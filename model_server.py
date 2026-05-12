"""
model_server.py — Inference wrapper for the custom diffusion model.

This module contains:
- DiffusionScheduler: noise schedule (linear beta schedule)
- TimeEmbedding, ResBlock, UNet: the denoising neural network
- InferenceWrapper: loads weights once, exposes generate / img2img / inpaint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Diffusion Scheduler
# ---------------------------------------------------------------------------

class DiffusionScheduler:
    def __init__(self, steps=500, beta_start=1e-4, beta_end=0.005):
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t):
        """Noisify a batch of images x_0 at specific timesteps t."""
        device = x_0.device
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise


# ---------------------------------------------------------------------------
# UNet components
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        )
        self.conv1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        h = self.conv1(x)
        t_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (scale + 1) + shift
        h = self.conv2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, ch_mults=(1, 2, 4, 8)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_ch),
            nn.Linear(base_ch, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )
        time_dim = base_ch * 4

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([])
        channels = base_ch
        for mult in ch_mults:
            out_c = base_ch * mult
            self.downs.append(
                nn.ModuleDict(
                    {
                        "block": ResBlock(channels, out_c, time_dim),
                        "down": nn.Conv2d(out_c, out_c, 4, 2, 1),
                    }
                )
            )
            channels = out_c

        # Middle
        self.mid_block = ResBlock(channels, channels, time_dim)

        # Upsampling
        self.ups = nn.ModuleList([])
        for mult in reversed(ch_mults):
            out_c = base_ch * mult
            self.ups.append(
                nn.ModuleDict(
                    {
                        "block": ResBlock(out_c * 2, out_c, time_dim),
                        "up": nn.ConvTranspose2d(channels, out_c, 4, 2, 1),
                    }
                )
            )
            channels = out_c

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)

        skip_conns = []
        for down in self.downs:
            x = down["block"](x, t)
            skip_conns.append(x)
            x = down["down"](x)

        x = self.mid_block(x, t)

        for up in self.ups:
            x = up["up"](x)
            skip = skip_conns.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat((x, skip), dim=1)
            x = up["block"](x, t)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Inference functions (ported from notebook)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_ddim(model, scheduler, image_size=64, ddim_steps=100):
    """Feature 1: Fast generation using DDIM sampling."""
    model.eval()
    device = next(model.parameters()).device

    # Clamp so step_size is always at least 1
    ddim_steps = min(ddim_steps, scheduler.steps)
    step_size = max(1, scheduler.steps // ddim_steps)
    timesteps = list(range(0, scheduler.steps, step_size))[::-1]

    x_t = torch.randn(1, 3, image_size, image_size, device=device)

    for i in range(len(timesteps)):
        t = timesteps[i]
        t_tensor = torch.tensor([t], device=device)

        alpha_prod_t = scheduler.alphas_cumprod[t].to(device)
        t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else -1
        if t_prev >= 0:
            alpha_prod_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=device)

        predicted_noise = model(x_t, t_tensor)
        pred_x0 = (x_t - torch.sqrt(1 - alpha_prod_t) * predicted_noise) / torch.sqrt(
            alpha_prod_t
        )
        dir_xt = torch.sqrt(1 - alpha_prod_t_prev) * predicted_noise
        x_t = torch.sqrt(alpha_prod_t_prev) * pred_x0 + dir_xt

    x_t = (x_t.clamp(-1, 1) + 1) / 2
    return x_t


@torch.no_grad()
def image_to_image(model, scheduler, init_image, strength=0.5):
    """Feature 2: SDEdit — transform an existing image."""
    model.eval()
    device = init_image.device

    start_step = int(scheduler.steps * strength) - 1
    noise = torch.randn_like(init_image)
    alpha_cumprod = scheduler.alphas_cumprod[start_step].to(device)

    x_t = torch.sqrt(alpha_cumprod) * init_image + torch.sqrt(1 - alpha_cumprod) * noise

    for i in reversed(range(start_step + 1)):
        t_tensor = torch.tensor([i], device=device)
        predicted_noise = model(x_t, t_tensor)

        alpha = scheduler.alphas[i].to(device)
        alpha_prod = scheduler.alphas_cumprod[i].to(device)
        beta = scheduler.betas[i].to(device)

        if i > 0:
            random_noise = torch.randn_like(x_t)
        else:
            random_noise = torch.zeros_like(x_t)

        noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_prod)
        x_t = (1 / torch.sqrt(alpha)) * (
            x_t - noise_factor * predicted_noise
        ) + torch.sqrt(beta) * random_noise

    x_t = (x_t.clamp(-1, 1) + 1) / 2
    return x_t


@torch.no_grad()
def inpaint(model, scheduler, init_image, mask):
    """Feature 3: Inpainting — fill masked (0) regions, keep mask=1 regions."""
    model.eval()
    device = init_image.device

    x_t = torch.randn_like(init_image)

    for i in reversed(range(scheduler.steps)):
        t_tensor = torch.tensor([i], device=device)

        alpha_cumprod = scheduler.alphas_cumprod[i].to(device)
        noise = torch.randn_like(init_image)
        real_x_t = (
            torch.sqrt(alpha_cumprod) * init_image
            + torch.sqrt(1 - alpha_cumprod) * noise
        )

        x_t = real_x_t * mask + x_t * (1 - mask)

        predicted_noise = model(x_t, t_tensor)

        alpha = scheduler.alphas[i].to(device)
        alpha_prod = scheduler.alphas_cumprod[i].to(device)
        beta = scheduler.betas[i].to(device)

        if i > 0:
            random_noise = torch.randn_like(x_t)
        else:
            random_noise = torch.zeros_like(x_t)

        noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_prod)
        x_t = (1 / torch.sqrt(alpha)) * (
            x_t - noise_factor * predicted_noise
        ) + torch.sqrt(beta) * random_noise

    x_0 = init_image * mask + x_t * (1 - mask)
    x_0 = (x_0.clamp(-1, 1) + 1) / 2
    return x_0


# ---------------------------------------------------------------------------
# High-level Inference Wrapper
# ---------------------------------------------------------------------------

# Standard transforms to prepare images for the model ([-1, 1] normalisation)
_to_tensor = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class InferenceWrapper:
    """Loads the model once and provides convenient methods for the Gradio app."""

    def __init__(
        self,
        weights_path: str = "models/diffusion_model_street.pth",
        steps: int = 500,
        beta_end: float = 0.005,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheduler = DiffusionScheduler(steps=steps, beta_end=beta_end)
        self.model = UNet(in_ch=3, out_ch=3, base_ch=64).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[InferenceWrapper] Model loaded on {self.device}")

    # -- helpers ----------------------------------------------------------

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a [1,3,64,64] tensor in [-1,1]."""
        return _to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(self.device)

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a [1,3,H,W] tensor in [0,1] to a PIL Image."""
        img = tensor[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    # -- public API -------------------------------------------------------

    def generate_fast(self, ddim_steps: int = 50) -> Image.Image:
        """DDIM fast generation → returns a PIL Image."""
        result = sample_ddim(self.model, self.scheduler, ddim_steps=ddim_steps)
        return self._tensor_to_pil(result)

    def transform_image(
        self, pil_img: Image.Image, strength: float = 0.6
    ) -> Image.Image:
        """SDEdit image-to-image → returns a PIL Image."""
        tensor = self._pil_to_tensor(pil_img)
        result = image_to_image(self.model, self.scheduler, tensor, strength=strength)
        return self._tensor_to_pil(result)

    def inpaint_image(
        self, pil_img: Image.Image, pil_mask: Image.Image
    ) -> Image.Image:
        """
        Inpaint masked regions.
        pil_mask should be an image where WHITE (255) = keep, BLACK (0) = replace.
        """
        tensor = self._pil_to_tensor(pil_img)

        # Convert mask to binary [1,1,64,64]: 1=keep, 0=replace
        mask_gray = pil_mask.convert("L").resize((64, 64), Image.NEAREST)
        mask_np = np.array(mask_gray).astype(np.float32) / 255.0
        # Threshold: anything > 0.5 is "keep"
        mask_np = (mask_np > 0.5).astype(np.float32)
        mask_tensor = (
            torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        result = inpaint(self.model, self.scheduler, tensor, mask_tensor)
        return self._tensor_to_pil(result)
