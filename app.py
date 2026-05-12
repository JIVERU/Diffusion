"""
app.py — Gradio web interface for the custom diffusion model.

Run with:
    .venv_gradio\Scripts\python app.py
"""

import gradio as gr
from PIL import Image
from model_server import InferenceWrapper

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print("Loading model — this may take a moment …")
wrapper = InferenceWrapper(weights_path="models/diffusion_model_street.pth")
print("Ready!\n")

DISPLAY_SIZE = 512  # upscale 64×64 → 512×512 for display


# ---------------------------------------------------------------------------
# Image-to-Image (SDEdit)
# ---------------------------------------------------------------------------
def transform_image(input_image, strength: float):
    """Apply SDEdit transformation to an uploaded image."""
    if input_image is None:
        return None, None
    pil_img = (
        Image.fromarray(input_image)
        if not isinstance(input_image, Image.Image)
        else input_image
    )
    original_resized = pil_img.convert("RGB").resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS
    )
    result = wrapper.transform_image(pil_img, strength=strength)
    result_upscaled = result.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
    return original_resized, result_upscaled


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
)

with gr.Blocks(title="Diffusion Image-to-Image") as demo:
    gr.Markdown("# Diffusion Image-to-Image")

    img_input = gr.Image(
        label="Upload Image",
        type="pil",
    )
    strength_slider = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.6,
        step=0.05,
        label="Strength",
    )
    transform_btn = gr.Button("✨ Transform", variant="primary")
    original_out = gr.Image(
        label="Original",
        type="pil",
    )
    transformed_out = gr.Image(
        label="Transformed",
        type="pil",
    )

    transform_btn.click(
        fn=transform_image,
        inputs=[img_input, strength_slider],
        outputs=[original_out, transformed_out],
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(theme=THEME)
