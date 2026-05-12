# Custom Diffusion Web App

A clean, standalone Gradio web interface for high-fidelity Image-to-Image transformations using a custom-trained Diffusion Model.

## Project Structure

```text
├── app.py                   # Main Gradio web interface
├── model_server.py          # PyTorch inference wrappers, UNet model, and SDEdit scheduler
├── final_enhanced.ipynb     # Complete training and data augmentation pipeline
├── requirements.txt         # Python dependencies
└── models/                  # Directory for pre-trained weights (ignored by Git)
```

## Setup Instructions

### 1. Create a Virtual Environment

Create and activate a Python virtual environment to manage dependencies cleanly:

**Windows (PowerShell):**
```powershell
python -m venv .venv_gradio
.\.venv_gradio\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv_gradio
source .venv_gradio/bin/activate
```

### 2. Install Dependencies

Install the required PyTorch, torchvision, and Gradio packages:

```bash
pip install -r requirements.txt
```

### 3. Provide Model Weights

Due to GitHub file size constraints (>100MB), the model weight checkpoints are not included in the repository repository history. 

Ensure you place your pre-trained model checkpoint inside the `models/` directory:
- Expected path by default: `models/diffusion_model_street.pth`

> **Note:** If you are starting from scratch, run the training pipeline inside `final_enhanced.ipynb` to train your model and generate the checkpoint files.

## Running the Web Application

Launch the Gradio application locally:

```bash
python app.py
```

Or using the direct path to the virtual environment's Python executable:

```powershell
.\.venv_gradio\Scripts\python app.py
```

Once loaded, the terminal will display a local web server URL (typically `http://127.0.0.0:7860`). Open this link in your browser to access the interactive diffusion editing suite.

## Features
- **SDEdit Image-to-Image Transformation:** Upload an image and adjust the **Strength** slider to control how much noise is injected before the custom UNet denoises the structure back into your model's target domain.
