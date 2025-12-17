import os
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np 

MODEL_DIR = "./dinov2-base"

def load_dinov2():
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device

def compute_embedding(img_input, processor, model, device):
    """
    Accepts either a file path (str/Path) or a PIL.Image.Image object.
    """
    if isinstance(img_input, str):  # If path, open file
        img = PILImage.open(img_input).convert('RGB')
    elif isinstance(img_input, PILImage.Image):  # If already a PIL Image
        img = img_input.convert('RGB')
    else:
        raise TypeError(f"img_input must be a file path or PIL.Image.Image, got {type(img_input)}")

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feature = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

    # L2-normalize for cosine similarity
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm

    return feature.tolist()
