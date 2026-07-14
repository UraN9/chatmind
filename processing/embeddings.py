"""
CLIP embeddings for chatmind.

Loads a pretrained CLIP model once and exposes two functions:
- embed_image(...): turn an image into a vector
- embed_text(...):  turn a text query into a vector

Both vectors live in the same embedding space, so a text query can be
compared directly against image embeddings stored in the database.
"""

from pathlib import Path
from typing import Union

import config  # noqa: F401  (sets KMP_DUPLICATE_LIB_OK before torch is imported)
import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

_device = "cuda" if torch.cuda.is_available() else "cpu"

_model, _, _preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
_tokenizer = open_clip.get_tokenizer(MODEL_NAME)

_model.to(_device)
_model.eval()


def embed_image(image: Union[str, Path, Image.Image]) -> list[float]:
    """
    Compute a CLIP embedding for an image.

    `image` can be a file path (str or Path) or an already-opened
    PIL.Image.Image.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    image_input = _preprocess(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = _model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()


def embed_text(text: str) -> list[float]:
    """
    Compute a CLIP embedding for a text query, in the same vector
    space as embed_image(), so it can be used to search for images.
    """
    tokens = _tokenizer([text]).to(_device)

    with torch.no_grad():
        features = _model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()