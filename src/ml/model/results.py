import torch
from dataclasses import dataclass
from typing import List, Any, Dict

@dataclass
class InferenceResult:
    """
    Class to hold the inference results.

    Attributes:
        original_images (List[Any]): Original images as python arrays or similar types.
        features (torch.Tensor): Feature maps from the model.
        logits (torch.Tensor): Raw model outputs.
        probs (torch.Tensor): Probabilities from the model outputs.
        class_ids (torch.Tensor): Predicted class IDs.
    """
    original_images: List[Any]
    features: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor
    class_ids: torch.Tensor
