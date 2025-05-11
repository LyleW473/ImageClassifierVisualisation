import torch
from dataclasses import dataclass

@dataclass
class InferenceResult:
    """
    Class to hold the inference results.

    Attributes:
        features (torch.Tensor): Feature maps from the model.
        logits (torch.Tensor): Raw model outputs.
        probs (torch.Tensor): Probabilities from the model outputs.
        class_ids (torch.Tensor): Predicted class IDs.
    """
    features: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor
    class_ids: torch.Tensor
