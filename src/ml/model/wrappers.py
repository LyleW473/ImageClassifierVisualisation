import torch
from torch.nn import functional as F
from typing import Tuple
from dataclasses import dataclass

from src.ml.model.image_preprocessor import ImagePreprocessor


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

class ModelWrapper:
    def __init__(
            self, 
            feature_extractor:torch.nn.Module,
            classifier:torch.nn.Module,
            image_preprocessor:ImagePreprocessor,
            ):
        """
        Initialises a model wrapper with all its components.

        Args:
            feature_extractor (torch.nn.Module): Feature extractor model.
            classifier (torch.nn.Module): Classifier model.
            image_preprocessor (ImagePreprocessor): Image preprocessor for input images.
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.image_preprocessor = image_preprocessor

    def forward(self, images:torch.Tensor) -> InferenceResult:
        """
        Forward pass through the model.

        Args:
            images (torch.Tensor): Batch of images to process.

        Returns:
            torch.Tensor: Model outputs.
        """
        images = self.image_preprocessor(images)
        features = self.feature_extractor(images)
        print("Features", [feature.shape for feature in features])
        logits, probs, class_ids = self.classifier(features[-1]) # Last feature map

        result = InferenceResult(
            features=features,
            logits=logits,
            probs=probs,
            class_ids=class_ids
        )
        return result

class ClassifierWrapper:
    def __init__(
                self, 
                global_pool:str,
                classifier_head:torch.nn.Module,
                ):
        """
        Initialises a classifier wrapper with the specified global pooling and classifier head.

        Args:
            global_pool (str): Type of global pooling to use.
            classifier_head (torch.nn.Module): Classifier head to use.
        """
        if global_pool == "avg":
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            raise NotImplementedError(f"Global pool {global_pool} not implemented.")
        self.classifier_head = classifier_head

    def __call__(self, features:torch.Tensor) -> Tuple[
                                                    torch.Tensor, 
                                                    torch.Tensor, 
                                                    torch.Tensor
                                                    ]:
        """
        Forward pass features to get the classifier outputs.

        Args:
            features (torch.Tensor): Features to process.
        """
        pooled_features = self.global_pool(features)
        print("Pooled features", pooled_features.shape)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        print("Pooled features reshaped", pooled_features.shape)
        logits = self.classifier_head(pooled_features)
        print("Logits", logits.shape)

        probs = F.softmax(logits, dim=1)
        
        print("Outputs", logits[0].min(), logits[0].max())
        print("Probs", probs[0].min(), probs[0].max())

        class_ids = torch.argmax(probs, dim=1)
        print("Class IDs", class_ids)

        return logits, probs, class_ids