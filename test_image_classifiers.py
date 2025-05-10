import timm
import torch
from torchvision.transforms import functional as torchvision_F
from torch.nn import functional as F
from typing import Dict, Any, Tuple 

class ImagePreprocessor:
    def __init__(self, data_config:Dict[str, Any]):
        
        self.resize_size = data_config["input_size"][1:] 
        self.crop_size = data_config["input_size"][1:]
        print(self.resize_size, self.crop_size)
        self.interpolation_mode = data_config["interpolation"]
        self.mean = data_config["mean"]
        self.std = data_config["std"]

    def __call__(self, images:torch.Tensor, normalize=True):
        """
        Preprocess a batch of images.

        1. Resize the images to the specified size.
        2. Center crop the images to the specified size.
        3. Normalize the images using the specified mean and std (optional).

        Args:
            images (torch.Tensor): Batch of images to preprocess.
            normalize (bool): Whether to normalize the images.
        """
        images = F.interpolate(images, size=self.resize_size, mode=self.interpolation_mode)
        images = torchvision_F.center_crop(images, self.crop_size)
        if normalize:
            images = torchvision_F.normalize(images, mean=self.mean, std=self.std)
        return images

class ModelWrapper:
    def __init__(
            self, 
            feature_extractor:torch.nn.Module,
            classifier:torch.nn.Module,
            image_preprocessor:ImagePreprocessor,
            ):
        """
        Initialises a model wrapper with all its components.
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.image_preprocessor = image_preprocessor

    def forward(self, images:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images (torch.Tensor): Batch of images to process.

        Returns:
            torch.Tensor: Model outputs.
        """
        images = self.image_preprocessor(images)
        features = self.feature_extractor(images)
        outputs = self.classifier(features)
        return features, outputs

if __name__ == "__main__":
    print("Testing image classifiers...")
    model_name = "maxvit_tiny_tf_224.in1k"
    feature_extractor = timm.create_model(model_name=model_name, pretrained=True, in_chans=3, features_only=True)
    classifier = timm.create_model(model_name=model_name, pretrained=True, in_chans=3, features_only=False).get_classifier()
    data_config = timm.data.resolve_model_data_config(feature_extractor)
    print("Data config:", data_config)

    image_preprocessor = ImagePreprocessor(data_config)
    model = ModelWrapper(feature_extractor=feature_extractor, classifier=classifier)

    test_batch = torch.randn(4, 3, 224, 224)

    preprocessed_images = image_preprocessor(test_batch)
    print("Preprocessed images shape:", preprocessed_images.shape)

    print("Model created successfully.")
