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
        print("Features", [feature.shape for feature in features])
        outputs = self.classifier(features[-1]) # Last feature map
        return features, outputs

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

    def __call__(self, features:torch.Tensor) -> torch.Tensor:
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
        return logits
        
def load_model(model_name:str) -> ModelWrapper:
    """
    Loads a model from the timm library with the specified name.

    Args:
        model_name (str): Name of the model to load.
    """
    feature_extractor = timm.create_model(model_name=model_name, pretrained=True, in_chans=3, features_only=True)
    classifier = timm.create_model(model_name=model_name, pretrained=True, in_chans=3, features_only=False)
    print(dir(classifier))

    classifier = ClassifierWrapper(global_pool=classifier.global_pool, classifier_head=classifier.get_classifier())

    data_config = timm.data.resolve_model_data_config(feature_extractor)
    image_preprocessor = ImagePreprocessor(data_config)
    model = ModelWrapper(feature_extractor=feature_extractor, classifier=classifier, image_preprocessor=image_preprocessor)
    return model

if __name__ == "__main__":
    print("Testing image classifiers...")
    model_name = "maxvit_tiny_tf_224.in1k"
    model = load_model(model_name=model_name)
    test_batch = torch.randn(4, 3, 224, 224)

    features, outputs = model.forward(test_batch)
    print("Features", [feature.shape for feature in features])
    print("Outputs", outputs.shape)
    print("Model created successfully.")