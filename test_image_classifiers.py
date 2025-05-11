import torch
import numpy as np

from src.ml.model.utils import load_model
from datasets import load_dataset
from PIL.JpegImagePlugin import JpegImageFile

if __name__ == "__main__":
    print("Testing image classifiers...")
    model_name = "maxvit_tiny_tf_224.in1k"
    model = load_model(model_name=model_name)
    model.eval()
    test_batch = torch.randn(4, 3, 224, 224)

    inference_result = model.forward(test_batch)
    features = inference_result.features
    logits = inference_result.logits
    probs = inference_result.probs
    class_ids = inference_result.class_ids
    print("Features", [feature.shape for feature in features])
    print("Logits", logits.shape)
    print("Probs", probs.shape)
    print("Class IDs", class_ids.shape)
    print("Model created successfully.")

    def visualise_image(image:JpegImageFile) -> None:
        """
        Visualises an image using matplotlib.
        
        Args:
            image (JpegImageFile): The image to be visualised.
        """
        import matplotlib.pyplot as plt

        # Convert the image tensor to a numpy array

        image_np = np.array(image)

        # Display the image
        plt.imshow(image_np)
        plt.axis('off')
        plt.show()


    # Dataset loading
    imagenet_dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)

    for sample in imagenet_dataset:
        print("Sample:", sample.keys())
        image = sample["image"]
        label = sample["label"]

        # Visualise the image
        print(type(image), type(label))
        visualise_image(image)

        # Image conversion
        image = image.convert("RGB") # JpegImageFile -> PIL Image
        print(type(image))
        image = np.array(image) # PIL Image -> np.ndarray
        image = torch.tensor(image, dtype=torch.float32) # np.ndarray -> torch.Tensor
        image = image.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        image = image.unsqueeze(0) # Add batch dimension

        print("Image shape:", image.shape)
        print("Label:", label)

        # Forward pass
        with torch.no_grad():
            inference_result = model.forward(image)
        features = inference_result.features
        logits = inference_result.logits
        probs = inference_result.probs
        class_ids = inference_result.class_ids
        print("Features", [feature.shape for feature in features])
        print("Logits", logits.shape)
        print("Probs", probs.shape)
        print("Class IDs", class_ids.shape)
        break