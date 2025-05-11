import torch
import numpy as np
import json
import random

from src.ml.model.utils import load_model
from datasets import load_dataset
from PIL.JpegImagePlugin import JpegImageFile
from typing import Dict

if __name__ == "__main__":
    print("Testing image classifiers...")
    model_name = "maxvit_tiny_tf_224.in1k"
    model = load_model(model_name=model_name)
    model.eval()

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


    def get_imagenet1k_sample_generator(buffer_size:int=100):
        """
        Generator function to yield samples from the ImageNet-1K dataset.
        - Returns a tuple of (image, label) for each sample.
        - The generator function will keep yielding samples indefinitely.
        - Samples are added to a buffer of size 'buffer_size' before yielding
        - If the dataset iterator is exhausted, it will reset and continue yielding samples.
        - Images in a buffer are shuffled before yielding.

        Args:
            buffer_size (int): The size of the buffer to hold samples before yielding.
        """
        # Load the ImageNet-1K dataset
        imagenet_dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)

        dataset_iter = iter(imagenet_dataset)

        while True:
            buffer = []

            try:
                # Fill the buffer with samples
                for _ in range(buffer_size):
                    sample = next(dataset_iter)
                    image = sample["image"]
                    label = sample["label"]

                    # Visualise the image
                    # visualise_image(image)

                    # Image conversion
                    image = image.convert("RGB") # JpegImageFile -> PIL Image
                    image = np.array(image) # PIL Image -> np.ndarray
                    image = torch.tensor(image, dtype=torch.float32) # np.ndarray -> torch.Tensor
                    image = image.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
                    image = image.unsqueeze(0) # Add batch dimension

                    buffer.append((image, label))
            
            except StopIteration:
                # If the dataset iterator is exhausted, reset it
                imagenet_dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
                dataset_iter = iter(imagenet_dataset)
                continue
            
            random.shuffle(buffer)
            
            # Yield samples from the buffer
            for sample in buffer:
                yield sample

    def get_answer_generator(class_index:Dict[str, str]):
        """
        Generator function to yield the answer for a sample from the ImageNet-1K dataset.
        - Returns a JSON object containing the feature, logits, confidence, predicted class name, and actual class name.

        Args:
            class_index (Dict[str, str]): A dictionary mapping class IDs to class names.
        """
        while True:
            image, label = next(imagenet1k_gen)
            
            # Forward pass
            with torch.no_grad():
                inference_result = model.forward(image)
            features = inference_result.features
            logits = inference_result.logits
            probs = inference_result.probs
            class_ids = inference_result.class_ids

            assert logits.shape == (1, 1000), f"Logits shape mismatch: {logits.shape}"
            assert probs.shape == (1, 1000), f"Probs shape mismatch: {probs.shape}"
            assert class_ids.shape == (1,), f"Class IDs shape mismatch: {class_ids.shape}"

            # Remove batch dimension for all tensors
            features = [feature.squeeze(0).numpy().tolist() for feature in features] 
            logits = logits.squeeze(0).numpy().tolist()
            class_id = class_ids.squeeze(0).numpy().tolist()

            # Extract confidence and class names.
            confidence = probs[0][class_id].numpy().tolist()
            predicted_class_name = class_index[str(class_id)]
            actual_class_name = class_index[str(label)]

            json_result = {
                # "feature": features,
                # "logits": logits,
                "confidence": confidence,
                "predicted_class_name": predicted_class_name,
                "actual_class_name": actual_class_name,
            }
            print("JSON result", json_result)
            yield json_result

    with open("in1k_cls_index.json", "r") as f:
        imagenet1k_cls_index = json.load(f)
    imagenet1k_gen = get_imagenet1k_sample_generator(buffer_size=10)
    answer_gen = get_answer_generator(class_index=imagenet1k_cls_index)
    
    for i in range(20):
        print("Iteration", i)
        pred_answer_json = next(answer_gen)
    