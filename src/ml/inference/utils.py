import torch
import numpy as np

from PIL import Image
from typing import Dict, Any, Tuple, List, Union

from src.ml.model.results import InferenceResult

def preprocess_dataset_image(sample:Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    """
    Preprocesses a sample from the ImageNet-1K dataset.
    - Converts the image from a JpegImageFile into a torch.Tensor.

    Args:
        sample (Dict[str, Any]): A dictionary containing the image and label.
                                    - "image": The image to be preprocessed.
                                    - "label": The label of the image.
    """
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

    new_sample = (image, label)
    return new_sample

def get_json_result(
                    inference_result:InferenceResult,
                    class_index:Dict[str, str], 
                    label:int
                    ) -> Dict[str, Any]:
    """
    Converts the inference result into a JSON object that 
    encapsulates the information about the model's prediction
    that can be easily interpreted.
    
    Args:
        inference_result (InferenceResult): The result of the model inference.
        class_index (Dict[str, str]): A dictionary mapping class IDs to class names.
        label (int): The actual label of the image.
    """
    original_images = inference_result.original_images
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
        "originalImages": original_images,
        "features": features,
        "confidence": confidence,
        "predictedClassName": predicted_class_name,
        "actualClassName": actual_class_name,
    }
    print("JSON result", json_result)
    return json_result

def save_image_locally(image:List[Union[float, int]], file_path:str) -> None:
    """
    Saves an image in format (C, H, W) to a file path in RGB format.

    Args:
        image (List[Union[float, int]]): The image to be saved, in the format (C, H, W).
                                         file_path (str): The path where the image will be saved.

    """
    image = np.array(image)
    image = image.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    image = Image.fromarray(image.astype("uint8"), "RGB")
    image.save(file_path)