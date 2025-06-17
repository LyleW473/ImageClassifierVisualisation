import torch
import numpy as np
import random

from datasets import load_dataset
from typing import Dict, Any, Iterator, Tuple

from src.ml.model.wrappers import ModelWrapper  
from src.ml.inference.utils import preprocess_dataset_image, get_json_result
from src.ml.model.results import InferenceResult

def get_imagenet1k_sample_generator(buffer_size:int=100) -> Iterator[Tuple[torch.Tensor, int]]:
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
                sample = preprocess_dataset_image(sample)
                buffer.append(sample)
        except StopIteration:
            # If the dataset iterator is exhausted, reset it
            imagenet_dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
            dataset_iter = iter(imagenet_dataset)
            continue
        
        random.shuffle(buffer)
        
        # Yield samples from the buffer
        for sample in buffer:
            yield sample

def get_answer_generator(
                        model:ModelWrapper,
                        sample_generator:iter,
                        class_index:Dict[str, str]
                        ):
    """
    Generator function to yield the answer for a sample from the ImageNet-1K dataset.
    - Returns a JSON object containing the feature, logits, confidence, predicted class name, and actual class name.

    Args:
        class_index (Dict[str, str]): A dictionary mapping class IDs to class names.
    """
    while True:
        sample:tuple = next(sample_generator)
        image:torch.Tensor = sample[0]
        label:int = sample[1]
        
        # Forward pass
        with torch.no_grad():
            inference_result:InferenceResult = model.forward(image)
        
        json_result: Dict[str, Any] = get_json_result(
                            inference_result=inference_result, 
                            class_index=class_index, 
                            label=label
                            )
        yield json_result