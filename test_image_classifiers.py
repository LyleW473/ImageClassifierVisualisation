import timm
import torch
from src.ml.model.utils import load_model

if __name__ == "__main__":
    print("Testing image classifiers...")
    model_name = "maxvit_tiny_tf_224.in1k"
    model = load_model(model_name=model_name)
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