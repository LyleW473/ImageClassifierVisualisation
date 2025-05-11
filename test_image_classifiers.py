import timm
import torch
from src.ml.model.utils import load_model

if __name__ == "__main__":
    print("Testing image classifiers...")
    model_name = "maxvit_tiny_tf_224.in1k"
    model = load_model(model_name=model_name)
    test_batch = torch.randn(4, 3, 224, 224)

    features, outputs = model.forward(test_batch)
    print("Features", [feature.shape for feature in features])
    print("Outputs", outputs.shape)
    print("Model created successfully.")