import json

from src.ml.model.utils import load_model
from src.ml.inference.generators import get_imagenet1k_sample_generator, get_answer_generator

if __name__ == "__main__":
    print("Testing image classifiers...")
    model_name = "maxvit_tiny_tf_224.in1k"
    model = load_model(model_name=model_name)
    model.eval()


    with open("in1k_cls_index.json", "r") as f:
        imagenet1k_cls_index = json.load(f)
    imagenet1k_gen = get_imagenet1k_sample_generator(buffer_size=10)
    answer_gen = get_answer_generator(
                                    model=model,
                                    sample_generator=imagenet1k_gen,
                                    class_index=imagenet1k_cls_index
                                    )
    
    for i in range(20):
        print("Iteration", i)
        pred_answer_json = next(answer_gen)
    