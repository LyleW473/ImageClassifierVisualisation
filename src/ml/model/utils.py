import timm
from src.ml.model.wrappers import ModelWrapper, ClassifierWrapper
from src.ml.model.image_preprocessor import ImagePreprocessor

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