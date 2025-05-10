import timm

if __name__ == "__main__":
    print("Testing image classifiers...")
    
    model = timm.create_model(model_name="maxvit_tiny_tf_224.in1k", pretrained=True, in_chans=3, features_only=True)
    data_config = timm.data.resolve_model_data_config(model)
    print("Data config:", data_config)
    print("Model created successfully.")
