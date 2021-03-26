"""Model config in json format"""

CFG = {
    "data": {
        "path_to_data" : "/workspace/product-classifier/data",
        "path_to_model" : "/workspace/product-classifier/saved_models/",
        "image_size": 224,
        "num_classes": 139,
        "load_with_info": True
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epochs": 10,
        "opt_lr" : 0.01,
        "fine_tune_at": 80,
        'dropout' : 0.3,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "name" : 'resnet_50_opt_lr__0.0001__fine_tune_at__120',
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}