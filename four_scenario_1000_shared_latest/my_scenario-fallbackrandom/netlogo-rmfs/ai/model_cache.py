import torch
import numpy as np
import random
_model_cache = {}

def get_model(create_fn, model_path=None):
    # Set torch seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if model_path not in _model_cache:
        model = create_fn
        model.load_model(model_path)
        _model_cache[model_path] = model
    return _model_cache[model_path]