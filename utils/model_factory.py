import timm
import copy
from typing import Any

_MODEL_CACHE = {}

def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> Any:
    """Centralized model creation with caching"""
    cache_key = f"{model_name}_{num_classes}_{pretrained}"
    
    if cache_key not in _MODEL_CACHE:
        # print(f"[🔧] Creating {model_name} model (downloading weights)")
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        _MODEL_CACHE[cache_key] = model
    else:
        pass
        # print(f"[♻️] Using cached {model_name} model")
    
    return copy.deepcopy(_MODEL_CACHE[cache_key])