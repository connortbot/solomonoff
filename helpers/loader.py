"""
Filename: loader.py
Description: Helpers for loading models using safetensors

Notes:

"""

import os
from collections import OrderedDict
import safetensors.torch

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    state_dict = OrderedDict()
    try:
        state_dict.update(safetensors.torch.load_file(path, device="cpu"))
    except Exception as e:
        raise ValueError(f"Error loading the model from {path}: {e}")
    
    return state_dict

if __name__ == "__main__":
    tinyllama_path = "files/TinyLlama-1.1B-Chat-v1.0/model.safetensors"
    try:
        tinyllama_state_dict = load_model(tinyllama_path)
        print(f"Successfully loaded state dictionary from {tinyllama_path}")
        with open("./files/safetensors-structure.txt", "w") as f:
            for key in list(tinyllama_state_dict.keys()):
                f.write(f"{key}: {tinyllama_state_dict[key].shape}\n")
    except Exception as e:
        print(f"Error: {e}")