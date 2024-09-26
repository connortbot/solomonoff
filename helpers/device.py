"""
Filename: device.py
Description: Device-related helpers

Notes:

"""

import torch

def get_device():
    if torch.cuda.is_available():
        dc = torch.cuda.device_count()
        device = f"cuda:{dc-1}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device