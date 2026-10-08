import torch

if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is NOT available. Using CPU.")
