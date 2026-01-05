import torch
print(f"PyTorch in Env: {torch.__version__}")
print(f"CUDA Active: {torch.cuda.is_available()}")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")