import torch

import sys;

print(sys.executable)

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Get current device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Print GPU name if available
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Additional verification: Allocate a tensor on GPU
    test_tensor = torch.randn(3, 3).cuda()
    print(f"Tensor device: {test_tensor.device}  (should be 'cuda:0')")
else:
    print("Running on CPU")


import torch
print(torch.__version__)
print(torch.version.cuda)  # Should print a CUDA version (e.g. '11.7')

print(f"Cuda Device Count: {torch.cuda.device_count()}")