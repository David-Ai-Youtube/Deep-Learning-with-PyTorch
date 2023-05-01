import torch
import numpy as np

# Creating tensors
tensor1 = torch.Tensor([[1, 2], [3, 4]])   # create a 2x2 tensor with default float type
tensor2 = torch.LongTensor([5, 6, 7])      # create a 1D tensor of long integers
tensor3 = torch.randn(3, 4)                # create a 3x4 tensor with random values

# Getting information from tensors
print(tensor1.size())                      # output: torch.Size([2, 2])
print(tensor2.dtype)                       # output: torch.int64
print(tensor3.mean())                      # output: tensor(0.1108)

# Manipulating tensors
tensor4 = tensor1 + tensor3                # add two tensors element-wise
tensor5 = torch.matmul(tensor1, tensor3.T) # matrix multiplication
tensor6 = torch.cat((tensor1, tensor2.view(1, -1)), dim=0) # concatenate two tensors

# Dealing with tensor shapes
tensor7 = tensor1.view(1, 4)               # reshape tensor1 into a 1x4 tensor
tensor8 = tensor2.unsqueeze(1)             # add a new dimension to tensor2 (1D -> 2D)

# Indexing on tensors
print(tensor1[0, 1])                       # output: tensor(2.)
print(tensor3[:, 1:3])                     # output: a 3x2 tensor

# Mixing PyTorch tensors and NumPy
numpy_array = np.array([[8, 9], [10, 11]])
tensor9 = torch.from_numpy(numpy_array)    # convert a NumPy array to a PyTorch tensor
numpy_array2 = tensor5.numpy()             # convert a PyTorch tensor to a NumPy array

# Reproducibility
torch.manual_seed(42)                      # set a manual seed for reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True  # ensure deterministic operations on GPU

# Running tensors on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    tensor1 = tensor1.to(device)            # move tensor1 to the GPU
