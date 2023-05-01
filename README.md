# Introduction to PyTorch Tensors: Creating, Manipulating, and Using Tensors for Machine Learning

This repository contains code examples and tutorials for working with PyTorch and NumPy, two powerful Python libraries commonly used in machine learning and deep learning applications. The examples cover various topics, including creating and manipulating tensors, dealing with tensor shapes, indexing tensors, mixing PyTorch and NumPy, and running tensors on GPUs for faster computations. Each example is commented and provides explanations of the concepts being used, making it a useful resource for beginners and experienced users alike. Whether you're looking to learn PyTorch and NumPy from scratch or want to expand your knowledge, this repository has something for you.

Creating tensors: The code creates tensors using the torch.Tensor and torch.LongTensor constructors, as well as using torch.randn to create a tensor with random values.

Getting information from tensors: The code demonstrates how to get information from tensors such as their size, data type, and mean.

Manipulating tensors: The code shows how to manipulate tensors by multiplying two tensors element-wise using torch.mul.

Dealing with tensor shapes: The code demonstrates reshaping tensors using tensor.view and adding a new dimension to a tensor using tensor.unsqueeze.

Indexing on tensors: The code shows how to index tensors using the square bracket notation, similar to indexing on Python lists or NumPy arrays.

Mixing PyTorch tensors and NumPy: The code demonstrates how to convert between PyTorch tensors and NumPy arrays using torch.from_numpy and .numpy().

Reproducibility: The code sets a manual seed using torch.manual_seed to ensure reproducibility in machine learning experiments.

Running tensors on GPU: The code does not explicitly run tensors on a GPU, but it demonstrates how to create tensors that can be run on a GPU by specifying a data type such as torch.cuda.FloatTensor.
