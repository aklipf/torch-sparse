# PyTorch Sparse Tensors

A small package that implements the basic sparse tensor of any dimension.

## Key features

* Torch-like API for sparse tensor of n-dimension
* Support basic element-wise operations
* Support reduction (mean and sum) over 1 or multiple dimensions
* Support reshaping and broadcasting
* Support concatenation over 1 or multiple dimensions

## Installation

```bash
pip install torch-sparse-tensor
```

## Demo

Import package
```python
import torch
from sparse import SparseTensor
```

Build sparse tensors
```python
a = SparseTensor(
    torch.tensor([[0, 3, 1, 1, 2, 2, 3], [0, 0, 1, 2, 1, 2, 3]], dtype=torch.long),
    torch.tensor([[1], [5], [1], [1], [1], [1], [1]], dtype=torch.float32),
    shape=(4, 4),
)
b = SparseTensor(
    torch.tensor([[0, 1, 1, 2, 3], [0, 1, 2, 2, 3]], dtype=torch.long),
    torch.tensor([[1], [2], [1], [1], [1]], dtype=torch.float32),
    shape=(4, 4),
)
```

To cuda device
```python
a = a.to("cuda")
b = b.to("cuda")
```

Conversion to dense representation
```python
print(a.to_dense())
print(b.to_dense())
```

Basic element-wise operations
```python
c=a+b
print(c.to_dense())

d=a*b
print(d.to_dense())
```

Reduction operations
```python
print(a.sum(0).to_dense())
print(a.mean(0).to_dense())
```

Indexing and broadcasting
```python
print((a[:, None, :] + a[:, :, None]).sum(0).to_dense())
```

Concatenation over several dimensions
```python
print(SparseTensor.cat((a, b), dim=1).to_dense())
print(SparseTensor.cat((a, b), dim=(0,1)).to_dense())
```