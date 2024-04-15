import torch

from sparse import SparseTensor

A = SparseTensor(
    torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
)  # .to("cuda")
B = SparseTensor(torch.tensor([[0, 1, 2], [0, 1, 2]]))  # .to("cuda")

print(A)
print(B)
res = SparseTensor.cat((A.unsqueeze(0), B.unsqueeze(0)), dim=(0, 1))
print(res)

x = SparseTensor(
    torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5],
        ]
    ),
)

print(x)
print(x.dense.squeeze())
print(x.sum(0).dense.squeeze())
print(x.sum(1).dense.squeeze())
print(x.sum((0, 1)).dense.squeeze())

print(x.reshape_((-1, 14)))
