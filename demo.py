import torch
from sparse.ops import SparseOpsMixin


if __name__ == "__main__":
    """
    print(
        SparseOpsMixin._select_values(
            torch.tensor(
                [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 1, 1]], dtype=torch.long
            ),
            torch.tensor(
                [
                    [1, 2, 3],
                    [2, 2, 3],
                    [3, 2, 3],
                    [4, 2, 3],
                    [5, 2, 3],
                    [6, 2, 3],
                    [7, 2, 3],
                    [8, 2, 3],
                ],
                dtype=torch.int32,
            ),
            torch.tensor([True, False, False, False, True, False, True, False]),
            2,
        )
    )

    exit(0)
    """

    a = SparseOpsMixin(
        torch.tensor([[0, 3, 1, 1, 2, 2, 3], [0, 0, 1, 2, 1, 2, 3]], dtype=torch.long),
        torch.tensor([[1], [5], [1], [1], [1], [1], [1]], dtype=torch.int32),
        shape=(4, 4),
    )
    b = SparseOpsMixin(
        torch.tensor([[0, 1, 1, 2, 3], [0, 1, 2, 2, 3]], dtype=torch.long),
        torch.tensor([[1], [2], [1], [1], [1]], dtype=torch.int32),
        shape=(4, 4),
    )
    c = SparseOpsMixin(
        torch.tensor([[0, 1, 1, 2, 2], [0, 1, 2, 2, 3]], dtype=torch.long),
        torch.tensor([[-2], [-2], [1], [1], [1]], dtype=torch.int32),
        shape=(4, 4),
    )

    print(a.to_dense())
    print(b.to_dense())
    print(c.to_dense())
    """
    print("test mul")
    print((a * b * c).to_dense())
    print(a.to_dense() * b.to_dense() * c.to_dense())
    print("test add")
    print((a + b + c).to_dense())
    print(a.to_dense() + b.to_dense() + c.to_dense())
    print("test sub")
    """
    print((a - b))
    print((a - b).to_dense())
    print((a - b - c).to_dense())
    print(a.to_dense() - b.to_dense() - c.to_dense())
    exit(0)
    result = a * b * c
    print(result)
    print(result.to_dense())
