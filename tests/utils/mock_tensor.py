from __future__ import annotations
import torch

from sparse.typing import Self


class MockTensor:
    """Good enouth for testing"""

    def __init__(
        self,
        shape: tuple = None,
        dtype: type = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, *args) -> MockTensor:
        out_shape = []
        in_shape = list(self.shape)

        for arg in args:
            if arg is None:
                out_shape.append(1)
            elif isinstance(arg, int):
                in_shape.pop(0)
            elif isinstance(arg, (list, tuple, torch.Tensor)):
                in_shape.pop(0)
                out_shape.append(len(arg))
            else:
                raise NotImplementedError

        out_shape.extend(in_shape)

        return MockTensor(shape=tuple(out_shape), dtype=self.dtype, device=self.device)

    def amax(self, **_) -> torch.LongTensor:
        return torch.ones(self.shape[0], dtype=torch.long)

    def view(self, *args) -> Self:
        numel = 1
        for s in self.shape:
            numel *= s

        numel_args = 1
        inferred = None
        for i, s in enumerate(args):
            assert inferred is None or s != -1
            if s == -1:
                inferred = i
            else:
                numel_args *= s

        if inferred is None:
            assert numel == numel_args
            return self.__class__(shape=args, dtype=self.dtype, device=self.device)

        shape_inferred = numel // numel_args
        assert numel == (shape_inferred * numel_args)

        args[inferred] = shape_inferred
        return self.__class__(shape=args, dtype=self.dtype, device=self.device)

    def reshape(self, *args) -> Self:
        return self.view(*args)

    def diff(self, **_) -> torch.LongTensor:
        return torch.tensor([[1]]).repeat(self.shape[0], 1)

    def pow(self, *exp) -> Self:
        return self

    def __repr__(self) -> str:
        return (
            f"MockTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"
        )
