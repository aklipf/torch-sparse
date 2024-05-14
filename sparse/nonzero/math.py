import torch
from .. import SparseTensor


def pow(x: SparseTensor, exp: int | float | torch.Tensor) -> SparseTensor:
    if isinstance(exp, torch.Tensor):
        return x.create_shared(x._values.pow(exp[None]))

    return x.create_shared(x._values.pow(exp))


def exp(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.exp())


def exp2(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.exp2())


def log(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.log())


def log2(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.log2())


def log10(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.log10())


def sqrt(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.sqrt())


def rsqrt(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.rsqrt())
