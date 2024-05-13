from .. import SparseTensor


def view(x: SparseTensor, *shape: int) -> SparseTensor:
    return x.create_shared(x._values.view(*shape))


def reshape(x: SparseTensor, *shape: int) -> SparseTensor:
    return x.create_shared(x._values.reshape(*shape))


def clip(x: SparseTensor, min: int = None, max: int = None) -> SparseTensor:
    return x.create_shared(x._values.clip(min=min, max=max))


def clamp(x: SparseTensor, min: int = None, max: int = None) -> SparseTensor:
    return x.create_shared(x._values.clamp(min=min, max=max))
