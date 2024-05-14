from .. import SparseTensor


def diag(x: SparseTensor, diagonal: int = 0) -> SparseTensor:
    return x.create_shared(x._values.diag(diagonal))


def det(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.det())


def trace(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.trace())


# TODO: norm, normalize
