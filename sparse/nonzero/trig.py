from .. import SparseTensor


def sin(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.sin())


def cos(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.cos())


def tan(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.tan())


def sinh(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.sinh())


def cosh(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.cosh())


def tanh(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.tanh())


def asin(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.asin())


def acos(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.acos())


def atan(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.atan())


def asinh(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.asinh())


def acosh(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.acosh())


def atanh(x: SparseTensor) -> SparseTensor:
    return x.create_shared(x._values.atanh())
