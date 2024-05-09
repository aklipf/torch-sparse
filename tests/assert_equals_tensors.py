import torch


def assert_equal_tensors(*tensors: torch.Tensor, precision: float = 1e-6):
    first, *_ = tensors

    for tensor in tensors[1:]:
        assert first.shape == tensor.shape
        assert first.device == tensor.device
        assert first.dtype == tensor.dtype
        assert (first - tensor).abs().amax() < precision


def assert_equal_bool_tensors(*tensors: torch.Tensor):
    first, *_ = tensors

    for tensor in tensors[1:]:
        assert first.shape == tensor.shape
        assert first.device == tensor.device
        assert first.dtype == tensor.dtype == torch.bool
        assert (first == tensor).all()
