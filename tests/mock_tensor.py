import torch


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

    def amax(self, **kwargs) -> torch.LongTensor:
        return torch.ones(self.shape[0], dtype=torch.long)

    def diff(self, **kwargs) -> torch.LongTensor:
        return torch.tensor([[1]]).repeat(self.shape[0], 1)

    def __repr__(self) -> str:
        return (
            f"MockTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"
        )
