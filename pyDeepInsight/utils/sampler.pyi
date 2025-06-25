from typing import Sequence, Iterator, List
from torch import Tensor
from torch.utils.data.sampler import Sampler


class StratifiedEventBatchSampler(Sampler):

    events: Tensor
    batch_size: int
    events0_idx: Tensor
    events1_idx: Tensor
    _len: int
    batch0_size: int
    batch1_size: int
    sampler0: Sampler
    sampler1: Sampler

    def __init__(self, events: Sequence[int], batch_size: int) -> None: ...

    def __iter0__(self) -> Iterator[List[int]]: ...

    def __iter1__(self) -> Iterator[List[int]]: ...

    def __iter__(self) -> Iterator[List[int]]: ...

