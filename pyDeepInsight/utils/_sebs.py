from typing import Sequence
import math
import torch
from torch.utils.data.sampler import Sampler, RandomSampler


class StratifiedEventBatchSampler(Sampler):
    """Samples elements with from a set with binary labelling to ensure
    the event label (1) is evenly distributed across batches.

    This sampler is useful when the loss function requires at least one event,
    such as in the case of a Cox Proportional Hazard based loss.
    """

    events: torch.Tensor
    batch_size: int

    def __init__(self, events: Sequence[int], batch_size: int) -> None:
        """Generate an StratifiedBinaryBatchSampler instance

        Args:
            events: int sequence of binary event labels (0, 1)
            batch_size: int that defines size of mini-batch.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.events = torch.as_tensor(events, dtype=torch.int64)
        self.batch_size = batch_size

        self.events0_idx = torch.where(self.events == 0)[0]
        self.events1_idx = torch.where(self.events == 1)[0]

        self._len = math.ceil(len(self.events) / self.batch_size)

        if self.events1_idx.shape[0] < self._len:
            raise ValueError("the number of events ({}) must be equal or "
                             "larger than the number of batches ({})"
                             .format(self.events1_idx.shape[0], self._len))

        # Get batch sizes for each
        self.batch0_size = math.ceil(
            self.events0_idx.shape[0] / self.events.shape[0] * batch_size)
        self.batch1_size = math.floor(
            self.events1_idx.shape[0] / self.events.shape[0] * batch_size)

        self.sampler0 = RandomSampler(self.events0_idx, replacement=False)
        self.sampler1 = RandomSampler(self.events1_idx, replacement=False)

    def __iter0__(self):
        """Iterate the non-event (0) label sampler"""
        batch = torch.tensor([], dtype=torch.int64)
        for idx in self.sampler0:
            idx0 = self.events0_idx[idx, None]
            batch = torch.cat((batch, idx0), 0)
            if batch.shape[0] == self.batch0_size:
                yield batch
                batch = torch.tensor([], dtype=torch.int64)
        if batch.shape[0] > 0:
            yield batch

    def __iter1__(self):
        """Iterate the event (0) label sampler"""
        batch = torch.tensor([], dtype=torch.int64)
        for idx in self.sampler1:
            idx1 = self.events1_idx[idx, None]
            batch = torch.cat((batch, idx1), 0)
            if batch.shape[0] == self.batch1_size:
                yield batch
                batch = torch.tensor([], dtype=torch.int64)
        if batch.shape[0] > 0:
            yield batch

    def __iter__(self):
        """Generate the indices for the next batch of elements"""
        for batch0, batch1 in zip(self.__iter0__(), self.__iter1__()):
            batch = torch.cat((batch0, batch1), 0).sort()[0]
            yield batch

    # Removed - https://github.com/Lightning-AI/lightning/issues/2429
    # def __len__(self):
    #     """Return the number of batches"""
    #     return self._len
