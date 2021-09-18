import bisect

from torch.utils.data import Dataset, ConcatDataset
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple


class ConcatData(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]):
        super().__init__(datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx], dataset_idx

    def update_length(self):
        self.cumulative_sizes = self.cumsum(self.datasets)
