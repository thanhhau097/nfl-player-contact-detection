from operator import itemgetter
from typing import Iterator, Optional

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DistributedSampler, Sampler
import random

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        import pdb

        pdb.set_trace()
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class ProportionalTwoClassesBatchSampler(Sampler):
    """
    dataset: DataSet class that returns torch tensors
    batch_size: Size of mini-batches
    minority_size_in_batch: Number of minority class samples in each mini-batch
    majority_priority: If it is True, iterations will include all majority
    samples in the data. Otherwise, it will be completed after all minority samples are used.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        minority_size_in_batch: int,
        world_size: int,
        majority_priority=True,
        
    ):
        super().__init__(labels)
        self.world_size = world_size
        self.labels = labels
        self.minority_size_in_batch = minority_size_in_batch
        self.batch_size = batch_size
        self.priority = majority_priority
        self._num_batches = (labels == 0).sum() // (batch_size - minority_size_in_batch) // self.world_size + 1
        

    def __len__(self):
        return len(self.labels) // self.world_size

    def __iter__(self):
        if self.minority_size_in_batch > self.batch_size:
            raise ValueError(
                "Number of minority samples in a batch must be lower than batch size!"
            )
        y_indices = [np.where(self.labels == label)[0] for label in np.unique(self.labels)]
        y_indices = sorted(y_indices, key=lambda x: x.shape)

        minority_copy = y_indices[0].copy()

        indices = []
        print("Prepare epoch's data")
        for _ in tqdm(range(self._num_batches)):
            if len(y_indices[0]) < self.minority_size_in_batch:
                if self.priority:
                    # reloading minority samples
                    y_indices[0] = minority_copy.copy()
            minority = np.random.choice(
                y_indices[0], size=self.minority_size_in_batch, replace=False
            )
            majority = np.random.choice(
                y_indices[1],
                size=(self.batch_size - self.minority_size_in_batch),
                replace=False,
            )
            # batch_inds = np.concatenate((minority, majority), axis=0)
            # batch_inds = np.random.permutation(batch_inds)
            batch_inds = []
            i_minor = 0
            j_major = 0
            for i in range(len(majority) + len(minority)):
                if i % 2 == 0:
                    batch_inds.append(minority[i_minor])
                    i_minor += 1
                else:
                    batch_inds.append(majority[j_major])
                    j_major += 1
            for i in range(len(batch_inds) // self.world_size):
                batch_indices_new = batch_inds[i*self.world_size:(i+1)*self.world_size]
                random.shuffle(batch_indices_new)
                batch_inds[i*self.world_size:(i+1)*self.world_size] = batch_indices_new

            y_indices[0] = np.setdiff1d(y_indices[0], minority)
            y_indices[1] = np.setdiff1d(y_indices[1], majority)
            indices.extend(batch_inds)
        return iter(indices[: len(self.labels) // self.world_size])


if __name__ == "__main__":
    import pandas as pd

    train_df = pd.read_csv("data/train_features.csv")
    sampler = DistributedSamplerWrapper(
        ProportionalTwoClassesBatchSampler(train_df["contact"].values, 8, 1), 1, 0
    )
    list(iter(sampler))