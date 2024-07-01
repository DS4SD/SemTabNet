"""Base class for dataset."""
import abc
from typing import Optional

from torch import Generator
from torch.utils.data import Dataset, random_split


class BaseDataset(Dataset):
    """Base Class for handling dataset."""

    allowed_tasks = ['ud2sd_table','tca1d','tca2d']

    newline_token = "<br>"
    sep_token = "<sep>"
    special_tokens = [newline_token, sep_token]

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of samples in dataset.

        Raises:
            NotImplementedError: if implementation is missing.

        Returns:
            int: number of samples in dataset.
        """
        raise NotImplementedError(
            f"Error: Class {self.__class__.__name__} has no implementation for __len__."
        )

    def random_split(self, dataset, lengths, seed: Optional[int] = None):
        """Random split of dataset into given lengths."""
        if seed is None:
            generator = Generator()
        else:
            generator = Generator().manual_seed(seed)
        return random_split(dataset=dataset, lengths=lengths, generator=generator)