from torch.utils.data import Sampler
from typing import List
from datasets import Dataset
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch


class BucketByAudioLength(Sampler):
    """
    Group audio samples which have equal or similar duration into the same bucket.
    The batch size is adapted with the duration of samples in each bucket.
    """

    def __init__(self,
                 dataset: Dataset,
                 bucket_boundaries: List[int] = [5, 10, 15, 20, 25],
                 bucket_batch_sizes: List[int] = [64, 32, 16, 8, 4, 2]
                 ):
        """
        Args:
            dataset: HuggingFace Dataset, but in general can use for any PyTorch Dataset class
            bucket_boundaries: define the maximum duration of audio sample in each bucket
            bucket_batch_sizes: define the batch size specified when loading samples from each bucket
        """
        super().__init__()
        self.bucket_boundaries = bucket_boundaries
        self.bucket_batch_sizes = bucket_batch_sizes
        self.dataset = dataset

        idx2length = []
        for idx, d in enumerate(self.dataset):
            idx2length.append((idx, d['duration']))
        self.idx2length = idx2length

    def element_to_bucket_id(self, seq_length):
        """
        Receive duration of an audio sample, return the index of the corresponding bucket.
        duration <= bucket duration boundary
        """
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.less(buckets_min, seq_length),
            np.less_equal(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

    def generate_batch_map(self):
        shuffle(self.idx2length)
        batch_map = dict()
        for idx, length in self.idx2length:
            bucket_id = self.element_to_bucket_id(length)
            if bucket_id not in batch_map:
                batch_map[bucket_id] = [idx]
            else:
                batch_map[bucket_id].append(idx)

        batch_list = []
        for batch_size_idx, data_incdices in batch_map.items():
            for group in [data_incdices[i:i + self.bucket_batch_sizes[batch_size_idx]] for i in
                          range(0, len(data_incdices), self.bucket_batch_sizes[batch_size_idx])]:
                batch_list.append(group)
        return batch_list

    @staticmethod
    def collate_fn(batch):
        """
        Just a toy collate_fn to test whether this bucket sampler work correctly or not
        """
        for data in batch:
            print(data['audio']['array'].shape[0] / 16000)
        return pad_sequence([torch.tensor(data['audio']['array']) for data in batch], batch_first=True)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        batch_list = self.generate_batch_map()
        shuffle(batch_list)
        for batch in batch_list:
            yield batch
