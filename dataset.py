import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import OrderedDict, Counter

PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

########################################
########## Paraphrase Dataset ##########
########################################
class ParaphraseDataset(Dataset):
    def __init__(
        self, 
        source_file, 
        target_file, 
        tokenizer,
        max_length=50
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        with open(source_file, 'r') as f:
            self._source = [line.strip() for line in f]
        with open(target_file, 'r') as f:
            self._target = [line.strip() for line in f]

        assert len(self._source) == len(self._target)

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, idx):
        src, tgt = self._source[idx], self._target[idx]
        src = [BOS_INDEX] + self.tokenizer(src)[:self.max_length - 2] + [EOS_INDEX]
        tgt = [BOS_INDEX] + self.tokenizer(tgt)[:self.max_length - 2] + [EOS_INDEX]
        return src, tgt

    @staticmethod
    def collate_fn(data):
        srcs = [torch.tensor(d[0]) for d in data]
        srcs = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True)
        tgts = [torch.tensor(d[1]) for d in data]
        tgts = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True)
        return srcs, tgts
