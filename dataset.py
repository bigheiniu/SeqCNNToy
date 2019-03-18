import torch
from torch.utils.data import Dataset
import numpy as np


class SeqDataSet(Dataset):
    def __init__(self, seq_data, label, num_class):
        super(SeqDataSet, self).__init__()
        self.seq_data = np.expand_dims(seq_data, axis=1)
        self.label = label
        self.num_class = num_class
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get label at index
        label = self.label[index] - 2
        label = np.array(label, dtype=np.long)
        one_hot_label = np.eye(self.num_class)[label]
        # 6 * 27 * 4
        seq = self.seq_data[index]
        return seq, one_hot_label