import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np

class CADVectorDataset(Dataset):
    def __init__(self, h5_dir, max_len=60):
        self.files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as f:
            vec = f['vec'][:] # (SeqLen, 33)
            
        # 数据填充 (Padding)
        seq_len = vec.shape[0]
        if seq_len < self.max_len:
            pad = np.tile(vec[-1], (self.max_len - seq_len, 1)) # 用最后一个向量填充 (通常是 EOS)
            vec = np.vstack([vec, pad])
        else:
            vec = vec[:self.max_len, :]

        # 转换为 Tensor
        return torch.from_numpy(vec).float()