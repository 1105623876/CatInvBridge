import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class CADRefineDataset(Dataset):
    def __init__(self, list_file, max_len=100):
        with open(list_file, "r") as f:
            self.files = f.read().splitlines()
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            with h5py.File(self.files[idx], 'r') as f:
                vec = f['vec'][:] # (N, 33)
        except:
            # 容错：如果某个文件坏了，随机返回另一个
            return self.__getitem__(np.random.randint(0, len(self.files)))

        n = vec.shape[0]
        
        # 1. 准备 Target (标签)
        target = np.zeros((self.max_len, 33))
        # WHUCAD 常用 -1 填充，这里我们统一用 0 填充，但保留原始 Command
        actual_len = min(n, self.max_len)
        target[:actual_len, :] = vec[:actual_len, :]
        
        # 2. 生成 Mask (True 表示是 padding 部分)
        padding_mask = np.zeros(self.max_len, dtype=bool)
        padding_mask[actual_len:] = True
        
        # 3. 准备 Input (加噪)
        noisy_input = target.copy()
        # 关键：归一化。把 0-255 映射到 0-1 之间，模型收敛快 10 倍
        noisy_input[:, 1:] = noisy_input[:, 1:] / 255.0 
        target[:, 1:] = target[:, 1:] / 255.0
        
        # 只给有效序列部分加噪声
        noise = (np.random.rand(actual_len, 32) - 0.5) * (0.5 / 255.0) # 模拟微小波动
        noisy_input[:actual_len, 1:] += noise
        
        return {
            "input": torch.FloatTensor(noisy_input),
            "target": torch.FloatTensor(target),
            "mask": torch.BoolTensor(padding_mask)
        }