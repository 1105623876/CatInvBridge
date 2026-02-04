import h5py
import numpy as np

h5_path = r"D:\0_WYW_0\WHU\WHUCAD-lab\CatInvBridge\00032641.h5"

with h5py.File(h5_path, 'r') as f:
    print("Keys in H5:", list(f.keys()))
    for key in f.keys():
        data = f[key][:]
        print(f"Dataset '{key}' shape: {data.shape}")
        if len(data) > 0:
            print(f"Sample row (first 10 elements of first row): {data[0][:10]}")