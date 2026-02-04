import numpy as np
import torch

def clean_and_normalize_vector(vec):
    """
    人为‘净化’向量，模拟理想的参数。
    比如：把靠近 128.0 的值强制设为 128.0，模拟‘对齐’。
    """
    cmd = vec[0]
    args = vec[1:]
    
    # 模拟净化：将坐标四舍五入到最近的规整值
    # 这样模型能学会从‘脏数据’恢复到‘干净数据’
    clean_args = np.round(args) 
    
    return np.concatenate([[cmd], clean_args])

def create_training_pair(original_vec):
    """
    生成一个训练对：(带噪声的输入, 干净的目标)
    """
    # 输入：原始 H5 里的数据（已经有量化噪声了）
    x = original_vec.copy()
    
    # 目标：我们希望它变成的样子（更规整的数值）
    y = clean_and_normalize_vector(original_vec)
    
    return x, y