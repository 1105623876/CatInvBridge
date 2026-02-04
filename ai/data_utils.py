import random
import numpy as np

def generate_random_extrude_pair():
    """
    随机生成一个『拉伸』特征的成对数据
    """
    # 1. 随机产生理想参数
    w = random.uniform(5.0, 50.0)
    h = random.uniform(5.0, 50.0)
    dist = random.uniform(10.0, 100.0)
    
    # 2. 模拟 CATIA 侧：加入量化噪声 (模拟 0-255 的离散感)
    cat_w = round(w * 2.5) / 2.5 
    cat_dist = round(dist * 2.5) / 2.5
    cat_vec = np.zeros(33)
    cat_vec[0] = 7 # Extrude
    cat_vec[13] = cat_dist # 假设的距离位
    
    # 3. 模拟 Inventor 侧：理想的连续值 JSON
    inv_json_params = {
        "distance": dist,
        "width": w,
        "height": h
    }
    
    return cat_vec, inv_json_params