import h5py
import json
import os
import numpy as np
from core.bridge import WHUToInventorBridge

def process_h5_file(h5_path, output_json_path, scale_factor=1.0):
    """
    读取 WHUCAD 的 .h5 向量文件并转换为 Inventor JSON
    :param h5_path: .h5 文件路径
    :param output_json_path: 输出 JSON 路径
    :param scale_factor: 缩放因子。WHUCAD 数据通常是归一化的(-1到1 或 0到256)，
                         Inventor 需要真实尺寸，建议设为 100.0 或根据 bounding_size 调整
    """
    if not os.path.exists(h5_path):
        print(f"Error: File {h5_path} not found.")
        return

    # 1. 加载 H5 数据
    with h5py.File(h5_path, 'r') as f:
        # WHUCAD 的 key 通常是 'vec'
        if 'vec' not in f.keys():
            print(f"Error: 'vec' dataset not found in {h5_path}. Keys: {list(f.keys())}")
            return
        
        whu_vectors = f['vec'][:] # 形状通常是 (N, 42)
    
    print(f"Successfully loaded {len(whu_vectors)} vectors from {h5_path}")

    # 2. 调用桥接器进行转换
    bridge = WHUToInventorBridge()
    common_objs = bridge.vector_to_common(whu_vectors)
    
    # --- 核心修复：相对于中心点 128 缩放 ---
    CENTER = 128.0 

    for obj in common_objs:
        # 1. 缩放拉伸长度 (长度是相对值，直接乘)
        obj.distance *= scale_factor
        
        # 2. 缩放平面原点 (这是绝对坐标，需要减去中心点再缩放)
        origin = np.array(obj.plane_origin)
        obj.plane_origin = ((origin - CENTER) * scale_factor).tolist()
        
        # 3. 缩放草图内的点 (同理，草图点通常也是相对于 128 偏移的)
        for ent in obj.sketch_entities:
            if ent.start_pt:
                pt = np.array(ent.start_pt)
                # 注意：如果草图坐标已经是相对于平面原点的局部坐标，则不需要减 CENTER
                # 但 WHUCAD 向量中的草图点通常还是量化在 0-255 空间的
                ent.start_pt = ((pt - CENTER) * scale_factor).tolist()
            if ent.end_pt:
                pt = np.array(ent.end_pt)
                ent.end_pt = ((pt - CENTER) * scale_factor).tolist()
            if ent.center:
                pt = np.array(ent.center)
                ent.center = ((pt - CENTER) * scale_factor).tolist()
            if ent.radius:
                ent.radius *= scale_factor

    # 3. 转为 Inventor JSON
    inventor_dict = bridge.common_to_inventor_json(common_objs)

    # 4. 保存文件
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(inventor_dict, f_out, indent=2)
    
    print(f"Conversion complete. JSON saved to: {output_json_path}")
    print(f"Total Features converted: {len(inventor_dict)}")

if __name__ == "__main__":
    input_h5 = "00032641.h5" 
    output_json = "converted_for_inventor.json"
    
    # 运行转换
    process_h5_file(input_h5, output_json, scale_factor=1)