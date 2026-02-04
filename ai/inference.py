import sys
import os
# 将当前文件的父目录（也就是根目录 CatInvBridge）加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import h5py
import json
import numpy as np
from model import CatInvTransformer
from core.bridge import WHUToInventorBridge
from core.macro import *

def inference(h5_path, model_path, output_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatInvTransformer(input_dim=33, max_seq_len=100).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with h5py.File(h5_path, 'r') as f:
        vec = f['vec'][:]
    
    original_len = vec.shape[0]
    input_vec = np.zeros((1, 100, 33))
    input_vec[0, :original_len, :] = vec
    input_tensor = torch.FloatTensor(input_vec).to(device)
    input_tensor[:, :, 1:] /= 255.0 

    with torch.no_grad():
        output = model(input_tensor)
    
    # 1. 还原 AI 预测值到 0-255
    refined_vec = output[0, :original_len, :].cpu().numpy()
    refined_vec[:, 1:] *= 255.0
    refined_vec[:, 0] = vec[:, 0] # 保持原始指令 ID
    
    # 2. 强力规整：AI 预测后的值先做一次四舍五入
    refined_vec = np.round(refined_vec)

    # 3. 核心修复：强制草图闭合逻辑
    # 遍历向量，寻找每个 Loop 的终点，强行将其改为起点
    loop_start_pt = None
    for i in range(len(refined_vec)):
        cmd = int(refined_vec[i, 0])
        if cmd == SOL_IDX:
            # 记录起点 (如果是 SOL 携带坐标的情况)
            if refined_vec[i, 1] != -1:
                loop_start_pt = refined_vec[i, 1:3].copy()
            else:
                # 如果 SOL 没坐标，下一条 LINE 的起点就是 last_point，
                # 我们在后面第一条 LINE 处记录
                loop_start_pt = None
        
        elif cmd == LINE_IDX or cmd == ARC_IDX:
            if loop_start_pt is None:
                # 记录这一组草图的第一条线的“逻辑起点”
                # 在我们的 bridge 里，第一条线的起点默认是 128, 128 或 SOL 定义的值
                loop_start_pt = np.array([128.0, 128.0]) # 默认值
            
        elif cmd in [EXT_IDX, POCKET_IDX, REV_IDX]:
            # 关键：在拉伸发生前，强行把上一条线的终点(index 1,2) 设为 loop_start_pt
            if i > 0 and loop_start_pt is not None:
                refined_vec[i-1, 1:3] = loop_start_pt
                print(f"DEBUG: 已强制闭合第 {i} 步之前的草图至 {loop_start_pt}")
            loop_start_pt = None # 重置，等待下一个特征

    # 4. 生成 JSON
    bridge = WHUToInventorBridge(snap_to_grid=True)
    common_objs = bridge.vector_to_common(refined_vec)
    
    # 5. 缩放逻辑 (0.1)
    scale_factor = 0.1
    CENTER = 128.0
    for obj in common_objs:
        obj.distance *= scale_factor
        obj.plane_origin = [float((x - CENTER) * scale_factor) for x in obj.plane_origin]
        
        for ent in obj.sketch_entities:
            # 必须确保 start_pt 和 end_pt 在计算后依然完全相等
            if ent.start_pt:
                ent.start_pt = [float(round((x - CENTER) * scale_factor, 4)) for x in ent.start_pt]
            if ent.end_pt:
                ent.end_pt = [float(round((x - CENTER) * scale_factor, 4)) for x in ent.end_pt]
            if ent.center:
                ent.center = [float(round((x - CENTER) * scale_factor, 4)) for x in ent.center]
            if ent.radius:
                ent.radius = float(round(ent.radius * scale_factor, 4))

    inv_json = bridge.common_to_inventor_json(common_objs)
    
    with open(output_json, 'w') as f:
        json.dump(inv_json, f, indent=2)
    print(f"✨ 拓扑修复后的 JSON 已保存")

if __name__ == "__main__":
    inference(
        h5_path=r"D:\0_WYW_0\WHU\WHUCAD-lab\CatInvBridge\00032641.h5",
        model_path="checkpoints/catinv_model_e100.pth",
        output_json="refined_ai_model.json"
    )