import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.macro import *
from core.geometry_utils import polar_to_cartesian

def visualize_cad_vectors(h5_path, scale=0.1, center=128.0):
    with h5py.File(h5_path, 'r') as f:
        vecs = f['vec'][:]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    current_sketch_segments = []
    last_pt_2d = np.array([center, center])
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    color_idx = 0

    for i, vec in enumerate(vecs):
        cmd = int(vec[0])
        
        if cmd == SOL_IDX:
            if vec[1] != -1: last_pt_2d = vec[1:3]
            continue

        if cmd == LINE_IDX:
            end_pt_2d = vec[1:3]
            current_sketch_segments.append((last_pt_2d.copy(), end_pt_2d.copy()))
            last_pt_2d = end_pt_2d

        elif cmd in [EXT_IDX, POCKET_IDX]:
            # 1. 提取平面参数
            theta, phi, gamma = vec[OFFSET_PLANE : OFFSET_PLANE+3]
            normal = polar_to_cartesian(theta, phi)
            origin = (vec[OFFSET_TRANS : OFFSET_TRANS+3] - center) * scale
            distance = vec[OFFSET_BODY] * scale
            
            # 2. 简单的坐标系对齐 (构建局部坐标系的 X 和 Y 轴)
            # 找一个不与法线平行的向量来做叉乘
            ref_vec = np.array([0, 1, 0]) if abs(normal[0]) > 0.9 else np.array([1, 0, 0])
            local_x = np.cross(ref_vec, normal)
            local_x /= np.linalg.norm(local_x)
            local_y = np.cross(normal, local_x)
            
            c = colors[color_idx % len(colors)]
            
            # 3. 转换并绘制草图
            for s2d, e2d in current_sketch_segments:
                # 将 2D 草图点转换为 3D 空间点
                def to_3d(pt2d):
                    u = (pt2d[0] - center) * scale
                    v = (pt2d[1] - center) * scale
                    return origin + u * local_x + v * local_y

                s3d = to_3d(s2d)
                e3d = to_3d(e2d)
                
                # 画底面草图线
                ax.plot([s3d[0], e3d[0]], [s3d[1], e3d[1]], [s3d[2], e3d[2]], color=c, linewidth=2)
                
                # 画顶面草图线 (底面 + 法线 * 距离)
                s3d_top = s3d + normal * distance
                e3d_top = e3d + normal * distance
                ax.plot([s3d_top[0], e3d_top[0]], [s3d_top[1], e3d_top[1]], [s3d_top[2], e3d_top[2]], 
                        color=c, linewidth=1, alpha=0.6)
                
                # 画侧边棱线 (连接底面和顶面)
                ax.plot([s3d[0], s3d_top[0]], [s3d[1], s3d_top[1]], [s3d[2], s3d_top[2]], 
                        color=c, linestyle='--', alpha=0.3)

            color_idx += 1
            current_sketch_segments = []
            last_pt_2d = np.array([center, center])

    # 设置相等的显示比例
    ax.set_box_aspect([1,1,1]) 
    max_range = 15.0 # 根据 scale=0.1 调整显示范围
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    plt.show()

if __name__ == "__main__":
    import os
    test_h5 = r"D:\0_WYW_0\WHU\WHUCAD-lab\CatInvBridge\00032641.h5"
    visualize_cad_vectors(test_h5, scale=0.1)