import numpy as np

def polar_to_cartesian(theta, phi, r=1.0):
    """球面坐标转笛卡尔向量 (用于处理平面法向)"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def get_arc_center(start_pt, end_pt, sweep_angle, clock_sign):
    """
    根据起点、终点、扫掠角和顺逆时针计算圆弧中心
    这是将 WHUCAD 转为 Inventor kCircularArcCurve2d 的关键
    """
    p1 = np.array(start_pt)
    p2 = np.array(end_pt)
    d = np.linalg.norm(p2 - p1)
    
    # 半径计算 r = (d/2) / sin(alpha/2)
    # 注意：WHUCAD 的参数通常经过了归一化，需要根据 ARGS_N 映射回弧度
    alpha = sweep_angle 
    if alpha <= 0: return p1.tolist() # 容错
    
    radius = (d / 2.0) / np.sin(alpha / 2.0)
    
    # 计算中点和垂直向量
    mid = (p1 + p2) / 2.0
    v = p2 - p1
    # 垂直向量 (2D)
    perp = np.array([-v[1], v[0]])
    perp = perp / np.linalg.norm(perp)
    
    # 计算弦心距
    h = np.sqrt(max(0, radius**2 - (d/2.0)**2))
    
    # 根据 clock_sign 确定中心点方向
    if clock_sign == 0: # 顺时针/逆时针 逻辑需对照 WHUCAD 源码确定
        center = mid + perp * h
    else:
        center = mid - perp * h
        
    return center.tolist(), radius