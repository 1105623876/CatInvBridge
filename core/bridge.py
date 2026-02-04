import numpy as np
from .macro import *
from .entities import NeutralSketchEntity, NeutralFeature
from .geometry_utils import polar_to_cartesian, get_arc_center

class WHUToInventorBridge:
    def __init__(self, snap_to_grid=True):
        self.snap_to_grid = snap_to_grid # 是否开启网格捕捉
        self.reset()

    def reset(self):
        self.current_sketch_entities = []
        self.last_point = [128.0, 128.0] # 默认初始点在中心

    def _snap(self, val):
        """将数值捕捉到整数网格，消除量化抖动"""
        if not self.snap_to_grid or val == -1:
            return val
        return float(round(val))

    def vector_to_common(self, whu_vectors):
        features = []
        for vec in whu_vectors:
            cmd = int(vec[0])
        
            if cmd == SOL_IDX:
                if vec[1] != -1:
                    self.last_point = [self._snap(vec[1]), self._snap(vec[2])]
                continue
            # --- 草图阶段 ---
            if cmd == LINE_IDX:
                #end_pt = vec[1:3].tolist()
                # 对终点进行网格捕捉
                end_pt = [self._snap(vec[1]), self._snap(vec[2])]
                # 只有线段长度大于0才添加（防止量化导致的零长线段）
                if not np.allclose(self.last_point, end_pt):
                    self.current_sketch_entities.append(
                        NeutralSketchEntity('Line', start_pt=self.last_point, end_pt=end_pt))
                self.last_point = end_pt
                
            # 圆弧 (Arc)
            elif cmd == ARC_IDX:
                end_pt = [self._snap(vec[1]), self._snap(vec[2])]
                # sweep_angle 通常不需要完全 round 到整数，但可以 round 到 1 位小数
                sweep_angle = vec[3] 
                clock_sign = int(vec[4])
                
                # 计算圆心和半径
                try:
                    center, radius = get_arc_center(self.last_point, end_pt, sweep_angle, clock_sign)
                    # 对圆心进行网格捕捉
                    snapped_center = [self._snap(center[0]), self._snap(center[1])]
                    snapped_radius = self._snap(radius)
                    
                    entity = NeutralSketchEntity(
                        'Arc', 
                        start_pt=self.last_point, 
                        end_pt=end_pt,
                        center=snapped_center,
                        radius=snapped_radius,
                        sweep=sweep_angle,
                        clock=clock_sign
                    )
                    self.current_sketch_entities.append(entity)
                except Exception as e:
                    print(f"Warning: Failed to calculate Arc at step {i}: {e}")
                
                self.last_point = end_pt

            # 圆 (Circle)
            elif cmd == CIRCLE_IDX:
                center = [self._snap(vec[1]), self._snap(vec[2])]
                radius = self._snap(vec[5]) # 根据 macro 定义，半径通常在 index 5
                
                entity = NeutralSketchEntity('Circle', center=center, radius=radius)
                self.current_sketch_entities.append(entity)
                # 圆不改变 last_point，因为它不一定是连续路径的一部分

            # --- 2. 实体特征生成阶段 ---
            elif cmd in [EXT_IDX, POCKET_IDX, REV_IDX]:
                if not self.current_sketch_entities:
                    print(f"Warning: Feature {cmd} at step {i} has no sketch. Skipping.")
                    continue
                
                feat = self._parse_feature(cmd, vec)
                features.append(feat)
                self.reset() # 完成一个特征，重置草图环境
                
        return features

    def _parse_feature(self, cmd, vec):
        """解析 Extrude/Pocket/Revolve 的参数并应用 Snap"""
        f_type = "ExtrudeFeature"
        if cmd == REV_IDX: f_type = "RevolveFeature"
        
        feat = NeutralFeature(f_type)
        feat.sketch_entities = self.current_sketch_entities
        
        # 1. 平面定位与对齐
        theta, phi, gamma = vec[OFFSET_PLANE : OFFSET_PLANE+3]
        feat.plane_normal = polar_to_cartesian(theta, phi).tolist()
        feat.plane_origin = [self._snap(x) for x in vec[OFFSET_TRANS : OFFSET_TRANS+3]]
        
        # 2. 特征特定参数
        if cmd in [EXT_IDX, POCKET_IDX]:
            # 捕捉拉伸长度
            feat.distance = self._snap(vec[OFFSET_BODY])
            feat.operation = "kCutOperation" if cmd == POCKET_IDX else "kJoinOperation"
        
        elif cmd == REV_IDX:
            # 捕捉旋转角度 (角度通常不需要 round 到整数，保持弧度)
            feat.angle = vec[OFFSET_BODY + 4] 
            feat.operation = "kJoinOperation"
            
        return feat

    def common_to_inventor_json(self, common_features):
        """
        将中立对象序列化为 Inventor 重建引擎需要的字典格式
        """
        inv_json = []
        for feat in common_features:
            item = {
                "type": feat.type,
                "name": f"{feat.type}_{len(inv_json)+1}",
                "operation": feat.operation,
                "isTwoDirectional": False,
                "profile": self._build_profile(feat)
            }
            
            if feat.type == "ExtrudeFeature":
                item.update({
                    "extentType": "kDistanceExtent",
                    "extent": {
                        "distance": {"value": feat.distance},
                        "direction": "kPositiveExtentDirection"
                    },
                    "extentTwo": None
                })
            elif feat.type == "RevolveFeature":
                item.update({
                    "extentType": "kAngleExtent",
                    "extent": {
                        "angle": {"value": getattr(feat, 'angle', 6.28)},
                        "direction": "kPositiveExtentDirection"
                    },
                    "axisEntity": self._build_default_axis()
                })
            
            inv_json.append(item)
        return inv_json

    def _build_profile(self, feat):
        """将中立草图实体列表转换为 Inventor 的 Profile 结构"""
        entities = []
        for ne in feat.sketch_entities:
            if ne.type == 'Line':
                entities.append({
                    "CurveType": "kLineSegmentCurve2d",
                    "StartSketchPoint": {"x": ne.start_pt[0], "y": ne.start_pt[1]},
                    "EndSketchPoint": {"x": ne.end_pt[0], "y": ne.end_pt[1]}
                })
            elif ne.type == 'Arc':
                entities.append({
                    "CurveType": "kCircularArcCurve2d",
                    "Curve": {
                        "center": {"x": ne.center[0], "y": ne.center[1]},
                        "radius": ne.radius,
                        "sweepAngle": ne.extra.get('sweep', 1.57)
                    },
                    "StartSketchPoint": {"x": ne.start_pt[0], "y": ne.start_pt[1]},
                    "EndSketchPoint": {"x": ne.end_pt[0], "y": ne.end_pt[1]}
                })
            elif ne.type == 'Circle':
                entities.append({
                    "CurveType": "kCircleCurve2d",
                    "Curve": {
                        "center": {"x": ne.center[0], "y": ne.center[1]},
                        "radius": ne.radius
                    }
                })
        
        return {
            "SketchPlane": {
                "geometry": {
                    "origin": {"x": feat.plane_origin[0], "y": feat.plane_origin[1], "z": feat.plane_origin[2]},
                    "normal": {"x": feat.plane_normal[0], "y": feat.plane_normal[1], "z": feat.plane_normal[2]},
                    "axis_x": {"x": 1, "y": 0, "z": 0}, "axis_y": {"x": 0, "y": 1, "z": 0}
                }
            },
            "ProfilePaths": [{"PathEntities": entities}]
        }

    def _build_default_axis(self):
        """为旋转特征提供一个默认的旋转轴 (中心 Y 轴)"""
        return {
            "metaType": "AxisEntity",
            "axisInfo": {
                "start_point": {"x": 0, "y": 0, "z": 0},
                "direction": {"x": 0, "y": 1, "z": 0}
            }
        }