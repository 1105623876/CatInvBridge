from .macro import *
from .entities import NeutralSketchEntity, NeutralFeature
from .geometry_utils import polar_to_cartesian, get_arc_center

class WHUToInventorBridge:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_sketch_entities = []
        self.last_point = [0.0, 0.0]

    def vector_to_common(self, whu_vectors):
        features = []
        for vec in whu_vectors:
            cmd = int(vec[0])
        
            # --- 草图阶段 ---
            if cmd == LINE_IDX:
                end_pt = vec[1:3].tolist()
                self.current_sketch_entities.append(
                    NeutralSketchEntity('Line', start_pt=self.last_point, end_pt=end_pt))
                self.last_point = end_pt
                
            elif cmd == ARC_IDX:
                end_pt = vec[1:3].tolist()
                # 假设 vec[3] 是角度, vec[4] 是方向
                center, radius = get_arc_center(self.last_point, end_pt, vec[3], vec[4])
                self.current_sketch_entities.append(
                    NeutralSketchEntity('Arc', start_pt=self.last_point, end_pt=end_pt, 
                                      center=center, radius=radius, sweep=vec[3]))
                self.last_point = end_pt

            elif cmd == CIRCLE_IDX:
                center = vec[1:3].tolist()
                radius = vec[5] # 根据 macro 定义
                self.current_sketch_entities.append(
                    NeutralSketchEntity('Circle', center=center, radius=radius))

            # --- 特征生成阶段 ---
            elif cmd in [EXT_IDX, POCKET_IDX, REV_IDX]:
                feat = self._parse_feature(cmd, vec)
                features.append(feat)
                self.reset() # 完成一个特征，重置草图状态
                
        return features

    def _parse_feature(self, cmd, vec):
        # 识别特征类型
        f_type = "ExtrudeFeature"
        if cmd == REV_IDX: f_type = "RevolveFeature"
        
        feat = NeutralFeature(f_type)
        feat.sketch_entities = self.current_sketch_entities
        
        # 1. 解析平面 (Theta, Phi -> Normal)
        theta, phi, gamma = vec[OFFSET_PLANE : OFFSET_PLANE+3]
        feat.plane_normal = polar_to_cartesian(theta, phi).tolist()
        feat.plane_origin = vec[OFFSET_TRANS : OFFSET_TRANS+3].tolist()
        
        # 2. 解析特征参数
        if cmd in [EXT_IDX, POCKET_IDX]:
            feat.distance = vec[OFFSET_BODY]
            feat.operation = "kCutOperation" if cmd == POCKET_IDX else "kJoinOperation"
        elif cmd == REV_IDX:
            feat.operation = "kJoinOperation"
            feat.angle = vec[OFFSET_BODY + 4] # WHUCAD angle_one 偏移
            
        return feat

    def common_to_inventor_json(self, common_features):
        inv_json = []
        for feat in common_features:
            # 基础结构
            item = {
                "type": feat.type,
                "name": f"{feat.type}_{len(inv_json)+1}",
                "operation": feat.operation,
                "isTwoDirectional": False,
                "profile": self._build_profile(feat)
            }
            
            # 特征特定参数
            if feat.type == "ExtrudeFeature":
                item.update({
                    "extentType": "kDistanceExtent",
                    "extent": {
                        "distance": {"value": feat.distance}, 
                        "direction": "kPositiveExtentDirection"
                    },
                    # 如果你的 rebuild 代码还检查 extentTwo，最好也给个默认值
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
        """构建 Inventor 复杂的 profile 结构"""
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
        """为 Revolve 提供默认旋转轴 (通常是 Y 轴)"""
        return {
            "metaType": "AxisEntity",
            "axisInfo": {
                "start_point": {"x": 0, "y": 0, "z": 0},
                "direction": {"x": 0, "y": 1, "z": 0}
            }
        }