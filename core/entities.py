class NeutralSketchEntity:
    def __init__(self, type, start_pt=None, end_pt=None, center=None, radius=None, **kwargs):
        self.type = type
        self.start_pt = start_pt # (x, y)
        self.end_pt = end_pt     # (x, y)
        self.center = center     # (x, y)
        self.radius = radius
        self.extra = kwargs      # 存放如 sweep_angle 等

class NeutralFeature:
    def __init__(self, type):
        self.type = type
        self.sketch_entities = []
        self.operation = "kJoinOperation"
        self.distance = 0.0
        # 绘图平面参数
        self.plane_origin = [0, 0, 0]
        self.plane_normal = [0, 0, 1]
        self.plane_x = [1, 0, 0]