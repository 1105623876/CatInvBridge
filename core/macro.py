import numpy as np

# WHUCAD 指令索引
LINE_IDX = 0
ARC_IDX = 1
CIRCLE_IDX = 2
SPLINE_IDX = 3
SCP_IDX = 4     # Spline Control Point
EOS_IDX = 5     # End of Sequence
SOL_IDX = 6     # Start of Loop
EXT_IDX = 7     # Extrude
REV_IDX = 8     # Revolve
POCKET_IDX = 9  # Pocket (Extrude-Cut)

# 参数偏移量 (基于 WHUCAD N_ARGS 定义)
# 向量结构: [CMD_ID, SKETCH_ARGS(5), PLANE_ARGS(3), TRANS_ARGS(4), BODY_ARGS(7), ...]
OFFSET_SKETCH = 1
OFFSET_PLANE = 6
OFFSET_TRANS = 9
OFFSET_BODY = 13

# 常用字符串映射
BOOLEAN_MAP = ["kJoinOperation", "kCutOperation", "kIntersectOperation"]