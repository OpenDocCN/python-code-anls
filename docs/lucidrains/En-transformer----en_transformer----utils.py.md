# `.\lucidrains\En-transformer\en_transformer\utils.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 sin, cos, atan2, acos 函数
from torch import sin, cos, atan2, acos

# 定义绕 z 轴旋转的函数，参数为旋转角度 gamma
def rot_z(gamma):
    # 返回一个包含 z 轴旋转矩阵的张量
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype = gamma.dtype)

# 定义绕 y 轴旋转的函数，参数为旋转角度 beta
def rot_y(beta):
    # 返回一个包含 y 轴旋转矩阵的张量
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype = beta.dtype)

# 定义绕任意轴旋转的函数，参数为三个旋转角度 alpha, beta, gamma
def rot(alpha, beta, gamma):
    # 返回绕 z 轴、y 轴、z 轴旋转矩阵的乘积
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)
```