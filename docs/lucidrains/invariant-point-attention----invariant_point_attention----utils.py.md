# `.\lucidrains\invariant-point-attention\invariant_point_attention\utils.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 sin, cos, atan2, acos 函数
from torch import sin, cos, atan2, acos
# 从 functools 库中导入 wraps 装饰器
from functools import wraps

# 定义一个装饰器函数，将输入转换为 torch 张量
def cast_torch_tensor(fn):
    # 定义内部函数，用于实际执行函数并进行类型转换
    @wraps(fn)
    def inner(t):
        # 如果输入不是 torch 张量，则将其转换为 torch 张量
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.get_default_dtype())
        # 调用原始函数并返回结果
        return fn(t)
    # 返回内部函数
    return inner

# 使用装饰器将 rot_z 函数转换为接受 torch 张量作为输入的函数
@cast_torch_tensor
def rot_z(gamma):
    # 返回绕 z 轴旋转角度 gamma 的旋转矩阵
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

# 使用装饰器将 rot_y 函数转换为接受 torch 张量作为输入的函数
@cast_torch_tensor
def rot_y(beta):
    # 返回绕 y 轴旋转角度 beta 的旋转矩阵
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

# 定义一个函数，通过组合旋转矩阵实现绕不同轴的旋转
def rot(alpha, beta, gamma):
    # 返回绕 z 轴旋转角度 alpha、绕 y 轴旋转角度 beta、绕 z 轴旋转角度 gamma 的组合旋转矩阵
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)
```