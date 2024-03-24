# `.\lucidrains\se3-transformer-pytorch\se3_transformer_pytorch\irr_repr.py`

```
# 导入所需的库
import os
import numpy as np
import torch
from torch import sin, cos, atan2, acos
from math import pi
from pathlib import Path
from functools import wraps

# 导入自定义的函数和类
from se3_transformer_pytorch.utils import exists, default, cast_torch_tensor, to_order
from se3_transformer_pytorch.spherical_harmonics import get_spherical_harmonics, clear_spherical_harmonics_cache

# 设置数据路径
DATA_PATH = path = Path(os.path.dirname(__file__)) / 'data'

# 尝试加载预先计算好的 J_dense 数据
try:
    path = DATA_PATH / 'J_dense.pt'
    Jd = torch.load(str(path))
except:
    # 如果加载失败，则加载 numpy 格式的数据并转换为 torch 格式
    path = DATA_PATH / 'J_dense.npy'
    Jd_np = np.load(str(path), allow_pickle = True)
    Jd = list(map(torch.from_numpy, Jd_np))

# 创建 Wigner D 矩阵
def wigner_d_matrix(degree, alpha, beta, gamma, dtype = None, device = None):
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    J = Jd[degree].type(dtype).to(device)
    order = to_order(degree)
    x_a = z_rot_mat(alpha, degree)
    x_b = z_rot_mat(beta, degree)
    x_c = z_rot_mat(gamma, degree)
    res = x_a @ J @ x_b @ J @ x_c
    return res.view(order, order)

# 创建绕 Z 轴旋转的矩阵
def z_rot_mat(angle, l):
    device, dtype = angle.device, angle.dtype
    order = to_order(l)
    m = angle.new_zeros((order, order))
    inds = torch.arange(0, order, 1, dtype=torch.long, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, dtype=torch.long, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)[None]

    m[inds, reversed_inds] = sin(frequencies * angle[None])
    m[inds, inds] = cos(frequencies * angle[None])
    return m

# 创建不可约表示
def irr_repr(order, alpha, beta, gamma, dtype = None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    cast_ = cast_torch_tensor(lambda t: t)
    dtype = default(dtype, torch.get_default_dtype())
    alpha, beta, gamma = map(cast_, (alpha, beta, gamma))
    return wigner_d_matrix(order, alpha, beta, gamma, dtype = dtype)

# 绕 Z 轴旋转
@cast_torch_tensor
def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

# 绕 Y 轴旋转
@cast_torch_tensor
def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

# 将球���上的点转换为 alpha 和 beta
@cast_torch_tensor
def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    x = x / torch.norm(x)
    beta = acos(x[2])
    alpha = atan2(x[1], x[0])
    return (alpha, beta)

# ZYZ 欧拉角旋转
def rot(alpha, beta, gamma):
    '''
    ZYZ Euler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

# 合成旋转
def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.])
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c

# 计算球谐函数
def spherical_harmonics(order, alpha, beta, dtype = None):
    return get_spherical_harmonics(order, theta = (pi - beta), phi = alpha)
```