# `.\lucidrains\equiformer-pytorch\equiformer_pytorch\irr_repr.py`

```
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 functools 模块中导入 partial 函数
from functools import partial

# 导入 torch 库
import torch
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 torch 库中导入 sin, cos, atan2, acos 函数
from torch import sin, cos, atan2, acos

# 从 einops 库中导入 rearrange, pack, unpack 函数
from einops import rearrange, pack, unpack

# 从 equiformer_pytorch.utils 模块中导入 exists, default, cast_torch_tensor, to_order, identity, l2norm 函数
from equiformer_pytorch.utils import (
    exists,
    default,
    cast_torch_tensor,
    to_order,
    identity,
    l2norm
)

# 定义 DATA_PATH 变量为当前文件路径的父目录下的 'data' 文件夹
DATA_PATH = Path(__file__).parents[0] / 'data'
# 定义 path 变量为 DATA_PATH 下的 'J_dense.pt' 文件
path = DATA_PATH / 'J_dense.pt'
# 从 'J_dense.pt' 文件中加载数据，赋值给 Jd 变量
Jd = torch.load(str(path))

# 定义 pack_one 函数，用于将输入张量 t 按照指定模式 pattern 进行打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 定义 unpack_one 函数，用于将输入张量 t 按照指定模式 pattern 进行解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 定义 wigner_d_matrix 函数，用于创建 ZYZ 欧拉角批量的维格纳 D 矩阵
def wigner_d_matrix(degree, alpha, beta, gamma, dtype = None, device = None):
    """Create wigner D matrices for batch of ZYZ Euler angles for degree l."""
    # 获取批量大小
    batch = alpha.shape[0]
    # 从 Jd 中获取 degree 对应的张量 J
    J = Jd[degree].type(dtype).to(device)
    # 根据 degree 创建对应的排序 order
    order = to_order(degree)
    # 计算 ZYZ 欧拉角的旋转矩阵
    x_a = z_rot_mat(alpha, degree)
    x_b = z_rot_mat(beta, degree)
    x_c = z_rot_mat(gamma, degree)
    res = x_a @ J @ x_b @ J @ x_c
    return res.view(batch, order, order)

# 定义 z_rot_mat 函数，用于创建绕 Z 轴旋转的旋转矩阵
def z_rot_mat(angle, l):
    device, dtype = angle.device, angle.dtype

    # 获取批量大小
    batch = angle.shape[0]
    # 创建 arange 函数的部分应用，指定设备为 device
    arange = partial(torch.arange, device = device)

    # 根据 degree 创建对应的排序 order
    order = to_order(l)

    # 初始化旋转矩阵 m
    m = angle.new_zeros((batch, order, order))

    # 创建批量范围
    batch_range = arange(batch, dtype = torch.long)[..., None]
    inds = arange(order, dtype = torch.long)[None, ...]
    reversed_inds = arange(2 * l, -1, -1, dtype = torch.long)[None, ...]
    frequencies = arange(l, -l - 1, -1, dtype = dtype)[None]

    # 计算旋转矩阵的值
    m[batch_range, inds, reversed_inds] = sin(frequencies * angle[..., None])
    m[batch_range, inds, inds] = cos(frequencies * angle[..., None])
    return m

# 定义 irr_repr 函数，用于计算 SO3 的不可约表示
def irr_repr(order, angles):
    """
    irreducible representation of SO3 - accepts multiple angles in tensor
    """
    dtype, device = angles.dtype, angles.device
    angles, ps = pack_one(angles, '* c')

    alpha, beta, gamma = angles.unbind(dim = -1)
    rep = wigner_d_matrix(order, alpha, beta, gamma, dtype = dtype, device = device)

    return unpack_one(rep, ps, '* o1 o2')

# 将 rot_z 函数的输出转换为 torch 张量
@cast_torch_tensor
def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    c = cos(gamma)
    s = sin(gamma)
    z = torch.zeros_like(gamma)
    o = torch.ones_like(gamma)

    out = torch.stack((
        c, -s, z,
        s, c, z,
        z, z, o
    ), dim = -1)

    return rearrange(out, '... (r1 r2) -> ... r1 r2', r1 = 3)

# 将 rot_y 函数的输出转换为 torch 张量
@cast_torch_tensor
def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    c = cos(beta)
    s = sin(beta)
    z = torch.zeros_like(beta)
    o = torch.ones_like(beta)

    out = torch.stack((
        c, z, s,
        z, o, z,
        -s, z, c
    ), dim = -1)

    return rearrange(out, '... (r1 r2) -> ... r1 r2', r1 = 3)

# 定义 rot 函数，用于计算 ZYZ 欧拉角的旋转矩阵
def rot(alpha, beta, gamma):
    '''
    ZYZ Euler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

# 定义 rot_to_euler_angles 函数，用于将旋转矩阵转换为 ZYZ 欧拉角
def rot_to_euler_angles(R):
    '''
    Rotation matrix to ZYZ Euler angles
    '''
    device, dtype = R.device, R.dtype
    xyz = R @ torch.tensor([0.0, 1.0, 0.0], device = device, dtype = dtype)
    xyz = l2norm(xyz).clamp(-1., 1.)

    b = acos(xyz[..., 1])
    a = atan2(xyz[..., 0], xyz[..., 2])

    R = rot(a, b, torch.zeros_like(a)).transpose(-1, -2) @ R
    c = atan2(R[..., 0, 2], R[..., 0, 0])
    return torch.stack((a, b, c), dim = -1)
```