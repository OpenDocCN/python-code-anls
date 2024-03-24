# `.\lucidrains\equiformer-pytorch\equiformer_pytorch\basis.py`

```py
# 导入必要的库
import os
from itertools import product
from collections import namedtuple

import torch
from einops import rearrange, repeat, reduce, einsum

# 导入自定义模块中的函数和工具
from equiformer_pytorch.irr_repr import (
    irr_repr,
    rot_to_euler_angles
)

from equiformer_pytorch.utils import (
    torch_default_dtype,
    cache_dir,
    exists,
    default,
    to_order,
    identity,
    l2norm,
    slice_for_centering_y_to_x
)

# 定义常量

# 设置缓存路径
CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.equivariant_attention'))
# 如果存在 CLEAR_CACHE 环境变量，则将缓存路径设置为 None
CACHE_PATH = CACHE_PATH if not exists(os.environ.get('CLEAR_CACHE')) else None

# 随机角度矩阵
RANDOM_ANGLES = torch.tensor([
    [4.41301023, 5.56684102, 4.59384642],
    [4.93325116, 6.12697327, 4.14574096],
    [0.53878964, 4.09050444, 5.36539036],
    [2.16017393, 3.48835314, 5.55174441],
    [2.52385107, 0.2908958, 3.90040975]
], dtype = torch.float64)

# 定义函数

# 获取矩阵的核空间的正交基
def get_matrix_kernel(A, eps = 1e-10):
    '''
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    '''
    A = rearrange(A, '... d -> (...) d')
    _u, s, v = torch.svd(A)
    kernel = v.t()[s < eps]
    return kernel

# 生成解决子空间 J 中 Sylvester 方程的 Kronecker 乘积矩阵
def sylvester_submatrix(order_out, order_in, J, a, b, c):
    ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
    angles = torch.stack((a, b, c), dim = -1)

    R_tensor = get_R_tensor(order_out, order_in, a, b, c)  # [m_out * m_in, m_out * m_in]

    R_irrep_J = irr_repr(J, angles)  # [m, m]
    R_irrep_J_T = rearrange(R_irrep_J, '... m n -> ... n m')

    R_tensor_identity = torch.eye(R_tensor.shape[-1])
    R_irrep_J_identity = torch.eye(R_irrep_J.shape[-1])

    return kron(R_tensor, R_irrep_J_identity) - kron(R_tensor_identity, R_irrep_J_T)  # [(m_out * m_in) * m, (m_out * m_in) * m]

# 计算两个矩阵的 Kronecker 乘积
def kron(a, b):
    """
    A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    res = einsum(a, b, '... i j, ... k l -> ... i k j l')
    return rearrange(res, '... i j k l -> ... (i j) (k l)')

# 获取 R 张量
def get_R_tensor(order_out, order_in, a, b, c):
    angles = torch.stack((a, b, c), dim = -1)
    return kron(irr_repr(order_out, angles), irr_repr(order_in, angles))

# 装饰器函数，用于缓存目录和设置默认数据类型
@cache_dir(CACHE_PATH)
@torch_default_dtype(torch.float64)
@torch.no_grad()
def basis_transformation_Q_J(J, order_in, order_out, random_angles = RANDOM_ANGLES):
    """
    :param J: order of the spherical harmonics
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: one part of the Q^-1 matrix of the article
    """
    sylvester_submatrices = sylvester_submatrix(order_out, order_in, J, *random_angles.unbind(dim = -1))
    null_space = get_matrix_kernel(sylvester_submatrices)

    assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
    Q_J = null_space[0] # [(m_out * m_in) * m]

    Q_J = rearrange(
        Q_J,
        '(oi m) -> oi m',
        m = to_order(J)
    )

    return Q_J.float()  # [m_out * m_in, m]

# 装饰器函数，用于缓存目录和设置默认数据类型
@cache_dir(CACHE_PATH)
@torch_default_dtype(torch.float64)
@torch.no_grad()
def get_basis(max_degree):
    """
    Return equivariant weight basis (basis)
    assuming edges are aligned to z-axis
    """
    basis = dict()

    # Equivariant basis (dict['<d_in><d_out>'])
    # 遍历输入输出度数的组合
    for d_in, d_out in product(range(max_degree+1), range(max_degree+1):
        # 存储每个 K_J 的列表
        K_Js = []

        # 计算输入输出度数的最小值
        d_min = min(d_in, d_out)

        # 将度数转换为顺序
        m_in, m_out, m_min = map(to_order, (d_in, d_out, d_min))
        # 为中心化 y 到 x 创建切片
        slice_in, slice_out = map(lambda t: slice_for_centering_y_to_x(t, m_min), (m_in, m_out))

        # 如果最小度数为0，则跳过当前循环
        if d_min == 0:
            continue

        # 遍历 J 的范围
        for J in range(abs(d_in - d_out), d_in + d_out + 1):

            # 获取球谐投影矩阵
            Q_J = basis_transformation_Q_J(J, d_in, d_out)

            # 将边（r_ij）与 z 轴对齐会导致稀疏球谐函数（例如度数为1 [0., 1., 0.]）- 因此只提取 mo 索引
            # 然后通过 equiformer v2 对 Y 进行归一化，以完全移除它
            mo_index = J
            K_J = Q_J[..., mo_index]

            # 重新排列 K_J 的维度
            K_J = rearrange(K_J, '... (o i) -> ... o i', o = m_out)
            K_J = K_J[..., slice_out, slice_in]

            # 对 K_J 进行降维操作，将矩阵转换为一维数组
            K_J = reduce(K_J, 'o i -> i', 'sum') # 矩阵是一个稀疏对角矩阵，但根据 J 的奇偶性会翻转

            # 将 K_J 添加到 K_Js 列表中
            K_Js.append(K_J)

        # 在最后一个维度上堆叠 K_Js 列表中的张量
        K_Js = torch.stack(K_Js, dim = -1)

        # 将 K_Js 存储到 basis 字典中
        basis[f'({d_in},{d_out})'] = K_Js # (mi, mf)

    # 返回 basis 字典
    return basis
# 用于将向量 r_ij 旋转到 z 轴的函数

def rot_x_to_y_direction(x, y, eps = 1e-6):
    '''
    将向量 x 旋转到与向量 y 相同的方向
    参考 https://math.stackexchange.com/a/2672702
    这种表述虽然不是最短路径，但旋转矩阵是对称的；两次旋转后可以回到 x
    '''
    n, dtype, device = x.shape[-1], x.dtype, x.device

    # 创建单位矩阵
    I = torch.eye(n, device = device, dtype = dtype)

    # 如果 x 和 y 非常接近，则返回单位矩阵
    if torch.allclose(x, y, atol = 1e-6):
        return I

    # 将 x 和 y 转换为双精度
    x, y = x.double(), y.double()

    # 对 x 和 y 进行 L2 范数归一化
    x, y = map(l2norm, (x, y))

    # 计算 x + y 和 x + y 的转置
    xy = rearrange(x + y, '... n -> ... n 1')
    xy_t = rearrange(xy, '... n 1 -> ... 1 n')

    # 计算旋转矩阵 R
    R = 2 * (xy @ xy_t) / (xy_t @ xy).clamp(min = eps) - I
    return R.type(dtype)

@torch.no_grad()
def get_D_to_from_z_axis(r_ij, max_degree):
    device, dtype = r_ij.device, r_ij.dtype

    D = dict()

    # 预先计算 D
    # 1. 计算旋转到 [0., 1., 0.] 的旋转
    # 2. 从该旋转计算 ZYZ 欧拉角
    # 3. 从 0 ... max_degree 计算 D 不可约表示（实际上不需要 0）

    z_axis = r_ij.new_tensor([0., 1., 0.])

    # 将 r_ij 旋转到 z 轴
    R = rot_x_to_y_direction(r_ij, z_axis)

    # 计算欧拉角
    angles = rot_to_euler_angles(R)

    # 计算每个阶数的 D 不可约表示
    for d in range(max_degree + 1):
        if d == 0:
            continue

        D[d] = irr_repr(d, angles)

    return D
```