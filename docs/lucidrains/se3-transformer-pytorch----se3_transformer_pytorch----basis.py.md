# `.\lucidrains\se3-transformer-pytorch\se3_transformer_pytorch\basis.py`

```py
# 导入必要的库
import os
from math import pi
import torch
from torch import einsum
from einops import rearrange
from itertools import product
from contextlib import contextmanager

# 导入自定义库
from se3_transformer_pytorch.irr_repr import irr_repr, spherical_harmonics
from se3_transformer_pytorch.utils import torch_default_dtype, cache_dir, exists, default, to_order
from se3_transformer_pytorch.spherical_harmonics import clear_spherical_harmonics_cache

# 常量定义

# 设置缓存路径，默认为用户主目录下的.cache.equivariant_attention文件夹
CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.equivariant_attention'))
# 如果环境变量CLEAR_CACHE存在，则将缓存路径设为None
CACHE_PATH = CACHE_PATH if not exists(os.environ.get('CLEAR_CACHE')) else None

# 随机角度列表
# todo (figure ot why this was hard coded in official repo)
RANDOM_ANGLES = [ 
    [4.41301023, 5.56684102, 4.59384642],
    [4.93325116, 6.12697327, 4.14574096],
    [0.53878964, 4.09050444, 5.36539036],
    [2.16017393, 3.48835314, 5.55174441],
    [2.52385107, 0.2908958, 3.90040975]
]

# 辅助函数

# 空上下文管理器
@contextmanager
def null_context():
    yield

# 函数定义

def get_matrix_kernel(A, eps = 1e-10):
    '''
    计算矩阵A的核的正交基(x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: 矩阵
    :return: 每行是A核的基向量的矩阵
    '''
    _u, s, v = torch.svd(A)
    kernel = v.t()[s < eps]
    return kernel

def get_matrices_kernel(As, eps = 1e-10):
    '''
    计算所有矩阵As的公共核
    '''
    matrix = torch.cat(As, dim=0)
    return get_matrix_kernel(matrix, eps)

def get_spherical_from_cartesian(cartesian, divide_radius_by = 1.0):
    """
    将笛卡尔坐标转换为球坐标

    # ON ANGLE CONVENTION
    #
    # sh has following convention for angles:
    # :param theta: the colatitude / polar angle, ranging from 0(North Pole, (X, Y, Z) = (0, 0, 1)) to pi(South Pole, (X, Y, Z) = (0, 0, -1)).
    # :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    #
    # the 3D steerable CNN code therefore (probably) has the following convention for alpha and beta:
    # beta = pi - theta; ranging from 0(South Pole, (X, Y, Z) = (0, 0, -1)) to pi(North Pole, (X, Y, Z) = (0, 0, 1).
    # alpha = phi
    #
    """
    # 初始化返回数组
    spherical = torch.zeros_like(cartesian)

    # 索引
    ind_radius, ind_alpha, ind_beta = 0, 1, 2

    cartesian_x, cartesian_y, cartesian_z = 2, 0, 1

    # 获取在xy平面上的投影半径
    r_xy = cartesian[..., cartesian_x] ** 2 + cartesian[..., cartesian_y] ** 2

    # 获取第二个角度
    # 版本 'elevation angle defined from Z-axis down'
    spherical[..., ind_beta] = torch.atan2(torch.sqrt(r_xy), cartesian[..., cartesian_z])

    # 获取xy平面上的角度
    spherical[...,ind_alpha] = torch.atan2(cartesian[...,cartesian_y], cartesian[...,cartesian_x])

    # 获取整体半径
    radius = torch.sqrt(r_xy + cartesian[...,cartesian_z]**2)

    if divide_radius_by != 1.0:
        radius /= divide_radius_by

    spherical[..., ind_radius] = radius
    return spherical

def kron(a, b):
    """
    计算矩阵a和b的Kronecker积
    """
    res = einsum('... i j, ... k l -> ... i k j l', a, b)
    return rearrange(res, '... i j k l -> ... (i j) (k l)')

def get_R_tensor(order_out, order_in, a, b, c):
    return kron(irr_repr(order_out, a, b, c), irr_repr(order_in, a, b, c)

def sylvester_submatrix(order_out, order_in, J, a, b, c):
    ''' 生成用于在子空间J中解Sylvester方程的Kronecker积矩阵 '''
    R_tensor = get_R_tensor(order_out, order_in, a, b, c)  # [m_out * m_in, m_out * m_in]
    R_irrep_J = irr_repr(J, a, b, c)  # [m, m]

    R_tensor_identity = torch.eye(R_tensor.shape[0])
    R_irrep_J_identity = torch.eye(R_irrep_J.shape[0]
    # 计算两个张量的 Kronecker 乘积，并返回结果
    return kron(R_tensor, R_irrep_J_identity) - kron(R_tensor_identity, R_irrep_J.t())  # [(m_out * m_in) * m, (m_out * m_in) * m]
# 使用缓存目录装饰器，指定缓存路径为 CACHE_PATH
# 使用默认的 torch 浮点数类型为 float64 装饰器
# 禁用 torch 的梯度计算功能装饰器
def basis_transformation_Q_J(J, order_in, order_out, random_angles = RANDOM_ANGLES):
    """
    :param J: 球谐函数的阶数
    :param order_in: 输入表示的阶数
    :param order_out: 输出表示的阶数
    :return: 文章中 Q^-1 矩阵的一部分
    """
    # 生成 Sylvester 子矩阵列表
    sylvester_submatrices = [sylvester_submatrix(order_out, order_in, J, a, b, c) for a, b, c in random_angles]
    # 获取 Sylvester 子矩阵的零空间
    null_space = get_matrices_kernel(sylvester_submatrices)
    # 断言零空间的大小为 1，即唯一的子空间解
    assert null_space.size(0) == 1, null_space.size()
    # 获取 Q_J 矩阵
    Q_J = null_space[0]  # [(m_out * m_in) * m]
    # 重塑 Q_J 矩阵的形状
    Q_J = Q_J.view(to_order(order_out) * to_order(order_in), to_order(J))  # [m_out * m_in, m]
    # 转换为 float 类型并返回
    return Q_J.float()  # [m_out * m_in, m]

# 预计算球谐函数直到最大阶数 max_J
def precompute_sh(r_ij, max_J):
    """
    预计算球谐函数直到最大阶数 max_J

    :param r_ij: 相对位置
    :param max_J: 整个网络中使用的最大阶数
    :return: 字典，每个条目的形状为 [B,N,K,2J+1]
    """
    i_alpha, i_beta = 1, 2
    # 生成球谐函数字典
    Y_Js = {J: spherical_harmonics(J, r_ij[...,i_alpha], r_ij[...,i_beta]) for J in range(max_J + 1)}
    # 清除球谐函数缓存
    clear_spherical_harmonics_cache()
    return Y_Js

# 获取等变权重基础（基础）函数
def get_basis(r_ij, max_degree, differentiable = False):
    """Return equivariant weight basis (basis)

    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal 
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function 
    can be shared as input across all SE(3)-Transformer layers in a model.

    Args:
        r_ij: relative positional vectors
        max_degree: non-negative int for degree of highest feature-type
        differentiable: whether r_ij should receive gradients from basis
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
    """

    # 相对位置编码（向量）
    context = null_context if not differentiable else torch.no_grad

    device, dtype = r_ij.device, r_ij.dtype

    with context():
        # 将笛卡尔坐标系转换为球坐标系
        r_ij = get_spherical_from_cartesian(r_ij)

        # 预计算球谐函数
        Y = precompute_sh(r_ij, 2 * max_degree)

        # 等变基础（字典['d_in><d_out>']）
        basis = {}
        for d_in, d_out in product(range(max_degree+1), range(max_degree+1)):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                # 获取球谐函数变换矩阵 Q_J
                Q_J = basis_transformation_Q_J(J, d_in, d_out)
                Q_J = Q_J.type(dtype).to(device)

                # 从球谐函数创建核
                K_J = torch.matmul(Y[J], Q_J.T)
                K_Js.append(K_J)

            # 重塑以便可以使用点积进行线性组合
            K_Js = torch.stack(K_Js, dim = -1)
            size = (*r_ij.shape[:-1], 1, to_order(d_out), 1, to_order(d_in), to_order(min(d_in,d_out)))
            basis[f'{d_in},{d_out}'] = K_Js.view(*size)

    # 额外的 detach 以确保安全
    if not differentiable:
        for k, v in basis.items():
            basis[k] = v.detach()

    return basis
```