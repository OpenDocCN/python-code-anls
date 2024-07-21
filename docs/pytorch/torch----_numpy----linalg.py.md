# `.\pytorch\torch\_numpy\linalg.py`

```py
# 忽略 mypy 的错误
# 从 __future__ 模块导入 annotations，用于支持类型注解中的类型自引用
import functools  # 导入 functools 模块，用于创建高阶函数和函数工具
import math  # 导入 math 模块，提供数学运算函数
from typing import Sequence  # 导入 Sequence 类型提示，用于表示序列类型

import torch  # 导入 PyTorch 库

from . import _dtypes_impl, _util  # 从当前包中导入 _dtypes_impl 和 _util 模块
from ._normalizations import ArrayLike, KeepDims, normalizer  # 从当前包中导入 ArrayLike 和 normalizer 函数


class LinAlgError(Exception):
    # 定义 LinAlgError 异常类，用于线性代数相关错误
    pass


def _atleast_float_1(a):
    # 函数：确保输入的张量 a 至少为浮点数类型或复数类型
    if not (a.dtype.is_floating_point or a.dtype.is_complex):
        a = a.to(_dtypes_impl.default_dtypes().float_dtype)  # 转换 a 的数据类型为默认的浮点数类型
    return a


def _atleast_float_2(a, b):
    # 函数：确保输入的张量 a 和 b 至少为相同的浮点数类型或复数类型
    dtyp = _dtypes_impl.result_type_impl(a, b)  # 获取 a 和 b 的结果数据类型
    if not (dtyp.is_floating_point or dtyp.is_complex):
        dtyp = _dtypes_impl.default_dtypes().float_dtype  # 如果不是浮点数或复数类型，则设置为默认的浮点数类型

    a = _util.cast_if_needed(a, dtyp)  # 根据需要转换 a 的数据类型
    b = _util.cast_if_needed(b, dtyp)  # 根据需要转换 b 的数据类型
    return a, b


def linalg_errors(func):
    # 装饰器函数：捕获 torch._C._LinAlgError 异常并抛出 LinAlgError 异常
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        try:
            return func(*args, **kwds)
        except torch._C._LinAlgError as e:
            raise LinAlgError(*e.args)  # 捕获 torch._C._LinAlgError 异常并转换为 LinAlgError 异常
                                      # noqa: B904: 忽略 B904 错误（raise 时未指定异常类）

    return wrapped


# ### Matrix and vector products ###


@normalizer
@linalg_errors
def matrix_power(a: ArrayLike, n):
    # 函数：计算矩阵 a 的 n 次幂
    a = _atleast_float_1(a)  # 确保 a 至少为浮点数类型或复数类型
    return torch.linalg.matrix_power(a, n)  # 调用 PyTorch 的矩阵幂计算函数


@normalizer
@linalg_errors
def multi_dot(inputs: Sequence[ArrayLike], *, out=None):
    # 函数：计算多个数组的点积
    return torch.linalg.multi_dot(inputs)  # 调用 PyTorch 的多点积计算函数


# ### Solving equations and inverting matrices ###


@normalizer
@linalg_errors
def solve(a: ArrayLike, b: ArrayLike):
    # 函数：解线性方程组 ax = b
    a, b = _atleast_float_2(a, b)  # 确保 a 和 b 至少为相同的浮点数类型或复数类型
    return torch.linalg.solve(a, b)  # 调用 PyTorch 的线性方程组求解函数


@normalizer
@linalg_errors
def lstsq(a: ArrayLike, b: ArrayLike, rcond=None):
    # 函数：最小二乘法求解线性矩阵方程
    a, b = _atleast_float_2(a, b)  # 确保 a 和 b 至少为相同的浮点数类型或复数类型
    # 根据运行环境选择适当的求解器（driver）
    driver = "gels" if a.is_cuda or b.is_cuda else "gelsd"
    return torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)  # 调用 PyTorch 的最小二乘法求解函数


@normalizer
@linalg_errors
def inv(a: ArrayLike):
    # 函数：计算矩阵 a 的逆矩阵
    a = _atleast_float_1(a)  # 确保 a 至少为浮点数类型或复数类型
    result = torch.linalg.inv(a)  # 调用 PyTorch 的矩阵逆计算函数
    return result


@normalizer
@linalg_errors
def pinv(a: ArrayLike, rcond=1e-15, hermitian=False):
    # 函数：计算矩阵 a 的伪逆矩阵
    a = _atleast_float_1(a)  # 确保 a 至少为浮点数类型或复数类型
    return torch.linalg.pinv(a, rtol=rcond, hermitian=hermitian)  # 调用 PyTorch 的矩阵伪逆计算函数


@normalizer
@linalg_errors
def tensorsolve(a: ArrayLike, b: ArrayLike, axes=None):
    # 函数：解张量方程 ax = b
    a, b = _atleast_float_2(a, b)  # 确保 a 和 b 至少为相同的浮点数类型或复数类型
    return torch.linalg.tensorsolve(a, b, dims=axes)  # 调用 PyTorch 的张量方程求解函数


@normalizer
@linalg_errors
def tensorinv(a: ArrayLike, ind=2):
    # 函数：计算张量 a 的逆张量
    a = _atleast_float_1(a)  # 确保 a 至少为浮点数类型或复数类型
    return torch.linalg.tensorinv(a, ind=ind)  # 调用 PyTorch 的张量逆计算函数


# ### Norms and other numbers ###


@normalizer
@linalg_errors
def det(a: ArrayLike):
    # 函数：计算矩阵 a 的行列式
    a = _atleast_float_1(a)  # 确保 a 至少为浮点数类型或复数类型
    return torch.linalg.det(a)  # 调用 PyTorch 的行列式计算函数


@normalizer
@linalg_errors
def slogdet(a: ArrayLike):
    # 函数：计算矩阵 a 的行列式的符号和自然对数
    a = _atleast_float_1(a)  # 确保 a 至少为浮点数类型或复数类型
    return torch.linalg.slogdet(a)  # 调用 PyTorch 的行列式符号和自然对数计算函数


@normalizer
@linalg_errors
def cond(x: ArrayLike, p=None):
    # 函数：计算矩阵 x 的条件数
    x = _atleast_float_1(x)  # 确保 x 至少为浮点数类型或复数类型

    # 检查是否为空矩阵
    # 参考：https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    # 检查张量 x 是否为空（元素数为零）且形状乘积也为零
    if x.numel() == 0 and math.prod(x.shape[-2:]) == 0:
        # 如果满足条件，抛出线性代数错误，说明条件数在空数组上未定义
        raise LinAlgError("cond is not defined on empty arrays")

    # 计算张量 x 的条件数，使用指定的矩阵范数 p
    result = torch.linalg.cond(x, p=p)

    # 将结果中的 NaN 替换为无穷大（与 NumPy 类似的行为，依赖于输入数组是否包含 NaN）
    # XXX: NumPy 在 https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744 中有相似的处理方式
    return torch.where(torch.isnan(result), float("inf"), result)
@normalizer
@linalg_errors
# 计算矩阵的秩
def matrix_rank(a: ArrayLike, tol=None, hermitian=False):
    # 确保数组至少是浮点型
    a = _atleast_float_1(a)

    # 如果数组的维度小于2，返回非零元素是否存在的整数值
    if a.ndim < 2:
        return int((a != 0).any())

    # 如果未提供公差参数 tol
    if tol is None:
        # 设置公差参数 atol 为 0，rtol 为最大形状的维度乘以数据类型的机器精度
        atol = 0
        rtol = max(a.shape[-2:]) * torch.finfo(a.dtype).eps
    else:
        # 使用提供的公差参数
        atol, rtol = tol, 0
    
    # 调用 torch 的矩阵秩计算函数
    return torch.linalg.matrix_rank(a, atol=atol, rtol=rtol, hermitian=hermitian)


@normalizer
@linalg_errors
# 计算范数
def norm(x: ArrayLike, ord=None, axis=None, keepdims: KeepDims = False):
    # 确保数组至少是浮点型
    x = _atleast_float_1(x)
    
    # 调用 torch 的范数计算函数
    return torch.linalg.norm(x, ord=ord, dim=axis)


# ### Decompositions ###


@normalizer
@linalg_errors
# Cholesky 分解
def cholesky(a: ArrayLike):
    # 确保数组至少是浮点型
    a = _atleast_float_1(a)
    
    # 调用 torch 的 Cholesky 分解函数
    return torch.linalg.cholesky(a)


@normalizer
@linalg_errors
# QR 分解
def qr(a: ArrayLike, mode="reduced"):
    # 确保数组至少是浮点型
    a = _atleast_float_1(a)
    
    # 调用 torch 的 QR 分解函数
    result = torch.linalg.qr(a, mode=mode)
    
    # 如果模式为 "r"，返回结果的 R 矩阵部分
    if mode == "r":
        result = result.R
    
    return result


@normalizer
@linalg_errors
# 奇异值分解
def svd(a: ArrayLike, full_matrices=True, compute_uv=True, hermitian=False):
    # 确保数组至少是浮点型
    a = _atleast_float_1(a)
    
    # 如果不需要计算 U 和 V 矩阵，直接返回奇异值
    if not compute_uv:
        return torch.linalg.svdvals(a)

    # 调用 torch 的奇异值分解函数
    result = torch.linalg.svd(a, full_matrices=full_matrices)
    return result


# ### Eigenvalues and eigenvectors ###


@normalizer
@linalg_errors
# 特征值和特征向量分解
def eig(a: ArrayLike):
    # 确保数组至少是浮点型
    a = _atleast_float_1(a)
    
    # 调用 torch 的特征值和特征向量分解函数
    w, vt = torch.linalg.eig(a)

    # 如果输入数组不是复数，但是得到的特征值是复数且虚部全为零，转换为实部
    if not a.is_complex() and w.is_complex() and (w.imag == 0).all():
        w = w.real
        vt = vt.real
    
    return w, vt


@normalizer
@linalg_errors
# 对称矩阵的特征值和特征向量分解
def eigh(a: ArrayLike, UPLO="L"):
    # 确保数组至少是浮点型
    a = _atleast_float_1(a)
    
    # 调用 torch 的对称矩阵特征值和特征向量分解函数
    return torch.linalg.eigh(a, UPLO=UPLO)


@normalizer
@linalg_errors
# 特征值分解
def eigvals(a: ArrayLike):
    # 确保数组至少是浮点型
    a = _atleast_float_1(a)
    
    # 调用 torch 的特征值计算函数
    result = torch.linalg.eigvals(a)
    
    # 如果输入数组不是复数，但是得到的特征值是复数且虚部全为零，转换为实部
    if not a.is_complex() and result.is_complex() and (result.imag == 0).all():
        result = result.real
    
    return result


@normalizer
@linalg_errors
# 对称矩阵的特征值分解
def eigvalsh(a: ArrayLike, UPLO="L"):
    # 确保数组至少是浮点型
    a = _atleast_float_1(a)
    
    # 调用 torch 的对称矩阵特征值计算函数
    return torch.linalg.eigvalsh(a, UPLO=UPLO)
```