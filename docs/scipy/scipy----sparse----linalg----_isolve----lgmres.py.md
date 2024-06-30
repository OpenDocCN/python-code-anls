# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\lgmres.py`

```
# 导入必要的库和模块
import numpy as np  # 导入NumPy库
from numpy.linalg import LinAlgError  # 导入NumPy的线性代数异常处理
from scipy.linalg import get_blas_funcs  # 从SciPy的线性代数模块导入获取BLAS函数的方法
from .iterative import _get_atol_rtol  # 从当前包的iterative模块导入_get_atol_rtol函数
from .utils import make_system  # 从当前包的utils模块导入make_system函数

# 导入C扩展模块_gcrotmk中的_fgmems函数
from ._gcrotmk import _fgmres

# 定义可以导出的函数或类列表
__all__ = ['lgmres']


def lgmres(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=1000, M=None, callback=None,
           inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True,
           prepend_outer_v=False):
    """
    使用LGMRES算法解决矩阵方程。

    LGMRES算法[1][2]旨在避免重启GMRES中的一些收敛问题，并通常在较少的迭代中收敛。

    Parameters
    ----------
    A : {稀疏矩阵, ndarray, LinearOperator}
        线性系统的实数或复数N×N矩阵。
        或者，``A``可以是一个线性操作符，它可以使用例如
        ``scipy.sparse.linalg.LinearOperator``来产生``Ax``。
    b : ndarray
        线性系统的右手边。形状为(N,)或(N,1)。
    x0 : ndarray
        解的初始猜测。
    rtol, atol : float, optional
        收敛测试的参数。对于收敛，
        应满足``norm(b - A @ x) <= max(rtol*norm(b), atol)``。
        默认值为``rtol=1e-5``，``atol``的默认值为``0.0``。
    maxiter : int, optional
        最大迭代次数。即使未达到指定的容差，迭代也将在maxiter步之后停止。
    M : {稀疏矩阵, ndarray, LinearOperator}, optional
        A的预条件器。预条件器应近似于A的逆。
        有效的预条件显著改善收敛速度，这意味着需要更少的迭代来达到给定的误差容限。
    callback : function, optional
        用户提供的函数，在每次迭代后调用。调用形式为callback(xk)，其中xk是当前解向量。
    inner_m : int, optional
        每个外部迭代中的内部GMRES迭代次数。
    outer_k : int, optional
        在内部GMRES迭代之间传递的向量数量。
        根据[1]_，好的值在1到3之间。
        但是，请注意，如果您希望使用额外的向量来加速解决多个类似问题，较大的值可能更有利。

    """
    A,M,x,b,postprocess = make_system(A,M,x0,b)


# 调用 make_system 函数构建线性方程组的参数和处理器
# A: 系数矩阵
# M: 预处理矩阵或 None
# x: 初始解向量
# b: 右端向量
# postprocess: 后处理器或 None
A,M,x,b,postprocess = make_system(A,M,x0,b)



    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")


# 检查右端向量 b 中是否包含有限数值，若不是则抛出 ValueError 异常
if not np.isfinite(b).all():
    raise ValueError("RHS must contain only finite numbers")



    matvec = A.matvec
    psolve = M.matvec


# 获取系数矩阵 A 和预处理矩阵 M 的向量乘法函数
matvec = A.matvec  # 系数矩阵 A 的向量乘法函数
psolve = M.matvec  # 预处理矩阵 M 的向量乘法函数



    if outer_v is None:
        outer_v = []


# 如果 outer_v 参数为 None，则初始化为一个空列表
if outer_v is None:
    outer_v = []



    axpy, dot, scal = None, None, None
    nrm2 = get_blas_funcs('nrm2', [b])


# 初始化线性代数库函数的变量
axpy, dot, scal = None, None, None  # BLAS 中的函数，用于向量运算
nrm2 = get_blas_funcs('nrm2', [b])  # 获取 BLAS 中的 nrm2 函数，用于计算向量的二范数



"""
注释结束
"""```
    # 计算向量 b 的二范数
    b_norm = nrm2(b)

    # 调用此函数以获取适当的 atol（绝对误差限）和 rtol（相对误差限），并在必要时引发错误
    atol, rtol = _get_atol_rtol('lgmres', b_norm, atol, rtol)

    # 如果向量 b 的二范数为 0，则直接返回 b 作为解向量 x，并将后处理应用到 x 上，返回解和迭代次数 0
    if b_norm == 0:
        x = b
        return (postprocess(x), 0)

    # 设置松弛因子的最大允许值
    ptol_max_factor = 1.0
    # 对外层循环进行迭代，最多执行maxiter次
    for k_outer in range(maxiter):
        # 计算当前解向量x的向量乘法结果，并与右侧向量b做差，得到残差r_outer
        r_outer = matvec(x) - b

        # -- 回调函数
        # 如果提供了回调函数callback，则调用它，参数是当前解向量x
        if callback is not None:
            callback(x)

        # -- 确定输入类型的例行程序
        if axpy is None:
            # 如果axpy函数未定义：
            # 如果r_outer是复数类型且x不是复数类型，则将x转换为r_outer的数据类型
            if np.iscomplexobj(r_outer) and not np.iscomplexobj(x):
                x = x.astype(r_outer.dtype)
            # 获取BLAS函数（axpy、dot、scal、nrm2），以便后续计算使用
            axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'],
                                                   (x, r_outer))

        # -- 检查停止条件
        # 计算残差r_outer的二范数
        r_norm = nrm2(r_outer)
        # 如果r_norm小于等于atol和rtol乘以右侧向量b的二范数的最大值，则结束迭代
        if r_norm <= max(atol, rtol * b_norm):
            break

        # -- 内部LGMRES迭代
        # 使用预处理函数psolve处理r_outer得到向量v0
        v0 = -psolve(r_outer)
        # 计算v0的二范数作为inner_res_0
        inner_res_0 = nrm2(v0)

        # 如果inner_res_0为0，表示预处理器返回了一个零向量，抛出运行时错误
        if inner_res_0 == 0:
            rnorm = nrm2(r_outer)
            raise RuntimeError("Preconditioner returned a zero vector; "
                               "|v| ~ %.1g, |M v| = 0" % rnorm)

        # 对v0进行缩放，使其二范数为1
        v0 = scal(1.0/inner_res_0, v0)

        # 计算允许的内部迭代容差ptol，根据最大容差因子和atol、rtol*b_norm/r_norm的最大值确定
        ptol = min(ptol_max_factor, max(atol, rtol*b_norm)/r_norm)

        try:
            # 调用_fgmlres函数进行FGMRES迭代，返回解y及其他中间结果
            Q, R, B, vs, zs, y, pres = _fgmres(matvec,
                                               v0,
                                               inner_m,
                                               lpsolve=psolve,
                                               atol=ptol,
                                               outer_v=outer_v,
                                               prepend_outer_v=prepend_outer_v)
            # 将y乘以inner_res_0以调整解的尺度
            y *= inner_res_0
            # 如果y中有非有限值，抛出线性代数错误
            if not np.isfinite(y).all():
                # 计算过程中发生溢出等问题，无法恢复，只能退出
                raise LinAlgError()
        except LinAlgError:
            # 线性代数错误，如浮点溢出等，报告失败并返回结果
            return postprocess(x), k_outer + 1

        # 内部循环容差控制
        # 如果pres大于ptol，则增加ptol_max_factor；否则减小ptol_max_factor
        if pres > ptol:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        else:
            ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)

        # -- GMRES终止：评估解
        # 计算解增量dx
        dx = zs[0]*y[0]
        # 使用axpy函数计算解增量dx的线性组合
        for w, yc in zip(zs[1:], y[1:]):
            dx = axpy(w, dx, dx.shape[0], yc)  # dx += w*yc

        # -- 存储LGMRES的增强向量
        # 计算dx的二范数nx
        nx = nrm2(dx)
        # 如果nx大于0，则存储增强向量dx/nx及相关信息
        if nx > 0:
            if store_outer_Av:
                q = Q.dot(R.dot(y))
                ax = vs[0]*q[0]
                for v, qc in zip(vs[1:], q[1:]):
                    ax = axpy(v, ax, ax.shape[0], qc)
                outer_v.append((dx/nx, ax/nx))
            else:
                outer_v.append((dx/nx, None))

        # -- 保留有限数量的增强向量
        # 当outer_v的长度超过outer_k时，删除最早的增强向量
        while len(outer_v) > outer_k:
            del outer_v[0]

        # -- 应用步长
        # 更新解向量x
        x += dx
    else:
        # 如果未收敛，则返回后处理后的解向量x及最大迭代次数maxiter
        return postprocess(x), maxiter

    # 成功收敛后，返回后处理后的解向量x及0
    return postprocess(x), 0
```