# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\bdf.py`

```
# 导入必要的库
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, EPS, num_jac, validate_first_step,
                     warn_extraneous)
from .base import OdeSolver, DenseOutput

# 定义常量
MAX_ORDER = 5  # 最大阶数
NEWTON_MAXITER = 4  # 牛顿迭代的最大次数
MIN_FACTOR = 0.2  # 最小因子
MAX_FACTOR = 10  # 最大因子


def compute_R(order, factor):
    """计算用于改变差分数组的矩阵."""
    # 创建索引向量
    I = np.arange(1, order + 1)[:, None]
    J = np.arange(1, order + 1)
    # 初始化矩阵
    M = np.zeros((order + 1, order + 1))
    # 计算差分矩阵
    M[1:, 1:] = (I - 1 - factor * J) / I
    M[0] = 1
    return np.cumprod(M, axis=0)


def change_D(D, order, factor):
    """在改变步长时就地改变差分数组."""
    # 计算用于改变差分数组的矩阵R
    R = compute_R(order, factor)
    U = compute_R(order, 1)
    # 计算R*U
    RU = R.dot(U)
    # 更新差分数组D
    D[:order + 1] = np.dot(RU.T, D[:order + 1])


def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    """解决由BDF方法产生的代数系统."""
    d = 0
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        # 计算当前时间和状态的右手边的函数值
        f = fun(t_new, y)
        # 检查是否所有元素都是有限的
        if not np.all(np.isfinite(f)):
            break

        # 解线性方程LU*dy = c*f - psi - d
        dy = solve_lu(LU, c * f - psi - d)
        # 计算归一化的更新步长
        dy_norm = norm(dy / scale)

        # 计算收敛率
        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old

        # 检查是否满足收敛条件
        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            break

        # 更新状态变量y和d
        y += dy
        d += dy

        # 检查是否已经收敛
        if (dy_norm == 0 or
                rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break

        # 更新旧的归一化步长
        dy_norm_old = dy_norm

    return converged, k + 1, y, d


class BDF(OdeSolver):
    """基于反向差分公式的隐式方法.

    这是一个可变阶方法，阶数自动从1到5变化。BDF算法的一般框架在文献[1]中描述。
    本类实现了在文献[2]中解释的准恒定步长。用于恒定步长BDF的误差估计策略在文献[3]中推导。
    还实现了使用修改的公式（NDF）[2]的精度增强。

    可以在复杂域中应用.

    Parameters
    ----------
    fun : callable
        系统右手边：状态``y``在时间``t``的时间导数. 调用签名为``fun(t, y)``，其中``t``是一个标量，``y``是一个形状为``(n,)``的数组. ``fun``必须返回与``y``形状相同的数组. 参见`vectorized`以获取更多信息.
    t0 : float
        初始时间.
    y0 : array_like, shape (n,)
        初始状态.
    ```
    # t_bound 是浮点数，表示边界时间，积分不会超过这个时间。它也决定了积分的方向。
    t_bound : float
    
    # first_step 是初始步长的大小，可以是浮点数或者 None。默认为 None，表示由算法自动选择初始步长。
    first_step : float or None, optional
    
    # max_step 是允许的最大步长大小。默认为 np.inf，即步长大小没有限制，完全由求解器确定。
    max_step : float, optional
    
    # rtol 和 atol 分别是相对误差和绝对误差的容许范围。求解器保持局部误差估计小于 atol + rtol * abs(y)。
    # 这里，rtol 控制相对精度（正确数字的数量），而 atol 控制绝对精度（正确小数位的数量）。
    # 为了达到所需的 rtol，需要将 atol 设置为小于从 rtol * abs(y) 得到的最小值，以便 rtol 主导允许的误差。
    # 如果 atol 大于 rtol * abs(y)，则不能保证正确数字的数量。
    # 相反，为了达到所需的 atol，需要设置 rtol，使得 rtol * abs(y) 总是小于 atol。
    # 如果 y 的分量具有不同的尺度，可能有利于通过传递形状为 (n,) 的数组来为不同的分量设置不同的 atol 值。
    # 默认值分别为 1e-3（rtol）和 1e-6（atol）。
    rtol, atol : float and array_like, optional
    
    # jac 是对于系统右手边的雅可比矩阵，对于该方法是必需的。
    # 雅可比矩阵的形状为 (n, n)，其元素 (i, j) 等于 d f_i / d y_j。
    # 定义雅可比矩阵有三种方式：
    #   * 如果是 array_like 或者 sparse_matrix，则假定雅可比矩阵是常数。
    #   * 如果是 callable，则假定雅可比矩阵依赖于 t 和 y；将会按需调用为 jac(t, y)。
    #     对于 'Radau' 和 'BDF' 方法，返回值可能是稀疏矩阵。
    #   * 如果是 None（默认），雅可比矩阵将会通过有限差分来近似计算。
    # 通常建议提供雅可比矩阵，而不是依赖有限差分近似。
    jac : {None, array_like, sparse_matrix, callable}, optional
    
    # jac_sparsity 定义了雅可比矩阵的稀疏结构，用于有限差分近似。
    # 其形状必须为 (n, n)。如果 jac 不是 None，则此参数被忽略。
    # 如果雅可比矩阵每行只有少量非零元素，提供稀疏结构将大大加快计算速度。
    # 零条目意味着雅可比矩阵对应元素始终为零。如果是 None（默认），假定雅可比矩阵是密集的。
    jac_sparsity : {None, array_like, sparse matrix}, optional
    # 是否可以对 `fun` 进行矢量化调用的标志，默认为 False。
    # 
    # 如果 `vectorized` 是 False，则 `fun` 将始终以 `y` 的形状 `(n,)` 调用，其中 `n = len(y0)`。
    # 
    # 如果 `vectorized` 是 True，则 `fun` 可能以 `y` 的形状 `(n, k)` 调用，其中 `k` 是一个整数。
    # 在这种情况下，`fun` 必须表现出 `fun(t, y)[:, i] == fun(t, y[:, i])` 的行为（即返回数组的每一列都是对应 `y` 列的状态的时间导数）。
    # 
    # 设置 `vectorized=True` 允许通过该方法更快地进行有限差分法求解雅可比矩阵，但在某些情况下（如 `len(y0)` 较小时）可能导致整体执行速度较慢。
    vectorized: bool, optional

    # 方程组的数量。
    n: int

    # 求解器当前的状态：'running'（运行中）、'finished'（完成）、'failed'（失败）。
    status: string

    # 边界时间。
    t_bound: float

    # 积分方向：+1 或 -1。
    direction: float

    # 当前时间。
    t: float

    # 当前状态。
    y: ndarray

    # 上一个时间步的时间。如果尚未进行过步骤，则为 None。
    t_old: float

    # 最后一次成功步骤的步长。如果尚未进行过步骤，则为 None。
    step_size: float

    # 右手边函数的评估次数。
    nfev: int

    # 雅可比矩阵的评估次数。
    njev: int

    # LU 分解的次数。
    nlu: int
    # 初始化方法，接受多个参数，并初始化对象状态
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, **extraneous):
        # 提示多余的参数（extraneous）可能会引发警告
        warn_extraneous(extraneous)
        # 调用父类的初始化方法，初始化基础属性和状态
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=True)
        # 验证和设置最大步长
        self.max_step = validate_max_step(max_step)
        # 验证和设置相对误差和绝对误差容差
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        # 计算初始函数值
        f = self.fun(self.t, self.y)
        # 如果没有指定首步长，选择初始步长
        if first_step is None:
            self.h_abs = select_initial_step(self.fun, self.t, self.y, 
                                             t_bound, max_step, f,
                                             self.direction, 1,
                                             self.rtol, self.atol)
        else:
            # 验证给定的首步长
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        # 初始化旧步长和误差范数
        self.h_abs_old = None
        self.error_norm_old = None

        # 设置牛顿法的误差容差
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))

        # 初始化雅可比矩阵相关属性
        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)
        # 如果雅可比矩阵是稀疏矩阵
        if issparse(self.J):
            # 定义稀疏矩阵的LU分解函数
            def lu(A):
                self.nlu += 1
                return splu(A)

            # 定义稀疏矩阵的LU分解求解函数
            def solve_lu(LU, b):
                return LU.solve(b)

            # 创建单位矩阵
            I = eye(self.n, format='csc', dtype=self.y.dtype)
        else:
            # 定义一般矩阵的LU分解函数
            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            # 定义一般矩阵的LU分解求解函数
            def solve_lu(LU, b):
                return lu_solve(LU, b, overwrite_b=True)

            # 创建单位矩阵
            I = np.identity(self.n, dtype=self.y.dtype)

        # 将LU分解函数、LU求解函数和单位矩阵保存到对象属性中
        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        # 初始化一些数学常数和数组
        kappa = np.array([0, -0.1850, -1/9, -0.0823, -0.0415, 0])
        self.gamma = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
        self.alpha = (1 - kappa) * self.gamma
        self.error_const = kappa * self.gamma + 1 / np.arange(1, MAX_ORDER + 2)

        # 初始化状态向量数组
        D = np.empty((MAX_ORDER + 3, self.n), dtype=self.y.dtype)
        D[0] = self.y
        D[1] = f * self.h_abs * self.direction
        self.D = D

        # 初始化阶数、相等步长计数和LU分解对象
        self.order = 1
        self.n_equal_steps = 0
        self.LU = None
    # 定义一个方法，用于验证雅可比矩阵 `jac` 的有效性，根据给定的稀疏度 `sparsity`
    def _validate_jac(self, jac, sparsity):
        # 将当前对象的时间 `t` 和状态 `y` 赋值给局部变量 t0 和 y0
        t0 = self.t
        y0 = self.y

        # 如果 `jac` 为 None
        if jac is None:
            # 如果给定了 `sparsity`，则需要进一步处理
            if sparsity is not None:
                # 如果 `sparsity` 是稀疏矩阵，则将其转换为 CSC 格式
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                # 对稀疏矩阵进行列分组
                groups = group_columns(sparsity)
                # 将 `sparsity` 更新为包含稀疏矩阵和列分组信息的元组
                sparsity = (sparsity, groups)

            # 定义一个内部函数 `jac_wrapped`，用于计算雅可比矩阵
            def jac_wrapped(t, y):
                # 增加雅可比计算次数统计
                self.njev += 1
                # 计算单个函数 `fun_single` 在给定时间 `t` 和状态 `y` 下的值
                f = self.fun_single(t, y)
                # 使用数值雅可比计算函数 `num_jac` 计算雅可比矩阵 `J`
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor,
                                             sparsity)
                return J
            # 使用当前时间 `t0` 和状态 `y0` 调用 `jac_wrapped` 函数，得到雅可比矩阵 `J`
            J = jac_wrapped(t0, y0)
        
        # 如果 `jac` 是可调用对象（函数）
        elif callable(jac):
            # 调用 `jac` 函数计算雅可比矩阵 `J`
            J = jac(t0, y0)
            # 增加雅可比计算次数统计
            self.njev += 1
            # 如果 `J` 是稀疏矩阵，则转换为 CSC 格式
            if issparse(J):
                J = csc_matrix(J, dtype=y0.dtype)

                # 定义一个内部函数 `jac_wrapped`，用于返回稀疏雅可比矩阵 `J`
                def jac_wrapped(t, y):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=y0.dtype)
            else:
                # 否则，将 `J` 转换为 `y0` 类型的 numpy 数组
                J = np.asarray(J, dtype=y0.dtype)

                # 定义一个内部函数 `jac_wrapped`，用于返回密集雅可比矩阵 `J`
                def jac_wrapped(t, y):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=y0.dtype)

            # 如果 `J` 的形状不是 (self.n, self.n)，则抛出 ValueError 异常
            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
        else:
            # 否则，如果 `jac` 是稀疏矩阵，则转换为 CSC 格式
            if issparse(jac):
                J = csc_matrix(jac, dtype=y0.dtype)
            else:
                # 否则，将 `jac` 转换为 `y0` 类型的 numpy 数组
                J = np.asarray(jac, dtype=y0.dtype)

            # 如果 `J` 的形状不是 (self.n, self.n)，则抛出 ValueError 异常
            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
            # 将 `jac_wrapped` 设置为 None
            jac_wrapped = None

        # 返回 `jac_wrapped` 函数和计算得到的雅可比矩阵 `J`
        return jac_wrapped, J

    # 定义一个方法 `_dense_output_impl`，用于返回 BdfDenseOutput 对象
    def _dense_output_impl(self):
        # 返回 BdfDenseOutput 对象，使用当前对象的时间历史 `self.t_old`、当前时间 `self.t`、
        # 绝对步长 `self.h_abs` 乘以方向 `self.direction`、阶数 `self.order`，以及前 `self.order + 1` 个项的复制 `self.D[:self.order + 1].copy()`
        return BdfDenseOutput(self.t_old, self.t, self.h_abs * self.direction,
                              self.order, self.D[:self.order + 1].copy())
# 创建一个名为 BdfDenseOutput 的类，它继承自 DenseOutput 类
class BdfDenseOutput(DenseOutput):
    # 定义初始化方法，接受 t_old, t, h, order, D 作为参数
    def __init__(self, t_old, t, h, order, D):
        # 调用父类 DenseOutput 的初始化方法，传入 t_old 和 t 参数
        super().__init__(t_old, t)
        # 将 order 参数赋值给实例变量 order
        self.order = order
        # 计算 t_shift，使用当前时间 t 减去 h 乘以从 0 到 order-1 的数组
        self.t_shift = self.t - h * np.arange(self.order)
        # 计算 denom，使用 h 乘以 1 加上从 0 到 order-1 的数组
        self.denom = h * (1 + np.arange(self.order))
        # 将参数 D 赋值给实例变量 D
        self.D = D

    # 定义私有方法 _call_impl，接受 t 参数
    def _call_impl(self, t):
        # 如果 t 的维度为 0
        if t.ndim == 0:
            # 计算 x，使用 t 减去 t_shift 后除以 denom
            x = (t - self.t_shift) / self.denom
            # 计算累积乘积 p，对 x 中的元素进行累积乘法
            p = np.cumprod(x)
        else:
            # 计算 x，使用 t 减去 t_shift 的每一列，然后除以 denom 的每一列
            x = (t - self.t_shift[:, None]) / self.denom[:, None]
            # 计算累积乘积 p，对 x 的每一列进行累积乘法
            p = np.cumprod(x, axis=0)

        # 计算输出 y，使用 D 的第 1 列开始的转置乘以 p
        y = np.dot(self.D[1:].T, p)
        
        # 如果 y 的维度为 1
        if y.ndim == 1:
            # 将 D 的第 0 列加到 y 上
            y += self.D[0]
        else:
            # 将 D 的第 0 列的每一列加到 y 的每一列上
            y += self.D[0, :, None]

        # 返回计算得到的 y
        return y
```