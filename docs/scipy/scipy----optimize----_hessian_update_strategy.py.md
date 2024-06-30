# `D:\src\scipysrc\scipy\scipy\optimize\_hessian_update_strategy.py`

```
# 导入必要的库和模块
"""Hessian update strategies for quasi-Newton optimization methods."""
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.linalg import norm  # 导入 norm 函数，用于计算向量范数
from scipy.linalg import get_blas_funcs, issymmetric  # 导入 get_blas_funcs 函数和 issymmetric 函数，用于获取 BLAS 函数和检查对称性
from warnings import warn  # 导入 warn 函数，用于发出警告


__all__ = ['HessianUpdateStrategy', 'BFGS', 'SR1']  # 指定模块导入时的公开接口


class HessianUpdateStrategy:
    """Interface for implementing Hessian update strategies.

    Many optimization methods make use of Hessian (or inverse Hessian)
    approximations, such as the quasi-Newton methods BFGS, SR1, L-BFGS.
    Some of these  approximations, however, do not actually need to store
    the entire matrix or can compute the internal matrix product with a
    given vector in a very efficiently manner. This class serves as an
    abstract interface between the optimization algorithm and the
    quasi-Newton update strategies, giving freedom of implementation
    to store and update the internal matrix as efficiently as possible.
    Different choices of initialization and update procedure will result
    in different quasi-Newton strategies.

    Four methods should be implemented in derived classes: ``initialize``,
    ``update``, ``dot`` and ``get_matrix``.

    Notes
    -----
    Any instance of a class that implements this interface,
    can be accepted by the method ``minimize`` and used by
    the compatible solvers to approximate the Hessian (or
    inverse Hessian) used by the optimization algorithms.
    """

    def initialize(self, n, approx_type):
        """Initialize internal matrix.

        Allocate internal memory for storing and updating
        the Hessian or its inverse.

        Parameters
        ----------
        n : int
            Problem dimension.
        approx_type : {'hess', 'inv_hess'}
            Selects either the Hessian or the inverse Hessian.
            When set to 'hess' the Hessian will be stored and updated.
            When set to 'inv_hess' its inverse will be used instead.
        """
        raise NotImplementedError("The method ``initialize(n, approx_type)``"
                                  " is not implemented.")  # 抛出未实现的错误提示信息

    def update(self, delta_x, delta_grad):
        """Update internal matrix.

        Update Hessian matrix or its inverse (depending on how 'approx_type'
        is defined) using information about the last evaluated points.

        Parameters
        ----------
        delta_x : ndarray
            The difference between two points the gradient
            function have been evaluated at: ``delta_x = x2 - x1``.
        delta_grad : ndarray
            The difference between the gradients:
            ``delta_grad = grad(x2) - grad(x1)``.
        """
        raise NotImplementedError("The method ``update(delta_x, delta_grad)``"
                                  " is not implemented.")  # 抛出未实现的错误提示信息
    # 计算内部矩阵与给定向量的乘积。
    def dot(self, p):
        """Compute the product of the internal matrix with the given vector.

        Parameters
        ----------
        p : array_like
            1-D array representing a vector.

        Returns
        -------
        Hp : array
            1-D represents the result of multiplying the approximation matrix
            by vector p.
        """
        # 抛出未实现异常，表示这个方法尚未在子类中实现。
        raise NotImplementedError("The method ``dot(p)``"
                                  " is not implemented.")

    # 返回当前内部矩阵。
    def get_matrix(self):
        """Return current internal matrix.

        Returns
        -------
        H : ndarray, shape (n, n)
            Dense matrix containing either the Hessian
            or its inverse (depending on how 'approx_type'
            is defined).
        """
        # 抛出未实现异常，表示这个方法尚未在子类中实现。
        raise NotImplementedError("The method ``get_matrix(p)``"
                                  " is not implemented.")
class FullHessianUpdateStrategy(HessianUpdateStrategy):
    """Hessian update strategy with full dimensional internal representation.
    """
    # 获取 BLAS 函数库中的对称矩阵更新函数（Symmetric rank 1 update）
    _syr = get_blas_funcs('syr', dtype='d')
    # 获取 BLAS 函数库中的对称矩阵更新函数（Symmetric rank 2 update）
    _syr2 = get_blas_funcs('syr2', dtype='d')
    # 获取 BLAS 函数库中的对称矩阵-向量乘法函数
    _symv = get_blas_funcs('symv', dtype='d')

    def __init__(self, init_scale='auto'):
        self.init_scale = init_scale
        # 在调用 initialize 方法前，类的其他属性都设置为 None
        self.first_iteration = None
        self.approx_type = None
        self.B = None
        self.H = None

    def initialize(self, n, approx_type):
        """Initialize internal matrix.

        Allocate internal memory for storing and updating
        the Hessian or its inverse.

        Parameters
        ----------
        n : int
            Problem dimension.
        approx_type : {'hess', 'inv_hess'}
            Selects either the Hessian or the inverse Hessian.
            When set to 'hess' the Hessian will be stored and updated.
            When set to 'inv_hess' its inverse will be used instead.
        """
        self.first_iteration = True
        self.n = n
        self.approx_type = approx_type
        if approx_type not in ('hess', 'inv_hess'):
            raise ValueError("`approx_type` must be 'hess' or 'inv_hess'.")
        # 创建一个单位矩阵
        if self.approx_type == 'hess':
            self.B = np.eye(n, dtype=float)
        else:
            self.H = np.eye(n, dtype=float)

    def _auto_scale(self, delta_x, delta_grad):
        # 根据第一次迭代的启发式方法缩放矩阵
        # 参考 Nocedal 和 Wright 的 "Numerical Optimization"
        # 第143页公式 (6.20) 描述
        s_norm2 = np.dot(delta_x, delta_x)
        y_norm2 = np.dot(delta_grad, delta_grad)
        ys = np.abs(np.dot(delta_grad, delta_x))
        if ys == 0.0 or y_norm2 == 0 or s_norm2 == 0:
            return 1
        if self.approx_type == 'hess':
            return y_norm2 / ys
        else:
            return ys / y_norm2

    def _update_implementation(self, delta_x, delta_grad):
        # 抛出未实现错误，提示方法 `_update_implementation` 没有实现
        raise NotImplementedError("The method ``_update_implementation``"
                                  " is not implemented.")

    def dot(self, p):
        """Compute the product of the internal matrix with the given vector.

        Parameters
        ----------
        p : array_like
            1-D array representing a vector.

        Returns
        -------
        Hp : array
            1-D represents the result of multiplying the approximation matrix
            by vector p.
        """
        if self.approx_type == 'hess':
            return self._symv(1, self.B, p)
        else:
            return self._symv(1, self.H, p)
    # 返回当前内部矩阵，可能是黑塞矩阵或其逆矩阵（取决于`approx_type`的定义）
    def get_matrix(self):
        """Return the current internal matrix.

        Returns
        -------
        M : ndarray, shape (n, n)
            Dense matrix containing either the Hessian or its inverse
            (depending on how `approx_type` was defined).
        """
        # 如果`approx_type`为'hess'，则复制对象`self.B`到`M`
        if self.approx_type == 'hess':
            M = np.copy(self.B)
        else:
            # 否则，复制对象`self.H`到`M`
            M = np.copy(self.H)
        
        # 获取`M`的下三角矩阵索引（不包括对角线）
        li = np.tril_indices_from(M, k=-1)
        # 将下三角区域的值与其对应的上三角区域值相同，使`M`成为对称矩阵
        M[li] = M.T[li]
        
        # 返回结果矩阵`M`
        return M
# 定义 BFGS 类，继承自 FullHessianUpdateStrategy 类
class BFGS(FullHessianUpdateStrategy):
    """Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.

    Parameters
    ----------
    exception_strategy : {'skip_update', 'damp_update'}, optional
        定义在曲率条件违反时如何处理的策略。
        设置为 'skip_update' 时跳过更新；或者设置为 'damp_update' 时，在实际 BFGS 结果和未修改矩阵之间插值。
        这两种异常策略在文献 [1]_, p.536-537 中有详细说明。
    min_curvature : float
        此数字乘以归一化因子，定义了允许不受异常策略影响的最小曲率 ``dot(delta_grad, delta_x)``。
        默认情况下，当 ``exception_strategy = 'skip_update'`` 时为 1e-8，当 ``exception_strategy = 'damp_update'`` 时为 0.2。
    init_scale : {float, np.array, 'auto'}
        此参数用于初始化 Hessian 或其逆。当给定 float 时，相关数组初始化为 ``np.eye(n) * init_scale``，
        其中 ``n`` 是问题的维度。或者，如果给定了精确的 ``(n, n)`` 形状的对称数组，则使用此数组。
        否则会生成错误。设置为 'auto' 以使用自动启发式方法选择初始尺度。
        启发式方法在文献 [1]_, p.143 中有描述。默认为 'auto'。

    Notes
    -----
    更新基于文献 [1]_, p.140 中的描述。

    References
    ----------
    .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """

    # 初始化方法，接受 exception_strategy、min_curvature 和 init_scale 参数
    def __init__(self, exception_strategy='skip_update', min_curvature=None,
                 init_scale='auto'):
        # 根据 exception_strategy 的值设置 min_curvature 的默认值
        if exception_strategy == 'skip_update':
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 1e-8
        elif exception_strategy == 'damp_update':
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 0.2
        else:
            # 如果 exception_strategy 不是 'skip_update' 或 'damp_update'，则抛出 ValueError
            raise ValueError("`exception_strategy` must be 'skip_update' "
                             "or 'damp_update'.")
        
        # 调用父类 FullHessianUpdateStrategy 的初始化方法，传入 init_scale 参数
        super().__init__(init_scale)
        
        # 将 exception_strategy 设置为类的属性
        self.exception_strategy = exception_strategy
    def _update_inverse_hessian(self, ys, Hy, yHy, s):
        """Update the inverse Hessian matrix.

        BFGS update using the formula:

            ``H <- H + ((H*y).T*y + s.T*y)/(s.T*y)^2 * (s*s.T)
                     - 1/(s.T*y) * ((H*y)*s.T + s*(H*y).T)``

        where ``s = delta_x`` and ``y = delta_grad``. This formula is
        equivalent to (6.17) in [1]_ written in a more efficient way
        for implementation.

        References
        ----------
        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        """
        # 更新逆黑塞矩阵，使用BFGS公式
        self.H = self._syr2(-1.0 / ys, s, Hy, a=self.H)
        # 继续更新逆黑塞矩阵，使用BFGS公式的另一部分
        self.H = self._syr((ys + yHy) / ys ** 2, s, a=self.H)

    def _update_hessian(self, ys, Bs, sBs, y):
        """Update the Hessian matrix.

        BFGS update using the formula:

            ``B <- B - (B*s)*(B*s).T/s.T*(B*s) + y*y^T/s.T*y``

        where ``s`` is short for ``delta_x`` and ``y`` is short
        for ``delta_grad``. Formula (6.19) in [1]_.

        References
        ----------
        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        """
        # 更新黑塞矩阵，使用BFGS公式
        self.B = self._syr(1.0 / ys, y, a=self.B)
        # 继续更新黑塞矩阵，使用BFGS公式的另一部分
        self.B = self._syr(-1.0 / sBs, Bs, a=self.B)
    # 辅助变量 w 和 z 根据近似类型选择不同的赋值顺序
    if self.approx_type == 'hess':
        w = delta_x
        z = delta_grad
    else:
        w = delta_grad
        z = delta_x
    
    # 计算向量 w 和 z 的内积
    wz = np.dot(w, z)
    
    # 计算向量 w 在当前矩阵 B（或 H）下的作用结果 Mw
    Mw = self.dot(w)
    
    # 计算向量 w 在当前矩阵 B（或 H）下的二次形式 wMw
    wMw = Mw.dot(w)
    
    # 确保 wMw 大于 0，若不是则重新初始化矩阵以避免非正定情况
    # 在精确算术中，wMw 一般大于 0，但由于舍入误差可能导致非正定矩阵出现
    if wMw <= 0.0:
        # 根据自动缩放因子重新初始化矩阵 B 或 H
        scale = self._auto_scale(delta_x, delta_grad)
        if self.approx_type == 'hess':
            self.B = scale * np.eye(self.n, dtype=float)
        else:
            self.H = scale * np.eye(self.n, dtype=float)
        
        # 更新重新初始化后的 Mw 和 wMw
        Mw = self.dot(w)
        wMw = Mw.dot(w)
    
    # 检查是否违反曲率条件
    if wz <= self.min_curvature * wMw:
        # 如果异常处理策略设置为跳过更新，则直接返回
        if self.exception_strategy == 'skip_update':
            return
        
        # 如果异常处理策略设置为阻尼更新，则进行插值操作
        elif self.exception_strategy == 'damp_update':
            update_factor = (1 - self.min_curvature) / (1 - wz / wMw)
            z = update_factor * z + (1 - update_factor) * Mw
            wz = np.dot(w, z)
    
    # 根据近似类型更新矩阵 B 或 H
    if self.approx_type == 'hess':
        self._update_hessian(wz, Mw, wMw, z)
    else:
        self._update_inverse_hessian(wz, Mw, wMw, z)
class SR1(FullHessianUpdateStrategy):
    """Symmetric-rank-1 Hessian update strategy.

    Parameters
    ----------
    min_denominator : float
        This number, scaled by a normalization factor,
        defines the minimum denominator magnitude allowed
        in the update. When the condition is violated we skip
        the update. By default uses ``1e-8``.
    init_scale : {float, np.array, 'auto'}, optional
        This parameter can be used to initialize the Hessian or its
        inverse. When a float is given, the relevant array is initialized
        to ``np.eye(n) * init_scale``, where ``n`` is the problem dimension.
        Alternatively, if a precisely ``(n, n)`` shaped, symmetric array is given,
        this array will be used. Otherwise an error is generated.
        Set it to 'auto' in order to use an automatic heuristic for choosing
        the initial scale. The heuristic is described in [1]_, p.143.
        The default is 'auto'.

    Notes
    -----
    The update is based on the description in [1]_, p.144-146.

    References
    ----------
    .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """

    def __init__(self, min_denominator=1e-8, init_scale='auto'):
        # 初始化 Symmetric-rank-1 更新策略，设置最小分母值和初始化规模
        self.min_denominator = min_denominator
        # 调用父类的初始化方法
        super().__init__(init_scale)

    def _update_implementation(self, delta_x, delta_grad):
        # Auxiliary variables w and z
        # 根据近似类型选择 w 和 z
        if self.approx_type == 'hess':
            w = delta_x
            z = delta_grad
        else:
            w = delta_grad
            z = delta_x
        # 做一些常见的操作
        Mw = self.dot(w)
        z_minus_Mw = z - Mw
        denominator = np.dot(w, z_minus_Mw)
        # 如果分母太小，跳过更新
        if np.abs(denominator) <= self.min_denominator * norm(w) * norm(z_minus_Mw):
            return
        # 更新矩阵
        if self.approx_type == 'hess':
            self.B = self._syr(1/denominator, z_minus_Mw, a=self.B)
        else:
            self.H = self._syr(1/denominator, z_minus_Mw, a=self.H)
```