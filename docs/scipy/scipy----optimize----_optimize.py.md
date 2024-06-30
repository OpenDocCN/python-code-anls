# `D:\src\scipysrc\scipy\scipy\optimize\_optimize.py`

```
# 设置文档格式为 restructuredtext，使用英文
__docformat__ = "restructuredtext en"
# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

# A collection of optimization algorithms. Version 0.5
# CHANGES
#  Added fminbound (July 2001)
#  Added brute (Aug. 2002)
#  Finished line search satisfying strong Wolfe conditions (Mar. 2004)
#  Updated strong Wolfe conditions line search to use
#  cubic-interpolation (Mar. 2004)


# Minimization routines

# 将以下名称添加到模块的公共接口中
__all__ = ['fmin', 'fmin_powell', 'fmin_bfgs', 'fmin_ncg', 'fmin_cg',
           'fminbound', 'brent', 'golden', 'bracket', 'rosen', 'rosen_der',
           'rosen_hess', 'rosen_hess_prod', 'brute', 'approx_fprime',
           'line_search', 'check_grad', 'OptimizeResult', 'show_options',
           'OptimizeWarning']

# 设置文档格式为 restructuredtext，使用英文
__docformat__ = "restructuredtext en"

# 导入标准库和第三方库
import math
import warnings
import sys
import inspect
# 导入 numpy 中的指定函数和模块
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
                   asarray, sqrt)
import numpy as np
# 导入 scipy 中的线性代数和稀疏线性代数函数
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
# 导入自定义模块中的函数
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
                          line_search_wolfe2 as line_search,
                          LineSearchWarning)
# 导入数值微分模块
from ._numdiff import approx_derivative
# 导入辅助工具函数和类
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import (MapWrapper, check_random_state, _RichResult,
                              _call_callback_maybe_halt)
# 导入优化器的不同类型和方法
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS


# 优化器的标准状态消息
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}


class MemoizeJac:
    """Decorator that caches the return values of a function returning ``(fun, grad)``
    each time it is called."""
    
    def __init__(self, fun):
        # 初始化装饰器，存储被装饰函数和相关变量
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None

    def _compute_if_needed(self, x, *args):
        # 如果需要，计算函数和梯度值，并进行缓存
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            fg = self.fun(x, *args)
            self.jac = fg[1]
            self._value = fg[0]
    # 定义对象的调用方法，用于计算函数值并返回
    def __call__(self, x, *args):
        """ returns the function value """
        # 如果需要的话，计算函数值
        self._compute_if_needed(x, *args)
        # 返回计算好的函数值
        return self._value

    # 定义对象的求导方法，用于计算函数的雅可比矩阵并返回
    def derivative(self, x, *args):
        # 如果需要的话，计算函数的雅可比矩阵
        self._compute_if_needed(x, *args)
        # 返回计算好的雅可比矩阵
        return self.jac
# 将用户提供的回调函数进行包装，以便可以附加属性。
def _wrap_callback(callback, method=None):
    # 如果回调函数为空或者方法是特定优化器中的一种，则不进行包装
    if callback is None or method in {'tnc', 'slsqp', 'cobyla', 'cobyqa'}:
        return callback  # 不进行包装

    # 获取回调函数的参数签名
    sig = inspect.signature(callback)

    # 根据不同的优化方法选择不同的包装方式
    if set(sig.parameters) == {'intermediate_result'}:
        # 如果参数只有一个叫做'intermediate_result'的参数
        def wrapped_callback(res):
            return callback(intermediate_result=res)
    elif method == 'trust-constr':
        # 对于'trust-constr'方法
        def wrapped_callback(res):
            return callback(np.copy(res.x), res)
    elif method == 'differential_evolution':
        # 对于'differential_evolution'方法
        def wrapped_callback(res):
            return callback(np.copy(res.x), res.convergence)
    else:
        # 对于其他情况
        def wrapped_callback(res):
            return callback(np.copy(res.x))

    # 设置包装后的回调函数的停止迭代属性为False
    wrapped_callback.stop_iteration = False
    return wrapped_callback


class OptimizeResult(_RichResult):
    """
    表示优化结果。

    Attributes
    ----------
    x : ndarray
        优化的解。
    success : bool
        优化器是否成功退出。
    status : int
        优化器的终止状态。其值取决于底层求解器。有关详细信息，请参阅 `message`。
    message : str
        终止原因的描述。
    fun, jac, hess: ndarray
        目标函数值，其雅可比矩阵和海森矩阵（如果可用）。这些海森矩阵可能是近似值，请参阅相应函数的文档。
    hess_inv : object
        目标函数海森矩阵的逆；可能是一个近似值。并非所有求解器都支持此属性。该属性的类型可能是 np.ndarray 或 scipy.sparse.linalg.LinearOperator。
    nfev, njev, nhev : int
        目标函数及其雅可比矩阵和海森矩阵的评估次数。
    nit : int
        优化器执行的迭代次数。
    maxcv : float
        最大约束违反值。

    Notes
    -----
    根据具体使用的求解器，`OptimizeResult` 可能不具有此处列出的所有属性，并且它们可能具有未在此处列出的额外属性。由于此类实质上是带有属性访问器的 dict 子类，可以使用 `OptimizeResult.keys` 方法查看可用的属性。
    """
    pass


class OptimizeWarning(UserWarning):
    pass


# 检查 Hk 是否为正定矩阵
def _check_positive_definite(Hk):
    # 判断矩阵 A 是否为正定矩阵的函数
    def is_pos_def(A):
        # 如果矩阵 A 是对称的
        if issymmetric(A):
            try:
                # 尝试进行 Cholesky 分解，若成功则返回 True，否则返回 False
                cholesky(A)
                return True
            except LinAlgError:
                return False
        else:
            return False

    # 如果 Hk 不为空，则检查其是否为正定矩阵，若不是则抛出 ValueError 异常
    if Hk is not None:
        if not is_pos_def(Hk):
            raise ValueError("'hess_inv0' matrix isn't positive definite.")


# 检查未知选项是否存在
def _check_unknown_options(unknown_options):
    # 这个函数暂时没有实现内容
    pass
    # 如果存在未知选项
    if unknown_options:
        # 将未知选项的键转换为字符串，并用逗号连接成一个消息字符串
        msg = ", ".join(map(str, unknown_options.keys()))
        # 发出警告，提示未知求解器选项，并指定警告类型为OptimizeWarning，堆栈级别为4
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, stacklevel=4)
# 定义一个函数，用于检测 `x` 是否为有限标量或有限数组标量
def is_finite_scalar(x):
    # 使用 NumPy 的 size 函数检查 x 是否只含有一个元素，并且该元素是有限的
    return np.size(x) == 1 and np.isfinite(x)


# 计算机器精度的平方根，并将结果赋值给 _epsilon
_epsilon = sqrt(np.finfo(float).eps)


# 定义一个向量范数的函数
def vecnorm(x, ord=2):
    # 如果 ord 是无穷大，则返回 x 中绝对值的最大值
    if ord == np.inf:
        return np.amax(np.abs(x))
    # 如果 ord 是负无穷，则返回 x 中绝对值的最小值
    elif ord == -np.inf:
        return np.amin(np.abs(x))
    else:
        # 否则，返回 x 的绝对值的 ord 次幂之和的 ord 次根
        return np.sum(np.abs(x)**ord, axis=0)**(1.0 / ord)


# 定义一个私有函数 _prepare_scalar_function，用于创建 ScalarFunction 对象，供标量最小化器使用
def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    """
    为标量最小化器（如 BFGS/LBFGSB/SLSQP/TNC/CG 等）创建一个 ScalarFunction 对象。

    Parameters
    ----------
    fun : callable
        要最小化的目标函数。

            ``fun(x, *args) -> float``

        其中 `x` 是形状为 (n,) 的 1-D 数组，`args` 是一个元组，包含完全
        确定函数所需的固定参数。
    x0 : ndarray, shape (n,)
        初始猜测值。一个大小为 (n,) 的实数组，其中 'n' 是独立变量的数量。
    jac : {callable, '2-point', '3-point', 'cs', None}, optional
        用于计算梯度向量的方法。如果是一个可调用对象，则应该是一个返回梯度向量的函数：

            ``jac(x, *args) -> array_like, shape (n,)``

        如果选择 `{'2-point', '3-point', 'cs'}` 中的一种，则使用相对步长计算梯度。
        如果是 `None`，则使用绝对步长的两点有限差分。
    args : tuple, optional
        传递给目标函数及其导数（`fun`，`jac` 函数）的额外参数。
    bounds : sequence, optional
        变量的边界。需要使用新样式的边界。
    eps : float or ndarray
        如果 ``jac is None``，则用于数值上近似雅可比矩阵的绝对步长。
    finite_diff_rel_step : None or array_like, optional
        如果 ``jac in ['2-point', '3-point', 'cs']``，则用于数值上近似雅可比矩阵的相对步长。
        绝对步长计算为 ``h = rel_step * sign(x0) * max(1, abs(x0))``，可能根据边界进行调整。
        对于 ``jac='3-point'``，忽略 `h` 的符号。如果为 None（默认），则步长将自动选择。
    hess : ndarray, optional
        Hessian 矩阵。如果没有给出，则使用近似值。

    """
    hess : {callable, '2-point', '3-point', 'cs', None}
        计算 Hessian 矩阵的选项。如果是可调用对象，应返回 Hessian 矩阵：
        ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
        或者可以选择有限差分数值估计的方式 {'2-point', '3-point', 'cs'}。
        当梯度通过有限差分估计时，Hessian 不能使用 {'2-point', '3-point', 'cs'} 选项估计，
        需要使用其中一种拟牛顿策略进行估计。

    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        # 如果 jac 是可调用对象，则将其作为梯度函数
        grad = jac
    elif jac in FD_METHODS:
        # 如果 jac 是有限差分方法中的一种，epsilon 设置为 None，以便 ScalarFunction 使用 rel_step
        epsilon = None
        grad = jac
    else:
        # 默认情况下（jac 是 None），使用 2-point 有限差分法并设置绝对步长
        grad = '2-point'
        epsilon = epsilon

    if hess is None:
        # 如果未提供 hess 函数，则提供一个占位实现返回 None，
        # 以便于下游的最小化器停止。不应使用 `fun.hess` 的结果。
        def hess(x, *args):
            return None

    if bounds is None:
        # 如果未提供 bounds 参数，默认设为 (-inf, inf)
        bounds = (-np.inf, np.inf)

    # ScalarFunction 的缓存。在计算梯度时重用 fun(x) 可减少总体函数评估次数。
    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf
# 定义一个函数 _clip_x_for_func，用于确保传递给 func 的 x 值在指定的 bounds 范围内
def _clip_x_for_func(func, bounds):
    # 内部函数 eval(x)，用于评估 func 在经过 bounds 裁剪后的 x 上的取值
    def eval(x):
        # 调用 _check_clip_x 函数来确保 x 在 bounds 范围内
        x = _check_clip_x(x, bounds)
        # 返回 func 在裁剪后的 x 上的计算结果
        return func(x)

    return eval


# 定义一个函数 _check_clip_x，用于检查并裁剪 x，确保其在指定的 bounds 范围内
def _check_clip_x(x, bounds):
    # 如果 x 中有任何值超出 bounds 的上下限，发出运行时警告并裁剪 x
    if (x < bounds[0]).any() or (x > bounds[1]).any():
        warnings.warn("Values in x were outside bounds during a "
                      "minimize step, clipping to bounds",
                      RuntimeWarning, stacklevel=3)
        # 使用 numpy 的 clip 函数裁剪 x 到 bounds 的范围内
        x = np.clip(x, bounds[0], bounds[1])
        return x

    return x


# 定义 Rosenbrock 函数 rosen，计算给定 x 的 Rosenbrock 函数值
def rosen(x):
    """
    The Rosenbrock function.

    The function computed is::

        sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    f : float
        The value of the Rosenbrock function.

    See Also
    --------
    rosen_der, rosen_hess, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen
    >>> X = 0.1 * np.arange(10)
    >>> rosen(X)
    76.56

    For higher-dimensional input ``rosen`` broadcasts.
    In the following example, we use this to plot a 2D landscape.
    Note that ``rosen_hess`` does not broadcast in this manner.

    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> x = np.linspace(-1, 1, 50)
    >>> X, Y = np.meshgrid(x, x)
    >>> ax = plt.subplot(111, projection='3d')
    >>> ax.plot_surface(X, Y, rosen([X, Y]))
    >>> plt.show()
    """
    x = asarray(x)
    # 计算 Rosenbrock 函数的值
    r = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                  axis=0)
    return r


# 定义 Rosenbrock 函数的导数 rosen_der，计算给定 x 的 Rosenbrock 函数的梯度
def rosen_der(x):
    """
    The derivative (i.e. gradient) of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the derivative is to be computed.

    Returns
    -------
    rosen_der : (N,) ndarray
        The gradient of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_hess, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen_der
    >>> X = 0.1 * np.arange(9)
    >>> rosen_der(X)
    array([ -2. ,  10.6,  15.6,  13.4,   6.4,  -3. , -12.4, -19.4,  62. ])

    """
    x = asarray(x)
    # 中间值计算，用于计算 Rosenbrock 函数的梯度
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    # 初始化梯度数组
    der = np.zeros_like(x)
    # 计算 Rosenbrock 函数在每个位置的梯度值
    der[1:-1] = (200 * (xm - xm_m1**2) -
                 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der


# 定义 Rosenbrock 函数的 Hessian 矩阵 rosen_hess，计算给定 x 的 Rosenbrock 函数的 Hessian 矩阵
def rosen_hess(x):
    """
    The Hessian matrix of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian is to be computed.

    Returns
    -------
    H : 2-D ndarray
        The Hessian matrix of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_der, rosen_hess_prod

    """
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.
        计算 Hessian 矩阵的点的一维数组。

    Returns
    -------
    rosen_hess : ndarray
        The Hessian matrix of the Rosenbrock function at `x`.
        在点 `x` 处 Rosenbrock 函数的 Hessian 矩阵。

    See Also
    --------
    rosen, rosen_der, rosen_hess_prod
        参见相关函数：rosen, rosen_der, rosen_hess_prod。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen_hess
    >>> X = 0.1 * np.arange(4)
    >>> rosen_hess(X)
    array([[-38.,   0.,   0.,   0.],
           [  0., 134., -40.,   0.],
           [  0., -40., 130., -80.],
           [  0.,   0., -80., 200.]])
        示例：
        导入必要的库和模块后，计算 Rosenbrock 函数在给定点 X 处的 Hessian 矩阵，并输出结果。

    """
    x = atleast_1d(x)  # 将输入的 x 转换为至少为一维的数组
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    # 构造 Hessian 矩阵的第一部分，使用 np.diag 创建对角矩阵，并偏移一位

    diagonal = np.zeros(len(x), dtype=x.dtype)
    diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
    # 设置对角线上的值，根据 Rosenbrock 函数的特定公式

    H = H + np.diag(diagonal)
    # 将两部分矩阵相加，得到最终的 Hessian 矩阵

    return H
    # 返回计算得到的 Hessian 矩阵
# 将给定的函数包装成一个新的函数，用于计算目标函数的调用次数，并方便地提供额外的参数。
def _wrap_scalar_function(function, args):
    # 初始化函数调用计数器
    ncalls = [0]
    # 如果传入的函数为空，则返回计数器和空函数
    if function is None:
        return ncalls, None

    # 定义函数的包装器，用于增加调用计数并调用目标函数
    def function_wrapper(x, *wrapper_args):
        ncalls[0] += 1
        # 将 x 的副本传递给用户定义的函数，并附加额外的参数
        fx = function(np.copy(x), *(wrapper_args + args))
        
        # 确保用户定义的目标函数返回一个标量值，而不是数组
        # 兼容性考虑：如果 fx 不是标量，则尝试将其转换为标量
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    # 返回函数调用计数器和包装后的函数
    return ncalls, function_wrapper
    def function_wrapper(x, *wrapper_args):
        # 如果函数调用次数已达到最大限制，则抛出错误
        if ncalls[0] >= maxfun:
            raise _MaxFuncCallError("Too many function calls")
        # 增加函数调用计数器
        ncalls[0] += 1
        # 将 x 的副本发送给用户定义的函数 (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        # 理想情况下，希望从 f(x) 中得到一个真正的标量返回值。为了向后兼容，也允许返回 np.array([1.3]), np.array([[1.3]]) 等。
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                # 如果用户提供的目标函数没有返回标量值，则抛出错误
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        # 返回计算得到的 fx
        return fx

    # 返回调用次数和函数包装器 function_wrapper
    return ncalls, function_wrapper
# 定义一个函数 fmin，使用下山简单型算法（Nelder-Mead simplex 算法）进行函数最小化
def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, initial_simplex=None):
    """
    Minimize a function using the downhill simplex algorithm.

    This algorithm only uses function values, not derivatives or second
    derivatives.

    Parameters
    ----------
    func : callable func(x,*args)
        The objective function to be minimized.
        要最小化的目标函数。
    x0 : ndarray
        Initial guess.
        初始猜测值。
    args : tuple, optional
        Extra arguments passed to func, i.e., ``f(x,*args)``.
        传递给 func 的额外参数，即 ``f(x,*args)``。
    xtol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
        允许收敛的迭代之间的 xopt 绝对误差。
    ftol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
        允许收敛的迭代之间的 func(xopt) 绝对误差。
    maxiter : int, optional
        Maximum number of iterations to perform.
        执行的最大迭代次数。
    maxfun : number, optional
        Maximum number of function evaluations to make.
        最大函数评估次数。
    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.
        如果希望输出 fopt 和 warnflag，则设置为 True。
    disp : bool, optional
        Set to True to print convergence messages.
        设置为 True 以打印收敛消息。
    retall : bool, optional
        Set to True to return list of solutions at each iteration.
        设置为 True 以返回每次迭代的解列表。
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
        在每次迭代后调用，参数为 callback(xk)，其中 xk 是当前的参数向量。
    initial_simplex : array_like of shape (N + 1, N), optional
        Initial simplex. If given, overrides `x0`.
        如果提供了初始单纯形，则覆盖 `x0`。
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
        ``initial_simplex[j,:]`` 应该包含单纯形中第 j 个顶点的坐标，其中
        ``N+1`` 是单纯形的顶点数，``N`` 是维数。

    Returns
    -------
    xopt : ndarray
        Parameter that minimizes function.
        最小化函数的参数。
    fopt : float
        Value of function at minimum: ``fopt = func(xopt)``.
        最小值处的函数值：``fopt = func(xopt)``。
    iter : int
        Number of iterations performed.
        执行的迭代次数。
    funcalls : int
        Number of function calls made.
        调用的函数次数。
    warnflag : int
        1 : Maximum number of function evaluations made.
            达到最大函数评估次数。
        2 : Maximum number of iterations reached.
            达到最大迭代次数。
        根据条件达到的终止标志。
    allvecs : list
        Solution at each iteration.
        每次迭代的解。

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Nelder-Mead' `method` in particular.
        minimize: 多变量函数最小化算法的接口。特别是查看 'Nelder-Mead' `method`。

    Notes
    -----
    Uses a Nelder-Mead simplex algorithm to find the minimum of function of
    one or more variables.
    使用 Nelder-Mead 单纯形算法来找到一个或多个变量函数的最小值。

    This algorithm has a long history of successful use in applications.
    But it will usually be slower than an algorithm that uses first or
    second derivative information. In practice, it can have poor
    performance in high-dimensional problems and is not robust to
    minimizing complicated functions. Additionally, there currently is no
    complete theory describing when the algorithm will successfully
    converge to the minimum, or how fast it will if it does. Both the ftol and
    xtol criteria must be met for convergence.
    该算法在应用中有着悠久的成功历史。
    但通常会比使用一阶或二阶导数信息的算法慢。在实践中，它可能在高维问题中表现不佳，并且对于最小化复杂函数不具有鲁棒性。
    此外，目前没有完整的理论描述算法何时会成功收敛到最小值，或者如果成功收敛时收敛速度如何。
    ftol 和 xtol 标准都必须满足才能收敛。

    Examples
    --------
    >>> def f(x):
    # 返回输入参数 x 的平方值
    return x**2



    # 从 scipy 库中导入 optimize 模块
    >>> from scipy import optimize

    # 使用 optimize 模块中的 fmin 函数找到函数 f 的最小值，并将结果赋给 minimum
    >>> minimum = optimize.fmin(f, 1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 17
             Function evaluations: 34

    # 输出最小值的第一个元素
    >>> minimum[0]
    -8.8817841970012523e-16

    # 引用文献参考
    References
    ----------
    .. [1] Nelder, J.A. and Mead, R. (1965), "A simplex method for function
           minimization", The Computer Journal, 7, pp. 308-313

    .. [2] Wright, M.H. (1996), "Direct Search Methods: Once Scorned, Now
           Respectable", in Numerical Analysis 1995, Proceedings of the
           1995 Dundee Biennial Conference in Numerical Analysis, D.F.
           Griffiths and G.A. Watson (Eds.), Addison Wesley Longman,
           Harlow, UK, pp. 191-208.



    # 创建选项字典 opts，包含算法的各种参数设置
    opts = {'xatol': xtol,
            'fatol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'return_all': retall,
            'initial_simplex': initial_simplex}

    # 使用 _wrap_callback 函数封装回调函数 callback
    callback = _wrap_callback(callback)

    # 使用 _minimize_neldermead 函数进行 Nelder-Mead 算法的最小化优化，并返回结果 res
    res = _minimize_neldermead(func, x0, args, callback=callback, **opts)

    # 如果 full_output 为 True，则返回详细输出结果
    if full_output:
        retlist = res['x'], res['fun'], res['nit'], res['nfev'], res['status']

        # 如果 retall 为 True，则将所有迭代结果加入返回列表
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        # 如果 retall 为 True，则仅返回最优解和所有迭代结果
        if retall:
            return res['x'], res['allvecs']
        else:
            # 否则仅返回最优解
            return res['x']
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization [1]_.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

        Note that this just clips all vertices in simplex based on
        the bounds.

    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277

    """
    # 检查并处理未知的可选参数
    _check_unknown_options(unknown_options)
    # 设置最大函数评估次数
    maxfun = maxfev
    # 设置是否返回所有迭代的最佳解标志
    retall = return_all

    # 将初始点 x0 转换为至少是一维数组，并展平
    x0 = np.atleast_1d(x0).flatten()
    # 确定 x0 数组的数据类型
    dtype = x0.dtype if np.issubdtype(x0.dtype, np.inexact) else np.float64
    x0 = np.asarray(x0, dtype=dtype)

    # 根据 adaptive 参数设置算法参数
    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    # 设置非零步长和零步长的初始值
    nonzdelt = 0.05
    zdelt = 0.00025
    # 如果给定了变量 bounds，则提取下界和上界
    if bounds is not None:
        lower_bound, upper_bound = bounds.lb, bounds.ub
        # 检查下界是否有任何一个大于对应的上界
        if (lower_bound > upper_bound).any():
            raise ValueError("Nelder Mead - one of the lower bounds "
                             "is greater than an upper bound.")
        # 检查初始猜测点 x0 是否在指定的边界范围内，发出警告如果不在范围内
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, stacklevel=3)

    # 如果 bounds 不为 None，则将初始猜测点 x0 裁剪到指定的边界范围内
    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)

    # 如果未提供初始简单形式（initial_simplex），则根据初始点 x0 创建 N+1 个顶点的初始简单形式
    if initial_simplex is None:
        N = len(x0)
        # 创建一个空的 N+1 行 N 列的数组 sim，用来存储初始简单形式的顶点
        sim = np.empty((N + 1, N), dtype=x0.dtype)
        # 第一个顶点为初始猜测点 x0
        sim[0] = x0
        # 生成其余 N 个顶点，每个顶点在一个特定维度上增加或减小一定的量
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        # 如果提供了初始简单形式，则复制它到 sim，并进行相关的验证
        sim = np.atleast_2d(initial_simplex).copy()
        dtype = sim.dtype if np.issubdtype(sim.dtype, np.inexact) else np.float64
        sim = np.asarray(sim, dtype=dtype)
        # 检查提供的初始简单形式是否符合要求
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    # 如果设置了 retall 参数，则记录所有迭代过程中的顶点
    if retall:
        allvecs = [sim[0]]

    # 如果未设置 maxiter 和 maxfun 参数，则将它们都设置为默认值
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # 如果只设置了 maxfun 而未设置 maxiter，则根据 maxfun 的值进行调整
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        # 如果只设置了 maxiter 而未设置 maxfun，则根据 maxiter 的值进行调整
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf

    # 如果设置了 bounds 参数，则对初始简单形式进行修正，确保所有顶点在指定的边界范围内
    if bounds is not None:
        # 检查简单形式中是否有任何顶点超出了上界，如果超出则将其反射到边界内部
        msk = sim > upper_bound
        sim = np.where(msk, 2*upper_bound - sim, sim)
        # 确保反射后的顶点不低于下界
        sim = np.clip(sim, lower_bound, upper_bound)

    # 创建一个从 1 到 N+1 的整数列表
    one2np1 = list(range(1, N + 1))
    # 创建一个大小为 N+1 的全为无穷大的数组，用来存储函数值
    fsim = np.full((N + 1,), np.inf, dtype=float)

    # 调用 _wrap_scalar_function_maxfun_validation 函数对 func 进行封装，以确保不超过最大函数调用次数 maxfun
    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    try:
        # 计算简单形式中每个顶点的函数值并存储在 fsim 数组中
        for k in range(N + 1):
            fsim[k] = func(sim[k])
    except _MaxFuncCallError:
        pass
    finally:
        # 对 fsim 数组中的函数值进行排序，并同时排序对应的顶点数组 sim
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
    # 使用 np.argsort 对 fsim 进行排序，并返回排序后的索引数组
    ind = np.argsort(fsim)
    # 使用 np.take 根据排序后的索引数组重新排序 fsim 数组
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    # 使用 np.take 根据排序后的索引数组重新排序 sim 数组
    sim = np.take(sim, ind, 0)

    # 初始化迭代次数为 1
    iterations = 1
    # 在函数调用次数未达到最大值且迭代次数未达到最大值时循环执行
    while (fcalls[0] < maxfun and iterations < maxiter):
        try:
            # 检查是否满足终止条件：解向量变化范围小于等于指定容差并且函数值变化小于等于指定容差
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                    np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
                break

            # 计算重心向量
            xbar = np.add.reduce(sim[:-1], 0) / N
            # 计算反射向量
            xr = (1 + rho) * xbar - rho * sim[-1]
            # 如果定义了边界，则将反射向量裁剪到指定边界内
            if bounds is not None:
                xr = np.clip(xr, lower_bound, upper_bound)
            # 计算反射向量的函数值
            fxr = func(xr)
            doshrink = 0

            # 判断反射向量的函数值是否比当前最小值更小
            if fxr < fsim[0]:
                # 计算扩展向量
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                # 如果定义了边界，则将扩展向量裁剪到指定边界内
                if bounds is not None:
                    xe = np.clip(xe, lower_bound, upper_bound)
                # 计算扩展向量的函数值
                fxe = func(xe)

                # 如果扩展向量的函数值比反射向量小，则更新对应的点和函数值
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                # 如果反射向量的函数值不比当前最小值小，继续判断是否比次小值更小
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # 执行收缩操作
                    if fxr < fsim[-1]:
                        # 计算外收缩向量
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        # 如果定义了边界，则将外收缩向量裁剪到指定边界内
                        if bounds is not None:
                            xc = np.clip(xc, lower_bound, upper_bound)
                        # 计算外收缩向量的函数值
                        fxc = func(xc)

                        # 如果外收缩向量的函数值小于等于反射向量的函数值，则更新对应的点和函数值
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        # 执行内收缩操作
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        # 如果定义了边界，则将内收缩向量裁剪到指定边界内
                        if bounds is not None:
                            xcc = np.clip(xcc, lower_bound, upper_bound)
                        # 计算内收缩向量的函数值
                        fxcc = func(xcc)

                        # 如果内收缩向量的函数值小于当前最小值的函数值，则更新对应的点和函数值
                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1

                    # 如果需要收缩操作，则执行缩小步长的操作
                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            if bounds is not None:
                                sim[j] = np.clip(
                                    sim[j], lower_bound, upper_bound)
                            fsim[j] = func(sim[j])
            iterations += 1
        except _MaxFuncCallError:
            pass
        finally:
            # 根据函数值的排序重新排列模拟点和对应的函数值
            ind = np.argsort(fsim)
            sim = np.take(sim, ind, 0)
            fsim = np.take(fsim, ind, 0)
            # 如果需要记录所有迭代点，则将当前最小值加入到记录列表中
            if retall:
                allvecs.append(sim[0])
            # 创建优化结果对象，包括当前最优解和对应的函数值
            intermediate_result = OptimizeResult(x=sim[0], fun=fsim[0])
            # 如果存在回调函数，则调用回调函数判断是否终止优化过程
            if _call_callback_maybe_halt(callback, intermediate_result):
                break

    # 设置最终返回的最优解和对应的函数值
    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0
    # 如果函数调用次数超过最大允许值
    if fcalls[0] >= maxfun:
        # 设置警告标志为1
        warnflag = 1
        # 获取最大函数评估次数警告信息
        msg = _status_message['maxfev']
        # 如果需要显示警告
        if disp:
            # 发出运行时警告，指定堆栈深度为3
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    # 否则，如果迭代次数超过最大允许值
    elif iterations >= maxiter:
        # 设置警告标志为2
        warnflag = 2
        # 获取最大迭代次数警告信息
        msg = _status_message['maxiter']
        # 如果需要显示警告
        if disp:
            # 发出运行时警告，指定堆栈深度为3
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    else:
        # 如果优化成功
        msg = _status_message['success']
        # 如果需要显示信息
        if disp:
            # 打印成功信息
            print(msg)
            # 打印当前函数值
            print("         Current function value: %f" % fval)
            # 打印迭代次数
            print("         Iterations: %d" % iterations)
            # 打印函数评估次数
            print("         Function evaluations: %d" % fcalls[0])

    # 创建优化结果对象
    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, final_simplex=(sim, fsim))
    # 如果需要返回所有向量
    if retall:
        # 将所有向量存储在结果对象中
        result['allvecs'] = allvecs
    # 返回优化结果对象
    return result
# 定义一个函数，用于计算标量或向量值函数的有限差分法导数估计
def approx_fprime(xk, f, epsilon=_epsilon, *args):
    """Finite difference approximation of the derivatives of a
    scalar or vector-valued function.

    If a function maps from :math:`R^n` to :math:`R^m`, its derivatives form
    an m-by-n matrix
    called the Jacobian, where an element :math:`(i, j)` is a partial
    derivative of f[i] with respect to ``xk[j]``.

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the gradient of `f`.
    f : callable
        Function of which to estimate the derivatives of. Has the signature
        ``f(xk, *args)`` where `xk` is the argument in the form of a 1-D array
        and `args` is a tuple of any additional fixed parameters needed to
        completely specify the function. The argument `xk` passed to this
        function is an ndarray of shape (n,) (never a scalar even if n=1).
        It must return a 1-D array_like of shape (m,) or a scalar.

        .. versionchanged:: 1.9.0
            `f` is now able to return a 1-D array-like, with the :math:`(m, n)`
            Jacobian being estimated.

    epsilon : {float, array_like}, optional
        Increment to `xk` to use for determining the function gradient.
        If a scalar, uses the same finite difference delta for all partial
        derivatives. If an array, should contain one value per element of
        `xk`. Defaults to ``sqrt(np.finfo(float).eps)``, which is approximately
        1.49e-08.
    \\*args : args, optional
        Any other arguments that are to be passed to `f`.

    Returns
    -------
    jac : ndarray
        The partial derivatives of `f` to `xk`.

    See Also
    --------
    check_grad : Check correctness of gradient function against approx_fprime.

    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::

                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[i] = ---------------------------------
                            epsilon[i]

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import optimize
    >>> def func(x, c0, c1):
    ...     "Coordinate vector `x` should be an array of size two."
    ...     return c0 * x[0]**2 + c1*x[1]**2

    >>> x = np.ones(2)
    >>> c0, c1 = (1, 200)
    >>> eps = np.sqrt(np.finfo(float).eps)
    >>> optimize.approx_fprime(x, func, [eps, np.sqrt(200) * eps], c0, c1)
    array([   2.        ,  400.00004208])

    """
    # 将输入参数 `xk` 转换为浮点数类型的 numpy 数组
    xk = np.asarray(xk, float)
    # 计算在 `xk` 处的函数值 `f0`
    f0 = f(xk, *args)

    # 调用 `approx_derivative` 函数，使用二点法计算导数估计
    return approx_derivative(f, xk, method='2-point', abs_step=epsilon,
                             args=args, f0=f0)


def check_grad(func, grad, x0, *args, epsilon=_epsilon,
                direction='all', seed=None):
    """Check the correctness of a gradient function by comparing it against a
    (forward) finite-difference approximation of the gradient.

    Parameters
    ----------
    func : callable ``func(x0, *args)``
        Function whose derivative is to be checked.
    # grad 是一个可调用对象，代表 func 的 Jacobian 矩阵
    grad : callable ``grad(x0, *args)``
        # Points to check `grad` against forward difference approximation of grad
        使用 `func` 的前向差分近似来检查 `grad` 的点。
    x0 : ndarray
        # 要检查 `grad` 的点，与 `func` 的前向差分近似 grad 对应。
    args : \\*args, optional
        # 传递给 `func` 和 `grad` 的额外参数。
    epsilon : float, optional
        # 用于有限差分近似的步长大小，默认为 ``sqrt(np.finfo(float).eps)``
        # 大约为 1.49e-08。
    direction : str, optional
        # 如果设置为 ``'random'``，则使用随机向量的梯度来检查 `grad` 的前向差分近似。
        # 默认为 ``'all'``，这种情况下，考虑所有的单热方向向量来检查 `grad`。
        # 如果 `func` 是向量值函数，则只能使用 ``'all'``。
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        # 如果 `seed` 是 None 或 `np.random`，则使用 `numpy.random.RandomState` 单例。
        # 如果 `seed` 是一个整数，则使用一个新的 ``RandomState`` 实例，并用 `seed` 初始化。
        # 如果 `seed` 已经是 ``Generator`` 或 ``RandomState`` 实例，则直接使用该实例。
        # 用于重现此函数返回值的种子。生成的随机数影响计算梯度的随机向量。注意，只有当 `direction` 参数设置为 `'random'` 时才使用 `seed`。

    Returns
    -------
    err : float
        # 差值的平方和的平方根（即 2-范数），即 ``grad(x0, *args)`` 与在点 `x0` 处使用 `func` 的有限差分近似的差值。

    See Also
    --------
    approx_fprime

    Examples
    --------
    >>> import numpy as np
    >>> def func(x):
    ...     return x[0]**2 - 0.5 * x[1]**3
    >>> def grad(x):
    ...     return [2 * x[0], -1.5 * x[1]**2]
    >>> from scipy.optimize import check_grad
    >>> check_grad(func, grad, [1.5, -1.5])
    2.9802322387695312e-08  # may vary
    >>> rng = np.random.default_rng()
    >>> check_grad(func, grad, [1.5, -1.5],
    ...             direction='random', seed=rng)
    2.9802322387695312e-08

    """
    # 设置步长为 epsilon
    step = epsilon
    # 将 x0 转换为 ndarray 格式
    x0 = np.asarray(x0)

    def g(w, func, x0, v, *args):
        # 返回 func(x0 + w*v, *args)
        return func(x0 + w*v, *args)

    if direction == 'random':
        # 将 grad(x0, *args) 转换为 ndarray
        _grad = np.asanyarray(grad(x0, *args))
        # 如果 _grad 的维度大于 1，则抛出 ValueError
        if _grad.ndim > 1:
            raise ValueError("'random' can only be used with scalar valued"
                             " func")
        # 检查随机状态
        random_state = check_random_state(seed)
        # 生成随机向量 v
        v = random_state.normal(0, 1, size=(x0.shape))
        # 构建参数元组 _args
        _args = (func, x0, v) + args
        # 定义函数 _func 为 g
        _func = g
        # 初始化变量 vars 为长度为 1 的零向量
        vars = np.zeros((1,))
        # 计算解析梯度为 _grad 和 v 的内积
        analytical_grad = np.dot(_grad, v)
    elif direction == 'all':
        # 如果 direction 参数为 'all'，则执行以下逻辑
        _args = args
        _func = func
        vars = x0
        # 计算给定函数在当前变量 vars 处的解析梯度
        analytical_grad = grad(x0, *args)
    else:
        # 如果 direction 参数不是 'all'，则抛出值错误异常
        raise ValueError(f"{direction} is not a valid string for "
                         "``direction`` argument")

    # 返回解析梯度与数值梯度之间的平方差的平方根
    return np.sqrt(np.sum(np.abs(
        (analytical_grad - approx_fprime(vars, _func, step, *_args))**2
    )))
# 使用 BFGS 算法最小化一个函数。

def fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
              epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
              retall=0, callback=None, xrtol=0, c1=1e-4, c2=0.9,
              hess_inv0=None):
    """
    Minimize a function using the BFGS algorithm.

    Parameters
    ----------
    f : callable ``f(x,*args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess, shape (n,)
    fprime : callable ``f'(x,*args)``, optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    gtol : float, optional
        Terminate successfully if gradient norm is less than `gtol`
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If `fprime` is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration. Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    """
    # 如果没有提供梯度函数 `fprime`，则需要先估算它，计算 `fprime(x0)`
    if fprime is None:
        raise ValueError("Gradient of the objective function `fprime` must be provided for BFGS optimization.")
    
    # 如果设置了回调函数，则每次迭代后调用回调函数
    if callback is not None:
        callback(x0)
    
    # 初始化当前迭代次数
    k = 0
    # 初始化 BFGS 近似的逆 Hessian 矩阵，如果未提供则默认为单位矩阵
    Hk = hess_inv0 if hess_inv0 is not None else np.eye(len(x0))
    
    # 迭代过程
    while True:
        # 计算梯度 fprime(xk)
        gfk = fprime(x0, *args)
        # 使用 BFGS 更新方程计算搜索方向 pk
        pk = -np.dot(Hk, gfk)
        try:
            # 进行线搜索，返回一个合适的步长 ret
            ret = _line_search_wolfe12(f, fprime, x0, pk, gfk, None, None, c1=c1, c2=c2)
        except _LineSearchError:
            # 如果线搜索失败，则退出迭代
            break
        
        # 获取步长和下一个迭代点 xk
        alpha_k = ret[0]
        xkp1 = x0 + alpha_k * pk
        sk = xkp1 - x0
        x0 = xkp1
        
        # 如果提供了最大迭代次数且达到了最大迭代次数，则退出迭代
        if maxiter is not None:
            if k >= maxiter:
                break
        
        # 如果提供了梯度范数的容忍度 gtol 并且满足条件，则退出迭代
        gnorm = np.linalg.norm(fprime(x0, *args), ord=norm)
        if gnorm < gtol:
            break
        
        # 更新 Hessian 的近似逆矩阵 Hk
        yk = fprime(xkp1, *args) - gfk
        ro = 1.0 / np.dot(yk, sk)
        A1 = np.eye(len(x0)) - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = np.eye(len(x0)) - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + ro * sk[:, np.newaxis] * sk[np.newaxis, :]

        # 更新迭代次数
        k += 1
    
    # 如果需要返回所有的迭代点，则将其存储在数组中
    if retall:
        return x0, f(x0), k, {'grad': gfk, 'task': 'CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL'}
    # 否则只返回最终的迭代点和最小值
    else:
        return x0, f(x0), k
    full_output : bool, optional
        # 如果为True，额外返回`fopt`、`func_calls`、`grad_calls`和`warnflag`
        如果为True，则返回`fopt`、`func_calls`、`grad_calls`和`warnflag`

    disp : bool, optional
        # 如果为True，打印收敛消息
        如果为True，则打印收敛消息

    retall : bool, optional
        # 如果为True，返回每次迭代的结果列表
        如果为True，则返回每次迭代的结果列表

    xrtol : float, default: 0
        # 相对于`x`的相对容差。如果步长小于`xk * xrtol`，则认为成功终止，其中`xk`是当前参数向量。
        相对于`x`的相对容差。如果步长小于`xk * xrtol`，则认为成功终止，其中`xk`是当前参数向量。

    c1 : float, default: 1e-4
        # Armijo条件规则的参数。
        Armijo条件规则的参数。

    c2 : float, default: 0.9
        # 曲率条件规则的参数。
        曲率条件规则的参数。

    hess_inv0 : None or ndarray, optional
        # 初始的Hessian逆矩阵估计值，形状为(n, n)。如果为None（默认），则使用单位矩阵。
        初始的Hessian逆矩阵估计值，形状为(n, n)。如果为None（默认），则使用单位矩阵。

    Returns
    -------
    xopt : ndarray
        # 最小化函数`f`的参数，即`f(xopt) == fopt`。
        最小化函数`f`的参数，即`f(xopt) == fopt`。

    fopt : float
        # 最小值。
        最小值。

    gopt : ndarray
        # 在最小处的梯度值，f'(xopt)，应接近0。
        在最小处的梯度值，f'(xopt)，应接近0。

    Bopt : ndarray
        # 1/f''(xopt)的值，即Hessian矩阵的逆。
        1/f''(xopt)的值，即Hessian矩阵的逆。

    func_calls : int
        # 调用的函数数。
        调用的函数数。

    grad_calls : int
        # 调用的梯度函数数。
        调用的梯度函数数。

    warnflag : integer
        # 警告标志：
        # 1：超过最大迭代次数。
        # 2：梯度和/或函数调用未改变。
        # 3：遇到NaN结果。
        警告标志：
        1：超过最大迭代次数。
        2：梯度和/或函数调用未改变。
        3：遇到NaN结果。

    allvecs : list
        # 每次迭代中`xopt`的值的列表。仅在`retall`为True时返回。
        每次迭代中`xopt`的值的列表。仅在`retall`为True时返回。

    Notes
    -----
    # 使用Broyden，Fletcher，Goldfarb和Shanno（BFGS）的拟牛顿方法优化函数`f`，其梯度由`fprime`给出。

    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
    # 参数`c1`和`c2`必须满足``0 < c1 < c2 < 1``。

    See Also
    --------
    minimize: 多变量函数的优化算法接口。特别是参见`method='BFGS'`。

    References
    ----------
    # Wright和Nocedal的《数值优化》，1999年，第198页。

    Examples
    --------
    # 示例
    >>> import numpy as np
    >>> from scipy.optimize import fmin_bfgs
    >>> def quadratic_cost(x, Q):
    ...     return x @ Q @ x
    ...
    >>> x0 = np.array([-3, -4])
    >>> cost_weight =  np.diag([1., 10.])
    >>> # 注意：单个元素的元组需要有一个尾随逗号
    >>> fmin_bfgs(quadratic_cost, x0, args=(cost_weight,))
    优化成功终止。
            当前函数值：0.000000
            迭代次数：7                   # 可能有所不同
            函数调用次数：24              # 可能有所不同
            梯度调用次数：8                # 可能有所不同
    array([ 2.85169950e-06, -4.61820139e-07])

    >>> def quadratic_cost_grad(x, Q):
    ...     return 2 * Q @ x
    ...
    >>> fmin_bfgs(quadratic_cost, x0, quadratic_cost_grad, args=(cost_weight,))
    """
    # 定义优化选项字典，用于传递给优化算法
    opts = {'gtol': gtol,         # 梯度的阈值，优化过程停止条件之一
            'norm': norm,         # 规范化梯度向量的方式
            'eps': epsilon,       # 控制数值梯度计算的步长
            'disp': disp,         # 是否打印优化过程信息
            'maxiter': maxiter,   # 最大迭代次数
            'return_all': retall, # 是否返回所有迭代点
            'xrtol': xrtol,       # 控制结果精度的相对公差
            'c1': c1,             # 光滑条件常数，用于线搜索
            'c2': c2,             # 曲率条件常数，用于线搜索
            'hess_inv0': hess_inv0}  # 初始的 Hessian 逆估计矩阵

    # 将回调函数包装后传递给优化算法
    callback = _wrap_callback(callback)
    # 使用 BFGS 方法进行优化，返回优化结果
    res = _minimize_bfgs(f, x0, args, fprime, callback=callback, **opts)

    # 如果需要完整输出
    if full_output:
        # 返回包含各种优化结果的元组
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'],
                   res['nfev'], res['njev'], res['status'])
        # 如果需要返回所有迭代点
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        # 如果需要返回所有迭代点
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']
    """
# 使用 BFGS 算法最小化一个或多个变量的标量函数。
def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-5, norm=np.inf, eps=_epsilon, maxiter=None,
                   disp=False, return_all=False, finite_diff_rel_step=None,
                   xrtol=0, c1=1e-4, c2=0.9,
                   hess_inv0=None, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages. 控制是否打印收敛信息。
    maxiter : int
        Maximum number of iterations to perform. 最大迭代次数。
    gtol : float
        Terminate successfully if gradient norm is less than `gtol`. 如果梯度范数小于 `gtol`，则成功终止。
    norm : float
        Order of norm (Inf is max, -Inf is min). 范数的阶数（Inf 表示最大，-Inf 表示最小）。
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences. 如果 `jac` 是 `None`，则是用于通过前向差分数值近似雅可比矩阵的绝对步长。
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations. 设置为 True 时，返回每次迭代的最佳解列表。
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically. 如果 `jac` 是 ['2-point', '3-point', 'cs'] 中的一种，则用于数值近似雅可比矩阵的相对步长。绝对步长的计算方式为 `h = rel_step * sign(x) * max(1, abs(x))`，可能会调整以适应边界。对于 `jac='3-point'`，忽略 `h` 的符号。如果为 None（默认），则自动选择步长。
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step size is
        less than ``xk * xrtol`` where ``xk`` is the current parameter vector.
        `x` 的相对容差。如果步长小于 `xk * xrtol`，则成功终止，其中 `xk` 是当前参数向量。
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule. Armijo 条件规则的参数。
    c2 : float, default: 0.9
        Parameter for curvature condition rule. 曲率条件规则的参数。
    hess_inv0 : None or ndarray, optional
        Initial inverse hessian estimate, shape (n, n). If None (default) then
        the identity matrix is used. 初始的逆 Hessian 矩阵估计值，形状为 (n, n)。如果为 None（默认），则使用单位矩阵。

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``. 参数 `c1` 和 `c2` 必须满足 ``0 < c1 < c2 < 1``。

    If minimization doesn't complete successfully, with an error message of
    ``Desired error not necessarily achieved due to precision loss``, then
    consider setting `gtol` to a higher value. This precision loss typically
    occurs when the (finite difference) numerical differentiation cannot provide
    sufficient precision to satisfy the `gtol` termination criterion.
    This can happen when working in single precision and a callable jac is not
    provided. For single precision problems a `gtol` of 1e-3 seems to work.
    如果最小化未能成功完成，并出现 ``Desired error not necessarily achieved due to precision loss`` 的错误消息，则考虑将 `gtol` 设置为较高的值。通常情况下，这种精度损失发生在（有限差分）数值微分无法提供足够精度以满足 `gtol` 终止标准时。这可能发生在单精度问题中，并且未提供可调用的 `jac`。对于单精度问题，`gtol` 设置为 1e-3 似乎有效。
    """
    _check_unknown_options(unknown_options)
    _check_positive_definite(hess_inv0)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    # 准备标量函数对象
    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    # 获取标量函数对象中的目标函数值
    f = sf.fun
    # 获取目标函数的梯度函数
    myfprime = sf.grad

    # 计算初始点处的目标函数值
    old_fval = f(x0)
    
    # 计算初始点处的梯度
    gfk = myfprime(x0)

    # 设置迭代次数 k 的初始值
    k = 0
    
    # 获取初始点的维度
    N = len(x0)
    
    # 创建单位矩阵 I，数据类型为整型
    I = np.eye(N, dtype=int)
    
    # 如果没有提供初始的逆海森矩阵估计值，则使用单位矩阵作为初始值
    Hk = I if hess_inv0 is None else hess_inv0

    # 设置初始步长估计值，dx 初始约为 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    # 设置当前迭代点 xk 的初始值为 x0
    xk = x0
    
    # 如果需要记录所有迭代点，则初始化列表并加入初始点 x0
    if retall:
        allvecs = [x0]
    
    # 警告标志，初始值为 0
    warnflag = 0
    
    # 计算初始梯度范数
    gnorm = vecnorm(gfk, ord=norm)
    
    # 迭代优化过程，终止条件为梯度范数小于 gtol 或达到最大迭代次数 maxiter
    while (gnorm > gtol) and (k < maxiter):
        # 计算搜索方向 pk
        pk = -np.dot(Hk, gfk)
        
        try:
            # 执行 Wolfe 线搜索，获取步长 alpha_k 及相关信息
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100,
                                          amax=1e100, c1=c1, c2=c2)
        except _LineSearchError:
            # 若线搜索未找到更好的解决方案，则设置警告标志为 2 并终止迭代
            warnflag = 2
            break

        # 计算更新步长 sk
        sk = alpha_k * pk
        
        # 计算更新后的迭代点 xkp1
        xkp1 = xk + sk
        
        # 若需要记录所有迭代点，则将当前迭代点 xkp1 加入列表
        if retall:
            allvecs.append(xkp1)
        
        # 更新迭代点 xk 为 xkp1
        xk = xkp1
        
        # 如果 gfkp1 为 None，则重新计算 xkp1 处的梯度 gfkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        # 计算更新的 yk
        yk = gfkp1 - gfk
        
        # 更新 gfk 为 gfkp1
        gfk = gfkp1
        
        # 更新迭代次数 k
        k += 1
        
        # 生成优化结果对象
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        
        # 检查是否应该调用回调函数并可能中止迭代
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
        
        # 计算更新后的梯度范数 gnorm
        gnorm = vecnorm(gfk, ord=norm)
        
        # 检查是否满足停止条件：alpha_k * ||pk|| <= xrtol * (xrtol + ||xk||)
        if (alpha_k * vecnorm(pk) <= xrtol * (xrtol + vecnorm(xk))):
            break
        
        # 检查目标函数值是否为有限值，若不是则设置警告标志为 2 并终止迭代
        if not np.isfinite(old_fval):
            warnflag = 2
            break

        # 计算更新的 rho_k 的逆
        rhok_inv = np.dot(yk, sk)
        
        # 若 rhok_inv 为 0，则设置 rhok 为 1000，并在需要时打印消息
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                msg = "Divide-by-zero encountered: rhok assumed large"
                _print_success_message_or_warn(True, msg)
        else:
            # 否则计算更新的 rho_k
            rhok = 1. / rhok_inv

        # 更新海森矩阵逆估计值 Hk
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])

    # 将最终的目标函数值设为 old_fval
    fval = old_fval
    
    # 根据警告标志设置相应的消息
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']
    # 如果 disp 参数为真，则打印成功消息或警告
    _print_success_message_or_warn(warnflag, msg)
    # 打印当前函数值
    print("         Current function value: %f" % fval)
    # 打印迭代次数
    print("         Iterations: %d" % k)
    # 打印函数评估次数
    print("         Function evaluations: %d" % sf.nfev)
    # 打印梯度评估次数
    print("         Gradient evaluations: %d" % sf.ngev)

result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                        njev=sf.ngev, status=warnflag,
                        success=(warnflag == 0), message=msg, x=xk,
                        nit=k)
# 如果 retall 参数为真，则将所有迭代向量保存在结果中
if retall:
    result['allvecs'] = allvecs
# 返回优化结果对象
return result
# 打印成功消息或警告信息
def _print_success_message_or_warn(warnflag, message, warntype=None):
    # 如果 warnflag 为 False，则打印消息 message
    if not warnflag:
        print(message)
    # 否则，发出警告消息 message，使用 warntype 或默认的 OptimizeWarning 类型
    else:
        warnings.warn(message, warntype or OptimizeWarning, stacklevel=3)


# 使用非线性共轭梯度算法最小化函数
def fmin_cg(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
            epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0,
            callback=None, c1=1e-4, c2=0.4):
    """
    Minimize a function using a nonlinear conjugate gradient algorithm.

    Parameters
    ----------
    f : callable, ``f(x, *args)``
        Objective function to be minimized. Here `x` must be a 1-D array of
        the variables that are to be changed in the search for a minimum, and
        `args` are the other (fixed) parameters of `f`.
    x0 : ndarray
        A user-supplied initial estimate of `xopt`, the optimal value of `x`.
        It must be a 1-D array of values.
    fprime : callable, ``fprime(x, *args)``, optional
        A function that returns the gradient of `f` at `x`. Here `x` and `args`
        are as described above for `f`. The returned value must be a 1-D array.
        Defaults to None, in which case the gradient is approximated
        numerically (see `epsilon`, below).
    args : tuple, optional
        Parameter values passed to `f` and `fprime`. Must be supplied whenever
        additional fixed parameters are needed to completely specify the
        functions `f` and `fprime`.
    gtol : float, optional
        Stop when the norm of the gradient is less than `gtol`.
    norm : float, optional
        Order to use for the norm of the gradient
        (``-np.inf`` is min, ``np.inf`` is max).
    epsilon : float or ndarray, optional
        Step size(s) to use when `fprime` is approximated numerically. Can be a
        scalar or a 1-D array. Defaults to ``sqrt(eps)``, with eps the
        floating point machine precision.  Usually ``sqrt(eps)`` is about
        1.5e-8.
    maxiter : int, optional
        Maximum number of iterations to perform. Default is ``200 * len(x0)``.
    full_output : bool, optional
        If True, return `fopt`, `func_calls`, `grad_calls`, and `warnflag` in
        addition to `xopt`.  See the Returns section below for additional
        information on optional return values.
    disp : bool, optional
        If True, return a convergence message, followed by `xopt`.
    retall : bool, optional
        If True, add to the returned values the results of each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each iteration.
        Called as ``callback(xk)``, where ``xk`` is the current value of `x0`.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.4
        Parameter for curvature condition rule.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    """
    fopt : float, optional
        最小化函数 f(xopt) 找到的最小值。仅当 `full_output` 为 True 时返回。

    func_calls : int, optional
        调用的函数计数。仅当 `full_output` 为 True 时返回。

    grad_calls : int, optional
        调用的梯度计数。仅当 `full_output` 为 True 时返回。

    warnflag : int, optional
        整数值，表示警告状态。仅当 `full_output` 为 True 时返回。

        0 : 成功。

        1 : 超过最大迭代次数。

        2 : 梯度和/或函数调用没有变化，可能表明精度丢失，即算法未收敛。

        3 : 遇到 NaN 结果。

    allvecs : list of ndarray, optional
        数组列表，包含每次迭代的结果。仅当 `retall` 为 True 时返回。

    See Also
    --------
    minimize : `scipy.optimize` 所有无约束和约束多变量函数最小化算法的通用接口。
               提供了调用 `fmin_cg` 的另一种方式，即指定 `method='CG'`。

    Notes
    -----
    本共轭梯度算法基于 Polak 和 Ribiere 的算法 [1]_。

    共轭梯度法在以下情况下通常效果更好：

    1. `f` 具有唯一的全局最小点，没有局部最小值或其他稳定点，
    2. `f` 在至少局部上可以由变量的二次函数合理近似，
    3. `f` 是连续的，并且具有连续的梯度，
    4. `fprime` 不太大，例如，其范数小于 1000，
    5. 初始猜测 `x0` 相对于 `f` 的全局最小化点 `xopt` 是合理接近的。

    参数 `c1` 和 `c2` 必须满足 ``0 < c1 < c2 < 1``。

    References
    ----------
    .. [1] Wright & Nocedal, "Numerical Optimization", 1999, pp. 120-122.

    Examples
    --------
    示例 1：寻找表达式 ``a*u**2 + b*u*v + c*v**2 + d*u + e*v + f`` 的最小值，
    给定参数值和初始猜测 ``(u, v) = (0, 0)``。

    >>> import numpy as np
    >>> args = (2, 3, 7, 8, 9, 10)  # 参数值
    >>> def f(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
    >>> def gradf(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     gu = 2*a*u + b*v + d     # 梯度的 u 分量
    ...     gv = b*u + 2*c*v + e     # 梯度的 v 分量
    ...     return np.asarray((gu, gv))
    >>> x0 = np.asarray((0, 0))  # 初始猜测
    >>> from scipy import optimize
    >>> res1 = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
    优化成功终止。
             当前函数值: 1.617021
             迭代次数: 4
             函数调用次数: 8
             梯度调用次数: 8
    opts = {'gtol': gtol,                   # 允许的最大梯度范数。默认为 1e-5。
            'norm': norm,                   # 梯度的范数类型。默认为无穷大。
            'eps': epsilon,                 # 用于数值梯度计算的步长。默认为 1.4901161193847656e-08。
            'disp': disp,                   # 是否显示优化过程信息。默认为 True。
            'maxiter': maxiter,             # 最大迭代次数。默认为 None。
            'return_all': retall}           # 是否返回所有迭代点。默认为 False。

    # 将回调函数包装成适当的形式
    callback = _wrap_callback(callback)
    # 使用共轭梯度法进行优化
    res = _minimize_cg(f, x0, args, fprime, callback=callback, c1=c1, c2=c2,
                       **opts)

    if full_output:
        # 如果需要完整输出
        retlist = res['x'], res['fun'], res['nfev'], res['njev'], res['status']
        if retall:
            # 如果需要返回所有迭代点的话，加入到返回列表中
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            # 如果需要返回所有迭代点的话，只返回最优解和所有迭代点
            return res['x'], res['allvecs']
        else:
            # 否则只返回最优解
            return res['x']
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.
    """

    # 检查未知选项，确保参数传递正确
    _check_unknown_options(unknown_options)

    # 是否返回所有迭代步骤的最佳解，默认为否
    retall = return_all

    # 将初始点 x0 转换为数组，并展平处理
    x0 = asarray(x0).flatten()

    # 如果未设置最大迭代次数，则默认为变量数乘以200
    if maxiter is None:
        maxiter = len(x0) * 200

    # 准备标量函数，包括计算函数值和梯度
    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    # 获取函数值和梯度函数
    f = sf.fun
    myfprime = sf.grad

    # 计算初始函数值和梯度
    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0  # 初始化迭代步数
    xk = x0  # 设置当前迭代点为初始点 x0

    # 设置初始步长估计为 dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    # 如果需要返回所有迭代步骤的解，则记录初始点
    if retall:
        allvecs = [xk]

    warnflag = 0  # 警告标志位初始化为0
    pk = -gfk  # 设置初始搜索方向为负梯度方向
    gnorm = vecnorm(gfk, ord=norm)  # 计算梯度的范数，用于收敛判据

    # Armijo条件的参数，控制步长选择
    c1 = 1e-4
    c2 = 0.4

    # 用于Wolfe条件的参数，控制曲率条件
    sigma_3 = 0.01
    while (gnorm > gtol) and (k < maxiter):
        deltak = np.dot(gfk, gfk)
        
        cached_step = [None]
        
        # 定义 Polak-Ribiere-Powell 方法的步骤函数
        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = vecnorm(gfkp1, ord=norm)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)
        
        # 定义下降条件函数
        def descent_condition(alpha, xkp1, fp1, gfkp1):
            # Polak-Ribiere+ 需要显式检查充分下降条件，这不被强 Wolfe 约束保证。
            #
            # 参见 Gilbert & Nocedal, "Global convergence properties of
            # conjugate gradient methods for optimization",
            # SIAM J. Optimization 2, 21 (1992).
            cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step
            
            # 如果步骤导致收敛，则接受该步骤
            if gnorm <= gtol:
                return True
            
            # 如果充分下降条件适用，则接受该步骤
            return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)
        
        try:
            # 进行线搜索以找到合适的步长 alpha_k
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval,
                                          old_old_fval, c1=c1, c2=c2, amin=1e-100,
                                          amax=1e100, extra_condition=descent_condition)
        except _LineSearchError:
            # 线搜索未能找到更好的解决方案。
            warnflag = 2
            break
        
        # 如果 alpha_k 与缓存步骤相同，则重复使用已计算的结果
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)
        
        # 如果需要保存所有迭代点，则将当前点 xk 添加到 allvecs 列表中
        if retall:
            allvecs.append(xk)
        
        # 增加迭代次数 k
        k += 1
        
        # 创建 OptimizeResult 对象作为中间结果
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        
        # 如果需要，调用回调函数，可能中断优化过程
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
    
    # 获取最终函数值
    fval = old_fval
    
    # 根据不同的终止条件设置相应的警告标志和消息
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']
    
    # 如果需要显示优化结果，则打印相关信息
    if disp:
        _print_success_message_or_warn(warnflag, msg)
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
    # 创建一个 OptimizeResult 对象，并初始化各个属性
    result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    # 如果 retall 为 True，则将 allvecs 添加到结果字典中
    if retall:
        result['allvecs'] = allvecs
    # 返回构建好的结果对象
    return result
def fmin_ncg(f, x0, fprime, fhess_p=None, fhess=None, args=(), avextol=1e-5,
             epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0,
             callback=None, c1=1e-4, c2=0.9):
    """
    Unconstrained minimization of a function using the Newton-CG method.

    Parameters
    ----------
    f : callable ``f(x, *args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable ``f'(x, *args)``
        Gradient of f.
    fhess_p : callable ``fhess_p(x, p, *args)``, optional
        Function which computes the Hessian of f times an
        arbitrary vector, p.
    fhess : callable ``fhess(x, *args)``, optional
        Function to compute the Hessian matrix of f.
    args : tuple, optional
        Extra arguments passed to f, fprime, fhess_p, and fhess
        (the same set of extra arguments is supplied to all of
        these functions).
    epsilon : float or ndarray, optional
        If fhess is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function which is called after
        each iteration. Called as callback(xk), where xk is the
        current parameter vector.
    avextol : float, optional
        Convergence is assumed when the average relative error in
        the minimizer falls below this amount.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return the optional outputs.
    disp : bool, optional
        If True, print convergence message.
    retall : bool, optional
        If True, return a list of results at each iteration.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float
        Value of the function at xopt, i.e., ``fopt = f(xopt)``.
    fcalls : int
        Number of function calls made.
    gcalls : int
        Number of gradient calls made.
    hcalls : int
        Number of Hessian calls made.
    warnflag : int
        Warnings generated by the algorithm.
        1 : Maximum number of iterations exceeded.
        2 : Line search failure (precision loss).
        3 : NaN result encountered.
    allvecs : list
        The result at each iteration, if retall is True (see below).

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Newton-CG' `method` in particular.

    Notes
    -----
    Only one of `fhess_p` or `fhess` need to be given.  If `fhess`
    is provided, then `fhess_p` will be ignored. If neither `fhess`
    nor `fhess_p` is provided, then the hessian product will be
    approximated using finite differences on `fprime`. `fhess_p`
    must compute the hessian times an arbitrary vector. If it is not
    """
    # 实现使用 Newton-CG 方法对无约束条件的函数进行最小化

    # 初始化参数及选项
    # 参数 x0：初始猜测
    # 参数 fprime：函数 f 的梯度
    # 参数 fhess_p：计算 f Hessian 矩阵与任意向量 p 的函数
    # 参数 fhess：计算 f 的 Hessian 矩阵的函数
    # 参数 args：传递给 f、fprime、fhess_p 和 fhess 的额外参数
    # 参数 epsilon：用于近似 fhess 时的步长大小
    # 参数 callback：可选的用户提供的每次迭代后调用的函数
    # 参数 avextol：收敛标准，当最小化器中的平均相对误差低于此值时认为收敛
    # 参数 maxiter：最大迭代次数
    # 参数 full_output：如果为 True，则返回可选的输出结果
    # 参数 disp：如果为 True，则打印收敛消息
    # 参数 retall：如果为 True，则返回每次迭代的结果列表
    # 参数 c1：Armijo 条件规则的参数
    # 参数 c2：曲率条件规则的参数

    # 返回值
    # 返回 xopt：使 f 最小化的参数，即 f(xopt) == fopt
    # 返回 fopt：在 xopt 处函数的值，即 fopt = f(xopt)
    # 返回 fcalls：调用的函数数
    # 返回 gcalls：调用的梯度数
    # 返回 hcalls：调用的 Hessian 数
    # 返回 warnflag：算法生成的警告
    # 返回 allvecs：如果 retall 为 True，则返回每次迭代的结果列表
    """
    given, finite-differences on `fprime` are used to compute
    it.

    Newton-CG methods are also called truncated Newton methods. This
    function differs from scipy.optimize.fmin_tnc because

    1. scipy.optimize.fmin_ncg is written purely in Python using NumPy
        and scipy while scipy.optimize.fmin_tnc calls a C function.
    2. scipy.optimize.fmin_ncg is only for unconstrained minimization
        while scipy.optimize.fmin_tnc is for unconstrained minimization
        or box constrained minimization. (Box constraints give
        lower and upper bounds for each variable separately.)

    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    References
    ----------
    Wright & Nocedal, 'Numerical Optimization', 1999, p. 140.
    """
    # 设置优化选项
    opts = {'xtol': avextol,  # 相对步长允许的最大误差
            'eps': epsilon,   # 数值梯度计算时使用的小步长
            'maxiter': maxiter,  # 最大迭代次数
            'disp': disp,     # 是否显示优化过程信息
            'return_all': retall}  # 是否返回所有优化过程中的解

    # 包装回调函数
    callback = _wrap_callback(callback)
    # 使用 Newton-CG 方法进行优化
    res = _minimize_newtoncg(f, x0, args, fprime, fhess, fhess_p,
                             callback=callback, c1=c1, c2=c2, **opts)

    # 如果需要完整输出
    if full_output:
        # 返回包含结果和统计信息的元组
        retlist = (res['x'], res['fun'], res['nfev'], res['njev'],
                   res['nhev'], res['status'])
        # 如果同时需要返回所有优化过程中的解
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        # 如果只需要最优解
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']
def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=_epsilon, maxiter=None,
                       disp=False, return_all=False, c1=1e-4, c2=0.9,
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Note that the `jac` parameter (Jacobian) is required.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
    """
    _check_unknown_options(unknown_options)  # 检查未知的选项参数

    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')  # 如果没有提供 Jacobi 矩阵，则抛出数值错误

    fhess_p = hessp  # 将参数 hessp 赋值给 fhess_p
    fhess = hess  # 将参数 hess 赋值给 fhess
    avextol = xtol  # 将参数 xtol 赋值给 avextol
    epsilon = eps  # 将参数 eps 赋值给 epsilon
    retall = return_all  # 将参数 return_all 赋值给 retall

    x0 = asarray(x0).flatten()  # 将 x0 转换为 numpy 数组并展平

    # TODO: add hessp (callable or FD) to ScalarFunction?
    # 准备标量函数对象，传入函数 fun、参数 x0、Jacobi 矩阵 jac，以及其他参数
    sf = _prepare_scalar_function(
        fun, x0, jac, args=args, epsilon=eps, hess=hess
    )
    f = sf.fun  # 获取标量函数的值
    fprime = sf.grad  # 获取标量函数的梯度
    _h = sf.hess(x0)  # 计算标量函数在 x0 处的黑塞矩阵

    # Logic for hess/hessp
    # - If a callable(hess) is provided, then use that
    # - If hess is a FD_METHOD, or the output from hess(x) is a LinearOperator
    #   then create a hessp function using those.
    # - If hess is None but you have callable(hessp) then use the hessp.
    # - If hess and hessp are None then approximate hessp using the grad/jac.

    # 如果 hess 是 FD_METHOD 中的一种或者 _h 是 LinearOperator 类型，则 fhess 设为 None
    if (hess in FD_METHODS or isinstance(_h, LinearOperator)):
        fhess = None

        # 定义 _hessp 函数，用于计算 hessp
        def _hessp(x, p, *args):
            return sf.hess(x).dot(p)

        fhess_p = _hessp

    # 定义终止优化的函数
    def terminate(warnflag, msg):
        if disp:
            _print_success_message_or_warn(warnflag, msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % sf.nfev)
            print("         Gradient evaluations: %d" % sf.ngev)
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                                njev=sf.ngev, nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    hcalls = 0  # 初始化黑塞矩阵的调用次数为 0
    # 如果 maxiter 参数未指定，则设为初始点 x0 的长度乘以 200
    if maxiter is None:
        maxiter = len(x0)*200
    # 设定共轭梯度法的最大迭代次数为初始点 x0 的长度乘以 20
    cg_maxiter = 20*len(x0)

    # 设定容许误差 xtol 为初始点 x0 的长度乘以 avextol
    xtol = len(x0) * avextol
    # 初始化更新 L1 范数为浮点数最大值，确保进入 while 循环
    update_l1norm = np.finfo(float).max
    # 复制初始点 x0 为当前迭代点 xk
    xk = np.copy(x0)
    # 如果需要记录每次迭代的点，则初始化列表并添加初始点 x0
    if retall:
        allvecs = [xk]
    # 初始化迭代次数 k 为 0
    k = 0
    # 初始化梯度 gfk 为 None
    gfk = None
    # 计算初始点 x0 的函数值作为 old_fval
    old_fval = f(x0)
    # 初始化 old_old_fval 为 None
    old_old_fval = None
    # 设定 float64eps 为 np.float64 类型的机器精度
    float64eps = np.finfo(np.float64).eps
    
    # 如果有异常情况，例如 old_fval 或 update_l1norm 为 NaN，则返回终止信息
    else:
        if np.isnan(old_fval) or np.isnan(update_l1norm):
            return terminate(3, _status_message['nan'])
    
        # 设置成功信息并返回终止信息，状态码为 0
        msg = _status_message['success']
        return terminate(0, msg)
def fminbound(func, x1, x2, args=(), xtol=1e-5, maxfun=500,
              full_output=0, disp=1):
    """Bounded minimization for scalar functions.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized (must accept and return scalars).
    x1, x2 : float or array scalar
        Finite optimization bounds.
    args : tuple, optional
        Extra arguments passed to function.
    xtol : float, optional
        The convergence tolerance.
    maxfun : int, optional
        Maximum number of function evaluations allowed.
    full_output : bool, optional
        If True, return optional outputs.
    disp : int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.


    Returns
    -------
    xopt : ndarray
        Parameters (over given interval) which minimize the
        objective function.
    fval : number
        (Optional output) The function value evaluated at the minimizer.
    ierr : int
        (Optional output) An error flag (0 if converged, 1 if maximum number of
        function calls reached).
    numfunc : int
        (Optional output) The number of function calls made.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Bounded' `method` in particular.

    Notes
    -----
    Finds a local minimizer of the scalar function `func` in the
    interval x1 < xopt < x2 using Brent's method. (See `brent`
    for auto-bracketing.)

    References
    ----------
    .. [1] Forsythe, G.E., M. A. Malcolm, and C. B. Moler. "Computer Methods
           for Mathematical Computations." Prentice-Hall Series in Automatic
           Computation 259 (1977).
    .. [2] Brent, Richard P. Algorithms for Minimization Without Derivatives.
           Courier Corporation, 2013.

    Examples
    --------
    `fminbound` finds the minimizer of the function in the given range.
    The following examples illustrate this.

    >>> from scipy import optimize
    >>> def f(x):
    ...     return (x-1)**2
    >>> minimizer = optimize.fminbound(f, -4, 4)
    >>> minimizer
    1.0
    >>> minimum = f(minimizer)
    >>> minimum
    0.0
    >>> res = optimize.fminbound(f, 3, 4, full_output=True)
    >>> minimizer, fval, ierr, numfunc = res
    >>> minimizer
    3.000005960860986
    >>> minimum = f(minimizer)
    >>> minimum, fval
    (4.000023843479476, 4.000023843479476)
    """
    
    # 定义优化参数
    options = {'xatol': xtol,
               'maxiter': maxfun,
               'disp': disp}

    # 调用内部函数进行标量函数的有界最小化
    res = _minimize_scalar_bounded(func, (x1, x2), args, **options)
    
    # 根据 full_output 参数决定返回值
    if full_output:
        return res['x'], res['fun'], res['status'], res['nfev']
    else:
        return res['x']
def _minimize_scalar_bounded(func, bounds, args=(),
                             xatol=1e-5, maxiter=500, disp=0,
                             **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.
    xatol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    # 检查并处理未知的可选参数
    _check_unknown_options(unknown_options)
    # 将 maxiter 赋值给 maxfun，这是为了与旧版本的兼容性
    maxfun = maxiter
    # 检查边界是否为正确的形式
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    # 将 bounds 分解为两个变量 x1 和 x2
    x1, x2 = bounds

    # 检查边界 x1 和 x2 是否是有限的标量
    if not (is_finite_scalar(x1) and is_finite_scalar(x2)):
        raise ValueError("Optimization bounds must be finite scalars.")

    # 检查 lower bound 是否小于等于 upper bound
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    # 初始化 flag 为 0
    flag = 0
    # 设置 header 用于打印表头信息
    header = ' Func-count     x          f(x)          Procedure'
    # 设置 step 的初始值
    step = '       initial'

    # 设置常数 sqrt_eps 和 golden_mean
    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    # 初始化 a 和 b 为 bounds 的两个端点
    a, b = x1, x2
    # 计算初始的 fulc
    fulc = a + golden_mean * (b - a)
    # 初始化 nfc 和 xf 为 fulc
    nfc, xf = fulc, fulc
    # 初始化 rat, e, x, fx, num, fmin_data, fu 的值
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = np.inf

    # 初始化 ffulc 和 fnfc 为 fx
    ffulc = fnfc = fx
    # 计算 xm, tol1 和 tol2
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    # 如果 disp 大于 2，则打印表头信息
    if disp > 2:
        print(" ")
        print(header)
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))
    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # 检查是否满足停止条件，即最优解未收敛到足够接近
        if np.abs(e) > tol1:
            golden = 0
            # 计算抛物线拟合的系数
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # 检查抛物线是否可接受
            if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'

                # 检查步长是否在边界容忍范围内
                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # 进行黄金分割步骤
                golden = 1

        if golden:  # 进行黄金分割步骤
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e
            step = '       golden'

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        # 计算函数在新点 x 处的值
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        # 如果需要打印详细信息
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        # 根据函数值的比较更新搜索区间和候选点
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            # 根据函数值的比较更新候选点
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        # 更新搜索区间的中点和容忍度
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        # 检查是否达到最大函数调用次数限制
        if num >= maxfun:
            flag = 1
            break

    # 如果最终找到的解包含 NaN 值，则设置标志为 2
    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    # 返回优化结果
    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)

    result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                            message={0: 'Solution found.',
                                     1: 'Maximum number of function calls '
                                        'reached.',
                                     2: _status_message['nan']}.get(flag, ''),
                            x=xf, nfev=num, nit=num)

    return result
class Brent:
    # 需要重新考虑 __init__ 的设计
    def __init__(self, func, args=(), tol=1.48e-8, maxiter=500,
                 full_output=0, disp=0):
        # 初始化 Brent 类的实例
        self.func = func  # 保存传入的目标函数
        self.args = args  # 保存传入的额外参数
        self.tol = tol  # 设置收敛精度
        self.maxiter = maxiter  # 设置最大迭代次数
        self._mintol = 1.0e-11  # 最小收敛精度
        self._cg = 0.3819660  # 常数值
        self.xmin = None  # 初始化最小值位置为 None
        self.fval = None  # 初始化最小函数值为 None
        self.iter = 0  # 初始化迭代次数为 0
        self.funcalls = 0  # 初始化函数调用次数为 0
        self.disp = disp  # 是否显示中间过程的标志

    # 需要重新考虑 set_bracket 的设计（新增选项等）
    def set_bracket(self, brack=None):
        # 设置 brack 属性为传入的 brack 参数
        self.brack = brack

    def get_bracket_info(self):
        # 准备数据
        func = self.func
        args = self.args
        brack = self.brack
        
        ### BEGIN core bracket_info code ###
        ### 仔细记录核心 bracket_info 代码的任何变更 ###
        if brack is None:
            # 调用 bracket 函数寻找初始的 bracket 区间和对应的函数值
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
        elif len(brack) == 2:
            # 使用传入的 brack 作为 bracket 区间的端点
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                       xb=brack[1], args=args)
        elif len(brack) == 3:
            # 使用传入的 brack 作为完整的 bracket 区间
            xa, xb, xc = brack
            if (xa > xc):  # 确保 xa < xc
                xc, xa = xa, xc
            if not ((xa < xb) and (xb < xc)):
                raise ValueError(
                    "Bracketing values (xa, xb, xc) do not"
                    " fulfill this requirement: (xa < xb) and (xb < xc)"
                )
            # 计算每个点的函数值
            fa = func(*((xa,) + args))
            fb = func(*((xb,) + args))
            fc = func(*((xc,) + args))
            if not ((fb < fa) and (fb < fc)):
                raise ValueError(
                    "Bracketing values (xa, xb, xc) do not fulfill"
                    " this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
                )
            funcalls = 3
        else:
            raise ValueError("Bracketing interval must be "
                             "length 2 or 3 sequence.")
        ### END core bracket_info code ###

        return xa, xb, xc, fa, fb, fc, funcalls

    def get_result(self, full_output=False):
        if full_output:
            # 返回完整输出结果
            return self.xmin, self.fval, self.iter, self.funcalls
        else:
            # 返回仅最小值位置的结果
            return self.xmin
    # 定义一个可选参数 `brack`，可以是一个三元组 `(xa, xb, xc)`，要求 `xa < xb < xc` 且 `func(xb) < func(xa) and func(xb) < func(xc)`，
    # 或者是一个二元组 `(xa, xb)`，作为下坡搜索的初始点 (参见 `scipy.optimize.bracket`).
    # 最小化器找到的最优点 `x` 不一定满足 `xa <= x <= xb` 的条件。
    tol : float, optional
        # 解决方案 `xopt` 的相对误差，用于判断收敛性。
    full_output : bool, optional
        # 如果为 True，则返回所有输出参数 (xmin, fval, iter, funcalls)。
    maxiter : int, optional
        # 解决方案的最大迭代次数。
    Returns
    -------
    xmin : ndarray
        # 最优点。
    fval : float
        # (可选输出) 最优函数值。
    iter : int
        # (可选输出) 迭代次数。
    funcalls : int
        # (可选输出) 目标函数调用次数。

    See also
    --------
    minimize_scalar: 用于标量单变量函数的最小化算法的接口。特别是查看 `Brent` 方法。

    Notes
    -----
    # 当可能时，使用反向抛物线插值来加速黄金分割法的收敛。
    # 不确保最小值在 `brack` 指定的范围内。参见 `scipy.optimize.fminbound`。

    Examples
    --------
    # 我们展示了在 `brack` 大小为 2 和 3 时函数的行为。当 `brack` 是 `(xa, xb)` 形式时，
    # 我们可以看到对于给定的值，输出不一定位于 `(xa, xb)` 的范围内。

    >>> def f(x):
    ...     return (x-1)**2

    >>> from scipy import optimize

    >>> minimizer = optimize.brent(f, brack=(1, 2))
    >>> minimizer
    1
    >>> res = optimize.brent(f, brack=(-1, 0.5, 2), full_output=True)
    >>> xmin, fval, iter, funcalls = res
    >>> f(xmin), fval
    (0.0, 0.0)
# 定义了一个使用 Brent 方法进行标量函数最小化的函数
def _minimize_scalar_brent(func, brack=None, args=(), xtol=1.48e-8,
                           maxiter=500, disp=0,
                           **unknown_options):
    """
    Options
    -------
    maxiter : int
        最大迭代次数。
    xtol : float
        允许的解 `xopt` 的相对误差，用于收敛判据。
    disp: int, optional
        如果非零，打印消息。
            0 : 不打印任何消息。
            1 : 仅打印非收敛通知消息。
            2 : 收敛时也打印消息。
            3 : 打印迭代结果。
    Notes
    -----
    当可能时，使用反向抛物线插值加速黄金分割法的收敛。

    """
    # 检查并处理未知选项
    _check_unknown_options(unknown_options)
    # 设置容差
    tol = xtol
    # 如果容差小于0，引发值错误
    if tol < 0:
        raise ValueError('tolerance should be >= 0, got %r' % tol)

    # 创建 Brent 对象，设置函数、参数、容差、全输出、最大迭代次数和显示标志
    brent = Brent(func=func, args=args, tol=tol,
                  full_output=True, maxiter=maxiter, disp=disp)
    # 设置初始搜索区间
    brent.set_bracket(brack)
    # 执行最优化
    brent.optimize()
    # 获取最优化结果，包括最小值点、函数值、迭代次数和函数调用次数
    x, fval, nit, nfev = brent.get_result(full_output=True)

    # 判断是否成功找到最小值
    success = nit < maxiter and not (np.isnan(x) or np.isnan(fval))

    # 如果成功找到最小值
    if success:
        message = ("\nOptimization terminated successfully;\n"
                   "The returned value satisfies the termination criteria\n"
                   f"(using xtol = {xtol} )")
    else:
        # 如果迭代次数超过了最大限制
        if nit >= maxiter:
            message = "\nMaximum number of iterations exceeded"
        # 如果最小值或函数值为 NaN
        if np.isnan(x) or np.isnan(fval):
            message = f"{_status_message['nan']}"

    # 如果需要显示消息
    if disp:
        # 打印成功消息或警告
        _print_success_message_or_warn(not success, message)

    # 返回优化结果对象
    return OptimizeResult(fun=fval, x=x, nit=nit, nfev=nfev,
                          success=success, message=message)


# 定义了使用黄金分割法寻找函数最小值的函数
def golden(func, args=(), brack=None, tol=_epsilon,
           full_output=0, maxiter=5000):
    """
    Return the minimizer of a function of one variable using the golden section
    method.

    Given a function of one variable and a possible bracketing interval,
    return a minimizer of the function isolated to a fractional precision of
    tol.

    Parameters
    ----------
    func : callable func(x,*args)
        Objective function to minimize.
    args : tuple, optional
        Additional arguments (if present), passed to func.
    brack : tuple, optional
        Either a triple ``(xa, xb, xc)`` where ``xa < xb < xc`` and
        ``func(xb) < func(xa) and  func(xb) < func(xc)``, or a pair (xa, xb)
        to be used as initial points for a downhill bracket search (see
        `scipy.optimize.bracket`).
        The minimizer ``x`` will not necessarily satisfy ``xa <= x <= xb``.
    tol : float, optional
        x tolerance stop criterion
    full_output : bool, optional
        If True, return optional outputs.
    maxiter : int
        Maximum number of iterations to perform.

    Returns
    -------
    xmin : ndarray
        Optimum point.

    """
    # 返回使用黄金分割法找到的函数的最小值点
    pass
    # fval : float
    #     (Optional output) Optimum function value.
    # funcalls : int
    #     (Optional output) Number of objective function evaluations made.

    # See also
    # --------
    # minimize_scalar: Interface to minimization algorithms for scalar
    #     univariate functions. See the 'Golden' `method` in particular.

    # Notes
    # -----
    # Uses analog of bisection method to decrease the bracketed
    # interval.

    # Examples
    # --------
    # We illustrate the behaviour of the function when `brack` is of
    # size 2 and 3, respectively. In the case where `brack` is of the
    # form (xa,xb), we can see for the given values, the output need
    # not necessarily lie in the range ``(xa, xb)``.
    options = {'xtol': tol, 'maxiter': maxiter}
    # 调用 _minimize_scalar_golden 函数进行最小化优化，传入参数和选项
    res = _minimize_scalar_golden(func, brack, args, **options)
    if full_output:
        # 如果 full_output 为 True，则返回包含最优解、最优函数值和评估次数的结果字典
        return res['x'], res['fun'], res['nfev']
    else:
        # 如果 full_output 为 False，则仅返回最优解
        return res['x']
def _minimize_scalar_golden(func, brack=None, args=(),
                            xtol=_epsilon, maxiter=5000, disp=0,
                            **unknown_options):
    """
    Options
    -------
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.
    """
    # 检查未知选项，确保没有未知参数传递给函数
    _check_unknown_options(unknown_options)
    
    # 设置容差值为相对误差容限
    tol = xtol
    
    # 根据提供的 brack 参数，选择合适的初始搜索区间
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                   xb=brack[1], args=args)
    elif len(brack) == 3:
        xa, xb, xc = brack
        # 如果 xa > xc，则交换它们，确保 xa < xc
        if (xa > xc):  
            xc, xa = xa, xc
        # 检查是否满足要求 (xa < xb) and (xb < xc)
        if not ((xa < xb) and (xb < xc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not"
                " fulfill this requirement: (xa < xb) and (xb < xc)"
            )
        # 计算初始点的函数值
        fa = func(*((xa,) + args))
        fb = func(*((xb,) + args))
        fc = func(*((xc,) + args))
        # 设置函数调用次数为 3
        funcalls = 3
    else:
        raise ValueError("Bracketing interval must be length 2 or 3 sequence.")
    
    # 黄金分割法中的常数设定
    _gR = 0.61803399  # 黄金分割比率的共轭值：2.0/(1.0+sqrt(5.0))
    _gC = 1.0 - _gR
    
    # 初始化搜索点
    x3 = xc
    x0 = xa
    # 根据初始点的距离选择初始的 x1 和 x2
    if (np.abs(xc - xb) > np.abs(xb - xa)):
        x1 = xb
        x2 = xb + _gC * (xc - xb)
    else:
        x2 = xb
        x1 = xb - _gC * (xb - xa)
    
    # 计算初始点的函数值
    f1 = func(*((x1,) + args))
    f2 = func(*((x2,) + args))
    # 增加函数调用次数
    funcalls += 2
    
    # 初始化迭代计数
    nit = 0
    
    # 如果 disp 大于 2，则打印表头和迭代结果
    if disp > 2:
        print(" ")
        print(f"{'Func-count':^12} {'x':^12} {'f(x)': ^12}")
    
    # 开始迭代优化过程
    for i in range(maxiter):
        # 判断是否达到收敛条件
        if np.abs(x3 - x0) <= tol * (np.abs(x1) + np.abs(x2)):
            break
        # 根据函数值比较更新搜索区间
        if (f2 < f1):
            x0 = x1
            x1 = x2
            x2 = _gR * x1 + _gC * x3
            f1 = f2
            f2 = func(*((x2,) + args))
        else:
            x3 = x2
            x2 = x1
            x1 = _gR * x2 + _gC * x0
            f2 = f1
            f1 = func(*((x1,) + args))
        # 增加函数调用次数
        funcalls += 1
        # 如果 disp 大于 2，则打印迭代结果
        if disp > 2:
            if (f1 < f2):
                xmin, fval = x1, f1
            else:
                xmin, fval = x2, f2
            print(f"{funcalls:^12g} {xmin:^12.6g} {fval:^12.6g}")
        
        # 增加迭代计数
        nit += 1
    
    # 迭代结束后确定最优解和最优函数值
    if (f1 < f2):
        xmin = x1
        fval = f1
    else:
        xmin = x2
        fval = f2
    # 检查优化是否成功：迭代次数小于最大迭代次数并且返回值和最小值均不是 NaN
    success = nit < maxiter and not (np.isnan(fval) or np.isnan(xmin))

    # 如果优化成功
    if success:
        # 设置成功消息，包括使用的 xtol 参数值
        message = ("\nOptimization terminated successfully;\n"
                   "The returned value satisfies the termination criteria\n"
                   f"(using xtol = {xtol} )")
    else:
        # 如果迭代次数超过了最大迭代次数
        if nit >= maxiter:
            # 设置消息为迭代次数超过最大值的提示
            message = "\nMaximum number of iterations exceeded"
        # 如果最小值或函数值是 NaN
        if np.isnan(xmin) or np.isnan(fval):
            # 设置消息为 NaN 相关的默认消息
            message = f"{_status_message['nan']}"

    # 如果 disp 参数为 True，则根据 success 输出成功或警告信息
    if disp:
        _print_success_message_or_warn(not success, message)

    # 返回优化结果对象，包括函数值、函数调用次数、最优解、迭代次数、成功标志和消息
    return OptimizeResult(fun=fval, nfev=funcalls, x=xmin, nit=nit,
                          success=success, message=message)
def bracket(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000):
    """
    Bracket the minimum of a function.

    Given a function and distinct initial points, search in the
    downhill direction (as defined by the initial points) and return
    three points that bracket the minimum of the function.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to minimize.
    xa, xb : float, optional
        Initial points. Defaults `xa` to 0.0, and `xb` to 1.0.
        A local minimum need not be contained within this interval.
    args : tuple, optional
        Additional arguments (if present), passed to `func`.
    grow_limit : float, optional
        Maximum grow limit.  Defaults to 110.0
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 1000.

    Returns
    -------
    xa, xb, xc : float
        Final points of the bracket.
    fa, fb, fc : float
        Objective function values at the bracket points.
    funcalls : int
        Number of function evaluations made.

    Raises
    ------
    BracketError
        If no valid bracket is found before the algorithm terminates.
        See notes for conditions of a valid bracket.

    Notes
    -----
    The algorithm attempts to find three strictly ordered points (i.e.
    :math:`x_a < x_b < x_c` or :math:`x_c < x_b < x_a`) satisfying
    :math:`f(x_b) ≤ f(x_a)` and :math:`f(x_b) ≤ f(x_c)`, where one of the
    inequalities must be satistfied strictly and all :math:`x_i` must be
    finite.

    Examples
    --------
    This function can find a downward convex region of a function:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.optimize import bracket
    >>> def f(x):
    ...     return 10*x**2 + 3*x + 5
    >>> x = np.linspace(-2, 2)
    >>> y = f(x)
    >>> init_xa, init_xb = 0.1, 1
    >>> xa, xb, xc, fa, fb, fc, funcalls = bracket(f, xa=init_xa, xb=init_xb)
    >>> plt.axvline(x=init_xa, color="k", linestyle="--")
    >>> plt.axvline(x=init_xb, color="k", linestyle="--")
    >>> plt.plot(x, y, "-k")
    >>> plt.plot(xa, fa, "bx")
    >>> plt.plot(xb, fb, "rx")
    >>> plt.plot(xc, fc, "bx")
    >>> plt.show()

    Note that both initial points were to the right of the minimum, and the
    third point was found in the "downhill" direction: the direction
    in which the function appeared to be decreasing (to the left).
    The final points are strictly ordered, and the function value
    at the middle point is less than the function values at the endpoints;
    it follows that a minimum must lie within the bracket.

    """
    _gold = 1.618034  # golden ratio: (1.0+sqrt(5.0))/2.0
    _verysmall_num = 1e-21
    # convert to numpy floats if not already
    xa, xb = np.asarray([xa, xb])  # 将初始点转换为numpy浮点数（如果尚未）
    fa = func(*(xa,) + args)  # 计算函数在xa处的值
    fb = func(*(xb,) + args)  # 计算函数在xb处的值
    if (fa < fb):                      # 如果fa < fb，则交换xa和xb，确保fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    # 计算新的试探点 xc，根据黄金分割法计算下一个试探点
    xc = xb + _gold * (xb - xa)
    
    # 计算函数在 xc 处的取值
    fc = func(*((xc,) + args))
    
    # 初始化函数调用次数和迭代次数
    funcalls = 3
    iter = 0
    
    # 开始循环直到找到合适的区间
    while (fc < fb):
        # 计算临时变量 tmp1 和 tmp2
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        
        # 计算新的试探点 w
        val = tmp2 - tmp1
        if np.abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        
        # 更新 wlim
        wlim = xb + grow_limit * (xc - xb)
        
        # 消息提示，用于迭代次数达到上限时抛出异常
        msg = ("No valid bracket was found before the iteration limit was "
               "reached. Consider trying different initial points or "
               "increasing `maxiter`.")
        if iter > maxiter:
            raise RuntimeError(msg)
        
        # 更新迭代次数
        iter += 1
        
        # 判断 w 是否在有效的区间内，并计算函数在 w 处的取值
        if (w - xc) * (xb - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            # 根据函数值更新区间和函数值
            if (fw < fc):
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif (fw > fb):
                xc = w
                fc = fw
                break
            # 更新 w 和 fw
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim)*(wlim - xc) >= 0.0:
            w = wlim
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim)*(xc - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            # 根据函数值更新区间和函数值
            if (fw < fc):
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(*((w,) + args))
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        
        # 更新 xa, xb, xc, fa, fb, fc
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    # 判断是否找到了有效的区间条件
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = (xa < xb < xc or xc < xb < xa)
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)
    
    # 若三个条件中有任何一个不满足，抛出 BracketError 异常
    msg = ("The algorithm terminated without finding a valid bracket. "
           "Consider trying different initial points.")
    if not (cond1 and cond2 and cond3):
        e = BracketError(msg)
        e.data = (xa, xb, xc, fa, fb, fc, funcalls)
        raise e
    
    # 返回找到的有效区间的端点和相关信息
    return xa, xb, xc, fa, fb, fc, funcalls
# 定义一个自定义的异常类，继承自 RuntimeError，用于处理括号错误
class BracketError(RuntimeError):
    pass


# 根据提供的参数和选项恢复从括号错误中，调用求解器 solver 处理函数 fun
def _recover_from_bracket_error(solver, fun, bracket, args, **options):
    try:
        # 使用提供的求解器 solver，以及参数 bracket 和其他选项调用处理函数 fun
        res = solver(fun, bracket, args, **options)
    except BracketError as e:
        # 如果捕获到 BracketError 异常，说明括号 bracket 不合法
        msg = str(e)
        # 从异常对象 e 中获取之前存储的数据
        xa, xb, xc, fa, fb, fc, funcalls = e.data
        # 检查是否有 NaN 值
        if np.any(np.isnan([xa, xb, xc, fa, fb, fc])):
            x, fun = np.nan, np.nan
        else:
            # 找到最小函数值的索引
            imin = np.argmin([fa, fb, fc])
            # 获取对应的最小点和函数值
            x, fun = [xa, xb, xc][imin], [fa, fb, fc][imin]
        # 返回优化结果对象 OptimizeResult，表示未成功
        return OptimizeResult(fun=fun, nfev=funcalls, x=x,
                              nit=0, success=False, message=msg)
    # 如果没有捕获到异常，直接返回正常的计算结果
    return res


# 给定参数向量 x0 和方向向量 alpha，以及每个参数的下界和上界，
# 计算标量 l 的边界，使得 lower_bound <= x0 + alpha * l <= upper_bound
def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # 获取 alpha 中非零元素的索引，以避免出现零除错误。
    # alpha 不会全为零，因为它是从 _linesearch_powell 函数调用而来，
    # 在该函数中已经对此进行了检查。
    nonzero, = alpha.nonzero()
    
    # 根据非零索引从 lower_bound 和 upper_bound 中取出相应的值
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    # 同样，从 x0 和 alpha 中也取出相应的非零元素
    x0, alpha = x0[nonzero], alpha[nonzero]
    
    # 计算 l 的下界和上界
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # 确定 alpha 中正数和负数的索引
    pos = alpha > 0

    # 计算 lmin 和 lmax 的正负值
    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    # 计算最终的 lmin 和 lmax
    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # 如果 x0 超出了边界，则说明当前方向 alpha 可能无法使参数回到边界内。
    # 在这种情况下，lmax < lmin。
    # 如果是这样，则返回 (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)
# 使用修改后的Powell方法进行函数最小化
def fmin_powell(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
                maxfun=None, full_output=0, disp=1, retall=0, callback=None,
                direc=None):
    """
    Minimize a function using modified Powell's method.

    This method only uses function values, not derivatives.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized.
        要最小化的目标函数。
    x0 : ndarray
        Initial guess.
        初始猜测值。
    args : tuple, optional
        Extra arguments passed to func.
        传递给函数的额外参数。

    xtol, ftol, maxiter, maxfun, full_output, disp, retall, callback, direc:
        控制算法行为的其他参数。
    """
    
    def _linesearch_powell(func, p, xi, tol=1e-3,
                           lower_bound=None, upper_bound=None, fval=None):
        """Line-search algorithm using fminbound.

        Find the minimum of the function ``func(x0 + alpha*direc)``.

        lower_bound : np.array.
            The lower bounds for each parameter in ``x0``. If the ``i``th
            parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
            should be ``-np.inf``.
            Note ``np.shape(lower_bound) == (n,)``.
        upper_bound : np.array.
            The upper bounds for each parameter in ``x0``. If the ``i``th
            parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
            should be ``np.inf``.
            Note ``np.shape(upper_bound) == (n,)``.
        fval : number.
            ``fval`` is equal to ``func(p)``, the idea is just to avoid
            recomputing it so we can limit the ``fevals``.

        """
        def myfunc(alpha):
            return func(p + alpha*xi)

        # if xi is zero, then don't optimize
        if not np.any(xi):
            return ((fval, p, xi) if fval is not None else (func(p), p, xi))
        elif lower_bound is None and upper_bound is None:
            # non-bounded minimization
            res = _recover_from_bracket_error(_minimize_scalar_brent,
                                              myfunc, None, tuple(), xtol=tol)
            alpha_min, fret = res.x, res.fun
            xi = alpha_min * xi
            return squeeze(fret), p + xi, xi
        else:
            bound = _line_for_search(p, xi, lower_bound, upper_bound)
            if np.isneginf(bound[0]) and np.isposinf(bound[1]):
                # equivalent to unbounded
                return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
            elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
                # we can use a bounded scalar minimization
                res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
                xi = res.x * xi
                return squeeze(res.fun), p + xi, xi
            else:
                # only bounded on one side. use the tangent function to convert
                # the infinity bound to a finite bound. The new bounded region
                # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
                bound = np.arctan(bound[0]), np.arctan(bound[1])
                res = _minimize_scalar_bounded(
                    lambda x: myfunc(np.tan(x)),
                    bound,
                    xatol=tol / 100)
                xi = np.tan(res.x) * xi
                return squeeze(res.fun), p + xi, xi
    xtol : float, optional
        # 控制线搜索的容差。
    ftol : float, optional
        # 相对误差，用于判断收敛时 `func(xopt)` 的可接受程度。
    maxiter : int, optional
        # 允许执行的最大迭代次数。
    maxfun : int, optional
        # 允许进行的最大函数评估次数。
    full_output : bool, optional
        # 如果为 True，则返回 `fopt`, `xi`, `direc`, `iter`, `funcalls`, 和 `warnflag`。
    disp : bool, optional
        # 如果为 True，则打印收敛信息。
    retall : bool, optional
        # 如果为 True，则返回每次迭代的解列表。
    callback : callable, optional
        # 可选的用户提供的函数，在每次迭代后调用。以 `callback(xk)` 形式调用，其中 `xk` 是当前的参数向量。
    direc : ndarray, optional
        # 初始拟合步长和参数顺序设置为 (N, N) 数组，其中 N 是 `x0` 中拟合参数的数量。默认为步长为 1.0，同时拟合所有参数 (`np.eye((N, N))`)。为了防止初始考虑步骤中的值或更改初始步长，在第 M 个块中将其设置为 0 或期望的步长大小，其中 J 是 `x0` 中的位置，M 是所需的评估步骤，步骤按索引顺序评估。步长和顺序会随着最小化过程自由变化。

    Returns
    -------
    xopt : ndarray
        # 使 `func` 最小化的参数。
    fopt : number
        # 最小值处函数的值：`fopt = func(xopt)`。
    direc : ndarray
        # 当前方向设置。
    iter : int
        # 迭代次数。
    funcalls : int
        # 执行的函数调用次数。
    warnflag : int
        # 整数警告标志：
            1 : 达到最大函数评估次数。
            2 : 达到最大迭代次数。
            3 : 遇到 NaN 结果。
            4 : 结果超出提供的边界。
    allvecs : list
        # 每次迭代的解列表。

    See also
    --------
    minimize: 用于多变量函数的无约束最小化算法的接口。特别参见 'Powell' 方法。

    Notes
    -----
    使用 Powell 方法的修改版来寻找 N 变量函数的最小值。Powell 方法是一种共轭方向方法。

    该算法有两个循环。外部循环仅迭代内部循环。内部循环在方向集中每个当前方向上进行最小化。在内部循环结束时，如果满足某些条件，则放弃给出最大减少的方向，并用当前估计的 x 与内部循环开始时估计的 x 的差替换该方向。

    替换最大增加方向的技术条件实际上是检查以下内容：
    """
    opts = {'xtol': xtol,
            'ftol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'direc': direc,
            'return_all': retall}

    # 将回调函数用_wrapp_callback包装
    callback = _wrap_callback(callback)
    # 调用_minimize_powell函数进行Powell方法的最小化优化
    res = _minimize_powell(func, x0, args, callback=callback, **opts)

    # 如果需要完整输出
    if full_output:
        # 设置返回结果的元组
        retlist = (res['x'], res['fun'], res['direc'], res['nit'],
                   res['nfev'], res['status'])
        # 如果需要返回所有迭代结果
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        # 如果需要返回所有迭代结果
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']
    ```
# 使用修改后的 Powell 算法最小化一个或多个变量的标量函数

def _minimize_powell(func, x0, args=(), callback=None, bounds=None,
                     xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
                     disp=False, direc=None, return_all=False,
                     **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    modified Powell algorithm.

    Parameters
    ----------
    func : callable
        要最小化的目标函数。

            ``func(x, *args) -> float``

        其中 ``x`` 是形状为 (n,) 的一维数组，``args`` 是一个元组，包含了完全
        确定函数所需的固定参数。
    x0 : ndarray, shape (n,)
        初始猜测。大小为 (n,) 的实数组成的数组，其中 ``n`` 是独立变量的数量。
    args : tuple, optional
        传递给目标函数及其导数 (`func`, `jac` 和 `hess` 函数) 的额外参数。
    bounds : sequence or `Bounds`, optional
        决策变量的边界。有两种指定边界的方式：

            1. `Bounds` 类的实例。
            2. 每个 `x` 元素的 ``(min, max)`` 对的序列。使用 `None` 表示无边界。

        如果未提供边界，则将使用无边界线性搜索。
        如果提供了边界并且初始猜测在边界内，则最小化过程中的每个函数评估都将在边界内。
        如果提供了边界，并且初始猜测在边界外，并且 `direc` 是全秩的（或者保持默认值），
        则第一次迭代期间的一些函数评估可能在边界外，但在第一次迭代后的每次函数评估都将在边界内。
        如果 `direc` 不是全秩的，则可能无法优化某些参数，并且不能保证解在边界内。

    xtol, ftol : float, optional
        允许的参数和函数值的容差，以确定是否终止优化。
    maxiter : int, optional
        执行的最大迭代次数。根据方法，每次迭代可能使用多个函数评估。
    maxfev : int, optional
        允许的最大函数调用次数。
    disp : bool, optional
        设置为 True 以打印收敛消息。
    direc : ndarray, optional
        搜索方向的初始向量。默认情况下为单位向量。
    return_all : bool, optional
        如果为 True，则将所有函数评估和迭代步骤的信息返回。

    callback : callable, optional
        每次迭代后调用的函数。签名为：

            ``callback(xk)``

        其中 ``xk`` 是当前参数向量。

    **unknown_options : dict, optional
        求解器选项的字典。所有方法都接受以下通用选项：

            maxiter : int
                执行的最大迭代次数。根据方法，每次迭代可能使用多个函数评估。
            disp : bool
                设置为 True 以打印收敛消息。

        请参阅 ``method='powell'`` 的特定于方法的选项。

    Returns
    -------
    ``OptimizeResult``
        包含有关最小化的信息的对象。

    """
    res : OptimizeResult
        # 优化结果，表示为 ``OptimizeResult`` 对象。
        # 重要的属性包括：``x`` 表示解决方案数组，``success`` 是一个布尔标志，指示优化器是否成功退出，
        # ``message`` 描述了终止的原因。详细属性描述请参见 `OptimizeResult`。

    Options
    -------
    disp : bool
        # 设置为 True 以打印收敛消息。
    xtol : float
        # 可接受的解 ``xopt`` 的相对误差，用于判定收敛。
    ftol : float
        # 可接受的 ``fun(xopt)`` 的相对误差，用于判定收敛。
    maxiter, maxfev : int
        # 允许的最大迭代次数和函数评估次数。
        # 如果未设置 `maxiter` 或 `maxfev`，将默认为 ``N*1000``，其中 ``N`` 是变量的数量。
        # 如果同时设置了 `maxiter` 和 `maxfev`，则优化会在第一个达到的条件下停止。
    direc : ndarray
        # Powell 方法的初始方向向量集合。
    return_all : bool, optional
        # 设置为 True 以返回每次迭代的最佳解的列表。
    """
    # 检查未知选项，如果存在未知选项则引发错误
    _check_unknown_options(unknown_options)

    # 将 maxfev 的值赋给 maxfun，以备后用
    maxfun = maxfev

    # 将 return_all 的值赋给 retall，以备后用
    retall = return_all

    # 将 x0 转换为一维数组并赋给 x
    x = asarray(x0).flatten()

    # 如果 retall 为 True，则创建一个包含初始解 x 的列表
    if retall:
        allvecs = [x]

    # 获取变量 x 的长度并赋给 N
    N = len(x)

    # 如果 maxiter 和 maxfun 都未设置，则将它们都设为默认值 N * 1000
    if maxiter is None and maxfun is None:
        maxiter = N * 1000
        maxfun = N * 1000
    elif maxiter is None:
        # 如果 maxiter 未设置，则根据 maxfun 的值决定是否将其设为 N * 1000 或 np.inf
        if maxfun == np.inf:
            maxiter = N * 1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        # 如果 maxfun 未设置，则根据 maxiter 的值决定是否将其设为 N * 1000 或 np.inf
        if maxiter == np.inf:
            maxfun = N * 1000
        else:
            maxfun = np.inf

    # 我们需要使用一个可变对象，在包装函数中更新它
    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    # 如果 direc 为 None，则使用单位矩阵作为初始方向向量集合
    if direc is None:
        direc = eye(N, dtype=float)
    else:
        # 如果 direc 不为 None，则将其转换为浮点类型的 ndarray
        direc = asarray(direc, dtype=float)
        # 如果 direc 不是全秩的，则发出警告
        if np.linalg.matrix_rank(direc) != direc.shape[0]:
            warnings.warn("direc input is not full rank, some parameters may "
                          "not be optimized",
                          OptimizeWarning, stacklevel=3)

    # 如果 bounds 为 None，则将 lower_bound 和 upper_bound 都设置为 None，避免不必要的检查
    if bounds is None:
        lower_bound, upper_bound = None, None
    else:
        # 如果不是第一次迭代，则使用规范化后的边界值
        lower_bound, upper_bound = bounds.lb, bounds.ub
        # 检查初始猜测点是否超出了指定的边界，如果是则发出警告
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, stacklevel=3)

    # 计算目标函数在点 x 处的值
    fval = squeeze(func(x))
    # 复制当前点 x 到 x1
    x1 = x.copy()
    # 初始化迭代次数
    iter = 0
    while True:
        try:
            # 将当前目标函数值保存在 fx 中
            fx = fval
            # 初始化参数
            bigind = 0
            delta = 0.0
            # 对于每个方向进行 Powell 方法的线搜索
            for i in range(N):
                direc1 = direc[i]
                # 在当前方向 direc1 上进行 Powell 方法的线搜索
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(func, x, direc1,
                                                     tol=xtol * 100,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     fval=fval)
                # 更新最大下降值和对应的方向索引
                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i
            # 迭代次数加一
            iter += 1
            # 如果需要记录所有中间结果，则将当前点 x 添加到 allvecs 中
            if retall:
                allvecs.append(x)
            # 构建当前迭代的结果对象
            intermediate_result = OptimizeResult(x=x, fun=fval)
            # 检查是否需要调用回调函数并可能终止优化过程
            if _call_callback_maybe_halt(callback, intermediate_result):
                break
            # 计算终止条件的容差
            bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
            # 检查是否满足终止条件
            if 2.0 * (fx - fval) <= bnd:
                break
            # 检查是否超过最大允许的函数调用次数
            if fcalls[0] >= maxfun:
                break
            # 检查是否超过最大允许的迭代次数
            if iter >= maxiter:
                break
            # 检查是否出现 NaN 值
            if np.isnan(fx) and np.isnan(fval):
                # 如果出现 NaN，则终止优化过程
                break

            # 构造外推点
            direc1 = x - x1
            x1 = x.copy()
            # 确保在外推时不超出指定的边界
            if lower_bound is None and upper_bound is None:
                lmax = 1
            else:
                _, lmax = _line_for_search(x, direc1, lower_bound, upper_bound)
            x2 = x + min(lmax, 1) * direc1
            fx2 = squeeze(func(x2))

            # 检查外推点是否比当前点更优
            if (fx > fx2):
                t = 2.0*(fx + fx2 - 2.0*fval)
                temp = (fx - fval - delta)
                t *= temp*temp
                temp = fx - fx2
                t -= delta*temp*temp
                # 如果满足一定条件，则进行额外的 Powell 方法的线搜索
                if t < 0.0:
                    fval, x, direc1 = _linesearch_powell(
                        func, x, direc1,
                        tol=xtol * 100,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        fval=fval
                    )
                    # 如果 direc1 非零，则更新 direc 中的方向
                    if np.any(direc1):
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1
        except _MaxFuncCallError:
            # 如果超过最大函数调用次数，则终止优化过程
            break

    # 初始化警告标志
    warnflag = 0
    # 设置成功完成优化的消息
    msg = _status_message['success']
    # 边界越界的警告比超过函数评估次数或迭代次数更为紧急，
    # 但不希望通过更改实现来引起不一致性
    # 如果定义了边界并且当前解超出了指定的上下界
    if bounds and (np.any(lower_bound > x) or np.any(x > upper_bound)):
        # 设置警告标志为4，表示超出边界
        warnflag = 4
        # 设置警告信息为“超出边界”的预定义消息
        msg = _status_message['out_of_bounds']
    
    # 如果函数调用次数超过了最大允许值
    elif fcalls[0] >= maxfun:
        # 设置警告标志为1，表示超过最大函数调用次数
        warnflag = 1
        # 设置警告信息为“超过最大函数调用次数（maxfev）”的预定义消息
        msg = _status_message['maxfev']
    
    # 如果迭代次数超过了最大允许值
    elif iter >= maxiter:
        # 设置警告标志为2，表示超过最大迭代次数
        warnflag = 2
        # 设置警告信息为“超过最大迭代次数（maxiter）”的预定义消息
        msg = _status_message['maxiter']
    
    # 如果目标函数值或当前解中存在NaN值
    elif np.isnan(fval) or np.isnan(x).any():
        # 设置警告标志为3，表示存在NaN值
        warnflag = 3
        # 设置警告信息为“存在NaN值”的预定义消息
        msg = _status_message['nan']

    # 如果显示标志为真
    if disp:
        # 打印成功消息或警告消息，根据警告标志和消息内容，使用RuntimeWarning作为警告类型
        _print_success_message_or_warn(warnflag, msg, RuntimeWarning)
        # 打印当前函数值
        print("         Current function value: %f" % fval)
        # 打印迭代次数
        print("         Iterations: %d" % iter)
        # 打印函数调用次数
        print("         Function evaluations: %d" % fcalls[0])

    # 创建OptimizeResult对象作为最优化结果
    result = OptimizeResult(fun=fval, direc=direc, nit=iter, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x)
    
    # 如果设置了retall标志，则将所有迭代向量存储在结果对象中
    if retall:
        result['allvecs'] = allvecs
    
    # 返回最优化结果对象
    return result
# 定义一个函数 `_endprint`，用于在优化结束后打印消息
def _endprint(x, flag, fval, maxfun, xtol, disp):
    # 如果优化成功完成
    if flag == 0:
        # 如果需要显示详细信息（disp > 1），则打印成功消息和终止标准（使用 xtol）
        if disp > 1:
            print("\nOptimization terminated successfully;\n"
                  "The returned value satisfies the termination criteria\n"
                  "(using xtol = ", xtol, ")")
        return

    # 如果优化中断，根据 flag 的值选择合适的错误消息
    if flag == 1:
        msg = ("\nMaximum number of function evaluations exceeded --- "
               "increase maxfun argument.\n")
    elif flag == 2:
        msg = "\n{}".format(_status_message['nan'])

    # 调用内部函数 _print_success_message_or_warn 打印消息
    _print_success_message_or_warn(flag, msg)
    return


# 定义一个函数 `brute`，用于通过蛮力法在给定范围内最小化一个函数
def brute(func, ranges, args=(), Ns=20, full_output=0, finish=fmin,
          disp=False, workers=1):
    """Minimize a function over a given range by brute force.

    Uses the "brute force" method, i.e., computes the function's value
    at each point of a multidimensional grid of points, to find the global
    minimum of the function.

    The function is evaluated everywhere in the range with the datatype of the
    first call to the function, as enforced by the ``vectorize`` NumPy
    function. The value and type of the function evaluation returned when
    ``full_output=True`` are affected in addition by the ``finish`` argument
    (see Notes).

    The brute force approach is inefficient because the number of grid points
    increases exponentially - the number of grid points to evaluate is
    ``Ns ** len(x)``. Consequently, even with coarse grid spacing, even
    moderately sized problems can take a long time to run, and/or run into
    memory limitations.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the
        form ``f(x, *args)``, where ``x`` is the argument in
        the form of a 1-D array and ``args`` is a tuple of any
        additional fixed parameters needed to completely specify
        the function.
    ranges : tuple
        Each component of the `ranges` tuple must be either a
        "slice object" or a range tuple of the form ``(low, high)``.
        The program uses these to create the grid of points on which
        the objective function will be computed. See `Note 2` for
        more detail.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify
        the function.
    Ns : int, optional
        Number of grid points along the axes, if not otherwise
        specified. See `Note2`.
    full_output : bool, optional
        If True, return the evaluation grid and the objective function's
        values on it.
    finish : callable, optional
        An optimization function that is called with the result of brute force
        minimization as initial guess. `finish` should take `func` and
        the initial guess as positional arguments, and take `args` as
        keyword arguments. It may additionally take `full_output`
        and/or `disp` as keyword arguments. Use None if no "polishing"
        function is to be used. See Notes for more details.
    disp : bool or int, optional
        If True, print messages to stdout. If > 1, print more verbose
        messages.
    workers : int, optional
        Number of parallel workers to use for function evaluation. Not
        supported yet.

    """
    # 函数文档字符串，解释了函数的作用和使用方法
    # 使用蛮力法（brute force）方法最小化给定范围内的函数

    # 调用了 NumPy 的 vectorize 函数，确保函数首次调用时的数据类型一致
    # full_output=True 时返回的评估网格及其上的函数值受 finish 参数影响（详见注释）

    # 蛮力法的效率低下，因为网格点的数量呈指数增长，评估的网格点数为 Ns ** len(x)
    # 因此，即使网格间距较粗，中等大小的问题也可能运行时间很长，或者遇到内存限制

    # 参数说明
    # func: 要最小化的目标函数，必须是形式为 f(x, *args) 的可调用对象
    # ranges: 定义了要计算函数值的多维网格的范围
    # args: func 所需的其他固定参数的元组
    # Ns: 沿轴的网格点数，默认为 20
    # full_output: 如果为 True，返回评估网格及其上的函数值
    # finish: 一个优化函数，用于蛮力最小化的结果作为初始猜测

    # 返回值：无
    # disp 参数控制是否打印消息到标准输出
    # workers 参数用于指定并行工作者数，当前不支持
    pass
    disp : bool, optional
        # 是否打印来自 `finish` 可调用对象的收敛信息的标志位
        Set to True to print convergence messages from the `finish` callable.
    workers : int or map-like callable, optional
        # 并行计算的工作进程数或者可映射的可调用对象
        If `workers` is an int the grid is subdivided into `workers`
        sections and evaluated in parallel (uses
        `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the grid in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        Requires that `func` be pickleable.

        .. versionadded:: 1.3.0

        # 版本新增内容说明
    Returns
    -------
    x0 : ndarray
        # 包含目标函数取得最小值的点的坐标的一维数组
        A 1-D array containing the coordinates of a point at which the
        objective function had its minimum value. (See `Note 1` for
        which point is returned.)
    fval : float
        # 在点 `x0` 处的函数值（当 `full_output` 为 True 时返回）
        Function value at the point `x0`. (Returned when `full_output` is
        True.)
    grid : tuple
        # 评估网格的表示形式，与 `x0` 的长度相同（当 `full_output` 为 True 时返回）
        Representation of the evaluation grid. It has the same
        length as `x0`. (Returned when `full_output` is True.)
    Jout : ndarray
        # 评估网格每个点的函数值，即 ``Jout = func(*grid)`` （当 `full_output` 为 True 时返回）
        Function values at each point of the evaluation
        grid, i.e., ``Jout = func(*grid)``. (Returned
        when `full_output` is True.)

    See Also
    --------
    basinhopping, differential_evolution

    Notes
    -----
    *Note 1*: The program finds the gridpoint at which the lowest value
    of the objective function occurs. If `finish` is None, that is the
    point returned. When the global minimum occurs within (or not very far
    outside) the grid's boundaries, and the grid is fine enough, that
    point will be in the neighborhood of the global minimum.

    However, users often employ some other optimization program to
    "polish" the gridpoint values, i.e., to seek a more precise
    (local) minimum near `brute's` best gridpoint.
    The `brute` function's `finish` option provides a convenient way to do
    that. Any polishing program used must take `brute's` output as its
    initial guess as a positional argument, and take `brute's` input values
    for `args` as keyword arguments, otherwise an error will be raised.
    It may additionally take `full_output` and/or `disp` as keyword arguments.

    `brute` assumes that the `finish` function returns either an
    `OptimizeResult` object or a tuple in the form:
    ``(xmin, Jmin, ... , statuscode)``, where ``xmin`` is the minimizing
    value of the argument, ``Jmin`` is the minimum value of the objective
    function, "..." may be some other returned values (which are not used
    by `brute`), and ``statuscode`` is the status code of the `finish` program.

    Note that when `finish` is not None, the values returned are those
    of the `finish` program, *not* the gridpoint ones. Consequently,
    while `brute` confines its search to the input grid points,
    the `finish` program's results usually will not coincide with any

        # 关于 `brute` 函数的注意事项
    ````
    # 获取范围的长度，即变量的数量
    N = len(ranges)
    # 如果变量数量超过40个，则抛出值错误异常
    if N > 40:
        raise ValueError("Brute Force not possible with more "
                         "than 40 variables.")
    # 将 ranges 转换为列表形式
    lrange = list(ranges)
    # 遍历范围参数列表 lrange 的长度 N
    for k in range(N):
        # 检查每个元素是否为切片对象，若不是则进行处理
        if not isinstance(lrange[k], slice):
            # 若元素不是切片且长度小于3，则将其转换为元组并添加一个复数形式的值
            if len(lrange[k]) < 3:
                lrange[k] = tuple(lrange[k]) + (complex(Ns),)
            # 将非切片对象转换为切片对象
            lrange[k] = slice(*lrange[k])
    
    # 若 N 等于1，则将 lrange 转换为其第一个元素
    if (N == 1):
        lrange = lrange[0]

    # 使用 np.mgrid 创建网格对象 grid
    grid = np.mgrid[lrange]

    # 获取 grid 的形状信息作为输入形状
    inpt_shape = grid.shape
    
    # 若 N 大于1，则将 grid 重新整形为二维数组
    if (N > 1):
        grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

    # 如果 args 不可迭代，则将其转换为元组
    if not np.iterable(args):
        args = (args,)

    # 创建一个函数包装器 _Brute_Wrapper，用于调用 func 函数
    wrapped_func = _Brute_Wrapper(func, args)

    # 使用 MapWrapper 类进行并行迭代
    with MapWrapper(pool=workers) as mapper:
        # 将 wrapped_func 应用于 grid 中的每个元素，并将结果转换为数组 Jout
        Jout = np.array(list(mapper(wrapped_func, grid)))
        
        # 如果 N 等于1，则调整 grid 和 Jout 的形状
        if (N == 1):
            grid = (grid,)
            Jout = np.squeeze(Jout)
        
        # 如果 N 大于1，则重新整形 Jout 和 grid 的形状
        elif (N > 1):
            Jout = np.reshape(Jout, inpt_shape[1:])
            grid = np.reshape(grid.T, inpt_shape)

    # 计算 Jout 的形状
    Nshape = shape(Jout)

    # 找到 Jout 的最小值的索引 indx
    indx = argmin(Jout.ravel(), axis=-1)
    
    # 创建一个长度为 N 的空数组 Nindx 和 xmin
    Nindx = np.empty(N, int)
    xmin = np.empty(N, float)
    
    # 从最后一个维度向前迭代，计算 Nindx 和 xmin
    for k in range(N - 1, -1, -1):
        thisN = Nshape[k]
        Nindx[k] = indx % Nshape[k]
        indx = indx // thisN
    
    # 根据 Nindx 获取最小值时的 grid 值作为 xmin
    for k in range(N):
        xmin[k] = grid[k][tuple(Nindx)]

    # 获取 Jout 在 Nindx 处的最小值作为 Jmin
    Jmin = Jout[tuple(Nindx)]
    
    # 若 N 等于1，则将 grid 和 xmin 转换为其第一个元素
    if (N == 1):
        grid = grid[0]
        xmin = xmin[0]

    # 如果 finish 是可调用对象，则进行最小化运算
    if callable(finish):
        # 获取 finish 函数的参数列表
        finish_args = _getfullargspec(finish).args
        finish_kwargs = dict()
        
        # 如果参数列表中包含 'full_output'，则设置 finish_kwargs 中的 'full_output' 为 1
        if 'full_output' in finish_args:
            finish_kwargs['full_output'] = 1
        
        # 如果参数列表中包含 'disp'，则设置 finish_kwargs 中的 'disp' 为 disp
        if 'disp' in finish_args:
            finish_kwargs['disp'] = disp
        # 如果参数列表中包含 'options'，则将 'disp' 作为 'options' 的一个选项传递
        elif 'options' in finish_args:
            finish_kwargs['options'] = {'disp': disp}

        # 运行优化器 finish 函数，将 func、xmin、args 和 finish_kwargs 作为参数传递
        res = finish(func, xmin, args=args, **finish_kwargs)

        # 根据返回结果的类型，更新 xmin、Jmin 和 success
        if isinstance(res, OptimizeResult):
            xmin = res.x
            Jmin = res.fun
            success = res.success
        else:
            xmin = res[0]
            Jmin = res[1]
            success = res[-1] == 0
        
        # 如果优化过程不成功，并且 disp 为 True，则发出警告
        if not success:
            if disp:
                warnings.warn("Either final optimization did not succeed or `finish` "
                              "does not return `statuscode` as its last argument.",
                              RuntimeWarning, stacklevel=2)

    # 如果需要完整输出，则返回 xmin、Jmin、grid 和 Jout；否则仅返回 xmin
    if full_output:
        return xmin, Jmin, grid, Jout
    else:
        return xmin
class _Brute_Wrapper:
    """
    Object to wrap user cost function for optimize.brute, allowing picklability
    """

    def __init__(self, f, args):
        # Initialize the wrapper object with a cost function and optional arguments
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        # Callable method to evaluate the wrapped function on input x
        # flatten needed for one dimensional case.
        return self.f(np.asarray(x).flatten(), *self.args)


def show_options(solver=None, method=None, disp=True):
    """
    Show documentation for additional options of optimization solvers.

    These are method-specific options that can be supplied through the
    ``options`` dict.

    Parameters
    ----------
    solver : str
        Type of optimization solver. One of 'minimize', 'minimize_scalar',
        'root', 'root_scalar', 'linprog', or 'quadratic_assignment'.
    method : str, optional
        If not given, shows all methods of the specified solver. Otherwise,
        show only the options for the specified method. Valid values
        corresponds to methods' names of respective solver (e.g., 'BFGS' for
        'minimize').
    disp : bool, optional
        Whether to print the result rather than returning it.

    Returns
    -------
    text
        Either None (for disp=True) or the text string (disp=False)

    Notes
    -----
    The solver-specific methods are:

    `scipy.optimize.minimize`

    - :ref:`Nelder-Mead <optimize.minimize-neldermead>`
    - :ref:`Powell      <optimize.minimize-powell>`
    - :ref:`CG          <optimize.minimize-cg>`
    - :ref:`BFGS        <optimize.minimize-bfgs>`
    - :ref:`Newton-CG   <optimize.minimize-newtoncg>`
    - :ref:`L-BFGS-B    <optimize.minimize-lbfgsb>`
    - :ref:`TNC         <optimize.minimize-tnc>`
    - :ref:`COBYLA      <optimize.minimize-cobyla>`
    - :ref:`COBYQA      <optimize.minimize-cobyqa>`
    - :ref:`SLSQP       <optimize.minimize-slsqp>`
    - :ref:`dogleg      <optimize.minimize-dogleg>`
    - :ref:`trust-ncg   <optimize.minimize-trustncg>`

    `scipy.optimize.root`

    - :ref:`hybr              <optimize.root-hybr>`
    - :ref:`lm                <optimize.root-lm>`
    - :ref:`broyden1          <optimize.root-broyden1>`
    - :ref:`broyden2          <optimize.root-broyden2>`
    - :ref:`anderson          <optimize.root-anderson>`
    - :ref:`linearmixing      <optimize.root-linearmixing>`
    - :ref:`diagbroyden       <optimize.root-diagbroyden>`
    - :ref:`excitingmixing    <optimize.root-excitingmixing>`
    - :ref:`krylov            <optimize.root-krylov>`
    - :ref:`df-sane           <optimize.root-dfsane>`

    `scipy.optimize.minimize_scalar`

    - :ref:`brent       <optimize.minimize_scalar-brent>`
    - :ref:`golden      <optimize.minimize_scalar-golden>`
    - :ref:`bounded     <optimize.minimize_scalar-bounded>`

    `scipy.optimize.root_scalar`

    - :ref:`bisect  <optimize.root_scalar-bisect>`
    - :ref:`brentq  <optimize.root_scalar-brentq>`
    - :ref:`brenth  <optimize.root_scalar-brenth>`
    - :ref:`ridder  <optimize.root_scalar-ridder>`
    """
    import textwrap
    # 导入 textwrap 模块，用于格式化文本的缩进和换行

    }

    # 如果未指定 solver，则生成关于所有可用求解器的文档
    if solver is None:
        # 初始化文本列表，用于存储生成的文档内容
        text = ["\n\n\n========\n", "minimize\n", "========\n"]
        # 添加 minimize 求解器的选项文档
        text.append(show_options('minimize', disp=False))
        # 继续添加 minimize_scalar 求解器的选项文档
        text.extend(["\n\n===============\n", "minimize_scalar\n",
                     "===============\n"])
        text.append(show_options('minimize_scalar', disp=False))
        # 添加 root 求解器的选项文档
        text.extend(["\n\n\n====\n", "root\n",
                     "====\n"])
        text.append(show_options('root', disp=False))
        # 添加 linprog 求解器的选项文档
        text.extend(['\n\n\n=======\n', 'linprog\n',
                     '=======\n'])
        text.append(show_options('linprog', disp=False))
        # 将所有文档内容连接成一个字符串
        text = "".join(text)
    else:
        # 将 solver 名称转换为小写
        solver = solver.lower()
        # 如果 solver 不在 doc_routines 中，抛出 ValueError
        if solver not in doc_routines:
            raise ValueError(f'Unknown solver {solver!r}')

        # 如果未指定 method，则生成特定求解器的所有文档
        if method is None:
            text = []
            # 遍历 solver 对应的文档例程
            for name, _ in doc_routines[solver]:
                # 添加求解器名称和分隔线
                text.extend(["\n\n" + name, "\n" + "="*len(name) + "\n\n"])
                # 添加该求解器及其方法的选项文档
                text.append(show_options(solver, name, disp=False))
            # 将所有文档内容连接成一个字符串
            text = "".join(text)
        else:
            # 将 method 名称转换为小写
            method = method.lower()
            # 获取 solver 对应的方法字典
            methods = dict(doc_routines[solver])
            # 如果 method 不在 methods 中，抛出 ValueError
            if method not in methods:
                raise ValueError(f"Unknown method {method!r}")
            # 获取 method 对应的名称
            name = methods[method]

            # 导入函数对象
            parts = name.split('.')
            mod_name = ".".join(parts[:-1])
            __import__(mod_name)
            obj = getattr(sys.modules[mod_name], parts[-1])

            # 获取函数对象的文档字符串
            doc = obj.__doc__
            if doc is not None:
                # 使用 textwrap 模块去除文档字符串的缩进并去除首尾空白
                text = textwrap.dedent(doc).strip()
            else:
                text = ""

    # 如果 disp 为 True，则打印生成的文档并返回
    if disp:
        print(text)
        return
    else:
        return text


如果条件不满足（即不是空字符串），则返回给定的文本。
```