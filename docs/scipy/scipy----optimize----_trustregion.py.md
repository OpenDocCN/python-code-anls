# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion.py`

```
"""Trust-region optimization."""
# 导入必要的库和模块
import math
import warnings

import numpy as np
import scipy.linalg
# 导入优化相关的模块和函数
from ._optimize import (_check_unknown_options, _status_message,
                        OptimizeResult, _prepare_scalar_function,
                        _call_callback_maybe_halt)
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.optimize._differentiable_functions import FD_METHODS
__all__ = []

def _wrap_function(function, args):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    # 定义一个函数包装器，用于计数评估次数并提供参数关键字。
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        # 将 x 的副本发送到用户函数（gh13740）
        return function(np.copy(x), *(wrapper_args + args))

    return ncalls, function_wrapper


class BaseQuadraticSubproblem:
    """
    Base/abstract class defining the quadratic model for trust-region
    minimization. Child classes must implement the ``solve`` method.

    Values of the objective function, Jacobian and Hessian (if provided) at
    the current iterate ``x`` are evaluated on demand and then stored as
    attributes ``fun``, ``jac``, ``hess``.
    """
    
    def __init__(self, x, fun, jac, hess=None, hessp=None):
        # 初始化方法，设置初始值和函数
        self._x = x
        self._f = None
        self._g = None
        self._h = None
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None
        self._fun = fun
        self._jac = jac
        self._hess = hess
        self._hessp = hessp

    def __call__(self, p):
        # 计算当前迭代点的目标函数值
        return self.fun + np.dot(self.jac, p) + 0.5 * np.dot(p, self.hessp(p))

    @property
    def fun(self):
        """Value of objective function at current iteration."""
        # 当前迭代点的目标函数值
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def jac(self):
        """Value of Jacobian of objective function at current iteration."""
        # 当前迭代点的目标函数的雅可比矩阵值
        if self._g is None:
            self._g = self._jac(self._x)
        return self._g

    @property
    def hess(self):
        """Value of Hessian of objective function at current iteration."""
        # 当前迭代点的目标函数的黑塞矩阵值
        if self._h is None:
            self._h = self._hess(self._x)
        return self._h

    def hessp(self, p):
        # 计算黑塞矩阵与向量 p 的乘积
        if self._hessp is not None:
            return self._hessp(self._x, p)
        else:
            return np.dot(self.hess, p)

    @property
    def jac_mag(self):
        """Magnitude of jacobian of objective function at current iteration."""
        # 当前迭代点的目标函数雅可比矩阵的模
        if self._g_mag is None:
            self._g_mag = scipy.linalg.norm(self.jac)
        return self._g_mag
    def get_boundaries_intersections(self, z, d, trust_radius):
        """
        Solve the scalar quadratic equation ``||z + t d|| == trust_radius``.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        # 计算方程的系数
        a = np.dot(d, d)  # a = d · d
        b = 2 * np.dot(z, d)  # b = 2 * z · d
        c = np.dot(z, z) - trust_radius**2  # c = z · z - trust_radius^2
        
        # 计算判别式的平方根
        sqrt_discriminant = math.sqrt(b*b - 4*a*c)
        
        # 通过这种方式计算 t1 和 t2 可以减小舍入误差
        aux = b + math.copysign(sqrt_discriminant, b)
        ta = -aux / (2*a)  # ta = (-b - sqrt_discriminant) / (2*a)
        tb = -2*c / aux    # tb = (-b + sqrt_discriminant) / (2*a)
        
        # 返回排序后的 t 值列表
        return sorted([ta, tb])

    def solve(self, trust_radius):
        raise NotImplementedError('The solve method should be implemented by '
                                  'the child class')
# 定义一个函数 `_minimize_trust_region`，用于使用信赖域算法最小化一个或多个变量的标量函数。

"""
Minimization of scalar function of one or more variables using a
trust-region algorithm.

Options for the trust-region algorithm are:
    initial_trust_radius : float
        Initial trust radius.
    max_trust_radius : float
        Never propose steps that are longer than this value.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than `gtol`
        before successful termination.
    maxiter : int
        Maximum number of iterations to perform.
    disp : bool
        If True, print convergence message.
    inexact : bool
        Accuracy to solve subproblems. If True requires less nonlinear
        iterations, but more vector products. Only effective for method
        trust-krylov.

This function is called by the `minimize` function.
It is not supposed to be called directly.
"""

# 检查是否有未知选项传入
_check_unknown_options(unknown_options)

# 如果未提供 jac 参数，则抛出 ValueError 异常
if jac is None:
    raise ValueError('Jacobian is currently required for trust-region '
                     'methods')
# 如果未提供 hess 和 hessp 参数，则抛出 ValueError 异常
if hess is None and hessp is None:
    raise ValueError('Either the Hessian or the Hessian-vector product '
                     'is currently required for trust-region methods')
# 如果未提供 subproblem 参数，则抛出 ValueError 异常
if subproblem is None:
    raise ValueError('A subproblem solving strategy is required for '
                     'trust-region methods')
# 检查 eta 值是否在有效范围内 (0 <= eta < 0.25)，否则抛出 Exception 异常
if not (0 <= eta < 0.25):
    raise Exception('invalid acceptance stringency')
# 检查 max_trust_radius 是否大于 0，否则抛出 Exception 异常
if max_trust_radius <= 0:
    raise Exception('the max trust radius must be positive')
# 检查 initial_trust_radius 是否大于 0，否则抛出 ValueError 异常
if initial_trust_radius <= 0:
    raise ValueError('the initial trust radius must be positive')
# 检查 initial_trust_radius 是否小于 max_trust_radius，否则抛出 ValueError 异常
if initial_trust_radius >= max_trust_radius:
    raise ValueError('the initial trust radius must be less than the '
                     'max trust radius')

# 强制将初始猜测值 x0 转换为一个平坦的 numpy 数组
x0 = np.asarray(x0).flatten()

# 准备标量函数对象 sf，用于缓存对 fun、jac、hess 的调用
sf = _prepare_scalar_function(fun, x0, jac=jac, hess=hess, args=args)
fun = sf.fun
jac = sf.grad
# 如果 hess 是可调用对象，则更新 hess 为 sf.hess
if callable(hess):
    hess = sf.hess
# 如果 hessp 是可调用对象，则维持原有 hess，这要比检查 hess 是否通过 FD 方法或 HessianUpdateStrategy 更优先
    # 如果 hess 属于 FD_METHODS 集合或者是 HessianUpdateStrategy 类型的实例，则执行以下代码块
    elif (hess in FD_METHODS or isinstance(hess, HessianUpdateStrategy)):
        # 如果 Hessian 是通过有限差分或者 Hessian 更新策略估计得到的，
        # 则 ScalarFunction.hess 返回一个 LinearOperator 或者 HessianUpdateStrategy。
        # 这使得可以计算/创建 hessp。但是，只有在用户没有提供 callable(hessp) 函数时才执行以下操作。
        hess = None

        # 定义 hessp 函数，用于计算 Hessian 矩阵与向量 p 的乘积
        def hessp(x, p, *args):
            return sf.hess(x).dot(p)
    else:
        # 如果既不是有限差分方法，也不是 Hessian 更新策略，则抛出 ValueError 异常
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is currently required for trust-region methods')

    # 对 hessp 函数进行包装，以适应特定的参数
    nhessp, hessp = _wrap_function(hessp, args)

    # 限制迭代次数的最大值
    if maxiter is None:
        maxiter = len(x0)*200

    # 初始化搜索状态警告标志
    warnflag = 0

    # 初始化搜索，设置初始信赖域半径和起始点 x
    trust_radius = initial_trust_radius
    x = x0

    # 如果需要记录所有迭代点，则初始化迭代点列表
    if return_all:
        allvecs = [x]

    # 计算子问题 m，传入初始点 x、目标函数 fun、雅可比矩阵 jac、Hessian 矩阵 hess、hessp 函数
    m = subproblem(x, fun, jac, hess, hessp)
    k = 0

    # 开始搜索目标函数的最小值
    # 如果梯度足够小，则不进行搜索
    while m.jac_mag >= gtol:
        # 当雅可比矩阵的范数大于等于 gtol 时执行循环

        # Solve the sub-problem.
        # 解决子问题。
        # 这给出了相对于当前位置的建议步长
        # 并告诉我们建议步长是否达到了信赖区域边界。
        try:
            p, hits_boundary = m.solve(trust_radius)
            # 尝试求解子问题，返回建议步长 p 和是否达到边界的标志 hits_boundary
        except np.linalg.LinAlgError:
            warnflag = 3
            break
            # 如果出现线性代数错误，则设置警告标志为 3 并退出循环

        # calculate the predicted value at the proposed point
        # 计算在建议点的预测值
        predicted_value = m(p)

        # define the local approximation at the proposed point
        # 定义在建议点的局部近似
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)
        # 使用给定的参数调用 subproblem 函数，得到在建议点的局部近似 m_proposed

        # evaluate the ratio defined in equation (4.4)
        # 计算方程(4.4)中定义的比率
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
            # 如果预测减少小于等于零，则设置警告标志为 2 并退出循环
        rho = actual_reduction / predicted_reduction

        # update the trust radius according to the actual/predicted ratio
        # 根据实际/预测比率更新信赖区域半径
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2*trust_radius, max_trust_radius)
            # 如果 rho 大于 0.75 且 hits_boundary 为真，则将信赖区域半径扩大至两倍，但不超过最大信赖区域半径

        # if the ratio is high enough then accept the proposed step
        # 如果比率足够高，则接受建议的步骤
        if rho > eta:
            x = x_proposed
            m = m_proposed
            # 如果 rho 大于 eta，则更新当前位置和当前局部近似 m

        # append the best guess, call back, increment the iteration count
        # 添加最佳猜测，回调函数，增加迭代计数
        if return_all:
            allvecs.append(np.copy(x))
            # 如果 return_all 为真，则将当前位置的拷贝添加到 allvecs 列表中
        k += 1
        # 增加迭代计数器 k 的值

        intermediate_result = OptimizeResult(x=x, fun=m.fun)
        # 创建一个中间结果对象，包含当前位置和当前局部近似的函数值
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
            # 如果回调函数返回真，则退出循环

        # check if the gradient is small enough to stop
        # 检查梯度是否足够小以停止
        if m.jac_mag < gtol:
            warnflag = 0
            break
            # 如果雅可比矩阵的范数小于 gtol，则设置警告标志为 0 并退出循环

        # check if we have looked at enough iterations
        # 检查是否已经达到足够的迭代次数
        if k >= maxiter:
            warnflag = 1
            break
            # 如果迭代计数器 k 大于等于最大迭代次数 maxiter，则设置警告标志为 1 并退出循环

    # print some stuff if requested
    # 如果需要，打印一些信息
    status_messages = (
            _status_message['success'],
            _status_message['maxiter'],
            'A bad approximation caused failure to predict improvement.',
            'A linalg error occurred, such as a non-psd Hessian.',
            )
    if disp:
        if warnflag == 0:
            print(status_messages[warnflag])
        else:
            warnings.warn(status_messages[warnflag], RuntimeWarning, stacklevel=3)
            # 如果警告标志为 0 以外的值，则发出运行时警告
        print("         Current function value: %f" % m.fun)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Hessian evaluations: %d" % (sf.nhev + nhessp[0]))

    result = OptimizeResult(x=x, success=(warnflag == 0), status=warnflag,
                            fun=m.fun, jac=m.jac, nfev=sf.nfev, njev=sf.ngev,
                            nhev=sf.nhev + nhessp[0], nit=k,
                            message=status_messages[warnflag])
    # 创建最终优化结果对象，包含最优解、成功与否标志、状态码、最终函数值、雅可比矩阵值等信息

    if hess is not None:
        result['hess'] = m.hess
        # 如果存在 Hessian 矩阵，则将其存储在结果对象中
    # 如果 return_all 标志为真，则将 allvecs 添加到结果字典中
    if return_all:
        result['allvecs'] = allvecs

    # 返回处理后的结果字典
    return result
```