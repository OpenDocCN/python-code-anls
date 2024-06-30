# `D:\src\scipysrc\scipy\scipy\optimize\_minpack_py.py`

```
````
import warnings  # 导入 warnings 模块，用于发出警告信息
from . import _minpack  # 从当前包导入 _minpack 模块

import numpy as np  # 导入 numpy 模块，并重命名为 np
from numpy import (atleast_1d, triu, shape, transpose, zeros, prod, greater,
                   asarray, inf, finfo, inexact, issubdtype, dtype)  # 从 numpy 模块导入多个函数和常量
from scipy import linalg  # 导入 scipy.linalg 模块
from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError  # 从 scipy.linalg 导入多个线性代数函数和异常
from scipy._lib._util import _asarray_validated, _lazywhere, _contains_nan  # 从 scipy._lib._util 导入多个工具函数
from scipy._lib._util import getfullargspec_no_self as _getfullargspec  # 从 scipy._lib._util 导入 getfullargspec_no_self 并重命名为 _getfullargspec
from ._optimize import OptimizeResult, _check_unknown_options, OptimizeWarning  # 从当前包导入 OptimizeResult, _check_unknown_options, 和 OptimizeWarning
from ._lsq import least_squares  # 从当前包导入 least_squares 函数
# from ._lsq.common import make_strictly_feasible  # 可能用于导入 make_strictly_feasible 函数（已注释）
from ._lsq.least_squares import prepare_bounds  # 从当前包导入 prepare_bounds 函数
from scipy.optimize._minimize import Bounds  # 从 scipy.optimize._minimize 导入 Bounds

__all__ = ['fsolve', 'leastsq', 'fixed_point', 'curve_fit']  # 定义模块的公共接口，包含 fsolve, leastsq, fixed_point, 和 curve_fit

def _check_func(checker, argname, thefunc, x0, args, numinputs, output_shape=None):
    # 检查函数的输出，验证其形状和数据类型
    res = atleast_1d(thefunc(*((x0[:numinputs],) + args)))  # 调用函数并将结果转换为至少一维数组
    if (output_shape is not None) and (shape(res) != output_shape):  # 如果指定了输出形状，检查结果形状是否匹配
        if (output_shape[0] != 1):
            if len(output_shape) > 1:
                if output_shape[1] == 1:
                    return shape(res)
            msg = f"{checker}: there is a mismatch between the input and output " \
                  f"shape of the '{argname}' argument"  # 构造错误信息
            func_name = getattr(thefunc, '__name__', None)  # 获取函数名
            if func_name:
                msg += " '%s'." % func_name  # 添加函数名到错误信息
            else:
                msg += "."  # 添加句点到错误信息
            msg += f'Shape should be {output_shape} but it is {shape(res)}.'  # 添加期望和实际形状信息
            raise TypeError(msg)  # 抛出类型错误
    if issubdtype(res.dtype, inexact):  # 如果结果数据类型是近似类型
        dt = res.dtype  # 设置数据类型为结果数据类型
    else:
        dt = dtype(float)  # 否则，设置数据类型为 float
    return shape(res), dt  # 返回结果的形状和数据类型

def fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-8, maxfev=0, band=None,
           epsfcn=None, factor=100, diag=None):
    """
    Find the roots of a function.

    Return the roots of the (non-linear) equations defined by
    ``func(x) = 0`` given a starting estimate.

    Parameters
    ----------
    func : callable ``f(x, *args)``
        A function that takes at least one (possibly vector) argument,
        and returns a value of the same length.
    x0 : ndarray
        The starting estimate for the roots of ``func(x) = 0``.
    args : tuple, optional
        Any extra arguments to `func`.
    fprime : callable ``f(x, *args)``, optional
        A function to compute the Jacobian of `func` with derivatives
        across the rows. By default, the Jacobian will be estimated.
    full_output : bool, optional
        If True, return optional outputs.
    col_deriv : bool, optional
        Specify whether the Jacobian function computes derivatives down
        the columns (faster, because there is no transpose operation).
    xtol : float, optional
        The calculation will terminate if the relative error between two
        consecutive iterates is at most `xtol`.
    # maxfev: int, optional
    #     最大函数调用次数。如果为零，则最大调用次数为 ``100*(N+1)``，其中 N 是 x0 的元素数量。
    # band: tuple, optional
    #     如果设置为包含 Jacobi 矩阵带中次对角线和超对角线数目的二元序列，则考虑带状 Jacobi 矩阵（仅适用于 ``fprime=None``）。
    # epsfcn: float, optional
    #     Jacobian 前向差分逼近的合适步长（对于 ``fprime=None``）。如果 `epsfcn` 小于机器精度，则假定函数的相对误差是机器精度的数量级。
    # factor: float, optional
    #     初始化步长界限的参数（``factor * || diag * x||``）。应在区间 ``(0.1, 100)`` 内。
    # diag: sequence, optional
    #     N 个正数条目，作为变量的缩放因子。

    # 返回值
    # -------
    # x: ndarray
    #     解（或最后一次迭代的结果，如果调用不成功）。
    # infodict: dict
    #     包含以下可选输出的字典，键如下：

    #     ``nfev``
    #         函数调用次数
    #     ``njev``
    #         Jacobian 矩阵调用次数
    #     ``fvec``
    #         输出处的函数值
    #     ``fjac``
    #         最终近似 Jacobian 矩阵的 QR 分解产生的正交矩阵 q，按列存储
    #     ``r``
    #         QR 分解相同矩阵的上三角矩阵
    #     ``qtf``
    #         向量 ``(transpose(q) * fvec)``

    # ier: int
    #     整数标志。如果找到解，则设置为 1，否则参考 `mesg` 获取更多信息。
    # mesg: str
    #     如果未找到解，则 `mesg` 详细说明失败的原因。

    # 参见
    # --------
    # root: 用于多变量函数根查找算法的接口。特别是看看 ``method='hybr'``。

    # 注意
    # -----
    # ``fsolve`` 是 MINPACK 的 hybrd 和 hybrj 算法的包装器。

    # 示例
    # --------
    # 找到以下方程组的解：
    # ``x0*cos(x1) = 4,  x1*x0 - x1 = 5``。
    #
    # >>> import numpy as np
    # >>> from scipy.optimize import fsolve
    # >>> def func(x):
    # ...     return [x[0] * np.cos(x[1]) - 4,
    # ...             x[1] * x[0] - x[1] - 5]
    # >>> root = fsolve(func, [1, 1])
    # >>> root
    # array([6.50409711, 0.90841421])
    # >>> np.isclose(func(root), [0.0, 0.0])  # func(root) 应接近 0.0。
    # array([ True,  True])

    def _wrapped_func(*fargs):
        """
        封装 `func` 函数，跟踪函数调用的次数。
        """
        _wrapped_func.nfev += 1
        return func(*fargs)

    _wrapped_func.nfev = 0
    # 构建包含选项参数的字典，用于传递给求根混合算法
    options = {'col_deriv': col_deriv,  # 是否使用列方向导数
               'xtol': xtol,            # 允许的最小误差
               'maxfev': maxfev,        # 允许的最大函数调用次数
               'band': band,            # 牛顿法中带状矩阵的半宽度
               'eps': epsfcn,           # 用于数值微分的步长
               'factor': factor,        # 用于确定初始步长的因子
               'diag': diag}            # 求根算法使用的对角元素

    # 调用求根混合算法，传递包装后的函数、初始值、额外参数、雅可比矩阵和选项字典
    res = _root_hybr(_wrapped_func, x0, args, jac=fprime, **options)
    # 将内部包装函数的函数调用次数设置到结果对象中
    res.nfev = _wrapped_func.nfev

    # 如果需要完整输出
    if full_output:
        # 提取结果中的解向量 x
        x = res['x']
        # 从结果中提取指定的信息，并组成字典
        info = {k: res.get(k)
                    for k in ('nfev', 'njev', 'fjac', 'r', 'qtf') if k in res}
        # 将函数值向量 fvec 添加到信息字典中
        info['fvec'] = res['fun']
        # 返回解向量 x、信息字典、状态和消息
        return x, info, res['status'], res['message']
    else:
        # 否则，只需要返回解向量 x
        status = res['status']
        msg = res['message']
        # 根据状态处理消息
        if status == 0:
            # 如果状态为 0，抛出类型错误异常
            raise TypeError(msg)
        elif status == 1:
            # 如果状态为 1，什么也不做
            pass
        elif status in [2, 3, 4, 5]:
            # 如果状态为 2、3、4 或 5，发出运行时警告
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            # 否则，抛出类型错误异常
            raise TypeError(msg)
        # 返回解向量 x
        return res['x']
def _root_hybr(func, x0, args=(), jac=None,
               col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, eps=None,
               factor=100, diag=None, **unknown_options):
    """
    Find the roots of a multivariate function using MINPACK's hybrd and
    hybrj routines (modified Powell method).

    Options
    -------
    col_deriv : bool
        Specify whether the Jacobian function computes derivatives down
        the columns (faster, because there is no transpose operation).
    xtol : float
        The calculation will terminate if the relative error between two
        consecutive iterates is at most `xtol`.
    maxfev : int
        The maximum number of calls to the function. If zero, then
        ``100*(N+1)`` is the maximum where N is the number of elements
        in `x0`.
    band : tuple
        If set to a two-sequence containing the number of sub- and
        super-diagonals within the band of the Jacobi matrix, the
        Jacobi matrix is considered banded (only for ``fprime=None``).
    eps : float
        A suitable step length for the forward-difference
        approximation of the Jacobian (for ``fprime=None``). If
        `eps` is less than the machine precision, it is assumed
        that the relative errors in the functions are of the order of
        the machine precision.
    factor : float
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in the interval
        ``(0.1, 100)``.
    diag : sequence
        N positive entries that serve as scale factors for the
        variables.

    """
    # 检查未知选项，确保没有未定义的选项传递进来
    _check_unknown_options(unknown_options)
    # 将 x0 转换为一维数组
    x0 = asarray(x0).flatten()
    # 获取 x0 的长度，即变量的数量
    n = len(x0)
    # 如果 args 不是元组，则转换为元组
    if not isinstance(args, tuple):
        args = (args,)
    # 检查函数 func 的输入输出，并返回其形状和数据类型
    shape, dtype = _check_func('fsolve', 'func', func, x0, args, n, (n,))
    # 如果 epsfcn 为 None，则设置为 dtype 的机器精度
    epsfcn = eps
    if epsfcn is None:
        epsfcn = finfo(dtype).eps
    # 设置 Dfun 为 jac
    Dfun = jac
    # 如果 Dfun 为 None，则进行以下处理
    if Dfun is None:
        # 如果 band 为 None，则 ml 和 mu 设置为 -10
        if band is None:
            ml, mu = -10, -10
        else:
            # 否则，从 band 中获取 ml 和 mu
            ml, mu = band[:2]
        # 如果 maxfev 为 0，则设置为默认值 200*(n+1)
        if maxfev == 0:
            maxfev = 200 * (n + 1)
        # 调用 _minpack._hybrd 函数进行求解
        retval = _minpack._hybrd(func, x0, args, 1, xtol, maxfev,
                                 ml, mu, epsfcn, factor, diag)
    else:
        # 检查函数 Dfun 的输入输出，并返回其形状和数据类型
        _check_func('fsolve', 'fprime', Dfun, x0, args, n, (n, n))
        # 如果 maxfev 为 0，则设置为默认值 100*(n+1)
        if (maxfev == 0):
            maxfev = 100 * (n + 1)
        # 调用 _minpack._hybrj 函数进行求解
        retval = _minpack._hybrj(func, Dfun, x0, args, 1,
                                 col_deriv, xtol, maxfev, factor, diag)

    # 获取返回值中的解 x 和状态 status
    x, status = retval[0], retval[-1]
    # 定义一个字典，用于存储不同状态下的错误信息
    errors = {
        0: "Improper input parameters were entered.",  # 状态0：输入参数不正确
        1: "The solution converged.",  # 状态1：求解收敛
        2: "The number of calls to function has reached maxfev = %d." % maxfev,  
           # 状态2：函数调用次数达到最大值 maxfev = %d
        3: "xtol=%f is too small, no further improvement in the approximate\n  solution is possible." % xtol,  
           # 状态3：xtol=%f 过小，无法进一步改善近似解
        4: "The iteration is not making good progress, as measured by the \n  improvement from the last five Jacobian evaluations.",  
           # 状态4：迭代进展不佳，根据最近五次雅可比矩阵评估改进情况
        5: "The iteration is not making good progress, as measured by the \n  improvement from the last ten iterations.",  
           # 状态5：迭代进展不佳，根据最近十次迭代改进情况
        'unknown': "An error occurred."  # 未知状态：发生了错误
    }
    
    # 从 retval 变量的第二个元素中获取信息
    info = retval[1]
    # 将 'fvec' 键改名为 'fun'
    info['fun'] = info.pop('fvec')
    # 创建一个 OptimizeResult 对象 sol，包括解 x、成功状态 success、求解状态 status、优化方法 "hybr"
    sol = OptimizeResult(x=x, success=(status == 1), status=status,
                         method="hybr")
    # 更新 sol 对象的信息
    sol.update(info)
    # 尝试获取对应 status 状态的错误信息并赋给 sol 的 'message' 键
    try:
        sol['message'] = errors[status]
    except KeyError:
        sol['message'] = errors['unknown']
    
    # 返回最终的解 sol
    return sol
# 定义成功情况下的最小二乘法迭代步骤
LEASTSQ_SUCCESS = [1, 2, 3, 4]
# 定义失败情况下的最小二乘法迭代步骤
LEASTSQ_FAILURE = [5, 6, 7, 8]

# 定义最小二乘法函数，用于最小化一组方程的平方和
def leastsq(func, x0, args=(), Dfun=None, full_output=False,
            col_deriv=False, ftol=1.49012e-8, xtol=1.49012e-8,
            gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    """
    Minimize the sum of squares of a set of equations.

    ::

        x = arg min(sum(func(y)**2,axis=0))
                 y

    Parameters
    ----------
    func : callable
        Should take at least one (possibly length ``N`` vector) argument and
        returns ``M`` floating point numbers. It must not return NaNs or
        fitting might fail. ``M`` must be greater than or equal to ``N``.
    x0 : ndarray
        The starting estimate for the minimization.
    args : tuple, optional
        Any extra arguments to func are placed in this tuple.
    Dfun : callable, optional
        A function or method to compute the Jacobian of func with derivatives
        across the rows. If this is None, the Jacobian will be estimated.
    full_output : bool, optional
        If ``True``, return all optional outputs (not just `x` and `ier`).
    col_deriv : bool, optional
        If ``True``, specify that the Jacobian function computes derivatives
        down the columns (faster, because there is no transpose operation).
    ftol : float, optional
        Relative error desired in the sum of squares.
    xtol : float, optional
        Relative error desired in the approximate solution.
    gtol : float, optional
        Orthogonality desired between the function vector and the columns of
        the Jacobian.
    maxfev : int, optional
        The maximum number of calls to the function. If `Dfun` is provided,
        then the default `maxfev` is 100*(N+1) where N is the number of elements
        in x0, otherwise the default `maxfev` is 200*(N+1).
    epsfcn : float, optional
        A variable used in determining a suitable step length for the forward-
        difference approximation of the Jacobian (for Dfun=None).
        Normally the actual step length will be sqrt(epsfcn)*x
        If epsfcn is less than the machine precision, it is assumed that the
        relative errors are of the order of the machine precision.
    factor : float, optional
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
    diag : sequence, optional
        N positive entries that serve as a scale factors for the variables.

    Returns
    -------
    x : ndarray
        The solution (or the result of the last iteration for an unsuccessful
        call).
    """
    # 将参数 x0 转换为 ndarray 并展平成一维数组
    x0 = asarray(x0).flatten()
    # 获取数组 x0 的长度，即参数个数
    n = len(x0)
    # 如果参数 args 不是元组，则将其转换为元组类型
    if not isinstance(args, tuple):
        args = (args,)
    # 检查函数 func 的返回值类型和形状，并返回形状和数据类型
    shape, dtype = _check_func('leastsq', 'func', func, x0, args, n)
    # 获取目标函数的行数，即数据点的数量
    m = shape[0]
    # 如果输入向量长度 N 大于输出向量长度 M，则抛出类型错误异常
    if n > m:
        raise TypeError(f"Improper input: func input vector length N={n} must"
                        f" not exceed func output vector length M={m}")

    # 如果 epsfcn 未指定，则设置为当前数据类型的机器精度
    if epsfcn is None:
        epsfcn = finfo(dtype).eps

    # 如果没有提供导数函数 Dfun，则根据条件设置 maxfev 的默认值
    if Dfun is None:
        # 如果 maxfev 为 0，则设置为默认值 200*(n + 1)
        if maxfev == 0:
            maxfev = 200*(n + 1)
        # 调用最小包 _minpack 的 _lmdif 函数进行非线性最小二乘拟合
        retval = _minpack._lmdif(func, x0, args, full_output, ftol, xtol,
                                 gtol, maxfev, epsfcn, factor, diag)
    else:
        # 如果需要考虑列优先导数计算，则检查函数 Dfun 的有效性
        if col_deriv:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (n, m))
        else:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (m, n))
        # 如果 maxfev 为 0，则设置为默认值 100*(n + 1)
        if maxfev == 0:
            maxfev = 100 * (n + 1)
        # 调用最小包 _minpack 的 _lmder 函数进行导数支持的非线性最小二乘拟合
        retval = _minpack._lmder(func, Dfun, x0, args, full_output,
                                 col_deriv, ftol, xtol, gtol, maxfev,
                                 factor, diag)

    # 错误字典，用于处理返回值不为零时的异常信息和类型
    errors = {0: ["Improper input parameters.", TypeError],
              1: ["Both actual and predicted relative reductions "
                  "in the sum of squares\n  are at most %f" % ftol, None],
              2: ["The relative error between two consecutive "
                  "iterates is at most %f" % xtol, None],
              3: ["Both actual and predicted relative reductions in "
                  f"the sum of squares\n  are at most {ftol:f} and the "
                  "relative error between two consecutive "
                  f"iterates is at \n  most {xtol:f}", None],
              4: ["The cosine of the angle between func(x) and any "
                  "column of the\n  Jacobian is at most %f in "
                  "absolute value" % gtol, None],
              5: ["Number of calls to function has reached "
                  "maxfev = %d." % maxfev, ValueError],
              6: ["ftol=%f is too small, no further reduction "
                  "in the sum of squares\n  is possible." % ftol,
                  ValueError],
              7: ["xtol=%f is too small, no further improvement in "
                  "the approximate\n  solution is possible." % xtol,
                  ValueError],
              8: ["gtol=%f is too small, func(x) is orthogonal to the "
                  "columns of\n  the Jacobian to machine "
                  "precision." % gtol, ValueError]}

    # 获取 FORTRAN 返回的信息值，其值范围为 0 到 8
    info = retval[-1]
    # 如果需要完整输出结果
    if full_output:
        # 初始化协方差矩阵为 None
        cov_x = None
        # 如果优化信息在成功的列表中
        if info in LEASTSQ_SUCCESS:
            # 获取 IPVT 数组，并将其元素减一（从1-based调整为0-based）
            perm = retval[1]['ipvt'] - 1
            # 获取 IPVT 的长度
            n = len(perm)
            # 提取并上三角化转置的雅可比矩阵的前 n 行
            r = triu(transpose(retval[1]['fjac'])[:n, :])
            # 获取用于计算三角矩阵的 LAPACK 函数
            inv_triu = linalg.get_lapack_funcs('trtri', (r,))
            try:
                # 计算上三角矩阵的逆以及返回的信息
                invR, trtri_info = inv_triu(r)  # 默认：上三角，非单位对角线
                # 如果 trtri 返回的信息不为 0，则抛出线性代数错误
                if trtri_info != 0:  # 明确比较以提高可读性
                    raise LinAlgError(f'trtri returned info {trtri_info}')
                # 将逆矩阵重新映射到原始顺序
                invR[perm] = invR.copy()
                # 计算协方差矩阵
                cov_x = invR @ invR.T
            except (LinAlgError, ValueError):
                pass
        # 返回完整输出结果，包括协方差矩阵和其它返回值
        return (retval[0], cov_x) + retval[1:-1] + (errors[info][0], info)
    else:
        # 如果不需要完整输出结果
        if info in LEASTSQ_FAILURE:
            # 在失败的情况下，发出运行时警告
            warnings.warn(errors[info][0], RuntimeWarning, stacklevel=2)
        elif info == 0:
            # 如果优化返回值为 0，抛出相应的优化错误
            raise errors[info][1](errors[info][0])
        # 返回简化的输出结果
        return retval[0], info
# 定义一个轻量级的记忆化函数装饰器，用于解决 gh-13670 的问题：仅记住第一组参数和对应的函数值，并仅尝试使用它们两次（函数在 x0 处评估的次数）。
def _lightweight_memoizer(f):
    def _memoized_func(params):
        # 如果 skip_lookup 为 True，则直接调用原始函数 f，并返回结果
        if _memoized_func.skip_lookup:
            return f(params)

        # 如果上一次调用的参数与当前参数相同，则直接返回上次计算的结果
        if np.all(_memoized_func.last_params == params):
            return _memoized_func.last_val
        # 如果不是第一次调用且参数不同，则设置 skip_lookup 为 True，以后直接调用原始函数 f
        elif _memoized_func.last_params is not None:
            _memoized_func.skip_lookup = True

        # 计算函数 f 在参数 params 处的值
        val = f(params)

        # 如果是第一次调用，则记录当前参数和计算结果
        if _memoized_func.last_params is None:
            _memoized_func.last_params = np.copy(params)
            _memoized_func.last_val = val

        return val

    # 初始化记忆化函数的状态变量
    _memoized_func.last_params = None
    _memoized_func.last_val = None
    _memoized_func.skip_lookup = False
    return _memoized_func


def _wrap_func(func, xdata, ydata, transform):
    # 如果 transform 为 None，则定义一个包装函数 func_wrapped，用于返回 func(xdata, *params) - ydata
    if transform is None:
        def func_wrapped(params):
            return func(xdata, *params) - ydata
    # 如果 transform 是标量或一维数组，则定义一个包装函数 func_wrapped，用于返回 transform * (func(xdata, *params) - ydata)
    elif transform.size == 1 or transform.ndim == 1:
        def func_wrapped(params):
            return transform * (func(xdata, *params) - ydata)
    else:
        # 如果 transform 是二维数组，则定义一个包装函数 func_wrapped，用于返回 solve_triangular(transform, func(xdata, *params) - ydata, lower=True)
        # 这里解释了使用线性变换 transform 进行 Chisq 计算的过程
        def func_wrapped(params):
            return solve_triangular(transform, func(xdata, *params) - ydata, lower=True)
    return func_wrapped


def _wrap_jac(jac, xdata, transform):
    # 如果 transform 为 None，则定义一个包装函数 jac_wrapped，用于返回 jac(xdata, *params)
    if transform is None:
        def jac_wrapped(params):
            return jac(xdata, *params)
    # 如果 transform 是一维数组，则定义一个包装函数 jac_wrapped，用于返回 transform[:, np.newaxis] * np.asarray(jac(xdata, *params))
    elif transform.ndim == 1:
        def jac_wrapped(params):
            return transform[:, np.newaxis] * np.asarray(jac(xdata, *params))
    else:
        # 如果 transform 是二维数组，则定义一个包装函数 jac_wrapped，用于返回 solve_triangular(transform, np.asarray(jac(xdata, *params)), lower=True)
        # 这里解释了使用线性变换 transform 进行雅可比矩阵的调整的过程
        def jac_wrapped(params):
            return solve_triangular(transform, np.asarray(jac(xdata, *params)), lower=True)
    return jac_wrapped


def _initialize_feasible(lb, ub):
    # 初始化参数 p0 为和 lb 具有相同形状的全为 1 的数组
    p0 = np.ones_like(lb)
    # 找到 lb 和 ub 都是有限值的位置，并将 p0 对应位置设置为 lb 和 ub 的中点
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)
    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])

    # 找到 lb 有限而 ub 无限的位置，并将 p0 对应位置设置为 lb 加 1
    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1

    # 找到 lb 无限而 ub 有限的位置，并将 p0 对应位置设置为 ub 减 1
    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1

    return p0


def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
              check_finite=None, bounds=(-np.inf, np.inf), method=None,
              jac=None, *, full_output=False, nan_policy=None,
              **kwargs):
    """
    Use non-linear least squares to fit a function, f, to data.

    Assumes ``ydata = f(xdata, *params) + eps``.

    Parameters
    ----------
    """
    # 定义模型函数，接受独立变量作为第一个参数，接下来的参数是要拟合的参数
    f : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    
    # 独立变量的数据，即数据测量时使用的自变量
    xdata : array_like
        The independent variable where the data is measured.
        Should usually be an M-length sequence or an (k,M)-shaped array for
        functions with k predictors, and each element should be float
        convertible if it is an array like object.
    
    # 依赖数据，一个长度为 M 的数组，通常是 f(xdata, ...) 的结果
    ydata : array_like
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
    
    # 参数的初始猜测值，长度为 N
    p0 : array_like, optional
        Initial guess for the parameters (length N). If None, then the
        initial values will all be 1 (if the number of parameters for the
        function can be determined using introspection, otherwise a
        ValueError is raised).
    
    # 确定 `ydata` 中的不确定性，影响拟合的结果
    sigma : None or scalar or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:
        
            - A scalar or 1-D `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.
              
            - A 2-D `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.
              
              .. versionadded:: 0.19
        
        None (default) is equivalent of 1-D `sigma` filled with ones.
    
    # 如果为 True，则 `sigma` 表示绝对误差，`pcov` 反映这些绝对值
    # 如果为 False（默认），则 `sigma` 只有相对大小是重要的
    # 返回的参数协方差矩阵 `pcov` 基于将 `sigma` 缩放一个常数因子
    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.
        
        If False (default), only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit. Default is False.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    
    # 如果为 True，则检查输入数组中是否包含 NaN 或 inf
    # 如果包含，则引发 ValueError；如果未显式指定 `nan_policy`，默认为 True
    check_finite : bool, optional
        If True, check that the input arrays do not contain nans or infs,
        and raise a ValueError if they do. Setting this parameter to
        False may silently produce nonsensical results if the input arrays
        do contain nans. Default is True if `nan_policy` is not specified
        explicitly and False otherwise.
    bounds : 2-tuple of array_like or `Bounds`, optional
        # 参数的下界和上界。可以是数组形式的二元组或者 `Bounds` 类的实例。默认情况下没有边界限制。
        There are two ways to specify the bounds:

            - Instance of `Bounds` class.

            - 2-tuple of array_like: Each element of the tuple must be either
              an array with the length equal to the number of parameters, or a
              scalar (in which case the bound is taken to be the same for all
              parameters). Use ``np.inf`` with an appropriate sign to disable
              bounds on all or some parameters.

    method : {'lm', 'trf', 'dogbox'}, optional
        # 优化方法选择。详见 `least_squares` 获取更多细节。默认是 'lm'，适用于无约束问题；如果提供了 `bounds`，则默认为 'trf'。
        # 当观测值数量少于变量数量时，'lm' 方法将不起作用，请使用 'trf' 或 'dogbox' 方法。
        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
        provided. The method 'lm' won't work when the number of observations
        is less than the number of variables, use 'trf' or 'dogbox' in this
        case.

        .. versionadded:: 0.17
    jac : callable, string or None, optional
        # 计算模型函数相对于参数的雅可比矩阵的函数。签名为 ``jac(x, ...)``。结果会根据提供的 `sigma` 进行缩放。
        # 如果为 None（默认），则将通过数值方法估算雅可比矩阵。
        # 对于 'trf' 和 'dogbox' 方法，可以使用字符串关键字来选择有限差分方案，详见 `least_squares`。

        .. versionadded:: 0.18
    full_output : boolean, optional
        # 如果为 True，则函数会返回额外的信息：`infodict`、`mesg` 和 `ier`。
        .. versionadded:: 1.9
    nan_policy : {'raise', 'omit', None}, optional
        # 定义如何处理输入中包含 NaN 的情况。
        # 可用选项如下（默认为 None）：
        # * 'raise': 抛出错误
        # * 'omit': 忽略包含 NaN 的值进行计算
        # * None: 不对 NaN 进行特殊处理（除了 `check_finite` 外）；当存在 NaN 时的行为依赖于具体实现，可能会变化。
        # 如果显式指定了此值（不是 None），则 `check_finite` 将被设置为 False。

        .. versionadded:: 1.11
    **kwargs
        # 传递给 `leastsq`（对于 ``method='lm'``）或者 `least_squares`（其他情况）的关键字参数。
    Returns
    -------
    popt : array
        # 最优参数值，使得 ``f(xdata, *popt) - ydata`` 的平方残差和最小化。
    pcov : 2-D array
        The estimated approximate covariance of popt. The diagonals provide
        the variance of the parameter estimate. To compute one standard
        deviation errors on the parameters, use
        ``perr = np.sqrt(np.diag(pcov))``. Note that the relationship between
        `cov` and parameter error estimates is derived based on a linear
        approximation to the model function around the optimum [1].
        When this approximation becomes inaccurate, `cov` may not provide an
        accurate measure of uncertainty.

        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.

        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix. Covariance matrices with large condition numbers
        (e.g. computed with `numpy.linalg.cond`) may indicate that results are
        unreliable.
    RuntimeError
        if the least-squares minimization fails.

    OptimizeWarning
        if covariance of the parameters can not be estimated.


    See Also
    --------
    least_squares : Minimize the sum of squares of nonlinear functions.
    scipy.stats.linregress : Calculate a linear least squares regression for
                             two sets of measurements.


    Notes
    -----
    Users should ensure that inputs `xdata`, `ydata`, and the output of `f`
    are ``float64``, or else the optimization may return incorrect results.


    With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm
    through `leastsq`. Note that this algorithm can only deal with
    unconstrained problems.


    Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to
    the docstring of `least_squares` for more information.


    Parameters to be fitted must have similar scale. Differences of multiple
    orders of magnitude can lead to incorrect results. For the 'trf' and
    'dogbox' methods, the `x_scale` keyword argument can be used to scale
    the parameters.


    References
    ----------
    [1] K. Vugrin et al. Confidence region estimation techniques for nonlinear
        regression in groundwater flow: Three case studies. Water Resources
        Research, Vol. 43, W03423, :doi:`10.1029/2005WR004804`


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.optimize import curve_fit


    >>> def func(x, a, b, c):
    ...     return a * np.exp(-b * x) + c


    Define the data to be fit with some noise:


    >>> xdata = np.linspace(0, 4, 50)
    >>> y = func(xdata, 2.5, 1.3, 0.5)
    >>> rng = np.random.default_rng()
    >>> y_noise = 0.2 * rng.normal(size=xdata.size)
    >>> ydata = y + y_noise
    >>> plt.plot(xdata, ydata, 'b-', label='data')


    Fit for the parameters a, b, c of the function `func`:


    >>> popt, pcov = curve_fit(func, xdata, ydata)
    >>> popt
    array([2.56274217, 1.37268521, 0.47427475])
    >>> plt.plot(xdata, func(xdata, *popt), 'r-',
    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


    Constrain the optimization to the region of ``0 <= a <= 3``,
    ``0 <= b <= 1`` and ``0 <= c <= 0.5``:


    >>> popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
    >>> popt
    array([2.43736712, 1.        , 0.34463856])
    >>> plt.plot(xdata, func(xdata, *popt), 'g--',
    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.legend()
    >>> plt.show()


    For reliable results, the model `func` should not be overparametrized;
    redundant parameters can cause unreliable covariance matrices and, in some
    cases, poorer quality fits. As a quick check of whether the model may be
    overparameterized, calculate the condition number of the covariance matrix:


    >>> np.linalg.cond(pcov)
    34.571092161547405  # may vary
    if p0 is None:
        # 如果初始参数 p0 为 None，则通过检查函数签名来确定参数数量
        sig = _getfullargspec(f)
        args = sig.args
        # 如果函数参数少于 2 个，抛出数值错误异常
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        # 计算参数数量，减去一个用于自变量 x
        n = len(args) - 1
    else:
        # 如果 p0 不为 None，则至少转换为 1 维数组，获取其大小作为参数数量
        p0 = np.atleast_1d(p0)
        n = p0.size

    # 如果 bounds 是 Bounds 类型的实例，则分别获取下界 lb 和上界 ub
    if isinstance(bounds, Bounds):
        lb, ub = bounds.lb, bounds.ub
    else:
        # 否则，准备边界
        lb, ub = prepare_bounds(bounds, n)

    # 如果 p0 为 None，则使用 _initialize_feasible 函数初始化可行的初始参数
    if p0 is None:
        p0 = _initialize_feasible(lb, ub)

    # 检查是否存在任何边界问题
    bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))

    # 如果未指定优化方法，则根据是否存在边界问题选择默认方法
    if method is None:
        if bounded_problem:
            method = 'trf'
        else:
            method = 'lm'

    # 如果选择的优化方法是 'lm' 且存在边界问题，则引发数值错误异常
    if method == 'lm' and bounded_problem:
        raise ValueError("Method 'lm' only works for unconstrained problems. "
                         "Use 'trf' or 'dogbox' instead.")

    # 如果未指定 check_finite 参数，则根据 nan_policy 来确定其值
    if check_finite is None:
        check_finite = True if nan_policy is None else False

    # 如果 check_finite 为 True，则将 ydata 转换为 float 类型的数组并检查其有限性
    # 否则，仅将 ydata 转换为 float 类型的数组
    if check_finite:
        ydata = np.asarray_chkfinite(ydata, float)
    else:
        ydata = np.asarray(ydata, float)
    # 检查 `xdata` 是否为列表、元组或者 NumPy 数组
    if isinstance(xdata, (list, tuple, np.ndarray)):
        # 如果 `check_finite` 为 True，则确保 `xdata` 中的元素为有限的浮点数数组
        # 否则，只将 `xdata` 转换为浮点数数组
        if check_finite:
            xdata = np.asarray_chkfinite(xdata, float)
        else:
            xdata = np.asarray(xdata, float)

    # 检查 `ydata` 的大小是否为 0，如果是则抛出 ValueError 异常
    if ydata.size == 0:
        raise ValueError("`ydata` must not be empty!")

    # 只有在 `check_finite` 为 False 且 `nan_policy` 不为 None 时才需要处理 NaN 值
    if not check_finite and nan_policy is not None:
        # 如果 `nan_policy` 为 "propagate"，则抛出 ValueError 异常
        if nan_policy == "propagate":
            raise ValueError("`nan_policy='propagate'` is not supported "
                             "by this function.")

        # 定义可能的 NaN 处理策略，并检查 `xdata` 和 `ydata` 中是否包含 NaN 值
        policies = [None, 'raise', 'omit']
        x_contains_nan, nan_policy = _contains_nan(xdata, nan_policy,
                                                   policies=policies)
        y_contains_nan, nan_policy = _contains_nan(ydata, nan_policy,
                                                   policies=policies)

        # 如果 `nan_policy` 为 'omit' 并且 `xdata` 或 `ydata` 中包含 NaN 值
        # 则忽略包含 NaN 值的数据点
        if (x_contains_nan or y_contains_nan) and nan_policy == 'omit':
            # 检查 `xdata` 和 `ydata` 中的 NaN 值，对 N 维数组忽略 NaN 值
            has_nan = np.isnan(xdata)
            has_nan = has_nan.any(axis=tuple(range(has_nan.ndim-1)))
            has_nan |= np.isnan(ydata)

            # 筛选出不包含 NaN 值的数据点
            xdata = xdata[..., ~has_nan]
            ydata = ydata[~has_nan]

    # 确定 sigma 的类型
    if sigma is not None:
        sigma = np.asarray(sigma)

        # 如果 sigma 是 1 维数组或标量，则认为是误差，定义 transform = 1/sigma
        if sigma.size == 1 or sigma.shape == (ydata.size, ):
            transform = 1.0 / sigma
        # 如果 sigma 是 2 维数组，则认为是协方差矩阵，定义 transform = L，使得 L L^T = C
        elif sigma.shape == (ydata.size, ydata.size):
            try:
                # 使用 cholesky 分解求解 L，要求 lower=True 返回 L L^T = A
                transform = cholesky(sigma, lower=True)
            except LinAlgError as e:
                raise ValueError("`sigma` must be positive definite.") from e
        else:
            raise ValueError("`sigma` has incorrect shape.")
    else:
        transform = None

    # 使用 memoizer 封装 `_wrap_func` 函数，创建 func 函数
    func = _lightweight_memoizer(_wrap_func(f, xdata, ydata, transform))

    # 如果 jac 是可调用对象，则使用 memoizer 封装 `_wrap_jac` 函数，创建 jac 函数
    # 如果 jac 为 None 并且 method 不为 'lm'，则使用 '2-point' 近似求导
    if callable(jac):
        jac = _lightweight_memoizer(_wrap_jac(jac, xdata, transform))
    elif jac is None and method != 'lm':
        jac = '2-point'

    # 如果 kwargs 中包含 'args' 关键字参数，则抛出 ValueError 异常
    if 'args' in kwargs:
        # 模型函数 `f` 的规范不支持额外的参数，参考 `curve_fit` 的文档了解 `f` 的接受的调用签名
        raise ValueError("'args' is not a supported keyword argument.")
    # 如果使用 Levenberg-Marquardt 方法进行拟合
    if method == 'lm':
        # 检查 ydata 的大小是否为 1，以确定是否进行广播处理
        if ydata.size != 1 and n > ydata.size:
            # 如果函数参数个数 n 超过了数据点个数 ydata.size，抛出异常
            raise TypeError(f"The number of func parameters={n} must not"
                            f" exceed the number of data points={ydata.size}")
        
        # 调用 leastsq 函数进行拟合
        res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
        popt, pcov, infodict, errmsg, ier = res
        
        # 获取残差向量的长度和成本（残差平方和）
        ysize = len(infodict['fvec'])
        cost = np.sum(infodict['fvec'] ** 2)
        
        # 如果拟合过程中出现错误码不在预期的范围内，抛出运行时异常
        if ier not in [1, 2, 3, 4]:
            raise RuntimeError("Optimal parameters not found: " + errmsg)
    else:
        # 如果使用其他方法进行拟合，将 maxfev 参数重命名为 max_nfev
        if 'max_nfev' not in kwargs:
            kwargs['max_nfev'] = kwargs.pop('maxfev', None)
        
        # 调用 least_squares 函数进行拟合
        res = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
                            **kwargs)
        
        # 如果拟合未成功，抛出运行时异常
        if not res.success:
            raise RuntimeError("Optimal parameters not found: " + res.message)
        
        # 提取拟合结果中的信息
        infodict = dict(nfev=res.nfev, fvec=res.fun)
        ier = res.status
        errmsg = res.message
        
        # 获取残差向量的长度和成本（残差平方和）
        ysize = len(res.fun)
        cost = 2 * res.cost  # res.cost 是残差平方和的一半
        popt = res.x
        
        # 计算 Moore-Penrose 伪逆，丢弃零奇异值
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
    
    # 如果协方差矩阵为 None 或包含 NaN 值，处理为不确定的协方差
    warn_cov = False
    if pcov is None or np.isnan(pcov).any():
        # 创建一个全零的协方差矩阵
        pcov = zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(inf)
        warn_cov = True
    elif not absolute_sigma:
        # 根据是否绝对标准差来调整协方差矩阵
        if ysize > p0.size:
            s_sq = cost / (ysize - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(inf)
            warn_cov = True
    
    # 如果协方差估计不可靠，发出警告
    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning, stacklevel=2)
    
    # 根据是否输出完整结果决定返回值
    if full_output:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov
# 定义函数，用于检查给定函数在某点处的梯度是否正确
def check_gradient(fcn, Dfcn, x0, args=(), col_deriv=0):
    """Perform a simple check on the gradient for correctness.
    
    Args:
    fcn : function
        要评估的函数。
    Dfcn : function
        函数的导数。
    x0 : array_like
        函数的参数，起始点。
    args : tuple, optional
        传递给函数和导数的额外参数，默认为空。
    col_deriv : int, optional
        指示是否按列返回导数，0表示是，1表示否，默认为0。

    Returns:
    tuple
        一个布尔值和一个浮点数数组，分别表示梯度检查是否通过和每个分量的误差。

    """

    # 将输入参数 x0 至少转换为一维数组
    x = atleast_1d(x0)
    # 获取数组 x 的长度
    n = len(x)
    # 将 x 重塑为一维数组
    x = x.reshape((n,))
    # 计算函数在 x 点处的函数值，并至少将其转换为一维数组
    fvec = atleast_1d(fcn(x, *args))
    # 获取数组 fvec 的长度
    m = len(fvec)
    # 将 fvec 重塑为一维数组
    fvec = fvec.reshape((m,))
    # 设置雅可比矩阵的行数为 m
    ldfjac = m
    # 计算函数在 x 点处的导数，并至少将其转换为一维数组
    fjac = atleast_1d(Dfcn(x, *args))
    # 将 fjac 重塑为 m 行 n 列的二维数组
    fjac = fjac.reshape((m, n))
    # 如果 col_deriv 为 0，则转置雅可比矩阵
    if col_deriv == 0:
        fjac = transpose(fjac)

    # 创建 n 维零向量 xp 和 m 维零向量 err
    xp = zeros((n,), float)
    err = zeros((m,), float)
    fvecp = None
    # 调用 Fortran 子程序 _minpack._chkder，进行梯度检查
    _minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 1, err)

    # 计算函数在 xp 点处的函数值，并至少将其转换为一维数组
    fvecp = atleast_1d(fcn(xp, *args))
    # 将 fvecp 重塑为一维数组
    fvecp = fvecp.reshape((m,))
    # 再次调用 Fortran 子程序 _minpack._chkder，进行梯度检查
    _minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 2, err)

    # 计算所有误差的元素乘积，并检查是否大于 0.5，得到一个布尔值数组
    good = (prod(greater(err, 0.5), axis=0))

    # 返回结果，包括布尔值数组和误差数组
    return (good, err)


# 定义函数，计算二维向量 p0 和 p1 之间的差的平方除以 d
def _del2(p0, p1, d):
    return p0 - np.square(p1 - p0) / d


# 定义函数，计算实际值与期望值之间的相对误差
def _relerr(actual, desired):
    return (actual - desired) / desired


# 定义函数，实现固定点迭代法的辅助函数
def _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel):
    """
    Helper function for finding a fixed point using specified method.

    Args:
    func : function
        Function to find the fixed point of.
    x0 : array_like
        Initial guess for the fixed point.
    args : tuple
        Extra arguments to `func`.
    xtol : float, optional
        Tolerance to achieve convergence, defaults to 1e-8.
    maxiter : int, optional
        Maximum number of iterations, defaults to 500.
    use_accel : bool
        Indicates whether to use acceleration for convergence.

    Returns:
    float
        Approximate fixed point of the function.

    Raises:
    RuntimeError
        If the fixed point cannot be found within the maximum number of iterations.

    """
    # 使用初始点 x0 开始迭代
    p0 = x0
    # 迭代最多 maxiter 次
    for i in range(maxiter):
        # 计算函数在 p0 点处的值
        p1 = func(p0, *args)
        # 如果使用加速方法，计算 p1 的加速值 p2
        if use_accel:
            p2 = func(p1, *args)
            d = p2 - 2.0 * p1 + p0
            # 使用 _del2 函数计算新的近似值 p
            p = _lazywhere(d != 0, (p0, p1, d), f=_del2, fillvalue=p2)
        else:
            # 如果不使用加速，直接取 p1 作为新的近似值 p
            p = p1
        # 计算当前近似值 p0 与新近似值 p 之间的相对误差
        relerr = _lazywhere(p0 != 0, (p, p0), f=_relerr, fillvalue=p)
        # 如果所有元素的绝对值都小于指定的容差 xtol，则认为收敛
        if np.all(np.abs(relerr) < xtol):
            return p
        # 更新 p0 为新的近似值，继续下一次迭代
        p0 = p
    # 如果达到最大迭代次数仍未收敛，则抛出运行时错误
    msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
    raise RuntimeError(msg)


# 定义函数，使用指定方法寻找函数的固定点
def fixed_point(func, x0, args=(), xtol=1e-8, maxiter=500, method='del2'):
    """
    Find a fixed point of the function.

    Given a function of one or more variables and a starting point, find a
    fixed point of the function: i.e., where ``func(x0) == x0``.

    Parameters
    ----------
    func : function
        Function to evaluate.
    x0 : array_like
        Fixed point of function.
    args : tuple, optional
        Extra arguments to `func`.
    xtol : float, optional
        Convergence tolerance, defaults to 1e-08.
    maxiter : int, optional
        Maximum number of iterations, defaults to 500.
    method : {"del2", "iteration"}, optional
        Method of finding the fixed-point, defaults to "del2",
        which uses Steffensen's Method with Aitken's ``Del^2``
        convergence acceleration [1]_. The "iteration" method simply iterates
        the function until convergence is detected, without attempting to
        accelerate the convergence.

    References
    ----------
    .. [1] Burden, Faires, "Numerical Analysis", 5th edition, pg. 80

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import optimize
    >>> def func(x, c1, c2):
    ...    return np.sqrt(c1/(x+c2))
    >>> c1 = np.array([10,12.])
    >>> c2 = np.array([3, 5.])
    >>> optimize.fixed_point(func, [1.2, 1.3], args=(c1,c2))
    array([ 1.4920333 ,  1.37228132])

    """
    # 根据指定方法选择是否使用加速
    use_accel = {'del2': True, 'iteration': False}[method]
    # 调用_asarray_validated函数，将x0转换为合适的数组，确保其为浮点数类型
    x0 = _asarray_validated(x0, as_inexact=True)
    # 调用_fixed_point_helper函数，进行迭代求解函数func的不动点
    # func: 待求解的函数
    # x0: 初始猜测值
    # args: 其他参数
    # xtol: 迭代收敛的容差
    # maxiter: 最大迭代次数
    # use_accel: 是否使用加速方法
    return _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel)
```