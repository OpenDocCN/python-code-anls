# `D:\src\scipysrc\scipy\scipy\optimize\_spectral.py`

```
"""
Spectral Algorithm for Nonlinear Equations
"""
# 导入必要的库和模块
import collections

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._optimize import _check_unknown_options
# 导入非单调线搜索方法
from ._linesearch import _nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng

# 定义一个异常类，用于表示算法未收敛的情况
class _NoConvergence(Exception):
    pass

# 定义一个函数，使用 DF-SANE 方法解非线性方程组
def _root_df_sane(func, x0, args=(), ftol=1e-8, fatol=1e-300, maxfev=1000,
                  fnorm=None, callback=None, disp=False, M=10, eta_strategy=None,
                  sigma_eps=1e-10, sigma_0=1.0, line_search='cruz', **unknown_options):
    r"""
    Solve nonlinear equation with the DF-SANE method

    Options
    -------
    ftol : float, optional
        Relative norm tolerance.
    fatol : float, optional
        Absolute norm tolerance.
        Algorithm terminates when ``||func(x)|| < fatol + ftol ||func(x_0)||``.
    fnorm : callable, optional
        Norm to use in the convergence check. If None, 2-norm is used.
    maxfev : int, optional
        Maximum number of function evaluations.
    disp : bool, optional
        Whether to print convergence process to stdout.
    eta_strategy : callable, optional
        Choice of the ``eta_k`` parameter, which gives slack for growth
        of ``||F||**2``.  Called as ``eta_k = eta_strategy(k, x, F)`` with
        `k` the iteration number, `x` the current iterate and `F` the current
        residual. Should satisfy ``eta_k > 0`` and ``sum(eta, k=0..inf) < inf``.
        Default: ``||F||**2 / (1 + k)**2``.
    sigma_eps : float, optional
        The spectral coefficient is constrained to ``sigma_eps < sigma < 1/sigma_eps``.
        Default: 1e-10
    sigma_0 : float, optional
        Initial spectral coefficient.
        Default: 1.0
    M : int, optional
        Number of iterates to include in the nonmonotonic line search.
        Default: 10
    line_search : {'cruz', 'cheng'}
        Type of line search to employ. 'cruz' is the original one defined in
        [Martinez & Raydan. Math. Comp. 75, 1429 (2006)], 'cheng' is
        a modified search defined in [Cheng & Li. IMA J. Numer. Anal. 29, 814 (2009)].
        Default: 'cruz'

    References
    ----------
    .. [1] "Spectral residual method without gradient information for solving
           large-scale nonlinear systems of equations." W. La Cruz,
           J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).
    .. [2] W. La Cruz, Opt. Meth. Software, 29, 24 (2014).
    .. [3] W. Cheng, D.-H. Li. IMA J. Numer. Anal. **29**, 814 (2009).

    """
    # 检查未知选项，确保它们是有效的
    _check_unknown_options(unknown_options)

    # 检查线搜索类型是否有效
    if line_search not in ('cheng', 'cruz'):
        raise ValueError(f"Invalid value {line_search!r} for 'line_search'")

    # 设置默认的指数
    nexp = 2

    # 如果未指定 eta_strategy，则使用默认的函数定义
    if eta_strategy is None:
        # 不同于文献 [1] 的选择，这里的 eta 不会随着 F 的缩放而变化
        def eta_strategy(k, x, F):
            # 从外部范围获取初始残差的平方二范数
            return f_0 / (1 + k)**2
    `
        if fnorm is None:
            def fnorm(F):
                # 从外部作用域获取当前残差的平方二范数
                return f_k**(1.0/nexp)
    
        def fmerit(F):
            # 计算 F 的 n 次方范数作为目标函数的代价
            return np.linalg.norm(F)**nexp
    
        nfev = [0]  # 初始化函数评估次数计数器
    
        # 调用 _wrap_func 函数包装 func 并初始化相关参数，返回函数值、当前位置、形状、当前函数值、残差等
        f, x_k, x_shape, f_k, F_k, is_complex = _wrap_func(func, x0, fmerit,
                                                           nfev, maxfev, args)
    
        k = 0  # 初始化迭代计数器
        f_0 = f_k  # 初始化 f_0 为初始函数值
        sigma_k = sigma_0  # 初始化谱参数 sigma_k
    
        F_0_norm = fnorm(F_k)  # 计算初始残差的范数
    
        # 初始化 'cruz' 线搜索的历史函数值队列，最大长度为 M
        prev_fs = collections.deque([f_k], M)
    
        # 初始化 'cheng' 线搜索相关参数
        Q = 1.0
        C = f_0
    
        converged = False  # 初始化收敛标志为 False
        message = "too many function evaluations required"  # 初始化消息
    
        while True:
            F_k_norm = fnorm(F_k)  # 计算当前残差的范数
    
            if disp:
                # 如果 disp 为 True，打印当前迭代信息
                print("iter %d: ||F|| = %g, sigma = %g" % (k, F_k_norm, sigma_k))
    
            if callback is not None:
                # 如果回调函数不为空，调用回调函数
                callback(x_k, F_k)
    
            if F_k_norm < ftol * F_0_norm + fatol:
                # 如果当前残差的范数小于目标容忍度，收敛
                message = "successful convergence"
                converged = True
                break
    
            # 控制谱参数，根据[2]中的公式
            if abs(sigma_k) > 1/sigma_eps:
                sigma_k = 1/sigma_eps * np.sign(sigma_k)
            elif abs(sigma_k) < sigma_eps:
                sigma_k = sigma_eps
    
            # 计算线搜索方向
            d = -sigma_k * F_k
    
            # 非单调线搜索策略
            eta = eta_strategy(k, x_k, F_k)
            try:
                if line_search == 'cruz':
                    # 使用 cruz 方法进行非单调线搜索
                    alpha, xp, fp, Fp = _nonmonotone_line_search_cruz(f, x_k, d, prev_fs,
                                                                      eta=eta)
                elif line_search == 'cheng':
                    # 使用 cheng 方法进行非单调线搜索
                    alpha, xp, fp, Fp, C, Q = _nonmonotone_line_search_cheng(f, x_k, d, f_k,
                                                                             C, Q, eta=eta)
            except _NoConvergence:
                # 如果非单调线搜索未能收敛，跳出循环
                break
    
            # 更新谱参数
            s_k = xp - x_k  # 更新步长 s_k
            y_k = Fp - F_k  # 更新残差 y_k
            sigma_k = np.vdot(s_k, s_k) / np.vdot(s_k, y_k)  # 更新谱参数
    
            # 执行一步更新
            x_k = xp
            F_k = Fp
            f_k = fp
    
            # 存储函数值，如果使用 cruz 线搜索，则将 fp 添加到历史队列
            if line_search == 'cruz':
                prev_fs.append(fp)
    
            k += 1  # 增加迭代次数
    
        x = _wrap_result(x_k, is_complex, shape=x_shape)  # 包装优化结果的 x
        F = _wrap_result(F_k, is_complex)  # 包装优化结果的 F
    
        result = OptimizeResult(x=x, success=converged,
                                message=message,
                                fun=F, nfev=nfev[0], nit=k, method="df-sane")
    
        return result  # 返回优化结果
def _wrap_func(func, x0, fmerit, nfev_list, maxfev, args=()):
    """
    Wrap a function and an initial value so that (i) complex values
    are wrapped to reals, and (ii) value for a merit function
    fmerit(x, f) is computed at the same time, (iii) iteration count
    is maintained and an exception is raised if it is exceeded.

    Parameters
    ----------
    func : callable
        Function to wrap
    x0 : ndarray
        Initial value
    fmerit : callable
        Merit function fmerit(f) for computing merit value from residual.
    nfev_list : list
        List to store number of evaluations in. Should be [0] in the beginning.
    maxfev : int
        Maximum number of evaluations before _NoConvergence is raised.
    args : tuple
        Extra arguments to func

    Returns
    -------
    wrap_func : callable
        Wrapped function, to be called as
        ``F, fp = wrap_func(x0)``
    x0_wrap : ndarray of float
        Wrapped initial value; raveled to 1-D and complex
        values mapped to reals.
    x0_shape : tuple
        Shape of the initial value array
    f : float
        Merit function at F
    F : ndarray of float
        Residual at x0_wrap
    is_complex : bool
        Whether complex values were mapped to reals

    """
    # 将 x0 转换为 NumPy 数组
    x0 = np.asarray(x0)
    # 记录 x0 的形状
    x0_shape = x0.shape
    # 计算初始值 x0 对应的函数值，并将其展平为一维数组
    F = np.asarray(func(x0, *args)).ravel()
    # 检查 x0 或 F 是否包含复数，以确定是否需要将复数映射到实数
    is_complex = np.iscomplexobj(x0) or np.iscomplexobj(F)
    # 将 x0 也展平为一维数组
    x0 = x0.ravel()

    # 初始化评估次数列表的第一个元素为 1
    nfev_list[0] = 1

    # 根据是否包含复数，定义不同的包装函数
    if is_complex:
        def wrap_func(x):
            # 检查评估次数是否达到最大值，如果是则抛出 _NoConvergence 异常
            if nfev_list[0] >= maxfev:
                raise _NoConvergence()
            # 增加评估次数计数
            nfev_list[0] += 1
            # 将实数数组 x 转换为复数形式，并按照 x0 的形状重新组织
            z = _real2complex(x).reshape(x0_shape)
            # 计算 z 对应的函数值，并将其展平为一维数组
            v = np.asarray(func(z, *args)).ravel()
            # 将复数结果映射回实数
            F = _complex2real(v)
            # 计算 Merit 函数的值
            f = fmerit(F)
            return f, F

        # 将初始值 x0 映射为实数
        x0 = _complex2real(x0)
        # 将函数值 F 映射为实数
        F = _complex2real(F)
    else:
        def wrap_func(x):
            # 检查评估次数是否达到最大值，如果是则抛出 _NoConvergence 异常
            if nfev_list[0] >= maxfev:
                raise _NoConvergence()
            # 增加评估次数计数
            nfev_list[0] += 1
            # 将 x 恢复为 x0 的形状
            x = x.reshape(x0_shape)
            # 计算 x 对应的函数值，并将其展平为一维数组
            F = np.asarray(func(x, *args)).ravel()
            # 计算 Merit 函数的值
            f = fmerit(F)
            return f, F

    # 返回包装后的函数及其相关数据
    return wrap_func, x0, x0_shape, fmerit(F), F, is_complex


def _wrap_result(result, is_complex, shape=None):
    """
    Convert from real to complex and reshape result arrays.
    """
    # 如果结果包含复数，将其从实数转换为复数形式
    if is_complex:
        z = _real2complex(result)
    else:
        z = result
    # 如果指定了形状，则将结果重新组织成指定的形状
    if shape is not None:
        z = z.reshape(shape)
    return z


def _real2complex(x):
    # 将实数数组转换为复数形式
    return np.ascontiguousarray(x, dtype=float).view(np.complex128)


def _complex2real(z):
    # 将复数数组转换为实数形式
    return np.ascontiguousarray(z, dtype=complex).view(np.float64)
```