# `D:\src\scipysrc\scipy\scipy\optimize\_tnc.py`

```
# TNC Python 接口
# @(#) $Jeannot: tnc.py,v 1.11 2005/01/28 18:27:31 js Exp $

# 版权所有 (c) 2004-2005，Jean-Sebastien Roy (js@jeannot.org)

# 根据以下条件，免费授予使用、复制、修改、合并、发布、分发、再许可和/或出售软件副本的权限：
# - 必须包含上述版权声明和本许可声明的软件副本或重要部分。

# 本软件按“原样”提供，不提供任何形式的明示或暗示担保，包括但不限于对适销性、特定用途的适用性和非侵权性的保证。 
# 作者或版权所有者在任何情况下均不对任何索赔、损害或其他责任负责，无论是合同行为、侵权行为还是其他行为。

"""
TNC：TNC非线性优化器的Python接口

TNC是一个非线性优化器。要使用它，必须提供一个要最小化的函数。该函数必须接受一个参数：要评估函数的坐标列表；并且它必须返回一个元组，其第一个元素是函数值，第二个元素是函数的梯度（作为值列表）；或者返回None，以中止最小化过程。
"""

from scipy.optimize import _moduleTNC as moduleTNC
from ._optimize import (MemoizeJac, OptimizeResult, _check_unknown_options,
                       _prepare_scalar_function)
from ._constraints import old_bound_to_new
from scipy._lib._array_api import atleast_nd, array_namespace

from numpy import inf, array, zeros

__all__ = ['fmin_tnc']

# 不同消息类型的标志
MSG_NONE = 0  # 没有消息
MSG_ITER = 1  # 每次迭代一行消息
MSG_INFO = 2  # 信息性消息
MSG_VERS = 4  # 版本信息
MSG_EXIT = 8  # 退出原因
MSG_ALL = MSG_ITER + MSG_INFO + MSG_VERS + MSG_EXIT

# 消息类型与描述的映射关系
MSGS = {
        MSG_NONE: "没有消息",
        MSG_ITER: "每次迭代一行消息",
        MSG_INFO: "信息性消息",
        MSG_VERS: "版本信息",
        MSG_EXIT: "退出原因",
        MSG_ALL: "所有消息"
}

# 不同退出原因的标识
INFEASIBLE = -1  # 不可行（下界大于上界）
LOCALMINIMUM = 0  # 达到局部最小值（|pg|约等于0）
FCONVERGED = 1  # 收敛（|f_n-f_(n-1)|约等于0）
XCONVERGED = 2  # 收敛（|x_n-x_(n-1)|约等于0）
MAXFUN = 3  # 达到最大函数评估次数
LSFAIL = 4  # 线性搜索失败
CONSTANT = 5  # 所有下界等于上界
NOPROGRESS = 6  # 无法进展
USERABORT = 7  # 用户请求终止最小化过程
# 定义一个全局字典，用于存储返回给用户的特定优化消息
RCSTRINGS = {
        INFEASIBLE: "Infeasible (lower bound > upper bound)",  # 无解（下界大于上界）
        LOCALMINIMUM: "Local minimum reached (|pg| ~= 0)",  # 达到局部最小值（|pg| ~= 0）
        FCONVERGED: "Converged (|f_n-f_(n-1)| ~= 0)",  # 收敛（|f_n-f_(n-1)| ~= 0）
        XCONVERGED: "Converged (|x_n-x_(n-1)| ~= 0)",  # 收敛（|x_n-x_(n-1)| ~= 0）
        MAXFUN: "Max. number of function evaluations reached",  # 达到最大函数评估次数
        LSFAIL: "Linear search failed",  # 线性搜索失败
        CONSTANT: "All lower bounds are equal to the upper bounds",  # 所有下界等于上界
        NOPROGRESS: "Unable to progress",  # 无法进展
        USERABORT: "User requested end of minimization"  # 用户请求终止最小化过程
}

# Travis Oliphant 于 2004 年 4 月对接口进行的更改，以便包含在 SciPy 中
# 为了使用截断牛顿算法最小化带有变量边界的函数，提供了一个函数接口的封装
def fmin_tnc(func, x0, fprime=None, args=(), approx_grad=0,
             bounds=None, epsilon=1e-8, scale=None, offset=None,
             messages=MSG_ALL, maxCGit=-1, maxfun=None, eta=-1,
             stepmx=0, accuracy=0, fmin=0, ftol=-1, xtol=-1, pgtol=-1,
             rescale=-1, disp=None, callback=None):
    """
    使用截断牛顿算法最小化带有变量边界的函数，此方法包装了算法的 C 实现。

    Parameters
    ----------
    func : callable ``func(x, *args)``
        要最小化的函数。必须满足以下条件之一：

        1. 返回 f 和 g，其中 f 是函数值，g 是其梯度（一个浮点数列表）。
        2. 返回函数值，但分别提供梯度函数作为 `fprime`。
        3. 返回函数值，并设置 ``approx_grad=True``。

        如果函数返回 None，则最小化过程将中止。
    x0 : array_like
        最小值的初始估计。
    fprime : callable ``fprime(x, *args)``, optional
        `func` 的梯度。如果为 None，则 `func` 必须返回函数值和梯度（``f,g = func(x, *args)``），
        或者必须设置 `approx_grad=True`。
    args : tuple, optional
        传递给函数的参数。
    approx_grad : bool, optional
        如果为真，则数值上近似梯度。
    bounds : list, optional
        对于 x0 中的每个元素，定义该参数的边界（min, max）对。当某个方向没有边界时，使用 None 或 +/-inf。
    epsilon : float, optional
        如果 approx_grad 为 True，则在有限差分逼近中使用的步长。
    scale : array_like, optional
        应用于每个变量的缩放因子。如果为 None，则对于区间边界变量，因子为上界减下界，对于其他变量为1+|x|。默认为 None。
    offset : array_like, optional
        每个变量要减去的值。如果为 None，则对于区间边界变量，偏移为（上界+下界）/2，对于其他变量为 x。默认为 None。
    messages : int, optional
        用于在最小化过程中选择消息显示的位掩码，值在 MSGS 字典中定义。默认为 MSG_ALL。
    maxCGit : int, optional
        允许的最大共轭梯度迭代次数。默认为 -1，即无限制。
    maxfun : int or None, optional
        允许的最大函数评估次数。默认为 None，即无限制。
    eta : float, optional
        罚项参数的起始值。默认为 -1。
    stepmx : float, optional
        在一步中允许的最大长度。默认为 0。
    accuracy : float, optional
        保留特征值分解的精度要求。默认为 0。
    fmin : float, optional
        求解过程中允许的最小函数值。默认为 0。
    ftol : float, optional
        终止条件之一：当函数值减小到这个值以下时，最小化过程终止。默认为 -1，即不使用此终止条件。
    xtol : float, optional
        终止条件之一：当参数值变化到这个程度以下时，最小化过程终止。默认为 -1，即不使用此终止条件。
    pgtol : float, optional
        终止条件之一：当梯度变化到这个程度以下时，最小化过程终止。默认为 -1，即不使用此终止条件。
    rescale : float, optional
        等比例因子。默认为 -1。
    disp : int or bool, optional
        是否显示详细信息。默认为 None。
    callback : callable, optional
        在每次迭代中调用的函数。默认为 None。
    """
    # disp: int, optional
    #     控制消息输出的整数接口。0 表示没有消息输出，5 表示输出所有消息。
    # maxCGit: int, optional
    #     主迭代过程中每次Hessian*向量评估的最大次数。如果 maxCGit == 0，选择方向为负梯度；
    #     如果 maxCGit < 0，maxCGit 被设置为 max(1, min(50, n/2))。默认为 -1。
    # maxfun: int, optional
    #     最大函数评估次数。如果为 None，则 maxfun 被设置为 max(100, 10*len(x0))。默认为 None。
    #     注意，由于通过数值微分评估梯度，此函数可能会超过限制。
    # eta: float, optional
    #     线搜索的严重性。如果 < 0 或 > 1，则设置为 0.25。默认为 -1。
    # stepmx: float, optional
    #     线搜索的最大步长。可以在调用过程中增加。如果太小，将设置为 10.0。默认为 0。
    # accuracy: float, optional
    #     有限差分计算的相对精度。如果 <= 机器精度，则设置为 sqrt(机器精度)。默认为 0。
    # fmin: float, optional
    #     函数值的最小估计。默认为 0。
    # ftol: float, optional
    #     停止标准中 f 值的精度目标。如果 ftol < 0.0，则 ftol 被设置为 0.0。默认为 -1。
    # xtol: float, optional
    #     停止标准中 x 值的精度目标（应用 x 缩放因子后）。如果 xtol < 0.0，则 xtol 被设置为 sqrt(机器精度)。默认为 -1。
    # pgtol: float, optional
    #     停止标准中投影梯度值的精度目标（应用 x 缩放因子后）。如果 pgtol < 0.0，则 pgtol 被设置为 1e-2 * sqrt(accuracy)。
    #     设置为 0.0 不推荐。默认为 -1。
    # rescale: float, optional
    #     触发 f 值重新缩放的缩放因子（以 log10 表示）。如果为 0，则每次迭代重新缩放。如果是一个较大的值，则永远不会重新缩放。
    #     如果 < 0，则 rescale 被设置为 1.3。默认为 0。
    # callback: callable, optional
    #     每次迭代后调用的回调函数，形式为 callback(xk)，其中 xk 是当前参数向量。

    # 返回值
    # ------
    # x: ndarray
    #     解决方案。
    # nfeval: int
    #     函数评估次数。
    # rc: int
    #     返回码，见下文

    # 参见
    # ----
    # minimize: 多变量函数最小化算法的接口。特别参见 'TNC' 方法。

    # 注释
    # ----
    # 底层算法是截断牛顿法，也称为牛顿共轭梯度法。该方法与 scipy.optimize.fmin_ncg 的不同之处在于：
    # 1. 它包装了该算法的C实现。
    # 2. 允许每个变量设定上下界约束。

    # 该算法通过像无约束截断牛顿法一样确定下降方向来整合界约束，但从不采取足以使空间离开的步长。
    # 处理 fprime/approx_grad 参数
    if approx_grad:
        fun = func
        jac = None
    elif fprime is None:
        # 如果没有提供精确的梯度，使用 MemoizeJac 对象封装 func 函数，并获取其导数
        fun = MemoizeJac(func)
        jac = fun.derivative
    else:
        fun = func
        jac = fprime

    # 根据 disp 参数确定消息输出级别
    if disp is not None:  # disp 参数优先于 messages 参数
        mesg_num = disp
    else:
        # 根据 messages 参数选择消息输出级别，默认为 MSG_ALL
        mesg_num = {0:MSG_NONE, 1:MSG_ITER, 2:MSG_INFO, 3:MSG_VERS,
                    4:MSG_EXIT, 5:MSG_ALL}.get(messages, MSG_ALL)
    
    # 构建选项字典，用于传递给 _minimize_tnc 函数
    opts = {'eps': epsilon,         # 控制步长的小幅度增量
            'scale': scale,         # 缩放参数
            'offset': offset,       # 偏移量
            'mesg_num': mesg_num,   # 消息输出级别
            'maxCGit': maxCGit,     # 允许的最大共轭梯度迭代次数
            'maxfun': maxfun,       # 允许的最大函数评估次数
            'eta': eta,             # 缩放参数
            'stepmx': stepmx,       # 最大步长
            'accuracy': accuracy,   # 函数值的精度要求
            'minfev': fmin,         # 函数的最小值
            'ftol': ftol,           # 函数值的公差
            'xtol': xtol,           # 变量的公差
            'gtol': pgtol,          # 梯度的公差
            'rescale': rescale,     # 是否对问题进行重新缩放
            'disp': False}          # 是否显示优化过程信息

    # 调用 _minimize_tnc 函数进行优化，返回优化结果字典中的 x 值、函数评估次数和状态
    res = _minimize_tnc(fun, x0, args, jac, bounds, callback=callback, **opts)

    # 返回优化结果中的最优解 x 值、函数评估次数和状态
    return res['x'], res['nfev'], res['status']
# 定义一个函数 _minimize_tnc，用于使用截断牛顿（TNC）算法最小化一个或多个变量的标量函数

def _minimize_tnc(fun, x0, args=(), jac=None, bounds=None,
                  eps=1e-8, scale=None, offset=None, mesg_num=None,
                  maxCGit=-1, eta=-1, stepmx=0, accuracy=0,
                  minfev=0, ftol=-1, xtol=-1, gtol=-1, rescale=-1, disp=False,
                  callback=None, finite_diff_rel_step=None, maxfun=None,
                  **unknown_options):
    """
    Minimize a scalar function of one or more variables using a truncated
    Newton (TNC) algorithm.
    
    Options
    -------
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    scale : list of floats
        Scaling factors to apply to each variable. If None, the
        factors are up-low for interval bounded variables and
        1+|x] for the others. Defaults to None.
    offset : float
        Value to subtract from each variable. If None, the
        offsets are (up+low)/2 for interval bounded variables
        and x for the others.
    disp : bool
       Set to True to print convergence messages.
    maxCGit : int
        Maximum number of hessian*vector evaluations per main
        iteration. If maxCGit == 0, the direction chosen is
        -gradient if maxCGit < 0, maxCGit is set to
        max(1,min(50,n/2)). Defaults to -1.
    eta : float
        Severity of the line search. If < 0 or > 1, set to 0.25.
        Defaults to -1.
    stepmx : float
        Maximum step for the line search. May be increased during
        call. If too small, it will be set to 10.0. Defaults to 0.
    accuracy : float
        Relative precision for finite difference calculations. If
        <= machine_precision, set to sqrt(machine_precision).
        Defaults to 0.
    minfev : float
        Minimum function value estimate. Defaults to 0.
    ftol : float
        Precision goal for the value of f in the stopping criterion.
        If ftol < 0.0, ftol is set to 0.0 defaults to -1.
    xtol : float
        Precision goal for the value of x in the stopping
        criterion (after applying x scaling factors). If xtol <
        0.0, xtol is set to sqrt(machine_precision). Defaults to
        -1.
    gtol : float
        Precision goal for the value of the projected gradient in
        the stopping criterion (after applying x scaling factors).
        If gtol < 0.0, gtol is set to 1e-2 * sqrt(accuracy).
        Setting it to 0.0 is not recommended. Defaults to -1.
    rescale : float
        Scaling factor (in log10) used to trigger f value
        rescaling.  If 0, rescale at each iteration.  If a large
        value, never rescale.  If < 0, rescale is set to 1.3.
    """
    pass  # 此处为函数体，暂未实现具体逻辑，因此使用 pass 作为占位符
    """
    finite_diff_rel_step : None or array_like, optional
        如果 `jac` 在 ['2-point', '3-point', 'cs'] 中，则用于数值近似雅可比矩阵的相对步长。
        绝对步长计算为 `h = rel_step * sign(x) * max(1, abs(x))`，可能会根据边界进行调整。
        对于 `method='3-point'`，忽略 `h` 的符号。如果为 None（默认），则步长会自动选择。
    maxfun : int
        最大函数评估次数。如果为 None，则 `maxfun` 设置为 max(100, 10*len(x0))。默认为 None。
    """
    # 检查未知选项
    _check_unknown_options(unknown_options)
    # 初始化 fmin 和 pgtol
    fmin = minfev
    pgtol = gtol

    # 使用适当的数组命名空间来处理输入的初始向量 x0
    xp = array_namespace(x0)
    # 将 x0 至少转换为一维数组，并根据命名空间重新分配内存
    x0 = atleast_nd(x0, ndim=1, xp=xp)
    # 确定数据类型，默认为 xp.float64
    dtype = xp.float64
    # 如果 x0 的数据类型是实数浮点型，则保留该数据类型
    if xp.isdtype(x0.dtype, "real floating"):
        dtype = x0.dtype
    # 将 x0 转换为指定数据类型的一维数组
    x0 = xp.reshape(xp.astype(x0, dtype), -1)

    # 确定 x0 的长度
    n = len(x0)

    # 如果 bounds 为 None，则初始化为 [(None,None)] * n
    if bounds is None:
        bounds = [(None,None)] * n
    # 如果 bounds 的长度与 x0 不相等，则引发 ValueError 异常
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    # 将原始 bounds 转换为新格式的 bounds
    new_bounds = old_bound_to_new(bounds)

    # 根据 mesg_num 设置消息类型
    if mesg_num is not None:
        messages = {0:MSG_NONE, 1:MSG_ITER, 2:MSG_INFO, 3:MSG_VERS,
                    4:MSG_EXIT, 5:MSG_ALL}.get(mesg_num, MSG_ALL)
    elif disp:
        messages = MSG_ALL
    else:
        messages = MSG_NONE

    # 准备标量函数 sf，包括计算函数值和梯度的方法
    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  bounds=new_bounds)
    func_and_grad = sf.fun_and_grad

    """
    low, up   : the bounds (lists of floats)
                if low is None, the lower bounds are removed.
                if up is None, the upper bounds are removed.
                low and up defaults to None
    """
    # 初始化 low 和 up 边界数组
    low = zeros(n)
    up = zeros(n)
    # 遍历每个维度，根据 bounds 设置对应的边界值
    for i in range(n):
        if bounds[i] is None:
            l, u = -inf, inf
        else:
            l,u = bounds[i]
            if l is None:
                low[i] = -inf
            else:
                low[i] = l
            if u is None:
                up[i] = inf
            else:
                up[i] = u

    # 如果 scale 为 None，则设置为空数组
    if scale is None:
        scale = array([])

    # 如果 offset 为 None，则设置为空数组
    if offset is None:
        offset = array([])

    # 如果 maxfun 为 None，则设置为 max(100, 10*len(x0))
    if maxfun is None:
        maxfun = max(100, 10*len(x0))

    # 调用 moduleTNC 中的 tnc_minimize 函数进行优化
    rc, nf, nit, x, funv, jacv = moduleTNC.tnc_minimize(
        func_and_grad, x0, low, up, scale,
        offset, messages, maxCGit, maxfun,
        eta, stepmx, accuracy, fmin, ftol,
        xtol, pgtol, rescale, callback
    )
    # 根据 TNC 文档，"输出的 x, f 和 g 可能由于缩放而略微不同步"，因此重新评估 func_and_grad 以使它们同步
    funv, jacv = func_and_grad(x)

    # 返回优化结果对象 OptimizeResult
    return OptimizeResult(x=x, fun=funv, jac=jacv, nfev=sf.nfev,
                          nit=nit, status=rc, message=RCSTRINGS[rc],
                          success=(-1 < rc < 3))
```