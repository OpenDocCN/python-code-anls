# `D:\src\scipysrc\scipy\scipy\optimize\_root.py`

```
"""
Unified interfaces to root finding algorithms.

Functions
---------
- root : find a root of a vector function.
"""
# 定义一个包含所有公开函数名的列表，只包含 root 函数
__all__ = ['root']

# 导入必要的库和模块
import numpy as np

from warnings import warn

# 导入优化相关模块和函数
from ._optimize import MemoizeJac, OptimizeResult, _check_unknown_options
from ._minpack_py import _root_hybr, leastsq
from ._spectral import _root_df_sane
from . import _nonlin as nonlin

# 定义支持的所有求根方法的列表
ROOT_METHODS = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson',
                'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov',
                'df-sane']


def root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None,
         options=None):
    r"""
    Find a root of a vector function.

    Parameters
    ----------
    fun : callable
        A vector function to find a root of.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its Jacobian.
    method : str, optional
        Type of solver. Should be one of

            - 'hybr'             :ref:`(see here) <optimize.root-hybr>`
            - 'lm'               :ref:`(see here) <optimize.root-lm>`
            - 'broyden1'         :ref:`(see here) <optimize.root-broyden1>`
            - 'broyden2'         :ref:`(see here) <optimize.root-broyden2>`
            - 'anderson'         :ref:`(see here) <optimize.root-anderson>`
            - 'linearmixing'     :ref:`(see here) <optimize.root-linearmixing>`
            - 'diagbroyden'      :ref:`(see here) <optimize.root-diagbroyden>`
            - 'excitingmixing'   :ref:`(see here) <optimize.root-excitingmixing>`
            - 'krylov'           :ref:`(see here) <optimize.root-krylov>`
            - 'df-sane'          :ref:`(see here) <optimize.root-dfsane>`

    jac : bool or callable, optional
        If `jac` is a Boolean and is True, `fun` is assumed to return the
        value of Jacobian along with the objective function. If False, the
        Jacobian will be estimated numerically.
        `jac` can also be a callable returning the Jacobian of `fun`. In
        this case, it must accept the same arguments as `fun`.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    callback : function, optional
        Optional callback function. It is called on every iteration as
        ``callback(x, f)`` where `x` is the current solution and `f`
        the corresponding residual. For all methods but 'hybr' and 'lm'.
    options : dict, optional
        A dictionary of solver options. E.g., `xtol` or `maxiter`, see
        :obj:`show_options()` for details.

    Returns
    -------
    """
    # root 函数的具体实现，用于求解向量函数的根
    pass
    # sol : OptimizeResult
    # 表示一个优化结果，以 `OptimizeResult` 对象形式返回。
    # 重要的属性包括：`x` 表示解的数组，`success` 表示算法是否成功退出的布尔标志，
    # `message` 描述终止的原因。查看 `OptimizeResult` 了解其他属性的描述。

    # See also
    # --------
    # show_options : solvers 接受的额外选项

    # Notes
    # -----
    # 此部分描述可以通过 'method' 参数选择的可用求解器。
    # 默认方法为 *hybr*。

    # 方法 *hybr* 使用 MINPACK [1]_ 中实现的 Powell 混合方法的修改版本。

    # 方法 *lm* 使用 MINPACK [1]_ 中实现的 Levenberg-Marquardt 算法的修改版本，
    # 以最小二乘意义上求解非线性方程组。

    # 方法 *df-sane* 是一种无导数谱方法。[3]_

    # 方法 *broyden1*, *broyden2*, *anderson*, *linearmixing*,
    # *diagbroyden*, *excitingmixing*, *krylov* 都是不精确 Newton 方法，
    # 使用回溯或全线搜索 [2]_。每种方法对应特定的雅可比逼近。

    # - 方法 *broyden1* 使用 Broyden 第一雅可比逼近，被称为 Broyden 的好方法。
    # - 方法 *broyden2* 使用 Broyden 第二雅可比逼近，被称为 Broyden 的坏方法。
    # - 方法 *anderson* 使用 (扩展的) Anderson 混合方法。
    # - 方法 *Krylov* 使用 Krylov 逆雅可比逼近，适用于大规模问题。
    # - 方法 *diagbroyden* 使用对角 Broyden 雅可比逼近。
    # - 方法 *linearmixing* 使用标量雅可比逼近。
    # - 方法 *excitingmixing* 使用调整的对角雅可比逼近。

    # .. warning::
    #     方法 *diagbroyden*, *linearmixing* 和 *excitingmixing* 实现的算法
    #     可能对特定问题有用，但其是否有效可能极大地依赖于问题本身。

    # .. versionadded:: 0.11.0

    # References
    # ----------
    # .. [1] More, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.
    #    1980. User Guide for MINPACK-1.
    # .. [2] C. T. Kelley. 1995. Iterative Methods for Linear and Nonlinear
    #    Equations. Society for Industrial and Applied Mathematics.
    #    <https://archive.siam.org/books/kelley/fr16/>
    # .. [3] W. La Cruz, J.M. Martinez, M. Raydan. Math. Comp. 75, 1429 (2006).

    # Examples
    # --------
    # 下面的函数定义了一个非线性方程组及其雅可比矩阵。

    # >>> import numpy as np
    # >>> def fun(x):
    # ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    # ...             0.5 * (x[1] - x[0])**3 + x[1]]

    # >>> def jac(x):
    # ...     return np.array([[1 + 1.5 * (x[0] - x[1])**2,
    # ...                       -1.5 * (x[0] - x[1])**2],
    # ...                      [-1.5 * (x[1] - x[0])**2,
    """
    def _wrapped_fun(*fargs):
        """
        Wrapped `func` to track the number of times
        the function has been called.
        """
        _wrapped_fun.nfev += 1  # 增加函数调用次数计数器
        return fun(*fargs)  # 调用原始函数并返回结果

    _wrapped_fun.nfev = 0  # 初始化函数调用次数计数器为0

    if not isinstance(args, tuple):
        args = (args,)  # 如果参数不是元组，则转换为元组

    meth = method.lower()  # 将求解方法名转换为小写
    if options is None:
        options = {}  # 如果选项为空，则初始化为空字典

    if callback is not None and meth in ('hybr', 'lm'):
        warn('Method %s does not accept callback.' % method,
             RuntimeWarning, stacklevel=2)  # 如果回调函数不为空且方法是'hybr'或'lm'，则发出警告

    # fun also returns the Jacobian
    if not callable(jac) and meth in ('hybr', 'lm'):
        if bool(jac):  # 如果jac不是可调用的且方法是'hybr'或'lm'
            fun = MemoizeJac(fun)  # 将原始函数包装为能缓存雅可比矩阵的函数
            jac = fun.derivative  # 使用缓存的雅可比矩阵
        else:
            jac = None  # 否则，设置雅可比矩阵为None

    # set default tolerances
    ```
    # 如果容差参数不为 None，则进行以下操作
    if tol is not None:
        # 将 options 转换为字典类型（如果尚未是字典的话）
        options = dict(options)
        # 根据求解方法的不同，设置容差参数
        if meth in ('hybr', 'lm'):
            options.setdefault('xtol', tol)  # 设置 x 的容差为 tol
        elif meth in ('df-sane',):
            options.setdefault('ftol', tol)  # 设置函数值的容差为 tol
        elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
                      'diagbroyden', 'excitingmixing', 'krylov'):
            options.setdefault('xtol', tol)  # 设置 x 的容差为 tol
            options.setdefault('xatol', np.inf)  # 设置绝对 x 容差为无穷大
            options.setdefault('ftol', np.inf)  # 设置函数值容差为无穷大
            options.setdefault('fatol', np.inf)  # 设置绝对函数值容差为无穷大
    
    # 根据求解方法选择相应的求解函数并返回求解结果
    if meth == 'hybr':
        sol = _root_hybr(_wrapped_fun, x0, args=args, jac=jac, **options)
    elif meth == 'lm':
        sol = _root_leastsq(_wrapped_fun, x0, args=args, jac=jac, **options)
    elif meth == 'df-sane':
        # 若方法为 df-sane，发出未使用雅可比矩阵警告，并调用相应的求解函数
        _warn_jac_unused(jac, method)
        sol = _root_df_sane(_wrapped_fun, x0, args=args, callback=callback,
                            **options)
    elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
                  'diagbroyden', 'excitingmixing', 'krylov'):
        # 若方法属于非线性求解方法之一，发出未使用雅可比矩阵警告，并调用相应的求解函数
        _warn_jac_unused(jac, method)
        sol = _root_nonlin_solve(_wrapped_fun, x0, args=args, jac=jac,
                                 _method=meth, _callback=callback,
                                 **options)
    else:
        # 若方法不在已知求解方法列表中，则抛出 ValueError 异常
        raise ValueError('Unknown solver %s' % method)
    
    # 设置求解结果的评估函数计数为 _wrapped_fun 的评估函数计数
    sol.nfev = _wrapped_fun.nfev
    # 返回求解结果
    return sol
# 如果 `jac` 不是 None，则发出警告，说明使用的方法不需要雅可比矩阵（jac）。
def _warn_jac_unused(jac, method):
    if jac is not None:
        warn(f'Method {method} does not use the jacobian (jac).',
             RuntimeWarning, stacklevel=2)


# 使用 Levenberg-Marquardt 方法求解最小二乘问题。
# Options 参数包括了多个控制算法行为的选项。
def _root_leastsq(fun, x0, args=(), jac=None,
                  col_deriv=0, xtol=1.49012e-08, ftol=1.49012e-08,
                  gtol=0.0, maxiter=0, eps=0.0, factor=100, diag=None,
                  **unknown_options):
    nfev = 0
    # 包装函数 `fun`，用于跟踪函数调用次数。
    def _wrapped_fun(*fargs):
        nonlocal nfev
        nfev += 1
        return fun(*fargs)

    # 检查并处理未知的选项参数。
    _check_unknown_options(unknown_options)
    # 调用 `leastsq` 函数进行最小二乘解算。
    x, cov_x, info, msg, ier = leastsq(_wrapped_fun, x0, args=args,
                                       Dfun=jac, full_output=True,
                                       col_deriv=col_deriv, xtol=xtol,
                                       ftol=ftol, gtol=gtol,
                                       maxfev=maxiter, epsfcn=eps,
                                       factor=factor, diag=diag)
    # 构建优化结果对象 `OptimizeResult`。
    sol = OptimizeResult(x=x, message=msg, status=ier,
                         success=ier in (1, 2, 3, 4), cov_x=cov_x,
                         fun=info.pop('fvec'), method="lm")
    sol.update(info)
    sol.nfev = nfev
    return sol


# 非线性求解函数，可以使用不同的算法进行求解。
# Options 参数包含了多个控制算法行为的选项。
def _root_nonlin_solve(fun, x0, args=(), jac=None,
                       _callback=None, _method=None,
                       nit=None, disp=False, maxiter=None,
                       ftol=None, fatol=None, xtol=None, xatol=None,
                       tol_norm=None, line_search='armijo', jac_options=None,
                       **unknown_options):
    # 检查并处理未知的选项参数。
    _check_unknown_options(unknown_options)

    # 将 `fatol` 赋值给 `f_tol`，将 `ftol` 赋值给 `f_rtol`，
    # 将 `xatol` 赋值给 `x_tol`，将 `xtol` 赋值给 `x_rtol`。
    f_tol = fatol
    f_rtol = ftol
    x_tol = xatol
    x_rtol = xtol
    # 设置详细模式为传入的显示参数
    verbose = disp
    # 如果未提供雅可比选项，则将其设为一个空字典
    if jac_options is None:
        jac_options = dict()
    
    # 根据给定的方法名称选择相应的雅可比函数类，并创建其实例
    jacobian = {'broyden1': nonlin.BroydenFirst,
                'broyden2': nonlin.BroydenSecond,
                'anderson': nonlin.Anderson,
                'linearmixing': nonlin.LinearMixing,
                'diagbroyden': nonlin.DiagBroyden,
                'excitingmixing': nonlin.ExcitingMixing,
                'krylov': nonlin.KrylovJacobian
                }[_method]
    
    # 如果存在额外的参数
    if args:
        # 如果 jac 参数为 True，则定义一个只返回函数值的函数 f(x)
        if jac is True:
            def f(x):
                return fun(x, *args)[0]
        # 否则定义一个返回完整输出的函数 f(x)
        else:
            def f(x):
                return fun(x, *args)
    # 如果没有额外参数，则直接使用给定的函数 fun 作为 f(x)
    else:
        f = fun
    
    # 调用非线性求解器 nonlin_solve 进行求解
    # 使用给定的初始值 x0，雅可比函数 jacobian，以及其他参数进行求解
    x, info = nonlin.nonlin_solve(f, x0, jacobian=jacobian(**jac_options),
                                  iter=nit, verbose=verbose,
                                  maxiter=maxiter, f_tol=f_tol,
                                  f_rtol=f_rtol, x_tol=x_tol,
                                  x_rtol=x_rtol, tol_norm=tol_norm,
                                  line_search=line_search,
                                  callback=_callback, full_output=True,
                                  raise_exception=False)
    
    # 创建一个 OptimizeResult 对象 sol，包含求解结果 x 和使用的方法 _method
    sol = OptimizeResult(x=x, method=_method)
    # 将详细信息 info 更新到 sol 中
    sol.update(info)
    # 返回求解结果 sol
    return sol
def _root_broyden1_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.
            alpha : float, optional
                Initial guess for the Jacobian is (-1/alpha).
            reduction_method : str or tuple, optional
                Method used in ensuring that the rank of the Broyden
                matrix stays low. Can either be a string giving the
                name of the method, or a tuple of the form ``(method,
                param1, param2, ...)`` that gives the name of the
                method and values for additional parameters.

                Methods available:

                    - ``restart``
                        Drop all matrix columns. Has no
                        extra parameters.
                    - ``simple``
                        Drop oldest matrix column. Has no
                        extra parameters.
                    - ``svd``
                        Keep only the most significant SVD
                        components.

                        Extra parameters:

                            - ``to_retain``
                                Number of SVD components to
                                retain when rank reduction is done.
                                Default is ``max_rank - 2``.
            max_rank : int, optional
                Maximum rank for the Broyden matrix.
                Default is infinity (i.e., no rank reduction).

    Examples
    --------
    >>> def func(x):
    ...     return np.cos(x) + x[::-1] - [1, 2, 3, 4]
    ...
    >>> from scipy import optimize
    >>> res = optimize.root(func, [1, 1, 1, 1], method='broyden1', tol=1e-14)
    >>> x = res.x
    >>> x
    array([4.04674914, 3.91158389, 2.71791677, 1.61756251])
    """
    # 此函数文档定义了使用 Broyden 方法求解非线性方程组时可用的选项和示例
    pass
    # 导入 NumPy 库，通常用 np 别名表示
    import numpy as np
    
    # 假设 x 是一个数组或者列表，计算 x 中每个元素的余弦值并与 x 的逆序数组拼接
    # 返回结果是一个 NumPy 数组，包含每个元素的余弦值与 x 逆序后对应位置元素的和
    np.cos(x) + x[::-1]
# 定义一个函数 _root_broyden2_doc，用于描述 Broyden 方法的根的优化算法的参数选项
def _root_broyden2_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            Initial guess for the Jacobian is (-1/alpha).
        reduction_method : str or tuple, optional
            Method used in ensuring that the rank of the Broyden
            matrix stays low. Can either be a string giving the
            name of the method, or a tuple of the form ``(method,
            param1, param2, ...)`` that gives the name of the
            method and values for additional parameters.

            Methods available:

                - ``restart``
                    Drop all matrix columns. Has no
                    extra parameters.
                - ``simple``
                    Drop oldest matrix column. Has no
                    extra parameters.
                - ``svd``
                    Keep only the most significant SVD
                    components.

                    Extra parameters:

                        - ``to_retain``
                            Number of SVD components to
                            retain when rank reduction is done.
                            Default is ``max_rank - 2``.
        max_rank : int, optional
            Maximum rank for the Broyden matrix.
            Default is infinity (i.e., no rank reduction).
    """
    pass

# 定义一个函数 _root_anderson_doc，用于描述 Anderson 方法的根的优化算法的参数选项
def _root_anderson_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    """
    pass
    # ftol : float, optional
    #     相对残差的容差。如果省略，则不使用。

    # fatol : float, optional
    #     绝对残差（最大范数）的容差。如果省略，默认为 6e-6。

    # xtol : float, optional
    #     相对最小步长。如果省略，则不使用。

    # xatol : float, optional
    #     绝对最小步长，从雅可比近似中确定。如果步长小于此值，则优化被视为成功终止。如果省略，则不使用。

    # tol_norm : function(vector) -> scalar, optional
    #     用于收敛检查的范数。默认为最大范数。

    # line_search : {None, 'armijo' (default), 'wolfe'}, optional
    #     用于确定在雅可比近似给定方向上的步长的线搜索类型。默认为 'armijo'。

    # jac_options : dict, optional
    #     雅可比近似的选项。

    #     alpha : float, optional
    #         初始猜测的雅可比矩阵是 (-1/alpha)。
    #     M : float, optional
    #         保留的先前向量数量。默认为 5。
    #     w0 : float, optional
    #         数值稳定性的正则化参数。相对于单位值，良好的值大约为 0.01。
def _root_linearmixing_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            initial guess for the jacobian is (-1/alpha).
    """
    pass

def _root_diagbroyden_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            initial guess for the jacobian is (-1/alpha).
    """
    pass

def _root_excitingmixing_doc():
    """
    Options
    -------
    """
    pass
    nit : int, optional
        # Number of iterations for the optimization algorithm. Default behavior is to iterate until tolerances are met.
    disp : bool, optional
        # Whether to print optimization progress to the console during each iteration.
    maxiter : int, optional
        # Maximum number of iterations allowed before terminating the optimization process.
    ftol : float, optional
        # Relative tolerance level for the residual. Optimization stops if the relative change is below this threshold.
    fatol : float, optional
        # Absolute tolerance level for the residual. Optimization stops if the absolute change is below this threshold.
        # Default value is 6e-6.
    xtol : float, optional
        # Relative minimum step size for termination. Optimization halts if the relative step size is below this value.
    xatol : float, optional
        # Absolute minimum step size for termination, determined from the Jacobian approximation.
        # If the step size falls below this, the optimization is considered successful.
    tol_norm : function(vector) -> scalar, optional
        # Norm function used to determine convergence. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        # Type of line search method used to find the step size along the Jacobian direction.
        # Default is 'armijo'.
    jac_options : dict, optional
        # Options for the Jacobian approximation.
        alpha : float, optional
            # Initial value used in Jacobian approximation. The initial approximation is (-1/alpha).
        alphamax : float, optional
            # Maximum value for diagonal entries of the Jacobian. Values are constrained to [alpha, alphamax].
# 定义函数 _root_krylov_doc，用于生成根据不同参数选项配置的文档字符串
def _root_krylov_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        rdiff : float, optional
            Relative step size to use in numerical differentiation.
        method : str or callable, optional
            Krylov method to use to approximate the Jacobian.  Can be a string,
            or a function implementing the same interface as the iterative
            solvers in `scipy.sparse.linalg`. If a string, needs to be one of:
            ``'lgmres'``, ``'gmres'``, ``'bicgstab'``, ``'cgs'``, ``'minres'``,
            ``'tfqmr'``.

            The default is `scipy.sparse.linalg.lgmres`.
        inner_M : LinearOperator or InverseJacobian
            Preconditioner for the inner Krylov iteration.
            Note that you can use also inverse Jacobians as (adaptive)
            preconditioners. For example,

            >>> jac = BroydenFirst()
            >>> kjac = KrylovJacobian(inner_M=jac.inverse).

            If the preconditioner has a method named 'update', it will
            be called as ``update(x, f)`` after each nonlinear step,
            with ``x`` giving the current point, and ``f`` the current
            function value.
        inner_tol, inner_maxiter, ...
            Parameters to pass on to the "inner" Krylov solver.
            See `scipy.sparse.linalg.gmres` for details.
        outer_k : int, optional
            Size of the subspace kept across LGMRES nonlinear
            iterations.

            See `scipy.sparse.linalg.lgmres` for details.
    """
    pass
```