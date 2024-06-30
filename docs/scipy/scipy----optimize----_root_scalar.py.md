# `D:\src\scipysrc\scipy\scipy\optimize\_root_scalar.py`

```
"""
Unified interfaces to root finding algorithms for real or complex
scalar functions.

Functions
---------
- root : find a root of a scalar function.
"""
import numpy as np

# 导入 Python 模块中的函数和类
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative

# 定义公开的接口列表
__all__ = ['root_scalar']

# 支持的根查找方法的列表
ROOT_SCALAR_METHODS = ['bisect', 'brentq', 'brenth', 'ridder', 'toms748',
                       'newton', 'secant', 'halley']


class MemoizeDer:
    """Decorator that caches the value and derivative(s) of function each
    time it is called.

    This is a simplistic memoizer that calls and caches a single value
    of ``f(x, *args)``.
    It assumes that `args` does not change between invocations.
    It supports the use case of a root-finder where `args` is fixed,
    `x` changes, and only rarely, if at all, does x assume the same value
    more than once."""
    
    def __init__(self, fun):
        self.fun = fun
        self.vals = None
        self.x = None
        self.n_calls = 0

    def __call__(self, x, *args):
        r"""Calculate f or use cached value if available"""
        # 如果没有缓存值或者 x 值发生了变化，则重新计算函数值和导数
        if self.vals is None or x != self.x:
            fg = self.fun(x, *args)
            self.x = x
            self.n_calls += 1
            self.vals = fg[:]
        return self.vals[0]

    def fprime(self, x, *args):
        r"""Calculate f' or use a cached value if available"""
        # 如果没有缓存值或者 x 值发生了变化，则重新计算一阶导数
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[1]

    def fprime2(self, x, *args):
        r"""Calculate f'' or use a cached value if available"""
        # 如果没有缓存值或者 x 值发生了变化，则重新计算二阶导数
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[2]

    def ncalls(self):
        # 返回函数调用次数
        return self.n_calls


def root_scalar(f, args=(), method=None, bracket=None,
                fprime=None, fprime2=None,
                x0=None, x1=None,
                xtol=None, rtol=None, maxiter=None,
                options=None):
    """
    Find a root of a scalar function.

    Parameters
    ----------
    f : callable
        A function to find a root of.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative(s).
    method : str, optional
        Type of solver.  Should be one of

            - 'bisect'    :ref:`(see here) <optimize.root_scalar-bisect>`
            - 'brentq'    :ref:`(see here) <optimize.root_scalar-brentq>`
            - 'brenth'    :ref:`(see here) <optimize.root_scalar-brenth>`
            - 'ridder'    :ref:`(see here) <optimize.root_scalar-ridder>`
            - 'toms748'    :ref:`(see here) <optimize.root_scalar-toms748>`
            - 'newton'    :ref:`(see here) <optimize.root_scalar-newton>`
            - 'secant'    :ref:`(see here) <optimize.root_scalar-secant>`
            - 'halley'    :ref:`(see here) <optimize.root_scalar-halley>`
    """
    # 这个函数实现了查找标量函数的根的功能
    pass
    # bracket: A sequence of 2 floats, optional
    # An interval bracketing a root. `f(x, *args)` must have different
    # signs at the two endpoints.

    # x0 : float, optional
    # Initial guess.

    # x1 : float, optional
    # A second guess.

    # fprime : bool or callable, optional
    # If `fprime` is a boolean and is True, `f` is assumed to return the
    # value of the objective function and of the derivative.
    # `fprime` can also be a callable returning the derivative of `f`. In
    # this case, it must accept the same arguments as `f`.

    # fprime2 : bool or callable, optional
    # If `fprime2` is a boolean and is True, `f` is assumed to return the
    # value of the objective function and of the first and second derivatives.
    # `fprime2` can also be a callable returning the second derivative of `f`.
    # In this case, it must accept the same arguments as `f`.

    # xtol : float, optional
    # Tolerance (absolute) for termination.

    # rtol : float, optional
    # Tolerance (relative) for termination.

    # maxiter : int, optional
    # Maximum number of iterations.

    # options : dict, optional
    # A dictionary of solver options. E.g., `k`, see
    # :obj:`show_options()` for details.

    # Returns
    # -------
    # sol : RootResults
    # The solution represented as a `RootResults` object.
    # Important attributes are: `root` the solution , `converged` a
    # boolean flag indicating if the algorithm exited successfully and
    # `flag` which describes the cause of the termination. See
    # `RootResults` for a description of other attributes.

    # See also
    # --------
    # show_options : Additional options accepted by the solvers
    # root : Find a root of a vector function.

    # Notes
    # -----
    # This section describes the available solvers that can be selected by the
    # 'method' parameter.

    # The default is to use the best method available for the situation
    # presented.
    # If a bracket is provided, it may use one of the bracketing methods.
    # If a derivative and an initial value are specified, it may
    # select one of the derivative-based methods.
    # If no method is judged applicable, it will raise an Exception.

    # Arguments for each method are as follows (x=required, o=optional).

    # +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    # |                    method                     | f | args | bracket | x0 | x1 | fprime | fprime2 | xtol | rtol | maxiter | options |
    # +===============================================+===+======+=========+====+====+========+=========+======+======+=========+=========+
    # | :ref:`bisect <optimize.root_scalar-bisect>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    # ```
    # 导入 optimize 模块中的所有内容
    from scipy import optimize
    
    # 定义一个简单的立方方程 f(x) = x^3 - 1，用于寻找其根
    def f(x):
        return (x**3 - 1)  # 只有一个实根在 x = 1 处
    
    # 定义 f(x) 的导数函数 fprime(x) = 3*x^2
    def fprime(x):
        return 3*x**2
    
    # 使用 brentq 方法寻找函数 f 在指定区间 [0, 3] 内的根
    sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')
    # 打印找到的根、迭代次数和函数调用次数
    sol.root, sol.iterations, sol.function_calls
    
    # 使用 newton 方法，从初始点 x0=0.2 开始，使用函数 f 和其导数 fprime 来寻找根
    sol = optimize.root_scalar(f, x0=0.2, fprime=fprime, method='newton')
    # 打印找到的根、迭代次数和函数调用次数
    sol.root, sol.iterations, sol.function_calls
    
    # 定义一个同时返回函数值及其导数和二阶导数的函数 f_p_pp(x) = (x^3 - 1), 3*x^2, 6*x
    def f_p_pp(x):
        return (x**3 - 1), 3*x**2, 6*x
    
    # 使用 newton 方法，从初始点 x0=0.2 开始，使用函数 f_p_pp，并指定 fprime=True 来寻找根
    sol = optimize.root_scalar(
        f_p_pp, x0=0.2, fprime=True, method='newton'
    )
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 11)

这行代码是一个示例，展示了如何使用 `optimize.root_scalar` 函数计算得到的解 `sol` 的根、迭代次数和函数调用次数。


    >>> sol = optimize.root_scalar(
    ...     f_p_pp, x0=0.2, fprime=True, fprime2=True, method='halley'
    ... )

调用 `optimize.root_scalar` 函数，使用 `halley` 方法求解方程 `f_p_pp` 的根，起始点 `x0` 为 `0.2`，同时指定了一阶导数 `fprime` 和二阶导数 `fprime2`。


    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 7, 8)

再次展示了使用 `optimize.root_scalar` 函数计算得到的解 `sol` 的根、迭代次数和函数调用次数，这次是 `halley` 方法的结果。

"""
if not isinstance(args, tuple):
    args = (args,)

如果 `args` 不是元组类型，则将其转换为单元素元组。


if options is None:
    options = {}

如果 `options` 参数为 `None`，则将其设为一个空字典，以便后续操作。


# fun also returns the derivative(s)
is_memoized = False
if fprime2 is not None and not callable(fprime2):
    if bool(fprime2):
        f = MemoizeDer(f)
        is_memoized = True
        fprime2 = f.fprime2
        fprime = f.fprime
    else:
        fprime2 = None
if fprime is not None and not callable(fprime):
    if bool(fprime):
        f = MemoizeDer(f)
        is_memoized = True
        fprime = f.fprime
    else:
        fprime = None

根据给定的 `fprime` 和 `fprime2` 参数，检查它们是否可调用，如果不可调用且为真值，则使用 `MemoizeDer` 对象对 `f` 进行装饰，并更新 `fprime` 和 `fprime2`。


# respect solver-specific default tolerances - only pass in if actually set
kwargs = {}
for k in ['xtol', 'rtol', 'maxiter']:
    v = locals().get(k)
    if v is not None:
        kwargs[k] = v

根据特定求解器的默认公差设置，只有在这些公差确实设置了的情况下才传递进去。


# Set any solver-specific options
if options:
    kwargs.update(options)
# Always request full_output from the underlying method as _root_scalar
# always returns a RootResults object
kwargs.update(full_output=True, disp=False)

设置任何特定求解器的选项，并总是请求底层方法返回完整的输出，并禁用显示（`disp=False`）。


# Pick a method if not specified.
# Use the "best" method available for the situation.
if not method:
    if bracket:
        method = 'brentq'
    elif x0 is not None:
        if fprime:
            if fprime2:
                method = 'halley'
            else:
                method = 'newton'
        elif x1 is not None:
            method = 'secant'
        else:
            method = 'newton'
if not method:
    raise ValueError('Unable to select a solver as neither bracket '
                     'nor starting point provided.')

如果未指定求解方法，则根据情况选择最佳方法。首先检查 `bracket` 参数，然后根据起始点 `x0`、一阶导数 `fprime` 和二阶导数 `fprime2` 的存在与否来选择适当的方法。


meth = method.lower()
map2underlying = {'halley': 'newton', 'secant': 'newton'}

try:
    methodc = getattr(optzeros, map2underlying.get(meth, meth))
except AttributeError as e:
    raise ValueError('Unknown solver %s' % meth) from e

将求解方法名称转换为小写，并尝试从 `optzeros` 模块中获取对应的求解器方法，如果找不到则引发 `ValueError` 异常。
    # 检查求解方法是否为二分法、Ridder 方法、Brentq 方法、Brenth 方法或者 Toms748 方法
    if meth in ['bisect', 'ridder', 'brentq', 'brenth', 'toms748']:
        # 如果 bracket 不是 list、tuple 或者 ndarray 类型，则抛出 ValueError 异常
        if not isinstance(bracket, (list, tuple, np.ndarray)):
            raise ValueError('Bracket needed for %s' % method)

        # 从 bracket 中取出前两个元素作为 a 和 b
        a, b = bracket[:2]
        try:
            # 调用 methodc 方法求解方程 f(x)=0，在区间 [a, b] 上
            r, sol = methodc(f, a, b, args=args, **kwargs)
        except ValueError as e:
            # 如果捕获到 ValueError 异常，则处理异常情况
            # gh-17622 通过引发错误（而不是返回不正确的结果）来修复低级求解器中的一些错误，
            # 当可调用函数返回 NaN 时。它通过包装可调用函数而不是修改编译代码来实现这一点，因此迭代计数不可用。
            if hasattr(e, "_x"):
                # 创建 RootResults 对象，传入异常信息的相关参数
                sol = optzeros.RootResults(root=e._x,
                                           iterations=np.nan,
                                           function_calls=e._function_calls,
                                           flag=str(e), method=method)
            else:
                raise

    # 如果求解方法是 secant 方法
    elif meth in ['secant']:
        # 如果 x0 为 None，则抛出 ValueError 异常
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        # 如果 kwargs 中有 'xtol' 参数，则将其重命名为 'tol'
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        # 调用 methodc 方法使用 secant 方法求解方程 f(x)=0
        r, sol = methodc(f, x0, args=args, fprime=None, fprime2=None,
                         x1=x1, **kwargs)

    # 如果求解方法是 newton 方法
    elif meth in ['newton']:
        # 如果 x0 为 None，则抛出 ValueError 异常
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        # 如果 fprime 为 None，则定义一个使用有限差分法近似 fprime 的函数
        if not fprime:
            def fprime(x, *args):
                # `root_scalar` 实际上似乎不支持 `newton` 的向量化使用。
                # 在这种情况下，`approx_derivative` 总是会得到标量输入。
                # 尽管如此，它总是返回一个数组，所以我们提取元素以产生标量输出。
                return approx_derivative(f, x, method='2-point', args=args)[0]

        # 如果 kwargs 中有 'xtol' 参数，则将其重命名为 'tol'
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        # 调用 methodc 方法使用 newton 方法求解方程 f(x)=0
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=None,
                         **kwargs)

    # 如果求解方法是 halley 方法
    elif meth in ['halley']:
        # 如果 x0 为 None，则抛出 ValueError 异常
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        # 如果 fprime 为 None，则抛出 ValueError 异常
        if not fprime:
            raise ValueError('fprime must be specified for %s' % method)
        # 如果 fprime2 为 None，则抛出 ValueError 异常
        if not fprime2:
            raise ValueError('fprime2 must be specified for %s' % method)
        # 如果 kwargs 中有 'xtol' 参数，则将其重命名为 'tol'
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        # 调用 methodc 方法使用 halley 方法求解方程 f(x)=0
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=fprime2, **kwargs)

    # 如果未知求解方法，则抛出 ValueError 异常
    else:
        raise ValueError('Unknown solver %s' % method)

    # 如果设置了 is_memoized 标志
    if is_memoized:
        # 替换 function_calls 计数为 memoized 计数，避免重复计数
        n_calls = f.n_calls
        sol.function_calls = n_calls

    # 返回求解结果 sol
    return sol
# 定义函数 _root_scalar_brentq_doc()，用于文档说明，无实际功能
r"""
Options
-------
args : tuple, optional
    传递给目标函数的额外参数。
bracket: A sequence of 2 floats, optional
    包围根的区间。在两个端点处，``f(x, *args)`` 必须具有不同的符号。
xtol : float, optional
    终止的绝对容差。
rtol : float, optional
    终止的相对容差。
maxiter : int, optional
    最大迭代次数。
options: dict, optional
    指定任何未涵盖上述内容的方法特定选项

"""

# 定义函数 _root_scalar_brenth_doc()，用于文档说明，无实际功能
r"""
Options
-------
args : tuple, optional
    传递给目标函数的额外参数。
bracket: A sequence of 2 floats, optional
    包围根的区间。在两个端点处，``f(x, *args)`` 必须具有不同的符号。
xtol : float, optional
    终止的绝对容差。
rtol : float, optional
    终止的相对容差。
maxiter : int, optional
    最大迭代次数。
options: dict, optional
    指定任何未涵盖上述内容的方法特定选项。

"""

# 定义函数 _root_scalar_toms748_doc()，用于文档说明，无实际功能
r"""
Options
-------
args : tuple, optional
    传递给目标函数的额外参数。
bracket: A sequence of 2 floats, optional
    包围根的区间。在两个端点处，``f(x, *args)`` 必须具有不同的符号。
xtol : float, optional
    终止的绝对容差。
rtol : float, optional
    终止的相对容差。
maxiter : int, optional
    最大迭代次数。
options: dict, optional
    指定任何未涵盖上述内容的方法特定选项。

"""

# 定义函数 _root_scalar_secant_doc()，用于文档说明，无实际功能
r"""
Options
-------
args : tuple, optional
    传递给目标函数的额外参数。
xtol : float, optional
    终止的绝对容差。
rtol : float, optional
    终止的相对容差。
maxiter : int, optional
    最大迭代次数。
x0 : float, required
    初始猜测值。
x1 : float, required
    第二个猜测值。
options: dict, optional
    指定任何未涵盖上述内容的方法特定选项。

"""

# 定义函数 _root_scalar_newton_doc()，用于文档说明，无实际功能
r"""
Options
-------
args : tuple, optional
    传递给目标函数及其导数的额外参数。
xtol : float, optional
    终止的绝对容差。
rtol : float, optional
    终止的相对容差。
maxiter : int, optional
    最大迭代次数。
x0 : float, required
    初始猜测值。

"""
    # `fprime` 参数，用于指定是否返回目标函数的导数值
    # 如果 `fprime` 是布尔类型且为 True，则假设 `f` 返回目标函数及其导数值
    # `fprime` 也可以是一个可调用对象，用于返回 `f` 的导数。在这种情况下，它必须接受与 `f` 相同的参数。
    options: dict, optional
        # 用于指定特定于方法的选项，这些选项未在上述说明中涵盖。
# 定义一个函数 _root_scalar_halley_doc，该函数的作用是提供 Halley 方法的文档字符串说明
def _root_scalar_halley_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    fprime : bool or callable, required
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of derivative along with the objective function.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, required
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of 1st and 2nd derivatives along with the objective function.
        `fprime2` can also be a callable returning the 2nd derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


# 定义一个函数 _root_scalar_ridder_doc，该函数的作用是提供 Ridder 方法的文档字符串说明
def _root_scalar_ridder_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


# 定义一个函数 _root_scalar_bisect_doc，该函数的作用是提供 Bisect 方法的文档字符串说明
def _root_scalar_bisect_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass
```