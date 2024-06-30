# `D:\src\scipysrc\sympy\sympy\solvers\ode\ode.py`

```
r"""
This module contains :py:meth:`~sympy.solvers.ode.dsolve` and different helper
functions that it uses.

:py:meth:`~sympy.solvers.ode.dsolve` solves ordinary differential equations.
See the docstring on the various functions for their uses.  Note that partial
differential equations support is in ``pde.py``.  Note that hint functions
have docstrings describing their various methods, but they are intended for
internal use.  Use ``dsolve(ode, func, hint=hint)`` to solve an ODE using a
specific hint.  See also the docstring on
:py:meth:`~sympy.solvers.ode.dsolve`.

**Functions in this module**

    These are the user functions in this module:

    - :py:meth:`~sympy.solvers.ode.dsolve` - Solves ODEs.
    - :py:meth:`~sympy.solvers.ode.classify_ode` - Classifies ODEs into
      possible hints for :py:meth:`~sympy.solvers.ode.dsolve`.
    - :py:meth:`~sympy.solvers.ode.checkodesol` - Checks if an equation is the
      solution to an ODE.
    - :py:meth:`~sympy.solvers.ode.homogeneous_order` - Returns the
      homogeneous order of an expression.
    - :py:meth:`~sympy.solvers.ode.infinitesimals` - Returns the infinitesimals
      of the Lie group of point transformations of an ODE, such that it is
      invariant.
    - :py:meth:`~sympy.solvers.ode.checkinfsol` - Checks if the given infinitesimals
      are the actual infinitesimals of a first order ODE.

    These are the non-solver helper functions that are for internal use.  The
    user should use the various options to
    :py:meth:`~sympy.solvers.ode.dsolve` to obtain the functionality provided
    by these functions:

    - :py:meth:`~sympy.solvers.ode.ode.odesimp` - Does all forms of ODE
      simplification.
    - :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity` - A key function for
      comparing solutions by simplicity.
    - :py:meth:`~sympy.solvers.ode.constantsimp` - Simplifies arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode.constant_renumber` - Renumber arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode._handle_Integral` - Evaluate unevaluated
      Integrals.

    See also the docstrings of these functions.

**Currently implemented solver methods**

The following methods are implemented for solving ordinary differential
equations.  See the docstrings of the various hint functions for more
information on each (run ``help(ode)``):

  - 1st order separable differential equations.
  - 1st order differential equations whose coefficients or `dx` and `dy` are
    functions homogeneous of the same order.
  - 1st order exact differential equations.
  - 1st order linear differential equations.
  - 1st order Bernoulli differential equations.
  - Power series solutions for first order differential equations.
  - Lie Group method of solving first order differential equations.
  - 2nd order Liouville differential equations.
  - Power series solutions for second order differential equations

"""
    # Solve `n`th order linear homogeneous differential equation with constant coefficients.
    # Solve `n`th order linear inhomogeneous differential equation with constant coefficients
    # using the method of undetermined coefficients.
    # Solve `n`th order linear inhomogeneous differential equation with constant coefficients
    # using the method of variation of parameters.
# 模块的设计理念是允许添加新的常微分方程（ODE）求解方法，而无需修改其他方法的求解代码。
# 核心函数是 `sympy.solvers.ode.classify_ode`，它接受一个ODE并告诉你哪些提示可以解决该ODE，而无需尝试解方程本身，因此速度较快。
# 每种求解方法都被视为一种提示（hint），并且有其自己的函数，命名为 `ode_<hint>`。
# 这些函数接受ODE和 `sympy.solvers.ode.classify_ode` 收集的匹配表达式，并返回一个求解结果。
# 如果结果包含积分，提示函数将返回一个未求值的 `sympy.integrals.integrals.Integral` 类。
# `sympy.solvers.ode.dsolve` 是用户包装函数，封装了所有这些操作，然后调用 `sympy.solvers.ode.ode.odesimp` 对结果进行简化处理。
# `odesimp` 的功能包括尝试为因变量（要解的函数）解方程，简化表达式中的任意常数，并在可能的情况下求解积分，如果提示允许的话。
# 作为一种最佳提示，如果包含“_Integral”可能会打乱“all_Integral”的目的，你需要在 :py:meth:`~sympy.solvers.ode.dsolve` 代码中手动移除它。
# 参见 :py:meth:`~sympy.solvers.ode.classify_ode` 的文档字符串，了解编写提示名称的准则。

# 在一般情况下，确定你的方法返回的解如何与可能解决相同常微分方程的其他方法进行比较。
# 然后，将你的提示放在 :py:data:`~sympy.solvers.ode.allhints` 元组中，按照应该调用它们的顺序排列。
# 这个元组的排序决定了哪些提示是默认的。注意，异常是可以接受的，因为用户可以使用 :py:meth:`~sympy.solvers.ode.dsolve` 选择单个提示。
# 通常情况下，“_Integral”变体应该放在列表的最后，“_best”变体应该放在它们适用的各种提示之前。
# 例如，“undetermined_coefficients”提示在“variation_of_parameters”提示之前，因为即使参数变分通常比不定系数更通用，不定系数对它能解决的常微分方程通常返回更干净的结果，并且不需要积分，因此速度更快。

# 接下来，你需要有一个匹配表达式或函数来匹配常微分方程的类型，你应该将它放在 :py:meth:`~sympy.solvers.ode.classify_ode` 中。
# 如果匹配函数不仅仅是几行代码，它应该尽可能地匹配常微分方程，以便 :py:meth:`~sympy.solvers.ode.classify_ode` 保持快速，并且不会受到解码中的错误的影响。记得考虑边界情况。
# 例如，如果你的解决方法涉及除以某个值，请确保排除除数为0的情况。

# 在大多数情况下，ODE 的匹配还将给出你解决它所需的各种部分。
# 你应该将其放在一个字典中（``.match()`` 将为你完成此操作），并将其添加为 ``matching_hints['hint'] = matchdict``，放在 :py:meth:`~sympy.solvers.ode.classify_ode` 的相关部分。
# :py:meth:`~sympy.solvers.ode.classify_ode` 将会把这些传递给 :py:meth:`~sympy.solvers.ode.dsolve`，后者将其作为 ``match`` 参数传递给你的函数。
# 你的函数应该命名为 ``ode_<hint>(eq, func, order, match)``。如果需要发送更多信息，请将其放入 ``match`` 字典中。
# 例如，如果在 :py:meth:`~sympy.solvers.ode.classify_ode` 中需要替换一个虚拟变量以匹配 ODE，你将需要使用 `match` 字典将其传递到你的函数中。
# 可以使用 ``func.args[0]`` 访问自变量，并且使用 ``func.func`` 访问因变量（即你要解的函数）。
# 如果在尝试解决 ODE 时发现无法解决，请引发 ``NotImplementedError``。
# :py:meth:`~sympy.solvers.ode.dsolve` 将使用 ``all`` 元提示捕获此错误，而不会导致整个例程失败。
# 为函数添加描述函数工作方式的文档字符串，包括如何在 SymPy 中添加 doctest，以及在 `test_ode.py` 中添加真实测试。
def your_function_name():
    """
    Add a docstring to your function that describes the method employed.  Like
    with anything else in SymPy, you will need to add a doctest to the docstring,
    in addition to real tests in ``test_ode.py``.  Try to maintain consistency
    with the other hint functions' docstrings.  Add your method to the list at the
    top of this docstring.  Also, add your method to ``ode.rst`` in the
    ``docs/src`` directory, so that the Sphinx docs will pull its docstring into
    the main SymPy documentation.  Be sure to make the Sphinx documentation by
    running ``make html`` from within the doc directory to verify that the
    docstring formats correctly.

    If your solution method involves integrating, use :py:obj:`~.Integral` instead of
    :py:meth:`~sympy.core.expr.Expr.integrate`.  This allows the user to bypass
    hard/slow integration by using the ``_Integral`` variant of your hint.  In
    most cases, calling :py:meth:`sympy.core.basic.Basic.doit` will integrate your
    solution.  If this is not the case, you will need to write special code in
    :py:meth:`~sympy.solvers.ode.ode._handle_Integral`.  Arbitrary constants should be
    symbols named ``C1``, ``C2``, and so on.  All solution methods should return
    an equality instance.  If you need an arbitrary number of arbitrary constants,
    you can use ``constants = numbered_symbols(prefix='C', cls=Symbol, start=1)``.
    If it is possible to solve for the dependent function in a general way, do so.
    Otherwise, do as best as you can, but do not call solve in your
    ``ode_<hint>()`` function.  :py:meth:`~sympy.solvers.ode.ode.odesimp` will attempt
    to solve the solution for you, so you do not need to do that.  Lastly, if your
    ODE has a common simplification that can be applied to your solutions, you can
    add a special case in :py:meth:`~sympy.solvers.ode.ode.odesimp` for it.  For
    example, solutions returned from the ``1st_homogeneous_coeff`` hints often
    have many :obj:`~sympy.functions.elementary.exponential.log` terms, so
    :py:meth:`~sympy.solvers.ode.ode.odesimp` calls
    :py:meth:`~sympy.simplify.simplify.logcombine` on them (it also helps to write
    the arbitrary constant as ``log(C1)`` instead of ``C1`` in this case).  Also
    consider common ways that you can rearrange your solution to have
    :py:meth:`~sympy.solvers.ode.constantsimp` take better advantage of it.  It is
    better to put simplification in :py:meth:`~sympy.solvers.ode.ode.odesimp` than in
    your method, because it can then be turned off with the simplify flag in
    :py:meth:`~sympy.solvers.ode.dsolve`.  If you have any extraneous
    simplification in your function, be sure to only run it using ``if
    match.get('simplify', True):``, especially if it can be slow or if it can
    reduce the domain of the solution.

    Finally, as with every contribution to SymPy, your method will need to be
    tested.  Add a test for each method in ``test_ode.py``.  Follow the
    conventions there, i.e., test the solver using ``dsolve(eq, f(x),
    hint=your_hint)``, and also test the solution using
    """
    pass
# 导入必要的模块和函数，从 sympy.core 开始
from sympy.core import Add, S, Mul, Pow, oo
# 导入容器类 Tuple
from sympy.core.containers import Tuple
# 导入表达式相关类 AtomicExpr, Expr
from sympy.core.expr import AtomicExpr, Expr
# 导入函数相关类 Function, Derivative, AppliedUndef 等
from sympy.core.function import Function, Derivative, AppliedUndef, diff, expand, expand_mul, Subs
# 导入多维向量处理相关函数 vectorize
from sympy.core.multidimensional import vectorize
# 导入数值相关类 Number, nan, zoo
from sympy.core.numbers import nan, zoo, Number
# 导入关系表达式类 Equality, Eq
from sympy.core.relational import Equality, Eq
# 导入排序函数 default_sort_key, ordered
from sympy.core.sorting import default_sort_key, ordered
# 导入符号类 Symbol,
#: 对于 :py:meth:`~sympy.solvers.ode.dsolve` 提供的特定ODE提示，可以被覆盖（见文档字符串）。
#:
#: 一般来说，``_Integral`` 提示会被分组放在列表的最后，除非有一个方法大部分时间返回一个无法求解的积分（这些通常也在列表末尾）。``default``、``all``、``best`` 和 ``all_Integral`` 元提示不应包含在此列表中，但 ``_best`` 和 ``_Integral`` 提示应该包含在内。
allhints = (
    "factorable",
    "nth_algebraic",
    "separable",
    "1st_exact",
    "1st_linear",
    "Bernoulli",
    "1st_rational_riccati",
    "Riccati_special_minus2",
    "1st_homogeneous_coeff_best",
    "1st_homogeneous_coeff_subs_indep_div_dep",
    "1st_homogeneous_coeff_subs_dep_div_indep",
    "almost_linear",
    "linear_coefficients",
    "separable_reduced",
    "1st_power_series",
    "lie_group",
    "nth_linear_constant_coeff_homogeneous",
    "nth_linear_euler_eq_homogeneous",
    "nth_linear_constant_coeff_undetermined_coefficients",
    "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients",
    "nth_linear_constant_coeff_variation_of_parameters",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters",
    "Liouville",
    "2nd_linear_airy",
    "2nd_linear_bessel",
    "2nd_hypergeometric",
    "2nd_hypergeometric_Integral",
    "nth_order_reducible",
    "2nd_power_series_ordinary",
    "2nd_power_series_regular",
    "nth_algebraic_Integral",
    "separable_Integral",
    "1st_exact_Integral",
    "1st_linear_Integral",
    "Bernoulli_Integral",
    "1st_homogeneous_coeff_subs_indep_div_dep_Integral",
    "1st_homogeneous_coeff_subs_dep_div_indep_Integral",
    "almost_linear_Integral",
    "linear_coefficients_Integral",
    "separable_reduced_Integral",
    "nth_linear_constant_coeff_variation_of_parameters_Integral",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters_Integral",
    "Liouville_Integral",
    "2nd_nonlinear_autonomous_conserved",
    "2nd_nonlinear_autonomous_conserved_Integral",
    )

def get_numbered_constants(eq, num=1, start=1, prefix='C'):
    """
    返回一个列表，包含在eq中尚未出现的常数。
    """

    ncs = iter_numbered_constants(eq, start, prefix)
    Cs = [next(ncs) for i in range(num)]
    return (Cs[0] if num == 1 else tuple(Cs))


def iter_numbered_constants(eq, start=1, prefix='C'):
    """
    返回一个迭代器，生成在eq中尚未出现的常数。
    """

    if isinstance(eq, (Expr, Eq)):
        eq = [eq]
    elif not iterable(eq):
        raise ValueError("Expected Expr or iterable but got %s" % eq)

    atom_set = set().union(*[i.free_symbols for i in eq])
    func_set = set().union(*[i.atoms(Function) for i in eq])
    if func_set:
        atom_set |= {Symbol(str(f.func)) for f in func_set}
    return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)


def dsolve(eq, func=None, hint="default", simplify=True,
    ics= None, xi=None, eta=None, x0=0, n=6, **kwargs):
    """
    解决任何（支持的）普通微分方程和普通微分方程系统。
    
    对于单个普通微分方程
    =======================
    
    当“eq”中的方程数量为一个时，它被分类为这种情况。
    **用法**
    
        ``dsolve(eq, f(x), hint)`` -> 解决函数``f(x)``的普通微分方程``eq``，使用方法``hint``。
    # `eq` 可以是任何支持的常微分方程，可以是一个相等关系 (`Equality`)，
    # 也可以是一个表达式，假定其等于 `0`。
    # 详见 sympy.solvers.ode 模块的文档，了解支持的解法方法。
    # 
    # `f(x)` 是一个关于一个变量的函数，其在该变量上的导数构成常微分方程 `eq`。
    # 在许多情况下，不必提供此函数；如果无法自动检测到，则会引发错误。
    # 
    # `hint` 是你想要 dsolve 使用的求解方法。使用 `classify_ode(eq, f(x))` 
    # 可以获取一个常微分方程的所有可能提示。默认提示 `default` 
    # 将使用 `classify_ode` 返回的第一个提示。更多提示选项请参见下面的提示部分。
    # 
    # `simplify` 启用通过 `odesimp` 进行简化。查看其文档以获取更多信息。
    # 例如，可以关闭此选项以禁用对 `func` 的解的求解或任意常数的简化。
    # 在启用此选项时，解可能包含比方程阶数更多的任意常数。
    # 
    # `xi` 和 `eta` 是普通微分方程的无穷小函数。它们是使得微分方程在Lie点变换群中
    # 保持不变的点变换的无穷小量。用户可以指定无穷小量的值。如果未指定任何值，
    # 将使用 `infinitesimals` 方法通过各种启发式方法计算 `xi` 和 `eta`。
    # 
    # `ics` 是微分方程的初始/边界条件集合。应以 `{f(x0): x1, f(x).diff(x).subs(x, x2): x3}` 
    # 的形式给出。对于幂级数解，如果未指定初始条件，则假定 `f(0)` 为 `C0`，
    # 并且将在0处计算幂级数解。
    # 
    # `x0` 是要评估微分方程幂级数解的点。
    # 
    # `n` 给出了要评估微分方程幂级数解的因变量的指数。
        **Hints**

        # 提示信息部分，介绍了在解常微分方程时使用的不同策略
        Aside from the various solving methods, there are also some meta-hints
        that you can pass to :py:meth:`~sympy.solvers.ode.dsolve`:

        ``default``:
                # 默认提示，使用 :py:meth:`~sympy.solvers.ode.classify_ode` 返回的第一个提示。
                This uses whatever hint is returned first by
                :py:meth:`~sympy.solvers.ode.classify_ode`. This is the
                default argument to :py:meth:`~sympy.solvers.ode.dsolve`.

        ``all``:
                # 应用所有相关的分类提示来解决方程，返回一个包含 ``hint:solution`` 对的字典。
                To make :py:meth:`~sympy.solvers.ode.dsolve` apply all
                relevant classification hints, use ``dsolve(ODE, func,
                hint="all")``.  This will return a dictionary of
                ``hint:solution`` terms.  If a hint causes dsolve to raise the
                ``NotImplementedError``, value of that hint's key will be the
                exception object raised.  The dictionary will also include
                some special keys:

                - ``order``: The order of the ODE.  See also
                  :py:meth:`~sympy.solvers.deutils.ode_order` in
                  ``deutils.py``.
                - ``best``: The simplest hint; what would be returned by
                  ``best`` below.
                - ``best_hint``: The hint that would produce the solution
                  given by ``best``.  If more than one hint produces the best
                  solution, the first one in the tuple returned by
                  :py:meth:`~sympy.solvers.ode.classify_ode` is chosen.
                - ``default``: The solution that would be returned by default.
                  This is the one produced by the hint that appears first in
                  the tuple returned by
                  :py:meth:`~sympy.solvers.ode.classify_ode`.

        ``all_Integral``:
                # 与 ``all`` 类似，但如果提示还有对应的 ``_Integral`` 提示，则只返回 ``_Integral`` 提示。
                This is the same as ``all``, except if a hint also has a
                corresponding ``_Integral`` hint, it only returns the
                ``_Integral`` hint.  This is useful if ``all`` causes
                :py:meth:`~sympy.solvers.ode.dsolve` to hang because of a
                difficult or impossible integral.  This meta-hint will also be
                much faster than ``all``, because
                :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive
                routine.

        ``best``:
                # 尝试所有方法并返回最简单的解决方案。
                To have :py:meth:`~sympy.solvers.ode.dsolve` try all methods
                and return the simplest one.  This takes into account whether
                the solution is solvable in the function, whether it contains
                any Integral classes (i.e.  unevaluatable integrals), and
                which one is the shortest in size.

        See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for
        more info on hints, and the :py:mod:`~sympy.solvers.ode` docstring for
        a list of all supported hints.
    # 导入必要的库和函数
    from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
    from sympy.abc import x
    
    # 定义函数 f(x) 作为符号函数
    f = Function('f')
    
    # 使用 dsolve 函数解决二阶线性常系数齐次ODE
    # Derivative(f(x), x, x) + 9*f(x) 表示要解决的ODE
    # f(x) 是待解的函数
    # 返回结果是一个方程 Eq，表示 f(x) 的解
    dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))
    
    # 示例输出:
    # Eq(f(x), C1*sin(3*x) + C2*cos(3*x))
    >>> eq = sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x)
    >>> dsolve(eq, hint='1st_exact')
    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
    
    >>> dsolve(eq, hint='almost_linear')
    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
    
    >>> t = symbols('t')
    >>> x, y = symbols('x, y', cls=Function)
    >>> eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))
    >>> dsolve(eq)
    [Eq(x(t), C1*x0(t) + C2*x0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t)),
     Eq(y(t), C1*y0(t) + C2*(y0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t) +
     exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)))]
    
    >>> eq = (Eq(Derivative(x(t),t),x(t)*y(t)*sin(t)), Eq(Derivative(y(t),t),y(t)**2*sin(t)))
    >>> dsolve(eq)
    {Eq(x(t), -exp(C1)/(C2*exp(C1) - cos(t))), Eq(y(t), -1/(C1 - cos(t)))}
    
    
    
    # 计算并解微分方程，返回解集合
    eq = sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x)
    # 解第一个微分方程，使用 '1st_exact' 提示
    dsolve(eq, hint='1st_exact')
    # 解第二个微分方程，使用 'almost_linear' 提示
    dsolve(eq, hint='almost_linear')
    
    # 定义符号变量 t
    t = symbols('t')
    # 定义函数变量 x(t) 和 y(t)
    x, y = symbols('x, y', cls=Function)
    # 定义一个包含两个微分方程的元组
    eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))
    # 解微分方程系统
    dsolve(eq)
    
    # 定义另一个包含两个微分方程的元组
    eq = (Eq(Derivative(x(t),t),x(t)*y(t)*sin(t)), Eq(Derivative(y(t),t),y(t)**2*sin(t)))
    # 解微分方程系统
    dsolve(eq)
    
    
    这段代码是一系列用来求解微分方程的示例，通过调用 `dsolve` 函数，并提供不同的提示 ('1st_exact', 'almost_linear') 来求解不同类型的微分方程。
    if iterable(eq):
        # 如果 eq 是可迭代的（即方程或方程组），则执行以下代码块

        from sympy.solvers.ode.systems import dsolve_system
        # 从 sympy.solvers.ode.systems 导入 dsolve_system 函数

        # 这部分可能在将来需要更改，特别是当我们有弱连接和强连接组件时。
        # 当我们需要展示尚未解决的系统时，这部分需要更改。
        try:
            # 尝试使用 dsolve_system 求解方程组 eq，使用给定的函数 func 和初始条件 ics
            sol = dsolve_system(eq, funcs=func, ics=ics, doit=True)
            return sol[0] if len(sol) == 1 else sol
            # 如果只有一个解，则返回该解；否则返回所有解
        except NotImplementedError:
            # 如果 dsolve_system 抛出 NotImplementedError 异常，则执行以下代码块
            pass

        # 根据方程 eq 和函数 func 进行分类
        match = classify_sysode(eq, func)

        # 更新 eq, func 和 t 到匹配的系统方程
        eq = match['eq']
        order = match['order']
        func = match['func']
        t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
        # 从第一个方程中获取导数符号，并找出其中的符号

        # 保持最高阶项的系数为正数
        for i in range(len(eq)):
            for func_ in func:
                if isinstance(func_, list):
                    pass
                else:
                    # 如果方程 eq[i] 中导数的系数为负数，则取相反数
                    if eq[i].coeff(diff(func[i],t,ode_order(eq[i], func[i]))).is_negative:
                        eq[i] = -eq[i]
        match['eq'] = eq

        # 如果方程组中方程的阶数不相等，则抛出 ValueError
        if len(set(order.values())) != 1:
            raise ValueError("It solves only those systems of equations whose orders are equal")
        match['order'] = list(order.values())[0]

        # 定义一个递归计算长度的函数 recur_len
        def recur_len(l):
            return sum(recur_len(item) if isinstance(item,list) else 1 for item in l)

        # 如果函数列表 func 的长度不等于方程列表 eq 的长度，则抛出 ValueError
        if recur_len(func) != len(eq):
            raise ValueError("dsolve() and classify_sysode() work with "
            "number of functions being equal to number of equations")

        # 如果方程的类型未知，则抛出 NotImplementedError
        if match['type_of_equation'] is None:
            raise NotImplementedError
        else:
            # 根据方程的线性性质选择求解函数
            if match['is_linear'] == True:
                solvefunc = globals()['sysode_linear_%(no_of_equation)seq_order%(order)s' % match]
            else:
                solvefunc = globals()['sysode_nonlinear_%(no_of_equation)seq_order%(order)s' % match]
            # 使用选择的求解函数求解方程组
            sols = solvefunc(match)

            # 如果有初始条件 ics，则求解方程组的常数项，并将其代入求解的结果中
            if ics:
                constants = Tuple(*sols).free_symbols - Tuple(*eq).free_symbols
                solved_constants = solve_ics(sols, func, constants, ics)
                return [sol.subs(solved_constants) for sol in sols]
            # 返回求解的结果 sols
            return sols
    else:
        given_hint = hint  # 用户提供的提示信息

        # 查看 _desolve 的文档字符串以获取更多细节。
        # 调用 _desolve 函数，求解微分方程 eq，返回提示信息和求解结果
        hints = _desolve(eq, func=func,
            hint=hint, simplify=True, xi=xi, eta=eta, type='ode', ics=ics,
            x0=x0, n=n, **kwargs)
        # 从提示信息中取出已解析的微分方程 eq
        eq = hints.pop('eq', eq)
        # 获取是否需要全部提示信息的标志
        all_ = hints.pop('all', False)
        # 如果需要全部提示信息
        if all_:
            # 初始化返回结果字典和失败提示字典
            retdict = {}
            failed_hints = {}
            # 根据微分方程分类，获取有序提示信息
            gethints = classify_ode(eq, dict=True, hint='all')
            orderedhints = gethints['ordered_hints']
            # 遍历已解析的提示信息
            for hint in hints:
                try:
                    # 尝试简化微分方程的解
                    rv = _helper_simplify(eq, hint, hints[hint], simplify)
                except NotImplementedError as detail:
                    # 记录未实现的提示信息和详细信息
                    failed_hints[hint] = detail
                else:
                    # 将简化后的解存入返回字典中
                    retdict[hint] = rv
            # 获取主函数
            func = hints[hint]['func']

            # 找出最简单的微分方程解
            retdict['best'] = min(list(retdict.values()), key=lambda x:
                ode_sol_simplicity(x, func, trysolving=not simplify))
            # 如果给定的提示是 'best'，则返回最简单的解
            if given_hint == 'best':
                return retdict['best']
            # 遍历有序的提示信息，找到与最佳解匹配的提示
            for i in orderedhints:
                if retdict['best'] == retdict.get(i, None):
                    retdict['best_hint'] = i
                    break
            # 存储默认提示和提示顺序
            retdict['default'] = gethints['default']
            retdict['order'] = gethints['order']
            # 将未成功的提示信息合并到返回字典中
            retdict.update(failed_hints)
            return retdict

        else:
            # 键 'hint' 存储需要解决的提示信息
            hint = hints['hint']
            # 调用辅助函数，简化微分方程的解
            return _helper_simplify(eq, hint, hints, simplify, ics=ics)
def _helper_simplify(eq, hint, match, simplify=True, ics=None, **kwargs):
    r"""
    Helper function of dsolve that calls the respective
    :py:mod:`~sympy.solvers.ode` functions to solve for the ordinary
    differential equations. This minimizes the computation in calling
    :py:meth:`~sympy.solvers.deutils._desolve` multiple times.
    """
    # 从参数 `match` 中提取必要的信息
    r = match
    # 获取函数名和阶数
    func = r['func']
    order = r['order']
    # 根据提示选择匹配的解决函数
    match = r[hint]

    if isinstance(match, SingleODESolver):
        # 如果匹配的对象是 SingleODESolver 类型，则直接使用
        solvefunc = match
    elif hint.endswith('_Integral'):
        # 如果提示以 '_Integral' 结尾，则调用对应的函数
        solvefunc = globals()['ode_' + hint[:-len('_Integral')]]
    else:
        # 否则根据提示调用对应的函数
        solvefunc = globals()['ode_' + hint]

    free = eq.free_symbols
    # 定义一个函数，用于提取未知符号中与自由符号不同的符号
    cons = lambda s: s.free_symbols.difference(free)

    if simplify:
        # 如果需要简化
        # odesimp() 尝试集成（如果必要），应用 constantsimp()，尝试解决 func，并应用任何其他特定提示的简化
        if isinstance(solvefunc, SingleODESolver):
            sols = solvefunc.get_general_solution()
        else:
            sols = solvefunc(eq, func, order, match)
        if iterable(sols):
            rv = []
            for s in sols:
                # 对每个解进行简化操作
                simp = odesimp(eq, s, func, hint)
                if iterable(simp):
                    rv.extend(simp)
                else:
                    rv.append(simp)
        else:
            rv = odesimp(eq, sols, func, hint)
    else:
        # 如果不需要简化
        # 我们仍然想要集成（可以单独使用提示禁用它）
        if isinstance(solvefunc, SingleODESolver):
            exprs = solvefunc.get_general_solution(simplify=False)
        else:
            match['simplify'] = False  # 有些提示可以利用这个选项
            exprs = solvefunc(eq, func, order, match)
        if isinstance(exprs, list):
            # 对每个表达式应用 _handle_Integral 处理
            rv = [_handle_Integral(expr, func, hint) for expr in exprs]
        else:
            rv = _handle_Integral(exprs, func, hint)

    if isinstance(rv, list):
        # 确保 rv 中的每个元素都是等式类型，否则抛出断言错误
        assert all(isinstance(i, Eq) for i in rv), rv  # if not => internal error
        if simplify:
            # 如果需要简化，则移除冗余的解决方案
            rv = _remove_redundant_solutions(eq, rv, order, func.args[0])
        if len(rv) == 1:
            rv = rv[0]
    if ics and 'power_series' not in hint:
        # 如果存在初始条件且提示中不包含 'power_series'
        if isinstance(rv, (Expr, Eq)):
            # 如果 rv 是表达式或等式类型，则解决常数
            solved_constants = solve_ics([rv], [r['func']], cons(rv), ics)
            rv = rv.subs(solved_constants)
        else:
            rv1 = []
            for s in rv:
                try:
                    # 对每个解进行初始条件的解决
                    solved_constants = solve_ics([s], [r['func']], cons(s), ics)
                except ValueError:
                    continue
                rv1.append(s.subs(solved_constants))
            if len(rv1) == 1:
                return rv1[0]
            rv = rv1
    return rv


def solve_ics(sols, funcs, constants, ics):
    """
    Solve for the constants given initial conditions

    ``sols`` is a list of solutions.

    ``funcs`` is a list of functions.
    """
    # 解决给定初始条件的常数
    # ``sols`` 是解决方案的列表
    # ``funcs`` 是函数的列表
    """
    ``constants`` is a list of constants.

    ``ics`` is the set of initial/boundary conditions for the differential
    equation. It should be given in the form of ``{f(x0): x1,
    f(x).diff(x).subs(x, x2):  x3}`` and so on.

    Returns a dictionary mapping constants to values.
    ``solution.subs(constants)`` will replace the constants in ``solution``.

    Example
    =======
    >>> # From dsolve(f(x).diff(x) - f(x), f(x))
    >>> from sympy import symbols, Eq, exp, Function
    >>> from sympy.solvers.ode.ode import solve_ics
    >>> f = Function('f')
    >>> x, C1 = symbols('x C1')
    >>> sols = [Eq(f(x), C1*exp(x))]
    >>> funcs = [f(x)]
    >>> constants = [C1]
    >>> ics = {f(0): 2}
    >>> solved_constants = solve_ics(sols, funcs, constants, ics)
    >>> solved_constants
    {C1: 2}
    >>> sols[0].subs(solved_constants)
    Eq(f(x), 2*exp(x))

    """
    # Assume ics are of the form f(x0): value or Subs(diff(f(x), x, n), (x,
    # x0)): value (currently checked by classify_ode). To solve, replace x
    # with x0, f(x0) with value, then solve for constants. For f^(n)(x0),
    # differentiate the solution n times, so that f^(n)(x) appears.
    x = funcs[0].args[0]  # Extracts the independent variable from the function
    diff_sols = []  # Initialize empty list for differential solutions
    subs_sols = []  # Initialize empty list for substituted solutions
    diff_variables = set()  # Initialize an empty set for differential variables

    for funcarg, value in ics.items():
        if isinstance(funcarg, AppliedUndef):
            x0 = funcarg.args[0]  # Extract x0 from the AppliedUndef instance
            matching_func = [f for f in funcs if f.func == funcarg.func][0]  # Finds the matching function in funcs
            S = sols  # Use original solutions
        elif isinstance(funcarg, (Subs, Derivative)):
            if isinstance(funcarg, Subs):
                funcarg = funcarg.doit()  # Ensure Subs object is evaluated
            if isinstance(funcarg, Subs):
                deriv = funcarg.expr  # Extract the derivative expression
                x0 = funcarg.point[0]  # Extract x0 from the Subs object
                variables = funcarg.expr.variables  # Extract variables from the derivative
                matching_func = deriv  # Set matching_func to the derivative
            elif isinstance(funcarg, Derivative):
                deriv = funcarg  # Set deriv to the Derivative object
                x0 = funcarg.variables[0]  # Extract x0 from the Derivative object
                variables = (x,) * len(funcarg.variables)  # Set variables to (x, x, ..., x) based on the number of variables
                matching_func = deriv.subs(x0, x)  # Substitute x0 with x in the derivative
            for sol in sols:
                if sol.has(deriv.expr.func):
                    diff_sols.append(Eq(sol.lhs.diff(*variables), sol.rhs.diff(*variables)))  # Append differential solutions
            diff_variables.add(variables)  # Add variables to the set of differential variables
            S = diff_sols  # Use differential solutions
        else:
            raise NotImplementedError("Unrecognized initial condition")  # Raise error for unrecognized initial condition

        for sol in S:
            if sol.has(matching_func):
                sol2 = sol.subs(x, x0)  # Substitute x with x0 in the solution
                sol2 = sol2.subs(funcarg, value)  # Substitute funcarg with value in the solution
                # This check is necessary because of issue #15724
                if not isinstance(sol2, BooleanAtom) or not subs_sols:
                    subs_sols = [s for s in subs_sols if not isinstance(s, BooleanAtom)]  # Remove BooleanAtom instances from subs_sols
                    subs_sols.append(sol2)  # Append sol2 to subs_sols

    # TODO: Use solveset here
    try:
        # 尝试解决代换后的方程组，返回解为字典形式
        solved_constants = solve(subs_sols, constants, dict=True)
    except NotImplementedError:
        # 如果 solve 函数不支持的操作抛出 NotImplementedError 异常，
        # 将解集合置为空列表
        solved_constants = []

    # XXX: 无法区分解不存在是因为初始条件无效还是 solve 函数不够智能。
    # 如果能使用 solveset 函数，也许可以改进此处。
    # 目前，我们在这种情况下使用 NotImplementedError。
    if not solved_constants:
        # 如果解集合为空列表，抛出 ValueError 异常
        raise ValueError("Couldn't solve for initial conditions")

    if solved_constants == True:
        # 如果解集合为 True（非空），抛出 ValueError 异常
        raise ValueError("Initial conditions did not produce any solutions for constants. Perhaps they are degenerate.")

    if len(solved_constants) > 1:
        # 如果解集合中的解数量超过一个，抛出 NotImplementedError 异常
        raise NotImplementedError("Initial conditions produced too many solutions for constants")

    # 返回解集合中的第一个解
    return solved_constants[0]
# 定义函数 classify_ode，用于分类给定的常微分方程（ODE）
def classify_ode(eq, func=None, dict=False, ics=None, *, prep=True, xi=None, eta=None, n=None, **kwargs):
    """
    Returns a tuple of possible :py:meth:`~sympy.solvers.ode.dsolve`
    classifications for an ODE.

    The tuple is ordered so that first item is the classification that
    :py:meth:`~sympy.solvers.ode.dsolve` uses to solve the ODE by default.  In
    general, classifications at the near the beginning of the list will
    produce better solutions faster than those near the end, thought there are
    always exceptions.  To make :py:meth:`~sympy.solvers.ode.dsolve` use a
    different classification, use ``dsolve(ODE, func,
    hint=<classification>)``.  See also the
    :py:meth:`~sympy.solvers.ode.dsolve` docstring for different meta-hints
    you can use.

    If ``dict`` is true, :py:meth:`~sympy.solvers.ode.classify_ode` will
    return a dictionary of ``hint:match`` expression terms. This is intended
    for internal use by :py:meth:`~sympy.solvers.ode.dsolve`.  Note that
    because dictionaries are ordered arbitrarily, this will most likely not be
    in the same order as the tuple.

    You can get help on different hints by executing
    ``help(ode.ode_hintname)``, where ``hintname`` is the name of the hint
    without ``_Integral``.

    See :py:data:`~sympy.solvers.ode.allhints` or the
    :py:mod:`~sympy.solvers.ode` docstring for a list of all supported hints
    that can be returned from :py:meth:`~sympy.solvers.ode.classify_ode`.

    Notes
    =====

    These are remarks on hint names.

    ``_Integral``

        If a classification has ``_Integral`` at the end, it will return the
        expression with an unevaluated :py:class:`~.Integral`
        class in it.  Note that a hint may do this anyway if
        :py:meth:`~sympy.core.expr.Expr.integrate` cannot do the integral,
        though just using an ``_Integral`` will do so much faster.  Indeed, an
        ``_Integral`` hint will always be faster than its corresponding hint
        without ``_Integral`` because
        :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive routine.
        If :py:meth:`~sympy.solvers.ode.dsolve` hangs, it is probably because
        :py:meth:`~sympy.core.expr.Expr.integrate` is hanging on a tough or
        impossible integral.  Try using an ``_Integral`` hint or
        ``all_Integral`` to get it return something.

        Note that some hints do not have ``_Integral`` counterparts. This is
        because :py:func:`~sympy.integrals.integrals.integrate` is not used in
        solving the ODE for those method. For example, `n`\th order linear
        homogeneous ODEs with constant coefficients do not require integration
        to solve, so there is no
        ``nth_linear_homogeneous_constant_coeff_Integrate`` hint. You can
        easily evaluate any unevaluated
        :py:class:`~sympy.integrals.integrals.Integral`\s in an expression by
        doing ``expr.doit()``.
    """
    # 将输入的初始条件转换为符号表达式
    ics = sympify(ics)
    # 如果 func 存在且其参数个数不为1，则抛出 ValueError 异常
    if func and len(func.args) != 1:
        raise ValueError("dsolve() and classify_ode() only "
                         "work with functions of one variable, not %s" % func)

    # 如果 eq 是 Equality 类的实例，则将其转换为等式的左侧减去右侧的形式
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs

    # 保存原始的方程式，用于某些方法需要未处理的方程式
    eq_orig = eq

    # 如果 prep 为真或者 func 为 None，则调用 _preprocess 函数处理方程式和函数
    if prep or func is None:
        eq, func_ = _preprocess(eq, func)
        # 如果 func 为 None，则将其设置为处理后的函数 func_
        if func is None:
            func = func_

    # 获取函数的自变量 x
    x = func.args[0]
    # 获取函数的主体部分 f
    f = func.func
    # 创建一个虚拟变量 y，用于一些内部计算
    y = Dummy('y')
    # 设置默认项个数为5，或者根据参数 n 的值来决定
    terms = 5 if n is None else n

    # 计算方程式的阶数
    order = ode_order(eq, f(x))
    # 构建匹配提示字典，包含方程式的阶数和其他默认信息
    matching_hints = {"order": order}

    # 计算函数 f(x) 对 x 的导数
    df = f(x).diff(x)
    # 定义多个 Wild 类，用于匹配未知参数
    a = Wild('a', exclude=[f(x)])
    d = Wild('d', exclude=[df, f(x).diff(x, 2)])
    e = Wild('e', exclude=[df])
    n = Wild('n', exclude=[x, f(x), df])
    c1 = Wild('c1', exclude=[x])
    a3 = Wild('a3', exclude=[f(x), df, f(x).diff(x, 2)])
    b3 = Wild('b3', exclude=[f(x), df, f(x).diff(x, 2)])
    c3 = Wild('c3', exclude=[f(x), df, f(x).diff(x, 2)])

    # 创建一个空字典 boundary，用于提取初始条件
    boundary = {}

    # 创建符号 C1，用于表示常数
    C1 = Symbol("C1")

    # 预处理以提取初始条件
    # 如果传入的 ics 不为 None，则进行以下操作
    if ics is not None:
        # 遍历 ics 中的每一个元素 funcarg
        for funcarg in ics:
            # 如果 funcarg 是 Subs 或者 Derivative 类型的实例
            if isinstance(funcarg, (Subs, Derivative)):
                # 如果 funcarg 是 Subs 类型的实例
                if isinstance(funcarg, Subs):
                    # 获取导数表达式和替换前后的变量
                    deriv = funcarg.expr
                    old = funcarg.variables[0]
                    new = funcarg.point[0]
                # 如果 funcarg 是 Derivative 类型的实例
                elif isinstance(funcarg, Derivative):
                    # 设置默认值为 x，因为没有具体信息
                    old = x
                    # 获取导数表达式和替换前后的变量
                    deriv = funcarg
                    new = funcarg.variables[0]

                # 检查是否符合特定的边界条件形式
                if (isinstance(deriv, Derivative) and isinstance(deriv.args[0],
                    AppliedUndef) and deriv.args[0].func == f and
                    len(deriv.args[0].args) == 1 and old == x and not
                    new.has(x) and all(i == deriv.variables[0] for i in
                    deriv.variables) and x not in ics[funcarg].free_symbols):

                    # 计算导数的阶数
                    dorder = ode_order(deriv, x)
                    # 生成一个临时的键 'f' + 阶数，并更新边界条件字典
                    temp = 'f' + str(dorder)
                    boundary.update({temp: new, temp + 'val': ics[funcarg]})
                else:
                    # 如果不符合条件，则引发值错误异常
                    raise ValueError("Invalid boundary conditions for Derivatives")

            # 如果 funcarg 是 AppliedUndef 类型的实例
            elif isinstance(funcarg, AppliedUndef):
                # 检查是否符合特定的边界条件形式
                if (funcarg.func == f and len(funcarg.args) == 1 and
                    not funcarg.args[0].has(x) and x not in ics[funcarg].free_symbols):
                    # 更新边界条件字典，设置 'f0' 为键
                    boundary.update({'f0': funcarg.args[0], 'f0val': ics[funcarg]})
                else:
                    # 如果不符合条件，则引发值错误异常
                    raise ValueError("Invalid boundary conditions for Function")

            else:
                # 如果 funcarg 类型不属于 Subs、Derivative 或 AppliedUndef，则引发值错误异常
                raise ValueError("Enter boundary conditions of the form ics={f(point): value, f(x).diff(x, order).subs(x, point): value}")

    # 创建 SingleODEProblem 的实例对象，使用给定的参数初始化
    ode = SingleODEProblem(eq_orig, func, x, prep=prep, xi=xi, eta=eta)
    # 获取 kwargs 中键为 'hint' 的值，如果不存在，则使用默认值 'default'
    user_hint = kwargs.get('hint', 'default')
    # 如果用户没有指定明确的提示，即 'default' 情况
    early_exit = (user_hint=='default')
    # 如果 user_hint 是以 '_Integral' 结尾，则去掉 '_Integral'
    if user_hint.endswith('_Integral'):
        user_hint = user_hint[:-len('_Integral')]
    # 设置 user_map 初始值为 solver_map
    user_map = solver_map
    # 如果用户指定了明确的提示，且该提示在 solver_map 中
    if user_hint not in ['default', 'all', 'all_Integral', 'best'] and user_hint in solver_map:
        # 将 user_map 更新为只包含指定的提示
        user_map = {user_hint: solver_map[user_hint]}

    # 遍历 user_map 中的每一个提示 hint
    for hint in user_map:
        # 使用 user_map[hint] 创建一个 solver 对象，解决 ODE 问题 ode
        solver = user_map[hint](ode)
        # 如果 solver 匹配成功
        if solver.matches():
            # 将匹配成功的 solver 添加到 matching_hints 字典中
            matching_hints[hint] = solver
            # 如果 solver 具有积分的能力，则将其也添加到 matching_hints 字典中
            if user_map[hint].has_integral:
                matching_hints[hint + "_Integral"] = solver
            # 如果 dict 存在并且 early_exit 为 True，则设置 'default' 键为 hint，并提前返回 matching_hints
            if dict and early_exit:
                matching_hints["default"] = hint
                return matching_hints

    # 对方程 eq 进行展开处理
    eq = expand(eq)
    # 尝试从最高阶导数中移除 f(x) 的前提条件
    # 初始化变量 reduced_eq 为 None
    reduced_eq = None
    # 如果方程 eq 是 Add 对象
    if eq.is_Add:
        # 计算方程关于 f(x) 的 order 阶导数的系数
        deriv_coef = eq.coeff(f(x).diff(x, order))
        # 如果导数系数不是 1 或者 0
        if deriv_coef not in (1, 0):
            # 尝试匹配 deriv_coef 是否符合形式 a*f(x)**c1
            r = deriv_coef.match(a*f(x)**c1)
            # 如果成功匹配并且 r[c1] 存在
            if r and r[c1]:
                # 构建分母 den 为 f(x) 的 r[c1] 次幂
                den = f(x)**r[c1]
                # 将方程 eq 中的每个参数除以 den，构成 reduced_eq
                reduced_eq = Add(*[arg/den for arg in eq.args])
    # 如果 reduced_eq 仍然是 None，则将其赋值为原始方程 eq
    if not reduced_eq:
        reduced_eq = eq

    # 如果 order 等于 1
    if order == 1:

        # 匹配方程 eq 关于 df 的 collect 形式 d + e * df
        r = collect(eq, df, exact=True).match(d + e * df)
        # 如果匹配成功
        if r:
            # 更新匹配结果中的变量名
            r['d'] = d
            r['e'] = e
            r['y'] = y
            # 将 r[d] 中的 f(x) 替换为 y
            r[d] = r[d].subs(f(x), y)
            # 将 r[e] 中的 f(x) 替换为 y
            r[e] = r[e].subs(f(x), y)

            # 检查 d/e 是否在某个给定点是解析的
            point = boundary.get('f0', 0)
            value = boundary.get('f0val', C1)
            check = cancel(r[d]/r[e])
            check1 = check.subs({x: point, y: value})
            # 检查 check1 中是否包含无穷大、无限大、NaN 或者 -无穷大
            if not check1.has(oo) and not check1.has(zoo) and \
                not check1.has(nan) and not check1.has(-oo):
                check2 = (check1.diff(x)).subs({x: point, y: value})
                # 检查 check2 中是否包含无穷大、无限大、NaN 或者 -无穷大
                if not check2.has(oo) and not check2.has(zoo) and \
                    not check2.has(nan) and not check2.has(-oo):
                    # 复制匹配结果 r 并更新 terms, f0, f0val
                    rseries = r.copy()
                    rseries.update({'terms': terms, 'f0': point, 'f0val': value})
                    # 将匹配到的一阶幂级数序列存入 matching_hints 字典中
                    matching_hints["1st_power_series"] = rseries
    elif order == 2:
        # 如果方程阶数为2
        # 形式为 a3*f(x).diff(x, 2) + b3*f(x).diff(x) + c3*f(x)
        # 在点 x0 处具有明确的幂级数解，条件是 b3/a3 和 c3/a3 在 x0 处是解析的。
        deq = a3*(f(x).diff(x, 2)) + b3*f(x).diff(x) + c3*f(x)
        # 对简化后的方程进行收集，匹配指定项
        r = collect(reduced_eq,
            [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
        ordinary = False
        if r:
            # 如果匹配成功，并且所有匹配到的项都是多项式
            if not all(r[key].is_polynomial() for key in r):
                n, d = reduced_eq.as_numer_denom()
                reduced_eq = expand(n)
                r = collect(reduced_eq,
                    [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
        if r and r[a3] != 0:
            # 计算 p 和 q 用于后续使用
            p = cancel(r[b3]/r[a3])  # 用于下面的计算
            q = cancel(r[c3]/r[a3])  # 用于下面的计算
            point = kwargs.get('x0', 0)
            check = p.subs(x, point)
            # 检查 p 在点 x0 处是否包含无穷大或非数字的特殊值
            if not check.has(oo, nan, zoo, -oo):
                check = q.subs(x, point)
                # 检查 q 在点 x0 处是否包含无穷大或非数字的特殊值
                if not check.has(oo, nan, zoo, -oo):
                    ordinary = True
                    r.update({'a3': a3, 'b3': b3, 'c3': c3, 'x0': point, 'terms': terms})
                    matching_hints["2nd_power_series_ordinary"] = r

            # 检查微分方程在 x0 处是否具有正则奇异点
            # 如果 (b3/a3)*(x - x0) 和 (c3/a3)*((x - x0)**2) 在 x0 处是解析的，则具有正则奇异点
            if not ordinary:
                p = cancel((x - point)*p)
                check = p.subs(x, point)
                if not check.has(oo, nan, zoo, -oo):
                    q = cancel(((x - point)**2)*q)
                    check = q.subs(x, point)
                    if not check.has(oo, nan, zoo, -oo):
                        coeff_dict = {'p': p, 'q': q, 'x0': point, 'terms': terms}
                        matching_hints["2nd_power_series_regular"] = coeff_dict

    # 根据所有的提示，按顺序排序键
    retlist = [i for i in allhints if i in matching_hints]
    if dict:
        # 由于字典是任意排序的，因此记录第一个用于 dsolve() 的提示。
        # 在 Python 3 中使用有序字典。
        matching_hints["default"] = retlist[0] if retlist else None
        matching_hints["ordered_hints"] = tuple(retlist)
        return matching_hints
    else:
        return tuple(retlist)
# 定义一个函数，用于分类描述常微分方程系统的参数和值
def classify_sysode(eq, funcs=None, **kwargs):
    """
    Returns a dictionary of parameter names and values that define the system
    of ordinary differential equations in ``eq``.
    The parameters are further used in
    :py:meth:`~sympy.solvers.ode.dsolve` for solving that system.

    Some parameter names and values are:

    'is_linear' (boolean), which tells whether the given system is linear.
    Note that "linear" here refers to the operator: terms such as ``x*diff(x,t)`` are
    nonlinear, whereas terms like ``sin(t)*diff(x,t)`` are still linear operators.

    'func' (list) contains the :py:class:`~sympy.core.function.Function`s that
    appear with a derivative in the ODE, i.e. those that we are trying to solve
    the ODE for.

    'order' (dict) with the maximum derivative for each element of the 'func'
    parameter.

    'func_coeff' (dict or Matrix) with the coefficient for each triple ``(equation number,
    function, order)```. The coefficients are those subexpressions that do not
    appear in 'func', and hence can be considered constant for purposes of ODE
    solving. The value of this parameter can also be a  Matrix if the system of ODEs are
    linear first order of the form X' = AX where X is the vector of dependent variables.
    Here, this function returns the coefficient matrix A.

    'eq' (list) with the equations from ``eq``, sympified and transformed into
    expressions (we are solving for these expressions to be zero).

    'no_of_equations' (int) is the number of equations (same as ``len(eq)``).

    'type_of_equation' (string) is an internal classification of the type of
    ODE.

    'is_constant' (boolean), which tells if the system of ODEs is constant coefficient
    or not. This key is temporary addition for now and is in the match dict only when
    the system of ODEs is linear first order constant coefficient homogeneous. So, this
    key's value is True for now if it is available else it does not exist.

    'is_homogeneous' (boolean), which tells if the system of ODEs is homogeneous. Like the
    key 'is_constant', this key is a temporary addition and it is True since this key value
    is available only when the system is linear first order constant coefficient homogeneous.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode-toc1.htm
    -A. D. Polyanin and A. V. Manzhirov, Handbook of Mathematics for Engineers and Scientists

    Examples
    ========

    >>> from sympy import Function, Eq, symbols, diff
    >>> from sympy.solvers.ode.ode import classify_sysode
    >>> from sympy.abc import t
    >>> f, x, y = symbols('f, x, y', cls=Function)
    >>> k, l, m, n = symbols('k, l, m, n', Integer=True)
    >>> x1 = diff(x(t), t) ; y1 = diff(y(t), t)
    >>> x2 = diff(x(t), t, t) ; y2 = diff(y(t), t, t)
    >>> eq = (Eq(x1, 12*x(t) - 6*y(t)), Eq(y1, 11*x(t) + 3*y(t)))
    >>> classify_sysode(eq)

    """
    """

    # Sympify equations and convert iterables of equations into
    # a list of equations
    # 将方程转换为SymPy对象，并将方程迭代器转换为方程列表
    def _sympify(eq):
        return list(map(sympify, eq if iterable(eq) else [eq]))

    # 将输入的方程组和函数符号列表转换为SymPy对象
    eq, funcs = (_sympify(w) for w in [eq, funcs])

    # 将等式形式的方程转换为一般形式，即将等式左右两边的差值作为方程
    for i, fi in enumerate(eq):
        if isinstance(fi, Equality):
            eq[i] = fi.lhs - fi.rhs

    # 获取方程中的自变量，这里假设自变量为 t
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]

    # 初始化匹配提示字典，设置方程数量信息
    matching_hints = {"no_of_equation":i+1}
    matching_hints['eq'] = eq

    # 如果方程数量为零，则抛出错误
    if i==0:
        raise ValueError("classify_sysode() works for systems of ODEs. "
        "For scalar ODEs, classify_ode should be used")

    # 如果未提供函数列表，则根据方程组提取所有函数
    order = {}
    if funcs==[None]:
        funcs = _extract_funcs(eq)

    # 去除重复的函数符号
    funcs = list(set(funcs))

    # 检查函数数量是否与方程数量一致，否则抛出错误
    if len(funcs) != len(eq):
        raise ValueError("Number of functions given is not equal to the number of equations %s" % funcs)

    # 初始化函数字典，用于存储每个函数及其对应的最高阶数
    func_dict = {}

    # 遍历所有函数，确定它们的最高阶数，并存储在函数字典中
    for func in funcs:
        if not order.get(func, False):
            max_order = 0
            for i, eqs_ in enumerate(eq):
                order_ = ode_order(eqs_,func)
                if max_order < order_:
                    max_order = order_
                    eq_no = i
            if eq_no in func_dict:
                func_dict[eq_no] = [func_dict[eq_no], func]
            else:
                func_dict[eq_no] = func
            order[func] = max_order

    # 将函数字典整理为函数列表
    funcs = [func_dict[i] for i in range(len(func_dict))]
    matching_hints['func'] = funcs

    # 检查每个函数是否只接受一个变量，否则抛出错误
    for func in funcs:
        if isinstance(func, list):
            for func_elem in func:
                if len(func_elem.args) != 1:
                    raise ValueError("dsolve() and classify_sysode() work with "
                    "functions of one variable only, not %s" % func)
        else:
            if func and len(func.args) != 1:
                raise ValueError("dsolve() and classify_sysode() work with "
                "functions of one variable only, not %s" % func)
    """
    # 将方程组中每个方程的阶数添加到匹配提示字典中
    matching_hints["order"] = order

    # 检查每个方程中函数 f(t)、diff(f(t),t) 及更高阶导数的系数
    # 同样地，对于其他函数 g(t)、diff(g(t),t)，以及所有方程中的这些函数
    # 这里 j 表示方程号，funcs[l] 表示当前讨论的函数，k 表示函数 funcs[l] 的阶数，
    # 我们正在计算其系数。
    def linearity_check(eqs, j, func, is_linear_):
        for k in range(order[func] + 1):
            # 收集方程式中关于 diff(func, t, k) 的系数
            func_coef[j, func, k] = collect(eqs.expand(), [diff(func, t, k)]).coeff(diff(func, t, k))
            if is_linear_ == True:
                # 如果是线性方程
                if func_coef[j, func, k] == 0:
                    if k == 0:
                        # 对于零阶导数的情况，检查非齐次项
                        coef = eqs.as_independent(func, as_Add=True)[1]
                        for xr in range(1, ode_order(eqs, func) + 1):
                            coef -= eqs.as_independent(diff(func, t, xr), as_Add=True)[1]
                        if coef != 0:
                            is_linear_ = False
                    else:
                        # 对于高阶导数的情况，检查非齐次项
                        if eqs.as_independent(diff(func, t, k), as_Add=True)[1]:
                            is_linear_ = False
                else:
                    # 检查系数中的依赖项
                    for func_ in funcs:
                        if isinstance(func_, list):
                            for elem_func_ in func_:
                                dep = func_coef[j, func, k].as_independent(elem_func_, as_Add=True)[1]
                                if dep != 0:
                                    is_linear_ = False
                        else:
                            dep = func_coef[j, func, k].as_independent(func_, as_Add=True)[1]
                            if dep != 0:
                                is_linear_ = False
        return is_linear_

    # 存储函数系数的字典
    func_coef = {}
    # 假设方程组是线性的
    is_linear = True
    # 遍历每个方程和每个函数，检查线性性
    for j, eqs in enumerate(eq):
        for func in funcs:
            if isinstance(func, list):
                for func_elem in func:
                    is_linear = linearity_check(eqs, j, func_elem, is_linear)
            else:
                is_linear = linearity_check(eqs, j, func, is_linear)
    
    # 将函数系数和线性性存储在匹配提示字典中
    matching_hints['func_coeff'] = func_coef
    matching_hints['is_linear'] = is_linear
    # 检查所有等式的解决顺序是否完全相同
    if len(set(order.values())) == 1:
        # 获取匹配提示中的解决顺序
        order_eq = list(matching_hints['order'].values())[0]
        # 如果等式是线性的
        if matching_hints['is_linear'] == True:
            # 如果等式数为2
            if matching_hints['no_of_equation'] == 2:
                # 如果解决顺序为1，调用线性二元一阶方程检查函数
                if order_eq == 1:
                    type_of_equation = check_linear_2eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            # 如果等式不匹配systems.py中的任何通用解法，并且等式数大于2，则应引发NotImplementedError
            else:
                type_of_equation = None

        # 如果等式不是线性的
        else:
            # 如果等式数为2
            if matching_hints['no_of_equation'] == 2:
                # 如果解决顺序为1，调用非线性二元一阶方程检查函数
                if order_eq == 1:
                    type_of_equation = check_nonlinear_2eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            # 如果等式数为3
            elif matching_hints['no_of_equation'] == 3:
                # 如果解决顺序为1，调用非线性三元一阶方程检查函数
                if order_eq == 1:
                    type_of_equation = check_nonlinear_3eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            else:
                type_of_equation = None
    else:
        type_of_equation = None

    # 将解方程类型存储在匹配提示字典中
    matching_hints['type_of_equation'] = type_of_equation

    # 返回匹配提示字典，包括解方程类型
    return matching_hints
# 检查线性一阶常微分方程组的特定类型条件，返回相应的类型或者None
def check_linear_2eq_order1(eq, func, func_coef):
    # 提取函数表达式中的变量 x 和 y
    x = func[0].func
    y = func[1].func
    # 函数系数
    fc = func_coef
    # 提取方程中的独立变量 t
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    r = {}

    # 对于方程组 Eq(a1*diff(x(t),t), b1*x(t) + c1*y(t) + d1)
    # 和 Eq(a2*diff(y(t),t), b2*x(t) + c2*y(t) + d2)，设置系数
    r['a1'] = fc[0,x(t),1] ; r['a2'] = fc[1,y(t),1]
    r['b1'] = -fc[0,x(t),0]/fc[0,x(t),1] ; r['b2'] = -fc[1,x(t),0]/fc[1,y(t),1]
    r['c1'] = -fc[0,y(t),0]/fc[0,x(t),1] ; r['c2'] = -fc[1,y(t),0]/fc[1,y(t),1]

    forcing = [S.Zero,S.Zero]
    # 计算非齐次项的常数部分
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j
    
    # 如果方程组没有 t 的非齐次项，可以处理齐次和简单常数非齐次项
    if not (forcing[0].has(t) or forcing[1].has(t)):
        r['d1'] = forcing[0]
        r['d2'] = forcing[1]
    else:
        # 非齐次线性系统不支持，返回 None
        return None

    # 检查 type 6 类型的条件，其方程为 Eq(diff(x(t),t), f(t)*x(t) + g(t)*y(t))
    # 和 Eq(diff(y(t),t), a*[f(t) + a*h(t)]x(t) + a*[g(t) - h(t)]*y(t))
    p = 0
    q = 0
    p1 = cancel(r['b2']/(cancel(r['b2']/r['c2']).as_numer_denom()[0]))
    p2 = cancel(r['b1']/(cancel(r['b1']/r['c1']).as_numer_denom()[0]))
    for n, i in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q and n==0:
                if ((r['b2']/j - r['b1'])/(r['c1'] - r['c2']/j)) == j:
                    p = 1
            elif q and n==1:
                if ((r['b1']/j - r['b2'])/(r['c2'] - r['c1']/j)) == j:
                    p = 2

    # 结束 type 6 类型条件的判断

    # 如果 d1 或 d2 不为零，返回 None
    if r['d1']!=0 or r['d2']!=0:
        return None
    else:
        # 如果 a1, a2, b1, b2, c1, c2 中有任何一个包含 t，返回 None
        if not any(r[k].has(t) for k in 'a1 a2 b1 b2 c1 c2'.split()):
            return None
        else:
            # 归一化 b1, b2, c1, c2
            r['b1'] = r['b1']/r['a1'] ; r['b2'] = r['b2']/r['a2']
            r['c1'] = r['c1']/r['a1'] ; r['c2'] = r['c2']/r['a2']
            if p:
                # 返回 type6 类型
                return "type6"
            else:
                # 返回 type7 类型
                return "type7"

# 检查非线性一阶常微分方程组的特定类型条件，当前实现仅为设置变量
def check_nonlinear_2eq_order1(eq, func, func_coef):
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    f = Wild('f')
    g = Wild('g')
    u, v = symbols('u, v', cls=Dummy)
    def check_type(x, y):
        # 尝试匹配第一个方程的模式
        r1 = eq[0].match(t*diff(x(t),t) - x(t) + f)
        # 尝试匹配第二个方程的模式
        r2 = eq[1].match(t*diff(y(t),t) - y(t) + g)
        # 如果没有同时匹配成功，则尝试备用模式
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t),t) - x(t)/t + f/t)
            r2 = eq[1].match(diff(y(t),t) - y(t)/t + g/t)
        # 如果还没有成功，则尝试负向备用模式
        if not (r1 and r2):
            r1 = (-eq[0]).match(t*diff(x(t),t) - x(t) + f)
            r2 = (-eq[1]).match(t*diff(y(t),t) - y(t) + g)
        # 如果依然没有成功，则尝试负向备用模式的第二种形式
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t),t) - x(t)/t + f/t)
            r2 = (-eq[1]).match(diff(y(t),t) - y(t)/t + g/t)
        # 如果成功匹配并且满足条件，则返回 'type5'，否则返回 None
        if r1 and r2 and not (r1[f].subs(diff(x(t),t),u).subs(diff(y(t),t),v).has(t) \
        or r2[g].subs(diff(x(t),t),u).subs(diff(y(t),t),v).has(t)):
            return 'type5'
        else:
            return None
    
    for func_ in func:
        # 如果 func_ 是列表，则获取其第一个元素的函数作为 x，第二个元素的函数作为 y
        if isinstance(func_, list):
            x = func[0][0].func
            y = func[0][1].func
            # 检查 x 和 y 的类型
            eq_type = check_type(x, y)
            if not eq_type:
                # 如果检查失败，则交换 x 和 y 的顺序再次检查
                eq_type = check_type(y, x)
            return eq_type
    
    # 如果 func 不是列表，则获取其第一个和第二个元素的函数分别作为 x 和 y
    x = func[0].func
    y = func[1].func
    fc = func_coef
    n = Wild('n', exclude=[x(t),y(t)])
    f1 = Wild('f1', exclude=[v,t])
    f2 = Wild('f2', exclude=[v,t])
    g1 = Wild('g1', exclude=[u,t])
    g2 = Wild('g2', exclude=[u,t])
    
    # 对每个方程进行处理
    for i in range(2):
        eqs = 0
        # 将每个方程中的项依次添加到 eqs 中
        for terms in Add.make_args(eq[i]):
            eqs += terms/fc[i,func[i],1]
        eq[i] = eqs
    
    # 尝试匹配第一个方程的模式
    r = eq[0].match(diff(x(t),t) - x(t)**n*f)
    if r:
        # 计算 g
        g = (diff(y(t),t) - eq[1])/r[f]
    if r and not (g.has(x(t)) or g.subs(y(t),v).has(t) or r[f].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type1'
    
    # 尝试匹配第一个方程的模式
    r = eq[0].match(diff(x(t),t) - exp(n*x(t))*f)
    if r:
        # 计算 g
        g = (diff(y(t),t) - eq[1])/r[f]
    if r and not (g.has(x(t)) or g.subs(y(t),v).has(t) or r[f].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type2'
    
    g = Wild('g')
    # 尝试匹配第一个方程的模式和第二个方程的模式
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    if r1 and r2 and not (r1[f].subs(x(t),u).subs(y(t),v).has(t) or \
    r2[g].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type3'
    
    # 尝试匹配第一个方程的模式和第二个方程的模式，然后求解出数值和分母
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    num, den = (
        (r1[f].subs(x(t),u).subs(y(t),v))/
        (r2[g].subs(x(t),u).subs(y(t),v))).as_numer_denom()
    R1 = num.match(f1*g1)
    R2 = den.match(f2*g2)
    # 如果成功匹配，则返回 'type4'，否则返回 None
    if R1 and R2:
        return 'type4'
    
    # 如果所有条件都不满足，则返回 None
    return None
# 检查非线性二阶方程组是否满足特定条件
def check_nonlinear_2eq_order2(eq, func, func_coef):
    return None

# 检查非线性一阶三元方程组是否满足特定条件
def check_nonlinear_3eq_order1(eq, func, func_coef):
    # 提取函数 func 中的三个函数并分别赋值给 x, y, z
    x = func[0].func
    y = func[1].func
    z = func[2].func
    # 将 func_coef 赋值给 fc
    fc = func_coef
    # 提取第一个方程中的自变量并赋值给 t
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    # 创建虚拟变量 u, v, w
    u, v, w = symbols('u, v, w', cls=Dummy)
    # 创建四个带有排除项的 Wild 对象
    a = Wild('a', exclude=[x(t), y(t), z(t), t])
    b = Wild('b', exclude=[x(t), y(t), z(t), t])
    c = Wild('c', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')
    # 创建三个方程的循环
    for i in range(3):
        # 初始化方程变量 eqs
        eqs = 0
        # 将每个 Add.make_args(eq[i]) 中的项加到 eqs 中
        for terms in Add.make_args(eq[i]):
            eqs += terms / fc[i, func[i], 1]
        # 将结果保存回原方程列表中
        eq[i] = eqs
    # 尝试匹配第一个方程的模式 diff(x(t),t) - a*y(t)*z(t)
    r1 = eq[0].match(diff(x(t), t) - a * y(t) * z(t))
    # 尝试匹配第二个方程的模式 diff(y(t),t) - b*z(t)*x(t)
    r2 = eq[1].match(diff(y(t), t) - b * z(t) * x(t))
    # 尝试匹配第三个方程的模式 diff(z(t),t) - c*x(t)*y(t)
    r3 = eq[2].match(diff(z(t), t) - c * x(t) * y(t))
    # 如果所有模式匹配成功
    if r1 and r2 and r3:
        # 将每个匹配项的分子和分母分离
        num1, den1 = r1[a].as_numer_denom()
        num2, den2 = r2[b].as_numer_denom()
        num3, den3 = r3[c].as_numer_denom()
        # 尝试解方程组，如果有解则返回 'type1'
        if solve([num1 * u - den1 * (v - w), num2 * v - den2 * (w - u), num3 * w - den3 * (u - v)], [u, v]):
            return 'type1'
    # 尝试匹配第一个方程的模式 diff(x(t),t) - y(t)*z(t)*f
    r = eq[0].match(diff(x(t), t) - y(t) * z(t) * f)
    if r:
        # 将匹配结果中的常数收集并匹配到 a*f
        r1 = collect_const(r[f]).match(a * f)
        # 尝试匹配第二个方程的模式 (diff(y(t),t) - eq[1])/r1[f]
        r2 = ((diff(y(t), t) - eq[1]) / r1[f]).match(b * z(t) * x(t))
        # 尝试匹配第三个方程的模式 (diff(z(t),t) - eq[2])/r1[f]
        r3 = ((diff(z(t), t) - eq[2]) / r1[f]).match(c * x(t) * y(t))
    # 如果所有模式匹配成功
    if r1 and r2 and r3:
        # 将每个匹配项的分子和分母分离
        num1, den1 = r1[a].as_numer_denom()
        num2, den2 = r2[b].as_numer_denom()
        num3, den3 = r3[c].as_numer_denom()
        # 尝试解方程组，如果有解则返回 'type2'
        if solve([num1 * u - den1 * (v - w), num2 * v - den2 * (w - u), num3 * w - den3 * (u - v)], [u, v]):
            return 'type2'
    # 尝试匹配第一个方程的模式 diff(x(t),t) - (F2-F3)
    r = eq[0].match(diff(x(t), t) - (F2 - F3))
    if r:
        # 将匹配结果中的常数收集并匹配到 c*F2
        r1 = collect_const(r[F2]).match(c * F2)
        # 将匹配结果中的常数收集并匹配到 b*F3
        r1.update(collect_const(r[F3]).match(b * F3))
        # 如果收集成功
        if r1:
            # 如果第二个方程中包含 r1[F2] 但不包含 r1[F3]
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                # 交换 r1[F2] 和 r1[F3]，并将 r1[c] 和 r1[b] 取反
                r1[F2], r1[F3] = r1[F3], r1[F2]
                r1[c], r1[b] = -r1[b], -r1[c]
            # 尝试匹配第二个方程的模式 diff(y(t),t) - a*r1[F3] + r1[c]*F1
            r2 = eq[1].match(diff(y(t), t) - a * r1[F3] + r1[c] * F1)
        # 如果匹配成功
        if r2:
            # 尝试匹配第三个方程的模式 eq[2] == diff(z(t),t) - r1[b]*y(t)*r2[F1] + r2[a]*r1[F2]
            r3 = (eq[2] == diff(z(t), t) - r1[b] * y(t) * r2[F1] + r2[a] * r1[F2])
        # 如果所有模式匹配成功
        if r1 and r2 and r3:
            return 'type3'
    # 尝试匹配第一个方程的模式 diff(x(t),t) - z(t)*F2 + y(t)*F3
    r = eq[0].match(diff(x(t), t) - z(t) * F2 + y(t) * F3)
    if r:
        # 将匹配结果中的常数收集并匹配到 c*F2
        r1 = collect_const(r[F2]).match(c * F2)
        # 将匹配结果中的常数收集并匹配到 b*F3
        r1.update(collect_const(r[F3]).match(b * F3))
        # 如果收集成功
        if r1:
            # 如果第二个方程中包含 r1[F2] 但不包含 r1[F3]
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                # 交换 r1[F2] 和 r1[F3]，并将 r1[c] 和 r1[b] 取反
                r1[F2], r1[F3] = r1[F3], r1[F2]
                r1[c], r1[b] = -r1[b], -r1[c]
            # 尝试匹配第二个方程的模式 (diff(y(t),t) - eq[1]).match(a*x(t)*r1[F3] - r1[c]*z(t)*F1)
            r2 = (diff(y(t), t) - eq[1]).match(a * x(t) * r1[F3] - r1[c] * z(t) * F1)
        # 如果匹配成功
        if r2:
            # 尝试匹配第三个方程的模式 (diff(z(t),t) - eq[2] == r1[b]*y(t)*r2[F1] - r2[a]*x(t)*r1[F2])
            r
    # 如果条件 r 成立，则执行以下语句块
    if r:
        # 使用 collect_const 函数从 r[F2] 中收集常数并匹配 c*F2 的模式，结果存储在 r1 中
        r1 = collect_const(r[F2]).match(c*F2)
        # 继续从 r[F3] 中收集常数并匹配 b*F3 的模式，更新 r1
        r1.update(collect_const(r[F3]).match(b*F3))
        # 如果 r1 不为空
        if r1:
            # 如果 eq[1] 中包含 r1[F2] 且不包含 r1[F3]
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                # 交换 r1[F2] 和 r1[F3]
                r1[F2], r1[F3] = r1[F3], r1[F2]
                # 取反 r1[c] 和 r1[b]
                r1[c], r1[b] = -r1[b], -r1[c]
            # 计算 y(t) 的导数与 eq[1] 的差，并尝试匹配 y(t)*(a*r1[F3] - r1[c]*F1) 的模式，结果存储在 r2 中
            r2 = (diff(y(t),t) - eq[1]).match(y(t)*(a*r1[F3] - r1[c]*F1))
        # 如果 r2 存在
        if r2:
            # 计算 z(t) 的导数与 eq[2] 的差，并尝试匹配 z(t)*(r1[b]*r2[F1] - r2[a]*r1[F2]) 的模式，结果存储在 r3 中
            r3 = (diff(z(t),t) - eq[2] == z(t)*(r1[b]*r2[F1] - r2[a]*r1[F2]))
        # 如果 r1、r2 和 r3 都存在
        if r1 and r2 and r3:
            # 返回字符串 'type5'
            return 'type5'
    # 如果以上条件都不满足，则返回 None
    return None
# 定义一个函数，用于检查三阶非线性方程的二阶导数项是否为零
def check_nonlinear_3eq_order2(eq, func, func_coef):
    return None


# 使用向量化装饰器对函数进行修饰，使其支持对单个参数的向量化操作
@vectorize(0)
def odesimp(ode, eq, func, hint):
    r"""
    简化常微分方程的解，尝试解出 `func`，并运行 :py:meth:`~sympy.solvers.ode.constantsimp`。

    可能根据 `hint` 返回的解类型进行附加简化。

    如果 `hint` 不是 `_Integral` 提示，则尝试集成表达式中的任何 :py:class:`~sympy.integrals.integrals.Integral`。

    此函数不应影响 :py:meth:`~sympy.solvers.ode.dsolve` 返回的表达式，因为
    :py:meth:`~sympy.solvers.ode.dsolve` 已经调用
    :py:meth:`~sympy.solvers.ode.ode.odesimp`，但个别提示函数不会调用
    :py:meth:`~sympy.solvers.ode.ode.odesimp`（因为 :py:meth:`~sympy.solvers.ode.dsolve` 包装器已经调用它）。
    因此，此函数主要设计用于内部使用。

    Examples
    ========

    >>> from sympy import sin, symbols, dsolve, pprint, Function
    >>> from sympy.solvers.ode.ode import odesimp
    >>> x, u2, C1= symbols('x,u2,C1')
    >>> f = Function('f')

    >>> eq = dsolve(x*f(x).diff(x) - f(x) - x*sin(f(x)/x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral',
    ... simplify=False)
    >>> pprint(eq, wrap_line=False)
                            x
                           ----
                           f(x)
                             /
                            |
                            |   /        1   \
                            |  -|u1 + -------|
                            |   |        /1 \|
                            |   |     sin|--||
                            |   \        \u1//
    log(f(x)) = log(C1) +   |  ---------------- d(u1)
                            |          2
                            |        u1
                            |
                           /

    >>> pprint(odesimp(eq, f(x), 1, {C1},
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep'
    ... )) #doctest: +SKIP
        x
    --------- = C1
       /f(x)\
    tan|----|
       \2*x /

    """
    # 获取函数的自变量
    x = func.args[0]
    # 获取函数的类型（如 Function('f')）
    f = func.func
    # 获取常数 C1
    C1 = get_numbered_constants(eq, num=1)
    # 获取方程中的所有常数
    constants = eq.free_symbols - ode.free_symbols

    # 首先，如果提示允许，进行积分处理
    eq = _handle_Integral(eq, func, hint)
    # 如果提示以 "nth_linear_euler_eq_nonhomogeneous" 开头，则简化方程
    if hint.startswith("nth_linear_euler_eq_nonhomogeneous"):
        eq = simplify(eq)
    # 确保 eq 是一个等式类型
    if not isinstance(eq, Equality):
        raise TypeError("eq should be an instance of Equality")

    # 在假设符号非零的情况下，允许简化
    eq = eq.xreplace((_:={i: Dummy(nonzero=True) for i in constants})).xreplace({_[i]: i for i in _})

    # 接下来，清理任意常数
    # 目前，nth linear hints 可以在表达式中放入多达 2*order 个常数。
    # 如果这个数量随着另一个提示而增长，则第三个参数
    # 使用 constantsimp 函数对方程进行简化，以处理常数
    eq = constantsimp(eq, constants)

    # 清理表达式后，尝试解 func
    # 当 solve() 实现了 CRootOf 时，我们希望每次返回 CRootOf 而不是 Equality
    # 尝试将 f(x) 移到等式左侧（如果可能）
    if eq.rhs == func and not eq.lhs.has(func):
        eq = [Eq(eq.rhs, eq.lhs)]

    # 确保我们处理的是简化形式的解列表
    if eq.lhs == func and not eq.rhs.has(func):
        # 解已经是简化形式
        eq = [eq]
    else:
        # 解未简化，尝试解决它
        try:
            # 检查是否有浮点数
            floats = any(i.is_Float for i in eq.atoms(Number))
            # 使用 solve() 尝试解方程 eq 关于 func 的值
            eqsol = solve(eq, func, force=True, rational=False if floats else None)
            if not eqsol:
                # 如果解为空，抛出未实现错误
                raise NotImplementedError
        except (NotImplementedError, PolynomialError):
            # 处理错误情况，将 eq 装入列表中
            eq = [eq]
        else:
            # 定义内部函数 _expand 用于扩展表达式
            def _expand(expr):
                numer, denom = expr.as_numer_denom()

                # 如果分母是加法，则返回原表达式
                if denom.is_Add:
                    return expr
                else:
                    # 否则对分子进行指数简化并扩展
                    return powsimp(expr.expand(), combine='exp', deep=True)

            # 对每个解进行处理，构建形如 f(x) = _expand(t) 的等式列表
            eq = [Eq(f(x), _expand(t)) for t in eqsol]

        # 对左侧进行特殊简化处理
        if hint.startswith("1st_homogeneous_coeff"):
            for j, eqi in enumerate(eq):
                # 使用 logcombine 函数强制合并对数项
                newi = logcombine(eqi, force=True)
                # 如果新等式的左侧是对数，并且右侧为0，则改写等式
                if isinstance(newi.lhs, log) and newi.rhs == 0:
                    newi = Eq(newi.lhs.args[0]/C1, C1)
                eq[j] = newi

    # 在返回之前，再次运行 constantsimp()，确保表达式简化
    for i, eqi in enumerate(eq):
        eq[i] = constantsimp(eqi, constants)
        # 使用 constant_renumber 函数重新编号常数
        eq[i] = constant_renumber(eq[i], ode.free_symbols)

    # 如果只有一个解，返回该解；否则返回解列表
    if len(eq) == 1:
        eq = eq[0]
    return eq
# 定义一个函数，评估给定常微分方程（ODE）的解的简单程度，并返回一个扩展的整数表示其简单程度
def ode_sol_simplicity(sol, func, trysolving=True):
    r"""
    返回一个扩展整数，表示常微分方程（ODE）的解的简单程度。

    从最简单到最复杂的顺序考虑以下几点：

    - ``sol`` 是关于 ``func`` 的解。
    - ``sol`` 不是关于 ``func`` 的解，但如果传递给 solve 函数，可以解决（例如，
      由 ``dsolve(ode, func, simplify=False)`` 返回的解）。
    - 如果 ``sol`` 不是关于 ``func`` 的解，则基于 ``sol`` 的长度来确定简单程度，
      长度由 ``len(str(sol))`` 计算。
    - 如果 ``sol`` 包含任何未评估的 :py:class:`~sympy.integrals.integrals.Integral` 对象，
      则自动认为比上述任何情况都不简单。

    该函数返回一个整数，使得如果解 A 按照上述度量比解 B 更简单，则
    ``ode_sol_simplicity(sola, func) < ode_sol_simplicity(solb, func)``。

    目前返回以下数字，但如果启发式算法改进，可能会更改。仅保证顺序。

    +----------------------------------------------+-------------------+
    | 简单程度                                     | 返回值             |
    +==============================================+===================+
    | ``sol`` 是关于 ``func`` 的解                | ``-2``            |
    +----------------------------------------------+-------------------+
    | ``sol`` 不是关于 ``func`` 的解，但可以解决   | ``-1``            |
    +----------------------------------------------+-------------------+
    | ``sol`` 既不是关于 ``func`` 的解，也不可解  | ``len(str(sol))`` |
    | 的，长度为                                   |                   |
    +----------------------------------------------+-------------------+
    | ``sol`` 包含一个                             | ``oo``            |
    | :obj:`~sympy.integrals.integrals.Integral`   |                   |
    +----------------------------------------------+-------------------+

    这里的 ``oo`` 表示 SymPy 中的无穷大，应该比任何整数都大。

    如果您已经知道 :py:meth:`~sympy.solvers.solvers.solve` 不能解决 ``sol``，
    可以使用 ``trysolving=False`` 跳过该步骤，这是唯一可能较慢的步骤。例如，
    :py:meth:`~sympy.solvers.ode.dsolve` 使用 ``simplify=False`` 标志应该这样做。

    如果 ``sol`` 是解列表，则如果列表中最差的解返回 ``oo``，则返回 ``oo``，
    否则返回 ``len(str(sol))``，即整个列表的字符串表示的长度。

    示例
    ========

    此函数设计为传递给 ``min`` 函数的关键字参数，例如 ``min(listofsolutions, key=lambda i: ode_sol_simplicity(i, f(x)))``。

    >>> from sympy import symbols, Function, Eq, tan, Integral
    >>> from sympy.solvers.ode.ode import ode_sol_simplicity
    >>> x, C1, C2 = symbols('x, C1, C2')
    >>> f = Function('f')
    # TODO: 如果对于 f(x) 解出两个解，我们仍然希望得到较简单的那一个
    
    # 查看类型为 sol 的对象是否可迭代，如果是，则遍历其中的每个元素 i
    for i in sol:
        # 如果解 ode_sol_simplicity(i, func, trysolving=trysolving) 的结果为无穷大 oo，则返回 oo
        if ode_sol_simplicity(i, func, trysolving=trysolving) == oo:
            return oo
    
        # 返回 sol 的字符串表示的长度
        return len(str(sol))
    
    # 如果 sol 包含 Integral，则返回无穷大 oo
    if sol.has(Integral):
        return oo
    
    # 尝试手动解析 func 的解
    if trysolving:
        try:
            # 尝试求解 sol 对于 func 的解
            sols = solve(sol, func)
            # 如果没有找到解，则抛出 NotImplementedError
            if not sols:
                raise NotImplementedError
        except NotImplementedError:
            pass
        else:
            return -1
    
    # 最后，基于表达式的字符串长度进行简单的计算。这可能会偏向合并的分数形式，因为它们不会有重复的分母，
    # 并且可能会稍微偏向具有较少加法和减法的表达式，因为这些操作符被打印时会用空格分隔。
    # 欢迎额外的简化启发式方法的想法，比如检查方程是否有较大的定义域，或者 constantsimp 是否引入了
    # 高于给定 ODE 阶数的任意编号的常数。
    return len(str(sol))
# 提取表达式列表中的所有导数节点，存入 derivs 列表
derivs = [node for node in preorder_traversal(eq) if isinstance(node, Derivative)]
# 初始化空列表 func 用于存放导数节点中的 AppliedUndef 对象
func = []
# 遍历 derivs 列表中的每个 Derivative 对象 d
for d in derivs:
    # 将 d 中的 AppliedUndef 对象转为列表，并加入 func 列表
    func += list(d.atoms(AppliedUndef))
# 将 func 列表中的元素添加到 funcs 列表中
for func_ in func:
    funcs.append(func_)

# 对 funcs 列表去重，确保每个元素只出现一次
funcs = list(uniq(funcs))

# 返回去重后的函数列表 funcs
return funcs


# 以递归方式获取表达式 expr 中的常数子表达式，Cs 是已知常数集合
Cs = set(Cs)
# 初始化空列表 Ces，用于存放找到的常数子表达式
Ces = []

def _recursive_walk(expr):
    # 获取表达式中的自由符号集合
    expr_syms = expr.free_symbols
    # 如果表达式中的所有自由符号都在 Cs 集合中，则将 expr 加入 Ces 列表
    if expr_syms and expr_syms.issubset(Cs):
        Ces.append(expr)
    else:
        # 处理指数函数 exp 的情况，展开表达式
        if expr.func == exp:
            expr = expr.expand(mul=True)
        # 处理加法和乘法函数的情况
        if expr.func in (Add, Mul):
            # 使用 sift 函数将表达式按照自由符号是否在 Cs 集合中分组
            d = sift(expr.args, lambda i : i.free_symbols.issubset(Cs))
            # 如果某一组中的元素超过一个，则创建新的表达式 x，并加入 Ces 列表
            if len(d[True]) > 1:
                x = expr.func(*d[True])
                if not x.is_number:
                    Ces.append(x)
        # 处理积分表达式的情况
        elif isinstance(expr, Integral):
            # 如果积分表达式中的自由符号都在 Cs 集合中，并且所有限制条件长度为 3，则加入 Ces 列表
            if expr.free_symbols.issubset(Cs) and \
                            all(len(x) == 3 for x in expr.limits):
                Ces.append(expr)
        # 递归处理表达式的每个参数
        for i in expr.args:
            _recursive_walk(i)
    return

# 调用递归函数 _recursive_walk 处理输入的表达式 expr
_recursive_walk(expr)
# 返回找到的所有常数子表达式列表 Ces
return Ces


# 去除表达式 expr 中线性冗余项，Cs 是已知的常数集合
# 计算表达式中每个常数的出现次数，并将出现次数大于 0 的常数加入 Cs 列表
cnts = {i: expr.count(i) for i in Cs}
Cs = [i for i in Cs if cnts[i] > 0]

def _linear(expr):
    # 如果表达式是加法表达式
    if isinstance(expr, Add):
        # 在 Cs 中查找每个常数的出现次数，并且其二阶导数为 0
        xs = [i for i in Cs if expr.count(i)==cnts[i] \
            and 0 == expr.diff(i, 2)]
        # 初始化空字典 d 用于存放线性冗余项
        d = {}
        for x in xs:
            # 计算表达式对 x 的一阶导数，并将结果加入 d 字典
            y = expr.diff(x)
            if y not in d:
                d[y]=[]
            d[y].append(x)
        for y in d:
            # 如果某一项 y 在 d 中出现超过一次，则将除第一个外的其他项 x 替换为 0
            if len(d[y]) > 1:
                d[y].sort(key=str)
                for x in d[y][1:]:
                    expr = expr.subs(x, 0)
    return expr

def _recursive_walk(expr):
    # 如果表达式的参数数量不为 0
    if len(expr.args) != 0:
        # 对表达式的每个参数递归调用 _recursive_walk 函数，并用结果重新构造表达式
        expr = expr.func(*[_recursive_walk(i) for i in expr.args])
    # 对表达式调用 _linear 函数，去除其中的线性冗余项
    expr = _linear(expr)
    return expr
    # 检查表达式是否为 Equality 类型
    if isinstance(expr, Equality):
        # 递归地处理表达式的每个参数，并将结果解构为 lhs 和 rhs
        lhs, rhs = [_recursive_walk(i) for i in expr.args]
        
        # 定义用于筛选数字或者已知符号的 lambda 函数
        f = lambda i: isinstance(i, Number) or i in Cs
        
        # 如果 lhs 是符号并且在 Cs 集合中
        if isinstance(lhs, Symbol) and lhs in Cs:
            # 调整 lhs 和 rhs 的位置
            rhs, lhs = lhs, rhs
        
        # 处理 lhs 和 rhs 是 Add 或者 Symbol 类型的情况
        if lhs.func in (Add, Symbol) and rhs.func in (Add, Symbol):
            # 分别筛选 lhs 和 rhs 中符合条件的部分
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            drhs = sift([rhs] if isinstance(rhs, AtomicExpr) else rhs.args, f)
            
            # 对于 True 和 False 两种情况，如果不存在于 dlhs 或 drhs 中，则添加默认值
            for i in [True, False]:
                for hs in [dlhs, drhs]:
                    if i not in hs:
                        hs[i] = [0]
            
            # 简化计算结果
            lhs = Add(*dlhs[False]) - Add(*drhs[False])
            rhs = Add(*drhs[True]) - Add(*dlhs[True])
        
        # 处理 lhs 和 rhs 是 Mul 或者 Symbol 类型的情况
        elif lhs.func in (Mul, Symbol) and rhs.func in (Mul, Symbol):
            # 筛选 lhs 中符合条件的部分
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            
            # 如果 dlhs 中存在 True，则处理 lhs 和 rhs 的乘法运算
            if True in dlhs:
                if False not in dlhs:
                    dlhs[False] = [1]
                lhs = Mul(*dlhs[False])
                rhs = rhs / Mul(*dlhs[True])
        
        # 返回等式 Eq(lhs, rhs)
        return Eq(lhs, rhs)
    
    else:
        # 对于非 Equality 类型的表达式，递归地处理
        return _recursive_walk(expr)
# 使用装饰器 @vectorize(0) 将函数 constantsimp 转换为向量化函数，使其能够处理输入的数组或向量
@vectorize(0)
# 定义一个函数 constantsimp，用于简化带有任意常数的表达式
def constantsimp(expr, constants):
    r"""
    Simplifies an expression with arbitrary constants in it.

    This function is written specifically to work with
    :py:meth:`~sympy.solvers.ode.dsolve`, and is not intended for general use.

    Simplification is done by "absorbing" the arbitrary constants into other
    arbitrary constants, numbers, and symbols that they are not independent
    of.

    The symbols must all have the same name with numbers after it, for
    example, ``C1``, ``C2``, ``C3``.  The ``symbolname`` here would be
    '``C``', the ``startnumber`` would be 1, and the ``endnumber`` would be 3.
    If the arbitrary constants are independent of the variable ``x``, then the
    independent symbol would be ``x``.  There is no need to specify the
    dependent function, such as ``f(x)``, because it already has the
    independent symbol, ``x``, in it.

    Because terms are "absorbed" into arbitrary constants and because
    constants are renumbered after simplifying, the arbitrary constants in
    expr are not necessarily equal to the ones of the same name in the
    returned result.

    If two or more arbitrary constants are added, multiplied, or raised to the
    power of each other, they are first absorbed together into a single
    arbitrary constant.  Then the new constant is combined into other terms if
    necessary.

    Absorption of constants is done with limited assistance:

    1. terms of :py:class:`~sympy.core.add.Add`\s are collected to try join
       constants so `e^x (C_1 \cos(x) + C_2 \cos(x))` will simplify to `e^x
       C_1 \cos(x)`;

    2. powers with exponents that are :py:class:`~sympy.core.add.Add`\s are
       expanded so `e^{C_1 + x}` will be simplified to `C_1 e^x`.

    Use :py:meth:`~sympy.solvers.ode.ode.constant_renumber` to renumber constants
    after simplification or else arbitrary numbers on constants may appear,
    e.g. `C_1 + C_3 x`.

    In rare cases, a single constant can be "simplified" into two constants.
    Every differential equation solution should have as many arbitrary
    constants as the order of the differential equation.  The result here will
    be technically correct, but it may, for example, have `C_1` and `C_2` in
    an expression, when `C_1` is actually equal to `C_2`.  Use your discretion
    in such situations, and also take advantage of the ability to use hints in
    :py:meth:`~sympy.solvers.ode.dsolve`.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.ode.ode import constantsimp
    >>> C1, C2, C3, x, y = symbols('C1, C2, C3, x, y')
    >>> constantsimp(2*C1*x, {C1, C2, C3})
    C1*x
    >>> constantsimp(C1 + 2 + x, {C1, C2, C3})
    C1 + x
    >>> constantsimp(C1*C2 + 2 + C2 + C3*x, {C1, C2, C3})
    C1 + C3*x

    """
    # This function works recursively.  The idea is that, for Mul,
    # Add, Pow, and Function, if the class has a constant in it, then
    # the constant is collected to be joined with other constants.
    # 我们可以简化它，通过递归向下和向上简化。否则，我们可以跳过表达式的那部分。

    Cs = constants  # 将常量列表赋值给变量Cs

    orig_expr = expr  # 将原始表达式保存在变量orig_expr中

    # 获取表达式中的常量子表达式列表
    constant_subexprs = _get_constant_subexpressions(expr, Cs)
    for xe in constant_subexprs:
        xes = list(xe.free_symbols)
        if not xes:  # 如果子表达式中没有自由符号，则跳过
            continue
        # 如果所有子表达式中的符号在整个表达式中出现次数相同，则简化表达式
        if all(expr.count(c) == xe.count(c) for c in xes):
            xes.sort(key=str)
            expr = expr.subs(xe, xes[0])

    # 尝试进行常见子表达式消除以简化常量项
    try:
        # 使用常见子表达式消除函数cse对表达式进行处理
        commons, rexpr = cse(expr)
        commons.reverse()  # 反转常见子表达式列表
        rexpr = rexpr[0]  # 获取简化后的表达式
        for s in commons:
            cs = list(s[1].atoms(Symbol))  # 获取子表达式中的符号集合
            # 如果子表达式中只有一个符号，且该符号在常量列表中，并且不在rexpr的符号集合中，
            # 并且没有其他常见子表达式中使用该符号，则将其替换为该符号
            if len(cs) == 1 and cs[0] in Cs and \
                cs[0] not in rexpr.atoms(Symbol) and \
                not any(cs[0] in ex for ex in commons if ex != s):
                rexpr = rexpr.subs(s[0], cs[0])
            else:
                rexpr = rexpr.subs(*s)  # 否则，替换所有出现的子表达式
        expr = rexpr  # 更新表达式为最终简化结果
    except IndexError:
        pass  # 捕获索引错误异常并忽略

    expr = __remove_linear_redundancies(expr, Cs)  # 移除表达式中的线性冗余项

    # 定义条件项的因式分解函数
    def _conditional_term_factoring(expr):
        # 对表达式进行因式分解处理，保持指数不变
        new_expr = terms_gcd(expr, clear=False, deep=True, expand=False)

        # 我们不想因式分解指数项，因此单独处理它们
        if new_expr.is_Mul:
            infac = False
            asfac = False
            for m in new_expr.args:
                if isinstance(m, exp):
                    asfac = True
                elif m.is_Add:
                    infac = any(isinstance(fi, exp) for t in m.args
                        for fi in Mul.make_args(t))
                if asfac and infac:
                    new_expr = expr
                    break
        return new_expr

    expr = _conditional_term_factoring(expr)  # 应用条件项的因式分解函数到表达式中

    # 如果原始表达式和当前表达式不同，则递归调用constantsimp函数进一步简化
    if orig_expr != expr:
        return constantsimp(expr, Cs)  # 递归调用constantsimp函数并返回结果
    return expr  # 返回最终的表达式结果
# 定义函数 constant_renumber，用于重新编号表达式中的任意常数，并按照给定的 newconstants 符号名称进行重新排序
def constant_renumber(expr, variables=None, newconstants=None):
    r"""
    Renumber arbitrary constants in ``expr`` to use the symbol names as given
    in ``newconstants``. In the process, this reorders expression terms in a
    standard way.

    If ``newconstants`` is not provided then the new constant names will be
    ``C1``, ``C2`` etc. Otherwise ``newconstants`` should be an iterable
    giving the new symbols to use for the constants in order.

    The ``variables`` argument is a list of non-constant symbols. All other
    free symbols found in ``expr`` are assumed to be constants and will be
    renumbered. If ``variables`` is not given then any numbered symbol
    beginning with ``C`` (e.g. ``C1``) is assumed to be a constant.

    Symbols are renumbered based on ``.sort_key()``, so they should be
    numbered roughly in the order that they appear in the final, printed
    expression.  Note that this ordering is based in part on hashes, so it can
    produce different results on different machines.

    The structure of this function is very similar to that of
    :py:meth:`~sympy.solvers.ode.constantsimp`.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.ode.ode import constant_renumber
    >>> x, C1, C2, C3 = symbols('x,C1:4')
    >>> expr = C3 + C2*x + C1*x**2
    >>> expr
    C1*x**2  + C2*x + C3
    >>> constant_renumber(expr)
    C1 + C2*x + C3*x**2

    The ``variables`` argument specifies which are constants so that the
    other symbols will not be renumbered:

    >>> constant_renumber(expr, [C1, x])
    C1*x**2  + C2 + C3*x

    The ``newconstants`` argument is used to specify what symbols to use when
    replacing the constants:

    >>> constant_renumber(expr, [x], newconstants=symbols('E1:4'))
    E1 + E2*x + E3*x**2

    """

    # 如果表达式是集合、列表或元组，则递归调用 constant_renumber，并返回相同类型的对象
    if isinstance(expr, (set, list, tuple)):
        return type(expr)(constant_renumber(Tuple(*expr),
                        variables=variables, newconstants=newconstants))

    # 如果提供了 variables 参数，则将其转换为集合，找出表达式中除了 variables 以外的所有自由符号，认定为常数符号
    if variables is not None:
        variables = set(variables)
        free_symbols = expr.free_symbols
        constantsymbols = list(free_symbols - variables)
    # 否则，假设任何以 'C' 开头且后跟数字的符号为常数
    else:
        variables = set()
        isconstant = lambda s: s.startswith('C') and s[1:].isdigit()
        constantsymbols = [sym for sym in expr.free_symbols if isconstant(sym.name)]

    # 如果未提供 newconstants，则使用 numbered_symbols 函数生成以 'C' 开头的新常数符号
    # 否则，使用 newconstants 中的符号进行重新编号，排除已在 variables 中的符号
    if newconstants is None:
        iter_constants = numbered_symbols(start=1, prefix='C', exclude=variables)
    else:
        iter_constants = (sym for sym in newconstants if sym not in variables)

    constants_found = []

    # 创建一个映射，将所有常数符号映射到 S.One，并用于确保术语排序不依赖于 C 的索引值
    C_1 = [(ci, S.One) for ci in constantsymbols]
    sort_key=lambda arg: default_sort_key(arg.subs(C_1))

    def _constant_renumber(expr):
        r"""
        内部递归函数，用于对表达式进行常数重新编号
        """

        # 处理表达式系统
        if isinstance(expr, Tuple):
            # 递归处理元组中的每个表达式
            renumbered = [_constant_renumber(e) for e in expr]
            return Tuple(*renumbered)

        # 处理等式表达式
        if isinstance(expr, Equality):
            # 递归处理等式左右两侧的表达式
            return Eq(
                _constant_renumber(expr.lhs),
                _constant_renumber(expr.rhs))

        # 处理非常数、非函数、非特定常数符号的表达式
        if type(expr) not in (Mul, Add, Pow) and not expr.is_Function and \
                not expr.has(*constantsymbols):
            # 基本情况，希望在其他类中没有常数，因为它们不会被重新编号
            return expr
        elif expr.is_Piecewise:
            # 处理分段函数表达式
            return expr
        elif expr in constantsymbols:
            # 处理常数符号
            if expr not in constants_found:
                constants_found.append(expr)
            return expr
        elif expr.is_Function or expr.is_Pow:
            # 处理函数或幂函数表达式
            return expr.func(
                *[_constant_renumber(x) for x in expr.args])
        else:
            # 处理一般情况，对表达式的参数进行排序
            sortedargs = list(expr.args)
            sortedargs.sort(key=sort_key)
            return expr.func(*[_constant_renumber(x) for x in sortedargs])

    # 对表达式进行常数重新编号操作
    expr = _constant_renumber(expr)

    # 在 ODE 中不重新编号已经存在的符号
    constants_found = [c for c in constants_found if c not in variables]

    # 执行常数重新编号的替换过程
    subs_dict = dict(zip(constants_found, iter_constants))
    expr = expr.subs(subs_dict, simultaneous=True)

    # 返回经过常数重新编号和替换的表达式
    return expr
def _handle_Integral(expr, func, hint):
    r"""
    Converts a solution with Integrals in it into an actual solution.

    For most hints, this simply runs ``expr.doit()``.

    """
    # 根据提示参数决定是否处理带有积分的表达式
    if hint == "nth_linear_constant_coeff_homogeneous":
        sol = expr  # 如果提示是特定类型的线性常系数齐次方程，则解为原始表达式
    elif not hint.endswith("_Integral"):
        sol = expr.doit()  # 如果提示不是以"_Integral"结尾，则调用doit()方法进行求解
    else:
        sol = expr  # 否则保持表达式不变，即表明它本身已经是处理过的积分表达式
    return sol


# XXX: Should this function maybe go somewhere else?


def homogeneous_order(eq, *symbols):
    r"""
    Returns the order `n` if `g` is homogeneous and ``None`` if it is not
    homogeneous.

    Determines if a function is homogeneous and if so of what order.  A
    function `f(x, y, \cdots)` is homogeneous of order `n` if `f(t x, t y,
    \cdots) = t^n f(x, y, \cdots)`.

    If the function is of two variables, `F(x, y)`, then `f` being homogeneous
    of any order is equivalent to being able to rewrite `F(x, y)` as `G(x/y)`
    or `H(y/x)`.  This fact is used to solve 1st order ordinary differential
    equations whose coefficients are homogeneous of the same order (see the
    docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`).

    Symbols can be functions, but every argument of the function must be a
    symbol, and the arguments of the function that appear in the expression
    must match those given in the list of symbols.  If a declared function
    appears with different arguments than given in the list of symbols,
    ``None`` is returned.

    Examples
    ========

    >>> from sympy import Function, homogeneous_order, sqrt
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> homogeneous_order(f(x), f(x)) is None
    True
    >>> homogeneous_order(f(x,y), f(y, x), x, y) is None
    True
    >>> homogeneous_order(f(x), f(x), x)
    1
    >>> homogeneous_order(x**2*f(x)/sqrt(x**2+f(x)**2), x, f(x))
    2
    >>> homogeneous_order(x**2+f(x), x, f(x)) is None
    True

    """
    # 检查是否提供了符号列表，若未提供则抛出异常
    if not symbols:
        raise ValueError("homogeneous_order: no symbols were given.")
    symset = set(symbols)
    eq = sympify(eq)

    # 若表达式包含 Order 或 Derivative，则不支持判断是否齐次
    if eq.has(Order, Derivative):
        return None

    # 若表达式为常数，则直接返回零
    if (eq.is_Number or
        eq.is_NumberSymbol or
        eq.is_number
            ):
        return S.Zero

    # 将所有函数用虚拟变量替换
    dum = numbered_symbols(prefix='d', cls=Dummy)
    newsyms = set()
    for i in [j for j in symset if getattr(j, 'is_Function')]:
        iargs = set(i.args)
        if iargs.difference(symset):
            return None
        else:
            dummyvar = next(dum)
            eq = eq.subs(i, dummyvar)
            symset.remove(i)
            newsyms.add(dummyvar)
    symset.update(newsyms)

    # 若表达式中不含给定的符号，则不是齐次的
    if not eq.free_symbols & symset:
        return None

    # 假设嵌套函数的阶数只能是零
    # 如果eq是Function类的实例，则执行以下操作
    if isinstance(eq, Function):
        # 如果eq.args[0]与symset中的所有符号具有同质次数，则返回S.Zero；否则返回None
        return None if homogeneous_order(
            eq.args[0], *tuple(symset)) != 0 else S.Zero

    # 使用t替换eq中的x，并检查是否可以因式分解出t
    t = Dummy('t', positive=True)  # 创建一个正数假变量t
    # 将eq中的每个符号i替换为t*i，并将结果按照t分离出来
    eqs = separatevars(eq.subs([(i, t*i) for i in symset]), [t], dict=True)[t]
    # 如果eqs等于S.One，则表示没有只包含t的项，返回S.Zero
    if eqs is S.One:
        return S.Zero
    # 将eqs分解为一个独立项i和一个剩余项d
    i, d = eqs.as_independent(t, as_Add=False)
    # 将剩余项d分解为底数b和指数e
    b, e = d.as_base_exp()
    # 如果底数b等于t，则返回指数e
    if b == t:
        return e
    x = func.args[0]
    # 获取函数的自变量（通常为 x）
    f = func.func
    # 获取函数的名称（通常为 f）
    C0, C1 = get_numbered_constants(eq, num=2)
    # 获取方程中的前两个常数 C0 和 C1
    n = Dummy("n", integer=True)
    # 创建一个符号 n，表示整数
    s = Wild("s")
    # 创建一个通配符 s，用于匹配模式
    k = Wild("k", exclude=[x])
    # 创建一个通配符 k，排除自变量 x
    x0 = match['x0']
    # 获取匹配对象中的 x0 值
    terms = match['terms']
    # 获取匹配对象中的 terms 值
    p = match[match['a3']]
    # 获取匹配对象中的 a3 对应的值
    q = match[match['b3']]
    # 获取匹配对象中的 b3 对应的值
    r = match[match['c3']]
    # 获取匹配对象中的 c3 对应的值
    seriesdict = {}
    # 创建一个空字典 seriesdict
    recurr = Function("r")
    # 创建一个函数对象 recurr，代表递归函数 r

    # Generating the recurrence relation which works this way:
    # for the second order term the summation begins at n = 2. The coefficients
    # p is multiplied with an*(n - 1)*(n - 2)*x**n-2 and a substitution is made such that
    # the exponent of x becomes n.
    # For example, if p is x, then the second degree recurrence term is
    # an*(n - 1)*(n - 2)*x**n-1, substituting (n - 1) as n, it transforms to
    # an+1*n*(n - 1)*x**n.
    # A similar process is done with the first order and zeroth order term.

    coefflist = [(recurr(n), r), (n*recurr(n), q), (n*(n - 1)*recurr(n), p)]
    # 创建一个系数列表，用于生成递推关系
    for index, coeff in enumerate(coefflist):
        # 遍历系数列表，获取索引和系数
        if coeff[1]:
            # 如果系数不为零
            f2 = powsimp(expand((coeff[1]*(x - x0)**(n - index)).subs(x, x + x0)))
            # 计算 f2，对其中的表达式进行简化和展开
            if f2.is_Add:
                addargs = f2.args
            else:
                addargs = [f2]
            # 检查 f2 是否是加法表达式，提取加法项
            for arg in addargs:
                # 遍历加法项
                powm = arg.match(s*x**k)
                # 匹配 s*x**k 形式的模式
                term = coeff[0]*powm[s]
                # 计算当前项的系数乘以匹配结果中的 s
                if not powm[k].is_Symbol:
                    term = term.subs(n, n - powm[k].as_independent(n)[0])
                # 如果 k 不是符号，则将其作为 n 的独立部分代入 term
                startind = powm[k].subs(n, index)
                # 获取开始指数 startind，将 n 替换为 index
                # Seeing if the startterm can be reduced further.
                # If it vanishes for n lesser than startind, it is
                # equal to summation from n.
                # 检查 startterm 是否可以进一步减少。
                # 如果对于小于 startind 的 n 而言它消失，则它等于从 n 开始的求和。
                if startind:
                    # 如果 startind 存在
                    for i in reversed(range(startind)):
                        # 反向遍历从 startind 到 0 的范围
                        if not term.subs(n, i):
                            seriesdict[term] = i
                            # 如果 term 在 n=i 处为零，则将其加入 seriesdict
                        else:
                            seriesdict[term] = i + 1
                            break
                            # 否则，将 term 加入 seriesdict，并且将其值设为 i+1，跳出循环
                else:
                    seriesdict[term] = S.Zero
                    # 如果 startind 不存在，则将 term 加入 seriesdict，并且将其值设为零

    # Stripping of terms so that the sum starts with the same number.
    # 剥离项以使求和从相同的数开始。
    teq = S.Zero
    suminit = seriesdict.values()
    rkeys = seriesdict.keys()
    req = Add(*rkeys)
    # 初始化 teq，获取 seriesdict 的值列表和键列表
    if any(suminit):
        # 如果 suminit 中有任何项
        maxval = max(suminit)
        # 找到 suminit 中的最大值
        for term in seriesdict:
            val = seriesdict[term]
            if val != maxval:
                # 如果 val 不等于 maxval
                for i in range(val, maxval):
                    teq += term.subs(n, val)
                    # 将 term 在 n=val 处的值加入 teq

    finaldict = {}
    # 初始化 finaldict 空字典
    if teq:
        fargs = teq.atoms(AppliedUndef)
        # 获取 teq 中的 AppliedUndef 原子
        if len(fargs) == 1:
            finaldict[fargs.pop()] = 0
            # 如果 fargs 只有一个元素，将其加入 finaldict，并设其值为零
        else:
            maxf = max(fargs, key = lambda x: x.args[0])
            # 找到 fargs 中 args[0] 最大的元素
            sol = solve(teq, maxf)
            if isinstance(sol, list):
                sol = sol[0]
            finaldict[maxf] = sol
            # 解 teq 关于 maxf 的方程，并将解加入 finaldict

    # Finding the recurrence relation in terms of the largest term.
    # 在最大项的术语中找到递归关系。
    fargs = req.atoms(AppliedUndef)
    # 获取 req 中的 AppliedUndef 原子
    maxf = max(fargs, key = lambda x: x.args[0])
    # 找到 fargs 中 args[0] 最大的元素
    minf = min(fargs, key = lambda x: x.args[0])
    # 找到 fargs 中 args[0] 最小的元素
    if minf.args[0].is_Symbol:
        startiter = 0
    else:
        startiter = -minf.args[0].as_independent(n)[0]
    # 确定 startiter 的初始值
    lhs = maxf
    rhs =  solve(req, maxf)
    if isinstance(rhs, list):
        rhs = rhs[0]
    # 解 req 关于 maxf 的方程，并将解赋给 rhs

    # Checking how many values are already present
    # 检查已经存在多少个值
    tcounter = len([t for t in finaldict.values() if t])

    for _ in range(tcounter, terms - 3):  # Assuming c0 and c1 to be arbitrary
        # 循环直到 finaldict 中有 terms - 3 个值为止（假设 c0 和 c1 是任意的）
        check = rhs.subs(n, startiter)
        nlhs = lhs.subs(n, startiter)
        nrhs = check.subs(finaldict)
        # 替换 n 后计算 check 和 nlhs，并用 finaldict 替换 nrhs
        finaldict[nlhs] = nrhs
        startiter += 1
        # 将 nlhs 和 nrhs 添加到 finaldict，并增加 startiter

    # Post processing
    # 后处理
    series = C0 + C1*(x - x0)
    # 初始化 series
    for term in finaldict:
        # 遍历 finaldict 中的项
        if finaldict[term]:
            # 如果 finaldict[term] 不为零
            fact = term.args[0]
            # 获取 term 的第一个参数
            series += (finaldict[term].subs([(recurr(0), C0), (recurr(1), C1)])*(
                x - x0)**fact)
            # 将 finaldict[term] 替换为 C0 和 C1，并将结果添加到 series
    series = collect(expand_mul(series), [C0, C1]) + Order(x**terms)
    # 将 series 展开并合并 C0 和 C1，然后加上 x**terms 的阶数
    return Eq(f(x), series)
    # 返回 f(x) 等于 series 的等式
    x = func.args[0]  # 从函数参数中获取自变量 x
    f = func.func  # 获取函数 f
    C0, C1 = get_numbered_constants(eq, num=2)  # 调用函数获取前两个常数 C0 和 C1
    m = Dummy("m")  # 创建一个虚拟变量 m，用于解决指标方程
    x0 = match['x0']  # 从匹配对象中获取 x0，即正则点
    terms = match['terms']  # 从匹配对象中获取项
    p = match['p']  # 从匹配对象中获取 p(x)
    q = match['q']  # 从匹配对象中获取 q(x)

    # 生成指标方程列表
    indicial = []
    # 遍历列表中的 p 和 q
    for term in [p, q]:
        # 检查 term 是否不包含变量 x
        if not term.has(x):
            # 如果不包含 x，则将 term 添加到 indicial 列表中
            indicial.append(term)
        else:
            # 如果包含 x，则调用 series 函数生成一个级数展开
            term = series(term, x=x, n=1, x0=x0)
            # 检查生成的 term 是否是 Order 类型
            if isinstance(term, Order):
                # 如果是 Order 类型，则将 S.Zero 添加到 indicial 列表中
                indicial.append(S.Zero)
            else:
                # 否则遍历 term 的 args
                for arg in term.args:
                    # 如果 arg 不包含变量 x，则将 arg 添加到 indicial 列表中并中断循环
                    if not arg.has(x):
                        indicial.append(arg)
                        break

    # 将 indicial 列表中的第一个和第二个元素分别赋值给 p0 和 q0
    p0, q0 = indicial

    # 解方程 m*(m - 1) + m*p0 + q0 = 0，得到 m 的解列表 sollist
    sollist = solve(m*(m - 1) + m*p0 + q0, m)

    # 检查解列表 sollist 是否存在且为列表类型，且所有解都是实数
    if sollist and isinstance(sollist, list) and all(sol.is_real for sol in sollist):
        # 初始化两个空字典 serdict1 和 serdict2
        serdict1 = {}
        serdict2 = {}

        # 如果解列表只有一个解
        if len(sollist) == 1:
            # 只有一个 Frobenius 级数解存在的情况
            m1 = m2 = sollist.pop()
            # 如果 terms-m1-1 <= 0，则返回一个 Order 类型的方程
            if terms-m1-1 <= 0:
                return Eq(f(x), Order(terms))
            # 调用 _frobenius 函数生成第一个 Frobenius 级数解的字典
            serdict1 = _frobenius(terms-m1-1, m1, p0, q0, p, q, x0, x, C0)

        else:
            # 解列表有两个解 m1 和 m2
            m1 = sollist[0]
            m2 = sollist[1]
            # 如果 m1 < m2，则交换它们的值
            if m1 < m2:
                m1, m2 = m2, m1
            # 无论 m1 - m2 是否为整数，至少存在一个 Frobenius 级数解
            serdict1 = _frobenius(terms-m1-1, m1, p0, q0, p, q, x0, x, C0)
            # 如果 m1 - m2 不是整数，则存在第二个 Frobenius 级数解
            if not (m1 - m2).is_integer:
                serdict2 = _frobenius(terms-m2-1, m2, p0, q0, p, q, x0, x, C1)
            else:
                # 否则检查是否存在第二个 Frobenius 级数解
                serdict2 = _frobenius(terms-m2-1, m2, p0, q0, p, q, x0, x, C1, check=m1)

        # 如果 serdict1 不为空
        if serdict1:
            # 初始化 finalseries1 为 C0
            finalseries1 = C0
            # 遍历 serdict1 中的键
            for key in serdict1:
                # 获取键名中的幂次数
                power = int(key.name[1:])
                # 构建 finalseries1
                finalseries1 += serdict1[key]*(x - x0)**power
            # 乘以 (x - x0)^m1
            finalseries1 = (x - x0)**m1 * finalseries1

            # 初始化 finalseries2 为 S.Zero
            finalseries2 = S.Zero
            # 如果 serdict2 不为空
            if serdict2:
                # 遍历 serdict2 中的键
                for key in serdict2:
                    # 获取键名中的幂次数
                    power = int(key.name[1:])
                    # 构建 finalseries2
                    finalseries2 += serdict2[key]*(x - x0)**power
                # 加上常数项 C1
                finalseries2 += C1
                # 乘以 (x - x0)^m2
                finalseries2 = (x - x0)**m2 * finalseries2

            # 返回整理后的级数方程 Eq(f(x), ...) 加上高阶无穷小 Order(x**terms)
            return Eq(f(x), collect(finalseries1 + finalseries2, [C0, C1]) + Order(x**terms))
# 移除冗余解决方案，保留不同的解决方案。
# 此函数用于确保 dsolve 不会返回冗余的解决方案。例如，当有多种解决方案时，可能会存在特定情况下一个解决方案是另一个的特例，
# 此时我们需要移除这样的特例，只保留不同的解。

def _remove_redundant_solutions(eq, solns, order, var):
    r"""
    Remove redundant solutions from the set of solutions.

    This function is needed because otherwise dsolve can return
    redundant solutions. As an example consider:

        eq = Eq((f(x).diff(x, 2))*f(x).diff(x), 0)

    There are two ways to find solutions to eq. The first is to solve f(x).diff(x, 2) = 0
    leading to solution f(x)=C1 + C2*x. The second is to solve the equation f(x).diff(x) = 0
    leading to the solution f(x) = C1. In this particular case we then see
    that the second solution is a special case of the first and we do not
    want to return it.

    This does not always happen. If we have

        eq = Eq((f(x)**2-4)*(f(x).diff(x)-4), 0)

    then we get the algebraic solution f(x) = [-2, 2] and the integral solution
    f(x) = x + C1 and in this case the two solutions are not equivalent wrt
    initial conditions so both should be returned.
    """
    
    # 判断一个解是否是另一个解的特例
    def is_special_case_of(soln1, soln2):
        return _is_special_case_of(soln1, soln2, eq, order, var)

    # 存储唯一的解决方案
    unique_solns = []
    # 遍历solns列表中的每一个元素soln1
    for soln1 in solns:
        # 遍历当前unique_solns列表的每一个元素soln2（复制一份用于迭代）
        for soln2 in unique_solns[:]:
            # 如果soln1是soln2的一个特殊情况，则终止当前循环
            if is_special_case_of(soln1, soln2):
                break
            # 如果soln2是soln1的一个特殊情况，则从unique_solns中移除soln2
            elif is_special_case_of(soln2, soln1):
                unique_solns.remove(soln2)
        else:
            # 如果内部循环正常结束（未通过break退出），将soln1添加到unique_solns中
            unique_solns.append(soln1)

    # 返回处理后的unique_solns列表，其中包含了solns中独特的解决方案
    return unique_solns
# 判断 soln1 是否是 soln2 的特殊情况，即是否存在某些常数值使得成立。如果是，返回 True；否则返回 False。
def _is_special_case_of(soln1, soln2, eq, order, var):
    r"""
    True if soln1 is found to be a special case of soln2 wrt some value of the
    constants that appear in soln2. False otherwise.
    """
    # 解方程 soln1 和 soln2 的差值，将隐式形式的解转换为显式形式
    soln1 = soln1.rhs - soln1.lhs
    soln2 = soln2.rhs - soln2.lhs

    # 处理级数解的情况
    if soln1.has(Order) and soln2.has(Order):
        if soln1.getO() == soln2.getO():
            soln1 = soln1.removeO()
            soln2 = soln2.removeO()
        else:
            return False
    elif soln1.has(Order) or soln2.has(Order):
        return False

    # 找出 soln1 和 soln2 中的自由符号集合，排除方程 eq 中的符号，得到 soln1 和 soln2 的常数集合
    constants1 = soln1.free_symbols.difference(eq.free_symbols)
    constants2 = soln2.free_symbols.difference(eq.free_symbols)

    # 获取与 soln1 和 soln2 相关的编号常数集合
    constants1_new = get_numbered_constants(Tuple(soln1, soln2), len(constants1))
    if len(constants1) == 1:
        constants1_new = {constants1_new}

    # 将 soln1 中的旧常数替换为新常数
    for c_old, c_new in zip(constants1, constants1_new):
        soln1 = soln1.subs(c_old, c_new)

    # 创建 n+1 个方程，其中第一个方程是 soln1 = soln2，其余为 soln1' = soln2'，...，soln1^(n) = soln2^(n)
    lhs = soln1
    rhs = soln2
    eqns = [Eq(lhs, rhs)]
    for n in range(1, order):
        lhs = lhs.diff(var)
        rhs = rhs.diff(var)
        eq = Eq(lhs, rhs)
        eqns.append(eq)

    # 排除掉无关紧要的布尔值 True 和 False
    if any(isinstance(eq, BooleanFalse) for eq in eqns):
        return False
    eqns = [eq for eq in eqns if not isinstance(eq, BooleanTrue)]

    # 尝试解这组方程，找到与 soln2 中的积分常数对应的解
    try:
        constant_solns = solve(eqns, constants2)
    except NotImplementedError:
        return False

    # 处理 solve 返回结果可能是字典或字典列表的情况
    if isinstance(constant_solns, dict):
        constant_solns = [constant_solns]

    # 针对每个解，检查是否满足所有方程，若不满足则返回 False
    for constant_soln in constant_solns:
        for eq in eqns:
            eq = eq.rhs - eq.lhs
            if checksol(eq, constant_soln) is not True:
                return False

    # 检查是否存在某个解使得 soln2 的所有常数表达式都不依赖于变量 var，若是则返回 True
    for constant_soln in constant_solns:
        if not any(c.has(var) for c in constant_soln.values()):
            return True

    # 若所有解均不符合条件，则返回 False
    return False
# 求解一阶微分方程的幂级数解法，基于幂级数展开给出微分方程的解
def ode_1st_power_series(eq, func, order, match):
    r"""
    幂级数解法是一种通过泰勒级数展开来求解微分方程解的方法。

    对于一阶微分方程 `\frac{dy}{dx} = h(x, y)`，在点 `x = x_{0}` 处存在幂级数解，如果 `h(x, y)` 在 `x_{0}` 处解析。
    解的形式为：

    .. math:: y(x) = y(x_{0}) + \sum_{n = 1}^{\infty} \frac{F_{n}(x_{0},b)(x - x_{0})^n}{n!},

    其中 `y(x_{0}) = b` 是初始值 `x_{0}` 处的 `y` 值。
    为了计算 `F_{n}(x_{0},b)` 的值，遵循以下算法，直到生成所需的项数为止。

    1. `F_1 = h(x_{0}, b)`
    2. `F_{n+1} = \frac{\partial F_{n}}{\partial x} + \frac{\partial F_{n}}{\partial y}F_{1}`

    示例
    ========

    >>> from sympy import Function, pprint, exp, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = exp(x)*(f(x).diff(x)) - f(x)
    >>> pprint(dsolve(eq, hint='1st_power_series'))
                           3       4       5
                       C1*x    C1*x    C1*x     / 6\
    f(x) = C1 + C1*x - ----- + ----- + ----- + O\x /
                         6       24      60


    参考文献
    ==========

    - Travis W. Walker, Analytic power series technique for solving first-order
      differential equations, p.p 17, 18

    """
    # 提取函数的自变量和因变量
    x = func.args[0]
    y = match['y']
    f = func.func
    # 提取微分方程中的函数h(x, y)
    h = -match[match['d']]/match[match['e']]
    # 获取初始点和初始值
    point = match['f0']
    value = match['f0val']
    # 获取需要生成的项数
    terms = match['terms']

    # 第一项
    F = h
    # 如果h为零，则返回常数解
    if not h:
        return Eq(f(x), value)

    # 初始化级数
    series = value
    # 如果需要生成的项数大于1
    if terms > 1:
        # 计算h在初始点的值
        hc = h.subs({x: point, y: value})
        # 如果hc包含无穷大或不是数或复无穷大
        if hc.has(oo) or hc.has(nan) or hc.has(zoo):
            # 导数不存在，不是解析的情况
            return Eq(f(x), oo)
        elif hc:
            series += hc*(x - point)

    # 生成剩余项
    for factcount in range(2, terms):
        # 计算新的F_{n}
        Fnew = F.diff(x) + F.diff(y)*h
        # 计算F_{n}在初始点的值
        Fnewc = Fnew.subs({x: point, y: value})
        # 如果F_{n}在初始点的值包含无穷大或不是数或负无穷大或复无穷大
        if Fnewc.has(oo) or Fnewc.has(nan) or Fnewc.has(-oo) or Fnewc.has(zoo):
            return Eq(f(x), oo)
        # 添加新项到级数中
        series += Fnewc*((x - point)**factcount)/factorial(factcount)
        F = Fnew
    # 添加O(x^terms)到级数中
    series += Order(x**terms)
    return Eq(f(x), series)


# 检查给定的无穷小是否是给定一阶微分方程的实际无穷小解
def checkinfsol(eq, infinitesimals, func=None, order=None):
    r"""
    此函数用于检查给定的无穷小是否是给定一阶微分方程的实际无穷小解。
    此方法特定于ODE的Lie群解法。

    目前，它仅通过将无穷小代入偏微分方程来检查。

    """
    # 这段代码定义了一个多变量微分方程的解析过程，通过计算部分微分方程的形式来计算给定条件下的方程解。
    """
    if isinstance(eq, Equality):
        # 如果输入的方程是一个等式，则转换成方程的左侧减去右侧
        eq = eq.lhs - eq.rhs
    if not func:
        # 如果没有给定函数，则预处理方程并返回处理后的方程和函数
        eq, func = _preprocess(eq)
    variables = func.args
    if len(variables) != 1:
        # 检查函数变量的数量，只能处理一个独立变量的常微分方程
        raise ValueError("ODE's have only one independent variable")
    else:
        # 取出唯一的独立变量
        x = variables[0]
        if not order:
            # 如果未提供方程的阶数，则计算方程的阶数
            order = ode_order(eq, func)
        if order != 1:
            # 只能处理一阶微分方程，如果方程阶数不是1，则报错
            raise NotImplementedError("Lie groups solver has been implemented "
            "only for first order differential equations")
        else:
            # 对函数关于独立变量求导数
            df = func.diff(x)
            # 定义通配符
            a = Wild('a', exclude = [df])
            b = Wild('b', exclude = [df])
            # 将方程展开并匹配模式
            match = collect(expand(eq), df).match(a*df + b)

            if match:
                # 如果匹配成功，则计算微分方程的 h 函数
                h = -simplify(match[b]/match[a])
            else:
                # 如果无法匹配，则尝试求解微分方程
                try:
                    sol = solve(eq, df)
                except NotImplementedError:
                    # 如果求解失败，则报告无法找到微分方程的无穷小项
                    raise NotImplementedError("Infinitesimals for the "
                        "first order ODE could not be found")
                else:
                    # 取得第一个解的无穷小项
                    h = sol[0]  # Find infinitesimals for one solution

            # 定义虚拟变量 y
            y = Dummy('y')
            # 替换 h 中的函数 func 为虚拟变量 y
            h = h.subs(func, y)
            # 定义函数 xi 和 eta
            xi = Function('xi')(x, y)
            eta = Function('eta')(x, y)
            # 定义函数 xi 和 eta 关于 func 的导数
            dxi = Function('xi')(x, func)
            deta = Function('eta')(x, func)
            # 构建偏微分方程
            pde = (eta.diff(x) + (eta.diff(y) - xi.diff(x))*h -
                (xi.diff(y))*h**2 - xi*(h.diff(x)) - eta*(h.diff(y)))
            # 初始化解元组
            soltup = []
            # 对每个给定的无穷小项求解
            for sol in infinitesimals:
                # 构建替换字典
                tsol = {xi: S(sol[dxi]).subs(func, y),
                    eta: S(sol[deta]).subs(func, y)}
                # 对偏微分方程进行替换并简化
                sol = simplify(pde.subs(tsol).doit())
                if sol:
                    # 如果存在解，则添加到解元组中
                    soltup.append((False, sol.subs(y, func)))
                else:
                    # 如果无解，则将 (True, 0) 添加到解元组中
                    soltup.append((True, 0))
            # 返回解元组
            return soltup
    # 从匹配对象中获取函数表达式 x(t) 和 y(t)
    x = match_['func'][0].func
    y = match_['func'][1].func
    # 获取匹配对象中的函数列表和函数系数
    func = match_['func']
    fc = match_['func_coeff']
    # 获取匹配对象中的方程列表
    eq = match_['eq']
    # 初始化结果字典
    r = {}
    # 获取方程中的独立变量 t
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]

    # 根据方程的系数将每个方程标准化为形式 a*diff(x(t),t) = b*x(t) + c*y(t) + k
    for i in range(2):
        eq[i] = Add(*[terms/fc[i,func[i],1] for terms in Add.make_args(eq[i])])

    # 解析方程中的系数 a, b, c, k
    r['a'] = -fc[0,x(t),0]/fc[0,x(t),1]
    r['c'] = -fc[1,x(t),0]/fc[1,y(t),1]
    r['b'] = -fc[0,y(t),0]/fc[0,x(t),1]
    r['d'] = -fc[1,y(t),0]/fc[1,y(t),1]

    # 初始化外力项列表
    forcing = [S.Zero, S.Zero]
    # 提取方程中的非齐次项作为外力项
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j

    # 如果外力项不含时间变量 t，则将其赋给结果字典中的 k1 和 k2
    if not (forcing[0].has(t) or forcing[1].has(t)):
        r['k1'] = forcing[0]
        r['k2'] = forcing[1]
    else:
        # 如果外力项含有时间变量 t，则抛出未实现错误
        raise NotImplementedError("Only homogeneous problems are supported" +
                                  " (and constant inhomogeneity)")

    # 根据方程类型调用相应的求解函数
    if match_['type_of_equation'] == 'type6':
        sol = _linear_2eq_order1_type6(x, y, t, r, eq)
    elif match_['type_of_equation'] == 'type7':
        sol = _linear_2eq_order1_type7(x, y, t, r, eq)
    return sol

# 解类型为 type6 的一阶线性常系数微分方程组
def _linear_2eq_order1_type6(x, y, t, r, eq):
    """
    这种类型的微分方程组的方程为：

    .. math:: x' = f(t) x + g(t) y

    .. math:: y' = a [f(t) + a h(t)] x + a [g(t) - h(t)] y

    通过将第一个方程乘以 `-a` 并加到第二个方程中，得到：

    .. math:: y' - a x' = -a h(t) (y - a x)

    设置 `U = y - ax` 并积分该方程，我们得到

    .. math:: y - ax = C_1 e^{-a \int h(t) \,dt}

    将 y 的值代入第一个方程可以得到一阶常微分方程。解出 `x` 后，再将其值代入第二个方程可以得到 `y`。

    """
    # 获取方程组的常数项
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    p = 0
    q = 0
    # 计算 p1 和 p2 的值
    p1 = cancel(r['c']/cancel(r['c']/r['d']).as_numer_denom()[0])
    p2 = cancel(r['a']/cancel(r['a']/r['b']).as_numer_denom()[0])
    for n, i in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q!=0 and n==0:
                if ((r['c']/j - r['a'])/(r['b'] - r['d']/j)) == j:
                    p = 1
                    s = j
                    break
            if q!=0 and n==1:
                if ((r['a']/j - r['c'])/(r['d'] - r['b']/j)) == j:
                    p = 2
                    s = j
                    break

    # 根据 p 的值选择合适的方程形式进行求解
    if p == 1:
        equ = diff(x(t),t) - r['a']*x(t) - r['b']*(s*x(t) + C1*exp(-s*Integral(r['b'] - r['d']/s, t)))
        hint1 = classify_ode(equ)[1]
        sol1 = dsolve(equ, hint=hint1+'_Integral').rhs
        sol2 = s*sol1 + C1*exp(-s*Integral(r['b'] - r['d']/s, t))
    # 如果条件 p 等于 2，执行以下操作
    elif p == 2:
        # 构造微分方程 equ：dy(t)/dt - r['c']*y(t) - r['d']*s*y(t) + C1*exp(-s*Integral(r['d'] - r['b']/s, t)) = 0
        equ = diff(y(t), t) - r['c']*y(t) - r['d']*s*y(t) + C1*exp(-s*Integral(r['d'] - r['b']/s, t))
        # 利用 classify_ode 函数对微分方程 equ 进行分类，获取其类别标识
        hint1 = classify_ode(equ)[1]
        # 求解微分方程 equ，使用 hint1+'_Integral' 的方法
        sol2 = dsolve(equ, hint=hint1+'_Integral').rhs
        # 计算第二个解 sol1，sol1 = s*sol2 + C1*exp(-s*Integral(r['d'] - r['b']/s, t))
        sol1 = s*sol2 + C1*exp(-s*Integral(r['d'] - r['b']/s, t))
    # 返回方程组 [x(t) = sol1, y(t) = sol2]
    return [Eq(x(t), sol1), Eq(y(t), sol2)]
def _linear_2eq_order1_type7(x, y, t, r, eq):
    r"""
    The equations of this type of ode are .

    .. math:: x' = f(t) x + g(t) y

    .. math:: y' = h(t) x + p(t) y

    Differentiating the first equation and substituting the value of `y`
    from second equation will give a second-order linear equation

    .. math:: g x'' - (fg + gp + g') x' + (fgp - g^{2} h + f g' - f' g) x = 0

    This above equation can be easily integrated if following conditions are satisfied.

    1. `fgp - g^{2} h + f g' - f' g = 0`

    2. `fgp - g^{2} h + f g' - f' g = ag, fg + gp + g' = bg`

    If first condition is satisfied then it is solved by current dsolve solver and in second case it becomes
    a constant coefficient differential equation which is also solved by current solver.

    Otherwise if the above condition fails then,
    a particular solution is assumed as `x = x_0(t)` and `y = y_0(t)`
    Then the general solution is expressed as

    .. math:: x = C_1 x_0(t) + C_2 x_0(t) \int \frac{g(t) F(t) P(t)}{x_0^{2}(t)} \,dt

    .. math:: y = C_1 y_0(t) + C_2 [\frac{F(t) P(t)}{x_0(t)} + y_0(t) \int \frac{g(t) F(t) P(t)}{x_0^{2}(t)} \,dt]

    where C1 and C2 are arbitrary constants and

    .. math:: F(t) = e^{\int f(t) \,dt}, P(t) = e^{\int p(t) \,dt}

    """
    # Get the numbered constants from the equation dictionary
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    
    # Calculate e1 and e2 based on the provided expressions
    e1 = r['a']*r['b']*r['c'] - r['b']**2*r['c'] + r['a']*diff(r['b'],t) - diff(r['a'],t)*r['b']
    e2 = r['a']*r['c']*r['d'] - r['b']*r['c']**2 + diff(r['c'],t)*r['d'] - r['c']*diff(r['d'],t)
    
    # Calculate m1 and m2 based on the provided expressions
    m1 = r['a']*r['b'] + r['b']*r['d'] + diff(r['b'],t)
    m2 = r['a']*r['c'] + r['c']*r['d'] + diff(r['c'],t)
    
    # Check conditions and solve the differential equations accordingly
    if e1 == 0:
        sol1 = dsolve(r['b']*diff(x(t),t,t) - m1*diff(x(t),t)).rhs
        sol2 = dsolve(diff(y(t),t) - r['c']*sol1 - r['d']*y(t)).rhs
    elif e2 == 0:
        sol2 = dsolve(r['c']*diff(y(t),t,t) - m2*diff(y(t),t)).rhs
        sol1 = dsolve(diff(x(t),t) - r['a']*x(t) - r['b']*sol2).rhs
    elif not (e1/r['b']).has(t) and not (m1/r['b']).has(t):
        sol1 = dsolve(diff(x(t),t,t) - (m1/r['b'])*diff(x(t),t) - (e1/r['b'])*x(t)).rhs
        sol2 = dsolve(diff(y(t),t) - r['c']*sol1 - r['d']*y(t)).rhs
    elif not (e2/r['c']).has(t) and not (m2/r['c']).has(t):
        sol2 = dsolve(diff(y(t),t,t) - (m2/r['c'])*diff(y(t),t) - (e2/r['c'])*y(t)).rhs
        sol1 = dsolve(diff(x(t),t) - r['a']*x(t) - r['b']*sol2).rhs
    else:
        # If none of the above conditions match, assume particular solutions x0 and y0
        x0 = Function('x0')(t)    # x0 and y0 being particular solutions
        y0 = Function('y0')(t)
        F = exp(Integral(r['a'],t))
        P = exp(Integral(r['d'],t))
        sol1 = C1*x0 + C2*x0*Integral(r['b']*F*P/x0**2, t)
        sol2 = C1*y0 + C2*(F*P/x0 + y0*Integral(r['b']*F*P/x0**2, t))
    
    # Return the differential equations representing the solutions
    return [Eq(x(t), sol1), Eq(y(t), sol2)]


def sysode_nonlinear_2eq_order1(match_):
    func = match_['func']
    eq = match_['eq']
    fc = match_['func_coeff']
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    # 如果方程的类型是 'type5'
    if match_['type_of_equation'] == 'type5':
        # 调用 _nonlinear_2eq_order1_type5 函数解决非线性二阶方程组类型5
        sol = _nonlinear_2eq_order1_type5(func, t, eq)
        # 返回解决方案
        return sol
    
    # 提取函数 func[0] 和 func[1] 的函数部分，并赋给 x 和 y
    x = func[0].func
    y = func[1].func
    
    # 对于 i 从 0 到 1 的循环
    for i in range(2):
        eqs = 0
        # 对于 Add.make_args(eq[i]) 中的每一项 terms
        for terms in Add.make_args(eq[i]):
            # 将 terms 除以 fc[i,func[i],1] 后加到 eqs 中
            eqs += terms / fc[i, func[i], 1]
        # 更新 eq[i] 为处理后的值
        eq[i] = eqs
    
    # 根据匹配类型 'type_of_equation' 的值选择合适的函数解决非线性二阶方程组
    if match_['type_of_equation'] == 'type1':
        sol = _nonlinear_2eq_order1_type1(x, y, t, eq)
    elif match_['type_of_equation'] == 'type2':
        sol = _nonlinear_2eq_order1_type2(x, y, t, eq)
    elif match_['type_of_equation'] == 'type3':
        sol = _nonlinear_2eq_order1_type3(x, y, t, eq)
    elif match_['type_of_equation'] == 'type4':
        sol = _nonlinear_2eq_order1_type4(x, y, t, eq)
    
    # 返回计算得到的解决方案
    return sol
def _nonlinear_2eq_order1_type3(x, y, t, eq):
    r"""
    Autonomous system of general form

    .. math:: x' = F(x,y)

    .. math:: y' = G(x,y)

    Assuming `y = y(x, C_1)` where `C_1` is an arbitrary constant is the general
    solution of the first-order equation

    .. math:: F(x,y) y'_x = G(x,y)

    Then the general solution of the original system of equations has the form

    .. math:: \int \frac{1}{F(x,y(x,C_1))} \,dx = t + C_1

    """
    # 获取方程组中的常数项 C1, C2, C3, C4
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    
    # 定义函数 v 和符号 u
    v = Function('v')
    u = Symbol('u')
    
    # 匹配方程的左侧表达式
    f = Wild('f')
    g = Wild('g')
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    
    # 替换 F 和 G 中的变量 x(t), y(t) 为 u 和 v(u)
    F = r1[f].subs(x(t), u).subs(y(t), v(u))
    G = r2[g].subs(x(t), u).subs(y(t), v(u))
    # 使用 dsolve 函数求解微分方程 Eq(diff(v(u), u), G/F)，返回解集 sol2r
    sol2r = dsolve(Eq(diff(v(u), u), G/F))
    
    # 如果 sol2r 是 Equality 类型的对象，则转换为列表，方便后续处理
    if isinstance(sol2r, Equality):
        sol2r = [sol2r]
    
    # 遍历 sol2r 中的每个解对象 sol2s
    for sol2s in sol2r:
        # 解方程 Integral(1/F.subs(v(u), sol2s.rhs), u) - t - C2 = 0，返回解集 sol1
        sol1 = solve(Integral(1/F.subs(v(u), sol2s.rhs), u).doit() - t - C2, u)
    
    # 初始化空列表 sol 用于存储最终的解集
    sol = []
    
    # 遍历 sol1 中的每个解对象 sols
    for sols in sol1:
        # 构造关于 x(t) 的方程 Eq(x(t), sols) 并加入 sol 列表
        sol.append(Eq(x(t), sols))
        # 构造关于 y(t) 的方程 Eq(y(t), (sol2s.rhs).subs(u, sols)) 并加入 sol 列表
        sol.append(Eq(y(t), (sol2s.rhs).subs(u, sols)))
    
    # 返回最终的解集 sol
    return sol
def _nonlinear_2eq_order1_type4(x, y, t, eq):
    r"""
    Equation:

    .. math:: x' = f_1(x) g_1(y) \phi(x,y,t)

    .. math:: y' = f_2(x) g_2(y) \phi(x,y,t)

    First integral:

    .. math:: \int \frac{f_2(x)}{f_1(x)} \,dx - \int \frac{g_1(y)}{g_2(y)} \,dy = C

    where `C` is an arbitrary constant.

    On solving the first integral for `x` (resp., `y` ) and on substituting the
    resulting expression into either equation of the original solution, one
    arrives at a first-order equation for determining `y` (resp., `x` ).

    """
    # 从方程中获取两个数值常数
    C1, C2 = get_numbered_constants(eq, num=2)
    # 定义符号变量 u, v 以及函数符号变量 U, V
    u, v = symbols('u, v')
    U, V = symbols('U, V', cls=Function)
    # 创建 Wild 匹配对象，用于匹配表达式中的模式
    f = Wild('f')
    g = Wild('g')
    f1 = Wild('f1', exclude=[v,t])
    f2 = Wild('f2', exclude=[v,t])
    g1 = Wild('g1', exclude=[u,t])
    g2 = Wild('g2', exclude=[u,t])
    # 匹配第一个方程的形式
    r1 = eq[0].match(diff(x(t),t) - f)
    # 匹配第二个方程的形式
    r2 = eq[1].match(diff(y(t),t) - g)
    # 计算两个匹配结果的比值
    num, den = (
        (r1[f].subs(x(t),u).subs(y(t),v))/
        (r2[g].subs(x(t),u).subs(y(t),v))).as_numer_denom()
    # 匹配分别匹配到的部分
    R1 = num.match(f1*g1)
    R2 = den.match(f2*g2)
    # 计算 phi 函数
    phi = (r1[f].subs(x(t),u).subs(y(t),v))/num
    F1 = R1[f1]; F2 = R2[f2]
    G1 = R1[g1]; G2 = R2[g2]
    # 解第一个积分得到的结果
    sol1r = solve(Integral(F2/F1, u).doit() - Integral(G1/G2,v).doit() - C1, u)
    # 解第二个积分得到的结果
    sol2r = solve(Integral(F2/F1, u).doit() - Integral(G1/G2,v).doit() - C1, v)
    sol = []
    # 遍历第一个解的结果
    for sols in sol1r:
        # 将得到的结果添加到解的列表中
        sol.append(Eq(y(t), dsolve(diff(V(t),t) - F2.subs(u,sols).subs(v,V(t))*G2.subs(v,V(t))*phi.subs(u,sols).subs(v,V(t))).rhs))
    # 遍历第二个解的结果
    for sols in sol2r:
        # 将得到的结果添加到解的列表中
        sol.append(Eq(x(t), dsolve(diff(U(t),t) - F1.subs(u,U(t))*G1.subs(v,sols).subs(u,U(t))*phi.subs(v,sols).subs(u,U(t))).rhs))
    # 返回结果集合
    return set(sol)

def _nonlinear_2eq_order1_type5(func, t, eq):
    r"""
    Clairaut system of ODEs

    .. math:: x = t x' + F(x',y')

    .. math:: y = t y' + G(x',y')

    The following are solutions of the system

    `(i)` straight lines:

    .. math:: x = C_1 t + F(C_1, C_2), y = C_2 t + G(C_1, C_2)

    where `C_1` and `C_2` are arbitrary constants;

    `(ii)` envelopes of the above lines;

    `(iii)` continuously differentiable lines made up from segments of the lines
    `(i)` and `(ii)`.

    """
    # 从方程中获取两个数值常数
    C1, C2 = get_numbered_constants(eq, num=2)
    # 创建 Wild 匹配对象，用于匹配表达式中的模式
    f = Wild('f')
    g = Wild('g')
    def check_type(x, y):
        # 尝试匹配第一种类型的方程形式
        r1 = eq[0].match(t*diff(x(t),t) - x(t) + f)
        r2 = eq[1].match(t*diff(y(t),t) - y(t) + g)
        # 如果匹配不成功，则尝试第二种类型的方程形式
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t),t) - x(t)/t + f/t)
            r2 = eq[1].match(diff(y(t),t) - y(t)/t + g/t)
        # 如果匹配不成功，则尝试第三种类型的方程形式（取反）
        if not (r1 and r2):
            r1 = (-eq[0]).match(t*diff(x(t),t) - x(t) + f)
            r2 = (-eq[1]).match(t*diff(y(t),t) - y(t) + g)
        # 如果匹配不成功，则尝试第四种类型的方程形式（取反）
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t),t) - x(t)/t + f/t)
            r2 = (-eq[1]).match(diff(y(t),t) - y(t)/t + g/t)
        # 返回匹配结果列表
        return [r1, r2]
    # 遍历 func 列表中的每个元素 func_
    for func_ in func:
        # 检查 func_ 是否为列表类型
        if isinstance(func_, list):
            # 获取 func 列表中第一个元素的第一个属性 func，并赋值给 x
            x = func[0][0].func
            # 获取 func 列表中第一个元素的第二个属性 func，并赋值给 y
            y = func[0][1].func
            # 调用 check_type 函数，检查 x 和 y 的类型
            [r1, r2] = check_type(x, y)
            # 如果 r1 和 r2 不同时为 True，则交换 x 和 y 并再次检查类型
            if not (r1 and r2):
                [r1, r2] = check_type(y, x)
                x, y = y, x
    # 计算 x 对 t 的导数，并赋值给 x1；计算 y 对 t 的导数，并赋值给 y1
    x1 = diff(x(t), t); y1 = diff(y(t), t)
    # 返回一个包含两个方程的字典，表示 x(t) 和 y(t) 的微分方程
    return {Eq(x(t), C1*t + r1[f].subs(x1, C1).subs(y1, C2)), Eq(y(t), C2*t + r2[g].subs(x1, C1).subs(y1, C2))}
# 解决非线性三阶一阶微分方程组的函数，根据给定的匹配信息 `match_` 进行处理
def sysode_nonlinear_3eq_order1(match_):
    # 从匹配信息中获取方程组中各个函数的表达式
    x = match_['func'][0].func
    y = match_['func'][1].func
    z = match_['func'][2].func
    # 获取方程组的表达式
    eq = match_['eq']
    # 从第一个方程的导数中获取独立变量 t
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]

    # 根据方程的类型选择相应的求解函数
    if match_['type_of_equation'] == 'type1':
        sol = _nonlinear_3eq_order1_type1(x, y, z, t, eq)
    elif match_['type_of_equation'] == 'type2':
        sol = _nonlinear_3eq_order1_type2(x, y, z, t, eq)
    elif match_['type_of_equation'] == 'type3':
        sol = _nonlinear_3eq_order1_type3(x, y, z, t, eq)
    elif match_['type_of_equation'] == 'type4':
        sol = _nonlinear_3eq_order1_type4(x, y, z, t, eq)
    elif match_['type_of_equation'] == 'type5':
        sol = _nonlinear_3eq_order1_type5(x, y, z, t, eq)

    # 返回求解结果
    return sol

# 处理非线性三阶一阶微分方程组类型1的函数
def _nonlinear_3eq_order1_type1(x, y, z, t, eq):
    """
    Equations:

    .. math:: a x' = (b - c) y z, \enspace b y' = (c - a) z x, \enspace c z' = (a - b) x y

    First Integrals:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2

    其中 `C_1` 和 `C_2` 是任意常数。通过解这些积分得到 `y` 和 `z` 的表达式，并将结果代入第一个方程，得到关于 `x` 的可分离一阶方程。类似地处理其他两个方程，可以得到关于 `y` 和 `z` 的一阶方程。

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0401.pdf
    """

    # 获取方程组中的常数项 C1 和 C2
    C1, C2 = get_numbered_constants(eq, num=2)
    # 定义新的符号 u, v, w
    u, v, w = symbols('u, v, w')
    # 定义匹配模式
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    # 匹配方程组的导数
    r = (diff(x(t),t) - eq[0]).match(p*y(t)*z(t))
    r.update((diff(y(t),t) - eq[1]).match(q*z(t)*x(t)))
    r.update((diff(z(t),t) - eq[2]).match(s*x(t)*y(t)))
    n1, d1 = r[p].as_numer_denom()
    n2, d2 = r[q].as_numer_denom()
    n3, d3 = r[s].as_numer_denom()
    # 解方程组得到的变量值
    val = solve([n1*u-d1*v+d1*w, d2*u+n2*v-d2*w, d3*u-d3*v-n3*w],[u,v])
    vals = [val[v], val[u]]
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])
    b = vals[0].subs(w, c)
    a = vals[1].subs(w, c)
    # 计算对应的 sqrt 表达式
    y_x = sqrt(((c*C1-C2) - a*(c-a)*x(t)**2)/(b*(c-b)))
    z_x = sqrt(((b*C1-C2) - a*(b-a)*x(t)**2)/(c*(b-c)))
    z_y = sqrt(((a*C1-C2) - b*(a-b)*y(t)**2)/(c*(a-c)))
    x_y = sqrt(((c*C1-C2) - b*(c-b)*y(t)**2)/(a*(c-a)))
    x_z = sqrt(((b*C1-C2) - c*(b-c)*z(t)**2)/(a*(b-a)))
    y_z = sqrt(((a*C1-C2) - c*(a-c)*z(t)**2)/(b*(a-b)))
    # 求解三个分离的一阶方程
    sol1 = dsolve(a*diff(x(t),t) - (b-c)*y_x*z_x)
    sol2 = dsolve(b*diff(y(t),t) - (c-a)*z_y*x_y)
    sol3 = dsolve(c*diff(z(t),t) - (a-b)*x_z*y_z)
    # 返回结果列表
    return [sol1, sol2, sol3]

# 处理非线性三阶一阶微分方程组类型2的函数
def _nonlinear_3eq_order1_type2(x, y, z, t, eq):
    """
    Equations:

    .. math:: a x' = (b - c) y z f(x, y, z, t)

    .. math:: b y' = (c - a) z x f(x, y, z, t)

    .. math:: c z' = (a - b) x y f(x, y, z, t)

    First Integrals:
    """
    """
    解方程组，并返回其一次解
    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2

    其中 `C_1` 和 `C_2` 是任意常数。在解 `y` 和 `z` 的积分，并将结果表达式代入系统的第一个方程，
    我们得到关于 `x` 的一阶微分方程。类似地，对其他两个方程做同样处理，我们得到关于 `y` 和 `z` 的一阶微分方程。

    参考资料：
    https://eqworld.ipmnet.ru/en/solutions/sysode/sode0402.pdf
    """

    # 获取方程中的两个编号常数 C1 和 C2
    C1, C2 = get_numbered_constants(eq, num=2)

    # 定义符号变量 u, v, w
    u, v, w = symbols('u, v, w')

    # 定义 Wild 对象，用于匹配方程中的模式，排除已知变量和时间 t
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')

    # 匹配微分方程的一阶解
    r1 = (diff(x(t),t) - eq[0]).match(y(t)*z(t)*f)
    r = collect_const(r1[f]).match(p*f)
    r.update(((diff(y(t),t) - eq[1])/r[f]).match(q*z(t)*x(t)))
    r.update(((diff(z(t),t) - eq[2])/r[f]).match(s*x(t)*y(t)))

    # 将匹配到的表达式分别作为分子和分母
    n1, d1 = r[p].as_numer_denom()
    n2, d2 = r[q].as_numer_denom()
    n3, d3 = r[s].as_numer_denom()

    # 解方程组得到 u, v 的值
    val = solve([n1*u-d1*v+d1*w, d2*u+n2*v-d2*w, -d3*u+d3*v+n3*w],[u,v])
    vals = [val[v], val[u]]

    # 计算最小公倍数
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])

    # 计算常数 a, b
    a = vals[0].subs(w, c)
    b = vals[1].subs(w, c)

    # 计算不同变量之间的关系
    y_x = sqrt(((c*C1-C2) - a*(c-a)*x(t)**2)/(b*(c-b)))
    z_x = sqrt(((b*C1-C2) - a*(b-a)*x(t)**2)/(c*(b-c)))
    z_y = sqrt(((a*C1-C2) - b*(a-b)*y(t)**2)/(c*(a-c)))
    x_y = sqrt(((c*C1-C2) - b*(c-b)*y(t)**2)/(a*(c-a)))
    x_z = sqrt(((b*C1-C2) - c*(b-c)*z(t)**2)/(a*(b-a)))
    y_z = sqrt(((a*C1-C2) - c*(a-c)*z(t)**2)/(b*(a-b)))

    # 求解微分方程
    sol1 = dsolve(a*diff(x(t),t) - (b-c)*y_x*z_x*r[f])
    sol2 = dsolve(b*diff(y(t),t) - (c-a)*z_y*x_y*r[f])
    sol3 = dsolve(c*diff(z(t),t) - (a-b)*x_z*y_z*r[f])

    # 返回结果
    return [sol1, sol2, sol3]
def _nonlinear_3eq_order1_type3(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: x' = c F_2 - b F_3, \enspace y' = a F_3 - c F_1, \enspace z' = b F_1 - a F_2

    where `F_n = F_n(x, y, z, t)`.

    1. First Integral:

    .. math:: a x + b y + c z = C_1,

    where C is an arbitrary constant.

    2. If we assume function `F_n` to be independent of `t`, i.e., `F_n = F_n(x, y, z)`
    Then, on eliminating `t` and `z` from the first two equations of the system, one
    arrives at the first-order equation

    .. math:: \frac{dy}{dx} = \frac{a F_3 (x, y, z) - c F_1 (x, y, z)}{c F_2 (x, y, z) -
                b F_3 (x, y, z)}

    where `z = \frac{1}{c} (C_1 - a x - b y)`

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0404.pdf

    """
    # 获取系统方程的第一个积分常数
    C1 = get_numbered_constants(eq, num=1)
    # 定义新的符号变量 u, v, w
    u, v, w = symbols('u, v, w')
    # 定义 u, v, w 的函数符号
    fu, fv, fw = symbols('u, v, w', cls=Function)
    # 定义排除 x(t), y(t), z(t), t 的 Wildcard 符号
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    # 定义 F1, F2, F3 的 Wildcard 符号
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    # 匹配 x'(t) 的表达式，并得到匹配结果 r1
    r1 = (diff(x(t), t) - eq[0]).match(F2 - F3)
    # 收集并匹配常数因子，更新匹配结果 r
    r = collect_const(r1[F2]).match(s * F2)
    r.update(collect_const(r1[F3]).match(q * F3))
    # 如果第二个方程包含 r[F2] 而不包含 r[F3]，则交换它们
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        r[F2], r[F3] = r[F3], r[F2]
        r[s], r[q] = -r[q], -r[s]
    # 更新并匹配第二个方程中的参数 p, q, s
    r.update((diff(y(t), t) - eq[1]).match(p * r[F3] - r[s] * F1))
    # 提取匹配结果中的系数 a, b, c
    a = r[p]; b = r[q]; c = r[s]
    # 分别用 u, v, w 替换 F1, F2, F3 中的 x(t), y(t), z(t)，得到新的 F1, F2, F3
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    # 计算 z = (C1 - a*u - b*v) / c
    z_xy = (C1 - a * u - b * v) / c
    # 计算 y = dsolve(dy/du - ((a*F3 - c*F1)/(c*F2 - b*F3)).subs(w,z_xy).subs(v,fv(u))).rhs
    y_x = dsolve(diff(fv(u), u) - ((a * F3 - c * F1) / (c * F2 - b * F3)).subs(w, z_xy).subs(v, fv(u))).rhs
    # 计算 z = dsolve(dz/du - ((b*F1 - a*F2)/(c*F2 - b*F3)).subs(v,y_zx).subs(w,fw(u))).rhs
    z_x = dsolve(diff(fw(u), u) - ((b * F1 - a * F2) / (c * F2 - b * F3)).subs(v, y_zx).subs(w, fw(u))).rhs
    # 计算 z = dsolve(dz/du - ((b*F1 - a*F2)/(a*F3 - c*F1)).subs(u,x_yz).subs(w,fw(v))).rhs
    z_y = dsolve(diff(fw(v), v) - ((b * F1 - a * F2) / (a * F3 - c * F1)).subs(u, x_yz).subs(w, fw(v))).rhs
    # 计算 y = dsolve(dy/du - ((a*F3 - c*F1)/(a*F3 - c*F1)).subs(w,z_xy).subs(u,fu(v))).rhs
    y_x = dsolve(diff(fv(v), v) - ((a * F3 - c * F1) / (a * F3 - c * F1)).subs(w, z_xy).subs(u, fv(v))).rhs
    # 计算 z = dsolve(dz/du - ((b*F1 - a*F2)/(b*F1 - a*F2)).subs(v,y_zx).subs(u,fu(w))).rhs
    z_x = dsolve(diff(fw(w), w) - ((b * F1 - a * F2) / (b * F1 - a * F2)).subs(v, y_zx).subs(u, fw(w))).rhs
    # 计算 x = dsolve(dx/du - (c*F2 - b*F3).subs(v,y_x).subs(w,z_x).subs(u,fu(t))).rhs
    sol1 = dsolve(diff(fu(t), t) - (c * F2 - b * F3).subs(v, y_x).subs(w, z_x).subs(u, fu(t))).rhs
    # 计算 x = dsolve(dx/du - (a*F3 - c*F1).subs(u,x_y).subs(w,z_y).subs(v,fv(t))).rhs
    sol2 = dsolve(diff(fv(t), t) - (a * F3 - c * F1).subs(u, x_y).subs(w, z_y).subs(v, fv(t))).rhs
    # 计算 x = dsolve(dx/du - (b*F1 - a*F2).subs(u,x_z).subs(v,y_z).subs(w,fw(t))).rhs
    sol3 = dsolve(diff(fw(t), t) - (b * F1 - a * F2).subs(u, x_z).subs(v, y_z).subs(w, fw(t))).rhs
    # 返回结果列表
    return [sol1, sol2, sol3]
    """
    C1 = get_numbered_constants(eq, num=1)
    # 从方程中获取常数C1
    
    u, v, w = symbols('u, v, w')
    # 定义符号变量u, v, w
    
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    # 定义通配符p, q, s，排除已有的函数x(t), y(t), z(t), t
    
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    # 定义通配符符号F1, F2, F3
    
    r1 = eq[0].match(diff(x(t),t) - z(t)*F2 + y(t)*F3)
    # 匹配方程的第一个项并存储到r1中
    
    r = collect_const(r1[F2]).match(s*F2)
    # 收集r1中F2项的常数并存储到r中
    
    r.update(collect_const(r1[F3]).match(q*F3))
    # 更新r中F3项的常数并存储到r中
    
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        # 如果方程eq[1]包含r中的F2项但不包含r中的F3项
        r[F2], r[F3] = r[F3], r[F2]
        # 交换r中的F2和F3项
        r[s], r[q] = -r[q], -r[s]
        # 调整r中的s和q项的符号
    
    r.update((diff(y(t),t) - eq[1]).match(p*x(t)*r[F3] - r[s]*z(t)*F1))
    # 更新r中的p, F3, s, z(t), F1项
    
    a = r[p]; b = r[q]; c = r[s]
    # 将r中的p, q, s项分别赋值给a, b, c
    
    F1 = r[F1].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    F2 = r[F2].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    F3 = r[F3].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    # 将r中的F1, F2, F3项分别替换为u, v, w，并赋值给F1, F2, F3
    
    x_yz = sqrt((C1 - b*v**2 - c*w**2)/a)
    # 计算x_yz
    
    y_zx = sqrt((C1 - c*w**2 - a*u**2)/b)
    # 计算y_zx
    
    z_xy = sqrt((C1 - a*u**2 - b*v**2)/c)
    # 计算z_xy
    
    y_x = dsolve(diff(v(u),u) - ((a*u*F3-c*w*F1)/(c*w*F2-b*v*F3)).subs(w,z_xy).subs(v,v(u))).rhs
    # 求解y_x的微分方程
    
    z_x = dsolve(diff(w(u),u) - ((b*v*F1-a*u*F2)/(c*w*F2-b*v*F3)).subs(v,y_zx).subs(w,w(u))).rhs
    # 求解z_x的微分方程
    
    z_y = dsolve(diff(w(v),v) - ((b*v*F1-a*u*F2)/(a*u*F3-c*w*F1)).subs(u,x_yz).subs(w,w(v))).rhs
    # 求解z_y的微分方程
    
    x_y = dsolve(diff(u(v),v) - ((c*w*F2-b*v*F3)/(a*u*F3-c*w*F1)).subs(w,z_xy).subs(u,u(v))).rhs
    # 求解x_y的微分方程
    
    y_z = dsolve(diff(v(w),w) - ((a*u*F3-c*w*F1)/(b*v*F1-a*u*F2)).subs(u,x_yz).subs(v,v(w))).rhs
    # 求解y_z的微分方程
    
    x_z = dsolve(diff(u(w),w) - ((c*w*F2-b*v*F3)/(b*v*F1-a*u*F2)).subs(v,y_zx).subs(u,u(w))).rhs
    # 求解x_z的微分方程
    
    sol1 = dsolve(diff(u(t),t) - (c*w*F2 - b*v*F3).subs(v,y_x).subs(w,z_x).subs(u,u(t))).rhs
    # 求解sol1的微分方程
    
    sol2 = dsolve(diff(v(t),t) - (a*u*F3 - c*w*F1).subs(u,x_y).subs(w,z_y).subs(v,v(t))).rhs
    # 求解sol2的微分方程
    
    sol3 = dsolve(diff(w(t),t) - (b*v*F1 - a*u*F2).subs(u,x_z).subs(v,y_z).subs(w,w(t))).rhs
    # 求解sol3的微分方程
    
    return [sol1, sol2, sol3]
    # 返回求解的结果列表
    """
# 定义一个函数用于解决非线性常微分方程组的一种类型
def _nonlinear_3eq_order1_type5(x, y, z, t, eq):
    r"""
    .. math:: x' = x (c F_2 - b F_3), \enspace y' = y (a F_3 - c F_1), \enspace z' = z (b F_1 - a F_2)

    where `F_n = F_n (x, y, z, t)` and are arbitrary functions.

    First Integral:

    .. math:: \left|x\right|^{a} \left|y\right|^{b} \left|z\right|^{c} = C_1

    where `C` is an arbitrary constant. If the function `F_n` is independent of `t`,
    then, by eliminating `t` and `z` from the first two equations of the system, one
    arrives at a first-order equation.

    References
    ==========
    - https://eqworld.ipmnet.ru/en/solutions/sysode/sode0406.pdf

    """
    # 获取方程中的常数C1
    C1 = get_numbered_constants(eq, num=1)
    # 定义符号变量u, v, w
    u, v, w = symbols('u, v, w')
    # 定义函数符号fu, fv, fw
    fu, fv, fw = symbols('u, v, w', cls=Function)
    # 定义通配符p, q, s，排除了方程中的x(t), y(t), z(t), t
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    # 定义通配符F1, F2, F3
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    # 匹配方程的第一个式子，得到关于x(t)的表达式r1
    r1 = eq[0].match(diff(x(t), t) - x(t)*F2 + x(t)*F3)
    # 收集常数项F2并匹配通配符s
    r = collect_const(r1[F2]).match(s*F2)
    # 更新收集常数项F3并匹配通配符q
    r.update(collect_const(r1[F3]).match(q*F3))
    # 如果方程2包含r[F2]但不包含r[F3]，则交换它们
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        r[F2], r[F3] = r[F3], r[F2]
        r[s], r[q] = -r[q], -r[s]
    # 更新匹配方程的第二个式子，得到关于y(t)和z(t)的表达式
    r.update((diff(y(t), t) - eq[1]).match(y(t)*(p*r[F3] - r[s]*F1)))
    # 分别赋值a, b, c为r[p], r[q], r[s]
    a = r[p]; b = r[q]; c = r[s]
    # 替换F1, F2, F3的表达式，将x(t), y(t), z(t)替换为u, v, w
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    # 计算三个第一积分条件
    x_yz = (C1*v**-b*w**-c)**-a
    y_zx = (C1*w**-c*u**-a)**-b
    z_xy = (C1*u**-a*v**-b)**-c
    # 求解对应的微分方程，得到各个变量的解函数
    y_x = dsolve(diff(fv(u), u) - ((v*(a*F3 - c*F1))/(u*(c*F2 - b*F3))).subs(w, z_xy).subs(v, fv(u))).rhs
    z_x = dsolve(diff(fw(u), u) - ((w*(b*F1 - a*F2))/(u*(c*F2 - b*F3))).subs(v, y_zx).subs(w, fw(u))).rhs
    z_y = dsolve(diff(fw(v), v) - ((w*(b*F1 - a*F2))/(v*(a*F3 - c*F1))).subs(u, x_yz).subs(w, fw(v))).rhs
    x_y = dsolve(diff(fu(v), v) - ((u*(c*F2 - b*F3))/(v*(a*F3 - c*F1))).subs(w, z_xy).subs(u, fu(v))).rhs
    y_z = dsolve(diff(fv(w), w) - ((v*(a*F3 - c*F1))/(w*(b*F1 - a*F2))).subs(u, x_yz).subs(v, fv(w))).rhs
    x_z = dsolve(diff(fu(w), w) - ((u*(c*F2 - b*F3))/(w*(b*F1 - a*F2))).subs(v, y_zx).subs(u, fu(w))).rhs
    # 求解原方程组得到的解
    sol1 = dsolve(diff(fu(t), t) - (u*(c*F2 - b*F3)).subs(v, y_x).subs(w, z_x).subs(u, fu(t))).rhs
    sol2 = dsolve(diff(fv(t), t) - (v*(a*F3 - c*F1)).subs(u, x_y).subs(w, z_y).subs(v, fv(t))).rhs
    sol3 = dsolve(diff(fw(t), t) - (w*(b*F1 - a*F2)).subs(u, x_z).subs(v, y_z).subs(w, fw(t))).rhs
    # 返回解组成的列表
    return [sol1, sol2, sol3]

# 在底部导入以避免循环导入问题
from .single import SingleODEProblem, SingleODESolver, solver_map
```