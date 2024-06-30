# `D:\src\scipysrc\sympy\sympy\solvers\ode\single.py`

```
# 导入必要的模块和类
from __future__ import annotations
from typing import ClassVar, Iterator

# 导入自定义的模块和函数
from .riccati import match_riccati, solve_riccati
from .hypergeometric import equivalence_hypergeometric, match_2nd_2F1_hypergeometric, \
    get_sol_2F1_hypergeometric, match_2nd_hypergeometric
from .nonhomogeneous import _get_euler_characteristic_eq_sols, _get_const_characteristic_eq_sols, \
    _solve_undetermined_coefficients, _solve_variation_of_parameters, _test_term, _undetermined_coefficients_match, \
    _get_simplified_sol
from .lie_group import _ode_lie_group

# 导入 sympy 中的核心模块和函数
from sympy.core import Add, S, Pow, Rational
from sympy.core.cache import cached_property
from sympy.core.exprtools import factor_terms
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, diff, Function, expand, Subs, _mexpand
from sympy.core.numbers import zoo
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Dummy, Wild
from sympy.core.mul import Mul
from sympy.functions import exp, tan, log, sqrt, besselj, bessely, cbrt, airyai, airybi
from sympy.integrals import Integral
from sympy.polys import Poly
from sympy.polys.polytools import cancel, factor, degree
from sympy.simplify import collect, simplify, separatevars, logcombine, posify  # type: ignore
from sympy.simplify.radsimp import fraction
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.solvers.deutils import ode_order, _preprocess
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.solvers import PolyNonlinearError

# 引入 SingleODESolver 类的错误
class ODEMatchError(NotImplementedError):
    """Raised if a SingleODESolver is asked to solve an ODE it does not match"""
    pass

# 定义一个表示单个常微分方程问题的类
class SingleODEProblem:
    """Represents an ordinary differential equation (ODE)

    This class is used internally in the by dsolve and related
    functions/classes so that properties of an ODE can be computed
    efficiently.

    Examples
    ========

    This class is used internally by dsolve. To instantiate an instance
    directly first define an ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now you can create a SingleODEProblem instance and query its properties:

    >>> from sympy.solvers.ode.single import SingleODEProblem
    >>> problem = SingleODEProblem(f(x).diff(x), f(x), x)
    >>> problem.eq
    Derivative(f(x), x)
    >>> problem.func
    f(x)
    >>> problem.sym
    x
    """

    # 实例属性:
    eq = None  # type: Expr         # ODE 方程的表达式
    func = None  # type: AppliedUndef  # 函数表达式
    sym = None  # type: Symbol      # 自变量符号
    _order = None  # type: int      # ODE 的阶数
    _eq_expanded = None  # type: Expr  # 展开后的方程表达式
    _eq_preprocessed = None  # type: Expr  # 预处理后的方程表达式
    _eq_high_order_free = None  # 去高阶项后的方程表达式
    def __init__(self, eq, func, sym, prep=True, **kwargs):
        # 构造函数，初始化对象
        assert isinstance(eq, Expr)  # 断言确保 eq 是 Expr 类型的对象
        assert isinstance(func, AppliedUndef)  # 断言确保 func 是 AppliedUndef 类型的对象
        assert isinstance(sym, Symbol)  # 断言确保 sym 是 Symbol 类型的对象
        assert isinstance(prep, bool)  # 断言确保 prep 是布尔类型
        self.eq = eq  # 将 eq 参数赋值给对象的 eq 属性
        self.func = func  # 将 func 参数赋值给对象的 func 属性
        self.sym = sym  # 将 sym 参数赋值给对象的 sym 属性
        self.prep = prep  # 将 prep 参数赋值给对象的 prep 属性
        self.params = kwargs  # 将额外的关键字参数赋值给对象的 params 属性

    @cached_property
    def order(self) -> int:
        # 返回微分方程的阶数
        return ode_order(self.eq, self.func)

    @cached_property
    def eq_preprocessed(self) -> Expr:
        # 返回预处理后的微分方程
        return self._get_eq_preprocessed()

    @cached_property
    def eq_high_order_free(self) -> Expr:
        # 尝试消除高阶导数中的函数 f(x) 的前提条件
        a = Wild('a', exclude=[self.func])
        c1 = Wild('c1', exclude=[self.sym])
        reduced_eq = None
        if self.eq.is_Add:
            # 如果方程是加法表达式
            deriv_coef = self.eq.coeff(self.func.diff(self.sym, self.order))
            # 获取最高阶导数的系数
            if deriv_coef not in (1, 0):
                r = deriv_coef.match(a*self.func**c1)
                # 匹配模式，尝试提取 a*f(x)^c1 形式的模式
                if r and r[c1]:
                    den = self.func**r[c1]
                    reduced_eq = Add(*[arg/den for arg in self.eq.args])
                    # 如果成功匹配，则将方程中的每一项除以 den
        if not reduced_eq:
            reduced_eq = expand(self.eq)
            # 如果未成功消除，则展开方程
        return reduced_eq

    @cached_property
    def eq_expanded(self) -> Expr:
        # 返回扩展后的预处理微分方程
        return expand(self.eq_preprocessed)

    def _get_eq_preprocessed(self) -> Expr:
        # 获取预处理后的微分方程
        if self.prep:
            process_eq, process_func = _preprocess(self.eq, self.func)
            # 如果需要预处理，则调用 _preprocess 处理方程和函数
            if process_func != self.func:
                raise ValueError
                # 如果处理后的函数不等于原始函数，则引发 ValueError 异常
        else:
            process_eq = self.eq
            # 如果不需要预处理，则直接使用原始方程
        return process_eq

    def get_numbered_constants(self, num=1, start=1, prefix='C') -> list[Symbol]:
        """
        Returns a list of constants that do not occur
        in eq already.
        """
        # 返回一个列表，列表中的常数不存在于方程 eq 中
        ncs = self.iter_numbered_constants(start, prefix)
        Cs = [next(ncs) for i in range(num)]
        # 获取 num 个编号常数，并存储在 Cs 列表中
        return Cs

    def iter_numbered_constants(self, start=1, prefix='C') -> Iterator[Symbol]:
        """
        Returns an iterator of constants that do not occur
        in eq already.
        """
        # 返回一个迭代器，迭代器中的常数不存在于方程 eq 中
        atom_set = self.eq.free_symbols
        # 获取方程中的自由符号集合
        func_set = self.eq.atoms(Function)
        # 获取方程中的函数集合
        if func_set:
            atom_set |= {Symbol(str(f.func)) for f in func_set}
            # 如果存在函数，则将函数名作为符号加入到 atom_set 中
        return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)
        # 返回一个编号常数的迭代器，排除了 atom_set 中的符号

    @cached_property
    def is_autonomous(self):
        # 返回方程是否为自治方程
        u = Dummy('u')
        x = self.sym
        syms = self.eq.subs(self.func, u).free_symbols
        # 将函数 func 替换为虚拟变量 u，获取方程中的自由符号
        return x not in syms
    def get_linear_coefficients(self, eq, func, order):
        r"""
        Matches a differential equation to the linear form:

        .. math:: a_n(x) y^{(n)} + \cdots + a_1(x)y' + a_0(x) y + B(x) = 0

        Returns a dict of order:coeff terms, where order is the order of the
        derivative on each term, and coeff is the coefficient of that derivative.
        The key ``-1`` holds the function `B(x)`. Returns ``None`` if the ODE is
        not linear.  This function assumes that ``func`` has already been checked
        to be good.

        Examples
        ========

        >>> from sympy import Function, cos, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import SingleODEProblem
        >>> f = Function('f')
        >>> eq = f(x).diff(x, 3) + 2*f(x).diff(x) + \
        ... x*f(x).diff(x, 2) + cos(x)*f(x).diff(x) + x - f(x) - \
        ... sin(x)
        >>> obj = SingleODEProblem(eq, f(x), x)
        >>> obj.get_linear_coefficients(eq, f(x), 3)
        {-1: x - sin(x), 0: -1, 1: cos(x) + 2, 2: x, 3: 1}
        >>> eq = f(x).diff(x, 3) + 2*f(x).diff(x) + \
        ... x*f(x).diff(x, 2) + cos(x)*f(x).diff(x) + x - f(x) - \
        ... sin(f(x))
        >>> obj = SingleODEProblem(eq, f(x), x)
        >>> obj.get_linear_coefficients(eq, f(x), 3) == None
        True

        """
        # 获取函数 `func`，并设置 `x` 为其唯一参数
        f = func.func
        x = func.args[0]
        # 创建一个包含函数 `f(x)` 及其各阶导数的集合 `symset`
        symset = {Derivative(f(x), x, i) for i in range(order+1)}
        try:
            # 将微分方程 `eq` 转化为字典形式 `rhs, lhs_terms`
            rhs, lhs_terms = _lin_eq2dict(eq, symset)
        except PolyNonlinearError:
            # 如果方程不是线性的，返回 `None`
            return None

        # 如果右侧或左侧任何一项包含 `func`，则返回 `None`
        if rhs.has(func) or any(c.has(func) for c in lhs_terms.values()):
            return None
        # 构建一个字典 `terms`，包含方程中各阶导数的系数
        terms = {i: lhs_terms.get(f(x).diff(x, i), S.Zero) for i in range(order+1)}
        # 将函数 `B(x)` 存储在键 `-1` 下
        terms[-1] = rhs
        return terms

    # TODO: Add methods that can be used by many ODE solvers:
    # order
    # is_linear()
    # get_linear_coefficients()
    # eq_prepared (the ODE in prepared form)
# 单一常微分方程求解器的基类
class SingleODESolver:
    """
    Base class for Single ODE solvers.

    Subclasses should implement the _matches and _get_general_solution
    methods. This class is not intended to be instantiated directly but its
    subclasses are as part of dsolve.

    Examples
    ========

    You can use a subclass of SingleODEProblem to solve a particular type of
    ODE. We first define a particular ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now we solve this problem using the NthAlgebraic solver which is a
    subclass of SingleODESolver:

    >>> from sympy.solvers.ode.single import NthAlgebraic, SingleODEProblem
    >>> problem = SingleODEProblem(eq, f(x), x)
    >>> solver = NthAlgebraic(problem)
    >>> solver.get_general_solution()
    [Eq(f(x), _C*x + _C)]

    The normal way to solve an ODE is to use dsolve (which would use
    NthAlgebraic and other solvers internally). When using dsolve a number of
    other things are done such as evaluating integrals, simplifying the
    solution and renumbering the constants:

    >>> from sympy import dsolve
    >>> dsolve(eq, hint='nth_algebraic')
    Eq(f(x), C1 + C2*x)
    """

    # Subclasses should store the hint name (the argument to dsolve) in this
    # attribute
    hint: ClassVar[str]

    # Subclasses should define this to indicate if they support an _Integral
    # hint.
    has_integral: ClassVar[bool]

    # The ODE to be solved
    ode_problem = None  # type: SingleODEProblem

    # Cache whether or not the equation has matched the method
    _matched: bool | None = None

    # Subclasses should store in this attribute the list of order(s) of ODE
    # that subclass can solve or leave it to None if not specific to any order
    order: list | None = None

    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def matches(self) -> bool:
        """
        Check if the solver matches the provided ODE problem's order.

        Returns:
            bool: True if the solver matches the problem, False otherwise.
        """
        if self.order is not None and self.ode_problem.order not in self.order:
            self._matched = False
            return self._matched

        if self._matched is None:
            self._matched = self._matches()
        return self._matched

    def get_general_solution(self, *, simplify: bool = True) -> list[Equality]:
        """
        Get the general solution of the ODE problem.

        Args:
            simplify (bool): Whether to simplify the solution (default is True).

        Returns:
            list[Equality]: List of general solutions as Equalities.

        Raises:
            ODEMatchError: If the solver cannot solve the given ODE problem.
        """
        if not self.matches():
            msg = "%s solver cannot solve:\n%s"
            raise ODEMatchError(msg % (self.hint, self.ode_problem.eq))
        return self._get_general_solution(simplify_flag=simplify)

    def _matches(self) -> bool:
        """
        Internal method to be implemented by subclasses to check if the solver
        matches the ODE problem.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        msg = "Subclasses of SingleODESolver should implement matches."
        raise NotImplementedError(msg)

    def _get_general_solution(self, *, simplify_flag: bool = True) -> list[Equality]:
        """
        Internal method to be implemented by subclasses to get the general
        solution of the ODE problem.

        Args:
            simplify_flag (bool): Whether to simplify the solution.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        msg = "Subclasses of SingleODESolver should implement get_general_solution."
        raise NotImplementedError(msg)


class SinglePatternODESolver(SingleODESolver):
    '''Superclass for ODE solvers based on pattern matching'''
    `
        # 返回一个 wilds 的结果，使用给定的 ODE 问题的函数、符号和阶数
        def wilds(self):
            prob = self.ode_problem  # 获取 ODE 问题对象
            f = prob.func.func  # 从问题对象获取函数 f
            x = prob.sym  # 从问题对象获取符号 x
            order = prob.order  # 从问题对象获取阶数
            return self._wilds(f, x, order)  # 调用 _wilds 方法，传入 f、x 和 order
    
        # 返回与 wilds 相关的匹配结果，使用 _wilds_match 方法的结果
        def wilds_match(self):
            match = self._wilds_match  # 获取 _wilds_match 属性
            return [match.get(w, S.Zero) for w in self.wilds()]  # 遍历 wilds 的结果，获取匹配结果列表
    
        # 检查 ODE 问题的匹配情况
        def _matches(self):
            eq = self.ode_problem.eq_expanded  # 获取 ODE 问题的展开方程
            f = self.ode_problem.func.func  # 从问题对象获取函数 f
            x = self.ode_problem.sym  # 从问题对象获取符号 x
            order = self.ode_problem.order  # 从问题对象获取阶数
            df = f(x).diff(x, order)  # 计算函数 f 对 x 的 order 阶导数
    
            # 检查阶数是否为 1 或 2
            if order not in [1, 2]:
                return False  # 如果阶数不是 1 或 2，返回 False
    
            pattern = self._equation(f(x), x, order)  # 获取匹配模式，调用 _equation 方法
    
            # 检查模式中导数项的系数是否包含 Wild
            if not pattern.coeff(df).has(Wild):
                eq = expand(eq / eq.coeff(df))  # 如果不包含 Wild，展开方程
    
            eq = eq.collect([f(x).diff(x), f(x)], func=cancel)  # 将方程按 f 的导数和函数本身聚合，进行简化
    
            self._wilds_match = match = eq.match(pattern)  # 匹配方程和模式
            if match is not None:
                return self._verify(f(x))  # 如果匹配成功，调用 _verify 方法验证
            return False  # 如果匹配失败，返回 False
    
        # 验证给定的函数 f(x)
        def _verify(self, fx) -> bool:
            return True  # 简单返回 True，表示验证通过
    
        # 定义 wilds 的抽象方法，子类必须实现
        def _wilds(self, f, x, order):
            msg = "Subclasses of SingleODESolver should implement _wilds"  # 提示信息
            raise NotImplementedError(msg)  # 抛出 NotImplementedError 异常
    
        # 定义方程的抽象方法，子类必须实现
        def _equation(self, fx, x, order):
            msg = "Subclasses of SingleODESolver should implement _equation"  # 提示信息
            raise NotImplementedError(msg)  # 抛出 NotImplementedError 异常
class NthAlgebraic(SingleODESolver):
    r"""
    Solves an `n`\th order ordinary differential equation using algebra and
    integrals.

    There is no general form for the kind of equation that this can solve. The
    the equation is solved algebraically treating differentiation as an
    invertible algebraic function.

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = Eq(f(x) * (f(x).diff(x)**2 - 1), 0)
    >>> dsolve(eq, f(x), hint='nth_algebraic')
    [Eq(f(x), 0), Eq(f(x), C1 - x), Eq(f(x), C1 + x)]

    Note that this solver can return algebraic solutions that do not have any
    integration constants (f(x) = 0 in the above example).
    """

    hint = 'nth_algebraic'  # 指示符，表明该类用于求解某种特定的高阶代数微分方程
    has_integral = True  # 标记该类支持积分，针对 nth_algebraic_Integral 提示

    def _matches(self):
        r"""
        Matches any differential equation that nth_algebraic can solve. Uses
        `sympy.solve` but teaches it how to integrate derivatives.

        This involves calling `sympy.solve` and does most of the work of finding a
        solution (apart from evaluating the integrals).
        """
        eq = self.ode_problem.eq  # 获取微分方程对象
        func = self.ode_problem.func  # 获取方程中的函数
        var = self.ode_problem.sym  # 获取方程中的自变量

        # Derivative that solve can handle:
        diffx = self._get_diffx(var)  # 获取可以处理的导数函数

        # Replace derivatives wrt the independent variable with diffx
        def replace(eq, var):
            def expand_diffx(*args):
                differand, diffs = args[0], args[1:]
                toreplace = differand
                for v, n in diffs:
                    for _ in range(n):
                        if v == var:
                            toreplace = diffx(toreplace)
                        else:
                            toreplace = Derivative(toreplace, v)
                return toreplace
            return eq.replace(Derivative, expand_diffx)

        # Restore derivatives in solution afterwards
        def unreplace(eq, var):
            return eq.replace(diffx, lambda e: Derivative(e, var))

        subs_eqn = replace(eq, var)  # 替换微分方程中的导数
        try:
            # turn off simplification to protect Integrals that have
            # _t instead of fx in them and would otherwise factor
            # as t_*Integral(1, x)
            solns = solve(subs_eqn, func, simplify=False)  # 解微分方程，不进行简化以保护积分表达式
        except NotImplementedError:
            solns = []

        solns = [simplify(unreplace(soln, var)) for soln in solns]  # 简化并恢复替换后的导数
        solns = [Equality(func, soln) for soln in solns]  # 将解转换为等式形式

        self.solutions = solns  # 将解存储在类的属性中
        return len(solns) != 0

    def _get_general_solution(self, *, simplify_flag: bool = True):
        return self.solutions

    # This needs to produce an invertible function but the inverse depends
    # which variable we are integrating with respect to. Since the class can
    # be stored in cached results we need to ensure that we always get the
    # correct form. 
    # 该方法需要生成可逆函数，但逆函数的形式取决于我们积分的自变量。
    # 由于该类可以被缓存，我们需要确保始终得到正确的形式。
    # 每个特定的积分变量返回相同的类，因此我们将这些类存储在全局字典中：
    _diffx_stored: dict[Symbol, type[Function]] = {}

    @staticmethod
    # 获取关于变量 var 的差分类
    def _get_diffx(var):
        # 从 _diffx_stored 字典中获取与 var 对应的差分类
        diffcls = NthAlgebraic._diffx_stored.get(var, None)

        if diffcls is None:
            # 如果没有找到对应的差分类，则定义一个新的类 diffx，
            # 这个类行为类似于对变量 var 的导数，但是是“可逆的”。
            class diffx(Function):
                # 定义 diffx 类的反函数
                def inverse(self):
                    # 在这里不使用 integrate，因为 fx 已经被 _t 替换了
                    # 在等式中；在求解过程中积分可能不正确。
                    return lambda expr: Integral(expr, var) + Dummy('C')

            # 将新定义的 diffx 类存储到 _diffx_stored 字典中与 var 对应的位置
            diffcls = NthAlgebraic._diffx_stored.setdefault(var, diffx)

        # 返回与 var 对应的差分类
        return diffcls
class FirstExact(SinglePatternODESolver):
    r"""
    Solves 1st order exact ordinary differential equations.

    A 1st order differential equation is called exact if it is the total
    differential of a function. That is, the differential equation

    .. math:: P(x, y) \,\partial{}x + Q(x, y) \,\partial{}y = 0

    is exact if there is some function `F(x, y)` such that `P(x, y) =
    \partial{}F/\partial{}x` and `Q(x, y) = \partial{}F/\partial{}y`.  It can
    be shown that a necessary and sufficient condition for a first order ODE
    to be exact is that `\partial{}P/\partial{}y = \partial{}Q/\partial{}x`.
    Then, the solution will be as given below::

        >>> from sympy import Function, Eq, Integral, symbols, pprint
        >>> x, y, t, x0, y0, C1= symbols('x,y,t,x0,y0,C1')
        >>> P, Q, F= map(Function, ['P', 'Q', 'F'])
        >>> pprint(Eq(Eq(F(x, y), Integral(P(t, y), (t, x0, x)) +
        ... Integral(Q(x0, t), (t, y0, y))), C1))
                    x                y
                    /                /
                   |                |
        F(x, y) =  |  P(t, y) dt +  |  Q(x0, t) dt = C1
                   |                |
                  /                /
                  x0               y0

    Where the first partials of `P` and `Q` exist and are continuous in a
    simply connected region.

    A note: SymPy currently has no way to represent inert substitution on an
    expression, so the hint ``1st_exact_Integral`` will return an integral
    with `dy`.  This is supposed to represent the function that you are
    solving for.

    Examples
    ========

    >>> from sympy import Function, dsolve, cos, sin
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x),
    ... f(x), hint='1st_exact')
    Eq(x*cos(f(x)) + f(x)**3/3, C1)

    References
    ==========

    - https://en.wikipedia.org/wiki/Exact_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 73

    # indirect doctest

    """
    
    # 设置求解器的提示信息为 "1st_exact"
    hint = "1st_exact"
    # 表明这个求解器可以处理含有积分的情况
    has_integral = True
    # 表明这个求解器适用于一阶方程
    order = [1]

    # 定义一个方法用于提取表达式中的通配符
    def _wilds(self, f, x, order):
        # 创建排除 f(x) 对 x 的导数之外的 Wild 占位符 P 和 Q
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return P, Q

    # 定义一个方法用于生成方程的表达式
    def _equation(self, fx, x, order):
        # 获取通配符 P 和 Q
        P, Q = self.wilds()
        # 返回方程表达式 P + Q*fx.diff(x)
        return P + Q*fx.diff(x)
    def _verify(self, fx) -> bool:
        # 获取自变量和因变量函数
        P, Q = self.wilds()
        # 获取常微分方程的自变量符号
        x = self.ode_problem.sym
        # 创建一个虚拟变量y作为替代
        y = Dummy('y')

        # 获取自变量和因变量的匹配项
        m, n = self.wilds_match()

        # 将自变量函数fx替换为虚拟变量y
        m = m.subs(fx, y)
        n = n.subs(fx, y)

        # 计算分子的差分并化简
        numerator = cancel(m.diff(y) - n.diff(x))

        # 检查分子是否为零，即是否为恰当方程
        if numerator.is_zero:
            # 是恰当方程
            return True
        else:
            # 下面的条件尝试将非恰当方程转换为恰当方程
            # 参考文献：
            # 1. Differential equations with applications and historical notes - George E. Simmons
            # 2. https://math.okstate.edu/people/binegar/2233-S99/2233-l12.pdf

            # 计算两个因子以便尝试转换方程
            factor_n = cancel(numerator / n)
            factor_m = cancel(-numerator / m)

            # 判断哪个因子不包含虚拟变量y或x，以决定使用哪个因子和积分变量
            if y not in factor_n.free_symbols:
                # 如果 (dP/dy - dQ/dx) / Q = f(x)
                # 则 exp(integral(f(x))*equation 可以使方程恰当
                factor = factor_n
                integration_variable = x
            elif x not in factor_m.free_symbols:
                # 如果 (dP/dy - dQ/dx) / -P = f(y)
                # 则 exp(integral(f(y))*equation 可以使方程恰当
                factor = factor_m
                integration_variable = y
            else:
                # 无法转换为恰当方程
                return False

            # 计算积分因子
            factor = exp(Integral(factor, integration_variable))
            m *= factor
            n *= factor

            # 更新匹配项的值
            self._wilds_match[P] = m.subs(y, fx)
            self._wilds_match[Q] = n.subs(y, fx)
            return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取自变量和因变量的匹配项
        m, n = self.wilds_match()
        # 获取自变量函数
        fx = self.ode_problem.func
        # 获取自变量符号
        x = self.ode_problem.sym
        # 获取常数C1
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        # 创建一个虚拟变量y作为替代
        y = Dummy('y')

        # 将自变量函数fx替换为虚拟变量y
        m = m.subs(fx, y)
        n = n.subs(fx, y)

        # 构建通解方程
        gen_sol = Eq(Subs(Integral(m, x)
                          + Integral(n - Integral(m, x).diff(y), y), y, fx), C1)
        return [gen_sol]
class FirstLinear(SinglePatternODESolver):
    r"""
    Solves 1st order linear differential equations.

    These are differential equations of the form

    .. math:: dy/dx + P(x) y = Q(x)\text{.}

    These kinds of differential equations can be solved in a general way.  The
    integrating factor `e^{\int P(x) \,dx}` will turn the equation into a
    separable equation.  The general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint, diff, sin
        >>> from sympy.abc import x
        >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
        >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x))
        >>> pprint(genform)
                    d
        P(x)*f(x) + --(f(x)) = Q(x)
                    dx
        >>> pprint(dsolve(genform, f(x), hint='1st_linear_Integral'))
                /       /                   \
                |      |                    |
                |      |         /          |     /
                |      |        |           |    |
                |      |        | P(x) dx   |  - | P(x) dx
                |      |        |           |    |
                |      |       /            |   /
        f(x) = |C1 +  | Q(x)*e           dx|*e
                |      |                    |
                \     /                     /


    Examples
    ========

    >>> f = Function('f')
    >>> pprint(dsolve(Eq(x*diff(f(x), x) - f(x), x**2*sin(x)),
    ... f(x), '1st_linear'))
    f(x) = x*(C1 - cos(x))

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation#First-order_equation_with_variable_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 92

    # indirect doctest

    """
    # 设置求解提示为 '1st_linear'
    hint = '1st_linear'
    # 声明是否有积分因子
    has_integral = True
    # 设置方程阶数为一阶
    order = [1]

    # 定义提取通用解中的变量函数
    def _wilds(self, f, x, order):
        # 使用 Wild 类型 P 和 Q 来匹配方程中的未知函数
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x), f(x).diff(x)])
        return P, Q

    # 定义差分方程
    def _equation(self, fx, x, order):
        # 提取通用解中的 P 和 Q
        P, Q = self.wilds()
        # 返回原始方程的形式
        return fx.diff(x) + P*fx - Q

    # 获取通用解表达式
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 提取通用解中的 P 和 Q
        P, Q = self.wilds_match()
        # 提取方程中的函数和自变量
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        # 获取方程中的常数项 C1
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        # 构建通用解表达式
        gensol = Eq(fx, ((C1 + Integral(Q*exp(Integral(P, x)), x))
            * exp(-Integral(P, x))))
        return [gensol]


class AlmostLinear(SinglePatternODESolver):
    r"""
    Solves an almost-linear differential equation.

    The general form of an almost linear differential equation is

    .. math:: a(x) g'(f(x)) f'(x) + b(x) g(f(x)) + c(x)

    Here `f(x)` is the function to be solved for (the dependent variable).
    The substitution `g(f(x)) = u(x)` leads to a linear differential equation
    for `u(x)` of the form `a(x) u' + b(x) u + c(x) = 0`. This can be solved
    for `u(x)` by the `first_linear` hint and then `f(x)` is found by solving
    `g(f(x)) = u(x)`.

    See Also
    ```
``````
class FirstLinear(SinglePatternODESolver):
    r"""
    Solves 1st order linear differential equations.

    These are differential equations of the form

    .. math:: dy/dx + P(x) y = Q(x)\text{.}

    These kinds of differential equations can be solved in a general way.  The
    integrating factor `e^{\int P(x) \,dx}` will turn the equation into a
    separable equation.  The general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint, diff, sin
        >>> from sympy.abc import x
        >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
        >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x))
        >>> pprint(genform)
                    d
        P(x)*f(x) + --(f(x)) = Q(x)
                    dx
        >>> pprint(dsolve(genform, f(x), hint='1st_linear_Integral'))
                /       /                   \
                |      |                    |
                |      |         /          |     /
                |      |        |           |    |
                |      |        | P(x) dx   |  - | P(x) dx
                |      |        |           |    |
                |      |       /            |   /
        f(x) = |C1 +  | Q(x)*e           dx|*e
                |      |                    |
                \     /                     /


    Examples
    ========

    >>> f = Function('f')
    >>> pprint(dsolve(Eq(x*diff(f(x), x) - f(x), x**2*sin(x)),
    ... f(x), '1st_linear'))
    f(x) = x*(C1 - cos(x))

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation#First-order_equation_with_variable_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 92

    # indirect doctest

    """
    # 设置求解提示为 '1st_linear'
    hint = '1st_linear'
    # 声明是否有积分因子
    has_integral = True
    # 设置方程阶数为一阶
    order = [1]

    # 定义提取通用解中的变量函数
    def _wilds(self, f, x, order):
        # 使用 Wild 类型 P 和 Q 来匹配方程中的未知函数
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x), f(x).diff(x)])
        return P, Q

    # 定义差分方程
    def _equation(self, fx, x, order):
        # 提取通用解中的 P 和 Q
        P, Q = self.wilds()
        # 返回原始方程的形式
        return fx.diff(x) + P*fx - Q

    # 获取通用解表达式
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 提取通用解中的 P 和 Q
        P, Q = self.wilds_match()
        # 提取方程中的函数和自变量
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        # 获取方程中的常数项 C1
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        # 构建通用解表达式
        gensol = Eq(fx, ((C1 + Integral(Q*exp(Integral(P, x)), x))
            * exp(-Integral(P, x))))
        return [gensol]


class AlmostLinear(SinglePatternODESolver):
    r"""
    Solves an almost-linear differential equation.

    The general form of an almost linear differential equation is

    .. math:: a(x) g'(f(x)) f'(x) + b(x) g(f(x)) + c(x)

    Here `f(x)` is the function to be solved for (the dependent variable).
    The substitution `g(f(x)) = u(x)` leads to a linear differential equation
    for `u(x)` of the form `a(x) u' + b(x) u + c(x) = 0`. This can be solved
    for `u(x)` by the `first_linear` hint and then `f(x)` is found by solving
    `g(f(x)) = u(x)`.

    See Also
    ```
    # 设置提示信息为 "almost_linear"
    hint = "almost_linear"
    # 设置具有积分的标志为 True
    has_integral = True
    # 设置微分方程的阶数为 [1]
    order = [1]
    
    # 定义函数 _wilds，用于识别方程中的未知函数及其导数
    def _wilds(self, f, x, order):
        # 创建两个野字符号 P 和 Q，但排除了 f(x) 的导数
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return P, Q
    
    # 定义函数 _equation，生成几乎线性微分方程的标准形式
    def _equation(self, fx, x, order):
        # 获取通配符 P 和 Q
        P, Q = self.wilds()
        return P*fx.diff(x) + Q
    
    # 定义函数 _verify，验证是否为几乎线性微分方程
    def _verify(self, fx):
        # 匹配得到 a 和 b
        a, b = self.wilds_match()
        # 如果 b 对 fx 的导数不为零，并且 b 对 a 的导数未包含 fx，则满足条件
        if b.diff(fx) != 0 and not simplify(b.diff(fx)/a).has(fx):
            # 分离出不包含 fx 的项，得到 c
            c, b = b.as_independent(fx) if b.is_Add else (S.Zero, b)
            # self.ly 为包含 fx 的项，即 l(y)
            self.ly = factor_terms(b).as_independent(fx, as_Add=False)[1]
            # 计算 ax 和 cx
            self.ax = a / self.ly.diff(fx)
            self.cx = -c  # cx 取为 -c(x)，以简化解的积分表达式
            # 计算 bx
            self.bx = factor_terms(b) / self.ly
            return True
    
        return False
    
    # 定义函数 _get_general_solution，生成几乎线性微分方程的一般解
    def _get_general_solution(self, *, simplify_flag: bool = True):
        x = self.ode_problem.sym
        # 获取常数 C1
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        # 构建一般解的表达式
        gensol = Eq(self.ly, ((C1 + Integral((self.cx/self.ax)*exp(Integral(self.bx/self.ax, x)), x))
                * exp(-Integral(self.bx/self.ax, x))))
    
        return [gensol]
# 定义 Bernoulli 类，继承自 SinglePatternODESolver 类
class Bernoulli(SinglePatternODESolver):
    # 解决 Bernoulli 微分方程
    # 这些方程形式如下
    #
    # dy/dx + P(x) y = Q(x) y^n, n ≠ 1
    #
    # 替换 w = 1/y^{1-n} 将这种形式的方程转换为线性方程（参见 sympy.solvers.ode.single.FirstLinear 的文档）
    # 一般解为:
    #
    # >>> from sympy import Function, dsolve, Eq, pprint
    # >>> from sympy.abc import x, n
    # >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
    # >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x)*f(x)**n)
    # >>> pprint(genform)
    #             d                n
    # P(x)*f(x) + --(f(x)) = Q(x)*f (x)
    #             dx
    # >>> pprint(dsolve(genform, f(x), hint='Bernoulli_Integral'), num_columns=110)
    #                                                                                      -1
    #                                                                                     -----
    #                                                                                     n - 1
    # //         /                                 /                            \                    \
    # ||        |                                 |                             |                    |
    # ||        |                  /              |                  /          |            /       |
    # ||        |                 |               |                 |           |           |        |
    # ||        |       -(n - 1)* | P(x) dx       |       -(n - 1)* | P(x) dx   |  (n - 1)* | P(x) dx|
    # ||        |                 |               |                 |           |           |        |
    # ||        |                /                |                /            |          /         |
    # f(x) = ||C1 - n* | Q(x)*e                    dx +  | Q(x)*e                    dx|*e                  |
    # ||        |                                 |                             |                    |
    # \\       /                                 /                              /                    /

    # 注意当 n = 1 时，方程是可分离的（参见 sympy.solvers.ode.single.Separable 的文档）
    #
    # >>> pprint(dsolve(Eq(f(x).diff(x) + P(x)*f(x), Q(x)*f(x)), f(x),
    # ... hint='separable_Integral'))
    # f(x)
    #     /
    # |                /
    # |  1            |
    # |  - dy = C1 +  | (-P(x) + Q(x)) dx
    # |  y            |
    # |              /
    # /

    # 示例
    # ========
    #
    # >>> from sympy import Function, dsolve, Eq, pprint, log
    # >>> from sympy.abc import x
    # >>> f = Function('f')
    #
    # >>> pprint(dsolve(Eq(x*f(x).diff(x) + f(x), log(x)*f(x)**2),
    # ... f(x), hint='Bernoulli'))
    #                 1
    f(x) =  -----------------
            C1*x + log(x) + 1

    References
    ==========

    - https://en.wikipedia.org/wiki/Bernoulli_differential_equation

    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 95

    # indirect doctest

    """
    hint = "Bernoulli"
    has_integral = True
    order = [1]

    定义一个类，处理伯努利微分方程的求解问题
    def _wilds(self, f, x, order):
        定义匹配模式，排除掉 f(x) 的部分
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x)])
        n = Wild('n', exclude=[x, f(x), f(x).diff(x)])
        返回匹配模式 P, Q, n

    定义求解伯努利微分方程的方法
    def _equation(self, fx, x, order):
        使用先前定义的匹配模式，构造微分方程
        P, Q, n = self.wilds()
        return fx.diff(x) + P*fx - Q*fx**n

    获取伯努利微分方程的通解
    def _get_general_solution(self, *, simplify_flag: bool = True):
        获取匹配后的参数 P, Q, n
        P, Q, n = self.wilds_match()
        获取微分方程的函数和符号变量
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        获取常数 C1
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        根据 n 的值选择不同的通解形式
        if n==1:
            对于 n=1 的情况，构造通解方程
            gensol = Eq(log(fx), (
            C1 + Integral((-P + Q), x)
        ))
        else:
            对于 n 不等于 1 的情况，构造通解方程
            gensol = Eq(fx**(1-n), (
                (C1 - (n - 1) * Integral(Q*exp(-n*Integral(P, x))
                            * exp(Integral(P, x)), x)
                ) * exp(-(1 - n)*Integral(P, x)))
            )
        返回通解方程列表
        return [gensol]
class Factorable(SingleODESolver):
    r"""
    Solves equations having a solvable factor.

    This function is used to solve the equation having factors. Factors may be of type algebraic or ode. It
    will try to solve each factor independently. Factors will be solved by calling dsolve. We will return the
    list of solutions.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = (f(x)**2-4)*(f(x).diff(x)+f(x))
    >>> pprint(dsolve(eq, f(x)))
                                    -x
    [f(x) = 2, f(x) = -2, f(x) = C1*e  ]


    """
    hint = "factorable"  # 提示该求解器适用于可分解因子的方程
    has_integral = False  # 不支持积分

    def _matches(self):
        eq_orig = self.ode_problem.eq  # 获取原始的方程
        f = self.ode_problem.func.func  # 获取方程中的函数
        x = self.ode_problem.sym  # 获取方程中的符号变量
        df = f(x).diff(x)  # 对函数求导
        self.eqs = []
        eq = eq_orig.collect(f(x), func = cancel)  # 将方程按照函数 f(x) 进行整理，并取消冗余项
        eq = fraction(factor(eq))[0]  # 对整理后的方程进行因式分解，并取出分母
        factors = Mul.make_args(factor(eq))  # 将因式分解的结果转化为因子列表
        roots = [fac.as_base_exp() for fac in factors if len(fac.args)!=0]  # 找出所有非空因子的底数和指数
        if len(roots)>1 or roots[0][1]>1:  # 如果有多个因子或第一个因子的指数大于1
            for base, expo in roots:
                if base.has(f(x)):  # 如果因子中包含函数 f(x)
                    self.eqs.append(base)  # 将该因子加入到方程列表中
            if len(self.eqs)>0:  # 如果方程列表不为空
                return True  # 返回匹配成功
        roots = solve(eq, df)  # 解方程关于 df 的根
        if len(roots)>0:  # 如果有根
            self.eqs = [(df - root) for root in roots]  # 计算得到方程的形式
            # 避免无限递归
            matches = self.eqs != [eq_orig]
            return matches  # 返回匹配成功
        for i in factors:  # 对于每一个因子
            if i.has(f(x)):  # 如果因子中包含函数 f(x)
                self.eqs.append(i)  # 将因子加入到方程列表中
        return len(self.eqs)>0 and len(factors)>1  # 返回是否存在多个因子且方程列表不为空

    def _get_general_solution(self, *, simplify_flag: bool = True):
        func = self.ode_problem.func.func  # 获取函数
        x = self.ode_problem.sym  # 获取符号变量
        eqns = self.eqs  # 获取方程列表
        sols = []
        for eq in eqns:  # 对于每一个方程
            try:
                sol = dsolve(eq, func(x))  # 尝试求解方程
            except NotImplementedError:
                continue
            else:
                if isinstance(sol, list):  # 如果求解结果是列表
                    sols.extend(sol)  # 将解加入到解列表中
                else:
                    sols.append(sol)  # 将解加入到解列表中

        if sols == []:  # 如果解列表为空
            raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                + " the factorable group method")  # 抛出未实现错误，说明方程无法用可分解因子组方法求解
        return sols  # 返回解列表
    # 导入必要的函数和类库：dsolve 用于求解常微分方程，Function 定义一个未知函数 f(x)
    >>> from sympy import dsolve, checkodesol, pprint, Function
    # 定义一个未知函数 f(x)
    >>> f = Function('f')
    # 将未知函数 f(x) 赋值给变量 y
    >>> y = f(x)
    # 定义一个特定形式的常微分方程，genform 表示 a*y.diff(x) - (b*y**2 + c*y/x + d/x**2)
    >>> genform = a*y.diff(x) - (b*y**2 + c*y/x + d/x**2)
    # 求解常微分方程 genform，使用 Riccati 特殊解法 hint="Riccati_special_minus2"
    >>> sol = dsolve(genform, y, hint="Riccati_special_minus2")
    # 打印求解结果 sol，不自动换行显示
    >>> pprint(sol, wrap_line=False)
            /                                 /        __________________       \\
            |           __________________    |       /                2        ||
            |          /                2     |     \/  4*b*d - (a + c)  *log(x)||
           -|a + c - \/  4*b*d - (a + c)  *tan|C1 + ----------------------------||
            \                                 \                 2*a             //
    f(x) = ------------------------------------------------------------------------
                                            2*b*x

    # 检查求解结果 sol 是否是常微分方程 genform 的解，返回 True 表示是解
    >>> checkodesol(genform, sol, order=1)[0]
    True

    References
    ==========

    - https://www.maplesoft.com/support/help/Maple/view.aspx?path=odeadvisor/Riccati
    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0106.pdf -
      https://eqworld.ipmnet.ru/en/solutions/ode/ode0123.pdf
    """
    # 定义变量 hint 为字符串 "Riccati_special_minus2"
    hint = "Riccati_special_minus2"
    # 定义变量 has_integral 为 False，表示未找到积分
    has_integral = False
    # 定义变量 order 为列表 [1]，表示方程的阶数为 1

    # 定义函数 _wilds，接受函数对象 self，未知函数 f，自变量 x 和方程阶数 order
    def _wilds(self, f, x, order):
        # 定义通配符对象，限制为非 x、f(x)、f(x).diff(x)、0 的符号
        a = Wild('a', exclude=[x, f(x), f(x).diff(x), 0])
        b = Wild('b', exclude=[x, f(x), f(x).diff(x), 0])
        c = Wild('c', exclude=[x, f(x), f(x).diff(x)])
        d = Wild('d', exclude=[x, f(x), f(x).diff(x)])
        return a, b, c, d

    # 定义函数 _equation，接受函数对象 self，未知函数 fx，自变量 x 和方程阶数 order
    def _equation(self, fx, x, order):
        # 从通配符对象 self.wilds() 中解包得到 a、b、c、d
        a, b, c, d = self.wilds()
        # 返回特定形式的常微分方程 a*fx.diff(x) + b*fx**2 + c*fx/x + d/x**2
        return a*fx.diff(x) + b*fx**2 + c*fx/x + d/x**2

    # 定义函数 _get_general_solution，接受函数对象 self 和一个布尔型参数 simplify_flag
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 从通配符对象 self.wilds_match() 中解包得到 a、b、c、d
        a, b, c, d = self.wilds_match()
        # 获取方程对象 self.ode_problem 的未知函数 fx 和自变量 x
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        # 获取方程对象 self.ode_problem 中的第一个编号常数 C1
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        # 计算 mu = sqrt(4*d*b - (a - c)**2)
        mu = sqrt(4*d*b - (a - c)**2)

        # 生成一般解 gensol，表示为方程 fx = (a - c - mu*tan(mu/(2*a)*log(x) + C1))/(2*b*x)
        gensol = Eq(fx, (a - c - mu*tan(mu/(2*a)*log(x) + C1))/(2*b*x))
        # 返回包含 gensol 的列表
        return [gensol]
class RationalRiccati(SinglePatternODESolver):
    r"""
    给出具有至少一个有理特解的一阶Riccati微分方程的一般解。

    .. math :: y' = b_0(x) + b_1(x) y + b_2(x) y^2

    其中 `b_0`, `b_1` 和 `b_2` 是 `x` 的有理函数，且 `b_2 \ne 0`
    （若 `b_2 = 0` 则为伯努利方程）。

    Examples
    ========

    >>> from sympy import Symbol, Function, dsolve, checkodesol
    >>> f = Function('f')
    >>> x = Symbol('x')

    >>> eq = -x**4*f(x)**2 + x**3*f(x).diff(x) + x**2*f(x) + 20
    >>> sol = dsolve(eq, hint="1st_rational_riccati")
    >>> sol
    Eq(f(x), (4*C1 - 5*x**9 - 4)/(x**2*(C1 + x**9 - 1)))
    >>> checkodesol(eq, sol)
    (True, 0)

    References
    ==========

    - Riccati ODE:  https://en.wikipedia.org/wiki/Riccati_equation
    - N. Thieu Vo - Rational and Algebraic Solutions of First-Order Algebraic ODEs:
      Algorithm 11, pp. 78 - https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf
    """
    has_integral = False
    hint = "1st_rational_riccati"
    order = [1]

    def _wilds(self, f, x, order):
        # 定义用于匹配的通配符，排除 `f(x)` 和 `f(x)` 的导数
        b0 = Wild('b0', exclude=[f(x), f(x).diff(x)])
        b1 = Wild('b1', exclude=[f(x), f(x).diff(x)])
        b2 = Wild('b2', exclude=[f(x), f(x).diff(x)])
        return (b0, b1, b2)

    def _equation(self, fx, x, order):
        # 定义 Riccati 方程
        b0, b1, b2 = self.wilds()
        return fx.diff(x) - b0 - b1*fx - b2*fx**2

    def _matches(self):
        # 检查方程是否为一阶，然后尝试匹配 Riccati 方程的形式
        eq = self.ode_problem.eq_expanded
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order

        if order != 1:
            return False

        match, funcs = match_riccati(eq, f, x)
        if not match:
            return False
        _b0, _b1, _b2 = funcs
        b0, b1, b2 = self.wilds()
        self._wilds_match = match = {b0: _b0, b1: _b1, b2: _b2}
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取一般解
        b0, b1, b2 = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return solve_riccati(fx, x, b0, b1, b2, gensol=True)


class SecondNonlinearAutonomousConserved(SinglePatternODESolver):
    r"""
    给出形式为

    .. math :: f''(x) = g(f(x))

    的自治二阶非线性微分方程的解。

    可以通过两边乘以 `f'(x)` 并积分来计算此微分方程的解，
    将其转换为一阶微分方程。

    Examples
    ========

    >>> from sympy import Function, symbols, dsolve
    >>> f, g = symbols('f g', cls=Function)
    >>> x = symbols('x')

    >>> eq = f(x).diff(x, 2) - g(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 - x)]


    """
    # 导入 sympy 中的 exp 和 log 函数
    >>> from sympy import exp, log
    # 定义微分方程
    >>> eq = f(x).diff(x, 2) - exp(f(x)) + log(f(x))
    # 求解微分方程，禁用简化
    >>> dsolve(eq, simplify=False)
    # 返回两个方程的列表，每个方程是一个积分方程
    [Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 - x)]

    References
    ==========

    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0301.pdf
    """
    # 提示信息，指示这是一个二阶非线性自治守恒型方程
    hint = "2nd_nonlinear_autonomous_conserved"
    # 标志：表示此方程含有积分
    has_integral = True
    # 方程的阶数，这里为二阶
    order = [2]

    # 返回一个元组，包含除了零、一阶导数和二阶导数之外的符号变量
    def _wilds(self, f, x, order):
        fy = Wild('fy', exclude=[0, f(x).diff(x), f(x).diff(x, 2)])
        return (fy, )

    # 定义微分方程的形式
    def _equation(self, fx, x, order):
        fy = self.wilds()[0]
        return fx.diff(x, 2) + fy

    # 验证微分方程是否是自治方程
    def _verify(self, fx):
        return self.ode_problem.is_autonomous

    # 获取通解的函数，如果 simplify_flag 为真则简化结果
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 匹配通解中的模式
        g = self.wilds_match()[0]
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        # 创建虚拟变量 u
        u = Dummy('u')
        # 用 u 替换 fx 中的函数
        g = g.subs(fx, u)
        # 获取两个常数 C1 和 C2
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        # 计算积分内部表达式
        inside = -2*Integral(g, u) + C1
        # 构造积分表达式
        lhs = Integral(1/sqrt(inside), (u, fx))
        # 返回通解方程的列表
        return [Eq(lhs, C2 + x), Eq(lhs, C2 - x)]
class Liouville(SinglePatternODESolver):
    r"""
    Solves 2nd order Liouville differential equations.

    The general form of a Liouville ODE is

    .. math:: \frac{d^2 y}{dx^2} + g(y) \left(\!
                \frac{dy}{dx}\!\right)^2 + h(x)
                \frac{dy}{dx}\text{.}

    The general solution is:

        >>> from sympy import Function, dsolve, Eq, pprint, diff
        >>> from sympy.abc import x
        >>> f, g, h = map(Function, ['f', 'g', 'h'])
        >>> genform = Eq(diff(f(x),x,x) + g(f(x))*diff(f(x),x)**2 +
        ... h(x)*diff(f(x),x), 0)
        >>> pprint(genform)
                          2                    2
                /d       \         d          d
        g(f(x))*|--(f(x))|  + h(x)*--(f(x)) + ---(f(x)) = 0
                \dx      /         dx           2
                                              dx
        >>> pprint(dsolve(genform, f(x), hint='Liouville_Integral'))
                                          f(x)
                  /                     /
                 |                     |
                 |     /               |     /
                 |    |                |    |
                 |  - | h(x) dx        |    | g(y) dy
                 |    |                |    |
                 |   /                 |   /
        C1 + C2* | e            dx +   |  e           dy = 0
                 |                     |
                /                     /

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(diff(f(x), x, x) + diff(f(x), x)**2/f(x) +
    ... diff(f(x), x)/x, f(x), hint='Liouville'))
               ________________           ________________
    [f(x) = -\/ C1 + C2*log(x) , f(x) = \/ C1 + C2*log(x) ]

    References
    ==========

    - Goldstein and Braun, "Advanced Methods for the Solution of Differential
      Equations", pp. 98
    - https://www.maplesoft.com/support/help/Maple/view.aspx?path=odeadvisor/Liouville

    # indirect doctest

    """
    hint = "Liouville"
    has_integral = True
    order = [2]

    def _wilds(self, f, x, order):
        # Define wildcards for matching patterns in the Liouville ODE
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        k = Wild('k', exclude=[f(x).diff(x)])
        return d, e, k

    def _equation(self, fx, x, order):
        # Return the Liouville ODE in the form:
        # d*fx.diff(x, 2) + e*fx.diff(x)**2 + k*fx.diff(x)
        # where d, e, k are patterns matched by wildcards from _wilds()
        d, e, k = self.wilds()
        return d*fx.diff(x, 2) + e*fx.diff(x)**2 + k*fx.diff(x)
    # 验证函数，用于验证给定函数 fx 是否符合特定的微分方程形式
    def _verify(self, fx):
        # 获取微分方程的系数 d, e, k
        d, e, k = self.wilds_match()
        # 创建一个名为 self.y 的虚拟变量
        self.y = Dummy('y')
        # 获取微分方程的自变量 x
        x = self.ode_problem.sym
        # 计算简化后的 g(x, y) = e(x)/d(x)，并将 fx 替换为 self.y
        self.g = simplify(e/d).subs(fx, self.y)
        # 计算简化后的 h(x) = k(x)/d(x)，并将 fx 替换为 self.y
        self.h = simplify(k/d).subs(fx, self.y)
        # 如果 self.y 或者 x 出现在 self.h 或者 self.g 的自由符号中，则返回 False
        if self.y in self.h.free_symbols or x in self.g.free_symbols:
            return False
        # 否则返回 True，表示验证通过
        return True

    # 获取通解函数，返回一个包含通解方程的列表
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取微分方程的系数 d, e, k
        d, e, k = self.wilds_match()
        # 获取微分方程的函数 fx 和自变量 x
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        # 获取微分方程的两个编号常数 C1 和 C2
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        # 计算积分 int = ∫exp(∫g(y)dy)，其中积分上限为 fx
        int = Integral(exp(Integral(self.g, self.y)), (self.y, None, fx))
        # 构造通解方程 gen_sol = int + C1*∫exp(-∫h(x)dx) + C2 = 0
        gen_sol = Eq(int + C1*Integral(exp(-Integral(self.h, x)), x) + C2, 0)

        # 返回包含通解方程的列表
        return [gen_sol]
class Separable(SinglePatternODESolver):
    r"""
    Solves separable 1st order differential equations.

    This is any differential equation that can be written as `P(y)
    \tfrac{dy}{dx} = Q(x)`.  The solution can then just be found by
    rearranging terms and integrating: `\int P(y) \,dy = \int Q(x) \,dx`.
    This hint uses :py:meth:`sympy.simplify.simplify.separatevars` as its back
    end, so if a separable equation is not caught by this solver, it is most
    likely the fault of that function.
    :py:meth:`~sympy.simplify.simplify.separatevars` is
    smart enough to do most expansion and factoring necessary to convert a
    separable equation `F(x, y)` into the proper form `P(x)\cdot{}Q(y)`.  The
    general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint
        >>> from sympy.abc import x
        >>> a, b, c, d, f = map(Function, ['a', 'b', 'c', 'd', 'f'])
        >>> genform = Eq(a(x)*b(f(x))*f(x).diff(x), c(x)*d(f(x)))
        >>> pprint(genform)
                     d
        a(x)*b(f(x))*--(f(x)) = c(x)*d(f(x))
                     dx
        >>> pprint(dsolve(genform, f(x), hint='separable_Integral'))
             f(x)
           /                  /
          |                  |
          |  b(y)            | c(x)
          |  ---- dy = C1 +  | ---- dx
          |  d(y)            | a(x)
          |                  |
         /                  /

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(Eq(f(x)*f(x).diff(x) + x, 3*x*f(x)**2), f(x),
    ... hint='separable', simplify=False))
       /   2       \         2
    log\3*f (x) - 1/        x
    ---------------- = C1 + --
           6                2

    References
    ==========

    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 52

    # indirect doctest

    """

    # 提示符号为“separable”
    hint = "separable"

    # 具有积分特性
    has_integral = True

    # 方程的阶数为1
    order = [1]

    # 提取通配符号函数
    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    # 构建微分方程
    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    # 验证方程是否满足分离变量法的条件
    def _verify(self, fx):
        d, e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        d = separatevars(d.subs(fx, self.y))
        e = separatevars(e.subs(fx, self.y))
        # m1[coeff]*m1[x]*m1[y] + m2[coeff]*m2[x]*m2[y]*y'
        self.m1 = separatevars(d, dict=True, symbols=(x, self.y))
        self.m2 = separatevars(e, dict=True, symbols=(x, self.y))
        return bool(self.m1 and self.m2)

    # 获取匹配对象
    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return self.m1, self.m2, x, fx
    # 定义一个方法 `_get_general_solution`，返回一个通解的列表
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 从 `_get_match_object` 方法中获取匹配对象 m1, m2, x, fx
        m1, m2, x, fx = self._get_match_object()
        
        # 从 ODE 问题中获取一个编号的常数 C1
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        
        # 创建一个积分对象 `int`，积分的被积函数是 m2['coeff']*m2[self.y]/m1[self.y]，积分变量是 self.y，积分区间是 [None, fx]
        int = Integral(m2['coeff']*m2[self.y]/m1[self.y],
                       (self.y, None, fx))
        
        # 创建一个一般解对象 `gen_sol`，等号左侧是上面定义的积分 `int`，右侧是另一个积分 `-m1['coeff']*m1[x]/m2[x]` 加上常数 C1
        gen_sol = Eq(int, Integral(-m1['coeff']*m1[x]/m2[x], x) + C1)
        
        # 返回一个包含这个一般解对象的列表
        return [gen_sol]
class SeparableReduced(Separable):
    r"""
    Solves a differential equation that can be reduced to the separable form.

    The general form of this equation is

    .. math:: y' + (y/x) H(x^n y) = 0\text{}.

    This can be solved by substituting `u(y) = x^n y`.  The equation then
    reduces to the separable form `\frac{u'}{u (\mathrm{power} - H(u))} -
    \frac{1}{x} = 0`.

    The general solution is:

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x, n
        >>> f, g = map(Function, ['f', 'g'])
        >>> genform = f(x).diff(x) + (f(x)/x)*g(x**n*f(x))
        >>> pprint(genform)
                         / n     \
        d          f(x)*g\x *f(x)/
        --(f(x)) + ---------------
        dx                x
        >>> pprint(dsolve(genform, hint='separable_reduced'))
         n
        x *f(x)
          /
         |
         |         1
         |    ------------ dy = C1 + log(x)
         |    y*(n - g(y))
         |
         /

    See Also
    ========
    :obj:`sympy.solvers.ode.single.Separable`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = (x - x**2*f(x))*d - f(x)
    >>> dsolve(eq, hint='separable_reduced')
    [Eq(f(x), (1 - sqrt(C1*x**2 + 1))/x), Eq(f(x), (sqrt(C1*x**2 + 1) + 1)/x)]
    >>> pprint(dsolve(eq, hint='separable_reduced'))
                   ___________            ___________
                  /     2                /     2
            1 - \/  C1*x  + 1          \/  C1*x  + 1  + 1
    [f(x) = ------------------, f(x) = ------------------]
                    x                          x

    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """

    # 提示符号，指明此类用于求解可化为可分离形式的微分方程
    hint = "separable_reduced"

    # 表示此类具有积分解
    has_integral = True

    # 方程的阶数列表，此处为一阶微分方程
    order = [1]

    def _degree(self, expr, x):
        # 计算表达式中关于 x 的次数
        # 如果表达式为 x**p*y 形式，返回 p
        # expr：待计算的表达式
        # x：变量
        for val in expr:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    return (val.as_base_exp()[1])
                elif val == x:
                    return (val.as_base_exp()[1])
                else:
                    return self._degree(val.args, x)
        return 0
    def _powers(self, expr):
        # 定义一个函数，用于计算相对于 f(x) 的不同幂次的 x 的幂次
        # 如果 expr = x**p * f(x)**q，则返回 {p/q}
        pows = set()  # 初始化一个空集合，用于存储幂次比
        fx = self.ode_problem.func  # 获取微分方程的函数部分
        x = self.ode_problem.sym  # 获取微分方程的符号变量部分
        self.y = Dummy('y')  # 创建一个虚拟符号 y
        if isinstance(expr, Add):
            exprs = expr.atoms(Add)
        elif isinstance(expr, Mul):
            exprs = expr.atoms(Mul)
        elif isinstance(expr, Pow):
            exprs = expr.atoms(Pow)
        else:
            exprs = {expr}
        
        for arg in exprs:
            if arg.has(x):
                _, u = arg.as_independent(x, fx)
                # 计算 u.subs(fx, self.y) 相对于 x 和 self.y 的次数比
                pow = self._degree((u.subs(fx, self.y), ), x)/self._degree((u.subs(fx, self.y), ), self.y)
                pows.add(pow)  # 将计算出的次数比添加到集合中
        
        return pows  # 返回所有的次数比集合

    def _verify(self, fx):
        num, den = self.wilds_match()  # 获取匹配的数值和分母
        x = self.ode_problem.sym  # 获取微分方程的符号变量部分
        factor = simplify(x/fx*num/den)  # 简化 x/fx*num/den 的表达式
        # 尝试将因子表示为 x^n*y 的形式
        # 其中 n 是因子中 x 的最低幂次
        # 首先从 factor.atoms(Mul) 中移除类似 sqrt(2)*3 的项
        num, dem = factor.as_numer_denom()
        num = expand(num)
        dem = expand(dem)
        pows = self._powers(num)  # 获取数值部分的幂次比
        pows.update(self._powers(dem))  # 添加分母部分的幂次比
        pows = list(pows)  # 转换为列表形式
        if(len(pows)==1) and pows[0]!=zoo:
            self.t = Dummy('t')  # 创建一个虚拟符号 t
            self.r2 = {'t': self.t}  # 设置 r2 字典的 't' 键为虚拟符号 t
            num = num.subs(x**pows[0]*fx, self.t)
            dem = dem.subs(x**pows[0]*fx, self.t)
            test = num/dem
            free = test.free_symbols
            if len(free) == 1 and free.pop() == self.t:
                # 如果 test 中只有一个自由符号且为 self.t，则更新 r2 字典
                self.r2.update({'power' : pows[0], 'u' : test})
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func  # 获取微分方程的函数部分
        x = self.ode_problem.sym  # 获取微分方程的符号变量部分
        u = self.r2['u'].subs(self.r2['t'], self.y)  # 替换 r2 中的 't' 为虚拟符号 y
        ycoeff = 1/(self.y*(self.r2['power'] - u))  # 计算 y 系数
        m1 = {self.y: 1, x: -1/x, 'coeff': 1}  # 定义匹配对象 m1
        m2 = {self.y: ycoeff, x: 1, 'coeff': 1}  # 定义匹配对象 m2
        return m1, m2, x, x**self.r2['power']*fx  # 返回匹配对象和表达式
# 定义名为 HomogeneousCoeffSubsDepDivIndep 的类，继承自 SinglePatternODESolver 类。
class HomogeneousCoeffSubsDepDivIndep(SinglePatternODESolver):
    r"""
    解决具有齐次系数的一阶微分方程，使用替换 `u_1 = \frac{\text{<dependent
    variable>}}{\text{<independent variable>}}`。

    This is a differential equation

    .. math:: P(x, y) + Q(x, y) dy/dx = 0

    such that `P` and `Q` are homogeneous and of the same order.  A function
    `F(x, y)` is homogeneous of order `n` if `F(x t, y t) = t^n F(x, y)`.
    Equivalently, `F(x, y)` can be rewritten as `G(y/x)` or `H(x/y)`.  See
    also the docstring of :py:meth:`~sympy.solvers.ode.homogeneous_order`.

    如果上述微分方程中的系数 `P` 和 `Q` 是同一阶的齐次函数，则可以证明替换 `y = u_1 x`（即 `u_1 = y/x`）
    将把微分方程转化为关于变量 `x` 和 `u` 可分离的方程。如果 `h(u_1)` 是通过在微分方程 `P(x, f(x)) +
    Q(x, f(x)) f'(x) = 0` 中进行替换 `u_1 = f(x)/x` 而得到的函数 `P(x, f(x))` 的结果，而 `g(u_2)`
    是在 `Q(x, f(x))` 中进行替换的结果，则一般解为：

    Where `u_1 h(u_1) + g(u_1) \ne 0` and `x \ne 0`.

    See also the docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`.

    Examples
    ========

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_subs_dep_div_indep', simplify=False))
                          /          3   \
                          |3*f(x)   f (x)|
                       log|------ + -----|
                          |  x         3 |
                          \           x  /
    log(x) = log(C1) - -------------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest
    """
    # 提示信息，指示函数的用途或含义
    hint = "1st_homogeneous_coeff_subs_dep_div_indep"
    
    # 标志，表示是否存在积分的结果
    has_integral = True
    
    # 方程的阶数列表，这里只有一个值为1的元素
    order = [1]

    # 定义一个方法，用于获取通配符表达式中排除特定导数的 Wild 对象
    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    # 定义一个方法，构建一个表达式，表示给定函数关于变量 x 的导数及其系数的线性组合
    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e * fx.diff(x)

    # 定义一个方法，验证给定的函数 fx 是否符合某个特定的方程形式
    def _verify(self, fx):
        # 匹配通配符表达式并赋值
        self.d, self.e = self.wilds_match()
        # 创建一个虚拟变量 y
        self.y = Dummy('y')
        x = self.ode_problem.sym
        # 将函数 fx 替换为虚拟变量 y 并分离变量
        self.d = separatevars(self.d.subs(fx, self.y))
        self.e = separatevars(self.e.subs(fx, self.y))
        # 计算 d 和 e 的齐次阶数
        ordera = homogeneous_order(self.d, x, self.y)
        orderb = homogeneous_order(self.e, x, self.y)
        # 如果两者阶数相等且不为 None
        if ordera == orderb and ordera is not None:
            # 创建虚拟变量 u
            self.u = Dummy('u')
            # 简化并检查方程是否为零
            if simplify((self.d + self.u * self.e).subs({x: 1, self.y: self.u})) != 0:
                return True
            return False
        return False

    # 定义一个方法，返回匹配对象的列表
    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        xarg = 0
        yarg = 0
        return [self.d, self.e, fx, x, self.u, self.u1, self.y, xarg, yarg]

    # 定义一个方法，获取通解表达式
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取匹配对象
        d, e, fx, x, u, u1, y, xarg, yarg = self._get_match_object()
        # 获取常数 C1
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        # 构建积分表达式
        int = Integral((-e / (d + u1 * e)).subs({x: 1, y: u1}), (u1, None, fx / x))
        # 合并对数表达式
        sol = logcombine(Eq(log(x), int + log(C1)), force=True)
        # 替换通解中的变量
        gen_sol = sol.subs(fx, u).subs(((u, u - yarg), (x, x - xarg), (u, fx)))
        return [gen_sol]
# 定义一个类 HomogeneousCoeffSubsIndepDivDep，继承自 SinglePatternODESolver 类
class HomogeneousCoeffSubsIndepDivDep(SinglePatternODESolver):
    # 类的文档字符串，解释了这个类的作用和使用方法
    r"""
    Solves a 1st order differential equation with homogeneous coefficients
    using the substitution `u_2 = \frac{\text{<independent
    variable>}}{\text{<dependent variable>}}`.

    This is a differential equation

    .. math:: P(x, y) + Q(x, y) dy/dx = 0

    such that `P` and `Q` are homogeneous and of the same order.  A function
    `F(x, y)` is homogeneous of order `n` if `F(x t, y t) = t^n F(x, y)`.
    Equivalently, `F(x, y)` can be rewritten as `G(y/x)` or `H(x/y)`.  See
    also the docstring of :py:meth:`~sympy.solvers.ode.homogeneous_order`.

    If the coefficients `P` and `Q` in the differential equation above are
    homogeneous functions of the same order, then it can be shown that the
    substitution `x = u_2 y` (i.e. `u_2 = x/y`) will turn the differential
    equation into an equation separable in the variables `y` and `u_2`.  If
    `h(u_2)` is the function that results from making the substitution `u_2 =
    x/f(x)` on `P(x, f(x))` and `g(u_2)` is the function that results from the
    substitution on `Q(x, f(x))` in the differential equation `P(x, f(x)) +
    Q(x, f(x)) f'(x) = 0`, then the general solution is:

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f, g, h = map(Function, ['f', 'g', 'h'])
    >>> genform = g(x/f(x)) + h(x/f(x))*f(x).diff(x)
    >>> pprint(genform)
     / x  \    / x  \ d
    g|----| + h|----|*--(f(x))
     \f(x)/    \f(x)/ dx
    >>> pprint(dsolve(genform, f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral'))
                 x
                ----
                f(x)
                  /
                 |
                 |       -g(u1)
                 |  ---------------- d(u1)
                 |  u1*g(u1) + h(u1)
                 |
                /
    <BLANKLINE>
    f(x) = C1*e

    Where `u_1 g(u_1) + h(u_1) \ne 0` and `f(x) \ne 0`.

    See also the docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`.

    Examples
    ========

    >>> from sympy import Function, pprint, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep',
    ... simplify=False))
                             /   2     \
                             |3*x      |
                          log|----- + 1|
                             | 2       |
                             \f (x)    /
    log(f(x)) = log(C1) - --------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    # 提示符号，表明这个类使用第一类齐次系数替换独立变量与依赖变量的分式形式
    hint = "1st_homogeneous_coeff_subs_indep_div_dep"
    # 类具有积分的能力，设置为 True
    has_integral = True
    order = [1]
    # 定义一个列表变量 order，包含元素 1

    def _wilds(self, f, x, order):
        # 定义一个方法 _wilds，接受参数 self, f, x, order
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        # 创建一个 Wild 对象 d，表示一个未知表达式，排除 f(x) 及其二阶导数
        e = Wild('e', exclude=[f(x).diff(x)])
        # 创建一个 Wild 对象 e，表示一个未知表达式，排除 f(x) 的一阶导数
        return d, e
        # 返回 Wild 对象 d 和 e

    def _equation(self, fx, x, order):
        # 定义一个方法 _equation，接受参数 self, fx, x, order
        d, e = self.wilds()
        # 调用 self.wilds() 方法获取 Wild 对象 d 和 e
        return d + e*fx.diff(x)
        # 返回表达式 d + e*fx.diff(x)，表示一个方程

    def _verify(self, fx):
        # 定义一个方法 _verify，接受参数 self, fx
        self.d, self.e = self.wilds_match()
        # 调用 self.wilds_match() 方法获取 Wild 对象 d 和 e，并赋值给 self.d 和 self.e
        self.y = Dummy('y')
        # 创建一个 Dummy 符号 y
        x = self.ode_problem.sym
        # 从 self.ode_problem 获取符号对象 x
        self.d = separatevars(self.d.subs(fx, self.y))
        # 将 self.d 中的 fx 替换为 self.y，并使用 separatevars 处理
        self.e = separatevars(self.e.subs(fx, self.y))
        # 将 self.e 中的 fx 替换为 self.y，并使用 separatevars 处理
        ordera = homogeneous_order(self.d, x, self.y)
        # 计算 self.d 关于 x 和 self.y 的齐次阶数
        orderb = homogeneous_order(self.e, x, self.y)
        # 计算 self.e 关于 x 和 self.y 的齐次阶数
        if ordera == orderb and ordera is not None:
            # 如果 ordera 等于 orderb 并且不为 None
            self.u = Dummy('u')
            # 创建一个 Dummy 符号 u
            if simplify((self.e + self.u*self.d).subs({x: self.u, self.y: 1})) != 0:
                # 如果简化后的表达式不等于 0
                return True
            return False
        return False

    def _get_match_object(self):
        # 定义一个方法 _get_match_object，不接受参数
        fx = self.ode_problem.func
        # 获取 self.ode_problem 的函数符号 fx
        x = self.ode_problem.sym
        # 获取 self.ode_problem 的符号对象 x
        self.u1 = Dummy('u1')
        # 创建一个 Dummy 符号 u1
        xarg = 0
        # 初始化变量 xarg 为 0
        yarg = 0
        # 初始化变量 yarg 为 0
        return [self.d, self.e, fx, x, self.u, self.u1, self.y, xarg, yarg]
        # 返回包含各个符号和变量的列表

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 定义一个方法 _get_general_solution，接受关键字参数 simplify_flag，默认为 True
        d, e, fx, x, u, u1, y, xarg, yarg = self._get_match_object()
        # 调用 self._get_match_object() 方法获取各个符号和变量
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        # 从 self.ode_problem 获取一个编号为 1 的常数 C1
        int = Integral(simplify((-d/(e + u1*d)).subs({x: u1, y: 1})), (u1, None, x/fx)) # type: ignore
        # 创建一个积分对象 int，表示一个积分表达式
        sol = logcombine(Eq(log(fx), int + log(C1)), force=True)
        # 调用 logcombine 函数，将 log(fx) 与积分表达式合并为一个等式 sol
        gen_sol = sol.subs(fx, u).subs(((u, u - yarg), (x, x - xarg), (u, fx)))
        # 将 sol 中的 fx 替换为 u，并做额外的替换操作，得到通解 gen_sol
        return [gen_sol]
        # 返回包含通解的列表
class HomogeneousCoeffBest(HomogeneousCoeffSubsIndepDivDep, HomogeneousCoeffSubsDepDivIndep):
    r"""
    Returns the best solution to an ODE from the two hints
    ``1st_homogeneous_coeff_subs_dep_div_indep`` and
    ``1st_homogeneous_coeff_subs_indep_div_dep``.

    This is as determined by :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity`.

    See the
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`
    and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`
    docstrings for more information on these hints.  Note that there is no
    ``ode_1st_homogeneous_coeff_best_Integral`` hint.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_best', simplify=False))
                             /   2     \
                             |3*x      |
                          log|----- + 1|
                             | 2       |
                             \f (x)    /
    log(f(x)) = log(C1) - --------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    # 设置提示信息为 "1st_homogeneous_coeff_best"
    hint = "1st_homogeneous_coeff_best"
    # 没有积分
    has_integral = False
    # 方程阶数为1
    order = [1]

    def _verify(self, fx):
        # 验证函数，调用两个父类的验证函数
        return HomogeneousCoeffSubsIndepDivDep._verify(self, fx) and \
               HomogeneousCoeffSubsDepDivIndep._verify(self, fx)

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 解一阶齐次常系数微分方程的通解
        # 有两种替换可以解决方程，u1=y/x 和 u2=x/y
        # 它们产生不同的积分，因此尝试两者并确定哪个更简单
        sol1 = HomogeneousCoeffSubsIndepDivDep._get_general_solution(self)
        sol2 = HomogeneousCoeffSubsDepDivIndep._get_general_solution(self)
        fx = self.ode_problem.func
        if simplify_flag:
            # 简化解，使用函数 odesimp
            sol1 = odesimp(self.ode_problem.eq, *sol1, fx, "1st_homogeneous_coeff_subs_indep_div_dep")
            sol2 = odesimp(self.ode_problem.eq, *sol2, fx, "1st_homogeneous_coeff_subs_dep_div_indep")
        # 返回两个解中较简单的一个
        return min([sol1, sol2], key=lambda x: ode_sol_simplicity(x, fx, trysolving=not simplify))


class LinearCoefficients(HomogeneousCoeffBest):
    r"""
    Solves a differential equation with linear coefficients.

    The general form of a differential equation with linear coefficients is

    .. math:: y' + F\left(\!\frac{a_1 x + b_1 y + c_1}{a_2 x + b_2 y +
                c_2}\!\right) = 0\text{,}

    where `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are constants and `a_1 b_2
    - a_2 b_1 \ne 0`.

    This can be solved by substituting:
    hint = "linear_coefficients"
    # 设置提示信息为"linear_coefficients"

    has_integral = True
    # 设定具有积分解

    order = [1]
    # 设置方程的阶数为1

    def _wilds(self, f, x, order):
        # 定义内部函数_wilds，接受参数self, f, x, order
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        # 使用Wild创建符号d，排除f(x)关于x的一阶和二阶导数
        e = Wild('e', exclude=[f(x).diff(x)])
        # 使用Wild创建符号e，排除f(x)关于x的一阶导数
        return d, e
        # 返回符号d和e

    def _equation(self, fx, x, order):
        # 定义内部函数_equation，接受参数self, fx, x, order
        d, e = self.wilds()
        # 调用wilds()方法获取符号d和e
        return d + e*fx.diff(x)
        # 返回表达式d + e*fx.diff(x)

    def _verify(self, fx):
        # 定义内部函数_verify，接受参数self, fx
        self.d, self.e = self.wilds_match()
        # 调用wilds_match()方法匹配符号d和e
        a, b = self.wilds()
        # 调用wilds()方法获取符号a和b
        F = self.d/self.e
        # 计算F = d/e
        x = self.ode_problem.sym
        # 获取ode_problem的sym属性赋给x
        params = self._linear_coeff_match(F, fx)
        # 调用_linear_coeff_match方法匹配F和fx
        if params:
            self.xarg, self.yarg = params
            # 如果有匹配到参数，将参数赋给self.xarg和self.yarg
            u = Dummy('u')
            t = Dummy('t')
            self.y = Dummy('y')
            # 创建虚拟变量u, t和y
            # 为df和f(x)进行虚拟替换
            dummy_eq = self.ode_problem.eq.subs(((fx.diff(x), t), (fx, u)))
            # 在ode_problem的方程中替换fx.diff(x)为t，fx为u
            reps = ((x, x + self.xarg), (u, u + self.yarg), (t, fx.diff(x)), (u, fx))
            # 创建替换列表
            dummy_eq = simplify(dummy_eq.subs(reps))
            # 简化替换后的方程dummy_eq
            # 获取e和d的重设值
            r2 = collect(expand(dummy_eq), [fx.diff(x), fx]).match(a*fx.diff(x) + b)
            # 收集并匹配方程中的fx.diff(x)和fx项
            if r2:
                self.d, self.e = r2[b], r2[a]
                # 如果匹配成功，将r2中的b和a赋给self.d和self.e
                orderd = homogeneous_order(self.d, x, fx)
                # 计算self.d关于fx的齐次阶数
                ordere = homogeneous_order(self.e, x, fx)
                # 计算self.e关于fx的齐次阶数
                if orderd == ordere and orderd is not None:
                    self.d = self.d.subs(fx, self.y)
                    # 如果阶数相同且不为None，用self.y替换self.d中的fx
                    self.e = self.e.subs(fx, self.y)
                    # 用self.y替换self.e中的fx
                    return True
                return False
            return False
    def _linear_coeff_match(self, expr, func):
        r"""
        Helper function to match hint ``linear_coefficients``.

        Matches the expression to the form `(a_1 x + b_1 f(x) + c_1)/(a_2 x + b_2
        f(x) + c_2)` where the following conditions hold:

        1. `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are Rationals;
        2. `c_1` or `c_2` are not equal to zero;
        3. `a_2 b_1 - a_1 b_2` is not equal to zero.

        Return ``xarg``, ``yarg`` where

        1. ``xarg`` = `(b_2 c_1 - b_1 c_2)/(a_2 b_1 - a_1 b_2)`
        2. ``yarg`` = `(a_1 c_2 - a_2 c_1)/(a_2 b_1 - a_1 b_2)`

        Examples
        ========

        >>> from sympy import Function, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import LinearCoefficients
        >>> f = Function('f')
        >>> eq = (-25*f(x) - 8*x + 62)/(4*f(x) + 11*x - 11)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (1/9, 22/9)
        >>> eq = sin((-5*f(x) - 8*x + 6)/(4*f(x) + x - 1))
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (19/27, 2/27)
        >>> eq = sin(f(x)/x)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))

        """
        # 获取函数对象和自变量 x
        f = func.func
        x = func.args[0]

        def abc(eq):
            r'''
            Internal function of _linear_coeff_match
            that returns Rationals a, b, c
            if eq is a*x + b*f(x) + c, else None.
            '''
            # 确保表达式扩展后的系数是有理数，并返回系数 a, b, c
            eq = _mexpand(eq)
            c = eq.as_independent(x, f(x), as_Add=True)[0]
            if not c.is_Rational:
                return
            a = eq.coeff(x)
            if not a.is_Rational:
                return
            b = eq.coeff(f(x))
            if not b.is_Rational:
                return
            # 如果表达式符合要求，则返回系数 a, b, c
            if eq == a*x + b*f(x) + c:
                return a, b, c

        def match(arg):
            r'''
            Internal function of _linear_coeff_match that returns Rationals a1,
            b1, c1, a2, b2, c2 and a2*b1 - a1*b2 of the expression (a1*x + b1*f(x)
            + c1)/(a2*x + b2*f(x) + c2) if one of c1 or c2 and a2*b1 - a1*b2 is
            non-zero, else None.
            '''
            # 将分子分母合并，然后调用 abc 函数获取各项系数，并确保条件满足
            n, d = arg.together().as_numer_denom()
            m = abc(n)
            if m is not None:
                a1, b1, c1 = m
                m = abc(d)
                if m is not None:
                    a2, b2, c2 = m
                    d = a2*b1 - a1*b2
                    if (c1 or c2) and d:
                        return a1, b1, c1, a2, b2, c2, d

        # 从表达式中获取所有的函数对象，找到符合条件的表达式，并进行匹配
        m = [fi.args[0] for fi in expr.atoms(Function) if fi.func != f and
            len(fi.args) == 1 and not fi.args[0].is_Function] or {expr}
        m1 = match(m.pop())
        if m1 and all(match(mi) == m1 for mi in m):
            a1, b1, c1, a2, b2, c2, denom = m1
            # 返回符合条件的结果 xarg 和 yarg
            return (b2*c1 - b1*c2)/denom, (a1*c2 - a2*c1)/denom
    # 定义一个方法 _get_match_object，用于获取匹配对象
    def _get_match_object(self):
        # 获取ODE问题的函数对象
        fx = self.ode_problem.func
        # 获取ODE问题的符号变量
        x = self.ode_problem.sym
        # 创建一个虚拟符号 'u1'
        self.u1 = Dummy('u1')
        # 创建一个虚拟符号 'u'
        u = Dummy('u')
        # 返回一个包含多个对象的列表，包括 self.d, self.e, fx, x, u, self.u1, self.y, self.xarg, self.yarg
        return [self.d, self.e, fx, x, u, self.u1, self.y, self.xarg, self.yarg]
class NthOrderReducible(SingleODESolver):
    r"""
    Solves ODEs that only involve derivatives of the dependent variable using
    a substitution of the form `f^n(x) = g(x)`.

    For example any second order ODE of the form `f''(x) = h(f'(x), x)` can be
    transformed into a pair of 1st order ODEs `g'(x) = h(g(x), x)` and
    `f'(x) = g(x)`. Usually the 1st order ODE for `g` is easier to solve. If
    that gives an explicit solution for `g` then `f` is found simply by
    integration.
    """

    hint = "nth_order_reducible"  # 提示信息，表明该类处理可化为低阶形式的微分方程
    has_integral = False  # 表示该类不处理积分情况的ODE

    def _matches(self):
        # 判断是否能使用替换和重复积分解决的任何ODE
        # 例如：`d^2/dx^2(y) + x*d/dx(y) = constant
        # f'(x)必须有限才能工作
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        x = self.ode_problem.sym
        r"""
        Matches any differential equation that can be rewritten with a smaller
        order. Only derivatives of ``func`` alone, wrt a single variable,
        are considered, and only in them should ``func`` appear.
        """
        # 确保只处理单变量函数的导数
        assert len(func.args) == 1
        # 找出ODE中出现的所有Derivative对象，并判断是否符合条件
        vc = [d.variable_count[0] for d in eq.atoms(Derivative)
              if d.expr == func and len(d.variable_count) == 1]
        ords = [c for v, c in vc if v == x]
        if len(ords) < 2:
            return False
        self.smallest = min(ords)
        # 确保func在导数外不出现
        D = Dummy()
        if eq.subs(func.diff(x, self.smallest), D).has(func):
            return False
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        n = self.smallest
        # 为g获取一个唯一的函数名
        names = [a.name for a in eq.atoms(AppliedUndef)]
        while True:
            name = Dummy().name
            if name not in names:
                g = Function(name)
                break
        w = f(x).diff(x, n)
        geq = eq.subs(w, g(x))
        gsol = dsolve(geq, g(x))

        if not isinstance(gsol, list):
            gsol = [gsol]

        # 可能存在多个简化后的ODE解：
        fsol = []
        for gsoli in gsol:
            fsoli = dsolve(gsoli.subs(g(x), w), f(x))  # 或进行n次积分
            fsol.append(fsoli)

        return fsol
    Solves 2nd order linear differential equations.

    It computes special function solutions which can be expressed using the
    2F1, 1F1 or 0F1 hypergeometric functions.

    .. math:: y'' + A(x) y' + B(x) y = 0\text{,}

    where `A` and `B` are rational functions.

    These kinds of differential equations have solution of non-Liouvillian form.

    Given linear ODE can be obtained from 2F1 given by

    .. math:: (x^2 - x) y'' + ((a + b + 1) x - c) y' + b a y = 0\text{,}

    where {a, b, c} are arbitrary constants.

    Notes
    =====

    The algorithm should find any solution of the form

    .. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

    where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
    Currently only the 2F1 case is implemented in SymPy but the other cases are
    described in the paper and could be implemented in future (contributions
    welcome!).


    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = (x*x - x)*f(x).diff(x,2) + (5*x - 1)*f(x).diff(x) + 4*f(x)
    >>> pprint(dsolve(eq, f(x), '2nd_hypergeometric'))
                                        _
           /        /           4  \\  |_  /-1, -1 |  \
           |C1 + C2*|log(x) + -----||* |   |       | x|
           \        \         x + 1// 2  1 \  1    |  /
    f(x) = --------------------------------------------
                                    3
                             (x - 1)


    References
    ==========

    - "Non-Liouvillian solutions for second order linear ODEs" by L. Chan, E.S. Cheb-Terrab

    """

    # 设置ODE求解的提示为"2nd_hypergeometric"
    hint = "2nd_hypergeometric"
    # 设置是否具有积分解的标志为True
    has_integral = True

    def _matches(self):
        # 获取预处理后的ODE方程和函数
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        # 尝试匹配二阶超几何类型的ODE
        r = match_2nd_hypergeometric(eq, func)
        self.match_object = None
        if r:
            A, B = r
            # 判断等效超几何方程类型，并尝试匹配2F1类型的ODE
            d = equivalence_hypergeometric(A, B, func)
            if d:
                if d['type'] == "2F1":
                    # 如果匹配成功，则更新匹配对象
                    self.match_object = match_2nd_2F1_hypergeometric(d['I0'], d['k'], d['sing_point'], func)
                    if self.match_object is not None:
                        self.match_object.update({'A':A, 'B':B})
            # 可以扩展为1F1和0F1类型
        # 返回是否有匹配对象
        return self.match_object is not None

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取ODE方程和函数
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        # 如果匹配对象类型是"2F1"
        if self.match_object['type'] == "2F1":
            # 调用函数获取2F1超几何方法的解
            sol = get_sol_2F1_hypergeometric(eq, func, self.match_object)
            # 如果解为None，则抛出未实现错误
            if sol is None:
                raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                    + " the hypergeometric method")

        # 返回解的列表
        return [sol]
class NthLinearConstantCoeffHomogeneous(SingleODESolver):
    r"""
    Solves an `n`\th order linear homogeneous differential equation with
    constant coefficients.

    This is an equation of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x)
                + a_0 f(x) = 0\text{.}

    These equations can be solved in a general manner, by taking the roots of
    the characteristic equation `a_n m^n + a_{n-1} m^{n-1} + \cdots + a_1 m +
    a_0 = 0`.  The solution will then be the sum of `C_n x^i e^{r x}` terms,
    for each where `C_n` is an arbitrary constant, `r` is a root of the
    characteristic equation and `i` is one of each from 0 to the multiplicity
    of the root - 1 (for example, a root 3 of multiplicity 2 would create the
    terms `C_1 e^{3 x} + C_2 x e^{3 x}`).  The exponential is usually expanded
    for complex roots using Euler's equation `e^{I x} = \cos(x) + I \sin(x)`.
    Complex roots always come in conjugate pairs in polynomials with real
    coefficients, so the two roots will be represented (after simplifying the
    constants) as `e^{a x} \left(C_1 \cos(b x) + C_2 \sin(b x)\right)`.

    If SymPy cannot find exact roots to the characteristic equation, a
    :py:class:`~sympy.polys.rootoftools.ComplexRootOf` instance will be return
    instead.

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(f(x).diff(x, 5) + 10*f(x).diff(x) - 2*f(x), f(x),
    ... hint='nth_linear_constant_coeff_homogeneous')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), C5*exp(x*CRootOf(_x**5 + 10*_x - 2, 0))
    + (C1*sin(x*im(CRootOf(_x**5 + 10*_x - 2, 1)))
    + C2*cos(x*im(CRootOf(_x**5 + 10*_x - 2, 1))))*exp(x*re(CRootOf(_x**5 + 10*_x - 2, 1)))
    + (C3*sin(x*im(CRootOf(_x**5 + 10*_x - 2, 3)))
    + C4*cos(x*im(CRootOf(_x**5 + 10*_x - 2, 3))))*exp(x*re(CRootOf(_x**5 + 10*_x - 2, 3))))

    Note that because this method does not involve integration, there is no
    ``nth_linear_constant_coeff_homogeneous_Integral`` hint.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 4) + 2*f(x).diff(x, 3) -
    ... 2*f(x).diff(x, 2) - 6*f(x).diff(x) + 5*f(x), f(x),
    ... hint='nth_linear_constant_coeff_homogeneous'))
                        x                            -2*x
    f(x) = (C1 + C2*x)*e  + (C3*sin(x) + C4*cos(x))*e

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation section:
      Nonhomogeneous_equation_with_constant_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 211

    # indirect doctest
    """

    # 提示符号，指示ODE解算器使用的方法
    hint = "nth_linear_constant_coeff_homogeneous"
    # 标志，指示此解算器方法是否有积分版本
    has_integral = False
    # 返回是否满足特定条件的函数，依赖于给定的常微分方程问题的属性
    def _matches(self):
        # 获取高阶自由方程
        eq = self.ode_problem.eq_high_order_free
        # 获取函数表达式
        func = self.ode_problem.func
        # 获取微分方程的阶数
        order = self.ode_problem.order
        # 获取符号变量
        x = self.ode_problem.sym
        # 获取线性系数
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        
        # 检查特定条件是否满足
        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            # 如果满足条件，返回True
            if not self.r[-1]:
                return True
            else:
                return False
        
        # 如果不满足条件，返回False
        return False

    # 获取常微分方程的一般解
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取函数表达式
        fx = self.ode_problem.func
        # 获取微分方程的阶数
        order = self.ode_problem.order
        # 根据线性系数和函数表达式获取常数和特征方程解
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, fx, order)
        # 获取编号常数的生成器
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        # 构建一般解表达式
        gsol = Add(*[i*j for (i, j) in zip(constants, roots)])
        # 将一般解表达式与原微分方程关联起来
        gsol = Eq(fx, gsol)
        
        # 如果指定了简化标志，对解进行简化处理
        if simplify_flag:
            gsol = _get_simplified_sol([gsol], fx, collectterms)

        # 返回简化后的一般解列表
        return [gsol]
# 定义一个类，继承自 SingleODESolver，用于解 n 阶带有常系数的线性微分方程
class NthLinearConstantCoeffVariationOfParameters(SingleODESolver):
    """
    Solves an `n`\th order linear differential equation with constant
    coefficients using the method of variation of parameters.

    This method works on any differential equations of the form

    .. math:: f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x) + a_0
                f(x) = P(x)\text{.}

    This method works by assuming that the particular solution takes the form

    .. math:: \sum_{x=1}^{n} c_i(x) y_i(x)\text{,}

    where `y_i` is the `i`\th solution to the homogeneous equation.  The
    solution is then solved using Wronskian's and Cramer's Rule.  The
    particular solution is given by

    .. math:: \sum_{x=1}^n \left( \int \frac{W_i(x)}{W(x)} \,dx
                \right) y_i(x) \text{,}

    where `W(x)` is the Wronskian of the fundamental system (the system of `n`
    linearly independent solutions to the homogeneous equation), and `W_i(x)`
    is the Wronskian of the fundamental system with the `i`\th column replaced
    with `[0, 0, \cdots, 0, P(x)]`.

    This method is general enough to solve any `n`\th order inhomogeneous
    linear differential equation with constant coefficients, but sometimes
    SymPy cannot simplify the Wronskian well enough to integrate it.  If this
    method hangs, try using the
    ``nth_linear_constant_coeff_variation_of_parameters_Integral`` hint and
    simplifying the integrals manually.  Also, prefer using
    ``nth_linear_constant_coeff_undetermined_coefficients`` when it
    applies, because it does not use integration, making it faster and more
    reliable.

    Warning, using simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters' in
    :py:meth:`~sympy.solvers.ode.dsolve` may cause it to hang, because it will
    not attempt to simplify the Wronskian before integrating.  It is
    recommended that you only use simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters_Integral' for this
    method, especially if the solution to the homogeneous equation has
    trigonometric functions in it.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, log
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 3) - 3*f(x).diff(x, 2) +
    ... 3*f(x).diff(x) - f(x) - exp(x)*log(x), f(x),
    ... hint='nth_linear_constant_coeff_variation_of_parameters'))
           /       /       /     x*log(x)   11*x\\\  x
    f(x) = |C1 + x*|C2 + x*|C3 + -------- - ----|||*e
           \       \       \        6        36 ///

    References
    ==========

    - https://en.wikipedia.org/wiki/Variation_of_parameters
    - https://planetmath.org/VariationOfParameters
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 233

    # indirect doctest

    """
    # 提示符号，表明使用参数变化法解法
    hint = "nth_linear_constant_coeff_variation_of_parameters"
    # 指示此类有积分方法
    has_integral = True
    # 获取当前ODE问题的高阶自由方程
    eq = self.ode_problem.eq_high_order_free
    # 获取ODE问题的函数表达式
    func = self.ode_problem.func
    # 获取ODE问题的阶数
    order = self.ode_problem.order
    # 获取ODE问题的符号变量
    x = self.ode_problem.sym
    # 获取ODE问题的线性系数
    self.r = self.ode_problem.get_linear_coefficients(eq, func, order)

    # 如果阶数不为零且存在线性系数且所有非负索引的线性系数中不含有符号变量x
    if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
        # 如果存在最后一个线性系数
        if self.r[-1]:
            # 返回True
            return True
        else:
            # 否则返回False
            return False
    # 默认返回False
    return False

    # 获取一般解
def _get_general_solution(self, *, simplify_flag: bool = True):
    # 获取当前ODE问题的高阶自由方程
    eq = self.ode_problem.eq_high_order_free
    # 获取ODE问题的函数
    f = self.ode_problem.func.func
    # 获取ODE问题的符号变量
    x = self.ode_problem.sym
    # 获取ODE问题的阶数
    order = self.ode_problem.order
    # 使用线性系数计算根和收集项
    roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
    # 获取编号的常数项生成器
    constants = self.ode_problem.get_numbered_constants(num=len(roots))
    # 计算齐次解
    homogen_sol = Add(*[i*j for (i, j) in zip(constants, roots)])
    # 将齐次解表示为方程
    homogen_sol = Eq(f(x), homogen_sol)
    # 使用参数变化法求解
    homogen_sol = _solve_variation_of_parameters(eq, f(x), roots, homogen_sol, order, self.r, simplify_flag)
    # 如果需要简化，则对简化后的解进行处理
    if simplify_flag:
        homogen_sol = _get_simplified_sol([homogen_sol], f(x), collectterms)
    # 返回结果
    return [homogen_sol]
# 定义一个继承自 SingleODESolver 的类，用于解 n 阶具有常系数的线性微分方程，使用待定系数法。

class NthLinearConstantCoeffUndeterminedCoefficients(SingleODESolver):
    r"""
    Solves an `n`\th order linear differential equation with constant
    coefficients using the method of undetermined coefficients.
    
    解决具有常系数的 n 阶线性微分方程，使用待定系数法。
    
    This method works on differential equations of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x)
                + a_0 f(x) = P(x)\text{,}

    where `P(x)` is a function that has a finite number of linearly
    independent derivatives.
    
    此方法适用于形如上述方程的微分方程，其中 `P(x)` 是具有有限个线性独立导数的函数。

    Functions that fit this requirement are finite sums functions of the form
    `a x^i e^{b x} \sin(c x + d)` or `a x^i e^{b x} \cos(c x + d)`, where `i`
    is a non-negative integer and `a`, `b`, `c`, and `d` are constants.  For
    example any polynomial in `x`, functions like `x^2 e^{2 x}`, `x \sin(x)`,
    and `e^x \cos(x)` can all be used.  Products of `\sin`'s and `\cos`'s have
    a finite number of derivatives, because they can be expanded into `\sin(a
    x)` and `\cos(b x)` terms.  However, SymPy currently cannot do that
    expansion, so you will need to manually rewrite the expression in terms of
    the above to use this method.  So, for example, you will need to manually
    convert `\sin^2(x)` into `(1 + \cos(2 x))/2` to properly apply the method
    of undetermined coefficients on it.
    
    满足上述要求的函数是形如 `a x^i e^{b x} \sin(c x + d)` 或 `a x^i e^{b x} \cos(c x + d)` 的有限和函数，其中 `i`
    是非负整数，`a`、`b`、`c` 和 `d` 是常数。例如任何 `x` 的多项式，像 `x^2 e^{2 x}`, `x \sin(x)`, 和 `e^x \cos(x)` 都可以使用。`\sin` 和 `\cos` 的乘积具有有限个导数，因为它们可以展开为 `\sin(a x)` 和 `\cos(b x)` 项。然而，SymPy 目前无法执行此展开，因此您需要手动将表达式转换为上述形式来使用此方法。例如，您需要手动将 `\sin^2(x)` 转换为 `(1 + \cos(2 x))/2` 以正确应用待定系数法。

    This method works by creating a trial function from the expression and all
    of its linear independent derivatives and substituting them into the
    original ODE.  The coefficients for each term will be a system of linear
    equations, which are be solved for and substituted, giving the solution.
    If any of the trial functions are linearly dependent on the solution to
    the homogeneous equation, they are multiplied by sufficient `x` to make
    them linearly independent.
    
    此方法通过从表达式及其所有线性独立的导数创建试探函数，并将其代入原始的微分方程中来工作。每个项的系数将形成一个线性方程组，解这些方程组并代入得到解。如果任何试探函数与齐次方程的解线性相关，则乘以足够的 `x` 使它们线性无关。

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 2) + 2*f(x).diff(x) + f(x) -
    ... 4*exp(-x)*x**2 + cos(2*x), f(x),
    ... hint='nth_linear_constant_coeff_undetermined_coefficients'))
           /       /      3\\
           |       |     x ||  -x   4*sin(2*x)   3*cos(2*x)
    f(x) = |C1 + x*|C2 + --||*e   - ---------- + ----------
           \       \     3 //           25           25

    References
    ==========

    - https://en.wikipedia.org/wiki/Method_of_undetermined_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 221
    
    # indirect doctest

    """
    
    # 提示符号，指示使用的求解方法为待定系数法
    hint = "nth_linear_constant_coeff_undetermined_coefficients"
    
    # 是否具有积分，这里设为 False，表示解法不涉及积分
    has_integral = False
    # 定义一个方法 `_matches`，用于检查给定的高阶自由常微分方程是否匹配特定条件
    def _matches(self):
        # 获取高阶自由常微分方程
        eq = self.ode_problem.eq_high_order_free
        # 获取函数表达式
        func = self.ode_problem.func
        # 获取方程的阶数
        order = self.ode_problem.order
        # 获取符号变量
        x = self.ode_problem.sym
        # 获取方程的线性系数
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        # 初始化匹配结果为 False
        does_match = False
        # 检查是否满足条件：方程阶数大于 0、存在线性系数且所有正系数中不含有符号变量 x
        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            # 检查最后一个线性系数是否存在
            if self.r[-1]:
                # 构造齐次方程
                eq_homogeneous = Add(eq, -self.r[-1])
                # 使用未定系数法检查是否匹配
                undetcoeff = _undetermined_coefficients_match(self.r[-1], x, func, eq_homogeneous)
                # 如果匹配成功
                if undetcoeff['test']:
                    # 设置试探集合
                    self.trialset = undetcoeff['trialset']
                    # 修改匹配结果为 True
                    does_match = True
        # 返回匹配结果
        return does_match

    # 定义一个方法 `_get_general_solution`，用于获取常微分方程的通解
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取常微分方程
        eq = self.ode_problem.eq
        # 获取函数表达式
        f = self.ode_problem.func.func
        # 获取符号变量
        x = self.ode_problem.sym
        # 获取方程的阶数
        order = self.ode_problem.order
        # 获取常数特征方程的解和收集项
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
        # 获取编号的常数集合
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        # 构造齐次解
        homogen_sol = Add(*[i*j for (i, j) in zip(constants, roots)])
        # 将齐次解表示为方程形式
        homogen_sol = Eq(f(x), homogen_sol)
        # 更新解的信息和简化标志
        self.r.update({'list': roots, 'sol': homogen_sol, 'simpliy_flag': simplify_flag})
        # 解非齐次方程，使用未定系数法和试探集合
        gsol = _solve_undetermined_coefficients(eq, f(x), order, self.r, self.trialset)
        # 如果需要简化解
        if simplify_flag:
            # 对解进行简化处理，使用收集项和函数表达式
            gsol = _get_simplified_sol([gsol], f(x), collectterms)
        # 返回简化后的解列表
        return [gsol]
# 定义一个类，继承自 SingleODESolver，用于解决 n 阶线性齐次可变系数 Cauchy-Euler 方程
class NthLinearEulerEqHomogeneous(SingleODESolver):
    r"""
    解决 n 阶线性齐次可变系数 Cauchy-Euler 方程。

    这种方程的形式为 `0 = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x) \cdots`。

    这些方程可以通过将形如 `f(x) = x^r` 的解代入，并为 `r` 求解特征方程来一般解。当存在重复根时，
    我们增加形如 `C_{r k} \ln^k(x) x^r` 的额外项，其中 `C_{r k}` 是任意积分常数，`r` 是特征方程的根，
    `k` 是 `r` 的重数。在根为复数的情况下，返回形如 `C_1 x^a \sin(b \log(x)) + C_2 x^a \cos(b \log(x))` 的解，
    基于 Euler 公式的展开。一般解是找到的所有项的和。如果 SymPy 无法找到特征方程的精确根，
    将返回 :py:obj:`~.ComplexRootOf` 实例。

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(4*x**2*f(x).diff(x, 2) + f(x), f(x),
    ... hint='nth_linear_euler_eq_homogeneous')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), sqrt(x)*(C1 + C2*log(x)))

    注意，因为这种方法不涉及积分，所以没有 ``nth_linear_euler_eq_homogeneous_Integral`` 提示。

    以下内容供内部使用：

    - ``returns = 'sol'`` 返回 ODE 的解。
    - ``returns = 'list'`` 返回一个线性无关解的列表，对应于基本解集，用于非齐次解法如参数变化法和未定系数法。
      注意，尽管解应该线性无关，但这个函数并没有显式检查。您可以使用 ``assert simplify(wronskian(sollist)) != 0`` 来检查线性无关性。
      同样， ``assert len(sollist) == order`` 也需要通过。
    - ``returns = 'both'``，返回一个字典 ``{'sol': <ODE 的解>, 'list': <线性无关解的列表>}``。

    示例
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)*x**2 - 4*f(x).diff(x)*x + 6*f(x)
    >>> pprint(dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_homogeneous'))
            2
    f(x) = x *(C1 + C2*x)

    参考资料
    ==========

    - https://en.wikipedia.org/wiki/Cauchy%E2%80%93Euler_equation
    - C. Bender & S. Orszag, "Advanced Mathematical Methods for Scientists and
      Engineers", Springer 1999, pp. 12

    # indirect doctest

    """
    # 提示类型，用于区分求解方法
    hint = "nth_linear_euler_eq_homogeneous"
    # 是否具有积分方法，这里为 False 表示不包含积分解法
    has_integral = False
    # 获取已预处理的常微分方程
    eq = self.ode_problem.eq_preprocessed
    # 获取函数 f(x)
    f = self.ode_problem.func.func
    # 获取方程的阶数
    order = self.ode_problem.order
    # 获取符号变量 x
    x = self.ode_problem.sym
    # 获取线性系数匹配的结果
    match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
    # 初始化结果 r
    self.r = None
    # 默认匹配结果为假
    does_match = False

    # 如果方程的阶数和匹配结果都存在
    if order and match:
        # 获取阶数对应的系数
        coeff = match[order]
        # 计算因子
        factor = x**order / coeff
        # 计算结果字典 r
        self.r = {i: factor*match[i] for i in match}
    
    # 如果结果 r 存在，并且所有非负指数的项均通过测试
    if self.r and all(_test_term(self.r[i], f(x), i) for i in self.r if i >= 0):
        # 如果最后一项为零
        if not self.r[-1]:
            # 则匹配成功
            does_match = True
    
    # 返回匹配结果
    return does_match

def _get_general_solution(self, *, simplify_flag: bool = True):
    # 获取函数对象
    fx = self.ode_problem.func
    # 获取常微分方程
    eq = self.ode_problem.eq
    # 获取齐次解
    homogen_sol = _get_euler_characteristic_eq_sols(eq, fx, self.r)[0]
    # 返回齐次解的列表
    return [homogen_sol]
# 定义一个类，用于解决第n阶线性非齐次Cauchy-Euler均匀方程，使用参数变化法
class NthLinearEulerEqNonhomogeneousVariationOfParameters(SingleODESolver):
    """
    解决第n阶线性非齐次Cauchy-Euler均匀方程，使用参数变化法。

    这种方程的形式为 `g(x) = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x) \cdots`。

    此方法通过假设特解的形式来工作:

    .. math:: \sum_{x=1}^{n} c_i(x) y_i(x) {a_n} {x^n} \text{, }

    其中 `y_i` 是齐次方程的第 `i` 个解。解由Wronskian和Cramer's Rule计算得出。
    特解通过以下方程乘以 `a_n x^{n}` 得到:

    .. math:: \sum_{x=1}^n \left( \int \frac{W_i(x)}{W(x)} \, dx \right) y_i(x) \text{, }

    其中 `W(x)` 是基础系统的Wronskian（齐次方程的 `n` 个线性独立解的系统），
    `W_i(x)` 是基础系统的Wronskian，第 `i` 列用 `[0, 0, \cdots, 0, \frac{x^{- n}}{a_n} g{\left(x \right)}]` 替换。

    此方法足够通用，可以解决任何 `n` 阶非齐次线性微分方程，但有时SymPy无法很好地简化Wronskian以便积分。
    如果此方法出现hang的情况，请尝试使用 ``nth_linear_constant_coeff_variation_of_parameters_Integral`` 提示，
    并手动简化积分。当适用时，建议使用 ``nth_linear_constant_coeff_undetermined_coefficients``，
    因为它不使用积分，速度更快且更可靠。

    警告，使用 'nth_linear_constant_coeff_variation_of_parameters' 和 simplify=False 在
    :py:meth:`~sympy.solvers.ode.dsolve` 可能会导致hang，因为在积分之前不会尝试简化Wronskian。
    建议您仅在这种方法中使用 simplify=False 和
    'nth_linear_constant_coeff_variation_of_parameters_Integral'，特别是如果齐次方程的解中包含三角函数。

    示例
    ========

    >>> from sympy import Function, dsolve, Derivative
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - x**4
    >>> dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_nonhomogeneous_variation_of_parameters').expand()
    Eq(f(x), C1*x + C2*x**2 + x**4/6)

    """
    # 提示信息，指示使用参数变化法解决非齐次Cauchy-Euler方程
    hint = "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters"
    # 标志，表明这个类具有积分的能力
    has_integral = True
    # 获取已预处理的微分方程
    eq = self.ode_problem.eq_preprocessed
    # 获取微分方程对应的函数
    f = self.ode_problem.func.func
    # 获取微分方程的阶数
    order = self.ode_problem.order
    # 获取微分方程的符号变量
    x = self.ode_problem.sym
    # 获取微分方程的线性系数
    match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
    # 初始化结果变量
    self.r = None
    # 初始化匹配结果为假
    does_match = False

    # 如果微分方程有阶数并且存在匹配
    if order and match:
        # 获取匹配结果中阶数对应的系数
        coeff = match[order]
        # 计算因子，用于归一化
        factor = x**order / coeff
        # 根据因子调整匹配结果中的每个系数
        self.r = {i: factor*match[i] for i in match}
    
    # 如果存在结果并且所有非负阶数的系数都通过测试
    if self.r and all(_test_term(self.r[i], f(x), i) for i in self.r if i >= 0):
        # 如果常数项不为零
        if self.r[-1]:
            # 匹配成功
            does_match = True

    # 返回匹配结果
    return does_match

# 获取通解
def _get_general_solution(self, *, simplify_flag: bool = True):
    # 获取未处理的微分方程
    eq = self.ode_problem.eq
    # 获取微分方程对应的函数
    f = self.ode_problem.func.func
    # 获取微分方程的符号变量
    x = self.ode_problem.sym
    # 获取微分方程的阶数
    order = self.ode_problem.order
    # 获取齐次解和特征方程的根
    homogen_sol, roots = _get_euler_characteristic_eq_sols(eq, f(x), self.r)
    # 调整结果中的常数项
    self.r[-1] = self.r[-1] / self.r[order]
    # 求解变分参数法得到的特解
    sol = _solve_variation_of_parameters(eq, f(x), roots, homogen_sol, order, self.r, simplify_flag)

    # 返回微分方程的通解
    return [Eq(f(x), homogen_sol.rhs + (sol.rhs - homogen_sol.rhs) * self.r[order])]
# 定义一个类，用于解决第n阶线性非齐次Cauchy-Euler等次常微分方程，采用未定系数法
class NthLinearEulerEqNonhomogeneousUndeterminedCoefficients(SingleODESolver):
    """
    解决第n阶线性非齐次Cauchy-Euler等次常微分方程，使用未定系数法。

    这种方程的形式为 `g(x) = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x) \cdots`。

    这些方程可以通过替换形如 `x = exp(t)` 的解，并推导出形如 `g(exp(t)) = b_0 f(t) + b_1 f'(t) + b_2 f''(t) \cdots` 的特征方程来通常解决，
    如果 `g(exp(t))` 具有有限个线性独立的导数。

    满足这一要求的函数是形如 `a x^i e^{b x} \sin(c x + d)` 或 `a x^i e^{b x} \cos(c x + d)` 的有限和函数，其中 `i` 是非负整数，
    而 `a`, `b`, `c`, 和 `d` 是常数。例如，任何多项式函数，像 `x^2 e^{2 x}`, `x \sin(x)`, 和 `e^x \cos(x)` 都可以使用。
    `\sin` 和 `\cos` 的乘积具有有限数量的导数，因为它们可以展开为 `\sin(a x)` 和 `\cos(b x)` 项。但是，SymPy 目前无法执行这种展开，
    因此您需要手动将表达式重写为上述形式，以便使用此方法。例如，您需要手动将 `\sin^2(x)` 转换为 `(1 + \cos(2 x))/2` 来正确应用未定系数法。

    在将 x 替换为 exp(t) 后，此方法通过将表达式及其所有线性独立的导数作为试验函数，并将它们代入原始ODE中来工作。每个项的系数将形成一个线性方程组，
    这些方程组将被求解并代入，从而得到解。如果任何试验函数在齐次方程的解上线性相关，则需要乘以足够的 `x` 使它们线性独立。

    Examples
    ========

    >>> from sympy import dsolve, Function, Derivative, log
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - log(x)
    >>> dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients').expand()
    Eq(f(x), C1*x + C2*x**2 + log(x)/2 + 3/4)

    """
    # 提示符号，指示使用非齐次Cauchy-Euler等次常微分方程的未定系数法解法
    hint = "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients"
    # 标志：不具备积分
    has_integral = False
    # 返回该方法是否与ODE问题的线性系数匹配的布尔值
    def _matches(self):
        # 获取高阶自由ODE问题的等式
        eq = self.ode_problem.eq_high_order_free
        # 获取ODE问题的函数
        f = self.ode_problem.func.func
        # 获取ODE问题的阶数
        order = self.ode_problem.order
        # 获取ODE问题的符号变量
        x = self.ode_problem.sym
        # 获取等式的线性系数匹配
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        # 初始化结果变量为None
        self.r = None
        # 初始化匹配状态为False
        does_match = False

        # 如果存在阶数和匹配
        if order and match:
            # 获取指定阶数的系数
            coeff = match[order]
            # 计算因子
            factor = x**order / coeff
            # 计算r，即每个匹配项乘以因子
            self.r = {i: factor*match[i] for i in match}
        
        # 如果r存在且所有大于等于0的项通过测试函数_test_term
        if self.r and all(_test_term(self.r[i], f(x), i) for i in self.r if i >= 0):
            # 如果r的最后一项不为零
            if self.r[-1]:
                # 对r的最后一项进行正化和指数操作
                e, re = posify(self.r[-1].subs(x, exp(x)))
                # 判断是否满足未定系数条件
                undetcoeff = _undetermined_coefficients_match(e.subs(re), x)
                # 如果测试通过，则匹配状态设置为True
                if undetcoeff['test']:
                    does_match = True
        
        # 返回匹配状态
        return does_match

    # 获取常规解的方法
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取ODE问题的函数
        f = self.ode_problem.func.func
        # 获取ODE问题的符号变量
        x = self.ode_problem.sym
        # 初始化字符方程、等式和符号
        chareq, eq, symbol = S.Zero, S.Zero, Dummy('x')
        
        # 遍历r的键
        for i in self.r.keys():
            # 如果键大于等于0
            if i >= 0:
                # 计算字符方程的贡献
                chareq += (self.r[i]*diff(x**symbol, x, i)*x**-symbol).expand()

        # 根据字符方程的次数，计算等式
        for i in range(1, degree(Poly(chareq, symbol))+1):
            eq += chareq.coeff(symbol**i)*diff(f(x), x, i)

        # 如果字符方程的常数项存在
        if chareq.as_coeff_add(symbol)[0]:
            # 添加到等式中
            eq += chareq.as_coeff_add(symbol)[0]*f(x)
        
        # 对r的最后一项进行正化和指数操作
        e, re = posify(self.r[-1].subs(x, exp(x)))
        # 将结果添加到等式中
        eq += e.subs(re)

        # 初始化常系数未定系数ODE问题的实例
        self.const_undet_instance = NthLinearConstantCoeffUndeterminedCoefficients(SingleODEProblem(eq, f(x), x))
        # 获取常规解
        sol = self.const_undet_instance.get_general_solution(simplify=simplify_flag)[0]
        # 将解中的x替换为log(x)
        sol = sol.subs(x, log(x))
        # 将解中的f(log(x))替换为f(x)，并展开
        sol = sol.subs(f(log(x)), f(x)).expand()

        # 返回解的列表
        return [sol]
# 定义 SecondLinearBessel 类，继承自 SingleODESolver 类，用于解决 Bessel 微分方程
class SecondLinearBessel(SingleODESolver):
    # 类的文档字符串，描述解 Bessel 微分方程的解法
    r"""
    Gives solution of the Bessel differential equation

    .. math :: x^2 \frac{d^2y}{dx^2} + x \frac{dy}{dx} y(x) + (x^2-n^2) y(x)

    if `n` is integer then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))`` as both the solutions are linearly independent else if
    `n` is a fraction then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 besselj(-n,x))`` which can also transform into ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))``.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import Symbol
    >>> v = Symbol('v', positive=True)
    >>> from sympy import dsolve, Function
    >>> f = Function('f')
    >>> y = f(x)
    >>> genform = x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - v**2)*y
    >>> dsolve(genform)
    Eq(f(x), C1*besselj(v, x) + C2*bessely(v, x))

    References
    ==========

    https://math24.net/bessel-differential-equation.html

    """
    # 提示信息，指明这是解二阶线性 Bessel 方程的类
    hint = "2nd_linear_bessel"
    # 标志，指示该类没有积分的能力
    has_integral = False
    # 定义一个方法 `_matches`，用于匹配微分方程的特定形式
    def _matches(self):
        # 获取高阶自由常数形式的微分方程
        eq = self.ode_problem.eq_high_order_free
        # 获取微分方程的函数
        f = self.ode_problem.func
        # 获取微分方程的阶数
        order = self.ode_problem.order
        # 获取微分方程的符号变量
        x = self.ode_problem.sym
        # 计算微分方程函数关于符号变量 x 的一阶导数
        df = f.diff(x)
        # 创建排除指定变量的通配符 Wild 对象
        a = Wild('a', exclude=[f, df])
        b = Wild('b', exclude=[x, f, df])
        a4 = Wild('a4', exclude=[x, f, df])
        b4 = Wild('b4', exclude=[x, f, df])
        c4 = Wild('c4', exclude=[x, f, df])
        d4 = Wild('d4', exclude=[x, f, df])
        # 创建排除指定变量和二阶导数的通配符 Wild 对象
        a3 = Wild('a3', exclude=[f, df, f.diff(x, 2)])
        b3 = Wild('b3', exclude=[f, df, f.diff(x, 2)])
        c3 = Wild('c3', exclude=[f, df, f.diff(x, 2)])
        # 构建微分方程的待匹配形式
        deq = a3 * (f.diff(x, 2)) + b3 * df + c3 * f
        # 对微分方程进行模式匹配
        r = collect(eq, [f.diff(x, 2), df, f]).match(deq)
        
        # 如果微分方程阶数为 2 并且匹配成功
        if order == 2 and r:
            # 检查所有匹配项是否都是多项式
            if not all(r[key].is_polynomial() for key in r):
                # 如果不是多项式，将方程转化为多项式形式
                n, d = eq.as_numer_denom()
                eq = expand(n)
                r = collect(eq, [f.diff(x, 2), df, f]).match(deq)

        # 如果匹配成功且 a3 不等于 0
        if r and r[a3] != 0:
            # 提取 f(x).diff(x, 2) 的主导系数
            coeff = factor(r[a3]).match(a4 * (x - b) ** b4)
            
            # 如果成功提取主导系数
            if coeff:
                # 如果主导系数 b4 为 0，则表示常数系数
                if coeff[b4] == 0:
                    return False
                # 提取点的位置
                point = coeff[b]
            else:
                return False
            
            # 如果存在点的位置
            if point:
                # 对 r[a3], r[b3], r[c3] 进行点移动变换
                r[a3] = simplify(r[a3].subs(x, x + point))
                r[b3] = simplify(r[b3].subs(x, x + point))
                r[c3] = simplify(r[c3].subs(x, x + point))
            
            # 将 a3 转化为 x**2 的形式
            r[a3] = cancel(r[a3] / (coeff[a4] * (x) ** (-2 + coeff[b4])))
            r[b3] = cancel(r[b3] / (coeff[a4] * (x) ** (-2 + coeff[b4])))
            r[c3] = cancel(r[c3] / (coeff[a4] * (x) ** (-2 + coeff[b4])))
            
            # 检查 b3 是否为 c*(x-b) 的形式
            coeff1 = factor(r[b3]).match(a4 * (x))
            if coeff1 is None:
                return False
            
            # 对 c3 进行简化检查，匹配 a - b 的形式
            _coeff2 = expand(r[c3]).match(a - b)
            if _coeff2 is None:
                return False
            
            # 匹配 c3 的标准形式
            coeff2 = factor(_coeff2[a]).match(c4 ** 2 * (x) ** (2 * a4))
            if coeff2 is None:
                return False
            
            # 如果 _coeff2[b] 为 0，设置 coeff2[d4] 为 0
            if _coeff2[b] == 0:
                coeff2[d4] = 0
            else:
                coeff2[d4] = factor(_coeff2[b]).match(d4 ** 2)[d4]
            
            # 设置匹配结果字典
            self.rn = {'n': coeff2[d4], 'a4': coeff2[c4], 'd4': coeff2[a4]}
            self.rn['c4'] = coeff1[a4]
            self.rn['b4'] = point
            return True
        
        # 匹配失败，返回 False
        return False
    # 定义一个方法来获取常规解，接受一个关键字参数 simplify_flag 用于指示是否简化
    def _get_general_solution(self, *, simplify_flag: bool = True):
        # 获取微分方程问题的函数对象
        f = self.ode_problem.func.func
        # 获取微分方程问题的符号变量
        x = self.ode_problem.sym
        # 获取属性 self.rn 中的参数 'n' 并赋值给变量 n
        n = self.rn['n']
        # 获取属性 self.rn 中的参数 'a4' 并赋值给变量 a4
        a4 = self.rn['a4']
        # 获取属性 self.rn 中的参数 'c4' 并赋值给变量 c4
        c4 = self.rn['c4']
        # 获取属性 self.rn 中的参数 'd4' 并赋值给变量 d4
        d4 = self.rn['d4']
        # 获取属性 self.rn 中的参数 'b4' 并赋值给变量 b4
        b4 = self.rn['b4']
        # 计算新的 n 值，使用 sqrt 函数和 Rational 类型
        n = sqrt(n**2 + Rational(1, 4)*(c4 - 1)**2)
        # 调用 self.ode_problem 的 get_numbered_constants 方法，获取两个编号的常数 C1 和 C2
        (C1, C2) = self.ode_problem.get_numbered_constants(num=2)
        # 返回一个列表，包含微分方程 f(x) 的等式表达式
        return [Eq(f(x), ((x**(Rational(1-c4,2)))*(C1*besselj(n/d4,a4*x**d4/d4)
            + C2*bessely(n/d4,a4*x**d4/d4))).subs(x, x-b4))]
class LieGroup(SingleODESolver):
    r"""
    This hint implements the Lie group method of solving first order differential
    equations. The aim is to convert the given differential equation from the
    given coordinate system into another coordinate system where it becomes
    invariant under the one-parameter Lie group of translations. The converted
    ODE can be easily solved by quadrature. It makes use of the
    :py:meth:`sympy.solvers.ode.infinitesimals` function which returns the
    infinitesimals of the transformation.

    The coordinates `r` and `s` can be found by solving the following Partial
    Differential Equations.

    .. math :: \xi\frac{\partial r}{\partial x} + \eta\frac{\partial r}{\partial y}
                  = 0

    .. math :: \xi\frac{\partial s}{\partial x} + \eta\frac{\partial s}{\partial y}
                  = 1

    The differential equation becomes separable in the new coordinate system

    .. math :: \frac{ds}{dr} = \frac{\frac{\partial s}{\partial x} +
                 h(x, y)\frac{\partial s}{\partial y}}{
                 \frac{\partial r}{\partial x} + h(x, y)\frac{\partial r}{\partial y}}

    After finding the solution by integration, it is then converted back to the original
    """
    """
    This class defines a solver for Ordinary Differential Equations (ODEs) using
    the Lie group method. It includes methods to check for additional parameters,
    match the differential equation, and retrieve general solutions.

    Attributes:
    hint : str
        A string indicating the method hint, in this case, "lie_group".
    has_integral : bool
        Flag indicating if an integral solution exists, initialized as False.

    Methods:
    _has_additional_params(self)
        Checks if additional parameters 'xi' and 'eta' are present in the ODE problem.
    
    _matches(self)
        Attempts to match the given ODE with a specific pattern involving 'd' and 'e'.
        Updates internal attributes if a match is found.
    
    _get_general_solution(self, *, simplify_flag: bool = True)
        Retrieves the general solution of the ODE using the Lie group method.
        Raises NotImplementedError if no solution is found.

    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14
    """
    hint = "lie_group"
    has_integral = False

    def _has_additional_params(self):
        return 'xi' in self.ode_problem.params and 'eta' in self.ode_problem.params

    def _matches(self):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f(x).diff(x)
        y = Dummy('y')
        d = Wild('d', exclude=[df, f(x).diff(x, 2)])
        e = Wild('e', exclude=[df])
        does_match = False
        if self._has_additional_params() and order == 1:
            xi = self.ode_problem.params['xi']
            eta = self.ode_problem.params['eta']
            self.r3 = {'xi': xi, 'eta': eta}
            r = collect(eq, df, exact=True).match(d + e * df)
            if r:
                r['d'] = d
                r['e'] = e
                r['y'] = y
                r[d] = r[d].subs(f(x), y)
                r[e] = r[e].subs(f(x), y)
                self.r3.update(r)
            does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        x = self.ode_problem.sym
        func = self.ode_problem.func
        order = self.ode_problem.order
        df = func.diff(x)

        try:
            eqsol = solve(eq, df)
        except NotImplementedError:
            eqsol = []

        desols = []
        for s in eqsol:
            sol = _ode_lie_group(s, func, order, match=self.r3)
            if sol:
                desols.extend(sol)

        if desols == []:
            raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                + " the lie group method")
        return desols
# 定义一个映射，将字符串映射到对应的类
solver_map = {
    'factorable': Factorable,  # 将字符串 'factorable' 映射到 Factorable 类
    'nth_linear_constant_coeff_homogeneous': NthLinearConstantCoeffHomogeneous,  # 将字符串映射到 NthLinearConstantCoeffHomogeneous 类
    'nth_linear_euler_eq_homogeneous': NthLinearEulerEqHomogeneous,  # 将字符串映射到 NthLinearEulerEqHomogeneous 类
    'nth_linear_constant_coeff_undetermined_coefficients': NthLinearConstantCoeffUndeterminedCoefficients,  # 将字符串映射到 NthLinearConstantCoeffUndeterminedCoefficients 类
    'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients': NthLinearEulerEqNonhomogeneousUndeterminedCoefficients,  # 将字符串映射到 NthLinearEulerEqNonhomogeneousUndeterminedCoefficients 类
    'separable': Separable,  # 将字符串 'separable' 映射到 Separable 类
    '1st_exact': FirstExact,  # 将字符串 '1st_exact' 映射到 FirstExact 类
    '1st_linear': FirstLinear,  # 将字符串 '1st_linear' 映射到 FirstLinear 类
    'Bernoulli': Bernoulli,  # 将字符串 'Bernoulli' 映射到 Bernoulli 类
    'Riccati_special_minus2': RiccatiSpecial,  # 将字符串 'Riccati_special_minus2' 映射到 RiccatiSpecial 类
    '1st_rational_riccati': RationalRiccati,  # 将字符串 '1st_rational_riccati' 映射到 RationalRiccati 类
    '1st_homogeneous_coeff_best': HomogeneousCoeffBest,  # 将字符串 '1st_homogeneous_coeff_best' 映射到 HomogeneousCoeffBest 类
    '1st_homogeneous_coeff_subs_indep_div_dep': HomogeneousCoeffSubsIndepDivDep,  # 将字符串映射到 HomogeneousCoeffSubsIndepDivDep 类
    '1st_homogeneous_coeff_subs_dep_div_indep': HomogeneousCoeffSubsDepDivIndep,  # 将字符串映射到 HomogeneousCoeffSubsDepDivIndep 类
    'almost_linear': AlmostLinear,  # 将字符串 'almost_linear' 映射到 AlmostLinear 类
    'linear_coefficients': LinearCoefficients,  # 将字符串 'linear_coefficients' 映射到 LinearCoefficients 类
    'separable_reduced': SeparableReduced,  # 将字符串 'separable_reduced' 映射到 SeparableReduced 类
    'nth_linear_constant_coeff_variation_of_parameters': NthLinearConstantCoeffVariationOfParameters,  # 将字符串映射到 NthLinearConstantCoeffVariationOfParameters 类
    'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters': NthLinearEulerEqNonhomogeneousVariationOfParameters,  # 将字符串映射到 NthLinearEulerEqNonhomogeneousVariationOfParameters 类
    'Liouville': Liouville,  # 将字符串 'Liouville' 映射到 Liouville 类
    '2nd_linear_airy': SecondLinearAiry,  # 将字符串映射到 SecondLinearAiry 类
    '2nd_linear_bessel': SecondLinearBessel,  # 将字符串映射到 SecondLinearBessel 类
    '2nd_hypergeometric': SecondHypergeometric,  # 将字符串映射到 SecondHypergeometric 类
    'nth_order_reducible': NthOrderReducible,  # 将字符串映射到 NthOrderReducible 类
    '2nd_nonlinear_autonomous_conserved': SecondNonlinearAutonomousConserved,  # 将字符串映射到 SecondNonlinearAutonomousConserved 类
    'nth_algebraic': NthAlgebraic,  # 将字符串映射到 NthAlgebraic 类
    'lie_group': LieGroup,  # 将字符串 'lie_group' 映射到 LieGroup 类
}

# 避免循环导入：
# 导入必要的模块或函数，这些函数可能在当前模块中用到
from .ode import dsolve, ode_sol_simplicity, odesimp, homogeneous_order
```