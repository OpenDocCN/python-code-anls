# `D:\src\scipysrc\sympy\sympy\solvers\ode\systems.py`

```
from sympy.core import Add, Mul, S  # 导入加法、乘法和S类（表示符号）
from sympy.core.containers import Tuple  # 导入元组容器
from sympy.core.exprtools import factor_terms  # 导入表达式工具中的因子项函数
from sympy.core.numbers import I  # 导入虚数单位I
from sympy.core.relational import Eq, Equality  # 导入等式和关系等式类
from sympy.core.sorting import default_sort_key, ordered  # 导入默认排序键和有序函数
from sympy.core.symbol import Dummy, Symbol  # 导入虚拟符号和符号类
from sympy.core.function import (expand_mul, expand, Derivative,  # 导入函数类和相关函数
                                 AppliedUndef, Function, Subs)
from sympy.functions import (exp, im, cos, sin, re, Piecewise,  # 导入指数、虚部、余弦、正弦、实部、分段函数等函数
                             piecewise_fold, sqrt, log)
from sympy.functions.combinatorial.factorials import factorial  # 导入阶乘函数
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye  # 导入矩阵和相关异常
from sympy.polys import Poly, together  # 导入多项式和合并函数
from sympy.simplify import collect, radsimp, signsimp  # 导入收集、有理化简和符号简化函数
from sympy.simplify.powsimp import powdenest, powsimp  # 导入幂简化函数
from sympy.simplify.ratsimp import ratsimp  # 导入有理化简函数
from sympy.simplify.simplify import simplify  # 导入简化函数
from sympy.sets.sets import FiniteSet  # 导入有限集类
from sympy.solvers.deutils import ode_order  # 导入ODE方程阶数计算函数
from sympy.solvers.solveset import NonlinearError, solveset  # 导入非线性方程求解相关异常和函数
from sympy.utilities.iterables import (connected_components, iterable,  # 导入可迭代工具函数
                                       strongly_connected_components)
from sympy.utilities.misc import filldedent  # 导入填充去除缩进的工具函数
from sympy.integrals.integrals import Integral, integrate  # 导入积分类和积分函数


def _get_func_order(eqs, funcs):
    """Return the maximum order of ODEs in eqs for each function in funcs."""
    return {func: max(ode_order(eq, func) for eq in eqs) for func in funcs}


class ODEOrderError(ValueError):
    """Raised by linear_ode_to_matrix if the system has the wrong order."""
    pass


class ODENonlinearError(NonlinearError):
    """Raised by linear_ode_to_matrix if the system is nonlinear."""
    pass


def _simpsol(soleq):
    """Simplify the solution equation soleq."""
    lhs = soleq.lhs  # 获取方程的左侧
    sol = soleq.rhs  # 获取方程的右侧
    sol = powsimp(sol)  # 幂简化右侧解
    gens = list(sol.atoms(exp))  # 获取解中的指数函数列表
    p = Poly(sol, *gens, expand=False)  # 使用给定的生成器创建多项式对象，不展开
    gens = [factor_terms(g) for g in gens]  # 对生成器进行因子项处理
    if not gens:
        gens = p.gens  # 如果没有生成器，则使用多项式的生成器
    syms = [Symbol('C1'), Symbol('C2')]  # 定义两个常数符号
    terms = []
    for coeff, monom in zip(p.coeffs(), p.monoms()):
        coeff = piecewise_fold(coeff)  # 对系数进行分段函数折叠
        if isinstance(coeff, Piecewise):
            coeff = Piecewise(*((ratsimp(coef).collect(syms), cond) for coef, cond in coeff.args))
        else:
            coeff = ratsimp(coeff).collect(syms)  # 对系数进行有理化简和常数收集
        monom = Mul(*(g ** i for g, i in zip(gens, monom)))  # 构造单项式
        terms.append(coeff * monom)  # 添加到项列表中
    return Eq(lhs, Add(*terms))  # 返回简化后的方程式


def _solsimp(e, t):
    """Simplify the expression e with respect to the independent variable t."""
    no_t, has_t = powsimp(expand_mul(e)).as_independent(t)  # 将表达式分解为不含t的部分和含t的部分

    no_t = ratsimp(no_t)  # 对不含t的部分进行有理化简
    has_t = has_t.replace(exp, lambda a: exp(factor_terms(a)))  # 对含t的部分进行指数因子项处理

    return no_t + has_t  # 返回简化后的表达式


def simpsol(sol, wrt1, wrt2, doit=True):
    """Simplify solutions from dsolve_system.

    Parameters:
    sol -- The solution as returned by dsolve (list of Eq).
    wrt1 -- List of symbols to be collected first.
    wrt2 -- List of symbols to be collected after wrt1.
    doit -- Optional flag to indicate if simplification should be applied immediately.
    """
    pass  # 此函数目前没有具体实现，仅定义了文档字符串和一个占位符
    # 定义函数 simprhs，用于简化常微分方程（ODE）解的右手边（rhs）部分
    def simprhs(rhs, rep, wrt1, wrt2):
        """Simplify the rhs of an ODE solution"""
        # 如果有替代参数 rep，则进行替换操作
        if rep:
            rhs = rhs.subs(rep)
        # 对 rhs 进行因子项化简
        rhs = factor_terms(rhs)
        # 对 rhs 进行 coeff_dep 简化，其中 wrt1 是变量，wrt2 是未知项
        rhs = simp_coeff_dep(rhs, wrt1, wrt2)
        # 使用 signsimp 函数对 rhs 进行符号化简
        rhs = signsimp(rhs)
        # 返回简化后的 rhs 结果
        return rhs
    def simp_coeff_dep(expr, wrt1, wrt2=None):
        """将右侧表达式分解为项，将项分解为依赖项和系数，并在依赖项上进行收集"""
        # 判断一个表达式是否为加法且包含 wrt1
        add_dep_terms = lambda e: e.is_Add and e.has(*wrt1)
        # 判断一个表达式是否为乘法且至少有一个因子满足 add_dep_terms
        expandable = lambda e: e.is_Mul and any(map(add_dep_terms, e.args))
        # 将可展开的表达式扩展为简化的乘积形式
        expand_func = lambda e: expand_mul(e, deep=False)
        # 替换表达式中可展开的部分
        expand_mul_mod = lambda e: e.replace(expandable, expand_func)
        # 将表达式分解为加法操作数
        terms = Add.make_args(expand_mul_mod(expr))
        # 初始化一个空字典
        dc = {}
        # 遍历每一个分解后的项
        for term in terms:
            # 将项分解为系数和依赖项
            coeff, dep = term.as_independent(*wrt1, as_Add=False)
            # 根据 simpdep 函数对依赖项进行标准化处理
            dep = simpdep(dep, wrt1)

            # 检查依赖项是否为常数
            if dep is not S.One:
                # 对依赖项进行因式分解
                dep2 = factor_terms(dep)
                # 如果依赖项不再包含 wrt1，将系数乘以 dep2
                if not dep2.has(*wrt1):
                    coeff *= dep2
                    dep = S.One

            # 将系数添加到字典中对应的依赖项中
            if dep not in dc:
                dc[dep] = coeff
            else:
                dc[dep] += coeff
        
        # 对系数应用递归方法，但这次是在 wrt2 上进行收集而不是 wrt1
        termpairs = ((simpcoeff(c, wrt2), d) for d, c in dc.items())
        if wrt2 is not None:
            termpairs = ((simp_coeff_dep(c, wrt2), d) for c, d in termpairs)
        # 返回收集后的加法表达式
        return Add(*(c * d for c, d in termpairs))

    def simpdep(term, wrt1):
        """使用 powsimp 对包含 t 的因子进行标准化，并重新组合 exp 函数"""
        def canonicalise(a):
            # 使用 factor_terms 并不完全正确，因为会产生我们不想要的表达式，例如 exp(t*(1+t))
            # 我们需要取消因子并提取公共分母，但理想情况下，分子应以 t 的标准形式多项式表示，因此我们在展开乘法后进行收集。
            a = factor_terms(a)
            num, den = a.as_numer_denom()
            num = expand_mul(num)
            num = collect(num, wrt1)
            return num / den

        # 对 term 使用 powsimp 函数
        term = powsimp(term)
        # 构建一个替换字典，用于将 exp 函数替换为 canonicalise 处理后的表达式
        rep = {e: exp(canonicalise(e.args[0])) for e in term.atoms(exp)}
        # 使用替换字典对 term 进行替换
        term = term.subs(rep)
        # 返回处理后的 term
        return term
    # 定义一个函数，用于对给定的系数进行简化并与 ratsimp 函数取消分数
    def simpcoeff(coeff, wrt2):
        """Bring to a common fraction and cancel with ratsimp"""
        # 将系数合并成一个公共分数
        coeff = together(coeff)
        
        # 如果系数是多项式，尝试通过 ratsimp 简化
        if coeff.is_polynomial():
            # 调用 ratsimp 可能会消耗较多时间。主要用于简化具有无理数分母的项的和，
            # 因此我们限制在表达式对任何符号都是多项式的情况下使用。也许有更好的方法...
            coeff = ratsimp(radsimp(coeff))
        
        # 首先在次要变量上收集，并在剩余符号上进行收集
        if wrt2 is not None:
            syms = list(wrt2) + list(ordered(coeff.free_symbols - set(wrt2)))
        else:
            syms = list(ordered(coeff.free_symbols))
        
        # 根据给定的符号进行收集
        coeff = collect(coeff, syms)
        
        # 再次尝试将系数合并成一个公共分数
        coeff = together(coeff)
        
        # 返回简化后的系数
        return coeff

    # 如果 doit 为真，收集唯一的积分表达式并对每个进行求值，然后将其代入最终结果中，
    # 替换解方程组中的每个解中的所有出现
    if doit:
        # 使用集合 union 方法收集所有解 sol 中的积分表达式
        integrals = set().union(*(s.atoms(Integral) for s in sol))
        # 对每个积分表达式进行因子项化和求值，并将结果存入 rep 字典中
        rep = {i: factor_terms(i).doit() for i in integrals}
    else:
        # 如果 doit 不为真，则 rep 字典为空
        rep = {}

    # 对每个解方程 s 中的右侧简化，并用 rep 字典中的值替换其中的积分表达式，最后构造成等式形式
    sol = [Eq(s.lhs, simprhs(s.rhs, rep, wrt1, wrt2)) for s in sol]
    
    # 返回简化后的解方程组
    return sol
def linodesolve_type(A, t, b=None):
    r"""
    Helper function that determines the type of the system of ODEs for solving with :obj:`sympy.solvers.ode.systems.linodesolve()`

    Explanation
    ===========

    This function takes in the coefficient matrix and/or the non-homogeneous term
    and returns the type of the equation that can be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`.

    If the system is constant coefficient homogeneous, then "type1" is returned

    If the system is constant coefficient non-homogeneous, then "type2" is returned

    If the system is non-constant coefficient homogeneous, then "type3" is returned

    If the system is non-constant coefficient non-homogeneous, then "type4" is returned

    If the system has a non-constant coefficient matrix which can be factorized into constant
    coefficient matrix, then "type5" or "type6" is returned for when the system is homogeneous or
    non-homogeneous respectively.

    Note that, if the system of ODEs is of "type3" or "type4", then along with the type,
    the commutative antiderivative of the coefficient matrix is also returned.

    If the system cannot be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`, then
    NotImplementedError is raised.

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of ODEs
    b : Matrix or None
        Non-homogeneous term of the system. The default value is None.
        If this argument is None, then the system is assumed to be homogeneous.

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import linodesolve_type
    >>> t = symbols("t")
    >>> A = Matrix([[1, 1], [2, 3]])
    >>> b = Matrix([t, 1])

    >>> linodesolve_type(A, t)
    {'antiderivative': None, 'type_of_equation': 'type1'}

    >>> linodesolve_type(A, t, b=b)
    {'antiderivative': None, 'type_of_equation': 'type2'}

    >>> A_t = Matrix([[1, t], [-t, 1]])

    >>> linodesolve_type(A_t, t)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type3'}

    >>> linodesolve_type(A_t, t, b=b)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type4'}

    >>> A_non_commutative = Matrix([[1, t], [t, -1]])
    >>> linodesolve_type(A_non_commutative, t)
    Traceback (most recent call last):
    ...
    NotImplementedError:
    The system does not have a commutative antiderivative, it cannot be
    solved by linodesolve.

    Returns
    =======

    Dict

    Raises
    ======

    NotImplementedError
        When the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linodesolve: Function for which linodesolve_type gets the information

    """

    match = {}  # 创建一个空字典，用于存储匹配类型和反导数
    is_non_constant = not _matrix_is_constant(A, t)  # 检查系数矩阵是否为非常数
    is_non_homogeneous = not (b is None or b.is_zero_matrix)  # 检查是否存在非齐次项
    # 根据输入的 is_non_constant 和 is_non_homogeneous 构造一个字符串，表示方程类型
    type = "type{}".format(int("{}{}".format(int(is_non_constant), int(is_non_homogeneous)), 2) + 1)

    # 初始化变量 B 为 None
    B = None

    # 更新 match 字典，添加方程类型和 B 的反导数信息
    match.update({"type_of_equation": type, "antiderivative": B})

    # 如果方程不是常数系数方程
    if is_non_constant:
        # 调用 _is_commutative_anti_derivative 函数计算反导数 B，并检查是否可交换
        B, is_commuting = _is_commutative_anti_derivative(A, t)
        
        # 如果不可交换，抛出 NotImplementedError 异常
        if not is_commuting:
            raise NotImplementedError(filldedent('''
                The system does not have a commutative antiderivative, it cannot be solved
                by linodesolve.
            '''))

        # 更新 match 字典，设置反导数 B 和一阶类型5或6的替换结果
        match['antiderivative'] = B
        match.update(_first_order_type5_6_subs(A, t, b=b))

    # 返回更新后的 match 字典
    return match
def _first_order_type5_6_subs(A, t, b=None):
    # 初始化一个空字典用于存储匹配结果
    match = {}

    # 调用_factor_matrix函数，计算给定矩阵A和变量t的因子项
    factor_terms = _factor_matrix(A, t)
    
    # 检查是否是齐次方程，即b是否为None或者零矩阵
    is_homogeneous = b is None or b.is_zero_matrix

    # 如果因子项不为None，则继续处理
    if factor_terms is not None:
        # 定义新的符号t_，表示对t的积分
        t_ = Symbol("{}_".format(t))
        # 对第一个因子项积分，得到F_t
        F_t = integrate(factor_terms[0], t)
        # 解方程 t_ = F_t，找到其反函数
        inverse = solveset(Eq(t_, F_t), t)

        # 如果inverse是有限集合且不包含Piecewise，且只有一个解
        if isinstance(inverse, FiniteSet) and not inverse.has(Piecewise) \
            and len(inverse) == 1:

            # 更新A为第二个因子项
            A = factor_terms[1]
            # 如果b不是齐次的，则将其除以第一个因子项，并在其中替换t为inverse的解
            if not is_homogeneous:
                b = b / factor_terms[0]
                b = b.subs(t, list(inverse)[0])
            
            # 确定方程的类型，以字符串形式存储在type中
            type = "type{}".format(5 + (not is_homogeneous))
            # 更新匹配结果字典
            match.update({'func_coeff': A, 'tau': F_t,
                          't_': t_, 'type_of_equation': type, 'rhs': b})

    # 返回最终的匹配结果字典
    return match


def linear_ode_to_matrix(eqs, funcs, t, order):
    r"""
    Convert a linear system of ODEs to matrix form

    Explanation
    ===========

    Express a system of linear ordinary differential equations as a single
    matrix differential equation [1]. For example the system $x' = x + y + 1$
    and $y' = x - y$ can be represented as

    .. math:: A_1 X' = A_0 X + b

    where $A_1$ and $A_0$ are $2 \times 2$ matrices and $b$, $X$ and $X'$ are
    $2 \times 1$ matrices with $X = [x, y]^T$.

    Higher-order systems are represented with additional matrices e.g. a
    second-order system would look like

    .. math:: A_2 X'' =  A_1 X' + A_0 X  + b

    Examples
    ========

    >>> from sympy import Function, Symbol, Matrix, Eq
    >>> from sympy.solvers.ode.systems import linear_ode_to_matrix
    >>> t = Symbol('t')
    >>> x = Function('x')
    >>> y = Function('y')

    We can create a system of linear ODEs like

    >>> eqs = [
    ...     Eq(x(t).diff(t), x(t) + y(t) + 1),
    ...     Eq(y(t).diff(t), x(t) - y(t)),
    ... ]
    >>> funcs = [x(t), y(t)]
    >>> order = 1 # 1st order system

    Now ``linear_ode_to_matrix`` can represent this as a matrix
    differential equation.

    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, order)
    >>> A1
    Matrix([
    [1, 0],
    [0, 1]])
    >>> A0
    Matrix([
    [1, 1],
    [1,  -1]])
    >>> b
    Matrix([
    [1],
    [0]])

    The original equations can be recovered from these matrices:

    >>> eqs_mat = Matrix([eq.lhs - eq.rhs for eq in eqs])
    >>> X = Matrix(funcs)
    >>> A1 * X.diff(t) - A0 * X - b == eqs_mat
    True

    If the system of equations has a maximum order greater than the
    order of the system specified, a ODEOrderError exception is raised.

    >>> eqs = [Eq(x(t).diff(t, 2), x(t).diff(t) + x(t)), Eq(y(t).diff(t), y(t) + x(t))]
    >>> linear_ode_to_matrix(eqs, funcs, t, 1)
    Traceback (most recent call last):
    ...
    ODEOrderError: Cannot represent system in 1-order form

    If the system of equations is nonlinear, then ODENonlinearError is
    raised.
    # 定义一个包含微分方程的列表，每个方程表示为 SymPy 表达式或等式
    eqs = [Eq(x(t).diff(t), x(t) + y(t)), Eq(y(t).diff(t), y(t)**2 + x(t))]
    # 调用函数 linear_ode_to_matrix 处理这些方程，以转换为矩阵形式
    # 如果方程系统非线性，会抛出 ODENonlinearError 异常
    linear_ode_to_matrix(eqs, funcs, t, 1)

    Parameters
    ==========

    eqs : list of SymPy expressions or equalities
        表达式列表，表示为零的微分方程
    funcs : list of applied functions
        微分方程的因变量列表
    t : symbol
        自变量
    order : int
        微分方程系统的阶数

    Returns
    =======

    返回一个元组 ``(As, b)``
    - ``As`` 是矩阵元组，表示微分方程的系数矩阵
    - ``b`` 是右侧常数矩阵，表示方程的右侧

    Raises
    ======

    ODEOrderError
        当微分方程系统的阶数高于指定的阶数时
    ODENonlinearError
        当微分方程系统是非线性的时候

    See Also
    ========

    linear_eq_to_matrix: 用于线性代数方程组的函数转换

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_differential_equation

    """
    # 导入 linear_eq_to_matrix 函数来处理线性方程组
    from sympy.solvers.solveset import linear_eq_to_matrix

    # 检查是否有任何方程的阶数超过指定的阶数 order
    if any(ode_order(eq, func) > order for eq in eqs for func in funcs):
        msg = "Cannot represent system in {}-order form"
        # 抛出 ODEOrderError 异常，指示无法按指定阶数表示系统
        raise ODEOrderError(msg.format(order))

    # 初始化一个空列表用于存储系数矩阵
    As = []

    # 从最高导数开始向下处理
    for o in range(order, -1, -1):
        # 获取当前阶数 o 下的所有变量（导数）
        syms = [func.diff(t, o) for func in funcs]

        try:
            # 将方程组转换为矩阵形式 Ai, b
            Ai, b = linear_eq_to_matrix(eqs, syms)
        except NonlinearError:
            # 如果方程组非线性，抛出 ODENonlinearError 异常
            raise ODENonlinearError("The system of ODEs is nonlinear.")

        # 对 Ai 中的每个元素应用 expand_mul 函数
        Ai = Ai.applyfunc(expand_mul)

        # 将 Ai 添加到 As 列表中，如果当前阶数 o 等于指定阶数 order，则添加 Ai，否则添加 -Ai
        As.append(Ai if o == order else -Ai)

        # 如果当前阶数不为零，更新方程组 eqs 为 b 的相反数列表，否则将 b 赋值给 rhs
        if o:
            eqs = [-eq for eq in b]
        else:
            rhs = b

    # 返回系数矩阵列表 As 和右侧常数矩阵 rhs
    return As, rhs
def matrix_exp(A, t):
    r"""
    Matrix exponential $\exp(A*t)$ for the matrix ``A`` and scalar ``t``.

    Explanation
    ===========

    This functions returns the $\exp(A*t)$ by doing a simple
    matrix multiplication:

    .. math:: \exp(A*t) = P * expJ * P^{-1}

    where $expJ$ is $\exp(J*t)$. $J$ is the Jordan normal
    form of $A$ and $P$ is matrix such that:

    .. math:: A = P * J * P^{-1}

    The matrix exponential $\exp(A*t)$ appears in the solution of linear
    differential equations. For example if $x$ is a vector and $A$ is a matrix
    then the initial value problem

    .. math:: \frac{dx(t)}{dt} = A \times x(t),   x(0) = x0

    has the unique solution

    .. math:: x(t) = \exp(A t) x0

    Examples
    ========

    >>> from sympy import Symbol, Matrix, pprint
    >>> from sympy.solvers.ode.systems import matrix_exp
    >>> t = Symbol('t')

    We will consider a 2x2 matrix for comupting the exponential

    >>> A = Matrix([[2, -5], [2, -4]])
    >>> pprint(A)
    [2  -5]
    [     ]
    [2  -4]

    Now, exp(A*t) is given as follows:

    >>> pprint(matrix_exp(A, t))
    [   -t           -t                    -t              ]
    [3*e  *sin(t) + e  *cos(t)         -5*e  *sin(t)       ]
    [                                                      ]
    [         -t                     -t           -t       ]
    [      2*e  *sin(t)         - 3*e  *sin(t) + e  *cos(t)]

    Parameters
    ==========

    A : Matrix
        The matrix $A$ in the expression $\exp(A*t)$
    t : Symbol
        The independent variable

    See Also
    ========

    matrix_exp_jordan_form: For exponential of Jordan normal form

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jordan_normal_form
    .. [2] https://en.wikipedia.org/wiki/Matrix_exponential

    """
    # Compute the Jordan normal form and exponential of the Jordan form
    P, expJ = matrix_exp_jordan_form(A, t)
    # Return the matrix exponential exp(A*t) using the formula P * expJ * P^{-1}
    return P * expJ * P.inv()


def matrix_exp_jordan_form(A, t):
    r"""
    Matrix exponential $\exp(A*t)$ for the matrix *A* and scalar *t*.

    Explanation
    ===========

    Returns the Jordan form of the $\exp(A*t)$ along with the matrix $P$ such that:

    .. math::
        \exp(A*t) = P * expJ * P^{-1}

    Examples
    ========

    >>> from sympy import Matrix, Symbol
    >>> from sympy.solvers.ode.systems import matrix_exp, matrix_exp_jordan_form
    >>> t = Symbol('t')

    We will consider a 2x2 defective matrix. This shows that our method
    works even for defective matrices.

    >>> A = Matrix([[1, 1], [0, 1]])

    It can be observed that this function gives us the Jordan normal form
    and the required invertible matrix P.

    >>> P, expJ = matrix_exp_jordan_form(A, t)

    Here, it is shown that P and expJ returned by this function is correct
    as they satisfy the formula: P * expJ * P_inverse = exp(A*t).

    >>> P * expJ * P.inv() == matrix_exp(A, t)
    True

    Parameters
    ==========

    A : Matrix
        The matrix $A$ in the expression $\exp(A*t)$

    """
    # Implementation computes the Jordan normal form and its matrix exponential
    # Return both the Jordan normal form matrix and its exponential
    pass
    t : Symbol
        The independent variable

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Defective_matrix
    .. [2] https://en.wikipedia.org/wiki/Jordan_matrix
    .. [3] https://en.wikipedia.org/wiki/Jordan_normal_form

    """

    # 获取矩阵 A 的形状信息
    N, M = A.shape
    # 检查矩阵 A 是否为方阵，若不是则抛出异常
    if N != M:
        raise ValueError('Needed square matrix but got shape (%s, %s)' % (N, M))
    # 检查矩阵 A 是否依赖于变量 t，若是则抛出异常
    elif A.has(t):
        raise ValueError('Matrix A should not depend on t')

    def jordan_chains(A):
        '''Chains from Jordan normal form analogous to M.eigenvects().
        Returns a dict with eignevalues as keys like:
            {e1: [[v111,v112,...], [v121, v122,...]], e2:...}
        where vijk is the kth vector in the jth chain for eigenvalue i.
        '''
        # 计算矩阵 A 的 Jordan 形式
        P, blocks = A.jordan_cells()
        # 构建 Jordan 链的基础向量列表
        basis = [P[:,i] for i in range(P.shape[1])]
        n = 0
        chains = {}
        # 遍历每个 Jordan 块
        for b in blocks:
            eigval = b[0, 0]
            size = b.shape[0]
            # 将基础向量按照对应的特征值组织成字典形式的链
            if eigval not in chains:
                chains[eigval] = []
            chains[eigval].append(basis[n:n+size])
            n += size
        return chains

    # 获取矩阵 A 的 Jordan 链
    eigenchains = jordan_chains(A)

    # 为了在不同 Python 版本间保持一致性而需要的迭代器
    eigenchains_iter = sorted(eigenchains.items(), key=default_sort_key)
    # 检查矩阵 A 是否不含有单位矩阵 I 的情况
    isreal = not A.has(I)

    blocks = []
    vectors = []
    seen_conjugate = set()
    # 遍历排序后的特征值和其对应的 Jordan 链
    for e, chains in eigenchains_iter:
        for chain in chains:
            n = len(chain)
            # 如果是实数域且特征值 e 和其共轭不相等且共轭存在于特征值中
            if isreal and e != e.conjugate() and e.conjugate() in eigenchains:
                # 如果已经见过特征值 e 的共轭，则跳过
                if e in seen_conjugate:
                    continue
                seen_conjugate.add(e.conjugate())
                # 计算指数项和三角函数项
                exprt = exp(re(e) * t)
                imrt = im(e) * t
                imblock = Matrix([[cos(imrt), sin(imrt)],
                                  [-sin(imrt), cos(imrt)]])
                expJblock2 = Matrix(n, n, lambda i,j:
                        imblock * t**(j-i) / factorial(j-i) if j >= i
                        else zeros(2, 2))
                expJblock = Matrix(2*n, 2*n, lambda i,j: expJblock2[i//2,j//2][i%2,j%2])

                blocks.append(exprt * expJblock)
                # 将实部和虚部向量加入向量列表
                for i in range(n):
                    vectors.append(re(chain[i]))
                    vectors.append(im(chain[i]))
            else:
                # 将链中的向量直接加入向量列表，并构建指数矩阵
                vectors.extend(chain)
                fun = lambda i,j: t**(j-i)/factorial(j-i) if j >= i else 0
                expJblock = Matrix(n, n, fun)
                blocks.append(exp(e * t) * expJblock)

    # 构建 Jordan 形式的矩阵 expJ
    expJ = Matrix.diag(*blocks)
    # 构建由 vectors 组成的矩阵 P
    P = Matrix(N, N, lambda i,j: vectors[j][i])

    return P, expJ
# Note: To add a docstring example with tau
def linodesolve(A, t, b=None, B=None, type="auto", doit=False,
                tau=None):
    r"""
    System of n equations linear first-order differential equations

    Explanation
    ===========

    This solver solves the system of ODEs of the following form:

    .. math::
        X'(t) = A(t) X(t) +  b(t)

    Here, $A(t)$ is the coefficient matrix, $X(t)$ is the vector of n independent variables,
    $b(t)$ is the non-homogeneous term and $X'(t)$ is the derivative of $X(t)$

    Depending on the properties of $A(t)$ and $b(t)$, this solver evaluates the solution
    differently.

    When $A(t)$ is constant coefficient matrix and $b(t)$ is zero vector i.e. system is homogeneous,
    the system is "type1". The solution is:

    .. math::
        X(t) = \exp(A t) C

    Here, $C$ is a vector of constants and $A$ is the constant coefficient matrix.

    When $A(t)$ is constant coefficient matrix and $b(t)$ is non-zero i.e. system is non-homogeneous,
    the system is "type2". The solution is:

    .. math::
        X(t) = e^{A t} ( \int e^{- A t} b \,dt + C)

    When $A(t)$ is coefficient matrix such that its commutative with its antiderivative $B(t)$ and
    $b(t)$ is a zero vector i.e. system is homogeneous, the system is "type3". The solution is:

    .. math::
        X(t) = \exp(B(t)) C

    When $A(t)$ is commutative with its antiderivative $B(t)$ and $b(t)$ is non-zero i.e. system is
    non-homogeneous, the system is "type4". The solution is:

    .. math::
        X(t) =  e^{B(t)} ( \int e^{-B(t)} b(t) \,dt + C)

    When $A(t)$ is a coefficient matrix such that it can be factorized into a scalar and a constant
    coefficient matrix:

    .. math::
        A(t) = f(t) * A

    Where $f(t)$ is a scalar expression in the independent variable $t$ and $A$ is a constant matrix,
    then we can do the following substitutions:

    .. math::
        tau = \int f(t) dt, X(t) = Y(tau), b(t) = b(f^{-1}(tau))

    Here, the substitution for the non-homogeneous term is done only when its non-zero.
    Using these substitutions, our original system becomes:

    .. math::
        Y'(tau) = A * Y(tau) + b(tau)/f(tau)

    The above system can be easily solved using the solution for "type1" or "type2" depending
    on the homogeneity of the system. After we get the solution for $Y(tau)$, we substitute the
    solution for $tau$ as $t$ to get back $X(t)$

    .. math::
        X(t) = Y(tau)

    Systems of "type5" and "type6" have a commutative antiderivative but we use this solution
    because its faster to compute.

    The final solution is the general solution for all the four equations since a constant coefficient
    matrix is always commutative with its antidervative.

    An additional feature of this function is, if someone wants to substitute for value of the independent
    variable, they can pass the substitution `tau` and the solution will have the independent variable
    """
    substituted with the passed expression(`tau`).

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of linear first order ODEs.
    t : Symbol
        Independent variable in the system of ODEs.
    b : Matrix or None
        Non-homogeneous term in the system of ODEs. If None is passed,
        a homogeneous system of ODEs is assumed.
    B : Matrix or None
        Antiderivative of the coefficient matrix. If the antiderivative
        is not passed and the solution requires the term, then the solver
        would compute it internally.
    type : String
        Type of the system of ODEs passed. Depending on the type, the
        solution is evaluated. The type values allowed and the corresponding
        system it solves are: "type1" for constant coefficient homogeneous
        "type2" for constant coefficient non-homogeneous, "type3" for non-constant
        coefficient homogeneous, "type4" for non-constant coefficient non-homogeneous,
        "type5" and "type6" for non-constant coefficient homogeneous and non-homogeneous
        systems respectively where the coefficient matrix can be factorized to a constant
        coefficient matrix.
        The default value is "auto" which will let the solver decide the correct type of
        the system passed.
    doit : Boolean
        Evaluate the solution if True, default value is False
    tau: Expression
        Used to substitute for the value of `t` after we get the solution of the system.

    Examples
    ========

    To solve the system of ODEs using this function directly, several things must be
    done in the right order. Wrong inputs to the function will lead to incorrect results.

    >>> from sympy import symbols, Function, Eq
    >>> from sympy.solvers.ode.systems import canonical_odes, linear_ode_to_matrix, linodesolve, linodesolve_type
    >>> from sympy.solvers.ode.subscheck import checkodesol
    >>> f, g = symbols("f, g", cls=Function)
    >>> x, a = symbols("x, a")
    >>> funcs = [f(x), g(x)]
    >>> eqs = [Eq(f(x).diff(x) - f(x), a*g(x) + 1), Eq(g(x).diff(x) + g(x), a*f(x))]

    Here, it is important to note that before we derive the coefficient matrix, it is
    important to get the system of ODEs into the desired form. For that we will use
    :obj:`sympy.solvers.ode.systems.canonical_odes()`.

    >>> eqs = canonical_odes(eqs, funcs, x)
    >>> eqs
    [[Eq(Derivative(f(x), x), a*g(x) + f(x) + 1), Eq(Derivative(g(x), x), a*f(x) - g(x))]]

    Now, we will use :obj:`sympy.solvers.ode.systems.linear_ode_to_matrix()` to get the coefficient matrix and the
    non-homogeneous term if it is there.

    >>> eqs = eqs[0]
    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)
    >>> A = A0

    We have the coefficient matrices and the non-homogeneous term ready. Now, we can use
    :obj:`sympy.solvers.ode.systems.linodesolve_type()` to get the information for the system of ODEs
    to finally pass it to the solver.
    >>> system_info = linodesolve_type(A, x, b=b)
    # 调用 linodesolve_type 函数，获取解系统类型及反导数信息
    >>> sol_vector = linodesolve(A, x, b=b, B=system_info['antiderivative'], type=system_info['type_of_equation'])
    # 调用 linodesolve 函数，解线性常系数常微分方程组，使用系统信息中的反导数和方程类型

    Now, we can prove if the solution is correct or not by using :obj:`sympy.solvers.ode.checkodesol()`

    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    # 根据函数列表 funcs 和解向量 sol_vector 构建方程组 sol
    >>> checkodesol(eqs, sol)
    # 调用 checkodesol 函数，验证求解的常微分方程组的解是否正确
    (True, [0, 0])

    We can also use the doit method to evaluate the solutions passed by the function.

    >>> sol_vector_evaluated = linodesolve(A, x, b=b, type="type2", doit=True)
    # 调用 linodesolve 函数，以 type="type2" 求解线性常系数非常微分方程组，并对解使用 doit 方法进行求值

    Now, we will look at a system of ODEs which is non-constant.

    >>> eqs = [Eq(f(x).diff(x), f(x) + x*g(x)), Eq(g(x).diff(x), -x*f(x) + g(x))]
    # 定义一个非常微分方程组 eqs，其中包含两个方程

    The system defined above is already in the desired form, so we do not have to convert it.

    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)
    # 调用 linear_ode_to_matrix 函数，将常微分方程组转换为系数矩阵形式
    >>> A = A0
    # 选择系数矩阵 A0 作为 A

    A user can also pass the commutative antiderivative required for type3 and type4 system of ODEs.
    Passing an incorrect one will lead to incorrect results. If the coefficient matrix is not commutative
    with its antiderivative, then :obj:`sympy.solvers.ode.systems.linodesolve_type()` raises a NotImplementedError.
    If it does have a commutative antiderivative, then the function just returns the information about the system.

    >>> system_info = linodesolve_type(A, x, b=b)
    # 调用 linodesolve_type 函数，获取解系统类型及反导数信息

    Now, we can pass the antiderivative as an argument to get the solution. If the system information is not
    passed, then the solver will compute the required arguments internally.

    >>> sol_vector = linodesolve(A, x, b=b)
    # 调用 linodesolve 函数，解线性常系数常微分方程组

    Once again, we can verify the solution obtained.

    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    # 根据函数列表 funcs 和解向量 sol_vector 构建方程组 sol
    >>> checkodesol(eqs, sol)
    # 调用 checkodesol 函数，验证求解的常微分方程组的解是否正确
    (True, [0, 0])

    Returns
    =======

    List

    Raises
    ======

    ValueError
        This error is raised when the coefficient matrix, non-homogeneous term
        or the antiderivative, if passed, are not a matrix or
        do not have correct dimensions
    NonSquareMatrixError
        When the coefficient matrix or its antiderivative, if passed is not a
        square matrix
    NotImplementedError
        If the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linear_ode_to_matrix: Coefficient matrix computation function
    canonical_odes: System of ODEs representation change
    linodesolve_type: Getting information about systems of ODEs to pass in this solver

    """

    if not isinstance(A, MatrixBase):
        raise ValueError(filldedent('''\
            The coefficients of the system of ODEs should be of type Matrix
        '''))

    if not A.is_square:
        raise NonSquareMatrixError(filldedent('''\
            The coefficient matrix must be a square
        '''))
    # 如果存在非空的非齐次项矩阵b，则进行类型检查，确保其为MatrixBase类型
    if b is not None:
        if not isinstance(b, MatrixBase):
            raise ValueError(filldedent('''\
                The non-homogeneous terms of the system of ODEs should be of type Matrix
            '''))

        # 检查非齐次项矩阵b的行数与系数矩阵A的行数是否相同
        if A.rows != b.rows:
            raise ValueError(filldedent('''\
                The system of ODEs should have the same number of non-homogeneous terms and the number of
                equations
            '''))

    # 如果存在反导数矩阵B，则进行类型检查，确保其为MatrixBase类型，并且是方阵
    if B is not None:
        if not isinstance(B, MatrixBase):
            raise ValueError(filldedent('''\
                The antiderivative of coefficients of the system of ODEs should be of type Matrix
            '''))

        # 检查反导数矩阵B是否为方阵
        if not B.is_square:
            raise NonSquareMatrixError(filldedent('''\
                The antiderivative of the coefficient matrix must be a square
            '''))

        # 检查系数矩阵A和反导数矩阵B的行数是否相同
        if A.rows != B.rows:
            raise ValueError(filldedent('''\
                        The coefficient matrix and its antiderivative should have same dimensions
                    '''))

    # 检查输入的类型是否为"auto"或者有效的"type1"到"type6"之一
    if not any(type == "type{}".format(i) for i in range(1, 7)) and not type == "auto":
        raise ValueError(filldedent('''\
                    The input type should be a valid one
                '''))

    # 获取系数矩阵A的行数
    n = A.rows

    # 创建一个由符号常数组成的矩阵Cvect，用于未知数的求解
    Cvect = Matrix([Dummy() for _ in range(n)])

    # 如果没有指定非齐次项矩阵b，并且类型为"type2", "type4", "type6"之一，则创建一个零矩阵作为b
    if b is None and any(type == typ for typ in ["type2", "type4", "type6"]):
        b = zeros(n, 1)

    # 标记是否进行了变换
    is_transformed = tau is not None
    # 保存传入的类型信息
    passed_type = type

    # 如果类型为"auto"，则调用linodesolve_type函数自动识别ODE系统的类型，并更新type和B
    if type == "auto":
        system_info = linodesolve_type(A, t, b=b)
        type = system_info["type_of_equation"]
        B = system_info["antiderivative"]

    # 如果类型为"type5"或"type6"，则需要进行特殊处理
    if type in ("type5", "type6"):
        is_transformed = True
        # 如果传入的类型不是"auto"，且未指定tau，则调用_first_order_type5_6_subs函数进行转换处理
        if passed_type != "auto":
            if tau is None:
                system_info = _first_order_type5_6_subs(A, t, b=b)
                if not system_info:
                    raise ValueError(filldedent('''
                        The system passed isn't {}.
                    '''.format(type)))

                # 更新tau, t, A, b为转换后的值
                tau = system_info['tau']
                t = system_info['t_']
                A = system_info['A']
                b = system_info['b']

    # 定义一个函数intx_wrtt用于积分，将x关于t积分
    intx_wrtt = lambda x: Integral(x, t) if x else 0

    # 根据不同的类型计算解向量sol_vector
    if type in ("type1", "type2", "type5", "type6"):
        # 计算矩阵指数和约当形式
        P, J = matrix_exp_jordan_form(A, t)
        P = simplify(P)

        # 根据不同类型计算解向量sol_vector
        if type in ("type1", "type5"):
            sol_vector = P * (J * Cvect)
        else:
            Jinv = J.subs(t, -t)
            sol_vector = P * J * ((Jinv * P.inv() * b).applyfunc(intx_wrtt) + Cvect)
    else:
        # 如果没有指定反导数矩阵B，则调用_is_commutative_anti_derivative函数计算B
        if B is None:
            B, _ = _is_commutative_anti_derivative(A, t)

        # 根据不同类型计算解向量sol_vector
        if type == "type3":
            sol_vector = B.exp() * Cvect
        else:
            sol_vector = B.exp() * (((-B).exp() * b).applyfunc(intx_wrtt) + Cvect)

    # 如果进行了变换，则用tau替换解向量中的t
    if is_transformed:
        sol_vector = sol_vector.subs(t, tau)

    # 获取解向量中所有指数函数的生成器
    gens = sol_vector.atoms(exp)
    # 如果 type 不等于 "type1"，对 sol_vector 中的每个元素执行 expand_mul 函数
    if type != "type1":
        sol_vector = [expand_mul(s) for s in sol_vector]

    # 对 sol_vector 中的每个元素执行 collect 函数，并使用 ordered(gens) 作为参数，exact=True 表示精确匹配
    sol_vector = [collect(s, ordered(gens), exact=True) for s in sol_vector]

    # 如果参数 doit 为真，则对 sol_vector 中的每个元素执行 doit 方法
    if doit:
        sol_vector = [s.doit() for s in sol_vector]

    # 返回处理后的 sol_vector 结果
    return sol_vector
def _matrix_is_constant(M, t):
    """检查矩阵 M 是否与变量 t 独立。"""
    # 检查矩阵 M 中的每个系数是否不依赖于变量 t
    return all(coef.as_independent(t, as_Add=True)[1] == 0 for coef in M)


def canonical_odes(eqs, funcs, t):
    r"""
    解决系统中最高阶导数的函数

    Explanation
    ===========

    此函数输入一个ODE系统，基于系统、依赖变量及其最高阶，返回以下形式的系统：
    
    .. math::
        X'(t) = A(t) X(t) + b(t)
    
    这里，$X(t)$ 是较低阶依赖变量的向量，$A(t)$ 是系数矩阵，$b(t)$ 是非齐次项，$X'(t)$ 是依赖变量在其最高阶。我们使用“规范形式”来表示上述形式的ODE系统。

    如果传入的系统具有非线性项且具有多个解，则返回其规范形式的系统列表。

    Parameters
    ==========

    eqs : List
        ODE系统的列表
    funcs : List
        依赖变量的列表
    t : Symbol
        自变量

    Examples
    ========

    >>> from sympy import symbols, Function, Eq, Derivative
    >>> from sympy.solvers.ode.systems import canonical_odes
    >>> f, g = symbols("f g", cls=Function)
    >>> x, y = symbols("x y")
    >>> funcs = [f(x), g(x)]
    >>> eqs = [Eq(f(x).diff(x) - 7*f(x), 12*g(x)), Eq(g(x).diff(x) + g(x), 20*f(x))]

    >>> canonical_eqs = canonical_odes(eqs, funcs, x)
    >>> canonical_eqs
    [[Eq(Derivative(f(x), x), 7*f(x) + 12*g(x)), Eq(Derivative(g(x), x), 20*f(x) - g(x))]]

    >>> system = [Eq(Derivative(f(x), x)**2 - 2*Derivative(f(x), x) + 1, 4), Eq(-y*f(x) + Derivative(g(x), x), 0)]

    >>> canonical_system = canonical_odes(system, funcs, x)
    >>> canonical_system
    [[Eq(Derivative(f(x), x), -1), Eq(Derivative(g(x), x), y*f(x))], [Eq(Derivative(f(x), x), 3), Eq(Derivative(g(x), x), y*f(x))]]

    Returns
    =======

    List

    """
    from sympy.solvers.solvers import solve

    # 获取ODE系统中每个函数的阶数
    order = _get_func_order(eqs, funcs)

    # 解ODE系统，返回其规范形式的解
    canon_eqs = solve(eqs, *[func.diff(t, order[func]) for func in funcs], dict=True)

    systems = []
    for eq in canon_eqs:
        # 根据解构建系统的列表
        system = [Eq(func.diff(t, order[func]), eq[func.diff(t, order[func])]) for func in funcs]
        systems.append(system)

    return systems


def _is_commutative_anti_derivative(A, t):
    r"""
    用于确定传递的矩阵是否与其相对于独立变量 t 的反导数可交换的辅助函数

    Explanation
    ===========

    此函数检查传递的矩阵 $A$ 是否与其相对于独立变量 $t$ 的反导数 $B(t)$ 可交换。

    .. math::
        B(t) = \int A(t) dt

    该函数输出两个值，第一个是反导数 $B(t)$，第二个是布尔值。若为True，则矩阵 $A(t)$ 可与 $B(t)$ 可交换，否则矩阵
    ```

    B(t)$ 可与 $B(t)$ 不可交换。

    Returns
    =======

    Tuple

    """
    # 这里是函数的具体实现，可以根据具体情况添加代码和注释
    # 对给定矩阵 A 进行积分，积分变量为 t
    B = integrate(A, t)
    
    # 检查积分后的矩阵 B 是否满足 B*A - A*B 的结果为零矩阵，
    # 并应用 expand 和 factor_terms 对每个元素进行处理
    is_commuting = (B*A - A*B).applyfunc(expand).applyfunc(factor_terms).is_zero_matrix
    
    # 如果 is_commuting 为 None，则将其设为 False
    is_commuting = False if is_commuting is None else is_commuting
    
    # 返回计算得到的矩阵 B 和布尔值 is_commuting
    return B, is_commuting
# 定义一个函数，用于从矩阵 A 中提取一个因子项
def _factor_matrix(A, t):
    term = None  # 初始化一个变量 term，用于存储提取出的因子项
    for element in A:
        temp_term = element.as_independent(t)[1]  # 获取元素中关于 t 的独立项
        if temp_term.has(t):
            term = temp_term  # 如果找到包含 t 的项，则将其赋给 term
            break

    if term is not None:
        # 如果找到了 term，则将 A 除以 term，并对结果应用 ratsimp 函数
        A_factored = (A/term).applyfunc(ratsimp)
        # 检查 A_factored 是否是常数矩阵
        can_factor = _matrix_is_constant(A_factored, t)
        term = (term, A_factored) if can_factor else None  # 如果可以因子化，则更新 term

    return term  # 返回提取出的因子项或者 None


# 定义一个函数，判断给定的矩阵 A 是否符合第二类二阶系统的形式
def _is_second_order_type2(A, t):
    term = _factor_matrix(A, t)  # 提取 A 的因子项
    is_type2 = False  # 初始化 is_type2 变量为 False

    if term is not None:
        term = 1/term[0]  # 计算 term 的倒数
        is_type2 = term.is_polynomial()  # 检查 term 是否是多项式

    if is_type2:
        poly = Poly(term.expand(), t)  # 创建一个 t 的多项式对象 poly
        monoms = poly.monoms()  # 获取 poly 的单项式列表

        # 检查 poly 的主要单项是否在 (2, 4) 中
        if monoms[0][0] in (2, 4):
            cs = _get_poly_coeffs(poly, 4)  # 获取 poly 的前四个系数
            a, b, c, d, e = cs  # 将系数分配给变量 a, b, c, d, e

            # 对 a, c, d 进行求解并应用 powdenest 和 sqrt 函数
            a1 = powdenest(sqrt(a), force=True)
            c1 = powdenest(sqrt(e), force=True)
            b1 = powdenest(sqrt(c - 2*a1*c1), force=True)

            # 检查二阶系统是否满足 type2 条件
            is_type2 = (b == 2*a1*b1) and (d == 2*b1*c1)
            term = a1*t**2 + b1*t + c1  # 更新 term 为二次多项式

        else:
            is_type2 = False  # 如果不满足 (2, 4) 中的任何一个，则设置为 False

    return is_type2, term  # 返回是否为 type2 和更新后的 term


# 定义一个函数，从多项式对象 poly 中获取指定阶数的系数列表
def _get_poly_coeffs(poly, order):
    cs = [0 for _ in range(order+1)]  # 初始化一个长度为 order+1 的系数列表 cs
    for c, m in zip(poly.coeffs(), poly.monoms()):
        cs[-1-m[0]] = c  # 将 poly 的系数与单项式对应的位置设置到 cs 中
    return cs  # 返回系数列表 cs


# 定义一个函数，用于匹配给定矩阵 A1 和 A0 是否符合第二类二阶系统的形式
def _match_second_order_type(A1, A0, t, b=None):
    r"""
    Works only for second order system in its canonical form.

    Type 0: Constant coefficient matrix, can be simply solved by
            introducing dummy variables.
    Type 1: When the substitution: $U = t*X' - X$ works for reducing
            the second order system to first order system.
    Type 2: When the system is of the form: $poly * X'' = A*X$ where
            $poly$ is square of a quadratic polynomial with respect to
            *t* and $A$ is a constant coefficient matrix.

    """
    match = {"type_of_equation": "type0"}  # 初始化一个包含类型信息的字典 match
    n = A1.shape[0]  # 获取 A1 的行数

    if _matrix_is_constant(A1, t) and _matrix_is_constant(A0, t):
        return match  # 如果 A1 和 A0 都是关于 t 的常数矩阵，则直接返回 match

    if (A1 + A0*t).applyfunc(expand_mul).is_zero_matrix:
        # 如果 A1 + A0*t 的每个元素经过 expand_mul 处理后为零矩阵
        match.update({"type_of_equation": "type1", "A1": A1})  # 更新 match 为 type1

    elif A1.is_zero_matrix and (b is None or b.is_zero_matrix):
        is_type2, term = _is_second_order_type2(A0, t)  # 判断 A0 是否满足 type2 条件
        if is_type2:
            a, b, c = _get_poly_coeffs(Poly(term, t), 2)  # 获取 term 的二次多项式系数
            A = (A0*(term**2).expand()).applyfunc(ratsimp) + (b**2/4 - a*c)*eye(n, n)
            tau = integrate(1/term, t)  # 计算 term 的积分 tau
            t_ = Symbol("{}_".format(t))  # 创建一个新的符号 t_

            # 更新 match 为 type2 并添加相关信息
            match.update({"type_of_equation": "type2", "A0": A,
                          "g(t)": sqrt(term), "tau": tau, "is_transformed": True,
                          "t_": t_})

    return match  # 返回匹配结果的字典 match


# 定义一个函数，用于执行第一类二阶系统的特定替换
def _second_order_subs_type1(A, b, funcs, t):
    r"""
    For a linear, second order system of ODEs, a particular substitution.

    A system of the below form can be reduced to a linear first order system of
    ODEs:
    .. math::
        X'' = A(t) * (t*X' - X) + b(t)

    By substituting:
    .. math::  U = t*X' - X

    """
    # 省略了具体的替换操作，文档字符串中提供了替换的形式和原理
    """
    To get the system:
    .. math::  U' = t*(A(t)*U + b(t))
    
    Where $U$ is the vector of dependent variables, $X$ is the vector of dependent
    variables in `funcs` and $X'$ is the first order derivative of $X$ with respect to
    $t$. It may or may not reduce the system into linear first order system of ODEs.
    
    Then a check is made to determine if the system passed can be reduced or not, if
    this substitution works, then the system is reduced and its solved for the new
    substitution. After we get the solution for $U$:
    
    .. math::  U = a(t)
    
    We substitute and return the reduced system:
    
    .. math::
        a(t) = t*X' - X
    
    Parameters
    ==========
    
    A: Matrix
        Coefficient matrix($A(t)*t$) of the second order system of this form.
    b: Matrix
        Non-homogeneous term($b(t)$) of the system of ODEs.
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the system of ODEs.
    
    Returns
    =======
    
    List
    """
    
    # 构建方程组的右侧向量 U = [t*func.diff(t) - func for func in funcs]
    U = Matrix([t*func.diff(t) - func for func in funcs])
    
    # 解二阶系统的非齐次线性常系数微分方程组，得到其解
    sol = linodesolve(A, t, t*b)
    
    # 将解映射为方程形式 Eq(u, s)，其中 u 是原方程中的变量，s 是解
    reduced_eqs = [Eq(u, s) for s, u in zip(sol, U)]
    
    # 对简化后的方程组进行规范化处理，返回规范化后的方程组中的第一个元素
    reduced_eqs = canonical_odes(reduced_eqs, funcs, t)[0]
    
    # 返回简化后的方程组
    return reduced_eqs
def _second_order_subs_type2(A, funcs, t_):
    r"""
    Returns a second order system based on the coefficient matrix passed.

    Explanation
    ===========

    This function returns a system of second order ODE of the following form:

    .. math::
        X'' = A * X

    Here, $X$ is the vector of dependent variables, but a bit modified, $A$ is the
    coefficient matrix passed.

    Along with returning the second order system, this function also returns the new
    dependent variables with the new independent variable `t_` passed.

    Parameters
    ==========

    A: Matrix
        Coefficient matrix of the system
    funcs: List
        List of old dependent variables
    t_: Symbol
        New independent variable

    Returns
    =======

    List, List

    """
    # Extract function names from funcs list
    func_names = [func.func.__name__ for func in funcs]
    # Create new dependent variables using Function and Dummy objects
    new_funcs = [Function(Dummy("{}_".format(name)))(t_) for name in func_names]
    # Compute right-hand side expressions of the second order ODE system
    rhss = A * Matrix(new_funcs)
    # Create equations representing the second order ODE system
    new_eqs = [Eq(func.diff(t_, 2), rhs) for func, rhs in zip(new_funcs, rhss)]

    return new_eqs, new_funcs


def _is_euler_system(As, t):
    # Check if all matrices As*t**i are constant coefficient matrices
    return all(_matrix_is_constant((A*t**i).applyfunc(ratsimp), t) for i, A in enumerate(As))


def _classify_linear_system(eqs, funcs, t, is_canon=False):
    r"""
    Returns a dictionary with details of the eqs if the system passed is linear
    and can be classified by this function else returns None

    Explanation
    ===========

    This function takes the eqs, converts it into a form Ax = b where x is a vector of terms
    containing dependent variables and their derivatives till their maximum order. If it is
    possible to convert eqs into Ax = b, then all the equations in eqs are linear otherwise
    they are non-linear.

    To check if the equations are constant coefficient, we need to check if all the terms in
    A obtained above are constant or not.

    To check if the equations are homogeneous or not, we need to check if b is a zero matrix
    or not.

    Parameters
    ==========

    eqs: List
        List of ODEs
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the equations in eqs
    is_canon: Boolean
        If True, then this function will not try to get the
        system in canonical form. Default value is False

    Returns
    =======

    match = {
        'no_of_equation': len(eqs),
        'eq': eqs,
        'func': funcs,
        'order': order,
        'is_linear': is_linear,
        'is_constant': is_constant,
        'is_homogeneous': is_homogeneous,
    }

    """
    # 返回值可以是字典、字典列表或者None
    # 字典包含以下键值对：
    #   1. no_of_equation: 方程的数量
    #   2. eq: 方程组
    #   3. func: 依赖变量的列表
    #   4. order: 字典，给出了依赖变量在方程组中的顺序
    #   5. is_linear: 布尔值，指示方程组是否线性
    #   6. is_constant: 布尔值，指示方程组系数是否常数
    #   7. is_homogeneous: 布尔值，指示方程组是否齐次
    #   8. commutative_antiderivative: 如果系数矩阵非常数且与其反导数交换，则是系数矩阵的反导数
    #   9. is_general: 布尔值，指示ODE系统是否可以使用通用求解器解决
    #   10. rhs: Matrix形式的非齐次ODE系统的右手边。此键可能不存在。
    #   11. is_higher_order: 如果系统的阶数大于1，则为True。此键可能不存在。
    #   12. is_second_order: 如果系统是二阶ODE，则为True。此键可能不存在。
    # 如果方程组是线性且系数常数，则返回此字典；否则返回None。
    """

    # 如果funcs的长度与eqs的长度不相等，则抛出ValueError异常
    if len(funcs) != len(eqs):
        raise ValueError("Number of functions given is not equal to the number of equations %s" % funcs)

    # 对每个函数检查其参数个数是否为1，如果不是则抛出ValueError异常
    for func in funcs:
        if len(func.args) != 1:
            raise ValueError("dsolve() and classify_sysode() work with "
            "functions of one variable only, not %s" % func)

    # 使用辅助函数获取func_dict和order
    order = _get_func_order(eqs, funcs)
    # 计算系统的最高阶数
    system_order = max(order[func] for func in funcs)
    # 检查系统是否为高阶ODE
    is_higher_order = system_order > 1
    # 检查系统是否为二阶ODE
    is_second_order = system_order == 2 and all(order[func] == 2 for func in funcs)

    # 如果每个函数的参数个数不是1，则不添加检查

    # 进行线性检查
    try:
        # 如果不是标准ODE形式，则canon_eqs可能是一个列表
        # canon_odes函数将eqs转换为标准ODE形式
        # 如果is_canon为假，则调用canonical_odes函数；否则直接返回eqs
        canon_eqs = canonical_odes(eqs, funcs, t) if not is_canon else [eqs]
        # 如果canon_eqs的长度为1，则调用linear_ode_to_matrix函数将其转换为矩阵形式的线性ODE
        if len(canon_eqs) == 1:
            As, b = linear_ode_to_matrix(canon_eqs[0], funcs, t, system_order)
        else:
            # 如果长度大于1，则构建匹配字典并返回
            match = {
                'is_implicit': True,
                'canon_eqs': canon_eqs
            }
            return match
    # 当ODE系统是非线性时，会引发ODENonlinearError错误。
    # 此函数捕获错误并返回None。
    except ODENonlinearError:
        return None

    # 判断ODE系统是否为线性
    is_linear = True

    # 检查ODE系统是否为齐次的
    is_homogeneous = True if b.is_zero_matrix else False

    # 用于匹配信息的字典，描述ODE系统的特征和属性
    match = {
        'no_of_equation': len(eqs),  # 方程的数量
        'eq': eqs,                   # 方程列表
        'func': funcs,               # 相关函数
        'order': order,              # 方程的阶数
        'is_linear': is_linear,      # 是否为线性ODE系统
        'is_homogeneous': is_homogeneous,  # 是否为齐次ODE系统
        'is_general': True           # 是否为一般情况（暂时设为True）
    }

    # 如果系统非齐次，则将右侧项b加入match字典
    if not is_homogeneous:
        match['rhs'] = b

    # 检查所有矩阵A_是否为常数矩阵
    is_constant = all(_matrix_is_constant(A_, t) for A_ in As)

    # 当函数准备好处理非线性ODE系统时，将添加match['is_linear']的检查

    # 如果不是高阶ODE系统
    if not is_higher_order:
        A = As[1]
        match['func_coeff'] = A  # 将函数系数加入match字典

        # 检查是否为常数系数矩阵
        is_constant = _matrix_is_constant(A, t)
        match['is_constant'] = is_constant

        try:
            # 尝试使用线性ODE求解函数linodesolve_type解决ODE系统
            system_info = linodesolve_type(A, t, b=b)
        except NotImplementedError:
            return None

        # 更新match字典
        match.update(system_info)
        antiderivative = match.pop("antiderivative")  # 弹出反导数项

        # 如果不是常数系数，则将交换性反导数项加入match字典
        if not is_constant:
            match['commutative_antiderivative'] = antiderivative

        return match
    else:
        match['type_of_equation'] = "type0"  # 类型0的ODE方程

        # 如果是二阶ODE系统
        if is_second_order:
            A1, A0 = As[1:]

            # 使用_match_second_order_type函数匹配二阶ODE系统类型
            match_second_order = _match_second_order_type(A1, A0, t)
            match.update(match_second_order)

            match['is_second_order'] = True  # 标记为二阶ODE系统

        # 如果类型为type0且不是常数系统，则检查是否为Euler形式
        if match['type_of_equation'] == "type0" and not is_constant:
            is_euler = _is_euler_system(As, t)

            # 如果是Euler形式，则进行转换
            if is_euler:
                t_ = Symbol('{}_'.format(t))
                match.update({'is_transformed': True, 'type_of_equation': 'type1',
                              't_': t_})
            else:
                # 如果不是Euler形式，则检查是否为Jordan块形式
                is_jordan = lambda M: M == Matrix.jordan_block(M.shape[0], M[0, 0])
                terms = _factor_matrix(As[-1], t)

                # 如果除了最后一项外都是零矩阵，并且terms不为空且不是Jordan块形式，则更新为type2形式
                if all(A.is_zero_matrix for A in As[1:-1]) and terms is not None and not is_jordan(terms[1]):
                    P, J = terms[1].jordan_form()
                    match.update({'type_of_equation': 'type2', 'J': J,
                                  'f(t)': terms[0], 'P': P, 'is_transformed': True})

            # 如果不是type0类型的ODE方程且为二阶ODE系统，则移除is_second_order项
            if match['type_of_equation'] != 'type0' and is_second_order:
                match.pop('is_second_order', None)

        match['is_higher_order'] = is_higher_order  # 标记是否为高阶ODE系统

        return match  # 返回匹配的字典
# 将传入的等式列表转换为经过预处理的标准形式列表
def _preprocess_eqs(eqs):
    processed_eqs = []
    for eq in eqs:
        # 如果等式已经是Equality类型，则直接添加到processed_eqs中，否则构造成Equality类型
        processed_eqs.append(eq if isinstance(eq, Equality) else Eq(eq, 0))
    return processed_eqs


# 将等式列表转换为两个字典：eqsorig 和 eqsmap
def _eqs2dict(eqs, funcs):
    eqsorig = {}  # 存放函数和其对应的等式原始形式
    eqsmap = {}   # 存放函数和其直接依赖的函数集合
    funcset = set(funcs)  # 转换函数列表为集合形式
    for eq in eqs:
        # 从等式的左边表达式中获取单个未知函数
        f1, = eq.lhs.atoms(AppliedUndef)
        # 从等式的右边表达式中获取除了f1外的其余未知函数，并且这些函数必须是funcs集合中的一部分
        f2s = (eq.rhs.atoms(AppliedUndef) - {f1}) & funcset
        # 将函数f1及其依赖的函数集合f2s添加到eqsmap字典中
        eqsmap[f1] = f2s
        # 将函数f1及其原始等式添加到eqsorig字典中
        eqsorig[f1] = eq
    return eqsmap, eqsorig


# 将字典表示的依赖关系转换为图的形式
def _dict2graph(d):
    nodes = list(d)  # 将字典的键转换为节点列表
    edges = [(f1, f2) for f1, f2s in d.items() for f2 in f2s]  # 生成节点之间的边列表
    G = (nodes, edges)  # 构建图的表示形式，用元组(nodes, edges)表示
    return G


# 判断给定的SCC是否满足类型1的条件
def _is_type1(scc, t):
    eqs, funcs = scc

    try:
        # 尝试将线性ODE转换为矩阵形式(A1, A0)和向量b
        (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, 1)
    except (ODENonlinearError, ODEOrderError):
        return False

    # 检查矩阵A0是否是常数矩阵且向量b是否是零矩阵
    if _matrix_is_constant(A0, t) and b.is_zero_matrix:
        return True

    return False


# 将类型1的子系统合并为更大的子系统
def _combine_type1_subsystems(subsystem, funcs, t):
    indices = [i for i, sys in enumerate(zip(subsystem, funcs)) if _is_type1(sys, t)]
    remove = set()
    for ip, i in enumerate(indices):
        for j in indices[ip+1:]:
            # 检查两个子系统之间是否有共享的未知函数，若有则将它们合并
            if any(eq2.has(funcs[i]) for eq2 in subsystem[j]):
                subsystem[j] = subsystem[i] + subsystem[j]
                remove.add(i)
    # 移除被合并的子系统
    subsystem = [sys for i, sys in enumerate(subsystem) if i not in remove]
    return subsystem


# 将等式组划分为多个子系统
def _component_division(eqs, funcs, t):

    # 假设每个等式在规范形式下，即[f(x).diff(x) = .., g(x).diff(x) = .., etc]
    # 并且传入的系统是它的一阶形式
    eqsmap, eqsorig = _eqs2dict(eqs, funcs)

    subsystems = []
    # 使用强连通分量分解连接图
    for cc in connected_components(_dict2graph(eqsmap)):
        # 从eqsmap中提取当前连通分量cc的依赖关系字典eqsmap_c
        eqsmap_c = {f: eqsmap[f] for f in cc}
        # 对当前连通分量进行强连通分量分解，得到子系统列表sccs
        sccs = strongly_connected_components(_dict2graph(eqsmap_c))
        # 将每个强连通分量转换为对应的等式列表，并尝试合并类型1的子系统
        subsystem = [[eqsorig[f] for f in scc] for scc in sccs]
        subsystem = _combine_type1_subsystems(subsystem, sccs, t)
        # 将处理后的子系统添加到subsystems列表中
        subsystems.append(subsystem)

    return subsystems


# 使用线性ODE求解器解决方程组，并返回等式列表
def _linear_ode_solver(match):
    t = match['t']
    funcs = match['func']

    rhs = match.get('rhs', None)
    tau = match.get('tau', None)
    t = match['t_'] if 't_' in match else t
    A = match['func_coeff']

    # 如果矩阵具有常数系数，则将B设置为None
    B = match.get('commutative_antiderivative', None)
    type = match['type_of_equation']

    # 使用linodesolve函数解线性ODE，并得到解向量sol_vector
    sol_vector = linodesolve(A, t, b=rhs, B=B,
                             type=type, tau=tau)

    # 将解向量转换为等式列表sol
    sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]

    return sol


# 根据指定的关键函数key，从等式列表中选取对应函数列表funcs的等式
def _select_equations(eqs, funcs, key=lambda x: x):
    # 构建等式字典，键为等式左边表达式，值为等式右边表达式
    eq_dict = {e.lhs: e.rhs for e in eqs}
    # 使用key函数从eq_dict中选取相应的等式，并构建成等式列表返回
    return [Eq(f, eq_dict[key(f)]) for f in funcs]


# 使用高阶ODE求解器解决方程组，并返回等式列表
def _higher_order_ode_solver(match):
    eqs = match["eq"]
    funcs = match["func"]
    t = match["t"]
    sysorder = match['order']
    type = match.get('type_of_equation', "type0")

    is_second_order = match.get('is_second_order', False)
    # 检查是否在匹配对象中存在 'is_transformed' 键，如果不存在则默认为 False
    is_transformed = match.get('is_transformed', False)

    # 如果 is_transformed 为 True，并且 type 等于 "type1"
    is_euler = is_transformed and type == "type1"

    # 如果 is_transformed 为 True，并且 type 等于 "type2"，并且 'P' 在匹配对象中
    is_higher_order_type2 = is_transformed and type == "type2" and 'P' in match

    # 如果 is_second_order 为 True，则调用 _second_order_to_first_order 函数转换方程和函数
    if is_second_order:
        new_eqs, new_funcs = _second_order_to_first_order(eqs, funcs, t,
                                                          A1=match.get("A1", None), A0=match.get("A0", None),
                                                          b=match.get("rhs", None), type=type,
                                                          t_=match.get("t_", None))
    else:
        # 否则调用 _higher_order_to_first_order 函数转换方程和函数
        new_eqs, new_funcs = _higher_order_to_first_order(eqs, sysorder, t, funcs=funcs,
                                                          type=type, J=match.get('J', None),
                                                          f_t=match.get('f(t)', None),
                                                          P=match.get('P', None), b=match.get('rhs', None))

    # 如果 is_transformed 为 True，则将 t 更新为匹配对象中的 't_'，否则保持不变
    if is_transformed:
        t = match.get('t_', t)

    # 如果不是 is_higher_order_type2，则从 new_eqs 中选择方程，要求其导数为 funcs 中的函数
    if not is_higher_order_type2:
        new_eqs = _select_equations(new_eqs, [f.diff(t) for f in new_funcs])

    # 初始化解为 None
    sol = None

    # 尝试使用 _strong_component_solver 解方程组 new_eqs
    # 如果抛出 NotImplementedError，则将解设为 None
    try:
        if not is_higher_order_type2:
            sol = _strong_component_solver(new_eqs, new_funcs, t)
    except NotImplementedError:
        sol = None

    # 当 sol 为 None 时，尝试使用 _component_solver 解方程组 new_eqs
    # 如果抛出 NotImplementedError，则将解设为 None
    if sol is None:
        try:
            sol = _component_solver(new_eqs, new_funcs, t)
        except NotImplementedError:
            sol = None

    # 如果最终 sol 仍然为 None，则返回 None
    if sol is None:
        return sol

    # 如果 is_second_order 为 True，并且 type 等于 "type2"
    is_second_order_type2 = is_second_order and type == "type2"

    # 根据 is_transformed 的值选择 underscores 的值为 '__' 或 '_'
    underscores = '__' if is_transformed else '_'

    # 选择方程组 sol 中的方程，并将它们重新定义为函数形式，以便后续处理
    sol = _select_equations(sol, funcs,
                            key=lambda x: Function(Dummy('{}{}0'.format(x.func.__name__, underscores)))(t))

    # 如果匹配对象中的 "is_transformed" 为 True
    if match.get("is_transformed", False):
        # 如果是 is_second_order_type2
        if is_second_order_type2:
            g_t = match["g(t)"]
            tau = match["tau"]
            # 对 sol 中的每个方程进行变换，使用 g_t 和 tau
            sol = [Eq(s.lhs, s.rhs.subs(t, tau) * g_t) for s in sol]
        # 如果是 is_euler
        elif is_euler:
            t = match['t']
            tau = match['t_']
            # 对 sol 中的每个方程进行变换，使用 log 函数
            sol = [s.subs(tau, log(t)) for s in sol]
        # 如果是 is_higher_order_type2
        elif is_higher_order_type2:
            P = match['P']
            sol_vector = P * Matrix([s.rhs for s in sol])
            # 将 sol 重新定义为函数和变换后的方程的对应关系
            sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]

    # 返回处理后的 sol
    return sol
# Returns: List of equations or None
# If None is returned by this solver, then the system
# of ODEs cannot be solved directly by dsolve_system.
def _strong_component_solver(eqs, funcs, t):
    from sympy.solvers.ode.ode import dsolve, constant_renumber

    # Classify the system of ODEs and identify its properties
    match = _classify_linear_system(eqs, funcs, t, is_canon=True)
    sol = None

    # Check if the system matches any specific classification
    if match:
        match['t'] = t

        # Solve higher order ODEs if detected
        if match.get('is_higher_order', False):
            sol = _higher_order_ode_solver(match)

        # Solve linear ODEs if detected
        elif match.get('is_linear', False):
            sol = _linear_ode_solver(match)

        # Handle single equation if no specific match found
        if sol is None and len(eqs) == 1:
            # Attempt to solve the equation using sympy's dsolve
            sol = dsolve(eqs[0], func=funcs[0])
            # Generate new constants for the solution
            variables = Tuple(eqs[0]).free_symbols
            new_constants = [Dummy() for _ in range(ode_order(eqs[0], funcs[0]))]
            sol = constant_renumber(sol, variables=variables, newconstants=new_constants)
            sol = [sol]

        # Non-linear case handling could be added in the future

    return sol


# Returns: List of Equations(a solution)
def _weak_component_solver(wcc, t):

    # Divide the weakly connected components (wcc) into individual systems
    eqs = []
    for scc in wcc:
        eqs += scc
    funcs = _get_funcs_from_canon(eqs)

    # Solve the system using the strong component solver
    sol = _strong_component_solver(eqs, funcs, t)
    if sol:
        return sol

    sol = []

    # Iterate over each strongly connected component (scc)
    for scc in wcc:
        eqs = scc
        funcs = _get_funcs_from_canon(eqs)

        # Substitute solutions from previous components into current equations
        comp_eqs = [eq.subs({s.lhs: s.rhs for s in sol}) for eq in eqs]
        scc_sol = _strong_component_solver(comp_eqs, funcs, t)

        # If the system cannot be solved, raise an error
        if scc_sol is None:
            raise NotImplementedError(filldedent('''
                The system of ODEs passed cannot be solved by dsolve_system.
            '''))

        # Accumulate solutions for each component
        sol += scc_sol

    return sol


# Returns: List of Equations(a solution)
def _component_solver(eqs, funcs, t):
    # Divide the system into components based on connectivity
    components = _component_division(eqs, funcs, t)
    sol = []

    # Iterate over each component and solve using the weak component solver
    for wcc in components:
        sol += _weak_component_solver(wcc, t)

    return sol


def _second_order_to_first_order(eqs, funcs, t, type="auto", A1=None,
                                 A0=None, b=None, t_=None):
    r"""
    Expects the system to be in second order and in canonical form

    Explanation
    ===========

    Reduces a second order system into a first order one depending on the type of second
    order system.
    ```
    """
    "type0": 如果传入此参数，则引入虚拟变量将系统降阶为一阶。
    
    "type1": 如果传入此参数，则使用特定的替换将系统降阶为一阶。
    
    "type2": 如果传入此参数，则通过引入新的因变量和自变量对系统进行转换。这种转换是解决对应常微分方程系统的一部分。
    
    `A1` 和 `A0` 是系统的系数矩阵，假设二阶系统的形式如下所示：
    
    .. math::
        A2 * X'' = A1 * X' + A0 * X + b
    
    这里，$A2$ 是向量 $X''$ 的系数矩阵，$b$ 是非齐次项。
    
    如果 `b` 的默认值为 None，但如果传入了 `A1` 和 `A0` 而未传入 `b`，则系统将被假定为齐次的。
    
    """
    
    # 检查是否 A1 和 A0 为 None
    is_a1 = A1 is None
    is_a0 = A0 is None
    
    # 根据不同情况进行处理
    if (type == "type1" and is_a1) or (type == "type2" and is_a0) or (type == "auto" and (is_a1 or is_a0)):
        # 将二阶线性方程组转换为矩阵形式
        (A2, A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, 2)
        
        # 如果 A2 不是单位矩阵，抛出错误
        if not A2.is_Identity:
            raise ValueError(filldedent('''
                The system must be in its canonical form.
            '''))
    
    # 如果 type 是 "auto"，根据二阶方程的类型进行匹配
    if type == "auto":
        match = _match_second_order_type(A1, A0, t)
        type = match["type_of_equation"]
        A1 = match.get("A1", None)
        A0 = match.get("A0", None)
    
    # 初始化系统的阶数为二阶
    sys_order = dict.fromkeys(funcs, 2)
    
    # 根据 type 的不同情况进行处理
    if type == "type1":
        # 如果 b 为 None，则初始化为长度与方程数量相同的零向量
        if b is None:
            b = zeros(len(eqs))
        # 使用 type1 替换进行二阶系统降阶为一阶
        eqs = _second_order_subs_type1(A1, b, funcs, t)
        sys_order = dict.fromkeys(funcs, 1)
    
    if type == "type2":
        # 如果 t_ 为 None，则将其初始化为新的符号变量
        if t_ is None:
            t_ = Symbol("{}_".format(t))
        t = t_
        # 使用 type2 替换进行二阶系统降阶为一阶
        eqs, funcs = _second_order_subs_type2(A0, funcs, t_)
        sys_order = dict.fromkeys(funcs, 2)
    
    # 将降阶后的方程组转换为一阶形式
    return _higher_order_to_first_order(eqs, sys_order, t, funcs=funcs)
# 将高阶类型2系统转换为子系统方程组
def _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, b=None, P=None):

    # 注意：为此 ValueError 添加测试
    # 检查参数 J、f_t 是否为 None 或矩阵 J 在时间 t 上是否非常量
    if J is None or f_t is None or not _matrix_is_constant(J, t):
        raise ValueError(filldedent('''
            Correctly input for args 'A' and 'f_t' for Linear, Higher Order,
            Type 2
        '''))

    # 如果 P 为 None，但 b 不为 None 且 b 不是零矩阵，则引发 ValueError
    if P is None and b is not None and not b.is_zero_matrix:
        raise ValueError(filldedent('''
            Provide the keyword 'P' for matrix P in A = P * J * P-1.
        '''))

    # 创建新的函数列表，根据原始函数 funcs 在时间 t 的值构造新函数
    new_funcs = Matrix([Function(Dummy('{}__0'.format(f.func.__name__)))(t) for f in funcs])
    
    # 构造新的方程组，将高阶方程转换为一阶方程
    new_eqs = new_funcs.diff(t, max_order) - f_t * J * new_funcs

    # 如果 b 不为 None 且不是零矩阵，则在方程中减去 P 的逆矩阵乘以 b
    if b is not None and not b.is_zero_matrix:
        new_eqs -= P.inv() * b

    # 将新方程组转换为规范形式
    new_eqs = canonical_odes(new_eqs, new_funcs, t)[0]

    # 返回转换后的新方程组和新函数列表
    return new_eqs, new_funcs


# 将高阶微分方程组转换为一阶方程组
def _higher_order_to_first_order(eqs, sys_order, t, funcs=None, type="type0", **kwargs):
    # 如果未指定 funcs，则使用 sys_order 中的函数列表
    if funcs is None:
        funcs = sys_order.keys()

    # 标准的 Cauchy Euler 系统
    # 如果类型为 "type1"，进行以下处理：
    if type == "type1":
        # 创建一个以 t 命名的符号对象 t_
        t_ = Symbol('{}_'.format(t))
        # 对于 funcs 中的每个函数 f，创建一个新的函数对象，并传入 t_ 作为参数
        new_funcs = [Function(Dummy('{}_'.format(f.func.__name__)))(t_) for f in funcs]
        # 计算 funcs 中每个函数的系统阶数的最大值
        max_order = max(sys_order[func] for func in funcs)
        # 创建一个替换字典，将 funcs 中的函数映射到 new_funcs 中对应的函数，并将 t 映射到 exp(t_)
        subs_dict = dict(zip(funcs, new_funcs))
        subs_dict[t] = exp(t_)

        # 创建一个未命名的自由函数对象
        free_function = Function(Dummy())

        # 定义一个函数，用于从替换后的表达式中获取系数
        def _get_coeffs_from_subs_expression(expr):
            # 如果表达式是 Subs 类型，则返回其表达式中的阶数为 1 的字典
            if isinstance(expr, Subs):
                free_symbol = expr.args[1][0]
                term = expr.args[0]
                return {ode_order(term, free_symbol): 1}

            # 如果表达式是 Mul 类型，则返回其第一个参数作为系数，第二个参数作为阶数
            if isinstance(expr, Mul):
                coeff = expr.args[0]
                order = list(_get_coeffs_from_subs_expression(expr.args[1]).keys())[0]
                return {order: coeff}

            # 如果表达式是 Add 类型，则遍历其每个参数
            if isinstance(expr, Add):
                coeffs = {}
                for arg in expr.args:
                    # 如果参数是 Mul 类型，则更新 coeffs 字典
                    if isinstance(arg, Mul):
                        coeffs.update(_get_coeffs_from_subs_expression(arg))
                    # 否则，获取其阶数并将其系数设为 1
                    else:
                        order = list(_get_coeffs_from_subs_expression(arg).keys())[0]
                        coeffs[order] = 1
                return coeffs

        # 对于每个阶数 o 从 1 到 max_order + 1
        for o in range(1, max_order + 1):
            # 构造一个表达式，计算自由函数 free_function 在 t_ 上对 t_ 的 o 阶导数乘以 t_ 的 o 次幂
            expr = free_function(log(t_)).diff(t_, o)*t_**o
            # 获取这个表达式中的系数字典
            coeff_dict = _get_coeffs_from_subs_expression(expr)
            # 根据阶数 o 构造系数列表 coeffs
            coeffs = [coeff_dict[order] if order in coeff_dict else 0 for order in range(o + 1)]
            # 构造用于替换的表达式 expr_to_subs
            expr_to_subs = sum(free_function(t_).diff(t_, i) * c for i, c in
                        enumerate(coeffs)) / t**o
            # 更新 subs_dict，将 funcs 中每个函数的 t 的 o 阶导数映射到 expr_to_subs
            subs_dict.update({f.diff(t, o): expr_to_subs.subs(free_function(t_), nf)
                              for f, nf in zip(funcs, new_funcs)})

        # 将 eqs 中的每个方程都应用 subs_dict 进行替换，得到 new_eqs
        new_eqs = [eq.subs(subs_dict) for eq in eqs]
        # 构造一个字典，将 funcs 映射到 new_funcs 中对应函数的系统阶数
        new_sys_order = {nf: sys_order[f] for f, nf in zip(funcs, new_funcs)}

        # 将 new_eqs 转换为标准的一阶常微分方程形式
        new_eqs = canonical_odes(new_eqs, new_funcs, t_)[0]

        # 调用函数将高阶微分方程组转换为一阶微分方程组并返回结果
        return _higher_order_to_first_order(new_eqs, new_sys_order, t_, funcs=new_funcs)

    # 如果类型为 "type2"，进行以下处理：
    # 对于形如 X(n)(t) = f(t)*A*X + b 的系统
    # 其中 X(n)(t) 是依赖变量向量关于自变量的第 n 阶导数，A 是常数矩阵
    if type == "type2":
        # 从 kwargs 中获取 J, f_t, b, P，并赋给相应变量
        J = kwargs.get('J', None)
        f_t = kwargs.get('f_t', None)
        b = kwargs.get('b', None)
        P = kwargs.get('P', None)
        # 计算 funcs 中每个函数的系统阶数的最大值
        max_order = max(sys_order[func] for func in funcs)

        # 调用函数将 type2 高阶微分方程组转换为子系统并返回结果
        return _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, P=P, b=b)

        # 注意：在默认情况下禁用 doit 选项后，应更改为以下代码
        # new_sysorder = _get_func_order(new_eqs, new_funcs)
        #
        # return _higher_order_to_first_order(new_eqs, new_sysorder, t, funcs=new_funcs)

    # 如果 type 不是 "type1" 或 "type2"，将 new_funcs 设为空列表
    new_funcs = []
    # 遍历给定的函数列表 funcs
    for prev_func in funcs:
        # 获取当前函数对象的名称
        func_name = prev_func.func.__name__
        # 创建一个新的函数对象 func，以当前时间 t 为参数
        func = Function(Dummy('{}_0'.format(func_name)))(t)
        # 将新函数对象添加到新函数列表 new_funcs 中
        new_funcs.append(func)
        # 创建一个替换字典，将 prev_func 映射到 func
        subs_dict = {prev_func: func}
        # 初始化一个空的新方程列表
        new_eqs = []

        # 对于当前函数 prev_func 的阶数 sys_order[prev_func] 中的每一个阶数 i
        for i in range(1, sys_order[prev_func]):
            # 创建一个新的函数对象 new_func，带有当前时间 t 的索引 i
            new_func = Function(Dummy('{}_{}'.format(func_name, i)))(t)
            # 将 prev_func 的 i 阶导数映射到 new_func
            subs_dict[prev_func.diff(t, i)] = new_func
            # 将新的函数对象 new_func 添加到新函数列表 new_funcs 中
            new_funcs.append(new_func)

            # 获取前一个函数 prev_f，用来构造新的方程
            prev_f = subs_dict[prev_func.diff(t, i-1)]
            # 创建一个新的方程 Eq，表示 prev_f 对时间 t 的导数等于 new_func
            new_eq = Eq(prev_f.diff(t), new_func)
            # 将新方程 new_eq 添加到新方程列表 new_eqs 中
            new_eqs.append(new_eq)

        # 使用替换字典 subs_dict 替换原始方程列表 eqs 中的所有方程
        eqs = [eq.subs(subs_dict) for eq in eqs] + new_eqs

    # 返回更新后的方程列表 eqs 和新函数列表 new_funcs
    return eqs, new_funcs
# 定义一个函数，用于求解任意（支持的）常微分方程组

# 导入必要的库和函数
def dsolve_system(eqs, funcs=None, t=None, ics=None, doit=False, simplify=True):
    r"""
    解决任意（支持的）常微分方程组

    说明
    ====

    此函数接受一个常微分方程组作为输入，确定是否可以通过此函数求解，并在找到解时返回解。

    此函数可以处理以下类型的系统：
    1. 线性、一阶、常系数齐次微分方程组
    2. 线性、一阶、常系数非齐次微分方程组
    3. 线性、一阶、非常数系数齐次微分方程组
    4. 线性、一阶、非常数系数非齐次微分方程组
    5. 可分解为上述4种形式微分方程组的任何隐式系统
    6. 可简化为上述5种形式的高阶线性微分方程组

    上述类型的系统不受方程数目的限制，即此函数可以解决上述类型的系统，而不论系统中包含多少方程。
    但是，系统越大，求解系统所需的时间就越长。

    此函数返回一个解的列表。每个解都是一个方程列表，其中左侧是因变量，右侧是自变量的表达式。

    在非常数系数类型中，并非所有系统都可以通过此函数求解。仅当系数矩阵具有可交换的反导数或者可以进一步分解以便分解系统具有具有可交换的反导数的系数矩阵时，才可以求解。

    参数
    ==========

    eqs : List
        待求解的微分方程组
    funcs : List 或 None
        构成微分方程组的因变量列表
    t : Symbol 或 None
        微分方程组中的自变量
    ics : Dict 或 None
        微分方程组的初始边界/条件集合
    doit : 布尔值
        如果为 True，则评估解。默认值为 True。如果积分评估耗时过长或不需要，则可设置为 False。
    simplify: 布尔值
        简化系统的解。默认值为 True。如果简化耗时过长或不需要，则可设置为 False。

    示例
    ========

    >>> from sympy import symbols, Eq, Function
    >>> from sympy.solvers.ode.systems import dsolve_system
    >>> f, g = symbols("f g", cls=Function)
    >>> x = symbols("x")

    >>> eqs = [Eq(f(x).diff(x), g(x)), Eq(g(x).diff(x), f(x))]
    >>> dsolve_system(eqs)
    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]

    也可以为微分方程组传递初始条件：

    >>> dsolve_system(eqs, ics={f(0): 1, g(0): 0})
    [[Eq(f(x), exp(x)/2 + exp(-x)/2), Eq(g(x), exp(x)/2 - exp(-x)/2)]]
    """
    Optionally, you can pass the dependent variables and the independent
    variable for which the system is to be solved:
    
    >>> funcs = [f(x), g(x)]
    >>> dsolve_system(eqs, funcs=funcs, t=x)
    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]
    
    Lets look at an implicit system of ODEs:
    
    >>> eqs = [Eq(f(x).diff(x)**2, g(x)**2), Eq(g(x).diff(x), g(x))]
    >>> dsolve_system(eqs)
    [[Eq(f(x), C1 - C2*exp(x)), Eq(g(x), C2*exp(x))], [Eq(f(x), C1 + C2*exp(x)), Eq(g(x), C2*exp(x))]]
    
    Returns
    =======
    
    List of List of Equations
    
    Raises
    ======
    
    NotImplementedError
        When the system of ODEs is not solvable by this function.
    ValueError
        When the parameters passed are not in the required form.
    """
    from sympy.solvers.ode.ode import solve_ics, _extract_funcs, constant_renumber
    
    if not iterable(eqs):
        raise ValueError(filldedent('''
            List of equations should be passed. The input is not valid.
        '''))
    
    eqs = _preprocess_eqs(eqs)
    
    if funcs is not None and not isinstance(funcs, list):
        raise ValueError(filldedent('''
            Input to the funcs should be a list of functions.
        '''))
    
    if funcs is None:
        funcs = _extract_funcs(eqs)
    
    if any(len(func.args) != 1 for func in funcs):
        raise ValueError(filldedent('''
            dsolve_system can solve a system of ODEs with only one independent
            variable.
        '''))
    
    if len(eqs) != len(funcs):
        raise ValueError(filldedent('''
            Number of equations and number of functions do not match
        '''))
    
    if t is not None and not isinstance(t, Symbol):
        raise ValueError(filldedent('''
            The independent variable must be of type Symbol
        '''))
    
    if t is None:
        t = list(list(eqs[0].atoms(Derivative))[0].atoms(Symbol))[0]
    
    sols = []
    canon_eqs = canonical_odes(eqs, funcs, t)
    
    for canon_eq in canon_eqs:
        try:
            sol = _strong_component_solver(canon_eq, funcs, t)
        except NotImplementedError:
            sol = None
    
        if sol is None:
            sol = _component_solver(canon_eq, funcs, t)
    
        sols.append(sol)
    
    if sols:
        final_sols = []
        variables = Tuple(*eqs).free_symbols
    
        for sol in sols:
            sol = _select_equations(sol, funcs)
            sol = constant_renumber(sol, variables=variables)
    
            if ics:
                constants = Tuple(*sol).free_symbols - variables
                solved_constants = solve_ics(sol, funcs, constants, ics)
                sol = [s.subs(solved_constants) for s in sol]
    
            if simplify:
                constants = Tuple(*sol).free_symbols - variables
                sol = simpsol(sol, [t], constants, doit=doit)
    
            final_sols.append(sol)
    
        sols = final_sols
    
    return sols
```