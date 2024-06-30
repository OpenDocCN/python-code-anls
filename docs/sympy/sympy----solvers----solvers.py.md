# `D:\src\scipysrc\sympy\sympy\solvers\solvers.py`

```
"""
This module contain solvers for all kinds of equations:

    - algebraic or transcendental, use solve()

    - recurrence, use rsolve()

    - differential, use dsolve()

    - nonlinear (numerically), use nsolve()
      (you will need a good starting point)

"""
# 导入 __future__ 模块，启用 Python 3.x 的特性支持
from __future__ import annotations

# 导入 sympy 的核心模块
from sympy.core import (S, Add, Symbol, Dummy, Expr, Mul)
# 导入用于检查假设的模块
from sympy.core.assumptions import check_assumptions
# 导入表达式工具模块，用于因式分解等操作
from sympy.core.exprtools import factor_terms
# 导入函数相关的核心模块
from sympy.core.function import (expand_mul, expand_log, Derivative,
                                 AppliedUndef, UndefinedFunction, nfloat,
                                 Function, expand_power_exp, _mexpand, expand,
                                 expand_func)
# 导入逻辑操作相关的核心模块
from sympy.core.logic import fuzzy_not
# 导入数字相关的核心模块
from sympy.core.numbers import Float, Rational, _illegal
# 导入整数函数相关的核心模块
from sympy.core.intfunc import integer_log, ilcm
# 导入幂函数相关的核心模块
from sympy.core.power import Pow
# 导入关系表达式相关的核心模块
from sympy.core.relational import Eq, Ne
# 导入排序和比较相关的核心模块
from sympy.core.sorting import ordered, default_sort_key
# 导入 sympify 函数，用于将字符串转换为 SymPy 表达式
from sympy.core.sympify import sympify, _sympify
# 导入遍历相关的核心模块
from sympy.core.traversal import preorder_traversal
# 导入布尔代数相关的模块
from sympy.logic.boolalg import And, BooleanAtom

# 导入函数模块，包括对数、指数等
from sympy.functions import (log, exp, LambertW, cos, sin, tan, acos, asin, atan,
                             Abs, re, im, arg, sqrt, atan2)
# 导入组合数学中阶乘相关的模块
from sympy.functions.combinatorial.factorials import binomial
# 导入双曲函数相关的模块
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
# 导入分段函数相关的模块
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
# 导入三角函数相关的模块
from sympy.functions.elementary.trigonometric import TrigonometricFunction
# 导入积分相关的模块
from sympy.integrals.integrals import Integral
# 导入数论相关的模块
from sympy.ntheory.factor_ import divisors
# 导入简化表达式相关的模块
from sympy.simplify import (simplify, collect, powsimp, posify,  # type: ignore
    powdenest, nsimplify, denom, logcombine, sqrtdenest, fraction,
    separatevars)
# 导入平方根简化相关的模块
from sympy.simplify.sqrtdenest import sqrt_depth
# 导入函数简化相关的模块
from sympy.simplify.fu import TR1, TR2i, TR10, TR11
# 导入重建相关的模块
from sympy.strategies.rl import rebuild
# 导入矩阵相关的异常模块
from sympy.matrices.exceptions import NonInvertibleMatrixError
# 导入矩阵和零矩阵相关的模块
from sympy.matrices import Matrix, zeros
# 导入多项式相关的模块，包括根、因式分解等
from sympy.polys import roots, cancel, factor, Poly
# 导入多项式求解相关的模块
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys
# 导入多项式相关的异常模块
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
# 导入多项式工具相关的模块，如最大公约数
from sympy.polys.polytools import gcd
# 导入 lambdify 函数，用于将 SymPy 表达式转换为可调用的 Python 函数
from sympy.utilities.lambdify import lambdify
# 导入实用工具相关的模块，包括文本格式化、调试等
from sympy.utilities.misc import filldedent, debugf
# 导入可迭代工具相关的模块，如生成子集等
from sympy.utilities.iterables import (connected_components,
    generate_bell, uniq, iterable, is_sequence, subsets, flatten, sift)
# 导入保留 mpmath 精度设置的装饰器
from sympy.utilities.decorator import conserve_mpmath_dps

# 导入 mpmath 中的 findroot 函数，用于数值求根
from mpmath import findroot

# 导入 sympy 中解多项式系统方程的函数
from sympy.solvers.polysys import solve_poly_system

# 导入 GeneratorType 类型，用于检查生成器对象
from types import GeneratorType
# 导入 defaultdict 类型，用于创建默认字典
from collections import defaultdict
# 导入 combinations 和 product 函数，用于生成组合和笛卡尔积
from itertools import combinations, product

# 导入警告模块，用于管理警告信息的显示
import warnings


def recast_to_symbols(eqs, symbols):
    """
    Return (e, s, d) where e and s are versions of *eqs* and
    *symbols* in which any non-Symbol objects in *symbols* have
    been replaced with symbols and dummy symbols have been used
    for undefined functions.

    """
    # 返回三个元素的元组，其中 e 和 s 是替换后的表达式和符号集合，d 是对未定义函数使用虚拟符号的结果
    return (eqs, symbols, defaultdict(Dummy, [(s, Symbol(s)) for s in symbols if not isinstance(s, Symbol)]))
    # 如果 eqs 不可迭代或 symbols 可迭代，则抛出 ValueError 异常
    if not iterable(eqs) and iterable(symbols):
        raise ValueError('Both eqs and symbols must be iterable')
    # 将 symbols 转换为有序列表，并保存原始符号列表
    orig = list(symbols)
    symbols = list(ordered(symbols))
    # 创建用于替换的 Dummy 符号字典 swap_sym
    swap_sym = {}
    i = 0
    # 遍历 symbols 列表，为非 Symbol 类型的元素或尚未在 swap_sym 中的元素创建 Dummy 符号
    for s in symbols:
        if not isinstance(s, Symbol) and s not in swap_sym:
            swap_sym[s] = Dummy('X%d' % i)
            i += 1
    # 创建新的表达式列表 new_f，将原始表达式中的符号用 swap_sym 中的 Dummy 符号替换
    new_f = []
    for i in eqs:
        isubs = getattr(i, 'subs', None)
        if isubs is not None:
            new_f.append(isubs(swap_sym))
        else:
            new_f.append(i)
    # 创建还原字典 restore，用于将 swap_sym 中的 Dummy 符号恢复为原始符号
    restore = {v: k for k, v in swap_sym.items()}
    # 返回替换后的新表达式列表 new_f、替换后的符号列表和恢复字典 restore
    return new_f, [swap_sym.get(i, i) for i in orig], restore
# 检查表达式 e 是否为 Pow 类型或 exp 类型，如果是则返回 True
def _ispow(e):
    """Return True if e is a Pow or is exp."""
    return isinstance(e, Expr) and (e.is_Pow or isinstance(e, exp))


# 简化函数，返回表达式 f 中所有分母的集合
def _simple_dens(f, symbols):
    # 当检查分母是否为零时，我们可以只检查具有非零指数的幂次的基数，
    # 因为如果基数为零，那么幂次也将为零。为了保持简单和快速，我们
    # 限制简化到指数为 Numbers 的情况
    dens = set()
    for d in denoms(f, symbols):
        if d.is_Pow and d.exp.is_Number:
            if d.exp.is_zero:
                continue  # foo**0 永远不为 0
            d = d.base
        dens.add(d)
    return dens


# 返回表达式 eq 中包含任何符号 symbols 的所有分母的集合
def denoms(eq, *symbols):
    """
    Return (recursively) set of all denominators that appear in *eq*
    that contain any symbol in *symbols*; if *symbols* are not
    provided then all denominators will be returned.
    
    如果提供了 *symbols*，则只返回包含这些符号的分母：
    - 如果 *symbols* 是空的，则返回所有分母。
    - 如果 *symbols* 是一个符号的集合，则检查是否包含这些符号。
    """

    pot = preorder_traversal(eq)
    dens = set()
    for p in pot:
        # 这里 p 可能是 Tuple 或 Relational
        # 在 pot 之后会遍历 Expr 的子树（如 lhs 和 rhs）
        if not isinstance(p, Expr):
            continue
        den = denom(p)
        if den is S.One:
            continue
        dens.update(Mul.make_args(den))
    if not symbols:
        return dens
    elif len(symbols) == 1:
        if iterable(symbols[0]):
            symbols = symbols[0]
    return {d for d in dens if any(s in d.free_symbols for s in symbols)}


# 检查是否 sol 是方程 f == 0 的解
def checksol(f, symbol, sol=None, **flags):
    """
    Checks whether sol is a solution of equation f == 0.

    输入可以是单个符号及其对应的值，也可以是符号及值的字典。
    如果给定了字典并且标志 ``simplify=True``，则字典中的值将被简化。
    *f* 可以是单个方程或方程组。一个解必须满足 *f* 中的所有方程才能被认为是有效的；
    如果一个解不满足任何一个方程，返回 False；如果一个或多个检查不确定（且没有 False），则返回 None。
    """
    `
        # 导入单位模块中的 Unit 类
        from sympy.physics.units import Unit
    
        # 从 flags 参数中获取 'minimal' 的值，默认为 False
        minimal = flags.get('minimal', False)
    
        # 如果 sol 不为 None，则将其转换为字典形式 {symbol: sol}
        if sol is not None:
            sol = {symbol: sol}
        # 如果 symbol 是字典类型，则直接将其赋给 sol
        elif isinstance(symbol, dict):
            sol = symbol
        else:
            # 抛出值错误异常，提示预期格式不匹配
            msg = 'Expecting (sym, val) or ({sym: val}, None) but got (%s, %s)'
            raise ValueError(msg % (symbol, sol))
    
        # 如果 f 是可迭代对象
        if iterable(f):
            # 如果 f 为空，则抛出值错误异常
            if not f:
                raise ValueError('no functions to check')
            # 初始化返回值 rv 为 True
            rv = True
            # 遍历 f 中的每个函数 fi
            for fi in f:
                # 调用 checksol() 函数检查 fi 是否满足 sol 中的解，传入 flags 参数
                check = checksol(fi, sol, **flags)
                # 如果 check 为真值（True 或非空对象），继续下一个函数检查
                if check:
                    continue
                # 如果 check 为 False，直接返回 False
                if check is False:
                    return False
                # 如果 check 为 None，说明还不能确定结果，将 rv 设为 None
                rv = None  # 不要返回，等待查看是否有 False 结果
            # 返回最终的 rv 值
            return rv
    
        # 将 f 转换为符号表达式
        f = _sympify(f)
    
        # 如果 f 是数值类型
        if f.is_number:
            return f.is_zero
    
        # 如果 f 是多项式类型 Poly
        if isinstance(f, Poly):
            # 将 f 转换为表达式形式
            f = f.as_expr()
        # 如果 f 是方程 Eq 或不等式 Ne
        elif isinstance(f, (Eq, Ne)):
            # 如果 f 的右侧是 S.true 或 S.false，将 f 反转
            if f.rhs in (S.true, S.false):
                f = f.reversed
            # 分别获取方程 f 的左侧 B 和右侧 E
            B, E = f.args
            # 如果 B 是布尔原子（BooleanAtom）类型
            if isinstance(B, BooleanAtom):
                # 将 f 中的符号替换为 sol 中的值
                f = f.subs(sol)
                # 如果结果不是布尔类型，返回 None
                if not f.is_Boolean:
                    return
            # 如果 f 是方程 Eq 类型
            elif isinstance(f, Eq):
                # 将 f 转换为 Add 类型，即 f.lhs + (-f.rhs)
                f = Add(f.lhs, -f.rhs, evaluate=False)
    
        # 如果 f 是布尔原子类型
        if isinstance(f, BooleanAtom):
            return bool(f)
        # 如果 f 不是关系表达式且不为假
        elif not f.is_Relational and not f:
            return True
    
        # 定义非法符号集合
        illegal = set(_illegal)
        # 检查 sol 中每个符号的原子，并与非法符号集合求交集
        if any(sympify(v).atoms() & illegal for k, v in sol.items()):
            return False
    
        # 初始化尝试次数 attempt 为 -1
        attempt = -1
        # 从 flags 参数中获取 'numerical' 的值，默认为 True
        numerical = flags.get('numerical', True)
    # 循环尝试求解过程
    while 1:
        # 尝试次数递增
        attempt += 1
        # 第一次尝试
        if attempt == 0:
            # 使用当前解代入函数，并计算结果
            val = f.subs(sol)
            # 如果结果是乘法表达式，取其独立部分
            if isinstance(val, Mul):
                val = val.as_independent(Unit)[0]
            # 如果结果中包含不合法的符号，则返回 False
            if val.atoms() & illegal:
                return False
        # 第二次尝试
        elif attempt == 1:
            # 如果结果不是数值
            if not val.is_number:
                # 如果结果不是常数，并且包含自由符号，则尝试简化处理
                if not val.is_constant(*list(sol.keys()), simplify=not minimal):
                    return False
                # 执行内容和原始分解
                _, val = val.as_content_primitive()
                # 对数值进行扩展处理
                val = _mexpand(val.as_numer_denom()[0], recursive=True)
        # 第三次尝试
        elif attempt == 2:
            # 如果是最小化处理
            if minimal:
                return
            # 如果设置了简化标志，则对解进行简化处理
            if flags.get('simplify', True):
                for k in sol:
                    sol[k] = simplify(sol[k])
            # 使用简化后的解重新开始，可能不使用失败的扩展形式
            val = simplify(f.subs(sol))
            # 如果设置了强制标志，则强制进行正数化处理
            if flags.get('force', True):
                val, reps = posify(val)
                # 现在可能扩展有效，因此再次尝试并检查
                exval = _mexpand(val, recursive=True)
                # 如果结果为数值，可以立即决定
                if exval.is_number:
                    val = exval
        # 其他情况
        else:
            # 如果没有根号和函数，则不可能为零
            pot = preorder_traversal(expand_mul(val))
            seen = set()
            saw_pow_func = False
            for p in pot:
                if p in seen:
                    continue
                seen.add(p)
                # 如果是幂次方函数且指数不是整数，或者是函数类型
                if p.is_Pow and not p.exp.is_Integer:
                    saw_pow_func = True
                elif p.is_Function:
                    saw_pow_func = True
                elif isinstance(p, UndefinedFunction):
                    saw_pow_func = True
                # 如果发现有幂次方或者函数，则终止检查
                if saw_pow_func:
                    break
            # 如果没有发现幂次方或函数类型，则返回 False
            if saw_pow_func is False:
                return False
            # 如果设置了强制标志，则根据正数假设进行替换
            if flags.get('force', True):
                val = val.subs(reps)
            # 模糊非零检查
            nz = fuzzy_not(val.is_zero)
            if nz is not None:
                # issue 5673: nz 可能为 True 即使实际为 False
                # 以下是一些 hack 以避免误判
                # HACK 1: LambertW (issue 5673)
                if val.is_number and val.has(LambertW):
                    # 如果结果是数值且包含 LambertW，则不评估以验证解决方案
                    return None
                # 可以在此添加其他 HACKs，否则假设 nz 的值是正确的
                return not nz
            # 中断循环
            break
        # 如果结果是有理数，则判断是否等于零
        if val.is_Rational:
            return val == 0
        # 如果数值为真，并且结果为数值类型，则判断其绝对值是否小于阈值
        if numerical and val.is_number:
            return (abs(val.n(18).n(12, chop=True)) < 1e-9) is S.true
    # 如果在 flags 字典中找到 'warn' 键并且其对应的值为 True，则发出警告
    if flags.get('warn', False):
        # 发出警告，指示未能验证解决方案的特定情况
        warnings.warn("\n\tWarning: could not verify solution %s." % sol)
    # 如果无法得出结论，则返回 None
    # TODO: 改进解决方案测试
# 定义函数 solve，用于代数方程和方程组的求解
def solve(f, *symbols, **flags):
    r"""
    Algebraically solves equations and systems of equations.

    Explanation
    ===========

    Currently supported:
        - polynomial
        - transcendental
        - piecewise combinations of the above
        - systems of linear and polynomial equations
        - systems containing relational expressions
        - systems implied by undetermined coefficients

    Examples
    ========

    The default output varies according to the input and might
    be a list (possibly empty), a dictionary, a list of
    dictionaries or tuples, or an expression involving relationals.
    For specifics regarding different forms of output that may appear, see :ref:`solve_output`.
    Let it suffice here to say that to obtain a uniform output from
    `solve` use ``dict=True`` or ``set=True`` (see below).

        >>> from sympy import solve, Poly, Eq, Matrix, Symbol
        >>> from sympy.abc import x, y, z, a, b

    The expressions that are passed can be Expr, Equality, or Poly
    classes (or lists of the same); a Matrix is considered to be a
    list of all the elements of the matrix:

        >>> solve(x - 3, x)
        [3]
        >>> solve(Eq(x, 3), x)
        [3]
        >>> solve(Poly(x - 3), x)
        [3]
        >>> solve(Matrix([[x, x + y]]), x, y) == solve([x, x + y], x, y)
        True

    If no symbols are indicated to be of interest and the equation is
    univariate, a list of values is returned; otherwise, the keys in
    a dictionary will indicate which (of all the variables used in
    the expression(s)) variables and solutions were found:

        >>> solve(x**2 - 4)
        [-2, 2]
        >>> solve((x - a)*(y - b))
        [{a: x}, {b: y}]
        >>> solve([x - 3, y - 1])
        {x: 3, y: 1}
        >>> solve([x - 3, y**2 - 1])
        [{x: 3, y: -1}, {x: 3, y: 1}]

    If you pass symbols for which solutions are sought, the output will vary
    depending on the number of symbols you passed, whether you are passing
    a list of expressions or not, and whether a linear system was solved.
    """
    Uniform output is attained by using ``dict=True`` or ``set=True``.

        >>> #### *** feel free to skip to the stars below *** ####
        >>> from sympy import TableForm  # 导入 sympy 中的 TableForm 函数
        >>> h = [None, ';|;'.join(['e', 's', 'solve(e, s)', 'solve(e, s, dict=True)',
        ... 'solve(e, s, set=True)']).split(';')]  # 定义表头 h，包含列名
        >>> t = []  # 初始化空列表 t 用于存放表格数据
        >>> for e, s in [  # 迭代计算每组方程 e 和符号 s
        ...         (x - y, y),
        ...         (x - y, [x, y]),
        ...         (x**2 - y, [x, y]),
        ...         ([x - 3, y -1], [x, y]),
        ...         ]:
        ...     how = [{}, dict(dict=True), dict(set=True)]  # 定义求解选项列表 how
        ...     res = [solve(e, s, **f) for f in how]  # 求解方程 e 和符号 s 的结果，使用不同的选项
        ...     t.append([e, '|', s, '|'] + [res[0], '|', res[1], '|', res[2]])  # 将每组结果添加到表格数据 t 中
        ...
        >>> # ******************************************************* #
        >>> TableForm(t, headings=h, alignments="<")  # 生成表格并指定表头和对齐方式
        e              | s      | solve(e, s)  | solve(e, s, dict=True) | solve(e, s, set=True)
        ---------------------------------------------------------------------------------------
        x - y          | y      | [x]          | [{y: x}]               | ([y], {(x,)})
        x - y          | [x, y] | [(y, y)]     | [{x: y}]               | ([x, y], {(y, y)})
        x**2 - y       | [x, y] | [(x, x**2)]  | [{y: x**2}]            | ([x, y], {(x, x**2)})
        [x - 3, y - 1] | [x, y] | {x: 3, y: 1} | [{x: 3, y: 1}]         | ([x, y], {(3, 1)})

        * If any equation does not depend on the symbol(s) given, it will be
          eliminated from the equation set and an answer may be given
          implicitly in terms of variables that were not of interest:

            >>> solve([x - y, y - 3], x)
            {x: y}

    When you pass all but one of the free symbols, an attempt
    is made to find a single solution based on the method of
    undetermined coefficients. If it succeeds, a dictionary of values
    is returned. If you want an algebraic solutions for one
    or more of the symbols, pass the expression to be solved in a list:

        >>> e = a*x + b - 2*x - 3  # 定义一个表达式 e
        >>> solve(e, [a, b])  # 解 e 关于 a 和 b 的值，返回一个字典
        {a: 2, b: 3}
        >>> solve([e], [a, b])  # 解 e（作为列表的单个元素）关于 a 和 b 的值，返回一个表达式
        {a: -b/x + (2*x + 3)/x}

    When there is no solution for any given symbol which will make all
    expressions zero, the empty list is returned (or an empty set in
    the tuple when ``set=True``):

        >>> from sympy import sqrt  # 导入 sympy 中的 sqrt 函数
        >>> solve(3, x)  # 对于方程 3 = x，无解，返回空列表
        []
        >>> solve(x - 3, y)  # 对于方程 x - 3 = y，无解，返回空列表
        []
        >>> solve(sqrt(x) + 1, x, set=True)  # 对于方程 sqrt(x) + 1 = 0，返回空元组（因为使用了 set=True）
        ([x], set())

    When an object other than a Symbol is given as a symbol, it is
    isolated algebraically and an implicit solution may be obtained.
    This is mostly provided as a convenience to save you from replacing
    the object with a Symbol and solving for that Symbol. It will only
    work if the specified object can be replaced with a Symbol using the
    subs method:

        >>> from sympy import exp, Function
        >>> f = Function('f')  # 创建一个名为 'f' 的未指定函数对象

        >>> solve(f(x) - x, f(x))  # 解方程 f(x) - x = 0，返回结果 [x]
        [x]
        >>> solve(f(x).diff(x) - f(x) - x, f(x).diff(x))  # 解微分方程 f'(x) - f(x) - x = 0，返回结果 [x + f(x)]
        [x + f(x)]
        >>> solve(f(x).diff(x) - f(x) - x, f(x))  # 解微分方程 f'(x) - f(x) - x = 0，返回结果 [-x + Derivative(f(x), x)]
        [-x + Derivative(f(x), x)]
        >>> solve(x + exp(x)**2, exp(x), set=True)  # 解方程 x + exp(x)^2 = 0，返回结果 ([exp(x)], {(-sqrt(-x),), (sqrt(-x),)})
        ([exp(x)], {(-sqrt(-x),), (sqrt(-x),)})

        >>> from sympy import Indexed, IndexedBase, Tuple
        >>> A = IndexedBase('A')  # 创建一个索引为 'A' 的 IndexedBase 对象
        >>> eqs = Tuple(A[1] + A[2] - 3, A[1] - A[2] + 1)  # 创建一个包含两个等式的元组
        >>> solve(eqs, eqs.atoms(Indexed))  # 解包含 Indexed 对象的等式组，返回结果 {A[1]: 1, A[2]: 2}

        * To solve for a function within a derivative, use :func:`~.dsolve`.  # 若要解导数内的函数，请使用 :func:`~.dsolve`。

    To solve for a symbol implicitly, use implicit=True:

        >>> solve(x + exp(x), x)  # 隐式解 x + exp(x) = 0，返回结果 [-LambertW(1)]
        [-LambertW(1)]
        >>> solve(x + exp(x), x, implicit=True)  # 隐式解 x + exp(x) = 0，返回结果 [-exp(x)]
        [-exp(x)]

    It is possible to solve for anything in an expression that can be
    replaced with a symbol using :obj:`~sympy.core.basic.Basic.subs`:

        >>> solve(x + 2 + sqrt(3), x + 2)  # 解 x + 2 + sqrt(3) = 0，返回结果 [-sqrt(3)]
        [-sqrt(3)]
        >>> solve((x + 2 + sqrt(3), x + 4 + y), y, x + 2)  # 解 (x + 2 + sqrt(3), x + 4 + y) = 0，返回结果 {y: -2 + sqrt(3), x + 2: -sqrt(3)}

        * Nothing heroic is done in this implicit solving so you may end up
          with a symbol still in the solution:

            >>> eqs = (x*y + 3*y + sqrt(3), x + 4 + y)
            >>> solve(eqs, y, x + 2)  # 解等式组 (x*y + 3*y + sqrt(3), x + 4 + y) = 0，返回结果 {y: -sqrt(3)/(x + 3), x + 2: -2*x/(x + 3) - 6/(x + 3) + sqrt(3)/(x + 3)}
            >>> solve(eqs, y*x, x)  # 解等式组 (x*y + 3*y + sqrt(3), x + 4 + y) = 0，返回结果 {x: -y - 4, x*y: -3*y - sqrt(3)}

        * If you attempt to solve for a number, remember that the number
          you have obtained does not necessarily mean that the value is
          equivalent to the expression obtained:

            >>> solve(sqrt(2) - 1, 1)  # 解 sqrt(2) - 1 = 0，返回结果 [sqrt(2)]
            >>> solve(x - y + 1, 1)  # 解 x - y + 1 = 0，返回结果 [x/(y - 1)]
            >>> [_.subs(z, -1) for _ in solve((x - y + 1).subs(-1, z), 1)]  # 将 z 替换为 -1 后重新解 x - y + 1 = 0，返回结果 [-x + y]

    **Additional Examples**

    ``solve()`` with check=True (default) will run through the symbol tags to
    eliminate unwanted solutions. If no assumptions are included, all possible
    solutions will be returned:

        >>> x = Symbol("x")
        >>> solve(x**2 - 1)  # 解 x^2 - 1 = 0，返回结果 [-1, 1]

    By setting the ``positive`` flag, only one solution will be returned:

        >>> pos = Symbol("pos", positive=True)
        >>> solve(pos**2 - 1)  # 解 pos^2 - 1 = 0，返回结果 [1]

    When the solutions are checked, those that make any denominator zero
    are automatically excluded. If you do not want to exclude such solutions,
    then use the check=False option:

        >>> from sympy import sin, limit
        >>> solve(sin(x)/x)  # 解 sin(x)/x = 0，返回结果 [pi]

    If ``check=False``, then a solution to the numerator being zero is found
    but the value of $x = 0$ is a spurious solution since $\sin(x)/x$ has the well
    known limit (without discontinuity) of 1 at $x = 0$:

        >>> solve(sin(x)/x, check=False)
        [0, pi]



# 解决 sin(x)/x 在 x = 0 处的极限，不进行检查条件
>>> solve(sin(x)/x, check=False)
# 返回解为 [0, pi]

    In the following case, however, the limit exists and is equal to the
    value of $x = 0$ that is excluded when check=True:

        >>> eq = x**2*(1/x - z**2/x)
        >>> solve(eq, x)
        []
        >>> solve(eq, x, check=False)
        [0]
        >>> limit(eq, x, 0, '-')
        0
        >>> limit(eq, x, 0, '+')
        0



# 在以下情况中，尽管在 check=True 时排除了 x = 0，但极限存在且等于 x = 0 的值：

# 创建表达式 eq
>>> eq = x**2*(1/x - z**2/x)
# 解决方程 eq 关于 x 的解
>>> solve(eq, x)
# 返回空列表 []

# 不进行检查条件，解决方程 eq 关于 x 的解
>>> solve(eq, x, check=False)
# 返回解为 [0]

# 计算 eq 在 x = 0 时从左侧的极限
>>> limit(eq, x, 0, '-')
# 返回值为 0

# 计算 eq 在 x = 0 时从右侧的极限
>>> limit(eq, x, 0, '+')
# 返回值为 0

    **Solving Relationships**

    When one or more expressions passed to ``solve`` is a relational,
    a relational result is returned (and the ``dict`` and ``set`` flags
    are ignored):



# 解决关系表达式时，``solve`` 返回关系结果（忽略 ``dict`` 和 ``set`` 标志）：

# 解决 x < 3 的关系
>>> solve(x < 3)
# 返回关系 (-oo < x) & (x < 3)

# 解决多个关系表达式 [x < 3, x**2 > 4] 关于 x 的解
>>> solve([x < 3, x**2 > 4], x)
# 返回关系 ((-oo < x) & (x < -2)) | ((2 < x) & (x < 3))

# 解决多个关系表达式 [x + y - 3, x > 3] 关于 x 的解
>>> solve([x + y - 3, x > 3], x)
# 返回关系 (3 < x) & (x < oo) & Eq(x, 3 - y)



# 虽然不检查关系中符号的假设，但设置假设会影响某些关系的自动简化：

# 解决 x**2 > 4 的关系
>>> solve(x**2 > 4)
# 返回关系 ((-oo < x) & (x < -2)) | ((2 < x) & (x < oo))

# 定义实数符号 r
>>> r = Symbol('r', real=True)
# 解决 r**2 > 4 的关系
>>> solve(r**2 > 4)
# 返回关系 (2 < r) | (r < -2)

# 当解决多个变量时，目前 SymPy 中没有算法允许使用关系解决多于一个变量的情况：

# 定义实数符号 r 和 q
>>> r, q = symbols('r, q', real=True)
# 解决多个关系表达式 [r + q - 3, r > 3] 关于 r 的解
>>> solve([r + q - 3, r > 3], r)
# 返回关系 (3 < r) & Eq(r, 3 - q)



# 可以直接调用 ``solve`` 遇到关系时调用的例程：:func:`~.reduce_inequalities`。
# 它将 Expr 视为等式：

# 导入 reduce_inequalities 函数
>>> from sympy import reduce_inequalities
# 解决不等式表达式 [x**2 - 4]
>>> reduce_inequalities([x**2 - 4])
# 返回等式 Eq(x, -2) | Eq(x, 2)

# 如果每个关系只包含一个感兴趣的符号，可以处理多个符号的表达式：

# 解决不等式表达式 [0 <= x  - 1, y < 3] 关于 [x, y] 的解
>>> reduce_inequalities([0 <= x  - 1, y < 3], [x, y])
# 返回关系 (-oo < y) & (1 <= x) & (x < oo) & (y < 3)

# 如果任何关系有多个感兴趣的符号，会引发错误：

# 解决不等式表达式 [0 <= x*y  - 1, y < 3] 关于 [x, y] 的解
>>> reduce_inequalities([0 <= x*y  - 1, y < 3], [x, y])
# 引发 NotImplementedError: inequality has more than one symbol of interest.

    **Disabling High-Order Explicit Solutions**

    When solving polynomial expressions, you might not want explicit solutions
    (which can be quite long). If the expression is univariate, ``CRootOf``



# 当解决多项式表达式时，可能不希望获得显式解（可能会非常长）。如果表达式是单变量的，``CRootOf``
    instances will be returned instead:  # 如果解是符号表达式，将会返回它们的实例

        >>> solve(x**3 - x + 1)  # 解方程 x**3 - x + 1
        [-1/((-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)) -
        (-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3,
        -(-1/2 + sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3 -
        1/((-1/2 + sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)),
        -(3*sqrt(69)/2 + 27/2)**(1/3)/3 -
        1/(3*sqrt(69)/2 + 27/2)**(1/3)]
        >>> solve(x**3 - x + 1, cubics=False)  # 解方程 x**3 - x + 1，不使用立方根求解
        [CRootOf(x**3 - x + 1, 0),
         CRootOf(x**3 - x + 1, 1),
         CRootOf(x**3 - x + 1, 2)]

    If the expression is multivariate, no solution might be returned:  # 如果表达式是多元的，则可能不返回解

        >>> solve(x**3 - x + a, x, cubics=False)  # 解方程 x**3 - x + a，指定变量为 x，不使用立方根求解
        []

    Sometimes solutions will be obtained even when a flag is False because the
    expression could be factored. In the following example, the equation can
    be factored as the product of a linear and a quadratic factor so explicit
    solutions (which did not require solving a cubic expression) are obtained:

        >>> eq = x**3 + 3*x**2 + x - 1
        >>> solve(eq, cubics=False)  # 解方程 eq，不使用立方根求解
        [-1, -1 + sqrt(2), -sqrt(2) - 1]

    **Solving Equations Involving Radicals**  # 解决涉及根式的方程

    Because of SymPy's use of the principle root, some solutions
    to radical equations will be missed unless check=False:  # 由于 SymPy 使用主根，除非 check=False，否则将错过某些根式方程的解

        >>> from sympy import root
        >>> eq = root(x**3 - 3*x**2, 3) + 1 - x
        >>> solve(eq)  # 解方程 eq
        []
        >>> solve(eq, check=False)  # 解方程 eq，忽略检查
        [1/3]

    In the above example, there is only a single solution to the
    equation. Other expressions will yield spurious roots which
    must be checked manually; roots which give a negative argument
    to odd-powered radicals will also need special checking:  # 在上面的例子中，方程只有一个解。其他表达式会产生虚根，需要手动检查；给奇数次方根带来负参数的根也需要特殊检查

        >>> from sympy import real_root, S
        >>> eq = root(x, 3) - root(x, 5) + S(1)/7
        >>> solve(eq)  # 解方程 eq，这里有两个解，但漏掉了一个
        [CRootOf(7*x**5 - 7*x**3 + 1, 1)**15,
        CRootOf(7*x**5 - 7*x**3 + 1, 2)**15]
        >>> sol = solve(eq, check=False)
        >>> [abs(eq.subs(x,i).n(2)) for i in sol]
        [0.48, 0.e-110, 0.e-110, 0.052, 0.052]

    The first solution is negative so ``real_root`` must be used to see that it
    satisfies the expression:  # 第一个解是负数，因此必须使用 ``real_root`` 确认其满足表达式

        >>> abs(real_root(eq.subs(x, sol[0])).n(2))
        0.e-110

    If the roots of the equation are not real then more care will be
    necessary to find the roots, especially for higher order equations.
    Consider the following expression:  # 如果方程的根不是实数，则需要更谨慎地找到根，特别是对于高阶方程。考虑以下表达式

        >>> expr = root(x, 3) - root(x, 5)

    We will construct a known value for this expression at x = 3 by selecting
    the 1-th root for each radical:  # 我们将通过选择每个根的第一个根来构造 x = 3 时此表达式的已知值

        >>> expr1 = root(x, 3, 1) - root(x, 5, 1)
        >>> v = expr1.subs(x, -3)

    The ``solve`` function is unable to find any exact roots to this equation:  # ``solve`` 函数无法找到此方程的精确根

        >>> eq = Eq(expr, v); eq1 = Eq(expr1, v)
        >>> solve(eq, check=False), solve(eq1, check=False)
        ([], [])

    The function ``unrad``, however, can be used to get a form of the equation  # 然而，函数 ``unrad`` 可以用来获得方程的一个形式
    # 求解可以找到数值根的方程:

        >>> from sympy.solvers.solvers import unrad
        >>> from sympy import nroots
        >>> e, (p, cov) = unrad(eq)
        >>> pvals = nroots(e)
        >>> inversion = solve(cov, x)[0]
        >>> xvals = [inversion.subs(p, i) for i in pvals]

    # 虽然 ``eq`` 或 ``eq1`` 可以用来找到 ``xvals``，但只有使用 ``expr1`` 才能验证解:

        >>> z = expr - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z.subs(x, xi).n()) < 1e-9]
        []
        >>> z1 = expr1 - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z1.subs(x, xi).n()) < 1e-9]
        [-3.0]

    # 参数
    # ==========

    # f :
    #     - 必须为零的单个表达式或多项式
    #     - 一个等式
    #     - 一个关系表达式
    #     - 一个布尔表达式
    #     - 上述任意组合的可迭代对象

    # symbols : （要解的对象）按以下方式指定
    #     - 未指定（将使用其他非数值对象）
    #     - 单个符号
    #     - 符号的展平列表
    #       （例如，``solve(f, x, y)``）
    #     - 符号的有序可迭代对象
    #       （例如，``solve(f, [x, y])``）

    # 另请参阅
    # ========

    # rsolve：用于解决递归关系
    # dsolve：用于解决微分方程

    """
    # 导入不等式模块以简化不等式
    from .inequalities import reduce_inequalities

    # 检查/记录标志
    ###########################################################################

    # 显式设置求解器类型；只要其中一个为 False，其他的都将为 False
    hints = ('cubics', 'quartics', 'quintics')
    default = True
    for k in hints:
        default = flags.setdefault(k, bool(flags.get(k, default)))

    # 如果为 True，则允许解包含符号
    implicit = flags.get('implicit', False)

    # 记录是否希望看到警告
    warn = flags.get('warn', False)

    # 此标志将在下面的快速退出中需要，因此现在记录
    as_set = flags.get('set', False)

    # 跟踪 f 的传递方式
    bare_f = not iterable(f)

    # 检查特殊/快速标志的使用情况，应该只用于方程组
    if flags.get('quick', None) is not None:
        if not flags.get('particular', None):
            raise ValueError('使用 `quick` 时，`particular` 应为 True')
    if flags.get('particular', False) and bare_f:
        raise ValueError(filldedent("""
            'particular/quick' 标志通常与方程组一起使用。请将您的方程传递给列表，
            或者考虑使用 `diophantine` 这样的求解器，如果您希望在整数域中寻找解。"""))

    # 将所有内容转换为 SymPy 表达式，创建表达式列表和符号列表
    ###########################################################################

    def _sympified_list(w):
        return list(map(sympify, w if iterable(w) else [w]))
    f, symbols = (_sympified_list(w) for w in [f, symbols])
    # 对 f 和 symbols 分别进行符号化处理，确保它们是符号表达式的列表或集合

    # preprocess symbol(s)
    ###########################################################################

    ordered_symbols = None  # were the symbols in a well defined order?
    # 初始化 ordered_symbols 变量，用于记录符号是否有明确定义的顺序

    if not symbols:
        # 如果 symbols 为空，则从方程式中获取符号
        symbols = set().union(*[fi.free_symbols for fi in f])
        # 遍历所有方程式 fi，将每个方程式中的自由符号集合并到 symbols 中
        if len(symbols) < len(f):
            for fi in f:
                pot = preorder_traversal(fi)
                # 使用前序遍历方式遍历方程式 fi 的表达式树
                for p in pot:
                    if isinstance(p, AppliedUndef):
                        # 如果当前节点是未定义函数应用
                        if not as_set:
                            flags['dict'] = True  # better show symbols
                            # 设置标志以显示符号
                        symbols.add(p)
                        # 将未定义函数添加到 symbols 中
                        pot.skip()  # don't go any deeper
                        # 跳过当前子树的深入遍历
        ordered_symbols = False
        # 标记符号没有明确定义的顺序
        symbols = list(ordered(symbols))  # to make it canonical
        # 将 symbols 转换为有序列表，以确保规范化顺序
    else:
        if len(symbols) == 1 and iterable(symbols[0]):
            symbols = symbols[0]
            # 如果 symbols 只有一个元素且是可迭代对象，则将其解包
        ordered_symbols = symbols and is_sequence(symbols,
                        include=GeneratorType)
        # 检查 symbols 是否是序列类型，并记录其是否为生成器类型
        _symbols = list(uniq(symbols))
        # 对 symbols 中的元素进行去重
        if len(_symbols) != len(symbols):
            ordered_symbols = False
            # 如果去重后的列表长度与原始 symbols 不同，则标记无序
            symbols = list(ordered(symbols))
            # 否则，将 symbols 转换为有序列表
        else:
            symbols = _symbols
            # 使用去重后的列表作为最终的 symbols

    # check for duplicates
    if len(symbols) != len(set(symbols)):
        raise ValueError('duplicate symbols given')
        # 检查是否存在重复的符号，如果有则引发 ValueError 异常

    # remove those not of interest
    exclude = flags.pop('exclude', set())
    # 从 flags 中获取并移除 'exclude' 键对应的值，并默认为一个空集合
    if exclude:
        if isinstance(exclude, Expr):
            exclude = [exclude]
            # 如果 exclude 是单个表达式，则将其放入列表中
        exclude = set().union(*[e.free_symbols for e in sympify(exclude)])
        # 将 sympify 处理后的 exclude 中每个表达式的自由符号并集到 exclude 中
        symbols = [s for s in symbols if s not in exclude]
        # 从 symbols 中移除不在 exclude 中的符号

    # preprocess equation(s)
    ###########################################################################

    # automatically ignore True values
    if isinstance(f, list):
        f = [s for s in f if s is not S.true]
        # 如果 f 是列表，则移除其中的 S.true 元素

    # handle canonicalization of equation types
    # 对列表 f 中的每个元素进行遍历，同时获取索引 i 和元素 fi
    for i, fi in enumerate(f):
        # 检查 fi 是否是 Eq 或者 Ne 类型的实例
        if isinstance(fi, (Eq, Ne)):
            # 如果 fi 中包含 ImmutableDenseMatrix 类型的参数
            if 'ImmutableDenseMatrix' in [type(a).__name__ for a in fi.args]:
                # 从 fi 中减去 rhs 以获得 lhs
                fi = fi.lhs - fi.rhs
            else:
                # 否则，获取 fi 的左右参数 L 和 R
                L, R = fi.args
                # 如果 R 是 BooleanAtom 类型，则交换 L 和 R
                if isinstance(R, BooleanAtom):
                    L, R = R, L
                # 如果 L 是 BooleanAtom 类型
                if isinstance(L, BooleanAtom):
                    # 如果 fi 是 Ne 类型，则取反 L
                    if isinstance(fi, Ne):
                        L = ~L
                    # 如果 R 是关系表达式，则根据 L 是否为 S.false 取反 R 或返回 R
                    if R.is_Relational:
                        fi = ~R if L is S.false else R
                    elif R.is_Symbol:
                        # 如果 R 是符号，则返回 L
                        return L
                    elif R.is_Boolean and (~R).is_Symbol:
                        # 如果 R 是布尔类型且取反后为符号，则返回 ~L
                        return ~L
                    else:
                        # 否则，抛出未实现的异常，提示 Eq 的其他参数情况为 True 或 False
                        raise NotImplementedError(filldedent('''
                            Unanticipated argument of Eq when other arg
                            is True or False.
                        '''))
                elif isinstance(fi, Eq):
                    # 如果 fi 是 Eq 类型，则将其转换为 lhs - rhs 的形式
                    fi = Add(fi.lhs, -fi.rhs, evaluate=False)
            # 更新列表 f 中的元素为处理后的 fi
            f[i] = fi

        # *** dispatch and handle as a system of relationals
        # **************************************************
        # 如果 fi 是关系表达式
        if fi.is_Relational:
            # 如果 symbols 的长度不为 1，则引发 ValueError
            if len(symbols) != 1:
                raise ValueError("can only solve for one symbol at a time")
            # 如果 warn 为真且 symbols[0] 的 assumptions0 不为空，则发出警告
            if warn and symbols[0].assumptions0:
                warnings.warn(filldedent("""
                    \tWarning: assumptions about variable '%s' are
                    not handled currently.""" % symbols[0]))
            # 调用 reduce_inequalities 处理 f 和 symbols，并返回结果
            return reduce_inequalities(f, symbols=symbols)

        # 如果 fi 是 Poly 类型，则将其转换为表达式
        if isinstance(fi, Poly):
            f[i] = fi.as_expr()

        # 将 f[i] 中的双曲函数替换为指数函数，如果它们具有 symbols 中的自由变量
        f[i] = f[i].replace(lambda w: isinstance(w, HyperbolicFunction) and \
            w.has_free(*symbols), lambda w: w.rewrite(exp))

        # 如果 f[i] 是矩阵
        if f[i].is_Matrix:
            # 设置 bare_f 为 False，并将 f[i] 中的元素扩展到列表 f 中
            bare_f = False
            f.extend(list(f[i]))
            # 将 f[i] 设置为零
            f[i] = S.Zero

        # 如果 f[i] 的自由符号为实数或虚数，则将其拆分为实部和虚部
        freei = f[i].free_symbols
        if freei and all(s.is_extended_real or s.is_imaginary for s in freei):
            fr, fi = f[i].as_real_imag()
            # 如果新的 re、im、arg 或 atan2 没有被引入，则接受 fr 和 fi
            had = f[i].atoms(re, im, arg, atan2)
            if fr and fi and fr != fi and not any(
                    i.atoms(re, im, arg, atan2) - had for i in (fr, fi)):
                # 如果 bare_f 为真，则设置为 False
                if bare_f:
                    bare_f = False
                # 替换 f[i:i+1] 为 [fr, fi]
                f[i: i + 1] = [fr, fi]

    # real/imag handling -----------------------------
    # 如果 f 中的任何元素是布尔类型或 BooleanAtom 类型
    if any(isinstance(fi, (bool, BooleanAtom)) for fi in f):
        # 如果 as_set 为真，则返回空列表和空集合
        if as_set:
            return [], set()
        # 否则，返回空列表
        return []
    # 对于列表 f 中的每个元素 fi，使用 enumerate 获取索引 i 和元素 fi
    for i, fi in enumerate(f):
        # 替换 fi 中的 Abs 函数调用，直到没有替换发生为止
        while True:
            was = fi  # 记录上一次的 fi
            # 使用 lambda 函数替换 Abs 函数调用，如果参数包含 symbols，则重新组织为 Piecewise 形式
            fi = fi.replace(Abs, lambda arg:
                separatevars(Abs(arg)).rewrite(Piecewise) if arg.has(*symbols)
                else Abs(arg))
            # 如果替换前后 fi 没有变化，则退出循环
            if was == fi:
                break

        # 检查 fi 中是否存在 Abs 函数调用，并且这些调用必须包含 symbols 中的符号，否则抛出 NotImplementedError
        for e in fi.find(Abs):
            if e.has(*symbols):
                raise NotImplementedError('solving %s when the argument '
                    'is not real or imaginary.' % e)

        # 替换 fi 中的 arg 函数调用，使用 lambda 函数重写为 atan2 或 atan 形式
        fi = fi.replace(arg, lambda a: arg(a).rewrite(atan2).rewrite(atan))

        # 将更新后的 fi 存回原来的列表位置
        f[i] = fi

    # 查找列表 f 中是否存在包含 re 或 im 的表达式
    freim = [fi for fi in f if fi.has(re, im)]
    if freim:
        irf = []
        # 遍历 symbols 中的符号
        for s in symbols:
            # 如果符号 s 是实数或虚数，则跳过，因为不会出现 re(x) 或 im(x)
            if s.is_real or s.is_imaginary:
                continue
            # 如果 freim 中任何表达式包含 re(s) 或 im(s)，则需要引入辅助方程
            if any(fi.has(re(s), im(s)) for fi in freim):
                irf.append((s, re(s) + S.ImaginaryUnit*im(s)))
        if irf:
            # 根据 irf 中的信息对 f 进行更新，并将相关的 re(s), im(s) 添加到 symbols 中
            for s, rhs in irf:
                f = [fi.xreplace({s: rhs}) for fi in f] + [s - rhs]
                symbols.extend([re(s), im(s)])
            # 如果原先 bare_f 为 True，则设置为 False
            if bare_f:
                bare_f = False
            # 设置 flags 中的 'dict' 为 True，表示发生了变化
            flags['dict'] = True

    # 实现对非符号实体的求解，将其替换为虚拟符号
    f, symbols, swap_sym = recast_to_symbols(f, symbols)
    # 下面需要用到这组符号（可能是重组过的）
    symset = set(symbols)

    # 移除没有感兴趣符号的方程式，这些方程不会被尝试求解，因为用户没有请求，可能很难求解
    # 这意味着解可能会以消除的方程形式给出，例如 solve((x-y, y-3), x) -> {x: y}
    newf = []
    # 对列表 f 中的每个元素进行循环处理
    for fi in f:
        # 让求解器处理以下类型的方程：
        # - 没有符号但是是表达式
        # - 包含感兴趣的符号
        # - 没有感兴趣的符号但是是常数
        # 但是对于一个既不是常数也没有感兴趣符号的表达式，它不能改变我们从剩余方程中获得的解，所以我们不包括它；
        # 如果它是零，可以将其移除；如果它不是零，则整个方程组没有解。
        #
        # 进行这种过滤的原因是允许解决类似 solve((x - y, y), x); 这样的查询，而不是返回空列表 []。
        
        ok = False
        if fi.free_symbols & symset:  # 如果 fi 的自由符号与 symset 有交集
            ok = True
        else:
            if fi.is_number:  # 如果 fi 是一个数值
                if fi.is_Number:  # 如果 fi 是一个数字
                    if fi.is_zero:  # 如果 fi 是零
                        continue  # 继续下一个循环
                    return []  # 返回空列表，表示没有解
                ok = True
            else:
                if fi.is_constant():  # 如果 fi 是一个常数
                    ok = True
        
        if ok:
            newf.append(fi)  # 将符合条件的 fi 添加到 newf 列表中
    
    if not newf:  # 如果 newf 列表为空
        if as_set:  # 如果参数 as_set 为真
            return symbols, set()  # 返回符号集合和空集合
        return []  # 否则返回空列表
    
    f = newf  # 将 f 更新为 newf
    del newf  # 删除 newf 变量

    # 掩盖掉我们不打算反转的任何对象：Derivative、Integral 等，以便解决它们包含的任何内容会给出一个隐式解
    seen = set()  # 创建一个空集合 seen
    non_inverts = set()  # 创建一个空集合 non_inverts
    for fi in f:  # 对列表 f 中的每个元素进行循环处理
        pot = preorder_traversal(fi)  # 对 fi 进行前序遍历
        for p in pot:  # 对 pot 中的每个元素进行循环处理
            if not isinstance(p, Expr) or isinstance(p, Piecewise):
                # 如果 p 不是 Expr 类型或者是 Piecewise 类型，则跳过
                pass
            elif (isinstance(p, bool) or
                    not p.args or
                    p in symset or
                    p.is_Add or p.is_Mul or
                    p.is_Pow and not implicit or
                    p.is_Function and not implicit) and p.func not in (re, im):
                # 如果满足条件，则跳过
                continue
            elif p not in seen:  # 如果 p 不在 seen 集合中
                seen.add(p)  # 将 p 添加到 seen 集合中
                if p.free_symbols & symset:  # 如果 p 的自由符号与 symset 有交集
                    non_inverts.add(p)  # 将 p 添加到 non_inverts 集合中
                else:
                    continue
            pot.skip()  # 跳过 pot 的当前元素
    
    del seen  # 删除 seen 变量
    non_inverts = dict(list(zip(non_inverts, [Dummy() for _ in non_inverts])))  # 创建 non_inverts 字典
    f = [fi.subs(non_inverts) for fi in f]  # 使用 non_inverts 进行替换操作

    # 下面的处理中需要同时使用 xreplace 和 subs：xreplace 用于强制在 Derivative 中进行替换，subs 用于处理非直接的替换
    non_inverts = [(v, k.xreplace(swap_sym).subs(swap_sym)) for k, v in non_inverts.items()]

    # 对浮点数进行有理化处理
    floats = False  # 创建布尔变量 floats，并初始化为 False
    if flags.get('rational', True) is not False:  # 如果 'rational' 标志为真或未提供
        for i, fi in enumerate(f):  # 对列表 f 中的每个元素进行循环处理
            if fi.has(Float):  # 如果 fi 包含 Float
                floats = True  # 设置 floats 为 True
                f[i] = nsimplify(fi, rational=True)  # 使用有理数化简 fi
        
    # 在重写之前捕获任何分母，因为在重写之后它们可能会消失，例如问题 14779
    flags['_denominators'] = _simple_dens(f[0], symbols)  # 设置 '_denominators' 标志
    # 如果存在嵌套的分段函数，需要将其移至顶层，以便选择适当的策略。
    # 然而，只有当其中一个分段函数依赖于我们要解的符号时，才需要这样做。
    def _has_piecewise(e):
        # 检查表达式是否为分段函数
        if e.is_Piecewise:
            # 检查分段函数是否依赖于给定的符号集合
            return e.has(*symbols)
        # 递归检查表达式中是否存在嵌套的分段函数
        return any(_has_piecewise(a) for a in e.args)

    # 遍历函数列表 f 中的每个函数 fi
    for i, fi in enumerate(f):
        # 如果函数 fi 中存在分段函数，则对其进行折叠处理
        if _has_piecewise(fi):
            f[i] = piecewise_fold(fi)

    # 展开和角度求和；一般来说，展开三角函数将有助于找到更多的根，但这不是
    # 返回参数化解决方案的好方法，否则会返回许多具有简单关系的值
    # 在 f 中的每个函数 fi 中找到所有的三角函数，存入集合 targs
    targs = {t for fi in f for t in fi.atoms(TrigonometricFunction)}
    # 如果 targs 中的元素数量大于 1
    if len(targs) > 1:
        # 使用 sift 函数将 targs 分为 add 和 other 两部分
        add, other = sift(targs, lambda x: x.args[0].is_Add, binary=True)
        # 筛选出 add 中不含任何自由符号的元素，并转换成列表
        add, other = [[i for i in l if i.has_free(*symbols)] for l in (add, other)]
        # 创建空字典 trep 用于存储替换规则
        trep = {}
        # 遍历 add 集合中的每个三角函数 t
        for t in add:
            # 提取 t 的第一个参数 a
            a = t.args[0]
            # 将 a 分解为与 symbols 相关和与 symbols 无关的部分
            ind, dep = a.as_independent(*symbols)
            # 如果 dep 是 symbols 中的一个或者其相反数
            if dep in symbols or -dep in symbols:
                # 如果 ind 不是数字，则创建一个虚拟符号 n
                n = Dummy() if not ind.is_Number else ind
                # 使用 TR10 函数对 t 进行处理，并更新 trep 中的替换规则
                trep[t] = TR10(t.func(dep + n)).xreplace({n: ind})
        # 如果 other 集合非空且元素个数小于等于 2
        if other and len(other) <= 2:
            # 如果 other 中有多个元素，则计算它们的最大公约数作为 base
            base = gcd(*[i.args[0] for i in other]) if len(other) > 1 else other[0].args[0]
            # 对 other 集合中的每个元素 i，使用 TR11 函数处理，并更新 trep 中的替换规则
            for i in other:
                trep[i] = TR11(i, base)
        # 将 f 中的每个函数 fi 进行替换处理，更新结果列表 f
        f = [fi.xreplace(trep) for fi in f]

    # 尝试获取一个解决方案
    ###########################################################################
    # 如果 bare_f 为真
    if bare_f:
        # 初始化 solution 为 None
        solution = None
        # 如果 symbols 中的符号数量不为 1，则尝试使用 _solve_undetermined 函数解决 f[0]
        if len(symbols) != 1:
            solution = _solve_undetermined(f[0], symbols, flags)
        # 如果解决方案仍然为空，则使用 _solve 函数尝试解决 f[0]
        if not solution:
            solution = _solve(f[0], *symbols, **flags)
    else:
        # 使用 _solve_system 函数尝试解决系统 f，并获取 linear 和 solution
        linear, solution = _solve_system(f, symbols, **flags)
    # 断言 solution 的类型为列表
    assert type(solution) is list
    # 断言 solution 为空或者第一个元素是字典类型
    assert not solution or type(solution[0]) is dict
    #
    # 后处理
    ###########################################################################
    # 获取标志位 dict 的值，作为 as_dict
    as_dict = flags.get('dict', False)

    # 定义如何解包 solution
    tuple_format = lambda s: [tuple([i.get(x, x) for x in symbols]) for i in s]
    # 如果标志位 as_dict 或 as_set 为真，则不需要解包
    if as_dict or as_set:
        unpack = None
    elif bare_f:
        # 如果 symbols 中只有一个符号，则解包方式为取值列表
        if len(symbols) == 1:
            unpack = lambda s: [i[symbols[0]] for i in s]
        # 如果 solution 中只有一个字典且其键数与 symbols 相同，则解包方式为取字典值
        elif len(solution) == 1 and len(solution[0]) == len(symbols):
            unpack = lambda s: s[0]
        # 如果 ordered_symbols 为真，则解包方式为元组格式化
        elif ordered_symbols:
            unpack = tuple_format
        else:
            # 否则解包方式为直接返回 solution
            unpack = lambda s: s
    else:
        if solution:
            if linear and len(solution) == 1:
                # 如果要获取线性情况下的元组解，使用 `set=True`
                unpack = lambda s: s[0]
            elif ordered_symbols:
                unpack = tuple_format
            else:
                unpack = lambda s: s
        else:
            unpack = None

    # 恢复被掩盖的对象
    if non_inverts and type(solution) is list:
        # 对于返回列表的情况，将每个解中的变量符号替换为非反转的值
        solution = [{k: v.subs(non_inverts) for k, v in s.items()}
            for s in solution]

    # 如果存在符号交换，则将符号列表中的符号替换为交换后的值
    if swap_sym:
        symbols = [swap_sym.get(k, k) for k in symbols]
        for i, sol in enumerate(solution):
            # 将每个解中的变量符号替换为交换后的值，并进行相关的符号替换
            solution[i] = {swap_sym.get(k, k): v.subs(swap_sym)
                      for k, v in sol.items()}

    # 获取关于符号的假设信息，以过滤解
    # 注意，如果无法验证关于解的假设，则仍然返回解
    check = flags.get('check', True)

    # 如果存在浮点数，且解不为空，则将解转换为浮点数表示
    if floats and solution and flags.get('rational', None) is None:
        solution = nfloat(solution, exponent=False)
        # nfloat 可能会显示更多重复解，因此需要去除重复的解
        solution = _remove_duplicate_solutions(solution)

    if check and solution:  # 对解进行假设检查
        warn = flags.get('warn', False)
        got_None = []  # 至少有一个符号值为 None 的解
        no_False = []  # 没有符号值为 False 的解
        for sol in solution:
            a_None = False
            for symb, val in sol.items():
                # 检查每个解中符号的假设
                test = check_assumptions(val, **symb.assumptions0)
                if test:
                    continue
                if test is False:
                    break
                a_None = True
            else:
                no_False.append(sol)
                if a_None:
                    got_None.append(sol)

        solution = no_False
        if warn and got_None:
            # 如果需要警告，并且有符号的假设无法验证的解，则发出警告
            warnings.warn(filldedent("""
                \tWarning: assumptions concerning following solution(s)
                cannot be checked:""" + '\n\t' +
                ', '.join(str(s) for s in got_None)))

    # 如果解为空，则根据 as_set 标志返回结果
    if not solution:
        if as_set:
            return symbols, set()
        return []

    # 对于返回的解列表中的每个字典，使排序规范化
    # 如果不需要作为集合输出，对解决方案进行处理：
    # - 将每个解决方案中的键按指定顺序重新排序，并将其转换为列表形式
    # - 根据默认排序键对解决方案列表进行排序
    if not as_set:  # for set, no point in ordering
        solution = [{k: s[k] for k in ordered(s)} for s in solution]
        solution.sort(key=default_sort_key)

    # 如果既不需要作为集合输出，也不需要作为字典输出，则解包解决方案并返回
    if not (as_set or as_dict):
        return unpack(solution)

    # 如果需要作为字典输出，则直接返回解决方案
    if as_dict:
        return solution

    # 如果需要作为集合输出，根据条件进行处理：
    # - 对于有序符号集合，保持符号的首选顺序
    # - 否则，统一所有解决方案中出现过的符号，构建符号列表
    if ordered_symbols:
        k = symbols  # keep preferred order
    else:
        # 统一所有解决方案中出现过的符号，构建符号列表
        k = list(ordered(set(flatten(tuple(i.keys()) for i in solution))))
    
    # 返回符号列表和解决方案结果的集合形式
    return k, {tuple([s.get(ki, ki) for ki in k]) for s in solution}
# 解决未定系数的辅助函数，返回包含一个字典（解决方案）的列表，否则返回None
def _solve_undetermined(g, symbols, flags):
    """
    A direct call to solve_undetermined_coeffs is more flexible and
    can return both multiple solutions and handle more than one independent
    variable. Here, we have to be more cautious to keep from solving
    something that does not look like an undetermined coeffs system --
    to minimize the surprise factor since singularities that cancel are not
    prohibited in solve_undetermined_coeffs.
    """
    # 如果 g 中的自由符号不在 symbols 中，则执行求解未定系数方程的计算
    if g.free_symbols - set(symbols):
        # 调用 solve_undetermined_coeffs 函数求解未定系数，并传入指定的 flags 参数
        sol = solve_undetermined_coeffs(g, symbols, **dict(flags, dict=True, set=None))
        # 如果仅有一个解，则返回该解
        if len(sol) == 1:
            return sol


# 解决函数 f 的辅助函数，以一个或多个符号的形式返回检查过的解决方案列表的字典。

def _solve(f, *symbols, **flags):
    """
    If no method is implemented to solve the equation, a NotImplementedError
    will be raised. In the case that conversion of an expression to a Poly
    gives None a ValueError will be raised.
    """
    not_impl_msg = "No algorithms are implemented to solve equation %s"
    # 抛出 NotImplementedError 异常，表明未实现求解该方程的算法
    raise NotImplementedError(not_impl_msg % f)
    # 如果符号列表长度不为1：
    # 寻找独立于已解决符号的所需符号的解决方案，例如如果我们解决 x = y
    # 那么在其解决方案中包含 x 的符号将不会被返回。

    # 首先解决线性符号（因为这更简单且限制解的大小），然后继续处理以非线性方式出现的符号。
    # 理想情况下，如果要为几个符号解决单个表达式，则它们必须在表达式的因子中出现，但这里不尝试因式分解。
    # XXX 或许应该首先处理乘法项（Mul），无论此例程中是否有一个或多个符号。

    # 初始化非线性符号列表和已获得符号集合
    nonlin_s = []
    got_s = set()
    rhs_s = set()
    result = []

    # 遍历每个符号进行求解
    for s in symbols:
        # 解决线性方程 xi = v，仅解决指定符号（symbols=[s]）
        xi, v = solve_linear(f, symbols=[s])
        if xi == s:
            # 如果 xi 等于 s，表示找到了符号 s 的解
            # 可选地进行简化操作
            if flags.get('simplify', True):
                v = simplify(v)
            # 获取 v 中的自由符号
            vfree = v.free_symbols
            # 如果 vfree 与已获得符号集合有交集，则忽略该解（线性关系冗余）
            if vfree & got_s:
                continue
            # 将 vfree 加入 rhs_s
            rhs_s |= vfree
            # 将 xi 加入已获得符号集合
            got_s.add(xi)
            # 将解添加到结果列表中
            result.append({xi: v})
        elif xi:  
            # 如果 xi 不为零，可能存在非线性解决方案
            nonlin_s.append(s)
    
    # 如果没有非线性符号，直接返回结果列表
    if not nonlin_s:
        return result
    
    # 对于每个非线性符号 s，尝试解决 f 的方程 _solve(f, s, **flags)
    for s in nonlin_s:
        try:
            soln = _solve(f, s, **flags)
            for sol in soln:
                # 如果解 sol 中的符号与已获得符号集合有交集，则忽略该解
                if sol[s].free_symbols & got_s:
                    continue
                # 将 s 加入已获得符号集合
                got_s.add(s)
                # 将解添加到结果列表中
                result.append(sol)
        except NotImplementedError:
            continue
    
    # 如果已获得符号集合不为空，返回结果列表
    if got_s:
        return result
    else:
        # 如果未获得任何符号，抛出未实现错误
        raise NotImplementedError(not_impl_msg % f)

# 解决单变量 f 的函数
symbol = symbols[0]

# 如果 f 包含未知符号时，仅展开二项式
f = f.replace(lambda e: isinstance(e, binomial) and e.has(symbol),
    lambda e: expand_func(e))

# 在递归调用之前，检查将被执行；变量 `checkdens` 和 `check` 在此处被捕获
# 如果标志值更改，以下内容供参考
flags['check'] = checkdens = check = flags.pop('check', True)

# 如果 f 是一个乘法表达式（Mul），构建解决方案
    # 如果表达式 f 是一个乘法表达式
    if f.is_Mul:
        # 初始化结果集合
        result = set()
        # 遍历乘法表达式的每个因子
        for m in f.args:
            # 如果因子 m 是特定的无限值或复数无穷大，则结果集合为空
            if m in {S.NegativeInfinity, S.ComplexInfinity, S.Infinity}:
                result = set()
                break
            # 对每个因子 m 解方程 _vsolve，并更新结果集合
            soln = _vsolve(m, symbol, **flags)
            result.update(set(soln))
        # 将结果格式化为字典的列表，每个字典表示一个解
        result = [{symbol: v} for v in result]
        # 如果需要进行额外的检查
        if check:
            # 检查每个解是否将任何因子的分母设置为零
            dens = flags.get('_denominators', _simple_dens(f, symbols))
            result = [s for s in result if
                      not any(checksol(den, s, **flags) for den in dens)]
        # 设置标志以便在最后快速退出；每个因子的解已经被检查和简化
        check = False
        flags['simplify'] = False

    # 如果表达式 f 是一个分段函数
    elif f.is_Piecewise:
        # 初始化结果集合
        result = set()
        # 如果任何分段的表达式为零，则简化分段函数 f
        if any(e.is_zero for e, c in f.args):
            f = f.simplify()  # 如果没有帮助，即将失败

        # 初始化条件和否定标志
        cond = neg = True
        # 遍历分段函数的每个表达式和条件
        for expr, cnd in f.args:
            # 当前表达式的显式条件是当前的条件，并且之前的条件都不满足
            cond = And(neg, cnd)
            neg = And(neg, ~cond)

            # 如果表达式为零且条件简化后不是 False，则抛出 NotImplementedError
            if expr.is_zero and cond.simplify() != False:
                raise NotImplementedError(filldedent('''
                    An expression is already zero when %s.
                    This means that in this *region* the solution
                    is zero but solve can only represent discrete,
                    not interval, solutions. If this is a spurious
                    interval it might be resolved with simplification
                    of the Piecewise conditions.''' % cond))

            # 解表达式 expr，并获取候选解集合
            candidates = _vsolve(expr, symbol, **flags)

            # 遍历候选解集合
            for candidate in candidates:
                # 如果候选解已经在结果集合中，则跳过
                if candidate in result:
                    continue
                try:
                    # 替换符号为候选解并简化条件 v
                    v = cond.subs(symbol, candidate)
                    _eval_simplify = getattr(v, '_eval_simplify', None)
                    if _eval_simplify is not None:
                        # 对 v 进行简化
                        v = _eval_simplify(ratio=2, measure=lambda x: 1)
                except TypeError:
                    # 与条件不兼容的类型
                    continue
                # 根据简化后的条件将候选解添加到结果集合中
                if v == False:
                    continue
                if v == True:
                    result.add(candidate)
                else:
                    result.add(Piecewise(
                        (candidate, v),
                        (S.NaN, True)))
        # 解已经被检查和简化
        # ****************************************
        # 将结果格式化为字典的列表，每个字典表示一个解
        return [{symbol: r} for r in result]
    # 如果既不是高阶多变量多项式
    Neither high-order multivariate polynomials
    # 如果结果变量不为空，则进行以下操作
    if result is not None:
        # 如果结果变量为 False，则尝试使用 _unrad 处理
        if result is False:
            # 弹出 _unrad 标志，如果不存在则默认为 True
            if flags.pop('_unrad', True):
                try:
                    # 使用 unrad 处理 f_num 关于 symbol 的解
                    u = unrad(f_num, symbol)
                except (ValueError, NotImplementedError):
                    u = False
                # 如果 u 不为 False
                if u:
                    # 解析 unrad 结果
                    eq, cov = u
                    # 如果存在覆盖（cov）
                    if cov:
                        isym, ieq = cov
                        # 对内部表达式 ieq 使用 _vsolve 求解 symbol
                        inv = _vsolve(ieq, symbol, **flags)[0]
                        # 对于每个 xi 使用 _solve 求解 eq
                        rv = {inv.subs(xi) for xi in _solve(eq, isym, **flags)}
                    else:
                        try:
                            # 使用 _vsolve 求解 eq 关于 symbol
                            rv = set(_vsolve(eq, symbol, **flags))
                        except NotImplementedError:
                            rv = None
                    # 如果 rv 不为 None
                    if rv is not None:
                        # 将结果转换为字典形式
                        result = [{symbol: v} for v in rv]
                        # 如果 simplify 标志未设置，则设置为 False，因为 unrad 结果可能很长或非常高阶
                        flags['simplify'] = flags.get('simplify', False)
                else:
                    pass  # 用于覆盖
    # 如果结果为 False，则允许在下一次处理中使用 tsolve
    if result is False:
        flags.pop('tsolve', None)  # 允许在下一次处理中使用 tsolve
        try:
            # 尝试使用 _tsolve 函数求解
            soln = _tsolve(f_num, symbol, **flags)
            # 如果求解结果不为 None，则将解添加到结果列表中
            if soln is not None:
                result = [{symbol: v} for v in soln]
        except PolynomialError:
            pass
    # ----------- fallback 结束 ----------------------------

    # 如果结果仍为 False，则抛出 NotImplementedError 异常
    if result is False:
        raise NotImplementedError('\n'.join([msg, not_impl_msg % f]))

    # 移除结果中的重复解
    result = _remove_duplicate_solutions(result)

    # 如果 flags 中的 simplify 标志为 True 或者未设置，则对结果进行化简操作
    if flags.get('simplify', True):
        # 对结果中每个解的每个变量进行化简
        result = [{k: d[k].simplify() for k in d} for d in result]
        # 化简操作可能揭示更多的重复解，再次移除
        result = _remove_duplicate_solutions(result)
        # 现在设置 simplify 标志为 False，以确保在 checksol() 中不再进行化简
        flags['simplify'] = False

    # 如果 checkdens 为 True，则检查结果中的解是否使任何分母为零
    if checkdens:
        # 获取表达式 f 的分母
        dens = _simple_dens(f, symbols)
        # 保留不使任何分母为零的解，否则保留不确定情况的解
        result = [r for r in result if
                  not any(checksol(d, r, **flags)
                          for d in dens)]

    # 如果 check 为 True，则仅保留通过 checksol 检查的解
    if check:
        # 保留 checksol 函数返回值不为 False 的解
        result = [r for r in result if
                  checksol(f_num, r, **flags) is not False]
    # 返回处理后的结果
    return result
# 从给定的解集中移除重复的解字典
def _remove_duplicate_solutions(solutions: list[dict[Expr, Expr]]
                                ) -> list[dict[Expr, Expr]]:
    # 使用集合来存储已经存在的解集合
    solutions_set = set()
    # 存储新的不重复解集的列表
    solutions_new = []

    # 遍历每个解字典
    for sol in solutions:
        # 将解字典转换为不可变的集合
        solset = frozenset(sol.items())
        # 如果当前解集合不在已存在的解集合中，则将其添加到新的解集合列表中，并记录该解集合
        if solset not in solutions_set:
            solutions_new.append(sol)
            solutions_set.add(solset)

    # 返回去重后的解集合列表
    return solutions_new


# 解决给定的方程组，返回一个二元组 ``(linear, solution)``
# 其中 ``linear`` 表示方程组是否为线性的，``solution`` 是一个包含每个符号解的字典列表
def _solve_system(exprs, symbols, **flags):
    # 如果表达式列表为空，则返回假和空列表
    if not exprs:
        return False, []

    # 如果指定的标志允许拆分系统，则进行拆分成连接的子组件
    if flags.pop('_split', True):
        # 使用表达式作为顶点集合
        V = exprs
        # 创建符号的集合
        symsset = set(symbols)
        # 计算每个表达式与符号的交集
        exprsyms = {e: e.free_symbols & symsset for e in exprs}
        # 边集合初始化为空列表
        E = []
        # 符号索引字典
        sym_indices = {sym: i for i, sym in enumerate(symbols)}
        
        # 遍历表达式列表，建立连接的边
        for n, e1 in enumerate(exprs):
            for e2 in exprs[:n]:
                # 如果两个表达式共享符号，则它们是连接的
                if exprsyms[e1] & exprsyms[e2]:
                    E.append((e1, e2))
        
        # 构建图 G = (V, E)
        G = V, E
        # 计算连接的子组件
        subexprs = connected_components(G)
        
        # 如果子组件数量大于 1，则逐个解决并合并解集合
        if len(subexprs) > 1:
            subsols = []
            linear = True
            
            # 遍历每个子组件
            for subexpr in subexprs:
                subsyms = set()
                # 计算当前子组件中涉及的符号集合
                for e in subexpr:
                    subsyms |= exprsyms[e]
                # 按符号索引排序符号集合
                subsyms = sorted(subsyms, key=lambda x: sym_indices[x])
                # 调用递归解决函数，跳过拆分步骤
                flags['_split'] = False
                _linear, subsol = _solve_system(subexpr, subsyms, **flags)
                
                # 更新整体线性标志
                if linear:
                    linear = linear and _linear
                
                # 如果子解不是列表，则转换为列表形式
                if not isinstance(subsol, list):
                    subsol = [subsol]
                
                # 将子解添加到子解列表中
                subsols.append(subsol)
            
            # 计算子系统的笛卡尔积，形成完整的解集合
            sols = []
            for soldicts in product(*subsols):
                sols.append(dict(item for sd in soldicts for item in sd.items()))
            
            # 返回整体线性标志和完整的解集合
            return linear, sols
    
    # 初始化多项式列表、密度集合、失败列表、结果列表和已解决符号列表
    polys = []
    dens = set()
    failed = []
    result = []
    solved_syms = []
    linear = True
    
    # 获取标志中的手动标志和检查标志
    manual = flags.get('manual', False)
    checkdens = check = flags.get('check', True)
    
    # 遍历表达式列表
    for j, g in enumerate(exprs):
        # 更新密度集合
        dens.update(_simple_dens(g, symbols))
        # 求解反函数
        i, d = _invert(g, *symbols)
        
        # 如果 d 是符号之一，则判断是否为线性方程
        if d in symbols:
            if linear:
                linear = solve_linear(g, 0, [d])[0] == d
        
        # 更新 g 为 d - i，获取其分子部分
        g = d - i
        g = g.as_numer_denom()[0]
        
        # 如果是手动模式，则将 g 添加到失败列表中
        if manual:
            failed.append(g)
            continue
        
        # 将 g 转换为多项式形式
        poly = g.as_poly(*symbols, extension=True)
        
        # 如果成功转换为多项式，则添加到多项式列表中；否则添加到失败列表中
        if poly is not None:
            polys.append(poly)
        else:
            failed.append(g)
    # 如果 polys 不为空，则执行以下代码块
    if polys:
        # 如果 polys 中所有的多项式都是线性的
        if all(p.is_linear for p in polys):
            # 获取 polys 和 symbols 的长度
            n, m = len(polys), len(symbols)
            # 创建一个 n x (m + 1) 的零矩阵
            matrix = zeros(n, m + 1)

            # 遍历 polys 中的每个多项式及其系数
            for i, poly in enumerate(polys):
                for monom, coeff in poly.terms():
                    try:
                        # 查找 monom 中系数为 1 的索引位置 j
                        j = monom.index(1)
                        # 将 coeff 放入 matrix 的对应位置
                        matrix[i, j] = coeff
                    except ValueError:
                        # 若 monom 中没有系数为 1 的项，则放入 matrix 的最后一列
                        matrix[i, m] = -coeff

            # 如果 flags 中有 'particular' 标志，则调用 minsolve_linear_system 解线性系统
            if flags.pop('particular', False):
                result = minsolve_linear_system(matrix, *symbols, **flags)
            else:
                # 否则调用 solve_linear_system 解线性系统
                result = solve_linear_system(matrix, *symbols, **flags)
            # 如果 result 不为空，则转换为列表，否则为空列表
            result = [result] if result else []
            # 如果 failed 不为空，则进行以下处理
            if failed:
                # 如果 result 不为空，则取第一个结果的键作为已解决的符号列表
                if result:
                    solved_syms = list(result[0].keys())  # 只有一个结果字典
                else:
                    solved_syms = []  # 否则为空列表
        else:
            # 如果 polys 中有非线性多项式，则设置 linear 为 False
            linear = False
            # 如果 symbols 的数量大于 polys 的数量
            if len(symbols) > len(polys):
                # 计算 polys 中所有多项式的自由符号的并集，并按 symbols 的顺序排序
                free = set().union(*[p.free_symbols for p in polys])
                free = list(ordered(free.intersection(symbols)))
                got_s = set()
                result = []
                # 对于 free 符号的所有子集进行遍历
                for syms in subsets(free, len(polys)):
                    try:
                        # 调用 solve_poly_system 解多项式系统
                        res = solve_poly_system(polys, *syms)
                        if res:
                            # 将解集中的每个解加入结果列表，排除依赖之前已解决符号的解
                            for r in set(res):
                                skip = False
                                for r1 in r:
                                    if got_s and any(ss in r1.free_symbols
                                           for ss in got_s):
                                        # 如果解依赖于之前已解决的符号，则丢弃该解
                                        skip = True
                                if not skip:
                                    got_s.update(syms)
                                    result.append(dict(list(zip(syms, r))))
                    except NotImplementedError:
                        pass
                # 如果得到了符号解集，则转换为已解决符号列表
                if got_s:
                    solved_syms = list(got_s)
                else:
                    # 否则抛出未实现的错误
                    raise NotImplementedError('no valid subset found')
            else:
                # 如果 symbols 的数量不大于 polys 的数量，则尝试解多项式系统
                try:
                    result = solve_poly_system(polys, *symbols)
                    if result:
                        # 如果得到解，则直接使用 symbols 作为已解决符号列表
                        solved_syms = symbols
                        result = [dict(list(zip(solved_syms, r))) for r in set(result)]
                except NotImplementedError:
                    # 如果解多项式系统未实现，则将 polys 中的每个多项式转为表达式，并加入 failed 列表
                    failed.extend([g.as_expr() for g in polys])
                    solved_syms = []

    # 将 result 的 None 或空列表转换为包含空字典的列表
    result = result or [{}]

    # 如果 result 为空，则返回 False 和空列表
    if not result:
        return False, []

    # 依赖于线性或多项式系统求解器来简化结果
    # 下面的测试显示，返回的表达式与应用简化后的表达式不同：
    #   sympy/solvers/ode/tests/test_systems/test__classify_linear_system
    #   sympy/solvers/tests/test_solvers/test_issue_4886
    # 因此，文档应该更新以反映这一点，或者以下代码应该是 `bool(failed) or not linear`
    default_simplify = bool(failed)

    # 如果 flags 中包含 'simplify' 标志或者默认情况下应用简化
    if flags.get('simplify', default_simplify):
        # 对结果中的每个字典进行遍历
        for r in result:
            for k in r:
                # 对 r[k] 应用简化操作
                r[k] = simplify(r[k])
        # 现在在 checksol 中不需要再进行简化，因此将 'simplify' 标志设为 False
        flags['simplify'] = False

    # 如果 checkdens 为真，则过滤掉结果中不满足任何 dens 中条件的 r
    if checkdens:
        result = [r for r in result
            if not any(checksol(d, r, **flags) for d in dens)]

    # 如果需要进行检查并且不是线性系统，则过滤掉结果中不满足任何 exprs 中条件的 r
    if check and not linear:
        result = [r for r in result
            if not any(checksol(e, r, **flags) is False for e in exprs)]

    # 过滤掉结果中为 False 的 r
    result = [r for r in result if r]
    # 返回 linear 和过滤后的结果列表
    return linear, result
# 定义 solve_linear 函数，用于解线性方程 f = lhs - rhs
def solve_linear(lhs, rhs=0, symbols=[], exclude=[]):
    # 函数返回一个元组，表示以下情况之一：(0, 1), (0, 0), (symbol, solution), (n, d)
    # 其中 f = lhs - rhs

    r"""
    Return a tuple derived from ``f = lhs - rhs`` that is one of
    the following: ``(0, 1)``, ``(0, 0)``, ``(symbol, solution)``, ``(n, d)``.

    Explanation
    ===========

    ``(0, 1)`` meaning that ``f`` is independent of the symbols in *symbols*
    that are not in *exclude*.

    ``(0, 0)`` meaning that there is no solution to the equation amongst the
    symbols given. If the first element of the tuple is not zero, then the
    function is guaranteed to be dependent on a symbol in *symbols*.

    ``(symbol, solution)`` where symbol appears linearly in the numerator of
    ``f``, is in *symbols* (if given), and is not in *exclude* (if given). No
    simplification is done to ``f`` other than a ``mul=True`` expansion, so the
    solution will correspond strictly to a unique solution.

    ``(n, d)`` where ``n`` and ``d`` are the numerator and denominator of ``f``
    when the numerator was not linear in any symbol of interest; ``n`` will
    never be a symbol unless a solution for that symbol was found (in which case
    the second element is the solution, not the denominator).

    Examples
    ========

    >>> from sympy import cancel, Pow

    ``f`` is independent of the symbols in *symbols* that are not in
    *exclude*:

    >>> from sympy import cos, sin, solve_linear
    >>> from sympy.abc import x, y, z
    >>> eq = y*cos(x)**2 + y*sin(x)**2 - y  # = y*(1 - 1) = 0
    >>> solve_linear(eq)
    (0, 1)
    >>> eq = cos(x)**2 + sin(x)**2  # = 1
    >>> solve_linear(eq)
    (0, 1)
    >>> solve_linear(x, exclude=[x])
    (0, 1)

    The variable ``x`` appears as a linear variable in each of the
    following:

    >>> solve_linear(x + y**2)
    (x, -y**2)
    >>> solve_linear(1/x - y**2)
    (x, y**(-2))

    When not linear in ``x`` or ``y`` then the numerator and denominator are
    returned:

    >>> solve_linear(x**2/y**2 - 3)
    (x**2 - 3*y**2, y**2)

    If the numerator of the expression is a symbol, then ``(0, 0)`` is
    returned if the solution for that symbol would have set any
    denominator to 0:

    >>> eq = 1/(1/x - 2)
    >>> eq.as_numer_denom()
    (x, 1 - 2*x)
    >>> solve_linear(eq)
    (0, 0)

    But automatic rewriting may cause a symbol in the denominator to
    appear in the numerator so a solution will be returned:

    >>> (1/x)**-1
    x
    >>> solve_linear((1/x)**-1)
    (x, 0)

    Use an unevaluated expression to avoid this:

    >>> solve_linear(Pow(1/x, -1, evaluate=False))
    (0, 0)

    If ``x`` is allowed to cancel in the following expression, then it
    appears to be linear in ``x``, but this sort of cancellation is not
    done by ``solve_linear`` so the solution will always satisfy the
    original expression without causing a division by zero error.

    >>> eq = x**2*(1/x - z**2/x)
    >>> solve_linear(cancel(eq))
    (x, 0)
    >>> solve_linear(eq)
    (x**2*(1 - z**2), x)
    """
    """
        A list of symbols for which a solution is desired may be given:
    
        >>> solve_linear(x + y + z, symbols=[y])
        (y, -x - z)
    
        A list of symbols to ignore may also be given:
    
        >>> solve_linear(x + y + z, exclude=[x])
        (y, -x - z)
    
        (A solution for ``y`` is obtained because it is the first variable
        from the canonically sorted list of symbols that had a linear
        solution.)
    
        """
        # 如果 lhs 是一个方程式 Eq 的实例
        if isinstance(lhs, Eq):
            # 如果 rhs 不为 0，则抛出 ValueError 异常
            if rhs:
                raise ValueError(filldedent('''
                If lhs is an Equality, rhs must be 0 but was %s''' % rhs))
            # 将 rhs 设置为 lhs 的右侧表达式
            rhs = lhs.rhs
            # 将 lhs 设置为 lhs 的左侧表达式
            lhs = lhs.lhs
    
        # 初始化 dens 为 None
        dens = None
        # 计算等式的差值 eq
        eq = lhs - rhs
        # 将 eq 表达式化为分子 n 和分母 d
        n, d = eq.as_numer_denom()
        # 如果分子 n 为 0，则返回解 (0, 1)
        if not n:
            return S.Zero, S.One
    
        # 获取分子 n 中的自由符号
        free = n.free_symbols
        # 如果没有指定 symbols，则将 symbols 设为自由符号 free
        if not symbols:
            symbols = free
        else:
            # 检查 symbols 中是否包含非符号的元素，并报错
            bad = [s for s in symbols if not s.is_Symbol]
            if bad:
                if len(bad) == 1:
                    bad = bad[0]
                if len(symbols) == 1:
                    eg = 'solve(%s, %s)' % (eq, symbols[0])
                else:
                    eg = 'solve(%s, *%s)' % (eq, list(symbols))
                raise ValueError(filldedent('''
                    solve_linear only handles symbols, not %s. To isolate
                    non-symbols use solve, e.g. >>> %s <<<.
                                 ''' % (bad, eg)))
            # 取 symbols 与自由符号 free 的交集，作为处理的符号集合
            symbols = free.intersection(symbols)
        # 从符号集合 symbols 中排除 exclude 中的符号
        symbols = symbols.difference(exclude)
        # 如果 symbols 集合为空，则返回解 (0, 1)
        if not symbols:
            return S.Zero, S.One
    
        # derivatives are easy to do but tricky to analyze to see if they
        # are going to disallow a linear solution, so for simplicity we
        # just evaluate the ones that have the symbols of interest
        # 初始化 derivs 为 defaultdict(list)
        derivs = defaultdict(list)
        # 遍历分子 n 中的 Derivative 类型的项
        for der in n.atoms(Derivative):
            # 找到 Derivative 中的自由符号与 symbols 的交集
            csym = der.free_symbols & symbols
            # 将找到的 Derivative 加入 derivs 对应的符号列表中
            for c in csym:
                derivs[c].append(der)
    
        # 初始化 all_zero 为 True
        all_zero = True
    for xi in sorted(symbols, key=default_sort_key):  # 按照默认排序键对符号进行排序，以确保顺序一致
        # 如果这个变量具有导数，现在计算它们
        if isinstance(derivs[xi], list):
            # 对每个导数进行计算，并更新到 derivs[xi] 中
            derivs[xi] = {der: der.doit() for der in derivs[xi]}
        
        # 计算替换后的表达式 newn = n.subs(derivs[xi])
        newn = n.subs(derivs[xi])
        
        # 计算 newn 对 xi 的偏导数
        dnewn_dxi = newn.diff(xi)
        
        # 如果 dnewn_dxi 非零，说明它在任意自由符号的微分中生效
        free = dnewn_dxi.free_symbols
        if dnewn_dxi and (not free or any(dnewn_dxi.diff(s) for s in free) or free == symbols):
            all_zero = False
            # 如果 dnewn_dxi 是 NaN，则跳出循环
            if dnewn_dxi is S.NaN:
                break
            
            # 如果 xi 不在 dnewn_dxi 的自由符号中
            if xi not in dnewn_dxi.free_symbols:
                # 计算 vi
                vi = -1/dnewn_dxi*(newn.subs(xi, 0))
                
                # 如果 dens 为 None，则调用 _simple_dens 函数获取密度
                if dens is None:
                    dens = _simple_dens(eq, symbols)
                
                # 如果对于所有 dens 中的 di，检查是否有解为真
                if not any(checksol(di, {xi: vi}, minimal=True) is True
                          for di in dens):
                    # 简化任何显而易见的积分
                    irep = [(i, i.doit()) for i in vi.atoms(Integral) if
                            i.function.is_number]
                    # 对 vi 进行一些简化处理
                    vi = expand_mul(vi.subs(irep))
                    # 返回找到的解 xi 和 vi
                    return xi, vi
    
    # 如果所有的 dnewn_dxi 都为零
    if all_zero:
        return S.Zero, S.One
    
    # 如果 n 是符号，表示未找到这个符号的解
    if n.is_Symbol:
        return S.Zero, S.Zero
    
    # 默认情况下返回 n 和 d
    return n, d
# 定义函数 minsolve_linear_system，用于求解线性方程组的一个特定解

r"""
Find a particular solution to a linear system.

Explanation
===========

In particular, try to find a solution with the minimal possible number
of non-zero variables using a naive algorithm with exponential complexity.
If ``quick=True``, a heuristic is used.

"""

quick = flags.get('quick', False)
# 检查是否需要快速求解
# flags.get('quick', False) 从 flags 参数中获取 quick 的值，默认为 False

# 求解给定线性方程组 system 的解
s0 = solve_linear_system(system, *symbols, **flags)
# 如果没有解或者所有变量均为零，则返回 s0
if not s0 or all(v == 0 for v in s0.values()):
    return s0

if quick:
    # 如果 quick=True，使用一种启发式方法求解
    # 解决方案 s 是整个线性系统的解
    s = solve_linear_system(system, *symbols)

    def update(determined, solution):
        # 更新解决方案，将已确定的变量代入到解中
        delete = []
        for k, v in solution.items():
            solution[k] = v.subs(determined)
            if not solution[k].free_symbols:
                delete.append(k)
                determined[k] = solution[k]
        for k in delete:
            del solution[k]

    determined = {}
    update(determined, s)

    while s:
        # 按照 default_sort_key 对解进行排序，以获得确定性的结果
        k = max((k for k in s.values()),
                key=lambda x: (len(x.free_symbols), default_sort_key(x)))
        kfree = k.free_symbols
        x = next(reversed(list(ordered(kfree))))
        if len(kfree) != 1:
            determined[x] = S.Zero
        else:
            # _vsolve 是一个未定义的函数，其作用是解决 k 中的变量 x
            val = _vsolve(k, x, check=False)[0]
            if not val and not any(v.subs(x, val) for v in s.values()):
                determined[x] = S.One
            else:
                determined[x] = val
        update(determined, s)

    return determined
    else:
        # 尝试选择 n 个我们希望非零的变量。
        # 其余变量假设为零。我们尝试解决修改后的系统。
        # 如果存在非平凡解，则将自由变量设为 1。
        # 通过增加 n 的值，尝试所有变量的组合，我们会找到一个最优解。
        # 通过在快速方法所管理的变量数目减一的基础上稍微加速。
        
        # 确定符号列表的长度
        N = len(symbols)
        # 使用快速方法找到系统的最小解
        bestsol = minsolve_linear_system(system, *symbols, quick=True)
        # 计算最小解中非零项的数量
        n0 = len([x for x in bestsol.values() if x != 0])
        # 从 n0 - 1 开始递减，尝试找到最优解
        for n in range(n0 - 1, 1, -1):
            # 输出调试信息，显示当前尝试的 n 值
            debugf('minsolve: %s', n)
            # 尝试所有在 N 中的 n 个非零项的组合
            for nonzeros in combinations(range(N), n):
                # 构建子矩阵，包含指定非零项列和常数项列
                subm = Matrix([system.col(i).T for i in nonzeros] + [system.col(-1).T]).T
                # 解子系统，获取变量的解
                s = solve_linear_system(subm, *[symbols[i] for i in nonzeros])
                # 如果存在解且不全为零，则进行进一步处理
                if s and not all(v == 0 for v in s.values()):
                    # 替换解中非零项对应的符号为 1
                    subs = [(symbols[v], S.One) for v in nonzeros]
                    for k, v in s.items():
                        s[k] = v.subs(subs)
                    # 将剩余未解的符号置为 1 或 0
                    for sym in symbols:
                        if sym not in s:
                            if symbols.index(sym) in nonzeros:
                                s[sym] = S.One
                            else:
                                s[sym] = S.Zero
                    thissol = s
                    break
            # 如果未找到解，则退出循环
            if thissol is None:
                break
            # 更新最优解为当前找到的解
            bestsol = thissol
        # 返回最优解
        return bestsol
def solve_linear_system(system, *symbols, **flags):
    r"""
    Solve system of $N$ linear equations with $M$ variables, which means
    both under- and overdetermined systems are supported.

    Explanation
    ===========

    The possible number of solutions is zero, one, or infinite. Respectively,
    this procedure will return None or a dictionary with solutions. In the
    case of underdetermined systems, all arbitrary parameters are skipped.
    This may cause a situation in which an empty dictionary is returned.
    In that case, all symbols can be assigned arbitrary values.

    Input to this function is a $N\times M + 1$ matrix, which means it has
    to be in augmented form. If you prefer to enter $N$ equations and $M$
    unknowns then use ``solve(Neqs, *Msymbols)`` instead. Note: a local
    copy of the matrix is made by this routine so the matrix that is
    passed will not be modified.

    The algorithm used here is fraction-free Gaussian elimination,
    which results, after elimination, in an upper-triangular matrix.
    Then solutions are found using back-substitution. This approach
    is more efficient and compact than the Gauss-Jordan method.

    Examples
    ========

    >>> from sympy import Matrix, solve_linear_system
    >>> from sympy.abc import x, y

    Solve the following system::

           x + 4 y ==  2
        -2 x +   y == 14

    >>> system = Matrix(( (1, 4, 2), (-2, 1, 14)))
    >>> solve_linear_system(system, x, y)
    {x: -6, y: 2}

    A degenerate system returns an empty dictionary:

    >>> system = Matrix(( (0,0,0), (0,0,0) ))
    >>> solve_linear_system(system, x, y)
    {}

    """
    # Assert that the number of columns in the system matrix matches the number of symbols plus one
    assert system.shape[1] == len(symbols) + 1

    # Wrapper function for solving the linear system using sympy_eqs_to_ring and solve_lin_sys
    # Multiply the system matrix by symbols extended with -1 to transform into equations
    eqs = list(system * Matrix(symbols + (-1,)))
    # Convert equations to a ring using sympy_eqs_to_ring
    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    # Solve the linear system using solve_lin_sys, disable raw output (_raw=False)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    # Filter out solutions where symbol equals value (sym != val) and return as a dictionary
    if sol is not None:
        sol = {sym: val for sym, val in sol.items() if sym != val}
    return sol
    # 如果未指定符号，并且 coeffs 不包含符号，则抛出值错误
    if not (coeffs and all(i.is_Symbol for i in coeffs)):
        raise ValueError('must provide symbols for coeffs')

    # 如果 equ 是方程实例，则将等式化简为左侧减右侧
    if isinstance(equ, Eq):
        eq = equ.lhs - equ.rhs
    else:
        eq = equ

    # 对等式进行化简和展开操作
    ceq = cancel(eq)  # 化简等式
    xeq = _mexpand(ceq.as_numer_denom()[0], recursive=True)  # 展开等式的分子部分

    # 获取等式中自由的符号变量
    free = xeq.free_symbols
    coeffs = free & set(coeffs)  # 确定等式中与给定系数相关的符号变量集合

    # 如果没有与系数相关的符号变量，根据标志返回空列表或空字典
    if not coeffs:
        return ([], {}) if flags.get('set', None) else []  # solve(0, x) -> []

    # 如果未提供符号变量 syms，则根据等式分离独立和依赖于系数的部分来确定符号变量
    if not syms:
        # 从等式中分离独立和依赖于系数的部分
        ind, dep = xeq.as_independent(*coeffs, as_Add=True)
        dfree = dep.free_symbols
        syms = dfree & ind.free_symbols  # 确定依赖部分与独立部分共同的符号变量
        if not syms:
            # 如果没有共同的符号变量，则根据依赖部分和系数集合确定符号变量
            syms = dfree - set(coeffs)
        if not syms:
            syms = [Dummy()]  # 如果仍然没有符号变量，则使用虚拟符号

    else:
        if len(syms) == 1 and iterable(syms[0]):
            syms = syms[0]
        e, s, _ = recast_to_symbols([xeq], syms)
        xeq = e[0]
        syms = s

    # 根据符号变量确定等式中系数的功能形式
    gens = set(xeq.as_coefficients_dict(*syms).keys()) - {1}  # 确定等式中的系数形式
    cset = set(coeffs)  # 将系数转换为集合形式
    # 检查是否有生成器包含系数符号
    if any(g.has_xfree(cset) for g in gens):
        return  # 如果有，直接返回，表示生成器包含系数符号

    # 确保我们在使用生成器符号进行操作

    # 将 xeqlist 转换为符号表示，gens 也转换为符号表示
    e, gens, _ = recast_to_symbols([xeq], list(gens))
    xeq = e[0]  # 更新 xeqlist 为符号表示后的结果

    # 收集在生成器前面的系数

    # 使用 gens 收集 xeq 中的各个生成器的系数，但不进行求值
    system = list(collect(xeq, gens, evaluate=False).values())

    # 获取一个解

    # 解 xeq 关于生成器的系数，得到解 soln
    soln = solve(system, coeffs, **flags)

    # 根据设置解包（除非明确指定字典形式）

    # 检查 soln 是否为字典形式，或者根据设置（flags）要求返回字典或集合，或者长度不为 1
    settings = flags.get('dict', None) or flags.get('set', None)
    if type(soln) is dict or settings or len(soln) != 1:
        return soln  # 直接返回 soln

    # 如果 soln 长度为 1，返回解的第一个元素
    return soln[0]
# 解决线性方程组的增广矩阵，使用 LU 分解求解，并返回一个字典，其中解按 *syms* 的顺序键入
def solve_linear_system_LU(matrix, syms):
    # 检查矩阵的行数是否等于列数减1，否则抛出错误
    if matrix.rows != matrix.cols - 1:
        raise ValueError("Rows should be equal to columns - 1")
    
    # 提取矩阵 A 和向量 b
    A = matrix[:matrix.rows, :matrix.rows]
    b = matrix[:, matrix.cols - 1:]
    
    # 使用 LU 分解求解线性方程组
    soln = A.LUsolve(b)
    
    # 将解按照给定的符号顺序存入字典中
    solutions = {}
    for i in range(soln.rows):
        solutions[syms[i]] = soln[i, 0]
    
    return solutions


# 返回矩阵 M 的行列式，通过使用排列来选择因子
def det_perm(M):
    args = []
    s = True
    n = M.rows
    list_ = M.flat()
    
    # 使用生成的 Bell 排列计算行列式
    for perm in generate_bell(n):
        fac = []
        idx = 0
        for j in perm:
            fac.append(list_[idx + j])
            idx += n
        term = Mul(*fac) # 通过乘积形式计算每个排列的行列式项
        args.append(term if s else -term) # 每隔一个改变符号
        s = not s
    
    return Add(*args)


# 返回矩阵 M 的行列式，通过使用子行列式计算，避免引入新的乘积嵌套
def det_minor(M):
    n = M.rows
    if n == 2:
        return M[0, 0]*M[1, 1] - M[1, 0]*M[0, 1]  # 2x2 矩阵的行列式计算公式
    else:
        return sum((1, -1)[i % 2]*Add(*[M[0, i]*d for d in
            Add.make_args(det_minor(M.minor_submatrix(0, i)))])  # 递归计算较大矩阵的行列式
            if M[0, i] else S.Zero for i in range(n))


# 返回矩阵 M 的行列式，根据矩阵特性快速计算，如果矩阵大小较小且包含符号则使用 det_perm，否则使用 det_minor
def det_quick(M, method=None):
    if any(i.has(Symbol) for i in M):
        if M.rows < 8 and all(i.has(Symbol) for i in M):
            return det_perm(M)  # 如果矩阵较小且包含符号，则使用排列方式计算行列式
        return det_minor(M)  # 否则使用子行列式方式计算行列式
    else:
        return M.det(method=method) if method else M.det()  # 如果没有符号且指定了方法，则使用指定方法计算行列式，否则使用默认方法


# 返回矩阵 M 的逆矩阵，假设矩阵中有大量零元素或者矩阵较小
def inv_quick(M):
    """Return the inverse of ``M``, assuming that either
    there are lots of zeros or the size of the matrix
    is small.
    """
    # 检查矩阵 M 中的所有元素是否都是数值类型
    if not all(i.is_Number for i in M):
        # 如果矩阵 M 中有任何一个元素不是数值类型
        if not any(i.is_Number for i in M):
            # 定义行列式函数为排列行列式函数 det_perm
            det = lambda _: det_perm(_)
        else:
            # 否则定义行列式函数为子矩阵行列式函数 det_minor
            det = lambda _: det_minor(_)
    else:
        # 如果矩阵 M 中所有元素都是数值类型，则直接返回 M 的逆矩阵
        return M.inv()
    
    # 获取矩阵 M 的行数
    n = M.rows
    # 计算矩阵 M 的行列式值
    d = det(M)
    
    # 如果矩阵 M 的行列式值为零，则抛出非可逆矩阵错误
    if d == S.Zero:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible")
    
    # 初始化一个全零矩阵 ret，大小为 n x n
    ret = zeros(n)
    # 初始化一个符号变量 s1，初始值为 -1
    s1 = -1
    
    # 遍历矩阵 M 的行索引
    for i in range(n):
        # 在每次循环开始时更新符号 s，实现交替变号
        s = s1 = -s1
        # 遍历矩阵 M 的列索引
        for j in range(n):
            # 计算当前子矩阵 M.minor_submatrix(i, j) 的行列式值
            di = det(M.minor_submatrix(i, j))
            # 计算当前位置 (j, i) 处的代数余子式，乘以符号 s，并除以矩阵 M 的行列式值 d
            ret[j, i] = s * di / d
            # 更新符号 s，变号
            s = -s
    
    # 返回计算得到的逆矩阵 ret
    return ret
# 这些是具有每个周期多个反向值的函数
multi_inverses = {
    sin: lambda x: (asin(x), S.Pi - asin(x)),  # 对于正弦函数，返回其反正弦值对
    cos: lambda x: (acos(x), 2*S.Pi - acos(x)),  # 对于余弦函数，返回其反余弦值对
}


def _vsolve(e, s, **flags):
    """返回方程 e 关于符号 s 的解的标量值列表"""
    return [i[s] for i in _solve(e, s, **flags)]


def _tsolve(eq, sym, **flags):
    """
    ``_solve`` 的辅助函数，用于解决关于给定符号的超越方程。
    可以解决包含幂和对数的各种方程。

    目前不能保证会返回所有解，也不能保证实数解优先于复数解。

    可能会返回潜在解的列表，或者在无法解决方程的情况下返回 None。
    所有其他错误（如无法将表达式转换为 Poly）均未处理。

    Examples
    ========

    >>> from sympy import log, ordered
    >>> from sympy.solvers.solvers import _tsolve as tsolve
    >>> from sympy.abc import x

    >>> list(ordered(tsolve(3**(2*x + 5) - 4, x)))
    [-5/2 + log(2)/log(3), (-5*log(3)/2 + log(2) + I*pi)/log(3)]

    >>> tsolve(log(x) + 2*x, x)
    [LambertW(2)/2]

    """
    if 'tsolve_saw' not in flags:
        flags['tsolve_saw'] = []
    if eq in flags['tsolve_saw']:
        return None
    else:
        flags['tsolve_saw'].append(eq)

    # 尝试反转方程
    rhs, lhs = _invert(eq, sym)

    # 如果左侧是符号本身，则返回右侧作为解
    if lhs == sym:
        return [rhs]
    
    # 如果未实现反转，则捕获异常
    except NotImplementedError:
        pass

    # 可能是 Lambert 模式
    # 如果标志中没有'bivariate'键，或者其对应的值为True，则执行以下代码块
    if flags.pop('bivariate', True):
        # lambert形式可能需要一些帮助来被识别，例如将表达式
        # 2**(3*x) + x**3*log(2)**3 + 3*x**2*log(2)**2 + 3*x*log(2) + 1
        # 转换为 2**(3*x) + (x*log(2) + 1)**3

        # 使log中的生成器具有指数1
        logs = eq.atoms(log)
        # 计算具有指数的最小值
        spow = min(
            {i.exp for j in logs for i in j.atoms(Pow)
             if i.base == sym} or {1})
        # 如果最小指数不为1，则进行下面的处理
        if spow != 1:
            # 构造p = sym**spow
            p = sym**spow
            # 创建一个虚拟变量
            u = Dummy('bivariate-cov')
            # 将eq中的p替换为u，生成新的等式ueq
            ueq = eq.subs(p, u)
            # 如果ueq中不含有自由变量sym，则调用_vsolve函数求解ueq
            if not ueq.has_free(sym):
                sol = _vsolve(ueq, u, **flags)
                # 求解p - u = 0，得到inv
                inv = _vsolve(p - u, sym)
                # 返回inv和sol的组合结果
                return [i.subs(u, s) for i in inv for s in sol]

        # 通过_eq_down获取与eq相关的信息
        g = _filtered_gens(eq.as_poly(), sym)
        # up_or_log设置为空集
        up_or_log = set()
        # 遍历g中的元素
        for gi in g:
            # 如果gi是exp、log类型，或者gi是Pow类型且其底数为S.Exp1
            if isinstance(gi, (exp, log)) or (gi.is_Pow and gi.base == S.Exp1):
                # 将gi添加到up_or_log集合中
                up_or_log.add(gi)
            # 如果gi是Pow类型
            elif gi.is_Pow:
                # 简化gisimp
                gisimp = powdenest(expand_power_exp(gi))
                # 如果gisimp是Pow类型，并且sym是其指数的自由变量
                if gisimp.is_Pow and sym in gisimp.exp.free_symbols:
                    # 将gi添加到up_or_log集合中
                    up_or_log.add(gi)
        # 将eq_down中的log展开，并将up_or_log对应的项设置为0，得到eq_down
        eq_down = expand_log(expand_power_exp(eq)).subs(
            dict(list(zip(up_or_log, [0]*len(up_or_log)))))
        # 将eq_down和(eq - eq_down)进行合并
        eq = expand_power_exp(factor(eq_down, deep=True) + (eq - eq_down))
        # 将lhs和rhs设置为eq的反向操作
        rhs, lhs = _invert(eq, sym)
        # 如果lhs中包含有sym
        if lhs.has(sym):
            try:
                # 将lhs转换为多项式形式
                poly = lhs.as_poly()
                # 获取过滤后的生成器g
                g = _filtered_gens(poly, sym)
                # 设置_eq为lhs - rhs
                _eq = lhs - rhs
                # 求解_lambert方程_eq，得到sols
                sols = _solve_lambert(_eq, sym, g)
                # 如果nsimplify(s)等于s并且其操作次数小于等于s的操作次数，则使用简化的形式
                for n, s in enumerate(sols):
                    ns = nsimplify(s)
                    if ns != s and ns.count_ops() <= s.count_ops():
                        # 检查简化后的ns是否满足_eq的条件
                        ok = checksol(_eq, sym, ns)
                        # 如果ok为None，则检查_eq.subs(sym, ns)是否等于0
                        if ok is None:
                            ok = _eq.subs(sym, ns).equals(0)
                        # 如果ok为True，则将sols中的第n个元素替换为ns
                        if ok:
                            sols[n] = ns
                # 返回求解结果sols
                return sols
            # 如果出现NotImplementedError异常
            except NotImplementedError:
                # 可能是一个复杂的函数
                # 如果g的长度为2
                if len(g) == 2:
                    try:
                        # 调用bivariate_type函数获取gpu
                        gpu = bivariate_type(lhs - rhs, *g)
                        # 如果gpu为None，则抛出NotImplementedError异常
                        if gpu is None:
                            raise NotImplementedError
                        # 从gpu中获取g、p、u
                        g, p, u = gpu
                        # 设置'bivariate'标志为False
                        flags['bivariate'] = False
                        # 调用_tsolve函数求解g - u = 0，得到inversion
                        inversion = _tsolve(g - u, sym, **flags)
                        # 如果inversion不为空
                        if inversion:
                            # 调用_vsolve函数求解p = u，得到sol
                            sol = _vsolve(p, u, **flags)
                            # 返回求解结果
                            return list({i.subs(u, s)
                                         for i in inversion for s in sol})
                    # 如果出现NotImplementedError异常
                    except NotImplementedError:
                        pass
                else:
                    pass
        else:
            pass
    # 如果 'force' 标志不存在或其值为真，则执行以下操作
    if flags.pop('force', True):
        # 将 'force' 标志设为假
        flags['force'] = False
        # 对 lhs - rhs 进行位置化，得到位置化的表达式和替换字典
        pos, reps = posify(lhs - rhs)
        # 如果 rhs 等于 S.ComplexInfinity，则返回空列表
        if rhs == S.ComplexInfinity:
            return []
        
        # 遍历替换字典中的项目，查找是否有与 sym 相等的 s 值
        for u, s in reps.items():
            if s == sym:
                break
        else:
            # 如果未找到与 sym 相等的 s 值，则将 u 设为 sym
            u = sym
        
        # 如果 pos 中包含 u
        if pos.has(u):
            try:
                # 尝试解决 pos 中的 u，使用给定的标志参数
                soln = _vsolve(pos, u, **flags)
                # 返回解的替换结果列表
                return [s.subs(reps) for s in soln]
            except NotImplementedError:
                pass
        else:
            # 如果 pos 不包含 u，则什么也不做
            pass  # 这里是为了覆盖率
        
    # 返回空，这里是为了覆盖率
    return  # 这里是为了覆盖率
# 定义一个装饰器函数，用于在数值求解时保存 mpmath 的精度设置
@conserve_mpmath_dps
def nsolve(*args, dict=False, **kwargs):
    r"""
    解数值非线性方程组：``nsolve(f, [args,] x0, modules=['mpmath'], **kwargs)``。

    说明
    ===========

    ``f`` 是表示系统的符号表达式向量函数。
    *args* 是变量。如果只有一个变量，可以省略这个参数。
    ``x0`` 是一个接近解的起始向量。

    使用 modules 关键字指定应该用哪些模块来评估函数和雅可比矩阵。
    确保使用支持矩阵的模块。有关语法的更多信息，请参阅 ``lambdify`` 的文档字符串。

    如果关键字参数包含 ``dict=True`` （默认为 False），则 ``nsolve`` 将返回一个解映射的列表（可能为空）。
    如果想要将 ``nsolve`` 作为解决方法的备用，这可能特别有用，因为使用 dict 参数对两种方法都产生一致类型结构的返回值。
    请注意：为了与 ``solve`` 保持一致，即使 ``nsolve`` （当前至少如此）一次只找到一个解，解也将作为列表返回。

    支持超定系统。

    示例
    ========

    >>> from sympy import Symbol, nsolve
    >>> import mpmath
    >>> mpmath.mp.dps = 15
    >>> x1 = Symbol('x1')
    >>> x2 = Symbol('x2')
    >>> f1 = 3 * x1**2 - 2 * x2**2 - 1
    >>> f2 = x1**2 - 2 * x1 + x2**2 + 2 * x2 - 8
    >>> print(nsolve((f1, f2), (x1, x2), (-1, 1)))
    Matrix([[-1.19287309935246], [1.27844411169911]])

    对于一维函数，语法更简化：

    >>> from sympy import sin, nsolve
    >>> from sympy.abc import x
    >>> nsolve(sin(x), x, 2)
    3.14159265358979
    >>> nsolve(sin(x), 2)
    3.14159265358979

    要比默认精度解决更高精度的问题，请使用 prec 参数：

    >>> from sympy import cos
    >>> nsolve(cos(x) - x, 1)
    0.739085133215161
    >>> nsolve(cos(x) - x, 1, prec=50)
    0.73908513321516064165531208767387340401341175890076
    >>> cos(_)
    0.73908513321516064165531208767387340401341175890076

    要解实函数的复根，必须指定一个非实的初始点：

    >>> from sympy import I
    >>> nsolve(x**2 + 2, I)
    1.4142135623731*I

    使用 ``mpmath.findroot`` 并可以找到它们更广泛的文档，特别是关于关键字参数和可用求解器的文档。
    请注意，如果函数在根附近非常陡峭，则可能会验证解失败。在这种情况下，应使用标志 ``verify=False`` 并独立验证解。

    >>> from sympy import cos, cosh
    >>> f = cos(x)*cosh(x) - 1
    >>> nsolve(f, 3.14*100)
    Traceback (most recent call last):
    ...
    ValueError: Could not find root within given tolerance. (1.39267e+230 > 2.1684e-19)
    >>> ans = nsolve(f, 3.14*100, verify=False); ans
    312.588469032184
    >>> f.subs(x, ans).n(2)
    2.1e+121
    >>> (f/f.diff(x)).subs(x, ans).n(2)
    7.4e-15

    One might safely skip the verification if bounds of the root are known
    and a bisection method is used:

    >>> bounds = lambda i: (3.14*i, 3.14*(i + 1))
    >>> nsolve(f, bounds(100), solver='bisect', verify=False)
    315.730061685774

    Alternatively, a function may be better behaved when the
    denominator is ignored. Since this is not always the case, however,
    the decision of what function to use is left to the discretion of
    the user.

    >>> eq = x**2/(1 - x)/(1 - 2*x)**2 - 100
    >>> nsolve(eq, 0.46)
    Traceback (most recent call last):
    ...
    ValueError: Could not find root within given tolerance. (10000 > 2.1684e-19)
    Try another starting point or tweak arguments.
    >>> nsolve(eq.as_numer_denom()[0], 0.46)
    0.46792545969349058

    """
    # there are several other SymPy functions that use method= so
    # guard against that here
    检查是否使用了不应在此上下文中使用的关键字"method"，这可能与某些 mpmath 求解器的直接使用有关，但在使用 nsolve（和 findroot）时，应使用关键字"solver"。
    if 'method' in kwargs:
        raise ValueError(filldedent('''
            Keyword "method" should not be used in this context.  When using
            some mpmath solvers directly, the keyword "method" is
            used, but when using nsolve (and findroot) the keyword to use is
            "solver".'''))

    if 'prec' in kwargs:
        import mpmath
        设置 mpmath 的精度为传入参数 'prec' 的值
        mpmath.mp.dps = kwargs.pop('prec')

    # keyword argument to return result as a dictionary
    将 as_dict 设置为 dict，以便返回结果作为字典
    as_dict = dict
    from builtins import dict  # to unhide the builtin

    # interpret arguments
    解析传入的参数
    if len(args) == 3:
        f = args[0]
        fargs = args[1]
        x0 = args[2]
        if iterable(fargs) and iterable(x0):
            检查参数 fargs 和 x0 是否可迭代，如果不可迭代则报错
            if len(x0) != len(fargs):
                抛出 TypeError 异常，指出 nsolve 预期的猜测向量数量与实际传入的数量不符
                                % (len(fargs), len(x0)))
    elif len(args) == 2:
        f = args[0]
        fargs = None
        x0 = args[1]
        if iterable(f):
            抛出 TypeError 异常，指出 nsolve 预期 3 个参数，但只传入了 2 个
    elif len(args) < 2:
        抛出 TypeError 异常，指出 nsolve 预期至少 2 个参数，但只传入了特定数量的参数
                        % len(args))
    else:
        抛出 TypeError 异常，指出 nsolve 预期最多 3 个参数，但传入了超过预期数量的参数
                        % len(args))
    modules = kwargs.get('modules', ['mpmath'])
    如果参数中包含 'modules' 关键字，则将其值设为列表 ['mpmath']

    if iterable(f):
        如果参数 f 是可迭代的对象，则将其转换为列表形式，并且如果其中的每个元素是方程类型的，将其转换为左侧减去右侧的形式
        f = list(f)
        for i, fi in enumerate(f):
            if isinstance(fi, Eq):
                f[i] = fi.lhs - fi.rhs
        将列表形式的 f 转换为矩阵，并转置
        f = Matrix(f).T

    if iterable(x0):
        如果参数 x0 是可迭代的对象，则将其转换为列表形式
        x0 = list(x0)
    # 检查变量 f 是否为 Matrix 类型的对象，如果不是，则假设为 SymPy 表达式
    if not isinstance(f, Matrix):
        # 如果 f 是一个方程对象 Eq，则转换为等式左边减去右边的形式
        if isinstance(f, Eq):
            f = f.lhs - f.rhs
        # 如果 f 是一个关系表达式，则抛出类型错误，因为 nsolve 无法处理不等式
        elif f.is_Relational:
            raise TypeError('nsolve cannot accept inequalities')
        # 获取表达式 f 中的自由符号集合
        syms = f.free_symbols
        # 如果未提供 fargs 参数，则选择 f 中的一个自由符号作为参数
        if fargs is None:
            fargs = syms.copy().pop()
        # 检查表达式是否为一维且包含数值函数，否则引发值错误
        if not (len(syms) == 1 and (fargs in syms or fargs[0] in syms)):
            raise ValueError('expected a one-dimensional and numerical function')

        # 如果表达式中有分母，函数的行为更加良好，但是否包含分母由用户决定
        # 例如，问题 11768
        # 使用 lambdify 函数将 SymPy 表达式转换为可调用的函数对象 f
        f = lambdify(fargs, f, modules)
        # 使用 findroot 函数找到表达式 f 在初始值 x0 处的根
        x = sympify(findroot(f, x0, **kwargs))
        # 如果指定返回结果为字典形式，则返回列表包含一个字典
        if as_dict:
            return [{fargs: x}]
        # 否则直接返回计算得到的根 x
        return x

    # 如果参数 fargs 的长度大于矩阵 f 的列数，则抛出未实现错误
    if len(fargs) > f.cols:
        raise NotImplementedError('need at least as many equations as variables')
    
    # 检查是否需要详细输出信息
    verbose = kwargs.get('verbose', False)
    if verbose:
        # 打印函数 f(x) 的信息
        print('f(x):')
        print(f)
    
    # 计算函数 f 对自变量 fargs 的雅可比矩阵 J
    J = f.jacobian(fargs)
    if verbose:
        # 打印雅可比矩阵 J(x) 的信息
        print('J(x):')
        print(J)
    
    # 使用 lambdify 函数将 SymPy 表达式 f.T 和雅可比矩阵 J 转换为可调用的函数对象
    f = lambdify(fargs, f.T, modules)
    J = lambdify(fargs, J, modules)
    
    # 使用 findroot 函数求解数值系统
    x = findroot(f, x0, J=J, **kwargs)
    if as_dict:
        # 如果指定返回结果为字典形式，则返回列表包含一个字典
        return [dict(zip(fargs, [sympify(xi) for xi in x]))]
    # 否则直接返回结果矩阵 x
    return Matrix(x)
def _invert(eq, *symbols, **kwargs):
    """
    Return tuple (i, d) where ``i`` is independent of *symbols* and ``d``
    contains symbols.

    Explanation
    ===========
    
    ``i`` and ``d`` are obtained after recursively using algebraic inversion
    until an uninvertible ``d`` remains. If there are no free symbols then
    ``d`` will be zero. Some (but not necessarily all) solutions to the
    expression ``i - d`` will be related to the solutions of the original
    expression.

    Examples
    ========

    >>> from sympy.solvers.solvers import _invert as invert
    >>> from sympy import sqrt, cos
    >>> from sympy.abc import x, y
    >>> invert(x - 3)
    (3, x)
    >>> invert(3)
    (3, 0)
    >>> invert(2*cos(x) - 1)
    (1/2, cos(x))
    >>> invert(sqrt(x) - 3)
    (3, sqrt(x))
    >>> invert(sqrt(x) + y, x)
    (-y, sqrt(x))
    >>> invert(sqrt(x) + y, y)
    (-sqrt(x), y)
    >>> invert(sqrt(x) + y, x, y)
    (0, sqrt(x) + y)

    If there is more than one symbol in a power's base and the exponent
    is not an Integer, then the principal root will be used for the
    inversion:

    >>> invert(sqrt(x + y) - 2)
    (4, x + y)
    >>> invert(sqrt(x + y) + 2)  # note +2 instead of -2
    (4, x + y)

    If the exponent is an Integer, setting ``integer_power`` to True
    will force the principal root to be selected:

    >>> invert(x**2 - 4, integer_power=True)
    (2, x)

    """
    # 将输入的表达式转换为符号表达式
    eq = sympify(eq)
    if eq.args:
        # 确保我们使用扁平化的表达式进行处理
        eq = eq.func(*eq.args)
    # 找到表达式中的自由符号
    free = eq.free_symbols
    # 如果没有指定符号，则默认使用所有自由符号
    if not symbols:
        symbols = free
    # 如果表达式中没有自由符号与给定的符号集合相交，则返回表达式和零
    if not free & set(symbols):
        return eq, S.Zero

    # 是否使用整数指数进行处理
    dointpow = bool(kwargs.get('integer_power', False))

    # 左右两侧的表达式
    lhs = eq
    rhs = S.Zero
    return rhs, lhs


def unrad(eq, *syms, **flags):
    """
    Remove radicals with symbolic arguments and return (eq, cov),
    None, or raise an error.

    Explanation
    ===========
    
    None is returned if there are no radicals to remove.

    NotImplementedError is raised if there are radicals and they cannot be
    removed or if the relationship between the original symbols and the
    change of variable needed to rewrite the system as a polynomial cannot
    be solved.

    Otherwise the tuple, ``(eq, cov)``, is returned where:

    *eq*, ``cov``
        *eq* is an equation without radicals (in the symbol(s) of
        interest) whose solutions are a superset of the solutions to the
        original expression. *eq* might be rewritten in terms of a new
        variable; the relationship to the original variables is given by
        ``cov`` which is a list containing ``v`` and ``v**p - b`` where
        ``p`` is the power needed to clear the radical and ``b`` is the
        radical now expressed as a polynomial in the symbols of interest.
        For example, for sqrt(2 - x) the tuple would be
        ``(c, c**2 - 2 + x)``. The solutions of *eq* will contain
        solutions to the original equation (if there are any).
    """
    # 此函数移除带有符号参数的根式，并返回（eq, cov），无效或引发错误。

    # 如果没有根式需要移除，则返回None。
    # 如果存在根式但无法移除，或者无法解决重写系统为多项式的变量关系，则引发NotImplementedError。
    # 否则返回元组（eq, cov），其中eq是不含根式的方程（在感兴趣的符号中），其解是原始表达式解的超集。
    # eq可能用新变量重写；与原始变量的关系由cov给出，它是一个列表，包含v和v**p - b，其中p是清除根式所需的幂次，b是根式在感兴趣符号中的多项式表示。
    # 例如，对于sqrt(2 - x)，元组将是(c, c**2 - 2 + x)。eq的解将包含原始方程的解（如果有）。
    uflags = {"check": False, "simplify": False}
    # 定义一个字典uflags，用于存储一些标志位，初始设置为{"check": False, "simplify": False}

    def _cov(p, e):
        if cov:
            # 如果cov已经存在，则执行以下操作
            oldp, olde = cov
            # 从cov中获取旧的p和e
            if Poly(e, p).degree(p) in (1, 2):
                # 如果e关于p的多项式次数是1或2
                cov[:] = [p, olde.subs(oldp, _vsolve(e, p, **uflags)[0])]
                # 更新cov，用p替换旧的p，并用_vsolve函数求解e关于p的变量，然后更新olde
            else:
                raise NotImplementedError
                # 抛出未实现的错误
        else:
            cov[:] = [p, e]
            # 否则，将cov设置为[p, e]

    def _canonical(eq, cov):
        if cov:
            # 如果cov存在，则执行以下操作
            p, e = cov
            # 获取cov中的p和e
            rep = {p: Dummy(p.name)}
            # 创建替换字典，将p映射为一个新的Dummy变量
            eq = eq.xreplace(rep)
            # 在eq中执行替换操作
            cov = [p.xreplace(rep), e.xreplace(rep)]
            # 更新cov，用新的Dummy变量替换p

        eq = factor_terms(_mexpand(eq.as_numer_denom()[0], recursive=True), clear=True)
        # 对eq的数值分子展开并进行因式分解和合并，清除冗余项
        if eq.is_Mul:
            # 如果eq是乘积表达式
            args = []
            for f in eq.args:
                if f.is_number:
                    continue
                if f.is_Pow:
                    args.append(f.base)
                    # 如果f是幂函数，则只取其底数
                else:
                    args.append(f)
                    # 否则直接添加到args中
            eq = Mul(*args)  # leave as Mul for more efficient solving
            # 将args中的项作为Mul的参数传递给eq，保持作为Mul对象以便于更高效的求解

        margs = list(Mul.make_args(eq))
        # 将eq中的乘法表达式转换为列表形式
        changed = False
        for i, m in enumerate(margs):
            if m.could_extract_minus_sign():
                # 如果m能够提取负号
                margs[i] = -m
                # 将m变为其相反数
                changed = True
        if changed:
            eq = Mul(*margs, evaluate=False)
            # 如果有改变，则将修改后的margs重新组合为Mul对象，但不进行求值

        return eq, cov
        # 返回修改后的eq和更新后的cov
    def _Q(pow):
        # 返回 Pow 指数的分母的主导有理数部分
        c = pow.as_base_exp()[1].as_coeff_Mul()[0]
        if not c.is_Rational:
            return S.One
        return c.q

    # 定义 _take 方法，确定是否对一个项感兴趣
    def _take(d):
        # 如果任何因子的指数的分母不为1，则返回 True
        for pow in Mul.make_args(d):
            if not pow.is_Pow:
                continue
            if _Q(pow) == 1:
                continue
            if pow.free_symbols & syms:
                return True
        return False
    _take = flags.setdefault('_take', _take)

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs  # XXX 遗留的 Eq 作为等式支持
    elif not isinstance(eq, Expr):
        return

    cov, nwas, rpt = [flags.setdefault(k, v) for k, v in
        sorted({"cov": [], "n": None, "rpt": 0}.items())]

    # 预处理
    eq = powdenest(factor_terms(eq, radical=True, clear=True))
    eq = eq.as_numer_denom()[0]
    eq = _mexpand(eq, recursive=True)
    if eq.is_number:
        return

    # 查看感兴趣的符号中是否有根式
    syms = set(syms) or eq.free_symbols  # _take 使用这个
    poly = eq.as_poly()
    gens = [g for g in poly.gens if _take(g)]
    if not gens:
        return

    # 用特征值生成重新构造多项式
    poly = eq.as_poly(*gens)

    # 如果不是多项式，例如 1 + sqrt(x)*exp(sqrt(x))，其中 gen 是 sqrt(x)
    if poly is None:
        return

    # - 指数包含感兴趣的符号（不予处理）
    if any(g.exp.has(*syms) for g in gens):
        return

    def _rads_bases_lcm(poly):
        # 如果所有的基数相同或者所有的根式都在一个项中，`lcm` 将是根式指数的分母的最小公倍数
        lcm = 1
        rads = set()
        bases = set()
        for g in poly.gens:
            q = _Q(g)
            if q != 1:
                rads.add(g)
                lcm = ilcm(lcm, q)
                bases.add(g.base)
        return rads, bases, lcm
    rads, bases, lcm = _rads_bases_lcm(poly)

    covsym = Dummy('p', nonnegative=True)

    # 只保留在根式中实际出现的符号，并更新 gens
    newsyms = set()
    for r in rads:
        newsyms.update(syms & r.free_symbols)
    if newsyms != syms:
        syms = newsyms
    # 将具有公共生成器的项放在一起
    drad = dict(zip(rads, range(len(rads))))
    rterms = {(): []}
    args = Add.make_args(poly.as_expr())
    for t in args:
        if _take(t):
            common = set(t.as_poly().gens).intersection(rads)
            key = tuple(sorted([drad[i] for i in common]))
        else:
            key = ()
        rterms.setdefault(key, []).append(t)
    others = Add(*rterms.pop(()))
    rterms = [Add(*rterms[k]) for k in rterms.keys()]

    # 输出将依赖于项处理的顺序，因此快速使其规范化
    # 将 rterms 列表反转，并转换为有序列表
    rterms = list(reversed(list(ordered(rterms))))

    # 初始条件：尚未找到解决方案
    ok = False  # we don't have a solution yet

    # 计算方程式的平方根深度
    depth = sqrt_depth(eq)

    # 如果 rterms 只有一个元素且不是 Add 类型并且 lcm 大于 2
    if len(rterms) == 1 and not (rterms[0].is_Add and lcm > 2):
        # 重置方程式为 rterms[0]^lcm - ((-others)^lcm)
        eq = rterms[0]**lcm - ((-others)**lcm)
        # 确认找到了解决方案
        ok = True

    # 如果找到了解决方案，更新新的平方根深度；否则保持初始的深度
    new_depth = sqrt_depth(eq) if ok else depth

    # rpt 增加 1，记录循环执行的次数，用于判断是否足够
    rpt += 1  # XXX how many repeats with others unchanging is enough?

    # 如果没有找到解决方案或者以下条件满足其中之一，则抛出未实现错误
    if not ok or (
                nwas is not None and len(rterms) == nwas and
                new_depth is not None and new_depth == depth and
                rpt > 3):
        raise NotImplementedError('Cannot remove all radicals')

    # 更新 flags 字典，包括 "cov" 和 rterms 的长度 "n"，以及 rpt 的次数 "rpt"
    flags.update({"cov": cov, "n": len(rterms), "rpt": rpt})

    # 对方程式进行反根操作，根据给定的符号和 flags 参数
    neq = unrad(eq, *syms, **flags)

    # 如果成功反根，则更新 eq 和 cov
    if neq:
        eq, cov = neq

    # 规范化方程式和覆盖率信息
    eq, cov = _canonical(eq, cov)

    # 返回处理后的方程式和覆盖率信息
    return eq, cov
# 导入延迟加载的模块
from sympy.solvers.bivariate import (
    bivariate_type, _solve_lambert, _filtered_gens)
```