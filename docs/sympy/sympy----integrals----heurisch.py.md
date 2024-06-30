# `D:\src\scipysrc\sympy\sympy\integrals\heurisch.py`

```
# 从 __future__ 模块导入 annotations，确保在 Python 3.7 之前的版本中支持类型注解
from __future__ import annotations

# 导入 collections 模块中的 defaultdict 类，用于创建默认字典
from collections import defaultdict
# 导入 functools 模块中的 reduce 函数，用于将一个函数累积地应用到序列的各个元素上
from functools import reduce
# 导入 itertools 模块中的 permutations 函数，用于生成指定序列的所有可能排列
from itertools import permutations

# 导入 sympy.core.add 模块中的 Add 类，表示 Sympy 中的加法表达式
from sympy.core.add import Add
# 导入 sympy.core.basic 模块中的 Basic 类，表示 Sympy 中所有基本表达式的基类
from sympy.core.basic import Basic
# 导入 sympy.core.mul 模块中的 Mul 类，表示 Sympy 中的乘法表达式
from sympy.core.mul import Mul
# 导入 sympy.core.symbol 模块中的 Wild, Dummy, Symbol 类，表示 Sympy 中的符号类
from sympy.core.symbol import Wild, Dummy, Symbol
# 导入 sympy.core.basic 模块中的 sympify 函数，用于将 Python 对象转换为 Sympy 对象
from sympy.core.basic import sympify
# 导入 sympy.core.numbers 模块中的 Rational, pi, I 类，表示 Sympy 中的有理数、圆周率和虚数单位
from sympy.core.numbers import Rational, pi, I
# 导入 sympy.core.relational 模块中的 Eq, Ne 类，表示 Sympy 中的相等和不等关系
from sympy.core.relational import Eq, Ne
# 导入 sympy.core.singleton 模块中的 S 类，表示 Sympy 中的单例对象
from sympy.core.singleton import S
# 导入 sympy.core.sorting 模块中的 ordered 函数，用于对表达式中的项进行排序
from sympy.core.sorting import ordered
# 导入 sympy.core.traversal 模块中的 iterfreeargs 函数，用于迭代表达式中的自由参数
from sympy.core.traversal import iterfreeargs

# 导入 sympy.functions 模块中的各种数学函数，如指数函数、三角函数等
from sympy.functions import exp, sin, cos, tan, cot, asin, atan
from sympy.functions import log, sinh, cosh, tanh, coth, asinh
from sympy.functions import sqrt, erf, erfi, li, Ei
from sympy.functions import besselj, bessely, besseli, besselk
from sympy.functions import hankel1, hankel2, jn, yn
# 导入 sympy.functions.elementary.complexes 模块中的复数函数，如绝对值、实部、虚部等
from sympy.functions.elementary.complexes import Abs, re, im, sign, arg
# 导入 sympy.functions.elementary.exponential 模块中的 LambertW 函数，表示 Lambert W 函数
from sympy.functions.elementary.exponential import LambertW
# 导入 sympy.functions.elementary.integers 模块中的 floor, ceiling 函数，表示向下取整和向上取整
from sympy.functions.elementary.integers import floor, ceiling
# 导入 sympy.functions.elementary.piecewise 模块中的 Piecewise 类，表示分段函数
from sympy.functions.elementary.piecewise import Piecewise
# 导入 sympy.functions.special.delta_functions 模块中的 Heaviside, DiracDelta 函数，表示 Heaviside 和 DiracDelta 函数
from sympy.functions.special.delta_functions import Heaviside, DiracDelta

# 导入 sympy.simplify.radsimp 模块中的 collect 函数，用于收集表达式中的同类项
from sympy.simplify.radsimp import collect

# 导入 sympy.logic.boolalg 模块中的 And, Or 类，表示逻辑与和逻辑或操作
from sympy.logic.boolalg import And, Or
# 导入 sympy.utilities.iterables 模块中的 uniq 函数，用于去除序列中的重复元素
from sympy.utilities.iterables import uniq

# 导入 sympy.polys 模块中的各种多项式相关函数和异常
from sympy.polys import quo, gcd, lcm, factor_list, cancel, PolynomialError
# 导入 sympy.polys.monomials 模块中的 itermonomials 函数，用于迭代多项式的单项式
from sympy.polys.monomials import itermonomials
# 导入 sympy.polys.polyroots 模块中的 root_factors 函数，用于计算多项式的根和因子
from sympy.polys.polyroots import root_factors

# 导入 sympy.polys.rings 模块中的 PolyRing 类，用于表示多项式环
from sympy.polys.rings import PolyRing
# 导入 sympy.polys.solvers 模块中的 solve_lin_sys 函数，用于解线性方程组
from sympy.polys.solvers import solve_lin_sys
# 导入 sympy.polys.constructor 模块中的 construct_domain 函数，用于构造域
from sympy.polys.constructor import construct_domain

# 导入 sympy.integrals.integrals 模块中的 integrate 函数，用于计算积分
from sympy.integrals.integrals import integrate

# 定义一个函数 components，接受表达式 f 和符号 x 作为参数，并返回表达式中的所有功能组件的集合
def components(f, x):
    """
    Returns a set of all functional components of the given expression
    which includes symbols, function applications and compositions and
    non-integer powers. Fractional powers are collected with
    minimal, positive exponents.

    Examples
    ========

    >>> from sympy import cos, sin
    >>> from sympy.abc import x
    >>> from sympy.integrals.heurisch import components

    >>> components(sin(x)*cos(x)**2, x)
    {x, sin(x), cos(x)}

    See Also
    ========

    heurisch
    """
    # 初始化结果集合
    result = set()

    # 如果表达式 f 包含符号 x
    if f.has_free(x):
        # 如果 f 是符号并且是可交换的
        if f.is_symbol and f.is_commutative:
            result.add(f)
        # 如果 f 是函数或者导数
        elif f.is_Function or f.is_Derivative:
            # 遍历 f 的参数
            for g in f.args:
                # 递归调用 components 函数，并将结果并入 result 集合
                result |= components(g, x)

            # 将 f 本身加入 result 集合
            result.add(f)
        # 如果 f 是幂
        elif f.is_Pow:
            # 将 f 的底数的组件加入 result 集合
            result |= components(f.base, x)

            # 如果 f 的指数不是整数
            if not f.exp.is_Integer:
                # 如果 f 的指数是有理数，将其添加为 f.base 的有理数幂
                if f.exp.is_Rational:
                    result.add(f.base**Rational(1, f.exp.q))
                # 否则递归调用 components 函数，并将结果并入 result 集合，同时将 f 加入集合
                else:
                    result |= components(f.exp, x) | {f}
        # 对于其他情况，递归调用 components 函数，并将结果并入 result 集合
        else:
            for g in f.args:
                result |= components(g, x)

    # 返回结果集合
    return result

# name -> [] of symbols 的缓存字典，存储符号列表
_symbols_cache: dict[str, list[Dummy]] = {}

# 定义 _symbols 函数，接受符号名称 name 和数量 n 作为参数
def _symbols(name, n):
    """获取本模块局部符号的向量"""
    try:
        # 尝试从符号缓存中获取指定名称的符号列表
        lsyms = _symbols_cache[name]
    except KeyError:
        # 如果符号缓存中不存在指定名称的符号列表，则创建一个空列表
        lsyms = []
        # 将新创建的空列表添加到符号缓存中
        _symbols_cache[name] = lsyms

    # 当符号列表长度小于指定的 n 时，继续添加新的 Dummy 符号
    while len(lsyms) < n:
        # 创建新的 Dummy 符号，并将其添加到符号列表中
        lsyms.append( Dummy('%s%i' % (name, len(lsyms))) )

    # 返回符号列表的前 n 个元素
    return lsyms[:n]
    # 引入 sympify 函数，将 f 转换为 SymPy 表达式
    f = sympify(f)
    # 如果 f 不含有自由变量 x，则返回 f*x
    if not f.has_free(x):
        return f*x

    # 使用 heurisch 函数计算 f 关于 x 的不定积分，并获取结果 res
    res = heurisch(f, x, rewrite, hints, mappings, retries, degree_offset,
                   unnecessary_permutations, _try_heurisch)
    # 如果 res 不是 Basic 对象（即 SymPy 的基本对象），直接返回 res
    if not isinstance(res, Basic):
        return res

    # 对 res 中的每个分母进行考虑，尝试找到可能为零的符号分母
    slns = []
    for d in ordered(denoms(res)):
        try:
            # 解方程 d=0，排除 x 后，得到解的列表，存入 slns 中
            slns += solve([d], dict=True, exclude=(x,))
        except NotImplementedError:
            pass
    # 如果没有找到任何解，直接返回 res
    if not slns:
        return res
    # 去重处理 slns 中的解
    slns = list(uniq(slns))

    # 移除在原始表达式中引起极点的解
    slns0 = []
    for d in denoms(f):
        try:
            slns0 += solve([d], dict=True, exclude=(x,))
        except NotImplementedError:
            pass
    slns = [s for s in slns if s not in slns0]
    # 如果没有剩余的解，直接返回 res
    if not slns:
        return res

    # 如果解的数量大于1，生成相应的方程，并将其与 slns 合并
    eqs = []
    for sub_dict in slns:
        eqs.extend([Eq(key, value) for key, value in sub_dict.items()])
    slns = solve(eqs, dict=True, exclude=(x,)) + slns

    # 对于 slns 中的每个情况，重新评估积分表达式
    pairs = []
    for sub_dict in slns:
        # 将 sub_dict 中的值代入 f 中，并重新使用 heurisch 函数计算积分表达式
        expr = heurisch(f.subs(sub_dict), x, rewrite, hints, mappings, retries,
                        degree_offset, unnecessary_permutations,
                        _try_heurisch)
        # 构造条件，当 sub_dict 中的键值对满足时，条件成立
        cond = And(*[Eq(key, value) for key, value in sub_dict.items()])
        # 通用情况为 sub_dict 中的键值对不满足时，条件成立
        generic = Or(*[Ne(key, value) for key, value in sub_dict.items()])
        # 如果 expr 为 None，则用 integrate 函数重新计算 f 的积分表达式
        if expr is None:
            expr = integrate(f.subs(sub_dict), x)
        # 将结果表达式 expr 和条件 cond 存入 pairs 中
        pairs.append((expr, cond))
    # 如果输入的列表 `pairs` 只有一个元素
    if len(pairs) == 1:
        # 将列表重新赋值为包含两个元组的列表：
        # 第一个元组：调用 heurisch 函数的结果和 `generic` 变量
        # 第二个元组：原先列表中的第一个元素的第一个元素和 True
        pairs = [
            (heurisch(f, x, rewrite, hints, mappings, retries,
                      degree_offset, unnecessary_permutations,
                      _try_heurisch),
             generic),
            (pairs[0][0], True)
        ]
    else:
        # 否则，向列表 `pairs` 添加一个元组：
        # 元组内容为调用 heurisch 函数的结果和 True
        pairs.append(
            (heurisch(f, x, rewrite, hints, mappings, retries,
                      degree_offset, unnecessary_permutations,
                      _try_heurisch),
             True)
        )
    # 返回一个 Piecewise 对象，使用 `pairs` 列表的内容作为参数
    return Piecewise(*pairs)
class BesselTable:
    """
    Derivatives of Bessel functions of orders n and n-1
    in terms of each other.

    See the docstring of DiffCache.
    """

    def __init__(self):
        # 初始化空的表格
        self.table = {}
        # 创建虚拟符号 n 和 z
        self.n = Dummy('n')
        self.z = Dummy('z')
        # 调用内部方法创建表格
        self._create_table()

    def _create_table(t):
        # 从实例 t 中获取表格、n 和 z
        table, n, z = t.table, t.n, t.z
        # 针对四种贝塞尔函数进行计算并填充表格
        for f in (besselj, bessely, hankel1, hankel2):
            table[f] = (f(n-1, z) - n*f(n, z)/z,
                        (n-1)*f(n-1, z)/z - f(n, z))

        # 计算贝塞尔修正函数的导数并添加到表格
        f = besseli
        table[f] = (f(n-1, z) - n*f(n, z)/z,
                    (n-1)*f(n-1, z)/z + f(n, z))
        f = besselk
        table[f] = (-f(n-1, z) - n*f(n, z)/z,
                    (n-1)*f(n-1, z)/z - f(n, z))

        # 针对第一类和第二类整数阶贝塞尔函数计算并填充表格
        for f in (jn, yn):
            table[f] = (f(n-1, z) - (n+1)*f(n, z)/z,
                        (n-1)*f(n-1, z)/z - f(n, z))

    def diffs(t, f, n, z):
        # 如果函数 f 在表格中，则返回其在点 (n, z) 处的两个导数
        if f in t.table:
            diff0, diff1 = t.table[f]
            repl = [(t.n, n), (t.z, z)]
            return (diff0.subs(repl), diff1.subs(repl))

    def has(t, f):
        # 检查函数 f 是否在表格中
        return f in t.table

_bessel_table = None

class DiffCache:
    """
    Store for derivatives of expressions.

    Explanation
    ===========

    The standard form of the derivative of a Bessel function of order n
    contains two Bessel functions of orders n-1 and n+1, respectively.
    Such forms cannot be used in parallel Risch algorithm, because
    there is a linear recurrence relation between the three functions
    while the algorithm expects that functions and derivatives are
    represented in terms of algebraically independent transcendentals.

    The solution is to take two of the functions, e.g., those of orders
    n and n-1, and to express the derivatives in terms of the pair.
    To guarantee that the proper form is used the two derivatives are
    cached as soon as one is encountered.

    Derivatives of other functions are also cached at no extra cost.
    All derivatives are with respect to the same variable `x`.
    """

    def __init__(self, x):
        # 初始化缓存和变量 x
        self.cache = {}
        self.x = x

        # 全局变量 _bessel_table 用于存储贝塞尔函数表格
        global _bessel_table
        if not _bessel_table:
            _bessel_table = BesselTable()

    def get_diff(self, f):
        # 获取缓存
        cache = self.cache

        # 如果 f 已经在缓存中，则直接返回
        if f in cache:
            pass
        # 如果 f 不是一个表达式或者其函数不在贝塞尔函数表格中，则计算其导数并添加到缓存
        elif (not hasattr(f, 'func') or
              not _bessel_table.has(f.func)):
            cache[f] = cancel(f.diff(self.x))
        # 否则，利用贝塞尔函数表格计算 f 的导数，并添加到缓存中
        else:
            n, z = f.args
            d0, d1 = _bessel_table.diffs(f.func, n, z)
            dz = self.get_diff(z)
            cache[f] = d0*dz
            cache[f.func(n-1, z)] = d1*dz

        return cache[f]

def heurisch(f, x, rewrite=False, hints=None, mappings=None, retries=3,
             degree_offset=0, unnecessary_permutations=None,
             _try_heurisch=None):
    """
    Compute indefinite integral using heuristic Risch algorithm.

    Explanation
    ===========

    This function computes the indefinite integral of expression f with respect to variable x
    using a heuristic variant of the Risch algorithm. Various options control the behavior
    of the integration process.

    """
    """
    This is a heuristic approach to indefinite integration in finite
    terms using the extended heuristic (parallel) Risch algorithm, based
    on Manuel Bronstein's "Poor Man's Integrator".

    The algorithm supports various classes of functions including
    transcendental elementary or special functions like Airy,
    Bessel, Whittaker and Lambert.

    Note that this algorithm is not a decision procedure. If it isn't
    able to compute the antiderivative for a given function, then this is
    not a proof that such a function does not exist. One should use
    recursive Risch algorithm in such case. It's an open question if
    this algorithm can be made a full decision procedure.

    This is an internal integrator procedure. You should use the top level
    'integrate' function in most cases, as this procedure needs some
    preprocessing steps and otherwise may fail.

    Specification
    =============

     heurisch(f, x, rewrite=False, hints=None)

       where
         f : expression
         x : symbol

         rewrite -> force rewrite 'f' in terms of 'tan' and 'tanh'
         hints   -> a list of functions that may appear in anti-derivative

          - hints = None          --> no suggestions at all
          - hints = [ ]           --> try to figure out
          - hints = [f1, ..., fn] --> we know better

    Examples
    ========

    >>> from sympy import tan
    >>> from sympy.integrals.heurisch import heurisch
    >>> from sympy.abc import x, y

    >>> heurisch(y*tan(x), x)
    y*log(tan(x)**2 + 1)/2

    See Manuel Bronstein's "Poor Man's Integrator":

    References
    ==========

    .. [1] https://www-sop.inria.fr/cafe/Manuel.Bronstein/pmint/index.html

    For more information on the implemented algorithm refer to:

    .. [2] K. Geddes, L. Stefanus, On the Risch-Norman Integration
       Method and its Implementation in Maple, Proceedings of
       ISSAC'89, ACM Press, 212-217.

    .. [3] J. H. Davenport, On the Parallel Risch Algorithm (I),
       Proceedings of EUROCAM'82, LNCS 144, Springer, 144-157.

    .. [4] J. H. Davenport, On the Parallel Risch Algorithm (III):
       Use of Tangents, SIGSAM Bulletin 16 (1982), 3-6.

    .. [5] J. H. Davenport, B. M. Trager, On the Parallel Risch
       Algorithm (II), ACM Transactions on Mathematical
       Software 11 (1985), 356-362.

    See Also
    ========

    sympy.integrals.integrals.Integral.doit
    sympy.integrals.integrals.Integral
    sympy.integrals.heurisch.components
    """

    # 将输入的表达式转换成 Sympy 表达式对象
    f = sympify(f)

    # 如果 _try_heurisch 不是 True，则检查表达式中是否包含无法处理的函数，
    # 如果有则直接返回，不尝试进行启发式积分
    if _try_heurisch is not True:
        if f.has(Abs, re, im, sign, Heaviside, DiracDelta, floor, ceiling, arg):
            return

    # 如果表达式 f 不含有自变量 x，则返回 f 乘以 x
    if not f.has_free(x):
        return f*x

    # 如果 f 不是加法表达式，则将其分解为独立部分和关于 x 的部分
    if not f.is_Add:
        indep, f = f.as_independent(x)
    else:
        indep = S.One
    # 定义可重写的函数集合，每个键是待重写的函数组合，值是对应的重写规则函数
    rewritables = {
        (sin, cos, cot): tan,
        (sinh, cosh, coth): tanh,
    }

    # 如果需要进行重写操作
    if rewrite:
        # 遍历可重写的函数组合及其对应的重写规则，对函数 f 进行重写操作
        for candidates, rule in rewritables.items():
            f = f.rewrite(candidates, rule)
    else:
        # 如果不需要重写，则检查是否有任何可重写的函数组合在 f 中存在
        for candidates in rewritables.keys():
            if f.has(*candidates):
                break
        else:
            # 如果没有找到可重写的函数组合，则设置 rewrite 为 True，准备进行重写操作
            rewrite = True

    # 计算函数 f 关于变量 x 的组成项
    terms = components(f, x)
    # 创建一个新的变量 x 的差分缓存
    dcache = DiffCache(x)

    # 对 terms 的副本进行操作，将其元素添加到 terms 中
    for g in set(terms):  # 使用 terms 的副本
        terms |= components(dcache.get_diff(g), x)

    # XXX: 下面的注释行使 heurisch 在 PYTHONHASHSEED 和集合的迭代顺序方面更加确定性。
    # 这里并不是唯一一个使用集合迭代的地方，但这个可能是最重要的地方。
    # 理论上，这里的顺序不应该影响结果，但不同的顺序可能会暴露出不同代码路径中的潜在 bug，
    # 所以保持非确定性可能更好一些。
    
    # 将 terms 转换为有序列表（已被注释掉的代码）
    #
    # terms = list(ordered(terms))

    # TODO: 缓存是让排列组合能够正常工作的重要因素。需要修改此处。
    
    # 创建变量 V，用于映射表达式，其长度与 terms 相等
    V = _symbols('x', len(terms))


    # 将表达式映射从最大到最小进行排序（最后一个始终是 x）。
    mapping = list(reversed(list(zip(*ordered(                          #
        [(a[0].as_independent(x)[1], a) for a in zip(terms, V)])))[1])) #
    rev_mapping = {v: k for k, v in mapping}                            #
    if mappings is None:                                                #
        # 优化映射的排列组合数量
        assert mapping[-1][0] == x # 如果不是，找到并修正这个注释
        unnecessary_permutations = [mapping.pop(-1)]
        # 只对对象类型进行排列组合，让类型的顺序决定替换的顺序
        types = defaultdict(list)
        for i in mapping:
            types[type(i)].append(i)
        mapping = [types[i] for i in types]
        # 定义一个函数，用于生成映射的所有排列组合
        def _iter_mappings():
            for i in permutations(mapping):
                yield [j for i in i for j in i]
        mappings = _iter_mappings()
    else:
        unnecessary_permutations = unnecessary_permutations or []

    # 定义一个函数，用于对表达式进行替换操作
    def _substitute(expr):
        return expr.subs(mapping)

    # 遍历所有的映射组合
    for mapping in mappings:
        mapping = list(mapping)
        mapping = mapping + unnecessary_permutations
        # 对 terms 中的每个 g，使用差分缓存对其进行替换，得到 diffs
        diffs = [ _substitute(dcache.get_diff(g)) for g in terms ]
        # 对 diffs 中的每个 g，获取其分母部分
        denoms = [ g.as_numer_denom()[1] for g in diffs ]
        # 如果所有的 denoms 都是 V 的多项式，并且对 f 进行替换后仍是有理函数，则执行以下操作
        if all(h.is_polynomial(*V) for h in denoms) and _substitute(f).is_rational_function(*V):
            # 计算 denoms 的最小公倍数作为 denom
            denom = reduce(lambda p, q: lcm(p, q, *V), denoms)
            break
    else:
        # 如果没有找到符合条件的映射组合，则根据需要设置 rewrite 为 True，调用 heurisch 函数
        if not rewrite:
            result = heurisch(f, x, rewrite=True, hints=hints,
                unnecessary_permutations=unnecessary_permutations)

            if result is not None:
                return indep*result
        # 如果以上条件都不符合，则返回 None
        return None

    # 对 diffs 中的每个 g，将 denom 与 g 相乘并化简，得到 numers
    numers = [ cancel(denom*g) for g in diffs ]
    # 定义一个内部函数_derivation，接受一个参数h，返回一个表达式
    def _derivation(h):
        # 返回一个加法表达式，其中每一项是数字乘以h对变量v的导数
        return Add(*[ d * h.diff(v) for d, v in zip(numers, V) ])

    # 定义一个内部函数_deflation，接受一个多项式参数p，返回一个简化后的多项式
    def _deflation(p):
        # 遍历变量集合V中的每个变量y
        for y in V:
            # 如果多项式p中不包含变量y，则继续下一个变量
            if not p.has(y):
                continue

            # 如果_derivation(p)不为零
            if _derivation(p) is not S.Zero:
                # 将p表示为y的多项式的原始部分和主要部分
                c, q = p.as_poly(y).primitive()
                # 递归调用_deflation，返回_c的deflation乘以gcd(q, q.diff(y))的表达式
                return _deflation(c)*gcd(q, q.diff(y)).as_expr()

        # 如果所有变量y都不符合条件，则返回原始多项式p
        return p

    # 定义一个内部函数_splitter，接受一个多项式参数p，返回一个元组
    def _splitter(p):
        # 遍历变量集合V中的每个变量y
        for y in V:
            # 如果多项式p中不包含变量y，则继续下一个变量
            if not p.has(y):
                continue

            # 如果_derivation(y)不为零
            if _derivation(y) is not S.Zero:
                # 将p表示为y的多项式的原始部分和主要部分
                c, q = p.as_poly(y).primitive()

                # 将q转换为表达式
                q = q.as_expr()

                # 计算gcd(q, _derivation(q), y)和quo(h, gcd(q, q.diff(y), y), y)
                h = gcd(q, _derivation(q), y)
                s = quo(h, gcd(q, q.diff(y), y), y)

                # 递归调用_splitter并处理cancel(q / s)的返回值
                c_split = _splitter(c)

                # 如果s作为y的多项式的次数为零
                if s.as_poly(y).degree() == 0:
                    return (c_split[0], q * c_split[1])

                # 递归调用_splitter并处理cancel(q / s)的返回值
                q_split = _splitter(cancel(q / s))

                # 返回结果元组
                return (c_split[0]*q_split[0]*s, c_split[1]*q_split[1])

        # 如果所有变量y都不符合条件，则返回(S.One, p)
        return (S.One, p)

    # 初始化一个空字典special
    special = {}

    # 遍历terms中的每个term
    for term in terms:
        # 如果term是函数
        if term.is_Function:
            # 如果term是tan函数实例
            if isinstance(term, tan):
                # 将特定的值1 + _substitute(term)**2映射到False
                special[1 + _substitute(term)**2] = False
            # 如果term是tanh函数实例
            elif isinstance(term, tanh):
                # 将特定的值1 + _substitute(term)和1 - _substitute(term)映射到False
                special[1 + _substitute(term)] = False
                special[1 - _substitute(term)] = False
            # 如果term是LambertW函数实例
            elif isinstance(term, LambertW):
                # 将_substitute(term)映射到True
                special[_substitute(term)] = True

    # 计算函数_f的值
    F = _substitute(f)

    # 将F表示为分子P和分母Q
    P, Q = F.as_numer_denom()

    # 分别对分母denom和分母Q进行_splitter操作
    u_split = _splitter(denom)
    v_split = _splitter(Q)

    # 将v_split中的项列表和特殊字典中的键组合成一个集合polys
    polys = set(list(v_split) + [ u_split[0] ] + list(special.keys()))

    # 计算s的值
    s = u_split[0] * Mul(*[ k for k, v in special.items() if v ])
    
    # 将s，P和Q转换为多项式列表polified
    polified = [ p.as_poly(*V) for p in [s, P, Q] ]

    # 如果polified中有任何值为None，则返回None
    if None in polified:
        return None

    #--- definitions for _integrate
    # 计算多项式polified中每个项的最大次数
    a, b, c = [ p.total_degree() for p in polified ]

    # 计算poly_denom的值
    poly_denom = (s * v_split[0] * _deflation(v_split[1])).as_expr()

    # 定义一个内部函数_exponent，计算g的幂
    def _exponent(g):
        # 如果g是幂
        if g.is_Pow:
            # 如果g.exp是有理数且分母不等于1
            if g.exp.is_Rational and g.exp.q != 1:
                # 如果g.exp的分子大于0
                if g.exp.p > 0:
                    return g.exp.p + g.exp.q - 1
                else:
                    return abs(g.exp.p + g.exp.q)
            else:
                return 1
        # 如果g不是原子并且有参数
        elif not g.is_Atom and g.args:
            # 返回参数中每个项的最大幂
            return max(_exponent(h) for h in g.args)
        else:
            return 1

    # 计算A和B的值
    A, B = _exponent(f), a + max(b, c)

    # 如果A大于1且B大于1
    if A > 1 and B > 1:
        # 计算所有符合条件的monoms元组
        monoms = tuple(ordered(itermonomials(V, A + B - 1 + degree_offset)))
    else:
        # 计算所有符合条件的monoms元组
        monoms = tuple(ordered(itermonomials(V, A + B + degree_offset)))

    # 根据monoms的长度定义一个多项式系数列表poly_coeffs
    poly_coeffs = _symbols('A', len(monoms))

    # 计算poly_part的值
    poly_part = Add(*[ poly_coeffs[i]*monomial
        for i, monomial in enumerate(monoms) ])

    # 初始化一个空集合reducibles
    reducibles = set()

    # 对polys中的每个多项式poly进行排序
    for poly in ordered(polys):
        # 将多项式poly分解为系数和因子列表
        coeff, factors = factor_list(poly, *V)
        # 将系数添加到reducibles集合中
        reducibles.add(coeff)
        # 将因子列表中的每个因子添加到reducibles集合中
        reducibles.update(fact for fact, mul in factors)

    # 如果V中的所有项都是符号
    if all(isinstance(_, Symbol) for _ in V):
        # 将F中的自由符号减去V中的符号添加到more_free中
        more_free = F.free_symbols - set(V)
    else:
        # 将表达式 F 转换为虚拟对象 Fd
        Fd = F.as_dummy()
        # 使用虚拟对象 Fd 替换 V 中的变量，生成一个更多自由符号的集合
        more_free = Fd.xreplace(dict(zip(V, (Dummy() for _ in V)))).free_symbols & Fd.free_symbols
    
    if not more_free:
        # 如果没有更多的自由生成器，则所有的自由生成器都已在 V 中被识别
        solution = _integrate('Q')

        # 如果无法找到解决方案，则尝试默认积分
        if solution is None:
            solution = _integrate()
    else:
        # 如果存在更多的自由生成器，则尝试默认积分
        solution = _integrate()

    if solution is not None:
        # 将解决方案中的变量反向映射替换为 antideriv
        antideriv = solution.subs(rev_mapping)
        # 取消 antideriv 中的公共因子并展开
        antideriv = cancel(antideriv).expand()

        # 如果 antideriv 是一个加法表达式，则取出其中关于 x 的部分
        if antideriv.is_Add:
            antideriv = antideriv.as_independent(x)[1]

        # 返回独立变量乘以 antideriv
        return indep * antideriv
    else:
        # 如果无法找到积分解，且重试次数 retries 大于等于 0，则尝试启发式积分
        if retries >= 0:
            result = heurisch(f, x, mappings=mappings, rewrite=rewrite, hints=hints, retries=retries - 1, unnecessary_permutations=unnecessary_permutations)

            # 如果找到结果，则返回独立变量乘以结果
            if result is not None:
                return indep * result

        # 如果无法找到任何解，则返回 None
        return None
```