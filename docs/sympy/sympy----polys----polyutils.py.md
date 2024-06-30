# `D:\src\scipysrc\sympy\sympy\polys\polyutils.py`

```
"""Useful utilities for higher level polynomial classes. """

# 引入 Python 未来的注释，以便支持 annotations
from __future__ import annotations

# 导入特定的外部模块或类型
from sympy.external.gmpy import GROUND_TYPES

# 导入 SymPy 核心模块及相关类
from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
    expand_mul, expand_multinomial)

# 导入 SymPy 表达式工具模块
from sympy.core.exprtools import decompose_power, decompose_power_rat

# 导入 SymPy 核心数字模块中的特定成员
from sympy.core.numbers import _illegal

# 导入 SymPy 多项式相关的异常类
from sympy.polys.polyerrors import PolynomialError, GeneratorsError

# 导入 SymPy 多项式选项构建器
from sympy.polys.polyoptions import build_options

# 导入正则表达式模块
import re

# 定义全局变量 _gens_order，用于指定生成器的顺序
_gens_order = {
    'a': 301, 'b': 302, 'c': 303, 'd': 304,
    'e': 305, 'f': 306, 'g': 307, 'h': 308,
    'i': 309, 'j': 310, 'k': 311, 'l': 312,
    'm': 313, 'n': 314, 'o': 315, 'p': 216,
    'q': 217, 'r': 218, 's': 219, 't': 220,
    'u': 221, 'v': 222, 'w': 223, 'x': 124,
    'y': 125, 'z': 126,
}

# 定义最大顺序常量
_max_order = 1000

# 编译正则表达式对象，用于解析生成器名称
_re_gen = re.compile(r"^(.*?)(\d*)$", re.MULTILINE)


def _nsort(roots, separated=False):
    """Sort the numerical roots putting the real roots first, then sorting
    according to real and imaginary parts. If ``separated`` is True, then
    the real and imaginary roots will be returned in two lists, respectively.

    This routine tries to avoid issue 6137 by separating the roots into real
    and imaginary parts before evaluation. In addition, the sorting will raise
    an error if any computation cannot be done with precision.
    """
    # 检查所有根是否为数值类型，否则抛出未实现的错误
    if not all(r.is_number for r in roots):
        raise NotImplementedError
    # 如果根列表为空，返回空列表或空的实数和虚数根列表
    if not len(roots):
        return [] if not separated else ([], [])
    # 获取每个根的实部和虚部的实数值
    key = [[i.n(2).as_real_imag()[0] for i in r.as_real_imag()] for r in roots]
    # 如果存在计算精度为1的情况，抛出未实现的错误
    if len(roots) > 1 and any(i._prec == 1 for k in key for i in k):
        raise NotImplementedError("could not compute root with precision")
    # 插入一个键来指示根是否有虚部
    key = [(1 if i else 0, r, i) for r, i in key]
    # 按照键对根进行排序
    key = sorted(zip(key, roots))
    # 如果需要分开返回实数和虚数根，则分开处理
    if separated:
        r = []
        i = []
        for (im, _, _), v in key:
            if im:
                i.append(v)
            else:
                r.append(v)
        return r, i
    # 否则返回按顺序排列的根列表
    _, roots = zip(*key)
    return list(roots)


def _sort_gens(gens, **args):
    """Sort generators in a reasonably intelligent way. """
    # 构建选项对象
    opt = build_options(args)

    # 初始化生成器顺序字典和关键字
    gens_order, wrt = {}, None

    # 如果选项不为空，则初始化生成器顺序和关键字
    if opt is not None:
        gens_order, wrt = {}, opt.wrt

        # 遍历选项中的排序列表，并为每个生成器分配顺序
        for i, gen in enumerate(opt.sort):
            gens_order[gen] = i + 1
    # 定义一个函数，用于生成排序键的函数
    def order_key(gen):
        # 将输入的生成器转换为字符串
        gen = str(gen)
    
        # 如果提供了 wrt 参数，则尝试使用它来确定排序顺序
        if wrt is not None:
            try:
                # 返回一个元组，元组包含三个值：按照 wrt 中 gen 的位置逆序排序的值、gen 本身、0
                return (-len(wrt) + wrt.index(gen), gen, 0)
            except ValueError:
                pass
    
        # 使用正则表达式匹配 gen，并提取名称和索引
        name, index = _re_gen.match(gen).groups()
    
        # 将索引转换为整数，如果没有索引，则默认为 0
        if index:
            index = int(index)
        else:
            index = 0
    
        # 尝试使用 gens_order 字典中的值来确定排序顺序
        try:
            return (gens_order[name], name, index)
        except KeyError:
            pass
    
        # 如果 gens_order 中没有对应的名称，尝试使用 _gens_order 字典中的值
        try:
            return (_gens_order[name], name, index)
        except KeyError:
            pass
    
        # 如果以上都失败，则返回一个默认的最大排序顺序值，名称和索引
        return (_max_order, name, index)
    
    try:
        # 对 gens 列表进行排序，使用 order_key 函数作为排序键
        gens = sorted(gens, key=order_key)
    except TypeError:  # 处理排序时可能出现的类型错误异常，通常不会触发，所以设置 pragma: no cover
        pass
    
    # 返回经过排序后的 gens 列表，转换为元组形式
    return tuple(gens)
def _unify_gens(f_gens, g_gens):
    """Unify generators in a reasonably intelligent way. """
    # 将生成器转换为列表
    f_gens = list(f_gens)
    g_gens = list(g_gens)

    # 如果两个生成器列表相同，则直接返回元组形式的 f_gens
    if f_gens == g_gens:
        return tuple(f_gens)

    gens, common, k = [], [], 0

    # 找出两个列表中共同的生成器
    for gen in f_gens:
        if gen in g_gens:
            common.append(gen)

    # 将 g_gens 中的共同生成器按顺序替换为 f_gens 中对应位置的生成器
    for i, gen in enumerate(g_gens):
        if gen in common:
            g_gens[i], k = common[k], k + 1

    # 根据共同生成器的顺序重新排列 f_gens 和 g_gens 中的生成器
    for gen in common:
        i = f_gens.index(gen)

        gens.extend(f_gens[:i])
        f_gens = f_gens[i + 1:]

        i = g_gens.index(gen)

        gens.extend(g_gens[:i])
        g_gens = g_gens[i + 1:]

        gens.append(gen)

    # 将剩余的生成器添加到 gens 中
    gens.extend(f_gens)
    gens.extend(g_gens)

    # 返回排序后的生成器列表
    return tuple(gens)


def _analyze_gens(gens):
    """Support for passing generators as `*gens` and `[gens]`. """
    # 如果 gens 中只有一个元素且该元素可迭代，则返回该元素的元组形式
    if len(gens) == 1 and hasattr(gens[0], '__iter__'):
        return tuple(gens[0])
    else:
        return tuple(gens)


def _sort_factors(factors, **args):
    """Sort low-level factors in increasing 'complexity' order. """

    # XXX: GF(p) does not support comparisons so we need a key function to sort
    # the factors if python-flint is being used. A better solution might be to
    # add a sort key method to each domain.

    # 定义排序因子的关键函数 order_key
    def order_key(factor):
        if isinstance(factor, _GF_types):
            return int(factor)
        elif isinstance(factor, list):
            return [order_key(f) for f in factor]
        else:
            return factor

    # 定义排序多重因子的关键函数 order_if_multiple_key
    def order_if_multiple_key(factor):
        (f, n) = factor
        return (len(f), n, order_key(f))

    # 定义排序非多重因子的关键函数 order_no_multiple_key
    def order_no_multiple_key(f):
        return (len(f), order_key(f))

    # 根据参数中的 'multiple' 键决定是否排序多重因子
    if args.get('multiple', True):
        return sorted(factors, key=order_if_multiple_key)
    else:
        return sorted(factors, key=order_no_multiple_key)


# 定义非法类型列表
illegal_types = [type(obj) for obj in _illegal]
# 创建包含 _illegal 列表中浮点数部分的 finf 列表
finf = [float(i) for i in _illegal[1:3]]


def _not_a_coeff(expr):
    """Do not treat NaN and infinities as valid polynomial coefficients. """
    # 如果 expr 的类型在 illegal_types 中或者是 finf 中的无穷大值，则返回 True
    if type(expr) in illegal_types or expr in finf:
        return True
    # 如果 expr 是浮点数且不等于自身，则可能是 NaN
    if isinstance(expr, float) and float(expr) != expr:
        return True  # nan
    return  # could be


def _parallel_dict_from_expr_if_gens(exprs, opt):
    """Transform expressions into a multinomial form given generators. """
    k, indices = len(opt.gens), {}

    # 根据 opt.gens 中的生成器创建索引字典 indices
    for i, g in enumerate(opt.gens):
        indices[g] = i

    polys = []
    # 遍历表达式列表中的每一个表达式
    for expr in exprs:
        # 初始化一个空的多项式字典
        poly = {}

        # 如果表达式是一个等式，将其转换为左侧减去右侧的形式
        if expr.is_Equality:
            expr = expr.lhs - expr.rhs

        # 遍历并处理表达式中的每一项
        for term in Add.make_args(expr):
            # 初始化系数和单项式指数列表
            coeff, monom = [], [0]*k

            # 遍历并处理每一项中的因子
            for factor in Mul.make_args(term):
                # 如果因子不是系数且是一个数字
                if not _not_a_coeff(factor) and factor.is_Number:
                    coeff.append(factor)
                else:
                    try:
                        # 解析因子的幂指数和基数，根据选项决定是否展开幂
                        if opt.series is False:
                            base, exp = decompose_power(factor)

                            # 如果指数为负数，转换为正指数和倒数的幂
                            if exp < 0:
                                exp, base = -exp, Pow(base, -S.One)
                        else:
                            base, exp = decompose_power_rat(factor)

                        # 根据基数确定其在指数数组中的位置，记录指数
                        monom[indices[base]] = exp
                    except KeyError:
                        # 如果因子包含自由生成元，则引发多项式错误
                        if not factor.has_free(*opt.gens):
                            coeff.append(factor)
                        else:
                            raise PolynomialError("%s contains an element of "
                                                  "the set of generators." % factor)

            # 将单项式指数数组转换为不可变元组
            monom = tuple(monom)

            # 如果该单项式已经在多项式字典中，则累加系数
            if monom in poly:
                poly[monom] += Mul(*coeff)
            else:
                poly[monom] = Mul(*coeff)

        # 将构建好的多项式字典添加到多项式列表中
        polys.append(poly)

    # 返回最终得到的多项式列表和生成元集合
    return polys, opt.gens
def _parallel_dict_from_expr_no_gens(exprs, opt):
    """Transform expressions into a multinomial form and figure out generators. """
    # 定义内部函数根据选项确定是否将因子视为系数
    if opt.domain is not None:
        def _is_coeff(factor):
            return factor in opt.domain
    elif opt.extension is True:
        def _is_coeff(factor):
            return factor.is_algebraic
    elif opt.greedy is not False:
        def _is_coeff(factor):
            return factor is S.ImaginaryUnit
    else:
        def _is_coeff(factor):
            return factor.is_number

    # 初始化生成器集合和表达式列表
    gens, reprs = set(), []

    # 遍历表达式列表中的每个表达式
    for expr in exprs:
        terms = []

        # 如果表达式是等式，则将其转换为左右两侧差的形式
        if expr.is_Equality:
            expr = expr.lhs - expr.rhs

        # 将表达式按照加法拆分成项
        for term in Add.make_args(expr):
            coeff, elements = [], {}

            # 对每个因子进行处理
            for factor in Mul.make_args(term):
                # 如果因子不是系数并且满足条件，则将其加入系数列表
                if not _not_a_coeff(factor) and (factor.is_Number or _is_coeff(factor)):
                    coeff.append(factor)
                else:
                    # 如果不符合系数条件，根据选项判断是否展开幂次
                    if opt.series is False:
                        base, exp = decompose_power(factor)

                        # 处理负指数
                        if exp < 0:
                            exp, base = -exp, Pow(base, -S.One)
                    else:
                        base, exp = decompose_power_rat(factor)

                    # 更新元素字典中的基数和对应的指数
                    elements[base] = elements.setdefault(base, 0) + exp
                    # 将基数添加到生成器集合中
                    gens.add(base)

            # 将当前项的系数和元素字典组成的元组添加到terms中
            terms.append((coeff, elements))

        # 将当前表达式的terms添加到reprs中
        reprs.append(terms)

    # 对生成器集合进行排序，并获取其长度和索引字典
    gens = _sort_gens(gens, opt=opt)
    k, indices = len(gens), {}

    # 将生成器和其索引添加到indices字典中
    for i, g in enumerate(gens):
        indices[g] = i

    # 初始化多项式列表
    polys = []

    # 遍历表达式列表中的每个terms
    for terms in reprs:
        poly = {}

        # 遍历terms中的每个系数和元素字典
        for coeff, term in terms:
            monom = [0]*k

            # 根据生成器的索引更新monom中的指数
            for base, exp in term.items():
                monom[indices[base]] = exp

            # 将monom转换为元组
            monom = tuple(monom)

            # 更新多项式字典中对应monom的系数
            if monom in poly:
                poly[monom] += Mul(*coeff)
            else:
                poly[monom] = Mul(*coeff)

        # 将当前多项式添加到polys中
        polys.append(poly)

    # 返回多项式列表和生成器元组
    return polys, tuple(gens)


def _dict_from_expr_if_gens(expr, opt):
    """Transform an expression into a multinomial form given generators. """
    # 调用并返回_parallel_dict_from_expr_if_gens函数的结果
    (poly,), gens = _parallel_dict_from_expr_if_gens((expr,), opt)
    return poly, gens


def _dict_from_expr_no_gens(expr, opt):
    """Transform an expression into a multinomial form and figure out generators. """
    # 调用并返回_parallel_dict_from_expr_no_gens函数的结果
    (poly,), gens = _parallel_dict_from_expr_no_gens((expr,), opt)
    return poly, gens


def parallel_dict_from_expr(exprs, **args):
    """Transform expressions into a multinomial form. """
    # 调用并返回_parallel_dict_from_expr函数的结果
    reps, opt = _parallel_dict_from_expr(exprs, build_options(args))
    return reps, opt.gens


def _parallel_dict_from_expr(exprs, opt):
    """Transform expressions into a multinomial form. """
    # 如果选项中expand不为False，则对每个表达式进行展开
    if opt.expand is not False:
        exprs = [ expr.expand() for expr in exprs ]

    # 如果表达式列表中有非交换表达式，则抛出异常
    if any(expr.is_commutative is False for expr in exprs):
        raise PolynomialError('non-commutative expressions are not supported')
    # 如果 opt.gens 为真，则调用 _parallel_dict_from_expr_if_gens 函数处理 exprs 和 opt 对象
    if opt.gens:
        reps, gens = _parallel_dict_from_expr_if_gens(exprs, opt)
    # 否则，调用 _parallel_dict_from_expr_no_gens 函数处理 exprs 和 opt 对象
    else:
        reps, gens = _parallel_dict_from_expr_no_gens(exprs, opt)

    # 返回处理后的结果 reps 和克隆了 gens 属性的新的 opt 对象
    return reps, opt.clone({'gens': gens})
def dict_from_expr(expr, **args):
    """将表达式转换为多项式形式的字典。"""
    # 调用内部函数 _dict_from_expr 处理表达式，使用传入的参数构建选项
    rep, opt = _dict_from_expr(expr, build_options(args))
    # 返回转换后的字典表示和生成器列表
    return rep, opt.gens


def _dict_from_expr(expr, opt):
    """将表达式转换为多项式形式的字典。"""
    # 检查表达式是否是可交换的，否则抛出异常
    if expr.is_commutative is False:
        raise PolynomialError('non-commutative expressions are not supported')

    def _is_expandable_pow(expr):
        # 检查是否可以展开幂运算的表达式
        return (expr.is_Pow and expr.exp.is_positive and expr.exp.is_Integer
                and expr.base.is_Add)

    # 如果选项中的 expand 不是 False，则尝试展开表达式
    if opt.expand is not False:
        if not isinstance(expr, (Expr, Eq)):
            raise PolynomialError('expression must be of type Expr')
        expr = expr.expand()
        # 循环展开表达式中可展开的幂运算和乘法
        while any(_is_expandable_pow(i) or i.is_Mul and
            any(_is_expandable_pow(j) for j in i.args) for i in
                Add.make_args(expr)):

            expr = expand_multinomial(expr)
        while any(i.is_Mul and any(j.is_Add for j in i.args) for i in Add.make_args(expr)):
            expr = expand_mul(expr)

    # 根据选项中的 gens 进行不同的处理分支
    if opt.gens:
        rep, gens = _dict_from_expr_if_gens(expr, opt)
    else:
        rep, gens = _dict_from_expr_no_gens(expr, opt)

    # 返回转换后的字典表示和更新后的选项
    return rep, opt.clone({'gens': gens})


def expr_from_dict(rep, *gens):
    """将多项式形式的字典转换为表达式。"""
    result = []

    for monom, coeff in rep.items():
        term = [coeff]
        for g, m in zip(gens, monom):
            if m:
                term.append(Pow(g, m))

        result.append(Mul(*term))

    return Add(*result)

# 以下是简化的别名定义，直接指向相应的函数
parallel_dict_from_basic = parallel_dict_from_expr
dict_from_basic = dict_from_expr
basic_from_dict = expr_from_dict


def _dict_reorder(rep, gens, new_gens):
    """使用字典表示重新排序级别。"""
    gens = list(gens)

    monoms = rep.keys()
    coeffs = rep.values()

    new_monoms = [ [] for _ in range(len(rep)) ]
    used_indices = set()

    # 根据新的生成器顺序重新排列
    for gen in new_gens:
        try:
            j = gens.index(gen)
            used_indices.add(j)

            for M, new_M in zip(monoms, new_monoms):
                new_M.append(M[j])
        except ValueError:
            for new_M in new_monoms:
                new_M.append(0)

    # 检查是否所有未使用的索引对应的单项式都为零
    for i, _ in enumerate(gens):
        if i not in used_indices:
            for monom in monoms:
                if monom[i]:
                    raise GeneratorsError("unable to drop generators")

    return map(tuple, new_monoms), coeffs


class PicklableWithSlots:
    """
    可以使用 ``__slots__`` 进行 pickle 的混合类。

    示例
    ========

    首先定义一个混合了 :class:`PicklableWithSlots` 的类::

        >>> from sympy.polys.polyutils import PicklableWithSlots
        >>> class Some(PicklableWithSlots):
        ...     __slots__ = ('foo', 'bar')
        ...
        ...     def __init__(self, foo, bar):
        ...         self.foo = foo
        ...         self.bar = bar


    """
    # To make :mod:`pickle` happy in doctest we have to use these hacks::
    # 在 doctest 中，为了使 :mod:`pickle` 模块正常工作，我们需要使用以下的技巧：

        >>> import builtins
        >>> builtins.Some = Some
        >>> from sympy.polys import polyutils
        >>> polyutils.Some = Some

    # Next lets see if we can create an instance, pickle it and unpickle::
    # 接下来，让我们看看是否可以创建一个实例，对其进行序列化和反序列化：

        >>> some = Some('abc', 10)
        >>> some.foo, some.bar
        ('abc', 10)

        >>> from pickle import dumps, loads
        >>> some2 = loads(dumps(some))

        >>> some2.foo, some2.bar
        ('abc', 10)

    """

    # 禁止动态添加实例属性，只允许存在指定的属性
    __slots__ = ()

    # 自定义对象序列化方法，以便将对象转换为可序列化的字典
    def __getstate__(self, cls=None):
        if cls is None:
            # This is the case for the instance that gets pickled
            # 对于被序列化的实例，cls 参数为 None
            cls = self.__class__

        d = {}

        # Get all data that should be stored from super classes
        # 获取所有应该从父类中存储的数据
        for c in cls.__bases__:
            # XXX: Python 3.11 defines object.__getstate__ and it does not
            # accept any arguments so we need to make sure not to call it with
            # an argument here. To be compatible with Python < 3.11 we need to
            # be careful not to assume that c or object has a __getstate__
            # method though.
            # Python 3.11 定义了 object.__getstate__ 方法，并且它不接受任何参数，
            # 因此我们在这里调用时要确保不传递参数。为了兼容 Python < 3.11，
            # 我们需要小心不要假设 c 或 object 拥有 __getstate__ 方法。
            getstate = getattr(c, "__getstate__", None)
            objstate = getattr(object, "__getstate__", None)
            if getstate is not None and getstate is not objstate:
                d.update(getstate(self, c))

        # Get all information that should be stored from cls and return the dict
        # 获取应该从 cls 中存储的所有信息，并返回字典
        for name in cls.__slots__:
            if hasattr(self, name):
                d[name] = getattr(self, name)

        return d

    # 自定义对象反序列化方法，将序列化的字典值赋给新创建的实例
    def __setstate__(self, d):
        # All values that were pickled are now assigned to a fresh instance
        # 将所有序列化的值赋给新创建的实例
        for name, value in d.items():
            setattr(self, name, value)
# 定义一个混合类，用于具有 `__mul__` 方法的类，支持自然的整数幂运算，采用二进制展开实现效率优化。

class IntegerPowerable:
    r"""
    Mixin class for classes that define a `__mul__` method, and want to be
    raised to integer powers in the natural way that follows. Implements
    powering via binary expansion, for efficiency.

    By default, only integer powers $\geq 2$ are supported. To support the
    first, zeroth, or negative powers, override the corresponding methods,
    `_first_power`, `_zeroth_power`, `_negative_power`, below.
    """

    def __pow__(self, e, modulo=None):
        # 如果指数 e 小于 2
        if e < 2:
            try:
                # 如果 e 等于 1，调用 `_first_power()` 方法
                if e == 1:
                    return self._first_power()
                # 如果 e 等于 0，调用 `_zeroth_power()` 方法
                elif e == 0:
                    return self._zeroth_power()
                # 否则，调用 `_negative_power(e, modulo=modulo)` 方法
                else:
                    return self._negative_power(e, modulo=modulo)
            except NotImplementedError:
                # 如果方法未实现，返回 NotImplemented
                return NotImplemented
        else:
            # 将指数 e 转换为二进制的位列表
            bits = [int(d) for d in reversed(bin(e)[2:])]
            n = len(bits)
            p = self
            first = True
            for i in range(n):
                # 如果当前位为 1
                if bits[i]:
                    # 如果是第一次计算
                    if first:
                        r = p
                        first = False
                    else:
                        r *= p
                        # 如果有模数，则对结果取模
                        if modulo is not None:
                            r %= modulo
                # 如果不是最后一位，将 p 自乘
                if i < n - 1:
                    p *= p
                    # 如果有模数，则对 p 取模
                    if modulo is not None:
                        p %= modulo
            # 返回最终结果 r
            return r

    def _negative_power(self, e, modulo=None):
        """
        Compute inverse of self, then raise that to the abs(e) power.
        For example, if the class has an `inv()` method,
            return self.inv() ** abs(e) % modulo
        """
        # 计算 self 的逆，然后将其提升为 abs(e) 次幂
        raise NotImplementedError

    def _zeroth_power(self):
        """Return unity element of algebraic struct to which self belongs."""
        # 返回 self 所属代数结构的单位元素
        raise NotImplementedError

    def _first_power(self):
        """Return a copy of self."""
        # 返回 self 的副本
        raise NotImplementedError


# 如果 GROUND_TYPES 等于 'flint'
if GROUND_TYPES == 'flint':
    # 导入 flint 模块
    import flint
    # 设置 _GF_types 为 flint 模块中的 nmod 和 fmpz_mod 类型元组
    _GF_types = (flint.nmod, flint.fmpz_mod)
# 否则
else:
    # 从 sympy.polys.domains.modularinteger 导入 ModularInteger
    from sympy.polys.domains.modularinteger import ModularInteger
    flint = None
    # 设置 _GF_types 为 ModularInteger 类型元组
    _GF_types = (ModularInteger,)
```