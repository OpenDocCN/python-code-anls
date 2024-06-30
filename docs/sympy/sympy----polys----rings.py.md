# `D:\src\scipysrc\sympy\sympy\polys\rings.py`

```
"""Sparse polynomial rings. """

from __future__ import annotations  # 允许类型提示中使用类型自身
from typing import Any  # 引入类型提示模块中的 Any 类型

from operator import add, mul, lt, le, gt, ge  # 引入运算符模块中的多个操作符
from functools import reduce  # 引入 functools 模块中的 reduce 函数
from types import GeneratorType  # 引入 types 模块中的 GeneratorType 类型

from sympy.core.expr import Expr  # 从 sympy 核心表达式模块中导入 Expr 类
from sympy.core.intfunc import igcd  # 从 sympy 核心整数函数模块中导入 igcd 函数
from sympy.core.symbol import Symbol, symbols as _symbols  # 从 sympy 核心符号模块中导入 Symbol 类和 symbols 函数
from sympy.core.sympify import CantSympify, sympify  # 从 sympy 核心 sympify 模块中导入 CantSympify 类和 sympify 函数
from sympy.ntheory.multinomial import multinomial_coefficients  # 从 sympy 多项式数论模块中导入 multinomial_coefficients 函数
from sympy.polys.compatibility import IPolys  # 从 sympy 多项式兼容性模块中导入 IPolys 类
from sympy.polys.constructor import construct_domain  # 从 sympy 多项式构造器模块中导入 construct_domain 函数
from sympy.polys.densebasic import ninf, dmp_to_dict, dmp_from_dict  # 从 sympy 密集基础模块中导入 ninf、dmp_to_dict 和 dmp_from_dict 函数
from sympy.polys.domains.domainelement import DomainElement  # 从 sympy 多项式域元素模块中导入 DomainElement 类
from sympy.polys.domains.polynomialring import PolynomialRing  # 从 sympy 多项式域模块中导入 PolynomialRing 类
from sympy.polys.heuristicgcd import heugcd  # 从 sympy 启发式最大公因数模块中导入 heugcd 函数
from sympy.polys.monomials import MonomialOps  # 从 sympy 单项式模块中导入 MonomialOps 类
from sympy.polys.orderings import lex  # 从 sympy 多项式排序模块中导入 lex 函数
from sympy.polys.polyerrors import (  # 从 sympy 多项式错误模块中导入多个错误类
    CoercionFailed, GeneratorsError,
    ExactQuotientFailed, MultivariatePolynomialError)
from sympy.polys.polyoptions import (Domain as DomainOpt,  # 从 sympy 多项式选项模块中导入 Domain 和 Order 类
                                     Order as OrderOpt, build_options)
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,  # 从 sympy 多项式工具模块中导入多个函数
                                   _parallel_dict_from_expr)
from sympy.printing.defaults import DefaultPrinting  # 从 sympy 打印默认模块中导入 DefaultPrinting 类
from sympy.utilities import public, subsets  # 从 sympy 实用工具模块中导入 public 和 subsets 函数
from sympy.utilities.iterables import is_sequence  # 从 sympy 可迭代工具模块中导入 is_sequence 函数
from sympy.utilities.magic import pollute  # 从 sympy 魔术模块中导入 pollute 函数

@public
def ring(symbols, domain, order=lex):
    """Construct a polynomial ring returning ``(ring, x_1, ..., x_n)``.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
        符号/表达式或者符号/表达式序列（非空）
    domain : :class:`~.Domain` or coercible
        :class:`~.Domain` 类或者可转换类型
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``
        :class:`~.MonomialOrder` 类或者可转换类型，默认为 ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> R, x, y, z = ring("x,y,z", ZZ, lex)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)  # 创建多项式环对象 _ring
    return (_ring,) + _ring.gens  # 返回多项式环对象 _ring 和其生成元组成的元组

@public
def xring(symbols, domain, order=lex):
    """Construct a polynomial ring returning ``(ring, (x_1, ..., x_n))``.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
        符号/表达式或者符号/表达式序列（非空）
    domain : :class:`~.Domain` or coercible
        :class:`~.Domain` 类或者可转换类型
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``
        :class:`~.MonomialOrder` 类或者可转换类型，默认为 ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import xring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> R, (x, y, z) = xring("x,y,z", ZZ, lex)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    # 创建一个多项式环对象 `_ring`，使用给定的符号、域和排序方式
    _ring = PolyRing(symbols, domain, order)
    # 返回一个元组，包含 `_ring` 对象和其生成元组 `_ring.gens`
    return (_ring, _ring.gens)
# 声明一个公共函数 `vring`，用于构建多项式环并将变量 `x_1, ..., x_n` 注入全局命名空间
@public
def vring(symbols, domain, order=lex):
    """Construct a polynomial ring and inject ``x_1, ..., x_n`` into the global namespace.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
        符号或表达式的字符串或字符串/表达式序列（非空）
    domain : :class:`~.Domain` or coercible
        :class:`~.Domain` 类型或可强制转换类型
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``
        :class:`~.MonomialOrder` 类型或可强制转换类型，可选，默认为 ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import vring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> vring("x,y,z", ZZ, lex)
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z # noqa:
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    # 创建多项式环对象 `_ring`，使用给定的符号、域和排序方式
    _ring = PolyRing(symbols, domain, order)
    # 将 `_ring` 中的符号名污染到全局命名空间中
    pollute([ sym.name for sym in _ring.symbols ], _ring.gens)
    # 返回构建的多项式环对象 `_ring`
    return _ring

# 声明一个公共函数 `sring`，用于从选项和输入表达式中构建一个环
@public
def sring(exprs, *symbols, **options):
    """Construct a ring deriving generators and domain from options and input expressions.

    Parameters
    ==========

    exprs : :class:`~.Expr` or sequence of :class:`~.Expr` (sympifiable)
        表达式或可 sympify 的表达式序列
    symbols : sequence of :class:`~.Symbol`/:class:`~.Expr`
        :class:`~.Symbol` 或 :class:`~.Expr` 的序列
    options : keyword arguments understood by :class:`~.Options`
        :class:`~.Options` 类型理解的关键字参数

    Examples
    ========

    >>> from sympy import sring, symbols

    >>> x, y, z = symbols("x,y,z")
    >>> R, f = sring(x + 2*y + 3*z)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> f
    x + 2*y + 3*z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    # 如果 `exprs` 不是序列，则将其转换为包含单个元素的列表
    single = False
    if not is_sequence(exprs):
        exprs, single = [exprs], True
    
    # 将所有表达式 sympify 化
    exprs = list(map(sympify, exprs))
    # 构建选项对象 `opt`
    opt = build_options(symbols, options)

    # 并行地从表达式和选项构建替换字典 `reps` 和更新选项 `opt`
    # TODO: 重写此部分，避免使用 expand()（参见 poly()）
    reps, opt = _parallel_dict_from_expr(exprs, opt)

    # 如果选项中的域为 None，则构造系数域并更新选项 `opt`
    if opt.domain is None:
        coeffs = sum([ list(rep.values()) for rep in reps ], [])
        opt.domain, coeffs_dom = construct_domain(coeffs, opt=opt)
        coeff_map = dict(zip(coeffs, coeffs_dom))
        reps = [{m: coeff_map[c] for m, c in rep.items()} for rep in reps]

    # 创建多项式环对象 `_ring`
    _ring = PolyRing(opt.gens, opt.domain, opt.order)
    # 将 `reps` 中的每个字典转换为多项式，并存储在 `polys` 列表中
    polys = list(map(_ring.from_dict, reps))

    # 如果 `single` 为 True，则返回 `_ring` 和第一个多项式；否则返回 `_ring` 和多项式列表
    if single:
        return (_ring, polys[0])
    else:
        return (_ring, polys)

# 声明一个函数 `_parse_symbols`，用于解析符号参数并返回相应的符号序列
def _parse_symbols(symbols):
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True) if symbols else ()
    elif isinstance(symbols, Expr):
        return (symbols,)
    elif is_sequence(symbols):
        if all(isinstance(s, str) for s in symbols):
            return _symbols(symbols)
        elif all(isinstance(s, Expr) for s in symbols):
            return symbols
    
    # 如果符号参数类型不符合预期，则抛出异常
    raise GeneratorsError("expected a string, Symbol or expression or a non-empty sequence of strings, Symbols or expressions")

# 声明一个字典 `_ring_cache`，用于缓存多项式环对象
_ring_cache: dict[Any, Any] = {}

# 定义类 `PolyRing`，表示多变量分布式多项式环
class PolyRing(DefaultPrinting, IPolys):
    """Multivariate distributed polynomial ring. """
    def __new__(cls, symbols, domain, order=lex):
        # 将符号转换为元组形式
        symbols = tuple(_parse_symbols(symbols))
        # 计算符号的数量
        ngens = len(symbols)
        # 预处理域
        domain = DomainOpt.preprocess(domain)
        # 预处理顺序
        order = OrderOpt.preprocess(order)

        # 构建用于哈希的元组
        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)
        # 尝试从缓存中获取已有的对象实例
        obj = _ring_cache.get(_hash_tuple)

        # 如果缓存中不存在该对象实例
        if obj is None:
            # 如果域是复合的并且与给定符号有交集，则抛出异常
            if domain.is_Composite and set(symbols) & set(domain.symbols):
                raise GeneratorsError("polynomial ring and it's ground domain share generators")

            # 创建一个新的对象实例
            obj = object.__new__(cls)
            # 设置对象的哈希值元组
            obj._hash_tuple = _hash_tuple
            # 计算对象的哈希值
            obj._hash = hash(_hash_tuple)
            # 定义元素类型
            obj.dtype = type("PolyElement", (PolyElement,), {"ring": obj})
            # 设置符号
            obj.symbols = symbols
            # 设置符号数量
            obj.ngens = ngens
            # 设置域
            obj.domain = domain
            # 设置顺序
            obj.order = order

            # 设置零单项式
            obj.zero_monom = (0,)*ngens
            # 生成多项式的生成器
            obj.gens = obj._gens()
            # 设置生成器的集合
            obj._gens_set = set(obj.gens)

            # 设置单位元
            obj._one = [(obj.zero_monom, domain.one)]

            # 如果有符号
            if ngens:
                # 创建单项式操作的代码生成器
                codegen = MonomialOps(ngens)
                # 设置单项式乘法
                obj.monomial_mul = codegen.mul()
                # 设置单项式幂
                obj.monomial_pow = codegen.pow()
                # 设置单项式乘幂
                obj.monomial_mulpow = codegen.mulpow()
                # 设置单项式左除
                obj.monomial_ldiv = codegen.ldiv()
                # 设置单项式除法
                obj.monomial_div = codegen.div()
                # 设置单项式最小公倍数
                obj.monomial_lcm = codegen.lcm()
                # 设置单项式最大公约数
                obj.monomial_gcd = codegen.gcd()
            else:
                # 如果没有符号，则定义一个空的单项式操作函数
                monunit = lambda a, b: ()
                obj.monomial_mul = monunit
                obj.monomial_pow = monunit
                obj.monomial_mulpow = lambda a, b, c: ()
                obj.monomial_ldiv = monunit
                obj.monomial_div = monunit
                obj.monomial_lcm = monunit
                obj.monomial_gcd = monunit

            # 根据顺序设置主导指数向量函数
            if order is lex:
                obj.leading_expv = max
            else:
                obj.leading_expv = lambda f: max(f, key=order)

            # 针对每个符号和生成器，将符号名作为对象的属性名，生成器作为属性值设置到对象中
            for symbol, generator in zip(obj.symbols, obj.gens):
                if isinstance(symbol, Symbol):
                    name = symbol.name

                    if not hasattr(obj, name):
                        setattr(obj, name, generator)

            # 将对象实例存入缓存
            _ring_cache[_hash_tuple] = obj

        # 返回对象实例
        return obj

    def _gens(self):
        """Return a list of polynomial generators. """
        # 获取域的单位元
        one = self.domain.one
        # 初始化生成器列表
        _gens = []
        # 对于每个生成器的索引
        for i in range(self.ngens):
            # 计算单项式的基础
            expv = self.monomial_basis(i)
            # 创建零多项式
            poly = self.zero
            # 设置指定指数的系数为单位元
            poly[expv] = one
            # 将生成的多项式添加到生成器列表中
            _gens.append(poly)
        # 返回生成器列表的元组形式
        return tuple(_gens)

    def __getnewargs__(self):
        # 返回用于对象序列化的参数元组
        return (self.symbols, self.domain, self.order)

    def __getstate__(self):
        # 复制对象的状态字典
        state = self.__dict__.copy()
        # 删除状态字典中的主导指数向量函数
        del state["leading_expv"]

        # 对于状态字典中以"monomial_"开头的键，删除这些键
        for key in state:
            if key.startswith("monomial_"):
                del state[key]

        # 返回更新后的状态字典
        return state
    # 定义对象的哈希方法，返回对象的哈希值
    def __hash__(self):
        return self._hash

    # 定义对象的相等性方法，比较两个对象是否相等
    def __eq__(self, other):
        return isinstance(other, PolyRing) and \
            (self.symbols, self.domain, self.ngens, self.order) == \
            (other.symbols, other.domain, other.ngens, other.order)

    # 定义对象的不相等性方法，与相等性方法相反
    def __ne__(self, other):
        return not self == other

    # 克隆当前对象，返回一个新的对象实例
    def clone(self, symbols=None, domain=None, order=None):
        return self.__class__(symbols or self.symbols, domain or self.domain, order or self.order)

    # 返回多项式环的第 i 个基础元素
    def monomial_basis(self, i):
        """Return the ith-basis element. """
        basis = [0]*self.ngens
        basis[i] = 1
        return tuple(basis)

    # 属性方法，返回零元素
    @property
    def zero(self):
        return self.dtype()

    # 属性方法，返回单位元素
    @property
    def one(self):
        return self.dtype(self._one)

    # 将给定元素转换为当前域中的元素
    def domain_new(self, element, orig_domain=None):
        return self.domain.convert(element, orig_domain)

    # 创建一个新的多项式环中的地面元素，使用给定系数
    def ground_new(self, coeff):
        return self.term_new(self.zero_monom, coeff)

    # 创建一个新的多项式，使用给定的单项式和系数
    def term_new(self, monom, coeff):
        coeff = self.domain_new(coeff)
        poly = self.zero
        if coeff:
            poly[monom] = coeff
        return poly

    # 将给定的元素转换为当前多项式环中的元素
    def ring_new(self, element):
        if isinstance(element, PolyElement):
            if self == element.ring:
                return element
            elif isinstance(self.domain, PolynomialRing) and self.domain.ring == element.ring:
                return self.ground_new(element)
            else:
                raise NotImplementedError("conversion")
        elif isinstance(element, str):
            raise NotImplementedError("parsing")
        elif isinstance(element, dict):
            return self.from_dict(element)
        elif isinstance(element, list):
            try:
                return self.from_terms(element)
            except ValueError:
                return self.from_list(element)
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)

    # 将当前对象作为函数调用时，调用 ring_new 方法
    __call__ = ring_new

    # 从字典形式的元素创建多项式
    def from_dict(self, element, orig_domain=None):
        domain_new = self.domain_new
        poly = self.zero

        for monom, coeff in element.items():
            coeff = domain_new(coeff, orig_domain)
            if coeff:
                poly[monom] = coeff

        return poly

    # 从项的列表形式创建多项式
    def from_terms(self, element, orig_domain=None):
        return self.from_dict(dict(element), orig_domain)

    # 从列表形式的元素创建多项式，使用 dmp_to_dict 转换
    def from_list(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain))
    def _rebuild_expr(self, expr, mapping):
        # 保存当前对象的域（domain）
        domain = self.domain

        # 定义内部函数_rebuild，用于重建表达式
        def _rebuild(expr):
            # 尝试从映射中获取表达式的生成器
            generator = mapping.get(expr)

            # 如果找到生成器，则直接返回
            if generator is not None:
                return generator
            # 如果表达式是加法表达式，则递归处理每个参数并返回求和结果
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            # 如果表达式是乘法表达式，则递归处理每个参数并返回乘积结果
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            else:
                # 如果表达式不是加法或乘法，则尝试将其分解为底数和指数
                base, exp = expr.as_base_exp()
                # 如果指数是整数且大于1，则递归处理底数并返回其指数次幂结果
                if exp.is_Integer and exp > 1:
                    return _rebuild(base)**int(exp)
                else:
                    # 否则，使用domain.convert将表达式转换为一个新的基础表达式
                    return self.ground_new(domain.convert(expr))

        # 调用_rebuild函数并返回结果
        return _rebuild(sympify(expr))

    def from_expr(self, expr):
        # 将symbols和gens两个列表打包成字典映射
        mapping = dict(list(zip(self.symbols, self.gens)))

        try:
            # 调用_rebuild_expr方法重建表达式为多项式
            poly = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            # 如果转换失败，抛出值错误异常
            raise ValueError("expected an expression convertible to a polynomial in %s, got %s" % (self, expr))
        else:
            # 否则，返回新生成的多项式环
            return self.ring_new(poly)

    def index(self, gen):
        """Compute index of ``gen`` in ``self.gens``. """
        # 如果gen为None，则根据ngens的值确定索引i
        if gen is None:
            if self.ngens:
                i = 0
            else:
                i = -1  # 表示不可能的选择
        # 如果gen是整数，则直接将其作为索引i
        elif isinstance(gen, int):
            i = gen
            # 检查i是否在合法范围内，否则抛出值错误异常
            if 0 <= i and i < self.ngens:
                pass
            elif -self.ngens <= i and i <= -1:
                i = -i - 1
            else:
                raise ValueError("invalid generator index: %s" % gen)
        # 如果gen是self.dtype类型，则尝试在gens中查找其索引
        elif isinstance(gen, self.dtype):
            try:
                i = self.gens.index(gen)
            except ValueError:
                raise ValueError("invalid generator: %s" % gen)
        # 如果gen是字符串类型，则尝试在symbols中查找其索引
        elif isinstance(gen, str):
            try:
                i = self.symbols.index(gen)
            except ValueError:
                raise ValueError("invalid generator: %s" % gen)
        else:
            # 如果gen不是预期的类型，则抛出值错误异常
            raise ValueError("expected a polynomial generator, an integer, a string or None, got %s" % gen)

        return i

    def drop(self, *gens):
        """Remove specified generators from this ring. """
        # 将gens中每个元素映射为其在symbols中的索引集合
        indices = set(map(self.index, gens))
        # 根据索引集合筛选出剩余的symbols
        symbols = [s for i, s in enumerate(self.symbols) if i not in indices]

        # 如果剩余的symbols为空，则返回domain
        if not symbols:
            return self.domain
        else:
            # 否则，克隆当前对象，替换symbols后返回
            return self.clone(symbols=symbols)

    def __getitem__(self, key):
        # 获取指定索引key处的symbols
        symbols = self.symbols[key]

        # 如果symbols为空，则返回domain
        if not symbols:
            return self.domain
        else:
            # 否则，克隆当前对象，替换symbols后返回
            return self.clone(symbols=symbols)
    def to_ground(self):
        # 如果当前环域是复合域或者具有子域属性，则返回一个新的对象，使用其子域作为域
        if self.domain.is_Composite or hasattr(self.domain, 'domain'):
            return self.clone(domain=self.domain.domain)
        else:
            # 抛出异常，表示当前域不是复合域
            raise ValueError("%s is not a composite domain" % self.domain)

    def to_domain(self):
        # 将当前环转换为一个多项式环对象，并返回
        return PolynomialRing(self)

    def to_field(self):
        # 导入分式域类 FracField
        from sympy.polys.fields import FracField
        # 返回一个新的分式域对象，使用当前环的符号、域和排序方式
        return FracField(self.symbols, self.domain, self.order)

    @property
    def is_univariate(self):
        # 检查当前环是否是单变量环，返回布尔值
        return len(self.gens) == 1

    @property
    def is_multivariate(self):
        # 检查当前环是否是多变量环，返回布尔值
        return len(self.gens) > 1

    def add(self, *objs):
        """
        添加一系列多项式或多项式容器。

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> R, x = ring("x", ZZ)
        >>> R.add([ x**2 + 2*i + 3 for i in range(4) ])
        4*x**2 + 24
        >>> _.factor_list()
        (4, [(x**2 + 6, 1)])

        """
        # 初始化结果多项式为零多项式
        p = self.zero

        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                # 如果 obj 是多项式序列，则递归调用 add 方法进行相加
                p += self.add(*obj)
            else:
                # 直接将 obj 加到 p 上
                p += obj

        return p

    def mul(self, *objs):
        """
        乘以一系列多项式或多项式容器。

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> R, x = ring("x", ZZ)
        >>> R.mul([ x**2 + 2*i + 3 for i in range(4) ])
        x**8 + 24*x**6 + 206*x**4 + 744*x**2 + 945
        >>> _.factor_list()
        (1, [(x**2 + 3, 1), (x**2 + 5, 1), (x**2 + 7, 1), (x**2 + 9, 1)])

        """
        # 初始化结果多项式为单位多项式
        p = self.one

        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                # 如果 obj 是多项式序列，则递归调用 mul 方法进行相乘
                p *= self.mul(*obj)
            else:
                # 直接将 obj 乘到 p 上
                p *= obj

        return p

    def drop_to_ground(self, *gens):
        r"""
        从环中移除指定的生成器，并将它们注入到域中。
        """
        # 获取要移除生成器的索引集合
        indices = set(map(self.index, gens))
        # 创建新的符号列表，不包含要移除的生成器对应的符号
        symbols = [s for i, s in enumerate(self.symbols) if i not in indices]
        # 创建新的生成器列表，不包含要移除的生成器
        gens = [gen for i, gen in enumerate(self.gens) if i not in indices]

        if not symbols:
            # 如果符号列表为空，则返回当前环对象
            return self
        else:
            # 否则，返回一个克隆对象，更新符号和域
            return self.clone(symbols=symbols, domain=self.drop(*gens))

    def compose(self, other):
        """将 ``other`` 的生成器添加到 ``self`` 中"""
        if self != other:
            # 如果当前环和给定的环不相等，合并它们的符号集合并返回一个新的环对象
            syms = set(self.symbols).union(set(other.symbols))
            return self.clone(symbols=list(syms))
        else:
            # 如果当前环和给定的环相等，则直接返回当前环对象
            return self

    def add_gens(self, symbols):
        """将 ``symbols`` 中的元素作为生成器添加到 ``self`` 中"""
        # 合并当前环的符号集合和给定符号集合，生成一个新的符号列表
        syms = set(self.symbols).union(set(symbols))
        # 返回一个克隆对象，更新符号列表
        return self.clone(symbols=list(syms))
    def symmetric_poly(self, n):
        """
        Return the elementary symmetric polynomial of degree *n* over
        this ring's generators.
        """
        # 如果 n 小于 0 或者大于 self.ngens，抛出数值错误异常
        if n < 0 or n > self.ngens:
            raise ValueError("Cannot generate symmetric polynomial of order %s for %s" % (n, self.gens))
        # 如果 n 等于 0，返回单位元素 self.one
        elif not n:
            return self.one
        else:
            # 初始化多项式为零
            poly = self.zero
            # 遍历生成器集合的所有大小为 n 的子集
            for s in subsets(range(self.ngens), int(n)):
                # 根据子集 s 创建对应的单项式的元组 monom
                monom = tuple(int(i in s) for i in range(self.ngens))
                # 使用 self.term_new 方法创建单项式，并加到多项式 poly 中
                poly += self.term_new(monom, self.domain.one)
            # 返回计算得到的多项式 poly
            return poly
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):
    """Element of multivariate distributed polynomial ring. """

    def new(self, init):
        # 创建并返回一个新的多项式元素，使用与当前对象相同的类
        return self.__class__(init)

    def parent(self):
        # 返回当前多项式元素所属的环的域
        return self.ring.to_domain()

    def __getnewargs__(self):
        # 返回一个元组，其中包含当前环和多项式元素的项列表，用于序列化
        return (self.ring, list(self.iterterms()))

    _hash = None

    def __hash__(self):
        # 计算当前多项式元素的哈希值
        # 注意：目前尚未保护字典免受修改，因此在任何使用中，修改都会导致哈希值错误。
        # 在不降低此低级类速度的情况下，小心使用此特性，直到找到如何创建安全 API 的方法。
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.ring, frozenset(self.items())))
        return _hash

    def copy(self):
        """Return a copy of polynomial self.

        Polynomials are mutable; if one is interested in preserving
        a polynomial, and one plans to use inplace operations, one
        can copy the polynomial. This method makes a shallow copy.
        
        返回当前多项式的副本。
        
        多项式是可变的；如果想要保留一个多项式，并且计划使用原地操作，可以复制多项式。此方法进行浅复制。

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> R, x, y = ring('x, y', ZZ)
        >>> p = (x + y)**2
        >>> p1 = p.copy()
        >>> p2 = p
        >>> p[R.zero_monom] = 3
        >>> p
        x**2 + 2*x*y + y**2 + 3
        >>> p1
        x**2 + 2*x*y + y**2
        >>> p2
        x**2 + 2*x*y + y**2 + 3

        """

        return self.new(self)

    def set_ring(self, new_ring):
        # 设置当前多项式元素的环
        if self.ring == new_ring:
            return self
        elif self.ring.symbols != new_ring.symbols:
            terms = list(zip(*_dict_reorder(self, self.ring.symbols, new_ring.symbols)))
            return new_ring.from_terms(terms, self.ring.domain)
        else:
            return new_ring.from_dict(self, self.ring.domain)

    def as_expr(self, *symbols):
        # 将多项式元素转换为表达式
        if not symbols:
            symbols = self.ring.symbols
        elif len(symbols) != self.ring.ngens:
            raise ValueError(
                "Wrong number of symbols, expected %s got %s" %
                (self.ring.ngens, len(symbols))
            )

        return expr_from_dict(self.as_expr_dict(), *symbols)

    def as_expr_dict(self):
        # 将多项式元素表示为表达式字典
        to_sympy = self.ring.domain.to_sympy
        return {monom: to_sympy(coeff) for monom, coeff in self.iterterms()}

    def clear_denoms(self):
        # 清除多项式元素中的分母
        domain = self.ring.domain

        if not domain.is_Field or not domain.has_assoc_Ring:
            return domain.one, self

        ground_ring = domain.get_ring()
        common = ground_ring.one
        lcm = ground_ring.lcm
        denom = domain.denom

        for coeff in self.values():
            common = lcm(common, denom(coeff))

        poly = self.new([ (k, v*common) for k, v in self.items() ])
        return common, poly
    def strip_zero(self):
        """Eliminate monomials with zero coefficient. """
        # 遍历多项式字典的键值对列表
        for k, v in list(self.items()):
            # 如果系数为零，则删除该键
            if not v:
                del self[k]

    def __eq__(p1, p2):
        """Equality test for polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = (x + y)**2 + (x - y)**2
        >>> p1 == 4*x*y
        False
        >>> p1 == 2*(x**2 + y**2)
        True

        """
        # 如果 p2 是空的，返回 p1 是否为空的结果
        if not p2:
            return not p1
        # 如果 p2 是多项式元素并且与 p1 的环相同，则比较它们的字典表示
        elif isinstance(p2, PolyElement) and p2.ring == p1.ring:
            return dict.__eq__(p1, p2)
        # 如果 p1 的长度大于 1，则返回 False
        elif len(p1) > 1:
            return False
        # 否则，比较 p1 的零单项式系数与 p2
        else:
            return p1.get(p1.ring.zero_monom) == p2

    def __ne__(p1, p2):
        return not p1 == p2

    def almosteq(p1, p2, tolerance=None):
        """Approximate equality test for polynomials. """
        # 获取 p1 的环
        ring = p1.ring

        # 如果 p2 是与环类型相同的实例
        if isinstance(p2, ring.dtype):
            # 如果两个多项式的键集合不相等，则返回 False
            if set(p1.keys()) != set(p2.keys()):
                return False

            # 获取环的近似相等性判断函数
            almosteq = ring.domain.almosteq

            # 遍历 p1 的键，并比较其值的近似相等性
            for k in p1.keys():
                if not almosteq(p1[k], p2[k], tolerance):
                    return False
            return True
        # 如果 p1 的长度大于 1，则返回 False
        elif len(p1) > 1:
            return False
        # 否则，尝试将 p2 转换为环的类型，并比较其与 p1 的近似相等性
        else:
            try:
                p2 = ring.domain.convert(p2)
            except CoercionFailed:
                return False
            else:
                return ring.domain.almosteq(p1.const(), p2, tolerance)

    def sort_key(self):
        return (len(self), self.terms())

    def _cmp(p1, p2, op):
        # 如果 p2 是与 p1 的环类型相同的实例，则使用指定的操作符比较它们的排序键
        if isinstance(p2, p1.ring.dtype):
            return op(p1.sort_key(), p2.sort_key())
        else:
            return NotImplemented

    def __lt__(p1, p2):
        return p1._cmp(p2, lt)
    def __le__(p1, p2):
        return p1._cmp(p2, le)
    def __gt__(p1, p2):
        return p1._cmp(p2, gt)
    def __ge__(p1, p2):
        return p1._cmp(p2, ge)

    def _drop(self, gen):
        # 获取多项式的环
        ring = self.ring
        # 获取生成器 gen 在环中的索引
        i = ring.index(gen)

        # 如果环只有一个生成器，则返回生成器索引和环的定义域
        if ring.ngens == 1:
            return i, ring.domain
        # 否则，返回生成器索引和去除生成器后的新环
        else:
            symbols = list(ring.symbols)
            del symbols[i]
            return i, ring.clone(symbols=symbols)

    def drop(self, gen):
        # 获取生成器索引和相应的环
        i, ring = self._drop(gen)

        # 如果多项式的环只有一个生成器
        if self.ring.ngens == 1:
            # 如果多项式是常数项，则返回该常数项
            if self.is_ground:
                return self.coeff(1)
            # 否则，引发异常，因为无法移除生成器
            else:
                raise ValueError("Cannot drop %s" % gen)
        # 否则，对于每个多项式的键值对
        else:
            # 初始化一个新的多项式
            poly = ring.zero

            # 遍历当前多项式的键值对
            for k, v in self.items():
                # 如果第 i 个生成器的次数为 0，则移除该生成器并更新多项式
                if k[i] == 0:
                    K = list(k)
                    del K[i]
                    poly[tuple(K)] = v
                # 否则，引发异常，因为无法移除生成器
                else:
                    raise ValueError("Cannot drop %s" % gen)

            # 返回更新后的多项式
            return poly
    def _drop_to_ground(self, gen):
        # 获取环对象的引用
        ring = self.ring
        # 获取生成元 gen 在环中的索引
        i = ring.index(gen)

        # 复制环对象的符号列表，并移除索引 i 处的符号
        symbols = list(ring.symbols)
        del symbols[i]
        # 返回索引 i 和更新后的环对象
        return i, ring.clone(symbols=symbols, domain=ring[i])

    def drop_to_ground(self, gen):
        # 如果环中只有一个生成元，则无法将其移至基础
        if self.ring.ngens == 1:
            raise ValueError("Cannot drop only generator to ground")

        # 调用 _drop_to_ground 方法，获取索引 i 和更新后的环对象 ring
        i, ring = self._drop_to_ground(gen)
        # 创建一个零多项式
        poly = ring.zero
        # 获取环中的唯一生成元
        gen = ring.domain.gens[0]

        # 遍历当前对象中的每一项
        for monom, coeff in self.iterterms():
            # 从单项式 monom 中删除索引 i 对应的指数
            mon = monom[:i] + monom[i+1:]
            # 如果 poly 中不存在单项式 mon，则添加 (gen^monom[i]) * coeff
            if mon not in poly:
                poly[mon] = (gen**monom[i]).mul_ground(coeff)
            else:
                # 如果 poly 中已经存在单项式 mon，则累加 (gen^monom[i]) * coeff
                poly[mon] += (gen**monom[i]).mul_ground(coeff)

        # 返回构建的多项式 poly
        return poly

    def to_dense(self):
        # 将当前对象转换为密集多项式表示
        return dmp_from_dict(self, self.ring.ngens-1, self.ring.domain)

    def to_dict(self):
        # 将当前对象转换为字典表示
        return dict(self)

    def str(self, printer, precedence, exp_pattern, mul_symbol):
        # 如果当前对象为空，则返回环域中的零元素的打印结果
        if not self:
            return printer._print(self.ring.domain.zero)

        # 获取乘法操作符的优先级
        prec_mul = precedence["Mul"]
        # 获取原子操作符的优先级
        prec_atom = precedence["Atom"]
        # 获取环对象的引用
        ring = self.ring
        # 获取环的符号列表
        symbols = ring.symbols
        # 获取环的生成元个数
        ngens = ring.ngens
        # 获取环的零单项式
        zm = ring.zero_monom
        # 初始化用于存储结果的列表
        sexpvs = []

        # 遍历当前对象中的每一项
        for expv, coeff in self.terms():
            # 判断系数是否为负数
            negative = ring.domain.is_negative(coeff)
            # 根据系数正负性确定符号
            sign = " - " if negative else " + "
            sexpvs.append(sign)

            # 如果指数为零单项式，则打印系数
            if expv == zm:
                scoeff = printer._print(coeff)
                if negative and scoeff.startswith("-"):
                    scoeff = scoeff[1:]
            else:
                # 否则，处理非零单项式的情况
                if negative:
                    coeff = -coeff
                if coeff != self.ring.domain.one:
                    scoeff = printer.parenthesize(coeff, prec_mul, strict=True)
                else:
                    scoeff = ''

            # 处理符号变量部分
            sexpv = []
            for i in range(ngens):
                exp = expv[i]
                if not exp:
                    continue
                # 打印符号并确定优先级
                symbol = printer.parenthesize(symbols[i], prec_atom, strict=True)
                if exp != 1:
                    if exp != int(exp) or exp < 0:
                        sexp = printer.parenthesize(exp, prec_atom, strict=False)
                    else:
                        sexp = exp
                    sexpv.append(exp_pattern % (symbol, sexp))
                else:
                    sexpv.append('%s' % symbol)

            # 将系数添加到符号列表中
            if scoeff:
                sexpv = [scoeff] + sexpv
            sexpvs.append(mul_symbol.join(sexpv))

        # 处理首个符号
        if sexpvs[0] in [" + ", " - "]:
            head = sexpvs.pop(0)
            if head == " - ":
                sexpvs.insert(0, "-")

        # 返回格式化后的结果字符串
        return "".join(sexpvs)

    @property
    def is_generator(self):
        # 检查当前对象是否是环的生成元之一
        return self in self.ring._gens_set

    @property
    def is_ground(self):
        # 检查当前对象是否为空，或者是否仅包含环的零单项式
        return not self or (len(self) == 1 and self.ring.zero_monom in self)
    # 检查多项式是否是单项式（只有常数项或者空）
    def is_monomial(self):
        return not self or (len(self) == 1 and self.LC == 1)

    # 属性方法，检查多项式是否是单项式或者空
    @property
    def is_term(self):
        return len(self) <= 1

    # 属性方法，检查多项式的首项系数是否为负数
    @property
    def is_negative(self):
        return self.ring.domain.is_negative(self.LC)

    # 属性方法，检查多项式的首项系数是否为正数
    @property
    def is_positive(self):
        return self.ring.domain.is_positive(self.LC)

    # 属性方法，检查多项式的首项系数是否为非负数
    @property
    def is_nonnegative(self):
        return self.ring.domain.is_nonnegative(self.LC)

    # 属性方法，检查多项式的首项系数是否为非正数
    @property
    def is_nonpositive(self):
        return self.ring.domain.is_nonpositive(self.LC)

    # 静态方法，检查多项式是否为零多项式
    @property
    def is_zero(f):
        return not f

    # 静态方法，检查多项式是否为单位多项式
    @property
    def is_one(f):
        return f == f.ring.one

    # 属性方法，检查多项式是否为首一多项式（首项系数为环的单位元）
    @property
    def is_monic(f):
        return f.ring.domain.is_one(f.LC)

    # 属性方法，检查多项式是否为原始多项式（内容的首项系数为环的单位元）
    @property
    def is_primitive(f):
        return f.ring.domain.is_one(f.content())

    # 属性方法，检查多项式是否为线性多项式（每个单项式的次数和不大于1）
    @property
    def is_linear(f):
        return all(sum(monom) <= 1 for monom in f.itermonoms())

    # 属性方法，检查多项式是否为二次多项式（每个单项式的次数和不大于2）
    @property
    def is_quadratic(f):
        return all(sum(monom) <= 2 for monom in f.itermonoms())

    # 属性方法，检查多项式是否为无平方因子多项式
    @property
    def is_squarefree(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_sqf_p(f)

    # 属性方法，检查多项式是否为不可约多项式
    @property
    def is_irreducible(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_irreducible_p(f)

    # 属性方法，检查多项式是否为周期多项式
    @property
    def is_cyclotomic(f):
        if f.ring.is_univariate:
            return f.ring.dup_cyclotomic_p(f)
        else:
            raise MultivariatePolynomialError("cyclotomic polynomial")

    # 取反运算符重载，返回取反后的多项式
    def __neg__(self):
        return self.new([ (monom, -coeff) for monom, coeff in self.iterterms() ])

    # 正运算符重载，返回多项式本身（不变）
    def __pos__(self):
        return self
    # 定义一个特殊方法，用于多项式对象的加法操作
    def __add__(p1, p2):
        """Add two polynomials.
    
        Examples
        ========
    
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring
    
        >>> _, x, y = ring('x, y', ZZ)
        >>> (x + y)**2 + (x - y)**2
        2*x**2 + 2*y**2
    
        """
        # 如果 p2 是空的，直接返回 p1 的副本
        if not p2:
            return p1.copy()
        
        # 获取 p1 所属的环（环论中的一种代数结构）
        ring = p1.ring
        
        # 如果 p2 是 p1 所属环的元素，则进行相应加法操作
        if isinstance(p2, ring.dtype):
            p = p1.copy()  # 复制 p1，以确保不修改原始对象
            get = p.get  # 获取 p 中的项
            zero = ring.domain.zero  # 获取环的零元素
            # 遍历 p2 的项，将其加到 p 中相应的项上
            for k, v in p2.items():
                v = get(k, zero) + v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        
        # 如果 p2 是多项式元素，则根据情况进行处理
        elif isinstance(p2, PolyElement):
            # 如果 p1 所属环的域是多项式环，并且 p2 的环与 p1 的环相同，则直接返回 p2 和 p1 的加法结果
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            # 如果 p2 的环的域是多项式环，并且 p2 的域的环与 p1 的环相同，则返回 p2 的右加法操作结果
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__radd__(p1)
            else:
                return NotImplemented
        
        # 尝试将 p2 转换为 p1 所属环的新域元素
        try:
            cp2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()  # 复制 p1，以确保不修改原始对象
            if not cp2:
                return p
            zm = ring.zero_monom  # 获取环的零单项式
            # 如果 p1 中不包含零单项式，则将 cp2 赋给 p 的零单项式
            if zm not in p1.keys():
                p[zm] = cp2
            else:
                # 否则，如果 p2 是 -p[zm]，则删除 p 的零单项式；否则，将 cp2 加到 p 的零单项式上
                if p2 == -p[zm]:
                    del p[zm]
                else:
                    p[zm] += cp2
            return p
    
    # 定义一个特殊方法，用于在多项式对象之前添加一个对象的加法操作
    def __radd__(p1, n):
        p = p1.copy()  # 复制 p1，以确保不修改原始对象
        if not n:
            return p
        ring = p1.ring
        try:
            n = ring.domain_new(n)  # 尝试将 n 转换为 p1 所属环的新域元素
        except CoercionFailed:
            return NotImplemented
        else:
            zm = ring.zero_monom  # 获取环的零单项式
            # 如果 p1 中不包含零单项式，则将 n 赋给 p 的零单项式
            if zm not in p1.keys():
                p[zm] = n
            else:
                # 否则，如果 n 是 -p[zm]，则删除 p 的零单项式；否则，将 n 加到 p 的零单项式上
                if n == -p[zm]:
                    del p[zm]
                else:
                    p[zm] += n
            return p
    def __sub__(p1, p2):
        """Subtract polynomial p2 from p1.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = x + y**2
        >>> p2 = x*y + y**2
        >>> p1 - p2
        -x*y + x

        """
        # 如果 p2 是零多项式，则返回 p1 的副本
        if not p2:
            return p1.copy()
        
        # 获取 p1 的环
        ring = p1.ring
        
        # 如果 p2 是 p1 的元素类型，则进行多项式相减操作
        if isinstance(p2, ring.dtype):
            p = p1.copy()  # 复制 p1 的多项式
            get = p.get     # 获取多项式项的方法
            zero = ring.domain.zero  # 获取环的零元素
            for k, v in p2.items():
                v = get(k, zero) - v  # 从 p1 中获取对应项，然后相减
                if v:
                    p[k] = v   # 如果结果不为零，则更新 p 中的值
                else:
                    del p[k]    # 如果结果为零，则删除 p 中的项
            return p
        
        # 如果 p2 是多项式元素类型
        elif isinstance(p2, PolyElement):
            # 如果 p1 的环的域是多项式环，并且与 p2 的环相同，则跳过
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            # 如果 p2 的环的域是多项式环，并且与 p1 的环相同，则调用 p2 的右减方法
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rsub__(p1)
            else:
                return NotImplemented
        
        # 尝试将 p2 转换为 p1 的环的新元素
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()  # 复制 p1 的多项式
            zm = ring.zero_monom  # 获取环的零单项式
            if zm not in p1.keys():
                p[zm] = -p2    # 如果零单项式不在 p1 的项中，则直接赋值为 -p2
            else:
                if p2 == p[zm]:
                    del p[zm]   # 如果 p2 等于 p1 的零单项式，则删除零单项式
                else:
                    p[zm] -= p2  # 否则，在 p1 的零单项式上减去 p2
            return p

    def __rsub__(p1, n):
        """n - p1 with n convertible to the coefficient domain.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y
        >>> 4 - p
        -x - y + 4

        """
        ring = p1.ring  # 获取 p1 的环
        try:
            n = ring.domain_new(n)  # 尝试将 n 转换为 p1 的环的新元素
        except CoercionFailed:
            return NotImplemented
        else:
            p = ring.zero   # 初始化一个零多项式
            for expv in p1:
                p[expv] = -p1[expv]  # 对 p1 的每个指数值，赋值为其相反数
            p += n  # 将 n 加到 p 中
            return p
    def __mul__(p1, p2):
        """Multiply two polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', QQ)
        >>> p1 = x + y
        >>> p2 = x - y
        >>> p1*p2
        x**2 - y**2

        """
        # 获取第一个多项式的环
        ring = p1.ring
        # 创建一个零多项式
        p = ring.zero
        # 如果任意一个多项式为空，则返回零多项式
        if not p1 or not p2:
            return p
        # 如果 p2 是环的元素
        elif isinstance(p2, ring.dtype):
            # 获取零元素和单项式乘法方法
            get = p.get
            zero = ring.domain.zero
            monomial_mul = ring.monomial_mul
            # 将 p2 转换为项列表
            p2it = list(p2.items())
            # 对 p1 中的每个项进行遍历
            for exp1, v1 in p1.items():
                # 对 p2 中的每个项进行遍历
                for exp2, v2 in p2it:
                    # 计算单项式乘积的指数
                    exp = monomial_mul(exp1, exp2)
                    # 计算对应的值并加到结果多项式中
                    p[exp] = get(exp, zero) + v1*v2
            # 去除结果多项式中的零项
            p.strip_zero()
            return p
        # 如果 p2 是多项式元素
        elif isinstance(p2, PolyElement):
            # 如果环的域是多项式环，并且 p2 的环与当前环相同，不进行任何操作
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            # 如果 p2 的环的域是多项式环，并且与当前环相同，则调用 p2 的右乘方法
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmul__(p1)
            else:
                return NotImplemented

        # 尝试将 p2 转换为当前环的元素
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            # 对 p1 中的每个项进行遍历，乘以 p2，并加到结果多项式中
            for exp1, v1 in p1.items():
                v = v1*p2
                if v:
                    p[exp1] = v
            return p

    def __rmul__(p1, p2):
        """p2 * p1 with p2 in the coefficient domain of p1.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y
        >>> 4 * p
        4*x + 4*y

        """
        # 创建一个零多项式
        p = p1.ring.zero
        # 如果 p2 是空的，则返回零多项式
        if not p2:
            return p
        # 尝试将 p2 转换为当前环的元素
        try:
            p2 = p.ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            # 对 p1 中的每个项进行遍历，乘以 p2，并加到结果多项式中
            for exp1, v1 in p1.items():
                v = p2*v1
                if v:
                    p[exp1] = v
            return p
    def __pow__(self, n):
        """将多项式提升到 `n` 次幂

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p**3
        x**3 + 3*x**2*y**2 + 3*x*y**4 + y**6

        """
        # 获取多项式所在的环
        ring = self.ring

        # 处理特殊情况：n 为 0
        if not n:
            if self:
                # 如果多项式不是零，则返回环的单位元素
                return ring.one
            else:
                # 否则抛出值错误异常，因为 0 的 0 次幂未定义
                raise ValueError("0**0")
        
        # 处理多项式长度为 1 的情况
        elif len(self) == 1:
            # 获取多项式的单项式和系数
            monom, coeff = list(self.items())[0]
            p = ring.zero
            # 如果系数是环的单位元素，则直接将单项式的 n 次幂作为结果的单项式
            if coeff == ring.domain.one:
                p[ring.monomial_pow(monom, n)] = coeff
            else:
                # 否则计算系数的 n 次幂，并将结果赋给对应的单项式
                p[ring.monomial_pow(monom, n)] = coeff**n
            return p

        # 对于环级数，只支持负指数和有理指数，且仅限于单项式
        n = int(n)
        if n < 0:
            # 如果指数为负数，则抛出值错误异常
            raise ValueError("Negative exponent")

        elif n == 1:
            # 如果指数为 1，则返回当前多项式的副本
            return self.copy()
        elif n == 2:
            # 如果指数为 2，则返回当前多项式的平方
            return self.square()
        elif n == 3:
            # 如果指数为 3，则返回当前多项式与其平方的乘积
            return self * self.square()
        elif len(self) <= 5: # TODO: use an actual density measure
            # 如果多项式长度小于等于 5（密度测量），则调用多项式的通用幂运算方法
            return self._pow_multinomial(n)
        else:
            # 否则调用通用幂运算方法
            return self._pow_generic(n)

    def _pow_generic(self, n):
        # 初始化结果多项式为环的单位元素
        p = self.ring.one
        # 初始化计算用的当前多项式为当前多项式本身
        c = self

        while True:
            if n & 1:
                # 如果 n 是奇数，则更新结果多项式为当前多项式与结果多项式的乘积
                p = p * c
                n -= 1
                if not n:
                    break

            # 更新当前多项式为其平方
            c = c.square()
            # 更新 n 为 n 的一半
            n = n // 2

        return p

    def _pow_multinomial(self, n):
        # 获取多项式长度与指数 n 的多项式系数
        multinomials = multinomial_coefficients(len(self), n).items()
        # 获取环的多项式乘幂与零单项式
        monomial_mulpow = self.ring.monomial_mulpow
        zero_monom = self.ring.zero_monom
        # 获取多项式的所有项
        terms = self.items()
        # 获取环的零元素和多项式
        zero = self.ring.domain.zero
        poly = self.ring.zero

        # 遍历多项式系数及其对应的多项式
        for multinomial, multinomial_coeff in multinomials:
            product_monom = zero_monom
            product_coeff = multinomial_coeff

            # 遍历当前多项式的每一项的指数与对应的单项式及系数
            for exp, (monom, coeff) in zip(multinomial, terms):
                if exp:
                    # 更新乘积单项式
                    product_monom = monomial_mulpow(product_monom, monom, exp)
                    # 更新乘积系数
                    product_coeff *= coeff**exp

            monom = tuple(product_monom)
            coeff = product_coeff

            # 将乘积系数与对应的单项式添加到结果多项式中
            coeff = poly.get(monom, zero) + coeff

            if coeff:
                poly[monom] = coeff
            elif monom in poly:
                del poly[monom]

        return poly
    # 定义一个方法，计算多项式的平方
    def square(self):
        """square of a polynomial

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p.square()
        x**2 + 2*x*y**2 + y**4

        """
        # 获取多项式环对象
        ring = self.ring
        # 初始化多项式 p 为零多项式
        p = ring.zero
        # 获取 p 的 get 方法
        get = p.get
        # 获取多项式的键列表
        keys = list(self.keys())
        # 获取环的零元素
        zero = ring.domain.zero
        # 获取环的 monomial_mul 方法，用于计算单项式乘积
        monomial_mul = ring.monomial_mul
        # 遍历多项式的键列表
        for i in range(len(keys)):
            k1 = keys[i]
            # 获取当前键 k1 对应的值
            pk = self[k1]
            # 再次遍历之前的键
            for j in range(i):
                k2 = keys[j]
                # 计算 k1 和 k2 的单项式乘积
                exp = monomial_mul(k1, k2)
                # 将结果加到 p[exp] 中
                p[exp] = get(exp, zero) + pk * self[k2]
        # 将 p 的值乘以 2
        p = p.imul_num(2)
        # 获取 p 的 get 方法
        get = p.get
        # 遍历多项式的键值对
        for k, v in self.items():
            # 计算 k 和 k 的单项式乘积
            k2 = monomial_mul(k, k)
            # 将结果加到 p[k2] 中
            p[k2] = get(k2, zero) + v ** 2
        # 去除 p 中的零项
        p.strip_zero()
        # 返回计算结果多项式 p
        return p

    # 定义多项式的右除法方法
    def __divmod__(p1, p2):
        # 获取多项式环对象
        ring = p1.ring

        # 如果 p2 是零多项式，则抛出 ZeroDivisionError 异常
        if not p2:
            raise ZeroDivisionError("polynomial division")
        # 如果 p2 是多项式环中的元素，则调用 p1.div(p2) 进行除法运算
        elif isinstance(p2, ring.dtype):
            return p1.div(p2)
        # 如果 p2 是 PolyElement 类型的实例
        elif isinstance(p2, PolyElement):
            # 如果当前环的域是多项式环，并且 p2 的环与当前环相同，则继续执行
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            # 如果 p2 的环的域是多项式环，并且 p2 的环与当前环相同，则调用 p2 的 __rdivmod__ 方法
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rdivmod__(p1)
            else:
                return NotImplemented

        try:
            # 尝试将 p2 转换为当前环的新元素
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            # 返回 p1 除以 p2 的商和余数
            return (p1.quo_ground(p2), p1.rem_ground(p2))

    # 定义多项式的右除法方法
    def __rdivmod__(p1, p2):
        # 返回 NotImplemented，表示不支持右除法
        return NotImplemented

    # 定义多项式的取模方法
    def __mod__(p1, p2):
        # 获取多项式环对象
        ring = p1.ring

        # 如果 p2 是零多项式，则抛出 ZeroDivisionError 异常
        if not p2:
            raise ZeroDivisionError("polynomial division")
        # 如果 p2 是多项式环中的元素，则调用 p1.rem(p2) 进行取模运算
        elif isinstance(p2, ring.dtype):
            return p1.rem(p2)
        # 如果 p2 是 PolyElement 类型的实例
        elif isinstance(p2, PolyElement):
            # 如果当前环的域是多项式环，并且 p2 的环与当前环相同，则继续执行
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            # 如果 p2 的环的域是多项式环，并且 p2 的环与当前环相同，则调用 p2 的 __rmod__ 方法
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmod__(p1)
            else:
                return NotImplemented

        try:
            # 尝试将 p2 转换为当前环的新元素
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            # 返回 p1 对 p2 取模的结果
            return p1.rem_ground(p2)

    # 定义多项式的右取模方法
    def __rmod__(p1, p2):
        # 返回 NotImplemented，表示不支持右取模
        return NotImplemented
    # 定义特殊方法 __truediv__，处理多项式的真除运算
    def __truediv__(p1, p2):
        # 获取 p1 的环
        ring = p1.ring

        # 检查除数是否为零
        if not p2:
            # 若除数为零，抛出 ZeroDivisionError 异常
            raise ZeroDivisionError("polynomial division")
        # 如果除数是多项式环的元素
        elif isinstance(p2, ring.dtype):
            # 如果除数是单项式
            if p2.is_monomial:
                # 返回 p1 乘以除数的逆
                return p1 * (p2 ** (-1))
            else:
                # 否则进行多项式的真除运算
                return p1.quo(p2)
        # 如果除数是多项式元素
        elif isinstance(p2, PolyElement):
            # 如果 p1 和 p2 的环相同
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            # 如果 p2 的环的域是多项式环，且和 p1 的环相同
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                # 调用 p2 的反向真除运算
                return p2.__rtruediv__(p1)
            else:
                # 其他情况返回 NotImplemented
                return NotImplemented

        # 尝试将 p2 转换为与 p1 环相同的类型
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            # 如果转换失败，返回 NotImplemented
            return NotImplemented
        else:
            # 否则进行 p1 对 p2 的地板除运算
            return p1.quo_ground(p2)

    # 定义特殊方法 __rtruediv__，默认返回 NotImplemented
    def __rtruediv__(p1, p2):
        return NotImplemented

    # 定义特殊方法 __floordiv__，与 __truediv__ 相同
    __floordiv__ = __truediv__

    # 定义特殊方法 __rfloordiv__，与 __rtruediv__ 相同
    __rfloordiv__ = __rtruediv__

    # TODO: use // (__floordiv__) for exquo()?

    # 定义 _term_div 方法，用于多项式的项级别的除法
    def _term_div(self):
        # 获取零单项式
        zm = self.ring.zero_monom
        # 获取环的域
        domain = self.ring.domain
        # 获取域的地板除法方法
        domain_quo = domain.quo
        # 获取单项式除法方法
        monomial_div = self.ring.monomial_div

        # 如果域是一个域
        if domain.is_Field:
            # 定义 term_div 函数进行项级别的除法
            def term_div(a_lm_a_lc, b_lm_b_lc):
                # 分离被除数的单项式和系数
                a_lm, a_lc = a_lm_a_lc
                # 分离除数的单项式和系数
                b_lm, b_lc = b_lm_b_lc
                # 如果除数的单项式是零单项式，通常情况下
                if b_lm == zm:
                    # 结果单项式为被除数的单项式
                    monom = a_lm
                else:
                    # 否则进行单项式的除法
                    monom = monomial_div(a_lm, b_lm)
                # 如果单项式除法成功
                if monom is not None:
                    # 返回结果单项式和系数的地板除法
                    return monom, domain_quo(a_lc, b_lc)
                else:
                    # 否则返回空
                    return None
        else:
            # 如果域不是一个域
            def term_div(a_lm_a_lc, b_lm_b_lc):
                # 分离被除数的单项式和系数
                a_lm, a_lc = a_lm_a_lc
                # 分离除数的单项式和系数
                b_lm, b_lc = b_lm_b_lc
                # 如果除数的单项式是零单项式，通常情况下
                if b_lm == zm:
                    # 结果单项式为被除数的单项式
                    monom = a_lm
                else:
                    # 否则进行单项式的除法
                    monom = monomial_div(a_lm, b_lm)
                # 如果单项式除法成功且系数取模为零
                if not (monom is None or a_lc % b_lc):
                    # 返回结果单项式和系数的地板除法
                    return monom, domain_quo(a_lc, b_lc)
                else:
                    # 否则返回空
                    return None

        # 返回 term_div 函数
        return term_div
    # 定义一个除法算法，参考[CLO]书中第64页内容
    # fv 是多项式数组，返回 qv, r 使得 self = sum(fv[i]*qv[i]) + r
    # 所有多项式都不能是 Laurent 多项式。

    # 如果 fv 是一个单一的多项式元素，则将其转换为列表形式
    ret_single = False
    if isinstance(fv, PolyElement):
        ret_single = True
        fv = [fv]

    # 如果 fv 中任意一个多项式为空，则抛出 ZeroDivisionError 异常
    if not all(fv):
        raise ZeroDivisionError("polynomial division")

    # 如果 self 是空多项式，则根据 ret_single 返回不同的值
    if not self:
        if ret_single:
            return ring.zero, ring.zero
        else:
            return [], ring.zero

    # 确保 self 和所有 fv 多项式具有相同的环（ring）
    for f in fv:
        if f.ring != ring:
            raise ValueError('self and f must have the same ring')

    # 初始化 qv 为长度为 s 的零多项式列表，p 为 self 的副本，r 为零多项式
    s = len(fv)
    qv = [ring.zero for i in range(s)]
    p = self.copy()
    r = ring.zero

    # 获取 self 的项除法函数
    term_div = self._term_div()

    # 获取 fv 中各个多项式的主导指数向量
    expvs = [fx.leading_expv() for fx in fv]

    # 当 p 非空时进行循环
    while p:
        i = 0
        divoccurred = 0
        # 在 fv 中遍历，尝试进行除法操作，直到找到可以除的项
        while i < s and divoccurred == 0:
            expv = p.leading_expv()
            # 调用 term_div 进行项除法，得到新的项和系数
            term = term_div((expv, p[expv]), (expvs[i], fv[i][expvs[i]]))
            if term is not None:
                expv1, c = term
                # 更新 qv[i]，将新的项添加到 qv[i] 中
                qv[i] = qv[i]._iadd_monom((expv1, c))
                # 更新 p，将 fv[i] 乘以新的项系数，然后加到 p 上
                p = p._iadd_poly_monom(fv[i], (expv1, -c))
                divoccurred = 1
            else:
                i += 1
        # 如果在 fv 中未找到可除项，则将 p 的主导项添加到余项 r 中
        if not divoccurred:
            expv = p.leading_expv()
            r = r._iadd_monom((expv, p[expv]))
            del p[expv]

    # 如果最后 p 变为空多项式，将其加到余项 r 中
    if expv == ring.zero_monom:
        r += p

    # 根据 ret_single 返回最终的结果
    if ret_single:
        if not qv:
            return ring.zero, r
        else:
            return qv[0], r
    else:
        return qv, r
    def rem(self, G):
        # 保存当前多项式对象的引用
        f = self
        # 如果 G 是 PolyElement 的实例，则转换为列表
        if isinstance(G, PolyElement):
            G = [G]
        # 如果 G 中存在空值，则引发 ZeroDivisionError 异常
        if not all(G):
            raise ZeroDivisionError("polynomial division")
        # 获取多项式对象的环
        ring = f.ring
        # 获取环的定义域
        domain = ring.domain
        # 获取定义域的零元素
        zero = domain.zero
        # 获取环的乘法操作函数 monomial_mul
        monomial_mul = ring.monomial_mul
        # 初始化余数 r
        r = ring.zero
        # 获取多项式对象的除法操作函数 term_div
        term_div = f._term_div()
        # 获取多项式对象的最高项 LT
        ltf = f.LT
        # 复制多项式对象 f
        f = f.copy()
        # 获取多项式对象的元素获取方法 get
        get = f.get
        # 循环处理多项式对象 f
        while f:
            # 遍历除数列表 G
            for g in G:
                # 计算当前多项式 f 的最高项与 g 的最高项的商
                tq = term_div(ltf, g.LT)
                if tq is not None:
                    m, c = tq
                    # 遍历 g 的每一项
                    for mg, cg in g.iterterms():
                        # 计算乘积的新单项式 m1
                        m1 = monomial_mul(mg, m)
                        # 计算新系数 c1
                        c1 = get(m1, zero) - c * cg
                        if not c1:
                            del f[m1]
                        else:
                            f[m1] = c1
                    # 更新 f 的最高项
                    ltm = f.leading_expv()
                    if ltm is not None:
                        ltf = ltm, f[ltm]
                    break
            else:
                # 如果没有可约项，则将当前最高项添加到余数 r 中
                ltm, ltc = ltf
                if ltm in r:
                    r[ltm] += ltc
                else:
                    r[ltm] = ltc
                del f[ltm]
                ltm = f.leading_expv()
                if ltm is not None:
                    ltf = ltm, f[ltm]

        # 返回最终的余数
        return r

    def quo(f, G):
        # 返回 f 与 G 的商
        return f.div(G)[0]

    def exquo(f, G):
        # 返回 f 与 G 的精确商
        q, r = f.div(G)

        # 如果余数 r 为零，则返回商 q
        if not r:
            return q
        # 否则引发 ExactQuotientFailed 异常
        else:
            raise ExactQuotientFailed(f, G)

    def _iadd_monom(self, mc):
        """add to self the monomial coeff*x0**i0*x1**i1*...
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x**4 + 2*y
        >>> m = (1, 2)
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        x**4 + 5*x*y**2 + 2*y
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        5*x*y**2 + x
        >>> p1 is p
        False

        """
        # 如果 self 是环的生成器之一，则复制 self
        if self in self.ring._gens_set:
            cpself = self.copy()
        else:
            cpself = self
        # 获取指数向量和系数
        expv, coeff = mc
        # 获取当前 self 中指数为 expv 的系数
        c = cpself.get(expv)
        # 如果系数 c 不存在，则将系数 coeff 添加到 self 中
        if c is None:
            cpself[expv] = coeff
        else:
            # 否则将 coeff 加到 c 上
            c += coeff
            if c:
                cpself[expv] = c
            else:
                # 如果 c 为零，则删除该项
                del cpself[expv]
        # 返回更新后的 self
        return cpself
    def _iadd_poly_monom(self, p2, mc):
        """
        将 (p)*(coeff*x0**i0*x1**i1*...) 的乘积添加到 self 中，除非 self 是生成元 —— 这种情况下只返回两者的和。

        mc 是一个元组，(monom, coeff)，其中 monom 是 (i0, i1, ...)。

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p1 = x**4 + 2*y
        >>> p2 = y + z
        >>> m = (1, 2, 3)
        >>> p1 = p1._iadd_poly_monom(p2, (m, 3))
        >>> p1
        x**4 + 3*x*y**3*z**3 + 3*x*y**2*z**4 + 2*y

        """
        p1 = self
        # 如果 self 是环的生成元，则复制一个副本以避免直接修改原对象
        if p1 in p1.ring._gens_set:
            p1 = p1.copy()
        (m, c) = mc
        get = p1.get
        zero = p1.ring.domain.zero
        monomial_mul = p1.ring.monomial_mul
        # 遍历 p2 中的每一项
        for k, v in p2.items():
            # 计算 k 和 m 的乘积，即新的键值对应的单项式
            ka = monomial_mul(k, m)
            # 获取当前 p1 中 ka 对应的系数，如果不存在则返回 zero
            coeff = get(ka, zero) + v*c
            if coeff:
                p1[ka] = coeff
            else:
                del p1[ka]
        return p1

    def degree(f, x=None):
        """
        返回 ``x`` 或者主变量中的主导次数。

        注意，0 的次数为负无穷 (``float('-inf')``)

        """
        i = f.ring.index(x)

        if not f:
            return ninf
        elif i < 0:
            return 0
        else:
            return max(monom[i] for monom in f.itermonoms())

    def degrees(f):
        """
        返回包含所有变量中的主导次数的元组。

        注意，0 的次数为负无穷 (``float('-inf')``)

        """
        if not f:
            return (ninf,)*f.ring.ngens
        else:
            return tuple(map(max, list(zip(*f.itermonoms()))))

    def tail_degree(f, x=None):
        """
        返回 ``x`` 或者主变量中的尾部次数。

        注意，0 的次数为负无穷 (``float('-inf')``)

        """
        i = f.ring.index(x)

        if not f:
            return ninf
        elif i < 0:
            return 0
        else:
            return min(monom[i] for monom in f.itermonoms())

    def tail_degrees(f):
        """
        返回包含所有变量中的尾部次数的元组。

        注意，0 的次数为负无穷 (``float('-inf')``)

        """
        if not f:
            return (ninf,)*f.ring.ngens
        else:
            return tuple(map(min, list(zip(*f.itermonoms()))))

    def leading_expv(self):
        """
        根据单项式顺序返回主导单项式元组。

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p = x**4 + x**3*y + x**2*z**2 + z**7
        >>> p.leading_expv()
        (4, 0, 0)

        """
        if self:
            return self.ring.leading_expv(self)
        else:
            return None
    def _get_coeff(self, expv):
        # 返回给定指数的系数，如果不存在则返回环的零元素
        return self.get(expv, self.ring.domain.zero)

    def coeff(self, element):
        """
        返回给定单项式旁边的系数。

        Parameters
        ==========

        element : PolyElement（带有 ``is_monomial = True``）或 1

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring("x,y,z", ZZ)
        >>> f = 3*x**2*y - x*y*z + 7*z**3 + 23

        >>> f.coeff(x**2*y)
        3
        >>> f.coeff(x*y)
        0
        >>> f.coeff(1)
        23

        """
        if element == 1:
            # 如果元素是 1，返回零单项式的系数
            return self._get_coeff(self.ring.zero_monom)
        elif isinstance(element, self.ring.dtype):
            # 如果元素是环的数据类型，则获取其所有单项式
            terms = list(element.iterterms())
            if len(terms) == 1:
                monom, coeff = terms[0]
                if coeff == self.ring.domain.one:
                    # 如果系数是环的单位元素，则返回指定单项式的系数
                    return self._get_coeff(monom)

        # 如果不符合上述条件，则引发异常
        raise ValueError("expected a monomial, got %s" % element)

    def const(self):
        """返回常数项系数。"""
        return self._get_coeff(self.ring.zero_monom)

    @property
    def LC(self):
        # 返回主项的首项系数
        return self._get_coeff(self.leading_expv())

    @property
    def LM(self):
        # 返回主项的首项单项式
        expv = self.leading_expv()
        if expv is None:
            return self.ring.zero_monom
        else:
            return expv

    def leading_monom(self):
        """
        返回作为多项式元素的主单项式。

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> (3*x*y + y**2).leading_monom()
        x*y

        """
        p = self.ring.zero
        expv = self.leading_expv()
        if expv:
            p[expv] = self.ring.domain.one
        return p

    @property
    def LT(self):
        # 返回主项的首项单项式和其系数
        expv = self.leading_expv()
        if expv is None:
            return (self.ring.zero_monom, self.ring.domain.zero)
        else:
            return (expv, self._get_coeff(expv))

    def leading_term(self):
        """返回作为多项式元素的主项。"""
        p = self.ring.zero
        expv = self.leading_expv()
        if expv is not None:
            p[expv] = self[expv]
        return p

    def _sorted(self, seq, order):
        if order is None:
            order = self.ring.order
        else:
            order = OrderOpt.preprocess(order)

        if order is lex:
            # 如果排序顺序是字典序，则按照单项式的字典序逆序排序
            return sorted(seq, key=lambda monom: monom[0], reverse=True)
        else:
            # 否则，按照给定的排序规则对单项式进行逆序排序
            return sorted(seq, key=lambda monom: order(monom[0]), reverse=True)
    def coeffs(self, order=None):
        """Ordered list of polynomial coefficients.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional
            The ordering of the polynomial coefficients.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.coeffs()
        [2, 1]
        >>> f.coeffs(grlex)
        [1, 2]

        """
        # 返回一个包含多项式系数的有序列表
        return [ coeff for _, coeff in self.terms(order) ]

    def monoms(self, order=None):
        """Ordered list of polynomial monomials.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional
            The ordering of the polynomial monomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.monoms()
        [(2, 3), (1, 7)]
        >>> f.monoms(grlex)
        [(1, 7), (2, 3)]

        """
        # 返回一个包含多项式单项式的有序列表
        return [ monom for monom, _ in self.terms(order) ]

    def terms(self, order=None):
        """Ordered list of polynomial terms.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional
            The ordering of the polynomial terms.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.terms()
        [((2, 3), 2), ((1, 7), 1)]
        >>> f.terms(grlex)
        [((1, 7), 1), ((2, 3), 2)]

        """
        # 返回一个包含多项式项的有序列表
        return self._sorted(list(self.items()), order)

    def itercoeffs(self):
        """Iterator over coefficients of a polynomial. """
        # 返回一个迭代器，迭代多项式的系数
        return iter(self.values())

    def itermonoms(self):
        """Iterator over monomials of a polynomial. """
        # 返回一个迭代器，迭代多项式的单项式
        return iter(self.keys())

    def iterterms(self):
        """Iterator over terms of a polynomial. """
        # 返回一个迭代器，迭代多项式的项
        return iter(self.items())

    def listcoeffs(self):
        """Unordered list of polynomial coefficients. """
        # 返回一个包含多项式系数的无序列表
        return list(self.values())

    def listmonoms(self):
        """Unordered list of polynomial monomials. """
        # 返回一个包含多项式单项式的无序列表
        return list(self.keys())

    def listterms(self):
        """Unordered list of polynomial terms. """
        # 返回一个包含多项式项的无序列表
        return list(self.items())
    def imul_num(p, c):
        """multiply inplace the polynomial p by an element in the
        coefficient ring, provided p is not one of the generators;
        else multiply not inplace

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p1 = p.imul_num(3)
        >>> p1
        3*x + 3*y**2
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p.imul_num(3)
        >>> p1
        3*x
        >>> p1 is p
        False

        """
        # 如果多项式 p 是生成元之一，则返回 p*c
        if p in p.ring._gens_set:
            return p*c
        # 如果 c 为零，则清空多项式 p 并返回
        if not c:
            p.clear()
            return
        # 遍历多项式 p 的每个指数，将系数乘以 c
        for exp in p:
            p[exp] *= c
        return p

    def content(f):
        """Returns GCD of polynomial's coefficients. """
        # 获取多项式的系数环
        domain = f.ring.domain
        # 初始化内容为零
        cont = domain.zero
        # 获取系数的最大公约数函数
        gcd = domain.gcd

        # 遍历多项式 f 的所有系数，并计算它们的最大公约数
        for coeff in f.itercoeffs():
            cont = gcd(cont, coeff)

        return cont

    def primitive(f):
        """Returns content and a primitive polynomial. """
        # 计算多项式的内容
        cont = f.content()
        # 将多项式除以内容，返回内容和结果多项式
        return cont, f.quo_ground(cont)

    def monic(f):
        """Divides all coefficients by the leading coefficient. """
        # 如果多项式为空，则返回空
        if not f:
            return f
        else:
            # 将多项式除以其首项系数，使其首项系数变为 1
            return f.quo_ground(f.LC)

    def mul_ground(f, x):
        if not x:
            return f.ring.zero

        # 将多项式的每一项乘以 x
        terms = [ (monom, coeff*x) for monom, coeff in f.iterterms() ]
        return f.new(terms)

    def mul_monom(f, monom):
        # 获取多项式的单项乘法函数
        monomial_mul = f.ring.monomial_mul
        # 将多项式的每一项乘以给定的单项 monom
        terms = [ (monomial_mul(f_monom, monom), f_coeff) for f_monom, f_coeff in f.items() ]
        return f.new(terms)

    def mul_term(f, term):
        monom, coeff = term

        # 如果多项式为空或系数为零，则返回零多项式
        if not f or not coeff:
            return f.ring.zero
        # 如果单项 monom 是零单项，则将多项式的每一项乘以系数 coeff
        elif monom == f.ring.zero_monom:
            return f.mul_ground(coeff)

        # 获取多项式的单项乘法函数
        monomial_mul = f.ring.monomial_mul
        # 将多项式的每一项乘以给定的单项 monom 和系数 coeff
        terms = [ (monomial_mul(f_monom, monom), f_coeff*coeff) for f_monom, f_coeff in f.items() ]
        return f.new(terms)

    def quo_ground(f, x):
        domain = f.ring.domain

        # 如果除数 x 为零，则抛出零除错误
        if not x:
            raise ZeroDivisionError('polynomial division')
        # 如果多项式为空或除数 x 等于环的单位元，则返回多项式本身
        if not f or x == domain.one:
            return f

        # 如果系数环是一个域，则使用域的商运算 quo
        if domain.is_Field:
            quo = domain.quo
            terms = [ (monom, quo(coeff, x)) for monom, coeff in f.iterterms() ]
        else:
            # 否则，直接进行整数除法
            terms = [ (monom, coeff // x) for monom, coeff in f.iterterms() if not (coeff % x) ]

        return f.new(terms)
    # 定义多项式的商操作函数，计算多项式 f 除以单项式 term 的商
    def quo_term(f, term):
        # 将 term 拆分为单项式 monom 和系数 coeff
        monom, coeff = term

        # 如果 coeff 为零，抛出 ZeroDivisionError 异常
        if not coeff:
            raise ZeroDivisionError("polynomial division")
        # 如果 f 为零多项式，则返回环中的零元素
        elif not f:
            return f.ring.zero
        # 如果 monom 等于 f 的零单项式，则返回 f 除以 coeff 的结果
        elif monom == f.ring.zero_monom:
            return f.quo_ground(coeff)

        # 获取多项式 f 的单项式除法函数
        term_div = f._term_div()

        # 对 f 的每一个单项式 t 进行 term 的除法操作，得到商的列表 terms
        terms = [term_div(t, term) for t in f.iterterms()]

        # 创建一个新的多项式，其系数为 terms 中不为 None 的项
        return f.new([t for t in terms if t is not None ])

    # 对多项式的系数进行模 p 的运算，返回结果的多项式
    def trunc_ground(f, p):
        # 如果环是整数环
        if f.ring.domain.is_ZZ:
            terms = []

            # 遍历 f 的每一个单项式 (monom, coeff)
            for monom, coeff in f.iterterms():
                # 计算 coeff 对 p 取模的结果
                coeff = coeff % p

                # 如果 coeff 大于 p 的一半，则将其调整为负数
                if coeff > p // 2:
                    coeff = coeff - p

                # 将调整后的 (monom, coeff) 添加到 terms 中
                terms.append((monom, coeff))
        else:
            # 如果环不是整数环，直接对每个 (monom, coeff) 求模 p
            terms = [(monom, coeff % p) for monom, coeff in f.iterterms()]

        # 根据 terms 创建一个新的多项式
        poly = f.new(terms)
        # 去除多项式 poly 的零系数项
        poly.strip_zero()
        return poly

    # 定义多项式的提取公因式操作，返回 f 和 g 的公因式 gcd，并将 f 和 g 除以 gcd 的结果返回
    def extract_ground(self, g):
        f = self
        # 计算 f 和 g 的内容系数
        fc = f.content()
        gc = g.content()

        # 计算 f 和 g 的内容系数的最大公因数 gcd
        gcd = f.ring.domain.gcd(fc, gc)

        # 将 f 和 g 分别除以 gcd，返回结果
        f = f.quo_ground(gcd)
        g = g.quo_ground(gcd)

        return gcd, f, g

    # 计算多项式的范数，使用给定的范数函数 norm_func
    def _norm(f, norm_func):
        # 如果 f 是零多项式，则返回环中的零元素
        if not f:
            return f.ring.domain.zero
        else:
            # 获取环中的绝对值函数 ground_abs
            ground_abs = f.ring.domain.abs
            # 对 f 的每个系数 coeff，计算其绝对值，然后使用 norm_func 对这些绝对值进行处理
            return norm_func([ground_abs(coeff) for coeff in f.itercoeffs() ])

    # 计算多项式的最大范数，即系数绝对值的最大值
    def max_norm(f):
        return f._norm(max)

    # 计算多项式的 L1 范数，即系数绝对值之和
    def l1_norm(f):
        return f._norm(sum)

    # 对多项式列表中的多项式进行通用的压缩操作
    def deflate(f, *G):
        # 获取 f 所在的环
        ring = f.ring
        # 将 f 和 G 合并为一个多项式列表 polys
        polys = [f] + list(G)

        # 初始化 J 为长度为环中生成元数量的全零列表
        J = [0]*ring.ngens

        # 遍历 polys 中的每一个多项式 p
        for p in polys:
            # 遍历 p 的每一个单项式的单项式指数 monom
            for monom in p.itermonoms():
                # 更新 J 中每个位置的值为 monom 中对应位置的值和 J 中当前位置的最大公因数
                for i, m in enumerate(monom):
                    J[i] = igcd(J[i], m)

        # 如果 J 中所有元素都是 1，则直接返回 J 和 polys
        if all(b == 1 for b in J):
            return J, polys

        # 否则，对 polys 中的每一个多项式 p 进行压缩操作，得到压缩后的多项式列表 H
        H = []

        for p in polys:
            h = ring.zero

            # 遍历 p 的每一个单项式 (I, coeff)
            for I, coeff in p.iterterms():
                # 计算 I 中每个位置的值除以 J 中对应位置的值，得到新的指数 N
                N = [i // j for i, j in zip(I, J)]
                # 将系数 coeff 添加到多项式 h 中对应指数 N 的位置
                h[tuple(N)] = coeff

            # 将压缩后的多项式 h 添加到 H 中
            H.append(h)

        return J, H

    # 根据指数 J 对多项式 f 进行扩展操作，返回扩展后的多项式
    def inflate(f, J):
        poly = f.ring.zero

        # 遍历 f 的每一个单项式 (I, coeff)
        for I, coeff in f.iterterms():
            # 计算 I 中每个位置的值乘以 J 中对应位置的值，得到新的指数 N
            N = [i*j for i, j in zip(I, J)]
            # 将系数 coeff 添加到多项式 poly 中对应指数 N 的位置
            poly[tuple(N)] = coeff

        return poly

    # 计算多项式 f 和 g 的最小公倍数
    def lcm(self, g):
        f = self
        domain = f.ring.domain

        # 如果环不是域，则先将 f 和 g 化为原始多项式
        if not domain.is_Field:
            fc, f = f.primitive()
            gc, g = g.primitive()
            # 计算 fc 和 gc 的最小公倍数 c
            c = domain.lcm(fc, gc)

        # 计算 f 和 g 的乘积除以 f 和 g 的最大公因数 h
        h = (f*g).quo(f.gcd(g))

        # 如果环不是域，则将 h 乘以 c 返回
        if not domain.is_Field:
            return h.mul_ground(c)
        else:
            # 如果环是域，则返回 h 的首一多项式
            return h.monic()

    # 计算多项式 f 和 g 的最大公因式
    def gcd(f, g):
        return f.cofactors(g)[0]
    def cofactors(f, g):
        # 如果 f 和 g 都为空，则获取零元素并返回三个零
        if not f and not g:
            zero = f.ring.zero
            return zero, zero, zero
        # 如果 f 为空，则调用 g 的 _gcd_zero 方法
        elif not f:
            h, cff, cfg = f._gcd_zero(g)
            return h, cff, cfg
        # 如果 g 为空，则调用 f 的 _gcd_zero 方法
        elif not g:
            h, cfg, cff = g._gcd_zero(f)
            return h, cff, cfg
        # 如果 f 是单项式，则调用 f 的 _gcd_monom 方法
        elif len(f) == 1:
            h, cff, cfg = f._gcd_monom(g)
            return h, cff, cfg
        # 如果 g 是单项式，则调用 g 的 _gcd_monom 方法
        elif len(g) == 1:
            h, cfg, cff = g._gcd_monom(f)
            return h, cff, cfg

        # 对于一般情况，使用 f 的 deflate 方法，并获取结果 J 和 (f, g)
        J, (f, g) = f.deflate(g)
        # 调用 f 的 _gcd 方法，计算结果 h, cff, cfg
        h, cff, cfg = f._gcd(g)

        # 将结果 h, cff, cfg 充气回 J 的结果并返回
        return (h.inflate(J), cff.inflate(J), cfg.inflate(J))

    def _gcd_zero(f, g):
        # 获取 f 的环中的单位元和零元素
        one, zero = f.ring.one, f.ring.zero
        # 如果 g 是非负的，则返回 g, zero, one
        if g.is_nonnegative:
            return g, zero, one
        # 否则返回 -g, zero, -one
        else:
            return -g, zero, -one

    def _gcd_monom(f, g):
        # 获取 f 的环
        ring = f.ring
        # 获取环中的最大公因式和商
        ground_gcd = ring.domain.gcd
        ground_quo = ring.domain.quo
        # 获取环中的单项式最大公因式和单项式左除
        monomial_gcd = ring.monomial_gcd
        monomial_ldiv = ring.monomial_ldiv
        # 获取 f 的首项
        mf, cf = list(f.iterterms())[0]
        _mgcd, _cgcd = mf, cf
        # 对 g 的每个项进行迭代
        for mg, cg in g.iterterms():
            # 计算单项式最大公因式和环中的最大公因式
            _mgcd = monomial_gcd(_mgcd, mg)
            _cgcd = ground_gcd(_cgcd, cg)
        # 构造新的多项式 h
        h = f.new([(_mgcd, _cgcd)])
        # 构造新的多项式 cff
        cff = f.new([(monomial_ldiv(mf, _mgcd), ground_quo(cf, _cgcd))])
        # 构造新的多项式 cfg
        cfg = f.new([(monomial_ldiv(mg, _mgcd), ground_quo(cg, _cgcd)) for mg, cg in g.iterterms()])
        # 返回 h, cff, cfg
        return h, cff, cfg

    def _gcd(f, g):
        # 获取 f 的环
        ring = f.ring
        # 如果环是有理数域，则调用 f 的 _gcd_QQ 方法
        if ring.domain.is_QQ:
            return f._gcd_QQ(g)
        # 如果环是整数环，则调用 f 的 _gcd_ZZ 方法
        elif ring.domain.is_ZZ:
            return f._gcd_ZZ(g)
        # 否则调用 PRS 算法中的内部 gcd 计算方法
        else: # TODO: don't use dense representation (port PRS algorithms)
            return ring.dmp_inner_gcd(f, g)

    def _gcd_ZZ(f, g):
        # 调用 heugcd 函数计算 f 和 g 的 gcd
        return heugcd(f, g)

    def _gcd_QQ(self, g):
        # 获取 self
        f = self
        # 获取 f 的环
        ring = f.ring
        # 获取环的新副本，域为环的域的环
        new_ring = ring.clone(domain=ring.domain.get_ring())

        # 清除 f 和 g 的分母，获取结果 cf 和 f, cg 和 g
        cf, f = f.clear_denoms()
        cg, g = g.clear_denoms()

        # 将 f 和 g 设置为新环
        f = f.set_ring(new_ring)
        g = g.set_ring(new_ring)

        # 调用 f 的 _gcd_ZZ 方法，获取结果 h, cff, cfg
        h, cff, cfg = f._gcd_ZZ(g)

        # 将 h 设置回原环，并获取 h 的首项和首项的最小多项式
        h = h.set_ring(ring)
        c, h = h.LC, h.monic()

        # 将 cff 设置回原环，并乘以环中的 c/cf
        cff = cff.set_ring(ring).mul_ground(ring.domain.quo(c, cf))
        # 将 cfg 设置回原环，并乘以环中的 c/cg
        cfg = cfg.set_ring(ring).mul_ground(ring.domain.quo(c, cg))

        # 返回 h, cff, cfg
        return h, cff, cfg
    def cancel(self, g):
        """
        Cancel common factors in a rational function ``f/g``.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x,y", ZZ)

        >>> (2*x**2 - 2).cancel(x**2 - 2*x + 1)
        (2*x + 2, x - 1)

        """
        # 将 self 赋值给 f
        f = self
        # 获取 f 所属的环
        ring = f.ring

        # 如果 f 是零多项式，则返回自身和环的单位元
        if not f:
            return f, ring.one

        # 获取 f 的定义域
        domain = ring.domain

        # 如果定义域不是域或者没有相关的环结构
        if not (domain.is_Field and domain.has_assoc_Ring):
            # 对 f 和 g 进行因式分解
            _, p, q = f.cofactors(g)
        else:
            # 克隆一个新的环，将其定义域设置为 domain 的环
            new_ring = ring.clone(domain=domain.get_ring())

            # 清除 f 和 g 的分母，并返回清除后的结果
            cq, f = f.clear_denoms()
            cp, g = g.clear_denoms()

            # 将 f 和 g 的环设置为新环 new_ring
            f = f.set_ring(new_ring)
            g = g.set_ring(new_ring)

            # 对 f 和 g 进行因式分解
            _, p, q = f.cofactors(g)
            # 对 cp 和 cq 进行因式分解
            _, cp, cq = new_ring.domain.cofactors(cp, cq)

            # 将 p 和 q 的环设置为原始环 ring
            p = p.set_ring(ring)
            q = q.set_ring(ring)

            # 将 p 和 q 乘以 cp 和 cq
            p = p.mul_ground(cp)
            q = q.mul_ground(cq)

        # 使 q 相对于符号或象限成为规范形式，对于 ZZ_I 或 QQ_I 类型的环。
        # 这确保了分母的 LC 通过乘以环的单位来变成规范形式。
        u = q.canonical_unit()
        if u == domain.one:
            pass
        elif u == -domain.one:
            p, q = -p, -q
        else:
            p = p.mul_ground(u)
            q = q.mul_ground(u)

        # 返回简化后的分子 p 和分母 q
        return p, q

    def canonical_unit(f):
        # 获取 f 的环的定义域
        domain = f.ring.domain
        # 返回 f.LC 的规范单位
        return domain.canonical_unit(f.LC)

    def diff(f, x):
        """Computes partial derivative in ``x``.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring("x,y", ZZ)
        >>> p = x + x**2*y**3
        >>> p.diff(x)
        2*x*y**3 + 1

        """
        # 获取 f 的环
        ring = f.ring
        # 获取变量 x 在环中的索引
        i = ring.index(x)
        # 获取 x 的单项式基础
        m = ring.monomial_basis(i)
        # 初始化结果 g 为零多项式
        g = ring.zero
        # 遍历 f 的所有项
        for expv, coeff in f.iterterms():
            # 如果当前项中包含 x
            if expv[i]:
                # 计算出当前项相对于 m 的商作为新的指数 e
                e = ring.monomial_ldiv(expv, m)
                # 将新项添加到结果 g 中，系数为 coeff*expv[i]
                g[e] = ring.domain_new(coeff * expv[i])
        # 返回求导后的多项式 g
        return g

    def __call__(f, *values):
        # 如果传入的值的数量大于 0 且不超过环的生成元数量
        if 0 < len(values) <= f.ring.ngens:
            # 调用 evaluate 方法，传入环的生成元和对应的值
            return f.evaluate(list(zip(f.ring.gens, values)))
        else:
            # 抛出值错误异常
            raise ValueError("expected at least 1 and at most %s values, got %s" % (f.ring.ngens, len(values)))
    # 定义一个方法 evaluate，用于计算多项式 f 在给定值 x 处的值，a 是可选的替换参数
    def evaluate(self, x, a=None):
        # 复制当前对象 f 到变量 f
        f = self

        # 如果 x 是列表且 a 为空，则将 x 的第一个元素解包为 (X, a)，并将剩余元素赋值给 x
        if isinstance(x, list) and a is None:
            (X, a), x = x[0], x[1:]
            # 递归调用 evaluate 方法，计算 f 在 X 处的值，并使用 a 替换参数
            f = f.evaluate(X, a)

            # 如果 x 现在为空列表，则返回 f
            if not x:
                return f
            else:
                # 对 x 中的每个元素进行操作，将其映射为 (Y.drop(X), a) 的形式，并递归调用 evaluate 方法
                x = [ (Y.drop(X), a) for (Y, a) in x ]
                return f.evaluate(x)

        # 获取多项式的环 ring
        ring = f.ring
        # 获取变量 x 在环中的索引
        i = ring.index(x)
        # 将 a 转换为环 ring 中对应的类型
        a = ring.domain.convert(a)

        # 如果环中生成元素个数为 1
        if ring.ngens == 1:
            # 初始化结果为环的零元素
            result = ring.domain.zero

            # 遍历多项式 f 中的每个项 (n,), coeff
            for (n,), coeff in f.iterterms():
                # 计算当前项的值，并加到结果中
                result += coeff*a**n

            # 返回计算后的结果
            return result
        else:
            # 在环 ring 中移除变量 x，得到一个新的多项式 poly，初始值为零多项式
            poly = ring.drop(x).zero

            # 遍历多项式 f 中的每个项 (monom, coeff)
            for monom, coeff in f.iterterms():
                # 获取 monom 中索引为 i 的项，并从 monom 中移除该项，得到新的 monom
                n, monom = monom[i], monom[:i] + monom[i+1:]
                # 计算当前项的值 coeff*a**n

                coeff = coeff*a**n

                # 如果 monom 已经在 poly 中
                if monom in poly:
                    # 将 coeff 加到 poly[monom] 上
                    coeff = coeff + poly[monom]

                    # 如果 coeff 不为零，则更新 poly[monom]；否则从 poly 中删除 monom
                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                else:
                    # 如果 coeff 不为零，则将 monom 加入 poly 中
                    if coeff:
                        poly[monom] = coeff

            # 返回最终的多项式 poly
            return poly

    # 定义一个方法 subs，用于将多项式 f 中的变量 x 替换为给定值 a
    def subs(self, x, a=None):
        # 复制当前对象 f 到变量 f
        f = self

        # 如果 x 是列表且 a 为空，则对列表中每对 (X, a) 逐个调用 subs 方法
        if isinstance(x, list) and a is None:
            for X, a in x:
                f = f.subs(X, a)
            return f

        # 获取多项式的环 ring
        ring = f.ring
        # 获取变量 x 在环中的索引
        i = ring.index(x)
        # 将 a 转换为环 ring 中对应的类型
        a = ring.domain.convert(a)

        # 如果环中生成元素个数为 1
        if ring.ngens == 1:
            # 初始化结果为环的零元素
            result = ring.domain.zero

            # 遍历多项式 f 中的每个项 (n,), coeff
            for (n,), coeff in f.iterterms():
                # 计算当前项的值，并加到结果中
                result += coeff*a**n

            # 返回将结果转换为环的元素类型后的值
            return ring.ground_new(result)
        else:
            # 初始化结果为环的零多项式
            poly = ring.zero

            # 遍历多项式 f 中的每个项 (monom, coeff)
            for monom, coeff in f.iterterms():
                # 获取 monom 中索引为 i 的项，并在其位置插入 0，得到新的 monom
                n, monom = monom[i], monom[:i] + (0,) + monom[i+1:]
                # 计算当前项的值 coeff*a**n

                coeff = coeff*a**n

                # 如果 monom 已经在 poly 中
                if monom in poly:
                    # 将 coeff 加到 poly[monom] 上
                    coeff = coeff + poly[monom]

                    # 如果 coeff 不为零，则更新 poly[monom]；否则从 poly 中删除 monom
                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                else:
                    # 如果 coeff 不为零，则将 monom 加入 poly 中
                    if coeff:
                        poly[monom] = coeff

            # 返回最终的多项式 poly
            return poly
    # 组合函数，用于生成与变量 x 的代数结构相关的多项式
    def compose(f, x, a=None):
        # 获取环对象
        ring = f.ring
        # 创建零多项式
        poly = ring.zero
        # 构建生成器映射，将生成器与其索引对应起来
        gens_map = dict(zip(ring.gens, range(ring.ngens)))

        # 如果提供了替换变量 a
        if a is not None:
            replacements = [(x, a)]
        else:
            # 如果 x 是列表，则直接复制
            if isinstance(x, list):
                replacements = list(x)
            # 如果 x 是字典，则按生成器映射对其进行排序
            elif isinstance(x, dict):
                replacements = sorted(x.items(), key=lambda k: gens_map[k[0]])
            else:
                # 抛出值错误，要求提供生成器值对序列
                raise ValueError("expected a generator, value pair a sequence of such pairs")

        # 对替换变量进行遍历
        for k, (x, g) in enumerate(replacements):
            replacements[k] = (gens_map[x], ring.ring_new(g))

        # 对 f 中的每个单项进行遍历
        for monom, coeff in f.iterterms():
            # 将单项转换为列表形式
            monom = list(monom)
            # 初始化子多项式
            subpoly = ring.one

            # 对替换变量进行遍历
            for i, g in replacements:
                n, monom[i] = monom[i], 0
                # 如果 n 不为零，则将 g 的 n 次幂乘到子多项式上
                if n:
                    subpoly *= g**n

            # 将计算得到的子多项式乘以原系数，并加到总多项式上
            subpoly = subpoly.mul_term((tuple(monom), coeff))
            poly += subpoly

        # 返回组合后的多项式
        return poly

    # 返回 self 中与 x**deg 对应的系数作为同一个环中的多项式
    def coeff_wrt(self, x, deg):
        """
        Coefficient of ``self`` with respect to ``x**deg``.

        Treating ``self`` as a univariate polynomial in ``x`` this finds the
        coefficient of ``x**deg`` as a polynomial in the other generators.

        Parameters
        ==========

        x : generator or generator index
            The generator or generator index to compute the expression for.
        deg : int
            The degree of the monomial to compute the expression for.

        Returns
        =======

        :py:class:`~.PolyElement`
            The coefficient of ``x**deg`` as a polynomial in the same ring.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y, z = ring("x, y, z", ZZ)

        >>> p = 2*x**4 + 3*y**4 + 10*z**2 + 10*x*z**2
        >>> deg = 2
        >>> p.coeff_wrt(2, deg) # Using the generator index
        10*x + 10
        >>> p.coeff_wrt(z, deg) # Using the generator
        10*x + 10
        >>> p.coeff(z**2) # shows the difference between coeff and coeff_wrt
        10

        See Also
        ========

        coeff, coeffs

        """
        p = self
        # 获取变量 x 在环中的索引
        i = p.ring.index(x)
        # 获取所有与 x**deg 对应的单项式及其系数
        terms = [(m, c) for m, c in p.iterterms() if m[i] == deg]

        # 如果没有找到对应的单项式，则返回零多项式
        if not terms:
            return p.ring.zero

        # 分离单项式和系数
        monoms, coeffs = zip(*terms)
        # 将每个单项式中的 x**deg 替换为 0
        monoms = [m[:i] + (0,) + m[i + 1:] for m in monoms]
        # 从单项式和系数的字典中重建多项式
        return p.ring.from_dict(dict(zip(monoms, coeffs)))
    def prem(self, g, x=None):
        """
        Pseudo-remainder of the polynomial ``self`` with respect to ``g``.

        The pseudo-quotient ``q`` and pseudo-remainder ``r`` with respect to
        ``z`` when dividing ``f`` by ``g`` satisfy ``m*f = g*q + r``,
        where ``deg(r,z) < deg(g,z)`` and
        ``m = LC(g,z)**(deg(f,z) - deg(g,z)+1)``.

        See :meth:`pdiv` for explanation of pseudo-division.


        Parameters
        ==========

        g : :py:class:`~.PolyElement`
            The polynomial to divide ``self`` by.
        x : generator or generator index, optional
            The main variable of the polynomials and default is first generator.

        Returns
        =======

        :py:class:`~.PolyElement`
            The pseudo-remainder polynomial.

        Raises
        ======

        ZeroDivisionError : If ``g`` is the zero polynomial.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x, y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2
        >>> f.prem(g) # first generator is chosen by default if it is not given
        -4*y + 4
        >>> f.rem(g) # shows the differnce between prem and rem
        x**2 + x*y
        >>> f.prem(g, y) # generator is given
        0
        >>> f.prem(g, 1) # generator index is given
        0

        See Also
        ========

        pdiv, pquo, pexquo, sympy.polys.domains.ring.Ring.rem

        """
        f = self                              # 将当前对象赋值给变量 f
        x = f.ring.index(x)                   # 获取变量 x 在多项式环中的索引
        df = f.degree(x)                      # 计算多项式 f 关于变量 x 的阶数
        dg = g.degree(x)                      # 计算多项式 g 关于变量 x 的阶数

        if dg < 0:
            raise ZeroDivisionError('polynomial division')  # 如果 g 是零多项式，则抛出 ZeroDivisionError 异常

        r, dr = f, df                         # 初始化 r 为 f，dr 为 df

        if df < dg:
            return r                          # 若 f 的阶数小于 g 的阶数，则返回 r

        N = df - dg + 1                       # 计算 N，用于后续计算

        lc_g = g.coeff_wrt(x, dg)             # 计算 g 在变量 x 的最高次系数

        xp = f.ring.gens[x]                   # 获取多项式环中索引为 x 的变量

        while True:
            lc_r = r.coeff_wrt(x, dr)         # 计算 r 在变量 x 的最高次系数
            j, N = dr - dg, N - 1             # 更新 j 和 N

            R = r * lc_g                      # 计算 R
            G = g * lc_r * xp**j              # 计算 G
            r = R - G                         # 更新 r

            dr = r.degree(x)                  # 更新 r 的阶数

            if dr < dg:
                break

        c = lc_g ** N                         # 计算 c

        return r * c                          # 返回 r 乘以 c

    def pquo(self, g, x=None):
        """
        Polynomial pseudo-quotient in multivariate polynomial ring.

        Examples
        ========
        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2*y
        >>> h = 2*x + 2
        >>> f.pquo(g)
        2*x
        >>> f.quo(g) # shows the difference between pquo and quo
        0
        >>> f.pquo(h)
        2*x + 2*y - 2
        >>> f.quo(h) # shows the difference between pquo and quo
        0

        See Also
        ========

        prem, pdiv, pexquo, sympy.polys.domains.ring.Ring.quo

        """
        f = self                              # 将当前对象赋值给变量 f
        return f.pdiv(g, x)[0]                # 调用对象的 pdiv 方法，并返回其第一个返回值
    def pexquo(self, g, x=None):
        """
        Polynomial exact pseudo-quotient in multivariate polynomial ring.

        Examples
        ========
        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2*y
        >>> h = 2*x + 2
        >>> f.pexquo(g)
        2*x
        >>> f.exquo(g) # shows the differnce between pexquo and exquo
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x + 2*y does not divide x**2 + x*y
        >>> f.pexquo(h)
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x + 2 does not divide x**2 + x*y

        See Also
        ========

        prem, pdiv, pquo, sympy.polys.domains.ring.Ring.exquo

        """
        f = self
        # Compute polynomial division of f by g, returning quotient q and remainder r
        q, r = f.pdiv(g, x)

        if r.is_zero:
            return q
        else:
            # If remainder r is non-zero, raise ExactQuotientFailed exception
            raise ExactQuotientFailed(f, g)

    def subresultants(self, g, x=None):
        """
        Computes the subresultant PRS of two polynomials ``self`` and ``g``.

        Parameters
        ==========

        g : :py:class:`~.PolyElement`
            The second polynomial.
        x : generator or generator index
            The variable with respect to which the subresultant sequence is computed.

        Returns
        =======

        R : list
            Returns a list polynomials representing the subresultant PRS.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x, y", ZZ)

        >>> f = x**2*y + x*y
        >>> g = x + y
        >>> f.subresultants(g) # first generator is chosen by default if not given
        [x**2*y + x*y, x + y, y**3 - y**2]
        >>> f.subresultants(g, 0) # generator index is given
        [x**2*y + x*y, x + y, y**3 - y**2]
        >>> f.subresultants(g, y) # generator is given
        [x**2*y + x*y, x + y, x**3 + x**2]

        """
        f = self
        # Get the index of variable x in the polynomial ring
        x = f.ring.index(x)
        # Compute degrees of f and g with respect to x
        n = f.degree(x)
        m = g.degree(x)

        # Swap f and g if degree of f is less than degree of g
        if n < m:
            f, g = g, f
            n, m = m, n

        # If f is zero, return [0, 0]
        if f == 0:
            return [0, 0]

        # If g is zero, return [f, 1]
        if g == 0:
            return [f, 1]

        R = [f, g]

        d = n - m
        b = (-1) ** (d + 1)

        # Compute the pseudo-remainder for f and g
        h = f.prem(g, x)
        h = h * b

        # Compute the leading coefficient of g with respect to x**m
        lc = g.coeff_wrt(x, m)

        c = lc ** d

        S = [1, c]

        c = -c

        while h:
            k = h.degree(x)

            R.append(h)
            f, g, m, d = g, h, k, m - k

            b = -lc * c ** d
            h = f.prem(g, x)
            h = h.exquo(b)

            lc = g.coeff_wrt(x, k)

            if d > 1:
                p = (-lc) ** d
                q = c ** (d - 1)
                c = p.exquo(q)
            else:
                c = -lc

            S.append(-c)

        return R
    # TODO: following methods should point to polynomial
    # representation independent algorithm implementations.

    # 使用半扩展欧几里得算法计算多项式 f 和 g 的半扩展欧几里得算法
    def half_gcdex(f, g):
        return f.ring.dmp_half_gcdex(f, g)

    # 使用扩展欧几里得算法计算多项式 f 和 g 的扩展欧几里得算法
    def gcdex(f, g):
        return f.ring.dmp_gcdex(f, g)

    # 计算多项式 f 和 g 的结果式
    def resultant(f, g):
        return f.ring.dmp_resultant(f, g)

    # 计算多项式 f 的判别式
    def discriminant(f):
        return f.ring.dmp_discriminant(f)

    # 对多项式 f 进行分解，若多项式是单变量的则使用对应的分解方法
    def decompose(f):
        if f.ring.is_univariate:
            return f.ring.dup_decompose(f)
        else:
            raise MultivariatePolynomialError("polynomial decomposition")

    # 对多项式 f 进行平移，若多项式是单变量的则使用对应的平移方法
    def shift(f, a):
        if f.ring.is_univariate:
            return f.ring.dup_shift(f, a)
        else:
            raise MultivariatePolynomialError("shift: use shift_list instead")

    # 对多项式 f 进行列表形式的平移
    def shift_list(f, a):
        return f.ring.dmp_shift(f, a)

    # 计算多项式 f 的斯图姆序列，若多项式是单变量的则使用对应的斯图姆序列方法
    def sturm(f):
        if f.ring.is_univariate:
            return f.ring.dup_sturm(f)
        else:
            raise MultivariatePolynomialError("sturm sequence")

    # 计算多项式 f 的不可约因子列表
    def gff_list(f):
        return f.ring.dmp_gff_list(f)

    # 计算多项式 f 的范数
    def norm(f):
        return f.ring.dmp_norm(f)

    # 计算多项式 f 的平方自然数
    def sqf_norm(f):
        return f.ring.dmp_sqf_norm(f)

    # 计算多项式 f 的平方部分
    def sqf_part(f):
        return f.ring.dmp_sqf_part(f)

    # 计算多项式 f 的平方因式列表，可选择是否返回所有结果
    def sqf_list(f, all=False):
        return f.ring.dmp_sqf_list(f, all=all)

    # 计算多项式 f 的因式列表
    def factor_list(f):
        return f.ring.dmp_factor_list(f)
```