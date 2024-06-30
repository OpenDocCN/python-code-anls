# `D:\src\scipysrc\sympy\sympy\polys\fields.py`

```
"""Sparse rational function fields. """

from __future__ import annotations  # 导入用于支持类型注解的未来特性
from typing import Any  # 导入用于类型提示的 Any 类型
from functools import reduce  # 导入用于函数式编程的 reduce 函数

from operator import add, mul, lt, le, gt, ge  # 导入各种运算符函数

from sympy.core.expr import Expr  # 导入 SymPy 表达式的基类
from sympy.core.mod import Mod  # 导入 SymPy 中的模运算类
from sympy.core.numbers import Exp1  # 导入 SymPy 中的自然常数 e
from sympy.core.singleton import S  # 导入 SymPy 中的单例 S
from sympy.core.symbol import Symbol  # 导入 SymPy 中的符号类
from sympy.core.sympify import CantSympify, sympify  # 导入 SymPy 中的符号化函数和异常
from sympy.functions.elementary.exponential import ExpBase  # 导入 SymPy 中的指数函数基类
from sympy.polys.domains.domainelement import DomainElement  # 导入 SymPy 中的域元素类
from sympy.polys.domains.fractionfield import FractionField  # 导入 SymPy 中的分式域类
from sympy.polys.domains.polynomialring import PolynomialRing  # 导入 SymPy 中的多项式环类
from sympy.polys.constructor import construct_domain  # 导入 SymPy 中的构造域函数
from sympy.polys.orderings import lex  # 导入 SymPy 中的词典序排序
from sympy.polys.polyerrors import CoercionFailed  # 导入 SymPy 中的多项式错误类
from sympy.polys.polyoptions import build_options  # 导入 SymPy 中的多项式选项构建函数
from sympy.polys.polyutils import _parallel_dict_from_expr  # 导入 SymPy 中的并行字典构建函数
from sympy.polys.rings import PolyElement  # 导入 SymPy 中的多项式元素类
from sympy.printing.defaults import DefaultPrinting  # 导入 SymPy 中的默认打印类
from sympy.utilities import public  # 导入 SymPy 中的公共接口装饰器
from sympy.utilities.iterables import is_sequence  # 导入 SymPy 中的判断是否为序列的函数
from sympy.utilities.magic import pollute  # 导入 SymPy 中的全局变量污染函数

@public
def field(symbols, domain, order=lex):
    """Construct new rational function field returning (field, x1, ..., xn). """
    _field = FracField(symbols, domain, order)  # 构造一个分式域对象
    return (_field,) + _field.gens  # 返回分式域对象及其生成元组成的元组

@public
def xfield(symbols, domain, order=lex):
    """Construct new rational function field returning (field, (x1, ..., xn)). """
    _field = FracField(symbols, domain, order)  # 构造一个分式域对象
    return (_field, _field.gens)  # 返回分式域对象及其生成元组成的元组

@public
def vfield(symbols, domain, order=lex):
    """Construct new rational function field and inject generators into global namespace. """
    _field = FracField(symbols, domain, order)  # 构造一个分式域对象
    pollute([ sym.name for sym in _field.symbols ], _field.gens)  # 将生成元素注入到全局命名空间中
    return _field  # 返回分式域对象本身

@public
def sfield(exprs, *symbols, **options):
    """Construct a field deriving generators and domain
    from options and input expressions.

    Parameters
    ==========

    exprs   : py:class:`~.Expr` or sequence of :py:class:`~.Expr` (sympifiable)
        表达式或可符号化的表达式序列

    symbols : sequence of :py:class:`~.Symbol`/:py:class:`~.Expr`
        符号或表达式的序列

    options : keyword arguments understood by :py:class:`~.Options`
        :py:class:`~.Options` 理解的关键字参数

    Examples
    ========

    >>> from sympy import exp, log, symbols, sfield

    >>> x = symbols("x")
    >>> K, f = sfield((x*log(x) + 4*x**2)*exp(1/x + log(x)/3)/x**2)
    >>> K
    Rational function field in x, exp(1/x), log(x), x**(1/3) over ZZ with lex order
    >>> f
    (4*x**2*(exp(1/x)) + x*(exp(1/x))*(log(x)))/((x**(1/3))**5)
    """
    single = False  # 标志变量，表示是否为单个表达式
    if not is_sequence(exprs):  # 如果 exprs 不是序列
        exprs, single = [exprs], True  # 将 exprs 转换为单元素列表，并设置标志变量为 True

    exprs = list(map(sympify, exprs))  # 将表达式序列符号化
    opt = build_options(symbols, options)  # 构建选项
    numdens = []
    for expr in exprs:
        numdens.extend(expr.as_numer_denom())  # 将表达式转换为分子和分母，并扩展到 numdens 列表中
    reps, opt = _parallel_dict_from_expr(numdens, opt)  # 并行地从表达式中构建替换字典和选项
    # 如果 opt.domain 为 None
    if opt.domain is None:
        # 注意：这段代码效率较低，因为 construct_domain() 函数会自动执行到目标域的转换，这不应该发生。
        # 因此，这里的操作可能包含不必要的转换过程。
        
        # 将 reps 中各个字典的值（coeffs）合并成一个列表
        coeffs = sum([list(rep.values()) for rep in reps], [])
        
        # 调用 construct_domain() 函数，获取生成的域和一个不使用的值
        opt.domain, _ = construct_domain(coeffs, opt=opt)

    # 创建一个分式域对象 _field
    _field = FracField(opt.gens, opt.domain, opt.order)
    
    # 初始化一个空列表 fracs，用于存放分式对象
    fracs = []
    
    # 遍历 reps 列表，步长为2
    for i in range(0, len(reps), 2):
        # 将 reps 中每两个元素作为元组传递给 _field，生成分式对象，并添加到 fracs 列表中
        fracs.append(_field(tuple(reps[i:i+2])))

    # 如果 single 参数为 True，则返回分式域对象 _field 和 fracs 列表的第一个元素
    if single:
        return (_field, fracs[0])
    else:
        # 否则返回分式域对象 _field 和 fracs 列表
        return (_field, fracs)
_field_cache: dict[Any, Any] = {}

class FracField(DefaultPrinting):
    """Multivariate distributed rational function field. """

    def __new__(cls, symbols, domain, order=lex):
        # 导入多项式环模块
        from sympy.polys.rings import PolyRing
        # 创建多项式环对象
        ring = PolyRing(symbols, domain, order)
        # 提取环对象的属性
        symbols = ring.symbols
        ngens = ring.ngens
        domain = ring.domain
        order = ring.order

        # 用于哈希的元组
        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)
        # 尝试从缓存中获取对象
        obj = _field_cache.get(_hash_tuple)

        # 如果缓存中不存在，则创建新对象
        if obj is None:
            obj = object.__new__(cls)
            obj._hash_tuple = _hash_tuple
            obj._hash = hash(_hash_tuple)
            obj.ring = ring
            obj.dtype = type("FracElement", (FracElement,), {"field": obj})
            obj.symbols = symbols
            obj.ngens = ngens
            obj.domain = domain
            obj.order = order

            # 设置零元素和单位元素
            obj.zero = obj.dtype(ring.zero)
            obj.one = obj.dtype(ring.one)

            # 获取生成元素列表
            obj.gens = obj._gens()

            # 将生成元素与符号关联到对象属性
            for symbol, generator in zip(obj.symbols, obj.gens):
                if isinstance(symbol, Symbol):
                    name = symbol.name

                    if not hasattr(obj, name):
                        setattr(obj, name, generator)

            # 将新创建的对象添加到缓存中
            _field_cache[_hash_tuple] = obj

        # 返回对象
        return obj

    def _gens(self):
        """Return a list of polynomial generators. """
        # 返回多项式生成元素列表
        return tuple([ self.dtype(gen) for gen in self.ring.gens ])

    def __getnewargs__(self):
        # 返回对象的构造参数
        return (self.symbols, self.domain, self.order)

    def __hash__(self):
        # 返回对象的哈希值
        return self._hash

    def index(self, gen):
        # 确定生成元素在环中的索引
        if isinstance(gen, self.dtype):
            return self.ring.index(gen.to_poly())
        else:
            raise ValueError("expected a %s, got %s instead" % (self.dtype,gen))

    def __eq__(self, other):
        # 检查两个对象是否相等
        return isinstance(other, FracField) and \
            (self.symbols, self.ngens, self.domain, self.order) == \
            (other.symbols, other.ngens, other.domain, other.order)

    def __ne__(self, other):
        # 检查两个对象是否不相等
        return not self == other

    def raw_new(self, numer, denom=None):
        # 创建新的分式元素
        return self.dtype(numer, denom)
    
    def new(self, numer, denom=None):
        # 创建新的分式元素，对分子分母进行约分
        if denom is None: denom = self.ring.one
        numer, denom = numer.cancel(denom)
        return self.raw_new(numer, denom)

    def domain_new(self, element):
        # 将给定元素转换为当前域
        return self.domain.convert(element)
    # 定义方法 `ground_new`，用于将给定的元素转换为当前对象类型的新实例
    def ground_new(self, element):
        try:
            # 尝试使用环 `ring` 的方法 `ground_new` 将元素转换为新实例
            return self.new(self.ring.ground_new(element))
        except CoercionFailed:
            # 如果转换失败，获取当前对象的定义域 `domain`
            domain = self.domain

            # 如果 `domain` 不是域（Field）且具有关联的域（Field），则执行以下操作
            if not domain.is_Field and domain.has_assoc_Field:
                # 获取环 `ring` 的实例
                ring = self.ring
                # 获取定义域 `domain` 的域（Field）对象
                ground_field = domain.get_field()
                # 将元素转换为域 `ground_field` 的形式
                element = ground_field.convert(element)
                # 使用环 `ring` 的方法将分子部分转换为环元素
                numer = ring.ground_new(ground_field.numer(element))
                # 使用环 `ring` 的方法将分母部分转换为环元素
                denom = ring.ground_new(ground_field.denom(element))
                # 返回使用转换后的分子和分母构造的新实例
                return self.raw_new(numer, denom)
            else:
                # 如果不满足上述条件，抛出异常 `CoercionFailed`
                raise

    # 定义方法 `field_new`，用于根据给定的元素创建新的实例
    def field_new(self, element):
        # 如果元素是 `FracElement` 类型
        if isinstance(element, FracElement):
            # 如果当前对象与元素的域相同，直接返回元素
            if self == element.field:
                return element

            # 如果当前对象的定义域是分式域（FractionField）且与元素的域相同
            if isinstance(self.domain, FractionField) and \
                self.domain.field == element.field:
                # 调用 `ground_new` 方法处理元素
                return self.ground_new(element)
            # 如果当前对象的定义域是多项式环（PolynomialRing）且与元素的域相同
            elif isinstance(self.domain, PolynomialRing) and \
                self.domain.ring.to_field() == element.field:
                # 调用 `ground_new` 方法处理元素
                return self.ground_new(element)
            else:
                # 如果条件都不满足，抛出未实现异常
                raise NotImplementedError("conversion")
        # 如果元素是 `PolyElement` 类型
        elif isinstance(element, PolyElement):
            # 清除元素中的分母，获取清除后的分子和分母
            denom, numer = element.clear_denoms()

            # 如果当前对象的定义域是多项式环且分子的环与当前环相同
            if isinstance(self.domain, PolynomialRing) and \
                numer.ring == self.domain.ring:
                # 使用环 `ring` 的方法将分子转换为环元素
                numer = self.ring.ground_new(numer)
            # 如果当前对象的定义域是分式域且分子的环与定义域的环相同
            elif isinstance(self.domain, FractionField) and \
                numer.ring == self.domain.field.to_ring():
                # 使用环 `ring` 的方法将分子转换为环元素
                numer = self.ring.ground_new(numer)
            else:
                # 否则，将分子设置为当前环元素
                numer = numer.set_ring(self.ring)

            # 使用环 `ring` 的方法将分母转换为环元素
            denom = self.ring.ground_new(denom)
            # 返回使用转换后的分子和分母构造的新实例
            return self.raw_new(numer, denom)
        # 如果元素是二元组且长度为2
        elif isinstance(element, tuple) and len(element) == 2:
            # 将二元组中的元素都使用环 `ring` 的方法 `ring_new` 转换为环元素
            numer, denom = list(map(self.ring.ring_new, element))
            # 返回使用转换后的分子和分母构造的新实例
            return self.new(numer, denom)
        # 如果元素是字符串类型
        elif isinstance(element, str):
            # 抛出未实现异常，暂不支持解析字符串
            raise NotImplementedError("parsing")
        # 如果元素是表达式类型
        elif isinstance(element, Expr):
            # 调用 `from_expr` 方法将表达式转换为当前对象类型的新实例
            return self.from_expr(element)
        else:
            # 否则，调用 `ground_new` 方法处理元素
            return self.ground_new(element)

    # 将 `__call__` 方法指向 `field_new` 方法，使当前对象可调用
    __call__ = field_new
    # 重新构建表达式，根据给定的映射关系进行替换和重建
    def _rebuild_expr(self, expr, mapping):
        # 获取当前对象的定义域
        domain = self.domain
        # 生成包含所有幂次表达式的元组，以便后续匹配和替换
        powers = tuple((gen, gen.as_base_exp()) for gen in mapping.keys()
            if gen.is_Pow or isinstance(gen, ExpBase))

        # 定义内部函数，用于递归地重建表达式
        def _rebuild(expr):
            # 根据表达式获取其对应的生成器
            generator = mapping.get(expr)

            if generator is not None:
                return generator
            # 处理加法表达式
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            # 处理乘法表达式
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            # 处理幂次表达式或者相关的表达式
            elif expr.is_Pow or isinstance(expr, (ExpBase, Exp1)):
                b, e = expr.as_base_exp()
                # 在已有的幂次表达式中查找匹配的项，进行替换
                for gen, (bg, eg) in powers:
                    if bg == b and Mod(e, eg) == 0:
                        return mapping.get(gen)**int(e/eg)
                # 处理整数幂次和非单位的情况
                if e.is_Integer and e is not S.One:
                    return _rebuild(b)**int(e)
            # 处理倒数的情况
            elif mapping.get(1/expr) is not None:
                return 1/mapping.get(1/expr)

            try:
                # 尝试将表达式转换到当前的定义域中
                return domain.convert(expr)
            except CoercionFailed:
                # 如果转换失败，根据定义域的特性进行额外处理
                if not domain.is_Field and domain.has_assoc_Field:
                    return domain.get_field().convert(expr)
                else:
                    raise

        return _rebuild(expr)

    # 根据表达式构建分式
    def from_expr(self, expr):
        # 将符号和生成器进行一一对应映射
        mapping = dict(list(zip(self.symbols, self.gens)))

        try:
            # 尝试重新构建表达式，并转换为分式
            frac = self._rebuild_expr(sympify(expr), mapping)
        except CoercionFailed:
            # 如果转换失败，则抛出异常
            raise ValueError("expected an expression convertible to a rational function in %s, got %s" % (self, expr))
        else:
            # 根据重建后的分式创建新的字段
            return self.field_new(frac)

    # 将当前对象转换为分式域
    def to_domain(self):
        return FractionField(self)

    # 将当前对象转换为环
    def to_ring(self):
        from sympy.polys.rings import PolyRing
        # 返回多项式环对象，使用当前对象的符号、定义域和排序方式
        return PolyRing(self.symbols, self.domain, self.order)
class FracElement(DomainElement, DefaultPrinting, CantSympify):
    """Element of multivariate distributed rational function field. """

    def __init__(self, numer, denom=None):
        # 初始化方法，用于创建分数元素对象
        if denom is None:
            denom = self.field.ring.one  # 如果分母为None，则默认为环的单位元素
        elif not denom:
            raise ZeroDivisionError("zero denominator")  # 如果分母为0，则抛出ZeroDivisionError异常

        self.numer = numer  # 设置分子
        self.denom = denom  # 设置分母

    def raw_new(f, numer, denom):
        # 创建新的分数元素对象，不进行约分
        return f.__class__(numer, denom)
        
    def new(f, numer, denom):
        # 创建新的分数元素对象，并进行约分操作
        return f.raw_new(*numer.cancel(denom))

    def to_poly(f):
        # 将分数元素对象转换为多项式
        if f.denom != 1:
            raise ValueError("f.denom should be 1")  # 如果分母不为1，则引发ValueError异常
        return f.numer  # 返回分子作为多项式

    def parent(self):
        # 返回当前分数元素对象所属的域
        return self.field.to_domain()

    def __getnewargs__(self):
        # 返回用于序列化的参数，包括域、分子和分母
        return (self.field, self.numer, self.denom)

    _hash = None

    def __hash__(self):
        # 计算对象的哈希值，用于散列存储和比较
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.field, self.numer, self.denom))
        return _hash

    def copy(self):
        # 复制当前分数元素对象
        return self.raw_new(self.numer.copy(), self.denom.copy())

    def set_field(self, new_field):
        # 将当前分数元素对象转移到新的域中
        if self.field == new_field:
            return self
        else:
            new_ring = new_field.ring
            numer = self.numer.set_ring(new_ring)
            denom = self.denom.set_ring(new_ring)
            return new_field.new(numer, denom)

    def as_expr(self, *symbols):
        # 将分数元素对象表示为表达式
        return self.numer.as_expr(*symbols)/self.denom.as_expr(*symbols)

    def __eq__(f, g):
        # 判断两个分数元素对象是否相等
        if isinstance(g, FracElement) and f.field == g.field:
            return f.numer == g.numer and f.denom == g.denom
        else:
            return f.numer == g and f.denom == f.field.ring.one

    def __ne__(f, g):
        # 判断两个分数元素对象是否不相等
        return not f == g

    def __bool__(f):
        # 判断分数元素对象是否为真
        return bool(f.numer)

    def sort_key(self):
        # 返回分数元素对象的排序键
        return (self.denom.sort_key(), self.numer.sort_key())

    def _cmp(f1, f2, op):
        # 比较两个分数元素对象的排序键
        if isinstance(f2, f1.field.dtype):
            return op(f1.sort_key(), f2.sort_key())
        else:
            return NotImplemented

    def __lt__(f1, f2):
        # 小于比较操作符重载
        return f1._cmp(f2, lt)
    def __le__(f1, f2):
        # 小于等于比较操作符重载
        return f1._cmp(f2, le)
    def __gt__(f1, f2):
        # 大于比较操作符重载
        return f1._cmp(f2, gt)
    def __ge__(f1, f2):
        # 大于等于比较操作符重载
        return f1._cmp(f2, ge)

    def __pos__(f):
        # 一元正号操作符重载，返回当前对象
        return f.raw_new(f.numer, f.denom)

    def __neg__(f):
        # 一元负号操作符重载，返回当前对象的相反数
        return f.raw_new(-f.numer, f.denom)
    def _extract_ground(self, element):
        # 获取当前字段对象
        domain = self.field.domain

        try:
            # 尝试将元素转换为当前字段的类型
            element = domain.convert(element)
        except CoercionFailed:
            # 如果转换失败且当前域不是字段并且有关联字段
            if not domain.is_Field and domain.has_assoc_Field:
                # 获取关联的字段对象
                ground_field = domain.get_field()

                try:
                    # 尝试将元素转换为关联字段的类型
                    element = ground_field.convert(element)
                except CoercionFailed:
                    # 如果转换失败，直接返回
                    pass
                else:
                    # 如果转换成功，返回转换后的元组
                    return -1, ground_field.numer(element), ground_field.denom(element)

            # 如果无法转换，返回默认值
            return 0, None, None
        else:
            # 如果转换成功，返回转换后的元素
            return 1, element, None

    def __add__(f, g):
        """Add rational functions ``f`` and ``g``. """
        # 获取当前对象的字段
        field = f.field

        # 如果 g 为空，则返回 f
        if not g:
            return f
        # 如果 f 为空，则返回 g
        elif not f:
            return g
        # 如果 g 是当前字段的实例
        elif isinstance(g, field.dtype):
            # 如果 f 和 g 的分母相同，直接相加分子
            if f.denom == g.denom:
                return f.new(f.numer + g.numer, f.denom)
            else:
                # 否则按照分数相加的规则进行操作
                return f.new(f.numer*g.denom + f.denom*g.numer, f.denom*g.denom)
        # 如果 g 是当前字段的环的实例
        elif isinstance(g, field.ring.dtype):
            # 按照分数和环的规则进行操作
            return f.new(f.numer + f.denom*g, f.denom)
        else:
            # 处理特定的元素类型
            if isinstance(g, FracElement):
                # 如果当前域是分数域且与 g 的字段相同，则直接通过
                if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                    pass
                # 如果 g 的字段是分数域且与当前字段相同，则调用 g 的 __radd__ 方法
                elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                    return g.__radd__(f)
                else:
                    # 否则返回未实现
                    return NotImplemented
            # 处理多项式元素
            elif isinstance(g, PolyElement):
                # 如果当前域是多项式环且与 g 的环相同，则通过
                if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                    pass
                else:
                    # 否则调用 g 的 __radd__ 方法
                    return g.__radd__(f)

        # 如果无法匹配以上情况，则调用 f 的 __radd__ 方法
        return f.__radd__(g)

    def __radd__(f, c):
        # 如果 c 是当前字段的环的实例
        if isinstance(c, f.field.ring.dtype):
            # 按照分数和环的规则进行操作
            return f.new(f.numer + f.denom*c, f.denom)

        # 提取 c 的地面对象信息
        op, g_numer, g_denom = f._extract_ground(c)

        # 根据地面对象信息执行操作
        if op == 1:
            return f.new(f.numer + f.denom*g_numer, f.denom)
        elif not op:
            return NotImplemented
        else:
            return f.new(f.numer*g_denom + f.denom*g_numer, f.denom*g_denom)
    def __sub__(f, g):
        """Subtract rational functions ``f`` and ``g``. """
        # 获取有理函数 ``f`` 的域
        field = f.field

        # 如果 ``g`` 是零函数，则返回 ``f``
        if not g:
            return f
        # 如果 ``f`` 是零函数，则返回 ``-g``
        elif not f:
            return -g
        # 如果 ``g`` 是 ``field.dtype`` 的实例
        elif isinstance(g, field.dtype):
            # 如果分母相同，则返回差的分子和分母不变
            if f.denom == g.denom:
                return f.new(f.numer - g.numer, f.denom)
            # 如果分母不同，则返回通分后的差
            else:
                return f.new(f.numer*g.denom - f.denom*g.numer, f.denom*g.denom)
        # 如果 ``g`` 是 ``field.ring.dtype`` 的实例
        elif isinstance(g, field.ring.dtype):
            return f.new(f.numer - f.denom*g, f.denom)
        # 否则，进入复杂类型判断
        else:
            # 如果 ``g`` 是 ``FracElement`` 的实例
            if isinstance(g, FracElement):
                # 如果 ``field.domain`` 是 ``FractionField`` 并且与 ``g.field`` 相同
                if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                    pass
                # 如果 ``g.field.domain`` 是 ``FractionField`` 并且与 ``field`` 相同，调用 ``g`` 的右减法方法
                elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                    return g.__rsub__(f)
                # 否则，返回未实现
                else:
                    return NotImplemented
            # 如果 ``g`` 是 ``PolyElement`` 的实例
            elif isinstance(g, PolyElement):
                # 如果 ``field.domain`` 是 ``PolynomialRing`` 并且与 ``g.ring`` 相同，继续执行
                if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                    pass
                # 否则，调用 ``g`` 的右减法方法
                else:
                    return g.__rsub__(f)

        # 提取常数项操作
        op, g_numer, g_denom = f._extract_ground(g)

        # 根据提取的操作类型进行不同处理
        if op == 1:
            return f.new(f.numer - f.denom*g_numer, f.denom)
        elif not op:
            return NotImplemented
        else:
            return f.new(f.numer*g_denom - f.denom*g_numer, f.denom*g_denom)

    def __rsub__(f, c):
        # 如果 ``c`` 是 ``f.field.ring.dtype`` 的实例
        if isinstance(c, f.field.ring.dtype):
            return f.new(-f.numer + f.denom*c, f.denom)

        # 提取常数项操作
        op, g_numer, g_denom = f._extract_ground(c)

        # 根据提取的操作类型进行不同处理
        if op == 1:
            return f.new(-f.numer + f.denom*g_numer, f.denom)
        elif not op:
            return NotImplemented
        else:
            return f.new(-f.numer*g_denom + f.denom*g_numer, f.denom*g_denom)

    def __mul__(f, g):
        """Multiply rational functions ``f`` and ``g``. """
        # 获取有理函数 ``f`` 的域
        field = f.field

        # 如果 ``f`` 或 ``g`` 是零函数，则返回域的零元素
        if not f or not g:
            return field.zero
        # 如果 ``g`` 是 ``field.dtype`` 的实例
        elif isinstance(g, field.dtype):
            return f.new(f.numer*g.numer, f.denom*g.denom)
        # 如果 ``g`` 是 ``field.ring.dtype`` 的实例
        elif isinstance(g, field.ring.dtype):
            return f.new(f.numer*g, f.denom)
        # 否则，进入复杂类型判断
        else:
            # 如果 ``g`` 是 ``FracElement`` 的实例
            if isinstance(g, FracElement):
                # 如果 ``field.domain`` 是 ``FractionField`` 并且与 ``g.field`` 相同
                if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                    pass
                # 如果 ``g.field.domain`` 是 ``FractionField`` 并且与 ``field`` 相同，调用 ``g`` 的右乘法方法
                elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                    return g.__rmul__(f)
                # 否则，返回未实现
                else:
                    return NotImplemented
            # 如果 ``g`` 是 ``PolyElement`` 的实例
            elif isinstance(g, PolyElement):
                # 如果 ``field.domain`` 是 ``PolynomialRing`` 并且与 ``g.ring`` 相同，继续执行
                if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                    pass
                # 否则，调用 ``g`` 的右乘法方法
                else:
                    return g.__rmul__(f)

        # 如果以上条件均不符合，则调用 ``f`` 的右乘法方法
        return f.__rmul__(g)
    # 定义了一个特殊方法 __rmul__，用于实现右乘操作符 `*` 的重载
    def __rmul__(f, c):
        # 检查右操作数 c 是否属于字段 f.field.ring.dtype 类型
        if isinstance(c, f.field.ring.dtype):
            # 如果是，则返回一个新的分数对象，其分子为 f.numer * c，分母为 f.denom
            return f.new(f.numer * c, f.denom)

        # 否则，尝试从 c 中提取出地面元素及其运算符号、数值部分和分母部分
        op, g_numer, g_denom = f._extract_ground(c)

        # 根据提取出的运算符号进行处理
        if op == 1:
            # 如果运算符号为 1，则返回一个新的分数对象，其分子为 f.numer * g_numer，分母为 f.denom
            return f.new(f.numer * g_numer, f.denom)
        elif not op:
            # 如果运算符号不明确，则返回 Not Implemented
            return NotImplemented
        else:
            # 否则，返回一个新的分数对象，其分子为 f.numer * g_numer，分母为 f.denom * g_denom
            return f.new(f.numer * g_numer, f.denom * g_denom)

    # 定义了一个特殊方法 __truediv__，用于实现真除操作符 `/` 的重载
    def __truediv__(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
        # 获取字段对象
        field = f.field

        # 处理除数为零的情况
        if not g:
            raise ZeroDivisionError
        # 如果 g 是字段对象的实例，则返回一个新的分数对象，其分子为 f.numer * g.denom，分母为 f.denom * g.numer
        elif isinstance(g, field.dtype):
            return f.new(f.numer * g.denom, f.denom * g.numer)
        # 如果 g 是字段的环对象的实例，则返回一个新的分数对象，其分子为 f.numer，分母为 f.denom * g
        elif isinstance(g, field.ring.dtype):
            return f.new(f.numer, f.denom * g)
        else:
            # 对于其他类型的 g 进行更详细的类型检查和处理
            if isinstance(g, FracElement):
                if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                    pass
                elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                    return g.__rtruediv__(f)
                else:
                    return NotImplemented
            elif isinstance(g, PolyElement):
                if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                    pass
                else:
                    return g.__rtruediv__(f)

        # 尝试从 g 中提取出地面元素及其运算符号、数值部分和分母部分
        op, g_numer, g_denom = f._extract_ground(g)

        # 根据提取出的运算符号进行处理
        if op == 1:
            # 如果运算符号为 1，则返回一个新的分数对象，其分子为 f.numer，分母为 f.denom * g_numer
            return f.new(f.numer, f.denom * g_numer)
        elif not op:
            # 如果运算符号不明确，则返回 Not Implemented
            return NotImplemented
        else:
            # 否则，返回一个新的分数对象，其分子为 f.numer * g_denom，分母为 f.denom * g_numer
            return f.new(f.numer * g_denom, f.denom * g_numer)

    # 定义了一个特殊方法 __rtruediv__，用于实现右真除操作符 `/` 的重载
    def __rtruediv__(f, c):
        # 处理被除数为零的情况
        if not f:
            raise ZeroDivisionError
        # 如果 c 是字段 f.field.ring.dtype 类型的实例，则返回一个新的分数对象，其分子为 f.denom * c，分母为 f.numer
        elif isinstance(c, f.field.ring.dtype):
            return f.new(f.denom * c, f.numer)

        # 尝试从 c 中提取出地面元素及其运算符号、数值部分和分母部分
        op, g_numer, g_denom = f._extract_ground(c)

        # 根据提取出的运算符号进行处理
        if op == 1:
            # 如果运算符号为 1，则返回一个新的分数对象，其分子为 f.denom * g_numer，分母为 f.numer
            return f.new(f.denom * g_numer, f.numer)
        elif not op:
            # 如果运算符号不明确，则返回 Not Implemented
            return NotImplemented
        else:
            # 否则，返回一个新的分数对象，其分子为 f.denom * g_numer，分母为 f.numer * g_denom
            return f.new(f.denom * g_numer, f.numer * g_denom)

    # 定义了一个特殊方法 __pow__，用于实现乘方操作符 `**` 的重载
    def __pow__(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        # 如果 n 是非负数，则返回一个新的分数对象，其分子为 f.numer 的 n 次方，分母为 f.denom 的 n 次方
        if n >= 0:
            return f.raw_new(f.numer ** n, f.denom ** n)
        # 如果 f 为零，则抛出 ZeroDivisionError 异常
        elif not f:
            raise ZeroDivisionError
        else:
            # 否则，返回一个新的分数对象，其分子为 f.denom 的 -n 次方，分母为 f.numer 的 -n 次方
            return f.raw_new(f.denom ** -n, f.numer ** -n)

    # 定义了一个方法 diff，用于计算分数 f 在变量 x 上的偏导数
    def diff(f, x):
        """Computes partial derivative in ``x``.

        Examples
        ========

        >>> from sympy.polys.fields import field
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = field("x,y,z", ZZ)
        >>> ((x**2 + y)/(z + 1)).diff(x)
        2*x/(z + 1)

        """
        # 将 x 转换为多项式对象
        x = x.to_poly()
        # 返回一个新的分数对象，其分子为 f.numer 对 x 的导数乘以 f.denom，分母为 f.denom 的平方
        return f.new(f.numer.diff(x) * f.denom - f.numer * f.denom.diff(x), f.denom ** 2)

    # 定义了一个特殊方法 __call__，用于实现调用操作符 `()` 的重载
    def __call__(f, *values):
        # 如果 values 的长度大于 0 且不超过 f.field.ngens，则调用 f.evaluate 方法并返回结果
        if 0 < len(values) <= f.field.ngens:
            return f.evaluate(list(zip(f.field.gens, values)))
        else:
            # 否则，抛出 ValueError 异常，说明传入的值的数量不符合预期
            raise ValueError("expected at least 1 and at most %s values, got %s" % (f.field.ngens, len(values)))
    # 定义一个函数 evaluate，用于计算多项式 f 在指定点 x 处的值
    def evaluate(f, x, a=None):
        # 如果 x 是列表且 a 为 None，则将 x 列表中的每个元素 (X, a) 转换为多项式并保留 a
        if isinstance(x, list) and a is None:
            x = [(X.to_poly(), a) for X, a in x]
            # 分别计算 f 的分子和分母在 x 处的值
            numer, denom = f.numer.evaluate(x), f.denom.evaluate(x)
        else:
            # 将 x 转换为多项式
            x = x.to_poly()
            # 分别计算 f 的分子和分母在 x 处的值
            numer, denom = f.numer.evaluate(x, a), f.denom.evaluate(x, a)
    
        # 将分子和分母转换成一个新的域元素并返回
        field = numer.ring.to_field()
        return field.new(numer, denom)
    
    # 定义一个函数 subs，用于替换多项式 f 中的变量 x 为给定值 a
    def subs(f, x, a=None):
        # 如果 x 是列表且 a 为 None，则将 x 列表中的每个元素 (X, a) 转换为多项式并保留 a
        if isinstance(x, list) and a is None:
            x = [(X.to_poly(), a) for X, a in x]
            # 分别替换 f 的分子和分母中的变量
            numer, denom = f.numer.subs(x), f.denom.subs(x)
        else:
            # 将 x 转换为多项式
            x = x.to_poly()
            # 分别替换 f 的分子和分母中的变量
            numer, denom = f.numer.subs(x, a), f.denom.subs(x, a)
    
        # 返回一个新的多项式对象，替换后的分子和分母
        return f.new(numer, denom)
    
    # 定义一个函数 compose，抛出未实现错误，暂未实现此函数
    def compose(f, x, a=None):
        raise NotImplementedError
```