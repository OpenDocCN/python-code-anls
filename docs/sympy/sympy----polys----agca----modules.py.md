# `D:\src\scipysrc\sympy\sympy\polys\agca\modules.py`

```
"""
Computations with modules over polynomial rings.

This module implements various classes that encapsulate groebner basis
computations for modules. Most of them should not be instantiated by hand.
Instead, use the constructing routines on objects you already have.

For example, to construct a free module over ``QQ[x, y]``, call
``QQ[x, y].free_module(rank)`` instead of the ``FreeModule`` constructor.
In fact ``FreeModule`` is an abstract base class that should not be
instantiated, the ``free_module`` method instead returns the implementing class
``FreeModulePolyRing``.

In general, the abstract base classes implement most functionality in terms of
a few non-implemented methods. The concrete base classes supply only these
non-implemented methods. They may also supply new implementations of the
convenience methods, for example if there are faster algorithms available.
"""


from copy import copy
from functools import reduce

from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable

# TODO
# - module saturation
# - module quotient/intersection for quotient rings
# - free resoltutions / syzygies
# - finding small/minimal generating sets
# - ...

##########################################################################
## Abstract base classes #################################################
##########################################################################


class Module:
    """
    Abstract base class for modules.

    Do not instantiate - use ring explicit constructors instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> QQ.old_poly_ring(x).free_module(2)
    QQ[x]**2

    Attributes:

    - dtype - type of elements
    - ring - containing ring

    Non-implemented methods:

    - submodule
    - quotient_module
    - is_zero
    - is_submodule
    - multiply_ideal

    The method convert likely needs to be changed in subclasses.
    """

    def __init__(self, ring):
        # 初始化模块对象，指定其所属环
        self.ring = ring

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into internal representation of this module.

        If ``M`` is not None, it should be a module containing it.
        """
        # 将元素``elem``转换为此模块的内部表示形式
        if not isinstance(elem, self.dtype):
            raise CoercionFailed
        return elem

    def submodule(self, *gens):
        """Generate a submodule."""
        # 生成一个子模块的抽象方法，子类需实现具体逻辑
        raise NotImplementedError

    def quotient_module(self, other):
        """Generate a quotient module."""
        # 生成一个商模块的抽象方法，子类需实现具体逻辑
        raise NotImplementedError

    def __truediv__(self, e):
        # 实现模块的除法运算，用于生成商模块
        if not isinstance(e, Module):
            e = self.submodule(*e)
        return self.quotient_module(e)
    def contains(self, elem):
        """Return True if ``elem`` is an element of this module."""
        try:
            # 尝试将元素转换为模块元素
            self.convert(elem)
            # 如果转换成功，返回 True
            return True
        except CoercionFailed:
            # 如果转换失败，返回 False
            return False

    def __contains__(self, elem):
        # 调用 contains 方法来检查元素是否在模块中
        return self.contains(elem)

    def subset(self, other):
        """
        Returns True if ``other`` is is a subset of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.subset([(1, x), (x, 2)])
        True
        >>> F.subset([(1/x, x), (x, 2)])
        False
        """
        # 对于 other 中的每个元素 x，检查其是否在当前模块中
        return all(self.contains(x) for x in other)

    def __eq__(self, other):
        # 判断当前模块是否与另一个模块相等
        return self.is_submodule(other) and other.is_submodule(self)

    def __ne__(self, other):
        # 判断当前模块是否与另一个模块不相等
        return not (self == other)

    def is_zero(self):
        """Returns True if ``self`` is a zero module."""
        # 判断当前模块是否为零模块，抛出未实现错误
        raise NotImplementedError

    def is_submodule(self, other):
        """Returns True if ``other`` is a submodule of ``self``."""
        # 判断 other 是否为当前模块的子模块，抛出未实现错误
        raise NotImplementedError

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.
        """
        # 将当前模块乘以理想 other，抛出未实现错误
        raise NotImplementedError

    def __mul__(self, e):
        # 定义乘法运算符 * 的行为
        if not isinstance(e, Ideal):
            try:
                # 尝试将 e 转换为当前环的理想
                e = self.ring.ideal(e)
            except (CoercionFailed, NotImplementedError):
                # 如果转换失败，则返回 Not Implemented
                return NotImplemented
        # 返回当前模块乘以理想 e 的结果
        return self.multiply_ideal(e)

    __rmul__ = __mul__  # 右乘与左乘相同的定义

    def identity_hom(self):
        """Return the identity homomorphism on ``self``."""
        # 返回在当前模块上的恒同同态，抛出未实现错误
        raise NotImplementedError
class ModuleElement:
    """
    Base class for module element wrappers.

    Use this class to wrap primitive data types as module elements. It stores
    a reference to the containing module, and implements all the arithmetic
    operators.

    Attributes:

    - module - containing module
    - data - internal data

    Methods that likely need change in subclasses:

    - add
    - mul
    - div
    - eq
    """

    def __init__(self, module, data):
        # 初始化方法，设置模块引用和数据
        self.module = module
        self.data = data

    def add(self, d1, d2):
        """Add data ``d1`` and ``d2``."""
        # 加法操作，返回两个数据的和
        return d1 + d2

    def mul(self, m, d):
        """Multiply module data ``m`` by coefficient d."""
        # 乘法操作，将模块数据 ``m`` 乘以系数 ``d``
        return m * d

    def div(self, m, d):
        """Divide module data ``m`` by coefficient d."""
        # 除法操作，将模块数据 ``m`` 除以系数 ``d``
        return m / d

    def eq(self, d1, d2):
        """Return true if d1 and d2 represent the same element."""
        # 判断相等操作，如果 ``d1`` 和 ``d2`` 表示相同的元素则返回 True
        return d1 == d2

    def __add__(self, om):
        # 定义加法操作符重载，支持模块元素的加法
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return NotImplemented
        # 返回一个新的 ModuleElement 对象，表示两个元素相加的结果
        return self.__class__(self.module, self.add(self.data, om.data))

    __radd__ = __add__  # 右加法与左加法一致

    def __neg__(self):
        # 定义取负操作符重载，支持模块元素的取负
        return self.__class__(self.module, self.mul(self.data,
                       self.module.ring.convert(-1)))

    def __sub__(self, om):
        # 定义减法操作符重载，支持模块元素的减法
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return NotImplemented
        # 返回一个新的 ModuleElement 对象，表示两个元素相减的结果
        return self.__add__(-om)

    def __rsub__(self, om):
        # 定义右减法操作符重载，支持模块元素的右减法
        return (-self).__add__(om)

    def __mul__(self, o):
        # 定义乘法操作符重载，支持模块元素的乘法
        if not isinstance(o, self.module.ring.dtype):
            try:
                o = self.module.ring.convert(o)
            except CoercionFailed:
                return NotImplemented
        # 返回一个新的 ModuleElement 对象，表示模块元素乘以指定值的结果
        return self.__class__(self.module, self.mul(self.data, o))

    __rmul__ = __mul__  # 右乘法与左乘法一致

    def __truediv__(self, o):
        # 定义除法操作符重载，支持模块元素的除法
        if not isinstance(o, self.module.ring.dtype):
            try:
                o = self.module.ring.convert(o)
            except CoercionFailed:
                return NotImplemented
        # 返回一个新的 ModuleElement 对象，表示模块元素除以指定值的结果
        return self.__class__(self.module, self.div(self.data, o))

    def __eq__(self, om):
        # 定义相等比较操作符重载，支持模块元素的相等比较
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return False
        # 返回比较结果，判断两个元素是否相等
        return self.eq(self.data, om.data)

    def __ne__(self, om):
        # 定义不等比较操作符重载，支持模块元素的不等比较
        return not self == om

##########################################################################
## Free Modules ##########################################################
##########################################################################


class FreeModuleElement(ModuleElement):
    # FreeModuleElement 类继承自 ModuleElement 类，表示自由模块中的元素
    """Element of a free module. Data stored as a tuple."""

    # 定义一个自由模块中的元素类，数据以元组形式存储

    def add(self, d1, d2):
        # 将两个元组 d1 和 d2 对应位置的元素相加，返回结果元组
        return tuple(x + y for x, y in zip(d1, d2))

    def mul(self, d, p):
        # 将元组 d 中的每个元素与标量 p 相乘，返回结果元组
        return tuple(x * p for x in d)

    def div(self, d, p):
        # 将元组 d 中的每个元素都除以标量 p，返回结果元组
        return tuple(x / p for x in d)

    def __repr__(self):
        # 返回该对象的字符串表示形式
        from sympy.printing.str import sstr
        data = self.data
        # 如果数据中有任何一个元素是 DMP 类型的对象，则将其转换为 SymPy 的表示形式
        if any(isinstance(x, DMP) for x in data):
            data = [self.module.ring.to_sympy(x) for x in data]
        # 返回格式化后的字符串，包含所有数据元素的字符串表示形式
        return '[' + ', '.join(sstr(x) for x in data) + ']'

    def __iter__(self):
        # 返回一个迭代器，迭代器遍历该对象的 data 属性
        return self.data.__iter__()

    def __getitem__(self, idx):
        # 返回对象的 data 属性中索引为 idx 的元素
        return self.data[idx]
class FreeModule(Module):
    """
    Abstract base class for free modules.

    Additional attributes:

    - rank - rank of the free module

    Non-implemented methods:

    - submodule
    """

    # 默认的元素类型为 FreeModuleElement
    dtype = FreeModuleElement

    def __init__(self, ring, rank):
        # 调用父类 Module 的初始化方法
        Module.__init__(self, ring)
        # 设置当前自由模的秩
        self.rank = rank

    def __repr__(self):
        # 返回当前自由模的表示形式
        return repr(self.ring) + "**" + repr(self.rank)

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> F.is_submodule(F)
        True
        >>> F.is_submodule(M)
        True
        >>> M.is_submodule(F)
        False
        """
        # 如果 other 是 SubModule 类的实例，则检查其容器是否为当前自由模
        if isinstance(other, SubModule):
            return other.container == self
        # 如果 other 是 FreeModule 类的实例，则检查其环和秩是否与当前自由模相同
        if isinstance(other, FreeModule):
            return other.ring == self.ring and other.rank == self.rank
        # 其他情况返回 False
        return False

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.convert([1, 0])
        [1, 0]
        """
        # 如果 elem 是 FreeModuleElement 的实例，并且属于当前自由模，则返回 elem
        if isinstance(elem, FreeModuleElement):
            if elem.module is self:
                return elem
            # 如果 elem 的模的秩不等于当前自由模的秩，则抛出 CoercionFailed 异常
            if elem.module.rank != self.rank:
                raise CoercionFailed
            # 转换 elem 的数据为当前自由模的环的表示形式
            return FreeModuleElement(self,
                     tuple(self.ring.convert(x, elem.module.ring) for x in elem.data))
        # 如果 elem 是可迭代对象
        elif iterable(elem):
            # 转换 elem 中的每个元素为当前自由模的环的表示形式，构成一个元组
            tpl = tuple(self.ring.convert(x) for x in elem)
            # 如果转换后的元组长度不等于当前自由模的秩，则抛出 CoercionFailed 异常
            if len(tpl) != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self, tpl)
        # 如果 elem 等于 0
        elif _aresame(elem, 0):
            # 返回一个元组，元组中每个元素都为当前自由模的环的 0 表示形式
            return FreeModuleElement(self, (self.ring.convert(0),)*self.rank)
        else:
            # 其他情况抛出 CoercionFailed 异常
            raise CoercionFailed

    def is_zero(self):
        """
        Returns True if ``self`` is a zero module.

        (If, as this implementation assumes, the coefficient ring is not the
        zero ring, then this is equivalent to the rank being zero.)

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(0).is_zero()
        True
        >>> QQ.old_poly_ring(x).free_module(1).is_zero()
        False
        """
        # 如果当前自由模的秩为 0，则返回 True，否则返回 False
        return self.rank == 0
    def basis(self):
        """
        Return a set of basis elements.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(3).basis()
        ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        """
        # 导入单位矩阵模块
        from sympy.matrices import eye
        # 创建一个大小为 self.rank x self.rank 的单位矩阵 M
        M = eye(self.rank)
        # 返回一个元组，其中每个元素是 self.convert(M.row(i)) 的结果，i 从 0 到 self.rank-1
        return tuple(self.convert(M.row(i)) for i in range(self.rank))

    def quotient_module(self, submodule):
        """
        Return a quotient module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2)
        >>> M.quotient_module(M.submodule([1, x], [x, 2]))
        QQ[x]**2/<[1, x], [x, 2]>

        Or more conicisely, using the overloaded division operator:

        >>> QQ.old_poly_ring(x).free_module(2) / [[1, x], [x, 2]]
        QQ[x]**2/<[1, x], [x, 2]>
        """
        # 返回一个 QuotientModule 对象，用于表示当前模块除以指定子模块的商模块
        return QuotientModule(self.ring, self, submodule)

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x)
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.multiply_ideal(I)
        <[x, 0], [0, x]>
        """
        # 返回当前模块与其基的理想的乘积
        return self.submodule(*self.basis()).multiply_ideal(other)

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2
        [0, 1]])
        """
        # 导入同态映射模块中的同态映射函数
        from sympy.polys.agca.homomorphisms import homomorphism
        # 返回一个表示在 self 上的恒同同态映射的对象
        return homomorphism(self, self, self.basis())
class FreeModulePolyRing(FreeModule):
    """
    Free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of the ring instead:

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(3)
    >>> F
    QQ[x]**3
    >>> F.contains([x, 1, 0])
    True
    >>> F.contains([1/x, 0, 1])
    False
    """

    def __init__(self, ring, rank):
        # 导入多项式环基类
        from sympy.polys.domains.old_polynomialring import PolynomialRingBase
        # 调用父类初始化方法
        FreeModule.__init__(self, ring, rank)
        # 检查环是否为多项式环，如果不是则抛出异常
        if not isinstance(ring, PolynomialRingBase):
            raise NotImplementedError('This implementation only works over '
                                      + 'polynomial rings, got %s' % ring)
        # 检查环的基域是否为域，如果不是则抛出异常
        if not isinstance(ring.dom, Field):
            raise NotImplementedError('Ground domain must be a field, '
                                      + 'got %s' % ring.dom)

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x, y).free_module(2).submodule([x, x + y])
        >>> M
        <[x, x + y]>
        >>> M.contains([2*x, 2*x + 2*y])
        True
        >>> M.contains([x, y])
        False
        """
        # 返回子模块对象
        return SubModulePolyRing(gens, self, **opts)


class FreeModuleQuotientRing(FreeModule):
    """
    Free module over a quotient ring.

    Do not instantiate this, use the constructor method of the ring instead:

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(3)
    >>> F
    (QQ[x]/<x**2 + 1>)**3

    Attributes

    - quot - the quotient module `R^n / IR^n`, where `R/I` is our ring
    """

    def __init__(self, ring, rank):
        # 导入商环基类
        from sympy.polys.domains.quotientring import QuotientRing
        # 调用父类初始化方法
        FreeModule.__init__(self, ring, rank)
        # 检查环是否为商环，如果不是则抛出异常
        if not isinstance(ring, QuotientRing):
            raise NotImplementedError('This implementation only works over '
                             + 'quotient rings, got %s' % ring)
        # 构建商模块
        F = self.ring.ring.free_module(self.rank)
        self.quot = F / (self.ring.base_ideal*F)

    def __repr__(self):
        # 返回对象的字符串表示形式
        return "(" + repr(self.ring) + ")" + "**" + repr(self.rank)

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x, y)/[x**2 - y**2]).free_module(2).submodule([x, x + y])
        >>> M
        <[x + <x**2 - y**2>, x + y + <x**2 - y**2>]>
        >>> M.contains([y**2, x**2 + x*y])
        True
        >>> M.contains([x, y])
        False
        """
        # 返回子模块对象
        return SubModuleQuotientRing(gens, self, **opts)
    def lift(self, elem):
        """
        将 self 中的元素 elem 提升到 self.quot 模块中。

        注意，self.quot 和 self 是同一个集合，只是作为 R 模块和 R/I 模块的不同表示方式。

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        >>> e = F.convert([1, 0])
        >>> e
        [1 + <x**2 + 1>, 0 + <x**2 + 1>]
        >>> L = F.quot
        >>> l = F.lift(e)
        >>> l
        [1, 0] + <[x**2 + 1, 0], [0, x**2 + 1]>
        >>> L.contains(l)
        True
        """
        # 将元素 elem 的数据转换为 self.quot 模块中的元素并返回
        return self.quot.convert([x.data for x in elem])

    def unlift(self, elem):
        """
        将 self.quot 中的元素下推到 self 中。

        这是 lift 的逆操作。

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        >>> e = F.convert([1, 0])
        >>> l = F.lift(e)
        >>> e == l
        False
        >>> e == F.unlift(l)
        True
        """
        # 将 self.quot 中的元素数据转换为 self 中的元素并返回
        return self.convert(elem.data)
# 定义一个名为 SubModule 的类，继承自 Module 类
class SubModule(Module):
    """
    Base class for submodules.

    Attributes:

    - container - containing module  # 包含这个子模块的父模块
    - gens - generators (subset of containing module)  # 生成元（父模块的子集）
    - rank - rank of containing module  # 父模块的秩

    Non-implemented methods:

    - _contains  # 未实现的方法：包含性检查
    - _syzygies  # 未实现的方法：关于自身生成元的举例计算
    - _in_terms_of_generators  # 未实现的方法：表达式转化为生成元的表达
    - _intersect  # 未实现的方法：交集操作
    - _module_quotient  # 未实现的方法：模块商空间操作

    Methods that likely need change in subclasses:

    - reduce_element  # 子类可能需要修改的方法：元素的简化
    """

    def __init__(self, gens, container):
        # 调用父类 Module 的初始化方法
        Module.__init__(self, container.ring)
        # 将生成元转换为容器模块中的元素，并作为元组存储在 gens 中
        self.gens = tuple(container.convert(x) for x in gens)
        # 设置容器模块
        self.container = container
        # 设置模块的秩
        self.rank = container.rank
        # 设置模块的环
        self.ring = container.ring
        # 设置数据类型
        self.dtype = container.dtype

    def __repr__(self):
        # 返回模块的字符串表示，包括所有生成元的表示
        return "<" + ", ".join(repr(x) for x in self.gens) + ">"

    def _contains(self, other):
        """Implementation of containment.
           Other is guaranteed to be FreeModuleElement."""
        # 包含性检查的实现，参数 other 必须是 FreeModuleElement 类型
        raise NotImplementedError

    def _syzygies(self):
        """Implementation of syzygy computation wrt self generators."""
        # 关于自身生成元计算举例的实现
        raise NotImplementedError

    def _in_terms_of_generators(self, e):
        """Implementation of expression in terms of generators."""
        # 表达式转化为生成元表达的实现
        raise NotImplementedError

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        Mostly called implicitly.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([1, x])
        >>> M.convert([2, 2*x])
        [2, 2*x]
        """
        # 将 elem 转换为内部表示
        if isinstance(elem, self.container.dtype) and elem.module is self:
            return elem
        r = copy(self.container.convert(elem, M))
        r.module = self
        if not self._contains(r):
            raise CoercionFailed
        return r

    def _intersect(self, other):
        """Implementation of intersection.
           Other is guaranteed to be a submodule of same free module."""
        # 交集操作的实现，other 必须是相同自由模块的子模块
        raise NotImplementedError

    def _module_quotient(self, other):
        """Implementation of quotient.
           Other is guaranteed to be a submodule of same free module."""
        # 模块商空间操作的实现，other 必须是相同自由模块的子模块
        raise NotImplementedError
    # 求两个子模块的交集。
    def intersect(self, other, **options):
        """
        返回 ``self`` 和子模块 ``other`` 的交集。

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x, y).free_module(2)
        >>> F.submodule([x, x]).intersect(F.submodule([y, y]))
        <[x*y, x*y]>

        有些实现允许传递更多选项。目前唯一实现的选项是 ``relations=True``，此时函数将返回一个三元组 ``(res, rela, relb)``，
        其中 ``res`` 是交集模块，而 ``rela`` 和 ``relb`` 是系数向量的列表，表示 ``res`` 的生成元在 ``self`` 的生成元和
        ``other`` 的生成元中的表达。

        >>> F.submodule([x, x]).intersect(F.submodule([y, y]), relations=True)
        (<[x*y, x*y]>, [(DMP_Python([[1, 0]], QQ),)], [(DMP_Python([[1], []], QQ),)])

        上述结果表示：交集模块由单个元素 `(-xy, -xy) = -y (x, x) = -x (y, y)` 生成，其中
        `(x, x)` 和 `(y, y)` 分别是两个被交集的模块的唯一生成元。
        """
        # 如果 ``other`` 不是 SubModule 类型，则抛出类型错误异常
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        # 如果 ``other`` 所在的容器与当前模块不同，则抛出值错误异常
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        # 调用实际的交集计算函数，并传递其他选项
        return self._intersect(other, **options)
    def module_quotient(self, other, **options):
        r"""
        Returns the module quotient of ``self`` by submodule ``other``.

        That is, if ``self`` is the module `M` and ``other`` is `N`, then
        return the ideal `\{f \in R | fN \subset M\}`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> F = QQ.old_poly_ring(x, y).free_module(2)
        >>> S = F.submodule([x*y, x*y])
        >>> T = F.submodule([x, x])
        >>> S.module_quotient(T)
        <y>

        Some implementations allow further options to be passed. Currently, the
        only one implemented is ``relations=True``, which may only be passed
        if ``other`` is principal. In this case the function
        will return a pair ``(res, rel)`` where ``res`` is the ideal, and
        ``rel`` is a list of coefficient vectors, expressing the generators of
        the ideal, multiplied by the generator of ``other`` in terms of
        generators of ``self``.

        >>> S.module_quotient(T, relations=True)
        (<y>, [[DMP_Python([[1]], QQ)]])

        This means that the quotient ideal is generated by the single element
        `y`, and that `y (x, x) = 1 (xy, xy)`, `(x, x)` and `(xy, xy)` being
        the generators of `T` and `S`, respectively.
        """
        # 检查参数 `other` 是否为 SubModule 类型，如果不是则抛出类型错误
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        # 检查 `other` 是否和当前对象属于同一个自由模块，如果不是则抛出值错误
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        # 调用内部方法 `_module_quotient` 处理模块商并返回结果
        return self._module_quotient(other, **options)

    def union(self, other):
        """
        Returns the module generated by the union of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(1)
        >>> M = F.submodule([x**2 + x]) # <x(x+1)>
        >>> N = F.submodule([x**2 - 1]) # <(x-1)(x+1)>
        >>> M.union(N) == F.submodule([x+1])
        True
        """
        # 检查参数 `other` 是否为 SubModule 类型，如果不是则抛出类型错误
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        # 检查 `other` 是否和当前对象属于同一个自由模块，如果不是则抛出值错误
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        # 返回一个新的模块对象，包含 `self` 和 `other` 的生成器的并集
        return self.__class__(self.gens + other.gens, self.container)

    def is_zero(self):
        """
        Return True if ``self`` is a zero module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_zero()
        False
        >>> F.submodule([0, 0]).is_zero()
        True
        """
        # 检查模块的所有生成器是否都为零
        return all(x == 0 for x in self.gens)
    def submodule(self, *gens):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([x, 1])
        >>> M.submodule([x**2, x])
        <[x**2, x]>
        """
        # 检查给定的生成元是否是当前模块的子集，否则引发 ValueError
        if not self.subset(gens):
            raise ValueError('%s not a subset of %s' % (gens, self))
        # 使用当前类的构造函数创建一个新的子模块对象，并传递生成元和容器
        return self.__class__(gens, self.container)

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        # 检查当前模块是否包含其所在容器的全部基向量
        return all(self.contains(x) for x in self.container.basis())

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> N = M.submodule([2*x, x**2])
        >>> M.is_submodule(M)
        True
        >>> M.is_submodule(N)
        True
        >>> N.is_submodule(M)
        False
        """
        # 如果 other 是 SubModule 类型，则检查容器和生成元是否与当前模块匹配
        if isinstance(other, SubModule):
            return self.container == other.container and \
                all(self.contains(x) for x in other.gens)
        # 如果 other 是 FreeModule 或 QuotientModule 类型，则检查容器是否匹配且当前模块是否为全模块
        if isinstance(other, (FreeModule, QuotientModule)):
            return self.container == other and self.is_full_module()
        # 其他情况返回 False
        return False
    def syzygy_module(self, **opts):
        r"""
        Compute the syzygy module of the generators of ``self``.

        Suppose `M` is generated by `f_1, \ldots, f_n` over the ring
        `R`. Consider the homomorphism `\phi: R^n \to M`, given by
        sending `(r_1, \ldots, r_n) \to r_1 f_1 + \cdots + r_n f_n`.
        The syzygy module is defined to be the kernel of `\phi`.

        Examples
        ========

        The syzygy module is zero iff the generators generate freely a free
        submodule:

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([1, 0], [1, 1]).syzygy_module().is_zero()
        True

        A slightly more interesting example:

        >>> M = QQ.old_poly_ring(x, y).free_module(2).submodule([x, 2*x], [y, 2*y])
        >>> S = QQ.old_poly_ring(x, y).free_module(2).submodule([y, -x])
        >>> M.syzygy_module() == S
        True
        """
        # 创建自由模块 F，其基数等于生成器 self.gens 的数量
        F = self.ring.free_module(len(self.gens))
        # 过滤掉零子模，这是为了方便 _syzygies 函数的使用，并非替代真正的“生成集约简”算法
        return F.submodule(*[x for x in self._syzygies() if F.convert(x) != 0],
                           **opts)

    def in_terms_of_generators(self, e):
        """
        Express element ``e`` of ``self`` in terms of the generators.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([1, 0], [1, 1])
        >>> M.in_terms_of_generators([x, x**2])  # doctest: +SKIP
        [DMP_Python([-1, 1, 0], QQ), DMP_Python([1, 0, 0], QQ)]
        """
        try:
            # 尝试将元素 e 转换成 self 所属的环的元素
            e = self.convert(e)
        except CoercionFailed:
            raise ValueError('%s is not an element of %s' % (e, self))
        # 调用 _in_terms_of_generators 方法，返回表达式 e 关于生成器的表示
        return self._in_terms_of_generators(e)

    def reduce_element(self, x):
        """
        Reduce the element ``x`` of our ring modulo the ideal ``self``.

        Here "reduce" has no specific meaning, it could return a unique normal
        form, simplify the expression a bit, or just do nothing.
        """
        # 函数返回参数 x，没有进行具体的缩减操作
        return x
    def quotient_module(self, other, **opts):
        """
        返回商模块。

        这等同于从包含模块的商集合中取子模块。

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> S1 = F.submodule([x, 1])
        >>> S2 = F.submodule([x**2, x])
        >>> S1.quotient_module(S2)
        <[x, 1] + <[x**2, x]>>

        或者更简洁地，使用重载的除法运算符：

        >>> F.submodule([x, 1]) / [(x**2, x)]
        <[x, 1] + <[x**2, x]>>
        """
        if not self.is_submodule(other):
            raise ValueError('%s not a submodule of %s' % (other, self))
        # 返回一个新的子商模块对象
        return SubQuotientModule(self.gens,
                self.container.quotient_module(other), **opts)

    def __add__(self, oth):
        # 返回将当前模块与另一个模块的商模块，再对另一个对象进行类型转换的结果
        return self.container.quotient_module(self).convert(oth)

    __radd__ = __add__

    def multiply_ideal(self, I):
        """
        将 ``self`` 乘以理想 ``I``。

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2)
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([1, 1])
        >>> I*M
        <[x**2, x**2]>
        """
        # 返回模块乘以理想生成的结果
        return self.submodule(*[x*g for [x] in I._module.gens for g in self.gens])

    def inclusion_hom(self):
        """
        返回表示包含映射的同态。

        即，从 ``self`` 到 ``self.container`` 的自然映射。

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([x, x]).inclusion_hom()
        Matrix([
        [1, 0], : <[x, x]> -> QQ[x]**2
        [0, 1]])
        """
        # 返回一个限制域为当前模块的同态矩阵
        return self.container.identity_hom().restrict_domain(self)

    def identity_hom(self):
        """
        返回 ``self`` 上的恒同同态。

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([x, x]).identity_hom()
        Matrix([
        [1, 0], : <[x, x]> -> <[x, x]>
        [0, 1]])
        """
        # 返回一个限制定义域和陪域为当前模块的恒同同态
        return self.container.identity_hom().restrict_domain(
            self).restrict_codomain(self)
class SubQuotientModule(SubModule):
    """
    Submodule of a quotient module.

    Equivalently, quotient module of a submodule.

    Do not instantiate this, instead use the submodule or quotient_module
    constructing methods:

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> S = F.submodule([1, 0], [1, x])
    >>> Q = F/[(1, 0)]
    >>> S/[(1, 0)] == Q.submodule([5, x])
    True

    Attributes:

    - base - base module we are quotient of
    - killed_module - submodule used to form the quotient
    """
    def __init__(self, gens, container, **opts):
        # 调用父类构造函数初始化子模块，gens为生成元，container为容器
        SubModule.__init__(self, gens, container)
        # 将被消去的子模块赋值给killed_module属性
        self.killed_module = self.container.killed_module
        # XXX it is important for some code below that the generators of base
        # are in this particular order!
        # 基模块的生成元需按特定顺序排列以便后续代码正确运行
        self.base = self.container.base.submodule(
            *[x.data for x in self.gens], **opts).union(self.killed_module)

    def _contains(self, elem):
        # 判断元素elem是否在基模块中
        return self.base.contains(elem.data)

    def _syzygies(self):
        # 返回基模块的syzygies（线性代数中的关系向量）
        # 计算方法如注释中所述
        return [X[:len(self.gens)] for X in self.base._syzygies()]

    def _in_terms_of_generators(self, e):
        # 返回元素e在生成元的表达式中的系数
        return self.base._in_terms_of_generators(e.data)[:len(self.gens)]

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        # 检查当前模块是否是完全模块（即整个自由模块）
        return self.base.is_full_module()

    def quotient_hom(self):
        """
        Return the quotient homomorphism to self.

        That is, return the natural map from ``self.base`` to ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x).free_module(2) / [(1, x)]).submodule([1, 0])
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : <[1, 0], [1, x]> -> <[1, 0] + <[1, x]>, [1, x] + <[1, x]>>
        [0, 1]])
        """
        # 返回到当前模块的商同态映射
        return self.base.identity_hom().quotient_codomain(self.killed_module)


# 下面是匿名函数，不需要额外的注释
_subs0 = lambda x: x[0]
_subs1 = lambda x: x[1:]
class ModuleOrder(ProductOrder):
    """A product monomial order with a zeroth term as module index."""

    def __init__(self, o1, o2, TOP):
        # 根据 TOP 的值选择不同的顺序初始化 ProductOrder
        if TOP:
            ProductOrder.__init__(self, (o2, _subs1), (o1, _subs0))
        else:
            ProductOrder.__init__(self, (o1, _subs0), (o2, _subs1))


class SubModulePolyRing(SubModule):
    """
    Submodule of a free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of FreeModule instead:

    >>> from sympy.abc import x, y
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x, y).free_module(2)
    >>> F.submodule([x, y], [1, 0])
    <[x, y], [1, 0]>

    Attributes:

    - order - monomial order used
    """

    #self._gb - cached groebner basis
    #self._gbe - cached groebner basis relations

    def __init__(self, gens, container, order="lex", TOP=True):
        # 调用父类 SubModule 的初始化方法
        SubModule.__init__(self, gens, container)
        # 如果 container 不是 FreeModulePolyRing 的实例，则抛出未实现的错误
        if not isinstance(container, FreeModulePolyRing):
            raise NotImplementedError('This implementation is for submodules of '
                                      + 'FreeModulePolyRing, got %s' % container)
        # 初始化 self.order，使用 ModuleOrder 包装 monomial_key(order) 和 self.ring.order
        self.order = ModuleOrder(monomial_key(order), self.ring.order, TOP)
        # 初始化 self._gb 和 self._gbe 为 None
        self._gb = None
        self._gbe = None

    def __eq__(self, other):
        # 检查是否与另一个 SubModulePolyRing 对象相等，包括顺序是否相同
        if isinstance(other, SubModulePolyRing) and self.order != other.order:
            return False
        return SubModule.__eq__(self, other)

    def _groebner(self, extended=False):
        """Returns a standard basis in sdm form."""
        # 导入所需的函数
        from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_mora
        # 如果 extended 为 True 并且 self._gbe 是 None，则计算扩展格罗布纳基
        if self._gbe is None and extended:
            gb, gbe = sdm_groebner(
                [self.ring._vector_to_sdm(x, self.order) for x in self.gens],
                sdm_nf_mora, self.order, self.ring.dom, extended=True)
            self._gb, self._gbe = tuple(gb), tuple(gbe)
        # 如果 self._gb 是 None，则计算标准格罗布纳基
        if self._gb is None:
            self._gb = tuple(sdm_groebner(
                             [self.ring._vector_to_sdm(x, self.order) for x in self.gens],
               sdm_nf_mora, self.order, self.ring.dom))
        # 如果 extended 为 True，则返回格罗布纳基和扩展基础关系
        if extended:
            return self._gb, self._gbe
        else:
            # 否则只返回标准格罗布纳基
            return self._gb

    def _groebner_vec(self, extended=False):
        """Returns a standard basis in element form."""
        # 如果 extended 为 False，则返回以元素形式表示的标准基
        if not extended:
            return [FreeModuleElement(self,
                        tuple(self.ring._sdm_to_vector(x, self.rank)))
                    for x in self._groebner()]
        # 否则返回扩展的格罗布纳基，转换为元素形式
        gb, gbe = self._groebner(extended=True)
        return ([self.convert(self.ring._sdm_to_vector(x, self.rank))
                 for x in gb],
                [self.ring._sdm_to_vector(x, len(self.gens)) for x in gbe])
    def _contains(self, x):
        # 导入分布式模块中的零元素和 Mora 正规形式函数
        from sympy.polys.distributedmodules import sdm_zero, sdm_nf_mora
        # 将输入向量 x 转换为分布式模块形式，然后与自身的 Groebner 基进行 Mora 正规形式比较
        return sdm_nf_mora(self.ring._vector_to_sdm(x, self.order),
                           self._groebner(), self.order, self.ring.dom) == \
            sdm_zero()

    def _syzygies(self):
        """Compute syzygies. See [SCA, algorithm 2.5.4]."""
        # 注意：如果 self.gens 是标准基础，则可以使用 Schreyer 定理更有效地完成这个计算

        # 第一条要点
        k = len(self.gens)
        r = self.rank
        zero = self.ring.convert(0)
        one = self.ring.convert(1)
        Rkr = self.ring.free_module(r + k)
        newgens = []
        for j, f in enumerate(self.gens):
            m = [0]*(r + k)
            for i, v in enumerate(f):
                m[i] = f[i]
            for i in range(k):
                m[r + i] = one if j == i else zero
            m = FreeModuleElement(Rkr, tuple(m))
            newgens.append(m)
        # 注意：我们需要按照模块索引的降序排列，并且 TOP=False 以获得一个消去序
        F = Rkr.submodule(*newgens, order='ilex', TOP=False)

        # 第二条要点：计算 F 的标准基
        G = F._groebner_vec()

        # 第三条要点：G0 = G 与新的 k 个分量的交集
        G0 = [x[r:] for x in G if all(y == zero for y in x[:r])]

        # 第四条和第五条要点：计算完成
        return G0

    def _in_terms_of_generators(self, e):
        """Expression in terms of generators. See [SCA, 2.8.1]."""
        # 注意：如果 gens 是标准基础，则可以更有效地完成这个计算
        M = self.ring.free_module(self.rank).submodule(*((e,) + self.gens))
        S = M.syzygy_module(
            order="ilex", TOP=False)  # 我们希望降序排列！
        G = S._groebner_vec()
        # 这个列表不可能为空，因为 e 是一个元素
        e = [x for x in G if self.ring.is_unit(x[0])][0]
        return [-x/e[0] for x in e[1:]]

    def reduce_element(self, x, NF=None):
        """
        Reduce the element ``x`` of our container modulo ``self``.

        This applies the normal form ``NF`` to ``x``. If ``NF`` is passed
        as none, the default Mora normal form is used (which is not unique!).
        """
        # 导入分布式模块中的 Mora 正规形式函数
        from sympy.polys.distributedmodules import sdm_nf_mora
        if NF is None:
            NF = sdm_nf_mora
        return self.container.convert(self.ring._sdm_to_vector(NF(
            self.ring._vector_to_sdm(x, self.order), self._groebner(),
            self.order, self.ring.dom),
            self.rank))
    def _intersect(self, other, relations=False):
        # See: [SCA, section 2.8.2]
        # 获取当前对象的生成元
        fi = self.gens
        # 获取另一个对象的生成元
        hi = other.gens
        # 获取当前对象的秩
        r = self.rank
        # 创建一个 r × 2r 的零矩阵
        ci = [[0]*(2*r) for _ in range(r)]
        # 在零矩阵中设置对角线上的值
        for k in range(r):
            ci[k][k] = 1
            ci[k][r + k] = 1
        # 将当前对象的生成元扩展为 r 维，并添加 r 个零
        di = [list(f) + [0]*r for f in fi]
        # 将另一个对象的生成元扩展为 r 维，并在前面添加 r 个零
        ei = [[0]*r + list(h) for h in hi]
        # 构建自由模 R^{2r} 的子模，包含 ci, di, ei，并计算其 syzygies
        syz = self.ring.free_module(2*r).submodule(*(ci + di + ei))._syzygies()
        # 找到非零的 syzygies，即其中至少有一个元素不为零的部分
        nonzero = [x for x in syz if any(y != self.ring.zero for y in x[:r])]
        # 构建当前对象的容器的子模，包含非零 syzygies 中的前 r 个元素的负值
        res = self.container.submodule(*([-y for y in x[:r]] for x in nonzero))
        # 提取出与生成元 fi 相关的关系
        reln1 = [x[r:r + len(fi)] for x in nonzero]
        # 提取出与生成元 hi 相关的关系
        reln2 = [x[r + len(fi):] for x in nonzero]
        # 如果需要关系信息，返回 res, reln1, reln2；否则只返回 res
        if relations:
            return res, reln1, reln2
        return res

    def _module_quotient(self, other, relations=False):
        # See: [SCA, section 2.8.4]
        # 如果要求关系，并且 other 的生成元数量不为 1，则抛出 NotImplementedError
        if relations and len(other.gens) != 1:
            raise NotImplementedError
        # 如果 other 的生成元数量为 0，则返回 self.ring.ideal(1)
        if len(other.gens) == 0:
            return self.ring.ideal(1)
        # 如果 other 的生成元数量为 1
        elif len(other.gens) == 1:
            # 进行一些技巧性操作。设 f 为生成 ``other`` 的向量，
            # f1, .., fn 为生成 self 的向量。
            # 考虑 R^{r+1} 中由 (f, 1) 和 {(fi, 0) | i} 生成的子模。
            # 然后与最后一个模块分量的交集得到商模。
            g1 = list(other.gens[0]) + [1]
            gi = [list(x) + [0] for x in self.gens]
            # 注意：我们需要使用消除顺序 'ilex'
            M = self.ring.free_module(self.rank + 1).submodule(*([g1] + gi),
                                            order='ilex', TOP=False)
            # 如果不需要关系信息，则返回 self.ring.ideal 中的结果
            if not relations:
                return self.ring.ideal(*[x[-1] for x in M._groebner_vec() if
                                         all(y == self.ring.zero for y in x[:-1])])
            else:
                # 否则，返回扩展的 Groebner 基和其余部分的负值的列表
                G, R = M._groebner_vec(extended=True)
                indices = [i for i, x in enumerate(G) if
                           all(y == self.ring.zero for y in x[:-1])]
                return (self.ring.ideal(*[G[i][-1] for i in indices]),
                        [[-x for x in R[i][1:]] for i in indices])
        # 对于更多的生成元，使用 I : <h1, .., hn> = {I : <hi> | i} 的交集
        # TODO：这可以更有效地完成
        return reduce(lambda x, y: x.intersect(y),
            (self._module_quotient(self.container.submodule(x)) for x in other.gens))
class SubModuleQuotientRing(SubModule):
    """
    Class for submodules of free modules over quotient rings.

    Do not instantiate this. Instead use the submodule methods.

    >>> from sympy.abc import x, y  # 导入符号变量 x, y
    >>> from sympy import QQ  # 导入有理数域 QQ
    >>> M = (QQ.old_poly_ring(x, y)/[x**2 - y**2]).free_module(2).submodule([x, x + y])  # 构建子模块 M
    >>> M  # 打印 M
    <[x + <x**2 - y**2>, x + y + <x**2 - y**2>]>
    >>> M.contains([y**2, x**2 + x*y])  # 检查 M 是否包含给定元素
    True
    >>> M.contains([x, y])  # 检查 M 是否包含给定元素
    False

    Attributes:

    - quot - the subquotient of `R^n/IR^n` generated by lifts of our generators
    """

    def __init__(self, gens, container):
        SubModule.__init__(self, gens, container)  # 调用父类构造函数
        self.quot = self.container.quot.submodule(
            *[self.container.lift(x) for x in self.gens])  # 计算并赋值 self.quot，表示模块的商模块

    def _contains(self, elem):
        return self.quot._contains(self.container.lift(elem))  # 检查元素是否属于商模块的包含关系

    def _syzygies(self):
        return [tuple(self.ring.convert(y, self.quot.ring) for y in x)
                for x in self.quot._syzygies()]  # 计算并返回商模块的所有零因子

    def _in_terms_of_generators(self, elem):
        return [self.ring.convert(x, self.quot.ring) for x in
            self.quot._in_terms_of_generators(self.container.lift(elem))]  # 将元素表示为生成元的线性组合


##########################################################################
## Quotient Modules ######################################################
##########################################################################


class QuotientModuleElement(ModuleElement):
    """Element of a quotient module."""

    def eq(self, d1, d2):
        """Equality comparison."""
        return self.module.killed_module.contains(d1 - d2)  # 判断两个元素是否相等

    def __repr__(self):
        return repr(self.data) + " + " + repr(self.module.killed_module)  # 返回元素的字符串表示形式


class QuotientModule(Module):
    """
    Class for quotient modules.

    Do not instantiate this directly. For subquotients, see the
    SubQuotientModule class.

    Attributes:

    - base - the base module we are a quotient of
    - killed_module - the submodule used to form the quotient
    - rank of the base
    """

    dtype = QuotientModuleElement  # 元素类型为 QuotientModuleElement

    def __init__(self, ring, base, submodule):
        Module.__init__(self, ring)  # 调用父类构造函数
        if not base.is_submodule(submodule):  # 检查 submodule 是否为 base 的子模块
            raise ValueError('%s is not a submodule of %s' % (submodule, base))
        self.base = base  # 基础模块
        self.killed_module = submodule  # 用于构造商模块的子模块
        self.rank = base.rank  # 基础模块的秩

    def __repr__(self):
        return repr(self.base) + "/" + repr(self.killed_module)  # 返回模块的字符串表示形式

    def is_zero(self):
        """
        Return True if ``self`` is a zero module.

        This happens if and only if the base module is the same as the
        submodule being killed.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> (F/[(1, 0)]).is_zero()
        False
        >>> (F/[(1, 0), (0, 1)]).is_zero()
        True
        """
        return self.base == self.killed_module  # 检查模块是否为零模块
    def is_submodule(self, other):
        """
        Return True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> Q = QQ.old_poly_ring(x).free_module(2) / [(x, x)]
        >>> S = Q.submodule([1, 0])
        >>> Q.is_submodule(S)
        True
        >>> S.is_submodule(Q)
        False
        """
        # 如果 `other` 是 QuotientModule 类型
        if isinstance(other, QuotientModule):
            # 返回是否满足相同的 killed_module 并且 base 也是 submodule 的条件
            return self.killed_module == other.killed_module and \
                self.base.is_submodule(other.base)
        # 如果 `other` 是 SubQuotientModule 类型
        if isinstance(other, SubQuotientModule):
            # 返回是否满足 other.container == self 的条件
            return other.container == self
        # 其它情况返回 False
        return False

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        This is the same as taking a quotient of a submodule of the base
        module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> Q = QQ.old_poly_ring(x).free_module(2) / [(x, x)]
        >>> Q.submodule([x, 0])
        <[x, 0] + <[x, x]>>
        """
        # 返回 SubQuotientModule 的实例，使用传入的生成器 `gens` 和选项 `opts`
        return SubQuotientModule(gens, self, **opts)

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> F.convert([1, 0])
        [1, 0] + <[1, 2], [1, x]>
        """
        # 如果 `elem` 是 QuotientModuleElement 类型，并且其 module 是 self
        if isinstance(elem, QuotientModuleElement):
            if elem.module is self:
                return elem
            # 如果 self.killed_module 是 elem.module.killed_module 的 submodule
            if self.killed_module.is_submodule(elem.module.killed_module):
                # 返回 QuotientModuleElement 的实例，使用 self.base.convert(elem.data)
                return QuotientModuleElement(self, self.base.convert(elem.data))
            # 如果以上条件都不满足，则抛出 CoercionFailed 异常
            raise CoercionFailed
        # 对于其它类型的 `elem`，返回 QuotientModuleElement 的实例，使用 self.base.convert(elem)
        return QuotientModuleElement(self, self.base.convert(elem))

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> M.identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2/<[1, 2], [1, x]> -> QQ[x]**2/<[1, 2], [1, x]>
        [0, 1]])
        """
        # 返回 self.base.identity_hom().quotient_codomain(self.killed_module).quotient_domain(self.killed_module) 的结果
        return self.base.identity_hom().quotient_codomain(
            self.killed_module).quotient_domain(self.killed_module)
    def quotient_hom(self):
        """
        返回到 ``self`` 的商同态。

        即返回一个同态映射，表示从 ``self.base`` 到 ``self`` 的自然映射。

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2/<[1, 2], [1, x]>
        [0, 1]])
        """
        # 返回由 self.base 的单位同态构成的同态映射，作用对象是 self.killed_module
        return self.base.identity_hom().quotient_codomain(
            self.killed_module)
```