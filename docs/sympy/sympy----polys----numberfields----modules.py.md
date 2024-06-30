# `D:\src\scipysrc\sympy\sympy\polys\numberfields\modules.py`

```
python
r"""Modules in number fields.

The classes defined here allow us to work with finitely generated, free
modules, whose generators are algebraic numbers.

There is an abstract base class called :py:class:`~.Module`, which has two
concrete subclasses, :py:class:`~.PowerBasis` and :py:class:`~.Submodule`.

Every module is defined by its basis, or set of generators:

* For a :py:class:`~.PowerBasis`, the generators are the first $n$ powers
  (starting with the zeroth) of an algebraic integer $\theta$ of degree $n$.
  The :py:class:`~.PowerBasis` is constructed by passing either the minimal
  polynomial of $\theta$, or an :py:class:`~.AlgebraicField` having $\theta$
  as its primitive element.

* For a :py:class:`~.Submodule`, the generators are a set of
  $\mathbb{Q}$-linear combinations of the generators of another module. That
  other module is then the "parent" of the :py:class:`~.Submodule`. The
  coefficients of the $\mathbb{Q}$-linear combinations may be given by an
  integer matrix, and a positive integer denominator. Each column of the matrix
  defines a generator.

>>> from sympy.polys import Poly, cyclotomic_poly, ZZ
>>> from sympy.abc import x
>>> from sympy.polys.matrices import DomainMatrix, DM
>>> from sympy.polys.numberfields.modules import PowerBasis
>>> T = Poly(cyclotomic_poly(5, x))
>>> A = PowerBasis(T)
>>> print(A)
PowerBasis(x**4 + x**3 + x**2 + x + 1)
>>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ), denom=3)
>>> print(B)
Submodule[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]/3
>>> print(B.parent)
PowerBasis(x**4 + x**3 + x**2 + x + 1)

Thus, every module is either a :py:class:`~.PowerBasis`,
or a :py:class:`~.Submodule`, some ancestor of which is a
:py:class:`~.PowerBasis`. (If ``S`` is a :py:class:`~.Submodule`, then its
ancestors are ``S.parent``, ``S.parent.parent``, and so on).

The :py:class:`~.ModuleElement` class represents a linear combination of the
generators of any module. Critically, the coefficients of this linear
combination are not restricted to be integers, but may be any rational
numbers. This is necessary so that any and all algebraic integers be
representable, starting from the power basis in a primitive element $\theta$
for the number field in question. For example, in a quadratic field
$\mathbb{Q}(\sqrt{d})$ where $d \equiv 1 \mod{4}$, a denominator of $2$ is
needed.

A :py:class:`~.ModuleElement` can be constructed from an integer column vector
and a denominator:

>>> U = Poly(x**2 - 5)
>>> M = PowerBasis(U)
>>> e = M(DM([[1], [1]], ZZ), denom=2)
>>> print(e)
[1, 1]/2
>>> print(e.module)
PowerBasis(x**2 - 5)

The :py:class:`~.PowerBasisElement` class is a subclass of
:py:class:`~.ModuleElement` that represents elements of a
:py:class:`~.PowerBasis`, and adds functionality pertinent to elements
represented directly over powers of the primitive element $\theta$.


Arithmetic with module elements
===============================

"""

# Modules in number fields allow manipulation of algebraic number generators.
# This section provides classes like PowerBasis and Submodule for module definition.
# PowerBasis uses powers of an algebraic integer as generators, while Submodule
# allows Q-linear combinations of another module's generators.

# Example usage of PowerBasis and Submodule classes to demonstrate module creation
# and submodule generation. These classes facilitate working with algebraic numbers
# in number fields, enabling representation and manipulation of module elements.

# Detailed examples show how to construct modules and submodules using SymPy's
# Poly and DomainMatrix utilities, ensuring clarity on module creation and usage.

# The ModuleElement class allows representing linear combinations of module
# generators with rational coefficients, crucial for handling algebraic integers
# across various number fields. This flexibility ensures accurate representation
# of elements like e, constructed from a Poly and DomainMatrix, with a specified
# denominator.

# PowerBasisElement extends ModuleElement, offering specialized functionalities
# for elements directly associated with powers of a primitive element theta in
# PowerBasis.

# This section emphasizes the versatility of these classes in arithmetic operations,
# enhancing computational capabilities in number field theory and algebraic
# number theory research and applications.
# 以下是关于模块元素的理论背景和操作说明的文档片段，介绍了如何在给定的模块中执行算术操作。

While a :py:class:`~.ModuleElement` represents a linear combination over the
generators of a particular module, recall that every module is either a
:py:class:`~.PowerBasis` or a descendant (along a chain of
:py:class:`~.Submodule` objects) thereof, so that in fact every
:py:class:`~.ModuleElement` represents an algebraic number in some field
$\mathbb{Q}(\theta)$, where $\theta$ is the defining element of some
:py:class:`~.PowerBasis`. It thus makes sense to talk about the number field
to which a given :py:class:`~.ModuleElement` belongs.

This means that any two :py:class:`~.ModuleElement` instances can be added,
subtracted, multiplied, or divided, provided they belong to the same number
field. Similarly, since $\mathbb{Q}$ is a subfield of every number field,
any :py:class:`~.ModuleElement` may be added, multiplied, etc. by any
rational number.

>>> from sympy import QQ
>>> from sympy.polys.numberfields.modules import to_col
>>> T = Poly(cyclotomic_poly(5))
>>> A = PowerBasis(T)
>>> C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
>>> e = A(to_col([0, 2, 0, 0]), denom=3)
>>> f = A(to_col([0, 0, 0, 7]), denom=5)
>>> g = C(to_col([1, 1, 1, 1]))

# 执行模块元素的加法操作
>>> e + f
[0, 10, 0, 21]/15
# 执行模块元素的减法操作
>>> e - f
[0, 10, 0, -21]/15
# 执行模块元素与子模块元素的减法操作
>>> e - g
[-9, -7, -9, -9]/3
# 执行模块元素与有理数的加法操作
>>> e + QQ(7, 10)
[21, 20, 0, 0]/30
# 执行模块元素的乘法操作
>>> e * f
[-14, -14, -14, -14]/15
# 执行模块元素的幂运算
>>> e ** 2
[0, 0, 4, 0]/9
# 执行模块元素与子模块元素的整除操作
>>> f // g
[7, 7, 7, 7]/15
# 执行模块元素与有理数的乘法操作
>>> f * QQ(2, 3)
[0, 0, 0, 14]/15

However, care must be taken with arithmetic operations on
:py:class:`~.ModuleElement`, because the module $C$ to which the result will
belong will be the nearest common ancestor (NCA) of the modules $A$, $B$ to
which the two operands belong, and $C$ may be different from either or both
of $A$ and $B$.

>>> A = PowerBasis(T)
>>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
>>> C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
>>> print((B(0) * C(0)).module == A)
True

Before the arithmetic operation is performed, copies of the two operands are
automatically converted into elements of the NCA (the operands themselves are
not modified). This upward conversion along an ancestor chain is easy: it just
requires the successive multiplication by the defining matrix of each
:py:class:`~.Submodule`.

Conversely, downward conversion, i.e. representing a given
:py:class:`~.ModuleElement` in a submodule, is also supported -- namely by
the :py:meth:`~sympy.polys.numberfields.modules.Submodule.represent` method
-- but is not guaranteed to succeed in general, since the given element may
not belong to the submodule. The main circumstance in which this issue tends
to arise is with multiplication, since modules, while closed under addition,
need not be closed under multiplication.

Multiplication
--------------

Generally speaking, a module need not be closed under multiplication, i.e. need
not form a ring. However, many of the modules we work with in the context of
number fields are in fact rings, and our classes do support multiplication.
from sympy.core.intfunc import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom



# 导入所需的各种 SymPy 模块和自定义异常类、实用工具
from sympy.core.intfunc import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom



def to_col(coeffs):
    r"""Transform a list of integer coefficients into a column vector."""
    return DomainMatrix([[ZZ(c) for c in coeffs]], (1, len(coeffs)), ZZ).transpose()



# 定义一个函数 to_col(coeffs)，将整数系数的列表转换为列向量
def to_col(coeffs):
    r"""Transform a list of integer coefficients into a column vector."""
    # 使用 DomainMatrix 创建一个列向量，其元素为整数系数 ZZ(c)，其中 c 是 coeffs 中的每个系数
    return DomainMatrix([[ZZ(c) for c in coeffs]], (1, len(coeffs)), ZZ).transpose()



class Module:
    """
    Generic finitely-generated module.

    This is an abstract base class, and should not be instantiated directly.
    The two concrete subclasses are :py:class:`~.PowerBasis` and
    :py:class:`~.Submodule`.

    Every :py:class:`~.Submodule` is derived from another module, referenced
    by its ``parent`` attribute. If ``S`` is a submodule, then we refer to
    ``S.parent``, ``S.parent.parent``, and so on, as the "ancestors" of
    ``S``. Thus, every :py:class:`~.Module` is either a
    :py:class:`~.PowerBasis` or a :py:class:`~.Submodule`, some ancestor of
    which is a :py:class:`~.PowerBasis`.
    """



# 定义一个通用的有限生成模块类 Module
class Module:
    """
    Generic finitely-generated module.

    This is an abstract base class, and should not be instantiated directly.
    The two concrete subclasses are :py:class:`~.PowerBasis` and
    :py:class:`~.Submodule`.

    Every :py:class:`~.Submodule` is derived from another module, referenced
    by its ``parent`` attribute. If ``S`` is a submodule, then we refer to
    ``S.parent``, ``S.parent.parent``, and so on, as the "ancestors" of
    ``S``. Thus, every :py:class:`~.Module` is either a
    :py:class:`~.PowerBasis` or a :py:class:`~.Submodule`, some ancestor of
    which is a :py:class:`~.PowerBasis`.
    """
    """

    @property
    def n(self):
        """The number of generators of this module."""
        # 返回该模块的生成元数量，这是一个属性方法
        raise NotImplementedError

    def mult_tab(self):
        """
        Get the multiplication table for this module (if closed under mult).

        Explanation
        ===========

        Computes a dictionary ``M`` of dictionaries of lists, representing the
        upper triangular half of the multiplication table.

        In other words, if ``0 <= i <= j < self.n``, then ``M[i][j]`` is the
        list ``c`` of coefficients such that
        ``g[i] * g[j] == sum(c[k]*g[k], k in range(self.n))``,
        where ``g`` is the list of generators of this module.

        If ``j < i`` then ``M[i][j]`` is undefined.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> print(A.mult_tab())  # doctest: +SKIP
        {0: {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0],     3: [0, 0, 0, 1]},
                          1: {1: [0, 0, 1, 0], 2: [0, 0, 0, 1],     3: [-1, -1, -1, -1]},
                                           2: {2: [-1, -1, -1, -1], 3: [1, 0, 0, 0]},
                                                                3: {3: [0, 1, 0, 0]}}

        Returns
        =======

        dict of dict of lists

        Raises
        ======

        ClosureFailure
            If the module is not closed under multiplication.

        """
        # 获取该模块的乘法表，如果模块在乘法下封闭
        raise NotImplementedError

    @property
    def parent(self):
        """
        The parent module, if any, for this module.

        Explanation
        ===========

        For a :py:class:`~.Submodule` this is its ``parent`` attribute; for a
        :py:class:`~.PowerBasis` this is ``None``.

        Returns
        =======

        :py:class:`~.Module`, ``None``

        See Also
        ========

        Module

        """
        # 返回此模块的父模块（如果有的话）
        return None

    def ancestors(self, include_self=False):
        """
        Return the list of ancestor modules of this module, from the
        foundational :py:class:`~.PowerBasis` downward, optionally including
        ``self``.

        See Also
        ========

        Module

        """
        # 返回此模块的祖先模块的列表，从基础的 PowerBasis 开始向下，可选地包括 self
        c = self.parent
        a = [] if c is None else c.ancestors(include_self=True)
        if include_self:
            a.append(self)
        return a

    def power_basis_ancestor(self):
        """
        Return the :py:class:`~.PowerBasis` that is an ancestor of this module.

        See Also
        ========

        Module

        """
        # 返回作为该模块祖先的 PowerBasis
        if isinstance(self, PowerBasis):
            return self
        c = self.parent
        if c is not None:
            return c.power_basis_ancestor()
        return None
    # 定义一个方法，用于查找当前模块和另一个模块的最近共同祖先

    Returns
    =======
    
    :py:class:`~.Module`, ``None``
    # 返回一个类型为 :py:class:`~.Module` 或者 ``None`` 的对象
    
    See Also
    ========
    
    Module
    # 参见 `Module` 类

    """
    # 获取包括当前模块在内的所有祖先模块列表
    sA = self.ancestors(include_self=True)
    # 获取包括另一个模块在内的所有祖先模块列表
    oA = other.ancestors(include_self=True)
    # 初始化最近共同祖先变量为 None
    nca = None
    # 遍历两个模块的祖先列表，按顺序比较对应位置的模块
    for sa, oa in zip(sA, oA):
        # 如果找到相同的祖先模块，则更新最近共同祖先变量
        if sa == oa:
            nca = sa
        else:
            # 如果找到第一个不同的祖先模块，则退出循环
            break
    # 返回最近共同祖先模块或者 None（如果没有共同祖先）
    return nca

@property
def number_field(self):
    r"""
    返回与之关联的 :py:class:`~.AlgebraicField`，如果有的话。

    Explanation
    ===========

    一个 :py:class:`~.PowerBasis` 可以构建在一个 :py:class:`~.Poly` $f$ 上，
    或者在一个 :py:class:`~.AlgebraicField` $K$ 上。在后一种情况下，
    :py:class:`~.PowerBasis` 及其所有后代模块的 ``.number_field`` 属性将返回 $K$，
    而在前一种情况下，它们将都返回 ``None``。

    Returns
    =======

    :py:class:`~.AlgebraicField`, ``None``
    # 返回一个类型为 :py:class:`~.AlgebraicField` 或者 ``None`` 的对象

    """
    # 返回当前模块的功用基础的关联数域
    return self.power_basis_ancestor().number_field

def is_compat_col(self, col):
    """判断 *col* 是否适合作为该模块的列向量。"""
    # 返回一个布尔值，判断 *col* 是否为 DomainMatrix 类型，并且形状为 (self.n, 1)，并且其域为整数环（is_ZZ）
    return isinstance(col, DomainMatrix) and col.shape == (self.n, 1) and col.domain.is_ZZ
    def __call__(self, spec, denom=1):
        r"""
        Generate a :py:class:`~.ModuleElement` belonging to this module.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> e = A(to_col([1, 2, 3, 4]), denom=3)
        >>> print(e)  # doctest: +SKIP
        [1, 2, 3, 4]/3
        >>> f = A(2)
        >>> print(f)  # doctest: +SKIP
        [0, 0, 1, 0]

        Parameters
        ==========

        spec : :py:class:`~.DomainMatrix`, int
            Specifies the numerators of the coefficients of the
            :py:class:`~.ModuleElement`. Can be either a column vector over
            :ref:`ZZ`, whose length must equal the number $n$ of generators of
            this module, or else an integer ``j``, $0 \leq j < n$, which is a
            shorthand for column $j$ of $I_n$, the $n \times n$ identity
            matrix.
        denom : int, optional (default=1)
            Denominator for the coefficients of the
            :py:class:`~.ModuleElement`.

        Returns
        =======

        :py:class:`~.ModuleElement`
            The coefficients are the entries of the *spec* vector, divided by
            *denom*.

        """
        # 如果 spec 是整数且在合法范围内，将其转换为单位矩阵的第 spec 列，并转为密集向量
        if isinstance(spec, int) and 0 <= spec < self.n:
            spec = DomainMatrix.eye(self.n, ZZ)[:, spec].to_dense()
        # 检查 spec 是否与当前模块兼容的列向量，否则抛出 ValueError
        if not self.is_compat_col(spec):
            raise ValueError('Compatible column vector required.')
        # 调用 make_mod_elt 函数生成一个模块元素对象，并返回
        return make_mod_elt(self, spec, denom=denom)

    def starts_with_unity(self):
        """Say whether the module's first generator equals unity."""
        # 抛出 NotImplementedError，表示该功能尚未实现
        raise NotImplementedError

    def basis_elements(self):
        """
        Get list of :py:class:`~.ModuleElement` being the generators of this
        module.
        """
        # 返回当前模块的所有基础元素，使用循环生成列表
        return [self(j) for j in range(self.n)]

    def zero(self):
        """Return a :py:class:`~.ModuleElement` representing zero."""
        # 返回一个表示零的模块元素对象
        return self(0) * 0

    def one(self):
        """
        Return a :py:class:`~.ModuleElement` representing unity,
        and belonging to the first ancestor of this module (including
        itself) that starts with unity.
        """
        # 返回一个表示单位元素的模块元素对象，使用 element_from_rational 函数生成
        return self.element_from_rational(1)
    def element_from_rational(self, a):
        """
        Return a :py:class:`~.ModuleElement` representing a rational number.

        Explanation
        ===========

        The returned :py:class:`~.ModuleElement` will belong to the first
        module on this module's ancestor chain (including this module
        itself) that starts with unity.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, QQ
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> a = A.element_from_rational(QQ(2, 3))
        >>> print(a)  # doctest: +SKIP
        [2, 0, 0, 0]/3

        Parameters
        ==========

        a : int, :ref:`ZZ`, :ref:`QQ`

        Returns
        =======

        :py:class:`~.ModuleElement`

        """
        raise NotImplementedError



        """
        Form the submodule generated by a list of :py:class:`~.ModuleElement`
        belonging to this module.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> gens = [A(0), 2*A(1), 3*A(2), 4*A(3)//5]
        >>> B = A.submodule_from_gens(gens)
        >>> print(B)  # doctest: +SKIP
        Submodule[[5, 0, 0, 0], [0, 10, 0, 0], [0, 0, 15, 0], [0, 0, 0, 4]]/5

        Parameters
        ==========

        gens : list of :py:class:`~.ModuleElement` belonging to this module.
        hnf : boolean, optional (default=True)
            If True, we will reduce the matrix into Hermite Normal Form before
            forming the :py:class:`~.Submodule`.
        hnf_modulus : int, None, optional (default=None)
            Modulus for use in the HNF reduction algorithm. See
            :py:func:`~sympy.polys.matrices.normalforms.hermite_normal_form`.

        Returns
        =======

        :py:class:`~.Submodule`

        See Also
        ========

        submodule_from_matrix

        """
        if not all(g.module == self for g in gens):
            raise ValueError('Generators must belong to this module.')
        n = len(gens)
        if n == 0:
            raise ValueError('Need at least one generator.')
        m = gens[0].n
        d = gens[0].denom if n == 1 else ilcm(*[g.denom for g in gens])
        B = DomainMatrix.zeros((m, 0), ZZ).hstack(*[(d // g.denom) * g.col for g in gens])
        if hnf:
            B = hermite_normal_form(B, D=hnf_modulus)
        return self.submodule_from_matrix(B, denom=d)
    def submodule_from_matrix(self, B, denom=1):
        """
        Form the submodule generated by the elements of this module indicated
        by the columns of a matrix, with an optional denominator.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
        >>> from sympy.polys.matrices import DM
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_matrix(DM([
        ...     [0, 10, 0, 0],
        ...     [0,  0, 7, 0],
        ... ], ZZ).transpose(), denom=15)
        >>> print(B)  # doctest: +SKIP
        Submodule[[0, 10, 0, 0], [0, 0, 7, 0]]/15

        Parameters
        ==========

        B : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            Each column gives the numerators of the coefficients of one
            generator of the submodule. Thus, the number of rows of *B* must
            equal the number of generators of the present module.
        denom : int, optional (default=1)
            Common denominator for all generators of the submodule.

        Returns
        =======

        :py:class:`~.Submodule`

        Raises
        ======

        ValueError
            If the given matrix *B* is not over :ref:`ZZ` or its number of rows
            does not equal the number of generators of the present module.

        See Also
        ========

        submodule_from_gens

        """
        # 获取矩阵 B 的行数和列数
        m, n = B.shape
        # 检查矩阵 B 的元素域是否为整数环 ZZ
        if not B.domain.is_ZZ:
            raise ValueError('Matrix must be over ZZ.')
        # 检查矩阵 B 的行数是否与当前模块的生成元数量相等
        if not m == self.n:
            raise ValueError('Matrix row count must match base module.')
        # 返回以当前模块和给定矩阵 B 构建的子模块
        return Submodule(self, B, denom=denom)

    def whole_submodule(self):
        """
        Return a submodule equal to this entire module.

        Explanation
        ===========

        This is useful when you have a :py:class:`~.PowerBasis` and want to
        turn it into a :py:class:`~.Submodule` (in order to use methods
        belonging to the latter).

        """
        # 创建一个单位矩阵 B，其大小为当前模块的生成元数量，元素域为整数环 ZZ
        B = DomainMatrix.eye(self.n, ZZ)
        # 返回以单位矩阵 B 构建的子模块
        return self.submodule_from_matrix(B)

    def endomorphism_ring(self):
        """Form the :py:class:`~.EndomorphismRing` for this module."""
        # 返回当前模块的自同态环
        return EndomorphismRing(self)
class PowerBasis(Module):
    """The module generated by the powers of an algebraic integer."""

    def __init__(self, T):
        """
        Parameters
        ==========

        T : :py:class:`~.Poly`, :py:class:`~.AlgebraicField`
            Either (1) the monic, irreducible, univariate polynomial over
            :ref:`ZZ`, a root of which is the generator of the power basis,
            or (2) an :py:class:`~.AlgebraicField` whose primitive element
            is the generator of the power basis.

        """
        K = None
        if isinstance(T, AlgebraicField):
            K, T = T, T.ext.minpoly_of_element()
        # Sometimes incoming Polys are formally over QQ, although all their
        # coeffs are integral. We want them to be formally over ZZ.
        T = T.set_domain(ZZ)
        self.K = K                   # 存储 AlgebraicField 对象或者 None
        self.T = T                   # 存储给定的多项式或代数域
        self._n = T.degree()         # 存储多项式的次数
        self._mult_tab = None        # 初始化乘法表为 None

    @property
    def number_field(self):
        return self.K                # 返回存储的代数域对象

    def __repr__(self):
        return f'PowerBasis({self.T.as_expr()})'  # 返回 PowerBasis 对象的字符串表示形式

    def __eq__(self, other):
        if isinstance(other, PowerBasis):
            return self.T == other.T  # 检查两个 PowerBasis 对象是否相等
        return NotImplemented

    @property
    def n(self):
        return self._n               # 返回存储的多项式的次数

    def mult_tab(self):
        if self._mult_tab is None:
            self.compute_mult_tab()  # 如果乘法表尚未计算，则进行计算
        return self._mult_tab         # 返回存储的乘法表

    def compute_mult_tab(self):
        theta_pow = AlgIntPowers(self.T)  # 创建 AlgIntPowers 对象
        M = {}
        n = self.n
        for u in range(n):
            M[u] = {}
            for v in range(u, n):
                M[u][v] = theta_pow[u + v]  # 计算乘法表中每个条目的值
        self._mult_tab = M               # 存储计算得到的乘法表

    def represent(self, elt):
        r"""
        Represent a module element as an integer-linear combination over the
        generators of this module.

        See Also
        ========

        .Module.represent
        .Submodule.represent

        """
        if elt.module == self and elt.denom == 1:
            return elt.column()    # 如果元素可以表示为整数线性组合，则返回其列向量
        else:
            raise ClosureFailure('Element not representable in ZZ[theta].')

    def starts_with_unity(self):
        return True                # 始终返回 True，表明以 1 开头

    def element_from_rational(self, a):
        return self(0) * a         # 返回零向量乘以给定有理数的结果

    def element_from_poly(self, f):
        """
        Produce an element of this module, representing *f* after reduction mod
        our defining minimal polynomial.

        Parameters
        ==========

        f : :py:class:`~.Poly` over :ref:`ZZ` in same var as our defining poly.

        Returns
        =======

        :py:class:`~.PowerBasisElement`

        """
        n, k = self.n, f.degree()
        if k >= n:
            f = f % self.T         # 若 f 的次数大于等于多项式的次数，则取 f 对多项式的模
        if f == 0:
            return self.zero()     # 若 f 等于零，则返回模块的零元素
        d, c = dup_clear_denoms(f.rep.to_list(), QQ, convert=True)
        c = list(reversed(c))
        ell = len(c)
        z = [ZZ(0)] * (n - ell)
        col = to_col(c + z)
        return self(col, denom=d)   # 返回用给定多项式 f 的列向量表示的模块元素
    def _element_from_rep_and_mod(self, rep, mod):
        """
        根据给定的代数数生成一个 PowerBasisElement 对象。

        Parameters
        ==========

        rep : list of coeffs
            用原始元素的多项式表示代数数。

        mod : list of coeffs
            原始元素的最小多项式的系数列表。

        Returns
        =======

        :py:class:`~.PowerBasisElement`
            返回表示代数数的 PowerBasisElement 对象。

        """
        # 如果给定的最小多项式与当前对象的最小多项式不同，抛出异常
        if mod != self.T.rep.to_list():
            raise UnificationFailed('Element does not appear to be in the same field.')
        # 使用多项式 rep 和当前对象的生成元创建 PowerBasisElement 对象
        return self.element_from_poly(Poly(rep, self.T.gen))

    def element_from_ANP(self, a):
        """
        将一个代数数域的代数数转换为 PowerBasisElement 对象。
        """
        # 将代数数 a 转换为系数列表，并使用其最小多项式的系数列表调用 _element_from_rep_and_mod 方法
        return self._element_from_rep_and_mod(a.to_list(), a.mod_to_list())

    def element_from_alg_num(self, a):
        """
        将一个 AlgebraicNumber 对象转换为 PowerBasisElement 对象。
        """
        # 将代数数 a 的系数列表和其最小多项式的系数列表传递给 _element_from_rep_and_mod 方法
        return self._element_from_rep_and_mod(a.rep.to_list(), a.minpoly.rep.to_list())
    def __init__(self, parent, matrix, denom=1, mult_tab=None):
        """
        Parameters
        ==========

        parent : :py:class:`~.Module`
            父模块，从中派生出当前子模块。
        matrix : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            矩阵，其列定义了该子模块生成器相对于父模块生成器的线性组合。
        denom : int, optional (default=1)
            矩阵系数的分母。
        mult_tab : dict, ``None``, optional
            如果已知，可以提供此模块的乘法表。

        """
        self._parent = parent  # 设置父模块
        self._matrix = matrix  # 设置矩阵
        self._denom = denom  # 设置分母
        self._mult_tab = mult_tab  # 设置乘法表
        self._n = matrix.shape[1]  # 计算矩阵列数作为模块的维数
        self._QQ_matrix = None  # 初始化 QQ 矩阵为 None
        self._starts_with_unity = None  # 初始化 starts_with_unity 属性为 None
        self._is_sq_maxrank_HNF = None  # 初始化 is_sq_maxrank_HNF 属性为 None

    def __repr__(self):
        r = 'Submodule' + repr(self.matrix.transpose().to_Matrix().tolist())  # 生成子模块的字符串表示，包括转置后的矩阵
        if self.denom > 1:
            r += f'/{self.denom}'  # 如果分母大于1，添加分母信息到字符串表示中
        return r

    def reduced(self):
        """
        生成该子模块的约简版本。

        Explanation
        ===========

        在约简版本中，确保分母为1，并且子模块矩阵的每个元素与分母互质。

        Returns
        =======

        :py:class:`~.Submodule`
        """
        if self.denom == 1:
            return self  # 如果分母已经为1，返回自身
        g = igcd(self.denom, *self.coeffs)  # 计算分母和所有系数的最大公约数
        if g == 1:
            return self  # 如果最大公约数为1，返回自身
        # 返回一个新的子模块对象，其矩阵为原矩阵除以最大公约数并转换为整数，分母为原分母除以最大公约数
        return type(self)(self.parent, (self.matrix / g).convert_to(ZZ), denom=self.denom // g, mult_tab=self._mult_tab)

    def discard_before(self, r):
        """
        根据给定索引 *r*，生成一个丢弃所有位于此索引之前的生成器的新模块。
        """
        W = self.matrix[:, r:]  # 提取从索引 r 开始的子矩阵 W
        s = self.n - r  # 计算新模块的维数
        M = None
        mt = self._mult_tab
        if mt is not None:
            M = {}
            for u in range(s):
                M[u] = {}
                for v in range(u, s):
                    M[u][v] = mt[r + u][r + v][r:]  # 从乘法表中提取相关数据并更新到新的乘法表 M
        # 返回一个新的子模块对象，其矩阵为 W，分母为原分母，乘法表为 M
        return Submodule(self.parent, W, denom=self.denom, mult_tab=M)

    @property
    def n(self):
        return self._n  # 返回模块的维数属性

    def mult_tab(self):
        if self._mult_tab is None:
            self.compute_mult_tab()  # 如果乘法表为空，计算并更新乘法表
        return self._mult_tab  # 返回乘法表

    def compute_mult_tab(self):
        gens = self.basis_element_pullbacks()  # 获取基础元素的逆映射
        M = {}
        n = self.n
        for u in range(n):
            M[u] = {}
            for v in range(u, n):
                M[u][v] = self.represent(gens[u] * gens[v]).flat()  # 计算并更新乘法表 M
        self._mult_tab = M  # 将计算得到的乘法表保存到对象属性中

    @property
    def parent(self):
        return self._parent  # 返回父模块属性

    @property
    def matrix(self):
        return self._matrix  # 返回矩阵属性
    # 返回一个扁平化后的矩阵的迭代器，即将矩阵展平成一维数组
    def coeffs(self):
        return self.matrix.flat()

    # 返回属性 _denom 的值，表示当前对象的分母
    @property
    def denom(self):
        return self._denom

    # 返回属性 _QQ_matrix 的值，即 QQ 域上的矩阵表示
    @property
    def QQ_matrix(self):
        """
        :py:class:`~.DomainMatrix` over :ref:`QQ`, equal to
        ``self.matrix / self.denom``, and guaranteed to be dense.

        Explanation
        ===========

        Depending on how it is formed, a :py:class:`~.DomainMatrix` may have
        an internal representation that is sparse or dense. We guarantee a
        dense representation here, so that tests for equivalence of submodules
        always come out as expected.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
        >>> from sympy.abc import x
        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5, x))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_matrix(3*DomainMatrix.eye(4, ZZ), denom=6)
        >>> C = A.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=2)
        >>> print(B.QQ_matrix == C.QQ_matrix)
        True

        Returns
        =======

        :py:class:`~.DomainMatrix` over :ref:`QQ`

        """
        # 如果 _QQ_matrix 尚未计算，则将 self.matrix / self.denom 转换为稠密表示
        if self._QQ_matrix is None:
            self._QQ_matrix = (self.matrix / self.denom).to_dense()
        return self._QQ_matrix

    # 返回属性 _starts_with_unity 的值，表示子模块是否以单位元素开头
    def starts_with_unity(self):
        if self._starts_with_unity is None:
            self._starts_with_unity = self(0).equiv(1)
        return self._starts_with_unity

    # 返回属性 _is_sq_maxrank_HNF 的值，表示当前矩阵是否为最大秩 Hermite 正则形式
    def is_sq_maxrank_HNF(self):
        if self._is_sq_maxrank_HNF is None:
            self._is_sq_maxrank_HNF = is_sq_maxrank_HNF(self._matrix)
        return self._is_sq_maxrank_HNF

    # 返回当前对象是否是幂基子模块的实例
    def is_power_basis_submodule(self):
        return isinstance(self.parent, PowerBasis)

    # 根据有理数 a 返回子模块的元素
    def element_from_rational(self, a):
        # 如果子模块以单位元素开头，则返回以 0 为参数的值乘以 a
        if self.starts_with_unity():
            return self(0) * a
        else:
            # 否则，返回父模块中的以 a 为参数的元素
            return self.parent.element_from_rational(a)

    # 返回该子模块基础元素的父模块表示列表
    def basis_element_pullbacks(self):
        """
        Return list of this submodule's basis elements as elements of the
        submodule's parent module.
        """
        # 返回每个基础元素在其父模块中的表示形式列表
        return [e.to_parent() for e in self.basis_elements()]
    def represent(self, elt):
        """
        Represent a module element as an integer-linear combination over the
        generators of this module.

        See Also
        ========

        .Module.represent
        .PowerBasis.represent

        """
        # 如果给定元素属于当前模块，则直接调用其 column 方法返回结果
        if elt.module == self:
            return elt.column()
        # 如果给定元素属于当前模块的父模块，尝试使用 QQ 域解决线性组合问题
        elif elt.module == self.parent:
            try:
                # 使用 QQ 矩阵和 QQ 列向量解决线性组合问题，并转换为 ZZ 类型
                A = self.QQ_matrix
                b = elt.QQ_col
                x = A._solve(b)[0].transpose()
                x = x.convert_to(ZZ)
            except DMBadInputError:
                # 如果无法在 QQ 跨度内解决，则抛出异常
                raise ClosureFailure('Element outside QQ-span of this basis.')
            except CoercionFailed:
                # 如果在 QQ 跨度内但不在 ZZ 跨度内，则抛出异常
                raise ClosureFailure('Element in QQ-span but not ZZ-span of this basis.')
            # 返回解决的线性组合结果
            return x
        # 如果当前模块的父对象是子模块类型，则递归调用父模块的 represent 方法
        elif isinstance(self.parent, Submodule):
            coeffs_in_parent = self.parent.represent(elt)
            parent_element = self.parent(coeffs_in_parent)
            return self.represent(parent_element)
        # 如果元素不在当前模块或其父模块链中，则抛出异常
        else:
            raise ClosureFailure('Element outside ancestor chain of this module.')

    def is_compat_submodule(self, other):
        # 判断是否是兼容的子模块，条件是其他对象是子模块并且父对象相同
        return isinstance(other, Submodule) and other.parent == self.parent

    def __eq__(self, other):
        # 自定义等于操作符，如果是兼容的子模块，则比较 QQ 矩阵是否相等
        if self.is_compat_submodule(other):
            return other.QQ_matrix == self.QQ_matrix
        return NotImplemented

    def add(self, other, hnf=True, hnf_modulus=None):
        """
        Add this :py:class:`~.Submodule` to another.

        Explanation
        ===========

        This represents the module generated by the union of the two modules'
        sets of generators.

        Parameters
        ==========

        other : :py:class:`~.Submodule`
        hnf : boolean, optional (default=True)
            If ``True``, reduce the matrix of the combined module to its
            Hermite Normal Form.
        hnf_modulus : :ref:`ZZ`, None, optional
            If a positive integer is provided, use this as modulus in the
            HNF reduction. See
            :py:func:`~sympy.polys.matrices.normalforms.hermite_normal_form`.

        Returns
        =======

        :py:class:`~.Submodule`

        """
        # 计算两个子模块的公倍数，并生成对应的合并矩阵
        d, e = self.denom, other.denom
        m = ilcm(d, e)
        a, b = m // d, m // e
        B = (a * self.matrix).hstack(b * other.matrix)
        # 如果需要进行 HNF（Hermite Normal Form）规范化，则对合并矩阵进行处理
        if hnf:
            B = hermite_normal_form(B, D=hnf_modulus)
        # 返回由合并矩阵生成的新子模块对象
        return self.parent.submodule_from_matrix(B, denom=m)

    def __add__(self, other):
        # 定义加法操作符，如果是兼容的子模块，则调用 add 方法进行操作
        if self.is_compat_submodule(other):
            return self.add(other)
        return NotImplemented

    __radd__ = __add__  # 右加操作符等同于加法操作符
    # 定义一个方法，用于将当前子模块乘以一个有理数、模块元素或另一个子模块。
    def mul(self, other, hnf=True, hnf_modulus=None):
        """
        Multiply this :py:class:`~.Submodule` by a rational number, a
        :py:class:`~.ModuleElement`, or another :py:class:`~.Submodule`.

        Explanation
        ===========

        To multiply by a rational number or :py:class:`~.ModuleElement` means
        to form the submodule whose generators are the products of this
        quantity with all the generators of the present submodule.

        To multiply by another :py:class:`~.Submodule` means to form the
        submodule whose generators are all the products of one generator from
        the one submodule, and one generator from the other.

        Parameters
        ==========

        other : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.ModuleElement`, :py:class:`~.Submodule`
            The object to multiply with the current submodule.
        hnf : boolean, optional (default=True)
            If ``True``, reduce the matrix of the product module to its
            Hermite Normal Form.
        hnf_modulus : :ref:`ZZ`, None, optional
            If provided, use this as modulus in the Hermite Normal Form reduction.

        Returns
        =======

        :py:class:`~.Submodule`
            The resulting submodule after multiplication.
        """
        # 如果 other 是有理数
        if is_rat(other):
            # 将有理数 other 分解为分子和分母
            a, b = get_num_denom(other)
            # 如果分子和分母都为 1，返回当前子模块自身
            if a == b == 1:
                return self
            else:
                # 否则，返回通过乘以有理数得到的新的子模块
                return Submodule(self.parent,
                             self.matrix * a, denom=self.denom * b,
                             mult_tab=None).reduced()
        # 如果 other 是模块元素，并且所属模块与当前子模块的父模块相同
        elif isinstance(other, ModuleElement) and other.module == self.parent:
            # 将当前子模块的基向量按照 other 进行乘法操作，生成新的基向量列表
            gens = [other * e for e in self.basis_element_pullbacks()]
            # 返回父模块中由这些新基向量生成的子模块
            return self.parent.submodule_from_gens(gens, hnf=hnf, hnf_modulus=hnf_modulus)
        # 如果当前子模块与 other 是兼容的子模块
        elif self.is_compat_submodule(other):
            # 获取当前子模块和 other 的基向量，分别进行乘法操作，生成新的基向量列表
            alphas, betas = self.basis_element_pullbacks(), other.basis_element_pullbacks()
            gens = [a * b for a in alphas for b in betas]
            # 返回父模块中由这些新基向量生成的子模块
            return self.parent.submodule_from_gens(gens, hnf=hnf, hnf_modulus=hnf_modulus)
        # 如果不满足以上条件，返回未实现对象
        return NotImplemented

    # 定义 __mul__ 方法，使其与 mul 方法等效
    def __mul__(self, other):
        return self.mul(other)

    # 定义 __rmul__ 方法，使其与 __mul__ 方法等效
    __rmul__ = __mul__

    # 定义一个私有方法，直接返回当前对象自身
    def _first_power(self):
        return self
    def reduce_element(self, elt):
        r"""
        If this submodule $B$ has defining matrix $W$ in square, maximal-rank
        Hermite normal form, then, given an element $x$ of the parent module
        $A$, we produce an element $y \in A$ such that $x - y \in B$, and the
        $i$th coordinate of $y$ satisfies $0 \leq y_i < w_{i,i}$. This
        representative $y$ is unique, in the sense that every element of
        the coset $x + B$ reduces to it under this procedure.

        Explanation
        ===========

        In the special case where $A$ is a power basis for a number field $K$,
        and $B$ is a submodule representing an ideal $I$, this operation
        represents one of a few important ways of reducing an element of $K$
        modulo $I$ to obtain a "small" representative. See [Cohen00]_ Section
        1.4.3.

        Examples
        ========

        >>> from sympy import QQ, Poly, symbols
        >>> t = symbols('t')
        >>> k = QQ.alg_field_from_poly(Poly(t**3 + t**2 - 2*t + 8))
        >>> Zk = k.maximal_order()
        >>> A = Zk.parent
        >>> B = (A(2) - 3*A(0))*Zk
        >>> B.reduce_element(A(2))
        [3, 0, 0]

        Parameters
        ==========

        elt : :py:class:`~.ModuleElement`
            An element of this submodule's parent module.

        Returns
        =======

        elt : :py:class:`~.ModuleElement`
            An element of this submodule's parent module.

        Raises
        ======

        NotImplementedError
            If the given :py:class:`~.ModuleElement` does not belong to this
            submodule's parent module.
        StructureError
            If this submodule's defining matrix is not in square, maximal-rank
            Hermite normal form.

        References
        ==========

        .. [Cohen00] Cohen, H. *Advanced Topics in Computational Number
           Theory.*

        """
        # 检查给定元素是否属于当前子模块的父模块
        if not elt.module == self.parent:
            raise NotImplementedError
        # 检查当前子模块的定义矩阵是否为方阵且具有最大秩的 Hermite 正规形式
        if not self.is_sq_maxrank_HNF():
            msg = "Reduction not implemented unless matrix square max-rank HNF"
            raise StructureError(msg)
        # 获取基元素的回拉
        B = self.basis_element_pullbacks()
        # 初始化元素 a 为给定的 elt
        a = elt
        # 从最后一个坐标开始向前遍历
        for i in range(self.n - 1, -1, -1):
            b = B[i]
            # 计算商 q
            q = a.coeffs[i]*b.denom // (b.coeffs[i]*a.denom)
            # 更新 a 为 a - q * b
            a -= q*b
        # 返回结果元素 a
        return a
def is_sq_maxrank_HNF(dm):
    r"""
    Say whether a :py:class:`~.DomainMatrix` is in that special case of Hermite
    Normal Form, in which the matrix is also square and of maximal rank.

    Explanation
    ===========

    We commonly work with :py:class:`~.Submodule` instances whose matrix is
    in this form, and it can be useful to be able to check that this condition is
    satisfied.

    For example this is the case with the :py:class:`~.Submodule` ``ZK``
    returned by :py:func:`~sympy.polys.numberfields.basis.round_two`, which
    represents the maximal order in a number field, and with ideals formed
    therefrom, such as ``2 * ZK``.

    """
    # 检查域是否为整数环、矩阵是否为方阵、以及矩阵是否为上三角矩阵
    if dm.domain.is_ZZ and dm.is_square and dm.is_upper:
        n = dm.shape[0]
        # 遍历矩阵的每一行
        for i in range(n):
            d = dm[i, i].element
            # 检查主对角元素是否大于0
            if d <= 0:
                return False
            # 检查严格上三角部分的元素是否符合条件
            for j in range(i + 1, n):
                if not (0 <= dm[i, j].element < d):
                    return False
        # 若所有条件满足，则返回True
        return True
    # 若不满足条件，则返回False
    return False


def make_mod_elt(module, col, denom=1):
    r"""
    Factory function which builds a :py:class:`~.ModuleElement`, but ensures
    that it is a :py:class:`~.PowerBasisElement` if the module is a
    :py:class:`~.PowerBasis`.
    """
    # 如果模块是PowerBasis类型，则创建PowerBasisElement对象
    if isinstance(module, PowerBasis):
        return PowerBasisElement(module, col, denom=denom)
    else:
        # 否则创建ModuleElement对象
        return ModuleElement(module, col, denom=denom)


class ModuleElement(IntegerPowerable):
    r"""
    Represents an element of a :py:class:`~.Module`.

    NOTE: Should not be constructed directly. Use the
    :py:meth:`~.Module.__call__` method or the :py:func:`make_mod_elt()`
    factory function instead.
    """

    def __init__(self, module, col, denom=1):
        """
        Parameters
        ==========

        module : :py:class:`~.Module`
            The module to which this element belongs.
        col : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            Column vector giving the numerators of the coefficients of this
            element.
        denom : int, optional (default=1)
            Denominator for the coefficients of this element.

        """
        self.module = module
        self.col = col
        self.denom = denom
        self._QQ_col = None

    def __repr__(self):
        r = str([int(c) for c in self.col.flat()])
        if self.denom > 1:
            r += f'/{self.denom}'
        return r

    def reduced(self):
        """
        Produce a reduced version of this ModuleElement, i.e. one in which the
        gcd of the denominator together with all numerator coefficients is 1.
        """
        # 如果分母为1，则返回自身
        if self.denom == 1:
            return self
        # 计算分母与所有分子系数的最大公约数
        g = igcd(self.denom, *self.coeffs)
        # 若最大公约数为1，则返回自身
        if g == 1:
            return self
        # 否则返回一个新的ModuleElement对象，分子系数除以最大公约数并转换为整数，分母除以最大公约数
        return type(self)(self.module,
                            (self.col / g).convert_to(ZZ),
                            denom=self.denom // g)
    def reduced_mod_p(self, p):
        """
        Produce a version of this :py:class:`~.ModuleElement` in which all
        numerator coefficients have been reduced mod *p*.
        """
        # 返回一个新的 :py:class:`~.ModuleElement` 对象，其分子系数在模 *p* 下进行了约简
        return make_mod_elt(self.module,
                            self.col.convert_to(FF(p)).convert_to(ZZ),
                            denom=self.denom)

    @classmethod
    def from_int_list(cls, module, coeffs, denom=1):
        """
        Make a :py:class:`~.ModuleElement` from a list of ints (instead of a
        column vector).
        """
        # 从整数列表创建一个 :py:class:`~.ModuleElement` 对象，而不是从列向量
        col = to_col(coeffs)
        return cls(module, col, denom=denom)

    @property
    def n(self):
        """The length of this element's column."""
        # 返回该元素列的长度
        return self.module.n

    def __len__(self):
        # 返回该元素列的长度，与 self.n 属性相同
        return self.n

    def column(self, domain=None):
        """
        Get a copy of this element's column, optionally converting to a domain.
        """
        if domain is None:
            # 返回该元素列的副本
            return self.col.copy()
        else:
            # 返回转换为指定域的该元素列的副本
            return self.col.convert_to(domain)

    @property
    def coeffs(self):
        # 返回该元素列的扁平化系数列表
        return self.col.flat()

    @property
    def QQ_col(self):
        """
        :py:class:`~.DomainMatrix` over :ref:`QQ`, equal to
        ``self.col / self.denom``, and guaranteed to be dense.

        See Also
        ========

        .Submodule.QQ_matrix

        """
        if self._QQ_col is None:
            # 如果尚未计算过，计算并返回 self.col / self.denom，并确保结果是密集的
            self._QQ_col = (self.col / self.denom).to_dense()
        return self._QQ_col

    def to_parent(self):
        """
        Transform into a :py:class:`~.ModuleElement` belonging to the parent of
        this element's module.
        """
        if not isinstance(self.module, Submodule):
            # 如果当前模块不是子模块的一个元素，则抛出错误
            raise ValueError('Not an element of a Submodule.')
        # 返回一个新的 :py:class:`~.ModuleElement` 对象，属于当前模块的父模块
        return make_mod_elt(
            self.module.parent, self.module.matrix * self.col,
            denom=self.module.denom * self.denom)

    def to_ancestor(self, anc):
        """
        Transform into a :py:class:`~.ModuleElement` belonging to a given
        ancestor of this element's module.

        Parameters
        ==========

        anc : :py:class:`~.Module`

        """
        if anc == self.module:
            # 如果给定的祖先与当前模块相同，直接返回当前对象
            return self
        else:
            # 否则，将当前对象转换为其父模块，并继续转换为给定祖先的元素
            return self.to_parent().to_ancestor(anc)

    def over_power_basis(self):
        """
        Transform into a :py:class:`~.PowerBasisElement` over our
        :py:class:`~.PowerBasis` ancestor.
        """
        # 将当前对象转换为在 :py:class:`~.PowerBasis` 祖先上的 :py:class:`~.PowerBasisElement`
        e = self
        while not isinstance(e.module, PowerBasis):
            e = e.to_parent()
        return e

    def is_compat(self, other):
        """
        Test whether other is another :py:class:`~.ModuleElement` with same
        module.
        """
        # 检查另一个对象是否是与当前对象同一模块的另一个 :py:class:`~.ModuleElement`
        return isinstance(other, ModuleElement) and other.module == self.module
    def unify(self, other):
        """
        Try to make a compatible pair of :py:class:`~.ModuleElement`, one
        equivalent to this one, and one equivalent to the other.

        Explanation
        ===========

        We search for the nearest common ancestor module for the pair of
        elements, and represent each one there.

        Returns
        =======

        Pair ``(e1, e2)``
            Each ``ei`` is a :py:class:`~.ModuleElement`, they belong to the
            same :py:class:`~.Module`, ``e1`` is equivalent to ``self``, and
            ``e2`` is equivalent to ``other``.

        Raises
        ======

        UnificationFailed
            If ``self`` and ``other`` have no common ancestor module.

        """
        # Check if both elements belong to the same module
        if self.module == other.module:
            return self, other
        # Find the nearest common ancestor module
        nca = self.module.nearest_common_ancestor(other.module)
        if nca is not None:
            # Return each element converted to the ancestor module
            return self.to_ancestor(nca), other.to_ancestor(nca)
        # Raise an exception if no common ancestor module is found
        raise UnificationFailed(f"Cannot unify {self} with {other}")

    def __eq__(self, other):
        # Check if `self` is equivalent to `other` under compatibility rules
        if self.is_compat(other):
            return self.QQ_col == other.QQ_col
        # Return NotImplemented if `self` is not comparable to `other`
        return NotImplemented

    def equiv(self, other):
        """
        A :py:class:`~.ModuleElement` may test as equivalent to a rational
        number or another :py:class:`~.ModuleElement`, if they represent the
        same algebraic number.

        Explanation
        ===========

        This method is intended to check equivalence only in those cases in
        which it is easy to test; namely, when *other* is either a
        :py:class:`~.ModuleElement` that can be unified with this one (i.e. one
        which shares a common :py:class:`~.PowerBasis` ancestor), or else a
        rational number (which is easy because every :py:class:`~.PowerBasis`
        represents every rational number).

        Parameters
        ==========

        other : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.ModuleElement`

        Returns
        =======

        bool

        Raises
        ======

        UnificationFailed
            If ``self`` and ``other`` do not share a common
            :py:class:`~.PowerBasis` ancestor.

        """
        # Check if `self` is equal to `other` directly
        if self == other:
            return True
        # Check if `other` is an instance of ModuleElement and unify
        elif isinstance(other, ModuleElement):
            a, b = self.unify(other)
            return a == b
        # Check if `other` is a rational number and handle accordingly
        elif is_rat(other):
            if isinstance(self, PowerBasisElement):
                return self == self.module(0) * other
            else:
                return self.over_power_basis().equiv(other)
        # If none of the above cases match, return False
        return False
    # 定义特殊方法 __add__，用于实现模块元素与有理数或另一个模块元素的加法操作
    def __add__(self, other):
        """
        A :py:class:`~.ModuleElement` can be added to a rational number, or to
        another :py:class:`~.ModuleElement`.

        Explanation
        ===========

        When the other summand is a rational number, it will be converted into
        a :py:class:`~.ModuleElement` (belonging to the first ancestor of this
        module that starts with unity).

        In all cases, the sum belongs to the nearest common ancestor (NCA) of
        the modules of the two summands. If the NCA does not exist, we return
        ``NotImplemented``.
        """
        # 如果 self 与 other 兼容（共享相同的模块），执行以下操作
        if self.is_compat(other):
            # 取得 self 和 other 的分母
            d, e = self.denom, other.denom
            # 计算最小公倍数
            m = ilcm(d, e)
            # 计算调整后的系数列
            col = to_col([u * a + v * b for a, b in zip(self.coeffs, other.coeffs)])
            # 返回一个新的 ModuleElement 对象，该对象包含调整后的系数列和最小公倍数作为分母
            return type(self)(self.module, col, denom=m).reduced()
        # 如果 other 是 ModuleElement 类型，尝试统一两者
        elif isinstance(other, ModuleElement):
            try:
                a, b = self.unify(other)
            except UnificationFailed:
                # 如果统一失败，返回 NotImplemented
                return NotImplemented
            # 递归调用 __add__ 方法
            return a + b
        # 如果 other 是有理数，将其转换为 ModuleElement 后再与 self 相加
        elif is_rat(other):
            return self + self.module.element_from_rational(other)
        # 如果无法处理其他类型，返回 NotImplemented
        return NotImplemented

    # 右加法的特殊方法等同于 __add__ 方法
    __radd__ = __add__

    # 定义特殊方法 __neg__，实现模块元素的取负操作
    def __neg__(self):
        return self * -1

    # 定义特殊方法 __sub__，实现模块元素与另一个元素的减法操作
    def __sub__(self, other):
        return self + (-other)

    # 右减法的特殊方法
    def __rsub__(self, other):
        return -self + other
    def __mul__(self, other):
        """
        A :py:class:`~.ModuleElement` can be multiplied by a rational number,
        or by another :py:class:`~.ModuleElement`.

        Explanation
        ===========

        When the multiplier is a rational number, the product is computed by
        operating directly on the coefficients of this
        :py:class:`~.ModuleElement`.

        When the multiplier is another :py:class:`~.ModuleElement`, the product
        will belong to the nearest common ancestor (NCA) of the modules of the
        two operands, and that NCA must have a multiplication table. If the NCA
        does not exist, we return ``NotImplemented``. If the NCA does not have
        a mult. table, ``ClosureFailure`` will be raised.
        """
        # Check if `other` is compatible for multiplication with `self`
        if self.is_compat(other):
            # Get multiplication table of the module
            M = self.module.mult_tab()
            # Flatten coefficients of self and other
            A, B = self.col.flat(), other.col.flat()
            n = self.n
            C = [0] * n
            # Iterate over pairs (u, v) in the module
            for u in range(n):
                for v in range(u, n):
                    c = A[u] * B[v]
                    if v > u:
                        c += A[v] * B[u]
                    # Multiply by corresponding entry in the multiplication table
                    if c != 0:
                        R = M[u][v]
                        for k in range(n):
                            C[k] += c * R[k]
            # Calculate the denominator of the resulting element
            d = self.denom * other.denom
            # Return the resulting module element constructed from C and denom d
            return self.from_int_list(self.module, C, denom=d)
        elif isinstance(other, ModuleElement):
            try:
                # Attempt to unify self and other to a common form
                a, b = self.unify(other)
            except UnificationFailed:
                # Return NotImplemented if unification fails
                return NotImplemented
            # Return the product of a and b
            return a * b
        elif is_rat(other):
            # If other is a rational number, decompose it into numerator a and denominator b
            a, b = get_num_denom(other)
            if a == b == 1:
                return self
            else:
                # Return the product of self and a, adjusted by denom b, and reduced
                return make_mod_elt(self.module, self.col * a, denom=self.denom * b).reduced()
        # Return NotImplemented for unsupported types
        return NotImplemented

    __rmul__ = __mul__  # Right multiplication is the same as left multiplication

    def _zeroth_power(self):
        # Return the zeroth power of the module element, which is 1
        return self.module.one()

    def _first_power(self):
        # Return the first power of the module element, which is itself
        return self

    def __floordiv__(self, a):
        # Floor division of self by a
        if is_rat(a):
            # Convert a to a rational number if it's not already
            a = QQ(a)
            # Return the result of self multiplied by the reciprocal of a
            return self * (1/a)
        elif isinstance(a, ModuleElement):
            # Return the result of self multiplied by the reciprocal of a
            return self * (1//a)
        # Return NotImplemented for unsupported types
        return NotImplemented

    def __rfloordiv__(self, a):
        # Right floor division of a by self
        return a // self.over_power_basis()


These annotations explain each method's purpose and the logic behind the operations performed within the methods, following the guidelines provided.
    # 定义一个特殊方法 `__mod__`，用于计算 `ModuleElement` 对象对一个 `Submodule` 的模运算
    r"""
    对此 :py:class:`~.ModuleElement` 对象执行模运算，以模 `:py:class:`~.Submodule`.

    参数
    ==========

    m : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.Submodule`
        如果是 :py:class:`~.Submodule`，则对 `self` 执行相对于此的模运算。
        如果是整数或有理数，则相对于我们自己模乘以此常数的 :py:class:`~.Submodule` 进行模运算。

    参见
    ========

    .Submodule.reduce_element

    """
    # 如果 m 是有理数，则将其乘以 `self.module.whole_submodule()` 的整体子模
    if is_rat(m):
        m = m * self.module.whole_submodule()
    # 如果 m 是 `Submodule` 类型，并且其父对象是 `self.module`
    if isinstance(m, Submodule) and m.parent == self.module:
        # 返回对 `m` 调用 `reduce_element` 方法后的结果
        return m.reduce_element(self)
    # 否则返回未实现的操作
    return NotImplemented
    class PowerBasisElement(ModuleElement):
        r"""
        Subclass for :py:class:`~.ModuleElement` instances whose module is a
        :py:class:`~.PowerBasis`.
        """

        @property
        def T(self):
            """Access the defining polynomial of the :py:class:`~.PowerBasis`."""
            return self.module.T
        # 返回与此对象相关的 :py:class:`~.PowerBasis` 的定义多项式

        def numerator(self, x=None):
            """Obtain the numerator as a polynomial over :ref:`ZZ`."""
            x = x or self.T.gen
            return Poly(reversed(self.coeffs), x, domain=ZZ)
        # 返回分子作为 :ref:`ZZ` 上的多项式

        def poly(self, x=None):
            """Obtain the number as a polynomial over :ref:`QQ`."""
            return self.numerator(x=x) // self.denom
        # 返回作为 :ref:`QQ` 上的多项式的数值

        @property
        def is_rational(self):
            """Say whether this element represents a rational number."""
            return self.col[1:, :].is_zero_matrix
        # 判断此元素是否表示有理数

        @property
        def generator(self):
            """
            Return a :py:class:`~.Symbol` to be used when expressing this element
            as a polynomial.

            If we have an associated :py:class:`~.AlgebraicField` whose primitive
            element has an alias symbol, we use that. Otherwise we use the variable
            of the minimal polynomial defining the power basis to which we belong.
            """
            K = self.module.number_field
            return K.ext.alias if K and K.ext.is_aliased else self.T.gen
        # 返回一个 :py:class:`~.Symbol` 用于表示此元素作为多项式时使用的变量

        def as_expr(self, x=None):
            """Create a Basic expression from ``self``. """
            return self.poly(x or self.generator).as_expr()
        # 从当前对象创建一个基本表达式

        def norm(self, T=None):
            """Compute the norm of this number."""
            T = T or self.T
            x = T.gen
            A = self.numerator(x=x)
            return T.resultant(A) // self.denom ** self.n
        # 计算此数的范数

        def inverse(self):
            f = self.poly()
            f_inv = f.invert(self.T)
            return self.module.element_from_poly(f_inv)
        # 返回此数的逆元素

        def __rfloordiv__(self, a):
            return self.inverse() * a
        # 右除运算符的重载，实现 a // self 的功能

        def _negative_power(self, e, modulo=None):
            return self.inverse() ** abs(e)
        # 返回此数的负幂次方，如果指定了模数，则进行模运算

        def to_ANP(self):
            """Convert to an equivalent :py:class:`~.ANP`. """
            return ANP(list(reversed(self.QQ_col.flat())), QQ.map(self.T.rep.to_list()), QQ)
        # 转换为等效的 :py:class:`~.ANP`
    def to_alg_num(self):
        """
        Try to convert to an equivalent :py:class:`~.AlgebraicNumber`.

        Explanation
        ===========

        In general, the conversion from an :py:class:`~.AlgebraicNumber` to a
        :py:class:`~.PowerBasisElement` throws away information, because an
        :py:class:`~.AlgebraicNumber` specifies a complex embedding, while a
        :py:class:`~.PowerBasisElement` does not. However, in some cases it is
        possible to convert a :py:class:`~.PowerBasisElement` back into an
        :py:class:`~.AlgebraicNumber`, namely when the associated
        :py:class:`~.PowerBasis` has a reference to an
        :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicNumber`

        Raises
        ======

        StructureError
            If the :py:class:`~.PowerBasis` to which this element belongs does
            not have an associated :py:class:`~.AlgebraicField`.

        """
        # 获取模块的数域对象 K
        K = self.module.number_field
        # 如果 K 存在
        if K:
            # 调用 K 的 to_alg_num 方法，传入当前元素的 to_ANP 方法的返回值，并返回结果
            return K.to_alg_num(self.to_ANP())
        # 如果 K 不存在，抛出 StructureError 异常
        raise StructureError("No associated AlgebraicField")
class ModuleHomomorphism:
    r"""A homomorphism from one module to another."""

    def __init__(self, domain, codomain, mapping):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain of the mapping.

        codomain : :py:class:`~.Module`
            The codomain of the mapping.

        mapping : callable
            An arbitrary callable is accepted, but should be chosen so as
            to represent an actual module homomorphism. In particular, should
            accept elements of *domain* and return elements of *codomain*.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, ModuleHomomorphism
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_gens([2*A(j) for j in range(4)])
        >>> phi = ModuleHomomorphism(A, B, lambda x: 6*x)
        >>> print(phi.matrix())  # doctest: +SKIP
        DomainMatrix([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]], (4, 4), ZZ)

        """
        # 初始化函数，设置域、陪域和映射
        self.domain = domain
        self.codomain = codomain
        self.mapping = mapping

    def matrix(self, modulus=None):
        r"""
        Compute the matrix of this homomorphism.

        Parameters
        ==========

        modulus : int, optional
            A positive prime number $p$ if the matrix should be reduced mod
            $p$.

        Returns
        =======

        :py:class:`~.DomainMatrix`
            The matrix is over :ref:`ZZ`, or else over :ref:`GF(p)` if a
            modulus was given.

        """
        # 获取域的基础元素
        basis = self.domain.basis_elements()
        # 对每个基础元素应用映射并表示在陪域中的结果
        cols = [self.codomain.represent(self.mapping(elt)) for elt in basis]
        # 若没有列，则返回零矩阵
        if not cols:
            return DomainMatrix.zeros((self.codomain.n, 0), ZZ).to_dense()
        # 按列堆叠所有表示结果，得到矩阵 M
        M = cols[0].hstack(*cols[1:])
        # 如果指定了模数，则将矩阵转换为有限域上的矩阵
        if modulus:
            M = M.convert_to(FF(modulus))
        return M
    # 计算表示这个同态核的子模块。

    Parameters
    ==========

    modulus : int, optional
        如果应该计算模 $p$ 下的核，则为正的素数 $p$。

    Returns
    =======

    :py:class:`~.Submodule`
        该子模块的生成元跨越了此同态的核，可以是整数环 :ref:`ZZ` 或者给定模数 :ref:`GF(p)` 下的核。

    """
    # 获取表示这个同态的矩阵
    M = self.matrix(modulus=modulus)
    if modulus is None:
        # 如果未指定模数，则将矩阵转换为有理数域 QQ 上的矩阵
        M = M.convert_to(QQ)

    # 注意：即使在有限域上工作，我们这里需要的是其对应到整数环上的拉回，
    # 因此在这种情况下，下面的转换为 ZZ 是适当的。在整数环上工作时，
    # 核应该是一个 ZZ-子模块，因此，虽然上面的转换到 QQ 是为了使零空间计算起作用，
    # 之后转换回 ZZ 应该总是有效的。
    
    # 计算矩阵的零空间并转换为整数环 ZZ 上的矩阵，然后转置
    K = M.nullspace().convert_to(ZZ).transpose()

    # 返回从矩阵 K 构建的子模块
    return self.domain.submodule_from_matrix(K)
class ModuleEndomorphism(ModuleHomomorphism):
    r"""A homomorphism from one module to itself."""

    def __init__(self, domain, mapping):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The common domain and codomain of the mapping.

        mapping : callable
            An arbitrary callable is accepted, but should be chosen so as
            to represent an actual module endomorphism. In particular, should
            accept and return elements of *domain*.

        """
        # 调用父类构造函数，初始化一个模块同态
        super().__init__(domain, domain, mapping)


class InnerEndomorphism(ModuleEndomorphism):
    r"""
    An inner endomorphism on a module, i.e. the endomorphism corresponding to
    multiplication by a fixed element.
    """

    def __init__(self, domain, multiplier):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain and codomain of the endomorphism.

        multiplier : :py:class:`~.ModuleElement`
            The element $a$ defining the mapping as $x \mapsto a x$.

        """
        # 调用父类构造函数，使用 lambda 表达式定义内部自同态
        super().__init__(domain, lambda x: multiplier * x)
        # 记录乘数（multiplier）作为类的属性
        self.multiplier = multiplier


class EndomorphismRing:
    r"""The ring of endomorphisms on a module."""

    def __init__(self, domain):
        """
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain and codomain of the endomorphisms.

        """
        # 初始化自同态环的域
        self.domain = domain

    def inner_endomorphism(self, multiplier):
        r"""
        Form an inner endomorphism belonging to this endomorphism ring.

        Parameters
        ==========

        multiplier : :py:class:`~.ModuleElement`
            Element $a$ defining the inner endomorphism $x \mapsto a x$.

        Returns
        =======

        :py:class:`~.InnerEndomorphism`

        """
        # 返回一个属于该自同态环的内部自同态对象
        return InnerEndomorphism(self.domain, multiplier)

def find_min_poly(alpha, domain, x=None, powers=None):
    r"""
    Find a polynomial of least degree (not necessarily irreducible) satisfied
    by an element of a finitely-generated ring with unity.

    Examples
    ========

    For the $n$th cyclotomic field, $n$ an odd prime, consider the quadratic
    equation whose roots are the two periods of length $(n-1)/2$. Article 356
    of Gauss tells us that we should get $x^2 + x - (n-1)/4$ or
    $x^2 + x + (n+1)/4$ according to whether $n$ is 1 or 3 mod 4, respectively.

    >>> from sympy import Poly, cyclotomic_poly, primitive_root, QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.modules import PowerBasis, find_min_poly
    >>> n = 13
    >>> g = primitive_root(n)
    >>> C = PowerBasis(Poly(cyclotomic_poly(n, x)))
    >>> ee = [g**(2*k+1) % n for k in range((n-1)//2)]
    >>> eta = sum(C(e) for e in ee)
    >>> print(find_min_poly(eta, QQ, x=x).as_expr())
    x**2 + x - 3
    >>> n = 19
    >>> g = primitive_root(n)
    >>> C = PowerBasis(Poly(cyclotomic_poly(n, x)))

    """
    # 在有单位元的有限生成环中，找到满足最小次数多项式的元素
    pass
    # 计算列表 ee 中的每个元素，使用给定的 g、k 和 n 计算模 n 的幂，放入列表中
    ee = [g**(2*k+2) % n for k in range((n-1)//2)]
    # 计算列表 ee 中所有元素的和，通过 C 函数将其转换为特定类型 eta
    eta = sum(C(e) for e in ee)
    # 调用 find_min_poly 函数，找到 eta 的最小多项式，使用 QQ 作为域，变量 x 作为多项式的变量
    print(find_min_poly(eta, QQ, x=x).as_expr())
    
    Parameters
    ==========
    
    alpha : :py:class:`~.ModuleElement`
        需要找到最小多项式的元素，其所属的模具有乘法并以单位元素开头。
    
    domain : :py:class:`~.Domain`
        所需的多项式域。
    
    x : :py:class:`~.Symbol`, optional
        多项式的变量，默认为 x。
    
    powers : list, optional
        如果需要，可以传入一个空列表。将记录 alpha 的幂（作为 :py:class:`~.ModuleElement` 实例），
        从零到最小多项式的阶数。
    
    Returns
    =======
    
    :py:class:`~.Poly`, ``None``
        alpha 的最小多项式，如果在所需的域中找不到多项式，则返回 ``None``。
    
    Raises
    ======
    
    MissingUnityError
        如果 alpha 所属的模具有不是以单位元素开头。
    ClosureFailure
        如果 alpha 所属的模不满足乘法封闭性。
    
    """
    R = alpha.module
    # 检查 alpha 所属的环是否以单位元素开头，如果不是则引发 MissingUnityError 异常
    if not R.starts_with_unity():
        raise MissingUnityError("alpha must belong to finitely generated ring with unity.")
    # 如果 powers 为 None，则初始化为空列表
    if powers is None:
        powers = []
    # 创建 R 的零元素
    one = R(0)
    # 将 R 的零元素添加到 powers 列表中
    powers.append(one)
    # 用 R 的零元素创建一个列向量，域为 domain
    powers_matrix = one.column(domain=domain)
    # 初始化 ak 为 alpha
    ak = alpha
    # 初始化 m 为 None
    m = None
    # 从 k=1 到 k=R.n 的循环
    for k in range(1, R.n + 1):
        # 将 alpha 的当前幂添加到 powers 列表中
        powers.append(ak)
        # 将 alpha 的当前幂作为列向量，域为 domain
        ak_col = ak.column(domain=domain)
        try:
            # 尝试解方程 powers_matrix * X = ak_col，找到 X
            X = powers_matrix._solve(ak_col)[0]
        except DMBadInputError:
            # 如果无法解决方程，说明 alpha^k 仍不在较低幂的域张成中
            powers_matrix = powers_matrix.hstack(ak_col)
            # ak 更新为 ak * alpha
            ak *= alpha
        else:
            # 如果能够解决方程，说明 alpha^k 在较低幂的域张成中，找到了 alpha 的最小度多项式
            coeffs = [1] + [-c for c in reversed(X.to_list_flat())]
            # 如果 x 为 None，则创建一个虚拟变量 x
            x = x or Dummy('x')
            # 根据域的类型创建多项式 m
            if domain.is_FF:
                m = Poly(coeffs, x, modulus=domain.mod)
            else:
                m = Poly(coeffs, x, domain=domain)
            break
    # 返回找到的最小多项式 m
    return m
```