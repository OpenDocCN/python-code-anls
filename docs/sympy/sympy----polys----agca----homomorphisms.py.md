# `D:\src\scipysrc\sympy\sympy\polys\agca\homomorphisms.py`

```
"""
Computations with homomorphisms of modules and rings.

This module implements classes for representing homomorphisms of rings and
their modules. Instead of instantiating the classes directly, you should use
the function ``homomorphism(from, to, matrix)`` to create homomorphism objects.
"""


from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
    SubModule, SubQuotientModule)
from sympy.polys.polyerrors import CoercionFailed

# The main computational task for module homomorphisms is kernels.
# For this reason, the concrete classes are organised by domain module type.


class ModuleHomomorphism:
    """
    Abstract base class for module homomoprhisms. Do not instantiate.

    Instead, use the ``homomorphism`` function:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])

    Attributes:

    - ring - the ring over which we are considering modules
    - domain - the domain module
    - codomain - the codomain module
    - _ker - cached kernel
    - _img - cached image

    Non-implemented methods:

    - _kernel
    - _image
    - _restrict_domain
    - _restrict_codomain
    - _quotient_domain
    - _quotient_codomain
    - _apply
    - _mul_scalar
    - _compose
    - _add
    """

    def __init__(self, domain, codomain):
        if not isinstance(domain, Module):
            raise TypeError('Source must be a module, got %s' % domain)
        if not isinstance(codomain, Module):
            raise TypeError('Target must be a module, got %s' % codomain)
        if domain.ring != codomain.ring:
            raise ValueError('Source and codomain must be over same ring, '
                             'got %s != %s' % (domain, codomain))
        self.domain = domain
        self.codomain = codomain
        self.ring = domain.ring
        self._ker = None
        self._img = None

    def kernel(self):
        r"""
        Compute the kernel of ``self``.

        That is, if ``self`` is the homomorphism `\phi: M \to N`, then compute
        `ker(\phi) = \{x \in M | \phi(x) = 0\}`.  This is a submodule of `M`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> homomorphism(F, F, [[1, 0], [x, 0]]).kernel()
        <[x, -1]>
        """
        # Check if the kernel has already been computed; if not, compute it
        if self._ker is None:
            self._ker = self._kernel()
        return self._ker
    def image(self):
        r"""
        Compute the image of ``self``.

        That is, if ``self`` is the homomorphism `\phi: M \to N`, then compute
        `im(\phi) = \{\phi(x) | x \in M \}`.  This is a submodule of `N`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> homomorphism(F, F, [[1, 0], [x, 0]]).image() == F.submodule([1, 0])
        True
        """
        # 如果尚未计算图像，则调用 _image() 方法计算并缓存
        if self._img is None:
            self._img = self._image()
        return self._img

    def _kernel(self):
        """Compute the kernel of ``self``."""
        # 此方法未实现，应当在子类中进行具体实现
        raise NotImplementedError

    def _image(self):
        """Compute the image of ``self``."""
        # 同样，此方法未实现，应当在子类中进行具体实现
        raise NotImplementedError

    def _restrict_domain(self, sm):
        """Implementation of domain restriction."""
        # 此方法未实现，应当在子类中进行具体实现
        raise NotImplementedError

    def _restrict_codomain(self, sm):
        """Implementation of codomain restriction."""
        # 此方法未实现，应当在子类中进行具体实现
        raise NotImplementedError

    def _quotient_domain(self, sm):
        """Implementation of domain quotient."""
        # 此方法未实现，应当在子类中进行具体实现
        raise NotImplementedError

    def _quotient_codomain(self, sm):
        """Implementation of codomain quotient."""
        # 此方法未实现，应当在子类中进行具体实现
        raise NotImplementedError

    def restrict_domain(self, sm):
        """
        Return ``self``, with the domain restricted to ``sm``.

        Here ``sm`` has to be a submodule of ``self.domain``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.restrict_domain(F.submodule([1, 0]))
        Matrix([
        [1, x], : <[1, 0]> -> QQ[x]**2
        [0, 0]])

        This is the same as just composing on the right with the submodule
        inclusion:

        >>> h * F.submodule([1, 0]).inclusion_hom()
        Matrix([
        [1, x], : <[1, 0]> -> QQ[x]**2
        [0, 0]])
        """
        # 检查 sm 是否是 self.domain 的子模块，如果不是则抛出 ValueError
        if not self.domain.is_submodule(sm):
            raise ValueError('sm must be a submodule of %s, got %s'
                             % (self.domain, sm))
        # 如果 sm 与 self.domain 相同，则直接返回 self
        if sm == self.domain:
            return self
        # 否则调用 _restrict_domain(sm) 方法返回限制后的对象
        return self._restrict_domain(sm)
    def restrict_codomain(self, sm):
        """
        Return ``self``, with codomain restricted to to ``sm``.

        Here ``sm`` has to be a submodule of ``self.codomain`` containing the
        image.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.restrict_codomain(F.submodule([1, 0]))
        Matrix([
        [1, x], : QQ[x]**2 -> <[1, 0]>
        [0, 0]])
        """
        # 检查 sm 是否是 self.image() 的子模块
        if not sm.is_submodule(self.image()):
            raise ValueError('the image %s must contain sm, got %s'
                             % (self.image(), sm))
        # 如果 sm 已经是当前的 codomain，则直接返回 self
        if sm == self.codomain:
            return self
        # 否则调用内部方法 _restrict_codomain 进行实际的限制操作
        return self._restrict_codomain(sm)

    def quotient_domain(self, sm):
        """
        Return ``self`` with domain replaced by ``domain/sm``.

        Here ``sm`` must be a submodule of ``self.kernel()``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.quotient_domain(F.submodule([-x, 1]))
        Matrix([
        [1, x], : QQ[x]**2/<[-x, 1]> -> QQ[x]**2
        [0, 0]])
        """
        # 检查 sm 是否是 self.kernel() 的子模块
        if not self.kernel().is_submodule(sm):
            raise ValueError('kernel %s must contain sm, got %s' %
                             (self.kernel(), sm))
        # 如果 sm 是零模块，则直接返回 self
        if sm.is_zero():
            return self
        # 否则调用内部方法 _quotient_domain 进行实际的商域操作
        return self._quotient_domain(sm)
    def quotient_codomain(self, sm):
        """
        Return ``self`` with codomain replaced by ``codomain/sm``.

        Here ``sm`` must be a submodule of ``self.codomain``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.quotient_codomain(F.submodule([1, 1]))
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2/<[1, 1]>
        [0, 0]])

        This is the same as composing with the quotient map on the left:

        >>> (F/[(1, 1)]).quotient_hom() * h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2/<[1, 1]>
        [0, 0]])
        """
        # 检查 sm 是否是 self.codomain 的子模块，如果不是则抛出 ValueError 异常
        if not self.codomain.is_submodule(sm):
            raise ValueError('sm must be a submodule of codomain %s, got %s'
                             % (self.codomain, sm))
        # 如果 sm 是零模块，则直接返回 self
        if sm.is_zero():
            return self
        # 调用内部方法 _quotient_codomain 处理替换 codomain 的操作
        return self._quotient_codomain(sm)

    def _apply(self, elem):
        """Apply ``self`` to ``elem``."""
        # 抽象方法，子类需要实现具体逻辑
        raise NotImplementedError

    def __call__(self, elem):
        # 调用 homomorphism 对象的 __call__ 方法，将 elem 转换为 self.domain 中的元素，然后应用 self._apply 方法
        return self.codomain.convert(self._apply(self.domain.convert(elem)))

    def _compose(self, oth):
        """
        Compose ``self`` with ``oth``, that is, return the homomorphism
        obtained by first applying then ``self``, then ``oth``.

        (This method is private since in this syntax, it is non-obvious which
        homomorphism is executed first.)
        """
        # 抽象方法，子类需要实现具体逻辑
        raise NotImplementedError

    def _mul_scalar(self, c):
        """Scalar multiplication. ``c`` is guaranteed in self.ring."""
        # 抽象方法，子类需要实现具体逻辑

    def _add(self, oth):
        """
        Homomorphism addition.
        ``oth`` is guaranteed to be a homomorphism with same domain/codomain.
        """
        # 抽象方法，子类需要实现具体逻辑

    def _check_hom(self, oth):
        """Helper to check that oth is a homomorphism with same domain/codomain."""
        # 检查 oth 是否为 ModuleHomomorphism 类型，并且其 domain 和 codomain 与 self 相同
        if not isinstance(oth, ModuleHomomorphism):
            return False
        return oth.domain == self.domain and oth.codomain == self.codomain

    def __mul__(self, oth):
        # 如果 oth 是 ModuleHomomorphism 类型，并且其 domain 和 self 的 codomain 相同，则返回 oth 与 self 的复合
        if isinstance(oth, ModuleHomomorphism) and self.domain == oth.codomain:
            return oth._compose(self)
        # 否则尝试进行标量乘法操作，如果失败则返回 NotImplemented
        try:
            return self._mul_scalar(self.ring.convert(oth))
        except CoercionFailed:
            return NotImplemented

    # NOTE: _compose will never be called from rmul
    # 因为在这个语法中，_compose 方法不会被 rmul 调用
    __rmul__ = __mul__

    def __truediv__(self, oth):
        # 尝试进行标量除法操作，如果失败则返回 NotImplemented
        try:
            return self._mul_scalar(1/self.ring.convert(oth))
        except CoercionFailed:
            return NotImplemented

    def __add__(self, oth):
        # 如果 oth 是同域同余域的 homomorphism，则返回它们的加法结果，否则返回 NotImplemented
        if self._check_hom(oth):
            return self._add(oth)
        return NotImplemented
    def __sub__(self, oth):
        # 如果自身与另一个对象类型相同，则执行减法操作
        if self._check_hom(oth):
            # 返回加法与相反数乘积的结果
            return self._add(oth._mul_scalar(self.ring.convert(-1)))
        # 如果类型不匹配，则返回 NotImplemented
        return NotImplemented

    def is_injective(self):
        """
        Return True if ``self`` is injective.

        That is, check if the elements of the domain are mapped to the same
        codomain element.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_injective()
        False
        >>> h.quotient_domain(h.kernel()).is_injective()
        True
        """
        # 返回核心是零的结果
        return self.kernel().is_zero()

    def is_surjective(self):
        """
        Return True if ``self`` is surjective.

        That is, check if every element of the codomain has at least one
        preimage.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_surjective()
        False
        >>> h.restrict_codomain(h.image()).is_surjective()
        True
        """
        # 返回映像与码域相等的结果
        return self.image() == self.codomain

    def is_isomorphism(self):
        """
        Return True if ``self`` is an isomorphism.

        That is, check if every element of the codomain has precisely one
        preimage. Equivalently, ``self`` is both injective and surjective.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h = h.restrict_codomain(h.image())
        >>> h.is_isomorphism()
        False
        >>> h.quotient_domain(h.kernel()).is_isomorphism()
        True
        """
        # 返回是否为单射且满射的结果
        return self.is_injective() and self.is_surjective()

    def is_zero(self):
        """
        Return True if ``self`` is a zero morphism.

        That is, check if every element of the domain is mapped to zero
        under self.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_zero()
        False
        >>> h.restrict_domain(F.submodule()).is_zero()
        True
        >>> h.quotient_codomain(h.image()).is_zero()
        True
        """
        # 返回映像是否为零的结果
        return self.image().is_zero()

    def __eq__(self, oth):
        # 尝试返回两个对象是否相等的结果，如果类型错误，则返回 False
        try:
            return (self - oth).is_zero()
        except TypeError:
            return False
    # 定义不等于运算符的特殊方法，用于判断当前对象是否不等于另一个对象
    def __ne__(self, oth):
        # 返回当前对象与另一个对象的相等性的反向结果
        return not (self == oth)
class MatrixHomomorphism(ModuleHomomorphism):
    r"""
    Helper class for all homomoprhisms which are expressed via a matrix.

    That is, for such homomorphisms ``domain`` is contained in a module
    generated by finitely many elements `e_1, \ldots, e_n`, so that the
    homomorphism is determined uniquely by its action on the `e_i`. It
    can thus be represented as a vector of elements of the codomain module,
    or potentially a supermodule of the codomain module
    (and hence conventionally as a matrix, if there is a similar interpretation
    for elements of the codomain module).

    Note that this class does *not* assume that the `e_i` freely generate a
    submodule, nor that ``domain`` is even all of this submodule. It exists
    only to unify the interface.

    Do not instantiate.

    Attributes:

    - matrix - the list of images determining the homomorphism.
    NOTE: the elements of matrix belong to either self.codomain or
          self.codomain.container

    Still non-implemented methods:

    - kernel
    - _apply
    """

    def __init__(self, domain, codomain, matrix):
        # 继承父类 ModuleHomomorphism 的初始化方法，设置 domain 和 codomain
        ModuleHomomorphism.__init__(self, domain, codomain)
        # 检查 matrix 的长度是否等于 domain 的秩，如果不等则抛出 ValueError 异常
        if len(matrix) != domain.rank:
            raise ValueError('Need to provide %s elements, got %s'
                             % (domain.rank, len(matrix)))

        # 设置转换器 converter 为 self.codomain.convert，若 codomain 是子模块或商模块则使用 container 的 convert 方法
        converter = self.codomain.convert
        if isinstance(self.codomain, (SubModule, SubQuotientModule)):
            converter = self.codomain.container.convert
        # 将 matrix 中的每个元素转换为 codomain 中的元素，并以元组形式存储在 self.matrix 中
        self.matrix = tuple(converter(x) for x in matrix)

    def _sympy_matrix(self):
        """Helper function which returns a SymPy matrix ``self.matrix``."""
        # 导入 SymPy 的 Matrix 类
        from sympy.matrices import Matrix
        # 定义转换函数 c，默认为恒等函数，若 codomain 是商模块或子商模块则为 x.data
        c = lambda x: x
        if isinstance(self.codomain, (QuotientModule, SubQuotientModule)):
            c = lambda x: x.data
        # 构造 SymPy 矩阵，转换 self.matrix 中的每个元素为 SymPy 格式，然后转置得到结果
        return Matrix([[self.ring.to_sympy(y) for y in c(x)] for x in self.matrix]).T

    def __repr__(self):
        # 调用 _sympy_matrix 方法获取 SymPy 格式的矩阵表示，并按照格式化输出
        lines = repr(self._sympy_matrix()).split('\n')
        t = " : %s -> %s" % (self.domain, self.codomain)
        s = ' '*len(t)
        n = len(lines)
        for i in range(n // 2):
            lines[i] += s
        lines[n // 2] += t
        for i in range(n//2 + 1, n):
            lines[i] += s
        return '\n'.join(lines)

    def _restrict_domain(self, sm):
        """Implementation of domain restriction."""
        # 返回一个新的 MatrixHomomorphism 对象，限制了 domain 为 sm
        return SubModuleHomomorphism(sm, self.codomain, self.matrix)

    def _restrict_codomain(self, sm):
        """Implementation of codomain restriction."""
        # 返回一个新的 MatrixHomomorphism 对象，限制了 codomain 为 sm
        return self.__class__(self.domain, sm, self.matrix)

    def _quotient_domain(self, sm):
        """Implementation of domain quotient."""
        # 返回一个新的 MatrixHomomorphism 对象，将 domain 除以 sm
        return self.__class__(self.domain/sm, self.codomain, self.matrix)
    # 定义一个方法，用于计算当前对象与给定对象 sm 的商空间
    def _quotient_codomain(self, sm):
        """Implementation of codomain quotient."""
        # 计算当前对象的 codomain 与 sm 的商空间 Q
        Q = self.codomain / sm
        # 获取一个用于转换的函数 converter
        converter = Q.convert
        # 如果当前对象的 codomain 是 SubModule 类的实例，则使用 Q.container 的转换函数
        if isinstance(self.codomain, SubModule):
            converter = Q.container.convert
        # 返回一个新的对象，表示当前对象的 domain 和 codomain 的商空间，以及经 converter 转换后的 matrix 列表
        return self.__class__(self.domain, self.codomain / sm,
            [converter(x) for x in self.matrix])

    # 定义一个方法，用于返回当前对象与另一个对象 oth 相加后的结果
    def _add(self, oth):
        return self.__class__(self.domain, self.codomain,
                              [x + y for x, y in zip(self.matrix, oth.matrix)])

    # 定义一个方法，用于返回当前对象与标量 c 相乘后的结果
    def _mul_scalar(self, c):
        return self.__class__(self.domain, self.codomain, [c * x for x in self.matrix])

    # 定义一个方法，用于返回当前对象与另一个对象 oth 组合后的结果
    def _compose(self, oth):
        return self.__class__(self.domain, oth.codomain, [oth(x) for x in self.matrix])
class FreeModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphisms with domain a free module or a quotient
    thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])
    """

    def _apply(self, elem):
        # If the domain is a quotient module, extract the actual data element.
        if isinstance(self.domain, QuotientModule):
            elem = elem.data
        # Perform the linear transformation using matrix multiplication.
        return sum(x * e for x, e in zip(elem, self.matrix))

    def _image(self):
        # Compute the submodule of the codomain generated by the image of the domain under this homomorphism.
        return self.codomain.submodule(*self.matrix)

    def _kernel(self):
        # Compute the kernel submodule of the domain under this homomorphism.
        # The kernel is essentially the syzygy module of the matrix entries.
        syz = self.image().syzygy_module()
        return self.domain.submodule(*syz.gens)


class SubModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphism with domain a submodule of a free module
    or a quotient thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> M = QQ.old_poly_ring(x).free_module(2)*x
    >>> homomorphism(M, M, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : <[x, 0], [0, x]> -> <[x, 0], [0, x]>
    [0, 1]])
    """

    def _apply(self, elem):
        # If the domain is a subquotient module, extract the actual data element.
        if isinstance(self.domain, SubQuotientModule):
            elem = elem.data
        # Perform the linear transformation using matrix multiplication.
        return sum(x * e for x, e in zip(elem, self.matrix))

    def _image(self):
        # Compute the submodule of the codomain generated by applying this homomorphism to each generator of the domain.
        return self.codomain.submodule(*[self(x) for x in self.domain.gens])

    def _kernel(self):
        # Compute the kernel submodule of the domain under this homomorphism.
        syz = self.image().syzygy_module()
        # Construct the submodule using linear combinations of generators.
        return self.domain.submodule(
            *[sum(xi*gi for xi, gi in zip(s, self.domain.gens))
              for s in syz.gens])


def homomorphism(domain, codomain, matrix):
    r"""
    Create a homomorphism object.

    This function tries to build a homomorphism from ``domain`` to ``codomain``
    via the matrix ``matrix``.

    Examples
    ========

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> R = QQ.old_poly_ring(x)
    >>> T = R.free_module(2)

    If ``domain`` is a free module generated by `e_1, \ldots, e_n`, then
    ``matrix`` should be an n-element iterable `(b_1, \ldots, b_n)` where
    the `b_i` are elements of ``codomain``. The constructed homomorphism is the
    unique homomorphism sending `e_i` to `b_i`.
    """
    >>> F = R.free_module(2)
    # 创建一个具有2个自由生成元的自由模块 F
    >>> h = homomorphism(F, T, [[1, x], [x**2, 0]])
    # 创建从 F 到 T 的同态映射 h，通过给定的矩阵 [[1, x], [x**2, 0]] 定义
    >>> h
    # 输出同态映射 h 的矩阵表示
    Matrix([
    [1, x**2], : QQ[x]**2 -> QQ[x]**2
    [x,    0]])
    # 显示 h 的矩阵表示，说明其定义域和值域
    >>> h([1, 0])
    # 应用同态映射 h 到向量 [1, 0]
    [1, x]
    # 得到向量 [1, x]，表示 h([1, 0]) 的结果
    >>> h([0, 1])
    # 应用同态映射 h 到向量 [0, 1]
    [x**2, 0]
    # 得到向量 [x**2, 0]，表示 h([0, 1]) 的结果
    >>> h([1, 1])
    # 应用同态映射 h 到向量 [1, 1]
    [x**2 + 1, x]
    # 得到向量 [x**2 + 1, x]，表示 h([1, 1]) 的结果

    If ``domain`` is a submodule of a free module, them ``matrix`` determines
    a homomoprhism from the containing free module to ``codomain``, and the
    homomorphism returned is obtained by restriction to ``domain``.

    # 如果 domain 是自由模块的子模块，则 matrix 确定从包含的自由模块到 codomain 的同态映射，
    # 并返回通过限制到 domain 而得到的同态映射。

    >>> S = F.submodule([1, 0], [0, x])
    # 创建 F 的一个子模块 S，通过生成元 [1, 0] 和 [0, x] 来定义
    >>> homomorphism(S, T, [[1, x], [x**2, 0]])
    # 创建从子模块 S 到 T 的同态映射，通过给定的矩阵 [[1, x], [x**2, 0]] 定义
    Matrix([
    [1, x**2], : <[1, 0], [0, x]> -> QQ[x]**2
    [x,    0]])

    If ``domain`` is a (sub)quotient `N/K`, then ``matrix`` determines a
    homomorphism from `N` to ``codomain``. If the kernel contains `K`, this
    homomorphism descends to ``domain`` and is returned; otherwise an exception
    is raised.

    # 如果 domain 是一个 (子)商模，那么 matrix 确定从 N 到 codomain 的同态映射。
    # 如果核包含 K，则这个同态映射降到 domain 并返回；否则会引发异常。

    >>> homomorphism(S/[(1, 0)], T, [0, [x**2, 0]])
    # 创建从商模 S/[(1, 0)] 到 T 的同态映射，通过给定的矩阵 [0, [x**2, 0]] 定义
    Matrix([
    [0, x**2], : <[1, 0] + <[1, 0]>, [0, x] + <[1, 0]>, [1, 0] + <[1, 0]>> -> QQ[x]**2
    [0,    0]])

    >>> homomorphism(S/[(0, x)], T, [0, [x**2, 0]])
    # 创建从商模 S/[(0, x)] 到 T 的同态映射，通过给定的矩阵 [0, [x**2, 0]] 定义
    Traceback (most recent call last):
    ...
    ValueError: kernel <[1, 0], [0, 0]> must contain sm, got <[0,x]>

    """
    def freepres(module):
        """
        Return a tuple ``(F, S, Q, c)`` where ``F`` is a free module, ``S`` is a
        submodule of ``F``, and ``Q`` a submodule of ``S``, such that
        ``module = S/Q``, and ``c`` is a conversion function.
        """
        # 根据输入的 module 返回元组 (F, S, Q, c)，其中 F 是自由模块，S 是 F 的子模块，
        # Q 是 S 的子模块，满足 module = S/Q，c 是一个转换函数。
        if isinstance(module, FreeModule):
            return module, module, module.submodule(), lambda x: module.convert(x)
        if isinstance(module, QuotientModule):
            return (module.base, module.base, module.killed_module,
                    lambda x: module.convert(x).data)
        if isinstance(module, SubQuotientModule):
            return (module.base.container, module.base, module.killed_module,
                    lambda x: module.container.convert(x).data)
        # an ordinary submodule
        return (module.container, module, module.submodule(),
                lambda x: module.container.convert(x))

    SF, SS, SQ, _ = freepres(domain)
    # 使用 freepres 函数得到 domain 的自由模块 SF、子模块 SS、子商模 SQ
    TF, TS, TQ, c = freepres(codomain)
    # 使用 freepres 函数得到 codomain 的自由模块 TF、子模块 TS、子商模 TQ
    # NOTE this is probably a bit inefficient (redundant checks)
    return FreeModuleHomomorphism(SF, TF, [c(x) for x in matrix]
         ).restrict_domain(SS).restrict_codomain(TS
         ).quotient_codomain(TQ).quotient_domain(SQ)
    # 创建自由模块同态映射 FreeModuleHomomorphism，使用矩阵转换函数 c，限制定义域和值域，
    # 并进行商模运算。
```