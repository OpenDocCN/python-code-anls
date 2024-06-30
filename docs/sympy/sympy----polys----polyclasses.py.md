# `D:\src\scipysrc\sympy\sympy\polys\polyclasses.py`

```
# 引入 __future__ 模块中的 annotations 特性，使得类可以引用自身作为类型注解
from __future__ import annotations

# 从 sympy.external.gmpy 模块中导入 GROUND_TYPES 常量
from sympy.external.gmpy import GROUND_TYPES

# 导入 sympy.utilities.exceptions 模块中的 sympy_deprecation_warning 函数
from sympy.utilities.exceptions import sympy_deprecation_warning

# 从 sympy.core.numbers 模块中导入 oo（无穷大）常量
from sympy.core.numbers import oo

# 从 sympy.core.sympify 模块中导入 CantSympify 异常类
from sympy.core.sympify import CantSympify

# 从 sympy.polys.polyutils 模块中导入 PicklableWithSlots 类和 _sort_factors 函数
from sympy.polys.polyutils import PicklableWithSlots, _sort_factors

# 从 sympy.polys.domains 模块中导入 Domain、ZZ、QQ 类（多项式相关的领域）
from sympy.polys.domains import Domain, ZZ, QQ

# 从 sympy.polys.polyerrors 模块中导入一系列异常类
from sympy.polys.polyerrors import (
    CoercionFailed,
    ExactQuotientFailed,
    DomainError,
    NotInvertible,
)

# 从 sympy.polys.densebasic 模块中导入多项式的基础操作函数和常量
from sympy.polys.densebasic import (
    ninf,
    dmp_validate,
    dup_normal, dmp_normal,
    dup_convert, dmp_convert,
    dmp_from_sympy,
    dup_strip,
    dmp_degree_in,
    dmp_degree_list,
    dmp_negative_p,
    dmp_ground_LC,
    dmp_ground_TC,
    dmp_ground_nth,
    dmp_one, dmp_ground,
    dmp_zero, dmp_zero_p, dmp_one_p, dmp_ground_p,
    dup_from_dict, dmp_from_dict,
    dmp_to_dict,
    dmp_deflate,
    dmp_inject, dmp_eject,
    dmp_terms_gcd,
    dmp_list_terms, dmp_exclude,
    dup_slice, dmp_slice_in, dmp_permute,
    dmp_to_tuple,
)

# 从 sympy.polys.densearith 模块中导入多项式的算术运算函数
from sympy.polys.densearith import (
    dmp_add_ground,
    dmp_sub_ground,
    dmp_mul_ground,
    dmp_quo_ground,
    dmp_exquo_ground,
    dmp_abs,
    dmp_neg,
    dmp_add,
    dmp_sub,
    dmp_mul,
    dmp_sqr,
    dmp_pow,
    dmp_pdiv,
    dmp_prem,
    dmp_pquo,
    dmp_pexquo,
    dmp_div,
    dmp_rem,
    dmp_quo,
    dmp_exquo,
    dmp_add_mul, dmp_sub_mul,
    dmp_max_norm,
    dmp_l1_norm,
    dmp_l2_norm_squared,
)

# 从 sympy.polys.densetools 模块中导入多项式的工具函数
from sympy.polys.densetools import (
    dmp_clear_denoms,
    dmp_integrate_in,
    dmp_diff_in,
    dmp_eval_in,
    dup_revert,
    dmp_ground_trunc,
    dmp_ground_content,
    dmp_ground_primitive,
    dmp_ground_monic,
    dmp_compose,
    dup_decompose,
    dup_shift,
    dmp_shift,
    dup_transform,
    dmp_lift,
)

# 从 sympy.polys.euclidtools 模块中导入欧几里得算法相关的函数
from sympy.polys.euclidtools import (
    dup_half_gcdex, dup_gcdex, dup_invert,
    dmp_subresultants,
    dmp_resultant,
    dmp_discriminant,
    dmp_inner_gcd,
    dmp_gcd,
    dmp_lcm,
    dmp_cancel,
)

# 从 sympy.polys.sqfreetools 模块中导入平方自由多项式相关的函数
from sympy.polys.sqfreetools import (
    dup_gff_list,
    dmp_norm,
    dmp_sqf_p,
    dmp_sqf_norm,
    dmp_sqf_part,
    dmp_sqf_list, dmp_sqf_list_include,
)

# 从 sympy.polys.factortools 模块中导入因式分解相关的函数
from sympy.polys.factortools import (
    dup_cyclotomic_p, dmp_irreducible_p,
    dmp_factor_list, dmp_factor_list_include,
)

# 从 sympy.polys.rootisolation 模块中导入多项式根隔离相关的函数
from sympy.polys.rootisolation import (
    dup_isolate_real_roots_sqf,
    dup_isolate_real_roots,
    dup_isolate_all_roots_sqf,
    dup_isolate_all_roots,
    dup_refine_real_root,
    dup_count_real_roots,
    dup_count_complex_roots,
    dup_sturm,
    dup_cauchy_upper_bound,
    dup_cauchy_lower_bound,
    dup_mignotte_sep_bound_squared,
)

# 从 sympy.polys.polyerrors 模块中导入多项式相关的异常类
from sympy.polys.polyerrors import (
    UnificationFailed,
    PolynomialError,
)

# 初始化 _flint_domains 变量为一个空元组
_flint_domains: tuple[Domain, ...]

# 根据 GROUND_TYPES 的值，设置 _flint_domains 变量
if GROUND_TYPES == 'flint':
    # 如果 GROUND_TYPES 是 'flint'，则导入 flint 模块并设置 _flint_domains 为 (ZZ, QQ)
    import flint
    _flint_domains = (ZZ, QQ)
else:
    # 否则，将 flint 设置为 None，并将 _flint_domains 设置为空元组
    flint = None
    _flint_domains = ()

# 定义 DMP 类，继承自 CantSympify 类，表示稠密多变量多项式
class DMP(CantSympify):
    """Dense Multivariate Polynomials over `K`. """
    pass
    __slots__ = ()
    # 空元组，限制实例的属性列表为空

    def __new__(cls, rep, dom, lev=None):
        # 构造函数，创建新实例

        if lev is None:
            # 如果未提供 lev 参数，进行数据验证和修正
            rep, lev = dmp_validate(rep)
        elif not isinstance(rep, list):
            # 如果 rep 不是列表类型，则抛出类型转换失败异常
            raise CoercionFailed("expected list, got %s" % type(rep))

        # 调用类方法 new 处理实例的创建和初始化
        return cls.new(rep, dom, lev)

    @classmethod
    def new(cls, rep, dom, lev):
        # 类方法：创建新实例，处理初始化逻辑

        # 在运行时总是调用 _validate_args 将导致性能问题
        # 理想情况下，应由静态类型检查器处理这种检查。
        #
        # cls._validate_args(rep, dom, lev)

        # 如果 flint 可用，并且 lev 为 0 且 dom 在 _flint_domains 中
        if flint is not None:
            if lev == 0 and dom in _flint_domains:
                # 返回 DUP_Flint 类型的新实例
                return DUP_Flint._new(rep, dom, lev)

        # 否则返回 DMP_Python 类型的新实例
        return DMP_Python._new(rep, dom, lev)

    @property
    def rep(f):
        """Get the representation of ``f``. """
        # 属性方法：获取 f 的表示形式

        sympy_deprecation_warning("""
        Accessing the ``DMP.rep`` attribute is deprecated. The internal
        representation of ``DMP`` instances can now be ``DUP_Flint`` when the
        ground types are ``flint``. In this case the ``DMP`` instance does not
        have a ``rep`` attribute. Use ``DMP.to_list()`` instead. Using
        ``DMP.to_list()`` also works in previous versions of SymPy.
        """,
            deprecated_since_version="1.13",
            active_deprecations_target="dmp-rep",
        )

        # 返回调用 f.to_list() 的结果
        return f.to_list()

    def to_best(f):
        """Convert to DUP_Flint if possible.

        This method should be used when the domain or level is changed and it
        potentially becomes possible to convert from DMP_Python to DUP_Flint.
        """
        # 方法：尝试转换为 DUP_Flint 类型，如果可能的话

        if flint is not None:
            if isinstance(f, DMP_Python) and f.lev == 0 and f.dom in _flint_domains:
                # 返回一个新的 DUP_Flint 实例
                return DUP_Flint.new(f._rep, f.dom, f.lev)

        # 否则返回原实例 f
        return f

    @classmethod
    def _validate_args(cls, rep, dom, lev):
        # 类方法：验证参数的正确性

        assert isinstance(dom, Domain)
        assert isinstance(lev, int) and lev >= 0

        def validate_rep(rep, lev):
            assert isinstance(rep, list)
            if lev == 0:
                assert all(dom.of_type(c) for c in rep)
            else:
                for r in rep:
                    validate_rep(r, lev - 1)

        # 调用内部函数进行递归验证
        validate_rep(rep, lev)

    @classmethod
    def from_dict(cls, rep, lev, dom):
        # 类方法：从字典创建实例

        # 使用 dmp_from_dict 转换 rep
        rep = dmp_from_dict(rep, lev, dom)
        # 调用 new 方法创建并返回新实例
        return cls.new(rep, dom, lev)

    @classmethod
    def from_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of native coefficients. """
        # 类方法：从列表创建实例，给定本地系数列表

        # 调用 dmp_convert 转换 rep，然后调用 new 方法创建并返回新实例
        return cls.new(dmp_convert(rep, lev, None, dom), dom, lev)

    @classmethod
    def from_sympy_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of SymPy coefficients. """
        # 类方法：从 SymPy 系数列表创建实例

        # 使用 dmp_from_sympy 转换 rep，然后调用 new 方法创建并返回新实例
        return cls.new(dmp_from_sympy(rep, lev, dom), dom, lev)

    @classmethod
    def from_monoms_coeffs(cls, monoms, coeffs, lev, dom):
        # 类方法：从单项式和系数创建实例

        # 创建字典并传入 monoms 和 coeffs，然后调用 new 方法创建并返回新实例
        return cls(dict(list(zip(monoms, coeffs))), dom, lev)
    def convert(f, dom):
        """Convert ``f`` to a ``DMP`` over the new domain. """
        # 如果函数的当前域与目标域相同，则直接返回原函数
        if f.dom == dom:
            return f
        # 如果函数的级数不为零或者 flint 是 None，则调用函数的 _convert 方法进行转换
        elif f.lev or flint is None:
            return f._convert(dom)
        # 如果函数是 DUP_Flint 类型，则根据目标域的情况进行不同的转换
        elif isinstance(f, DUP_Flint):
            if dom in _flint_domains:
                return f._convert(dom)
            else:
                return f.to_DMP_Python()._convert(dom)
        # 如果函数是 DMP_Python 类型，则根据目标域的情况进行不同的转换
        elif isinstance(f, DMP_Python):
            if dom in _flint_domains:
                return f._convert(dom).to_DUP_Flint()
            else:
                return f._convert(dom)
        # 如果以上条件都不满足，则抛出运行时错误
        else:
            raise RuntimeError("unreachable code")

    def _convert(f, dom):
        # 抛出未实现错误，表明该函数在子类中需要被实现
        raise NotImplementedError

    @classmethod
    def zero(cls, lev, dom):
        # 创建一个新的零多项式实例
        return DMP(dmp_zero(lev), dom, lev)

    @classmethod
    def one(cls, lev, dom):
        # 创建一个新的单位多项式实例
        return DMP(dmp_one(lev, dom), dom, lev)

    def _one(f):
        # 抛出未实现错误，表明该函数在子类中需要被实现
        raise NotImplementedError

    def __repr__(f):
        # 返回该对象的字符串表示，包括类名、系数列表和域
        return "%s(%s, %s)" % (f.__class__.__name__, f.to_list(), f.dom)

    def __hash__(f):
        # 返回该对象的哈希值，包括类名、系数元组、级数和域
        return hash((f.__class__.__name__, f.to_tuple(), f.lev, f.dom))

    def __getnewargs__(self):
        # 返回该对象的参数元组，用于序列化和反序列化
        return self.to_list(), self.dom, self.lev

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        # 抛出未实现错误，表明该函数在子类中需要被实现
        raise NotImplementedError

    def unify_DMP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        # 如果 g 不是 DMP 类型或者级数不同，则引发统一失败异常
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        # 如果函数的域不同，则尝试统一两者的域
        if f.dom != g.dom:
            dom = f.dom.unify(g.dom)
            f = f.convert(dom)
            g = g.convert(dom)

        # 返回统一后的两个函数实例
        return f, g

    def to_dict(f, zero=False):
        """Convert ``f`` to a dict representation with native coefficients. """
        # 使用函数的系数列表、级数和域生成一个字典表示
        return dmp_to_dict(f.to_list(), f.lev, f.dom, zero=zero)

    def to_sympy_dict(f, zero=False):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        # 转换为普通系数字典，然后将每个系数转换为 SymPy 类型
        rep = f.to_dict(zero=zero)

        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)

        return rep

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        # 将函数转换为列表表示，其中每个系数都转换为 SymPy 类型
        def sympify_nested_list(rep):
            out = []
            for val in rep:
                if isinstance(val, list):
                    out.append(sympify_nested_list(val))
                else:
                    out.append(f.dom.to_sympy(val))
            return out

        return sympify_nested_list(f.to_list())

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        # 抛出未实现错误，表明该函数在子类中需要被实现
        raise NotImplementedError

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        # 抛出未实现错误，表明该函数在子类中需要被实现
        raise NotImplementedError
    def to_ring(f):
        """将多项式 ``f`` 的基础域转换为环。"""
        return f.convert(f.dom.get_ring())

    def to_field(f):
        """将多项式 ``f`` 的基础域转换为域。"""
        return f.convert(f.dom.get_field())

    def to_exact(f):
        """使多项式 ``f`` 的基础域变为精确。"""
        return f.convert(f.dom.get_exact())

    def slice(f, m, n, j=0):
        """从多项式 ``f`` 中取连续子序列。"""
        if not f.lev and not j:
            return f._slice(m, n)
        else:
            return f._slice_lev(m, n, j)

    def _slice(f, m, n):
        """未实现的方法：对多项式 ``f`` 进行切片。"""
        raise NotImplementedError

    def _slice_lev(f, m, n, j):
        """未实现的方法：对多项式 ``f`` 进行按级别切片。"""
        raise NotImplementedError

    def coeffs(f, order=None):
        """按词典序返回多项式 ``f`` 中所有非零系数。"""
        return [ c for _, c in f.terms(order=order) ]

    def monoms(f, order=None):
        """按词典序返回多项式 ``f`` 中所有非零单项式。"""
        return [ m for m, _ in f.terms(order=order) ]

    def terms(f, order=None):
        """按词典序返回多项式 ``f`` 中所有非零项。"""
        if f.is_zero:
            zero_monom = (0,)*(f.lev + 1)
            return [(zero_monom, f.dom.zero)]
        else:
            return f._terms(order=order)

    def _terms(f, order=None):
        """未实现的方法：按指定顺序返回多项式 ``f`` 中的所有非零项。"""
        raise NotImplementedError

    def all_coeffs(f):
        """返回多项式 ``f`` 中的所有系数。"""
        if f.lev:
            raise PolynomialError('不支持多变量多项式')

        if not f:
            return [f.dom.zero]
        else:
            return list(f.to_list())

    def all_monoms(f):
        """返回多项式 ``f`` 中的所有单项式。"""
        if f.lev:
            raise PolynomialError('不支持多变量多项式')

        n = f.degree()

        if n < 0:
            return [(0,)]
        else:
            return [ (n - i,) for i, c in enumerate(f.to_list()) ]

    def all_terms(f):
        """返回多项式 ``f`` 中的所有项。"""
        if f.lev:
            raise PolynomialError('不支持多变量多项式')

        n = f.degree()

        if n < 0:
            return [((0,), f.dom.zero)]
        else:
            return [ ((n - i,), c) for i, c in enumerate(f.to_list()) ]

    def lift(f):
        """将多项式 ``f`` 中的代数系数提升为有理数。"""
        return f._lift().to_best()

    def _lift(f):
        """未实现的方法：将多项式 ``f`` 中的系数提升。"""
        raise NotImplementedError

    def deflate(f):
        """通过将 `x_i^m` 映射到 `y_i` 来减小多项式 `f` 的次数。"""
        raise NotImplementedError

    def inject(f, front=False):
        """将基础域生成元注入到多项式 ``f`` 中。"""
        raise NotImplementedError

    def eject(f, dom, front=False):
        """将选定的生成元注入到基础域中。"""
        raise NotImplementedError
    def exclude(f):
        r"""
        Remove useless generators from ``f``.

        Returns the removed generators and the new excluded ``f``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(1)]], [[ZZ(1)], [ZZ(2)]]], ZZ).exclude()
        ([2], DMP_Python([[1], [1, 2]], ZZ))

        """
        # 调用 f 对象的 _exclude 方法，移除无用的生成器
        J, F = f._exclude()
        # 返回被移除的生成器 J 和处理后的 F
        return J, F.to_best()

    def _exclude(f):
        # 抛出未实现的错误，暂未定义该方法的功能
        raise NotImplementedError

    def permute(f, P):
        r"""
        Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 0, 2])
        DMP_Python([[[2], []], [[1, 0], []]], ZZ)

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 2, 0])
        DMP_Python([[[1], []], [[2, 0], []]], ZZ)

        """
        # 调用 f 对象的 _permute 方法，根据排列 P 返回置换后的多项式
        return f._permute(P)

    def _permute(f, P):
        # 抛出未实现的错误，暂未定义该方法的功能
        raise NotImplementedError

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        # 抛出未实现的错误，暂未定义该方法的功能
        raise NotImplementedError

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        # 抛出未实现的错误，暂未定义该方法的功能
        raise NotImplementedError

    def neg(f):
        """Negate all coefficients in ``f``. """
        # 抛出未实现的错误，暂未定义该方法的功能
        raise NotImplementedError

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        # 调用 f 对象的 _add_ground 方法，将地面域的元素 c 添加到 f 中
        return f._add_ground(f.dom.convert(c))

    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        # 调用 f 对象的 _sub_ground 方法，从 f 中减去地面域的元素 c
        return f._sub_ground(f.dom.convert(c))

    def mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        # 调用 f 对象的 _mul_ground 方法，将 f 乘以地面域的元素 c
        return f._mul_ground(f.dom.convert(c))

    def quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        # 调用 f 对象的 _quo_ground 方法，计算 f 除以地面域的元素 c 的商
        return f._quo_ground(f.dom.convert(c))

    def exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
        # 调用 f 对象的 _exquo_ground 方法，计算 f 除以地面域的元素 c 的精确商
        return f._exquo_ground(f.dom.convert(c))

    def add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        # 将 f 和 g 转化为相同类型后，调用其 _add 方法进行多变量多项式的加法运算
        F, G = f.unify_DMP(g)
        return F._add(G)

    def sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        # 将 f 和 g 转化为相同类型后，调用其 _sub 方法进行多变量多项式的减法运算
        F, G = f.unify_DMP(g)
        return F._sub(G)

    def mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        # 将 f 和 g 转化为相同类型后，调用其 _mul 方法进行多变量多项式的乘法运算
        F, G = f.unify_DMP(g)
        return F._mul(G)

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        # 调用 f 对象的 _sqr 方法，对多变量多项式 f 进行平方运算
        return f._sqr()

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        # 检查 n 是否为整数，然后调用 f 对象的 _pow 方法，对多变量多项式 f 进行指数运算
        if not isinstance(n, int):
            raise TypeError("``int`` expected, got %s" % type(n))
        return f._pow(n)
    def pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一为相同的多项式类别
        F, G = f.unify_DMP(g)
        # 调用多项式类中的 _pdiv 方法进行伪除操作
        return F._pdiv(G)

    def prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一为相同的多项式类别
        F, G = f.unify_DMP(g)
        # 调用多项式类中的 _prem 方法进行伪余操作
        return F._prem(G)

    def pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一为相同的多项式类别
        F, G = f.unify_DMP(g)
        # 调用多项式类中的 _pquo 方法进行伪商操作
        return F._pquo(G)

    def pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一为相同的多项式类别
        F, G = f.unify_DMP(g)
        # 调用多项式类中的 _pexquo 方法进行精确伪商操作
        return F._pexquo(G)

    def div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一为相同的多项式类别
        F, G = f.unify_DMP(g)
        # 调用多项式类中的 _div 方法进行带余除法操作
        return F._div(G)

    def rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一为相同的多项式类别
        F, G = f.unify_DMP(g)
        # 调用多项式类中的 _rem 方法计算多项式余数
        return F._rem(G)

    def quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一为相同的多项式类别
        F, G = f.unify_DMP(g)
        # 调用多项式类中的 _quo 方法计算多项式商
        return F._quo(G)

    def exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一为相同的多项式类别
        F, G = f.unify_DMP(g)
        # 调用多项式类中的 _exquo 方法计算多项式精确商
        return F._exquo(G)

    def _add_ground(f, c):
        raise NotImplementedError

    def _sub_ground(f, c):
        raise NotImplementedError

    def _mul_ground(f, c):
        raise NotImplementedError

    def _quo_ground(f, c):
        raise NotImplementedError

    def _exquo_ground(f, c):
        raise NotImplementedError

    def _add(f, g):
        raise NotImplementedError

    def _sub(f, g):
        raise NotImplementedError

    def _mul(f, g):
        raise NotImplementedError

    def _sqr(f):
        raise NotImplementedError

    def _pow(f, n):
        raise NotImplementedError

    def _pdiv(f, g):
        raise NotImplementedError

    def _prem(f, g):
        raise NotImplementedError

    def _pquo(f, g):
        raise NotImplementedError

    def _pexquo(f, g):
        raise NotImplementedError

    def _div(f, g):
        raise NotImplementedError

    def _rem(f, g):
        raise NotImplementedError

    def _quo(f, g):
        raise NotImplementedError

    def _exquo(f, g):
        raise NotImplementedError

    def degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        # 检查 j 是否为整数，如果不是则引发类型错误异常
        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        # 调用多项式类中的 _degree 方法获取多项式在 x_j 上的最高次数
        return f._degree(j)

    def _degree(f, j):
        raise NotImplementedError

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        raise NotImplementedError

    def total_degree(f):
        """Returns the total degree of ``f``. """
        raise NotImplementedError
    def homogenize(f, s):
        """Return homogeneous polynomial of ``f``"""
        # 计算多项式 f 的总次数
        td = f.total_degree()
        # 初始化结果字典
        result = {}
        # 判断是否需要添加新的符号
        new_symbol = (s == len(f.terms()[0][0]))
        # 遍历 f 的每一个项
        for term in f.terms():
            # 计算当前项的次数
            d = sum(term[0])
            # 计算需要补齐的次数
            if d < td:
                i = td - d
            else:
                i = 0
            # 如果需要添加新符号，则直接将当前项添加到结果中
            if new_symbol:
                result[term[0] + (i,)] = term[1]
            else:
                # 否则，在指定位置 s 上增加补齐次数 i
                l = list(term[0])
                l[s] += i
                result[tuple(l)] = term[1]
        # 使用结果字典构造一个新的多项式对象，并返回
        return DMP.from_dict(result, f.lev + int(new_symbol), f.dom)

    def homogeneous_order(f):
        """Returns the homogeneous order of ``f``. """
        # 如果 f 是零多项式，返回负无穷
        if f.is_zero:
            return -oo

        # 获取所有单项式，并计算第一个单项式的总次数
        monoms = f.monoms()
        tdeg = sum(monoms[0])

        # 检查每个单项式的总次数是否与第一个单项式相同
        for monom in monoms:
            _tdeg = sum(monom)
            if _tdeg != tdeg:
                return None

        # 返回多项式的总次数
        return tdeg

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError

    def nth(f, *N):
        """Returns the ``n``-th coefficient of ``f``. """
        # 检查参数 N 是否全为整数
        if all(isinstance(n, int) for n in N):
            return f._nth(N)
        else:
            # 如果参数不全为整数，抛出类型错误
            raise TypeError("a sequence of integers expected")

    def _nth(f, N):
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError

    def integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        # 检查参数 m 和 j 是否为整数
        if not isinstance(m, int):
            raise TypeError("``int`` expected, got %s" % type(m))

        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        # 调用 f 的私有方法 _integrate 进行积分计算
        return f._integrate(m, j)

    def _integrate(f, m, j):
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError

    def diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
        # 检查参数 m 和 j 是否为整数
        if not isinstance(m, int):
            raise TypeError("``int`` expected, got %s" % type(m))

        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        # 调用 f 的私有方法 _diff 进行求导计算
        return f._diff(m, j)

    def _diff(f, m, j):
        # 未实现此功能，抛出未实现错误
        raise NotImplementedError
    # 定义一个函数 eval，用于在给定点 a 处计算多项式 f 的值
    def eval(f, a, j=0):
        """Evaluates ``f`` at the given point ``a`` in ``x_j``. """
        # 检查 j 是否为整数类型
        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))
        # 检查 j 是否在有效的变量索引范围内
        elif not (0 <= j <= f.lev):
            raise ValueError("invalid variable index %s" % j)

        # 如果多项式 f 的级数大于 0，则调用 _eval_lev 方法计算
        if f.lev:
            return f._eval_lev(a, j)
        # 否则调用 _eval 方法计算
        else:
            return f._eval(a)

    # 定义一个未实现的方法 _eval
    def _eval(f, a):
        raise NotImplementedError

    # 定义一个未实现的方法 _eval_lev
    def _eval_lev(f, a, j):
        raise NotImplementedError

    # 定义一个函数 half_gcdex，实现半扩展欧几里得算法（如果是单变量多项式）
    def half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
        # 将 f 和 g 统一为相同的多项式类型
        F, G = f.unify_DMP(g)

        # 如果多项式 F 的级数不为 0，则抛出异常
        if F.lev:
            raise ValueError('univariate polynomial expected')

        # 调用 F 的 _half_gcdex 方法计算结果
        return F._half_gcdex(G)

    # 定义一个未实现的方法 _half_gcdex
    def _half_gcdex(f, g):
        raise NotImplementedError

    # 定义一个函数 gcdex，实现扩展欧几里得算法（如果是单变量多项式）
    def gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
        # 将 f 和 g 统一为相同的多项式类型
        F, G = f.unify_DMP(g)

        # 如果多项式 F 的级数不为 0，则抛出异常
        if F.lev:
            raise ValueError('univariate polynomial expected')

        # 如果 F 的定义域不是一个域，则抛出域错误异常
        if not F.dom.is_Field:
            raise DomainError('ground domain must be a field')

        # 调用 F 的 _gcdex 方法计算结果
        return F._gcdex(G)

    # 定义一个未实现的方法 _gcdex
    def _gcdex(f, g):
        raise NotImplementedError

    # 定义一个函数 invert，实现在模 g 下对 f 进行求逆（如果可能）
    def invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        # 将 f 和 g 统一为相同的多项式类型
        F, G = f.unify_DMP(g)

        # 如果多项式 F 的级数不为 0，则抛出异常
        if F.lev:
            raise ValueError('univariate polynomial expected')

        # 调用 F 的 _invert 方法计算结果
        return F._invert(G)

    # 定义一个未实现的方法 _invert
    def _invert(f, g):
        raise NotImplementedError

    # 定义一个函数 revert，计算多项式 f 在模 x**n 下的逆
    def revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        # 如果多项式 f 的级数不为 0，则抛出异常
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 调用 f 的 _revert 方法计算结果
        return f._revert(n)

    # 定义一个未实现的方法 _revert
    def _revert(f, n):
        raise NotImplementedError

    # 定义一个函数 subresultants，计算多项式 f 和 g 的子结果序列
    def subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        # 将 f 和 g 统一为相同的多项式类型
        F, G = f.unify_DMP(g)
        # 调用 F 的 _subresultants 方法计算结果
        return F._subresultants(G)

    # 定义一个未实现的方法 _subresultants
    def _subresultants(f, g):
        raise NotImplementedError

    # 定义一个函数 resultant，计算多项式 f 和 g 的结果式
    def resultant(f, g, includePRS=False):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        # 将 f 和 g 统一为相同的多项式类型
        F, G = f.unify_DMP(g)
        # 如果 includePRS 参数为 True，则调用 F 的 _resultant_includePRS 方法计算结果
        if includePRS:
            return F._resultant_includePRS(G)
        # 否则调用 F 的 _resultant 方法计算结果
        else:
            return F._resultant(G)

    # 定义一个未实现的方法 _resultant
    def _resultant(f, g, includePRS=False):
        raise NotImplementedError

    # 定义一个函数 discriminant，计算多项式 f 的判别式
    def discriminant(f):
        """Computes discriminant of ``f``. """
        raise NotImplementedError

    # 定义一个函数 cofactors，返回多项式 f 和 g 的最大公因式及其因式
    def cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        # 将 f 和 g 统一为相同的多项式类型
        F, G = f.unify_DMP(g)
        # 调用 F 的 _cofactors 方法计算结果
        return F._cofactors(G)

    # 定义一个未实现的方法 _cofactors
    def _cofactors(f, g):
        raise NotImplementedError

    # 定义一个函数 gcd，返回多项式 f 和 g 的最大公因式
    def gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        # 将 f 和 g 统一为相同的多项式类型
        F, G = f.unify_DMP(g)
        # 调用 F 的 _gcd 方法计算结果
        return F._gcd(G)

    # 定义一个未实现的方法 _gcd
    def _gcd(f, g):
        raise NotImplementedError

    # 定义一个函数 lcm，返回多项式 f 和 g 的最小公倍式
    def lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        # 将 f 和 g 统一为相同的多项式类型
        F, G = f.unify_DMP(g)
        # 调用 F 的 _lcm 方法计算结果
        return F._lcm(G)

    # 定义一个未实现的方法 _lcm
    def _lcm(f, g):
        raise NotImplementedError
    # 取消有理函数 ``f/g`` 中的公因子。
    def cancel(f, g, include=True):
        """Cancel common factors in a rational function ``f/g``. """
        # 将多项式 f 和 g 统一到同一个表示形式
        F, G = f.unify_DMP(g)

        # 根据 include 参数选择具体的取消公因子方法
        if include:
            return F._cancel_include(G)
        else:
            return F._cancel(G)

    # 抛出未实现错误，用于私有方法 _cancel
    def _cancel(f, g):
        raise NotImplementedError

    # 抛出未实现错误，用于私有方法 _cancel_include
    def _cancel_include(f, g):
        raise NotImplementedError

    # 将多项式 f 对常数 p 取模，返回结果
    def trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f._trunc(f.dom.convert(p))

    # 抛出未实现错误，用于私有方法 _trunc
    def _trunc(f, p):
        raise NotImplementedError

    # 使多项式 f 变成首一多项式（所有系数除以 f 的首项系数）
    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        raise NotImplementedError

    # 返回多项式系数的最大公约数（GCD）
    def content(f):
        """Returns GCD of polynomial coefficients. """
        raise NotImplementedError

    # 返回多项式的内容和其原始形式
    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        raise NotImplementedError

    # 计算函数复合 f(g(x))
    def compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        # 将多项式 f 和 g 统一到同一个表示形式
        F, G = f.unify_DMP(g)
        # 调用 F 的私有方法 _compose 计算复合函数结果
        return F._compose(G)

    # 抛出未实现错误，用于私有方法 _compose
    def _compose(f, g):
        raise NotImplementedError

    # 计算多项式 f 的函数分解
    def decompose(f):
        """Computes functional decomposition of ``f``. """
        # 如果 f 的维数不为零，抛出值错误
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 调用 f 的私有方法 _decompose 计算函数分解结果
        return f._decompose()

    # 抛出未实现错误，用于私有方法 _decompose
    def _decompose(f):
        raise NotImplementedError

    # 计算多项式 f 的泰勒平移 f(x + a)
    def shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        # 如果 f 的维数不为零，抛出值错误
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 将 a 转换为 f.dom 中的对象，并调用 f 的私有方法 _shift 计算泰勒平移结果
        return f._shift(f.dom.convert(a))

    # 计算多项式 f 的列表形式的泰勒平移 f(X + A)
    def shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
        # 将列表 a 中的每个元素转换为 f.dom 中的对象
        a = [f.dom.convert(ai) for ai in a]
        # 调用 f 的私有方法 _shift_list 计算列表形式的泰勒平移结果
        return f._shift_list(a)

    # 抛出未实现错误，用于私有方法 _shift
    def _shift(f, a):
        raise NotImplementedError

    # 计算函数变换 q**n * f(p/q)
    def transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        # 如果 f 的维数不为零，抛出值错误
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 将 p 和 q 统一到同一个表示形式
        P, Q = p.unify_DMP(q)
        # 将 f 和 P 统一到同一个表示形式，并且 Q 也统一到同一个表示形式
        F, P = f.unify_DMP(P)
        F, Q = F.unify_DMP(Q)

        # 调用 F 的私有方法 _transform 计算函数变换结果
        return F._transform(P, Q)

    # 抛出未实现错误，用于私有方法 _transform
    def _transform(f, p, q):
        raise NotImplementedError

    # 计算多项式 f 的斯托姆序列
    def sturm(f):
        """Computes the Sturm sequence of ``f``. """
        # 如果 f 的维数不为零，抛出值错误
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 调用 f 的私有方法 _sturm 计算斯托姆序列
        return f._sturm()

    # 抛出未实现错误，用于私有方法 _sturm
    def _sturm(f):
        raise NotImplementedError

    # 计算多项式 f 的根的柯西上界
    def cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        # 如果 f 的维数不为零，抛出值错误
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 调用 f 的私有方法 _cauchy_upper_bound 计算柯西上界
        return f._cauchy_upper_bound()

    # 抛出未实现错误，用于私有方法 _cauchy_upper_bound
    def _cauchy_upper_bound(f):
        raise NotImplementedError

    # 计算多项式 f 的非零根的柯西下界
    def cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        # 如果 f 的维数不为零，抛出值错误
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 调用 f 的私有方法 _cauchy_lower_bound 计算柯西下界
        return f._cauchy_lower_bound()

    # 抛出未实现错误，用于私有方法 _cauchy_lower_bound
    def _cauchy_lower_bound(f):
        raise NotImplementedError
    def mignotte_sep_bound_squared(f):
        """计算 ``f`` 的平方Mignotte边界上的根分离。"""
        # 如果 f 是多变量多项式，则引发错误
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 调用内部方法计算平方Mignotte边界上的根分离
        return f._mignotte_sep_bound_squared()

    def _mignotte_sep_bound_squared(f):
        # 抛出未实现错误
        raise NotImplementedError

    def gff_list(f):
        """计算 ``f`` 的最大阶乘因子分解。"""
        # 如果 f 是多变量多项式，则引发错误
        if f.lev:
            raise ValueError('univariate polynomial expected')

        # 调用内部方法计算最大阶乘因子分解
        return f._gff_list()

    def _gff_list(f):
        # 抛出未实现错误
        raise NotImplementedError

    def norm(f):
        """计算 ``f`` 的 Norm。"""
        # 抛出未实现错误
        raise NotImplementedError

    def sqf_norm(f):
        """计算 ``f`` 的平方自由范数。"""
        # 抛出未实现错误
        raise NotImplementedError

    def sqf_part(f):
        """计算 ``f`` 的平方自由部分。"""
        # 抛出未实现错误
        raise NotImplementedError

    def sqf_list(f, all=False):
        """返回 ``f`` 的平方自由因子列表。"""
        # 抛出未实现错误
        raise NotImplementedError

    def sqf_list_include(f, all=False):
        """返回 ``f`` 的平方自由因子列表。"""
        # 抛出未实现错误
        raise NotImplementedError

    def factor_list(f):
        """返回 ``f`` 的不可约因子列表。"""
        # 抛出未实现错误
        raise NotImplementedError

    def factor_list_include(f):
        """返回 ``f`` 的不可约因子列表。"""
        # 抛出未实现错误
        raise NotImplementedError

    def intervals(f, all=False, eps=None, inf=None, sup=None, fast=False, sqf=False):
        """计算 ``f`` 的根的隔离区间。"""
        # 如果 f 是多变量多项式，则引发错误
        if f.lev:
            raise PolynomialError("Cannot isolate roots of a multivariate polynomial")

        # 根据参数返回不同类型的根隔离区间
        if all and sqf:
            return f._isolate_all_roots_sqf(eps=eps, inf=inf, sup=sup, fast=fast)
        elif all and not sqf:
            return f._isolate_all_roots(eps=eps, inf=inf, sup=sup, fast=fast)
        elif not all and sqf:
            return f._isolate_real_roots_sqf(eps=eps, inf=inf, sup=sup, fast=fast)
        else:
            return f._isolate_real_roots(eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        # 抛出未实现错误
        raise NotImplementedError

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        # 抛出未实现错误
        raise NotImplementedError

    def _isolate_real_roots(f, eps, inf, sup, fast):
        # 抛出未实现错误
        raise NotImplementedError

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        # 抛出未实现错误
        raise NotImplementedError

    def refine_root(f, s, t, eps=None, steps=None, fast=False):
        """
        将隔离区间精确到给定的精度。

        ``eps`` 应为有理数。

        """
        # 如果 f 是多变量多项式，则引发错误
        if f.lev:
            raise PolynomialError(
                "Cannot refine a root of a multivariate polynomial")

        # 调用内部方法将根的隔离区间精确到给定的精度
        return f._refine_real_root(s, t, eps=eps, steps=steps, fast=fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        # 抛出未实现错误
        raise NotImplementedError
    # 定义一个方法，用于计算多项式 ``f`` 在区间 ``[inf, sup]`` 中的实根数量
    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        raise NotImplementedError

    # 定义一个方法，用于计算多项式 ``f`` 在区间 ``[inf, sup]`` 中的复根数量
    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 是零多项式
    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 是单位多项式
    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 是域的元素
    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 是平方自由的
    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 的首项系数为1
    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 的系数的最大公因数为1
    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 对所有变量都是线性的
    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 对所有变量都是二次的
    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 是单项式（零或只有一个项）
    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 是齐次多项式
    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 是不可约的
    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        raise NotImplementedError

    # 属性方法，返回 ``True`` 如果多项式 ``f`` 是旋转多项式
    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        raise NotImplementedError

    # 方法，返回多项式 ``f`` 的绝对值
    def __abs__(f):
        return f.abs()

    # 方法，返回多项式 ``f`` 的相反数
    def __neg__(f):
        return f.neg()

    # 方法，返回多项式 ``f`` 与另一个对象 ``g`` 的和
    def __add__(f, g):
        if isinstance(g, DMP):
            return f.add(g)
        else:
            try:
                return f.add_ground(g)
            except CoercionFailed:
                return NotImplemented

    # 方法，返回多项式 ``f`` 与另一个对象 ``g`` 的和（右操作数版本）
    def __radd__(f, g):
        return f.__add__(g)

    # 方法，返回多项式 ``f`` 与另一个对象 ``g`` 的差
    def __sub__(f, g):
        if isinstance(g, DMP):
            return f.sub(g)
        else:
            try:
                return f.sub_ground(g)
            except CoercionFailed:
                return NotImplemented

    # 方法，返回多项式 ``f`` 与另一个对象 ``g`` 的差（右操作数版本）
    def __rsub__(f, g):
        return (-f).__add__(g)

    # 方法，返回多项式 ``f`` 与另一个对象 ``g`` 的积
    def __mul__(f, g):
        if isinstance(g, DMP):
            return f.mul(g)
        else:
            try:
                return f.mul_ground(g)
            except CoercionFailed:
                return NotImplemented

    # 方法，返回多项式 ``f`` 与另一个对象 ``g`` 的积（右操作数版本）
    def __rmul__(f, g):
        return f.__mul__(g)
    def __truediv__(f, g):
        # 如果 g 是 DMP 类型的对象，则调用 f 对象的 exquo 方法进行除法运算
        if isinstance(g, DMP):
            return f.exquo(g)
        else:
            # 否则，尝试调用 f 对象的 mul_ground 方法进行乘法运算
            try:
                return f.mul_ground(g)
            except CoercionFailed:
                # 如果类型转换失败，则返回 NotImplemented
                return NotImplemented

    def __rtruediv__(f, g):
        # 如果 g 是 DMP 类型的对象，则调用 g 对象的 exquo 方法进行除法运算
        if isinstance(g, DMP):
            return g.exquo(f)
        else:
            # 否则，尝试调用 f 对象的 _one().mul_ground(g).exquo(f) 方法进行乘法运算和除法运算
            try:
                return f._one().mul_ground(g).exquo(f)
            except CoercionFailed:
                # 如果类型转换失败，则返回 NotImplemented
                return NotImplemented

    def __pow__(f, n):
        # 返回 f 对象的 pow 方法计算 f 的 n 次方
        return f.pow(n)

    def __divmod__(f, g):
        # 返回 f 对象的 div 方法计算 f 除以 g 的商和余数
        return f.div(g)

    def __mod__(f, g):
        # 返回 f 对象的 rem 方法计算 f 除以 g 的余数
        return f.rem(g)

    def __floordiv__(f, g):
        # 如果 g 是 DMP 类型的对象，则调用 f 对象的 quo 方法进行整数除法运算
        if isinstance(g, DMP):
            return f.quo(g)
        else:
            # 否则，尝试调用 f 对象的 quo_ground 方法进行整数除法运算
            try:
                return f.quo_ground(g)
            except TypeError:
                # 如果类型转换失败，则返回 NotImplemented
                return NotImplemented

    def __eq__(f, g):
        # 如果 f 和 g 是同一个对象，则返回 True
        if f is g:
            return True
        # 如果 g 不是 DMP 类型的对象，则返回 NotImplemented
        if not isinstance(g, DMP):
            return NotImplemented
        try:
            # 尝试将 f 和 g 统一成同一类型对象，并比较它们是否严格相等
            F, G = f.unify_DMP(g)
        except UnificationFailed:
            # 如果统一失败，则返回 False
            return False
        else:
            # 如果统一成功，则调用 F 对象的 _strict_eq 方法比较是否严格相等
            return F._strict_eq(G)

    def _strict_eq(f, g):
        # 抛出 NotImplementedError 异常，表示严格相等的比较未实现
        raise NotImplementedError

    def eq(f, g, strict=False):
        # 如果 strict 参数为 False，则调用 f 对象的 __eq__ 方法进行比较
        if not strict:
            return f == g
        else:
            # 否则，调用 f 对象的 _strict_eq 方法进行严格相等的比较
            return f._strict_eq(g)

    def ne(f, g, strict=False):
        # 返回 f 对象与 g 对象是否不相等的结果，根据 strict 参数决定是否严格比较
        return not f.eq(g, strict=strict)

    def __lt__(f, g):
        # 将 f 和 g 对象统一成同一类型对象，然后比较它们的值的大小关系
        F, G = f.unify_DMP(g)
        return F.to_list() < G.to_list()

    def __le__(f, g):
        # 将 f 和 g 对象统一成同一类型对象，然后比较它们的值的大小关系
        F, G = f.unify_DMP(g)
        return F.to_list() <= G.to_list()

    def __gt__(f, g):
        # 将 f 和 g 对象统一成同一类型对象，然后比较它们的值的大小关系
        F, G = f.unify_DMP(g)
        return F.to_list() > G.to_list()

    def __ge__(f, g):
        # 将 f 和 g 对象统一成同一类型对象，然后比较它们的值的大小关系
        F, G = f.unify_DMP(g)
        return F.to_list() >= G.to_list()

    def __bool__(f):
        # 返回 f 对象是否不为零
        return not f.is_zero
class DMP_Python(DMP):
    """Dense Multivariate Polynomials over `K`. """

    __slots__ = ('_rep', 'dom', 'lev')

    @classmethod
    def _new(cls, rep, dom, lev):
        obj = object.__new__(cls)  # 创建一个新的实例对象，继承自当前类（DMP_Python）
        obj._rep = rep  # 设置实例对象的内部表示（_rep）为给定的表示（rep）
        obj.lev = lev  # 设置实例对象的级别（lev）为给定的级别（lev）
        obj.dom = dom  # 设置实例对象的域（dom）为给定的域（dom）
        return obj  # 返回创建的实例对象

    def _strict_eq(f, g):
        if type(f) != type(g):
            return False
        return f.lev == g.lev and f.dom == g.dom and f._rep == g._rep

    def per(f, rep):
        """Create a DMP out of the given representation. """
        return f._new(rep, f.dom, f.lev)  # 使用给定的表示（rep）创建一个新的 DMP 实例对象

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        return f._new(dmp_ground(coeff, f.lev), f.dom, f.lev)  # 使用给定的系数（coeff）构造一个新的 DMP 实例对象

    def _one(f):
        return f.one(f.lev, f.dom)  # 返回表示多项式 1 的 DMP 实例对象

    def unify(f, g):
        """Unify representations of two multivariate polynomials. """
        # XXX: This function is not really used any more since there is
        # unify_DMP now.
        if not isinstance(g, DMP) or f.lev != g.lev:  # 如果 g 不是 DMP 类型或者级别不同，抛出 UnificationFailed 异常
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:  # 如果域相同，直接返回对应的信息
            return f.lev, f.dom, f.per, f._rep, g._rep
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)  # 否则，统一域到一个新的域（dom）

            F = dmp_convert(f._rep, lev, f.dom, dom)  # 将 f 的表示转换到新的域（dom）
            G = dmp_convert(g._rep, lev, g.dom, dom)  # 将 g 的表示转换到新的域（dom）

            def per(rep):
                return f._new(rep, dom, lev)  # 返回在新域上创建的 DMP 实例对象

            return lev, dom, per, F, G  # 返回级别、新域、创建函数、转换后的 f 和 g

    def to_DUP_Flint(f):
        """Convert ``f`` to a Flint representation. """
        return DUP_Flint._new(f._rep, f.dom, f.lev)  # 将当前对象转换为 Flint 表示的实例对象

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return list(f._rep)  # 将多项式的表示转换为具有本地系数的列表表示

    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
        return dmp_to_tuple(f._rep, f.lev)  # 将多项式的表示转换为具有本地系数的元组表示

    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
        return f._new(dmp_convert(f._rep, f.lev, f.dom, dom), dom, f.lev)  # 将多项式的域转换为给定的新域（dom）

    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
        rep = dup_slice(f._rep, m, n, f.dom)  # 获取多项式在指定范围内的连续子序列表示
        return f._new(rep, f.dom, f.lev)  # 返回新的 DMP 实例对象

    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
        rep = dmp_slice_in(f._rep, m, n, j, f.lev, f.dom)  # 获取多项式在指定级别和范围内的连续子序列表示
        return f._new(rep, f.dom, f.lev)  # 返回新的 DMP 实例对象

    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        return dmp_list_terms(f._rep, f.lev, f.dom, order=order)  # 返回多项式中所有非零项的列表表示，按照指定的顺序

    def _lift(f):
        """Convert algebraic coefficients to rationals. """
        r = dmp_lift(f._rep, f.lev, f.dom)  # 将多项式中的代数系数转换为有理数
        return f._new(r, f.dom.dom, f.lev)  # 返回新的 DMP 实例对象，有理数系数

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        J, F = dmp_deflate(f._rep, f.lev, f.dom)  # 减少多项式的次数，将 x_i^m 映射到 y_i
        return J, f.per(F)  # 返回减少次数后的级别 J 和新的 DMP 实例对象
    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        # 调用多项式操作函数，将地面域生成器注入到多项式 ``f`` 中
        F, lev = dmp_inject(f._rep, f.lev, f.dom, front=front)
        # XXX: domain and level changed here
        # 返回一个新的多项式对象，包含注入后的表示和域信息
        return f._new(F, f.dom.dom, lev)

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        # 从多项式 ``f`` 中选择的生成器弹出到地面域
        F = dmp_eject(f._rep, f.lev, dom, front=front)
        # XXX: domain and level changed here
        # 返回一个新的多项式对象，包含弹出后的表示和级别信息更新
        return f._new(F, dom, f.lev - len(dom.symbols))

    def _exclude(f):
        """Remove useless generators from ``f``. """
        # 从多项式 ``f`` 中移除无用的生成器
        J, F, u = dmp_exclude(f._rep, f.lev, f.dom)
        # XXX: level changed here
        # 返回一个元组，包含移除后的表示、更新后的域信息和级别信息
        return J, f._new(F, f.dom, u)

    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
        # 返回在 `K[x_{P(1)}, ..., x_{P(n)}]` 中的多项式
        return f.per(dmp_permute(f._rep, P, f.lev, f.dom))

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        # 从多项式 ``f`` 中移除项的最大公约数
        J, F = dmp_terms_gcd(f._rep, f.lev, f.dom)
        # 返回一个元组，包含移除后的表示和更新后的多项式
        return J, f.per(F)

    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        # 将地面域的元素添加到多项式 ``f`` 中
        return f.per(dmp_add_ground(f._rep, c, f.lev, f.dom))

    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        # 从多项式 ``f`` 中减去地面域的元素
        return f.per(dmp_sub_ground(f._rep, c, f.lev, f.dom))

    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        # 将多项式 ``f`` 乘以地面域的元素
        return f.per(dmp_mul_ground(f._rep, c, f.lev, f.dom))

    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        # 多项式 ``f`` 除以地面域的元素
        return f.per(dmp_quo_ground(f._rep, c, f.lev, f.dom))

    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
        # 多项式 ``f`` 对地面域的元素进行精确的除法
        return f.per(dmp_exquo_ground(f._rep, c, f.lev, f.dom))

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        # 使多项式 ``f`` 中所有系数变为正数
        return f.per(dmp_abs(f._rep, f.lev, f.dom))

    def neg(f):
        """Negate all coefficients in ``f``. """
        # 反转多项式 ``f`` 中所有系数的符号
        return f.per(dmp_neg(f._rep, f.lev, f.dom))

    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        # 将两个多变量多项式 ``f`` 和 ``g`` 相加
        return f.per(dmp_add(f._rep, g._rep, f.lev, f.dom))

    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        # 将两个多变量多项式 ``f`` 和 ``g`` 相减
        return f.per(dmp_sub(f._rep, g._rep, f.lev, f.dom))

    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        # 将两个多变量多项式 ``f`` 和 ``g`` 相乘
        return f.per(dmp_mul(f._rep, g._rep, f.lev, f.dom))

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        # 对多变量多项式 ``f`` 进行平方
        return f.per(dmp_sqr(f._rep, f.lev, f.dom))

    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        # 将多变量多项式 ``f`` 提升到非负整数幂 ``n``
        return f.per(dmp_pow(f._rep, n, f.lev, f.dom))

    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        # 对多项式 ``f`` 和 ``g`` 进行多项式伪除法
        q, r = dmp_pdiv(f._rep, g._rep, f.lev, f.dom)
        # 返回伪商和余数，作为新的多项式对象
        return f.per(q), f.per(r)
    def _prem(f, g):
        """计算多项式 f 和 g 的伪余数。"""
        # 调用底层函数 dmp_prem 计算多项式 f 和 g 的伪余数
        return f.per(dmp_prem(f._rep, g._rep, f.lev, f.dom))

    def _pquo(f, g):
        """计算多项式 f 和 g 的伪商。"""
        # 调用底层函数 dmp_pquo 计算多项式 f 和 g 的伪商
        return f.per(dmp_pquo(f._rep, g._rep, f.lev, f.dom))

    def _pexquo(f, g):
        """计算多项式 f 和 g 的精确伪商。"""
        # 调用底层函数 dmp_pexquo 计算多项式 f 和 g 的精确伪商
        return f.per(dmp_pexquo(f._rep, g._rep, f.lev, f.dom))

    def _div(f, g):
        """计算多项式 f 和 g 的带余除法。"""
        # 调用底层函数 dmp_div 计算多项式 f 和 g 的带余除法
        q, r = dmp_div(f._rep, g._rep, f.lev, f.dom)
        return f.per(q), f.per(r)

    def _rem(f, g):
        """计算多项式 f 对 g 的余数。"""
        # 调用底层函数 dmp_rem 计算多项式 f 对 g 的余数
        return f.per(dmp_rem(f._rep, g._rep, f.lev, f.dom))

    def _quo(f, g):
        """计算多项式 f 除以 g 的商。"""
        # 调用底层函数 dmp_quo 计算多项式 f 除以 g 的商
        return f.per(dmp_quo(f._rep, g._rep, f.lev, f.dom))

    def _exquo(f, g):
        """计算多项式 f 除以 g 的精确商。"""
        # 调用底层函数 dmp_exquo 计算多项式 f 除以 g 的精确商
        return f.per(dmp_exquo(f._rep, g._rep, f.lev, f.dom))

    def _degree(f, j=0):
        """返回多项式 f 在 x_j 中的最高次数。"""
        # 调用底层函数 dmp_degree_in 返回多项式 f 在 x_j 中的最高次数
        return dmp_degree_in(f._rep, j, f.lev)

    def degree_list(f):
        """返回多项式 f 的各项的次数列表。"""
        # 调用底层函数 dmp_degree_list 返回多项式 f 的各项的次数列表
        return dmp_degree_list(f._rep, f.lev)

    def total_degree(f):
        """返回多项式 f 的总次数。"""
        # 利用 f.monoms() 计算多项式 f 的各单项式的次数之和，并返回最大值
        return max(sum(m) for m in f.monoms())

    def LC(f):
        """返回多项式 f 的首项系数。"""
        # 调用底层函数 dmp_ground_LC 返回多项式 f 的首项系数
        return dmp_ground_LC(f._rep, f.lev, f.dom)

    def TC(f):
        """返回多项式 f 的末项系数。"""
        # 调用底层函数 dmp_ground_TC 返回多项式 f 的末项系数
        return dmp_ground_TC(f._rep, f.lev, f.dom)

    def _nth(f, N):
        """返回多项式 f 的第 N 项系数。"""
        # 调用底层函数 dmp_ground_nth 返回多项式 f 的第 N 项系数
        return dmp_ground_nth(f._rep, N, f.lev, f.dom)

    def max_norm(f):
        """返回多项式 f 的最大范数。"""
        # 调用底层函数 dmp_max_norm 返回多项式 f 的最大范数
        return dmp_max_norm(f._rep, f.lev, f.dom)

    def l1_norm(f):
        """返回多项式 f 的 L1 范数。"""
        # 调用底层函数 dmp_l1_norm 返回多项式 f 的 L1 范数
        return dmp_l1_norm(f._rep, f.lev, f.dom)

    def l2_norm_squared(f):
        """返回多项式 f 的平方 L2 范数。"""
        # 调用底层函数 dmp_l2_norm_squared 返回多项式 f 的平方 L2 范数
        return dmp_l2_norm_squared(f._rep, f.lev, f.dom)

    def clear_denoms(f):
        """清除多项式 f 的分母，但保留其基础域。"""
        # 调用底层函数 dmp_clear_denoms 清除多项式 f 的分母，并返回清除分母后的系数和新的多项式
        coeff, F = dmp_clear_denoms(f._rep, f.lev, f.dom)
        return coeff, f.per(F)

    def _integrate(f, m=1, j=0):
        """计算多项式 f 在 x_j 上的 m 阶不定积分。"""
        # 调用底层函数 dmp_integrate_in 计算多项式 f 在 x_j 上的 m 阶不定积分
        return f.per(dmp_integrate_in(f._rep, m, j, f.lev, f.dom))

    def _diff(f, m=1, j=0):
        """计算多项式 f 在 x_j 上的 m 阶导数。"""
        # 调用底层函数 dmp_diff_in 计算多项式 f 在 x_j 上的 m 阶导数
        return f.per(dmp_diff_in(f._rep, m, j, f.lev, f.dom))

    def _eval(f, a):
        """计算多项式 f 在点 a 处的值。"""
        return dmp_eval_in(f._rep, f.dom.convert(a), 0, f.lev, f.dom)

    def _eval_lev(f, a, j):
        """计算多项式 f 在点 a 处、相对于 x_j 的值。"""
        # 调用底层函数 dmp_eval_in 计算多项式 f 在点 a 处、相对于 x_j 的值
        rep = dmp_eval_in(f._rep, f.dom.convert(a), j, f.lev, f.dom)
        return f.new(rep, f.dom, f.lev - 1)
    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
        # 调用底层函数 dup_half_gcdex 计算两个多项式 f 和 g 的半扩展欧几里得算法
        s, h = dup_half_gcdex(f._rep, g._rep, f.dom)
        # 将结果映射到当前多项式环上并返回
        return f.per(s), f.per(h)

    def _gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
        # 调用底层函数 dup_gcdex 计算两个多项式 f 和 g 的扩展欧几里得算法
        s, t, h = dup_gcdex(f._rep, g._rep, f.dom)
        # 将结果映射到当前多项式环上并返回
        return f.per(s), f.per(t), f.per(h)

    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        # 调用底层函数 dup_invert 计算多项式 f 对 g 取模的乘法逆元
        s = dup_invert(f._rep, g._rep, f.dom)
        # 将结果映射到当前多项式环上并返回
        return f.per(s)

    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        # 调用底层函数 dup_revert 计算多项式 f 在 x**n 下的乘法逆元
        return f.per(dup_revert(f._rep, n, f.dom))

    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        # 调用底层函数 dmp_subresultants 计算多项式 f 和 g 的子结果ant PRS 序列
        R = dmp_subresultants(f._rep, g._rep, f.lev, f.dom)
        # 将结果映射到当前多项式环上并返回列表
        return list(map(f.per, R))

    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        # 调用底层函数 dmp_resultant 计算多项式 f 和 g 的结果ant，包括 PRS 信息
        res, R = dmp_resultant(f._rep, g._rep, f.lev, f.dom, includePRS=True)
        # 如果结果的级别大于零，则调整结果的级别
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        # 将结果映射到当前多项式环上并返回结果和列表
        return res, list(map(f.per, R))

    def _resultant(f, g):
        # 调用底层函数 dmp_resultant 计算多项式 f 和 g 的结果ant
        res = dmp_resultant(f._rep, g._rep, f.lev, f.dom)
        # 如果结果的级别大于零，则调整结果的级别
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        # 将结果映射到当前多项式环上并返回
        return res

    def discriminant(f):
        """Computes discriminant of ``f``. """
        # 调用底层函数 dmp_discriminant 计算多项式 f 的判别式
        res = dmp_discriminant(f._rep, f.lev, f.dom)
        # 如果结果的级别大于零，则调整结果的级别
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        # 将结果映射到当前多项式环上并返回
        return res

    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        # 调用底层函数 dmp_inner_gcd 计算多项式 f 和 g 的最大公因式以及它们的互素部分
        h, cff, cfg = dmp_inner_gcd(f._rep, g._rep, f.lev, f.dom)
        # 将结果映射到当前多项式环上并返回
        return f.per(h), f.per(cff), f.per(cfg)

    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        # 调用底层函数 dmp_gcd 计算多项式 f 和 g 的最大公因式
        return f.per(dmp_gcd(f._rep, g._rep, f.lev, f.dom))

    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        # 调用底层函数 dmp_lcm 计算多项式 f 和 g 的最小公倍式
        return f.per(dmp_lcm(f._rep, g._rep, f.lev, f.dom))

    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        # 调用底层函数 dmp_cancel 取消有理函数 f/g 中的公因式
        cF, cG, F, G = dmp_cancel(f._rep, g._rep, f.lev, f.dom, include=False)
        # 将结果映射到当前多项式环上并返回
        return cF, cG, f.per(F), f.per(G)

    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        # 调用底层函数 dmp_cancel 取消有理函数 f/g 中的公因式，包括其结果
        F, G = dmp_cancel(f._rep, g._rep, f.lev, f.dom, include=True)
        # 将结果映射到当前多项式环上并返回
        return f.per(F), f.per(G)

    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        # 调用底层函数 dmp_ground_trunc 将多项式 f 对常数 p 取模
        return f.per(dmp_ground_trunc(f._rep, p, f.lev, f.dom))

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        # 调用底层函数 dmp_ground_monic 使多项式 f 变为首项系数为 1 的首一多项式
        return f.per(dmp_ground_monic(f._rep, f.lev, f.dom))

    def content(f):
        """Returns GCD of polynomial coefficients. """
        # 调用底层函数 dmp_ground_content 计算多项式 f 的系数的最大公因数
        return dmp_ground_content(f._rep, f.lev, f.dom)

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        # 调用底层函数 dmp_ground_primitive 计算多项式 f 的内容和其原始形式
        cont, F = dmp_ground_primitive(f._rep, f.lev, f.dom)
        # 将结果映射到当前多项式环上并返回
        return cont, f.per(F)
    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        # 使用函数 dmp_compose 计算函数 f 和 g 的组合结果，并返回
        return f.per(dmp_compose(f._rep, g._rep, f.lev, f.dom))

    def _decompose(f):
        """Computes functional decomposition of ``f``. """
        # 使用函数 dup_decompose 对函数 f 进行分解，返回分解后的结果列表
        return list(map(f.per, dup_decompose(f._rep, f.dom)))

    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        # 使用函数 dup_shift 计算函数 f 在变量 x 上的 Taylor 平移结果，返回平移后的函数对象
        return f.per(dup_shift(f._rep, a, f.dom))

    def _shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
        # 使用函数 dmp_shift 计算函数 f 在向量 X 上的 Taylor 平移结果，返回平移后的函数对象
        return f.per(dmp_shift(f._rep, a, f.lev, f.dom))

    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        # 使用函数 dup_transform 对函数 f 进行变换计算，返回变换后的函数对象
        return f.per(dup_transform(f._rep, p._rep, q._rep, f.dom))

    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
        # 使用函数 dup_sturm 计算函数 f 的斯图姆序列，返回序列中各项的函数对象列表
        return list(map(f.per, dup_sturm(f._rep, f.dom)))

    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        # 使用函数 dup_cauchy_upper_bound 计算函数 f 的根的上确界，返回上确界值
        return dup_cauchy_upper_bound(f._rep, f.dom)

    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        # 使用函数 dup_cauchy_lower_bound 计算函数 f 非零根的下确界，返回下确界值
        return dup_cauchy_lower_bound(f._rep, f.dom)

    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        # 使用函数 dup_mignotte_sep_bound_squared 计算函数 f 根之间分离的平方米诺特界限，返回界限值
        return dup_mignotte_sep_bound_squared(f._rep, f.dom)

    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        # 使用函数 dup_gff_list 计算函数 f 的最大阶乘因子分解，返回分解后的列表
        return [ (f.per(g), k) for g, k in dup_gff_list(f._rep, f.dom) ]

    def norm(f):
        """Computes ``Norm(f)``."""
        # 使用函数 dmp_norm 计算函数 f 的范数，返回计算后的函数对象
        r = dmp_norm(f._rep, f.lev, f.dom)
        return f.new(r, f.dom.dom, f.lev)

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        # 使用函数 dmp_sqf_norm 计算函数 f 的无平方因子范数，返回计算后的结果元组
        s, g, r = dmp_sqf_norm(f._rep, f.lev, f.dom)
        return s, f.per(g), f.new(r, f.dom.dom, f.lev)

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        # 使用函数 dmp_sqf_part 计算函数 f 的无平方因子部分，返回计算后的函数对象
        return f.per(dmp_sqf_part(f._rep, f.lev, f.dom))

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        # 使用函数 dmp_sqf_list 获取函数 f 的无平方因子列表，返回系数和因子对的列表
        coeff, factors = dmp_sqf_list(f._rep, f.lev, f.dom, all)
        return coeff, [ (f.per(g), k) for g, k in factors ]

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        # 使用函数 dmp_sqf_list_include 获取函数 f 的无平方因子列表，返回因子对的列表
        factors = dmp_sqf_list_include(f._rep, f.lev, f.dom, all)
        return [ (f.per(g), k) for g, k in factors ]

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
        # 使用函数 dmp_factor_list 获取函数 f 的不可约因子列表，返回系数和因子对的列表
        coeff, factors = dmp_factor_list(f._rep, f.lev, f.dom)
        return coeff, [ (f.per(g), k) for g, k in factors ]

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        # 使用函数 dmp_factor_list_include 获取函数 f 的不可约因子列表，返回因子对的列表
        factors = dmp_factor_list_include(f._rep, f.lev, f.dom)
        return [ (f.per(g), k) for g, k in factors ]
    # 调用函数 dup_isolate_real_roots，用于在指定区间内孤立多项式 f 的实根
    def _isolate_real_roots(f, eps, inf, sup, fast):
        return dup_isolate_real_roots(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    # 调用函数 dup_isolate_real_roots_sqf，用于在指定区间内孤立多项式 f 的实根（对平方因式分解形式的多项式特化）
    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        return dup_isolate_real_roots_sqf(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    # 调用函数 dup_isolate_all_roots，用于在指定区间内孤立多项式 f 的所有根（实根和复根）
    def _isolate_all_roots(f, eps, inf, sup, fast):
        return dup_isolate_all_roots(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    # 调用函数 dup_isolate_all_roots_sqf，用于在指定区间内孤立多项式 f 的所有根（对平方因式分解形式的多项式特化）
    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        return dup_isolate_all_roots_sqf(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    # 调用函数 dup_refine_real_root，用于在指定区间内精炼多项式 f 的实根
    def _refine_real_root(f, s, t, eps, steps, fast):
        return dup_refine_real_root(f._rep, s, t, f.dom, eps=eps, steps=steps, fast=fast)

    # 调用函数 dup_count_real_roots，用于计算多项式 f 在指定区间内的实根个数
    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        return dup_count_real_roots(f._rep, f.dom, inf=inf, sup=sup)

    # 调用函数 dup_count_complex_roots，用于计算多项式 f 在指定区间内的复根个数
    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        return dup_count_complex_roots(f._rep, f.dom, inf=inf, sup=sup)

    # 定义属性方法 is_zero，用于判断多项式 f 是否为零多项式
    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        return dmp_zero_p(f._rep, f.lev)

    # 定义属性方法 is_one，用于判断多项式 f 是否为单位多项式
    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        return dmp_one_p(f._rep, f.lev, f.dom)

    # 定义属性方法 is_ground，用于判断多项式 f 是否为地面域中的元素
    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return dmp_ground_p(f._rep, None, f.lev)

    # 定义属性方法 is_sqf，用于判断多项式 f 是否为平方自由多项式
    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        return dmp_sqf_p(f._rep, f.lev, f.dom)

    # 定义属性方法 is_monic，用于判断多项式 f 的首项系数是否为 1
    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        return f.dom.is_one(dmp_ground_LC(f._rep, f.lev, f.dom))

    # 定义属性方法 is_primitive，用于判断多项式 f 的系数的最大公约数是否为 1
    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        return f.dom.is_one(dmp_ground_content(f._rep, f.lev, f.dom))

    # 定义属性方法 is_linear，用于判断多项式 f 是否在所有变量上是线性的
    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        return all(sum(monom) <= 1 for monom in dmp_to_dict(f._rep, f.lev, f.dom).keys())

    # 定义属性方法 is_quadratic，用于判断多项式 f 是否在所有变量上是二次的
    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        return all(sum(monom) <= 2 for monom in dmp_to_dict(f._rep, f.lev, f.dom).keys())

    # 定义属性方法 is_monomial，用于判断多项式 f 是否为零多项式或者只有一个项
    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        return len(f.to_dict()) <= 1

    # 定义属性方法 is_homogeneous，用于判断多项式 f 是否为齐次多项式
    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        return f.homogeneous_order() is not None

    # 定义属性方法 is_irreducible，用于判断多项式 f 是否在其域上不可约
    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        return dmp_irreducible_p(f._rep, f.lev, f.dom)
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        # 检查多项式的级是否为零，如果是，则调用 dup_cyclotomic_p 函数进行判定
        if not f.lev:
            return dup_cyclotomic_p(f._rep, f.dom)
        else:
            # 如果多项式的级不为零，则返回 False，表示不是环多项式
            return False
class DUP_Flint(DMP):
    """Dense Multivariate Polynomials over `K`. """

    lev = 0  # 初始化 lev 为 0

    __slots__ = ('_rep', 'dom', '_cls')  # 定义实例的槽，限制只能有 _rep, dom, _cls 这三个属性

    def __reduce__(self):
        # 返回对象在 pickle 时的自定义序列化方式，包括使用 to_list() 转换为列表表示
        return self.__class__, (self.to_list(), self.dom, self.lev)

    @classmethod
    def _new(cls, rep, dom, lev):
        # 创建一个新的 DUP_Flint 对象，使用 _flint_poly 方法将 rep 转换为 flint_poly 类型
        rep = cls._flint_poly(rep[::-1], dom, lev)
        return cls.from_rep(rep, dom)

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return f._rep.coeffs()[::-1]  # 返回多项式的系数列表，颠倒顺序以匹配 flint 的表示

    @classmethod
    def _flint_poly(cls, rep, dom, lev):
        # 使用 flint 库创建 flint_poly 对象，并进行一些断言检查
        assert dom in _flint_domains
        assert lev == 0
        flint_cls = cls._get_flint_poly_cls(dom)
        return flint_cls(rep)

    @classmethod
    def _get_flint_poly_cls(cls, dom):
        # 根据 dom 类型选择合适的 flint 多项式类
        if dom.is_ZZ:
            return flint.fmpz_poly
        elif dom.is_QQ:
            return flint.fmpq_poly
        else:
            raise RuntimeError("Domain %s is not supported with flint" % dom)

    @classmethod
    def from_rep(cls, rep, dom):
        """Create a DMP from the given representation. """
        # 根据给定的表示创建 DUP_Flint 对象
        if dom.is_ZZ:
            assert isinstance(rep, flint.fmpz_poly)
            _cls = flint.fmpz_poly
        elif dom.is_QQ:
            assert isinstance(rep, flint.fmpq_poly)
            _cls = flint.fmpq_poly
        else:
            raise RuntimeError("Domain %s is not supported with flint" % dom)

        obj = object.__new__(cls)  # 创建一个新的实例
        obj.dom = dom  # 设置对象的 dom 属性
        obj._rep = rep  # 设置对象的 _rep 属性
        obj._cls = _cls  # 设置对象的 _cls 属性

        return obj

    def _strict_eq(f, g):
        # 比较两个多项式是否严格相等
        if type(f) != type(g):
            return False
        return f.dom == g.dom and f._rep == g._rep

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        # 构造一个新的基础实例，使用给定的系数 coeff
        return f.from_rep(f._cls([coeff]), f.dom)

    def _one(f):
        # 返回多项式的单位元
        return f.ground_new(f.dom.one)

    def unify(f, g):
        """Unify representations of two polynomials. """
        # 统一两个多项式的表示，抛出运行时异常
        raise RuntimeError

    def to_DMP_Python(f):
        """Convert ``f`` to a Python native representation. """
        # 将多项式转换为 Python 的本地表示
        return DMP_Python._new(f.to_list(), f.dom, f.lev)

    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
        # 将多项式转换为元组表示，使用本地系数
        return tuple(f.to_list())

    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
        # 转换多项式的基础域
        if dom == QQ and f.dom == ZZ:
            return f.from_rep(flint.fmpq_poly(f._rep), dom)
        elif dom == ZZ and f.dom == QQ:
            # XXX: python-flint should provide a faster way to do this.
            return f.to_DMP_Python()._convert(dom).to_DUP_Flint()
        else:
            raise RuntimeError(f"DUP_Flint: Cannot convert {f.dom} to {dom}")

    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
        # 返回多项式中从 m 到 n 的连续子序列
        coeffs = f._rep.coeffs()[m:n]
        return f.from_rep(f._cls(coeffs), f.dom)
    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
        # 只对多变量多项式有意义
        raise NotImplementedError

    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        if order is None or order.alias == 'lex':
            # 如果没有指定排序方式或者排序方式是词典序（lex）
            terms = [ ((n,), c) for n, c in enumerate(f._rep.coeffs()) if c ]
            return terms[::-1]  # 返回按词典序倒序排列的所有非零项
        else:
            # XXX: InverseOrder (ilex) comes here. We could handle that case
            # efficiently by reversing the coefficients but it is not clear
            # how to test if the order is InverseOrder.
            #
            # Otherwise why would the order ever be different for univariate
            # polynomials?
            # 如果是 InverseOrder (ilex) 排序，则处理这种情况（暂未实现）；
            # 对于单变量多项式，不清楚为什么需要不同的排序方式
            return f.to_DMP_Python()._terms(order=order)

    def _lift(f):
        """Convert algebraic coefficients to rationals. """
        # 用于将代数系数转换为有理数
        raise NotImplementedError

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        # XXX: Check because otherwise this segfaults with python-flint:
        #
        #  >>> flint.fmpz_poly([]).deflation()
        #  Exception (fmpz_poly_deflate). Division by zero.
        #  Aborted (core dumped
        #
        if f.is_zero:
            return (1,), f
        g, n = f._rep.deflation()
        return (n,), f.from_rep(g, f.dom)  # 返回降低了次数的多项式

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        # 地面域需要是一个多项式环
        raise NotImplementedError

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        # 只对多变量多项式有意义
        raise NotImplementedError

    def _exclude(f):
        """Remove useless generators from ``f``. """
        # 只对多变量多项式有意义
        raise NotImplementedError

    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
        # 只对多变量多项式有意义
        raise NotImplementedError

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        # XXX: python-flint should have primitive, content, etc methods.
        J, F = f.to_DMP_Python().terms_gcd()
        return J, F.to_DUP_Flint()  # 返回去除多项式项的最大公因子后的结果

    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.from_rep(f._rep + c, f.dom)  # 将地面域的元素添加到多项式 f 中

    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.from_rep(f._rep - c, f.dom)  # 从多项式 f 中减去地面域的元素

    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f.from_rep(f._rep * c, f.dom)  # 将多项式 f 乘以地面域的元素

    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f.from_rep(f._rep // c, f.dom)  # 将多项式 f 除以地面域的元素得到商
    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by an element of the ground domain."""
        # Perform division of the internal representation of f by c
        q, r = divmod(f._rep, c)
        # If there's a non-zero remainder, raise an exception
        if r:
            raise ExactQuotientFailed(f, c)
        # Return the result as a new polynomial
        return f.from_rep(q, f.dom)

    def abs(f):
        """Make all coefficients in ``f`` positive."""
        # Convert f to a Python dict format, apply absolute value, then convert back
        return f.to_DMP_Python().abs().to_DUP_Flint()

    def neg(f):
        """Negate all coefficients in ``f``."""
        # Negate the internal representation of f and return as a new polynomial
        return f.from_rep(-f._rep, f.dom)

    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``."""
        # Add the internal representations of f and g, return as a new polynomial
        return f.from_rep(f._rep + g._rep, f.dom)

    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``."""
        # Subtract the internal representations of g from f, return as a new polynomial
        return f.from_rep(f._rep - g._rep, f.dom)

    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``."""
        # Multiply the internal representations of f and g, return as a new polynomial
        return f.from_rep(f._rep * g._rep, f.dom)

    def sqr(f):
        """Square a multivariate polynomial ``f``."""
        # Square the internal representation of f, return as a new polynomial
        return f.from_rep(f._rep ** 2, f.dom)

    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``."""
        # Raise the internal representation of f to the power of n, return as a new polynomial
        return f.from_rep(f._rep ** n, f.dom)

    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``."""
        # Calculate degree difference between f and g
        d = f.degree() - g.degree() + 1
        # Perform pseudo-division of f and g, return quotient and remainder as new polynomials
        q, r = divmod(g.LC()**d * f._rep, g._rep)
        return f.from_rep(q, f.dom), f.from_rep(r, f.dom)

    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``."""
        # Calculate degree difference between f and g
        d = f.degree() - g.degree() + 1
        # Compute pseudo-remainder of f and g, return as a new polynomial
        q = (g.LC()**d * f._rep) % g._rep
        return f.from_rep(q, f.dom)

    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``."""
        # Calculate degree difference between f and g
        d = f.degree() - g.degree() + 1
        # Compute pseudo-quotient of f and g, return as a new polynomial
        r = (g.LC()**d * f._rep) // g._rep
        return f.from_rep(r, f.dom)

    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``."""
        # Calculate degree difference between f and g
        d = f.degree() - g.degree() + 1
        # Perform exact pseudo-quotient of f and g, raise exception if remainder is non-zero
        q, r = divmod(g.LC()**d * f._rep, g._rep)
        if r:
            raise ExactQuotientFailed(f, g)
        return f.from_rep(q, f.dom)

    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``."""
        if f.dom.is_Field:
            # If the domain is a field, perform division directly
            q, r = divmod(f._rep, g._rep)
            return f.from_rep(q, f.dom), f.from_rep(r, f.dom)
        else:
            # Otherwise, handle division differently for a specific case
            # XXX: python-flint defines division in ZZ[x] differently
            q, r = f.to_DMP_Python()._div(g.to_DMP_Python())
            return q.to_DUP_Flint(), r.to_DUP_Flint()

    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``."""
        # Compute remainder of f divided by g, return as a new polynomial
        return f.from_rep(f._rep % g._rep, f.dom)

    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``."""
        # Compute quotient of f divided by g, return as a new polynomial
        return f.from_rep(f._rep // g._rep, f.dom)

    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``."""
        # Compute exact quotient of f divided by g, raise exception if remainder is non-zero
        q, r = f._div(g)
        if r:
            raise ExactQuotientFailed(f, g)
        return q
    def _degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        # 获取多项式 ``f`` 在变量 ``x_j`` 中的主导次数
        d = f._rep.degree()
        # 如果次数为 -1，则将其视为负无穷大
        if d == -1:
            d = ninf
        return d

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        # 返回多项式 ``f`` 的次数列表
        return ( f._degree() ,)

    def total_degree(f):
        """Returns the total degree of ``f``. """
        # 返回多项式 ``f`` 的总次数
        return f._degree()

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        # 返回多项式 ``f`` 的主导系数
        return f._rep[f._rep.degree()]

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        # 返回多项式 ``f`` 的尾随系数
        return f._rep[0]

    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
        [n] = N
        # 返回多项式 ``f`` 的第 ``n`` 个系数
        return f._rep[n]

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        # 返回多项式 ``f`` 的最大范数
        return f.to_DMP_Python().max_norm()

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        # 返回多项式 ``f`` 的 l1 范数
        return f.to_DMP_Python().l1_norm()

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        # 返回多项式 ``f`` 的 l2 范数的平方
        return f.to_DMP_Python().l2_norm_squared()

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        # 清除多项式 ``f`` 的分母，但保留其基础域
        denom = f._rep.denom()
        numer = f.from_rep(f._cls(f._rep.numer()), f.dom)
        return denom, numer

    def _integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        # 计算多项式 ``f`` 在变量 ``x_j`` 中的 ``m`` 阶不定积分
        assert j == 0
        if f.dom.is_QQ:
            rep = f._rep
            for i in range(m):
                rep = rep.integral()
            return f.from_rep(rep, f.dom)
        else:
            return f.to_DMP_Python()._integrate(m=m, j=j).to_DUP_Flint()

    def _diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f``. """
        # 计算多项式 ``f`` 的 ``m`` 阶导数
        assert j == 0
        rep = f._rep
        for i in range(m):
            rep = rep.derivative()
        return f.from_rep(rep, f.dom)

    def _eval(f, a):
        # Evaluate polynomial ``f`` at ``a``
        return f.to_DMP_Python()._eval(a)

    def _eval_lev(f, a, j):
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm. """
        # 半扩展欧几里得算法，返回多项式 ``f`` 和 ``g`` 的半扩展欧几里得结果
        s, h = f.to_DMP_Python()._half_gcdex(g.to_DMP_Python())
        return s.to_DUP_Flint(), h.to_DUP_Flint()

    def _gcdex(f, g):
        """Extended Euclidean algorithm. """
        # 扩展欧几里得算法，返回多项式 ``f`` 和 ``g`` 的扩展欧几里得结果
        h, s, t = f._rep.xgcd(g._rep)
        return f.from_rep(s, f.dom), f.from_rep(t, f.dom), f.from_rep(h, f.dom)

    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        # 尝试在模 ``g`` 的情况下反转多项式 ``f``
        if f.dom.is_QQ:
            gcd, F_inv, _ = f._rep.xgcd(g._rep)
            if gcd != 1:
                raise NotInvertible("zero divisor")
            return f.from_rep(F_inv, f.dom)
        else:
            return f.to_DMP_Python()._invert(g.to_DMP_Python()).to_DUP_Flint()

    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        # 计算 ``f**(-1)`` 对 ``x**n`` 取模的结果
        return f.to_DMP_Python()._revert(n).to_DUP_Flint()
    def _subresultants(f, g):
        """计算多项式 f 和 g 的子结果式 PRS 序列。"""
        # 将 f 和 g 转换为 Python-DMP 形式，并计算它们的子结果式 PRS 序列
        R = f.to_DMP_Python()._subresultants(g.to_DMP_Python())
        # 返回结果，其中每个元素都转换为 Flint-DUP 形式
        return [ g.to_DUP_Flint() for g in R ]

    def _resultant_includePRS(f, g):
        """通过 PRS 计算多项式 f 和 g 的结果式。"""
        # 将 f 和 g 转换为 Python-DMP 形式，并通过 PRS 计算它们的结果式和 PRS 序列
        res, R = f.to_DMP_Python()._resultant_includePRS(g.to_DMP_Python())
        # 返回结果式和 PRS 序列中每个元素转换为 Flint-DUP 形式
        return res, [ g.to_DUP_Flint() for g in R ]

    def _resultant(f, g):
        """计算多项式 f 和 g 的结果式。"""
        # 将 f 和 g 转换为 Python-DMP 形式，并计算它们的结果式
        return f.to_DMP_Python()._resultant(g.to_DMP_Python())

    def discriminant(f):
        """计算多项式 f 的判别式。"""
        # 将 f 转换为 Python-DMP 形式，并计算其判别式
        return f.to_DMP_Python().discriminant()

    def _cofactors(f, g):
        """返回多项式 f 和 g 的最大公因式及它们的因式。"""
        # 计算 f 和 g 的最大公因式
        h = f.gcd(g)
        # 返回最大公因式 h 及 f 和 g 除以 h 的结果
        return h, f.exquo(h), g.exquo(h)

    def _gcd(f, g):
        """返回多项式 f 和 g 的最大公因式。"""
        # 使用 f 和 g 的表示形式的最大公因式计算结果，并转换为 f 的域
        return f.from_rep(f._rep.gcd(g._rep), f.dom)

    def _lcm(f, g):
        """返回多项式 f 和 g 的最小公倍式。"""
        # XXX: python-flint 应该有一个 lcm 方法
        if not (f and g):
            return f.ground_new(f.dom.zero)

        # 计算 f 和 g 的乘积，并除以它们的最大公因式
        l = f._mul(g)._exquo(f._gcd(g))

        # 如果 f 和 g 的域是一个域，则将结果转换为首一多项式
        if l.dom.is_Field:
            l = l.monic()
        # 如果结果的首项系数小于 0，则取其相反数
        elif l.LC() < 0:
            l = l.neg()

        return l

    def _cancel(f, g):
        """取消有理函数 f/g 的公因子。"""
        # 如果域不是 ZZ 或 QQ，则抛出断言错误
        assert f.dom == g.dom in (ZZ, QQ)

        if f.dom.is_QQ:
            # 如果域是 QQ，则分别将 f 和 g 清除分母
            cG, F = f.clear_denoms()
            cF, G = g.clear_denoms()
        else:
            # 如果域是 ZZ，则设置 cG, F, cF, G 为默认值
            cG, F = f.dom.one, f
            cF, G = g.dom.one, g

        # 计算 cF 和 cG 的最大公因子
        cH = cF.gcd(cG)
        # 将 cF 和 cG 除以它们的最大公因子
        cF, cG = cF // cH, cG // cH

        # 计算 F 和 G 的最大公因式
        H = F._gcd(G)
        # 将 F 和 G 分别除以它们的最大公因式
        F, G = F.exquo(H), G.exquo(H)

        # 检查 F 和 G 的首项系数是否小于 0，若是，则取其相反数
        f_neg = F.LC() < 0
        g_neg = G.LC() < 0

        if f_neg and g_neg:
            F, G = F.neg(), G.neg()
        elif f_neg:
            cF, F = -cF, F.neg()
        elif g_neg:
            cF, G = -cF, G.neg()

        # 返回结果
        return cF, cG, F, G

    def _cancel_include(f, g):
        """取消有理函数 f/g 的公因子。"""
        # 调用 _cancel 方法取消 f 和 g 的公因子
        cF, cG, F, G = f._cancel(g)
        # 返回取消公因子后的 F 和 G
        return F._mul_ground(cF), G._mul_ground(cG)

    def _trunc(f, p):
        """将多项式 f 模除常数 p。"""
        # 将 f 转换为 Python-DMP 形式，并将其模除常数 p，然后转换为 Flint-DUP 形式
        return f.to_DMP_Python()._trunc(p).to_DUP_Flint()

    def monic(f):
        """将多项式 f 的所有系数除以 LC(f)。"""
        # 将 f 的首项系数除以 LC(f)
        return f._exquo_ground(f.LC())

    def content(f):
        """返回多项式系数的最大公因数。"""
        # XXX: python-flint 应该有一个 content 方法
        # 将 f 转换为 Python-DMP 形式，并计算其系数的最大公因数
        return f.to_DMP_Python().content()

    def primitive(f):
        """返回多项式的最大公因数及其原始形式。"""
        # 计算 f 的最大公因数
        cont = f.content()
        # 将 f 除以其最大公因数，得到原始形式的多项式
        prim = f._exquo_ground(cont)
        # 返回最大公因数及原始形式
        return cont, prim
    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        # 使用 f 的 _rep 方法和 g 的 _rep 方法来计算函数组合
        return f.from_rep(f._rep(g._rep), f.dom)

    def _decompose(f):
        """Computes functional decomposition of ``f``. """
        # 将函数 f 转换成多项式表示，然后进行分解为 DUP_Flint 对象的列表
        return [ g.to_DUP_Flint() for g in f.to_DMP_Python()._decompose() ]

    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        # 构造 x + a 的多项式表示，并利用 from_rep 方法进行变换计算
        x_plus_a = f._cls([a, f.dom.one])
        return f.from_rep(f._rep(x_plus_a), f.dom)

    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        # 将函数 f、p 和 q 转换为多项式表示，然后进行函数变换计算并返回 DUP_Flint 对象
        F, P, Q = f.to_DMP_Python(), p.to_DMP_Python(), q.to_DMP_Python()
        return F.transform(P, Q).to_DUP_Flint()

    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
        # 将函数 f 转换成多项式表示，然后计算其 Sturm 序列并返回 DUP_Flint 对象的列表
        return [ g.to_DUP_Flint() for g in f.to_DMP_Python()._sturm() ]

    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        # 将函数 f 转换成多项式表示，然后计算其根的 Cauchy 上界
        return f.to_DMP_Python()._cauchy_upper_bound()

    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        # 将函数 f 转换成多项式表示，然后计算其非零根的 Cauchy 下界
        return f.to_DMP_Python()._cauchy_lower_bound()

    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        # 将函数 f 转换成多项式表示，然后计算其根分离的 Mignotte 平方界
        return f.to_DMP_Python()._mignotte_sep_bound_squared()

    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        # 将函数 f 转换成多项式表示，然后计算其最大阶乘因子分解，并返回 DUP_Flint 对象和整数对的列表
        F = f.to_DMP_Python()
        return [ (g.to_DUP_Flint(), k) for g, k in F.gff_list() ]

    def norm(f):
        """Computes ``Norm(f)``."""
        # 对于 DUP_Flint 不支持的代数数域，抛出未实现错误
        raise NotImplementedError

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        # 对于 DUP_Flint 不支持的代数数域，抛出未实现错误
        raise NotImplementedError

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        # 计算函数 f 的平方自由部分，返回 DUP_Flint 对象
        return f._exquo(f._gcd(f._diff()))

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        # 将函数 f 转换成多项式表示，然后计算其平方自由因子的列表，返回系数和 DUP_Flint 对象的列表
        coeff, factors = f.to_DMP_Python().sqf_list(all=all)
        return coeff, [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        # 将函数 f 转换成多项式表示，然后计算其包含系数的平方自由因子的列表，返回 DUP_Flint 对象和整数对的列表
        factors = f.to_DMP_Python().sqf_list_include(all=all)
        return [ (g.to_DUP_Flint(), k) for g, k in factors ]
    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """

        if f.dom.is_ZZ:
            # 如果定义域是整数环（ZZ），使用 python-flint 匹配多项式
            coeff, factors = f._rep.factor()
            # 转换每个因子到多项式表示形式
            factors = [ (f.from_rep(g, f.dom), k) for g, k in factors ]

        elif f.dom.is_QQ:
            # 如果定义域是有理数环（QQ），python-flint 返回单位首一的因子，而 polys 返回去除分母的因子
            coeff, factors = f._rep.factor()
            factors_monic = [ (f.from_rep(g, f.dom), k) for g, k in factors ]

            # 将分母吸收到 coeff 中
            factors = []
            for g, k in factors_monic:
                d, g = g.clear_denoms()
                coeff /= d**k
                factors.append((g, k))

        else:
            # 在添加更多定义域时需要小心...
            raise RuntimeError("Domain %s is not supported with flint" % f.dom)

        # 需要按照 polys 的方式排序因子
        factors = f._sort_factors(factors)

        return coeff, factors

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        # XXX: factor_list_include 在一般情况下似乎存在问题：
        #
        #   >>> Poly(2*(x - 1)**3, x).factor_list_include()
        #   [(Poly(2*x - 2, x, domain='ZZ'), 3)]
        #
        # 在这里我们不尝试实现它。
        # 使用 to_DMP_Python 将 f 转换成多项式，然后调用 factor_list_include 方法获取因子列表
        factors = f.to_DMP_Python().factor_list_include()
        return [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def _sort_factors(f, factors):
        """Sort a list of factors to canonical order. """
        # 将因子转换为列表形式，并使用 polys 的 _sort_factors 函数排序
        factors = [ (g.to_list(), k) for g, k in factors ]
        factors = _sort_factors(factors, multiple=True)
        to_dup_flint = lambda g: f.from_rep(f._cls(g[::-1]), f.dom)
        return [ (to_dup_flint(g), k) for g, k in factors ]

    def _isolate_real_roots(f, eps, inf, sup, fast):
        # 调用 to_DMP_Python 转换 f 到多项式，然后调用 _isolate_real_roots 方法
        return f.to_DMP_Python()._isolate_real_roots(eps, inf, sup, fast)

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        # 调用 to_DMP_Python 转换 f 到多项式，然后调用 _isolate_real_roots_sqf 方法
        return f.to_DMP_Python()._isolate_real_roots_sqf(eps, inf, sup, fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        # 调用 to_DMP_Python 转换 f 到多项式，然后调用 _isolate_all_roots 方法
        return f.to_DMP_Python()._isolate_all_roots(eps, inf, sup, fast)

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        # 调用 to_DMP_Python 转换 f 到多项式，然后调用 _isolate_all_roots_sqf 方法
        return f.to_DMP_Python()._isolate_all_roots_sqf(eps, inf, sup, fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        # 调用 to_DMP_Python 转换 f 到多项式，然后调用 _refine_real_root 方法
        return f.to_DMP_Python()._refine_real_root(s, t, eps, steps, fast)

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        # 调用 to_DMP_Python 转换 f 到多项式，然后调用 count_real_roots 方法
        return f.to_DMP_Python().count_real_roots(inf=inf, sup=sup)

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        # 调用 to_DMP_Python 转换 f 到多项式，然后调用 count_complex_roots 方法
        return f.to_DMP_Python().count_complex_roots(inf=inf, sup=sup)

    @property
    # 这是一个装饰器，用于定义一个属性
    # 返回 ``True`` 如果 ``f`` 是零多项式。
    def is_zero(f):
        return not f._rep

    # 返回 ``True`` 如果 ``f`` 是单位多项式。
    @property
    def is_one(f):
        return f._rep == f.dom.one

    # 返回 ``True`` 如果 ``f`` 是属于基础域的元素。
    @property
    def is_ground(f):
        return f._rep.degree() <= 0

    # 返回 ``True`` 如果 ``f`` 在所有变量中是线性的。
    @property
    def is_linear(f):
        return f._rep.degree() <= 1

    # 返回 ``True`` 如果 ``f`` 在所有变量中是二次的。
    @property
    def is_quadratic(f):
        return f._rep.degree() <= 2

    # 返回 ``True`` 如果 ``f`` 是零多项式或者只有一个项。
    @property
    def is_monomial(f):
        return f.to_DMP_Python().is_monomial

    # 返回 ``True`` 如果 ``f`` 的主导系数是一。
    @property
    def is_monic(f):
        return f.LC() == f.dom.one

    # 返回 ``True`` 如果 ``f`` 的系数的最大公约数是一。
    @property
    def is_primitive(f):
        return f.to_DMP_Python().is_primitive

    # 返回 ``True`` 如果 ``f`` 是齐次多项式。
    @property
    def is_homogeneous(f):
        return f.to_DMP_Python().is_homogeneous

    # 返回 ``True`` 如果 ``f`` 是无平方因子的多项式。
    @property
    def is_sqf(f):
        return f.to_DMP_Python().is_sqf

    # 返回 ``True`` 如果 ``f`` 在其域上没有因子。
    @property
    def is_irreducible(f):
        return f.to_DMP_Python().is_irreducible

    # 返回 ``True`` 如果 ``f`` 是一个旋转多项式。
    @property
    def is_cyclotomic(f):
        if f.dom.is_ZZ:
            return bool(f._rep.is_cyclotomic())
        else:
            return f.to_DMP_Python().is_cyclotomic
# 创建一个初始化 DMF 对象的函数，接受数字的分子、分母、级别和定义域作为参数
def init_normal_DMF(num, den, lev, dom):
    # 调用 dmp_normal 函数分别处理分子和分母，返回处理后的结果，并将其作为参数传递给 DMF 类的构造函数
    return DMF(dmp_normal(num, lev, dom),
               dmp_normal(den, lev, dom), dom, lev)

# 定义 DMF 类，继承自 PicklableWithSlots 和 CantSympify
class DMF(PicklableWithSlots, CantSympify):
    """Dense Multivariate Fractions over `K`. """

    # 设置 __slots__ 属性以优化内存使用，仅包含 'num', 'den', 'lev', 'dom' 四个实例变量
    __slots__ = ('num', 'den', 'lev', 'dom')

    # DMF 类的构造函数，接受 rep、dom 和 lev 三个参数
    def __init__(self, rep, dom, lev=None):
        # 调用 _parse 方法解析 rep，得到分子 num、分母 den 和级别 lev
        num, den, lev = self._parse(rep, dom, lev)
        # 调用 dmp_cancel 函数对 num 和 den 进行约简操作，lev 和 dom 作为参数传递
        num, den = dmp_cancel(num, den, lev, dom)

        # 将处理后的 num、den、lev 和 dom 分别赋值给实例变量
        self.num = num
        self.den = den
        self.lev = lev
        self.dom = dom

    # 类方法 new，用于创建新的 DMF 对象，接受 rep、dom 和 lev 三个参数
    @classmethod
    def new(cls, rep, dom, lev=None):
        # 调用 _parse 方法解析 rep，得到分子 num、分母 den 和级别 lev
        num, den, lev = cls._parse(rep, dom, lev)

        # 使用 object.__new__ 方法创建一个新的 DMF 类对象
        obj = object.__new__(cls)

        # 将解析后的 num、den、lev 和 dom 分别赋值给新对象的实例变量
        obj.num = num
        obj.den = den
        obj.lev = lev
        obj.dom = dom

        # 返回创建的对象
        return obj

    # ground_new 方法，接受 rep 参数，使用当前对象的 dom 和 lev 创建新的 DMF 对象
    def ground_new(self, rep):
        return self.new(rep, self.dom, self.lev)

    # 类方法 _parse，用于解析 rep，接受 rep、dom 和 lev 三个参数
    @classmethod
    def _parse(cls, rep, dom, lev=None):
        # 如果 rep 是元组，则分别解析出 num 和 den
        if isinstance(rep, tuple):
            num, den = rep

            # 如果 lev 不为 None，且 num 是字典，则调用 dmp_from_dict 将 num 转换为多项式表示
            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)

                # 同样地，如果 den 是字典，则调用 dmp_from_dict 将 den 转换为多项式表示
                if isinstance(den, dict):
                    den = dmp_from_dict(den, lev, dom)
            else:
                # 如果 lev 是 None，则调用 dmp_validate 验证 num 和 den，并获取它们的级别
                num, num_lev = dmp_validate(num)
                den, den_lev = dmp_validate(den)

                # 如果 num 和 den 的级别相同，则将 lev 设置为 num 的级别
                if num_lev == den_lev:
                    lev = num_lev
                else:
                    # 如果级别不同，则抛出 ValueError 异常
                    raise ValueError('inconsistent number of levels')

            # 如果 den 是零多项式，则抛出 ZeroDivisionError 异常
            if dmp_zero_p(den, lev):
                raise ZeroDivisionError('fraction denominator')

            # 如果 num 是零多项式，则将 den 设置为单位多项式
            if dmp_zero_p(num, lev):
                den = dmp_one(lev, dom)
            else:
                # 如果 den 是负多项式，则将 num 和 den 都取负
                if dmp_negative_p(den, lev, dom):
                    num = dmp_neg(num, lev, dom)
                    den = dmp_neg(den, lev, dom)
        else:
            # 如果 rep 不是元组，则将 rep 赋给 num
            num = rep

            # 如果 lev 不为 None，且 num 是字典，则调用 dmp_from_dict 将 num 转换为多项式表示
            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)
                # 如果 num 不是列表，则将其转换为基础类型的多项式表示
                elif not isinstance(num, list):
                    num = dmp_ground(dom.convert(num), lev)
            else:
                # 如果 lev 是 None，则调用 dmp_validate 验证 num，并获取它的级别
                num, lev = dmp_validate(num)

            # 将 den 设置为单位多项式
            den = dmp_one(lev, dom)

        # 返回解析后的 num、den 和 lev
        return num, den, lev

    # 定义对象的字符串表示形式
    def __repr__(f):
        return "%s((%s, %s), %s)" % (f.__class__.__name__, f.num, f.den, f.dom)

    # 定义对象的哈希值
    def __hash__(f):
        return hash((f.__class__.__name__, dmp_to_tuple(f.num, f.lev),
            dmp_to_tuple(f.den, f.lev), f.lev, f.dom))
    def poly_unify(f, g):
        """Unify a multivariate fraction and a polynomial. """
        # 检查 g 是否为 DMP 类型且与 f 的层次相同，否则抛出异常
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        # 如果 f 的域与 g 的域相同，返回 f 的层次、域、per 函数、(f.num, f.den) 和 g._rep
        if f.dom == g.dom:
            return (f.lev, f.dom, f.per, (f.num, f.den), g._rep)
        else:
            # 否则，计算新的层次 lev 和域 dom
            lev, dom = f.lev, f.dom.unify(g.dom)

            # 将 f 的分子和分母转换到新的域 dom
            F = (dmp_convert(f.num, lev, f.dom, dom),
                 dmp_convert(f.den, lev, f.dom, dom))

            # 将 g 的表示 g._rep 转换到新的域 dom
            G = dmp_convert(g._rep, lev, g.dom, dom)

            # 定义 per 函数，用于创建新的多变量分数对象
            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num/den
                    else:
                        lev = lev - 1

                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)

                return f.__class__.new((num, den), dom, lev)

            return lev, dom, per, F, G

    def frac_unify(f, g):
        """Unify representations of two multivariate fractions. """
        # 检查 g 是否为 DMF 类型且与 f 的层次相同，否则抛出异常
        if not isinstance(g, DMF) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        # 如果 f 的域与 g 的域相同，返回 f 的层次、域、per 函数、(f.num, f.den) 和 (g.num, g.den)
        if f.dom == g.dom:
            return (f.lev, f.dom, f.per, (f.num, f.den), (g.num, g.den))
        else:
            # 否则，计算新的层次 lev 和域 dom
            lev, dom = f.lev, f.dom.unify(g.dom)

            # 将 f 的分子和分母转换到新的域 dom
            F = (dmp_convert(f.num, lev, f.dom, dom),
                 dmp_convert(f.den, lev, f.dom, dom))

            # 将 g 的分子和分母转换到新的域 dom
            G = (dmp_convert(g.num, lev, g.dom, dom),
                 dmp_convert(g.den, lev, g.dom, dom))

            # 定义 per 函数，用于创建新的多变量分数对象
            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num/den
                    else:
                        lev = lev - 1

                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)

                return f.__class__.new((num, den), dom, lev)

            return lev, dom, per, F, G

    def per(f, num, den, cancel=True, kill=False):
        """Create a DMF out of the given representation. """
        # 获取 f 的层次和域
        lev, dom = f.lev, f.dom

        # 如果 kill 为真且层次 lev 为 0，则返回 num/den
        if kill:
            if not lev:
                return num/den
            else:
                lev -= 1

        # 如果 cancel 为真，取消 num 和 den 的公因式
        if cancel:
            num, den = dmp_cancel(num, den, lev, dom)

        # 使用 f 类的构造函数创建新的多变量分数对象
        return f.__class__.new((num, den), dom, lev)

    def half_per(f, rep, kill=False):
        """Create a DMP out of the given representation. """
        # 获取 f 的层次
        lev = f.lev

        # 如果 kill 为真且层次 lev 为 0，则直接返回 rep
        if kill:
            if not lev:
                return rep
            else:
                lev -= 1

        # 使用 DMP 类创建新的多项式对象
        return DMP(rep, f.dom, lev)

    @classmethod
    def zero(cls, lev, dom):
        # 类方法：返回一个新的 cls 类对象，表示层次 lev 和域 dom 的零元素
        return cls.new(0, dom, lev)

    @classmethod
    def one(cls, lev, dom):
        # 类方法：返回一个新的 cls 类对象，表示层次 lev 和域 dom 的单位元素
        return cls.new(1, dom, lev)

    def numer(f):
        """Returns the numerator of ``f``. """
        # 返回 f 对象的分子部分
        return f.half_per(f.num)
    def denom(f):
        """Returns the denominator of ``f``. """
        # 返回分式 ``f`` 的分母
        return f.half_per(f.den)

    def cancel(f):
        """Remove common factors from ``f.num`` and ``f.den``. """
        # 从 ``f.num`` 和 ``f.den`` 中移除公因子
        return f.per(f.num, f.den)

    def neg(f):
        """Negate all coefficients in ``f``. """
        # 对 ``f`` 中的所有系数取负
        return f.per(dmp_neg(f.num, f.lev, f.dom), f.den, cancel=False)

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        # 将一个地面域的元素加到 ``f`` 中
        return f + f.ground_new(c)

    def add(f, g):
        """Add two multivariate fractions ``f`` and ``g``. """
        # 添加两个多变量分式 ``f`` 和 ``g``
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_add_mul(F_num, F_den, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_add(dmp_mul(F_num, G_den, lev, dom),
                          dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def sub(f, g):
        """Subtract two multivariate fractions ``f`` and ``g``. """
        # 减去两个多变量分式 ``f`` 和 ``g``
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_sub_mul(F_num, F_den, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_sub(dmp_mul(F_num, G_den, lev, dom),
                          dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def mul(f, g):
        """Multiply two multivariate fractions ``f`` and ``g``. """
        # 乘以两个多变量分式 ``f`` 和 ``g``
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_mul(F_num, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_mul(F_num, G_num, lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        # 将 ``f`` 的非负整数次幂为 ``n``
        if isinstance(n, int):
            num, den = f.num, f.den
            if n < 0:
                num, den, n = den, num, -n
            return f.per(dmp_pow(num, n, f.lev, f.dom),
                         dmp_pow(den, n, f.lev, f.dom), cancel=False)
        else:
            raise TypeError("``int`` expected, got %s" % type(n))
    def quo(f, g):
        """计算分数 ``f`` 和 ``g`` 的商。"""
        # 如果 g 是 DMP 类型的对象
        if isinstance(g, DMP):
            # 调用 poly_unify 方法统一多项式对象 f 和 g
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            # 计算分子和分母
            num, den = F_num, dmp_mul(F_den, G, lev, dom)
        else:
            # 否则调用 frac_unify 方法统一分数对象 f 和 g
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            # 计算分子和分母
            num = dmp_mul(F_num, G_den, lev, dom)
            den = dmp_mul(F_den, G_num, lev, dom)

        # 返回分数的商
        return per(num, den)

    exquo = quo

    def invert(f, check=True):
        """计算分数 ``f`` 的倒数。"""
        # 调用 per 方法计算分数的倒数
        return f.per(f.den, f.num, cancel=False)

    @property
    def is_zero(f):
        """如果分数 ``f`` 是零返回 ``True``。"""
        # 调用 dmp_zero_p 方法判断分数是否为零
        return dmp_zero_p(f.num, f.lev)

    @property
    def is_one(f):
        """如果分数 ``f`` 是单位分数返回 ``True``。"""
        # 调用 dmp_one_p 方法判断分数是否为单位分数
        return dmp_one_p(f.num, f.lev, f.dom) and \
            dmp_one_p(f.den, f.lev, f.dom)

    def __neg__(f):
        # 调用 neg 方法实现分数的取负操作
        return f.neg()

    def __add__(f, g):
        # 如果 g 是 DMP 或 DMF 类型的对象，则调用 add 方法实现加法
        if isinstance(g, (DMP, DMF)):
            return f.add(g)
        # 如果 g 是 f.dom 中的元素，则转换后调用 add_ground 方法实现加法
        elif g in f.dom:
            return f.add_ground(f.dom.convert(g))

        try:
            # 否则尝试调用 half_per 方法处理 g 后再调用 add 方法实现加法
            return f.add(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            # 处理类型错误、转换失败或未实现错误，返回 NotImplemented
            return NotImplemented

    def __radd__(f, g):
        # 实现反向加法操作
        return f.__add__(g)

    def __sub__(f, g):
        # 如果 g 是 DMP 或 DMF 类型的对象，则调用 sub 方法实现减法
        if isinstance(g, (DMP, DMF)):
            return f.sub(g)

        try:
            # 否则尝试调用 half_per 方法处理 g 后再调用 sub 方法实现减法
            return f.sub(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            # 处理类型错误、转换失败或未实现错误，返回 NotImplemented
            return NotImplemented

    def __rsub__(f, g):
        # 实现反向减法操作
        return (-f).__add__(g)

    def __mul__(f, g):
        # 如果 g 是 DMP 或 DMF 类型的对象，则调用 mul 方法实现乘法
        if isinstance(g, (DMP, DMF)):
            return f.mul(g)

        try:
            # 否则尝试调用 half_per 方法处理 g 后再调用 mul 方法实现乘法
            return f.mul(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            # 处理类型错误、转换失败或未实现错误，返回 NotImplemented
            return NotImplemented

    def __rmul__(f, g):
        # 实现反向乘法操作
        return f.__mul__(g)

    def __pow__(f, n):
        # 实现指数操作
        return f.pow(n)

    def __truediv__(f, g):
        # 如果 g 是 DMP 或 DMF 类型的对象，则调用 quo 方法实现除法
        if isinstance(g, (DMP, DMF)):
            return f.quo(g)

        try:
            # 否则尝试调用 half_per 方法处理 g 后再调用 quo 方法实现除法
            return f.quo(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            # 处理类型错误、转换失败或未实现错误，返回 NotImplemented
            return NotImplemented

    def __rtruediv__(self, g):
        # 实现反向除法操作
        return self.invert(check=False)*g

    def __eq__(f, g):
        try:
            # 如果 g 是 DMP 类型的对象，则调用 poly_unify 方法统一多项式对象 f 和 g
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)

                # 如果 f 和 g 的级数相同，则比较分母是否为单位多项式且分子是否与 G 相等
                if f.lev == g.lev:
                    return dmp_one_p(F_den, f.lev, f.dom) and F_num == G
            else:
                # 否则调用 frac_unify 方法统一分数对象 f 和 g
                _, _, _, F, G = f.frac_unify(g)

                # 如果 f 和 g 的级数相同，则比较 F 和 G 是否相等
                if f.lev == g.lev:
                    return F == G
        except UnificationFailed:
            pass

        # 若出现统一失败或其它异常，则返回 False
        return False
    # 比较两个对象 f 和 g 是否不相等
    def __ne__(f, g):
        try:
            # 如果 g 是 DMP 类型，则尝试多项式统一操作
            if isinstance(g, DMP):
                # 获取多项式统一后的结果
                _, _, _, (F_num, F_den), G = f.poly_unify(g)

                # 如果 f 和 g 的级别相同
                if f.lev == g.lev:
                    # 返回是否不满足 F_den 是单位元并且 F_num 等于 G 的条件
                    return not (dmp_one_p(F_den, f.lev, f.dom) and F_num == G)
            else:
                # 否则尝试分式统一操作
                _, _, _, F, G = f.frac_unify(g)

                # 如果 f 和 g 的级别相同
                if f.lev == g.lev:
                    # 返回 F 是否不等于 G 的结果
                    return F != G
        except UnificationFailed:
            pass

        # 异常情况下，默认返回 True
        return True

    # 比较两个对象 f 和 g 是否小于
    def __lt__(f, g):
        # 获取分式统一操作后的结果
        _, _, _, F, G = f.frac_unify(g)
        # 返回 F 是否小于 G 的结果
        return F < G

    # 比较两个对象 f 和 g 是否小于等于
    def __le__(f, g):
        # 获取分式统一操作后的结果
        _, _, _, F, G = f.frac_unify(g)
        # 返回 F 是否小于等于 G 的结果
        return F <= G

    # 比较两个对象 f 和 g 是否大于
    def __gt__(f, g):
        # 获取分式统一操作后的结果
        _, _, _, F, G = f.frac_unify(g)
        # 返回 F 是否大于 G 的结果
        return F > G

    # 比较两个对象 f 和 g 是否大于等于
    def __ge__(f, g):
        # 获取分式统一操作后的结果
        _, _, _, F, G = f.frac_unify(g)
        # 返回 F 是否大于等于 G 的结果
        return F >= G

    # 判断对象 f 是否为真
    def __bool__(f):
        # 返回对象 f 的 num 属性是否非零
        return not dmp_zero_p(f.num, f.lev)
# 创建一个名为 init_normal_ANP 的函数，接受三个参数 rep、mod 和 dom，并返回一个 ANP 对象
def init_normal_ANP(rep, mod, dom):
    # 调用 ANP 类的构造函数，使用 dup_normal 函数处理 rep 和 mod 参数，并传入 dom 参数
    return ANP(dup_normal(rep, dom),
               dup_normal(mod, dom), dom)


class ANP(CantSympify):
    """Dense Algebraic Number Polynomials over a field. """

    # 使用 __slots__ 定义仅有的实例变量 _rep、_mod 和 dom
    __slots__ = ('_rep', '_mod', 'dom')

    # ANP 类的构造函数，接受 rep、mod 和 dom 三个参数
    def __new__(cls, rep, mod, dom):
        # 如果 rep 是 DMP 类的实例，则保持不变
        if isinstance(rep, DMP):
            pass
        # 如果 rep 是字典类型（不使用 isinstance），则转换为 DMP 类型
        elif type(rep) is dict:
            rep = DMP(dup_from_dict(rep, dom), dom, 0)
        else:
            # 否则，如果 rep 是列表，则逐个转换为 dom 所属的类型
            if isinstance(rep, list):
                rep = [dom.convert(a) for a in rep]
            else:
                rep = [dom.convert(rep)]
            # 将 rep 转换为 DMP 类型
            rep = DMP(dup_strip(rep), dom, 0)

        # 如果 mod 是 DMP 类的实例，则保持不变
        if isinstance(mod, DMP):
            pass
        # 如果 mod 是字典类型，则转换为 DMP 类型
        elif isinstance(mod, dict):
            mod = DMP(dup_from_dict(mod, dom), dom, 0)
        else:
            # 否则，将 mod 转换为 DMP 类型
            mod = DMP(dup_strip(mod), dom, 0)

        # 调用父类的构造函数创建新的 ANP 对象
        return cls.new(rep, mod, dom)

    @classmethod
    # 类方法 new，用于创建并返回一个新的 ANP 对象
    def new(cls, rep, mod, dom):
        # 检查 rep、mod 和 dom 是否属于同一域，若不是则引发 RuntimeError
        if not (rep.dom == mod.dom == dom):
            raise RuntimeError("Inconsistent domain")
        # 使用父类的 __new__ 方法创建 ANP 类的新实例对象
        obj = super().__new__(cls)
        # 初始化实例对象的 _rep、_mod 和 dom 属性
        obj._rep = rep
        obj._mod = mod
        obj.dom = dom
        return obj

    # XXX: 应该能够使用 __getnewargs__ 而不是 __reduce__
    # 但由于某些原因这并不起作用。可能如果 python-flint 支持多项式类型的 pickling 就更容易了。
    # __reduce__ 方法，用于序列化 ANP 对象
    def __reduce__(self):
        return ANP, (self.rep, self.mod, self.dom)

    # 返回 _rep 属性的列表形式
    @property
    def rep(self):
        return self._rep.to_list()

    # 返回 _mod 属性的列表形式
    @property
    def mod(self):
        return self.mod_to_list()  # 此处应该是 self._mod.to_list()，修正为 self.mod_to_list()

    # 返回 _rep 属性
    def to_DMP(self):
        return self._rep

    # 返回 _mod 属性
    def mod_to_DMP(self):
        return self._mod

    # 将 f 的 _mod 属性替换为新的 rep，返回更新后的 ANP 对象
    def per(f, rep):
        return f.new(rep, f._mod, f.dom)

    # 返回 ANP 对象的字符串表示形式
    def __repr__(f):
        return "%s(%s, %s, %s)" % (f.__class__.__name__, f._rep.to_list(), f._mod.to_list(), f.dom)

    # 返回 ANP 对象的哈希值
    def __hash__(f):
        return hash((f.__class__.__name__, f.to_tuple(), f._mod.to_tuple(), f.dom))

    # 将 ANP 对象转换为新域 dom 的 ANP 对象
    def convert(f, dom):
        """Convert ``f`` to a ``ANP`` over a new domain. """
        # 如果当前对象的域和目标域相同，则直接返回当前对象
        if f.dom == dom:
            return f
        else:
            # 否则，将 _rep 和 _mod 转换为目标域 dom 的类型，并创建新的 ANP 对象
            return f.new(f._rep.convert(dom), f._mod.convert(dom), dom)
    def unify(f, g):
        """Unify representations of two algebraic numbers. """
        # 检查是否需要使用 unify_ANP 方法，此方法不再使用，而是使用 unify_ANP 方法代替
        # XXX: This unify method is not used any more because unify_ANP is used
        # instead.

        # 检查输入参数 g 是否为 ANP 类型，并且模数是否相同，否则抛出 UnificationFailed 异常
        if not isinstance(g, ANP) or f.mod != g.mod:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        # 如果两个对象的定义域相同，则直接返回该定义域、per 方法、f 的表示、g 的表示和模数
        if f.dom == g.dom:
            return f.dom, f.per, f.rep, g.rep, f.mod
        else:
            # 否则，统一两个对象的定义域，并将各自的表示转换为新的定义域下的表示
            dom = f.dom.unify(g.dom)

            # 将 f 的表示和 g 的表示转换为新的定义域下的表示
            F = dup_convert(f.rep, f.dom, dom)
            G = dup_convert(g.rep, g.dom, dom)

            # 根据情况转换模数
            if dom != f.dom and dom != g.dom:
                mod = dup_convert(f.mod, f.dom, dom)
            else:
                if dom == f.dom:
                    mod = f.mod
                else:
                    mod = g.mod

            # 定义一个函数 per，用于将表示转换为 ANP 对象
            per = lambda rep: ANP(rep, mod, dom)

        # 返回统一后的定义域、per 方法、F 表示、G 表示和模数
        return dom, per, F, G, mod

    def unify_ANP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        # 检查输入参数 g 是否为 ANP 类型，并且模数是否相同，否则抛出 UnificationFailed 异常
        if not isinstance(g, ANP) or f._mod != g._mod:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        # 如果 f 和 g 的定义域不同，则统一它们的定义域，并将它们转换为相同定义域下的对象
        if f.dom != g.dom:
            dom = f.dom.unify(g.dom)
            f = f.convert(dom)
            g = g.convert(dom)

        # 返回 f 和 g 的表示、模数和定义域
        return f._rep, g._rep, f._mod, f.dom

    @classmethod
    def zero(cls, mod, dom):
        # 返回一个具有给定模数和定义域的 ANP 对象，表示数值 0
        return ANP(0, mod, dom)

    @classmethod
    def one(cls, mod, dom):
        # 返回一个具有给定模数和定义域的 ANP 对象，表示数值 1
        return ANP(1, mod, dom)

    def to_dict(f):
        """Convert ``f`` to a dict representation with native coefficients. """
        # 将对象 f 的表示转换为字典形式，字典中的值为本地系数
        return f._rep.to_dict()

    def to_sympy_dict(f):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        # 将对象 f 的表示转换为字典形式，字典中的值为 SymPy 系数
        rep = dmp_to_dict(f.rep, 0, f.dom)

        # 将字典中的值转换为 SymPy 类型
        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)

        # 返回转换后的字典表示
        return rep

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        # 将对象 f 的表示转换为列表形式，列表中的值为本地系数
        return f._rep.to_list()

    def mod_to_list(f):
        """Return ``f.mod`` as a list with native coefficients. """
        # 将对象 f 的模数转换为列表形式，列表中的值为本地系数
        return f._mod.to_list()

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        # 将对象 f 的表示转换为列表形式，列表中的值为 SymPy 系数
        return [ f.dom.to_sympy(c) for c in f.to_list() ]

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        # 将对象 f 的表示转换为元组形式，元组中的值为本地系数
        return f._rep.to_tuple()

    @classmethod
    def from_list(cls, rep, mod, dom):
        # 从给定的列表 rep、模数 mod 和定义域 dom 创建一个 ANP 对象
        return ANP(dup_strip(list(map(dom.convert, rep))), mod, dom)

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        # 将对象 f 的表示中加上一个来自基础域的元素 c
        return f.per(f._rep.add_ground(c))

    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        # 将对象 f 的表示中减去一个来自基础域的元素 c
        return f.per(f._rep.sub_ground(c))
    # 将多项式 ``f`` 乘以一个来自于基本域的元素
    def mul_ground(f, c):
        """Multiply ``f`` by an element of the ground domain. """
        return f.per(f._rep.mul_ground(c))

    # 将多项式 ``f`` 除以一个来自于基本域的元素
    def quo_ground(f, c):
        """Quotient of ``f`` by an element of the ground domain. """
        return f.per(f._rep.quo_ground(c))

    # 返回多项式 ``f`` 的相反数
    def neg(f):
        return f.per(f._rep.neg())

    # 返回多项式 ``f`` 与 ``g`` 的加法结果
    def add(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.add(G), mod, dom)

    # 返回多项式 ``f`` 与 ``g`` 的减法结果
    def sub(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.sub(G), mod, dom)

    # 返回多项式 ``f`` 与 ``g`` 的乘法结果
    def mul(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.mul(G).rem(mod), mod, dom)

    # 将多项式 ``f`` 升至非负整数幂 ``n``
    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if not isinstance(n, int):
            raise TypeError("``int`` expected, got %s" % type(n))

        mod = f._mod
        F = f._rep

        if n < 0:
            F, n = F.invert(mod), -n

        # XXX: 需要为 DMP 提供 pow_mod 方法
        return f.new(F.pow(n).rem(f._mod), mod, f.dom)

    # 返回多项式 ``f`` 与 ``g`` 的商
    def exquo(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.mul(G.invert(mod)).rem(mod), mod, dom)

    # 返回多项式 ``f`` 与 ``g`` 的商及余数
    def div(f, g):
        return f.exquo(g), f.zero(f._mod, f.dom)

    # 返回多项式 ``f`` 与 ``g`` 的商
    def quo(f, g):
        return f.exquo(g)

    # 返回多项式 ``f`` 与 ``g`` 的余数
    def rem(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        s, h = F.half_gcdex(G)

        if h.is_one:
            return f.zero(mod, dom)
        else:
            raise NotInvertible("zero divisor")

    # 返回多项式 ``f`` 的主导系数
    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return f._rep.LC()

    # 返回多项式 ``f`` 的尾系数
    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return f._rep.TC()

    # 返回布尔值，指示多项式 ``f`` 是否为零代数数
    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero algebraic number. """
        return f._rep.is_zero

    # 返回布尔值，指示多项式 ``f`` 是否为单位代数数
    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit algebraic number. """
        return f._rep.is_one

    # 返回布尔值，指示多项式 ``f`` 是否为基本域的元素
    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return f._rep.is_ground

    # 返回多项式 ``f`` 自身，表示正
    def __pos__(f):
        return f

    # 返回多项式 ``f`` 的负值
    def __neg__(f):
        return f.neg()

    # 返回多项式 ``f`` 与另一多项式 ``g`` 的加法结果
    def __add__(f, g):
        if isinstance(g, ANP):
            return f.add(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.add_ground(g)

    # 返回多项式 ``f`` 与另一多项式 ``g`` 的反向加法结果
    def __radd__(f, g):
        return f.__add__(g)

    # 返回多项式 ``f`` 与另一多项式 ``g`` 的减法结果
    def __sub__(f, g):
        if isinstance(g, ANP):
            return f.sub(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.sub_ground(g)

    # 返回多项式 ``f`` 与另一多项式 ``g`` 的反向减法结果
    def __rsub__(f, g):
        return (-f).__add__(g)

    # 返回多项式 ``f`` 与另一多项式 ``g`` 的乘法结果
    def __mul__(f, g):
        if isinstance(g, ANP):
            return f.mul(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.mul_ground(g)
    # 定义特殊方法 __rmul__，实现右乘操作
    def __rmul__(f, g):
        return f.__mul__(g)

    # 定义特殊方法 __pow__，实现乘方操作
    def __pow__(f, n):
        return f.pow(n)

    # 定义特殊方法 __divmod__，实现除余操作
    def __divmod__(f, g):
        return f.div(g)

    # 定义特殊方法 __mod__，实现取模操作
    def __mod__(f, g):
        return f.rem(g)

    # 定义特殊方法 __truediv__，实现真除操作
    def __truediv__(f, g):
        # 如果 g 是 ANP 类型，则进行真除操作
        if isinstance(g, ANP):
            return f.quo(g)
        # 否则尝试将 g 转换为 f.dom 所属的类型
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.quo_ground(g)

    # 定义特殊方法 __eq__，实现相等比较
    def __eq__(f, g):
        # 尝试统一两个对象为 ANP 类型
        try:
            F, G, _, _ = f.unify_ANP(g)
        except UnificationFailed:
            return NotImplemented
        return F == G

    # 定义特殊方法 __ne__，实现不等比较
    def __ne__(f, g):
        # 尝试统一两个对象为 ANP 类型
        try:
            F, G, _, _ = f.unify_ANP(g)
        except UnificationFailed:
            return NotImplemented
        return F != G

    # 定义特殊方法 __lt__，实现小于比较
    def __lt__(f, g):
        # 统一两个对象为 ANP 类型并比较
        F, G, _, _ = f.unify_ANP(g)
        return F < G

    # 定义特殊方法 __le__，实现小于等于比较
    def __le__(f, g):
        # 统一两个对象为 ANP 类型并比较
        F, G, _, _ = f.unify_ANP(g)
        return F <= G

    # 定义特殊方法 __gt__，实现大于比较
    def __gt__(f, g):
        # 统一两个对象为 ANP 类型并比较
        F, G, _, _ = f.unify_ANP(g)
        return F > G

    # 定义特殊方法 __ge__，实现大于等于比较
    def __ge__(f, g):
        # 统一两个对象为 ANP 类型并比较
        F, G, _, _ = f.unify_ANP(g)
        return F >= G

    # 定义特殊方法 __bool__，实现对象的布尔值返回
    def __bool__(f):
        return bool(f._rep)
```