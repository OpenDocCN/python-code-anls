# `D:\src\scipysrc\sympy\sympy\polys\rootoftools.py`

```
# 导入必要的模块和类
from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
    symbols, sympify, Rational, Dummy)
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    GeneratorsNeeded,
    PolynomialError,
    DomainError)
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
    roots_linear, roots_quadratic, roots_binomial,
    preprocess_roots, roots)
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
    dup_isolate_complex_roots_sqf,
    dup_isolate_real_roots_sqf)
from sympy.utilities import lambdify, public, sift, numbered_symbols

# 导入外部库和函数
from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain

# 指定模块的公开接口
__all__ = ['CRootOf']

# 定义一个类 _pure_key_dict，用于管理以 PurePoly 实例为键的数据
class _pure_key_dict:
    """A minimal dictionary that makes sure that the key is a
    univariate PurePoly instance.

    Examples
    ========

    Only the following actions are guaranteed:

    >>> from sympy.polys.rootoftools import _pure_key_dict
    >>> from sympy import PurePoly
    >>> from sympy.abc import x, y

    1) creation

    >>> P = _pure_key_dict()

    2) assignment for a PurePoly or univariate polynomial

    >>> P[x] = 1
    >>> P[PurePoly(x - y, x)] = 2

    3) retrieval based on PurePoly key comparison (use this
       instead of the get method)

    >>> P[y]
    1

    4) KeyError when trying to retrieve a nonexisting key

    >>> P[y + 1]
    Traceback (most recent call last):
    ...
    KeyError: PurePoly(y + 1, y, domain='ZZ')

    5) ability to query with ``in``

    >>> x + 1 in P
    False

    NOTE: this is a *not* a dictionary. It is a very basic object
    for internal use that makes sure to always address its cache
    via PurePoly instances. It does not, for example, implement
    ``get`` or ``setdefault``.
    """
    def __init__(self):
        self._dict = {}

    def __getitem__(self, k):
        # 如果键不是 PurePoly 实例，则尝试将其转换为 PurePoly 实例
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise KeyError
            k = PurePoly(k, expand=False)
        return self._dict[k]

    def __setitem__(self, k, v):
        # 如果键不是 PurePoly 实例，则尝试将其转换为 PurePoly 实例
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise ValueError('expecting univariate expression')
            k = PurePoly(k, expand=False)
        self._dict[k] = v

    def __contains__(self, k):
        # 检查给定的键是否存在于字典中，通过 PurePoly 实例进行比较
        try:
            self[k]
            return True
        except KeyError:
            return False

# 创建两个 _pure_key_dict 实例，用于缓存实数和复数根的计算结果
_reals_cache = _pure_key_dict()
_complexes_cache = _pure_key_dict()
# 定义一个函数 _pure_factors，用于获取多项式的纯因子
def _pure_factors(poly):
    # 使用多项式对象的 factor_list 方法获取其因子分解结果
    _, factors = poly.factor_list()
    # 返回一个列表，包含每个因子以及其对应的指数
    return [(PurePoly(f, expand=False), m) for f, m in factors]


# 定义一个函数 _imag_count_of_factor，用于计算不可约单变量多项式的虚根数量
def _imag_count_of_factor(f):
    """Return the number of imaginary roots for irreducible
    univariate polynomial ``f``.
    """
    # 提取多项式 f 的所有项，并将其格式化为 (i, j) 形式的列表
    terms = [(i, j) for (i,), j in f.terms()]
    # 如果存在任何指数为奇数的项，则返回虚根数量为 0
    if any(i % 2 for i, j in terms):
        return 0
    # 更新所有项的符号，生成新的多项式对象 even
    even = [(i, I**i*j) for i, j in terms]
    even = Poly.from_dict(dict(even), Dummy('x'))
    # 计算 even 多项式在实数范围 (-oo, oo) 内的根的数量，并转换为整数返回
    return int(even.count_roots(-oo, oo))


# 定义一个公共函数 rootof，用于返回单变量多项式的索引根
@public
def rootof(f, x, index=None, radicals=True, expand=True):
    """An indexed root of a univariate polynomial.

    Returns either a :obj:`ComplexRootOf` object or an explicit
    expression involving radicals.

    Parameters
    ==========

    f : Expr
        Univariate polynomial.
    x : Symbol, optional
        Generator for ``f``.
    index : int or Integer
    radicals : bool
               Return a radical expression if possible.
    expand : bool
             Expand ``f``.
    """
    # 调用 CRootOf 类构造函数，返回 f 的根对象
    return CRootOf(f, x, index=index, radicals=radicals, expand=expand)


# 定义一个公共类 RootOf，表示单变量多项式的根
@public
class RootOf(Expr):
    """Represents a root of a univariate polynomial.

    Base class for roots of different kinds of polynomials.
    Only complex roots are currently supported.
    """

    __slots__ = ('poly',)

    def __new__(cls, f, x, index=None, radicals=True, expand=True):
        """Construct a new ``CRootOf`` object for ``k``-th root of ``f``."""
        # 调用 rootof 函数，返回 f 的根对象
        return rootof(f, x, index=index, radicals=radicals, expand=expand)


# 定义一个公共类 ComplexRootOf，表示多项式的索引复根
@public
class ComplexRootOf(RootOf):
    """Represents an indexed complex root of a polynomial.

    Roots of a univariate polynomial separated into disjoint
    real or complex intervals and indexed in a fixed order:

    * real roots come first and are sorted in increasing order;
    * complex roots come next and are sorted primarily by increasing
      real part, secondarily by increasing imaginary part.

    Currently only rational coefficients are allowed.
    Can be imported as ``CRootOf``. To avoid confusion, the
    generator must be a Symbol.


    Examples
    ========

    >>> from sympy import CRootOf, rootof
    >>> from sympy.abc import x

    CRootOf is a way to reference a particular root of a
    polynomial. If there is a rational root, it will be returned:

    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> CRootOf(x**2 - 4, 0)
    -2

    Whether roots involving radicals are returned or not
    depends on whether the ``radicals`` flag is true (which is
    set to True with rootof):

    >>> CRootOf(x**2 - 3, 0)
    CRootOf(x**2 - 3, 0)
    >>> CRootOf(x**2 - 3, 0, radicals=True)
    -sqrt(3)
    >>> rootof(x**2 - 3, 0)
    -sqrt(3)

    The following cannot be expressed in terms of radicals:

    >>> r = rootof(4*x**5 + 16*x**3 + 12*x**2 + 7, 0); r
    CRootOf(4*x**5 + 16*x**3 + 12*x**2 + 7, 0)

    The root bounds can be seen, however, and they are used by the
    """
    # 定义一个类，表示代数方程的复数根
    class CRootOf:
    
        # 类的专用属性，用于存储单个整数索引
        __slots__ = ('index',)
    
        # 表示这个类的实例是复数
        is_complex = True
        # 表示这个类的实例是一个数值
        is_number = True
        # 表示这个类的实例是有限的（即非无穷大或无穷小）
        is_finite = True
    
    
    这段代码定义了一个名为 `CRootOf` 的类，用于表示代数方程的复数根。类具有三个属性 `__slots__`、`is_complex`、`is_number` 和 `is_finite`，分别用于存储索引和描述实例的特性。
    def __new__(cls, f, x, index=None, radicals=False, expand=True):
        """ 构造一个多项式的索引复根对象。

        参见 ``rootof`` 以了解参数详情。

        默认情况下，``radicals`` 的值为 ``False``，以满足 ``eval(srepr(expr)) == expr`` 的要求。
        """
        x = sympify(x)  # 将 x 转换为 SymPy 表达式

        if index is None and x.is_Integer:
            x, index = None, x
        else:
            index = sympify(index)

        if index is not None and index.is_Integer:
            index = int(index)
        else:
            raise ValueError("expected an integer root index, got %s" % index)

        poly = PurePoly(f, x, greedy=False, expand=expand)  # 创建多项式对象

        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")  # 多项式必须是一元的

        if not poly.gen.is_Symbol:
            # 生成器必须是符号变量
            raise PolynomialError("generator must be a Symbol")

        degree = poly.degree()  # 多项式的次数

        if degree <= 0:
            raise PolynomialError("Cannot construct CRootOf object for %s" % f)  # 无法为该多项式构造 CRootOf 对象

        if index < -degree or index >= degree:
            raise IndexError("root index out of [%d, %d] range, got %d" %
                             (-degree, degree - 1, index))  # 索引超出范围

        elif index < 0:
            index += degree

        dom = poly.get_domain()  # 获取多项式的定义域

        if not dom.is_Exact:
            poly = poly.to_exact()  # 将多项式转换为精确表示

        roots = cls._roots_trivial(poly, radicals)  # 计算多项式的根

        if roots is not None:
            return roots[index]  # 如果根已经计算过，则直接返回相应索引的根

        coeff, poly = preprocess_roots(poly)  # 预处理多项式的根
        dom = poly.get_domain()

        if not dom.is_ZZ:
            raise NotImplementedError("CRootOf is not supported over %s" % dom)  # 不支持非整数域上的 CRootOf

        root = cls._indexed_root(poly, index, lazy=True)  # 计算多项式的索引根
        return coeff * cls._postprocess_root(root, radicals)  # 对计算出的根进行后处理并返回结果

    @classmethod
    def _new(cls, poly, index):
        """从原始数据构造一个新的 ``CRootOf`` 对象。"""
        obj = Expr.__new__(cls)  # 创建新的表达式对象

        obj.poly = PurePoly(poly)  # 使用给定的多项式创建 PurePoly 对象
        obj.index = index  # 设置索引值

        try:
            _reals_cache[obj.poly] = _reals_cache[poly]
            _complexes_cache[obj.poly] = _complexes_cache[poly]
        except KeyError:
            pass

        return obj

    def _hashable_content(self):
        return (self.poly, self.index)  # 返回对象的可散列内容

    @property
    def expr(self):
        return self.poly.as_expr()  # 返回对象对应的表达式

    @property
    def args(self):
        return (self.expr, Integer(self.index))  # 返回对象的参数

    @property
    def free_symbols(self):
        # CRootOf 目前仅适用于无自由符号的一元表达式
        # 其 poly 属性应为无自由符号的 PurePoly
        return set()  # 返回空集，表示没有自由符号

    def _eval_is_real(self):
        """如果根是实数，则返回 ``True``。"""
        self._ensure_reals_init()  # 确保实数初始化
        return self.index < len(_reals_cache[self.poly])  # 检查索引是否在实数缓存范围内
    def _eval_is_imaginary(self):
        """Return ``True`` if the root is imaginary. """
        # Ensure that the real roots cache is initialized
        self._ensure_reals_init()
        # Check if the index of the current root is beyond the cached real roots
        if self.index >= len(_reals_cache[self.poly]):
            # Get the interval containing the root
            ivl = self._get_interval()
            # Determine if the interval spans across zero, indicating imaginary roots
            return ivl.ax * ivl.bx <= 0  # all others are on one side or the other
        # Return False if the root is not imaginary
        return False  # XXX is this necessary?

    @classmethod
    def real_roots(cls, poly, radicals=True):
        """Get real roots of a polynomial. """
        # Call _get_roots method to retrieve real roots of the polynomial
        return cls._get_roots("_real_roots", poly, radicals)

    @classmethod
    def all_roots(cls, poly, radicals=True):
        """Get real and complex roots of a polynomial. """
        # Call _get_roots method to retrieve all roots of the polynomial
        return cls._get_roots("_all_roots", poly, radicals)

    @classmethod
    def _get_reals_sqf(cls, currentfactor, use_cache=True):
        """Get real root isolating intervals for a square-free factor."""
        # Check if the real root isolating intervals for the current factor are cached
        if use_cache and currentfactor in _reals_cache:
            real_part = _reals_cache[currentfactor]
        else:
            # Compute real root isolating intervals and cache them if needed
            _reals_cache[currentfactor] = real_part = \
                dup_isolate_real_roots_sqf(
                    currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)

        return real_part

    @classmethod
    def _get_complexes_sqf(cls, currentfactor, use_cache=True):
        """Get complex root isolating intervals for a square-free factor."""
        # Check if the complex root isolating intervals for the current factor are cached
        if use_cache and currentfactor in _complexes_cache:
            complex_part = _complexes_cache[currentfactor]
        else:
            # Compute complex root isolating intervals and cache them if needed
            _complexes_cache[currentfactor] = complex_part = \
                dup_isolate_complex_roots_sqf(
                currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)
        return complex_part

    @classmethod
    def _get_reals(cls, factors, use_cache=True):
        """Compute real root isolating intervals for a list of factors. """
        reals = []

        for currentfactor, k in factors:
            try:
                # Attempt to retrieve real root intervals from cache
                if not use_cache:
                    raise KeyError
                r = _reals_cache[currentfactor]
                # Extend the list of real root intervals
                reals.extend([(i, currentfactor, k) for i in r])
            except KeyError:
                # Compute real root intervals for the current factor if not cached
                real_part = cls._get_reals_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in real_part]
                reals.extend(new)

        # Sort and return the list of real root intervals
        reals = cls._reals_sorted(reals)
        return reals
    @classmethod
    # 声明一个类方法，用于获取复数根的隔离区间
    def _get_complexes(cls, factors, use_cache=True):
        """Compute complex root isolating intervals for a list of factors. """
        # 初始化一个空列表用于存储复数根的隔离区间
        complexes = []

        # 遍历因子列表，并按顺序处理
        for currentfactor, k in ordered(factors):
            try:
                # 尝试从缓存中获取当前因子的复数根隔离区间
                if not use_cache:
                    raise KeyError
                c = _complexes_cache[currentfactor]
                # 将获取到的复数根隔离区间添加到结果列表中
                complexes.extend([(i, currentfactor, k) for i in c])
            except KeyError:
                # 如果缓存中不存在，则计算当前因子的复数根隔离区间
                complex_part = cls._get_complexes_sqf(currentfactor, use_cache)
                # 将计算得到的复数根隔离区间添加到结果列表中
                new = [(root, currentfactor, k) for root in complex_part]
                complexes.extend(new)

        # 对复数根隔离区间进行排序和处理
        complexes = cls._complexes_sorted(complexes)
        # 返回最终的复数根隔离区间列表
        return complexes
    def _refine_complexes(cls, complexes):
        """
        返回复数，以确保非共轭根的边界矩形不相交。此外，确保 ay 和 by 均不为零，
        以保证非实根与实根在 y 边界上的区别。
        """
        # 逐对获取间隔不相交的复数。
        # 如果围绕边界矩形的坐标画矩形，则经过此过程后不会有矩形相交。
        for i, (u, f, k) in enumerate(complexes):
            for j, (v, g, m) in enumerate(complexes[i + 1:]):
                u, v = u.refine_disjoint(v)
                complexes[i + j + 1] = (v, g, m)

            complexes[i] = (u, f, k)

        # 直到对于非虚根，x 边界明确为正或负为止进行精化
        complexes = cls._refine_imaginary(complexes)

        # 确保所有 y 边界都远离实轴并在同一侧
        for i, (u, f, k) in enumerate(complexes):
            while u.ay * u.by <= 0:
                u = u.refine()
            complexes[i] = u, f, k
        return complexes

    @classmethod
    def _complexes_sorted(cls, complexes):
        """使复数隔离区间不相交并排序根。"""
        complexes = cls._refine_complexes(complexes)
        # XXX 在确定与索引方法兼容之前不要排序，但断言所需状态未被破坏
        C, F = 0, 1  # ComplexInterval 和 factor 的位置
        fs = {i[F] for i in complexes}
        for i in range(1, len(complexes)):
            if complexes[i][F] != complexes[i - 1][F]:
                # 如果此处失败，则根的因子不连续，因为不连续应该只发生一次
                fs.remove(complexes[i - 1][F])
        for i, cmplx in enumerate(complexes):
            # 负虚部（conj=True）位于正虚部（conj=False）之前
            assert cmplx[C].conj is (i % 2 == 0)

        # 更新缓存
        cache = {}
        # -- 汇总
        for root, currentfactor, _ in complexes:
            cache.setdefault(currentfactor, []).append(root)
        # -- 存储
        for currentfactor, root in cache.items():
            _complexes_cache[currentfactor] = root

        return complexes

    @classmethod
    def _reals_index(cls, reals, index):
        """
        将初始实根索引映射到根所属因子的索引。
        """
        i = 0

        for j, (_, currentfactor, k) in enumerate(reals):
            if index < i + k:
                poly, index = currentfactor, 0

                for _, currentfactor, _ in reals[:j]:
                    if currentfactor == poly:
                        index += 1

                return poly, index
            else:
                i += k
    @classmethod
    def _complexes_index(cls, complexes, index):
        """
        Map initial complex root index to an index in a factor where
        the root belongs.
        """
        # 初始化索引值
        i = 0
        # 遍历复数根列表中的每个元素，解包得到索引 j、当前因子 currentfactor、次数 k
        for j, (_, currentfactor, k) in enumerate(complexes):
            # 如果所需索引在当前因子的根范围内
            if index < i + k:
                # 设置 poly 为当前因子，并将索引重置为 0
                poly, index = currentfactor, 0

                # 计算当前因子在 complexes 列表中出现的次数，更新 index
                for _, currentfactor, _ in complexes[:j]:
                    if currentfactor == poly:
                        index += 1

                # 加上在 _reals_cache 中的 currentfactor 的根数
                index += len(_reals_cache[poly])

                # 返回找到的 poly 和对应的 index
                return poly, index
            else:
                # 更新当前索引 i，移动到下一个因子的根范围
                i += k

    @classmethod
    def _count_roots(cls, roots):
        """Count the number of real or complex roots with multiplicities."""
        # 返回所有根的总数，包括其重数
        return sum(k for _, _, k in roots)

    @classmethod
    def _indexed_root(cls, poly, index, lazy=False):
        """Get a root of a composite polynomial by index. """
        # 提取 poly 的纯因子
        factors = _pure_factors(poly)

        # 如果 poly 已经是不可约的，并且 lazy 标志为 True，则直接返回该 poly 和 index
        if lazy and len(factors) == 1 and factors[0][1] == 1:
            return factors[0][0], index

        # 获取 poly 的实数根列表
        reals = cls._get_reals(factors)
        # 计算实数根的总数（包括其重数）
        reals_count = cls._count_roots(reals)

        # 如果所需索引在实数根的范围内，则返回实数根的索引
        if index < reals_count:
            return cls._reals_index(reals, index)
        else:
            # 否则，获取 poly 的复数根列表，并计算索引
            complexes = cls._get_complexes(factors)
            return cls._complexes_index(complexes, index - reals_count)

    def _ensure_reals_init(self):
        """Ensure that our poly has entries in the reals cache. """
        # 如果当前 poly 不在 _reals_cache 中，则初始化其实数根
        if self.poly not in _reals_cache:
            self._indexed_root(self.poly, self.index)

    def _ensure_complexes_init(self):
        """Ensure that our poly has entries in the complexes cache. """
        # 如果当前 poly 不在 _complexes_cache 中，则初始化其复数根
        if self.poly not in _complexes_cache:
            self._indexed_root(self.poly, self.index)

    @classmethod
    def _real_roots(cls, poly):
        """Get real roots of a composite polynomial. """
        # 提取 poly 的纯因子
        factors = _pure_factors(poly)

        # 获取 poly 的实数根列表
        reals = cls._get_reals(factors)
        # 计算实数根的总数（包括其重数）
        reals_count = cls._count_roots(reals)

        # 初始化存放实数根的列表
        roots = []

        # 遍历所有实数根的索引，将其添加到 roots 列表中
        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))

        # 返回实数根列表
        return roots

    def _reset(self):
        """
        Reset all intervals
        """
        # 重置所有与 poly 相关的区间信息，强制刷新缓存
        self._all_roots(self.poly, use_cache=False)

    @classmethod
    @classmethod
    def _all_roots(cls, poly, use_cache=True):
        """获取复合多项式的实数和复数根。"""
        # 将多项式分解为纯因子
        factors = _pure_factors(poly)

        # 获取实数根
        reals = cls._get_reals(factors, use_cache=use_cache)
        # 计算实数根的数量
        reals_count = cls._count_roots(reals)

        # 初始化根列表
        roots = []

        # 将实数根加入根列表
        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))

        # 获取复数根
        complexes = cls._get_complexes(factors, use_cache=use_cache)
        # 计算复数根的数量
        complexes_count = cls._count_roots(complexes)

        # 将复数根加入根列表
        for index in range(0, complexes_count):
            roots.append(cls._complexes_index(complexes, index))

        # 返回所有根
        return roots

    @classmethod
    @cacheit
    def _roots_trivial(cls, poly, radicals):
        """计算线性、二次和二项式情况下的根。"""
        # 如果多项式是一次的，直接返回线性根
        if poly.degree() == 1:
            return roots_linear(poly)

        # 如果不使用根式，返回空
        if not radicals:
            return None

        # 如果多项式是二次的，返回二次根
        if poly.degree() == 2:
            return roots_quadratic(poly)
        # 如果多项式长度为2且有TC，返回二项式根
        elif poly.length() == 2 and poly.TC():
            return roots_binomial(poly)
        else:
            return None

    @classmethod
    def _preprocess_roots(cls, poly):
        """采取措施使多项式与'CRootOf'兼容。"""
        # 获取多项式的定义域
        dom = poly.get_domain()

        # 如果定义域不是精确的，转换为精确的
        if not dom.is_Exact:
            poly = poly.to_exact()

        # 预处理多项式的根
        coeff, poly = preprocess_roots(poly)
        dom = poly.get_domain()

        # 如果定义域不是整数环，抛出未实现的错误
        if not dom.is_ZZ:
            raise NotImplementedError(
                "sorted roots not supported over %s" % dom)

        # 返回系数和处理后的多项式
        return coeff, poly

    @classmethod
    def _postprocess_root(cls, root, radicals):
        """如果根是微不足道的或'CRootOf'对象，则返回根。"""
        # 获取多项式和索引
        poly, index = root
        # 计算根的简单情况或'CRootOf'对象
        roots = cls._roots_trivial(poly, radicals)

        # 如果根不是空，则返回根的索引
        if roots is not None:
            return roots[index]
        else:
            # 否则，创建并返回新的'CRootOf'对象
            return cls._new(poly, index)

    @classmethod
    def _get_roots(cls, method, poly, radicals):
        """返回指定类型的后处理根。"""
        # 如果多项式不是单变量的，抛出多项式错误
        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")

        # 替换多项式的生成器为虚拟符号
        d = Dummy()
        poly = poly.subs(poly.gen, d)
        x = symbols('x')

        # 查看剩余的符号并选择x或编号的x，确保不冲突
        free_names = {str(i) for i in poly.free_symbols}
        for x in chain((symbols('x'),), numbered_symbols('x')):
            if x.name not in free_names:
                poly = poly.xreplace({d: x})
                break

        # 预处理多项式的根
        coeff, poly = cls._preprocess_roots(poly)
        # 初始化根列表
        roots = []

        # 对每个根使用指定的方法，并将结果加入根列表
        for root in getattr(cls, method)(poly):
            roots.append(coeff*cls._postprocess_root(root, radicals))

        # 返回所有后处理的根
        return roots
    def clear_cache(cls):
        """Reset cache for reals and complexes.

        The intervals used to approximate a root instance are updated
        as needed. When a request is made to see the intervals, the
        most current values are shown. `clear_cache` will reset all
        CRootOf instances back to their original state.

        See Also
        ========

        _reset
        """
        # 重置实部和复部的缓存
        global _reals_cache, _complexes_cache
        # 使用纯键字典重置实部缓存
        _reals_cache = _pure_key_dict()
        # 使用纯键字典重置复部缓存
        _complexes_cache = _pure_key_dict()

    def _get_interval(self):
        """Internal function for retrieving isolation interval from cache. """
        # 确保实部初始化
        self._ensure_reals_init()
        # 如果是实数根，返回实部缓存中的区间
        if self.is_real:
            return _reals_cache[self.poly][self.index]
        else:
            # 否则，获取实部缓存的长度
            reals_count = len(_reals_cache[self.poly])
            # 确保复部初始化
            self._ensure_complexes_init()
            # 返回复部缓存中对应位置的区间
            return _complexes_cache[self.poly][self.index - reals_count]

    def _set_interval(self, interval):
        """Internal function for updating isolation interval in cache. """
        # 确保实部初始化
        self._ensure_reals_init()
        # 如果是实数根，更新实部缓存中的区间
        if self.is_real:
            _reals_cache[self.poly][self.index] = interval
        else:
            # 否则，获取实部缓存的长度
            reals_count = len(_reals_cache[self.poly])
            # 确保复部初始化
            self._ensure_complexes_init()
            # 更新复部缓存中对应位置的区间
            _complexes_cache[self.poly][self.index - reals_count] = interval

    def _eval_subs(self, old, new):
        # 不允许替换操作改变任何东西，直接返回自身
        # 不进行任何修改
        return self

    def _eval_conjugate(self):
        if self.is_real:
            return self
        expr, i = self.args
        # 如果是复数根，根据当前区间的共轭性进行求解
        return self.func(expr, i + (1 if self._get_interval().conj else -1))

    def _eval_evalf(self, prec, **kwargs):
        """Evaluate this complex root to the given precision."""
        # 忽略所有的kwargs参数
        # 通过有理数求值后再精确求值
        return self.eval_rational(n=prec_to_dps(prec))._evalf(prec)
# 将 ComplexRootOf 别名为 CRootOf
CRootOf = ComplexRootOf

# 定义一个特化函数，处理两个 ComplexRootOf 对象的比较，避免无限递归
@dispatch(ComplexRootOf, ComplexRootOf)
def _eval_is_eq(lhs, rhs): # noqa:F811
    # 如果在此处使用 is_eq 进行检查，会导致无限递归
    return lhs == rhs

# 定义一个特化函数，处理 ComplexRootOf 对象与 Basic 类型对象的比较
@dispatch(ComplexRootOf, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    # CRootOf 表示一个根，如果 rhs 是该根，则表达式应为零，且应在 CRootOf 实例的区间内。
    # 同时 rhs 必须是与 CRootOf 实例的 is_real 值相符的数字。
    if not rhs.is_number:
        return None
    if not rhs.is_finite:
        return False
    # 将 lhs.expr 中的自由符号中的一个替换为 rhs，并检查表达式是否为零
    z = lhs.expr.subs(lhs.expr.free_symbols.pop(), rhs).is_zero
    if z is False:  # 所有的根都会使 z 为 True，但我们不知道是否是正确的根如果 z 为 True
        return False
    o = rhs.is_real, rhs.is_imaginary
    s = lhs.is_real, lhs.is_imaginary
    assert None not in s  # 这是初始细化的一部分
    # 如果 o 与 s 不相等，并且 o 中没有 None，则返回 False
    if o != s and None not in o:
        return False
    # 获取 rhs 的实部和虚部
    re, im = rhs.as_real_imag()
    if lhs.is_real:
        if im:
            return False
        # 获取 lhs 的区间并判断 rhs 是否在该区间内
        i = lhs._get_interval()
        a, b = [Rational(str(_)) for _ in (i.a, i.b)]
        return sympify(a <= rhs and rhs <= b)
    # 获取 lhs 的区间并判断 rhs 的实部和虚部是否在该区间内
    i = lhs._get_interval()
    r1, r2, i1, i2 = [Rational(str(j)) for j in (
        i.ax, i.bx, i.ay, i.by)]
    return is_le(r1, re) and is_le(re,r2) and is_le(i1,im) and is_le(im,i2)

# 定义一个公共类 RootSum，表示一元多项式的所有根的和
@public
class RootSum(Expr):
    """Represents a sum of all roots of a univariate polynomial. """

    # 类的槽位，用于限制可以动态添加的属性
    __slots__ = ('poly', 'fun', 'auto')
    def __new__(cls, expr, func=None, x=None, auto=True, quadratic=False):
        """构造一个新的``RootSum``实例，用于多项式的根的求和。"""
        # 转换表达式为系数和多项式
        coeff, poly = cls._transform(expr, x)

        # 如果多项式不是一元的，则抛出异常
        if not poly.is_univariate:
            raise MultivariatePolynomialError(
                "only univariate polynomials are allowed")

        # 如果未指定函数，则默认为多项式的恒等函数
        if func is None:
            func = Lambda(poly.gen, poly.gen)
        else:
            # 检查传入的函数是否符合预期
            is_func = getattr(func, 'is_Function', False)

            if is_func and 1 in func.nargs:
                # 如果是单变量函数，转换为Lambda函数
                if not isinstance(func, Lambda):
                    func = Lambda(poly.gen, func(poly.gen))
            else:
                # 抛出异常，期望的是单变量函数
                raise ValueError(
                    "expected a univariate function, got %s" % func)

        # 提取函数的变量和表达式
        var, expr = func.variables[0], func.expr

        # 如果系数不是1，则乘以多项式的变量
        if coeff is not S.One:
            expr = expr.subs(var, coeff*var)

        # 计算多项式的次数
        deg = poly.degree()

        # 如果表达式中不包含多项式的变量，则返回次数乘以表达式
        if not expr.has(var):
            return deg*expr

        # 将表达式分解为常数项和主项
        if expr.is_Add:
            add_const, expr = expr.as_independent(var)
        else:
            add_const = S.Zero

        # 将表达式分解为乘法常数和主项
        if expr.is_Mul:
            mul_const, expr = expr.as_independent(var)
        else:
            mul_const = S.One

        # 重新定义函数为Lambda函数
        func = Lambda(var, expr)

        # 检查函数是否为有理函数
        rational = cls._is_func_rational(poly, func)
        # 分解多项式为纯因子和项列表
        factors, terms = _pure_factors(poly), []

        # 遍历纯因子
        for poly, k in factors:
            if poly.is_linear:
                # 如果是一次方程，则计算根并应用函数
                term = func(roots_linear(poly)[0])
            elif quadratic and poly.is_quadratic:
                # 如果开启了二次项，并且是二次方程，则计算根并应用函数
                term = sum(map(func, roots_quadratic(poly)))
            else:
                # 如果不是线性或者二次项，根据情况递归构建新的RootSum实例或者使用有理情况处理
                if not rational or not auto:
                    term = cls._new(poly, func, auto)
                else:
                    term = cls._rational_case(poly, func)

            terms.append(k*term)

        # 返回计算结果，主项乘以乘法常数，加上常数项乘以多项式的次数
        return mul_const*Add(*terms) + deg*add_const

    @classmethod
    def _new(cls, poly, func, auto=True):
        """构造一个新的原始``RootSum``实例。"""
        obj = Expr.__new__(cls)

        obj.poly = poly
        obj.fun = func
        obj.auto = auto

        return obj

    @classmethod
    def new(cls, poly, func, auto=True):
        """构造一个新的``RootSum``实例。"""
        # 如果函数表达式中不包含任何变量，则直接返回函数表达式
        if not func.expr.has(*func.variables):
            return func.expr

        # 检查函数是否为有理函数
        rational = cls._is_func_rational(poly, func)

        # 如果不是有理函数或者不自动计算，则调用_new方法构造新的实例
        if not rational or not auto:
            return cls._new(poly, func, auto)
        else:
            # 否则调用有理情况处理方法
            return cls._rational_case(poly, func)

    @classmethod
    def _transform(cls, expr, x):
        """将表达式转换为多项式。"""
        poly = PurePoly(expr, x, greedy=False)
        return preprocess_roots(poly)

    @classmethod
    def _is_func_rational(cls, poly, func):
        """检查Lambda函数是否为有理函数。"""
        var, expr = func.variables[0], func.expr
        return expr.is_rational_function(var)

    @classmethod
    def _rational_case(cls, poly, func):
        """处理有理函数的情况。"""
        # 实现有理函数的特殊处理逻辑
        pass
    def _rational_case(cls, poly, func):
        """处理有理函数情况。"""
        # 定义多项式的根符号
        roots = symbols('r:%d' % poly.degree())
        # 获取函数的变量和表达式
        var, expr = func.variables[0], func.expr

        # 计算函数在各根处的值的和
        f = sum(expr.subs(var, r) for r in roots)
        # 对函数求通分后的分子和分母
        p, q = together(f).as_numer_denom()

        # 多项式的定义域
        domain = QQ[roots]

        # 尝试将分子转化为多项式
        p = p.expand()
        q = q.expand()

        try:
            p = Poly(p, domain=domain, expand=False)
        except GeneratorsNeeded:
            p, p_coeff = None, (p,)
        else:
            p_monom, p_coeff = zip(*p.terms())

        try:
            q = Poly(q, domain=domain, expand=False)
        except GeneratorsNeeded:
            q, q_coeff = None, (q,)
        else:
            q_monom, q_coeff = zip(*q.terms())

        # 对多项式系数进行对称化处理
        coeffs, mapping = symmetrize(p_coeff + q_coeff, formal=True)
        # 计算维特方程的结果
        formulas, values = viete(poly, roots), []

        # 将映射和维特方程结果结合得到值
        for (sym, _), (_, val) in zip(mapping, formulas):
            values.append((sym, val))

        # 对系数进行值替换
        for i, (coeff, _) in enumerate(coeffs):
            coeffs[i] = coeff.subs(values)

        # 分离出分子和分母的系数
        n = len(p_coeff)
        p_coeff = coeffs[:n]
        q_coeff = coeffs[n:]

        # 构建分子和分母的表达式
        if p is not None:
            p = Poly(dict(zip(p_monom, p_coeff)), *p.gens).as_expr()
        else:
            (p,) = p_coeff

        if q is not None:
            q = Poly(dict(zip(q_monom, q_coeff)), *q.gens).as_expr()
        else:
            (q,) = q_coeff

        # 返回分式化简后的结果
        return factor(p/q)

    def _hashable_content(self):
        """返回对象的可散列内容。"""
        return (self.poly, self.fun)

    @property
    def expr(self):
        """返回对象的表达式形式。"""
        return self.poly.as_expr()

    @property
    def args(self):
        """返回对象的参数元组。"""
        return (self.expr, self.fun, self.poly.gen)

    @property
    def free_symbols(self):
        """返回对象的自由符号集合。"""
        return self.poly.free_symbols | self.fun.free_symbols

    @property
    def is_commutative(self):
        """返回对象是否可交换。"""
        return True

    def doit(self, **hints):
        """如果根不是必需的，则不进行计算。"""
        if not hints.get('roots', True):
            return self

        # 计算多项式的全部根
        _roots = roots(self.poly, multiple=True)

        # 如果计算得到的根少于多项式的次数，则返回对象本身
        if len(_roots) < self.poly.degree():
            return self
        else:
            # 否则计算函数在所有根处的值的和
            return Add(*[self.fun(r) for r in _roots])

    def _eval_evalf(self, prec):
        """使用数值计算精度进行估值计算。"""
        try:
            # 计算多项式的数值根
            _roots = self.poly.nroots(n=prec_to_dps(prec))
        except (DomainError, PolynomialError):
            # 如果计算出错，则返回对象本身
            return self
        else:
            # 否则计算函数在所有数值根处的值的和
            return Add(*[self.fun(r) for r in _roots])

    def _eval_derivative(self, x):
        """计算函数的导数。"""
        # 提取函数的变量和表达式
        var, expr = self.fun.args
        # 定义新函数，表示给定变量对应的导数
        func = Lambda(var, expr.diff(x))
        # 返回一个新的对象，表示原对象的导数
        return self.new(self.poly, func, self.auto)
```