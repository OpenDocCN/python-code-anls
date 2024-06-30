# `D:\src\scipysrc\sympy\sympy\polys\domains\old_fractionfield.py`

```
"""Implementation of :class:`FractionField` class. """

# 导入所需的模块和类
from sympy.polys.domains.field import Field
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.polyclasses import DMF
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public

# 声明 FractionField 类，继承自 Field 和 CompositeDomain
@public
class FractionField(Field, CompositeDomain):
    """A class for representing rational function fields. """

    # 设置类属性
    dtype = DMF
    is_FractionField = is_Frac = True

    # 指定关联的环和域
    has_assoc_Ring = True
    has_assoc_Field = True

    # 构造函数，接受域和生成元作为参数
    def __init__(self, dom, *gens):
        # 如果没有生成元，则抛出异常
        if not gens:
            raise GeneratorsNeeded("generators not specified")

        # 计算生成元的数量
        lev = len(gens) - 1
        self.ngens = len(gens)

        # 初始化零元素和单位元素
        self.zero = self.dtype.zero(lev, dom)
        self.one = self.dtype.one(lev, dom)

        # 设置域和生成元
        self.domain = self.dom = dom
        self.symbols = self.gens = gens

    # 设置域的方法，返回一个新的分式域对象
    def set_domain(self, dom):
        """Make a new fraction field with given domain. """
        return self.__class__(dom, *self.gens)

    # 创建新元素的方法
    def new(self, element):
        return self.dtype(element, self.dom, len(self.gens) - 1)

    # 返回对象的字符串表示
    def __str__(self):
        return str(self.dom) + '(' + ','.join(map(str, self.gens)) + ')'

    # 返回对象的哈希值
    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom, self.gens))

    # 比较两个分式域对象是否相等的方法
    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, FractionField) and \
            self.dtype == other.dtype and self.dom == other.dom and self.gens == other.gens

    # 将对象转换为 SymPy 对象的方法
    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return (basic_from_dict(a.numer().to_sympy_dict(), *self.gens) /
                basic_from_dict(a.denom().to_sympy_dict(), *self.gens))

    # 将 SymPy 表达式转换为当前域中的元素的方法
    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        p, q = a.as_numer_denom()

        # 将分子和分母转换为字典表示
        num, _ = dict_from_basic(p, gens=self.gens)
        den, _ = dict_from_basic(q, gens=self.gens)

        # 转换字典中的每个值为当前域中的元素
        for k, v in num.items():
            num[k] = self.dom.from_sympy(v)

        for k, v in den.items():
            den[k] = self.dom.from_sympy(v)

        # 返回分子分母构造的新对象，并进行约分
        return self((num, den)).cancel()

    # 从整数转换为当前域中的元素的静态方法
    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    # 从整数转换为当前域中的元素的静态方法（Python 实现）
    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    # 从有理数转换为当前域中的元素的静态方法（Python 实现）
    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    # 从 GMPY 大整数转换为当前域中的元素的静态方法
    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    # 从 GMPY 大有理数转换为当前域中的元素的静态方法
    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))
    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        # 将 mpmath 的 mpf 对象转换为指定的 dtype 类型
        return K1(K1.dom.convert(a, K0))
    
    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert a ``DMF`` object to ``dtype``. """
        # 将 ``DMF`` 对象转换为指定的 dtype 类型
        if K1.gens == K0.gens:
            if K1.dom == K0.dom:
                # 如果生成元和域相同，则直接转换为列表
                return K1(a.to_list())
            else:
                # 否则，先进行域的转换，再转换为列表
                return K1(a.convert(K1.dom).to_list())
        else:
            # 对于生成元不同的情况，重新排序并转换系数的字典表示
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.gens, K1.gens)
    
            if K1.dom != K0.dom:
                # 如果域不同，则转换系数
                coeffs = [ K1.dom.convert(c, K0.dom) for c in coeffs ]
    
            return K1(dict(zip(monoms, coeffs)))
    
    def from_FractionField(K1, a, K0):
        """
        Convert a fraction field element to another fraction field.
    
        Examples
        ========
    
        >>> from sympy.polys.polyclasses import DMF
        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.abc import x
    
        >>> f = DMF(([ZZ(1), ZZ(2)], [ZZ(1), ZZ(1)]), ZZ)
    
        >>> QQx = QQ.old_frac_field(x)
        >>> ZZx = ZZ.old_frac_field(x)
    
        >>> QQx.from_FractionField(f, ZZx)
        DMF([1, 2], [1, 1], QQ)
    
        """
        if K1.gens == K0.gens:
            if K1.dom == K0.dom:
                # 如果生成元和域相同，则直接返回
                return a
            else:
                # 否则，分别转换分子和分母的域，并转换为列表
                return K1((a.numer().convert(K1.dom).to_list(),
                           a.denom().convert(K1.dom).to_list()))
        elif set(K0.gens).issubset(K1.gens):
            # 如果 K0 的生成元是 K1 的子集，则重新排序并转换系数的字典表示
            nmonoms, ncoeffs = _dict_reorder(
                a.numer().to_dict(), K0.gens, K1.gens)
            dmonoms, dcoeffs = _dict_reorder(
                a.denom().to_dict(), K0.gens, K1.gens)
    
            if K1.dom != K0.dom:
                # 如果域不同，则转换系数
                ncoeffs = [ K1.dom.convert(c, K0.dom) for c in ncoeffs ]
                dcoeffs = [ K1.dom.convert(c, K0.dom) for c in dcoeffs ]
    
            return K1((dict(zip(nmonoms, ncoeffs)), dict(zip(dmonoms, dcoeffs))))
    
    def get_ring(self):
        """Returns a ring associated with ``self``. """
        # 返回与当前对象关联的环
        from sympy.polys.domains import PolynomialRing
        return PolynomialRing(self.dom, *self.gens)
    
    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. `K[X]`. """
        # 返回一个多项式环，即 `K[X]`
        raise NotImplementedError('nested domains not allowed')
    
    def frac_field(self, *gens):
        """Returns a fraction field, i.e. `K(X)`. """
        # 返回一个分式域，即 `K(X)`
        raise NotImplementedError('nested domains not allowed')
    
    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        # 判断给定对象是否为正数
        return self.dom.is_positive(a.numer().LC())
    
    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        # 判断给定对象是否为负数
        return self.dom.is_negative(a.numer().LC())
    
    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        # 判断给定对象是否为非正数
        return self.dom.is_nonpositive(a.numer().LC())
    
    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        # 判断给定对象是否为非负数
        return self.dom.is_nonnegative(a.numer().LC())
    # 定义一个方法 `numer`，接受参数 `a`
    def numer(self, a):
        # 调用参数 `a` 的 `numer()` 方法，返回其分子部分
        """Returns numerator of ``a``. """
        return a.numer()

    # 定义一个方法 `denom`，接受参数 `a`
    def denom(self, a):
        # 调用参数 `a` 的 `denom()` 方法，返回其分母部分
        """Returns denominator of ``a``. """
        return a.denom()

    # 定义一个方法 `factorial`，接受参数 `a`
    def factorial(self, a):
        # 调用当前对象的 `dtype` 属性的 `factorial` 方法，传入参数 `a`，
        # 返回参数 `a` 的阶乘结果
        """Returns factorial of ``a``. """
        return self.dtype(self.dom.factorial(a))
```