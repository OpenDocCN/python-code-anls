# `D:\src\scipysrc\sympy\sympy\polys\agca\extensions.py`

```
# 导入需要的模块和类
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import (CoercionFailed, NotInvertible,
        GeneratorsError)
from sympy.polys.polytools import Poly
from sympy.printing.defaults import DefaultPrinting

# 定义有限扩展元素类，继承自域元素类和默认打印类
class ExtensionElement(DomainElement, DefaultPrinting):
    """
    Element of a finite extension.

    A class of univariate polynomials modulo the ``modulus``
    of the extension ``ext``. It is represented by the
    unique polynomial ``rep`` of lowest degree. Both
    ``rep`` and the representation ``mod`` of ``modulus``
    are of class DMP.

    """
    # 定义槽位以限制属性
    __slots__ = ('rep', 'ext')

    # 初始化方法，接受表示和扩展对象作为参数
    def __init__(self, rep, ext):
        self.rep = rep  # 设置扩展元素的多项式表示
        self.ext = ext  # 设置元素所属的扩展域对象

    # 父域访问方法，返回元素所属的扩展域对象
    def parent(f):
        return f.ext

    # 将元素表示为 SymPy 表达式的方法
    def as_expr(f):
        return f.ext.to_sympy(f)

    # 布尔值转换方法，判断元素是否为非零
    def __bool__(f):
        return bool(f.rep)

    # 正数运算符重载，返回元素本身
    def __pos__(f):
        return f

    # 负数运算符重载，返回相反数的扩展元素
    def __neg__(f):
        return ExtElem(-f.rep, f.ext)

    # 获取多项式表示的私有方法，支持元素与其他元素的加法操作
    def _get_rep(f, g):
        if isinstance(g, ExtElem):
            if g.ext == f.ext:
                return g.rep
            else:
                return None
        else:
            try:
                g = f.ext.convert(g)
                return g.rep
            except CoercionFailed:
                return None

    # 加法运算符重载，支持元素与其他元素的加法操作
    def __add__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep + rep, f.ext)
        else:
            return NotImplemented

    # 右加法运算符重载，与加法运算符重载相同
    __radd__ = __add__

    # 减法运算符重载，支持元素与其他元素的减法操作
    def __sub__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep - rep, f.ext)
        else:
            return NotImplemented

    # 右减法运算符重载，支持其他元素与元素的减法操作
    def __rsub__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(rep - f.rep, f.ext)
        else:
            return NotImplemented

    # 乘法运算符重载，支持元素与其他元素的乘法操作
    def __mul__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem((f.rep * rep) % f.ext.mod, f.ext)
        else:
            return NotImplemented

    # 右乘法运算符重载，与乘法运算符重载相同
    __rmul__ = __mul__

    # 除法检查方法，检查是否可以进行除法操作
    def _divcheck(f):
        """Raise if division is not implemented for this divisor"""
        if not f:
            raise NotInvertible('Zero divisor')
        elif f.ext.is_Field:
            return True
        elif f.rep.is_ground and f.ext.domain.is_unit(f.rep.LC()):
            return True
        else:
            # Some cases like (2*x + 2)/2 over ZZ will fail here. It is
            # unclear how to implement division in general if the ground
            # domain is not a field so for now it was decided to restrict the
            # implementation to division by invertible constants.
            msg = (f"Can not invert {f} in {f.ext}. "
                    "Only division by invertible constants is implemented.")
            raise NotImplementedError(msg)
    def inverse(f):
        """求乘法逆元。

        Raises
        ======

        NotInvertible
            如果元素是零除法器。

        """
        # 调用 _divcheck 方法，确保除法操作合法
        f._divcheck()

        if f.ext.is_Field:
            # 如果扩展域是一个域，使用扩展元素的表示来求取其乘法逆元
            invrep = f.rep.invert(f.ext.mod)
        else:
            # 否则，使用环 R 的 exquo 方法，计算 f.rep 对 1/f.rep 的商
            R = f.ext.ring
            invrep = R.exquo(R.one, f.rep)

        # 返回一个新的扩展元素，其表示为 invrep，扩展为 f.ext
        return ExtElem(invrep, f.ext)

    def __truediv__(f, g):
        # 获取 g 的表示形式
        rep = f._get_rep(g)
        if rep is None:
            return NotImplemented
        # 构造一个扩展元素 g
        g = ExtElem(rep, f.ext)

        try:
            # 求 g 的乘法逆元
            ginv = g.inverse()
        except NotInvertible:
            # 如果求逆失败，抛出 ZeroDivisionError 异常
            raise ZeroDivisionError(f"{f} / {g}")

        # 返回 f 乘以 g 的乘法逆元
        return f * ginv

    __floordiv__ = __truediv__

    def __rtruediv__(f, g):
        try:
            # 尝试将 g 转换为与 f.ext 兼容的类型
            g = f.ext.convert(g)
        except CoercionFailed:
            return NotImplemented
        # 返回 g 除以 f 的结果
        return g / f

    __rfloordiv__ = __rtruediv__

    def __mod__(f, g):
        # 获取 g 的表示形式
        rep = f._get_rep(g)
        if rep is None:
            return NotImplemented
        # 构造一个扩展元素 g
        g = ExtElem(rep, f.ext)

        try:
            # 调用 g 的 _divcheck 方法，确保除法操作合法
            g._divcheck()
        except NotInvertible:
            # 如果除法操作不合法，抛出 ZeroDivisionError 异常
            raise ZeroDivisionError(f"{f} % {g}")

        # 返回零元素，因为在定义域内，除法操作总是精确的，没有余数
        return f.ext.zero

    def __rmod__(f, g):
        try:
            # 尝试将 g 转换为与 f.ext 兼容的类型
            g = f.ext.convert(g)
        except CoercionFailed:
            return NotImplemented
        # 返回 g 对 f 求模的结果
        return g % f

    def __pow__(f, n):
        if not isinstance(n, int):
            raise TypeError("指数应为 'int' 类型")
        if n < 0:
            try:
                # 如果 n 为负数，尝试计算 f 的逆元，并取其绝对值
                f, n = f.inverse(), -n
            except NotImplementedError:
                # 如果无法求逆，抛出 ValueError
                raise ValueError("负指数未定义")

        b = f.rep
        m = f.ext.mod
        r = f.ext.one.rep
        while n > 0:
            if n % 2:
                # 如果 n 是奇数，更新 r 为 (r*b) % m
                r = (r*b) % m
            # 更新 b 为 b*b % m，n 取其整数除以 2
            b = (b*b) % m
            n //= 2

        # 返回一个新的扩展元素，其表示为 r，扩展为 f.ext
        return ExtElem(r, f.ext)

    def __eq__(f, g):
        if isinstance(g, ExtElem):
            # 如果 g 是 ExtElem 类型，比较 f 的表示和扩展域与 g 的相同性
            return f.rep == g.rep and f.ext == g.ext
        else:
            return NotImplemented

    def __ne__(f, g):
        # 返回 f 和 g 是否不相等的结果
        return not f == g

    def __hash__(f):
        # 返回 f 的哈希值，基于其表示和扩展域
        return hash((f.rep, f.ext))

    def __str__(f):
        # 导入字符串表示方法 sstr，返回 f 的表达式的字符串形式
        from sympy.printing.str import sstr
        return sstr(f.as_expr())

    __repr__ = __str__

    @property
    def is_ground(f):
        # 返回 f.rep 是否为一个地面元素
        return f.rep.is_ground

    def to_ground(f):
        # 将 f.rep 转换为其列表的唯一元素，并返回
        [c] = f.rep.to_list()
        return c
# 定义一个别名指向 ExtensionElement
ExtElem = ExtensionElement

# 定义一个有限域扩展类，继承自 Domain
class MonogenicFiniteExtension(Domain):
    r"""
    由一个整数元素生成的有限扩展。

    生成元由参数 `mod` 表示的首一一元多项式定义。

    一个更短的别名是 `FiniteExtension`。

    Examples
    ========

    二次整数环 $\mathbb{Z}[\sqrt2]$:

    >>> from sympy import Symbol, Poly
    >>> from sympy.polys.agca.extensions import FiniteExtension
    >>> x = Symbol('x')
    >>> R = FiniteExtension(Poly(x**2 - 2)); R
    ZZ[x]/(x**2 - 2)
    >>> R.rank
    2
    >>> R(1 + x)*(3 - 2*x)
    x - 1

    由原始多项式 $x^3 + x^2 + 2$（在 $\mathbb{Z}_5$ 上）定义的有限域 $GF(5^3)$。

    >>> F = FiniteExtension(Poly(x**3 + x**2 + 2, modulus=5)); F
    GF(5)[x]/(x**3 + x**2 + 2)
    >>> F.basis
    (1, x, x**2)
    >>> F(x + 3)/(x**2 + 2)
    -2*x**2 + x + 2

    椭圆曲线的函数域：

    >>> t = Symbol('t')
    >>> FiniteExtension(Poly(t**2 - x**3 - x + 1, t, field=True))
    ZZ(x)[t]/(t**2 - x**3 - x + 1)

    """
    
    # 声明这是一个有限扩展
    is_FiniteExtension = True

    # 类型定义为 ExtensionElement
    dtype = ExtensionElement

    # 初始化方法，接受一个 mod 参数作为输入
    def __init__(self, mod):
        # 检查 mod 是否为一元多项式
        if not (isinstance(mod, Poly) and mod.is_univariate):
            raise TypeError("modulus must be a univariate Poly")
        
        # 将 mod 转换为首一形式，auto=False 可能会引发除法不精确的异常
        mod = mod.monic(auto=False)
        
        # 计算多项式的次数，并存储在 rank 中
        self.rank = mod.degree()
        # 存储原始多项式在 modulus 中
        self.modulus = mod
        # 存储 mod 的 DMP 表示
        self.mod = mod.rep

        # 获取 mod 的定义域
        self.domain = dom = mod.domain
        # 创建 mod 的旧多项式环
        self.ring = dom.old_poly_ring(*mod.gens)

        # 将零元素转换为当前环的表示
        self.zero = self.convert(self.ring.zero)
        # 将一元素转换为当前环的表示
        self.one = self.convert(self.ring.one)

        # 获取生成元和符号
        gen = self.ring.gens[0]
        self.symbol = self.ring.symbols[0]
        # 将生成元转换为当前环的表示，并构建基底元组
        self.generator = self.convert(gen)
        self.basis = tuple(self.convert(gen**i) for i in range(self.rank))

        # XXX: 这里可能需要检查 mod 是否为不可约多项式
        # 判断定义域是否为域
        self.is_Field = self.domain.is_Field

    # 返回一个新的 ExtensionElement 对象，代表输入参数 arg 的扩展
    def new(self, arg):
        rep = self.ring.convert(arg)
        return ExtElem(rep % self.mod, self)

    # 判断两个有限扩展是否相等
    def __eq__(self, other):
        if not isinstance(other, FiniteExtension):
            return False
        return self.modulus == other.modulus

    # 计算对象的哈希值
    def __hash__(self):
        return hash((self.__class__.__name__, self.modulus))

    # 返回对象的字符串表示形式
    def __str__(self):
        return "%s/(%s)" % (self.ring, self.modulus.as_expr())

    # 使用 __str__ 的结果作为 __repr__ 的返回值
    __repr__ = __str__

    # 返回定义域是否具有零特征
    @property
    def has_CharacteristicZero(self):
        return self.domain.has_CharacteristicZero

    # 返回定义域的特征
    def characteristic(self):
        return self.domain.characteristic()

    # 将输入多项式 f 转换为当前环的表示
    def convert(self, f, base=None):
        rep = self.ring.convert(f, base)
        return ExtElem(rep % self.mod, self)
    # 将给定的多项式 f 转换到当前有限扩展域上，并取模于当前模数
    def convert_from(self, f, base):
        rep = self.ring.convert(f, base)
        # 创建一个新的有限扩展域元素，使用取模后的剩余类 rep % self.mod
        return ExtElem(rep % self.mod, self)

    # 将当前有限扩展域元素转换为 SymPy 多项式
    def to_sympy(self, f):
        return self.ring.to_sympy(f.rep)

    # 从 SymPy 多项式 f 转换到当前有限扩展域元素
    def from_sympy(self, f):
        return self.convert(f)

    # 设置当前有限扩展域的定义域为 K，返回一个新的扩展域对象
    def set_domain(self, K):
        mod = self.modulus.set_domain(K)
        return self.__class__(mod)

    # 从当前有限扩展域中移除指定的符号，返回一个新的扩展域对象
    def drop(self, *symbols):
        if self.symbol in symbols:
            # 如果要移除的符号是当前扩展域的生成元，抛出异常
            raise GeneratorsError('Can not drop generator from FiniteExtension')
        # 从当前定义域中移除指定的符号，得到新的定义域 K，并设置当前扩展域的定义域为 K
        K = self.domain.drop(*symbols)
        return self.set_domain(K)

    # 在当前有限扩展域中计算 f 除以 g 的商
    def quo(self, f, g):
        return self.exquo(f, g)

    # 在当前有限扩展域中计算 f 除以 g 的剩余类，并返回一个新的扩展域元素
    def exquo(self, f, g):
        rep = self.ring.exquo(f.rep, g.rep)
        # 创建一个新的有限扩展域元素，使用取模后的剩余类 rep % self.mod
        return ExtElem(rep % self.mod, self)

    # 判断给定的元素 a 是否为负数，总是返回 False
    def is_negative(self, a):
        return False

    # 判断给定的元素 a 是否为单位元（可逆元）
    def is_unit(self, a):
        # 如果当前扩展域是一个域（Field），则元素 a 非零即为单位元
        if self.is_Field:
            return bool(a)
        # 否则，如果 a 是一个地面元（ground），则检查其是否为当前定义域的单位元
        elif a.is_ground:
            return self.domain.is_unit(a.to_ground())
# 将MonogenicFiniteExtension类赋值给FiniteExtension，作为别名或者兼容性命名。
FiniteExtension = MonogenicFiniteExtension
```