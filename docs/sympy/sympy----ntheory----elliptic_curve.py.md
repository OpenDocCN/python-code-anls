# `D:\src\scipysrc\sympy\sympy\ntheory\elliptic_curve.py`

```
# 从 sympy.core.numbers 模块导入无穷大 oo
# 从 sympy.core.symbol 模块导入 symbols 符号函数
# 从 sympy.polys.domains 模块导入有限域 FiniteField, QQ, RationalField, FF
# 从 sympy.polys.polytools 模块导入多项式类 Poly
# 从 sympy.solvers.solvers 模块导入 solve 解方程函数
# 从 sympy.utilities.iterables 模块导入 is_sequence 判断是否为序列函数
# 从 sympy.utilities.misc 模块导入 as_int 将输入转换为整数函数
# 从 .factor_ 模块导入 divisors 因子函数
# 从 .residue_ntheory 模块导入 polynomial_congruence 多项式同余函数
# 导入 EllipticCurvePoint 类
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.polys.polytools import Poly
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence

# 定义一个椭圆曲线类 EllipticCurve
class EllipticCurve:
    """
    Create the following Elliptic Curve over domain.

    `y^{2} + a_{1} x y + a_{3} y = x^{3} + a_{2} x^{2} + a_{4} x + a_{6}`

    The default domain is ``QQ``. If no coefficient ``a1``, ``a2``, ``a3``,
    is given then it creates a curve with the following form:

    `y^{2} = x^{3} + a_{4} x + a_{6}`

    Examples
    ========

    References
    ==========

    .. [1] J. Silverman "A Friendly Introduction to Number Theory" Third Edition
    .. [2] https://mathworld.wolfram.com/EllipticDiscriminant.html
    .. [3] G. Hardy, E. Wright "An Introduction to the Theory of Numbers" Sixth Edition

    """

    # 构造函数，初始化椭圆曲线的系数和域
    def __init__(self, a4, a6, a1=0, a2=0, a3=0, modulus=0):
        # 如果 modulus 为 0，则使用有理数域 QQ
        if modulus == 0:
            domain = QQ
        else:
            # 否则使用有限域 FF(modulus)
            domain = FF(modulus)
        # 将 a1, a2, a3, a4, a6 转换为指定域上的元素
        a1, a2, a3, a4, a6 = map(domain.convert, (a1, a2, a3, a4, a6))
        self._domain = domain
        self.modulus = modulus
        # 计算判别式
        b2 = a1**2 + 4 * a2
        b4 = 2 * a4 + a1 * a3
        b6 = a3**2 + 4 * a6
        b8 = a1**2 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3**2 - a4**2
        self._b2, self._b4, self._b6, self._b8 = b2, b4, b6, b8
        self._discrim = -b2**2 * b8 - 8 * b4**3 - 27 * b6**2 + 9 * b2 * b4 * b6
        self._a1 = a1
        self._a2 = a2
        self._a3 = a3
        self._a4 = a4
        self._a6 = a6
        x, y, z = symbols('x y z')
        self.x, self.y, self.z = x, y, z
        # 创建多项式对象，表示椭圆曲线方程
        self._poly = Poly(y**2*z + a1*x*y*z + a3*y*z**2 - x**3 - a2*x**2*z - a4*x*z**2 - a6*z**3, domain=domain)
        # 如果域是有限域，则设置秩为 0；如果是有理数域，则设置秩为 None
        if isinstance(self._domain, FiniteField):
            self._rank = 0
        elif isinstance(self._domain, RationalField):
            self._rank = None

    # 调用实例时返回一个椭圆曲线点对象 EllipticCurvePoint
    def __call__(self, x, y, z=1):
        return EllipticCurvePoint(x, y, z, self)

    # 判断点是否在椭圆曲线上
    def __contains__(self, point):
        if is_sequence(point):
            if len(point) == 2:
                z1 = 1
            else:
                z1 = point[2]
            x1, y1 = point[:2]
        elif isinstance(point, EllipticCurvePoint):
            x1, y1, z1 = point.x, point.y, point.z
        else:
            raise ValueError('Invalid point.')
        # 如果特征为 0 且 z1 为 0，则认为点在曲线上
        if self.characteristic == 0 and z1 == 0:
            return True
        # 否则检查点是否满足椭圆曲线方程
        return self._poly.subs({self.x: x1, self.y: y1, self.z: z1}) == 0

    # 返回椭圆曲线的字符串表示
    def __repr__(self):
        return self._poly.__repr__()
    # 返回椭圆曲线的最小魏尔斯特拉斯方程
    def minimal(self):
        # 获取曲线的特征（素数）
        char = self.characteristic
        # 如果特征是2，则返回当前曲线对象
        if char == 2:
            return self
        # 如果特征是3，则返回一个新的椭圆曲线对象，使用简化的系数
        if char == 3:
            return EllipticCurve(self._b4/2, self._b6/4, a2=self._b2/4, modulus=self.modulus)
        # 计算常数 c4 和 c6
        c4 = self._b2**2 - 24*self._b4
        c6 = -self._b2**3 + 36*self._b2*self._b4 - 216*self._b6
        # 返回一个新的椭圆曲线对象，使用最小化的魏尔斯特拉斯方程的系数
        return EllipticCurve(-27*c4, -54*c6, modulus=self.modulus)

    # 返回有限域上曲线的点集合
    def points(self):
        # 获取曲线的特征（素数）
        char = self.characteristic
        # 初始化一个空集合，用于存储所有点
        all_pt = set()
        # 如果特征大于等于1
        if char >= 1:
            # 对于每一个从0到特征-1的整数
            for i in range(char):
                # 替换多项式中的变量 x 和 z，得到一个同余方程
                congruence_eq = self._poly.subs({self.x: i, self.z: 1}).expr
                # 解决同余方程，得到解集合
                sol = polynomial_congruence(congruence_eq, char)
                # 将解集合中的点添加到 all_pt 集合中
                all_pt.update((i, num) for num in sol)
            # 返回所有点的集合
            return all_pt
        else:
            # 如果特征小于1，则抛出异常，因为点的数量是无限的
            raise ValueError("Infinitely many points")

    # 返回给定 x 坐标上的曲线点集合
    def points_x(self, x):
        # 初始化一个空列表，用于存储点坐标
        pt = []
        # 如果定义域是有理数集 QQ
        if self._domain == QQ:
            # 解多项式中以 x 为变量的方程，得到 y 坐标
            for y in solve(self._poly.subs(self.x, x)):
                pt.append((x, y))
        else:
            # 否则，计算在给定 x 和 z=1 条件下的同余方程
            congruence_eq = self._poly.subs({self.x: x, self.z: 1}).expr
            # 解同余方程，得到所有可能的 y 坐标
            for y in polynomial_congruence(congruence_eq, self.characteristic):
                pt.append((x, y))
        # 返回所有点的坐标列表
        return pt
    def torsion_points(self):
        """
        Return torsion points of curve over Rational number.

        Return point objects those are finite order.
        According to Nagell-Lutz theorem, torsion point p(x, y)
        x and y are integers, either y = 0 or y**2 is divisor
        of discriminent. According to Mazur's theorem, there are
        at most 15 points in torsion collection.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(-43, 166)
        >>> sorted(e2.torsion_points())
        [(-5, -16), (-5, 16), O, (3, -8), (3, 8), (11, -32), (11, 32)]

        """
        # 如果椭圆曲线的特征大于0，则抛出异常，有限域没有挠点
        if self.characteristic > 0:
            raise ValueError("No torsion point for Finite Field.")
        # 初始列表，包含无穷远点
        l = [EllipticCurvePoint.point_at_infinity(self)]
        # 对于解析曲线的 y=0 的多项式方程解
        for xx in solve(self._poly.subs({self.y: 0, self.z: 1})):
            # 如果解是有理数
            if xx.is_rational:
                # 将解作为 x 坐标构造挠点并加入列表
                l.append(self(xx, 0))
        # 对于判别式的所有因子
        for i in divisors(self.discriminant, generator=True):
            j = int(i**.5)
            # 如果 j 的平方等于当前因子
            if j**2 == i:
                # 对于解析曲线的 y=j 的多项式方程解
                for xx in solve(self._poly.subs({self.y: j, self.z: 1})):
                    # 如果解不是有理数，则继续下一个解
                    if not xx.is_rational:
                        continue
                    # 构造挠点 p 并检查其阶是否有限
                    p = self(xx, j)
                    if p.order() != oo:
                        # 将 p 及其负点加入列表
                        l.extend([p, -p])
        # 返回所有挠点的列表
        return l

    @property
    def characteristic(self):
        """
        Return domain characteristic.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(-43, 166)
        >>> e2.characteristic
        0

        """
        # 返回域的特征
        return self._domain.characteristic()

    @property
    def discriminant(self):
        """
        Return curve discriminant.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(0, 17)
        >>> e2.discriminant
        -124848

        """
        # 返回曲线的判别式
        return int(self._discrim)

    @property
    def is_singular(self):
        """
        Return True if curve discriminant is equal to zero.
        """
        # 如果曲线判别式为零，则返回 True
        return self.discriminant == 0

    @property
    def j_invariant(self):
        """
        Return curve j-invariant.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e1 = EllipticCurve(-2, 0, 0, 1, 1)
        >>> e1.j_invariant
        1404928/389

        """
        # 计算并返回曲线的 j 不变量
        c4 = self._b2**2 - 24*self._b4
        return self._domain.to_sympy(c4**3 / self._discrim)
    # 计算椭圆曲线上点的数量，基于有限域的定义
    def order(self):
        """
        Number of points in Finite field.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(1, 0, modulus=19)
        >>> e2.order
        19

        """
        # 如果特征为 0，即特征是有限域中的特性，抛出未实现错误
        if self.characteristic == 0:
            raise NotImplementedError("Still not implemented")
        # 返回点集的长度，即椭圆曲线上的点的数量
        return len(self.points())

    @property
    # 计算椭圆曲线上无穷阶点的独立数量
    def rank(self):
        """
        Number of independent points of infinite order.

        For Finite field, it must be 0.
        """
        # 如果已经计算过排名，则直接返回已有的值
        if self._rank is not None:
            return self._rank
        # 如果尚未实现计算排名，抛出未实现错误
        raise NotImplementedError("Still not implemented")
    """
    Point of Elliptic Curve

    Examples
    ========

    >>> from sympy.ntheory.elliptic_curve import EllipticCurve
    >>> e1 = EllipticCurve(-17, 16)
    >>> p1 = e1(0, -4, 1)
    >>> p2 = e1(1, 0)
    >>> p1 + p2
    (15, -56)
    >>> e3 = EllipticCurve(-1, 9)
    >>> e3(1, -3) * 3
    (664/169, 17811/2197)
    >>> (e3(1, -3) * 3).order()
    oo
    >>> e2 = EllipticCurve(-2, 0, 0, 1, 1)
    >>> p = e2(-1,1)
    >>> q = e2(0, -1)
    >>> p+q
    (4, 8)
    >>> p-q
    (1, 0)
    >>> 3*p-5*q
    (328/361, -2800/6859)
    """

    @staticmethod
    # 返回代表无穷远点的 EllipticCurvePoint 对象
    def point_at_infinity(curve):
        return EllipticCurvePoint(0, 1, 0, curve)

    def __init__(self, x, y, z, curve):
        dom = curve._domain.convert
        self.x = dom(x)  # 将 x 坐标转换为椭圆曲线域的元素
        self.y = dom(y)  # 将 y 坐标转换为椭圆曲线域的元素
        self.z = dom(z)  # 将 z 坐标转换为椭圆曲线域的元素
        self._curve = curve  # 设置椭圆曲线对象
        self._domain = self._curve._domain  # 设置椭圆曲线的域
        if not self._curve.__contains__(self):  # 如果椭圆曲线不包含当前点，则抛出异常
            raise ValueError("The curve does not contain this point")

    def __add__(self, p):
        if self.z == 0:
            return p  # 如果当前点是无穷远点，则返回另一个点 p
        if p.z == 0:
            return self  # 如果点 p 是无穷远点，则返回当前点
        x1, y1 = self.x/self.z, self.y/self.z  # 将当前点转换为 affine 坐标系下的坐标
        x2, y2 = p.x/p.z, p.y/p.z  # 将点 p 转换为 affine 坐标系下的坐标
        a1 = self._curve._a1
        a2 = self._curve._a2
        a3 = self._curve._a3
        a4 = self._curve._a4
        a6 = self._curve._a6
        if x1 != x2:
            slope = (y1 - y2) / (x1 - x2)  # 计算斜率
            yint = (y1 * x2 - y2 * x1) / (x2 - x1)  # 计算 y 截距
        else:
            if (y1 + y2) == 0:
                return self.point_at_infinity(self._curve)  # 如果两点垂直且在同一条直线上，则返回无穷远点
            slope = (3 * x1**2 + 2*a2*x1 + a4 - a1*y1) / (a1 * x1 + a3 + 2 * y1)  # 计算斜率
            yint = (-x1**3 + a4*x1 + 2*a6 - a3*y1) / (a1*x1 + a3 + 2*y1)  # 计算 y 截距
        x3 = slope**2 + a1*slope - a2 - x1 - x2  # 计算新点 x 坐标
        y3 = -(slope + a1) * x3 - yint - a3  # 计算新点 y 坐标
        return self._curve(x3, y3, 1)  # 返回曲线上的新点

    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)  # 比较两个点的大小

    def __mul__(self, n):
        n = as_int(n)  # 将 n 转换为整数
        r = self.point_at_infinity(self._curve)  # 初始化结果点为无穷远点
        if n == 0:
            return r  # 如果 n 为 0，则返回无穷远点
        if n < 0:
            return -self * -n  # 如果 n 为负数，则返回自身的相反数乘以正数 n
        p = self
        while n:
            if n & 1:
                r = r + p  # 如果 n 的最低位为 1，则将当前点加到结果点上
            n >>= 1  # 右移一位，相当于 n //= 2
            p = p + p  # 当前点加倍
        return r  # 返回乘法结果点

    def __rmul__(self, n):
        return self * n  # 右乘运算符的实现

    def __neg__(self):
        return EllipticCurvePoint(self.x, -self.y - self._curve._a1*self.x - self._curve._a3, self.z, self._curve)
        # 返回当前点的相反点

    def __repr__(self):
        if self.z == 0:
            return 'O'  # 如果当前点是无穷远点，则返回字符串 'O'
        dom = self._curve._domain
        try:
            return '({}, {})'.format(dom.to_sympy(self.x), dom.to_sympy(self.y))  # 尝试将点的坐标转换为 SymPy 的表示
        except TypeError:
            pass
        return '({}, {})'.format(self.x, self.y)  # 否则返回点的坐标

    def __sub__(self, other):
        return self.__add__(-other)  # 减法运算符的实现，等同于加上相反数
    # 定义一个方法 `order`，用于计算点在椭圆曲线上的阶数 n，使得 nP = 0。

        """
        Return point order n where nP = 0.
        返回满足 nP = 0 的点的阶数 n。
        
        """
        # 如果当前点在无穷远处（Z 坐标为 0），返回阶数为 1
        if self.z == 0:
            return 1
        # 如果当前点的 Y 坐标为 0，表明点 P 与 -P 重合，返回阶数为 2
        if self.y == 0:  # P = -P
            return 2
        # 计算点的双倍 p
        p = self * 2
        # 如果点的双倍 p 的 Y 坐标等于当前点 P 的负数 Y 坐标，返回阶数为 3
        if p.y == -self.y:  # 2P = -P
            return 3
        # 初始化计数器 i 为 2
        i = 2
        # 如果当前点不在 QQ 域上，进行以下循环
        if self._domain != QQ:
            # 当 p 的 X 和 Y 坐标都为整数时，反复计算 P + p，并增加计数器 i
            while int(p.x) == p.x and int(p.y) == p.y:
                p = self + p
                i += 1
                # 如果计算结果的 Z 坐标为 0，返回当前的计数器 i
                if p.z == 0:
                    return i
            # 若超过阶数 12 仍未找到满足条件的点，返回无穷大 oo
            return oo
        # 在 QQ 域上进行以下循环
        while p.x.numerator == p.x and p.y.numerator == p.y:
            p = self + p
            i += 1
            # 如果计数器超过 12，返回无穷大 oo
            if i > 12:
                return oo
            # 如果计算结果的 Z 坐标为 0，返回当前的计数器 i
            if p.z == 0:
                return i
        # 若超过阶数 12 仍未找到满足条件的点，返回无穷大 oo
        return oo
```