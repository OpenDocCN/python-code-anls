# `D:\src\scipysrc\sympy\sympy\polys\polyquinticconst.py`

```
# 导入符号计算相关的模块和函数
from sympy.core import Symbol
from sympy.core.evalf import N
from sympy.core.numbers import I, Rational
from sympy.functions import sqrt
from sympy.polys.polytools import Poly
from sympy.utilities import public

# 创建符号变量 x
x = Symbol('x')

@public
class PolyQuintic:
    """Solvable quintic polynomial class with special functions"""

    # 初始化方法，接收一个多项式对象，并提取其系数
    def __init__(self, poly):
        _, _, self.p, self.q, self.r, self.s = poly.all_coeffs()
        
        # 计算四个复数根
        self.zeta1 = Rational(-1, 4) + (sqrt(5)/4) + I*sqrt((sqrt(5)/8) + Rational(5, 8))
        self.zeta2 = (-sqrt(5)/4) - Rational(1, 4) + I*sqrt((-sqrt(5)/8) + Rational(5, 8))
        self.zeta3 = (-sqrt(5)/4) - Rational(1, 4) - I*sqrt((-sqrt(5)/8) + Rational(5, 8))
        self.zeta4 = Rational(-1, 4) + (sqrt(5)/4) - I*sqrt((sqrt(5)/8) + Rational(5, 8))

    # 计算特定的多项式 f20，返回一个多项式对象
    @property
    def f20(self):
        # 提取系数 p, q, r, s
        p, q, r, s = self.p, self.q, self.r, self.s
        # 构造多项式 f20
        f20 = q**8 - 13*p*q**6*r + p**5*q**2*r**2 + 65*p**2*q**4*r**2 - 4*p**6*r**3 - 128*p**3*q**2*r**3 + 17*q**4*r**3 + 48*p**4*r**4 - 16*p*q**2*r**4 - 192*p**2*r**5 + 256*r**6 - 4*p**5*q**3*s - 12*p**2*q**5*s + 18*p**6*q*r*s + 12*p**3*q**3*r*s - 124*q**5*r*s + 196*p**4*q*r**2*s + 590*p*q**3*r**2*s - 160*p**2*q*r**3*s - 1600*q*r**4*s - 27*p**7*s**2 - 150*p**4*q**2*s**2 - 125*p*q**4*s**2 - 99*p**5*r*s**2 - 725*p**2*q**2*r*s**2 + 1200*p**3*r**2*s**2 + 3250*q**2*r**2*s**2 - 2000*p*r**3*s**2 - 1250*p*q*r*s**3 + 3125*p**2*s**4 - 9375*r*s**4-(2*p*q**6 - 19*p**2*q**4*r + 51*p**3*q**2*r**2 - 3*q**4*r**2 - 32*p**4*r**3 - 76*p*q**2*r**3 + 256*p**2*r**4 - 512*r**5 + 31*p**3*q**3*s + 58*q**5*s - 117*p**4*q*r*s - 105*p*q**3*r*s - 260*p**2*q*r**2*s + 2400*q*r**3*s + 108*p**5*s**2 + 325*p**2*q**2*s**2 - 525*p**3*r*s**2 - 2750*q**2*r*s**2 + 500*p*r**2*s**2 - 625*p*q*s**3 + 3125*s**4)*x+(p**2*q**4 - 6*p**3*q**2*r - 8*q**4*r + 9*p**4*r**2 + 76*p*q**2*r**2 - 136*p**2*r**3 + 400*r**4 - 50*p*q**3*s + 90*p**2*q*r*s - 1400*q*r**2*s + 625*q**2*s**2 + 500*p*r*s**2)*x**2-(2*q**4 - 21*p*q**2*r + 40*p**2*r**2 - 160*r**3 + 15*p**2*q*s + 400*q*r*s - 125*p*s**2)*x**3+(2*p*q**2 - 6*p**2*r + 40*r**2 - 50*q*s)*x**4 + 8*r*x**5 + x**6
        return Poly(f20, x)

    # 这里缺少进一步的代码，需要根据实际情况补全
    # 定义函数 F，计算多项式表达式的值
    def F(self):
        # 将类中的变量解包赋值给局部变量 p, q, r, s
        p, q, r, s = self.p, self.q, self.r, self.s
        # 多项式 F 的具体表达式，包含多个项
        F = 4*p**6*q**6 + 59*p**3*q**8 + 216*q**10 - 36*p**7*q**4*r - 623*p**4*q**6*r - 2610*p*q**8*r + 81*p**8*q**2*r**2 + 2015*p**5*q**4*r**2 + 10825*p**2*q**6*r**2 - 1800*p**6*q**2*r**3 - 17500*p**3*q**4*r**3 + 625*q**6*r**3 + 10000*p**4*q**2*r**4 + 108*p**8*q**3*s + 1584*p**5*q**5*s + 5700*p**2*q**7*s - 486*p**9*q*r*s - 9720*p**6*q**3*r*s - 45050*p**3*q**5*r*s - 9000*q**7*r*s + 10800*p**7*q*r**2*s + 92500*p**4*q**3*r**2*s + 32500*p*q**5*r**2*s - 60000*p**5*q*r**3*s - 50000*p**2*q**3*r**3*s + 729*p**10*s**2 + 12150*p**7*q**2*s**2 + 60000*p**4*q**4*s**2 + 93750*p*q**6*s**2 - 18225*p**8*r*s**2 - 175500*p**5*q**2*r*s**2 - 478125*p**2*q**4*r*s**2 + 135000*p**6*r**2*s**2 + 850000*p**3*q**2*r**2*s**2 + 15625*q**4*r**2*s**2 - 250000*p**4*r**3*s**2 + 225000*p**3*q**3*s**3 + 175000*q**5*s**3 - 1012500*p**4*q*r*s**3 - 1187500*p*q**3*r*s**3 + 1250000*p**2*q*r**2*s**3 + 928125*p**5*s**4 + 1875000*p**2*q**2*s**4 - 2812500*p**3*r*s**4 - 390625*q**2*r*s**4 - 9765625*s**6
        return F

    # 定义函数 l0，计算多项式 a 在 theta 处的值除以多项式 F
    def l0(self, theta):
        F = self.F  # 获取函数 F 的值
        a = self.a  # 获取多项式 a
        # 计算多项式 a 在 theta 处的值除以 F，并返回结果
        l0 = Poly(a, x).eval(theta)/F
        return l0

    # 定义函数 T，计算多项式列表 b 在 theta 处的值除以 d 乘以多项式 F 的一半
    def T(self, theta, d):
        F = self.F  # 获取函数 F 的值
        T = [0]*5  # 创建长度为 5 的零列表 T
        b = self.b  # 获取多项式列表 b
        # 计算不同子列表的 b 在 theta 处的值除以对应系数乘以 F 的一半，并存储在 T 中
        T[1] = -Poly(b[1], x).eval(theta)/(2*F)
        T[2] = Poly(b[2], x).eval(theta)/(2*d*F)
        T[3] = Poly(b[3], x).eval(theta)/(2*F)
        T[4] = Poly(b[4], x).eval(theta)/(2*d*F)
        return T

    # 定义函数 order，计算多项式 o 在 theta 处的值除以 d 乘以多项式 F，并返回结果的数值化版本
    def order(self, theta, d):
        F = self.F  # 获取函数 F 的值
        o = self.o  # 获取多项式 o
        # 计算多项式 o 在 theta 处的值除以 d 乘以 F，并返回数值化结果
        order = Poly(o, x).eval(theta)/(d*F)
        return N(order)

    # 定义函数 uv，计算多项式 c 在 theta 处的值除以 2*d 乘以函数 F 的值，并返回结果的数值化版本
    def uv(self, theta, d):
        c = self.c  # 获取多项式 c
        u = self.q*Rational(-25, 2)  # 计算 u 的值
        v = Poly(c, x).eval(theta)/(2*d*self.F)  # 计算 v 的值
        return N(u), N(v)  # 返回 u 和 v 的数值化结果

    @property
    def zeta(self):
        # 返回类中定义的 zeta1, zeta2, zeta3, zeta4 属性的列表
        return [self.zeta1, self.zeta2, self.zeta3, self.zeta4]
```