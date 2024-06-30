# `D:\src\scipysrc\sympy\sympy\polys\agca\tests\test_extensions.py`

```
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix

from sympy.testing.pytest import raises

from sympy.abc import x, y, t


def test_FiniteExtension():
    # 测试有限扩展域的功能

    # Gaussian integers
    A = FiniteExtension(Poly(x**2 + 1, x))
    # 断言域的秩为2
    assert A.rank == 2
    # 断言域的字符串表示为 'ZZ[x]/(x**2 + 1)'
    assert str(A) == 'ZZ[x]/(x**2 + 1)'
    # 获取域的生成元
    i = A.generator
    # 断言生成元的父对象为 A
    assert i.parent() is A

    # 断言生成元的平方等于 -1
    assert i*i == A(-1)
    # 断言生成元乘以空参数会引发 TypeError
    raises(TypeError, lambda: i*())

    # 断言域的基向量包含单位元和生成元 i
    assert A.basis == (A.one, i)
    # 断言 A(1) 等于域的单位元
    assert A(1) == A.one
    # 断言生成元的平方等于 -1
    assert i**2 == A(-1)
    # 断言生成元的平方不等于 Python 中的整数 -1（不进行强制类型转换）
    assert i**2 != -1  # no coercion
    # 断言复合表达式 (2 + i)*(1 - i) 的结果为 3 - i
    assert (2 + i)*(1 - i) == 3 - i
    # 断言 (1 + i) 的八次方等于 A(16)
    assert (1 + i)**8 == A(16)
    # 断言 A(1) 的逆元为 A(1)
    assert A(1).inverse() == A(1)
    # 断言 A(2) 没有定义逆元时会引发 NotImplementedError
    raises(NotImplementedError, lambda: A(2).inverse())

    # Finite field of order 27
    F = FiniteExtension(Poly(x**3 - x + 1, x, modulus=3))
    # 断言域的秩为 3
    assert F.rank == 3
    # 获取域的生成元 a，同时也是生成循环群 F - {0} 的元素
    a = F.generator
    # 断言域的基向量包含单位元和生成元 a 的幂次
    assert F.basis == (F(1), a, a**2)
    # 断言 a 的 27 次方等于 a
    assert a**27 == a
    # 断言 a 的 26 次方等于 F(1)
    assert a**26 == F(1)
    # 断言 a 的 13 次方等于 F(-1)
    assert a**13 == F(-1)
    # 断言 a 的 9 次方等于 a + 1
    assert a**9 == a + 1
    # 断言 a 的 3 次方等于 a - 1
    assert a**3 == a - 1
    # 断言 a 的 6 次方等于 a**2 + a + 1
    assert a**6 == a**2 + a + 1
    # 断言 F(x**2 + x) 的逆元为 1 - a
    assert F(x**2 + x).inverse() == 1 - a
    # 断言 F(x + 2)**(-1) 等于 F(x + 2) 的逆元
    assert F(x + 2)**(-1) == F(x + 2).inverse()
    # 断言 a**19 * a**(-19) 等于 F(1)
    assert a**19 * a**(-19) == F(1)
    # 断言 (a - 1) / (2*a**2 - 1) 等于 a**2 + 1
    assert (a - 1) / (2*a**2 - 1) == a**2 + 1
    # 断言 (a - 1) // (2*a**2 - 1) 等于 a**2 + 1
    assert (a - 1) // (2*a**2 - 1) == a**2 + 1
    # 断言 2 / (a**2 + 1) 等于 a**2 - a + 1
    assert 2/(a**2 + 1) == a**2 - a + 1
    # 断言 (a**2 + 1) / 2 等于 -a**2 - 1
    assert (a**2 + 1)/2 == -a**2 - 1
    # 断言当 F(0) 没有定义逆元时会引发 NotInvertible 异常
    raises(NotInvertible, lambda: F(0).inverse())

    # Function field of an elliptic curve
    K = FiniteExtension(Poly(t**2 - x**3 - x + 1, t, field=True))
    # 断言域的秩为 2
    assert K.rank == 2
    # 断言域的字符串表示为 'ZZ(x)[t]/(t**2 - x**3 - x + 1)'
    assert str(K) == 'ZZ(x)[t]/(t**2 - x**3 - x + 1)'
    # 获取域的生成元 y
    y = K.generator
    # 定义常数 c
    c = 1/(x**3 - x**2 + x - 1)
    # 断言 ((y + x)*(y - x)).inverse() 等于 K(c)
    assert ((y + x)*(y - x)).inverse() == K(c)
    # 断言 (y + x)*(y - x)*c 等于 K(1)，即 y + x 和 y - x 的显式逆元
    assert (y + x)*(y - x)*c == K(1)


def test_FiniteExtension_eq_hash():
    # 测试有限扩展域的相等性和哈希

    # 定义多项式 p1 和 p2
    p1 = Poly(x**2 - 2, x, domain=ZZ)
    p2 = Poly(x**2 - 2, x, domain=QQ)
    # 创建有限扩展域 K1 和 K2
    K1 = FiniteExtension(p1)
    K2 = FiniteExtension(p2)
    # 断言 K1 等于 FiniteExtension(Poly(x**2 - 2))
    assert K1 == FiniteExtension(Poly(x**2 - 2))
    # 断言 K2 不等于 FiniteExtension(Poly(x**2 - 2))
    assert K2 != FiniteExtension(Poly(x**2 - 2))
    # 断言 {K1, K2, FiniteExtension(p1)} 的长度为 2
    assert len({K1, K2, FiniteExtension(p1)}) == 2


def test_FiniteExtension_mod():
    # 测试有限扩展域的取模操作

    # 创建有理域上多项式的有限扩展域 K
    K = FiniteExtension(Poly(x**3 + 1, x, domain=QQ))
    # 定义变量 xf
    xf = K(x)
    # 断言 (xf**2 - 1) % 1 等于 K 的零元
    assert (xf**2 - 1) % 1 == K.zero
    # 断言 1 % (xf**2 - 1) 等于 K 的零元
    assert 1 % (xf**2 - 1) == K.zero
    # 断言 (xf**2 - 1) / (xf - 1) 等于 xf + 1
    assert (xf**2 - 1) / (xf - 1) == xf + 1
    # 断言 (xf**2 - 1) // (xf - 1) 等于 xf + 1
    assert (xf**2 - 1) // (xf - 1) == xf + 1
    # 断言 (xf**2 - 1) % (xf - 1) 等于 K 的零
    # 调用 raises 函数，期望它抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: (xf**2 - 1) % (xf - 1))
def test_FiniteExtension_from_sympy():
    # Test to_sympy/from_sympy functions of FiniteExtension class

    # 创建一个有限扩展对象 K，其基础为 x^3 + 1 多项式，系数域为整数环 ZZ
    K = FiniteExtension(Poly(x**3 + 1, x, domain=ZZ))
    
    # 在 K 上构造 x 的对象 xf
    xf = K(x)
    
    # 断言从 sympy 对象 x 转换到 K 中的对象与 xf 相等
    assert K.from_sympy(x) == xf
    
    # 断言从 K 中的对象 xf 转换到 sympy 对象中与 x 相等
    assert K.to_sympy(xf) == x


def test_FiniteExtension_set_domain():
    # Test set_domain method of FiniteExtension class

    # 创建两个有限扩展对象，分别使用整数环 ZZ 和有理数域 QQ 构造 x^2 + 1 多项式
    KZ = FiniteExtension(Poly(x**2 + 1, x, domain='ZZ'))
    KQ = FiniteExtension(Poly(x**2 + 1, x, domain='QQ'))
    
    # 断言将 KZ 对象的系数域设置为 QQ 后，得到的对象与 KQ 对象相等
    assert KZ.set_domain(QQ) == KQ


def test_FiniteExtension_exquo():
    # Test exquo method of FiniteExtension class

    # 创建一个有限扩展对象 K，其基础为 x^4 + 1 多项式
    K = FiniteExtension(Poly(x**4 + 1))
    
    # 在 K 上构造 x 的对象 xf
    xf = K(x)
    
    # 断言在 K 中计算 (xf^2 - 1) / (xf - 1) 的结果为 xf + 1
    assert K.exquo(xf**2 - 1, xf - 1) == xf + 1


def test_FiniteExtension_convert():
    # Test convert and convert_from methods of FiniteExtension class

    # 创建两个有限扩展对象 K1 和 K2，分别基于 x^2 + 1 和 x^2 - 1 多项式构造
    K1 = FiniteExtension(Poly(x**2 + 1))
    K2 = QQ[x]
    
    # 在 K1 和 K2 上分别构造 x 的对象 x1 和 x2
    x1, x2 = K1(x), K2(x)
    
    # 断言将 x2 转换为 K1 中的对象与 x1 相等
    assert K1.convert(x2) == x1
    
    # 断言将 x1 转换为 K2 中的对象与 x2 相等
    assert K2.convert(x1) == x2

    # 创建一个有限扩展对象 K，其基础为 x^2 - 1 多项式，系数域为有理数域 QQ
    K = FiniteExtension(Poly(x**2 - 1, domain=QQ))
    
    # 断言将 QQ(1/2) 转换为 K 中的对象等于 K.one / 2
    assert K.convert_from(QQ(1, 2), QQ) == K.one/2


def test_FiniteExtension_division_ring():
    # Test division properties of FiniteExtension class over different domains

    # 创建四个有限扩展对象，分别基于 x^2 - 1 多项式，系数域为 QQ, ZZ, QQ[t], QQ(t)
    KQ = FiniteExtension(Poly(x**2 - 1, x, domain=QQ))
    KZ = FiniteExtension(Poly(x**2 - 1, x, domain=ZZ))
    KQt = FiniteExtension(Poly(x**2 - 1, x, domain=QQ[t]))
    KQtf = FiniteExtension(Poly(x**2 - 1, x, domain=QQ.frac_field(t)))
    
    # 断言 KQ 是一个域（即可逆元素存在），KZ 不是域
    assert KQ.is_Field is True
    assert KZ.is_Field is False
    assert KQt.is_Field is False
    assert KQtf.is_Field is True
    
    # 对每个 K 进行如下断言：
    for K in KQ, KZ, KQt, KQtf:
        xK = K.convert(x)
        
        # 断言 xK 除以 K.one 等于 xK 自身
        assert xK / K.one == xK
        
        # 断言 xK 整除 K.one 等于 xK 自身
        assert xK // K.one == xK
        
        # 断言 xK 对 K.one 取模等于 K.zero
        assert xK % K.one == K.zero
        
        # 断言除以 K.zero 会抛出 ZeroDivisionError 异常
        raises(ZeroDivisionError, lambda: xK / K.zero)
        raises(ZeroDivisionError, lambda: xK // K.zero)
        raises(ZeroDivisionError, lambda: xK % K.zero)
        
        # 如果 K 是域，则进行如下断言：
        if K.is_Field:
            assert xK / xK == K.one
            assert xK // xK == K.one
            assert xK % xK == K.zero
        else:
            # 如果 K 不是域，则对除法操作会抛出 NotImplementedError 异常
            raises(NotImplementedError, lambda: xK / xK)
            raises(NotImplementedError, lambda: xK // xK)
            raises(NotImplementedError, lambda: xK % xK)


def test_FiniteExtension_Poly():
    # Test Polynomial operations in FiniteExtension class

    # 创建一个有限扩展对象 K，其基础为 x^2 - 2 多项式
    K = FiniteExtension(Poly(x**2 - 2))
    
    # 创建一个多项式 p，系数域为 K
    p = Poly(x, y, domain=K)
    
    # 断言 p 的系数域为 K，p 的表达式为 x
    assert p.domain == K
    assert p.as_expr() == x
    
    # 断言 p 的平方的表达式为 2
    assert (p**2).as_expr() == 2

    # 创建一个有限扩展对象 K，其基础为 x^2 - 2 多项式，系数域为有理数域 QQ
    K = FiniteExtension(Poly(x**2 - 2, x, domain=QQ))
    
    # 创建一个 K2 对象，基于 t^2 - 2 多项式，其系数域为 K
    K2 = FiniteExtension(Poly(t**2 - 2, t, domain=K))
    
    # 断言 K2 的字符串表示为 'QQ[x]/(x**2 - 2)[t]/(t**2 - 2)'
    assert str(K2) == 'QQ[x]/(x**2 - 2)[t]/(t**2 - 2)'
    
    # 创建一个 K2 中 x + t 的对象 eK
    eK = K2.convert(x + t)
    
    # 断言将 eK 转换为 sympy 对象等于 x + t
    assert K2.to_sympy(eK) == x + t
    
    # 断言将 eK 的平方转换为 sympy 对象等于 4 + 2*x*t
    assert K2.to_sympy(eK ** 2) == 4 + 2*x*t
    
    # 创建一个多项式 p，系数域为 K2
    p = Poly(x + t, y, domain=K2)
    
    # 断言 p 的平方为 Poly(4 + 2*x*t, y, domain=K2)
    assert p**2 == Poly(4 + 2*x*t, y, domain=K2)


def test_FiniteExtension_sincos_jacobian():
    # Test using FiniteExtension to compute the Jacobian involving sin and cos

    # 定义符号 r, p, t
    r, p, t = symbols('rho, phi, theta')
    
    # 创建一个矩阵元素列表，包含 sin 和 cos 不同符号的组合
    elements = [
        [sin(p)*cos(t), r*cos(p)*cos(t), -r*sin(p)*sin(t)],
        [sin(p)*sin(t), r*cos(p)*sin(t),  r*sin(p)*cos(t)],
        [       cos(p),       -r*sin(p),                0],
    ]
    # 定义一个函数 `make_extension`，用于生成有限扩展域对象 `K`
    def make_extension(K):
        # 使用给定的多项式生成一个新的有限扩展域对象 `K`
        K = FiniteExtension(Poly(sin(p)**2 + cos(p)**2 - 1, sin(p), domain=K[cos(p)]))
        # 使用另一个多项式生成另一个新的有限扩展域对象 `K`
        K = FiniteExtension(Poly(sin(t)**2 + cos(t)**2 - 1, sin(t), domain=K[cos(t)]))
        # 返回生成的有限扩展域对象 `K`
        return K

    # 使用函数 `make_extension` 分别生成两个有限扩展域对象 `Ksc1` 和 `Ksc2`
    Ksc1 = make_extension(ZZ[r])
    # 从 `ZZ` 上生成一个有限扩展域对象 `K`，然后应用 `r` 作为变量
    Ksc2 = make_extension(ZZ)[r]

    # 对每个有限扩展域对象 `Ksc1` 和 `Ksc2` 执行以下操作
    for K in [Ksc1, Ksc2]:
        # 将 `elements` 中的每个元素转换为当前有限扩展域 `K` 中的元素，形成一个新的列表
        elements_K = [[K.convert(e) for e in row] for row in elements]
        # 创建一个域矩阵 `J`，使用转换后的元素列表 `elements_K`，并指定尺寸和域 `K`
        J = DomainMatrix(elements_K, (3, 3), K)
        # 计算矩阵 `J` 的特征多项式，提取其常数项并乘以 `-K.one` 的立方
        det = J.charpoly()[-1] * (-K.one)**3
        # 断言特征多项式的常数项等于 `r^2 * sin(p)` 在域 `K` 中的转换结果
        assert det == K.convert(r**2 * sin(p))
```