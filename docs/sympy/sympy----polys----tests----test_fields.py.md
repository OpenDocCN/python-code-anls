# `D:\src\scipysrc\sympy\sympy\polys\tests\test_fields.py`

```
"""Test sparse rational functions. """

# 导入所需的库和模块
from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
# 导入测试用例所需的辅助函数和对象
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt

# 测试初始化 FracField 类的方法
def test_FracField___init__():
    F1 = FracField("x,y", ZZ, lex)
    F2 = FracField("x,y", ZZ, lex)
    F3 = FracField("x,y,z", ZZ, lex)

    assert F1.x == F1.gens[0]
    assert F1.y == F1.gens[1]
    assert F1.x == F2.x
    assert F1.y == F2.y
    assert F1.x != F3.x
    assert F1.y != F3.y

# 测试 FracField 类的哈希方法
def test_FracField___hash__():
    F, x, y, z = field("x,y,z", QQ)
    assert hash(F)

# 测试 FracField 类的相等性方法
def test_FracField___eq__():
    assert field("x,y,z", QQ)[0] == field("x,y,z", QQ)[0]
    assert field("x,y,z", QQ)[0] is field("x,y,z", QQ)[0]

    assert field("x,y,z", QQ)[0] != field("x,y,z", ZZ)[0]
    assert field("x,y,z", QQ)[0] is not field("x,y,z", ZZ)[0]

    assert field("x,y,z", ZZ)[0] != field("x,y,z", QQ)[0]
    assert field("x,y,z", ZZ)[0] is not field("x,y,z", QQ)[0]

    assert field("x,y,z", QQ)[0] != field("x,y", QQ)[0]
    assert field("x,y,z", QQ)[0] is not field("x,y", QQ)[0]

    assert field("x,y", QQ)[0] != field("x,y,z", QQ)[0]
    assert field("x,y", QQ)[0] is not field("x,y,z", QQ)[0]

# 测试 sfield 函数
def test_sfield():
    x = symbols("x")

    F = FracField((E, exp(exp(x)), exp(x)), ZZ, lex)
    e, exex, ex = F.gens
    assert sfield(exp(x)*exp(exp(x) + 1 + log(exp(x) + 3)/2)**2/(exp(x) + 3)) \
        == (F, e**2*exex**2*ex)

    F = FracField((x, exp(1/x), log(x), x**QQ(1, 3)), ZZ, lex)
    _, ex, lg, x3 = F.gens
    assert sfield(((x-3)*log(x)+4*x**2)*exp(1/x+log(x)/3)/x**2) == \
        (F, (4*F.x**2*ex + F.x*ex*lg - 3*ex*lg)/x3**5)

    F = FracField((x, log(x), sqrt(x + log(x))), ZZ, lex)
    _, lg, srt = F.gens
    assert sfield((x + 1) / (x * (x + log(x))**QQ(3, 2)) - 1/(x * log(x)**2)) \
        == (F, (F.x*lg**2 - F.x*srt + lg**2 - lg*srt)/
            (F.x**2*lg**2*srt + F.x*lg**3*srt))

# 测试 FracElement 类的哈希方法
def test_FracElement___hash__():
    F, x, y, z = field("x,y,z", QQ)
    assert hash(x*y/z)

# 测试 FracElement 类的复制方法
def test_FracElement_copy():
    F, x, y, z = field("x,y,z", ZZ)

    f = x*y/3*z
    g = f.copy()

    assert f == g
    g.numer[(1, 1, 1)] = 7
    assert f != g

# 测试 FracElement 类的表达式转换方法
def test_FracElement_as_expr():
    F, x, y, z = field("x,y,z", ZZ)
    f = (3*x**2*y - x*y*z)/(7*z**3 + 1)

    X, Y, Z = F.symbols
    g = (3*X**2*Y - X*Y*Z)/(7*Z**3 + 1)

    assert f != g
    assert f.as_expr() == g

    X, Y, Z = symbols("x,y,z")
    g = (3*X**2*Y - X*Y*Z)/(7*Z**3 + 1)

    assert f != g
    assert f.as_expr(X, Y, Z) == g

    raises(ValueError, lambda: f.as_expr(X))

# 测试从表达式创建 FracElement 对象的方法
def test_FracElement_from_expr():
    x, y, z = symbols("x,y,z")
    F, X, Y, Z = field((x, y, z), ZZ)

    f = F.from_expr(1)
    # 断言表达式 f == 1 并且 f 是 F.dtype 的实例
    assert f == 1 and isinstance(f, F.dtype)
    
    # 使用有理数 Rational(3, 7) 创建表达式 f，并断言 f 等于 F(3)/7 并且 f 是 F.dtype 的实例
    f = F.from_expr(Rational(3, 7))
    assert f == F(3)/7 and isinstance(f, F.dtype)
    
    # 使用变量 x 创建表达式 f，并断言 f 等于 X 并且 f 是 F.dtype 的实例
    f = F.from_expr(x)
    assert f == X and isinstance(f, F.dtype)
    
    # 使用有理数 Rational(3, 7) 乘以变量 x 创建表达式 f，并断言 f 等于 X*Rational(3, 7) 并且 f 是 F.dtype 的实例
    f = F.from_expr(Rational(3,7)*x)
    assert f == X*Rational(3, 7) and isinstance(f, F.dtype)
    
    # 使用表达式 1/x 创建表达式 f，并断言 f 等于 1/X 并且 f 是 F.dtype 的实例
    f = F.from_expr(1/x)
    assert f == 1/X and isinstance(f, F.dtype)
    
    # 使用表达式 x*y*z 创建表达式 f，并断言 f 等于 X*Y*Z 并且 f 是 F.dtype 的实例
    f = F.from_expr(x*y*z)
    assert f == X*Y*Z and isinstance(f, F.dtype)
    
    # 使用表达式 x*y/z 创建表达式 f，并断言 f 等于 X*Y/Z 并且 f 是 F.dtype 的实例
    f = F.from_expr(x*y/z)
    assert f == X*Y/Z and isinstance(f, F.dtype)
    
    # 使用表达式 x*y*z + x*y + x 创建表达式 f，并断言 f 等于 X*Y*Z + X*Y + X 并且 f 是 F.dtype 的实例
    f = F.from_expr(x*y*z + x*y + x)
    assert f == X*Y*Z + X*Y + X and isinstance(f, F.dtype)
    
    # 使用表达式 (x*y*z + x*y + x)/(x*y + 7) 创建表达式 f，并断言 f 等于 (X*Y*Z + X*Y + X)/(X*Y + 7) 并且 f 是 F.dtype 的实例
    f = F.from_expr((x*y*z + x*y + x)/(x*y + 7))
    assert f == (X*Y*Z + X*Y + X)/(X*Y + 7) and isinstance(f, F.dtype)
    
    # 使用表达式 x**3*y*z + x**2*y**7 + 1 创建表达式 f，并断言 f 等于 X**3*Y*Z + X**2*Y**7 + 1 并且 f 是 F.dtype 的实例
    f = F.from_expr(x**3*y*z + x**2*y**7 + 1)
    assert f == X**3*Y*Z + X**2*Y**7 + 1 and isinstance(f, F.dtype)
    
    # 使用 lambda 函数抛出 ValueError 异常来测试 F.from_expr(2**x) 的行为
    raises(ValueError, lambda: F.from_expr(2**x))
    # 使用 lambda 函数抛出 ValueError 异常来测试 F.from_expr(7*x + sqrt(2)) 的行为
    raises(ValueError, lambda: F.from_expr(7*x + sqrt(2)))
    
    # 断言表达式 isinstance(ZZ[2**x].get_field().convert(2**(-x)), FracElement)
    # 检查将 ZZ[2**x] 转换为分数域后，结果是否是 FracElement 的实例
    assert isinstance(ZZ[2**x].get_field().convert(2**(-x)), FracElement)
    
    # 断言表达式 isinstance(ZZ[x**2].get_field().convert(x**(-6)), FracElement)
    # 检查将 ZZ[x**2] 转换为分数域后，结果是否是 FracElement 的实例
    assert isinstance(ZZ[x**2].get_field().convert(x**(-6)), FracElement)
    
    # 断言表达式 isinstance(ZZ[exp(Rational(1, 3))].get_field().convert(E), FracElement)
    # 检查将 ZZ[exp(Rational(1, 3))] 转换为分数域后，结果是否是 FracElement 的实例
    assert isinstance(ZZ[exp(Rational(1, 3))].get_field().convert(E), FracElement)
def test_FracField_nested():
    # 定义符号变量
    a, b, x = symbols('a b x')
    # 创建有理分式域 F1 = QQ(a, b)
    F1 = ZZ.frac_field(a, b)
    # 在 F1 的基础上再创建有理分式域 F2 = QQ(a, b)(x)
    F2 = F1.frac_field(x)
    # 创建分式 frac = F2(a + b)
    frac = F2(a + b)
    # 断言分子等于 F1(x)[a + b]
    assert frac.numer == F1.poly_ring(x)(a + b)
    # 断言分子的系数列表为 [F1(a + b)]
    assert frac.numer.coeffs() == [F1(a + b)]
    # 断言分母等于 F1(x)[1]
    assert frac.denom == F1.poly_ring(x)(1)

    # 创建多项式环 F3 = ZZ[a, b]
    F3 = ZZ.poly_ring(a, b)
    # 在 F3 的基础上创建有理分式域 F4 = QQ(a, b)(x)
    F4 = F3.frac_field(x)
    # 创建分式 frac = F4(a + b)
    frac = F4(a + b)
    # 断言分子等于 F3(x)[a + b]
    assert frac.numer == F3.poly_ring(x)(a + b)
    # 断言分子的系数列表为 [F3(a + b)]
    assert frac.numer.coeffs() == [F3(a + b)]
    # 断言分母等于 F3(x)[1]
    assert frac.denom == F3.poly_ring(x)(1)

    # 创建分式 frac = F2(F3(a + b))
    frac = F2(F3(a + b))
    # 断言分子等于 F1(x)[a + b]
    assert frac.numer == F1.poly_ring(x)(a + b)
    # 断言分子的系数列表为 [F1(a + b)]
    assert frac.numer.coeffs() == [F1(a + b)]
    # 断言分母等于 F1(x)[1]
    assert frac.denom == F1.poly_ring(x)(1)

    # 创建分式 frac = F4(F1(a + b))
    frac = F4(F1(a + b))
    # 断言分子等于 F3(x)[a + b]
    assert frac.numer == F3.poly_ring(x)(a + b)
    # 断言分子的系数列表为 [F3(a + b)]
    assert frac.numer.coeffs() == [F3(a + b)]
    # 断言分母等于 F3(x)[1]
    assert frac.denom == F3.poly_ring(x)(1)


def test_FracElement__lt_le_gt_ge__():
    # 定义域 F 和符号变量 x, y
    F, x, y = field("x,y", ZZ)

    # 断言比较操作
    assert F(1) < 1/x < 1/x**2 < 1/x**3
    assert F(1) <= 1/x <= 1/x**2 <= 1/x**3

    assert -7/x < 1/x < 3/x < y/x < 1/x**2
    assert -7/x <= 1/x <= 3/x <= y/x <= 1/x**2

    assert 1/x**3 > 1/x**2 > 1/x > F(1)
    assert 1/x**3 >= 1/x**2 >= 1/x >= F(1)

    assert 1/x**2 > y/x > 3/x > 1/x > -7/x
    assert 1/x**2 >= y/x >= 3/x >= 1/x >= -7/x


def test_FracElement___neg__():
    # 定义域 F 和符号变量 x, y
    F, x, y = field("x,y", QQ)

    # 创建分式 f 和 g
    f = (7*x - 9)/y
    g = (-7*x + 9)/y

    # 断言取反操作
    assert -f == g
    assert -g == f


def test_FracElement___add__():
    # 定义域 F 和符号变量 x, y
    F, x, y = field("x,y", QQ)

    # 创建分式 f 和 g
    f, g = 1/x, 1/y
    # 断言加法操作
    assert f + g == g + f == (x + y)/(x*y)

    assert x + F.ring.gens[0] == F.ring.gens[0] + x == 2*x

    # 重新定义域 F 和符号变量 x, y
    F, x, y = field("x,y", ZZ)
    # 断言加法操作
    assert x + 3 == 3 + x
    assert x + QQ(3,7) == QQ(3,7) + x == (7*x + 3)/7

    # 创建环 Ruv 和符号变量 u, v
    Fuv, u, v = field("u,v", ZZ)
    # 创建有理分式域 Fxyzt 和符号变量 x, y, z, t
    Fxyzt, x, y, z, t = field("x,y,z,t", Fuv)

    # 创建分式 f
    f = (u*v + x)/(y + u*v)
    # 断言分子和分母的字典表示
    assert dict(f.numer) == {(1, 0, 0, 0): 1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0): u*v}

    # 创建环 Ruv 和符号变量 u, v
    Ruv, u, v = ring("u,v", ZZ)
    # 创建有理分式域 Fxyzt 和符号变量 x, y, z, t
    Fxyzt, x, y, z, t = field("x,y,z,t", Ruv)

    # 创建分式 f
    f = (u*v + x)/(y + u*v)
    # 断言分子和分母的字典表示
    assert dict(f.numer) == {(1, 0, 0, 0): 1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0): u*v}


def test_FracElement___sub__():
    # 定义域 F 和符号变量 x, y
    F, x, y = field("x,y", QQ)

    # 创建分式 f 和 g
    f, g = 1/x, 1/y
    # 断言减法操作
    assert f - g == (-x + y)/(x*y)

    assert x - F.ring.gens[0] == F.ring.gens[0] - x == 0

    # 重新定义域 F 和符号变量 x, y
    F, x, y = field("x,y", ZZ)
    # 断言减法操作
    assert x - 3 == -(3 - x)
    assert x - QQ(3,7) == -(QQ(3,7) - x) == (7*x - 3)/7

    # 创建环 Fuv 和符号变量 u, v
    Fuv, u, v = field("u,v", ZZ)
    # 创建有理分式域 Fxyzt 和符号变量 x, y, z, t
    Fxyzt, x, y, z, t = field("x,y,z,t", Fuv)

    # 创建分式 f
    f = (u*v - x)/(y - u*v)
    # 断言分子和分母的字典表示
    assert dict(f.numer) == {(1, 0, 0, 0): -1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0): -u*v}

    # 创建环 Ruv 和符号变量 u, v
    Ruv, u, v = ring("u,v", ZZ)
    # 创建有理分式域 Fxyzt 和符号变量 x, y, z, t
    Fxyzt, x, y, z, t = field("x,y,z,t", Ruv)

    # 创建分式 f
    f = (u*v
    # 断言，验证 f*g 等于 g*f 等于 1/(x*y)
    assert f*g == g*f == 1/(x*y)

    # 断言，验证 x 乘以 F 环的第一个生成元等于 F 环的第一个生成元乘以 x 等于 x 的平方
    assert x*F.ring.gens[0] == F.ring.gens[0]*x == x**2

    # 创建有理函数域 F，并定义变量 x 和 y
    F, x, y = field("x,y", ZZ)
    # 断言，验证 x 乘以整数 3 等于 3 乘以 x
    assert x*3 == 3*x
    # 断言，验证 x 乘以有理数 QQ(3,7) 等于 QQ(3,7) 乘以 x 等于 x 乘以有理数 Rational(3, 7)
    assert x*QQ(3,7) == QQ(3,7)*x == x*Rational(3, 7)

    # 创建有理函数域 Fuv，并定义变量 u 和 v
    Fuv, u, v = field("u,v", ZZ)
    # 在 Fuv 上创建扩展域 Fxyzt，并定义变量 x, y, z, t
    Fxyzt, x, y, z, t = field("x,y,z,t", Fuv)

    # 定义有理函数 f
    f = ((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)
    # 断言，验证 f 的分子部分转换为字典后应为 {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    assert dict(f.numer) == {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    # 断言，验证 f 的分母部分转换为字典后应为 {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u*v, (0, 0, 0, 0): -1}
    assert dict(f.denom) == {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u*v, (0, 0, 0, 0): -1}

    # 创建整数环 Ruv，并定义变量 u 和 v
    Ruv, u, v = ring("u,v", ZZ)
    # 在 Ruv 上创建扩展域 Fxyzt，并定义变量 x, y, z, t
    Fxyzt, x, y, z, t = field("x,y,z,t", Ruv)

    # 定义有理函数 f
    f = ((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)
    # 断言，验证 f 的分子部分转换为字典后应为 {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    assert dict(f.numer) == {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    # 断言，验证 f 的分母部分转换为字典后应为 {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u*v, (0, 0, 0, 0): -1}
    assert dict(f.denom) == {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u*v, (0, 0, 0, 0): -1}
def test_FracElement___truediv__():
    # 定义有理函数域 F, 变量 x, y 为有理数域 QQ 的符号
    F, x, y = field("x,y", QQ)

    # 定义有理函数 f = 1/x, g = 1/y
    f, g = 1/x, 1/y
    # 断言 f/g 的结果等于 y/x
    assert f/g == y/x

    # 断言 x 除以 F 中的第一个生成元等于 1，以及 F 中的第一个生成元除以 x 等于 1
    assert x/F.ring.gens[0] == F.ring.gens[0]/x == 1

    # 重新定义有理函数域 F, 变量 x, y 为整数环 ZZ 的符号
    F, x, y = field("x,y", ZZ)
    # 断言 x 乘以 3 等于 3 乘以 x
    assert x*3 == 3*x
    # 断言 x 除以 QQ(3,7) 等于 (QQ(3,7)/x) 的倒数，等于 x 乘以 Rational(7,3)
    assert x/QQ(3,7) == (QQ(3,7)/x)**-1 == x*Rational(7, 3)

    # 检查是否会引发 ZeroDivisionError 异常：x 除以 0
    raises(ZeroDivisionError, lambda: x/0)
    # 检查是否会引发 ZeroDivisionError 异常：1 除以 (x - x)
    raises(ZeroDivisionError, lambda: 1/(x - x))
    # 检查是否会引发 ZeroDivisionError 异常：x 除以 (x - x)
    raises(ZeroDivisionError, lambda: x/(x - x))

    # 定义有理数环 Ruv, 变量 u, v 为整数环 ZZ 的符号
    Fuv, u, v = field("u,v", ZZ)
    # 定义有理函数域 Fxyzt, 变量 x, y, z, t 为 Fuv 的符号
    Fxyzt, x, y, z, t = field("x,y,z,t", Fuv)

    # 定义有理函数 f = (u*v)/(x*y)
    f = (u*v)/(x*y)
    # 断言 f 的分子以字典形式表示为 {(0, 0, 0, 0): u*v}
    assert dict(f.numer) == {(0, 0, 0, 0): u*v}
    # 断言 f 的分母以字典形式表示为 {(1, 1, 0, 0): 1}
    assert dict(f.denom) == {(1, 1, 0, 0): 1}

    # 定义有理函数 g = (x*y)/(u*v)
    g = (x*y)/(u*v)
    # 断言 g 的分子以字典形式表示为 {(1, 1, 0, 0): 1}
    assert dict(g.numer) == {(1, 1, 0, 0): 1}
    # 断言 g 的分母以字典形式表示为 {(0, 0, 0, 0): u*v}
    assert dict(g.denom) == {(0, 0, 0, 0): u*v}

    # 定义整数环 Ruv, 变量 u, v 为整数环 ZZ 的符号
    Ruv, u, v = ring("u,v", ZZ)
    # 定义有理函数域 Fxyzt, 变量 x, y, z, t 为 Ruv 的符号
    Fxyzt, x, y, z, t = field("x,y,z,t", Ruv)

    # 定义有理函数 f = (u*v)/(x*y)
    f = (u*v)/(x*y)
    # 断言 f 的分子以字典形式表示为 {(0, 0, 0, 0): u*v}
    assert dict(f.numer) == {(0, 0, 0, 0): u*v}
    # 断言 f 的分母以字典形式表示为 {(1, 1, 0, 0): 1}
    assert dict(f.denom) == {(1, 1, 0, 0): 1}

    # 定义有理函数 g = (x*y)/(u*v)
    g = (x*y)/(u*v)
    # 断言 g 的分子以字典形式表示为 {(1, 1, 0, 0): 1}
    assert dict(g.numer) == {(1, 1, 0, 0): 1}
    # 断言 g 的分母以字典形式表示为 {(0, 0, 0, 0): u*v}
    assert dict(g.denom) == {(0, 0, 0, 0): u*v}

def test_FracElement___pow__():
    # 定义有理函数域 F, 变量 x, y 为有理数域 QQ 的符号
    F, x, y = field("x,y", QQ)

    # 定义有理函数 f, g = 1/x, 1/y
    f, g = 1/x, 1/y

    # 断言 f 的立方等于 1/x 的立方
    assert f**3 == 1/x**3
    # 断言 g 的立方等于 1/y 的立方
    assert g**3 == 1/y**3

    # 断言 (f*g) 的立方等于 1/(x**3 * y**3)
    assert (f*g)**3 == 1/(x**3 * y**3)
    # 断言 (f*g) 的负三次方等于 (x*y) 的立方
    assert (f*g)**-3 == (x*y)**3

    # 检查是否会引发 ZeroDivisionError 异常：(x - x) 的负三次方
    raises(ZeroDivisionError, lambda: (x - x)**-3)

def test_FracElement_diff():
    # 定义有理函数域 F, 变量 x, y, z 为整数环 ZZ 的符号
    F, x, y, z = field("x,y,z", ZZ)

    # 断言 ((x**2 + y)/(z + 1)) 对 x 的偏导数等于 2*x/(z + 1)
    assert ((x**2 + y)/(z + 1)).diff(x) == 2*x/(z + 1)

@XFAIL
def test_FracElement___call__():
    # 定义有理函数域 F, 变量 x, y, z 为整数环 ZZ 的符号
    F, x, y, z = field("x,y,z", ZZ)
    # 定义有理函数 f = (x**2 + 3*y)/z

    # 计算 f 在 (1, 1, 1) 处的值，并断言结果为 4，且返回结果不是 FracElement 类型
    r = f(1, 1, 1)
    assert r == 4 and not isinstance(r, FracElement)
    # 检查是否会引发 ZeroDivisionError 异常：f 在 (1, 1, 0) 处的值
    raises(ZeroDivisionError, lambda: f(1, 1, 0))

def test_FracElement_evaluate():
    # 定义有理函数域 F, 变量 x, y, z 为整数环 ZZ 的符号
    F, x, y, z = field("x,y,z", ZZ)
    # 定义有理函数域 Fyz, 变量 y, z 为整数环 ZZ 的符号
    Fyz = field("y,z", ZZ)[0]
    # 定义有理函数 f = (x**2 + 3*y)/z

    # 断言 f 在 x=0, y 保持不变时的求值结果等于 3*Fyz.y/Fyz.z
    assert f.evaluate(x, 0) == 3*Fyz.y/Fyz.z
    # 检查是否会引发 ZeroDivisionError 异常：f 在 z=0, x 保持不变时的求值
    raises(ZeroDivisionError, lambda: f.evaluate(z, 0))

def test_FracElement_subs():
    # 定义有理函数域 F, 变量 x, y, z 为整数环 ZZ 的
```