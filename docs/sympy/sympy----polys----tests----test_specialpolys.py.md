# `D:\src\scipysrc\sympy\sympy\polys\tests\test_specialpolys.py`

```
# 导入所需的 SymPy 模块和函数
from sympy.core.add import Add
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import prime
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import permute_signs
from sympy.testing.pytest import raises

# 导入 SymPy 中定义的特殊多项式函数
from sympy.polys.specialpolys import (
    swinnerton_dyer_poly,
    cyclotomic_poly,
    symmetric_poly,
    random_poly,
    interpolating_poly,
    fateman_poly_F_1,
    dmp_fateman_poly_F_1,
    fateman_poly_F_2,
    dmp_fateman_poly_F_2,
    fateman_poly_F_3,
    dmp_fateman_poly_F_3,
)

# 导入 SymPy 中的符号变量
from sympy.abc import x, y, z


# 定义测试函数：测试 Swinnerton-Dyer 多项式生成函数
def test_swinnerton_dyer_poly():
    # 测试当输入参数为0时，是否引发 ValueError 异常
    raises(ValueError, lambda: swinnerton_dyer_poly(0, x))

    # 测试生成 Swinnerton-Dyer 多项式，使用 polys=True 时返回多项式对象 Poly(x**2 - 2)
    assert swinnerton_dyer_poly(1, x, polys=True) == Poly(x**2 - 2)

    # 测试生成 Swinnerton-Dyer 多项式，不使用 polys=True 时返回表达式 x**2 - 2
    assert swinnerton_dyer_poly(1, x) == x**2 - 2
    assert swinnerton_dyer_poly(2, x) == x**4 - 10*x**2 + 1
    assert swinnerton_dyer_poly(3, x) == x**8 - 40*x**6 + 352*x**4 - 960*x**2 + 576

    # 测试根据 polys=True 返回多项式对象时，其所有根的精确值是否正确
    p = [sqrt(prime(i)) for i in range(1, 5)]
    assert str([i.n(3) for i in swinnerton_dyer_poly(4, polys=True).all_roots()]) == str(sorted([Add(*i).n(3) for i in permute_signs(p)]))


# 定义测试函数：测试 Cyclotomic 多项式生成函数
def test_cyclotomic_poly():
    # 测试当输入参数为0时，是否引发 ValueError 异常
    raises(ValueError, lambda: cyclotomic_poly(0, x))

    # 测试生成 Cyclotomic 多项式，使用 polys=True 时返回多项式对象 Poly(x - 1)
    assert cyclotomic_poly(1, x, polys=True) == Poly(x - 1)

    # 测试生成 Cyclotomic 多项式，不使用 polys=True 时返回表达式 x - 1 等其他情况
    assert cyclotomic_poly(1, x) == x - 1
    assert cyclotomic_poly(2, x) == x + 1
    assert cyclotomic_poly(3, x) == x**2 + x + 1
    assert cyclotomic_poly(4, x) == x**2 + 1
    assert cyclotomic_poly(5, x) == x**4 + x**3 + x**2 + x + 1
    assert cyclotomic_poly(6, x) == x**2 - x + 1


# 定义测试函数：测试对称多项式生成函数
def test_symmetric_poly():
    # 测试当输入参数为负数和大于等于5时，是否引发 ValueError 异常
    raises(ValueError, lambda: symmetric_poly(-1, x, y, z))
    raises(ValueError, lambda: symmetric_poly(5, x, y, z))

    # 测试生成对称多项式，使用 polys=True 时返回多项式对象 Poly(x + y + z)
    assert symmetric_poly(1, x, y, z, polys=True) == Poly(x + y + z)
    assert symmetric_poly(1, (x, y, z), polys=True) == Poly(x + y + z)

    # 测试生成对称多项式，不使用 polys=True 时返回表达式等其他情况
    assert symmetric_poly(0, x, y, z) == 1
    assert symmetric_poly(1, x, y, z) == x + y + z
    assert symmetric_poly(2, x, y, z) == x*y + x*z + y*z
    assert symmetric_poly(3, x, y, z) == x*y*z


# 定义测试函数：测试随机多项式生成函数
def test_random_poly():
    # 测试生成随机多项式，检查其次数是否为指定值，系数是否在指定范围内
    poly = random_poly(x, 10, -100, 100, polys=False)
    assert Poly(poly).degree() == 10
    assert all(-100 <= coeff <= 100 for coeff in Poly(poly).coeffs()) is True

    # 测试生成随机多项式（多项式对象形式），检查其次数是否为指定值，系数是否在指定范围内
    poly = random_poly(x, 10, -100, 100, polys=True)
    assert poly.degree() == 10
    assert all(-100 <= coeff <= 100 for coeff in poly.coeffs()) is True


# 定义测试函数：测试插值多项式生成函数
def test_interpolating_poly():
    x0, x1, x2, x3, y0, y1, y2, y3 = symbols('x:4, y:4')

    # 测试插值多项式生成函数，当输入参数为0和1时的情况
    assert interpolating_poly(0, x) == 0
    assert interpolating_poly(1, x) == y0

    # 测试插值多项式生成函数，当输入参数为2时的情况
    assert interpolating_poly(2, x) == y0*(x - x1)/(x0 - x1) + y1*(x - x0)/(x1 - x0)
    # 断言：验证对于插值多项式的特定阶数和变量 x 的计算结果是否符合预期
    assert interpolating_poly(3, x) == \
        y0*(x - x1)*(x - x2)/((x0 - x1)*(x0 - x2)) + \
        y1*(x - x0)*(x - x2)/((x1 - x0)*(x1 - x2)) + \
        y2*(x - x0)*(x - x1)/((x2 - x0)*(x2 - x1))

    # 断言：验证对于插值多项式的另一个阶数和变量 x 的计算结果是否符合预期
    assert interpolating_poly(4, x) == \
        y0*(x - x1)*(x - x2)*(x - x3)/((x0 - x1)*(x0 - x2)*(x0 - x3)) + \
        y1*(x - x0)*(x - x2)*(x - x3)/((x1 - x0)*(x1 - x2)*(x1 - x3)) + \
        y2*(x - x0)*(x - x1)*(x - x3)/((x2 - x0)*(x2 - x1)*(x2 - x3)) + \
        y3*(x - x0)*(x - x1)*(x - x2)/((x3 - x0)*(x3 - x1)*(x3 - x2))

    # 断言：验证对于不合法参数配置时，插值多项式函数是否会引发 ValueError 异常
    raises(ValueError, lambda:
        interpolating_poly(2, x, (x, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, x, (x + y, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, x + y, (x, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, 3, (4, 5), (6, 7)))
    raises(ValueError, lambda:
        interpolating_poly(2, 3, (4, 5), (6, 7, 8)))

    # 断言：验证对于特定的插值阶数和给定数据点，插值多项式的计算结果是否符合预期
    assert interpolating_poly(0, x, (1, 2), (3, 4)) == 0
    assert interpolating_poly(1, x, (1, 2), (3, 4)) == 3
    assert interpolating_poly(2, x, (1, 2), (3, 4)) == x + 2
# 定义测试函数，测试 fateman_poly_F_1 函数
def test_fateman_poly_F_1():
    # 调用 fateman_poly_F_1 函数，返回三个多项式 f, g, h
    f, g, h = fateman_poly_F_1(1)
    # 调用 dmp_fateman_poly_F_1 函数，传入参数 1 和 ZZ，返回三个多项式 F, G, H
    F, G, H = dmp_fateman_poly_F_1(1, ZZ)

    # 断言：将 f, g, h 的表示转换为列表，与 F, G, H 的列表表示进行比较
    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    # 再次调用 fateman_poly_F_1 函数，传入参数 3，返回三个多项式 f, g, h
    f, g, h = fateman_poly_F_1(3)
    # 再次调用 dmp_fateman_poly_F_1 函数，传入参数 3 和 ZZ，返回三个多项式 F, G, H
    F, G, H = dmp_fateman_poly_F_1(3, ZZ)

    # 断言：将 f, g, h 的表示转换为列表，与 F, G, H 的列表表示进行比较
    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]


# 定义测试函数，测试 fateman_poly_F_2 函数
def test_fateman_poly_F_2():
    # 调用 fateman_poly_F_2 函数，返回三个多项式 f, g, h
    f, g, h = fateman_poly_F_2(1)
    # 调用 dmp_fateman_poly_F_2 函数，传入参数 1 和 ZZ，返回三个多项式 F, G, H
    F, G, H = dmp_fateman_poly_F_2(1, ZZ)

    # 断言：将 f, g, h 的表示转换为列表，与 F, G, H 的列表表示进行比较
    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    # 再次调用 fateman_poly_F_2 函数，传入参数 3，返回三个多项式 f, g, h
    f, g, h = fateman_poly_F_2(3)
    # 再次调用 dmp_fateman_poly_F_2 函数，传入参数 3 和 ZZ，返回三个多项式 F, G, H
    F, G, H = dmp_fateman_poly_F_2(3, ZZ)

    # 断言：将 f, g, h 的表示转换为列表，与 F, G, H 的列表表示进行比较
    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]


# 定义测试函数，测试 fateman_poly_F_3 函数
def test_fateman_poly_F_3():
    # 调用 fateman_poly_F_3 函数，返回三个多项式 f, g, h
    f, g, h = fateman_poly_F_3(1)
    # 调用 dmp_fateman_poly_F_3 函数，传入参数 1 和 ZZ，返回三个多项式 F, G, H
    F, G, H = dmp_fateman_poly_F_3(1, ZZ)

    # 断言：将 f, g, h 的表示转换为列表，与 F, G, H 的列表表示进行比较
    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    # 再次调用 fateman_poly_F_3 函数，传入参数 3，返回三个多项式 f, g, h
    f, g, h = fateman_poly_F_3(3)
    # 再次调用 dmp_fateman_poly_F_3 函数，传入参数 3 和 ZZ，返回三个多项式 F, G, H
    F, G, H = dmp_fateman_poly_F_3(3, ZZ)

    # 断言：将 f, g, h 的表示转换为列表，与 F, G, H 的列表表示进行比较
    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]
```