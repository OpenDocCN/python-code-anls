# `D:\src\scipysrc\sympy\sympy\polys\tests\test_sqfreetools.py`

```
"""Tests for square-free decomposition algorithms and related tools. """

from sympy.polys.rings import ring  # 导入ring函数，用于创建多项式环
from sympy.polys.domains import FF, ZZ, QQ  # 导入FF, ZZ, QQ，分别表示有限域、整数环和有理数环
from sympy.polys.specialpolys import f_polys  # 导入f_polys函数，用于生成特殊多项式

from sympy.testing.pytest import raises  # 导入raises函数，用于测试时引发异常
from sympy.external.gmpy import MPQ  # 导入MPQ，外部依赖的GMPY库

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = f_polys()  # 调用f_polys函数生成七个特殊多项式

def test_dup_sqf():
    R, x = ring("x", ZZ)  # 创建整数环上的多项式环R，并指定变量为x

    assert R.dup_sqf_part(0) == 0  # 测试多项式0的平方因子部分是否为0
    assert R.dup_sqf_p(0) is True  # 测试多项式0是否为平方自由多项式

    assert R.dup_sqf_part(7) == 1  # 测试多项式7的平方因子部分是否为1
    assert R.dup_sqf_p(7) is True  # 测试多项式7是否为平方自由多项式

    assert R.dup_sqf_part(2*x + 2) == x + 1  # 测试多项式2*x + 2的平方因子部分是否为x + 1
    assert R.dup_sqf_p(2*x + 2) is True  # 测试多项式2*x + 2是否为平方自由多项式

    assert R.dup_sqf_part(x**3 + x + 1) == x**3 + x + 1  # 测试多项式x**3 + x + 1的平方因子部分是否为x**3 + x + 1
    assert R.dup_sqf_p(x**3 + x + 1) is True  # 测试多项式x**3 + x + 1是否为平方自由多项式

    assert R.dup_sqf_part(-x**3 + x + 1) == x**3 - x - 1  # 测试多项式-x**3 + x + 1的平方因子部分是否为x**3 - x - 1
    assert R.dup_sqf_p(-x**3 + x + 1) is True  # 测试多项式-x**3 + x + 1是否为平方自由多项式

    assert R.dup_sqf_part(2*x**3 + 3*x**2) == 2*x**2 + 3*x  # 测试多项式2*x**3 + 3*x**2的平方因子部分是否为2*x**2 + 3*x
    assert R.dup_sqf_p(2*x**3 + 3*x**2) is False  # 测试多项式2*x**3 + 3*x**2是否为平方自由多项式

    assert R.dup_sqf_part(-2*x**3 + 3*x**2) == 2*x**2 - 3*x  # 测试多项式-2*x**3 + 3*x**2的平方因子部分是否为2*x**2 - 3*x
    assert R.dup_sqf_p(-2*x**3 + 3*x**2) is False  # 测试多项式-2*x**3 + 3*x**2是否为平方自由多项式

    assert R.dup_sqf_list(0) == (0, [])  # 测试多项式0的平方自由列表是否为(0, [])
    assert R.dup_sqf_list(1) == (1, [])  # 测试多项式1的平方自由列表是否为(1, [])

    assert R.dup_sqf_list(x) == (1, [(x, 1)])  # 测试多项式x的平方自由列表是否为(1, [(x, 1)])
    assert R.dup_sqf_list(2*x**2) == (2, [(x, 2)])  # 测试多项式2*x**2的平方自由列表是否为(2, [(x, 2)])
    assert R.dup_sqf_list(3*x**3) == (3, [(x, 3)])  # 测试多项式3*x**3的平方自由列表是否为(3, [(x, 3)])

    assert R.dup_sqf_list(-x**5 + x**4 + x - 1) == \
        (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])  # 测试多项式-x**5 + x**4 + x - 1的平方自由列表是否为(-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dup_sqf_list(x**8 + 6*x**6 + 12*x**4 + 8*x**2) == \
        (1, [(x, 2), (x**2 + 2, 3)])  # 测试多项式x**8 + 6*x**6 + 12*x**4 + 8*x**2的平方自由列表是否为(1, [(x, 2), (x**2 + 2, 3)])

    assert R.dup_sqf_list(2*x**2 + 4*x + 2) == (2, [(x + 1, 2)])  # 测试多项式2*x**2 + 4*x + 2的平方自由列表是否为(2, [(x + 1, 2)])

    R, x = ring("x", QQ)  # 创建有理数环上的多项式环R，并指定变量为x
    assert R.dup_sqf_list(2*x**2 + 4*x + 2) == (2, [(x + 1, 2)])  # 测试多项式2*x**2 + 4*x + 2在有理数环上的平方自由列表是否为(2, [(x + 1, 2)])

    R, x = ring("x", FF(2))  # 创建特征为2的有限域上的多项式环R，并指定变量为x
    assert R.dup_sqf_list(x**2 + 1) == (1, [(x + 1, 2)])  # 测试多项式x**2 + 1在特征为2的有限域上的平方自由列表是否为(1, [(x + 1, 2)])

    R, x = ring("x", FF(3))  # 创建特征为3的有限域上的多项式环R，并指定变量为x
    assert R.dup_sqf_list(x**10 + 2*x**7 + 2*x**4 + x) == \
        (1, [(x, 1),
             (x + 1, 3),
             (x + 2, 6)])  # 测试多项式x**10 + 2*x**7 + 2*x**4 + x在特征为3的有限域上的平方自由列表是否为(1, [(x, 1), (x + 1, 3), (x + 2, 6)])

    R1, x = ring("x", ZZ)  # 创建新的整数环上的多项式环R1，并指定变量为x
    R2, y = ring("y", FF(3))  # 创建特征为3的有限域上的多项式环R2，并指定变量为y

    f = x**3 + 1  # 定义多项式f
    g = y**3 + 1  # 定义多项式g

    assert R1.dup_sqf_part(f) == f  # 测试多项式f的平方因子部分是否为f
    assert R2.dup_sqf_part(g) == y + 1  # 测试多项式g的平方因子部分是否为y + 1

    assert R1.dup_sqf_p(f) is True  # 测试多项式f是否为平方自由多项式
    assert R2.dup_sqf_p(g) is False  # 测试多项式g是否为平方自由多项式

    R, x, y = ring("x,y",
    # 断言 f_2**2 不是平方自由的
    assert R.dmp_sqf_p(f_2**2) is False
    # 断言 f_3 是平方自由的
    assert R.dmp_sqf_p(f_3) is True
    # 断言 f_3**2 不是平方自由的
    assert R.dmp_sqf_p(f_3**2) is False
    # 断言 f_5 不是平方自由的
    assert R.dmp_sqf_p(f_5) is False
    # 断言 f_5**2 不是平方自由的
    assert R.dmp_sqf_p(f_5**2) is False

    # 断言 f_4 是平方自由的
    assert R.dmp_sqf_p(f_4) is True
    # 断言 f_4 的平方自由部分是 -f_4
    assert R.dmp_sqf_part(f_4) == -f_4

    # 断言 f_5 的平方自由部分是 x + y - z
    assert R.dmp_sqf_part(f_5) == x + y - z

    # 创建环 R，变量为 x, y, z, t
    R, x, y, z, t = ring("x,y,z,t", ZZ)
    # 断言 f_6 是平方自由的
    assert R.dmp_sqf_p(f_6) is True
    # 断言 f_6 的平方自由部分是 f_6
    assert R.dmp_sqf_part(f_6) == f_6

    # 创建环 R，变量为 x
    R, x = ring("x", ZZ)
    # 定义多项式 f = -x**5 + x**4 + x - 1
    f = -x**5 + x**4 + x - 1

    # 断言 f 的平方自由列表是 (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dmp_sqf_list(f) == (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    # 断言 f 的平方自由列表包含 (-x**3 - x**2 - x - 1, 1), (x - 1, 2)
    assert R.dmp_sqf_list_include(f) == [(-x**3 - x**2 - x - 1, 1), (x - 1, 2)]

    # 创建环 R，变量为 x, y
    R, x, y = ring("x,y", ZZ)
    # 重新定义多项式 f = -x**5 + x**4 + x - 1
    f = -x**5 + x**4 + x - 1

    # 断言 f 的平方自由列表是 (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dmp_sqf_list(f) == (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    # 断言 f 的平方自由列表包含 (-x**3 - x**2 - x - 1, 1), (x - 1, 2)
    assert R.dmp_sqf_list_include(f) == [(-x**3 - x**2 - x - 1, 1), (x - 1, 2)]

    # 重新定义多项式 f = -x**2 + 2*x - 1
    f = -x**2 + 2*x - 1
    # 断言 f 的平方自由列表包含 (-1, 1), (x - 1, 2)
    assert R.dmp_sqf_list_include(f) == [(-1, 1), (x - 1, 2)]

    # 重新定义多项式 f = (y**2 + 1)**2*(x**2 + 2*x + 2)
    f = (y**2 + 1)**2*(x**2 + 2*x + 2)
    # 断言 f 不是平方自由的
    assert R.dmp_sqf_p(f) is False
    # 断言 f 的平方自由列表是 (1, [(x**2 + 2*x + 2, 1), (y**2 + 1, 2)])
    assert R.dmp_sqf_list(f) == (1, [(x**2 + 2*x + 2, 1), (y**2 + 1, 2)])

    # 创建环 R，变量为 x, y，有限域 FF(2)
    R, x, y = ring("x,y", FF(2))
    # 使用 lambda 函数引发 NotImplementedError 异常，因为平方自由分解尚未实现
    raises(NotImplementedError, lambda: R.dmp_sqf_list(y**2 + 1))
# 定义一个测试函数，用于测试多项式环中的重复因子列表功能
def test_dup_gff_list():
    # 创建一个多项式环 R，其中 x 是环中的符号变量
    R, x = ring("x", ZZ)

    # 定义一个多项式 f
    f = x**5 + 2*x**4 - x**3 - 2*x**2
    # 断言调用 R.dup_gff_list(f) 返回的结果符合预期的重复因子列表 [(x, 1), (x + 2, 4)]
    assert R.dup_gff_list(f) == [(x, 1), (x + 2, 4)]

    # 定义另一个多项式 g
    g = x**9 - 20*x**8 + 166*x**7 - 744*x**6 + 1965*x**5 - 3132*x**4 + 2948*x**3 - 1504*x**2 + 320*x
    # 断言调用 R.dup_gff_list(g) 返回的结果符合预期的重复因子列表 [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]
    assert R.dup_gff_list(g) == [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]

    # 断言对于输入为 0 的情况，会引发 ValueError 异常
    raises(ValueError, lambda: R.dup_gff_list(0))

# 定义另一个测试函数，用于测试多元多项式环中的平方因子列表功能
def test_issue_26178():
    # 创建一个多元多项式环 R，其中 x, y, z 是环中的符号变量
    R, x, y, z = ring(['x', 'y', 'z'], QQ)
    
    # 断言对于多项式 x**2 - 2*y**2 + 1，其平方因子列表(sqf_list)的预期结果是 (MPQ(1,1), [(x**2 - 2*y**2 + 1, 1)])
    assert (x**2 - 2*y**2 + 1).sqf_list() == (MPQ(1,1), [(x**2 - 2*y**2 + 1, 1)])
    
    # 断言对于多项式 x**2 - 2*z**2 + 1，其平方因子列表(sqf_list)的预期结果是 (MPQ(1,1), [(x**2 - 2*z**2 + 1, 1)])
    assert (x**2 - 2*z**2 + 1).sqf_list() == (MPQ(1,1), [(x**2 - 2*z**2 + 1, 1)])
    
    # 断言对于多项式 y**2 - 2*z**2 + 1，其平方因子列表(sqf_list)的预期结果是 (MPQ(1,1), [(y**2 - 2*z**2 + 1, 1)])
    assert (y**2 - 2*z**2 + 1).sqf_list() == (MPQ(1,1), [(y**2 - 2*z**2 + 1, 1)])
```