# `D:\src\scipysrc\sympy\sympy\polys\numberfields\tests\test_galoisgroups.py`

```
"""Tests for computing Galois groups. """

# 从 sympy.abc 模块导入符号 x
from sympy.abc import x
# 从 sympy.combinatorics.galois 模块导入各阶置换群的子群
from sympy.combinatorics.galois import (
    S1TransitiveSubgroups, S2TransitiveSubgroups, S3TransitiveSubgroups,
    S4TransitiveSubgroups, S5TransitiveSubgroups, S6TransitiveSubgroups,
)
# 从 sympy.polys.domains.rationalfield 模块导入有理数域 QQ
from sympy.polys.domains.rationalfield import QQ
# 从 sympy.polys.numberfields.galoisgroups 模块导入 Galois 相关函数和类
from sympy.polys.numberfields.galoisgroups import (
    tschirnhausen_transformation,
    galois_group,
    _galois_group_degree_4_root_approx,
    _galois_group_degree_5_hybrid,
)
# 从 sympy.polys.numberfields.subfield 模块导入子域同构函数
from sympy.polys.numberfields.subfield import field_isomorphism
# 从 sympy.polys.polytools 模块导入多项式类 Poly
from sympy.polys.polytools import Poly
# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises


# 定义测试函数 test_tschirnhausen_transformation
def test_tschirnhausen_transformation():
    # 遍历多项式列表
    for T in [
        Poly(x**2 - 2),
        Poly(x**2 + x + 1),
        Poly(x**4 + 1),
        Poly(x**4 - x**3 + x**2 - x + 1),
    ]:
        # 对每个多项式 T 进行 Tschirnhausen 变换
        _, U = tschirnhausen_transformation(T)
        # 断言变换后的多项式 U 度数与 T 相同
        assert U.degree() == T.degree()
        # 断言变换后的多项式 U 是首一多项式
        assert U.is_monic
        # 断言变换后的多项式 U 是不可约多项式
        assert U.is_irreducible
        # 构建 T 和 U 的代数数域扩展
        K = QQ.alg_field_from_poly(T)
        L = QQ.alg_field_from_poly(U)
        # 断言 T 和 U 的代数数域扩展之间存在子域同构
        assert field_isomorphism(K.ext, L.ext) is not None


# Test polys are from:
# Cohen, H. *A Course in Computational Algebraic Number Theory*.
# 定义测试多项式的字典 test_polys_by_deg，按照多项式的阶数分类存储
test_polys_by_deg = {
    # Degree 1
    1: [
        (x, S1TransitiveSubgroups.S1, True)
    ],
    # Degree 2
    2: [
        (x**2 + x + 1, S2TransitiveSubgroups.S2, False)
    ],
    # Degree 3
    3: [
        (x**3 + x**2 - 2*x - 1, S3TransitiveSubgroups.A3, True),
        (x**3 + 2, S3TransitiveSubgroups.S3, False),
    ],
    # Degree 4
    4: [
        (x**4 + x**3 + x**2 + x + 1, S4TransitiveSubgroups.C4, False),
        (x**4 + 1, S4TransitiveSubgroups.V, True),
        (x**4 - 2, S4TransitiveSubgroups.D4, False),
        (x**4 + 8*x + 12, S4TransitiveSubgroups.A4, True),
        (x**4 + x + 1, S4TransitiveSubgroups.S4, False),
    ],
    # Degree 5
    5: [
        (x**5 + x**4 - 4*x**3 - 3*x**2 + 3*x + 1, S5TransitiveSubgroups.C5, True),
        (x**5 - 5*x + 12, S5TransitiveSubgroups.D5, True),
        (x**5 + 2, S5TransitiveSubgroups.M20, False),
        (x**5 + 20*x + 16, S5TransitiveSubgroups.A5, True),
        (x**5 - x + 1, S5TransitiveSubgroups.S5, False),
    ],
    # Degree 6
    6: [
        # 等待后续添加
    ],
}
    6: [
        # Tuple 1: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + x**5 + x**4 + x**3 + x**2 + x + 1, S6TransitiveSubgroups.C6, False),
        # Tuple 2: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 108, S6TransitiveSubgroups.S3, False),
        # Tuple 3: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 2, S6TransitiveSubgroups.D6, False),
        # Tuple 4: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 - 3*x**2 - 1, S6TransitiveSubgroups.A4, True),
        # Tuple 5: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 3*x**3 + 3, S6TransitiveSubgroups.G18, False),
        # Tuple 6: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 - 3*x**2 + 1, S6TransitiveSubgroups.A4xC2, False),
        # Tuple 7: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 - 4*x**2 - 1, S6TransitiveSubgroups.S4p, True),
        # Tuple 8: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 - 3*x**5 + 6*x**4 - 7*x**3 + 2*x**2 + x - 4, S6TransitiveSubgroups.S4m, False),
        # Tuple 9: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 2*x**3 - 2, S6TransitiveSubgroups.G36m, False),
        # Tuple 10: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 2*x**2 + 2, S6TransitiveSubgroups.S4xC2, False),
        # Tuple 11: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 10*x**5 + 55*x**4 + 140*x**3 + 175*x**2 + 170*x + 25, S6TransitiveSubgroups.PSL2F5, True),
        # Tuple 12: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 10*x**5 + 55*x**4 + 140*x**3 + 175*x**2 - 3019*x + 25, S6TransitiveSubgroups.PGL2F5, False),
        # Tuple 13: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 6*x**4 + 2*x**3 + 9*x**2 + 6*x - 4, S6TransitiveSubgroups.G36p, True),
        # Tuple 14: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 2*x**4 + 2*x**3 + x**2 + 2*x + 2, S6TransitiveSubgroups.G72, False),
        # Tuple 15: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + 24*x - 20, S6TransitiveSubgroups.A6, True),
        # Tuple 16: Represents a polynomial and its associated transitive subgroup and a boolean flag
        (x**6 + x + 1, S6TransitiveSubgroups.S6, False),
    ],
# 测试 Galois 群函数的单元测试函数，用于验证不同多项式的 Galois 群计算是否正确
def test_galois_group():
    """
    Try all the test polys.
    遍历所有测试多项式。
    """
    for deg in range(1, 7):
        # 获取指定度数的测试多项式列表
        polys = test_polys_by_deg[deg]
        for T, G, alt in polys:
            # 断言计算得到的 Galois 群与预期的相同
            assert galois_group(T, by_name=True) == (G, alt)


# 测试 Galois 群函数对于超出度数范围的多项式是否能正确引发 ValueError
def test_galois_group_degree_out_of_bounds():
    raises(ValueError, lambda: galois_group(Poly(0, x)))
    raises(ValueError, lambda: galois_group(Poly(1, x)))
    raises(ValueError, lambda: galois_group(Poly(x ** 7 + 1)))


# 测试 Galois 群函数在不通过名称指定情况下的表现
def test_galois_group_not_by_name():
    """
    Check at least one polynomial of each supported degree, to see that
    conversion from name to group works.
    检查每个支持的度数至少一个多项式，以确保从名称到群的转换正常工作。
    """
    for deg in range(1, 7):
        # 获取指定度数的第一个测试多项式及其预期的 Galois 群名称
        T, G_name, _ = test_polys_by_deg[deg][0]
        # 计算多项式 T 的 Galois 群，并检查是否与预期的群相同
        G, _ = galois_group(T)
        assert G == G_name.get_perm_group()


# 测试 Galois 群函数在非首一多项式上的表现
def test_galois_group_not_monic_over_ZZ():
    """
    Check that we can work with polys that are not monic over ZZ.
    检查我们能够处理非首一多项式的情况。
    """
    for deg in range(1, 7):
        # 获取指定度数的第一个测试多项式及其预期的 Galois 群和替代群
        T, G, alt = test_polys_by_deg[deg][0]
        # 计算 T/2 的 Galois 群，并检查是否与预期的群相同
        assert galois_group(T/2, by_name=True) == (G, alt)


# 测试 _galois_group_degree_4_root_approx 函数对度数为 4 的多项式的根近似计算
def test__galois_group_degree_4_root_approx():
    for T, G, alt in test_polys_by_deg[4]:
        # 断言 _galois_group_degree_4_root_approx 函数计算的结果与预期的 Galois 群和替代群相同
        assert _galois_group_degree_4_root_approx(Poly(T)) == (G, alt)


# 测试 _galois_group_degree_5_hybrid 函数对度数为 5 的多项式的混合方法计算
def test__galois_group_degree_5_hybrid():
    for T, G, alt in test_polys_by_deg[5]:
        # 断言 _galois_group_degree_5_hybrid 函数计算的结果与预期的 Galois 群和替代群相同
        assert _galois_group_degree_5_hybrid(Poly(T)) == (G, alt)


# 测试 AlgebraicField 类的 Galois 群函数
def test_AlgebraicField_galois_group():
    # 使用 x^4 + 1 构造代数域，并检查其 Galois 群是否与预期的 S4TransitiveSubgroups.V 相同
    k = QQ.alg_field_from_poly(Poly(x**4 + 1))
    G, _ = k.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.V

    # 使用 x^4 - 2 构造代数域，并检查其 Galois 群是否与预期的 S4TransitiveSubgroups.D4 相同
    k = QQ.alg_field_from_poly(Poly(x**4 - 2))
    G, _ = k.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.D4
```