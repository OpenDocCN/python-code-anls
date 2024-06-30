# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_homomorphisms.py`

```
# 导入必要的模块和函数
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.homomorphisms import homomorphism, group_isomorphism, is_isomorphic
from sympy.combinatorics.free_groups import free_group
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.named_groups import AlternatingGroup, DihedralGroup, CyclicGroup
from sympy.testing.pytest import raises

# 定义测试函数 test_homomorphism
def test_homomorphism():
    # 创建自由群 F 和生成元 a, b
    F, a, b = free_group("a, b")
    # 创建 FpGroup G，其中的生成元和关系定义
    G = FpGroup(F, [a**3, b**3, (a*b)**2])

    # 创建置换 Permutation c 和 d
    c = Permutation(3)(0, 1, 2)
    d = Permutation(3)(1, 2, 3)
    # 创建交错群 AlternatingGroup A
    A = AlternatingGroup(4)
    # 创建从 FpGroup 到 AlternatingGroup 的同态映射 T
    T = homomorphism(G, A, [a, b], [c, d])
    # 断言 T 的计算结果
    assert T(a*b**2*a**-1) == c*d**2*c**-1
    # 断言 T 是同构映射
    assert T.is_isomorphism()
    # 断言 T 的逆映射计算结果
    assert T(T.invert(Permutation(3)(0, 2, 3))) == Permutation(3)(0, 2, 3)

    # 重新使用 T 创建从 G 到 AlternatingGroup 的同态映射
    T = homomorphism(G, AlternatingGroup(4), G.generators)
    # 断言 T 是平凡映射
    assert T.is_trivial()
    # 断言 T 的核的阶等于 G 的阶
    assert T.kernel().order() == G.order()

    # 创建自由群 E 和生成元 e
    E, e = free_group("e")
    # 创建 FpGroup G，其中的生成元和关系定义
    G = FpGroup(E, [e**8])
    # 创建置换群 P
    P = PermutationGroup([Permutation(0, 1, 2, 3), Permutation(0, 2)])
    # 创建从 G 到 P 的同态映射 T
    T = homomorphism(G, P, [e], [Permutation(0, 1, 2, 3)])
    # 断言 T 的像的阶等于 4
    assert T.image().order() == 4
    # 断言 T 的逆映射计算结果
    assert T(T.invert(Permutation(0, 2)(1, 3))) == Permutation(0, 2)(1, 3)

    # 创建从自由群 E 到 AlternatingGroup 的同态映射 T
    T = homomorphism(E, AlternatingGroup(4), E.generators, [c])
    # 断言 T 的逆映射计算结果
    assert T.invert(c**2) == e**-1  # order(c) == 3 so c**2 == c**-1

    # 创建从自由群 F 到自由群 E 的同态映射 T
    T = homomorphism(F, E, [a], [e])
    # 断言 T 对于给定的表达式是单位元
    assert T(a**-2*b**4*a**2).is_identity

    # 创建从自由群 F 到 FpGroup G 的同态映射 T
    G = FpGroup(F, [a*b*a**-1*b**-1])
    T = homomorphism(F, G, F.generators, G.generators)
    # 断言 T 的逆映射计算结果
    assert T.invert(a**-1*b**-1*a**2) == a*b**-1

    # 创建置换群 P 和置换 p
    D = DihedralGroup(8)
    p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    P = PermutationGroup(p)
    # 创建从置换群 P 到 DihedralGroup 的同态映射 T
    T = homomorphism(P, D, [p], [p])
    # 断言 T 是单射
    assert T.is_injective()
    # 断言 T 不是同构映射
    assert not T.is_isomorphism()
    # 断言 T 的逆映射计算结果
    assert T.invert(p**3) == p**3

    # 创建从自由群 F 到置换群 P 的同态映射 T2
    T2 = homomorphism(F, P, [F.generators[0]], P.generators)
    # 将 T2 与 T 合并
    T = T.compose(T2)
    # 断言 T 的定义域和值域
    assert T.domain == F
    assert T.codomain == D
    # 断言 T 的计算结果
    assert T(a*b) == p

    # 创建 DihedralGroup D3
    D3 = DihedralGroup(3)
    # 创建从 D3 到 D3 的同构映射 T
    T = homomorphism(D3, D3, D3.generators, D3.generators)
    # 断言 T 是同构映射


# 定义测试函数 test_isomorphisms
def test_isomorphisms():
    # 创建自由群 F 和生成元 a, b
    F, a, b = free_group("a, b")
    # 创建 FpGroup G 和 H，分别用不同顺序的关系定义
    G = FpGroup(F, [a**2, b**3])
    H = FpGroup(F, [b**3, a**2])
    # 断言 G 和 H 是同构的
    assert is_isomorphic(G, H)

    # 创建 FpGroup G 和 H，其中的生成元和关系定义不同
    H = FpGroup(F, [a**3, b**3, (a*b)**2])
    # 创建自由群 F 和生成元 c, d
    F, c, d = free_group("c, d")
    # 创建 FpGroup G，其中的生成元和关系定义不同
    G = FpGroup(F, [c**3, d**3, (c*d)**2])
    # 检查并获取从 G 到 H 的群同构映射 T
    check, T =  group_isomorphism(G, H)
    # 断言群同构检查结果和映射 T
    assert check
    # 断言 T 的计算结果
    assert T(c**3*d**2) == a**3*b**2

    # 创建自由群 F 和生成元 a, b
    F, a, b = free_group("a, b")
    # 创建 FpGroup G，其中的生成元和关系定义
    G = FpGroup(F, [a**3, b**3, (a*b)**2])
    # 创建交错群 AlternatingGroup H
    H = AlternatingGroup(4)
    # 检查并获取从 G 到 H 的群同构映射 T
    check, T = group_isomorphism(G, H)
    # 断言群同构检查结果
    assert check
    # 断言：验证给定的置换是否等于指定的置换
    assert T(b*a*b**-1*a**-1*b**-1) == Permutation(0, 2, 3)
    assert T(b*a*b*a**-1*b**-1) == Permutation(0, 3, 2)

    # 创建二面体群 DihedralGroup(8) 和置换群 PermutationGroup，验证它们不同构
    D = DihedralGroup(8)
    p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    P = PermutationGroup(p)
    assert not is_isomorphic(D, P)

    # 创建循环群 A 和 B，并验证它们不同构
    A = CyclicGroup(5)
    B = CyclicGroup(7)
    assert not is_isomorphic(A, B)

    # 断言：生成的 FpGroup 和循环群的群阶相等，并验证它们同构
    G = FpGroup(F, [a, b**5])
    H = CyclicGroup(5)
    assert G.order() == H.order()
    assert is_isomorphic(G, H)
# 定义一个测试函数，用于检查同态映射
def test_check_homomorphism():
    # 创建置换对象a，表示置换(1,2,3,4)
    a = Permutation(1,2,3,4)
    # 创建置换对象b，表示置换(1,3)
    b = Permutation(1,3)
    # 创建置换群G，包含置换a和b
    G = PermutationGroup([a, b])
    # 使用homomorphism函数检查从群G到自身的同态映射，预期引发值错误
    raises(ValueError, lambda: homomorphism(G, G, [a], [a]))
```