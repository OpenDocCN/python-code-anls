# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_fp_groups.py`

```
# 导入所需的符号计算模块
from sympy.core.singleton import S
from sympy.combinatorics.fp_groups import (FpGroup, low_index_subgroups,
                                   reidemeister_presentation, FpSubgroup,
                                           simplify_presentation)
from sympy.combinatorics.free_groups import (free_group, FreeGroup)

# 导入测试相关模块
from sympy.testing.pytest import slow

"""
References
==========

[1] Holt, D., Eick, B., O'Brien, E.
"Handbook of Computational Group Theory"

[2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
"Implementation and Analysis of the Todd-Coxeter Algorithm"

[3] PROC. SECOND  INTERNAT. CONF. THEORY OF GROUPS, CANBERRA 1973,
pp. 347-356. "A Reidemeister-Schreier program" by George Havas.
http://staff.itee.uq.edu.au/havas/1973cdhw.pdf

"""

# 定义测试函数：测试低指数子群算法
def test_low_index_subgroups():
    # 创建自由群 F 以及其生成元 x, y
    F, x, y = free_group("x, y")

    # 创建 FpGroup 对象 f，使用给定的生成元和关系构造群的表示
    f = FpGroup(F, [x**2, y**3, (x*y)**4])
    # 计算 f 中低指数子群 L，直到索引为 4
    L = low_index_subgroups(f, 4)
    # 预期结果 t1 是一个包含预定义子群表的列表
    t1 = [[[0, 0, 0, 0]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 3, 3]],
          [[0, 0, 1, 2], [2, 2, 2, 0], [1, 1, 0, 1]],
          [[1, 1, 0, 0], [0, 0, 1, 1]]]
    # 验证计算结果与预期结果是否一致
    for i in range(len(t1)):
        assert L[i].table == t1[i]

    # 重新定义群 f 的关系并再次计算低指数子群
    f = FpGroup(F, [x**2, y**3, (x*y)**7])
    L = low_index_subgroups(f, 15)
    # 验证计算结果与预期结果是否一致
    for i in range(len(t2)):
        assert L[i].table == t2[i]

    # 重新定义群 f 的关系和生成元，再次计算低指数子群
    f = FpGroup(F, [x**2, y**3, (x*y)**7])
    L = low_index_subgroups(f, 10, [x])
    # 预期结果 t3 是一个包含预定义子群表的列表
    t3 = [[[0, 0, 0, 0]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5], [4, 4, 5, 3],
           [6, 6, 3, 4], [5, 5, 6, 6]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5], [6, 6, 5, 3],
           [5, 5, 3, 4], [4, 4, 6, 6]],
          [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8],
           [6, 6, 6, 3], [5, 5, 3, 5], [8, 8, 8, 4], [7, 7, 4, 7]]]
    # 验证计算结果与预期结果是否一致
    for i in range(len(t3)):
        assert L[i].table == t3[i]


# 定义测试函数：测试子群表示
def test_subgroup_presentations():
    # 创建自由群 F 以及其生成元 x, y
    F, x, y = free_group("x, y")
    
    # 创建 FpGroup 对象 f，使用给定的生成元和关系构造群的表示
    f = FpGroup(F, [x**3, y**5, (x*y)**2])
    # 定义子群 H
    H = [x*y, x**-1*y**-1*x*y*x]
    # 计算 f 关于 H 的雷德迈斯特-施雷尔表示
    p1 = reidemeister_presentation(f, H)
    # 验证计算结果与预期结果是否一致
    assert str(p1) == "((y_1, y_2), (y_1**2, y_2**3, y_2*y_1*y_2*y_1*y_2*y_1))"

    # 创建 H 的子群对象，并验证其生成元和关系是否与 p1 相同
    H = f.subgroup(H)
    assert (H.generators, H.relators) == p1

    # 重新定义群 f 的关系并再次计算子群 H 的表示
    f = FpGroup(F, [x**3, y**3, (x*y)**3])
    H = [x*y, x*y**-1]
    p2 = reidemeister_presentation(f, H)
    # 验证计算结果与预期结果是否一致
    assert str(p2) == "((x_0, y_0), (x_0**3, y_0**3, x_0*y_0*x_0*y_0*x_0*y_0))"

    # 重新定义群 f 的关系并再次计算子群 H 的表示
    f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    H = [x]
    p3 = reidemeister_presentation(f, H)
    # 验证计算结果与预期结果是否一致
    assert str(p3) == "((x_0,), (x_0**4,))"

    # 重新定义群 f 的关系并再次计算子群 H 的表示
    f = FpGroup(F, [x**3*y**-3, (x*y)**3, (x*y**-1)**2])
    H = [x]
    p4 = reidemeister_presentation(f, H)
    # 验证计算结果与预期结果是否一致
    assert str(p4) == "((x_0,), (x_0**6,))"

    # 注释：这个表示可以改进，最简化的形式为 <a, b | a^11, b^2, (a*b)^3, (a^4*b*a^-5*b)^2>
    # 参见 [2] Pg 474 group PSL_2(11)
    # 这是群 PSL_2(11) 的生成和关系的计算
    
    # 创建自由群 F，并定义生成元 a, b, c
    F, a, b, c = free_group("a, b, c")
    
    # 使用生成元和关系列表创建自由群的赋环 FpGroup 对象
    f = FpGroup(F, [a**11, b**5, c**4, (b*c**2)**2, (a*b*c)**3, (a**4*c**2)**3, b**2*c**-1*b**-1*c, a**4*b**-1*a**-1*b])
    
    # 定义子群 H，包含生成元 a, b 以及 c 的平方
    H = [a, b, c**2]
    
    # 通过 Reidemeister 算法计算生成元和关系的演示
    gens, rels = reidemeister_presentation(f, H)
    
    # 断言生成元的字符串表示是否为 "(b_1, c_3)"
    assert str(gens) == "(b_1, c_3)"
    
    # 断言关系列表的长度是否为 18
    assert len(rels) == 18
# 标记为慢速测试
@slow
def test_order():
    # 创建自由群 F，其中包含生成元 x 和 y
    F, x, y = free_group("x, y")
    # 创建 FpGroup 对象 f，指定生成元和关系
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    # 断言 f 的阶（群的大小）为 8
    assert f.order() == 8

    # 创建另一个 FpGroup 对象 f，指定新的关系
    f = FpGroup(F, [x*y*x**-1*y**-1, y**2])
    # 断言 f 的阶为无穷大
    assert f.order() is S.Infinity

    # 创建包含更多生成元的自由群 F
    F, a, b, c = free_group("a, b, c")
    # 创建具有更多关系的 FpGroup 对象 f
    f = FpGroup(F, [a**250, b**2, c*b*c**-1*b, c**4, c**-1*a**-1*c*a, a**-1*b**-1*a*b])
    # 断言 f 的阶为 2000
    assert f.order() == 2000

    # 创建只包含一个生成元 x 的自由群 F
    F, x = free_group("x")
    # 创建没有关系的 FpGroup 对象 f
    f = FpGroup(F, [])
    # 断言 f 的阶为无穷大
    assert f.order() is S.Infinity

    # 创建空自由群并创建 FpGroup 对象 f
    f = FpGroup(free_group('')[0], [])
    # 断言 f 的阶为 1
    assert f.order() == 1

# 测试 FpSubgroup 的功能
def test_fp_subgroup():
    # 定义内部函数 _test_subgroup，用于测试子群的特性
    def _test_subgroup(K, T, S):
        # 断言 K 的生成元都在 S 中
        assert all(elem in S for elem in T(K.generators))
        # 断言 T 是单射（保持群结构映射）
        assert T.is_injective()
        # 断言 T 映射后的群的阶与 S 的阶相同
        assert T.image().order() == S.order()

    # 创建自由群 F，包含生成元 x 和 y
    F, x, y = free_group("x, y")
    # 创建 FpGroup 对象 f，指定生成元和关系
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    # 创建包含生成元 x*y 的 FpSubgroup 对象 S
    S = FpSubgroup(f, [x*y])
    # 断言 (x*y)^-3 属于 S
    assert (x*y)**-3 in S
    # 调用 f 的 subgroup 方法，获取生成同态 K 和 T
    K, T = f.subgroup([x*y], homomorphism=True)
    # 断言 T(K.generators) 等于 [y*x^-1]
    assert T(K.generators) == [y*x**-1]
    # 调用内部测试函数 _test_subgroup，验证 K、T 和 S 的特性
    _test_subgroup(K, T, S)

    # 创建包含生成元 x^-1*y*x 的 FpSubgroup 对象 S
    S = FpSubgroup(f, [x**-1*y*x])
    # 断言 x^-1*y^4*x 属于 S
    assert x**-1*y**4*x in S
    # 断言 x^-1*y^4*x^2 不属于 S
    assert x**-1*y**4*x**2 not in S
    # 调用 f 的 subgroup 方法，获取生成同态 K 和 T
    K, T = f.subgroup([x**-1*y*x], homomorphism=True)
    # 断言 T(K.generators[0]^3) 等于 y^3
    assert T(K.generators[0]**3) == y**3
    # 调用内部测试函数 _test_subgroup，验证 K、T 和 S 的特性
    _test_subgroup(K, T, S)

    # 创建 FpGroup 对象 f，指定生成元和关系
    f = FpGroup(F, [x**3, y**5, (x*y)**2])
    # 创建包含生成元 x*y 和 x^-1*y^-1*x*y*x 的 FpSubgroup 对象 S
    H = [x*y, x**-1*y**-1*x*y*x]
    # 调用 f 的 subgroup 方法，获取生成同态 K 和 T
    K, T = f.subgroup(H, homomorphism=True)
    # 创建包含 H 的 FpSubgroup 对象 S
    S = FpSubgroup(f, H)
    # 调用内部测试函数 _test_subgroup，验证 K、T 和 S 的特性
    _test_subgroup(K, T, S)

# 测试置换群方法
def test_permutation_methods():
    # 创建自由群 F，包含生成元 x 和 y
    F, x, y = free_group("x, y")
    # 创建 FpGroup 对象 G，指定生成元和关系，生成 DihedralGroup(8)
    G = FpGroup(F, [x**2, y**8, x*y*x**-1*y])
    # 调用 _to_perm_group 方法，获取生成同态 T
    T = G._to_perm_group()[1]
    # 断言 T 是同构映射
    assert T.is_isomorphism()
    # 断言 G 的中心包含元素 [y^4]
    assert G.center() == [y**4]

    # 创建 FpGroup 对象 G，指定生成元和关系，生成 DihedralGroup(4)
    G = FpGroup(F, [x**2, y**4, x*y*x**-1*y])
    # 创建包含生成元 x 的正规闭包的 FpSubgroup 对象 S
    S = FpSubgroup(G, G.normal_closure([x]))
    # 断言 x 属于 S
    assert x in S
    # 断言 y^-1*x*y 属于 S
    assert y**-1*x*y in S

    # 创建 FpGroup 对象 G，指定生成元和关系，生成 Z_5xZ_4
    G = FpGroup(F, [x*y*x**-1*y**-1, y**5, x**4])
    # 断言 G 是可交换的（阿贝尔群）
    assert G.is_abelian
    # 断言 G 是可解的
    assert G.is_solvable

    # 创建 FpGroup 对象 G，指定生成元和关系，生成 AlternatingGroup(5)
    G = FpGroup(F, [x**3, y**2, (x*y)**5])
    # 断言 G 不是可解的
    assert not G.is_solvable

    # 创建 FpGroup 对象 G，指定生成元和关系，生成 AlternatingGroup(4)
    G = FpGroup(F, [x**3, y**2, (x*y)**3])
    # 断言 G 的导出级数长度为 3
    assert len(G.derived_series()) == 3
    # 创建包含 G 的导出子群的 FpSubgroup 对象 S
    S = FpSubgroup(G, G.derived_subgroup())
    # 断言 S 的阶为 4
    assert S.order() == 4

# 测试简化表示方法
def test_simplify_presentation():
    # 创建空自由群并调用 simplify_presentation 函数，获取简化后的 FpGroup 对象 G
    G = simplify_presentation(FpGroup(FreeGroup([]), []))
    # 断言 G 的生成元为空
    assert not G.generators
    # 断言 G 的关系为空

    # 创建自由群 F，包含生成元 x 和 y
    F, x, y = free_group("x, y")
    # 创建 FpGroup 对象 G，指定生成元和关系，生成 CyclicGroup(3)
    G = simplify_presentation(FpGroup(F, [x**2, x**5, y**3]))
    # 断言 x 属于 G 的关系（第二个生成元是由于关系 {x^2, x^5} 是平
    # 断言检查 f 对象的阿贝尔不变量是否为 [2]
    assert f.abelian_invariants() == [2]
    # 使用 FpGroup 类创建一个 FpGroup 对象 f，定义生成元为 x 和 y，以及对应的关系 x**4, y**2, x*y*x**-1*y
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    # 断言检查更新后的 f 对象的阿贝尔不变量是否为 [2, 4]
    assert f.abelian_invariants() == [2, 4]
```