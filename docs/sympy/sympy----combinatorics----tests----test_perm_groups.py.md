# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_perm_groups.py`

```
from sympy.core.containers import Tuple  # 导入 Tuple 类
from sympy.combinatorics.generators import rubik_cube_generators  # 导入魔方生成器
from sympy.combinatorics.homomorphisms import is_isomorphic  # 导入同态映射判断函数
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
    DihedralGroup, AlternatingGroup, AbelianGroup, RubikGroup  # 导入命名群组类
from sympy.combinatorics.perm_groups import (PermutationGroup,
    _orbit_transversal, Coset, SymmetricPermutationGroup)  # 导入置换群相关类
from sympy.combinatorics.permutations import Permutation  # 导入置换类
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube  # 导入多面体类（简称）
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
    _verify_normal_closure  # 导入测试工具函数
from sympy.testing.pytest import skip, XFAIL, slow  # 导入测试装饰器

rmul = Permutation.rmul  # 将 Permutation 类的 rmul 方法赋值给 rmul


def test_has():
    a = Permutation([1, 0])  # 创建置换 a
    G = PermutationGroup([a])  # 使用 a 创建置换群 G
    assert G.is_abelian  # 断言群 G 是否为可交换群

    a = Permutation([2, 0, 1])  # 创建置换 a
    b = Permutation([2, 1, 0])  # 创建置换 b
    G = PermutationGroup([a, b])  # 使用 a 和 b 创建置换群 G
    assert not G.is_abelian  # 断言群 G 是否为可交换群

    G = PermutationGroup([a])  # 使用 a 创建置换群 G
    assert G.has(a)  # 断言群 G 是否包含置换 a
    assert not G.has(b)  # 断言群 G 是否不包含置换 b

    a = Permutation([2, 0, 1, 3, 4, 5])  # 创建置换 a
    b = Permutation([0, 2, 1, 3, 4])  # 创建置换 b
    assert PermutationGroup(a, b).degree == \
        PermutationGroup(a, b).degree == 6  # 断言由 a 和 b 构成的群的度为 6

    g = PermutationGroup(Permutation(0, 2, 1))  # 创建置换群 g
    assert Tuple(1, g).has(g)  # 断言元组 (1, g) 是否包含 g


def test_generate():
    a = Permutation([1, 0])  # 创建置换 a
    g = list(PermutationGroup([a]).generate())  # 生成由 a 生成的群 G 的所有元素列表
    assert g == [Permutation([0, 1]), Permutation([1, 0])]  # 断言生成结果是否符合预期
    assert len(list(PermutationGroup(Permutation((0, 1))).generate())) == 1  # 断言生成的元素数量是否符合预期
    g = PermutationGroup([a]).generate(method='dimino')  # 使用 'dimino' 方法生成群 G 的所有元素
    assert list(g) == [Permutation([0, 1]), Permutation([1, 0])]  # 断言生成结果是否符合预期

    a = Permutation([2, 0, 1])  # 创建置换 a
    b = Permutation([2, 1, 0])  # 创建置换 b
    G = PermutationGroup([a, b])  # 使用 a 和 b 创建置换群 G
    g = G.generate()  # 生成群 G 的所有元素
    v1 = [p.array_form for p in list(g)]  # 将生成的置换转换为数组形式列表
    v1.sort()  # 对列表进行排序
    assert v1 == [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0,
        1], [2, 1, 0]]  # 断言生成结果是否符合预期
    v2 = list(G.generate(method='dimino', af=True))  # 使用 'dimino' 方法生成群 G 的所有元素并转换为数组形式
    assert v1 == sorted(v2)  # 断言生成结果是否与排序后的 v2 相同

    a = Permutation([2, 0, 1, 3, 4, 5])  # 创建置换 a
    b = Permutation([2, 1, 3, 4, 5, 0])  # 创建置换 b
    g = PermutationGroup([a, b]).generate(af=True)  # 使用数组形式生成群 G 的所有元素
    assert len(list(g)) == 360  # 断言生成的元素数量是否符合预期


def test_order():
    a = Permutation([2, 0, 1, 3, 4, 5, 6, 7, 8, 9])  # 创建置换 a
    b = Permutation([2, 1, 3, 4, 5, 6, 7, 8, 9, 0])  # 创建置换 b
    g = PermutationGroup([a, b])  # 使用 a 和 b 创建置换群 g
    assert g.order() == 1814400  # 断言群 g 的阶数是否符合预期
    assert PermutationGroup().order() == 1  # 断言空置换群的阶数是否为 1


def test_equality():
    p_1 = Permutation(0, 1, 3)  # 创建置换 p_1
    p_2 = Permutation(0, 2, 3)  # 创建置换 p_2
    p_3 = Permutation(0, 1, 2)  # 创建置换 p_3
    p_4 = Permutation(0, 1, 3)  # 创建置换 p_4
    g_1 = PermutationGroup(p_1, p_2)  # 使用 p_1 和 p_2 创建置换群 g_1
    g_2 = PermutationGroup(p_3, p_4)  # 使用 p_3 和 p_4 创建置换群 g_2
    g_3 = PermutationGroup(p_2, p_1)  # 使用 p_2 和 p_1 创建置换群 g_3
    g_4 = PermutationGroup(p_1, p_2)  # 使用 p_1 和 p_2 创建置换群 g_4

    assert g_1 != g_2  # 断言 g_1 和 g_2 是否不相等
    assert g_1.generators != g_2.generators  # 断言 g_1 和 g_2 的生成器是否不相等
    assert g_1.equals(g_2)  # 断言 g_1 是否等于 g_2
    assert g_1 != g_3  # 断言 g_1 和 g_3 是否不相等
    assert g_1.equals(g_3)  # 断言 g_1 是否等于 g_3
    assert g_1 == g_4  # 断言 g_1 是否等于 g_4


def test_stabilizer():
    S = SymmetricGroup(2)  # 创建对称群 S
    H = S.stabilizer(0)  # 计算 S 中保定点 0 的稳定子群 H
    assert H.generators == [Permutation(1)]  # 断言 H 的生成器是否符合预期
    a = Permutation([2, 0, 1, 3, 4, 5])  # 创建置换 a
    # 创建置换 b，其作用是将第一个元素与第二个元素互换
    b = Permutation([2, 1, 3, 4, 5, 0])

    # 创建包含置换 a 和 b 的置换群 G
    G = PermutationGroup([a, b])

    # 计算 G 中保持元素 0 固定的子群 G0
    G0 = G.stabilizer(0)

    # 断言 G0 的阶（元素个数）为 60
    assert G0.order() == 60

    # 定义两个生成元组成的列表 gens_cube
    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]

    # 将 gens_cube 中的每个列表转换为置换对象，并存储在 gens 列表中
    gens = [Permutation(p) for p in gens_cube]

    # 创建由 gens 列表中所有置换生成的置换群 G
    G = PermutationGroup(gens)

    # 计算 G 中保持元素 2 固定的子群 G2
    G2 = G.stabilizer(2)

    # 断言 G2 的阶为 6
    assert G2.order() == 6

    # 计算 G2 中保持元素 1 固定的子群 G2_1
    G2_1 = G2.stabilizer(1)

    # 生成 G2_1 中的所有元素，并转换为列表 v
    v = list(G2_1.generate(af=True))

    # 断言 v 应为包含两个列表的列表
    assert v == [[0, 1, 2, 3, 4, 5, 6, 7], [3, 1, 2, 0, 7, 5, 6, 4]]

    # 定义三个元组作为生成元 gens
    gens = (
        (1, 2, 0, 4, 5, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        (0, 1, 2, 3, 4, 5, 19, 6, 8, 9, 10, 11, 12, 13, 14,
         15, 16, 7, 17, 18),
        (0, 1, 2, 3, 4, 5, 6, 7, 9, 18, 16, 11, 12, 13, 14, 15, 8, 17, 10, 19))

    # 将 gens 中的每个元组转换为置换对象，并存储在 gens 列表中
    gens = [Permutation(p) for p in gens]

    # 创建由 gens 列表中所有置换生成的置换群 G
    G = PermutationGroup(gens)

    # 计算 G 中保持元素 2 固定的子群 G2
    G2 = G.stabilizer(2)

    # 断言 G2 的阶为 181440
    assert G2.order() == 181440

    # 创建对称群 S3
    S = SymmetricGroup(3)

    # 断言 S3 中基本稳定子群的阶分别为 6 和 2
    assert [G.order() for G in S.basic_stabilizers] == [6, 2]
def test_center():
    # the center of the dihedral group D_n is of order 2 for even n
    for i in (4, 6, 10):
        # Create a DihedralGroup object with order i
        D = DihedralGroup(i)
        # Assert that the order of the center of D is 2
        assert (D.center()).order() == 2

    # the center of the dihedral group D_n is of order 1 for odd n>2
    for i in (3, 5, 7):
        # Create a DihedralGroup object with order i
        D = DihedralGroup(i)
        # Assert that the order of the center of D is 1
        assert (D.center()).order() == 1

    # the center of an abelian group is the group itself
    for i in (2, 3, 5):
        for j in (1, 5, 7):
            for k in (1, 1, 11):
                # Create an AbelianGroup object with dimensions i, j, k
                G = AbelianGroup(i, j, k)
                # Assert that the center of G is a subgroup of G itself
                assert G.center().is_subgroup(G)

    # the center of a nonabelian simple group is trivial
    for i in(1, 5, 9):
        # Create an AlternatingGroup object with order i
        A = AlternatingGroup(i)
        # Assert that the order of the center of A is 1
        assert (A.center()).order() == 1

    # brute-force verifications
    D = DihedralGroup(5)
    A = AlternatingGroup(3)
    C = CyclicGroup(4)
    # Check if G is a subgroup of D * A * C
    G.is_subgroup(D*A*C)
    # Assert the result of verifying centralizer for G with itself
    assert _verify_centralizer(G, G)


def test_centralizer():
    # the centralizer of the trivial group is the entire group
    S = SymmetricGroup(2)
    # Assert that the centralizer of a permutation in S is a subgroup of S
    assert S.centralizer(Permutation(list(range(2)))).is_subgroup(S)

    A = AlternatingGroup(5)
    # Assert that the centralizer of a permutation in A is a subgroup of A
    assert A.centralizer(Permutation(list(range(5)))).is_subgroup(A)

    # a centralizer in the trivial group is the trivial group itself
    triv = PermutationGroup([Permutation([0, 1, 2, 3])])
    D = DihedralGroup(4)
    # Assert that the centralizer of triv in D is triv itself
    assert triv.centralizer(D).is_subgroup(triv)

    # brute-force verifications for centralizers of groups
    for i in (4, 5, 6):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        C = CyclicGroup(i)
        D = DihedralGroup(i)
        for gp in (S, A, C, D):
            for gp2 in (S, A, C, D):
                if not gp2.is_subgroup(gp):
                    # Assert the result of verifying centralizer for gp and gp2
                    assert _verify_centralizer(gp, gp2)

    # verify the centralizer for all elements of several groups
    S = SymmetricGroup(5)
    elements = list(S.generate_dimino())
    for element in elements:
        # Assert the result of verifying centralizer for S and each element
        assert _verify_centralizer(S, element)

    A = AlternatingGroup(5)
    elements = list(A.generate_dimino())
    for element in elements:
        # Assert the result of verifying centralizer for A and each element
        assert _verify_centralizer(A, element)

    D = DihedralGroup(7)
    elements = list(D.generate_dimino())
    for element in elements:
        # Assert the result of verifying centralizer for D and each element
        assert _verify_centralizer(D, element)

    # verify centralizers of small groups within small groups
    small = []
    for i in (1, 2, 3):
        small.append(SymmetricGroup(i))
        small.append(AlternatingGroup(i))
        small.append(DihedralGroup(i))
        small.append(CyclicGroup(i))
    for gp in small:
        for gp2 in small:
            if gp.degree == gp2.degree:
                # Assert the result of verifying centralizer for gp and gp2
                assert _verify_centralizer(gp, gp2)


def test_coset_rank():
    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    gens = [Permutation(p) for p in gens_cube]
    G = PermutationGroup(gens)
    i = 0
    for h in G.generate(af=True):
        # Calculate the coset rank of h in G
        rk = G.coset_rank(h)
        # Assert that the coset rank is equal to i
        assert rk == i
        # Calculate the coset represented by rk in G
        h1 = G.coset_unrank(rk, af=True)
        # Assert that h and h1 represent the same coset in G
        assert h == h1
        i += 1
    # 断言：验证 coset_unrank(48) 返回 None
    assert G.coset_unrank(48) is None
    
    # 断言：验证 coset_unrank(coset_rank(gens[0])) 返回 gens[0]
    assert G.coset_unrank(G.coset_rank(gens[0])) == gens[0]
# 定义一个测试函数，用于测试余类分解相关功能
def test_coset_factor():
    # 创建置换对象 a = (0 2 1)
    a = Permutation([0, 2, 1])
    # 创建包含置换 a 的置换群 G
    G = PermutationGroup([a])
    # 创建置换对象 c = (2 1 0)
    c = Permutation([2, 1, 0])
    # 断言在 G 中没有 c 的余类分解
    assert not G.coset_factor(c)
    # 断言 G 中 c 的余类秩为 None
    assert G.coset_rank(c) is None

    # 创建置换对象 a = (2 0 1 3 4 5)
    a = Permutation([2, 0, 1, 3, 4, 5])
    # 创建置换对象 b = (2 1 3 4 5 0)
    b = Permutation([2, 1, 3, 4, 5, 0])
    # 创建包含置换 a 和 b 的置换群 g
    g = PermutationGroup([a, b])
    # 断言 g 的阶数为 360
    assert g.order() == 360
    # 创建置换对象 d = (1 0 2 3 4 5)
    d = Permutation([1, 0, 2, 3, 4, 5])
    # 断言在 g 中没有 d 的余类分解（使用数组形式）
    assert not g.coset_factor(d.array_form)
    # 断言 g 不包含置换 d
    assert not g.contains(d)
    # 断言置换对象 Permutation(2) 在 G 中
    assert Permutation(2) in G
    # 创建置换对象 c = (1 0 2 3 5 4)
    c = Permutation([1, 0, 2, 3, 5, 4])
    # 获取 c 在 g 中的余类分解，strict=True
    v = g.coset_factor(c, True)
    # 获取 g 的基本横跨集
    tr = g.basic_transversals
    # 使用基本横跨集计算置换 p
    p = Permutation.rmul(*[tr[i][v[i]] for i in range(len(g.base))])
    # 断言 p 等于 c
    assert p == c
    # 获取 c 在 g 中的余类分解，strict=False
    v = g.coset_factor(c)
    # 计算置换 p
    p = Permutation.rmul(*v)
    # 断言 p 等于 c
    assert p == c
    # 断言 g 包含置换 c
    assert g.contains(c)

    # 创建只包含置换 (2 1 0) 的置换群 G
    G = PermutationGroup([Permutation([2, 1, 0])])
    # 创建置换对象 p = (1 0 2)
    p = Permutation([1, 0, 2])
    # 断言 G 中 p 的余类分解为空列表
    assert G.coset_factor(p) == []


# 定义一个测试函数，用于测试轨道相关功能
def test_orbits():
    # 创建置换对象 a = (2 0 1)
    a = Permutation([2, 0, 1])
    # 创建置换对象 b = (2 1 0)
    b = Permutation([2, 1, 0])
    # 创建包含置换 a 和 b 的置换群 g
    g = PermutationGroup([a, b])
    # 断言 g 中 0 的轨道为 {0, 1, 2}
    assert g.orbit(0) == {0, 1, 2}
    # 断言 g 的所有轨道列表为 [{0, 1, 2}]
    assert g.orbits() == [{0, 1, 2}]
    # 断言 g 是传递的，并且在非严格模式下也是传递的
    assert g.is_transitive() and g.is_transitive(strict=False)
    # 断言 g 中 0 的轨道横跨列表
    assert g.orbit_transversal(0) == [
        Permutation([0, 1, 2]), Permutation([2, 0, 1]), Permutation([1, 2, 0])]
    # 断言 g 中 0 的轨道横跨列表，包括索引
    assert g.orbit_transversal(0, True) == [
        (0, Permutation([0, 1, 2])), (2, Permutation([2, 0, 1])),
        (1, Permutation([1, 2, 0]))]

    # 创建 Dihedral 群对象 G
    G = DihedralGroup(6)
    # 使用 _orbit_transversal 函数计算 G 的度数、生成器、起始点 0 的轨道横跨列表和生成的置换列表
    transversal, slps = _orbit_transversal(G.degree, G.generators, 0, True, slp=True)
    # 遍历轨道横跨列表，验证生成的置换 slp
    for i, t in transversal:
        slp = slps[i]
        w = G.identity
        for s in slp:
            w = G.generators[s]*w
        assert w == t

    # 创建置换对象 a = (1 2 ... 99 0)
    a = Permutation(list(range(1, 100)) + [0])
    # 创建包含置换 a 的置换群 G
    G = PermutationGroup([a])
    # 断言 G 的所有轨道的最小元素列表为 [0]
    assert [min(o) for o in G.orbits()] == [0]
    
    # 使用魔方生成器函数生成魔方置换群 G
    G = PermutationGroup(rubik_cube_generators())
    # 断言 G 的所有轨道的最小元素列表为 [0, 1]
    assert [min(o) for o in G.orbits()] == [0, 1]
    # 断言 G 不是传递的，并且在非严格模式下也不是传递的
    assert not G.is_transitive() and not G.is_transitive(strict=False)

    # 创建只包含置换 (0 1 3) 和 (3)(0 1) 的置换群 G
    G = PermutationGroup([Permutation(0, 1, 3), Permutation(3)(0, 1)])
    # 断言 G 不是传递的，并且在非严格模式下是传递的
    assert not G.is_transitive() and G.is_transitive(strict=False)
    # 断言 PermutationGroup(Permutation(3)) 在非严格模式下不是传递的
    assert PermutationGroup(Permutation(3)).is_transitive(strict=False) is False


# 定义一个测试函数，用于测试正规子群相关功能
def test_is_normal():
    # 创建 S5 置换群的生成器列表 gens_s5
    gens_s5 = [Permutation(p) for p in [[1, 2, 3, 4, 0], [2, 1, 4, 0, 3]]]
    # 创建包含 gens_s5 生成器的置换群 G1
    G1 = PermutationGroup(gens_s5)
    # 断言 G1 的阶数为 120
    assert G1.order() == 120

    # 创建 A5 置换群的生成器列表 gens_a5
    gens_a5 = [Permutation(p) for p in [[1, 0, 3, 2, 4], [2, 1, 4, 3, 0]]]
    # 创建包含 gens_a5 生成器的置换群 G2
    G2 = PermutationGroup(gens_a5)
    # 断言 G2 的阶数为 60
    assert G2.order() == 60
    # 断言 G2 是 G1 的正规子群
    assert G2.is_normal(G1)

    # 创建生成器列表 gens3
    gens3 = [Permutation(p) for p in [[2, 1, 3, 0, 4], [1, 2, 0, 3, 4]]]
    # 创建包含 gens3 生成器的置换群 G3
    G3 = PermutationGroup(gens3)
    #
    # 确保 G1 不是 G4 的子群
    assert not G1.is_subgroup(G4)
    
    # 确保 G2 是 G4 的子群
    assert G2.is_subgroup(G4)
    
    # 创建包含一个长度为 4 的置换的置换群 I5
    I5 = PermutationGroup(Permutation(4))
    
    # 确保 I5 是 G5 的正规子群
    assert I5.is_normal(G5)
    
    # 确保 I5 是 G6 的正规子群，允许松散判定
    assert I5.is_normal(G6, strict=False)
    
    # 创建多个置换
    p1 = Permutation([1, 0, 2, 3, 4])
    p2 = Permutation([0, 1, 2, 4, 3])
    p3 = Permutation([3, 4, 2, 1, 0])
    id_ = Permutation([0, 1, 2, 3, 4])
    
    # 创建包含 p1 和 p3 的置换群 H
    H = PermutationGroup([p1, p3])
    
    # 创建包含 p1 和 p2 的置换群 H_n1
    H_n1 = PermutationGroup([p1, p2])
    
    # 创建只包含 p1 的置换群 H_n2_1
    H_n2_1 = PermutationGroup(p1)
    
    # 创建只包含 p2 的置换群 H_n2_2
    H_n2_2 = PermutationGroup(p2)
    
    # 创建只包含 id_ 的置换群 H_id
    H_id = PermutationGroup(id_)
    
    # 确保 H_n1 是 H 的正规子群
    assert H_n1.is_normal(H)
    
    # 确保 H_n2_1 是 H_n1 的正规子群
    assert H_n2_1.is_normal(H_n1)
    
    # 确保 H_n2_2 是 H_n1 的正规子群
    assert H_n2_2.is_normal(H_n1)
    
    # 确保 H_id 是 H_n2_1 的正规子群
    assert H_id.is_normal(H_n2_1)
    
    # 确保 H_id 是 H_n1 的正规子群
    assert H_id.is_normal(H_n1)
    
    # 确保 H_id 是 H 的正规子群
    assert H_id.is_normal(H)
    
    # 确保 H_n2_1 不是 H 的正规子群
    assert not H_n2_1.is_normal(H)
    
    # 确保 H_n2_2 不是 H 的正规子群
    assert not H_n2_2.is_normal(H)
# 定义测试函数 test_eq
def test_eq():
    # 初始化排列列表 a
    a = [[1, 2, 0, 3, 4, 5], [1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5], [
        1, 2, 0, 3, 4, 5]]
    # 将每个列表转换为 Permutation 对象并存储在 a 中
    a = [Permutation(p) for p in a + [[1, 2, 3, 4, 5, 0]]]
    # 创建一个 Permutation 对象 g
    g = Permutation([1, 2, 3, 4, 5, 0])
    # 分别用前两个、后两个以及 g 和 g 的平方创建三个 PermutationGroup 对象 G1, G2, G3
    G1, G2, G3 = [PermutationGroup(x) for x in [a[:2], a[2:4], [g, g**2]]]
    # 断言 G1, G2, G3 的阶数均为 6
    assert G1.order() == G2.order() == G3.order() == 6
    # 断言 G1 是 G2 的子群
    assert G1.is_subgroup(G2)
    # 断言 G1 不是 G3 的子群
    assert not G1.is_subgroup(G3)
    # 创建包含 Permutation([0, 1]) 的 PermutationGroup 对象 G4
    G4 = PermutationGroup([Permutation([0, 1])])
    # 断言 G1 不是 G4 的子群
    assert not G1.is_subgroup(G4)
    # 断言 G4 是 G1 的子群，使用默认生成器下标 0
    assert G4.is_subgroup(G1, 0)
    # 断言 PermutationGroup(g, g) 是 PermutationGroup(g) 的子群
    assert PermutationGroup(g, g).is_subgroup(PermutationGroup(g))
    # 断言 SymmetricGroup(3) 是 SymmetricGroup(4) 的子群，使用默认生成器下标 0
    assert SymmetricGroup(3).is_subgroup(SymmetricGroup(4), 0)
    # 断言 SymmetricGroup(3) 是 SymmetricGroup(3) * CyclicGroup(5) 的子群，使用默认生成器下标 0
    assert SymmetricGroup(3).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)
    # 断言 CyclicGroup(5) 不是 SymmetricGroup(3) * CyclicGroup(5) 的子群，使用默认生成器下标 0
    assert not CyclicGroup(5).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)
    # 断言 CyclicGroup(3) 是 SymmetricGroup(3) * CyclicGroup(5) 的子群，使用默认生成器下标 0
    assert CyclicGroup(3).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)


# 定义测试函数 test_derived_subgroup
def test_derived_subgroup():
    # 创建排列 a 和 b
    a = Permutation([1, 0, 2, 4, 3])
    b = Permutation([0, 1, 3, 2, 4])
    # 用 a 和 b 创建 PermutationGroup 对象 G
    G = PermutationGroup([a, b])
    # 求 G 的导出子群 C
    C = G.derived_subgroup()
    # 断言 C 的阶数为 3
    assert C.order() == 3
    # 断言 C 是 G 的正规子群
    assert C.is_normal(G)
    # 断言 C 是 G 的子群，使用默认生成器下标 0
    assert C.is_subgroup(G, 0)
    # 断言 G 不是 C 的子群，使用默认生成器下标 0
    assert not G.is_subgroup(C, 0)
    # 创建 Rubik 魔方的生成器列表 gens_cube
    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    # 将 gens_cube 中每个列表转换为 Permutation 对象并存储在 gens 中
    gens = [Permutation(p) for p in gens_cube]
    # 用 gens 创建 PermutationGroup 对象 G
    G = PermutationGroup(gens)
    # 求 G 的导出子群 C
    C = G.derived_subgroup()
    # 断言 C 的阶数为 12
    assert C.order() == 12


# 定义测试函数 test_is_solvable
def test_is_solvable():
    # 创建排列 a 和 b
    a = Permutation([1, 2, 0])
    b = Permutation([1, 0, 2])
    # 用 a 和 b 创建 PermutationGroup 对象 G
    G = PermutationGroup([a, b])
    # 断言 G 是可解的
    assert G.is_solvable
    # 用 a 创建 PermutationGroup 对象 G
    G = PermutationGroup([a])
    # 断言 G 是可解的
    assert G.is_solvable
    # 创建排列 a 和 b
    a = Permutation([1, 2, 3, 4, 0])
    b = Permutation([1, 0, 2, 3, 4])
    # 用 a 和 b 创建 PermutationGroup 对象 G
    G = PermutationGroup([a, b])
    # 断言 G 不是可解的
    assert not G.is_solvable
    # 创建 SymmetricGroup(10) 的 Sylow 子群 S
    P = SymmetricGroup(10)
    S = P.sylow_subgroup(3)
    # 断言 S 是可解的
    assert S.is_solvable


# 定义测试函数 test_rubik1
def test_rubik1():
    # 获取 Rubik 魔方的生成器列表 gens
    gens = rubik_cube_generators()
    # 创建 G1，用最后一个生成器和其余生成器的平方创建 PermutationGroup 对象
    gens1 = [gens[-1]] + [p**2 for p in gens[1:]]
    G1 = PermutationGroup(gens1)
    # 断言 G1 的阶数为 19508428800
    assert G1.order() == 19508428800
    # 用所有生成器的平方创建 PermutationGroup 对象 G2
    gens2 = [p**2 for p in gens]
    G2 = PermutationGroup(gens2)
    # 断言 G2 的阶数为 663552
    assert G2.order() == 663552
    # 断言 G2 是 G1 的子群，使用默认生成器下标 0
    assert G2.is_subgroup(G1, 0)
    # 求 G1 的导出子群 C1
    C1 = G1.derived_subgroup()
    # 断言 C1 的阶数为 4877107200
    assert C1.order() == 4877107200
    # 断言 C1 是 G1 的子群，使用默认生成器下标 0
    assert C1.is_subgroup(G1, 0)
    # 断言 G2 不是 C1 的子群，使用默认生成器下标 0
    assert not G2.is_subgroup(C1, 0)
    # 创建 2 阶 Rubik 魔方的 RubikGroup 对象 G
    G = RubikGroup(2)
    # 断言 G 的阶数为 3674160


# 定义测试函数 test_direct_product
def test_direct_product():
    # 创建循环群 C 和二面体群 D
    C = CyclicGroup(4)
    D = DihedralGroup(4)
    # 创建 C*C*C 的直积群 G
    G = C*C*C
    # 断言 G 的阶数为 64
    assert G.order() == 64
    # 断言 G 的度为 12
    assert G.degree == 12
    # 断言 G 的轨道数为 3
    assert len(G.orbits()) == 3
    # 断言 G 是 Abel 群
    assert G.is_abelian is True
    # 创建 D*C 的直积群 H
    H
    # 断言检查 G.orbit_rep(1, 3) 返回的结果是否在给定的排列列表中
    assert G.orbit_rep(1, 3) in [Permutation([2, 3, 4, 5, 0, 1]),
                                Permutation([4, 3, 2, 1, 0, 5])]
    # 创建一个新的循环群 H，该群是循环群 CyclicGroup(4) 和 G 的直积
    H = CyclicGroup(4) * G
    # 断言检查 H.orbit_rep(1, 5) 的返回值是否为 False
    assert H.orbit_rep(1, 5) is False
# 定义一个测试函数，用于测试 Schreier 向量的计算
def test_schreier_vector():
    # 创建一个循环群对象 G，包含 50 个元素
    G = CyclicGroup(50)
    # 创建一个长度为 50 的列表 v，所有元素初始化为 0
    v = [0]*50
    # 将列表 v 的第 23 个元素设为 -1
    v[23] = -1
    # 断言调用 G 的 schreier_vector 方法返回的结果与预期的向量 v 相同
    assert G.schreier_vector(23) == v

    # 创建一个 8 面体群对象 H
    H = DihedralGroup(8)
    # 断言调用 H 的 schreier_vector 方法返回的结果与预期的向量 [0, 1, -1, 0, 0, 1, 0, 0] 相同
    assert H.schreier_vector(2) == [0, 1, -1, 0, 0, 1, 0, 0]

    # 创建一个对称群对象 L，包含 4 个元素
    L = SymmetricGroup(4)
    # 断言调用 L 的 schreier_vector 方法返回的结果与预期的向量 [1, -1, 0, 0] 相同
    assert L.schreier_vector(1) == [1, -1, 0, 0]


# 定义一个测试函数，用于测试随机生成置换的功能
def test_random_pr():
    # 创建一个 6 边形对称群对象 D
    D = DihedralGroup(6)
    r = 11
    n = 3
    # 初始化一个包含随机精度信息的字典 _random_prec_n
    _random_prec_n = {}
    _random_prec_n[0] = {'s': 7, 't': 3, 'x': 2, 'e': -1}
    _random_prec_n[1] = {'s': 5, 't': 5, 'x': 1, 'e': -1}
    _random_prec_n[2] = {'s': 3, 't': 4, 'x': 2, 'e': 1}
    # 调用 D 的 _random_pr_init 方法，初始化随机生成器
    D._random_pr_init(r, n, _random_prec_n=_random_prec_n)
    # 断言调用 D 的 _random_gens 字典中键为 11 的值与预期的列表 [0, 1, 2, 3, 4, 5] 相同
    assert D._random_gens[11] == [0, 1, 2, 3, 4, 5]
    # 初始化一个随机精度信息字典 _random_prec
    _random_prec = {'s': 2, 't': 9, 'x': 1, 'e': -1}
    # 断言调用 D 的 random_pr 方法返回的结果是一个特定的置换
    assert D.random_pr(_random_prec=_random_prec) == Permutation([0, 5, 4, 3, 2, 1])


# 定义一个测试函数，用于测试交替对称性质的评估
def test_is_alt_sym():
    # 创建一个 10 面体二面体群对象 G
    G = DihedralGroup(10)
    # 断言调用 G 的 is_alt_sym 方法返回 False
    assert G.is_alt_sym() is False
    # 断言调用 G 的 _eval_is_alt_sym_naive 方法返回 False
    assert G._eval_is_alt_sym_naive() is False
    # 断言调用 G 的 _eval_is_alt_sym_naive 方法并传入 only_alt=True 返回 False
    assert G._eval_is_alt_sym_naive(only_alt=True) is False
    # 断言调用 G 的 _eval_is_alt_sym_naive 方法并传入 only_sym=True 返回 False

    # 创建一个对称群对象 S，包含 10 个元素
    S = SymmetricGroup(10)
    # 断言调用 S 的 _eval_is_alt_sym_naive 方法返回 True
    assert S._eval_is_alt_sym_naive() is True
    # 断言调用 S 的 _eval_is_alt_sym_naive 方法并传入 only_alt=True 返回 False
    assert S._eval_is_alt_sym_naive(only_alt=True) is False
    # 断言调用 S 的 _eval_is_alt_sym_naive 方法并传入 only_sym=True 返回 True

    # 初始化一个包含多个随机置换的字典 _random_prec
    N_eps = 10
    _random_prec = {
        'N_eps': N_eps,
        0: Permutation([[2], [1, 4], [0, 6, 7, 8, 9, 3, 5]]),
        1: Permutation([[1, 8, 7, 6, 3, 5, 2, 9], [0, 4]]),
        2: Permutation([[5, 8], [4, 7], [0, 1, 2, 3, 6, 9]]),
        3: Permutation([[3], [0, 8, 2, 7, 4, 1, 6, 9, 5]]),
        4: Permutation([[8], [4, 7, 9], [3, 6], [0, 5, 1, 2]]),
        5: Permutation([[6], [0, 2, 4, 5, 1, 8, 3, 9, 7]]),
        6: Permutation([[6, 9, 8], [4, 5], [1, 3, 7], [0, 2]]),
        7: Permutation([[4], [0, 2, 9, 1, 3, 8, 6, 5, 7]]),
        8: Permutation([[1, 5, 6, 3], [0, 2, 7, 8, 4, 9]]),
        9: Permutation([[8], [6, 7], [2, 3, 4, 5], [0, 1, 9]])
    }
    # 断言调用 S 的 is_alt_sym 方法并传入 _random_prec 返回 True
    assert S.is_alt_sym(_random_prec=_random_prec) is True

    # 创建一个交替群对象 A，包含 10 个元素
    A = AlternatingGroup(10)
    # 断言调用 A 的 _eval_is_alt_sym_naive 方法返回 True
    assert A._eval_is_alt_sym_naive() is True
    # 断言调用 A 的 _eval_is_alt_sym_naive 方法并传入 only_alt=True 返回 True
    assert A._eval_is_alt_sym_naive(only_alt=True) is True
    # 断言调用 A 的 _eval_is_alt_sym_naive 方法并传入 only_sym=True 返回 False

    # 初始化一个包含多个随机置换的字典 _random_prec
    _random_prec = {
        'N_eps': N_eps,
        0: Permutation([[1, 6, 4, 2, 7, 8, 5, 9, 3], [0]]),
        1: Permutation([[1], [0, 5, 8, 4, 9, 2, 3, 6, 7]]),
        2: Permutation([[1, 9, 8, 3, 2, 5], [0, 6, 7, 4]]),
        3: Permutation([[6, 8, 9], [4, 5], [1, 3, 7, 2], [0]]),
        4: Permutation([[8], [5], [4], [2, 6, 9, 3], [1], [0, 7]]),
        5: Permutation([[3, 6], [0, 8, 1, 7, 5, 9, 4, 2]]),
        6: Permutation([[5], [2, 9], [1, 8, 3], [0, 4, 7, 6]]),
        7: Permutation([[1, 8, 4, 7, 2, 3], [0, 6, 9, 5]]),
        8: Permutation([[5, 8, 7], [3], [1, 4, 2, 6], [0,
    # 创建置换群 G，包含两个置换：(1, 3)(0, 2, 4, 6) 和 (5, 7)(0, 2, 4, 6)，总大小为 8
    G = PermutationGroup(
        Permutation(1, 3, size=8)(0, 2, 4, 6),
        Permutation(5, 7, size=8)(0, 2, 4, 6))
    # 断言 G 不是交替对称群
    assert G.is_alt_sym() is False
    
    # 对于 Monte-Carlo c_n 参数设置的测试，保证结果为 False
    G = DihedralGroup(10)
    assert G._eval_is_alt_sym_monte_carlo() is False
    G = DihedralGroup(20)
    assert G._eval_is_alt_sym_monte_carlo() is False
    
    # 一次干运行测试，检查是否查找更新后的缓存
    G = DihedralGroup(6)
    # 调用 G 的 is_alt_sym 方法，并断言其结果为 False
    G.is_alt_sym()
    assert G.is_alt_sym() is False
def test_minimal_block():
    # 创建一个6阶二面体群对象
    D = DihedralGroup(6)
    # 对给定的区块 [0, 3] 计算最小区块系统
    block_system = D.minimal_block([0, 3])
    # 检查前三个元素与后三个元素是否相等
    for i in range(3):
        assert block_system[i] == block_system[i + 3]
    # 创建一个6阶对称群对象
    S = SymmetricGroup(6)
    # 检查对称群中 [0, 1] 区块的最小区块系统是否为 [0, 0, 0, 0, 0, 0]
    assert S.minimal_block([0, 1]) == [0, 0, 0, 0, 0, 0]

    # 棱柱群的类属性 pgroup 中，对 [0, 1] 区块的最小区块系统是否为 [0, 0, 0, 0]
    assert Tetra.pgroup.minimal_block([0, 1]) == [0, 0, 0, 0]

    # 创建一个置换群对象 P1，包含两个置换生成元
    P1 = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    # 创建一个置换群对象 P2，包含两个置换生成元（顺序不同）
    P2 = PermutationGroup(Permutation(0, 1, 2, 3, 4, 5), Permutation(1, 5)(2, 4))
    # 检查 P1 中 [0, 2] 区块的最小区块系统是否为 [0, 1, 0, 1, 0, 1]
    assert P1.minimal_block([0, 2]) == [0, 1, 0, 1, 0, 1]
    # 检查 P2 中 [0, 2] 区块的最小区块系统是否为 [0, 1, 0, 1, 0, 1]
    assert P2.minimal_block([0, 2]) == [0, 1, 0, 1, 0, 1]


def test_minimal_blocks():
    # 创建一个置换群对象 P，包含两个置换生成元
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    # 检查 P 的所有最小区块系统是否为 [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]]
    assert P.minimal_blocks() == [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]]

    # 创建一个5阶对称群对象 P
    P = SymmetricGroup(5)
    # 检查 P 的所有最小区块系统是否为 [[0, 0, 0, 0, 0]]
    assert P.minimal_blocks() == [[0]*5]

    # 创建一个置换群对象 P，包含一个置换生成元 (0, 3)
    P = PermutationGroup(Permutation(0, 3))
    # 检查 P 的最小区块系统是否为 False
    assert P.minimal_blocks() is False


def test_max_div():
    # 创建一个10阶对称群对象 S
    S = SymmetricGroup(10)
    # 检查 S 的最大不变子群阶数是否为 5
    assert S.max_div == 5


def test_is_primitive():
    # 创建一个5阶对称群对象 S
    S = SymmetricGroup(5)
    # 检查 S 是否是原始的
    assert S.is_primitive() is True
    # 创建一个7阶循环群对象 C
    C = CyclicGroup(7)
    # 检查 C 是否是原始的
    assert C.is_primitive() is True

    # 创建两个置换 a 和 b，生成一个置换群对象 G
    a = Permutation(0, 1, 2, size=6)
    b = Permutation(3, 4, 5, size=6)
    G = PermutationGroup(a, b)
    # 检查 G 是否是原始的
    assert G.is_primitive() is False


def test_random_stab():
    # 创建一个5阶对称群对象 S
    S = SymmetricGroup(5)
    # 创建一个随机置换 _random_el
    _random_el = Permutation([1, 3, 2, 0, 4])
    # 创建一个包含随机置换的字典 _random_prec
    _random_prec = {'rand': _random_el}
    # 计算 S 在给定精确数据的下，保持点集 {1, 2} 的随机稳定子群
    g = S.random_stab(2, _random_prec=_random_prec)
    # 检查计算结果是否与预期的随机置换相同
    assert g == Permutation([1, 3, 2, 0, 4])
    # 计算 S 在给定精确数据的下，保持点 1 的随机稳定子群
    h = S.random_stab(1)
    # 检查计算结果是否保持点 1 不变
    assert h(1) == 1


def test_transitivity_degree():
    # 创建一个置换 perm
    perm = Permutation([1, 2, 0])
    # 创建一个包含置换 perm 的置换群对象 C
    C = PermutationGroup([perm])
    # 检查 C 的传递度是否为 1
    assert C.transitivity_degree == 1
    # 创建两个置换生成元 gen1 和 gen2
    gen1 = Permutation([1, 2, 0, 3, 4])
    gen2 = Permutation([1, 2, 3, 4, 0])
    # 创建一个阶为 5 的交错群对象 Alt
    Alt = PermutationGroup([gen1, gen2])
    # 检查 Alt 的传递度是否为 3
    assert Alt.transitivity_degree == 3


def test_schreier_sims_random():
    # 检查 Tetra.pgroup 的基是否为 [0, 1] 的排序列表
    assert sorted(Tetra.pgroup.base) == [0, 1]

    # 创建一个3阶对称群对象 S
    S = SymmetricGroup(3)
    # 创建一个基 base 和强生成元列表 strong_gens
    base = [0, 1]
    strong_gens = [Permutation([1, 2, 0]), Permutation([1, 0, 2]),
                   Permutation([0, 2, 1])]
    # 检查 S 在给定基和强生成元下的随机 Schreier Sims 运算是否返回原始的基和强生成元
    assert S.schreier_sims_random(base, strong_gens, 5) == (base, strong_gens)

    # 创建一个3阶二面体群对象 D
    D = DihedralGroup(3)
    # 创建一个包含随机生成元的字典 _random_prec
    _random_prec = {'g': [Permutation([2, 0, 1]), Permutation([1, 2, 0]),
                          Permutation([1, 0, 2])]}
    # 创建一个基 base 和强生成元列表 strong_gens
    base = [0, 1]
    strong_gens = [Permutation([1, 2, 0]), Permutation([2, 1, 0]),
                   Permutation([0, 2, 1])]
    # 检查 D 在空基和其所有生成元下的随机 Schreier Sims 运算是否返回给定的基和强生成元
    assert D.schreier_sims_random([], D.generators, 2,
                                   _random_prec=_random_prec) == (base, strong_gens)


def test_baseswap():
    # 创建一个4阶对称群对象 S
    S = SymmetricGroup(4)
    # 执行 Schreier Sims
    # 断言：验证通过确定性参数的 BSGS 算法结果为 True
    assert _verify_bsgs(S, deterministic[0], deterministic[1]) is True
    
    # 断言：验证随机化参数的第一个元素等于列表 [0, 2, 1]
    assert randomized[0] == [0, 2, 1]
    
    # 断言：验证通过随机化参数的 BSGS 算法结果为 True
    assert _verify_bsgs(S, randomized[0], randomized[1]) is True
def test_schreier_sims_incremental():
    # 创建一个包含单位置换的群
    identity = Permutation([0, 1, 2, 3, 4])
    TrivialGroup = PermutationGroup([identity])
    # 使用 Schreier-Sims 算法增量方式计算基和强生成元
    base, strong_gens = TrivialGroup.schreier_sims_incremental(base=[0, 1, 2])
    # 验证基和强生成元是否满足 BSGS 条件
    assert _verify_bsgs(TrivialGroup, base, strong_gens) is True

    # 创建对称群，并使用 Schreier-Sims 算法增量方式计算基和强生成元
    S = SymmetricGroup(5)
    base, strong_gens = S.schreier_sims_incremental(base=[0, 1, 2])
    # 验证基和强生成元是否满足 BSGS 条件
    assert _verify_bsgs(S, base, strong_gens) is True

    # 创建二面角群，并使用 Schreier-Sims 算法增量方式计算基和强生成元
    D = DihedralGroup(2)
    base, strong_gens = D.schreier_sims_incremental(base=[1])
    # 验证基和强生成元是否满足 BSGS 条件
    assert _verify_bsgs(D, base, strong_gens) is True

    # 创建7阶交错群，并处理生成元
    A = AlternatingGroup(7)
    gens = A.generators[:]
    gen0 = gens[0]
    gen1 = gens[1]
    gen1 = rmul(gen1, ~gen0)  # 计算 gen1 的逆乘 gen0 的结果
    gen0 = rmul(gen0, gen1)   # 计算 gen0 乘 gen1 的结果
    gen1 = rmul(gen0, gen1)   # 计算 gen1 乘 gen0 乘 gen1 的结果
    # 使用 Schreier-Sims 算法增量方式计算基和强生成元
    base, strong_gens = A.schreier_sims_incremental(base=[0, 1], gens=gens)
    # 验证基和强生成元是否满足 BSGS 条件
    assert _verify_bsgs(A, base, strong_gens) is True

    # 创建11阶循环群，并使用 Schreier-Sims 算法增量方式计算基和强生成元
    C = CyclicGroup(11)
    gen = C.generators[0]
    base, strong_gens = C.schreier_sims_incremental(gens=[gen**3])
    # 验证基和强生成元是否满足 BSGS 条件
    assert _verify_bsgs(C, base, strong_gens) is True


def _subgroup_search(i, j, k):
    # 定义不同的属性测试函数
    prop_true = lambda x: True
    prop_fix_points = lambda x: [x(point) for point in points] == points
    prop_comm_g = lambda x: rmul(x, g) == rmul(g, x)
    prop_even = lambda x: x.is_even
    # 循环测试不同阶数的群
    for i in range(i, j, k):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        C = CyclicGroup(i)
        # 在对称群中搜索满足 prop_true 的子群，并断言其确实是 S 的子群
        Sym = S.subgroup_search(prop_true)
        assert Sym.is_subgroup(S)
        # 在对称群中搜索满足 prop_even 的子群，并断言其确实是 A 的子群
        Alt = S.subgroup_search(prop_even)
        assert Alt.is_subgroup(A)
        # 在对称群中搜索满足 prop_true 的子群，初始子群为 C，并断言其确实是 S 的子群
        Sym = S.subgroup_search(prop_true, init_subgroup=C)
        assert Sym.is_subgroup(S)
        # 设定测试点为 [7]，并断言其稳定子群为 S 中满足 prop_fix_points 的子群
        points = [7]
        assert S.stabilizer(7).is_subgroup(S.subgroup_search(prop_fix_points))
        # 设定测试点为 [3, 4]，并断言其稳定子群为 S 中稳定子群的稳定子群中满足 prop_fix_points 的子群
        points = [3, 4]
        assert S.stabilizer(3).stabilizer(4).is_subgroup(
            S.subgroup_search(prop_fix_points))
        # 设定测试点为 [3, 5]，并断言 A 中满足 prop_fix_points 的子群，初始子群为 fix35
        points = [3, 5]
        fix35 = A.subgroup_search(prop_fix_points)
        points = [5]
        fix5 = A.subgroup_search(prop_fix_points)
        assert A.subgroup_search(prop_fix_points, init_subgroup=fix35
            ).is_subgroup(fix5)
        # 使用 Schreier-Sims 算法增量方式计算 A 的基和强生成元
        base, strong_gens = A.schreier_sims_incremental()
        g = A.generators[0]
        # 在 A 中搜索满足 prop_comm_g 的子群，基为 base，强生成元为 strong_gens
        comm_g = \
            A.subgroup_search(prop_comm_g, base=base, strong_gens=strong_gens)
        # 验证基和强生成元是否满足 BSGS 条件
        assert _verify_bsgs(comm_g, base, comm_g.generators) is True
        # 断言 comm_g 中所有生成元满足 prop_comm_g
        assert [prop_comm_g(gen) is True for gen in comm_g.generators]


def test_subgroup_search():
    # 调用 _subgroup_search 函数测试阶数在 [10, 15) 范围内的群
    _subgroup_search(10, 15, 2)


@XFAIL
def test_subgroup_search2():
    # 由于运行时间过长，标记为测试失败，输出跳过信息
    skip('takes too much time')
    # 调用 _subgroup_search 函数测试阶数为 16 的群
    _subgroup_search(16, 17, 1)


def test_normal_closure():
    # 对称群 S3 的单位元的正规闭包是它自己
    S = SymmetricGroup(3)
    identity = Permutation([0, 1, 2])
    closure = S.normal_closure(identity)
    assert closure.is_trivial
    # 交错群 A4 的整个群的正规闭包是它自己
    A = AlternatingGroup(4)
    assert A.normal_closure(A).is_subgroup(A)
    # 针对子群进行暴力验证
    for i in (3, 4, 5):
        # 创建对称群、交错群、二面角群和循环群的实例
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        D = DihedralGroup(i)
        C = CyclicGroup(i)
        for gp in (A, D, C):
            # 断言子群 gp 的正规闭包在对称群 S 中
            assert _verify_normal_closure(S, gp)
    
    # 针对群中所有元素进行暴力验证
    S = SymmetricGroup(5)
    elements = list(S.generate_dimino())
    for element in elements:
        # 断言群 S 的每个元素的正规闭包在群 S 中
        assert _verify_normal_closure(S, element)
    
    # 小群的处理
    small = []
    for i in (1, 2, 3):
        # 添加对称群、交错群、二面角群和循环群的实例到列表中
        small.append(SymmetricGroup(i))
        small.append(AlternatingGroup(i))
        small.append(DihedralGroup(i))
        small.append(CyclicGroup(i))
    for gp in small:
        for gp2 in small:
            # 如果 gp2 是 gp 的子群且阶数相同，则进行断言
            if gp2.is_subgroup(gp, 0) and gp2.degree == gp.degree:
                assert _verify_normal_closure(gp, gp2)
def test_derived_series():
    # 测试从派生序列的构建
    # 对于平凡群的派生序列，只包含平凡群本身
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert triv.derived_series()[0].is_subgroup(triv)
    
    # 对于简单群的派生序列，只包含群本身
    for i in (5, 6, 7):
        A = AlternatingGroup(i)
        assert A.derived_series()[0].is_subgroup(A)
    
    # 对于 S_4 的派生序列是 S_4 > A_4 > K_4 > triv
    S = SymmetricGroup(4)
    series = S.derived_series()
    assert series[1].is_subgroup(AlternatingGroup(4))
    assert series[2].is_subgroup(DihedralGroup(2))
    assert series[3].is_trivial


def test_lower_central_series():
    # 测试从下中心级数的构建
    # 对于平凡群的下中心级数，只包含平凡群本身
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert triv.lower_central_series()[0].is_subgroup(triv)
    
    # 对于简单群的下中心级数，只包含群本身
    for i in (5, 6, 7):
        A = AlternatingGroup(i)
        assert A.lower_central_series()[0].is_subgroup(A)
    
    # 对于 GAP 验证的示例
    S = SymmetricGroup(6)
    series = S.lower_central_series()
    assert len(series) == 2
    assert series[1].is_subgroup(AlternatingGroup(6))


def test_commutator():
    # 测试交换子的性质
    # 平凡群与平凡群的交换子是平凡群
    S = SymmetricGroup(3)
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert S.commutator(triv, triv).is_subgroup(triv)
    
    # 平凡群与任何其他群的交换子仍然是平凡群
    A = AlternatingGroup(3)
    assert S.commutator(triv, A).is_subgroup(triv)
    
    # 交换子是可交换的
    for i in (3, 4, 5):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        D = DihedralGroup(i)
        assert S.commutator(A, D).is_subgroup(S.commutator(D, A))
    
    # 交换子对于阿贝尔群是平凡的
    S = SymmetricGroup(7)
    A1 = AbelianGroup(2, 5)
    A2 = AbelianGroup(3, 4)
    triv = PermutationGroup([Permutation([0, 1, 2, 3, 4, 5, 6])])
    assert S.commutator(A1, A1).is_subgroup(triv)
    assert S.commutator(A2, A2).is_subgroup(triv)
    
    # 通过手工计算得出的例子
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    assert S.commutator(A, S).is_subgroup(A)


def test_is_nilpotent():
    # 测试是否幂零的性质
    # 每个阿贝尔群都是幂零的
    for i in (1, 2, 3):
        C = CyclicGroup(i)
        Ab = AbelianGroup(i, i + 2)
        assert C.is_nilpotent
        assert Ab.is_nilpotent
    
    Ab = AbelianGroup(5, 7, 10)
    assert Ab.is_nilpotent
    
    # A_5 不是可解的，因此不是幂零的
    assert AlternatingGroup(5).is_nilpotent is False


def test_is_trivial():
    # 测试群是否平凡的性质
    for i in range(5):
        triv = PermutationGroup([Permutation(list(range(i)))])
        assert triv.is_trivial


def test_pointwise_stabilizer():
    # 测试点稳定子的性质
    S = SymmetricGroup(2)
    stab = S.pointwise_stabilizer([0])
    assert stab.generators == [Permutation(1)]
    
    S = SymmetricGroup(5)
    # 创建一个空列表，用于存储点的集合
    points = []
    # 将变量 stab 初始化为 S
    stab = S
    # 遍历固定顺序的点集合 (2, 0, 3, 4, 1)
    for point in (2, 0, 3, 4, 1):
        # 调用 stab 对象的 stabilizer 方法，更新 stab 变量为当前点的稳定子群
        stab = stab.stabilizer(point)
        # 将当前点添加到 points 列表中
        points.append(point)
        # 断言：S 中点逐点的稳定子群是 stab 的子群
        assert S.pointwise_stabilizer(points).is_subgroup(stab)
# 测试函数，用于测试 make_perm 函数是否正确生成置换对象
def test_make_perm():
    # 断言：调用 cube.pgroup.make_perm 函数生成置换对象，与预期的 Permutation 对象相等
    assert cube.pgroup.make_perm(5, seed=list(range(5))) == \
        Permutation([4, 7, 6, 5, 0, 3, 2, 1])
    # 断言：调用 cube.pgroup.make_perm 函数生成置换对象，与预期的 Permutation 对象相等
    assert cube.pgroup.make_perm(7, seed=list(range(7))) == \
        Permutation([6, 7, 3, 2, 5, 4, 0, 1])


# 测试函数，用于测试 elements 方法是否正确生成置换群的全部元素
def test_elements():
    # 导入必要的模块和类
    from sympy.sets.sets import FiniteSet

    # 创建置换对象 p
    p = Permutation(2, 3)
    # 断言：将 PermutationGroup(p) 的全部元素转为集合，与预期的集合相等
    assert set(PermutationGroup(p).elements) == {Permutation(3), Permutation(2, 3)}
    # 断言：将 PermutationGroup(p) 的全部元素转为有限集，与预期的有限集相等
    assert FiniteSet(*PermutationGroup(p).elements) \
        == FiniteSet(Permutation(2, 3), Permutation(3))


# 测试函数，用于测试 is_group 方法是否能正确判断置换群是否为群
def test_is_group():
    # 断言：创建的 PermutationGroup 包含的置换构成的对象是否为群
    assert PermutationGroup(Permutation(1,2), Permutation(2,4)).is_group is True
    # 断言：SymmetricGroup(4) 是否为群
    assert SymmetricGroup(4).is_group is True


# 测试函数，用于测试 PermutationGroup 对象的相等性和是否等于 0
def test_PermutationGroup():
    # 断言：创建的空 PermutationGroup 对象是否等于只包含单个 Permutation 对象的 PermutationGroup 对象
    assert PermutationGroup() == PermutationGroup(Permutation())
    # 断言：创建的空 PermutationGroup 对象是否不等于 0
    assert (PermutationGroup() == 0) is False


# 测试函数，用于测试 AlternatingGroup 的 coset_transversal 方法是否正确生成陪集系统
def test_coset_transvesal():
    # 创建 AlternatingGroup 和一个 PermutationGroup
    G = AlternatingGroup(5)
    H = PermutationGroup(Permutation(0,1,2),Permutation(1,2)(3,4))
    # 断言：使用 G.coset_transversal(H) 生成的陪集系统列表，与预期的列表相等
    assert G.coset_transversal(H) == \
        [Permutation(4), Permutation(2, 3, 4), Permutation(2, 4, 3),
         Permutation(1, 2, 4), Permutation(4)(1, 2, 3), Permutation(1, 3)(2, 4),
         Permutation(0, 1, 2, 3, 4), Permutation(0, 1, 2, 4, 3),
         Permutation(0, 1, 3, 2, 4), Permutation(0, 2, 4, 1, 3)]


# 测试函数，用于测试 PermutationGroup 的 coset_table 方法是否正确生成陪集表
def test_coset_table():
    # 创建两个 PermutationGroup 对象 G 和 H
    G = PermutationGroup(Permutation(0,1,2,3), Permutation(0,1,2),
         Permutation(0,4,2,7), Permutation(5,6), Permutation(0,7));
    H = PermutationGroup(Permutation(0,1,2,3), Permutation(0,7))
    # 断言：使用 G.coset_table(H) 生成的陪集表，与预期的列表相等
    assert G.coset_table(H) == \
        [[0, 0, 0, 0, 1, 2, 3, 3, 0, 0], [4, 5, 2, 5, 6, 0, 7, 7, 1, 1],
         [5, 4, 5, 1, 0, 6, 8, 8, 6, 6], [3, 3, 3, 3, 7, 8, 0, 0, 3, 3],
         [2, 1, 4, 4, 4, 4, 9, 9, 4, 4], [1, 2, 1, 2, 5, 5, 10, 10, 5, 5],
         [6, 6, 6, 6, 2, 1, 11, 11, 2, 2], [9, 10, 8, 10, 11, 3, 1, 1, 7, 7],
         [10, 9, 10, 7, 3, 11, 2, 2, 11, 11], [8, 7, 9, 9, 9, 9, 4, 4, 9, 9],
         [7, 8, 7, 8, 10, 10, 5, 5, 10, 10], [11, 11, 11, 11, 8, 7, 6, 6, 8, 8]]


# 测试函数，用于测试 subgroup 方法是否正确生成子群
def test_subgroup():
    # 创建一个 PermutationGroup 对象 G，生成其一个子群 H
    G = PermutationGroup(Permutation(0,1,2), Permutation(0,2,3))
    H = G.subgroup([Permutation(0,1,3)])
    # 断言：子群 H 是否为 G 的子群
    assert H.is_subgroup(G)


# 测试函数，用于测试 generator_product 方法是否正确生成生成元的乘积
def test_generator_product():
    # 创建 SymmetricGroup(5) 对象 G 和一个 Permutation 对象 p
    G = SymmetricGroup(5)
    p = Permutation(0, 2, 3)(1, 4)
    # 调用 G.generator_product 方法生成生成元的乘积列表 gens
    gens = G.generator_product(p)
    # 断言：gens 中的每个元素是否都在 G 的强生成元中
    assert all(g in G.strong_gens for g in gens)
    # 初始化置换 w 为 G 的单位元
    w = G.identity
    # 循环计算所有生成元的乘积，检查是否等于 p
    for g in gens:
        w = g * w
    assert w == p


# 测试函数，用于测试 sylow_subgroup 方法是否正确生成 Sylow 子群
def test_sylow_subgroup():
    # 创建 PermutationGroup 对象 P，测试其 sylow_subgroup 方法
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    # 断言：生成的 Sylow 2-子群 S 的阶是否为 4
    S = P.sylow_subgroup(2)
    assert S.order() == 4

    # 创建 DihedralGroup(12) 对象 P，测试其 sylow_subgroup 方法
    P = DihedralGroup(12)
    # 断言：生成的 Sylow 3-子群 S 的阶是否为 3
    S = P.sylow_subgroup(3)
    assert S.order() == 3

    # 创建另一个 PermutationGroup 对象 P，测试其 sylow_subgroup 方法
    P = PermutationGroup(
        Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5), Permutation(0, 2))
    # 断言：生成的 Sylow 3-子群 S 的阶是否为 9
    S = P.sylow_subgroup(3)
    assert S.order() == 9
    # 断言：生成的 Sylow 2
    # 获取 SymmetricGroup 对象 P 中的一个 5-Sylow 子群 S
    S = P.sylow_subgroup(5)
    # 断言 S 的阶数为 25
    assert S.order() == 25

    # 下降中心级数的长度
    # 对于 Sym(n) 的一个 p-Sylow 子群来说，其下降中心级数的长度随着
    # 最高指数 exp 的增长而增长，其中 p 的指数 exp 满足 n >= p**exp
    exp = 1
    length = 0
    for i in range(2, 9):
        # 创建 SymmetricGroup 对象 P，其阶数为 i
        P = SymmetricGroup(i)
        # 获取 P 中的一个 2-Sylow 子群 S
        S = P.sylow_subgroup(2)
        # 获取 S 的下降中心级数
        ls = S.lower_central_series()
        if i // 2**exp > 0:
            # 断言下降中心级数的长度随着指数的增加而增加
            assert len(ls) > length
            length = len(ls)
            exp += 1
        else:
            # 断言下降中心级数的长度保持不变
            assert len(ls) == length

    # 创建 SymmetricGroup 对象 G，其阶数为 100
    G = SymmetricGroup(100)
    # 获取 G 中的一个 3-Sylow 子群 S
    S = G.sylow_subgroup(3)
    # 断言 G 的阶数能被 S 的阶数整除
    assert G.order() % S.order() == 0
    # 断言 G 的阶数除以 S 的阶数不被 3 整除
    assert G.order()/S.order() % 3 > 0

    # 创建 AlternatingGroup 对象 G，其阶数为 100
    G = AlternatingGroup(100)
    # 获取 G 中的一个 2-Sylow 子群 S
    S = G.sylow_subgroup(2)
    # 断言 G 的阶数能被 S 的阶数整除
    assert G.order() % S.order() == 0
    # 断言 G 的阶数除以 S 的阶数不被 2 整除
    assert G.order()/S.order() % 2 > 0

    # 创建 DihedralGroup 对象 G，其阶数为 18
    G = DihedralGroup(18)
    # 获取 G 中的一个 2-Sylow 子群 S
    S = G.sylow_subgroup(p=2)
    # 断言 S 的阶数为 4
    assert S.order() == 4

    # 创建 DihedralGroup 对象 G，其阶数为 50
    G = DihedralGroup(50)
    # 获取 G 中的一个 2-Sylow 子群 S
    S = G.sylow_subgroup(p=2)
    # 断言 S 的阶数为 4
    assert S.order() == 4
# 标记为慢速测试
@slow
# 定义名为 test_presentation 的测试函数
def test_presentation():
    # 定义内部函数 _test，用于测试给定置换群 P 的演示
    def _test(P):
        # 获取 P 的演示，生成置换群 G
        G = P.presentation()
        # 检查 G 的阶是否与 P 的阶相等，返回布尔值
        return G.order() == P.order()

    # 定义内部函数 _strong_test，用于测试给定置换群 P 的强演示
    def _strong_test(P):
        # 获取 P 的强演示，生成置换群 G
        G = P.strong_presentation()
        # 检查 G 的生成元数量是否与 P 的强生成元数量相等
        chk = len(G.generators) == len(P.strong_gens)
        # 检查 G 的阶是否与 P 的阶相等，并返回两个条件的布尔值
        return chk and G.order() == P.order()

    # 创建置换群 P，包含两个置换
    P = PermutationGroup(Permutation(0,1,5,2)(3,7,4,6), Permutation(0,3,5,4)(1,6,2,7))
    # 断言调用 _test 函数返回 True
    assert _test(P)

    # 创建 5 阶交错群
    P = AlternatingGroup(5)
    # 断言调用 _test 函数返回 True
    assert _test(P)

    # 创建 5 阶对称群
    P = SymmetricGroup(5)
    # 断言调用 _test 函数返回 True
    assert _test(P)

    # 创建置换群 P，包含三个置换
    P = PermutationGroup(
        [Permutation(0,3,1,2), Permutation(3)(0,1), Permutation(0,1)(2,3)])
    # 断言调用 _strong_test 函数返回 True
    assert _strong_test(P)

    # 创建 6 阶二面体群
    P = DihedralGroup(6)
    # 断言调用 _strong_test 函数返回 True
    assert _strong_test(P)

    # 创建三个置换 a, b, c，并作为参数创建置换群 P
    a = Permutation(0,1)(2,3)
    b = Permutation(0,2)(3,1)
    c = Permutation(4,5)
    P = PermutationGroup(c, a, b)
    # 断言调用 _strong_test 函数返回 True
    assert _strong_test(P)


# 定义测试函数 test_polycyclic
def test_polycyclic():
    # 创建置换 a, b，并作为参数创建置换群 G
    a = Permutation([0, 1, 2])
    b = Permutation([2, 1, 0])
    G = PermutationGroup([a, b])
    # 断言 G 是否为多周期置换群
    assert G.is_polycyclic is True

    # 创建置换 a, b，并作为参数创建置换群 G
    a = Permutation([1, 2, 3, 4, 0])
    b = Permutation([1, 0, 2, 3, 4])
    G = PermutationGroup([a, b])
    # 断言 G 是否为多周期置换群
    assert G.is_polycyclic is False


# 定义测试函数 test_elementary
def test_elementary():
    # 创建置换 a，并作为参数创建置换群 G
    a = Permutation([1, 5, 2, 0, 3, 6, 4])
    G = PermutationGroup([a])
    # 断言 G 是否为 7 阶元素置换群
    assert G.is_elementary(7) is False

    # 创建置换 a, b，并作为参数创建置换群 G
    a = Permutation(0, 1)(2, 3)
    b = Permutation(0, 2)(3, 1)
    G = PermutationGroup([a, b])
    # 断言 G 是否为 2 阶元素置换群
    assert G.is_elementary(2) is True
    # 创建置换 c，并将其加入置换群 G
    c = Permutation(4, 5, 6)
    G = PermutationGroup([a, b, c])
    # 断言 G 是否为 2 阶元素置换群
    assert G.is_elementary(2) is False

    # 获取 4 阶对称群的 2-Sylow 子群 G
    G = SymmetricGroup(4).sylow_subgroup(2)
    # 断言 G 是否为 2 阶元素置换群
    assert G.is_elementary(2) is False
    # 获取 4 阶交错群的 2-Sylow 子群 H
    H = AlternatingGroup(4).sylow_subgroup(2)
    # 断言 H 是否为 2 阶元素置换群
    assert H.is_elementary(2) is True


# 定义测试函数 test_perfect
def test_perfect():
    # 创建 3 阶交错群 G
    G = AlternatingGroup(3)
    # 断言 G 是否为完美群
    assert G.is_perfect is False
    # 创建 5 阶交错群 G
    G = AlternatingGroup(5)
    # 断言 G 是否为完美群
    assert G.is_perfect is True


# 定义测试函数 test_index
def test_index():
    # 创建置换群 G，包含两个置换
    G = PermutationGroup(Permutation(0,1,2), Permutation(0,2,3))
    # 获取 G 关于子群 H 的指数，并断言其为 4
    H = G.subgroup([Permutation(0,1,3)])
    assert G.index(H) == 4


# 定义测试函数 test_cyclic
def test_cyclic():
    # 创建对称群 S_2，并断言其为循环群
    G = SymmetricGroup(2)
    assert G.is_cyclic
    # 创建阿贝尔群 Z_3 × Z_7，并断言其为循环群
    G = AbelianGroup(3, 7)
    assert G.is_cyclic
    # 创建阿贝尔群 Z_7 × Z_7，并断言其不为循环群
    G = AbelianGroup(7, 7)
    assert not G.is_cyclic
    # 创建 3 阶交错群，并断言其为循环群
    G = AlternatingGroup(3)
    assert G.is_cyclic
    # 创建 4 阶交错群，并断言其不为循环群
    G = AlternatingGroup(4)
    assert not G.is_cyclic

    # 创建 3 阶置换群 G，包含两个置换，并断言其为循环群
    G = PermutationGroup(Permutation(0, 1, 2), Permutation(0, 2, 1))
    assert G.is_cyclic
    # 创建 4 阶置换群 G，包含两个置换，并断言其为循环群
    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(0, 2)(1, 3)
    )
    assert G.is_cyclic
    # 创建 4 阶置换群 G，包含四个置换，并断言其不为循环群
    G = PermutationGroup(
        Permutation(3),
        Permutation(0, 1)(2, 3),
        Permutation(0, 2)(1, 3),
        Permutation(0, 3)(1, 2)
    )
    assert G.is_cyclic is False

    # 创建 15 阶置换群 G，包含两个 15 阶置换，并断言其为循环群
    G = PermutationGroup(
        Permutation(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        Permutation(0, 2, 4, 6, 8, 10, 12,
    # 使用 _distinct_primes_lemma 方法检查是否满足素数分离引理，预期返回 True
    assert PermutationGroup._distinct_primes_lemma([5, 7]) is True
    # 使用 _distinct_primes_lemma 方法检查是否满足素数分离引理，预期返回 None
    assert PermutationGroup._distinct_primes_lemma([2, 3]) is None
    # 使用 _distinct_primes_lemma 方法检查是否满足素数分离引理，预期返回 None
    assert PermutationGroup._distinct_primes_lemma([3, 5, 7]) is None
    # 使用 _distinct_primes_lemma 方法检查是否满足素数分离引理，预期返回 True
    assert PermutationGroup._distinct_primes_lemma([5, 7, 13]) is True

    # 创建一个置换群 G，包含两个置换：(0, 1, 2, 3) 和 (0, 2)(1, 3)
    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(0, 2)(1, 3))
    # 断言群 G 是否是循环群
    assert G.is_cyclic
    # 断言群 G 是否是交换群（阿贝尔群）
    assert G._is_abelian

    # 创建一个非阿贝尔群 G，因此不是循环群
    G = PermutationGroup(*SymmetricGroup(3).generators)
    assert G.is_cyclic is False

    # 创建一个阿贝尔群 G，包含两个置换：(0, 1, 2, 3) 和 (4, 5, 6)，因此是循环群
    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(4, 5, 6)
    )
    assert G.is_cyclic

    # 创建一个阿贝尔群 G，包含三个置换：(0, 1)，(2, 3)，(4, 5, 6)，因此不是循环群
    G = PermutationGroup(
        Permutation(0, 1),
        Permutation(2, 3),
        Permutation(4, 5, 6)
    )
    assert G.is_cyclic is False
def test_dihedral():
    # 创建对称群 S_2，并检查是否是二面体群
    G = SymmetricGroup(2)
    assert G.is_dihedral
    # 创建对称群 S_3，并检查是否是二面体群
    G = SymmetricGroup(3)
    assert G.is_dihedral

    # 创建阿贝尔群 Z_2 x Z_2，并检查是否是二面体群
    G = AbelianGroup(2, 2)
    assert G.is_dihedral
    # 创建循环群 Z_4，并检查是否是二面体群
    G = CyclicGroup(4)
    assert not G.is_dihedral

    # 创建阿贝尔群 Z_3 x Z_5，并检查是否是二面体群
    G = AbelianGroup(3, 5)
    assert not G.is_dihedral
    # 创建阿贝尔群 Z_2，并检查是否是二面体群
    G = AbelianGroup(2)
    assert G.is_dihedral
    # 创建阿贝尔群 Z_6，并检查是否是二面体群
    G = AbelianGroup(6)
    assert not G.is_dihedral

    # 创建 D_6，由两个相邻的翻转生成
    G = PermutationGroup(
        Permutation(1, 5)(2, 4),
        Permutation(0, 1)(3, 4)(2, 5))
    assert G.is_dihedral

    # 创建 D_7，由一个翻转和一个旋转生成
    G = PermutationGroup(
        Permutation(1, 6)(2, 5)(3, 4),
        Permutation(0, 1, 2, 3, 4, 5, 6))
    assert G.is_dihedral

    # 创建 S_4，由三个生成器表示，因具有恰好 9 个阶为 2 的元素而失败
    G = PermutationGroup(
        Permutation(0, 1), Permutation(0, 2),
        Permutation(0, 3))
    assert not G.is_dihedral

    # 创建 D_7，由三个生成器表示
    G = PermutationGroup(
        Permutation(1, 6)(2, 5)(3, 4),
        Permutation(2, 0)(3, 6)(4, 5),
        Permutation(0, 1, 2, 3, 4, 5, 6))
    assert G.is_dihedral


def test_abelian_invariants():
    # 创建阿贝尔群 Z_2 x Z_3 x Z_4，并验证其阿贝尔不变量
    G = AbelianGroup(2, 3, 4)
    assert G.abelian_invariants() == [2, 3, 4]
    
    # 创建置换群 S_6，并验证其阿贝尔不变量
    G = PermutationGroup([Permutation(1, 2, 3, 4), Permutation(1, 2), Permutation(5, 6)])
    assert G.abelian_invariants() == [2, 2]
    
    # 创建交错群 A_7，并验证其阿贝尔不变量
    G = AlternatingGroup(7)
    assert G.abelian_invariants() == []
    
    # 创建交错群 A_4，并验证其阿贝尔不变量
    G = AlternatingGroup(4)
    assert G.abelian_invariants() == [3]
    
    # 创建二面体群 D_4，并验证其阿贝尔不变量
    G = DihedralGroup(4)
    assert G.abelian_invariants() == [2, 2]

    # 创建置换群 S_7，并验证其阿贝尔不变量
    G = PermutationGroup([Permutation(1, 2, 3, 4, 5, 6, 7)])
    assert G.abelian_invariants() == [7]
    
    # 创建二面体群 D_12，并获取其 3 阶 Sylow 子群，验证其阿贝尔不变量
    G = DihedralGroup(12)
    S = G.sylow_subgroup(3)
    assert S.abelian_invariants() == [3]
    
    # 创建置换群 S_4，并验证其阿贝尔不变量
    G = PermutationGroup(Permutation(0, 1, 2), Permutation(0, 2, 3))
    assert G.abelian_invariants() == [3]
    
    # 创建置换群 S_8，并验证其阿贝尔不变量
    G = PermutationGroup([Permutation(0, 1), Permutation(0, 2, 4, 6)(1, 3, 5, 7)])
    assert G.abelian_invariants() == [2, 4]
    
    # 创建对称群 S_30，并获取其 2 阶 Sylow 子群，验证其阿贝尔不变量
    G = SymmetricGroup(30)
    S = G.sylow_subgroup(2)
    assert S.abelian_invariants() == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    
    # 获取对称群 S_30 的 3 阶 Sylow 子群，验证其阿贝尔不变量
    S = G.sylow_subgroup(3)
    assert S.abelian_invariants() == [3, 3, 3, 3]
    

def test_composition_series():
    a = Permutation(1, 2, 3)
    b = Permutation(1, 2)
    # 创建置换群 S_3，并验证其组合序列和导出序列相等
    G = PermutationGroup([a, b])
    comp_series = G.composition_series()
    assert comp_series == G.derived_series()
    
    # 创建对称群 S_4，并验证其组合序列的第一个群为其自身，最后一个群为平凡群
    S = SymmetricGroup(4)
    assert S.composition_series()[0] == S
    assert len(S.composition_series()) == 5
    
    # 创建交错群 A_4，并验证其组合序列的第一个群为其自身，最后一个群为平凡群
    A = AlternatingGroup(4)
    assert A.composition_series()[0] == A
    assert len(A.composition_series()) == 4

    # 创建循环群 C_8，并验证其组合序列
    G = CyclicGroup(8)
    # 调用对象 G 的 composition_series 方法，返回一个群的构成序列
    series = G.composition_series()
    # 断言序列的第一个元素与一个阶为 4 的循环群同构
    assert is_isomorphic(series[1], CyclicGroup(4))
    # 断言序列的第二个元素与一个阶为 2 的循环群同构
    assert is_isomorphic(series[2], CyclicGroup(2))
    # 断言序列的第三个元素是一个平凡群（即单位元素）
    assert series[3].is_trivial
# 定义测试函数，用于验证置换群是否对称
def test_is_symmetric():
    # 创建置换 a 和 b
    a = Permutation(0, 1, 2)
    b = Permutation(0, 1, size=3)
    # 断言两个置换组成的置换群是对称的
    assert PermutationGroup(a, b).is_symmetric is True

    a = Permutation(0, 2, 1)
    b = Permutation(1, 2, size=3)
    assert PermutationGroup(a, b).is_symmetric is True

    a = Permutation(0, 1, 2, 3)
    b = Permutation(0, 3)(1, 2)
    # 断言两个置换组成的置换群不是对称的
    assert PermutationGroup(a, b).is_symmetric is False

# 定义测试函数，用于验证对称群的共轭类
def test_conjugacy_class():
    # 创建度为 4 的对称群
    S = SymmetricGroup(4)
    x = Permutation(1, 2, 3)
    # 定义预期的共轭类集合 C
    C = {Permutation(0, 1, 2, size=4), Permutation(0, 1, 3),
         Permutation(0, 2, 1, size=4), Permutation(0, 2, 3),
         Permutation(0, 3, 1), Permutation(0, 3, 2),
         Permutation(1, 2, 3), Permutation(1, 3, 2)}
    # 断言给定置换 x 的共轭类与预期相等
    assert S.conjugacy_class(x) == C

# 定义测试函数，用于验证对称群的所有共轭类
def test_conjugacy_classes():
    # 创建度为 3 的对称群
    S = SymmetricGroup(3)
    # 定义预期的所有共轭类列表 expected
    expected = [{Permutation(size=3)},
                {Permutation(0, 1, size=3), Permutation(0, 2), Permutation(1, 2)},
                {Permutation(0, 1, 2), Permutation(0, 2, 1)}]
    # 计算得到对称群的所有共轭类 computed
    computed = S.conjugacy_classes()

    # 断言预期共轭类数量与计算出的共轭类数量相等
    assert len(expected) == len(computed)
    # 断言所有预期共轭类都包含在计算出的共轭类中
    assert all(e in computed for e in expected)

# 定义测试函数，用于验证置换群的左右陪集
def test_coset_class():
    a = Permutation(1, 2)
    b = Permutation(0, 1)
    G = PermutationGroup([a, b])
    # 创建右陪集 rht_coset
    rht_coset = G * a
    # 断言 rht_coset 是右陪集而不是左陪集
    assert rht_coset.is_right_coset
    assert not rht_coset.is_left_coset
    # 创建右陪集的列表表示 list_repr
    list_repr = rht_coset.as_list()
    expected = [Permutation(0, 2), Permutation(0, 2, 1), Permutation(1, 2),
                Permutation(2), Permutation(2)(0, 1), Permutation(0, 1, 2)]
    # 断言 list_repr 中的每个元素都在预期的列表 expected 中
    for ele in list_repr:
        assert ele in expected

    # 创建左陪集 left_coset
    left_coset = a * G
    # 断言 left_coset 是左陪集而不是右陪集
    assert not left_coset.is_right_coset
    assert left_coset.is_left_coset
    # 创建左陪集的列表表示 list_repr
    list_repr = left_coset.as_list()
    expected = [Permutation(2)(0, 1), Permutation(0, 1, 2), Permutation(1, 2),
                Permutation(2), Permutation(0, 2), Permutation(0, 2, 1)]
    # 断言 list_repr 中的每个元素都在预期的列表 expected 中
    for ele in list_repr:
        assert ele in expected

    # 创建置换群 G 和 H
    G = PermutationGroup(Permutation(1, 2, 3, 4), Permutation(2, 3, 4))
    H = PermutationGroup(Permutation(1, 2, 3, 4))
    g = Permutation(1, 3)(2, 4)
    # 创建右陪集 rht_coset
    rht_coset = Coset(g, H, G, dir='+')
    # 断言 rht_coset 是右陪集
    assert rht_coset.is_right_coset
    # 创建右陪集的列表表示 list_repr
    list_repr = rht_coset.as_list()
    expected = [Permutation(1, 2, 3, 4), Permutation(4), Permutation(1, 3)(2, 4),
                Permutation(1, 4, 3, 2)]
    # 断言 list_repr 中的每个元素都在预期的列表 expected 中
    for ele in list_repr:
        assert ele in expected

# 定义测试函数，用于验证对称置换群的性质
def test_symmetricpermutationgroup():
    a = SymmetricPermutationGroup(5)
    # 断言对称置换群的度为 5
    assert a.degree == 5
    # 断言对称置换群的阶数为 120
    assert a.order() == 120
    # 断言对称置换群的单位元素是置换 (4)
    assert a.identity() == Permutation(4)
```