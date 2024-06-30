# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_util.py`

```
from sympy.combinatorics.named_groups import SymmetricGroup, DihedralGroup,\
    AlternatingGroup
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.util import _check_cycles_alt_sym, _strip,\
    _distribute_gens_by_base, _strong_gens_from_distr,\
    _orbits_transversals_from_bsgs, _handle_precomputed_bsgs, _base_ordering,\
    _remove_gens
from sympy.combinatorics.testutil import _verify_bsgs

# 定义测试函数，检查带有交替或对称循环的置换是否满足条件
def test_check_cycles_alt_sym():
    # 创建置换对象，定义置换的循环结构
    perm1 = Permutation([[0, 1, 2, 3, 4, 5, 6], [7], [8], [9]])
    perm2 = Permutation([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]])
    perm3 = Permutation([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    # 断言每个置换是否符合交替或对称循环的条件
    assert _check_cycles_alt_sym(perm1) is True
    assert _check_cycles_alt_sym(perm2) is False
    assert _check_cycles_alt_sym(perm3) is False

# 定义测试函数，检查在置换群中剥离置换的作用
def test_strip():
    # 创建二面体群对象，进行 Schreier-Sims 算法初始化
    D = DihedralGroup(5)
    D.schreier_sims()
    # 定义几个置换对象
    member = Permutation([4, 0, 1, 2, 3])
    not_member1 = Permutation([0, 1, 4, 3, 2])
    not_member2 = Permutation([3, 1, 4, 2, 0])
    identity = Permutation([0, 1, 2, 3, 4])
    # 使用 _strip 函数分别处理这些置换
    res1 = _strip(member, D.base, D.basic_orbits, D.basic_transversals)
    res2 = _strip(not_member1, D.base, D.basic_orbits, D.basic_transversals)
    res3 = _strip(not_member2, D.base, D.basic_orbits, D.basic_transversals)
    # 断言处理结果是否符合预期
    assert res1[0] == identity
    assert res1[1] == len(D.base) + 1
    assert res2[0] == not_member1
    assert res2[1] == len(D.base) + 1
    assert res3[0] != identity
    assert res3[1] == 2

# 定义测试函数，检查生成器如何根据基底分布
def test_distribute_gens_by_base():
    base = [0, 1, 2]
    gens = [Permutation([0, 1, 2, 3]), Permutation([0, 1, 3, 2]),
           Permutation([0, 2, 3, 1]), Permutation([3, 2, 1, 0])]
    # 使用 _distribute_gens_by_base 函数按照给定基底分发生成器
    assert _distribute_gens_by_base(base, gens) == [gens,
                                                   [Permutation([0, 1, 2, 3]),
                                                   Permutation([0, 1, 3, 2]),
                                                   Permutation([0, 2, 3, 1])],
                                                   [Permutation([0, 1, 2, 3]),
                                                   Permutation([0, 1, 3, 2])]]

# 定义测试函数，检查如何从分布中获取强生成器
def test_strong_gens_from_distr():
    strong_gens_distr = [[Permutation([0, 2, 1]), Permutation([1, 2, 0]),
                  Permutation([1, 0, 2])], [Permutation([0, 2, 1])]]
    # 使用 _strong_gens_from_distr 函数获取强生成器列表
    assert _strong_gens_from_distr(strong_gens_distr) == \
        [Permutation([0, 2, 1]),
         Permutation([1, 2, 0]),
         Permutation([1, 0, 2])]

# 定义测试函数，检查如何从基底与强生成器集合生成轨道与横截面
def test_orbits_transversals_from_bsgs():
    # 创建对称群对象，进行 Schreier-Sims 算法初始化
    S = SymmetricGroup(4)
    S.schreier_sims()
    base = S.base
    strong_gens = S.strong_gens
    # 使用 _distribute_gens_by_base 函数分发强生成器到基底
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    # 使用 _orbits_transversals_from_bsgs 函数计算轨道与横截面
    result = _orbits_transversals_from_bsgs(base, strong_gens_distr)
    orbits = result[0]
    transversals = result[1]
    base_len = len(base)
    # 遍历每个基的长度范围
    for i in range(base_len):
        # 遍历当前基的轨道中的每个元素
        for el in orbits[i]:
            # 断言当前基上的转置映射函数应用于元素 el 返回 el 本身
            assert transversals[i][el](base[i]) == el
            # 遍历之前的每个基
            for j in range(i):
                # 断言当前基上的转置映射函数应用于元素 el 返回之前基 j 上的对应元素
                assert transversals[i][el](base[j]) == base[j]
    
    # 计算整体的顺序(order)，即为每个基上轨道数量的乘积
    order = 1
    for i in range(base_len):
        order *= len(orbits[i])
    
    # 断言给定集合 S 的顺序等于之前计算的顺序
    assert S.order() == order
# 测试处理预先计算的波尔斯特朗-汤普森算法（BSGS）的函数
def test_handle_precomputed_bsgs():
    # 创建一个阿尔特五群的对象
    A = AlternatingGroup(5)
    # 进行施雷尔-辛斯算法，准备生成算法
    A.schreier_sims()
    # 获取生成算法的基
    base = A.base
    # 获取强生成元
    strong_gens = A.strong_gens
    # 调用处理预先计算的BSGS算法，得到结果
    result = _handle_precomputed_bsgs(base, strong_gens)
    # 将生成元按基分布的结果
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    # 断言生成元按基分布的结果应与处理预先计算BSGS算法的结果的第三项相等
    assert strong_gens_distr == result[2]
    # 获取转移系数
    transversals = result[0]
    # 获取轨道
    orbits = result[1]
    # 获取基的长度
    base_len = len(base)
    # 对每个基进行遍历
    for i in range(base_len):
        # 对每个轨道中的元素进行遍历
        for el in orbits[i]:
            # 断言转移系数应满足对于每个基的元素el，应有transversals[i][el](base[i]) == el
            assert transversals[i][el](base[i]) == el
            # 对于每个小于i的基元素进行遍历
            for j in range(i):
                # 断言转移系数应满足对于每个基的元素el，应有transversals[i][el](base[j]) == base[j]
                assert transversals[i][el](base[j]) == base[j]
    # 计算群的阶
    order = 1
    for i in range(base_len):
        order *= len(orbits[i])
    # 断言阿尔特五群的阶应与计算得到的群的阶相等
    assert A.order() == order


# 测试基的排序函数
def test_base_ordering():
    # 定义基和度
    base = [2, 4, 5]
    degree = 7
    # 断言基的排序应与预期的结果相等
    assert _base_ordering(base, degree) == [3, 4, 0, 5, 1, 2, 6]


# 测试移除生成元的函数
def test_remove_gens():
    # 创建对称群的对象
    S = SymmetricGroup(10)
    # 进行增量式施雷尔-辛斯算法，获取基和强生成元
    base, strong_gens = S.schreier_sims_incremental()
    # 调用移除生成元的函数，得到新的生成元
    new_gens = _remove_gens(base, strong_gens)
    # 断言新的生成元是否满足验证BSGS的条件
    assert _verify_bsgs(S, base, new_gens) is True
    # 创建阿尔特七群的对象
    A = AlternatingGroup(7)
    # 进行增量式施雷尔-辛斯算法，获取基和强生成元
    base, strong_gens = A.schreier_sims_incremental()
    # 调用移除生成元的函数，得到新的生成元
    new_gens = _remove_gens(base, strong_gens)
    # 断言新的生成元是否满足验证BSGS的条件
    assert _verify_bsgs(A, base, new_gens) is True
    # 创建二面体群的对象
    D = DihedralGroup(2)
    # 进行增量式施雷尔-辛斯算法，获取基和强生成元
    base, strong_gens = D.schreier_sims_incremental()
    # 调用移除生成元的函数，得到新的生成元
    new_gens = _remove_gens(base, strong_gens)
    # 断言新的生成元是否满足验证BSGS的条件
    assert _verify_bsgs(D, base, new_gens) is True
```