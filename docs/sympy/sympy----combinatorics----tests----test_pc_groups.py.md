# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_pc_groups.py`

```
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup, DihedralGroup
from sympy.matrices import Matrix

def test_pc_presentation():
    # 定义要测试的群列表，包括不同的对称群、二面体群等
    Groups = [SymmetricGroup(3), SymmetricGroup(4), SymmetricGroup(9).sylow_subgroup(3),
         SymmetricGroup(9).sylow_subgroup(2), SymmetricGroup(8).sylow_subgroup(2), DihedralGroup(10)]

    # 对 SymmetricGroup(125) 中的 Sylow 5-子群进行处理
    S = SymmetricGroup(125).sylow_subgroup(5)
    # 获取该群的第二个导出级数
    G = S.derived_series()[2]
    Groups.append(G)

    # 对 SymmetricGroup(25) 中的 Sylow 5-子群进行处理
    G = SymmetricGroup(25).sylow_subgroup(5)
    Groups.append(G)

    # 对 SymmetricGroup(11^2) 中的 Sylow 11-子群进行处理
    S = SymmetricGroup(11**2).sylow_subgroup(11)
    # 获取该群的第二个导出级数
    G = S.derived_series()[2]
    Groups.append(G)

    # 遍历所有群
    for G in Groups:
        # 获取 G 的拟周期群
        PcGroup = G.polycyclic_group()
        # 获取 PcGroup 的 collector 对象
        collector = PcGroup.collector
        # 获取 PcGroup 的拟周期表达
        pc_presentation = collector.pc_presentation

        # 获取 PcGroup 的生成器集合
        pcgs = PcGroup.pcgs
        # 获取 collector 的自由群对象
        free_group = collector.free_group
        # 构建自由群到置换的映射
        free_to_perm = {}
        for s, g in zip(free_group.symbols, pcgs):
            free_to_perm[s] = g

        # 遍历拟周期表达的键值对
        for k, v in pc_presentation.items():
            # 获取 k 的数组形式
            k_array = k.array_form
            # 如果 v 非空
            if v != ():
                # 获取 v 的数组形式
                v_array = v.array_form

            # 初始化左手置换
            lhs = Permutation()
            # 根据 k_array 构建 lhs 置换
            for gen in k_array:
                s = gen[0]
                e = gen[1]
                lhs = lhs * free_to_perm[s] ** e

            # 如果 v 是空元组
            if v == ():
                # 断言 lhs 是恒等置换
                assert lhs.is_identity
                continue

            # 初始化右手置换
            rhs = Permutation()
            # 根据 v_array 构建 rhs 置换
            for gen in v_array:
                s = gen[0]
                e = gen[1]
                rhs = rhs * free_to_perm[s] ** e

            # 断言 lhs 等于 rhs
            assert lhs == rhs


def test_exponent_vector():
    # 定义要测试的群列表，包括不同的对称群、二面体群等
    Groups = [SymmetricGroup(3), SymmetricGroup(4), SymmetricGroup(9).sylow_subgroup(3),
         SymmetricGroup(9).sylow_subgroup(2), SymmetricGroup(8).sylow_subgroup(2)]

    # 遍历所有群
    for G in Groups:
        # 获取 G 的拟周期群
        PcGroup = G.polycyclic_group()
        # 获取 PcGroup 的 collector 对象
        collector = PcGroup.collector

        # 获取 PcGroup 的生成器集合
        pcgs = PcGroup.pcgs
        # 遍历 G 的生成器
        for gen in G.generators:
            # 获取 gen 的指数向量
            exp = collector.exponent_vector(gen)
            # 初始化置换 g
            g = Permutation()
            # 根据指数向量构建置换 g
            for i in range(len(exp)):
                g = g * pcgs[i] ** exp[i] if exp[i] else g
            # 断言 g 等于 gen
            assert g == gen


def test_induced_pcgs():
    # 定义要测试的群列表，包括不同的对称群、二面体群、交错群等
    G = [SymmetricGroup(9).sylow_subgroup(3), SymmetricGroup(20).sylow_subgroup(2), AlternatingGroup(4),
    DihedralGroup(4), DihedralGroup(10), DihedralGroup(9), SymmetricGroup(3), SymmetricGroup(4)]

    # 遍历所有群
    for g in G:
        # 获取 g 的拟周期群
        PcGroup = g.polycyclic_group()
        # 获取 PcGroup 的 collector 对象
        collector = PcGroup.collector
        # 获取 g 的生成器列表
        gens = list(g.generators)
        # 获取 g 诱导的拟周期生成子集合
        ipcgs = collector.induced_pcgs(gens)
        # 初始化指数矩阵 m
        m = []
        # 构建每个拟周期生成子的指数向量并添加到 m 中
        for i in ipcgs:
            m.append(collector.exponent_vector(i))
        # 断言矩阵 m 是上三角矩阵
        assert Matrix(m).is_upper
```