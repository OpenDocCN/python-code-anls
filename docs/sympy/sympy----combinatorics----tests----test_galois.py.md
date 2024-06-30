# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_galois.py`

```
"""Test groups defined by the galois module. """

# 导入需要的模块和类
from sympy.combinatorics.galois import (
    S4TransitiveSubgroups, S5TransitiveSubgroups, S6TransitiveSubgroups,
    find_transitive_subgroups_of_S6,
)
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import (
    SymmetricGroup, AlternatingGroup, CyclicGroup,
)


# 测试 S4 的传递子群
def test_four_group():
    G = S4TransitiveSubgroups.V.get_perm_group()
    A4 = AlternatingGroup(4)
    assert G.is_subgroup(A4)           # 断言 G 是 A4 的子群
    assert G.degree == 4                # 断言 G 的次数为 4
    assert G.is_transitive()            # 断言 G 是传递的
    assert G.order() == 4               # 断言 G 的阶数为 4
    assert not G.is_cyclic             # 断言 G 不是循环群


# 测试 M20 的子群
def test_M20():
    G = S5TransitiveSubgroups.M20.get_perm_group()
    S5 = SymmetricGroup(5)
    A5 = AlternatingGroup(5)
    assert G.is_subgroup(S5)            # 断言 G 是 S5 的子群
    assert not G.is_subgroup(A5)        # 断言 G 不是 A5 的子群
    assert G.degree == 5                # 断言 G 的次数为 5
    assert G.is_transitive()            # 断言 G 是传递的
    assert G.order() == 20              # 断言 G 的阶数为 20


# 设置此值为 True 表示对 S6 的每个传递子群运行测试
INCLUDE_SEARCH_REPS = False
S6_randomized = {}

# 如果 INCLUDE_SEARCH_REPS 为 True，则从 S6TransitiveSubgroups 中获取所有传递子群的测试表示
if INCLUDE_SEARCH_REPS:
    S6_randomized = find_transitive_subgroups_of_S6(*list(S6TransitiveSubgroups))


# 获取给定名称的 S6 子群的不同版本
def get_versions_of_S6_subgroup(name):
    vers = [name.get_perm_group()]
    if INCLUDE_SEARCH_REPS:
        vers.append(S6_randomized[name])
    return vers


# 测试 S6 的传递子群
def test_S6_transitive_subgroups():
    """
    Test enough characteristics to distinguish all 16 transitive subgroups.
    """
    ts = S6TransitiveSubgroups
    A6 = AlternatingGroup(6)

    # 遍历每个子群的特征，并进行相应的断言
    for name, alt, order, is_isom, not_isom in [
        (ts.C6,     False,    6, CyclicGroup(6), None),
        (ts.S3,     False,    6, SymmetricGroup(3), None),
        (ts.D6,     False,   12, None, None),
        (ts.A4,     True,    12, None, None),
        (ts.G18,    False,   18, None, None),
        (ts.A4xC2,  False,   24, None, SymmetricGroup(4)),
        (ts.S4m,    False,   24, SymmetricGroup(4), None),
        (ts.S4p,    True,    24, None, None),
        (ts.G36m,   False,   36, None, None),
        (ts.G36p,   True,    36, None, None),
        (ts.S4xC2,  False,   48, None, None),
        (ts.PSL2F5, True,    60, None, None),
        (ts.G72,    False,   72, None, None),
        (ts.PGL2F5, False,  120, None, None),
        (ts.A6,     True,   360, None, None),
        (ts.S6,     False,  720, None, None),
    ]:
        for G in get_versions_of_S6_subgroup(name):
            assert G.is_transitive()         # 断言 G 是传递的
            assert G.degree == 6              # 断言 G 的次数为 6
            assert G.is_subgroup(A6) is alt   # 断言 G 是否是 A6 的子群，根据 alt 的值进行判断
            assert G.order() == order         # 断言 G 的阶数为给定的 order
            if is_isom:
                assert is_isomorphic(G, is_isom)     # 断言 G 与 is_isom 同构
            if not_isom:
                assert not is_isomorphic(G, not_isom)  # 断言 G 与 not_isom 不同构
```