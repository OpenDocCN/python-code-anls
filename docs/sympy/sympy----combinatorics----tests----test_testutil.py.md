# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_testutil.py`

```
# 从 sympy.combinatorics.named_groups 模块导入 SymmetricGroup、AlternatingGroup 和 CyclicGroup 类
# 从 sympy.combinatorics.testutil 模块导入一些测试工具函数
# 从 sympy.combinatorics.permutations 模块导入 Permutation 类
# 从 sympy.combinatorics.perm_groups 模块导入 PermutationGroup 类
# 从 sympy.core.random 模块导入 shuffle 函数
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup,\
    CyclicGroup
from sympy.combinatorics.testutil import _verify_bsgs, _cmp_perm_lists,\
    _naive_list_centralizer, _verify_centralizer,\
    _verify_normal_closure
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.random import shuffle


def test_cmp_perm_lists():
    # 创建 SymmetricGroup(4) 对象 S，生成其所有的四阶置换元素，并转换为列表 els
    S = SymmetricGroup(4)
    els = list(S.generate_dimino())
    # 复制 els 列表到 other
    other = els[:]
    # 打乱 other 列表中元素的顺序
    shuffle(other)
    # 断言调用 _cmp_perm_lists 函数，比较 els 和 other 列表是否相同
    assert _cmp_perm_lists(els, other) is True


def test_naive_list_centralizer():
    # 创建 SymmetricGroup(3) 和 AlternatingGroup(3) 对象 S 和 A
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    # 断言调用 _naive_list_centralizer 函数，验证 S 和 S 的中心化子是否为 [Permutation([0, 1, 2])]
    assert _naive_list_centralizer(S, S) == [Permutation([0, 1, 2])]
    # 断言调用 PermutationGroup 构造函数，创建中心化子为 _naive_list_centralizer(S, A) 的群，并检验是否为 A 的子群
    assert PermutationGroup(_naive_list_centralizer(S, A)).is_subgroup(A)


def test_verify_bsgs():
    # 创建 SymmetricGroup(5) 对象 S，并调用其 schreier_sims 方法
    S = SymmetricGroup(5)
    S.schreier_sims()
    # 获取生成的基 base 和强生成器 strong_gens
    base = S.base
    strong_gens = S.strong_gens
    # 断言调用 _verify_bsgs 函数，验证 BSGS（基与强生成器系统）是否有效
    assert _verify_bsgs(S, base, strong_gens) is True
    # 断言调用 _verify_bsgs 函数，验证不同的 base 是否导致 BSGS 无效
    assert _verify_bsgs(S, base[:-1], strong_gens) is False
    # 断言调用 _verify_bsgs 函数，验证不同的生成器列表是否导致 BSGS 无效
    assert _verify_bsgs(S, base, S.generators) is False


def test_verify_centralizer():
    # 创建 SymmetricGroup(3) 和 AlternatingGroup(3) 对象 S 和 A
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    # 创建仅包含恒等置换的 PermutationGroup 对象 triv
    triv = PermutationGroup([Permutation([0, 1, 2])])
    # 断言调用 _verify_centralizer 函数，验证 S 和 S 的中心化子是否为 triv
    assert _verify_centralizer(S, S, centr=triv)
    # 断言调用 _verify_centralizer 函数，验证 S 和 A 的中心化子是否为 A 自身
    assert _verify_centralizer(S, A, centr=A)


def test_verify_normal_closure():
    # 创建 SymmetricGroup(3) 和 AlternatingGroup(3) 对象 S 和 A
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    # 断言调用 _verify_normal_closure 函数，验证 S 的正规闭包是否为 A
    assert _verify_normal_closure(S, A, closure=A)
    # 创建 SymmetricGroup(5)、AlternatingGroup(5) 和 CyclicGroup(5) 对象 S、A 和 C
    S = SymmetricGroup(5)
    A = AlternatingGroup(5)
    C = CyclicGroup(5)
    # 断言调用 _verify_normal_closure 函数，验证 S 的正规闭包是否为 A
    assert _verify_normal_closure(S, A, closure=A)
    # 断言调用 _verify_normal_closure 函数，验证 S 的正规闭包是否为 A
    assert _verify_normal_closure(S, C, closure=A)
```