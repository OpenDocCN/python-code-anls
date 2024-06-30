# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_named_groups.py`

```
# 导入 sympy 库中的具名群组模块
from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup,
                                              DihedralGroup, AlternatingGroup,
                                              AbelianGroup, RubikGroup)
# 导入 pytest 中的 raises 函数，用于测试异常情况
from sympy.testing.pytest import raises


# 测试 SymmetricGroup 类
def test_SymmetricGroup():
    # 创建一个 SymmetricGroup 对象 G，阶数为 5
    G = SymmetricGroup(5)
    # 生成群 G 的所有元素并转换为列表
    elements = list(G.generate())
    # 断言：G 的第一个生成元素的阶数为 5
    assert (G.generators[0]).size == 5
    # 断言：群 G 的元素个数为 120
    assert len(elements) == 120
    # 断言：群 G 不可解
    assert G.is_solvable is False
    # 断言：群 G 不可交换
    assert G.is_abelian is False
    # 断言：群 G 不可幂零
    assert G.is_nilpotent is False
    # 断言：群 G 是传递群
    assert G.is_transitive() is True
    # 创建一个阶数为 1 的 SymmetricGroup 对象 H
    H = SymmetricGroup(1)
    # 断言：群 H 的阶数为 1
    assert H.order() == 1
    # 创建一个阶数为 2 的 SymmetricGroup 对象 L
    L = SymmetricGroup(2)
    # 断言：群 L 的阶数为 2
    assert L.order() == 2


# 测试 CyclicGroup 类
def test_CyclicGroup():
    # 创建一个 CyclicGroup 对象 G，阶数为 10
    G = CyclicGroup(10)
    # 生成群 G 的所有元素并转换为列表
    elements = list(G.generate())
    # 断言：群 G 的元素个数为 10
    assert len(elements) == 10
    # 断言：G 的导出子群的阶数为 1
    assert (G.derived_subgroup()).order() == 1
    # 断言：群 G 可交换
    assert G.is_abelian is True
    # 断言：群 G 可解
    assert G.is_solvable is True
    # 断言：群 G 可幂零
    assert G.is_nilpotent is True
    # 创建一个阶数为 1 的 CyclicGroup 对象 H
    H = CyclicGroup(1)
    # 断言：群 H 的阶数为 1
    assert H.order() == 1
    # 创建一个阶数为 2 的 CyclicGroup 对象 L
    L = CyclicGroup(2)
    # 断言：群 L 的阶数为 2
    assert L.order() == 2


# 测试 DihedralGroup 类
def test_DihedralGroup():
    # 创建一个 DihedralGroup 对象 G，阶数为 6
    G = DihedralGroup(6)
    # 生成群 G 的所有元素并转换为列表
    elements = list(G.generate())
    # 断言：群 G 的元素个数为 12
    assert len(elements) == 12
    # 断言：群 G 是传递群
    assert G.is_transitive() is True
    # 断言：群 G 不可交换
    assert G.is_abelian is False
    # 断言：群 G 可解
    assert G.is_solvable is True
    # 断言：群 G 不可幂零
    assert G.is_nilpotent is False
    # 创建一个阶数为 1 的 DihedralGroup 对象 H
    H = DihedralGroup(1)
    # 断言：群 H 的阶数为 2
    assert H.order() == 2
    # 创建一个阶数为 2 的 DihedralGroup 对象 L
    L = DihedralGroup(2)
    # 断言：群 L 的阶数为 4
    assert L.order() == 4
    # 断言：群 L 可交换
    assert L.is_abelian is True
    # 断言：群 L 可幂零
    assert L.is_nilpotent is True


# 测试 AlternatingGroup 类
def test_AlternatingGroup():
    # 创建一个 AlternatingGroup 对象 G，阶数为 5
    G = AlternatingGroup(5)
    # 生成群 G 的所有元素并转换为列表
    elements = list(G.generate())
    # 断言：群 G 的元素个数为 60
    assert len(elements) == 60
    # 断言：群 G 中所有置换都是偶置换
    assert [perm.is_even for perm in elements] == [True]*60
    # 创建一个阶数为 1 的 AlternatingGroup 对象 H
    H = AlternatingGroup(1)
    # 断言：群 H 的阶数为 1
    assert H.order() == 1
    # 创建一个阶数为 2 的 AlternatingGroup 对象 L
    L = AlternatingGroup(2)
    # 断言：群 L 的阶数为 1
    assert L.order() == 1


# 测试 AbelianGroup 类
def test_AbelianGroup():
    # 创建一个阶数为 3, 3, 3 的 AbelianGroup 对象 A
    A = AbelianGroup(3, 3, 3)
    # 断言：群 A 的阶数为 27
    assert A.order() == 27
    # 断言：群 A 可交换
    assert A.is_abelian is True


# 测试 RubikGroup 类
def test_RubikGroup():
    # 使用 lambda 表达式测试创建 RubikGroup(1) 时抛出 ValueError 异常
    raises(ValueError, lambda: RubikGroup(1))
```