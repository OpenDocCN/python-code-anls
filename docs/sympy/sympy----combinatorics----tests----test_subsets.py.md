# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_subsets.py`

```
# 从 sympy.combinatorics.subsets 模块中导入 Subset 和 ksubsets 类
from sympy.combinatorics.subsets import Subset, ksubsets
# 从 sympy.testing.pytest 模块中导入 raises 函数，用于测试异常情况

# 定义测试函数 test_subset，用于测试 Subset 类的各种方法和属性
def test_subset():
    # 创建一个 Subset 对象 a，表示从集合 ['a', 'b', 'c', 'd'] 中选择子集 ['c', 'd']
    a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
    # 断言下一个二进制子集是 ['b']
    assert a.next_binary() == Subset(['b'], ['a', 'b', 'c', 'd'])
    # 断言前一个二进制子集是 ['c']
    assert a.prev_binary() == Subset(['c'], ['a', 'b', 'c', 'd'])
    # 断言下一个字典序子集是 ['d']
    assert a.next_lexicographic() == Subset(['d'], ['a', 'b', 'c', 'd'])
    # 断言前一个字典序子集是 ['c']
    assert a.prev_lexicographic() == Subset(['c'], ['a', 'b', 'c', 'd'])
    # 断言下一个格雷码子集是 ['c']
    assert a.next_gray() == Subset(['c'], ['a', 'b', 'c', 'd'])
    # 断言前一个格雷码子集是 ['d']
    assert a.prev_gray() == Subset(['d'], ['a', 'b', 'c', 'd'])
    # 断言二进制排名是 3
    assert a.rank_binary == 3
    # 断言字典序排名是 14
    assert a.rank_lexicographic == 14
    # 断言格雷码排名是 2
    assert a.rank_gray == 2
    # 断言基数是 16
    assert a.cardinality == 16
    # 断言集合大小是 2
    assert a.size == 2
    # 断言从子集到二进制位列表的转换结果为 '0011'

    # 创建一个 Subset 对象 a，表示从集合 [1, 2, 3, 4, 5, 6, 7] 中选择子集 [2, 5, 7]
    a = Subset([2, 5, 7], [1, 2, 3, 4, 5, 6, 7])
    # 断言下一个二进制子集是 [2, 5, 6]
    assert a.next_binary() == Subset([2, 5, 6], [1, 2, 3, 4, 5, 6, 7])
    # 断言前一个二进制子集是 [2, 5]
    assert a.prev_binary() == Subset([2, 5], [1, 2, 3, 4, 5, 6, 7])
    # 断言下一个字典序子集是 [2, 6]
    assert a.next_lexicographic() == Subset([2, 6], [1, 2, 3, 4, 5, 6, 7])
    # 断言前一个字典序子集是 [2, 5, 6, 7]
    assert a.prev_lexicographic() == Subset([2, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
    # 断言下一个格雷码子集是 [2, 5, 6, 7]
    assert a.next_gray() == Subset([2, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
    # 断言前一个格雷码子集是 [2, 5]
    assert a.prev_gray() == Subset([2, 5], [1, 2, 3, 4, 5, 6, 7])
    # 断言二进制排名是 37
    assert a.rank_binary == 37
    # 断言字典序排名是 93
    assert a.rank_lexicographic == 93
    # 断言格雷码排名是 57
    assert a.rank_gray == 57
    # 断言基数是 128
    assert a.cardinality == 128

    # 创建超集 superset 为 ['a', 'b', 'c', 'd']
    superset = ['a', 'b', 'c', 'd']
    # 断言在 superset 中二进制排名为 4 的子集的二进制排名是 4
    assert Subset.unrank_binary(4, superset).rank_binary == 4
    # 断言在 superset 中格雷码排名为 10 的子集的格雷码排名是 10
    assert Subset.unrank_gray(10, superset).rank_gray == 10

    # 创建超集 superset 为 [1, 2, 3, 4, 5, 6, 7, 8, 9]
    superset = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 断言在 superset 中二进制排名为 33 的子集的二进制排名是 33
    assert Subset.unrank_binary(33, superset).rank_binary == 33
    # 断言在 superset 中格雷码排名为 25 的子集的格雷码排名是 25
    assert Subset.unrank_gray(25, superset).rank_gray == 25

    # 创建一个 Subset 对象 a，表示从空集中选择子集 ['a', 'b', 'c', 'd']
    a = Subset([], ['a', 'b', 'c', 'd'])
    # 初始化计数器 i
    i = 1
    # 循环直到子集变为 ['d']
    while a.subset != Subset(['d'], ['a', 'b', 'c', 'd']).subset:
        # 获取下一个字典序子集
        a = a.next_lexicographic()
        # 计数器加一
        i = i + 1
    # 断言计数器 i 的值为 16
    assert i == 16

    # 初始化计数器 i
    i = 1
    # 循环直到子集变为空集
    while a.subset != Subset([], ['a', 'b', 'c', 'd']).subset:
        # 获取前一个字典序子集
        a = a.prev_lexicographic()
        # 计数器加一
        i = i + 1
    # 断言计数器 i 的值为 16
    assert i == 16

    # 断言当尝试创建不合法的 Subset 对象时会抛出 ValueError 异常
    raises(ValueError, lambda: Subset(['a', 'b'], ['a']))
    raises(ValueError, lambda: Subset(['a'], ['b', 'c']))
    raises(ValueError, lambda: Subset.subset_from_bitlist(['a', 'b'], '010'))

    # 断言两个不同的 Subset 对象相等性测试
    assert Subset(['a'], ['a', 'b']) != Subset(['b'], ['a', 'b'])
    assert Subset(['a'], ['a', 'b']) != Subset(['a'], ['a', 'c'])

# 定义测试函数 test_ksubsets，用于测试 ksubsets 函数
def test_ksubsets():
    # 断言从集合 [1, 2, 3] 中选择 2 个元素的所有子集
    assert list(ksubsets([1, 2, 3], 2)) == [(1, 2), (1, 3), (2, 3)]
    # 断言从集合 [1, 2, 3, 4, 5] 中选择 2 个元素的所有子集
    assert list(ksubsets([1, 2, 3, 4, 5], 2)) == [(1, 2), (1, 3), (1, 4),
                                                 (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
```