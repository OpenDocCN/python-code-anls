# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_partitions.py`

```
# 导入所需模块和函数
from sympy.core.sorting import ordered, default_sort_key
from sympy.combinatorics.partitions import (Partition, IntegerPartition,
                                            RGS_enum, RGS_unrank, RGS_rank,
                                            random_integer_partition)
from sympy.testing.pytest import raises
from sympy.utilities.iterables import partitions
from sympy.sets.sets import Set, FiniteSet

# 定义测试函数，检查 Partition 类的构造函数
def test_partition_constructor():
    # 检查不合法的参数输入
    raises(ValueError, lambda: Partition([1, 1, 2]))
    raises(ValueError, lambda: Partition([1, 2, 3], [2, 3, 4]))
    raises(ValueError, lambda: Partition(1, 2, 3))
    raises(ValueError, lambda: Partition(*list(range(3))))

    # 检查 Partition 对象相等性
    assert Partition([1, 2, 3], [4, 5]) == Partition([4, 5], [1, 2, 3])
    assert Partition({1, 2, 3}, {4, 5}) == Partition([1, 2, 3], [4, 5])

    # 使用 FiniteSet 创建 Partition 对象，并进行比较
    a = FiniteSet(1, 2, 3)
    b = FiniteSet(4, 5)
    assert Partition(a, b) == Partition([1, 2, 3], [4, 5])
    assert Partition({a, b}) == Partition(FiniteSet(a, b))
    assert Partition({a, b}) != Partition(a, b)

# 定义测试函数，检查 Partition 类的功能
def test_partition():
    # 导入符号 x
    from sympy.abc import x

    # 创建几个 Partition 对象并排序
    a = Partition([1, 2, 3], [4])
    b = Partition([1, 2], [3, 4])
    c = Partition([x])
    l = [a, b, c]
    l.sort(key=default_sort_key)
    assert l == [c, a, b]
    l.sort(key=lambda w: default_sort_key(w, order='rev-lex'))
    assert l == [c, a, b]

    # 检查 Partition 对象的比较运算
    assert (a == b) is False
    assert a <= b
    assert (a > b) is False
    assert a != b
    assert a < b

    # 检查 Partition 对象的算术运算
    assert (a + 2).partition == [[1, 2], [3, 4]]
    assert (b - 1).partition == [[1, 2, 4], [3]]
    assert (a - 1).partition == [[1, 2, 3, 4]]
    assert (a + 1).partition == [[1, 2, 4], [3]]
    assert (b + 1).partition == [[1, 2], [3], [4]]

    # 检查 Partition 对象的属性
    assert a.rank == 1
    assert b.rank == 3
    assert a.RGS == (0, 0, 0, 1)
    assert b.RGS == (0, 0, 1, 1)

# 定义测试函数，检查 IntegerPartition 类的功能
def test_integer_partition():
    # 检查 IntegerPartition 构造函数对参数的限制
    raises(ValueError, lambda: IntegerPartition(list(range(3))))
    raises(ValueError, lambda: IntegerPartition(100, list(range(1, 3))))

    # 创建几个 IntegerPartition 对象，并进行比较
    a = IntegerPartition(8, [1, 3, 4])
    b = a.next_lex()
    c = IntegerPartition([1, 3, 4])
    d = IntegerPartition(8, {1: 3, 3: 1, 2: 1})
    assert a == c
    assert a.integer == d.integer
    assert a.conjugate == [3, 2, 2, 1]
    assert (a == b) is False
    assert a <= b
    assert (a > b) is False
    assert a != b

    # 遍历整数的分割，并检查 lex 下一个和前一个分割
    for i in range(1, 11):
        next = set()
        prev = set()
        a = IntegerPartition([i])
        ans = {IntegerPartition(p) for p in partitions(i)}
        n = len(ans)
        for j in range(n):
            next.add(a)
            a = a.next_lex()
            IntegerPartition(i, a.partition)  # check it by giving i
        for j in range(n):
            prev.add(a)
            a = a.prev_lex()
            IntegerPartition(i, a.partition)  # check it by giving i
        assert next == ans
        assert prev == ans

    # 检查 IntegerPartition 对象的 as_ferrers 方法
    assert IntegerPartition([1, 2, 3]).as_ferrers() == '###\n##\n#'
    # 断言：验证 IntegerPartition([1, 1, 3]) 的 as_ferrers('o') 方法返回值是否为 'ooo\no\no'
    assert IntegerPartition([1, 1, 3]).as_ferrers('o') == 'ooo\no\no'
    
    # 断言：验证 IntegerPartition([1, 1, 3]) 转换为字符串的结果是否为 '[3, 1, 1]'
    assert str(IntegerPartition([1, 1, 3])) == '[3, 1, 1]'
    
    # 断言：验证 IntegerPartition([1, 1, 3]) 的 partition 属性是否为 [3, 1, 1]
    assert IntegerPartition([1, 1, 3]).partition == [3, 1, 1]
    
    # 断言：验证调用 random_integer_partition(-1) 时是否会抛出 ValueError 异常
    raises(ValueError, lambda: random_integer_partition(-1))
    
    # 断言：验证调用 random_integer_partition(1) 返回值是否为 [1]
    assert random_integer_partition(1) == [1]
    
    # 断言：验证使用 seed=[1, 3, 2, 1, 5, 1] 调用 random_integer_partition(10) 的返回值是否为 [5, 2, 1, 1, 1]
    assert random_integer_partition(10, seed=[1, 3, 2, 1, 5, 1]) == [5, 2, 1, 1, 1]
# 定义测试函数 test_rgs，用于测试 RGS_unrank 和 Partition 类的相关功能
def test_rgs():
    # 检查 RGS_unrank 函数对于无效参数（负数）是否引发 ValueError 异常
    raises(ValueError, lambda: RGS_unrank(-1, 3))
    # 检查 RGS_unrank 函数对于无效参数（宽度为零）是否引发 ValueError 异常
    raises(ValueError, lambda: RGS_unrank(3, 0))
    # 检查 RGS_unrank 函数对于无效参数（长度大于宽度）是否引发 ValueError 异常
    raises(ValueError, lambda: RGS_unrank(10, 1))

    # 检查 Partition.from_rgs 函数对于不合规的 RGS 序列是否引发 ValueError 异常
    raises(ValueError, lambda: Partition.from_rgs(list(range(3)), list(range(2))))
    # 检查 Partition.from_rgs 函数对于不合规的 RGS 序列是否引发 ValueError 异常
    raises(ValueError, lambda: Partition.from_rgs(list(range(1, 3)), list(range(2))))
    
    # 断言 RGS_enum 函数对于负数的枚举结果应为 0
    assert RGS_enum(-1) == 0
    # 断言 RGS_enum 函数对于长度为 1 的枚举结果应为 1
    assert RGS_enum(1) == 1
    
    # 断言 RGS_unrank 函数对于特定参数的结果是否符合预期
    assert RGS_unrank(7, 5) == [0, 0, 1, 0, 2]
    assert RGS_unrank(23, 14) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2]
    
    # 断言 RGS_rank 函数对于特定 RGS 序列的排名结果是否符合预期
    assert RGS_rank(RGS_unrank(40, 100)) == 40

# 定义测试函数 test_ordered_partition_9608，用于测试 ordered 函数对 Partition 实例的排序功能
def test_ordered_partition_9608():
    # 创建两个 Partition 实例 a 和 b
    a = Partition([1, 2, 3], [4])
    b = Partition([1, 2], [3, 4])
    # 断言对给定的 Partition 实例列表进行 ordered 函数排序的结果是否符合预期
    assert list(ordered([a, b], Set._infimum_key))
```