# `D:\src\scipysrc\sympy\sympy\sets\tests\test_powerset.py`

```
# 从 sympy.core.expr 模块导入 unchanged 函数
from sympy.core.expr import unchanged
# 从 sympy.core.singleton 模块导入 S 单例
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Symbol 符号类
from sympy.core.symbol import Symbol
# 从 sympy.sets.contains 模块导入 Contains 类
from sympy.sets.contains import Contains
# 从 sympy.sets.fancysets 模块导入 Interval 类
from sympy.sets.fancysets import Interval
# 从 sympy.sets.powerset 模块导入 PowerSet 类
from sympy.sets.powerset import PowerSet
# 从 sympy.sets.sets 模块导入 FiniteSet 类
from sympy.sets.sets import FiniteSet
# 从 sympy.testing.pytest 模块导入 raises 和 XFAIL 函数
from sympy.testing.pytest import raises, XFAIL


# 定义测试函数，验证 PowerSet 的创建不改变指定的 FiniteSet 对象
def test_powerset_creation():
    # 断言 PowerSet(FiniteSet(1, 2)) 不会改变
    assert unchanged(PowerSet, FiniteSet(1, 2))
    # 断言 PowerSet(S.EmptySet) 不会改变
    assert unchanged(PowerSet, S.EmptySet)
    # 断言创建 PowerSet(123) 时会引发 ValueError 异常
    raises(ValueError, lambda: PowerSet(123))
    # 断言 PowerSet(S.Reals) 不会改变
    assert unchanged(PowerSet, S.Reals)
    # 断言 PowerSet(S.Integers) 不会改变
    assert unchanged(PowerSet, S.Integers)


# 定义测试函数，验证 PowerSet 对 FiniteSet 的重写操作
def test_powerset_rewrite_FiniteSet():
    # 断言对 FiniteSet(1, 2) 使用 FiniteSet 重写后结果符合预期
    assert PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet) == \
        FiniteSet(S.EmptySet, FiniteSet(1), FiniteSet(2), FiniteSet(1, 2))
    # 断言对 S.EmptySet 使用 FiniteSet 重写后结果符合预期
    assert PowerSet(S.EmptySet).rewrite(FiniteSet) == FiniteSet(S.EmptySet)
    # 断言对 S.Naturals 使用 FiniteSet 重写后结果符合预期
    assert PowerSet(S.Naturals).rewrite(FiniteSet) == PowerSet(S.Naturals)


# 定义测试函数，验证 FiniteSet 对 PowerSet 的重写操作
def test_finiteset_rewrite_powerset():
    # 断言对 FiniteSet(S.EmptySet) 使用 PowerSet 重写后结果符合预期
    assert FiniteSet(S.EmptySet).rewrite(PowerSet) == PowerSet(S.EmptySet)
    # 断言对 FiniteSet(S.EmptySet, FiniteSet(1), FiniteSet(2), FiniteSet(1, 2)) 使用 PowerSet 重写后结果符合预期
    assert FiniteSet(
        S.EmptySet, FiniteSet(1),
        FiniteSet(2), FiniteSet(1, 2)).rewrite(PowerSet) == \
            PowerSet(FiniteSet(1, 2))
    # 断言对 FiniteSet(1, 2, 3) 使用 PowerSet 重写后结果符合预期
    assert FiniteSet(1, 2, 3).rewrite(PowerSet) == FiniteSet(1, 2, 3)


# 定义测试函数，验证 PowerSet 的 __contains__ 方法
def test_powerset__contains__():
    # 定义包含各种子集的列表 subset_series
    subset_series = [
        S.EmptySet,
        FiniteSet(1, 2),
        S.Naturals,
        S.Naturals0,
        S.Integers,
        S.Rationals,
        S.Reals,
        S.Complexes]

    # 获取 subset_series 的长度
    l = len(subset_series)
    # 嵌套循环遍历 subset_series
    for i in range(l):
        for j in range(l):
            # 如果 i <= j，则断言 subset_series[i] 在 PowerSet(subset_series[j], evaluate=False) 中
            if i <= j:
                assert subset_series[i] in \
                    PowerSet(subset_series[j], evaluate=False)
            # 否则断言 subset_series[i] 不在 PowerSet(subset_series[j], evaluate=False) 中
            else:
                assert subset_series[i] not in \
                    PowerSet(subset_series[j], evaluate=False)


# 定义预期失败的测试函数，验证失败的 PowerSet 的 __contains__ 方法
@XFAIL
def test_failing_powerset__contains__():
    # 以下断言在 evaluate=True 时会失败，但在未评估的 PowerSet 下可以正常工作
    assert FiniteSet(1, 2) not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Naturals0 not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals0 not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Integers not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Integers not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Rationals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Rationals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Reals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Reals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Complexes not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Complexes not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)


# 定义测试函数，验证 PowerSet 的 __len__ 方法
def test_powerset__len__():
    # 创建不评估的 PowerSet(S.EmptySet) 对象 A
    A = PowerSet(S.EmptySet, evaluate=False)
    # 断言 A 的长度为 1
    assert len(A) == 1
    # 调用名为 PowerSet 的函数来生成集合 A 的幂集，但不进行求值
    A = PowerSet(A, evaluate=False)
    # 断言集合 A 的长度为 2
    assert len(A) == 2
    # 再次调用 PowerSet 函数生成集合 A 的幂集，但不进行求值
    A = PowerSet(A, evaluate=False)
    # 断言集合 A 的长度为 4
    assert len(A) == 4
    # 第三次调用 PowerSet 函数生成集合 A 的幂集，但不进行求值
    A = PowerSet(A, evaluate=False)
    # 断言集合 A 的长度为 16
    assert len(A) == 16
# 定义测试函数，用于测试 PowerSet 类的 __iter__ 方法
def test_powerset__iter__():
    # 创建 PowerSet 对象，包含 {1, 2} 的所有子集，并获取其迭代器
    a = PowerSet(FiniteSet(1, 2)).__iter__()
    # 断言下一个子集是空集
    assert next(a) == S.EmptySet
    # 断言下一个子集是 {1}
    assert next(a) == FiniteSet(1)
    # 断言下一个子集是 {2}
    assert next(a) == FiniteSet(2)
    # 断言下一个子集是 {1, 2}
    assert next(a) == FiniteSet(1, 2)

    # 创建 PowerSet 对象，包含自然数集合的所有子集，并获取其迭代器
    a = PowerSet(S.Naturals).__iter__()
    # 断言下一个子集是空集
    assert next(a) == S.EmptySet
    # 断言下一个子集是 {1}
    assert next(a) == FiniteSet(1)
    # 断言下一个子集是 {2}
    assert next(a) == FiniteSet(2)
    # 断言下一个子集是 {1, 2}
    assert next(a) == FiniteSet(1, 2)
    # 断言下一个子集是 {3}
    assert next(a) == FiniteSet(3)
    # 断言下一个子集是 {1, 3}
    assert next(a) == FiniteSet(1, 3)
    # 断言下一个子集是 {2, 3}
    assert next(a) == FiniteSet(2, 3)
    # 断言下一个子集是 {1, 2, 3}
    assert next(a) == FiniteSet(1, 2, 3)


# 定义测试函数，用于测试 PowerSet 类的 contains 方法
def test_powerset_contains():
    # 创建 PowerSet 对象，包含 {1} 的所有子集，不进行求值
    A = PowerSet(FiniteSet(1), evaluate=False)
    # 断言 PowerSet 对象是否包含元素 2
    assert A.contains(2) == Contains(2, A)

    # 创建符号变量 x
    x = Symbol('x')

    # 创建 PowerSet 对象，包含 {x} 的所有子集，不进行求值
    A = PowerSet(FiniteSet(x), evaluate=False)
    # 断言 PowerSet 对象是否包含集合 {1}
    assert A.contains(FiniteSet(1)) == Contains(FiniteSet(1), A)


# 定义测试函数，用于测试有限集合的 powerset 方法
def test_powerset_method():
    # 创建空集合 A
    A = FiniteSet()
    # 计算 A 的幂集
    pset = A.powerset()
    # 断言幂集的长度为 1
    assert len(pset) == 1
    # 断言幂集只包含空集
    assert pset == FiniteSet(S.EmptySet)

    # 创建集合 A 包含 {1, 2}
    A = FiniteSet(1, 2)
    # 计算 A 的幂集
    pset = A.powerset()
    # 断言幂集的长度为 2^|A|
    assert len(pset) == 2**len(A)
    # 断言幂集包含的元素为 { {}, {1}, {2}, {1, 2} }
    assert pset == FiniteSet(FiniteSet(), FiniteSet(1),
                             FiniteSet(2), A)

    # 创建区间 A 包含 [0, 1]
    A = Interval(0, 1)
    # 断言 A 的幂集等于其幂集对象
    assert A.powerset() == PowerSet(A)


# 定义测试函数，用于测试 PowerSet 类的 is_subset 方法
def test_is_subset():
    # 创建 PowerSet 对象，包含 {1} 的所有子集
    subset = PowerSet(FiniteSet(1))
    # 创建 PowerSet 对象，包含 {1, 2} 的所有子集
    pset = PowerSet(FiniteSet(1, 2))
    # 创建 PowerSet 对象，包含 {2, 3} 的所有子集
    bad_set = PowerSet(FiniteSet(2, 3))
    # 断言 subset 是否是 pset 的子集，预期为 True
    assert subset.is_subset(pset)
    # 断言 bad_set 是否是 pset 的子集，预期为 False
    assert not pset.is_subset(bad_set)
```