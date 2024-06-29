# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_setops.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 测试框架，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 数据分析库
from pandas import (  # 从 Pandas 中导入以下模块和类
    CategoricalIndex,
    DataFrame,
    Index,
    IntervalIndex,
    MultiIndex,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块，用于测试辅助函数
from pandas.api.types import (  # 从 Pandas API 中导入以下数据类型判断函数
    is_float_dtype,
    is_unsigned_integer_dtype,
)


@pytest.mark.parametrize("case", [0.5, "xxx"])  # 参数化测试，参数为 0.5 和 "xxx"
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_set_ops_error_cases(idx, case, sort, method):
    # 非可迭代输入的情况下
    msg = "Input must be Index or array-like"
    with pytest.raises(TypeError, match=msg):  # 使用 Pytest 检查是否抛出预期的 TypeError 异常，匹配异常消息
        getattr(idx, method)(case, sort=sort)


@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])  # 参数化测试，测试不同的数据类型
def test_intersection_base(idx, sort, klass):
    first = idx[2::-1]  # 取索引中前三个元素并反转顺序
    second = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)  # 如果 klass 不是 MultiIndex 类型，则将 second 转换为 klass 类型

    intersect = first.intersection(second, sort=sort)  # 计算两个索引的交集
    if sort is None:
        expected = first.sort_values()  # 如果 sort 为 None，则对 first 进行排序
    else:
        expected = first
    tm.assert_index_equal(intersect, expected)  # 使用 Pandas 测试工具比较预期的交集与计算结果是否相等

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):  # 使用 Pytest 检查是否抛出预期的 TypeError 异常，匹配异常消息
        first.intersection([1, 2, 3], sort=sort)


@pytest.mark.arm_slow  # 标记测试为较慢的测试
@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])  # 参数化测试，测试不同的数据类型
def test_union_base(idx, sort, klass):
    first = idx[::-1]  # 反转索引顺序
    second = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)  # 如果 klass 不是 MultiIndex 类型，则将 second 转换为 klass 类型

    union = first.union(second, sort=sort)  # 计算两个索引的并集
    if sort is None:
        expected = first.sort_values()  # 如果 sort 为 None，则对 first 进行排序
    else:
        expected = first
    tm.assert_index_equal(union, expected)  # 使用 Pandas 测试工具比较预期的并集与计算结果是否相等

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):  # 使用 Pytest 检查是否抛出预期的 TypeError 异常，匹配异常消息
        first.union([1, 2, 3], sort=sort)


def test_difference_base(idx, sort):
    second = idx[4:]  # 取索引中从第四个元素开始到末尾的部分
    answer = idx[:4]  # 取索引中前四个元素
    result = idx.difference(second, sort=sort)  # 计算两个索引的差集

    if sort is None:
        answer = answer.sort_values()  # 如果 sort 为 None，则对 answer 进行排序

    assert result.equals(answer)  # 断言计算结果与预期结果是否相等
    tm.assert_index_equal(result, answer)  # 使用 Pandas 测试工具比较预期的差集与计算结果是否相等

    # GH 10149
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = idx.difference(case, sort=sort)  # 计算索引与不同数据类型的差集
        tm.assert_index_equal(result, answer)  # 使用 Pandas 测试工具比较预期的差集与计算结果是否相等

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):  # 使用 Pytest 检查是否抛出预期的 TypeError 异常，匹配异常消息
        idx.difference([1, 2, 3], sort=sort)


def test_symmetric_difference(idx, sort):
    first = idx[1:]  # 取索引中从第二个元素开始到末尾的部分
    second = idx[:-1]  # 取索引中从开始到倒数第二个元素的部分
    answer = idx[[-1, 0]]  # 选择索引中的特定元素
    result = first.symmetric_difference(second, sort=sort)  # 计算两个索引的对称差集

    if sort is None:
        answer = answer.sort_values()  # 如果 sort 为 None，则对 answer 进行排序

    tm.assert_index_equal(result, answer)  # 使用 Pandas 测试工具比较预期的对称差集与计算结果是否相等

    # GH 10149
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = first.symmetric_difference(case, sort=sort)  # 计算索引与不同数据类型的对称差集
        tm.assert_index_equal(result, answer)  # 使用 Pandas 测试工具比较预期的对称差集与计算结果是否相等
    # 设置错误消息，用于断言异常抛出时的匹配
    msg = "other must be a MultiIndex or a list of tuples"
    # 使用 pytest.raises 断言捕获 TypeError 异常，并验证其错误消息与设定的 msg 是否匹配
    with pytest.raises(TypeError, match=msg):
        # 调用 first 对象的 symmetric_difference 方法，传入参数 [1, 2, 3]，并指定 sort=sort 参数
        first.symmetric_difference([1, 2, 3], sort=sort)
# 定义测试函数，用于测试多重索引的对称差集操作
def test_multiindex_symmetric_difference():
    # GH 13490：GitHub issue编号，用于跟踪问题
    idx = MultiIndex.from_product([["a", "b"], ["A", "B"]], names=["a", "b"])
    # 对同一索引进行对称差集操作
    result = idx.symmetric_difference(idx)
    # 断言结果的名称与原索引相同
    assert result.names == idx.names

    # 复制索引并重命名其标签
    idx2 = idx.copy().rename(["A", "B"])
    # 对不同索引进行对称差集操作
    result = idx.symmetric_difference(idx2)
    # 断言结果的名称为 [None, None]
    assert result.names == [None, None]


# 定义测试函数，用于测试空索引的相关操作
def test_empty(idx):
    # GH 15270：GitHub issue编号，用于跟踪问题
    assert not idx.empty
    # 断言取出的空切片索引为空
    assert idx[:0].empty


# 定义测试函数，用于测试索引差集操作
def test_difference(idx, sort):
    first = idx
    # 对索引进行差集操作，排除最后三个元素，根据sort参数进行排序
    result = first.difference(idx[-3:], sort=sort)
    vals = idx[:-3].values

    if sort is None:
        vals = sorted(vals)

    # 构建预期的多重索引
    expected = MultiIndex.from_tuples(vals, sortorder=0, names=idx.names)

    # 断言结果类型为MultiIndex
    assert isinstance(result, MultiIndex)
    # 断言结果与预期相等
    assert result.equals(expected)
    # 断言结果的名称与原索引相同
    assert result.names == idx.names
    # 使用测试模块的方法进行索引的比较
    tm.assert_index_equal(result, expected)

    # 空差集操作：自反性
    result = idx.difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # 空差集操作：超集
    result = idx[-3:].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # 空差集操作：退化
    result = idx[:0].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # 名称不同的情况
    chunklet = idx[-3:]
    chunklet.names = ["foo", "baz"]
    result = first.difference(chunklet, sort=sort)
    # 断言结果的名称为 (None, None)
    assert result.names == (None, None)

    # 空的，但非相等
    result = idx.difference(idx.sortlevel(1)[0], sort=sort)
    # 断言结果长度为0
    assert len(result) == 0

    # 抛出异常：使用非MultiIndex类型调用
    result = first.difference(first.values, sort=sort)
    assert result.equals(first[:0])

    # 来自空数组的名称
    result = first.difference([], sort=sort)
    # 断言原索引与结果相等，且名称相同
    assert first.equals(result)
    assert first.names == result.names

    # 来自非空数组的名称
    result = first.difference([("foo", "one")], sort=sort)
    expected = MultiIndex.from_tuples(
        [("bar", "one"), ("baz", "two"), ("foo", "two"), ("qux", "one"), ("qux", "two")]
    )
    expected.names = first.names
    # 断言结果的名称与预期相同
    assert first.names == result.names

    # 错误消息：other必须是MultiIndex或元组列表
    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.difference([1, 2, 3, 4, 5], sort=sort)


# 定义测试函数，用于测试特殊的排序差集操作
def test_difference_sort_special():
    # GH-24959：GitHub issue编号，用于跟踪问题
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # sort=None，默认排序方式
    result = idx.difference([])
    # 使用测试模块的方法断言结果与原索引相同
    tm.assert_index_equal(result, idx)


# 定义测试函数，用于测试特殊的排序差集操作（sort=True）
def test_difference_sort_special_true():
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # 使用sort=True进行排序差集操作
    result = idx.difference([], sort=True)
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    # 使用测试模块的方法断言结果与预期相同
    tm.assert_index_equal(result, expected)


# 定义测试函数，用于测试不可比较的排序差集操作
def test_difference_sort_incomparable():
    # GH-24959：GitHub issue编号，用于跟踪问题
    # 创建一个多级索引对象 idx，包含两级，第一级包含整数 1、时间戳 "2000"、整数 2，第二级包含字符串 "a" 和 "b"
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])
    
    # 创建另一个多级索引对象 other，包含两级，第一级包含整数 3、时间戳 "2000"、整数 4，第二级包含字符串 "c" 和 "d"
    other = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])
    
    # 设置 sort=None，这是默认值。如果有不可比较的对象，排序顺序未定义时，会产生 RuntimeWarning 警告。
    msg = "sort order is undefined for incomparable objects"
    with tm.assert_produces_warning(RuntimeWarning, match=msg):
        # 计算 idx 与 other 之间的差集
        result = idx.difference(other)
    
    # 断言 result 等于 idx，即 idx 和 other 之间的差集应该等于 idx 自身
    tm.assert_index_equal(result, idx)
    
    # 设置 sort=False
    # 计算 idx 与 other 之间的差集，此时指定不排序
    result = idx.difference(other, sort=False)
    
    # 断言 result 等于 idx，即 idx 和 other 之间的差集应该等于 idx 自身，因为此时不会排序
    tm.assert_index_equal(result, idx)
# 测试函数：测试在 MultiIndex 上执行 difference 操作时的行为
def test_difference_sort_incomparable_true():
    # 创建两个 MultiIndex 实例，每个实例都由两个层级组成，包含不同的值和类型
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])
    other = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])

    # 异常消息，指示在构建分类数据时调用 algos.safe_sort 时引发错误
    msg = "'values' is not ordered, please explicitly specify the categories order "
    # 使用 pytest 框架断言 TypeError 异常被正确地引发，并且异常消息与预期消息匹配
    with pytest.raises(TypeError, match=msg):
        idx.difference(other, sort=True)


# 测试函数：测试 MultiIndex 上的 union 方法
def test_union(idx, sort):
    # 从 MultiIndex 中选择一部分数据，并逆序排列，形成 piece1
    piece1 = idx[:5][::-1]
    # 从 MultiIndex 中选择另一部分数据，形成 piece2
    piece2 = idx[3:]

    # 使用 union 方法将 piece1 和 piece2 合并，根据 sort 参数进行排序或不排序
    the_union = piece1.union(piece2, sort=sort)

    # 根据 sort 参数的不同值，执行不同的断言
    if sort in (None, False):
        # 如果 sort 是 None 或 False，则断言排序后的结果与原始 idx 排序后的结果相等
        tm.assert_index_equal(the_union.sort_values(), idx.sort_values())
    else:
        # 否则，断言合并后的结果与原始 idx 相等
        tm.assert_index_equal(the_union, idx)

    # 对于特殊情况：传入相同的 MultiIndex 或空集合
    the_union = idx.union(idx, sort=sort)
    tm.assert_index_equal(the_union, idx)

    the_union = idx.union(idx[:0], sort=sort)
    tm.assert_index_equal(the_union, idx)

    # 使用 values 属性创建的元组，并将其与 idx 的一部分进行合并
    tuples = idx.values
    result = idx[:4].union(tuples[4:], sort=sort)
    if sort is None:
        # 如果 sort 是 None，则断言排序后的结果与原始 idx 排序后的结果相等
        tm.assert_index_equal(result.sort_values(), idx.sort_values())
    else:
        # 否则，断言合并后的结果与原始 idx 相等
        assert result.equals(idx)


# 测试函数：测试 Regular Index 与 MultiIndex 的 union 方法
def test_union_with_regular_index(idx, using_infer_string):
    # 创建一个 Regular Index
    other = Index(["A", "B", "C"])

    # 使用 union 方法将 Regular Index other 与 MultiIndex idx 合并
    result = other.union(idx)
    # 断言结果中包含指定的元组和值
    assert ("foo", "one") in result
    assert "B" in result

    # 根据 using_infer_string 参数值选择不同的断言方式
    if using_infer_string:
        # 如果 using_infer_string 为 True，则断言引发 NotImplementedError 异常，并且异常消息匹配
        with pytest.raises(NotImplementedError, match="Can only union"):
            idx.union(other)
    else:
        # 否则，断言引发 RuntimeWarning 警告，并且警告消息匹配
        msg = "The values in the array are unorderable"
        with tm.assert_produces_warning(RuntimeWarning, match=msg):
            result2 = idx.union(other)
        # 这种情况下更一致，如果排序失败，则完全不排序 MultiIndex 的情况
        assert not result.equals(result2)


# 测试函数：测试 MultiIndex 上的 intersection 方法
def test_intersection(idx, sort):
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]

    # 使用 intersection 方法找到 piece1 和 piece2 的交集，根据 sort 参数进行排序或不排序
    the_int = piece1.intersection(piece2, sort=sort)

    # 根据 sort 参数的不同值，执行不同的断言
    if sort in (None, True):
        # 如果 sort 是 None 或 True，则断言交集结果与预期的 idx[3:5] 相等
        tm.assert_index_equal(the_int, idx[3:5])
    else:
        # 否则，断言排序后的交集结果与预期的 idx[3:5] 相等
        tm.assert_index_equal(the_int.sort_values(), idx[3:5])

    # 对于特殊情况：传入相同的 MultiIndex
    the_int = idx.intersection(idx, sort=sort)
    tm.assert_index_equal(the_int, idx)

    # 空交集：不相交的情况
    empty = idx[:2].intersection(idx[2:], sort=sort)
    expected = idx[:0]
    assert empty.equals(expected)

    # 使用 values 属性创建的元组，并将其与 idx 进行交集操作
    tuples = idx.values
    result = idx.intersection(tuples)
    assert result.equals(idx)


# 使用 parametrize 装饰器定义的参数化测试：测试 MultiIndex 上的集合操作方法
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(idx, sort, method):
    # 将 MultiIndex 转换为扁平索引，并将其转换为类别类型
    other = idx.to_flat_index().astype("category")
    # 创建一个与 MultiIndex 层级数相同的空列表
    res_names = [None] * idx.nlevels

    # 调用指定的集合操作方法，并根据 sort 参数进行排序或不排序
    result = getattr(idx, method)(other, sort=sort)
    # 获取预期的结果并重命名
    expected = getattr(idx, method)(idx, sort=sort).rename(res_names)
    # 断言结果与预期结果相等
    tm.assert_index_equal(result, expected)
    # 调用对象 idx 的方法 method，并将 other 切片的前五个元素作为参数传入，使用 sort 参数进行排序
    result = getattr(idx, method)(other[:5], sort=sort)
    # 调用对象 idx 的方法 method，并将 idx 切片的前五个元素作为参数传入，使用 sort 参数进行排序，然后对结果重命名为 res_names
    expected = getattr(idx, method)(idx[:5], sort=sort).rename(res_names)
    # 使用测试工具包中的方法，验证 result 和 expected 是否相等
    tm.assert_index_equal(result, expected)
# 定义一个函数用于测试索引对象之间的交集操作，不使用对象排序
def test_intersection_non_object(idx, sort):
    # 创建一个名为"foo"的索引对象，包含范围为0到2的整数
    other = Index(range(3), name="foo")

    # 执行索引对象idx与other的交集操作，根据传入的sort参数进行排序
    result = idx.intersection(other, sort=sort)
    # 创建一个期望的多级索引对象，其级别、编码、名称均与idx相同
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=None)
    # 使用测试工具验证结果与期望是否相等，要求完全相等
    tm.assert_index_equal(result, expected, exact=True)

    # 如果传入长度为0的ndarray（即没有名称），保留idx的名称进行交集操作
    result = idx.intersection(np.asarray(other)[:0], sort=sort)
    # 创建另一个期望的多级索引对象，级别、编码与idx相同，名称与idx保持一致
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=idx.names)
    # 使用测试工具验证结果与期望是否相等，要求完全相等
    tm.assert_index_equal(result, expected, exact=True)

    # 抛出TypeError异常，要求错误消息为"other must be a MultiIndex or a list of tuples"
    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        # 当传入非零长度的非索引对象时，尝试将其转换为元组并失败
        idx.intersection(np.asarray(other), sort=sort)


# 定义一个函数用于测试两个相等的索引对象之间的交集操作，确保排序设置正确
def test_intersect_equal_sort():
    # 创建一个多级索引对象idx，其元素为[1, 0]和["a", "b"]的笛卡尔积
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # 使用测试工具验证无排序要求时的交集操作结果与idx是否相等
    tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
    # 使用测试工具验证sort=None时的交集操作结果与idx是否相等
    tm.assert_index_equal(idx.intersection(idx, sort=None), idx)


# 定义一个函数用于测试两个相等的索引对象之间的交集操作，要求进行排序
def test_intersect_equal_sort_true():
    # 创建一个多级索引对象idx，其元素为[1, 0]和["a", "b"]的笛卡尔积
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # 创建一个期望的多级索引对象，其元素为[0, 1]和["a", "b"]的笛卡尔积
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    # 执行索引对象idx与自身的交集操作，要求进行排序
    result = idx.intersection(idx, sort=True)
    # 使用测试工具验证结果与期望是否相等
    tm.assert_index_equal(result, expected)


# 使用参数化测试，测试当slice_为[slice(None), slice(0)]时的联合操作，要求不进行排序
@pytest.mark.parametrize("slice_", [slice(None), slice(0)])
def test_union_sort_other_empty(slice_):
    # 创建一个多级索引对象idx，其元素为[1, 0]和["a", "b"]的笛卡尔积
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])

    # 创建一个包含slice_条件的索引对象other
    other = idx[slice_]
    # 使用测试工具验证默认情况下的联合操作结果与idx是否相等，不要求排序
    tm.assert_index_equal(idx.union(other), idx)
    # 使用测试工具验证other与idx的联合操作结果是否相等，不要求排序
    tm.assert_index_equal(other.union(idx), idx)

    # 使用测试工具验证不要求排序时，idx与other的联合操作结果是否相等
    tm.assert_index_equal(idx.union(other, sort=False), idx)


# 定义一个函数用于测试当other为空时进行排序的联合操作
def test_union_sort_other_empty_sort():
    # 创建一个多级索引对象idx，其元素为[1, 0]和["a", "b"]的笛卡尔积
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # 创建一个空的索引对象other
    other = idx[:0]
    # 执行排序要求的idx与other的联合操作
    result = idx.union(other, sort=True)
    # 创建一个期望的多级索引对象，其元素为[0, 1]和["a", "b"]的笛卡尔积
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    # 使用测试工具验证结果与期望是否相等
    tm.assert_index_equal(result, expected)


# 定义一个函数用于测试当other包含无法比较的元素时的联合操作，不进行排序
def test_union_sort_other_incomparable():
    # 创建一个多级索引对象idx，其元素包含整数1和日期"2000-01-01"的笛卡尔积
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])

    # 执行默认情况下，idx与其自身一部分的联合操作
    with tm.assert_produces_warning(RuntimeWarning, match="are unorderable"):
        result = idx.union(idx[:1])
    # 使用测试工具验证结果与idx是否相等
    tm.assert_index_equal(result, idx)

    # 执行不要求排序的idx与其自身一部分的联合操作
    result = idx.union(idx[:1], sort=False)
    # 使用测试工具验证结果与idx是否相等
    tm.assert_index_equal(result, idx)


# 定义一个函数用于测试当other包含无法比较的元素时的联合操作，要求进行排序
def test_union_sort_other_incomparable_sort():
    # 创建一个多级索引对象idx，其元素包含整数1和日期"2000-01-01"的笛卡尔积
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])
    # 期望抛出TypeError异常，错误消息为"'<' not supported between instances of 'Timestamp' and 'int'"
    msg = "'<' not supported between instances of 'Timestamp' and 'int'"
    with pytest.raises(TypeError, match=msg):
        # 执行要求排序的idx与其自身一部分的联合操作
        idx.union(idx[:1], sort=True)


# 定义一个函数用于测试当索引对象包含非对象数据类型时的联合操作，预期抛出NotImplementedError异常
def test_union_non_object_dtype_raises():
    # 创建一个包含元素为["a", "b"]和[1, 2]的笛卡尔积的多级索引对象mi
    mi = MultiIndex.from_product([["a", "b"], [1, 2]])

    # 创建一个包含mi第二级别元素的索引对象idx
    idx = mi.levels[1]

    # 定义错误消息为"Can only union MultiIndex with MultiIndex or Index of tuples"
    msg = "Can only union MultiIndex with MultiIndex or Index of tuples"
    # 使用 pytest 的上下文管理器检查是否抛出 NotImplementedError 异常，并验证异常消息是否与给定的 msg 匹配
    with pytest.raises(NotImplementedError, match=msg):
        # 调用 mi 对象的 union 方法，并期望其抛出 NotImplementedError 异常
        mi.union(idx)
def test_union_empty_self_different_names():
    # GH#38423
    # 创建一个空的 MultiIndex 对象
    mi = MultiIndex.from_arrays([[]])
    # 创建另一个 MultiIndex 对象，包含两个层级的索引，并指定名称
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    # 对两个 MultiIndex 对象执行并集操作
    result = mi.union(mi2)
    # 创建预期结果的 MultiIndex 对象
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]])
    # 断言两个 MultiIndex 对象是否相等
    tm.assert_index_equal(result, expected)


def test_union_multiindex_empty_rangeindex():
    # GH#41234
    # 创建一个 MultiIndex 对象，包含两个层级的索引，并指定名称
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    # 创建一个空的 RangeIndex 对象
    ri = pd.RangeIndex(0)

    # 对 mi 和 ri 执行并集操作，结果赋值给 result_left
    result_left = mi.union(ri)
    # 断言 mi 和 result_left 是否相等，不检查名称
    tm.assert_index_equal(mi, result_left, check_names=False)

    # 对 ri 和 mi 执行并集操作，结果赋值给 result_right
    result_right = ri.union(mi)
    # 断言 mi 和 result_right 是否相等，不检查名称
    tm.assert_index_equal(mi, result_right, check_names=False)


@pytest.mark.parametrize(
    "method", ["union", "intersection", "difference", "symmetric_difference"]
)
def test_setops_sort_validation(method):
    # 创建两个 MultiIndex 对象，每个对象包含两个层级的笛卡尔积索引
    idx1 = MultiIndex.from_product([["a", "b"], [1, 2]])
    idx2 = MultiIndex.from_product([["b", "c"], [1, 2]])

    # 使用 pytest 检查是否会引发 ValueError 异常，匹配错误消息字符串
    with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
        # 使用 getattr 调用指定的 set 操作方法（如 union），传递 idx2 和 sort 参数
        getattr(idx1, method)(idx2, sort=2)

    # 当 sort=True 时支持的情况，无需保存结果
    getattr(idx1, method)(idx2, sort=True)


@pytest.mark.parametrize("val", [pd.NA, 100])
def test_difference_keep_ea_dtypes(any_numeric_ea_dtype, val):
    # GH#48606
    # 创建一个 MultiIndex 对象，包含两个层级的索引，每个层级由 Series 对象构成
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    # 创建另一个 MultiIndex 对象，包含两个层级的索引，每个层级由 Series 对象构成
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    # 对 midx 和 midx2 执行差集操作
    result = midx.difference(midx2)
    # 创建预期结果的 MultiIndex 对象
    expected = MultiIndex.from_arrays([Series([1], dtype=any_numeric_ea_dtype), [2]])
    # 断言结果是否与预期结果相等
    tm.assert_index_equal(result, expected)

    # 对 midx 和按降序排序后的 midx 执行差集操作
    result = midx.difference(midx.sort_values(ascending=False))
    # 创建预期结果的 MultiIndex 对象，包含空的 Series 对象
    expected = MultiIndex.from_arrays(
        [Series([], dtype=any_numeric_ea_dtype), Series([], dtype=np.int64)],
        names=["a", None],
    )
    # 断言结果是否与预期结果相等
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("val", [pd.NA, 5])
def test_symmetric_difference_keeping_ea_dtype(any_numeric_ea_dtype, val):
    # GH#48607
    # 创建一个 MultiIndex 对象，包含两个层级的索引，每个层级由 Series 对象构成
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    # 创建另一个 MultiIndex 对象，包含两个层级的索引，每个层级由 Series 对象构成
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    # 对 midx 和 midx2 执行对称差集操作
    result = midx.symmetric_difference(midx2)
    # 创建预期结果的 MultiIndex 对象
    expected = MultiIndex.from_arrays(
        [Series([1, 1, val], dtype=any_numeric_ea_dtype), [1, 2, 3]]
    )
    # 断言结果是否与预期结果相等
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    ("tuples", "exp_tuples"),
    [
        ([("val1", "test1")], [("val1", "test1")]),
        ([("val1", "test1"), ("val1", "test1")], [("val1", "test1")]),
        (
            [("val2", "test2"), ("val1", "test1")],
            [("val2", "test2"), ("val1", "test1")],
        ),
    ],
)
def test_intersect_with_duplicates(tuples, exp_tuples):
    # GH#36915
    # 创建一个 MultiIndex 对象，从元组列表中创建，指定层级的名称
    left = MultiIndex.from_tuples(tuples, names=["first", "second"])
    # 使用元组列表创建一个多级索引对象 `right`
    right = MultiIndex.from_tuples(
        [("val1", "test1"), ("val1", "test1"), ("val2", "test2")],
        names=["first", "second"],
    )
    # 对左侧的多级索引对象 `left` 和 `right` 进行交集操作，返回结果 `result`
    result = left.intersection(right)
    # 使用预期的元组列表 `exp_tuples` 创建一个多级索引对象 `expected`
    expected = MultiIndex.from_tuples(exp_tuples, names=["first", "second"])
    # 使用断言函数 `tm.assert_index_equal` 检查 `result` 是否与 `expected` 相等
    tm.assert_index_equal(result, expected)
# 使用 pytest 模块的 mark.parametrize 装饰器，为函数 test_maybe_match_names 提供参数化测试数据
@pytest.mark.parametrize(
    "data, names, expected",
    [
        ((1,), None, [None, None]),
        ((1,), ["a"], [None, None]),
        ((1,), ["b"], [None, None]),
        ((1, 2), ["c", "d"], [None, None]),
        ((1, 2), ["b", "a"], [None, None]),
        ((1, 2, 3), ["a", "b", "c"], [None, None]),
        ((1, 2), ["a", "c"], ["a", None]),
        ((1, 2), ["c", "b"], [None, "b"]),
        ((1, 2), ["a", "b"], ["a", "b"]),
        ((1, 2), [None, "b"], [None, "b"]),
    ],
)
def test_maybe_match_names(data, names, expected):
    # GH#38323
    # 创建空的 MultiIndex 对象 mi，命名为 ["a", "b"]
    mi = MultiIndex.from_tuples([], names=["a", "b"])
    # 使用给定的 data 和 names 创建新的 MultiIndex 对象 mi2
    mi2 = MultiIndex.from_tuples([data], names=names)
    # 调用 _maybe_match_names 方法，对比 mi 和 mi2 的命名匹配情况，返回结果
    result = mi._maybe_match_names(mi2)
    # 断言 result 是否等于预期的 expected 值
    assert result == expected


# 定义单元测试函数 test_intersection_equal_different_names
def test_intersection_equal_different_names():
    # GH#30302
    # 创建 mi1，mi2 两个 MultiIndex 对象，分别用 from_arrays 方法从数组创建，设置不同的命名
    mi1 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["c", "b"])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])

    # 调用 mi1 的 intersection 方法，传入 mi2，计算交集
    result = mi1.intersection(mi2)
    # 创建期望的 MultiIndex 对象 expected，命名为 [None, "b"]
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]], names=[None, "b"])
    # 使用 tm.assert_index_equal 进行断言，判断 result 是否等于 expected
    tm.assert_index_equal(result, expected)


# 定义单元测试函数 test_intersection_different_names
def test_intersection_different_names():
    # GH#38323
    # 创建 mi 和 mi2 两个 MultiIndex 对象，分别用 from_arrays 方法从数组创建，设置不同的命名
    mi = MultiIndex.from_arrays([[1], [3]], names=["c", "b"])
    mi2 = MultiIndex.from_arrays([[1], [3]])
    # 调用 mi 的 intersection 方法，传入 mi2，计算交集
    result = mi.intersection(mi2)
    # 使用 tm.assert_index_equal 进行断言，判断 result 是否等于 mi2
    tm.assert_index_equal(result, mi2)


# 定义单元测试函数 test_intersection_with_missing_values_on_both_sides
def test_intersection_with_missing_values_on_both_sides(nulls_fixture):
    # GH#38623
    # 创建 mi1 和 mi2 两个 MultiIndex 对象，分别用 from_arrays 方法从数组创建，包含空值处理
    mi1 = MultiIndex.from_arrays([[3, nulls_fixture, 4, nulls_fixture], [1, 2, 4, 2]])
    mi2 = MultiIndex.from_arrays([[3, nulls_fixture, 3], [1, 2, 4]])
    # 调用 mi1 的 intersection 方法，传入 mi2，计算交集
    result = mi1.intersection(mi2)
    # 创建期望的 MultiIndex 对象 expected，命名为 [[3, nulls_fixture], [1, 2]]
    expected = MultiIndex.from_arrays([[3, nulls_fixture], [1, 2]])
    # 使用 tm.assert_index_equal 进行断言，判断 result 是否等于 expected
    tm.assert_index_equal(result, expected)


# 定义单元测试函数 test_union_with_missing_values_on_both_sides
def test_union_with_missing_values_on_both_sides(nulls_fixture):
    # GH#38623
    # 创建 mi1 和 mi2 两个 MultiIndex 对象，分别用 from_arrays 方法从数组创建，包含空值处理
    mi1 = MultiIndex.from_arrays([[1, nulls_fixture]])
    mi2 = MultiIndex.from_arrays([[1, nulls_fixture, 3]])
    # 调用 mi1 的 union 方法，传入 mi2，计算并集
    result = mi1.union(mi2)
    # 创建期望的 MultiIndex 对象 expected，命名为 [[1, 3, nulls_fixture]]
    expected = MultiIndex.from_arrays([[1, 3, nulls_fixture]])
    # 使用 tm.assert_index_equal 进行断言，判断 result 是否等于 expected
    tm.assert_index_equal(result, expected)


# 使用 pytest 模块的 mark.parametrize 装饰器，为函数 test_union_nan_got_duplicated 提供参数化测试数据
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("sort", [None, False])
def test_union_nan_got_duplicated(dtype, sort):
    # GH#38977, GH#49010
    # 创建 mi1 和 mi2 两个 MultiIndex 对象，分别用 from_arrays 方法从数组创建，包含 NaN 值
    mi1 = MultiIndex.from_arrays([pd.array([1.0, np.nan], dtype=dtype), [2, 3]])
    mi2 = MultiIndex.from_arrays([pd.array([1.0, np.nan, 3.0], dtype=dtype), [2, 3, 4]])
    # 调用 mi1 的 union 方法，传入 mi2，并指定排序方式 sort
    result = mi1.union(mi2, sort=sort)
    if sort is None:
        # 创建期望的 MultiIndex 对象 expected，根据 sort 参数判断排序后的结果
        expected = MultiIndex.from_arrays(
            [pd.array([1.0, 3.0, np.nan], dtype=dtype), [2, 4, 3]]
        )
    else:
        expected = mi2
    # 使用 tm.assert_index_equal 进行断言，判断 result 是否等于 expected
    tm.assert_index_equal(result, expected)


# 使用 pytest 模块的 mark.parametrize 装饰器，为函数 test_union_keep_ea_dtype 提供参数化测试数据
@pytest.mark.parametrize("val", [4, 1])
def test_union_keep_ea_dtype(any_numeric_ea_dtype, val):
    # GH#48505
    # 创建 Series 对象 arr1 和 arr2，包含特定类型的数值数据
    arr1 = Series([val, 2], dtype=any_numeric_ea_dtype)
    arr2 = Series([2, 1], dtype=any_numeric_ea_dtype)
    # 使用 from_arrays 方法创建 MultiIndex 对象 midx，命名为 ["a", None]
    midx = MultiIndex.from_arrays([arr1, [1, 2]], names=["a", None])
    # 使用给定的两个数组创建一个 MultiIndex 对象，其中 arr2 是第二级索引
    midx2 = MultiIndex.from_arrays([arr2, [2, 1]])
    # 将当前的 MultiIndex 对象 midx 与 midx2 合并，生成新的 MultiIndex 对象 result
    result = midx.union(midx2)
    # 检查 val 是否等于 4
    if val == 4:
        # 如果 val 等于 4，创建一个预期的 MultiIndex 对象，第一级索引包含 [1, 2, 4]，第二级索引为 [1, 2, 1]
        expected = MultiIndex.from_arrays(
            [Series([1, 2, 4], dtype=any_numeric_ea_dtype), [1, 2, 1]]
        )
    else:
        # 如果 val 不等于 4，创建一个预期的 MultiIndex 对象，第一级索引包含 [1, 2]，第二级索引为 [1, 2]
        expected = MultiIndex.from_arrays(
            [Series([1, 2], dtype=any_numeric_ea_dtype), [1, 2]]
        )
    # 使用断言检查 result 和 expected 是否相等
    tm.assert_index_equal(result, expected)
@pytest.mark.parametrize("dupe_val", [3, pd.NA])
def test_union_with_duplicates_keep_ea_dtype(dupe_val, any_numeric_ea_dtype):
    # 定义一个参数化测试函数，用于测试合并多重索引并保持元素数据类型
    # GH48900
    
    # 创建第一个多重索引对象 mi1，包含两个相同的系列，数据类型为任意数值类型
    mi1 = MultiIndex.from_arrays(
        [
            Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype),
            Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype),
        ]
    )
    
    # 创建第二个多重索引对象 mi2，包含两个相同的系列，数据类型为任意数值类型
    mi2 = MultiIndex.from_arrays(
        [
            Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
            Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
        ]
    )
    
    # 对 mi1 和 mi2 进行合并操作
    result = mi1.union(mi2)
    
    # 创建预期的多重索引对象 expected，包含合并后的系列，数据类型为任意数值类型
    expected = MultiIndex.from_arrays(
        [
            Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
            Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
        ]
    )
    
    # 断言合并结果与预期结果相等
    tm.assert_index_equal(result, expected)


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_union_duplicates(index, request):
    # 定义一个测试函数，用于测试合并多重索引中的重复值
    # GH#38977
    
    # 如果索引为空或者是区间索引或分类索引，则跳过测试
    if index.empty or isinstance(index, (IntervalIndex, CategoricalIndex)):
        pytest.skip(f"No duplicates in an empty {type(index).__name__}")
    
    # 获取索引的唯一值列表
    values = index.unique().values.tolist()
    
    # 创建第一个多重索引对象 mi1，包含唯一值及对应的整数系列
    mi1 = MultiIndex.from_arrays([values, [1] * len(values)])
    
    # 创建第二个多重索引对象 mi2，包含第一个值重复一次后的唯一值及对应的整数系列
    mi2 = MultiIndex.from_arrays([[values[0]] + values, [1] * (len(values) + 1)])
    
    # 对 mi2 和 mi1 进行合并操作
    result = mi2.union(mi1)
    
    # 对合并后的结果进行排序作为预期结果
    expected = mi2.sort_values()
    
    # 断言合并结果与预期结果相等
    tm.assert_index_equal(result, expected)
    
    # 如果 mi2 的第一级别是无符号整数类型并且所有值小于 2**63
    if (
        is_unsigned_integer_dtype(mi2.levels[0])
        and (mi2.get_level_values(0) < 2**63).all()
    ):
        # 对预期结果的第一级别进行类型转换为 np.int64 类型
        expected = expected.set_levels(
            [expected.levels[0].astype(np.int64), expected.levels[1]]
        )
    elif is_float_dtype(mi2.levels[0]):
        # 如果 mi2 的第一级别是浮点数类型，则对预期结果的第一级别进行类型转换为 float 类型
        expected = expected.set_levels(
            [expected.levels[0].astype(float), expected.levels[1]]
        )
    
    # 再次对 mi1 和 mi2 进行合并操作
    result = mi1.union(mi2)
    
    # 断言合并结果与预期结果相等
    tm.assert_index_equal(result, expected)


def test_union_keep_dtype_precision(any_real_numeric_dtype):
    # 定义一个测试函数，用于测试保持多重索引的数据类型精度
    # GH#48498
    
    # 创建第一个系列 arr1，包含数值，并指定数据类型为任意实数数值类型
    arr1 = Series([4, 1, 1], dtype=any_real_numeric_dtype)
    
    # 创建第二个系列 arr2，包含数值，并指定数据类型为任意实数数值类型
    arr2 = Series([1, 4], dtype=any_real_numeric_dtype)
    
    # 创建多重索引对象 midx，由 arr1 和一个对应的系列组成
    midx = MultiIndex.from_arrays([arr1, [2, 1, 1]], names=["a", None])
    
    # 创建多重索引对象 midx2，由 arr2 和一个对应的系列组成
    midx2 = MultiIndex.from_arrays([arr2, [1, 2]], names=["a", None])
    
    # 对 midx 和 midx2 进行合并操作
    result = midx.union(midx2)
    
    # 创建预期的多重索引对象 expected，包含合并后的系列，数据类型为任意实数数值类型
    expected = MultiIndex.from_arrays(
        ([Series([1, 1, 4], dtype=any_real_numeric_dtype), [1, 1, 2]]),
        names=["a", None],
    )
    
    # 断言合并结果与预期结果相等
    tm.assert_index_equal(result, expected)


def test_union_keep_ea_dtype_with_na(any_numeric_ea_dtype):
    # 定义一个测试函数，用于测试在包含 NA 值时保持多重索引的元素数据类型
    # GH#48498
    
    # 创建系列 arr1，包含数值和 NA 值，并指定数据类型为任意数值元素数据类型
    arr1 = Series([4, pd.NA], dtype=any_numeric_ea_dtype)
    # 创建一个 Series 对象，包含两个元素：1 和 pd.NA，数据类型为 any_numeric_ea_dtype 指定的类型
    arr2 = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
    # 使用两个数组创建一个 MultiIndex 对象，其中第一个数组是 arr1，第二个数组是 [2, 1]，指定第二个数组的名称为 "a"
    midx = MultiIndex.from_arrays([arr1, [2, 1]], names=["a", None])
    # 使用两个数组创建另一个 MultiIndex 对象，其中第一个数组是 arr2，第二个数组是 [1, 2]
    midx2 = MultiIndex.from_arrays([arr2, [1, 2]])
    # 对两个 MultiIndex 对象执行 union 操作，生成一个新的 MultiIndex 对象 result
    result = midx.union(midx2)
    # 创建一个预期的 MultiIndex 对象 expected，其中第一个数组是包含 [1, 4, pd.NA, pd.NA] 的 Series 对象，数据类型为 any_numeric_ea_dtype 指定的类型；第二个数组是 [1, 2, 1, 2]
    expected = MultiIndex.from_arrays(
        [Series([1, 4, pd.NA, pd.NA], dtype=any_numeric_ea_dtype), [1, 2, 1, 2]]
    )
    # 使用测试框架中的方法，验证 result 和 expected 两个 MultiIndex 对象是否相等
    tm.assert_index_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器来定义多组参数化测试参数
@pytest.mark.parametrize(
    "levels1, levels2, codes1, codes2, names",
    [
        (
            [["a", "b", "c"], [0, ""]],  # 第一组参数的 levels1, levels2 数组
            [["c", "d", "b"], [""]],    # 第一组参数的 codes1, codes2 数组
            [[0, 1, 2], [1, 1, 1]],     # 第一组参数的 names 数组
            [[0, 1, 2], [0, 0, 0]],     # 第二组参数的 names 数组
            ["name1", "name2"],        # 参数 names 数组
        ),
    ],
)
# 定义测试函数 test_intersection_lexsort_depth
def test_intersection_lexsort_depth(levels1, levels2, codes1, codes2, names):
    # GH#25169: 标识 GitHub 上的 issue 编号
    # 创建 MultiIndex 对象 mi1 和 mi2，使用给定的 levels, codes 和 names 参数
    mi1 = MultiIndex(levels=levels1, codes=codes1, names=names)
    mi2 = MultiIndex(levels=levels2, codes=codes2, names=names)
    # 对 mi1 和 mi2 执行 intersection 操作，获取交集结果 mi_int
    mi_int = mi1.intersection(mi2)
    # 断言 mi_int 对象的 _lexsort_depth 属性等于 2
    assert mi_int._lexsort_depth == 2


# 使用 pytest 的 parametrize 装饰器定义多组参数化测试参数
@pytest.mark.parametrize(
    "a",
    [pd.Categorical(["a", "b"], categories=["a", "b"]), ["a", "b"]],
)
# 参数化测试 b_ordered 参数，分别为 True 和 False
@pytest.mark.parametrize("b_ordered", [True, False])
# 定义测试函数 test_intersection_with_non_lex_sorted_categories
def test_intersection_with_non_lex_sorted_categories(a, b_ordered):
    # GH#49974: 标识 GitHub 上的 issue 编号
    other = ["1", "2"]
    # 创建 pd.Categorical 对象 b，指定 categories 和 ordered 参数
    b = pd.Categorical(["a", "b"], categories=["b", "a"], ordered=b_ordered)
    # 创建 DataFrame df1 和 df2，包含指定的 Series 列 'x' 和 'y'
    df1 = DataFrame({"x": a, "y": other})
    df2 = DataFrame({"x": b, "y": other})

    # 期望的 MultiIndex 对象，从 df1 的 'x' 和 'y' 列创建
    expected = MultiIndex.from_arrays([a, other], names=["x", "y"])

    # 执行四种不同排序方式的 intersection 操作，得到结果 res1, res2, res3, res4
    res1 = MultiIndex.from_frame(df1).intersection(
        MultiIndex.from_frame(df2.sort_values(["x", "y"]))
    )
    res2 = MultiIndex.from_frame(df1).intersection(MultiIndex.from_frame(df2))
    res3 = MultiIndex.from_frame(df1.sort_values(["x", "y"])).intersection(
        MultiIndex.from_frame(df2)
    )
    res4 = MultiIndex.from_frame(df1.sort_values(["x", "y"])).intersection(
        MultiIndex.from_frame(df2.sort_values(["x", "y"]))
    )

    # 断言四个结果与期望结果相等
    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)


# 使用 pytest 的 parametrize 装饰器定义多组参数化测试参数
@pytest.mark.parametrize("val", [pd.NA, 100])
# 定义测试函数 test_intersection_keep_ea_dtypes
def test_intersection_keep_ea_dtypes(val, any_numeric_ea_dtype):
    # GH#48604: 标识 GitHub 上的 issue 编号
    # 创建 MultiIndex 对象 midx 和 midx2，使用指定的 Series 列作为数组
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    # 执行 midx 和 midx2 的 intersection 操作，得到结果 result
    result = midx.intersection(midx2)
    # 期望的 MultiIndex 对象，从指定的 Series 列创建
    expected = MultiIndex.from_arrays([Series([2], dtype=any_numeric_ea_dtype), [1]])
    # 断言 result 与 expected 结果相等
    tm.assert_index_equal(result, expected)


# 定义测试函数 test_union_with_na_when_constructing_dataframe
def test_union_with_na_when_constructing_dataframe():
    # GH43222: 标识 GitHub 上的 issue 编号
    # 创建 Series 对象 series1 和 series2，分别使用 MultiIndex.from_arrays 和 from_tuples 构造索引
    series1 = Series(
        (1,),
        index=MultiIndex.from_arrays(
            [Series([None], dtype="string"), Series([None], dtype="string")]
        ),
    )
    series2 = Series((10, 20), index=MultiIndex.from_tuples(((None, None), ("a", "b"))))
    # 创建 DataFrame 对象 result，包含两个 Series
    result = DataFrame([series1, series2])
    # 期望的 DataFrame 对象，包含指定的数据和索引
    expected = DataFrame({(np.nan, np.nan): [1.0, 10.0], ("a", "b"): [np.nan, 20.0]})
    # 断言 result 与 expected 的内容相等
    tm.assert_frame_equal(result, expected)
```