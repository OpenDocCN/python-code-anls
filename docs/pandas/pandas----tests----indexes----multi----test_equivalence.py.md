# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_equivalence.py`

```
# 导入必要的库
import numpy as np  # 导入numpy库，用于数组操作
import pytest  # 导入pytest库，用于单元测试

# 导入pandas库及其子模块和函数
from pandas.core.dtypes.common import is_any_real_numeric_dtype
import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm  # 导入pandas测试工具模块

# 定义测试函数，用于测试索引对象的相等性
def test_equals(idx):
    # 断言索引对象与自身相等
    assert idx.equals(idx)
    # 断言索引对象与其副本相等
    assert idx.equals(idx.copy())
    # 断言索引对象与转换为object类型后的对象相等
    assert idx.equals(idx.astype(object))
    # 断言索引对象与转换为flat index后的对象相等
    assert idx.equals(idx.to_flat_index())
    # 断言索引对象与转换为category类型后的flat index对象相等
    assert idx.equals(idx.to_flat_index().astype("category"))

    # 断言索引对象与其转换为列表后的对象不相等
    assert not idx.equals(list(idx))
    # 断言索引对象与其转换为numpy数组后的对象不相等
    assert not idx.equals(np.array(idx))

    # 创建与idx相同值的Index对象，但dtype为object
    same_values = Index(idx, dtype=object)
    # 断言索引对象与same_values对象相等
    assert idx.equals(same_values)
    # 断言same_values对象与idx对象相等
    assert same_values.equals(idx)

    # 如果索引对象只有一个层级
    if idx.nlevels == 1:
        # 不应测试MultiIndex对象的相等性
        assert not idx.equals(Series(idx))

# 定义测试函数，用于测试索引对象的操作相等性
def test_equals_op(idx):
    # GH9947, GH10637
    index_a = idx

    # 获取index_a的长度
    n = len(index_a)
    # 根据index_a生成新的索引对象index_b，长度比index_a小1
    index_b = index_a[0:-1]
    # 根据index_a生成新的索引对象index_c，长度比index_a小1，并添加最后一个元素
    index_c = index_a[0:-1].append(index_a[-2:-1])
    # 根据index_a生成新的索引对象index_d，仅包含第一个元素
    index_d = index_a[0:1]
    
    # 断言index_a与index_b相等会抛出ValueError异常，提示长度不匹配
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == index_b
    # 创建预期的布尔数组expected1，所有元素为True，长度为n
    expected1 = np.array([True] * n)
    # 创建预期的布尔数组expected2，前n-1个元素为True，最后一个元素为False，长度为n
    expected2 = np.array([True] * (n - 1) + [False])
    # 使用测试工具模块中的函数，断言index_a与index_a的操作相等性
    tm.assert_numpy_array_equal(index_a == index_a, expected1)
    # 使用测试工具模块中的函数，断言index_a与index_c的操作相等性
    tm.assert_numpy_array_equal(index_a == index_c, expected2)

    # 使用numpy数组与索引对象进行比较的测试
    array_a = np.array(index_a)
    array_b = np.array(index_a[0:-1])
    array_c = np.array(index_a[0:-1].append(index_a[-2:-1]))
    array_d = np.array(index_a[0:1])
    # 断言index_a与array_b相等会抛出ValueError异常，提示长度不匹配
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == array_b
    # 使用测试工具模块中的函数，断言index_a与array_a的操作相等性
    tm.assert_numpy_array_equal(index_a == array_a, expected1)
    # 使用测试工具模块中的函数，断言index_a与array_c的操作相等性
    tm.assert_numpy_array_equal(index_a == array_c, expected2)

    # 使用Series对象与索引对象进行比较的测试
    series_a = Series(array_a)
    series_b = Series(array_b)
    series_c = Series(array_c)
    series_d = Series(array_d)
    # 断言index_a与series_b相等会抛出ValueError异常，提示长度不匹配
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == series_b

    # 使用测试工具模块中的函数，断言index_a与series_a的操作相等性
    tm.assert_numpy_array_equal(index_a == series_a, expected1)
    # 使用测试工具模块中的函数，断言index_a与series_c的操作相等性
    tm.assert_numpy_array_equal(index_a == series_c, expected2)

    # 当其中一个对象长度为1时的特殊情况处理
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == index_d
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == series_d
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == array_d
    msg = "Can only compare identically-labeled Series objects"
    with pytest.raises(ValueError, match=msg):
        series_a == series_d
    with pytest.raises(ValueError, match="Lengths must match"):
        series_a == array_d

    # 与标量比较应该进行广播处理；注意，这里排除了MultiIndex，因为在这种情况下，索引的每个项都是长度为2的元组，
    # 因此在比较中被视为长度为2的数组，而不是标量
    # 检查 index_a 是否不是 MultiIndex 类型
    if not isinstance(index_a, MultiIndex):
        # 创建一个布尔数组 expected3，长度为 index_a 的长度减去 2，最后两个元素分别为 False 和 True
        expected3 = np.array([False] * (len(index_a) - 2) + [True, False])
        
        # 假设倒数第二个元素在数据中是唯一的
        item = index_a[-2]
        
        # 使用断言验证 index_a 是否与 item 相等，结果应为 expected3
        tm.assert_numpy_array_equal(index_a == item, expected3)
        
        # 使用断言验证 series_a 是否与 item 相等，结果应为一个包含 expected3 的 Series 对象
        tm.assert_series_equal(series_a == item, Series(expected3))
def test_compare_tuple():
    # GH#21517
    # 创建一个多级索引，包含由 [1, 2] 组成的笛卡尔积
    mi = MultiIndex.from_product([[1, 2]] * 2)

    # 创建一个包含全部为 False 的 NumPy 数组
    all_false = np.array([False, False, False, False])

    # 比较多级索引 mi 与 mi 的第一个元素
    result = mi == mi[0]
    expected = np.array([True, False, False, False])
    # 断言两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 比较多级索引 mi 与 mi 的第一个元素是否不等
    result = mi != mi[0]
    # 断言 result 与 expected 的逻辑非是否相等
    tm.assert_numpy_array_equal(result, ~expected)

    # 比较多级索引 mi 是否小于 mi 的第一个元素
    result = mi < mi[0]
    # 断言 result 是否与 all_false 数组相等
    tm.assert_numpy_array_equal(result, all_false)

    # 比较多级索引 mi 是否小于等于 mi 的第一个元素
    result = mi <= mi[0]
    # 断言 result 是否与 expected 数组相等
    tm.assert_numpy_array_equal(result, expected)

    # 比较多级索引 mi 是否大于 mi 的第一个元素
    result = mi > mi[0]
    # 断言 result 与 expected 的逻辑非是否相等
    tm.assert_numpy_array_equal(result, ~expected)

    # 比较多级索引 mi 是否大于等于 mi 的第一个元素
    result = mi >= mi[0]
    # 断言 result 与 all_false 的逻辑非是否相等
    tm.assert_numpy_array_equal(result, ~all_false)


def test_compare_tuple_strs():
    # GH#34180
    # 使用元组列表创建多级索引 mi
    mi = MultiIndex.from_tuples([("a", "b"), ("b", "c"), ("c", "a")])

    # 比较多级索引 mi 是否等于元组 ("c", "a")
    result = mi == ("c", "a")
    expected = np.array([False, False, True])
    # 断言两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 比较多级索引 mi 是否等于元组 ("c",)
    result = mi == ("c",)
    expected = np.array([False, False, False])
    # 断言两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


def test_equals_multi(idx):
    # 断言 idx 是否与自身相等
    assert idx.equals(idx)
    # 断言 idx 与其值不相等
    assert not idx.equals(idx.values)
    # 断言 idx 是否与其值为元素的 Index 相等
    assert idx.equals(Index(idx.values))

    # 断言 idx 的层级是否相等
    assert idx.equal_levels(idx)
    # 断言 idx 是否与其前 n-1 个元素不相等
    assert not idx.equals(idx[:-1])
    # 断言 idx 是否与其最后一个元素不相等
    assert not idx.equals(idx[-1])

    # 创建一个不同层数的 MultiIndex 对象 index
    index = MultiIndex(
        levels=[Index(list(range(4))), Index(list(range(4))), Index(list(range(4)))],
        codes=[
            np.array([0, 0, 1, 2, 2, 2, 3, 3]),
            np.array([0, 1, 0, 0, 0, 1, 0, 1]),
            np.array([1, 0, 1, 1, 0, 0, 1, 0]),
        ],
    )

    # 创建一个与 index 不同层数的 MultiIndex 对象 index2
    index2 = MultiIndex(levels=index.levels[:-1], codes=index.codes[:-1])
    # 断言 index 与 index2 是否不相等
    assert not index.equals(index2)
    # 断言 index 的层级是否不相等
    assert not index.equal_levels(index2)

    # 创建不同层级的 MultiIndex 对象 index
    major_axis = Index(list(range(4)))
    minor_axis = Index(list(range(2)))

    major_codes = np.array([0, 0, 1, 2, 2, 3])
    minor_codes = np.array([0, 1, 0, 0, 1, 0])

    index = MultiIndex(
        levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
    )
    # 断言 idx 与 index 是否不相等
    assert not idx.equals(index)
    # 断言 idx 的层级是否不相等
    assert not idx.equal_levels(index)

    # 创建带有不同标签的 MultiIndex 对象 index
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 2, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])

    index = MultiIndex(
        levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
    )
    # 断言 idx 与 index 是否不相等
    assert not idx.equals(index)


def test_identical(idx):
    # 创建 idx 的副本 mi 和 mi2
    mi = idx.copy()
    mi2 = idx.copy()
    # 断言 mi 与 mi2 是否相同
    assert mi.identical(mi2)

    # 修改 mi 的名称为 ["new1", "new2"]
    mi = mi.set_names(["new1", "new2"])
    # 断言 mi 与 mi2 是否相等
    assert mi.equals(mi2)
    # 断言 mi 与 mi2 是否不完全相同
    assert not mi.identical(mi2)

    # 修改 mi2 的名称为 ["new1", "new2"]
    mi2 = mi2.set_names(["new1", "new2"])
    # 断言 mi 与 mi2 是否完全相同
    assert mi.identical(mi2)

    # 创建一个不元组化列的 Index 对象 mi4
    mi4 = Index(mi.tolist(), tupleize_cols=False)
    # 断言 mi 与 mi4 是否不完全相同
    assert not mi.identical(mi4)
    # 断言 mi 与 mi4 是否相等
    assert mi.equals(mi4)


def test_equals_operator(idx):
    # GH9785
    # 断言 idx 是否与自身完全相等
    assert (idx == idx).all()


def test_equals_missing_values():
    # 未提供此函数的完整代码，无法提供注释
    # 创建一个包含多层索引的 MultiIndex 对象，包括两个元组：(0, NaT) 和 (0, 2013年1月1日的时间戳)
    i = MultiIndex.from_tuples([(0, pd.NaT), (0, pd.Timestamp("20130101"))])
    # 检查 i[0:1] 是否等于 i[0]，返回比较结果
    result = i[0:1].equals(i[0])
    # 使用断言确保 result 的值为 False
    assert not result
    # 检查 i[1:2] 是否等于 i[1]，返回比较结果
    result = i[1:2].equals(i[1])
    # 使用断言确保 result 的值为 False
    assert not result
def test_equals_missing_values_differently_sorted():
    # GH#38439
    # 创建包含缺失值的多级索引 mi1 和 mi2
    mi1 = MultiIndex.from_tuples([(81.0, np.nan), (np.nan, np.nan)])
    mi2 = MultiIndex.from_tuples([(np.nan, np.nan), (81.0, np.nan)])
    # 断言 mi1 和 mi2 不相等
    assert not mi1.equals(mi2)

    # 更新 mi2 使其与 mi1 相等
    mi2 = MultiIndex.from_tuples([(81.0, np.nan), (np.nan, np.nan)])
    # 断言 mi1 和 mi2 相等
    assert mi1.equals(mi2)


def test_is_():
    # 创建一个包含整数对的多级索引 mi
    mi = MultiIndex.from_tuples(zip(range(10), range(10)))
    # 断言 mi 等于自身
    assert mi.is_(mi)
    # 断言 mi 等于其视图的对象
    assert mi.is_(mi.view())
    # 连续视图的断言
    assert mi.is_(mi.view().view().view().view())
    mi2 = mi.view()
    # 修改元数据名称不会改变对象 id
    mi2.names = ["A", "B"]
    # 断言 mi2 与 mi 相等
    assert mi2.is_(mi)
    # 断言 mi 与 mi2 相等
    assert mi.is_(mi2)

    # 断言 mi 与设置名称后的对象不相等
    assert not mi.is_(mi.set_names(["C", "D"]))
    # 设置级别属性后，对象 id 发生变化
    mi3 = mi2.set_levels([list(range(10)), list(range(10))])
    # 断言 mi3 与 mi2 不相等
    assert not mi3.is_(mi2)
    # 断言 mi2 与 mi 相等
    assert mi2.is_(mi)
    mi4 = mi3.view()

    # GH 17464 - 删除重复的 MultiIndex 级别
    mi4 = mi4.set_levels([list(range(10)), list(range(10))])
    # 断言 mi4 与 mi3 不相等
    assert not mi4.is_(mi3)
    mi5 = mi.view()
    mi5 = mi5.set_levels(mi5.levels)
    # 断言 mi5 与 mi 不相等
    assert not mi5.is_(mi)


def test_is_all_dates(idx):
    # 断言 idx 的 _is_all_dates 属性为 False
    assert not idx._is_all_dates


def test_is_numeric(idx):
    # 多级索引 idx 永远不是数值型
    assert not is_any_real_numeric_dtype(idx)


def test_multiindex_compare():
    # GH 21149
    # 确保具有 nlevels == 1 的 MultiIndex 的比较操作与具有 nlevels > 1 的一致行为

    # 创建一个包含单个级别的 MultiIndex
    midx = MultiIndex.from_product([[0, 1]])

    # 自身相等性测试：MultiIndex 对象与自身比较
    expected = Series([True, True])
    result = Series(midx == midx)
    tm.assert_series_equal(result, expected)

    # 大于比较：MultiIndex 对象与自身比较
    expected = Series([False, False])
    result = Series(midx > midx)
    tm.assert_series_equal(result, expected)


def test_equals_ea_int_regular_int():
    # GH#46026
    # 创建包含整数和标准整数的不同 MultiIndex 对象 mi1 和 mi2
    mi1 = MultiIndex.from_arrays([Index([1, 2], dtype="Int64"), [3, 4]])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]])
    # 断言 mi1 和 mi2 不相等
    assert not mi1.equals(mi2)
    assert not mi2.equals(mi1)
```