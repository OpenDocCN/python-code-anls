# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_isin.py`

```
import numpy as np  # 导入NumPy库，用于处理数组和数值计算
import pytest  # 导入pytest库，用于编写和运行测试

from pandas import MultiIndex  # 从pandas库中导入MultiIndex类，用于处理多层索引
import pandas._testing as tm  # 导入pandas内部测试工具模块，命名为tm

def test_isin_nan():
    idx = MultiIndex.from_arrays([["foo", "bar"], [1.0, np.nan]])  # 创建一个包含NaN的MultiIndex对象
    tm.assert_numpy_array_equal(idx.isin([("bar", np.nan)]), np.array([False, True]))  # 断言MultiIndex对象中是否包含指定元素
    tm.assert_numpy_array_equal(
        idx.isin([("bar", float("nan"))]), np.array([False, True])
    )  # 断言MultiIndex对象中是否包含指定NaN值的元组


def test_isin_missing(nulls_fixture):
    # GH48905
    mi1 = MultiIndex.from_tuples([(1, nulls_fixture)])  # 创建一个包含空值的MultiIndex对象
    mi2 = MultiIndex.from_tuples([(1, 1), (1, 2)])  # 创建另一个MultiIndex对象
    result = mi2.isin(mi1)  # 判断mi2中的元组是否在mi1中存在
    expected = np.array([False, False])  # 期望的结果数组
    tm.assert_numpy_array_equal(result, expected)  # 断言结果是否与期望一致


def test_isin():
    values = [("foo", 2), ("bar", 3), ("quux", 4)]  # 定义一个包含元组的列表

    idx = MultiIndex.from_arrays([["qux", "baz", "foo", "bar"], np.arange(4)])  # 创建一个包含多层索引的MultiIndex对象
    result = idx.isin(values)  # 判断MultiIndex对象中是否包含给定的元组
    expected = np.array([False, False, True, True])  # 期望的结果数组
    tm.assert_numpy_array_equal(result, expected)  # 断言结果是否与期望一致

    # 空的MultiIndex对象，返回布尔类型的结果数组
    idx = MultiIndex.from_arrays([[], []])
    result = idx.isin(values)
    assert len(result) == 0  # 断言结果数组的长度为0
    assert result.dtype == np.bool_  # 断言结果数组的数据类型为布尔类型


def test_isin_level_kwarg():
    idx = MultiIndex.from_arrays([["qux", "baz", "foo", "bar"], np.arange(4)])  # 创建一个包含多层索引的MultiIndex对象

    vals_0 = ["foo", "bar", "quux"]
    vals_1 = [2, 3, 10]

    expected = np.array([False, False, True, True])  # 期望的结果数组
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level=0))  # 断言在指定层级上是否包含给定元素
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level=-2))  # 使用负数索引也可以指定层级

    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level=1))  # 断言在指定层级上是否包含给定元素
    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level=-1))  # 使用负数索引也可以指定层级

    # 测试超出索引范围的情况，应该触发IndexError异常
    msg = "Too many levels: Index has only 2 levels, not 6"
    with pytest.raises(IndexError, match=msg):
        idx.isin(vals_0, level=5)
    msg = "Too many levels: Index has only 2 levels, -5 is not a valid level number"
    with pytest.raises(IndexError, match=msg):
        idx.isin(vals_0, level=-5)

    # 测试指定不存在的层级，应该触发KeyError异常
    with pytest.raises(KeyError, match=r"'Level 1\.0 not found'"):
        idx.isin(vals_0, level=1.0)
    with pytest.raises(KeyError, match=r"'Level -1\.0 not found'"):
        idx.isin(vals_1, level=-1.0)
    with pytest.raises(KeyError, match="'Level A not found'"):
        idx.isin(vals_1, level="A")

    idx.names = ["A", "B"]  # 设置MultiIndex对象的层级名称
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level="A"))  # 断言在指定层级名称上是否包含给定元素
    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level="B"))  # 断言在指定层级名称上是否包含给定元素

    # 测试指定不存在的层级名称，应该触发KeyError异常
    with pytest.raises(KeyError, match="'Level C not found'"):
        idx.isin(vals_1, level="C")


@pytest.mark.parametrize(
    "labels,expected,level",
    [
        ([("b", np.nan)], [False, False, True], None),  # 参数化测试数据，包含NaN值的元组
        ([np.nan, "a"], [True, True, False], 0),  # 参数化测试数据，包含NaN值的元素
        (["d", np.nan], [False, True, True], 1),  # 参数化测试数据，包含NaN值的元素
    ],
)
def test_isin_multi_index_with_missing_value(labels, expected, level):
    # GH 19132
    midx = MultiIndex.from_arrays([[np.nan, "a", "b"], ["c", "d", np.nan]])  # 创建一个包含NaN值的MultiIndex对象
    result = midx.isin(labels, level=level)  # 判断MultiIndex对象中是否包含指定的元素
    expected = np.array(expected)  # 期望的结果数组
    # 使用测试工具库中的函数比较两个 NumPy 数组 result 和 expected 是否相等
    tm.assert_numpy_array_equal(result, expected)
def test_isin_empty():
    # GH#51599
    # 创建一个多级索引对象，使用两个数组作为索引的级别
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]])
    # 对多级索引对象进行isin操作，判断是否包含空列表中的元素，返回布尔数组
    result = midx.isin([])
    # 期望的结果是一个NumPy数组，表示预期的isin操作结果
    expected = np.array([False, False])
    # 使用测试工具函数检查result和expected是否相等
    tm.assert_numpy_array_equal(result, expected)


def test_isin_generator():
    # GH#52568
    # 创建一个多级索引对象，使用元组列表作为索引的级别
    midx = MultiIndex.from_tuples([(1, 2)])
    # 对多级索引对象进行isin操作，判断是否包含生成器生成的元组，返回布尔数组
    result = midx.isin(x for x in [(1, 2)])
    # 期望的结果是一个NumPy数组，表示预期的isin操作结果
    expected = np.array([True])
    # 使用测试工具函数检查result和expected是否相等
    tm.assert_numpy_array_equal(result, expected)
```