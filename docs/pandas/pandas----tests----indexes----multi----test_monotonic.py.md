# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_monotonic.py`

```
# 导入必要的库
import numpy as np
import pytest

# 从 pandas 库中导入 Index 和 MultiIndex 类
from pandas import (
    Index,
    MultiIndex,
)


# 定义测试函数，测试带有词典序排列的两级字符串 MultiIndex
def test_is_monotonic_increasing_lexsorted(lexsorted_two_level_string_multiindex):
    # 使用 lexsorted_two_level_string_multiindex 变量表示词典序排列的 MultiIndex
    mi = lexsorted_two_level_string_multiindex
    # 断言 MultiIndex 不是单调递增的
    assert mi.is_monotonic_increasing is False
    # 将 MultiIndex 的值转换为 Index 对象后，断言其不是单调递增的
    assert Index(mi.values).is_monotonic_increasing is False
    # 断言 MultiIndex 不是严格单调递增的
    assert mi._is_strictly_monotonic_increasing is False
    # 将 MultiIndex 的值转换为 Index 对象后，断言其不是严格单调递增的
    assert Index(mi.values)._is_strictly_monotonic_increasing is False


# 定义测试函数，测试 MultiIndex 是否单调递增
def test_is_monotonic_increasing():
    # 创建一个由两个 np.arange(10) 的笛卡尔积构成的 MultiIndex，命名为 "one" 和 "two"
    i = MultiIndex.from_product([np.arange(10), np.arange(10)], names=["one", "two"])
    # 断言 MultiIndex 是单调递增的
    assert i.is_monotonic_increasing is True
    # 断言 MultiIndex 是严格单调递增的
    assert i._is_strictly_monotonic_increasing is True
    # 将 MultiIndex 的值转换为 Index 对象后，断言其是单调递增的
    assert Index(i.values).is_monotonic_increasing is True
    # 再次断言 MultiIndex 是严格单调递增的
    assert i._is_strictly_monotonic_increasing is True

    # 创建一个由递减序列和递增序列的笛卡尔积构成的 MultiIndex，命名为 "one" 和 "two"
    i = MultiIndex.from_product(
        [np.arange(10, 0, -1), np.arange(10)], names=["one", "two"]
    )
    # 断言 MultiIndex 不是单调递增的
    assert i.is_monotonic_increasing is False
    # 断言 MultiIndex 不是严格单调递增的
    assert i._is_strictly_monotonic_increasing is False
    # 将 MultiIndex 的值转换为 Index 对象后，断言其不是单调递增的
    assert Index(i.values).is_monotonic_increasing is False
    # 将 MultiIndex 的值转换为 Index 对象后，断言其不是严格单调递增的
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    # 创建一个由两个递增序列的笛卡尔积和包含 NaN 的序列构成的 MultiIndex
    i = MultiIndex.from_product(
        [np.arange(10), np.arange(10, 0, -1)], names=["one", "two"]
    )
    # 断言 MultiIndex 不是单调递增的
    assert i.is_monotonic_increasing is False
    # 断言 MultiIndex 不是严格单调递增的
    assert i._is_strictly_monotonic_increasing is False
    # 将 MultiIndex 的值转换为 Index 对象后，断言其不是单调递增的
    assert Index(i.values).is_monotonic_increasing is False
    # 将 MultiIndex 的值转换为 Index 对象后，断言其不是严格单调递增的
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    # 创建一个由两个列表组成的 MultiIndex，包含字符串和 NaN
    i = MultiIndex.from_product([[1.0, np.nan, 2.0], ["a", "b", "c"]])
    # 断言 MultiIndex 不是单调递增的
    assert i.is_monotonic_increasing is False
    # 断言 MultiIndex 不是严格单调递增的
    assert i._is_strictly_monotonic_increasing is False
    # 将 MultiIndex 的值转换为 Index 对象后，断言其不是单调递增的
    assert Index(i.values).is_monotonic_increasing is False
    # 将 MultiIndex 的值转换为 Index 对象后，断言其不是严格单调递增的
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    # 创建一个预定义的 MultiIndex，由 levels 和 codes 参数指定
    i = MultiIndex(
        levels=[["bar", "baz", "foo", "qux"], ["mom", "next", "zenith"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    # 断言 MultiIndex 是单调递增的
    assert i.is_monotonic_increasing is True
    # 将 MultiIndex 的值转换为 Index 对象后，断言其是单调递增的
    assert Index(i.values).is_monotonic_increasing is True
    # 断言 MultiIndex 是严格单调递增的
    assert i._is_strictly_monotonic_increasing is True
    # 将 MultiIndex 的值转换为 Index 对象后，断言其是严格单调递增的
    assert Index(i.values)._is_strictly_monotonic_increasing is True

    # 创建一个包含类型不一致的 MultiIndex，其中包含一个无效的代码索引
    i = MultiIndex(
        levels=[
            [1, 2, 3, 4],
            [
                "gb00b03mlx29",
                "lu0197800237",
                "nl0000289783",
                "nl0000289965",
                "nl0000301109",
            ],
        ],
        codes=[[0, 1, 1, 2, 2, 2, 3], [4, 2, 0, 0, 1, 3, -1]],
        names=["household_id", "asset_id"],
    )
    # 断言 MultiIndex 不是单调递增的
    assert i.is_monotonic_increasing is False
    # 断言 MultiIndex 不是严格单调递增的
    assert i._is_strictly_monotonic_increasing is False

    # 创建一个空的 MultiIndex
    i = MultiIndex.from_arrays([[], []])
    # 断言空的 MultiIndex 是单调递增的
    assert i.is_monotonic_increasing is True
    # 将 MultiIndex 的值转换为 Index 对象后，断言其是单调递增的
    assert Index(i.values).is_monotonic_increasing is True
    # 断言空的 MultiIndex 是严格单调递增的
    assert i._is_strictly_monotonic_increasing is True
    # 断言语句，用于验证条件是否为真，如果不是，则抛出异常
    assert Index(i.values)._is_strictly_monotonic_increasing is True
def test_is_monotonic_decreasing():
    # 创建一个 MultiIndex 对象，包含递减的整数索引对
    i = MultiIndex.from_product(
        [np.arange(9, -1, -1), np.arange(9, -1, -1)], names=["one", "two"]
    )
    # 断言索引是否单调递减
    assert i.is_monotonic_decreasing is True
    # 断言严格单调递减属性为真
    assert i._is_strictly_monotonic_decreasing is True
    # 将 MultiIndex 的值转为 Index 对象后，断言其是否单调递减
    assert Index(i.values).is_monotonic_decreasing is True
    # 再次断言严格单调递减属性为真
    assert i._is_strictly_monotonic_decreasing is True

    # 创建一个 MultiIndex 对象，包含不递减的整数索引对
    i = MultiIndex.from_product(
        [np.arange(10), np.arange(10, 0, -1)], names=["one", "two"]
    )
    # 断言索引是否不单调递减
    assert i.is_monotonic_decreasing is False
    # 断言严格单调递减属性为假
    assert i._is_strictly_monotonic_decreasing is False
    # 将 MultiIndex 的值转为 Index 对象后，断言其是否不单调递减
    assert Index(i.values).is_monotonic_decreasing is False
    # 再次断言严格单调递减属性为假
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    # 创建一个 MultiIndex 对象，包含不递减的整数索引对
    i = MultiIndex.from_product(
        [np.arange(10, 0, -1), np.arange(10)], names=["one", "two"]
    )
    # 断言索引是否不单调递减
    assert i.is_monotonic_decreasing is False
    # 断言严格单调递减属性为假
    assert i._is_strictly_monotonic_decreasing is False
    # 将 MultiIndex 的值转为 Index 对象后，断言其是否不单调递减
    assert Index(i.values).is_monotonic_decreasing is False
    # 再次断言严格单调递减属性为假
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    # 创建一个 MultiIndex 对象，包含混合类型值对
    i = MultiIndex.from_product([[2.0, np.nan, 1.0], ["c", "b", "a"]])
    # 断言索引是否不单调递减
    assert i.is_monotonic_decreasing is False
    # 断言严格单调递减属性为假
    assert i._is_strictly_monotonic_decreasing is False
    # 将 MultiIndex 的值转为 Index 对象后，断言其是否不单调递减
    assert Index(i.values).is_monotonic_decreasing is False
    # 再次断言严格单调递减属性为假
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    # 创建一个 MultiIndex 对象，包含字符串排序的值对
    i = MultiIndex(
        levels=[["qux", "foo", "baz", "bar"], ["three", "two", "one"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    # 断言索引是否不单调递减
    assert i.is_monotonic_decreasing is False
    # 将 MultiIndex 的值转为 Index 对象后，断言其是否不单调递减
    assert Index(i.values).is_monotonic_decreasing is False
    # 断言严格单调递减属性为假
    assert i._is_strictly_monotonic_decreasing is False
    # 再次断言严格单调递减属性为假
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    # 创建一个 MultiIndex 对象，包含字符串排序的值对
    i = MultiIndex(
        levels=[["qux", "foo", "baz", "bar"], ["zenith", "next", "mom"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    # 断言索引是否单调递减
    assert i.is_monotonic_decreasing is True
    # 将 MultiIndex 的值转为 Index 对象后，断言其是否单调递减
    assert Index(i.values).is_monotonic_decreasing is True
    # 断言严格单调递减属性为真
    assert i._is_strictly_monotonic_decreasing is True
    # 再次断言严格单调递减属性为真
    assert Index(i.values)._is_strictly_monotonic_decreasing is True

    # 创建一个 MultiIndex 对象，包含混合类型值对，会触发 TypeError
    i = MultiIndex(
        levels=[
            [4, 3, 2, 1],
            [
                "nl0000301109",
                "nl0000289965",
                "nl0000289783",
                "lu0197800237",
                "gb00b03mlx29",
            ],
        ],
        codes=[[0, 1, 1, 2, 2, 2, 3], [4, 2, 0, 0, 1, 3, -1]],
        names=["household_id", "asset_id"],
    )
    # 断言索引是否不单调递减
    assert i.is_monotonic_decreasing is False
    # 断言严格单调递减属性为假
    assert i._is_strictly_monotonic_decreasing is False

    # 创建一个空 MultiIndex 对象
    i = MultiIndex.from_arrays([[], []])
    # 断言索引是否单调递减
    assert i.is_monotonic_decreasing is True
    # 将 MultiIndex 的值转为 Index 对象后，断言其是否单调递减
    assert Index(i.values).is_monotonic_decreasing is True
    # 断言严格单调递减属性为真
    assert i._is_strictly_monotonic_decreasing is True
    # 断言语句，用于验证索引对象 i 的值是否严格单调递减，并断定结果为 True。
    assert Index(i.values)._is_strictly_monotonic_decreasing is True
# 定义测试函数，用于检查 MultiIndex 对象是否严格单调递增
def test_is_strictly_monotonic_increasing():
    # 创建一个 MultiIndex 对象，包含两级索引，每级索引的取值和编码
    idx = MultiIndex(
        levels=[["bar", "baz"], ["mom", "next"]],
        codes=[[0, 0, 1, 1], [0, 0, 0, 1]]
    )
    # 断言索引对象是否单调递增，预期为 True
    assert idx.is_monotonic_increasing is True
    # 断言索引对象是否严格单调递增，预期为 False
    assert idx._is_strictly_monotonic_increasing is False


# 定义测试函数，用于检查 MultiIndex 对象是否严格单调递减
def test_is_strictly_monotonic_decreasing():
    # 创建一个 MultiIndex 对象，包含两级索引，每级索引的取值和编码
    idx = MultiIndex(
        levels=[["baz", "bar"], ["next", "mom"]],
        codes=[[0, 0, 1, 1], [0, 0, 0, 1]]
    )
    # 断言索引对象是否单调递减，预期为 True
    assert idx.is_monotonic_decreasing is True
    # 断言索引对象是否严格单调递减，预期为 False
    assert idx._is_strictly_monotonic_decreasing is False


# 定义参数化测试函数，用于检查带有 NaN 值的情况下索引对象是否单调
@pytest.mark.parametrize("attr", ["is_monotonic_increasing", "is_monotonic_decreasing"])
@pytest.mark.parametrize(
    "values",
    [[(np.nan,), (1,), (2,)], [(1,), (np.nan,), (2,)], [(1,), (2,), (np.nan,)]],
)
def test_is_monotonic_with_nans(values, attr):
    # 创建一个 MultiIndex 对象，根据给定的值组成元组，并指定索引的名称为 "test"
    idx = MultiIndex.from_tuples(values, names=["test"])
    # 断言指定属性在含有 NaN 的情况下是否为 False
    assert getattr(idx, attr) is False
```