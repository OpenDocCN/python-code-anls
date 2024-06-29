# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_reshape.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime
# 导入 zoneinfo 模块，用于时区信息处理
import zoneinfo

# 导入 numpy 库，常用于数值计算
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas 库，用于数据分析和处理
import pandas as pd
# 从 pandas 库中导入 Index 和 MultiIndex 类
from pandas import (
    Index,
    MultiIndex,
)
# 导入 pandas._testing 模块，用于测试辅助函数
import pandas._testing as tm


def test_insert(idx):
    # 在索引 idx 的开头插入元素 ("bar", "two")
    new_index = idx.insert(0, ("bar", "two"))
    # 断言新索引 new_index 的层级结构与 idx 相同
    assert new_index.equal_levels(idx)
    # 断言新索引的第一个元素为 ("bar", "two")
    assert new_index[0] == ("bar", "two")

    # 在索引 idx 的开头插入元素 ("abc", "three")
    new_index = idx.insert(0, ("abc", "three"))

    # 预期新索引的第一层级为扩展后的列表，名称为 "first"
    exp0 = Index(list(idx.levels[0]) + ["abc"], name="first")
    tm.assert_index_equal(new_index.levels[0], exp0)
    # 断言新索引的名称为 ["first", "second"]
    assert new_index.names == ["first", "second"]

    # 预期新索引的第二层级为扩展后的列表，名称为 "second"
    exp1 = Index(list(idx.levels[1]) + ["three"], name="second")
    tm.assert_index_equal(new_index.levels[1], exp1)
    # 断言新索引的第一个元素为 ("abc", "three")
    assert new_index[0] == ("abc", "three")

    # 插入的元素长度不符合索引的层级数，预期抛出 ValueError 异常
    msg = "Item must have length equal to number of levels"
    with pytest.raises(ValueError, match=msg):
        idx.insert(0, ("foo2",))

    # 创建 DataFrame，并设置索引为 ["1st", "2nd"]
    left = pd.DataFrame([["a", "b", 0], ["b", "d", 1]], columns=["1st", "2nd", "3rd"])
    left.set_index(["1st", "2nd"], inplace=True)
    # 深拷贝 left 的 "3rd" 列到 ts
    ts = left["3rd"].copy(deep=True)

    # 修改 left 中的特定索引处的 "3rd" 值
    left.loc[("b", "x"), "3rd"] = 2
    left.loc[("b", "a"), "3rd"] = -1
    left.loc[("b", "b"), "3rd"] = 3
    left.loc[("a", "x"), "3rd"] = 4
    left.loc[("a", "w"), "3rd"] = 5
    left.loc[("a", "a"), "3rd"] = 6

    # 修改 ts 中的特定索引处的值
    ts.loc[("b", "x")] = 2
    ts.loc["b", "a"] = -1
    ts.loc[("b", "b")] = 3
    ts.loc["a", "x"] = 4
    ts.loc[("a", "w")] = 5
    ts.loc["a", "a"] = 6

    # 创建预期的 right DataFrame，设置相同的索引 ["1st", "2nd"]
    right = pd.DataFrame(
        [
            ["a", "b", 0],
            ["b", "d", 1],
            ["b", "x", 2],
            ["b", "a", -1],
            ["b", "b", 3],
            ["a", "x", 4],
            ["a", "w", 5],
            ["a", "a", 6],
        ],
        columns=["1st", "2nd", "3rd"],
    )
    right.set_index(["1st", "2nd"], inplace=True)
    # 检查 left 和 right 的数据是否相等，忽略数据类型的检查
    # 由于插入 NaN 的中间操作，数据类型可能发生变化
    tm.assert_frame_equal(left, right, check_dtype=False)
    # 检查 ts 和 right["3rd"] 的 Series 是否相等
    tm.assert_series_equal(ts, right["3rd"])


def test_insert2():
    # 测试用例 GH9250
    idx = (
        [("test1", i) for i in range(5)]
        + [("test2", i) for i in range(6)]
        + [("test", 17), ("test", 18)]
    )

    # 创建具有 MultiIndex 的 Series left
    left = pd.Series(np.linspace(0, 10, 11), MultiIndex.from_tuples(idx[:-2]))

    # 修改 left 中特定索引处的值
    left.loc[("test", 17)] = 11
    left.loc[("test", 18)] = 12

    # 创建预期的 Series right，具有相同的 MultiIndex
    right = pd.Series(np.linspace(0, 12, 13), MultiIndex.from_tuples(idx))

    # 检查 left 和 right 的 Series 是否相等
    tm.assert_series_equal(left, right)


def test_append(idx):
    # 将 idx 的前三个元素与其余元素连接，检查结果是否与 idx 相等
    result = idx[:3].append(idx[3:])
    assert result.equals(idx)

    # 将 idx 划分为三个子索引，分别进行连接，检查结果是否与 idx 相等
    foos = [idx[:1], idx[1:3], idx[3:]]
    result = foos[0].append(foos[1:])
    assert result.equals(idx)

    # 将空列表追加到 idx，检查结果是否与 idx 相等
    result = idx.append([])
    assert result.equals(idx)


def test_append_index():
    # 创建 Index 对象 idx1、idx2 和 idx3
    idx1 = Index([1.1, 1.2, 1.3])
    idx2 = pd.date_range("2011-01-01", freq="D", periods=3, tz="Asia/Tokyo")
    idx3 = Index(["A", "B", "C"])

    # 创建具有两个层级的 MultiIndex 对象 midx_lv2
    midx_lv2 = MultiIndex.from_arrays([idx1, idx2])
    # 使用给定的 idx1, idx2, idx3 创建一个三级多重索引对象
    midx_lv3 = MultiIndex.from_arrays([idx1, idx2, idx3])

    # 将 idx1 追加到 midx_lv2 中，生成新的索引对象 result
    result = idx1.append(midx_lv2)

    # 为了解决 gh-7112，创建一个代表 "Asia/Tokyo" 时区的 zoneinfo.ZoneInfo 对象
    tz = zoneinfo.ZoneInfo("Asia/Tokyo")
    
    # 创建预期的元组列表 expected_tuples，每个元组包含浮点数和带有时区信息的日期时间对象
    expected_tuples = [
        (1.1, datetime(2011, 1, 1, tzinfo=tz)),
        (1.2, datetime(2011, 1, 2, tzinfo=tz)),
        (1.3, datetime(2011, 1, 3, tzinfo=tz)),
    ]
    
    # 创建预期的索引对象 expected，包括浮点数和上述元组列表的所有元素
    expected = Index([1.1, 1.2, 1.3] + expected_tuples)
    
    # 断言 result 和 expected 的索引是否相等
    tm.assert_index_equal(result, expected)

    # 将 idx1 追加到 midx_lv2 中，生成新的索引对象 result
    result = midx_lv2.append(idx1)
    
    # 创建预期的索引对象 expected，包括上述元组列表和浮点数的所有元素
    expected = Index(expected_tuples + [1.1, 1.2, 1.3])
    
    # 断言 result 和 expected 的索引是否相等
    tm.assert_index_equal(result, expected)

    # 将 midx_lv2 追加到自身，生成新的多级索引对象 result
    result = midx_lv2.append(midx_lv2)
    
    # 创建预期的多级索引对象 expected，包括 idx1 和 idx2 各自追加自身后的结果
    expected = MultiIndex.from_arrays([idx1.append(idx1), idx2.append(idx2)])
    
    # 断言 result 和 expected 的索引是否相等
    tm.assert_index_equal(result, expected)

    # 将 midx_lv2 追加到 midx_lv3 中，生成新的多级索引对象 result
    result = midx_lv2.append(midx_lv3)
    
    # 断言 result 和 expected 的索引是否相等
    tm.assert_index_equal(result, expected)

    # 将 midx_lv3 追加到 midx_lv2 中，生成新的索引对象 result
    result = midx_lv3.append(midx_lv2)
    
    # 创建预期的索引对象 expected，包括浮点数、带时区信息的日期时间对象和字符 "A", "B", "C" 的组合
    expected = Index._simple_new(
        np.array(
            [
                (1.1, datetime(2011, 1, 1, tzinfo=tz), "A"),
                (1.2, datetime(2011, 1, 2, tzinfo=tz), "B"),
                (1.3, datetime(2011, 1, 3, tzinfo=tz), "C"),
            ]
            + expected_tuples,
            dtype=object,
        ),
        None,
    )
    
    # 断言 result 和 expected 的索引是否相等
    tm.assert_index_equal(result, expected)
@pytest.mark.parametrize("name, exp", [("b", "b"), ("c", None)])
# 使用 pytest 的 parametrize 装饰器，指定参数化测试的参数和期望值
def test_append_names_match(name, exp):
    # GH#48288
    # 创建一个 MultiIndex 对象 midx，由两个数组组成，分别是 [1, 2] 和 [3, 4]，设置列名为 ["a", "b"]
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    # 创建另一个 MultiIndex 对象 midx2，由两个数组组成，分别是 [3] 和 [5]，设置列名为 ["a", name]
    midx2 = MultiIndex.from_arrays([[3], [5]], names=["a", name])
    # 将 midx2 追加到 midx 中
    result = midx.append(midx2)
    # 创建期望的 MultiIndex 对象 expected，由两个数组组成，第一个数组包含 [1, 2, 3]，第二个数组包含 [3, 4, 5]，设置列名为 ["a", exp]
    expected = MultiIndex.from_arrays([[1, 2, 3], [3, 4, 5]], names=["a", exp])
    # 断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)


def test_append_names_dont_match():
    # GH#48288
    # 创建一个 MultiIndex 对象 midx，由两个数组组成，分别是 [1, 2] 和 [3, 4]，设置列名为 ["a", "b"]
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    # 创建另一个 MultiIndex 对象 midx2，由两个数组组成，分别是 [3] 和 [5]，设置列名为 ["x", "y"]
    midx2 = MultiIndex.from_arrays([[3], [5]], names=["x", "y"])
    # 将 midx2 追加到 midx 中
    result = midx.append(midx2)
    # 创建期望的 MultiIndex 对象 expected，没有设置列名
    expected = MultiIndex.from_arrays([[1, 2, 3], [3, 4, 5]], names=None)
    # 断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)


def test_append_overlapping_interval_levels():
    # GH 54934
    # 创建一个 IntervalIndex 对象 ivl1，从断点 [0.0, 1.0, 2.0] 创建
    ivl1 = pd.IntervalIndex.from_breaks([0.0, 1.0, 2.0])
    # 创建另一个 IntervalIndex 对象 ivl2，从断点 [0.5, 1.5, 2.5] 创建
    ivl2 = pd.IntervalIndex.from_breaks([0.5, 1.5, 2.5])
    # 使用 ivl1 的笛卡尔积创建 MultiIndex 对象 mi1
    mi1 = MultiIndex.from_product([ivl1, ivl1])
    # 使用 ivl2 的笛卡尔积创建 MultiIndex 对象 mi2
    mi2 = MultiIndex.from_product([ivl2, ivl2])
    # 将 mi2 追加到 mi1 中
    result = mi1.append(mi2)
    # 创建期望的 MultiIndex 对象 expected，包含多个元组，表示重叠的区间
    expected = MultiIndex.from_tuples(
        [
            (pd.Interval(0.0, 1.0), pd.Interval(0.0, 1.0)),
            (pd.Interval(0.0, 1.0), pd.Interval(1.0, 2.0)),
            (pd.Interval(1.0, 2.0), pd.Interval(0.0, 1.0)),
            (pd.Interval(1.0, 2.0), pd.Interval(1.0, 2.0)),
            (pd.Interval(0.5, 1.5), pd.Interval(0.5, 1.5)),
            (pd.Interval(0.5, 1.5), pd.Interval(1.5, 2.5)),
            (pd.Interval(1.5, 2.5), pd.Interval(0.5, 1.5)),
            (pd.Interval(1.5, 2.5), pd.Interval(1.5, 2.5)),
        ]
    )
    # 断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)


def test_repeat():
    # 设置重复次数
    reps = 2
    # 创建一个数字列表
    numbers = [1, 2, 3]
    # 创建一个名称数组
    names = np.array(["foo", "bar"])

    # 使用 numbers 和 names 的笛卡尔积创建 MultiIndex 对象 m，并设置列名为 names
    m = MultiIndex.from_product([numbers, names], names=names)
    # 创建期望的 MultiIndex 对象 expected，重复 names 中的每个元素 reps 次
    expected = MultiIndex.from_product([numbers, names.repeat(reps)], names=names)
    # 断言 m 重复 reps 次后与 expected 相等
    tm.assert_index_equal(m.repeat(reps), expected)


def test_insert_base(idx):
    # 提取 idx 中第 1 到第 4 个元素作为结果 result
    result = idx[1:4]

    # 测试第 0 个元素是否可以插入到 result 的开头
    assert idx[0:4].equals(result.insert(0, idx[0]))


def test_delete_base(idx):
    # 期望的结果是删除 idx 的第一个元素后得到 expected
    expected = idx[1:]
    # 删除 idx 的第一个元素，得到结果 result
    result = idx.delete(0)
    # 断言 result 和 expected 相等
    assert result.equals(expected)
    # 断言 result 的名称与 expected 的名称相同
    assert result.name == expected.name

    # 期望的结果是删除 idx 的最后一个元素后得到 expected
    expected = idx[:-1]
    # 删除 idx 的最后一个元素，得到结果 result
    result = idx.delete(-1)
    # 断言 result 和 expected 相等
    assert result.equals(expected)
    # 断言 result 的名称与 expected 的名称相同
    assert result.name == expected.name

    # 测试删除超出索引边界的情况
    msg = "index 6 is out of bounds for axis 0 with size 6"
    with pytest.raises(IndexError, match=msg):
        # 尝试删除超出 idx 大小的索引，应该引发 IndexError 异常并匹配指定消息
        idx.delete(len(idx))
```