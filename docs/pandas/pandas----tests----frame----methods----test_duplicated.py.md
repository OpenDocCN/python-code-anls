# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_duplicated.py`

```
import re  # 导入正则表达式模块
import sys  # 导入系统模块，用于设置递归限制

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

from pandas import (  # 从pandas库中导入以下对象
    DataFrame,  # 数据帧对象
    Series,  # 系列对象
    date_range,  # 日期范围生成器
)
import pandas._testing as tm  # 导入pandas测试工具模块


@pytest.mark.parametrize("subset", ["a", ["a"], ["a", "B"]])
def test_duplicated_with_misspelled_column_name(subset):
    # GH 19730
    # 创建一个包含三列的DataFrame对象
    df = DataFrame({"A": [0, 0, 1], "B": [0, 0, 1], "C": [0, 0, 1]})
    # 准备匹配的错误列名消息
    msg = re.escape("Index(['a'], dtype=")

    # 断言在DataFrame上使用错误的列名会引发KeyError异常，并匹配指定的错误消息
    with pytest.raises(KeyError, match=msg):
        df.duplicated(subset)


def test_duplicated_implemented_no_recursion():
    # gh-21524
    # 确保duplicated方法不使用可能在宽DataFrame上失败的递归实现
    df = DataFrame(np.random.default_rng(2).integers(0, 1000, (10, 1000)))
    # 获取当前递归限制
    rec_limit = sys.getrecursionlimit()
    try:
        # 临时设置递归限制为100
        sys.setrecursionlimit(100)
        # 调用DataFrame的duplicated方法
        result = df.duplicated()
    finally:
        # 恢复原始的递归限制
        sys.setrecursionlimit(rec_limit)

    # 断言结果为布尔Series对象，且数据类型为np.bool_
    assert isinstance(result, Series)
    assert result.dtype == np.bool_


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, False, True])),
        ("last", Series([True, True, False, False, False])),
        (False, Series([True, True, True, False, True])),
    ],
)
def test_duplicated_keep(keep, expected):
    # 创建一个包含两列的DataFrame对象
    df = DataFrame({"A": [0, 1, 1, 2, 0], "B": ["a", "b", "b", "c", "a"]})

    # 调用duplicated方法，指定keep参数
    result = df.duplicated(keep=keep)
    # 使用测试工具模块中的函数来断言结果与期望值的一致性
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(reason="GH#21720; nan/None falsely considered equal")
@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, False, True])),
        ("last", Series([True, True, False, False, False])),
        (False, Series([True, True, True, False, True])),
    ],
)
def test_duplicated_nan_none(keep, expected):
    # 创建一个包含两列的DataFrame对象，其中一列包含NaN和None值
    df = DataFrame({"C": [np.nan, 3, 3, None, np.nan], "x": 1}, dtype=object)

    # 调用duplicated方法，指定keep参数
    result = df.duplicated(keep=keep)
    # 使用测试工具模块中的函数来断言结果与期望值的一致性
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("subset", [None, ["A", "B"], "A"])
def test_duplicated_subset(subset, keep):
    # 创建一个包含三列的DataFrame对象，包括一列包含NaN和None值
    df = DataFrame(
        {
            "A": [0, 1, 1, 2, 0],
            "B": ["a", "b", "b", "c", "a"],
            "C": [np.nan, 3, 3, None, np.nan],
        }
    )

    if subset is None:
        subset = list(df.columns)
    elif isinstance(subset, str):
        # 需要DataFrame对象而不是Series对象，因此选择使用包含一个元素的列表而不是字符串
        subset = [subset]

    # 获取期望值，使用指定的keep和subset参数
    expected = df[subset].duplicated(keep=keep)
    # 调用duplicated方法，指定keep和subset参数
    result = df.duplicated(keep=keep, subset=subset)
    # 使用测试工具模块中的函数来断言结果与期望值的一致性
    tm.assert_series_equal(result, expected)


def test_duplicated_on_empty_frame():
    # GH 25184

    # 创建一个空的DataFrame对象，包含两列
    df = DataFrame(columns=["a", "b"])
    # 在空DataFrame上调用duplicated方法，指定列名
    dupes = df.duplicated("a")

    # 获取结果DataFrame，应该与原始DataFrame相等
    result = df[dupes]
    expected = df.copy()
    # 使用测试工具模块中的函数来断言结果DataFrame与期望值的一致性
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于验证日期时间处理的功能是否正确
def test_frame_datetime64_duplicated():
    # 生成一个日期范围，从"2010-07-01"到"2010-08-05"
    dates = date_range("2010-07-01", end="2010-08-05")

    # 创建一个数据框，包含一个名为"symbol"的列和一个"date"的日期列
    tst = DataFrame({"symbol": "AAA", "date": dates})
    # 检查数据框中是否有重复的行，根据"date"和"symbol"列进行判断
    result = tst.duplicated(["date", "symbol"])
    # 断言结果中所有的重复行都应为False（即没有重复）
    assert (-result).all()

    # 创建一个只包含"date"列的数据框
    tst = DataFrame({"date": dates})
    # 检查数据框中是否有重复的日期
    result = tst.date.duplicated()
    # 断言结果中所有的重复行都应为False（即没有重复）
    assert (-result).all()
```