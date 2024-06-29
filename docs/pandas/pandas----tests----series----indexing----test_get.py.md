# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_get.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库，并从中导入特定的模块和函数
import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Series,
    date_range,
)
# 导入 pandas 内部的测试工具模块
import pandas._testing as tm


# 定义测试函数 test_get
def test_get():
    # GH 6383
    # 创建一个 Series 对象，其中包含一维的 numpy 数组
    s = Series(
        np.array(
            [
                43,
                48,
                60,
                48,
                50,
                51,
                50,
                45,
                57,
                48,
                56,
                45,
                51,
                39,
                55,
                43,
                54,
                52,
                51,
                54,
            ]
        )
    )

    # 使用 get 方法获取索引为 25 的元素，默认值为 0
    result = s.get(25, 0)
    expected = 0
    # 断言结果与预期相符
    assert result == expected

    # 创建带有索引的 Series 对象
    s = Series(
        np.array(
            [
                43,
                48,
                60,
                48,
                50,
                51,
                50,
                45,
                57,
                48,
                56,
                45,
                51,
                39,
                55,
                43,
                54,
                52,
                51,
                54,
            ]
        ),
        index=Index(
            [
                25.0,
                36.0,
                49.0,
                64.0,
                81.0,
                100.0,
                121.0,
                144.0,
                169.0,
                196.0,
                1225.0,
                1296.0,
                1369.0,
                1444.0,
                1521.0,
                1600.0,
                1681.0,
                1764.0,
                1849.0,
                1936.0,
            ],
            dtype=np.float64,
        ),
    )

    # 使用 get 方法获取索引为 25 的元素，默认值为 0
    result = s.get(25, 0)
    expected = 43
    # 断言结果与预期相符
    assert result == expected

    # GH 7407
    # 使用布尔访问器创建 DataFrame 对象
    df = pd.DataFrame({"i": [0] * 3, "b": [False] * 3})
    # 对列 'i' 进行值计数
    vc = df.i.value_counts()
    # 使用 get 方法获取键为 99 的值，默认为 "Missing"
    result = vc.get(99, default="Missing")
    assert result == "Missing"

    # 对列 'b' 进行值计数
    vc = df.b.value_counts()
    # 使用 get 方法获取键为 False 的值，默认为 "Missing"
    result = vc.get(False, default="Missing")
    assert result == 3

    # 使用 get 方法获取键为 True 的值，默认为 "Missing"
    result = vc.get(True, default="Missing")
    assert result == "Missing"


# 定义测试函数 test_get_nan，参数为 float_numpy_dtype
def test_get_nan(float_numpy_dtype):
    # GH 8569
    # 创建一个 Index 对象，并将其转换为 Series 对象
    s = Index(range(10), dtype=float_numpy_dtype).to_series()
    # 使用 get 方法获取键为 np.nan 的值，预期为 None
    assert s.get(np.nan) is None
    # 使用 get 方法获取键为 np.nan 的值，默认为 "Missing"
    assert s.get(np.nan, default="Missing") == "Missing"


# 定义测试函数 test_get_nan_multiple，参数为 float_numpy_dtype
def test_get_nan_multiple(float_numpy_dtype):
    # GH 8569
    # 确保上面修复的 "test_get_nan" 没有破坏具有多个元素的 get 方法
    # 创建一个 Index 对象，并将其转换为 Series 对象
    s = Index(range(10), dtype=float_numpy_dtype).to_series()

    # 测试多个键的情况
    idx = [2, 30]
    assert s.get(idx) is None

    idx = [2, np.nan]
    assert s.get(idx) is None

    # GH 17295 - 所有缺失的键
    idx = [20, 30]
    assert s.get(idx) is None

    idx = [np.nan, np.nan]
    assert s.get(idx) is None


# 定义测试函数 test_get_with_default
def test_get_with_default():
    # GH#7725
    # 创建一个包含字符串的列表
    d0 = ["a", "b", "c", "d"]
    # 创建一个包含四个 int64 类型元素的 NumPy 数组
    d1 = np.arange(4, dtype="int64")

    # 对于两组数据 (d0, d1) 和 (d1, d0)，分别执行以下操作
    for data, index in ((d0, d1), (d1, d0)):
        # 使用给定的数据和索引创建一个 Pandas Series 对象
        s = Series(data, index=index)

        # 遍历索引和数据，使用 s.get 方法进行多种断言操作
        for i, d in zip(index, data):
            # 断言：获取索引 i 对应的值应该等于数据 d
            assert s.get(i) == d
            # 断言：获取索引 i 对应的值，如果不存在则返回默认值 d，结果应该等于数据 d
            assert s.get(i, d) == d
            # 断言：获取索引 i 对应的值，如果不存在则返回默认值 "z"，结果应该等于数据 d
            assert s.get(i, "z") == d

            # 断言：获取不存在的索引 "e"，返回默认值 "z"
            assert s.get("e", "z") == "z"
            # 断言：获取不存在的索引 "e"，返回默认值 "e"
            assert s.get("e", "e") == "e"

            # 断言：获取不存在的索引 10，返回默认值 "z"
            assert s.get(10, "z") == "z"
            # 断言：获取不存在的索引 10，返回默认值 10
            assert s.get(10, 10) == 10
# 使用 pytest.mark.parametrize 装饰器标记这个测试函数，参数化测试输入数组 arr
@pytest.mark.parametrize(
    "arr",
    [
        np.random.default_rng(2).standard_normal(10),  # 生成一个包含10个标准正态分布随机数的数组
        DatetimeIndex(date_range("2020-01-01", periods=10), name="a").tz_localize(
            tz="US/Eastern"
        ),  # 生成一个具有时区信息的日期索引对象
    ],
)
def test_get_with_ea(arr):
    # GH#21260
    # 创建一个 Series 对象 ser，使用数组 arr 和自定义索引
    ser = Series(arr, index=[2 * i for i in range(len(arr))])
    # 断言 ser 对象使用 get 方法获取索引为 4 的值等于使用 iloc 方法获取索引为 2 的值
    assert ser.get(4) == ser.iloc[2]

    # 使用 get 方法获取索引为 4 和 6 的值，与使用 iloc 方法获取索引为 2 和 3 的值进行比较
    result = ser.get([4, 6])
    expected = ser.iloc[[2, 3]]
    tm.assert_series_equal(result, expected)

    # 使用 get 方法获取切片索引为 slice(2) 的值，与使用 iloc 方法获取索引为 0 和 1 的值进行比较
    result = ser.get(slice(2))
    expected = ser.iloc[[0, 1]]
    tm.assert_series_equal(result, expected)

    # 断言 ser 对象使用 get 方法获取索引为 -1 的值为 None
    assert ser.get(-1) is None
    # 断言 ser 对象使用 get 方法获取最大索引 + 1 的值为 None
    assert ser.get(ser.index.max() + 1) is None

    # 创建一个新的 Series 对象 ser，使用数组 arr 的前6个元素和自定义索引
    ser = Series(arr[:6], index=list("abcdef"))
    # 断言 ser 对象使用 get 方法获取索引为 'c' 的值等于使用 iloc 方法获取索引为 2 的值
    assert ser.get("c") == ser.iloc[2]

    # 使用 get 方法获取从 'b' 到 'd' 的切片索引的值，与使用 iloc 方法获取索引为 1、2 和 3 的值进行比较
    result = ser.get(slice("b", "d"))
    expected = ser.iloc[[1, 2, 3]]
    tm.assert_series_equal(result, expected)

    # 断言 ser 对象使用 get 方法获取索引为 'Z' 的值为 None
    result = ser.get("Z")
    assert result is None

    # 断言 ser 对象中使用 get 方法获取整数索引为 4 的值为 None，表明整数被视为标签
    assert ser.get(4) is None
    # 断言 ser 对象中使用 get 方法获取索引为 -1 和长度超出索引的值为 None
    assert ser.get(-1) is None
    assert ser.get(len(ser)) is None

    # GH#21257
    # 创建一个新的 Series 对象 ser，使用数组 arr
    ser = Series(arr)
    # 对 ser 进行切片操作，步长为2，得到 ser2
    ser2 = ser[::2]
    # 断言 ser2 对象使用 get 方法获取索引为 1 的值为 None
    assert ser2.get(1) is None


# 定义测试函数 test_getitem_get，接受两个 Series 对象 string_series 和 object_series
def test_getitem_get(string_series, object_series):
    # 遍历对象数组 [string_series, object_series]
    for obj in [string_series, object_series]:
        # 获取对象 obj 的第5个索引值 idx
        idx = obj.index[5]

        # 断言对象 obj 使用索引 idx 的普通获取方式等于使用 get 方法获取的值
        assert obj[idx] == obj.get(idx)
        # 断言对象 obj 使用索引 idx 的普通获取方式等于使用 iloc 方法获取索引为 5 的值
        assert obj[idx] == obj.iloc[5]

    # 断言 string_series 对象使用 get 方法获取索引为 -1 的值为 None
    assert string_series.get(-1) is None
    # 断言 string_series 对象使用 iloc 方法获取索引为 5 的值等于使用 get 方法获取索引为 5 的值
    assert string_series.iloc[5] == string_series.get(string_series.index[5])


# 定义测试函数 test_get_none
def test_get_none():
    # GH#5652
    # 创建一个空的 Series 对象 s1 和带有索引 'abc' 的 Series 对象 s2
    s1 = Series(dtype=object)
    s2 = Series(dtype=object, index=list("abc"))
    # 遍历对象数组 [s1, s2]
    for s in [s1, s2]:
        # 使用 get 方法获取空索引 None 的结果，断言结果为 None
        result = s.get(None)
        assert result is None
```