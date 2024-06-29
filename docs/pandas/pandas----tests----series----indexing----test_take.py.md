# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_take.py`

```
import pytest  # 导入 pytest 库

import pandas as pd  # 导入 pandas 库并使用别名 pd
from pandas import Series  # 从 pandas 中导入 Series 类
import pandas._testing as tm  # 导入 pandas 内部测试模块


def test_take_validate_axis():
    # GH#51022: 此测试用例是为了验证在 Series 对象中使用不存在的轴名称 "foo" 时是否会抛出 ValueError 异常

    ser = Series([-1, 5, 6, 2, 4])  # 创建一个 Series 对象

    msg = "No axis named foo for object type Series"
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并验证异常消息是否包含指定文本
    with pytest.raises(ValueError, match=msg):
        ser.take([1, 2], axis="foo")


def test_take():
    ser = Series([-1, 5, 6, 2, 4])  # 创建一个 Series 对象

    actual = ser.take([1, 3, 4])  # 从 Series 中获取指定位置的元素
    expected = Series([5, 2, 4], index=[1, 3, 4])  # 预期的 Series 结果
    tm.assert_series_equal(actual, expected)  # 使用测试模块中的 assert_series_equal 函数进行结果比较

    actual = ser.take([-1, 3, 4])  # 从 Series 中获取指定位置的元素，包括负索引
    expected = Series([4, 2, 4], index=[4, 3, 4])  # 预期的 Series 结果
    tm.assert_series_equal(actual, expected)  # 使用测试模块中的 assert_series_equal 函数进行结果比较

    msg = "indices are out-of-bounds"
    # 使用 pytest.raises 检查是否会抛出 IndexError 异常，并验证异常消息是否包含指定文本
    with pytest.raises(IndexError, match=msg):
        ser.take([1, 10])
    with pytest.raises(IndexError, match=msg):
        ser.take([2, 5])


def test_take_categorical():
    # https://github.com/pandas-dev/pandas/issues/20664
    ser = Series(pd.Categorical(["a", "b", "c"]))  # 创建一个包含分类数据的 Series 对象
    result = ser.take([-2, -2, 0])  # 从 Series 中获取指定位置的元素，包括负索引
    expected = Series(
        pd.Categorical(["b", "b", "a"], categories=["a", "b", "c"]), index=[1, 1, 0]
    )  # 预期的 Series 结果，包含特定的分类和索引
    tm.assert_series_equal(result, expected)  # 使用测试模块中的 assert_series_equal 函数进行结果比较


def test_take_slice_raises():
    ser = Series([-1, 5, 6, 2, 4])  # 创建一个 Series 对象

    msg = "Series.take requires a sequence of integers, not slice"
    # 使用 pytest.raises 检查是否会抛出 TypeError 异常，并验证异常消息是否包含指定文本
    with pytest.raises(TypeError, match=msg):
        ser.take(slice(0, 3, 1))
```