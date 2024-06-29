# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_set_value.py`

```
# 导入datetime模块中的datetime类，用于处理日期和时间
from datetime import datetime

# 导入numpy库，并用np作为别名
import numpy as np

# 从pandas库中导入Index和Series类
from pandas import (
    Index,
    Series,
)

# 导入pandas的测试模块，使用别名tm
import pandas._testing as tm


# 定义一个测试函数，用于测试Series类的_set_value方法
def test_series_set_value():
    # GH#1561, GH#51363 表示GitHub上的两个issue编号，作为提醒和参考注释
    # 自版本3.0起，Index.insert方法中不进行推断

    # 创建包含两个datetime对象的列表
    dates = [datetime(2001, 1, 1), datetime(2001, 1, 2)]
    # 使用dates列表创建一个Index对象，dtype指定为object类型
    index = Index(dates, dtype=object)

    # 创建一个空的Series对象，dtype指定为object类型
    s = Series(dtype=object)
    # 在Series对象s中设置日期dates[0]对应的值为1.0
    s._set_value(dates[0], 1.0)
    # 在Series对象s中设置日期dates[1]对应的值为NaN（Not a Number）
    s._set_value(dates[1], np.nan)

    # 创建一个预期的Series对象，包含索引和对应的值
    expected = Series([1.0, np.nan], index=index)

    # 使用pandas测试模块tm中的assert_series_equal函数验证s和expected是否相等
    tm.assert_series_equal(s, expected)


# 定义一个测试函数，用于测试datetime_series对象的_set_value方法
def test_set_value_dt64(datetime_series):
    # 获取datetime_series的索引中第11个位置的值，并赋给idx变量
    idx = datetime_series.index[10]
    # 使用datetime_series的_set_value方法将idx位置的值设置为0，返回值为None
    res = datetime_series._set_value(idx, 0)
    # 断言res是否为None
    assert res is None
    # 断言datetime_series中idx位置的值是否为0
    assert datetime_series[idx] == 0


# 定义一个测试函数，用于测试string_series对象的_set_value方法
def test_set_value_str_index(string_series):
    # 创建string_series的副本，并赋给ser变量
    ser = string_series.copy()
    # 使用ser的_set_value方法将索引为"foobar"的位置的值设置为0，返回值为None
    res = ser._set_value("foobar", 0)
    # 断言res是否为None
    assert res is None
    # 断言ser的索引中最后一个值是否为"foobar"
    assert ser.index[-1] == "foobar"
    # 断言ser中索引为"foobar"的位置的值是否为0
    assert ser["foobar"] == 0

    # 创建string_series的副本，并赋给ser2变量
    ser2 = string_series.copy()
    # 使用ser2的loc属性，直接设置索引为"foobar"的位置的值为0
    ser2.loc["foobar"] = 0
    # 断言ser2的索引中最后一个值是否为"foobar"
    assert ser2.index[-1] == "foobar"
    # 断言ser2中索引为"foobar"的位置的值是否为0
    assert ser2["foobar"] == 0
```