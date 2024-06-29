# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimelike_\test_nat.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

# 从 Pandas 库中导入多个类和函数
from pandas import (
    DatetimeIndex,  # 日期时间索引类
    NaT,  # "Not a Time" 表示缺失日期时间值的常量
    PeriodIndex,  # 时期索引类
    TimedeltaIndex,  # 时间增量索引类
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


# 使用 pytest.mark.parametrize 装饰器，定义多组参数化测试数据
@pytest.mark.parametrize(
    "index_without_na",  # 参数化的测试数据名称
    [
        TimedeltaIndex(["1 days", "2 days"]),  # 时间增量索引，包含两个元素
        PeriodIndex(["2011-01-01", "2011-01-02"], freq="D"),  # 时期索引，包含两个日期
        DatetimeIndex(["2011-01-01", "2011-01-02"]),  # 日期时间索引，包含两个日期
        DatetimeIndex(["2011-01-01", "2011-01-02"], tz="UTC"),  # 包含时区信息的日期时间索引
    ],
)
def test_nat(index_without_na):
    # 创建空索引，即长度为0的原始索引
    empty_index = index_without_na[:0]

    # 复制原始索引，进行深拷贝，并设置第二个元素为 NaT（缺失时间）
    index_with_na = index_without_na.copy(deep=True)
    index_with_na._data[1] = NaT

    # 断言空索引的 _na_value 属性为 NaT
    assert empty_index._na_value is NaT
    # 断言带缺失值的索引的 _na_value 属性为 NaT
    assert index_with_na._na_value is NaT
    # 断言原始索引的 _na_value 属性为 NaT
    assert index_without_na._na_value is NaT

    # 将索引赋值给 idx 变量
    idx = index_without_na
    # 断言索引支持 NaN 值
    assert idx._can_hold_na

    # 使用测试模块中的函数，比较索引的 _isnan 属性与预期的布尔数组
    tm.assert_numpy_array_equal(idx._isnan, np.array([False, False]))
    # 断言索引不含有 NaN 值
    assert idx.hasnans is False

    # 将带缺失值的索引赋值给 idx 变量
    idx = index_with_na
    # 断言索引支持 NaN 值
    assert idx._can_hold_na

    # 使用测试模块中的函数，比较索引的 _isnan 属性与预期的布尔数组
    tm.assert_numpy_array_equal(idx._isnan, np.array([False, True]))
    # 断言索引含有 NaN 值
    assert idx.hasnans is True
```