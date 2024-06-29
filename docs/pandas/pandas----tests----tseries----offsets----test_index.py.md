# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_index.py`

```
"""
Tests for offset behavior with indices.
"""

# 导入 pytest 测试框架
import pytest

# 导入 pandas 库中的 Series 和 date_range 函数
from pandas import (
    Series,
    date_range,
)

# 导入 pandas 库中时间序列偏移量相关的类
from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
)

# 使用 pytest 的参数化装饰器指定测试参数 n 和 cls
@pytest.mark.parametrize("n", [-2, 1])
@pytest.mark.parametrize(
    "cls",
    [
        MonthBegin,
        MonthEnd,
        BMonthBegin,
        BMonthEnd,
        QuarterBegin,
        QuarterEnd,
        BQuarterBegin,
        BQuarterEnd,
        YearBegin,
        YearEnd,
        BYearBegin,
        BYearEnd,
    ],
)
# 定义测试函数 test_apply_index，参数为偏移量类 cls 和 n
def test_apply_index(cls, n):
    # 创建偏移量对象
    offset = cls(n=n)
    # 创建时间范围，从 "1/1/2000" 开始，频率为每分钟，共 100000 个时间点
    rng = date_range(start="1/1/2000", periods=100000, freq="min")
    # 将时间范围转换为 Series 对象
    ser = Series(rng)

    # 对时间范围应用偏移量，生成结果 res
    res = rng + offset
    # 断言结果 res 的频率为 None，表示未保留频率信息
    assert res.freq is None
    # 断言结果 res 的第一个元素等于原始时间范围 rng 的第一个元素加上偏移量 offset
    assert res[0] == rng[0] + offset
    # 断言结果 res 的最后一个元素等于原始时间范围 rng 的最后一个元素加上偏移量 offset
    assert res[-1] == rng[-1] + offset

    # 对 Series 对象应用偏移量，生成结果 res2
    res2 = ser + offset
    # 断言结果 res2 的第一个元素等于原始 Series 对象 ser 的第一个元素加上偏移量 offset
    assert res2.iloc[0] == ser.iloc[0] + offset
    # 断言结果 res2 的最后一个元素等于原始 Series 对象 ser 的最后一个元素加上偏移量 offset
    assert res2.iloc[-1] == ser.iloc[-1] + offset
```