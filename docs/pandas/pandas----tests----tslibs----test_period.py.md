# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_period.py`

```
import numpy as np
import pytest

from pandas._libs.tslibs import (
    iNaT,  # 导入特定时间序列库的模块和函数
    to_offset,
)
from pandas._libs.tslibs.period import (
    extract_ordinals,  # 导入特定时间周期库的模块和函数
    get_period_field_arr,
    period_asfreq,
    period_ordinal,
)

import pandas._testing as tm  # 导入用于测试的 pandas 工具模块


def get_freq_code(freqstr: str) -> int:
    off = to_offset(freqstr, is_period=True)  # 根据频率字符串创建偏移量对象
    # 获取偏移量对象的周期数据类型码
    code = off._period_dtype_code  # type: ignore[attr-defined]
    return code


@pytest.mark.parametrize(
    "freq1,freq2,expected",
    [
        ("D", "h", 24),    # 测试日到小时的频率转换因子
        ("D", "min", 1440),    # 测试日到分钟的频率转换因子
        ("D", "s", 86400),    # 测试日到秒的频率转换因子
        ("D", "ms", 86400000),    # 测试日到毫秒的频率转换因子
        ("D", "us", 86400000000),    # 测试日到微秒的频率转换因子
        ("D", "ns", 86400000000000),    # 测试日到纳秒的频率转换因子
        ("h", "min", 60),    # 测试小时到分钟的频率转换因子
        ("h", "s", 3600),    # 测试小时到秒的频率转换因子
        ("h", "ms", 3600000),    # 测试小时到毫秒的频率转换因子
        ("h", "us", 3600000000),    # 测试小时到微秒的频率转换因子
        ("h", "ns", 3600000000000),    # 测试小时到纳秒的频率转换因子
        ("min", "s", 60),    # 测试分钟到秒的频率转换因子
        ("min", "ms", 60000),    # 测试分钟到毫秒的频率转换因子
        ("min", "us", 60000000),    # 测试分钟到微秒的频率转换因子
        ("min", "ns", 60000000000),    # 测试分钟到纳秒的频率转换因子
        ("s", "ms", 1000),    # 测试秒到毫秒的频率转换因子
        ("s", "us", 1000000),    # 测试秒到微秒的频率转换因子
        ("s", "ns", 1000000000),    # 测试秒到纳秒的频率转换因子
        ("ms", "us", 1000),    # 测试毫秒到微秒的频率转换因子
        ("ms", "ns", 1000000),    # 测试毫秒到纳秒的频率转换因子
        ("us", "ns", 1000),    # 测试微秒到纳秒的频率转换因子
    ],
)
def test_intra_day_conversion_factors(freq1, freq2, expected):
    assert (
        period_asfreq(1, get_freq_code(freq1), get_freq_code(freq2), False) == expected
    )


@pytest.mark.parametrize(
    "freq,expected", [("Y", 0), ("M", 0), ("W", 1), ("D", 0), ("B", 0)]
)
def test_period_ordinal_start_values(freq, expected):
    # 用于特定日期（1970年1月1日）的周期序数值计算
    assert period_ordinal(1970, 1, 1, 0, 0, 0, 0, 0, get_freq_code(freq)) == expected


@pytest.mark.parametrize(
    "dt,expected",
    [
        ((1970, 1, 4, 0, 0, 0, 0, 0), 1),    # 测试特定日期的周期序数值计算
        ((1970, 1, 5, 0, 0, 0, 0, 0), 2),    # 测试特定日期的周期序数值计算
        ((2013, 10, 6, 0, 0, 0, 0, 0), 2284),    # 测试特定日期的周期序数值计算
        ((2013, 10, 7, 0, 0, 0, 0, 0), 2285),    # 测试特定日期的周期序数值计算
    ],
)
def test_period_ordinal_week(dt, expected):
    args = dt + (get_freq_code("W"),)
    assert period_ordinal(*args) == expected


@pytest.mark.parametrize(
    "day,expected",
    [
        (3, 11415),    # 测试特定日期的工作日周期序数值计算
        (4, 11416),    # 测试特定日期的工作日周期序数值计算
        (5, 11417),    # 测试特定日期的工作日周期序数值计算
        (6, 11417),    # 测试特定日期的工作日周期序数值计算
        (7, 11417),    # 测试特定日期的工作日周期序数值计算
        (8, 11418),    # 测试特定日期的工作日周期序数值计算
    ],
)
def test_period_ordinal_business_day(day, expected):
    # 用于特定日期的工作日周期序数值计算
    args = (2013, 10, day, 0, 0, 0, 0, 0, 5000)
    assert period_ordinal(*args) == expected


class TestExtractOrdinals:
    def test_extract_ordinals_raises(self):
        # 确保传入非对象类型时引发 TypeError 而非段错误
        arr = np.arange(5)
        freq = to_offset("D")
        with pytest.raises(TypeError, match="values must be object-dtype"):
            extract_ordinals(arr, freq)
    # 定义测试方法，用于测试 extract_ordinals 函数在处理二维数组时的行为
    def test_extract_ordinals_2d(self):
        # 设置时间频率为日（每天）
        freq = to_offset("D")
        # 创建一个长度为10的空对象数组
        arr = np.empty(10, dtype=object)
        # 将数组所有元素设置为 iNaT（缺失时间戳的特殊值）
        arr[:] = iNaT

        # 调用 extract_ordinals 函数处理一维数组 arr
        res = extract_ordinals(arr, freq)
        # 将一维数组 arr 重塑为二维数组（5行2列），再次调用 extract_ordinals 函数
        res2 = extract_ordinals(arr.reshape(5, 2), freq)
        # 使用测试框架的函数验证两次调用的结果是否相等
        tm.assert_numpy_array_equal(res, res2.reshape(-1))
# 定义一个测试函数，用于测试在越界情况下是否会引发异常
def test_get_period_field_array_raises_on_out_of_range():
    # 错误消息字符串，用于匹配异常信息
    msg = "Buffer dtype mismatch, expected 'const int64_t' but got 'double'"
    # 使用 pytest 模块的 raises 函数检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用被测试函数 get_period_field_arr，并传入预期越界的参数
        get_period_field_arr(-1, np.empty(1), 0)
```