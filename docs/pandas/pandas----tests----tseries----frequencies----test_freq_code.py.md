# `D:\src\scipysrc\pandas\pandas\tests\tseries\frequencies\test_freq_code.py`

```
# 导入所需的模块和库
import numpy as np
import pytest

# 从 pandas._libs.tslibs 中导入 Period 类和 to_offset 函数
from pandas._libs.tslibs import (
    Period,
    to_offset,
)

# 使用 pytest 的 parametrize 装饰器定义多组参数化测试用例
@pytest.mark.parametrize(
    "freqstr,exp_freqstr",
    [("D", "D"), ("W", "D"), ("ME", "D"), ("s", "s"), ("min", "s"), ("h", "s")],
)
def test_get_to_timestamp_base(freqstr, exp_freqstr):
    # 根据频率字符串创建时间偏移量对象
    off = to_offset(freqstr)
    # 使用偏移量对象创建周期对象
    per = Period._from_ordinal(1, off)
    # 根据预期频率字符串创建时间偏移量对象，获取其内部的周期数据类型代码
    exp_code = to_offset(exp_freqstr)._period_dtype_code

    # 获取周期对象内部的周期数据类型代码
    result_code = per._dtype._get_to_timestamp_base()
    # 断言实际结果与预期结果相等
    assert result_code == exp_code


@pytest.mark.parametrize(
    "args,expected",
    [
        ((1.5, "min"), (90, "s")),
        ((62.4, "min"), (3744, "s")),
        ((1.04, "h"), (3744, "s")),
        ((1, "D"), (1, "D")),
        ((0.342931, "h"), (1234551600, "us")),
        ((1.2345, "D"), (106660800, "ms")),
    ],
)
def test_resolution_bumping(args, expected):
    # 查看 GitHub issue gh-14378 的相关信息
    off = to_offset(str(args[0]) + args[1])
    # 断言时间偏移量对象的周期数量与预期结果相等
    assert off.n == expected[0]
    # 断言时间偏移量对象的前缀与预期结果相等
    assert off._prefix == expected[1]


@pytest.mark.parametrize(
    "args",
    [
        (0.5, "ns"),
        # 输入精度过高可能导致的异常情况
        (0.3429324798798269273987982, "h"),
    ],
)
def test_cat(args):
    # 定义错误消息字符串
    msg = "Invalid frequency"

    # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配错误消息字符串
    with pytest.raises(ValueError, match=msg):
        # 尝试根据参数创建时间偏移量对象
        to_offset(str(args[0]) + args[1])


@pytest.mark.parametrize(
    "freqstr,expected",
    [
        ("1h", "2021-01-01T09:00:00"),
        ("1D", "2021-01-02T08:00:00"),
        ("1W", "2021-01-03T08:00:00"),
        ("1ME", "2021-01-31T08:00:00"),
        ("1YE", "2021-12-31T08:00:00"),
    ],
)
def test_compatibility(freqstr, expected):
    # 创建一个 NumPy datetime64 对象
    ts_np = np.datetime64("2021-01-01T08:00:00.00")
    # 根据频率字符串创建时间偏移量对象
    do = to_offset(freqstr)
    # 断言计算后的时间与预期结果相等
    assert ts_np + do == np.datetime64(expected)
```