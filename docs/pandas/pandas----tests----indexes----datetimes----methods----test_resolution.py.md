# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_resolution.py`

```
# 从 dateutil.tz 模块导入 tzlocal 时区信息
# 导入 pytest 测试框架
from dateutil.tz import tzlocal
import pytest

# 从 pandas.compat 中导入 IS64 和 WASM
from pandas.compat import (
    IS64,
    WASM,
)

# 从 pandas 中导入 date_range 函数
from pandas import date_range

# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize(
    "freq,expected",
    [
        ("YE", "day"),
        ("QE", "day"),
        ("ME", "day"),
        ("D", "day"),
        ("h", "hour"),
        ("min", "minute"),
        ("s", "second"),
        ("ms", "millisecond"),
        ("us", "microsecond"),
    ],
)
# 使用 pytest.mark.skipif 装饰器标记测试用例，当 WASM 为真时跳过，原因是在 WASM 上会收到 OverflowError
@pytest.mark.skipif(WASM, reason="OverflowError received on WASM")
# 定义测试函数 test_dti_resolution，接受 request、tz_naive_fixture、freq、expected 参数
def test_dti_resolution(request, tz_naive_fixture, freq, expected):
    # 将 tz_naive_fixture 赋值给 tz 变量
    tz = tz_naive_fixture
    # 如果 freq 为 "YE" 且 (非 IS64 或 WASM 为真) 且 tz 是 tzlocal 类型的实例
    if freq == "YE" and ((not IS64) or WASM) and isinstance(tz, tzlocal):
        # 在 request 上应用 pytest.mark.xfail 标记，原因是 tzlocal 在 2038 年后可能会导致 OverflowError
        request.applymarker(
            pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
        )

    # 使用 date_range 函数生成时间索引 idx，起始时间为 "2013-04-01"，长度为 30，频率为 freq，时区为 tz
    idx = date_range(start="2013-04-01", periods=30, freq=freq, tz=tz)
    # 断言 idx 的分辨率为 expected
    assert idx.resolution == expected
```