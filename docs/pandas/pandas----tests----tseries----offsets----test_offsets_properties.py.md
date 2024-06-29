# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_offsets_properties.py`

```
"""
Behavioral based tests for offsets and date_range.

This file is adapted from https://github.com/pandas-dev/pandas/pull/18761 -
which was more ambitious but less idiomatic in its use of Hypothesis.

You may wish to consult the previous version for inspiration on further
tests, or when trying to pin down the bugs exposed by the tests below.
"""

# 导入必要的库和模块
from hypothesis import (
    assume,
    given,
)
import pytest
import pytz

import pandas as pd
from pandas._testing._hypothesis import (
    DATETIME_JAN_1_1900_OPTIONAL_TZ,
    YQM_OFFSET,
)

# ----------------------------------------------------------------
# Offset-specific behaviour tests

# 标记测试为慢速测试
@pytest.mark.arm_slow
@given(DATETIME_JAN_1_1900_OPTIONAL_TZ, YQM_OFFSET)
def test_on_offset_implementations(dt, offset):
    # 假设偏移量不需要标准化
    assume(not offset.normalize)

    # 检查类特定的 is_on_offset 实现是否与一般情况定义匹配
    # (dt + offset) - offset == dt
    try:
        compare = (dt + offset) - offset
    except (pytz.NonExistentTimeError, pytz.AmbiguousTimeError):
        # 当 dt + offset 不存在或者是 DST-不明确时，假设(False)表明这不是一个有效的测试案例
        # DST-不明确的例子 (GH41906):
        # dt = datetime.datetime(1900, 1, 1, tzinfo=pytz.timezone('Africa/Kinshasa'))
        # offset = MonthBegin(66)
        assume(False)

    # 断言测试结果
    assert offset.is_on_offset(dt) == (compare == dt)


@given(YQM_OFFSET)
def test_shift_across_dst(offset):
    # GH#18319 检查 1) 时区是否正确标准化，和
    # 2) 标准化是否正确地更改了小时数
    assume(not offset.normalize)

    # 注意 dti 包含跨越夏令时边界的转换
    dti = pd.date_range(
        start="2017-10-30 12:00:00", end="2017-11-06", freq="D", tz="US/Eastern"
    )
    assert (dti.hour == 12).all()  # 还没有出错

    # 对时间序列进行偏移操作
    res = dti + offset
    assert (res.hour == 12).all()
```