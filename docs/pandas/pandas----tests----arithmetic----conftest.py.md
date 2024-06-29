# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\conftest.py`

```
# 导入 numpy 库并将其命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 pandas 库并将其命名为 pd
import pandas as pd
# 从 pandas 库中导入 Index 类
from pandas import Index


# ------------------------------------------------------------------
# Scalar Fixtures

# 定义名为 one 的 pytest fixture，接受参数为 1 和 np.array(1, dtype=np.int64)
@pytest.fixture(params=[1, np.array(1, dtype=np.int64)])
def one(request):
    """
    Several variants of integer value 1. The zero-dim integer array
    behaves like an integer.

    This fixture can be used to check that datetimelike indexes handle
    addition and subtraction of integers and zero-dimensional arrays
    of integers.

    Examples
    --------
    dti = pd.date_range('2016-01-01', periods=2, freq='h')
    dti
    DatetimeIndex(['2016-01-01 00:00:00', '2016-01-01 01:00:00'],
    dtype='datetime64[ns]', freq='h')
    dti + one
    DatetimeIndex(['2016-01-01 01:00:00', '2016-01-01 02:00:00'],
    dtype='datetime64[ns]', freq='h')
    """
    return request.param


# 定义名为 zeros 的列表，包含各种类型的零值和零向量
zeros = [
    # 生成长度为 5 的零值向量，使用不同的数据类型
    box_cls([0] * 5, dtype=dtype)
    for box_cls in [Index, np.array, pd.array]  # 遍历 Index, np.array, pd.array
    for dtype in [np.int64, np.uint64, np.float64]  # 遍历 np.int64, np.uint64, np.float64
]
# 添加包含负零的浮点数向量，使用不同的数据类型
zeros.extend([box_cls([-0.0] * 5, dtype=np.float64) for box_cls in [Index, np.array]])
# 添加不同数据类型的单个零值
zeros.extend([np.array(0, dtype=dtype) for dtype in [np.int64, np.uint64, np.float64]])
# 添加包含负零的单个浮点数零值
zeros.extend([np.array(-0.0, dtype=np.float64)])
# 添加普通的整数、浮点数和负零
zeros.extend([0, 0.0, -0.0])


# 定义名为 zero 的 pytest fixture，接受参数为 zeros 列表中的各种零值
@pytest.fixture(params=zeros)
def zero(request):
    """
    Several types of scalar zeros and length 5 vectors of zeros.

    This fixture can be used to check that numeric-dtype indexes handle
    division by any zero numeric-dtype.

    Uses vector of length 5 for broadcasting with `numeric_idx` fixture,
    which creates numeric-dtype vectors also of length 5.

    Examples
    --------
    arr = RangeIndex(5)
    arr / zeros
    Index([nan, inf, inf, inf, inf], dtype='float64')
    """
    return request.param


# 定义名为 scalar_td 的 pytest fixture，接受参数为几种 Timedelta 标量值
@pytest.fixture(
    params=[
        pd.Timedelta("10m7s").to_pytimedelta(),
        pd.Timedelta("10m7s"),
        pd.Timedelta("10m7s").to_timedelta64(),
    ],
    ids=lambda x: type(x).__name__,
)
def scalar_td(request):
    """
    Several variants of Timedelta scalars representing 10 minutes and 7 seconds.
    """
    return request.param


# 定义名为 three_days 的 pytest fixture，接受参数为几种代表三天的 timedelta-like 和 DateOffset 对象
@pytest.fixture(
    params=[
        pd.offsets.Day(3),
        pd.offsets.Hour(72),
        pd.Timedelta(days=3).to_pytimedelta(),
        pd.Timedelta("72:00:00"),
        np.timedelta64(3, "D"),
        np.timedelta64(72, "h"),
    ],
    ids=lambda x: type(x).__name__,
)
def three_days(request):
    """
    Several timedelta-like and DateOffset objects that each represent
    a 3-day timedelta
    """
    return request.param


# 定义名为 two_hours 的 pytest fixture，接受参数为几种代表两小时的 timedelta-like 和 DateOffset 对象
@pytest.fixture(
    params=[
        pd.offsets.Hour(2),
        pd.offsets.Minute(120),
        pd.Timedelta(hours=2).to_pytimedelta(),
        pd.Timedelta(seconds=2 * 3600),
        np.timedelta64(2, "h"),
        np.timedelta64(120, "m"),
    ],
    ids=lambda x: type(x).__name__,
)
def two_hours(request):
    """
    Several timedelta-like and DateOffset objects that each represent
    a 2-hour timedelta
    """
    # 返回函数调用的参数值
    return request.param
# 定义了一个列表 `_common_mismatch`，包含了多个时间偏移对象
_common_mismatch = [
    pd.offsets.YearBegin(2),    # 表示从每年开始的第2年
    pd.offsets.MonthBegin(1),   # 表示从每月开始的第1月
    pd.offsets.Minute(),        # 表示每分钟的时间偏移量
]

# 使用 pytest 框架的 fixture 装饰器定义了一个名为 `not_daily` 的测试数据生成器
@pytest.fixture(
    params=[
        np.timedelta64(4, "h"),  # 表示4小时的 numpy 时间增量
        pd.Timedelta(hours=23).to_pytimedelta(),  # 表示23小时的 pandas 时间增量
        pd.Timedelta("23:00:00"),  # 表示23小时的 pandas 时间增量，另一种表达方式
    ]
    + _common_mismatch  # 将之前定义的 `_common_mismatch` 列表中的内容添加到参数列表中
)
def not_daily(request):
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Daily frequencies.
    """
    return request.param  # 返回通过参数化生成的每一个测试数据
```