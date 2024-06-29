# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_maybe_box_native.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 pytest 库，用于测试框架
import pytest

# 从 pandas 库的 core.dtypes.cast 模块导入 maybe_box_native 函数
from pandas.core.dtypes.cast import maybe_box_native

# 从 pandas 库导入 Interval, Period, Timedelta, Timestamp 这些类
from pandas import (
    Interval,
    Period,
    Timedelta,
    Timestamp,
)

# 使用 pytest 的 parametrize 装饰器定义测试参数化
@pytest.mark.parametrize(
    "obj,expected_dtype",  # 测试参数包括对象和预期数据类型
    [
        (b"\x00\x10", bytes),                      # 字节对象应返回 bytes 类型
        (int(4), int),                              # 整数应返回 int 类型
        (np.uint(4), int),                          # numpy 无符号整数应返回 int 类型
        (np.int32(-4), int),                        # numpy 32位整数应返回 int 类型
        (np.uint8(4), int),                         # numpy 8位无符号整数应返回 int 类型
        (float(454.98), float),                     # 浮点数应返回 float 类型
        (np.float16(0.4), float),                   # numpy 16位浮点数应返回 float 类型
        (np.float64(1.4), float),                   # numpy 64位浮点数应返回 float 类型
        (np.bool_(False), bool),                    # numpy 布尔值应返回 bool 类型
        (datetime(2005, 2, 25), datetime),          # datetime 对象应返回 datetime 类型
        (np.datetime64("2005-02-25"), Timestamp),   # numpy 日期时间对象应返回 Timestamp 类型
        (Timestamp("2005-02-25"), Timestamp),       # pandas Timestamp 对象应返回 Timestamp 类型
        (np.timedelta64(1, "D"), Timedelta),        # numpy 时间差对象应返回 Timedelta 类型
        (Timedelta(1, "D"), Timedelta),             # pandas Timedelta 对象应返回 Timedelta 类型
        (Interval(0, 1), Interval),                 # pandas Interval 对象应返回 Interval 类型
        (Period("4Q2005"), Period),                 # pandas Period 对象应返回 Period 类型
    ],
)
def test_maybe_box_native(obj, expected_dtype):
    # 调用 maybe_box_native 函数对输入对象进行处理
    boxed_obj = maybe_box_native(obj)
    # 获取处理后对象的类型
    result_dtype = type(boxed_obj)
    # 断言处理后对象的类型与预期数据类型相符
    assert result_dtype is expected_dtype
```