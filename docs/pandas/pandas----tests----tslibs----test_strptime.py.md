# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_strptime.py`

```
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.strptime import array_strptime

from pandas import (
    NaT,
    Timestamp,
)
import pandas._testing as tm

# 从NpyDatetimeUnit中导入NPY_FR_GENERIC的枚举值作为creso_infer变量的值
creso_infer = NpyDatetimeUnit.NPY_FR_GENERIC.value

# 定义一个测试类TestArrayStrptimeResolutionInference
class TestArrayStrptimeResolutionInference:
    
    # 测试方法：检查在数组全为NaT（Not a Time）时的解析结果
    def test_array_strptime_resolution_all_nat(self):
        # 创建包含NaT和NaN的numpy数组，dtype为object
        arr = np.array([NaT, np.nan], dtype=object)

        # 时间格式字符串
        fmt = "%Y-%m-%d %H:%M:%S"
        # 调用array_strptime函数解析数组arr，不使用UTC时间，使用creso_infer作为解析分辨率
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        # 断言解析结果的dtype应为"M8[s]"，即精确到秒的datetime64类型
        assert res.dtype == "M8[s]"

        # 同上，但使用UTC时间
        res, _ = array_strptime(arr, fmt=fmt, utc=True, creso=creso_infer)
        assert res.dtype == "M8[s]"

    # 参数化测试方法：检查对于同质字符串数组的解析结果推断
    @pytest.mark.parametrize("tz", [None, timezone.utc])
    def test_array_strptime_resolution_inference_homogeneous_strings(self, tz):
        # 创建带有时区信息的datetime对象
        dt = datetime(2016, 1, 2, 3, 4, 5, 678900, tzinfo=tz)

        # 时间格式字符串
        fmt = "%Y-%m-%d %H:%M:%S"
        # 将datetime对象格式化为字符串
        dtstr = dt.strftime(fmt)
        # 创建包含三个相同字符串的numpy数组，dtype为object
        arr = np.array([dtstr] * 3, dtype=object)
        # 创建预期的numpy数组，其中时区信息被移除，dtype为"M8[s]"
        expected = np.array([dt.replace(tzinfo=None)] * 3, dtype="M8[s]")

        # 调用array_strptime函数解析数组arr，不使用UTC时间，使用creso_infer作为解析分辨率
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        # 断言解析结果与预期数组相等
        tm.assert_numpy_array_equal(res, expected)

        # 时间格式字符串精确到微秒
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        dtstr = dt.strftime(fmt)
        arr = np.array([dtstr] * 3, dtype=object)
        # 创建预期的numpy数组，精确到微秒，dtype为"M8[us]"
        expected = np.array([dt.replace(tzinfo=None)] * 3, dtype="M8[us]")

        # 调用array_strptime函数解析数组arr，不使用UTC时间，使用creso_infer作为解析分辨率
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        # 断言解析结果与预期数组相等
        tm.assert_numpy_array_equal(res, expected)

        # 时间格式字符串为ISO8601
        fmt = "ISO8601"
        # 调用array_strptime函数解析数组arr，不使用UTC时间，使用creso_infer作为解析分辨率
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        # 断言解析结果与预期数组相等
        tm.assert_numpy_array_equal(res, expected)

    # 参数化测试方法：检查对于混合类型数组的解析结果推断
    @pytest.mark.parametrize("tz", [None, timezone.utc])
    def test_array_strptime_resolution_mixed(self, tz):
        # 创建带有时区信息的datetime对象
        dt = datetime(2016, 1, 2, 3, 4, 5, 678900, tzinfo=tz)

        # 创建Timestamp对象并将其精确到纳秒
        ts = Timestamp(dt).as_unit("ns")

        # 创建包含datetime对象和Timestamp对象的numpy数组，dtype为object
        arr = np.array([dt, ts], dtype=object)
        # 创建预期的numpy数组，其中datetime对象被转换为纳秒精度的numpy datetime64类型，dtype为"M8[ns]"
        expected = np.array(
            [Timestamp(dt).as_unit("ns").asm8, ts.asm8],
            dtype="M8[ns]",
        )

        # 时间格式字符串
        fmt = "%Y-%m-%d %H:%M:%S"
        # 调用array_strptime函数解析数组arr，不使用UTC时间，使用creso_infer作为解析分辨率
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        # 断言解析结果与预期数组相等
        tm.assert_numpy_array_equal(res, expected)

        # 时间格式字符串为ISO8601
        # 调用array_strptime函数解析数组arr，不使用UTC时间，使用creso_infer作为解析分辨率
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        # 断言解析结果与预期数组相等
        tm.assert_numpy_array_equal(res, expected)
    # 定义测试方法，用于验证数组时间解析函数在包含"today"或具体日期的数组中的行为
    def test_array_strptime_resolution_todaynow(self):
        # 创建包含字符串"today"和日期对象的 NumPy 数组，数据类型为 object
        vals = np.array(["today", np.datetime64("2017-01-01", "us")], dtype=object)

        # 获取当前时间戳的纳秒级表示
        now = Timestamp("now").asm8
        
        # 调用数组时间解析函数，解析包含"today"在前的数组，指定日期格式为"%Y-%m-%d"，非 UTC 时间
        res, _ = array_strptime(vals, fmt="%Y-%m-%d", utc=False, creso=creso_infer)
        
        # 调用数组时间解析函数，解析翻转数组（"today"在后），指定日期格式为"%Y-%m-%d"，非 UTC 时间
        res2, _ = array_strptime(
            vals[::-1], fmt="%Y-%m-%d", utc=False, creso=creso_infer
        )

        # 设置函数调用耗时的阈值为1秒；在本地测试中，实际耗时差异约为250微秒
        tolerance = np.timedelta64(1, "s")

        # 断言结果数组的数据类型为纳秒级日期时间
        assert res.dtype == "M8[us]"
        # 断言第一个元素的时间与当前时间戳的差值小于设定的耐受度
        assert abs(res[0] - now) < tolerance
        # 断言第二个元素与原始数组中的日期对象相等
        assert res[1] == vals[1]

        # 断言结果数组的数据类型为纳秒级日期时间
        assert res2.dtype == "M8[us]"
        # 断言第二个元素的时间与当前时间戳的差值小于设定的耐受度的两倍
        assert abs(res2[1] - now) < tolerance * 2
        # 断言第一个元素与原始数组中的日期对象相等
        assert res2[0] == vals[1]

    # 定义测试方法，用于验证数组时间解析函数在超出纳秒范围的日期字符串上的行为
    def test_array_strptime_str_outside_nano_range(self):
        # 创建包含日期字符串的 NumPy 数组，数据类型为 object
        vals = np.array(["2401-09-15"], dtype=object)
        # 创建预期的日期时间数组，数据类型为秒级日期时间
        expected = np.array(["2401-09-15"], dtype="M8[s]")
        # 指定日期格式为 ISO8601，调用数组时间解析函数，使用默认的时间解析推断策略
        res, _ = array_strptime(vals, fmt="ISO8601", creso=creso_infer)
        # 断言返回的日期时间数组与预期结果相等
        tm.assert_numpy_array_equal(res, expected)

        # 创建包含非 ISO 格式日期字符串的 NumPy 数组，数据类型为 object
        vals2 = np.array(["Sep 15, 2401"], dtype=object)
        # 创建预期的日期时间数组，数据类型为秒级日期时间
        expected2 = np.array(["2401-09-15"], dtype="M8[s]")
        # 指定日期格式为 "%b %d, %Y"，调用数组时间解析函数，使用默认的时间解析推断策略
        res2, _ = array_strptime(vals2, fmt="%b %d, %Y", creso=creso_infer)
        # 断言返回的日期时间数组与预期结果相等
        tm.assert_numpy_array_equal(res2, expected2)
```