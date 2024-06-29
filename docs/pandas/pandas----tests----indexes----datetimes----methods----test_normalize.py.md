# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_normalize.py`

```
from dateutil.tz import tzlocal  # 导入本地时区工具
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

from pandas.compat import WASM  # 导入WASM兼容性模块
import pandas.util._test_decorators as td  # 导入测试装饰器模块

from pandas import (  # 从Pandas库中导入多个模块
    DatetimeIndex,
    NaT,
    Timestamp,
    date_range,
)
import pandas._testing as tm  # 导入Pandas测试模块

class TestNormalize:  # 定义测试类TestNormalize
    def test_normalize(self):  # 定义测试方法test_normalize
        rng = date_range("1/1/2000 9:30", periods=10, freq="D")  # 创建日期范围对象rng

        result = rng.normalize()  # 对日期范围对象进行归一化操作
        expected = date_range("1/1/2000", periods=10, freq="D")  # 创建期望的日期范围对象expected
        tm.assert_index_equal(result, expected)  # 使用测试模块验证结果与期望相等

        arr_ns = np.array([1380585623454345752, 1380585612343234312]).astype(
            "datetime64[ns]"
        )
        rng_ns = DatetimeIndex(arr_ns)  # 创建纳秒级日期时间索引对象rng_ns
        rng_ns_normalized = rng_ns.normalize()  # 对日期时间索引对象进行归一化操作

        arr_ns = np.array([1380585600000000000, 1380585600000000000]).astype(
            "datetime64[ns]"
        )
        expected = DatetimeIndex(arr_ns)  # 创建期望的日期时间索引对象expected
        tm.assert_index_equal(rng_ns_normalized, expected)  # 使用测试模块验证结果与期望相等

        assert result.is_normalized  # 断言结果已归一化
        assert not rng.is_normalized  # 断言原始日期范围对象未归一化

    def test_normalize_nat(self):  # 定义测试方法test_normalize_nat
        dti = DatetimeIndex([NaT, Timestamp("2018-01-01 01:00:00")])  # 创建包含NaT和时间戳的日期时间索引对象dti
        result = dti.normalize()  # 对日期时间索引对象进行归一化操作
        expected = DatetimeIndex([NaT, Timestamp("2018-01-01")])  # 创建期望的日期时间索引对象expected
        tm.assert_index_equal(result, expected)  # 使用测试模块验证结果与期望相等

    def test_normalize_tz(self):  # 定义测试方法test_normalize_tz
        rng = date_range("1/1/2000 9:30", periods=10, freq="D", tz="US/Eastern")  # 创建带有时区的日期范围对象rng

        result = rng.normalize()  # 对日期范围对象进行归一化操作，不保留频率信息
        expected = date_range("1/1/2000", periods=10, freq="D", tz="US/Eastern")  # 创建期望的带有时区的日期范围对象expected
        tm.assert_index_equal(result, expected._with_freq(None))  # 使用测试模块验证结果与期望相等，不保留频率信息

        assert result.is_normalized  # 断言结果已归一化
        assert not rng.is_normalized  # 断言原始日期范围对象未归一化

        rng = date_range("1/1/2000 9:30", periods=10, freq="D", tz="UTC")  # 创建带有UTC时区的日期范围对象rng

        result = rng.normalize()  # 对日期范围对象进行归一化操作
        expected = date_range("1/1/2000", periods=10, freq="D", tz="UTC")  # 创建期望的带有UTC时区的日期范围对象expected
        tm.assert_index_equal(result, expected)  # 使用测试模块验证结果与期望相等

        assert result.is_normalized  # 断言结果已归一化
        assert not rng.is_normalized  # 断言原始日期范围对象未归一化

        rng = date_range("1/1/2000 9:30", periods=10, freq="D", tz=tzlocal())  # 创建带有本地时区的日期范围对象rng
        result = rng.normalize()  # 对日期范围对象进行归一化操作，不保留频率信息
        expected = date_range("1/1/2000", periods=10, freq="D", tz=tzlocal())  # 创建期望的带有本地时区的日期范围对象expected
        tm.assert_index_equal(result, expected._with_freq(None))  # 使用测试模块验证结果与期望相等，不保留频率信息

        assert result.is_normalized  # 断言结果已归一化
        assert not rng.is_normalized  # 断言原始日期范围对象未归一化

    @td.skip_if_windows  # 使用测试装饰器，如果运行环境为Windows则跳过测试
    @pytest.mark.skipif(
        WASM, reason="tzset is available only on Unix-like systems, not WASM"
    )
    @pytest.mark.parametrize(
        "timezone",
        [
            "US/Pacific",
            "US/Eastern",
            "UTC",
            "Asia/Kolkata",
            "Asia/Shanghai",
            "Australia/Canberra",
        ],
    )
    # 定义测试方法，用于测试时区归一化功能
    def test_normalize_tz_local(self, timezone):
        # GH#13459: 对应GitHub上的issue编号，用于跟踪问题

        # 使用指定的时区设置测试环境
        with tm.set_timezone(timezone):
            # 创建一个日期范围，从 "1/1/2000 9:30" 开始，包含10个日期，频率为每日，使用本地时区
            rng = date_range("1/1/2000 9:30", periods=10, freq="D", tz=tzlocal())

            # 对日期范围进行归一化处理
            result = rng.normalize()

            # 期望的归一化结果，从 "1/1/2000" 开始，包含10个日期，频率为每日，使用本地时区
            expected = date_range("1/1/2000", periods=10, freq="D", tz=tzlocal())

            # 将期望结果的频率设置为None
            expected = expected._with_freq(None)

            # 断言归一化后的结果与期望结果相等
            tm.assert_index_equal(result, expected)

            # 断言归一化后的结果已经是归一化状态
            assert result.is_normalized

            # 断言原始日期范围未归一化
            assert not rng.is_normalized
```