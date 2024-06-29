# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_round.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入Pytest库，用于单元测试

import pandas as pd  # 导入Pandas库，用于数据处理
from pandas import Series  # 从Pandas库中导入Series数据结构
import pandas._testing as tm  # 导入Pandas测试模块，用于测试辅助功能


class TestSeriesRound:
    def test_round(self, datetime_series):
        datetime_series.index.name = "index_name"  # 设置日期时间Series的索引名称为"index_name"
        result = datetime_series.round(2)  # 对日期时间Series进行小数点后两位的四舍五入
        expected = Series(
            np.round(datetime_series.values, 2), index=datetime_series.index, name="ts"
        )  # 创建期望的Series，其值为日期时间Series四舍五入后的结果，索引和名称与原Series相同
        tm.assert_series_equal(result, expected)  # 断言结果与期望的Series相等
        assert result.name == datetime_series.name  # 断言结果的名称与原日期时间Series的名称相同

    def test_round_numpy(self, any_float_dtype):
        # See GH#12600
        ser = Series([1.53, 1.36, 0.06], dtype=any_float_dtype)  # 创建一个带有任意浮点类型的Series
        out = np.round(ser, decimals=0)  # 使用NumPy对Series中的值进行四舍五入，小数点位数为0
        expected = Series([2.0, 1.0, 0.0], dtype=any_float_dtype)  # 创建期望的Series，其值为四舍五入后的结果
        tm.assert_series_equal(out, expected)  # 断言结果与期望的Series相等

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):  # 使用Pytest断言抛出值错误异常，并匹配指定的消息
            np.round(ser, decimals=0, out=ser)  # 使用NumPy对Series中的值进行四舍五入，并尝试将结果写入到原Series中

    def test_round_numpy_with_nan(self, any_float_dtype):
        # See GH#14197
        ser = Series([1.53, np.nan, 0.06], dtype=any_float_dtype)  # 创建一个带有NaN值的任意浮点类型的Series
        with tm.assert_produces_warning(None):  # 使用Pandas测试模块断言不会产生警告
            result = ser.round()  # 对Series中的值进行四舍五入
        expected = Series([2.0, np.nan, 0.0], dtype=any_float_dtype)  # 创建期望的Series，其值为四舍五入后的结果
        tm.assert_series_equal(result, expected)  # 断言结果与期望的Series相等

    def test_round_builtin(self, any_float_dtype):
        ser = Series(
            [1.123, 2.123, 3.123],
            index=range(3),
            dtype=any_float_dtype,
        )  # 创建一个带有任意浮点类型的Series，包含指定的索引和dtype
        result = round(ser)  # 使用Python内置的round函数对Series中的值进行四舍五入
        expected_rounded0 = Series(
            [1.0, 2.0, 3.0], index=range(3), dtype=any_float_dtype
        )  # 创建期望的Series，其值为四舍五入后的结果，索引和dtype与原Series相同
        tm.assert_series_equal(result, expected_rounded0)  # 断言结果与期望的Series相等

        decimals = 2  # 设置小数点位数为2
        expected_rounded = Series(
            [1.12, 2.12, 3.12], index=range(3), dtype=any_float_dtype
        )  # 创建期望的Series，其值为四舍五入后的结果，索引和dtype与原Series相同
        result = round(ser, decimals)  # 使用Python内置的round函数对Series中的值进行指定小数点位数的四舍五入
        tm.assert_series_equal(result, expected_rounded)  # 断言结果与期望的Series相等

    @pytest.mark.parametrize("method", ["round", "floor", "ceil"])
    @pytest.mark.parametrize("freq", ["s", "5s", "min", "5min", "h", "5h"])
    def test_round_nat(self, method, freq, unit):
        # GH14940, GH#56158
        ser = Series([pd.NaT], dtype=f"M8[{unit}]")  # 创建一个带有NaT值的Pandas时间类型的Series
        expected = Series(pd.NaT, dtype=f"M8[{unit}]")  # 创建期望的Series，其值为NaT，dtype与原Series相同
        round_method = getattr(ser.dt, method)  # 获取Series.dt对象中指定方法的方法对象
        result = round_method(freq)  # 调用指定方法对象，使用指定频率对Series中的时间值进行舍入操作
        tm.assert_series_equal(result, expected)  # 断言结果与期望的Series相等

    def test_round_ea_boolean(self):
        # GH#55936
        ser = Series([True, False], dtype="boolean")  # 创建一个布尔类型的Series
        expected = ser.copy()  # 复制原Series作为期望的Series
        result = ser.round(2)  # 对布尔类型的Series进行四舍五入，小数点位数为2
        tm.assert_series_equal(result, expected)  # 断言结果与期望的Series相等
        result.iloc[0] = False  # 修改结果Series的第一个元素为False
        tm.assert_series_equal(ser, expected)  # 断言原Series与期望的Series相等
```