# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_clip.py`

```
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 导入 numpy 模块并使用别名 np
import numpy as np
# 导入 pytest 模块
import pytest

# 从 pandas.errors 模块中导入 OutOfBoundsDatetime 类
from pandas.errors import OutOfBoundsDatetime

# 导入 pandas 模块并使用别名 pd
import pandas as pd
# 从 pandas 模块中导入 Series, Timestamp, isna, notna 函数
from pandas import (
    Series,
    Timestamp,
    isna,
    notna,
)
# 导入 pandas._testing 模块并使用别名 tm
import pandas._testing as tm

# 定义 TestSeriesClip 类
class TestSeriesClip:
    # 定义 test_clip 方法，传入 datetime_series 参数
    def test_clip(self, datetime_series):
        # 计算 datetime_series 的中位数
        val = datetime_series.median()

        # 断言 datetime_series 调用 clip 方法后的最小值等于中位数
        assert datetime_series.clip(lower=val).min() == val
        # 断言 datetime_series 调用 clip 方法后的最大值等于中位数
        assert datetime_series.clip(upper=val).max() == val

        # 对 datetime_series 调用 clip 方法，指定范围为 -0.5 到 0.5
        result = datetime_series.clip(-0.5, 0.5)
        # 使用 np.clip 函数对 datetime_series 进行裁剪，范围为 -0.5 到 0.5
        expected = np.clip(datetime_series, -0.5, 0.5)
        # 断言 result 和 expected 相等
        tm.assert_series_equal(result, expected)
        # 断言 expected 的类型为 Series
        assert isinstance(expected, Series)

    # 定义 test_clip_types_and_nulls 方法
    def test_clip_types_and_nulls(self):
        # 创建包含不同类型数据的 Series 列表
        sers = [
            Series([np.nan, 1.0, 2.0, 3.0]),
            Series([None, "a", "b", "c"]),
            Series(pd.to_datetime([np.nan, 1, 2, 3], unit="D")),
        ]

        # 遍历 sers 列表
        for s in sers:
            # 获取 s 的第二个元素作为阈值
            thresh = s[2]
            # 对 s 调用 clip 方法，指定下限为阈值
            lower = s.clip(lower=thresh)
            # 对 s 调用 clip 方法，指定上限为阈值
            upper = s.clip(upper=thresh)
            # 断言 lower 中非空值的最小值等于阈值
            assert lower[notna(lower)].min() == thresh
            # 断言 upper 中非空值的最大值等于阈值
            assert upper[notna(upper)].max() == thresh
            # 断言 s 中的缺失值与 lower 中的缺失值相同
            assert list(isna(s)) == list(isna(lower))
            # 断言 s 中的缺失值与 upper 中的缺失值相同
            assert list(isna(s)) == list(isna(upper))

    # 定义 test_series_clipping_with_na_values 方法，传入 any_numeric_ea_dtype 和 nulls_fixture 参数
    def test_series_clipping_with_na_values(self, any_numeric_ea_dtype, nulls_fixture):
        # 确保裁剪方法可以处理 NA 值而不会失败
        # GH#40581

        if nulls_fixture is pd.NaT:
            # 如果 nulls_fixture 是 pd.NaT，则跳过
            pytest.skip("See test_constructor_mismatched_null_nullable_dtype")

        # 创建包含 NA 值的 Series
        ser = Series([nulls_fixture, 1.0, 3.0], dtype=any_numeric_ea_dtype)
        # 对 ser 调用 clip 方法，指定上限为 2.0
        s_clipped_upper = ser.clip(upper=2.0)
        # 对 ser 调用 clip 方法，指定下限为 2.0
        s_clipped_lower = ser.clip(lower=2.0)

        # 创建期望的上限 Series
        expected_upper = Series([nulls_fixture, 1.0, 2.0], dtype=any_numeric_ea_dtype)
        # 创建期望的下限 Series
        expected_lower = Series([nulls_fixture, 2.0, 3.0], dtype=any_numeric_ea_dtype)

        # 断言 s_clipped_upper 和 expected_upper 相等
        tm.assert_series_equal(s_clipped_upper, expected_upper)
        # 断言 s_clipped_lower 和 expected_lower 相等
        tm.assert_series_equal(s_clipped_lower, expected_lower)

    # 定义 test_clip_with_na_args 方法
    def test_clip_with_na_args(self):
        """Should process np.nan argument as None"""
        # GH#17276
        # 创建包含整数的 Series
        s = Series([1, 2, 3])

        # 断言 s 调用 clip 方法，传入 np.nan 后结果与原始 Series 相等
        tm.assert_series_equal(s.clip(np.nan), Series([1, 2, 3]))
        # 断言 s 调用 clip 方法，传入上限和下限均为 np.nan 后结果与原始 Series 相等
        tm.assert_series_equal(s.clip(upper=np.nan, lower=np.nan), Series([1, 2, 3]))

        # GH#19992

        # 对 s 调用 clip 方法，传入下限为 [0, 4, np.nan]
        res = s.clip(lower=[0, 4, np.nan])
        # 断言 res 与期望的 Series 相等
        tm.assert_series_equal(res, Series([1, 4, 3.0]))
        # 对 s 调用 clip 方法，传入上限为 [1, np.nan, 1]
        res = s.clip(upper=[1, np.nan, 1])
        # 断言 res 与期望的 Series 相等
        tm.assert_series_equal(res, Series([1, 2, 1.0]))

        # GH#40420
        # 创建包含整数的 Series
        s = Series([1, 2, 3])
        # 对 s 调用 clip 方法，传入下限为 0，上限为 [np.nan, np.nan, np.nan]
        result = s.clip(0, [np.nan, np.nan, np.nan])
        # 断言 s 和 result 相等
        tm.assert_series_equal(s, result)
    # 定义测试方法，用于测试 Series 对象的 clip 方法与 Series 对象之间的交互作用
    def test_clip_against_series(self):
        # GH#6966
        # 创建包含浮点数的 Series 对象
        s = Series([1.0, 1.0, 4.0])

        # 创建两个 Series 对象作为下限和上限
        lower = Series([1.0, 2.0, 3.0])
        upper = Series([1.5, 2.5, 3.5])

        # 调用 Series 对象的 clip 方法，对 s 应用下限和上限，验证结果是否符合预期
        tm.assert_series_equal(s.clip(lower, upper), Series([1.0, 2.0, 3.5]))
        
        # 再次调用 clip 方法，但这次上限是标量值，验证结果是否符合预期
        tm.assert_series_equal(s.clip(1.5, upper), Series([1.5, 1.5, 3.5]))

    # 使用 pytest 的参数化装饰器标记，定义另一个测试方法，用于测试 Series 对象的 clip 方法与列表类似对象之间的交互作用
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("upper", [[1, 2, 3], np.asarray([1, 2, 3])])
    def test_clip_against_list_like(self, inplace, upper):
        # GH#15390
        # 创建原始的 Series 对象
        original = Series([5, 6, 7])
        
        # 调用 Series 对象的 clip 方法，应用上限和是否原地修改的参数，验证结果是否符合预期
        result = original.clip(upper=upper, inplace=inplace)
        
        # 创建预期的 Series 对象，用于与结果进行比较
        expected = Series([1, 2, 3])

        # 如果 inplace 参数为 True，则结果应该是原始对象，再次赋值给 result 进行验证
        if inplace:
            result = original
        
        # 使用测试模块的方法验证 result 是否与 expected 一致
        tm.assert_series_equal(result, expected, check_exact=True)

    # 定义测试方法，用于测试 Series 对象的 clip 方法与日期时间对象之间的交互作用
    def test_clip_with_datetimes(self):
        # GH#11838
        # 创建一个时间戳对象 t
        t = Timestamp("2015-12-01 09:30:30")
        
        # 创建包含时间戳对象的 Series 对象 s
        s = Series([Timestamp("2015-12-01 09:30:00"), Timestamp("2015-12-01 09:31:00")])
        
        # 调用 Series 对象的 clip 方法，应用上限 t，验证结果是否符合预期
        result = s.clip(upper=t)
        
        # 创建预期的 Series 对象，用于与结果进行比较
        expected = Series(
            [Timestamp("2015-12-01 09:30:00"), Timestamp("2015-12-01 09:30:30")]
        )
        
        # 使用测试模块的方法验证 result 是否与 expected 一致
        tm.assert_series_equal(result, expected)

        # 创建一个带时区的时间戳对象 t
        t = Timestamp("2015-12-01 09:30:30", tz="US/Eastern")
        
        # 创建带时区的 Series 对象 s
        s = Series(
            [
                Timestamp("2015-12-01 09:30:00", tz="US/Eastern"),
                Timestamp("2015-12-01 09:31:00", tz="US/Eastern"),
            ]
        )
        
        # 再次调用 Series 对象的 clip 方法，应用上限 t，验证结果是否符合预期
        result = s.clip(upper=t)
        
        # 创建预期的 Series 对象，用于与结果进行比较
        expected = Series(
            [
                Timestamp("2015-12-01 09:30:00", tz="US/Eastern"),
                Timestamp("2015-12-01 09:30:30", tz="US/Eastern"),
            ]
        )
        
        # 使用测试模块的方法验证 result 是否与 expected 一致
        tm.assert_series_equal(result, expected)

    # 定义测试方法，用于测试 Series 对象的 clip 方法与包含超出范围的日期时间对象之间的交互作用
    def test_clip_with_timestamps_and_oob_datetimes_object(self):
        # GH-42794
        # 创建包含 datetime 对象的 Series 对象 ser
        ser = Series([datetime(1, 1, 1), datetime(9999, 9, 9)], dtype=object)
        
        # 调用 Series 对象的 clip 方法，应用下限和上限，验证结果是否符合预期
        result = ser.clip(lower=Timestamp.min, upper=Timestamp.max)
        
        # 创建预期的 Series 对象，用于与结果进行比较
        expected = Series([Timestamp.min, Timestamp.max], dtype=object)
        
        # 使用测试模块的方法验证 result 是否与 expected 一致
        tm.assert_series_equal(result, expected)

    # 定义测试方法，用于测试 Series 对象的 clip 方法与包含超出范围的日期时间对象（非纳秒精度）之间的交互作用
    def test_clip_with_timestamps_and_oob_datetimes_non_nano(self):
        # GH#56410
        # 指定 dtype 为 "M8[us]" 的时间戳对象 ser
        dtype = "M8[us]"
        ser = Series([datetime(1, 1, 1), datetime(9999, 9, 9)], dtype=dtype)

        # 定义异常消息内容
        msg = (
            r"Incompatible \(high-resolution\) value for dtype='datetime64\[us\]'. "
            "Explicitly cast before operating"
        )
        
        # 使用 pytest 的异常断言方法，验证超出范围的操作是否会引发 OutOfBoundsDatetime 异常
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            ser.clip(lower=Timestamp.min, upper=Timestamp.max)

        # 将 Timestamp.min 和 Timestamp.max 转换为微秒单位，并调用 Series 对象的 clip 方法，验证结果是否符合预期
        lower = Timestamp.min.as_unit("us")
        upper = Timestamp.max.as_unit("us")
        result = ser.clip(lower=lower, upper=upper)
        
        # 创建预期的 Series 对象，用于与结果进行比较
        expected = Series([lower, upper], dtype=dtype)
        
        # 使用测试模块的方法验证 result 是否与 expected 一致
        tm.assert_series_equal(result, expected)
```