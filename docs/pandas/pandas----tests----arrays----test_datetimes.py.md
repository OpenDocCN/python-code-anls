# `D:\src\scipysrc\pandas\pandas\tests\arrays\test_datetimes.py`

```
"""
Tests for DatetimeArray
"""

# 导入必要的模块和类
from __future__ import annotations

from datetime import timedelta  # 导入 timedelta 类
import operator  # 导入 operator 模块，用于操作符函数

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas._libs.tslibs import tz_compare  # 导入时区比较函数

from pandas.core.dtypes.dtypes import DatetimeTZDtype  # 导入 DatetimeTZDtype 类

import pandas as pd  # 导入 Pandas 库，用于数据分析
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.core.arrays import (  # 导入 Pandas 数组相关模块
    DatetimeArray,  # DatetimeArray 类，处理日期时间数组
    TimedeltaArray,  # TimedeltaArray 类，处理时间间隔数组
)


class TestNonNano:
    @pytest.fixture(params=["s", "ms", "us"])
    def unit(self, request):
        """Fixture returning parametrized time units"""
        return request.param  # 返回参数化的时间单位

    @pytest.fixture
    def dtype(self, unit, tz_naive_fixture):
        tz = tz_naive_fixture
        if tz is None:
            return np.dtype(f"datetime64[{unit}]")  # 返回不带时区的 dtype
        else:
            return DatetimeTZDtype(unit=unit, tz=tz)  # 返回带时区的 DatetimeTZDtype

    @pytest.fixture
    def dta_dti(self, unit, dtype):
        tz = getattr(dtype, "tz", None)

        dti = pd.date_range("2016-01-01", periods=55, freq="D", tz=tz)
        if tz is None:
            arr = np.asarray(dti).astype(f"M8[{unit}]")  # 转换为特定单位的 NumPy 数组
        else:
            arr = np.asarray(dti.tz_convert("UTC").tz_localize(None)).astype(
                f"M8[{unit}]"
            )  # 转换为 UTC 时区的特定单位的 NumPy 数组

        dta = DatetimeArray._simple_new(arr, dtype=dtype)  # 创建 DatetimeArray 对象
        return dta, dti  # 返回 DatetimeArray 对象和日期时间索引对象

    @pytest.fixture
    def dta(self, dta_dti):
        dta, dti = dta_dti
        return dta  # 返回 DatetimeArray 对象

    def test_non_nano(self, unit, dtype):
        arr = np.arange(5, dtype=np.int64).view(f"M8[{unit}]")
        dta = DatetimeArray._simple_new(arr, dtype=dtype)

        assert dta.dtype == dtype  # 断言 DatetimeArray 的 dtype 是否符合预期
        assert dta[0].unit == unit  # 断言 DatetimeArray 中第一个元素的单位是否符合预期
        assert tz_compare(dta.tz, dta[0].tz)  # 断言时区是否相同
        assert (dta[0] == dta[:1]).all()  # 断言第一个元素和切片后的元素是否相等

    @pytest.mark.parametrize(
        "field", DatetimeArray._field_ops + DatetimeArray._bool_ops
    )
    def test_fields(self, unit, field, dtype, dta_dti):
        dta, dti = dta_dti

        assert (dti == dta).all()  # 断言 DatetimeArray 和日期时间索引对象是否相等

        res = getattr(dta, field)  # 获取 DatetimeArray 对象的特定属性
        expected = getattr(dti._data, field)  # 获取日期时间索引对象的数据的特定属性
        tm.assert_numpy_array_equal(res, expected)  # 断言两者特定属性的数组是否相等

    def test_normalize(self, unit):
        dti = pd.date_range("2016-01-01 06:00:00", periods=55, freq="D")
        arr = np.asarray(dti).astype(f"M8[{unit}]")

        dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)

        assert not dta.is_normalized  # 断言 DatetimeArray 对象是否未被标准化

        # TODO: simplify once we can just .astype to other unit
        exp = np.asarray(dti.normalize()).astype(f"M8[{unit}]")
        expected = DatetimeArray._simple_new(exp, dtype=exp.dtype)

        res = dta.normalize()  # 标准化 DatetimeArray 对象
        tm.assert_extension_array_equal(res, expected)  # 断言标准化后的结果是否与预期一致

    def test_simple_new_requires_match(self, unit):
        arr = np.arange(5, dtype=np.int64).view(f"M8[{unit}]")
        dtype = DatetimeTZDtype(unit, "UTC")

        dta = DatetimeArray._simple_new(arr, dtype=dtype)
        assert dta.dtype == dtype  # 断言创建的 DatetimeArray 对象的 dtype 是否符合预期

        wrong = DatetimeTZDtype("ns", "UTC")
        with pytest.raises(AssertionError, match=""):
            DatetimeArray._simple_new(arr, dtype=wrong)  # 断言创建 DatetimeArray 对象时，传入不匹配的 dtype 是否抛出异常
    # 定义一个测试方法，用于测试非纳秒分辨率的日期时间数组
    def test_std_non_nano(self, unit):
        # 创建一个从"2016-01-01"开始的日期时间索引，包含55天，频率为每天("D")
        dti = pd.date_range("2016-01-01", periods=55, freq="D")
        # 将日期时间索引转换为 numpy 数组，并指定数据类型为指定的时间单位
        arr = np.asarray(dti).astype(f"M8[{unit}]")

        # 调用 DatetimeArray 的静态方法 _simple_new，创建一个新的 DatetimeArray 对象
        dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)

        # 计算日期时间数组的标准差，并与日期时间索引的标准差（按给定单位向下取整）进行比较
        res = dta.std()
        assert res._creso == dta._creso
        assert res == dti.std().floor(unit)

    # 标记此测试以忽略警告信息"Converting to PeriodArray.*:UserWarning"
    @pytest.mark.filterwarnings("ignore:Converting to PeriodArray.*:UserWarning")
    # 定义一个测试方法，用于测试将 DatetimeArray 转换为 PeriodArray 的功能
    def test_to_period(self, dta_dti):
        dta, dti = dta_dti
        # 调用 DatetimeArray 对象的 to_period 方法，将其转换为以天("D")为周期的 PeriodArray 对象
        result = dta.to_period("D")
        # 将对应的日期时间索引对象转换为以天("D")为周期的 PeriodArray 对象，作为预期结果
        expected = dti._data.to_period("D")

        # 使用测试工具函数 tm.assert_extension_array_equal 检查结果与预期是否相等
        tm.assert_extension_array_equal(result, expected)

    # 定义一个测试方法，用于测试日期时间数组的迭代功能
    def test_iter(self, dta):
        # 获取日期时间数组的第一个元素
        res = next(iter(dta))
        # 获取日期时间数组对象的第一个元素，作为预期结果
        expected = dta[0]

        # 断言返回的 res 是 pd.Timestamp 类型
        assert type(res) is pd.Timestamp
        # 断言 res 的值与预期结果的值相等
        assert res._value == expected._value
        # 断言 res 的分辨率与预期结果的分辨率相等
        assert res._creso == expected._creso
        # 断言 res 与预期结果相等
        assert res == expected

    # 定义一个测试方法，用于测试将日期时间数组转换为对象类型的功能
    def test_astype_object(self, dta):
        # 将日期时间数组转换为对象类型
        result = dta.astype(object)
        # 断言转换后的每个元素的分辨率与原始数组的分辨率相同
        assert all(x._creso == dta._creso for x in result)
        # 断言转换后的结果与原始数组的每个元素相等
        assert all(x == y for x, y in zip(result, dta))

    # 定义一个测试方法，用于测试将日期时间数组转换为 Python datetime 对象的功能
    def test_to_pydatetime(self, dta_dti):
        dta, dti = dta_dti

        # 将日期时间数组转换为 Python datetime 对象
        result = dta.to_pydatetime()
        # 将日期时间索引对象转换为 Python datetime 对象，作为预期结果
        expected = dti.to_pydatetime()
        # 使用测试工具函数 tm.assert_numpy_array_equal 检查结果与预期是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 标记此测试方法为参数化测试，参数为["time", "timetz", "date"]
    @pytest.mark.parametrize("meth", ["time", "timetz", "date"])
    # 定义一个测试方法，用于测试日期时间数组的时间和日期方法
    def test_time_date(self, dta_dti, meth):
        dta, dti = dta_dti

        # 获取日期时间数组对象的指定方法（time, timetz, date）
        result = getattr(dta, meth)
        # 获取日期时间索引对象的相应方法（time, timetz, date），作为预期结果
        expected = getattr(dti, meth)
        # 使用测试工具函数 tm.assert_numpy_array_equal 检查结果与预期是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法，用于测试将日期时间数组格式化为本机类型的功能
    def test_format_native_types(self, unit, dtype, dta_dti):
        # 在这种情况下，我们应该得到与非纳秒版本的日期时间索引 dti._data 相同格式的结果
        dta, dti = dta_dti

        # 调用日期时间数组对象的 _format_native_types 方法，返回格式化后的结果
        res = dta._format_native_types()
        # 调用日期时间索引对象 dti._data 的 _format_native_types 方法，返回格式化后的结果，作为预期结果
        exp = dti._data._format_native_types()
        # 使用测试工具函数 tm.assert_numpy_array_equal 检查结果与预期是否相等
        tm.assert_numpy_array_equal(res, exp)

    # 定义一个测试方法，用于测试日期时间数组的字符串表示形式
    def test_repr(self, dta_dti, unit):
        dta, dti = dta_dti

        # 断言日期时间数组对象的字符串表示形式与日期时间索引对象 dti._data 的字符串表示形式相同，
        # 将"[ns"替换为给定的时间单位[unit]
        assert repr(dta) == repr(dti._data).replace("[ns", f"[{unit}")

    # TODO: tests with td64
    # 定义测试函数，用于比较不匹配分辨率的情况下的操作
    def test_compare_mismatched_resolutions(self, comparison_op):
        # 将传入的比较操作符保存到变量op中
        op = comparison_op

        # 获取np.int64的信息
        iinfo = np.iinfo(np.int64)
        # 创建包含最小值、最小值+1、最大值的np.int64数组
        vals = np.array([iinfo.min, iinfo.min + 1, iinfo.max], dtype=np.int64)

        # 将vals数组视图转换为"M8[ns]"类型的数组arr
        arr = np.array(vals).view("M8[ns]")
        # 将arr数组视图转换为"M8[s]"类型的数组arr2
        arr2 = arr.view("M8[s]")

        # 使用DatetimeArray类的_simple_new方法创建左右两个时间数组
        left = DatetimeArray._simple_new(arr, dtype=arr.dtype)
        right = DatetimeArray._simple_new(arr2, dtype=arr2.dtype)

        # 根据不同的比较操作符，设定期望的比较结果数组
        if comparison_op is operator.eq:
            expected = np.array([False, False, False])
        elif comparison_op is operator.ne:
            expected = np.array([True, True, True])
        elif comparison_op in [operator.lt, operator.le]:
            expected = np.array([False, False, True])
        else:
            expected = np.array([False, True, False])

        # 对左右时间数组进行操作，并使用tm.assert_numpy_array_equal断言结果与期望相等
        result = op(left, right)
        tm.assert_numpy_array_equal(result, expected)

        # 对左右时间数组的第二个元素分别进行操作，并使用tm.assert_numpy_array_equal断言结果与期望相等
        result = op(left[1], right)
        tm.assert_numpy_array_equal(result, expected)

        # 如果操作符不是等于或不等于，进行进一步的检查
        if op not in [operator.eq, operator.ne]:
            # 检查numpy是否仍然得出错误的结果；如果问题已修复，则可以移除compare_mismatched_resolutions函数
            np_res = op(left._ndarray, right._ndarray)
            tm.assert_numpy_array_equal(np_res[1:], ~expected[1:])

    # 定义测试函数，用于测试不匹配分辨率时不会降级的情况
    def test_add_mismatched_reso_doesnt_downcast(self):
        # 创建微秒为1的Timedelta对象td
        td = pd.Timedelta(microseconds=1)
        # 创建从"2016-01-01"开始的3个日期范围，每个日期减去td
        dti = pd.date_range("2016-01-01", periods=3) - td
        # 将dti的数据作为微秒单位返回给dta
        dta = dti._data.as_unit("us")

        # 将dta与td相加，结果单位仍为"us"，断言结果单位为"us"
        res = dta + td.as_unit("us")
        assert res.unit == "us"

    # 使用pytest.mark.parametrize装饰器定义测试函数，测试不匹配分辨率时的时间增量标量操作
    @pytest.mark.parametrize(
        "scalar",
        [
            timedelta(hours=2),
            pd.Timedelta(hours=2),
            np.timedelta64(2, "h"),
            np.timedelta64(2 * 3600 * 1000, "ms"),
            pd.offsets.Minute(120),
            pd.offsets.Hour(2),
        ],
    )
    def test_add_timedeltalike_scalar_mismatched_reso(self, dta_dti, scalar):
        # 获取dta_dti元组中的dta和dti
        dta, dti = dta_dti

        # 创建Timedelta对象td，单位为传入的标量scalar
        td = pd.Timedelta(scalar)
        # 获取最细粒度的单位exp_unit，作为期望的时间单位
        exp_unit = tm.get_finest_unit(dta.unit, td.unit)

        # 计算预期结果expected，使用tm.assert_extension_array_equal断言结果与期望相等
        expected = (dti + td)._data.as_unit(exp_unit)
        result = dta + scalar
        tm.assert_extension_array_equal(result, expected)

        # 将标量scalar与dta相加，使用tm.assert_extension_array_equal断言结果与期望相等
        result = scalar + dta
        tm.assert_extension_array_equal(result, expected)

        # 计算预期结果expected，使用tm.assert_extension_array_equal断言结果与期望相等
        expected = (dti - td)._data.as_unit(exp_unit)
        result = dta - scalar
        tm.assert_extension_array_equal(result, expected)
    # 定义测试函数，测试日期时间运算中标量单位不匹配的情况
    def test_sub_datetimelike_scalar_mismatch(self):
        # 创建一个包含三个日期时间的日期范围
        dti = pd.date_range("2016-01-01", periods=3)
        # 将日期时间转换为微秒单位的时间戳
        dta = dti._data.as_unit("us")

        # 获取第一个时间戳，并转换为秒单位
        ts = dta[0].as_unit("s")

        # 计算时间戳数组与第一个时间戳的差值
        result = dta - ts
        # 计算预期结果：日期时间范围减去第一个日期时间的差值，单位为微秒
        expected = (dti - dti[0])._data.as_unit("us")
        
        # 断言结果的数据类型为微秒精度的 datetime64
        assert result.dtype == "m8[us]"
        # 断言计算结果与预期结果相等
        tm.assert_extension_array_equal(result, expected)

    # 定义测试函数，测试 datetime64 对象分辨率不匹配的情况
    def test_sub_datetime64_reso_mismatch(self):
        # 创建一个包含三个日期时间的日期范围
        dti = pd.date_range("2016-01-01", periods=3)
        # 将日期时间转换为秒单位的时间戳
        left = dti._data.as_unit("s")
        # 将左侧时间戳转换为毫秒单位
        right = left.as_unit("ms")

        # 计算左侧时间戳与右侧时间戳的差值
        result = left - right
        # 创建预期的时间差数组，单位为毫秒
        exp_values = np.array([0, 0, 0], dtype="m8[ms]")
        expected = TimedeltaArray._simple_new(
            exp_values,
            dtype=exp_values.dtype,
        )
        # 断言计算结果与预期结果相等
        tm.assert_extension_array_equal(result, expected)
        
        # 计算右侧时间戳与左侧时间戳的差值
        result2 = right - left
        # 断言计算结果与预期结果相等
        tm.assert_extension_array_equal(result2, expected)
class TestDatetimeArrayComparisons:
    # 将此类测试合并到 tests/arithmetic/test_datetime64 中，一旦足够稳定

    def test_cmp_dt64_arraylike_tznaive(self, comparison_op):
        # 使用传入的比较操作函数
        op = comparison_op

        # 创建一个时区非感知的 DatetimeIndex，以"2016-01-01"为起始，月度频率，9个周期
        dti = pd.date_range("2016-01-1", freq="MS", periods=9, tz=None)
        # 从 DatetimeIndex 中获取其底层数据数组
        arr = dti._data
        # 断言数组的频率与原 DatetimeIndex 的频率相同
        assert arr.freq == dti.freq
        # 断言数组是时区非感知的
        assert arr.tz == dti.tz

        # 将 right 设置为 dti
        right = dti

        # 创建一个全为 True 的预期结果数组，数据类型为布尔型
        expected = np.ones(len(arr), dtype=bool)
        # 如果 comparison_op 是 "ne", "gt", "lt" 中的一种，则预期结果应全为 False
        if comparison_op.__name__ in ["ne", "gt", "lt"]:
            expected = ~expected

        # 对数组 arr 执行比较操作，期望结果与预期结果相等
        result = op(arr, arr)
        tm.assert_numpy_array_equal(result, expected)

        # 遍历不同类型的 other 对象进行比较
        for other in [
            right,
            np.array(right),
            list(right),
            tuple(right),
            right.astype(object),
        ]:
            result = op(arr, other)
            tm.assert_numpy_array_equal(result, expected)

            result = op(other, arr)
            tm.assert_numpy_array_equal(result, expected)


class TestDatetimeArray:
    def test_astype_ns_to_ms_near_bounds(self):
        # GH#55979

        # 创建一个时间戳 ts
        ts = pd.Timestamp("1677-09-21 00:12:43.145225")
        # 将时间戳转换为毫秒单位
        target = ts.as_unit("ms")

        # 从时间戳创建一个 DatetimeArray，数据类型为纳秒级
        dta = DatetimeArray._from_sequence([ts], dtype="M8[ns]")
        # 断言 DatetimeArray 的视图与原时间戳的纳秒值相等
        assert (dta.view("i8") == ts.as_unit("ns").value).all()

        # 将 DatetimeArray 类型转换为毫秒级
        result = dta.astype("M8[ms]")
        # 断言转换后的结果与目标时间戳相等
        assert result[0] == target

        # 创建一个预期的 DatetimeArray，数据类型为毫秒级
        expected = DatetimeArray._from_sequence([ts], dtype="M8[ms]")
        # 断言预期的 DatetimeArray 视图与目标时间戳的整数值相等
        assert (expected.view("i8") == target._value).all()

        # 断言转换后的结果与预期结果相等
        tm.assert_datetime_array_equal(result, expected)

    def test_astype_non_nano_tznaive(self):
        # 创建一个时区非感知的 DatetimeIndex
        dti = pd.date_range("2016-01-01", periods=3)

        # 将 DatetimeIndex 转换为秒级
        res = dti.astype("M8[s]")
        # 断言结果的数据类型为秒级
        assert res.dtype == "M8[s]"

        # 获取 DatetimeIndex 的底层数据数组
        dta = dti._data
        # 将底层数据数组转换为秒级
        res = dta.astype("M8[s]")
        # 断言结果的数据类型为秒级
        assert res.dtype == "M8[s]"
        # 断言结果是 DatetimeArray 类型，而不是 ndarray
        assert isinstance(res, pd.core.arrays.DatetimeArray)  # used to be ndarray

    def test_astype_non_nano_tzaware(self):
        # 创建一个时区感知的 DatetimeIndex
        dti = pd.date_range("2016-01-01", periods=3, tz="UTC")

        # 将 DatetimeIndex 转换为秒级，时区设置为 US/Pacific
        res = dti.astype("M8[s, US/Pacific]")
        # 断言结果的数据类型为秒级，时区为 US/Pacific
        assert res.dtype == "M8[s, US/Pacific]"

        # 获取 DatetimeIndex 的底层数据数组
        dta = dti._data
        # 将底层数据数组转换为秒级，时区设置为 US/Pacific
        res = dta.astype("M8[s, US/Pacific]")
        # 断言结果的数据类型为秒级，时区为 US/Pacific
        assert res.dtype == "M8[s, US/Pacific]"

        # 将结果从非纳秒级转换回非纳秒级，保持精度
        res2 = res.astype("M8[s, UTC]")
        # 断言结果的数据类型为秒级，时区为 UTC
        assert res2.dtype == "M8[s, UTC]"
        # 断言结果与原结果不共享内存
        assert not tm.shares_memory(res2, res)

        # 将结果从非纳秒级转换回非纳秒级，保持精度，不复制数据
        res3 = res.astype("M8[s, UTC]", copy=False)
        # 断言结果的数据类型为秒级，时区为 UTC
        assert res2.dtype == "M8[s, UTC]"
        # 断言结果与原结果共享内存
        assert tm.shares_memory(res3, res)

    def test_astype_to_same(self):
        # 创建一个包含时区的 DatetimeArray
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        # 将 DatetimeArray 转换为相同时区的类型，不复制数据
        result = arr.astype(DatetimeTZDtype(tz="US/Central"), copy=False)
        # 断言结果与原数组相同
        assert result is arr

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[ns, UTC]"])
    @pytest.mark.parametrize(
        "other", ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, CET]"]
    )
    # 定义测试函数，使用参数化测试不同的 'other' 值
    def test_astype_copies(self, dtype, other):
        # 创建一个包含整数的 Pandas Series，指定 dtype
        ser = pd.Series([1, 2], dtype=dtype)
        # 复制原始 Series
        orig = ser.copy()

        err = False
        # 检查是否有错误条件
        if (dtype == "datetime64[ns]") ^ (other == "datetime64[ns]"):
            # 若条件满足，标记错误
            err = True

        if err:
            # 若存在错误
            if dtype == "datetime64[ns]":
                # 设置错误消息，提示使用 tz_localize 替代
                msg = "Use obj.tz_localize instead or series.dt.tz_localize instead"
            else:
                # 设置错误消息，从时区感知类型到无时区感知类型的转换
                msg = "from timezone-aware dtype to timezone-naive dtype"
            # 断言会引发 TypeError 异常，并匹配特定的错误消息
            with pytest.raises(TypeError, match=msg):
                ser.astype(other)
        else:
            # 若无错误
            t = ser.astype(other)
            # 将 t 的所有元素设置为 NaT（Not a Time）
            t[:] = pd.NaT
            # 断言两个 Series 相等
            tm.assert_series_equal(ser, orig)

    @pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
    # 定义测试函数，使用参数化测试不同的 'dtype' 值
    def test_astype_int(self, dtype):
        # 创建一个日期时间数组
        arr = DatetimeArray._from_sequence(
            [pd.Timestamp("2000"), pd.Timestamp("2001")], dtype="M8[ns]"
        )

        if np.dtype(dtype) != np.int64:
            # 若 'dtype' 不是 int64 类型，断言会引发 TypeError 异常，并匹配特定的错误消息
            with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
                arr.astype(dtype)
            return

        # 将日期时间数组转换为指定 'dtype' 类型
        result = arr.astype(dtype)
        # 期望结果为数组的视图类型为 'i8'
        expected = arr._ndarray.view("i8")
        # 断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_to_sparse_dt64(self):
        # GH#50082
        # 创建一个日期时间索引
        dti = pd.date_range("2016-01-01", periods=4)
        # 获取日期时间索引的内部数据
        dta = dti._data
        # 将内部数据转换为稀疏的 datetime64[ns] 类型
        result = dta.astype("Sparse[datetime64[ns]]")

        # 断言结果的 dtype 为 'Sparse[datetime64[ns]]'
        assert result.dtype == "Sparse[datetime64[ns]]"
        # 断言结果与原始数据完全相等
        assert (result == dta).all()

    def test_tz_setter_raises(self):
        # 创建一个具有时区信息的日期时间数组
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        # 断言设置时会引发 AttributeError 异常，并匹配特定的错误消息
        with pytest.raises(AttributeError, match="tz_localize"):
            arr.tz = "UTC"

    def test_setitem_str_impute_tz(self, tz_naive_fixture):
        # 类似于 getitem，如果传入类似于时区无关的字符串，则自动补充时区信息
        tz = tz_naive_fixture

        # 创建一个包含日期时间数据的 NumPy 数组
        data = np.array([1, 2, 3], dtype="M8[ns]")
        # 根据是否有时区信息，选择对应的数据类型
        dtype = data.dtype if tz is None else DatetimeTZDtype(tz=tz)
        # 创建日期时间数组
        arr = DatetimeArray._from_sequence(data, dtype=dtype)
        # 复制预期的结果
        expected = arr.copy()

        # 创建一个带时区信息的 Timestamp 对象
        ts = pd.Timestamp("2020-09-08 16:50").tz_localize(tz)
        # 转换为字符串并去除时区信息
        setter = str(ts.tz_localize(None))

        # 设置一个时区无关的标量字符串
        expected[0] = ts
        arr[0] = setter
        # 断言 arr 等于预期的结果
        tm.assert_equal(arr, expected)

        # 设置一个时区无关的字符串列表
        expected[1] = ts
        arr[:2] = [setter, setter]
        # 断言 arr 等于预期的结果
        tm.assert_equal(arr, expected)
    def test_setitem_different_tz_raises(self):
        # 测试在不同时区时设置元素会引发异常
        # 创建包含日期时间数组的 NumPy 数组
        data = np.array([1, 2, 3], dtype="M8[ns]")
        # 使用 DatetimeArray._from_sequence 方法创建 DatetimeArray 对象
        arr = DatetimeArray._from_sequence(
            data, copy=False, dtype=DatetimeTZDtype(tz="US/Central")
        )
        # 使用 pytest.raises 检查是否引发 TypeError 异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match="Cannot compare tz-naive and tz-aware"):
            arr[0] = pd.Timestamp("2000")

        # 创建带有 US/Eastern 时区的 Timestamp 对象
        ts = pd.Timestamp("2000", tz="US/Eastern")
        # 将 DatetimeArray 中的第一个元素设置为 ts
        arr[0] = ts
        # 断言 DatetimeArray 中的第一个元素是否等于经过时区转换后的 ts
        assert arr[0] == ts.tz_convert("US/Central")

    def test_setitem_clears_freq(self):
        # 测试设置元素后是否清除频率信息
        # 创建带有 US/Central 时区的日期范围，并获取其内部数据
        a = pd.date_range("2000", periods=2, freq="D", tz="US/Central")._data
        # 将 a 中的第一个元素设置为带有 US/Central 时区的 Timestamp 对象
        a[0] = pd.Timestamp("2000", tz="US/Central")
        # 断言 a 的频率是否为 None
        assert a.freq is None

    @pytest.mark.parametrize(
        "obj",
        [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-01").to_datetime64(),
            pd.Timestamp("2021-01-01").to_pydatetime(),
        ],
    )
    def test_setitem_objects(self, obj):
        # 测试是否接受 datetime64 和 datetime 类型的对象
        # 创建日期范围，并获取其内部数据
        dti = pd.date_range("2000", periods=2, freq="D")
        arr = dti._data

        # 将 arr 中的第一个元素设置为参数化传入的对象 obj
        arr[0] = obj
        # 断言 arr 中的第一个元素是否等于 obj
        assert arr[0] == obj

    def test_repeat_preserves_tz(self):
        # 测试重复操作是否保留时区信息
        # 创建带有 US/Central 时区的日期范围，并获取其内部数据
        dti = pd.date_range("2000", periods=2, freq="D", tz="US/Central")
        arr = dti._data

        # 对 arr 应用 repeat 方法，指定重复次数
        repeated = arr.repeat([1, 1])

        # 创建期望的 DatetimeArray 对象，从 arr 的 asi8 属性中生成
        expected = DatetimeArray._from_sequence(arr.asi8, dtype=arr.dtype)
        # 使用 tm.assert_equal 断言 repeated 和 expected 是否相等
        tm.assert_equal(repeated, expected)

    def test_value_counts_preserves_tz(self):
        # 测试 value_counts 方法是否保留时区信息
        # 创建带有 US/Central 时区的日期范围，并重复其中的元素
        dti = pd.date_range("2000", periods=2, freq="D", tz="US/Central")
        arr = dti._data.repeat([4, 3])

        # 对 arr 应用 value_counts 方法
        result = arr.value_counts()

        # 断言 result 的索引是否与 dti 相等
        assert result.index.equals(dti)

        # 将 arr 中的倒数第二个元素设置为 NaT（Not a Time）
        arr[-2] = pd.NaT
        # 再次应用 value_counts 方法，包括 NaN 值
        result = arr.value_counts(dropna=False)
        # 创建期望的 Series 对象，包括 NaN 值的计数
        expected = pd.Series([4, 2, 1], index=[dti[0], dti[1], pd.NaT], name="count")
        # 使用 tm.assert_series_equal 断言 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_preserves_tz(self, method):
        # 测试 fillna 方法是否保留时区信息
        # 创建带有 US/Central 时区的日期范围，并从中生成 DatetimeArray 对象
        dti = pd.date_range("2000-01-01", periods=5, freq="D", tz="US/Central")
        arr = DatetimeArray._from_sequence(dti, copy=True)
        # 将 arr 中的第二个元素设置为 NaT
        arr[2] = pd.NaT

        # 根据填充方法选择填充值
        fill_val = dti[1] if method == "pad" else dti[3]
        # 创建期望的 DatetimeArray 对象，包括填充后的值
        expected = DatetimeArray._from_sequence(
            [dti[0], dti[1], fill_val, dti[3], dti[4]],
            dtype=DatetimeTZDtype(tz="US/Central"),
        )

        # 使用 arr 的 _pad_or_backfill 方法填充缺失值
        result = arr._pad_or_backfill(method=method)
        # 使用 tm.assert_extension_array_equal 断言 result 和 expected 是否相等
        tm.assert_extension_array_equal(result, expected)

        # 断言 arr 和 dti 是否没有被原地修改
        assert arr[2] is pd.NaT
        assert dti[2] == pd.Timestamp("2000-01-03", tz="US/Central")
    # 定义测试方法，测试填充二维数组的功能
    def test_fillna_2d(self):
        # 创建一个包含时区信息的日期时间索引
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        # 将日期时间索引的数据重塑为3行2列的二维数组副本
        dta = dti._data.reshape(3, 2).copy()
        # 在数组中人为设定两个位置为 NaT（不可用的日期时间）
        dta[0, 1] = pd.NaT
        dta[1, 0] = pd.NaT

        # 使用填充方法"pad"对数组进行填充或者向后填充
        res1 = dta._pad_or_backfill(method="pad")
        expected1 = dta.copy()
        expected1[1, 0] = dta[0, 0]
        # 断言填充后的结果与预期结果相等
        tm.assert_extension_array_equal(res1, expected1)

        # 使用填充方法"backfill"对数组进行填充或者向前填充
        res2 = dta._pad_or_backfill(method="backfill")
        expected2 = dta.copy()
        expected2[1, 0] = dta[2, 0]
        expected2[0, 1] = dta[1, 1]
        # 断言填充后的结果与预期结果相等
        tm.assert_extension_array_equal(res2, expected2)

        # 使用特定的顺序创建数据的副本，并验证其连续性
        dta2 = dta._from_backing_data(dta._ndarray.copy(order="F"))
        assert dta2._ndarray.flags["F_CONTIGUOUS"]
        assert not dta2._ndarray.flags["C_CONTIGUOUS"]
        # 断言两个数组在值上相等
        tm.assert_extension_array_equal(dta, dta2)

        # 对新数组使用填充方法"pad"，验证结果与之前的预期结果一致
        res3 = dta2._pad_or_backfill(method="pad")
        tm.assert_extension_array_equal(res3, expected1)

        # 对新数组使用填充方法"backfill"，验证结果与之前的预期结果一致
        res4 = dta2._pad_or_backfill(method="backfill")
        tm.assert_extension_array_equal(res4, expected2)

        # 在此处测试 DataFrame 的填充方法
        df = pd.DataFrame(dta)
        # 使用前向填充方法填充 DataFrame
        res = df.ffill()
        expected = pd.DataFrame(expected1)
        # 断言填充后的 DataFrame 与预期结果相等
        tm.assert_frame_equal(res, expected)

        # 使用后向填充方法填充 DataFrame
        res = df.bfill()
        expected = pd.DataFrame(expected2)
        # 断言填充后的 DataFrame 与预期结果相等
        tm.assert_frame_equal(res, expected)

    # 测试带有时区信息的数组接口
    def test_array_interface_tz(self):
        tz = "US/Central"
        # 创建包含时区信息的日期时间索引，并获取其内部数据
        data = pd.date_range("2017", periods=2, tz=tz)._data
        # 将日期时间索引的数据转换为 NumPy 数组
        result = np.asarray(data)

        # 创建预期的 NumPy 数组，其中包含与给定时区相符的日期时间戳
        expected = np.array(
            [
                pd.Timestamp("2017-01-01T00:00:00", tz=tz),
                pd.Timestamp("2017-01-02T00:00:00", tz=tz),
            ],
            dtype=object,
        )
        # 断言 NumPy 数组的内容与预期结果相等
        tm.assert_numpy_array_equal(result, expected)

        # 将日期时间索引的数据转换为具有对象类型的 NumPy 数组
        result = np.asarray(data, dtype=object)
        # 断言 NumPy 数组的内容与预期结果相等
        tm.assert_numpy_array_equal(result, expected)

        # 将日期时间索引的数据转换为特定类型的 NumPy 数组（"M8[ns]"）
        result = np.asarray(data, dtype="M8[ns]")

        # 创建预期的 NumPy 数组，使用给定类型存储日期时间戳
        expected = np.array(
            ["2017-01-01T06:00:00", "2017-01-02T06:00:00"], dtype="M8[ns]"
        )
        # 断言 NumPy 数组的内容与预期结果相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试日期时间索引的数组接口
    def test_array_interface(self):
        # 创建不包含时区信息的日期时间索引，并获取其内部数据
        data = pd.date_range("2017", periods=2)._data
        # 创建预期的 NumPy 数组，其中存储日期时间戳，带有特定的数据类型
        expected = np.array(
            ["2017-01-01T00:00:00", "2017-01-02T00:00:00"], dtype="datetime64[ns]"
        )

        # 将日期时间索引的数据转换为 NumPy 数组
        result = np.asarray(data)
        # 断言 NumPy 数组的内容与预期结果相等
        tm.assert_numpy_array_equal(result, expected)

        # 将日期时间索引的数据转换为具有对象类型的 NumPy 数组
        result = np.asarray(data, dtype=object)
        # 创建预期的 NumPy 数组，其中存储日期时间戳的对象
        expected = np.array(
            [pd.Timestamp("2017-01-01T00:00:00"), pd.Timestamp("2017-01-02T00:00:00")],
            dtype=object,
        )
        # 断言 NumPy 数组的内容与预期结果相等
        tm.assert_numpy_array_equal(result, expected)

    # 使用参数化测试，测试索引为真和假的情况
    @pytest.mark.parametrize("index", [True, False])
    # 定义一个测试方法，用于测试带有不同时区信息的 DatetimeIndex 的 searchsorted 方法
    def test_searchsorted_different_tz(self, index):
        # 创建一个包含 0 到 9 的整数数组，每个元素乘以一天的纳秒数，组成时间戳数据
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        # 创建一个以日为频率的 DatetimeIndex，并设定其时区为 "Asia/Tokyo"
        arr = pd.DatetimeIndex(data, freq="D")._data.tz_localize("Asia/Tokyo")
        # 如果 index 为真，则将 arr 转换为 pd.Index 类型
        if index:
            arr = pd.Index(arr)

        # 计算预期的 searchsorted 结果，用于后续断言比较
        expected = arr.searchsorted(arr[2])
        # 将 arr 中第二个元素转换为 UTC 时区后，再次计算其在 arr 中的搜索位置
        result = arr.searchsorted(arr[2].tz_convert("UTC"))
        # 断言结果与预期相同
        assert result == expected

        # 计算一组范围在 arr[2] 到 arr[5] 的元素在 arr 中的搜索位置
        expected = arr.searchsorted(arr[2:6])
        # 将这些元素转换为 UTC 时区后，再次计算它们在 arr 中的搜索位置
        result = arr.searchsorted(arr[2:6].tz_convert("UTC"))
        # 使用测试工具库 tm 来断言结果与预期相同
        tm.assert_equal(result, expected)

    # 使用 pytest 参数化装饰器定义一个测试方法，用于测试时区感知性的 searchsorted 方法兼容性
    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_tzawareness_compat(self, index):
        # 创建一个包含 0 到 9 的整数数组，每个元素乘以一天的纳秒数，组成时间戳数据
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        # 创建一个以日为频率的 DatetimeIndex
        arr = pd.DatetimeIndex(data, freq="D")._data
        # 如果 index 为真，则将 arr 转换为 pd.Index 类型
        if index:
            arr = pd.Index(arr)

        # 将 arr 的时区设定为 "Asia/Tokyo"
        mismatch = arr.tz_localize("Asia/Tokyo")

        # 定义错误信息字符串，用于验证异常消息
        msg = "Cannot compare tz-naive and tz-aware datetime-like objects"

        # 使用 pytest 断言异常处理上下文，验证 searchsorted 方法对不兼容的时区对象的处理
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(mismatch[0])
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(mismatch)

        with pytest.raises(TypeError, match=msg):
            mismatch.searchsorted(arr[0])
        with pytest.raises(TypeError, match=msg):
            mismatch.searchsorted(arr)

    # 使用 pytest 参数化装饰器定义一个测试方法，用于测试 searchsorted 方法对不兼容类型的处理
    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            1.0,
            np.timedelta64("NaT"),
            pd.Timedelta(days=2),
            "invalid",
            np.arange(10, dtype="i8") * 24 * 3600 * 10**9,
            np.arange(10).view("timedelta64[ns]") * 24 * 3600 * 10**9,
            pd.Timestamp("2021-01-01").to_period("D"),
        ],
    )
    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_invalid_types(self, other, index):
        # 创建一个包含 0 到 9 的整数数组，每个元素乘以一天的纳秒数，组成时间戳数据
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        # 创建一个以日为频率的 DatetimeIndex
        arr = pd.DatetimeIndex(data, freq="D")._data
        # 如果 index 为真，则将 arr 转换为 pd.Index 类型
        if index:
            arr = pd.Index(arr)

        # 定义用于验证异常消息的正则表达式字符串
        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Timestamp', 'NaT', or array of those. Got",
            ]
        )

        # 使用 pytest 断言异常处理上下文，验证 searchsorted 方法对不兼容类型的处理
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(other)

    # 定义一个测试方法，用于测试 shift 方法中的填充值设置
    def test_shift_fill_value(self):
        # 创建一个包含 2016-01-01 到 2016-01-03 的日期范围
        dti = pd.date_range("2016-01-01", periods=3)

        # 获取日期时间数组的内部数据
        dta = dti._data
        # 创建一个预期的日期时间数组，其值为 dta 的循环移位结果
        expected = DatetimeArray._from_sequence(np.roll(dta._ndarray, 1))

        # 获取数组中的最后一个值作为填充值
        fv = dta[-1]

        # 遍历不同类型的填充值进行测试
        for fill_value in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
            # 使用 shift 方法对 dta 进行移位，设置不同的填充值，并使用测试工具库 tm 来断言结果与预期相同
            result = dta.shift(1, fill_value=fill_value)
            tm.assert_datetime_array_equal(result, expected)

        # 将 dta 的时区设定为 UTC，并更新预期值的时区为 UTC
        dta = dta.tz_localize("UTC")
        expected = expected.tz_localize("UTC")
        # 获取数组中的最后一个值作为填充值
        fv = dta[-1]

        # 再次遍历不同类型的填充值进行测试
        for fill_value in [fv, fv.to_pydatetime()]:
            # 使用 shift 方法对 dta 进行移位，设置不同的填充值，并使用测试工具库 tm 来断言结果与预期相同
            result = dta.shift(1, fill_value=fill_value)
            tm.assert_datetime_array_equal(result, expected)
    # 测试函数，用于验证在时区匹配不一致的情况下的移位操作是否会触发异常
    def test_shift_value_tzawareness_mismatch(self):
        # 创建一个日期范围对象，从 "2016-01-01" 开始，包含 3 个时间点
        dti = pd.date_range("2016-01-01", periods=3)
        
        # 获取日期范围对象内部的底层数据
        dta = dti._data
        
        # 从底层数据中取出最后一个时间点，并将其设为 UTC 时区
        fv = dta[-1].tz_localize("UTC")
        
        # 遍历包含无效时间点的列表，分别是 fv 和其转换为 Python datetime 对象后的值
        for invalid in [fv, fv.to_pydatetime()]:
            # 使用 pytest 检查是否会抛出 TypeError 异常，异常信息中包含 "Cannot compare"
            with pytest.raises(TypeError, match="Cannot compare"):
                # 对日期范围数据进行移位操作，填充值为 invalid
                dta.shift(1, fill_value=invalid)
        
        # 将整个日期范围对象的时区设为 UTC
        dta = dta.tz_localize("UTC")
        
        # 取出最后一个时间点，并将其时区设为 None
        fv = dta[-1].tz_localize(None)
        
        # 遍历包含无效时间点的列表，分别是 fv、其转换为 Python datetime 对象后的值，以及转换为 datetime64 后的值
        for invalid in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
            # 使用 pytest 检查是否会抛出 TypeError 异常，异常信息中包含 "Cannot compare"
            with pytest.raises(TypeError, match="Cannot compare"):
                # 对日期范围数据进行移位操作，填充值为 invalid
                dta.shift(1, fill_value=invalid)

    # 测试函数，验证移位操作是否要求时区匹配
    def test_shift_requires_tzmatch(self):
        # 创建一个日期范围对象，从 "2016-01-01" 开始，包含 3 个时间点，时区设为 UTC
        dti = pd.date_range("2016-01-01", periods=3, tz="UTC")
        
        # 获取日期范围对象内部的底层数据
        dta = dti._data
        
        # 创建一个填充值，时区为 "US/Pacific"
        fill_value = pd.Timestamp("2020-10-18 18:44", tz="US/Pacific")
        
        # 进行移位操作，填充值为 fill_value
        result = dta.shift(1, fill_value=fill_value)
        
        # 创建一个预期结果，进行移位操作，填充值为 fill_value 且时区转换为 UTC
        expected = dta.shift(1, fill_value=fill_value.tz_convert("UTC"))
        
        # 使用 pytest 检查 result 和 expected 是否相等
        tm.assert_equal(result, expected)

    # 测试函数，验证时区本地化和去除时区的操作
    def test_tz_localize_t2d(self):
        # 创建一个日期范围对象，从 "1994-05-12" 开始，包含 12 个时间点，时区为 "US/Pacific"
        dti = pd.date_range("1994-05-12", periods=12, tz="US/Pacific")
        
        # 获取日期范围对象内部的底层数据，并将其重新排列成 3 行 4 列的形式
        dta = dti._data.reshape(3, 4)
        
        # 对底层数据进行时区去除操作
        result = dta.tz_localize(None)
        
        # 创建一个预期结果，对底层数据的扁平化结果进行时区去除，并重新形状为原来的形式
        expected = dta.ravel().tz_localize(None).reshape(dta.shape)
        
        # 使用测试模块中的方法检查 result 和 expected 是否相等
        tm.assert_datetime_array_equal(result, expected)
        
        # 对 expected 结果进行再次时区本地化，时区设为 "US/Pacific"
        roundtrip = expected.tz_localize("US/Pacific")
        
        # 使用测试模块中的方法检查 roundtrip 和原始底层数据是否相等
        tm.assert_datetime_array_equal(roundtrip, dta)

    # 使用参数化测试，验证在不同时区设置下的迭代行为
    @pytest.mark.parametrize(
        "tz", ["US/Eastern", "dateutil/US/Eastern", "pytz/US/Eastern"]
    )
    def test_iter_zoneinfo_fold(self, tz):
        # 如果时区以 "pytz/" 开头，则导入 pytest 并跳过测试
        if tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        
        # 创建一个包含时间戳的 NumPy 数组，时间戳表示为 UTC 时间
        utc_vals = np.array(
            [1320552000, 1320555600, 1320559200, 1320562800], dtype=np.int64
        )
        utc_vals *= 1_000_000_000
        
        # 从时间戳数组创建 DatetimeArray 对象，并依次进行时区本地化和转换
        dta = DatetimeArray._from_sequence(utc_vals).tz_localize("UTC").tz_convert(tz)
        
        # 取出第三个时间点，分别命名为 left 和 right
        left = dta[2]
        right = list(dta)[2]
        
        # 使用断言检查 left 和 right 的字符串表示是否相等
        assert str(left) == str(right)
        
        # 使用断言检查 left 和 right 的 UTC 偏移是否相等
        assert left.utcoffset() == right.utcoffset()
        
        # 对 dta 进行对象类型转换，并取出第三个时间点，命名为 right2
        right2 = dta.astype(object)[2]
        
        # 使用断言检查 left 和 right2 的字符串表示是否相等
        assert str(left) == str(right2)
        
        # 使用断言检查 left 和 right2 的 UTC 偏移是否相等
        assert left.utcoffset() == right2.utcoffset()
    # 测试给定的频率（freq）是否会引发异常
    def test_date_range_frequency_M_Q_Y_raises(self, freq):
        # 构造错误消息字符串，指示无效的频率
        msg = f"Invalid frequency: {freq}"

        # 使用 pytest 来验证调用 pd.date_range 函数时是否会抛出 ValueError 异常，并且异常消息与 msg 匹配
        with pytest.raises(ValueError, match=msg):
            pd.date_range("1/1/2000", periods=4, freq=freq)

    # 测试已废弃的大写频率（freq_depr）是否会引发警告
    @pytest.mark.parametrize("freq_depr", ["2H", "2CBH", "2MIN", "2S", "2mS", "2Us"])
    def test_date_range_uppercase_frequency_deprecated(self, freq_depr):
        # GH#9586, GH#54939 提供了频率（freq_depr）的废弃说明
        depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a " \
                   f"future version. Please use '{freq_depr.lower()[1:]}' instead."

        # 期望的日期范围对象，使用小写形式的 freq_depr 作为频率
        expected = pd.date_range("1/1/2000", periods=4, freq=freq_depr.lower())

        # 使用 tm.assert_produces_warning 来验证调用 pd.date_range 函数时是否会产生 FutureWarning 警告，并且警告消息与 depr_msg 匹配
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            # 调用 pd.date_range 函数，传入大写形式的 freq_depr 作为频率，保存结果到 result
            result = pd.date_range("1/1/2000", periods=4, freq=freq_depr)

        # 使用 tm.assert_index_equal 来验证 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)

    # 测试已废弃的小写频率（freq）是否会引发异常
    @pytest.mark.parametrize(
        "freq",
        [
            "2ye-mar",
            "2ys",
            "2qe",
            "2qs-feb",
            "2bqs",
            "2sms",
            "2bms",
            "2cbme",
            "2me",
        ],
    )
    def test_date_range_lowercase_frequency_raises(self, freq):
        # 构造错误消息字符串，指示无效的频率
        msg = f"Invalid frequency: {freq}"

        # 使用 pytest 来验证调用 pd.date_range 函数时是否会抛出 ValueError 异常，并且异常消息与 msg 匹配
        with pytest.raises(ValueError, match=msg):
            pd.date_range("1/1/2000", periods=4, freq=freq)

    # 测试已废弃的小写频率（"w"）是否会引发警告
    def test_date_range_lowercase_frequency_deprecated(self):
        # GH#9586, GH#54939 提供了频率（"w"）的废弃说明
        depr_msg = "'w' is deprecated and will be removed in a future version"

        # 期望的日期范围对象，使用大写形式的 "W" 作为频率
        expected = pd.date_range("1/1/2000", periods=4, freq="2W")

        # 使用 tm.assert_produces_warning 来验证调用 pd.date_range 函数时是否会产生 FutureWarning 警告，并且警告消息与 depr_msg 匹配
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            # 调用 pd.date_range 函数，传入小写形式的 "w" 作为频率，保存结果到 result
            result = pd.date_range("1/1/2000", periods=4, freq="2w")

        # 使用 tm.assert_index_equal 来验证 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)

    # 测试指定年度频率（freq）是否会引发异常
    @pytest.mark.parametrize("freq", ["1A", "2A-MAR", "2a-mar"])
    def test_date_range_frequency_A_raises(self, freq):
        # 构造错误消息字符串，指示无效的频率
        msg = f"Invalid frequency: {freq}"

        # 使用 pytest 来验证调用 pd.date_range 函数时是否会抛出 ValueError 异常，并且异常消息与 msg 匹配
        with pytest.raises(ValueError, match=msg):
            pd.date_range("1/1/2000", periods=4, freq=freq)
# 定义一个测试函数，用于测试 factorize 方法在不指定频率的情况下的行为
def test_factorize_sort_without_freq():
    # 创建一个 DatetimeArray 对象，包含三个日期时间值 [0, 2, 1]，数据类型为 "M8[ns]"
    dta = DatetimeArray._from_sequence([0, 2, 1], dtype="M8[ns]")

    # 定义一个正则表达式消息，用于匹配 pytest.raises 抛出的 NotImplementedError 异常的消息
    msg = r"call pd.factorize\(obj, sort=True\) instead"
    
    # 使用 pytest 框架来验证是否会抛出 NotImplementedError 异常，异常消息需匹配定义的 msg
    with pytest.raises(NotImplementedError, match=msg):
        # 调用 DatetimeArray 对象的 factorize 方法，并期望抛出异常，指定 sort=True 参数
        dta.factorize(sort=True)

    # 在同一作用域内，测试 TimedeltaArray 类型的行为
    # 计算时间差 TimedeltaArray，基于 dta 中的值与第一个值的差值
    tda = dta - dta[0]
    
    # 再次使用 pytest 框架来验证是否会抛出 NotImplementedError 异常，异常消息需匹配定义的 msg
    with pytest.raises(NotImplementedError, match=msg):
        # 调用 TimedeltaArray 对象的 factorize 方法，并期望抛出异常，指定 sort=True 参数
        tda.factorize(sort=True)
```