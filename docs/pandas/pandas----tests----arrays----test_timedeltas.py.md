# `D:\src\scipysrc\pandas\pandas\tests\arrays\test_timedeltas.py`

```
from datetime import timedelta  # 导入timedelta类，用于处理时间间隔的计算

import numpy as np  # 导入NumPy库，用于数组操作
import pytest  # 导入pytest库，用于编写和运行测试

import pandas as pd  # 导入Pandas库，用于数据分析
from pandas import Timedelta  # 导入Timedelta类，处理Pandas中的时间间隔数据
import pandas._testing as tm  # 导入Pandas测试模块，用于测试工具
from pandas.core.arrays import (  # 从Pandas核心数组模块导入以下类
    DatetimeArray,  # 时间数组类
    TimedeltaArray,  # 时间间隔数组类
)


class TestNonNano:
    @pytest.fixture(params=["s", "ms", "us"])  # 使用pytest的装饰器fixture，参数化测试时间单位
    def unit(self, request):
        return request.param

    @pytest.fixture  # 使用pytest的装饰器fixture，创建时间间隔数组的测试数据
    def tda(self, unit):
        arr = np.arange(5, dtype=np.int64).view(f"m8[{unit}]")  # 创建NumPy数组，视图为指定单位的时间间隔
        return TimedeltaArray._simple_new(arr, dtype=arr.dtype)

    def test_non_nano(self, unit):
        arr = np.arange(5, dtype=np.int64).view(f"m8[{unit}]")  # 创建NumPy数组，视图为指定单位的时间间隔
        tda = TimedeltaArray._simple_new(arr, dtype=arr.dtype)  # 创建时间间隔数组对象

        assert tda.dtype == arr.dtype  # 断言时间间隔数组的数据类型与原始数组相同
        assert tda[0].unit == unit  # 断言时间间隔数组的第一个元素的单位与参数化的单位相同

    def test_as_unit_raises(self, tda):
        # GH#50616
        with pytest.raises(ValueError, match="Supported units"):  # 使用pytest断言抛出指定异常和匹配的消息
            tda.as_unit("D")  # 调用as_unit方法，尝试转换时间间隔单位为天

        tdi = pd.Index(tda)  # 创建Pandas索引对象
        with pytest.raises(ValueError, match="Supported units"):  # 使用pytest断言抛出指定异常和匹配的消息
            tdi.as_unit("D")  # 调用as_unit方法，尝试转换时间间隔单位为天

    @pytest.mark.parametrize("field", TimedeltaArray._field_ops)  # 使用pytest的参数化测试，遍历时间间隔数组的字段操作
    def test_fields(self, tda, field):
        as_nano = tda._ndarray.astype("m8[ns]")  # 将时间间隔数组的底层NumPy数组转换为纳秒单位的视图
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)  # 创建纳秒单位的时间间隔数组对象

        result = getattr(tda, field)  # 获取时间间隔数组对象的指定字段操作结果
        expected = getattr(tda_nano, field)  # 获取纳秒时间间隔数组对象的指定字段操作结果
        tm.assert_numpy_array_equal(result, expected)  # 使用Pandas测试工具断言两个数组相等

    def test_to_pytimedelta(self, tda):
        as_nano = tda._ndarray.astype("m8[ns]")  # 将时间间隔数组的底层NumPy数组转换为纳秒单位的视图
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)  # 创建纳秒单位的时间间隔数组对象

        result = tda.to_pytimedelta()  # 调用to_pytimedelta方法，将时间间隔数组转换为Python原生timedelta对象
        expected = tda_nano.to_pytimedelta()  # 将纳秒时间间隔数组转换为Python原生timedelta对象
        tm.assert_numpy_array_equal(result, expected)  # 使用Pandas测试工具断言两个数组相等

    def test_total_seconds(self, unit, tda):
        as_nano = tda._ndarray.astype("m8[ns]")  # 将时间间隔数组的底层NumPy数组转换为纳秒单位的视图
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)  # 创建纳秒单位的时间间隔数组对象

        result = tda.total_seconds()  # 计算时间间隔数组中每个元素的总秒数
        expected = tda_nano.total_seconds()  # 计算纳秒时间间隔数组中每个元素的总秒数
        tm.assert_numpy_array_equal(result, expected)  # 使用Pandas测试工具断言两个数组相等

    def test_timedelta_array_total_seconds(self):
        # GH34290
        expected = Timedelta("2 min").total_seconds()  # 计算2分钟的总秒数

        result = pd.array([Timedelta("2 min")]).total_seconds()[0]  # 计算数组中元素（2分钟时间间隔）的总秒数
        assert result == expected  # 断言计算结果与期望结果相等

    def test_total_seconds_nanoseconds(self):
        # issue #48521
        start_time = pd.Series(["2145-11-02 06:00:00"]).astype("datetime64[ns]")  # 创建Pandas Series对象，表示起始时间
        end_time = pd.Series(["2145-11-02 07:06:00"]).astype("datetime64[ns]")  # 创建Pandas Series对象，表示结束时间
        expected = (end_time - start_time).values / np.timedelta64(1, "s")  # 计算时间间隔的总秒数
        result = (end_time - start_time).dt.total_seconds().values  # 计算时间间隔的总秒数（通过dt属性）
        assert result == expected  # 断言计算结果与期望结果相等

    @pytest.mark.parametrize(
        "nat", [np.datetime64("NaT", "ns"), np.datetime64("NaT", "us")]
    )  # 使用pytest的参数化测试，指定NaT（Not a Time）的时间单位为纳秒和微秒
    # 测试方法：对DatetimeArray类型实例和标量nat进行加法操作
    def test_add_nat_datetimelike_scalar(self, nat, tda):
        # 执行加法操作
        result = tda + nat
        # 断言结果类型为DatetimeArray
        assert isinstance(result, DatetimeArray)
        # 断言结果的_creso属性与tda的_creso属性相同
        assert result._creso == tda._creso
        # 断言结果中所有值都为NaN
        assert result.isna().all()

        # 交换操作数进行加法操作
        result = nat + tda
        # 断言结果类型为DatetimeArray
        assert isinstance(result, DatetimeArray)
        # 断言结果的_creso属性与tda的_creso属性相同
        assert result._creso == tda._creso
        # 断言结果中所有值都为NaN
        assert result.isna().all()

    # 测试方法：对DatetimeArray类型实例和pd.NaT进行加法操作
    def test_add_pdnat(self, tda):
        # 执行加法操作
        result = tda + pd.NaT
        # 断言结果类型为TimedeltaArray
        assert isinstance(result, TimedeltaArray)
        # 断言结果的_creso属性与tda的_creso属性相同
        assert result._creso == tda._creso
        # 断言结果中所有值都为NaN
        assert result.isna().all()

        # 交换操作数进行加法操作
        result = pd.NaT + tda
        # 断言结果类型为TimedeltaArray
        assert isinstance(result, TimedeltaArray)
        # 断言结果的_creso属性与tda的_creso属性相同
        assert result._creso == tda._creso
        # 断言结果中所有值都为NaN
        assert result.isna().all()

    # TODO: 2022-07-11 这是唯一一个会涉及到DTA.tz_convert或tz_localize的测试用例; 实现针对此功能的特定测试。
    # 测试方法：对DatetimeArray类型实例和标量ts进行加法操作
    def test_add_datetimelike_scalar(self, tda, tz_naive_fixture):
        # 创建一个带时区的Timestamp实例并转换为纳秒单位
        ts = pd.Timestamp("2016-01-01", tz=tz_naive_fixture).as_unit("ns")

        # 计算预期结果
        expected = tda.as_unit("ns") + ts
        # 执行加法操作
        res = tda + ts
        # 使用测试框架断言扩展数组相等
        tm.assert_extension_array_equal(res, expected)
        # 执行交换操作数的加法
        res = ts + tda
        # 使用测试框架断言扩展数组相等
        tm.assert_extension_array_equal(res, expected)

        # 在Timestamp实例上增加一天，这是无法无损转换的情况
        ts += Timedelta(1)

        # 计算期望结果的数值部分
        exp_values = tda._ndarray + ts.asm8
        # 构建预期结果的DatetimeArray实例，设定其时区为UTC并转换为ts的时区
        expected = (
            DatetimeArray._simple_new(exp_values, dtype=exp_values.dtype)
            .tz_localize("UTC")
            .tz_convert(ts.tz)
        )

        # 执行加法操作
        result = tda + ts
        # 使用测试框架断言扩展数组相等
        tm.assert_extension_array_equal(result, expected)

        # 执行交换操作数的加法
        result = ts + tda
        # 使用测试框架断言扩展数组相等
        tm.assert_extension_array_equal(result, expected)

    # 测试方法：对DatetimeArray类型实例和标量other进行乘法操作
    def test_mul_scalar(self, tda):
        # 定义乘法的标量other
        other = 2
        # 执行乘法操作
        result = tda * other
        # 构建预期的TimedeltaArray实例
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        # 使用测试框架断言扩展数组相等
        tm.assert_extension_array_equal(result, expected)
        # 断言结果的_creso属性与tda的_creso属性相同
        assert result._creso == tda._creso

    # 测试方法：对DatetimeArray类型实例和数组other进行乘法操作
    def test_mul_listlike(self, tda):
        # 创建一个与tda长度相同的数组other
        other = np.arange(len(tda))
        # 执行乘法操作
        result = tda * other
        # 构建预期的TimedeltaArray实例
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        # 使用测试框架断言扩展数组相等
        tm.assert_extension_array_equal(result, expected)
        # 断言结果的_creso属性与tda的_creso属性相同
        assert result._creso == tda._creso

    # 测试方法：对DatetimeArray类型实例和以对象形式表示的数组other进行乘法操作
    def test_mul_listlike_object(self, tda):
        # 创建一个与tda长度相同的数组other，并转换为对象数组
        other = np.arange(len(tda))
        result = tda * other.astype(object)
        # 构建预期的TimedeltaArray实例
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        # 使用测试框架断言扩展数组相等
        tm.assert_extension_array_equal(result, expected)
        # 断言结果的_creso属性与tda的_creso属性相同
        assert result._creso == tda._creso

    # 测试方法：对DatetimeArray类型实例和数值标量other进行除法操作
    def test_div_numeric_scalar(self, tda):
        # 定义除法的数值标量other
        other = 2
        # 执行除法操作
        result = tda / other
        # 构建预期的TimedeltaArray实例
        expected = TimedeltaArray._simple_new(tda._ndarray / other, dtype=tda.dtype)
        # 使用测试框架断言扩展数组相等
        tm.assert_extension_array_equal(result, expected)
        # 断言结果的_creso属性与tda的_creso属性相同
        assert result._creso == tda._creso
    # 定义测试方法，用于测试时间增量数组的标量除法
    def test_div_td_scalar(self, tda):
        # 创建一个时间增量对象，代表1秒的时间增量
        other = timedelta(seconds=1)
        # 对时间增量数组进行除法操作
        result = tda / other
        # 计算预期结果，将时间增量数组转换为以秒为单位的numpy数组
        expected = tda._ndarray / np.timedelta64(1, "s")
        # 使用断言检查两个numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法，用于测试时间增量数组与数值数组的除法
    def test_div_numeric_array(self, tda):
        # 创建一个与时间增量数组长度相同的数值数组
        other = np.arange(len(tda))
        # 对时间增量数组进行除法操作
        result = tda / other
        # 计算预期结果，使用时间增量数组与数值数组的简单元素级除法
        expected = TimedeltaArray._simple_new(tda._ndarray / other, dtype=tda.dtype)
        # 使用断言检查两个扩展数组是否相等
        tm.assert_extension_array_equal(result, expected)
        # 使用断言检查结果数组的某个属性与原始时间增量数组的某个属性是否相等
        assert result._creso == tda._creso

    # 定义测试方法，用于测试时间增量数组与自身的数组操作
    def test_div_td_array(self, tda):
        # 创建一个数组，该数组是时间增量数组与其最后一个元素相加的结果
        other = tda._ndarray + tda._ndarray[-1]
        # 对时间增量数组进行除法操作
        result = tda / other
        # 计算预期结果，时间增量数组与自身数组的元素级除法
        expected = tda._ndarray / other
        # 使用断言检查两个numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法，用于测试时间增量数组与类似时间增量数组的加减法操作
    def test_add_timedeltaarraylike(self, tda):
        # 将时间增量数组转换为以纳秒为单位的数组
        tda_nano = tda.astype("m8[ns]")

        # 计算预期结果，时间增量数组乘以2
        expected = tda_nano * 2
        # 执行时间增量数组与时间增量数组类似的加法操作
        res = tda_nano + tda
        # 使用断言检查两个扩展数组是否相等
        tm.assert_extension_array_equal(res, expected)
        
        # 执行时间增量数组与时间增量数组类似的加法操作（交换顺序）
        res = tda + tda_nano
        # 使用断言检查两个扩展数组是否相等
        tm.assert_extension_array_equal(res, expected)

        # 计算预期结果，时间增量数组减去自身
        expected = tda_nano * 0
        # 执行时间增量数组与时间增量数组类似的减法操作
        res = tda - tda_nano
        # 使用断言检查两个扩展数组是否相等
        tm.assert_extension_array_equal(res, expected)

        # 执行时间增量数组与时间增量数组类似的减法操作（交换顺序）
        res = tda_nano - tda
        # 使用断言检查两个扩展数组是否相等
        tm.assert_extension_array_equal(res, expected)
class TestTimedeltaArray:
    # 测试将时间增量数组转换为整数类型的方法
    def test_astype_int(self, any_int_numpy_dtype):
        # 创建时间增量数组，包含两个小时的时间增量
        arr = TimedeltaArray._from_sequence(
            [Timedelta("1h"), Timedelta("2h")], dtype="m8[ns]"
        )

        # 如果指定的 NumPy 数据类型不是 int64 类型，验证是否会抛出 TypeError 异常
        if np.dtype(any_int_numpy_dtype) != np.int64:
            with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
                arr.astype(any_int_numpy_dtype)
            return

        # 将时间增量数组转换为指定的整数类型
        result = arr.astype(any_int_numpy_dtype)
        # 期望结果是时间增量数组的底层 ndarray 视图
        expected = arr._ndarray.view("i8")
        tm.assert_numpy_array_equal(result, expected)

    # 测试设置数组元素并清除频率设置
    def test_setitem_clears_freq(self):
        # 创建包含两个小时时间增量的时间范围对象的数据表示
        a = pd.timedelta_range("1h", periods=2, freq="h")._data
        # 修改第一个元素为1小时的时间增量
        a[0] = Timedelta("1h")
        # 验证修改后频率是否为 None
        assert a.freq is None

    # 参数化测试，验证设置不同对象类型到数组元素的行为
    @pytest.mark.parametrize(
        "obj",
        [
            Timedelta(seconds=1),
            Timedelta(seconds=1).to_timedelta64(),
            Timedelta(seconds=1).to_pytimedelta(),
        ],
    )
    def test_setitem_objects(self, obj):
        # 创建包含四个小时时间增量的时间范围对象的数据表示
        tdi = pd.timedelta_range("2 Days", periods=4, freq="h")
        arr = tdi._data

        # 将不同类型的时间增量对象设置到数组的第一个元素
        arr[0] = obj
        # 验证第一个元素是否为 Timedelta(seconds=1)
        assert arr[0] == Timedelta(seconds=1)

    # 参数化测试，验证对不兼容类型执行 searchsorted 方法时的行为
    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            1.0,
            np.datetime64("NaT"),
            pd.Timestamp("2021-01-01"),
            "invalid",
            np.arange(10, dtype="i8") * 24 * 3600 * 10**9,
            (np.arange(10) * 24 * 3600 * 10**9).view("datetime64[ns]"),
            pd.Timestamp("2021-01-01").to_period("D"),
        ],
    )
    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_invalid_types(self, other, index):
        # 创建包含十个元素的时间增量索引数据表示
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.TimedeltaIndex(data, freq="D")._data
        if index:
            arr = pd.Index(arr)

        # 构造异常消息正则表达式
        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Timedelta', 'NaT', or array of those. Got",
            ]
        )
        # 验证在给定的其他类型下执行 searchsorted 是否会抛出 TypeError 异常并匹配消息
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(other)


class TestUnaryOps:
    # 测试时间增量数组的绝对值运算
    def test_abs(self):
        # 创建包含多种时间增量的 NumPy 数组
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(vals)

        # 创建期望结果的时间增量数组
        evals = np.array([3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        expected = TimedeltaArray._from_sequence(evals)

        # 测试时间增量数组的绝对值操作
        result = abs(arr)
        tm.assert_timedelta_array_equal(result, expected)

        # 测试 NumPy 的绝对值函数对时间增量数组的影响
        result2 = np.abs(arr)
        tm.assert_timedelta_array_equal(result2, expected)
    # 定义测试正数操作的方法
    def test_pos(self):
        # 创建包含 timedelta 数据的 NumPy 数组，包括一个大负数、"NaT" 和一个大正数
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        # 使用 TimedeltaArray 的 _from_sequence 方法生成 TimedeltaArray 对象
        arr = TimedeltaArray._from_sequence(vals)

        # 执行正数操作，结果应与原数组相同
        result = +arr
        tm.assert_timedelta_array_equal(result, arr)
        # 检查结果与原数组不共享内存
        assert not tm.shares_memory(result, arr)

        # 使用 NumPy 的 positive 函数执行相同的正数操作
        result2 = np.positive(arr)
        tm.assert_timedelta_array_equal(result2, arr)
        # 检查结果与原数组不共享内存
        assert not tm.shares_memory(result2, arr)

    # 定义测试负数操作的方法
    def test_neg(self):
        # 创建包含 timedelta 数据的 NumPy 数组，包括一个大负数、"NaT" 和一个大正数
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        # 使用 TimedeltaArray 的 _from_sequence 方法生成 TimedeltaArray 对象
        arr = TimedeltaArray._from_sequence(vals)

        # 创建预期的结果 TimedeltaArray，包含相反符号的值
        evals = np.array([3600 * 10**9, "NaT", -7200 * 10**9], dtype="m8[ns]")
        expected = TimedeltaArray._from_sequence(evals)

        # 执行负数操作，结果应与预期的结果相等
        result = -arr
        tm.assert_timedelta_array_equal(result, expected)

        # 使用 NumPy 的 negative 函数执行相同的负数操作
        result2 = np.negative(arr)
        tm.assert_timedelta_array_equal(result2, expected)

    # 定义测试带频率的负数操作的方法
    def test_neg_freq(self):
        # 创建一个包含时间增量的时间段索引
        tdi = pd.timedelta_range("2 Days", periods=4, freq="h")
        # 获取时间段索引的数据部分
        arr = tdi._data

        # 创建预期的结果，对时间段索引数据的每个值执行负数操作
        expected = -tdi._data

        # 执行负数操作，结果应与预期的结果相等
        result = -arr
        tm.assert_timedelta_array_equal(result, expected)

        # 使用 NumPy 的 negative 函数执行相同的负数操作
        result2 = np.negative(arr)
        tm.assert_timedelta_array_equal(result2, expected)
```