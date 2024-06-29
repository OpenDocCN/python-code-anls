# `D:\src\scipysrc\pandas\pandas\tests\arrays\datetimes\test_constructors.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas._libs import iNaT  # 导入 pandas 库中的 iNaT 对象，表示缺失的时间戳

from pandas.core.dtypes.dtypes import DatetimeTZDtype  # 导入 pandas 库中的 DatetimeTZDtype 类型，用于处理带时区的日期时间数据类型

import pandas as pd  # 导入 pandas 库，用于数据分析和操作

import pandas._testing as tm  # 导入 pandas 库中的测试工具模块，用于测试结果的验证

from pandas.core.arrays import DatetimeArray  # 导入 pandas 库中的 DatetimeArray 类，用于处理日期时间数组的操作


class TestDatetimeArrayConstructor:
    def test_from_sequence_invalid_type(self):
        # 创建一个 MultiIndex 对象 mi，其中包含两个 np.arange(5) 的笛卡尔积
        mi = pd.MultiIndex.from_product([np.arange(5), np.arange(5)])
        # 使用 pytest 检测是否抛出 TypeError 异常，异常消息包含 "Cannot create a DatetimeArray"
        with pytest.raises(TypeError, match="Cannot create a DatetimeArray"):
            DatetimeArray._from_sequence(mi, dtype="M8[ns]")

    @pytest.mark.parametrize(
        "meth",
        [
            DatetimeArray._from_sequence,  # 将 DatetimeArray._from_sequence 添加到参数列表中
            pd.to_datetime,  # 将 pd.to_datetime 添加到参数列表中
            pd.DatetimeIndex,  # 将 pd.DatetimeIndex 添加到参数列表中
        ],
    )
    def test_mixing_naive_tzaware_raises(self, meth):
        # 创建一个包含两个时间戳的 numpy 数组 arr，一个是无时区的，另一个是使用 "CET" 时区的
        arr = np.array([pd.Timestamp("2000"), pd.Timestamp("2000", tz="CET")])

        # 设置错误消息，用于匹配 ValueError 异常，包含两种情况："Cannot mix tz-aware with tz-naive values" 或者 "Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True"
        msg = (
            "Cannot mix tz-aware with tz-naive values|"
            "Tz-aware datetime.datetime cannot be converted "
            "to datetime64 unless utc=True"
        )

        # 遍历数组的两种倒序情况
        for obj in [arr, arr[::-1]]:
            # 检查无论是先出现 naive 还是 aware，都会引发 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                meth(obj)

    def test_from_pandas_array(self):
        # 创建一个包含 [0, 1, 2, 3, 4] 的 numpy 数组，转换为 pandas 中的 Array，再乘以 3600 * 10**9
        arr = pd.array(np.arange(5, dtype=np.int64)) * 3600 * 10**9

        # 使用 DatetimeArray._from_sequence 方法，创建一个 DatetimeArray 对象，指定 dtype="M8[ns]"
        result = DatetimeArray._from_sequence(arr, dtype="M8[ns]")._with_freq("infer")

        # 创建一个预期的日期范围 expected，从 "1970-01-01" 开始，频率为每小时，共 5 个周期
        expected = pd.date_range("1970-01-01", periods=5, freq="h")._data
        # 使用测试工具模块 tm 来断言 result 和 expected 相等
        tm.assert_datetime_array_equal(result, expected)

    def test_bool_dtype_raises(self):
        # 创建一个包含 [1, 2, 3] 的 numpy 数组，数据类型为布尔型
        arr = np.array([1, 2, 3], dtype="bool")

        # 设置错误消息，用于匹配 TypeError 异常，表明布尔类型不能转换为 "M8[ns]" 类型
        msg = r"dtype bool cannot be converted to datetime64\[ns\]"

        # 使用 pytest 检测是否抛出 TypeError 异常，异常消息匹配 msg
        with pytest.raises(TypeError, match=msg):
            DatetimeArray._from_sequence(arr, dtype="M8[ns]")

        with pytest.raises(TypeError, match=msg):
            pd.DatetimeIndex(arr)

        with pytest.raises(TypeError, match=msg):
            pd.to_datetime(arr)

    def test_copy(self):
        # 创建一个包含 [1, 2, 3] 的 numpy 数组，数据类型为 "M8[ns]"
        data = np.array([1, 2, 3], dtype="M8[ns]")
        # 使用 DatetimeArray._from_sequence 方法，创建一个 DatetimeArray 对象，指定 copy=False
        arr = DatetimeArray._from_sequence(data, copy=False)
        # 断言 arr 的底层 ndarray 与 data 是同一个对象
        assert arr._ndarray is data

        # 使用 DatetimeArray._from_sequence 方法，创建一个 DatetimeArray 对象，指定 copy=True
        arr = DatetimeArray._from_sequence(data, copy=True)
        # 断言 arr 的底层 ndarray 与 data 不是同一个对象
        assert arr._ndarray is not data

    def test_numpy_datetime_unit(self, unit):
        # 创建一个包含 [1, 2, 3] 的 numpy 数组，数据类型为 f"M8[{unit}]"
        data = np.array([1, 2, 3], dtype=f"M8[{unit}]")
        # 使用 DatetimeArray._from_sequence 方法，创建一个 DatetimeArray 对象
        arr = DatetimeArray._from_sequence(data)
        # 断言 arr 的单位（unit）与输入的 unit 相等
        assert arr.unit == unit
        # 断言 arr 的第一个元素的单位（unit）与输入的 unit 相等
        assert arr[0].unit == unit


class TestSequenceToDT64NS:
    def test_tz_dtype_mismatch_raises(self):
        # 创建一个 DatetimeArray 对象 arr，从包含一个字符串 "2000" 的序列开始，指定 dtype=DatetimeTZDtype(tz="US/Central")
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        # 使用 pytest 检测是否抛出 TypeError 异常，异常消息包含 "data is already tz-aware"
        with pytest.raises(TypeError, match="data is already tz-aware"):
            # 使用 DatetimeArray._from_sequence 方法，尝试转换 arr 的时区为 "UTC"
            DatetimeArray._from_sequence(arr, dtype=DatetimeTZDtype(tz="UTC"))
    # 定义一个测试函数，用于验证带时区信息的日期时间类型匹配
    def test_tz_dtype_matches(self):
        # 创建一个带有指定时区的日期时间类型
        dtype = DatetimeTZDtype(tz="US/Central")
        # 从字符串序列创建日期时间数组，并指定数据类型为上述创建的日期时间类型
        arr = DatetimeArray._from_sequence(["2000"], dtype=dtype)
        # 再次从日期时间数组创建一个新的日期时间数组，数据类型仍为指定的日期时间类型
        result = DatetimeArray._from_sequence(arr, dtype=dtype)
        # 使用测试工具函数验证两个日期时间数组是否相等
        tm.assert_equal(arr, result)

    # 使用参数化装饰器标记测试函数，参数为数组的排序方式（F：Fortran风格，C：C语言风格）
    @pytest.mark.parametrize("order", ["F", "C"])
    # 定义一个测试函数，用于验证二维日期时间数组的处理
    def test_2d(self, order):
        # 创建一个具有指定时区的日期时间索引
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        # 将日期时间索引转换为对象类型的NumPy数组，并按照对象类型重塑为3行2列的数组
        arr = np.array(dti, dtype=object).reshape(3, 2)
        # 如果排序方式为F（Fortran风格），则转置数组
        if order == "F":
            arr = arr.T

        # 从数组序列创建日期时间数组，指定数据类型为日期时间索引的数据类型
        res = DatetimeArray._from_sequence(arr, dtype=dti.dtype)
        # 从扁平化的数组序列再次创建日期时间数组，指定数据类型为日期时间索引的数据类型，并按原形状重塑
        expected = DatetimeArray._from_sequence(arr.ravel(), dtype=dti.dtype).reshape(
            arr.shape
        )
        # 使用测试工具函数验证两个日期时间数组是否相等
        tm.assert_datetime_array_equal(res, expected)
# ----------------------------------------------------------------------------
# Arrow interaction

# 定义包含极端值的列表，包括整数、None和特殊常量iNaT
EXTREME_VALUES = [0, 123456789, None, iNaT, 2**63 - 1, -(2**63) + 1]
# 安全的精细到粗略转换的数据列表，包括整数和None
FINE_TO_COARSE_SAFE = [123_000_000_000, None, -123_000_000_000]
# 安全的粗略到精细转换的数据列表，包括整数和None
COARSE_TO_FINE_SAFE = [123, None, -123]

# 使用pytest的@parametrize装饰器定义多个测试参数组合
@pytest.mark.parametrize(
    ("pa_unit", "pd_unit", "pa_tz", "pd_tz", "data"),
    [
        ("s", "s", "UTC", "UTC", EXTREME_VALUES),
        ("ms", "ms", "UTC", "Europe/Berlin", EXTREME_VALUES),
        ("us", "us", "US/Eastern", "UTC", EXTREME_VALUES),
        ("ns", "ns", "US/Central", "Asia/Kolkata", EXTREME_VALUES),
        ("ns", "s", "UTC", "UTC", FINE_TO_COARSE_SAFE),
        ("us", "ms", "UTC", "Europe/Berlin", FINE_TO_COARSE_SAFE),
        ("ms", "us", "US/Eastern", "UTC", COARSE_TO_FINE_SAFE),
        ("s", "ns", "US/Central", "Asia/Kolkata", COARSE_TO_FINE_SAFE),
    ],
)
# 定义测试函数，测试不同单位和时区下的时间转换功能
def test_from_arrow_with_different_units_and_timezones_with(
    pa_unit, pd_unit, pa_tz, pd_tz, data
):
    # 导入pytest模块，并跳过如果pyarrow未安装的情况
    pa = pytest.importorskip("pyarrow")

    # 根据单位和时区创建pyarrow的时间戳类型
    pa_type = pa.timestamp(pa_unit, tz=pa_tz)
    # 根据给定数据创建pyarrow数组
    arr = pa.array(data, type=pa_type)
    # 创建DatetimeTZDtype对象，用于Pandas的时间类型
    dtype = DatetimeTZDtype(unit=pd_unit, tz=pd_tz)

    # 使用dtype对象的__from_arrow__方法从pyarrow数组转换为Pandas的扩展时间数组
    result = dtype.__from_arrow__(arr)
    # 根据原始数据创建预期的Pandas时间数组
    expected = DatetimeArray._from_sequence(data, dtype=f"M8[{pa_unit}, UTC]").astype(
        dtype, copy=False
    )
    # 断言结果与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 使用dtype对象的__from_arrow__方法从pyarrow的分块数组转换为Pandas的扩展时间数组
    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    # 再次断言结果与预期相等
    tm.assert_extension_array_equal(result, expected)


# 使用pytest的@parametrize装饰器定义另一组测试参数
@pytest.mark.parametrize(
    ("unit", "tz"),
    [
        ("s", "UTC"),
        ("ms", "Europe/Berlin"),
        ("us", "US/Eastern"),
        ("ns", "Asia/Kolkata"),
        ("ns", "UTC"),
    ],
)
# 定义测试函数，测试从空数组创建时间的功能
def test_from_arrow_from_empty(unit, tz):
    # 导入pytest模块，并跳过如果pyarrow未安装的情况
    pa = pytest.importorskip("pyarrow")

    # 创建空的pyarrow数组
    data = []
    arr = pa.array(data)
    # 创建DatetimeTZDtype对象，用于Pandas的时间类型
    dtype = DatetimeTZDtype(unit=unit, tz=tz)

    # 使用dtype对象的__from_arrow__方法从pyarrow数组转换为Pandas的扩展时间数组
    result = dtype.__from_arrow__(arr)
    # 根据原始数据创建预期的Pandas时间数组，并进行时区本地化
    expected = DatetimeArray._from_sequence(np.array(data, dtype=f"datetime64[{unit}]"))
    expected = expected.tz_localize(tz=tz)
    # 断言结果与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 使用dtype对象的__from_arrow__方法从pyarrow的分块数组转换为Pandas的扩展时间数组
    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    # 再次断言结果与预期相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数，测试从整数数组创建时间的功能
def test_from_arrow_from_integers():
    # 导入pytest模块，并跳过如果pyarrow未安装的情况
    pa = pytest.importorskip("pyarrow")

    # 创建包含整数的pyarrow数组
    data = [0, 123456789, None, 2**63 - 1, iNaT, -123456789]
    arr = pa.array(data)
    # 创建DatetimeTZDtype对象，用于Pandas的时间类型
    dtype = DatetimeTZDtype(unit="ns", tz="UTC")

    # 使用dtype对象的__from_arrow__方法从pyarrow数组转换为Pandas的扩展时间数组
    result = dtype.__from_arrow__(arr)
    # 根据原始数据创建预期的Pandas时间数组，并进行时区本地化
    expected = DatetimeArray._from_sequence(np.array(data, dtype="datetime64[ns]"))
    expected = expected.tz_localize("UTC")
    # 断言结果与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 使用dtype对象的__from_arrow__方法从pyarrow的分块数组转换为Pandas的扩展时间数组
    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    # 再次断言结果与预期相等
    tm.assert_extension_array_equal(result, expected)
```