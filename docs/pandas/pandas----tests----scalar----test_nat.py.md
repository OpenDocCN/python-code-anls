# `D:\src\scipysrc\pandas\pandas\tests\scalar\test_nat.py`

```
# 导入 datetime 和 timedelta 类
from datetime import (
    datetime,
    timedelta,
)
# 导入 operator 模块
import operator
# 导入 zoneinfo 模块
import zoneinfo

# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 pandas 库中的各种类和函数
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
    DatetimeIndex,
    DatetimeTZDtype,
    Index,
    NaT,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    isna,
    offsets,
)
# 导入 pandas 测试模块并使用别名 tm
import pandas._testing as tm
# 导入 pandas 核心模块中的 roperator
from pandas.core import roperator
# 导入 pandas 核心数组相关类
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)


class TestNaTFormatting:
    # 测试 NaT 类的 repr 方法
    def test_repr(self):
        assert repr(NaT) == "NaT"

    # 测试 NaT 类的 str 方法
    def test_str(self):
        assert str(NaT) == "NaT"

    # 测试 NaT 类的 isoformat 方法
    def test_isoformat(self):
        assert NaT.isoformat() == "NaT"


# 参数化测试函数，测试不同的自然缺失时间值和对应的数组类型
@pytest.mark.parametrize(
    "nat,idx",
    [
        (Timestamp("NaT"), DatetimeArray),
        (Timedelta("NaT"), TimedeltaArray),
        (Period("NaT", freq="M"), PeriodArray),
    ],
)
def test_nat_fields(nat, idx):
    # 遍历字段操作集合
    for field in idx._field_ops:
        # 对于 'weekday' 字段，NaT/Timestamp 兼容性处理，直接跳过
        if field == "weekday":
            continue

        # 获取 NaT 对象的字段值并断言为 NaN
        result = getattr(NaT, field)
        assert np.isnan(result)

        # 获取给定自然缺失时间对象的字段值并断言为 NaN
        result = getattr(nat, field)
        assert np.isnan(result)

    # 遍历布尔操作集合
    for field in idx._bool_ops:
        # 获取 NaT 对象的布尔字段值并断言为 False
        result = getattr(NaT, field)
        assert result is False

        # 获取给定自然缺失时间对象的布尔字段值并断言为 False
        result = getattr(nat, field)
        assert result is False


# 测试自然缺失时间对象的向量字段访问
def test_nat_vector_field_access():
    # 创建日期时间索引对象 idx
    idx = DatetimeIndex(["1/1/2000", None, None, "1/4/2000"])

    # 遍历字段操作集合
    for field in DatetimeArray._field_ops:
        # 对于 'weekday' 字段，DatetimeArray 中处理方式不同，直接跳过
        if field == "weekday":
            continue

        # 获取日期时间索引对象 idx 的字段值
        result = getattr(idx, field)
        # 创建预期的索引对象 expected
        expected = Index([getattr(x, field) for x in idx])
        # 使用测试模块中的方法断言结果与预期相等
        tm.assert_index_equal(result, expected)

    # 创建日期时间序列对象 ser
    ser = Series(idx)

    # 遍历字段操作集合
    for field in DatetimeArray._field_ops:
        # 对于 'weekday' 字段，DatetimeArray 中处理方式不同，直接跳过
        if field == "weekday":
            continue

        # 获取日期时间序列对象 ser 的字段值
        result = getattr(ser.dt, field)
        # 创建预期的序列对象 expected
        expected = [getattr(x, field) for x in idx]
        # 使用测试模块中的方法断言结果序列与预期序列相等
        tm.assert_series_equal(result, Series(expected))

    # 遍历布尔操作集合
    for field in DatetimeArray._bool_ops:
        # 获取日期时间序列对象 ser 的布尔字段值
        result = getattr(ser.dt, field)
        # 创建预期的布尔序列对象 expected
        expected = [getattr(x, field) for x in idx]
        # 使用测试模块中的方法断言结果序列与预期序列相等
        tm.assert_series_equal(result, Series(expected))


# 参数化测试函数，测试自然缺失时间对象的身份测试
@pytest.mark.parametrize("klass", [Timestamp, Timedelta, Period])
@pytest.mark.parametrize(
    "value", [None, np.nan, iNaT, float("nan"), NaT, "NaT", "nat", "", "NAT"]
)
def test_identity(klass, value):
    # 断言传入值经过类构造后应为 NaT 对象
    assert klass(value) is NaT


# 参数化测试函数，测试自然缺失时间对象的舍入操作
@pytest.mark.parametrize("klass", [Timestamp, Timedelta])
@pytest.mark.parametrize("method", ["round", "floor", "ceil"])
@pytest.mark.parametrize("freq", ["s", "5s", "min", "5min", "h", "5h"])
def test_round_nat(klass, method, freq):
    # 见 GitHub issue 14940，测试舍入操作
    pass


这段代码的注释按照要求逐行对每个代码语句进行了解释，确保每行代码的作用清晰明了。
    # 创建一个名为 `ts` 的对象实例，类型为 "nat"
    ts = klass("nat")
    
    # 使用内置函数 `getattr` 获取 `ts` 对象中名为 `method` 的方法或属性，并将其赋值给 `round_method` 变量
    round_method = getattr(ts, method)
    
    # 使用断言语句检查调用 `round_method` 方法（或属性）时传入 `freq` 参数返回的结果是否为 `ts` 对象本身，如果不是则抛出 AssertionError
    assert round_method(freq) is ts
@pytest.mark.parametrize(
    "method",
    [
        "astimezone",
        "combine",
        "ctime",
        "dst",
        "fromordinal",
        "fromtimestamp",
        "fromisocalendar",
        "isocalendar",
        "strftime",
        "strptime",
        "time",
        "timestamp",
        "timetuple",
        "timetz",
        "toordinal",
        "tzname",
        "utcfromtimestamp",
        "utcnow",
        "utcoffset",
        "utctimetuple",
    ],
)
def test_nat_methods_raise(method):
    # 创建错误信息，说明 NaTType 不支持特定的方法调用
    msg = f"NaTType does not support {method}"

    # 使用 pytest.raises 检查是否会抛出 ValueError，并匹配预期的错误信息
    with pytest.raises(ValueError, match=msg):
        getattr(NaT, method)()


@pytest.mark.parametrize("method", ["weekday", "isoweekday"])
def test_nat_methods_nan(method):
    # 创建断言，检查 NaTType 对象调用特定方法后是否返回 NaN
    assert np.isnan(getattr(NaT, method)())


@pytest.mark.parametrize(
    "method", ["date", "now", "replace", "today", "tz_convert", "tz_localize"]
)
def test_nat_methods_nat(method):
    # 创建断言，检查 NaTType 对象调用特定方法后是否返回 NaT
    assert getattr(NaT, method)() is NaT


@pytest.mark.parametrize(
    "get_nat", [lambda x: NaT, lambda x: Timedelta(x), lambda x: Timestamp(x)]
)
def test_nat_iso_format(get_nat):
    # 创建断言，检查 NaTType 对象调用 isoformat 方法后是否返回 "NaT"
    assert get_nat("NaT").isoformat() == "NaT"
    assert get_nat("NaT").isoformat(timespec="nanoseconds") == "NaT"


@pytest.mark.parametrize(
    "klass,expected",
    [
        (Timestamp, ["normalize", "to_julian_date", "to_period", "unit"]),
        (
            Timedelta,
            [
                "components",
                "resolution_string",
                "to_pytimedelta",
                "to_timedelta64",
                "unit",
                "view",
            ],
        ),
    ],
)
def test_missing_public_nat_methods(klass, expected):
    # 创建断言，检查 NaTType 缺少哪些 Timestamp 或 Timedelta 的公共方法
    # 忽略任何缺少的私有方法
    nat_names = dir(NaT)
    klass_names = dir(klass)

    missing = [x for x in klass_names if x not in nat_names and not x.startswith("_")]
    missing.sort()

    assert missing == expected


def _get_overlap_public_nat_methods(klass, as_tuple=False):
    """
    Get overlapping public methods between NaT and another class.

    Parameters
    ----------
    klass : type
        The class to compare with NaT
    as_tuple : bool, default False
        Whether to return a list of tuples of the form (klass, method).

    Returns
    -------
    overlap : list
    """
    nat_names = dir(NaT)
    klass_names = dir(klass)

    # 寻找 NaTType 和另一个类之间的公共方法
    overlap = [
        x
        for x in nat_names
        if x in klass_names and not x.startswith("_") and callable(getattr(klass, x))
    ]

    # 如果 klass 是 Timedelta，则 Timestamp 的方法优先于 Timedelta
    if klass is Timedelta:
        ts_names = dir(Timestamp)
        overlap = [x for x in overlap if x not in ts_names]
    # 如果参数 as_tuple 为真，则将 overlap 列表中的每个方法作为元组 (klass, method) 形式存储在 overlap 中
    if as_tuple:
        overlap = [(klass, method) for method in overlap]

    # 对 overlap 列表进行排序，排序依据为列表中的元素，默认是按第一个元素排序，即 klass
    overlap.sort()

    # 返回排序后的 overlap 列表，该列表包含按 klass 排序的 (klass, method) 元组
    return overlap
@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，为测试用例提供不同的输入参数
    "klass,expected",  # 参数化的参数：类名和预期输出列表
    [  # 参数化的输入数据列表开始
        (
            Timestamp,  # 第一个参数化的类为 Timestamp
            [  # 第一个参数化类的预期输出方法列表开始
                "as_unit",  # 方法名：as_unit
                "astimezone",  # 方法名：astimezone
                "ceil",  # 方法名：ceil
                "combine",  # 方法名：combine
                "ctime",  # 方法名：ctime
                "date",  # 方法名：date
                "day_name",  # 方法名：day_name
                "dst",  # 方法名：dst
                "floor",  # 方法名：floor
                "fromisocalendar",  # 方法名：fromisocalendar
                "fromisoformat",  # 方法名：fromisoformat
                "fromordinal",  # 方法名：fromordinal
                "fromtimestamp",  # 方法名：fromtimestamp
                "isocalendar",  # 方法名：isocalendar
                "isoformat",  # 方法名：isoformat
                "isoweekday",  # 方法名：isoweekday
                "month_name",  # 方法名：month_name
                "now",  # 方法名：now
                "replace",  # 方法名：replace
                "round",  # 方法名：round
                "strftime",  # 方法名：strftime
                "strptime",  # 方法名：strptime
                "time",  # 方法名：time
                "timestamp",  # 方法名：timestamp
                "timetuple",  # 方法名：timetuple
                "timetz",  # 方法名：timetz
                "to_datetime64",  # 方法名：to_datetime64
                "to_numpy",  # 方法名：to_numpy
                "to_pydatetime",  # 方法名：to_pydatetime
                "today",  # 方法名：today
                "toordinal",  # 方法名：toordinal
                "tz_convert",  # 方法名：tz_convert
                "tz_localize",  # 方法名：tz_localize
                "tzname",  # 方法名：tzname
                "utcfromtimestamp",  # 方法名：utcfromtimestamp
                "utcnow",  # 方法名：utcnow
                "utcoffset",  # 方法名：utcoffset
                "utctimetuple",  # 方法名：utctimetuple
                "weekday",  # 方法名：weekday
            ],  # 第一个参数化类的预期输出方法列表结束
        ),  # 第一个参数化的输入元组结束
        (
            Timedelta,  # 第二个参数化的类为 Timedelta
            ["total_seconds"],  # 第二个参数化类的预期输出方法列表为 total_seconds
        ),  # 第二个参数化的输入元组结束
    ],  # 参数化的输入数据列表结束
)
def test_overlap_public_nat_methods(klass, expected):
    # see gh-17327
    #
    # NaT should have *most* of the Timestamp and Timedelta methods.
    # In case when Timestamp, Timedelta, and NaT are overlap, the overlap
    # is considered to be with Timestamp and NaT, not Timedelta.
    assert _get_overlap_public_nat_methods(klass) == expected


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，为测试用例提供不同的输入参数
    "compare",  # 参数化的参数：比较结果
    (  # 参数化的输入数据列表开始
        _get_overlap_public_nat_methods(Timestamp, True)  # 调用函数获取 Timestamp 类的公共方法重叠结果
        + _get_overlap_public_nat_methods(Timedelta, True)  # 调用函数获取 Timedelta 类的公共方法重叠结果
    ),  # 参数化的输入数据列表结束
    ids=lambda x: f"{x[0].__name__}.{x[1]}",  # 使用 Lambda 函数生成测试用例的 ID
)
def test_nat_doc_strings(compare):
    # see gh-17327
    #
    # The docstrings for overlapping methods should match.
    klass, method = compare  # 解包比较结果元组为类和方法
    klass_doc = getattr(klass, method).__doc__  # 获取方法的文档字符串

    if klass == Timestamp and method == "isoformat":
        pytest.skip(
            "Ignore differences with Timestamp.isoformat() as they're intentional"
        )  # 如果是 Timestamp 的 isoformat 方法，跳过测试

    if method == "to_numpy":
        # GH#44460 can return either dt64 or td64 depending on dtype,
        #  different docstring is intentional
        pytest.skip(f"different docstring for {method} is intentional")  # 如果是 to_numpy 方法，跳过测试并注明文档字符串差异是有意的


_ops = {  # 定义操作字典
    "left_plus_right": lambda a, b: a + b,  # 键：left_plus_right，值：lambda 函数执行 a + b
    "right_plus_left": lambda a, b: b + a,  # 键：right_plus_left，值：lambda 函数执行 b + a
    "left_minus_right": lambda a, b: a - b,  # 键：left_minus_right，值：lambda 函数执行 a - b
    "right_minus_left": lambda a, b: b - a,  # 键：right_minus_left，值：lambda 函数执行 b - a
    "left_times_right": lambda a, b: a * b,  # 键：left_times_right，值：lambda 函数执行 a * b
    "right_times_left": lambda a, b: b * a,  # 键：right_times_left，值：lambda 函数执行 b * a
    "left_div_right": lambda a, b: a / b,  # 键：left_div_right，值：lambda 函数执行 a / b
    "right_div_left": lambda a, b: b / a,  # 键：right_div_left，值：lambda 函数执行 b / a
}


@pytest.mark.parametrize("op_name", list(_ops.keys()))  # 使用 pytest 的参数化装饰器，为测试用例提供不同的输入操作名
@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，为测试用例提供不同的输入参数
    "value,val_type",

    [
        (_ops[op_name], "function")  # 参数化的输入数据列表：操作函数和字符串 "function"
    ],  # 参数化的输入数据列表结束
)
    [
        (2, "scalar"),  # 整数类型的标量值
        (1.5, "floating"),  # 浮点数类型的标量值
        (np.nan, "floating"),  # 浮点数类型的标量值，表示 NaN（非数字）
        ("foo", "str"),  # 字符串类型的值
        (timedelta(3600), "timedelta"),  # 时间增量对象，表示一小时
        (Timedelta("5s"), "timedelta"),  # 时间增量对象，表示5秒
        (datetime(2014, 1, 1), "timestamp"),  # Python datetime 对象，表示日期时间
        (Timestamp("2014-01-01"), "timestamp"),  # Pandas Timestamp 对象，表示日期时间
        (Timestamp("2014-01-01", tz="UTC"), "timestamp"),  # 带有时区信息的 Pandas Timestamp 对象
        (Timestamp("2014-01-01", tz="US/Eastern"), "timestamp"),  # 带有不同时区信息的 Pandas Timestamp 对象
        (datetime(2014, 1, 1).astimezone(zoneinfo.ZoneInfo("Asia/Tokyo")), "timestamp"),  # 带有特定时区信息的 datetime 对象
    ],
# 测试自然数运算的标量操作，针对特定操作名、值和值类型进行测试
def test_nat_arithmetic_scalar(op_name, value, val_type):
    # 定义不支持的操作集合，根据值类型进行分类
    invalid_ops = {
        "scalar": {"right_div_left"},
        "floating": {
            "right_div_left",
            "left_minus_right",
            "right_minus_left",
            "left_plus_right",
            "right_plus_left",
        },
        "str": set(_ops.keys()),  # 使用所有操作的键集合
        "timedelta": {"left_times_right", "right_times_left"},
        "timestamp": {
            "left_times_right",
            "right_times_left",
            "left_div_right",
            "right_div_left",
        },
    }

    # 获取操作函数对象
    op = _ops[op_name]

    # 如果操作名在对应值类型的不支持操作集合中
    if op_name in invalid_ops.get(val_type, set()):
        # 对于时间增量类型且操作名包含 "times" 并且值类型是时间增量的情况
        if (
            val_type == "timedelta"
            and "times" in op_name
            and isinstance(value, Timedelta)
        ):
            typs = "(Timedelta|NaTType)"
            # 构造特定的错误信息字符串
            msg = rf"unsupported operand type\(s\) for \*: '{typs}' and '{typs}'"
        elif val_type == "str":
            # 对字符串类型进行未具体化检查，因为消息来源于字符串且根据方法而变化
            msg = "|".join(
                [
                    "can only concatenate str",
                    "unsupported operand type",
                    "can't multiply sequence",
                    "Can't convert 'NaTType'",
                    "must be str, not NaTType",
                ]
            )
        else:
            msg = "unsupported operand type"

        # 使用 pytest 断言预期会抛出 TypeError 异常，并匹配预设消息
        with pytest.raises(TypeError, match=msg):
            op(NaT, value)
    else:
        # 如果值类型是时间增量且操作名包含 "div" 的情况下
        if val_type == "timedelta" and "div" in op_name:
            expected = np.nan  # 预期结果为 NaN
        else:
            expected = NaT  # 否则预期结果为 NaT

        # 使用断言检查操作的结果与预期结果是否一致
        assert op(NaT, value) is expected


# 参数化测试，测试时间增量与值之间的右向地板除法操作
@pytest.mark.parametrize(
    "val,expected", [(np.nan, NaT), (NaT, np.nan), (np.timedelta64("NaT"), np.nan)]
)
def test_nat_rfloordiv_timedelta(val, expected):
    # 详见 gh-#18846
    #
    # 另请参阅 test_timedelta.TestTimedeltaArithmetic.test_floordiv
    # 创建一个时间增量对象，3小时4分钟
    td = Timedelta(hours=3, minutes=4)
    # 使用断言检查时间增量对象与参数化提供的值进行右向地板除法操作的结果是否等于预期结果
    assert td // val is expected


# 参数化测试，测试自然数运算与索引操作的各种组合
@pytest.mark.parametrize(
    "op_name",
    ["left_plus_right", "right_plus_left", "left_minus_right", "right_minus_left"],
)
@pytest.mark.parametrize(
    "value",
    [
        DatetimeIndex(["2011-01-01", "2011-01-02"], dtype="M8[ns]", name="x"),
        DatetimeIndex(
            ["2011-01-01", "2011-01-02"], dtype="M8[ns, US/Eastern]", name="x"
        ),
        DatetimeArray._from_sequence(["2011-01-01", "2011-01-02"], dtype="M8[ns]"),
        DatetimeArray._from_sequence(
            ["2011-01-01", "2011-01-02"], dtype=DatetimeTZDtype(tz="US/Pacific")
        ),
        TimedeltaIndex(["1 day", "2 day"], name="x"),
    ],
)
def test_nat_arithmetic_index(op_name, value):
    # 详见 gh-11718
    exp_name = "x"
    exp_data = [NaT] * 2

    # 如果值的数据类型的种类是 'M' 并且操作名包含 "plus" 的情况
    if value.dtype.kind == "M" and "plus" in op_name:
        # 创建一个预期的日期时间索引对象，其数据与提供的值相同，并保持时区与值的时区一致
        expected = DatetimeIndex(exp_data, tz=value.tz, name=exp_name)
    else:
        # 否则创建一个预期的时间增量索引对象，其数据与提供的值相同
        expected = TimedeltaIndex(exp_data, name=exp_name)
    # 将预期值转换为与值单位相同的单位
    expected = expected.as_unit(value.unit)

    # 如果值不是索引对象，则将预期值转换为其数组形式
    if not isinstance(value, Index):
        expected = expected.array

    # 从操作名映射中获取操作函数
    op = _ops[op_name]
    
    # 使用 NaT 和给定值执行操作，并获取结果
    result = op(NaT, value)
    
    # 使用测试模块中的函数验证结果与预期值是否相等
    tm.assert_equal(result, expected)
@pytest.mark.parametrize(
    "op_name",
    ["left_plus_right", "right_plus_left", "left_minus_right", "right_minus_left"],
)
@pytest.mark.parametrize("box", [TimedeltaIndex, Series, TimedeltaArray._from_sequence])
# 定义测试函数，用于测试NaN时间操作的向量化计算
def test_nat_arithmetic_td64_vector(op_name, box):
    # 见问题报告 gh-19124
    # 创建一个包含两个时间差字符串的时间差索引/序列/数组
    vec = box(["1 day", "2 day"], dtype="timedelta64[ns]")
    # 创建一个包含两个NaT的相同类型对象
    box_nat = box([NaT, NaT], dtype="timedelta64[ns]")
    # 断言NaN时间与特定操作的结果与预期相等
    tm.assert_equal(_ops[op_name](vec, NaT), box_nat)


@pytest.mark.parametrize(
    "dtype,op,out_dtype",
    [
        ("datetime64[ns]", operator.add, "datetime64[ns]"),
        ("datetime64[ns]", roperator.radd, "datetime64[ns]"),
        ("datetime64[ns]", operator.sub, "timedelta64[ns]"),
        ("datetime64[ns]", roperator.rsub, "timedelta64[ns]"),
        ("timedelta64[ns]", operator.add, "datetime64[ns]"),
        ("timedelta64[ns]", roperator.radd, "datetime64[ns]"),
        ("timedelta64[ns]", operator.sub, "datetime64[ns]"),
        ("timedelta64[ns]", roperator.rsub, "timedelta64[ns]"),
    ],
)
# 定义测试函数，用于测试NaN时间与数组的算术运算
def test_nat_arithmetic_ndarray(dtype, op, out_dtype):
    # 创建一个特定类型的数组，其中元素为0到9
    other = np.arange(10).astype(dtype)
    # 执行NaN时间与数组元素的特定操作
    result = op(NaT, other)

    # 创建一个预期的结果数组，其元素都为NaT
    expected = np.empty(other.shape, dtype=out_dtype)
    expected.fill("NaT")
    # 断言结果与预期数组相等
    tm.assert_numpy_array_equal(result, expected)


def test_nat_pinned_docstrings():
    # 见问题报告 gh-17327
    # 断言NaT对象的ctime方法文档字符串与Timestamp对象的相同
    assert NaT.ctime.__doc__ == Timestamp.ctime.__doc__


def test_to_numpy_alias():
    # GH 24653: 为标量提供.to_numpy()的别名
    # 断言NaT对象通过.to_numpy()返回的结果是NaN
    expected = NaT.to_datetime64()
    result = NaT.to_numpy()

    assert isna(expected) and isna(result)

    # GH#44460
    # 测试NaT对象在指定不同时间单位的情况下通过.to_numpy()返回的结果类型与dtype
    result = NaT.to_numpy("M8[s]")
    assert isinstance(result, np.datetime64)
    assert result.dtype == "M8[s]"

    result = NaT.to_numpy("m8[ns]")
    assert isinstance(result, np.timedelta64)
    assert result.dtype == "m8[ns]"

    result = NaT.to_numpy("m8[s]")
    assert isinstance(result, np.timedelta64)
    assert result.dtype == "m8[s]"

    # 使用pytest.raises检查NaT.to_numpy()传入非法dtype时是否引发异常
    with pytest.raises(ValueError, match="NaT.to_numpy dtype must be a "):
        NaT.to_numpy(np.int64)


@pytest.mark.parametrize(
    "other",
    [
        Timedelta(0),
        Timedelta(0).to_pytimedelta(),
        pytest.param(
            Timedelta(0).to_timedelta64(),
            marks=pytest.mark.xfail(
                not np_version_gte1p24p3,
                reason="td64 doesn't return NotImplemented, see numpy#17017",
                # 当此xfail修复后，可以移除对test_nat_comparisons_numpy的xfail标记
            ),
        ),
        Timestamp(0),
        Timestamp(0).to_pydatetime(),
        pytest.param(
            Timestamp(0).to_datetime64(),
            marks=pytest.mark.xfail(
                not np_version_gte1p24p3,
                reason="dt64 doesn't return NotImplemented, see numpy#17017",
            ),
        ),
        Timestamp(0).tz_localize("UTC"),
        NaT,
    ],
)
# 定义测试函数，用于测试NaN时间的比较操作
def test_nat_comparisons(compare_operators_no_eq_ne, other):
    # GH 26039
    opname = compare_operators_no_eq_ne
    # 断言检查指定操作名在 NaT 对象上执行给定参数时的返回值是否为 False
    assert getattr(NaT, opname)(other) is False
    
    # 获取指定操作名（去除开头和结尾的下划线）对应的操作函数
    op = getattr(operator, opname.strip("_"))
    # 断言检查使用获取到的操作函数对 NaT 对象和给定参数执行操作时的返回值是否为 False
    assert op(NaT, other) is False
    
    # 断言检查使用获取到的操作函数对给定参数和 NaT 对象执行操作时的返回值是否为 False
    assert op(other, NaT) is False
@pytest.mark.parametrize("other", [np.timedelta64(0, "ns"), np.datetime64("now", "ns")])
# 使用 pytest.mark.parametrize 注解，对函数 test_nat_comparisons_numpy 进行参数化测试，参数为 np.timedelta64 和 np.datetime64 类型的对象
def test_nat_comparisons_numpy(other):
    # 一旦修复了 numpy#17017 和 test_nat_comparisons 中的 xfailed 测试案例通过，可以移除这个测试
    assert not NaT == other
    # 断言 NaT 不等于 other
    assert NaT != other
    # 断言 NaT 不小于 other
    assert not NaT < other
    # 断言 NaT 不大于 other
    assert not NaT > other
    # 断言 NaT 不小于等于 other
    assert not NaT <= other
    # 断言 NaT 不大于等于 other
    assert not NaT >= other


@pytest.mark.parametrize("other_and_type", [("foo", "str"), (2, "int"), (2.0, "float")])
@pytest.mark.parametrize(
    "symbol_and_op",
    [("<=", operator.le), ("<", operator.lt), (">=", operator.ge), (">", operator.gt)],
)
# 使用 pytest.mark.parametrize 注解，对函数 test_nat_comparisons_invalid 进行参数化测试，参数为 other_and_type 和 symbol_and_op
def test_nat_comparisons_invalid(other_and_type, symbol_and_op):
    # GH#35585
    other, other_type = other_and_type
    # 解包 other_and_type 元组，分别赋值给 other 和 other_type
    symbol, op = symbol_and_op
    # 解包 symbol_and_op 元组，分别赋值给 symbol 和 op

    assert not NaT == other
    # 断言 NaT 不等于 other
    assert not other == NaT
    # 断言 other 不等于 NaT

    assert NaT != other
    # 断言 NaT 不等于 other
    assert other != NaT
    # 断言 other 不等于 NaT

    msg = f"'{symbol}' not supported between instances of 'NaTType' and '{other_type}'"
    # 构建错误消息，指示 'NaTType' 和 '{other_type}' 之间不支持 'symbol' 操作
    with pytest.raises(TypeError, match=msg):
        op(NaT, other)
    # 使用 pytest.raises 断言捕获 TypeError 异常，匹配错误消息 msg，执行 op(NaT, other)

    msg = f"'{symbol}' not supported between instances of '{other_type}' and 'NaTType'"
    # 构建错误消息，指示 '{other_type}' 和 'NaTType' 之间不支持 'symbol' 操作
    with pytest.raises(TypeError, match=msg):
        op(other, NaT)
    # 使用 pytest.raises 断言捕获 TypeError 异常，匹配错误消息 msg，执行 op(other, NaT)


@pytest.mark.parametrize(
    "other",
    [
        np.array(["foo"] * 2, dtype=object),
        np.array([2, 3], dtype="int64"),
        np.array([2.0, 3.5], dtype="float64"),
    ],
    ids=["str", "int", "float"],
)
# 使用 pytest.mark.parametrize 注解，对函数 test_nat_comparisons_invalid_ndarray 进行参数化测试，参数为 np.array 对象数组
def test_nat_comparisons_invalid_ndarray(other):
    # GH#40722
    expected = np.array([False, False])
    # 创建预期的结果数组，全为 False
    result = NaT == other
    # 使用 NaT 和 other 进行等于比较
    tm.assert_numpy_array_equal(result, expected)
    # 使用 tm.assert_numpy_array_equal 断言 result 和 expected 数组相等
    result = other == NaT
    # 使用 other 和 NaT 进行等于比较
    tm.assert_numpy_array_equal(result, expected)
    # 使用 tm.assert_numpy_array_equal 断言 result 和 expected 数组相等

    expected = np.array([True, True])
    # 创建预期的结果数组，全为 True
    result = NaT != other
    # 使用 NaT 和 other 进行不等于比较
    tm.assert_numpy_array_equal(result, expected)
    # 使用 tm.assert_numpy_array_equal 断言 result 和 expected 数组相等
    result = other != NaT
    # 使用 other 和 NaT 进行不等于比较
    tm.assert_numpy_array_equal(result, expected)
    # 使用 tm.assert_numpy_array_equal 断言 result 和 expected 数组相等

    for symbol, op in [
        ("<=", operator.le),
        ("<", operator.lt),
        (">=", operator.ge),
        (">", operator.gt),
    ]:
        # 遍历 symbol 和 op 元组列表
        msg = f"'{symbol}' not supported between"
        # 构建错误消息，指示不支持 symbol 操作

        with pytest.raises(TypeError, match=msg):
            op(NaT, other)
        # 使用 pytest.raises 断言捕获 TypeError 异常，匹配错误消息 msg，执行 op(NaT, other)

        if other.dtype == np.dtype("object"):
            # 如果 other 的数据类型是 np.dtype("object")
            # 使用相反的操作符，所以符号会变化
            msg = None
            # 错误消息为空

        with pytest.raises(TypeError, match=msg):
            op(other, NaT)
        # 使用 pytest.raises 断言捕获 TypeError 异常，匹配错误消息 msg，执行 op(other, NaT)


def test_compare_date(fixed_now_ts):
    # GH#39151 comparing NaT with date object is deprecated
    # See also: tests.scalar.timestamps.test_comparisons::test_compare_date

    dt = fixed_now_ts.to_pydatetime().date()
    # 获取固定时间戳的日期部分，转换为 datetime.date 对象

    msg = "Cannot compare NaT with datetime.date object"
    # 设置错误消息，指示无法比较 NaT 和 datetime.date 对象
    # 使用元组列表 [(NaT, dt), (dt, NaT)] 中的每对元组(left, right)进行以下操作：
    for left, right in [(NaT, dt), (dt, NaT)]:
        # 断言左边的值不等于右边的值
        assert not left == right
        # 断言左边的值不等于右边的值（另一种形式的断言）
        assert left != right

        # 使用 pytest 的断言来捕获期望的 TypeError 异常，并匹配给定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            # 尝试比较 left < right，预期会引发 TypeError 异常
            left < right
        with pytest.raises(TypeError, match=msg):
            # 尝试比较 left <= right，预期会引发 TypeError 异常
            left <= right
        with pytest.raises(TypeError, match=msg):
            # 尝试比较 left > right，预期会引发 TypeError 异常
            left > right
        with pytest.raises(TypeError, match=msg):
            # 尝试比较 left >= right，预期会引发 TypeError 异常
            left >= right
# 使用 pytest.mark.parametrize 装饰器，为 test_nat_addsub_tdlike_scalar 函数添加参数化测试
@pytest.mark.parametrize(
    "obj",
    [
        offsets.YearEnd(2),                 # 创建 YearEnd 偏移量对象，参数为 2
        offsets.YearBegin(2),               # 创建 YearBegin 偏移量对象，参数为 2
        offsets.MonthBegin(1),              # 创建 MonthBegin 偏移量对象，参数为 1
        offsets.MonthEnd(2),                # 创建 MonthEnd 偏移量对象，参数为 2
        offsets.MonthEnd(12),               # 创建 MonthEnd 偏移量对象，参数为 12
        offsets.Day(2),                     # 创建 Day 偏移量对象，参数为 2
        offsets.Day(5),                     # 创建 Day 偏移量对象，参数为 5
        offsets.Hour(24),                   # 创建 Hour 偏移量对象，参数为 24
        offsets.Hour(3),                    # 创建 Hour 偏移量对象，参数为 3
        offsets.Minute(),                   # 创建 Minute 偏移量对象，无参数
        np.timedelta64(3, "h"),             # 创建 numpy 时间间隔对象，表示 3 小时
        np.timedelta64(4, "h"),             # 创建 numpy 时间间隔对象，表示 4 小时
        np.timedelta64(3200, "s"),          # 创建 numpy 时间间隔对象，表示 3200 秒
        np.timedelta64(3600, "s"),          # 创建 numpy 时间间隔对象，表示 3600 秒
        np.timedelta64(3600 * 24, "s"),     # 创建 numpy 时间间隔对象，表示 3600 * 24 秒
        np.timedelta64(2, "D"),             # 创建 numpy 时间间隔对象，表示 2 天
        np.timedelta64(365, "D"),           # 创建 numpy 时间间隔对象，表示 365 天
        timedelta(-2),                      # 创建 timedelta 对象，表示 -2 天
        timedelta(365),                     # 创建 timedelta 对象，表示 365 天
        timedelta(minutes=120),             # 创建 timedelta 对象，表示 120 分钟
        timedelta(days=4, minutes=180),     # 创建 timedelta 对象，表示 4 天 180 分钟
        timedelta(hours=23),                # 创建 timedelta 对象，表示 23 小时
        timedelta(hours=23, minutes=30),    # 创建 timedelta 对象，表示 23 小时 30 分钟
        timedelta(hours=48),                # 创建 timedelta 对象，表示 48 小时
    ],
)
# 定义测试函数 test_nat_addsub_tdlike_scalar，用于测试 NaT（Not a Time）对象与时间偏移量的加减操作
def test_nat_addsub_tdlike_scalar(obj):
    # 测试 NaT 加上时间偏移量的结果是否仍为 NaT
    assert NaT + obj is NaT
    # 测试时间偏移量加上 NaT 的结果是否仍为 NaT
    assert obj + NaT is NaT
    # 测试 NaT 减去时间偏移量的结果是否仍为 NaT
    assert NaT - obj is NaT


# 定义测试函数 test_pickle，用于测试 NaT 对象的序列化与反序列化
def test_pickle():
    # GH#4606 表示 GitHub 问题编号 4606，测试 NaT 对象的 pickle 序列化和反序列化
    p = tm.round_trip_pickle(NaT)
    # 断言序列化后的对象 p 是否仍为 NaT
    assert p is NaT
```