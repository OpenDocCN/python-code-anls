# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_find_common_type.py`

```
# 导入所需的库
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

# 从pandas库中导入相关模块和类
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas import (
    Categorical,
    Index,
)

# 使用pytest的装饰器@parametrize标记，定义测试参数化
@pytest.mark.parametrize(
    "source_dtypes,expected_common_dtype",
    [
        # 各种基本类型的组合
        ((np.int64,), np.int64),
        ((np.uint64,), np.uint64),
        ((np.float32,), np.float32),
        ((object,), object),
        # 整数类型之间的转换
        ((np.int16, np.int64), np.int64),
        ((np.int32, np.uint32), np.int64),
        ((np.uint16, np.uint64), np.uint64),
        # 浮点数类型之间的转换
        ((np.float16, np.float32), np.float32),
        ((np.float16, np.int16), np.float32),
        ((np.float32, np.int16), np.float32),
        ((np.uint64, np.int64), np.float64),
        ((np.int16, np.float64), np.float64),
        ((np.float16, np.int64), np.float64),
        # 其他类型之间的转换
        ((np.complex128, np.int32), np.complex128),
        ((object, np.float32), object),
        ((object, np.int16), object),
        # 布尔型和整数之间的转换
        ((np.dtype("bool"), np.int64), object),
        ((np.dtype("bool"), np.int32), object),
        ((np.dtype("bool"), np.int16), object),
        ((np.dtype("bool"), np.int8), object),
        ((np.dtype("bool"), np.uint64), object),
        ((np.dtype("bool"), np.uint32), object),
        ((np.dtype("bool"), np.uint16), object),
        ((np.dtype("bool"), np.uint8), object),
        # 布尔型和浮点数之间的转换
        ((np.dtype("bool"), np.float64), object),
        ((np.dtype("bool"), np.float32), object),
        # 日期时间和时间间隔类型的处理
        ((np.dtype("datetime64[ns]"), np.dtype("datetime64[ns]")), np.dtype("datetime64[ns]")),
        ((np.dtype("timedelta64[ns]"), np.dtype("timedelta64[ns]")), np.dtype("timedelta64[ns]")),
        ((np.dtype("datetime64[ns]"), np.dtype("datetime64[ms]")), np.dtype("datetime64[ns]")),
        ((np.dtype("timedelta64[ms]"), np.dtype("timedelta64[ns]")), np.dtype("timedelta64[ns]")),
        ((np.dtype("datetime64[ns]"), np.dtype("timedelta64[ns]")), object),
        ((np.dtype("datetime64[ns]"), np.int64), object),
    ],
)
# 定义测试函数test_numpy_dtypes，测试find_common_type函数是否返回正确的结果
def test_numpy_dtypes(source_dtypes, expected_common_dtype):
    # 将source_dtypes列表中的每个元素转换为pandas数据类型
    source_dtypes = [pandas_dtype(x) for x in source_dtypes]
    # 断言调用find_common_type函数后的返回结果是否与expected_common_dtype相等
    assert find_common_type(source_dtypes) == expected_common_dtype


# 定义测试函数test_raises_empty_input，测试find_common_type函数对空输入的处理是否抛出异常
def test_raises_empty_input():
    # 使用pytest的上下文管理器，断言调用find_common_type函数时传入空列表是否抛出值错误异常，并匹配"no types given"字符串
    with pytest.raises(ValueError, match="no types given"):
        find_common_type([])


# 使用pytest的装饰器@parametrize标记，定义测试参数化
@pytest.mark.parametrize(
    "dtypes,exp_type",
    [
        # 测试CategoricalDtype类型的转换
        ([CategoricalDtype()], "category"),
        ([object, CategoricalDtype()], object),
        ([CategoricalDtype(), CategoricalDtype()], "category"),
    ],
)
# 定义测试函数test_categorical_dtype，测试find_common_type函数对CategoricalDtype类型的处理是否正确
def test_categorical_dtype(dtypes, exp_type):
    # 断言调用find_common_type函数后的返回结果是否与exp_type相等
    assert find_common_type(dtypes) == exp_type
# 定义测试函数，用于验证 DatetimeTZDtype 的匹配情况
def test_datetimetz_dtype_match():
    # 创建一个 DatetimeTZDtype 对象，指定单位为纳秒，时区为美国东部
    dtype = DatetimeTZDtype(unit="ns", tz="US/Eastern")
    # 断言调用 find_common_type 函数，传入两个相同的 DatetimeTZDtype 对象，返回值应为字符串 "datetime64[ns, US/Eastern]"
    assert find_common_type([dtype, dtype]) == "datetime64[ns, US/Eastern]"


# 使用 pytest 的 parametrize 装饰器，定义测试函数，用于验证 DatetimeTZDtype 的不匹配情况
@pytest.mark.parametrize(
    "dtype2",
    [
        DatetimeTZDtype(unit="ns", tz="Asia/Tokyo"),
        np.dtype("datetime64[ns]"),
        object,
        np.int64,
    ],
)
def test_datetimetz_dtype_mismatch(dtype2):
    # 创建一个 DatetimeTZDtype 对象，单位为纳秒，时区为美国东部
    dtype = DatetimeTZDtype(unit="ns", tz="US/Eastern")
    # 断言调用 find_common_type 函数，传入两个不同的 DatetimeTZDtype 对象，预期返回值为 object
    assert find_common_type([dtype, dtype2]) == object
    # 断言调用 find_common_type 函数，传入两个不同顺序的 DatetimeTZDtype 对象，预期返回值为 object
    assert find_common_type([dtype2, dtype]) == object


# 定义测试函数，用于验证 PeriodDtype 的匹配情况
def test_period_dtype_match():
    # 创建一个 PeriodDtype 对象，频率为每天
    dtype = PeriodDtype(freq="D")
    # 断言调用 find_common_type 函数，传入两个相同的 PeriodDtype 对象，返回值应为字符串 "period[D]"
    assert find_common_type([dtype, dtype]) == "period[D]"


# 使用 pytest 的 parametrize 装饰器，定义测试函数，用于验证 PeriodDtype 的不匹配情况
@pytest.mark.parametrize(
    "dtype2",
    [
        DatetimeTZDtype(unit="ns", tz="Asia/Tokyo"),
        PeriodDtype(freq="2D"),
        PeriodDtype(freq="h"),
        np.dtype("datetime64[ns]"),
        object,
        np.int64,
    ],
)
def test_period_dtype_mismatch(dtype2):
    # 创建一个 PeriodDtype 对象，频率为每天
    dtype = PeriodDtype(freq="D")
    # 断言调用 find_common_type 函数，传入一个 PeriodDtype 对象和另一种 dtype2，预期返回值为 object
    assert find_common_type([dtype, dtype2]) == object
    # 断言调用 find_common_type 函数，传入一个 dtype2 和 PeriodDtype 对象，预期返回值为 object
    assert find_common_type([dtype2, dtype]) == object


# 定义一个包含多种 IntervalDtype 类型对象的列表
interval_dtypes = [
    IntervalDtype(np.int64, "right"),
    IntervalDtype(np.float64, "right"),
    IntervalDtype(np.uint64, "right"),
    IntervalDtype(DatetimeTZDtype(unit="ns", tz="US/Eastern"), "right"),
    IntervalDtype("M8[ns]", "right"),
    IntervalDtype("m8[ns]", "right"),
]


# 使用 pytest 的 parametrize 装饰器，为两个参数 left 和 right 分别传入 interval_dtypes 列表中的元素
@pytest.mark.parametrize("left", interval_dtypes)
@pytest.mark.parametrize("right", interval_dtypes)
def test_interval_dtype(left, right):
    # 调用 find_common_type 函数，传入两个 IntervalDtype 对象 left 和 right
    result = find_common_type([left, right])

    # 如果 left 和 right 是同一个对象
    if left is right:
        # 断言返回值应为 left
        assert result is left

    # 如果 left 的 subtype 的种类为整数、无符号整数或浮点数
    elif left.subtype.kind in ["i", "u", "f"]:
        # 如果 right 的 subtype 也为整数、无符号整数或浮点数
        if right.subtype.kind in ["i", "u", "f"]:
            # 断言返回值应为具有共同数值 subtype 的 IntervalDtype 对象
            expected = IntervalDtype(np.float64, "right")
            assert result == expected
        else:
            # 断言返回值应为 object
            assert result == object

    else:
        # 断言返回值应为 object
        assert result == object


# 使用 pytest 的 parametrize 装饰器，传入 interval_dtypes 列表中的每个元素作为 dtype 参数
@pytest.mark.parametrize("dtype", interval_dtypes)
def test_interval_dtype_with_categorical(dtype):
    # 创建一个 Index 对象，数据为空列表，dtype 为 interval 类型中的一个
    obj = Index([], dtype=dtype)
    # 创建一个 Categorical 对象，数据为空列表，categories 参数为 obj
    cat = Categorical([], categories=obj)

    # 调用 find_common_type 函数，传入一个 interval 类型的 dtype 和 Categorical 对象的 dtype
    result = find_common_type([dtype, cat.dtype])
    # 断言返回值应为 dtype
    assert result == dtype
```