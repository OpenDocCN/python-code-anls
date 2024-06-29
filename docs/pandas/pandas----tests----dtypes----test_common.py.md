# `D:\src\scipysrc\pandas\pandas\tests\dtypes\test_common.py`

```
from __future__ import annotations  # 导入未来的 annotations 特性，使得函数的类型提示更加精确

import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas.util._test_decorators as td  # 导入 pandas 内部测试装饰器模块

from pandas.core.dtypes.astype import astype_array  # 导入 pandas 中的数据类型转换函数
import pandas.core.dtypes.common as com  # 导入 pandas 中的常用数据类型函数
from pandas.core.dtypes.dtypes import (  # 导入 pandas 中的各种数据类型
    CategoricalDtype,
    CategoricalDtypeType,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import isna  # 导入 pandas 中的缺失值处理函数

import pandas as pd  # 导入 pandas 库
import pandas._testing as tm  # 导入 pandas 内部测试模块
from pandas.api.types import pandas_dtype  # 导入 pandas 中的数据类型检查函数
from pandas.arrays import SparseArray  # 导入 pandas 中的稀疏数组模块

# EA & Actual Dtypes
def to_ea_dtypes(dtypes):
    """将字符串数据类型列表转换为 EA 数据类型"""
    return [getattr(pd, dt + "Dtype") for dt in dtypes]  # 使用 getattr 获取 pandas 库中对应字符串的数据类型对象

def to_numpy_dtypes(dtypes):
    """将字符串数据类型列表转换为 NumPy 数据类型"""
    return [getattr(np, dt) for dt in dtypes if isinstance(dt, str)]  # 使用 getattr 获取 NumPy 库中对应字符串的数据类型对象

class TestNumpyEADtype:
    # Passing invalid dtype, both as a string or object, must raise TypeError
    # Per issue GH15520
    @pytest.mark.parametrize("box", [pd.Timestamp, "pd.Timestamp", list])
    def test_invalid_dtype_error(self, box):
        with pytest.raises(TypeError, match="not understood"):
            com.pandas_dtype(box)  # 调用 pandas_dtype 函数，测试是否能正确处理无效的数据类型

    @pytest.mark.parametrize(
        "dtype",
        [
            object,
            "float64",
            np.object_,
            np.dtype("object"),
            "O",
            np.float64,
            float,
            np.dtype("float64"),
            "object_",
        ],
    )
    def test_pandas_dtype_valid(self, dtype):
        assert com.pandas_dtype(dtype) == dtype  # 测试 pandas_dtype 函数能否正确处理有效的数据类型

    @pytest.mark.parametrize(
        "dtype", ["M8[ns]", "m8[ns]", "object", "float64", "int64"]
    )
    def test_numpy_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == np.dtype(dtype)  # 测试 pandas_dtype 函数是否能正确处理 NumPy 数据类型

    def test_numpy_string_dtype(self):
        # do not parse freq-like string as period dtype
        assert com.pandas_dtype("U") == np.dtype("U")  # 测试 pandas_dtype 函数是否能正确处理字符串类型 'U'
        assert com.pandas_dtype("S") == np.dtype("S")  # 测试 pandas_dtype 函数是否能正确处理字符串类型 'S'

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[ns, US/Eastern]",
            "datetime64[ns, Asia/Tokyo]",
            "datetime64[ns, UTC]",
            # GH#33885 check that the M8 alias is understood
            "M8[ns, US/Eastern]",
            "M8[ns, Asia/Tokyo]",
            "M8[ns, UTC]",
        ],
    )
    def test_datetimetz_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == DatetimeTZDtype.construct_from_string(dtype)  # 测试 pandas_dtype 函数是否能正确处理时区日期时间类型
        assert com.pandas_dtype(dtype) == dtype  # 测试 pandas_dtype 函数是否能正确处理时区日期时间类型的字符串表示

    def test_categorical_dtype(self):
        assert com.pandas_dtype("category") == CategoricalDtype()  # 测试 pandas_dtype 函数是否能正确处理分类数据类型

    @pytest.mark.parametrize(
        "dtype",
        [
            "period[D]",
            "period[3M]",
            "period[us]",
            "Period[D]",
            "Period[3M]",
            "Period[us]",
        ],
    )
    # 定义一个测试函数，用于验证数据类型是否正确转换为期间数据类型
    def test_period_dtype(self, dtype):
        # 断言传入的数据类型经过 pandas_dtype 转换后不等于其对应的 PeriodDtype 类型对象
        assert com.pandas_dtype(dtype) is not PeriodDtype(dtype)
        # 断言传入的数据类型经过 pandas_dtype 转换后等于其对应的 PeriodDtype 类型对象
        assert com.pandas_dtype(dtype) == PeriodDtype(dtype)
        # 断言传入的数据类型经过 pandas_dtype 转换后等于其自身数据类型
        assert com.pandas_dtype(dtype) == dtype
# 定义一个包含不同数据类型的字典，每个数据类型由字符串键和相应的 Pandas 或 NumPy 数据类型对象值组成
dtypes = {
    "datetime_tz": com.pandas_dtype("datetime64[ns, US/Eastern]"),  # 定义带时区的日期时间数据类型
    "datetime": com.pandas_dtype("datetime64[ns]"),  # 定义日期时间数据类型
    "timedelta": com.pandas_dtype("timedelta64[ns]"),  # 定义时间间隔数据类型
    "period": PeriodDtype("D"),  # 定义周期数据类型，每天为一个周期
    "integer": np.dtype(np.int64),  # 定义整数数据类型
    "float": np.dtype(np.float64),  # 定义浮点数数据类型
    "object": np.dtype(object),  # 定义对象数据类型
    "category": com.pandas_dtype("category"),  # 定义分类数据类型
    "string": pd.StringDtype(),  # 定义字符串数据类型
}


# 使用参数化测试，对每对数据类型进行测试
@pytest.mark.parametrize("name1,dtype1", list(dtypes.items()), ids=lambda x: str(x))
@pytest.mark.parametrize("name2,dtype2", list(dtypes.items()), ids=lambda x: str(x))
def test_dtype_equal(name1, dtype1, name2, dtype2):
    # 检查数据类型是否等于自身，但不等于其它数据类型
    assert com.is_dtype_equal(dtype1, dtype1)
    if name1 != name2:
        assert not com.is_dtype_equal(dtype1, dtype2)


# 使用参数化测试，对每种数据类型进行测试，检查是否能正确处理 pyarrow 字符串类型导入错误
@pytest.mark.parametrize("name,dtype", list(dtypes.items()), ids=lambda x: str(x))
def test_pyarrow_string_import_error(name, dtype):
    # GH-44276：检查是否不等于 pyarrow 字符串数据类型
    assert not com.is_dtype_equal(dtype, "string[pyarrow]")


# 使用参数化测试，对每对数据类型进行测试，检查严格的数据类型相等性
@pytest.mark.parametrize(
    "dtype1,dtype2",
    [
        (np.int8, np.int64),
        (np.int16, np.int64),
        (np.int32, np.int64),
        (np.float32, np.float64),
        (PeriodDtype("D"), PeriodDtype("2D")),  # 周期类型
        (
            com.pandas_dtype("datetime64[ns, US/Eastern]"),
            com.pandas_dtype("datetime64[ns, CET]"),
        ),  # 日期时间类型
        (None, None),  # gh-15941: 不应该引发异常
    ],
)
def test_dtype_equal_strict(dtype1, dtype2):
    # 检查是否不等于严格的数据类型
    assert not com.is_dtype_equal(dtype1, dtype2)


def get_is_dtype_funcs():
    """
    获取 pandas.core.dtypes.common 中以 'is_' 开头且以 'dtype' 结尾的所有函数
    """
    fnames = [f for f in dir(com) if (f.startswith("is_") and f.endswith("dtype"))]
    fnames.remove("is_string_or_object_np_dtype")  # 快速路径需要 np.dtype 对象
    return [getattr(com, fname) for fname in fnames]


# 使用参数化测试，对每个数据类型检查函数进行测试，确保不会引发异常
@pytest.mark.filterwarnings(
    "ignore:is_categorical_dtype is deprecated:DeprecationWarning"
)
@pytest.mark.parametrize("func", get_is_dtype_funcs(), ids=lambda x: x.__name__)
def test_get_dtype_error_catch(func):
    # 见 gh-15941
    #
    # 不应该引发异常
    msg = f"{func.__name__} is deprecated"
    warn = None
    if (
        func is com.is_int64_dtype
        or func is com.is_interval_dtype
        or func is com.is_datetime64tz_dtype
        or func is com.is_categorical_dtype
        or func is com.is_period_dtype
    ):
        warn = DeprecationWarning

    with tm.assert_produces_warning(warn, match=msg):
        assert not func(None)


# 测试 is_object_dtype 函数，验证它的行为是否符合预期
def test_is_object():
    assert com.is_object_dtype(object)
    assert com.is_object_dtype(np.array([], dtype=object))

    assert not com.is_object_dtype(int)
    assert not com.is_object_dtype(np.array([], dtype=int))
    assert not com.is_object_dtype([1, 2, 3])


# 使用参数化测试，测试是否能正确处理 scipy 的情况
@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
)
# 测试函数，检查函数 is_sparse 是否已被弃用
def test_is_sparse(check_scipy):
    # 弃用警告消息
    msg = "is_sparse is deprecated"
    # 断言 is_sparse 对 SparseArray([1, 2, 3]) 返回 True
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert com.is_sparse(SparseArray([1, 2, 3]))

        # 断言 is_sparse 对 np.array([1, 2, 3]) 返回 False
        assert not com.is_sparse(np.array([1, 2, 3]))

        # 如果 check_scipy 为真，检查 scipy.sparse.bsr_matrix([1, 2, 3]) 是否不是稀疏矩阵
        if check_scipy:
            import scipy.sparse
            # 断言 is_sparse 对 scipy.sparse.bsr_matrix([1, 2, 3]) 返回 False
            assert not com.is_sparse(scipy.sparse.bsr_matrix([1, 2, 3]))


# 测试函数，检查函数 is_scipy_sparse 的功能
def test_is_scipy_sparse():
    # 导入 scipy.sparse 库，如果不存在则跳过测试
    sp_sparse = pytest.importorskip("scipy.sparse")

    # 断言 is_scipy_sparse 对 sp_sparse.bsr_matrix([1, 2, 3]) 返回 True
    assert com.is_scipy_sparse(sp_sparse.bsr_matrix([1, 2, 3]))

    # 断言 is_scipy_sparse 对 SparseArray([1, 2, 3]) 返回 False
    assert not com.is_scipy_sparse(SparseArray([1, 2, 3]))


# 测试函数，检查函数 is_datetime64_dtype 的功能
def test_is_datetime64_dtype():
    # 断言 is_datetime64_dtype 对 object 返回 False
    assert not com.is_datetime64_dtype(object)
    # 断言 is_datetime64_dtype 对 [1, 2, 3] 返回 False
    assert not com.is_datetime64_dtype([1, 2, 3])
    # 断言 is_datetime64_dtype 对 np.array([], dtype=int) 返回 False
    assert not com.is_datetime64_dtype(np.array([], dtype=int))

    # 断言 is_datetime64_dtype 对 np.datetime64 返回 True
    assert com.is_datetime64_dtype(np.datetime64)
    # 断言 is_datetime64_dtype 对 np.array([], dtype=np.datetime64) 返回 True
    assert com.is_datetime64_dtype(np.array([], dtype=np.datetime64))


# 测试函数，检查函数 is_datetime64tz_dtype 的功能
def test_is_datetime64tz_dtype():
    # 弃用警告消息
    msg = "is_datetime64tz_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 断言 is_datetime64tz_dtype 对 object 返回 False
        assert not com.is_datetime64tz_dtype(object)
        # 断言 is_datetime64tz_dtype 对 [1, 2, 3] 返回 False
        assert not com.is_datetime64tz_dtype([1, 2, 3])
        # 断言 is_datetime64tz_dtype 对 pd.DatetimeIndex([1, 2, 3]) 返回 False
        assert not com.is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))
        # 断言 is_datetime64tz_dtype 对 pd.DatetimeIndex(["2000"], tz="US/Eastern") 返回 True
        assert com.is_datetime64tz_dtype(pd.DatetimeIndex(["2000"], tz="US/Eastern"))


# 测试函数，检查自定义的类 NotTZDtype 的功能
def test_custom_ea_kind_M_not_datetime64tz():
    # GH 34986
    # 定义一个不是时区类型的自定义类 NotTZDtype
    class NotTZDtype(ExtensionDtype):
        @property
        def kind(self) -> str:
            return "M"

    # 创建 NotTZDtype 类的实例
    not_tz_dtype = NotTZDtype()
    # 弃用警告消息
    msg = "is_datetime64tz_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 断言 is_datetime64tz_dtype 对 not_tz_dtype 返回 False
        assert not com.is_datetime64tz_dtype(not_tz_dtype)
        # 断言 needs_i8_conversion 对 not_tz_dtype 返回 False
        assert not com.needs_i8_conversion(not_tz_dtype)


# 测试函数，检查函数 is_timedelta64_dtype 的功能
def test_is_timedelta64_dtype():
    # 断言 is_timedelta64_dtype 对 object 返回 False
    assert not com.is_timedelta64_dtype(object)
    # 断言 is_timedelta64_dtype 对 None 返回 False
    assert not com.is_timedelta64_dtype(None)
    # 断言 is_timedelta64_dtype 对 [1, 2, 3] 返回 False
    assert not com.is_timedelta64_dtype([1, 2, 3])
    # 断言 is_timedelta64_dtype 对 np.array([], dtype=np.datetime64) 返回 False
    assert not com.is_timedelta64_dtype(np.array([], dtype=np.datetime64))
    # 断言 is_timedelta64_dtype 对 "0 days" 返回 False
    assert not com.is_timedelta64_dtype("0 days")
    # 断言 is_timedelta64_dtype 对 "0 days 00:00:00" 返回 False
    assert not com.is_timedelta64_dtype("0 days 00:00:00")
    # 断言 is_timedelta64_dtype 对 ["0 days 00:00:00"] 返回 False
    assert not com.is_timedelta64_dtype(["0 days 00:00:00"])
    # 断言 is_timedelta64_dtype 对 "NO DATE" 返回 False
    assert not com.is_timedelta64_dtype("NO DATE")

    # 断言 is_timedelta64_dtype 对 np.timedelta64 返回 True
    assert com.is_timedelta64_dtype(np.timedelta64)
    # 断言 is_timedelta64_dtype 对 pd.Series([], dtype="timedelta64[ns]") 返回 True
    assert com.is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    # 断言 is_timedelta64_dtype 对 pd.to_timedelta(["0 days", "1 days"]) 返回 True
    assert com.is_timedelta64_dtype(pd.to_timedelta(["0 days", "1 days"]))


# 测试函数，检查函数 is_period_dtype 的功能
def test_is_period_dtype():
    # 弃用警告消息
    msg = "is_period_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 断言 is_period_dtype 对 object 返回 False
        assert not com.is_period_dtype(object)
        # 断言 is_period_dtype 对 [1, 2, 3] 返回 False
        assert not com.is_period_dtype([1, 2, 3])
        # 断言 is_period_dtype 对 pd.Period("2017-01-01") 返回 False
        assert not com.is_period_dtype(pd.Period("2017-01-01"))

        # 断言 is_period_dtype 对 PeriodDtype(freq="D") 返回 True
        assert com.is_period_dtype(PeriodDtype(freq="D"))
        # 断言 is_period_dtype 对 pd.PeriodIndex([], freq="Y") 返回 True
        assert com.is_period_dtype(pd.PeriodIndex([], freq="Y"))


# 测试函数，检查函数 is_interval_dtype 的功能
def test_is_interval_dtype():
    # 弃用警告消息
    msg = "is_interval_dtype is deprecated"
    # 使用上下文管理器确保产生特定警告类型的警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 检查对象是否不是间隔数据类型
        assert not com.is_interval_dtype(object)
        # 检查列表是否不是间隔数据类型
        assert not com.is_interval_dtype([1, 2, 3])
    
        # 创建一个间隔数据类型的对象，并检查是否是间隔数据类型
        assert com.is_interval_dtype(IntervalDtype())
    
        # 创建一个右闭区间 [1, 2] 的对象，并检查是否不是间隔数据类型
        interval = pd.Interval(1, 2, closed="right")
        assert not com.is_interval_dtype(interval)
        # 创建一个包含上述间隔对象的间隔索引，并检查是否是间隔数据类型
        assert com.is_interval_dtype(pd.IntervalIndex([interval]))
def test_is_categorical_dtype():
    msg = "is_categorical_dtype is deprecated"
    # 使用上下文管理器检查是否会产生特定的 DeprecationWarning，匹配特定消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 检查 object 不是分类数据类型
        assert not com.is_categorical_dtype(object)
        # 检查列表 [1, 2, 3] 不是分类数据类型
        assert not com.is_categorical_dtype([1, 2, 3])

        # 检查 CategoricalDtype() 是分类数据类型
        assert com.is_categorical_dtype(CategoricalDtype())
        # 检查 pd.Categorical([1, 2, 3]) 是分类数据类型
        assert com.is_categorical_dtype(pd.Categorical([1, 2, 3]))
        # 检查 pd.CategoricalIndex([1, 2, 3]) 是分类数据类型
        assert com.is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (int, False),
        (pd.Series([1, 2]), False),
        (str, True),
        (object, True),
        (np.array(["a", "b"]), True),
        (pd.StringDtype(), True),
        (pd.Index([], dtype="O"), True),
    ],
)
def test_is_string_dtype(dtype, expected):
    # GH#54661
    # 测试 com.is_string_dtype 函数对给定 dtype 的返回结果是否符合预期
    result = com.is_string_dtype(dtype)
    assert result is expected


@pytest.mark.parametrize(
    "data",
    [[(0, 1), (1, 1)], pd.Categorical([1, 2, 3]), np.array([1, 2], dtype=object)],
)
def test_is_string_dtype_arraylike_with_object_elements_not_strings(data):
    # GH 15585
    # 测试 com.is_string_dtype 函数对给定的数组或类数组对象是否返回 False
    assert not com.is_string_dtype(pd.Series(data))


def test_is_string_dtype_nullable(nullable_string_dtype):
    # 检查给定的 nullable_string_dtype 是否被 com.is_string_dtype 函数识别为字符串类型
    assert com.is_string_dtype(pd.array(["a", "b"], dtype=nullable_string_dtype))


integer_dtypes: list = []

@pytest.mark.parametrize(
    "dtype",
    integer_dtypes
    + [pd.Series([1, 2])]
    + tm.ALL_INT_NUMPY_DTYPES
    + to_numpy_dtypes(tm.ALL_INT_NUMPY_DTYPES)
    + tm.ALL_INT_EA_DTYPES
    + to_ea_dtypes(tm.ALL_INT_EA_DTYPES),
)
def test_is_integer_dtype(dtype):
    # 检查 com.is_integer_dtype 函数对给定 dtype 是否返回 True
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype=np.timedelta64),
    ],
)
def test_is_not_integer_dtype(dtype):
    # 检查 com.is_integer_dtype 函数对给定 dtype 是否返回 False
    assert not com.is_integer_dtype(dtype)


signed_integer_dtypes: list = []

@pytest.mark.parametrize(
    "dtype",
    signed_integer_dtypes
    + [pd.Series([1, 2])]
    + tm.SIGNED_INT_NUMPY_DTYPES
    + to_numpy_dtypes(tm.SIGNED_INT_NUMPY_DTYPES)
    + tm.SIGNED_INT_EA_DTYPES
    + to_ea_dtypes(tm.SIGNED_INT_EA_DTYPES),
)
def test_is_signed_integer_dtype(dtype):
    # 检查 com.is_integer_dtype 函数对给定 dtype 是否返回 True
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype=np.timedelta64),
    ]
    + tm.UNSIGNED_INT_NUMPY_DTYPES
    + to_numpy_dtypes(tm.UNSIGNED_INT_NUMPY_DTYPES)
    + tm.UNSIGNED_INT_EA_DTYPES
    + to_ea_dtypes(tm.UNSIGNED_INT_EA_DTYPES),
)
def test_is_not_signed_integer_dtype(dtype):
    # 检查 com.is_integer_dtype 函数对给定 dtype 是否返回 False
    assert not com.is_signed_integer_dtype(dtype)


unsigned_integer_dtypes: list = []

@pytest.mark.parametrize(
    "dtype",
    unsigned_integer_dtypes
    + [pd.Series([1, 2], dtype=np.uint32)]
    + tm.UNSIGNED_INT_NUMPY_DTYPES
    # 将 tm.UNSIGNED_INT_NUMPY_DTYPES 转换为 NumPy 数据类型
    + to_numpy_dtypes(tm.UNSIGNED_INT_NUMPY_DTYPES)
    # 添加 tm.UNSIGNED_INT_EA_DTYPES 到当前语句中
    + tm.UNSIGNED_INT_EA_DTYPES
    # 将 tm.UNSIGNED_INT_EA_DTYPES 转换为 ea 数据类型
    + to_ea_dtypes(tm.UNSIGNED_INT_EA_DTYPES),
def test_is_unsigned_integer_dtype(dtype):
    # 断言给定的 dtype 是无符号整数类型
    assert com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),  # 创建一个包含整数和浮点数的索引对象
        np.array(["a", "b"]),  # 创建一个包含字符串的 NumPy 数组
        np.array([], dtype=np.timedelta64),  # 创建一个空的时间间隔 NumPy 数组
    ]
    + tm.SIGNED_INT_NUMPY_DTYPES  # 使用 tm 模块中定义的有符号整数 NumPy 类型
    + to_numpy_dtypes(tm.SIGNED_INT_NUMPY_DTYPES)  # 转换 tm 模块中定义的有符号整数 NumPy 类型
    + tm.SIGNED_INT_EA_DTYPES  # 使用 tm 模块中定义的有符号整数 EA 类型
    + to_ea_dtypes(tm.SIGNED_INT_EA_DTYPES),  # 转换 tm 模块中定义的有符号整数 EA 类型
)
def test_is_not_unsigned_integer_dtype(dtype):
    # 断言给定的 dtype 不是无符号整数类型
    assert not com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype", [np.int64, np.array([1, 2], dtype=np.int64), "Int64", pd.Int64Dtype]
)
def test_is_int64_dtype(dtype):
    # 使用已废弃的 is_int64_dtype 进行断言，同时匹配废弃警告信息
    msg = "is_int64_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert com.is_int64_dtype(dtype)


def test_type_comparison_with_numeric_ea_dtype(any_numeric_ea_dtype):
    # GH#43038
    # 断言 pandas_dtype 函数返回的类型与给定的任意数值 EA 类型相等
    assert pandas_dtype(any_numeric_ea_dtype) == any_numeric_ea_dtype


def test_type_comparison_with_real_numpy_dtype(any_real_numpy_dtype):
    # GH#43038
    # 断言 pandas_dtype 函数返回的类型与给定的任意实数 NumPy 类型相等
    assert pandas_dtype(any_real_numpy_dtype) == any_real_numpy_dtype


def test_type_comparison_with_signed_int_ea_dtype_and_signed_int_numpy_dtype(
    any_signed_int_ea_dtype, any_signed_int_numpy_dtype
):
    # GH#43038
    # 断言 pandas_dtype 函数返回的类型与给定的任意有符号整数 EA 类型不相等
    assert not pandas_dtype(any_signed_int_ea_dtype) == any_signed_int_numpy_dtype


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.int32,
        np.uint64,
        pd.Index([1, 2.0]),  # 创建一个包含整数和浮点数的索引对象
        np.array(["a", "b"]),  # 创建一个包含字符串的 NumPy 数组
        np.array([1, 2], dtype=np.uint32),  # 创建一个包含无符号整数的 NumPy 数组
        "int8",
        "Int8",
        pd.Int8Dtype,
    ],
)
def test_is_not_int64_dtype(dtype):
    # 断言给定的 dtype 不是 int64 类型，并匹配废弃警告信息
    msg = "is_int64_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_int64_dtype(dtype)


def test_is_datetime64_any_dtype():
    # 断言 com 模块中的函数对不同类型的数据返回正确的 datetime64 结果
    assert not com.is_datetime64_any_dtype(int)
    assert not com.is_datetime64_any_dtype(str)
    assert not com.is_datetime64_any_dtype(np.array([1, 2]))
    assert not com.is_datetime64_any_dtype(np.array(["a", "b"]))

    assert com.is_datetime64_any_dtype(np.datetime64)
    assert com.is_datetime64_any_dtype(np.array([], dtype=np.datetime64))
    assert com.is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_any_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]")
    )


def test_is_datetime64_ns_dtype():
    # 断言 com 模块中的函数对不同类型的数据返回正确的 datetime64[ns] 结果
    assert not com.is_datetime64_ns_dtype(int)
    assert not com.is_datetime64_ns_dtype(str)
    assert not com.is_datetime64_ns_dtype(np.datetime64)
    assert not com.is_datetime64_ns_dtype(np.array([1, 2]))
    assert not com.is_datetime64_ns_dtype(np.array(["a", "b"]))
    assert not com.is_datetime64_ns_dtype(np.array([], dtype=np.datetime64))

    # 这个 datetime 数组的单位不正确 (ps 而不是 ns)
    assert not com.is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))
    # 使用 assert 断言检查给定的 DatetimeTZDtype 对象是否为 'ns' 纳秒精度的 datetime64 类型，应返回 True
    assert com.is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    
    # 使用 assert 断言检查给定的 DatetimeIndex 对象是否包含 'ns' 纳秒精度的 datetime64 类型，应返回 True
    assert com.is_datetime64_ns_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype=np.dtype("datetime64[ns]"))
    )
    
    # 使用 assert 断言检查给定的 DatetimeTZDtype 对象是否不是 'ns' 纳秒精度的 datetime64tz 类型，应返回 False
    # 这里使用的是 'us' 微秒精度
    assert not com.is_datetime64_ns_dtype(DatetimeTZDtype("us", "US/Eastern"))
# 测试函数，用于检查是否为 'timedelta64[ps]' 类型
def test_is_timedelta64_ns_dtype():
    # 断言所给数据类型不是 'timedelta64[ps]'
    assert not com.is_timedelta64_ns_dtype(np.dtype("m8[ps]"))
    # 断言所给数组不是 'timedelta64[ps]' 类型
    assert not com.is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))

    # 断言所给数据类型是 'timedelta64[ns]'
    assert com.is_timedelta64_ns_dtype(np.dtype("m8[ns]"))
    # 断言所给数组是 'timedelta64[ns]' 类型
    assert com.is_timedelta64_ns_dtype(np.array([1, 2], dtype="m8[ns]"))


# 测试函数，检查是否为数值或字符串类型
def test_is_numeric_v_string_like():
    # 断言所给数组不是数值或字符串类型
    assert not com.is_numeric_v_string_like(np.array([1]), 1)
    # 断言所给数组不是数值或字符串类型
    assert not com.is_numeric_v_string_like(np.array([1]), np.array([2]))
    # 断言所给数组不是数值或字符串类型
    assert not com.is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))

    # 断言所给数组是数值或字符串类型
    assert com.is_numeric_v_string_like(np.array([1]), "foo")
    # 断言所给数组是数值或字符串类型
    assert com.is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    # 断言所给数组是数值或字符串类型
    assert com.is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))


# 测试函数，检查是否需要进行 'i8' 类型的转换
def test_needs_i8_conversion():
    # 断言所给类型不需要 'i8' 类型的转换
    assert not com.needs_i8_conversion(str)
    # 断言所给类型不需要 'i8' 类型的转换
    assert not com.needs_i8_conversion(np.int64)
    # 断言所给 Series 不需要 'i8' 类型的转换
    assert not com.needs_i8_conversion(pd.Series([1, 2]))
    # 断言所给数组不需要 'i8' 类型的转换
    assert not com.needs_i8_conversion(np.array(["a", "b"]))

    # 断言所给类型不需要 'i8' 类型的转换
    assert not com.needs_i8_conversion(np.datetime64)
    # 断言所给 dtype 需要 'i8' 类型的转换
    assert com.needs_i8_conversion(np.dtype(np.datetime64))
    # 断言所给 Series 需要 'i8' 类型的转换
    assert not com.needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    # 断言所给 dtype 需要 'i8' 类型的转换
    assert com.needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]").dtype)
    # 断言所给 DatetimeIndex 不需要 'i8' 类型的转换
    assert not com.needs_i8_conversion(pd.DatetimeIndex(["2000"], tz="US/Eastern"))
    # 断言所给 dtype 需要 'i8' 类型的转换
    assert com.needs_i8_conversion(pd.DatetimeIndex(["2000"], tz="US/Eastern").dtype)


# 测试函数，检查是否为数值类型
def test_is_numeric_dtype():
    # 断言所给类型不是数值类型
    assert not com.is_numeric_dtype(str)
    # 断言所给类型不是数值类型
    assert not com.is_numeric_dtype(np.datetime64)
    # 断言所给类型不是数值类型
    assert not com.is_numeric_dtype(np.timedelta64)
    # 断言所给数组不是数值类型
    assert not com.is_numeric_dtype(np.array(["a", "b"]))
    # 断言所给数组不是数值类型
    assert not com.is_numeric_dtype(np.array([], dtype=np.timedelta64))

    # 断言所给类型是数值类型
    assert com.is_numeric_dtype(int)
    # 断言所给类型是数值类型
    assert com.is_numeric_dtype(float)
    # 断言所给类型是数值类型
    assert com.is_numeric_dtype(np.uint64)
    # 断言所给 Series 是数值类型
    assert com.is_numeric_dtype(pd.Series([1, 2]))
    # 断言所给 Index 是数值类型
    assert com.is_numeric_dtype(pd.Index([1, 2.0]))

    # 自定义的数值类型类，断言其为数值类型
    class MyNumericDType(ExtensionDtype):
        @property
        def type(self):
            return str

        @property
        def name(self):
            raise NotImplementedError

        @classmethod
        def construct_array_type(cls):
            raise NotImplementedError

        def _is_numeric(self) -> bool:
            return True

    assert com.is_numeric_dtype(MyNumericDType())


# 测试函数，检查是否为任何实数类型
def test_is_any_real_numeric_dtype():
    # 断言所给类型不是任何实数类型
    assert not com.is_any_real_numeric_dtype(str)
    # 断言所给类型不是任何实数类型
    assert not com.is_any_real_numeric_dtype(bool)
    # 断言所给类型不是任何实数类型
    assert not com.is_any_real_numeric_dtype(complex)
    # 断言所给类型不是任何实数类型
    assert not com.is_any_real_numeric_dtype(object)
    # 断言所给类型不是任何实数类型
    assert not com.is_any_real_numeric_dtype(np.datetime64)
    # 断言所给数组不是任何实数类型
    assert not com.is_any_real_numeric_dtype(np.array(["a", "b", complex(1, 2)]))
    # 断言所给 DataFrame 不是任何实数类型
    assert not com.is_any_real_numeric_dtype(pd.DataFrame([complex(1, 2), True]))

    # 断言所给类型是任何实数类型
    assert com.is_any_real_numeric_dtype(int)
    # 断言所给类型是任何实数类型
    assert com.is_any_real_numeric_dtype(float)
    # 使用断言检查给定的 NumPy 数组是否包含任何实数类型的数据
    assert com.is_any_real_numeric_dtype(np.array([1, 2.5]))
# 检查输入类型是否为浮点数类型，使用 pandas、numpy 和内置类型作为示例
def test_is_float_dtype():
    # 确认 str 类型不是浮点数类型
    assert not com.is_float_dtype(str)
    # 确认 int 类型不是浮点数类型
    assert not com.is_float_dtype(int)
    # 确认包含整数的 pandas Series 不是浮点数类型
    assert not com.is_float_dtype(pd.Series([1, 2]))
    # 确认包含字符串的 numpy array 不是浮点数类型
    assert not com.is_float_dtype(np.array(["a", "b"]))

    # 确认 float 类型是浮点数类型
    assert com.is_float_dtype(float)
    # 确认包含浮点数的 pandas Index 是浮点数类型
    assert com.is_float_dtype(pd.Index([1, 2.0]))


# 检查输入类型是否为布尔类型，使用 pandas、numpy 和内置类型作为示例
def test_is_bool_dtype():
    # 确认 int 类型不是布尔类型
    assert not com.is_bool_dtype(int)
    # 确认 str 类型不是布尔类型
    assert not com.is_bool_dtype(str)
    # 确认包含整数的 pandas Series 不是布尔类型
    assert not com.is_bool_dtype(pd.Series([1, 2]))
    # 确认包含类别数据的 pandas Series 不是布尔类型
    assert not com.is_bool_dtype(pd.Series(["a", "b"], dtype="category"))
    # 确认包含字符串的 numpy array 不是布尔类型
    assert not com.is_bool_dtype(np.array(["a", "b"]))
    # 确认包含字符串的 pandas Index 不是布尔类型
    assert not com.is_bool_dtype(pd.Index(["a", "b"]))
    # 确认字符串 "Int64" 不是布尔类型
    assert not com.is_bool_dtype("Int64")

    # 确认 bool 类型是布尔类型
    assert com.is_bool_dtype(bool)
    # 确认 np.bool_ 类型是布尔类型
    assert com.is_bool_dtype(np.bool_)
    # 确认包含类别数据的 pandas Series 是布尔类型
    assert com.is_bool_dtype(pd.Series([True, False], dtype="category"))
    # 确认包含布尔值的 numpy array 是布尔类型
    assert com.is_bool_dtype(np.array([True, False]))
    # 确认包含布尔值的 pandas Index 是布尔类型
    assert com.is_bool_dtype(pd.Index([True, False]))

    # 确认 pd.BooleanDtype() 是布尔类型
    assert com.is_bool_dtype(pd.BooleanDtype())
    # 确认包含布尔值的 pd.array 是布尔类型
    assert com.is_bool_dtype(pd.array([True, False, None], dtype="boolean"))
    # 确认字符串 "boolean" 是布尔类型
    assert com.is_bool_dtype("boolean")


# 检查输入类型是否为扩展数组类型，使用 pandas 和 numpy 的扩展数组作为示例
def test_is_extension_array_dtype(check_scipy):
    # 确认包含整数的列表不是扩展数组类型
    assert not com.is_extension_array_dtype([1, 2, 3])
    # 确认包含整数的 numpy array 不是扩展数组类型
    assert not com.is_extension_array_dtype(np.array([1, 2, 3]))
    # 确认包含整数的 pandas DatetimeIndex 不是扩展数组类型
    assert not com.is_extension_array_dtype(pd.DatetimeIndex([1, 2, 3]))

    # 确认 pandas Categorical 对象是扩展数组类型
    cat = pd.Categorical([1, 2, 3])
    assert com.is_extension_array_dtype(cat)
    # 确认包含 pandas Categorical 对象的 pandas Series 是扩展数组类型
    assert com.is_extension_array_dtype(pd.Series(cat))
    # 确认 SparseArray 对象是扩展数组类型
    assert com.is_extension_array_dtype(SparseArray([1, 2, 3]))
    # 确认带时区的 pandas DatetimeIndex 是扩展数组类型
    assert com.is_extension_array_dtype(pd.DatetimeIndex(["2000"], tz="US/Eastern"))

    # 使用自定义数据类型创建 pandas Series，并确认其为扩展数组类型
    dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    s = pd.Series([], dtype=dtype)
    assert com.is_extension_array_dtype(s)

    # 如果 check_scipy 为 True，确认 scipy.sparse.bsr_matrix 不是扩展数组类型
    if check_scipy:
        import scipy.sparse
        assert not com.is_extension_array_dtype(scipy.sparse.bsr_matrix([1, 2, 3]))


# 检查输入类型是否为复数类型，使用 numpy 的复数类型作为示例
def test_is_complex_dtype():
    # 确认 int 类型不是复数类型
    assert not com.is_complex_dtype(int)
    # 确认 str 类型不是复数类型
    assert not com.is_complex_dtype(str)
    # 确认包含整数的 pandas Series 不是复数类型
    assert not com.is_complex_dtype(pd.Series([1, 2]))
    # 确认包含字符串的 numpy array 不是复数类型
    assert not com.is_complex_dtype(np.array(["a", "b"]))

    # 确认 numpy 的 np.complex128 类型是复数类型
    assert com.is_complex_dtype(np.complex128)
    # 确认内置的 complex 类型是复数类型
    assert com.is_complex_dtype(complex)
    # 确认包含复数的 numpy array 是复数类型
    assert com.is_complex_dtype(np.array([1 + 1j, 5]))


# 使用 pytest 的 parametrize 功能对 check_scipy 参数化，验证 scipy 是否支持扩展数组类型
@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
)
    [
        # 定义一个包含多个元组的列表，每个元组包含两个元素：数据类型和对应的 NumPy 数据类型
        (int, np.dtype(int)),
        # 整数类型和其对应的 NumPy int32 数据类型
        ("int32", np.dtype("int32")),
        # 浮点数类型和其对应的 NumPy float 数据类型
        (float, np.dtype(float)),
        # 浮点数类型和其对应的 NumPy float64 数据类型
        ("float64", np.dtype("float64")),
        # NumPy float64 类型和其对应的 NumPy float64 数据类型
        (np.dtype("float64"), np.dtype("float64")),
        # 字符串类型和其对应的 NumPy str 数据类型
        (str, np.dtype(str)),
        # Pandas Series，包含整数的 Series 和其对应的 NumPy int16 数据类型
        (pd.Series([1, 2], dtype=np.dtype("int16")), np.dtype("int16")),
        # Pandas Series，包含字符串的 Series 和其对应的 NumPy object 数据类型
        (pd.Series(["a", "b"], dtype=object), np.dtype(object)),
        # Pandas Index，包含整数的 Index 和其对应的 NumPy int64 数据类型
        (pd.Index([1, 2]), np.dtype("int64")),
        # Pandas Index，包含字符串的 Index 和其对应的 NumPy object 数据类型
        (pd.Index(["a", "b"], dtype=object), np.dtype(object)),
        # 字符串 "category" 和其对应的 Pandas Categorical 数据类型 "category"
        ("category", "category"),
        # Pandas Categorical 数据类型，包含字符串的 Series 的 dtype 和其对应的 CategoricalDtype
        (pd.Categorical(["a", "b"]).dtype, CategoricalDtype(["a", "b"])),
        # Pandas Categorical 类型，包含字符串的 Categorical 对象 和其对应的 CategoricalDtype
        (pd.Categorical(["a", "b"]), CategoricalDtype(["a", "b"])),
        # Pandas CategoricalIndex，包含字符串的 Index 的 dtype 和其对应的 CategoricalDtype
        (pd.CategoricalIndex(["a", "b"]).dtype, CategoricalDtype(["a", "b"])),
        # Pandas CategoricalIndex，包含字符串的 CategoricalIndex 对象 和其对应的 CategoricalDtype
        (pd.CategoricalIndex(["a", "b"]), CategoricalDtype(["a", "b"])),
        # 空的 CategoricalDtype 对象 和其对应的空的 CategoricalDtype 对象
        (CategoricalDtype(), CategoricalDtype()),
        # Pandas DatetimeIndex，包含整数的 DatetimeIndex 和其对应的 NumPy datetime64[ns] 数据类型
        (pd.DatetimeIndex([1, 2]), np.dtype("=M8[ns]")),
        # Pandas DatetimeIndex 的 dtype 和其对应的 NumPy datetime64[ns] 数据类型
        (pd.DatetimeIndex([1, 2]).dtype, np.dtype("=M8[ns]")),
        # 字符串 "<M8[ns]" 和其对应的 NumPy datetime64[ns] 数据类型
        ("<M8[ns]", np.dtype("<M8[ns]")),
        # 字符串 "datetime64[ns, Europe/London]" 和其对应的带时区信息的 DatetimeTZDtype 类型
        ("datetime64[ns, Europe/London]", DatetimeTZDtype("ns", "Europe/London")),
        # Pandas PeriodDtype(freq="D") 和其对应的 PeriodDtype(freq="D") 类型
        (PeriodDtype(freq="D"), PeriodDtype(freq="D")),
        # 字符串 "period[D]" 和其对应的 PeriodDtype(freq="D") 类型
        ("period[D]", PeriodDtype(freq="D")),
        # 空的 IntervalDtype 对象 和其对应的空的 IntervalDtype 对象
        (IntervalDtype(), IntervalDtype()),
    ],
# 测试函数，验证 com._get_dtype 是否返回预期结果
def test_get_dtype(input_param, result):
    assert com._get_dtype(input_param) == result


# 使用 pytest 的参数化功能，定义多组输入参数及其对应的预期错误消息
@pytest.mark.parametrize(
    "input_param,expected_error_message",
    [
        (None, "Cannot deduce dtype from null object"),
        (1, "data type not understood"),
        (1.2, "data type not understood"),
        ("random string", "data type [\"']random string[\"'] not understood"),
        (pd.DataFrame([1, 2]), "data type not understood"),
    ],
)
# 测试 com._get_dtype 函数是否会抛出预期的 TypeError 异常
def test_get_dtype_fails(input_param, expected_error_message):
    expected_error_message += f"|Cannot interpret '{input_param}' as a data type"
    with pytest.raises(TypeError, match=expected_error_message):
        com._get_dtype(input_param)


# 使用 pytest 的参数化功能，定义多组输入参数及其对应的预期输出结果
@pytest.mark.parametrize(
    "input_param,result",
    [
        (int, np.dtype(int).type),
        ("int32", np.int32),
        (float, np.dtype(float).type),
        ("float64", np.float64),
        (np.dtype("float64"), np.float64),
        (str, np.dtype(str).type),
        (pd.Series([1, 2], dtype=np.dtype("int16")), np.int16),
        (pd.Series(["a", "b"], dtype=object), np.object_),
        (pd.Index([1, 2], dtype="int64"), np.int64),
        (pd.Index(["a", "b"], dtype=object), np.object_),
        ("category", CategoricalDtypeType),
        (pd.Categorical(["a", "b"]).dtype, CategoricalDtypeType),
        (pd.Categorical(["a", "b"]), CategoricalDtypeType),
        (pd.CategoricalIndex(["a", "b"]).dtype, CategoricalDtypeType),
        (pd.CategoricalIndex(["a", "b"]), CategoricalDtypeType),
        (pd.DatetimeIndex([1, 2]), np.datetime64),
        (pd.DatetimeIndex([1, 2]).dtype, np.datetime64),
        ("<M8[ns]", np.datetime64),
        (pd.DatetimeIndex(["2000"], tz="Europe/London"), pd.Timestamp),
        (pd.DatetimeIndex(["2000"], tz="Europe/London").dtype, pd.Timestamp),
        ("datetime64[ns, Europe/London]", pd.Timestamp),
        (PeriodDtype(freq="D"), pd.Period),
        ("period[D]", pd.Period),
        (IntervalDtype(), pd.Interval),
        (None, type(None)),
        (1, type(None)),
        (1.2, type(None)),
        (pd.DataFrame([1, 2]), type(None)),
    ],
)
# 测试 com._is_dtype_type 函数是否返回预期结果
def test__is_dtype_type(input_param, result):
    assert com._is_dtype_type(input_param, lambda tipo: tipo == result)


# 测试函数，验证 astype_array 函数在 copy=False 模式下的行为
def test_astype_nansafe_copy_false(any_int_numpy_dtype):
    # 创建包含任意整数类型的 NumPy 数组
    arr = np.array([1, 2, 3], dtype=any_int_numpy_dtype)

    # 指定目标数据类型为 float64，使用 astype_array 进行类型转换，不进行拷贝
    dtype = np.dtype("float64")
    result = astype_array(arr, dtype, copy=False)

    # 预期的转换结果
    expected = np.array([1.0, 2.0, 3.0], dtype=dtype)
    # 使用测试工具包的函数进行数组相等性断言
    tm.assert_numpy_array_equal(result, expected)


# 使用 pytest 的参数化功能，定义不同的 from_type 输入参数
@pytest.mark.parametrize("from_type", [np.datetime64, np.timedelta64])
# 测试函数，验证 astype_array 函数对于 datetime 和 timedelta 类型的处理
def test_astype_object_preserves_datetime_na(from_type):
    # 创建包含 NaT 值的 NumPy 数组，使用 from_type 指定数据类型
    arr = np.array([from_type("NaT", "ns")])
    # 使用 astype_array 将数组转换为 object 类型
    result = astype_array(arr, dtype=np.dtype("object"))

    # 断言转换后的结果是否保留了 NaN 值
    assert isna(result)[0]
# 定义一个测试函数，用于验证 com.validate_all_hashable 函数的行为
def test_validate_allhashable():
    # 断言调用 com.validate_all_hashable 函数返回 None，验证传入参数 1 和 "a" 均为可哈希类型
    assert com.validate_all_hashable(1, "a") is None

    # 使用 pytest 断言语法，期望调用 com.validate_all_hashable 函数传入空列表时抛出 TypeError 异常，并匹配特定错误信息
    with pytest.raises(TypeError, match="All elements must be hashable"):
        com.validate_all_hashable([])

    # 使用 pytest 断言语法，期望调用 com.validate_all_hashable 函数传入空列表时抛出 TypeError 异常，并匹配特定错误信息 "list"
    with pytest.raises(TypeError, match="list must be a hashable type"):
        com.validate_all_hashable([], error_name="list")


# 定义一个测试函数，用于验证 pandas_dtype 函数对 numpy 数据类型的警告行为
def test_pandas_dtype_numpy_warning():
    # GH#51523
    # 使用 tm.assert_produces_warning 上下文管理器，期望产生 DeprecationWarning 警告，匹配特定的警告信息
    with tm.assert_produces_warning(
        DeprecationWarning,
        check_stacklevel=False,
        match="Converting `np.integer` or `np.signedinteger` to a dtype is deprecated",
    ):
        # 调用 pandas_dtype 函数，传入 np.integer 数据类型


# 定义一个测试函数，用于验证 pandas_dtype 函数对非实例化 CategoricalDtype 类型的警告行为
def test_pandas_dtype_ea_not_instance():
    # GH 31356 GH 54592
    # 使用 tm.assert_produces_warning 上下文管理器，期望产生 UserWarning 警告，匹配特定的警告信息 "without any arguments"
    with tm.assert_produces_warning(UserWarning, match="without any arguments"):
        # 断言调用 pandas_dtype 函数传入 CategoricalDtype 类型返回的结果与 CategoricalDtype() 相等
        assert pandas_dtype(CategoricalDtype) == CategoricalDtype()
```