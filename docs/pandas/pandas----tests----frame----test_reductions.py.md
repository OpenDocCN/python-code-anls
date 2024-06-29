# `D:\src\scipysrc\pandas\pandas\tests\frame\test_reductions.py`

```
# 导入需要的模块和函数
from datetime import timedelta
from decimal import Decimal
import re

from dateutil.tz import tzlocal  # 导入时区本地化函数
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

from pandas._config import using_pyarrow_string_dtype  # 导入配置模块中的函数

from pandas.compat import (
    IS64,
    is_platform_windows,
)  # 导入兼容性相关的函数和常量

from pandas.compat.numpy import np_version_gt2  # 导入NumPy版本检测函数
import pandas.util._test_decorators as td  # 导入测试相关的装饰器函数

import pandas as pd  # 导入Pandas库
from pandas import (  # 导入Pandas中的一些核心对象和函数
    Categorical,
    CategoricalDtype,
    DataFrame,
    DatetimeIndex,
    Index,
    PeriodIndex,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm  # 导入Pandas测试相关的工具函数

from pandas.core import (  # 导入Pandas核心模块中的算法和空值处理函数
    algorithms,
    nanops,
)

# 检查操作系统是否为Windows且运行的Python是否为32位，确定一个布尔值
is_windows_np2_or_is32 = (is_platform_windows() and not np_version_gt2) or not IS64
is_windows_or_is32 = is_platform_windows() or not IS64


def make_skipna_wrapper(alternative, skipna_alternative=None):
    """
    Create a function for calling on an array.

    Parameters
    ----------
    alternative : function
        The function to be called on the array with no NaNs.
        Only used when 'skipna_alternative' is None.
    skipna_alternative : function
        The function to be called on the original array

    Returns
    -------
    function
    """
    if skipna_alternative:
        # 如果skipna_alternative存在，则创建一个处理NaN安全的函数
        def skipna_wrapper(x):
            return skipna_alternative(x.values)

    else:
        # 否则创建一个函数，处理数组中的NaN值
        def skipna_wrapper(x):
            nona = x.dropna()
            if len(nona) == 0:
                return np.nan
            return alternative(nona)

    return skipna_wrapper


def assert_stat_op_calc(
    opname,
    alternative,
    frame,
    has_skipna=True,
    check_dtype=True,
    check_dates=False,
    rtol=1e-5,
    atol=1e-8,
    skipna_alternative=None,
):
    """
    Check that operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    check_dtype : bool, default True
        Whether the dtypes of the result of "frame.opname()" and
        "alternative(frame)" should be checked.
    check_dates : bool, default false
        Whether opname should be tested on a Datetime Series
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    skipna_alternative : function, default None
        NaN-safe version of alternative
    """
    # 获取DataFrame对象上的操作函数
    f = getattr(frame, opname)
    # 如果需要检查日期
    if check_dates:
        # 创建一个包含日期范围的 DataFrame，列名为'b'
        df = DataFrame({"b": date_range("1/1/2001", periods=2)})
        # 断言操作名称返回的结果是 Series 类型
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)

        # 在 DataFrame 中添加列'a'，其值为索引
        df["a"] = range(len(df))
        # 断言操作名称返回的结果是 Series 类型
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)
        # 断言结果长度不为零
        assert len(result)

    # 如果允许跳过缺失值
    if has_skipna:
        # 定义一个包装函数，对输入进行处理并返回 alternative(x.values) 的结果
        def wrapper(x):
            return alternative(x.values)

        # 创建跳过缺失值的包装器
        skipna_wrapper = make_skipna_wrapper(alternative, skipna_alternative)
        # 对 axis=0 方向进行函数 f 的调用，跳过缺失值
        result0 = f(axis=0, skipna=False)
        # 对 axis=1 方向进行函数 f 的调用，跳过缺失值
        result1 = f(axis=1, skipna=False)
        # 断言结果0与 DataFrame 按照 wrapper 函数应用后的结果相等
        tm.assert_series_equal(
            result0, frame.apply(wrapper), check_dtype=check_dtype, rtol=rtol, atol=atol
        )
        # 断言结果1与 DataFrame 按照 wrapper 函数在 axis=1 上应用后的结果相等
        tm.assert_series_equal(
            result1,
            frame.apply(wrapper, axis=1),
            rtol=rtol,
            atol=atol,
        )
    else:
        # 如果不允许跳过缺失值，则直接使用 alternative 作为包装器
        skipna_wrapper = alternative

    # 对 axis=0 方向进行函数 f 的调用
    result0 = f(axis=0)
    # 对 axis=1 方向进行函数 f 的调用
    result1 = f(axis=1)
    # 断言结果0与 DataFrame 按照 skipna_wrapper 函数应用后的结果相等
    tm.assert_series_equal(
        result0,
        frame.apply(skipna_wrapper),
        check_dtype=check_dtype,
        rtol=rtol,
        atol=atol,
    )

    # 如果操作名称是'sum'或'prod'
    if opname in ["sum", "prod"]:
        # 预期结果是在 axis=1 方向上应用 skipna_wrapper 后的结果
        expected = frame.apply(skipna_wrapper, axis=1)
        # 断言结果1与预期结果相等，忽略数据类型的检查
        tm.assert_series_equal(
            result1, expected, check_dtype=False, rtol=rtol, atol=atol
        )

    # 检查数据类型是否一致
    if check_dtype:
        # 获取 DataFrame 的值的数据类型
        lcd_dtype = frame.values.dtype
        # 断言结果0的数据类型与 DataFrame 的值的数据类型一致
        assert lcd_dtype == result0.dtype
        # 断言结果1的数据类型与 DataFrame 的值的数据类型一致
        assert lcd_dtype == result1.dtype

    # 错误的轴
    with pytest.raises(ValueError, match="No axis named 2"):
        # 调用 f 函数，期望引发错误，因为轴的值为2不存在
        f(axis=2)

    # 全为 NA 的情况
    if has_skipna:
        # 创建一个全为 NaN 的 DataFrame
        all_na = frame * np.nan
        # 对 axis=0 方向进行操作名称的调用
        r0 = getattr(all_na, opname)(axis=0)
        # 对 axis=1 方向进行操作名称的调用
        r1 = getattr(all_na, opname)(axis=1)
        # 如果操作名称是'sum'或'prod'
        if opname in ["sum", "prod"]:
            # 对结果0进行比较，预期结果是全为单位值（0或1）
            unit = 1 if opname == "prod" else 0  # 空sum/prod的结果
            expected = Series(unit, index=r0.index, dtype=r0.dtype)
            # 断言结果0与预期结果相等
            tm.assert_series_equal(r0, expected)
            expected = Series(unit, index=r1.index, dtype=r1.dtype)
            # 断言结果1与预期结果相等
            tm.assert_series_equal(r1, expected)
@pytest.fixture
def bool_frame_with_na():
    """
    生成包含布尔值的DataFrame的fixture，索引为唯一字符串

    列为 ['A', 'B', 'C', 'D']；部分条目缺失
    """
    df = DataFrame(
        np.concatenate(
            [np.ones((15, 4), dtype=bool), np.zeros((15, 4), dtype=bool)], axis=0
        ),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
        dtype=object,
    )
    # 设置一些缺失值
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


@pytest.fixture
def float_frame_with_na():
    """
    生成包含浮点数的DataFrame的fixture，索引为唯一字符串

    列为 ['A', 'B', 'C', 'D']；部分条目缺失
    """
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
    )
    # 设置一些缺失值
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


class TestDataFrameAnalytics:
    # ---------------------------------------------------------------------
    # Reductions
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "opname",
        [
            "count",
            "sum",
            "mean",
            "product",
            "median",
            "min",
            "max",
            "nunique",
            "var",
            "std",
            "sem",
            pytest.param("skew", marks=td.skip_if_no("scipy")),
            pytest.param("kurt", marks=td.skip_if_no("scipy")),
        ],
    )
    def test_stat_op_api_float_string_frame(
        self, float_string_frame, axis, opname, using_infer_string
    ):
        """
        测试DataFrame的统计操作API，适用于包含浮点数和字符串的DataFrame

        参数:
        - float_string_frame: 包含浮点数和字符串的DataFrame
        - axis: 统计操作的轴向 (0或1)
        - opname: 统计操作的名称
        - using_infer_string: 是否使用字符串推断

        """
    ):
        # 检查操作名是否为"sum", "min", "max"且轴为0，或者为"count", "nunique"之一
        if (
            (opname in ("sum", "min", "max") and axis == 0)
            or opname
            in (
                "count",
                "nunique",
            )
        ) and not (using_infer_string and opname == "sum"):
            # 如果符合条件，则调用float_string_frame对象的对应操作方法，仅指定轴参数
            getattr(float_string_frame, opname)(axis=axis)
        else:
            # 否则根据操作名选择错误消息
            if opname in ["var", "std", "sem", "skew", "kurt"]:
                msg = "could not convert string to float: 'bar'"
            elif opname == "product":
                # 根据轴的不同选择不同的乘法错误消息
                if axis == 1:
                    msg = "can't multiply sequence by non-int of type 'float'"
                else:
                    msg = "can't multiply sequence by non-int of type 'str'"
            elif opname == "sum":
                msg = r"unsupported operand type\(s\) for \+: 'float' and 'str'"
            elif opname == "mean":
                # 根据轴的不同选择不同的均值错误消息
                if axis == 0:
                    # 不同构建的不同消息
                    msg = "|".join(
                        [
                            r"Could not convert \['.*'\] to numeric",
                            "Could not convert string '(bar){30}' to numeric",
                        ]
                    )
                else:
                    msg = r"unsupported operand type\(s\) for \+: 'float' and 'str'"
            elif opname in ["min", "max"]:
                msg = "'[><]=' not supported between instances of 'float' and 'str'"
            elif opname == "median":
                # 正则表达式匹配的消息
                msg = re.compile(
                    r"Cannot convert \[.*\] to numeric|does not support", flags=re.S
                )
            # 如果消息不是正则表达式对象，则添加后缀
            if not isinstance(msg, re.Pattern):
                msg = msg + "|does not support"
            # 使用pytest检查是否引发TypeError异常，并匹配消息
            with pytest.raises(TypeError, match=msg):
                getattr(float_string_frame, opname)(axis=axis)
        # 如果操作名不是"nunique"，则再次调用操作方法，指定numeric_only参数为True
        if opname != "nunique":
            getattr(float_string_frame, opname)(axis=axis, numeric_only=True)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "opname",
        [
            "count",
            "sum",
            "mean",
            "product",
            "median",
            "min",
            "max",
            "var",
            "std",
            "sem",
            pytest.param("skew", marks=td.skip_if_no("scipy")),
            pytest.param("kurt", marks=td.skip_if_no("scipy")),
        ],
    )
    # 测试统计操作的API，应用于浮点数框架，根据参数化给定的操作名和轴
    def test_stat_op_api_float_frame(self, float_frame, axis, opname):
        getattr(float_frame, opname)(axis=axis, numeric_only=False)
    # 定义一个嵌套函数 `count`，计算 Series 中非缺失值的数量
    def count(s):
        return notna(s).sum()

    # 定义一个嵌套函数 `nunique`，计算 Series 中的唯一值数量
    def nunique(s):
        return len(algorithms.unique1d(s.dropna()))

    # 定义一个嵌套函数 `var`，使用 numpy 计算 Series 或 DataFrame 列的方差（无偏估计）
    def var(x):
        return np.var(x, ddof=1)

    # 定义一个嵌套函数 `std`，使用 numpy 计算 Series 或 DataFrame 列的标准差（无偏估计）
    def std(x):
        return np.std(x, ddof=1)

    # 定义一个嵌套函数 `sem`，使用 numpy 计算 Series 或 DataFrame 列的标准误（无偏估计）
    def sem(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `nunique` 的结果
    assert_stat_op_calc(
        "nunique",
        nunique,
        float_frame_with_na,
        has_skipna=False,
        check_dtype=False,
        check_dates=True,
    )

    # GH#32571: 为了修复持续集成中的问题，需要为混合类型的 DataFrame 执行 sum 操作，使用 float32 进行类型转换
    # 期望结果相对误差为 1e-3
    assert_stat_op_calc(
        "sum",
        np.sum,
        mixed_float_frame.astype("float32"),
        check_dtype=False,
        rtol=1e-3,
    )

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `sum` 的结果
    assert_stat_op_calc(
        "sum", np.sum, float_frame_with_na, skipna_alternative=np.nansum
    )

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `mean` 的结果
    assert_stat_op_calc("mean", np.mean, float_frame_with_na, check_dates=True)

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `product` 的结果
    assert_stat_op_calc(
        "product", np.prod, float_frame_with_na, skipna_alternative=np.nanprod
    )

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `var` 的结果
    assert_stat_op_calc("var", var, float_frame_with_na)

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `std` 的结果
    assert_stat_op_calc("std", std, float_frame_with_na)

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `sem` 的结果
    assert_stat_op_calc("sem", sem, float_frame_with_na)

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `count` 的结果
    assert_stat_op_calc(
        "count",
        count,
        float_frame_with_na,
        has_skipna=False,
        check_dtype=False,
        check_dates=True,
    )

# 定义一个测试函数 `test_stat_op_calc_skew_kurtosis`，用于验证偏度和峰度的统计操作
def test_stat_op_calc_skew_kurtosis(self, float_frame_with_na):
    # 导入 scipy.stats 库，如果不存在则跳过测试
    sp_stats = pytest.importorskip("scipy.stats")

    # 定义一个嵌套函数 `skewness`，计算 Series 或 DataFrame 列的偏度
    def skewness(x):
        if len(x) < 3:
            return np.nan
        return sp_stats.skew(x, bias=False)

    # 定义一个嵌套函数 `kurt`，计算 Series 或 DataFrame 列的峰度
    def kurt(x):
        if len(x) < 4:
            return np.nan
        return sp_stats.kurtosis(x, bias=False)

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `skewness` 的结果
    assert_stat_op_calc("skew", skewness, float_frame_with_na)

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `kurt` 的结果
    assert_stat_op_calc("kurt", kurt, float_frame_with_na)

# 定义一个测试函数 `test_median`，用于验证中位数的统计操作
def test_median(self, float_frame_with_na, int_frame):
    # 定义一个嵌套函数 `wrapper`，计算 Series 或 DataFrame 列的中位数，如果有缺失值则返回 NaN
    def wrapper(x):
        if isna(x).any():
            return np.nan
        return np.median(x)

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `wrapper` 对于 `float_frame_with_na` 的结果
    assert_stat_op_calc("median", wrapper, float_frame_with_na, check_dates=True)

    # 使用自定义的 `assert_stat_op_calc` 函数进行测试，并验证 `wrapper` 对于 `int_frame` 的结果
    assert_stat_op_calc(
        "median", wrapper, int_frame, check_dtype=False, check_dates=True
    )

# 使用 pytest 的参数化装饰器，对下列方法进行多次测试，分别测试 "sum", "mean", "prod", "var", "std", "skew", "min", "max"
@pytest.mark.parametrize(
    "method", ["sum", "mean", "prod", "var", "std", "skew", "min", "max"]
)
    @pytest.mark.parametrize(
        "df",
        [  # 参数化测试数据框
            DataFrame(  # 创建一个数据框，包含三列a, b, c，数据类型为对象
                {
                    "a": [
                        -0.00049987540199591344,
                        -0.0016467257772919831,
                        0.00067695870775883013,
                    ],
                    "b": [-0, -0, 0.0],
                    "c": [
                        0.00031111847529610595,
                        0.0014902627951905339,
                        -0.00094099200035979691,
                    ],
                },
                index=["foo", "bar", "baz"],  # 指定索引为'foo', 'bar', 'baz'
                dtype="O",  # 指定数据类型为对象
            ),
            DataFrame({0: [np.nan, 2], 1: [np.nan, 3], 2: [np.nan, 4]}, dtype=object),  # 创建另一个数据框，数据类型为对象
        ],
    )
    @pytest.mark.filterwarnings("ignore:Mismatched null-like values:FutureWarning")
    def test_stat_operators_attempt_obj_array(self, method, df, axis):
        # GH#676
        assert df.values.dtype == np.object_  # 断言数据框的值的数据类型为np.object_
        result = getattr(df, method)(axis=axis)  # 调用数据框的方法并传入指定的轴参数
        expected = getattr(df.astype("f8"), method)(axis=axis).astype(object)  # 以浮点数形式复制数据框，并调用方法，再转换为对象型
        if axis in [1, "columns"] and method in ["min", "max"]:  # 如果轴在[1, "columns"]中且方法在["min", "max"]中
            expected[expected.isna()] = None  # 将期望的结果中NaN值替换为None
        tm.assert_series_equal(result, expected)  # 使用测试工具断言结果和期望值相等

    @pytest.mark.parametrize("op", ["mean", "std", "var", "skew", "kurt", "sem"])
    def test_mixed_ops(self, op):
        # GH#16116
        df = DataFrame(  # 创建一个数据框，包含三列'int', 'float', 'str'
            {
                "int": [1, 2, 3, 4],
                "float": [1.0, 2.0, 3.0, 4.0],
                "str": ["a", "b", "c", "d"],
            }
        )
        msg = "|".join(  # 创建匹配的错误信息字符串
            [
                "Could not convert",
                "could not convert",
                "can't multiply sequence by non-int",
                "does not support",
            ]
        )
        with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获特定的TypeError异常，并匹配预期的错误信息
            getattr(df, op)()  # 调用数据框的方法

        with pd.option_context("use_bottleneck", False):  # 设置Pandas选项上下文，禁用Bottleneck优化
            msg = "|".join(  # 创建匹配的错误信息字符串
                [
                    "Could not convert",
                    "could not convert",
                    "can't multiply sequence by non-int",
                    "does not support",
                ]
            )
            with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获特定的TypeError异常，并匹配预期的错误信息
                getattr(df, op)()  # 调用数据框的方法

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="sum doesn't work for arrow strings"
    )
    def test_reduce_mixed_frame(self):
        # GH 6806
        df = DataFrame(  # 创建一个包含三列'bool_data', 'int_data', 'string_data'的数据框
            {
                "bool_data": [True, True, False, False, False],
                "int_data": [10, 20, 30, 40, 50],
                "string_data": ["a", "b", "c", "d", "e"],
            }
        )
        df.reindex(columns=["bool_data", "int_data", "string_data"])  # 重新索引数据框的列
        test = df.sum(axis=0)  # 计算每列的和
        tm.assert_numpy_array_equal(  # 使用测试工具断言numpy数组的相等性
            test.values, np.array([2, 150, "abcde"], dtype=object)  # 指定期望的numpy数组
        )
        alt = df.T.sum(axis=1)  # 计算数据框转置后每行的和
        tm.assert_series_equal(test, alt)  # 使用测试工具断言结果和备选值的序列相等
    # 定义一个测试方法，用于测试 DataFrame 的 nunique 方法
    def test_nunique(self):
        # 创建一个包含三列的 DataFrame，每列包含不同类型的数据
        df = DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": [1, np.nan, 3]})
        # 断言调用 df.nunique() 后返回的 Series 与给定的 Series 相等
        tm.assert_series_equal(df.nunique(), Series({"A": 1, "B": 3, "C": 2}))
        # 断言调用 df.nunique(dropna=False) 后返回的 Series 与给定的 Series 相等
        tm.assert_series_equal(
            df.nunique(dropna=False), Series({"A": 1, "B": 3, "C": 3})
        )
        # 断言调用 df.nunique(axis=1) 后返回的 Series 与给定的 Series 相等
        tm.assert_series_equal(df.nunique(axis=1), Series({0: 1, 1: 2, 2: 2}))
        # 断言调用 df.nunique(axis=1, dropna=False) 后返回的 Series 与给定的 Series 相等
        tm.assert_series_equal(
            df.nunique(axis=1, dropna=False), Series({0: 1, 1: 3, 2: 2})
        )

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_mean_mixed_datetime_numeric(self, tz):
        # 测试混合数据类型的平均值计算，包括日期时间和数值类型
        # 创建一个包含日期时间和数值数据的 DataFrame
        df = DataFrame({"A": [1, 1], "B": [Timestamp("2000", tz=tz)] * 2})
        # 计算 DataFrame 的平均值
        result = df.mean()
        # 创建一个期望的 Series 对象，用于比较计算结果
        expected = Series([1.0, Timestamp("2000", tz=tz)], index=["A", "B"])
        # 断言计算结果与期望值相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_mean_includes_datetimes(self, tz):
        # 测试包含日期时间的平均值计算
        # 创建一个包含日期时间数据的 DataFrame
        df = DataFrame({"A": [Timestamp("2000", tz=tz)] * 2})
        # 计算 DataFrame 的平均值
        result = df.mean()

        # 创建一个期望的 Series 对象，用于比较计算结果
        expected = Series([Timestamp("2000", tz=tz)], index=["A"])
        # 断言计算结果与期望值相等
        tm.assert_series_equal(result, expected)

    def test_mean_mixed_string_decimal(self):
        # 测试包含字符串和小数的平均值计算
        # 创建一个包含不同数据类型的数据集
        d = [
            {"A": 2, "B": None, "C": Decimal("628.00")},
            {"A": 1, "B": None, "C": Decimal("383.00")},
            {"A": 3, "B": None, "C": Decimal("651.00")},
            {"A": 2, "B": None, "C": Decimal("575.00")},
            {"A": 4, "B": None, "C": Decimal("1114.00")},
            {"A": 1, "B": "TEST", "C": Decimal("241.00")},
            {"A": 2, "B": None, "C": Decimal("572.00")},
            {"A": 4, "B": None, "C": Decimal("609.00")},
            {"A": 3, "B": None, "C": Decimal("820.00")},
            {"A": 5, "B": None, "C": Decimal("1223.00")},
        ]
        # 创建 DataFrame 对象
        df = DataFrame(d)

        # 使用 pytest 断言捕获 TypeError 异常
        with pytest.raises(
            TypeError, match="unsupported operand type|does not support"
        ):
            # 尝试计算整个 DataFrame 的平均值，预期会抛出异常
            df.mean()
        
        # 计算 DataFrame 中特定列（"A" 和 "C"）的平均值
        result = df[["A", "C"]].mean()
        # 创建一个期望的 Series 对象，用于比较计算结果
        expected = Series([2.7, 681.6], index=["A", "C"], dtype=object)
        # 断言计算结果与期望值相等
        tm.assert_series_equal(result, expected)
    # 对给定的 datetime_frame 计算标准差，使用自由度为4
    result = datetime_frame.std(ddof=4)
    # 期望值是对 datetime_frame 每列应用标准差计算的结果
    expected = datetime_frame.apply(lambda x: x.std(ddof=4))
    # 使用 pytest 的准确近似相等断言比较结果和期望值
    tm.assert_almost_equal(result, expected)

    # 对给定的 datetime_frame 计算方差，使用自由度为4
    result = datetime_frame.var(ddof=4)
    # 期望值是对 datetime_frame 每列应用方差计算的结果
    expected = datetime_frame.apply(lambda x: x.var(ddof=4))
    # 使用 pytest 的准确近似相等断言比较结果和期望值
    tm.assert_almost_equal(result, expected)

    # 创建一个形状为 (1, 1000) 的随机数组，并对其在第 0 轴上进行方差计算
    arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
    result = nanops.nanvar(arr, axis=0)
    # 断言结果数组中没有负数
    assert not (result < 0).any()

    # 在禁用 bottleneck 的上下文中再次计算 arr 的方差
    with pd.option_context("use_bottleneck", False):
        result = nanops.nanvar(arr, axis=0)
        # 断言结果数组中没有负数
        assert not (result < 0).any()

@pytest.mark.parametrize("meth", ["sem", "var", "std"])
def test_numeric_only_flag(self, meth):
    # GH 9201
    # 创建一个形状为 (5, 3) 的随机数据框 df1，并指定列名
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((5, 3)),
        columns=["foo", "bar", "baz"],
    )
    # 将 "foo" 列强制转换为对象类型，以避免后面将该条目设置为 "100" 时的隐式转换
    df1 = df1.astype({"foo": object})
    # 将 "foo" 列的一项设置为字符串格式的数字 "100"
    df1.loc[0, "foo"] = "100"

    # 创建另一个形状为 (5, 3) 的随机数据框 df2，并指定列名
    df2 = DataFrame(
        np.random.default_rng(2).standard_normal((5, 3)),
        columns=["foo", "bar", "baz"],
    )
    # 将 "foo" 列强制转换为对象类型，以避免后面将该条目设置为 "a" 时的隐式转换
    df2 = df2.astype({"foo": object})
    # 将 "foo" 列的一项设置为非数字字符串 "a"
    df2.loc[0, "foo"] = "a"

    # 使用 getattr 动态调用 df1 的 meth 方法（"sem", "var", "std" 中的一个），仅数值列参与计算
    result = getattr(df1, meth)(axis=1, numeric_only=True)
    # 期望值是对 df1 的 ["bar", "baz"] 列应用 meth 方法（"sem", "var", "std" 中的一个）后的结果
    expected = getattr(df1[["bar", "baz"]], meth)(axis=1)
    # 使用 pytest 的 Series 相等断言比较期望值和结果
    tm.assert_series_equal(expected, result)

    # 使用 getattr 动态调用 df2 的 meth 方法（"sem", "var", "std" 中的一个），仅数值列参与计算
    result = getattr(df2, meth)(axis=1, numeric_only=True)
    # 期望值是对 df2 的 ["bar", "baz"] 列应用 meth 方法（"sem", "var", "std" 中的一个）后的结果
    expected = getattr(df2[["bar", "baz"]], meth)(axis=1)
    # 使用 pytest 的 Series 相等断言比较期望值和结果
    tm.assert_series_equal(expected, result)

    # df1 中全为数字，df2 中包含字母
    # 断言抛出 TypeError 异常，并验证消息中包含指定文本
    msg = r"unsupported operand type\(s\) for -: 'float' and 'str'"
    with pytest.raises(TypeError, match=msg):
        getattr(df1, meth)(axis=1, numeric_only=False)
    # 断言抛出 TypeError 异常，并验证消息中包含指定文本
    msg = "could not convert string to float: 'a'"
    with pytest.raises(TypeError, match=msg):
        getattr(df2, meth)(axis=1, numeric_only=False)

def test_sem(self, datetime_frame):
    # 对给定的 datetime_frame 计算标准误差，使用自由度为4
    result = datetime_frame.sem(ddof=4)
    # 期望值是对 datetime_frame 每列应用标准差计算后再除以该列长度的结果
    expected = datetime_frame.apply(lambda x: x.std(ddof=4) / np.sqrt(len(x)))
    # 使用 pytest 的准确近似相等断言比较结果和期望值
    tm.assert_almost_equal(result, expected)

    # 创建一个形状为 (1, 1000) 的随机数组，并对其在第 0 轴上进行标准误差计算
    arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
    result = nanops.nansem(arr, axis=0)
    # 断言结果数组中没有负数
    assert not (result < 0).any()

    # 在禁用 bottleneck 的上下文中再次计算 arr 的标准误差
    with pd.option_context("use_bottleneck", False):
        result = nanops.nansem(arr, axis=0)
        # 断言结果数组中没有负数
        assert not (result < 0).any()
    # 使用 pytest 的 @parametrize 装饰器，为一个测试函数定义多组参数和预期输出
    @pytest.mark.parametrize(
        "dropna, expected",
        [
            (
                True,  # 当 dropna=True 时的预期输出字典
                {
                    "A": [12],  # 键 'A' 对应的值是列表 [12]
                    "B": [10.0],  # 键 'B' 对应的值是列表 [10.0]
                    "C": [1.0],  # 键 'C' 对应的值是列表 [1.0]
                    "D": ["a"],  # 键 'D' 对应的值是列表 ["a"]
                    "E": Categorical(["a"], categories=["a"]),  # 键 'E' 对应的值是 Categorical 对象
                    "F": DatetimeIndex(["2000-01-02"], dtype="M8[ns]"),  # 键 'F' 对应的值是 DatetimeIndex 对象
                    "G": to_timedelta(["1 days"]),  # 键 'G' 对应的值是时间差对象列表
                },
            ),
            (
                False,  # 当 dropna=False 时的预期输出字典
                {
                    "A": [12],  # 键 'A' 对应的值是列表 [12]
                    "B": [10.0],  # 键 'B' 对应的值是列表 [10.0]
                    "C": [np.nan],  # 键 'C' 对应的值是包含 np.nan 的列表
                    "D": np.array([np.nan], dtype=object),  # 键 'D' 对应的值是包含 np.nan 的数组
                    "E": Categorical([np.nan], categories=["a"]),  # 键 'E' 对应的值是 Categorical 对象
                    "F": DatetimeIndex([pd.NaT], dtype="M8[ns]"),  # 键 'F' 对应的值是 DatetimeIndex 对象
                    "G": to_timedelta([pd.NaT]),  # 键 'G' 对应的值是时间差对象列表
                },
            ),
            (
                True,  # 当 dropna=True 时的另一组预期输出字典
                {
                    "H": [8, 9, np.nan, np.nan],  # 键 'H' 对应的值是包含数值和 np.nan 的列表
                    "I": [8, 9, np.nan, np.nan],  # 键 'I' 对应的值是包含数值和 np.nan 的列表
                    "J": [1, np.nan, np.nan, np.nan],  # 键 'J' 对应的值是包含数值和 np.nan 的列表
                    "K": Categorical(["a", np.nan, np.nan, np.nan], categories=["a"]),  # 键 'K' 对应的值是 Categorical 对象
                    "L": DatetimeIndex(["2000-01-02", "NaT", "NaT", "NaT"], dtype="M8[ns]"),  # 键 'L' 对应的值是 DatetimeIndex 对象
                    "M": to_timedelta(["1 days", "nan", "nan", "nan"]),  # 键 'M' 对应的值是时间差对象列表
                    "N": [0, 1, 2, 3],  # 键 'N' 对应的值是列表 [0, 1, 2, 3]
                },
            ),
            (
                False,  # 当 dropna=False 时的另一组预期输出字典
                {
                    "H": [8, 9, np.nan, np.nan],  # 键 'H' 对应的值是包含数值和 np.nan 的列表
                    "I": [8, 9, np.nan, np.nan],  # 键 'I' 对应的值是包含数值和 np.nan 的列表
                    "J": [1, np.nan, np.nan, np.nan],  # 键 'J' 对应的值是包含数值和 np.nan 的列表
                    "K": Categorical([np.nan, "a", np.nan, np.nan], categories=["a"]),  # 键 'K' 对应的值是 Categorical 对象
                    "L": DatetimeIndex(["NaT", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"),  # 键 'L' 对应的值是 DatetimeIndex 对象
                    "M": to_timedelta(["nan", "1 days", "nan", "nan"]),  # 键 'M' 对应的值是时间差对象列表
                    "N": [0, 1, 2, 3],  # 键 'N' 对应的值是列表 [0, 1, 2, 3]
                },
            ),
        ],
    )
    # 定义测试函数，用于测试数据框的 mode 方法，检验 dropna 参数是否正常工作
    def test_mode_dropna(self, dropna, expected):
        # 创建包含各种数据类型的数据框 df
        df = DataFrame(
            {
                "A": [12, 12, 19, 11],
                "B": [10, 10, np.nan, 3],
                "C": [1, np.nan, np.nan, np.nan],
                "D": Series([np.nan, np.nan, "a", np.nan], dtype=object),
                "E": Categorical([np.nan, np.nan, "a", np.nan]),
                "F": DatetimeIndex(["NaT", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"),
                "G": to_timedelta(["1 days", "nan", "nan", "nan"]),
                "H": [8, 8, 9, 9],
                "I": [9, 9, 8, 8],
                "J": [1, 1, np.nan, np.nan],
                "K": Categorical(["a", np.nan, "a", np.nan]),
                "L": DatetimeIndex(
                    ["2000-01-02", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"
                ),
                "M": to_timedelta(["1 days", "nan", "1 days", "nan"]),
                "N": np.arange(4, dtype="int64"),
            }
        )

        # 使用数据框的 mode 方法计算众数，按照预期的列顺序排序结果
        result = df[sorted(expected.keys())].mode(dropna=dropna)
        # 将预期结果转换为数据框格式
        expected = DataFrame(expected)
        # 使用测试框架中的 assert_frame_equal 方法比较计算结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，验证 mode 方法在无法对结果进行排序时是否会引发警告
    def test_mode_sortwarning(self, using_infer_string):
        # 创建包含一列数据的数据框 df
        df = DataFrame({"A": [np.nan, np.nan, "a", "a"]})
        # 创建预期的数据框，包含按 A 列排序的结果
        expected = DataFrame({"A": ["a", np.nan]})

        # 根据 using_infer_string 参数判断是否期望引发警告
        warning = None if using_infer_string else UserWarning
        # 使用测试框架中的 assert_produces_warning 方法，检查是否引发指定警告类型，并匹配警告消息 "Unable to sort modes"
        with tm.assert_produces_warning(warning, match="Unable to sort modes"):
            # 调用数据框的 mode 方法计算众数
            result = df.mode(dropna=False)
            # 对结果按 A 列进行排序，并重设索引
            result = result.sort_values(by="A").reset_index(drop=True)

        # 使用测试框架中的 assert_frame_equal 方法比较计算结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，验证对空数据框调用 mode 方法的行为
    def test_mode_empty_df(self):
        # 创建一个空的数据框 df，指定列名为 ["a", "b"]
        df = DataFrame([], columns=["a", "b"])
        # 调用数据框的 mode 方法计算众数
        result = df.mode()
        # 创建一个预期的空数据框，列名为 ["a", "b"]，索引为空
        expected = DataFrame([], columns=["a", "b"], index=Index([], dtype=np.int64))
        # 使用测试框架中的 assert_frame_equal 方法比较计算结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)
    def test_operators_timedelta64(self):
        # 创建一个包含日期数据的DataFrame，列名为A、B、C
        df = DataFrame(
            {
                "A": date_range("2012-1-1", periods=3, freq="D"),  # 从2012年1月1日开始，生成3个日期，频率为每天
                "B": date_range("2012-1-2", periods=3, freq="D"),  # 从2012年1月2日开始，生成3个日期，频率为每天
                "C": Timestamp("20120101") - timedelta(minutes=5, seconds=5),  # 计算时间戳20120101减去5分钟5秒的时间差
            }
        )

        # 计算DataFrame列'A'和'C'的时间差，列'B'和'A'的时间差
        diffs = DataFrame({"A": df["A"] - df["C"], "B": df["A"] - df["B"]})

        # 求每列的最小值
        result = diffs.min()
        assert result.iloc[0] == diffs.loc[0, "A"]  # 断言第一行第一列的最小值等于'A'列的第一行值
        assert result.iloc[1] == diffs.loc[0, "B"]  # 断言第一行第二列的最小值等于'B'列的第一行值

        # 沿着列方向求最小值
        result = diffs.min(axis=1)
        assert (result == diffs.loc[0, "B"]).all()  # 断言结果的所有值都等于'B'列的第一行值

        # 求每列的最大值
        result = diffs.max()
        assert result.iloc[0] == diffs.loc[2, "A"]  # 断言最后一行第一列的最大值等于'A'列的最后一行值
        assert result.iloc[1] == diffs.loc[2, "B"]  # 断言最后一行第二列的最大值等于'B'列的最后一行值

        # 沿着列方向求最大值
        result = diffs.max(axis=1)
        assert (result == diffs["A"]).all()  # 断言结果的所有值都等于'A'列的值

        # 求绝对值
        result = diffs.abs()
        result2 = abs(diffs)
        expected = DataFrame({"A": df["A"] - df["C"], "B": df["B"] - df["A"]})  # 期望的DataFrame结果
        tm.assert_frame_equal(result, expected)  # 使用测试工具断言两个DataFrame相等
        tm.assert_frame_equal(result2, expected)  # 使用测试工具断言两个DataFrame相等

        # 混合数据帧
        mixed = diffs.copy()
        mixed["C"] = "foo"
        mixed["D"] = 1
        mixed["E"] = 1.0
        mixed["F"] = Timestamp("20130101")

        # 导致一个对象数组
        result = mixed.min()
        expected = Series(
            [
                pd.Timedelta(timedelta(seconds=5 * 60 + 5)),  # 时间差为5分5秒
                pd.Timedelta(timedelta(days=-1)),  # 时间差为-1天
                "foo",  # 字符串 "foo"
                1,  # 整数 1
                1.0,  # 浮点数 1.0
                Timestamp("20130101"),  # 时间戳为20130101
            ],
            index=mixed.columns,  # 索引为数据帧的列名
        )
        tm.assert_series_equal(result, expected)  # 使用测试工具断言两个Series相等

        # 排除非数值列
        result = mixed.min(axis=1, numeric_only=True)
        expected = Series([1, 1, 1.0], index=[0, 1, 2])  # 期望的Series结果
        tm.assert_series_equal(result, expected)  # 使用测试工具断言两个Series相等

        # 当仅选择这些列时工作
        result = mixed[["A", "B"]].min(axis=1)
        expected = Series([timedelta(days=-1)] * 3)  # 期望的Series结果，每个元素为时间差-1天，共3个元素
        tm.assert_series_equal(result, expected)  # 使用测试工具断言两个Series相等

        result = mixed[["A", "B"]].min()
        expected = Series(
            [timedelta(seconds=5 * 60 + 5), timedelta(days=-1)], index=["A", "B"]
        )  # 期望的Series结果，包含'A'和'B'列的最小时间差
        tm.assert_series_equal(result, expected)  # 使用测试工具断言两个Series相等

        # GH 3106
        df = DataFrame(
            {
                "time": date_range("20130102", periods=5),  # 生成从20130102开始的5个日期
                "time2": date_range("20130105", periods=5),  # 生成从20130105开始的5个日期
            }
        )
        df["off1"] = df["time2"] - df["time"]  # 计算时间差列off1
        assert df["off1"].dtype == "timedelta64[ns]"  # 断言时间差列的数据类型为纳秒级别的时间差

        df["off2"] = df["time"] - df["time2"]  # 计算时间差列off2
        df._consolidate_inplace()  # 在原地重新整理数据框
        assert df["off1"].dtype == "timedelta64[ns]"  # 断言时间差列off1的数据类型为纳秒级别的时间差
        assert df["off2"].dtype == "timedelta64[ns]"  # 断言时间差列off2的数据类型为纳秒级别的时间差
    # 定义测试方法，用于测试 pandas 中的标准差计算，skipna 参数设置为 False
    def test_std_timedelta64_skipna_false(self):
        # GH#37392: GitHub issue 跟踪号码，用于追踪相关问题
        # 创建一个时间间隔范围，从 "1 Day" 开始，包含 10 个周期
        tdi = pd.timedelta_range("1 Day", periods=10)
        # 创建一个包含两列的 DataFrame，每列都使用上述时间间隔范围 tdi，复制数据以保留原始数据的不变性
        df = DataFrame({"A": tdi, "B": tdi}, copy=True)
        # 将 DataFrame 中倒数第二行、倒数第一列的值设置为 pd.NaT（表示缺失的时间值）
        df.iloc[-2, -1] = pd.NaT

        # 计算 DataFrame 每列的标准差，skipna 参数为 False，即不跳过 NaN 值
        result = df.std(skipna=False)
        # 期望得到的标准差 Series，索引为 ["A", "B"]，数据类型为 timedelta64[ns]
        expected = Series(
            [df["A"].std(), pd.NaT], index=["A", "B"], dtype="timedelta64[ns]"
        )
        # 断言计算结果与期望值相等
        tm.assert_series_equal(result, expected)

        # 计算 DataFrame 每行的标准差，skipna 参数为 False
        result = df.std(axis=1, skipna=False)
        # 期望得到的标准差 Series，包含 8 个值为 pd.Timedelta(0)，最后两个值分别为 pd.NaT 和 pd.Timedelta(0)
        expected = Series([pd.Timedelta(0)] * 8 + [pd.NaT, pd.Timedelta(0)])
        # 断言计算结果与期望值相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values", [["2022-01-01", "2022-01-02", pd.NaT, "2022-01-03"], 4 * [pd.NaT]]
    )
    # 定义参数化测试，测试包含 NaN 值的 datetime64 数据的标准差计算
    def test_std_datetime64_with_nat(self, values, skipna, request, unit):
        # GH#51335: GitHub issue 跟踪号码，用于追踪相关问题
        # 将输入值转换为 datetime64 类型，并按照指定的 unit 进行单位转换
        dti = to_datetime(values).as_unit(unit)
        # 创建一个包含一列的 DataFrame，列名为 "a"，数据为上述转换后的 dti
        df = DataFrame({"a": dti})
        # 计算 DataFrame 的标准差，skipna 参数控制是否跳过 NaN 值
        result = df.std(skipna=skipna)
        # 根据 skipna 的值构建期望的标准差 Series
        if not skipna or all(value is pd.NaT for value in values):
            expected = Series({"a": pd.NaT}, dtype=f"timedelta64[{unit}]")
        else:
            # 如果 skipna 不为 False 且 values 不全为 pd.NaT，则期望的标准差为 1 天对应的纳秒数
            # 86400000000000ns == 1 day
            expected = Series({"a": 86400000000000}, dtype=f"timedelta64[{unit}]")
        # 断言计算结果与期望值相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试空 DataFrame 的求和情况
    def test_sum_corner(self):
        # 创建一个空的 DataFrame
        empty_frame = DataFrame()

        # 对空 DataFrame 沿 axis=0 求和，返回结果应为 Series 类型
        axis0 = empty_frame.sum(axis=0)
        # 对空 DataFrame 沿 axis=1 求和，返回结果应为 Series 类型
        axis1 = empty_frame.sum(axis=1)
        # 断言 axis0 和 axis1 的长度都为 0
        assert len(axis0) == 0
        assert len(axis1) == 0

    @pytest.mark.parametrize(
        "index",
        [
            RangeIndex(0),  # 测试使用 RangeIndex 创建空索引
            DatetimeIndex([]),  # 测试使用 DatetimeIndex 创建空索引
            Index([], dtype=np.int64),  # 测试使用 Index 创建空索引，数据类型为 int64
            Index([], dtype=np.float64),  # 测试使用 Index 创建空索引，数据类型为 float64
            DatetimeIndex([], freq="ME"),  # 测试使用 DatetimeIndex 创建空索引，频率为 "ME"
            PeriodIndex([], freq="D"),  # 测试使用 PeriodIndex 创建空索引，频率为 "D"
        ],
    )
    # 参数化测试方法，测试当 axis=1 时，对空 DataFrame 进行不同计算的行为
    def test_axis_1_empty(self, all_reductions, index):
        # 创建一个列为 "a"、索引为 index 的 DataFrame
        df = DataFrame(columns=["a"], index=index)
        # 调用 DataFrame 对象的 all_reductions 方法，axis=1 进行计算
        result = getattr(df, all_reductions)(axis=1)
        # 根据不同的 all_reductions 类型，确定期望结果的数据类型
        if all_reductions in ("any", "all"):
            expected_dtype = "bool"
        elif all_reductions == "count":
            expected_dtype = "int64"
        else:
            expected_dtype = "object"
        # 构建期望的 Series 结果，数据为空，索引为 index，数据类型为 expected_dtype
        expected = Series([], index=index, dtype=expected_dtype)
        # 断言计算结果与期望值相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method, unit", [("sum", 0), ("prod", 1)])
    @pytest.mark.parametrize("numeric_only", [None, True, False])
    # 定义一个测试函数，用于测试 DataFrame 对象的 sum 和 prod 方法在处理 NaN 值时的行为
    def test_sum_prod_nanops(self, method, unit, numeric_only):
        # 创建一个索引列表
        idx = ["a", "b", "c"]
        # 创建一个包含 NaN 值的 DataFrame 对象
        df = DataFrame({"a": [unit, unit], "b": [unit, np.nan], "c": [np.nan, np.nan]})
        # 调用指定的 DataFrame 方法，计算结果并赋值给 result
        result = getattr(df, method)(numeric_only=numeric_only)
        # 创建预期结果的 Series 对象，用于断言比较
        expected = Series([unit, unit, unit], index=idx, dtype="float64")
        # 断言结果与预期是否相等
        tm.assert_series_equal(result, expected)

        # 使用 min_count=1 参数进行测试
        result = getattr(df, method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, unit, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # 使用 min_count=0 参数进行测试
        result = getattr(df, method)(numeric_only=numeric_only, min_count=0)
        expected = Series([unit, unit, unit], index=idx, dtype="float64")
        tm.assert_series_equal(result, expected)

        # 对 df 的子集进行操作并进行测试
        result = getattr(df.iloc[1:], method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, np.nan, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # 使用 min_count 大于 1 进行测试
        df = DataFrame({"A": [unit] * 10, "B": [unit] * 5 + [np.nan] * 5})
        result = getattr(df, method)(numeric_only=numeric_only, min_count=5)
        expected = Series(result, index=["A", "B"])
        tm.assert_series_equal(result, expected)

        result = getattr(df, method)(numeric_only=numeric_only, min_count=6)
        expected = Series(result, index=["A", "B"])
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，用于测试 DataFrame 对象的 sum 方法在处理时间增量（timedelta）数据类型时的行为
    def test_sum_nanops_timedelta(self):
        # 创建一个索引列表
        idx = ["a", "b", "c"]
        # 创建一个包含时间增量数据的 DataFrame 对象
        df = DataFrame({"a": [0, 0], "b": [0, np.nan], "c": [np.nan, np.nan]})
        # 对 df 应用时间增量转换函数
        df2 = df.apply(to_timedelta)

        # 默认情况下的测试
        result = df2.sum()
        expected = Series([0, 0, 0], dtype="m8[ns]", index=idx)
        tm.assert_series_equal(result, expected)

        # 使用 min_count=0 参数进行测试
        result = df2.sum(min_count=0)
        tm.assert_series_equal(result, expected)

        # 使用 min_count=1 参数进行测试
        result = df2.sum(min_count=1)
        expected = Series([0, 0, np.nan], dtype="m8[ns]", index=idx)
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，用于测试 DataFrame 对象的 sum 方法在使用 min_count 参数时的行为
    def test_sum_nanops_min_count(self):
        # https://github.com/pandas-dev/pandas/issues/39738
        # 创建一个包含整数数据的 DataFrame 对象
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        # 使用 min_count=10 参数进行测试
        result = df.sum(min_count=10)
        expected = Series([np.nan, np.nan], index=["x", "y"])
        tm.assert_series_equal(result, expected)

    # 使用 pytest 的参数化装饰器，定义一个测试函数，测试不同浮点数类型和不同参数组合时的 sum 方法行为
    @pytest.mark.parametrize("float_type", ["float16", "float32", "float64"])
    @pytest.mark.parametrize(
        "kwargs, expected_result",
        [
            ({"axis": 1, "min_count": 2}, [3.2, 5.3, np.nan]),
            ({"axis": 1, "min_count": 3}, [np.nan, np.nan, np.nan]),
            ({"axis": 1, "skipna": False}, [3.2, 5.3, np.nan]),
        ],
    )
    def test_sum_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
        # GH#46947
        # 创建一个包含两列的 DataFrame，其中包括浮点数和 NaN 值
        df = DataFrame({"a": [1.0, 2.3, 4.4], "b": [2.2, 3, np.nan]}, dtype=float_type)
        # 对 DataFrame 进行 sum 操作，使用给定的 kwargs 参数
        result = df.sum(**kwargs)
        # 创建一个预期的 Series 结果，将其转换为指定的浮点数类型
        expected = Series(expected_result).astype(float_type)
        # 使用 pytest 中的 assert 函数检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("float_type", ["float16", "float32", "float64"])
    @pytest.mark.parametrize(
        "kwargs, expected_result",
        [
            ({"axis": 1, "min_count": 2}, [2.0, 4.0, np.nan]),
            ({"axis": 1, "min_count": 3}, [np.nan, np.nan, np.nan]),
            ({"axis": 1, "skipna": False}, [2.0, 4.0, np.nan]),
        ],
    )
    def test_prod_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
        # GH#46947
        # 创建一个包含两列的 DataFrame，其中包括浮点数和 NaN 值
        df = DataFrame(
            {"a": [1.0, 2.0, 4.4], "b": [2.0, 2.0, np.nan]}, dtype=float_type
        )
        # 对 DataFrame 进行 prod 操作，使用给定的 kwargs 参数
        result = df.prod(**kwargs)
        # 创建一个预期的 Series 结果，将其转换为指定的浮点数类型
        expected = Series(expected_result).astype(float_type)
        # 使用 pytest 中的 assert 函数检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_sum_object(self, float_frame):
        # 将 float_frame 中的值转换为整数类型，并创建一个新的 DataFrame
        values = float_frame.values.astype(int)
        frame = DataFrame(values, index=float_frame.index, columns=float_frame.columns)
        # 将 frame 中的值乘以 timedelta(1)，得到时间间隔的数据
        deltas = frame * timedelta(1)
        # 对 deltas 中的值进行 sum 操作
        deltas.sum()

    def test_sum_bool(self, float_frame):
        # 确保以下操作可行，这是一个 bug 报告
        # 检查 float_frame 中的 NaN 值，生成一个布尔值的 DataFrame
        bools = np.isnan(float_frame)
        # 对 axis=1 方向上的布尔值进行 sum 操作
        bools.sum(axis=1)
        # 对 axis=0 方向上的布尔值进行 sum 操作
        bools.sum(axis=0)

    def test_sum_mixed_datetime(self):
        # GH#30886
        # 创建一个包含日期时间和整数的 DataFrame，并重新索引
        df = DataFrame({"A": date_range("2000", periods=4), "B": [1, 2, 3, 4]}).reindex(
            [2, 3, 4]
        )
        # 使用 pytest 的 assert 函数，验证对 df 执行 sum 操作是否会引发 TypeError 异常
        with pytest.raises(TypeError, match="does not support operation 'sum'"):
            df.sum()

    def test_mean_corner(self, float_frame, float_string_frame):
        # 对具有对象数据的单元测试
        msg = "Could not convert|does not support"
        # 使用 pytest 的 assert 函数，验证对 float_string_frame 执行 mean 操作是否会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            float_string_frame.mean(axis=0)

        # 对混合类型的 xs 进行 mean 操作，仅仅希望确认它能够工作
        with pytest.raises(TypeError, match="unsupported operand type"):
            float_string_frame.mean(axis=1)

        # 对布尔列进行均值计算
        float_frame["bool"] = float_frame["A"] > 0
        means = float_frame.mean(axis=0)
        # 使用 assert 断言，验证 means 中的 bool 列均值是否与 float_frame["bool"] 的值均值相等
        assert means["bool"] == float_frame["bool"].values.mean()
    def test_mean_datetimelike(self):
        # GH#24757 检查默认情况下datetimelike是否被排除，numeric_only=True时是否正确处理
        # 截至2.0版本，datetimelike在numeric_only=None时*不*被排除

        # 创建包含不同类型数据的DataFrame
        df = DataFrame(
            {
                "A": np.arange(3),
                "B": date_range("2016-01-01", periods=3),
                "C": pd.timedelta_range("1D", periods=3),
                "D": pd.period_range("2016", periods=3, freq="Y"),
            }
        )
        # 计算DataFrame每列的均值，仅计算数值类型列
        result = df.mean(numeric_only=True)
        # 期望的均值结果，这里只关注"A"列的值
        expected = Series({"A": 1.0})
        # 使用测试框架检查结果是否符合预期
        tm.assert_series_equal(result, expected)

        # 使用pytest检查对于PeriodArray是否正确抛出TypeError异常
        with pytest.raises(TypeError, match="mean is not implemented for PeriodArray"):
            df.mean()

    def test_mean_datetimelike_numeric_only_false(self):
        # 创建包含不同类型数据的DataFrame
        df = DataFrame(
            {
                "A": np.arange(3),
                "B": date_range("2016-01-01", periods=3),
                "C": pd.timedelta_range("1D", periods=3),
            }
        )

        # 对于datetime和timedelta类型，numeric_only=False时仍然可以计算均值
        result = df.mean(numeric_only=False)
        # 期望的均值结果，包括"A"列的数值，"B"列和"C"列的特定值
        expected = Series({"A": 1, "B": df.loc[1, "B"], "C": df.loc[1, "C"]})
        # 使用测试框架检查结果是否符合预期
        tm.assert_series_equal(result, expected)

        # 添加一个Period列，检查是否正确抛出TypeError异常
        df["D"] = pd.period_range("2016", periods=3, freq="Y")

        with pytest.raises(TypeError, match="mean is not implemented for Period"):
            df.mean(numeric_only=False)

    def test_mean_extensionarray_numeric_only_true(self):
        # https://github.com/pandas-dev/pandas/issues/33256
        # 创建一个包含随机整数的DataFrame
        arr = np.random.default_rng(2).integers(1000, size=(10, 5))
        df = DataFrame(arr, dtype="Int64")
        # 计算DataFrame每列的均值，仅计算数值类型列
        result = df.mean(numeric_only=True)
        # 期望的均值结果，转换为Float64类型以保持一致性
        expected = DataFrame(arr).mean().astype("Float64")
        # 使用测试框架检查结果是否符合预期
        tm.assert_series_equal(result, expected)

    def test_stats_mixed_type(self, float_string_frame):
        # 使用pytest检查是否正确抛出TypeError异常，匹配给定的错误信息
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.std(axis=1)
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.var(axis=1)
        with pytest.raises(TypeError, match="unsupported operand type"):
            float_string_frame.mean(axis=1)
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.skew(axis=1)

    def test_sum_bools(self):
        # 创建一个DataFrame，索引为0，列数为10
        df = DataFrame(index=range(1), columns=range(10))
        # 计算DataFrame中每行中True值的总数，预期结果为10
        bools = isna(df)
        assert bools.sum(axis=1)[0] == 10

    # ----------------------------------------------------------------------
    # Index of max / min

    @pytest.mark.parametrize("axis", [0, 1])
    def test_idxmin(self, float_frame, int_frame, skipna, axis):
        # 使用 float_frame 进行测试
        frame = float_frame
        # 将第5到第10行的所有列设置为 NaN
        frame.iloc[5:10] = np.nan
        # 将第15到第20行的倒数第2列和倒数第1列设置为 NaN
        frame.iloc[15:20, -2:] = np.nan
        # 遍历两个数据框：float_frame 和 int_frame
        for df in [frame, int_frame]:
            # 如果不跳过 NaN 值或者操作轴为1，并且当前 df 不是 int_frame
            if (not skipna or axis == 1) and df is not int_frame:
                # 如果要求跳过 NaN 值，则设置错误信息为 "Encountered all NA values"
                if skipna:
                    msg = "Encountered all NA values"
                # 否则，设置错误信息为 "Encountered an NA value"
                else:
                    msg = "Encountered an NA value"
                # 使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误信息
                with pytest.raises(ValueError, match=msg):
                    df.idxmin(axis=axis, skipna=skipna)
                # 再次使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误信息
                with pytest.raises(ValueError, match=msg):
                    df.idxmin(axis=axis, skipna=skipna)
            else:
                # 计算 df 在指定轴上的最小值索引
                result = df.idxmin(axis=axis, skipna=skipna)
                # 期望的结果是应用 Series.idxmin 函数后的值
                expected = df.apply(Series.idxmin, axis=axis, skipna=skipna)
                # 将期望的结果转换为与 df 索引相同的数据类型
                expected = expected.astype(df.index.dtype)
                # 使用 tm.assert_series_equal 检查结果与期望是否相等
                tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_idxmin_empty(self, index, skipna, axis):
        # GH53265
        # 如果轴为0，则创建一个只有索引的 DataFrame
        if axis == 0:
            frame = DataFrame(index=index)
        # 否则，创建一个列为索引的空 DataFrame
        else:
            frame = DataFrame(columns=index)

        # 计算 DataFrame 在指定轴上的最小值索引
        result = frame.idxmin(axis=axis, skipna=skipna)
        # 创建一个预期结果为指定索引数据类型的 Series
        expected = Series(dtype=index.dtype)
        # 使用 tm.assert_series_equal 检查结果与期望是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_idxmin_numeric_only(self, numeric_only):
        # 创建一个 DataFrame 包含三列，每列包含的数据不同
        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1], "c": list("xyx")})
        # 计算 DataFrame 在指定条件下的最小值索引
        result = df.idxmin(numeric_only=numeric_only)
        # 如果 numeric_only 为 True，则预期结果为包含两个索引的 Series
        if numeric_only:
            expected = Series([2, 1], index=["a", "b"])
        # 否则，预期结果为包含三个索引的 Series
        else:
            expected = Series([2, 1, 0], index=["a", "b", "c"])
        # 使用 tm.assert_series_equal 检查结果与期望是否相等
        tm.assert_series_equal(result, expected)

    def test_idxmin_axis_2(self, float_frame):
        # 使用 float_frame 进行测试
        frame = float_frame
        # 设置错误信息为 "No axis named 2 for object type DataFrame"
        msg = "No axis named 2 for object type DataFrame"
        # 使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误信息
        with pytest.raises(ValueError, match=msg):
            frame.idxmin(axis=2)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_idxmax(self, float_frame, int_frame, skipna, axis):
        # 使用 float_frame 进行测试
        frame = float_frame
        # 将第5到第10行的所有列设置为 NaN
        frame.iloc[5:10] = np.nan
        # 将第15到第20行的倒数第2列和倒数第1列设置为 NaN
        frame.iloc[15:20, -2:] = np.nan
        # 遍历两个数据框：float_frame 和 int_frame
        for df in [frame, int_frame]:
            # 如果不跳过 NaN 值或者操作轴为1，并且当前 df 是 frame
            if (skipna is False or axis == 1) and df is frame:
                # 如果不跳过 NaN 值，则设置错误信息为 "Encountered all NA values"
                if skipna:
                    msg = "Encountered all NA values"
                # 否则，设置错误信息为 "Encountered an NA value"
                else:
                    msg = "Encountered an NA value"
                # 使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误信息
                with pytest.raises(ValueError, match=msg):
                    df.idxmax(axis=axis, skipna=skipna)
                # 直接返回，不再执行后续代码
                return

            # 计算 df 在指定轴上的最大值索引
            result = df.idxmax(axis=axis, skipna=skipna)
            # 期望的结果是应用 Series.idxmax 函数后的值
            expected = df.apply(Series.idxmax, axis=axis, skipna=skipna)
            # 将期望的结果转换为与 df 索引相同的数据类型
            expected = expected.astype(df.index.dtype)
            # 使用 tm.assert_series_equal 检查结果与期望是否相等
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    # 使用 pytest.mark.filterwarnings 注册一个测试标记，用于忽略特定的警告信息
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 定义一个测试函数 test_idxmax_empty，接受 index、skipna、axis 三个参数
    def test_idxmax_empty(self, index, skipna, axis):
        # GH53265：标识 GitHub 上的 issue 53265
        # 根据 axis 参数的不同，创建一个空的 DataFrame 对象 frame
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)
    
        # 调用 DataFrame 对象的 idxmax 方法，计算最大值的索引，根据 skipna 参数确定是否跳过 NaN 值
        result = frame.idxmax(axis=axis, skipna=skipna)
        # 创建一个预期结果的 Series 对象 expected，数据类型根据 index 的数据类型决定
        expected = Series(dtype=index.dtype)
        # 使用 pytest 的断言方法 assert_series_equal 检查结果和预期是否相等
        tm.assert_series_equal(result, expected)
    
    # 使用 pytest.mark.parametrize 标记，参数化测试函数 test_idxmax_numeric_only 中的 numeric_only 参数
    @pytest.mark.parametrize("numeric_only", [True, False])
    # 定义一个测试函数 test_idxmax_numeric_only，接受 numeric_only 参数
    def test_idxmax_numeric_only(self, numeric_only):
        # 创建一个包含指定数据的 DataFrame 对象 df
        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1], "c": list("xyx")})
        # 调用 DataFrame 对象的 idxmax 方法，计算最大值的索引，根据 numeric_only 参数确定是否仅对数值列进行计算
        result = df.idxmax(numeric_only=numeric_only)
        # 根据 numeric_only 参数选择不同的预期结果 expected，以 Series 对象返回
        if numeric_only:
            expected = Series([1, 0], index=["a", "b"])
        else:
            expected = Series([1, 0, 1], index=["a", "b", "c"])
        # 使用 pytest 的断言方法 assert_series_equal 检查结果和预期是否相等
        tm.assert_series_equal(result, expected)
    
    # 定义一个测试函数 test_idxmax_arrow_types，不接受任何参数
    def test_idxmax_arrow_types(self):
        # GH#55368：标识 GitHub 上的 issue 55368
        # 导入 pyarrow 模块，如果未安装，则跳过该测试
        pytest.importorskip("pyarrow")
    
        # 创建一个包含指定数据和类型的 DataFrame 对象 df
        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1]}, dtype="int64[pyarrow]")
        # 调用 DataFrame 对象的 idxmax 方法，计算最大值的索引
        result = df.idxmax()
        # 创建一个预期结果的 Series 对象 expected
        expected = Series([1, 0], index=["a", "b"])
        # 使用 pytest 的断言方法 assert_series_equal 检查结果和预期是否相等
        tm.assert_series_equal(result, expected)
    
        # 调用 DataFrame 对象的 idxmin 方法，计算最小值的索引
        result = df.idxmin()
        # 创建一个预期结果的 Series 对象 expected
        expected = Series([2, 1], index=["a", "b"])
        # 使用 pytest 的断言方法 assert_series_equal 检查结果和预期是否相等
        tm.assert_series_equal(result, expected)
    
        # 创建一个包含指定数据和类型的 DataFrame 对象 df
        df = DataFrame({"a": ["b", "c", "a"]}, dtype="string[pyarrow]")
        # 调用 DataFrame 对象的 idxmax 方法，计算最大值的索引，numeric_only=False 表示考虑所有列
        result = df.idxmax(numeric_only=False)
        # 创建一个预期结果的 Series 对象 expected
        expected = Series([1], index=["a"])
        # 使用 pytest 的断言方法 assert_series_equal 检查结果和预期是否相等
        tm.assert_series_equal(result, expected)
    
        # 调用 DataFrame 对象的 idxmin 方法，计算最小值的索引，numeric_only=False 表示考虑所有列
        result = df.idxmin(numeric_only=False)
        # 创建一个预期结果的 Series 对象 expected
        expected = Series([2], index=["a"])
        # 使用 pytest 的断言方法 assert_series_equal 检查结果和预期是否相等
        tm.assert_series_equal(result, expected)
    
    # 使用 pytest.mark.parametrize 标记，参数化测试函数 test_idxmax_axis_2 中的 float_frame 参数
    @pytest.mark.parametrize("float_frame", [DataFrame([[1.0, 2.0], [3.0, 4.0]])])
    # 定义一个测试函数 test_idxmax_axis_2，接受 float_frame 参数
    def test_idxmax_axis_2(self, float_frame):
        # 将参数 float_frame 赋值给局部变量 frame
        frame = float_frame
        # 设置错误消息 msg
        msg = "No axis named 2 for object type DataFrame"
        # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误消息 msg
        with pytest.raises(ValueError, match=msg):
            # 调用 DataFrame 对象的 idxmax 方法，指定 axis=2，引发 ValueError 异常
            frame.idxmax(axis=2)
    # 测试在混合数据类型中使用 idxmax 方法
    def test_idxmax_mixed_dtype(self):
        # 创建一个日期范围对象，从"2016-01-01"开始，包含3个日期
        dti = date_range("2016-01-01", periods=3)
        # 创建一个DataFrame，包含三列数据：第一列是 [0, 2, 1]，第二列是 [2, 1, 0]，第三列是日期范围对象dti
        df = DataFrame({1: [0, 2, 1], 2: range(3)[::-1], 3: dti})

        # 计算DataFrame每列的最大值所在的索引
        result = df.idxmax()
        # 创建一个期望的Series，包含预期的最大值索引和对应的索引值
        expected = Series([1, 0, 2], index=[1, 2, 3])
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 计算DataFrame每列的最小值所在的索引
        result = df.idxmin()
        # 创建一个期望的Series，包含预期的最小值索引和对应的索引值
        expected = Series([0, 2, 0], index=[1, 2, 3])
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 将第一行第三列的值设置为NaT（Not a Time）
        df.loc[0, 3] = pd.NaT
        # 计算DataFrame每列的最大值所在的索引
        result = df.idxmax()
        # 创建一个期望的Series，包含预期的最大值索引和对应的索引值
        expected = Series([1, 0, 2], index=[1, 2, 3])
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 计算DataFrame每列的最小值所在的索引
        result = df.idxmin()
        # 创建一个期望的Series，包含预期的最小值索引和对应的索引值
        expected = Series([0, 2, 1], index=[1, 2, 3])
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 添加一个新的列，其值为日期范围对象dti的倒序
        df[4] = dti[::-1]
        # 对DataFrame进行就地合并
        df._consolidate_inplace()

        # 计算DataFrame每列的最大值所在的索引
        result = df.idxmax()
        # 创建一个期望的Series，包含预期的最大值索引和对应的索引值
        expected = Series([1, 0, 2, 0], index=[1, 2, 3, 4])
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 计算DataFrame每列的最小值所在的索引
        result = df.idxmin()
        # 创建一个期望的Series，包含预期的最小值索引和对应的索引值
        expected = Series([0, 2, 1, 2], index=[1, 2, 3, 4])
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op, expected_value",
        [("idxmax", [0, 4]), ("idxmin", [0, 5])],
    )
    # 测试在转换数据类型时使用idxmax和idxmin方法
    def test_idxmax_idxmin_convert_dtypes(self, op, expected_value):
        # 创建一个包含两列的DataFrame，第一列是ID，包含多个相同的值，第二列是value
        df = DataFrame(
            {
                "ID": [100, 100, 100, 200, 200, 200],
                "value": [0, 0, 0, 1, 2, 0],
            },
            dtype="Int64",
        )
        # 按照ID分组DataFrame
        df = df.groupby("ID")

        # 根据op指定的操作，计算每组DataFrame的最大值或最小值索引
        result = getattr(df, op)()
        # 创建一个期望的DataFrame，包含预期的最大值或最小值索引和对应的索引名
        expected = DataFrame(
            {"value": expected_value},
            index=Index([100, 200], name="ID", dtype="Int64"),
        )
        # 断言计算结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试在DataFrame的多列Datetime64块中使用idxmax方法
    def test_idxmax_dt64_multicolumn_axis1(self):
        # 创建一个包含两列的DataFrame，第一列和第二列是相同的日期范围对象dti
        dti = date_range("2016-01-01", periods=3)
        df = DataFrame({3: dti, 4: dti[::-1]}, copy=True)
        # 将第一行第一列的值设置为NaT（Not a Time）
        df.iloc[0, 0] = pd.NaT

        # 对DataFrame进行就地合并
        df._consolidate_inplace()

        # 计算DataFrame每行的最大值所在的索引
        result = df.idxmax(axis=1)
        # 创建一个期望的Series，包含预期的最大值索引和对应的行索引
        expected = Series([4, 3, 3])
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 计算DataFrame每行的最小值所在的索引
        result = df.idxmin(axis=1)
        # 创建一个期望的Series，包含预期的最小值索引和对应的行索引
        expected = Series([4, 3, 4])
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

    # ----------------------------------------------------------------------
    # 逻辑约简

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("bool_only", [False, True])
    # 测试在包含浮点数和字符串的混合类型DataFrame上应用any和all操作
    def test_any_all_mixed_float(
        self, all_boolean_reductions, axis, bool_only, float_string_frame
    ):
        # 确保操作可以在混合类型的DataFrame上正常工作
        mixed = float_string_frame
        # 添加一个名为"_bool_"的列，其值为根据条件生成的随机布尔值
        mixed["_bool_"] = np.random.default_rng(2).standard_normal(len(mixed)) > 0.5

        # 调用DataFrame的逻辑约简函数，根据参数axis和bool_only指定的条件进行操作
        getattr(mixed, all_boolean_reductions)(axis=axis, bool_only=bool_only)

    @pytest.mark.parametrize("axis", [0, 1])
    # 定义测试方法，用于测试带有 NaN 的布尔类型 DataFrame 的各种布尔归约操作
    def test_any_all_bool_with_na(
        self, all_boolean_reductions, axis, bool_frame_with_na
    ):
        # 调用 DataFrame 对象的指定布尔归约方法，传入 axis 参数和 bool_only=False 参数
        getattr(bool_frame_with_na, all_boolean_reductions)(axis=axis, bool_only=False)

    # 定义测试方法，测试带有 NaN 的布尔类型 DataFrame 的布尔归约操作
    def test_any_all_bool_frame(self, all_boolean_reductions, bool_frame_with_na):
        # GH#12863: numpy gives back non-boolean data for object type
        # 对于对象类型，NumPy 返回非布尔数据，因此使用 fillna(True) 填充 NaN，以便与 pandas 行为进行比较
        frame = bool_frame_with_na.fillna(True)
        # 获取 numpy 库中对应的布尔归约函数
        alternative = getattr(np, all_boolean_reductions)
        # 获取 DataFrame 对象中对应的布尔归约方法
        f = getattr(frame, all_boolean_reductions)

        # 定义用于跳过 NaN 值的包装函数
        def skipna_wrapper(x):
            nona = x.dropna().values
            return alternative(nona)

        # 定义一般情况下的包装函数
        def wrapper(x):
            return alternative(x.values)

        # 计算在 axis=0 和 axis=1 上的布尔归约结果，跳过 NaN 值
        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)

        # 断言结果与 DataFrame 应用包装函数后的结果相等
        tm.assert_series_equal(result0, frame.apply(wrapper))
        tm.assert_series_equal(result1, frame.apply(wrapper, axis=1))

        # 计算在 axis=0 和 axis=1 上的布尔归约结果，不跳过 NaN 值
        result0 = f(axis=0)
        result1 = f(axis=1)

        # 断言结果与 DataFrame 应用 skipna_wrapper 后的结果相等，忽略数据类型检查
        tm.assert_series_equal(result0, frame.apply(skipna_wrapper))
        tm.assert_series_equal(
            result1, frame.apply(skipna_wrapper, axis=1), check_dtype=False
        )

        # 测试错误的 axis 参数情况
        with pytest.raises(ValueError, match="No axis named 2"):
            f(axis=2)

        # 测试全为 NA 的情况
        all_na = frame * np.nan
        r0 = getattr(all_na, all_boolean_reductions)(axis=0)
        r1 = getattr(all_na, all_boolean_reductions)(axis=1)
        if all_boolean_reductions == "any":
            # 断言结果 r0 和 r1 均为 False
            assert not r0.any()
            assert not r1.any()
        else:
            # 断言结果 r0 和 r1 均为 True
            assert r0.all()
            assert r1.all()

    # 定义额外的测试方法，测试 DataFrame 的 any 和 all 方法
    def test_any_all_extra(self):
        # 创建 DataFrame 对象 df
        df = DataFrame(
            {
                "A": [True, False, False],
                "B": [True, True, False],
                "C": [True, True, True],
            },
            index=["a", "b", "c"],
        )
        # 计算 df 中选择列 "A" 和 "B" 的 any 结果，期望得到的结果是 Series 对象 expected
        result = df[["A", "B"]].any(axis=1)
        expected = Series([True, True, False], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        # 计算 df 中选择列 "A" 和 "B" 的 any 结果，设置 bool_only=True
        result = df[["A", "B"]].any(axis=1, bool_only=True)
        tm.assert_series_equal(result, expected)

        # 计算 df 中所有元素在 axis=1 上的 all 结果
        result = df.all(axis=1)
        expected = Series([True, False, False], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        # 计算 df 中所有元素在 axis=1 上的 all 结果，设置 bool_only=True
        result = df.all(axis=1, bool_only=True)
        tm.assert_series_equal(result, expected)

        # 测试 axis=None 的情况
        result = df.all(axis=None).item()
        assert result is False

        result = df.any(axis=None).item()
        assert result is True

        result = df[["C"]].all(axis=None).item()
        assert result is True

    # 使用 pytest 的参数化功能定义测试方法，测试对象类型的 any 和 all 方法
    @pytest.mark.parametrize("axis", [0, 1])
    def test_any_all_object_dtype(
        self, axis, all_boolean_reductions, skipna, using_infer_string
    ):
        # GH#35450
        df = DataFrame(
            data=[
                [1, np.nan, np.nan, True],  # 创建一个包含空值和布尔值的DataFrame
                [np.nan, 2, np.nan, True],
                [np.nan, np.nan, np.nan, True],
                [np.nan, np.nan, "5", np.nan],
            ]
        )
        if using_infer_string:
            # 如果使用推断字符串类型，则设定特定的布尔值变量
            val = not axis == 0 and not skipna and all_boolean_reductions == "all"
        else:
            # 否则，将val设为True
            val = True
        # 调用DataFrame对象的布尔操作函数，根据指定的轴和参数进行操作
        result = getattr(df, all_boolean_reductions)(axis=axis, skipna=skipna)
        # 创建预期的结果Series对象，包含True值和val的值
        expected = Series([True, True, val, True])
        # 断言结果Series与预期的Series相等
        tm.assert_series_equal(result, expected)

    def test_any_datetime(self):
        # GH 23070
        float_data = [1, np.nan, 3, np.nan]
        datetime_data = [
            Timestamp("1960-02-15"),  # 创建Timestamp对象
            Timestamp("1960-02-16"),
            pd.NaT,  # 创建pandas NaT对象
            pd.NaT,
        ]
        # 创建包含浮点数和日期时间数据的DataFrame对象
        df = DataFrame({"A": float_data, "B": datetime_data})

        # 定义错误消息
        msg = "datetime64 type does not support operation 'any'"
        # 使用pytest断言，期望引发特定类型错误并包含指定消息
        with pytest.raises(TypeError, match=msg):
            df.any(axis=1)

    def test_any_all_bool_only(self):
        # GH 25101
        # 创建包含特定列和空值的DataFrame对象，使用dtype指定列的数据类型
        df = DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [None, None, None]},
            columns=Index(["col1", "col2", "col3"], dtype=object),
        )

        # 调用DataFrame对象的布尔操作函数，只考虑布尔值列，并生成结果Series对象
        result = df.all(bool_only=True)
        # 创建预期的结果Series对象，指定数据类型为布尔值
        expected = Series(dtype=np.bool_, index=[])
        # 断言结果Series与预期的Series相等
        tm.assert_series_equal(result, expected)

        # 创建包含特定列和布尔值的DataFrame对象
        df = DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [None, None, None],
                "col4": [False, False, True],  # 包含布尔值列
            }
        )

        # 调用DataFrame对象的布尔操作函数，只考虑布尔值列，并生成结果Series对象
        result = df.all(bool_only=True)
        # 创建预期的结果Series对象，包含布尔值为False的列
        expected = Series({"col4": False})
        # 断言结果Series与预期的Series相等
        tm.assert_series_equal(result, expected)
    # 使用 pytest.mark.parametrize 装饰器标记测试参数化，用于多组输入数据的测试
    @pytest.mark.parametrize(
        "func, data, expected",
        [
            # 测试 np.any 函数，传入空字典，期望结果为 False
            (np.any, {}, False),
            # 测试 np.all 函数，传入空字典，期望结果为 True
            (np.all, {}, True),
            # 测试 np.any 函数，传入包含空列表的字典，期望结果为 False
            (np.any, {"A": []}, False),
            # 测试 np.all 函数，传入包含空列表的字典，期望结果为 True
            (np.all, {"A": []}, True),
            # 测试 np.any 函数，传入包含两个 False 值的列表，期望结果为 False
            (np.any, {"A": [False, False]}, False),
            # 测试 np.all 函数，传入包含两个 False 值的列表，期望结果为 False
            (np.all, {"A": [False, False]}, False),
            # 测试 np.any 函数，传入包含一个 True 和一个 False 的列表，期望结果为 True
            (np.any, {"A": [True, False]}, True),
            # 测试 np.all 函数，传入包含一个 True 和一个 False 的列表，期望结果为 False
            (np.all, {"A": [True, False]}, False),
            # 测试 np.any 函数，传入包含两个 True 值的列表，期望结果为 True
            (np.any, {"A": [True, True]}, True),
            # 测试 np.all 函数，传入包含两个 True 值的列表，期望结果为 True
            (np.all, {"A": [True, True]}, True),
            # 测试 np.any 函数，传入包含一个 False 的 A 列和一个 False 的 B 列，期望结果为 False
            (np.any, {"A": [False], "B": [False]}, False),
            # 测试 np.all 函数，传入包含一个 False 的 A 列和一个 False 的 B 列，期望结果为 False
            (np.all, {"A": [False], "B": [False]}, False),
            # 测试 np.any 函数，传入包含两个 False 的 A 列和一个 False 一个 True 的 B 列，期望结果为 True
            (np.any, {"A": [False, False], "B": [False, True]}, True),
            # 测试 np.all 函数，传入包含两个 False 的 A 列和一个 False 一个 True 的 B 列，期望结果为 False
            (np.all, {"A": [False, False], "B": [False, True]}, False),
            # 测试 np.all 函数，传入包含浮点数序列的 Series 类型数据，期望结果为 False
            (np.all, {"A": Series([0.0, 1.0], dtype="float")}, False),
            # 测试 np.any 函数，传入包含浮点数序列的 Series 类型数据，期望结果为 True
            (np.any, {"A": Series([0.0, 1.0], dtype="float")}, True),
            # 测试 np.all 函数，传入包含整数序列的 Series 类型数据，期望结果为 False
            (np.all, {"A": Series([0, 1], dtype=int)}, False),
            # 测试 np.any 函数，传入包含整数序列的 Series 类型数据，期望结果为 True
            (np.any, {"A": Series([0, 1], dtype=int)}, True),
            # 测试 np.all 函数，传入包含 datetime 类型序列的 Series 数据，期望结果为 False
            pytest.param(np.all, {"A": Series([0, 1], dtype="M8[ns]")}, False),
            # 测试 np.all 函数，传入包含带时区的 datetime 类型序列的 Series 数据，期望结果为 False
            pytest.param(np.all, {"A": Series([0, 1], dtype="M8[ns, UTC]")}, False),
            # 测试 np.any 函数，传入包含 datetime 类型序列的 Series 数据，期望结果为 True
            pytest.param(np.any, {"A": Series([0, 1], dtype="M8[ns]")}, True),
            # 测试 np.any 函数，传入包含带时区的 datetime 类型序列的 Series 数据，期望结果为 True
            pytest.param(np.any, {"A": Series([0, 1], dtype="M8[ns, UTC]")}, True),
            # 测试 np.all 函数，传入包含整数序列的 Series 数据，期望结果为 True
            pytest.param(np.all, {"A": Series([1, 2], dtype="M8[ns]")}, True),
            # 测试 np.all 函数，传入包含带时区的 datetime 类型序列的 Series 数据，期望结果为 True
            pytest.param(np.all, {"A": Series([1, 2], dtype="M8[ns, UTC]")}, True),
            # 测试 np.any 函数，传入包含整数序列的 Series 数据，期望结果为 True
            pytest.param(np.any, {"A": Series([1, 2], dtype="M8[ns]")}, True),
            # 测试 np.any 函数，传入包含带时区的 datetime 类型序列的 Series 数据，期望结果为 True
            pytest.param(np.any, {"A": Series([1, 2], dtype="M8[ns, UTC]")}, True),
            # 测试 np.all 函数，传入包含整数序列的 Series 数据，期望结果为 False
            pytest.param(np.all, {"A": Series([0, 1], dtype="m8[ns]")}, False),
            # 测试 np.any 函数，传入包含整数序列的 Series 数据，期望结果为 True
            pytest.param(np.any, {"A": Series([0, 1], dtype="m8[ns]")}, True),
            # 测试 np.all 函数，传入包含整数序列的 Series 数据，期望结果为 True
            pytest.param(np.all, {"A": Series([1, 2], dtype="m8[ns]")}, True),
            # 测试 np.any 函数，传入包含整数序列的 Series 数据，期望结果为 True
            pytest.param(np.any, {"A": Series([1, 2], dtype="m8[ns]")}, True),
            # 测试 np.all 函数，传入包含分类类型序列的 Series 数据，期望结果为 True
            # 由于 np.all 在分类类型上抛出异常，所以降级为对空 Series 的操作，结果为 True
            (np.all, {"A": Series([0, 1], dtype="category")}, True),
            # 测试 np.any 函数，传入包含分类类型序列的 Series 数据，期望结果为 False
            (np.any, {"A": Series([0, 1], dtype="category")}, False),
            # 测试 np.all 函数，传入包含分类类型序列的 Series 数据，期望结果为 True
            (np.all, {"A": Series([1, 2], dtype="category")}, True),
            # 测试 np.any 函数，传入包含分类类型序列的 Series 数据，期望结果为 False
            (np.any, {"A": Series([1, 2], dtype="category")}, False),
            # 混合类型测试 GH#21484
            # 测试 np.all 函数，传入包含 datetime 类型和 timedelta 类型序列的 Series 数据，期望结果为 True
            pytest.param(
                np.all,
                {
                    "A": Series([10, 20], dtype="M8[ns]"),
                    "B": Series([10, 20], dtype="m8[ns]"),
                },
                True,
            ),
        ],
    )
    # GH 19976
    # 使用 DataFrame 封装输入的数据
    data = DataFrame(data)

    # 如果数据中包含任何 CategoricalDtype 类型的列，则期望抛出 TypeError 异常
    if any(isinstance(x, CategoricalDtype) for x in data.dtypes):
        # 验证函数调用时抛出特定异常
        with pytest.raises(
            TypeError, match=".* dtype category does not support operation"
        ):
            func(data)

        # 以方法调用形式验证函数调用时抛出特定异常
        with pytest.raises(
            TypeError, match=".* dtype category does not support operation"
        ):
            getattr(DataFrame(data), func.__name__)(axis=None)

    # 如果数据中包含任何 datetime64 类型的列，则期望抛出 TypeError 异常
    if data.dtypes.apply(lambda x: x.kind == "M").any():
        msg = "datetime64 type does not support operation '(any|all)'"
        # 验证函数调用时抛出特定异常
        with pytest.raises(TypeError, match=msg):
            func(data)

        # 以方法调用形式验证函数调用时抛出特定异常
        with pytest.raises(TypeError, match=msg):
            getattr(DataFrame(data), func.__name__)(axis=None)

    # 如果数据中不包含 category 类型的列，则执行以下操作
    elif data.dtypes.apply(lambda x: x != "category").any():
        # 调用函数并验证结果类型为 np.bool_
        result = func(data)
        assert isinstance(result, np.bool_)
        # 验证函数返回结果与期望值一致
        assert result.item() is expected

        # 以方法调用形式调用函数并验证结果类型为 np.bool_
        result = getattr(DataFrame(data), func.__name__)(axis=None)
        assert isinstance(result, np.bool_)
        # 验证函数返回结果与期望值一致
        assert result.item() is expected
    def test_series_broadcasting(self):
        # 定义一个测试方法，用于测试 Series 和 DataFrame 的广播操作
        # GH 16378, GH 16306 是 GitHub 上的问题编号，可能与此测试相关

        # 创建一个包含三个相同值的 DataFrame 对象
        df = DataFrame([1.0, 1.0, 1.0])
        # 创建一个包含 NaN 值的 DataFrame 对象，其中列 'A' 包含 NaN 值
        df_nan = DataFrame({"A": [np.nan, 2.0, np.nan]})
        # 创建一个包含三个相同值的 Series 对象
        s = Series([1, 1, 1])
        # 创建一个包含 NaN 值的 Series 对象
        s_nan = Series([np.nan, np.nan, 1])

        # 使用上下文管理器来检查是否产生了任何警告
        with tm.assert_produces_warning(None):
            # 对 df_nan 应用 clip 方法，使用 s 作为 lower 边界，axis=0 表示按行操作
            df_nan.clip(lower=s, axis=0)
            # 对于每一个操作符 op（如 lt, le, gt, ge, eq, ne），在 df 上调用对应的比较方法
            for op in ["lt", "le", "gt", "ge", "eq", "ne"]:
                getattr(df, op)(s_nan, axis=0)
class TestDataFrameReductions:
    # 测试在包含 NaT 的情况下使用 dt64 的最小最大值
    def test_min_max_dt64_with_NaT(self):
        # 创建包含 NaT 和 Timestamp 的 DataFrame
        df = DataFrame({"foo": [pd.NaT, pd.NaT, Timestamp("2012-05-01")]})
        
        # 计算 DataFrame 的最小值
        res = df.min()
        # 期望的最小值 Series
        exp = Series([Timestamp("2012-05-01")], index=["foo"])
        tm.assert_series_equal(res, exp)
        
        # 计算 DataFrame 的最大值
        res = df.max()
        # 期望的最大值 Series
        exp = Series([Timestamp("2012-05-01")], index=["foo"])
        tm.assert_series_equal(res, exp)
        
        # GH12941, 只有 NaT 在 DataFrame 中
        df = DataFrame({"foo": [pd.NaT, pd.NaT]})
        
        # 计算 DataFrame 的最小值
        res = df.min()
        # 期望的最小值 Series
        exp = Series([pd.NaT], index=["foo"])
        tm.assert_series_equal(res, exp)
        
        # 计算 DataFrame 的最大值
        res = df.max()
        # 期望的最大值 Series
        exp = Series([pd.NaT], index=["foo"])
        tm.assert_series_equal(res, exp)

    # 测试在包含 NaT 的情况下使用 dt64 的最小最大值，且 skipna 参数为 False
    def test_min_max_dt64_with_NaT_skipna_false(self, request, tz_naive_fixture):
        # GH#36907
        tz = tz_naive_fixture
        if isinstance(tz, tzlocal) and is_platform_windows():
            pytest.skip(
                "GH#37659 OSError raised within tzlocal bc Windows "
                "chokes in times before 1970-01-01"
            )
        
        # 创建包含不同时间戳和 NaT 的 DataFrame
        df = DataFrame(
            {
                "a": [
                    Timestamp("2020-01-01 08:00:00", tz=tz),
                    Timestamp("1920-02-01 09:00:00", tz=tz),
                ],
                "b": [Timestamp("2020-02-01 08:00:00", tz=tz), pd.NaT],
            }
        )
        
        # 计算每行的最小值，不跳过 NaN
        res = df.min(axis=1, skipna=False)
        expected = Series([df.loc[0, "a"], pd.NaT])
        assert expected.dtype == df["a"].dtype
        tm.assert_series_equal(res, expected)
        
        # 计算每行的最大值，不跳过 NaN
        res = df.max(axis=1, skipna=False)
        expected = Series([df.loc[0, "b"], pd.NaT])
        assert expected.dtype == df["a"].dtype
        tm.assert_series_equal(res, expected)

    # 测试在空 Series/DataFrames 上调用 min/max API 的一致性，其中包含 NaT
    def test_min_max_dt64_api_consistency_with_NaT(self):
        # 调用以下求和函数，对于 dataframes 返回错误，而对于 series 返回 NaT。这些测试检查 API 在空 Series/DataFrames 上的 min/max 调用是否一致。
        # 参见 GH:33704 获取更多信息
        df = DataFrame({"x": to_datetime([])})
        expected_dt_series = Series(to_datetime([]))
        
        # 检查 axis 0
        assert (df.min(axis=0).x is pd.NaT) == (expected_dt_series.min() is pd.NaT)
        assert (df.max(axis=0).x is pd.NaT) == (expected_dt_series.max() is pd.NaT)
        
        # 检查 axis 1
        tm.assert_series_equal(df.min(axis=1), expected_dt_series)
        tm.assert_series_equal(df.max(axis=1), expected_dt_series)
    def test_min_max_dt64_api_consistency_empty_df(self):
        # 检查在空的 DataFrame/Series 上调用 min/max 方法时，DataFrame/Series 的 API 一致性。
        # 创建一个空的 DataFrame
        df = DataFrame({"x": []})
        # 创建一个预期的空浮点数 Series
        expected_float_series = Series([], dtype=float)
        
        # 检查在轴 0 上的最小值和最大值
        assert np.isnan(df.min(axis=0).x) == np.isnan(expected_float_series.min())
        assert np.isnan(df.max(axis=0).x) == np.isnan(expected_float_series.max())
        
        # 检查在轴 1 上的最小值
        tm.assert_series_equal(df.min(axis=1), expected_float_series)
        # 再次检查在轴 1 上的最小值（这里有可能是一个错误，应该是检查最大值）
        tm.assert_series_equal(df.min(axis=1), expected_float_series)

    @pytest.mark.parametrize(
        "initial",
        ["2018-10-08 13:36:45+00:00", "2018-10-08 13:36:45+03:00"],  # 非UTC时区
    )
    @pytest.mark.parametrize("method", ["min", "max"])
    def test_preserve_timezone(self, initial: str, method):
        # GH 28552
        # 将初始时间字符串转换为日期时间对象
        initial_dt = to_datetime(initial)
        expected = Series([initial_dt])
        # 创建一个包含预期 Series 的 DataFrame
        df = DataFrame([expected])
        # 调用指定方法（min/max）在轴 1 上进行计算
        result = getattr(df, method)(axis=1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_minmax_tzaware_skipna_axis_1(self, method, skipna):
        # GH#51242
        # 创建一个包含时区感知对象的 DataFrame
        val = to_datetime("1900-01-01", utc=True)
        df = DataFrame(
            {"a": Series([pd.NaT, pd.NaT, val]), "b": Series([pd.NaT, val, val])}
        )
        # 获取指定方法（min/max）的操作对象
        op = getattr(df, method)
        # 在轴 1 上执行操作，根据 skipna 参数决定是否跳过 NaN 值
        result = op(axis=1, skipna=skipna)
        if skipna:
            expected = Series([pd.NaT, val, val])
        else:
            expected = Series([pd.NaT, pd.NaT, val])
        tm.assert_series_equal(result, expected)

    def test_frame_any_with_timedelta(self):
        # GH#17667
        # 创建一个包含 timedelta 的 DataFrame
        df = DataFrame(
            {
                "a": Series([0, 0]),
                "t": Series([to_timedelta(0, "s"), to_timedelta(1, "ms")]),
            }
        )

        # 在轴 0 上应用 any 方法
        result = df.any(axis=0)
        expected = Series(data=[False, True], index=["a", "t"])
        tm.assert_series_equal(result, expected)

        # 在轴 1 上应用 any 方法
        result = df.any(axis=1)
        expected = Series(data=[False, True])
        tm.assert_series_equal(result, expected)

    def test_reductions_skipna_none_raises(
        self, request, frame_or_series, all_reductions
    ):
        # 如果是计数操作，则标记为预期失败，因为计数操作不接受 skipna 参数
        if all_reductions == "count":
            request.applymarker(
                pytest.mark.xfail(reason="Count does not accept skipna")
            )
        # 创建一个 DataFrame 或 Series 对象
        obj = frame_or_series([1, 2, 3])
        # 准备一个预期的错误消息
        msg = 'For argument "skipna" expected type bool, received type NoneType.'
        # 断言调用指定的聚合函数时会抛出 ValueError 异常，且错误消息符合预期
        with pytest.raises(ValueError, match=msg):
            getattr(obj, all_reductions)(skipna=None)
    def test_reduction_timestamp_smallest_unit(self):
        # GH#52524
        # 创建一个 DataFrame 对象，包含两列：
        # - 列 'a' 包含一个 datetime64 秒精度的时间戳 '2019-12-31'
        # - 列 'b' 包含一个 datetime64 毫秒精度的时间戳 '2019-12-31 00:00:00.123'
        df = DataFrame(
            {
                "a": Series([Timestamp("2019-12-31")], dtype="datetime64[s]"),
                "b": Series(
                    [Timestamp("2019-12-31 00:00:00.123")], dtype="datetime64[ms]"
                ),
            }
        )
        # 对 DataFrame 执行最大值计算
        result = df.max()
        # 创建预期的 Series 对象，包含两个时间戳：
        # - 时间戳 '2019-12-31'，精度为毫秒
        # - 时间戳 '2019-12-31 00:00:00.123'，精度为毫秒
        expected = Series(
            [Timestamp("2019-12-31"), Timestamp("2019-12-31 00:00:00.123")],
            dtype="datetime64[ms]",
            index=["a", "b"],
        )
        # 使用测试框架的方法来断言结果与预期是否相等
        tm.assert_series_equal(result, expected)

    def test_reduction_timedelta_smallest_unit(self):
        # GH#52524
        # 创建一个 DataFrame 对象，包含两列：
        # - 列 'a' 包含一个 timedelta64 秒精度的时间差 '1 days'
        # - 列 'b' 包含一个 timedelta64 毫秒精度的时间差 '1 days'
        df = DataFrame(
            {
                "a": Series([pd.Timedelta("1 days")], dtype="timedelta64[s]"),
                "b": Series([pd.Timedelta("1 days")], dtype="timedelta64[ms]"),
            }
        )
        # 对 DataFrame 执行最大值计算
        result = df.max()
        # 创建预期的 Series 对象，包含两个时间差：
        # - 时间差 '1 days'，精度为毫秒
        # - 时间差 '1 days'，精度为毫秒
        expected = Series(
            [pd.Timedelta("1 days"), pd.Timedelta("1 days")],
            dtype="timedelta64[ms]",
            index=["a", "b"],
        )
        # 使用测试框架的方法来断言结果与预期是否相等
        tm.assert_series_equal(result, expected)
    # 定义一个测试类 TestNuisanceColumns，用于测试处理DataFrame中的特定情况
    class TestNuisanceColumns:
        
        # 测试当存在任意或全部分类数据类型的干扰列时的行为
        def test_any_all_categorical_dtype_nuisance_column(self, all_boolean_reductions):
            # 创建一个名为 "A" 的分类数据类型的Series
            ser = Series([0, 1], dtype="category", name="A")
            # 将Series转换为DataFrame
            df = ser.to_frame()

            # 确认Series的行为是抛出TypeError异常
            with pytest.raises(TypeError, match="does not support operation"):
                # 调用指定的逻辑运算方法（例如 all() 或 any()）于Series对象上
                getattr(ser, all_boolean_reductions)()

            with pytest.raises(TypeError, match="does not support operation"):
                # 在Series对象上调用numpy函数，期望抛出TypeError异常
                getattr(np, all_boolean_reductions)(ser)

            with pytest.raises(TypeError, match="does not support operation"):
                # 在DataFrame对象上调用指定的逻辑运算方法，禁用仅限布尔值的条件
                getattr(df, all_boolean_reductions)(bool_only=False)

            with pytest.raises(TypeError, match="does not support operation"):
                # 在DataFrame对象上调用指定的逻辑运算方法，禁用空条件
                getattr(df, all_boolean_reductions)(bool_only=None)

            with pytest.raises(TypeError, match="does not support operation"):
                # 在numpy函数上调用DataFrame对象，期望抛出TypeError异常
                getattr(np, all_boolean_reductions)(df, axis=0)

        # 测试当存在分类数据类型的干扰列时，DataFrame.median的行为
        def test_median_categorical_dtype_nuisance_column(self):
            # 创建一个包含分类数据类型列 "A" 的DataFrame
            df = DataFrame({"A": Categorical([1, 2, 2, 2, 3])})
            # 获取DataFrame中 "A" 列的Series对象
            ser = df["A"]

            # 确认Series的行为是抛出TypeError异常
            with pytest.raises(TypeError, match="does not support operation"):
                # 调用Series对象的 median() 方法
                ser.median()

            with pytest.raises(TypeError, match="does not support operation"):
                # 在DataFrame对象上调用 median() 方法，禁用仅数值的条件
                df.median(numeric_only=False)

            with pytest.raises(TypeError, match="does not support operation"):
                # 在DataFrame对象上调用 median() 方法
                df.median()

            # 将 "A" 列转换为整数类型，并添加到DataFrame作为 "B" 列
            df["B"] = df["A"].astype(int)

            with pytest.raises(TypeError, match="does not support operation"):
                # 在DataFrame对象上调用 median() 方法，禁用仅数值的条件
                df.median(numeric_only=False)

            with pytest.raises(TypeError, match="does not support operation"):
                # 在DataFrame对象上调用 median() 方法
                df.median()

            # TODO: np.median(df, axis=0) gives np.array([2.0, 2.0]) instead
            #  of expected.values

        @pytest.mark.parametrize("method", ["min", "max"])
    # 定义一个测试方法，用于测试非有序分类数据类型的最小和最大值计算行为
    def test_min_max_categorical_dtype_non_ordered_nuisance_column(self, method):
        # 创建一个无序的分类数据对象
        cat = Categorical(["a", "b", "c", "b"], ordered=False)
        # 将分类数据对象转换为 Series
        ser = Series(cat)
        # 将 Series 转换为 DataFrame，列名为"A"
        df = ser.to_frame("A")

        # 验证 Series 的行为
        # 使用 pytest 断言检查是否会抛出 TypeError，并检查异常消息是否包含特定字符串
        with pytest.raises(TypeError, match="is not ordered for operation"):
            # 调用 method 方法（参数 method 是传入的方法名）对 ser 进行操作
            getattr(ser, method)()

        # 针对 numpy 函数对 Series 执行相同的操作验证
        with pytest.raises(TypeError, match="is not ordered for operation"):
            # 调用 numpy 的 method 方法对 ser 进行操作
            getattr(np, method)(ser)

        # 针对 DataFrame 的列"A"执行操作，numeric_only=False
        with pytest.raises(TypeError, match="is not ordered for operation"):
            # 调用 method 方法对 df 进行操作，numeric_only=False
            getattr(df, method)(numeric_only=False)

        # 针对 DataFrame 的列"A"执行操作，不指定 numeric_only
        with pytest.raises(TypeError, match="is not ordered for operation"):
            # 调用 method 方法对 df 进行操作，不指定 numeric_only
            getattr(df, method)()

        # 针对 DataFrame 对整体执行操作，axis=0
        with pytest.raises(TypeError, match="is not ordered for operation"):
            # 调用 numpy 的 method 方法对 df 进行操作，axis=0
            getattr(np, method)(df, axis=0)

        # 在 DataFrame 中添加一个非分类的列"B"
        df["B"] = df["A"].astype(object)
        # 针对包含非分类列的 DataFrame 执行操作
        with pytest.raises(TypeError, match="is not ordered for operation"):
            # 调用 method 方法对 df 进行操作
            getattr(df, method)()

        # 针对包含非分类列的 DataFrame 执行 numpy 操作，axis=0
        with pytest.raises(TypeError, match="is not ordered for operation"):
            # 调用 numpy 的 method 方法对 df 进行操作，axis=0
            getattr(np, method)(df, axis=0)
class TestEmptyDataFrameReductions:
    @pytest.mark.parametrize(
        "opname, dtype, exp_value, exp_dtype",
        [
            ("sum", np.int8, 0, np.int64),  # 定义参数：操作名称为sum，数据类型为np.int8，期望值为0，期望数据类型为np.int64
            ("prod", np.int8, 1, np.int_),  # 定义参数：操作名称为prod，数据类型为np.int8，期望值为1，期望数据类型为np.int_
            ("sum", np.int64, 0, np.int64),  # 定义参数：操作名称为sum，数据类型为np.int64，期望值为0，期望数据类型为np.int64
            ("prod", np.int64, 1, np.int64),  # 定义参数：操作名称为prod，数据类型为np.int64，期望值为1，期望数据类型为np.int64
            ("sum", np.uint8, 0, np.uint64),  # 定义参数：操作名称为sum，数据类型为np.uint8，期望值为0，期望数据类型为np.uint64
            ("prod", np.uint8, 1, np.uint),  # 定义参数：操作名称为prod，数据类型为np.uint8，期望值为1，期望数据类型为np.uint
            ("sum", np.uint64, 0, np.uint64),  # 定义参数：操作名称为sum，数据类型为np.uint64，期望值为0，期望数据类型为np.uint64
            ("prod", np.uint64, 1, np.uint64),  # 定义参数：操作名称为prod，数据类型为np.uint64，期望值为1，期望数据类型为np.uint64
            ("sum", np.float32, 0, np.float32),  # 定义参数：操作名称为sum，数据类型为np.float32，期望值为0，期望数据类型为np.float32
            ("prod", np.float32, 1, np.float32),  # 定义参数：操作名称为prod，数据类型为np.float32，期望值为1，期望数据类型为np.float32
            ("sum", np.float64, 0, np.float64),  # 定义参数：操作名称为sum，数据类型为np.float64，期望值为0，期望数据类型为np.float64
        ],
    )
    def test_df_empty_min_count_0(self, opname, dtype, exp_value, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)  # 创建一个空的DataFrame，指定数据类型为dtype
        result = getattr(df, opname)(min_count=0)  # 调用DataFrame对象的opname方法（如sum或prod），min_count设为0

        expected = Series([exp_value, exp_value], dtype=exp_dtype)  # 创建期望的Series对象，包含预期值和数据类型
        tm.assert_series_equal(result, expected)  # 使用测试工具tm断言result与expected相等

    @pytest.mark.parametrize(
        "opname, dtype, exp_dtype",
        [
            ("sum", np.int8, np.float64),  # 定义参数：操作名称为sum，数据类型为np.int8，期望数据类型为np.float64
            ("prod", np.int8, np.float64),  # 定义参数：操作名称为prod，数据类型为np.int8，期望数据类型为np.float64
            ("sum", np.int64, np.float64),  # 定义参数：操作名称为sum，数据类型为np.int64，期望数据类型为np.float64
            ("prod", np.int64, np.float64),  # 定义参数：操作名称为prod，数据类型为np.int64，期望数据类型为np.float64
            ("sum", np.uint8, np.float64),  # 定义参数：操作名称为sum，数据类型为np.uint8，期望数据类型为np.float64
            ("prod", np.uint8, np.float64),  # 定义参数：操作名称为prod，数据类型为np.uint8，期望数据类型为np.float64
            ("sum", np.uint64, np.float64),  # 定义参数：操作名称为sum，数据类型为np.uint64，期望数据类型为np.float64
            ("prod", np.uint64, np.float64),  # 定义参数：操作名称为prod，数据类型为np.uint64，期望数据类型为np.float64
            ("sum", np.float32, np.float32),  # 定义参数：操作名称为sum，数据类型为np.float32，期望数据类型为np.float32
            ("prod", np.float32, np.float32),  # 定义参数：操作名称为prod，数据类型为np.float32，期望数据类型为np.float32
            ("sum", np.float64, np.float64),  # 定义参数：操作名称为sum，数据类型为np.float64，期望数据类型为np.float64
        ],
    )
    def test_df_empty_min_count_1(self, opname, dtype, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)  # 创建一个空的DataFrame，指定数据类型为dtype
        result = getattr(df, opname)(min_count=1)  # 调用DataFrame对象的opname方法（如sum或prod），min_count设为1

        expected = Series([np.nan, np.nan], dtype=exp_dtype)  # 创建期望的Series对象，包含预期的NaN值和数据类型
        tm.assert_series_equal(result, expected)  # 使用测试工具tm断言result与expected相等

    @pytest.mark.parametrize(
        "opname, dtype, exp_value, exp_dtype",
        [
            ("sum", "Int8", 0, ("Int32" if is_windows_np2_or_is32 else "Int64")),  # 定义参数：操作名称为sum，数据类型为"Int8"，期望值为0，期望数据类型为"Int32"或"Int64"
            ("prod", "Int8", 1, ("Int32" if is_windows_np2_or_is32 else "Int64")),  # 定义参数：操作名称为prod，数据类型为"Int8"，期望值为1，期望数据类型为"Int32"或"Int64"
            ("sum", "Int64", 0, "Int64"),  # 定义参数：操作名称为sum，数据类型为"Int64"，期望值为0，期望数据类型为"Int64"
            ("prod", "Int64", 1, "Int64"),  # 定义参数：操作名称为prod，数据类型为"Int64"，期望值为1，期望数据类型为"Int64"
            ("sum", "UInt8", 0, ("UInt32" if is_windows_np2_or_is32 else "UInt64")),  # 定义参数：操作名称为sum，数据类型为"UInt8"，期望值为0，期望数据类型为"UInt32"或"UInt64"
            ("prod", "UInt8", 1, ("UInt32" if is_windows_np2_or_is32 else "UInt64")),  # 定义参数：操作名称为prod，数据类型为"UInt8"，期望值为1，期望数据类型为"UInt32"或"UInt64"
            ("sum", "UInt64", 0, "UInt64"),  # 定义参数：操作名称为sum，数据类型为"UInt64"，期望值为0，期望数据类型为"UInt64"
            ("prod", "UInt64", 1, "UInt64"),  # 定义参数：操作名称为prod，数据类型为"UInt64"，期望值为1，期望数据类型为"UInt64"
            ("sum", "Float32", 0, "Float32"),  # 定义参数：操作名称为sum，数据类型为"Float32"，期望值为0，期望数据类型为"Float32"
            ("prod", "Float32", 1, "Float32"),  # 定义参数：操作名称为prod，数据类型为"Float32"，期望值为1，期望数据类型为"Float32"
            ("sum", "Float64", 0, "Float64"),  # 定义参数：操作名称为sum，数据类型为"Float64"，期望值为0，期望数据类型为"Float64"
        ],
    )
    def test_df_empty_nullable_min_count_0(self,
    # 使用 pytest 的 parametrize 装饰器为测试方法提供多组参数化输入
    @pytest.mark.parametrize(
        "opname, dtype, exp_dtype",
        [
            ("sum", "Int8", ("Int32" if is_windows_or_is32 else "Int64")),  # 设置操作名，数据类型及预期输出数据类型
            ("prod", "Int8", ("Int32" if is_windows_or_is32 else "Int64")),  # 设置操作名，数据类型及预期输出数据类型
            ("sum", "Int64", "Int64"),  # 设置操作名，数据类型及预期输出数据类型
            ("prod", "Int64", "Int64"),  # 设置操作名，数据类型及预期输出数据类型
            ("sum", "UInt8", ("UInt32" if is_windows_or_is32 else "UInt64")),  # 设置操作名，数据类型及预期输出数据类型
            ("prod", "UInt8", ("UInt32" if is_windows_or_is32 else "UInt64")),  # 设置操作名，数据类型及预期输出数据类型
            ("sum", "UInt64", "UInt64"),  # 设置操作名，数据类型及预期输出数据类型
            ("prod", "UInt64", "UInt64"),  # 设置操作名，数据类型及预期输出数据类型
            ("sum", "Float32", "Float32"),  # 设置操作名，数据类型及预期输出数据类型
            ("prod", "Float32", "Float32"),  # 设置操作名，数据类型及预期输出数据类型
            ("sum", "Float64", "Float64"),  # 设置操作名，数据类型及预期输出数据类型
        ],
    )
    # 定义测试方法，测试 DataFrame 对象在空数据列上进行操作时的行为
    def test_df_empty_nullable_min_count_1(self, opname, dtype, exp_dtype):
        # 创建一个空的 DataFrame 对象，指定列的数据类型
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        # 调用 DataFrame 对象的指定操作方法，设置 min_count 参数为 1
        result = getattr(df, opname)(min_count=1)
    
        # 创建一个期望的 Series 对象，包含两个空值(pd.NA)，并指定数据类型
        expected = Series([pd.NA, pd.NA], dtype=exp_dtype)
        # 使用 pytest 的 assert_series_equal 函数验证操作结果与期望值是否相等
        tm.assert_series_equal(result, expected)
def test_sum_timedelta64_skipna_false():
    # GH#17235
    # 创建一个包含整数数据的 NumPy 数组，转换为 timedelta64[s] 类型，并设定一个元素为 NaT
    arr = np.arange(8).astype(np.int64).view("m8[s]").reshape(4, 2)
    arr[-1, -1] = "Nat"

    # 根据 NumPy 数组创建 DataFrame
    df = DataFrame(arr)
    # 检查 DataFrame 的数据类型是否与数组的数据类型一致
    assert (df.dtypes == arr.dtype).all()

    # 对 DataFrame 进行 skipna=False 的求和操作
    result = df.sum(skipna=False)
    # 创建预期的结果 Series，包含 timedelta64[s] 类型的数据和 NaT
    expected = Series([pd.Timedelta(seconds=12), pd.NaT], dtype="m8[s]")
    # 断言结果是否与预期相等
    tm.assert_series_equal(result, expected)

    # 按列进行 skipna=False 的求和操作
    result = df.sum(axis=0, skipna=False)
    # 断言结果是否与预期相等
    tm.assert_series_equal(result, expected)

    # 按行进行 skipna=False 的求和操作
    result = df.sum(axis=1, skipna=False)
    # 创建预期的结果 Series，包含 timedelta64[s] 类型的数据和 NaT
    expected = Series(
        [
            pd.Timedelta(seconds=1),
            pd.Timedelta(seconds=5),
            pd.Timedelta(seconds=9),
            pd.NaT,
        ],
        dtype="m8[s]",
    )
    # 断言结果是否与预期相等
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(
    using_pyarrow_string_dtype(), reason="sum doesn't work with arrow strings"
)
def test_mixed_frame_with_integer_sum():
    # https://github.com/pandas-dev/pandas/issues/34520
    # 创建一个包含字符串和整数的 DataFrame
    df = DataFrame([["a", 1]], columns=list("ab"))
    # 将列'b'的数据类型转换为 'Int64'
    df = df.astype({"b": "Int64"})
    # 对 DataFrame 进行求和操作
    result = df.sum()
    # 创建预期的结果 Series
    expected = Series(["a", 1], index=["a", "b"])
    # 断言结果是否与预期相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False, None])
@pytest.mark.parametrize("method", ["min", "max"])
def test_minmax_extensionarray(method, numeric_only):
    # https://github.com/pandas-dev/pandas/issues/32651
    # 获取 'int64' 的信息
    int64_info = np.iinfo("int64")
    # 创建一个包含 pd.Int64Dtype() 类型的 Series
    ser = Series([int64_info.max, None, int64_info.min], dtype=pd.Int64Dtype())
    # 根据 Series 创建 DataFrame
    df = DataFrame({"Int64": ser})
    # 根据给定的方法名进行相应的操作（如最小值、最大值等）
    result = getattr(df, method)(numeric_only=numeric_only)
    # 创建预期的结果 Series
    expected = Series(
        [getattr(int64_info, method)],
        dtype="Int64",
        index=Index(["Int64"]),
    )
    # 断言结果是否与预期相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ts_value", [Timestamp("2000-01-01"), pd.NaT])
def test_frame_mixed_numeric_object_with_timestamp(ts_value):
    # GH 13912
    # 创建一个包含不同数据类型的 DataFrame
    df = DataFrame({"a": [1], "b": [1.1], "c": ["foo"], "d": [ts_value]})
    # 使用 pytest 断言检查是否会引发 TypeError 异常
    with pytest.raises(TypeError, match="does not support operation"):
        df.sum()


def test_prod_sum_min_count_mixed_object():
    # https://github.com/pandas-dev/pandas/issues/41074
    # 创建一个包含混合类型数据的 DataFrame
    df = DataFrame([1, "a", True])

    # 对 DataFrame 进行 prod 操作，numeric_only=False，min_count=1
    result = df.prod(axis=0, min_count=1, numeric_only=False)
    # 创建预期的结果 Series
    expected = Series(["a"], dtype=object)
    # 断言结果是否与预期相等
    tm.assert_series_equal(result, expected)

    # 使用 pytest 断言检查是否会引发 TypeError 异常
    msg = re.escape("unsupported operand type(s) for +: 'int' and 'str'")
    with pytest.raises(TypeError, match=msg):
        df.sum(axis=0, min_count=1, numeric_only=False)


@pytest.mark.parametrize("method", ["min", "max", "mean", "median", "skew", "kurt"])
@pytest.mark.parametrize("numeric_only", [True, False])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_reduction_axis_none_returns_scalar(method, numeric_only, dtype):
    # GH#21597 As of 2.0, axis=None reduces over all axes.
    # 根据给定的 dtype 创建一个包含随机数据的 DataFrame
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), dtype=dtype)
    # 使用 getattr 函数从 DataFrame 对象 df 中动态调用指定方法，并传入参数
    result = getattr(df, method)(axis=None, numeric_only=numeric_only)
    
    # 将 DataFrame 转换为 NumPy 数组，数据类型为 np.float64
    np_arr = df.to_numpy(dtype=np.float64)
    
    # 如果 method 是 {"skew", "kurt"} 中的一种
    if method in {"skew", "kurt"}:
        # 导入 pytest 模块，如果导入失败则跳过测试
        comp_mod = pytest.importorskip("scipy.stats")
        
        # 如果 method 是 "kurt"，则将其修改为 "kurtosis"
        if method == "kurt":
            method = "kurtosis"
        
        # 使用 getattr 函数从 comp_mod 模块中动态调用指定方法，并传入参数
        expected = getattr(comp_mod, method)(np_arr, bias=False, axis=None)
        
        # 使用 tm.assert_almost_equal 函数比较 result 和 expected，要求几乎相等
        tm.assert_almost_equal(result, expected)
    
    # 如果 method 不在 {"skew", "kurt"} 中
    else:
        # 使用 getattr 函数从 np 模块中动态调用指定方法，并传入参数
        expected = getattr(np, method)(np_arr, axis=None)
        
        # 断言 result 等于 expected
        assert result == expected
@pytest.mark.parametrize(
    "kernel",
    [
        "corr",
        "corrwith",
        "cov",
        "idxmax",
        "idxmin",
        "kurt",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "quantile",
        "sem",
        "skew",
        "std",
        "sum",
        "var",
    ],
)
# 定义一个参数化测试函数，测试不同的数据操作方法（如相关性计算、最大最小值等）
def test_fails_on_non_numeric(kernel):
    # GH#46852
    # 创建一个 DataFrame 对象，包含一个整数列和一个对象列
    df = DataFrame({"a": [1, 2, 3], "b": object})
    # 如果 kernel 是 "corrwith"，则传入 df，否则传入空参数元组
    args = (df,) if kernel == "corrwith" else ()
    # 构建异常匹配的消息，包括多种可能的错误信息
    msg = "|".join(
        [
            "not allowed for this dtype",
            "argument must be a string or a number",
            "not supported between instances of",
            "unsupported operand type",
            "argument must be a string or a real number",
        ]
    )
    # 如果 kernel 是 "median"，则重新定义 msg 以适应不同的构建版本
    if kernel == "median":
        msg1 = (
            r"Cannot convert \[\[<class 'object'> <class 'object'> "
            r"<class 'object'>\]\] to numeric"
        )
        msg2 = (
            r"Cannot convert \[<class 'object'> <class 'object'> "
            r"<class 'object'>\] to numeric"
        )
        msg = "|".join([msg1, msg2])
    # 使用 pytest 的 raises 方法检查是否抛出 TypeError 异常，并匹配预期的异常消息
    with pytest.raises(TypeError, match=msg):
        getattr(df, kernel)(*args)


@pytest.mark.parametrize(
    "method",
    [
        "all",
        "any",
        "count",
        "idxmax",
        "idxmin",
        "kurt",
        "kurtosis",
        "max",
        "mean",
        "median",
        "min",
        "nunique",
        "prod",
        "product",
        "sem",
        "skew",
        "std",
        "sum",
        "var",
    ],
)
@pytest.mark.parametrize("min_count", [0, 2])
# 定义一个参数化测试函数，测试不同的统计方法和不同的最小计数情况
def test_numeric_ea_axis_1(method, skipna, min_count, any_numeric_ea_dtype):
    # GH 54341
    # 创建一个 DataFrame 对象，包含两列，每列数据类型为 any_numeric_ea_dtype 指定的数值类型
    df = DataFrame(
        {
            "a": Series([0, 1, 2, 3], dtype=any_numeric_ea_dtype),
            "b": Series([0, 1, pd.NA, 3], dtype=any_numeric_ea_dtype),
        },
    )
    # 期望的 DataFrame 结构，将 NaN 替换为相应的浮点数
    expected_df = DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [0.0, 1.0, np.nan, 3.0],
        },
    )
    # 根据方法选择预期的数据类型
    if method in ("count", "nunique"):
        expected_dtype = "int64"
    elif method in ("all", "any"):
        expected_dtype = "boolean"
    elif method in (
        "kurt",
        "kurtosis",
        "mean",
        "median",
        "sem",
        "skew",
        "std",
        "var",
    ) and not any_numeric_ea_dtype.startswith("Float"):
        expected_dtype = "Float64"
    else:
        expected_dtype = any_numeric_ea_dtype

    kwargs = {}
    # 如果方法不是 "count", "nunique", "quantile"，则设置 skipna 参数
    if method not in ("count", "nunique", "quantile"):
        kwargs["skipna"] = skipna
    # 如果方法是 "prod", "product", "sum"，则设置 min_count 参数
    if method in ("prod", "product", "sum"):
        kwargs["min_count"] = min_count
    # 如果不跳过 NA 值并且方法是 "idxmax" 或 "idxmin"
    if not skipna and method in ("idxmax", "idxmin"):
        # GH#57745 - EAs use groupby for axis=1 which still needs a proper deprecation.
        # 准备警告消息，指出DataFrame.{method}在全是NA值情况下的行为
        msg = f"The behavior of DataFrame.{method} with all-NA values"
        # 断言产生 FutureWarning 警告，消息内容匹配指定的消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 调用 DataFrame 的指定方法（如 idxmax 或 idxmin），对 axis=1 进行操作，传递额外参数
            getattr(df, method)(axis=1, **kwargs)
        # 断言抛出 ValueError 异常，消息内容匹配指定的消息
        with pytest.raises(ValueError, match="Encountered an NA value"):
            # 调用 expected_df 的指定方法（如 idxmax 或 idxmin），对 axis=1 进行操作，传递额外参数
            getattr(expected_df, method)(axis=1, **kwargs)
        # 返回，结束函数执行
        return
    
    # 调用 DataFrame 的指定方法（如 mean、sum），对 axis=1 进行操作，传递额外参数
    result = getattr(df, method)(axis=1, **kwargs)
    # 调用 expected_df 的指定方法（如 mean、sum），对 axis=1 进行操作，传递额外参数
    expected = getattr(expected_df, method)(axis=1, **kwargs)
    
    # 如果方法不是 "idxmax" 或 "idxmin"，将 expected 转换为指定的数据类型 expected_dtype
    if method not in ("idxmax", "idxmin"):
        expected = expected.astype(expected_dtype)
    
    # 使用测试模块中的 assert_series_equal 函数比较 result 和 expected 的序列是否相等
    tm.assert_series_equal(result, expected)
```