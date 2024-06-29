# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_numeric_only.py`

```
import re  # 导入 re 模块，用于正则表达式操作

import pytest  # 导入 pytest 模块，用于单元测试

from pandas._libs import lib  # 从 pandas._libs 中导入 lib

import pandas as pd  # 导入 pandas 库并简写为 pd
from pandas import (  # 从 pandas 中导入以下对象：
    DataFrame,  # 数据帧对象
    Index,  # 索引对象
    Series,  # 序列对象
    Timestamp,  # 时间戳对象
    date_range,  # 日期范围生成函数
)
import pandas._testing as tm  # 导入 pandas 测试相关的模块，简写为 tm
from pandas.tests.groupby import get_groupby_method_args  # 从 pandas.tests.groupby 中导入 get_groupby_method_args 函数


class TestNumericOnly:
    # 确保我们将 kwargs 传递给聚合函数

    @pytest.fixture
    def df(self):
        # GH3668
        # GH5724
        # 创建一个包含不同数据类型列的 DataFrame 对象
        df = DataFrame(
            {
                "group": [1, 1, 2],
                "int": [1, 2, 3],
                "float": [4.0, 5.0, 6.0],
                "string": list("abc"),
                "category_string": Series(list("abc")).astype("category"),
                "category_int": [7, 8, 9],
                "datetime": date_range("20130101", periods=3),
                "datetimetz": date_range("20130101", periods=3, tz="US/Eastern"),
                "timedelta": pd.timedelta_range("1 s", periods=3, freq="s"),
            },
            columns=[
                "group",
                "int",
                "float",
                "string",
                "category_string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ],
        )
        return df  # 返回创建的 DataFrame 对象

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_averages(self, df, method):
        # 计算均值或中位数
        expected_columns_numeric = Index(["int", "float", "category_int"])

        gb = df.groupby("group")  # 根据 'group' 列分组
        # 预期的聚合结果 DataFrame
        expected = DataFrame(
            {
                "category_int": [7.5, 9],
                "float": [4.5, 6.0],
                "timedelta": [pd.Timedelta("1.5s"), pd.Timedelta("3s")],
                "int": [1.5, 3],
                "datetime": [
                    Timestamp("2013-01-01 12:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                "datetimetz": [
                    Timestamp("2013-01-01 12:00:00", tz="US/Eastern"),
                    Timestamp("2013-01-03 00:00:00", tz="US/Eastern"),
                ],
            },
            index=Index([1, 2], name="group"),  # 索引为 'group' 列的值
            columns=[
                "int",
                "float",
                "category_int",
            ],
        )

        result = getattr(gb, method)(numeric_only=True)  # 调用聚合方法（均值或中位数），只考虑数值列
        tm.assert_frame_equal(result.reindex_like(expected), expected)  # 断言结果与预期相同

        expected_columns = expected.columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["min", "max"])
    # 定义一个测试函数，用于测试数据框在特定方法下的行为
    def test_extrema(self, df, method):
        # TODO: min, max *should* handle
        # categorical (ordered) dtype
        # 设置预期的列名索引，包括整数、浮点数、字符串、有序类别整数、日期时间、带时区日期时间、时间增量
        expected_columns = Index(
            [
                "int",
                "float",
                "string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ]
        )
        expected_columns_numeric = expected_columns

        # 调用内部方法进行检查
        self._check(df, method, expected_columns, expected_columns_numeric)

    # 使用 pytest 的参数化装饰器，分别测试 "first" 和 "last" 方法
    @pytest.mark.parametrize("method", ["first", "last"])
    def test_first_last(self, df, method):
        # 设置预期的列名索引，包括整数、浮点数、字符串、类别字符串、有序类别整数、日期时间、带时区日期时间、时间增量
        expected_columns = Index(
            [
                "int",
                "float",
                "string",
                "category_string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ]
        )
        expected_columns_numeric = expected_columns

        # 调用内部方法进行检查
        self._check(df, method, expected_columns, expected_columns_numeric)

    # 使用 pytest 的参数化装饰器，分别测试 "sum" 和 "cumsum" 方法
    @pytest.mark.parametrize("method", ["sum", "cumsum"])
    def test_sum_cumsum(self, df, method):
        # 设置预期的数值列名索引，包括整数、浮点数、类别整数
        expected_columns_numeric = Index(["int", "float", "category_int"])
        # 设置预期的列名索引，包括整数、浮点数、字符串、类别整数、时间增量；在累积和的情况下，丢失字符串列
        expected_columns = Index(
            ["int", "float", "string", "category_int", "timedelta"]
        )
        if method == "cumsum":
            # 在累积和方法下，进一步精简预期的列名索引，丢弃字符串列
            expected_columns = Index(["int", "float", "category_int", "timedelta"])

        # 调用内部方法进行检查
        self._check(df, method, expected_columns, expected_columns_numeric)

    # 使用 pytest 的参数化装饰器，分别测试 "prod" 和 "cumprod" 方法
    @pytest.mark.parametrize("method", ["prod", "cumprod"])
    def test_prod_cumprod(self, df, method):
        # 设置预期的列名索引，包括整数、浮点数、类别整数
        expected_columns = Index(["int", "float", "category_int"])
        expected_columns_numeric = expected_columns

        # 调用内部方法进行检查
        self._check(df, method, expected_columns, expected_columns_numeric)

    # 使用 pytest 的参数化装饰器，分别测试 "cummin" 和 "cummax" 方法
    @pytest.mark.parametrize("method", ["cummin", "cummax"])
    def test_cummin_cummax(self, df, method):
        # 设置预期的列名索引，包括整数、浮点数、类别整数、日期时间、带时区日期时间、时间增量
        expected_columns = Index(
            ["int", "float", "category_int", "datetime", "datetimetz", "timedelta"]
        )

        # GH#15561: numeric_only=False set by default like min/max
        # 设置预期的数值列名索引，与预期的列名索引一致
        expected_columns_numeric = expected_columns

        # 调用内部方法进行检查
        self._check(df, method, expected_columns, expected_columns_numeric)
    def _check(self, df, method, expected_columns, expected_columns_numeric):
        # 按照 "group" 列对 DataFrame 进行分组
        gb = df.groupby("group")

        # 如果方法以 "cum" 开头，则抛出 NotImplementedError，否则抛出 TypeError
        exception = NotImplementedError if method.startswith("cum") else TypeError

        # 根据不同的方法进行异常检查和处理
        if method in ("min", "max", "cummin", "cummax", "cumsum", "cumprod"):
            # 这些方法默认 numeric_only=False，并且会引发 TypeError
            msg = "|".join(
                [
                    "Categorical is not ordered",
                    f"Cannot perform {method} with non-ordered Categorical",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    # cumsum/cummin/cummax/cumprod
                    "function is not implemented for this dtype",
                ]
            )
            # 使用 pytest 检查是否抛出期望的异常，并匹配消息内容
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        elif method in ("sum", "mean", "median", "prod"):
            # 针对 sum、mean、median、prod 方法的异常消息
            msg = "|".join(
                [
                    "category type does not support sum operations",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    re.escape(f"agg function failed [how->{method},dtype->string]"),
                ]
            )
            # 使用 pytest 检查是否抛出期望的异常，并匹配消息内容
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        else:
            # 对于其余方法，执行并验证结果的列是否符合预期的数值列
            result = getattr(gb, method)()
            tm.assert_index_equal(result.columns, expected_columns_numeric)

        # 如果方法不是 "first" 或 "last"，则进行额外的异常检查
        if method not in ("first", "last"):
            msg = "|".join(
                [
                    "Categorical is not ordered",
                    "category type does not support",
                    "function is not implemented for this dtype",
                    f"Cannot perform {method} with non-ordered Categorical",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    re.escape(f"agg function failed [how->{method},dtype->string]"),
                ]
            )
            # 使用 pytest 检查是否抛出期望的异常，并匹配消息内容
            with pytest.raises(exception, match=msg):
                getattr(gb, method)(numeric_only=False)
        else:
            # 对于 "first" 或 "last" 方法，执行并验证结果的列是否符合预期的数值列
            result = getattr(gb, method)(numeric_only=False)
            tm.assert_index_equal(result.columns, expected_columns_numeric)
@pytest.mark.parametrize(
    "kernel, has_arg",
    [  # 定义参数化测试的核心函数和是否具有numeric_only参数的元组列表
        ("all", False),  # 使用所有数据的核心功能，没有numeric_only参数
        ("any", False),  # 使用任意数据的核心功能，没有numeric_only参数
        ("bfill", False),  # 使用后向填充的核心功能，没有numeric_only参数
        ("corr", True),  # 使用相关性计算的核心功能，并有numeric_only参数
        ("corrwith", True),  # 使用两个数据帧之间相关性计算的核心功能，并有numeric_only参数
        ("cov", True),  # 使用协方差计算的核心功能，并有numeric_only参数
        ("cummax", True),  # 使用累计最大值计算的核心功能，并有numeric_only参数
        ("cummin", True),  # 使用累计最小值计算的核心功能，并有numeric_only参数
        ("cumprod", True),  # 使用累计乘积计算的核心功能，并有numeric_only参数
        ("cumsum", True),  # 使用累计和计算的核心功能，并有numeric_only参数
        ("diff", False),  # 使用差分计算的核心功能，没有numeric_only参数
        ("ffill", False),  # 使用前向填充的核心功能，没有numeric_only参数
        ("first", True),  # 使用首个值计算的核心功能，并有numeric_only参数
        ("idxmax", True),  # 使用最大索引值计算的核心功能，并有numeric_only参数
        ("idxmin", True),  # 使用最小索引值计算的核心功能，并有numeric_only参数
        ("last", True),  # 使用最后一个值计算的核心功能，并有numeric_only参数
        ("max", True),  # 使用最大值计算的核心功能，并有numeric_only参数
        ("mean", True),  # 使用均值计算的核心功能，并有numeric_only参数
        ("median", True),  # 使用中位数计算的核心功能，并有numeric_only参数
        ("min", True),  # 使用最小值计算的核心功能，并有numeric_only参数
        ("nth", False),  # 使用第n个值计算的核心功能，没有numeric_only参数
        ("nunique", False),  # 使用唯一值计算的核心功能，没有numeric_only参数
        ("pct_change", False),  # 使用百分比变化计算的核心功能，没有numeric_only参数
        ("prod", True),  # 使用乘积计算的核心功能，并有numeric_only参数
        ("quantile", True),  # 使用分位数计算的核心功能，并有numeric_only参数
        ("sem", True),  # 使用标准误差计算的核心功能，并有numeric_only参数
        ("skew", True),  # 使用偏度计算的核心功能，并有numeric_only参数
        ("std", True),  # 使用标准差计算的核心功能，并有numeric_only参数
        ("sum", True),  # 使用求和计算的核心功能，并有numeric_only参数
        ("var", True),  # 使用方差计算的核心功能，并有numeric_only参数
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False, lib.no_default])  # 参数化测试numeric_only的三种可能取值
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])  # 参数化测试groupby操作的keys参数
def test_numeric_only(kernel, has_arg, numeric_only, keys):
    # GH#46072
    # drops_nuisance: Whether the op drops nuisance columns even when numeric_only=False
    # has_arg: Whether the op has a numeric_only arg

    # 创建一个DataFrame，包含多列数据，其中a1和a2列包含数值，a3列包含对象
    df = DataFrame({"a1": [1, 1], "a2": [2, 2], "a3": [5, 6], "b": 2 * [object]})

    # 调用函数获取groupby方法的参数
    args = get_groupby_method_args(kernel, df)
    # 如果numeric_only不是lib.no_default，则将其作为关键字参数传递给kwargs
    kwargs = {} if numeric_only is lib.no_default else {"numeric_only": numeric_only}

    # 对DataFrame进行分组操作
    gb = df.groupby(keys)
    # 获取groupby对象中相应核心函数的方法
    method = getattr(gb, kernel)

    # 如果核心函数有numeric_only参数，并且numeric_only为True
    if has_arg and numeric_only is True:
        # 对于核心函数为"corrwith"时，发出未来警告
        if kernel == "corrwith":
            warn = FutureWarning
            msg = "DataFrameGroupBy.corrwith is deprecated"
        else:
            warn = None
            msg = ""
        # 使用assert_produces_warning上下文管理器来检查是否生成了特定警告消息
        with tm.assert_produces_warning(warn, match=msg):
            # 调用方法并获取结果
            result = method(*args, **kwargs)
        # 断言结果中不包含'b'列
        assert "b" not in result.columns
    # 否则，如果核心函数适用于任何数据类型并且没有numeric_only参数
    elif (
        kernel in ("first", "last")
        or (
            kernel in ("any", "all", "bfill", "ffill", "fillna", "nth", "nunique")
            and numeric_only is lib.no_default
        )
    ):
        # 对于核心函数为"fillna"时，发出未来警告
        warn = FutureWarning if kernel == "fillna" else None
        msg = "DataFrameGroupBy.fillna is deprecated"
        # 使用assert_produces_warning上下文管理器来检查是否生成了特定警告消息
        with tm.assert_produces_warning(warn, match=msg):
            # 调用方法并获取结果
            result = method(*args, **kwargs)
        # 断言结果中包含'b'列
        assert "b" in result.columns
    # 如果存在参数，执行以下逻辑
    elif has_arg:
        # 断言 numeric_only 不是 True
        assert numeric_only is not True
        # 如果 kernel 以 "cum" 开头，则抛出 NotImplementedError，否则抛出 TypeError
        # 对于对象类型的转换，在 Cython 中未实现，也没有 Python 的回退
        exception = NotImplementedError if kernel.startswith("cum") else TypeError
        
        # 构造异常匹配的消息内容
        msg = "|".join(
            [
                "not allowed for this dtype",  # 此 dtype 不支持此操作
                "cannot be performed against 'object' dtypes",  # 不能对 'object' dtype 执行此操作
                "must be a string or a real number",  # 必须是字符串或实数
                "unsupported operand type",  # 不支持的操作数类型
                "function is not implemented for this dtype",  # 此 dtype 尚未实现该函数
                re.escape(f"agg function failed [how->{kernel},dtype->object]"),  # 聚合函数失败消息
            ]
        )
        
        # 如果 kernel 是 "idxmin"，则修改消息为指定的错误信息
        if kernel == "idxmin":
            msg = "'<' not supported between instances of 'type' and 'type'"
        # 如果 kernel 是 "idxmax"，则修改消息为指定的错误信息
        elif kernel == "idxmax":
            msg = "'>' not supported between instances of 'type' and 'type'"
        
        # 使用 pytest.raises 检查是否抛出特定类型的异常，并匹配特定消息
        with pytest.raises(exception, match=msg):
            # 如果 kernel 是 "corrwith"，设置警告为 FutureWarning，消息为相关警告信息
            if kernel == "corrwith":
                warn = FutureWarning
                msg = "DataFrameGroupBy.corrwith is deprecated"
            else:
                warn = None
                msg = ""
            # 使用 tm.assert_produces_warning 检查是否产生特定警告，并匹配特定消息
            with tm.assert_produces_warning(warn, match=msg):
                # 执行特定的方法调用，传入参数和关键字参数
                method(*args, **kwargs)
    
    # 如果不存在参数，并且 numeric_only 不是 lib.no_default
    elif not has_arg and numeric_only is not lib.no_default:
        # 使用 pytest.raises 检查是否抛出 TypeError 异常，匹配特定消息
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'numeric_only'"
        ):
            # 执行特定的方法调用，传入参数和关键字参数
            method(*args, **kwargs)
    
    # 其他情况
    else:
        # 断言 kernel 是 "diff" 或 "pct_change"
        assert kernel in ("diff", "pct_change")
        # 断言 numeric_only 是 lib.no_default
        assert numeric_only is lib.no_default
        # 没有 numeric_only 参数，且在无用列上失败
        # 使用 pytest.raises 检查是否抛出 TypeError 异常，匹配特定消息
        with pytest.raises(TypeError, match=r"unsupported operand type"):
            # 执行特定的方法调用，传入参数和关键字参数
            method(*args, **kwargs)
# 使用 pytest.mark.parametrize 注解为 test_deprecate_numeric_only_series 函数添加参数 dtype，分别测试 bool、int、float 和 object 类型
@pytest.mark.parametrize("dtype", [bool, int, float, object])
def test_deprecate_numeric_only_series(dtype, groupby_func, request):
    # GH#46560
    # 创建一个用于分组的列表
    grouper = [0, 0, 1]

    # 创建一个 Series 对象 ser，包含数值 [1, 0, 0]，使用给定的 dtype
    ser = Series([1, 0, 0], dtype=dtype)
    # 对 ser 应用 groupby 方法，使用 grouper 进行分组
    gb = ser.groupby(grouper)

    # 如果 groupby_func 是 "corrwith"，则断言 SeriesGroupBy 对象 gb 不包含属性 groupby_func，因为 corrwith 方法在 SeriesGroupBy 上没有实现
    if groupby_func == "corrwith":
        assert not hasattr(gb, groupby_func)
        return

    # 获取 gb 对象的特定方法，该方法由 groupby_func 指定
    method = getattr(gb, groupby_func)

    # 创建一个期望的 Series 对象 expected_ser，其数值与 ser 相同
    expected_ser = Series([1, 0, 0])
    # 对 expected_ser 应用 groupby 方法，使用 grouper 进行分组
    expected_gb = expected_ser.groupby(grouper)
    # 获取期望的 gb 对象的特定方法，该方法由 groupby_func 指定
    expected_method = getattr(expected_gb, groupby_func)

    # 获取用于调用 method 方法的参数
    args = get_groupby_method_args(groupby_func, ser)

    # 定义对于 object 类型输入而言会失败的操作集合
    fails_on_numeric_object = (
        "corr",
        "cov",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "quantile",
    )
    # 对于给定对象输入会产生对象结果的操作集合
    obj_result = (
        "first",
        "last",
        "nth",
        "bfill",
        "ffill",
        "shift",
        "sum",
        "diff",
        "pct_change",
        "var",
        "mean",
        "median",
        "min",
        "max",
        "prod",
        "skew",
    )

    # 测试默认行为；未来可能会启用失败的内核，但不应允许成功的内核失败（至少不应该没有废弃）
    # 如果 groupby_func 在 fails_on_numeric_object 中，并且 dtype 是 object 类型
    if groupby_func in fails_on_numeric_object and dtype is object:
        if groupby_func == "quantile":
            msg = "cannot be performed against 'object' dtypes"
        else:
            msg = "is not supported for object dtype"
        # 使用 pytest.raises 断言会抛出 TypeError 异常，并且匹配特定的错误消息 msg
        with pytest.raises(TypeError, match=msg):
            method(*args)
    # 如果 dtype 是 object 类型
    elif dtype is object:
        # 调用 method 方法，并将结果存储在 result 变量中
        result = method(*args)
        # 调用期望的方法 expected_method，并将结果存储在 expected 变量中
        expected = expected_method(*args)
        # 如果 groupby_func 在 obj_result 集合中，则将 expected 转换为 object 类型
        if groupby_func in obj_result:
            expected = expected.astype(object)
        # 使用 tm.assert_series_equal 比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 定义仅适用于数值输入的操作集合
    has_numeric_only = (
        "first",
        "last",
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
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
    )
    # 如果 groupby_func 不在 has_numeric_only 集合中
    if groupby_func not in has_numeric_only:
        # 断言会抛出 TypeError 异常，并且匹配特定的错误消息 msg
        msg = "got an unexpected keyword argument 'numeric_only'"
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    # 如果 dtype 是 object 类型
    elif dtype is object:
        # 定义一个包含多个错误消息的正则表达式模式 msg
        msg = "|".join(
            [
                "SeriesGroupBy.sem called with numeric_only=True and dtype object",
                "Series.skew does not allow numeric_only=True with non-numeric",
                "cum(sum|prod|min|max) is not supported for object dtype",
                r"Cannot use numeric_only=True with SeriesGroupBy\..* and non-numeric",
            ]
        )
        # 使用 pytest.raises 断言会抛出 TypeError 异常，并且匹配特定的错误消息 msg
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    # 如果数据类型是布尔型且分组函数为“quantile”，则抛出类型错误并匹配特定消息
    elif dtype == bool and groupby_func == "quantile":
        msg = "Cannot use quantile with bool dtype"
        with pytest.raises(TypeError, match=msg):
            # 在特定 GitHub 问题号 #51424 的上下文中，调用指定方法并传入参数，确保在 numeric_only=False 的情况下抛出异常
            method(*args, numeric_only=False)
    else:
        # 否则，以 numeric_only=True 的模式调用方法，获取结果
        result = method(*args, numeric_only=True)
        # 再以 numeric_only=False 的模式调用方法，获取预期结果
        expected = method(*args, numeric_only=False)
        # 使用测试工具 (tm) 检查返回的结果和预期结果是否相等
        tm.assert_series_equal(result, expected)
```