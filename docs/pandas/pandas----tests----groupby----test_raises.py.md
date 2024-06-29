# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_raises.py`

```
# 导入必要的模块和库
import datetime  # 导入datetime模块，用于处理日期和时间
import re  # 导入re模块，用于正则表达式操作

import numpy as np  # 导入numpy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas import (  # 从pandas库中导入以下子模块或类
    Categorical,  # 类别数据类型，用于有限的可能取值
    DataFrame,  # 数据框，用于处理二维数据
    Grouper,  # 分组器，用于按指定条件分组数据
    Series,  # 系列，用于处理一维数据
)
import pandas._testing as tm  # 导入pandas内部测试工具模块
from pandas.tests.groupby import get_groupby_method_args  # 从pandas测试模块中导入函数

# 定义参数化测试用例的装饰器，分别对应不同的分组方法
@pytest.fixture(
    params=[
        "a",  # 单个字符串
        ["a"],  # 字符串列表
        ["a", "b"],  # 多个字符串列表
        Grouper(key="a"),  # Grouper对象，按'a'键分组
        lambda x: x % 2,  # 匿名函数，对输入取模2
        [0, 0, 0, 1, 2, 2, 2, 3, 3],  # 数字列表
        np.array([0, 0, 0, 1, 2, 2, 2, 3, 3]),  # NumPy数组
        dict(zip(range(9), [0, 0, 0, 1, 2, 2, 2, 3, 3])),  # 字典，键值对
        Series([1, 1, 1, 1, 1, 2, 2, 2, 2]),  # Pandas Series对象
        [Series([1, 1, 1, 1, 1, 2, 2, 2, 2]), Series([3, 3, 4, 4, 4, 4, 4, 3, 3])],  # Series列表
    ]
)
def by(request):
    return request.param  # 返回参数化测试用例的参数值


# 定义参数化测试用例的装饰器，测试分组是否为Series对象
@pytest.fixture(params=[True, False])
def groupby_series(request):
    return request.param  # 返回参数化测试用例的参数值


# 定义数据框，包含一个字符串列的测试数据
@pytest.fixture
def df_with_string_col():
    df = DataFrame(  # 创建数据框对象
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],  # 列'a'的数据
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],  # 列'b'的数据
            "c": range(9),  # 生成0到8的整数，列'c'的数据
            "d": list("xyzwtyuio"),  # 列'd'的数据，为字符列表转换成的字符串
        }
    )
    return df  # 返回数据框对象


# 定义数据框，包含一个日期时间列的测试数据
@pytest.fixture
def df_with_datetime_col():
    df = DataFrame(  # 创建数据框对象
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],  # 列'a'的数据
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],  # 列'b'的数据
            "c": range(9),  # 生成0到8的整数，列'c'的数据
            "d": datetime.datetime(2005, 1, 1, 10, 30, 23, 540000),  # 列'd'的数据，为指定日期时间
        }
    )
    return df  # 返回数据框对象


# 定义数据框，包含一个分类列的测试数据
@pytest.fixture
def df_with_cat_col():
    df = DataFrame(  # 创建数据框对象
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],  # 列'a'的数据
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],  # 列'b'的数据
            "c": range(9),  # 生成0到8的整数，列'c'的数据
            "d": Categorical(  # 列'd'的数据，为分类类型数据
                ["a", "a", "a", "a", "b", "b", "b", "b", "c"],  # 分类数据的值
                categories=["a", "b", "c", "d"],  # 分类数据的类别
                ordered=True,  # 分类数据是否有序
            ),
        }
    )
    return df  # 返回数据框对象


# 定义用于测试的函数，用于检查方法调用是否引发了预期的异常
def _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg=""):
    warn_klass = None if warn_msg == "" else FutureWarning  # 如果警告消息为空，则warn_klass为None，否则为FutureWarning类
    with tm.assert_produces_warning(warn_klass, match=warn_msg, check_stacklevel=False):  # 使用tm.assert_produces_warning断言，检查是否产生警告
        if klass is None:  # 如果klass为None
            if how == "method":  # 如果how为'method'
                getattr(gb, groupby_func)(*args)  # 调用gb对象的groupby_func方法，传递args作为参数
            elif how == "agg":  # 如果how为'agg'
                gb.agg(groupby_func, *args)  # 使用gb对象的agg方法，传递groupby_func和args作为参数
            else:  # 否则
                gb.transform(groupby_func, *args)  # 使用gb对象的transform方法，传递groupby_func和args作为参数
        else:  # 如果klass不为None
            with pytest.raises(klass, match=msg):  # 使用pytest.raises断言，检查是否引发klass类型的异常，并匹配msg字符串
                if how == "method":  # 如果how为'method'
                    getattr(gb, groupby_func)(*args)  # 调用gb对象的groupby_func方法，传递args作为参数
                elif how == "agg":  # 如果how为'agg'
                    gb.agg(groupby_func, *args)  # 使用gb对象的agg方法，传递groupby_func和args作为参数
                else:  # 否则
                    gb.transform(groupby_func, *args)  # 使用gb对象的transform方法，传递groupby_func和args作为参数


# 参数化测试用例，测试groupby方法在不同how下是否引发字符串列异常
@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_string(
    how, by, groupby_series, groupby_func, df_with_string_col
):
    df = df_with_string_col  # 使用包含字符串列的数据框对象df_with_string_col
    args = get_groupby_method_args(groupby_func, df)  # 调用get_groupby_method_args函数，获取groupby方法的参数
    gb = df.groupby(by=by)  # 对数据框df按by进行分组
    # 如果 groupby_series 为真，则对 gb 使用 "d" 字段进行处理
    if groupby_series:
        gb = gb["d"]

        # 如果 groupby_func 为 "corrwith"，则检查 gb 没有 "corrwith" 属性，否则返回
        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    # 根据 groupby_func 的不同取值，选择不同的 klass 和 msg
    klass, msg = {
        "all": (None, ""),
        "any": (None, ""),
        "bfill": (None, ""),
        "corrwith": (TypeError, "Could not convert"),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (
            (NotImplementedError, TypeError),
            "(function|cummax) is not (implemented|supported) for (this|object) dtype",
        ),
        "cummin": (
            (NotImplementedError, TypeError),
            "(function|cummin) is not (implemented|supported) for (this|object) dtype",
        ),
        "cumprod": (
            (NotImplementedError, TypeError),
            "(function|cumprod) is not (implemented|supported) for (this|object) dtype",
        ),
        "cumsum": (
            (NotImplementedError, TypeError),
            "(function|cumsum) is not (implemented|supported) for (this|object) dtype",
        ),
        "diff": (TypeError, "unsupported operand type"),
        "ffill": (None, ""),
        "fillna": (None, ""),
        "first": (None, ""),
        "idxmax": (None, ""),
        "idxmin": (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (
            TypeError,
            re.escape("agg function failed [how->mean,dtype->object]"),
        ),
        "median": (
            TypeError,
            re.escape("agg function failed [how->median,dtype->object]"),
        ),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (TypeError, "unsupported operand type"),
        "prod": (
            TypeError,
            re.escape("agg function failed [how->prod,dtype->object]"),
        ),
        "quantile": (TypeError, "cannot be performed against 'object' dtypes!"),
        "rank": (None, ""),
        "sem": (ValueError, "could not convert string to float"),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (ValueError, "could not convert string to float"),
        "std": (ValueError, "could not convert string to float"),
        "sum": (None, ""),
        "var": (
            TypeError,
            re.escape("agg function failed [how->var,dtype->"),
        ),
    }[groupby_func]

    # 根据 groupby_func 的值，设置 warn_msg 不同的警告信息
    if groupby_func == "fillna":
        kind = "Series" if groupby_series else "DataFrame"
        warn_msg = f"{kind}GroupBy.fillna is deprecated"
    elif groupby_func == "corrwith":
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn_msg = ""
    
    # 调用 _call_and_check 函数，传入相应的参数进行调用和检查
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)
# 使用 pytest 的 parametrize 装饰器为 test_groupby_raises_string_udf 函数参数化，how 参数可以是 "agg" 或 "transform"
@pytest.mark.parametrize("how", ["agg", "transform"])
# 定义测试函数 test_groupby_raises_string_udf，测试 groupby 对象应用自定义函数时引发异常的情况
def test_groupby_raises_string_udf(how, by, groupby_series, df_with_string_col):
    # 使用 df_with_string_col 初始化 DataFrame df
    df = df_with_string_col
    # 根据 by 参数对 df 进行分组操作，生成 GroupBy 对象 gb
    gb = df.groupby(by=by)

    # 如果 groupby_series 为真，则对 gb 只保留 "d" 列的分组结果
    if groupby_series:
        gb = gb["d"]

    # 定义一个会抛出 TypeError 的自定义函数 func
    def func(x):
        raise TypeError("Test error message")

    # 使用 pytest 的 raises 断言捕获 TypeError 异常，并验证异常消息为 "Test error message"
    with pytest.raises(TypeError, match="Test error message"):
        # 调用 gb 对象的 how 方法（如 agg 或 transform），传入自定义函数 func
        getattr(gb, how)(func)


# 使用 pytest 的 parametrize 装饰器为 test_groupby_raises_string_np 函数参数化，how 参数可以是 "agg" 或 "transform"
@pytest.mark.parametrize("how", ["agg", "transform"])
# 使用 parametrize 装饰器为 groupby_func_np 参数参数化，可以是 np.sum 或 np.mean
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
# 定义测试函数 test_groupby_raises_string_np，测试 groupby 对象应用 NumPy 函数时引发异常的情况
def test_groupby_raises_string_np(
    how, by, groupby_series, groupby_func_np, df_with_string_col
):
    # GH#50749
    # 使用 df_with_string_col 初始化 DataFrame df
    df = df_with_string_col
    # 根据 by 参数对 df 进行分组操作，生成 GroupBy 对象 gb
    gb = df.groupby(by=by)

    # 如果 groupby_series 为真，则对 gb 只保留 "d" 列的分组结果
    if groupby_series:
        gb = gb["d"]

    # 根据 groupby_func_np 的值选择对应的异常类型 klass 和异常消息 msg
    klass, msg = {
        np.sum: (None, ""),
        np.mean: (
            TypeError,
            "Could not convert string .* to numeric",
        ),
    }[groupby_func_np]
    
    # 调用 _call_and_check 函数，验证对 gb 对象应用 how 方法（如 agg 或 transform）时是否会引发预期的异常
    _call_and_check(klass, msg, how, gb, groupby_func_np, ())


# 使用 pytest 的 parametrize 装饰器为 test_groupby_raises_datetime 函数参数化，how 参数可以是 "method"、"agg" 或 "transform"
@pytest.mark.parametrize("how", ["method", "agg", "transform"])
# 定义测试函数 test_groupby_raises_datetime，测试 groupby 对象应用日期时间函数时的特定情况
def test_groupby_raises_datetime(
    how, by, groupby_series, groupby_func, df_with_datetime_col
):
    # 使用 df_with_datetime_col 初始化 DataFrame df
    df = df_with_datetime_col
    # 根据 groupby_func 和 df 获取相应的分组方法参数 args
    args = get_groupby_method_args(groupby_func, df)
    # 根据 by 参数对 df 进行分组操作，生成 GroupBy 对象 gb
    gb = df.groupby(by=by)

    # 如果 groupby_series 为真，则对 gb 只保留 "d" 列的分组结果
    if groupby_series:
        gb = gb["d"]

    # 当 groupby_func 为 "corrwith" 时的特殊情况处理：验证 gb 对象是否没有 corrwith 属性，然后直接返回
    if groupby_func == "corrwith":
        assert not hasattr(gb, "corrwith")
        return
    # 根据 groupby_func 的值选择合适的异常类和异常消息
    klass, msg = {
        "all": (TypeError, "'all' with datetime64 dtypes is no longer supported"),
        "any": (TypeError, "'any' with datetime64 dtypes is no longer supported"),
        "bfill": (None, ""),
        "corrwith": (TypeError, "cannot perform __mul__ with this index type"),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (None, ""),
        "cummin": (None, ""),
        "cumprod": (TypeError, "datetime64 type does not support operation 'cumprod'"),
        "cumsum": (TypeError, "datetime64 type does not support operation 'cumsum'"),
        "diff": (None, ""),
        "ffill": (None, ""),
        "fillna": (None, ""),
        "first": (None, ""),
        "idxmax": (None, ""),
        "idxmin": (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (None, ""),
        "median": (None, ""),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (TypeError, "cannot perform __truediv__ with this index type"),
        "prod": (TypeError, "datetime64 type does not support operation 'prod'"),
        "quantile": (None, ""),
        "rank": (None, ""),
        "sem": (None, ""),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (
            TypeError,
            "|".join(
                [
                    r"dtype datetime64\[ns\] does not support operation",
                    "datetime64 type does not support operation 'skew'",
                ]
            ),
        ),
        "std": (None, ""),
        "sum": (TypeError, "datetime64 type does not support operation 'sum"),
        "var": (TypeError, "datetime64 type does not support operation 'var'"),
    }[groupby_func]

    # 处理特定的警告消息
    if groupby_func == "fillna":
        # 根据 groupby_series 决定警告消息的种类
        kind = "Series" if groupby_series else "DataFrame"
        warn_msg = f"{kind}GroupBy.fillna is deprecated"
    elif groupby_func == "corrwith":
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn_msg = ""

    # 调用 _call_and_check 函数，传递异常类、异常消息、处理方式、分组对象、分组函数、额外参数和警告消息
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg=warn_msg)
@pytest.mark.parametrize("how", ["agg", "transform"])
# 定义测试函数，用于测试 groupby 对象调用 'agg' 或 'transform' 方法时抛出异常
def test_groupby_raises_datetime_udf(how, by, groupby_series, df_with_datetime_col):
    df = df_with_datetime_col
    # 根据指定列或列组进行分组
    gb = df.groupby(by=by)

    # 如果是按照单列分组，则只选择 'd' 列
    if groupby_series:
        gb = gb["d"]

    # 定义一个抛出异常的函数
    def func(x):
        raise TypeError("Test error message")

    # 使用 pytest 检查是否抛出指定异常和消息
    with pytest.raises(TypeError, match="Test error message"):
        getattr(gb, how)(func)


@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
# 定义测试函数，用于测试 groupby 对象调用 'agg' 或 'transform' 方法时抛出异常
def test_groupby_raises_datetime_np(
    how, by, groupby_series, groupby_func_np, df_with_datetime_col
):
    # GH#50749
    df = df_with_datetime_col
    # 根据指定列或列组进行分组
    gb = df.groupby(by=by)

    # 如果是按照单列分组，则只选择 'd' 列
    if groupby_series:
        gb = gb["d"]

    # 定义期望抛出异常类型和消息的字典
    klass, msg = {
        np.sum: (
            TypeError,
            re.escape("datetime64[us] does not support operation 'sum'"),
        ),
        np.mean: (None, ""),
    }[groupby_func_np]

    # 调用辅助函数来检查是否抛出期望的异常
    _call_and_check(klass, msg, how, gb, groupby_func_np, ())


@pytest.mark.parametrize("func", ["prod", "cumprod", "skew", "var"])
# 定义测试函数，用于测试 groupby 对象调用不支持的函数时抛出异常
def test_groupby_raises_timedelta(func):
    # 创建包含时间差数据的 DataFrame
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": datetime.timedelta(days=1),
        }
    )
    # 根据指定列 'a' 进行分组
    gb = df.groupby(by="a")

    # 调用辅助函数来检查是否抛出期望的异常
    _call_and_check(
        TypeError,
        "timedelta64 type does not support .* operations",
        "method",
        gb,
        func,
        [],
    )


@pytest.mark.parametrize("how", ["method", "agg", "transform"])
# 定义测试函数，用于测试 groupby 对象调用不支持的函数时抛出异常
def test_groupby_raises_category(
    how, by, groupby_series, groupby_func, df_with_cat_col
):
    # GH#50749
    df = df_with_cat_col
    # 根据指定列或列组进行分组
    gb = df.groupby(by=by)

    # 如果是按照单列分组，则只选择 'd' 列
    if groupby_series:
        gb = gb["d"]

        # 对于 'corrwith' 函数，如果对象没有 'corrwith' 属性，则直接返回
        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    # 根据不同的 groupby 函数选择相应的异常类型和消息
    klass, msg = {
        groupby_func: (
            TypeError,
            re.escape(f"{groupby_func} is not supported for categorical data"),
        ),
        "fillna": (FutureWarning, ""),
    }[groupby_func]

    # 如果是 'fillna' 函数，生成相应的警告消息
    if groupby_func == "fillna":
        kind = "Series" if groupby_series else "DataFrame"
        warn_msg = f"{kind}GroupBy.fillna is deprecated"
    elif groupby_func == "corrwith":
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn_msg = ""

    # 调用辅助函数来检查是否抛出期望的异常
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)


@pytest.mark.parametrize("how", ["agg", "transform"])
# 定义测试函数，用于测试 groupby 对象调用 'agg' 或 'transform' 方法时抛出异常
def test_groupby_raises_category_udf(how, by, groupby_series, df_with_cat_col):
    # GH#50749
    df = df_with_cat_col
    # 根据指定列或列组进行分组
    gb = df.groupby(by=by)

    # 如果是按照单列分组，则只选择 'd' 列
    if groupby_series:
        gb = gb["d"]

    # 定义一个抛出异常的函数
    def func(x):
        raise TypeError("Test error message")

    # 使用 pytest 检查是否抛出指定异常和消息
    with pytest.raises(TypeError, match="Test error message"):
        getattr(gb, how)(func)


@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
# 定义测试函数，用于测试 groupby 对象调用 'agg' 或 'transform' 方法时抛出异常
def test_groupby_raises_category_np(
    how, by, groupby_series, groupby_func_np, df_with_cat_col
):
    # GH#50749
    df = df_with_cat_col
    # 根据指定的列名 `by` 对 DataFrame `df` 进行分组操作，并返回一个 GroupBy 对象
    gb = df.groupby(by=by)

    # 如果 `groupby_series` 为真值（True），则重新将 `gb` 对象限制在包含列名为 "d" 的分组结果上
    if groupby_series:
        gb = gb["d"]

    # 根据 `groupby_func_np` 所对应的函数选择合适的异常类型 `klass` 和错误消息 `msg`
    klass, msg = {
        np.sum: (TypeError, "dtype category does not support operation 'sum'"),
        np.mean: (
            TypeError,
            "dtype category does not support operation 'mean'",
        ),
    }[groupby_func_np]
    
    # 调用 `_call_and_check` 函数，检查在分组操作中是否发生了异常情况
    _call_and_check(klass, msg, how, gb, groupby_func_np, ())
@pytest.mark.parametrize("how", ["method", "agg", "transform"])
# 使用 pytest 的 parametrize 装饰器，定义测试参数化，参数名为 'how'，取值为 ["method", "agg", "transform"]

def test_groupby_raises_category_on_category(
    how,
    by,
    groupby_series,
    groupby_func,
    observed,
    df_with_cat_col,
):
    # GH#50749
    # 标识 GitHub issue 编号为 50749，可能与问题相关

    df = df_with_cat_col
    # 将 df_with_cat_col 赋值给 df，假设它是一个包含分类列的 DataFrame

    df["a"] = Categorical(
        ["a", "a", "a", "a", "b", "b", "b", "b", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    # 在 df 中创建一个名为 'a' 的分类列，设置其值为指定的列表，并指定分类的类别和顺序

    args = get_groupby_method_args(groupby_func, df)
    # 使用 get_groupby_method_args 函数获取根据 groupby_func 得到的分组方法的参数列表

    gb = df.groupby(by=by, observed=observed)
    # 根据给定的分组依据 by 和观察模式 observed 进行分组操作，返回 GroupBy 对象 gb

    if groupby_series:
        # 如果 groupby_series 为真（即非空）
        
        gb = gb["d"]
        # 将 gb 对象限制为仅包含 'd' 列的 GroupBy 对象

        if groupby_func == "corrwith":
            # 如果 groupby_func 为 "corrwith"
            
            assert not hasattr(gb, "corrwith")
            # 断言 gb 对象没有 'corrwith' 属性
            return
            # 测试完成，退出函数

    empty_groups = not observed and any(group.empty for group in gb.groups.values())
    # 计算是否存在空分组，如果 observed 为假且 gb 中任意分组为空，则 empty_groups 为真

    if how == "transform":
        # 如果 how 为 "transform"
        
        # 空分组将被忽略
        empty_groups = False

    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)
    # 调用 _call_and_check 函数，传递相应参数进行检查和处理
```