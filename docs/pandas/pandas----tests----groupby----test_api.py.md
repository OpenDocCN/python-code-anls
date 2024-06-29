# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_api.py`

```
"""
Tests of the groupby API, including internal consistency and with other pandas objects.

Tests in this file should only check the existence, names, and arguments of groupby
methods. It should not test the results of any groupby operation.
"""

# 导入必要的模块和库
import inspect  # 导入inspect模块，用于检查对象的属性和方法

import pytest  # 导入pytest模块，用于编写和运行测试

from pandas import (  # 从pandas库中导入DataFrame和Series类
    DataFrame,
    Series,
)
from pandas.core.groupby.base import (  # 从pandas核心模块中导入groupby相关的基础方法
    groupby_other_methods,
    reduction_kernels,
    transformation_kernels,
)
from pandas.core.groupby.generic import (  # 从pandas核心模块中导入groupby相关的泛化方法
    DataFrameGroupBy,
    SeriesGroupBy,
)


def test_tab_completion(multiindex_dataframe_random_data):
    # 对多级索引的随机数据框进行分组，根据'second'级别进行分组
    grp = multiindex_dataframe_random_data.groupby(level="second")
    # 获取分组对象的所有非下划线开头的属性名称集合
    results = {v for v in dir(grp) if not v.startswith("_")}
    # 预期的分组属性名称集合
    expected = {
        "A",
        "B",
        "C",
        "agg",
        "aggregate",
        "apply",
        "boxplot",
        "filter",
        "first",
        "get_group",
        "groups",
        "hist",
        "indices",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "ngroups",
        "nth",
        "ohlc",
        "plot",
        "prod",
        "size",
        "std",
        "sum",
        "transform",
        "var",
        "sem",
        "count",
        "nunique",
        "head",
        "describe",
        "cummax",
        "quantile",
        "rank",
        "cumprod",
        "tail",
        "resample",
        "cummin",
        "cumsum",
        "cumcount",
        "ngroup",
        "all",
        "shift",
        "skew",
        "take",
        "pct_change",
        "any",
        "corr",
        "corrwith",
        "cov",
        "ndim",
        "diff",
        "idxmax",
        "idxmin",
        "ffill",
        "bfill",
        "rolling",
        "expanding",
        "pipe",
        "sample",
        "ewm",
        "value_counts",
    }
    # 断言实际的属性名称集合与预期的完全一致
    assert results == expected


def test_all_methods_categorized(multiindex_dataframe_random_data):
    # 根据数据框的第一列进行分组
    grp = multiindex_dataframe_random_data.groupby(
        multiindex_dataframe_random_data.iloc[:, 0]
    )
    # 获取分组对象的所有属性名称集合，排除数据框列名
    names = {_ for _ in dir(grp) if not _.startswith("_")} - set(
        multiindex_dataframe_random_data.columns
    )
    # 创建新的属性名称集合，排除已分类的核心方法
    new_names = set(names)
    new_names -= reduction_kernels
    new_names -= transformation_kernels
    new_names -= groupby_other_methods

    # 断言核心方法之间没有重叠
    assert not reduction_kernels & transformation_kernels
    assert not reduction_kernels & groupby_other_methods
    assert not transformation_kernels & groupby_other_methods

    # 如果存在新的公共方法，则抛出断言错误
    if new_names:
        msg = f"""
There are uncategorized methods defined on the Grouper class:
{new_names}.

Was a new method recently added?

Every public method On Grouper must appear in exactly one the
following three lists defined in pandas.core.groupby.base:
- `reduction_kernels`
- `transformation_kernels`
- `groupby_other_methods`
see the comments in pandas/core/groupby/base.py for guidance on
how to fix this test.
        """
        raise AssertionError(msg)
    # 将三个集合合并为一个新的集合，包含了所有归约内核、转换内核和其他分组方法的名称
    all_categorized = reduction_kernels | transformation_kernels | groupby_other_methods
    # 检查给定的名称集合是否与合并后的集合不同
    if names != all_categorized:
        # 构建包含错误消息的多行字符串，其中包含了占位符
        msg = f"""
"""
Some methods which are supposed to be on the Grouper class
are missing:
{all_categorized - names}.

They're still defined in one of the lists that live in pandas/core/groupby/base.py.
If you removed a method, you should update them
"""
# 如果在 Grouper 类中缺少了一些预期的方法，会引发此异常
raise AssertionError(msg)


def test_frame_consistency(groupby_func):
    # GH#48028
    # 如果 groupby_func 是 "first" 或者 "last"，跳过测试并给出相应的消息
    if groupby_func in ("first", "last"):
        msg = "first and last don't exist for DataFrame anymore"
        pytest.skip(reason=msg)

    # 如果 groupby_func 是 "cumcount" 或者 "ngroup"，则检查 DataFrame 是否没有此方法
    if groupby_func in ("cumcount", "ngroup"):
        assert not hasattr(DataFrame, groupby_func)
        return

    # 获取 DataFrame 中的 frame_method 和 DataFrameGroupBy 中的 gb_method
    frame_method = getattr(DataFrame, groupby_func)
    gb_method = getattr(DataFrameGroupBy, groupby_func)

    # 使用反射机制获取 gb_method 方法的参数列表，并转换成集合
    result = set(inspect.signature(gb_method).parameters)

    # 对于不同的 groupby_func，预期的参数列表也会有所不同
    if groupby_func == "size":
        # 对于 "size" 方法，DataFrame 中的预期参数是 {"self"}
        expected = {"self"}
    else:
        # 对于其他方法，使用反射机制获取 frame_method 方法的参数列表，并转换成集合
        expected = set(inspect.signature(frame_method).parameters)

    # 根据 groupby_func 的不同，排除特定的参数来匹配预期结果和实际结果
    exclude_expected, exclude_result = set(), set()
    if groupby_func in ("any", "all"):
        exclude_expected = {"kwargs", "bool_only", "axis"}
    elif groupby_func in ("count",):
        exclude_expected = {"numeric_only", "axis"}
    elif groupby_func in ("nunique",):
        exclude_expected = {"axis"}
    elif groupby_func in ("max", "min"):
        exclude_expected = {"axis", "kwargs", "skipna"}
        exclude_result = {"min_count", "engine", "engine_kwargs"}
    elif groupby_func in ("mean", "std", "sum", "var"):
        exclude_expected = {"axis", "kwargs", "skipna"}
        exclude_result = {"engine", "engine_kwargs"}
    elif groupby_func in ("median", "prod", "sem"):
        exclude_expected = {"axis", "kwargs", "skipna"}
    elif groupby_func in ("bfill", "ffill"):
        exclude_expected = {"inplace", "axis", "limit_area"}
    elif groupby_func in ("cummax", "cummin"):
        exclude_expected = {"axis", "skipna", "args"}
    elif groupby_func in ("cumprod", "cumsum"):
        exclude_expected = {"axis", "skipna", "numeric_only"}
    elif groupby_func in ("pct_change",):
        exclude_expected = {"kwargs"}
    elif groupby_func in ("rank",):
        exclude_expected = {"numeric_only"}
    elif groupby_func in ("quantile",):
        exclude_expected = {"method", "axis"}
    elif groupby_func in ["corrwith"]:
        exclude_expected = {"min_periods"}

    # 如果 groupby_func 不是 "pct_change" 或者 "size"，则排除参数 "axis"
    if groupby_func not in ["pct_change", "size"]:
        exclude_expected |= {"axis"}

    # 确保排除的参数确实在签名中存在
    assert result & exclude_result == exclude_result
    assert expected & exclude_expected == exclude_expected

    # 更新 result 和 expected，移除排除的参数后再进行比较
    result -= exclude_result
    expected -= exclude_expected
    assert result == expected
    # 检查 groupby_func 是否为 "first" 或者 "last"
    if groupby_func in ("first", "last"):
        # 如果是，则生成跳过测试的消息
        msg = "first and last don't exist for Series anymore"
        # 使用 pytest 跳过测试，并传入消息
        pytest.skip(msg)

    # 检查 groupby_func 是否为 "cumcount", "corrwith", "ngroup" 中的一个
    if groupby_func in ("cumcount", "corrwith", "ngroup"):
        # 断言 Series 没有名为 groupby_func 的属性
        assert not hasattr(Series, groupby_func)
        # 直接返回，不再继续执行后面的代码
        return

    # 获取 Series 类中名为 groupby_func 的方法对象
    series_method = getattr(Series, groupby_func)
    # 获取 SeriesGroupBy 类中名为 groupby_func 的方法对象
    gb_method = getattr(SeriesGroupBy, groupby_func)
    # 使用 inspect 模块获取 gb_method 方法的参数集合
    result = set(inspect.signature(gb_method).parameters)

    # 如果 groupby_func 是 "size"，则预期参数集合为 {"self"}
    if groupby_func == "size":
        expected = {"self"}
    else:
        # 否则，使用 inspect 模块获取 series_method 方法的参数集合
        expected = set(inspect.signature(series_method).parameters)

    # 根据 groupby_func 的不同值，排除特定的参数
    exclude_expected, exclude_result = set(), set()
    if groupby_func in ("any", "all"):
        exclude_expected = {"kwargs", "bool_only", "axis"}
    elif groupby_func in ("max", "min"):
        exclude_expected = {"axis", "kwargs", "skipna"}
        exclude_result = {"min_count", "engine", "engine_kwargs"}
    elif groupby_func in ("mean", "std", "sum", "var"):
        exclude_expected = {"axis", "kwargs", "skipna"}
        exclude_result = {"engine", "engine_kwargs"}
    elif groupby_func in ("median", "prod", "sem"):
        exclude_expected = {"axis", "kwargs", "skipna"}
    elif groupby_func in ("bfill", "ffill"):
        exclude_expected = {"inplace", "axis", "limit_area"}
    elif groupby_func in ("cummax", "cummin"):
        exclude_expected = {"skipna", "args"}
        exclude_result = {"numeric_only"}
    elif groupby_func in ("cumprod", "cumsum"):
        exclude_expected = {"skipna"}
    elif groupby_func in ("pct_change",):
        exclude_expected = {"kwargs"}
    elif groupby_func in ("rank",):
        exclude_expected = {"numeric_only"}
    elif groupby_func in ("idxmin", "idxmax"):
        exclude_expected = {"args", "kwargs"}
    elif groupby_func in ("quantile",):
        exclude_result = {"numeric_only"}

    # 如果 groupby_func 不在以下列表中，则将 "axis" 参数也添加到排除的预期参数集合中
    if groupby_func not in [
        "diff",
        "pct_change",
        "count",
        "nunique",
        "quantile",
        "size",
    ]:
        exclude_expected |= {"axis"}

    # 断言排除的结果参数确实在结果集合中
    assert result & exclude_result == exclude_result
    # 断言排除的预期参数确实在预期集合中
    assert expected & exclude_expected == exclude_expected

    # 从结果集合中移除排除的结果参数
    result -= exclude_result
    # 从预期集合中移除排除的预期参数
    expected -= exclude_expected
    # 断言结果集合等于预期集合，以确保参数匹配
    assert result == expected
```