# `D:\src\scipysrc\pandas\pandas\conftest.py`

```
"""
This file is very long and growing, but it was decided to not split it yet, as
it's still manageable (2020-03-17, ~1.1k LoC). See gh-31989

Instead of splitting it was decided to define sections here:
- Configuration / Settings
- Autouse fixtures
- Common arguments
- Missing values & co.
- Classes
- Indices
- Series'
- DataFrames
- Operators & Operations
- Data sets/files
- Time zones
- Dtypes
- Misc
"""

from __future__ import annotations  # 导入使用未来版本的类型注解

from collections import abc  # 导入 collections 模块中的 abc 子模块
from datetime import (  # 从 datetime 模块导入多个对象
    date,
    datetime,
    time,
    timedelta,
    timezone,
)
from decimal import Decimal  # 导入 decimal 模块中的 Decimal 类
import gc  # 导入垃圾回收模块
import operator  # 导入 operator 模块
import os  # 导入操作系统相关模块
from typing import TYPE_CHECKING  # 导入类型提示模块中的 TYPE_CHECKING 类型

import uuid  # 导入 uuid 模块

from dateutil.tz import (  # 从 dateutil.tz 模块导入多个对象
    tzlocal,
    tzutc,
)
import hypothesis  # 导入 hypothesis 库
from hypothesis import strategies as st  # 导入 hypothesis 库中的 strategies 子模块，并命名为 st
import numpy as np  # 导入 numpy 库，并命名为 np
import pytest  # 导入 pytest 库
from pytz import (  # 从 pytz 模块导入多个对象
    FixedOffset,
    utc,
)

import pandas.util._test_decorators as td  # 导入 pandas.util._test_decorators 模块，并命名为 td

from pandas.core.dtypes.dtypes import (  # 从 pandas.core.dtypes.dtypes 模块导入多个对象
    DatetimeTZDtype,
    IntervalDtype,
)

import pandas as pd  # 导入 pandas 库，并命名为 pd
from pandas import (  # 从 pandas 库导入多个对象
    CategoricalIndex,
    DataFrame,
    Interval,
    IntervalIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm  # 导入 pandas._testing 模块，并命名为 tm
from pandas.core import ops  # 导入 pandas.core 中的 ops 模块
from pandas.core.indexes.api import (  # 从 pandas.core.indexes.api 模块导入多个对象
    Index,
    MultiIndex,
)
from pandas.util.version import Version  # 导入 pandas.util.version 模块中的 Version 类型

if TYPE_CHECKING:  # 如果 TYPE_CHECKING 为真
    from collections.abc import (  # 导入 collections.abc 模块中的多个对象
        Callable,
        Hashable,
        Iterator,
    )

try:
    import pyarrow as pa  # 尝试导入 pyarrow 库，并命名为 pa
except ImportError:
    has_pyarrow = False  # 如果导入失败，则将 has_pyarrow 设置为 False
else:
    del pa  # 如果导入成功，删除 pa 对象
    has_pyarrow = True  # 将 has_pyarrow 设置为 True

import zoneinfo  # 导入 zoneinfo 模块

try:
    zoneinfo.ZoneInfo("UTC")  # 尝试创建一个 "UTC" 时区对象
except zoneinfo.ZoneInfoNotFoundError:
    zoneinfo = None  # 如果时区未找到，则将 zoneinfo 设置为 None（忽略类型检查）

# ----------------------------------------------------------------
# Configuration / Settings
# ----------------------------------------------------------------

# pytest


def pytest_addoption(parser) -> None:
    """
    添加 pytest 的命令行选项。

    Parameters
    ----------
    parser : object
        pytest 命令行解析器对象
    """
    parser.addoption(
        "--no-strict-data-files",
        action="store_false",
        help="Don't fail if a test is skipped for missing data file.",
    )


def ignore_doctest_warning(item: pytest.Item, path: str, message: str) -> None:
    """
    忽略 doctest 的警告信息。

    Parameters
    ----------
    item : pytest.Item
        当前 pytest 测试项
    path : str
        Python 对象的模块路径，例如 "pandas.DataFrame.append"。当 item.name 以给定路径结尾时，将会过滤警告。
    message : str
        要过滤的警告信息。
    """
    if item.name.endswith(path):
        item.add_marker(pytest.mark.filterwarnings(f"ignore:{message}"))


def pytest_collection_modifyitems(items, config) -> None:
    """
    修改 pytest 测试项的集合。

    Parameters
    ----------
    items : list
        pytest 测试项列表
    config : object
        pytest 的配置对象
    """
    is_doctest = config.getoption("--doctest-modules") or config.getoption(
        "--doctest-cython", default=False
    )

    # Warnings from doctests that can be ignored; place reason in comment above.
    # 每个条目指定（路径，消息）- 参见 ignore_doctest_warning 函数
    ignored_doctest_warnings = [
        ("is_int64_dtype", "is_int64_dtype is deprecated"),  # is_int64_dtype 方法已弃用警告
        ("is_interval_dtype", "is_interval_dtype is deprecated"),  # is_interval_dtype 方法已弃用警告
        ("is_period_dtype", "is_period_dtype is deprecated"),  # is_period_dtype 方法已弃用警告
        ("is_datetime64tz_dtype", "is_datetime64tz_dtype is deprecated"),  # is_datetime64tz_dtype 方法已弃用警告
        ("is_categorical_dtype", "is_categorical_dtype is deprecated"),  # is_categorical_dtype 方法已弃用警告
        ("is_sparse", "is_sparse is deprecated"),  # is_sparse 方法已弃用警告
        ("DataFrameGroupBy.fillna", "DataFrameGroupBy.fillna is deprecated"),  # DataFrameGroupBy.fillna 方法已弃用警告
        ("DataFrameGroupBy.corrwith", "DataFrameGroupBy.corrwith is deprecated"),  # DataFrameGroupBy.corrwith 方法已弃用警告
        ("NDFrame.replace", "Series.replace without 'value'"),  # NDFrame.replace 方法已弃用警告，不带 'value' 参数
        ("NDFrame.clip", "Downcasting behavior in Series and DataFrame methods"),  # Series 和 DataFrame 方法中的降级行为
        ("Series.idxmin", "The behavior of Series.idxmin"),  # Series.idxmin 方法的行为
        ("Series.idxmax", "The behavior of Series.idxmax"),  # Series.idxmax 方法的行为
        ("SeriesGroupBy.fillna", "SeriesGroupBy.fillna is deprecated"),  # SeriesGroupBy.fillna 方法已弃用警告
        ("SeriesGroupBy.idxmin", "The behavior of Series.idxmin"),  # SeriesGroupBy.idxmin 方法的行为
        ("SeriesGroupBy.idxmax", "The behavior of Series.idxmax"),  # SeriesGroupBy.idxmax 方法的行为
        ("to_pytimedelta", "The behavior of TimedeltaProperties.to_pytimedelta"),  # TimedeltaProperties.to_pytimedelta 方法的行为
        ("NDFrame.reindex_like", "keyword argument 'method' is deprecated"),  # NDFrame.reindex_like 方法的 'method' 参数已弃用
        # Docstring divides by zero to show behavior difference
        ("missing.mask_zero_div_zero", "divide by zero encountered"),  # 除零错误遇到
        (
            "pandas.core.generic.NDFrame.first",
            "first is deprecated and will be removed in a future version. "
            "Please create a mask and filter using `.loc` instead",
        ),  # first 方法已弃用，并将在未来版本中移除
        (
            "Resampler.fillna",
            "DatetimeIndexResampler.fillna is deprecated",
        ),  # Resampler.fillna 方法已弃用
        (
            "DataFrameGroupBy.fillna",
            "DataFrameGroupBy.fillna with 'method' is deprecated",
        ),  # 带 'method' 参数的 DataFrameGroupBy.fillna 方法已弃用
        ("read_parquet", "Passing a BlockManager to DataFrame is deprecated"),  # 将 BlockManager 传递给 DataFrame 已弃用警告
    ]

    # 如果是 doctest 环境
    if is_doctest:
        # 对于每个测试项
        for item in items:
            # 遍历忽略的 doctest 警告列表
            for path, message in ignored_doctest_warnings:
                # 调用 ignore_doctest_warning 函数，传入路径和消息
                ignore_doctest_warning(item, path, message)
# 创建一个健康检查列表，初始包含 'too_slow' 健康检查
hypothesis_health_checks = [hypothesis.HealthCheck.too_slow]

# 如果 Hypothesis 的版本号大于等于 "6.83.2"，则添加 'differing_executors' 健康检查
if Version(hypothesis.__version__) >= Version("6.83.2"):
    hypothesis_health_checks.append(hypothesis.HealthCheck.differing_executors)

# 注册名为 "ci" 的 Hypothesis 配置文件
hypothesis.settings.register_profile(
    "ci",
    # Hypothesis 的时间检查默认针对标量调整，所以将测试用例的默认超时时间从200ms提高到500ms。
    # 如果某个测试用例仍然过慢，(a) 尝试优化其性能，(b) 如果确实很慢，可以添加 `@settings(deadline=...)` 设置一个适当的超时值，
    # 或者使用 `deadline=None` 完全禁用该测试用例的超时限制。
    # 2022-02-09: 将超时时间从500调整为None。超时时间导致CI测试失败，不具备操作性 (# GH 24641, 44969, 45118, 44969)
    deadline=None,
    suppress_health_check=tuple(hypothesis_health_checks),
)

# 加载名为 "ci" 的 Hypothesis 配置文件
hypothesis.settings.load_profile("ci")

# 注册以下策略，使它们通过 `st.from_type` 在全局范围内可用，用于测试/tseries/offsets/test_offsets_properties.py中的偏移量
for name in "MonthBegin MonthEnd BMonthBegin BMonthEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls, st.builds(cls, n=st.integers(-99, 99), normalize=st.booleans())
    )

for name in "YearBegin YearEnd BYearBegin BYearEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls,
        st.builds(
            cls,
            n=st.integers(-5, 5),
            normalize=st.booleans(),
            month=st.integers(min_value=1, max_value=12),
        ),
    )

for name in "QuarterBegin QuarterEnd BQuarterBegin BQuarterEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls,
        st.builds(
            cls,
            n=st.integers(-24, 24),
            normalize=st.booleans(),
            startingMonth=st.integers(min_value=1, max_value=12),
        ),
    )


# ----------------------------------------------------------------
# Autouse fixtures
# ----------------------------------------------------------------


# https://github.com/pytest-dev/pytest/issues/11873
# 希望避免使用 autouse=True，但由于 pytest 8.0.0 的限制，无法避免
# 为 doctests 添加必要的导入，使 `np` 和 `pd` 名称在 doctests 中可用
@pytest.fixture(autouse=True)
def add_doctest_imports(doctest_namespace) -> None:
    """
    Make `np` and `pd` names available for doctests.
    """
    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd


# 为所有测试和测试模块配置设置
@pytest.fixture(autouse=True)
def configure_tests() -> None:
    """
    Configure settings for all tests and test modules.
    """
    pd.set_option("chained_assignment", "raise")


# ----------------------------------------------------------------
# Common arguments
# ----------------------------------------------------------------

# 返回一个DataFrame的轴编号参数的fixture，包括 0, 1, "index", "columns" 四种选项
@pytest.fixture(params=[0, 1, "index", "columns"], ids=lambda x: f"axis={x!r}")
def axis(request):
    """
    Fixture for returning the axis numbers of a DataFrame.
    """
    return request.param
@pytest.fixture(params=[True, False])
def observed(request):
    """
    是否将观察关键字传递给groupby，取值为[True, False]
    表示分类变量是否应返回未在分组器中出现的值[False / None]，或仅返回出现在分组器中的值[True]。
    未来支持[None]以便于向后兼容，如果我们决定更改默认值（并且需要警告如果没有传递此参数）。
    """
    return request.param


@pytest.fixture(params=[True, False, None])
def ordered(request):
    """
    Categorical的布尔型'ordered'参数。
    """
    return request.param


@pytest.fixture(params=[True, False])
def dropna(request):
    """
    布尔型'dropna'参数。
    """
    return request.param


@pytest.fixture(params=[True, False])
def sort(request):
    """
    布尔型'sort'参数。
    """
    return request.param


@pytest.fixture(params=[True, False])
def skipna(request):
    """
    布尔型'skipna'参数。
    """
    return request.param


@pytest.fixture(params=["first", "last", False])
def keep(request):
    """
    .duplicated或.drop_duplicates中'keep'参数的有效取值。
    """
    return request.param


@pytest.fixture(params=["both", "neither", "left", "right"])
def inclusive_endpoints_fixture(request):
    """
    试验所有区间'inclusive'参数的夹点的装置。
    """
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def closed(request):
    """
    试验所有区间闭合参数的闭区间的装置。
    """
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def other_closed(request):
    """
    次要的闭合装置，允许参数化所有对的闭合参数。
    """
    return request.param


@pytest.fixture(
    params=[
        None,
        "gzip",
        "bz2",
        "zip",
        "xz",
        "tar",
        pytest.param("zstd", marks=td.skip_if_no("zstandard")),
    ]
)
def compression(request):
    """
    在压缩测试中尝试常见的压缩类型的装置。
    """
    return request.param


@pytest.fixture(
    params=[
        "gzip",
        "bz2",
        "zip",
        "xz",
        "tar",
        pytest.param("zstd", marks=td.skip_if_no("zstandard")),
    ]
)
def compression_only(request):
    """
    在压缩测试中尝试常见的压缩类型的装置，不包括未压缩情况。
    """
    return request.param


@pytest.fixture(params=[True, False])
def writable(request):
    """
    数组是否可写的装置。
    """
    return request.param


@pytest.fixture(params=["inner", "outer", "left", "right"])
def join_type(request):
    """
    试验所有类型的连接操作的装置。
    """
    return request.param


@pytest.fixture(params=["nlargest", "nsmallest"])
def nselect_method(request):
    """
    试验所有nselect方法的装置。
    """
    return request.param
    return request.param
# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 'na_action' 参数
@pytest.fixture(params=[None, "ignore"])
def na_action(request):
    """
    Fixture for 'na_action' argument in map.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 'ascending' 参数
@pytest.fixture(params=[True, False])
def ascending(request):
    """
    Fixture for 'na_action' argument in sort_values/sort_index/rank.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 'rank_method' 参数
@pytest.fixture(params=["average", "min", "max", "first", "dense"])
def rank_method(request):
    """
    Fixture for 'rank' argument in rank.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 'as_index' 参数
@pytest.fixture(params=[True, False])
def as_index(request):
    """
    Fixture for 'as_index' argument in groupby.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 'cache' 参数
@pytest.fixture(params=[True, False])
def cache(request):
    """
    Fixture for 'cache' argument in to_datetime.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 'parallel' 参数
@pytest.fixture(params=[True, False])
def parallel(request):
    """
    Fixture for parallel keyword argument for numba.jit.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 'nogil' 参数
@pytest.fixture(params=[False])
def nogil(request):
    """
    Fixture for nogil keyword argument for numba.jit.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 'nopython' 参数
@pytest.fixture(params=[True])
def nopython(request):
    """
    Fixture for nopython keyword argument for numba.jit.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理不同空值类型
@pytest.fixture(params=tm.NULL_OBJECTS, ids=lambda x: type(x).__name__)
def nulls_fixture(request):
    """
    Fixture for each null type in pandas.
    """
    return request.param


# 将 nulls_fixture 复制给 nulls_fixture2，以生成 nulls_fixture 的笛卡尔积
nulls_fixture2 = nulls_fixture  # Generate cartesian product of nulls_fixture


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理不同唯一空值类型
@pytest.fixture(params=[None, np.nan, pd.NaT])
def unique_nulls_fixture(request):
    """
    Fixture for each null type in pandas, each null type exactly once.
    """
    return request.param


# 将 unique_nulls_fixture 复制给 unique_nulls_fixture2，以生成 unique_nulls_fixture 的笛卡尔积
unique_nulls_fixture2 = unique_nulls_fixture


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理不同 NaT 类型
@pytest.fixture(params=tm.NP_NAT_OBJECTS, ids=lambda x: type(x).__name__)
def np_nat_fixture(request):
    """
    Fixture for each NaT type in numpy.
    """
    return request.param


# 将 np_nat_fixture 复制给 np_nat_fixture2，以生成 np_nat_fixture 的笛卡尔积
np_nat_fixture2 = np_nat_fixture


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 DataFrame 和 Series 类型
@pytest.fixture(params=[DataFrame, Series])
def frame_or_series(request):
    """
    Fixture to parametrize over DataFrame and Series.
    """
    return request.param


# 使用 pytest.fixture 装饰器定义一个参数化的测试 fixture，用于处理 Index 和 Series 类型
@pytest.fixture(params=[Index, Series], ids=["index", "series"])
def index_or_series(request):
    """
    Fixture to parametrize over Index and Series, made necessary by a mypy
    bug, giving an error:
    """
    return request.param
    # 返回 request.param 参数值
    """
    List item 0 has incompatible type "Type[Series]"; expected "Type[PandasObject]"

    See GH#29725
    """
    return request.param
# ----------------------------------------------------------------
# Indices
# ----------------------------------------------------------------

# 返回一个参数化的 fixture，包含 Index、Series 和 pd.array
@pytest.fixture(params=[Index, Series, pd.array], ids=["index", "series", "array"])
def index_or_series_or_array(request):
    """
    Fixture to parametrize over Index, Series, and ExtensionArray
    """
    return request.param


# 返回一个参数化的 fixture，包含 Index、Series、DataFrame 和 pd.array，
# 使用 lambda 函数为每个参数设置 ids
@pytest.fixture(params=[Index, Series, DataFrame, pd.array], ids=lambda x: x.__name__)
def box_with_array(request):
    """
    Fixture to test behavior for Index, Series, DataFrame, and pandas Array
    classes
    """
    return request.param


# 创建一个别名 fixture，与 box_with_array 相同
box_with_array2 = box_with_array


# 返回一个类型为 TestSubDict 的 fixture，为字典的子类
@pytest.fixture
def dict_subclass() -> type[dict]:
    """
    Fixture for a dictionary subclass.
    """

    class TestSubDict(dict):
        def __init__(self, *args, **kwargs) -> None:
            dict.__init__(self, *args, **kwargs)

    return TestSubDict


# 返回一个类型为 TestNonDictMapping 的 fixture，为非映射字典的子类
@pytest.fixture
def non_dict_mapping_subclass() -> type[abc.Mapping]:
    """
    Fixture for a non-mapping dictionary subclass.
    """

    class TestNonDictMapping(abc.Mapping):
        def __init__(self, underlying_dict) -> None:
            self._data = underlying_dict

        def __getitem__(self, key):
            return self._data.__getitem__(key)

        def __iter__(self) -> Iterator:
            return self._data.__iter__()

        def __len__(self) -> int:
            return self._data.__len__()

    return TestNonDictMapping


# 返回一个 DataFrame，包含了三层 MultiIndex（年、月、日）和随机数据，
# 覆盖了 2000-01-01 起的前 100 个工作日
@pytest.fixture
def multiindex_year_month_day_dataframe_random_data():
    """
    DataFrame with 3 level MultiIndex (year, month, day) covering
    first 100 business days from 2000-01-01 with random data
    """
    tdf = DataFrame(
        np.random.default_rng(2).standard_normal((100, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=100, freq="B"),
    )
    ymd = tdf.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day]).sum()
    # 使用 int64 Index，确保一切正常运作
    ymd.index = ymd.index.set_levels([lev.astype("i8") for lev in ymd.index.levels])
    ymd.index.set_names(["year", "month", "day"], inplace=True)
    return ymd


# 返回一个 lexsorted 的两层字符串 MultiIndex
@pytest.fixture
def lexsorted_two_level_string_multiindex() -> MultiIndex:
    """
    2-level MultiIndex, lexsorted, with string names.
    """
    return MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )


# 返回一个带有两层 MultiIndex 和随机数据的 DataFrame
@pytest.fixture
def multiindex_dataframe_random_data(
    lexsorted_two_level_string_multiindex,
) -> DataFrame:
    """DataFrame with 2 level MultiIndex with random data"""
    index = lexsorted_two_level_string_multiindex
    return DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        index=index,
        columns=Index(["A", "B", "C"], name="exp"),
    )


# 创建一个未完成的函数定义，用于创建 MultiIndex
def _create_multiindex():
    """
    # 创建一个用于测试对象通用功能的 MultiIndex 多级索引对象
    """
    
    # See Also: tests.multi.conftest.idx
    # 定义主要轴索引，包含元素 ["foo", "bar", "baz", "qux"]
    major_axis = Index(["foo", "bar", "baz", "qux"])
    # 定义次要轴索引，包含元素 ["one", "two"]
    minor_axis = Index(["one", "two"])
    
    # 创建主要轴的编码数组，指定每个元素的索引位置
    major_codes = np.array([0, 0, 1, 2, 3, 3])
    # 创建次要轴的编码数组，指定每个元素的索引位置
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    # 定义索引的名称列表
    index_names = ["first", "second"]
    
    # 返回一个 MultiIndex 多级索引对象，其中包含主要轴和次要轴的索引及其编码
    return MultiIndex(
        levels=[major_axis, minor_axis],  # 指定索引的层次结构，主要和次要轴的索引
        codes=[major_codes, minor_codes],  # 指定每个轴的索引编码
        names=index_names,  # 指定索引的名称
        verify_integrity=False,  # 设置不验证索引的完整性
    )
def _create_mi_with_dt64tz_level():
    """
    MultiIndex with a level that is a tzaware DatetimeIndex.
    创建一个包含 tzaware DatetimeIndex 级别的 MultiIndex。
    """
    # GH#8367 round trip with pickle
    # 使用 pickle 进行 round trip 测试
    return MultiIndex.from_product(
        [[1, 2], ["a", "b"], date_range("20130101", periods=3, tz="US/Eastern")],
        names=["one", "two", "three"],
    )


indices_dict = {
    "string": Index([f"pandas_{i}" for i in range(10)]),
    "datetime": date_range("2020-01-01", periods=10),
    "datetime-tz": date_range("2020-01-01", periods=10, tz="US/Pacific"),
    "period": period_range("2020-01-01", periods=10, freq="D"),
    "timedelta": timedelta_range(start="1 day", periods=10, freq="D"),
    "range": RangeIndex(10),
    "int8": Index(np.arange(10), dtype="int8"),
    "int16": Index(np.arange(10), dtype="int16"),
    "int32": Index(np.arange(10), dtype="int32"),
    "int64": Index(np.arange(10), dtype="int64"),
    "uint8": Index(np.arange(10), dtype="uint8"),
    "uint16": Index(np.arange(10), dtype="uint16"),
    "uint32": Index(np.arange(10), dtype="uint32"),
    "uint64": Index(np.arange(10), dtype="uint64"),
    "float32": Index(np.arange(10), dtype="float32"),
    "float64": Index(np.arange(10), dtype="float64"),
    "bool-object": Index([True, False] * 5, dtype=object),
    "bool-dtype": Index([True, False] * 5, dtype=bool),
    "complex64": Index(
        np.arange(10, dtype="complex64") + 1.0j * np.arange(10, dtype="complex64")
    ),
    "complex128": Index(
        np.arange(10, dtype="complex128") + 1.0j * np.arange(10, dtype="complex128")
    ),
    "categorical": CategoricalIndex(list("abcd") * 2),
    "interval": IntervalIndex.from_breaks(np.linspace(0, 100, num=11)),
    "empty": Index([]),
    "tuples": MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3])),
    "mi-with-dt64tz-level": _create_mi_with_dt64tz_level(),
    "multi": _create_multiindex(),
    "repeats": Index([0, 0, 1, 1, 2, 2]),
    "nullable_int": Index(np.arange(10), dtype="Int64"),
    "nullable_uint": Index(np.arange(10), dtype="UInt16"),
    "nullable_float": Index(np.arange(10), dtype="Float32"),
    "nullable_bool": Index(np.arange(10).astype(bool), dtype="boolean"),
    "string-python": Index(
        pd.array([f"pandas_{i}" for i in range(10)], dtype="string[python]")
    ),
}
if has_pyarrow:
    idx = Index(pd.array([f"pandas_{i}" for i in range(10)], dtype="string[pyarrow]"))
    indices_dict["string-pyarrow"] = idx


@pytest.fixture(params=indices_dict.keys())
def index(request):
    """
    Fixture for many "simple" kinds of indices.

    These indices are unlikely to cover corner cases, e.g.
        - no names
        - no NaTs/NaNs
        - no values near implementation bounds
        - ...
    """
    # copy to avoid mutation, e.g. setting .name
    # 复制以避免变异，例如设置 .name
    return indices_dict[request.param].copy()


@pytest.fixture(
    params=[
        key for key, value in indices_dict.items() if not isinstance(value, MultiIndex)
    ]
)
def index_flat(request):
    """
    Fixture for "flat" indices (not MultiIndex instances).
    为“平坦”索引（非 MultiIndex 实例）提供 Fixture。
    """
    # 获取参数化测试函数的参数值作为键值
    key = request.param
    # 返回参数对应的 indices_dict 中的值的副本
    return indices_dict[key].copy()
@pytest.fixture(
    params=[
        # 从indices_dict中选择项作为参数，条件是：
        # - 键名不以"int"、"uint"、"float"开头，
        #   且不是"range"、"empty"、"repeats"、"bool-dtype"中的任何一个
        # - 值不是MultiIndex类型的
        key
        for key, value in indices_dict.items()
        if not (
            key.startswith(("int", "uint", "float"))
            or key in ["range", "empty", "repeats", "bool-dtype"]
        )
        and not isinstance(value, MultiIndex)
    ]
)
def index_with_missing(request):
    """
    Fixture for indices with missing values.

    Integer-dtype and empty cases are excluded because they cannot hold missing
    values.

    MultiIndex is excluded because isna() is not defined for MultiIndex.
    """

    # GH 35538. Use deep copy to avoid illusive bug on np-dev
    # GHA pipeline that writes into indices_dict despite copy
    # 根据请求的参数（即indices_dict中的某个项），进行深拷贝以避免在np-dev的GHA流水线中出现的bug
    ind = indices_dict[request.param].copy(deep=True)
    vals = ind.values.copy()
    if request.param in ["tuples", "mi-with-dt64tz-level", "multi"]:
        # 对于MultiIndex顶层设置缺失值的情况
        vals = ind.tolist()
        vals[0] = (None,) + vals[0][1:]
        vals[-1] = (None,) + vals[-1][1:]
        return MultiIndex.from_tuples(vals)
    else:
        # 在非MultiIndex情况下，设置首尾两个值为None表示缺失
        vals[0] = None
        vals[-1] = None
        return type(ind)(vals)


# ----------------------------------------------------------------
# Series'
# ----------------------------------------------------------------
@pytest.fixture
def string_series() -> Series:
    """
    Fixture for Series of floats with Index of unique strings
    """
    return Series(
        np.arange(30, dtype=np.float64) * 1.1,
        index=Index([f"i_{i}" for i in range(30)], dtype=object),
        name="series",
    )


@pytest.fixture
def object_series() -> Series:
    """
    Fixture for Series of dtype object with Index of unique strings
    """
    data = [f"foo_{i}" for i in range(30)]
    index = Index([f"bar_{i}" for i in range(30)], dtype=object)
    return Series(data, index=index, name="objects", dtype=object)


@pytest.fixture
def datetime_series() -> Series:
    """
    Fixture for Series of floats with DatetimeIndex
    """
    return Series(
        np.random.default_rng(2).standard_normal(30),
        index=date_range("2000-01-01", periods=30, freq="B"),
        name="ts",
    )


def _create_series(index):
    """Helper for the _series dict"""
    size = len(index)
    data = np.random.default_rng(2).standard_normal(size)
    return Series(data, index=index, name="a", copy=False)


_series = {
    f"series-with-{index_id}-index": _create_series(index)
    for index_id, index in indices_dict.items()
}


@pytest.fixture
def series_with_simple_index(index) -> Series:
    """
    Fixture for tests on series with changing types of indices.
    """
    return _create_series(index)


_narrow_series = {
    f"{dtype.__name__}-series": Series(
        range(30), index=[f"i-{i}" for i in range(30)], name="a", dtype=dtype
    )
    for dtype in tm.NARROW_NP_DTYPES
}


_index_or_series_objs = {**indices_dict, **_series, **_narrow_series}


@pytest.fixture(params=_index_or_series_objs.keys())
# 返回根据请求参数所对应的 _index_or_series_objs 中的对象的深拷贝副本
def index_or_series_obj(request):
    """
    Fixture for tests on indexes, series and series with a narrow dtype
    copy to avoid mutation, e.g. setting .name
    """
    return _index_or_series_objs[request.param].copy(deep=True)


# 创建一个字典，键为形如 "{dtype.__name__}-series" 的字符串，值为对应 dtype 的 Series 对象
_typ_objects_series = {
    f"{dtype.__name__}-series": Series(dtype) for dtype in tm.PYTHON_DATA_TYPES
}


# 创建一个字典，将 indices_dict, _series, _narrow_series, _typ_objects_series 的键值对合并到一个新字典中
_index_or_series_memory_objs = {
    **indices_dict,
    **_series,
    **_narrow_series,
    **_typ_objects_series,
}


# ----------------------------------------------------------------
# DataFrames
# ----------------------------------------------------------------

# 返回一个由整数构成的 DataFrame，索引是唯一字符串的 Index 对象
# 列为 ['A', 'B', 'C', 'D']
@pytest.fixture
def int_frame() -> DataFrame:
    """
    Fixture for DataFrame of ints with index of unique strings

    Columns are ['A', 'B', 'C', 'D']
    """
    return DataFrame(
        np.ones((30, 4), dtype=np.int64),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
    )


# 返回一个由随机浮点数构成的 DataFrame，索引是唯一字符串的 Index 对象
# 列为 ['A', 'B', 'C', 'D']
@pytest.fixture
def float_frame() -> DataFrame:
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    return DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)]),
        columns=Index(list("ABCD")),
    )


# 返回一个带有重复的 DatetimeIndex 的 Series 对象
@pytest.fixture
def rand_series_with_duplicate_datetimeindex() -> Series:
    """
    Fixture for Series with a DatetimeIndex that has duplicates.
    """
    dates = [
        datetime(2000, 1, 2),
        datetime(2000, 1, 2),
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
        datetime(2000, 1, 3),
        datetime(2000, 1, 3),
        datetime(2000, 1, 4),
        datetime(2000, 1, 4),
        datetime(2000, 1, 4),
        datetime(2000, 1, 5),
    ]

    return Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)


# ----------------------------------------------------------------
# Scalars
# ----------------------------------------------------------------

# 返回元组中的标量和对应的类型描述
@pytest.fixture(
    params=[
        (Interval(left=0, right=5), IntervalDtype("int64", "right")),
        (Interval(left=0.1, right=0.5), IntervalDtype("float64", "right")),
        (Period("2012-01", freq="M"), "period[M]"),
        (Period("2012-02-01", freq="D"), "period[D]"),
        (
            Timestamp("2011-01-01", tz="US/Eastern"),
            DatetimeTZDtype(unit="s", tz="US/Eastern"),
        ),
        (Timedelta(seconds=500), "timedelta64[ns]"),
    ]
)
def ea_scalar_and_dtype(request):
    return request.param


# ----------------------------------------------------------------
# Operators & Operations
# ----------------------------------------------------------------

@pytest.fixture(params=tm.arithmetic_dunder_methods)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations.

    Parameters
    ----------
    request : object
        A request object from pytest that encapsulates the parameterization.

    Returns
    -------
    object
        The specific arithmetic operator or method to be tested.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        ops.radd,
        operator.sub,
        ops.rsub,
        operator.mul,
        ops.rmul,
        operator.truediv,
        ops.rtruediv,
        operator.floordiv,
        ops.rfloordiv,
        operator.mod,
        ops.rmod,
        operator.pow,
        ops.rpow,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.and_,
        ops.rand_,
        operator.xor,
        ops.rxor,
        operator.or_,
        ops.ror_,
    ]
)
def all_binary_operators(request):
    """
    Fixture for operator and roperator arithmetic, comparison, and logical ops.

    Parameters
    ----------
    request : object
        A request object from pytest that encapsulates the parameterization.

    Returns
    -------
    object
        The specific operator function to be tested.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        ops.radd,
        operator.sub,
        ops.rsub,
        operator.mul,
        ops.rmul,
        operator.truediv,
        ops.rtruediv,
        operator.floordiv,
        ops.rfloordiv,
        operator.mod,
        ops.rmod,
        operator.pow,
        ops.rpow,
    ]
)
def all_arithmetic_functions(request):
    """
    Fixture for operator and roperator arithmetic functions.

    Parameters
    ----------
    request : object
        A request object from pytest that encapsulates the parameterization.

    Returns
    -------
    object
        The specific arithmetic function to be tested.
    """
    return request.param


_all_numeric_reductions = [
    "count",
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
    "sem",
]

@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.

    Parameters
    ----------
    request : object
        A request object from pytest that encapsulates the parameterization.

    Returns
    -------
    str
        The specific numeric reduction name to be tested.
    """
    return request.param


_all_boolean_reductions = ["all", "any"]

@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.

    Parameters
    ----------
    request : object
        A request object from pytest that encapsulates the parameterization.

    Returns
    -------
    str
        The specific boolean reduction name to be tested.
    """
    return request.param


_all_reductions = _all_numeric_reductions + _all_boolean_reductions

@pytest.fixture(params=_all_reductions)
def all_reductions(request):
    """
    Fixture for all (boolean + numeric) reduction names.

    Parameters
    ----------
    request : object
        A request object from pytest that encapsulates the parameterization.

    Returns
    -------
    str
        The specific reduction name to be tested.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    ]
)
def comparison_op(request):
    """
    Fixture for operator module comparison functions.

    Parameters
    ----------
    request : object
        A request object from pytest that encapsulates the parameterization.

    Returns
    -------
    object
        The specific comparison operator function to be tested.
    """
    return request.param


@pytest.fixture(params=["__le__", "__lt__", "__ge__", "__gt__"])
def compare_operators_no_eq_ne(request):
    """
    Fixture for dunder names for compare operations except == and !=

    Parameters
    ----------
    request : object
        A request object from pytest that encapsulates the parameterization.

    Returns
    -------
    str
        The specific dunder name for comparison operation to be tested.
    """
    return request.param


@pytest.fixture(
    # 定义一个列表，包含了特殊方法名称 "__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"
    params=["__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"]
# ----------------------------------------------------------------
# Data sets/files
# ----------------------------------------------------------------

# 返回 pytest 配置中 `--no-strict-data-files` 的设置值
@pytest.fixture
def strict_data_files(pytestconfig):
    """
    Returns the configuration for the test setting `--no-strict-data-files`.
    """
    return pytestconfig.getoption("--no-strict-data-files")


# 返回一个函数，用于获取数据文件的路径
# 如果文件路径不存在且 `--no-strict-data-files` 选项未设置，则抛出 ValueError 异常
# 参数：
# path : str
#     文件路径，相对于 `pandas/tests/`
# 返回：
# path，包括 `pandas/tests` 在内的完整路径
# 异常：
# ValueError
#     如果路径不存在且未设置 `--no-strict-data-files` 选项
@pytest.fixture
def datapath(strict_data_files: str) -> Callable[..., str]:
    """
    Get the path to a data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``pandas/tests/``

    Returns
    -------
    path including ``pandas/tests``.

    Raises
    ------
    ValueError
        If the path doesn't exist and the --no-strict-data-files option is not set.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "tests")

    def deco(*args):
        path = os.path.join(BASE_PATH, *args)
        if not os.path.exists(path):
            if strict_data_files:
                raise ValueError(
                    f"Could not find file {path} and --no-strict-data-files is not set."
                )
            pytest.skip(f"Could not find {path}.")
        return path

    return deco


# ----------------------------------------------------------------
# Time zones
# ----------------------------------------------------------------

# 定义多个时区常量
TIMEZONES = [
    None,
    "UTC",
    "US/Eastern",
    "Asia/Tokyo",
    "dateutil/US/Pacific",
    "dateutil/Asia/Singapore",
    "+01:15",
    "-02:15",
    "UTC+01:15",
    "UTC-02:15",
    tzutc(),
    tzlocal(),
    FixedOffset(300),
    FixedOffset(0),
    FixedOffset(-300),
    timezone.utc,
    timezone(timedelta(hours=1)),
    timezone(timedelta(hours=-1), name="foo"),
]

# 如果 zoneinfo 可用，则添加额外的时区信息到 TIMEZONES 中
if zoneinfo is not None:
    TIMEZONES.extend(
        [
            zoneinfo.ZoneInfo("US/Pacific"),  # type: ignore[list-item]
            zoneinfo.ZoneInfo("UTC"),  # type: ignore[list-item]
        ]
    )

# 创建包含所有 TIMEZONES 元素的字符串表示列表
TIMEZONE_IDS = [repr(i) for i in TIMEZONES]


# 用于 parametrize 的文档化装饰器，参数为 TIMEZONE_IDS 的字符串形式
@td.parametrize_fixture_doc(str(TIMEZONE_IDS))
# 返回一个 pytest fixture，参数为 TIMEZONES 列表中的元素，使用 TIMEZONE_IDS 作为标识符
@pytest.fixture(params=TIMEZONES, ids=TIMEZONE_IDS)
def tz_naive_fixture(request):
    """
    Fixture for trying timezones including default (None): {0}
    """
    return request.param


# 用于 parametrize 的文档化装饰器，参数为 TIMEZONE_IDS[1:] 的字符串形式
@td.parametrize_fixture_doc(str(TIMEZONE_IDS[1:]))
# 返回一个 pytest fixture，参数为 TIMEZONES 列表中除第一个元素外的其它元素，使用 TIMEZONE_IDS[1:] 作为标识符
@pytest.fixture(params=TIMEZONES[1:], ids=TIMEZONE_IDS[1:])
def tz_aware_fixture(request):
    """
    Fixture for trying explicit timezones: {0}
    """
    return request.param


# 定义 UTC 相关常量列表
_UTCS = ["utc", "dateutil/UTC", utc, tzutc(), timezone.utc]

# 如果 zoneinfo 可用，则添加额外的 UTC 信息到 _UTCS 中
if zoneinfo is not None:
    _UTCS.extend(
        [
            zoneinfo.ZoneInfo("US/Pacific"),  # type: ignore[list-item]
            zoneinfo.ZoneInfo("UTC"),  # type: ignore[list-item]
        ]
    )
    # 向全局列表 _UTCS 中添加一个代表"UTC"时区的 ZoneInfo 对象
    _UTCS.append(zoneinfo.ZoneInfo("UTC"))
@pytest.fixture(params=_UTCS)
def utc_fixture(request):
    """
    Fixture to provide variants of UTC timezone strings and tzinfo objects.
    """
    return request.param


utc_fixture2 = utc_fixture


@pytest.fixture(params=["s", "ms", "us", "ns"])
def unit(request):
    """
    Fixture to provide datetime64 units we support.
    """
    return request.param


unit2 = unit


# ----------------------------------------------------------------
# Dtypes
# ----------------------------------------------------------------

@pytest.fixture(params=tm.STRING_DTYPES)
def string_dtype(request):
    """
    Parametrized fixture for string dtypes.

    * str
    * 'str'
    * 'U'
    """
    return request.param


@pytest.fixture(
    params=[
        "string[python]",
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
    ]
)
def nullable_string_dtype(request):
    """
    Parametrized fixture for nullable string dtypes.

    * 'string[python]'
    * 'string[pyarrow]'
    """
    return request.param


@pytest.fixture(
    params=[
        "python",
        pytest.param("pyarrow", marks=td.skip_if_no("pyarrow")),
        pytest.param("pyarrow_numpy", marks=td.skip_if_no("pyarrow")),
    ]
)
def string_storage(request):
    """
    Parametrized fixture for string storage modes.

    * 'python'
    * 'pyarrow'
    * 'pyarrow_numpy'
    """
    return request.param


@pytest.fixture(
    params=[
        "numpy_nullable",
        pytest.param("pyarrow", marks=td.skip_if_no("pyarrow")),
    ]
)
def dtype_backend(request):
    """
    Parametrized fixture for dtype backend modes.

    * 'numpy_nullable'
    * 'pyarrow'
    """
    return request.param


# Alias so we can test with cartesian product of string_storage
string_storage2 = string_storage


@pytest.fixture(params=tm.BYTES_DTYPES)
def bytes_dtype(request):
    """
    Parametrized fixture for bytes dtypes.

    * bytes
    * 'bytes'
    """
    return request.param


@pytest.fixture(params=tm.OBJECT_DTYPES)
def object_dtype(request):
    """
    Parametrized fixture for object dtypes.

    * object
    * 'object'
    """
    return request.param


@pytest.fixture(
    params=[
        "object",
        "string[python]",
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("string[pyarrow_numpy]", marks=td.skip_if_no("pyarrow")),
    ]
)
def any_string_dtype(request):
    """
    Parametrized fixture for any string dtypes.

    * 'object'
    * 'string[python]'
    * 'string[pyarrow]'
    """
    return request.param


@pytest.fixture(params=tm.DATETIME64_DTYPES)
def datetime64_dtype(request):
    """
    Parametrized fixture for datetime64 dtypes.

    * 'datetime64[ns]'
    * 'M8[ns]'
    """
    return request.param


@pytest.fixture(params=tm.TIMEDELTA64_DTYPES)
def timedelta64_dtype(request):
    """
    Parametrized fixture for timedelta64 dtypes.

    * 'timedelta64[ns]'
    * 'm8[ns]'
    """
    return request.param
def fixed_now_ts() -> Timestamp:
    """
    Fixture emits fixed Timestamp.now()
    """
    # 返回一个固定的时间戳，代表特定日期和时间的时间戳对象
    return Timestamp(
        year=2021, month=1, day=1, hour=12, minute=4, second=13, microsecond=22
    )


@pytest.fixture(params=tm.FLOAT_NUMPY_DTYPES)
def float_numpy_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    """
    # 返回一个参数化的 fixture，用于 float 数据类型
    return request.param


@pytest.fixture(params=tm.FLOAT_EA_DTYPES)
def float_ea_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * 'Float32'
    * 'Float64'
    """
    # 返回一个参数化的 fixture，用于 EA（Extended Architecture）中的 float 数据类型
    return request.param


@pytest.fixture(params=tm.ALL_FLOAT_DTYPES)
def any_float_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    * 'Float32'
    * 'Float64'
    """
    # 返回一个参数化的 fixture，包含各种 float 数据类型
    return request.param


@pytest.fixture(params=tm.COMPLEX_DTYPES)
def complex_dtype(request):
    """
    Parameterized fixture for complex dtypes.

    * complex
    * 'complex64'
    * 'complex128'
    """
    # 返回一个参数化的 fixture，用于复数数据类型
    return request.param


@pytest.fixture(params=tm.SIGNED_INT_NUMPY_DTYPES)
def any_signed_int_numpy_dtype(request):
    """
    Parameterized fixture for signed integer dtypes.

    * int
    * 'int8'
    * 'int16'
    * 'int32'
    * 'int64'
    """
    # 返回一个参数化的 fixture，用于有符号整数数据类型
    return request.param


@pytest.fixture(params=tm.UNSIGNED_INT_NUMPY_DTYPES)
def any_unsigned_int_numpy_dtype(request):
    """
    Parameterized fixture for unsigned integer dtypes.

    * 'uint8'
    * 'uint16'
    * 'uint32'
    * 'uint64'
    """
    # 返回一个参数化的 fixture，用于无符号整数数据类型
    return request.param


@pytest.fixture(params=tm.ALL_INT_NUMPY_DTYPES)
def any_int_numpy_dtype(request):
    """
    Parameterized fixture for any integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    """
    # 返回一个参数化的 fixture，包含各种整数数据类型
    return request.param


@pytest.fixture(params=tm.ALL_INT_EA_DTYPES)
def any_int_ea_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    # 返回一个参数化的 fixture，包含 EA（Extended Architecture）中的可空整数数据类型
    return request.param


@pytest.fixture(params=tm.ALL_INT_DTYPES)
def any_int_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    # 返回一个参数化的 fixture，包含各种可空整数数据类型
    return request.param


@pytest.fixture(params=tm.ALL_INT_EA_DTYPES + tm.FLOAT_EA_DTYPES)
def any_numeric_ea_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype and
    any float ea dtypes.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    """
    # 返回一个参数化的 fixture，包含 EA（Extended Architecture）中的可空整数和浮点数数据类型
    return request.param
#  Unsupported operand types for + ("List[Union[str, ExtensionDtype, dtype[Any],
#  Type[object]]]" and "List[str]")
@pytest.fixture(
    params=tm.ALL_INT_EA_DTYPES
    + tm.FLOAT_EA_DTYPES
    + tm.ALL_INT_PYARROW_DTYPES_STR_REPR
    + tm.FLOAT_PYARROW_DTYPES_STR_REPR  # type: ignore[operator]
)
def any_numeric_ea_and_arrow_dtype(request):
    """
    Parameterized fixture for any numeric dtype usable with pandas.

    This fixture combines multiple lists of data types:
    - ALL_INT_EA_DTYPES: List of all integer dtypes for pandas extension arrays.
    - FLOAT_EA_DTYPES: List of all float dtypes for pandas extension arrays.
    - ALL_INT_PYARROW_DTYPES_STR_REPR: List of all integer dtypes in PyArrow format strings.
    - FLOAT_PYARROW_DTYPES_STR_REPR: List of all float dtypes in PyArrow format strings.

    It provides a parameterized fixture that tests all combinations of these dtypes.
    """
    return request.param


@pytest.fixture(params=tm.SIGNED_INT_EA_DTYPES)
def any_signed_int_ea_dtype(request):
    """
    Parameterized fixture for any signed integer dtype usable with pandas.

    This fixture provides a parameterized test for signed integer dtypes
    compatible with pandas extension arrays (ea).
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_NUMPY_DTYPES)
def any_real_numpy_dtype(request):
    """
    Parameterized fixture for any purely real numeric dtype usable with numpy.

    This fixture tests various numeric dtypes:
    - Integer dtypes: int8, uint8, int16, uint16, int32, uint32, int64, uint64
    - Float dtypes: float32, float64

    It provides a parameterized fixture that tests all these dtypes.
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_DTYPES)
def any_real_numeric_dtype(request):
    """
    Parameterized fixture for any purely real numeric dtype.

    This fixture tests various purely real numeric dtypes:
    - Integer dtypes: int8, uint8, int16, uint16, int32, uint32, int64, uint64
    - Float dtypes: float32, float64
    - Additional extension array dtypes associated with these types.

    It provides a parameterized fixture that tests all these dtypes and their associated extension array types.
    """
    return request.param


@pytest.fixture(params=tm.ALL_NUMPY_DTYPES)
def any_numpy_dtype(request):
    """
    Parameterized fixture for all numpy dtypes.

    This fixture tests all standard numpy dtypes including:
    - bool, int, uint8, int8, uint16, int16, uint32, int32, uint64, int64
    - float32, float64, complex64, complex128
    - str ('str', 'U'), bytes ('bytes'), datetime64[ns], timedelta64[ns], object

    It provides a parameterized fixture that tests all these dtypes.
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_NULLABLE_DTYPES)
def any_real_nullable_dtype(request):
    """
    Parameterized fixture for all real dtypes that can hold NA.

    This fixture tests all real dtypes that can hold NA values, including:
    - float32, float64, 'Float32', 'Float64'
    - 'UInt8', 'UInt16', 'UInt32', 'UInt64', 'Int8', 'Int16', 'Int32', 'Int64'
    - 'uint8[pyarrow]', 'uint16[pyarrow]', 'uint32[pyarrow]', 'uint64[pyarrow]',
      'int8[pyarrow]', 'int16[pyarrow]', 'int32[pyarrow]', 'int64[pyarrow]',
      'float[pyarrow]'

    It provides a parameterized fixture that tests all these dtypes.
    """
    return request.param
    # 返回传入的 request.param 参数，通常用于 pytest 中的参数化测试，表示需要传入一个 'double[pyarrow]' 的参数值
    def return_param(request):
        """
        return request.param
        """
@pytest.fixture(params=tm.ALL_NUMERIC_DTYPES)
def any_numeric_dtype(request):
    """
    Parameterized fixture for all numeric dtypes.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    * complex
    * 'complex64'
    * 'complex128'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


_any_skipna_inferred_dtype = [
    ("string", ["a", np.nan, "c"]),  # 列类型为字符串，包含字符串值和缺失值
    ("string", ["a", pd.NA, "c"]),   # 列类型为字符串，包含字符串值和pandas的NA值
    ("mixed", ["a", pd.NaT, "c"]),   # 混合类型列，包含字符串值、pandas的NaT和字符串值
    ("bytes", [b"a", np.nan, b"c"]), # 字节类型列，包含字节值和缺失值
    ("empty", [np.nan, np.nan, np.nan]),  # 空类型列，全是缺失值
    ("empty", []),  # 空类型列，没有值
    ("mixed-integer", ["a", np.nan, 2]),  # 混合整数类型列，包含字符串值、缺失值和整数值
    ("mixed", ["a", np.nan, 2.0]),   # 混合类型列，包含字符串值、缺失值和浮点数值
    ("floating", [1.0, np.nan, 2.0]),  # 浮点数类型列，包含浮点数值和缺失值
    ("integer", [1, np.nan, 2]),    # 整数类型列，包含整数值和缺失值
    ("mixed-integer-float", [1, np.nan, 2.0]),  # 混合整数和浮点数类型列，包含整数值、缺失值和浮点数值
    ("decimal", [Decimal(1), np.nan, Decimal(2)]),  # 十进制类型列，包含十进制数值和缺失值
    ("boolean", [True, np.nan, False]),  # 布尔类型列，包含布尔值和缺失值
    ("boolean", [True, pd.NA, False]),  # 布尔类型列，包含布尔值和pandas的NA值
    ("datetime64", [np.datetime64("2013-01-01"), np.nan, np.datetime64("2018-01-01")]),  # 日期时间类型列，包含日期时间值和缺失值
    ("datetime", [Timestamp("20130101"), np.nan, Timestamp("20180101")]),  # 日期时间类型列，包含日期时间对象和缺失值
    ("date", [date(2013, 1, 1), np.nan, date(2018, 1, 1)]),  # 日期类型列，包含日期对象和缺失值
    ("complex", [1 + 1j, np.nan, 2 + 2j]),  # 复数类型列，包含复数值和缺失值
    ("timedelta", [timedelta(1), np.nan, timedelta(2)]),  # 时间间隔类型列，包含时间间隔对象和缺失值
    ("time", [time(1), np.nan, time(2)]),  # 时间类型列，包含时间对象和缺失值
    ("period", [Period(2013), pd.NaT, Period(2018)]),  # 时间段类型列，包含时间段对象和pandas的NaT值
    ("interval", [Interval(0, 1), np.nan, Interval(0, 2)]),  # 区间类型列，包含区间对象和缺失值
]
ids, _ = zip(*_any_skipna_inferred_dtype)  # 使用推断出的类型作为fixture-id


@pytest.fixture(params=_any_skipna_inferred_dtype, ids=ids)
def any_skipna_inferred_dtype(request):
    """
    Fixture for all inferred dtypes from _libs.lib.infer_dtype

    The covered (inferred) types are:
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'
    * 'mixed-integer-float'
    * 'floating'
    * 'integer'
    * 'decimal'
    * 'boolean'
    * 'datetime64'
    * 'datetime'
    * 'date'
    * 'timedelta'
    * 'time'
    * 'period'
    * 'interval'

    Returns
    -------
    inferred_dtype : str
        The string for the inferred dtype from _libs.lib.infer_dtype
    values : np.ndarray
        An array of object dtype that will be inferred to have
        `inferred_dtype`

    Examples
    --------
    >>> from pandas._libs import lib
    >>>
    >>> def test_something(any_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_skipna_inferred_dtype
    ...     # will pass
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    """
    inferred_dtype, values = request.param
    # 将列表 values 转换为 NumPy 数组，使用 dtype=object 以避免类型转换
    values = np.array(values, dtype=object)  # object dtype to avoid casting

    # 在 tests/dtypes/test_inference.py 中测试推断的数据类型的正确性
    # 返回推断出的数据类型 inferred_dtype 和转换后的值数组 values
    return inferred_dtype, values
# ----------------------------------------------------------------
# Misc
# ----------------------------------------------------------------

# 定义一个 pytest fixture，返回一个 IPython.InteractiveShell 实例
@pytest.fixture
def ip():
    """
    Get an instance of IPython.InteractiveShell.

    Will raise a skip if IPython is not installed.
    """
    # 检查并导入 IPython 版本大于等于 6.0.0
    pytest.importorskip("IPython", minversion="6.0.0")
    from IPython.core.interactiveshell import InteractiveShell

    # 导入 Config 类来配置 InteractiveShell
    from traitlets.config import Config  # isort:skip

    c = Config()
    # 设置历史记录文件为内存模式
    c.HistoryManager.hist_file = ":memory:"

    return InteractiveShell(config=c)


# 定义一个 pytest fixture，用于确保在测试周围清理 Matplotlib
@pytest.fixture
def mpl_cleanup():
    """
    Ensure Matplotlib is cleaned up around a test.

    Before a test is run:

    1) Set the backend to "template" to avoid requiring a GUI.

    After a test is run:

    1) Reset units registry
    2) Reset rc_context
    3) Close all figures

    See matplotlib/testing/decorators.py#L24.
    """
    # 导入 Matplotlib 库，如果不存在则跳过
    mpl = pytest.importorskip("matplotlib")
    mpl_units = pytest.importorskip("matplotlib.units")
    plt = pytest.importorskip("matplotlib.pyplot")
    # 备份原始的单位注册表
    orig_units_registry = mpl_units.registry.copy()
    try:
        # 进入 Matplotlib 的上下文环境
        with mpl.rc_context():
            mpl.use("template")  # 设置后端为 "template"
            yield  # 执行测试
    finally:
        # 清理单位注册表
        mpl_units.registry.clear()
        mpl_units.registry.update(orig_units_registry)
        plt.close("all")  # 关闭所有图形
        # 强制执行垃圾回收以避免 Figure 关闭时的内存泄漏
        gc.collect(1)


# 定义一个 pytest fixture，返回 pd.offsets 中的 Tick 类型的日期偏移类
@pytest.fixture(
    params=[
        getattr(pd.offsets, o)
        for o in pd.offsets.__all__
        if issubclass(getattr(pd.offsets, o), pd.offsets.Tick) and o != "Tick"
    ]
)
def tick_classes(request):
    """
    Fixture for Tick based datetime offsets available for a time series.
    """
    return request.param


# 定义一个 pytest fixture，用于测试排序方法中的键
@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


# 定义一个 pytest fixture，返回三元组，用于操作数和结果的名称
@pytest.fixture(
    params=[
        ("foo", None, None),
        ("Egon", "Venkman", None),
        ("NCC1701D", "NCC1701D", "NCC1701D"),
        # 可能匹配的 NA 值
        (np.nan, np.nan, np.nan),
        (np.nan, pd.NaT, None),
        (np.nan, pd.NA, None),
        (pd.NA, pd.NA, pd.NA),
    ]
)
def names(request) -> tuple[Hashable, Hashable, Hashable]:
    """
    A 3-tuple of names, the first two for operands, the last for a result.
    """
    return request.param


# 定义一个 pytest fixture，用于在测试中对 __setitem__, loc.__setitem__, iloc.__setitem__ 进行参数化
@pytest.fixture(params=[tm.setitem, tm.loc, tm.iloc])
def indexer_sli(request):
    """
    Parametrize over __setitem__, loc.__setitem__, iloc.__setitem__
    """
    return request.param


# 定义一个 pytest fixture，用于在测试中对 loc.__getitem__, iloc.__getitem__ 进行参数化
@pytest.fixture(params=[tm.loc, tm.iloc])
def indexer_li(request):
    """
    Parametrize over loc.__getitem__, iloc.__getitem__
    """
    return request.param
# 参数化夹具，返回请求的参数
def indexer_si(request):
    """
    Parametrize over __setitem__, iloc.__setitem__
    """
    return request.param


# 参数化夹具，返回请求的参数
@pytest.fixture(params=[tm.setitem, tm.loc])
def indexer_sl(request):
    """
    Parametrize over __setitem__, loc.__setitem__
    """
    return request.param


# 参数化夹具，返回请求的参数
@pytest.fixture(params=[tm.at, tm.loc])
def indexer_al(request):
    """
    Parametrize over at.__setitem__, loc.__setitem__
    """
    return request.param


# 参数化夹具，返回请求的参数
@pytest.fixture(params=[tm.iat, tm.iloc])
def indexer_ial(request):
    """
    Parametrize over iat.__setitem__, iloc.__setitem__
    """
    return request.param


# 夹具，生成布尔值或性能警告类型的迭代器
@pytest.fixture
def performance_warning(request) -> Iterator[bool | type[Warning]]:
    """
    Fixture to check if performance warnings are enabled. Either produces
    ``PerformanceWarning`` if they are enabled, otherwise ``False``.
    """
    with pd.option_context("mode.performance_warnings", request.param):
        yield pd.errors.PerformanceWarning if request.param else False


# 夹具，返回推断字符串选项是否启用的布尔值
@pytest.fixture
def using_infer_string() -> bool:
    """
    Fixture to check if infer string option is enabled.
    """
    return pd.options.future.infer_string is True


# 列表，包含了表示华沙时区的字符串
warsaws = ["Europe/Warsaw", "dateutil/Europe/Warsaw"]
# 如果 zoneinfo 可用，添加 zoneinfo.ZoneInfo 对象到列表中
if zoneinfo is not None:
    warsaws.append(zoneinfo.ZoneInfo("Europe/Warsaw"))  # type: ignore[arg-type]


# 夹具，返回请求的华沙时区字符串
@pytest.fixture(params=warsaws)
def warsaw(request) -> str:
    """
    tzinfo for Europe/Warsaw using pytz, dateutil, or zoneinfo.
    """
    return request.param


# 夹具，返回字符串数据类型的 PyArrow 存储字段可能的取值
@pytest.fixture
def arrow_string_storage():
    """
    Fixture that lists possible PyArrow values for StringDtype storage field.
    """
    return ("pyarrow", "pyarrow_numpy")


# 夹具，生成用于测试的临时文件路径
@pytest.fixture
def temp_file(tmp_path):
    """
    Generate a unique file for testing use. See link for removal policy.
    https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-default-base-temporary-directory
    """
    file_path = tmp_path / str(uuid.uuid4())
    file_path.touch()
    return file_path
```