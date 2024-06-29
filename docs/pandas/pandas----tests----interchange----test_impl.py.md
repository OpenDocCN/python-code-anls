# `D:\src\scipysrc\pandas\pandas\tests\interchange\test_impl.py`

```
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas.compat import (
    is_ci_environment,
    is_platform_windows,
)

import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
    ColumnNullType,
    DtypeKind,
)
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes


@pytest.mark.parametrize("data", [("ordered", True), ("unordered", False)])
def test_categorical_dtype(data):
    data_categorical = {
        "ordered": pd.Categorical(list("testdata") * 30, ordered=True),
        "unordered": pd.Categorical(list("testdata") * 30, ordered=False),
    }
    # 创建包含分类数据的 DataFrame
    df = pd.DataFrame({"A": (data_categorical[data[0]])})

    # 从 DataFrame 获取列对象
    col = df.__dataframe__().get_column_by_name("A")
    # 断言列的数据类型为分类类型
    assert col.dtype[0] == DtypeKind.CATEGORICAL
    # 断言列中的空值数量为0
    assert col.null_count == 0
    # 断言列的空值描述为使用特殊标记，并且值为-1
    assert col.describe_null == (ColumnNullType.USE_SENTINEL, -1)
    # 断言列的数据块数量为1
    assert col.num_chunks() == 1
    # 获取列的分类描述信息
    desc_cat = col.describe_categorical
    # 断言列的分类是否有序
    assert desc_cat["is_ordered"] == data[1]
    # 断言列的分类是否为字典编码
    assert desc_cat["is_dictionary"] is True
    # 断言列的分类包含的具体类别
    assert isinstance(desc_cat["categories"], PandasColumn)
    tm.assert_series_equal(
        desc_cat["categories"]._col, pd.Series(["a", "d", "e", "s", "t"])
    )

    # 断言从 DataFrame 到数据帧的转换是否保持一致
    tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


def test_categorical_pyarrow():
    # GH 49889
    pa = pytest.importorskip("pyarrow", "11.0.0")

    # 创建包含字典编码的分类数据表
    arr = ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", "Sun"]
    table = pa.table({"weekday": pa.array(arr).dictionary_encode()})
    # 转换为 DataFrame
    exchange_df = table.__dataframe__()
    # 从数据帧转换回来
    result = from_dataframe(exchange_df)
    # 创建 Pandas 的分类数据
    weekday = pd.Categorical(
        arr, categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    expected = pd.DataFrame({"weekday": weekday})
    # 断言转换结果是否与预期一致
    tm.assert_frame_equal(result, expected)


def test_empty_categorical_pyarrow():
    # https://github.com/pandas-dev/pandas/issues/53077
    pa = pytest.importorskip("pyarrow", "11.0.0")

    # 创建包含空分类的数据表
    arr = [None]
    table = pa.table({"arr": pa.array(arr, "float64").dictionary_encode()})
    # 转换为 DataFrame
    exchange_df = table.__dataframe__()
    # 使用 Pandas API 从数据帧转换
    result = pd.api.interchange.from_dataframe(exchange_df)
    # 创建包含 NaN 的分类数据
    expected = pd.DataFrame({"arr": pd.Categorical([np.nan])})
    # 断言转换结果是否与预期一致
    tm.assert_frame_equal(result, expected)


def test_large_string_pyarrow():
    # GH 52795
    pa = pytest.importorskip("pyarrow", "11.0.0")

    # 创建包含大字符串的数据表
    arr = ["Mon", "Tue"]
    table = pa.table({"weekday": pa.array(arr, "large_string")})
    # 转换为 DataFrame
    exchange_df = table.__dataframe__()
    # 从数据帧转换回来
    result = from_dataframe(exchange_df)
    expected = pd.DataFrame({"weekday": ["Mon", "Tue"]})
    # 断言转换结果是否与预期一致
    tm.assert_frame_equal(result, expected)

    # 检查往返转换是否保持一致
    assert pa.Table.equals(pa.interchange.from_dataframe(result), table)


@pytest.mark.parametrize(
    ("offset", "length", "expected_values"),
    [
        # 第一个元组：索引为0，没有额外信息，包含三个浮点数元素
        (0, None, [3.3, float("nan"), 2.1]),
        # 第二个元组：索引为1，没有额外信息，包含两个浮点数元素（其中一个为NaN）
        (1, None, [float("nan"), 2.1]),
        # 第三个元组：索引为2，没有额外信息，包含一个浮点数元素
        (2, None, [2.1]),
        # 第四个元组：索引为0，包含两个额外信息，包含两个浮点数元素（其中一个为NaN）
        (0, 2, [3.3, float("nan")]),
        # 第五个元组：索引为0，包含一个额外信息，包含一个浮点数元素
        (0, 1, [3.3]),
        # 第六个元组：索引为1，包含一个额外信息，包含一个NaN 元素
        (1, 1, [float("nan")]),
    ],
# 定义一个测试函数，用于测试从 PyArrow 到 Pandas DataFrame 的转换
def test_bitmasks_pyarrow(offset, length, expected_values):
    # 导入 PyArrow 库，如果版本小于 11.0.0，则跳过测试
    pa = pytest.importorskip("pyarrow", "11.0.0")

    # 创建一个包含浮点数、空值和浮点数的数组
    arr = [3.3, None, 2.1]
    # 使用 PyArrow 创建表格，并从中切片出一部分
    table = pa.table({"arr": arr}).slice(offset, length)
    # 将 PyArrow 表格转换为 Pandas DataFrame
    exchange_df = table.__dataframe__()
    # 调用 from_dataframe 函数，将 Pandas DataFrame 转换回结果
    result = from_dataframe(exchange_df)
    # 创建期望的 Pandas DataFrame，以便进行比较
    expected = pd.DataFrame({"arr": expected_values})
    # 使用 Pandas 提供的工具进行 DataFrame 的比较
    tm.assert_frame_equal(result, expected)

    # 检查往返转换的结果是否一致
    assert pa.Table.equals(pa.interchange.from_dataframe(result), table)


# 使用 pytest 的 parametrize 装饰器定义多组输入数据进行测试
@pytest.mark.parametrize(
    "data",
    [
        # 使用 lambda 函数生成随机整数数组
        lambda: np.random.default_rng(2).integers(-100, 100),
        lambda: np.random.default_rng(2).integers(1, 100),
        # 使用 lambda 函数生成随机浮点数
        lambda: np.random.default_rng(2).random(),
        # 使用 lambda 函数生成随机布尔值
        lambda: np.random.default_rng(2).choice([True, False]),
        # 使用 lambda 函数生成随机日期
        lambda: datetime(
            year=np.random.default_rng(2).integers(1900, 2100),
            month=np.random.default_rng(2).integers(1, 12),
            day=np.random.default_rng(2).integers(1, 20),
        ),
    ],
)
# 定义测试函数，用于测试生成的 DataFrame 的列数和行数
def test_dataframe(data):
    NCOLS, NROWS = 10, 20
    # 生成包含随机数据的字典
    data = {
        f"col{int((i - NCOLS / 2) % NCOLS + 1)}": [data() for _ in range(NROWS)]
        for i in range(NCOLS)
    }
    # 使用 Pandas 创建 DataFrame
    df = pd.DataFrame(data)

    # 调用 DataFrame 的 __dataframe__ 方法，返回 PyArrow 表格
    df2 = df.__dataframe__()

    # 断言转换后的 PyArrow 表格列数与预期相符
    assert df2.num_columns() == NCOLS
    # 断言转换后的 PyArrow 表格行数与预期相符
    assert df2.num_rows() == NROWS

    # 断言转换后的 PyArrow 表格列名列表与原始数据字典的键列表相符
    assert list(df2.column_names()) == list(data.keys())

    # 选取特定列的索引并调用 from_dataframe 函数进行转换
    indices = (0, 2)
    names = tuple(list(data.keys())[idx] for idx in indices)
    result = from_dataframe(df2.select_columns(indices))
    expected = from_dataframe(df2.select_columns_by_name(names))
    # 使用 Pandas 提供的工具进行 DataFrame 的比较
    tm.assert_frame_equal(result, expected)

    # 断言结果 DataFrame 的属性中包含 Protocol Buffers 列表
    assert isinstance(result.attrs["_INTERCHANGE_PROTOCOL_BUFFERS"], list)
    assert isinstance(expected.attrs["_INTERCHANGE_PROTOCOL_BUFFERS"], list)


# 定义测试函数，用于测试处理带有缺失值的 Pandas DataFrame 的情况
def test_missing_from_masked():
    # 创建包含带有缺失值的 Pandas DataFrame
    df = pd.DataFrame(
        {
            "x": np.array([1.0, 2.0, 3.0, 4.0, 0.0]),
            "y": np.array([1.5, 2.5, 3.5, 4.5, 0]),
            "z": np.array([1.0, 0.0, 1.0, 1.0, 1.0]),
        }
    )

    # 使用随机数生成器创建随机的缺失值字典
    rng = np.random.default_rng(2)
    dict_null = {col: rng.integers(low=0, high=len(df)) for col in df.columns}
    # 遍历字典，将指定数量的索引位置设为 None
    for col, num_nulls in dict_null.items():
        null_idx = df.index[
            rng.choice(np.arange(len(df)), size=num_nulls, replace=False)
        ]
        df.loc[null_idx, col] = None

    # 调用 DataFrame 的 __dataframe__ 方法，返回 PyArrow 表格
    df2 = df.__dataframe__()

    # 断言 PyArrow 表格中特定列的缺失值数量与预期相符
    assert df2.get_column_by_name("x").null_count == dict_null["x"]
    assert df2.get_column_by_name("y").null_count == dict_null["y"]
    assert df2.get_column_by_name("z").null_count == dict_null["z"]


# 使用 pytest 的 parametrize 装饰器定义多组混合数据进行测试
@pytest.mark.parametrize(
    "data",
    [
        # 包含两列的字典数据
        {"x": [1.5, 2.5, 3.5], "y": [9.2, 10.5, 11.8]},
        {"x": [1, 2, 0], "y": [9.2, 10.5, 11.8]},
        # 包含三列的 Numpy 数组数据
        {
            "x": np.array([True, True, False]),
            "y": np.array([1, 2, 0]),
            "z": np.array([9.2, 10.5, 11.8]),
        },
    ],
)
# 定义测试函数，用于测试包含混合数据的 Pandas DataFrame 转换
def test_mixed_data(data):
    # 使用字典数据创建 Pandas DataFrame
    df = pd.DataFrame(data)
    # 调用 DataFrame 的 __dataframe__ 方法，返回 PyArrow 表格
    df2 = df.__dataframe__()
    # 遍历数据框中的每一列名
    for col_name in df.columns:
        # 使用断言检查数据框 df2 中按列名获取的列的空值数量是否为 0
        assert df2.get_column_by_name(col_name).null_count == 0
def test_mixed_missing():
    # 创建包含混合数据类型和缺失值的DataFrame
    df = pd.DataFrame(
        {
            "x": np.array([True, None, False, None, True]),  # 布尔型和缺失值
            "y": np.array([None, 2, None, 1, 2]),  # 整数和缺失值
            "z": np.array([9.2, 10.5, None, 11.8, None]),  # 浮点数和缺失值
        }
    )

    # 调用未定义的方法 '__dataframe__'，将会引发 AttributeError
    df2 = df.__dataframe__()

    # 遍历DataFrame的每一列，断言其新返回的DataFrame中每列的缺失值数量为2
    for col_name in df.columns:
        assert df2.get_column_by_name(col_name).null_count == 2


def test_string():
    # 字符串数据字典
    string_data = {
        "separator data": [
            "abC|DeF,Hik",
            "234,3245.67",
            "gSaf,qWer|Gre",
            "asd3,4sad|",
            np.nan,
        ]
    }
    # 添加空字符串，创建DataFrame
    test_str_data = string_data["separator data"] + [""]
    df = pd.DataFrame({"A": test_str_data})
    # 获取列 'A' 对应的列对象
    col = df.__dataframe__().get_column_by_name("A")

    # 断言列的大小为6
    assert col.size() == 6
    # 断言列的缺失值数量为1
    assert col.null_count == 1
    # 断言列的数据类型为字符串
    assert col.dtype[0] == DtypeKind.STRING
    # 断言列的描述缺失值方式为使用位掩码
    assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)

    # 对切片后的DataFrame进行同样的断言
    df_sliced = df[1:]
    col = df_sliced.__dataframe__().get_column_by_name("A")
    assert col.size() == 5
    assert col.null_count == 1
    assert col.dtype[0] == DtypeKind.STRING
    assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)


def test_nonstring_object():
    # 包含非字符串对象的DataFrame
    df = pd.DataFrame({"A": ["a", 10, 1.0, ()]})
    # 获取列 'A' 对应的列对象
    col = df.__dataframe__().get_column_by_name("A")
    # 调用未定义的属性 'dtype'，将引发 NotImplementedError
    with pytest.raises(NotImplementedError, match="not supported yet"):
        col.dtype


def test_datetime():
    # 包含日期时间数据的DataFrame
    df = pd.DataFrame({"A": [pd.Timestamp("2022-01-01"), pd.NaT]})
    # 获取列 'A' 对应的列对象
    col = df.__dataframe__().get_column_by_name("A")

    # 断言列的大小为2
    assert col.size() == 2
    # 断言列的缺失值数量为1
    assert col.null_count == 1
    # 断言列的数据类型为日期时间
    assert col.dtype[0] == DtypeKind.DATETIME
    # 断言列的描述缺失值方式为使用 sentinel 值 iNaT
    assert col.describe_null == (ColumnNullType.USE_SENTINEL, iNaT)

    # 使用 from_dataframe 函数将DataFrame与其 __dataframe__() 方法返回的DataFrame进行比较
    tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


def test_categorical_to_numpy_dlpack():
    # 使用类别数据创建DataFrame
    df = pd.DataFrame({"A": pd.Categorical(["a", "b", "a"])})
    # 获取列 'A' 对应的列对象
    col = df.__dataframe__().get_column_by_name("A")
    # 从列对象获取缓冲区数据，并使用 np.from_dlpack 将其转换为 numpy 数组
    result = np.from_dlpack(col.get_buffers()["data"][0])
    expected = np.array([0, 1, 0], dtype="int8")
    # 断言转换后的 numpy 数组与预期结果相等
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("data", [{}, {"a": []}])
def test_empty_pyarrow(data):
    # 引入 pyarrow 包，并从数据字典创建预期的 DataFrame
    pytest.importorskip("pyarrow", "11.0.0")
    from pyarrow.interchange import from_dataframe as pa_from_dataframe

    expected = pd.DataFrame(data)
    # 使用 pyarrow 的 from_dataframe 函数创建箭头格式的 DataFrame
    arrow_df = pa_from_dataframe(expected)
    # 使用 from_dataframe 函数将箭头格式的 DataFrame 转换为 pandas 格式，并与预期结果比较
    result = from_dataframe(arrow_df)
    tm.assert_frame_equal(result, expected)


def test_multi_chunk_pyarrow() -> None:
    # 引入 pyarrow 包，并创建包含多个分块的数组
    pa = pytest.importorskip("pyarrow", "11.0.0")
    n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
    names = ["n_legs"]
    table = pa.table([n_legs], names=names)
    # 使用 from_dataframe 函数尝试从箭头格式的表格创建 pandas DataFrame，预期引发 RuntimeError
    with pytest.raises(
        RuntimeError,
        match="To join chunks a copy is required which is "
        "forbidden by allow_copy=False",
    ):
        pd.api.interchange.from_dataframe(table, allow_copy=False)


def test_multi_chunk_column() -> None:
    # 未完成的测试函数，待补充
    pass
    # 导入 pytest 库，并检查是否可以导入 pyarrow 模块，否则跳过测试
    pytest.importorskip("pyarrow", "11.0.0")
    # 创建一个 Pandas Series，包含整数和空值，数据类型为 Int64[pyarrow]
    ser = pd.Series([1, 2, None], dtype="Int64[pyarrow]")
    # 将两个相同的 Series 水平拼接成一个 DataFrame，忽略原始索引
    df = pd.concat([ser, ser], ignore_index=True).to_frame("a")
    # 复制原始 DataFrame 用于后续比较
    df_orig = df.copy()
    # 使用 pytest 的断言，检查是否抛出 RuntimeError 异常，并且异常信息中包含特定字符串
    with pytest.raises(
        RuntimeError, match="Found multi-chunk pyarrow array, but `allow_copy` is False"
    ):
        # 调用 Pandas API 以交换格式导出 DataFrame，传入 allow_copy=False 参数
        pd.api.interchange.from_dataframe(df.__dataframe__(allow_copy=False))
    # 调用 Pandas API 以交换格式导出 DataFrame，传入 allow_copy=True 参数
    result = pd.api.interchange.from_dataframe(df.__dataframe__(allow_copy=True))
    # 创建预期的 DataFrame，期望结果中的列为 'float64' 类型
    expected = pd.DataFrame({"a": [1.0, 2.0, None, 1.0, 2.0, None]}, dtype="float64")
    # 使用 Pandas Testing 模块比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 检查重分块操作是否修改了原始 DataFrame
    tm.assert_frame_equal(df, df_orig)
    # 断言 DataFrame 中列 'a' 的数组中的分块数量为 2
    assert len(df["a"].array._pa_array.chunks) == 2
    # 断言原始 DataFrame 中列 'a' 的数组中的分块数量为 2
    assert len(df_orig["a"].array._pa_array.chunks) == 2
# 导入必要的模块 pytest 和 pandas
import pytest
import pandas as pd
# 导入 datetime 模块中的 datetime 类
from datetime import datetime
# 导入 pandas.testing 模块中的 tm 对象，用于断言数据框是否相等
import pandas.testing as tm
# 导入 pyarrow 模块，引入为 pa，并导入其 compute 模块作为 pc
import pyarrow as pa
import pyarrow.compute as pc
# 导入 is_platform_windows 和 is_ci_environment 函数
from pandas._testing import is_platform_windows, is_ci_environment
# 导入 DtypeKind 和 ArrowCTypes 枚举
from pandas.core.arrays._arrow_utils import DtypeKind, ArrowCTypes

# 定义测试函数 test_timestamp_ns_pyarrow，用于测试时间戳与 pyarrow 的交换
def test_timestamp_ns_pyarrow():
    # 检查是否能导入 pyarrow 11.0.0 及以上版本，否则跳过测试
    pytest.importorskip("pyarrow", "11.0.0")
    # 定义时间戳参数字典
    timestamp_args = {
        "year": 2000,
        "month": 1,
        "day": 1,
        "hour": 1,
        "minute": 1,
        "second": 1,
    }
    # 创建一个包含时间戳的 pandas Series 对象 df，指定数据类型为 "timestamp[ns][pyarrow]"，列名为 "col0"
    df = pd.Series(
        [datetime(**timestamp_args)],
        dtype="timestamp[ns][pyarrow]",
        name="col0",
    ).to_frame()

    # 调用私有方法 __dataframe__() 将 df 转换为交换对象 dfi
    dfi = df.__dataframe__()
    # 调用 from_dataframe() 方法将 dfi 转换回 pandas 数据结构，并获取其中 "col0" 列的元素
    result = pd.api.interchange.from_dataframe(dfi)["col0"].item()

    # 创建预期的时间戳对象 expected
    expected = pd.Timestamp(**timestamp_args)
    # 使用断言检查 result 是否与 expected 相等
    assert result == expected


# 使用 pytest 的 parametrize 装饰器定义参数化测试函数 test_datetimetzdtype
@pytest.mark.parametrize("tz", ["UTC", "US/Pacific"])
def test_datetimetzdtype(tz, unit):
    # GH 54239
    # 创建时区相关的时间数据 tz_data
    tz_data = (
        pd.date_range("2018-01-01", periods=5, freq="D").tz_localize(tz).as_unit(unit)
    )
    # 创建包含 tz_data 的数据框 df
    df = pd.DataFrame({"ts_tz": tz_data})
    # 使用 assert_frame_equal 方法断言 df 与从交换对象 df.__dataframe__() 转换而来的数据框是否相等
    tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


# 定义测试函数 test_interchange_from_non_pandas_tz_aware，用于测试非 pandas 时区感知对象的转换
def test_interchange_from_non_pandas_tz_aware(request):
    # GH 54239, 54287
    # 检查是否能导入 pyarrow 11.0.0 及以上版本，否则跳过测试
    pa = pytest.importorskip("pyarrow", "11.0.0")
    import pyarrow.compute as pc

    # 如果在 Windows 平台且是 CI 环境，则标记为预期失败，原因是需要设置 ARROW_TIMEZONE_DATABASE 环境变量
    if is_platform_windows() and is_ci_environment():
        mark = pytest.mark.xfail(
            raises=pa.ArrowInvalid,
            reason=(
                "TODO: Set ARROW_TIMEZONE_DATABASE environment variable "
                "on CI to path to the tzdata for pyarrow."
            ),
        )
        # 应用标记 mark 到当前测试请求对象 request
        request.applymarker(mark)

    # 创建包含 datetime 对象的数组 arr
    arr = pa.array([datetime(2020, 1, 1), None, datetime(2020, 1, 2)])
    # 将 arr 转换为 "Asia/Kathmandu" 时区的时间数组
    arr = pc.assume_timezone(arr, "Asia/Kathmandu")
    # 创建包含 arr 的箭头表格 table
    table = pa.table({"arr": arr})
    # 使用私有方法 __dataframe__() 将 table 转换为交换对象 exchange_df
    exchange_df = table.__dataframe__()
    # 使用 from_dataframe() 方法将 exchange_df 转换为 pandas 数据结构
    result = from_dataframe(exchange_df)

    # 创建预期的 pandas 数据框对象 expected
    expected = pd.DataFrame(
        ["2020-01-01 00:00:00+05:45", "NaT", "2020-01-02 00:00:00+05:45"],
        columns=["arr"],
        dtype="datetime64[us, Asia/Kathmandu]",
    )
    # 使用断言方法 assert_frame_equal 检查 result 是否与 expected 相等
    tm.assert_frame_equal(expected, result)


# 定义测试函数 test_interchange_from_corrected_buffer_dtypes，测试修正后的缓冲区数据类型转换
def test_interchange_from_corrected_buffer_dtypes(monkeypatch) -> None:
    # https://github.com/pandas-dev/pandas/issues/54781
    # 创建包含字符串列 "a" 的数据框 df
    df = pd.DataFrame({"a": ["foo", "bar"]}).__dataframe__()
    # 使用私有方法 __dataframe__() 将 df 转换为交换对象 interchange
    interchange = df.__dataframe__()
    # 获取列名为 "a" 的列对象 column
    column = interchange.get_column_by_name("a")
    # 获取列对象 column 的缓冲区数据
    buffers = column.get_buffers()
    # 获取缓冲区数据的类型 buffer_dtype
    buffers_data = buffers["data"]
    buffer_dtype = buffers_data[1]
    # 将 buffer_dtype 中的数据类型调整为 (DtypeKind.UINT, 8, ArrowCTypes.UINT8, buffer_dtype[3])
    buffer_dtype = (
        DtypeKind.UINT,
        8,
        ArrowCTypes.UINT8,
        buffer_dtype[3],
    )
    # 更新 buffers 中的 "data" 缓冲区数据
    buffers["data"] = (buffers_data[0], buffer_dtype)
    # 使用 lambda 函数修改 column 的 get_buffers 方法，使其返回修改后的 buffers
    column.get_buffers = lambda: buffers
    # 使用 lambda 函数修改 interchange 的 get_column_by_name 方法，使其返回 column
    interchange.get_column_by_name = lambda _: column
    # 使用 monkeypatch 替换 df 的 __dataframe__ 方法，使其返回修改后的 interchange
    monkeypatch.setattr(df, "__dataframe__", lambda allow_copy: interchange)
    # 调用 pd.api.interchange.from_dataframe 方法，将 df 转换为 pandas 数据结构


# 定义测试函数 test_empty_string_column，测试空字符串列的转换
def test_empty_string_column():
    # https://github.com/pandas-dev/pandas/issues/56703
    # 创建只包含空字符串列 "a" 的数据框 df
    df = pd.DataFrame({"a": []}, dtype=str)
    # 使用私有方法 __dataframe__() 将 df 转换为交换对象 df2
    df2 = df.__dataframe__()
    # 使用 pd.api.interchange.from_dataframe 方法将 df2 转换为 pandas 数据结构 result
    result = pd.api.interchange.from_dataframe(df2)
    # 使用 assert_frame_equal 方法断言 df 与 result 是否相等
    tm.assert_frame_equal(df, result)


# 定义测试函数 test_large_string，测试大字符串的处理
def test_large_string():
    # GH#56702
    # 检查是否能导入 pyarrow，否则跳过测试
    pytest.importorskip("pyarrow")
    # 创建一个包含单列 'a' 的 Pandas DataFrame，列类型为 'large_string[pyarrow]'
    df = pd.DataFrame({"a": ["x"]}, dtype="large_string[pyarrow]")
    # 使用 Pandas 提供的 API 将 DataFrame 转换为内部数据结构，并返回结果
    result = pd.api.interchange.from_dataframe(df.__dataframe__())
    # 创建一个预期的 Pandas DataFrame，包含单列 'a'，列类型为 'object'
    expected = pd.DataFrame({"a": ["x"]}, dtype="object")
    # 使用 Pandas 的测试工具 (tm.assert_frame_equal) 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_non_str_names():
    # GitHub问题链接: https://github.com/pandas-dev/pandas/issues/56701
    # 创建一个包含单个整数列的DataFrame
    df = pd.Series([1, 2, 3], name=0).to_frame()
    # 获取DataFrame的列名列表
    names = df.__dataframe__().column_names()
    # 断言列名列表与预期相同
    assert names == ["0"]


def test_non_str_names_w_duplicates():
    # GitHub问题链接: https://github.com/pandas-dev/pandas/issues/56701
    # 创建一个包含重复列名的DataFrame
    df = pd.DataFrame({"0": [1, 2, 3], 0: [4, 5, 6]})
    # 获取DataFrame的内部表示
    dfi = df.__dataframe__()
    # 使用pytest的断言检查是否引发了预期的TypeError异常
    with pytest.raises(
        TypeError,
        match=(
            "Expected a Series, got a DataFrame. This likely happened because you "
            "called __dataframe__ on a DataFrame which, after converting column "
            r"names to string, resulted in duplicated names: Index\(\['0', '0'\], "
            r"dtype='object'\). Please rename these columns before using the "
            "interchange protocol."
        ),
    ):
        # 使用pandas的API转换函数，并期望不允许复制操作
        pd.api.interchange.from_dataframe(dfi, allow_copy=False)


@pytest.mark.parametrize(
    ("data", "dtype", "expected_dtype"),
    [
        ([1, 2, None], "Int64", "int64"),
        ([1, 2, None], "Int64[pyarrow]", "int64"),
        ([1, 2, None], "Int8", "int8"),
        ([1, 2, None], "Int8[pyarrow]", "int8"),
        (
            [1, 2, None],
            "UInt64",
            "uint64",
        ),
        (
            [1, 2, None],
            "UInt64[pyarrow]",
            "uint64",
        ),
        ([1.0, 2.25, None], "Float32", "float32"),
        ([1.0, 2.25, None], "Float32[pyarrow]", "float32"),
        ([True, False, None], "boolean", "bool"),
        ([True, False, None], "boolean[pyarrow]", "bool"),
        (["much ado", "about", None], "string[pyarrow_numpy]", "large_string"),
        (["much ado", "about", None], "string[pyarrow]", "large_string"),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), None],
            "timestamp[ns][pyarrow]",
            "timestamp[ns]",
        ),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), None],
            "timestamp[us][pyarrow]",
            "timestamp[us]",
        ),
        (
            [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                None,
            ],
            "timestamp[us, Asia/Kathmandu][pyarrow]",
            "timestamp[us, tz=Asia/Kathmandu]",
        ),
    ],
)
def test_pandas_nullable_with_missing_values(
    data: list, dtype: str, expected_dtype: str
) -> None:
    # GitHub问题链接: https://github.com/pandas-dev/pandas/issues/57643
    # GitHub问题链接: https://github.com/pandas-dev/pandas/issues/57664
    # 导入必要的pytest库
    pa = pytest.importorskip("pyarrow", "11.0.0")
    # 导入pyarrow中的interchange模块
    import pyarrow.interchange as pai

    # 如果预期的数据类型是特定的时区时间戳，则修改预期数据类型
    if expected_dtype == "timestamp[us, tz=Asia/Kathmandu]":
        expected_dtype = pa.timestamp("us", "Asia/Kathmandu")

    # 创建一个包含单列数据的DataFrame，使用指定的数据类型
    df = pd.DataFrame({"a": data}, dtype=dtype)
    # 使用pyarrow的from_dataframe函数转换DataFrame，并获取其中的"a"列
    result = pai.from_dataframe(df.__dataframe__())["a"]
    # 断言结果的数据类型与预期数据类型相同
    assert result.type == expected_dtype
    # 断言结果的第一个和第二个元素与输入数据的第一个和第二个元素相同
    assert result[0].as_py() == data[0]
    assert result[1].as_py() == data[1]
    # 断言语句，验证 result 列表中索引为 2 的元素的值是否为 None
    assert result[2].as_py() is None
@pytest.mark.parametrize(
    ("data", "dtype", "expected_dtype"),
    [  # 参数化测试用例，包括数据、数据类型和预期数据类型
        ([1, 2, 3], "Int64", "int64"),  # 整数列表，数据类型为 Int64，预期数据类型为 int64
        ([1, 2, 3], "Int64[pyarrow]", "int64"),  # 同上，使用 pyarrow 扩展
        ([1, 2, 3], "Int8", "int8"),  # 整数列表，数据类型为 Int8，预期数据类型为 int8
        ([1, 2, 3], "Int8[pyarrow]", "int8"),  # 同上，使用 pyarrow 扩展
        (
            [1, 2, 3],
            "UInt64",
            "uint64",
        ),  # 无符号整数列表，数据类型为 UInt64，预期数据类型为 uint64
        (
            [1, 2, 3],
            "UInt64[pyarrow]",
            "uint64",
        ),  # 同上，使用 pyarrow 扩展
        ([1.0, 2.25, 5.0], "Float32", "float32"),  # 浮点数列表，数据类型为 Float32，预期数据类型为 float32
        ([1.0, 2.25, 5.0], "Float32[pyarrow]", "float32"),  # 同上，使用 pyarrow 扩展
        ([True, False, False], "boolean", "bool"),  # 布尔值列表，数据类型为 boolean，预期数据类型为 bool
        ([True, False, False], "boolean[pyarrow]", "bool"),  # 同上，使用 pyarrow 扩展
        (
            ["much ado", "about", "nothing"],
            "string[pyarrow_numpy]",
            "large_string",
        ),  # 字符串列表，数据类型为 string[pyarrow_numpy]，预期数据类型为 large_string
        (
            ["much ado", "about", "nothing"],
            "string[pyarrow]",
            "large_string",
        ),  # 同上，使用 pyarrow 扩展
        (
            [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
            ],
            "timestamp[ns][pyarrow]",
            "timestamp[ns]",
        ),  # 时间戳列表，数据类型为 timestamp[ns][pyarrow]，预期数据类型为 timestamp[ns]
        (
            [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
            ],
            "timestamp[us][pyarrow]",
            "timestamp[us]",
        ),  # 同上，时间精度为微秒
        (
            [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                datetime(2020, 1, 3, tzinfo=timezone.utc),
            ],
            "timestamp[us, Asia/Kathmandu][pyarrow]",
            "timestamp[us, tz=Asia/Kathmandu]",
        ),  # 同上，带有时区信息
    ],
)
def test_pandas_nullable_without_missing_values(
    data: list, dtype: str, expected_dtype: str
) -> None:
    # 测试 pandas 的可空数据类型处理，检查相关 GitHub 问题
    pa = pytest.importorskip("pyarrow", "11.0.0")  # 导入并检查 pyarrow 版本
    import pyarrow.interchange as pai  # 导入 pyarrow.interchange 模块

    if expected_dtype == "timestamp[us, tz=Asia/Kathmandu]":
        expected_dtype = pa.timestamp("us", "Asia/Kathmandu")  # 处理特定的时间戳格式

    df = pd.DataFrame({"a": data}, dtype=dtype)  # 创建包含测试数据的 DataFrame
    result = pai.from_dataframe(df.__dataframe__())["a"]  # 从 DataFrame 转换为 pyarrow 的数据格式
    assert result.type == expected_dtype  # 断言结果的数据类型与预期相符
    assert result[0].as_py() == data[0]  # 断言第一个元素与原始数据一致
    assert result[1].as_py() == data[1]  # 断言第二个元素与原始数据一致
    assert result[2].as_py() == data[2]  # 断言第三个元素与原始数据一致


def test_string_validity_buffer() -> None:
    # 测试 pandas 大字符串类型的有效性缓冲区问题，相关 GitHub 问题链接
    pytest.importorskip("pyarrow", "11.0.0")  # 导入并检查 pyarrow 版本
    df = pd.DataFrame({"a": ["x"]}, dtype="large_string[pyarrow]")  # 创建包含大字符串的 DataFrame
    result = df.__dataframe__().get_column_by_name("a").get_buffers()["validity"]  # 获取有效性缓冲区
    assert result is None  # 断言结果为空


def test_string_validity_buffer_no_missing() -> None:
    # 测试 pandas 大字符串类型的有效性缓冲区处理无缺失数据的情况，相关 GitHub 问题链接
    pytest.importorskip("pyarrow", "11.0.0")  # 导入并检查 pyarrow 版本
    df = pd.DataFrame({"a": ["x", None]}, dtype="large_string[pyarrow]")  # 创建包含大字符串的 DataFrame
    validity = df.__dataframe__().get_column_by_name("a").get_buffers()["validity"]  # 获取有效性缓冲区
    assert validity is not None  # 断言有效性缓冲区不为空
    result = validity[1]  # 获取第二个元素
    expected = (DtypeKind.BOOL, 1, ArrowCTypes.BOOL, "=")  # 预期的元组格式
    assert result == expected  # 断言结果与预期相符


def test_empty_dataframe():
    # 测试空 DataFrame 的情况，相关 GitHub 问题链接
    # 创建一个空的 Pandas 数据帧 df，只有一个列 "a"，且数据类型为 int8
    df = pd.DataFrame({"a": []}, dtype="int8")
    
    # 调用 DataFrame 对象的私有方法 __dataframe__()，返回内部数据帧对象 dfi
    dfi = df.__dataframe__()
    
    # 使用 Pandas 的 api.interchange.from_dataframe() 方法，从数据帧 dfi 中生成交换数据，
    # 禁止复制操作（allow_copy=False）
    result = pd.api.interchange.from_dataframe(dfi, allow_copy=False)
    
    # 创建一个预期结果的空数据帧 expected，只有一个列 "a"，且数据类型为 int8
    expected = pd.DataFrame({"a": []}, dtype="int8")
    
    # 使用 Pandas 的测试工具 tm.assert_frame_equal()，断言 result 和 expected 的内容是否相同
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    ("data", "expected_dtype", "expected_buffer_dtype"),
    [  # 参数化测试数据列表
        (
            pd.Series(["a", "b", "a"], dtype="category"),  # 第一组数据：包含分类数据的 Pandas Series
            (DtypeKind.CATEGORICAL, 8, "c", "="),  # 期望的数据类型元组和缓冲区数据类型元组
            (DtypeKind.INT, 8, "c", "|"),  # 期望的数据类型元组和缓冲区数据类型元组
        ),
        (
            pd.Series(
                [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)],
                dtype="M8[ns]",
            ),  # 第二组数据：包含日期时间数据的 Pandas Series
            (DtypeKind.DATETIME, 64, "tsn:", "="),  # 期望的数据类型元组和缓冲区数据类型元组
            (DtypeKind.INT, 64, ArrowCTypes.INT64, "="),  # 期望的数据类型元组和缓冲区数据类型元组
        ),
        (
            pd.Series(["a", "bc", None]),  # 第三组数据：包含字符串和缺失值的 Pandas Series
            (DtypeKind.STRING, 8, ArrowCTypes.STRING, "="),  # 期望的数据类型元组和缓冲区数据类型元组
            (DtypeKind.UINT, 8, ArrowCTypes.UINT8, "="),  # 期望的数据类型元组和缓冲区数据类型元组
        ),
        (
            pd.Series([1, 2, 3]),  # 第四组数据：包含整数的 Pandas Series
            (DtypeKind.INT, 64, ArrowCTypes.INT64, "="),  # 期望的数据类型元组和缓冲区数据类型元组
            (DtypeKind.INT, 64, ArrowCTypes.INT64, "="),  # 期望的数据类型元组和缓冲区数据类型元组
        ),
        (
            pd.Series([1.5, 2, 3]),  # 第五组数据：包含浮点数的 Pandas Series
            (DtypeKind.FLOAT, 64, ArrowCTypes.FLOAT64, "="),  # 期望的数据类型元组和缓冲区数据类型元组
            (DtypeKind.FLOAT, 64, ArrowCTypes.FLOAT64, "="),  # 期望的数据类型元组和缓冲区数据类型元组
        ),
    ],
)
def test_buffer_dtype_categorical(
    data: pd.Series,
    expected_dtype: tuple[DtypeKind, int, str, str],
    expected_buffer_dtype: tuple[DtypeKind, int, str, str],
) -> None:
    """
    测试函数：test_buffer_dtype_categorical
    参数：
    - data: 测试用的 Pandas Series 数据
    - expected_dtype: 期望的列数据类型元组 (数据类型种类, 数据类型大小, 特征字符, 对齐方式)
    - expected_buffer_dtype: 期望的缓冲区数据类型元组 (数据类型种类, 数据类型大小, C 类型, 对齐方式)
    """
    # 创建包含测试数据的 DataFrame
    df = pd.DataFrame({"data": data})
    # 获取 DataFrame 内部数据结构的引用
    dfi = df.__dataframe__()
    # 通过列名获取特定列对象
    col = dfi.get_column_by_name("data")
    # 断言列的数据类型与期望的数据类型元组相等
    assert col.dtype == expected_dtype
    # 断言列的缓冲区数据类型与期望的缓冲区数据类型元组相等
    assert col.get_buffers()["data"][1] == expected_buffer_dtype
```