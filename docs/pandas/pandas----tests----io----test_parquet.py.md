# `D:\src\scipysrc\pandas\pandas\tests\io\test_parquet.py`

```
# 导入必要的模块和库
"""test parquet compat"""

import datetime  # 导入日期时间模块
from decimal import Decimal  # 导入 Decimal 类
from io import BytesIO  # 导入 BytesIO 类
import os  # 导入操作系统相关的模块
import pathlib  # 导入处理路径相关的模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from pandas.compat import is_platform_windows  # 导入平台兼容性模块
from pandas.compat.pyarrow import (  # 导入 PyArrow 相关模块和函数
    pa_version_under11p0,
    pa_version_under13p0,
    pa_version_under15p0,
    pa_version_under17p0,
)

import pandas as pd  # 导入 Pandas 库
import pandas._testing as tm  # 导入 Pandas 测试工具模块
from pandas.util.version import Version  # 导入版本管理模块

from pandas.io.parquet import (  # 导入 Parquet 文件读写相关模块和函数
    FastParquetImpl,
    PyArrowImpl,
    get_engine,
    read_parquet,
    to_parquet,
)

try:
    import pyarrow  # 尝试导入 PyArrow 库
    _HAVE_PYARROW = True
except ImportError:
    _HAVE_PYARROW = False

try:
    import fastparquet  # 尝试导入 fastparquet 库
    _HAVE_FASTPARQUET = True
except ImportError:
    _HAVE_FASTPARQUET = False


pytestmark = [  # 设定 pytest 的标记
    pytest.mark.filterwarnings("ignore:DataFrame._data is deprecated:FutureWarning"),
    pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
    ),
]


# 设置引擎和跳过条件
@pytest.fixture(  # 定义 pytest 的 fixture 函数
    params=[
        pytest.param(  # 定义参数化的 pytest 参数
            "fastparquet",  # 使用 fastparquet 引擎
            marks=pytest.mark.skipif(  # 使用 pytest 标记跳过条件
                not _HAVE_FASTPARQUET,  # 如果 fastparquet 未安装
                reason="fastparquet is not installed",  # 给出跳过的原因
            ),
        ),
        pytest.param(  # 定义参数化的 pytest 参数
            "pyarrow",  # 使用 pyarrow 引擎
            marks=pytest.mark.skipif(  # 使用 pytest 标记跳过条件
                not _HAVE_PYARROW,  # 如果 pyarrow 未安装
                reason="pyarrow is not installed"  # 给出跳过的原因
            ),
        ),
    ]
)
def engine(request):  # 定义 pytest fixture 的参数化函数
    return request.param  # 返回 fixture 的参数值


@pytest.fixture  # 定义 pytest fixture 函数
def pa():  # 如果有 pyarrow 安装，则返回 "pyarrow"
    if not _HAVE_PYARROW:  # 如果 pyarrow 未安装
        pytest.skip("pyarrow is not installed")  # 跳过测试，给出提示信息
    return "pyarrow"  # 返回 "pyarrow"


@pytest.fixture  # 定义 pytest fixture 函数
def fp():  # 如果有 fastparquet 安装，则返回 "fastparquet"
    if not _HAVE_FASTPARQUET:  # 如果 fastparquet 未安装
        pytest.skip("fastparquet is not installed")  # 跳过测试，给出提示信息
    return "fastparquet"  # 返回 "fastparquet"


@pytest.fixture  # 定义 pytest fixture 函数
def df_compat():  # 返回一个包含指定数据的 DataFrame
    return pd.DataFrame({"A": [1, 2, 3], "B": "foo"})  # 返回 DataFrame 对象


@pytest.fixture  # 定义 pytest fixture 函数
def df_cross_compat():  # 返回一个包含多种数据类型的 DataFrame
    df = pd.DataFrame(  # 创建 DataFrame 对象
        {
            "a": list("abc"),  # 包含字符列 'a'
            "b": list(range(1, 4)),  # 包含整数列 'b'
            # 'c': np.arange(3, 6).astype('u1'),  # 暂时注释掉的列 'c'
            "d": np.arange(4.0, 7.0, dtype="float64"),  # 包含浮点数列 'd'
            "e": [True, False, True],  # 包含布尔值列 'e'
            "f": pd.date_range("20130101", periods=3),  # 包含日期时间列 'f'
            # 'g': pd.date_range('20130101', periods=3,  # 暂时注释掉的列 'g'
            #                    tz='US/Eastern'),
            # 'h': pd.date_range('20130101', periods=3, freq='ns')  # 暂时注释掉的列 'h'
        }
    )
    return df  # 返回 DataFrame 对象


@pytest.fixture  # 定义 pytest fixture 函数
def df_full():  # 返回一个完整的 DataFrame，但此处未给出实现
    pass  # 空函数体，暂未实现
    # 返回一个 Pandas DataFrame 对象，其中包含不同数据类型的列
    return pd.DataFrame(
        {
            "string": list("abc"),  # 包含字符串类型列，每个元素为单个字符 'a', 'b', 'c'
            "string_with_nan": ["a", np.nan, "c"],  # 包含字符串类型列，包括 NaN 值
            "string_with_none": ["a", None, "c"],  # 包含字符串类型列，包括 None 值
            "bytes": [b"foo", b"bar", b"baz"],  # 包含字节串类型列，每个元素为字节串
            "unicode": ["foo", "bar", "baz"],  # 包含 Unicode 字符串类型列
            "int": list(range(1, 4)),  # 包含整数类型列，值从 1 到 3
            "uint": np.arange(3, 6).astype("u1"),  # 包含无符号整数类型列，值从 3 到 5
            "float": np.arange(4.0, 7.0, dtype="float64"),  # 包含浮点数类型列，值从 4.0 到 6.0
            "float_with_nan": [2.0, np.nan, 3.0],  # 包含浮点数类型列，包括 NaN 值
            "bool": [True, False, True],  # 包含布尔类型列，值为 True 或 False
            "datetime": pd.date_range("20130101", periods=3),  # 包含日期时间类型列，从 '20130101' 开始的 3 个日期
            "datetime_with_nat": [  # 包含日期时间类型列，包括 NaT (Not a Time) 值
                pd.Timestamp("20130101"),
                pd.NaT,
                pd.Timestamp("20130103"),
            ],
        }
    )
@pytest.fixture(
    params=[
        datetime.datetime.now(datetime.timezone.utc),  # 使用当前 UTC 时间创建参数化测试的日期时间对象
        datetime.datetime.now(datetime.timezone.min),  # 使用最小时区的当前时间创建参数化测试的日期时间对象
        datetime.datetime.now(datetime.timezone.max),  # 使用最大时区的当前时间创建参数化测试的日期时间对象
        datetime.datetime.strptime("2019-01-04T16:41:24+0200", "%Y-%m-%dT%H:%M:%S%z"),  # 解析指定格式的带时区日期时间对象
        datetime.datetime.strptime("2019-01-04T16:41:24+0215", "%Y-%m-%dT%H:%M:%S%z"),  # 解析指定格式的带时区日期时间对象
        datetime.datetime.strptime("2019-01-04T16:41:24-0200", "%Y-%m-%dT%H:%M:%S%z"),  # 解析指定格式的带时区日期时间对象
        datetime.datetime.strptime("2019-01-04T16:41:24-0215", "%Y-%m-%dT%H:%M:%S%z"),  # 解析指定格式的带时区日期时间对象
    ]
)
def timezone_aware_date_list(request):
    return request.param


def check_round_trip(
    df,
    engine=None,
    path=None,
    write_kwargs=None,
    read_kwargs=None,
    expected=None,
    check_names=True,
    check_like=False,
    check_dtype=True,
    repeat=2,
):
    """Verify parquet serializer and deserializer produce the same results.

    Performs a pandas to disk and disk to pandas round trip,
    then compares the 2 resulting DataFrames to verify equality.

    Parameters
    ----------
    df: Dataframe  # 待序列化和反序列化的 Pandas DataFrame
    engine: str, optional  # 使用的序列化引擎，可以是 'pyarrow' 或 'fastparquet'
        'pyarrow' or 'fastparquet'
    path: str, optional  # 写入和读取数据的文件路径
    write_kwargs: dict of str:str, optional  # 写入时的参数字典
    read_kwargs: dict of str:str, optional  # 读取时的参数字典
    expected: DataFrame, optional  # 期望的反序列化结果 DataFrame，默认与 df 相同
        Expected deserialization result, otherwise will be equal to `df`
    check_names: list of str, optional  # 需要比较的列名列表
        Closed set of column names to be compared
    check_like: bool, optional  # 是否忽略索引和列的顺序
        If True, ignore the order of index & columns.
    repeat: int, optional  # 测试重复执行的次数
        How many times to repeat the test
    """
    write_kwargs = write_kwargs or {"compression": None}  # 如果 write_kwargs 未提供，默认为不压缩
    read_kwargs = read_kwargs or {}  # 如果 read_kwargs 未提供，默认为空字典

    if expected is None:
        expected = df  # 如果未提供期望的结果，默认与 df 相同

    if engine:
        write_kwargs["engine"] = engine  # 如果指定了 engine，设置写入参数中的引擎
        read_kwargs["engine"] = engine  # 设置读取参数中的引擎

    def compare(repeat):
        for _ in range(repeat):
            df.to_parquet(path, **write_kwargs)  # 将 DataFrame 序列化为 Parquet 格式并写入文件
            actual = read_parquet(path, **read_kwargs)  # 从文件中读取 Parquet 格式数据并反序列化为 DataFrame

            if "string_with_nan" in expected:
                expected.loc[1, "string_with_nan"] = None  # 如果期望的 DataFrame 中包含 'string_with_nan'，将其置为 None
            tm.assert_frame_equal(
                expected,
                actual,
                check_names=check_names,
                check_like=check_like,
                check_dtype=check_dtype,
            )  # 使用测试工具比较期望和实际的 DataFrame 结果

    if path is None:
        with tm.ensure_clean() as path:  # 使用测试工具确保测试路径的干净性，并执行比较
            compare(repeat)
    else:
        compare(repeat)  # 直接执行比较操作


def check_partition_names(path, expected):
    """Check partitions of a parquet file are as expected.

    Parameters
    ----------
    path: str  # 数据集的路径
        Path of the dataset.
    expected: iterable of str  # 预期的分区名称列表
        Expected partition names.
    """
    import pyarrow.dataset as ds  # 导入 PyArrow 的 dataset 模块

    dataset = ds.dataset(path, partitioning="hive")  # 使用 Hive 分区方案打开指定路径的数据集
    assert dataset.partitioning.schema.names == expected  # 断言数据集的分区方案的名称与预期相符


def test_invalid_engine(df_compat):
    msg = "engine must be one of 'pyarrow', 'fastparquet'"  # 引擎参数无效的错误消息
    # 使用 pytest 提供的上下文管理器 `pytest.raises` 来捕获函数调用中引发的 ValueError 异常
    # 并验证异常消息与变量 `msg` 匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 `check_round_trip` 函数，期望其抛出 ValueError 异常，并验证异常消息
        check_round_trip(df_compat, "foo", "bar")
# 设置选项上下文，指定使用 pyarrow 引擎，然后调用 check_round_trip 函数
def test_options_py(df_compat, pa):
    with pd.option_context("io.parquet.engine", "pyarrow"):
        check_round_trip(df_compat)

# 设置选项上下文，指定使用 fastparquet 引擎，然后调用 check_round_trip 函数
def test_options_fp(df_compat, fp):
    with pd.option_context("io.parquet.engine", "fastparquet"):
        check_round_trip(df_compat)

# 设置选项上下文，指定使用 auto 模式选择引擎，然后调用 check_round_trip 函数
def test_options_auto(df_compat, fp, pa):
    with pd.option_context("io.parquet.engine", "auto"):
        check_round_trip(df_compat)

# 测试获取引擎函数 get_engine，验证使用 "pyarrow" 引擎返回的类型是否是 PyArrowImpl 类型
# 然后验证使用 "fastparquet" 引擎返回的类型是否是 FastParquetImpl 类型
# 最后，在不同引擎选项下，验证 "auto" 引擎返回的类型是否符合预期
def test_options_get_engine(fp, pa):
    assert isinstance(get_engine("pyarrow"), PyArrowImpl)
    assert isinstance(get_engine("fastparquet"), FastParquetImpl)

    with pd.option_context("io.parquet.engine", "pyarrow"):
        assert isinstance(get_engine("auto"), PyArrowImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)

    with pd.option_context("io.parquet.engine", "fastparquet"):
        assert isinstance(get_engine("auto"), FastParquetImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)

    with pd.option_context("io.parquet.engine", "auto"):
        assert isinstance(get_engine("auto"), PyArrowImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)

# 测试获取引擎函数 get_engine 在 "auto" 模式下返回不同的错误消息情况
def test_get_engine_auto_error_message():
    # 从 pandas.compat._optional 导入 VERSIONS 模块
    from pandas.compat._optional import VERSIONS

    # 检查是否安装了引擎，并验证其版本是否满足最小要求
    pa_min_ver = VERSIONS.get("pyarrow")
    fp_min_ver = VERSIONS.get("fastparquet")
    have_pa_bad_version = (
        False
        if not _HAVE_PYARROW
        else Version(pyarrow.__version__) < Version(pa_min_ver)
    )
    have_fp_bad_version = (
        False
        if not _HAVE_FASTPARQUET
        else Version(fastparquet.__version__) < Version(fp_min_ver)
    )

    # 检查是否安装了可用的引擎版本
    have_usable_pa = _HAVE_PYARROW and not have_pa_bad_version
    have_usable_fp = _HAVE_FASTPARQUET and not have_fp_bad_version
    # 如果没有可用的 .pyarrow 和 .fastparquet 引擎，则执行以下操作
    if not have_usable_pa and not have_usable_fp:
        # 没有找到可用的引擎。
        
        # 如果 .pyarrow 版本不符合要求，则抛出 ImportError 异常并验证匹配的错误消息
        if have_pa_bad_version:
            match = f"Pandas requires version {pa_min_ver} or newer of pyarrow."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")
        else:
            # 如果没有 .pyarrow 的可选依赖，则抛出 ImportError 异常并验证匹配的错误消息
            match = "Missing optional dependency pyarrow."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")

        # 如果 .fastparquet 版本不符合要求，则抛出 ImportError 异常并验证匹配的错误消息
        if have_fp_bad_version:
            match = f"Pandas requires version {fp_min_ver} or newer of fastparquet."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")
        else:
            # 如果没有 .fastparquet 的可选依赖，则抛出 ImportError 异常并验证匹配的错误消息
            match = "Missing optional dependency fastparquet."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")
# cross-compat with differing reading/writing engines
def test_cross_engine_pa_fp(df_cross_compat, pa, fp):
    # 使用不同的读写引擎进行交叉兼容性测试

    df = df_cross_compat
    # 确保路径干净，使用上下文管理器
    with tm.ensure_clean() as path:
        # 将 DataFrame 写入 Parquet 文件，指定使用指定的引擎（pa），无压缩
        df.to_parquet(path, engine=pa, compression=None)

        # 使用指定的引擎（fp）读取 Parquet 文件，返回结果并断言与原始 DataFrame 相等
        result = read_parquet(path, engine=fp)
        tm.assert_frame_equal(result, df)

        # 使用指定的引擎（fp）读取 Parquet 文件的部分列，返回结果并断言与原始 DataFrame 的选定列相等
        result = read_parquet(path, engine=fp, columns=["a", "d"])
        tm.assert_frame_equal(result, df[["a", "d"]])


# cross-compat with differing reading/writing engines
def test_cross_engine_fp_pa(df_cross_compat, pa, fp):
    # 使用不同的读写引擎进行交叉兼容性测试

    df = df_cross_compat
    # 确保路径干净，使用上下文管理器
    with tm.ensure_clean() as path:
        # 将 DataFrame 写入 Parquet 文件，指定使用指定的引擎（fp），无压缩
        df.to_parquet(path, engine=fp, compression=None)

        # 使用指定的引擎（pa）读取 Parquet 文件，返回结果并断言与原始 DataFrame 相等
        result = read_parquet(path, engine=pa)
        tm.assert_frame_equal(result, df)

        # 使用指定的引擎（pa）读取 Parquet 文件的部分列，返回结果并断言与原始 DataFrame 的选定列相等
        result = read_parquet(path, engine=pa, columns=["a", "d"])
        tm.assert_frame_equal(result, df[["a", "d"]])


class Base:
    def check_error_on_write(self, df, engine, exc, err_msg):
        # 检查写入时是否引发异常
        with tm.ensure_clean() as path:
            # 使用 pytest 检查引发特定异常（exc）和匹配错误消息（err_msg）
            with pytest.raises(exc, match=err_msg):
                to_parquet(df, path, engine, compression=None)

    def check_external_error_on_write(self, df, engine, exc):
        # 检查外部库在写入时是否引发异常
        with tm.ensure_clean() as path:
            # 使用 tm.external_error_raised 检查外部库引发的特定异常（exc）
            with tm.external_error_raised(exc):
                to_parquet(df, path, engine, compression=None)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_parquet_read_from_url(self, httpserver, datapath, df_compat, engine):
        # 如果 engine 不是 "auto"，则检查引入该引擎的能力
        if engine != "auto":
            pytest.importorskip(engine)
        # 使用 HTTP 服务器提供的 Parquet 数据，读取并断言结果 DataFrame 与预期相等
        with open(datapath("io", "data", "parquet", "simple.parquet"), mode="rb") as f:
            httpserver.serve_content(content=f.read())
            df = read_parquet(httpserver.url)
        tm.assert_frame_equal(df, df_compat)


class TestBasic(Base):
    def test_error(self, engine):
        # 对不支持的对象进行写入时的异常检查
        for obj in [
            pd.Series([1, 2, 3]),
            1,
            "foo",
            pd.Timestamp("20130101"),
            np.array([1, 2, 3]),
        ]:
            msg = "to_parquet only supports IO with DataFrames"
            self.check_error_on_write(obj, engine, ValueError, msg)

    def test_columns_dtypes(self, engine):
        # 测试列名称和数据类型的一致性
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})

        # 修改列名为 Unicode 编码
        df.columns = ["foo", "bar"]
        # 调用函数检查 DataFrame 的读写往返过程
        check_round_trip(df, engine)

    @pytest.mark.parametrize("compression", [None, "gzip", "snappy", "brotli"])
    def test_compression(self, engine, compression):
        # 测试不同压缩格式的 Parquet 文件写入和读取
        df = pd.DataFrame({"A": [1, 2, 3]})
        # 调用函数检查 DataFrame 的读写往返过程，并传递压缩参数
        check_round_trip(df, engine, write_kwargs={"compression": compression})
    # 定义测试函数，测试读取指定列的数据
    def test_read_columns(self, engine):
        # GH18154: 标识此测试的编号或描述
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})
        
        # 期望的数据框，只包含 "string" 列
        expected = pd.DataFrame({"string": list("abc")})
        
        # 调用自定义函数检查数据框的往返转换是否正确
        check_round_trip(
            df, engine, expected=expected, read_kwargs={"columns": ["string"]}
        )

    # 定义测试函数，测试读取时的过滤器功能
    def test_read_filters(self, engine, tmp_path):
        # 创建包含 "int" 和 "part" 列的数据框
        df = pd.DataFrame(
            {
                "int": list(range(4)),
                "part": list("aabb"),
            }
        )
        
        # 期望的数据框，只包含满足过滤条件的数据
        expected = pd.DataFrame({"int": [0, 1]})
        
        # 调用自定义函数检查数据框的往返转换是否正确，同时指定写入和读取的参数
        check_round_trip(
            df,
            engine,
            path=tmp_path,
            expected=expected,
            write_kwargs={"partition_cols": ["part"]},
            read_kwargs={"filters": [("part", "==", "a")], "columns": ["int"]},
            repeat=1,
        )

    # 定义测试函数，测试写入时的索引处理
    def test_write_index(self):
        # 导入 pyarrow 库，如果库不存在则跳过此测试
        pytest.importorskip("pyarrow")
        
        # 创建包含 "A" 列的数据框
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        # 调用自定义函数检查数据框的往返转换是否正确，使用 pyarrow 引擎
        check_round_trip(df, "pyarrow")

        # 不同类型的索引
        indexes = [
            [2, 3, 4],  # 整数索引
            pd.date_range("20130101", periods=3),  # 日期索引
            list("abc"),  # 字符串索引
            [1, 3, 4],  # 整数索引
        ]
        
        # 遍历不同类型的索引，测试其往返转换是否正确
        for index in indexes:
            df.index = index
            if isinstance(index, pd.DatetimeIndex):
                df.index = df.index._with_freq(None)  # 针对日期索引，去除频率信息以避免信息丢失
            check_round_trip(df, "pyarrow")

        # 带有元数据的索引
        df.index = [0, 1, 2]
        df.index.name = "foo"
        
        # 调用自定义函数检查数据框的往返转换是否正确，使用 pyarrow 引擎
        check_round_trip(df, "pyarrow")

    # 定义测试函数，测试写入多级索引时的处理
    def test_write_multiindex(self, pa):
        # 使用给定的 pyarrow 引擎
        engine = pa
        
        # 创建包含 "A" 列和多级索引的数据框
        df = pd.DataFrame({"A": [1, 2, 3]})
        index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
        df.index = index
        
        # 调用自定义函数检查数据框的往返转换是否正确
        check_round_trip(df, engine)

    # 定义测试函数，测试带有列和多级索引的数据框的处理
    def test_multiindex_with_columns(self, pa):
        # 使用给定的 pyarrow 引擎
        engine = pa
        
        # 创建包含随机数据的数据框，并设置多级索引
        dates = pd.date_range("01-Jan-2018", "01-Dec-2018", freq="MS")
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((2 * len(dates), 3)),
            columns=list("ABC"),
        )
        index1 = pd.MultiIndex.from_product(
            [["Level1", "Level2"], dates], names=["level", "date"]
        )
        index2 = index1.copy(names=None)
        
        # 遍历不同类型的索引，测试其往返转换是否正确
        for index in [index1, index2]:
            df.index = index

            # 调用自定义函数检查数据框的往返转换是否正确，同时指定读取的列和期望的结果
            check_round_trip(df, engine)
            check_round_trip(
                df, engine, read_kwargs={"columns": ["A", "B"]}, expected=df[["A", "B"]]
            )
    def test_write_ignoring_index(self, engine):
        # ENH 20768
        # 确保设置 index=False 时，从写入的 Parquet 文件中省略索引。
        
        # 创建一个包含两列 'a' 和 'b' 的 DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["q", "r", "s"]})

        # 定义写入参数，包括不压缩和不写入索引
        write_kwargs = {"compression": None, "index": False}

        # 由于我们省略了索引，期望加载的 DataFrame 会有默认的整数索引。
        expected = df.reset_index(drop=True)

        # 调用函数检查数据往返，即写入再读取，确保写入和读取的数据一致
        check_round_trip(df, engine, write_kwargs=write_kwargs, expected=expected)

        # 忽略自定义索引
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": ["q", "r", "s"]}, index=["zyx", "wvu", "tsr"]
        )

        # 再次进行数据往返检查
        check_round_trip(df, engine, write_kwargs=write_kwargs, expected=expected)

        # 同样忽略多级索引
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        df = pd.DataFrame(
            {"one": list(range(8)), "two": [-i for i in range(8)]}, index=arrays
        )

        expected = df.reset_index(drop=True)
        # 再次进行数据往返检查
        check_round_trip(df, engine, write_kwargs=write_kwargs, expected=expected)

    def test_write_column_multiindex(self, engine):
        # 无法使用非字符串列名写入列多级索引。
        
        # 创建一个带有非字符串列名的列多级索引的 DataFrame
        mi_columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)), columns=mi_columns
        )

        # 根据引擎类型检查写入时是否会出错
        if engine == "fastparquet":
            # 在 fastparquet 引擎下，检查是否会因为列名不是字符串而引发 TypeError
            self.check_error_on_write(
                df, engine, TypeError, "Column name must be a string"
            )
        elif engine == "pyarrow":
            # 在 pyarrow 引擎下，进行数据往返检查
            check_round_trip(df, engine)

    def test_write_column_multiindex_nonstring(self, engine):
        # GH #34777
        
        # 无法使用非字符串列名写入列多级索引
        
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            [1, 2, 1, 2, 1, 2, 1, 2],
        ]
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 8)), columns=arrays
        )
        df.columns.names = ["Level1", "Level2"]
        
        # 根据引擎类型检查写入时是否会出错
        if engine == "fastparquet":
            # 在 fastparquet 引擎下，检查是否会因为列名问题而引发 ValueError
            self.check_error_on_write(df, engine, ValueError, "Column name")
        elif engine == "pyarrow":
            # 在 pyarrow 引擎下，进行数据往返检查
            check_round_trip(df, engine)
    # 测试函数：写入具有多级索引且使用字符串列名的列
    def test_write_column_multiindex_string(self, pa):
        # GH #34777
        # Not supported in fastparquet as of 0.1.3
        # 引擎选择，通常由测试框架传入
        engine = pa

        # 使用字符串作为列名创建多级索引
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        # 创建一个随机数据的 DataFrame，形状为 (8, 8)，列名为 arrays
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 8)), columns=arrays
        )
        # 设置列级别名称为 ["ColLevel1", "ColLevel2"]
        df.columns.names = ["ColLevel1", "ColLevel2"]

        # 调用函数检查写入和读取的一致性
        check_round_trip(df, engine)

    # 测试函数：写入具有字符串列名的列索引
    def test_write_column_index_string(self, pa):
        # GH #34777
        # Not supported in fastparquet as of 0.1.3
        # 引擎选择，通常由测试框架传入
        engine = pa

        # 使用字符串作为列名创建索引
        arrays = ["bar", "baz", "foo", "qux"]
        # 创建一个随机数据的 DataFrame，形状为 (8, 4)，列名为 arrays
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)), columns=arrays
        )
        # 设置列名称为 "StringCol"
        df.columns.name = "StringCol"

        # 调用函数检查写入和读取的一致性
        check_round_trip(df, engine)

    # 测试函数：写入具有非字符串列名的列索引
    def test_write_column_index_nonstring(self, engine):
        # GH #34777

        # 使用非字符串作为列名创建索引
        arrays = [1, 2, 3, 4]
        # 创建一个随机数据的 DataFrame，形状为 (8, 4)，列名为 arrays
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)), columns=arrays
        )
        # 设置列名称为 "NonStringCol"
        df.columns.name = "NonStringCol"
        
        # 如果引擎是 "fastparquet"，则检查写入时的错误处理情况
        if engine == "fastparquet":
            self.check_error_on_write(
                df, engine, TypeError, "Column name must be a string"
            )
        else:
            # 调用函数检查写入和读取的一致性
            check_round_trip(df, engine)
    # 定义一个测试函数，测试不同的数据类型后端
    def test_dtype_backend(self, engine, request):
        # 导入并检查是否存在 pyarrow.parquet，若不存在则跳过测试
        pq = pytest.importorskip("pyarrow.parquet")

        if engine == "fastparquet":
            # 当引擎为 "fastparquet" 时，手动禁用其可空数据类型支持，待讨论
            mark = pytest.mark.xfail(
                reason="Fastparquet nullable dtype support is disabled"
            )
            # 应用标记以跳过测试
            request.applymarker(mark)

        # 创建一个包含不同数据类型的 pyarrow 表
        table = pyarrow.table(
            {
                "a": pyarrow.array([1, 2, 3, None], "int64"),
                "b": pyarrow.array([1, 2, 3, None], "uint8"),
                "c": pyarrow.array(["a", "b", "c", None]),
                "d": pyarrow.array([True, False, True, None]),
                # 测试即使没有空值，也使用可空数据类型
                "e": pyarrow.array([1, 2, 3, 4], "int64"),
                # GH 45694
                "f": pyarrow.array([1.0, 2.0, 3.0, None], "float32"),
                "g": pyarrow.array([1.0, 2.0, 3.0, None], "float64"),
            }
        )
        
        # 确保在一个干净的环境中执行以下操作
        with tm.ensure_clean() as path:
            # 使用 pyarrow 手动写入表格到 parquet 文件
            pq.write_table(table, path)
            # 使用指定的引擎读取 parquet 文件
            result1 = read_parquet(path, engine=engine)
            # 使用 numpy_nullable 数据类型后端读取 parquet 文件
            result2 = read_parquet(path, engine=engine, dtype_backend="numpy_nullable")

        # 断言结果的数据类型是否符合预期
        assert result1["a"].dtype == np.dtype("float64")
        # 创建一个预期的 Pandas DataFrame
        expected = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3, None], dtype="Int64"),
                "b": pd.array([1, 2, 3, None], dtype="UInt8"),
                "c": pd.array(["a", "b", "c", None], dtype="string"),
                "d": pd.array([True, False, True, None], dtype="boolean"),
                "e": pd.array([1, 2, 3, 4], dtype="Int64"),
                "f": pd.array([1.0, 2.0, 3.0, None], dtype="Float32"),
                "g": pd.array([1.0, 2.0, 3.0, None], dtype="Float64"),
            }
        )

        if engine == "fastparquet":
            # 当引擎为 "fastparquet" 时，不支持字符串列
            # 只支持整数和布尔类型
            result2 = result2.drop("c", axis=1)
            expected = expected.drop("c", axis=1)

        # 使用测试模块中的方法来比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result2, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            "Int64",
            "UInt8",
            "boolean",
            "object",
            "datetime64[ns, UTC]",
            "float",
            "period[D]",
            "Float64",
            "string",
        ],
    )
    # 定义一个测试方法，测试读取空数组的情况，传入参数包括 pa 和 dtype
    def test_read_empty_array(self, pa, dtype):
        # GH #41241
        # 创建一个空的 pandas DataFrame 对象，其中包含一个空的 pd.array 对象，数据类型为参数 dtype 指定的类型
        df = pd.DataFrame(
            {
                "value": pd.array([], dtype=dtype),
            }
        )
        # 导入 pyarrow 库，如果版本小于 11.0.0 则跳过测试
        pytest.importorskip("pyarrow", "11.0.0")
        # GH 45694
        # 初始化变量 expected 为 None
        expected = None
        # 如果 dtype 参数为 "float"，则创建一个预期的 pandas DataFrame 对象
        if dtype == "float":
            expected = pd.DataFrame(
                {
                    "value": pd.array([], dtype="Float64"),
                }
            )
        # 调用 check_round_trip 函数，测试数据框 df 和 pa 对象，传入额外的读取参数，并传入预期的结果 expected
        check_round_trip(
            df, pa, read_kwargs={"dtype_backend": "numpy_nullable"}, expected=expected
        )
    @pytest.mark.xfail(reason="datetime_with_nat unit doesn't round-trip")
    # 标记该测试为预期失败，原因是datetime_with_nat单元不能往返处理

    def test_basic(self, pa, df_full):
        df = df_full
        pytest.importorskip("pyarrow", "11.0.0")
        # 导入pyarrow库，要求版本不低于11.0.0

        # additional supported types for pyarrow
        dti = pd.date_range("20130101", periods=3, tz="Europe/Brussels")
        dti = dti._with_freq(None)  # freq doesn't round-trip
        # 创建包含时区信息的日期时间索引，并移除频率信息，因为频率不能往返处理
        df["datetime_tz"] = dti
        df["bool_with_none"] = [True, None, True]
        # 向DataFrame中添加额外支持的类型，如带有时区的日期时间和带有None值的布尔类型

        check_round_trip(df, pa)
        # 调用函数检查DataFrame是否能在pyarrow中进行往返处理

    def test_basic_subset_columns(self, pa, df_full):
        # GH18628
        # GitHub issue 18628

        df = df_full
        # 使用完整的DataFrame

        # additional supported types for pyarrow
        df["datetime_tz"] = pd.date_range("20130101", periods=3, tz="Europe/Brussels")
        # 向DataFrame中添加带有时区信息的日期时间列

        check_round_trip(
            df,
            pa,
            expected=df[["string", "int"]],
            read_kwargs={"columns": ["string", "int"]},
        )
        # 调用函数检查DataFrame是否能在pyarrow中进行往返处理，并指定预期输出和读取参数

    def test_to_bytes_without_path_or_buf_provided(self, pa, df_full):
        # GH 37105
        # GitHub issue 37105

        buf_bytes = df_full.to_parquet(engine=pa)
        # 将DataFrame写入parquet格式的字节流
        assert isinstance(buf_bytes, bytes)
        # 断言字节流的类型为bytes

        buf_stream = BytesIO(buf_bytes)
        # 创建字节流对象
        res = read_parquet(buf_stream)
        # 读取parquet格式数据

        expected = df_full.copy()
        expected.loc[1, "string_with_nan"] = None
        if pa_version_under11p0:
            expected["datetime_with_nat"] = expected["datetime_with_nat"].astype(
                "M8[ns]"
            )
        else:
            expected["datetime_with_nat"] = expected["datetime_with_nat"].astype(
                "M8[ms]"
            )
        # 根据pyarrow版本不同，将列datetime_with_nat的数据类型转换为特定的时间戳格式

        tm.assert_frame_equal(res, expected)
        # 断言读取的DataFrame与预期的DataFrame相等

    def test_duplicate_columns(self, pa):
        # not currently able to handle duplicate columns
        # 当前无法处理重复列名

        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list("aaa")).copy()
        # 创建包含重复列名的DataFrame

        self.check_error_on_write(df, pa, ValueError, "Duplicate column names found")
        # 调用函数检查在写入时是否能正确处理重复列名的错误情况

    def test_timedelta(self, pa):
        df = pd.DataFrame({"a": pd.timedelta_range("1 day", periods=3)})
        # 创建包含时间差数据的DataFrame

        check_round_trip(df, pa)
        # 调用函数检查DataFrame是否能在pyarrow中进行往返处理

    def test_unsupported(self, pa):
        # mixed python objects
        # 包含混合Python对象

        df = pd.DataFrame({"a": ["a", 1, 2.0]})
        # 创建包含混合数据类型的DataFrame

        # pyarrow 0.11 raises ArrowTypeError
        # older pyarrows raise ArrowInvalid
        self.check_external_error_on_write(df, pa, pyarrow.ArrowException)
        # 调用函数检查在写入时是否能正确处理混合数据类型的异常情况

    def test_unsupported_float16(self, pa):
        # #44847, #44914
        # Not able to write float 16 column using pyarrow.
        # 无法使用pyarrow写入float16类型列

        data = np.arange(2, 10, dtype=np.float16)
        df = pd.DataFrame(data=data, columns=["fp16"])
        # 创建包含float16类型数据的DataFrame

        if pa_version_under15p0:
            self.check_external_error_on_write(df, pa, pyarrow.ArrowException)
            # 如果pyarrow版本低于15.0，调用函数检查在写入时是否能正确处理float16类型的异常情况
        else:
            check_round_trip(df, pa)
            # 否则，调用函数检查DataFrame是否能在pyarrow中进行往返处理

    @pytest.mark.xfail(
        is_platform_windows(),
        reason=(
            "PyArrow does not cleanup of partial files dumps when unsupported "
            "dtypes are passed to_parquet function in windows"
        ),
    )
    # 标记该测试为预期失败，原因是当在Windows中传递不支持的数据类型给to_parquet函数时，pyarrow无法清理部分文件的转储
    @pytest.mark.skipif(not pa_version_under15p0, reason="float16 works on 15")
    @pytest.mark.parametrize("path_type", [str, pathlib.Path])
    def test_unsupported_float16_cleanup(self, pa, path_type):
        # 标记此测试用例为条件性跳过，如果不满足条件 pa_version_under15p0
        # 该测试用例测试 pyarrow 无法处理 float16 列的情况
        # 测试在出现错误时 pyarrow 的清理行为
        data = np.arange(2, 10, dtype=np.float16)
        # 创建包含 float16 数据的 DataFrame
        df = pd.DataFrame(data=data, columns=["fp16"])

        with tm.ensure_clean() as path_str:
            # 确保路径 path_str 为空并返回
            path = path_type(path_str)
            # 使用指定的路径类型 path_type 创建路径对象 path
            with tm.external_error_raised(pyarrow.ArrowException):
                # 在外部引发 pyarrow.ArrowException 异常的上下文中执行以下代码块
                df.to_parquet(path=path, engine=pa)
            # 断言路径 path 不是一个文件
            assert not os.path.isfile(path)

    def test_categorical(self, pa):
        # 支持自版本 0.7.0 起
        # 创建一个空的 DataFrame
        df = pd.DataFrame()
        # 创建一个包含分类数据的列 'a'
        df["a"] = pd.Categorical(list("abcdef"))

        # 测试空值、无序值以及未观察到的分类
        df["b"] = pd.Categorical(
            ["bar", "foo", "foo", "bar", None, "bar"],
            dtype=pd.CategoricalDtype(["foo", "bar", "baz"]),
        )

        # 测试有序标志
        df["c"] = pd.Categorical(
            ["a", "b", "c", "a", "c", "b"], categories=["b", "c", "d"], ordered=True
        )

        # 调用函数 check_round_trip 以验证 DataFrame 的 round-trip
        check_round_trip(df, pa)

    @pytest.mark.single_cpu
    def test_s3_roundtrip_explicit_fs(self, df_compat, s3_public_bucket, pa, s3so):
        # GH #19134
        # 设置 s3 文件系统选项
        s3so = {"storage_options": s3so}
        # 调用函数 check_round_trip 以验证 DataFrame 在 S3 中的 round-trip
        check_round_trip(
            df_compat,
            pa,
            path=f"{s3_public_bucket.name}/pyarrow.parquet",
            read_kwargs=s3so,
            write_kwargs=s3so,
        )

    @pytest.mark.single_cpu
    def test_s3_roundtrip(self, df_compat, s3_public_bucket, pa, s3so):
        # GH #19134
        # 设置 s3 文件系统选项
        s3so = {"storage_options": s3so}
        # 调用函数 check_round_trip 以验证 DataFrame 在 S3 中的 round-trip
        check_round_trip(
            df_compat,
            pa,
            path=f"s3://{s3_public_bucket.name}/pyarrow.parquet",
            read_kwargs=s3so,
            write_kwargs=s3so,
        )

    @pytest.mark.single_cpu
    @pytest.mark.parametrize("partition_col", [["A"], []])
    def test_s3_roundtrip_for_dir(
        self, df_compat, s3_public_bucket, pa, partition_col, s3so
    ):
        # 测试 DataFrame 在 S3 中目录的 round-trip
    ):
        # 导入必要的依赖库 pytest 和 s3fs，并在缺失时跳过测试
        pytest.importorskip("s3fs")
        # 创建一个预期的数据框副本，用于后续比较
        expected_df = df_compat.copy()

        # 如果存在分区列 partition_col，则进行以下处理
        if partition_col:
            # 将预期的数据框中指定的分区列 partition_col 的数据类型转换为 np.int32
            expected_df = expected_df.astype(dict.fromkeys(partition_col, np.int32))
            # 将分区列 partition_col 的数据类型设定为 "category"
            partition_col_type = "category"
            expected_df[partition_col] = expected_df[partition_col].astype(
                partition_col_type
            )

        # 调用 check_round_trip 函数，验证数据框的序列化和反序列化过程
        check_round_trip(
            df_compat,
            pa,
            expected=expected_df,
            path=f"s3://{s3_public_bucket.name}/parquet_dir",
            read_kwargs={"storage_options": s3so},
            write_kwargs={
                "partition_cols": partition_col,
                "compression": None,
                "storage_options": s3so,
            },
            check_like=True,
            repeat=1,
        )

    # 测试函数：验证从文件类对象读取支持
    def test_read_file_like_obj_support(self, df_compat):
        # 导入必要的依赖库 pytest 和 pyarrow，并在缺失时跳过测试
        pytest.importorskip("pyarrow")
        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 将数据框 df_compat 写入 Parquet 格式到缓冲区
        df_compat.to_parquet(buffer)
        # 从缓冲区读取 Parquet 格式的数据框
        df_from_buf = read_parquet(buffer)
        # 使用 assert_frame_equal 检查两个数据框是否相等
        tm.assert_frame_equal(df_compat, df_from_buf)

    # 测试函数：验证 expanduser 函数的支持
    def test_expand_user(self, df_compat, monkeypatch):
        # 导入必要的依赖库 pytest 和 pyarrow，并在缺失时跳过测试
        pytest.importorskip("pyarrow")
        # 使用 monkeypatch 设置环境变量 HOME 和 USERPROFILE 为 "TestingUser"
        monkeypatch.setenv("HOME", "TestingUser")
        monkeypatch.setenv("USERPROFILE", "TestingUser")
        # 使用 pytest.raises 检测是否抛出 OSError 异常，并匹配包含 "TestingUser" 的错误消息
        with pytest.raises(OSError, match=r".*TestingUser.*"):
            read_parquet("~/file.parquet")
        with pytest.raises(OSError, match=r".*TestingUser.*"):
            df_compat.to_parquet("~/file.parquet")

    # 测试函数：验证分区列的支持
    def test_partition_cols_supported(self, tmp_path, pa, df_full):
        # GH #23283
        # 指定分区列名列表
        partition_cols = ["bool", "int"]
        # 使用完整数据框 df_full
        df = df_full
        # 将数据框 df 写入 Parquet 格式到临时路径 tmp_path，并指定分区列和压缩方式
        df.to_parquet(tmp_path, partition_cols=partition_cols, compression=None)
        # 检查临时路径 tmp_path 下的分区名称是否符合预期
        check_partition_names(tmp_path, partition_cols)
        # 使用 read_parquet 函数读取 Parquet 格式的数据框，并检查其形状是否与 df 相同
        assert read_parquet(tmp_path).shape == df.shape

    # 测试函数：验证字符串分区列的支持
    def test_partition_cols_string(self, tmp_path, pa, df_full):
        # GH #27117
        # 指定字符串形式的分区列名
        partition_cols = "bool"
        partition_cols_list = [partition_cols]
        # 使用完整数据框 df_full
        df = df_full
        # 将数据框 df 写入 Parquet 格式到临时路径 tmp_path，并指定分区列和压缩方式
        df.to_parquet(tmp_path, partition_cols=partition_cols, compression=None)
        # 检查临时路径 tmp_path 下的分区名称是否符合预期
        check_partition_names(tmp_path, partition_cols_list)
        # 使用 read_parquet 函数读取 Parquet 格式的数据框，并检查其形状是否与 df 相同
        assert read_parquet(tmp_path).shape == df.shape

    # 测试函数：验证 pathlib.Path 类型的路径分区列支持
    @pytest.mark.parametrize(
        "path_type", [str, lambda x: x], ids=["string", "pathlib.Path"]
    )
    def test_partition_cols_pathlib(self, tmp_path, pa, df_compat, path_type):
        # GH 35902
        # 指定分区列名
        partition_cols = "B"
        partition_cols_list = [partition_cols]
        # 使用兼容数据框 df_compat
        df = df_compat

        # 使用 path_type 参数构建路径对象 path
        path = path_type(tmp_path)
        # 将数据框 df 写入 Parquet 格式到路径 path，并指定分区列
        df.to_parquet(path, partition_cols=partition_cols_list)
        # 使用 read_parquet 函数读取 Parquet 格式的数据框，并检查其形状是否与 df 相同
        assert read_parquet(path).shape == df.shape

    # 测试函数：验证空数据框的支持
    def test_empty_dataframe(self, pa):
        # GH #27339
        # 创建一个空的 Pandas 数据框
        df = pd.DataFrame(index=[], columns=[])
        # 调用 check_round_trip 函数，验证空数据框的序列化和反序列化过程
        check_round_trip(df, pa)
    def test_write_with_schema(self, pa):
        import pyarrow  # 导入 pyarrow 库

        df = pd.DataFrame({"x": [0, 1]})  # 创建一个包含一列 'x' 的 DataFrame
        schema = pyarrow.schema([pyarrow.field("x", type=pyarrow.bool_())])  # 创建一个包含 'x' 列的布尔类型 schema
        out_df = df.astype(bool)  # 将 DataFrame 的数据类型转换为布尔类型
        check_round_trip(df, pa, write_kwargs={"schema": schema}, expected=out_df)  # 使用给定的 schema 进行往返检查

    def test_additional_extension_arrays(self, pa):
        # 测试通过 __arrow_array__ 协议支持的额外 ExtensionArrays
        pytest.importorskip("pyarrow")  # 如果没有 pyarrow 库则跳过测试
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype="Int64"),  # 创建包含 'a' 列的 Int64 类型 Series
                "b": pd.Series([1, 2, 3], dtype="UInt32"),  # 创建包含 'b' 列的 UInt32 类型 Series
                "c": pd.Series(["a", None, "c"], dtype="string"),  # 创建包含 'c' 列的字符串类型 Series
            }
        )
        check_round_trip(df, pa)  # 进行 DataFrame 和 pyarrow 库之间的往返检查

        df = pd.DataFrame({"a": pd.Series([1, 2, 3, None], dtype="Int64")})  # 创建包含 'a' 列的 Int64 类型 Series
        check_round_trip(df, pa)  # 进行 DataFrame 和 pyarrow 库之间的往返检查

    def test_pyarrow_backed_string_array(self, pa, string_storage):
        # 测试通过 __arrow_array__ 协议支持的 ArrowStringArray
        pytest.importorskip("pyarrow")  # 如果没有 pyarrow 库则跳过测试
        df = pd.DataFrame({"a": pd.Series(["a", None, "c"], dtype="string[pyarrow]")})  # 创建包含 'a' 列的 pyarrow 支持的字符串类型 Series
        with pd.option_context("string_storage", string_storage):  # 设置字符串存储选项
            check_round_trip(df, pa, expected=df.astype(f"string[{string_storage}]"))  # 进行 DataFrame 和 pyarrow 库之间的往返检查，期望结果是特定的字符串存储类型

    def test_additional_extension_types(self, pa):
        # 测试通过 __arrow_array__ 协议 + 自定义 ExtensionType 支持的额外 ExtensionArrays
        pytest.importorskip("pyarrow")  # 如果没有 pyarrow 库则跳过测试
        df = pd.DataFrame(
            {
                "c": pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)]),  # 创建包含 'c' 列的 IntervalIndex
                "d": pd.period_range("2012-01-01", periods=3, freq="D"),  # 创建包含 'd' 列的 PeriodIndex
                # GH-45881 issue with interval with datetime64[ns] subtype
                "e": pd.IntervalIndex.from_breaks(  # 创建包含 'e' 列的 IntervalIndex
                    pd.date_range("2012-01-01", periods=4, freq="D")  # 使用日期范围创建 IntervalIndex
                ),
            }
        )
        check_round_trip(df, pa)  # 进行 DataFrame 和 pyarrow 库之间的往返检查

    def test_timestamp_nanoseconds(self, pa):
        # 在版本 2.6 中，pyarrow 默认写入纳秒级时间戳，所以这应该可以正常工作
        # 注意，在之前的 pyarrow (<7.0.0) 中，只有伪版本 2.0 是可用的
        ver = "2.6"  # 设定版本号为 "2.6"
        df = pd.DataFrame({"a": pd.date_range("2017-01-01", freq="1ns", periods=10)})  # 创建包含 'a' 列的纳秒级日期范围 DataFrame
        check_round_trip(df, pa, write_kwargs={"version": ver})  # 进行 DataFrame 和 pyarrow 库之间的往返检查，使用指定的版本号参数
    # 测试时区感知的索引功能
    def test_timezone_aware_index(self, request, pa, timezone_aware_date_list):
        # 导入 pyarrow，如果版本低于 11.0.0 则跳过测试
        pytest.importorskip("pyarrow", "11.0.0")

        # 如果日期列表不是 UTC 时区，则标记为预期失败，并给出原因链接
        if timezone_aware_date_list.tzinfo != datetime.timezone.utc:
            request.applymarker(
                pytest.mark.xfail(
                    reason="temporary skip this test until it is properly resolved: "
                    "https://github.com/pandas-dev/pandas/issues/37286"
                )
            )

        # 创建包含五个相同日期的索引
        idx = 5 * [timezone_aware_date_list]
        
        # 创建 DataFrame，使用上述索引和数据列名为 'index_as_col'
        df = pd.DataFrame(index=idx, data={"index_as_col": idx})

        # 见 gh-36004
        # 仅比较时间（区域）值，跳过它们的类：
        # pyarrow 总是使用 pytz.FixedOffset() 创建固定偏移的时区
        # 即使最初是 datetime.timezone()
        #
        # 在技术上它们是相同的：
        # 它们都实现了 datetime.tzinfo
        # 它们都包装了 datetime.timedelta()
        # 此用例将分辨率设置为 1 分钟

        # 期望结果等于 DataFrame 本身
        expected = df[:]

        # 如果 pyarrow 版本小于 11.0.0，则将索引转换为纳秒单位
        if pa_version_under11p0:
            expected.index = expected.index.as_unit("ns")

        # 调用函数检查数据的往返传输
        check_round_trip(df, pa, check_dtype=False, expected=expected)


    # 测试过滤行组功能
    def test_filter_row_groups(self, pa):
        # 导入 pyarrow，如果未安装则跳过
        pytest.importorskip("pyarrow")

        # 创建包含一列 0 到 2 的 DataFrame
        df = pd.DataFrame({"a": list(range(3))})

        # 确保在使用路径之前进行清理操作
        with tm.ensure_clean() as path:
            # 将 DataFrame 保存为 parquet 文件
            df.to_parquet(path, engine=pa)

            # 读取 parquet 文件，应用指定的过滤条件
            result = read_parquet(path, pa, filters=[("a", "==", 0)])

        # 断言结果中的行数为 1
        assert len(result) == 1


    # 测试读取 dtype 后端配置
    @pytest.mark.filterwarnings("ignore:make_block is deprecated:DeprecationWarning")
    def test_read_dtype_backend_pyarrow_config(self, pa, df_full):
        import pyarrow

        # 使用完整的 DataFrame
        df = df_full

        # 创建带有时区的日期时间索引
        dti = pd.date_range("20130101", periods=3, tz="Europe/Brussels")
        dti = dti._with_freq(None)  # 频率不会往返传输
        df["datetime_tz"] = dti

        # 添加一个包含布尔值和 None 的列
        df["bool_with_none"] = [True, None, True]

        # 将 Pandas DataFrame 转换为 pyarrow Table
        pa_table = pyarrow.Table.from_pandas(df)

        # 预期结果从 pyarrow Table 转换回 Pandas DataFrame
        expected = pa_table.to_pandas(types_mapper=pd.ArrowDtype)

        # 如果 pyarrow 版本低于 13.0.0
        if pa_version_under13p0:
            # pyarrow 将日期时间推断为微秒而不是纳秒
            expected["datetime"] = expected["datetime"].astype("timestamp[us][pyarrow]")

            # 为带有欧洲布鲁塞尔时区的 datetime_tz 列设置类型
            expected["datetime_tz"] = expected["datetime_tz"].astype(
                pd.ArrowDtype(pyarrow.timestamp(unit="us", tz="Europe/Brussels"))
            )

        # 设置带有 NAT 的 datetime_with_nat 列的类型
        expected["datetime_with_nat"] = expected["datetime_with_nat"].astype(
            "timestamp[ms][pyarrow]"
        )

        # 调用函数检查数据的往返传输
        check_round_trip(
            df,
            engine=pa,
            read_kwargs={"dtype_backend": "pyarrow"},
            expected=expected,
        )
    def test_read_dtype_backend_pyarrow_config_index(self, pa):
        # 使用 pyarrow 引擎测试读取和写入 DataFrame，设置索引的数据类型为 'int64[pyarrow]'
        df = pd.DataFrame(
            {"a": [1, 2]}, index=pd.Index([3, 4], name="test"), dtype="int64[pyarrow]"
        )
        # 复制 DataFrame 以备后续比较
        expected = df.copy()
        import pyarrow

        # 检查 pyarrow 版本，如果大于 "11.0.0"，则修改索引的数据类型为 'int64[pyarrow]'
        if Version(pyarrow.__version__) > Version("11.0.0"):
            expected.index = expected.index.astype("int64[pyarrow]")
        # 调用检查函数，验证数据框的往返写入
        check_round_trip(
            df,
            engine=pa,
            read_kwargs={"dtype_backend": "pyarrow"},
            expected=expected,
        )

    @pytest.mark.xfail(
        pa_version_under17p0, reason="pa.pandas_compat passes 'datetime64' to .astype"
    )
    def test_columns_dtypes_not_invalid(self, pa):
        # 创建包含不同数据类型列的 DataFrame
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})

        # 设置列名为数字，验证数据框的往返写入
        df.columns = [0, 1]
        check_round_trip(df, pa)

        # 设置列名为字节，预期会触发 NotImplementedError 异常，匹配错误信息中包含 '|S3'
        df.columns = [b"foo", b"bar"]
        with pytest.raises(NotImplementedError, match="|S3"):
            # 在读取 Parquet 格式时，字节列写入失败
            check_round_trip(df, pa)

        # 设置列名为 Python 对象（日期时间），验证数据框的往返写入
        df.columns = [
            datetime.datetime(2011, 1, 1, 0, 0),
            datetime.datetime(2011, 1, 1, 1, 1),
        ]
        check_round_trip(df, pa)

    def test_empty_columns(self, pa):
        # GH 52034：测试创建只有索引没有列的 DataFrame
        df = pd.DataFrame(index=pd.Index(["a", "b", "c"], name="custom name"))
        check_round_trip(df, pa)

    def test_df_attrs_persistence(self, tmp_path, pa):
        # 测试 DataFrame 的属性持久性
        path = tmp_path / "test_df_metadata.p"
        df = pd.DataFrame(data={1: [1]})
        df.attrs = {"test_attribute": 1}
        # 将 DataFrame 写入 Parquet 文件
        df.to_parquet(path, engine=pa)
        # 读取 Parquet 文件，并验证属性与原始 DataFrame 是否相同
        new_df = read_parquet(path, engine=pa)
        assert new_df.attrs == df.attrs

    def test_string_inference(self, tmp_path, pa):
        # GH#54431：测试推断字符串类型的处理
        path = tmp_path / "test_string_inference.p"
        df = pd.DataFrame(data={"a": ["x", "y"]}, index=["a", "b"])
        # 将 DataFrame 写入 Parquet 文件，使用 pyarrow 引擎
        df.to_parquet(path, engine="pyarrow")
        with pd.option_context("future.infer_string", True):
            # 读取 Parquet 文件，预期推断数据类型为 'string[pyarrow_numpy]'
            result = read_parquet(path, engine="pyarrow")
        # 准备预期的 DataFrame 结果，包含推断后的数据类型
        expected = pd.DataFrame(
            data={"a": ["x", "y"]},
            dtype="string[pyarrow_numpy]",
            index=pd.Index(["a", "b"], dtype="string[pyarrow_numpy]"),
        )
        # 比较结果 DataFrame 与预期 DataFrame 是否相同
        tm.assert_frame_equal(result, expected)

    @pytest.mark.skipif(pa_version_under11p0, reason="not supported before 11.0")
    def test_roundtrip_decimal(self, tmp_path, pa):
        # GH#54768：测试 Decimal 类型的往返写入
        import pyarrow as pa

        path = tmp_path / "decimal.p"
        # 创建包含 Decimal 类型数据的 DataFrame，数据类型设置为 'string[pyarrow]'
        df = pd.DataFrame({"a": [Decimal("123.00")]}, dtype="string[pyarrow]")
        # 将 DataFrame 写入 Parquet 文件，指定 schema 为 Decimal 类型
        df.to_parquet(path, schema=pa.schema([("a", pa.decimal128(5))]))
        # 读取 Parquet 文件，验证写入后的数据是否正确
        result = read_parquet(path)
        # 准备预期的 DataFrame 结果，数据类型为 'string[python]'
        expected = pd.DataFrame({"a": ["123"]}, dtype="string[python]")
        # 比较结果 DataFrame 与预期 DataFrame 是否相同
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试推断大字符串类型
    def test_infer_string_large_string_type(self, tmp_path, pa):
        # 导入必要的库：pyarrow和pyarrow.parquet
        import pyarrow as pa
        import pyarrow.parquet as pq

        # 创建临时文件路径
        path = tmp_path / "large_string.p"

        # 创建一个包含大字符串类型数据的表格
        table = pa.table({"a": pa.array([None, "b", "c"], pa.large_string())})
        
        # 将表格写入Parquet文件
        pq.write_table(table, path)

        # 使用未来的推断字符串选项上下文环境
        with pd.option_context("future.infer_string", True):
            # 读取Parquet文件
            result = read_parquet(path)
        
        # 期望的结果DataFrame
        expected = pd.DataFrame(
            data={"a": [None, "b", "c"]},
            dtype="string[pyarrow_numpy]",
            columns=pd.Index(["a"], dtype="string[pyarrow_numpy]"),
        )
        
        # 断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # NOTE: 这个测试默认情况下不会运行，因为它需要大量内存 (>5GB)
    # @pytest.mark.slow
    # def test_string_column_above_2GB(self, tmp_path, pa):
    #     # https://github.com/pandas-dev/pandas/issues/55606
    #     # above 2GB of string data
    #     v1 = b"x" * 100000000
    #     v2 = b"x" * 147483646
    #     df = pd.DataFrame({"strings": [v1] * 20 + [v2] + ["x"] * 20}, dtype="string")
    #     df.to_parquet(tmp_path / "test.parquet")
    #     result = read_parquet(tmp_path / "test.parquet")
    #     assert result["strings"].dtype == "string"
    # FIXME: 不要留下注释掉的代码
    # 定义一个名为 TestParquetFastParquet 的测试类，继承自 Base 类
    class TestParquetFastParquet(Base):
        
        # 使用 pytest.mark.xfail 装饰器标记该测试方法为预期失败，原因是 datetime_with_nat 会得到不正确的值
        @pytest.mark.xfail(reason="datetime_with_nat gets incorrect values")
        def test_basic(self, fp, df_full):
            # 导入并验证是否存在 pytz 模块，如果不存在则跳过测试
            pytz = pytest.importorskip("pytz")
            # 设置时区为 US/Eastern
            tz = pytz.timezone("US/Eastern")
            # 使用预设的数据框 df_full
            df = df_full

            # 创建一个带时区的日期时间索引，从 "20130101" 开始，周期为 3 天，时区为 tz
            dti = pd.date_range("20130101", periods=3, tz=tz)
            # 移除日期时间索引的频率信息，因为频率信息不能往返
            dti = dti._with_freq(None)  # freq doesn't round-trip
            # 将生成的日期时间索引添加到数据框 df 中的 "datetime_tz" 列
            df["datetime_tz"] = dti
            # 创建一个时间增量的时间间隔，从 "1 day" 开始，周期为 3
            df["timedelta"] = pd.timedelta_range("1 day", periods=3)
            # 使用自定义函数 check_round_trip 检查数据框 df 的写入和读取过程
            check_round_trip(df, fp)

        # 检查列名数据类型无效的情况
        def test_columns_dtypes_invalid(self, fp):
            # 创建一个包含字符串和整数的数据框 df
            df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})

            # 设置预期的错误类型为 TypeError
            err = TypeError
            # 设置预期的错误消息为 "Column name must be a string"
            msg = "Column name must be a string"

            # 当列名是数字时，检查是否会触发错误
            df.columns = [0, 1]
            self.check_error_on_write(df, fp, err, msg)

            # 当列名是字节时，检查是否会触发错误
            df.columns = [b"foo", b"bar"]
            self.check_error_on_write(df, fp, err, msg)

            # 当列名是 Python 对象时，检查是否会触发错误
            df.columns = [
                datetime.datetime(2011, 1, 1, 0, 0),
                datetime.datetime(2011, 1, 1, 1, 1),
            ]
            self.check_error_on_write(df, fp, err, msg)

        # 检查处理重复列名的情况
        def test_duplicate_columns(self, fp):
            # 创建一个含有重复列名的数据框 df，目前 Fastparquet 不支持处理重复列名
            df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list("aaa")).copy()
            # 设置预期的错误消息为 "Cannot create parquet dataset with duplicate column names"
            msg = "Cannot create parquet dataset with duplicate column names"
            self.check_error_on_write(df, fp, ValueError, msg)

        # 检查布尔类型和 None 值的处理
        def test_bool_with_none(self, fp):
            # 创建一个包含布尔值和 None 的数据框 df
            df = pd.DataFrame({"a": [True, None, False]})
            # 创建一个预期结果数据框 expected，将 None 转换为 NaN，数据类型为 float16
            expected = pd.DataFrame({"a": [1.0, np.nan, 0.0]}, dtype="float16")
            # 使用自定义函数 check_round_trip 检查数据框 df 的写入和读取过程，忽略数据类型的检查
            # 由于 Fastparquet 0.7.1 版本的 bug，dtype 可能会变为 float64
            check_round_trip(df, fp, expected=expected, check_dtype=False)

        # 检查不支持的数据类型处理
        def test_unsupported(self, fp):
            # 创建一个包含周期类型的数据框 df
            df = pd.DataFrame({"a": pd.period_range("2013", freq="M", periods=3)})
            # 设置预期的错误类型为 ValueError，不检查具体的错误消息
            # fastparquet 会抛出错误，不关心具体的错误消息
            self.check_error_on_write(df, fp, ValueError, None)

            # 创建一个包含混合类型的数据框 df
            df = pd.DataFrame({"a": ["a", 1, 2.0]})
            # 设置预期的错误类型为 ValueError，预期的错误消息为 "Can't infer object conversion type"
            msg = "Can't infer object conversion type"
            self.check_error_on_write(df, fp, ValueError, msg)

        # 检查分类数据的处理
        def test_categorical(self, fp):
            # 创建一个包含分类数据的数据框 df
            df = pd.DataFrame({"a": pd.Categorical(list("abc"))})
            # 使用自定义函数 check_round_trip 检查数据框 df 的写入和读取过程
            check_round_trip(df, fp)

        # 检查过滤行组的处理
        def test_filter_row_groups(self, fp):
            d = {"a": list(range(3))}
            df = pd.DataFrame(d)
            # 确保路径 path 是干净的，避免文件已经存在
            with tm.ensure_clean() as path:
                # 将数据框 df 写入 Parquet 文件到指定路径 path，使用引擎 fp，不压缩，行组偏移量设置为 1
                df.to_parquet(path, engine=fp, compression=None, row_group_offsets=1)
                # 使用自定义函数 read_parquet 读取指定路径 path 的 Parquet 文件，使用引擎 fp，应用过滤条件 [("a", "==", 0)]
                result = read_parquet(path, fp, filters=[("a", "==", 0)])
            # 断言结果的长度为 1
            assert len(result) == 1

        # 使用 pytest.mark.single_cpu 标记下一个测试方法为单 CPU 执行
        @pytest.mark.single_cpu
    # 测试确保在读写时没有数据损失或格式错误，关注GitHub问题 #19134
    def test_s3_roundtrip(self, df_compat, s3_public_bucket, fp, s3so):
        check_round_trip(
            df_compat,
            fp,
            path=f"s3://{s3_public_bucket.name}/fastparquet.parquet",
            read_kwargs={"storage_options": s3so},
            write_kwargs={"compression": None, "storage_options": s3so},
        )

    # 测试确保分区列在fastparquet中受到支持，关注GitHub问题 #23283
    def test_partition_cols_supported(self, tmp_path, fp, df_full):
        partition_cols = ["bool", "int"]
        df = df_full
        df.to_parquet(
            tmp_path,
            engine="fastparquet",
            partition_cols=partition_cols,
            compression=None,
        )
        assert os.path.exists(tmp_path)
        import fastparquet
        # 获取实际分区列
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 2

    # 测试确保字符串形式的分区列在fastparquet中受到支持，关注GitHub问题 #27117
    def test_partition_cols_string(self, tmp_path, fp, df_full):
        partition_cols = "bool"
        df = df_full
        df.to_parquet(
            tmp_path,
            engine="fastparquet",
            partition_cols=partition_cols,
            compression=None,
        )
        assert os.path.exists(tmp_path)
        import fastparquet
        # 获取实际分区列
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 1

    # 测试确保在fastparquet中使用partition_on选项支持分区，关注GitHub问题 #23283
    def test_partition_on_supported(self, tmp_path, fp, df_full):
        partition_cols = ["bool", "int"]
        df = df_full
        df.to_parquet(
            tmp_path,
            engine="fastparquet",
            compression=None,
            partition_on=partition_cols,
        )
        assert os.path.exists(tmp_path)
        import fastparquet
        # 获取实际分区列
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 2

    # 测试当同时使用partition_cols和partition_on时是否抛出错误，关注GitHub问题 #23283
    def test_error_on_using_partition_cols_and_partition_on(
        self, tmp_path, fp, df_full
    ):
        partition_cols = ["bool", "int"]
        df = df_full
        msg = (
            "Cannot use both partition_on and partition_cols. Use partition_cols for "
            "partitioning data"
        )
        # 断言抛出值错误异常，与期望消息匹配
        with pytest.raises(ValueError, match=msg):
            df.to_parquet(
                tmp_path,
                engine="fastparquet",
                compression=None,
                partition_on=partition_cols,
                partition_cols=partition_cols,
            )

    # 测试空DataFrame是否能正确进行往返操作，关注GitHub问题 #27339
    def test_empty_dataframe(self, fp):
        df = pd.DataFrame()
        expected = df.copy()
        check_round_trip(df, fp, expected=expected)

    # 标记为预期失败的测试，注释中提供了GitHub问题链接，关注fastparquet中的问题 #891
    @pytest.mark.xfail(
        reason="fastparquet passed mismatched values/dtype to DatetimeArray "
        "constructor, see https://github.com/dask/fastparquet/issues/891"
    )
    # 定义测试方法，用于测试时区感知索引的处理
    def test_timezone_aware_index(self, fp, timezone_aware_date_list):
        # 创建一个长度为5的列表，每个元素为时区感知日期列表的引用
        idx = 5 * [timezone_aware_date_list]

        # 使用索引为idx，数据为{"index_as_col": idx}创建DataFrame对象
        df = pd.DataFrame(index=idx, data={"index_as_col": idx})

        # 复制DataFrame对象作为预期结果
        expected = df.copy()
        # 设置预期DataFrame的索引名称为"index"
        expected.index.name = "index"
        # 调用check_round_trip函数验证DataFrame的往返操作
        check_round_trip(df, fp, expected=expected)

    # 定义测试方法，用于测试读取错误时关闭文件句柄的行为
    def test_close_file_handle_on_read_error(self):
        # 使用tm.ensure_clean("test.parquet")确保测试时的干净环境，并获取文件路径
        with tm.ensure_clean("test.parquet") as path:
            # 将"breakit"字节写入路径对应的文件
            pathlib.Path(path).write_bytes(b"breakit")
            # 使用tm.external_error_raised(Exception)检测是否引发异常
            with tm.external_error_raised(Exception):
                # 调用read_parquet函数尝试读取parquet文件
                read_parquet(path, engine="fastparquet")
            # 如果文件仍然打开，下一行代码在Windows上会引发错误
            pathlib.Path(path).unlink(missing_ok=False)

    # 定义测试方法，用于测试带有字节文件名的情况
    def test_bytes_file_name(self, engine):
        # 创建包含数据的DataFrame对象
        df = pd.DataFrame(data={"A": [0, 1], "B": [1, 0]})
        # 使用tm.ensure_clean("test.parquet")确保测试时的干净环境，并获取文件路径
        with tm.ensure_clean("test.parquet") as path:
            # 打开路径对应的文件并以二进制写入模式写入数据
            with open(path.encode(), "wb") as f:
                # 将DataFrame对象写入parquet格式
                df.to_parquet(f)

            # 使用指定引擎(engine参数)读取parquet文件并获取结果
            result = read_parquet(path, engine=engine)
        # 断言读取的结果与原始DataFrame相等
        tm.assert_frame_equal(result, df)

    # 定义测试方法，用于测试未实现的文件系统异常情况
    def test_filesystem_notimplemented(self):
        # 导入fastparquet模块，如果不存在则跳过此测试
        pytest.importorskip("fastparquet")
        # 创建包含数据的DataFrame对象
        df = pd.DataFrame(data={"A": [0, 1], "B": [1, 0]})
        # 使用tm.ensure_clean()确保测试时的干净环境，并获取文件路径
        with tm.ensure_clean() as path:
            # 使用pytest.raises捕获预期的NotImplementedError异常，并检查消息
            with pytest.raises(
                NotImplementedError, match="filesystem is not implemented"
            ):
                # 尝试将DataFrame对象写入parquet格式，并指定不支持的文件系统"foo"
                df.to_parquet(path, engine="fastparquet", filesystem="foo")

        # 使用tm.ensure_clean()确保测试时的干净环境，并获取文件路径
        with tm.ensure_clean() as path:
            # 向路径对应的文件写入字节数据"foo"
            pathlib.Path(path).write_bytes(b"foo")
            # 使用pytest.raises捕获预期的NotImplementedError异常，并检查消息
            with pytest.raises(
                NotImplementedError, match="filesystem is not implemented"
            ):
                # 尝试使用指定引擎和不支持的文件系统"foo"读取parquet文件
                read_parquet(path, engine="fastparquet", filesystem="foo")

    # 定义测试方法，用于测试无效的文件系统参数异常情况
    def test_invalid_filesystem(self):
        # 导入pyarrow模块，如果不存在则跳过此测试
        pytest.importorskip("pyarrow")
        # 创建包含数据的DataFrame对象
        df = pd.DataFrame(data={"A": [0, 1], "B": [1, 0]})
        # 使用tm.ensure_clean()确保测试时的干净环境，并获取文件路径
        with tm.ensure_clean() as path:
            # 使用pytest.raises捕获预期的ValueError异常，并检查消息
            with pytest.raises(
                ValueError, match="filesystem must be a pyarrow or fsspec FileSystem"
            ):
                # 尝试将DataFrame对象写入parquet格式，并指定无效的文件系统"foo"
                df.to_parquet(path, engine="pyarrow", filesystem="foo")

        # 使用tm.ensure_clean()确保测试时的干净环境，并获取文件路径
        with tm.ensure_clean() as path:
            # 向路径对应的文件写入字节数据"foo"
            pathlib.Path(path).write_bytes(b"foo")
            # 使用pytest.raises捕获预期的ValueError异常，并检查消息
            with pytest.raises(
                ValueError, match="filesystem must be a pyarrow or fsspec FileSystem"
            ):
                # 尝试使用指定引擎和无效的文件系统"foo"读取parquet文件
                read_parquet(path, engine="pyarrow", filesystem="foo")
    # 定义测试函数，用于测试不支持的 Parquet 文件系统存储选项
    def test_unsupported_pa_filesystem_storage_options(self):
        # 导入 pyarrow 文件系统模块，如果不存在则跳过测试
        pa_fs = pytest.importorskip("pyarrow.fs")
        # 创建一个包含数据的 DataFrame
        df = pd.DataFrame(data={"A": [0, 1], "B": [1, 0]})
        # 使用 tm.ensure_clean() 确保路径干净，使用 path 变量作为上下文管理器
        with tm.ensure_clean() as path:
            # 在上下文中，使用 pytest.raises 检查是否会抛出 NotImplementedError 异常，
            # 匹配的错误信息为 "storage_options not supported with a pyarrow FileSystem."
            with pytest.raises(
                NotImplementedError,
                match="storage_options not supported with a pyarrow FileSystem.",
            ):
                # 将 DataFrame 写入 Parquet 格式到指定路径，使用 pyarrow 引擎和本地文件系统，
                # 并设置存储选项为 {"foo": "bar"}，这会导致 NotImplementedError 异常
                df.to_parquet(
                    path,
                    engine="pyarrow",
                    filesystem=pa_fs.LocalFileSystem(),
                    storage_options={"foo": "bar"},
                )

        # 再次使用 tm.ensure_clean() 确保路径干净，使用 path 变量作为上下文管理器
        with tm.ensure_clean() as path:
            # 在上下文中，使用 pathlib.Path().write_bytes() 写入字节数据 "foo" 到指定路径
            pathlib.Path(path).write_bytes(b"foo")
            # 在上下文中，使用 pytest.raises 检查是否会抛出 NotImplementedError 异常，
            # 匹配的错误信息同样为 "storage_options not supported with a pyarrow FileSystem."
            with pytest.raises(
                NotImplementedError,
                match="storage_options not supported with a pyarrow FileSystem.",
            ):
                # 调用 read_parquet() 函数读取 Parquet 格式的数据，
                # 使用 pyarrow 引擎和本地文件系统，并设置存储选项为 {"foo": "bar"}，
                # 这同样会导致 NotImplementedError 异常
                read_parquet(
                    path,
                    engine="pyarrow",
                    filesystem=pa_fs.LocalFileSystem(),
                    storage_options={"foo": "bar"},
                )

    # 定义测试函数，用于测试无效的 dtype_backend 参数
    def test_invalid_dtype_backend(self, engine):
        # 错误消息字符串
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        # 创建一个包含数据的 DataFrame
        df = pd.DataFrame({"int": list(range(1, 4))})
        # 使用 tm.ensure_clean("tmp.parquet") 确保路径 "tmp.parquet" 干净，作为上下文管理器
        with tm.ensure_clean("tmp.parquet") as path:
            # 将 DataFrame 写入 Parquet 格式到指定路径
            df.to_parquet(path)
            # 在上下文中，使用 pytest.raises 检查是否会抛出 ValueError 异常，
            # 匹配的错误信息为预先定义的错误消息字符串
            with pytest.raises(ValueError, match=msg):
                # 调用 read_parquet() 函数读取 Parquet 格式的数据，
                # 指定 dtype_backend 参数为 "numpy"，这会导致 ValueError 异常
                read_parquet(path, dtype_backend="numpy")
```