# `D:\src\scipysrc\pandas\pandas\tests\io\test_orc.py`

```
"""test orc compat"""

# 导入必要的库和模块
import datetime  # 处理日期时间
from decimal import Decimal  # 处理十进制数据
from io import BytesIO  # 用于字节流操作
import os  # 提供操作系统相关功能
import pathlib  # 提供路径操作功能

import numpy as np  # 数组操作库
import pytest  # 测试框架

import pandas as pd  # 数据分析库
from pandas import read_orc  # 读取 ORC 文件功能
import pandas._testing as tm  # 测试辅助功能
from pandas.core.arrays import StringArray  # 字符串数组操作

pytest.importorskip("pyarrow.orc")  # 确保安装了 pyarrow 的 ORC 支持

import pyarrow as pa  # Apache Arrow 数据处理库

# 忽略特定警告以及相关过滤器设置
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def dirpath(datapath):
    """返回 ORC 文件所在目录的路径"""
    return datapath("io", "data", "orc")


def test_orc_reader_empty(dirpath):
    """测试读取空的 ORC 文件"""
    # 定义要读取的列和对应的数据类型
    columns = [
        "boolean1",
        "byte1",
        "short1",
        "int1",
        "long1",
        "float1",
        "double1",
        "bytes1",
        "string1",
    ]
    dtypes = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "object",
        "object",
    ]
    # 创建预期的空 DataFrame
    expected = pd.DataFrame(index=pd.RangeIndex(0))
    for colname, dtype in zip(columns, dtypes):
        expected[colname] = pd.Series(dtype=dtype)

    # 构造测试数据文件路径
    inputfile = os.path.join(dirpath, "TestOrcFile.emptyFile.orc")
    # 读取 ORC 文件并获取结果
    got = read_orc(inputfile, columns=columns)

    # 断言预期结果和实际结果相等
    tm.assert_equal(expected, got)


def test_orc_reader_basic(dirpath):
    """测试基本的 ORC 文件读取功能"""
    # 定义测试数据
    data = {
        "boolean1": np.array([False, True], dtype="bool"),
        "byte1": np.array([1, 100], dtype="int8"),
        "short1": np.array([1024, 2048], dtype="int16"),
        "int1": np.array([65536, 65536], dtype="int32"),
        "long1": np.array([9223372036854775807, 9223372036854775807], dtype="int64"),
        "float1": np.array([1.0, 2.0], dtype="float32"),
        "double1": np.array([-15.0, -5.0], dtype="float64"),
        "bytes1": np.array([b"\x00\x01\x02\x03\x04", b""], dtype="object"),
        "string1": np.array(["hi", "bye"], dtype="object"),
    }
    # 从字典构建预期的 DataFrame
    expected = pd.DataFrame.from_dict(data)

    # 构造测试数据文件路径
    inputfile = os.path.join(dirpath, "TestOrcFile.test1.orc")
    # 读取 ORC 文件并获取结果
    got = read_orc(inputfile, columns=data.keys())

    # 断言预期结果和实际结果相等
    tm.assert_equal(expected, got)


def test_orc_reader_decimal(dirpath):
    """测试读取包含十进制数据的 ORC 文件"""
    # 只测试前 10 行数据
    data = {
        "_col0": np.array(
            [
                Decimal("-1000.50000"),
                Decimal("-999.60000"),
                Decimal("-998.70000"),
                Decimal("-997.80000"),
                Decimal("-996.90000"),
                Decimal("-995.10000"),
                Decimal("-994.11000"),
                Decimal("-993.12000"),
                Decimal("-992.13000"),
                Decimal("-991.14000"),
            ],
            dtype="object",
        )
    }
    # 从字典构建预期的 DataFrame
    expected = pd.DataFrame.from_dict(data)

    # 构造测试数据文件路径
    inputfile = os.path.join(dirpath, "TestOrcFile.decimal.orc")
    # 读取 ORC 文件的前 10 行数据并获取结果
    got = read_orc(inputfile).iloc[:10]

    # 断言预期结果和实际结果相等
    tm.assert_equal(expected, got)


def test_orc_reader_date_low(dirpath):
    """测试读取包含日期数据的 ORC 文件"""
    # 创建一个包含时间和日期数据的字典
    data = {
        "time": np.array(
            [
                "1900-05-05 12:34:56.100000",
                "1900-05-05 12:34:56.100100",
                "1900-05-05 12:34:56.100200",
                "1900-05-05 12:34:56.100300",
                "1900-05-05 12:34:56.100400",
                "1900-05-05 12:34:56.100500",
                "1900-05-05 12:34:56.100600",
                "1900-05-05 12:34:56.100700",
                "1900-05-05 12:34:56.100800",
                "1900-05-05 12:34:56.100900",
            ],
            dtype="datetime64[ns]",
        ),
        "date": np.array(
            [
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
            ],
            dtype="object",
        ),
    }
    
    # 从字典数据创建一个期望的 Pandas DataFrame
    expected = pd.DataFrame.from_dict(data)
    
    # 构建输入文件的完整路径
    inputfile = os.path.join(dirpath, "TestOrcFile.testDate1900.orc")
    
    # 调用 read_orc 函数读取 ORC 文件数据，获取前十行
    got = read_orc(inputfile).iloc[:10]
    
    # 使用测试工具库中的方法比较期望的 DataFrame 和读取到的 DataFrame
    tm.assert_equal(expected, got)
# 定义测试函数，用于测试读取 ORC 文件中高时间戳日期的情况
def test_orc_reader_date_high(dirpath):
    # 准备测试数据，包括时间戳和日期数组
    data = {
        "time": np.array(
            [
                "2038-05-05 12:34:56.100000",
                "2038-05-05 12:34:56.100100",
                "2038-05-05 12:34:56.100200",
                "2038-05-05 12:34:56.100300",
                "2038-05-05 12:34:56.100400",
                "2038-05-05 12:34:56.100500",
                "2038-05-05 12:34:56.100600",
                "2038-05-05 12:34:56.100700",
                "2038-05-05 12:34:56.100800",
                "2038-05-05 12:34:56.100900",
            ],
            dtype="datetime64[ns]",
        ),
        "date": np.array(
            [
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
            ],
            dtype="object",
        ),
    }
    # 从数据字典创建期望的 DataFrame
    expected = pd.DataFrame.from_dict(data)

    # 构造输入文件路径
    inputfile = os.path.join(dirpath, "TestOrcFile.testDate2038.orc")
    # 读取 ORC 文件并取前10行数据
    got = read_orc(inputfile).iloc[:10]

    # 使用 pytest 模块的断言函数检查期望和实际结果是否相等
    tm.assert_equal(expected, got)


# 定义测试函数，用于测试读取 ORC 文件中 Snappy 压缩数据的情况
def test_orc_reader_snappy_compressed(dirpath):
    # 准备测试数据，包括整数和字符串数组
    data = {
        "int1": np.array(
            [
                -1160101563,
                1181413113,
                2065821249,
                -267157795,
                172111193,
                1752363137,
                1406072123,
                1911809390,
                -1308542224,
                -467100286,
            ],
            dtype="int32",
        ),
        "string1": np.array(
            [
                "f50dcb8",
                "382fdaaa",
                "90758c6",
                "9e8caf3f",
                "ee97332b",
                "d634da1",
                "2bea4396",
                "d67d89e8",
                "ad71007e",
                "e8c82066",
            ],
            dtype="object",
        ),
    }
    # 从数据字典创建期望的 DataFrame
    expected = pd.DataFrame.from_dict(data)

    # 构造输入文件路径
    inputfile = os.path.join(dirpath, "TestOrcFile.testSnappy.orc")
    # 读取 ORC 文件并取前10行数据
    got = read_orc(inputfile).iloc[:10]

    # 使用 pytest 模块的断言函数检查期望和实际结果是否相等
    tm.assert_equal(expected, got)


# 定义测试函数，用于测试 ORC 文件的完整读写往返过程
def test_orc_roundtrip_file(dirpath):
    # 注释：GH44554
    # 如果缺少 PyArrow 模块，则跳过当前测试
    pytest.importorskip("pyarrow")
    # 创建包含不同数据类型的字典
    data = {
        "boolean1": np.array([False, True], dtype="bool"),        # 包含布尔类型的数组
        "byte1": np.array([1, 100], dtype="int8"),                # 包含字节类型的数组
        "short1": np.array([1024, 2048], dtype="int16"),          # 包含短整型数组
        "int1": np.array([65536, 65536], dtype="int32"),          # 包含整型数组
        "long1": np.array([9223372036854775807, 9223372036854775807], dtype="int64"),  # 包含长整型数组
        "float1": np.array([1.0, 2.0], dtype="float32"),          # 包含单精度浮点数数组
        "double1": np.array([-15.0, -5.0], dtype="float64"),      # 包含双精度浮点数数组
        "bytes1": np.array([b"\x00\x01\x02\x03\x04", b""], dtype="object"),  # 包含字节串数组
        "string1": np.array(["hi", "bye"], dtype="object"),       # 包含字符串数组
    }
    
    # 根据数据字典创建预期的 Pandas DataFrame
    expected = pd.DataFrame.from_dict(data)

    # 使用临时路径确保环境清洁
    with tm.ensure_clean() as path:
        # 将预期的 DataFrame 写入 ORC 格式文件
        expected.to_orc(path)
        # 从 ORC 文件中读取数据
        got = read_orc(path)

        # 断言预期结果和读取结果是否相等
        tm.assert_equal(expected, got)
def test_orc_roundtrip_bytesio():
    # GH44554
    # 确保PyArrow支持当前参数顺序的ORC写入
    pytest.importorskip("pyarrow")

    # 准备测试数据
    data = {
        "boolean1": np.array([False, True], dtype="bool"),
        "byte1": np.array([1, 100], dtype="int8"),
        "short1": np.array([1024, 2048], dtype="int16"),
        "int1": np.array([65536, 65536], dtype="int32"),
        "long1": np.array([9223372036854775807, 9223372036854775807], dtype="int64"),
        "float1": np.array([1.0, 2.0], dtype="float32"),
        "double1": np.array([-15.0, -5.0], dtype="float64"),
        "bytes1": np.array([b"\x00\x01\x02\x03\x04", b""], dtype="object"),
        "string1": np.array(["hi", "bye"], dtype="object"),
    }
    expected = pd.DataFrame.from_dict(data)

    # 将期望的DataFrame转换为ORC格式的字节流
    bytes = expected.to_orc()

    # 读取ORC格式的字节流，返回读取到的DataFrame
    got = read_orc(BytesIO(bytes))

    # 使用测试工具比较预期结果和实际结果
    tm.assert_equal(expected, got)


@pytest.mark.parametrize(
    "orc_writer_dtypes_not_supported",
    [
        np.array([1, 20], dtype="uint64"),
        pd.Series(["a", "b", "a"], dtype="category"),
        [pd.Interval(left=0, right=2), pd.Interval(left=0, right=5)],
        [pd.Period("2022-01-03", freq="D"), pd.Period("2022-01-04", freq="D")],
    ],
)
def test_orc_writer_dtypes_not_supported(orc_writer_dtypes_not_supported):
    # GH44554
    # 确保PyArrow支持当前参数顺序的ORC写入
    pytest.importorskip("pyarrow")

    # 创建包含不支持的数据类型的DataFrame
    df = pd.DataFrame({"unimpl": orc_writer_dtypes_not_supported})
    msg = "The dtype of one or more columns is not supported yet."

    # 断言期望的异常被抛出，匹配预期的消息
    with pytest.raises(NotImplementedError, match=msg):
        df.to_orc()


def test_orc_dtype_backend_pyarrow():
    # 确保PyArrow库可用
    pytest.importorskip("pyarrow")

    # 创建包含各种数据类型的DataFrame
    df = pd.DataFrame(
        {
            "string": list("abc"),
            "string_with_nan": ["a", np.nan, "c"],
            "string_with_none": ["a", None, "c"],
            "bytes": [b"foo", b"bar", None],
            "int": list(range(1, 4)),
            "float": np.arange(4.0, 7.0, dtype="float64"),
            "float_with_nan": [2.0, np.nan, 3.0],
            "bool": [True, False, True],
            "bool_with_na": [True, False, None],
            "datetime": pd.date_range("20130101", periods=3),
            "datetime_with_nat": [
                pd.Timestamp("20130101"),
                pd.NaT,
                pd.Timestamp("20130103"),
            ],
        }
    )

    # FIXME: 如果不转换为纳秒精度，无法正确地往返处理
    df["datetime_with_nat"] = df["datetime_with_nat"].astype("M8[ns]")

    # 将DataFrame转换为ORC格式的字节流
    bytes_data = df.copy().to_orc()

    # 读取ORC格式的字节流，使用PyArrow作为dtype后端
    result = read_orc(BytesIO(bytes_data), dtype_backend="pyarrow")

    # 创建期望的DataFrame，使用ArrowExtensionArray来存储数据
    expected = pd.DataFrame(
        {
            col: pd.arrays.ArrowExtensionArray(pa.array(df[col], from_pandas=True))
            for col in df.columns
        }
    )

    # 使用测试工具比较预期结果和实际结果
    tm.assert_frame_equal(result, expected)


def test_orc_dtype_backend_numpy_nullable():
    # GH#50503
    # 确保PyArrow库可用
    pytest.importorskip("pyarrow")
    # 创建一个 Pandas DataFrame，包含多列数据
    df = pd.DataFrame(
        {
            "string": list("abc"),  # 字符串列，包含 'a', 'b', 'c'
            "string_with_nan": ["a", np.nan, "c"],  # 字符串列，包含 'a', NaN, 'c'
            "string_with_none": ["a", None, "c"],  # 字符串列，包含 'a', None, 'c'
            "int": list(range(1, 4)),  # 整数列，包含 1, 2, 3
            "int_with_nan": pd.Series([1, pd.NA, 3], dtype="Int64"),  # 整数列，包含 1, NaN, 3
            "na_only": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Int64"),  # 整数列，全为 NaN
            "float": np.arange(4.0, 7.0, dtype="float64"),  # 浮点数列，包含 4.0, 5.0, 6.0
            "float_with_nan": [2.0, np.nan, 3.0],  # 浮点数列，包含 2.0, NaN, 3.0
            "bool": [True, False, True],  # 布尔列，包含 True, False, True
            "bool_with_na": [True, False, None],  # 布尔列，包含 True, False, None
        }
    )
    
    # 复制 DataFrame，并将其转换为 ORC 格式的字节数据
    bytes_data = df.copy().to_orc()
    
    # 调用 read_orc 函数，读取 ORC 格式的字节数据，使用 numpy nullable 类型的后端处理
    result = read_orc(BytesIO(bytes_data), dtype_backend="numpy_nullable")
    
    # 创建期望的 Pandas DataFrame，包含与 result 数据相匹配的期望值
    expected = pd.DataFrame(
        {
            "string": StringArray(np.array(["a", "b", "c"], dtype=np.object_)),  # 字符串列，使用 StringArray 存储
            "string_with_nan": StringArray(
                np.array(["a", pd.NA, "c"], dtype=np.object_)
            ),  # 字符串列，包含 'a', NaN, 'c'，使用 StringArray 存储
            "string_with_none": StringArray(
                np.array(["a", pd.NA, "c"], dtype=np.object_)
            ),  # 字符串列，包含 'a', NaN, 'c'，使用 StringArray 存储
            "int": pd.Series([1, 2, 3], dtype="Int64"),  # 整数列，包含 1, 2, 3
            "int_with_nan": pd.Series([1, pd.NA, 3], dtype="Int64"),  # 整数列，包含 1, NaN, 3
            "na_only": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Int64"),  # 整数列，全为 NaN
            "float": pd.Series([4.0, 5.0, 6.0], dtype="Float64"),  # 浮点数列，包含 4.0, 5.0, 6.0
            "float_with_nan": pd.Series([2.0, pd.NA, 3.0], dtype="Float64"),  # 浮点数列，包含 2.0, NaN, 3.0
            "bool": pd.Series([True, False, True], dtype="boolean"),  # 布尔列，包含 True, False, True
            "bool_with_na": pd.Series([True, False, pd.NA], dtype="boolean"),  # 布尔列，包含 True, False, NaN
        }
    )
    
    # 使用 assert_frame_equal 检查 result 是否与期望的 expected 相等
    tm.assert_frame_equal(result, expected)
# 定义测试函数，测试读取 ORC 文件时传入 URI 路径的情况
def test_orc_uri_path():
    # 创建预期的 DataFrame，包含一列整数，范围为 1 到 3
    expected = pd.DataFrame({"int": list(range(1, 4))})
    # 使用 tm.ensure_clean 确保在临时路径上操作，返回路径名
    with tm.ensure_clean("tmp.orc") as path:
        # 将预期的 DataFrame 写入 ORC 文件
        expected.to_orc(path)
        # 将路径转换为 URI 格式
        uri = pathlib.Path(path).as_uri()
        # 调用 read_orc 函数读取 URI 对应的 ORC 文件内容
        result = read_orc(uri)
    # 使用 tm.assert_frame_equal 检查读取结果是否与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 定义参数化测试，测试不同类型的索引
@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(start=2, stop=5, step=1),
        pd.RangeIndex(start=0, stop=3, step=1, name="non-default"),
        pd.Index([1, 2, 3]),
    ],
)
# 定义测试函数，测试将 DataFrame 写入 ORC 文件时使用非默认索引的情况
def test_to_orc_non_default_index(index):
    # 创建包含一列整数的 DataFrame，使用不同类型的索引
    df = pd.DataFrame({"a": [1, 2, 3]}, index=index)
    # 定义错误消息，用于捕获预期的 ValueError 异常
    msg = (
        "orc does not support serializing a non-default index|"
        "orc does not serialize index meta-data"
    )
    # 使用 pytest.raises 检查是否抛出预期的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        # 调用 DataFrame 的 to_orc 方法写入 ORC 文件
        df.to_orc()


# 定义测试函数，测试传入无效 dtype_backend 参数时是否抛出 ValueError 异常
def test_invalid_dtype_backend():
    # 定义错误消息，指出无效的 dtype_backend 参数
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    # 创建包含一列整数的 DataFrame
    df = pd.DataFrame({"int": list(range(1, 4))})
    # 使用 tm.ensure_clean 确保在临时路径上操作，返回路径名
    with tm.ensure_clean("tmp.orc") as path:
        # 将 DataFrame 写入 ORC 文件
        df.to_orc(path)
        # 使用 pytest.raises 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            # 调用 read_orc 函数读取 ORC 文件，并传入无效的 dtype_backend 参数
            read_orc(path, dtype_backend="numpy")


# 定义测试函数，测试读取 ORC 文件时推断字符串类型的情况
def test_string_inference(tmp_path):
    # 创建临时路径下的文件名
    # GH#54431
    path = tmp_path / "test_string_inference.p"
    # 创建包含一列字符串的 DataFrame
    df = pd.DataFrame(data={"a": ["x", "y"]})
    # 将 DataFrame 写入 ORC 文件
    df.to_orc(path)
    # 使用 pd.option_context 设置 future.infer_string 为 True，读取 ORC 文件
    with pd.option_context("future.infer_string", True):
        # 调用 read_orc 函数读取 ORC 文件内容
        result = read_orc(path)
    # 创建预期的 DataFrame，指定列为字符串类型，使用 pyarrow_numpy 作为 dtype
    expected = pd.DataFrame(
        data={"a": ["x", "y"]},
        dtype="string[pyarrow_numpy]",
        columns=pd.Index(["a"], dtype="string[pyarrow_numpy]"),
    )
    # 使用 tm.assert_frame_equal 检查读取结果是否与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)
```