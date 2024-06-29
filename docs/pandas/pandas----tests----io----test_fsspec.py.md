# `D:\src\scipysrc\pandas\pandas\tests\io\test_fsspec.py`

```
import io  # 导入 io 模块，用于处理文件流

import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入多个模块和函数
    DataFrame,  # DataFrame 数据结构，用于处理二维表格数据
    date_range,  # 生成日期范围
    read_csv,  # 读取 CSV 文件到 DataFrame
    read_excel,  # 读取 Excel 文件到 DataFrame
    read_feather,  # 读取 Feather 格式文件到 DataFrame
    read_json,  # 读取 JSON 文件到 DataFrame
    read_parquet,  # 读取 Parquet 文件到 DataFrame
    read_pickle,  # 读取 Pickle 文件到 DataFrame
    read_stata,  # 读取 Stata 文件到 DataFrame
    read_table,  # 读取表格型文件到 DataFrame
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块
from pandas.util import _test_decorators as td  # 导入 pandas 内部测试修饰符模块

pytestmark = pytest.mark.filterwarnings(  # 设置 pytest 的标记，忽略特定的警告信息
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def fsspectest():  # 定义 pytest 的 fixture 函数，用于测试 fsspec 相关功能
    pytest.importorskip("fsspec")  # 如果没有 fsspec 库则跳过测试
    from fsspec import register_implementation  # 导入注册实现函数
    from fsspec.implementations.memory import MemoryFileSystem  # 导入内存文件系统
    from fsspec.registry import _registry as registry  # 导入注册表

    class TestMemoryFS(MemoryFileSystem):  # 定义一个测试用的内存文件系统类
        protocol = "testmem"  # 协议名称
        test = [None]  # 测试数据

        def __init__(self, **kwargs) -> None:
            self.test[0] = kwargs.pop("test", None)  # 初始化测试数据
            super().__init__(**kwargs)

    register_implementation("testmem", TestMemoryFS, clobber=True)  # 注册测试内存文件系统
    yield TestMemoryFS()  # 返回测试内存文件系统对象
    registry.pop("testmem", None)  # 清除注册的测试内存文件系统
    TestMemoryFS.test[0] = None  # 重置测试数据
    TestMemoryFS.store.clear()  # 清空存储


@pytest.fixture
def df1():  # 定义 pytest 的 fixture 函数，返回一个 DataFrame 示例
    return DataFrame(  # 创建 DataFrame 对象
        {
            "int": [1, 3],  # 整数列
            "float": [2.0, np.nan],  # 浮点数列
            "str": ["t", "s"],  # 字符串列
            "dt": date_range("2018-06-18", periods=2),  # 日期时间列
        }
    )


@pytest.fixture
def cleared_fs():  # 定义 pytest 的 fixture 函数，用于清除文件系统数据
    fsspec = pytest.importorskip("fsspec")  # 导入 fsspec 库，如果不存在则跳过测试

    memfs = fsspec.filesystem("memory")  # 创建内存文件系统对象
    yield memfs  # 返回内存文件系统对象
    memfs.store.clear()  # 清空内存文件系统存储数据


def test_read_csv(cleared_fs, df1):  # 定义测试函数，测试读取 CSV 文件功能
    text = str(df1.to_csv(index=False)).encode()  # 将 DataFrame 转换为 CSV 文本，并转换为字节流
    with cleared_fs.open("test/test.csv", "wb") as w:  # 在清空的内存文件系统中写入 CSV 文件
        w.write(text)  # 写入 CSV 文本数据
    df2 = read_csv("memory://test/test.csv", parse_dates=["dt"])  # 从内存文件系统中读取 CSV 文件到 DataFrame

    expected = df1.copy()  # 复制原始 DataFrame 作为预期结果
    expected["dt"] = expected["dt"].astype("M8[s]")  # 将日期时间列转换为 datetime64[s] 类型
    tm.assert_frame_equal(df2, expected)  # 断言读取的 DataFrame 与预期结果相同


def test_reasonable_error(monkeypatch, cleared_fs):  # 定义测试函数，测试合理的错误处理
    from fsspec.registry import known_implementations  # 导入已知实现的注册表

    with pytest.raises(ValueError, match="nosuchprotocol"):  # 断言引发 ValueError 异常，匹配错误信息
        read_csv("nosuchprotocol://test/test.csv")  # 尝试读取不存在的协议的 CSV 文件
    err_msg = "test error message"  # 错误消息字符串
    monkeypatch.setitem(  # 使用 monkeypatch 修改已知实现的注册表
        known_implementations,
        "couldexist",
        {"class": "unimportable.CouldExist", "err": err_msg},  # 添加新的实现信息
    )
    with pytest.raises(ImportError, match=err_msg):  # 断言引发 ImportError 异常，匹配错误信息
        read_csv("couldexist://test/test.csv")  # 尝试读取存在的协议但无法导入的 CSV 文件


def test_to_csv(cleared_fs, df1):  # 定义测试函数，测试将 DataFrame 写入 CSV 文件
    df1.to_csv("memory://test/test.csv", index=True)  # 将 DataFrame 写入内存文件系统中的 CSV 文件

    df2 = read_csv("memory://test/test.csv", parse_dates=["dt"], index_col=0)  # 从内存文件系统中读取 CSV 文件到 DataFrame

    expected = df1.copy()  # 复制原始 DataFrame 作为预期结果
    expected["dt"] = expected["dt"].astype("M8[s]")  # 将日期时间列转换为 datetime64[s] 类型
    tm.assert_frame_equal(df2, expected)  # 断言读取的 DataFrame 与预期结果相同


def test_to_excel(cleared_fs, df1):  # 定义测试函数，测试将 DataFrame 写入 Excel 文件
    pytest.importorskip("openpyxl")  # 导入 openpyxl 库，如果不存在则跳过测试
    ext = "xlsx"  # 文件扩展名
    path = f"memory://test/test.{ext}"  # 文件路径
    df1.to_excel(path, index=True)  # 将 DataFrame 写入内存文件系统中的 Excel 文件

    df2 = read_excel(path, parse_dates=["dt"], index_col=0)  # 从内存文件系统中读取 Excel 文件到 DataFrame

    expected = df1.copy()  # 复制原始 DataFrame 作为预期结果
    expected["dt"] = expected["dt"].astype("M8[s]")  # 将日期时间列转换为 datetime64[s] 类型
    tm.assert_frame_equal(df2, expected)  # 断言读取的 DataFrame 与预期结果相同


@pytest.mark.parametrize("binary_mode", [False, True])  # 使用 pytest 的参数化标记，测试二进制模式与非二进制模式
# 定义测试函数，用于测试将 DataFrame 写入和从文件系统读取的操作
def test_to_csv_fsspec_object(cleared_fs, binary_mode, df1):
    # 导入 pytest 库并检查是否可用
    fsspec = pytest.importorskip("fsspec")

    # 设置文件路径
    path = "memory://test/test.csv"
    # 根据二进制模式确定打开文件的模式（二进制或文本）
    mode = "wb" if binary_mode else "w"
    
    # 使用 fsspec 打开指定路径的文件对象，并确保文件未关闭
    with fsspec.open(path, mode=mode).open() as fsspec_object:
        # 将 DataFrame df1 写入到 fsspec_object 中作为 CSV 文件，包括索引
        df1.to_csv(fsspec_object, index=True)
        # 断言文件对象没有关闭
        assert not fsspec_object.closed

    # 切换模式为读取模式
    mode = mode.replace("w", "r")
    # 再次使用 fsspec 打开文件对象，用于读取之前写入的 CSV 数据
    with fsspec.open(path, mode=mode) as fsspec_object:
        # 从 fsspec_object 中读取 CSV 数据，指定解析日期列和索引列
        df2 = read_csv(
            fsspec_object,
            parse_dates=["dt"],
            index_col=0,
        )
        # 断言文件对象没有关闭
        assert not fsspec_object.closed

    # 创建期望的 DataFrame 结果，将日期列转换为时间戳秒级精度
    expected = df1.copy()
    expected["dt"] = expected["dt"].astype("M8[s]")
    # 断言读取的 DataFrame df2 与期望的结果 DataFrame 相等
    tm.assert_frame_equal(df2, expected)


# 定义测试函数，用于测试 CSV 写入和读取的选项设置
def test_csv_options(fsspectest):
    # 创建包含单个元素的 DataFrame 对象 df
    df = DataFrame({"a": [0]})
    # 将 df 写入到指定路径的 CSV 文件，设置存储选项为 "csv_write"，不包括索引
    df.to_csv(
        "testmem://test/test.csv", storage_options={"test": "csv_write"}, index=False
    )
    # 断言存储选项包含 "csv_write"
    assert fsspectest.test[0] == "csv_write"
    # 读取指定路径的 CSV 文件，设置存储选项为 "csv_read"
    read_csv("testmem://test/test.csv", storage_options={"test": "csv_read"})
    # 断言存储选项包含 "csv_read"
    assert fsspectest.test[0] == "csv_read"


# 定义测试函数，用于测试从表格文件中读取的选项设置
def test_read_table_options(fsspectest):
    # 创建包含单个元素的 DataFrame 对象 df
    df = DataFrame({"a": [0]})
    # 将 df 写入到指定路径的 CSV 文件，设置存储选项为 "csv_write"，不包括索引
    df.to_csv(
        "testmem://test/test.csv", storage_options={"test": "csv_write"}, index=False
    )
    # 断言存储选项包含 "csv_write"
    assert fsspectest.test[0] == "csv_write"
    # 读取指定路径的表格文件，设置存储选项为 "csv_read"
    read_table("testmem://test/test.csv", storage_options={"test": "csv_read"})
    # 断言存储选项包含 "csv_read"
    assert fsspectest.test[0] == "csv_read"


# 定义测试函数，用于测试 Excel 文件写入和读取的选项设置
def test_excel_options(fsspectest):
    # 导入 openpyxl 库并检查是否可用
    pytest.importorskip("openpyxl")
    # 设置文件扩展名为 "xlsx"
    extension = "xlsx"

    # 创建包含单个元素的 DataFrame 对象 df
    df = DataFrame({"a": [0]})

    # 设置文件路径
    path = f"testmem://test/test.{extension}"

    # 将 df 写入到指定路径的 Excel 文件，设置存储选项为 "write"，不包括索引
    df.to_excel(path, storage_options={"test": "write"}, index=False)
    # 断言存储选项包含 "write"
    assert fsspectest.test[0] == "write"
    # 从指定路径读取 Excel 文件，设置存储选项为 "read"
    read_excel(path, storage_options={"test": "read"})
    # 断言存储选项包含 "read"
    assert fsspectest.test[0] == "read"


# 定义测试函数，用于测试将 DataFrame 写入 Parquet 文件的选项设置
def test_to_parquet_new_file(cleared_fs, df1):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    # 导入 fastparquet 库并检查是否可用
    pytest.importorskip("fastparquet")

    # 将 DataFrame df1 写入到指定路径的 Parquet 文件，包括索引，使用 fastparquet 引擎，不压缩
    df1.to_parquet(
        "memory://test/test.csv", index=True, engine="fastparquet", compression=None
    )


# 定义测试函数，用于测试 Arrow Parquet 文件写入和读取的选项设置
def test_arrowparquet_options(fsspectest):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    # 导入 pyarrow 库并检查是否可用
    pytest.importorskip("pyarrow")
    # 创建包含单个元素的 DataFrame 对象 df
    df = DataFrame({"a": [0]})
    # 将 df 写入到指定路径的 Arrow Parquet 文件，使用 pyarrow 引擎，不压缩，存储选项设置为 "parquet_write"
    df.to_parquet(
        "testmem://test/test.csv",
        engine="pyarrow",
        compression=None,
        storage_options={"test": "parquet_write"},
    )
    # 断言存储选项包含 "parquet_write"
    assert fsspectest.test[0] == "parquet_write"
    # 从指定路径读取 Arrow Parquet 文件，使用 pyarrow 引擎，存储选项设置为 "parquet_read"
    read_parquet(
        "testmem://test/test.csv",
        engine="pyarrow",
        storage_options={"test": "parquet_read"},
    )
    # 断言存储选项包含 "parquet_read"
    assert fsspectest.test[0] == "parquet_read"


# 定义测试函数，用于测试 FastParquet 文件写入的选项设置
def test_fastparquet_options(fsspectest):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    # 导入 fastparquet 库并检查是否可用
    pytest.importorskip("fastparquet")

    # 创建包含单个元素的 DataFrame 对象 df
    df = DataFrame({"a": [0]})
    # 略
    # 将数据框 df 写入 Parquet 格式到内存中的 testmem://test/test.csv 文件
    # 使用 fastparquet 引擎进行写入，不进行压缩
    # 设置存储选项，指定测试参数 "test" 为 "parquet_write"
    df.to_parquet(
        "testmem://test/test.csv",
        engine="fastparquet",
        compression=None,
        storage_options={"test": "parquet_write"},
    )
    
    # 断言，验证 fsspectest.test 列表中的第一个元素是否为 "parquet_write"
    assert fsspectest.test[0] == "parquet_write"
    
    # 从内存中的 testmem://test/test.csv 文件读取 Parquet 格式的数据
    # 使用 fastparquet 引擎进行读取
    # 设置存储选项，指定测试参数 "test" 为 "parquet_read"
    read_parquet(
        "testmem://test/test.csv",
        engine="fastparquet",
        storage_options={"test": "parquet_read"},
    )
    
    # 断言，验证 fsspectest.test 列表中的第一个元素是否为 "parquet_read"
    assert fsspectest.test[0] == "parquet_read"
@pytest.mark.single_cpu
# 标记测试函数为单CPU运行的测试用例
def test_from_s3_csv(s3_public_bucket_with_data, tips_file, s3so):
    pytest.importorskip("s3fs")
    # 如果没有安装s3fs库，则跳过这个测试用例

    tm.assert_equal(
        read_csv(
            f"s3://{s3_public_bucket_with_data.name}/tips.csv", storage_options=s3so
        ),
        read_csv(tips_file),
    )
    # 比较从S3读取的CSV文件和本地文件的内容是否相等

    # the following are decompressed by pandas, not fsspec
    tm.assert_equal(
        read_csv(
            f"s3://{s3_public_bucket_with_data.name}/tips.csv.gz", storage_options=s3so
        ),
        read_csv(tips_file),
    )
    # 比较从S3读取的Gzip压缩的CSV文件和本地文件的内容是否相等

    tm.assert_equal(
        read_csv(
            f"s3://{s3_public_bucket_with_data.name}/tips.csv.bz2", storage_options=s3so
        ),
        read_csv(tips_file),
    )
    # 比较从S3读取的Bzip2压缩的CSV文件和本地文件的内容是否相等


@pytest.mark.single_cpu
@pytest.mark.parametrize("protocol", ["s3", "s3a", "s3n"])
# 标记测试函数为单CPU运行的测试用例，并使用参数化测试三种S3协议
def test_s3_protocols(s3_public_bucket_with_data, tips_file, protocol, s3so):
    pytest.importorskip("s3fs")
    # 如果没有安装s3fs库，则跳过这个测试用例

    tm.assert_equal(
        read_csv(
            f"{protocol}://{s3_public_bucket_with_data.name}/tips.csv",
            storage_options=s3so,
        ),
        read_csv(tips_file),
    )
    # 比较从指定S3协议读取的CSV文件和本地文件的内容是否相等


@pytest.mark.single_cpu
# 标记测试函数为单CPU运行的测试用例
def test_s3_parquet(s3_public_bucket, s3so, df1):
    pytest.importorskip("fastparquet")
    # 如果没有安装fastparquet库，则跳过这个测试用例
    pytest.importorskip("s3fs")
    # 如果没有安装s3fs库，则跳过这个测试用例

    fn = f"s3://{s3_public_bucket.name}/test.parquet"
    df1.to_parquet(
        fn, index=False, engine="fastparquet", compression=None, storage_options=s3so
    )
    # 将DataFrame df1以Parquet格式存储到指定S3路径下

    df2 = read_parquet(fn, engine="fastparquet", storage_options=s3so)
    # 从指定S3路径读取Parquet格式的数据到DataFrame df2

    tm.assert_equal(df1, df2)
    # 比较两个DataFrame df1和df2是否相等


@td.skip_if_installed("fsspec")
# 如果安装了fsspec库，则跳过这个测试用例
def test_not_present_exception():
    msg = "Missing optional dependency 'fsspec'|fsspec library is required"
    # 定义异常消息字符串
    with pytest.raises(ImportError, match=msg):
        read_csv("memory://test/test.csv")
    # 测试读取不存在的文件时是否触发指定异常


def test_feather_options(fsspectest):
    pytest.importorskip("pyarrow")
    # 如果没有安装pyarrow库，则跳过这个测试用例

    df = DataFrame({"a": [0]})
    df.to_feather("testmem://mockfile", storage_options={"test": "feather_write"})
    # 将DataFrame df以Feather格式存储到指定内存路径下

    assert fsspectest.test[0] == "feather_write"
    # 断言fsspectest对象的test属性值是否为"feather_write"

    out = read_feather("testmem://mockfile", storage_options={"test": "feather_read"})
    # 从指定内存路径读取Feather格式的数据到DataFrame out

    assert fsspectest.test[0] == "feather_read"
    # 断言fsspectest对象的test属性值是否为"feather_read"

    tm.assert_frame_equal(df, out)
    # 比较两个DataFrame df和out是否相等


def test_pickle_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_pickle("testmem://mockfile", storage_options={"test": "pickle_write"})
    # 将DataFrame df以Pickle格式存储到指定内存路径下

    assert fsspectest.test[0] == "pickle_write"
    # 断言fsspectest对象的test属性值是否为"pickle_write"

    out = read_pickle("testmem://mockfile", storage_options={"test": "pickle_read"})
    # 从指定内存路径读取Pickle格式的数据到DataFrame out

    assert fsspectest.test[0] == "pickle_read"
    # 断言fsspectest对象的test属性值是否为"pickle_read"

    tm.assert_frame_equal(df, out)
    # 比较两个DataFrame df和out是否相等


def test_json_options(fsspectest, compression):
    df = DataFrame({"a": [0]})
    df.to_json(
        "testmem://mockfile",
        compression=compression,
        storage_options={"test": "json_write"},
    )
    # 将DataFrame df以JSON格式存储到指定内存路径下，带有压缩选项

    assert fsspectest.test[0] == "json_write"
    # 断言fsspectest对象的test属性值是否为"json_write"

    out = read_json(
        "testmem://mockfile",
        compression=compression,
        storage_options={"test": "json_read"},
    )
    # 从指定内存路径读取JSON格式的数据到DataFrame out，带有压缩选项

    assert fsspectest.test[0] == "json_read"
    # 断言fsspectest对象的test属性值是否为"json_read"

    tm.assert_frame_equal(df, out)
    # 比较两个DataFrame df和out是否相等
# 测试 Stata 存储选项的功能
def test_stata_options(fsspectest):
    # 创建一个包含单列 'a' 的 DataFrame 对象
    df = DataFrame({"a": [0]})
    # 将 DataFrame 写入到指定的 testmem://mockfile 地址，使用 Stata 格式存储，不写入索引信息
    df.to_stata(
        "testmem://mockfile", storage_options={"test": "stata_write"}, write_index=False
    )
    # 断言 fsspectest.test 列表的第一个元素是否为 "stata_write"
    assert fsspectest.test[0] == "stata_write"
    # 从 testmem://mockfile 地址读取数据，使用 Stata 格式读取
    out = read_stata("testmem://mockfile", storage_options={"test": "stata_read"})
    # 断言 fsspectest.test 列表的第一个元素是否为 "stata_read"
    assert fsspectest.test[0] == "stata_read"
    # 断言读取的 DataFrame 是否与原始 df 数据相等，且数据类型为 int64
    tm.assert_frame_equal(df, out.astype("int64"))


# 测试 Markdown 存储选项的功能
def test_markdown_options(fsspectest):
    # 确保 tabulate 库已导入，否则跳过该测试
    pytest.importorskip("tabulate")
    # 创建一个包含单列 'a' 的 DataFrame 对象
    df = DataFrame({"a": [0]})
    # 将 DataFrame 写入到指定的 testmem://mockfile 地址，使用 Markdown 格式存储
    df.to_markdown("testmem://mockfile", storage_options={"test": "md_write"})
    # 断言 fsspectest.test 列表的第一个元素是否为 "md_write"
    assert fsspectest.test[0] == "md_write"
    # 断言确保 testmem://mockfile 地址存在
    assert fsspectest.cat("testmem://mockfile")


# 测试非 fsspec 存储选项的处理
def test_non_fsspec_options():
    # 确保 pyarrow 库已导入，否则跳过该测试
    pytest.importorskip("pyarrow")
    # 使用 pytest 引发 ValueError 异常，匹配错误信息 "storage_options"
    with pytest.raises(ValueError, match="storage_options"):
        read_csv("localfile", storage_options={"a": True})
    # 使用 pytest 引发 ValueError 异常，匹配错误信息 "storage_options"
    with pytest.raises(ValueError, match="storage_options"):
        # 对 parquet 单独进行测试，因其具有不同的代码路径
        read_parquet("localfile", storage_options={"a": True})
    # 创建一个空的 BytesIO 对象
    by = io.BytesIO()
    # 使用 pytest 引发 ValueError 异常，匹配错误信息 "storage_options"
    with pytest.raises(ValueError, match="storage_options"):
        read_csv(by, storage_options={"a": True})
    # 创建一个包含单列 'a' 的 DataFrame 对象
    df = DataFrame({"a": [0]})
    # 使用 pytest 引发 ValueError 异常，匹配错误信息 "storage_options"
    with pytest.raises(ValueError, match="storage_options"):
        df.to_parquet("nonfsspecpath", storage_options={"a": True})
```