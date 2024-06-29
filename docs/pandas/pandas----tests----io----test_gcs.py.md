# `D:\src\scipysrc\pandas\pandas\tests\io\test_gcs.py`

```
# 从 io 模块导入 BytesIO 类
from io import BytesIO
# 导入 os 模块
import os
# 导入 pathlib 模块
import pathlib
# 导入 tarfile 模块
import tarfile
# 导入 zipfile 模块
import zipfile

# 导入 numpy 库，并将其重命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas 库中导入多个类和函数
from pandas import (
    DataFrame,      # 数据帧类
    Index,          # 索引类
    date_range,     # 时间范围生成函数
    read_csv,       # 读取 CSV 文件函数
    read_excel,     # 读取 Excel 文件函数
    read_json,      # 读取 JSON 文件函数
    read_parquet,   # 读取 Parquet 文件函数
)
# 导入 pandas 测试工具作为 tm
import pandas._testing as tm
# 从 pandas.util 中导入 _test_decorators 并重命名为 td
from pandas.util import _test_decorators as td

# 为 pytestmark 赋值，标记用于忽略特定警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 定义一个 pytest 的 fixture，模拟使用二进制缓冲区的 GCS 环境
@pytest.fixture
def gcs_buffer():
    """Emulate GCS using a binary buffer."""
    # 导入 gcsfs 库，如果不存在则跳过测试
    pytest.importorskip("gcsfs")
    # 导入 fsspec 库，如果不存在则跳过测试
    fsspec = pytest.importorskip("fsspec")

    # 创建一个 BytesIO 对象作为模拟的 GCS 缓冲区
    gcs_buffer = BytesIO()
    # 定义 gcs_buffer 对象的 close 方法，使其返回 True
    gcs_buffer.close = lambda: True

    # 定义一个 MockGCSFileSystem 类，模拟 GCS 文件系统
    class MockGCSFileSystem(fsspec.AbstractFileSystem):
        @staticmethod
        def open(*args, **kwargs):
            # 重置 gcs_buffer 的指针位置到起始处
            gcs_buffer.seek(0)
            return gcs_buffer

        def ls(self, path, **kwargs):
            # 返回一个字典列表，表示模拟的文件系统中存在一个文件
            return [{"name": path, "type": "file"}]

    # 将 MockGCSFileSystem 注册为 gs 协议的实现，覆盖默认实现
    fsspec.register_implementation("gs", MockGCSFileSystem, clobber=True)

    # 返回创建的模拟 GCS 缓冲区对象
    return gcs_buffer


# 标记测试用例为单 CPU 运行，并为 format 参数进行参数化
@pytest.mark.single_cpu
@pytest.mark.parametrize("format", ["csv", "json", "parquet", "excel", "markdown"])
def test_to_read_gcs(gcs_buffer, format, monkeypatch, capsys):
    """
    Test that many to/read functions support GCS.

    GH 33987
    """

    # 创建一个包含数据的 DataFrame 对象 df1
    df1 = DataFrame(
        {
            "int": [1, 3],
            "float": [2.0, np.nan],
            "str": ["t", "s"],
            "dt": date_range("2018-06-18", periods=2),
        }
    )

    # 根据不同的格式进行条件分支处理
    path = f"gs://test/test.{format}"

    if format == "csv":
        # 将 df1 数据写入到 CSV 文件
        df1.to_csv(path, index=True)
        # 读取并解析 CSV 文件内容到 DataFrame df2
        df2 = read_csv(path, parse_dates=["dt"], index_col=0)
    elif format == "excel":
        # 将 df1 数据写入到 Excel 文件
        path = "gs://test/test.xlsx"
        df1.to_excel(path)
        # 读取并解析 Excel 文件内容到 DataFrame df2
        df2 = read_excel(path, parse_dates=["dt"], index_col=0)
    elif format == "json":
        # 将 df1 数据写入到 JSON 文件
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 检测是否产生特定警告信息，并进行 JSON 文件的读取
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df1.to_json(path)
        df2 = read_json(path, convert_dates=["dt"])
    # 如果格式是 "parquet"，则执行以下代码块
    elif format == "parquet":
        # 导入 pytest 并检查是否存在 pyarrow，如果不存在则跳过测试
        pytest.importorskip("pyarrow")
        # 导入 pytest 并检查是否存在 pyarrow.fs，如果不存在则跳过测试
        pa_fs = pytest.importorskip("pyarrow.fs")

        # 定义 MockFileSystem 类，继承自 pa_fs.FileSystem
        class MockFileSystem(pa_fs.FileSystem):
            # 静态方法：根据给定的路径创建 MockFileSystem 实例
            @staticmethod
            def from_uri(path):
                # 输出提示信息，表明正在使用 pyarrow 文件系统
                print("Using pyarrow filesystem")
                # 将路径中的 "gs://" 替换为本地文件系统路径，并转换为 URI
                to_local = pathlib.Path(path.replace("gs://", "")).absolute().as_uri()
                # 返回一个新的本地文件系统实例
                return pa_fs.LocalFileSystem(to_local)

        # 使用 monkeypatch 上下文
        with monkeypatch.context() as m:
            # 替换 pa_fs.FileSystem 为 MockFileSystem 类
            m.setattr(pa_fs, "FileSystem", MockFileSystem)
            # 将 DataFrame df1 保存为 parquet 文件到指定路径
            df1.to_parquet(path)
            # 使用 read_parquet 函数读取 parquet 文件，并将结果赋给 df2
            df2 = read_parquet(path)
        
        # 读取 capsys 捕获的标准输出和标准错误流
        captured = capsys.readouterr()
        # 断言捕获的标准输出是否包含指定的提示信息
        assert captured.out == "Using pyarrow filesystem\nUsing pyarrow filesystem\n"

    # 如果格式是 "markdown"，则执行以下代码块
    elif format == "markdown":
        # 导入 pytest 并检查是否存在 tabulate，如果不存在则跳过测试
        pytest.importorskip("tabulate")
        # 将 DataFrame df1 转换为 markdown 格式并保存到指定路径
        df1.to_markdown(path)
        # 将 df2 设置为 df1 的副本
        df2 = df1

    # 将 DataFrame df1 的预期值复制给 expected
    expected = df1[:]
    # 如果格式是 "csv" 或 "excel"，则执行以下代码块
    if format in ["csv", "excel"]:
        # 将 expected DataFrame 的 "dt" 列转换为秒单位
        expected["dt"] = expected["dt"].dt.as_unit("s")

    # 使用 tm.assert_frame_equal 函数断言 df2 和 expected 是否相等
    tm.assert_frame_equal(df2, expected)
def assert_equal_zip_safe(result: bytes, expected: bytes, compression: str):
    """
    For zip compression, only compare the CRC-32 checksum of the file contents
    to avoid checking the time-dependent last-modified timestamp which
    in some CI builds is off-by-one

    See https://en.wikipedia.org/wiki/ZIP_(file_format)#File_headers
    """
    # 如果压缩方法为 zip
    if compression == "zip":
        # 仅比较文件内容的 CRC 校验和
        with (
            zipfile.ZipFile(BytesIO(result)) as exp,  # 使用结果的字节流创建 ZipFile 对象
            zipfile.ZipFile(BytesIO(expected)) as res,  # 使用期望的字节流创建 ZipFile 对象
        ):
            # 遍历结果和期望的文件信息列表
            for res_info, exp_info in zip(res.infolist(), exp.infolist()):
                # 断言结果和期望文件的 CRC 校验和相等
                assert res_info.CRC == exp_info.CRC
    # 如果压缩方法为 tar
    elif compression == "tar":
        with (
            tarfile.open(fileobj=BytesIO(result)) as tar_exp,  # 使用结果的字节流创建 TarFile 对象
            tarfile.open(fileobj=BytesIO(expected)) as tar_res,  # 使用期望的字节流创建 TarFile 对象
        ):
            # 遍历结果和期望的文件成员列表
            for tar_res_info, tar_exp_info in zip(
                tar_res.getmembers(), tar_exp.getmembers()
            ):
                # 获取实际文件和期望文件对象
                actual_file = tar_res.extractfile(tar_res_info)
                expected_file = tar_exp.extractfile(tar_exp_info)
                # 断言实际文件是否为空与期望文件是否为空的相等性
                assert (actual_file is None) == (expected_file is None)
                # 如果实际文件和期望文件都不为空，则断言它们的内容相等
                if actual_file is not None and expected_file is not None:
                    assert actual_file.read() == expected_file.read()
    # 如果压缩方法不是 zip 或 tar，则直接比较结果和期望是否相等
    else:
        assert result == expected


@pytest.mark.parametrize("encoding", ["utf-8", "cp1251"])
def test_to_csv_compression_encoding_gcs(
    gcs_buffer, compression_only, encoding, compression_to_extension
):
    """
    Compression and encoding should with GCS.

    GH 35677 (to_csv, compression), GH 26124 (to_csv, encoding), and
    GH 32392 (read_csv, encoding)
    """
    # 创建一个 DataFrame 对象
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),  # 使用 numpy 创建数据
        columns=Index(list("ABCD"), dtype=object),  # 设置列索引
        index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 设置行索引
    )

    # 设置压缩方式和相关参数的字典
    compression = {"method": compression_only}
    # 如果压缩方式为 gzip，则设置可重现的修改时间
    if compression_only == "gzip":
        compression["mtime"] = 1  # be reproducible
    # 创建一个字节流对象
    buffer = BytesIO()
    # 将 DataFrame 对象写入到字节流中，使用指定的压缩和编码方式
    df.to_csv(buffer, compression=compression, encoding=encoding, mode="wb")

    # 将压缩后的文件写入到指定路径，使用指定的压缩和编码方式
    path_gcs = "gs://test/test.csv"
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    # 获取写入到 GCS 缓冲区的结果
    res = gcs_buffer.getvalue()
    # 获取预期结果的字节流
    expected = buffer.getvalue()
    # 调用 assert_equal_zip_safe 函数比较结果和预期结果，使用指定的压缩方法
    assert_equal_zip_safe(res, expected, compression_only)

    # 读取存储在 GCS 中的 CSV 文件到 DataFrame 对象
    read_df = read_csv(
        path_gcs, index_col=0, compression=compression_only, encoding=encoding
    )
    # 使用断言验证读取的 DataFrame 和原始 DataFrame 相等
    tm.assert_frame_equal(df, read_df)

    # 写入压缩文件，使用隐式推断的压缩方法
    file_ext = compression_to_extension[compression_only]
    compression["method"] = "infer"
    path_gcs += f".{file_ext}"
    df.to_csv(path_gcs, compression=compression, encoding=encoding)

    # 获取写入到 GCS 缓冲区的结果
    res = gcs_buffer.getvalue()
    # 获取预期结果的字节流
    expected = buffer.getvalue()
    # 使用自定义的断言函数 assert_equal_zip_safe 检查 res 和 expected 是否相等，考虑到压缩格式
    assert_equal_zip_safe(res, expected, compression_only)

    # 从指定路径读取 CSV 文件到 DataFrame read_df，推断压缩格式，使用指定编码解析
    read_df = read_csv(path_gcs, index_col=0, compression="infer", encoding=encoding)
    
    # 使用测试工具（如 pytest 的 assert_frame_equal）比较 df 和 read_df 是否相等
    tm.assert_frame_equal(df, read_df)
# 使用 monkeypatch 和 tmpdir 作为参数的测试函数，用于测试写入尚不存在的 GCS Parquet 文件的回归情况
def test_to_parquet_gcs_new_file(monkeypatch, tmpdir):
    # 检查是否导入了 fastparquet 库，否则跳过测试
    pytest.importorskip("fastparquet")
    # 检查是否导入了 gcsfs 库，否则跳过测试
    pytest.importorskip("gcsfs")

    # 导入 AbstractFileSystem 类
    from fsspec import AbstractFileSystem

    # 创建一个 DataFrame 对象 df1
    df1 = DataFrame(
        {
            "int": [1, 3],
            "float": [2.0, np.nan],
            "str": ["t", "s"],
            "dt": date_range("2018-06-18", periods=2),
        }
    )

    # 定义 MockGCSFileSystem 类，继承自 AbstractFileSystem
    class MockGCSFileSystem(AbstractFileSystem):
        # 重写 open 方法
        def open(self, path, mode="r", *args):
            # 如果模式中不包含 'w'，则抛出 FileNotFoundError 异常
            if "w" not in mode:
                raise FileNotFoundError
            # 返回一个以 UTF-8 编码打开的文件对象，路径为 tmpdir/test.parquet
            return open(os.path.join(tmpdir, "test.parquet"), mode, encoding="utf-8")

    # 使用 monkeypatch 替换 gcsfs.GCSFileSystem 类为 MockGCSFileSystem 类
    monkeypatch.setattr("gcsfs.GCSFileSystem", MockGCSFileSystem)
    
    # 将 df1 对象写入 Parquet 格式到 gs://test/test.csv，包括索引，使用 fastparquet 引擎，无压缩
    df1.to_parquet(
        "gs://test/test.csv", index=True, engine="fastparquet", compression=None
    )


# 使用 @td.skip_if_installed 装饰的测试函数，如果 gcsfs 已安装则跳过测试
def test_gcs_not_present_exception():
    # 使用 tm.external_error_raised 捕获 ImportError 异常
    with tm.external_error_raised(ImportError):
        # 调用 read_csv 函数尝试读取 gs://test/test.csv 文件
        read_csv("gs://test/test.csv")
```