# `D:\src\scipysrc\pandas\pandas\tests\io\test_compression.py`

```
# 导入必要的模块和库
import gzip  # 导入用于 gzip 压缩的模块
import io  # 导入用于处理 IO 的模块
import os  # 导入操作系统相关功能的模块
from pathlib import Path  # 导入处理文件路径的模块
import subprocess  # 导入执行外部命令的模块
import sys  # 导入系统相关的模块
import tarfile  # 导入处理 tar 文件的模块
import textwrap  # 导入处理文本缩进和换行的模块
import time  # 导入处理时间的模块
import zipfile  # 导入处理 zip 文件的模块

import numpy as np  # 导入数值计算库 numpy
import pytest  # 导入用于编写和运行测试的 pytest

from pandas.compat import is_platform_windows  # 导入判断平台是否为 Windows 的函数

import pandas as pd  # 导入数据处理和分析的库 pandas
import pandas._testing as tm  # 导入 pandas 内部测试工具

import pandas.io.common as icom  # 导入 pandas IO 公共模块


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_compression_size(obj, method, compression_only):
    if compression_only == "tar":
        compression_only = {"method": "tar", "mode": "w:gz"}

    # 确保测试环境干净，创建临时文件路径
    with tm.ensure_clean() as path:
        # 使用对象的指定方法将数据写入路径，同时进行压缩
        getattr(obj, method)(path, compression=compression_only)
        # 获取压缩后文件的大小
        compressed_size = os.path.getsize(path)
        # 使用相同路径但不进行压缩的方式再次写入数据
        getattr(obj, method)(path, compression=None)
        # 获取未压缩文件的大小
        uncompressed_size = os.path.getsize(path)
        # 断言未压缩文件大小大于压缩文件大小
        assert uncompressed_size > compressed_size


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_csv", "to_json"])
def test_compression_size_fh(obj, method, compression_only):
    # 确保测试环境干净，创建临时文件路径
    with tm.ensure_clean() as path:
        # 获取文件句柄，根据压缩选项选择不同的模式
        with icom.get_handle(
            path,
            "w:gz" if compression_only == "tar" else "w",
            compression=compression_only,
        ) as handles:
            # 使用对象的指定方法将数据写入文件句柄
            getattr(obj, method)(handles.handle)
            # 断言文件句柄未关闭
            assert not handles.handle.closed
        # 获取压缩后文件的大小
        compressed_size = os.path.getsize(path)
    with tm.ensure_clean() as path:
        # 获取文件句柄，不进行压缩
        with icom.get_handle(path, "w", compression=None) as handles:
            # 使用对象的指定方法将数据写入文件句柄
            getattr(obj, method)(handles.handle)
            # 断言文件句柄未关闭
            assert not handles.handle.closed
        # 获取未压缩文件的大小
        uncompressed_size = os.path.getsize(path)
        # 断言未压缩文件大小大于压缩文件大小
        assert uncompressed_size > compressed_size


@pytest.mark.parametrize(
    "write_method, write_kwargs, read_method",
    [
        ("to_csv", {"index": False}, pd.read_csv),
        ("to_json", {}, pd.read_json),
        ("to_pickle", {}, pd.read_pickle),
    ],
)
def test_dataframe_compression_defaults_to_infer(
    write_method, write_kwargs, read_method, compression_only, compression_to_extension
):
    # GH22004
    # 创建输入的 DataFrame
    input = pd.DataFrame([[1.0, 0, -4], [3.4, 5, 2]], columns=["X", "Y", "Z"])
    # 根据压缩选项获取对应的文件扩展名
    extension = compression_to_extension[compression_only]
    # 确保测试环境干净，创建带有扩展名的临时文件路径
    with tm.ensure_clean("compressed" + extension) as path:
        # 使用指定的写入方法将输入数据写入路径
        getattr(input, write_method)(path, **write_kwargs)
        # 使用指定的读取方法读取路径中的数据，并与输入数据进行比较
        output = read_method(path, compression=compression_only)
    # 断言读取输出与输入数据相等
    tm.assert_frame_equal(output, input)


@pytest.mark.parametrize(
    "write_method,write_kwargs,read_method,read_kwargs",
    # 包含三个元组的列表，每个元组描述了一个操作：
    [
        # 第一个元组描述了使用 pandas 的 read_csv 函数进行操作
        ("to_csv", {"index": False, "header": True}, pd.read_csv, {"squeeze": True}),
        # 第二个元组描述了使用 pandas 的 read_json 函数进行操作
        ("to_json", {}, pd.read_json, {"typ": "series"}),
        # 第三个元组描述了使用 pandas 的 read_pickle 函数进行操作
        ("to_pickle", {}, pd.read_pickle, {}),
    ],
# 定义一个测试函数，用于检查默认情况下系列数据的压缩行为是否符合预期
def test_series_compression_defaults_to_infer(
    write_method,
    write_kwargs,
    read_method,
    read_kwargs,
    compression_only,
    compression_to_extension,
):
    # GH22004: GitHub issue编号，标识此测试的相关问题
    # 创建一个包含整数的系列数据，命名为"X"
    input = pd.Series([0, 5, -2, 10], name="X")
    # 根据压缩类型获取相应的文件扩展名
    extension = compression_to_extension[compression_only]
    # 确保在测试期间创建并使用干净的文件路径
    with tm.ensure_clean("compressed" + extension) as path:
        # 调用输入系列数据的指定写入方法，将数据写入到文件中
        getattr(input, write_method)(path, **write_kwargs)
        # 如果在读取参数中存在"squeeze"选项
        if "squeeze" in read_kwargs:
            # 复制读取参数，删除其中的"squeeze"选项
            kwargs = read_kwargs.copy()
            del kwargs["squeeze"]
            # 使用指定的读取方法从文件中读取数据，根据需要进行数据压缩解码，并压缩掉"columns"轴
            output = read_method(path, compression=compression_only, **kwargs).squeeze(
                "columns"
            )
        else:
            # 使用指定的读取方法从文件中读取数据，根据需要进行数据压缩解码
            output = read_method(path, compression=compression_only, **read_kwargs)
    # 断言输出的系列数据与输入的系列数据相等，忽略名称的检查
    tm.assert_series_equal(output, input, check_names=False)


# 定义一个测试函数，用于验证在指定压缩协议时，将文件对象传递给to_csv会触发RuntimeWarning
def test_compression_warning(compression_only):
    # 创建一个包含浮点数的数据框
    df = pd.DataFrame(
        100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        columns=["X", "Y", "Z"],
    )
    # 确保在测试期间创建并使用干净的文件路径
    with tm.ensure_clean() as path:
        # 使用指定的压缩协议打开文件句柄，指定写入方式为"w"
        with icom.get_handle(path, "w", compression=compression_only) as handles:
            # 断言在运行期间会产生RuntimeWarning，并匹配指定的警告信息
            with tm.assert_produces_warning(RuntimeWarning, match="has no effect"):
                # 将数据框df的内容以CSV格式写入到文件句柄中，指定压缩协议
                df.to_csv(handles.handle, compression=compression_only)


# 定义一个测试函数，用于验证二进制文件句柄是否支持压缩功能
def test_compression_binary(compression_only):
    """
    二进制文件句柄支持压缩功能。

    GH22555
    """
    # 创建一个包含浮点数的数据框
    df = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD"), dtype=object),
        index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # 使用文件路径测试
    with tm.ensure_clean() as path:
        # 使用指定的文件路径和写入模式"wb"，将数据框df以CSV格式写入文件，并指定压缩协议
        with open(path, mode="wb") as file:
            df.to_csv(file, mode="wb", compression=compression_only)
            file.seek(0)  # 保持文件未关闭状态
        # 断言读取的数据框与原始数据框df相等，指定压缩协议
        tm.assert_frame_equal(
            df, pd.read_csv(path, index_col=0, compression=compression_only)
        )

    # 使用BytesIO测试
    file = io.BytesIO()
    # 将数据框df以CSV格式写入BytesIO对象中，指定压缩协议
    df.to_csv(file, mode="wb", compression=compression_only)
    file.seek(0)  # 保持文件未关闭状态
    # 断言从BytesIO对象读取的数据框与原始数据框df相等，指定压缩协议
    tm.assert_frame_equal(
        df, pd.read_csv(file, index_col=0, compression=compression_only)
    )


# 定义一个测试函数，用于验证gzip是否能够创建具有可复现性的归档文件名
def test_gzip_reproducibility_file_name():
    """
    Gzip应该在mtime下创建可复现的归档。

    注意：使用不同文件名创建的归档文件将不同！

    GH 28103
    """
    # 创建一个包含浮点数的数据框
    df = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD"), dtype=object),
        index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 压缩选项设置，包括压缩方法和mtime值
    compression_options = {"method": "gzip", "mtime": 1}

    # 测试文件名情况
    # 使用tm.ensure_clean()上下文管理器确保操作的文件路径在操作完成后被正确清理
    with tm.ensure_clean() as path:
        # 将路径转换为Path对象
        path = Path(path)
        # 将DataFrame保存为CSV文件到指定路径，并使用给定的压缩选项进行压缩
        df.to_csv(path, compression=compression_options)
        # 等待0.1秒，确保文件操作完成
        time.sleep(0.1)
        # 读取保存在path路径下的文件的字节内容
        output = path.read_bytes()
        # 将DataFrame再次保存为CSV文件到相同的路径，使用相同的压缩选项
        df.to_csv(path, compression=compression_options)
        # 断言先前读取的文件字节内容与当前再次读取的文件字节内容相同
        assert output == path.read_bytes()
# 定义名为 test_gzip_reproducibility_file_object 的测试函数，测试 gzip 在保证修改时间的情况下创建可复现的存档。

df = pd.DataFrame(
    1.1 * np.arange(120).reshape((30, 4)),  # 创建一个 30 行 4 列的 DataFrame，数据是从 0 到 119 的浮点数乘以 1.1
    columns=pd.Index(list("ABCD"), dtype=object),  # 设置列索引为 ['A', 'B', 'C', 'D']，数据类型为 object
    index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),  # 设置行索引为 ['i-0', 'i-1', ..., 'i-29']，数据类型为 object
)
compression_options = {"method": "gzip", "mtime": 1}  # 定义压缩选项，使用 gzip 方法，并设置修改时间为 1

# 将 DataFrame 内容写入到字节流 buffer 中，采用 gzip 压缩方式，写入模式为二进制
buffer = io.BytesIO()
df.to_csv(buffer, compression=compression_options, mode="wb")
output = buffer.getvalue()  # 获取 buffer 中的数据作为输出

time.sleep(0.1)  # 等待 0.1 秒

# 重新创建一个字节流 buffer
buffer = io.BytesIO()
df.to_csv(buffer, compression=compression_options, mode="wb")
assert output == buffer.getvalue()  # 断言输出的内容与重新获取的 buffer 数据相同


@pytest.mark.single_cpu
def test_with_missing_lzma():
    """Tests if import pandas works when lzma is not present."""
    # https://github.com/pandas-dev/pandas/issues/27575
    code = textwrap.dedent(
        """\
        import sys
        sys.modules['lzma'] = None
        import pandas
        """
    )
    subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.PIPE)


@pytest.mark.single_cpu
def test_with_missing_lzma_runtime():
    """Tests if ModuleNotFoundError is hit when calling lzma without
    having the module available.
    """
    # 动态生成代码，模拟在缺少 lzma 模块时调用 pandas 的情况
    code = textwrap.dedent(
        """
        import sys
        import pytest
        sys.modules['lzma'] = None
        import pandas as pd
        df = pd.DataFrame()
        # 使用 pytest 断言捕获 ModuleNotFoundError 异常，匹配字符串 'import of lzma'
        with pytest.raises(ModuleNotFoundError, match='import of lzma'):
            df.to_csv('foo.csv', compression='xz')
        """
    )
    subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.PIPE)


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_gzip_compression_level(obj, method):
    # GH33196
    # 使用 tm.ensure_clean() 确保路径干净，执行不同的数据对象和方法，测试 gzip 压缩级别
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression="gzip")
        compressed_size_default = os.path.getsize(path)
        getattr(obj, method)(path, compression={"method": "gzip", "compresslevel": 1})
        compressed_size_fast = os.path.getsize(path)
        assert compressed_size_default < compressed_size_fast


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_xz_compression_level_read(obj, method):
    # 使用 tm.ensure_clean() 确保在退出代码块时清理临时文件
    with tm.ensure_clean() as path:
        # 调用 obj 对象的 method 方法，将结果压缩为 xz 格式并写入 path
        getattr(obj, method)(path, compression="xz")
        # 获取 path 文件的大小，即默认压缩方式下的文件大小
        compressed_size_default = os.path.getsize(path)
        # 再次调用 obj 对象的 method 方法，使用更快速的 xz 压缩方法，并写入 path
        getattr(obj, method)(path, compression={"method": "xz", "preset": 1})
        # 获取 path 文件使用更快速方法压缩后的大小
        compressed_size_fast = os.path.getsize(path)
        # 断言默认压缩方式的文件大小应小于更快速压缩方式的文件大小
        assert compressed_size_default < compressed_size_fast
        # 如果 method 是 "to_csv"，则使用 xz 压缩方式读取 path 文件内容为 pandas 数据框
        if method == "to_csv":
            pd.read_csv(path, compression="xz")
@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),  # 创建一个包含指定数据和列名的 DataFrame 对象
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),  # 创建一个包含指定数据和名称的 Series 对象
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_bzip_compression_level(obj, method):
    """GH33196 bzip needs file size > 100k to show a size difference between
    compression levels, so here we just check if the call works when
    compression is passed as a dict.
    """
    with tm.ensure_clean() as path:  # 在一个临时路径下进行操作，确保环境干净
        getattr(obj, method)(path, compression={"method": "bz2", "compresslevel": 1})  # 调用对象的指定方法，使用 bzip2 压缩算法，并设置压缩级别为 1


@pytest.mark.parametrize(
    "suffix,archive",
    [
        (".zip", zipfile.ZipFile),  # 指定后缀为 .zip 的文件及其操作对象为 ZipFile
        (".tar", tarfile.TarFile),  # 指定后缀为 .tar 的文件及其操作对象为 TarFile
    ],
)
def test_empty_archive_zip(suffix, archive):
    with tm.ensure_clean(filename=suffix) as path:  # 在一个临时路径下进行操作，确保环境干净，文件名后缀为 suffix
        with archive(path, "w"):  # 使用指定的 archive 对象创建文件 path，并以写入模式打开
            pass  # 空操作，即不往文件中写入内容
        with pytest.raises(ValueError, match="Zero files found"):  # 断言在读取时抛出 ValueError 异常，且异常信息为 "Zero files found"
            pd.read_csv(path)  # 读取 CSV 文件内容


def test_ambiguous_archive_zip():
    with tm.ensure_clean(filename=".zip") as path:  # 在一个临时路径下进行操作，确保环境干净，文件名后缀为 .zip
        with zipfile.ZipFile(path, "w") as file:  # 创建一个 ZipFile 对象并以写入模式打开文件 path
            file.writestr("a.csv", "foo,bar")  # 向 Zip 文件中写入字符串 "foo,bar"，文件名为 "a.csv"
            file.writestr("b.csv", "foo,bar")  # 向 Zip 文件中写入字符串 "foo,bar"，文件名为 "b.csv"
        with pytest.raises(ValueError, match="Multiple files found in ZIP file"):  # 断言在读取时抛出 ValueError 异常，且异常信息为 "Multiple files found in ZIP file"
            pd.read_csv(path)  # 读取 ZIP 文件内容


def test_ambiguous_archive_tar(tmp_path):
    csvAPath = tmp_path / "a.csv"  # 定义临时目录下的文件路径 csvAPath
    with open(csvAPath, "w", encoding="utf-8") as a:  # 打开 csvAPath 文件以写入模式，编码为 UTF-8
        a.write("foo,bar\n")  # 向文件中写入字符串 "foo,bar" 加换行符
    csvBPath = tmp_path / "b.csv"  # 定义临时目录下的文件路径 csvBPath
    with open(csvBPath, "w", encoding="utf-8") as b:  # 打开 csvBPath 文件以写入模式，编码为 UTF-8
        b.write("foo,bar\n")  # 向文件中写入字符串 "foo,bar" 加换行符

    tarpath = tmp_path / "archive.tar"  # 定义临时目录下的 Tar 文件路径 tarpath
    with tarfile.TarFile(tarpath, "w") as tar:  # 创建 TarFile 对象并以写入模式打开 tarpath
        tar.add(csvAPath, "a.csv")  # 将 csvAPath 文件添加到 Tar 文件中，目标路径为 "a.csv"
        tar.add(csvBPath, "b.csv")  # 将 csvBPath 文件添加到 Tar 文件中，目标路径为 "b.csv"

    with pytest.raises(ValueError, match="Multiple files found in TAR archive"):  # 断言在读取时抛出 ValueError 异常，且异常信息为 "Multiple files found in TAR archive"
        pd.read_csv(tarpath)  # 读取 Tar 文件内容


def test_tar_gz_to_different_filename():
    with tm.ensure_clean(filename=".foo") as file:  # 在一个临时路径下进行操作，确保环境干净，文件名后缀为 .foo
        pd.DataFrame(  # 创建一个 DataFrame 对象
            [["1", "2"]],  # 数据为 [["1", "2"]]
            columns=["foo", "bar"],  # 列名为 ["foo", "bar"]
        ).to_csv(file, compression={"method": "tar", "mode": "w:gz"}, index=False)  # 将 DataFrame 对象以 CSV 格式写入到文件 file 中，使用 gzip 压缩算法写入

        with gzip.open(file) as uncompressed:  # 打开 gzip 压缩文件 file
            with tarfile.TarFile(fileobj=uncompressed) as archive:  # 创建 TarFile 对象，从未压缩的文件对象中读取
                members = archive.getmembers()  # 获取 Tar 文件中的成员列表
                assert len(members) == 1  # 断言 Tar 文件中只有一个成员
                content = archive.extractfile(members[0]).read().decode("utf8")  # 读取 Tar 文件中第一个成员的内容，并解码为 UTF-8

                if is_platform_windows():  # 如果是 Windows 平台
                    expected = "foo,bar\r\n1,2\r\n"  # 预期的内容为 "foo,bar\r\n1,2\r\n"
                else:  # 如果是其他平台
                    expected = "foo,bar\n1,2\n"  # 预期的内容为 "foo,bar\n1,2\n"

                assert content == expected  # 断言读取的内容与预期的内容相同


def test_tar_no_error_on_close():
    with io.BytesIO() as buffer:  # 创建一个字节流缓冲区对象 buffer
        with icom._BytesTarFile(fileobj=buffer, mode="w"):  # 使用自定义的 _BytesTarFile 对象创建 Tar 文件，写入模式
            pass  # 空操作，即不往文件中写入内容
```