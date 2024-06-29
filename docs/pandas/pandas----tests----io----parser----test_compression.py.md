# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_compression.py`

```
"""
Tests compressed data parsing functionality for all
of the parsers defined in parsers.py
"""

# 导入必要的库和模块
import os
from pathlib import Path
import tarfile
import zipfile

# 导入 pytest 库
import pytest

# 导入 pandas 库的 DataFrame 类和测试模块
from pandas import DataFrame
import pandas._testing as tm

# 设置 pytest 标记来忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


# 定义一个 pytest fixture，参数为 True 或 False
@pytest.fixture(params=[True, False])
def buffer(request):
    return request.param


# 定义一个 pytest fixture，返回 parser 和 csv1 文件内容
@pytest.fixture
def parser_and_data(all_parsers, csv1):
    parser = all_parsers

    # 打开并读取 csv1 文件的内容
    with open(csv1, "rb") as f:
        data = f.read()

    # 使用 parser 解析 csv1 文件，得到预期结果 expected
    expected = parser.read_csv(csv1)

    return parser, data, expected


# 使用 pytest.mark.parametrize 来定义压缩方式的参数化测试
@pytest.mark.parametrize("compression", ["zip", "infer", "zip2"])
def test_zip(parser_and_data, compression):
    parser, data, expected = parser_and_data

    # 确保在 "test_file.zip" 文件上进行测试，并在测试结束后清理文件
    with tm.ensure_clean("test_file.zip") as path:
        # 创建一个新的 ZipFile 对象 tmp，并将数据写入其中的 "test_file" 文件
        with zipfile.ZipFile(path, mode="w") as tmp:
            tmp.writestr("test_file", data)

        # 根据不同的 compression 参数读取压缩文件，得到结果 result
        if compression == "zip2":
            with open(path, "rb") as f:
                result = parser.read_csv(f, compression="zip")
        else:
            result = parser.read_csv(path, compression=compression)

        # 使用 pandas 的 assert_frame_equal 函数比较 result 和 expected
        tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 定义压缩方式的参数化测试
@pytest.mark.parametrize("compression", ["zip", "infer"])
def test_zip_error_multiple_files(parser_and_data, compression):
    parser, data, expected = parser_and_data

    # 确保在 "combined_zip.zip" 文件上进行测试，并在测试结束后清理文件
    with tm.ensure_clean("combined_zip.zip") as path:
        inner_file_names = ["test_file", "second_file"]

        # 创建一个新的 ZipFile 对象 tmp，并将多个文件数据写入其中
        with zipfile.ZipFile(path, mode="w") as tmp:
            for file_name in inner_file_names:
                tmp.writestr(file_name, data)

        # 使用 parser.read_csv 读取压缩文件，预期会抛出 ValueError 异常
        with pytest.raises(ValueError, match="Multiple files"):
            parser.read_csv(path, compression=compression)


# 定义测试函数来测试当没有文件存在时的错误情况
def test_zip_error_no_files(parser_and_data):
    parser, _, _ = parser_and_data

    # 确保在一个临时文件上进行测试，并在测试结束后清理文件
    with tm.ensure_clean() as path:
        # 创建一个空的 ZipFile 对象
        with zipfile.ZipFile(path, mode="w"):
            pass

        # 使用 parser.read_csv 读取空的压缩文件，预期会抛出 ValueError 异常
        with pytest.raises(ValueError, match="Zero files"):
            parser.read_csv(path, compression="zip")


# 定义测试函数来测试当压缩文件无效时的错误情况
def test_zip_error_invalid_zip(parser_and_data):
    parser, _, _ = parser_and_data

    # 确保在一个临时文件上进行测试，并在测试结束后清理文件
    with tm.ensure_clean() as path:
        # 打开临时文件，并预期会抛出 zipfile.BadZipFile 异常
        with open(path, "rb") as f:
            with pytest.raises(zipfile.BadZipFile, match="File is not a zip file"):
                parser.read_csv(f, compression="zip")


# 使用 pytest.mark.parametrize 来定义测试文件名参数化测试
@pytest.mark.parametrize("filename", [None, "test.{ext}"])
def test_compression(
    request,
    parser_and_data,
    compression_only,
    buffer,
    filename,
    compression_to_extension,
):
    parser, data, expected = parser_and_data
    compress_type = compression_only

    # 根据 compression_only 映射出对应的文件扩展名 ext
    ext = compression_to_extension[compress_type]
    # 如果 filename 存在且 buffer 为真，则应用 pytest.mark.xfail 标记
    filename = filename if filename is None else filename.format(ext=ext)

    if filename and buffer:
        request.applymarker(
            pytest.mark.xfail(
                reason="Cannot deduce compression from buffer of compressed data."
            )
        )
    # 使用tm.ensure_clean上下文管理器确保在处理文件时可以安全地清理
    with tm.ensure_clean(filename=filename) as path:
        # 使用tm.write_to_compressed函数将数据写入指定路径的压缩文件中
        tm.write_to_compressed(compress_type, path, data)
        # 根据是否提供了文件名，决定压缩类型的选择，若未提供则使用推测模式（infer）
        compression = "infer" if filename else compress_type

        # 如果buffer为真，使用上下文管理器打开文件并读取数据
        if buffer:
            with open(path, "rb") as f:
                # 使用parser.read_csv从打开的文件对象中读取CSV数据，指定压缩类型
                result = parser.read_csv(f, compression=compression)
        else:
            # 否则直接使用parser.read_csv从文件路径中读取CSV数据，指定压缩类型
            result = parser.read_csv(path, compression=compression)

        # 使用tm.assert_frame_equal断言读取的结果与期望的结果相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("ext", [None, "gz", "bz2"])
def test_infer_compression(all_parsers, csv1, buffer, ext):
    # 使用 pytest.mark.parametrize 装饰器为 test_infer_compression 函数定义参数化测试
    # 这里的参数 ext 可以是 None、"gz" 或 "bz2"
    # 详见 GitHub 问题 gh-9770
    parser = all_parsers
    kwargs = {"index_col": 0, "parse_dates": True}

    # 从 csv1 文件中读取数据，使用给定的解析器 parser 和 kwargs 参数
    expected = parser.read_csv(csv1, **kwargs)
    kwargs["compression"] = "infer"  # 设定 compression 参数为 "infer"

    if buffer:
        # 如果 buffer 为 True，则使用文件对象 f 来读取 csv1 文件
        with open(csv1, encoding="utf-8") as f:
            result = parser.read_csv(f, **kwargs)
    else:
        # 否则，根据 ext 的值（如果有的话），构造文件名并读取数据
        ext = "." + ext if ext else ""
        result = parser.read_csv(csv1 + ext, **kwargs)

    # 使用 pandas.testing.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_compression_utf_encoding(all_parsers, csv_dir_path, utf_value, encoding_fmt):
    # 详见 GitHub 问题 gh-18071 和 gh-24130
    parser = all_parsers
    encoding = encoding_fmt.format(utf_value)
    # 构造带有特定编码的文件路径
    path = os.path.join(csv_dir_path, f"utf{utf_value}_ex_small.zip")

    # 使用 parser 读取压缩文件 path，指定编码和压缩类型为 "zip"，分隔符为制表符
    result = parser.read_csv(path, encoding=encoding, compression="zip", sep="\t")

    # 创建预期的 DataFrame 对象
    expected = DataFrame(
        {
            "Country": ["Venezuela", "Venezuela"],
            "Twitter": ["Hugo Chávez Frías", "Henrique Capriles R."],
        }
    )

    # 使用 pandas.testing.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("invalid_compression", ["sfark", "bz3", "zipper"])
def test_invalid_compression(all_parsers, invalid_compression):
    parser = all_parsers
    compress_kwargs = {"compression": invalid_compression}

    # 构造错误信息字符串，包含无法识别的压缩类型
    msg = f"Unrecognized compression type: {invalid_compression}"

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并验证异常信息是否符合预期
    with pytest.raises(ValueError, match=msg):
        parser.read_csv("test_file.zip", **compress_kwargs)


def test_compression_tar_archive(all_parsers, csv_dir_path):
    parser = all_parsers
    # 构造 tar.gz 文件的路径
    path = os.path.join(csv_dir_path, "tar_csv.tar.gz")

    # 使用 parser 读取 tar.gz 文件，并将结果存储在 DataFrame 对象 df 中
    df = parser.read_csv(path)

    # 使用断言验证 df 的列是否为 ["a"]
    assert list(df.columns) == ["a"]


def test_ignore_compression_extension(all_parsers):
    parser = all_parsers
    df = DataFrame({"a": [0, 1]})

    # 确保创建具有 zip 扩展名但未压缩的文件
    with tm.ensure_clean("test.csv") as path_csv:
        with tm.ensure_clean("test.csv.zip") as path_zip:
            # 将 DataFrame df 写入 path_csv 文件，不包括索引
            df.to_csv(path_csv, index=False)

            # 将 path_csv 的内容复制到 path_zip，创建未压缩的 zip 文件
            Path(path_zip).write_text(
                Path(path_csv).read_text(encoding="utf-8"), encoding="utf-8"
            )

            # 使用 parser 读取 path_zip 文件，确认 compression 参数为 None，验证结果与 df 相等
            tm.assert_frame_equal(parser.read_csv(path_zip, compression=None), df)


def test_writes_tar_gz(all_parsers):
    parser = all_parsers
    data = DataFrame(
        {
            "Country": ["Venezuela", "Venezuela"],
            "Twitter": ["Hugo Chávez Frías", "Henrique Capriles R."],
        }
    )

    # 确保创建具有 .tar.gz 扩展名的文件
    with tm.ensure_clean("test.tar.gz") as tar_path:
        data.to_csv(tar_path, index=False)

        # 使用 parser 读取 tar_path 文件，并使用 pandas.testing.assert_frame_equal 验证结果与 data 相等
        tm.assert_frame_equal(parser.read_csv(tar_path), data)

        # 验证文件确实是被 gzip 压缩的
        with tarfile.open(tar_path, "r:gz") as tar:
            # 读取 tar 文件的第一个文件对象，并推断其压缩类型
            result = parser.read_csv(
                tar.extractfile(tar.getnames()[0]), compression="infer"
            )

            # 使用 pandas.testing.assert_frame_equal 验证 result 和 data 是否相等
            tm.assert_frame_equal(result, data)
```