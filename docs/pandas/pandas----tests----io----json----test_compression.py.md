# `D:\src\scipysrc\pandas\pandas\tests\io\json\test_compression.py`

```
# 导入必要的模块和函数
from io import (
    BytesIO,      # 从 io 模块导入 BytesIO 类，用于处理二进制数据流
    StringIO,     # 从 io 模块导入 StringIO 类，用于处理文本数据流
)

import pytest     # 导入 pytest 模块，用于单元测试框架

import pandas as pd                 # 导入 pandas 库并简写为 pd
import pandas.util._test_decorators as td  # 导入 pandas 内部测试装饰器
import pandas._testing as tm         # 导入 pandas 内部测试工具


def test_compression_roundtrip(compression):
    # 创建一个包含数字和索引的 DataFrame 对象
    df = pd.DataFrame(
        [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        index=["A", "B"],
        columns=["X", "Y", "Z"],
    )

    # 在确保环境清洁的情况下，将 DataFrame 对象以 JSON 格式写入到文件
    with tm.ensure_clean() as path:
        df.to_json(path, compression=compression)
        # 断言：读取的 DataFrame 应当与原始的 DataFrame 相等
        tm.assert_frame_equal(df, pd.read_json(path, compression=compression))

        # 显式确认文件已压缩
        with tm.decompress_file(path, compression) as fh:
            result = fh.read().decode("utf8")
            data = StringIO(result)
        # 断言：解压后的 DataFrame 应当与原始的 DataFrame 相等
        tm.assert_frame_equal(df, pd.read_json(data))


def test_read_zipped_json(datapath):
    # 读取未压缩的 JSON 文件中的 DataFrame
    uncompressed_path = datapath("io", "json", "data", "tsframe_v012.json")
    uncompressed_df = pd.read_json(uncompressed_path)

    # 读取经压缩的 JSON 文件中的 DataFrame
    compressed_path = datapath("io", "json", "data", "tsframe_v012.json.zip")
    compressed_df = pd.read_json(compressed_path, compression="zip")

    # 断言：未压缩和经压缩的 DataFrame 应当相等
    tm.assert_frame_equal(uncompressed_df, compressed_df)


@td.skip_if_not_us_locale
@pytest.mark.single_cpu
def test_with_s3_url(compression, s3_public_bucket, s3so):
    # 创建一个简单的 JSON 字符串，并读取为 DataFrame
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))

    # 在确保环境清洁的情况下，将 DataFrame 对象以 JSON 格式写入到文件，并上传到 S3
    with tm.ensure_clean() as path:
        df.to_json(path, compression=compression)
        with open(path, "rb") as f:
            s3_public_bucket.put_object(Key="test-1", Body=f)

    # 从 S3 中读取文件并将其转换为 DataFrame
    roundtripped_df = pd.read_json(
        f"s3://{s3_public_bucket.name}/test-1",
        compression=compression,
        storage_options=s3so,
    )
    # 断言：上传和下载的 DataFrame 应当相等
    tm.assert_frame_equal(df, roundtripped_df)


def test_lines_with_compression(compression):
    # 在确保环境清洁的情况下，将 DataFrame 对象以行的形式写入到 JSON 文件
    with tm.ensure_clean() as path:
        df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
        df.to_json(path, orient="records", lines=True, compression=compression)
        # 以行的形式读取经压缩的 JSON 文件，并将其转换为 DataFrame
        roundtripped_df = pd.read_json(path, lines=True, compression=compression)
        # 断言：读取的行格式 JSON 文件的 DataFrame 应当与原始的 DataFrame 相等
        tm.assert_frame_equal(df, roundtripped_df)


def test_chunksize_with_compression(compression):
    # 在确保环境清洁的情况下，将 DataFrame 对象以行的形式写入到 JSON 文件
    with tm.ensure_clean() as path:
        df = pd.read_json(StringIO('{"a": ["foo", "bar", "baz"], "b": [4, 5, 6]}'))
        df.to_json(path, orient="records", lines=True, compression=compression)

        # 使用 chunksize 为 1 读取经压缩的 JSON 文件，并将其合并为 DataFrame
        with pd.read_json(
            path, lines=True, chunksize=1, compression=compression
        ) as res:
            roundtripped_df = pd.concat(res)
        # 断言：按块读取的 DataFrame 应当与原始的 DataFrame 相等
        tm.assert_frame_equal(df, roundtripped_df)


def test_write_unsupported_compression_type():
    # 创建一个简单的 JSON 字符串，并读取为 DataFrame
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
    with tm.ensure_clean() as path:
        # 断言：尝试使用不支持的压缩类型时，应当抛出 ValueError 异常
        msg = "Unrecognized compression type: unsupported"
        with pytest.raises(ValueError, match=msg):
            df.to_json(path, compression="unsupported")


def test_read_unsupported_compression_type():
    # 使用 tm.ensure_clean() 上下文管理器确保操作后资源被正确清理
    with tm.ensure_clean() as path:
        # 定义错误消息字符串，用于匹配 pytest 抛出的 ValueError 异常
        msg = "Unrecognized compression type: unsupported"
        # 使用 pytest.raises() 检查是否抛出指定类型的异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 读取指定路径的 JSON 文件，使用不支持的压缩类型会抛出异常
            pd.read_json(path, compression="unsupported")
@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
@pytest.mark.parametrize("to_infer", [True, False])
@pytest.mark.parametrize("read_infer", [True, False])
def test_to_json_compression(
    compression_only, read_infer, to_infer, compression_to_extension, infer_string
):
    with pd.option_context("future.infer_string", infer_string):
        # 设置 'future.infer_string' 上下文选项为 infer_string 的值
        compression = compression_only
        # 将 compression_only 参数赋给 compression 变量

        # 我们将逐步完整文件扩展名。
        filename = "test."
        filename += compression_to_extension[compression]
        # 构建文件名，通过 compression_to_extension 字典获取对应的扩展名

        df = pd.DataFrame({"A": [1]})
        # 创建一个包含一列为 [1] 的 DataFrame

        to_compression = "infer" if to_infer else compression
        # 如果 to_infer 为 True，则将 to_compression 设置为 "infer"，否则设置为 compression
        read_compression = "infer" if read_infer else compression
        # 如果 read_infer 为 True，则将 read_compression 设置为 "infer"，否则设置为 compression

        with tm.ensure_clean(filename) as path:
            # 在临时路径下操作
            df.to_json(path, compression=to_compression)
            # 将 DataFrame df 以指定的压缩方式 to_compression 写入到 JSON 文件
            result = pd.read_json(path, compression=read_compression)
            # 从指定的 JSON 文件中读取数据到 result DataFrame
            tm.assert_frame_equal(result, df)
            # 使用 tm.assert_frame_equal 检查 result 和 df 是否相等


def test_to_json_compression_mode(compression):
    # GH 39985 (read_json does not support user-provided binary files)
    expected = pd.DataFrame({"A": [1]})
    # 创建一个包含一列为 [1] 的 DataFrame，作为预期结果

    with BytesIO() as buffer:
        # 使用 BytesIO 创建一个内存缓冲区
        expected.to_json(buffer, compression=compression)
        # 将 DataFrame expected 以指定的压缩方式 compression 写入到 JSON 格式到缓冲区
        # df = pd.read_json(buffer, compression=compression)
        # 从缓冲区中读取 JSON 数据到 DataFrame df
        # tm.assert_frame_equal(expected, df)
        # 使用 tm.assert_frame_equal 检查 expected 和 df 是否相等，但此行被注释掉了
```