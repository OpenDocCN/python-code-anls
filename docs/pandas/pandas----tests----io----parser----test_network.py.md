# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_network.py`

```
"""
Tests parsers ability to read and parse non-local files
and hence require a network connection to be read.
"""

# 导入所需模块和库
from io import BytesIO  # 从 io 模块导入 BytesIO 类
import logging  # 导入 logging 模块，用于记录日志
import re  # 导入 re 模块，用于正则表达式操作

import numpy as np  # 导入 numpy 库，命名为 np
import pytest  # 导入 pytest 库

import pandas.util._test_decorators as td  # 导入 pandas.util._test_decorators 模块，命名为 td

from pandas import DataFrame  # 从 pandas 库导入 DataFrame 类
import pandas._testing as tm  # 导入 pandas._testing 模块，命名为 tm

from pandas.io.feather_format import read_feather  # 从 pandas.io.feather_format 导入 read_feather 函数
from pandas.io.parsers import read_csv  # 从 pandas.io.parsers 导入 read_csv 函数

# 设置 pytest 的标记来忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.mark.network  # 标记为网络测试
@pytest.mark.single_cpu  # 标记为单核 CPU 测试
@pytest.mark.parametrize("mode", ["explicit", "infer"])  # 参数化测试，测试模式为 explicit 和 infer
@pytest.mark.parametrize("engine", ["python", "c"])  # 参数化测试，测试引擎为 python 和 c
def test_compressed_urls(
    httpserver,
    datapath,
    salaries_table,
    mode,
    engine,
    compression_only,
    compression_to_extension,
):
    # 测试读取带压缩格式的 URL，使用不同引擎和扩展名推断
    if compression_only == "tar":
        pytest.skip("TODO: Add tar salaraies.csv to pandas/io/parsers/data")

    extension = compression_to_extension[compression_only]
    with open(datapath("io", "parser", "data", "salaries.csv" + extension), "rb") as f:
        httpserver.serve_content(content=f.read())

    url = httpserver.url + "/salaries.csv" + extension

    if mode != "explicit":
        compression_only = mode

    # 使用 read_csv 函数读取 URL 数据，设置分隔符为制表符，压缩类型为 compression_only，引擎为 engine
    url_table = read_csv(url, sep="\t", compression=compression_only, engine=engine)
    # 断言读取的数据框架与预期的 salaries_table 相等
    tm.assert_frame_equal(url_table, salaries_table)


@pytest.mark.network  # 标记为网络测试
@pytest.mark.single_cpu  # 标记为单核 CPU 测试
def test_url_encoding_csv(httpserver, datapath):
    """
    read_csv should honor the requested encoding for URLs.

    GH 10424
    """
    with open(datapath("io", "parser", "data", "unicode_series.csv"), "rb") as f:
        httpserver.serve_content(content=f.read())
        # 使用 read_csv 函数读取 URL 数据，指定编码为 latin-1，无标题
        df = read_csv(httpserver.url, encoding="latin-1", header=None)
    # 断言特定位置的数据与预期相等
    assert df.loc[15, 1] == "Á köldum klaka (Cold Fever) (1994)"


@pytest.fixture
def tips_df(datapath):
    """DataFrame with the tips dataset."""
    # 返回读取的 tips.csv 文件内容生成的数据框架
    return read_csv(datapath("io", "data", "csv", "tips.csv"))


@pytest.mark.single_cpu  # 标记为单核 CPU 测试
@pytest.mark.usefixtures("s3_resource")
@td.skip_if_not_us_locale()
class TestS3:
    def test_parse_public_s3_bucket(self, s3_public_bucket_with_data, tips_df, s3so):
        # 对公共 S3 存储桶的解析测试，包含对非公共内容的集成测试部分
        # 可能可以使用模拟来实现这一部分
        pytest.importorskip("s3fs")
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            # 使用 read_csv 函数读取 S3 存储桶中的数据，设置压缩类型和存储选项
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                compression=comp,
                storage_options=s3so,
            )
            # 断言返回的对象为 DataFrame 类型且非空，并与预期的 tips_df 相等
            assert isinstance(df, DataFrame)
            assert not df.empty
            tm.assert_frame_equal(df, tips_df)
    def test_parse_private_s3_bucket(self, s3_private_bucket_with_data, tips_df, s3so):
        # 如果没有安装 s3fs 模块，跳过当前测试
        pytest.importorskip("s3fs")
        # 从私有 S3 存储桶中读取名为 "tips.csv" 的文件
        df = read_csv(
            f"s3://{s3_private_bucket_with_data.name}/tips.csv", storage_options=s3so
        )
        # 断言返回的数据类型为 DataFrame
        assert isinstance(df, DataFrame)
        # 断言 DataFrame 不为空
        assert not df.empty
        # 比较读取的 DataFrame 和预期的 tips_df 是否相等
        tm.assert_frame_equal(df, tips_df)

    def test_parse_public_s3n_bucket(self, s3_public_bucket_with_data, tips_df, s3so):
        # 从公共 S3 存储桶中读取名为 "tips.csv" 的文件，使用 s3n 协议
        df = read_csv(
            f"s3n://{s3_public_bucket_with_data.name}/tips.csv",
            nrows=10,
            storage_options=s3so,
        )
        # 断言返回的数据类型为 DataFrame
        assert isinstance(df, DataFrame)
        # 断言 DataFrame 不为空
        assert not df.empty
        # 比较读取的 DataFrame 和预期的 tips_df 的前 10 行是否相等
        tm.assert_frame_equal(tips_df.iloc[:10], df)

    def test_parse_public_s3a_bucket(self, s3_public_bucket_with_data, tips_df, s3so):
        # 从公共 S3 存储桶中读取名为 "tips.csv" 的文件，使用 s3a 协议
        df = read_csv(
            f"s3a://{s3_public_bucket_with_data.name}/tips.csv",
            nrows=10,
            storage_options=s3so,
        )
        # 断言返回的数据类型为 DataFrame
        assert isinstance(df, DataFrame)
        # 断言 DataFrame 不为空
        assert not df.empty
        # 比较读取的 DataFrame 和预期的 tips_df 的前 10 行是否相等
        tm.assert_frame_equal(tips_df.iloc[:10], df)

    def test_parse_public_s3_bucket_nrows(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        # 循环处理不同的文件扩展名和压缩格式，从公共 S3 存储桶中读取文件 "tips.csv"
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                nrows=10,
                compression=comp,
                storage_options=s3so,
            )
            # 断言返回的数据类型为 DataFrame
            assert isinstance(df, DataFrame)
            # 断言 DataFrame 不为空
            assert not df.empty
            # 比较读取的 DataFrame 和预期的 tips_df 的前 10 行是否相等
            tm.assert_frame_equal(tips_df.iloc[:10], df)

    def test_parse_public_s3_bucket_chunked(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        # 使用分块读取的方式处理文件，从公共 S3 存储桶中读取文件 "tips.csv"
        chunksize = 5
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            with read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                chunksize=chunksize,
                compression=comp,
                storage_options=s3so,
            ) as df_reader:
                # 断言返回的数据流对象的块大小是否为预期的 chunksize
                assert df_reader.chunksize == chunksize
                for i_chunk in [0, 1, 2]:
                    # 逐块读取数据并确保数据正确性
                    df = df_reader.get_chunk()
                    # 断言返回的数据类型为 DataFrame
                    assert isinstance(df, DataFrame)
                    # 断言 DataFrame 不为空
                    assert not df.empty
                    # 获取预期的数据块
                    true_df = tips_df.iloc[
                        chunksize * i_chunk : chunksize * (i_chunk + 1)
                    ]
                    # 比较读取的 DataFrame 块和预期的数据块是否相等
                    tm.assert_frame_equal(true_df, df)
    ):
        # 使用 Python 解析器以块大小读取数据
        chunksize = 5
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            # 使用 read_csv 函数读取指定 S3 存储桶中的 tips.csv 文件，并指定压缩格式和存储选项
            with read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                chunksize=chunksize,
                compression=comp,
                engine="python",
                storage_options=s3so,
            ) as df_reader:
                # 断言每个 chunk 的大小与设定的 chunksize 一致
                assert df_reader.chunksize == chunksize
                for i_chunk in [0, 1, 2]:
                    # 读取几个 chunk 并确保能正确读取
                    df = df_reader.get_chunk()
                    assert isinstance(df, DataFrame)
                    assert not df.empty
                    # 从 tips_df 中获取与当前 chunk 对应的真实数据，进行比较
                    true_df = tips_df.iloc[
                        chunksize * i_chunk : chunksize * (i_chunk + 1)
                    ]
                    tm.assert_frame_equal(true_df, df)

    def test_parse_public_s3_bucket_python(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            # 使用 read_csv 函数读取指定 S3 存储桶中的 tips.csv 文件，并指定压缩格式和存储选项
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                engine="python",
                compression=comp,
                storage_options=s3so,
            )
            # 断言返回的数据类型为 DataFrame，并且不为空
            assert isinstance(df, DataFrame)
            assert not df.empty
            # 使用测试框架确保返回的 DataFrame 与预期的 tips_df 相等
            tm.assert_frame_equal(df, tips_df)

    def test_infer_s3_compression(self, s3_public_bucket_with_data, tips_df, s3so):
        for ext in ["", ".gz", ".bz2"]:
            # 使用 read_csv 函数读取指定 S3 存储桶中的 tips.csv 文件，并指定压缩格式为推测模式和存储选项
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                engine="python",
                compression="infer",
                storage_options=s3so,
            )
            # 断言返回的数据类型为 DataFrame，并且不为空
            assert isinstance(df, DataFrame)
            assert not df.empty
            # 使用测试框架确保返回的 DataFrame 与预期的 tips_df 相等
            tm.assert_frame_equal(df, tips_df)

    def test_parse_public_s3_bucket_nrows_python(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            # 使用 read_csv 函数读取指定 S3 存储桶中的 tips.csv 文件，并指定行数、压缩格式和存储选项
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                engine="python",
                nrows=10,
                compression=comp,
                storage_options=s3so,
            )
            # 断言返回的数据类型为 DataFrame，并且不为空
            assert isinstance(df, DataFrame)
            assert not df.empty
            # 使用测试框架确保返回的 DataFrame 前 10 行与预期的 tips_df 的前 10 行相等
            tm.assert_frame_equal(tips_df.iloc[:10], df)

    def test_read_s3_fails(self, s3so):
        msg = "The specified bucket does not exist"
        # 使用 pytest 断言读取不存在的 S3 存储桶时会抛出 OSError 异常，并包含指定的错误消息
        with pytest.raises(OSError, match=msg):
            read_csv("s3://nyqpug/asdf.csv", storage_options=s3so)
    def test_read_s3_fails_private(self, s3_private_bucket, s3so):
        # 定义错误消息，用于匹配权限错误异常
        msg = "The specified bucket does not exist"
        # 当尝试读取私有桶时，预期会收到权限错误异常
        # 这里注明了虽然实际上这不是一个表格的问题，但这对本测试来说并不相关。
        with pytest.raises(OSError, match=msg):
            # 调用 read_csv 函数，尝试读取 s3://<bucket_name>/file.csv 文件
            read_csv(f"s3://{s3_private_bucket.name}/file.csv")

    @pytest.mark.xfail(reason="GH#39155 s3fs upgrade", strict=False)
    def test_write_s3_csv_fails(self, tips_df, s3so):
        # GH 32486
        # 尝试向无效的 S3 路径写入数据应该引发异常
        import botocore

        # GH 34087
        # 捕获 ClientError，因为 AWS 服务错误是动态定义的
        error = (FileNotFoundError, botocore.exceptions.ClientError)

        with pytest.raises(error, match="The specified bucket does not exist"):
            # 将 tips_df DataFrame 写入到指定的 S3 路径，测试是否引发预期异常
            tips_df.to_csv(
                "s3://an_s3_bucket_data_doesnt_exit/not_real.csv", storage_options=s3so
            )

    @pytest.mark.xfail(reason="GH#39155 s3fs upgrade", strict=False)
    def test_write_s3_parquet_fails(self, tips_df, s3so):
        # GH 27679
        # 尝试向无效的 S3 路径写入数据应该引发异常
        pytest.importorskip("pyarrow")
        import botocore

        # GH 34087
        # 捕获 ClientError，因为 AWS 服务错误是动态定义的
        error = (FileNotFoundError, botocore.exceptions.ClientError)

        with pytest.raises(error, match="The specified bucket does not exist"):
            # 将 tips_df DataFrame 写入到指定的 S3 路径，测试是否引发预期异常
            tips_df.to_parquet(
                "s3://an_s3_bucket_data_doesnt_exit/not_real.parquet",
                storage_options=s3so,
            )

    @pytest.mark.single_cpu
    def test_read_csv_handles_boto_s3_object(
        self, s3_public_bucket_with_data, tips_file
    ):
        # see gh-16135

        # 获取 S3 桶中的 tips.csv 对象
        s3_object = s3_public_bucket_with_data.Object("tips.csv")

        with BytesIO(s3_object.get()["Body"].read()) as buffer:
            # 使用 read_csv 函数读取缓冲区中的数据，指定编码为 utf8
            result = read_csv(buffer, encoding="utf8")
        # 断言返回的 result 是一个 DataFrame
        assert isinstance(result, DataFrame)
        # 断言 result 不为空
        assert not result.empty

        # 从 tips_file 中读取预期的 DataFrame
        expected = read_csv(tips_file)
        # 使用测试工具比较 result 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single_cpu
    # 测试从 S3 下载 CSV 文件的分块读取功能
    def test_read_csv_chunked_download(self, s3_public_bucket, caplog, s3so):
        # 创建一个包含 100000 行和 4 列的数据框，初始值为零
        df = DataFrame(np.zeros((100000, 4)), columns=list("abcd"))
        # 将数据框转换为 CSV 格式，并编码为 UTF-8，存入字节流
        with BytesIO(df.to_csv().encode("utf-8")) as buf:
            # 将 CSV 数据以对象形式上传到 S3 桶中，键为 "large-file.csv"
            s3_public_bucket.put_object(Key="large-file.csv", Body=buf)
            # 构建 CSV 文件的 S3 URI
            uri = f"{s3_public_bucket.name}/large-file.csv"
            # 创建用于匹配日志消息的正则表达式
            match_re = re.compile(rf"^Fetch: {uri}, 0-(?P<stop>\d+)$")
            # 设置 caplog 记录日志的调试级别为 DEBUG，记录器为 "s3fs"
            with caplog.at_level(logging.DEBUG, logger="s3fs"):
                # 从 S3 中读取 CSV 文件的前 5 行数据
                read_csv(
                    f"s3://{uri}",
                    nrows=5,
                    storage_options=s3so,
                )
                # 遍历 caplog 中的每条消息
                for log in caplog.messages:
                    # 如果消息与匹配正则表达式匹配成功
                    if match := re.match(match_re, log):
                        # 断言截止位置小于 8000000，即文件大小小于 8 MB
                        assert int(match.group("stop")) < 8000000

    # 测试带有键中哈希值的 S3 对象读取功能
    def test_read_s3_with_hash_in_key(self, s3_public_bucket_with_data, tips_df, s3so):
        # 使用 read_csv 函数读取带有哈希值的键的 CSV 文件
        result = read_csv(
            f"s3://{s3_public_bucket_with_data.name}/tips#1.csv", storage_options=s3so
        )
        # 断言读取结果与预期的数据框 tips_df 相等
        tm.assert_frame_equal(tips_df, result)

    # 测试从 S3 中读取 Feather 格式文件的功能
    def test_read_feather_s3_file_path(
        self, s3_public_bucket_with_data, feather_file, s3so
    ):
        # 导入 pytest 中的 pyarrow 模块，如果不存在则跳过测试
        pytest.importorskip("pyarrow")
        # 使用 read_feather 函数读取预期的 Feather 文件
        expected = read_feather(feather_file)
        # 使用 read_feather 函数从 S3 中读取指定路径下的 Feather 文件
        res = read_feather(
            f"s3://{s3_public_bucket_with_data.name}/simple_dataset.feather",
            storage_options=s3so,
        )
        # 断言读取结果与预期的数据框 expected 相等
        tm.assert_frame_equal(expected, res)
```