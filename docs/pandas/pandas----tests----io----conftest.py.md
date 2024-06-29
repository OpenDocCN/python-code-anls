# `D:\src\scipysrc\pandas\pandas\tests\io\conftest.py`

```
# 导入必要的模块和库
import shlex  # 提供了一个用于解析 shell 命令的模块
import subprocess  # 提供了一个用于执行外部命令的模块
import time  # 提供时间相关的功能
import uuid  # 提供生成唯一标识符的功能

import pytest  # 导入 pytest 测试框架

# 从 pandas.compat 模块中导入一些平台相关的函数和变量
from pandas.compat import (
    is_ci_environment,  # 检测是否在持续集成环境中运行
    is_platform_arm,  # 检测是否在 ARM 平台上运行
    is_platform_mac,  # 检测是否在 macOS 平台上运行
    is_platform_windows,  # 检测是否在 Windows 平台上运行
)

# 导入 pandas.util._test_decorators 模块，并命名为 td
import pandas.util._test_decorators as td

# 导入 pandas.io.common 模块，并命名为 icom
import pandas.io.common as icom

# 从 pandas.io.parsers 模块中导入 read_csv 函数
from pandas.io.parsers import read_csv


@pytest.fixture
def compression_to_extension():
    # 创建一个字典，将扩展名映射到压缩格式的反向映射
    return {value: key for key, value in icom.extension_to_compression.items()}


@pytest.fixture
def tips_file(datapath):
    """Path to the tips dataset"""
    # 返回 tips 数据集的路径，使用了 datapath 函数
    return datapath("io", "data", "csv", "tips.csv")


@pytest.fixture
def jsonl_file(datapath):
    """Path to a JSONL dataset"""
    # 返回一个 JSONL 数据集的路径，使用了 datapath 函数
    return datapath("io", "parser", "data", "items.jsonl")


@pytest.fixture
def salaries_table(datapath):
    """DataFrame with the salaries dataset"""
    # 返回一个包含薪资数据集的 DataFrame，使用了 read_csv 函数，并指定了分隔符
    return read_csv(datapath("io", "parser", "data", "salaries.csv"), sep="\t")


@pytest.fixture
def feather_file(datapath):
    # 返回 feather 数据文件的路径，使用了 datapath 函数
    return datapath("io", "data", "feather", "feather-0_3_1.feather")


@pytest.fixture
def xml_file(datapath):
    # 返回 XML 文件的路径，使用了 datapath 函数
    return datapath("io", "data", "xml", "books.xml")


@pytest.fixture
def s3_base(worker_id, monkeypatch):
    """
    Fixture for mocking S3 interaction.

    Sets up moto server in separate process locally
    Return url for motoserver/moto CI service
    """
    # 确保 s3fs 和 boto3 模块可用，否则跳过测试
    pytest.importorskip("s3fs")
    pytest.importorskip("boto3")

    # 设置环境变量模拟 AWS 访问密钥
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "foobar_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "foobar_secret")

    # 如果在持续集成环境中运行，且不在 Windows/macOS/ARM 平台上，则跳过测试
    if is_ci_environment():
        if is_platform_arm() or is_platform_mac() or is_platform_windows():
            # 在 Windows/macOS/ARM 平台上跳过 S3 测试
            pytest.skip(
                "S3 tests do not have a corresponding service in "
                "Windows, macOS or ARM platforms"
            )
        else:
            # 返回本地 Moto 服务器的 URL，用于 CI 服务
            yield "http://localhost:5000"
    else:
        # 导入 requests 库，如果导入失败则跳过测试
        requests = pytest.importorskip("requests")
        # 导入 moto 库，如果导入失败则跳过测试
        pytest.importorskip("moto")
        # 导入 flask 库，服务器模式需要 flask
        pytest.importorskip("flask")  # server mode needs flask too

        # 在服务器模式下启动 moto，即作为一个独立进程
        # 在 localhost 上使用 S3 端点

        # 根据 worker_id 调整 worker_id，如果是 "master" 则设为 "5"，否则移除开头的 "gw"
        worker_id = "5" if worker_id == "master" else worker_id.lstrip("gw")
        # 根据 worker_id 构建端口号
        endpoint_port = f"555{worker_id}"
        # 构建 S3 端点的 URI
        endpoint_uri = f"http://127.0.0.1:{endpoint_port}/"

        # 将子进程的标准输出和标准错误都重定向到空设备，避免在终端输出日志
        with subprocess.Popen(
            shlex.split(f"moto_server s3 -p {endpoint_port}"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as proc:
            timeout = 5
            while timeout > 0:
                try:
                    # 当服务器接受连接后即可继续
                    r = requests.get(endpoint_uri)
                    if r.ok:
                        break
                except Exception:
                    pass
                # 每次循环减少超时时间，然后休眠 0.1 秒
                timeout -= 0.1
                time.sleep(0.1)
            # 返回 S3 端点的 URI
            yield endpoint_uri

            # 终止子进程
            proc.terminate()
@pytest.fixture
def s3so(s3_base):
    """
    Fixture for creating S3 client configuration.

    Returns a dictionary with client_kwargs containing endpoint_url set to s3_base.
    """
    return {"client_kwargs": {"endpoint_url": s3_base}}


@pytest.fixture
def s3_resource(s3_base):
    """
    Fixture for creating an S3 resource object.

    Uses boto3 to create an S3 resource object with endpoint_url set to s3_base.
    """
    import boto3

    s3 = boto3.resource("s3", endpoint_url=s3_base)
    return s3


@pytest.fixture
def s3_public_bucket(s3_resource):
    """
    Fixture for creating a public S3 bucket.

    Creates a new S3 bucket with a unique name derived from uuid.uuid4().
    Deletes the bucket and its contents after yielding.
    """
    bucket = s3_resource.Bucket(f"pandas-test-{uuid.uuid4()}")
    bucket.create()
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture
def s3_public_bucket_with_data(
    s3_public_bucket, tips_file, jsonl_file, feather_file, xml_file
):
    """
    Fixture for creating a public S3 bucket and loading test data into it.

    Loads multiple files into the bucket using put_object method.
    """
    test_s3_files = [
        ("tips#1.csv", tips_file),
        ("tips.csv", tips_file),
        ("tips.csv.gz", tips_file + ".gz"),
        ("tips.csv.bz2", tips_file + ".bz2"),
        ("items.jsonl", jsonl_file),
        ("simple_dataset.feather", feather_file),
        ("books.xml", xml_file),
    ]
    for s3_key, file_name in test_s3_files:
        with open(file_name, "rb") as f:
            s3_public_bucket.put_object(Key=s3_key, Body=f)
    return s3_public_bucket


@pytest.fixture
def s3_private_bucket(s3_resource):
    """
    Fixture for creating a private S3 bucket.

    Creates a new S3 bucket with a unique name derived from uuid.uuid4(),
    with ACL set to 'private'.
    Deletes the bucket and its contents after yielding.
    """
    bucket = s3_resource.Bucket(f"cant_get_it-{uuid.uuid4()}")
    bucket.create(ACL="private")
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture
def s3_private_bucket_with_data(
    s3_private_bucket, tips_file, jsonl_file, feather_file, xml_file
):
    """
    Fixture for creating a private S3 bucket and loading test data into it.

    Loads multiple files into the bucket using put_object method.
    """
    test_s3_files = [
        ("tips#1.csv", tips_file),
        ("tips.csv", tips_file),
        ("tips.csv.gz", tips_file + ".gz"),
        ("tips.csv.bz2", tips_file + ".bz2"),
        ("items.jsonl", jsonl_file),
        ("simple_dataset.feather", feather_file),
        ("books.xml", xml_file),
    ]
    for s3_key, file_name in test_s3_files:
        with open(file_name, "rb") as f:
            s3_private_bucket.put_object(Key=s3_key, Body=f)
    return s3_private_bucket


_compression_formats_params = [
    (".no_compress", None),
    ("", None),
    (".gz", "gzip"),
    (".GZ", "gzip"),
    (".bz2", "bz2"),
    (".BZ2", "bz2"),
    (".zip", "zip"),
    (".ZIP", "zip"),
    (".xz", "xz"),
    (".XZ", "xz"),
    pytest.param((".zst", "zstd"), marks=td.skip_if_no("zstandard")),
    pytest.param((".ZST", "zstd"), marks=td.skip_if_no("zstandard")),
]


@pytest.fixture(params=_compression_formats_params[1:])
def compression_format(request):
    """
    Fixture for providing different compression formats.

    Returns a tuple with file extension and compression type, skipping the first entry.
    """
    return request.param


@pytest.fixture(params=_compression_formats_params)
def compression_ext(request):
    """
    Fixture for providing different compression extensions.

    Returns only the file extension part of the tuple.
    """
    return request.param[0]


@pytest.fixture(
    params=[
        "python",
        pytest.param("pyarrow", marks=td.skip_if_no("pyarrow")),
    ]
)
def string_storage(request):
    """
    Parametrized fixture for pd.options.mode.string_storage.

    Provides different string storage options for pandas.

    - 'python'
    - 'pyarrow' (conditionally, depending on the availability of 'pyarrow')
    """
    return request.param
```