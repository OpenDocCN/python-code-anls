# `D:\src\scipysrc\pandas\pandas\tests\io\test_s3.py`

```
# 引入 BytesIO 模块，用于处理字节流
from io import BytesIO

# 引入 pytest 模块，用于单元测试
import pytest

# 引入 pandas 中的 read_csv 函数，用于读取 CSV 文件
from pandas import read_csv


# 测试函数：测试从 S3 对象存储流式读取对象
def test_streaming_s3_objects():
    # GH17135
    # botocore 在 1.10.47 版本中增加了迭代支持，现在可以在 read_* 中使用
    pytest.importorskip("botocore", minversion="1.10.47")
    # 从 botocore.response 模块引入 StreamingBody 对象
    from botocore.response import StreamingBody

    # 示例数据，包含两个字节流对象
    data = [b"foo,bar,baz\n1,2,3\n4,5,6\n", b"just,the,header\n"]
    # 遍历数据列表
    for el in data:
        # 使用 BytesIO 创建 StreamingBody 对象，指定内容长度
        body = StreamingBody(BytesIO(el), content_length=len(el))
        # 调用 read_csv 函数读取 StreamingBody 中的内容
        read_csv(body)


# 测试函数：从公共 S3 存储桶读取未使用凭证的数据
@pytest.mark.single_cpu
def test_read_without_creds_from_pub_bucket(s3_public_bucket_with_data, s3so):
    # GH 34626
    # 确保 s3fs 模块已导入
    pytest.importorskip("s3fs")
    # 使用 read_csv 函数读取公共 S3 存储桶中的 tips.csv 文件，指定读取行数和存储选项
    result = read_csv(
        f"s3://{s3_public_bucket_with_data.name}/tips.csv",
        nrows=3,
        storage_options=s3so,
    )
    # 断言读取结果的行数为 3
    assert len(result) == 3


# 测试函数：从公共 S3 存储桶读取使用凭证的数据
@pytest.mark.single_cpu
def test_read_with_creds_from_pub_bucket(s3_public_bucket_with_data, s3so):
    # Ensure we can read from a public bucket with credentials
    # GH 34626
    # 确保 s3fs 模块已导入
    pytest.importorskip("s3fs")
    # 使用 read_csv 函数读取公共 S3 存储桶中的 tips.csv 文件，指定读取行数、无标题行和存储选项
    df = read_csv(
        f"s3://{s3_public_bucket_with_data.name}/tips.csv",
        nrows=5,
        header=None,
        storage_options=s3so,
    )
    # 断言读取结果的行数为 5
    assert len(df) == 5
```