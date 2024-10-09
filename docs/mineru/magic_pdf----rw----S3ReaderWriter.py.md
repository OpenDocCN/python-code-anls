# `.\MinerU\magic_pdf\rw\S3ReaderWriter.py`

```
# 从魔法 PDF 库导入抽象读写器类
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
# 从公共库导入参数解析和路径连接函数
from magic_pdf.libs.commons import parse_aws_param, parse_bucket_key, join_path
# 导入 boto3 库以与 AWS 进行交互
import boto3
# 导入日志记录库
from loguru import logger
# 导入 botocore 配置
from botocore.config import Config


# 定义 S3 读写器类，继承自抽象读写器类
class S3ReaderWriter(AbsReaderWriter):
    # 初始化方法，接收 AWS 访问密钥和其他配置参数
    def __init__(
        self,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = "auto",
        parent_path: str = "",
    ):
        # 获取 S3 客户端并设置父路径
        self.client = self._get_client(ak, sk, endpoint_url, addressing_style)
        self.path = parent_path

    # 私有方法，用于创建 S3 客户端
    def _get_client(self, ak: str, sk: str, endpoint_url: str, addressing_style: str):
        # 创建 boto3 S3 客户端并配置
        s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=endpoint_url,
            config=Config(
                s3={"addressing_style": addressing_style},
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )
        # 返回创建的 S3 客户端
        return s3_client

    # 读取指定 S3 路径的内容
    def read(self, s3_relative_path, mode=AbsReaderWriter.MODE_TXT, encoding="utf-8"):
        # 检查路径是否以 "s3://" 开头
        if s3_relative_path.startswith("s3://"):
            s3_path = s3_relative_path
        else:
            # 如果不是，则与父路径连接
            s3_path = join_path(self.path, s3_relative_path)
        # 解析桶名称和键
        bucket_name, key = parse_bucket_key(s3_path)
        # 从 S3 获取对象
        res = self.client.get_object(Bucket=bucket_name, Key=key)
        # 读取对象的主体
        body = res["Body"].read()
        # 根据模式解码或返回原始数据
        if mode == AbsReaderWriter.MODE_TXT:
            data = body.decode(encoding)  # 解码字节为文本
        elif mode == AbsReaderWriter.MODE_BIN:
            data = body
        else:
            # 如果模式无效，则抛出异常
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")
        # 返回读取的数据
        return data

    # 写入内容到指定 S3 路径
    def write(self, content, s3_relative_path, mode=AbsReaderWriter.MODE_TXT, encoding="utf-8"):
        # 检查路径是否以 "s3://" 开头
        if s3_relative_path.startswith("s3://"):
            s3_path = s3_relative_path
        else:
            # 如果不是，则与父路径连接
            s3_path = join_path(self.path, s3_relative_path)
        # 根据模式编码内容为字节
        if mode == AbsReaderWriter.MODE_TXT:
            body = content.encode(encoding)  # 将文本数据编码为字节
        elif mode == AbsReaderWriter.MODE_BIN:
            body = content
        else:
            # 如果模式无效，则抛出异常
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")
        # 解析桶名称和键
        bucket_name, key = parse_bucket_key(s3_path)
        # 将对象写入 S3
        self.client.put_object(Body=body, Bucket=bucket_name, Key=key)
        # 记录写入操作的信息
        logger.info(f"内容已写入 {s3_path} ")

    # 以偏移量读取指定 S3 路径的部分内容
    def read_offset(self, path: str, offset=0, limit=None) -> bytes:
        # 检查路径是否以 "s3://" 开头
        if path.startswith("s3://"):
            s3_path = path
        else:
            # 如果不是，则与父路径连接
            s3_path = join_path(self.path, path)
        # 解析桶名称和键
        bucket_name, key = parse_bucket_key(s3_path)

        # 设置字节范围头部
        range_header = (
            f"bytes={offset}-{offset+limit-1}" if limit else f"bytes={offset}-"
        )
        # 从 S3 获取指定范围的对象
        res = self.client.get_object(Bucket=bucket_name, Key=key, Range=range_header)
        # 返回读取的字节数据
        return res["Body"].read()


# 如果该脚本是主程序，则执行以下代码
if __name__ == "__main__":
    # 判断条件，当前为假，不执行内部代码块
    if 0:
        # 配置连接信息的注释
        # AK（Access Key）为空字符串
        ak = ""
        # SK（Secret Key）为空字符串
        sk = ""
        # 存储端点的 URL 为空字符串
        endpoint_url = ""
        # 地址风格设置为自动
        addressing_style = "auto"
        # 存储桶名称为空字符串
        bucket_name = ""
        # 创建一个 S3ReaderWriter 对象，初始化连接信息
        s3_reader_writer = S3ReaderWriter(
            ak, sk, endpoint_url, addressing_style, "s3://bucket_name/"
        )

        # 定义要写入 S3 的文本数据
        text_data = "This is some text data"
        # 将文本数据写入 S3，指定相对路径和模式
        s3_reader_writer.write(
            text_data,
            s3_relative_path=f"s3://{bucket_name}/ebook/test/test.json",
            mode=AbsReaderWriter.MODE_TXT,
        )

        # 从 S3 读取文本数据
        text_data_read = s3_reader_writer.read(
            s3_relative_path=f"s3://{bucket_name}/ebook/test/test.json", mode=AbsReaderWriter.MODE_TXT
        )
        # 记录读取的文本数据
        logger.info(f"Read text data from S3: {text_data_read}")
        # 定义要写入 S3 的二进制数据
        binary_data = b"This is some binary data"
        # 将二进制数据写入 S3，指定相对路径和模式
        s3_reader_writer.write(
            text_data,
            s3_relative_path=f"s3://{bucket_name}/ebook/test/test.json",
            mode=AbsReaderWriter.MODE_BIN,
        )

        # 从 S3 读取二进制数据
        binary_data_read = s3_reader_writer.read(
            s3_relative_path=f"s3://{bucket_name}/ebook/test/test.json", mode=AbsReaderWriter.MODE_BIN
        )
        # 记录读取的二进制数据
        logger.info(f"Read binary data from S3: {binary_data_read}")

        # 从 S3 按范围读取文本数据
        binary_data_read = s3_reader_writer.read_offset(
            path=f"s3://{bucket_name}/ebook/test/test.json", offset=0, limit=10
        )
        # 记录按范围读取的二进制数据
        logger.info(f"Read binary data from S3: {binary_data_read}")
    # 判断条件，当前为真，执行内部代码块
    if 1:
        # 导入操作系统和 JSON 模块
        import os
        import json

        # 从环境变量获取 AK（Access Key），如果不存在则为默认空字符串
        ak = os.getenv("AK", "")
        # 从环境变量获取 SK（Secret Key），如果不存在则为默认空字符串
        sk = os.getenv("SK", "")
        # 从环境变量获取端点 URL，如果不存在则为默认空字符串
        endpoint_url = os.getenv("ENDPOINT", "")
        # 从环境变量获取存储桶名称，如果不存在则为默认空字符串
        bucket = os.getenv("S3_BUCKET", "")
        # 从环境变量获取前缀，如果不存在则为默认空字符串
        prefix = os.getenv("S3_PREFIX", "")
        # 从环境变量获取键的基本名称，如果不存在则为默认空字符串
        key_basename = os.getenv("S3_KEY_BASENAME", "")
        # 创建一个 S3ReaderWriter 对象，初始化连接信息
        s3_reader_writer = S3ReaderWriter(
            ak, sk, endpoint_url, "auto", f"s3://{bucket}/{prefix}"
        )
        # 从 S3 按偏移量读取内容
        content_bin = s3_reader_writer.read_offset(key_basename)
        # 断言读取内容的前十个字节符合预期
        assert content_bin[:10] == b'{"track_id'
        # 断言读取内容的最后十个字节符合预期
        assert content_bin[-10:] == b'r":null}}\n'

        # 从 S3 按偏移量和限制读取内容
        content_bin = s3_reader_writer.read_offset(key_basename, offset=424, limit=426)
        # 将读取的内容转换为 JSON 格式
        jso = json.dumps(content_bin.decode("utf-8"))
        # 打印 JSON 格式的内容
        print(jso)
```