# `.\MinerU\tests\test_commons.py`

```
# 导入 io 模块，用于处理字节流
import io
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 os 模块，用于与操作系统交互
import os

# 导入 boto3 库，用于与 AWS S3 进行交互
import boto3
# 从 botocore.config 导入 Config 类，用于配置 S3 客户端
from botocore.config import Config

# 从魔法 PDF 库中导入 fitz，用于处理 PDF 文件
from magic_pdf.libs.commons import fitz
# 从配置读取器中导入获取 S3 配置字典的函数
from magic_pdf.libs.config_reader import get_s3_config_dict

# 从魔法 PDF 库中导入各种工具函数
from magic_pdf.libs.commons import join_path, json_dump_path, read_file, parse_bucket_key
# 导入日志记录工具 loguru
from loguru import logger

# 定义测试 PDF 文件的 S3 存储路径
test_pdf_dir_path = "s3://llm-pdf-text/unittest/pdf/"

# 定义获取测试 PDF JSON 数据的函数，接受书名作为参数
def get_test_pdf_json(book_name):
    # 生成 JSON 文件的路径
    json_path = join_path(json_dump_path, book_name + ".json")
    # 获取 S3 配置字典
    s3_config = get_s3_config_dict(json_path)
    # 读取 JSON 文件内容
    file_content = read_file(json_path, s3_config)
    # 将文件内容解码为 UTF-8 字符串
    json_str = file_content.decode('utf-8')
    # 将 JSON 字符串加载为 Python 对象
    json_object = json.loads(json_str)
    # 返回 JSON 对象
    return json_object

# 定义读取测试文件的函数，接受书名作为参数
def read_test_file(book_name):
    # 生成测试 PDF 文件的路径
    test_pdf_path = join_path(test_pdf_dir_path, book_name + ".pdf")
    # 获取 S3 配置字典
    s3_config = get_s3_config_dict(test_pdf_path)
    try:
        # 尝试读取测试 PDF 文件内容
        file_content = read_file(test_pdf_path, s3_config)
        # 返回文件内容
        return file_content
    except Exception as e:
        # 如果捕获到文件未找到的异常
        if "NoSuchKey" in str(e):
            # 记录警告日志，表示文件未找到，将从原始 S3 PDF 路径下载
            logger.warning("File not found in test_pdf_path. Downloading from orig_s3_pdf_path.")
            try:
                # 获取与书名相关的 JSON 数据
                json_object = get_test_pdf_json(book_name)
                # 从 JSON 对象中获取原始 S3 PDF 路径
                orig_s3_pdf_path = json_object.get('file_location')
                # 获取原始 PDF 文件的 S3 配置字典
                s3_config = get_s3_config_dict(orig_s3_pdf_path)
                # 读取原始 PDF 文件内容
                file_content = read_file(orig_s3_pdf_path, s3_config)
                # 获取 S3 客户端
                s3_client = get_s3_client(test_pdf_path)
                # 解析测试 PDF 路径中的桶名和键
                bucket_name, bucket_key = parse_bucket_key(test_pdf_path)
                # 将文件内容封装为字节流对象
                file_obj = io.BytesIO(file_content)
                # 将字节流对象上传到指定的 S3 桶和键
                s3_client.upload_fileobj(file_obj, bucket_name, bucket_key)
                # 返回文件内容
                return file_content
            except Exception as e:
                # 记录异常日志
                logger.exception(e)
        else:
            # 记录异常日志
            logger.exception(e)

# 定义从测试 PDF 获取文档的函数，接受书名作为参数
def get_docs_from_test_pdf(book_name):
    # 读取测试 PDF 文件内容
    file_content = read_test_file(book_name)
    # 使用 fitz 打开 PDF 文件并返回文档对象
    return fitz.open("pdf", file_content)

# 定义获取测试 JSON 数据的函数，接受目录路径和 JSON 文件名作为参数
def get_test_json_data(directory_path, json_file_name):
    # 打开指定路径的 JSON 文件进行读取
    with open(os.path.join(directory_path, json_file_name), "r", encoding='utf-8') as f:
        # 将 JSON 文件内容加载为 Python 对象
        test_data = json.load(f)
    # 返回测试数据
    return test_data

# 定义获取 S3 客户端的函数，接受路径作为参数
def get_s3_client(path):
    # 获取 S3 配置字典
    s3_config = get_s3_config_dict(path)
    try:
        # 尝试创建 S3 客户端并设置参数
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8, "mode": "standard"}),
        )
    except:
        # 如果旧版本 boto3 不支持 retries.mode 参数，则创建不带该参数的 S3 客户端
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8}),
        )
```