# `.\MinerU\magic_pdf\libs\commons.py`

```
# 导入datetime模块，用于处理时间
import datetime
# 导入json模块，用于处理JSON数据
import json
# 导入os模块，提供与操作系统交互的功能
import os, re, configparser
# 导入subprocess模块，用于执行外部命令
import subprocess
# 导入time模块，用于时间相关的操作
import time

# 导入boto3库，AWS的Python SDK
import boto3
# 导入loguru库，用于日志记录
from loguru import logger
# 从boto3的s3模块导入TransferConfig类，处理S3传输配置
from boto3.s3.transfer import TransferConfig
# 从botocore导入Config类，用于配置AWS客户端
from botocore.config import Config

# 导入fitz库，处理PDF文件
import fitz # 1.23.9中已经切换到rebase
# import fitz_old as fitz  # 使用1.23.9之前的pymupdf库


# 定义获取时间差的函数，输入为时间戳
def get_delta_time(input_time):
    # 计算当前时间与输入时间的差值，并四舍五入到小数点后两位
    return round(time.time() - input_time, 2)


# 定义连接路径的函数，支持多个路径参数
def join_path(*args):
    # 将路径参数连接成一个字符串，去除每个部分末尾的斜杠
    return '/'.join(str(s).rstrip('/') for s in args)


# 配置全局的错误日志路径，方便在多个地方引用
error_log_path = "s3://llm-pdf-text/err_logs/"
# json_dump_path用于临时本地测试，不应提交到主分支
json_dump_path = "s3://llm-pdf-text/json_dump/"

# s3_image_save_path不应在基础库中定义，应在业务代码中定义


# 定义获取列表中前百分之多少元素的函数
def get_top_percent_list(num_list, percent):
    """
    获取列表中前百分之多少的元素
    :param num_list:
    :param percent:
    :return:
    """
    # 如果列表为空，返回空列表
    if len(num_list) == 0:
        top_percent_list = []
    else:
        # 将输入列表按降序排序
        sorted_imgs_len_list = sorted(num_list, reverse=True)
        # 计算前percent的索引
        top_percent_index = int(len(sorted_imgs_len_list) * percent)
        # 获取排序后的前percent元素
        top_percent_list = sorted_imgs_len_list[:top_percent_index]
    # 返回前percent的元素列表
    return top_percent_list


# 定义格式化时间的函数，输入为时间戳
def formatted_time(time_stamp):
    # 将时间戳转换为datetime对象
    dt_object = datetime.datetime.fromtimestamp(time_stamp)
    # 将datetime对象格式化为字符串
    output_time = dt_object.strftime("%Y-%m-%d-%H:%M:%S")
    # 返回格式化后的时间字符串
    return output_time


# 定义获取列表最大值的函数
def mymax(alist: list):
    # 如果列表为空，返回0
    if len(alist) == 0:
        return 0  # 空是0， 0*0也是0大小q
    else:
        # 返回列表中的最大值
        return max(alist)

# 定义解析AWS配置参数的函数，输入为profile
def parse_aws_param(profile):
    # 如果profile是字符串类型
    if isinstance(profile, str):
        # 解析配置文件路径
        config_file = join_path(os.path.expanduser("~"), ".aws", "config")
        credentials_file = join_path(os.path.expanduser("~"), ".aws", "credentials")
        # 创建ConfigParser对象
        config = configparser.ConfigParser()
        # 读取凭证文件和配置文件
        config.read(credentials_file)
        config.read(config_file)
        # 获取AWS账户相关信息
        ak = config.get(profile, "aws_access_key_id")
        sk = config.get(profile, "aws_secret_access_key")
        # 获取S3相关的配置
        if profile == "default":
            s3_str = config.get(f"{profile}", "s3")
        else:
            s3_str = config.get(f"profile {profile}", "s3")
        # 正则匹配endpoint_url
        end_match = re.search("endpoint_url[\s]*=[\s]*([^\s\n]+)[\s\n]*$", s3_str, re.MULTILINE)
        if end_match:
            endpoint = end_match.group(1)
        else:
            raise ValueError(f"aws 配置文件中没有找到 endpoint_url")
        # 正则匹配addressing_style
        style_match = re.search("addressing_style[\s]*=[\s]*([^\s\n]+)[\s\n]*$", s3_str, re.MULTILINE)
        if style_match:
            addressing_style = style_match.group(1)
        else:
            addressing_style = "path"
    # 如果profile是字典类型
    elif isinstance(profile, dict):
        ak = profile["ak"]
        sk = profile["sk"]
        endpoint = profile["endpoint"]
        addressing_style = "auto"

    # 返回访问密钥、秘密密钥、endpoint和地址样式
    return ak, sk, endpoint, addressing_style


# 定义解析S3路径的函数，输入为完整的S3路径
def parse_bucket_key(s3_full_path: str):
    """
    输入 s3://bucket/path/to/my/file.txt
    输出 bucket, path/to/my/file.txt
    """
    # 去除 S3 路径两端的空白字符
        s3_full_path = s3_full_path.strip()
        # 检查路径是否以 "s3://" 开头，如果是则移除前缀
        if s3_full_path.startswith("s3://"):
            s3_full_path = s3_full_path[5:]
        # 检查路径是否以 "/" 开头，如果是则移除前导斜杠
        if s3_full_path.startswith("/"):
            s3_full_path = s3_full_path[1:]
        # 按照第一个 "/" 分割路径，得到桶名和对象键
        bucket, key = s3_full_path.split("/", 1)
        # 返回桶名和对象键
        return bucket, key
# 定义函数，读取 PDF 文件，支持 S3 存储和本地文件
def read_file(pdf_path: str, s3_profile):
    # 检查路径是否以 "s3://" 开头，判断是否为 S3 文件
    if pdf_path.startswith("s3://"):
        # 解析 AWS 参数以获取访问密钥和端点
        ak, sk, end_point, addressing_style = parse_aws_param(s3_profile)
        # 创建 S3 客户端，配置访问参数
        cli = boto3.client(service_name="s3", aws_access_key_id=ak, aws_secret_access_key=sk, endpoint_url=end_point,
                           config=Config(s3={'addressing_style': addressing_style}, retries={'max_attempts': 10, 'mode': 'standard'}))
        # 解析 S3 路径以获取桶名称和对象键
        bucket_name, bucket_key = parse_bucket_key(pdf_path)
        # 从 S3 获取对象并读取其内容
        res = cli.get_object(Bucket=bucket_name, Key=bucket_key)
        file_content = res["Body"].read()
        # 返回文件内容
        return file_content
    else:
        # 如果不是 S3 路径，打开本地 PDF 文件并读取其内容
        with open(pdf_path, "rb") as f:
            return f.read()


# 定义函数，获取指定页面的 DOCX 模型输出
def get_docx_model_output(pdf_model_output, page_id):
    # 从 PDF 模型输出中获取指定页面的 JSON 数据
    model_output_json = pdf_model_output[page_id]
    # 返回该页面的 JSON 数据
    return model_output_json


# 定义函数，列出指定目录下的所有文件，支持 S3 和本地目录
def list_dir(dir_path:str, s3_profile:str):
    """
    列出dir_path下的所有文件
    """
    # 初始化一个列表，用于存储结果
    ret = []
    
    # 检查目录路径是否以 "s3" 开头，判断是否为 S3 路径
    if dir_path.startswith("s3"):
        # 解析 AWS 参数以获取访问密钥和端点
        ak, sk, end_point, addressing_style = parse_aws_param(s3_profile)
        # 使用正则表达式从 S3 路径中提取桶名称和路径
        s3info = re.findall(r"s3:\/\/([^\/]+)\/(.*)", dir_path)
        bucket, path = s3info[0][0], s3info[0][1]
        try:
            # 创建 S3 客户端，配置访问参数
            cli = boto3.client(service_name="s3", aws_access_key_id=ak, aws_secret_access_key=sk, endpoint_url=end_point,
                                            config=Config(s3={'addressing_style': addressing_style}))
            # 定义内部生成器函数，用于分批列出 S3 对象
            def list_obj_scluster():
                marker = None
                while True:
                    # 设置列出对象的参数
                    list_kwargs = dict(MaxKeys=1000, Bucket=bucket, Prefix=path)
                    if marker:
                        list_kwargs['Marker'] = marker
                    # 获取 S3 对象列表
                    response = cli.list_objects(**list_kwargs)
                    contents = response.get("Contents", [])
                    yield from contents
                    # 检查是否还有更多对象可列出
                    if not response.get("IsTruncated") or len(contents)==0:
                        break
                    marker = contents[-1]['Key']

            # 遍历 S3 对象，筛选出 JSON 文件并添加到结果列表
            for info in list_obj_scluster():
                file_path = info['Key']
                #size = info['Size']

                if path!="":
                    afile = file_path[len(path):]
                    if afile.endswith(".json"):
                        ret.append(f"s3://{bucket}/{file_path}")
                        
            # 返回结果列表
            return ret

        except Exception as e:
            # 记录异常并退出
            logger.exception(e)
            exit(-1)
    else: # 如果是本地目录，遍历本地目录以获取所有 JSON 文件
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".json"):
                    # 添加符合条件的文件路径到结果列表
                    ret.append(join_path(root, file))
        # 对结果列表进行排序
        ret.sort()
        # 返回结果列表
        return ret

# 定义函数，获取 S3 客户端，保存图像
def get_img_s3_client(save_path:str, image_s3_config:str):
    """
    """
    # 检查保存路径是否以 "s3://" 开头，以决定是否创建 S3 客户端
    if save_path.startswith("s3://"):  # 放这里是为了最少创建一个s3 client
        # 解析 AWS 参数，包括访问密钥、秘密密钥、端点和地址样式
        ak, sk, end_point, addressing_style = parse_aws_param(image_s3_config)
        # 创建 S3 客户端，使用解析出的 AWS 参数
        img_s3_client = boto3.client(
            service_name="s3",  # 指定服务名称为 S3
            aws_access_key_id=ak,  # 设置 AWS 访问密钥
            aws_secret_access_key=sk,  # 设置 AWS 秘密密钥
            endpoint_url=end_point,  # 设置 S3 端点 URL
            # 配置 S3 的地址样式和重试策略
            config=Config(s3={"addressing_style": addressing_style}, retries={'max_attempts': 5, 'mode': 'standard'}),
        )
    else:
        # 如果保存路径不是 S3，则将 S3 客户端设置为 None
        img_s3_client = None
        
    # 返回创建的 S3 客户端或 None
    return img_s3_client
# 当此脚本被直接运行时执行以下代码
if __name__=="__main__":
    # 定义一个 S3 路径，指向特定 PDF 文件存储位置
    s3_path = "s3://llm-pdf-text/layout_det/scihub/scimag07865000-07865999/10.1007/s10729-011-9175-6.pdf/"
    # 设置 S3 配置文件名，可能用于身份验证
    s3_profile = "langchao"
    # 调用 list_dir 函数，获取指定 S3 路径下的目录列表
    ret = list_dir(s3_path, s3_profile)
    # 打印返回的目录列表
    print(ret)
```