# `.\MinerU\magic_pdf\libs\config_reader.py`

```
# 根据bucket的名字返回对应的s3 AK， SK，endpoint三元组

import json  # 导入json模块，用于处理JSON数据
import os  # 导入os模块，用于操作文件和目录

from loguru import logger  # 导入logger用于记录日志

from magic_pdf.libs.commons import parse_bucket_key  # 从commons模块导入parse_bucket_key函数

# 定义配置文件名常量
CONFIG_FILE_NAME = "magic-pdf.json"  # 配置文件的名称常量


def read_config():
    home_dir = os.path.expanduser("~")  # 获取当前用户的主目录

    config_file = os.path.join(home_dir, CONFIG_FILE_NAME)  # 构建配置文件的完整路径

    if not os.path.exists(config_file):  # 检查配置文件是否存在
        raise FileNotFoundError(f"{config_file} not found")  # 如果不存在，抛出文件未找到异常

    with open(config_file, "r", encoding="utf-8") as f:  # 以只读模式打开配置文件
        config = json.load(f)  # 读取文件内容并解析为JSON格式
    return config  # 返回配置字典


def get_s3_config(bucket_name: str):
    """
    ~/magic-pdf.json 读出来
    """
    config = read_config()  # 调用read_config函数读取配置

    bucket_info = config.get("bucket_info")  # 从配置中获取bucket_info
    if bucket_name not in bucket_info:  # 如果bucket_name不在bucket_info中
        access_key, secret_key, storage_endpoint = bucket_info["[default]"]  # 使用默认的AK、SK和endpoint
    else:  # 如果存在指定的bucket_name
        access_key, secret_key, storage_endpoint = bucket_info[bucket_name]  # 获取对应的AK、SK和endpoint

    if access_key is None or secret_key is None or storage_endpoint is None:  # 检查是否有缺失的凭证
        raise Exception(f"ak, sk or endpoint not found in {CONFIG_FILE_NAME}")  # 抛出异常

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")  # 记录获取的配置信息

    return access_key, secret_key, storage_endpoint  # 返回AK、SK和endpoint


def get_s3_config_dict(path: str):
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))  # 获取bucket对应的配置
    return {"ak": access_key, "sk": secret_key, "endpoint": storage_endpoint}  # 返回包含AK、SK和endpoint的字典


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)  # 解析path获取bucket和key
    return bucket  # 返回bucket名称


def get_local_models_dir():
    config = read_config()  # 读取配置
    models_dir = config.get("models-dir")  # 获取模型目录配置
    if models_dir is None:  # 如果未找到模型目录配置
        logger.warning(f"'models-dir' not found in {CONFIG_FILE_NAME}, use '/tmp/models' as default")  # 记录警告
        return "/tmp/models"  # 返回默认模型目录
    else:  # 如果找到配置
        return models_dir  # 返回模型目录


def get_device():
    config = read_config()  # 读取配置
    device = config.get("device-mode")  # 获取设备模式配置
    if device is None:  # 如果未找到设备模式配置
        logger.warning(f"'device-mode' not found in {CONFIG_FILE_NAME}, use 'cpu' as default")  # 记录警告
        return "cpu"  # 返回默认设备模式
    else:  # 如果找到配置
        return device  # 返回设备模式


def get_table_recog_config():
    config = read_config()  # 读取配置
    table_config = config.get("table-config")  # 获取表格识别配置
    if table_config is None:  # 如果未找到表格识别配置
        logger.warning(f"'table-config' not found in {CONFIG_FILE_NAME}, use 'False' as default")  # 记录警告
        return json.loads('{"is_table_recog_enable": false, "max_time": 400}')  # 返回默认配置
    else:  # 如果找到配置
        return table_config  # 返回表格识别配置


if __name__ == "__main__":  # 如果脚本是主程序运行
    ak, sk, endpoint = get_s3_config("llm-raw")  # 获取名为"llm-raw"的bucket的AK、SK和endpoint
```