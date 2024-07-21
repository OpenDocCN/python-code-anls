# `.\pytorch\tools\linter\adapters\update_s3.py`

```py
"""Uploads a new binary to s3 and updates its hash in the config file.

You'll need to have appropriate credentials on the PyTorch AWS buckets, see:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
for how to configure them.
"""

# 引入所需的模块和库
import argparse  # 用于解析命令行参数
import hashlib  # 用于计算文件的哈希值
import json  # 用于处理 JSON 格式的配置文件
import logging  # 用于日志记录
import os  # 用于操作系统相关功能

import boto3  # 引入 AWS 的 boto3 客户端库


def compute_file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file and return it as a hex string."""
    # 如果文件不存在，则返回空字符串
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha256()

    # 以二进制模式打开文件并计算其哈希值
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)

    # 将哈希值以十六进制字符串形式返回
    return hash.hexdigest()


def main() -> None:
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="s3 binary updater",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config-json",
        required=True,
        help="path to config json that you are trying to update",
    )
    parser.add_argument(
        "--linter",
        required=True,
        help="name of linter you're trying to update",
    )
    parser.add_argument(
        "--platform",
        required=True,
        help="which platform you are uploading the binary for",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="file to upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="if set, don't actually upload/write hash",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)  # 配置日志记录器，设置记录日志的级别为 INFO

    config = json.load(open(args.config_json))  # 加载配置文件
    linter_config = config[args.linter][args.platform]  # 获取特定 linter 和 platform 的配置信息
    bucket = linter_config["s3_bucket"]  # 获取 S3 存储桶名称
    object_name = linter_config["object_name"]  # 获取对象名称

    # 上传文件到 S3 存储桶
    logging.info(
        "Uploading file %s to s3 bucket: %s, object name: %s",
        args.file,
        bucket,
        object_name,
    )
    if not args.dry_run:
        s3_client = boto3.client("s3")  # 创建 S3 客户端
        s3_client.upload_file(args.file, bucket, object_name)  # 上传文件到指定 S3 存储桶

    # 计算新上传二进制文件的哈希值
    hash_of_new_binary = compute_file_sha256(args.file)
    logging.info("Computed new hash for binary %s", hash_of_new_binary)

    linter_config["hash"] = hash_of_new_binary  # 更新配置文件中的哈希值字段
    config_dump = json.dumps(config, indent=4, sort_keys=True)  # 将更新后的配置文件转换为 JSON 格式的字符串

    logging.info("Writing out new config:")  # 记录写入新配置文件的操作
    logging.info(config_dump)  # 打印新配置文件的内容
    if not args.dry_run:
        with open(args.config_json, "w") as f:
            f.write(config_dump)  # 将新配置文件写入磁盘


if __name__ == "__main__":
    main()
```