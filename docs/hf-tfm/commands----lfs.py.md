# `.\transformers\commands\lfs.py`

```py
"""
Implementation of a custom transfer agent for the transfer type "multipart" for git-lfs.

Inspired by: github.com/cbartz/git-lfs-swift-transfer-agent/blob/master/git_lfs_swift_transfer.py

Spec is: github.com/git-lfs/git-lfs/blob/master/docs/custom-transfers.md


To launch debugger while developing:

``` [lfs "customtransfer.multipart"]
path = /path/to/transformers/.env/bin/python args = -m debugpy --listen 5678 --wait-for-client
/path/to/transformers/src/transformers/commands/transformers_cli.py lfs-multipart-upload ```"""

import json
import os
import subprocess
import sys
import warnings
from argparse import ArgumentParser
from contextlib import AbstractContextManager
from typing import Dict, List, Optional

import requests

from ..utils import logging
from . import BaseTransformersCLICommand

# 获取日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义 lfs-multipart-upload 命令
LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"

# 自定义 LfsCommands 类，实现 git-lfs 的 "multipart" 传输类型的自定义传输代理
class LfsCommands(BaseTransformersCLICommand):
    """
    Implementation of a custom transfer agent for the transfer type "multipart" for git-lfs. This lets users upload
    large files >5GB 🔥. Spec for LFS custom transfer agent is:
    https://github.com/git-lfs/git-lfs/blob/master/docs/custom-transfers.md

    This introduces two commands to the CLI:

    1. $ transformers-cli lfs-enable-largefiles

    This should be executed once for each model repo that contains a model file >5GB. It's documented in the error
    message you get if you just try to git push a 5GB file without having enabled it before.

    2. $ transformers-cli lfs-multipart-upload

    This command is called by lfs directly and is not meant to be called by the user.
    """

    # 注册子命令
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 添加 lfs-enable-largefiles 命令
        enable_parser = parser.add_parser(
            "lfs-enable-largefiles",
            help=(
                "Deprecated: use `huggingface-cli` instead. Configure your repository to enable upload of files > 5GB."
            ),
        )
        enable_parser.add_argument("path", type=str, help="Local path to repository you want to configure.")
        enable_parser.set_defaults(func=lambda args: LfsEnableCommand(args))

        # 添加 lfs-multipart-upload 命令
        upload_parser = parser.add_parser(
            LFS_MULTIPART_UPLOAD_COMMAND,
            help=(
                "Deprecated: use `huggingface-cli` instead. "
                "Command will get called by git-lfs, do not call it directly."
            ),
        )
        upload_parser.set_defaults(func=lambda args: LfsUploadCommand(args))

# 定义 LfsEnableCommand 类
class LfsEnableCommand:
    def __init__(self, args):
        self.args = args
    # 在运行方法中发出警告，提示通过 transformers-cli 管理仓库已不推荐，建议使用 `huggingface-cli` 代替
    warnings.warn(
        "Managing repositories through transformers-cli is deprecated. Please use `huggingface-cli` instead."
    )
    # 获取本地路径并确保其为有效的 Git 仓库路径
    local_path = os.path.abspath(self.args.path)
    if not os.path.isdir(local_path):
        # 若路径不是有效的 Git 仓库，则打印错误信息并退出
        print("This does not look like a valid git repo.")
        exit(1)
    # 设置 Git LFS 的自定义传输配置，使用 transformers-cli
    subprocess.run(
        "git config lfs.customtransfer.multipart.path transformers-cli".split(), check=True, cwd=local_path
    )
    # 设置 Git LFS 的自定义传输参数，指定上传命令
    subprocess.run(
        f"git config lfs.customtransfer.multipart.args {LFS_MULTIPART_UPLOAD_COMMAND}".split(),
        check=True,
        cwd=local_path,
    )
    # 打印提示信息，说明本地仓库已经设置好以处理大文件
    print("Local repo set up for largefiles")
# 定义一个函数，将消息以行分隔的 JSON 格式写入标准输出
def write_msg(msg: Dict):
    # 将消息字典转换成 JSON 格式并添加换行符
    msg = json.dumps(msg) + "\n"
    # 将消息写入标准输出
    sys.stdout.write(msg)
    # 立即刷新标准输出缓冲区
    sys.stdout.flush()

# 定义一个函数，从标准输入读取行分隔的 JSON 格式消息
def read_msg() -> Optional[Dict]:
    # 从标准输入读取一行并去除首尾空白字符，解析成 JSON 格式消息
    msg = json.loads(sys.stdin.readline().strip())

    # 如果消息中含有 "terminate" 字段，表示接收到终止消息，则返回 None
    if "terminate" in (msg.get("type"), msg.get("event")):
        # 收到终止消息
        return None

    # 如果消息中的 "event" 字段不是 "download" 或 "upload"，则记录日志并退出程序
    if msg.get("event") not in ("download", "upload"):
        logger.critical("Received unexpected message")
        sys.exit(1)

    # 返回解析后的消息字典
    return msg

# 定义一个上下文管理器类，实现仅读取文件的指定部分
class FileSlice(AbstractContextManager):
    """
    File-like object that only reads a slice of a file

    Inspired by stackoverflow.com/a/29838711/593036
    """

    # 初始化方法，接受文件路径、起始偏移和读取限制参数
    def __init__(self, filepath: str, seek_from: int, read_limit: int):
        self.filepath = filepath
        self.seek_from = seek_from
        self.read_limit = read_limit
        self.n_seen = 0

    # 进入上下文时执行的方法
    def __enter__(self):
        # 打开文件并将文件指针移动到指定位置
        self.f = open(self.filepath, "rb")
        self.f.seek(self.seek_from)
        return self

    # 返回文件内容的长度
    def __len__(self):
        # 获取文件总长度
        total_length = os.fstat(self.f.fileno()).st_size
        # 返回实际可读取的长度，不超过读取限制
        return min(self.read_limit, total_length - self.seek_from)

    # 读取文件内容的方法
    def read(self, n=-1):
        # 如果已经读取了指定长度的内容，则返回空字节串
        if self.n_seen >= self.read_limit:
            return b""
        # 计算剩余可读取的字节数
        remaining_amount = self.read_limit - self.n_seen
        # 读取文件内容，不超过剩余可读取的字节数
        data = self.f.read(remaining_amount if n < 0 else min(n, remaining_amount))
        # 更新已读取的字节数
        self.n_seen += len(data)
        return data

    # 迭代器方法，每次迭代返回读取的数据
    def __iter__(self):
        yield self.read(n=4 * 1024 * 1024)

    # 退出上下文时执行的方法，关闭文件
    def __exit__(self, *args):
        self.f.close()

# 定义一个类，表示 LFS 上传命令
class LfsUploadCommand:
    # 初始化方法，接受参数并保存到实例属性中
    def __init__(self, args):
        self.args = args
    # 定义一个方法，用于运行自定义的传输过程
    def run(self):
        # 从标准输入读取一行数据，并将其解析为 JSON 格式，获取初始化信息
        init_msg = json.loads(sys.stdin.readline().strip())
        # 检查初始化信息是否正确，如果不正确则发送错误消息并退出程序
        if not (init_msg.get("event") == "init" and init_msg.get("operation") == "upload"):
            write_msg({"error": {"code": 32, "message": "Wrong lfs init operation"}})
            sys.exit(1)

        # 响应初始化信息，发送一个空的确认消息到标准输出
        write_msg({})

        # 在初始化交换之后，git-lfs 将会发送任意数量的传输请求到传输过程的标准输入中，按顺序进行处理
        while True:
            # 读取传输请求消息
            msg = read_msg()
            if msg is None:
                # 当所有传输请求都被处理完毕时，git-lfs 将会发送一个终止事件到传输过程的标准输入中
                # 接收到此消息后，传输过程应该进行清理并终止，不需要响应
                sys.exit(0)

            # 获取传输请求中的相关信息
            oid = msg["oid"]
            filepath = msg["path"]
            completion_url = msg["action"]["href"]
            header = msg["action"]["header"]
            chunk_size = int(header.pop("chunk_size"))
            presigned_urls: List[str] = list(header.values())

            parts = []
            # 遍历预签名 URL 列表，按照指定的块大小上传数据
            for i, presigned_url in enumerate(presigned_urls):
                with FileSlice(filepath, seek_from=i * chunk_size, read_limit=chunk_size) as data:
                    r = requests.put(presigned_url, data=data)
                    r.raise_for_status()
                    # 将上传结果添加到 parts 列表中
                    parts.append(
                        {
                            "etag": r.headers.get("etag"),
                            "partNumber": i + 1,
                        }
                    )
                    # 为了支持数据上传/下载时的进度报告，传输过程应该向标准输出发送消息
                    write_msg(
                        {
                            "event": "progress",
                            "oid": oid,
                            "bytesSoFar": (i + 1) * chunk_size,
                            "bytesSinceLast": chunk_size,
                        }
                    )
                    # 不是精确的，但可以接受

            # 向完成 URL 发送 POST 请求，包含上传完成的信息
            r = requests.post(
                completion_url,
                json={
                    "oid": oid,
                    "parts": parts,
                },
            )
            r.raise_for_status()

            # 发送完成事件消息到标准输出
            write_msg({"event": "complete", "oid": oid})
```