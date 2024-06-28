# `.\commands\lfs.py`

```
"""
Implementation of a custom transfer agent for the transfer type "multipart" for git-lfs.

Inspired by: github.com/cbartz/git-lfs-swift-transfer-agent/blob/master/git_lfs_swift_transfer.py

Spec is: github.com/git-lfs/git-lfs/blob/master/docs/custom-transfers.md


To launch debugger while developing:

``` [lfs "customtransfer.multipart"]
path = /path/to/transformers/.env/bin/python args = -m debugpy --listen 5678 --wait-for-client
/path/to/transformers/src/transformers/commands/transformers_cli.py lfs-multipart-upload ```"""

import json  # 导入处理 JSON 的模块
import os  # 导入操作系统功能的模块
import subprocess  # 导入运行外部命令的模块
import sys  # 导入与 Python 解释器交互的模块
import warnings  # 导入警告处理的模块
from argparse import ArgumentParser  # 从 argparse 模块中导入 ArgumentParser 类
from contextlib import AbstractContextManager  # 从 contextlib 模块中导入 AbstractContextManager 类
from typing import Dict, List, Optional  # 导入类型提示相关的模块

import requests  # 导入处理 HTTP 请求的模块

from ..utils import logging  # 从相对路径中导入 logging 模块
from . import BaseTransformersCLICommand  # 从当前目录中导入 BaseTransformersCLICommand 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象，并赋值给 logger 变量  # pylint: disable=invalid-name

LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"  # 定义一个常量，指定 LFS 多部分上传命令的名称

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

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        enable_parser = parser.add_parser(
            "lfs-enable-largefiles",
            help=(
                "Deprecated: use `huggingface-cli` instead. Configure your repository to enable upload of files > 5GB."
            ),
        )
        enable_parser.add_argument("path", type=str, help="Local path to repository you want to configure.")
        enable_parser.set_defaults(func=lambda args: LfsEnableCommand(args))  # 设置默认的命令处理函数为 LfsEnableCommand 类的实例化

        upload_parser = parser.add_parser(
            LFS_MULTIPART_UPLOAD_COMMAND,
            help=(
                "Deprecated: use `huggingface-cli` instead. "
                "Command will get called by git-lfs, do not call it directly."
            ),
        )
        upload_parser.set_defaults(func=lambda args: LfsUploadCommand(args))  # 设置默认的命令处理函数为 LfsUploadCommand 类的实例化

class LfsEnableCommand:
    def __init__(self, args):
        self.args = args  # 初始化类实例时，将参数保存到实例属性中
    def run(self):
        # 发出警告信息，提示使用 `huggingface-cli` 取代 `transformers-cli` 管理仓库
        warnings.warn(
            "Managing repositories through transformers-cli is deprecated. Please use `huggingface-cli` instead."
        )
        # 获取指定路径的绝对路径
        local_path = os.path.abspath(self.args.path)
        # 如果指定路径不是一个目录，则输出错误信息并退出程序
        if not os.path.isdir(local_path):
            print("This does not look like a valid git repo.")
            exit(1)
        # 设置 git-lfs 的自定义传输程序路径为 `transformers-cli`，在指定路径下执行
        subprocess.run(
            "git config lfs.customtransfer.multipart.path transformers-cli".split(), check=True, cwd=local_path
        )
        # 设置 git-lfs 的自定义传输程序参数为预定义的 `LFS_MULTIPART_UPLOAD_COMMAND` 值，在指定路径下执行
        subprocess.run(
            f"git config lfs.customtransfer.multipart.args {LFS_MULTIPART_UPLOAD_COMMAND}".split(),
            check=True,
            cwd=local_path,
        )
        # 输出信息，表示本地仓库已设置好以处理大文件
        print("Local repo set up for largefiles")
# 将字典消息转换为 JSON 格式并写入标准输出
def write_msg(msg: Dict):
    msg = json.dumps(msg) + "\n"  # 转换字典消息为 JSON 字符串，并添加换行符
    sys.stdout.write(msg)  # 将 JSON 字符串写入标准输出
    sys.stdout.flush()  # 刷新标准输出缓冲区，确保消息被写入

# 从标准输入读取一行 JSON 格式的消息
def read_msg() -> Optional[Dict]:
    msg = json.loads(sys.stdin.readline().strip())  # 读取并解析 JSON 格式的消息

    if "terminate" in (msg.get("type"), msg.get("event")):
        # 如果消息中包含 "terminate" 类型或事件，表示终止消息已接收
        return None

    if msg.get("event") not in ("download", "upload"):
        logger.critical("Received unexpected message")  # 记录关键错误日志，表示接收到意外的消息
        sys.exit(1)  # 非预期消息时退出程序

    return msg  # 返回解析后的消息字典

# 用于从文件中读取指定范围的数据的上下文管理器类
class FileSlice(AbstractContextManager):
    """
    File-like object that only reads a slice of a file

    Inspired by stackoverflow.com/a/29838711/593036
    """

    def __init__(self, filepath: str, seek_from: int, read_limit: int):
        self.filepath = filepath  # 文件路径
        self.seek_from = seek_from  # 读取起始位置
        self.read_limit = read_limit  # 读取数据限制大小
        self.n_seen = 0  # 已读取的字节数

    def __enter__(self):
        self.f = open(self.filepath, "rb")  # 打开文件以供读取
        self.f.seek(self.seek_from)  # 设置文件读取的起始位置
        return self  # 返回 FileSlice 对象本身作为上下文管理器

    def __len__(self):
        total_length = os.fstat(self.f.fileno()).st_size  # 获取文件总大小
        return min(self.read_limit, total_length - self.seek_from)  # 返回实际可读取的数据长度

    def read(self, n=-1):
        if self.n_seen >= self.read_limit:
            return b""  # 如果已读取数据超出限制，则返回空字节串

        remaining_amount = self.read_limit - self.n_seen  # 剩余可读取的数据量
        # 读取数据，不超过剩余可读取的数据量或指定的 n 字节
        data = self.f.read(remaining_amount if n < 0 else min(n, remaining_amount))
        self.n_seen += len(data)  # 更新已读取的字节数
        return data  # 返回读取的数据

    def __iter__(self):
        yield self.read(n=4 * 1024 * 1024)  # 以迭代器方式返回每次最多 4MB 的数据

    def __exit__(self, *args):
        self.f.close()  # 关闭文件

# LFS 上传命令类，初始化时接收参数
class LfsUploadCommand:
    def __init__(self, args):
        self.args = args  # 初始化 LFS 上传命令的参数
    def run(self):
        # 立即在调用自定义传输过程后，git-lfs通过标准输入发送初始化数据到进程中。
        # 这向进程提供了关于配置的有用信息。
        init_msg = json.loads(sys.stdin.readline().strip())
        # 如果初始化消息不是"init"事件且操作不是"upload"，则写入错误消息并退出程序。
        if not (init_msg.get("event") == "init" and init_msg.get("operation") == "upload"):
            write_msg({"error": {"code": 32, "message": "Wrong lfs init operation"}})
            sys.exit(1)

        # 传输过程应使用初始化结构中的信息，并执行任何一次性设置任务。
        # 然后通过标准输出响应一个简单的空确认结构。
        write_msg({})

        # 初始化交换后，git-lfs将按序列发送任意数量的传输请求到传输进程的标准输入。
        while True:
            msg = read_msg()
            if msg is None:
                # 当所有传输都已处理完毕时，git-lfs将向传输进程的标准输入发送终止事件。
                # 收到此消息后，传输进程应清理并终止。不需要响应。
                sys.exit(0)

            oid = msg["oid"]
            filepath = msg["path"]
            completion_url = msg["action"]["href"]
            header = msg["action"]["header"]
            chunk_size = int(header.pop("chunk_size"))
            presigned_urls: List[str] = list(header.values())

            parts = []
            for i, presigned_url in enumerate(presigned_urls):
                # 使用FileSlice从文件中读取数据片段，根据chunk_size和偏移量进行读取。
                with FileSlice(filepath, seek_from=i * chunk_size, read_limit=chunk_size) as data:
                    # 发送PUT请求上传数据片段到预签名的URL。
                    r = requests.put(presigned_url, data=data)
                    r.raise_for_status()
                    # 添加上传片段的ETag和序号到parts列表。
                    parts.append(
                        {
                            "etag": r.headers.get("etag"),
                            "partNumber": i + 1,
                        }
                    )
                    # 为了支持数据上传/下载过程中的进度报告，
                    # 传输进程应向标准输出发送消息。
                    write_msg(
                        {
                            "event": "progress",
                            "oid": oid,
                            "bytesSoFar": (i + 1) * chunk_size,
                            "bytesSinceLast": chunk_size,
                        }
                    )
                    # 不是精确的进度报告，但可以接受。

            # 发送包含oid和已上传部分信息的POST请求到完成URL。
            r = requests.post(
                completion_url,
                json={
                    "oid": oid,
                    "parts": parts,
                },
            )
            r.raise_for_status()

            # 发送完成事件到标准输出。
            write_msg({"event": "complete", "oid": oid})
```