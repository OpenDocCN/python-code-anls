# `MetaGPT\metagpt\utils\file.py`

```

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : file.py
@Describe : General file operations.
"""
# 导入 Path 类
from pathlib import Path
# 导入 aiofiles 模块
import aiofiles
# 导入日志记录器
from metagpt.logs import logger
# 导入异常处理装饰器
from metagpt.utils.exceptions import handle_exception

# 定义文件类
class File:
    """A general util for file operations."""

    # 定义块大小
    CHUNK_SIZE = 64 * 1024

    # 写文件方法
    @classmethod
    @handle_exception
    async def write(cls, root_path: Path, filename: str, content: bytes) -> Path:
        """Write the file content to the local specified path.

        Args:
            root_path: The root path of file, such as "/data".
            filename: The name of file, such as "test.txt".
            content: The binary content of file.

        Returns:
            The full filename of file, such as "/data/test.txt".

        Raises:
            Exception: If an unexpected error occurs during the file writing process.
        """
        # 创建目录
        root_path.mkdir(parents=True, exist_ok=True)
        # 拼接完整路径
        full_path = root_path / filename
        # 异步打开文件并写入内容
        async with aiofiles.open(full_path, mode="wb") as writer:
            await writer.write(content)
            logger.debug(f"Successfully write file: {full_path}")
            return full_path

    # 读文件方法
    @classmethod
    @handle_exception
    async def read(cls, file_path: Path, chunk_size: int = None) -> bytes:
        """Partitioning read the file content from the local specified path.

        Args:
            file_path: The full file name of file, such as "/data/test.txt".
            chunk_size: The size of each chunk in bytes (default is 64kb).

        Returns:
            The binary content of file.

        Raises:
            Exception: If an unexpected error occurs during the file reading process.
        """
        # 如果未指定块大小，则使用默认块大小
        chunk_size = chunk_size or cls.CHUNK_SIZE
        # 异步打开文件并按块读取内容
        async with aiofiles.open(file_path, mode="rb") as reader:
            chunks = list()
            while True:
                chunk = await reader.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
            content = b"".join(chunks)
            logger.debug(f"Successfully read file, the path of file: {file_path}")
            return content

```