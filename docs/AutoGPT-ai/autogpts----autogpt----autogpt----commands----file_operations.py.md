# `.\AutoGPT\autogpts\autogpt\autogpt\commands\file_operations.py`

```py
# 导入必要的库和模块
from __future__ import annotations
import hashlib
import logging
import os
import os.path
from pathlib import Path
from typing import Iterator, Literal

from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import DuplicateOperationError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.memory.vector import MemoryItemFactory, VectorMemory

# 导入自定义的装饰器和工具函数
from .decorators import sanitize_path_arg
from .file_operations_utils import decode_textual_file

# 定义命令的分类和标题
COMMAND_CATEGORY = "file_operations"
COMMAND_CATEGORY_TITLE = "File Operations"

# 导入文件上下文管理器
from .file_context import open_file, open_folder  # NOQA

# 获取日志记录器对象
logger = logging.getLogger(__name__)

# 定义操作类型
Operation = Literal["write", "append", "delete"]

# 计算文本的 MD5 校验和
def text_checksum(text: str) -> str:
    """Get the hex checksum for the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# 从日志文件中解析文件操作记录
def operations_from_log(
    log_path: str | Path,
) -> Iterator[
    tuple[Literal["write", "append"], str, str] | tuple[Literal["delete"], str, None]
]:
    """Parse the file operations log and return a tuple containing the log entries"""
    try:
        log = open(log_path, "r", encoding="utf-8")
    except FileNotFoundError:
        return

    for line in log:
        line = line.replace("File Operation Logger", "").strip()
        if not line:
            continue
        operation, tail = line.split(": ", maxsplit=1)
        operation = operation.strip()
        if operation in ("write", "append"):
            path, checksum = (x.strip() for x in tail.rsplit(" #", maxsplit=1))
            yield (operation, path, checksum)
        elif operation == "delete":
            yield (operation, tail.strip(), None)

    log.close()

# 获取文件操作的状态
def file_operations_state(log_path: str | Path) -> dict[str, str]:
    """Iterates over the operations log and returns the expected state.
    # 解析文件管理器中文件操作日志路径下的日志文件，构建一个字典，将每个写入或追加的文件路径映射到其校验和。已删除的文件将从字典中移除。
    # 返回一个将文件路径映射到其校验和的字典。
    # 如果找不到文件管理器中的文件操作日志路径，则引发 FileNotFoundError 异常。
    # 如果日志文件内容不符合预期格式，则引发 ValueError 异常。
    """
    state = {}
    # 遍历从日志文件中获取的操作、路径和校验和
    for operation, path, checksum in operations_from_log(log_path):
        # 如果操作是写入或追加，则将路径映射到校验和
        if operation in ("write", "append"):
            state[path] = checksum
        # 如果操作是删除，则从字典中删除该路径
        elif operation == "delete":
            del state[path]
    # 返回最终的状态字典
    return state
# 对文件路径参数进行清理，使其相对路径，并检查是否为重复操作
@sanitize_path_arg("file_path", make_relative=True)
def is_duplicate_operation(
    operation: Operation, file_path: Path, agent: Agent, checksum: str | None = None
) -> bool:
    """Check if the operation has already been performed

    Args:
        operation: The operation to check for
        file_path: The name of the file to check for
        agent: The agent
        checksum: The checksum of the contents to be written

    Returns:
        True if the operation has already been performed on the file
    """
    # 获取文件操作状态
    state = file_operations_state(agent.file_manager.file_ops_log_path)
    # 如果操作为删除且文件路径不在状态中，则返回True
    if operation == "delete" and str(file_path) not in state:
        return True
    # 如果操作为写入且文件路径在状态中且对应的校验和与给定校验和相同，则返回True
    if operation == "write" and state.get(str(file_path)) == checksum:
        return True
    return False


# 对文件路径参数进行清理，使其相对路径，并将文件操作记录到file_logger.log中
@sanitize_path_arg("file_path", make_relative=True)
def log_operation(
    operation: Operation,
    file_path: str | Path,
    agent: Agent,
    checksum: str | None = None,
) -> None:
    """Log the file operation to the file_logger.log

    Args:
        operation: The operation to log
        file_path: The name of the file the operation was performed on
        checksum: The checksum of the contents to be written
    """
    # 构建日志条目
    log_entry = f"{operation}: {file_path}"
    if checksum is not None:
        log_entry += f" #{checksum}"
    # 记录文件操作到日志
    logger.debug(f"Logging file operation: {log_entry}")
    append_to_file(
        agent.file_manager.file_ops_log_path, f"{log_entry}\n", agent, should_log=False
    )


# 定义命令"read_file"，用于读取现有文件
@command(
    "read_file",
    "Read an existing file",
    {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the file to read",
            required=True,
        )
    },
)
def read_file(filename: str | Path, agent: Agent) -> str:
    """Read a file and return the contents

    Args:
        filename (Path): The name of the file to read

    Returns:
        str: The contents of the file
    """
    # 使用代理对象的工作空间打开指定文件，以二进制模式
    file = agent.workspace.open_file(filename, binary=True)
    # 解码文本文件内容，根据文件扩展名和记录器
    content = decode_textual_file(file, os.path.splitext(filename)[1], logger)

    # # TODO: 当文件被编辑时，使内存无效/更新
    # # 创建文本文件的内存项，使用内容、文件名和代理配置
    # file_memory = MemoryItem.from_text_file(content, str(filename), agent.config)
    # 如果文件内存项的块数大于1，则返回文件内存项的摘要
    # if len(file_memory.chunks) > 1:
    #     return file_memory.summary

    # 返回文件内容
    return content
# 从文件中读取内容，将其分割成具有指定最大长度和重叠的块，并将这些块添加到内存存储中
def ingest_file(
    filename: str,
    memory: VectorMemory,
) -> None:
    try:
        # 记录日志，表示正在处理文件
        logger.info(f"Ingesting file {filename}")
        # 读取文件内容
        content = read_file(filename)

        # TODO: 区分不同类型的文件
        # 根据文件内容和文件名创建 MemoryItem 对象
        file_memory = MemoryItemFactory.from_text_file(content, filename)
        # 记录调试日志，显示创建的 MemoryItem 信息
        logger.debug(f"Created memory: {file_memory.dump(True)}")
        # 将 MemoryItem 添加到内存存储中
        memory.add(file_memory)

        # 记录日志，表示已从文件中提取了多少块数据
        logger.info(f"Ingested {len(file_memory.e_chunks)} chunks from {filename}")
    except Exception as err:
        # 记录警告日志，表示在处理文件时出现错误
        logger.warning(f"Error while ingesting file '{filename}': {err}")


# 命令函数，用于写入文件，如果文件不存在则创建，如果文件存在则覆盖
@command(
    "write_file",
    "Write a file, creating it if necessary. If the file exists, it is overwritten.",
    {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to write to",
            required=True,
        ),
        "contents": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The contents to write to the file",
            required=True,
        ),
    },
    aliases=["create_file"],
)
async def write_to_file(filename: str | Path, contents: str, agent: Agent) -> str:
    """Write contents to a file

    Args:
        filename (Path): The name of the file to write to
        contents (str): The contents to write to the file

    Returns:
        str: A message indicating success or failure
    """
    # 计算内容的校验和
    checksum = text_checksum(contents)
    # 检查是否存在重复的写入操作
    if is_duplicate_operation("write", Path(filename), agent, checksum):
        # 如果存在重复操作，则抛出异常
        raise DuplicateOperationError(f"File {filename} has already been updated.")
    # 如果文件名中包含目录信息，则获取目录路径
    if directory := os.path.dirname(filename):
        # 获取目录路径对应的路径对象，并创建目录（如果不存在）
        agent.workspace.get_path(directory).mkdir(exist_ok=True)
    # 异步写入文件内容到工作空间
    await agent.workspace.write_file(filename, contents)
    # 记录写入操作的日志
    log_operation("write", filename, agent, checksum)
    # 返回写入成功的消息
    return f"File {filename} has been written successfully."
# 将文本追加到文件末尾
def append_to_file(
    filename: Path, text: str, agent: Agent, should_log: bool = True
) -> None:
    # 获取文件所在目录路径
    directory = os.path.dirname(filename)
    # 如果目录不存在则创建目录
    os.makedirs(directory, exist_ok=True)
    # 以追加模式打开文件
    with open(filename, "a") as f:
        # 写入文本内容
        f.write(text)

    # 如果需要记录日志
    if should_log:
        # 以只读模式打开文件
        with open(filename, "r") as f:
            # 计算文本内容的校验和
            checksum = text_checksum(f.read())
        # 记录操作日志
        log_operation("append", filename, agent, checksum=checksum)


# 命令装饰器，用于定义 list_folder 命令
@command(
    "list_folder",
    "List the items in a folder",
    {
        "folder": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The folder to list files in",
            required=True,
        )
    },
)
# 列出指定文件夹中的所有文件（包括子文件夹中的文件）
def list_folder(folder: str | Path, agent: Agent) -> list[str]:
    # 搜索指定文件夹中的所有文件
    return [str(p) for p in agent.workspace.list(folder)]
```