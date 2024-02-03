# `.\AutoGPT\autogpts\autogpt\autogpt\commands\file_context.py`

```py
# 导入必要的模块和库
from __future__ import annotations
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

# 导入自定义模块和类
from autogpt.agents.features.context import ContextMixin, get_agent_context
from autogpt.agents.utils.exceptions import CommandExecutionError, DuplicateOperationError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.context_item import FileContextItem, FolderContextItem
from .decorators import sanitize_path_arg

# 定义命令的类别和标题
COMMAND_CATEGORY = "file_operations"
COMMAND_CATEGORY_TITLE = "File Operations"

# 检查类型是否为 TYPE_CHECKING
if TYPE_CHECKING:
    from autogpt.agents import Agent, BaseAgent

# 检查代理是否实现了上下文
def agent_implements_context(agent: BaseAgent) -> bool:
    return isinstance(agent, ContextMixin)

# 命令装饰器，用于打开文件
@command(
    "open_file",
    "Opens a file for editing or continued viewing; creates it if it does not exist yet. Note: If you only need to read or write a file once, use `write_to_file` instead.",
    {
        "file_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the file to open",
            required=True,
        )
    },
    available=agent_implements_context,
)
# 对文件路径进行清理
@sanitize_path_arg("file_path")
def open_file(file_path: Path, agent: Agent) -> tuple[str, FileContextItem]:
    """Open a file and return a context item

    Args:
        file_path (Path): The path of the file to open

    Returns:
        str: A status message indicating what happened
        FileContextItem: A ContextItem representing the opened file
    """
    # 尝试将文件路径转换为相对路径
    relative_file_path = None
    with contextlib.suppress(ValueError):
        relative_file_path = file_path.relative_to(agent.workspace.root)

    # 获取代理上下文
    assert (agent_context := get_agent_context(agent)) is not None

    # 检查文件是否存在，如果不存在则创建
    created = False
    if not file_path.exists():
        file_path.touch()
        created = True
    # 如果文件路径不是一个文件，则抛出命令执行错误
    elif not file_path.is_file():
        raise CommandExecutionError(f"{file_path} exists but is not a file")

    # 如果存在相对文件路径，则使用相对文件路径，否则使用原始文件路径
    file_path = relative_file_path or file_path

    # 创建文件上下文项，包括文件在工作空间中的路径和代理的工作空间根路径
    file = FileContextItem(
        file_path_in_workspace=file_path,
        workspace_path=agent.workspace.root,
    )
    
    # 如果文件已经在代理上下文中，则抛出重复操作错误
    if file in agent_context:
        raise DuplicateOperationError(f"The file {file_path} is already open")

    # 返回包含文件路径和是否已创建的消息，并将文件添加到上下文中
    return (
        f"File {file_path}{' created,' if created else ''} has been opened"
        " and added to the context ✅",
        file,
    )
# 定义一个名为 open_folder 的命令，用于打开一个文件夹并跟踪其内容
@command(
    "open_folder",
    "Open a folder to keep track of its content",
    {
        "path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the folder to open",
            required=True,
        )
    },
    available=agent_implements_context,
)
# 对路径参数进行路径清理
@sanitize_path_arg("path")
def open_folder(path: Path, agent: Agent) -> tuple[str, FolderContextItem]:
    """Open a folder and return a context item

    Args:
        path (Path): The path of the folder to open

    Returns:
        str: A status message indicating what happened
        FolderContextItem: A ContextItem representing the opened folder
    """
    # 尝试将路径转换为相对路径
    relative_path = None
    with contextlib.suppress(ValueError):
        relative_path = path.relative_to(agent.workspace.root)

    # 确保代理上下文存在
    assert (agent_context := get_agent_context(agent)) is not None

    # 如果路径不存在，则抛出文件或目录不存在的错误
    if not path.exists():
        raise FileNotFoundError(f"open_folder {path} failed: no such file or directory")
    # 如果路径存在但不是文件夹，则抛出命令执行错误
    elif not path.is_dir():
        raise CommandExecutionError(f"{path} exists but is not a folder")

    # 使用相对路径或原始路径更新路径
    path = relative_path or path

    # 创建一个表示打开文件夹的 ContextItem 对象
    folder = FolderContextItem(
        path_in_workspace=path,
        workspace_path=agent.workspace.root,
    )
    # 如果文件夹已经在代理上下文中，则抛出重复操作错误
    if folder in agent_context:
        raise DuplicateOperationError(f"The folder {path} is already open")

    # 返回成功消息和打开的文件夹对象
    return f"Folder {path} has been opened and added to the context ✅", folder
```