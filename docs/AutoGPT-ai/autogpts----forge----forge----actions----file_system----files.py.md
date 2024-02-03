# `.\AutoGPT\autogpts\forge\forge\actions\file_system\files.py`

```py
# 导入 List 类型
from typing import List

# 从 registry 模块中导入 action 装饰器
from ..registry import action

# 定义 list_files 动作函数，用于列出目录中的文件
@action(
    name="list_files",
    description="List files in a directory",
    parameters=[
        {
            "name": "path",
            "description": "Path to the directory",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)
async def list_files(agent, task_id: str, path: str) -> List[str]:
    """
    List files in a workspace directory
    """
    # 调用 agent 对象的 workspace 属性的 list 方法，列出目录中的文件
    return agent.workspace.list(task_id=task_id, path=str(path))

# 定义 write_file 动作函数，用于向文件写入数据
@action(
    name="write_file",
    description="Write data to a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "data",
            "description": "Data to write to the file",
            "type": "bytes",
            "required": True,
        },
    ],
    output_type="None",
)
async def write_file(agent, task_id: str, file_path: str, data: bytes):
    """
    Write data to a file
    """
    # 如果 data 是字符串类型，则转换为字节类型
    if isinstance(data, str):
        data = data.encode()

    # 调用 agent 对象的 workspace 属性的 write 方法，向文件写入数据
    agent.workspace.write(task_id=task_id, path=file_path, data=data)
    # 创建一个 artifact 并返回
    return await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_path.split("/")[-1],
        relative_path=file_path,
        agent_created=True,
    )

# 定义 read_file 动作函数，用于从文件中读取数据
@action(
    name="read_file",
    description="Read data from a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="bytes",
)
async def read_file(agent, task_id: str, file_path: str) -> bytes:
    """
    Read data from a file
    """
    # 调用 agent 对象的 workspace 属性的 read 方法，从文件中读取数据
    return agent.workspace.read(task_id=task_id, path=file_path)
```