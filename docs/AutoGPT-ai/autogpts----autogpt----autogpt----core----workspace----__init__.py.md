# `.\AutoGPT\autogpts\autogpt\autogpt\core\workspace\__init__.py`

```py
# 工作空间是 Agent 在磁盘资源上的中央枢纽
"""The workspace is the central hub for the Agent's on disk resources."""

# 导入 Workspace 类
from autogpt.core.workspace.base import Workspace
# 导入 SimpleWorkspace 和 WorkspaceSettings 类
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings

# 暴露给外部的类列表
__all__ = [
    "SimpleWorkspace",
    "Workspace",
    "WorkspaceSettings",
]
```