# `.\AutoGPT\autogpts\forge\forge\sdk\__init__.py`

```py
"""
The Forge SDK. This is the core of the Forge. It contains the agent protocol, which is the
core of the Forge.
"""
# 导入 Forge SDK 的核心模块，包含 Forge 的核心功能
from ..llm import chat_completion_request, create_embedding_request, transcribe_audio
# 导入 agent 模块
from .agent import Agent
# 导入 AgentDB 和 Base 类
from .db import AgentDB, Base
# 导入错误相关的模块
from .errors import *
# 导入 ForgeLogger 类
from .forge_log import ForgeLogger
# 导入模型相关的类
from .model import (
    Artifact,
    ArtifactUpload,
    Pagination,
    Status,
    Step,
    StepOutput,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
# 导入提示引擎模块
from .prompting import PromptEngine
# 导入本地工作空间和工作空间类
from .workspace import LocalWorkspace, Workspace
```