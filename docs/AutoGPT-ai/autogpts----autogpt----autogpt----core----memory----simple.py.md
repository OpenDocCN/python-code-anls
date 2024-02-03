# `.\AutoGPT\autogpts\autogpt\autogpt\core\memory\simple.py`

```py
# 导入所需的模块
import json
import logging

# 导入自定义模块
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory
from autogpt.core.workspace import Workspace

# 定义内存配置类，继承自系统配置类
class MemoryConfiguration(SystemConfiguration):
    pass

# 定义内存设置类，继承自系统设置类
class MemorySettings(SystemSettings):
    configuration: MemoryConfiguration

# 定义消息历史类
class MessageHistory:
    def __init__(self, previous_message_history: list[str]):
        self._message_history = previous_message_history

# 定义简单内存类，继承自内存类和可配置类
class SimpleMemory(Memory, Configurable):
    # 默认设置为简单内存设置
    default_settings = MemorySettings(
        name="simple_memory",
        description="A simple memory.",
        configuration=MemoryConfiguration(),
    )

    # 初始化方法
    def __init__(
        self,
        settings: MemorySettings,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        # 设置内存配置
        self._configuration = settings.configuration
        # 设置日志记录器
        self._logger = logger
        # 加载消息历史
        self._message_history = self._load_message_history(workspace)

    # 加载消息历史的静态方法
    @staticmethod
    def _load_message_history(workspace: Workspace):
        # 获取消息历史文件路径
        message_history_path = workspace.get_path("message_history.json")
        # 如果消息历史文件存在
        if message_history_path.exists():
            # 读取消息历史文件内容
            with message_history_path.open("r") as f:
                message_history = json.load(f)
        else:
            message_history = []
        # 返回消息历史对象
        return MessageHistory(message_history)
```