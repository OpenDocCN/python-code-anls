# `.\AutoGPT\autogpts\autogpt\autogpt\core\agent\base.py`

```py
# 导入 abc 模块，用于定义抽象基类
import abc
# 导入 logging 模块，用于记录日志
import logging
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 定义 Agent 类，作为抽象基类
class Agent(abc.ABC):
    # 抽象方法，用于初始化 Agent 类
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    # 类方法，从工作空间路径和日志记录器创建 Agent 实例
    @classmethod
    @abc.abstractmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "Agent":
        ...

    # 抽象方法，用于确定下一个能力
    @abc.abstractmethod
    async def determine_next_ability(self, *args, **kwargs):
        ...

    # 抽象方法，用于返回对象的字符串表示形式
    @abc.abstractmethod
    def __repr__(self):
        ...
```