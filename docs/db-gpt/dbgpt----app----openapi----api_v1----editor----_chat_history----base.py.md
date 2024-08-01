# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\editor\_chat_history\base.py`

```py
# 引入了 __future__ 模块，用于支持注解（annotations）特性
from __future__ import annotations

# 从 abc 模块中引入 ABC 抽象基类和 abstractmethod 装饰器
from abc import ABC, abstractmethod

# 从 enum 模块引入 Enum 枚举类型
from enum import Enum

# 从 typing 模块引入 Dict、List、Optional 类型注解
from typing import Dict, List, Optional

# 从 dbgpt.core.interface.message 模块引入 OnceConversation 类
from dbgpt.core.interface.message import OnceConversation

# 定义一个枚举类型 MemoryStoreType，表示内存存储的类型
class MemoryStoreType(Enum):
    DB = "db"

# 定义一个抽象基类 BaseChatHistoryMemory，继承自 ABC 抽象基类
class BaseChatHistoryMemory(ABC):
    # 定义一个实例属性 store_type，类型为 MemoryStoreType，用于标识存储类型
    store_type: MemoryStoreType

    # 构造函数，初始化实例属性 conversations 为一个空列表
    def __init__(self):
        self.conversations: List[OnceConversation] = []

    # 抽象方法，返回值为 List[OnceConversation]，用于获取消息列表
    @abstractmethod
    def messages(self) -> List[OnceConversation]:  # type: ignore
        """Retrieve the messages from the local file"""

    # 抽象方法，接受一个 OnceConversation 类型参数，用于追加消息记录到本地文件
    @abstractmethod
    def append(self, message: OnceConversation) -> None:
        """Append the message to the record in the local file"""

    # 抽象方法，接受一个 List[OnceConversation] 类型参数，用于更新消息记录到本地文件
    @abstractmethod
    def update(self, messages: List[OnceConversation]) -> None:
        pass

    # 抽象方法，无参数，返回值为 bool 类型，用于删除本地文件的消息记录
    @abstractmethod
    def delete(self) -> bool:
        pass

    # 抽象方法，无参数，返回值为 List[Dict]，获取本地文件的消息列表
    @abstractmethod
    def get_messages(self) -> List[Dict]:
        pass

    # 静态方法，接受两个可选参数 user_name 和 sys_code，返回值为 List[Dict]，获取用户的对话列表
    @staticmethod
    @abstractmethod
    def conv_list(
        user_name: Optional[str] = None, sys_code: Optional[str] = None
    ) -> List[Dict]:
        """get user's conversation list"""
```