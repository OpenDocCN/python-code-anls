# `.\pytorch\torch\_streambase.py`

```py
# mypy: allow-untyped-defs
# 引入ABC抽象基类，用于定义抽象方法
from abc import ABC, abstractmethod


# 定义流对象基类，用于多后端流的继承
class _StreamBase(ABC):
    r"""Base stream class abstraction for multi backends Stream to herit from"""

    # 等待事件的抽象方法声明，无返回值
    @abstractmethod
    def wait_event(self, event) -> None:
        raise NotImplementedError

    # 等待流的抽象方法声明，无返回值
    @abstractmethod
    def wait_stream(self, stream) -> None:
        raise NotImplementedError

    # 记录事件的抽象方法声明，无返回值
    @abstractmethod
    def record_event(self, event=None) -> None:
        raise NotImplementedError

    # 查询状态的抽象方法声明，返回布尔值
    @abstractmethod
    def query(self) -> bool:
        raise NotImplementedError

    # 同步操作的抽象方法声明，无返回值
    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError

    # 定义等于操作的抽象方法声明，返回布尔值
    @abstractmethod
    def __eq__(self, stream) -> bool:
        raise NotImplementedError


# 定义事件对象基类，用于多后端事件的继承
class _EventBase(ABC):
    r"""Base Event class abstraction for multi backends Event to herit from"""

    # 等待事件的抽象方法声明，无返回值
    @abstractmethod
    def wait(self, stream=None) -> None:
        raise NotImplementedError

    # 查询状态的抽象方法声明，返回布尔值
    @abstractmethod
    def query(self) -> bool:
        raise NotImplementedError

    # 同步操作的抽象方法声明，无返回值
    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError
```