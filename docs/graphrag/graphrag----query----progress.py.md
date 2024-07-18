# `.\graphrag\graphrag\query\progress.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Status Reporter for orchestration."""

# 导入必要的模块
from abc import ABCMeta, abstractmethod
from typing import Any


# 定义一个抽象基类 StatusReporter，用于报告管道中的状态更新
class StatusReporter(metaclass=ABCMeta):
    """Provides a way to report status updates from the pipeline."""

    @abstractmethod
    def error(self, message: str, details: dict[str, Any] | None = None):
        """Report an error."""

    @abstractmethod
    def warning(self, message: str, details: dict[str, Any] | None = None):
        """Report a warning."""

    @abstractmethod
    def log(self, message: str, details: dict[str, Any] | None = None):
        """Report a log."""


# 定义 ConsoleStatusReporter 类，它是 StatusReporter 的子类，将状态报告输出到控制台
class ConsoleStatusReporter(StatusReporter):
    """A reporter that writes to a console."""

    def error(self, message: str, details: dict[str, Any] | None = None):
        """Report an error."""
        # 打印错误消息和细节到控制台，忽略 T201 规则（指示不要在注释中使用标点符号）
        print(message, details)  # noqa T201

    def warning(self, message: str, details: dict[str, Any] | None = None):
        """Report a warning."""
        # 打印警告消息到控制台，调用 _print_warning 函数来格式化输出
        _print_warning(message)

    def log(self, message: str, details: dict[str, Any] | None = None):
        """Report a log."""
        # 打印日志消息和细节到控制台，忽略 T201 规则
        print(message, details)  # noqa T201


# 定义一个辅助函数 _print_warning，用于格式化打印警告消息到控制台
def _print_warning(skk):
    print(f"\033[93m {skk}\033[00m")  # noqa T201
```