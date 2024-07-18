# `.\graphrag\graphrag\index\progress\types.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Types for status reporting."""

from abc import ABC, abstractmethod  # 导入 ABC 类和 abstractmethod 装饰器

from datashaper import Progress  # 导入 Progress 类


class ProgressReporter(ABC):
    """
    Abstract base class for progress reporters.

    This is used to report workflow processing progress via mechanisms like progress-bars.
    """

    @abstractmethod
    def __call__(self, update: Progress):
        """Update progress."""
    
    @abstractmethod
    def dispose(self):
        """Dispose of the progress reporter."""
    
    @abstractmethod
    def child(self, prefix: str, transient=True) -> "ProgressReporter":
        """Create a child progress bar."""
    
    @abstractmethod
    def force_refresh(self) -> None:
        """Force a refresh."""
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the progress reporter."""
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Report an error."""
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Report a warning."""
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Report information."""
    
    @abstractmethod
    def success(self, message: str) -> None:
        """Report success."""


class NullProgressReporter(ProgressReporter):
    """A progress reporter that does nothing."""

    def __call__(self, update: Progress) -> None:
        """Update progress."""
        pass

    def dispose(self) -> None:
        """Dispose of the progress reporter."""
        pass

    def child(self, prefix: str, transient: bool = True) -> ProgressReporter:
        """Create a child progress bar."""
        return self

    def force_refresh(self) -> None:
        """Force a refresh."""
        pass

    def stop(self) -> None:
        """Stop the progress reporter."""
        pass

    def error(self, message: str) -> None:
        """Report an error."""
        pass

    def warning(self, message: str) -> None:
        """Report a warning."""
        pass

    def info(self, message: str) -> None:
        """Report information."""
        pass

    def success(self, message: str) -> None:
        """Report success."""
        pass


class PrintProgressReporter(ProgressReporter):
    """A progress reporter that prints progress dots."""

    prefix: str

    def __init__(self, prefix: str):
        """Create a new progress reporter."""
        self.prefix = prefix
        print(f"\n{self.prefix}", end="")  # noqa T201

    def __call__(self, update: Progress) -> None:
        """Update progress."""
        print(".", end="")  # noqa T201

    def dispose(self) -> None:
        """Dispose of the progress reporter."""
        pass

    def child(self, prefix: str, transient: bool = True) -> "ProgressReporter":
        """Create a child progress bar."""
        return PrintProgressReporter(prefix)

    def stop(self) -> None:
        """Stop the progress reporter."""
        pass

    def force_refresh(self) -> None:
        """Force a refresh."""
        pass
    # 定义一个方法用于报告错误信息，接收一个字符串参数 message
    def error(self, message: str) -> None:
        """Report an error."""
        # 打印带有前缀的错误消息，使用 T201 告诉 linter 忽略该行的格式错误
        print(f"\n{self.prefix}ERROR: {message}")  # noqa T201

    # 定义一个方法用于报告警告信息，接收一个字符串参数 message
    def warning(self, message: str) -> None:
        """Report a warning."""
        # 打印带有前缀的警告消息，使用 T201 告诉 linter 忽略该行的格式错误
        print(f"\n{self.prefix}WARNING: {message}")  # noqa T201

    # 定义一个方法用于报告信息，接收一个字符串参数 message
    def info(self, message: str) -> None:
        """Report information."""
        # 打印带有前缀的信息消息，使用 T201 告诉 linter 忽略该行的格式错误
        print(f"\n{self.prefix}INFO: {message}")  # noqa T201

    # 定义一个方法用于报告成功信息，接收一个字符串参数 message
    def success(self, message: str) -> None:
        """Report success."""
        # 打印带有前缀的成功消息，使用 T201 告诉 linter 忽略该行的格式错误
        print(f"\n{self.prefix}SUCCESS: {message}")  # noqa T201
```