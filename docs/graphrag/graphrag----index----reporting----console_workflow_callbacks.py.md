# `.\graphrag\graphrag\index\reporting\console_workflow_callbacks.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Console-based reporter for the workflow engine."""

# 从 datashaper 模块导入 NoopWorkflowCallbacks 类
from datashaper import NoopWorkflowCallbacks


class ConsoleWorkflowCallbacks(NoopWorkflowCallbacks):
    """A reporter that writes to a console."""

    # 当发生错误时的回调函数
    def on_error(
        self,
        message: str,
        cause: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ):
        """Handle when an error occurs."""
        # 打印错误信息、异常、堆栈信息和详细信息（如果提供）
        print(message, str(cause), stack, details)  # noqa T201

    # 当发生警告时的回调函数
    def on_warning(self, message: str, details: dict | None = None):
        """Handle when a warning occurs."""
        # 调用 _print_warning 函数打印警告信息
        _print_warning(message)

    # 当产生日志消息时的回调函数
    def on_log(self, message: str, details: dict | None = None):
        """Handle when a log message is produced."""
        # 打印日志消息及其详细信息（如果提供）
        print(message, details)  # noqa T201


# 定义一个辅助函数，用于打印警告信息
def _print_warning(skk):
    # 打印带有黄色前景色的警告信息
    print("\033[93m {}\033[00m".format(skk))  # noqa T201
```