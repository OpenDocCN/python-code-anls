# `.\graphrag\graphrag\index\reporting\file_workflow_callbacks.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A reporter that writes to a file."""

# 引入所需模块和库
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
from io import TextIOWrapper  # 导入文本 I/O 封装模块
from pathlib import Path  # 导入路径操作模块

from datashaper import NoopWorkflowCallbacks  # 从 datashaper 模块导入 NoopWorkflowCallbacks 类

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class FileWorkflowCallbacks(NoopWorkflowCallbacks):
    """A reporter that writes to a file."""

    _out_stream: TextIOWrapper  # 定义类变量 _out_stream，类型为 TextIOWrapper

    def __init__(self, directory: str):
        """Create a new file-based workflow reporter."""
        # 根据给定的目录路径创建目录，如果存在则忽略
        Path(directory).mkdir(parents=True, exist_ok=True)
        # 打开 logs.json 文件，追加模式，使用 UTF-8 编码
        self._out_stream = open(
            Path(directory) / "logs.json", "a", encoding="utf-8"
        )

    def on_error(
        self,
        message: str,
        cause: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ):
        """Handle when an error occurs."""
        # 将错误信息、堆栈信息、错误原因、详细信息写入 _out_stream
        self._out_stream.write(
            json.dumps({
                "type": "error",
                "data": message,
                "stack": stack,
                "source": str(cause),
                "details": details,
            })
            + "\n"
        )
        # 记录错误信息到日志
        message = f"{message} details={details}"
        log.info(message)

    def on_warning(self, message: str, details: dict | None = None):
        """Handle when a warning occurs."""
        # 将警告信息和详细信息写入 _out_stream
        self._out_stream.write(
            json.dumps({"type": "warning", "data": message, "details": details}) + "\n"
        )
        # 调用 _print_warning 函数记录警告信息到日志
        _print_warning(message)

    def on_log(self, message: str, details: dict | None = None):
        """Handle when a log message is produced."""
        # 将日志信息和详细信息写入 _out_stream
        self._out_stream.write(
            json.dumps({"type": "log", "data": message, "details": details}) + "\n"
        )
        # 记录日志信息到日志
        message = f"{message} details={details}"
        log.info(message)


def _print_warning(skk):
    log.warning(skk)
```