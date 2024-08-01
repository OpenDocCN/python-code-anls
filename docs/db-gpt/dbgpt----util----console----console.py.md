# `.\DB-GPT-src\dbgpt\util\console\console.py`

```py
"""Console utility functions for CLI."""
# 导入必要的模块和类
import dataclasses
import sys
from functools import lru_cache
from typing import Any

# 导入丰富输出相关的类和函数
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.theme import Theme


@dataclasses.dataclass
class Output:
    """Output file."""
    # 定义一个数据类用于表示输出文件，包含标题和文件路径两个字段
    title: str
    file: str


def _get_theme():
    # 返回一个自定义主题对象
    return Theme(
        {
            "success": "green",
            "info": "bright_blue",
            "warning": "bright_yellow",
            "error": "red",
        }
    )


@lru_cache(maxsize=None)
def get_console(output: Output | None = None) -> Console:
    # 返回一个控制台对象，如果指定了输出对象，则将输出流设置为该对象的文件流
    return Console(
        force_terminal=True,
        color_system="standard",
        theme=_get_theme(),
        file=output.file if output else None,
    )


class CliLogger:
    def __init__(self, output: Output | None = None):
        # 初始化时创建一个控制台对象，用于日志输出
        self.console = get_console(output)

    def success(self, msg: str, **kwargs):
        # 打印成功信息，使用绿色字体
        self.console.print(f"[success]{msg}[/]", **kwargs)

    def info(self, msg: str, **kwargs):
        # 打印信息，使用亮蓝色字体
        self.console.print(f"[info]{msg}[/]", **kwargs)

    def warning(self, msg: str, **kwargs):
        # 打印警告信息，使用亮黄色字体
        self.console.print(f"[warning]{msg}[/]", **kwargs)

    def error(self, msg: str, exit_code: int = 0, **kwargs):
        # 打印错误信息，使用红色字体，并根据指定的退出码决定是否退出程序
        self.console.print(f"[error]{msg}[/]", **kwargs)
        if exit_code != 0:
            sys.exit(exit_code)

    def debug(self, msg: str, **kwargs):
        # 打印调试信息，使用青色字体
        self.console.print(f"[cyan]{msg}[/]", **kwargs)

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", **kwargs):
        # 普通打印函数，接受多个对象并打印到控制台
        self.console.print(*objects, sep=sep, end=end, **kwargs)

    def markdown(self, msg: str, **kwargs):
        # 将输入的 Markdown 格式文本渲染后打印到控制台
        md = Markdown(msg)
        self.console.print(md, **kwargs)

    def ask(self, msg: str, **kwargs):
        # 使用富文本方式向用户提问，并等待用户输入回答
        return Prompt.ask(msg, **kwargs)
```