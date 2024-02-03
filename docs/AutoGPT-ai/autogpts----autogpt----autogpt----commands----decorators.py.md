# `.\AutoGPT\autogpts\autogpt\autogpt\commands\decorators.py`

```py
# 导入 functools 模块，提供高阶函数和操作工具
# 导入 logging 模块，用于记录日志信息
# 导入 re 模块，用于正则表达式操作
# 从 pathlib 模块中导入 Path 类，用于处理文件路径
# 从 typing 模块中导入 Callable、ParamSpec、TypeVar 类型
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar
import functools
import logging
import re

# 从 autogpt.agents.agent 模块中导入 Agent 类
from autogpt.agents.agent import Agent

# 定义 ParamSpec 类型参数 P
P = ParamSpec("P")
# 定义 TypeVar 类型参数 T
T = TypeVar("T")

# 获取当前模块的 logger 对象
logger = logging.getLogger(__name__)

# 定义装饰器函数 sanitize_path_arg，用于处理指定路径参数并将其解析为 Path 对象
def sanitize_path_arg(
    arg_name: str, make_relative: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Sanitizes the specified path (str | Path) argument, resolving it to a Path"""

    # 返回装饰器函数 decorator
    return decorator
```