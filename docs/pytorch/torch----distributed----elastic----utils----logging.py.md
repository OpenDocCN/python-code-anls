# `.\pytorch\torch\distributed\elastic\utils\logging.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect  # 导入inspect模块，用于获取和处理对象的信息
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，提供了与操作系统交互的功能
import warnings  # 导入warnings模块，用于处理警告信息
from typing import Optional  # 导入Optional类型提示，用于可选类型声明

from torch.distributed.elastic.utils.log_level import get_log_level  # 导入获取日志级别的函数


def get_logger(name: Optional[str] = None):
    """
    Util function to set up a simple logger that writes
    into stderr. The loglevel is fetched from the LOGLEVEL
    env. variable or WARNING as default. The function will use the
    module name of the caller if no name is provided.

    Args:
        name: Name of the logger. If no name provided, the name will
              be derived from the call stack.
    """

    # 根据给定的name或从调用栈中获取的模块名，设置并返回一个简单的日志记录器对象
    return _setup_logger(name or _derive_module_name(depth=2))


def _setup_logger(name: Optional[str] = None):
    # 创建一个日志记录器对象
    logger = logging.getLogger(name)
    # 设置日志记录器对象的日志级别为环境变量LOGLEVEL的值，如果未设置则使用默认的日志级别（通过get_log_level函数获取）
    logger.setLevel(os.environ.get("LOGLEVEL", get_log_level()))
    return logger  # 返回设置好的日志记录器对象


def _derive_module_name(depth: int = 1) -> Optional[str]:
    """
    Derives the name of the caller module from the stack frames.

    Args:
        depth: The position of the frame in the stack.
    """
    try:
        stack = inspect.stack()  # 获取当前调用栈信息
        assert depth < len(stack)
        # 获取指定深度处的栈帧信息
        frame_info = stack[depth]

        module = inspect.getmodule(frame_info[0])  # 获取栈帧所属的模块对象
        if module:
            module_name = module.__name__  # 如果模块对象存在，获取模块名
        else:
            # 如果获取模块对象失败，例如在优化模式下的二进制文件中，使用文件名（去除扩展名）作为模块名
            filename = frame_info[1]
            module_name = os.path.splitext(os.path.basename(filename))[0]
        return module_name  # 返回推导出的模块名
    except Exception as e:
        # 捕获异常并记录警告信息，返回None作为模块名
        warnings.warn(
            f"Error deriving logger module name, using <None>. Exception: {e}",
            RuntimeWarning,
        )
        return None  # 返回None作为模块名
```