# `D:\src\scipysrc\pandas\pandas\util\_exceptions.py`

```
from __future__ import annotations
# 引入在Python 3.10之前使类型提示更清晰的特性

import contextlib
# 导入用于创建上下文管理器的标准库模块

import inspect
# 导入用于检查对象的属性和方法的模块

import os
# 导入操作系统相关功能的标准库模块

import re
# 导入用于正则表达式操作的标准库模块

from typing import TYPE_CHECKING
# 从 typing 模块中导入 TYPE_CHECKING，用于类型检查时的条件导入

import warnings
# 导入用于警告处理的标准库模块

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FrameType
    # 如果是类型检查阶段，则从标准库中导入 Generator 类型和 FrameType 类型


@contextlib.contextmanager
def rewrite_exception(old_name: str, new_name: str) -> Generator[None, None, None]:
    """
    Rewrite the message of an exception.
    """
    try:
        yield
    except Exception as err:
        # 捕获所有异常并进行处理
        if not err.args:
            raise
        # 如果异常没有附带参数，则直接抛出异常
        msg = str(err.args[0])
        # 获取异常的消息，并转换为字符串
        msg = msg.replace(old_name, new_name)
        # 替换异常消息中的旧名称为新名称
        args: tuple[str, ...] = (msg,)
        # 构造新的异常参数元组
        if len(err.args) > 1:
            args = args + err.args[1:]
        err.args = args
        # 更新异常对象的参数
        raise


def find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """
    import pandas as pd
    # 导入 pandas 库

    pkg_dir = os.path.dirname(pd.__file__)
    # 获取 pandas 库的安装目录路径
    test_dir = os.path.join(pkg_dir, "tests")
    # 构造测试目录的路径

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    # 获取当前的调用栈帧
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            # 获取栈帧对应的文件名
            if filename.startswith(pkg_dir) and not filename.startswith(test_dir):
                # 判断文件名是否位于 pandas 安装目录内但不在测试目录内
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # 无论如何都要确保释放栈帧对象
        del frame
    return n
    # 返回找到的栈级别


@contextlib.contextmanager
def rewrite_warning(
    target_message: str,
    target_category: type[Warning],
    new_message: str,
    new_category: type[Warning] | None = None,
) -> Generator[None, None, None]:
    """
    Rewrite the message of a warning.

    Parameters
    ----------
    target_message : str
        Warning message to match.
    target_category : Warning
        Warning type to match.
    new_message : str
        New warning message to emit.
    new_category : Warning or None, default None
        New warning type to emit. When None, will be the same as target_category.
    """
    if new_category is None:
        new_category = target_category
    # 如果未指定新的警告类型，则使用目标警告类型

    with warnings.catch_warnings(record=True) as record:
        yield
    # 使用警告记录器记录所有警告信息

    if len(record) > 0:
        match = re.compile(target_message)
        # 编译目标警告消息为正则表达式

        for warning in record:
            if warning.category is target_category and re.search(
                match, str(warning.message)
            ):
                category = new_category
                message: Warning | str = new_message
            else:
                category, message = warning.category, warning.message
            # 根据条件选择要发出的新警告类型和消息

            warnings.warn_explicit(
                message=message,
                category=category,
                filename=warning.filename,
                lineno=warning.lineno,
            )
            # 发出新警告，保留原警告的位置信息
```