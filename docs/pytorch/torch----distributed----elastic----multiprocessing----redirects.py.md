# `.\pytorch\torch\distributed\elastic\multiprocessing\redirects.py`

```py
"""
# mypy: allow-untyped-defs
# !/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Taken and modified from original source:
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial

# 判断操作系统类型
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

# 设置日志记录器
logger = logging.getLogger(__name__)

# 获取 libc 库，用于控制 C 标准输出
def get_libc():
    # 如果是 Windows 或者 MacOS，给出警告并返回 None
    if IS_WINDOWS or IS_MACOS:
        logger.warning(
            "NOTE: Redirects are currently not supported in Windows or MacOs."
        )
        return None
    else:
        # 否则加载 libc.so.6 库并返回
        return ctypes.CDLL("libc.so.6")

# 获取 libc 对象
libc = get_libc()

# 获取 C 标准输出流
def _c_std(stream: str):
    return ctypes.c_void_p.in_dll(libc, stream)

# 获取 Python 标准输出流
def _python_std(stream: str):
    return {"stdout": sys.stdout, "stderr": sys.stderr}[stream]

# 支持的标准输出类型
_VALID_STD = {"stdout", "stderr"}

# 上下文管理器，用于重定向标准输出或错误到文件
@contextmanager
def redirect(std: str, to_file: str):
    """
    Redirect ``std`` (one of ``"stdout"`` or ``"stderr"``) to a file in the path specified by ``to_file``.

    This method redirects the underlying std file descriptor (not just python's ``sys.stdout|stderr``).
    See usage for details.

    Directory of ``dst_filename`` is assumed to exist and the destination file
    is overwritten if it already exists.

    .. note:: Due to buffering cross source writes are not guaranteed to
              appear in wall-clock order. For instance in the example below
              it is possible for the C-outputs to appear before the python
              outputs in the log file.

    Usage:

    ::

     # syntactic-sugar for redirect("stdout", "tmp/stdout.log")
     with redirect_stdout("/tmp/stdout.log"):
        print("python stdouts are redirected")
        libc = ctypes.CDLL("libc.so.6")
        libc.printf(b"c stdouts are also redirected"
        os.system("echo system stdouts are also redirected")

     print("stdout restored")

    """
    # 检查标准输出类型是否合法
    if std not in _VALID_STD:
        raise ValueError(
            f"unknown standard stream <{std}>, must be one of {_VALID_STD}"
        )

    # 获取 C 标准输出流和 Python 标准输出流的文件描述符
    c_std = _c_std(std)
    python_std = _python_std(std)
    std_fd = python_std.fileno()

    # 内部函数，用于实际的重定向操作
    def _redirect(dst):
        libc.fflush(c_std)  # 刷新 C 标准输出
        python_std.flush()  # 刷新 Python 标准输出
        os.dup2(dst.fileno(), std_fd)  # 将目标文件描述符复制到标准输出文件描述符上

    # 使用副本打开当前标准输出流，并以写入二进制模式打开目标文件
    with os.fdopen(os.dup(std_fd)) as orig_std, open(to_file, mode="w+b") as dst:
        _redirect(dst)  # 执行重定向操作
        try:
            yield  # 执行被重定向的代码块
        finally:
            _redirect(orig_std)  # 恢复原始的标准输出流

# 使用偏函数创建更方便的函数 redirect_stdout 和 redirect_stderr
redirect_stdout = partial(redirect, "stdout")
redirect_stderr = partial(redirect, "stderr")
"""
```