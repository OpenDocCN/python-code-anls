# `.\pytorch\torch\distributed\elastic\multiprocessing\errors\error_handler.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库和模块
import faulthandler  # 引入 faulthandler 库，用于处理程序中的异常
import json  # 引入 json 库，用于 JSON 数据的处理
import logging  # 引入 logging 库，用于记录日志
import os  # 引入 os 库，用于操作系统相关功能
import time  # 引入 time 库，用于时间相关操作
import traceback  # 引入 traceback 库，用于获取异常的堆栈信息
import warnings  # 引入 warnings 库，用于处理警告信息
from typing import Any, Dict, Optional  # 引入 typing 库中的类型定义

# 定义导出的模块变量
__all__ = ["ErrorHandler"]

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


# 定义错误处理器类
class ErrorHandler:
    """
    Write the provided exception object along with some other metadata about
    the error in a structured way in JSON format to an error file specified by the
    environment variable: ``TORCHELASTIC_ERROR_FILE``. If this environment
    variable is not set, then simply logs the contents of what would have been
    written to the error file.

    This handler may be subclassed to customize the handling of the error.
    Subclasses should override ``initialize()`` and ``record_exception()``.
    """

    def _get_error_file_path(self) -> Optional[str]:
        """
        Return the error file path.

        May return ``None`` to have the structured error be logged only.
        """
        # 返回环境变量 TORCHELASTIC_ERROR_FILE 对应的文件路径，如果未设置则返回 None
        return os.environ.get("TORCHELASTIC_ERROR_FILE", None)

    def initialize(self) -> None:
        """
        Call prior to running code that we wish to capture errors/exceptions.

        Typically registers signal/fault handlers. Users can override this
        function to add custom initialization/registrations that aid in
        propagation/information of errors/signals/exceptions/faults.
        """
        try:
            # 尝试启用所有线程的故障处理器
            faulthandler.enable(all_threads=True)
        except Exception as e:
            # 如果失败，记录警告信息
            warnings.warn(f"Unable to enable fault handler. {type(e).__name__}: {e}")

    def _write_error_file(self, file_path: str, error_msg: str) -> None:
        """Write error message to the file."""
        try:
            # 尝试将错误信息写入指定的文件中
            with open(file_path, "w") as fp:
                fp.write(error_msg)
        except Exception as e:
            # 如果写入失败，记录警告信息
            warnings.warn(f"Unable to write error to file. {type(e).__name__}: {e}")

    def record_exception(self, e: BaseException) -> None:
        """
        Write a structured information about the exception into an error file in JSON format.

        If the error file cannot be determined, then logs the content
        that would have been written to the error file.
        """
        # 获取错误文件的路径
        file = self._get_error_file_path()
        if file:
            # 构造错误信息的结构化 JSON 数据
            data = {
                "message": {
                    "message": f"{type(e).__name__}: {e}",
                    "extraInfo": {
                        "py_callstack": traceback.format_exc(),
                        "timestamp": str(int(time.time())),
                    },
                }
            }
            # 将数据写入错误文件
            with open(file, "w") as fp:
                json.dump(data, fp)
    def override_error_code_in_rootcause_data(
        self,
        rootcause_error_file: str,
        rootcause_error: Dict[str, Any],
        error_code: int = 0,
    ):
        """Modify the rootcause_error read from the file, to correctly set the exit code."""
        # 检查 rootcause_error 中是否包含 "message" 字段
        if "message" not in rootcause_error:
            # 如果缺少 "message" 字段，记录警告日志并提示无法覆盖错误码
            logger.warning(
                "child error file (%s) does not have field `message`. \n"
                "cannot override error code: %s",
                rootcause_error_file,
                error_code,
            )
        # 如果 "message" 字段存在且为字符串类型
        elif isinstance(rootcause_error["message"], str):
            # 如果 "message" 字段为字符串类型，记录警告日志并提示跳过错误码覆盖
            logger.warning(
                "child error file (%s) has a new message format. \n"
                "skipping error code override",
                rootcause_error_file,
            )
        else:
            # 否则，将错误码写入 rootcause_error 中的 "message" 字段的 "errorCode" 键
            rootcause_error["message"]["errorCode"] = error_code

    def dump_error_file(self, rootcause_error_file: str, error_code: int = 0):
        """Dump parent error file from child process's root cause error and error code."""
        # 打开指定路径的错误文件，并加载其内容为 JSON 格式
        with open(rootcause_error_file) as fp:
            rootcause_error = json.load(fp)
            # 覆盖错误码，用于处理由于信号（如 SIGSEGV）导致的子进程终止无法捕获错误码的情况
            if error_code:
                self.override_error_code_in_rootcause_data(
                    rootcause_error_file, rootcause_error, error_code
                )
            # 记录调试日志，输出 rootcause_error 的内容，格式化输出为缩进为 2 的 JSON 字符串
            logger.debug(
                "child error file (%s) contents:\n" "%s",
                rootcause_error_file,
                json.dumps(rootcause_error, indent=2),
            )

        # 获取父进程的错误文件路径
        my_error_file = self._get_error_file_path()
        if my_error_file:
            # 如果存在父进程的错误文件路径
            # 防止已存在的错误文件被覆盖
            # 当使用 multiprocessing 创建子进程时，可能会出现这种情况
            # 如果子进程在包装函数执行之前接收到信号，信号处理程序会写入错误文件
            # 此时子进程会写入父进程的错误文件，记录警告日志并覆盖错误文件
            self._rm(my_error_file)
            self._write_error_file(my_error_file, json.dumps(rootcause_error))
            logger.info("dumped error file to parent's %s", my_error_file)
        else:
            # 如果未定义父进程的错误文件路径，记录错误日志
            logger.error(
                "no error file defined for parent, to copy child error file (%s)",
                rootcause_error_file,
            )
    def _rm(self, my_error_file):
        # 检查给定路径是否是一个文件
        if os.path.isfile(my_error_file):
            # 记录原始文件的内容到日志
            with open(my_error_file) as fp:
                try:
                    # 尝试加载并转换文件内容为格式化的 JSON 字符串
                    original = json.dumps(json.load(fp), indent=2)
                    # 记录警告信息，包括文件已存在并将被覆盖，以及原始内容
                    logger.warning(
                        "%s already exists"
                        " and will be overwritten."
                        " Original contents:\n%s",
                        my_error_file,
                        original,
                    )
                except json.decoder.JSONDecodeError:
                    # 如果加载文件内容失败，记录警告信息，文件已存在并将被覆盖，但无法加载原始内容
                    logger.warning(
                        "%s already exists"
                        " and will be overwritten."
                        " Unable to load original contents:\n",
                        my_error_file,
                    )
            # 删除文件
            os.remove(my_error_file)
```