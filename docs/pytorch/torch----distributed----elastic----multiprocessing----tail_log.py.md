# `.\pytorch\torch\distributed\elastic\multiprocessing\tail_log.py`

```py
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging  # 导入日志模块
import os  # 导入操作系统接口模块
import time  # 导入时间模块
from concurrent.futures.thread import ThreadPoolExecutor  # 导入线程池执行器
from threading import Event  # 导入事件对象
from typing import Dict, List, Optional, TextIO, TYPE_CHECKING  # 导入类型注解相关模块

if TYPE_CHECKING:
    from concurrent.futures._base import Future  # 导入Future类型的类型注解

__all__ = ["tail_logfile", "TailLog"]  # 模块公开的接口列表

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


def tail_logfile(
    header: str, file: str, dst: TextIO, finished: Event, interval_sec: float
):
    # 循环等待文件存在，直到文件被创建或者结束事件被设置
    while not os.path.exists(file):
        if finished.is_set():
            return
        time.sleep(interval_sec)

    # 打开文件，使用替换错误的方式处理异常字符，逐行读取文件内容
    with open(file, errors="replace") as fp:
        while True:
            line = fp.readline()

            if line:
                dst.write(f"{header}{line}")  # 将带有指定头部的行写入目标文件对象
            else:  # 已到达文件末尾
                if finished.is_set():
                    # 日志行生成器已完成
                    break
                else:
                    # 日志行生成器仍在运行
                    # 在下一次循环前等待一段时间
                    time.sleep(interval_sec)


class TailLog:
    """
    Tail the given log files.

    The log files do not have to exist when the ``start()`` method is called. The tail-er will gracefully wait until
    the log files are created by the producer and will tail the contents of the
    log files until the ``stop()`` method is called.

    .. warning:: ``TailLog`` will wait indefinitely for the log file to be created!

    Each log file's line will be suffixed with a header of the form: ``[{name}{idx}]:``,
    where the ``name`` is user-provided and ``idx`` is the index of the log file
    in the ``log_files`` mapping. ``log_line_prefixes`` can be used to override the
    header for each log file.

    Usage:

    ::

     log_files = {0: "/tmp/0_stdout.log", 1: "/tmp/1_stdout.log"}
     tailer = TailLog("trainer", log_files, sys.stdout).start()
     # actually run the trainers to produce 0_stdout.log and 1_stdout.log
     run_trainers()
     tailer.stop()

     # once run_trainers() start writing the ##_stdout.log files
     # the tailer will print to sys.stdout:
     # >>> [trainer0]:log_line1
     # >>> [trainer1]:log_line1
     # >>> [trainer0]:log_line2
     # >>> [trainer0]:log_line3
     # >>> [trainer1]:log_line2

    .. note:: Due to buffering log lines between files may not necessarily
              be printed out in order. You should configure your application's
              logger to suffix each log line with a proper timestamp.

    """

    def __init__(
        self,
        name: str,
        log_files: Dict[int, str],
        dst: TextIO,
        log_line_prefixes: Optional[Dict[int, str]] = None,
        interval_sec: float = 0.1,
    ):
        """
        Initialize the TailLog instance.

        Args:
            name (str): The name prefix for log lines.
            log_files (Dict[int, str]): A mapping of log file indices to their paths.
            dst (TextIO): The destination where merged log lines will be written.
            log_line_prefixes (Optional[Dict[int, str]], optional): Mapping to override default log line prefixes.
            interval_sec (float, optional): Interval in seconds to check for new log lines.
        """
        self.name = name
        self.log_files = log_files
        self.dst = dst
        self.log_line_prefixes = log_line_prefixes or {}
        self.interval_sec = interval_sec
    ):
        # 获取日志文件数量
        n = len(log_files)
        # 初始化线程池对象，如果有日志文件则创建对应数量的线程池
        self._threadpool = None
        if n > 0:
            self._threadpool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix=f"{self.__class__.__qualname__}_{name}",
            )

        # 设置对象属性
        self._name = name  # 设置日志尾追器的名称
        self._dst = dst  # 设置日志输出目标
        self._log_files = log_files  # 设置日志文件字典
        self._log_line_prefixes = log_line_prefixes  # 设置日志行前缀字典
        # 创建完成事件字典，每个日志文件对应一个事件对象
        self._finished_events: Dict[int, Event] = {
            local_rank: Event() for local_rank in log_files.keys()
        }
        self._futs: List[Future] = []  # 初始化任务 Future 列表
        self._interval_sec = interval_sec  # 设置检查间隔秒数
        self._stopped = False  # 初始化停止状态为 False

    def start(self) -> "TailLog":
        # 如果没有线程池，则直接返回对象自身
        if not self._threadpool:
            return self

        # 遍历日志文件字典，为每个文件创建尾追任务
        for local_rank, file in self._log_files.items():
            header = f"[{self._name}{local_rank}]:"
            # 如果指定了日志行前缀，则使用该前缀
            if self._log_line_prefixes and local_rank in self._log_line_prefixes:
                header = self._log_line_prefixes[local_rank]
            # 提交尾追日志文件的任务到线程池，并将 Future 对象加入列表
            self._futs.append(
                self._threadpool.submit(
                    tail_logfile,
                    header=header,
                    file=file,
                    dst=self._dst,
                    finished=self._finished_events[local_rank],
                    interval_sec=self._interval_sec,
                )
            )
        return self

    def stop(self) -> None:
        # 设置所有完成事件为已触发状态
        for finished in self._finished_events.values():
            finished.set()

        # 等待所有尾追任务完成，如果有异常则记录日志
        for local_rank, f in enumerate(self._futs):
            try:
                f.result()
            except Exception as e:
                logger.error(
                    "error in log tailor for %s%s. %s: %s",
                    self._name,
                    local_rank,
                    e.__class__.__qualname__,
                    e,
                )

        # 如果有线程池，则关闭线程池
        if self._threadpool:
            self._threadpool.shutdown(wait=True)

        # 设置停止状态为 True
        self._stopped = True

    def stopped(self) -> bool:
        # 返回当前停止状态
        return self._stopped
```