# `.\pytorch\torch\distributed\elastic\timer\local_timer.py`

```
# mypy: allow-untyped-defs
# 允许未经类型注解的定义，用于静态类型检查工具
# Copyright (c) Facebook, Inc. and its affiliates.
# 版权声明，保留所有权利
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# 此源代码采用 BSD 风格许可证授权，许可证文件位于根目录下

import logging
# 导入日志记录模块
import multiprocessing as mp
# 导入多进程模块
import os
# 导入操作系统相关功能模块
import signal
# 导入信号处理模块
import time
# 导入时间模块
from queue import Empty
# 从队列模块导入 Empty 异常
from typing import Any, Dict, List, Set, Tuple
# 导入类型注解相关模块

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer
# 从当前包的 api 模块导入 RequestQueue, TimerClient, TimerRequest, TimerServer

__all__ = ["LocalTimerClient", "MultiprocessingRequestQueue", "LocalTimerServer"]
# 模块的公共接口，包括 LocalTimerClient, MultiprocessingRequestQueue, LocalTimerServer

logger = logging.getLogger(__name__)
# 获取当前模块的日志记录器对象

class LocalTimerClient(TimerClient):
    """
    Client side of ``LocalTimerServer``. This client is meant to be used
    on the same host that the ``LocalTimerServer`` is running on and uses
    pid to uniquely identify a worker. This is particularly useful in situations
    where one spawns a subprocess (trainer) per GPU on a host with multiple
    GPU devices.
    """
    # 本地定时器客户端，继承自 TimerClient 类

    def __init__(self, mp_queue):
        super().__init__()
        self._mp_queue = mp_queue
        # 初始化方法，接收一个 multiprocessing.Queue 对象作为参数，用于通信

    def acquire(self, scope_id, expiration_time):
        pid = os.getpid()
        # 获取当前进程的 PID
        acquire_request = TimerRequest(pid, scope_id, expiration_time)
        # 创建一个 TimerRequest 对象，表示获取资源的请求
        self._mp_queue.put(acquire_request)
        # 将请求放入 multiprocessing.Queue 中

    def release(self, scope_id):
        pid = os.getpid()
        # 获取当前进程的 PID
        release_request = TimerRequest(pid, scope_id, -1)
        # 创建一个 TimerRequest 对象，表示释放资源的请求
        self._mp_queue.put(release_request)
        # 将请求放入 multiprocessing.Queue 中

class MultiprocessingRequestQueue(RequestQueue):
    """
    A ``RequestQueue`` backed by python ``multiprocessing.Queue``
    """
    # 使用 Python multiprocessing.Queue 支持的请求队列

    def __init__(self, mp_queue: mp.Queue):
        super().__init__()
        self._mp_queue = mp_queue
        # 初始化方法，接收一个 multiprocessing.Queue 对象作为参数，用于存储请求

    def size(self) -> int:
        return self._mp_queue.qsize()
        # 返回队列中当前的请求数量

    def get(self, size, timeout: float) -> List[TimerRequest]:
        requests = []
        wait = timeout
        # 初始化请求列表和等待时间

        for _ in range(0, size):
            start = time.time()
            # 记录开始时间

            try:
                r = self._mp_queue.get(block=True, timeout=wait)
                # 从 multiprocessing.Queue 中获取请求，阻塞直到有请求或超时
            except Empty:
                break
                # 如果队列为空，结束获取请求的操作

            requests.append(r)
            # 将获取到的请求加入到列表中
            wait = wait - (time.time() - start)
            # 计算剩余等待时间
            if wait <= 0:
                break
                # 如果剩余等待时间小于等于 0，结束获取请求的操作

        return requests
        # 返回获取到的请求列表

class LocalTimerServer(TimerServer):
    """
    Server that works with ``LocalTimerClient``. Clients are expected to be
    subprocesses to the parent process that is running this server. Each host
    in the job is expected to start its own timer server locally and each
    server instance manages timers for local workers (running on processes
    on the same host).
    """
    # 本地定时器服务器，与 LocalTimerClient 配合使用

    def __init__(
        self, mp_queue: mp.Queue, max_interval: float = 60, daemon: bool = True
    ):
        super().__init__(MultiprocessingRequestQueue(mp_queue), max_interval, daemon)
        # 初始化方法，接收一个 multiprocessing.Queue 对象作为参数，并调用父类的初始化方法

        self._timers: Dict[Tuple[Any, str], TimerRequest] = {}
        # 初始化定时器字典，用于存储定时器相关信息
    # 注册定时器请求，将每个请求加入到定时器字典中
    def register_timers(self, timer_requests: List[TimerRequest]) -> None:
        for request in timer_requests:
            pid = request.worker_id
            scope_id = request.scope_id
            expiration_time = request.expiration_time

            # 如果过期时间小于0，表示释放调用，从定时器中移除对应的请求
            if expiration_time < 0:
                self._timers.pop((pid, scope_id), None)
            else:
                # 否则将请求加入到定时器字典中
                self._timers[(pid, scope_id)] = request

    # 清除特定工作进程ID对应的所有定时器请求
    def clear_timers(self, worker_ids: Set[int]) -> None:
        for pid, scope_id in list(self._timers.keys()):
            if pid in worker_ids:
                # 如果工作进程ID在给定的工作进程ID集合中，则从定时器字典中移除该请求
                self._timers.pop((pid, scope_id))

    # 获取所有已过期的定时器请求，以字典形式返回
    def get_expired_timers(self, deadline: float) -> Dict[Any, List[TimerRequest]]:
        # pid -> [timer_requests...]
        expired_timers: Dict[Any, List[TimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                # 如果请求的过期时间早于等于截止时间，则将其加入到已过期定时器字典中
                expired_scopes = expired_timers.setdefault(request.worker_id, [])
                expired_scopes.append(request)
        return expired_timers

    # 终止指定工作进程
    def _reap_worker(self, worker_id: int) -> bool:
        try:
            # 尝试发送 SIGKILL 信号给指定工作进程ID
            os.kill(worker_id, signal.SIGKILL)
            return True
        except ProcessLookupError:
            # 如果进程不存在，则记录日志并跳过
            logger.info("Process with pid=%s does not exist. Skipping", worker_id)
            return True
        except Exception:
            # 捕获其他异常情况，记录错误日志
            logger.exception("Error terminating pid=%s", worker_id)
        return False
```