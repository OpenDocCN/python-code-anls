# `.\pytorch\torch\distributed\elastic\timer\api.py`

```py
# mypy: allow-untyped-defs
# 上面的注释指定了允许未类型化的函数定义，这对于类型检查工具很有用
# Copyright (c) Facebook, Inc. and its affiliates.
# 版权声明，指明代码的版权归属和保留情况
# All rights reserved.
# 版权声明结束
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# 指明此源代码使用 BSD 风格许可证，许可证文件位于根目录下的 LICENSE 文件中
import abc
# 导入抽象基类模块
import logging
# 导入日志模块
import threading
# 导入线程模块
import time
# 导入时间模块
from contextlib import contextmanager
# 从 contextlib 模块导入上下文管理器
from inspect import getframeinfo, stack
# 从 inspect 模块导入获取调用栈信息的函数和栈信息函数
from typing import Any, Dict, List, Optional, Set
# 导入类型提示相关的类和函数


__all__ = [
    "TimerRequest",
    "TimerClient",
    "RequestQueue",
    "TimerServer",
    "configure",
    "expires",
]
# 定义了模块中可以被导入的公共接口列表

logger = logging.getLogger(__name__)
# 创建了一个名为 __name__ 的 logger 对象，用于记录当前模块的日志信息


class TimerRequest:
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``TimerClient`` and ``TimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.

    .. note:: the type of ``worker_id`` is implementation specific.
              It is whatever the TimerServer and TimerClient implementations
              have on to uniquely identify a worker.
    """
    # 表示倒计时定时器获取和释放的数据对象，用于 TimerClient 和 TimerServer 之间的通信

    __slots__ = ["worker_id", "scope_id", "expiration_time"]
    # 使用 __slots__ 来优化内存使用，限制实例可以拥有的属性

    def __init__(self, worker_id: Any, scope_id: str, expiration_time: float):
        # 初始化方法，接受 worker_id、scope_id 和 expiration_time 作为参数
        self.worker_id = worker_id
        self.scope_id = scope_id
        self.expiration_time = expiration_time

    def __eq__(self, other):
        # 自定义的相等比较方法，用于比较两个 TimerRequest 对象是否相等
        if isinstance(other, TimerRequest):
            return (
                self.worker_id == other.worker_id
                and self.scope_id == other.scope_id
                and self.expiration_time == other.expiration_time
            )
        return False


class TimerClient(abc.ABC):
    """
    Client library to acquire and release countdown timers by communicating
    with the TimerServer.
    """
    # 用于与 TimerServer 通信以获取和释放倒计时定时器的客户端库

    @abc.abstractmethod
    def acquire(self, scope_id: str, expiration_time: float) -> None:
        """
        Acquires a timer for the worker that holds this client object
        given the scope_id and expiration_time. Typically registers
        the timer with the TimerServer.
        """
        pass
        # 抽象方法，用于获取给定 scope_id 和 expiration_time 的计时器，通常在 TimerServer 上注册计时器

    @abc.abstractmethod
    def release(self, scope_id: str):
        """
        Releases the timer for the ``scope_id`` on the worker this
        client represents. After this method is
        called, the countdown timer on the scope is no longer in effect.
        """
        pass
        # 抽象方法，释放该客户端表示的 worker 上的 scope_id 的计时器，调用此方法后，该作用域上的倒计时器将不再生效


class RequestQueue(abc.ABC):
    """
    Consumer queue holding timer acquisition/release requests
    """
    # 用于保存计时器获取/释放请求的消费者队列

    @abc.abstractmethod
    # 抽象方法声明，用于定义一个抽象方法
    # 定义一个方法，返回当前队列的大小
    def size(self) -> int:
        """
        Returns the size of the queue at the time this method is called.
        Note that by the time ``get`` is called the size of the queue
        may have increased. The size of the queue should not decrease
        until the ``get`` method is called. That is, the following assertion
        should hold:

        size = q.size()
        res = q.get(size, timeout=0)
        assert size == len(res)

        -- or --

        size = q.size()
        res = q.get(size * 2, timeout=1)
        assert size <= len(res) <= size * 2
        """
        pass

    @abc.abstractmethod
    def get(self, size: int, timeout: float) -> List[TimerRequest]:
        """
        Gets up to ``size`` number of timer requests in a blocking fashion
        (no more than ``timeout`` seconds).
        """
        pass
class TimerServer(abc.ABC):
    """
    Entity that monitors active timers and expires them
    in a timely fashion. This server is responsible for
    reaping workers that have expired timers.
    """

    def __init__(
        self, request_queue: RequestQueue, max_interval: float, daemon: bool = True
    ):
        """
        :param request_queue: Consumer ``RequestQueue`` - 用于存放请求的队列
        :param max_interval: max time (in seconds) to wait
                             for an item in the request_queue - 在请求队列中等待的最大时间（秒）
        :param daemon: whether to run the watchdog thread as a daemon - 是否将看门狗线程设置为守护线程
        """
        super().__init__()
        self._request_queue = request_queue  # 存储请求队列的实例
        self._max_interval = max_interval  # 存储最大等待时间
        self._daemon = daemon  # 存储是否为守护线程
        self._watchdog_thread: Optional[threading.Thread] = None  # 看门狗线程的引用
        self._stop_signaled = False  # 标记是否收到停止信号

    @abc.abstractmethod
    def register_timers(self, timer_requests: List[TimerRequest]) -> None:
        """
        Processes the incoming timer requests and registers them with the server.
        The timer request can either be a acquire-timer or release-timer request.
        Timer requests with a negative expiration_time should be interpreted
        as a release-timer request.
        """
        pass

    @abc.abstractmethod
    def clear_timers(self, worker_ids: Set[Any]) -> None:
        """
        Clears all timers for the given ``worker_ids``.
        """
        pass

    @abc.abstractmethod
    def get_expired_timers(self, deadline: float) -> Dict[str, List[TimerRequest]]:
        """
        Returns all expired timers for each worker_id. An expired timer
        is a timer for which the expiration_time is less than or equal to
        the provided deadline.
        """
        pass

    @abc.abstractmethod
    def _reap_worker(self, worker_id: Any) -> bool:
        """
        Reaps the given worker. Returns True if the worker has been
        successfully reaped, False otherwise. If any uncaught exception
        is thrown from this method, the worker is considered reaped
        and all associated timers will be removed.
        """

    def _reap_worker_no_throw(self, worker_id: Any) -> bool:
        """
        Wraps ``_reap_worker(worker_id)``, if an uncaught exception is
        thrown, then it considers the worker as reaped.
        """
        try:
            return self._reap_worker(worker_id)  # 尝试执行 _reap_worker(worker_id)
        except Exception:
            logger.exception(
                "Uncaught exception thrown from _reap_worker(), "
                "check that the implementation correctly catches exceptions",
            )
            return True

    def _watchdog_loop(self):
        while not self._stop_signaled:  # 循环直到收到停止信号
            try:
                self._run_watchdog()  # 运行看门狗任务
            except Exception:
                logger.exception("Error running watchdog")  # 记录异常信息
    # 在运行监控任务时，确定批处理大小，至少为1，并考虑请求队列的当前大小
    batch_size = max(1, self._request_queue.size())
    # 从请求队列获取一批定时器请求，并指定最大等待时间
    timer_requests = self._request_queue.get(batch_size, self._max_interval)
    # 注册这些定时器请求
    self.register_timers(timer_requests)
    # 获取当前时间戳
    now = time.time()
    # 存储已经被清除的工作线程 ID 集合
    reaped_worker_ids = set()
    # 对于每个过期的定时器，获取相关的工作线程 ID 和过期定时器列表
    for worker_id, expired_timers in self.get_expired_timers(now).items():
        # 记录信息日志，指明正在清除的工作线程 ID 和过期定时器的作用域
        logger.info(
            "Reaping worker_id=[%s]." " Expired timers: %s",
            worker_id,
            self._get_scopes(expired_timers),
        )
        # 尝试清除指定的工作线程，并记录操作成功的信息
        if self._reap_worker_no_throw(worker_id):
            logger.info("Successfully reaped worker=[%s]", worker_id)
            # 将成功清除的工作线程 ID 加入集合
            reaped_worker_ids.add(worker_id)
        else:
            # 若清除失败，记录错误日志，并打算在下次监控任务中重试
            logger.error(
                "Error reaping worker=[%s]. Will retry on next watchdog.", worker_id
            )
    # 清除所有已被重新分配或清除的定时器
    self.clear_timers(reaped_worker_ids)

    # 返回一组定时器请求的作用域 ID 列表
def _get_scopes(self, timer_requests):
        return [r.scope_id for r in timer_requests]

    # 启动当前实例的操作，包括记录日志信息和启动守护线程
    logger.info(
        "Starting %s..." " max_interval=%s," " daemon=%s",
        type(self).__name__,
        self._max_interval,
        self._daemon,
    )
    # 创建并启动监控线程
    self._watchdog_thread = threading.Thread(
        target=self._watchdog_loop, daemon=self._daemon
    )
    logger.info("Starting watchdog thread...")
    self._watchdog_thread.start()

    # 停止当前实例的操作，包括设置停止信号和等待守护线程结束
    logger.info("Stopping %s", type(self).__name__)
    self._stop_signaled = True
    if self._watchdog_thread:
        logger.info("Stopping watchdog thread...")
        # 等待守护线程结束，指定最长等待时间为 _max_interval
        self._watchdog_thread.join(self._max_interval)
        self._watchdog_thread = None
    else:
        # 若没有守护线程在运行，则记录日志并不执行任何操作
        logger.info("No watchdog thread running, doing nothing")
# 可选类型的全局变量，用于存储计时器客户端对象
_timer_client: Optional[TimerClient] = None

# 配置计时器客户端，必须在使用 expires 前调用
def configure(timer_client: TimerClient):
    """
    配置计时器客户端。必须在使用 expires 前调用。
    """
    global _timer_client
    _timer_client = timer_client
    # 记录日志，指示已配置计时器客户端的类型
    logger.info("Timer client configured to: %s", type(_timer_client).__name__)


# 上下文管理器，获取一个在指定时间后过期的倒计时器
def expires(
    after: float, scope: Optional[str] = None, client: Optional[TimerClient] = None
):
    """
    获取一个在 ``after`` 秒后过期的倒计时器。除非其包裹的代码块在指定时间内完成，否则计时器将过期。
    当计时器过期时，该工作进程有资格被收回。"收回"的确切含义取决于客户端的实现。在大多数情况下，收回意味着终止工作进程。
    注意，工作进程不保证会在确切的 ``time.now() + after`` 时被收回，而是在此时变得"有资格"被收回，
    最终决定何时以及如何收回带有过期计时器的工作进程的是客户端所连接的 TimerServer。

    用法::

        torch.distributed.elastic.timer.configure(LocalTimerClient())
        with expires(after=10):
            torch.distributed.all_reduce(...)
    """
    # 如果未提供客户端，则使用全局配置的计时器客户端
    if client is None:
        if _timer_client is None:
            raise RuntimeError("Configure timer client before using countdown timers.")
        client = _timer_client
    # 如果未提供作用域，则获取调用者的文件名和行号
    if scope is None:
        caller = getframeinfo(stack()[1][0])
        scope = f"{caller.filename}#{caller.lineno}"
    # 计算过期时间
    expiration = time.time() + after
    # 向客户端申请计时器
    client.acquire(scope, expiration)
    try:
        yield
    finally:
        # 释放计时器
        client.release(scope)
```