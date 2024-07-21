# `.\pytorch\torch\distributed\elastic\timer\file_based_local_timer.py`

```
import io  # 导入 io 模块，用于处理文件流
import json  # 导入 json 模块，用于 JSON 数据的编码和解码
import os  # 导入 os 模块，提供了访问操作系统服务的功能
import select  # 导入 select 模块，用于在多个文件对象上进行 I/O 多路复用
import signal  # 导入 signal 模块，用于处理信号
import sys  # 导入 sys 模块，提供了对 Python 解释器的访问
import threading  # 导入 threading 模块，提供了多线程相关的操作
import time  # 导入 time 模块，提供时间相关的函数

from typing import Callable, Dict, List, Optional, Set, Tuple  # 导入类型提示相关的类和函数

from torch.distributed.elastic.timer.api import TimerClient, TimerRequest  # 导入定时器客户端和请求类
from torch.distributed.elastic.timer.debug_info_logging import (
    log_debug_info_for_expired_timers,  # 导入用于记录调试信息的函数
)
from torch.distributed.elastic.utils.logging import get_logger  # 导入获取日志记录器的函数


__all__ = ["FileTimerClient", "FileTimerRequest", "FileTimerServer"]  # 定义模块的公开接口列表

logger = get_logger(__name__)  # 获取当前模块的日志记录器对象


class FileTimerRequest(TimerRequest):
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``FileTimerClient`` and ``FileTimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.
    ``signal`` is the signal to reap the worker process from the server
    process.
    """

    __slots__ = ["version", "worker_pid", "scope_id", "expiration_time", "signal"]

    def __init__(
        self, worker_pid: int, scope_id: str, expiration_time: float, signal: int = 0
    ) -> None:
        self.version = 1  # 初始化版本号为 1
        self.worker_pid = worker_pid  # 设置工作进程的 PID
        self.scope_id = scope_id  # 设置作用域 ID
        self.expiration_time = expiration_time  # 设置过期时间
        self.signal = signal  # 设置信号

    def __eq__(self, other) -> bool:
        """
        比较函数，用于判断两个 FileTimerRequest 对象是否相等。
        """
        if isinstance(other, FileTimerRequest):
            return (
                self.version == other.version
                and self.worker_pid == other.worker_pid
                and self.scope_id == other.scope_id
                and self.expiration_time == other.expiration_time
                and self.signal == other.signal
            )
        return False

    def to_json(self) -> str:
        """
        将对象转换为 JSON 格式的字符串表示。
        """
        return json.dumps(
            {
                "version": self.version,
                "pid": self.worker_pid,
                "scope_id": self.scope_id,
                "expiration_time": self.expiration_time,
                "signal": self.signal,
            },
        )


class FileTimerClient(TimerClient):
    """
    Client side of ``FileTimerServer``. This client is meant to be used
    on the same host that the ``FileTimerServer`` is running on and uses
    pid to uniquely identify a worker.
    This client uses a named_pipe to send timer requests to the
    ``FileTimerServer``. This client is a producer while the
    ``FileTimerServer`` is a consumer. Multiple clients can work with
    the same ``FileTimerServer``.

    Args:

        file_path: str, the path of a FIFO special file. ``FileTimerServer``
                        must have created it by calling os.mkfifo().

        signal: signal, the signal to use to kill the process. Using a
                        negative or zero signal will not kill the process.
    """
    # 初始化方法，接受文件路径和信号参数，默认使用SIGKILL信号（非Windows平台）或CTRL_C_EVENT信号（Windows平台）
    def __init__(
        self,
        file_path: str,
        signal=(signal.SIGKILL if sys.platform != "win32" else signal.CTRL_C_EVENT),  # type: ignore[attr-defined]
    ) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 将文件路径保存到实例变量中
        self._file_path = file_path
        # 保存信号参数到实例变量中，用于后续发送请求时使用
        self.signal = signal

    # 打开文件的非阻塞方法，返回一个可写的文本I/O包装器对象或者None
    def _open_non_blocking(self) -> Optional[io.TextIOWrapper]:
        try:
            # 打开文件描述符，在写入模式下以非阻塞方式
            fd = os.open(self._file_path, os.O_WRONLY | os.O_NONBLOCK)
            # 使用文件描述符创建文本I/O对象，并返回
            return os.fdopen(fd, "wt")
        except Exception:
            # 如果发生异常（如文件不存在或权限问题），返回None
            return None

    # 发送文件定时请求的方法，接受一个FileTimerRequest对象作为参数
    def _send_request(self, request: FileTimerRequest) -> None:
        # 服务器可能已经崩溃或尚未启动。
        # 在这种情况下，使用阻塞模式调用open()会阻塞客户端。
        # 为避免这种问题，以非阻塞模式打开文件，如果服务器不可用，将引发OSError异常。
        file = self._open_non_blocking()
        # 如果打开文件失败（file为None），抛出BrokenPipeError异常
        if file is None:
            raise BrokenPipeError(
                "Could not send the FileTimerRequest because FileTimerServer is not available."
            )
        # 使用文件上下文管理器确保操作安全关闭文件
        with file:
            # 将请求对象转换为JSON格式
            json_request = request.to_json()
            # 如果JSON请求的长度超过系统管道缓冲区大小（select.PIPE_BUF），抛出RuntimeError异常
            if len(json_request) > select.PIPE_BUF:
                raise RuntimeError(
                    f"FileTimerRequest larger than {select.PIPE_BUF} bytes "
                    f"is not supported: {json_request}"
                )
            # 将JSON请求写入文件
            file.write(json_request + "\n")

    # 获取资源的方法，发送一个FileTimerRequest对象，指定作用域ID和过期时间
    def acquire(self, scope_id: str, expiration_time: float) -> None:
        self._send_request(
            request=FileTimerRequest(
                worker_pid=os.getpid(),
                scope_id=scope_id,
                expiration_time=expiration_time,
                signal=self.signal,
            ),
        )

    # 释放资源的方法，发送一个FileTimerRequest对象，指定作用域ID和过期时间为-1
    def release(self, scope_id: str) -> None:
        self._send_request(
            request=FileTimerRequest(
                worker_pid=os.getpid(), scope_id=scope_id, expiration_time=-1, signal=0
            ),
        )
class FileTimerServer:
    """
    Server that works with ``FileTimerClient``. Clients are expected to be
    running on the same host as the process that is running this server.
    Each host in the job is expected to start its own timer server locally
    and each server instance manages timers for local workers (running on
    processes on the same host).

    Args:

        file_path: str, the path of a FIFO special file to be created.

        max_interval: float, max interval in seconds for each watchdog loop.

        daemon: bool, running the watchdog thread in daemon mode or not.
                      A daemon thread will not block a process to stop.
        log_event: Callable[[Dict[str, str]], None], an optional callback for
                logging the events in JSON format.
    """

    def __init__(
        self,
        file_path: str,
        run_id: str,
        max_interval: float = 10,
        daemon: bool = True,
        log_event: Optional[Callable[[str, Optional[FileTimerRequest]], None]] = None,
    ) -> None:
        # 设置实例变量，保存传入的参数
        self._file_path = file_path
        self._run_id = run_id
        self._max_interval = max_interval
        self._daemon = daemon
        # 初始化计时器字典，用于存储定时器请求
        self._timers: Dict[Tuple[int, str], FileTimerRequest] = {}
        self._stop_signaled = False
        self._watchdog_thread: Optional[threading.Thread] = None
        # 如果存在同名文件，先移除
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        # 创建 FIFO 特殊文件，用于进程间通信
        os.mkfifo(self._file_path)
        # 仅用于测试，计数接收到的请求数量
        self._request_count = 0
        # 仅用于测试，控制仅处理一次请求然后停止服务器
        self._run_once = False
        # 设置日志事件回调函数，默认为无操作函数
        self._log_event = (
            log_event if log_event is not None else lambda name, request: None
        )
        # 记录上次进度时间戳，初始值为当前时间戳
        self._last_progress_time = int(time.time())

    def start(self) -> None:
        # 记录服务器启动信息，包括最大间隔和是否为守护线程模式
        logger.info(
            "Starting %s..." " max_interval=%s," " daemon=%s",
            type(self).__name__,
            self._max_interval,
            self._daemon,
        )
        # 创建并启动看门狗线程
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=self._daemon
        )
        logger.info("Starting watchdog thread...")
        self._watchdog_thread.start()
        # 记录看门狗线程启动事件
        self._log_event("watchdog started", None)

    def stop(self) -> None:
        # 记录服务器停止信息
        logger.info("Stopping %s", type(self).__name__)
        # 发送停止信号，通知看门狗线程退出
        self._stop_signaled = True
        if self._watchdog_thread:
            # 等待看门狗线程结束，最长等待时间为最大间隔
            logger.info("Stopping watchdog thread...")
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            # 若没有正在运行的看门狗线程，记录无操作事件
            logger.info("No watchdog thread running, doing nothing")
        # 移除 FIFO 文件
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        # 记录看门狗停止事件
        self._log_event("watchdog stopped", None)
    def run_once(self) -> None:
        # 将 _run_once 标志设为 True，表示运行一次标志已激活
        self._run_once = True
        if self._watchdog_thread:
            # 如果存在 watchdog 线程，则记录日志信息
            logger.info("Stopping watchdog thread...")
            # 等待 watchdog 线程结束
            self._watchdog_thread.join()
            # 置空 watchdog 线程引用
            self._watchdog_thread = None
        else:
            # 如果没有 watchdog 线程在运行，则记录日志信息
            logger.info("No watchdog thread running, doing nothing")
        if os.path.exists(self._file_path):
            # 如果指定文件路径存在，则删除该文件
            os.remove(self._file_path)

    @staticmethod
    def is_process_running(pid: int):
        """
        function to check process is running or not
        """
        try:
            # 尝试向指定 PID 发送信号，检查进程是否存在
            os.kill(pid, 0)
            return True
        except OSError:
            # 如果出现 OSError 则进程不存在
            return False

    def _watchdog_loop(self) -> None:
        # 使用阻塞模式打开管道，会阻塞服务器线程。
        # 这是合理的原因有：
        #  1. 通常情况下不会没有客户端连接。
        #  2. 我们在单独的守护线程中运行 watchdog 循环，这不会阻塞进程停止。
        with open(self._file_path) as fd:
            while not self._stop_signaled:
                try:
                    # 获取 _run_once 的当前值
                    run_once = self._run_once
                    # 运行 watchdog 方法处理文件描述符
                    self._run_watchdog(fd)
                    # 如果 _run_once 为真，则退出循环
                    if run_once:
                        break
                    # 更新上次进度时间为当前时间的整数值
                    self._last_progress_time = int(time.time())
                except Exception:
                    # 捕获任何异常并记录日志
                    logger.exception("Error running watchdog")
    # 运行 watchdog 功能的私有方法，监控文件描述符 fd 上的请求
    def _run_watchdog(self, fd: io.TextIOWrapper) -> None:
        # 获取基于文件描述符的定时器请求列表，最大间隔为 self._max_interval
        timer_requests = self._get_requests(fd, self._max_interval)
        # 注册这些定时器请求
        self.register_timers(timer_requests)
        # 获取当前时间戳
        now = time.time()
        # 存储已经回收的 worker 进程的 PID
        reaped_worker_pids = set()

        # 获取所有已经过期的定时器
        all_expired_timers = self.get_expired_timers(now)
        # 记录已过期定时器的调试信息
        log_debug_info_for_expired_timers(
            self._run_id,
            {
                # 遍历每个 worker 进程的过期定时器，获取它们的作用域
                pid: self._get_scopes(expired_timers)
                for pid, expired_timers in all_expired_timers.items()
            },
        )

        # 遍历每个 worker 进程及其过期的定时器
        for worker_pid, expired_timers in all_expired_timers.items():
            # 记录信息：回收 worker 进程的 PID，以及其过期的定时器作用域
            logger.info(
                "Reaping worker_pid=[%s]. Expired timers: %s",
                worker_pid,
                self._get_scopes(expired_timers),
            )
            # 将 worker 进程的 PID 添加到已回收的集合中
            reaped_worker_pids.add(worker_pid)
            
            # 对过期的定时器按照过期时间排序
            expired_timers.sort(key=lambda timer: timer.expiration_time)
            signal = 0
            expired_timer = None
            # 遍历过期定时器，找到第一个带有有效信号（>0）的定时器
            for timer in expired_timers:
                self._log_event("timer expired", timer)
                if timer.signal > 0:
                    signal = timer.signal
                    expired_timer = timer
                    break
            # 若未找到有效信号的定时器，则记录日志并继续下一个 worker 进程
            if signal <= 0:
                logger.info(
                    "No signal specified with worker=[%s]. Do not reap it.", worker_pid
                )
                continue
            # 尝试使用指定信号回收 worker 进程
            if self._reap_worker(worker_pid, signal):
                logger.info(
                    "Successfully reaped worker=[%s] with signal=%s", worker_pid, signal
                )
                # 记录事件日志：杀死 worker 进程
                self._log_event("kill worker process", expired_timer)
            else:
                # 若回收失败，则记录错误日志，并计划在下一个 watchdog 循环重试
                logger.error(
                    "Error reaping worker=[%s]. Will retry on next watchdog.",
                    worker_pid,
                )
        
        # 清理已回收的 worker 进程的定时器
        self.clear_timers(reaped_worker_pids)

    # 获取定时器请求列表中每个请求的作用域 ID
    def _get_scopes(self, timer_requests: List[FileTimerRequest]) -> List[str]:
        return [r.scope_id for r in timer_requests]

    # 获取基于文件描述符的定时器请求列表
    def _get_requests(
        self, fd: io.TextIOWrapper, max_interval: float
    ) -> List[FileTimerRequest]:
        # 省略部分代码，具体实现依赖具体功能需求
    ) -> List[FileTimerRequest]:
        # 记录开始时间
        start = time.time()
        # 初始化请求列表
        requests = []
        # 当停止信号未被发出或者仅需运行一次时执行循环
        while not self._stop_signaled or self._run_once:
            # 对于命名管道，当至少有一个写入者打开时，readline() 是阻塞的。
            # 它仅在写入端调用 flush() 时返回。
            # 注意 flush() 在 close() 内部自动调用。
            # 当最后一个写入者关闭时，readline() 不再阻塞。
            # 当处于文件结尾时，它会返回空字符串。
            # 由于客户端总是打开管道，写入一条消息并立即关闭管道，
            # 下面的 readline() 调用不会长时间阻塞。
            json_request = fd.readline()
            if len(json_request) == 0:
                # 如果读取到空字符串，且是单次运行模式，则跳出循环
                if self._run_once:
                    break
                # 否则等待一段时间后继续
                time.sleep(min(max_interval, 1))
            else:
                # 解析 JSON 请求
                request = json.loads(json_request)
                pid = request["pid"]
                scope_id = request["scope_id"]
                expiration_time = request["expiration_time"]
                signal = request["signal"]
                # 将请求添加到请求列表中
                requests.append(
                    FileTimerRequest(
                        worker_pid=pid,
                        scope_id=scope_id,
                        expiration_time=expiration_time,
                        signal=signal,
                    )
                )
            # 更新当前时间
            now = time.time()
            # 如果超过最大时间间隔，则跳出循环
            if now - start > max_interval:
                break
        # 返回所有请求的列表
        return requests

    def register_timers(self, timer_requests: List[FileTimerRequest]) -> None:
        # 遍历所有计时器请求
        for request in timer_requests:
            pid = request.worker_pid
            scope_id = request.scope_id
            expiration_time = request.expiration_time
            # 增加请求计数
            self._request_count += 1

            key = (pid, scope_id)
            # 负的过期时间表示释放调用的代理
            if expiration_time < 0:
                # 如果键存在于计时器中，则删除它
                if key in self._timers:
                    del self._timers[key]
            else:
                # 否则将请求加入计时器中
                self._timers[key] = request

    def clear_timers(self, worker_pids: Set[int]) -> None:
        # 遍历计时器字典的拷贝列表
        for pid, scope_id in list(self._timers.keys()):
            # 如果 pid 在工作进程集中或者进程已经结束，则删除计时器
            if pid in worker_pids or not FileTimerServer.is_process_running(pid):
                del self._timers[(pid, scope_id)]

    def get_expired_timers(self, deadline: float) -> Dict[int, List[FileTimerRequest]]:
        # 过期计时器字典，键为进程 ID，值为计时器请求列表
        expired_timers: Dict[int, List[FileTimerRequest]] = {}
        # 遍历所有计时器请求
        for request in self._timers.values():
            # 如果计时器请求的过期时间小于等于截止时间，则加入到过期计时器字典中
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_pid, [])
                expired_scopes.append(request)
        # 返回过期计时器字典
        return expired_timers
    # 尝试向指定的 worker_pid 进程发送信号以终止其运行
    def _reap_worker(self, worker_pid: int, signal: int) -> bool:
        try:
            # 发送信号给指定的进程
            os.kill(worker_pid, signal)
            # 如果成功发送信号，则返回 True
            return True
        except ProcessLookupError:
            # 如果进程不存在，则记录日志并跳过
            logger.info("Process with pid=%s does not exist. Skipping", worker_pid)
            return True
        except Exception:
            # 捕获其他异常情况，并记录相关错误信息
            logger.exception("Error terminating pid=%s", worker_pid)
        # 发生异常或者无法终止进程时，返回 False
        return False

    # 返回保存在实例变量 _last_progress_time 中的最后一次进度时间
    def get_last_progress_time(self) -> int:
        return self._last_progress_time
```