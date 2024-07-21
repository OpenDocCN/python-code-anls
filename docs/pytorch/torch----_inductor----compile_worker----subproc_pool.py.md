# `.\pytorch\torch\_inductor\compile_worker\subproc_pool.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import functools  # 提供高阶函数和函数操作的功能
import itertools  # 提供用于操作迭代器的函数
import logging  # 提供日志记录功能
import multiprocessing  # 提供多进程支持
import os  # 提供与操作系统交互的功能
import pickle  # 提供对象序列化和反序列化功能
import struct  # 提供处理字节数据的功能
import subprocess  # 提供创建和管理子进程的功能
import sys  # 提供与 Python 解释器交互的功能
import threading  # 提供多线程支持
import traceback  # 提供跟踪异常调用栈的功能
import typing  # 提供类型提示支持
from concurrent.futures import Future, ProcessPoolExecutor  # 异步执行任务的支持
from concurrent.futures.process import BrokenProcessPool  # 处理进程池异常的支持
from typing import Any, Callable, Dict  # 类型提示支持

from torch._inductor import config  # 导入与 Torch 相关的配置模块
from torch._inductor.compile_worker.watchdog import _async_compile_initializer  # 导入异步编译的初始化函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器实例


class Pipe(typing.Protocol):
    """
    定义 Pipe 协议，规定了 Pipe 类需要实现的方法。
    """

    def write(self, data: bytes):
        """
        写入字节数据到 Pipe 中。

        Args:
            data: 要写入的字节数据。
        """
        ...

    def read(self, n: int) -> bytes:
        """
        从 Pipe 中读取指定长度的字节数据。

        Args:
            n: 要读取的字节数。

        Returns:
            读取的字节数据。
        """
        ...

    def close(self):
        """
        关闭 Pipe 连接。
        """
        ...

    def flush(self):
        """
        刷新 Pipe 的缓冲区。
        """
        ...


def _pack_msg(job_id, length):
    """
    将任务 ID 和数据长度打包成字节数据。

    Args:
        job_id: 任务 ID。
        length: 数据长度。

    Returns:
        打包后的字节数据。
    """
    return struct.pack("nn", job_id, length)


def _unpack_msg(data):
    """
    解析字节数据，获取任务 ID 和数据长度。

    Args:
        data: 要解析的字节数据。

    Returns:
        解析后的任务 ID 和数据。
    """
    if not data:
        return -1, -1
    return struct.unpack("nn", data)


msg_bytes = len(_pack_msg(0, 0))  # 计算打包消息后的字节数


def _send_msg(write_pipe, job_id, job_data=b""):
    """
    发送消息到管道中。

    Args:
        write_pipe: 写入数据的管道。
        job_id: 任务 ID。
        job_data: 要发送的数据，默认为空字节串。
    """
    length = len(job_data)
    write_pipe.write(_pack_msg(job_id, length))  # 写入任务 ID 和数据长度
    if length > 0:
        write_pipe.write(job_data)  # 如果有数据，写入数据到管道
    write_pipe.flush()  # 刷新管道，确保数据被发送


def _recv_msg(read_pipe):
    """
    从管道中接收消息。

    Args:
        read_pipe: 读取数据的管道。

    Returns:
        任务 ID 和接收到的数据。
    """
    job_id, length = _unpack_msg(read_pipe.read(msg_bytes))  # 从管道中读取任务 ID 和数据长度
    data = read_pipe.read(length) if length > 0 else b""  # 根据数据长度读取实际数据
    return job_id, data


def _get_ld_library_path():
    """
    获取 LD_LIBRARY_PATH 环境变量的路径。

    Returns:
        LD_LIBRARY_PATH 的路径。
    """
    path = os.environ.get("LD_LIBRARY_PATH", "")  # 获取环境变量 LD_LIBRARY_PATH 的值
    if config.is_fbcode():
        from libfb.py.parutil import get_runtime_path

        runtime_path = get_runtime_path()
        if runtime_path:
            lib_path = os.path.join(runtime_path, "runtime", "lib")
            path = os.pathsep.join([lib_path, path]) if path else lib_path  # 拼接运行时库的路径到 LD_LIBRARY_PATH
    return path


class _SubprocExceptionInfo:
    """
    封装子进程中的异常信息，用于在主进程中传递异常。
    traceback 对象不可序列化，因此将异常跟踪信息存储为字符串。
    """

    def __init__(self, details):
        """
        初始化异常信息。

        Args:
            details: 异常的详细信息。
        """
        self.details = details


class SubprocException(Exception):
    """
    当子进程中的任务引发异常时抛出的异常。
    """

    def __init__(self, details):
        """
        初始化异常信息。

        Args:
            details: 异常的详细信息。
        """
        super().__init__(f"An exception occurred in a subprocess:\n\n{details}")


class SubprocPool:
    """
    模拟 concurrent.futures.ProcessPoolExecutor，但在 subprocess.Popen() 中封装，
    以尝试避免 fork/spawn 问题。
    """
    def __init__(self, nprocs: int):
        # 获取主文件 __main__.py 的完整路径
        entry = os.path.join(os.path.dirname(__file__), "__main__.py")
        # 组装执行命令，指定使用当前解释器执行主文件，设置参数
        cmd = [
            sys.executable,
            entry,
            f"--workers={nprocs}",  # 指定工作进程数量
            f"--parent={os.getpid()}",  # 指定父进程的 PID
        ]
        # 启动子进程，并传入命令、标准输入输出管道以及环境变量设置
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env={
                **os.environ,  # 继承当前环境变量
                "PYTHONPATH": os.pathsep.join(sys.path),  # 设置 PYTHONPATH 以便子进程找到 torch
                "TORCH_WARM_POOL": "0",  # 禁用 torch 进程池预热
                "LD_LIBRARY_PATH": _get_ld_library_path(),  # 设置特定的 LD_LIBRARY_PATH
            },
        )
        # 设置写入管道和写入锁
        self.write_pipe: Pipe = typing.cast(Pipe, self.process.stdin)
        self.write_lock = threading.Lock()
        # 设置读取管道和读取线程
        self.read_pipe: Pipe = typing.cast(Pipe, self.process.stdout)
        self.read_thread = threading.Thread(target=self._read_thread, daemon=True)

        # 设置用于管理未完成任务的锁和字典
        self.futures_lock = threading.Lock()
        self.pending_futures: Dict[int, Future[Any]] = {}
        # 计数器，用于生成唯一的作业 ID
        self.job_id_count = itertools.count()

        # 标记线程已启动
        self.running = True

        # 启动读取线程，确保所有成员变量初始化后再启动
        self.read_thread.start()

    def submit(self, job_fn: Callable[..., Any], *args):
        # 如果有额外参数，则使用 functools.partial 创建带参数的函数
        if args:
            job_fn = functools.partial(job_fn, *args)
        # 序列化作业函数及其参数，使用 pickle 进行高级协议的序列化
        job_data = pickle.dumps(job_fn, pickle.HIGHEST_PROTOCOL)
        future: Future[Any]
        # 使用 futures_lock 确保并发安全地分配作业 ID 和设置未完成的任务
        with self.futures_lock:
            job_id = next(self.job_id_count)
            self.pending_futures[job_id] = future = Future()
        # 标记 future 为运行中或通知取消
        future.set_running_or_notify_cancel()
        # 使用写入锁确保并发安全地向子进程发送作业消息
        with self.write_lock:
            if not self.running:
                raise RuntimeError("submit() on closed pool")
            _send_msg(self.write_pipe, job_id, job_data)
        # 返回 future 对象，以便调用者可以跟踪任务状态和结果
        return future
    # 定义一个内部方法 `_read_thread`，用于处理子进程池中读取数据的线程
    def _read_thread(self):
        try:
            # 循环执行，持续读取管道中的消息
            while True:
                # 调用 `_recv_msg` 方法从管道中接收消息，返回作业 ID 和数据
                job_id, data = _recv_msg(self.read_pipe)
                # 如果接收到的作业 ID 小于 0，表示接收到退出信号
                if job_id < 0:
                    # 如果子进程池仍在运行状态，记录警告信息
                    if self.running:
                        log.warning("SubprocPool unclean exit")
                    # 关闭读管道并退出方法
                    self.read_pipe.close()
                    return
                # 使用 pickle 反序列化接收到的数据
                result = pickle.loads(data)
                # 使用 futures 锁，确保线程安全
                with self.futures_lock:
                    # 如果子进程池已经停止运行，退出方法
                    if not self.running:
                        return
                    # 如果接收到的结果是 `_SubprocExceptionInfo` 类型的异常信息
                    if isinstance(result, _SubprocExceptionInfo):
                        # 设置对应作业的异常状态，抛出 `SubprocException`
                        self.pending_futures[job_id].set_exception(
                            SubprocException(result.details)
                        )
                    # 如果接收到的结果是一般异常对象
                    elif isinstance(result, Exception):
                        # 设置对应作业的异常状态
                        self.pending_futures[job_id].set_exception(result)
                    # 如果接收到的结果正常
                    else:
                        # 设置对应作业的结果状态
                        self.pending_futures[job_id].set_result(result)
                    # 从待处理作业字典中移除已处理的作业 ID
                    del self.pending_futures[job_id]
        # 捕获所有异常
        except Exception:
            # 记录异常信息到日志
            log.exception("failure in SubprocPool._read_thread")

    # 定义关闭子进程池的方法
    def shutdown(self):
        try:
            # 使用写锁，确保关闭操作的线程安全性
            with self.write_lock:
                # 如果子进程池已经停止运行，直接返回
                if not self.running:
                    return
                # 将运行状态置为 False
                self.running = False
                # 向写管道发送结束信号 `-1`
                _send_msg(self.write_pipe, -1)
                # 关闭写管道
                self.write_pipe.close()
            # 等待子进程的退出，超时时间为 10 秒
            self.process.wait(10)
        # 捕获 OSError 异常
        except OSError as e:
            # 记录警告信息，忽略 OSError 异常
            log.warning("Ignored OSError in pool shutdown:  %s", e)
        # 最终处理，确保待处理作业字典中的所有未完成作业都被取消或设置异常状态
        finally:
            # 使用 futures 锁，确保线程安全
            with self.futures_lock:
                # 遍历所有待处理作业的 future 对象
                for future in self.pending_futures.values():
                    # 如果未能成功取消作业
                    if not future.cancel():
                        # 设置异常状态，表明子进程池已关闭
                        future.set_exception(RuntimeError("SubprocPool closed"))
                # 清空待处理作业字典
                self.pending_futures.clear()
class SubprocMain:
    """Communicates with a SubprocPool in the parent process, called by __main__.py"""

    def __init__(self, nprocs: int, read_pipe: Pipe, write_pipe: Pipe):
        self.read_pipe = read_pipe  # 保存读取管道对象的引用
        self.write_pipe = write_pipe  # 保存写入管道对象的引用
        self.write_lock = threading.Lock()  # 创建一个线程锁对象，用于保护写入操作
        self.nprocs = nprocs  # 记录进程池中进程的数量
        self.pool = self._new_pool(nprocs, True)  # 创建一个新的进程池对象
        self.running = True  # 标志位，表示进程是否在运行中

    def _new_pool(self, nprocs, warm):
        pool = ProcessPoolExecutor(
            nprocs,
            mp_context=multiprocessing.get_context("fork"),  # 使用 "fork" 上下文来创建多进程
            initializer=functools.partial(_async_compile_initializer, os.getpid()),  # 初始化函数，设置进程环境
        )
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)  # 注册一个进程退出时的清理函数
        if warm:
            _warm_process_pool(pool, nprocs)  # 预热进程池，提前创建所需数量的进程
        return pool

    def main(self):
        while True:
            job_id, data = _recv_msg(self.read_pipe)  # 从读取管道中接收消息和数据
            if job_id < 0:
                return self._shutdown()  # 如果接收到的任务 ID 小于 0，执行关闭操作
            self.submit(job_id, data)  # 提交任务给进程池处理

    def _shutdown(self):
        with self.write_lock:
            self.running = False  # 设置运行状态为 False，表示进程池要关闭
            try:
                _send_msg(self.write_pipe, -1)  # 向写入管道发送结束信号
                self.write_pipe.close()  # 关闭写入管道
            except BrokenPipeError:
                pass  # 父进程已经关闭，不处理管道异常
            self.read_pipe.close()  # 关闭读取管道
        self.pool.shutdown()  # 关闭进程池，等待所有任务完成后退出

    def submit(self, job_id, data):
        while self.running:
            try:
                self._submit_inner(job_id, data)  # 提交任务给进程池的内部处理函数
                return
            except BrokenProcessPool:
                # 如果进程池中的任一子进程崩溃，会抛出 BrokenProcessPool 异常，整个进程池变得不可用
                # 处理崩溃情况，重新创建进程池并重新提交任务
                self.pool = self._new_pool(self.nprocs, False)

    def _submit_inner(self, job_id, data):
        future = self.pool.submit(functools.partial(SubprocMain.do_job, data))  # 在进程池中提交任务

        def callback(_):
            if not self.running:
                return
            try:
                result = future.result()  # 获取任务执行的结果
            except Exception as e:
                log.exception("Error in subprocess")  # 记录子进程中的异常信息
                result = pickle.dumps(e, pickle.HIGHEST_PROTOCOL)  # 将异常信息序列化为字节流
            assert isinstance(result, bytes)
            with self.write_lock:
                if self.running:
                    _send_msg(self.write_pipe, job_id, result)  # 向写入管道发送任务执行结果

        future.add_done_callback(callback)  # 注册任务完成时的回调函数

    @staticmethod
    def do_job(data):
        # 在子子进程中执行 pickle/unpickle 操作
        job = pickle.loads(data)  # 反序列化任务数据
        try:
            result = job()  # 执行任务
        except Exception as e:
            result = _SubprocExceptionInfo(traceback.format_exc())  # 处理任务执行异常
        return pickle.dumps(result, pickle.HIGHEST_PROTOCOL)  # 将任务执行结果序列化为字节流


AnyPool = typing.Union[ProcessPoolExecutor, SubprocPool]


def _warm_process_pool(pool: AnyPool, n: int):
    if isinstance(pool, SubprocPool):
        return  # 对于 SubprocPool 类型的进程池，不需要预热
    # 断言确保参数 pool 是 ProcessPoolExecutor 类型的实例

    # 我们需要为编译器工作进程进行分叉，但是加载的内存和其他资源越多，os.fork 的时间就会越慢，严重影响性能。
    # 它还会持有全局解释器锁（GIL），因此我们无法将其放在另一个线程中执行。

    # 举例：
    # 简单的 x + x + x 脚本：在程序中段时大约需要 10 毫秒，启动时约需 2 毫秒
    # tf_efficientnet_b0 基准测试：在程序中段时需要高达 50 毫秒！启动时约需 3 毫秒

    # 因此，我们希望在成本较低时尽早启动工作进程，并且让工作进程能够在有任务到来前准备好。

    # ProcessPoolExecutor 在找到所有工作进程都处于空闲状态之前不会启动这些进程。
    # 但是如果等到那时再进行分叉，分叉时间会很长，我们将等待进程初始化完成。

    # 我们在这里强制启动它们，使用了一些内部方法来强制启动。

    # TODO(masnesral): 这些信息是否仍然相关？

    # 如果 pool 具有 "_start_queue_management_thread" 方法，则调用该方法启动队列管理线程
    if hasattr(pool, "_start_queue_management_thread"):
        pool._start_queue_management_thread()
    else:
        # 否则，通过多次调用 "_adjust_process_count" 方法来调整进程数量，确保进程池中的进程已经启动
        for _ in range(n):
            pool._adjust_process_count()
        # 如果 pool 具有 "_start_executor_manager_thread" 方法，则调用该方法启动执行器管理线程
        if hasattr(pool, "_start_executor_manager_thread"):
            pool._start_executor_manager_thread()
# 定义一个名为 TestException 的自定义异常类，继承自 RuntimeError
class TestException(RuntimeError):
    # pass 语句表示该类没有额外的行为或属性，只是一个占位符
    pass


# 定义一个函数 raise_testexc，用于抛出 TestException 异常
def raise_testexc():
    # 抛出 TestException 异常
    raise TestException
```