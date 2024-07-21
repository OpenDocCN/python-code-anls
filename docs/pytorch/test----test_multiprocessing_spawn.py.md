# `.\pytorch\test\test_multiprocessing_spawn.py`

```
# Owner(s): ["module: multiprocessing"]

# 导入必要的模块
import os  # 提供与操作系统交互的功能
import pickle  # 提供序列化和反序列化 Python 对象的工具
import random  # 提供生成随机数的功能
import signal  # 提供处理信号的功能
import sys  # 提供访问与 Python 解释器相关的变量和函数
import time  # 提供处理时间的功能
import unittest  # 提供编写和运行单元测试的框架

# 导入 Torch 的测试相关模块和函数
from torch.testing._internal.common_utils import (TestCase, run_tests, IS_WINDOWS, NO_MULTIPROCESSING_SPAWN)
import torch.multiprocessing as mp  # 提供多进程功能的模块


def _test_success_func(i):
    pass
    # 空函数，用于测试成功的情况


def _test_success_single_arg_func(i, arg):
    if arg:
        arg.put(i)
    # 如果参数 arg 不为空，则将 i 放入队列 arg 中


def _test_exception_single_func(i, arg):
    if i == arg:
        raise ValueError("legitimate exception from process %d" % i)
    time.sleep(1.0)
    # 如果 i 等于 arg，则抛出 ValueError 异常，否则睡眠 1.0 秒


def _test_exception_all_func(i):
    time.sleep(random.random() / 10)
    raise ValueError("legitimate exception from process %d" % i)
    # 睡眠一个随机时间然后抛出 ValueError 异常，模拟异常情况


def _test_terminate_signal_func(i):
    if i == 0:
        os.kill(os.getpid(), signal.SIGABRT)
    time.sleep(1.0)
    # 如果 i 等于 0，则向当前进程发送 SIGABRT 信号终止进程，否则睡眠 1.0 秒


def _test_terminate_exit_func(i, arg):
    if i == 0:
        sys.exit(arg)
    time.sleep(1.0)
    # 如果 i 等于 0，则以 arg 作为退出码退出进程，否则睡眠 1.0 秒


def _test_success_first_then_exception_func(i, arg):
    if i == 0:
        return
    time.sleep(0.1)
    raise ValueError("legitimate exception")
    # 如果 i 等于 0，则直接返回，否则睡眠 0.1 秒后抛出 ValueError 异常


def _test_nested_child_body(i, ready_queue, nested_child_sleep):
    ready_queue.put(None)
    time.sleep(nested_child_sleep)
    # 将 None 放入 ready_queue 中，并睡眠 nested_child_sleep 秒


def _test_infinite_task(i):
    while True:
        time.sleep(1)
    # 无限循环，每次睡眠 1 秒


def _test_process_exit(idx):
    sys.exit(12)
    # 以退出码 12 退出进程


def _test_nested(i, pids_queue, nested_child_sleep, start_method):
    context = mp.get_context(start_method)
    nested_child_ready_queue = context.Queue()
    nprocs = 2
    mp_context = mp.start_processes(
        fn=_test_nested_child_body,
        args=(nested_child_ready_queue, nested_child_sleep),
        nprocs=nprocs,
        join=False,
        daemon=False,
        start_method=start_method,
    )
    pids_queue.put(mp_context.pids())

    # 等待两个子进程都已启动，确保它们调用了 prctl(2) 注册了父进程终止信号
    for _ in range(nprocs):
        nested_child_ready_queue.get()

    # 杀死当前进程，应同时关闭子进程
    os.kill(os.getpid(), signal.SIGTERM)
    # 使用 SIGTERM 信号杀死当前进程


class _TestMultiProcessing:
    start_method = None

    def test_success(self):
        mp.start_processes(_test_success_func, nprocs=2, start_method=self.start_method)
        # 启动两个进程运行 _test_success_func 函数，测试成功情况

    def test_success_non_blocking(self):
        mp_context = mp.start_processes(_test_success_func, nprocs=2, join=False, start_method=self.start_method)

        # 所有进程（nproc=2）都加入后必须返回 True
        mp_context.join(timeout=None)
        mp_context.join(timeout=None)
        self.assertTrue(mp_context.join(timeout=None))
        # 断言所有进程已成功加入

    def test_first_argument_index(self):
        context = mp.get_context(self.start_method)
        queue = context.SimpleQueue()
        mp.start_processes(_test_success_single_arg_func, args=(queue,), nprocs=2, start_method=self.start_method)
        self.assertEqual([0, 1], sorted([queue.get(), queue.get()]))
        # 启动两个进程运行 _test_success_single_arg_func 函数，测试第一个参数的索引
    # 定义测试函数，验证在单个进程中抛出异常的情况
    def test_exception_single(self):
        # 设置进程数量
        nprocs = 2
        # 循环创建多个进程进行测试
        for i in range(nprocs):
            # 使用断言验证异常是否被正确捕获和报告
            with self.assertRaisesRegex(
                Exception,
                "\nValueError: legitimate exception from process %d$" % i,
            ):
                # 启动多进程，调用指定的测试函数并传递参数，验证异常抛出
                mp.start_processes(_test_exception_single_func, args=(i,), nprocs=nprocs, start_method=self.start_method)

    # 定义测试函数，验证在所有进程中抛出异常的情况
    def test_exception_all(self):
        # 使用断言验证异常是否被正确捕获和报告
        with self.assertRaisesRegex(
            Exception,
            "\nValueError: legitimate exception from process (0|1)$",
        ):
            # 启动多进程，调用指定的测试函数并传递参数，验证异常抛出
            mp.start_processes(_test_exception_all_func, nprocs=2, start_method=self.start_method)

    # 定义测试函数，验证进程收到终止信号的情况
    def test_terminate_signal(self):
        # 定义预期的终止信号消息
        # 在 multiprocessing 中，通过负的退出码表示进程被信号终止
        message = "process 0 terminated with signal (SIGABRT|SIGIOT)"

        # 在 Windows 平台上，退出码总是正数，因此会导致不同的异常消息
        # 退出码 22 表示 "ERROR_BAD_COMMAND"
        if IS_WINDOWS:
            message = "process 0 terminated with exit code 22"

        # 使用断言验证异常是否被正确捕获和报告
        with self.assertRaisesRegex(Exception, message):
            # 启动多进程，调用指定的测试函数，验证进程是否受到了预期的信号终止
            mp.start_processes(_test_terminate_signal_func, nprocs=2, start_method=self.start_method)

    # 定义测试函数，验证进程正常退出的情况
    def test_terminate_exit(self):
        # 定义预期的退出码
        exitcode = 123
        # 使用断言验证异常是否被正确捕获和报告
        with self.assertRaisesRegex(
            Exception,
            "process 0 terminated with exit code %d" % exitcode,
        ):
            # 启动多进程，调用指定的测试函数并传递参数，验证进程是否以预期的退出码结束
            mp.start_processes(_test_terminate_exit_func, args=(exitcode,), nprocs=2, start_method=self.start_method)

    # 定义测试函数，验证首先成功然后抛出异常的情况
    def test_success_first_then_exception(self):
        # 定义预期的异常消息
        exitcode = 123
        # 使用断言验证异常是否被正确捕获和报告
        with self.assertRaisesRegex(
            Exception,
            "ValueError: legitimate exception",
        ):
            # 启动多进程，调用指定的测试函数并传递参数，验证在成功后抛出预期的异常
            mp.start_processes(_test_success_first_then_exception_func, args=(exitcode,), nprocs=2, start_method=self.start_method)

    # 使用 unittest 的条件跳过装饰器，仅在 Linux 平台下运行该测试
    @unittest.skipIf(
        sys.platform != "linux",
        "Only runs on Linux; requires prctl(2)",
    )
    # 定义一个测试方法，用于测试嵌套进程的行为
    def _test_nested(self):
        # 获取指定启动方法的进程上下文
        context = mp.get_context(self.start_method)
        # 创建一个进程间通信的队列
        pids_queue = context.Queue()
        # 嵌套子进程的睡眠时间
        nested_child_sleep = 20.0
        # 启动一个子进程来执行测试函数 `_test_nested`
        mp_context = mp.start_processes(
            fn=_test_nested,
            args=(pids_queue, nested_child_sleep, self.start_method),
            nprocs=1,
            join=False,
            daemon=False,
            start_method=self.start_method,
        )

        # 等待嵌套的子进程在规定时间内终止
        pids = pids_queue.get()
        start = time.time()
        while len(pids) > 0:
            for pid in pids:
                try:
                    # 尝试发送信号到指定 PID 的进程，检测其是否存活
                    os.kill(pid, 0)
                except ProcessLookupError:
                    # 如果进程不存在，则从 PID 列表中移除
                    pids.remove(pid)
                    break

            # 如果任何嵌套子进程在 (nested_child_sleep / 2) 秒后仍然存活，则断言失败
            # 测试在 (nested_child_sleep / 2) 秒后会超时并抛出断言错误
            self.assertLess(time.time() - start, nested_child_sleep / 2)
            # 等待一小段时间后重新检查
            time.sleep(0.1)
@unittest.skipIf(
    NO_MULTIPROCESSING_SPAWN,
    "Disabled for environments that don't support the spawn start method")
# 如果 NO_MULTIPROCESSING_SPAWN 为真，则跳过测试，给出相应的注释
class SpawnTest(TestCase, _TestMultiProcessing):
    start_method = 'spawn'

    def test_exception_raises(self):
        # 测试确保 mp.spawn 函数能够捕获 mp.ProcessRaisedException 异常
        with self.assertRaises(mp.ProcessRaisedException):
            mp.spawn(_test_success_first_then_exception_func, args=(), nprocs=1)

    def test_signal_raises(self):
        # 创建一个多进程上下文 context，并启动一个进程执行 _test_infinite_task 函数
        context = mp.spawn(_test_infinite_task, args=(), nprocs=1, join=False)
        # 对 context 中的每个进程 pid 发送 SIGTERM 信号
        for pid in context.pids():
            os.kill(pid, signal.SIGTERM)
        # 确保在发送信号后，能够捕获 mp.ProcessExitedException 异常
        with self.assertRaises(mp.ProcessExitedException):
            context.join()

    def _test_process_exited(self):
        # 测试确保 mp.spawn 函数能够捕获 mp.ProcessExitedException 异常，并验证退出码为 12
        with self.assertRaises(mp.ProcessExitedException) as e:
            mp.spawn(_test_process_exit, args=(), nprocs=1)
            self.assertEqual(12, e.exit_code)


@unittest.skipIf(
    IS_WINDOWS,
    "Fork is only available on Unix",
)
# 如果 IS_WINDOWS 为真，则跳过测试，因为 fork 方法仅在 Unix 上可用
class ForkTest(TestCase, _TestMultiProcessing):
    start_method = 'fork'


class ErrorTest(TestCase):
    def test_errors_pickleable(self):
        # 测试确保 mp.ProcessRaisedException 和 mp.ProcessExitedException 类型的错误对象可序列化和反序列化
        for error in (
            mp.ProcessRaisedException("Oh no!", 1, 1),
            mp.ProcessExitedException("Oh no!", 1, 1, 1),
        ):
            pickle.loads(pickle.dumps(error))


if __name__ == '__main__':
    run_tests()
```