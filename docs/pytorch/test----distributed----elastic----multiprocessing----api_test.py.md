# `.\pytorch\test\distributed\elastic\multiprocessing\api_test.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库和模块
import ctypes  # 提供与 C 语言兼容的数据类型
import multiprocessing  # 多进程处理模块
import os  # 提供与操作系统交互的功能
import shutil  # 提供高级文件操作功能
import signal  # 提供处理信号的功能
import sys  # 提供与 Python 解释器交互的功能
import tempfile  # 提供创建临时文件和目录的功能
import time  # 提供时间相关的功能
from itertools import product  # 提供迭代器工具函数
from typing import Callable, Dict, List, Union  # 提供类型提示支持
from unittest import mock  # 提供单元测试时的模拟对象支持

import torch  # PyTorch 深度学习库
import torch.multiprocessing as mp  # 多进程处理的 PyTorch 扩展模块
from torch.distributed.elastic.multiprocessing import ProcessFailure, start_processes  # 弹性分布式训练相关模块和函数
from torch.distributed.elastic.multiprocessing.api import (
    _validate_full_rank,
    _wrap,
    DefaultLogsSpecs,
    MultiprocessContext,
    RunProcsResult,
    SignalException,
    Std,
    to_map,
)  # 弹性分布式训练 API 相关功能
from torch.distributed.elastic.multiprocessing.errors import ErrorHandler  # 处理分布式训练中的错误
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_MACOS,
    IS_WINDOWS,
    NO_MULTIPROCESSING_SPAWN,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skip_if_pytest,
    TEST_WITH_ASAN,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_TSAN,
    TestCase,
)  # PyTorch 内部测试相关的实用函数和标志


class RunProcResultsTest(TestCase):
    def setUp(self):
        super().setUp()
        # 创建临时测试目录
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

    def tearDown(self):
        super().tearDown()
        # 删除临时测试目录及其内容
        shutil.rmtree(self.test_dir)

    def test_is_failed(self):
        # 测试 RunProcsResult 类的 is_failed 方法
        pr_success = RunProcsResult(return_values={0: "a", 1: "b"})
        self.assertFalse(pr_success.is_failed())

        # 创建一个模拟的进程失败对象，并测试 is_failed 方法
        fail0 = ProcessFailure(
            local_rank=0, pid=998, exitcode=1, error_file="ignored.json"
        )
        pr_fail = RunProcsResult(failures={0: fail0})
        self.assertTrue(pr_fail.is_failed())

    def test_get_failures(self):
        # 准备用于记录异常的文件路径
        error_file0 = os.path.join(self.test_dir, "error0.json")
        error_file1 = os.path.join(self.test_dir, "error1.json")
        eh = ErrorHandler()

        # 模拟记录第一个异常
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": error_file0}):
            eh.record_exception(RuntimeError("error 0"))

        # 模拟记录第二个异常
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": error_file1}):
            eh.record_exception(RuntimeError("error 1"))

        # 创建模拟的进程失败对象，并测试时间戳的顺序
        fail0 = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=error_file0
        )
        fail1 = ProcessFailure(
            local_rank=1, pid=998, exitcode=3, error_file=error_file1
        )
        fail2 = ProcessFailure(
            local_rank=2, pid=999, exitcode=15, error_file="no_exist.json"
        )

        # 断言失败对象的时间戳顺序
        self.assertLessEqual(fail0.timestamp, fail1.timestamp)
        self.assertLessEqual(fail1.timestamp, fail2.timestamp)


class StdTest(TestCase):
    def test_from_value(self):
        # 测试 Std 类的 from_str 方法
        self.assertEqual(Std.NONE, Std.from_str("0"))
        self.assertEqual(Std.OUT, Std.from_str("1"))
        self.assertEqual(Std.ERR, Std.from_str("2"))
        self.assertEqual(Std.ALL, Std.from_str("3"))
    # 定义单元测试方法，测试从值映射字符串到标准输出的转换
    def test_from_value_map(self):
        # 断言从字符串 "0:1" 转换后的结果与预期的标准输出字典 {0: Std.OUT} 相等
        self.assertEqual({0: Std.OUT}, Std.from_str("0:1"))
        # 断言从字符串 "0:1,1:1" 转换后的结果与预期的标准输出字典 {0: Std.OUT, 1: Std.OUT} 相等
        self.assertEqual({0: Std.OUT, 1: Std.OUT}, Std.from_str("0:1,1:1"))

    # 定义单元测试方法，测试处理不良输入时是否抛出 ValueError 异常
    def test_from_str_bad_input(self):
        # 不良输入的字符串列表
        bad_inputs = ["0:1,", "11", "0:1,1", "1,0:1"]
        # 对每个不良输入进行循环测试
        for bad in bad_inputs:
            # 使用子测试（subTest）运行当前不良输入的测试，并命名为 'bad'
            with self.subTest(bad=bad):
                # 断言处理当前不良输入时是否抛出 ValueError 异常
                with self.assertRaises(ValueError):
                    Std.from_str(bad)
def echo0(msg: str) -> None:
    """
    void function
    """
    # 打印消息到标准输出
    print(msg)


def echo1(msg: str, exitcode: int = 0) -> str:
    """
    returns ``msg`` or exits with the given exitcode (if nonzero)
    """
    # 从环境变量中获取 RANK 的整数值
    rank = int(os.environ["RANK"])
    # 如果 exitcode 非零，输出错误信息到标准错误并退出程序
    if exitcode != 0:
        print(f"exit {exitcode} from {rank}", file=sys.stderr)
        sys.exit(exitcode)
    else:
        # 否则，输出消息到标准输出和标准错误，并返回带有排名后缀的消息字符串
        print(f"{msg} stdout from {rank}")
        print(f"{msg} stderr from {rank}", file=sys.stderr)
        return f"{msg}_{rank}"


def echo2(msg: str, fail: bool = False) -> str:
    """
    returns ``msg`` or raises a RuntimeError if ``fail`` is set
    """
    # 如果 fail 为真，抛出运行时错误，否则返回原始消息
    if fail:
        raise RuntimeError(msg)
    return msg


def echo_large(size: int) -> Dict[int, str]:
    """
    returns a large output ({0: test0", 1: "test1", ..., (size-1):f"test{size-1}"})
    """
    # 创建一个包含指定大小的索引和测试消息的字典，并返回
    out = {}
    for idx in range(0, size):
        out[idx] = f"test{idx}"
    return out


def echo3(msg: str, fail: bool = False) -> str:
    """
    returns ``msg`` or induces a SIGSEGV if ``fail`` is set
    """
    # 如果 fail 为真，调用 ctypes 函数触发 SIGSEGV 信号，否则返回原始消息
    if fail:
        ctypes.string_at(0)
    return msg


def dummy_compute() -> torch.Tensor:
    """
    returns a predefined size random Tensor
    """
    # 返回一个预定义大小的随机 Tensor
    return torch.rand(100, 100)


def redirects_oss_test() -> List[Std]:
    """
    returns a list with a single element, Std.NONE
    """
    return [
        Std.NONE,
    ]


def redirects_all() -> List[Std]:
    """
    returns a list with multiple elements of Std enumeration
    """
    return [
        Std.NONE,
        Std.OUT,
        Std.ERR,
        Std.ALL,
    ]


def bin(name: str):
    """
    returns a path by joining directory of this file with 'bin' and given name
    """
    # 获取当前文件所在目录，并返回一个包含 'bin' 目录和指定名称的路径
    dir = os.path.dirname(__file__)
    return os.path.join(dir, "bin", name)


def wait_fn(wait_time: int = 300) -> None:
    """
    sleeps for specified wait_time seconds and prints a message upon completion
    """
    # 等待指定时间后输出完成等待的消息
    time.sleep(wait_time)
    print("Finished waiting")


def start_processes_zombie_test(
    idx: int,
    entrypoint: Union[str, Callable],
    mp_queue: mp.Queue,
    log_dir: str,
    nproc: int = 2,
) -> None:
    """
    Starts processes
    """
    # 准备进程的参数和环境变量字典
    args = {}
    envs = {}
    for idx in range(nproc):
        args[idx] = ()
        envs[idx] = {}

    # 调用 start_processes 函数开始进程，并将主进程及其子进程的 PID 放入 multiprocessing 队列
    pc = start_processes(
        name="zombie_test",
        entrypoint=entrypoint,
        args=args,
        envs=envs,
        logs_specs=DefaultLogsSpecs(log_dir=log_dir),
    )
    my_pid = os.getpid()
    mp_queue.put(my_pid)
    for child_pid in pc.pids().values():
        mp_queue.put(child_pid)

    try:
        # 等待进程完成或超时，如果接收到信号异常，使用信号值关闭进程组
        pc.wait(period=1, timeout=300)
    except SignalException as e:
        pc.close(e.sigval)


# tests incompatible with tsan or asan
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS):
    # 在不支持 tsan 或 asan 的环境中运行测试

# tests incompatible with tsan or asan, the redirect functionality does not work on macos or windows
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS):
    # 在不支持 tsan 或 asan，且在 macOS 或 Windows 系统上不支持重定向功能的环境中运行测试

# tests incompatible with tsan or asan, the redirect functionality does not work on macos or windows
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS or IS_CI):
    # 在不支持 tsan 或 asan，且在 macOS、Windows 或 CI 环境上不支持重定向功能的环境中运行测试

if __name__ == "__main__":
    run_tests()  # 如果脚本被直接执行，运行测试函数
```