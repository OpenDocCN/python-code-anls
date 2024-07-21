# `.\pytorch\test\distributed\elastic\timer\file_based_local_timer_test.py`

```py
# Owner(s): ["oncall: r2p"]

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import multiprocessing as mp  # 导入 multiprocessing 库，用于实现多进程功能
import signal  # 导入 signal 库，用于处理信号
import time  # 导入 time 库，提供时间相关功能
import unittest  # 导入 unittest 框架，用于编写和运行单元测试
import unittest.mock as mock  # 导入 unittest.mock 模块，用于模拟测试

import uuid  # 导入 uuid 模块，用于生成唯一标识符

import torch.distributed.elastic.timer as timer  # 导入 torch 分布式 elastic 模块中的 timer
from torch.testing._internal.common_utils import (  # 导入 torch 测试内部通用工具模块中的函数和变量
    IS_MACOS,  # 导入操作系统检测变量 IS_MACOS
    IS_WINDOWS,  # 导入操作系统检测变量 IS_WINDOWS
    run_tests,  # 导入运行测试的函数 run_tests
    TEST_WITH_TSAN,  # 导入是否使用 TSAN 测试的变量 TEST_WITH_TSAN
    TestCase,  # 导入单元测试基类 TestCase
)

# 如果不是在 Windows 或 MacOS 上运行
if not (IS_WINDOWS or IS_MACOS):
    # func2 函数用于执行超时操作
    def func2(n, file_path):
        if file_path is not None:
            timer.configure(timer.FileTimerClient(file_path))  # 配置文件定时器客户端
        if n > 0:
            with timer.expires(after=0.1):  # 设置超时时间为 0.1 秒
                func2(n - 1, None)  # 递归调用 func2 函数，逐层减少 n
                time.sleep(0.2)  # 等待 0.2 秒

    # _request_on_interval 函数用于按间隔时间发送定时请求
    def _request_on_interval(file_path, n, interval, sem):
        """
        enqueues ``n`` timer requests into ``mp_queue`` one element per
        interval seconds. Releases the given semaphore once before going to work.
        """
        client = timer.FileTimerClient(file_path)  # 创建文件定时器客户端对象
        sem.release()  # 释放信号量
        for i in range(0, n):
            client.acquire("test_scope", 0)  # 请求定时器的资源
            time.sleep(interval)  # 按指定间隔时间等待

    # FileTimerClientTest 类，用于测试文件定时器客户端
    class FileTimerClientTest(TestCase):
        def test_send_request_without_server(self):
            client = timer.FileTimerClient("test_file")  # 创建文件定时器客户端对象
            timer.configure(client)  # 配置定时器
            with self.assertRaises(BrokenPipeError):  # 检测是否抛出 BrokenPipeError 异常
                with timer.expires(after=0.1):  # 设置超时时间为 0.1 秒
                    time.sleep(0.1)  # 等待 0.1 秒

# 如果是主程序入口
if __name__ == "__main__":
    run_tests()  # 运行单元测试
```