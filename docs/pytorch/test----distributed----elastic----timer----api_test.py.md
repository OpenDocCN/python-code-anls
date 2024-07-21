# `.\pytorch\test\distributed\elastic\timer\api_test.py`

```py
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 引入单元测试框架 unittest
import unittest
# 引入 unittest.mock 作为 mock 模块
import unittest.mock as mock

# 从 torch.distributed.elastic.timer 中引入 TimerServer 类
from torch.distributed.elastic.timer import TimerServer
# 从 torch.distributed.elastic.timer.api 中引入 RequestQueue 和 TimerRequest 类
from torch.distributed.elastic.timer.api import RequestQueue, TimerRequest

# MockRequestQueue 类，继承自 RequestQueue 类，用于模拟请求队列
class MockRequestQueue(RequestQueue):
    def size(self):
        return 2

    def get(self, size, timeout):
        # 返回一个包含两个 TimerRequest 对象的列表，模拟从队列中获取请求
        return [TimerRequest(1, "test_1", 0), TimerRequest(2, "test_2", 0)]

# MockTimerServer 类，继承自 TimerServer 类，用于模拟计时器服务器
class MockTimerServer(TimerServer):
    """
     Mock implementation of TimerServer for testing purposes.
     This mock has the following behavior:

     1. reaping worker 1 throws
     2. reaping worker 2 succeeds
     3. reaping worker 3 fails (caught exception)

    For each workers 1 - 3 returns 2 expired timers
    """

    def __init__(self, request_queue, max_interval):
        super().__init__(request_queue, max_interval)

    def register_timers(self, timer_requests):
        # 注册计时器请求的方法，此处仅占位不实现
        pass

    def clear_timers(self, worker_ids):
        # 清除计时器的方法，此处仅占位不实现
        pass

    def get_expired_timers(self, deadline):
        # 返回一个字典，模拟过期计时器的获取，每个 worker 返回两个过期的计时器请求
        return {
            i: [TimerRequest(i, f"test_{i}_0", 0), TimerRequest(i, f"test_{i}_1", 0)]
            for i in range(1, 4)
        }

    def _reap_worker(self, worker_id):
        # 根据不同的 worker_id 模拟不同的工作器行为
        if worker_id == 1:
            raise RuntimeError("test error")  # 模拟抛出运行时错误的情况
        elif worker_id == 2:
            return True  # 模拟成功 reap 的情况
        elif worker_id == 3:
            return False  # 模拟失败 reap 的情况

# TimerApiTest 类，继承自 unittest.TestCase，用于测试计时器 API
class TimerApiTest(unittest.TestCase):
    @mock.patch.object(MockTimerServer, "register_timers")
    @mock.patch.object(MockTimerServer, "clear_timers")
    def test_run_watchdog(self, mock_clear_timers, mock_register_timers):
        """
        tests that when a ``_reap_worker()`` method throws an exception
        for a particular worker_id, the timers for successfully reaped workers
        are cleared properly
        """
        max_interval = 1
        request_queue = mock.Mock(wraps=MockRequestQueue())  # 使用 MockRequestQueue 创建一个模拟的请求队列
        timer_server = MockTimerServer(request_queue, max_interval)  # 使用 MockTimerServer 创建一个模拟的计时器服务器
        timer_server._run_watchdog()  # 调用计时器服务器的 _run_watchdog 方法

        request_queue.size.assert_called_once()  # 断言请求队列的 size 方法被调用一次
        request_queue.get.assert_called_with(request_queue.size(), max_interval)  # 断言请求队列的 get 方法被正确调用
        mock_register_timers.assert_called_with(request_queue.get(2, 1))  # 断言 register_timers 方法被正确调用
        mock_clear_timers.assert_called_with({1, 2})  # 断言 clear_timers 方法被正确调用
```