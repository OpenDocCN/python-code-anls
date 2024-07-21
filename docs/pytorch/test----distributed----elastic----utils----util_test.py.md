# `.\pytorch\test\distributed\elastic\utils\util_test.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime  # 导入处理日期时间的模块
from multiprocessing.pool import ThreadPool  # 导入多线程池模块
from typing import List  # 导入类型提示模块

import torch.distributed as dist  # 导入PyTorch分布式模块

import torch.distributed.elastic.utils.store as store_util  # 导入自定义存储工具模块
from torch.distributed.elastic.utils.logging import get_logger  # 导入日志记录模块
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关模块


class MockStore:
    def __init__(self):
        self.ops = []  # 初始化操作记录列表

    def set_timeout(self, timeout: float) -> None:
        self.ops.append(("set_timeout", timeout))  # 记录设置超时操作

    @property
    def timeout(self) -> datetime.timedelta:
        self.ops.append(("timeout",))  # 记录获取超时时间操作

        return datetime.timedelta(seconds=1234)  # 返回固定的时间间隔

    def set(self, key: str, value: str) -> None:
        self.ops.append(("set", key, value))  # 记录设置键值对操作

    def get(self, key: str) -> str:
        self.ops.append(("get", key))  # 记录获取键值对操作
        return "value"  # 返回固定的值作为示例

    def multi_get(self, keys: List[str]) -> List[str]:
        self.ops.append(("multi_get", keys))  # 记录批量获取键值对操作
        return ["value"] * len(keys)  # 返回固定的值列表，长度与输入键列表相同

    def add(self, key: str, val: int) -> int:
        self.ops.append(("add", key, val))  # 记录增加操作
        return 3  # 返回固定的整数作为示例


class StoreUtilTest(TestCase):
    def test_get_all_rank_0(self):
        world_size = 3

        store = MockStore()  # 创建模拟存储对象

        store_util.get_all(store, 0, "test/store", world_size)  # 调用测试函数获取所有数据

        self.assertListEqual(
            store.ops,
            [
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),  # 批量获取指定键的值
                ("add", "test/store/finished/num_members", 1),  # 增加成员数量计数器
                ("set", "test/store/finished/last_member", "<val_ignored>"),  # 设置最后一个成员信息
                ("get", "test/store/finished/last_member"),  # 获取最后一个成员信息
            ],
        )

    def test_get_all_rank_n(self):
        store = MockStore()  # 创建模拟存储对象
        world_size = 3
        store_util.get_all(store, 1, "test/store", world_size)  # 调用测试函数获取所有数据

        self.assertListEqual(
            store.ops,
            [
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),  # 批量获取指定键的值
                ("add", "test/store/finished/num_members", 1),  # 增加成员数量计数器
                ("set", "test/store/finished/last_member", "<val_ignored>"),  # 设置最后一个成员信息
            ],
        )
    # 定义一个名为 test_synchronize 的测试方法，测试 synchronize 函数
    def test_synchronize(self):
        # 创建一个 MockStore 对象作为存储模拟
        store = MockStore()

        # 定义一个名为 data 的字节串数据
        data = b"data0"
        # 调用 synchronize 函数，将数据同步到 store 中，指定 key 前缀为 "test/store"
        store_util.synchronize(store, data, 0, 3, key_prefix="test/store")

        # 使用断言检查 store.ops 的值是否符合预期
        self.assertListEqual(
            store.ops,
            [
                ("timeout",),  # 设置超时操作
                ("set_timeout", datetime.timedelta(seconds=300)),  # 设置超时时间为 300 秒
                ("set", "test/store0", data),  # 设置键为 "test/store0" 的值为 data
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),  # 批量获取指定键的值
                ("add", "test/store/finished/num_members", 1),  # 将 "test/store/finished/num_members" 的值加 1
                ("set", "test/store/finished/last_member", "<val_ignored>"),  # 设置 "test/store/finished/last_member" 的值为 "<val_ignored>"
                ("get", "test/store/finished/last_member"),  # 获取 "test/store/finished/last_member" 的值
                ("set_timeout", datetime.timedelta(seconds=1234)),  # 设置超时时间为 1234 秒
            ],
        )

    # 定义一个名为 test_synchronize_hash_store 的测试方法，测试 synchronize 函数与 HashStore
    def test_synchronize_hash_store(self) -> None:
        # 定义常量 N，值为 4
        N = 4

        # 创建一个 HashStore 对象作为分布式存储
        store = dist.HashStore()

        # 定义一个函数 f，用于调用 synchronize 函数，将不同的数据同步到 store 中，指定 key 前缀为 "test/store"
        def f(i: int):
            return store_util.synchronize(
                store, f"data{i}", i, N, key_prefix="test/store"
            )

        # 使用线程池 ThreadPool 创建具有 N 个线程的线程池
        with ThreadPool(N) as pool:
            # 使用 map 函数并发执行函数 f，将结果保存在 out 中
            out = pool.map(f, range(N))

        # 使用断言检查 out 的值是否符合预期，每个元素都是一个字节串数据
        self.assertListEqual(out, [[f"data{i}".encode() for i in range(N)]] * N)

    # 定义一个名为 test_barrier 的测试方法，测试 barrier 函数
    def test_barrier(self):
        # 创建一个 MockStore 对象作为存储模拟
        store = MockStore()

        # 调用 barrier 函数，设置屏障，要求等待 3 个成员到达，指定 key 前缀为 "test/store"
        store_util.barrier(store, 3, key_prefix="test/store")

        # 使用断言检查 store.ops 的值是否符合预期
        self.assertListEqual(
            store.ops,
            [
                ("timeout",),  # 设置超时操作
                ("set_timeout", datetime.timedelta(seconds=300)),  # 设置超时时间为 300 秒
                ("add", "test/store/num_members", 1),  # 将 "test/store/num_members" 的值加 1
                ("set", "test/store/last_member", "<val_ignored>"),  # 设置 "test/store/last_member" 的值为 "<val_ignored>"
                ("get", "test/store/last_member"),  # 获取 "test/store/last_member" 的值
                ("set_timeout", datetime.timedelta(seconds=1234)),  # 设置超时时间为 1234 秒
            ],
        )

    # 定义一个名为 test_barrier_hash_store 的测试方法，测试 barrier 函数与 HashStore
    def test_barrier_hash_store(self) -> None:
        # 定义常量 N，值为 4
        N = 4

        # 创建一个 HashStore 对象作为分布式存储
        store = dist.HashStore()

        # 定义一个函数 f，用于调用 barrier 函数，设置屏障，等待 N 个成员到达，指定 key 前缀为 "test/store"
        def f(i: int):
            store_util.barrier(store, N, key_prefix="test/store")

        # 使用线程池 ThreadPool 创建具有 N 个线程的线程池
        with ThreadPool(N) as pool:
            # 使用 map 函数并发执行函数 f，将结果保存在 out 中
            out = pool.map(f, range(N))

        # 使用断言检查 out 的值是否符合预期，每个元素都应为 None
        self.assertEqual(out, [None] * N)
# 定义一个测试类 UtilTest，继承自 TestCase，用于测试日志记录器功能
class UtilTest(TestCase):

    # 测试函数：测试获取不同名称的日志记录器，确保它们的名称不相同
    def test_get_logger_different(self):
        # 调用 get_logger 函数，获取名为 "name1" 的日志记录器
        logger1 = get_logger("name1")
        # 再次调用 get_logger 函数，获取名为 "name2" 的日志记录器
        logger2 = get_logger("name2")
        # 断言：确保两个日志记录器的名称不相同
        self.assertNotEqual(logger1.name, logger2.name)

    # 测试函数：测试获取默认名称的日志记录器，确保其名称与当前模块的名称相同
    def test_get_logger(self):
        # 调用 get_logger 函数，获取默认名称的日志记录器
        logger1 = get_logger()
        # 断言：确保获取的日志记录器名称与当前模块的名称相同
        self.assertEqual(__name__, logger1.name)

    # 测试函数：测试获取空名称的日志记录器，确保其名称与当前模块的名称相同
    def test_get_logger_none(self):
        # 调用 get_logger 函数，获取空名称的日志记录器
        logger1 = get_logger(None)
        # 断言：确保获取的日志记录器名称与当前模块的名称相同
        self.assertEqual(__name__, logger1.name)

    # 测试函数：测试获取自定义名称的日志记录器，确保其名称与指定的名称相同
    def test_get_logger_custom_name(self):
        # 调用 get_logger 函数，获取名为 "test.module" 的日志记录器
        logger1 = get_logger("test.module")
        # 断言：确保获取的日志记录器名称与指定的名称相同
        self.assertEqual("test.module", logger1.name)


# 如果该脚本作为主程序执行，则运行所有的测试用例
if __name__ == "__main__":
    run_tests()
```