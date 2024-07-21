# `.\pytorch\test\distributed\elastic\utils\distributed_test.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# 导入必要的库和模块
import multiprocessing as mp  # 多进程支持
import os  # 系统操作
import socket  # 网络通信
import sys  # 系统参数和函数
import unittest  # 单元测试框架
from contextlib import closing  # 上下文管理器

# 导入需要的 Torch 分布式模块和工具函数
from torch.distributed import DistNetworkError, DistStoreError
from torch.distributed.elastic.utils.distributed import (
    create_c10d_store,
    get_socket_with_port,
)
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    TEST_WITH_TSAN,
    TestCase,
)

# 定义一个用于在多进程中创建 C10d 存储的函数
def _create_c10d_store_mp(is_server, server_addr, port, world_size, wait_for_workers):
    # 调用工具函数创建 C10d 存储
    store = create_c10d_store(
        is_server,
        server_addr,
        port,
        world_size,
        wait_for_workers=wait_for_workers,
        timeout=2,
    )
    # 如果存储创建失败则抛出断言错误
    if store is None:
        raise AssertionError

    # 在存储中设置一个测试键值对
    store.set(f"test_key/{os.getpid()}", b"test_value")


# 如果操作系统为 Windows 或 macOS，则输出一条兼容性警告并退出
if IS_WINDOWS or IS_MACOS:
    print("tests incompatible with tsan or asan", file=sys.stderr)
    sys.exit(0)


# 定义一个测试类，用于测试分布式工具函数的功能
class DistributedUtilTest(TestCase):
    # 测试单服务器模式下创建存储
    def test_create_store_single_server(self):
        store = create_c10d_store(is_server=True, server_addr=socket.gethostname())
        self.assertIsNotNone(store)

    # 测试在多进程环境下创建存储时，未提供端口号是否会引发值错误
    def test_create_store_no_port_multi(self):
        with self.assertRaises(ValueError):
            create_c10d_store(
                is_server=True, server_addr=socket.gethostname(), world_size=2
            )

    # 跳过使用 TSAN 工具运行的测试用例
    @unittest.skipIf(TEST_WITH_TSAN, "test incompatible with tsan")
    def test_create_store_multi(self):
        world_size = 3
        wait_for_workers = False
        localhost = socket.gethostname()

        # 在主进程上使用可用端口号启动服务器
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )

        # 记录分配给服务器的端口号，工作进程将使用此端口
        server_port = store.port

        # 创建两个工作进程，并启动它们
        worker0 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size, wait_for_workers),
        )
        worker1 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size, wait_for_workers),
        )

        worker0.start()
        worker1.start()

        worker0.join()
        worker1.join()

        # 检查测试键的值是否等于预期的 "test_value"
        self.assertEqual(
            "test_value", store.get(f"test_key/{worker0.pid}").decode("UTF-8")
        )
        self.assertEqual(
            "test_value", store.get(f"test_key/{worker1.pid}").decode("UTF-8")
        )

        # 检查工作进程的退出码是否为 0
        self.assertEqual(0, worker0.exitcode)
        self.assertEqual(0, worker1.exitcode)
    def test_create_store_timeout_on_server(self):
        with self.assertRaises(DistStoreError):
            # 在服务器端创建分布式存储，期望超时
            # 使用任意可用端口（端口 0）
            create_c10d_store(
                is_server=True,
                server_addr=socket.gethostname(),
                server_port=0,
                world_size=2,
                timeout=1,
            )

    def test_create_store_timeout_on_worker(self):
        with self.assertRaises(DistNetworkError):
            # 在工作节点创建分布式存储，期望超时
            # 使用任意可用端口（端口 0）
            create_c10d_store(
                is_server=False,
                server_addr=socket.gethostname(),
                server_port=0,
                world_size=2,
                timeout=1,
            )

    def test_create_store_with_libuv_support(self):
        world_size = 1
        wait_for_workers = False
        localhost = socket.gethostname()

        # 创建支持 libuv 的 C10d 存储
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
            use_libuv=False,
        )
        self.assertFalse(store.libuvBackend)  # 确保不使用 libuv 后端

        # 创建支持 libuv 的 C10d 存储
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
            use_libuv=True,
        )
        self.assertTrue(store.libuvBackend)  # 确保使用了 libuv 后端

    def test_port_already_in_use_on_server(self):
        # 尝试在相同端口上两次创建 TCPStore 服务器
        # 第二次由于端口冲突应该失败
        # 第一个存储绑定到一个空闲端口上
        server_addr = socket.gethostname()
        pick_free_port = 0
        store1 = create_c10d_store(
            is_server=True,
            server_addr=server_addr,
            server_port=pick_free_port,
            timeout=1,
        )
        with self.assertRaises(RuntimeError):
            # 尝试在第一个存储绑定的端口上创建第二个存储
            create_c10d_store(
                is_server=True, server_addr=server_addr, server_port=store1.port
            )

    def test_port_already_in_use_on_worker(self):
        sock = get_socket_with_port()
        with closing(sock):
            port = sock.getsockname()[1]
            # 在工作节点上，端口冲突不应该影响，应该只是超时
            # 因为我们从未创建过服务器
            with self.assertRaises(DistNetworkError):
                create_c10d_store(
                    is_server=False,
                    server_addr=socket.gethostname(),
                    server_port=port,
                    timeout=1,
                )
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```