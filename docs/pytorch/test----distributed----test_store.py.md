# `.\pytorch\test\distributed\test_store.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的标准库和第三方库
import datetime
import os
import socket
import sys
import tempfile
import threading
import time
from datetime import timedelta
from sys import platform

# 导入 PyTorch 分布式相关模块
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.distributed.rpc as rpc
from torch.distributed import DistError, DistNetworkError, DistStoreError
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests

# 如果 torch.distributed 不可用，则输出消息并退出
if not dist.is_available():
    print("torch.distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入其他必要的测试辅助函数和类
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import (
    create_tcp_store,
    skip_if_win32,
    tp_transports,
)
from torch.testing._internal.common_utils import (
    ADDRESS_IN_USE,
    CONNECT_TIMEOUT,
    load_tests,
    retry_on_connect_failures,
    run_tests,
    TestCase,
)

# load_tests 函数用于在 sandcastle 上自动过滤测试用例，此行代码用于消除 flake 警告
load_tests = load_tests

# 根据平台设置回环接口的名称
if platform == "darwin":
    LOOPBACK = "lo0"
else:
    LOOPBACK = "lo"

# 默认的主机名为 localhost
DEFAULT_HOSTNAME = "localhost"

# 禁用 CUDA 的 TF32 模式
torch.backends.cuda.matmul.allow_tf32 = False


def gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    # 获取所有可见的 GPU 设备列表
    visible_devices = list(range(torch.cuda.device_count()))
    # 计算每个进程应使用的 GPU 数量
    gpus_per_process = torch.cuda.device_count() // world_size
    gpus_for_rank = []
    # 遍历每个进程的排名，确定其使用的 GPU 子集
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


class StoreTestBase:
    def _create_store(self, i):
        raise RuntimeError("not implemented")
    # 测试设置、获取和检查操作的方法
    def _test_set_get_check(self, fs):
        # 向存储中添加键 "key" 的多个值
        fs.add("key", 1)
        fs.add("key", 2)
        fs.add("key", 3)
        # 设置键 "key0" 的值为 "value0"
        fs.set("key0", "value0")
        fs.add("key3", 1)
        # 设置键 "key1" 的值为 "value1"
        fs.set("key1", "value1")
        fs.add("key3", 2)
        # 设置键 "key2" 的值为 "value2"
        fs.set("key2", "value2")
        fs.add("key3", 3)
        fs.add("key3", 4)
        fs.add("key3", 5)
        fs.add("key3", 6)
        # 断言键的数量与预期的总数相等
        self.assertEqual(fs.num_keys(), self.num_keys_total)
        # 断言获取键 "key" 的值为 b"6"
        self.assertEqual(b"6", fs.get("key"))
        # 断言获取键 "key0" 的值为 b"value0"
        self.assertEqual(b"value0", fs.get("key0"))
        # 断言获取键 "key1" 的值为 b"value1"
        self.assertEqual(b"value1", fs.get("key1"))
        # 断言获取键 "key2" 的值为 b"value2"
        self.assertEqual(b"value2", fs.get("key2"))
        # 断言获取键 "key3" 的值为 b"21"
        self.assertEqual(b"21", fs.get("key3"))
        # 检查键 "key3" 是否存在于存储中，预期为 True
        self.assertTrue(fs.check(["key3"]))
        # 检查键 "Randomkey3" 是否存在于存储中，预期为 False
        self.assertFalse(fs.check(["Randomkey3"]))

        # 将键 "-key3" 的值设置为 "7"
        fs.set("-key3", "7")
        # 断言获取键 "-key3" 的值为 b"7"
        self.assertEqual(b"7", fs.get("-key3"))
        # 删除键 "-key3"
        fs.delete_key("-key3")
        # 断言键的数量与预期的总数相等
        self.assertEqual(fs.num_keys(), self.num_keys_total)

    # 测试设置、获取和检查操作的方法
    def test_set_get_check(self):
        # 使用内部方法 _create_store() 创建存储并进行测试
        self._test_set_get_check(self._create_store())

    # 测试比较设置操作的方法
    def _test_compare_set(self, store):
        # 测试在键 "cs_key0" 不存在时的比较设置操作
        missing_key_result = store.compare_set(
            "cs_key0", "wrong_old_value", "new_value0"
        )
        # 断言比较设置返回的旧值为 b"wrong_old_value"
        self.assertEqual(b"wrong_old_value", missing_key_result)

        # 设置键 "cs_key0" 的值为 "value0"
        store.set("cs_key0", "value0")
        # 断言获取键 "cs_key0" 的值为 b"value0"
        self.assertEqual(b"value0", store.get("cs_key0"))
        # 测试在键 "cs_key0" 的旧值为 "wrong_old_value" 时的比较设置操作
        old_value_result = store.compare_set("cs_key0", "wrong_old_value", "new_value0")
        # 断言比较设置返回的旧值为 b"value0"
        self.assertEqual(b"value0", old_value_result)
        # 断言获取键 "cs_key0" 的值为 b"value0"
        self.assertEqual(b"value0", store.get("cs_key0"))
        # 测试在键 "cs_key0" 的旧值为 "value0" 时的比较设置操作
        new_value_result = store.compare_set("cs_key0", "value0", "new_value0")
        # 断言比较设置返回的新值为 b"new_value0"
        self.assertEqual(b"new_value0", new_value_result)
        # 断言获取键 "cs_key0" 的值为 b"new_value0"
        self.assertEqual(b"new_value0", store.get("cs_key0"))
        # 测试在键 "cs_key1" 不存在且旧值为空字符串时的比较设置操作
        empty_old_value_result = store.compare_set("cs_key1", "", "new_value1")
        # 断言比较设置返回的新值为 b"new_value1"
        self.assertEqual(b"new_value1", empty_old_value_result)
        # 断言获取键 "cs_key1" 的值为 b"new_value1"

    # 测试比较设置操作的方法
    def test_compare_set(self):
        # 使用内部方法 _create_store() 创建存储并进行测试
        self._test_compare_set(self._create_store())

    # 测试简单等待操作的方法
    def _test_simple_wait(self, fs):
        # 使用断言检测预期的运行时错误是否被抛出
        with self.assertRaisesRegex(RuntimeError, "[t -i]imeout"):
            fs.wait(["bad_key"], timedelta(seconds=0.25))
        # 向存储中添加键 "good_key" 的值为 1
        fs.add("good_key", 1)
        # 等待存储中键 "good_key" 的存在

    # 测试简单等待操作的方法
    def test_simple_wait(self):
        # 使用内部方法 _create_store() 创建存储并进行测试
        self._test_simple_wait(self._create_store())

    # 测试追加操作的方法
    def _test_append(self, store):
        # 如果存储不支持扩展 API，则跳过测试
        if not store.has_extended_api():
            self.skipTest("Store doesn't support extended APIs")
        # 设置键 "foo" 的值为 "po"
        store.set("foo", "po")
        # 向键 "foo" 追加值 "tato"
        store.append("foo", "tato")
        # 向键 "bar" 追加值 "po"
        store.append("bar", "po")
        # 向键 "bar" 追加值 "tato"
        store.append("bar", "tato")
        # 断言获取键 "foo" 的值为 b"potato"
        self.assertEqual(b"potato", store.get("foo"))
        # 断言获取键 "bar" 的值为 b"potato"
        self.assertEqual(b"potato", store.get("bar"))

    # 测试追加操作的方法
    def test_append(self):
        # 使用内部方法 _create_store() 创建存储并进行测试
        self._test_append(self._create_store())
    # 定义一个名为 _test_multi_set 的测试方法，接受一个 store 参数
    def _test_multi_set(self, store):
        # 检查存储是否支持扩展 API，如果不支持则跳过测试
        if not store.has_extended_api():
            self.skipTest("Store doesn't support extended APIs")
        # 使用 multi_set 方法批量设置键值对 "foo" -> "po", "bar" -> "tato"
        store.multi_set(["foo", "bar"], ["po", "tato"])
        # 断言 "foo" 键的值为 b"po"
        self.assertEqual(b"po", store.get("foo"))
        # 断言 "bar" 键的值为 b"tato"
        self.assertEqual(b"tato", store.get("bar"))

    # 定义名为 test_multi_set 的测试方法
    def test_multi_set(self):
        # 调用 _create_store 方法创建一个存储对象，并传递给 _test_multi_set 测试
        self._test_multi_set(self._create_store())

    # 定义一个名为 _test_multi_get 的测试方法，接受一个 store 参数
    def _test_multi_get(self, store):
        # 检查存储是否支持扩展 API，如果不支持则跳过测试
        if not store.has_extended_api():
            self.skipTest("Store doesn't support extended APIs")
        # 设置键 "foo" 的值为 "po"，键 "bar" 的值为 "tato"
        store.set("foo", "po")
        store.set("bar", "tato")
        # 使用 multi_get 方法获取多个键的值，返回值分别赋给 v0 和 v1
        v0, v1 = store.multi_get(["foo", "bar"])
        # 断言 "foo" 键的值为 b"po"
        self.assertEqual(b"po", v0)
        # 断言 "bar" 键的值为 b"tato"
        self.assertEqual(b"tato", v1)

    # 定义名为 test_multi_get 的测试方法
    def test_multi_get(self):
        # 调用 _create_store 方法创建一个存储对象，并传递给 _test_multi_get 测试
        self._test_multi_get(self._create_store())

    # 这是在 test_set_get 方法中使用的键的数量。将其作为类属性，而不是硬编码在测试中，
    # 因为某些 Store 实现可能有不同数量的键。基本情况下将有 5 个键: key, key0, key1, key2, key3。
    @property
    def num_keys_total(self):
        # 返回值为 5，表示测试用例中使用的键的总数
        return 5
class FileStoreTest(TestCase, StoreTestBase):
    # 设置测试前的准备工作，继承自 TestCase 和 StoreTestBase
    def setUp(self):
        super().setUp()
        # 创建一个临时文件，并且设置不自动删除
        self.file = tempfile.NamedTemporaryFile(delete=False)

    # 创建一个 FileStore 实例的辅助方法
    def _create_store(self):
        # 使用临时文件名创建 FileStore 对象，设置版本号为 1
        store = dist.FileStore(self.file.name, 1)
        # 设置存储的超时时间为 300 秒
        store.set_timeout(timedelta(seconds=300))
        return store

    # 测试使用相同文件进行初始化 RPC 和 PG
    def test_init_pg_and_rpc_with_same_file(self):
        file = tempfile.NamedTemporaryFile(delete=False)
        # 使用文件初始化 RPC
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
        rpc_backend_options.init_method = f"file://{file.name}"
        rpc_backend_options._transports = tp_transports()
        rpc.init_rpc(
            "worker", rank=0, world_size=1, rpc_backend_options=rpc_backend_options
        )

        # 使用文件初始化 PG
        dist.init_process_group(
            "gloo", rank=0, world_size=1, init_method=f"file://{file.name}"
        )
        dist.destroy_process_group()
        # 断言文件确实存在
        assert os.path.exists(file.name)

        # 关闭 RPC
        rpc.shutdown()
        # 移除临时文件
        os.remove(file.name)

    # 测试引用计数
    def test_refcount(self):
        file = tempfile.NamedTemporaryFile(delete=False)
        # 创建 FileStore 实例
        store = dist.FileStore(file.name, 1)
        store2 = dist.FileStore(file.name, 1)

        # 删除第一个 FileStore 实例
        del store
        # 断言文件存在
        assert os.path.exists(file.name)
        # 删除第二个 FileStore 实例
        del store2
        # 断言文件不存在
        assert not os.path.exists(file.name)

    # 返回总键数的属性方法
    @property
    def num_keys_total(self):
        return 6


@skip_if_win32()
class HashStoreTest(TestCase, StoreTestBase):
    # 创建 HashStore 的测试类，继承自 TestCase 和 StoreTestBase
    def _create_store(self):
        # 创建 HashStore 实例
        store = dist.HashStore()
        # 设置存储的超时时间为 300 秒
        store.set_timeout(timedelta(seconds=300))
        return store


class PrefixStoreTest(TestCase):
    # 创建 PrefixStore 的测试类，继承自 TestCase
    def setUp(self):
        # 设置测试前的准备工作，创建一个临时文件并且不自动删除
        self.file = tempfile.NamedTemporaryFile(delete=False)

    # 测试获取底层存储的方法
    def test_get_underlying_store(self):
        # 创建 TCPStore 实例
        tcp_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME, port=0, world_size=1, is_master=True
        )
        # 创建 HashStore 实例
        hash_store = dist.HashStore()
        # 创建 FileStore 实例
        file_store = dist.FileStore(self.file.name, world_size=1)
        for store in [tcp_store, hash_store, file_store]:
            with self.subTest(f"Testing getting underlying_store for {type(store)}"):
                # 创建 PrefixStore 实例
                prefix_store = dist.PrefixStore("prefix", store)
                # 断言 PrefixStore 的底层存储与当前 store 相同
                self.assertEqual(prefix_store.underlying_store, store)


class PrefixFileStoreTest(TestCase, StoreTestBase):
    # 创建 PrefixFileStore 的测试类，继承自 TestCase 和 StoreTestBase
    def setUp(self):
        super().setUp()
        # 设置测试前的准备工作，创建一个临时文件并且不自动删除
        self.file = tempfile.NamedTemporaryFile(delete=False)
        # 创建 FileStore 实例
        self.filestore = dist.FileStore(self.file.name, 1)
        self.prefix = "test_prefix"
        # 设置存储的超时时间为 300 秒
        self.filestore.set_timeout(timedelta(seconds=300))

    # 创建 PrefixStore 实例的辅助方法
    def _create_store(self):
        return dist.PrefixStore(self.prefix, self.filestore)

    # 返回总键数的属性方法
    @property
    def num_keys_total(self):
        return 6


class TCPStoreTest(TestCase, StoreTestBase):
    # TCPStore 测试类，继承自 TestCase 和 StoreTestBase
    _use_libuv = False
    # 创建一个 TCP 存储对象，使用指定的库（如果启用 libuv）
    def _create_store(self):
        store = create_tcp_store(use_libuv=self._use_libuv)
        # 设置存储对象超时时间为 300 秒
        store.set_timeout(timedelta(seconds=300))
        return store

    # 创建一个带有 WebSocket 的 TCP 存储对象
    def _create_store_with_ws(self, addr, world_size):
        return create_tcp_store(
            addr, world_size, wait_for_workers=False, use_libuv=self._use_libuv
        )

    # 测试当地址已被使用时的行为
    def test_address_already_in_use(self):
        # 正则表达式用于匹配运行时错误消息
        err_msg_reg = "^The server socket has failed to listen on any local "
        with self.assertRaisesRegex(RuntimeError, err_msg_reg):
            addr = DEFAULT_HOSTNAME
            port = common.find_free_port()

            # Use noqa to silence flake8.
            # 需要将对象存储在未使用的变量中，以确保第一个对象在第二个对象创建之前不会被销毁。
            store1 = dist.TCPStore(
                addr, port, 1, True, use_libuv=self._use_libuv
            )  # noqa: F841
            store2 = dist.TCPStore(
                addr, port, 1, True, use_libuv=self._use_libuv
            )  # noqa: F841
            # 断言两个存储对象使用的 libuv 后端与预期相符
            self.assertEqual(store1.libuvBackend, self._use_libuv)
            self.assertEqual(store2.libuvBackend, self._use_libuv)

    # 在连接失败时进行重试的多租户测试
    @retry_on_connect_failures
    def test_multitenancy(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        # Use noqa to silence flake8.
        # 需要将对象存储在未使用的变量中，以确保第一个对象在第二个对象创建之前不会被销毁。
        store1 = dist.TCPStore(
            addr, port, 1, True, multi_tenant=True, use_libuv=self._use_libuv
        )  # type: ignore[call-arg] # noqa: F841
        store2 = dist.TCPStore(
            addr, port, 1, True, multi_tenant=True, use_libuv=self._use_libuv
        )  # type: ignore[call-arg] # noqa: F841
        # 断言两个存储对象使用的 libuv 后端与预期相符
        self.assertEqual(store1.libuvBackend, self._use_libuv)
        self.assertEqual(store2.libuvBackend, self._use_libuv)

    # 如果运行环境是 Windows，跳过测试
    @skip_if_win32()
    @retry_on_connect_failures
    def test_init_pg_and_rpc_with_same_socket(self):
        # 设置默认地址和查找空闲端口
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        # 设置环境变量以指定主地址和端口号
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        # 设置使用的库，根据 self._use_libuv 的值
        os.environ["USE_LIBUV"] = "1" if self._use_libuv else "0"
        
        # 初始化分布式进程组，使用 'gloo' 后端，使用环境变量的方式
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=0,
            world_size=1,
        )

        # 配置 TensorPipe RPC 后端选项，使用给定的地址和端口
        backend_opts = rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{addr}:{port}", _transports=tp_transports()
        )
        
        # 初始化 RPC，指定名称、等级和总体规模
        rpc.init_rpc(
            name="worker0",
            rank=0,
            world_size=1,
            rpc_backend_options=backend_opts,
        )

        # 清除环境变量 'USE_LIBUV'
        del os.environ["USE_LIBUV"]
        
        # 断言检查 'USE_LIBUV' 是否不在环境变量中
        assert "USE_LIBUV" not in os.environ
        
        # 关闭 RPC
        rpc.shutdown()
        
        # 销毁分布式进程组
        dist.destroy_process_group()

    @skip_if_win32()
    def test_take_over_listen_socket(self):
        # 创建监听套接字并绑定到本地地址的随机端口
        listen_sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.bind(("localhost", 0))
        addr, port, *_ = listen_sock.getsockname()
        
        # 分离监听套接字的文件描述符
        listen_fd = listen_sock.detach()

        # 创建 TCPStore 对象，使用给定的地址、端口和文件描述符，设置为主节点，根据 self._use_libuv 的值
        store = dist.TCPStore(
            addr,
            port,
            1,
            is_master=True,
            master_listen_fd=listen_fd,
            use_libuv=self._use_libuv,
        )

        # 断言检查库的后端是否与 self._use_libuv 的值一致
        self.assertEqual(store.libuvBackend, self._use_libuv)
        
        # 设置键值对到存储中
        store.set("key", "value")
        
        # 断言检查从存储中获取的值是否与预期相符
        self.assertEqual(b"value", store.get("key"))

    # 在 test_set_get 中，TCPStore 包含 6 个键。其中 5 个键是用户添加的，还有一个额外的键用于协调所有的工作节点。
    @property
    def num_keys_total(self):
        # 返回总键数为 6
        return 6

    def _test_numkeys_delkeys(self, fs):
        # 初始化时，存储中有一个初始键用于协调工作节点
        self.assertEqual(fs.num_keys(), 1)
        
        # 添加键 'key' 的多个值
        fs.add("key", 1)
        fs.add("key", 2)
        fs.add("key", 3)
        
        # 设置键 'key0' 的值
        fs.set("key0", "value0")
        
        # 添加键 'key3' 的值
        fs.add("key3", 1)
        
        # 设置键 'key1' 的值
        fs.set("key1", "value1")
        
        # 断言检查当前存储中的键数为 5
        self.assertEqual(fs.num_keys(), 5)
        
        # 删除键 'key'
        fs.delete_key("key")
        
        # 断言检查当前存储中的键数为 4
        self.assertEqual(fs.num_keys(), 4)
        
        # 设置存储的超时时间为 2 秒
        fs.set_timeout(timedelta(seconds=2))
        
        # 使用断言检查运行时错误是否被引发
        with self.assertRaises(RuntimeError):
            fs.get("key")
        
        # 删除键 'key0' 和 'key3'
        fs.delete_key("key0")
        fs.delete_key("key3")
        
        # 断言检查当前存储中的键数为 2
        self.assertEqual(fs.num_keys(), 2)
        
        # 设置键 'key4' 的值
        fs.set("key4", "value2")
        
        # 断言检查当前存储中的键数为 3
        self.assertEqual(fs.num_keys(), 3)
        
        # 断言检查从存储中获取键 'key1' 的值是否与预期相符
        self.assertEqual(b"value1", fs.get("key1"))
        
        # 断言检查从存储中获取键 'key4' 的值是否与预期相符
        self.assertEqual(b"value2", fs.get("key4"))

    def test_numkeys_delkeys(self):
        # 执行 _test_numkeys_delkeys 方法，传入一个创建的存储对象
        self._test_numkeys_delkeys(self._create_store())
    `
        # 创建一个客户端的私有方法，用于与指定地址和端口的服务器进行通信
        def _create_client(self, index, addr, port, world_size):
            # 创建一个基于TCP的分布式存储客户端
            client_store = dist.TCPStore(
                addr,
                port,
                world_size=world_size,
                timeout=timedelta(seconds=10),
                use_libuv=self._use_libuv,
            )
            # 断言客户端从服务器存储中获取的值为"value"
            self.assertEqual(b"value", client_store.get("key"))
            # 向服务器存储中设置新的键值对
            client_store.set(f"new_key{index}", f"new_value{index}")
            # 断言使用CAS操作成功设置的值与预期值相匹配
            self.assertEqual(
                f"next_value{index}".encode(),
                client_store.compare_set(
                    f"new_key{index}", f"new_value{index}", f"next_value{index}"
                ),
            )
    
        # 辅助方法，用于在多个工作节点之间执行测试
        def _multi_worker_helper(self, world_size):
            # 默认主机名
            addr = DEFAULT_HOSTNAME
            # 使用指定的世界大小创建服务器存储
            server_store = self._create_store_with_ws(addr, world_size)
            # 断言服务器存储的libuv后端与预期的使用状态相符
            self.assertEqual(server_store.libuvBackend, self._use_libuv)
            # 在服务器存储中设置键值对
            server_store.set("key", "value")
            # 获取服务器存储的端口号
            port = server_store.port
    
            # 确定要创建的客户端数量
            num_indices = world_size if world_size else 1
            # 循环创建每个客户端并与服务器进行通信
            for i in range(num_indices):
                self._create_client(i, addr, port, world_size)
    
        # 测试固定世界大小的多工作节点情况
        def test_multi_worker_with_fixed_world_size(self):
            self._multi_worker_helper(5)
    
        # 测试非固定世界大小的多工作节点情况
        def test_multi_worker_with_nonfixed_world_size(self):
            self._multi_worker_helper(None)
    
        # 测试在存储中追加值的情况
        def test_append(self):
            store = self._create_store()
            # 断言存储的libuv后端与预期的使用状态相符
            self.assertEqual(store.libuvBackend, self._use_libuv)
            # 设置存储中的键值对
            store.set("foo", "po")
            # 在已有键的值后追加新值
            store.append("foo", "tato")
            store.append("bar", "po")
            store.append("bar", "tato")
            # 断言获取到的键的值与预期值匹配
            self.assertEqual(b"potato", store.get("foo"))
            self.assertEqual(b"potato", store.get("bar"))
    
        # 测试在存储中批量设置值的情况
        def test_multi_set(self):
            store = self._create_store()
            # 断言存储的libuv后端与预期的使用状态相符
            self.assertEqual(store.libuvBackend, self._use_libuv)
            # 批量设置存储中的多个键值对
            store.multi_set(["foo", "bar"], ["po", "tato"])
            # 断言获取到的键的值与预期值匹配
            self.assertEqual(b"po", store.get("foo"))
            self.assertEqual(b"tato", store.get("bar"))
    
        # 测试在存储中批量获取值的情况
        def test_multi_get(self):
            store = self._create_store()
            # 断言存储的libuv后端与预期的使用状态相符
            self.assertEqual(store.libuvBackend, self._use_libuv)
            # 设置存储中的键值对
            store.set("foo", "po")
            store.set("bar", "tato")
            # 批量获取存储中多个键的值
            v0, v1 = store.multi_get(["foo", "bar"])
            # 断言获取到的值与预期值匹配
            self.assertEqual(b"po", v0)
            self.assertEqual(b"tato", v1)
    def test_store_timeout_on_missing_clients(self):
        with self.assertRaisesRegex(
            DistStoreError,  # 使用assertRaisesRegex断言捕获DistStoreError异常
            r"Timed out after \d+ seconds waiting for clients. \d+/\d+ clients joined.",  # 断言异常消息的正则表达式模式
        ):
            # world_size is 2 so it should timeout
            dist.TCPStore(
                "localhost",  # 主机名为localhost
                0,  # 端口号为0
                2,  # world_size为2
                True,  # async_store为True
                timeout=timedelta(seconds=2),  # 设置超时时间为2秒
                use_libuv=self._use_libuv,  # 根据self._use_libuv确定是否使用libuv
            )

        # when wait_for_workers is not set, then there should be no exception raised
        dist.TCPStore(
            "localhost",  # 主机名为localhost
            0,  # 端口号为0
            2,  # world_size为2
            True,  # async_store为True
            timeout=timedelta(seconds=2),  # 设置超时时间为2秒
            wait_for_workers=False,  # 设置wait_for_workers为False，不等待workers加入
            use_libuv=self._use_libuv,  # 根据self._use_libuv确定是否使用libuv
        )
class LibUvTCPStoreTest(TCPStoreTest):
    _use_libuv = True  # 设置一个类变量，表示使用 libuv

    def _create_store(self):
        store = create_tcp_store(use_libuv=True)  # 创建一个 TCP 存储对象，使用 libuv
        store.set_timeout(timedelta(seconds=300))  # 设置存储对象的超时时间为 300 秒
        return store

    def _create_store_with_ws(self, addr, world_size):
        return create_tcp_store(
            addr, world_size, wait_for_workers=False, use_libuv=True
        )  # 创建一个带有指定地址和世界大小的 TCP 存储对象，不等待工作节点连接，使用 libuv

    def test_take_over_listen_socket(self):
        """
        override the take_over_listen_socket test in TCPStoreTest.
        Reason: we have not thoroughly tested libuv TCPStore initialization using
        open Socket so we decide to not support this use for now.
        TODO (xilunwu): enable this use case
        """
        listen_sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.bind(("localhost", 0))  # 将监听套接字绑定到本地地址的任意可用端口
        addr, port, *_ = listen_sock.getsockname()  # 获取绑定后的地址和端口号
        listen_fd = listen_sock.detach()  # 分离监听套接字的文件描述符

        err_msg_reg = (
            "^The libuv TCPStore backend does not support "
            "initialization with an listen fd"
        )  # 定义一个错误消息的正则表达式，用于捕获不支持监听文件描述符初始化的异常信息

        with self.assertRaisesRegex(NotImplementedError, err_msg_reg):
            store = dist.TCPStore(
                addr,
                port,
                1,
                is_master=True,
                master_listen_fd=listen_fd,
                use_libuv=self._use_libuv,
            )  # 创建一个 TCP 存储对象，期望捕获指定错误类型和消息的异常

class PrefixTCPStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super().setUp()
        self.tcpstore = create_tcp_store()  # 创建一个 TCP 存储对象
        self.prefix = "test_prefix"  # 设置存储前缀
        self.tcpstore.set_timeout(timedelta(seconds=300))  # 设置存储对象的超时时间为 300 秒

    def _create_store(self):
        return dist.PrefixStore(self.prefix, self.tcpstore)  # 创建一个带前缀的前缀存储对象，使用现有的 TCP 存储对象

    # The PrefixTCPStore has 6 keys in test_set_get. It contains the 5 keys
    # added by the user and one additional key used for coordinate all the
    # workers.
    @property
    def num_keys_total(self):
        return 6  # 返回 PrefixTCPStore 在 test_set_get 中包含的键的总数

    def test_underlying_non_prefix_store(self):
        store = self._create_store()
        wrapped_store = dist.PrefixStore(
            self.prefix, dist.PrefixStore(self.prefix, store)
        )  # 创建一个嵌套的前缀存储对象，内部包含包含前缀的前缀存储对象
        self.assertEqual(self.tcpstore, store._underlying_non_prefix_store)  # 断言内部非前缀存储对象与初始化的 TCP 存储对象相等
        self.assertEqual(self.tcpstore, wrapped_store._underlying_non_prefix_store)  # 断言嵌套存储对象的内部非前缀存储对象与初始化的 TCP 存储对象相等


class MyPythonStore(dist.Store):
    def __init__(self):
        super().__init__()
        self.store = {}  # 初始化一个空字典作为存储对象

    def set(self, key, value):
        if not isinstance(key, (str, bytes)):
            raise AssertionError("Expected set to be called with string key")  # 如果键不是字符串或字节，则抛出断言错误
        if type(value) is not bytes:
            raise AssertionError("Expected set to be called with bytes value")  # 如果值不是字节，则抛出断言错误
        self.store[key] = value  # 将键值对存储到内部字典中

    def get(self, key):
        value = self.store.get(key, b"")  # 获取指定键的值，若不存在则返回空字节串
        if type(value) is not bytes:
            raise AssertionError("Expected get to return bytes value")  # 如果值不是字节，则抛出断言错误
        return value  # 返回获取到的值
    # 在给定键（key）的存储中，将一个整数值加到现有值上，并返回新值
    def add(self, key, value):
        # 获取键对应的当前值，如果不存在则默认为0，加上新值
        new = int(self.store.get(key, 0)) + value
        # 将新值转换为字节类型并设置到存储中
        self.set(key, bytes(str(new).encode("utf-8")))
        # 返回更新后的值
        return new

    # 比较存储中指定键的当前值是否与期望值相同，如果相同则更新为新值
    def compare_set(self, key, expected, newValue):
        # 如果期望值不是字节类型，则抛出异常
        if type(expected) is not bytes:
            raise AssertionError("compare_set::expected not bytes")
        # 如果新值不是字节类型，则抛出异常
        if type(newValue) is not bytes:
            raise AssertionError("compare_set::newValue not bytes")

        # 获取指定键的当前值
        val = self.store.get(key, None)
        # 如果当前值等于期望值或者当前值为None，则更新存储中的值为新值
        if expected == val or val is None:
            val = self.store[key] = newValue
        # 返回更新后的值
        return val
class PythonStoreTest(TestCase):
    def test_set_get(self):
        # 如果我们从 StoreTestBase 继承并尝试使用其 test_set_get 函数，
        # 那么我们将直接调用 Python API，而不是通过 C++ 的 trampoline。
        # 我们关心测试 C++ 的 trampoline，因此运行与 StoreTestBase.test_set_get
        # 相当的函数，从而从 C++ 中执行。
        # 参见 `torch/csrc/distributed/c10d/init.cpp` 查看此测试函数的定义。
        dist._test_python_store(MyPythonStore())


class RendezvousTest(TestCase):
    def test_unknown_handler(self):
        # 使用断言确保在运行时捕获到 RuntimeError，并且错误信息以 "No rendezvous handler" 开头
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            dist.rendezvous("invalid://")

    def test_url_with_node_params(self):
        # 使用断言确保在运行时捕获到 AssertionError，并且错误信息包含 "has node-specific arguments"
        with self.assertRaisesRegex(AssertionError, "has node-specific arguments"):
            dist.rendezvous("file://foo?rank=12&world_size=16", 12, 16)


class RendezvousEnvTest(TestCase):
    @retry_on_connect_failures
    def test_nominal(self):
        # 设置环境变量来模拟单一节点环境
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())

        # 设置环境变量中的 RANK 为 "0"，创建并使用 dist.rendezvous("env://") 生成器
        # 从生成器中获取第一个结果（store0, rank0, size0），并进行断言验证
        os.environ["RANK"] = "0"
        gen0 = dist.rendezvous("env://")
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(1, size0)

        # 在 store0 中设置键值对 "key0": "value0"
        store0.set("key0", "value0")

        # 使用 get 方法检查键 "key0" 对应的值是否为 b"value0"
        self.assertEqual(b"value0", store0.get("key0"))


class RendezvousFileTest(TestCase):
    def test_common_errors(self):
        # 使用断言确保在运行时捕获到 ValueError，并且错误信息包含 "path missing"
        with self.assertRaisesRegex(ValueError, "path missing"):
            gen = dist.rendezvous("file://?rank=0&world_size=1")
            next(gen)
        # 使用断言确保在运行时捕获到 ValueError，并且错误信息包含 "rank parameter missing"
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            gen = dist.rendezvous("file:///tmp/foo?world_size=1")
            next(gen)
        # 使用断言确保在运行时捕获到 ValueError，并且错误信息包含 "size parameter missing"
        with self.assertRaisesRegex(ValueError, "size parameter missing"):
            gen = dist.rendezvous("file:///tmp/foo?rank=0")
            next(gen)

    def test_nominal(self):
        # 创建临时文件，并构建 URL，其中包含文件名和 world_size 参数
        with tempfile.NamedTemporaryFile(delete=False) as file:
            url = f'file:///{file.name.replace(os.path.sep, "/")}?world_size=2'
            # 创建并使用 dist.rendezvous 生成器，从生成器中获取第一个结果
            # （store0, rank0, size0），并进行断言验证
            gen0 = dist.rendezvous(url + "&rank=0")
            store0, rank0, size0 = next(gen0)
            self.assertEqual(0, rank0)
            self.assertEqual(2, size0)
            # 创建并使用 dist.rendezvous 生成器，从生成器中获取第二个结果
            # （store1, rank1, size1），并进行断言验证
            gen1 = dist.rendezvous(url + "&rank=1")
            store1, rank1, size1 = next(gen1)
            self.assertEqual(1, rank1)
            self.assertEqual(2, size1)

            # 在 store0 和 store1 中分别设置键值对
            store0.set("key0", "value0")
            store1.set("key1", "value1")

            # 在 store1 中检查键 "key0" 对应的值是否为 b"value0"
            self.assertEqual(b"value0", store1.get("key0"))
            # 在 store0 中检查键 "key1" 对应的值是否为 b"value1"
            self.assertEqual(b"value1", store0.get("key1"))


@skip_if_win32()
class RendezvousTCPTest(TestCase):
    pass
    # 创建一个 TCP URL 地址，默认使用主机名 DEFAULT_HOSTNAME
    def create_tcp_url(self):
        # 查找一个空闲的端口号
        port = common.find_free_port()
        # 构建格式化的 TCP URL 字符串，包括地址、端口和世界大小
        url = "tcp://%s:%d?world_size=%d" % (addr, port, 1)
        return url

    # 测试常见错误情况
    def test_common_errors(self):
        # 测试缺少端口号参数时是否会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "port number missing"):
            gen = dist.rendezvous("tcp://127.0.0.1?rank=0&world_size=1")
            next(gen)
        # 测试缺少 rank 参数时是否会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            gen = dist.rendezvous("tcp://127.0.0.1:23456?world_size=1")
            next(gen)
        # 测试缺少 size 参数时是否会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "size parameter missing"):
            gen = dist.rendezvous("tcp://127.0.0.1:23456?rank=0")
            next(gen)

    # 测试 DNS 超时错误
    def test_dns_timeout(self):
        with self.assertRaisesRegex(
            DistNetworkError, "client socket has timed out after.*dnsnotexist"
        ) as manager:
            # 使用不存在的域名进行测试，设置超时时间为1秒
            gen = dist.rendezvous(
                "tcp://dnsnotexist:23456?world_size=2&rank=0",
                timeout=timedelta(seconds=1),
            )
            next(gen)
        self.assertTrue(isinstance(manager.exception, DistError))

    # 测试正常情况
    @retry_on_connect_failures
    def test_nominal(self):
        # 创建一个 TCP URL 地址
        url = self.create_tcp_url()
        # 开始连接并获取第一个 store 的信息
        gen0 = dist.rendezvous(url + "&rank=0")
        store0, rank0, size0 = next(gen0)
        # 验证返回的 rank 和 size 是否符合预期
        self.assertEqual(0, rank0)
        self.assertEqual(1, size0)

        # 在单个 store 上设置一个键值对
        store0.set("key0", "value0")

        # 使用 get 方法验证设置的值是否正确
        self.assertEqual(b"value0", store0.get("key0"))

    # 测试 TCP 存储超时设置
    @retry_on_connect_failures(connect_errors=(CONNECT_TIMEOUT, ADDRESS_IN_USE))
    def test_tcp_store_timeout_set(self):
        # 创建一个 TCP URL 地址
        url = self.create_tcp_url()
        # 设置测试的存储超时时间为10秒
        test_store_timeout = timedelta(seconds=10)
        # 开始连接并获取第一个 store 的信息，同时设置超时时间
        gen0 = dist.rendezvous(url + "&rank=0", timeout=test_store_timeout)
        store0, rank0, size0 = next(gen0)
        # 使用 get 方法尝试获取不存在的键，预期会抛出 RuntimeError 异常并包含 "Timeout" 字符串
        with self.assertRaisesRegex(RuntimeError, "Timeout"):
            store0.get("nonexistant key")

        # 计算实际执行时间与预期超时时间的差异
        end = time.time()
        time_diff = end - start
        # 验证实际执行时间是否大于预期的超时时间
        self.assertGreater(test_store_timeout.seconds * 10, time_diff)
    # 测试 TCP 存储超时是否会中断客户端操作
    def test_tcp_store_timeout_doest_break_client(self):
        # 创建一个 TCP URL
        url = self.create_tcp_url()
        # 设置测试存储超时时间为10秒
        test_store_timeout = timedelta(seconds=10)
        # 生成一个迭代器，用于建立与指定 URL 的连接，并设置超时时间
        gen0 = dist.rendezvous(url + "&rank=0", timeout=test_store_timeout)
        # 从迭代器中获取存储对象、排名和大小
        store0, rank0, size0 = next(gen0)
        
        # 这里应该在10秒内超时。如果传入 rendezvous 的超时时间没有被尊重，
        # 它会花费更长时间才能超时。
        start = time.time()
        # 断言抛出 RuntimeError 异常，并且异常信息包含 "Timeout"
        with self.assertRaisesRegex(RuntimeError, "Timeout"):
            store0.get("the_key")

        # 设置键值对 "the_key": "x" 到存储对象中
        store0.set("the_key", "x")

        # 断言从存储对象中获取键 "the_key" 的值为 b"x"
        self.assertEqual(b"x", store0.get("the_key"))

        # 计算测试执行结束时间
        end = time.time()
        # 计算测试执行时间差
        time_diff = end - start
        # 断言测试执行时间大于等于测试存储超时时间的10倍
        self.assertGreater(test_store_timeout.seconds * 10, time_diff)

    # 测试带有 libuv 的 TCP 存储 URL
    def test_tcp_store_url_with_libuv(self):
        # 创建一个 TCP URL
        url = self.create_tcp_url()
        # 生成一个迭代器，用于建立与指定 URL 的连接，并启用 libuv 后端
        gen0 = dist.rendezvous(url + "&rank=0&use_libuv=1")
        # 从迭代器中获取存储对象、排名和大小
        store0, rank0, size0 = next(gen0)
        # 断言存储对象的 libuv 后端为真
        self.assertTrue(store0.libuvBackend)
class DummyStore(dist.Store):
    # 定义一个虚拟的存储类，继承自 dist.Store
    def __init__(self):
        # 初始化实例变量，用于记录操作历史的列表
        self.appends = []
        self.multi_sets = []
        self.multi_gets = []
        self.multi_get_res = []
        # 调用父类的初始化方法
        super().__init__()

    # 添加键值对到 appends 列表
    def append(self, key, value):
        self.appends.append((key, value))

    # 记录调用 multi_get 方法时传入的 keys
    # 并返回事先设定好的结果列表中的第一个结果
    def multi_get(self, keys):
        self.multi_gets.append(keys)
        return self.multi_get_res.pop(0)

    # 记录调用 multi_set 方法时传入的 keys 和 values
    def multi_set(self, keys, values):
        self.multi_sets.append((keys, values))

    # 虚拟的方法，返回 True，表示支持扩展的 API
    def has_extended_api(self):
        return True


class TestPythonStore(TestCase):
    # 测试未实现必要方法时是否会抛出异常
    def test_optional_methods_fail(self):
        # 定义一个未实现必要方法的 TestStore 类
        class TestStore(dist.Store):
            pass

        # 创建 TestStore 的实例
        store = TestStore()
        # 断言 has_extended_api 方法返回 False
        self.assertFalse(store.has_extended_api())
        # 测试调用 append 方法时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.append("foo", "bar")
        # 测试调用 multi_get 方法时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_get(["foo", "bar"])
        # 测试调用 multi_set 方法时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_set(["foo", "bar"], [b"v", b"v"])

    # 测试通过 PrefixStore 封装后，未实现必要方法是否会抛出异常
    def test_has_extended_api_passthrough(self):
        # 定义一个未实现必要方法的 TestStore 类
        class TestStore(dist.Store):
            pass

        # 创建 TestStore 的实例
        test_store = TestStore()
        # 使用 PrefixStore 封装 TestStore
        store = dist.PrefixStore("p", test_store)
        # 断言 has_extended_api 方法返回 False
        self.assertFalse(store.has_extended_api())
        # 测试调用 append 方法时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.append("foo", "bar")
        # 测试调用 multi_get 方法时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_get(["foo", "bar"])
        # 测试调用 multi_set 方法时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_set(["foo", "bar"], [b"v", b"v"])

    # 测试 DummyStore 实现了扩展 API 的情况
    def test_has_extended_api_roundtrip(self):
        # 创建 DummyStore 的实例
        store = DummyStore()
        # 使用 PrefixStore 封装 DummyStore
        prefix = dist.PrefixStore("p", store)
        # 断言 has_extended_api 方法返回 True
        self.assertTrue(prefix.has_extended_api())

    # 测试 append 方法的正确性
    def test_append_roundtrip(self):
        # 创建 DummyStore 的实例
        store = DummyStore()
        # 使用 PrefixStore 封装 DummyStore
        prefix = dist.PrefixStore("p", store)
        # 调用 append 方法
        prefix.append("foo", "bar")
        # 断言 appends 列表中有一项，并且其内容正确
        self.assertEqual(1, len(store.appends))
        self.assertEqual(("p/foo", b"bar"), store.appends[0])

    # 测试 multi_get 方法的正确性
    def test_multi_get_roundtrip(self):
        # 创建 DummyStore 的实例
        store = DummyStore()
        # 使用 PrefixStore 封装 DummyStore
        prefix = dist.PrefixStore("p", store)
        # 设置 multi_get 方法的返回结果
        store.multi_get_res.append([b"x", b"y"])
        # 调用 multi_get 方法
        res = prefix.multi_get(["foo", "bar"])
        # 断言 multi_gets 列表中有一项，并且其内容正确
        self.assertEqual(1, len(store.multi_gets))
        self.assertEqual(["p/foo", "p/bar"], store.multi_gets[0])
        # 断言 multi_get 方法返回的结果正确
        self.assertEqual([b"x", b"y"], res)

    # 测试 multi_set 方法的正确性
    def test_multi_set_roundtrip(self):
        # 创建 DummyStore 的实例
        store = DummyStore()
        # 使用 PrefixStore 封装 DummyStore
        prefix = dist.PrefixStore("p", store)
        # 调用 multi_set 方法
        prefix.multi_set(["foo", "bar"], [b"x", b"y"])
        # 断言 multi_sets 列表中有一项，并且其内容正确
        self.assertEqual(1, len(store.multi_sets))
        self.assertEqual(["p/foo", "p/bar"], store.multi_sets[0][0])
        self.assertEqual([b"x", b"y"], store.multi_sets[0][1])
    # 定义一个测试方法，测试扩展方法的回退机制
    def test_extended_methods_fallbacks(self):
        # 创建一个自定义的 Python 存储对象
        test_store = MyPythonStore()
        # 创建一个带有前缀的存储对象，基于 test_store
        store = dist.PrefixStore("p", test_store)
        # 断言当前存储对象不支持扩展 API
        self.assertFalse(store.has_extended_api())
        # 在存储对象中追加键 "foo" 对应的值为 b"po"
        store.append("foo", b"po")
        # 再次在存储对象中追加键 "foo" 对应的值为 b"tato"
        store.append("foo", b"tato")
        # 断言获取键 "foo" 对应的值等于 b"potato"
        self.assertEqual(store.get("foo"), b"potato")

        # 使用 multi_set 方法批量设置键 "a", "b" 对应的值为 b"c", b"d"
        store.multi_set(["a", "b"], [b"c", b"d"])
        # 断言获取键 "a", "b", "foo" 对应的值分别为 b"c", b"d", b"potato"
        self.assertEqual(store.multi_get(["a", "b", "foo"]), [b"c", b"d", b"potato"])
# 定义一个测试类 `TestMultiThreadedWait`，继承自 `MultiThreadedTestCase` 类
class TestMultiThreadedWait(MultiThreadedTestCase):
    # 创建一个文件存储实例 `file_store`，使用临时命名的文件名，并设置存储容量为 1
    file_store = dist.FileStore(tempfile.NamedTemporaryFile(delete=False).name, 1)
    # 创建一个哈希存储实例 `hash_store`
    hash_store = dist.HashStore()

    # 创建一个 TCP 存储实例 `tcp_store`，禁用 libuv
    tcp_store = create_tcp_store(use_libuv=False)
    # 创建一个 TCP 存储实例 `tcp_store_uv`，启用 libuv
    tcp_store_uv = create_tcp_store(use_libuv=True)

    # 定义一个属性 `world_size`，返回 2，表示测试中的分布式环境中的节点数
    @property
    def world_size(self):
        return 2

    # 设置测试的准备工作，在每个测试方法执行前调用
    def setUp(self):
        super().setUp()
        # 启动多线程
        self._spawn_threads()

    # 定义一个私有方法 `_test_wait`，接收一个存储实例 `store` 作为参数
    def _test_wait(self, store):
        # 设置存储实例的超时时间为 2 秒钟
        store.set_timeout(timedelta(seconds=2))
        # 如果当前进程的分布式节点标识为 0
        if dist.get_rank() == 0:
            # 等待键名为 "key1" 的数据存储到达，然后进行断言检查
            store.wait(["key1"])
            self.assertEqual(b"value1", store.get("key1"))
        # 如果当前进程的分布式节点标识为 1
        if dist.get_rank() == 1:
            # 设置键名为 "key1" 的值为 "value1"
            store.set("key1", "value1")

    # 定义测试方法 `test_wait_hash_store`，测试哈希存储 `hash_store` 的等待功能
    def test_wait_hash_store(self):
        self._test_wait(self.hash_store)

    # 定义测试方法 `test_wait_file_store`，测试文件存储 `file_store` 的等待功能
    def test_wait_file_store(self):
        self._test_wait(self.file_store)

    # 定义测试方法 `test_wait_prefix_file_store`，测试带前缀的文件存储的等待功能
    def test_wait_prefix_file_store(self):
        # 创建一个带有前缀的文件存储 `store`，前缀为 "pre"，基于 `file_store`
        store = dist.PrefixStore("pre", self.file_store)
        self._test_wait(store)

    # 定义私有方法 `_test_wait_tcp_store`，接收一个主存储 `master_store` 作为参数
    def _test_wait_tcp_store(self, master_store):
        # 根据当前节点标识选择性地创建 TCP 存储 `store`
        store = (
            master_store
            if dist.get_rank() == 0
            else dist.TCPStore(
                host_name=master_store.host,
                port=master_store.port,
                is_master=False,
                wait_for_workers=False,
                use_libuv=False,
            )
        )
        # 使用 `_test_wait` 方法测试当前 TCP 存储 `store`
        self._test_wait(store)

        # 创建一个带有前缀的 TCP 存储 `prefix_store`，前缀为 "pre"，基于 `store`
        prefix_store = dist.PrefixStore("pre", store)
        self._test_wait(prefix_store)

    # 定义测试方法 `test_wait_tcp_store`，测试非 libuv 版本的 TCP 存储 `tcp_store` 的等待功能
    def test_wait_tcp_store(self):
        self._test_wait_tcp_store(self.tcp_store)

    # 定义测试方法 `test_wait_tcp_store_uv`，测试 libuv 版本的 TCP 存储 `tcp_store_uv` 的等待功能
    def test_wait_tcp_store_uv(self):
        self._test_wait_tcp_store(self.tcp_store_uv)


# 实例化参数化测试类 `TestMultiThreadedWait`
instantiate_parametrized_tests(TestMultiThreadedWait)

# 跳过在 Windows 平台上运行的测试类 `TimeoutTest`
@skip_if_win32()
class TimeoutTest(TestCase):
    # 在测试结束时，忽略 SIGUSR1 信号
    def tearDown(self):
        import signal

        super().tearDown()
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    # 定义一个测试函数，验证中断信号不会打断等待操作
    def test_interrupt_doesnt_break_wait(self):
        # 导入信号处理模块
        import signal

        # 存储各个进程的结果
        rank_res = [None, None]

        # 定义运行函数，接收进程号和存储对象作为参数
        def run(rank, my_store):
            nonlocal rank_res
            try:
                # 如果是第一个进程
                if rank == 0:
                    # 等待4秒钟
                    time.sleep(4)
                    # 设置键值对到存储对象中
                    my_store.set("foo", "bar")
                else:
                    # 等待其他进程设置的键"foo"出现，最多等待10秒钟
                    my_store.wait(["foo"], datetime.timedelta(seconds=10))
                # 标记该进程已完成
                rank_res[rank] = True
            except Error as e:  # noqa: F821
                # 如果出现异常，记录异常信息
                rank_res[rank] = e
            # 等待1秒钟
            time.sleep(1)

        # 创建两个 TCP 存储对象，模拟两个进程间的通信
        rank0_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME,
            port=0,
            world_size=2,
            is_master=True,
            wait_for_workers=False,
        )
        rank1_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME,
            port=rank0_store.port,
            world_size=2,
            is_master=False,
            wait_for_workers=False,
        )

        # 创建线程列表
        ths = []
        for i in range(2):
            # 为每个进程创建线程
            t = threading.Thread(
                target=run,
                args=(
                    i,
                    [rank0_store, rank1_store][i],
                ),
            )
            # 启动线程
            t.start()
            # 将线程对象加入列表
            ths.append(t)

        # 定义信号处理函数
        def handler(a, b):
            pass

        # 设置 SIGUSR1 信号的处理函数
        signal.signal(signal.SIGUSR1, handler)
        # 等待1秒钟
        time.sleep(1)
        # 向第二个线程发送 SIGUSR1 信号
        signal.pthread_kill(ths[1].ident, signal.SIGUSR1)

        # 等待所有线程结束
        for t in ths:
            t.join()
        
        # 断言第一个进程和第二个进程都已完成
        self.assertTrue(rank_res[0], "rank0")
        self.assertTrue(rank_res[1], "rank1")
class InitPgWithNonUvStore(TestCase):
    """
    This test case demonstrates the usage of the legacy TCPStore (non-libuv) backend 
    when libuv is the default backend.
    """

    def tearDown(self):
        super().tearDown()
        # 清除环境变量中与连接和端口相关的设定
        os.environ.pop("USE_LIBUV", None)
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)

    def test_with_url_param(self):
        # 找到一个可用的端口号
        port = common.find_free_port()
        # 使用指定的 URL 参数初始化进程组，指定使用非 libuv 后端
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=1,
            init_method=f"tcp://{DEFAULT_HOSTNAME}:{port}?use_libuv=0",
        )
        self._run_test()

    def test_with_env_var(self):
        # 找到一个可用的端口号
        port = common.find_free_port()
        # 设置环境变量使用非 libuv 后端，并指定主节点地址和端口号
        os.environ["USE_LIBUV"] = "0"
        os.environ["MASTER_ADDR"] = DEFAULT_HOSTNAME
        os.environ["MASTER_PORT"] = str(port)
        # 使用环境变量中的设定初始化进程组
        dist.init_process_group("gloo", rank=0, world_size=1, init_method="env://")
        self._run_test()

    def _run_test(self):
        # 获取当前进程组
        pg = dist.group.WORLD
        # 获取进程组的存储对象
        store = c10d._get_process_group_store(pg)
        # 断言存储对象是 PrefixStore 类型
        self.assertTrue(isinstance(store, dist.PrefixStore))
        # c10d 进行多层封装
        while isinstance(store, dist.PrefixStore):
            store = store.underlying_store
        # 断言存储对象是 TCPStore 类型
        self.assertTrue(isinstance(store, dist.TCPStore))
        # 断言存储对象不是 libuv 后端
        self.assertFalse(store.libuvBackend)
        # 销毁进程组
        dist.destroy_process_group()


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
```