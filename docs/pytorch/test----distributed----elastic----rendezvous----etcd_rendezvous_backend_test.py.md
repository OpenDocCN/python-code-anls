# `.\pytorch\test\distributed\elastic\rendezvous\etcd_rendezvous_backend_test.py`

```py
# 导入需要的模块和类
import subprocess
from base64 import b64encode
from typing import cast, ClassVar
from unittest import TestCase

# 导入特定的异常类
from etcd import EtcdKeyNotFound  # type: ignore[import]

# 导入测试用的混合类
from rendezvous_backend_test import RendezvousBackendTestMixin

# 导入需要测试的模块和类
from torch.distributed.elastic.rendezvous import (
    RendezvousConnectionError,
    RendezvousParameters,
)
from torch.distributed.elastic.rendezvous.etcd_rendezvous_backend import (
    create_backend,
    EtcdRendezvousBackend,
)
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.rendezvous.etcd_store import EtcdStore


class EtcdRendezvousBackendTest(TestCase, RendezvousBackendTestMixin):
    _server: ClassVar[EtcdServer]

    @classmethod
    def setUpClass(cls) -> None:
        # 设置类变量 _server 为 EtcdServer 实例，并启动服务
        cls._server = EtcdServer()
        cls._server.start(stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        # 停止 _server 实例服务
        cls._server.stop()

    def setUp(self) -> None:
        # 设置测试方法的实例变量 _client 为 _server 的客户端实例

        # 尝试删除 "/dummy_prefix" 及其所有子目录，保证测试环境干净
        try:
            self._client.delete("/dummy_prefix", recursive=True, dir=True)
        except EtcdKeyNotFound:
            pass

        # 创建 EtcdRendezvousBackend 的实例，使用 _client、"dummy_run_id"、"/dummy_prefix" 作为参数
        self._backend = EtcdRendezvousBackend(
            self._client, "dummy_run_id", "/dummy_prefix"
        )

    def _corrupt_state(self) -> None:
        # 在 etcd 中写入非 base64 编码的数据，用于模拟状态损坏
        self._client.write("/dummy_prefix/dummy_run_id", "non_base64")


class CreateBackendTest(TestCase):
    _server: ClassVar[EtcdServer]

    @classmethod
    def setUpClass(cls) -> None:
        # 设置类变量 _server 为 EtcdServer 实例，并启动服务
        cls._server = EtcdServer()
        cls._server.start(stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        # 停止 _server 实例服务
        cls._server.stop()

    def setUp(self) -> None:
        # 设置测试方法的实例变量 _params 为 RendezvousParameters 的实例
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint=self._server.get_endpoint(),
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            protocol="hTTp",  # 指定通信协议为 HTTP（不规范的大小写）
            read_timeout="10",  # 设置读取超时时间为 10 秒
        )

        # 设置预期的读取超时时间为 10
        self._expected_read_timeout = 10
    # 测试函数：测试创建后端和存储对象是否返回正确
    def test_create_backend_returns_backend(self) -> None:
        # 调用创建后端和存储对象的函数
        backend, store = create_backend(self._params)

        # 断言：检查返回的后端名称是否为 "etcd-v2"
        self.assertEqual(backend.name, "etcd-v2")

        # 断言：检查返回的存储对象是否为 EtcdStore 类型的实例
        self.assertIsInstance(store, EtcdStore)

        # 强制类型转换：将 store 转换为 EtcdStore 类型
        etcd_store = cast(EtcdStore, store)

        # 断言：检查 etcd_store 的客户端读取超时时间是否与预期相等
        self.assertEqual(etcd_store.client.read_timeout, self._expected_read_timeout)  # type: ignore[attr-defined]

        # 获取服务器端的客户端对象
        client = self._server.get_client()

        # 设置后端状态为 b"dummy_state"
        backend.set_state(b"dummy_state")

        # 从客户端获取指定路径的结果数据
        result = client.get("/torch/elastic/rendezvous/" + self._params.run_id)

        # 断言：检查获取的结果值是否为 base64 编码后的 "dummy_state"
        self.assertEqual(result.value, b64encode(b"dummy_state").decode())

        # 断言：检查获取的 TTL（生存时间）是否不超过 7200 秒
        self.assertLessEqual(result.ttl, 7200)

        # 在存储对象中设置键 "dummy_key" 的值为 "dummy_value"
        store.set("dummy_key", "dummy_value")

        # 从客户端获取指定路径的结果数据
        result = client.get("/torch/elastic/store/" + b64encode(b"dummy_key").decode())

        # 断言：检查获取的结果值是否为 base64 编码后的 "dummy_value"
        self.assertEqual(result.value, b64encode(b"dummy_value").decode())

    # 测试函数：如果协议未指定，则测试创建后端和存储对象是否返回正确
    def test_create_backend_returns_backend_if_protocol_is_not_specified(self) -> None:
        # 删除参数配置中的 "protocol" 键值对
        del self._params.config["protocol"]

        # 调用相同的测试函数以测试未指定协议时的行为
        self.test_create_backend_returns_backend()

    # 测试函数：如果读取超时时间未指定，则测试创建后端和存储对象是否返回正确
    def test_create_backend_returns_backend_if_read_timeout_is_not_specified(
        self,
    ) -> None:
        # 删除参数配置中的 "read_timeout" 键值对
        del self._params.config["read_timeout"]

        # 设置预期的读取超时时间为 60 秒
        self._expected_read_timeout = 60

        # 调用相同的测试函数以测试未指定读取超时时间时的行为
        self.test_create_backend_returns_backend()

    # 测试函数：如果 etcd 服务器不可达，则测试是否会引发连接错误
    def test_create_backend_raises_error_if_etcd_is_unreachable(self) -> None:
        # 修改参数中的端点地址为 "dummy:1234"
        self._params.endpoint = "dummy:1234"

        # 使用断言检查是否会引发 RendezvousConnectionError 异常
        with self.assertRaisesRegex(
            RendezvousConnectionError,
            r"^The connection to etcd has failed. See inner exception for details.$",
        ):
            # 调用创建后端和存储对象的函数
            create_backend(self._params)

    # 测试函数：如果协议无效，则测试是否会引发值错误
    def test_create_backend_raises_error_if_protocol_is_invalid(self) -> None:
        # 将参数配置中的 "protocol" 设置为 "dummy"
        self._params.config["protocol"] = "dummy"

        # 使用断言检查是否会引发 ValueError 异常
        with self.assertRaisesRegex(
            ValueError, r"^The protocol must be HTTP or HTTPS.$"
        ):
            # 调用创建后端和存储对象的函数
            create_backend(self._params)

    # 测试函数：如果读取超时时间无效，则测试是否会引发值错误
    def test_create_backend_raises_error_if_read_timeout_is_invalid(self) -> None:
        # 遍历无效的读取超时时间列表
        for read_timeout in ["0", "-10"]:
            with self.subTest(read_timeout=read_timeout):
                # 设置参数配置中的 "read_timeout" 为当前的无效读取超时时间
                self._params.config["read_timeout"] = read_timeout

                # 使用断言检查是否会引发 ValueError 异常
                with self.assertRaisesRegex(
                    ValueError, r"^The read timeout must be a positive integer.$"
                ):
                    # 调用创建后端和存储对象的函数
                    create_backend(self._params)
```