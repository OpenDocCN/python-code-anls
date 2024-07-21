# `.\pytorch\test\distributed\elastic\rendezvous\dynamic_rendezvous_test.py`

```
# 导入必要的库和模块
import copy  # 导入 copy 模块，用于对象的深拷贝和浅拷贝操作
import os  # 导入 os 模块，提供与操作系统相关的功能
import pickle  # 导入 pickle 模块，用于序列化和反序列化 Python 对象
import socket  # 导入 socket 模块，提供网络通信的功能
import threading  # 导入 threading 模块，用于多线程编程
import time  # 导入 time 模块，提供时间相关的功能

from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 和 abstractmethod 装饰器
from base64 import b64encode  # 从 base64 模块导入 b64encode 函数，用于编码和解码 Base64 数据
from datetime import datetime, timedelta  # 从 datetime 模块导入 datetime 和 timedelta 类
from typing import Callable, cast, Optional, Tuple  # 导入类型提示相关的功能
from unittest import TestCase  # 从 unittest 模块导入 TestCase 类
from unittest.mock import call, MagicMock, Mock, patch, PropertyMock  # 导入 mock 相关的功能

import torch.distributed as dist  # 导入 PyTorch 分布式模块

from torch.distributed import HashStore, Store  # 导入 HashStore 和 Store 类
from torch.distributed.elastic.rendezvous import (  # 导入弹性训练相关的 rendezvous 模块
    RendezvousClosedError,  # 导入 RendezvousClosedError 异常类
    RendezvousError,  # 导入 RendezvousError 异常类
    RendezvousInfo,  # 导入 RendezvousInfo 类
    RendezvousParameters,  # 导入 RendezvousParameters 类
    RendezvousStateError,  # 导入 RendezvousStateError 异常类
    RendezvousStoreInfo,  # 导入 RendezvousStoreInfo 类
    RendezvousTimeoutError,  # 导入 RendezvousTimeoutError 异常类
)
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (  # 导入动态 rendezvous 相关的模块和类
    _Action,  # 导入 _Action 枚举类
    _BackendRendezvousStateHolder,  # 导入 _BackendRendezvousStateHolder 类
    _DistributedRendezvousOpExecutor,  # 导入 _DistributedRendezvousOpExecutor 类
    _NodeDesc,  # 导入 _NodeDesc 类
    _NodeDescGenerator,  # 导入 _NodeDescGenerator 类
    _RendezvousCloseOp,  # 导入 _RendezvousCloseOp 类
    _RendezvousContext,  # 导入 _RendezvousContext 类
    _RendezvousExitOp,  # 导入 _RendezvousExitOp 类
    _RendezvousJoinOp,  # 导入 _RendezvousJoinOp 类
    _RendezvousKeepAliveOp,  # 导入 _RendezvousKeepAliveOp 类
    _RendezvousState,  # 导入 _RendezvousState 类
    _RendezvousStateHolder,  # 导入 _RendezvousStateHolder 类
    create_handler,  # 导入 create_handler 函数
    DynamicRendezvousHandler,  # 导入 DynamicRendezvousHandler 类
    RendezvousBackend,  # 导入 RendezvousBackend 类
    RendezvousSettings,  # 导入 RendezvousSettings 类
    RendezvousTimeout,  # 导入 RendezvousTimeout 类
    Token,  # 导入 Token 类
)


class CustomAssertMixin:
    assertDictEqual: Callable

    def assert_state_equal(
        self, actual: _RendezvousState, expected: _RendezvousState
    ) -> None:
        self.assertDictEqual(vars(actual), vars(expected))

    def assert_state_empty(self, actual: _RendezvousState) -> None:
        self.assertDictEqual(vars(actual), vars(_RendezvousState()))


class RendezvousTimeoutTest(TestCase):
    def test_init_initializes_timeout(self) -> None:
        # 创建 RendezvousTimeout 对象，并初始化各个超时时间
        timeout = RendezvousTimeout(
            timedelta(seconds=50),  # 设置加入超时时间为 50 秒
            timedelta(seconds=60),  # 设置最后调用超时时间为 60 秒
            timedelta(seconds=70),  # 设置关闭超时时间为 70 秒
            timedelta(seconds=80),  # 设置心跳超时时间为 80 秒
        )

        # 断言各个超时时间的初始化值是否符合预期
        self.assertEqual(timeout.join, timedelta(seconds=50))
        self.assertEqual(timeout.last_call, timedelta(seconds=60))
        self.assertEqual(timeout.close, timedelta(seconds=70))
        self.assertEqual(timeout.heartbeat, timedelta(seconds=80))

    def test_init_initializes_timeout_if_no_timeout_is_specified(self) -> None:
        # 创建 RendezvousTimeout 对象，若未指定超时时间，则使用默认值
        timeout = RendezvousTimeout()

        # 断言各个超时时间的初始化默认值是否符合预期
        self.assertEqual(timeout.join, timedelta(seconds=600))  # 默认加入超时时间为 600 秒
        self.assertEqual(timeout.last_call, timedelta(seconds=30))  # 默认最后调用超时时间为 30 秒
        self.assertEqual(timeout.close, timedelta(seconds=30))  # 默认关闭超时时间为 30 秒
        self.assertEqual(timeout.heartbeat, timedelta(seconds=5))  # 默认心跳超时时间为 5 秒
    def test_init_raises_error_if_timeout_is_not_positive(self) -> None:
        # 定义一个列表，包含两个 timedelta 对象，分别表示0秒和-1秒的时间间隔
        join_timeouts = [timedelta(seconds=0), timedelta(seconds=-1)]

        # 对于每一个 join_timeout，在测试中执行以下操作
        for join_timeout in join_timeouts:
            # 使用 self.subTest() 创建一个子测试上下文，传入参数 join_timeout
            with self.subTest(join_timeout=join_timeout):
                # 使用 self.assertRaisesRegex() 断言引发 ValueError 异常，并匹配指定的正则表达式消息
                # 消息格式要求 join_timeout 的值必须是正数
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The join timeout \({join_timeout}\) must be positive.$",
                ):
                    # 创建 RendezvousTimeout 对象，传入 join_timeout 作为参数
                    timeout = RendezvousTimeout(join_timeout)
class NodeDescTest(TestCase):
    # NodeDesc 类的单元测试
    def test_repr(self) -> None:
        # 创建一个 _NodeDesc 实例，使用 "dummy_fqdn", 3, 5 作为参数
        desc = _NodeDesc("dummy_fqdn", 3, 5)

        # 断言 _NodeDesc 实例的 repr 方法返回字符串 "dummy_fqdn_3_5"
        self.assertEqual(repr(desc), "dummy_fqdn_3_5")

    # NodeDesc 类的另一个单元测试
    def test_hash(self) -> None:
        # 创建两个不同的 _NodeDesc 实例，分别使用参数 "dummy_fqdn", 2, 4 和 "dummy_fqdn", 3, 5
        desc1 = _NodeDesc("dummy_fqdn", 2, 4)
        desc2 = _NodeDesc("dummy_fqdn", 3, 5)

        # 将两个 _NodeDesc 实例放入集合 descs 中
        descs = {desc1, desc2}

        # 断言 desc1 和 desc2 分别存在于集合 descs 中
        self.assertIn(desc1, descs)
        self.assertIn(desc2, descs)


class NodeDescGeneratorTest(TestCase):
    # NodeDescGenerator 类的单元测试
    def test_generate(self) -> None:
        # 创建一个 _NodeDescGenerator 实例
        desc_generator = _NodeDescGenerator()

        # 获取本地主机的完全限定域名
        fqdn = socket.getfqdn()

        # 获取当前进程的进程 ID
        pid = os.getpid()

        # 迭代生成四次，每次生成一个新的 _NodeDesc 实例
        for local_id in range(4):
            # 使用子测试进行测试，子测试的描述信息包括 fqdn、pid 和 local_id
            with self.subTest(fqdn=fqdn, pid=pid, local_id=local_id):
                # 调用 desc_generator 的 generate 方法生成一个新的 _NodeDesc 实例
                desc = desc_generator.generate()

                # 断言 _NodeDesc 实例的 repr 方法返回符合预期的字符串
                self.assertEqual(repr(desc), f"{fqdn}_{pid}_{local_id}")


class RendezvousStateTest(TestCase):
    # RendezvousState 类的单元测试
    def test_encoded_size_is_within_expected_limit(self) -> None:
        # 创建一个 _RendezvousState 实例
        state = _RendezvousState()

        # 设置 _RendezvousState 实例的一些属性
        state.round = 1
        state.complete = True
        state.deadline = datetime.utcnow()
        state.closed = True

        # 定义期望的最大尺寸列表，每个元组包含一个整数和一个字节大小的上限
        # fmt: off
        expected_max_sizes = (
            (   5,    2 * (2 ** 10),),  #    10 machines <=   2KB  # noqa: E201, E241, E262
            (  50,   16 * (2 ** 10),),  #   100 machines <=  16KB  # noqa: E201, E241, E262
            ( 500,  160 * (2 ** 10),),  #  1000 machines <= 160KB  # noqa: E201, E241, E262
            (5000, 1600 * (2 ** 10),),  # 10000 machines <= 1.6MB  # noqa: E201, E241, E262
        )
        # fmt: on

        # 遍历期望的最大尺寸列表
        for num_nodes, max_byte_size in expected_max_sizes:
            # 使用子测试进行测试，子测试的描述信息包括 num_nodes 和 max_byte_size
            with self.subTest(num_nodes=num_nodes, max_byte_size=max_byte_size):
                # 对每个 num_nodes 的值循环生成 _NodeDesc 实例，并进行相关状态设置
                for i in range(num_nodes):
                    # 创建两个不同的 _NodeDesc 实例，每个实例使用不同的域名和端口号
                    node_running = _NodeDesc(
                        f"dummy{i}.dummy1-dummy1-dummy1-dummy1.com", 12345, i
                    )
                    node_waiting = _NodeDesc(
                        f"dummy{i}.dummy2-dummy2-dummy2-dummy2.com", 67890, i
                    )

                    # 将 node_running 添加到 participants 字典中，键为 _NodeDesc 实例，值为 i
                    state.participants[node_running] = i

                    # 将 node_waiting 添加到 wait_list 集合中
                    state.wait_list.add(node_waiting)

                    # 设置 node_running 和 node_waiting 的最后心跳时间为当前时间
                    state.last_heartbeats[node_running] = datetime.utcnow()
                    state.last_heartbeats[node_waiting] = datetime.utcnow()

                # 使用 pickle.dumps 方法将 state 对象序列化为字节流
                bits = pickle.dumps(state)

                # 使用 base64 编码将序列化后的字节流转换为 base64 编码的字节
                base64_bits = b64encode(bits)

                # 断言 base64 编码后的长度不超过指定的 max_byte_size
                self.assertLessEqual(len(base64_bits), max_byte_size)


class FakeRendezvousBackend(RendezvousBackend):
    # FakeRendezvousBackend 类，继承自 RendezvousBackend 类
    _state: Optional[bytes]
    _token: int

    # 构造方法，初始化 _state 和 _token 属性
    def __init__(self) -> None:
        self._state = None
        self._token = 0

    # name 属性的 getter 方法，返回字符串 "fake_backend"
    @property
    def name(self) -> str:
        return "fake_backend"

    # 获取当前状态和令牌的方法，返回元组 (self._state, self._token)
    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        if self._token == 0:
            return None

        return self._state, self._token  # type: ignore[return-value]

    # 设置状态的方法，接受 state 和 token 两个参数
    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        # 如果 token 参数为 None，则将其设置为 0
        if token is None:
            token = 0

        # 如果 token 等于当前对象的 _token 属性值
        if token == self._token:
            # 将对象的 _state 属性设置为给定的 state 参数值
            self._state = state
            # 将对象的 _token 属性增加 1
            self._token += 1

            # 设置标志位表示状态已设置成功
            has_set = True
        else:
            # 如果 token 不等于当前对象的 _token 属性值，表示状态设置失败
            has_set = False

        # 返回当前对象的 _state, _token, has_set 属性值作为元组，忽略类型检查
        return self._state, self._token, has_set  # type: ignore[return-value]

    def get_state_internal(self) -> _RendezvousState:
        # 返回从对象的 _state 属性中反序列化得到的状态对象
        return pickle.loads(cast(bytes, self._state))

    def set_state_internal(self, state: _RendezvousState) -> None:
        # 将给定的 state 参数序列化为字节流，并设置为对象的 _state 属性值
        self._state = pickle.dumps(state)
        # 增加对象的 _token 属性值，用于跟踪状态更改
        self._token += 1

    def corrupt_state(self) -> None:
        # 将对象的 _state 属性设置为表示损坏状态的字节流
        self._state = b"corrupt_state"
        # 增加对象的 _token 属性值，用于跟踪状态更改
        self._token += 1
class BackendRendezvousStateHolderTest(TestCase, CustomAssertMixin):
    # 设置测试用例的初始化方法
    def setUp(self) -> None:
        # 创建一个 FakeRendezvousBackend 的实例作为被测对象的模拟
        self._backend = FakeRendezvousBackend()

        # 创建 Mock 对象来包装 _backend 的 get_state 和 set_state 方法
        mock_get_state = MagicMock(wraps=self._backend.get_state)
        mock_set_state = MagicMock(wraps=self._backend.set_state)

        # 创建一个 Mock 对象来模拟后端的行为
        self._mock_backend = Mock()
        self._mock_backend.get_state = mock_get_state
        self._mock_backend.set_state = mock_set_state

        # 使用 setattr 将 Mock 对象设置为 _backend 的 get_state 和 set_state 方法
        setattr(self._backend, "get_state", mock_get_state)  # noqa: B010
        setattr(self._backend, "set_state", mock_set_state)  # noqa: B010

        # 设置用于测试的 RendezvousSettings 对象
        self._settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            timeout=RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=30),
            keep_alive_max_attempt=3,
        )

        # 设置缓存持续时间
        self._cache_duration = 0

        # 设置当前时间为 2000 年 1 月 1 日 0 时 0 分
        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        # 使用 patch 创建一个 Mock 对象来模拟 datetime 模块中的 utcnow 方法，并设置返回值为 self._now
        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )
        mock_datetime = self._datetime_patch.start()
        mock_datetime.utcnow.return_value = self._now

    # 设置测试用例的清理方法
    def tearDown(self) -> None:
        # 停止对 datetime 模块的 patch
        self._datetime_patch.stop()

    # 创建 _RendezvousState 对象的辅助方法
    def _create_state(self) -> _RendezvousState:
        state = _RendezvousState()
        state.round = 999
        state.complete = True
        state.deadline = self._now
        state.closed = True
        state.participants = {
            _NodeDesc("dummy1", 1, 1): 0,
            _NodeDesc("dummy2", 1, 1): 1,
            _NodeDesc("dummy3", 1, 1): 2,
        }
        state.wait_list = {
            _NodeDesc("dummy4", 1, 1),
            _NodeDesc("dummy5", 1, 1),
        }
        state.last_heartbeats = {
            _NodeDesc("dummy1", 1, 1): self._now,
            _NodeDesc("dummy2", 1, 1): self._now - timedelta(seconds=15),
            _NodeDesc("dummy3", 1, 1): self._now - timedelta(seconds=30),
            _NodeDesc("dummy4", 1, 1): self._now - timedelta(seconds=60),
            _NodeDesc("dummy5", 1, 1): self._now - timedelta(seconds=90),
        }
        return state

    # 创建 _BackendRendezvousStateHolder 对象的辅助方法
    def _create_state_holder(self) -> _BackendRendezvousStateHolder:
        return _BackendRendezvousStateHolder(
            self._backend, self._settings, self._cache_duration
        )

    # 测试用例：验证初始化方法是否正确初始化了状态持有器对象
    def test_init_initializes_state_holder(self) -> None:
        state_holder = self._create_state_holder()

        # 断言状态持有器的初始状态为空
        self.assert_state_empty(state_holder.state)

        # 断言 Mock 对象的方法未被调用
        self._mock_backend.assert_not_called()

    # 测试用例：如果后端状态不存在，则同步方法应返回空状态
    def test_sync_gets_empty_state_if_backend_state_does_not_exist(self) -> None:
        state_holder = self._create_state_holder()

        # 调用同步方法并获取返回结果
        has_set = state_holder.sync()

        # 断言返回结果为 None
        self.assertIsNone(has_set)

        # 断言状态持有器的状态为空
        self.assert_state_empty(state_holder.state)

        # 断言 Mock 对象的 get_state 方法被调用一次，set_state 方法未被调用
        self.assertEqual(self._mock_backend.get_state.call_count, 1)
        self.assertEqual(self._mock_backend.set_state.call_count, 0)
    # 测试同步操作，如果本地状态干净，则获取后端状态
    def test_sync_gets_backend_state_if_local_state_is_clean(self) -> None:
        # 创建状态持有器对象
        state_holder = self._create_state_holder()

        # 创建预期状态对象
        expected_state = self._create_state()

        # 进行三次尝试
        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                # 设置预期状态对象的轮次属性为当前尝试次数
                expected_state.round = attempt

                # 将预期状态对象设置到后端
                self._backend.set_state_internal(expected_state)

                # 执行同步操作，返回值应为 None
                has_set = state_holder.sync()

                # 断言同步后状态持有器的状态与预期状态一致
                self.assertIsNone(has_set)

                # 断言获取后端状态的调用次数为1
                self.assertEqual(self._mock_backend.get_state.call_count, 1)
                # 断言设置后端状态的调用次数为0
                self.assertEqual(self._mock_backend.set_state.call_count, 0)

                # 重置模拟后端的调用记录
                self._mock_backend.reset_mock()

    # 测试同步操作，如果本地状态过时且脏，则获取后端状态
    def test_sync_gets_backend_state_if_local_state_is_old_and_dirty(self) -> None:
        # 创建状态持有器对象
        state_holder = self._create_state_holder()

        # 创建预期状态对象
        expected_state = self._create_state()

        # 进行三次尝试
        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                # 将预期状态对象设置到后端（增量令牌）
                self._backend.set_state_internal(expected_state)

                # 将状态持有器对象的轮次属性设置为当前尝试次数
                state_holder.state.round = attempt
                # 标记状态持有器对象为脏状态
                state_holder.mark_dirty()

                # 执行同步操作，返回值应为 False，表示未设置到后端
                has_set = state_holder.sync()

                # 断言同步后状态持有器的状态与预期状态一致
                self.assertFalse(has_set)

                # 断言获取后端状态的调用次数为0
                self.assertEqual(self._mock_backend.get_state.call_count, 0)
                # 断言设置后端状态的调用次数为1
                self.assertEqual(self._mock_backend.set_state.call_count, 1)

                # 重置模拟后端的调用记录
                self._mock_backend.reset_mock()

    # 测试同步操作，如果本地状态是新的且脏，则设置后端状态
    def test_sync_sets_backend_state_if_local_state_is_new_and_dirty(self) -> None:
        # 创建状态持有器对象
        state_holder = self._create_state_holder()

        # 进行三次尝试
        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                # 将状态持有器对象的轮次属性设置为当前尝试次数
                state_holder.state.round = attempt
                # 标记状态持有器对象为脏状态
                state_holder.mark_dirty()

                # 执行同步操作，返回值应为 True，表示已将本地状态设置到后端
                has_set = state_holder.sync()

                # 断言同步后状态持有器的状态与从后端获取的预期状态一致
                expected_state = self._backend.get_state_internal()
                self.assert_state_equal(state_holder.state, expected_state)

                # 断言获取后端状态的调用次数为0
                self.assertEqual(self._mock_backend.get_state.call_count, 0)
                # 断言设置后端状态的调用次数为1
                self.assertEqual(self._mock_backend.set_state.call_count, 1)

                # 重置模拟后端的调用记录
                self._mock_backend.reset_mock()
    # 测试函数：当缓存持续时间被指定时，同步使用缓存状态
    def test_sync_uses_cached_state_if_cache_duration_is_specified(self) -> None:
        # 创建状态对象
        state = self._create_state()

        # 将状态设置到后端
        self._backend.set_state_internal(state)

        # 使用 patch 方法模拟时间
        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        ) as mock_time:
            # 对于不同的缓存持续时间进行循环测试
            for cache_duration in [1, 5, 10]:
                with self.subTest(cache_duration=cache_duration):
                    # 设置当前缓存持续时间
                    self._cache_duration = cache_duration

                    # 创建状态持有者对象
                    state_holder = self._create_state_holder()

                    # 模拟时间函数返回值为 5
                    mock_time.monotonic.return_value = 5

                    # 执行同步操作
                    state_holder.sync()

                    # 再次执行同步操作，检查是否设置了新状态
                    has_set = state_holder.sync()

                    # 应该返回空，表示没有设置新状态
                    self.assertIsNone(has_set)

                    # 检查获取状态的调用次数应为 1
                    self.assertEqual(self._mock_backend.get_state.call_count, 1)
                    # 检查设置状态的调用次数应为 0
                    self.assertEqual(self._mock_backend.set_state.call_count, 0)

                    # 模拟时间函数返回值增加缓存持续时间后的值
                    mock_time.monotonic.return_value = 5 + self._cache_duration

                    # 执行同步操作
                    state_holder.sync()

                    # 再次执行同步操作，检查是否设置了新状态
                    has_set = state_holder.sync()

                    # 应该返回空，表示没有设置新状态
                    self.assertIsNone(has_set)

                    # 检查获取状态的调用次数应为 1，因为缓存尚未过期
                    self.assertEqual(self._mock_backend.get_state.call_count, 1)
                    # 检查设置状态的调用次数应为 0，因为缓存尚未过期

        # 重置获取状态的模拟调用次数
        self._mock_backend.get_state.reset_mock()

    # 测试函数：如果缓存状态已过期，则同步获取后端状态
    def test_sync_gets_backend_state_if_cached_state_has_expired(self) -> None:
        # 创建状态对象
        state = self._create_state()

        # 将状态设置到后端
        self._backend.set_state_internal(state)

        # 使用 patch 方法模拟时间
        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        ) as mock_time:
            # 设置当前缓存持续时间为 1
            self._cache_duration = 1

            # 创建状态持有者对象
            state_holder = self._create_state_holder()

            # 模拟时间函数返回值为 5
            mock_time.monotonic.return_value = 5

            # 执行同步操作
            state_holder.sync()

            # 再次执行同步操作，检查是否设置了新状态
            has_set = state_holder.sync()

            # 应该返回空，表示没有设置新状态
            self.assertIsNone(has_set)

            # 检查获取状态的调用次数应为 1
            self.assertEqual(self._mock_backend.get_state.call_count, 1)
            # 检查设置状态的调用次数应为 0
            self.assertEqual(self._mock_backend.set_state.call_count, 0)

            # 模拟时间函数返回值增加缓存持续时间后的值再稍微增加一点
            mock_time.monotonic.return_value = 5 + self._cache_duration + 0.01

            # 执行同步操作
            state_holder.sync()

            # 再次执行同步操作，检查是否设置了新状态
            has_set = state_holder.sync()

            # 应该返回空，表示没有设置新状态
            self.assertIsNone(has_set)

            # 检查获取状态的调用次数应为 2，因为缓存已经过期
            self.assertEqual(self._mock_backend.get_state.call_count, 2)
            # 检查设置状态的调用次数应为 0，因为缓存已经过期
    # 测试函数：确保同步功能可以正确地清理状态
    
    def test_sync_sanitizes_state(self) -> None:
        # 创建初始状态
        state = self._create_state()
    
        # 深拷贝初始状态，作为预期状态的参考
        expected_state = copy.deepcopy(state)
    
        # 创建几个虚拟的死节点
        dead_node1 = _NodeDesc("dead1", 1, 1)
        dead_node2 = _NodeDesc("dead2", 1, 1)
        dead_node3 = _NodeDesc("dead3", 1, 1)
        dead_node4 = _NodeDesc("dead4", 1, 1)
        dead_node5 = _NodeDesc("dead5", 1, 1)
    
        # 更新几个死节点的最后心跳时间
        state.last_heartbeats[dead_node1] = self._now - timedelta(seconds=91)
        state.last_heartbeats[dead_node2] = self._now - timedelta(seconds=100)
        state.last_heartbeats[dead_node3] = self._now - timedelta(seconds=110)
        state.last_heartbeats[dead_node4] = self._now - timedelta(seconds=120)
        state.last_heartbeats[dead_node5] = self._now - timedelta(seconds=130)
    
        # 标记几个死节点为非参与者
        state.participants[dead_node1] = 0
        state.participants[dead_node2] = 0
        state.participants[dead_node3] = 0
    
        # 将几个死节点添加到等待列表中
        state.wait_list.add(dead_node4)
        state.wait_list.add(dead_node5)
    
        # 将更新后的状态存储到后端
        self._backend.set_state_internal(state)
    
        # 创建状态持有器
        state_holder = self._create_state_holder()
    
        # 执行同步操作
        state_holder.sync()
    
        # 断言同步后的状态与预期的状态相等
        self.assert_state_equal(state_holder.state, expected_state)
    
    
    # 测试函数：如果没有剩余参与者，则确保同步功能可以正确地清理状态
    
    def test_sync_sanitizes_state_if_no_participants_is_left(self) -> None:
        # 创建初始状态
        state = self._create_state()
    
        # 深拷贝初始状态，作为预期状态的参考
        expected_state = copy.deepcopy(state)
    
        # 更新所有节点的最后心跳时间
        for node in state.last_heartbeats:
            state.last_heartbeats[node] = self._now - timedelta(seconds=100)
    
        # 更新预期状态的一些字段
        expected_state.complete = False
        expected_state.round = 1000
        expected_state.participants = {}
        expected_state.wait_list = set()
        expected_state.last_heartbeats = {}
    
        # 将更新后的状态存储到后端
        self._backend.set_state_internal(state)
    
        # 创建状态持有器
        state_holder = self._create_state_holder()
    
        # 执行同步操作
        state_holder.sync()
    
        # 断言同步后的状态与预期的状态相等
        self.assert_state_equal(state_holder.state, expected_state)
    
    
    # 测试函数：如果后端状态损坏，则确保同步功能会引发错误
    
    def test_sync_raises_error_if_backend_state_is_corrupt(self) -> None:
        # 模拟后端状态损坏
        self._backend.corrupt_state()
    
        # 创建状态持有器
        state_holder = self._create_state_holder()
    
        # 断言调用同步操作会引发特定异常
        with self.assertRaisesRegex(
            RendezvousStateError,
            r"^The rendezvous state is corrupt. See inner exception for details.$",
        ):
            state_holder.sync()
class FakeRendezvousStateHolder(_RendezvousStateHolder):
    _state: _RendezvousState  # 定义一个私有属性 _state，类型为 _RendezvousState
    _dirty: Optional[bool]  # 定义一个私有属性 _dirty，可选布尔类型

    def __init__(self) -> None:
        self._state = _RendezvousState()  # 初始化 _state 属性为 _RendezvousState 类的实例
        self._dirty = None  # 初始化 _dirty 属性为 None

    @property
    def state(self) -> _RendezvousState:
        return self._state  # 返回 _state 属性的值作为 state 属性的 getter 方法

    @state.setter
    def state(self, value) -> None:
        self._state = value  # 设置 _state 属性的值作为 state 属性的 setter 方法

    def sync(self) -> Optional[bool]:
        self._dirty, dirty = None, self._dirty  # 将 _dirty 属性设为 None，同时获取原先的 _dirty 值

        return dirty  # 返回原先的 _dirty 值

    def mark_dirty(self) -> None:
        self._dirty = True  # 将 _dirty 属性设置为 True，表示状态已变更


class DistributedRendezvousOpExecutorTest(TestCase, CustomAssertMixin):
    def setUp(self) -> None:
        self._node = _NodeDesc("this_node", 1, 1)  # 创建一个 _NodeDesc 实例

        self._state_holder = FakeRendezvousStateHolder()  # 创建 FakeRendezvousStateHolder 的实例

        mock_sync = MagicMock(wraps=self._state_holder.sync)  # 创建 sync 方法的 Mock 对象
        mock_mark = MagicMock(wraps=self._state_holder.mark_dirty)  # 创建 mark_dirty 方法的 Mock 对象

        self._mock_state_holder = Mock()  # 创建 Mock 对象
        self._mock_state_holder.sync = mock_sync  # 将 sync 方法设置为 mock_sync 的包装
        self._mock_state_holder.mark = mock_mark  # 将 mark 方法设置为 mock_mark 的包装

        setattr(self._state_holder, "sync", mock_sync)  # 动态设置 _state_holder 的 sync 方法
        setattr(self._state_holder, "mark_dirty", mock_mark)  # 动态设置 _state_holder 的 mark_dirty 方法

        self._state = self._state_holder.state  # 获取 _state_holder 的 state 属性的值

        self._min_nodes = 1  # 设置最小节点数为 1
        self._max_nodes = 1  # 设置最大节点数为 1

        self._timeout = RendezvousTimeout()  # 创建 RendezvousTimeout 的实例

        self._now = datetime(2000, 1, 1, hour=0, minute=0)  # 创建特定日期时间的 datetime 实例

        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )  # 创建 datetime 的 patch 对象

        mock_datetime = self._datetime_patch.start()  # 启动 datetime 的 patch
        mock_datetime.utcnow.return_value = self._now  # 设置 mock_datetime 的 utcnow 方法返回固定时间

    def tearDown(self) -> None:
        self._datetime_patch.stop()  # 停止 datetime 的 patch

    def _create_settings(self) -> RendezvousSettings:
        return RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=self._timeout,
            keep_alive_interval=timedelta(seconds=30),
            keep_alive_max_attempt=3,
        )  # 创建并返回 RendezvousSettings 的实例

    def _create_op_executor(
        self, settings: Optional[RendezvousSettings] = None
    ) -> _DistributedRendezvousOpExecutor:
        self._state_holder.state = self._state  # 设置 _state_holder 的 state 属性为 self._state

        if settings is None:
            settings = self._create_settings()  # 如果 settings 为 None，则创建默认设置

        return _DistributedRendezvousOpExecutor(
            self._node, self._state_holder, settings
        )  # 返回 _DistributedRendezvousOpExecutor 的实例

    def _run_action(self, action: _Action) -> None:
        op_executor = self._create_op_executor()  # 创建操作执行器实例

        op = MagicMock(side_effect=[action, _Action.FINISH])  # 创建操作的 MagicMock 对象

        op_executor.run(op, deadline=1)  # 运行操作执行器的 run 方法，设置截止时间为 1

    def _assert_action(self, action: _Action, expected_state: _RendezvousState) -> None:
        self._run_action(action)  # 执行操作

        self.assert_state_equal(self._state, expected_state)  # 断言当前状态与预期状态相等

        self.assertListEqual(
            self._mock_state_holder.mock_calls, [call.sync(), call.mark(), call.sync()]
        )  # 断言 mock_state_holder 的 mock_calls 是否符合预期
    # 测试确保运行操作时传递正确的上下文和截止时间给状态处理器
    def test_run_passes_expected_context_and_deadline_to_state_handler(self) -> None:
        # 创建测试所需的设置
        settings = self._create_settings()

        # 创建操作执行器
        op_executor = self._create_op_executor(settings)

        # 创建一个 Mock 对象代表操作，并使其返回 _Action.FINISH
        op = MagicMock(return_value=_Action.FINISH)

        # 运行操作执行器，并设置截止时间为 3
        op_executor.run(op, deadline=3)

        # 获取操作调用时的第一个参数（上下文）和第二个参数（截止时间）
        ctx, deadline = op.call_args[0]  # args

        # 断言上下文的节点是预期的节点
        self.assertIs(ctx.node, self._node)
        # 断言上下文的状态是预期的状态
        self.assertIs(ctx.state, self._state)
        # 断言上下文的设置是预期的设置
        self.assertIs(ctx.settings, settings)

        # 断言截止时间是预期的截止时间
        self.assertEqual(deadline, 3)

    # 测试确保运行操作时保持存活状态
    def test_run_keeps_alive(self) -> None:
        # 创建预期的状态对象
        expected_state = _RendezvousState()

        # 设置预期状态的最后心跳时间为当前时间
        expected_state.last_heartbeats[self._node] = self._now

        # 断言操作为 KEEP_ALIVE 并验证状态
        self._assert_action(_Action.KEEP_ALIVE, expected_state)

    # 测试确保运行操作时将节点添加到参与者列表中
    def test_run_adds_to_participants(self) -> None:
        # 创建预期的状态对象
        expected_state = _RendezvousState()

        # 将节点添加到参与者列表中，并设置初始计数为 0
        expected_state.participants[self._node] = 0

        # 设置预期状态的最后心跳时间为当前时间
        expected_state.last_heartbeats[self._node] = self._now

        # 设置最小和最大节点数为 2
        self._min_nodes = 2
        self._max_nodes = 2

        # 断言操作为 ADD_TO_PARTICIPANTS 并验证状态
        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

    # 测试确保运行操作时，如果节点已在等待列表中，则将其添加到参与者列表中
    def test_run_adds_to_participants_if_node_was_in_waitlist(self) -> None:
        # 将节点添加到状态的等待列表中
        self._state.wait_list.add(self._node)

        # 创建预期的状态对象
        expected_state = _RendezvousState()

        # 将节点添加到参与者列表中，并设置初始计数为 0
        expected_state.participants[self._node] = 0

        # 设置预期状态的最后心跳时间为当前时间
        expected_state.last_heartbeats[self._node] = self._now

        # 设置最小和最大节点数为 2
        self._min_nodes = 2
        self._max_nodes = 2

        # 断言操作为 ADD_TO_PARTICIPANTS 并验证状态
        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

    # 将指定数量的参与者添加到状态中
    def _add_participants(
        self, num_participants: int, state: _RendezvousState, ranked: bool = False
    ) -> None:
        # 循环添加指定数量的参与者
        for i in range(num_participants):
            if ranked:
                # 如果按排名添加，则创建具有特定名称和排名的节点描述对象
                node = _NodeDesc(f"dummy{i}", 1, 1)
                rank = i
            else:
                # 否则，以相反顺序添加，创建具有特定名称和默认排名的节点描述对象
                node = _NodeDesc(
                    f"dummy{num_participants - i - 1}", 1, 1
                )  # Add in reverse.
                rank = 0

            # 将节点添加到状态的参与者列表中，并设置其排名
            state.participants[node] = rank

            # 设置节点的最后心跳时间为当前时间
            state.last_heartbeats[node] = self._now

    # 测试确保在达到最小节点数时将节点添加到参与者列表中，并开始最后调用
    def test_run_adds_to_participants_and_starts_last_call_if_min_nodes_is_reached(
        self,
    ) -> None:
        # 循环测试不同数量的参与者
        for num_participants in range(3):
            # 创建空的会合状态对象
            self._state = _RendezvousState()

            # 向状态中添加指定数量的参与者
            self._add_participants(num_participants, self._state)

            # 将节点添加到状态的等待列表中
            self._state.wait_list.add(self._node)

            # 创建预期的状态对象
            expected_state = _RendezvousState()

            # 向预期状态中添加相同数量的参与者
            self._add_participants(num_participants, expected_state)

            # 将节点添加到预期状态的参与者列表中，并设置初始计数为 0
            expected_state.participants[self._node] = 0

            # 设置预期状态的最后心跳时间为当前时间
            expected_state.last_heartbeats[self._node] = self._now

            # 设置预期状态的截止时间为当前时间加上最后调用的超时时间
            expected_state.deadline = self._now + self._timeout.last_call

            # 在子测试中执行以下操作
            with self.subTest(num_participants=num_participants):
                # 设置最小节点数为当前参与者数加 1
                self._min_nodes = num_participants + 1
                # 设置最大节点数为当前参与者数加 2
                self._max_nodes = num_participants + 2

                # 断言操作为 ADD_TO_PARTICIPANTS 并验证状态
                self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

                # 重置 mock 对象的调用记录
                self._mock_state_holder.reset_mock()
    # 测试函数：当达到最大节点数时，验证添加参与者并完成会面
    def test_run_adds_to_participants_and_completes_rendezvous_if_max_nodes_is_reached(
        self,
    ) -> None:
        # 遍历两种情况：最小和最大节点数是否相等
        for min_max_nodes_equal in [False, True]:
            # 遍历不同数量的参与者
            for num_participants in range(3):
                # 排名即为参与者的编号
                rank = num_participants

                # 初始化测试状态
                self._state = _RendezvousState()

                # 添加参与者到状态中
                self._add_participants(num_participants, self._state)

                # 将当前节点添加到等待列表中
                self._state.wait_list.add(self._node)

                # 设置截止时间为当前时间加上最后一次调用的超时时间
                self._state.deadline = self._now + self._timeout.last_call

                # 预期的状态对象
                expected_state = _RendezvousState()

                # 添加参与者到预期状态中，按排名添加
                self._add_participants(num_participants, expected_state, ranked=True)

                # 将当前节点的排名和最后心跳时间添加到预期状态中
                expected_state.participants[self._node] = rank
                expected_state.last_heartbeats[self._node] = self._now

                # 设置预期状态为完成状态，截止时间为None
                expected_state.complete = True
                expected_state.deadline = None

                # 使用子测试检查不同参与者数量时的行为
                with self.subTest(num_participants=num_participants):
                    # 根据最小最大节点数是否相等来设置最小节点数
                    self._min_nodes = num_participants + 1 if min_max_nodes_equal else 0
                    self._max_nodes = num_participants + 1

                    # 断言执行添加到参与者的动作，检查状态是否符合预期
                    self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

                    # 重置模拟状态的调用情况
                    self._mock_state_holder.reset_mock()

    # 测试函数：验证添加到等待列表的行为
    def test_run_adds_to_waitlist(self) -> None:
        # 预期的状态对象
        expected_state = _RendezvousState()

        # 将当前节点添加到等待列表中
        expected_state.wait_list.add(self._node)

        # 将当前节点的最后心跳时间设置为当前时间
        expected_state.last_heartbeats[self._node] = self._now

        # 断言执行添加到等待列表的动作，检查状态是否符合预期
        self._assert_action(_Action.ADD_TO_WAIT_LIST, expected_state)

    # 测试函数：验证从参与者中移除的行为
    def test_run_removes_from_participants(self) -> None:
        # 遍历两种情况：会议是否完成，最后调用的截止时间是当前时间还是None
        for complete, last_call_deadline in [(False, self._now), (True, None)]:
            # 初始化测试状态
            self._state = _RendezvousState()

            # 添加两个参与者到状态中
            self._add_participants(2, self._state)

            # 将当前节点的参与者信息设置为第一个参与者，排名为0
            self._state.participants[self._node] = 0

            # 将当前节点的最后心跳时间设置为当前时间
            self._state.last_heartbeats[self._node] = self._now

            # 设置会议是否完成和最后调用的截止时间
            self._state.complete = complete
            self._state.deadline = last_call_deadline

            # 设置当前轮次为1
            self._state.round = 1

            # 预期的状态对象
            expected_state = _RendezvousState()

            # 添加两个参与者到预期状态中
            self._add_participants(2, expected_state)

            # 设置预期状态的完成状态和截止时间
            expected_state.complete = complete
            expected_state.deadline = last_call_deadline

            # 设置预期状态的轮次为1
            expected_state.round = 1

            # 使用子测试检查不同完成状态下的行为
            with self.subTest(complete=complete):
                # 断言执行从参与者中移除的动作，检查状态是否符合预期
                self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

                # 重置模拟状态的调用情况
                self._mock_state_holder.reset_mock()

    # 测试函数：验证从参与者中移除并进入下一轮次的行为，如果节点是最后一个参与者
    def test_run_removes_from_participants_and_moves_to_next_round_if_node_is_last_participant(
        self,
    ) -> None:
        # 将当前节点的参与者信息设置为第一个参与者，排名为0
        self._state.participants[self._node] = 0

        # 将当前节点的最后心跳时间设置为当前时间
        self._state.last_heartbeats[self._node] = self._now

        # 设置会议完成状态为True
        self._state.complete = True

        # 设置当前轮次为1
        self._state.round = 1

        # 预期的状态对象
        expected_state = _RendezvousState()

        # 设置预期状态的完成状态为False
        expected_state.complete = False

        # 设置预期状态的轮次为2
        expected_state.round = 2

        # 断言执行从参与者中移除的动作，检查状态是否符合预期
        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)
    # 测试如果会合点少于最小节点数时，从参与者中移除并清除最后一次调用
    def test_run_removes_from_participants_and_clears_last_call_if_rendezvous_has_less_than_min_nodes(
        self,
    ) -> None:
        # 向状态中添加两个参与者
        self._add_participants(2, self._state)
        
        # 将当前节点的参与者数量设置为0
        self._state.participants[self._node] = 0
        
        # 记录当前节点的最后心跳时间
        self._state.last_heartbeats[self._node] = self._now
        
        # 将截止时间设置为当前时间
        self._state.deadline = self._now
        
        # 创建一个期望的状态对象
        expected_state = _RendezvousState()
        
        # 向期望的状态对象中添加两个参与者
        self._add_participants(2, expected_state)
        
        # 设置最小节点数和最大节点数
        self._min_nodes = 3
        self._max_nodes = 4
        
        # 断言执行动作，预期结果是从参与者中移除
        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

    # 测试如果从等待列表中移除
    def test_run_removes_from_waitlist(self) -> None:
        # 将当前节点添加到等待列表中
        self._state.wait_list.add(self._node)
        
        # 记录当前节点的最后心跳时间
        self._state.last_heartbeats[self._node] = self._now
        
        # 创建一个期望的状态对象
        expected_state = _RendezvousState()
        
        # 断言执行动作，预期结果是从等待列表中移除
        self._assert_action(_Action.REMOVE_FROM_WAIT_LIST, expected_state)

    # 测试标记会合关闭
    def test_run_marks_rendezvous_closed(self) -> None:
        # 创建一个期望的状态对象
        expected_state = _RendezvousState()
        
        # 将会合状态设置为关闭
        expected_state.closed = True
        
        # 断言执行动作，预期结果是标记会合关闭
        self._assert_action(_Action.MARK_RENDEZVOUS_CLOSED, expected_state)

    # 测试如果会合已关闭时引发错误
    def test_run_raises_error_if_rendezvous_is_closed(self) -> None:
        # 使用断言检查是否引发了会合关闭错误
        with self.assertRaises(RendezvousClosedError):
            self._run_action(_Action.ERROR_CLOSED)
        
        # 使用断言检查模拟状态持有者的调用
        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync()])

    # 测试如果操作超时时引发错误
    def test_run_raises_error_if_operation_timed_out(self) -> None:
        # 使用断言检查是否引发了会合超时错误
        with self.assertRaises(RendezvousTimeoutError):
            self._run_action(_Action.ERROR_TIMEOUT)
        
        # 使用断言检查模拟状态持有者的调用
        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync()])

    # 测试如果请求同步则延迟执行
    def test_run_delays_execution_if_sync_requested(self) -> None:
        # 使用模拟延迟功能进行补丁
        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous._delay"
        ) as mock_delay:
            # 执行同步动作
            self._run_action(_Action.SYNC)
            
            # 断言检查是否调用了模拟延迟功能
            mock_delay.assert_called_once_with(seconds=1)
        
        # 使用断言检查模拟状态持有者的调用
        self.assertListEqual(
            self._mock_state_holder.mock_calls, [call.sync(), call.sync()]
        )
class AbstractTestRendezvousOp(ABC):
    assertEqual: Callable  # 定义一个类型提示，表明assertEqual是一个可调用对象

    def setUp(self) -> None:
        self._node = _NodeDesc("this_node", 1, 1)
        # 设置测试环境：当前节点的描述信息

        self._min_nodes = 1
        self._max_nodes = 2
        # 设置最小和最大节点数

        self._keep_alive_interval = timedelta(seconds=30)
        # 设置保持活跃的时间间隔为30秒

        self._state = _RendezvousState()
        self._state.participants[_NodeDesc("dummy1", 1, 1)] = 1
        # 初始化会议状态，并添加一个参与者

        self._now = datetime(2000, 1, 1, hour=0, minute=0)
        # 设置当前时间为2000年1月1日零点零分

        self._deadline = 10
        # 设置截止时间为10秒

        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )
        # 用于模拟日期时间相关操作的patch对象

        mock_datetime = self._datetime_patch.start()
        mock_datetime.utcnow.return_value = self._now
        # 启动日期时间的模拟，并设定模拟的当前时间为_now

        self._time_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        )
        # 用于模拟时间相关操作的patch对象

        mock_time = self._time_patch.start()
        mock_time.monotonic.return_value = self._deadline
        # 启动时间的模拟，并设定模拟的时间为_deadline

    def tearDown(self) -> None:
        self._time_patch.stop()
        self._datetime_patch.stop()
        # 停止时间和日期时间的模拟

    def _get_next_action(self) -> _Action:
        op = self._create_op()
        # 获取下一个操作对象

        settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=RendezvousTimeout(),
            keep_alive_interval=self._keep_alive_interval,
            keep_alive_max_attempt=3,
        )
        # 设置会议的参数和配置

        ctx = _RendezvousContext(self._node, self._state, settings)
        # 创建会议的上下文对象，包含节点、状态和设置信息

        return op(ctx, self._deadline)
        # 执行操作并返回操作的结果

    @abstractmethod
    def _create_op(self) -> Callable:
        pass
        # 抽象方法，用于创建具体的操作对象

    def _assert_action(self, expected_action) -> None:
        action = self._get_next_action()
        # 获取下一个操作动作

        self.assertEqual(action, expected_action)
        # 断言动作是否符合预期


class TestRendezvousExitOp(AbstractTestRendezvousOp, TestCase):
    def _create_op(self) -> Callable:
        return _RendezvousExitOp()
        # 创建_RendezvousExitOp操作对象的具体实现

    def test_removes_from_participants_if_node_is_participant(self) -> None:
        self._state.participants[self._node] = 1
        # 设置当前节点为参与者

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS)
        # 断言操作动作为从参与者中移除的动作

    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        self._deadline = 0
        # 设置截止时间为0，表示超时

        self._state.participants[self._node] = 1
        # 设置当前节点为参与者

        self._assert_action(_Action.ERROR_TIMEOUT)
        # 断言操作动作为超时错误的动作

    def test_finishes_if_node_is_not_participant(self) -> None:
        self._assert_action(_Action.FINISH)
        # 断言操作动作为完成的动作，因为当前节点不是参与者


class TestRendezvousJoinOp(AbstractTestRendezvousOp, TestCase):
    def _create_op(self) -> Callable:
        return _RendezvousJoinOp()
        # 创建_RendezvousJoinOp操作对象的具体实现

    def test_raises_closed_if_rendezvous_is_closed(self) -> None:
        self._state.closed = True
        # 设置会议为已关闭状态

        self._assert_action(_Action.ERROR_CLOSED)
        # 断言操作动作为会议已关闭的错误动作

    def test_finishes_if_rendezvous_is_complete_and_node_is_participant(self) -> None:
        self._state.participants[self._node] = 0
        # 设置当前节点为参与者

        self._state.complete = True
        # 设置会议为已完成状态

        self._assert_action(_Action.FINISH)
        # 断言操作动作为完成的动作
    # 检查是否等待完成会面的验证，并计算保持活动的时间
    def _assert_waits_rendezvous_completion(self) -> None:
        keep_alive_time = self._now - self._keep_alive_interval

        # 对于两种预期的动作，分别设置最后心跳时间
        for delta, expected_action in [
            (timedelta(seconds=0), _Action.KEEP_ALIVE),
            (timedelta(seconds=1), _Action.SYNC),
        ]:
            self._state.last_heartbeats[self._node] = keep_alive_time + delta

            # 断言预期的动作
            self._assert_action(expected_action)

    # 测试：如果会面已完成，则将当前节点视为冗余节点，添加到冗余列表
    def test_treat_as_redundancy_for_next_rendezvous_if_rendezvous_is_complete(
        self,
    ) -> None:
        self._max_nodes = 1

        self._state.complete = True

        # 断言动作为添加到冗余列表
        self._assert_action(_Action.ADD_TO_REDUNDANCY_LIST)

    # 测试：如果会面已完成且节点已被视为冗余，则等待下一轮会面
    def test_waits_next_round_if_rendezvous_is_complete_and_node_is_redundant(
        self,
    ) -> None:
        self._state.redundancy_list.add(self._node)

        self._max_nodes = 1

        self._state.complete = True

        # 执行等待会面完成的验证
        self._assert_waits_rendezvous_completion()

    # 测试：如果会面已完成且节点在等待列表中，则等待下一轮会面
    def test_waits_next_round_if_rendezvous_is_complete_and_node_is_in_wait_list(
        self,
    ) -> None:
        self._state.wait_list.add(self._node)

        self._state.complete = True

        # 执行等待会面完成的验证
        self._assert_waits_rendezvous_completion()

    # 测试：如果会面已完成且当前节点数小于最大节点数，则将当前节点添加到等待列表
    def test_adds_to_wait_list_if_rendezvous_is_complete_and_num_nodes_is_less_than_max_nodes(
        self,
    ) -> None:
        self._state.complete = True

        # 断言动作为添加到等待列表
        self._assert_action(_Action.ADD_TO_WAIT_LIST)

    # 测试：如果节点是会面的参与者，则等待会面完成
    def test_waits_rendezvous_to_complete_if_node_is_participant(self) -> None:
        self._max_nodes = 3

        self._state.participants[self._node] = 0

        self._state.deadline = self._now

        # 执行等待会面完成的验证
        self._assert_waits_rendezvous_completion()

    # 测试：如果节点是会面的参与者且超过了最后调用截止时间，则标记会面为完成
    def test_marks_rendezvous_complete_if_node_is_participant_and_last_call_deadline_exceeded(
        self,
    ) -> None:
        self._max_nodes = 3

        self._state.participants[self._node] = 0

        self._state.deadline = self._now - timedelta(seconds=1)

        # 断言动作为标记会面为完成
        self._assert_action(_Action.MARK_RENDEZVOUS_COMPLETE)

    # 测试：将节点添加到参与者列表
    def test_adds_to_participants(self) -> None:
        # 断言动作为添加到参与者列表
        self._assert_action(_Action.ADD_TO_PARTICIPANTS)

    # 测试：如果截止时间已过，则抛出超时错误
    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        self._deadline = 0

        # 断言动作为超时错误
        self._assert_action(_Action.ERROR_TIMEOUT)

    # 测试：如果回滚截止时间已过且节点是参与者，则抛出超时错误
    def test_raises_timeout_if_rollback_deadline_exceeded_and_node_is_participant(
        self,
    ) -> None:
        self._deadline = 0

        self._state.participants[self._node] = 0

        # 断言动作为超时错误
        self._assert_action(_Action.ERROR_TIMEOUT)

    # 测试：如果回滚截止时间已过且节点在等待列表中，则抛出超时错误
    def test_raises_timeout_if_rollback_deadline_exceeded_and_node_is_in_wait_list(
        self,
    ) -> None:
        self._deadline = 0

        self._state.wait_list.add(self._node)

        # 断言动作为超时错误
        self._assert_action(_Action.ERROR_TIMEOUT)
    # 当回滚截止时间未达到时，如果参与者已超时，则从参与者列表中移除该节点
    def test_removes_from_participants_if_timed_out_but_rollback_deadline_is_not_reached(
        self,
    ) -> None:
        # 设置测试用的截止时间为5
        self._deadline = 5
    
        # 在状态对象的参与者字典中标记节点为超时
        self._state.participants[self._node] = 0
    
        # 断言执行某个动作，期望动作为从参与者列表中移除
        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS)
    
    # 当回滚截止时间未达到时，如果在等待列表中的节点已超时，则从等待列表中移除该节点
    def test_removes_from_wait_list_if_timed_out_but_rollback_deadline_is_not_reached(
        self,
    ) -> None:
        # 设置测试用的截止时间为5
        self._deadline = 5
    
        # 将节点添加到状态对象的等待列表中
        self._state.wait_list.add(self._node)
    
        # 断言执行某个动作，期望动作为从等待列表中移除
        self._assert_action(_Action.REMOVE_FROM_WAIT_LIST)
    
    # 对于冗余节点，不会因为超时而触发任何操作
    def test_no_timeout_for_redundant_node(self) -> None:
        # 设置最大节点数为1
        self._max_nodes = 1
        # 设置截止时间为0，即没有超时限制
        self._deadline = 0
        # 设置状态为已完成
        self._state.complete = True
    
        # 将节点添加到状态对象的冗余节点列表中
        self._state.redundancy_list.add(self._node)
    
        # 断言执行某个动作，期望动作为同步操作
        self._assert_action(_Action.SYNC)
    
    # 对于冗余节点，即使超时时间已过，仍需要保持其活跃状态
    def test_keep_alive_for_redundant_node(self) -> None:
        # 设置截止时间为0，即没有超时限制
        self._deadline = 0
        # 设置最大节点数为1
        self._max_nodes = 1
        # 设置状态为已完成
        self._state.complete = True
    
        # 将节点添加到状态对象的冗余节点列表中
        self._state.redundancy_list.add(self._node)
    
        # 计算保持活跃的时间点，即当前时间减去保持活跃间隔
        keep_alive_time = self._now - self._keep_alive_interval
        # 更新状态对象中节点的最后心跳时间
        self._state.last_heartbeats[self._node] = keep_alive_time
    
        # 断言执行某个动作，期望动作为保持活跃操作
        self._assert_action(_Action.KEEP_ALIVE)
class TestRendezvousCloseOp(AbstractTestRendezvousOp, TestCase):
    # TestRendezvousCloseOp 类，继承自 AbstractTestRendezvousOp 和 TestCase
    def _create_op(self) -> Callable:
        # 返回一个 _RendezvousCloseOp 实例的回调函数
        return _RendezvousCloseOp()

    def test_finishes_if_rendezvous_is_closed(self) -> None:
        # 设置 self._state.closed 为 True，模拟会话已关闭的情况
        self._state.closed = True

        # 断言执行 _Action.FINISH 动作
        self._assert_action(_Action.FINISH)

    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        # 设置 self._deadline 为 0，模拟超过截止时间的情况
        self._deadline = 0

        # 断言执行 _Action.ERROR_TIMEOUT 动作
        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_marks_rendezvous_closed(self) -> None:
        # 断言执行 _Action.MARK_RENDEZVOUS_CLOSED 动作
        self._assert_action(_Action.MARK_RENDEZVOUS_CLOSED)


class TestRendezvousKeepAliveOp(AbstractTestRendezvousOp, TestCase):
    # TestRendezvousKeepAliveOp 类，继承自 AbstractTestRendezvousOp 和 TestCase
    def _create_op(self) -> Callable:
        # 返回一个 _RendezvousKeepAliveOp 实例的回调函数
        return _RendezvousKeepAliveOp()

    def test_updates_keep_alive_if_needed(self) -> None:
        # 计算 keep_alive_time，用于模拟最后心跳时间的设定
        keep_alive_time = self._now - self._keep_alive_interval

        # 遍历两种 delta 值，测试心跳更新的情况
        for delta in [timedelta(seconds=0), timedelta(seconds=-1)]:
            with self.subTest(delta=delta):
                # 设置 self._state.last_heartbeats[self._node]，模拟节点的最后心跳时间
                self._state.last_heartbeats[self._node] = keep_alive_time + delta

                # 断言执行 _Action.KEEP_ALIVE 动作
                self._assert_action(_Action.KEEP_ALIVE)

    def test_raises_timeout_if_deadlined_exceeded(self) -> None:
        # 设置 self._deadline 为 0，模拟超过截止时间的情况
        self._deadline = 0

        # 设置 self._state.last_heartbeats[self._node]，模拟节点的最后心跳时间
        self._state.last_heartbeats[self._node] = self._now - self._keep_alive_interval

        # 断言执行 _Action.ERROR_TIMEOUT 动作
        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_finishes_if_no_keep_alive_update_is_needed(self) -> None:
        # 设置 delta 为 timedelta(seconds=1)，模拟最后心跳时间的设定
        delta = timedelta(seconds=1)

        # 设置 self._state.last_heartbeats[self._node]，模拟节点的最后心跳时间
        self._state.last_heartbeats[self._node] = (
            self._now - self._keep_alive_interval + delta
        )

        # 断言执行 _Action.FINISH 动作
        self._assert_action(_Action.FINISH)


class DummyStore(Store):
    # DummyStore 类，继承自 Store 类，用作测试用途的虚拟存储类
    pass


class DynamicRendezvousHandlerTest(TestCase):
    # DynamicRendezvousHandlerTest 类，继承自 TestCase 类，用于测试动态会话处理器
    def setUp(self) -> None:
        # 设置测试的节点描述符 _NodeDesc
        self._node = _NodeDesc("this_node", 1, 1)

        # 设置最小和最大节点数
        self._min_nodes = 1
        self._max_nodes = 1

        # 初始化各种超时时间设定为 None
        self._join_timeout: Optional[timedelta] = None
        self._close_timeout: Optional[timedelta] = None
        self._heartbeat_timeout: Optional[timedelta] = None

        # 设置心跳间隔为 30 秒的时间段
        self._keep_alive_interval = timedelta(seconds=30)

        # 创建一个 DummyStore 的实例作为测试使用的存储
        self._store = DummyStore()

        # 设置 MagicMock 作为 get 和 set 方法的模拟
        self._mock_store_get = MagicMock(return_value=b"123")
        self._mock_store_set = MagicMock()

        # 将模拟的 get 和 set 方法绑定到 self._store 对象上
        setattr(self._store, "get", self._mock_store_get)  # noqa: B010
        setattr(self._store, "set", self._mock_store_set)  # noqa: B010

        # 创建一个 FakeRendezvousStateHolder 的实例用作状态保持
        self._state_holder = FakeRendezvousStateHolder()

        # 设置 MagicMock 作为同步方法的模拟
        self._mock_sync = MagicMock(wraps=self._state_holder.sync)

        # 将模拟的 sync 方法绑定到 self._state_holder 上
        setattr(self._state_holder, "sync", self._mock_sync)  # noqa: B010

        # 获取状态对象 self._state
        self._state = self._state_holder.state

        # 创建一个 DummyStore 的实例作为 TCP 存储的模拟
        self._tcp_store_mock = DummyStore()

        # 使用 patch.object 方法，替换 DynamicRendezvousHandler 类的 _create_tcp_store_server 方法
        # 返回 self._tcp_store_mock
        patcher = patch.object(
            DynamicRendezvousHandler,
            "_create_tcp_store_server",
            return_value=self._tcp_store_mock,
        )
        # 启动 patcher，并在测试结束后停止
        patcher.start()
        self.addCleanup(patcher.stop)
    # 创建动态会议处理器的方法，返回一个动态会议处理器对象
    def _create_handler(self) -> DynamicRendezvousHandler:
        # 定义会议设置对象，包括运行 ID、最小和最大节点数、超时设置（加入、关闭、心跳）、保持活跃间隔及最大尝试次数
        settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=RendezvousTimeout(
                join=self._join_timeout,
                close=self._close_timeout,
                heartbeat=self._heartbeat_timeout,
            ),
            keep_alive_interval=self._keep_alive_interval,
            keep_alive_max_attempt=3,
        )

        # 将当前状态设置到状态持有器中
        self._state_holder.state = self._state

        # 创建并返回动态会议处理器对象，使用节点、设置、后端名称、存储和状态持有器作为参数
        return DynamicRendezvousHandler(
            self._node, settings, "dummy_backend", self._store, self._state_holder
        )

    # 测试共享存储创建 TCP 存储
    def test_share_store_creates_tcp_store(self):
        # 创建会议处理器对象
        handler = self._create_handler()

        # 定义共享存储信息，并使用 Mock 对象模拟构建方法返回的结果
        shared_store_info = RendezvousStoreInfo("host", 54321)
        with patch.object(RendezvousStoreInfo, "build", return_value=shared_store_info):
            # 获取下一个会议信息
            rdzv_info = handler.next_rendezvous()
            # 断言引导存储信息的主地址和端口
            self.assertEqual(rdzv_info.bootstrap_store_info.master_addr, "host")
            self.assertEqual(rdzv_info.bootstrap_store_info.master_port, 54321)
        # 断言共享 TCP 存储服务器等于预期的 TCP 存储模拟对象
        self.assertEqual(handler._shared_tcp_store_server, self._tcp_store_mock)

        # 再次获取下一个会议信息
        rdzv_info = handler.next_rendezvous()
        # 断言共享 TCP 存储服务器等于预期的 TCP 存储模拟对象
        self.assertEqual(handler._shared_tcp_store_server, self._tcp_store_mock)

    # 测试共享存储在存在 TCP 存储时的行为
    def test_share_store_when_tcp_store(self):
        # 创建会议处理器对象
        handler = self._create_handler()

        # 使用 Mock 对象替换 PrefixStore 类的构造函数，模拟 TCPStore 类的行为
        with patch.object(dist, "PrefixStore", new=Mock):
            handler._store = Mock(spec=dist.TCPStore)
            type(handler._store).host = PropertyMock(return_value="host")
            type(handler._store).port = PropertyMock(return_value=54321)
            # 获取下一个会议信息
            rdzv_info = handler.next_rendezvous()
            # 断言引导存储信息的主地址和端口
            self.assertEqual(rdzv_info.bootstrap_store_info.master_addr, "host")
            self.assertEqual(rdzv_info.bootstrap_store_info.master_port, 54321)
            # 断言共享 TCP 存储服务器等于当前会议处理器的存储对象
            self.assertEqual(handler._shared_tcp_store_server, handler._store)

            # 再次获取下一个会议信息
            rdzv_info = handler.next_rendezvous()
            # 断言引导存储信息的主地址和端口
            self.assertEqual(rdzv_info.bootstrap_store_info.master_addr, "host")
            self.assertEqual(rdzv_info.bootstrap_store_info.master_port, 54321)
            # 断言共享 TCP 存储服务器等于当前会议处理器的存储对象
            self.assertEqual(handler._shared_tcp_store_server, handler._store)

    # 使用延迟模拟对象测试下一个会议方法在第一次加入尝试时的行为
    @patch("torch.distributed.elastic.rendezvous.dynamic_rendezvous._delay")
    def test_next_rendezvous_skews_the_first_join_attempt(self, mock_delay) -> None:
        # 对于每一轮测试，断言调用次数的期望结果
        for round, expected_call_count in [(0, True), (1, False)]:
            with self.subTest(round=round):
                # 设置当前状态的轮数
                self._state.round = round

                # 创建会议处理器对象
                handler = self._create_handler()

                # 获取下一个会议信息
                handler.next_rendezvous()

                # 断言延迟模拟对象的调用次数符合预期
                self.assertEqual(mock_delay.call_count, expected_call_count)

                # 重置延迟模拟对象的调用次数记录
                mock_delay.reset_mock()
    # 测试函数：验证 next_rendezvous 方法返回预期的值
    def test_next_rendezvous_returns_expected_value(self) -> None:
        # 设置两个虚拟节点参与者，并初始化它们的状态为0
        self._state.participants[_NodeDesc("dummy1", 1, 1)] = 0
        self._state.participants[_NodeDesc("dummy2", 1, 1)] = 0

        # 设置最大节点数为3
        self._max_nodes = 3

        # 创建处理器对象
        handler = self._create_handler()

        # 调用 next_rendezvous 方法，获取 rendezvous 信息
        rdzv_info = handler.next_rendezvous()

        # 断言：验证返回的 rdzv_info 中 rank 为 2
        self.assertEqual(rdzv_info.rank, 2)
        # 断言：验证返回的 rdzv_info 中 world_size 为 3
        self.assertEqual(rdzv_info.world_size, 3)

        # 使用 store 获取 dummy_key
        _ = rdzv_info.store.get("dummy_key")

        # 断言：验证 mock 的 store_get 方法是否被调用，传入特定参数
        self._mock_store_get.assert_called_with(
            "torch.rendezvous.dummy_run_id.0/dummy_key"
        )

    # 测试函数：验证 next_rendezvous 方法在请求的超时期间能够抛出 RendezvousTimeoutError
    def test_next_rendezvous_respects_the_requested_timeout(self) -> None:
        # 模拟同步操作导致的 0.3 秒延迟
        self._mock_sync.side_effect = lambda: time.sleep(0.3)

        # 设置加入超时时间为 0.2 秒
        self._join_timeout = timedelta(seconds=0.2)

        # 创建处理器对象
        handler = self._create_handler()

        # 断言：验证调用 next_rendezvous 方法是否会引发 RendezvousTimeoutError
        with self.assertRaises(RendezvousTimeoutError):
            handler.next_rendezvous()

    # 测试函数：验证多次调用 next_rendezvous 方法会使 round 逐步递增
    def test_next_rendezvous_moves_to_next_round_if_called_repeatedly(self) -> None:
        # 创建处理器对象
        handler = self._create_handler()

        # 迭代4次，每次调用 next_rendezvous 方法
        for i in range(4):
            handler.next_rendezvous()

            # 断言：验证状态对象中的 round 值与当前迭代变量 i 相等
            self.assertEqual(self._state.round, i)

    # 测试函数：验证 is_closed 方法返回预期的值
    def test_is_closed_returns_expected_value(self) -> None:
        # 针对 closed 变量的两种情况进行迭代
        for closed in [False, True]:
            with self.subTest(closed=closed):
                # 设置状态对象的 closed 属性
                self._state.closed = closed

                # 创建处理器对象
                handler = self._create_handler()

                # 断言：验证 is_closed 方法返回值与设置的 closed 变量相等
                self.assertEqual(handler.is_closed(), closed)

                # 断言：验证 mock 的 sync 方法是否被调用一次
                self._mock_sync.assert_called_once()

                # 重置 mock 对象的调用记录
                self._mock_sync.reset_mock()

    # 测试函数：验证 is_closed 方法在抛出异常时能够记录事件
    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_is_closed_records_and_raises_exceptions(self, record_mock) -> None:
        # 模拟同步操作引发的 RendezvousError 异常
        self._mock_sync.side_effect = RendezvousError("test error")

        # 创建处理器对象
        handler = self._create_handler()

        # 断言：验证调用 is_closed 方法会引发 RendezvousError 异常
        with self.assertRaises(RendezvousError):
            handler.is_closed()
            # 断言：验证 mock 的 record_rdzv_event 方法是否被调用一次
            record_mock.assert_called_once()

    # 测试函数：验证 set_closed 方法能够正确关闭 rendezvous
    def test_set_closed_closes_rendezvous(self) -> None:
        # 创建处理器对象
        handler = self._create_handler()

        # 调用 set_closed 方法
        handler.set_closed()

        # 断言：验证状态对象的 closed 属性为 True
        self.assertTrue(self._state.closed)

    # 测试函数：验证 set_closed 方法在请求的超时期间能够抛出 RendezvousTimeoutError
    def test_set_closed_respects_the_requested_timeout(self) -> None:
        # 模拟同步操作导致的 0.3 秒延迟
        self._mock_sync.side_effect = lambda: time.sleep(0.3)

        # 设置关闭超时时间为 0.2 秒
        self._close_timeout = timedelta(seconds=0.2)

        # 创建处理器对象
        handler = self._create_handler()

        # 断言：验证调用 set_closed 方法是否会引发 RendezvousTimeoutError
        with self.assertRaises(RendezvousTimeoutError):
            handler.set_closed()

    # 测试函数：验证可以多次调用 set_closed 方法
    def test_set_closed_can_be_called_multiple_times(self) -> None:
        # 创建处理器对象
        handler = self._create_handler()

        # 连续调用两次 set_closed 方法
        handler.set_closed()
        handler.set_closed()

        # 断言：验证状态对象的 closed 属性为 True
        self.assertTrue(self._state.closed)

    # 测试函数：验证 record_rdzv_event 方法被正确调用
    @patch("torch.distributed.elastic.events.record_rdzv_event")
    # 测试设置关闭记录和引发异常
    def test_set_closed_records_and_raises_exceptions(self, record_mock) -> None:
        # 使用 patch 对象替换 DynamicRendezvousHandler 类的 _close 方法
        with patch.object(DynamicRendezvousHandler, "_close") as close_mock:
            # 设置 close_mock 的副作用为引发 RendezvousError 异常
            close_mock.side_effect = RendezvousError("test error")
            # 创建测试处理程序实例
            handler = self._create_handler()
            # 断言抛出 RendezvousError 异常
            with self.assertRaises(RendezvousError):
                # 调用 set_closed 方法
                handler.set_closed()
                # 断言 record_mock.assert_called_once() 被调用一次
                record_mock.assert_called_once()

    # 测试 num_nodes_waiting 方法返回预期值
    def test_num_nodes_waiting_returns_expected_value(self) -> None:
        # 向 wait_list 中添加两个 _NodeDesc 实例
        self._state.wait_list.add(_NodeDesc("dummy1", 1, 1))
        self._state.wait_list.add(_NodeDesc("dummy2", 1, 1))

        # 创建测试处理程序实例
        handler = self._create_handler()

        # 断言 handler.num_nodes_waiting() 返回值为 2
        self.assertEqual(handler.num_nodes_waiting(), 2)

        # 断言 self._mock_sync.assert_called_once() 被调用一次
        self._mock_sync.assert_called_once()

    # 测试 num_nodes_waiting 方法记录事件并引发异常
    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_num_nodes_waiting_records_and_raises_exceptions(self, record_mock) -> None:
        # 设置 self._mock_sync 的副作用为引发 RendezvousError 异常
        self._mock_sync.side_effect = RendezvousError("test error")
        # 创建测试处理程序实例
        handler = self._create_handler()
        # 断言抛出 RendezvousError 异常
        with self.assertRaises(RendezvousError):
            # 调用 num_nodes_waiting 方法
            handler.num_nodes_waiting()
            # 断言 record_mock.assert_called_once() 被调用一次
            record_mock.assert_called_once()

    # 测试 shutdown 方法关闭会议并返回 True
    def test_shutdown_closes_rendezvous_and_returns_true(self) -> None:
        # 创建测试处理程序实例
        handler = self._create_handler()

        # 调用 shutdown 方法并获取返回结果
        result = handler.shutdown()

        # 断言结果为 True
        self.assertTrue(result)

        # 断言 self._state.closed 为 True
        self.assertTrue(self._state.closed)

    # 测试 shutdown 方法无法关闭会议时返回 False
    def test_shutdown_returns_false_if_rendezvous_cannot_be_closed(self) -> None:
        # 设置 self._mock_sync 的副作用为返回 RendezvousError 异常
        self._mock_sync.side_effect = [RendezvousError]

        # 创建测试处理程序实例
        handler = self._create_handler()

        # 调用 shutdown 方法并获取返回结果
        result = handler.shutdown()

        # 断言结果为 False
        self.assertFalse(result)

    # 测试 shutdown 方法可以多次调用
    def test_shutdown_can_be_called_multiple_times(self) -> None:
        # 创建测试处理程序实例
        handler = self._create_handler()

        # 多次调用 shutdown 方法
        handler.shutdown()
        handler.shutdown()

        # 断言 self._state.closed 为 True
        self.assertTrue(self._state.closed)

    # 测试 shutdown 方法记录事件并引发异常
    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_shutdown_records_and_raises_exceptions(self, record_mock) -> None:
        # 使用 patch 对象替换 DynamicRendezvousHandler 类的 _close 方法
        with patch.object(DynamicRendezvousHandler, "_close") as close_mock:
            # 设置 close_mock 的副作用为引发 RuntimeError 异常
            close_mock.side_effect = RuntimeError("test error")
            # 创建测试处理程序实例
            handler = self._create_handler()
            # 断言抛出 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                # 调用 shutdown 方法
                handler.shutdown()
                # 断言 record_mock.assert_called_once() 被调用一次
                record_mock.assert_called_once()

    # 测试 keep_alive 方法更新最后心跳时间
    @patch("torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime")
    def test_keep_alive_updates_last_heartbeat(self, mock_datetime) -> None:
        # 创建一个特定时间的 datetime 实例
        now = datetime(2000, 1, 1, hour=0, minute=0)

        # 设置 mock_datetime.utcnow().return_value 返回值为 now
        mock_datetime.utcnow.return_value = now

        # 设置 self._state.last_heartbeats[self._node] 的值
        self._state.last_heartbeats[self._node] = now - (self._keep_alive_interval * 2)

        # 创建测试处理程序实例
        handler = self._create_handler()

        # 调用 _keep_alive 方法
        handler._keep_alive()

        # 断言 self._state.last_heartbeats[self._node] 的值为 now
        self.assertEqual(self._state.last_heartbeats[self._node], now)
    # 确保_keep_alive_swallows_rendezvous_errors方法能够正确处理心跳异常
    def _assert_keep_alive_swallows_rendezvous_errors(self) -> None:
        # 计算最后心跳时间为当前时间减去两倍的保持活跃间隔
        last_heartbeat_time = datetime.utcnow() - (self._keep_alive_interval * 2)
    
        # 更新状态中对应节点的最后心跳时间
        self._state.last_heartbeats[self._node] = last_heartbeat_time
    
        # 创建处理器对象
        handler = self._create_handler()
    
        # 执行保持活跃操作
        handler._keep_alive()
    
        # 断言状态中对应节点的最后心跳时间与预期一致
        self.assertEqual(self._state.last_heartbeats[self._node], last_heartbeat_time)
    
    
    # 测试保持活跃操作能够正确处理RendezvousError异常
    def test_keep_alive_swallows_rendezvous_errors(self) -> None:
        # 使用mock对象模拟同步过程中发生RendezvousError异常
        self._mock_sync.side_effect = [RendezvousError]
    
        # 调用_assert_keep_alive_swallows_rendezvous_errors方法进行测试
        self._assert_keep_alive_swallows_rendezvous_errors()
    
    
    # 测试保持活跃操作能够在指定的超时时间内完成
    def test_keep_alive_respects_the_requested_timeout(self) -> None:
        # 使用lambda表达式作为mock对象的side_effect，模拟保持活跃操作耗时0.3秒
        self._mock_sync.side_effect = lambda: time.sleep(0.3)
    
        # 设置心跳超时时间为0.2秒
        self._heartbeat_timeout = timedelta(seconds=0.2)
    
        # 调用_assert_keep_alive_swallows_rendezvous_errors方法进行测试
        self._assert_keep_alive_swallows_rendezvous_errors()
    
    
    # 测试在调用next_rendezvous后启动线程，并在调用shutdown后关闭线程
    def test_keep_alive_thread_is_started_with_next_rendezvous_and_stopped_with_shutdown(
        self,
    ) -> None:
        # 设置节点描述信息
        self._node = _NodeDesc("this_node", 1, 2)
    
        # 定义线程名称
        name = "RendezvousKeepAliveTimer_2"
    
        # 创建处理器对象
        handler = self._create_handler()
    
        # 断言当前线程中不存在名称为name的线程
        self.assertTrue(all(t.name != name for t in threading.enumerate()))
    
        # 调用处理器的next_rendezvous方法
        handler.next_rendezvous()
    
        # 断言当前线程中存在名称为name的线程
        self.assertTrue(any(t.name == name for t in threading.enumerate()))
    
        # 调用处理器的shutdown方法
        handler.shutdown()
    
        # 断言当前线程中不存在名称为name的线程
        self.assertTrue(all(t.name != name for t in threading.enumerate()))
    
    
    # 测试在调用next_rendezvous后启动线程，并在处理器被删除后关闭线程
    def test_keep_alive_thread_is_started_with_next_rendezvous_and_stopped_with_finalizer(
        self,
    ) -> None:
        # 设置节点描述信息
        self._node = _NodeDesc("this_node", 1, 3)
    
        # 定义线程名称
        name = "RendezvousKeepAliveTimer_3"
    
        # 创建处理器对象
        handler = self._create_handler()
    
        # 断言当前线程中不存在名称为name的线程
        self.assertTrue(all(t.name != name for t in threading.enumerate()))
    
        # 调用处理器的next_rendezvous方法
        handler.next_rendezvous()
    
        # 断言当前线程中存在名称为name的线程
        self.assertTrue(any(t.name == name for t in threading.enumerate()))
    
        # 删除处理器对象
        del handler
    
        # 断言当前线程中不存在名称为name的线程
        self.assertTrue(all(t.name != name for t in threading.enumerate()))
#`
class DummyRendezvousBackend(RendezvousBackend):
    # 定义一个名为 DummyRendezvousBackend 的类，继承自 RendezvousBackend
    @property
    def name(self):
        # 返回一个字符串 "dummy_backend"，表示后端名称
        return "dummy_backend"

    def get_state(self):
        # 返回 None，表示获取状态的方法，当前后端没有状态
        return None

    def set_state(self, state, token):
        # 返回 None，表示设置状态的方法，当前后端不处理状态设置
        return None


class DynamicRendezvousHandlerFromBackendTest(TestCase):
    def setUp(self) -> None:
        # 初始化测试所需的各种属性
        self._run_id = "dummy_run_id"  # 设置运行 ID
        self._store = DummyStore()  # 初始化一个 DummyStore 对象
        self._backend = DummyRendezvousBackend()  # 初始化一个 DummyRendezvousBackend 对象
        self._min_nodes = 3  # 设置最小节点数
        self._max_nodes = 6  # 设置最大节点数
        self._timeout: Optional[RendezvousTimeout] = RendezvousTimeout()  # 设置超时时间，可选类型

    def _create_handler(self) -> DynamicRendezvousHandler:
        # 创建 DynamicRendezvousHandler 对象，从后端初始化
        return DynamicRendezvousHandler.from_backend(
            run_id=self._run_id,  # 传入运行 ID
            store=self._store,  # 传入存储对象
            backend=self._backend,  # 传入后端对象
            min_nodes=self._min_nodes,  # 传入最小节点数
            max_nodes=self._max_nodes,  # 传入最大节点数
            timeout=self._timeout,  # 传入超时时间
        )

    def test_init_initializes_handler(self) -> None:
        # 测试初始化处理器的方法，确保所有属性都正确设置
        handler = self._create_handler()  # 创建处理器实例

        # 检查后端名称是否正确
        self.assertEqual(handler.get_backend(), self._backend.name)
        # 检查运行 ID 是否正确
        self.assertEqual(handler.get_run_id(), self._run_id)
        # 检查设置中的运行 ID 是否正确
        self.assertEqual(handler.settings.run_id, self._run_id)
        # 检查设置中的最小节点数是否正确
        self.assertEqual(handler.settings.min_nodes, self._min_nodes)
        # 检查设置中的最大节点数是否正确
        self.assertEqual(handler.settings.max_nodes, self._max_nodes)

        # 检查超时时间是否设置正确
        if self._timeout is None:
            self.assertIsNotNone(handler.settings.timeout)  # 如果超时时间为 None，确保不为 None
        else:
            self.assertIs(handler.settings.timeout, self._timeout)  # 否则，检查超时时间是否正确

    def test_init_initializes_handler_if_timeout_is_not_specified(self) -> None:
        # 测试如果没有指定超时时间，初始化处理器的行为
        self._timeout = None

        self.test_init_initializes_handler()  # 调用前一个测试方法

    def test_init_initializes_handler_if_min_and_max_nodes_are_equal(self) -> None:
        # 测试如果最小节点数和最大节点数相等，初始化处理器的行为
        self._min_nodes = 3
        self._max_nodes = 3

        self.test_init_initializes_handler()  # 调用前一个测试方法

    def test_init_raises_error_if_min_nodes_is_not_positive(self) -> None:
        # 测试最小节点数不是正数时，初始化处理器是否会抛出错误
        for num in [0, -10]:  # 测试值包括 0 和负数
            with self.subTest(min_nodes=num):  # 为每个测试值创建子测试
                self._min_nodes = num  # 设置最小节点数

                # 检查在设置无效最小节点数时是否抛出 ValueError
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The minimum number of nodes \({num}\) must be greater than zero.$",
                ):
                    self._create_handler()  # 创建处理器

    def test_init_raises_error_if_max_nodes_is_less_than_min(self) -> None:
        # 测试最大节点数小于最小节点数时，初始化处理器是否会抛出错误
        self._min_nodes = 3
        self._max_nodes = 2

        # 检查在设置最大节点数小于最小节点数时是否抛出 ValueError
        with self.assertRaisesRegex(
            ValueError,
            rf"^The maximum number of nodes \({self._max_nodes}\) must be greater than or equal to "
            "the minimum number of nodes "
            rf"\({self._min_nodes}\).$",
        ):
            self._create_handler()  # 创建处理器


class CreateHandlerTest(TestCase):
    # 定义 CreateHandlerTest 类，继承自 TestCase，可能包含更多测试用例
    # 设置测试环境，初始化虚拟存储和虚拟后端
    def setUp(self) -> None:
        self._store = DummyStore()  # 创建虚拟存储对象

        self._backend = DummyRendezvousBackend()  # 创建虚拟后端对象

        # 设置会议参数对象，包括后端名称、端点、运行ID、最小和最大节点数以及各种超时时间
        self._params = RendezvousParameters(
            backend=self._backend.name,
            endpoint="dummy_endpoint",
            run_id="dummy_run_id",
            min_nodes=3,
            max_nodes=6,
            join_timeout="50",  # 加入超时时间为50秒
            last_call_timeout="60",  # 最后调用超时时间为60秒
            close_timeout="70",  # 关闭超时时间为70秒
        )

        # 预期的超时时间对象，基于给定的秒数创建时间段对象
        self._expected_timeout = RendezvousTimeout(
            timedelta(seconds=50), timedelta(seconds=60), timedelta(seconds=70)
        )

    # 测试函数：验证 create_handler 函数返回正确的处理器对象
    def test_create_handler_returns_handler(self) -> None:
        # 调用 create_handler 函数创建处理器对象
        handler = create_handler(self._store, self._backend, self._params)

        # 断言处理器对象的各个属性与预期相符
        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._params.run_id)
        self.assertEqual(handler.settings.min_nodes, self._params.min_nodes)
        self.assertEqual(handler.settings.max_nodes, self._params.max_nodes)
        self.assertEqual(handler.settings.timeout.join, self._expected_timeout.join)
        self.assertEqual(
            handler.settings.timeout.last_call, self._expected_timeout.last_call
        )
        self.assertEqual(handler.settings.timeout.close, self._expected_timeout.close)

    # 测试函数：验证 create_handler 函数在超时未指定时返回正确的处理器对象
    def test_create_handler_returns_handler_if_timeout_is_not_specified(self) -> None:
        # 删除参数中的超时配置项
        del self._params.config["join_timeout"]
        del self._params.config["last_call_timeout"]
        del self._params.config["close_timeout"]

        # 重新设置预期的超时时间为默认值
        self._expected_timeout = RendezvousTimeout()

        # 调用测试 create_handler 返回处理器对象的方法
        self.test_create_handler_returns_handler()

    # 测试函数：验证 create_handler 函数记录事件并引发异常
    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_create_handler_records_and_raises_exceptions(self, record_mock) -> None:
        # 使用 patch 替换 DynamicRendezvousHandler 类的 from_backend 方法
        with patch.object(DynamicRendezvousHandler, "from_backend") as from_mock:
            # 设置 from_mock 的 side_effect 为引发 RendezvousError 异常
            from_mock.side_effect = RendezvousError("test error")
            # 断言调用 create_handler 函数时引发 RendezvousError 异常，并记录事件
            with self.assertRaises(RendezvousError):
                create_handler(self._store, self._backend, self._params)
                record_mock.assert_called_once()
# 定义一个函数来忽略特定类型的异常并执行指定的函数
def _ignore_exception(exception_type: Exception, fn: Callable):
    try:
        fn()  # 调用传入的函数
    except exception_type as e:
        pass  # 忽略特定类型的异常

# 定义一个函数来等待条件的满足，超时时间默认为10秒，检查间隔默认为1秒
def _wait_for(condition, timeout=10, interval=1, name=None):
    # 定义一个内部函数来持续检查条件是否满足
    def _wait_while():
        while True:
            if condition():  # 如果条件满足
                break  # 跳出循环
            else:
                time.sleep(interval)  # 如果条件不满足，则休眠一段时间后再次检查

    # 创建一个线程，目标为内部的等待函数，设置线程名
    wait_thread = threading.Thread(target=_wait_while, name=name)
    wait_thread.start()  # 启动线程
    wait_thread.join(timeout=timeout)  # 等待线程执行完毕或超时

# 定义一个线程类，用于捕获线程执行后的结果
class _CapturingThread(threading.Thread):
    def __init__(self, target=None, name=None, args=None, kwargs=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        threading.Thread.__init__(
            self, target=target, args=args, kwargs=kwargs, name=name
        )
        self._result = None  # 初始化结果为 None

    def run(self):
        if self._target is not None:
            self._result = self._target(*self._args, **self._kwargs)  # 执行目标函数并存储结果

    def join(self, *args):
        threading.Thread.join(self, *args)  # 调用父类的 join 方法等待线程执行完毕
        return self._result  # 返回线程执行的结果

# 定义一个集成测试类
class IntegrationTest(TestCase):
    def setUp(self) -> None:
        self._store = HashStore()  # 初始化哈希存储
        self._handlers = []  # 初始化处理器列表
        self._backend = _InMemoryRendezvousBackend()  # 初始化内存会合后端

    def tearDown(self) -> None:
        for handler in self._handlers:
            handler._stop_heartbeats()  # 停止所有处理器的心跳

    # 创建一个处理器对象，并添加到处理器列表中
    def _create_handler(self, **kwargs) -> DynamicRendezvousHandler:
        params = {
            "backend": self._backend.name,
            "endpoint": "dummy_endpoint",
            "run_id": "dummy_run_id",
            "min_nodes": 2,
            "max_nodes": 2,
            "join_timeout": "5",
            "local_addr": f"address_{len(self._handlers)}",
        }
        params.update(**kwargs)  # 更新参数

        rzdv_params = RendezvousParameters(**params)  # 创建会合参数对象

        handler = create_handler(self._store, self._backend, rzdv_params)  # 创建处理器对象
        self._handlers.append(handler)  # 将处理器对象添加到处理器列表中
        return handler  # 返回创建的处理器对象

    # 测试所有节点能够加入会合
    def test_all_nodes_join_rendezvous(self) -> None:
        handler1 = self._create_handler(min_nodes=2, max_nodes=2)  # 创建第一个处理器对象
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)  # 创建第二个处理器对象

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)  # 创建捕获线程，目标为第一个处理器的下一个会合方法
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)  # 创建捕获线程，目标为第二个处理器的下一个会合方法

        handler1_thread.start()  # 启动第一个线程
        handler2_thread.start()  # 启动第二个线程

        rdzv_info1: RendezvousInfo = handler1_thread.join()  # 等待第一个线程执行完毕，并获取返回的会合信息
        rdzv_info2: RendezvousInfo = handler2_thread.join()  # 等待第二个线程执行完毕，并获取返回的会合信息

        self.assertEqual(rdzv_info1.store.underlying_store, self._store)  # 断言第一个会合信息中的存储与测试中使用的存储相同
        self.assertEqual(rdzv_info2.store.underlying_store, self._store)  # 断言第二个会合信息中的存储与测试中使用的存储相同

        self.assertNotEqual(rdzv_info1.rank, rdzv_info2.rank)  # 断言两个会合信息中的排名不相同

        self.assertEqual(rdzv_info1.world_size, 2)  # 断言第一个会合信息中的世界大小为2
        self.assertEqual(rdzv_info2.world_size, 2)  # 断言第二个会合信息中的世界大小为2
    # 测试方法：验证处理程序在冗余列表中的注册是否正常
    def test_redundancy_list(self) -> None:
        # 创建三个处理程序实例，每个实例有两个节点的设置
        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)
        handler3 = self._create_handler(min_nodes=2, max_nodes=2)

        # 分别为每个处理程序创建线程，以捕获下一个会合操作
        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)
        handler3_thread = _CapturingThread(
            target=_ignore_exception,
            args=(RendezvousTimeoutError, lambda: handler3.next_rendezvous()),
        )

        # 启动 handler1 和 handler2 的线程
        handler1_thread.start()
        handler2_thread.start()

        # 等待 handler1 和 handler2 的线程结束，确保成功进行会合
        handler1_thread.join()
        handler2_thread.join()

        # 启动 handler3 的线程，期望其注册到冗余列表中
        handler3_thread.start()

        # 等待直到 handler3 注册到冗余列表中
        _wait_for(lambda: pickle.loads(self._backend.get_state()[0]).redundancy_list)

        # 获取当前状态并验证冗余列表中的地址是否符合预期
        state_and_token = self._backend.get_state()
        state = pickle.loads(state_and_token[0])
        addresses = [node.addr for node in state.redundancy_list]
        self.assertListEqual(addresses, ["address_2"])

    # 测试方法：验证处理程序在从等待列表到会合后的过渡是否正确
    def test_redundancy_transition_to_wait_list_then_join_rendezvous(self) -> None:
        # 创建三个处理程序实例，每个实例具有不同的配置
        handler1 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )
        handler2 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
            keep_alive_interval=timedelta(seconds=1),
        )
        handler3 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )

        # 分别为每个处理程序创建线程，以捕获下一个会合操作
        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler3_thread = _CapturingThread(
            target=_ignore_exception,
            args=(RendezvousTimeoutError, lambda: handler3.next_rendezvous()),
        )

        # 启动 handler1 和 handler2 的线程
        handler1_thread.start()
        handler2_thread.start()

        # 等待 handler1 和 handler2 的线程结束，确保成功进行会合
        handler1_thread.join()
        handler2_thread.join()

        # 启动 handler3 的线程
        handler3_thread.start()

        # 等待直到 handler3 注册到冗余列表中
        _wait_for(lambda: pickle.loads(self._backend.get_state()[0]).redundancy_list)

        # 停止 handler2 的心跳
        handler2._stop_heartbeats()

        # 等待直到参与者数量为 1
        _wait_for(
            lambda: len(pickle.loads(self._backend.get_state()[0]).participants) == 1
        )

        # 等待直到等待列表中的数量为 1
        _wait_for(
            lambda: len(pickle.loads(self._backend.get_state()[0]).wait_list) == 1
        )

    # 测试方法：验证处理程序默认情况下是否启用代理存储
    def test_use_agent_store_is_true_by_default(self):
        # 创建处理程序实例，使用默认配置
        handler = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )

        # 断言处理程序的 use_agent_store 属性为 True
        self.assertTrue(handler.use_agent_store)

    # 测试方法：验证禁用 TORCH_DISABLE_SHARE_RDZV_TCP_STORE 后处理程序是否正确禁用代理存储
    @patch.dict(os.environ, {"TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1"})
    def test_use_agent_store_is_disabled(self):
        # 创建处理程序实例，使用环境变量禁用代理存储的配置
        handler = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )

        # 断言处理程序的 use_agent_store 属性为 False
        self.assertFalse(handler.use_agent_store)
    # 使用装饰器 patch.object，将 dist.PrefixStore 替换为 Mock 对象
    @patch.object(dist, "PrefixStore")
    # 定义测试方法，用于验证从后端共享 TCP 存储
    def test_share_tcp_store_from_backend(self, prefix_store_class_mock):
        # 创建 Mock 对象作为 prefix_store，并设置装饰器返回该对象
        prefix_store = Mock(spec=dist.PrefixStore)
        prefix_store_class_mock.return_value = prefix_store

        # 创建 Mock 对象作为 tcp_store，并设置主机地址和端口号
        tcp_store = Mock(spec=dist.TCPStore)
        expected_addr = "expected_address"
        expected_port = 54321
        type(tcp_store).host = PropertyMock(return_value=expected_addr)
        type(tcp_store).port = PropertyMock(return_value=expected_port)
        # 将 tcp_store 注入 self._store
        self._store = tcp_store

        # 创建两个处理器对象，并启动线程执行其 next_rendezvous 方法
        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler1_thread.start()
        handler2_thread.start()

        # 等待线程执行完成并获取返回的 RendezvousInfo 对象
        rdzv_info1: RendezvousInfo = handler1_thread.join()
        rdzv_info2: RendezvousInfo = handler2_thread.join()

        # 断言确保 rdzv_info1 和 rdzv_info2 的 store 属性与 prefix_store 相等
        self.assertEqual(rdzv_info1.store, prefix_store)
        self.assertEqual(rdzv_info2.store, prefix_store)
        # 验证 prefix_store_class_mock 被调用，参数为指定的运行 ID 和 tcp_store 对象
        prefix_store_class_mock.assert_called_with(
            "torch.rendezvous.dummy_run_id.0", tcp_store
        )

        # 断言验证 rdzv_info1 和 rdzv_info2 的 bootstrap_store_info 属性相等
        self.assertEqual(
            rdzv_info1.bootstrap_store_info, rdzv_info2.bootstrap_store_info
        )

        # 断言验证 rdzv_info1 的 bootstrap_store_info 的 master_addr 和 master_port 与期望值相等
        self.assertEqual(rdzv_info1.bootstrap_store_info.master_addr, expected_addr)
        self.assertEqual(rdzv_info1.bootstrap_store_info.master_port, expected_port)

    # 使用装饰器 patch.dict，设置环境变量以禁用共享 RDZV TCP 存储
    @patch.dict(os.environ, {"TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1"})
    # 使用装饰器 patch.object，将 dist.PrefixStore 替换为 Mock 对象
    @patch.object(dist, "PrefixStore")
    # 定义测试方法，用于验证共享 TCP 存储被禁用
    def test_share_tcp_store_is_disabled(self, prefix_store_class_mock):
        # 创建 Mock 对象作为 prefix_store，并设置装饰器返回该对象
        prefix_store = Mock()
        prefix_store_class_mock.return_value = prefix_store

        # 设置 prefix_store 的 set 和 get 方法的返回值
        prefix_store.set.return_value = None
        prefix_store.get.return_value = b"123"

        # 创建 Mock 对象作为 tcp_store，并将其注入 self._store
        tcp_store = Mock(spec=dist.TCPStore)
        self._store = tcp_store

        # 创建两个处理器对象，并启动线程执行其 next_rendezvous 方法
        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler1_thread.start()
        handler2_thread.start()

        # 等待线程执行完成并获取返回的 RendezvousInfo 对象
        rdzv_info1: RendezvousInfo = handler1_thread.join()
        rdzv_info2: RendezvousInfo = handler2_thread.join()

        # 断言确保 rdzv_info1 和 rdzv_info2 的 store 属性与 prefix_store 相等
        self.assertEqual(rdzv_info1.store, prefix_store)
        self.assertEqual(rdzv_info2.store, prefix_store)
        # 验证 prefix_store_class_mock 被调用，参数为指定的运行 ID 和 self._store 对象
        prefix_store_class_mock.assert_called_with(
            "torch.rendezvous.dummy_run_id.0", self._store
        )
        # 断言验证 rdzv_info1 和 rdzv_info2 的 bootstrap_store_info 的 master_port 与预期值相等
        self.assertEqual(rdzv_info1.bootstrap_store_info.master_port, 123)
        self.assertEqual(rdzv_info2.bootstrap_store_info.master_port, 123)
# 定义一个名为 _InMemoryRendezvousBackend 的类，继承自 RendezvousBackend 类
class _InMemoryRendezvousBackend(RendezvousBackend):
    # 初始化方法，创建 _InMemoryRendezvousBackend 实例时调用
    def __init__(self):
        # 创建一个线程锁对象，用于线程同步
        self._lock = threading.Lock()
        # 初始化状态变量为 None
        self._state = None
        # 初始化令牌变量为 None
        self._token = None

    # 返回属性 name 的值为 "_in_memory_backend"
    @property
    def name(self):
        return "_in_memory_backend"

    # 获取当前状态和令牌的元组，使用线程锁确保线程安全
    def get_state(self):
        with self._lock:
            # 如果状态为 None，则返回 None
            if self._state is None:
                return None
            # 否则返回状态和令牌的元组
            return (self._state, self._token)

        # 在 with 语句外返回状态，保留该行为但实际上不执行
        return self._state

    # 设置状态和令牌，确保状态不为 None，并使用线程锁确保线程安全
    def set_state(self, state, token):
        # 如果状态为 None，则抛出 ValueError 异常
        if state is None:
            raise ValueError("State cannot be None.")
        with self._lock:
            # 如果令牌为 None 且当前令牌不为 None，则返回 None
            if token is None and self._token is not None:
                return None
            # 如果当前令牌与传入令牌不相等，则返回 None
            if self._token != token:
                return None

            # 设置新的状态和更新令牌
            self._state = state
            self._token = self._token + 1 if self._token is not None else 0
```