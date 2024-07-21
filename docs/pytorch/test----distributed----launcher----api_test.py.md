# `.\pytorch\test\distributed\launcher\api_test.py`

```
# 指定 Python 解释器路径，用于环境变量中的 shebang
#!/usr/bin/env python3

# 代码拥有者信息，用列表标明负责人信息，包括值班负责人为 'r2p'
# Owner(s): ["oncall: r2p"]

# 版权声明，声明代码版权归 Facebook, Inc. 及其关联公司所有
# Copyright (c) Facebook, Inc. and its affiliates.
# 保留所有权利。
#
# 此源代码在根目录下的 LICENSE 文件中以 BSD 风格许可证授权

# 导入必要的库和模块
import multiprocessing as mp  # 多进程支持模块
import os  # 操作系统接口模块
import shutil  # 文件操作模块
import signal  # 信号处理模块
import sys  # 系统相关的参数和函数
import tempfile  # 创建临时文件和目录模块
import time  # 时间操作模块
import unittest  # 单元测试框架模块
import uuid  # 生成唯一标识符模块
from contextlib import closing  # 上下文管理模块，用于创建上下文管理器
from typing import Any, Dict, Optional  # 类型提示模块，用于静态类型检查

from unittest import mock  # 单元测试模拟对象模块
from unittest.mock import MagicMock, Mock, patch  # 单元测试模拟对象的相关方法和类

import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # 分布式计算模块
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState  # 弹性训练代理服务器 API 模块
from torch.distributed.elastic.multiprocessing.api import SignalException  # 弹性训练多进程 API 模块中的异常类
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError  # 弹性训练多进程错误处理模块
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer  # Etcd 服务器模块
from torch.distributed.elastic.utils import get_socket_with_port  # 获取带有指定端口的套接字模块
from torch.distributed.launcher.api import (
    _get_entrypoint_name,  # 获取入口点名称函数
    elastic_launch,  # 弹性启动函数
    launch_agent,  # 启动代理函数
    LaunchConfig,  # 启动配置类
)
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,  # 如果在沙堡中跳过但通过的测试函数
    TEST_WITH_DEV_DBG_ASAN,  # 开发调试 ASAN 的测试函数
)


def path(script):
    # 返回当前脚本文件所在目录与指定脚本文件名拼接后的完整路径
    return os.path.join(os.path.dirname(__file__), script)


def simple_rank_scale():
    # 获取环境变量中的 RANK 值并转换为整数
    rank = int(os.environ["RANK"])
    # 返回 10 加上当前进程的排名值
    return 10 + rank


def function_with_bug():
    # 抛出一个运行时错误，用于测试错误处理机制
    raise RuntimeError("test error")


def get_test_launch_config(
    rdzv_endpoint: str,
    min_nodes: int,
    max_nodes: int,
    nproc_per_node: int,
    run_id: str = "",
    rdzv_backend: str = "etcd",
    config: Optional[Dict[str, Any]] = None,
) -> LaunchConfig:
    rdzv_configs = {}
    if config:
        rdzv_configs.update(config)
    # 返回一个启动配置对象，包括最小节点数、最大节点数、每节点进程数等信息
    return LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id=run_id,
        rdzv_endpoint=rdzv_endpoint,
        monitor_interval=0.1,
        rdzv_backend=rdzv_backend,
        start_method="spawn",
        max_restarts=0,
        rdzv_configs=rdzv_configs,
    )


def elastic_launch_wrapper(
    test_dir: str,
    rdzv_endpoint: str,
    min_nodes: int,
    max_nodes: int,
    nproc_per_node: int,
    run_id: str,
):
    """一个包装函数，用于类 `elastic_launch`，以确保多进程返回正确的退出码。"""
    elastic_launch(
        get_test_launch_config(
            rdzv_endpoint, min_nodes, max_nodes, nproc_per_node, run_id
        ),
        sys.executable,
    )("-u", path("bin/test_script.py"), f"--touch-file-dir={test_dir}")


def _dist_sum(wait=0):
    # 获取环境变量中的 RANK 值并转换为整数
    rank = int(os.environ["RANK"])
    # 使用 Gloo 后端初始化进程组
    dist.init_process_group(backend="gloo")
    # 创建一个包含当前进程排名的张量
    t = torch.tensor(rank)

    # 等待指定秒数
    time.sleep(wait)
    # 对所有进程的张量 t 进行全局求和操作
    dist.all_reduce(t, op=dist.reduce_op.SUM)
    # 返回全局求和后的张量 t 的数值部分
    return t.item()


# 常量定义：指定 torch.distributed.launcher.api.LocalElasticAgent.run 的值
ELASTIC_AGENT_RUN = "torch.distributed.launcher.api.LocalElasticAgent.run"
# 常量定义：指定 torch.distributed.launcher.api.events.record 的值
EVENTS_RECORD = "torch.distributed.launcher.api.events.record"
# 常量定义：指定 get_rdsv_handler 的值（未完整定义）
GET_RDZV_HANDLER = (
    # 导入PyTorch分布式弹性训练中的Rendezvous注册表模块中的获取Rendezvous处理程序函数
    "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
# 定义一个自定义异常类 MockException，继承自内置异常类 Exception
class MockException(Exception):
    pass

# 生成一个短 UUID 的字符串，只使用 UUID 的前八位字符
def short_hash():
    return str(uuid.uuid4()).split("-")[0]

# 定义一个单元测试类 ElasticLaunchTest，继承自 unittest.TestCase
class ElasticLaunchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 启动一个独立的、单进程的 etcd 服务器，用于所有测试
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()
        # 获取 etcd 服务器的地址端点
        cls._etcd_endpoint = cls._etcd_server.get_endpoint()

    @classmethod
    def tearDownClass(cls):
        # 停止 etcd 服务器
        cls._etcd_server.stop()

    def setUp(self):
        # 创建一个临时目录用于测试
        self.test_dir = tempfile.mkdtemp()

        # 清除所有可能存在的环境变量
        for env in os.environ.keys():
            if env.startswith("PET_"):
                del os.environ[env]

        # 在父进程设置一个环境变量作为标志
        # 这个变量在子进程中应该存在，并在 ``bin/test_script.py`` 中进行断言
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"
        # 设置 OpenMP 线程数为 1
        os.environ["OMP_NUM_THREADS"] = str(1)

    def tearDown(self):
        # 删除临时测试目录及其内容
        shutil.rmtree(self.test_dir)

    def check_works_ran(self, world_size: int):
        # 断言测试目录中存在的文件集合与预期的全局进程数相匹配
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_python(self):
        nnodes = 1
        nproc_per_node = 4

        # 使用 elastic_launch 函数启动 Python 脚本
        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

        # 确保所有 worker 运行完毕
        # 每个 worker 会创建一个文件，文件名为其全局排名
        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_python_local_rank_transfer(self):
        nnodes = 1
        nproc_per_node = 4

        # 使用 elastic_launch 函数启动 Python 脚本
        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

        # 确保所有 worker 运行完毕
        # 每个 worker 会创建一个文件，文件名为其全局排名
        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_bash(self):
        nnodes = 1
        nproc_per_node = 4

        # 使用 elastic_launch 函数启动 Bash 脚本
        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            path("bin/test_script.sh"),
        )(f"{self.test_dir}")

        # 确保所有 worker 运行完毕
        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)
    # 在满足特定条件时跳过测试，条件是在开发/调试模式下使用地址消毒工具不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_function(self):
        # 设置节点数和每个节点的进程数
        nnodes = 1
        nproc_per_node = 4
    
        # 使用 elastic_launch 启动函数，返回结果
        res = elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            simple_rank_scale,
        )()
    
        # 预期结果
        expected_res = [10, 11, 12, 13]
        # 实际结果，对结果值进行排序
        actual_res = sorted(value for value in res.values())
        # 断言预期结果和实际结果相等
        self.assertEqual(expected_res, actual_res)
    
    # 在满足特定条件时跳过测试，条件是在开发/调试模式下使用地址消毒工具不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_dist_sum_with_static_rdzv(self):
        # 设置节点数和每个节点的进程数
        nnodes = 1
        nproc_per_node = 4
        # 获取带有端口的套接字
        sock = get_socket_with_port()
        with closing(sock):
            # 获取主机端口号
            master_port = sock.getsockname()[1]
        # 设置静态 Rendezvous 点的地址
        rdzv_endpoint = f"127.0.0.1:{master_port}"
        rank = 0
        rdzv_config = {
            "rank": rank,
        }
    
        # 使用 elastic_launch 启动函数，返回结果
        res = elastic_launch(
            get_test_launch_config(
                rdzv_endpoint,
                nnodes,
                nnodes,
                nproc_per_node,
                rdzv_backend="static",
                config=rdzv_config,
            ),
            _dist_sum,
        )()
    
        # 预期结果为每个进程数值的和组成的列表
        expected_res = [sum(range(nproc_per_node))] * nproc_per_node
        # 实际结果，对结果值进行排序
        actual_res = sorted(value for value in res.values())
        # 断言预期结果和实际结果相等
        self.assertEqual(expected_res, actual_res)
    
    # 在满足特定条件时跳过测试，条件是在开发/调试模式下使用地址消毒工具不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic(self):
        nproc_per_node = 4
    
        # 使用 elastic_launch 启动函数，执行测试脚本
        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, 1, 2, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")
    
        world_size = nproc_per_node
        # 检查运行结果是否正常
        self.check_works_ran(world_size)
    
    # 使用 mock.patch 修饰器模拟 torch.distributed.elastic.events.record 方法
    @mock.patch("torch.distributed.elastic.events.record")
    def test_launch_elastic_worker_raise_exception(self, record_mock):
        """
        断言当工作程序失败并且启动器引发异常时，指示工作进程失败。
        """
        nproc_per_node = 4
    
        # 断言引发 ChildFailedError 异常
        with self.assertRaises(ChildFailedError):
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 2, nproc_per_node),
                sys.executable,
            )("-u", path("bin/test_script.py"), "--fail")
    
        # 断言 record_mock 方法被调用一次
        record_mock.assert_called_once()
    
    # 使用两个 mock.patch 修饰器分别模拟 torch.distributed.elastic.events.record 和 torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent.run 方法
    @mock.patch("torch.distributed.elastic.events.record")
    @mock.patch(
        "torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent.run"
    )
    def test_launch_elastic_agent_raise_exception(self, record_mock, mock_agent_run):
        """
        Asserts that when the agent raises an exception
        the launcher re-raises the original exception.
        """
        # 设置 mock_agent_run 方法的副作用为引发 MockException 异常
        mock_agent_run.side_effect = MockException
        # 断言调用 elastic_launch 时会抛出 MockException 异常
        with self.assertRaises(MockException):
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 2, 4),
                sys.executable,
            )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")
        # 断言 record_mock.assert_called_once() 方法被调用一次
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_multiple_agents(self):
        # 设置测试参数
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        nnodes = 2
        run_id = str(uuid.uuid4().int)

        # 创建进程列表
        procs = []
        # 使用 spawn 上下文来获取 multiprocessing 上下文
        ctx = mp.get_context("spawn")
        # 循环创建 nnodes - 1 个进程
        for _ in range(nnodes - 1):
            p = ctx.Process(
                target=elastic_launch_wrapper,
                args=(
                    self.test_dir,
                    self._etcd_endpoint,
                    min_nodes,
                    max_nodes,
                    nproc_per_node,
                    run_id,
                ),
            )
            procs.append(p)
            p.start()

        # 调用 elastic_launch_wrapper 启动一个进程
        elastic_launch_wrapper(
            self.test_dir,
            self._etcd_endpoint,
            min_nodes,
            max_nodes,
            nproc_per_node,
            run_id,
        )

        # 等待并验证每个进程的退出码为 0
        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        # 确保所有 worker 进程都运行过
        # 每个 worker 进程会创建一个以其全局排名命名的文件
        world_size = nnodes * nproc_per_node
        # 断言目录中的文件集合与预期的全局排名集合相等
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    def test_launch_shutdown(self, agent_mock_cls):
        # 创建 Mock 对象并设置其 run 方法的返回值为成功的 RunResult
        agent_mock = Mock()
        agent_mock.run.return_value = RunResult(WorkerState.SUCCEEDED)
        agent_mock_cls.return_value = agent_mock
        rdzv_handler_mock = Mock()
        # 使用 patch 替换 "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler" 的调用结果为 rdzv_handler_mock
        with patch(
            "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
        ) as param_mock:
            param_mock.return_value = rdzv_handler_mock
            # 调用 elastic_launch 启动一个进程
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 1, 4),
                sys.executable,
            )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

            # 断言 rdzv_handler_mock.shutdown 方法被调用一次
            rdzv_handler_mock.shutdown.assert_called_once()
    # 测试函数，用于验证 _get_entrypoint_name 函数的不同输入情况下的返回值是否符合预期
    def test_get_entrypoint_name(self):
        # 验证当函数参数为 simple_rank_scale 时，返回值是否为 "simple_rank_scale"
        self.assertEqual(
            "simple_rank_scale", _get_entrypoint_name(simple_rank_scale, [])
        )
        # 验证当函数参数为 sys.executable 时，且参数列表为空时，返回值是否为空字符串
        self.assertEqual("", _get_entrypoint_name(sys.executable, []))
        # 验证当函数参数为 sys.executable 时，且参数列表为 ["-u"] 时，返回值是否为空字符串
        self.assertEqual("", _get_entrypoint_name(sys.executable, ["-u"]))
        # 验证当函数参数为 sys.executable 时，且参数列表为 ["-u", "test_script.py"] 时，返回值是否为 "test_script.py"
        self.assertEqual(
            "test_script.py",
            _get_entrypoint_name(sys.executable, ["-u", "test_script.py"]),
        )
        # 验证当函数参数为 None 时，返回值是否为空字符串
        self.assertEqual("", _get_entrypoint_name(None, []))

    # 使用 patch 装饰器模拟对 ELASTIC_AGENT_RUN 和 GET_RDZV_HANDLER 的操作
    @patch(ELASTIC_AGENT_RUN)
    @patch(GET_RDZV_HANDLER)
    # 测试函数，验证在代理信号发生时是否正确处理 rendezvous handler 的关闭
    def test_rdzv_handler_shutdown_on_agent_signal(self, mock_get_rdzv, mock_agent_run):
        # 获取测试用的启动配置
        config = get_test_launch_config(
            self._etcd_endpoint, min_nodes=1, max_nodes=1, nproc_per_node=1
        )

        # 对于每个信号值进行测试
        for sigval in [signal.SIGTERM, signal.SIGINT]:
            # 在 patch EVENTS_RECORD 的上下文中执行
            with patch(EVENTS_RECORD) as record_event_mock:
                # 创建 MagicMock 对象作为 rendezvous handler mock
                rdzv_handler_mock = MagicMock()
                # 设置 mock 的 get_run_id 方法返回一个短哈希字符串
                rdzv_handler_mock.get_run_id.return_value = short_hash()
                # mock_get_rdzv 返回该 mock 对象
                mock_get_rdzv.return_value = rdzv_handler_mock

                # 设置 mock_agent_run 抛出 SignalException 异常
                mock_agent_run.side_effect = SignalException("test", sigval)
                # 验证是否抛出 SignalException 异常
                with self.assertRaises(SignalException):
                    # 调用 launch_agent 函数，验证是否正确抛出异常
                    launch_agent(config, simple_rank_scale, [])
                # 验证 rdzv_handler_mock.shutdown 方法未被调用
                rdzv_handler_mock.shutdown.assert_not_called()
                # 验证 record_event_mock.assert_called_once() 方法被调用一次
                record_event_mock.assert_called_once()

    # 使用 patch 装饰器模拟对 ELASTIC_AGENT_RUN 和 GET_RDZV_HANDLER 的操作
    @patch(ELASTIC_AGENT_RUN)
    @patch(GET_RDZV_HANDLER)
    # 测试函数，验证在代理运行中出现运行时错误时是否正确处理 rendezvous handler 的关闭
    def test_rdzv_handler_shutdown_on_agent_error(self, mock_get_rdzv, mock_agent_run):
        # 获取测试用的启动配置
        config = get_test_launch_config(
            self._etcd_endpoint, min_nodes=1, max_nodes=1, nproc_per_node=1
        )

        # 在 patch EVENTS_RECORD 的上下文中执行
        with patch(EVENTS_RECORD) as record_event_mock:
            # 创建 MagicMock 对象作为 rendezvous handler mock
            rdzv_handler_mock = MagicMock()
            # 设置 mock 的 get_run_id 方法返回一个短哈希字符串
            rdzv_handler_mock.get_run_id.return_value = short_hash()
            # mock_get_rdzv 返回该 mock 对象
            mock_get_rdzv.return_value = rdzv_handler_mock

            # 设置 mock_agent_run 抛出 RuntimeError 异常
            mock_agent_run.side_effect = RuntimeError("any other exception")
            # 验证是否抛出 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                # 调用 launch_agent 函数，验证是否正确抛出异常
                launch_agent(config, simple_rank_scale, [])
            # 验证 rdzv_handler_mock.shutdown 方法被调用一次
            rdzv_handler_mock.shutdown.assert_called_once()
            # 验证 record_event_mock.assert_called_once() 方法被调用一次
            record_event_mock.assert_called_once()
```