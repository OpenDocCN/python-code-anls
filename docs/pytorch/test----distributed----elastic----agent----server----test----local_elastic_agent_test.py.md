# `.\pytorch\test\distributed\elastic\agent\server\test\local_elastic_agent_test.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json  # 导入 JSON 模块
import multiprocessing as mp  # 导入 multiprocessing 模块，用于多进程处理
import os  # 导入 os 模块，提供操作系统相关的功能
import shutil  # 导入 shutil 模块，提供高级文件操作
import signal  # 导入 signal 模块，用于处理信号
import socket  # 导入 socket 模块，用于网络通信
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import time  # 导入 time 模块，提供时间相关的功能
import unittest  # 导入 unittest 模块，用于编写和运行测试
import uuid  # 导入 uuid 模块，提供唯一标识符的生成
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于声明数据类
from typing import Callable, Dict, List, Optional, Tuple  # 导入类型提示相关的功能
from unittest import mock  # 导入 unittest 中的 mock 模块，用于模拟对象
from unittest.mock import Mock, patch  # 导入 unittest.mock 中的 Mock 和 patch

import torch  # 导入 PyTorch 模块
import torch.distributed as dist  # 导入 PyTorch 分布式模块
import torch.distributed.elastic.rendezvous.registry as rdzv_registry  # 导入 PyTorch Elastic 中的注册表
import torch.distributed.rpc as rpc  # 导入 PyTorch RPC 模块
from torch.distributed.elastic.agent.server.api import (  # 导入 PyTorch Elastic 中的服务器 API 相关类
    RunResult,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import (  # 导入 PyTorch Elastic 中的本地弹性代理
    LocalElasticAgent,
    TORCHELASTIC_HEALTH_CHECK_PORT,
    TORCHELASTIC_TIMER_FILE,
)
from torch.distributed.elastic.multiprocessing import (  # 导入 PyTorch Elastic 中的多进程支持相关类和函数
    DefaultLogsSpecs,
    Std,
)
from torch.distributed.elastic.multiprocessing.errors import (  # 导入 PyTorch Elastic 中的错误处理类
    ChildFailedError,
    record,
)
from torch.distributed.elastic.rendezvous import RendezvousParameters  # 导入 PyTorch Elastic 中的会面参数
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer  # 导入 PyTorch Elastic 中的 Etcd 服务器
from torch.distributed.rpc.backend_registry import BackendType  # 导入 PyTorch RPC 的后端类型
from torch.testing._internal.common_utils import (  # 导入 PyTorch 内部测试中的实用函数
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_TSAN,
)


def init_rpc(name, backend):
    """
    Initializes RPC with given name and backend.

    Args:
        name (str): Name of the RPC instance.
        backend (BackendType): Backend type for RPC.

    Environment Variables:
        RANK (str): Rank of the current process.
        WORLD_SIZE (str): Total number of processes.

    Raises:
        KeyError: If RANK or WORLD_SIZE environment variables are not set.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rpc.init_rpc(
        name=name,
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def rpc_master(msg):
    """
    RPC function for master node.

    Args:
        msg (str): Message to send to worker.

    Returns:
        str: Response message from worker.
    """
    init_rpc("master", BackendType.TENSORPIPE)
    ret = rpc.rpc_sync(to="worker", func=_echo, args=(msg,))
    rpc.shutdown()
    return f"{ret} from worker"


def rpc_worker():
    """
    RPC function for worker node.
    """
    init_rpc("worker", BackendType.TENSORPIPE)
    rpc.shutdown()


def _happy_function():
    """
    A happy function that does nothing.
    """
    return


def _sad_function():
    """
    A function that raises a RuntimeError.
    """
    raise RuntimeError("sad because i throw")


def dummy_compute() -> torch.Tensor:
    """
    returns a predefined size random Tensor
    """
    return torch.rand(100, 100)


def dummy_compute_simulate_rank_failure() -> torch.Tensor:
    """
    fails rank 1 once
    in other cases, returns a predefined size random Tensor
    """
    if os.environ["RANK"] == "1" and os.environ["TORCHELASTIC_RESTART_COUNT"] == "0":
        os.kill(os.getpid(), 9)
    return torch.rand(100, 100)


def _fatal_signal_function(expected_error_index: int, sig: int):
    """
    Sends a fatal signal to the current process if its rank matches the expected_error_index.

    Args:
        expected_error_index (int): Rank index expected to receive the signal.
        sig (int): Signal number.
    """
    rank = int(os.environ["RANK"])
    if rank == expected_error_index:
        os.kill(os.getpid(), sig)


def _check_master_port_addr_override(
    expected_master_addr: str, expected_master_port: int
):
    """
    Checks if the master address and port are overridden correctly in the environment variables.

    Args:
        expected_master_addr (str): Expected master address.
        expected_master_port (int): Expected master port.

    Environment Variables:
        MASTER_ADDR (str): Actual master address.
        MASTER_PORT (str): Actual master port.
    """
    actual_master_addr = os.environ["MASTER_ADDR"]
    actual_master_port = int(os.environ["MASTER_PORT"])
    # 如果预期的主地址和实际的主地址不匹配，并且预期的主端口和实际的主端口也不匹配，则执行下面的代码块
    raise RuntimeError(
        f"Expected addr: {expected_master_addr}:{expected_master_port}, got addr: {actual_master_addr}:{actual_master_port}"
    )
def _bipolar_function():
    # 从环境变量中获取当前进程的排名
    rank = int(os.environ["RANK"])
    # 如果排名是偶数，则调用 _happy_function()
    if rank % 2 == 0:
        _happy_function()
    else:
        # 否则调用 _sad_function()
        _sad_function()


def _bipolar_sleep_function(sleep_sec):
    # 从环境变量中获取当前进程的排名
    rank = int(os.environ["RANK"])
    # 如果排名是偶数，则调用 _sleep(sleep_sec)
    if rank % 2 == 0:
        _sleep(sleep_sec)
    else:
        # 否则调用 _sad_function()
        _sad_function()


def _dist_sum(wait=0):
    # 从环境变量中获取当前进程的排名和总进程数
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # 初始化分布式进程组，使用 gloo 后端
    dist.init_process_group(backend="gloo")
    # 创建包含当前排名的张量
    t = torch.tensor(rank)

    # 等待一段时间
    time.sleep(wait)
    # 对所有进程的排名进行求和，使用 SUM 操作
    dist.all_reduce(t, op=dist.reduce_op.SUM)

    # 预期的排名总和是所有进程排名的累加和
    expected_sum = sum(range(world_size))
    actual = t.item()
    # 检查实际排名总和是否与预期相符
    if expected_sum != actual:
        raise RuntimeError(f"Expected rank sum {expected_sum}, got {actual}")


def _sleep(sleep_sec) -> int:
    # 线程休眠指定秒数
    time.sleep(sleep_sec)
    # 返回当前进程的排名作为整数
    return int(os.environ["RANK"])


@dataclass
class RankInfo:
    rank: int
    role_rank: int
    group_rank: int
    role_world_size: int
    world_size: int


def _get_role_info() -> RankInfo:
    # 从环境变量中获取当前进程的各种角色信息
    rank = int(os.environ["RANK"])
    role_rank = int(os.environ["ROLE_RANK"])
    group_rank = int(os.environ["GROUP_RANK"])
    role_world_size = int(os.environ["ROLE_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    # 返回一个包含角色信息的 RankInfo 对象
    return RankInfo(rank, role_rank, group_rank, role_world_size, world_size)


def _echo(msg):
    # 返回传入的消息参数
    return msg


def _check_env_function():
    # 检查以下环境变量是否存在，如果不存在会抛出异常
    # 用途是验证环境设置是否齐全
    env_vars = [
        "RANK",
        "LOCAL_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "GROUP_RANK",
        "LOCAL_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "WORLD_SIZE",
        "GROUP_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ]
    for var in env_vars:
        _ = os.environ[var]


def _check_env_value(key: str, expected: str):
    # 检查指定的环境变量是否存在，并且其值是否与预期值匹配
    # 如果不匹配则抛出异常
    if key not in os.environ:
        raise RuntimeError(f"Environment variable {key} not found in os.environ")
    else:
        actual = os.getenv(key)
        if expected != actual:
            raise RuntimeError(
                f"os.environ['{key}']={actual}"
                f" does not equal the expected value: {expected}"
            )


def _check_local_watchdog_setup(key: str, should_exist: bool):
    # 检查指定的环境变量是否存在或不存在，并根据情况抛出异常
    if should_exist and key not in os.environ:
        raise RuntimeError(f"Environment variable {key} not found in os.environ")
    if not should_exist and key in os.environ:
        raise RuntimeError(f"Environment variable {key} found in os.environ")


def acquire_available_port():
    """
    使用 sockets 获取操作系统中可用的端口，用于其他用途
    """
    """
    Note: To reduce the race condition where another process grabs the port
          after this function returns an available port, we should aim to use
          the port as quickly as possible.
    """

    # 获取本地地址信息列表，包括 IPv4 和 IPv6 地址族，以及对应的套接字类型
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )

    # 遍历地址信息列表
    for addr in addrs:
        # 解构地址信息
        family, type, proto, _, _ = addr
        try:
            # 创建套接字对象
            s = socket.socket(family, type, proto)
            # 绑定套接字到本地随机端口
            s.bind(("localhost", 0))
            # 开始监听连接
            s.listen(0)
            # 获取绑定后的套接字地址信息，返回的是 (host, port)
            port = s.getsockname()[1]
            # 关闭套接字
            s.close()
            # 返回获取到的可用端口号
            return port
        except OSError as e:
            # 关闭套接字，在捕获 OSError 异常后处理
            s.close()
            # 打印套接字创建失败的错误信息
            print(f"Socket creation attempt failed: {e}")

    # 如果所有地址尝试都失败，则抛出运行时错误
    raise RuntimeError("Failed to create a socket")
@dataclass
class Conf:
    """
    Holds arguments to launch an agent (e.g. simulates an agent run on a node).

    """

    entrypoint: Callable
    local_world_size: int
    args: Tuple = ()
    role: str = "default"
    redirects: Std = Std.NONE
    tee: Std = Std.NONE



class LocalElasticAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 启动一个独立的单进程 etcd 服务器，用于所有测试
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # 停止独立的 etcd 服务器
        cls._etcd_server.stop()

    def setUp(self):
        # 创建一个临时目录，用于当前测试类的临时文件存储
        self._test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        # 生成一个随机的运行 ID，并截取前部分作为当前测试的运行 ID
        self._run_id = str(uuid.uuid4()).split("-")[0]

    def tearDown(self):
        # 删除临时目录及其所有内容
        shutil.rmtree(self._test_dir)

    def log_dir(self) -> str:
        # 在临时目录中创建一个以 'torchelastic_' 开头的临时日志目录，并返回其路径
        return tempfile.mkdtemp(prefix="torchelastic_", dir=self._test_dir)

    def get_worker_spec(
        self,
        node_config: Conf,
        min_nodes=1,
        max_nodes=1,
        max_restarts=0,
        monitor_interval=0.01,
        master_addr_override: Optional[str] = None,
        master_port_override: Optional[int] = None,
        is_host=True,
    ):
        # 创建 RendezvousParameters 对象，用于描述分布式同步的参数
        rdzv_params = RendezvousParameters(
            backend=self._backend,
            endpoint=self._endpoint,
            run_id=self._run_id,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            is_host=is_host,
        )
        # 根据 rdzv_params 获取对应的 RendezvousHandler 对象
        rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_params)
        # 返回 WorkerSpec 对象，描述了一个 worker 的规格及配置
        return WorkerSpec(
            role=node_config.role,
            local_world_size=node_config.local_world_size,
            entrypoint=node_config.entrypoint,
            args=node_config.args,
            rdzv_handler=rdzv_handler,
            max_restarts=max_restarts,
            monitor_interval=monitor_interval,
            master_addr=master_addr_override,
            master_port=master_port_override,
        )

    def get_agent(
        self,
        spec: WorkerSpec,
        node_config: Conf,
        start_method: str = "spawn",
        exit_barrier_timeout=5,
        log_line_prefix_template: Optional[str] = None,
    ) -> LocalElasticAgent:
        # 创建 LocalElasticAgent 对象，用于管理本地弹性代理的启动和管理
        return LocalElasticAgent(
            spec,
            start_method=start_method,
            exit_barrier_timeout=exit_barrier_timeout,
            logs_specs=DefaultLogsSpecs(
                log_dir=self.log_dir(),
                redirects=node_config.redirects,
                tee=node_config.tee,
            ),
            log_line_prefix_template=log_line_prefix_template,
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.multiprocessing.errors.record`.
    @record



        # 此处使用 @record 装饰器来标记一个函数，具体功能由装饰器的实现决定
    def run_agent(
        self,
        conf: Conf,
        agent_results: Optional[mp.Queue] = None,  # 定义一个可选的多进程队列，用于存储代理程序的结果
        min_nodes=1,  # 最小节点数，默认为1
        max_nodes=1,  # 最大节点数，默认为1
        start_method: str = "spawn",  # 启动方法，默认为"spawn"
        max_restarts: int = 0,  # 最大重启次数，默认为0
        exit_barrier_timeout=5,  # 退出屏障超时时间，默认为5秒
        master_addr_override: Optional[str] = None,  # 主地址覆盖，可选参数，默认为None
        master_port_override: Optional[int] = None,  # 主端口覆盖，可选参数，默认为None
        is_host=True,  # 是否为主机，默认为True
        monitor_interval=0.01,  # 监控间隔，默认为0.01秒
        log_line_prefix_template: Optional[str] = None,  # 日志行前缀模板，可选参数，默认为None
    ) -> Optional[RunResult]:
        """
        Runs a single agent. This method can be called either on a separate process
        or the main test process. When calling this method on a separate process make
        sure to pass the ``agent_results`` multiprocessing Queue so that the agent's
        run results can be returned. If ``agent_results`` is omitted, then the
        run result is returned from the method.
        """
        
        # 根据参数配置获取工作节点规格
        spec = self.get_worker_spec(
            node_config=conf,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            max_restarts=max_restarts,
            master_addr_override=master_addr_override,
            master_port_override=master_port_override,
            is_host=is_host,
            monitor_interval=monitor_interval,
        )
        
        # 根据规格获取代理程序实例
        agent = self.get_agent(
            spec=spec,
            node_config=conf,
            start_method=start_method,
            exit_barrier_timeout=exit_barrier_timeout,
            log_line_prefix_template=log_line_prefix_template,
        )
        
        # 运行代理程序并获取结果
        result = agent.run()
        
        # 关闭分布式协调处理器的处理
        spec.rdzv_handler.shutdown()

        # 如果提供了代理结果队列，将结果放入队列中
        if agent_results:
            agent_results.put((conf.role, result))

        # 如果运行失败，则抛出异常
        if result.is_failed():
            raise ChildFailedError(spec.get_entrypoint_name(), result.failures)
        else:
            # 如果没有提供代理结果队列，则直接返回结果
            if not agent_results:
                return result

    def run_job(
        self,
        node_configs: List[Conf],
        exit_barrier_timeout: int = 5,
        log_line_prefix_template: Optional[str] = None,
    ) -> Dict[str, List[RunResult]]:
        """
        Simulates running a distributed job by running multiple agents
        (one on each process). Agent 0 is run on the main process for
        test coverage and ease of debugging
        """

        # 获取节点配置的数量
        nnodes = len(node_configs)

        # 创建一个多进程队列，用于存储每个代理运行后的结果 (role, RunResult)
        agent_results = mp.Queue()

        # 在主进程上运行第一个节点配置的第一个代理，以进行测试覆盖和调试方便
        # 注意，反向循环很重要，因为在主进程上运行 fn 会阻塞
        procs = []
        for node_idx in reversed(range(len(node_configs))):
            conf = node_configs[node_idx]
            run_agent_args = {
                "conf": conf,
                "agent_results": agent_results,
                "min_nodes": nnodes,
                "max_nodes": nnodes,
                "start_method": "spawn",
                "max_restarts": 0,
                "exit_barrier_timeout": exit_barrier_timeout,
                "is_host": node_idx == 0,
                "log_line_prefix_template": log_line_prefix_template,
            }
            # 创建一个新的进程来运行代理，并传递运行代理所需的参数
            p = mp.Process(target=self.run_agent, kwargs=run_agent_args)
            procs.append(p)
            p.start()
        for p in procs:
            p.join()

        # 收集并组织代理运行后的结果到一个字典中
        results: Dict[str, List[RunResult]] = {}
        while not agent_results.empty():
            role, run_result = agent_results.get()
            results.setdefault(role, []).append(run_result)
        return results

    def run_test_with_backend(self, backend: str, test_to_run: Callable):
        """
        Sets the backend and determines the endpoint before running the
        given test.

        Note: This method must be invoked to run any test functions that spawn
              an agent. This is because this function sets the backend and
              endpoint parameters.
        """
        # 设置后端类型
        self._backend = backend

        # 根据后端类型设置相应的端点地址
        if self._backend == "etcd-v2" or self._backend == "etcd":
            self._endpoint = self._etcd_server.get_endpoint()
        else:
            # 默认使用 c10d 后端
            self._endpoint = f"localhost:{acquire_available_port()}"

        # 运行指定的测试函数
        test_to_run()

    def dummy_compute(self):
        # 运行一个代理，并验证返回结果
        res = self.run_agent(Conf(entrypoint=dummy_compute, local_world_size=2))
        self.assertFalse(res.is_failed())
        for return_value in res.return_values.values():
            self.assertIsInstance(return_value, torch.Tensor)
            self.assertEqual((100, 100), return_value.shape)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_dummy_compute_c10d(self):
        # 使用 c10d 后端运行 dummy_compute 测试函数
        self.run_test_with_backend(backend="c10d", test_to_run=self.dummy_compute)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_dummy_compute_etcd(self):
        # 使用 etcd 后端运行 dummy_compute 测试
        self.run_test_with_backend(backend="etcd", test_to_run=self.dummy_compute)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_dummy_compute_etcd_v2(self):
        # 使用 etcd-v2 后端运行 dummy_compute 测试
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.dummy_compute)

    def run_happy_function(self):
        # 运行 happy_function，检查结果是否失败并且返回值为 None
        res = self.run_agent(Conf(entrypoint=_happy_function, local_world_size=2))
        self.assertFalse(res.is_failed())
        self.assertIsNone(res.return_values[0])
        self.assertIsNone(res.return_values[1])

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_happy_function_c10d(self):
        # 使用 c10d 后端运行 run_happy_function 测试
        self.run_test_with_backend(backend="c10d", test_to_run=self.run_happy_function)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_happy_function_etcd(self):
        # 使用 etcd 后端运行 run_happy_function 测试
        self.run_test_with_backend(backend="etcd", test_to_run=self.run_happy_function)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_happy_function_etcd_v2(self):
        # 使用 etcd-v2 后端运行 run_happy_function 测试
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_happy_function
        )

    def check_master_addr_port_override(self):
        # 检查主节点地址和端口是否正确设置
        master_addr = "test_host"
        master_port = 42
        res = self.run_agent(
            Conf(
                entrypoint=_check_master_port_addr_override,
                args=(master_addr, master_port),
                local_world_size=1,
            ),
            master_addr_override=master_addr,
            master_port_override=master_port,
        )
        self.assertFalse(res.is_failed())
        self.assertIsNone(res.return_values[0])

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_check_master_addr_port_override_etcd(self):
        # 使用 etcd 后端运行 check_master_addr_port_override 测试
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.check_master_addr_port_override
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_check_master_addr_port_override_etcd_v2(self):
        # 使用 etcd-v2 后端运行 check_master_addr_port_override 测试
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.check_master_addr_port_override
        )

    def run_check_env_function(self):
        # 检查所有必需设置在用户脚本中的环境变量是否已设置
        res = self.run_agent(Conf(entrypoint=_check_env_function, local_world_size=1))
        self.assertFalse(res.is_failed())
    def run_check_nccl_async_error_handling_env(self):
        # 确保尊重 os.environ 中设置的 TORCH_NCCL_ASYNC_ERROR_HANDLING
        with patch.dict(os.environ, {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "0"}):
            # 运行代理，检查环境变量设置是否正确
            res = self.run_agent(
                Conf(
                    entrypoint=_check_env_value,
                    local_world_size=1,
                    args=("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0"),
                )
            )
            # 断言检查结果不是失败状态
            self.assertFalse(res.is_failed())

    def run_check_nccl_async_error_handling_env_default(self):
        # 如果环境变量未设置，默认为 1
        res = self.run_agent(
            Conf(
                entrypoint=_check_env_value,
                local_world_size=1,
                args=("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
            )
        )
        # 断言检查结果不是失败状态
        self.assertFalse(res.is_failed())

    def run_agent_local_watchdog_setup_enabled(self):
        # 设置 watchdog 的环境变量
        watchdog_env_name = TORCHELASTIC_TIMER_FILE
        watchdog_file_path = "/tmp/watchdog_timer_" + str(uuid.uuid4())
        os.environ[watchdog_env_name] = watchdog_file_path
        # 运行代理
        node_conf = Conf(
            entrypoint=_check_local_watchdog_setup,
            local_world_size=1,
            args=(TORCHELASTIC_TIMER_FILE, True),
        )
        # 获取代理的规格
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        # 运行代理并获取结果
        res = agent.run()
        # 断言检查结果不是失败状态
        self.assertFalse(res.is_failed())

    def run_agent_local_watchdog_setup_disabled(self):
        # 不设置 watchdog 的环境变量
        watchdog_env_name = TORCHELASTIC_TIMER_FILE
        if watchdog_env_name in os.environ:
            del os.environ[watchdog_env_name]
        # 运行代理
        node_conf = Conf(
            entrypoint=_check_local_watchdog_setup,
            local_world_size=1,
            args=(TORCHELASTIC_TIMER_FILE, False),
        )
        # 获取代理的规格
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        # 运行代理并获取结果
        res = agent.run()
        # 断言检查结果不是失败状态
        self.assertFalse(res.is_failed())

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_local_watchdog_setup_enabled_etcd(self):
        # 在 Sandcastle 环境中跳过，除非 TEST_WITH_DEV_DBG_ASAN 为真
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_local_watchdog_setup_enabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_local_watchdog_setup_enabled_c10d(self):
        # 在 Sandcastle 环境中跳过，除非 TEST_WITH_DEV_DBG_ASAN 为真
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_local_watchdog_setup_enabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_local_watchdog_setup_disabled_etcd(self):
        # 使用 etcd 后端运行 run_agent_local_watchdog_setup_disabled 测试
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_local_watchdog_setup_disabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_local_watchdog_setup_disabled_c10d(self):
        # 如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试，因为不兼容 dev/dbg asan
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_local_watchdog_setup_disabled
        )

    def run_agent_healthcheck_setup_enabled(self):
        # 设置健康检查的环境变量
        healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT
        os.environ[healthcheck_port_env_name] = "12345"
        # 运行代理程序
        node_conf = Conf(
            entrypoint=_check_local_watchdog_setup,
            local_world_size=1,
            args=(TORCHELASTIC_HEALTH_CHECK_PORT, True),
        )
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        res = agent.run()
        self.assertFalse(res.is_failed())

    def run_agent_healthcheck_setup_disabled(self):
        # 不设置健康检查的环境变量
        healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT
        if healthcheck_port_env_name in os.environ:
            del os.environ[healthcheck_port_env_name]
        # 运行代理程序
        node_conf = Conf(
            entrypoint=_check_local_watchdog_setup,
            local_world_size=1,
            args=(TORCHELASTIC_HEALTH_CHECK_PORT, False),
        )
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        res = agent.run()
        self.assertFalse(res.is_failed())

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_healthcheck_setup_enabled_etcd(self):
        # 使用 etcd 后端运行 run_agent_healthcheck_setup_enabled 测试（跳过测试如果与 dev/dbg asan 不兼容）
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_healthcheck_setup_enabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_healthcheck_setup_enabled_c10d(self):
        # 使用 c10d 后端运行 run_agent_healthcheck_setup_enabled 测试（跳过测试如果与 dev/dbg asan 不兼容）
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_healthcheck_setup_enabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_healthcheck_setup_disabled_etcd(self):
        # 使用 etcd 后端运行 run_agent_healthcheck_setup_disabled 测试（跳过测试如果与 dev/dbg asan 不兼容）
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_healthcheck_setup_disabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_healthcheck_setup_disabled_c10d(self):
        # 使用 c10d 后端运行 run_agent_healthcheck_setup_disabled 测试（跳过测试如果与 dev/dbg asan 不兼容）
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_healthcheck_setup_disabled
        )
    # 根据条件跳过，但在 Sandcastle 中传递，如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 etcd 后端运行 check_env_function 测试
    def test_run_check_env_function_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_check_env_function
        )
    
    # 根据条件跳过，但在 Sandcastle 中传递，如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 c10d 后端运行 check_nccl_async_error_handling_env 测试
    def test_run_check_nccl_async_error_handling_env_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_check_nccl_async_error_handling_env
        )
    
    # 根据条件跳过，但在 Sandcastle 中传递，如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 c10d 后端运行 check_nccl_async_error_handling_env_default 测试
    def test_run_check_nccl_async_error_handling_env_default_c10d(self):
        self.run_test_with_backend(
            backend="c10d",
            test_to_run=self.run_check_nccl_async_error_handling_env_default,
        )
    
    # 运行带有返回值的函数测试，验证返回的结果
    def run_function_with_return_value(self):
        res = self.run_agent(Conf(entrypoint=_echo, args=("foo",), local_world_size=2))
        self.assertFalse(res.is_failed())
        self.assertEqual("foo", res.return_values[0])
        self.assertEqual("foo", res.return_values[1])
    
    # 根据条件跳过，但在 Sandcastle 中传递，如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 c10d 后端运行带有返回值的函数测试
    def test_run_function_with_return_value_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_function_with_return_value
        )
    
    # 根据条件跳过，但在 Sandcastle 中传递，如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 etcd 后端运行带有返回值的函数测试
    def test_run_function_with_return_value_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_function_with_return_value
        )
    
    # 根据条件跳过，但在 Sandcastle 中传递，如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 etcd-v2 后端运行带有返回值的函数测试
    def test_run_function_with_return_value_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_function_with_return_value
        )
    
    # 运行 simple_dist_sum 测试，验证分布式求和功能
    def simple_dist_sum(self):
        res = self.run_agent(Conf(entrypoint=_dist_sum, local_world_size=2))
        self.assertFalse(res.is_failed())
        # _dist_sum 内部检查计算的总和是否有效
    
    # 根据条件跳过，但在 Sandcastle 中传递，如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 c10d 后端运行 simple_dist_sum 测试
    def test_simple_dist_sum_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.simple_dist_sum)
    
    # 根据条件跳过，但在 Sandcastle 中传递，如果 TEST_WITH_DEV_DBG_ASAN 为真，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 etcd 后端运行 simple_dist_sum 测试
    def test_simple_dist_sum_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.simple_dist_sum)
    def test_simple_dist_sum_etcd_v2(self):
        # 运行使用 etcd-v2 后端的简单分布式求和测试
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.simple_dist_sum)

    def run_distributed_sum_homogeneous(
        self, log_line_prefix_template: Optional[str] = None
    ):
        # 定义节点配置列表
        node_configs = [
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=4, tee=Std.ALL),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=4, tee=Std.ALL),
        ]
        # 当进程方法为 spawn 时，覆盖率收集器会挂起，
        # 原因是在等待 TCPStore 工作进程加入集群时卡住在 _dist_sum 上
        # TODO(aivanou): t83447589 提出正确的修复方法
        # 运行作业并获取结果
        res = self.run_job(
            node_configs, log_line_prefix_template=log_line_prefix_template
        )
        # 断言返回的结果集合长度为 2
        self.assertEqual(2, len(res["sum"]))
        ranks = set()
        # 遍历每个运行结果的返回值，检查是否有失败的情况，并更新 ranks 集合
        for run_results in res["sum"]:
            self.assertFalse(run_results.is_failed())
            ranks.update(run_results.return_values.keys())
        # 断言 ranks 集合包含了预期的排名范围
        self.assertSetEqual(set(range(4 + 4)), ranks)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_c10d(self):
        # 使用 c10d 后端运行分布式同质求和测试
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_distributed_sum_homogeneous
        )

    def test_run_with_custom_log_lines(self):
        # 定义自定义日志行前缀模板
        log_line_prefix_template = "[${role_name}-${local_rank}:${rank}]:"
        # 使用 c10d 后端运行分布式同质求和测试，传入自定义的日志行前缀模板
        self.run_test_with_backend(
            backend="c10d",
            test_to_run=lambda: self.run_distributed_sum_homogeneous(
                log_line_prefix_template
            ),
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_etcd(self):
        # 使用 etcd 后端运行分布式同质求和测试
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_distributed_sum_homogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_etcd_v2(self):
        # 使用 etcd-v2 后端运行分布式同质求和测试
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_distributed_sum_homogeneous
        )
    def run_distributed_sum_heterogeneous(self):
        # sums all ranks on 3 agents; each running 1, 2, 3 workers respectively
        # sum should be equal to 0 + (1 + 2) + (3 + 4 + 5) = 15
        # sum asserted inside _dist_sum()
        # 定义节点配置列表，每个节点扮演"sum"角色，使用_dist_sum作为入口函数，并指定本地节点数量
        node_configs = [
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=1),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=2),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=3),
        ]
        # 当进程方法为spawn时，覆盖收集器在等待TCPStore工作进程加入集群时会挂起
        # TODO(aivanou): t83447589 需要找到正确的修复方法
        # 运行任务并获取结果
        res = self.run_job(node_configs)
        # 断言结果列表长度为3
        self.assertEqual(3, len(res["sum"]))
        ranks = set()
        # 遍历结果中的每个运行结果
        for run_results in res["sum"]:
            # 断言运行结果未失败
            self.assertFalse(run_results.is_failed())
            # 更新排名集合
            ranks.update(run_results.return_values.keys())
        # 断言排名集合包含范围为0到5的所有整数
        self.assertSetEqual(set(range(1 + 2 + 3)), ranks)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_distributed_sum_heterogeneous_c10d(self):
        # 使用c10d后端运行test_run_distributed_sum_heterogeneous方法的测试
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_distributed_sum_heterogeneous
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_distributed_sum_heterogeneous_etcd(self):
        # 使用etcd后端运行test_run_distributed_sum_heterogeneous方法的测试
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_distributed_sum_heterogeneous
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_distributed_sum_heterogeneous_etcd_v2(self):
        # 使用etcd-v2后端运行test_run_distributed_sum_heterogeneous方法的测试
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_distributed_sum_heterogeneous
        )

    def run_sad_function(self):
        """
        checks error propagation logic
        """
        # 设置环境变量TORCHELASTIC_ERROR_FILE为error.json文件路径
        replyfile = os.path.join(self._test_dir, "error.json")
        # 使用mock.patch.dict将TORCHELASTIC_ERROR_FILE设置为replyfile路径，运行self.run_agent方法
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": replyfile}):
            with self.assertRaises(ChildFailedError) as cm:
                self.run_agent(Conf(entrypoint=_sad_function, local_world_size=2))

            # 获取异常中的第一个失败项的排名和失败信息
            rank, failure = cm.exception.get_first_failure()
            failure_data = failure.error_file_data["message"]
            # 打开replyfile，加载其中的JSON数据
            with open(replyfile) as fp:
                data = json.load(fp)["message"]

                # 断言运行了两次，两次都失败，第一个失败的排名为0或1
                self.assertTrue(rank in {0, 1})
                self.assertTrue(failure.local_rank in {0, 1})
                self.assertEqual(1, failure.exitcode)
                # 断言数据信息一致
                self.assertEqual(data["message"], failure_data["message"])
                self.assertEqual(int(data["extraInfo"]["timestamp"]), failure.timestamp)
    # 根据条件决定是否在Sandcastle环境中跳过测试，条件为TEST_WITH_DEV_DBG_ASAN，因为它与dev/dbg asan不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_sad_function_c10d(self):
        # 使用c10d后端运行run_sad_function测试
        self.run_test_with_backend(backend="c10d", test_to_run=self.run_sad_function)
    
    # 根据条件决定是否在Sandcastle环境中跳过测试，条件为TEST_WITH_DEV_DBG_ASAN，因为它与dev/dbg asan不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_sad_function_etcd(self):
        # 使用etcd后端运行run_sad_function测试
        self.run_test_with_backend(backend="etcd", test_to_run=self.run_sad_function)
    
    # 根据条件决定是否在Sandcastle环境中跳过测试，条件为TEST_WITH_DEV_DBG_ASAN，因为它与dev/dbg asan不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_sad_function_etcd_v2(self):
        # 使用etcd-v2后端运行run_sad_function测试
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.run_sad_function)
    
    def run_bipolar_function(self):
        """
        检查代理失败处理逻辑
        """
        # 配置节点参数，设置入口点为_bipolar_function，本地节点数为4
        node_conf = Conf(entrypoint=_bipolar_function, local_world_size=4)
        # 获取工作节点规范，最大重启次数为2
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        # 获取代理对象，使用给定的节点配置
        agent = self.get_agent(spec, node_config=node_conf)
        # 运行代理的方法
        run_result = agent.run()
        # 断言运行结果为失败状态
        self.assertTrue(run_result.is_failed())
        # 断言代理的剩余重启次数为0
        self.assertEqual(0, agent._remaining_restarts)
        # 断言工作组的状态为FAILED
        self.assertEqual(WorkerState.FAILED, agent.get_worker_group().state)
        # 断言代理的总执行时间大于0
        self.assertTrue(agent._total_execution_time > 0)
    
    # 根据条件决定是否在Sandcastle环境中跳过测试，条件为TEST_WITH_DEV_DBG_ASAN，因为它与dev/dbg asan不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_bipolar_function_c10d(self):
        # 使用c10d后端运行run_bipolar_function测试
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_bipolar_function
        )
    
    # 根据条件决定是否在Sandcastle环境中跳过测试，条件为TEST_WITH_DEV_DBG_ASAN，因为它与dev/dbg asan不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_bipolar_function_etcd(self):
        # 使用etcd后端运行run_bipolar_function测试
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_bipolar_function
        )
    
    # 根据条件决定是否在Sandcastle环境中跳过测试，条件为TEST_WITH_DEV_DBG_ASAN，因为它与dev/dbg asan不兼容
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_bipolar_function_etcd_v2(self):
        # 使用etcd-v2后端运行run_bipolar_function测试
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_bipolar_function
        )
    # 定义一个方法，用于测试异构节点配置的正确性
    def correct_rank_assignment_heterogeneous():
        # 定义不同节点的配置信息列表
        node_configs = [
            Conf(role="master", entrypoint=_get_role_info, local_world_size=8),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=1),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=2),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=3),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=5),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=2),
        ]
        # 运行作业并获取结果
        results = self.run_job(node_configs)
        # 打印异构作业的结果
        print(f"heterogeneous job result: {results}")
        # 断言确保结果中“master”角色的数量为1
        self.assertEqual(1, len(results["master"]))
        # 断言确保结果中“trainer”角色的数量为4
        self.assertEqual(4, len(results["trainer"]))
        # 断言确保结果中“ps”角色的数量为2
        self.assertEqual(2, len(results["ps"]))
        # 断言确保结果中角色的数量与期望的角色世界大小一致性
        self.assert_rank_consistency(
            results,
            expected_role_world_sizes={
                "master": 8,
                "trainer": 1 + 2 + 3 + 4,
                "ps": 5 + 2,
            },
        )

    # 如果运行环境是开发/调试 ASAN 或 TSAN，则跳过该测试
    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    # 定义一个方法，用于测试使用 etcd 后端的异构节点配置的正确性
    def test_correct_rank_assignment_heterogeneous_etcd():
        # 运行带有 etcd 后端的测试，使用 correct_rank_assignment_heterogeneous 方法
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.correct_rank_assignment_heterogeneous
        )

    # 如果运行环境是开发/调试 ASAN 或 TSAN，则跳过该测试
    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    # 定义一个方法，用于测试使用 etcd-v2 后端的异构节点配置的正确性
    def test_correct_rank_assignment_heterogeneous_etcd_v2():
        # 运行带有 etcd-v2 后端的测试，使用 correct_rank_assignment_heterogeneous 方法
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.correct_rank_assignment_heterogeneous
        )

    # 定义一个方法，用于测试同构节点配置的正确性
    def correct_rank_assignment_homogeneous():
        # 定义不同节点的配置信息列表
        node_configs = [
            Conf(role="master", entrypoint=_get_role_info, local_world_size=1),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=3),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=3),
        ]
        # 运行作业并获取结果
        results = self.run_job(node_configs)
        # 打印同构作业的结果
        print(f"homogeneous job result: {results}")
        # 断言确保结果中“master”角色的数量为1
        self.assertEqual(1, len(results["master"]))
        # 断言确保结果中“trainer”角色的数量为4
        self.assertEqual(4, len(results["trainer"]))
        # 断言确保结果中“ps”角色的数量为2
        self.assertEqual(2, len(results["ps"]))
        # 断言确保结果中角色的数量与期望的角色世界大小一致性
        self.assert_rank_consistency(
            results,
            expected_role_world_sizes={"master": 1, "trainer": 4 * 4, "ps": 3 * 2},
        )
    def test_correct_rank_assignment_homogeneous_etcd(self):
        # 调用 run_test_with_backend 方法，使用 etcd 后端运行 correct_rank_assignment_homogeneous 测试
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.correct_rank_assignment_homogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_homogeneous_etcd_v2(self):
        # 如果 TEST_WITH_DEV_DBG_ASAN 或 TEST_WITH_TSAN 为真，则跳过测试，因为与开发/调试 asan 或 tsan 不兼容
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.correct_rank_assignment_homogeneous
        )

    def assert_rank_consistency(
        self,
        run_results: Dict[str, List[RunResult]],
        expected_role_world_sizes: Dict[str, int],
    ):
        """
        Asserts that ranks are consecutive w.r.t role_rank. If local world sizes are 4:
        role_rank_0 -> ranks: 0,1,2,3
        role_rank_1 -> ranks: 4,5,6,7
        ... etc ...
        """
        # 初始化空列表用于存储全局 ranks
        global_ranks: List[int] = []
        # 用于存储每个角色的角色 ranks 的字典
        role_ranks: Dict[str, List[int]] = {}
        # 用于存储分组 ranks 的字典，每个分组包含 (rank, role_rank) 元组的列表
        grouped_ranks: Dict[int, List[Tuple[int, int]]] = {}

        # 计算全局世界大小，即所有角色世界大小之和
        expected_world_size = sum(expected_role_world_sizes.values())
        # 遍历运行结果字典，其中每个角色对应一个结果列表
        for role, results in run_results.items():
            for result in results:
                res = result.return_values
                # 遍历返回值中的角色信息
                for role_info in res.values():
                    rank = role_info.rank
                    role_rank = role_info.role_rank
                    group_rank = role_info.group_rank
                    role_world_size = role_info.role_world_size
                    world_size = role_info.world_size

                    # 断言全局世界大小与角色世界大小相等
                    self.assertEqual(expected_world_size, world_size)
                    # 断言期望的角色世界大小与角色信息中的角色世界大小相等
                    self.assertEqual(expected_role_world_sizes[role], role_world_size)
                    # 将 (rank, role_rank) 元组添加到分组 ranks 字典中对应的分组中
                    grouped_ranks.setdefault(group_rank, []).append((rank, role_rank))
                    # 将角色 ranks 添加到角色 ranks 字典中对应角色的列表中
                    role_ranks.setdefault(role, []).append(role_rank)
                    # 将 rank 添加到全局 ranks 列表中
                    global_ranks.append(rank)

        # 对全局 ranks 列表进行排序
        global_ranks = sorted(global_ranks)
        # 断言全局 ranks 应为连续整数序列
        self.assertEqual(list(range(expected_world_size)), global_ranks)
        # 对每个角色及其期望的角色世界大小进行断言，角色 ranks 应为连续整数序列
        for role, expected_role_world_size in expected_role_world_sizes.items():
            self.assertEqual(
                list(range(expected_role_world_size)), sorted(role_ranks[role])
            )
        # 确保每个 agent 将连续的 ranks 分配给 workers
        # 第一个参数是全局 ranks，第二个参数是角色 ranks
        for ranks_lst in grouped_ranks.values():
            self.assert_ranks_sequential(ranks_lst, 0)
            self.assert_ranks_sequential(ranks_lst, 1)

    def assert_ranks_sequential(self, ranks_pairs, rank_idx):
        # 对 ranks_pairs 中的 ranks 按 rank_idx 进行排序
        ranks = sorted(rank_pair[rank_idx] for rank_pair in ranks_pairs)
        start_rank, end_rank = ranks[0], ranks[-1]
        # 断言 ranks 应为连续整数序列
        self.assertEqual(list(range(start_rank, end_rank + 1)), ranks)
    def double_agent_fault_tolerance(self):
        """
        启动 ``nnodes`` 个代理，杀死并重新启动奇数编号的代理，验证容错性能
        """
        nnodes = 2
        wait = 2
        # 定义节点配置
        node_conf = Conf(entrypoint=_dist_sum, args=(wait,), local_world_size=2)
        # 创建一个进程通信队列
        agent_results = mp.Queue()
        # 定义代理参数
        agent_args = {
            "conf": node_conf,
            "agent_results": agent_results,
            "min_nodes": nnodes,
            "max_nodes": nnodes,
            "max_restarts": 2,
        }

        procs = []
        # 启动 nnodes 个代理进程
        for _ in range(nnodes):
            p = mp.Process(
                target=self.run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # 重新启动奇数编号的代理
        for i in range(nnodes):
            if i % 2 != 0:
                # 杀死当前进程
                procs[i].kill()
                # 创建新的代理进程
                p = mp.Process(
                    target=self.run_agent,
                    kwargs=agent_args,
                )
                procs[i] = p
                p.start()

        # 等待所有代理进程结束并验证退出码为 0
        for i in range(nnodes):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_fault_tolerance_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.double_agent_fault_tolerance
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_fault_tolerance_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.double_agent_fault_tolerance
        )
    def no_exit_barrier_on_failure(self):
        """
        start ``nnodes`` agents, kill and restart odd ones, validate fault-tolerance works
        """
        # 定义节点数量
        nnodes = 2
        # 等待时间设定为20秒
        wait = 20
        # 配置节点的参数，包括入口点函数和参数
        node_conf = Conf(
            entrypoint=_bipolar_sleep_function, args=(wait,), local_world_size=2
        )
        # 创建用于存储代理结果的队列
        agent_results = mp.Queue()
        # 监控间隔设定为0.5秒
        monitor_interval_s = 0.5
        # 定义代理的参数字典
        agent_args = {
            "conf": node_conf,
            "agent_results": agent_results,
            "min_nodes": nnodes,
            "max_nodes": nnodes,
            "max_restarts": 0,
            "exit_barrier_timeout": 300,
            "monitor_interval": monitor_interval_s,
        }

        # 初始化进程列表
        procs = []
        # 启动指定数量的代理进程
        for _ in range(nnodes):
            p = mp.Process(
                target=self.run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # 等待所有进程结束，并检查其退出码
        exit_interval_between_agents = 0
        for i in range(nnodes):
            p = procs[i]
            p.join()
            # 断言每个进程的退出码不为0，验证故障容错性能
            self.assertNotEqual(0, p.exitcode)
            # 计算进程结束之间的时间间隔
            exit_interval_between_agents = (
                time.monotonic() - exit_interval_between_agents
            )

        # 验证所有进程的结束时间接近
        # 使用略高于2 * monitor_interval_s (0.01秒)的超时时间，减少波动性
        self.assertGreater(
            2 * monitor_interval_s,
            exit_interval_between_agents,
            "Agents are not cleaned up until 2 * monitor_interval",
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_no_exit_barrier_on_failure(self):
        # 运行后端测试，使用c10d后端，并执行no_exit_barrier_on_failure方法
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.no_exit_barrier_on_failure
        )
    def double_agent_elastic(self):
        """
        start ``nnodes`` agents, kill odd ones (do not restart), validate
        elasticity (scale-down) works. (scale-up covered in fault_tolerance test)
        """
        # 设置最小节点数为1，最大节点数为2，等待时间为2秒
        min_nodes = 1
        max_nodes = 2
        wait = 2
        # 配置节点的运行参数，包括入口点函数和参数，本地世界大小为2
        node_conf = Conf(entrypoint=_dist_sum, args=(wait,), local_world_size=2)
        # 创建用于存放代理进程结果的队列
        agent_results = mp.Queue()
        # 设定代理进程的参数
        agent_args = {
            "conf": node_conf,
            "agent_results": agent_results,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "max_restarts": 2,
        }

        # 创建代理进程列表
        procs = []
        for _ in range(max_nodes):
            # 启动代理进程，每个进程使用相同的参数agent_args
            p = mp.Process(
                target=self.run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # 杀死奇数索引位置的代理进程
        for i in range(max_nodes):
            if i % 2 != 0:
                procs[i].kill()

        # 等待所有代理进程结束
        for i in range(max_nodes):
            p = procs[i]
            p.join()
            # 验证偶数索引位置的进程退出码为0
            if i % 2 == 0:
                self.assertEqual(0, p.exitcode)
            # 验证奇数索引位置的进程退出码为负的SIGKILL信号值
            else:
                self.assertEqual(-signal.SIGKILL, p.exitcode)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_c10d(self):
        # 使用c10d后端运行test_double_agent_elastic测试方法
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.double_agent_elastic
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_etcd(self):
        # 使用etcd后端运行test_double_agent_elastic测试方法
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.double_agent_elastic
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_etcd_v2(self):
        # 使用etcd-v2后端运行test_double_agent_elastic测试方法
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.double_agent_elastic
        )
    def torch_rpc(self):
        """
        Simple torch rpc example with torchelastic.
        Creates two agents (to simulate two node job),
        each agent runs a single worker. worker0 calls an rpc_sync on
        worker1.
        """
        # 定义消息内容
        msg = "hello world"
        # 定义节点配置列表
        node_configs = [
            Conf(
                role="master",
                entrypoint=rpc_master,
                args=(msg,),
                local_world_size=1,
                tee=Std.ALL,
            ),
            Conf(
                role="worker",
                entrypoint=rpc_worker,
                args=(),
                local_world_size=1,
                tee=Std.ALL,
            ),
        ]

        # 运行作业，并获取结果
        results = self.run_job(node_configs)
        # 获取主节点返回的值
        master_retvals = results["master"][0].return_values
        # 由于全局排名不稳定，将主节点返回的值作为集合进行比较
        self.assertEqual([f"{msg} from worker"], list(master_retvals.values()))

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_c10d(self):
        # 运行指定后端的测试（这里是c10d），测试函数为torch_rpc
        self.run_test_with_backend(backend="c10d", test_to_run=self.torch_rpc)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_etcd(self):
        # 运行指定后端的测试（这里是etcd），测试函数为torch_rpc
        self.run_test_with_backend(backend="etcd", test_to_run=self.torch_rpc)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_etcd_v2(self):
        # 运行指定后端的测试（这里是etcd-v2），测试函数为torch_rpc
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.torch_rpc)

    def workers_drift_success(self):
        """
        two agents (one worker each) finishes within ``sec`` seconds of each other,
        exit barrier timeout set to ``sec * 2 * 2``.
        """
        # 设置时间间隔
        sec = 1
        # 定义节点配置列表
        node_configs = [
            Conf(role="zzz", entrypoint=_sleep, args=(0 * sec,), local_world_size=1),
            Conf(role="zzz", entrypoint=_sleep, args=(2 * sec,), local_world_size=1),
        ]
        # 运行作业，并设置退出屏障超时时间
        results = self.run_job(node_configs, exit_barrier_timeout=2 * 2 * sec)
        # 遍历结果列表
        for i in range(2):
            run_results = results["zzz"][i]
            # 断言作业运行结果不是失败的
            self.assertFalse(run_results.is_failed())
            # 遍历每个运行结果的返回值
            for rank, output in run_results.return_values.items():
                # 断言返回值与其排名相同（由_sleep()函数返回）
                self.assertEqual(rank, output)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_success_etcd(self):
        # 运行指定后端的测试（这里是etcd），测试函数为workers_drift_success
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.workers_drift_success
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_success_etcd_v2(self):
        # 使用特定后端（etcd-v2）运行 workers_drift_success 测试
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.workers_drift_success
        )

    def workers_drift_fail(self):
        """
        two agents (one worker each) finishes within ``4 x sec`` seconds of each other,
        exit barrier timeout set to 0. Exit barriers should NOT fail the job.
        """
        sec = 1
        node_configs = [
            # 第一个 agent 的配置，角色为 "zzz"，入口点是 _sleep 函数，参数为 0 秒
            Conf(role="zzz", entrypoint=_sleep, args=(0 * sec,), local_world_size=1),
            # 第二个 agent 的配置，角色为 "zzz"，入口点是 _sleep 函数，参数为 4 秒
            Conf(role="zzz", entrypoint=_sleep, args=(4 * sec,), local_world_size=1),
        ]
        # 运行作业并设置退出屏障超时为 0
        results = self.run_job(node_configs, exit_barrier_timeout=0)
        for i in range(2):
            # 检查每个 agent 的运行结果，应该不是失败的
            run_results = results["zzz"][i]
            self.assertFalse(run_results.is_failed())
            for rank, output in run_results.return_values.items():
                # _sleep() 函数返回其自身的排名
                self.assertEqual(rank, output)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_fail_etcd(self):
        # 使用特定后端（etcd）运行 workers_drift_fail 测试
        self.run_test_with_backend(backend="etcd", test_to_run=self.workers_drift_fail)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_fail_etcd_v2(self):
        # 使用特定后端（etcd-v2）运行 workers_drift_fail 测试
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.workers_drift_fail
        )

    @patch("torch.distributed.elastic.utils.store.barrier")
    def barrier_failed(self, barrier_mock):
        """
        Failure during the barrier should NOT fail the job.
        """
        # 设置 barrier_mock 的 side_effect 为 RuntimeError("test error")
        barrier_mock.side_effect = RuntimeError("test error")
        # 运行代理，并期望结果不是失败的
        res = self.run_agent(Conf(entrypoint=_happy_function, local_world_size=1))
        self.assertFalse(res.is_failed())
        # 确保 barrier_mock 被调用了一次
        barrier_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_barrier_failed_c10d(self):
        # 使用特定后端（c10d）运行 barrier_failed 测试
        self.run_test_with_backend(backend="c10d", test_to_run=self.barrier_failed)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_barrier_failed_etcd(self):
        # 使用特定后端（etcd）运行 barrier_failed 测试
        self.run_test_with_backend(backend="etcd", test_to_run=self.barrier_failed)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_barrier_failed_etcd_v2(self):
        # 使用特定后端（etcd-v2）运行 barrier_failed 测试
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.barrier_failed)

    @patch("torch.distributed.elastic.agent.server.local_elastic_agent.start_processes")
    # 当 shutdown_called 方法被调用时，模拟并返回一个进程上下文的 Mock 对象
    def shutdown_called(self, start_processes_mock):
        # 创建进程上下文的 Mock 对象，并设定其中的进程 ID 为 {0: 0}
        pcontext_mock = Mock()
        pcontext_mock.pids.return_value = {0: 0}
        # 设定 start_processes_mock 方法的返回值为 pcontext_mock
        start_processes_mock.return_value = pcontext_mock
        # 创建一个 Conf 对象，其中包含入口点为 _happy_function，本地世界大小为 1
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)
        # 使用 self.get_worker_spec 方法获取工作节点的规格
        spec = self.get_worker_spec(node_conf, max_restarts=0)
        # 使用 self.get_agent 方法创建一个代理对象，使用指定的规格和节点配置
        agent = self.get_agent(spec, node_config=node_conf)
        # 使用 patch.object 方法创建 agent 的 _monitor_workers 方法的 Mock 对象
        with patch.object(agent, "_monitor_workers") as monitor_mock:
            # 设定 monitor_mock 方法的返回值为一个成功的运行结果对象
            monitor_mock.return_value = RunResult(
                state=WorkerState.SUCCEEDED, return_values={0: 0}
            )
            # 运行 agent 的 run 方法，参数为 "worker"
            agent.run("worker")
        # 断言 pcontext_mock.close 方法仅被调用一次
        pcontext_mock.close.assert_called_once()

    # 如果 TEST_WITH_DEV_DBG_ASAN 为真，则在 Sandcastle 中跳过执行，但在其他情况下仍然执行
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 c10d 后端运行 shutdown_called 方法的测试
    def test_shutdown_called_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.shutdown_called)

    # 如果 TEST_WITH_DEV_DBG_ASAN 为真，则在 Sandcastle 中跳过执行，但在其他情况下仍然执行
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 etcd 后端运行 shutdown_called 方法的测试
    def test_shutdown_called_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.shutdown_called)

    # 如果 TEST_WITH_DEV_DBG_ASAN 为真，则在 Sandcastle 中跳过执行，但在其他情况下仍然执行
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 etcd-v2 后端运行 shutdown_called 方法的测试
    def test_shutdown_called_etcd_v2(self):
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.shutdown_called)

    # 运行 fail_rank_one_once 方法以测试在故障后重启的情况
    def fail_rank_one_once(self):
        # 运行代理对象的 run_agent 方法，使用指定的 Conf 对象和最大重启次数
        res = self.run_agent(
            Conf(entrypoint=dummy_compute_simulate_rank_failure, local_world_size=2),
            max_restarts=3,
        )
        # 断言结果不是失败状态
        self.assertFalse(res.is_failed())
        # 对于每个返回值，断言其类型为 torch.Tensor
        for return_value in res.return_values.values():
            self.assertIsInstance(return_value, torch.Tensor)
            # 断言返回值的形状为 (100, 100)
            self.assertEqual((100, 100), return_value.shape)

    # 如果 TEST_WITH_DEV_DBG_ASAN 为真，则在 Sandcastle 中跳过执行，但在其他情况下仍然执行
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 使用 c10d 后端运行 fail_rank_one_once 方法的测试
    def test_rank_restart_after_failure(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.fail_rank_one_once)
```