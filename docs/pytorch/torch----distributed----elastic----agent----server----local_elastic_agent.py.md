# `.\pytorch\torch\distributed\elastic\agent\server\local_elastic_agent.py`

```py
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入所需的标准库和第三方库
import json  # 导入处理 JSON 数据的模块
import os  # 导入处理操作系统相关功能的模块
import signal  # 导入处理信号的模块
import socket  # 导入实现网络通信的模块
import time  # 导入处理时间的模块
import uuid  # 导入生成唯一标识符的模块
from string import Template  # 导入字符串模板的模块
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING  # 导入类型注解相关的功能

# 导入 Torch Elastic 的相关模块和类
import torch.distributed.elastic.timer as timer  # 导入 Torch Elastic 的定时器模块
from torch.distributed.elastic import events  # 导入 Torch Elastic 的事件模块
from torch.distributed.elastic.agent.server.api import (  # 导入 Torch Elastic 的代理服务器 API 相关类
    RunResult,
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.health_check_server import (  # 导入 Torch Elastic 的健康检查服务器相关类
    create_healthcheck_server,
    HealthCheckServer,
)
from torch.distributed.elastic.metrics.api import prof  # 导入 Torch Elastic 的性能指标 API
from torch.distributed.elastic.multiprocessing import (  # 导入 Torch Elastic 的多进程相关功能
    LogsSpecs,
    PContext,
    start_processes,
)
from torch.distributed.elastic.utils import macros  # 导入 Torch Elastic 的宏工具
from torch.distributed.elastic.utils.logging import get_logger  # 导入获取日志记录器的函数

# 如果支持类型检查，引入类型检查相关的导入
if TYPE_CHECKING:
    from torch.distributed.elastic.events.api import EventMetadataValue  # 导入事件元数据值类型注解

# 获取当前模块的日志记录器
logger = get_logger(__name__)

# 定义公开的符号列表
__all__ = [
    "LocalElasticAgent",
    "TORCHELASTIC_ENABLE_FILE_TIMER",
    "TORCHELASTIC_TIMER_FILE",
    "TORCHELASTIC_HEALTH_CHECK_PORT",
]

# 定义 Torch Elastic 配置项常量
TORCHELASTIC_ENABLE_FILE_TIMER = "TORCHELASTIC_ENABLE_FILE_TIMER"
TORCHELASTIC_HEALTH_CHECK_PORT = "TORCHELASTIC_HEALTH_CHECK_PORT"
TORCHELASTIC_TIMER_FILE = "TORCHELASTIC_TIMER_FILE"

# 定义本地弹性代理类，继承自 SimpleElasticAgent 类
class LocalElasticAgent(SimpleElasticAgent):
    """An implementation of :py:class:`torchelastic.agent.server.ElasticAgent` that handles host-local workers.

    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.

    The worker function and argument passed to the worker function must be
    python multiprocessing compatible. To pass multiprocessing data structures
    to the workers you may create the data structure in the same multiprocessing
    context as the specified ``start_method`` and pass it as a function argument.

    The ``exit_barrier_timeout`` specifies the amount of time (in seconds) to wait
    for other agents to finish. This acts as a safety net to handle cases where
    workers finish at different times, to prevent agents from viewing workers
    that finished early as a scale-down event. It is strongly advised that the
    user code deal with ensuring that workers are terminated in a synchronous
    manner rather than relying on the exit_barrier_timeout.

    A named pipe based watchdog can be enabled in ```LocalElasticAgent```py if an
    """
    # 定义一个类 LocalElasticAgent，代表本地弹性代理
    def __init__(
        self,
        spec: WorkerSpec,  # 参数 spec: WorkerSpec 类型，描述工作进程的规范
        logs_specs: LogsSpecs,  # 参数 logs_specs: LogsSpecs 类型，描述日志规范
        start_method="spawn",  # 参数 start_method: str 类型，默认为 "spawn"，指定启动方法
        exit_barrier_timeout: float = 300,  # 参数 exit_barrier_timeout: float 类型，默认为 300 秒，退出屏障超时时间
        log_line_prefix_template: Optional[str] = None,  # 参数 log_line_prefix_template: Optional[str] 类型，可选，日志行前缀模板
    ):
        # 调用父类构造函数，传递特定参数和退出屏障超时时间
        super().__init__(spec, exit_barrier_timeout)
        # 设置起始方法
        self._start_method = start_method
        # 初始化 PContext 为可选的空值
        self._pcontext: Optional[PContext] = None
        # 获取规范中的调度器处理器
        self._rdzv_handler = spec.rdzv_handler
        # 设置日志行前缀模板
        self._log_line_prefix_template = log_line_prefix_template
        # 初始化工作进程看门狗为可选的空值
        self._worker_watchdog: Optional[timer.FileTimerServer] = None
        # 设置日志规范
        self._logs_specs = logs_specs
        # 初始化健康检查服务器为可选的空值
        self._health_check_server: Optional[HealthCheckServer] = None

    def _setup_local_watchdog(self, envs: Dict[int, Dict[str, str]]) -> None:
        # 获取环境变量 TORCHELASTIC_ENABLE_FILE_TIMER 的值
        enable_watchdog_env_name = TORCHELASTIC_ENABLE_FILE_TIMER
        watchdog_enabled = os.getenv(enable_watchdog_env_name)
        # 获取环境变量 TORCHELASTIC_TIMER_FILE 的路径
        watchdog_file_env_name = TORCHELASTIC_TIMER_FILE
        watchdog_file_path = os.getenv(watchdog_file_env_name)
        # 如果看门狗启用且为字符串 "1"
        if watchdog_enabled is not None and str(watchdog_enabled) == "1":
            # 如果未指定看门狗文件路径，则创建一个临时路径
            if watchdog_file_path is None:
                watchdog_file_path = "/tmp/watchdog_timer_" + str(uuid.uuid4())
            # 记录启动 FileTimerServer 的日志信息
            logger.info("Starting a FileTimerServer with %s ...", watchdog_file_path)
            # 如果环境变量列表为空，则警告并使用空的 run_id 启动 FileTimerServer
            if not envs:
                logger.warning(
                    "Empty envs variables, using empty run_id for FileTimerServer"
                )
                run_id = ""
            else:
                # 否则从环境变量中获取 TORCHELASTIC_RUN_ID
                run_id = envs[0]["TORCHELASTIC_RUN_ID"]
            # 初始化 FileTimerServer 对象，并启动它
            self._worker_watchdog = timer.FileTimerServer(
                file_path=watchdog_file_path,
                run_id=run_id,
                max_interval=0.1,
                daemon=True,
                log_event=self._log_watchdog_event,
            )
            self._worker_watchdog.start()
            # 记录 FileTimerServer 启动成功的日志信息
            logger.info("FileTimerServer started")
        else:
            # 如果环境变量 TORCHELASTIC_ENABLE_FILE_TIMER 未找到，则记录日志信息
            logger.info(
                "Environment variable '%s' not found. Do not start FileTimerServer.",
                enable_watchdog_env_name,
            )
        # 将看门狗文件路径环境变量传播给工作进程
        if watchdog_file_path is not None:
            for worker_env in envs.values():
                worker_env[watchdog_file_env_name] = watchdog_file_path

    @staticmethod
    def _get_current_time_secs() -> int:
        # 返回当前时间的整数秒数
        return int(time.time())
    # 设置健康检查功能，根据环境变量 TORCHELASTIC_HEALTH_CHECK_PORT 获取健康检查端口号
    healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT
    # 获取环境变量中健康检查端口号的数值表示
    healthcheck_port = os.getenv(healthcheck_port_env_name)
    # 如果成功获取到健康检查端口号
    if healthcheck_port is not None:
        # 记录找到的健康检查端口号及其环境变量名
        logger.info(
            "Found healthcheck port %s: %s",
            healthcheck_port_env_name,
            healthcheck_port,
        )
        # 如果 FileTimerServer 不存在
        if self._worker_watchdog is None:
            # 记录使用当前时间作为虚拟回调，因为 FileTimerServer 不存在
            logger.info(
                "FileTimerServer doesn't exist, using current time as dummy callback"
            )
            # 使用 LocalElasticAgent 类的 _get_current_time_secs 方法作为存活回调
            alive_callback = LocalElasticAgent._get_current_time_secs
        else:
            # 使用 _worker_watchdog 的 get_last_progress_time 方法作为存活回调
            alive_callback = self._worker_watchdog.get_last_progress_time

        # 创建健康检查服务器，并启动
        self._health_check_server = create_healthcheck_server(
            alive_callback=alive_callback,
            port=int(healthcheck_port),
            timeout=60,
        )
        self._health_check_server.start()
    else:
        # 如果未找到健康检查端口号的环境变量，记录此信息，不启动健康检查
        logger.info(
            "Environment variable '%s' not found. Do not start health check.",
            healthcheck_port_env_name,
        )

# 获取完全限定的主机名
def _get_fq_hostname(self) -> str:
    return socket.getfqdn(socket.gethostname())

# 记录看门狗事件，包括一些请求的详细信息
def _log_watchdog_event(
    self,
    name: str,
    request: Optional[timer.FileTimerRequest],
) -> None:
    # 获取当前工作组的规格
    wg = self._worker_group
    spec = wg.spec
    # 创建包含看门狗事件的元数据字典
    md = {"watchdog_event": name}
    # 如果有请求对象，添加更多的详细信息到元数据字典中
    if request is not None:
        md["worker_pid"] = str(request.worker_pid)
        md["scope_id"] = request.scope_id
        md["expiration_time"] = str(request.expiration_time)
        md["signal"] = str(request.signal)
    # 将元数据字典转换成 JSON 字符串
    md_str = json.dumps(md)
    # 设置事件状态为 RUNNING
    state = "RUNNING"
    # 创建事件的元数据字典，包含运行 ID、组内排名、角色、主机名等信息
    metadata: Dict[str, EventMetadataValue] = {
        "run_id": spec.rdzv_handler.get_run_id(),
        "global_rank": None,
        "group_rank": wg.group_rank,
        "worker_id": None,
        "role": spec.role,
        "hostname": self._get_fq_hostname(),
        "state": state,
        "total_run_time": self._total_execution_time,
        "rdzv_backend": spec.rdzv_handler.get_backend(),
        "raw_error": None,
        "metadata": md_str,
        "agent_restarts": spec.max_restarts - self._remaining_restarts,
    }
    # 创建事件对象，并记录该事件
    # 注意：事件的 'metadata' 字段稍后会被转换成 TorchelasticStatusLogEntry。
    #       事件的 'name' 字段在 TorchelasticStatusLogEntry 中不会被使用。
    event = events.Event(
        name=name, source=events.EventSource.AGENT, metadata=metadata
    )
    events.record(event)

# 使用性能分析装饰器 `prof` 来标记方法 `_stop_workers`，这可能与 Pyre 类型推断有关
@prof
def _stop_workers(
    self, worker_group: WorkerGroup, is_restart: bool = False
) -> None:
    # 调用 _shutdown 方法来停止工作进程
    self._shutdown(is_restart=is_restart)
    # 应用装饰器 `prof` 到当前函数或方法
    @prof
    # 启动工作进程的方法，返回一个字典，包含进程ID到任意类型数据的映射关系
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        # 获取工作组的规格信息
        spec = worker_group.spec
        # 获取工作组的存储信息
        store = worker_group.store
        # 断言存储信息不为空
        assert store is not None
        # 计算重启次数，为最大重启次数减去已经使用的重启次数
        restart_count = spec.max_restarts - self._remaining_restarts

        # 获取是否使用代理存储的标志
        use_agent_store: bool = spec.rdzv_handler.use_agent_store
        # 记录是否使用代理存储的信息到日志中
        logger.info("use_agent_store: %s", use_agent_store)

        # 准备存储参数的字典
        args: Dict[int, Tuple] = {}
        # 准备环境变量的字典
        envs: Dict[int, Dict[str, str]] = {}
        # 如果有日志行前缀模板，则准备日志行前缀的字典，否则为None
        log_line_prefixes: Optional[Dict[int, str]] = (
            {} if self._log_line_prefix_template else None
        )

        # 遍历工作组中的每个工作进程
        for worker in worker_group.workers:
            # 获取本地进程的本地排名
            local_rank = worker.local_rank
            # 准备本地进程的环境变量
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": worker_group.master_addr,
                "MASTER_PORT": str(worker_group.master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING", str(1)
                ),
            }
            # 如果环境中存在OMP_NUM_THREADS，则添加到工作进程的环境变量中
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            # 如果有日志行前缀模板，则根据模板创建日志行前缀，并存储在对应本地排名的字典中
            if self._log_line_prefix_template:
                log_line_prefix = Template(
                    self._log_line_prefix_template
                ).safe_substitute(
                    role_name=spec.role,
                    rank=worker.global_rank,
                    local_rank=local_rank,
                )
                log_line_prefixes[local_rank] = log_line_prefix

            # 将本地进程的环境变量加入envs字典
            envs[local_rank] = worker_env

            # 复制工作进程的参数列表，进行宏替换，然后存储在args字典中对应本地排名的位置
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        # 设置本地监控
        self._setup_local_watchdog(envs=envs)
        # 设置健康检查
        self._setup_healthcheck()

        # 断言入口点不为空
        assert spec.entrypoint is not None
        # 断言日志规范不为空
        assert self._logs_specs is not None

        # 启动进程，并返回进程上下文中的进程ID
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            logs_specs=self._logs_specs,
            log_line_prefixes=log_line_prefixes,
            start_method=self._start_method,
        )

        # 返回启动的进程ID字典
        return self._pcontext.pids()
    # 关闭各种监视和服务器，用于程序关闭操作
    def _shutdown(
        self, death_sig: signal.Signals = signal.SIGTERM, is_restart: bool = False
    ) -> None:
        # 如果存在 worker 监视对象，停止它
        if self._worker_watchdog is not None:
            self._worker_watchdog.stop()
            self._worker_watchdog = None
        # 如果存在健康检查服务器，停止它
        if self._health_check_server is not None:
            self._health_check_server.stop()
            self._health_check_server = None
        # 如果存在进程上下文对象，关闭它
        if self._pcontext:
            self._pcontext.close(death_sig)
        # 如果不是重启操作，并且存在 _rdzv_handler，执行其关闭操作
        if not is_restart and self._rdzv_handler:
            self._rdzv_handler.shutdown()

    # 使用 @prof 装饰器，用于监视工作进程的运行情况
    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        # 获取工作组的角色信息
        role = worker_group.spec.role
        # 获取工作组中所有 worker 的进程 ID 集合
        worker_pids = {w.id for w in worker_group.workers}
        # 断言进程上下文对象不为 None
        assert self._pcontext is not None
        # 获取当前进程上下文对象中所有进程的 ID 集合
        pc_pids = set(self._pcontext.pids().values())
        # 检查工作组中的 worker 进程 ID 是否与进程上下文中的匹配
        if worker_pids != pc_pids:
            # 记录错误日志，说明 worker 进程 ID 与进程上下文中的不匹配
            logger.error(
                "[%s] worker pids do not match process_context pids."
                " Expected: %s, actual: %s",
                role,
                worker_pids,
                pc_pids,
            )
            # 返回工作状态为 UNKNOWN 的运行结果对象
            return RunResult(state=WorkerState.UNKNOWN)

        # 等待当前进程上下文对象的运行结果
        result = self._pcontext.wait(0)
        if result:
            if result.is_failed():
                # 将本地 rank 的失败映射到全局 rank
                worker_failures = {}
                for local_rank, failure in result.failures.items():
                    worker = worker_group.workers[local_rank]
                    worker_failures[worker.global_rank] = failure
                # 返回工作状态为 FAILED 的运行结果对象，并包含失败的信息
                return RunResult(
                    state=WorkerState.FAILED,
                    failures=worker_failures,
                )
            else:
                # 将返回值队列复制到带有全局 rank 的映射中
                workers_ret_vals = {}
                for local_rank, ret_val in result.return_values.items():
                    worker = worker_group.workers[local_rank]
                    workers_ret_vals[worker.global_rank] = ret_val
                # 返回工作状态为 SUCCEEDED 的运行结果对象，并包含返回值的信息
                return RunResult(
                    state=WorkerState.SUCCEEDED,
                    return_values=workers_ret_vals,
                )
        else:
            # 如果没有运行结果，则返回工作状态为 HEALTHY 的运行结果对象
            return RunResult(state=WorkerState.HEALTHY)
```