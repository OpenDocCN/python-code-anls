# `.\pytorch\torch\distributed\launcher\api.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入系统模块
import sys
# 导入生成 UUID 的模块
import uuid
# 导入用于数据类的模块
from dataclasses import dataclass, field
# 导入类型提示模块
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 Torch Elastic 的模块
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic import events, metrics
from torch.distributed.elastic.agent.server.api import WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing import (
    DefaultLogsSpecs,
    LogsSpecs,
    SignalException,
)
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
from torch.distributed.elastic.utils.logging import get_logger

# 设置对外导出的符号
__all__ = ["LaunchConfig", "elastic_launch", "launch_agent"]

# 获取当前模块的日志记录器对象
logger = get_logger(__name__)


@dataclass
class LaunchConfig:
    """
    Creates a rendezvous config.
    """
    # 数据类字段：用于定义多节点训练的 Worker 规格
    worker_spec: Optional[WorkerSpec] = None
    # 数据类字段：指定默认的日志配置规格
    logs: LogsSpecs = field(default_factory=DefaultLogsSpecs)
    # 数据类字段：指定 Elastic Agent 的超时时间
    agent_timeout: Optional[int] = None
    # 数据类字段：定义应用程序 ID
    app_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # 数据类字段：定义会议参数对象
    rendezvous: Optional[RendezvousParameters] = None
    # 数据类字段：定义强制信号异常
    signal_exception: Optional[SignalException] = None
    # 数据类字段：定义事件处理器
    event_handler: Optional[Callable[[events.Event], None]] = None
    # 数据类字段：定义指标收集器
    metrics_handler: Optional[Callable[[metrics.Metric], None]] = None
    # 最小节点数：用户函数将在达到此节点数时启动。弹性代理确保用户函数仅在达到最小节点数时才开始运行。
    min_nodes: int
    # 最大节点数：用户函数将启动的最大节点数。
    max_nodes: int
    # 每个节点的处理器数量：每个节点上弹性代理将启动的工作进程数量，用于执行用户定义的函数。
    nproc_per_node: int
    # rdzv_backend：在会合中使用的后端（例如 zeus-adapter、etcd）。
    rdzv_backend: str
    # rdzv_endpoint：会合同步存储的终端点。
    rdzv_endpoint: str
    # rdzv_configs：指定会合特定配置的键值对。
    rdzv_configs: Dict[str, Any]
    # rdzv_timeout：会合的超时时间（旧参数，将在未来版本中移除）。建议通过 `rdzv_configs['timeout']` 设置超时时间。
    rdzv_timeout: int
    # run_id：作业的唯一运行 ID（如果未传递，则从运行环境中推断出唯一 ID）。
    run_id: str
    # role：工作节点的用户定义角色（默认为 "trainer"）。
    role: str
    # max_restarts：弹性代理在工作失败之前将进行的最大重启次数。
    max_restarts: int
    # monitor_interval：弹性代理用于监视工作进程的间隔（秒）。
    monitor_interval: float
    # start_method：弹性代理用于启动工作进程的方法（spawn、fork、forkserver）。
    start_method: str
    # metrics_cfg：初始化指标的配置。
    metrics_cfg: Dict[str, str]
    # local_addr：本地节点的地址（如果有）。如果未设置，则会执行本地机器的 FQDN 查找。
    local_addr: Optional[str]
    # 在对象初始化完成后执行的方法，用来设定默认的超时时间
    def __post_init__(self):
        # 默认超时时间为900秒
        default_timeout = 900
        # 如果用户提供了rdzv_timeout参数，则使用用户提供的超时时间
        if self.rdzv_timeout != -1:
            self.rdzv_configs["timeout"] = self.rdzv_timeout
        # 如果用户未提供rdzv_timeout参数，并且rdzv_configs中未定义timeout，则使用默认超时时间
        elif "timeout" not in self.rdzv_configs:
            self.rdzv_configs["timeout"] = default_timeout

        # 后处理操作，用于根据非torchrun API使用情况引入logs_specs参数
        # 如果logs_specs参数为None，则使用默认的日志规格DefaultLogsSpecs()
        if self.logs_specs is None:
            self.logs_specs = DefaultLogsSpecs()
class elastic_launch:
    """
    Launches an torchelastic agent on the container that invoked the entrypoint.

        1. Pass the ``entrypoint`` arguments as non ``kwargs`` (e.g. no named parameters)/
           ``entrypoint`` can be a function or a command.
        2. The return value is a map of each worker's output mapped
           by their respective global rank.

    Usage

    ::

    def worker_fn(foo):
        # ...

    def main():
        # entrypoint is a function.
        outputs = elastic_launch(LaunchConfig, worker_fn)(foo)
        # return rank 0's output
        return outputs[0]

        # entrypoint is a command and ``script.py`` is the python module.
        outputs = elastic_launch(LaunchConfig, "script.py")(args)
        outputs = elastic_launch(LaunchConfig, "python")("script.py")
    """

    def __init__(
        self,
        config: LaunchConfig,
        entrypoint: Union[Callable, str, None],
    ):
        """
        Initialize the elastic_launch object with provided configuration and entrypoint.

        Args:
            config (LaunchConfig): Configuration object specifying launch parameters.
            entrypoint (Union[Callable, str, None]): Function or command to be executed.
        """
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args):
        """
        Call method to launch the agent with given arguments.

        Args:
            *args: Arguments to be passed to the entrypoint.

        Returns:
            Dict[int, Any]: Dictionary mapping each worker's global rank to its output.
        """
        return launch_agent(self._config, self._entrypoint, list(args))


def _get_entrypoint_name(
    entrypoint: Union[Callable, str, None], args: List[Any]
) -> str:
    """
    Retrieve entrypoint name with the rule:
    1. If entrypoint is a function, use ``entrypoint.__qualname__``.
    2. If entrypoint is a string, check its value:
        2.1 if entrypoint equals to ``sys.executable`` (like "python"), use the first element from ``args``
            which does not start with hifen letter (for example, "-u" will be skipped).
        2.2 otherwise, use ``entrypoint`` value.
    3. Otherwise, return empty string.

    Args:
        entrypoint (Union[Callable, str, None]): Function or command to be checked.
        args (List[Any]): List of arguments associated with the entrypoint.

    Returns:
        str: Name of the entrypoint determined by the rules specified.
    """
    if isinstance(entrypoint, Callable):  # type: ignore[arg-type]
        return entrypoint.__name__  # type: ignore[union-attr]
    elif isinstance(entrypoint, str):
        if entrypoint == sys.executable:
            return next((arg for arg in args if arg[0] != "-"), "")
        else:
            return entrypoint
    else:
        return ""


def _get_addr_and_port(
    rdzv_parameters: RendezvousParameters,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Get the master address and port from the rendezvous parameters.

    Args:
        rdzv_parameters (RendezvousParameters): Parameters specifying rendezvous details.

    Returns:
        Tuple[Optional[str], Optional[int]]: Tuple containing master address and port.
            If backend is not "static", returns (None, None).
    """
    if rdzv_parameters.backend != "static":
        return (None, None)
    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError(
            "Endpoint is missing in endpoint. Try to add --master-addr and --master-port"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    if master_port == -1:
        raise ValueError(
            f"port is missing in endpoint: {endpoint}. Try to specify --master-port"
        )
    return (master_addr, master_port)


def launch_agent(
    config: LaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> Dict[int, Any]:
    """
    Launch the agent based on the provided configuration and entrypoint.

    Args:
        config (LaunchConfig): Configuration object specifying launch parameters.
        entrypoint (Union[Callable, str, None]): Function or command to be executed.
        args (List[Any]): List of arguments to be passed to the entrypoint.

    Returns:
        Dict[int, Any]: Dictionary mapping each worker's global rank to its output.
    """
    # 如果配置文件中没有指定运行 ID，则生成一个随机的运行 ID
    if not config.run_id:
        run_id = str(uuid.uuid4().int)
        # 记录警告日志，说明生成了一个随机的运行 ID
        logger.warning("config has no run_id, generated a random run_id: %s", run_id)
        # 将生成的运行 ID 赋值给配置文件中的 run_id 属性
        config.run_id = run_id

    # 获取入口点的名称，用于日志记录
    entrypoint_name = _get_entrypoint_name(entrypoint, args)

    # 记录信息日志，显示 elastic_operator 的启动配置参数
    logger.info(
        "Starting elastic_operator with launch configs:\n"
        "  entrypoint       : %(entrypoint)s\n"
        "  min_nodes        : %(min_nodes)s\n"
        "  max_nodes        : %(max_nodes)s\n"
        "  nproc_per_node   : %(nproc_per_node)s\n"
        "  run_id           : %(run_id)s\n"
        "  rdzv_backend     : %(rdzv_backend)s\n"
        "  rdzv_endpoint    : %(rdzv_endpoint)s\n"
        "  rdzv_configs     : %(rdzv_configs)s\n"
        "  max_restarts     : %(max_restarts)s\n"
        "  monitor_interval : %(monitor_interval)s\n"
        "  log_dir          : %(log_dir)s\n"
        "  metrics_cfg      : %(metrics_cfg)s\n",
        {
            "entrypoint": entrypoint_name,
            "min_nodes": config.min_nodes,
            "max_nodes": config.max_nodes,
            "nproc_per_node": config.nproc_per_node,
            "run_id": config.run_id,
            "rdzv_backend": config.rdzv_backend,
            "rdzv_endpoint": config.rdzv_endpoint,
            "rdzv_configs": config.rdzv_configs,
            "max_restarts": config.max_restarts,
            "monitor_interval": config.monitor_interval,
            "log_dir": config.logs_specs.root_log_dir,  # 日志目录路径
            "metrics_cfg": config.metrics_cfg,
        },
    )

    # 创建 RendezvousParameters 对象，用于协调分布式任务的节点间通信
    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        local_addr=config.local_addr,
        **config.rdzv_configs,  # 其他额外的协调参数
    )

    # 获取主节点的地址和端口号
    master_addr, master_port = _get_addr_and_port(rdzv_parameters)

    # 创建 WorkerSpec 对象，描述每个工作节点的配置和任务信息
    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(args),
        rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),  # 获取协调处理器
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        master_addr=master_addr,
        master_port=master_port,
        local_addr=config.local_addr,
    )

    # 创建 LocalElasticAgent 实例，用于管理本地的弹性任务执行代理
    agent = LocalElasticAgent(
        spec=spec,
        logs_specs=config.logs_specs,  # 日志配置
        start_method=config.start_method,  # 启动方法
        log_line_prefix_template=config.log_line_prefix_template,  # 日志行前缀模板
    )

    # 标记需要关闭协调服务的标志
    shutdown_rdzv = True
    try:
        # 初始化指标系统，使用给定的配置文件初始化
        metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))

        # 运行 agent 对象的任务
        result = agent.run()
        # 记录 agent.run() 成功完成的事件，而不是 workers 成功的事件
        events.record(agent.get_event_succeeded())

        # 检查任务执行结果是否失败
        if result.is_failed():
            # 如果失败，抛出 ChildFailedError 异常，将失败信息传递给异常
            # ChildFailedError 被 @record 处理特殊情况，
            # 如果存在子进程的错误文件，则 @record 将复制第一个错误（根本原因）
            # 到启动器进程的错误文件中。
            raise ChildFailedError(
                name=entrypoint_name,
                failures=result.failures,
            )

        # 如果任务执行成功，返回执行结果的返回值
        return result.return_values
    except ChildFailedError:
        # 如果抛出 ChildFailedError 异常，则直接抛出该异常
        raise
    except SignalException:
        # 当 agent 因信号而终止时，不关闭 rdzv_handler
        # 因为这会永久关闭此 rdzv_id 上的会合，并阻止任何额外的缩放事件
        shutdown_rdzv = False
        # 记录 agent 执行失败的事件
        events.record(agent.get_event_failed())
        # 再次抛出异常
        raise
    except Exception:
        # 捕获所有其他异常
        # 记录 agent 执行失败的事件
        events.record(agent.get_event_failed())
        # 再次抛出异常
        raise
    finally:
        # 最终执行的清理工作，如果需要关闭 rdzv_handler，则执行关闭操作
        if shutdown_rdzv:
            spec.rdzv_handler.shutdown()
```