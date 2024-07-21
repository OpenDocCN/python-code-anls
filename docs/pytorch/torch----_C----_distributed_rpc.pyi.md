# `.\pytorch\torch\_C\_distributed_rpc.pyi`

```
# 引入类型检查相关的声明
# 允许未标注类型的函数定义
# 禁止类型错误码为 "type-arg"
from datetime import timedelta
from typing import Any, Generic, overload, TypeVar

# 引入 PyTorch 库
import torch
from torch._C import Future
from torch._C._autograd import ProfilerEvent
from torch._C._distributed_c10d import Store
from torch._C._profiler import ProfilerConfig

# 默认的 RPC 初始化方法和工作线程数量
_DEFAULT_INIT_METHOD: str
_DEFAULT_NUM_WORKER_THREADS: int
_UNSET_RPC_TIMEOUT: float
_DEFAULT_RPC_TIMEOUT_SEC: float

_T = TypeVar("_T")

# 定义 RPC 后端选项类
class RpcBackendOptions:
    rpc_timeout: float
    init_method: str
    def __init__(
        self,
        rpc_timeout: float = ...,
        init_method: str = ...,
    ) -> None: ...

# 表示远程工作节点的信息
class WorkerInfo:
    def __init__(self, name: str, worker_id: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def id(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

# 表示 RPC 代理的接口
class RpcAgent:
    # 加入 RPC 环境，可选择是否关闭并设置超时时间
    def join(self, shutdown: bool = False, timeout: float = 0): ...
    # 同步方法，确保所有操作完成
    def sync(self): ...
    # 关闭 RPC 代理
    def shutdown(self): ...
    # 获取当前工作节点信息（重载方法）
    @overload
    def get_worker_info(self) -> WorkerInfo: ...
    @overload
    def get_worker_info(self, workerName: str) -> WorkerInfo: ...
    # 获取所有工作节点信息列表
    def get_worker_infos(self) -> list[WorkerInfo]: ...
    # 获取到目标节点的设备映射
    def _get_device_map(self, dst: WorkerInfo) -> dict[torch.device, torch.device]: ...
    # 获取调试信息
    def get_debug_info(self) -> dict[str, str]: ...
    # 获取度量信息
    def get_metrics(self) -> dict[str, str]: ...

# 表示 Python 远程引用的类
class PyRRef(Generic[_T]):
    def __init__(self, value: _T, type_hint: Any = None) -> None: ...
    # 判断是否为所有者
    def is_owner(self) -> bool: ...
    # 是否已被所有者确认
    def confirmed_by_owner(self) -> bool: ...
    # 返回所有者信息
    def owner(self) -> WorkerInfo: ...
    # 返回所有者名称
    def owner_name(self) -> str: ...
    # 同步方法，获取本地值
    def to_here(self, timeout: float = ...) -> _T: ...
    # 返回本地值
    def local_value(self) -> Any: ...
    # 远程过程调用，同步方式
    def rpc_sync(self, timeout: float = ...) -> Any: ...
    # 远程过程调用，异步方式
    def rpc_async(self, timeout: float = ...) -> Any: ...
    # 远程调用
    def remote(self, timeout: float = ...) -> Any: ...
    # 序列化方法
    def _serialize(self) -> tuple: ...
    # 反序列化静态方法
    @staticmethod
    def _deserialize(tp: tuple) -> PyRRef: ...
    # 返回类型
    def _get_type(self) -> type[_T]: ...
    # 返回异步操作的未来对象
    def _get_future(self) -> Future[_T]: ...
    # 返回分析异步操作的未来对象
    def _get_profiling_future(self) -> Future[_T]: ...
    # 设置分析异步操作的未来对象
    def _set_profiling_future(self, profilingFuture: Future[_T]): ...

# 基于 TensorPipe 的 RPC 后端选项类
class _TensorPipeRpcBackendOptionsBase(RpcBackendOptions):
    num_worker_threads: int
    device_maps: dict[str, dict[torch.device, torch.device]]
    devices: list[torch.device]
    def __init__(
        self,
        num_worker_threads: int,
        _transports: list | None,
        _channels: list | None,
        rpc_timeout: float = ...,
        init_method: str = ...,
        device_maps: dict[str, dict[torch.device, torch.device]] = {},  # noqa: B006
        devices: list[torch.device] = [],  # noqa: B006
    ) -> None: ...
    # 设置设备映射
    def _set_device_map(
        self,
        to: str,
        device_map: dict[torch.device, torch.device],
    ): ...

# 基于 TensorPipe 的 RPC 代理类
class TensorPipeAgent(RpcAgent):
    # 初始化方法，用于设置RPC worker的基本信息和参数
    def __init__(
        self,
        store: Store,  # 存储对象，用于RPC调用过程中存储和检索数据
        name: str,  # RPC worker的名称
        worker_id: int,  # RPC worker的唯一标识符
        world_size: int | None,  # RPC group中的worker数量，如果未知则为None
        opts: _TensorPipeRpcBackendOptionsBase,  # TensorPipe RPC后端选项对象
        reverse_device_maps: dict[str, dict[torch.device, torch.device]],  # 反向设备映射字典，用于设备间的映射关系
        devices: list[torch.device],  # 该worker使用的设备列表
    ) -> None: ...
    
    # 加入RPC group的方法，可选择是否关闭和设定超时时间
    def join(self, shutdown: bool = False, timeout: float = 0): ...
    
    # 关闭RPC worker的方法
    def shutdown(self): ...
    
    # 根据不同的参数重载方法，获取特定worker的信息
    @overload
    def get_worker_info(self) -> WorkerInfo: ...  # 获取本地worker的信息
    @overload
    def get_worker_info(self, workerName: str) -> WorkerInfo: ...  # 根据worker名称获取信息
    @overload
    def get_worker_info(self, id: int) -> WorkerInfo: ...  # 根据worker id获取信息
    def get_worker_infos(self) -> list[WorkerInfo]: ...  # 获取所有worker的信息列表
    
    # 获取与目标worker通信的设备映射关系
    def _get_device_map(self, dst: WorkerInfo) -> dict[torch.device, torch.device]: ...
    
    # 更新RPC group的成员关系，包括设备映射等
    def _update_group_membership(
        self,
        worker_info: WorkerInfo,
        my_devices: list[torch.device],
        reverse_device_map: dict[str, dict[torch.device, torch.device]],
        is_join: bool,
    ): ...
    
    # 获取RPC后端选项的方法
    def _get_backend_options(self) -> _TensorPipeRpcBackendOptionsBase: ...
    
    # 判断RPC group是否为静态的属性
    @property
    def is_static_group(self) -> bool: ...
    
    # 获取存储对象的属性
    @property
    def store(self) -> Store: ...
# 检查当前是否设置了 RPC 代理，并返回布尔值
def _is_current_rpc_agent_set() -> bool: ...

# 返回当前的 RPC 代理对象
def _get_current_rpc_agent() -> RpcAgent: ...

# 设置并启动指定的 RPC 代理
def _set_and_start_rpc_agent(agent: RpcAgent): ...

# 重置当前的 RPC 代理
def _reset_current_rpc_agent(): ...

# 删除所有用户和未分叉所有者的远程引用，并设置超时时间
def _delete_all_user_and_unforked_owner_rrefs(timeout: timedelta = ...): ...

# 销毁远程引用上下文，可选择忽略引用泄漏
def _destroy_rref_context(ignoreRRefLeak: bool): ...

# 获取远程引用上下文的调试信息，返回字典形式的调试信息
def _rref_context_get_debug_info() -> dict[str, str]: ...

# 清理 Python RPC 处理器的资源
def _cleanup_python_rpc_handler(): ...

# 调用内置的 RPC 方法，向目标节点发送指定操作名和参数
def _invoke_rpc_builtin(
    dst: WorkerInfo,
    opName: str,
    rpcTimeoutSeconds: float,
    *args: Any,
    **kwargs: Any,
): ...

# 调用 Python UDF 的 RPC 方法，向目标节点发送序列化后的 Python UDF 和张量
def _invoke_rpc_python_udf(
    dst: WorkerInfo,
    pickledPythonUDF: str,
    tensors: list[torch.Tensor],
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
): ...

# 调用 TorchScript 的 RPC 方法，向目标工作节点发送 TorchScript 函数和参数
def _invoke_rpc_torchscript(
    dstWorkerName: str,
    qualifiedNameStr: str,
    argsTuple: tuple,
    kwargsDict: dict,
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
): ...

# 调用内置的远程方法，向目标节点发送指定操作名和参数
def _invoke_remote_builtin(
    dst: WorkerInfo,
    opName: str,
    rpcTimeoutSeconds: float,
    *args: Any,
    **kwargs: Any,
): ...

# 调用 Python UDF 的远程方法，向目标节点发送序列化后的 Python UDF 和张量
def _invoke_remote_python_udf(
    dst: WorkerInfo,
    pickledPythonUDF: str,
    tensors: list[torch.Tensor],
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
): ...

# 调用 TorchScript 的远程方法，向目标节点发送 TorchScript 函数和参数
def _invoke_remote_torchscript(
    dstWorkerName: WorkerInfo,
    qualifiedNameStr: str,
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
    *args: Any,
    **kwargs: Any,
): ...

# 获取 RPC 超时时间
def get_rpc_timeout() -> float: ...

# 启用或禁用 GIL 分析
def enable_gil_profiling(flag: bool): ...

# 设置 RPC 超时时间
def _set_rpc_timeout(rpcTimeoutSeconds: float): ...

class RemoteProfilerManager:
    # 设置当前的性能分析器键
    @staticmethod
    def set_current_profiling_key(key: str): ...

# 启用服务器进程全局性能分析器，使用新的配置
def _enable_server_process_global_profiler(new_config: ProfilerConfig): ...

# 禁用服务器进程全局性能分析器，并返回所有事件的列表
def _disable_server_process_global_profiler() -> list[list[list[ProfilerEvent]]]: ...

# 设置性能分析器节点 ID 的默认值
def _set_profiler_node_id(default_node_id: int): ...

# 启用 JIT 对远程引用的序列化和反序列化
def _enable_jit_rref_pickle(): ...

# 禁用 JIT 对远程引用的序列化和反序列化
def _disable_jit_rref_pickle(): ...
```