# `.\pytorch\torch\_C\_distributed_c10d.pyi`

```py
# mypy: allow-untyped-defs
# mypy: disable-error-code="type-arg"
# 从 datetime 模块导入 timedelta 类型
from datetime import timedelta
# 从 enum 模块导入 Enum 类型
from enum import Enum
# 从 typing 模块导入 Any 和 overload 类型
from typing import Any, overload

# 导入 torch 库
import torch
# 从 torch 库导入 Tensor 类型
from torch import Tensor
# 从 torch._C 模块导入 ScriptObject 类型
from torch._C import ScriptObject
# 从 torch.futures 模块导入 Future 类型
from torch.futures import Future

# 这个模块在 torch/csrc/distributed/c10d/init.cpp 中定义

# 定义默认的第一个桶的字节数
_DEFAULT_FIRST_BUCKET_BYTES: int
# 定义默认的无超时时间间隔
_DEFAULT_NO_TIMEOUT: timedelta
# 定义默认的进程组超时时间间隔
_DEFAULT_PG_TIMEOUT: timedelta
# 定义默认的进程组 NCCL 超时时间间隔
_DEFAULT_PG_NCCL_TIMEOUT: timedelta

# 定义内置通信钩子类型的枚举
class BuiltinCommHookType(Enum):
    ALLREDUCE = ...
    FP16_COMPRESS = ...

# 注册通信钩子函数，接受 Reducer 对象、任意状态和通信钩子对象
def _register_comm_hook(reducer: Reducer, state: Any, comm_hook: Any): ...

# 注册内置通信钩子函数，接受 Reducer 对象、内置通信钩子类型
def _register_builtin_comm_hook(
    reducer: Reducer,
    comm_hook_type: BuiltinCommHookType,
): ...

# 设置全局排名，接受整数类型的排名参数，无返回值
def _set_global_rank(rank: int) -> None: ...

# 计算张量列表的哈希值，返回整数
def _hash_tensors(tensors: list[Tensor]) -> int: ...

# 定义梯度桶类
class GradBucket:
    # 返回桶的索引，返回整数
    def index(self) -> int: ...
    # 返回桶的缓冲区张量，返回 Tensor 类型
    def buffer(self) -> Tensor: ...
    # 返回桶中的梯度张量列表，返回 Tensor 类型的列表
    def gradients(self) -> list[Tensor]: ...
    # 判断当前桶是否为最后一个桶，返回布尔值
    def is_last(self) -> bool: ...
    # 设置桶的缓冲区张量，接受 Tensor 类型参数，无返回值
    def set_buffer(self, tensor: Tensor) -> None: ...
    # 返回桶中的参数张量列表，返回 Tensor 类型的列表
    def parameters(self) -> list[Tensor]: ...

# 定义减少器类
class Reducer:
    # 初始化减少器对象，接受多种参数，无返回值
    def __init__(
        self,
        params: list[Tensor],
        bucket_indices: list[list[int]],
        per_bucket_size_limits: list[int],
        process_group: ProcessGroup,
        expect_sparse_gradients: list[bool] = ...,
        bucket_bytes_cap: int = ...,  # kDefaultBucketBytesCap in reducer.hpp
        find_unused_parameters: bool = ...,
        gradient_as_bucket_view: bool = ...,
        param_to_name_mapping: dict[int, str] = ...,
        first_bucket_types_cap: int = ...,  # kDefaultFirstBucketBytes in reducer.hpp
    ) -> None: ...
    # 准备进行前向传播，无返回值
    def prepare_for_forward(self) -> None: ...
    # 准备进行反向传播，接受输出张量列表参数，无返回值
    def prepare_for_backward(self, output: list[Tensor]) -> None: ...
    # 获取反向传播的统计信息，返回整数列表
    def get_backward_stats(self) -> list[int]: ...
    # 安装后向传播后的未来对象，接受未来对象列表参数，无返回值
    def _install_post_backward_futures(self, futures: list[Future]) -> None: ...
    # 重建所有桶，返回布尔值
    def _rebuild_buckets(self) -> bool: ...
    # 获取与梯度大小相同的零张量桶列表，返回 GradBucket 类型的列表
    def _get_zeros_like_grad_buckets(self) -> list[GradBucket]: ...
    # 推送所有重建参数，无返回值
    def _push_all_rebuilt_params(self) -> None: ...
    # 设置前向传播工作句柄，接受工作对象和是否使用静态世界大小的布尔参数，无返回值
    def _set_forward_pass_work_handle(
        self,
        work: Work,
        use_static_world_size: bool,
    ): ...
    # 获取本地使用映射，返回 Tensor 类型
    def _get_local_used_map(self) -> Tensor: ...
    # 设置 DDP 运行时日志记录的采样率，接受采样率整数参数，无返回值
    def _set_ddp_runtime_logging_sample_rate(self, sample_rate: int) -> None: ...
    # 设置静态图，无返回值
    def _set_static_graph(self) -> None: ...
    # 运行通信钩子，接受梯度桶参数，返回 Future 类型
    def _run_comm_hook(self, bucket: GradBucket) -> Future: ...
    # 设置日志记录器，接受日志记录器对象参数，无返回值
    def set_logger(self, logger: Logger) -> None: ...
    # 移除自动求导钩子，无返回值
    def _remove_autograd_hooks(self) -> None: ...
    # 检查减少器是否已完成最终化，无返回值
    def _check_reducer_finalized(self) -> None: ...
    # 设置稀疏元数据，接受全局唯一 ID 字典参数，无返回值
    def _set_sparse_metadata(self, global_unique_ids: dict[str, Tensor]) -> None: ...
    # 重置状态，无返回值
    def _reset_state(self) -> None: ...
    # 更新进程组，接受新进程组参数，无返回值
    def _update_process_group(self, new_process_group: ProcessGroup) -> None: ...

# 定义分布式数据并行日志数据类
class DDPLoggingData:
    strs_map: dict[str, str]
    ints_map: dict[str, int]

# 定义日志记录器类，接受减少器对象参数，无返回值
class Logger:
    def __init__(self, reducer: Reducer) -> None: ...
    # 设置构造数据和记录日志的方法，接受多个参数：
    #   module_name: 模块名称，字符串类型
    #   device_ids: 设备 ID 列表，包含整数类型元素
    #   output_device: 输出设备 ID，整数类型
    #   broadcast_buffers: 是否广播缓冲区，布尔类型
    #   has_sync_bn: 是否使用同步 BatchNorm，布尔类型
    #   static_graph: 是否使用静态图，布尔类型
    def set_construction_data_and_log(
        self,
        module_name: str,
        device_ids: list[int],
        output_device: int,
        broadcast_buffers: bool,
        has_sync_bn: bool,
        static_graph: bool,
    ): ...
    
    # 设置运行时统计信息和记录日志的方法，没有返回值
    def set_runtime_stats_and_log(self) -> None: ...
    
    # 设置错误并记录日志的方法，接受一个错误消息作为参数
    #   error: 错误消息，字符串类型
    def set_error_and_log(self, error: str) -> None: ...
    
    # 获取分布式数据并记录日志的方法，返回类型为 DDPLoggingData 对象
    def _get_ddp_logging_data(self) -> DDPLoggingData: ...
    
    # 设置通信钩子名称的方法，接受一个字符串作为通信钩子的名称
    #   comm_hook: 通信钩子的名称，字符串类型
    def _set_comm_hook_name(self, comm_hook: str) -> None: ...
    
    # 设置不均匀输入连接的方法，没有返回值
    def _set_uneven_input_join(self) -> None: ...
    
    # 设置静态图的方法，没有返回值
    def _set_static_graph(self) -> None: ...
# 定义一个名为 _WorkerServer 的私有类，用于管理与工作服务器相关的功能
class _WorkerServer:
    # 初始化方法，接受一个字符串参数 socket_path，用于指定服务器的套接字路径
    def __init__(self, socket_path: str) -> None: ...
    
    # 关闭工作服务器的方法，没有返回值
    def shutdown(self) -> None: ...

# 获取调试级别的函数，返回当前调试级别
def get_debug_level(): ...

# 设置调试级别的函数，接受一个参数用于设置新的调试级别，没有返回值
def set_debug_level(): ...

# 从环境变量中设置调试级别的函数，没有参数和返回值
def set_debug_level_from_env(): ...

# 调试级别的枚举类，定义了几种调试级别选项
class DebugLevel(Enum):
    OFF = ...
    INFO = ...
    DETAIL = ...

# ReduceOp 类，用于定义不同的减少操作
class ReduceOp:
    # 初始化方法，接受一个 RedOpType 类型的参数 op
    def __init__(self, op: RedOpType) -> None: ...

    # 下面是几种预定义的减少操作类型常量
    SUM: RedOpType = ...
    AVG: RedOpType = ...
    PRODUCT: RedOpType = ...
    MIN: RedOpType = ...
    MAX: RedOpType = ...
    BAND: RedOpType = ...
    BOR: RedOpType = ...
    BXOR: RedOpType = ...
    PREMUL_SUM: RedOpType = ...
    UNUSED: RedOpType = ...

    # 减少操作类型的枚举定义在 RedOpType 类中
    class RedOpType(Enum): ...

# BroadcastOptions 类，定义了广播操作的选项
class BroadcastOptions:
    # 广播的根排名
    rootRank: int
    # 广播的根张量
    rootTensor: int
    # 超时时间
    timeout: timedelta
    # 是否异步操作
    asyncOp: bool

# AllreduceOptions 类，定义了全局归约操作的选项
class AllreduceOptions:
    # 归约操作的类型
    reduceOp: ReduceOp
    # 超时时间
    timeout: timedelta

# AllreduceCoalescedOptions 类，继承自 AllreduceOptions，定义了累积全局归约操作的选项
class AllreduceCoalescedOptions(AllreduceOptions): ...

# ReduceOptions 类，定义了减少操作的选项
class ReduceOptions:
    # 减少操作的类型
    reduceOp: ReduceOp
    # 减少操作的根排名
    rootRank: int
    # 减少操作的根张量
    rootTensor: int
    # 超时时间
    timeout: timedelta

# AllgatherOptions 类，定义了全收集操作的选项
class AllgatherOptions:
    # 超时时间
    timeout: timedelta
    # 是否异步操作
    asyncOp: bool

# GatherOptions 类，定义了收集操作的选项
class GatherOptions:
    # 收集的根排名
    rootRank: int
    # 超时时间
    timeout: timedelta

# ScatterOptions 类，定义了散列操作的选项
class ScatterOptions:
    # 散列的根排名
    rootRank: int
    # 超时时间
    timeout: timedelta
    # 是否异步操作
    asyncOp: bool

# ReduceScatterOptions 类，定义了减少散列操作的选项
class ReduceScatterOptions:
    # 减少散列操作的类型
    reduceOp: ReduceOp
    # 超时时间
    timeout: timedelta
    # 是否异步操作
    asyncOp: bool

# BarrierOptions 类，定义了屏障操作的选项
class BarrierOptions:
    # 设备 ID 列表
    device_ids: list[int]
    # 设备
    device: torch.device
    # 超时时间
    timeout: timedelta

# AllToAllOptions 类，定义了全对全操作的选项
class AllToAllOptions:
    # 超时时间
    timeout: timedelta

# Store 类，定义了键值存储的通用接口
class Store:
    # 设置键值对的方法
    def set(self, key: str, value: str): ...
    # 获取键对应值的方法，返回字节类型
    def get(self, key: str) -> bytes: ...
    # 添加整数值到键对应值的方法，返回新的值
    def add(self, key: str, value: int) -> int: ...
    # 比较并设置键对应值的方法，返回字节类型
    def compare_set(
        self,
        key: str,
        expected_value: str,
        desired_value: str,
    ) -> bytes: ...
    # 删除键对应值的方法，返回是否删除成功的布尔值
    def delete_key(self, key: str) -> bool: ...
    # 返回键的数量
    def num_keys(self) -> int: ...
    # 设置超时时间的方法
    def set_timeout(self, timeout: timedelta): ...
    # 等待键列表的方法，重载版本
    @overload
    def wait(self, keys: list[str]): ...
    # 等待键列表的方法，带超时参数的重载版本
    @overload
    def wait(self, keys: list[str], timeout: timedelta): ...

# FileStore 类，继承自 Store，实现了基于文件的键值存储
class FileStore(Store):
    # 初始化方法，接受路径参数 path 和可选的工作者数量参数 numWorkers
    def __init__(self, path: str, numWorkers: int = ...) -> None: ...

# HashStore 类，继承自 Store，实现了基于哈希表的键值存储
class HashStore(Store):
    # 初始化方法，没有参数
    def __init__(self) -> None: ...

# TCPStore 类，继承自 Store，实现了基于 TCP 协议的键值存储
class TCPStore(Store):
    # 初始化方法，接受多个参数来配置 TCP 连接的各种选项
    def __init__(
        self,
        host_name: str,
        port: int,
        world_size: int | None = ...,
        is_master: bool = ...,
        timeout: timedelta = ...,
        wait_for_workers: bool = ...,
        multi_tenant: bool = ...,
        master_listen_fd: int | None = ...,
        use_libuv: bool | None = ...,
    ) -> None: ...
    
    # 主机名属性，返回 TCP 连接的主机名
    @property
    def host(self) -> str: ...
    
    # 端口属性，返回 TCP 连接的端口号
    @property
    def port(self) -> int: ...

# PrefixStore 类，继承自 Store，实现了带有前缀的键值存储
class PrefixStore(Store):
    # 初始化方法，接受前缀字符串和基础存储对象作为参数
    def __init__(self, prefix: str, store: Store) -> None: ...
    
    # 基础存储对象的属性，返回被封装的基础存储对象
    @property
    def underlying_store(self) -> Store: ...

# _ControlCollectives 类，定义了控制集合操作的接口
class _ControlCollectives:
    # 屏障操作的方法，接受键、超时时间和阻塞标志作为参数，没有返回值
    def barrier(self, key: str, timeout: timedelta, blocking: bool) -> None: ...
    
    # 广播发送操作的方法，接受键、数据和超时时间作为参数，没有返回值
    def broadcast_send(self, key: str, data: str, timeout: timedelta) -> None: ...
    # 接收广播消息，根据键名和超时时间
    def broadcast_recv(self, key: str, timeout: timedelta) -> str:
        ...
    
    # 发送数据进行聚合，根据键名和数据内容以及超时时间
    def gather_send(self, key: str, data: str, timeout: timedelta) -> None:
        ...
    
    # 接收聚合数据，根据键名和超时时间
    def gather_recv(self, key: str, timeout: timedelta) -> str:
        ...
    
    # 发送数据进行分发，根据键名和数据内容以及超时时间
    def scatter_send(self, key: str, data: str, timeout: timedelta) -> None:
        ...
    
    # 接收分发数据，根据键名和超时时间
    def scatter_recv(self, key: str, timeout: timedelta) -> str:
        ...
    
    # 聚合所有节点的数据，根据键名和数据内容以及超时时间
    def all_gather(self, key: str, data: str, timeout: timedelta) -> str:
        ...
    
    # 对所有节点的整数数据进行求和，根据键名和整数数据以及超时时间
    def all_sum(self, key: str, data: int, timeout: timedelta) -> int:
        ...
class _StoreCollectives(_ControlCollectives):
    # _StoreCollectives 类继承自 _ControlCollectives 类，用于管理存储集合操作
    def __init__(self, store: Store, rank: int, world_size: int) -> None:
        # 初始化方法，接受 store（存储对象）、rank（当前进程在集合中的排名）、world_size（集合中的进程总数）作为参数
        ...


class _DistributedBackendOptions:
    # _DistributedBackendOptions 类定义了分布式后端的选项
    def __init__(self) -> None:
        # 初始化方法，不接受参数
        ...

    @property
    def store(self) -> Store:
        # 获取存储对象的属性方法
        ...

    @store.setter
    def store(self, store: Store) -> None:
        # 设置存储对象的属性方法
        ...

    @property
    def group_rank(self) -> int:
        # 获取集合中当前进程的排名属性方法
        ...

    @group_rank.setter
    def group_rank(self, rank: int) -> None:
        # 设置集合中当前进程的排名属性方法
        ...

    @property
    def group_size(self) -> int:
        # 获取集合中进程总数的属性方法
        ...

    @group_size.setter
    def group_size(self, size: int) -> None:
        # 设置集合中进程总数的属性方法
        ...

    @property
    def timeout(self) -> timedelta:
        # 获取超时时间的属性方法，返回 timedelta 对象
        ...

    @timeout.setter
    def timeout(self, timeout: timedelta) -> None:
        # 设置超时时间的属性方法，参数为 timedelta 对象
        ...

    @property
    def group_id(self) -> str:
        # 获取集合的唯一标识符属性方法
        ...

    @group_id.setter
    def group_id(self, group_id: str) -> None:
        # 设置集合的唯一标识符属性方法
        ...

    @property
    def global_ranks_in_group(self) -> list[int]:
        # 获取集合中全局排名列表的属性方法
        ...

    @global_ranks_in_group.setter
    def global_ranks_in_group(self, ranks: list[int]) -> None:
        # 设置集合中全局排名列表的属性方法
        ...


class Work:
    # Work 类定义了与工作任务相关的方法和属性
    def is_completed(self) -> bool:
        # 检查工作是否已完成的方法，返回布尔值
        ...

    def is_success(self) -> bool:
        # 检查工作是否成功的方法，返回布尔值
        ...

    def exception(self) -> Any:
        # 获取与工作相关的异常信息的方法，返回任意类型
        ...

    def wait(self, timeout: timedelta = ...) -> bool:
        # 等待工作完成的方法，可选超时时间参数为 timedelta 对象，默认为省略符号
        ...

    def get_future(self) -> Future:
        # 获取工作的未来结果对象的方法，返回 Future 对象
        ...

    def source_rank(self) -> int:
        # 获取工作来源进程的排名的方法，返回整数
        ...

    def _source_rank(self) -> int:
        # 私有方法，获取工作来源进程的排名的方法，返回整数
        ...

    def result(self) -> list[Tensor]:
        # 获取工作的结果张量列表的方法，返回列表
        ...

    def synchronize(self):
        # 同步工作任务的方法
        ...

    def boxed(self) -> ScriptObject:
        # 封装工作任务为脚本对象的方法，返回 ScriptObject 对象
        ...

    @staticmethod
    def unbox(obj: ScriptObject) -> Work:
        # 静态方法，从脚本对象中解封工作任务的方法，返回 Work 对象
        ...


class Backend:
    # Backend 类定义了后端处理的方法和属性
    def __init__(
        self,
        rank: int,
        size: int,
    ) -> None:
        # 初始化方法，接受 rank（当前进程在集合中的排名）和 size（集合中的进程总数）作为参数
        ...

    @property
    def supports_splitting(self) -> bool:
        # 判断后端是否支持分割的属性方法，返回布尔值
        ...

    def rank(self) -> int:
        # 获取当前进程在集合中的排名的方法，返回整数
        ...

    def size(self) -> int:
        # 获取集合中的进程总数的方法，返回整数
        ...

    def eager_connect_single_device(self, device: torch.device | None) -> None:
        # 在单一设备上进行快速连接的方法，设备参数可以为 None
        ...

    def _set_sequence_number_for_group(self) -> None:
        # 设置集合的序列号的方法，无返回值
        ...


class ProcessGroup:
    # ProcessGroup 类定义了处理组的方法和属性
    class Options:
        # Options 类定义了处理组选项的初始化方法和属性
        def __init__(self, backend: str, timeout: timedelta = ...) -> None:
            # 初始化方法，接受 backend（后端名称）和 timeout（超时时间，默认为省略符号）作为参数
            ...

        @property
        def backend(self) -> str:
            # 获取后端名称的属性方法，返回字符串
            ...

        @property
        def _timeout(self) -> timedelta:
            # 获取超时时间的属性方法，返回 timedelta 对象
            ...

        @_timeout.setter
        def _timeout(self, val: timedelta) -> None:
            # 设置超时时间的属性方法，参数为 timedelta 对象，无返回值
            ...

    class BackendType(Enum):
        # 定义处理组后端类型的枚举
        UNDEFINED = ...
        GLOO = ...
        NCCL = ...
        UCC = ...
        MPI = ...
        CUSTOM = ...

    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        options: Options,
    ) -> None:
        # 初始化方法，接受 store（存储对象）、rank（当前进程在集合中的排名）、size（集合中的进程总数）、options（处理组选项）作为参数
        ...

    def rank(self) -> int:
        # 获取当前进程在集合中的排名的方法，返回整数
        ...

    def size(self) -> int:
        # 获取集合中的进程总数的方法，返回整数
        ...

    @overload
    def broadcast(
        self,
        tensors: list[Tensor],
        opts=...,
    ) -> Work:
        # 广播张量列表的重载方法，返回 Work 对象
        ...

    @overload
    def broadcast(
        self,
        tensor: Tensor,
        root: int,
    ) -> Work:
        # 广播单个张量的重载方法，接受 tensor（张量对象）和 root（广播的根进程排名）作为参数，返回 Work 对象
        ...

    @overload
    def allreduce(
        self,
        tensors: list[Tensor],
        opts: AllreduceOptions = ...,
    ) -> Work:
        # 执行张量列表全局归约的重载方法，接受 tensors（张量对象列表）和 opts（全局归约选项）作为参数，返回 Work 对象
        ...

    @overload
    def allreduce(
        self,
        tensors: list[Tensor],
        op=...,
    ) -> Work:
        # 执行张量列表全局归约的重载方法，接受 tensors（张量对象列表）和 op（归约操作）作为参数，返回 Work 对象
        ...
    # 定义 allreduce 方法，用于在分布式环境中对张量进行全局归约操作
    def allreduce(
        self,
        tensor: Tensor,
        op=...,
    ) -> Work: ...

    # 定义 allreduce_coalesced 方法，用于在分布式环境中对多个张量进行全局归约操作
    def allreduce_coalesced(
        self,
        tensors: list[Tensor],
        opts=...,
    ) -> Work: ...

    # 定义 reduce_scatter_tensor_coalesced 方法，用于在分布式环境中对多个输入张量进行 reduce scatter 操作
    def reduce_scatter_tensor_coalesced(
        self,
        outputTensors: list[Tensor],
        inputTensors: list[Tensor],
        opts: ReduceScatterOptions | None = None,
    ) -> Work: ...

    # 重载方法定义：reduce 方法，支持对单个张量或多个张量进行归约操作
    @overload
    def reduce(
        self,
        tensors: list[Tensor],
        opts=...,
    ) -> Work: ...

    # 重载方法定义：reduce 方法，支持对单个张量在指定根节点上进行归约操作
    @overload
    def reduce(
        self,
        tensor: Tensor,
        root: int,
        op=...,
    ) -> Work: ...

    # 重载方法定义：allgather 方法，支持将多个输入张量按照指定选项进行全收集操作
    @overload
    def allgather(
        self,
        output_tensors: list[list[Tensor]],
        input_tensors: list[Tensor],
        opts=...,
    ) -> Work: ...

    # 重载方法定义：allgather 方法，支持将单个输入张量进行全收集操作
    @overload
    def allgather(
        self,
        output_tensors: list[Tensor],
        input_tensor: Tensor,
    ) -> Work: ...

    # 定义 _allgather_base 方法，用于实现基础的全收集操作
    def _allgather_base(
        self,
        output: Tensor,
        input: Tensor,
        opts=...,
    ) -> Work: ...

    # 定义 allgather_coalesced 方法，用于在分布式环境中对多个输入张量进行 coalesced 全收集操作
    def allgather_coalesced(
        self,
        output_lists: list[list[Tensor]],
        input_list: list[Tensor],
        opts=...,
    ) -> Work: ...

    # 定义 allgather_into_tensor_coalesced 方法，用于在分布式环境中将多个输入张量 coalesced 全收集到单个张量中
    def allgather_into_tensor_coalesced(
        self,
        output_lists: list[Tensor],
        input_list: list[Tensor],
        opts=...,
    ) -> Work: ...

    # 重载方法定义：gather 方法，支持将多个输入张量按照指定选项进行 gather 操作
    @overload
    def gather(
        self,
        output_tensors: list[list[Tensor]],
        input_tensors: list[Tensor],
        opts=...,
    ) -> Work: ...

    # 重载方法定义：gather 方法，支持将单个输入张量在指定根节点上进行 gather 操作
    @overload
    def gather(
        self,
        output_tensors: list[Tensor],
        input_tensor: Tensor,
        root: int,
    ) -> Work: ...

    # 重载方法定义：scatter 方法，支持将单个输出张量分散到多个输入张量上
    @overload
    def scatter(
        self,
        output_tensors: list[Tensor],
        input_tensors: list[list[Tensor]],
        opts=...,
    ) -> Work: ...

    # 重载方法定义：scatter 方法，支持将单个输出张量分散到指定根节点的多个输入张量上
    @overload
    def scatter(
        self,
        output_tensor: Tensor,
        input_tensors: list[Tensor],
        root: int,
    ) -> Work: ...

    # 重载方法定义：reduce_scatter 方法，支持对多个输入张量按照指定选项进行 reduce scatter 操作
    @overload
    def reduce_scatter(
        self,
        output_tensors: list[Tensor],
        input_tensors: list[list[Tensor]],
        opts=...,
    ) -> Work: ...

    # 重载方法定义：reduce_scatter 方法，支持将单个输入张量在多个输入张量上进行 reduce scatter 操作
    @overload
    def reduce_scatter(
        self,
        output_tensors: Tensor,
        input_tensor: list[Tensor],
    ) -> Work: ...

    # 定义 _reduce_scatter_base 方法，用于实现基础的 reduce scatter 操作
    def _reduce_scatter_base(
        self,
        outputTensor: Tensor,
        inputTensor: Tensor,
        opts: ReduceScatterOptions | None,
    ) -> Work: ...

    # 重载方法定义：alltoall_base 方法，支持将单个输入张量分发到多个输出张量上，并按照指定选项进行分发
    @overload
    def alltoall_base(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts=...,
    ) -> Work: ...

    # 重载方法定义：alltoall_base 方法，支持将单个输入张量分发到多个输出张量上
    @overload
    def alltoall_base(
        self,
        output: Tensor,
        input: Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
    ) -> Work: ...
    # 定义类方法 `alltoall`，用于执行所有节点间的张量交换操作
    def alltoall(
        self,
        output_tensor: list[Tensor],  # 输出张量列表
        input_tensor: list[Tensor],   # 输入张量列表
        opts=...,                     # 可选参数
    ) -> Work:                        # 返回一个 `Work` 对象，表示执行的任务
    
    # 函数重载：定义类方法 `alltoall` 的另一版本，不包含可选参数 `opts`
    @overload
    def alltoall(
        self,
        output: list[Tensor],  # 输出张量列表
        input: list[Tensor],   # 输入张量列表
    ) -> Work:                  # 返回一个 `Work` 对象，表示执行的任务
    
    # 定义类方法 `send`，用于发送张量数据到指定节点
    def send(
        self,
        tensors: list[Tensor],  # 待发送的张量列表
        dstRank: int,           # 目标节点的排名
        tag: int,               # 标记，用于区分不同的通信
    ) -> Work:                   # 返回一个 `Work` 对象，表示发送操作的任务
    
    # 定义类方法 `recv`，用于接收指定节点发送的张量数据
    def recv(
        self,
        tensors: list[Tensor],  # 接收到的张量将存储在这个列表中
        srcRank: int,           # 源节点的排名
        tag: int,               # 标记，用于区分不同的通信
    ) -> Work:                   # 返回一个 `Work` 对象，表示接收操作的任务
    
    # 定义类方法 `recv_anysource`，用于从任意节点接收张量数据
    def recv_anysource(self, tensors: list[Tensor], tag: int) -> Work:
        # 接收到的张量存储在指定的张量列表中，标记用于区分不同的通信
        ...
    
    # 定义类方法 `barrier`，用于同步多个节点的操作
    def barrier(self, opts=...) -> Work:
        # 可选参数用于设置特定的同步行为
        ...
    
    # 定义类方法 `boxed`，返回一个 `ScriptObject` 对象
    def boxed(self) -> ScriptObject:
        ...
    
    # 定义静态方法 `unbox`，用于从 `ScriptObject` 解包出 `ProcessGroup` 对象
    @staticmethod
    def unbox(obj: ScriptObject) -> ProcessGroup:
        ...
    
    # 定义类方法 `_start_coalescing`，用于在指定设备上开始合并操作
    def _start_coalescing(self, device: torch.device) -> None:
        ...
    
    # 定义类方法 `_end_coalescing`，用于在指定设备上结束合并操作
    def _end_coalescing(self, device: torch.device) -> Work:
        ...
    
    # 定义类方法 `_get_backend_name`，返回当前使用的后端名称
    def _get_backend_name(self) -> str:
        ...
    
    # 定义类方法 `_backend_id`，返回指定类型后端的 ID
    def _backend_id(self, backend_type: BackendType) -> int:
        ...
    
    # 定义属性 `_device_types`，返回支持的设备类型列表
    @property
    def _device_types(self) -> list[torch.device]:
        ...
    
    # 定义类方法 `_get_backend`，返回指定设备的后端对象
    def _get_backend(self, device: torch.device) -> Backend:
        ...
    
    # 定义类方法 `_register_backend`，注册指定设备的后端对象
    def _register_backend(
        self,
        device: torch.device,
        backend_type: BackendType,
        backend: Backend | None,
    ) -> None:
        ...
    
    # 定义类方法 `_set_group_name`，设置当前通信组的名称
    def _set_group_name(self, name: str) -> None:
        ...
    
    # 定义类方法 `_set_group_desc`，设置当前通信组的描述信息
    def _set_group_desc(self, desc: str) -> None:
        ...
    
    # 定义实例方法 `name`，返回对象的名称
    def name(self) -> str:
        ...
    
    # 定义类方法 `_has_hooks`，判断是否有注册的钩子函数
    def _has_hooks(self) -> bool:
        ...
    
    # 定义类方法 `_wait_for_pending_works`，等待当前所有未完成的任务完成
    def _wait_for_pending_works(self) -> None:
        ...
    
    # 定义类方法 `_set_sequence_number_for_group`，设置当前通信组的序列号
    def _set_sequence_number_for_group(self) -> None:
        ...
    
    # 定义属性 `bound_device_id`，返回绑定的设备 ID，可能为空
    @property
    def bound_device_id(self) -> torch.device | None:
        ...
    
    # 定义属性 `bound_device_id` 的 setter 方法，设置绑定的设备 ID
    @bound_device_id.setter
    def bound_device_id(self, device: torch.device | None) -> None:
        ...
    
    # 定义属性 `group_name`，返回当前通信组的名称
    @property
    def group_name(self) -> str:
        ...
    
    # 定义属性 `group_desc`，返回当前通信组的描述信息
    @property
    def group_desc(self) -> str:
        ...
class ProcessGroupRoundRobin(ProcessGroup):
    # 定义一个继承自 ProcessGroup 的 RoundRobin 过程组类

def _round_robin_process_groups(
    process_groups: list[ProcessGroup],
) -> ProcessGroupRoundRobin:
    # 返回一个 RoundRobin 过程组对象，接受一个 ProcessGroup 列表作为参数

class ProcessGroupGloo(Backend):
    # 定义一个继承自 Backend 的 Gloo 过程组类
    class Device: ...
    # 定义 Gloo 过程组的设备类
    class Options: ...
    # 定义 Gloo 过程组的选项类

    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ) -> None:
        # Gloo 过程组的初始化方法，接受存储、排名、大小和超时时间作为参数

    @staticmethod
    def create_device(hostname="", interface="") -> Device:
        # 静态方法：创建一个指定主机名和接口的设备对象

    @staticmethod
    def create_default_device() -> Device:
        # 静态方法：创建默认设备对象

    def _set_default_timeout(self, timeout) -> None:
        # 设置默认超时时间的方法

class _ProcessGroupWrapper(Backend):
    # 定义一个继承自 Backend 的过程组包装类

    def __init__(self, pg: Backend, gloo_pg: ProcessGroupGloo) -> None:
        # 过程组包装类的初始化方法，接受两个参数：pg 和 gloo_pg

    wrapped_pg: Backend
    # 包装后的过程组对象

class ProcessGroupNCCL(Backend):
    # 定义一个继承自 Backend 的 NCCL 过程组类

    class Options:
        # NCCL 过程组的选项类

        def __init__(self, timeout: timedelta | None = None) -> None:
            # 初始化选项类，接受超时时间作为参数

        @property
        def backend(self) -> str:
            # 返回后端名称的属性方法

        @property
        def _timeout(self) -> timedelta:
            # 返回超时时间的属性方法

        @_timeout.setter
        def _timeout(self, val: timedelta) -> None:
            # 设置超时时间的属性方法

        @property
        def _is_high_priority_stream(self) -> bool:
            # 返回是否高优先级流的属性方法

        @_is_high_priority_stream.setter
        def _is_high_priority_stream(self, val: bool) -> None:
            # 设置是否高优先级流的属性方法

    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ) -> None:
        # NCCL 过程组的初始化方法，接受存储、排名、大小和超时时间作为参数

    def _group_start(self) -> None:
        # 启动过程组的方法

    def _group_end(self) -> None:
        # 结束过程组的方法

    def _set_default_timeout(self, timeout) -> None:
        # 设置默认超时时间的方法

    def _shutdown(self) -> None:
        # 关闭 NCCL 过程组的方法

    @property
    def uid(self) -> int:
        # 返回过程组唯一标识符的属性方法

class ProcessGroupUCC(Backend):
    # 定义一个继承自 Backend 的 UCC 过程组类

    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ) -> None:
        # UCC 过程组的初始化方法，接受存储、排名、大小和超时时间作为参数

class ProcessGroupMPI(Backend):
    # 定义一个继承自 Backend 的 MPI 过程组类

    def __init__(
        self,
        rank: int,
        size: int,
        pgComm: int,
    ) -> None:
        # MPI 过程组的初始化方法，接受排名、大小和进程组通信对象作为参数

    @staticmethod
    def create(ranks: list[int]) -> ProcessGroupMPI:
        # 静态方法：根据排名列表创建 MPI 过程组对象

def _compute_bucket_assignment_by_size(
    tensors: list[Tensor],
    bucket_size_limits: list[int],
    expect_sparse_gradient: list[bool] = ...,
    tensor_indices: list[int] = ...,
) -> tuple[list[list[int]], list[int]]:
    # 根据张量大小分配桶的方法，返回分配结果和桶的大小限制列表

def _broadcast_coalesced(
    process_group: ProcessGroup,
    tensors: list[Tensor],
    buffer_size: int,
    src: int,
):
    # 将张量在指定过程组中广播和合并的方法，接受过程组、张量列表、缓冲区大小和源作为参数

def _test_python_store(store: Store):
    # 测试 Python 存储的方法，接受存储对象作为参数

def _verify_params_across_processes(
    process_group: ProcessGroup,
    params: list[Tensor],
    logger: Logger | None,
):
    # 在多个进程间验证参数的方法，接受过程组、张量列表和日志对象作为参数

def _make_nccl_premul_sum(factor: float | list[Tensor]) -> ReduceOp:
    # 创建 NCCL 预乘和的方法，接受浮点数或张量列表作为参数

def _register_process_group(
    group_name: str,
    process_group: ProcessGroup,
) -> None:
    # 注册过程组的方法，接受组名和过程组对象作为参数

def _resolve_process_group(group_name: str) -> ProcessGroup:
    # 解析过程组的方法，接受组名作为参数并返回对应的过程组对象

def _unregister_all_process_groups() -> None:
    # 注销所有过程组的方法

def _unregister_process_group(group_name: str) -> None:
    # 注销特定过程组的方法，接受组名作为参数

class _SymmetricMemory:
    # 对称内存类
    @staticmethod
    # 静态方法
    # 定义静态方法 set_group_info，用于设置组的信息
    def set_group_info(
        group_name: str,
        rank: int,
        world_size: int,
        store: Store,
    ) -> None: ...

    # 定义静态方法 empty_strided_p2p，用于创建空的分布式张量
    def empty_strided_p2p(
        size: torch.types._size,
        stride: torch.types._size,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str,
    ) -> torch.Tensor: ...

    # 定义属性方法 rank，返回当前进程的排名
    @property
    def rank(self) -> int: ...

    # 定义属性方法 world_size，返回当前组中的进程总数
    @property
    def world_size(self) -> int: ...

    # 定义静态方法 rendezvous，用于协调张量在分布式环境中的同步
    def rendezvous(tensor: torch.Tensor) -> _SymmetricMemory: ...

    # 定义实例方法 get_buffer，用于获取指定进程的数据缓冲区
    def get_buffer(
        self,
        rank: int,
        sizes: torch.types._size,
        dtype: torch.dtype,
        storage_offset: int | None = 0,
    ) -> torch.Tensor: ...

    # 定义实例方法 barrier，用于实现进程间的同步屏障操作
    def barrier(self, channel: int = 0) -> None: ...

    # 定义实例方法 put_signal，用于向指定进程发送信号
    def put_signal(self, dst_rank: int, channel: int = 0) -> None: ...

    # 定义实例方法 wait_signal，用于等待来自指定进程的信号
    def wait_signal(self, src_rank: int, channel: int = 0) -> None: ...
```