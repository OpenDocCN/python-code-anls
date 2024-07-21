# `.\pytorch\torch\distributed\rpc\__init__.py`

```
# 设置类型检查时允许未标注的函数定义
# 导入日志模块
import logging
# 导入操作系统功能模块
import os
# 导入线程模块
import threading
# 导入警告模块
import warnings
# 导入时间间隔模块
from datetime import timedelta
# 导入生成器和元组类型
from typing import Generator, Tuple
# 导入 URL 解析模块
from urllib.parse import urlparse

# 导入 PyTorch 模块
import torch
# 导入分布式通信模块
import torch.distributed as dist

# 定义公开的模块成员列表
__all__ = ["is_available"]

# 设置日志记录器
logger = logging.getLogger(__name__)

# 初始化计数器
_init_counter = 0
# 初始化计数器的线程锁
_init_counter_lock = threading.Lock()

# 检查是否支持 RPC 功能，并检查 RPC 是否初始化成功
def is_available() -> bool:
    return hasattr(torch._C, "_rpc_init")

# 如果支持 RPC 但 RPC 未初始化成功，则抛出运行时错误
if is_available() and not torch._C._rpc_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc")

# 如果支持 RPC 功能
if is_available():
    # 导入必要的模块和类
    import numbers
    import torch.distributed.autograd as dist_autograd
    from torch._C._distributed_c10d import Store
    from torch._C._distributed_rpc import (
        # 导入 RPC 相关函数和常量
        _cleanup_python_rpc_handler,
        _DEFAULT_INIT_METHOD,
        _DEFAULT_NUM_WORKER_THREADS,
        _DEFAULT_RPC_TIMEOUT_SEC,
        _delete_all_user_and_unforked_owner_rrefs,
        _destroy_rref_context,
        _disable_jit_rref_pickle,
        _disable_server_process_global_profiler,
        _enable_jit_rref_pickle,
        _enable_server_process_global_profiler,
        _get_current_rpc_agent,
        _invoke_remote_builtin,
        _invoke_remote_python_udf,
        _invoke_remote_torchscript,
        _invoke_rpc_builtin,
        _invoke_rpc_python_udf,
        _invoke_rpc_torchscript,
        _is_current_rpc_agent_set,
        _reset_current_rpc_agent,
        _rref_context_get_debug_info,
        _set_and_start_rpc_agent,
        _set_profiler_node_id,
        _set_rpc_timeout,
        _TensorPipeRpcBackendOptionsBase,
        _UNSET_RPC_TIMEOUT,
        enable_gil_profiling,
        get_rpc_timeout,
        PyRRef,
        RemoteProfilerManager,
        RpcAgent,
        RpcBackendOptions,
        TensorPipeAgent,
        WorkerInfo,
    )
    # 导入自定义模块
    from . import api, backend_registry, functions
    # 导入特定的 API 函数
    from .api import *  # noqa: F401,F403
    # 导入后端注册表
    from .backend_registry import BackendType
    # 导入 TensorPipe RPC 后端选项
    from .options import TensorPipeRpcBackendOptions  # noqa: F401
    # 导入服务器进程全局分析器
    from .server_process_global_profiler import _server_process_global_profile

    # 定义用于会合的迭代器类型
    rendezvous_iterator: Generator[Tuple[Store, int, int], None, None]

    # 扩展公开的模块成员列表
    __all__ += ["init_rpc", "BackendType", "TensorPipeRpcBackendOptions"]
    # 将 API 模块和后端注册表模块中的成员添加到 __all__ 列表中
    __all__ = __all__ + api.__all__ + backend_registry.__all__  # noqa: PLE0605

    # 初始化 RPC 功能
    def init_rpc(
        name,
        backend=None,
        rank=-1,
        world_size=None,
        rpc_backend_options=None,
        ```
        初始化 RPC 环境，配置名称、后端、等级、全局大小和选项
        ```
    # 验证远程过程调用（RPC）的参数是否符合预期类型，抛出异常如果有任何参数类型不匹配
    def _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options):
        # 定义参数与其预期类型的映射关系
        type_mapping = {
            backend: backend_registry.BackendType,  # backend 应该是 BackendType 类型
            store: dist.Store,  # store 应该是 dist.Store 类型
            name: str,  # name 应该是字符串类型
            rank: numbers.Integral,  # rank 应该是整数类型
            # world_size 对于动态组可以是 None
            world_size: (numbers.Integral, type(None)),  # world_size 可以是整数类型或者 None
            rpc_backend_options: RpcBackendOptions,  # rpc_backend_options 应该是 RpcBackendOptions 类型
        }
        # 遍历映射关系，检查每个参数是否符合预期类型
        for arg, arg_type in type_mapping.items():
            if not isinstance(arg, arg_type):  # 如果参数类型不符合预期
                raise RuntimeError(
                    f"Argument {arg} must be of type {arg_type} but got type {type(arg)}"
                )

    # 初始化 RPC 后端
    def _init_rpc_backend(
        backend=BackendType.TENSORPIPE,  # RPC 后端的默认值为 TENSORPIPE
        store=None,  # 存储后端的默认值为 None
        name=None,  # 名称的默认值为 None
        rank=-1,  # 排序的默认值为 -1
        world_size=None,  # 群组大小的默认值为 None
        rpc_backend_options=None,  # RPC 后端选项的默认值为 None
    ):
        # 验证 RPC 初始化的参数
        _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options)

        # 检查当前是否已经初始化了 RPC
        if _is_current_rpc_agent_set():
            raise RuntimeError("RPC is already initialized")

        # 初始化 RPC Agent
        rpc_agent = backend_registry.init_backend(
            backend,
            store=store,
            name=name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # 初始化 RPC 状态
        api._init_rpc_states(rpc_agent)

    # 要求 RPC 已初始化后才能执行的函数装饰器
    @api._require_initialized
    def _get_debug_info():
        # 获取调试信息
        info = _rref_context_get_debug_info()
        # 更新调试信息
        info.update(api._get_current_rpc_agent().get_debug_info())
        info.update(dist_autograd._get_debug_info())
        return info
```