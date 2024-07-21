# `.\pytorch\torch\distributed\rpc\backend_registry.py`

```py
# mypy: allow-untyped-defs

# 导入必要的模块和库
import collections
import enum
from typing import cast, Dict, List, Set, Tuple

import torch
import torch.distributed as dist

# 导入自定义模块和常量
from . import api, constants as rpc_constants
from ._utils import _group_membership_management, _update_group_membership

# 声明公开的接口名称列表
__all__ = [
    "backend_registered",
    "register_backend",
    "construct_rpc_backend_options",
    "init_backend",
    "BackendValue",
    "BackendType",
]

# 定义一个命名元组类型，表示RPC后端的值
BackendValue = collections.namedtuple(
    "BackendValue", ["construct_rpc_backend_options_handler", "init_backend_handler"]
)

# 定义一个函数，用于返回BackendType枚举成员的字符串表示
def _backend_type_repr(self):
    return "BackendType." + self.name

# BackendType的文档字符串
_backend_type_doc = """
    An enum class of available backends.

    PyTorch ships with a builtin ``BackendType.TENSORPIPE`` backend.
    Additional ones can be registered using the
    :func:`~torch.distributed.rpc.backend_registry.register_backend` function.
"""

# 创建一个枚举类BackendType，初始时没有成员
BackendType = enum.Enum(value="BackendType", names=dict())  # type: ignore[misc]
# 为了解决mypy报告的问题，忽略对函数的Enum API和方法分配
BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]

# 如果BackendType已有文档字符串，则将其设置为_backend_type_doc
if BackendType.__doc__:
    BackendType.__doc__ = _backend_type_doc

# 检查给定的后端名称是否已注册为RPC后端
def backend_registered(backend_name):
    """
    Checks if backend_name is registered as an RPC backend.

    Args:
        backend_name (str): string to identify the RPC backend.
    Returns:
        True if the backend has been registered with ``register_backend``, else
        False.
    """
    return backend_name in BackendType.__members__.keys()

# 注册一个新的RPC后端
def register_backend(
    backend_name, construct_rpc_backend_options_handler, init_backend_handler
):
    """Registers a new RPC backend.

    Args:
        backend_name (str): backend string to identify the handler.
        construct_rpc_backend_options_handler (function):
            Handler that is invoked when
            rpc_backend.construct_rpc_backend_options(**dict) is called.
        init_backend_handler (function): Handler that is invoked when the
            `_init_rpc_backend()` function is called with a backend.
             This returns the agent.
    """
    global BackendType
    # 如果已注册相同的后端名称，则引发运行时错误
    if backend_registered(backend_name):
        raise RuntimeError(f"RPC backend {backend_name}: already registered")
    # 获取当前BackendType的所有成员，并扩展新的后端类型字典
    existing_enum_dict = {member.name: member.value for member in BackendType}
    extended_enum_dict = dict(
        {
            backend_name: BackendValue(
                construct_rpc_backend_options_handler=construct_rpc_backend_options_handler,
                init_backend_handler=init_backend_handler,
            )
        },
        **existing_enum_dict,
    )
    # 为了解决mypy报告的问题，忽略对函数的Enum API和方法分配
    BackendType = enum.Enum(value="BackendType", names=extended_enum_dict)  # type: ignore[misc]
    # 将 _backend_type_repr 方法赋值给 BackendType 类的 __repr__ 方法，忽略类型检查错误 # type: ignore[assignment]
    BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]
    # 如果 BackendType 类有文档字符串，则将其替换为 _backend_type_doc 方法返回的文档字符串
    if BackendType.__doc__:
        BackendType.__doc__ = _backend_type_doc
    # 返回 BackendType 中指定名称的成员
    return BackendType[backend_name]
# 构建 RPC 后端选项的函数，使用给定的后端对象和参数
def construct_rpc_backend_options(
    backend,
    rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
    init_method=rpc_constants.DEFAULT_INIT_METHOD,
    **kwargs,
):
    # 调用后端对象的方法来构建 RPC 后端选项
    return backend.value.construct_rpc_backend_options_handler(
        rpc_timeout, init_method, **kwargs
    )


# 初始化后端的函数，使用给定的后端对象和参数
def init_backend(backend, *args, **kwargs):
    # 调用后端对象的方法来初始化后端
    return backend.value.init_backend_handler(*args, **kwargs)


# 初始化进程组的函数，使用给定的存储对象、排名和世界大小
def _init_process_group(store, rank, world_size):
    # 设置进程组超时时间为默认值
    process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT

    # 使用 Gloo 后端创建进程组对象
    group = dist.ProcessGroupGloo(store, rank, world_size, process_group_timeout)

    # 断言确保进程组对象已成功初始化
    assert group is not None, "Failed to initialize default ProcessGroup."

    # 检查排名是否与进程组的排名匹配
    if (rank != -1) and (rank != group.rank()):
        raise RuntimeError(f"rank argument {rank} doesn't match pg rank {group.rank()}")
    
    # 检查世界大小是否与进程组的大小匹配
    if (world_size != -1) and (world_size != group.size()):
        raise RuntimeError(
            f"world_size argument {world_size} doesn't match pg size {group.size()}"
        )
    
    # 返回初始化后的进程组对象
    return group


# 使用 TensorPipe 后端构建 RPC 后端选项的处理函数，使用给定的超时、初始化方法和其他可选参数
def _tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads=rpc_constants.DEFAULT_NUM_WORKER_THREADS,
    _transports=None,
    _channels=None,
    **kwargs,
):
    # 导入 TensorPipeRpcBackendOptions 类
    from . import TensorPipeRpcBackendOptions

    # 创建 TensorPipeRpcBackendOptions 对象并返回
    return TensorPipeRpcBackendOptions(
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        num_worker_threads=num_worker_threads,
        _transports=_transports,
        _channels=_channels,
    )


# 验证给定设备列表中的设备是否有效的函数
def _tensorpipe_validate_devices(devices, device_count):
    # 检查所有设备是否是有效的 CPU 或者 CUDA 设备
    return all(
        d.type == "cpu" or (d.type == "cuda" and 0 <= d.index < device_count)
        for d in devices
    )


# 检测是否有任何 worker 的设备映射配置无效，并返回反向设备映射和本地处理的设备列表
def _tensorpipe_exchange_and_check_all_device_maps(
    my_name, my_device_count, my_device_maps, my_devices, group
):
    # 初始化一个空的列表用于收集所有 worker 的名称、设备数量、设备映射和设备列表
    gathered: List[
        Tuple[str, int, Dict[str, Dict[torch.device, torch.device]], List[torch.device]]
    ] = [("", 0, {}, []) for _ in range(group.size())]

    # 使用分布式通信库的 all_gather_object 方法收集所有 worker 的设备映射相关信息
    dist.all_gather_object(
        gathered, (my_name, my_device_count, my_device_maps, my_devices), group
    )

    # 提取所有收集到的 worker 的名称、设备数量、设备映射和设备列表
    all_names = [name for name, _, _, _ in gathered]
    all_device_counts = {name: count for name, count, _, _ in gathered}
    all_device_maps = {name: map_ for name, _, map_, _ in gathered}
    all_devices = {name: devices for name, _, _, devices in gathered}

    # 验证所有收集到的设备映射配置是否有效
    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices)

    # 生成反向设备映射，并获取本地处理的设备列表
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)
    my_devices = _create_device_list(my_devices, my_device_maps, reverse_device_maps)

    # 返回反向设备映射和本地处理的设备列表
    return reverse_device_maps, my_devices
# 验证设备映射的有效性，确保设备没有重复，且索引有效
def _validate_device_maps(
    all_names, all_device_counts, all_device_maps, all_devices, is_static_group=True
):
    # 遍历所有节点名称
    for node in all_names:
        # 获取节点对应的设备列表
        devices = all_devices[node]
        # 检查设备列表是否存在重复的设备
        if len(set(devices)) != len(devices):
            # 如果存在重复设备，则抛出数值错误异常
            raise ValueError(
                f"Node {node} has duplicated devices\n" f"devices = {devices}"
            )
        # 调用函数验证设备的索引是否有效
        if not _tensorpipe_validate_devices(devices, all_device_counts[node]):
            # 如果设备索引无效，则抛出数值错误异常
            raise ValueError(
                f"Node {node} has devices with invalid indices\n"
                f"devices = {devices}\n"
                f"device count = {all_device_counts[node]}"
            )
    for source_node in all_names:
        # 对于动态组（非静态），不需要检查目标节点名称，因为可能尚未加入
        if is_static_group and not set(all_device_maps[source_node].keys()).issubset(
            all_names
        ):
            # 如果是静态组且设备映射中的目标节点名称不在所有节点名中，则抛出异常
            raise ValueError(
                f"Node {source_node} has invalid target node names in its device maps\n"
                f"device maps = {all_device_maps[source_node].keys()}\n"
                f"node names = {all_names}"
            )
        for target_node, map_ in all_device_maps[source_node].items():
            if len(set(map_.values())) != len(map_):
                # 如果设备映射中目标设备有重复，则抛出异常
                raise ValueError(
                    f"Node {source_node} has duplicated target devices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}"
                )
            if all_devices[source_node]:
                if not set(map_.keys()).issubset(all_devices[source_node]):
                    # 如果设备映射中的源设备不在允许的设备列表中，则抛出异常
                    raise ValueError(
                        f"Node {source_node} has unexpected source devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[source_node]}"
                    )
            elif not _tensorpipe_validate_devices(
                map_.keys(), all_device_counts[source_node]
            ):
                # 如果设备映射中的源设备索引无效，则抛出异常
                raise ValueError(
                    f"Node {source_node} has source devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[source_node]}"
                )
            if all_devices.get(target_node, []):
                if not set(map_.values()).issubset(all_devices[target_node]):
                    # 如果设备映射中的目标设备不在允许的设备列表中，则抛出异常
                    raise ValueError(
                        f"Node {source_node} has unexpected target devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[target_node]}"
                    )
            elif target_node in all_device_counts and not _tensorpipe_validate_devices(
                map_.values(), all_device_counts[target_node]
            ):
                # 如果设备映射中的目标设备索引无效，则抛出异常
                raise ValueError(
                    f"Node {source_node} has target devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[target_node]}"
                )
def _create_device_list(my_devices, my_device_maps, reverse_device_maps):
    # 如果没有指定自己的设备列表，则初始化一个空集合
    if not my_devices:
        devices_set: Set[torch.device] = set()
        # 遍历所有设备映射表，更新设备集合
        for map_ in my_device_maps.values():
            devices_set.update(map_.keys())
        # 遍历所有反向设备映射表，更新设备集合
        for map_ in reverse_device_maps.values():
            devices_set.update(map_.keys())
        # 移除集合中的 CPU 设备
        devices_set.discard(torch.device("cpu"))
        # 将集合转换为列表并按设备索引排序
        my_devices = list(devices_set)
    # 按设备索引排序返回设备列表
    my_devices = sorted(my_devices, key=lambda d: d.index)
    return my_devices


def _create_reverse_mapping(my_name, all_names, all_device_maps):
    # 初始化空的反向设备映射字典
    reverse_device_maps: Dict[str, Dict[torch.device, torch.device]] = {}
    # 遍历所有节点名称
    for node in all_names:
        # 如果当前节点包含目标名称在其设备映射表中
        if my_name in all_device_maps[node]:
            # 构建当前节点到目标名称的反向设备映射
            reverse_device_maps[node] = {
                v: k for k, v in all_device_maps[node][my_name].items()
            }
    return reverse_device_maps


def _get_device_infos():
    from . import TensorPipeAgent

    # 从当前 API 获取 TensorPipeAgent 实例，并转换类型为 TensorPipeAgent
    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    # 获取当前 RPC 代理的后端选项
    opts = agent._get_backend_options()
    # 获取 CUDA 设备数量
    device_count = torch.cuda.device_count()
    # 如果 CUDA 可用且存在后端选项的设备列表，则初始化 CUDA
    if torch.cuda.is_available() and opts.devices:
        torch.cuda.init()
    return device_count, opts.device_maps, opts.devices


def _set_devices_and_reverse_device_map(agent):
    from . import TensorPipeAgent

    # 将传入的代理对象强制类型转换为 TensorPipeAgent 类型
    agent = cast(TensorPipeAgent, agent)
    # 从代理对象中获取当前工作节点的信息
    my_worker_info = agent.get_worker_info()
    my_name = my_worker_info.name
    # 获取所有工作节点的信息列表
    all_worker_infos = agent.get_worker_infos()
    # 初始化空字典和列表来存储设备相关信息
    all_device_counts, all_device_maps, all_devices, all_names = {}, {}, {}, []
    # 遍历所有工作节点的信息
    for worker_info in all_worker_infos:
        worker_name = worker_info.name
        # 如果当前工作节点不是自己
        if worker_name != my_name:
            # TODO: make async? (待改进：是否异步处理)
            # 通过 RPC 同步调用获取远程工作节点的设备信息
            device_count, device_map, devices = api.rpc_sync(
                worker_name, _get_device_infos
            )
        else:
            # 如果是自己，则直接从当前代理对象获取设备信息
            opts = agent._get_backend_options()
            device_count, device_map, devices = (
                torch.cuda.device_count(),
                opts.device_maps,
                opts.devices,
            )
        # 将获取到的设备数量、设备映射和设备列表存储到相应的字典中
        all_device_counts[worker_name] = device_count
        all_device_maps[worker_name] = device_map
        all_devices[worker_name] = devices
        # 将工作节点名称添加到名称列表中
        all_names.append(worker_name)

    # 对所有设备映射进行验证，确保一致性和正确性
    _validate_device_maps(
        all_names,
        all_device_counts,
        all_device_maps,
        all_devices,
        is_static_group=False,
    )
    # 创建当前节点到所有节点的反向设备映射
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)

    # 向所有工作节点（包括自身）发起 RPC 调用，以包含新加入的工作节点信息和设备映射
    # 遍历所有工作人员的名称列表
    for worker_name in all_names:
        # 为每个工作人员设置设备列表
        all_devices[worker_name] = _create_device_list(
            all_devices[worker_name], all_device_maps[worker_name], reverse_device_maps
        )
        # 使用 RPC 同步调用更新组成员资格
        api.rpc_sync(
            worker_name,
            _update_group_membership,
            args=(my_worker_info, all_devices[worker_name], reverse_device_maps, True),
        )
# 定义一个函数用于初始化 TensorPipe 后端处理程序，接受多个参数：存储对象、名称、当前进程的等级、总进程数、RPC 后端选项
def _tensorpipe_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    # 导入 TensorPipeAgent 和 TensorPipeRpcBackendOptions 类
    from . import TensorPipeAgent, TensorPipeRpcBackendOptions

    # 检查存储对象是否为 c10d::Store 类型，如果不是则抛出类型错误异常
    if not isinstance(store, dist.Store):
        raise TypeError(f"`store` must be a c10d::Store. {store}")

    # 检查 RPC 后端选项是否为 TensorPipeRpcBackendOptions 类型，如果不是则抛出类型错误异常
    if not isinstance(rpc_backend_options, TensorPipeRpcBackendOptions):
        raise TypeError(
            f"`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`. {rpc_backend_options}"
        )

    # 获取当前可用的 CUDA 设备数量
    device_count = torch.cuda.device_count()

    # 判断是否为静态群组，如果 world_size 不为 None 则为静态群组
    is_static_group = True if world_size else False
    # world_size 被指定，因此这是一个静态群组（等级不能加入或离开）
    if is_static_group:
        # 创建一个进程组，用于代理的加入方法，需要类似于屏障和执行集体操作的过程组，而不是在 RPC 上重新实现这些功能
        group = _init_process_group(store, rank, world_size)

        # 交换和检查所有设备映射，获取反向设备映射和设备列表
        reverse_device_maps, devices = _tensorpipe_exchange_and_check_all_device_maps(
            name,
            device_count,
            rpc_backend_options.device_maps,
            rpc_backend_options.devices,
            group,
        )

        # 如果 CUDA 可用且存在设备列表，则在此初始化 PyTorch CUDA 状态
        if torch.cuda.is_available() and devices:
            # 必须在此处初始化 PyTorch CUDA 状态（例如，CUDACachingAllocator）。如果缺少此步骤，
            # 可能会出现“allocator not initialized”等错误，因为其他进程可能在用户代码初始化其
            # PyTorch CUDA 状态之前向该进程发送与 CUDA 相关的 RPC 请求。
            torch.cuda.init()

        # 创建 TensorPipeAgent 实例，传入所需参数和设备信息
        agent = TensorPipeAgent(
            store,
            name,
            rank,
            world_size,
            rpc_backend_options,
            reverse_device_maps,
            devices,
        )

        # 初始化 RPC 状态
        api._init_rpc_states(agent)

        # 执行一轮虚拟的 RPC，以初始化通道/传输。如果没有此操作，如果在 rpc.shutdown() 之前没有
        # 其他 RPC 请求到达该进程，则很容易超时，因为代理的初始化可能需要超过 5 秒。
        api._all_gather(None, timeout=rpc_backend_options.rpc_timeout)
        # 在这里需要一个屏障以确保在等级 0 完成 _all_gather 之前没有同行离开
        group.barrier().wait()

        # 返回代理对象
        return agent
    # 动态 RPC 初始化（等级可以加入和离开）
    else:
        with _group_membership_management(store, name, True):
            # 使用 _group_membership_management 函数处理组成员管理，确保在此过程中进入临界区
            # 创建一个 TensorPipeAgent 对象，使用给定参数初始化，其中 reverse_device_map 和 devices 初始为空
            # 这些属性在初始化后将被更新
            agent = TensorPipeAgent(
                store,
                name,
                rank,
                world_size,
                rpc_backend_options,
                {},
                [],
            )
            # 初始化 RPC 状态
            api._init_rpc_states(agent)

            try:
                # 通知组内所有工作节点，此节点已加入，并设置设备和反向设备映射
                # 这是一个同步操作，直到所有现有节点的更新完成
                _set_devices_and_reverse_device_map(agent)
                pass
            except Exception:
                # 出现异常时，调用 API 关闭 RPC
                api.shutdown()
                raise
            # 返回创建的 agent 对象
            return agent
# 注册后端，将 "TENSORPIPE" 字符串作为后端的标识
# 使用 _tensorpipe_construct_rpc_backend_options_handler 函数作为构造 RPC 后端选项处理程序
# 使用 _tensorpipe_init_backend_handler 函数作为初始化后端处理程序
register_backend(
    "TENSORPIPE",
    _tensorpipe_construct_rpc_backend_options_handler,
    _tensorpipe_init_backend_handler,
)
```