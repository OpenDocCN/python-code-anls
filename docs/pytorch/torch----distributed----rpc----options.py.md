# `.\pytorch\torch\distributed\rpc\options.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型定义
from typing import Dict, List, Optional, Union

# 导入 torch 库
import torch
# 导入 Torch 分布式 RPC 的后端选项基类
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase

# 导入本地的 constants 模块作为 rpc_constants
from . import constants as rpc_constants

# 定义 DeviceType 为可以是整数、字符串或 torch.device 类型的联合类型
DeviceType = Union[int, str, torch.device]

# 导出的类名列表，只包含 TensorPipeRpcBackendOptions
__all__ = ["TensorPipeRpcBackendOptions"]


# 将给定的 device 转换为 torch.device 对象
def _to_device(device: DeviceType) -> torch.device:
    device = torch.device(device)
    # 如果设备类型不是 "cuda"，抛出 ValueError 异常
    if device.type != "cuda":
        raise ValueError(
            "`set_devices` expect a list of CUDA devices, but got "
            f"device type {device.type}."
        )
    return device


# 将 device_map 字典中的键和值都转换为 torch.device 对象的映射
def _to_device_map(
    device_map: Dict[DeviceType, DeviceType]
) -> Dict[torch.device, torch.device]:
    # 定义两个空的设备映射字典
    full_device_map: Dict[torch.device, torch.device] = {}
    reverse_map: Dict[torch.device, torch.device] = {}
    # 遍历 device_map 中的每对键值
    for k, v in device_map.items():
        # 将键和值都转换为 torch.device 对象
        k, v = torch.device(k), torch.device(v)
        # 检查是否已经存在相同值的映射
        if v in reverse_map:
            raise ValueError(
                "`device_map` only supports 1-to-1 mapping, "
                f"trying to map {k} and {reverse_map[v]} to {v}"
            )
        # 更新完整的设备映射和反向映射
        full_device_map[k] = v
        reverse_map[v] = k
    return full_device_map


# 将 devices 列表中的每个元素转换为 torch.device 对象的列表
def _to_device_list(devices: List[DeviceType]) -> List[torch.device]:
    return list(map(_to_device, devices))


# TensorPipeRpcBackendOptions 类继承自 _TensorPipeRpcBackendOptionsBase 类
class TensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    r"""
    :class:`~torch.distributed.rpc.TensorPipeAgent` 的后端选项，
    派生自 :class:`~torch.distributed.rpc.RpcBackendOptions`。
    """
    """
    初始化一个基于TensorPipe的RPC代理对象。
    
    Args:
        num_worker_threads (int, optional): 用于执行请求的线程池中的线程数
            (:class:`~torch.distributed.rpc.TensorPipeAgent`的默认值为16)。
        rpc_timeout (float, optional): RPC请求的默认超时时间，单位为秒
            (默认为60秒)。如果在此时间内RPC未完成，将引发超时异常。
            调用者可以在单个RPC中覆盖此超时时间，使用
            :meth:`~torch.distributed.rpc.rpc_sync` 和
            :meth:`~torch.distributed.rpc.rpc_async`。
        init_method (str, optional): 初始化用于约会的分布式存储的URL。
            接受与 :meth:`~torch.distributed.init_process_group` 相同参数的任意值
            (默认为 ``env://``)。
        device_maps (Dict[str, Dict], optional): 从此工作进程到被调用者的设备放置映射。
            键是被调用者的工作进程名称，值是字典(``Dict``)，包含了将此工作进程的设备映射到
            被调用者工作进程的设备的映射关系。
            (默认为 ``None``)
        devices (List[int, str, or ``torch.device``], optional): RPC代理使用的所有本地CUDA设备。
            默认情况下，它将从自己的 ``device_maps`` 和对等方的 ``device_maps`` 初始化到所有本地设备。
            在处理CUDA RPC请求时，代理将为此 ``List`` 中的所有设备适当同步CUDA流。
            (默认为 ``None``)
    
    """
    
    def __init__(
        self,
        *,
        num_worker_threads: int = rpc_constants.DEFAULT_NUM_WORKER_THREADS,
        rpc_timeout: float = rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
        init_method: str = rpc_constants.DEFAULT_INIT_METHOD,
        device_maps: Optional[Dict[str, Dict[DeviceType, DeviceType]]] = None,
        devices: Optional[List[DeviceType]] = None,
        _transports: Optional[List] = None,
        _channels: Optional[List] = None,
    ):
        """
        构造函数，初始化TensorPipe RPC代理对象。
    
        full_device_maps = (
            {} if device_maps is None
            else {k: _to_device_map(v) for k, v in device_maps.items()}
        )
        如果 `device_maps` 为 `None`，则 `full_device_maps` 初始化为空字典；
        否则将 `device_maps` 中的每个项 `_to_device_map` 转换为设备映射，并存入 `full_device_maps`。
    
        full_device_list = [] if devices is None else _to_device_list(devices)
        如果 `devices` 为 `None`，则 `full_device_list` 初始化为空列表；
        否则将 `devices` 中的每个设备项 `_to_device_list` 转换为设备列表，并存入 `full_device_list`。
    
        调用父类的构造函数，传递以下参数初始化：
            - num_worker_threads: 线程池中的线程数
            - _transports: 传输方式列表
            - _channels: 通道列表
            - rpc_timeout: RPC请求超时时间
            - init_method: 分布式存储的初始化URL
            - full_device_maps: 完整的设备映射
            - full_device_list: 完整的设备列表
        """
        super().__init__(
            num_worker_threads,
            _transports,
            _channels,
            rpc_timeout,
            init_method,
            full_device_maps,
            full_device_list,
        )
    def set_device_map(self, to: str, device_map: Dict[DeviceType, DeviceType]):
        r"""
        Set device mapping between each RPC caller and callee pair. This
        function can be called multiple times to incrementally add
        device placement configurations.

        Args:
            to (str): Callee name.
            device_map (Dict of int, str, or torch.device): Device placement
                mappings from this worker to the callee. This map must be
                invertible.

        Example:
            >>> # xdoctest: +SKIP("distributed")
            >>> # both workers
            >>> def add(x, y):
            >>>     print(x)  # tensor([1., 1.], device='cuda:1')
            >>>     return x + y, (x + y).to(2)
            >>>
            >>> # on worker 0
            >>> options = TensorPipeRpcBackendOptions(
            >>>     num_worker_threads=8,
            >>>     device_maps={"worker1": {0: 1}}
            >>>     # maps worker0's cuda:0 to worker1's cuda:1
            >>> )
            >>> options.set_device_map("worker1", {1: 2})
            >>> # maps worker0's cuda:1 to worker1's cuda:2
            >>>
            >>> rpc.init_rpc(
            >>>     "worker0",
            >>>     rank=0,
            >>>     world_size=2,
            >>>     backend=rpc.BackendType.TENSORPIPE,
            >>>     rpc_backend_options=options
            >>> )
            >>>
            >>> x = torch.ones(2)
            >>> rets = rpc.rpc_sync("worker1", add, args=(x.to(0), 1))
            >>> # The first argument will be moved to cuda:1 on worker1. When
            >>> # sending the return value back, it will follow the invert of
            >>> # the device map, and hence will be moved back to cuda:0 and
            >>> # cuda:1 on worker0
            >>> print(rets[0])  # tensor([2., 2.], device='cuda:0')
            >>> print(rets[1])  # tensor([2., 2.], device='cuda:1')
        """
        # 获取完整的设备映射字典
        full_device_map = _to_device_map(device_map)
        # 获取当前设备映射字典
        curr_device_maps = super().device_maps

        # 如果目标 'to' 在当前设备映射中已存在
        if to in curr_device_maps:
            # 遍历新的设备映射字典
            for k, v in full_device_map.items():
                # 如果当前 'to' 的设备映射中已经存在键 k，并且对应的值 v 不等于当前映射中的值
                if k in curr_device_maps[to] and v != curr_device_maps[to][k]:
                    # 抛出数值错误，说明只支持一对一映射
                    raise ValueError(
                        "`set_device_map` only supports 1-to-1 mapping, trying"
                        f" to map {k} to {v} and {curr_device_maps[to][k]}"
                    )

        # 调用父类方法来设置设备映射
        super()._set_device_map(to, full_device_map)

    def set_devices(self, devices: List[DeviceType]):
        r"""
        Set local devices used by the TensorPipe RPC agent. When processing
        CUDA RPC requests, the TensorPipe RPC agent will properly synchronize
        CUDA streams for all devices in this ``List``.

        Args:
            devices (List of int, str, or torch.device): local devices used by
                the TensorPipe RPC agent.
        """
        # 将输入的设备列表转换为设备列表
        self.devices = _to_device_list(devices)
```