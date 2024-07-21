# `.\pytorch\torch\distributed\_tensor\random.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入上下文管理模块
import contextlib
# 导入警告模块
import warnings
# 导入类型提示模块
from typing import Dict, List, Optional

# 导入PyTorch模块
import torch
# 导入分布式通信模块
import torch.distributed as dist
# 导入张量模块
from torch import Tensor
# 导入分布式张量规格和分片类型
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
# 导入设备网格模块
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh

# 定义全局变量，用于跟踪随机数生成器状态
_rng_tracker: Optional["_RNGStateTracker"] = None


def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    """Checks if the current device of `device_mesh` supports DTensor's random APIs.
    Currently DTensor Random APIs only supports cuda/cuda-like devices. We suggest
    users call this API to test the availability before using our random APIs.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh on which we check if the
            random ops APIs are supported.

    Returns:
        A bool value. True if `device_mesh` supports DTensor Random APIs; False otherwise.

    .. warning::
        Currently we only support correct RNG on cuda/cuda-like devices.
    """
    # 获取设备句柄
    device_handle = _get_device_handle(device_mesh.device_type)
    # 检查设备句柄是否存在并且具有set_rng_state方法
    if device_handle and hasattr(device_handle, "set_rng_state"):
        return True
    else:
        # 如果不支持，发出警告
        warnings.warn(
            f"DTensor random operators may not have complete support on {device_mesh.device_type} device mesh"
        )
        return False


def manual_seed(seed: int, device_mesh: DeviceMesh) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed.

    Returns:
        None

    .. warning::
        When calling this function, :func:`manual_seed` must be called from all ranks of the
        default `ProcessGroup` even if some ranks may not be a part of the `device_mesh`,
        with the same `seed` value.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `manual_seed` will not set its GPU device's generator seed.
        Current implementation only supports a GPU device mesh.
    """
    # 获取设备句柄
    device_handle = _get_device_handle(device_mesh.device_type)
    # 如果设备句柄不存在，抛出异常
    if not device_handle:
        raise NotImplementedError(
            f"DTensor randomness only supports cuda/cuda-like device type, but got {device_mesh.device_type}"
        )

    # 在默认进程组上收集所有进程的种子值
    object_list = [seed] * dist.get_world_size()
    dist.all_gather_object(object_list, seed)
    # 检查收集到的种子值是否一致
    for rank, object in enumerate(object_list):
        if seed != int(object):
            raise RuntimeError(
                f"calling manual_seed function over {device_mesh} but received different seed values on ranks:",
                f"seed on rank {dist.get_rank()} is {seed}, and seed on rank {rank} is {object}!",
            )
    # 如果尚未实例化RNG跟踪器，则实例化之
    # 默认情况下，DTensor使用一个
    # 如果全局变量 _rng_tracker 未定义，初始化为 OffsetBasedRNGTracker 对象，
    # 用于执行随机操作。
    global _rng_tracker
    if not _rng_tracker:
        _rng_tracker = OffsetBasedRNGTracker(device_mesh.device_type)
    
    # 检查设备网格中当前的坐标是否存在
    if device_mesh.get_coordinate() is not None:
        # 如果 _rng_tracker 是 TensorParallelRNGTracker 类型的实例，
        # 则调用其 _manual_seed 方法设置随机种子。
        if isinstance(_rng_tracker, TensorParallelRNGTracker):
            _rng_tracker._manual_seed(device_mesh, seed)
        # 如果 _rng_tracker 是 OffsetBasedRNGTracker 类型的实例，
        # 则调用其 _manual_seed 方法设置随机种子。
        elif isinstance(_rng_tracker, OffsetBasedRNGTracker):
            _rng_tracker._manual_seed(seed)
        else:
            # 如果 _rng_tracker 不是已知类型，则抛出 RuntimeError 异常。
            raise RuntimeError(
                f"Unknown type of cuda RNG state tracker: _rng_tracker = {_rng_tracker}"
            )
class _RNGStateTracker:
    """
    _RNGStateTracker stores Random Number Generator (RNG) state (a ByteTensor object)
    in a dict, mapping from a corresponding tag to each state tensor. It also provides
    a set of convenient utility methods to help access/modify the state tensors. The most
    important interface is _distribute_region which will be used when DTensor executes
    a random op (an operator that calls RNG).
    """

    def __init__(self, device_type: str = "cuda"):
        # 初始化 RNGStateTracker 对象，指定设备类型（默认为 CUDA）
        self._device_type = device_type
        # 获取指定设备类型的设备句柄
        self._device_handle = _get_device_handle(device_type)
        # 如果设备句柄不可用，抛出运行时错误
        if not (self._device_handle and self._device_handle.is_available()):
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of CUDA/CUDA-like device"
            )

        # 初始化 RNG 状态字典
        self._states: Dict[str, Tensor] = {}
        # 记录设备列表，当前设备是当前设备句柄的当前设备
        self._devices = [self._device_handle.current_device()]
        # 启用分布式区域分配标志
        self._use_distribute_region = True

    @property
    def rng_states(self) -> Dict[str, Tensor]:
        # 返回 RNG 状态字典
        return self._states

    @property
    def distribute_region_enabled(self) -> bool:
        # 返回分布式区域分配是否启用的标志
        return self._use_distribute_region

    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value) -> None:
        # 设置分布式区域分配是否启用的标志
        self._use_distribute_region = value

    def rng_state_is_sync(self, name) -> bool:
        # 检查指定名称的 RNG 状态是否已同步
        return name in self.rng_states

    def get_seed(self, name: str) -> int:
        # 获取指定名称的 RNG 种子值
        if name not in self.rng_states:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state for {name}"
            )

        # 获取种子张量，并将其视图转换为 int64 类型后返回
        seed_tensor = (self.rng_states[name])[0:8].view(dtype=torch.int64)
        return int(seed_tensor.item())

    def set_seed(self, name: str, seed: int) -> None:
        # 设置指定名称的 RNG 种子值
        seed_tensor = torch.tensor([seed]).view(torch.uint8)
        offset_tensor = torch.tensor([0]).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

    def _distribute_region(self, spec: DTensorSpec):
        # 留空，用于定义分布式区域分配的具体实现
        pass


class OffsetBasedRNGTracker(_RNGStateTracker):
    """
    This subclass of `_RNGStateTracker` defines the default policy of how RNG states
    should be shared and synchronized among all ranks to respect the semantics of DTensor
    random operators.
    """

    def __init__(self, device_type: str = "cuda"):
        # 调用父类的初始化方法，指定设备类型
        super().__init__(device_type)
        # 使用设备句柄获取当前 RNG 状态，并将其广播给所有进程，然后存储到 RNG 状态字典中
        rng_state = self._device_handle.get_rng_state().to(device_type)
        dist.broadcast(rng_state, 0)
        self.rng_states["parallel-rng"] = rng_state.to("cpu")

    def _manual_seed(self, parallel_seed: int) -> None:
        # 设置并行 RNG 的种子值
        self.set_seed("parallel-rng", parallel_seed)

    @contextlib.contextmanager
    # 分发区域的方法，根据指定的 DTensorSpec 参数进行操作
    def _distribute_region(self, spec: DTensorSpec):
        # 检查并确保并行随机数生成器状态已经同步
        if not self.rng_state_is_sync("parallel-rng"):
            # 如果未同步，抛出运行时错误
            raise RuntimeError(
                "OffsetBasedRNGTracker requires the random state to be synchronized "
                "before entering into a distribute region!"
            )

        # 如果启用了分发区域
        if self.distribute_region_enabled:
            # 获取进入分发区域前的随机数偏移量
            old_offset = self.get_offset("parallel-rng")
            # 设置预操作偏移量
            self._set_pre_op_offset(spec)
            # 在指定设备类型和设备上分叉随机数生成器状态
            with torch.random.fork_rng(self._devices, device_type=self._device_type):
                # 恢复并设置随机数生成器状态为当前对象的并行随机数状态
                self._device_handle.set_rng_state(self.rng_states["parallel-rng"])
                try:
                    yield  # 执行区域内的代码
                finally:
                    # 更新偏移量以在不同进程之间同步
                    self._set_post_op_offset(spec, old_offset)
        else:
            yield  # 如果未启用分发区域，则直接生成器返回

    # 获取指定名称随机数生成器的偏移量
    def get_offset(self, name: str) -> int:
        if name not in self.rng_states:
            # 如果指定名称的随机数状态不存在，抛出运行时错误
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state for {name}"
            )

        # 从随机数状态中获取偏移量张量，并将其视图转换为 int64 类型
        offset_tensor = (self.rng_states[name])[8:].view(dtype=torch.int64)
        return int(offset_tensor.item())

    # 设置指定名称随机数生成器的偏移量
    def set_offset(self, name: str, offset: int) -> None:
        if name not in self.rng_states:
            # 如果指定名称的随机数状态不存在，抛出运行时错误
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state for {name}"
            )

        # 从随机数状态中获取种子张量
        seed_tensor = (self.rng_states[name])[0:8]
        # 创建包含偏移量的 uint8 类型张量，并将其设置到随机数状态中
        offset_tensor = torch.tensor([offset]).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

    # 在执行本地随机操作后，设置随机数生成器为同步状态
    def _set_post_op_offset(self, spec: DTensorSpec, old_offset: int) -> None:
        """Sets the RNG to a synchronized state after running the local random op. Every
        rank should set its RNG offset to `old_offset + DTensor.numel()` where old_offset is
        the offset before calling `set_pre_op_offset` i.e. the offset before running DTensor
        random ops.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we post-process the offset for running random ops.

        Returns:
            None
        """
        # 获取 DTensor 对象的形状
        dtensor_shape = spec.shape

        # 导入 torch 分布式计算库中的 prod 函数
        from torch.distributed._tensor.ops.utils import prod

        # 计算 DTensor 中元素的总数
        numel = prod(dtensor_shape)
        # PyTorch 要求偏移量必须是 4 的倍数，因此将 numel 向上舍入到最接近的 4 的倍数
        numel = (numel + 3) // 4 * 4
        # 设置并更新并行随机数生成器的偏移量
        self.set_offset("parallel-rng", old_offset + numel)

    # 计算分片的线性索引
    def _calc_shard_linear_idx(
        self, shard_coord: List[int], shard_size: List[int]
    ) -> int:
        # 计算分片的线性索引

        # 初始化分片的线性索引为0
        shard_linear_idx = 0
        # 初始化分片坐标步长为1
        shard_coord_stride = 1
        
        # 使用反向遍历的方式计算分片的线性索引
        for idx, size in zip(reversed(shard_coord), reversed(shard_size)):
            # 计算当前维度的分片索引对应的线性偏移量并累加到总索引中
            shard_linear_idx += idx * shard_coord_stride
            # 更新步长，用于计算下一个维度的偏移量
            shard_coord_stride *= size

        # 返回计算得到的分片线性索引
        return shard_linear_idx
class TensorParallelRNGTracker(_RNGStateTracker):
    # 初始化函数，继承自 _RNGStateTracker 类
    def __init__(self, device_type: str = "cuda"):
        super().__init__(device_type)
        # 复制默认的随机数生成器状态
        self.rng_states["tensor-parallel-rng"] = self._device_handle.get_rng_state()

    # 手动设置种子的方法
    def _manual_seed(
        self,
        tp_mesh: DeviceMesh,
        base_seed: int = 1234,
    ):
        # 获取当前 tensor parallel 的排名
        tensor_parallel_rank = tp_mesh.get_local_rank()
        # 这个神奇的数字 2718 来自 Megatron 的代码
        # (https://github.com/NVIDIA/Megatron-LM/blob/060415572f4365a2e895f8036c4e37dad0efbdf5/megatron/core/tensor_parallel/random.py#L162-L163)
        MegatronMagicNum = 2718
        # 计算 tensor parallel 的种子值
        tensor_parallel_seed = base_seed + MegatronMagicNum + tensor_parallel_rank
        self.set_seed("tensor-parallel-rng", tensor_parallel_seed)

    # 上下文管理器，用于进入分布区域
    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec):
        # 检查 tensor parallel 的随机数状态是否已同步
        if not self.rng_state_is_sync("tensor-parallel-rng"):
            # 若未同步，则抛出运行时错误
            raise RuntimeError(
                "TensorParallelRNGTracker requires the random state to be synchronized "
                "before entering into a distribute region!"
            )

        # 如果启用了分布区域
        if self.distribute_region_enabled:
            with torch.random.fork_rng(self._devices, device_type=self._device_type):
                # 设置 RNG 状态为 tensor-parallel-rng 对应的状态
                self._device_handle.set_rng_state(
                    self.rng_states["tensor-parallel-rng"]
                )
                try:
                    yield
                finally:
                    # 更新 RNG 状态为当前设备的状态
                    self.rng_states[
                        "tensor-parallel-rng"
                    ] = self._device_handle.get_rng_state()
        else:
            # 如果未启用分布区域，则直接执行 yield
            yield
```