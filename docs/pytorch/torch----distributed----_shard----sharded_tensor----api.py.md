# `.\pytorch\torch\distributed\_shard\sharded_tensor\api.py`

```
# mypy: allow-untyped-defs
from __future__ import annotations  # type: ignore[attr-defined]

import copy  # 导入copy模块，用于对象的深拷贝操作
import operator  # 导入operator模块，提供了一些Python内置运算符的函数实现
import threading  # 导入threading模块，支持多线程编程
import warnings  # 导入warnings模块，用于警告控制
import weakref  # 导入weakref模块，支持弱引用对象
from dataclasses import dataclass  # 导入dataclass装饰器，用于声明数据类
from functools import reduce  # 导入functools模块的reduce函数，用于对可迭代对象进行累积计算
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import deprecated  # 导入deprecated装饰器，用于标记过时的函数或类

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
import torch.distributed._shard.sharding_spec as shard_spec  # 导入分片规范模块
from torch.distributed import distributed_c10d, rpc  # 导入分布式C10d和RPC模块
from torch.distributed._shard._utils import DEPRECATE_MSG  # 导入分片工具模块中的DEPRECATE_MSG
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,  # 导入检查张量函数
    validate_non_overlapping_shards_metadata,  # 导入验证非重叠分片元数据函数
)
from torch.distributed._shard.sharding_spec.api import (
    _dispatch_custom_op,  # 导入调度自定义操作函数
    _has_custom_op,  # 导入判断是否有自定义操作函数
)
from torch.distributed.remote_device import _remote_device  # 导入远程设备模块中的_remote_device函数
from torch.utils import _pytree as pytree  # 导入_pytree模块，支持树形数据结构操作

from .metadata import ShardedTensorMetadata, TensorProperties  # 导入元数据和张量属性类
from .reshard import reshard_local_shard, reshuffle_local_shard  # 导入重新分片函数
from .shard import Shard  # 导入分片类
from .utils import (
    _flatten_tensor_size,  # 导入张量大小展平函数
    _parse_and_validate_remote_device,  # 导入解析和验证远程设备函数
    _validate_output_tensor_for_gather,  # 导入验证聚合输出张量函数
    build_global_metadata,  # 导入构建全局元数据函数
    build_metadata_from_local_shards,  # 导入从本地分片构建元数据函数
)


if TYPE_CHECKING:
    from torch.distributed._shard.metadata import ShardMetadata  # 导入分片元数据类

# 线程锁，用于保护共享资源_sharded_tensor_map
_sharded_tensor_lock = threading.Lock()

# 当前分片张量的唯一标识符计数器
_sharded_tensor_current_id = 0

# 映射，存储分片张量对象的弱引用，键为分片张量的唯一标识符，值为对分片张量的弱引用
_sharded_tensor_map: Dict[int, weakref.ReferenceType[ShardedTensor]] = {}

# 默认分片操作的字典
_SHARDED_OPS: Dict[Callable, Callable] = {}

# 自定义用户分片操作的字典
_CUSTOM_SHARDED_OPS: Dict[Callable, Callable] = {}


def _register_remote_shards(
    sharded_tensor_id: int, rrefs: List[rpc.RRef[Shard]], rpc_rank: int
):
    """
    注册远程分片到分片张量对象中。

    Args:
        sharded_tensor_id (int): 分片张量对象的唯一标识符
        rrefs (List[rpc.RRef[Shard]]): 包含分片的远程引用列表
        rpc_rank (int): RPC进程的排名
    """
    with _sharded_tensor_lock:
        # 检查_sharded_tensor_map中是否存在给定的分片张量ID
        if sharded_tensor_id not in _sharded_tensor_map:
            raise RuntimeError(
                f"Could not find sharded_tensor_id: {sharded_tensor_id} in map: {_sharded_tensor_map.keys()}"
            )

        # 从_sharded_tensor_map中获取分片张量对象的弱引用
        sharded_tensor = _sharded_tensor_map[sharded_tensor_id]()
        if sharded_tensor is None:
            # 如果分片张量对象的弱引用已经被释放，则抛出运行时错误
            raise RuntimeError("ShardedTensor weakref has been deallocated")
        else:
            # 否则，调用分片张量对象的方法注册远程分片
            sharded_tensor._register_remote_shards(rrefs, rpc_rank)


class ShardedTensorBase(torch.Tensor):
    """
    分片张量的基类，继承自PyTorch的张量类torch.Tensor。
    """
    _sharding_spec: shard_spec.ShardingSpec  # 分片规范对象
    _metadata: ShardedTensorMetadata  # 分片张量的元数据
    _local_shards: List[Shard]  # 本地分片列表
    def __new__(cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs):
        """
        Use __new__ to construct a wrapper tensor, for recording tensor
        properties and logging purposes.
        """
        # 记录 API 使用情况，一次性记录到日志中
        torch._C._log_api_usage_once("torch.distributed._shard.sharded_tensor")

        # 检查分片规格并构建分布式张量元数据
        if not isinstance(sharding_spec, shard_spec.ShardingSpec):
            raise ValueError(f"Expecting ShardingSpec but got: {type(sharding_spec)}")

        sizes = _flatten_tensor_size(size)
        dtype = kwargs["dtype"]
        layout = kwargs["layout"]
        pin_memory = kwargs["pin_memory"]
        requires_grad = kwargs["requires_grad"]

        if dtype is None:
            dtype = torch.get_default_dtype()

        # 创建张量属性对象
        tensor_properties = TensorProperties(
            dtype, layout, requires_grad, pin_memory=pin_memory
        )
        # 基于分片规格构建分布式张量元数据
        sharded_tensor_metadata = sharding_spec.build_metadata(
            sizes, tensor_properties=tensor_properties
        )

        # 创建张量的包装子类
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            sizes,
            dtype=dtype,
            layout=layout,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        # 设置分片规格
        r._sharding_spec = sharding_spec
        # 设置元数据
        r._metadata = sharded_tensor_metadata
        # 设置本地分片
        r._local_shards = []
        return r

    def metadata(self) -> ShardedTensorMetadata:
        """
        Returns a :class:`ShardedTensorMetadata` object corresponding to the
        metadata for the entire tensor.
        """
        return self._metadata

    def local_shards(self) -> List[Shard]:
        """
        Returns a list of :class:`Shard` corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    @classmethod
    def _init_from_local_shards_and_global_metadata(
        cls,
        local_shards: List[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        sharding_spec=None,
    ) -> ShardedTensorBase:
        """
        Initialize a ShardedTensorBase with local shards and a global
        ShardedTensorMetadata built on each rank.
        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        # 从输入参数 sharded_tensor_metadata 中获取分片元数据和张量属性
        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        # 如果 shards_metadata 为空，抛出异常
        if len(shards_metadata) == 0:
            raise ValueError("shards_metadata must not be empty!")

        # 检查张量布局是否为 torch.strided，否则抛出异常
        if tensor_properties.layout != torch.strided:
            raise ValueError("Only torch.strided layout is currently supported")

        # 根据输入的 sharding_spec 或者根据 shards_metadata 推断分片规格
        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        # 创建 ShardedTensorBase 对象
        sharded_tensor_base = ShardedTensorBase.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # 检查 shards_metadata 是否有重叠的分片
        validate_non_overlapping_shards_metadata(shards_metadata)

        # 检查 shards_metadata 是否与分片张量的整体大小兼容
        check_tensor(shards_metadata, list(sharded_tensor_metadata.size))

        # 完成验证，将 local_shards 添加到 sharded_tensor_base 中
        sharded_tensor_base._local_shards = local_shards
        return sharded_tensor_base

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 抛出运行时异常，提示缺少 __torch_dispatch__ 方法的实现
        raise RuntimeError(
            f"A {cls.__name__} object is being used from c++ while calling {func.__module__}.{func.__name__} "
            "but there is no custom __torch_dispatch__ implementation for it."
        )
# 定义了一个名为 ShardedTensor 的类，继承自 ShardedTensorBase 类
class ShardedTensor(ShardedTensorBase):
    """
    ShardedTensor is an torch.Tensor subclass to represent Tensors that are sharded
    across multiple devices and multiple processes.

    ShardedTensor 是 torch.Tensor 的子类，用于表示跨多个设备和多个进程分片的张量。

    ShardedTensor is initialized in an SPMD like fashion where each rank
    initializes the ShardedTensor. The ShardedTensor object on each rank
    then only stores the local shard for the Tensor and provides global
    metadata for all the shards.

    ShardedTensor 被初始化为 SPMD 样式，每个等级初始化一个 ShardedTensor。
    每个等级上的 ShardedTensor 对象只存储张量的本地分片，并提供所有分片的全局元数据。

    ShardedTensor doesn't provide any Tensor like operations but is a wrapper
    providing the Tensor representing the local shard and the global metadata.
    Using these, users can build their custom distributed._sharded computations
    on top of this primitive. The local shards are all initialized using the
    create_op specified by tensor_init_params.create_op, e.g., torch.ones, or
    torch.empty

    ShardedTensor 不提供任何类似张量的操作，而是提供一个包装器，包含表示本地分片的张量和全局元数据。
    使用这些，用户可以在这个基本结构上构建他们自己的分布式 _sharded 计算。
    所有本地分片都使用 tensor_init_params.create_op 指定的 create_op 进行初始化，例如 torch.ones 或 torch.empty

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    .. note:: ShardedTensor uses collectives to do various operations, i.e. it
        uses all_gather to do cross rank validations. For NCCL-based process
        groups, internal tensor representations of objects must be moved to the
        GPU device before communication takes place. In this case, the device
        used is given by ``torch.cuda.current_device()`` and it is the user's
        responsibility to ensure that this is set so that each rank has an
        individual GPU, via ``torch.cuda.set_device()``

    """

    def __new__(cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs):
        # 调用父类的 __new__ 方法来创建新的实例
        self = super().__new__(cls, sharding_spec, *size, **kwargs)
        return self
    def __init__(
        self,
        sharding_spec: shard_spec.ShardingSpec,
        *size,
        dtype=None,
        layout=torch.strided,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
        process_group=None,
        init_rrefs=False,
    ):
        """
        构造函数，初始化一个分布式张量对象。

        Args:
            sharding_spec (shard_spec.ShardingSpec): 分片规格对象，描述如何分片张量。
            *size: 张量的大小参数。
            dtype: 张量数据类型。
            layout: 张量的布局，默认为 torch.strided。
            requires_grad: 是否需要梯度。
            pin_memory: 是否将张量存储在固定内存中。
            memory_format: 张量的内存格式，默认为 torch.contiguous_format。
            process_group: 进程组对象，用于分布式通信。
            init_rrefs: 是否初始化远程引用对象。
        """
        # 准备初始化工作，初始化诸如 _process_group、_local_shards 等字段。
        self._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # 检查布局是否为 torch.strided，否则抛出异常。
        if layout != torch.strided:
            raise ValueError("Only torch.strided layout is currently supported")

        # 检查内存格式是否为 torch.contiguous_format，否则抛出异常。
        if memory_format != torch.contiguous_format:
            raise ValueError(
                "Only torch.contiguous_format memory_format is currently supported"
            )

        # 设置张量属性中的内存格式。
        self._metadata.tensor_properties.memory_format = memory_format

        # 获取全局进程的排名。
        current_rank = dist.get_rank()  # global rank

        # 遍历分片元数据列表。
        for shard_metadata in self._metadata.shards_metadata:
            # 解析和验证远程设备的排名和设备。
            rank, device = _parse_and_validate_remote_device(
                self._process_group, shard_metadata.placement
            )
            # 如果排名与当前进程的排名相同，则创建本地张量。
            if rank == current_rank:
                local_tensor = _create_tensor_from_params(
                    shard_metadata.shard_sizes,
                    local_device=device,
                    tensor_properties=self._metadata.tensor_properties,
                )
                # 将本地分片添加到 _local_shards 列表中。
                self._local_shards.append(Shard(local_tensor, shard_metadata))

        # 进行后续初始化工作（如注册分布式张量 ID，初始化 RPC）。
        self._post_init()

    def _prepare_init(self, process_group=None, init_rrefs=False):
        """
        准备初始化工作，设置初始状态和属性。

        Args:
            process_group: 进程组对象，用于分布式通信。
            init_rrefs: 是否初始化远程引用对象。
        """
        self._init_rrefs = init_rrefs
        self._sharded_tensor_id = None

        # 规范化处理进程组对象。
        self._process_group = self._normalize_pg(process_group)
        # 存储远程分片的字典。
        self._remote_shards: Dict[int, List[rpc.RRef[Shard]]] = {}

    def _post_init(self):
        """
        后续初始化工作，处理特定的后续操作，如初始化 RPC。
        """
        # 如果设置了 init_rrefs，则初始化 RPC。
        if self._init_rrefs:
            with _sharded_tensor_lock:
                global _sharded_tensor_current_id, _sharded_tensor_map
                # 设置分布式张量 ID。
                self._sharded_tensor_id = _sharded_tensor_current_id
                # 将当前对象添加到全局映射中。
                _sharded_tensor_map[self._sharded_tensor_id] = weakref.ref(self)
                _sharded_tensor_current_id += 1

            # 如果当前 RPC 代理未初始化，则抛出运行时异常。
            if not rpc._is_current_rpc_agent_set():
                raise RuntimeError(
                    "RPC Framework needs to be initialized using"
                    " torch.distributed.rpc.init_rpc if init_rrefs is set to True"
                )
            # 初始化 RPC。
            self._init_rpc()

    def __del__(self):
        """
        析构函数，清理对象的全局映射。
        """
        with _sharded_tensor_lock:
            global _sharded_tensor_current_id, _sharded_tensor_map
            # 如果存在 _sharded_tensor_id 并且其存在于映射中，则将其移除。
            if (
                hasattr(self, "_sharded_tensor_id")
                and self._sharded_tensor_id in _sharded_tensor_map
            ):
                _sharded_tensor_map.pop(self._sharded_tensor_id)  # type: ignore[call-overload]
    def _init_rpc(self):
        # Validate PG and RPC ranks match.
        pg_rank = dist.get_rank()  # 获取当前进程组的排名
        rpc_rank = rpc.get_worker_info().id  # 获取当前 RPC worker 的排名
        if pg_rank != rpc_rank:  # 检查进程组排名和 RPC 排名是否匹配
            raise ValueError(
                f"Default ProcessGroup and RPC ranks must be "
                f"the same for ShardedTensor, found process group rank: "
                f"{pg_rank} and RPC rank: {rpc_rank}"
            )

        self._remote_shards = {}  # 初始化远程分片字典

        # Gather all the sharded tensor ids.
        worker_infos = rpc._get_current_rpc_agent().get_worker_infos()  # 获取当前 RPC agent 的 worker 信息
        rank_to_name = {}  # 创建排名到名称的映射字典
        name_to_rank = {}  # 创建名称到排名的映射字典

        for worker_info in worker_infos:
            rank_to_name[worker_info.id] = worker_info.name  # 填充排名到名称的映射字典
            name_to_rank[worker_info.name] = worker_info.id  # 填充名称到排名的映射字典

        all_tensor_ids = rpc.api._all_gather(self._sharded_tensor_id)  # 收集所有分布式张量的 ID

        # Share the local shards to the entire world.
        futs = []  # 初始化 futures 列表
        rpc_rank = rpc.get_worker_info().id  # 获取当前 RPC worker 的排名
        for rank in range(dist.get_world_size()):  # 遍历所有的进程组排名
            # Skip self.
            if rank == dist.get_rank():  # 跳过当前进程组的排名
                continue

            if len(self.local_shards()) != 0:  # 如果本地分片不为空
                rrefs: List[rpc.RRef[Shard]] = [
                    rpc.RRef(shard) for shard in self.local_shards()  # 创建本地分片的远程引用列表
                ]
                fut = rpc.rpc_async(
                    rank,
                    _register_remote_shards,  # 远程注册分片函数的调用
                    args=(all_tensor_ids[rank_to_name[rank]], rrefs, rpc_rank),  # 参数包括分布式张量 ID 和远程引用列表
                )
                futs.append(fut)  # 将 future 对象添加到 futures 列表中

        torch.futures.wait_all(futs)  # 等待所有 future 对象完成

        # Barrier for all RPCs to finish on all ranks.
        rpc.api._all_gather(None)  # 执行全局同步等待所有 RPC 完成

    def _get_preferred_device(self) -> torch.device:
        """
        Return the preferred device to be used when creating tensors for collectives.
        This method takes into account the associated process group
        """
        if dist.get_backend(self._process_group) == dist.Backend.NCCL:  # 如果使用 NCCL 后端
            return torch.device(torch.cuda.current_device())  # 返回当前 CUDA 设备
        return torch.device("cpu")  # 默认返回 CPU 设备

    def gather(  # type: ignore[override]
        self,
        dst: int = 0,
        out: Optional[torch.Tensor] = None,
        enforce_dtype: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        # Method to gather tensors from different ranks to a destination rank.
        ...

    def cpu(
        self, memory_format=torch.preserve_format, process_group=None
    ):
        # Method to move tensors to CPU.
        ...
    ) -> ShardedTensor:
        """
        Returns a copy of this object in CPU memory.

        If this ShardedTensor is already on CPU memory, then no copy is
        performed and the original object is returned.

        .. note:: When moving a ShardedTensor from GPU to CPU, the ShardedTensor might
            need to be managed by a different type of ProcessGroup (i.e., ProcessGroupGloo),
            it is the user's responsibility to explicitly pass in a new process_group that
            is compatible with CPU.
        """
        # TODO: make this a __torch_function__ op once ShardedTensor becomes a
        # torch.Tensor subclass, see https://github.com/pytorch/pytorch/issues/75402
        
        # Check if the memory format is either `torch.preserve_format` or `torch.contiguous_format`
        if (
            memory_format != torch.preserve_format
            and memory_format != torch.contiguous_format
        ):
            raise RuntimeError(
                "Only `torch.contiguous_format` or "
                "`torch.preserve_format` is supported!"
            )
        
        # Check if all shards of the ShardedTensor are already on CPU
        all_on_cpu = True
        for meta in self.metadata().shards_metadata:
            all_on_cpu &= meta.placement.device().type == "cpu"  # type: ignore[union-attr]

        # if every shard is already on CPU, return the original object
        if all_on_cpu:
            return self

        # if not, return a copy of this object on CPU
        list_shards: List[Shard] = []
        # move all local shards to CPU, and change metadata accordingly
        for shard in self._local_shards:
            cpu_tensor = shard.tensor.cpu(memory_format=memory_format)  # type: ignore[call-arg]
            metadata = copy.deepcopy(shard.metadata)
            metadata.placement._device = torch.device("cpu")  # type: ignore[union-attr]
            list_shards.append(Shard(cpu_tensor, metadata))

        # Update the metadata for the entire ShardedTensor to reflect CPU placement
        st_meta = copy.deepcopy(self.metadata())
        for meta in st_meta.shards_metadata:
            if meta.placement.device().type != "cpu":  # type: ignore[union-attr]
                meta.placement._device = torch.device("cpu")  # type: ignore[union-attr]

        # Determine the process group to use for the CPU version
        pg = self._process_group if process_group is None else process_group

        # Initialize a new ShardedTensor instance from local CPU shards and updated metadata
        st_cpu = ShardedTensor._init_from_local_shards_and_global_metadata(
            list_shards,
            sharded_tensor_metadata=st_meta,
            process_group=pg,
            init_rrefs=self._init_rrefs,
        )
        return st_cpu
    ) -> ShardedTensor:
        """
        Returns a copy of this object in CUDA memory, if the original ShardedTensor
        is on CPU, we will move the local shard to the current GPU device of each
        process in a SPMD fashion.
        If this ShardedTensor is already on CUDA memory and local shards on each rank are
        already on current device, we still returns a new ShardedTensor object with new
        metadata, but no underlying data movements are performed.
        .. note:: When moving a ShardedTensor from CPU to GPU, the ShardedTensor might
            need to be managed by a different type of ProcessGroup(i.e. ProcessGroupNCCL),
            it is the user's responsiblity to explicitly pass in a new process_group that
            is compatible with GPU.
        """
        # 检查内存格式是否为支持的格式，只支持连续格式或保留格式
        if (
            memory_format != torch.preserve_format
            and memory_format != torch.contiguous_format
        ):
            raise RuntimeError(
                "Only `torch.contiguous_format` or "
                "`torch.preserve_format` is supported!"
            )

        # 如果指定了设备，则将其转换为torch.device对象，确保设备是当前CUDA设备
        if device is not None:
            device = torch.device(device) if isinstance(device, str) else device
            assert (
                isinstance(device, torch.device)
                and device.index == torch.cuda.current_device()
            ), """Only device without device id (e.g. "cpu" or "cuda") is expected for ShardedTensor!"""

        # 获取当前CUDA设备
        current_device = torch.device(torch.cuda.current_device())
        # 返回一个空的Shard列表
        list_shards: List[Shard] = []
        
        # 遍历本地shard，将每个shard移到当前设备，并更新元数据
        # 如果本地shard已经在当前设备上，则只复制元数据，不进行实际数据移动
        for shard in self._local_shards:
            cuda_tensor = shard.tensor.cuda(
                device=current_device,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )  # type: ignore[call-arg]
            metadata = copy.deepcopy(shard.metadata)
            metadata.placement._device = current_device  # type: ignore[union-attr]

            list_shards.append(Shard(cuda_tensor, metadata))

        # 深拷贝ShardedTensor的元数据
        st_meta = copy.deepcopy(self.metadata())
        # 更新所有shards的元数据，确保它们的设备为当前CUDA设备
        for meta in st_meta.shards_metadata:
            if meta.placement.device().type != "cuda":  # type: ignore[union-attr]
                meta.placement._device = current_device  # type: ignore[union-attr]

        # 如果未提供process_group，则使用self._process_group
        pg = self._process_group if process_group is None else process_group
        # 使用init_from_local_shards方法通信并更新sharding spec和shards metadata
        st_cuda = ShardedTensor._init_from_local_shards_and_global_metadata(
            list_shards,
            sharded_tensor_metadata=st_meta,
            process_group=pg,
            init_rrefs=self._init_rrefs,
        )
        # 返回位于CUDA上的新的ShardedTensor对象
        return st_cuda
    @classmethod
    # 静态方法：用于规范化处理组对象。如果未提供处理组对象，则返回默认处理组对象。
    def _normalize_pg(
        cls, process_group: Optional[dist.ProcessGroup]
    ) -> dist.ProcessGroup:
        if process_group is not None:
            return process_group
        return distributed_c10d._get_default_group()

    @classmethod
    # 静态方法：从本地碎片初始化分布式张量。
    def _init_from_local_shards(
        cls,
        local_shards: List[Shard],
        *global_size,
        process_group=None,
        init_rrefs=False,
    ):
        # STEP 1: Validate the Shardmetadatas locally
        # 步骤1：本地验证碎片元数据
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # 获取全局排名

        # 获取处理组中的全局大小
        world_size = dist.get_world_size(process_group)

        local_sharded_tensor_metadata: Optional[ShardedTensorMetadata] = None
        global_tensor_size = _flatten_tensor_size(global_size)

        if len(local_shards) > 0:
            # 从本地碎片构建本地分片张量元数据
            local_sharded_tensor_metadata = build_metadata_from_local_shards(
                local_shards, global_tensor_size, current_rank, process_group
            )

        # STEP 2. Validate metadata across ranks, and build a global sharded tensor
        # metadata by gathering local ShardedTensorMetadata
        # 步骤2：跨排名验证元数据，并通过收集本地ShardedTensorMetadata构建全局分片张量元数据
        gathered_metadatas: List[Optional[ShardedTensorMetadata]] = []
        if world_size > 1:
            gathered_metadatas = [None for _ in range(world_size)]

            # 使用dist.all_gather_object收集本地分片张量元数据
            dist.all_gather_object(
                gathered_metadatas, local_sharded_tensor_metadata, group=process_group
            )
        else:
            gathered_metadatas = [local_sharded_tensor_metadata]

        # 构建全局元数据
        global_sharded_tensor_metadata = build_global_metadata(gathered_metadatas)
        tensor_properties = global_sharded_tensor_metadata.tensor_properties

        # STEP 3: Validation done, create the actual ShardedTensor and populate fields
        # prepare initialization
        # 步骤3：验证完成，创建实际的ShardedTensor并填充字段，准备初始化
        spec = shard_spec._infer_sharding_spec_from_shards_metadata(
            global_sharded_tensor_metadata.shards_metadata
        )
        sharded_tensor = cls.__new__(
            cls,
            spec,
            global_sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # attach local_shards to the ShardedTensor created
        # 将本地碎片附加到创建的ShardedTensor上
        sharded_tensor._local_shards = local_shards

        # run post initialization, i.e. map registration, rpc initialization
        # 运行后初始化，例如映射注册，rpc初始化
        sharded_tensor._post_init()
        return sharded_tensor

    @classmethod
    @deprecated(DEPRECATE_MSG, category=FutureWarning)
    # 静态方法（已弃用）：从本地张量初始化分片张量。
    def _init_from_local_tensor(
        cls,
        local_tensor: torch.Tensor,
        sharding_spec: shard_spec.ShardingSpec,
        *global_size: Sequence[int],
        process_group: Optional[dist.ProcessGroup] = None,
        init_rrefs=False,
    ):
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: List[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ):
        """
        Initializes a sharded tensor from local shards and global metadata.

        Args:
            cls: Class reference (implicitly passed).
            local_shards: List of local shards composing the sharded tensor.
            sharded_tensor_metadata: Metadata describing the sharded tensor.
            process_group: Optional process group for collective operations.
            init_rrefs: Flag indicating whether to initialize remote references.
            sharding_spec: Optional specification for sharding the tensor.
        """

    def sharding_spec(self) -> shard_spec.ShardingSpec:
        """
        Returns the ShardingSpec object associated with the tensor.
        """
        return self._sharding_spec

    @deprecated(DEPRECATE_MSG, category=FutureWarning)
    def local_tensor(self) -> torch.Tensor:
        """
        Returns the local tensor of the sharded tensor. Currently supports only a single local shard.

        Returns:
            A :class:`torch.Tensor` representing the local shard.
        """
        if len(self.local_shards()) != 1:
            raise NotImplementedError("Only single local shard is supported.")
        return self.local_shards()[0].tensor

    @classmethod
    @deprecated(DEPRECATE_MSG, category=FutureWarning)
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Overrides the torch function behavior for ShardedTensor.

        Args:
            cls: Class reference (implicitly passed).
            func: Function being called.
            types: Types associated with the function call.
            args: Positional arguments for the function call.
            kwargs: Keyword arguments for the function call.

        Returns:
            Result of the dispatched function call on ShardedTensor instance.
        """
        def dispatch(st: ShardedTensor, func: Callable):
            # Dispatch to custom user provided op first if it exists.
            if func in _CUSTOM_SHARDED_OPS:
                return _CUSTOM_SHARDED_OPS[func](types, args, kwargs, st._process_group)

            # Dispatch to custom sharding spec op if it has one.
            if _has_custom_op(st._sharding_spec, func):
                return _dispatch_custom_op(
                    st._sharding_spec, func, types, args, kwargs, st._process_group
                )

            if func in _SHARDED_OPS:
                return _SHARDED_OPS[func](types, args, kwargs, st._process_group)

            raise RuntimeError(
                f"torch function '{func.__name__}', with args: {args} and "
                f"kwargs: {kwargs} not supported for ShardedTensor!"
            )

        # Find ShardedTensor instance to get process_group and sharding_spec.
        st_instance = None

        def find_sharded_tensor(e):
            nonlocal st_instance
            if st_instance is None and isinstance(e, ShardedTensor):
                st_instance = e

        pytree.tree_map_(find_sharded_tensor, args)
        pytree.tree_map_(find_sharded_tensor, kwargs)

        if st_instance is not None:
            return dispatch(st_instance, func)

        raise RuntimeError(
            f"torch function '{func.__name__}', with args: {args} and "
            f"kwargs: {kwargs} not supported for ShardedTensor!"
        )

    def is_pinned(self) -> bool:  # type: ignore[override]
        """
        Returns True if the sharded tensor (each local shard) resides in pinned memory.
        """
        return self._metadata.tensor_properties.pin_memory

    def _register_remote_shards(
        self, remote_shards: List[rpc.RRef[Shard]], rpc_rank: int
    ):
        """
        Registers remote shards for a specific RPC rank.

        Args:
            remote_shards: List of remote shards represented as RRefs.
            rpc_rank: Rank of the RPC process where the shards reside.
        """
        self._remote_shards[rpc_rank] = remote_shards
    # 返回远程分片的字典，键为RPC排名，值为该排名上分片的RRef列表
    def remote_shards(self) -> Dict[int, List[rpc.RRef[Shard]]]:
        """
        Returns a Dict[int, RRef] with keys being the RPC rank and values
        being RRefs to shards on that rank. Need to initialize the
        RPC framework for this functionality.

        Raises an exception if ShardedTensor was created with ``init_rrefs=False``
        """
        if not self._init_rrefs:
            # 如果 ShardedTensor 是以 init_rrefs=False 创建的，则抛出运行时异常
            raise RuntimeError(
                "ShardedTensor created with init_rrefs=False, no RRefs to remote shards available"
            )
        # 返回存储在实例变量 _remote_shards 中的远程分片字典
        return self._remote_shards

    # 定义对象的哈希值方法
    def __hash__(self):
        return id(self)

    # 定义对象的字符串表示形式方法
    def __repr__(self):
        return f"ShardedTensor({self._metadata})"

    # 定义数据类，用于存储进程组状态的序列化和反序列化
    @dataclass
    class ProcessGroupState:
        """
        State for ser-de of process group
        """

        local_rank: int
        global_rank: int
        local_world_size: int
        global_world_size: int

    # 定义对象的序列化方法
    def __getstate__(self):
        # 创建 ProcessGroupState 实例，存储当前进程组的状态信息
        pg_state = ShardedTensor.ProcessGroupState(
            distributed_c10d.get_rank(self._process_group),
            distributed_c10d.get_rank(),  # 获取当前进程在全局中的排名
            distributed_c10d.get_world_size(self._process_group),
            distributed_c10d.get_world_size(),  # 获取当前进程组的全局世界大小
        )

        # 返回当前对象的状态信息，包括本地分片、元数据、进程组状态、分片策略和初始化标志
        return (
            self._local_shards,
            self._metadata,
            pg_state,
            self._sharding_spec,
            self._init_rrefs,
        )
    # 定义 __setstate__ 方法，用于反序列化对象状态
    def __setstate__(self, state):
        # 初始化 _sharded_tensor_id 属性为 None
        self._sharded_tensor_id = None
        
        # 检查是否已初始化分布式进程组，如果没有则抛出运行时错误
        if not distributed_c10d.is_initialized():
            raise RuntimeError(
                "Need to initialize default process group using "
                '"init_process_group" before loading ShardedTensor'
            )

        # 解包传入的状态，分别赋值给各个属性
        (
            self._local_shards,
            self._metadata,
            pg_state,
            self._sharding_spec,
            self._init_rrefs,
        ) = state

        # 导入 _get_current_process_group 函数并调用，设置当前进程组
        from torch.distributed._shard.api import _get_current_process_group
        self._process_group = _get_current_process_group()

        # 验证本地进程在保存时和加载时的本地排名是否一致
        local_rank = distributed_c10d.get_rank(self._process_group)
        if pg_state.local_rank != local_rank:
            raise RuntimeError(
                f"Local rank at save time was {pg_state.local_rank}, but at "
                f"load time was {local_rank}"
            )

        # 验证全局进程在保存时和加载时的全局排名是否一致
        global_rank = distributed_c10d.get_rank()
        if pg_state.global_rank != global_rank:
            raise RuntimeError(
                f"Global rank at save time was {pg_state.global_rank}, but at "
                f"load time was {global_rank}"
            )

        # 验证本地进程组在保存时和加载时的本地世界大小是否一致
        local_world_size = distributed_c10d.get_world_size(self._process_group)
        if pg_state.local_world_size != local_world_size:
            raise RuntimeError(
                f"Local world size at save time was {pg_state.local_world_size}, "
                f"but at load time was {local_world_size}"
            )

        # 验证全局世界大小在保存时和加载时的全局世界大小是否一致
        global_world_size = distributed_c10d.get_world_size()
        if pg_state.global_world_size != global_world_size:
            raise RuntimeError(
                f"Global world size at save time was {pg_state.global_world_size}, "
                f"but at load time was {global_world_size}"
            )

        # 调用 _post_init 方法，完成对象状态的后续初始化
        self._post_init()
# 根据给定参数创建一个张量
def _create_tensor_from_params(
    *size, local_device, tensor_properties: TensorProperties
):
    """Helper to construct tensor from size, device and common params."""
    # 从参数中获取张量的数据类型
    dtype = tensor_properties.dtype
    # 从参数中获取张量的布局
    layout = tensor_properties.layout
    # 从参数中获取张量是否需要梯度信息
    requires_grad = tensor_properties.requires_grad
    # 从参数中获取张量的内存格式
    memory_format = tensor_properties.memory_format
    # 从参数中获取张量是否需要被固定到内存中
    pin_memory = tensor_properties.pin_memory

    # 使用torch.empty函数创建一个未初始化的张量
    return torch.empty(
        *size,
        dtype=dtype,
        layout=layout,
        device=local_device,
        requires_grad=requires_grad,
        memory_format=memory_format,
        pin_memory=pin_memory,
    )
```