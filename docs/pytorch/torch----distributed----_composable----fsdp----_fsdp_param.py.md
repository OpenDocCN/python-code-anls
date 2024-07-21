# `.\pytorch\torch\distributed\_composable\fsdp\_fsdp_param.py`

```
"""
导入所需的模块和库
"""
# mypy: allow-untyped-defs
import itertools  # 导入 itertools 模块，用于高效的迭代操作
from dataclasses import dataclass, field  # 导入 dataclass 和 field 用于定义数据类
from enum import auto, Enum  # 导入 auto 和 Enum 用于定义枚举类型
from typing import Any, cast, List, Optional, Sequence, Tuple  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 深度学习框架
import torch._dynamo.compiled_autograd as ca  # 导入 compiled_autograd 模块
import torch.nn as nn  # 导入神经网络模块
from torch._prims_common import make_contiguous_strides_for  # 导入 prims_common 模块中的函数
from torch.distributed._functional_collectives import AsyncCollectiveTensor  # 导入分布式相关模块
from torch.distributed._tensor import DTensor, Replicate, Shard  # 导入分布式张量相关模块
from torch.distributed._tensor.device_mesh import _mesh_resources  # 导入设备网格相关模块
from torch.distributed._tensor.placement_types import DTensorSpec, Placement, TensorMeta  # 导入张量放置相关模块

from ._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy  # 导入 FSDP API 相关模块
from ._fsdp_common import (  # 导入 FSDP 公共函数和类
    _chunk_with_empty,
    _from_local_no_grad,
    _get_dim0_chunked_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    FSDPMeshInfo,
    HSDPMeshInfo,
)

"""
FSDP tensors 的说明和注释
"""

lib = torch.library.Library("fsdp", "FRAGMENT")  # 创建一个名为 "fsdp" 的库对象，标记为 FRAGMENT 类型  # noqa: TOR901

lib.define("set_(Tensor(a!) tensor, Tensor data) -> ()")  # 在库对象上定义一个名为 "set_" 的函数，接受两个张量参数并无返回值


@torch.library.impl(lib, "set_", "Meta")  # 在 "set_" 函数上定义一个 Meta 类型的实现
@torch.library.impl(lib, "set_", "CUDA")  # 在 "set_" 函数上定义一个 CUDA 类型的实现
# 使用装饰器定义了一个 Torch 库的实现函数，名称为 `set_`，作用于 CPU 设备
@torch.library.impl(lib, "set_", "CPU")
def set_(tensor, data):
    # 调用 Tensor 对象的 `set_` 方法，用给定数据 `data` 来设置 `tensor` 的值
    tensor.set_(data)


"""
[Note: Avoiding functionalization for fsdp.set_ and inductor.resize_storage_bytes_(0)]

Currently we don't functionalize `fsdp.set_` op or `inductor.resize_storage_bytes_(0)` op
(i.e. they show up as a mutation op in the middle of the AOT joint graph).

Reason:
Traceable FSDP2 compiled autograd BWD graph have the following traits:
(1) Two inputs of the graph were aliased to each other (one from hook closed-over tensors, one from FWD saved tensors).
(2) One of them is mutated (set_ and resize_(0) to handle the all-gathered param).
(3) They are both subclasses.
The combination of these traits is not supported by AOTAutograd (it's difficult to reason about subclass aliasing).
So this doesn't work at all for Traceable FSDP2.

The compromise we use is to avoid functionalization for the FSDP2 set_ and resize_(0) ops.
This avoids the problem above, because from AOTAutograd point-of-view there are no mutations
that functionalization needs to handle. (Although we need to be careful not to DCE those mutable ops.)

We can avoid this functionalization because:
(1) The nn.Parameter is never used before its .set_() is called in eager code (i.e. no alias of it is created),
so it's safe to call .set_() in the middle of the graph to swap out its storage and start using the nn.Parameter downstream.
(2) We always re-allocate the buffer for nn.Parameter to store the AllGather output and to be used in downstream user ops.
So calling resize-to-0 in the middle of the graph to free nn.Parameter memory after use should always be okay
(since we always allocate anew next time we need it, we strictly don't need to keep the old tensor storage around anymore).

Q: But doesn't the torch.compile stack have the "functional graph" assumption in many places?
A: Yes - this is WIP but we will try to get back to functional graph as early as possible in the lowering process.
Specifically, we believe we can move both .set_ and .resize_(0) ops to end of graph in AOT joint graph before partitioner
(i.e. effectively "re-functionalizing" those ops). Put it in another way, we avoid functionalization for those two ops just to
make AOTAutograd alias analysis happy, and as soon as we are past that point, we "re-functionalize" the graph.
This requires a custom FX pass but we believe it's not hard to write and maintain.

Q: What's the importance of partitioner not saving views of nn.Parameter as FWD saved tensors?
A: This is critical: we do want to save FWD nn.Parameter graph input (instead of its view) for BWD use,
so that downstream ops in BWD graph uses the post-`.set_` nn.Parameter instead of any of its saved views as input.
This is because .set_ will not update any of the nn.Parameter's views, so BWD downstream ops must use the original
nn.Parameter in order to see the result of .set_.
"""


@torch.library.impl(lib, "set_", "Functionalize")
def set__functionalize(tensor, data):
    # 定义一个 Torch 库的实现函数，名称为 `set_`，但这次是用于 Functionalize 模式
    # 该函数可能会对 `tensor` 进行函数化处理，使得其在处理时更符合功能图的需求
    # 调用 torch 模块的 _sync 函数，同步传入的张量 tensor
    torch._sync(tensor)
    # 调用 torch 模块的 _sync 函数，同步传入的数据 data
    torch._sync(data)
    # 使用 torch 模块的 _from_functional_tensor 函数，创建一个新的张量 tensor_inner，以便从函数性张量 tensor 中获取数据
    tensor_inner = torch._from_functional_tensor(tensor)
    # 使用 torch 模块的 _from_functional_tensor 函数，创建一个新的数据 data_inner，以便从函数性数据 data 中获取数据
    data_inner = torch._from_functional_tensor(data)
    # 将 data_inner 的数据复制到 tensor_inner 中，用于替换 tensor_inner 的数据内容，同时忽略类型检查
    tensor_inner.set_(data_inner)  # type: ignore[call-overload]
class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``SHARDED_POST_FORWARD``: The unsharded parameter is resharded to a
      smaller world size. Since this data should not be used for computation,
      we do not register it to the module. Users should reshard the module
      before any in-place modifications. Both it and the sharded parameter
      contribute to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """

    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
    UNSHARDED = auto()
    # 定义枚举类型，表示参数在模块中的不同分片状态，每种状态对模型参数内存有不同贡献


@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``. We additionally save
    shared modules and shared parameter names to update them accordingly.
    """

    # Parameter names are unprefixed, e.g. "weight", not "lin.weight"
    module: nn.Module
    param_name: str
    shared_modules: List[nn.Module] = field(default_factory=list)
    shared_param_names: List[str] = field(default_factory=list)
    # 存储模块和参数名，支持通过 setattr(module, param_name, ...) 进行参数交换或者
    # 通过 getattr(module, param_name) 获取参数。额外保存共享模块和参数名以便进行相应更新。


@dataclass
class ExtensionsData:
    # User-defined metadata passed from pre to post-all-gather
    all_gather_metadata: Optional[Any] = None
    # Save the all-gather input sizes to unflatten the all-gather outputs to ND
    all_gather_input_sizes: Sequence[torch.Size] = ()  # ND

    def clear(self):
        self.all_gather_metadata = None
        self.all_gather_input_sizes = ()
    # 用于存储从预处理到全聚合后传递的用户定义元数据。
    # 存储全聚合的输入大小以便将全聚合输出展开为多维张量（ND）。
    # 提供清除方法用于重置数据。


class FSDPParam:
    """
    This class manages a parameter with FSDP or FSDP variants applied,
    implementing dim-0 per-parameter sharding.
    """

    orig_dtype: torch.dtype
    param_dtype: Optional[torch.dtype]
    reduce_dtype: Optional[torch.dtype]
    _orig_size: torch.Size  # ND
    sharded_size: torch.Size  # ND
    contiguous_sharded_stride: Tuple[int, ...]
    padded_sharded_param_size: torch.Size  # ND
    sharded_post_forward_size: torch.Size  # ND
    contiguous_sharded_post_forward_stride: Tuple[int, ...]
    _sharded_param_data: torch.Tensor  # 1D
    sharded_param: nn.Parameter  # ND
    _sharded_post_forward_param_data: Optional[torch.Tensor]  # 1D
    _sharded_post_forward_param: Optional[nn.Parameter]  # ND
    _unsharded_param: nn.Parameter  # ND
    unsharded_accumulated_grad: Optional[torch.Tensor]  # ND
    _sharding_spec: DTensorSpec
    # DTensor attributes (only defined for DTensor `param`):
    _tp_spec: DTensorSpec
    all_gather_outputs: List[torch.Tensor]  # 1D
    # All-gather extension attributes
    _extensions_data: ExtensionsData
    _unsharded_inner_tensors: List[torch.Tensor]
    # 管理应用了 FSDP 或其变体的参数的类。
    # 实现基于参数的零维分片（dim-0 per-parameter sharding）。
    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ):
        self._module_info: ParamModuleInfo = module_info  # 存储模块信息
        self.mesh_info = mesh_info  # 存储网格信息
        self.post_forward_mesh_info = post_forward_mesh_info  # 存储后向传播网格信息
        self.device = device  # 存储设备信息
        self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)  # 根据 offload_policy 判断是否需要向 CPU offload
        self.pin_memory = (
            self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
        )  # 如果需要向 CPU offload，则存储 pin_memory 的设置
        self.grad_offload_event: Optional[torch.cuda.Event] = None  # 初始化梯度 offload 的事件为 None
        self._init_sharded_param(param, device)  # 初始化分片参数
        if self.post_forward_mesh_info:  # 如果存在后向传播网格信息
            self._init_sharded_post_forward_param_metadata(param)  # 初始化后向传播参数的元数据
        self._init_extensions()  # 初始化扩展
        self.all_gather_outputs: List[torch.Tensor] = []  # 初始化用于聚合的输出列表为空列表
        self.unsharded_accumulated_grad = None  # 初始化未分片的累积梯度为 None
        self._param_fqn: Optional[str] = None  # 初始化参数全限定名为 None，从根模块前缀
        # TODO: Remove this padding logic once DTensor pads the local tensor:
        # https://github.com/pytorch/pytorch/issues/113045
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()  # 注册加载状态字典后钩子，用于重置分片参数
            )
        )

    @torch.no_grad()
    def _init_sharded_post_forward_param_metadata(self, param: torch.Tensor) -> None:
        mesh_info = self.post_forward_mesh_info  # 获取后向传播网格信息
        assert mesh_info is not None  # 确保后向传播网格信息不为 None
        param_data = param._local_tensor if isinstance(param, DTensor) else param  # 如果参数是 DTensor，则获取其本地张量，否则使用原参数
        chunks = _chunk_with_empty(param_data, mesh_info.shard_mesh_size, dim=0)  # 使用空值分块函数对参数数据进行分块处理
        self.sharded_post_forward_size = _get_dim0_chunked_size(
            chunks[mesh_info.shard_mesh_rank], param_data.size()
        )  # 获取分片后向传播尺寸
        self.contiguous_sharded_post_forward_stride = make_contiguous_strides_for(
            self.sharded_post_forward_size
        )  # 生成连续的分片后向传播步长

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)  # 获取混合精度策略的参数类型和减少类型
        self.orig_dtype = self.sharded_param.dtype  # 存储原始的参数类型
        # Clamp `param_dtype` to `None` if no casting is required
        if param_dtype == self.orig_dtype:  # 如果参数类型等于原始类型，则将 param_dtype 设置为 None
            param_dtype = None
        self.param_dtype = param_dtype  # 存储参数类型
        self.reduce_dtype = reduce_dtype  # 存储减少类型
        # None indicates that the mixed precision is not enabled  # None 表示未启用混合精度
    # 初始化扩展属性，检查是否有fsdp_pre_all_gather和fsdp_post_all_gather属性
    inner_tensor = self._sharded_local_tensor
    has_fsdp_pre_all_gather = hasattr(inner_tensor, "fsdp_pre_all_gather")
    has_fsdp_post_all_gather = hasattr(inner_tensor, "fsdp_post_all_gather")
    
    # 如果fsdp_pre_all_gather和fsdp_post_all_gather属性不一致，抛出断言错误
    if has_fsdp_pre_all_gather != has_fsdp_post_all_gather:
        raise AssertionError(
            "Both fsdp_pre_all_gather and fsdp_post_all_gather should be defined "
            f"if using all-gather extensions: {inner_tensor}"
        )
    
    # 如果有fsdp_pre_all_gather属性
    if has_fsdp_pre_all_gather:
        # 检查参数的大小是否符合预期
        if self.padded_sharded_param_size != self._sharded_local_tensor.size():
            raise NotImplementedError(
                "FSDP all-gather extensions require even sharding on dim-0.\n"
                f"{self._orig_size} is not divisible by FSDP world size {self.mesh_info.mesh.size()}."
            )
        
        # 初始化扩展数据对象
        self._extensions_data = ExtensionsData()
    
    # 初始化未分片的内部张量列表
    self._unsharded_inner_tensors: List[torch.Tensor] = []

def init_all_gather_outputs(
    self,
    all_gather_input_numels: List[int],
    all_gather_input_dtypes: List[torch.dtype],
    world_size: int,
    device: torch.device,
    force_recreate: bool = False,
):
    # 如果不强制重新创建并且已经初始化了all_gather_outputs，则直接返回
    if not force_recreate and len(self.all_gather_outputs) > 0:
        return
    
    # 根据输入的大小和数据类型创建all_gather_outputs列表
    self.all_gather_outputs = [
        torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
    ]

def _unflatten_all_gather_outputs(self) -> Tuple[torch.Tensor, ...]:
    # 对all_gather_outputs进行展开操作，返回一个元组，每个元素是一个张量
    return tuple(
        t.view(-1, *s[1:])
        for t, s in zip(
            self.all_gather_outputs, self._extensions_data.all_gather_input_sizes
        )
    )

def to_sharded(self) -> None:
    # 将参数设置为分片状态，并在模块上设置属性
    self._setattr_on_modules(self.sharded_param)
    
    # 释放未分片的参数
    self.free_unsharded_param()
    
    # 将sharded_state设置为SHARDED状态
    self.sharded_state = ShardedState.SHARDED
    # 将当前对象转换为未分片状态的后向传递过程
    def to_sharded_post_forward(self) -> None:
        # 如果当前对象是一个分布式张量，则抛出未实现错误
        if self.is_dtensor:
            raise NotImplementedError(
                "Resharding to smaller mesh with TP is not supported yet"
            )
        # 断言当前状态为未分片状态
        self._assert_in_states(ShardedState.UNSHARDED)
        # 确保后向传递网格信息不为None
        assert self.post_forward_mesh_info is not None  # mypy
        # 确保all_gather_outputs列表长度为1
        assert len(self.all_gather_outputs) == 1
        # 获取分片世界大小
        shard_world_size = self.post_forward_mesh_info.shard_mesh_size
        # 计算all_gather输出的元素数量
        if (numel := self.all_gather_outputs[0].numel()) % shard_world_size != 0:
            # 如果不是分片世界大小的倍数，则抛出断言异常
            _raise_assert_with_print(
                f"All-gather output size ({numel}) must be divisible by the shard "
                f"world size ({shard_world_size})"
            )
        # 获取当前对象在分片网格中的排名
        shard_rank = self.post_forward_mesh_info.shard_mesh_rank
        # 计算每个分片的元素数量
        sharded_numel = numel // shard_world_size
        # 从all_gather_outputs[0]中截取分片后的参数数据，并克隆一份以释放all-gather输出
        self._sharded_post_forward_param_data = (
            self.all_gather_outputs[0].narrow(
                0, sharded_numel * shard_rank, sharded_numel
            )
        ).clone()
        # 使用torch.as_strided创建分片后的后向传递张量
        sharded_post_forward_tensor = torch.as_strided(
            self._sharded_post_forward_param_data,
            size=self.sharded_post_forward_size,
            stride=self.contiguous_sharded_post_forward_stride,
            storage_offset=0,
        )
        # 将分片后的后向传递张量转换为nn.Parameter对象
        self._sharded_post_forward_param = nn.Parameter(
            self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor)
        )
        # 在相关模块上设置当前参数
        self._setattr_on_modules(self._sharded_post_forward_param)
        # 释放未分片参数
        self.free_unsharded_param()
        # 将对象的分片状态设置为分片后向传递状态
        self.sharded_state = ShardedState.SHARDED_POST_FORWARD

    # 将对象转换为未分片状态
    def to_unsharded(self) -> None:
        # 如果需要，设置参数的梯度属性
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        # 在相关模块上设置未分片参数
        self._setattr_on_modules(self._unsharded_param)
        # 如果对象当前处于分片后向传递状态
        if self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            # 释放分片后向传递参数及其数据
            self._sharded_post_forward_param = None
            self._sharded_post_forward_param_data = None  # free
        # 将对象的分片状态设置为未分片状态
        self.sharded_state = ShardedState.UNSHARDED

    # 在相关模块上设置参数
    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        # 使用不安全的方法设置当前模块的参数
        unsafe_setattr_param(
            self._module_info.module, self._module_info.param_name, param
        )
        # 对于每个共享模块及其参数，使用不安全的方法设置参数
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            unsafe_setattr_param(shared_module, shared_param_name, param)
    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        # 检查输入的张量形状是否与预期的分片大小相匹配
        if tensor.shape != self.sharded_size:
            _raise_assert_with_print(
                f"Expects size {self.sharded_size} but got {tensor.shape}"
            )
        # 将本地张量转换为分片后的 DTensor，无梯度操作
        return _from_local_no_grad(
            tensor,
            self._sharding_spec,
        )

    def to_sharded_post_forward_dtensor(self, tensor: torch.Tensor) -> DTensor:
        # 检查输入的张量形状是否与预期的后向传播分片大小相匹配
        if tensor.shape != self.sharded_post_forward_size:
            _raise_assert_with_print(
                f"Expects size {self.sharded_post_forward_size} but got {tensor.shape}"
            )
        # 确保 self.post_forward_mesh_info 是 HSDPMeshInfo 类型
        assert isinstance(self.post_forward_mesh_info, HSDPMeshInfo)
        # 创建后向传播分片的 DTensorSpec，包括网格信息和张量元数据
        post_forward_sharding_spec = DTensorSpec(
            self.post_forward_mesh_info.mesh,
            (Replicate(), Shard(0)),
            tensor_meta=self._sharding_spec.tensor_meta,
        )
        # 将输入张量转换为后向传播分片的 DTensor，无梯度操作
        return _from_local_no_grad(tensor, post_forward_sharding_spec)

    def to_accumulated_grad_if_needed(self) -> None:
        # 访问 `_unsharded_param` 以绕过分片状态检查，因为我们更倾向于在将梯度提升前重新分片以节省内存
        if (
            self.reduce_dtype is None
            or self._unsharded_param.grad is None
            or self._unsharded_param.grad.dtype == self.reduce_dtype
        ):
            return
        # 如果需要，将非分片梯度提升到指定数据类型
        unsharded_grad = self._unsharded_param.grad
        self._unsharded_param.grad = None
        self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    def accumulate_unsharded_grad_if_needed(self) -> None:
        # 如果非分片累积梯度不为 None，并且非分片参数的梯度不为 None，则累积梯度
        if (
            self.unsharded_accumulated_grad is not None
            and self.unsharded_param.grad is not None
        ):
            self.unsharded_accumulated_grad += self.unsharded_param.grad
            self.unsharded_param.grad = None

    def alloc_all_gather_outputs(self) -> None:
        # 为所有收集输出分配存储空间
        for tensor in self.all_gather_outputs:
            alloc_storage(tensor)

    def free_unsharded_param(self) -> None:
        # 释放所有收集输出和非分片内部张量的存储空间
        for tensor in itertools.chain(
            self.all_gather_outputs, self._unsharded_inner_tensors
        ):
            free_storage(tensor)
        # 如果编译自动求导启用，则重置所有收集输出和非分片内部张量
        if ca.compiled_autograd_enabled:
            self.all_gather_outputs = []
            self._unsharded_inner_tensors = []

    @property
    # 返回一个包含所有输入张量的列表，每个张量是一维的
    def all_gather_inputs(self) -> List[torch.Tensor]:  # 1D
        # 确保当前对象处于SHARDED或SHARDED_POST_FORWARD状态
        self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)
        
        # 如果当前状态是SHARDED
        if self.sharded_state == ShardedState.SHARDED:
            # 如果未启用编译自动梯度并且sharded_local_tensor具有fsdp_pre_all_gather属性
            if not ca.compiled_autograd_enabled and hasattr(
                self._sharded_local_tensor, "fsdp_pre_all_gather"
            ):
                # 获取sharded_local_tensor，并根据需要将其转移到self.device上
                sharded_local_tensor = self._sharded_local_tensor
                if self.offload_to_cpu:
                    sharded_local_tensor = sharded_local_tensor.to(
                        self.device, non_blocking=True
                    )
                
                # 调用fsdp_pre_all_gather方法进行数据预处理
                (
                    all_gather_inputs,
                    self._extensions_data.all_gather_metadata,
                ) = sharded_local_tensor.fsdp_pre_all_gather(self.mesh_info.mesh)
                
                # 记录每个输入张量的大小信息
                self._extensions_data.all_gather_input_sizes = [
                    t.size() for t in all_gather_inputs
                ]
                
                # 返回重新形状为一维的所有输入张量列表
                return [t.view(-1) for t in all_gather_inputs]
            
            # 如果未满足上述条件，则返回未分片的参数数据，根据需要转换数据类型
            sharded_param_data = self._sharded_param_data
            if self.offload_to_cpu:
                sharded_param_data = sharded_param_data.to(
                    self.device, non_blocking=True
                )
            return [_to_dtype_if_needed(sharded_param_data, self.param_dtype)]
        
        # 如果当前状态是SHARDED_POST_FORWARD，抛出NotImplementedError异常
        elif self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            if not ca.compiled_autograd_enabled and hasattr(
                self._sharded_local_tensor, "fsdp_pre_all_gather"
            ):
                raise NotImplementedError
            
            # 返回转换数据类型后的后向传播参数数据
            all_gather_input = _to_dtype_if_needed(
                cast(torch.Tensor, self._sharded_post_forward_param_data),
                self.param_dtype,
            )
            return [all_gather_input]
        
        # 如果当前状态既不是SHARDED也不是SHARDED_POST_FORWARD，则返回一个空张量列表
        return [torch.empty(0)]  # mypy

    # 返回未分片的参数作为一个nn.Parameter对象，这里假设是多维的
    @property
    def unsharded_param(self) -> nn.Parameter:  # ND
        # 确保当前对象处于UNSHARDED状态
        self._assert_in_states(ShardedState.UNSHARDED)
        return self._unsharded_param

    # 返回未分片参数的梯度数据作为torch.Tensor对象
    @property
    def unsharded_grad_data(self) -> torch.Tensor:
        # 获取未分片参数的梯度数据，确保梯度不为None
        grad = self.unsharded_param.grad
        assert grad is not None, "Expects unsharded_param.grad to not be None"
        return self._get_grad_inner_tensor(grad)

    # 返回未分片累积梯度数据作为torch.Tensor对象
    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor:
        # 获取未分片累积梯度数据，确保梯度不为None
        grad = self.unsharded_accumulated_grad
        assert grad is not None, "Expects unsharded_accumulated_grad to not be None"
        return self._get_grad_inner_tensor(grad)
    # 获取内部梯度张量，根据是否是分布式张量进行处理
    def _get_grad_inner_tensor(self, grad: torch.Tensor) -> torch.Tensor:
        if self.is_dtensor:
            # 如果梯度是 AsyncCollectiveTensor 类型，则等待其完成
            if isinstance(grad, AsyncCollectiveTensor):
                grad = grad.wait()
            # 断言梯度是 DTensor 类型，否则引发异常
            assert isinstance(grad, DTensor), f"{type(grad)}"
            # 如果梯度中有任何部分是分布式的，创建对应的复制策略并重新分布梯度
            if any(pl.is_partial() for pl in grad.placements):
                placements = [
                    Replicate() if pl.is_partial() else pl for pl in grad.placements
                ]
                grad = grad.redistribute(placements=placements)
            # 获取梯度的本地张量
            grad = grad._local_tensor
        return grad

    # 返回分片的本地张量
    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        return cast(DTensor, self.sharded_param)._local_tensor

    # 断言当前分布式状态在指定状态中
    def _assert_in_states(self, *states: ShardedState) -> None:
        if self.sharded_state not in states:
            _raise_assert_with_print(
                f"Expects to be in one of {states}, not {self.sharded_state}"
            )

    # 重设分片参数
    def reset_sharded_param(self):
        # 用于操作如 `nn.Module._apply` 或 `load_state_dict(assign=True)`，
        # 这些操作可能会改变分片参数张量，需要重新填充分片的本地张量并重新保存引用。
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        # 如果新的参数不是当前的分片参数，则更新分片参数并重新设置
        if new_param is not self.sharded_param:
            if torch.__future__.get_swap_module_params_on_conversion():
                raise AssertionError(
                    f"Expects swap_tensors to preserve object but got {new_param} "
                    f"instead of {self.sharded_param}"
                )
            self.sharded_param = new_param
        # 获取新参数的本地张量
        local_tensor = new_param._local_tensor
        # 如果本地张量是元数据，则直接返回
        if local_tensor.is_meta:
            return
        # 获取填充后的分片参数大小
        padded_sharded_size = self.padded_sharded_param_size
        # 如果本地张量的大小不等于填充后的分片参数大小，则进行填充操作
        if local_tensor.size() != padded_sharded_size:
            padded_local_tensor = local_tensor.new_zeros(padded_sharded_size)
            padded_local_tensor[: local_tensor.size(0)].copy_(local_tensor)
            local_tensor = padded_local_tensor
        # 如果需要固定内存并且本地张量未被固定，则将其移到 CPU 固定内存上
        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.cpu().pin_memory()
        # 将分片参数的数据视图设置为本地张量的展平版本
        self._sharded_param_data = local_tensor.view(-1)
        # 断言分片参数是 DTensor 类型，用于类型检查
        assert isinstance(self.sharded_param, DTensor)  # mypy
        # 更新分片参数的本地张量为前 sharded_size[0] 个元素
        self.sharded_param._local_tensor = local_tensor[: self.sharded_size[0]]
# 分配存储空间给张量，确保其存储的大小足够
def alloc_storage(tensor: torch.Tensor) -> None:
    # 计算张量所需的存储空间大小
    size = tensor.numel() * tensor.itemsize
    # 获取张量的未类型化存储，并检查其大小是否与所需大小相同
    if (storage := tensor.untyped_storage()).size() != size:
        # 调整存储空间大小以满足需求
        storage.resize_(size)


# 释放张量的存储空间
def free_storage(tensor: torch.Tensor) -> None:
    # 获取张量的未类型化存储，并检查其大小是否不为零
    if (storage := tensor.untyped_storage()).size() != 0:
        # 调整存储空间大小为零，释放存储空间
        storage.resize_(0)


# 注意：这些操作绕过了 `nn.Module.__setattr__` 的检查，避免了在模块没有重写该方法时带来的非常规 CPU 开销。
# 对于 FSDP（Fully Sharded Data Parallel）来说，在分片和非分片参数之间切换时，我们知道这些检查是不必要的。
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    # 如果模块的 `__setattr__` 方法未被重写，则直接设置参数到 `_parameters` 字典中
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:  # 慢速路径：如果模块的 `__setattr__` 方法被重写，则通过通用的 `setattr` 方法设置参数
        setattr(module, param_name, param)


# 根据需要设置张量的 `requires_grad` 属性，避免不必要的 Python <-> C++ 上下文切换开销
def set_requires_grad_if_needed(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> None:
    # 只有在需要时才调用 `requires_grad_` 方法，以避免不必要的开销
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        # 将目标张量的 `requires_grad` 属性设置为源张量的 `requires_grad` 属性
        dst_tensor.requires_grad_(src_tensor.requires_grad)
```