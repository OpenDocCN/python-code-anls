# `.\pytorch\torch\distributed\fsdp\_flat_param.py`

```py
# mypy: allow-untyped-defs
# 导入上下文管理、函数工具、日志记录、操作系统功能、警告管理等模块
import contextlib
import functools
import logging
import os
import warnings
# 从枚举模块中导入自动枚举和自定义枚举类型
from enum import auto, Enum
# 从迭代工具模块中导入累积函数和链式迭代器
from itertools import accumulate, chain
# 导入类型提示模块中多种类型的定义
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
    no_type_check,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# 导入PyTorch相关模块和类
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# 从PyTorch分布式FSDP模块中导入共享的内部工具函数和类
from torch.distributed.fsdp._common_utils import (
    _FSDPDeviceHandle,
    _named_parameters_with_duplicates,
    _no_dispatch_record_stream,
    _set_fsdp_flattened,
    HandleTrainingState,
)
# 从PyTorch分布式工具模块中导入内存管理和断言工具函数
from torch.distributed.utils import (
    _alloc_storage,
    _data_ptr_allocated,
    _free_storage,
    _p_assert,
)
# 从PyTorch参数模块中导入参数元信息类
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
# 从PyTorch内部测试分布式模块中导入虚假进程组类
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup

# 导入FSDP扩展模块中的相关函数
from ._fsdp_extensions import (
    _ext_post_unflatten_transform,
    _ext_pre_flatten_transform,
    FSDPExtensions,
)

# 定义可导出的模块成员列表
__all__ = [
    "FlatParameter",
    "FlatParamHandle",
    "FlatParamShardMetadata",
    "ParamInfo",
    "SharedParamInfo",
    "HandleShardingStrategy",
]

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)

"""
[注意：完全分片模块]
我们定义“完全分片模块”为拥有 FlatParamHandle 的原始 nn.Module，它负责给定前向或反向传播的单个 unshard/reshard 对。
完全分片模块应该传递给 FlatParamHandle 构造函数。

对于包装器代码路径：
- FullyShardedDataParallel 模块包装完全分片模块，并通过重写 nn.Module.forward 来运行 unshard/reshard。
- 完全分片模块正是传递给 FullyShardedDataParallel 构造函数中的 module 参数。

对于非包装器代码路径：
- 在完全分片模块上注册的钩子运行 unshard/reshard。
- 完全分片模块可以直接作为 fully_shard 的参数，或者根据提供的包装策略选择子模块。
"""

# 环境变量，用于切换是否在 `_use_sharded_views()` 和 `_use_unsharded_views()` 中使用不安全的 `setattr()`
# 默认情况下应该使用 'safe'，因为它尊重方法覆盖，但对于特殊情况，如高CPU开销或故意绕过检查的情况，可以使用 'unsafe'。
_FSDP_USE_UNSAFE_SETATTR = "FSDP_USE_UNSAFE_SETATTR"

# 环境变量，用于切换是否在FSDP初始化后检查参数/梯度写回，以防它们的存储发生更改
# 默认情况下应该检查，因为它可以防止潜在的正确性错误，但如果这种更改不寻常，可以跳过检查以节省CPU开销，特别是因为检查发生在前向和
# 每次迭代前向处理的跳过写回检查。
_FSDP_SKIP_WRITEBACK_CHECK = "FSDP_SKIP_WRITEBACK_CHECK"

# 当模型处于.eval()模式时，环境变量控制是否在fp32或者降低精度下运行。
_FSDP_USE_FULL_PREC_IN_EVAL = "FSDP_USE_FULL_PREC_IN_EVAL"

# 用于调试时设置张量填充的某个值。
_FLAT_PARAM_PADDING_VALUE = 42

# 用于消除全聚合和减少分散通信操作的环境变量，用于消融研究。
# 注意：禁用这些通信操作会导致训练不收敛，可能需要在模型中禁用正确性检查。
_FSDP_USE_FAKE_ALL_GATHER = "FSDP_USE_FAKE_ALL_GATHER"
_FSDP_USE_FAKE_REDUCE = "FSDP_USE_FAKE_REDUCE"


# TODO: 暂时定义此处以避免循环导入。查看是否可以移除。
class HandleShardingStrategy(Enum):
    FULL_SHARD = auto()  # 完全分片策略
    SHARD_GRAD_OP = auto()  # 梯度操作分片策略
    NO_SHARD = auto()  # 无分片策略
    HYBRID_SHARD = auto()  # 混合分片策略
    _HYBRID_SHARD_ZERO2 = auto()  # 特定混合分片策略


# 在前向处理后重新分片的处理策略。
RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.FULL_SHARD,
    HandleShardingStrategy.HYBRID_SHARD,
)

# 在前向处理后不重新分片的处理策略。
NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.SHARD_GRAD_OP,
    HandleShardingStrategy._HYBRID_SHARD_ZERO2,
)


class ParamInfo(NamedTuple):
    """原始参数的信息。"""

    param_name: str  # 未加前缀的参数名
    module: nn.Module  # 参数所属的模块
    module_name: str  # 参数所属模块的名称


class SharedParamInfo(NamedTuple):
    """
    共享参数的附加信息。

    对于每个共享参数，我们指定一个模块和其参数变量作为主要所有者，确定为参数遍历中遇到的第一个。
    这些带有“prim”前缀。主要模块和参数没有自己的:class:`SharedParamInfo`实例。
    """

    param_name: str  # 未加前缀的参数名
    module: nn.Module  # 参数所属的模块
    module_name: str  # 参数所属模块的名称
    prim_param_name: str  # 未加前缀的主要参数名
    prim_module: nn.Module  # 主要参数所属的模块
    prim_module_name: str  # 主要参数所属模块的名称


class _ShardParamInfo(NamedTuple):
    """原始参数的分片相关信息。"""

    in_shard: bool  # 是否在分片中
    # 用于索引分片后的平坦参数，例如 `flat_param[offset_in_shard : offset_in_shard + numel_in_shard]`
    offset_in_shard: Optional[int]  # 在分片中的偏移量
    numel_in_shard: Optional[int]  # 分片中的元素数目
    # 用于从未分片参数的平坦版本中获取本地分片中的部分参数，例如 `param.flatten()[intra_param_start_idx : intra_param_end_idx + 1]`
    intra_param_start_idx: Optional[int]  # 参数在本地分片中的起始索引
    intra_param_end_idx: Optional[int]  # 参数在本地分片中的结束索引（包含）


class FlatParamShardMetadata(NamedTuple):
    """
    包含特定于此排名分片的平坦参数的元数据。
    """
    # 属性定义：本类的参数名称，是由 FlatParameter 类处理后的带前缀的参数名称元组
    param_names: Tuple[str, ...]
    # 属性定义：本类的参数形状，是由 FlatParameter 类处理后的参数形状元组
    param_shapes: Tuple[torch.Size, ...]
    # 属性定义：本类的参数元素个数，是由 FlatParameter 类处理后的参数元素个数元组
    param_numels: Tuple[int, ...]
    # 属性定义：本类的参数偏移量，是每个参数在扁平化后的起始和结束位置（单位为元素）元组的元组
    param_offsets: Tuple[Tuple[int, int], ...]
class _FlatParameterMeta(_ParameterMeta):
    # 定义一个元类 _FlatParameterMeta，继承自 _ParameterMeta
    # 使得 isinstance(t, FlatParameter) 可以对具有 _is_flat_param 标志的自定义张量实例返回 True，以支持向后兼容性
    def __instancecheck__(self, instance):
        # 注意：不要测试超类的实现
        return isinstance(instance, torch.Tensor) and getattr(
            instance, "_is_flat_param", False
        )


class FlatParameter(nn.Parameter, metaclass=_FlatParameterMeta):
    """
    这是由 :class:`FullyShardedDataParallel` 使用的平坦参数。

    它由一个或多个原始参数组成，这些参数被展平并连接以构建平坦参数。

    在当前设计下，该参数在逻辑上代表未分片和分片的平坦参数，并且其数据会动态地更改存储位置。
        - 在 :class:`FullyShardedDataParallel` 构造函数中，参数被初始化为未分片，然后在原地分片。
        - 在运行时，参数是惰性地（重新）初始化的。分片参数数据保存在 ``self._local_shard`` 中，并且创建了一个新的 ``Tensor``
          ``self._full_param_padded``，它是全局聚集的目标，并且此后拥有未分片参数的存储。 （参见 :meth:`FlatParamHandle.init_flat_param_attributes`。）
        - 在整个运行时期间，参数数据根据需要更改存储位置，例如到分片的平坦参数、低精度分片的平坦参数或未分片的平坦参数。

    注意：由于 ``use_orig_params=True`` 支持内部的 ``FlatParameter`` 填充，我们有两个版本的每个参数的 numels，一个包括填充（``_numels_with_padding``），
    一个不包括（``_numels``）。前者的长度可能比其他数据结构长，而后者与实际原始参数的数量一样长，类似其他每个参数的数据结构。

    注意：这不是一个真正的类；相反，如果尝试创建这种类的实例，您将始终得到一个 Parameter。 这类似于我们为 Parameter 实现的技巧，以使其与子类一起使用；
    这主要是为了使 FlatParameter 支持与 FakeTensor 的组合。

    """

    _unpadded_unsharded_size: torch.Size
    _padded_unsharded_size: torch.Size
    _sharded_size: torch.Size
    _num_params: int
    _param_infos: Tuple[ParamInfo, ...]
    _shapes: Tuple[torch.Size, ...]
    _fqns: Tuple[str, ...]
    _param_extensions: Tuple[Optional[Any], ...]
    _numels_with_padding: Tuple[int, ...]
    _numels: Tuple[int, ...]
    _shard_param_infos: Tuple[_ShardParamInfo, ...]
    _shared_param_infos: Tuple[SharedParamInfo, ...]
    _modules: Set[nn.Module]
    _shard_numel_padded: int
    _local_shard: Tensor
    _full_param_padded: Tensor
    _full_prec_full_param_padded: Tensor
    # 仅在 Eager 模式下存在
    _post_backward_hook_state: Tuple[Any, Any]
    # 仅在编译模式下存在
    # 定义私有变量，用于存储反向钩子处理的句柄
    _post_backward_hook_handle: Any
    # 存储张量数据的变量，用于多处理单元 (MPU)
    _mp_shard: Tensor
    # CPU 上的梯度张量
    _cpu_grad: Tensor
    # 存储梯度的张量，用于后续步骤
    _saved_grad_shard: Tensor
    # 可选的参数列表，存储该对象的参数
    _params: Optional[List[nn.Parameter]]
    # 可选的共享参数列表
    _shared_params: Optional[List[nn.Parameter]]
    # 可选的张量列表，用于存储数据
    _tensors: Optional[List[Optional[Tensor]]]
    # 可选的布尔值列表，用于指示梯度是否为 None
    _is_grad_none_mask: Optional[List[bool]]

    # 布尔值列表，指示是否为填充遮罩
    _is_padding_mask: List[bool]

    # 构造函数，创建 FlatParameter 的实例
    def __new__(cls, data=None, requires_grad=True):
        # 断言确保当前类是 FlatParameter，不支持子类
        assert cls is FlatParameter, "subclasses FlatParameter not supported"
        # 调用父类 nn.Parameter 的构造函数，创建参数对象 r
        r = nn.Parameter.__new__(nn.Parameter, data, requires_grad)  # type: ignore[call-arg]
        # 设置标志，指示这是一个扁平参数对象
        r._is_flat_param = True  # type: ignore[attr-defined]
        # 返回创建的参数对象
        return r

    # 类方法，用于初始化元数据
    # 注意：这不是一个常规方法，因为 FlatParameters 实际上不是该类的实例（参见上面的 __new__ 方法）
    # 因此，必须通过类方法间接调用此方法。
    @classmethod
    def _init_metadata(
        cls,
        self,
        param_infos: List[ParamInfo],
        numels: List[int],
        shapes: List[torch.Size],
        fqns: List[str],
        shared_param_infos: List[SharedParamInfo],
        param_extensions: List[Optional[Any]],
        params: Optional[List[nn.Parameter]],
        shared_params: Optional[List[nn.Parameter]],
        is_padding_mask: List[bool],
    ) -> None:
        """
        初始化方法，用于设置关于原始参数的元数据信息，这些参数构成了平坦参数。

        我们将此方法单独暴露出来，而不放在构造函数中，以保持构造函数仅负责平坦参数的张量数据。
        此方法每个模型只应调用一次，而构造函数可能会被多次调用，例如从检查点重新加载时，只需将张量数据传递给构造函数。
        由于 `load_state_dict` 方法通过 `copy_` 实现，因此正确假定元数据未更改。

        Args:
            param_infos: 参数信息列表，详见类文档字符串中的属性描述。
            shapes: 参数的形状列表。
            fqns: 参数的全限定名称列表。
            param_extensions: 参数扩展信息列表。
            is_padding_mask: 布尔型列表，指示是否为填充遮罩。

        Raises:
            AssertionError: 如果传入的各列表长度不一致。

        """
        assert len(param_infos) == len(shapes)
        assert len(param_infos) == len(fqns)
        assert len(param_infos) == len(param_extensions)

        # 设置参数数量
        self._num_params = len(param_infos)
        # 设置参数信息列表
        self._param_infos = param_infos
        # 设置参数形状列表
        self._shapes = shapes
        # 设置参数全限定名称列表
        self._fqns = fqns
        # 设置参数扩展信息列表
        self._param_extensions = param_extensions
        # 设置是否为填充遮罩
        self._is_padding_mask = is_padding_mask

        # 计算不包含填充的参数元素数列表
        numels_without_padding: List[int] = []
        for numel, is_padding in zip(numels, is_padding_mask):
            if not is_padding:
                numels_without_padding.append(numel)
        # 设置不包含填充的参数元素数元组
        self._numels = tuple(numels_without_padding)
        # 设置包含填充的参数元素数元组
        self._numels_with_padding = tuple(numels)
        assert len(self._numels) == self._num_params

        # 设置共享参数信息元组
        self._shared_param_infos = tuple(shared_param_infos)
        # 计算模块集合，包括参数信息和共享参数信息中的模块
        self._modules = {pi.module for pi in self._param_infos}.union(
            {spi.module for spi in self._shared_param_infos}
        )
        assert (params is None) == (shared_params is None)
        if params is not None:
            assert shared_params is not None and len(shared_params) == len(
                shared_param_infos
            )
            # 设置参数列表，仅包含非填充参数
            self._params = []
            for param, is_padding in zip(params, is_padding_mask):
                if not is_padding:
                    self._params.append(param)
            # 设置共享参数列表
            self._shared_params = shared_params
            # 对原始参数进行标记，避免在递归构造过程中再次将其扁平化为另一个 `FlatParameter`
            for param in chain(self._params, self._shared_params):
                _set_fsdp_flattened(param)
            # 初始化梯度是否为 None 的遮罩
            self._is_grad_none_mask = [False for _ in range(self._num_params)]
            # 初始化张量列表
            self._tensors = [None for _ in range(self._num_params)]
        else:
            # 如果没有传入参数和共享参数，则置为 None
            self._params = None
            self._shared_params = None
            self._is_grad_none_mask = None
            self._tensors = None
        # 计算未填充未分片大小
        self._unpadded_unsharded_size = self.size()
        # 标记当前对象为扁平化的 `FlatParameter`
        _set_fsdp_flattened(self)
        # 跟踪 `FlatParameter` 的反向传播钩子是否已调用，以修改后向传播回调的行为
        self._post_backward_called = False
    ##################
    # INITIALIZATION #
    ##################

    # 初始化 FlatParamHandle 类的实例
    def __init__(
        self,
        params: Sequence[Union[nn.Parameter, Tensor]],
        fully_sharded_module: nn.Module,
        device: torch.device,
        sharding_strategy: HandleShardingStrategy,
        offload_params: bool,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
        keep_low_precision_grads: bool,
        process_group: dist.ProcessGroup,
        use_orig_params: bool,
        *,
        fsdp_extension: Optional[FSDPExtensions] = None,
    ):
        # 检查是否启用不安全的 setattr
        use_unsafe_setattr = os.environ.get(_FSDP_USE_UNSAFE_SETATTR, "") == "1"

        # 设置用于设置 tensor 和 param 的函数
        self._setattr_tensor: Callable[[nn.Module, str, Tensor], None]
        self._setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
        
        # 根据使用情况选择设置函数
        if use_unsafe_setattr:
            self._setattr_tensor = _unsafe_setattr_tensor
            self._setattr_param = _unsafe_setattr_param
        else:
            self._setattr_tensor = _safe_setattr_tensor_or_param
            self._setattr_param = _safe_setattr_tensor_or_param

        # 初始化获取非扁平化视图的函数
        def _init_get_unflat_views_fn(self, align_addresses: bool):
            # 根据 align_addresses 参数选择对齐或非对齐的获取视图函数
            self._get_unflat_views = (
                self._get_unflat_views_aligned
                if align_addresses
                else self._get_unflat_views_unaligned
            )
    def _init_flat_param_and_metadata(
        self,
        params: List[Union[Tensor, nn.Parameter]],
        module: nn.Module,
        aligned_numel: int,
        use_orig_params: bool,
    ):
        """Initialize flat parameters and metadata necessary for flattening."""
        dtype: Optional[torch.dtype] = None
        # Return as the logical OR over each tensor's value
        flat_param_requires_grad: Optional[bool] = None
        device: Optional[torch.device] = None
        # For `use_orig_params=True`, permit non-uniform `requires_grad`
        for tensor in tensors:
            if isinstance(tensor, FlatParameter):
                raise ValueError("Cannot flatten a `FlatParameter`")
            if dtype is None and not tensor.is_floating_point():
                raise ValueError("Cannot flatten integer dtype tensors")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Must flatten tensors with uniform dtype but got {dtype} "
                    f"and {tensor.dtype}"
                )
            if (
                not self._use_orig_params
                and flat_param_requires_grad is not None
                and tensor.requires_grad != flat_param_requires_grad
            ):
                raise ValueError(
                    "Must flatten tensors with uniform `requires_grad` when "
                    "`use_orig_params=False`"
                )
            if device is not None and tensor.device != device:
                raise ValueError(
                    "Must flatten tensors on the same device but got both "
                    f"{device} and {tensor.device}"
                )
            dtype = tensor.dtype
            flat_param_requires_grad = flat_param_requires_grad or tensor.requires_grad
            device = tensor.device
        assert flat_param_requires_grad is not None, "Requires non-empty `tensors` list"
        return dtype, flat_param_requires_grad, device

    def flatten_tensors(
        self,
        tensors: List[Tensor],
        aligned_numel: int,
        use_orig_params: bool,
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        """Flatten the tensors into a single tensor and return metadata."""
    ) -> Tensor:
        """
        Flatten ``tensors`` into a single flat tensor.

        The flattening optionally includes
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
        # 检查输入的`tensors`列表是否为空，如果为空则抛出异常
        if len(tensors) == 0:
            raise ValueError("Expects non-empty `tensors`")
        # 检查`aligned_numel`是否小于0，如果是则抛出异常
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        # 验证`tensors`中的张量，获取数据类型(dtype)、设备(device)
        dtype, _, device = self._validate_tensors_to_flatten(tensors)
        # 用于存放扁平化后的张量列表
        flat_tensors: List[Tensor] = []
        # 如果需要填充对齐
        if aligned_numel > 0:
            total_numel = 0
            # 遍历输入的每个张量
            for tensor in tensors:
                # 计算需要填充的元素数
                numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                # 如果需要填充且填充元素数大于0且小于aligned_numel，则创建填充张量
                if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                    padding_tensor = _construct_padding_tensor(
                        numel_to_pad, dtype, False, device
                    )
                    flat_tensors.append(padding_tensor)
                    total_numel += numel_to_pad
                # 将张量扁平化后加入列表，并更新总元素数
                flat_tensors.append(torch.flatten(_detach_if_needed(tensor)))
                total_numel += tensor.numel()
            # 计算最后一部分的填充元素数，确保对齐到world_size
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                flat_tensors.append(padding_tensor)
                total_numel += numel_to_pad
        else:
            # 如果不需要填充对齐，直接扁平化所有张量
            flat_tensors = [
                torch.flatten(_detach_if_needed(tensor)) for tensor in tensors
            ]
        # 使用torch.cat函数将所有扁平化后的张量拼接成一个张量并返回
        return torch.cat(flat_tensors, dim=0)

    def flatten_tensors_into_flat_param(
        self,
        tensors: List[Tensor],
        aligned_numel: int,
        requires_grad: bool,
    ) -> FlatParameter:
        # 调用flatten_tensors方法将输入的张量列表扁平化
        flat_param_data = self.flatten_tensors(tensors, aligned_numel)
        # 创建FlatParameter对象并返回
        return FlatParameter(flat_param_data, requires_grad=requires_grad)

    def _init_param_reduce_dtypes(
        self,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
    ) -> None:
        """
        Initialize param and reduce dtypes.

        Precondition: ``self.flat_param`` is set. This ensures that this
        handle's parameters have a single dtype.

        Postcondition: This sets ``self._fwd_bwd_param_dtype`` and
        ``self._reduce_dtype``. If ``mp_param_dtype`` or ``mp_reduce_dtype``
        is ``None``, then we assume the original parameter dtype. One special
        case is if ``mp_param_dtype`` is not ``None`` and ``mp_reduce_dtype``
        is ``None``, in which case we assume the gradient reduction dtype
        matches the forward/backward parameter dtype.
        """
        # Save whether these dtypes were specified so that we permit the
        # parameter dtype to change up until the lazy initialization
        # 保存是否明确指定了这些数据类型，允许在延迟初始化之前更改参数数据类型
        self._low_prec_param_dtype_specified = mp_param_dtype is not None
        self._low_prec_reduce_dtype_specified = mp_reduce_dtype is not None
        if (
            self._low_prec_param_dtype_specified
            and not self._low_prec_reduce_dtype_specified
        ):
            # Special case: infer gradient reduction mixed precision
            # 特殊情况：推断梯度减少的混合精度
            self._fwd_bwd_param_dtype = mp_param_dtype
            self._reduce_dtype = self._fwd_bwd_param_dtype
        else:
            self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
            self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
        assert self._fwd_bwd_param_dtype is not None
        assert self._reduce_dtype is not None

    ###################################
    # SHARD INITIALIZATION & METADATA #
    ###################################
    @torch.no_grad()
    def shard(self):
        """
        Shard the handle's ``FlatParameter``.

        This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
        # 将 self.flat_param 赋值给局部变量 flat_param
        flat_param = self.flat_param
        # 如果未使用分片策略
        if not self.uses_sharded_strategy:
            # 初始化分片元数据，设置 unsharded 区间
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
        else:
            # 断言 flat_param 的存储偏移为 0，确保 FlatParameter 是其存储的唯一占用者
            _p_assert(
                flat_param.storage_offset() == 0,
                "The `FlatParameter` is not the sole occupant of its storage",
            )
            # 获取分片后的 flat_param 和填充后的元素数
            sharded_flat_param, numel_padded = FlatParamHandle._get_shard(
                flat_param, self.rank, self.world_size
            )
            # 如果不是在 TorchDynamo 编译过程中
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # 检查 flat_param 的存储是否已分配内存
                allocated = flat_param._typed_storage()._size() > 0
                if allocated:
                    # 如果已分配，则将其大小调整为 0
                    flat_param._typed_storage()._resize_(0)
            # 将 flat_param 设置为分片后的 sharded_flat_param
            flat_param.set_(sharded_flat_param)  # type: ignore[call-overload]
            # 计算当前分片的起始和结束索引
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive
            # 初始化分片元数据，设置 numel_padded、start_idx 和 end_idx
            self._init_shard_metadata(numel_padded, start_idx, end_idx)
        # 如果使用原始参数，则使用分片后的视图
        if self._use_orig_params:
            self._use_sharded_views()

    def _init_shard_metadata(
        self,
        numel_padded: int,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> None:
        """
        Initialize shard-related metadata for this rank's shard of the flat parameter.

        This includes ``_sharded_size``, ``_shard_param_infos``, and ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
        # 获取 self 对象中的 flat_param 属性
        flat_param = self.flat_param
        # 设置 flat_param 对象的 _sharded_size 属性为 flat_param 的尺寸
        flat_param._sharded_size = flat_param.size()  # type: ignore[attr-defined]
        # 获取 flat_param 对象的元素数量（包括 numel_padded）
        sharded_flat_param_numel = flat_param.numel()  # includes `numel_padded`
        # 断言：确保 unsharded_start_idx 在合理范围内
        _p_assert(
            unsharded_start_idx >= 0 and unsharded_start_idx <= unsharded_end_idx,
            f"unsharded_start_idx: {unsharded_start_idx} unsharded_end_idx: {unsharded_end_idx}",
        )
        # 断言：确保 numel_padded 不超过 sharded_flat_param_numel
        _p_assert(
            numel_padded <= sharded_flat_param_numel,
            f"numel_padded: {numel_padded} "
            f"sharded_flat_param_numel: {sharded_flat_param_numel}",
        )
        # 获取分片参数的元数据信息
        shard_param_infos = self._get_shard_metadata(
            unsharded_start_idx, unsharded_end_idx
        )
        # 断言：确保 shard_param_infos 的长度与 flat_param._num_params 相符
        assert (
            len(shard_param_infos) == flat_param._num_params
        ), f"Expects length {flat_param._num_params} but got {len(shard_param_infos)}"
        # 设置 flat_param 对象的 _shard_param_infos 属性为 shard_param_infos
        flat_param._shard_param_infos = shard_param_infos  # type: ignore[attr-defined]
        # 设置 flat_param 对象的 _shard_numel_padded 属性为 numel_padded
        flat_param._shard_numel_padded = numel_padded  # type: ignore[attr-defined]

    def _get_shard_metadata(
        self,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> Tuple[_ShardParamInfo, ...]:
        """
        Compute the shard metadata based on ``unsharded_start_idx`` and ``unsharded_end_idx`` (inclusive).

        ``unsharded_start_idx`` and ``unsharded_end_idx`` give the interval of the
        unsharded flat parameter specifying the shard.
        """
        # 获取扁平参数的偏移量列表
        flat_param_offsets = self._get_flat_param_offsets()
        # 断言：确保扁平参数的偏移量列表长度与带填充元素的扁平参数长度相等
        assert len(flat_param_offsets) == len(
            self.flat_param._numels_with_padding
        ), f"Expected {len(self.flat_param._numels_with_padding)} but got {len(flat_param_offsets)}"
        # 初始化一个空列表用于存储分片参数信息
        shard_param_infos: List[_ShardParamInfo] = []
        # 计算分片扁平参数的元素数量
        sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1
        # 遍历扁平参数的偏移量和填充掩码
        for i, (
            (unsharded_param_start_idx, unsharded_param_end_idx),
            is_padding,
        ) in enumerate(zip(flat_param_offsets, self.flat_param._is_padding_mask)):
            # 如果是填充参数，则跳过
            if is_padding:
                continue
            # 判断当前参数是否在分片的扁平参数内
            in_sharded_flat_param = (
                unsharded_start_idx <= unsharded_param_end_idx
                and unsharded_end_idx >= unsharded_param_start_idx
            )
            # 根据是否在分片内构造参数的分片信息对象
            if not in_sharded_flat_param:
                shard_param_info = _ShardParamInfo(False, None, None, None, None)
            else:
                if unsharded_start_idx <= unsharded_param_start_idx:
                    # 这个分支只会执行一次，因为rank的未分片开始索引只会与一个参数交集
                    intra_param_start_idx = 0
                    offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
                else:
                    intra_param_start_idx = (
                        unsharded_start_idx - unsharded_param_start_idx
                    )
                    offset_in_shard = 0
                # 断言：确保分片偏移量在有效范围内
                assert (
                    offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel
                ), (
                    f"Invalid `offset_in_shard` of {offset_in_shard} for "
                    f"sharded flat parameter with {sharded_flat_param_numel} numel"
                )
                # 计算参数在分片内的结束索引和元素数量
                intra_param_end_idx = (
                    min(unsharded_param_end_idx, unsharded_end_idx)
                    - unsharded_param_start_idx
                )
                numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
                # 创建参数分片信息对象
                shard_param_info = _ShardParamInfo(
                    True,
                    offset_in_shard,
                    numel_in_shard,
                    intra_param_start_idx,
                    intra_param_end_idx,
                )
            # 将参数分片信息对象添加到列表中
            shard_param_infos.append(shard_param_info)
        # 返回元组形式的参数分片信息列表
        return tuple(shard_param_infos)
    def _get_unpadded_shard(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[Tensor, int]:
        """
        Return the unpadded shard of ``tensor`` for the given ``rank`` and ``world_size``.

        The returned value is a tuple of the shard of ``tensor`` without any
        padding and the numel to pad for that shard.

        If ``tensor`` is already flattened or may be viewed in the flattened
        shape (which is true in the expected usage), then this method does not
        allocate any new tensor memory.
        """
        # 将 tensor 扁平化后分割成多个块，每个块对应一个 rank
        chunks = torch.flatten(tensor).chunk(world_size)
        if len(chunks) < (rank + 1):
            # 如果 chunks 数量小于 (rank + 1)，说明该 rank 获得一个全零填充的空块
            chunk = chunks[0].new_empty(0)
        else:
            # 否则，该 rank 获得对应的 chunk
            chunk = chunks[rank]
        # 计算需要填充的元素数目
        numel_to_pad = chunks[0].numel() - chunk.numel()
        # 断言确保填充的元素数不会超过第一个 chunk 的大小
        assert (
            numel_to_pad >= 0
        ), "Chunk's size should be at most the first chunk's size"
        return chunk, numel_to_pad

    @staticmethod
    def _get_shard(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[Tensor, int]:
        """
        Return the shard of ``tensor`` with padding for the given ``rank`` and ``world_size`` and the numel padded for that shard.

        This method allocates new memory (via :meth:`clone`) since the
        unsharded ``tensor`` may be deallocated after this method returns.
        """
        # 获取未填充的 shard 和需要填充的元素数
        chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(
            tensor, rank, world_size
        )
        # 克隆未填充的 shard
        shard = chunk.clone()
        if numel_to_pad > 0:
            # 如果需要填充的元素数大于 0，则使用 F.pad 进行填充
            shard = F.pad(shard, [0, numel_to_pad])
        return shard, numel_to_pad

    @staticmethod
    def _get_sharded_size(tensor: Tensor, rank: int, world_size: int) -> torch.Size:
        """
        Return the shape of ``tensor`` after sharding including padding.

        This requires ``tensor`` to have 1D shape and ensures that the returned
        shape is 1D.
        """
        # 断言确保 tensor 是 1 维的
        assert len(tensor.shape) == 1, f"{tensor.shape}"
        # 获取未填充的 shard 和需要填充的元素数
        unpadded_sharded_tensor, numel_to_pad = FlatParamHandle._get_unpadded_shard(
            tensor, rank, world_size
        )
        # 获取未填充的 shard 的尺寸
        unpadded_sharded_size = unpadded_sharded_tensor.size()
        # 断言确保未填充的 shard 的尺寸也是 1 维的
        assert len(unpadded_sharded_size) == 1, f"{unpadded_sharded_size}"
        # 返回包含填充的 tensor 的尺寸
        return torch.Size([unpadded_sharded_size[0] + numel_to_pad])
    # 返回一个包含每个原始参数在未分片的扁平参数中的起始和结束偏移量的列表
    def _get_flat_param_offsets(self) -> List[Tuple[int, int]]:
        """
        Return [start, end] offsets of each original parameter's flattened data in the unsharded flat parameter (without padding).

        NOTE: The returned list includes elements for alignment padding.
        """
        # 计算未分片的扁平参数中每个参数数据的累积和
        cumulative_sum = list(accumulate(self.flat_param._numels_with_padding))
        # 计算每个参数数据的起始偏移量列表
        starts = [0] + cumulative_sum[:-1]
        # 计算每个参数数据的结束偏移量列表（包含最后一个元素）
        ends = [end - 1 for end in cumulative_sum]  # inclusive
        # 将起始和结束偏移量组成一个元组列表
        param_offsets = list(zip(starts, ends))
        return param_offsets

    @no_type_check
    def shard_metadata(
        self,
    ) -> FlatParamShardMetadata:
        """
        Return the shard-related metadata specific to this rank's shard of the flat parameter.

        NOTE: The returned tuple does not include elements for alignment
        padding but does account for the padding.
        """
        fqns_list = []
        shapes_list = []
        numels_list = []
        shard_param_offsets = []
        # 遍历每个扁平参数的元数据
        for fqn, shape, numel, shard_param_info in zip(
            self.flat_param._fqns,
            self.flat_param._shapes,
            self.flat_param._numels,
            self.flat_param._shard_param_infos,
        ):
            # 如果不在当前分片内，则跳过
            if not shard_param_info.in_shard:
                continue
            # 收集参数的全限定名、形状、元素数和分片参数的偏移信息
            fqns_list.append(fqn)
            shapes_list.append(shape)
            numels_list.append(numel)
            shard_param_offsets.append(
                (
                    shard_param_info.intra_param_start_idx,
                    shard_param_info.intra_param_end_idx,
                )
            )
        # 返回扁平参数分片的元数据对象
        return FlatParamShardMetadata(
            tuple(fqns_list),
            tuple(shapes_list),
            tuple(numels_list),
            tuple(shard_param_offsets),
        )

    @no_type_check
    @torch.no_grad()
    ###################
    # UNSHARD/RESHARD #
    ###################
    def pre_unshard(self) -> bool:
        """
        Return ``False`` if this is a no-op and ``True`` otherwise.

        Postcondition: ``self.flat_param`` 's data is on the device for
        communication and is what should be all-gathered. This means that it
        matches the dtype of the expected unsharded parameter.
        """
        if (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            and self._skipped_use_sharded_views
        ):
            # 如果 `_training_state` 为 `HandleTrainingState.SUMMON_FULL_PARAMS` 并且 `_skipped_use_sharded_views` 为真，
            # 使用分片视图以重用特定处理的逻辑（例如强制完整精度）来处理未分片的平坦参数
            self._use_sharded_views()
        ret = False
        if self._use_orig_params and not self._skip_writeback_check:
            # 如果 `_use_orig_params` 为真且 `_skip_writeback_check` 为假，
            # 执行写回原始参数的操作，并将结果赋给 `ret`
            ret = self._writeback_orig_params()
        if (
            self.uses_sharded_strategy
            and not self._offload_params
            and not self.needs_unshard()
        ):
            # 如果使用了分片策略，并且没有卸载参数并且不需要取消分片操作，
            # 则这是一个空操作
            pass  # no-op
        elif self._uses_param_mixed_precision and not self._force_full_precision:
            # 如果使用参数混合精度并且没有强制使用完整精度，
            # 使用低精度分片参数
            self._use_low_precision_shard()
            ret = True
        elif self._offload_params and self.flat_param.device != self.device:
            # 如果卸载了参数并且 `flat_param` 的设备与当前设备不同，
            # 将 `flat_param` 移动到指定设备（非阻塞）
            # 注意：这会创建一个与任何属性都不同的新张量
            self.flat_param_to(self.device, non_blocking=True)
            ret = True
        self._check_on_compute_device(self.flat_param)
        # 检查 `flat_param` 是否位于计算设备上，并返回 `ret`
        return ret

    def _use_low_precision_shard(self):
        """Allocate on the compute device and switch to using the low precision sharded flat parameter."""
        # 检查并确保低精度分片的正确性
        self._check_low_precision_shard()
        # 获取当前的平坦参数
        flat_param = self.flat_param
        # 分配存储空间给低精度分片，并将本地分片的大小传递给 `_alloc_storage`
        _alloc_storage(
            flat_param._mp_shard, flat_param._local_shard.size()  # type: ignore[attr-defined]
        )
        # `copy_()` 隐式地将数据复制到低精度
        flat_param._mp_shard.copy_(  # type: ignore[attr-defined]
            flat_param._local_shard.to(  # type: ignore[attr-defined]
                self.device, non_blocking=True
            )
        )
        # 不变条件：`_mp_shard` 始终位于计算设备上
        flat_param.data = flat_param._mp_shard  # type: ignore[attr-defined]
    def unshard(self):
        """
        Run the unshard logic.

        This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        # 检查是否需要执行unshard操作
        if not self.needs_unshard():
            # 即使不需要unshard，我们也应该切换到使用unsharded flat parameter
            unsharded_flat_param = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(unsharded_flat_param)
            return
        
        # 分配padded unsharded flat parameter
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        # all-gather操作，获取padded unsharded flat parameter
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
        # 切换到使用padded unsharded flat parameter
        self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def needs_unshard(self) -> bool:
        """Return if the handle's flat parameter needs to be unsharded."""
        # 如果不使用sharded策略，则不需要unshard
        if not self.uses_sharded_strategy:
            return False
        # 获取padded unsharded flat parameter
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        # 检查是否已经是unsharded状态
        already_unsharded = _same_storage_size(
            unsharded_flat_param, unsharded_flat_param.numel()
        )
        return not already_unsharded

    def _alloc_padded_unsharded_flat_param(self):
        """
        Allocate the *padded* unsharded flat parameter.

        The unpadded unsharded
        flat parameter is always a view into the padded one. This padded
        parameter is saved to a different attribute on the ``FlatParameter``
        depending on if we force full precision.
        """
        # 检查是否使用sharded策略
        self._check_sharded_strategy()
        # 获取flat parameter
        flat_param = self.flat_param
        # 获取padded unsharded flat parameter
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        # 检查存储空间是否已释放
        self._check_storage_freed(unsharded_flat_param)
        # 分配存储空间给padded unsharded flat parameter
        _alloc_storage(unsharded_flat_param, flat_param._padded_unsharded_size)  # type: ignore[attr-defined]
        return unsharded_flat_param
    def _get_padded_unsharded_flat_param(self) -> torch.Tensor:
        """
        Return a reference to the padded unsharded flat parameter depending on the calling context.

        This should only be called if using a sharded strategy.
        """
        # 检查当前是否使用了分片策略，如果没有则抛出异常
        self._check_sharded_strategy()
        
        # 获取当前对象的 flat_param 属性
        flat_param = self.flat_param
        
        # 如果强制使用全精度且使用了参数混合精度
        if self._force_full_precision and self._uses_param_mixed_precision:
            # 当启用参数混合精度时，使用另一个张量作为 all-gather 目标，
            # 以保证 `_full_param_padded` 保持低精度的不变性
            unsharded_flat_param = flat_param._full_prec_full_param_padded  # type: ignore[attr-defined]
            
            # 断言当前全精度的参数类型与 `_fwd_bwd_param_dtype` 不一致时抛出异常
            _p_assert(
                unsharded_flat_param.dtype != self._fwd_bwd_param_dtype,
                f"Expects full precision but got {self._fwd_bwd_param_dtype}",
            )
            
            # 对于前向传播后不重新分片的策略，`_full_param_padded` 可能仍然
            # 是前向传播前分配的。由于我们在这里强制使用全精度，因此全精度的
            # 未分片副本可能会被修改，使现有的低精度未分片副本无效化，因此
            # 我们应该在这里释放它，以确保下一个前向/反向计算的新全收集持
            # 续修改。
            if flat_param._full_param_padded.untyped_storage().size() > 0:
                _free_storage(flat_param._full_param_padded)
        else:
            # 否则直接使用 flat_param 的 `_full_param_padded` 属性作为未分片的平坦参数
            unsharded_flat_param = flat_param._full_param_padded  # type: ignore[attr-defined]
        
        # 返回未分片的平坦参数
        return unsharded_flat_param
    ) -> Tensor:
        """
        All-gather the handle's flat parameter to the destination ``padded_unsharded_flat_param``.

        Then switch to use the all-gathered tensor.
        """
        _p_assert(
            hasattr(self, "process_group") and hasattr(self, "world_size"),
            "Expects a process group and world size to have been set via `shard()`",
        )
        # 获取分片后的参数数据
        sharded_flat_param = self.flat_param.data
        # 计算期望的参数数量
        expected_numel = sharded_flat_param.numel() * self.world_size
        _p_assert(
            padded_unsharded_flat_param.numel() == expected_numel,
            f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
        )

        # 获取当前使用的进程组
        pg = (
            self._fake_process_group
            if self._use_fake_all_gather
            else self.process_group
        )

        # HACK this should be handled by C10D
        # 如果参数在 CPU 上
        if sharded_flat_param.is_cpu:  # type: ignore[attr-defined]
            # 将 padded_unsharded_flat_param 拆分成列表，列表长度为进程组的数量
            tensor_list = list(
                torch.chunk(padded_unsharded_flat_param, dist.get_world_size(pg))
            )
            # 使用分布式库进行全局聚集操作
            dist.all_gather(tensor_list, sharded_flat_param, group=pg)
        else:
            # 使用低级 API 进行全局聚集操作
            dist.all_gather_into_tensor(
                padded_unsharded_flat_param,
                sharded_flat_param,
                pg,
            )

        # 如果启用参数的卸载
        if self._offload_params:
            # 对于卸载情况下，`flat_param.data`（即分片参数）是在预未分片流上创建的。
            # 我们需要将其交给未分片流以进行全局聚集。
            _no_dispatch_record_stream(
                sharded_flat_param,
                self._device_handle.current_stream(),  # unshard_stream
            )
        # 返回未分片的填充扁平参数
        return padded_unsharded_flat_param
    ) -> None:
        """
        Switch to use the *unpadded* unsharded flat parameter.

        This is a view into the *padded* unsharded flat parameter.
        """
        # 获取未填充的、未分片的平坦参数的大小
        unsharded_size = self.flat_param._unpadded_unsharded_size
        # 从填充的未分片平坦参数中提取与未分片大小相符的部分
        flat_param_part = padded_unsharded_flat_param[: unsharded_size.numel()]
        # 由于使用了 `.data`，切片操作 `[:]` 对 autograd 不可见
        # 将切片后的部分赋值给 `self.flat_param.data`
        self.flat_param.data = flat_param_part
        # 检查当前训练状态是否为前向传播
        in_forward = self._training_state == HandleTrainingState.FORWARD
        # 检查当前训练状态是否为前向传播前的状态
        in_pre_backward = self._training_state == HandleTrainingState.BACKWARD_PRE
        # 如果使用原始参数
        if self._use_orig_params:
            # 如果跳过了使用分片视图而且处于前向传播前的状态
            if self._skipped_use_sharded_views and in_pre_backward:
                # 这次调用对应于补充的前向传播前的 `_use_unsharded_views()`，
                # 用于跳过前向传播的 `_use_sharded_views()`，因此我们也应跳过这次调用
                return
            # 在前向传播中使用 `Tensor` 视图以便被 autograd 跟踪
            # 在前向传播前的状态中同样使用它们，以支持可重入激活检查点，
            # 后者需要这些视图在反向传播的重计算前被 autograd 跟踪
            self._use_unsharded_views(
                as_params=(not in_forward and not in_pre_backward)
            )
        # 如果不使用原始参数且当前是前向传播状态
        elif in_forward:
            self._use_unsharded_views(as_params=False)

    def post_unshard(self):
        """
        Run the post-unshard logic.

        This includes freeing the low precision shard if needed.
        """
        # 如果使用参数混合精度且采用了分片策略
        if self._uses_param_mixed_precision and self.uses_sharded_strategy:
            # 释放低精度分片参数
            self._free_low_precision_sharded_param()
        # 检查并确保 `self.flat_param` 在计算设备上
        self._check_on_compute_device(self.flat_param)

    def _free_low_precision_sharded_param(self):
        """Frees the low precision sharded flat parameter."""
        # 检查低精度分片参数
        self._check_low_precision_shard()
        # `_mp_shard` 在预分片流中分配，在分片策略的分片流中使用，
        # 以及在 `NO_SHARD` 中在分片和默认流中使用。
        # 对于分片策略，当前流是分片流；对于 `NO_SHARD`，当前流是默认流。
        # 对于 `NO_SHARD`，仅记录默认流即可，因为默认流会等待分片流。
        _no_dispatch_record_stream(
            self.flat_param._mp_shard, self._device_handle.current_stream()  # type: ignore[attr-defined]
        )
        # 释放 `_mp_shard` 的存储空间
        _free_storage(self.flat_param._mp_shard)  # type: ignore[attr-defined]

    @torch.no_grad()
    def unshard_grad(self):
        """
        Unshard the handle's ``FlatParameter``'s gradient.

        If all ranks have
        ``None`` gradient, then all original parameters will as well. This
        method performs an all-reduce and an all-gather. The additional
        all-reduce is tolerable since this method is not meant to be used on
        the computation critical path.

        Postcondition: ``_saved_grad_shard`` is defined and contains the value
        to set ``flat_param.grad`` after gradients are resharded.
        """
        # 如果没有使用分片策略，则切换到使用未分片的梯度视图并返回
        if not self.uses_sharded_strategy:
            self._use_unsharded_grad_views()
            return
        
        # 获取 FlatParameter 对象
        flat_param = self.flat_param
        # 检查 flat_param 是否已经是未分片状态
        self._check_unsharded(flat_param)

        # 检查所有进程是否都有 None 梯度
        num_grad_none = torch.zeros(1, dtype=torch.int32, device=self.device)
        num_grad_none[0] = flat_param.grad is None
        dist.all_reduce(num_grad_none, group=self.process_group)
        
        # 如果所有进程都有 None 梯度，则设置 _saved_grad_shard 为 None，切换到未分片梯度视图并返回
        if num_grad_none[0] == self.world_size:
            flat_param._saved_grad_shard = None  # type: ignore[assignment]
            self._use_unsharded_grad_views()
            return

        # 如果 flat_param.grad 是 None，则在部分进程有 None 梯度的情况下，使用零张量来近似
        if flat_param.grad is None:
            if self._debug_level == dist.DebugLevel.INFO:
                # 发出警告，说明只有部分进程的 FlatParameter 梯度为 None
                warnings.warn(
                    f"[Rank {self.rank}] Only some but not all ranks have a "
                    "`None` `FlatParameter` gradient, so FSDP is using zeros to "
                    "approximate those ranks' sharded gradients being `None`"
                )
            flat_param._saved_grad_shard = None  # type: ignore[assignment]
            # 创建零张量作为 sharded_grad，类型为未分片大小
            sharded_grad = torch.zeros(flat_param._sharded_size, device=self.device)  # type: ignore[attr-defined]
        else:
            # 检查 flat_param.grad 是否为分片状态
            self._check_sharded(flat_param.grad)
            # 将 flat_param.grad 存储到 _saved_grad_shard 中
            flat_param._saved_grad_shard = flat_param.grad  # type: ignore[attr-defined]
            # 将 sharded_grad 设置为 flat_param._saved_grad_shard
            sharded_grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
        
        # 创建空的 padded_unsharded_grad 张量，用于存储未分片梯度
        padded_unsharded_grad = torch.empty(
            flat_param._padded_unsharded_size,  # type: ignore[attr-defined]
            device=self.device,
            dtype=sharded_grad.dtype,
        )
        # 执行全局 all_gather 操作，将 sharded_grad 收集到 padded_unsharded_grad 中
        dist.all_gather_into_tensor(
            padded_unsharded_grad, sharded_grad, self.process_group
        )
        # 计算未分片梯度的大小
        unsharded_size = self.flat_param._unpadded_unsharded_size
        # 将 padded_unsharded_grad 的前 unsharded_size.numel() 个元素视图为 unsharded_size 大小的 flat_param.grad
        flat_param.grad = padded_unsharded_grad[: unsharded_size.numel()].view(
            unsharded_size
        )
        # 使用未分片梯度视图
        self._use_unsharded_grad_views()

    def reshard_grad(self):
        # 如果使用原始参数，则切换到使用分片梯度视图
        if self._use_orig_params:
            self._use_sharded_grad_views()
        # 如果不使用分片策略，则直接返回
        if not self.uses_sharded_strategy:
            return
        # 恢复 flat_param 的梯度为 _saved_grad_shard
        self.flat_param.grad = self.flat_param._saved_grad_shard  # type: ignore[attr-defined]
        # 删除 flat_param 的 _saved_grad_shard 属性
        delattr(self.flat_param, "_saved_grad_shard")
    def prepare_gradient_for_optim(self):
        """Prepare the gradient for optimizer computation by moving the sharded gradient to the ``.grad`` attribute."""

        def cast_grad_to_param_dtype_if_needed(flat_param):
            # TODO (rohan-varma): test for full precision with keep_low_precision_grads
            # 如果不强制全精度且保留低精度梯度，确保梯度不为空，将梯度数据类型转换为设定的前向/后向参数数据类型
            if not self._force_full_precision and self._keep_low_precision_grads:
                _p_assert(flat_param.grad is not None, "Unexpected None grad!")
                if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                    flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                    if self._use_orig_params:
                        self._use_sharded_grad_views()

        flat_param = self.flat_param
        # TODO (awgu): We should replace these conditional checks to encode
        # the logical intention more directly.
        # 如果 `flat_param` 具有 `_cpu_grad` 属性
        if hasattr(flat_param, "_cpu_grad"):
            # NOTE: This branch includes `NO_SHARD`.
            # 检查是否为分片参数，执行分片检查
            self._check_sharded(flat_param)
            # 检查梯度是否在 CPU 上
            self._check_on_cpu(flat_param)
            # 将 `_cpu_grad` 赋值给 `flat_param.grad`
            flat_param.grad = flat_param._cpu_grad  # type: ignore[attr-defined]
            # 根据需要将梯度数据类型转换为设定的前向/后向参数数据类型
            cast_grad_to_param_dtype_if_needed(flat_param)
        # 如果 `flat_param` 具有 `_saved_grad_shard` 属性
        elif hasattr(flat_param, "_saved_grad_shard"):
            # 检查是否为分片参数，执行分片检查
            self._check_sharded(flat_param)
            # 检查梯度是否在计算设备上
            self._check_on_compute_device(flat_param)
            # 如果 `_saved_grad_shard` 不为空，检查其是否在计算设备上
            if flat_param._saved_grad_shard is not None:
                self._check_on_compute_device(flat_param._saved_grad_shard)  # type: ignore[attr-defined]
            # 如果在后向传播后调用了 `_post_backward_called`，则将 `_saved_grad_shard` 赋给 `flat_param.grad`
            if flat_param._post_backward_called:  # type: ignore[attr-defined]
                flat_param.grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
                # 如果梯度不为空，根据需要将梯度数据类型转换为设定的前向/后向参数数据类型
                if flat_param.grad is not None:
                    cast_grad_to_param_dtype_if_needed(flat_param)
        else:
            _p_assert(
                not self.uses_sharded_strategy
                or not flat_param._post_backward_called,  # type: ignore[attr-defined]
                "All sharded parameters that received a gradient in the "
                "post-backward should use `_saved_grad_shard`",
            )
        # 删除 `_saved_grad_shard`，因为其存在表示在后向传播钩子中累积前一个梯度
        if hasattr(flat_param, "_saved_grad_shard"):
            delattr(flat_param, "_saved_grad_shard")
    def to_cpu(self):
        """
        Move the unpadded unsharded flat parameter to CPU while in the context and moves it back to the previous device upon exit.

        For now, this assumes the ``FlatParameter`` is the unpadded unsharded flat parameter
        since (1) there is no reason to include the padding in the copy and (2)
        there is no use case for the sharded flat parameter.

        Precondition: ``self.flat_param`` 's data is the unpadded unsharded
        flat parameter on the compute device, and the handle uses a sharded
        strategy.
        Postcondition: Same as the precondition.
        """
        # 检查当前是否使用分片策略
        self._check_sharded_strategy()
        # 断言当前 flat_param 的大小与未填充、未分片大小相同
        _p_assert(
            self.flat_param.size() == self.flat_param._unpadded_unsharded_size,
            f"Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}",
        )
        # 检查 flat_param 是否在计算设备上
        self._check_on_compute_device(self.flat_param)
        # 断言未填充、未分片的 flat_param 是预期填充的 flat_param 的视图
        # 注意：这个检查对正确性来说不是严格必需的，但对于检查张量仅在内部使用时是有用的健全性检查
        _p_assert(
            _same_storage(self.flat_param, self._get_padded_unsharded_flat_param()),
            "Expects the unpadded parameter to be a view into the padded parameter",
        )
        # 将 flat_param 移动到 CPU
        self.flat_param_to(torch.device("cpu"))
        # 释放未分片的 flat_param
        self._free_unsharded_flat_param()
        try:
            # 执行代码块
            yield
        finally:
            # 再次断言当前 flat_param 的大小与未填充、未分片大小相同
            _p_assert(
                self.flat_param.size() == self.flat_param._unpadded_unsharded_size,
                f"Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}",
            )
            # 分配填充的未分片 flat_param
            padded_unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
            # 从 CPU 复制回计算设备
            padded_unsharded_flat_param[: self.flat_param.numel()].copy_(
                self.flat_param
            )
            # 使用未分片的 flat_param
            self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def reshard(self, free_unsharded_flat_param: bool):
        """
        Run the reshard logic.

        This includes freeing the unsharded flat
        parameter if ``free_unsharded_flat_param`` and switching to using the
        sharded flat parameter. Note that this also implicitly offloads
        the sharded flat parameter (if CPU offload is enabled) by pointing
        it to the ``_local_shard`` attribute which resides on CPU.
        """
        # 切换到分片的 `FlatParameter`，在释放之前防止与外部分析工具中的“使用后释放”类似的 bug，
        # 对于 `use_orig_params=True`，在 `_use_sharded_views()` 中设置 `param.data = ...` 时，
        # `param` 指向的内存不是有效的
        self._use_sharded_flat_param()
        # 如果需要释放未分片的 flat_param
        if free_unsharded_flat_param:
            self._free_unsharded_flat_param()
    def post_reshard(self):
        """
        Run the post-reshard logic.

        This includes freeing any memory that
        can now be freed given that the ``FlatParameter`` points to the full
        precision sharded flat parameter.

        Precondition: ``self.flat_param`` 's data points to the full precision
        sharded flat parameter.
        """
        # 如果当前对象使用混合精度参数，并且没有使用分片策略，并且没有强制使用完整精度的低精度分片
        if (
            self._uses_param_mixed_precision
            and not self.uses_sharded_strategy
            and not self._force_full_precision  # did not use the low precision shard
        ):
            # 调用方法释放低精度的分片参数
            self._free_low_precision_sharded_param()

    def _free_unsharded_flat_param(self):
        """
        Free the padded unsharded flat parameter. We allow this
        function to be called even when storage is not allocated

        The tensor to free depends
        on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
        # 检查是否使用了分片策略
        self._check_sharded_strategy()
        # 获取未分片的填充平坦参数
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        # 检查未分片的平坦参数是否在计算设备上
        self._check_on_compute_device(unsharded_flat_param)
        # 等待当前流中的所有操作完成后再释放内存
        _no_dispatch_record_stream(
            unsharded_flat_param, self._device_handle.current_stream()
        )
        # 调用内部方法释放存储空间
        _free_storage(unsharded_flat_param)
    # 定义一个方法 `_use_sharded_flat_param`，用于处理参数的分片平坦化使用
    def _use_sharded_flat_param(self) -> None:
        """Switches to using the sharded flat parameter."""
        # 获取当前对象的平坦参数
        flat_param = self.flat_param
        # 如果正在使用原始参数
        if self._use_orig_params:
            # 判断是否在前向传播过程中
            in_forward = self._training_state == HandleTrainingState.FORWARD
            # 判断是否应该跳过使用分片视图
            skip_use_sharded_views = (
                torch.is_grad_enabled()
                and in_forward
                and self._sharding_strategy
                in NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
            )
            # 如果需要跳过使用分片视图，则只在必要时进行额外的 `.data` 调用
            if skip_use_sharded_views:
                unsharded_flat_param = flat_param.data
        # 如果需要卸载参数
        if self._offload_params:
            # 获取本地分片的设备信息，期望在 CPU 上
            device = flat_param._local_shard.device  # type: ignore[attr-defined]
            _p_assert(
                device == torch.device("cpu"),
                f"Expects the local shard to be on CPU but got {device}",
            )
        # 将参数数据设置为本地分片数据
        flat_param.data = flat_param._local_shard  # type: ignore[attr-defined]
        # 如果正在使用原始参数
        if self._use_orig_params:
            # 如果跳过使用分片视图已定义
            if skip_use_sharded_views:  # type: ignore[possibly-undefined]
                # 存储未分片的参数数据，用于跳过视图时使用
                self._unsharded_flat_param_for_skipped_views = unsharded_flat_param  # type: ignore[possibly-undefined]
            else:
                # 使用分片视图
                self._use_sharded_views()
            # 对于后向传播后的重新分片，我们可能尝试使用分片梯度视图
            # （或者如果在 `no_sync()` 中累积了梯度，则使用未分片梯度视图），
            # 但对于后向传播后的重新分片，我们将调用后进行 reduce-scatter。
            if (
                in_forward  # type: ignore[possibly-undefined]
                # 如果跳过使用分片视图，则跳过使用梯度视图
                # 因为暴露未分片参数与分片梯度可能会让用户感到困惑
                and not self._skipped_use_sharded_views
            ):
                # TODO: 如果我们直接使用填充进行梯度计算，则修改 `_unpadded_unsharded_size`
                accumulated_grad_in_no_sync = (
                    flat_param.grad is not None
                    and self.uses_sharded_strategy
                    and flat_param.grad.shape == flat_param._unpadded_unsharded_size
                )
                # 如果在 `no_sync()` 中累积了梯度，则使用未分片梯度视图
                if accumulated_grad_in_no_sync:
                    self._use_unsharded_grad_views()
                else:
                    self._use_sharded_grad_views()

    #########
    # VIEWS #
    #########
    # 定义一个私有方法 `_get_unflat_views_unaligned`，用于获取非对齐的未分块视图
    @no_type_check
    def _get_unflat_views_unaligned(
        self,
        tensor: Optional[torch.Tensor] = None,
    ) -> Iterator[Tensor]:
        """
        Return unflattened ``Tensor`` views into ``tensor``.

        If `tensor`` is ``None``,  ``flat_param`` is used. The unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
        # 获取当前对象的 flat_param 属性，即扁平化参数
        flat_param = self.flat_param
        # 如果传入的 tensor 参数为 None，则使用 flat_param 作为默认参数
        if tensor is None:
            tensor = flat_param
        # 生成器表达式，返回每个子张量的视图及其扩展处理后的结果
        views = (
            _ext_post_unflatten_transform(
                subtensor.view(shape),  # 将子张量按给定的形状进行视图重塑
                param_extension,         # 参数的扩展处理
                self._fsdp_extension,    # FSDP 扩展处理
            )
            for (subtensor, shape, param_extension) in zip(
                torch.split(tensor, flat_param._numels, dim=0),   # 按照 flat_param 的分割数量在指定维度上切分 tensor
                flat_param._shapes,     # 扁平参数的形状列表
                flat_param._param_extensions,   # 扁平参数的扩展列表
            )
        )
        return views

    @no_type_check
    def _get_unflat_views_aligned(
        self,
        tensor: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Return unflattened ``Tensor`` views into ``tensor`` with handling for padding.

        This method has the same contract as :meth:`_get_unflat_views_unaligned`
        except it checks for ``None`` placeholders representing padding for
        alignment, which may incur slightly more CPU overhead.
        """
        # 获取当前对象的 flat_param 属性，即扁平化参数
        flat_param = self.flat_param
        # 如果传入的 tensor 参数为 None，则使用 flat_param 作为默认参数
        if tensor is None:
            tensor = flat_param
        # 按照 flat_param 的带填充数目在指定维度上切分 tensor，得到切分后的张量列表
        splits: List[Tensor] = torch.split(
            tensor, flat_param._numels_with_padding, dim=0
        )
        idx = 0
        views: List[Tensor] = []
        # 遍历切分后的张量及其对应的填充掩码
        for split, is_padding in zip(splits, flat_param._is_padding_mask):
            # 如果是填充张量，则跳过处理
            if is_padding:
                continue
            # 对非填充张量进行视图重塑和扩展处理后加入 views 列表
            views.append(
                _ext_post_unflatten_transform(
                    split.view(flat_param._shapes[idx]),   # 将切分后的张量按指定形状进行视图重塑
                    flat_param._param_extensions[idx],     # 参数的扩展处理
                    self._fsdp_extension,                  # FSDP 扩展处理
                )
            )
            idx += 1
        return views

    @no_type_check
    @torch.enable_grad()
    @no_type_check
    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator:
        """
        Unflatten the original parameters.

        The function assumes that the flat parameter is unsharded. When in the context,
        unflattens the original parameters as ``nn.Parameter`` views into the
        flat parameter, and after the context, restores the original parameters
        as ``Tensor`` views into the flat parameter.
        """
        # 使用未分片视图作为参数调用 _use_unsharded_views 方法
        self._use_unsharded_views(as_params=True)
        try:
            yield   # 执行上下文管理器内部逻辑
        finally:
            # 在上下文结束后，使用非参数视图调用 _use_unsharded_views 方法
            self._use_unsharded_views(as_params=False)

    @no_type_check
    @torch.no_grad()
    def _use_sharded_views(self) -> None:
        """
        将原始参数变量的数据设置为对分片平坦参数的展平视图。

        这些视图保持为展平状态，以简化参数在多个进程间分片的情况。那些数据不在分片平坦参数中的参数，其数据被设置为大小为 0 的空张量。
        我们不删除它们，以确保保留模型可打印性等预期行为。数据存在于分片平坦参数中的参数必须保留其变量，以便传递给优化器。
        """
        self._unsharded_flat_param_for_skipped_views = None
        if not self.uses_sharded_strategy:
            """
            对于 `NO_SHARD`，使用 *未展平* 的未分片视图，因为我们有未分片参数。
            """
            self._use_unsharded_views(as_params=True)
            return
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        """
        为所有未在本地分片中的参数构造一次并重复使用。
        """
        size_0_empty_tensor = torch.empty(
            0,
            dtype=self.flat_param.dtype,  # 如果 `flat_param` 改变了 dtype
            device=self.flat_param.device,
            requires_grad=False,
        )
        for param, shard_param_info, (param_name, module, _) in zip(
            flat_param._params, flat_param._shard_param_infos, flat_param._param_infos
        ):
            self._setattr_param(module, param_name, param)
            if not shard_param_info.in_shard:
                """
                允许通过垃圾回收释放原始数据。
                """
                param.data = size_0_empty_tensor
            else:
                offset = shard_param_info.offset_in_shard
                numel_in_shard = shard_param_info.numel_in_shard
                param.data = flat_param[offset : offset + numel_in_shard]
        assert self.flat_param._shared_params is not None
        for i, (
            param,
            (param_name, module, _, prim_param_name, prim_module, _),
        ) in enumerate(
            zip(self.flat_param._shared_params, self.flat_param._shared_param_infos)
        ):
            self._setattr_param(module, param_name, param)
            prim_param = getattr(prim_module, prim_param_name)
            param.data = prim_param  # 可能为空，也可能非空
        if self._training_state == HandleTrainingState.BACKWARD_POST:
            """
            由于现在不再需要，清除保存的 `Tensor`。
            """
            for i in range(len(self.flat_param._tensors)):
                self.flat_param._tensors[i] = None
    # 设置原始参数变量的梯度为对分片平坦参数梯度的展平视图
    def _use_sharded_grad_views(self) -> None:
        """
        Set the original parameter variables' gradients to be flattened views into the sharded flat parameter's gradient.

        This is a no-op if there is no gradient.

        Parameters whose data is not present in the sharded flat parameter and
        parameters with ``requires_grad=False`` have their gradients set to
        ``None``. Since the gradient variables do not need to be preserved,
        this method does not manipulate existing ``Tensor`` data directly and
        creates new ``Tensor`` variables instead.
        """
        # 获取平坦参数
        flat_param = self.flat_param
        # 检查平坦参数是否已分片
        self._check_sharded(flat_param)
        # 获取分片梯度
        grad = self.sharded_grad
        # 如果梯度为None，则将参数的梯度设置为None
        if grad is None:
            for param in chain(flat_param._params, flat_param._shared_params):
                param.grad = None
            return
        # 再次检查平坦参数是否已分片
        self._check_sharded(grad)
        # 遍历参数、分片参数信息和梯度是否为None的标志
        for param, shard_param_info, is_grad_none in zip(
            flat_param._params,
            flat_param._shard_param_infos,
            flat_param._is_grad_none_mask,
        ):
            # 如果参数不在分片中，则将梯度设置为None
            if not shard_param_info.in_shard:
                param.grad = None
            else:
                numel_in_shard = shard_param_info.numel_in_shard
                # 如果参数需要梯度且梯度不为None
                if param.requires_grad and not is_grad_none:
                    offset = shard_param_info.offset_in_shard
                    # 如果需要保留低精度梯度或参数的数据类型与梯度的数据类型不同
                    if self._keep_low_precision_grads or param.dtype != grad.dtype:
                        # 注意：这是一个使用`.data`的技巧，绕过参数/梯度数据类型匹配的检查。
                        # 这里，`param`具有全精度；`grad`具有低精度。
                        if param.grad is None:
                            # `.grad`必须与`param`具有相同的形状
                            param.grad = torch.empty_like(param)
                        param.grad.data = grad[
                            offset : offset + numel_in_shard
                        ].reshape(param.shape)
                    else:
                        param.grad = grad[offset : offset + numel_in_shard].reshape(
                            param.shape
                        )
                else:
                    param.grad = None
        # 断言共享参数不为None
        assert flat_param._shared_params is not None
        # 遍历共享参数和共享参数信息
        for i, (param, (_, _, _, prim_param_name, prim_module, _)) in enumerate(
            zip(flat_param._shared_params, flat_param._shared_param_infos)
        ):
            in_sharded_flat_param = hasattr(prim_module, prim_param_name)
            # 如果在分片平坦参数中且参数需要梯度
            if in_sharded_flat_param and param.requires_grad:
                prim_param = getattr(prim_module, prim_param_name)
                param.grad = prim_param.grad  # 共享相同的引用
            else:
                param.grad = None

    @no_type_check
    @torch.no_grad()
    def _writeback_tensor(
        self,
        src_tensor: Optional[Tensor],
        dst_tensor: Tensor,
        tensor_index: int,
        expected_shape: torch.Size,
        offset: int,
        is_param: bool,  # else gradient
    ) -> None:
        """
        Write back ``src_tensor`` to ``dst_tensor`` at offset ``offset``, where ``src_tensor`` should have shape ``expected_shape``.

        ``is_param`` indicates if the tensor is the parameter (if ``True``) or gradient (if
        ``False``). If ``src_tensor`` is ``None``, then the effect is zeroing
        instead of copying. ``tensor_index`` gives the index of ``src_tensor``
        in the metadata structures.

        Raises:
            RuntimeError: If the ``src_tensor`` does not have the expected
            shape.
        """
        # 检查期望的形状是否为一维，如果不是则抛出断言错误
        _p_assert(
            len(expected_shape) == 1,
            f"Expects a 1D expected shape but got {expected_shape}",
        )
        # 如果调试级别是 INFO，则获取当前进程的排名，并发出警告信息
        if self._debug_level == dist.DebugLevel.INFO:
            rank = self.rank if hasattr(self, "rank") else dist.get_rank()
            # 如果存在 src_tensor，则获取其形状和设备信息；否则置为 None
            src_shape = src_tensor.shape if src_tensor is not None else None
            src_device = src_tensor.device if src_tensor is not None else None
            warnings.warn(
                f"[Rank {rank}] {'Parameter' if is_param else 'Gradient'} needs "
                f"writeback in {self._training_state}\n"
                f"expected shape={expected_shape} shape={src_shape} "
                f"expected device={dst_tensor.device} device={src_device}"
            )
        # 如果 src_tensor 存在且其形状与期望形状不一致，则抛出运行时错误
        if src_tensor is not None and src_tensor.shape != expected_shape:
            raise RuntimeError(
                f"Cannot writeback when the {'parameter' if is_param else 'gradient'} "
                f"shape changes\nExpects {expected_shape} but got {src_tensor.shape}"
            )
        # 如果 src_tensor 存在，则将其内容复制到 dst_tensor 的指定偏移位置
        if src_tensor is not None:
            dst_tensor[offset : offset + expected_shape.numel()].copy_(src_tensor)
        # 否则，在 dst_tensor 的指定偏移位置进行零填充，并设置梯度不存在的掩码标记
        else:
            dst_tensor[offset : offset + expected_shape.numel()].zero_()
            assert self.flat_param._is_grad_none_mask is not None
            self.flat_param._is_grad_none_mask[tensor_index] = True
    def _reset_flat_param_grad_info_if_needed(self):
        """
        Reset ``flat_param.grad`` if needed.

        When ``use_orig_params=True``:
        (1) sets the underlying ``flat_param.grad`` to ``None`` if *all* of the
        original parameters' ``.grad`` are ``None``, and
        (2) sets ``flat_param.requires_grad=False`` if *none* of the original
        parameters require gradient.
        For (1), this is targeting ``optim.zero_grad(set_to_none=True)``, in
        which case we want to free the gradients as soon after the
        ``zero_grad()`` call as possible.
        """
        # 如果不使用原始参数，则直接返回
        if not self._use_orig_params:
            return
        # 获取当前的 flat_param
        flat_param = self.flat_param
        # 确保 flat_param._params 不为 None，这里是类型检查
        assert flat_param._params is not None  # mypy
        # 初始化标志位，判断是否所有原始参数的梯度都为 None，并且是否有参数需要梯度
        all_grad_none = True
        requires_grad = False
        # 遍历 flat_param 的所有参数
        for param in flat_param._params:
            # 判断是否所有参数的梯度都为 None
            all_grad_none &= param.grad is None
            # 判断是否有参数需要梯度
            requires_grad |= param.requires_grad
        # 如果所有参数的梯度都为 None，则将 flat_param.grad 设置为 None
        if all_grad_none:
            flat_param.grad = None
        # 只要有一个参数需要梯度，就将 flat_param.requires_grad 设为 True
        flat_param.requires_grad = requires_grad

    def _deregister_orig_params(self):
        # 遍历 flat_param._param_infos，删除模块中注册的原始参数
        for param_info in self.flat_param._param_infos:
            param_name, module, _ = param_info
            if hasattr(module, param_name):
                delattr(module, param_name)
        # 遍历 flat_param._shared_param_infos，删除共享参数中模块中注册的参数
        for param_name, module, _, _, _, _ in self.flat_param._shared_param_infos:
            if hasattr(module, param_name):
                delattr(module, param_name)

    ###########
    # HELPERS #
    ###########

    def flat_param_to(self, *args, **kwargs):
        """Wrap an in-place call to ``.to()`` for ``self.flat_param``."""
        # 将 self.flat_param 转移到指定的设备上
        self.flat_param.data = self.flat_param.to(*args, **kwargs)
        # 如果使用原始参数，刷新视图，因为它们的存储可能已经改变
        if self._use_orig_params:
            # 如果 flat_param 是分片的，则使用分片视图
            if self.is_sharded(self.flat_param):
                self._use_sharded_views()
            else:
                self._use_unsharded_views(as_params=True)

    def _get_modules(self) -> Set[nn.Module]:
        """Return a :class:`set` of the modules whose parameters are included in this handle's flat parameter."""
        # 返回包含在 flat_param._param_infos 和 flat_param._shared_param_infos 中的模块集合
        return {pi.module for pi in self.flat_param._param_infos}.union(
            {spi.module for spi in self.flat_param._shared_param_infos}
        )

    def is_sharded(self, tensor: Tensor) -> bool:
        """
        Return whether ``tensor`` is *currently* sharded.

        For ``NO_SHARD``, we choose to have this always return ``False`` for clarity.
        """
        # 如果 flat_param 没有 _sharded_size 属性或者不使用分片策略，则返回 False
        if (
            not hasattr(self.flat_param, "_sharded_size")
            or not self.uses_sharded_strategy
        ):
            return False
        # 获取 flat_param 的分片大小
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        # 判断 tensor 的大小是否与分片大小相同
        return tensor.size() == sharded_size
    def param_module_names(self) -> Iterator[Tuple[str, str]]:
        # 创建一个包含参数信息的列表，其中每个元素是 ParamInfo 对象，表示参数名、模块和模块名
        shared_param_infos = [
            ParamInfo(param_name, module, module_name)
            for (
                param_name,
                module,
                module_name,
                _,
                _,  # 忽略的额外项
                _,  # 忽略的额外项
            ) in self.flat_param._shared_param_infos
        ]
        # 遍历所有参数信息，包括 flat_param._param_infos 和 shared_param_infos
        for param_info in chain(self.flat_param._param_infos, shared_param_infos):
            param_name, _, module_name = param_info  # type: ignore[misc]  # 解包 param_info
            yield (param_name, module_name)  # 返回参数名和模块名的元组迭代器

    def shared_param_module_names(self) -> Iterator[Tuple[str, str]]:
        # 遍历所有共享参数的信息，创建 ParamInfo 对象，表示参数名、模块和模块名
        for param_name, _, module_name in [
            ParamInfo(param_name, module, module_name)
            for (
                param_name,
                module,
                module_name,
                _,
                _,  # 忽略的额外项
                _,  # 忽略的额外项
            ) in self.flat_param._shared_param_infos
        ]:
            yield (param_name, module_name)  # 返回参数名和模块名的元组迭代器

    @property
    def _fqns_in_shard(self) -> List[str]:
        """Return the FQNs of the parameters present in this rank's shard."""
        fqns_in_shard: List[str] = []  # 存储此排名分片中存在的参数的全限定名列表
        # 遍历 flat_param._fqns 和 flat_param._shard_param_infos 的并行元素
        for fqn, shard_param_info in zip(
            self.flat_param._fqns, self.flat_param._shard_param_infos  # type: ignore[attr-defined]
        ):
            if shard_param_info.in_shard:  # 如果 shard_param_info 表示参数在分片中
                fqns_in_shard.append(fqn)  # 将全限定名添加到列表中
        return fqns_in_shard  # 返回全限定名列表

    @property
    def sharded_grad(self) -> Optional[Tensor]:
        """Return the handle's sharded gradient."""
        flat_param = self.flat_param  # 获取 flat_param 的引用
        # 优先级为非 `None` 的情况： `_cpu_grad` > `_saved_grad_shard` > `grad`
        # - CPU offloading: `_cpu_grad`
        # - 无 CPU offloading + 分片策略: `_saved_grad_shard`
        # - 无 CPU offloading + `NO_SHARD`: `grad`
        grad: Optional[Tensor]
        if hasattr(flat_param, "_cpu_grad"):  # 如果 flat_param 有 `_cpu_grad` 属性
            grad = flat_param._cpu_grad  # type: ignore[attr-defined]
        elif hasattr(flat_param, "_saved_grad_shard"):  # 如果 flat_param 有 `_saved_grad_shard` 属性
            # 在后向传播钩子中，分片梯度仍在 `_saved_grad_shard` 中
            grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
        else:
            # 如果处于 IDLE 或 FORWARD 状态，则可能存在（累积的）梯度
            # 如果在 IDLE 状态访问，则应该是因为重新注册原始参数（例如在状态字典加载时）
            _p_assert(
                flat_param.grad is None
                or not self.uses_sharded_strategy
                or self._training_state
                in (HandleTrainingState.FORWARD, HandleTrainingState.IDLE),
                "Sharded strategies should use `_cpu_grad` or `_saved_grad_shard` "
                "unless in IDLE or FORWARD",
            )
            grad = flat_param.grad  # 否则使用 flat_param 的梯度
        return grad  # 返回梯度对象或 None
    #######################
    # CHECKS & INVARIANTS #
    #######################

    # 检查分片策略是否启用，若未启用则引发断言错误
    def _check_sharded_strategy(self):
        _p_assert(self.uses_sharded_strategy, "Expects sharded strategy")

    # 检查张量是否位于计算设备上，若不在指定设备上则引发断言错误
    def _check_on_compute_device(self, tensor: Tensor):
        _p_assert(
            tensor.device == self.device,
            f"Expects tensor to be on the compute device {self.device}, was on {tensor.device}",
        )

    # 检查张量是否位于 CPU 上，若不在 CPU 上则引发断言错误
    def _check_on_cpu(self, tensor: Tensor):
        _p_assert(
            tensor.device == torch.device("cpu"),
            f"Expects tensor to be on CPU but got {tensor.device}",
        )

    # 静态方法：检查张量的存储是否已被释放，若未释放则引发断言错误
    @staticmethod
    def _check_storage_freed(tensor: Tensor):
        # 在编译期间不进行调整大小
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            _p_assert(
                _same_storage_size(tensor, 0),
                "Expects storage to be freed but got storage with size > 0",
            )

    # 静态方法：检查张量的存储是否已分配，若未分配则引发断言错误
    @staticmethod
    def _check_storage_allocated(tensor: Tensor):
        _p_assert(_storage_size_allocated(tensor), "Expects storage to be allocated")

    # 检查是否使用低精度分片策略，若未使用则引发断言错误
    def _check_low_precision_shard(self):
        _p_assert(
            self._uses_param_mixed_precision,
            "Not using low precision for parameters",
        )
        _p_assert(
            getattr(self.flat_param, "_mp_shard", None) is not None,
            "Expects `_mp_shard` to exist",
        )
        device = self.flat_param._mp_shard.device  # type: ignore[attr-defined]
        _p_assert(
            device == self.device,
            f"Expects the low precision shard to be on {self.device} but got {device}",
        )
    # 检查未分片张量的有效性，并确保其大小与期望的未分片大小一致
    def _check_unsharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be unsharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        # 获取未分片参数的未填充未分片大小
        unsharded_size = self.flat_param._unpadded_unsharded_size
        _p_assert(
            tensor.size() == unsharded_size,
            msg_prefix + f"with size {unsharded_size} but got {tensor.size()}",
        )

    # 检查分片张量的有效性，并确保其大小与期望的分片大小一致
    def _check_sharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be sharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        # 获取分片参数的分片大小
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        _p_assert(
            tensor.size() == sharded_size,
            msg_prefix + f"with size {sharded_size} but got {tensor.size()}",
        )

    ##############
    # PROPERTIES #
    ##############

    @property
    def uses_sharded_strategy(self) -> bool:
        # 检查当前是否使用分片策略
        return self._sharding_strategy != HandleShardingStrategy.NO_SHARD

    @property
    def _uses_param_mixed_precision(self) -> bool:
        # 检查是否使用参数混合精度
        return self._fwd_bwd_param_dtype != self._orig_param_dtype

    @property
    def _uses_reduce_mixed_precision(self) -> bool:
        # 检查是否使用减少混合精度
        return self._reduce_dtype != self._orig_param_dtype

    @property
    def _force_full_precision(self) -> bool:
        # 检查是否强制使用完整精度，如果参数或减少使用混合精度，并且处于召唤完整参数状态或模型评估模式未使用混合精度，则返回True
        return (
            self._uses_param_mixed_precision or self._uses_reduce_mixed_precision
        ) and (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            or
            # 在模型评估模式下，如果未配置使用完整精度，也禁用混合精度
            (not self._fully_sharded_module.training and self._use_full_prec_in_eval)
        )

    @property
    def _skipped_use_sharded_views(self) -> bool:
        """
        This property is used for sharding strategies that do not free after forward with ``use_orig_params=True``.

        This returns if this handle is
        currently in a state where it has skipped using sharded views, in which
        case it can restore view invariants via ``_use_sharded_views()``.
        """
        # 返回是否当前处于跳过使用分片视图状态，适用于在前向传播中未释放的分片策略，并通过``use_orig_params=True``还原视图不变性
        return self._unsharded_flat_param_for_skipped_views is not None
# NOTE: These are hacks to bypass `nn.Module.__setattr__` checks.
# 给 `nn.Module.__setattr__` 检查绕过的一些变量设置函数

def _unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    # 直接设置参数到 `_parameters` 字典中，绕过可能存在的 `nn.Module` 子类的重写
    module._parameters[param_name] = param
    # 绕过 `nn.Module` 子类重写的设置，直接调用父类 `nn.Module` 的 `__setattr__` 方法
    super(nn.Module, module).__setattr__(param_name, param)


def _unsafe_setattr_tensor(module: nn.Module, param_name: str, tensor: Tensor) -> None:
    # 从 `_parameters` 字典中移除参数名对应的项，绕过可能存在的 `nn.Module` 子类的重写
    module._parameters.pop(param_name, None)
    # 绕过 `nn.Module` 子类重写的设置，直接调用父类 `nn.Module` 的 `__setattr__` 方法
    super(nn.Module, module).__setattr__(param_name, tensor)


def _safe_setattr_tensor_or_param(
    module: nn.Module, param_name: str, tensor_or_param: Union[Tensor, nn.Parameter]
):
    # 调用 `delattr()` 和 `setattr()` 方法，以通过 `nn.Module` 的检查
    if hasattr(module, param_name):
        delattr(module, param_name)
    setattr(module, param_name, tensor_or_param)


def _convert_to_params(
    tensors: List[Union[torch.Tensor, nn.Parameter]]
) -> List[nn.Parameter]:
    # 将输入列表中的张量转换为 `nn.Parameter` 类型的列表
    return [t if isinstance(t, nn.Parameter) else nn.Parameter(t) for t in tensors]


def _detach_if_needed(param_or_tensor: Union[nn.Parameter, Tensor]) -> Tensor:
    # 如果输入是 `nn.Parameter` 类型，则执行 `detach()` 方法，否则直接返回输入
    return (
        param_or_tensor.detach()
        if isinstance(param_or_tensor, nn.Parameter)
        else param_or_tensor
    )


def _get_aligned_numel(unsharded_dtype: torch.dtype):
    # NOTE: This alignment constraint comes from TorchInductor.
    # 根据 TorchInductor 给出的对齐约束，设置对齐的元素数量
    ALIGNMENT = 16  # bytes
    # 计算未分片数据类型的字节大小
    unsharded_dtype_size = _get_dtype_size(unsharded_dtype)
    # 计算对齐后的元素数量
    aligned_numel = ALIGNMENT // unsharded_dtype_size
    return aligned_numel


@functools.lru_cache(8)
def _get_dtype_size(dtype):
    # 返回给定数据类型 `dtype` 的元素大小
    return torch.empty((), dtype=dtype).element_size()


def _construct_padding_tensor(
    padding_numel: int, dtype: torch.dtype, requires_grad: bool, device: torch.device
):
    # NOTE: Set the padding value as a magic number for debuggability. The
    # value itself should never be used in any user-facing computation.
    # 为了便于调试性，设置填充值为一个魔术数，实际计算中不应使用这个值
    return (
        torch.ones(
            (padding_numel,), dtype=dtype, requires_grad=requires_grad, device=device
        )
        * _FLAT_PARAM_PADDING_VALUE
    )


# Use `lru_cache(1)` to only log the warning once (assuming the fixed warning
# messasge is passed in)
@functools.lru_cache(1)
def _warn_skip_writeback_check(log: logging.Logger, warning: str):
    # 使用 `lru_cache(1)` 以确保只记录一次警告
    logger.warning(warning)


# Use `lru_cache(1)` to only log the warning once
@functools.lru_cache(1)
def _warn_use_fake_all_gather(log: logging.Logger, warning: str):
    # 使用 `lru_cache(1)` 以确保只记录一次警告
    logger.warning(warning)


# Use `lru_cache(1)` to only log the warning once
@functools.lru_cache(1)
def _warn_use_fake_reduce(log: logging.Logger, warning: str):
    # 使用 `lru_cache(1)` 以确保只记录一次警告
    logger.warning(warning)


def _same_storage(a, b):
    # Params are DTensors in backward
    # with SHARD_GRAD_OP + TP
    from torch.distributed._tensor import DTensor

    if isinstance(a, DTensor):
        # 如果 `a` 是 `DTensor` 类型，则获取其本地张量 `_local_tensor`
        a = a._local_tensor
    # 检查变量 b 是否是 DTensor 类的实例
    if isinstance(b, DTensor):
        # 如果是 DTensor 类的实例，则将 b 替换为其内部的本地张量（_local_tensor）
        b = b._local_tensor
    
    # 返回变量 a 的未类型化存储的数据指针是否等于变量 b 的未类型化存储的数据指针
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
# 检查两个张量的存储大小是否相等，使用除元素大小外的存储大小进行比较
def _same_storage_size(a: torch.Tensor, b: int):
    # 获得张量的未类型化存储，并计算其大小，然后除以元素大小来得到存储大小
    return a.untyped_storage().size() // a.element_size() == b

# 检查给定张量的存储大小是否大于零
def _storage_size_allocated(tensor: Tensor):
    # 获得张量的未类型化存储，并获取其大小，然后检查其是否大于零
    storage_size: int = tensor.untyped_storage().size()
    return storage_size > 0
```