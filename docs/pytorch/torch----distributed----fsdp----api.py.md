# `.\pytorch\torch\distributed\fsdp\api.py`

```
"""
This file includes public APIs for FSDP such as the classes used for the
constructor arguments.
"""

# 导入必要的库和模块
from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type

# 导入 PyTorch 库
import torch
from torch.nn.modules.batchnorm import _BatchNorm

# 定义导出的模块列表
__all__ = [
    "ShardingStrategy",
    "BackwardPrefetch",
    "MixedPrecision",
    "CPUOffload",
    "StateDictType",
    "StateDictConfig",
    "FullStateDictConfig",
    "LocalStateDictConfig",
    "ShardedStateDictConfig",
    "OptimStateDictConfig",
    "FullOptimStateDictConfig",
    "LocalOptimStateDictConfig",
    "ShardedOptimStateDictConfig",
    "StateDictSettings",
]

# 定义分片策略枚举类
class ShardingStrategy(Enum):
    """
    This specifies the sharding strategy to be used for distributed training by
    :class:`FullyShardedDataParallel`.

    - ``FULL_SHARD``: Parameters, gradients, and optimizer states are sharded.
      For the parameters, this strategy unshards (via all-gather) before the
      forward, reshards after the forward, unshards before the backward
      computation, and reshards after the backward computation. For gradients,
      it synchronizes and shards them (via reduce-scatter) after the backward
      computation. The sharded optimizer states are updated locally per rank.
    - ``SHARD_GRAD_OP``: Gradients and optimizer states are sharded during
      computation, and additionally, parameters are sharded outside
      computation. For the parameters, this strategy unshards before the
      forward, does not reshard them after the forward, and only reshards them
      after the backward computation. The sharded optimizer states are updated
      locally per rank. Inside ``no_sync()``, the parameters are not resharded
      after the backward computation.
    - ``NO_SHARD``: Parameters, gradients, and optimizer states are not sharded
      but instead replicated across ranks similar to PyTorch's
      :class:`DistributedDataParallel` API. For gradients, this strategy
      synchronizes them (via all-reduce) after the backward computation. The
      unsharded optimizer states are updated locally per rank.
    - ``HYBRID_SHARD``: Apply ``FULL_SHARD`` within a node, and replicate parameters across
      nodes. This results in reduced communication volume as expensive all-gathers and
      reduce-scatters are only done within a node, which can be more performant for medium
      -sized models.
    - ``_HYBRID_SHARD_ZERO2``: Apply ``SHARD_GRAD_OP`` within a node, and replicate parameters across
      nodes. This is like ``HYBRID_SHARD``, except this may provide even higher throughput
      since the unsharded parameters are not freed after the forward pass, saving the
      all-gathers in the pre-backward.
    """

    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


class BackwardPrefetch(Enum):
    """
    Placeholder for the BackwardPrefetch enumeration class.
    """
    """
    This configures explicit backward prefetching, which improves throughput by
    enabling communication and computation overlap in the backward pass at the
    cost of slightly increased memory usage.

    - ``BACKWARD_PRE``: This enables the most overlap but increases memory
      usage the most. This prefetches the next set of parameters *before* the
      current set of parameters' gradient computation. This overlaps the *next
      all-gather* and the *current gradient computation*, and at the peak, it
      holds the current set of parameters, next set of parameters, and current
      set of gradients in memory.
    - ``BACKWARD_POST``: This enables less overlap but requires less memory
      usage. This prefetches the next set of parameters *after* the current
      set of parameters' gradient computation. This overlaps the *current
      reduce-scatter* and the *next gradient computation*, and it frees the
      current set of parameters before allocating memory for the next set of
      parameters, only holding the next set of parameters and current set of
      gradients in memory at the peak.
    - FSDP's ``backward_prefetch`` argument accepts ``None``, which disables
      the backward prefetching altogether. This has no overlap and does not
      increase memory usage. In general, we do not recommend this setting since
      it may degrade throughput significantly.

    For more technical context: For a single process group using NCCL backend,
    any collectives, even if issued from different streams, contend for the
    same per-device NCCL stream, which implies that the relative order in which
    the collectives are issued matters for overlapping. The two backward
    prefetching values correspond to different issue orders.
    """

    # NOTE: For both modes, the ordering that defines "current" and "next" is
    # not always exact in the current implementation. A mistargeted prefetch
    # simply means that the parameter memory is allocated earlier than needed,
    # possibly increasing peak memory usage, but does not affect correctness.
    
    # 使用 auto() 自动分配唯一值给 BACKWARD_PRE
    BACKWARD_PRE = auto()
    # 使用 auto() 自动分配唯一值给 BACKWARD_POST
    BACKWARD_POST = auto()
# 使用 @dataclass 装饰器声明一个数据类，用于配置 FSDP 本地混合精度训练
@dataclass
class MixedPrecision:
    """
    This configures FSDP-native mixed precision training.

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: In ``summon_full_params``, parameters are forced to full
        precision, but buffers are not.

    .. note:: Layer norm and batch norm accumulate in ``float32`` even when
        their inputs are in a low precision like ``float16`` or ``bfloat16``.
        Disabling FSDP's mixed precision for those norm modules only means that
        the affine parameters are kept in ``float32``. However, this incurs
        separate all-gathers and reduce-scatters for those norm modules, which
        may be inefficient, so if the workload permits, the user should prefer
        to still apply mixed precision to those modules.

    .. note:: By default, if the user passes a model with any ``_BatchNorm``
        modules and specifies an ``auto_wrap_policy``, then the batch norm
        modules will have FSDP applied to them separately with mixed precision
        disabled. See the ``_module_classes_to_ignore`` argument.

    .. note:: ``MixedPrecision`` has ``cast_root_forward_inputs=True`` and
        ``cast_forward_inputs=False`` by default. For the root FSDP instance,
        its ``cast_root_forward_inputs`` takes precedence over its
        ``cast_forward_inputs``. For non-root FSDP instances, their
        ``cast_root_forward_inputs`` values are ignored. The default setting is
        sufficient for the typical case where each FSDP instance has the same
        ``MixedPrecision`` configuration and only needs to cast inputs to the
        ``param_dtype`` at the beginning of the model's forward pass.
    """
    # 参数类型：用于指定模型参数的数据类型，可以为None表示未指定
    param_dtype: Optional[torch.dtype] = None
    # 减少类型：用于指定减少操作的数据类型，可以为None表示未指定
    reduce_dtype: Optional[torch.dtype] = None
    # 缓冲区类型：用于指定模型缓冲区的数据类型，可以为None表示未指定
    buffer_dtype: Optional[torch.dtype] = None
    # 保持低精度梯度：是否保持低精度的梯度，默认为False
    keep_low_precision_grads: bool = False
    # 转换前向输入：是否在前向传播时转换输入数据的类型，默认为False
    cast_forward_inputs: bool = False
    # 转换根前向输入：是否在根模块前向传播时转换输入数据的类型，默认为True
    cast_root_forward_inputs: bool = True
    # 忽略的模块类：不需要处理的模块类列表，默认为(_BatchNorm,)
    _module_classes_to_ignore: Sequence[Type[torch.nn.Module]] = (_BatchNorm,)
@dataclass
class CPUOffload:
    """
    This configures CPU offloading.

    Attributes:
        offload_params (bool): This specifies whether to offload parameters to
            CPU when not involved in computation. If ``True``, then this
            offloads gradients to CPU as well, meaning that the optimizer step
            runs on CPU.
    """

    offload_params: bool = False
    # 默认情况下，不启用参数的CPU卸载


class StateDictType(Enum):
    """
    This enum indicates that which type of ``state_dict`` the FSDP module is
    currently processing (returning or loading).
    The default value is FULL_STATE_DICT to comply the PyTorch convention.
    ..note::
        FSDP currently supports three types of ``state_dict``:
            1. ``state_dict/load_state_dict``: this pair of APIs return and load
               the non-sharded, unflattened parameters. The semantics is the
               same as using DDP.
            2. ``_local_state_dict/_load_local_state_dict``: this pair of APIs return
               and load local sharded, flattened parameters. The values returned
               by ``_local_state_dict`` can be directly used by FSDP and is only
               meaningful to FSDP (because parameters are flattened). Note that
               these APIs are meant for use via the :func:`state_dict_type`
               context manager as follows:
                   >>> # xdoctest: +SKIP("undefined variables")
                   >>> with fsdp.state_dict_type(StateDictType.LOCAL_STATE_DICT):
                   ...     state = fsdp.state_dict()  # loads local state dict
            3. ``_sharded_state_dict/_load_sharded_state_dict``: this pair of APIs
               return and load sharded, unflattened parameters. The ``state_dict``
               return by ``sharded_state_dict`` can be used by all other parallel
               schemes (resharding may be required).
    """

    FULL_STATE_DICT = auto()
    LOCAL_STATE_DICT = auto()
    SHARDED_STATE_DICT = auto()
    # 定义了三种state_dict类型：FULL_STATE_DICT、LOCAL_STATE_DICT和SHARDED_STATE_DICT


@dataclass
class StateDictConfig:
    """
    ``StateDictConfig`` is the base class for all ``state_dict`` configuration
    classes. Users should instantiate a child class (e.g.
    ``FullStateDictConfig``) in order to configure settings for the
    corresponding ``state_dict`` type supported by FSDP.

    Attributes:
        offload_to_cpu (bool): If ``True``, then FSDP offloads the state dict
            values to CPU, and if ``False``, then FSDP keeps them on GPU.
            (Default: ``False``)
    """

    offload_to_cpu: bool = False
    # 默认情况下，不将state dict的值卸载到CPU上


@dataclass
class FullStateDictConfig(StateDictConfig):
    """
    ``FullStateDictConfig`` is a config class meant to be used with
    ``StateDictType.FULL_STATE_DICT``. We recommend enabling both
    ``offload_to_cpu=True`` and ``rank0_only=True`` when saving full state
    dicts to save GPU memory and CPU memory, respectively. This config class
    is meant to be used via the :func:`state_dict_type` context manager as
    """
    # 如果为 True，则仅在进程的 rank 0 保存完整的状态字典，非零 rank 的进程保存空字典；如果为 False，则所有进程保存完整的状态字典。默认为 False。
    rank0_only: bool = False
@dataclass
class LocalStateDictConfig(StateDictConfig):
    pass


# `LocalStateDictConfig`类继承自`StateDictConfig`，没有额外的属性或方法
@dataclass
class ShardedStateDictConfig(StateDictConfig):
    """
    ``ShardedStateDictConfig``是一个配置类，用于与``StateDictType.SHARDED_STATE_DICT``一起使用。

    Attributes:
        _use_dtensor (bool): 如果为``True``，则FSDP将状态字典值保存为``DTensor``，
            如果为``False``，则保存为``ShardedTensor``。 (默认为``False``)

    .. warning:: ``_use_dtensor``是:class:`ShardedStateDictConfig`的私有字段，
        FSDP用它来确定状态字典值的类型。用户不应手动修改``_use_dtensor``。
    """

    _use_dtensor: bool = False


@dataclass
class OptimStateDictConfig:
    """
    ``OptimStateDictConfig``是所有``optim_state_dict``配置类的基类。
    用户应实例化子类（例如``FullOptimStateDictConfig``），以配置FSDP支持的相应``optim_state_dict``类型的设置。

    Attributes:
        offload_to_cpu (bool): 如果为``True``，则FSDP将状态字典的张量值卸载到CPU，
            如果为``False``，则FSDP将它们保留在原始设备上（默认为GPU，除非启用了CPU卸载参数）。 (默认为``True``)
    """

    offload_to_cpu: bool = True


@dataclass
class FullOptimStateDictConfig(OptimStateDictConfig):
    """
    Attributes:
        rank0_only (bool): 如果为``True``，则仅rank 0保存完整状态字典，
            非零rank保存空字典。如果为``False``，则所有rank保存完整状态字典。 (默认为``False``)
    """

    rank0_only: bool = False


@dataclass
class LocalOptimStateDictConfig(OptimStateDictConfig):
    offload_to_cpu: bool = False


@dataclass
class ShardedOptimStateDictConfig(OptimStateDictConfig):
    """
    ``ShardedOptimStateDictConfig``是一个配置类，用于与``StateDictType.SHARDED_STATE_DICT``一起使用。

    Attributes:
        _use_dtensor (bool): 如果为``True``，则FSDP将状态字典值保存为``DTensor``，
            如果为``False``，则保存为``ShardedTensor``。 (默认为``False``)

    .. warning:: ``_use_dtensor``是:class:`ShardedOptimStateDictConfig`的私有字段，
        FSDP用它来确定状态字典值的类型。用户不应手动修改``_use_dtensor``。
    """

    _use_dtensor: bool = False


@dataclass
class StateDictSettings:
    state_dict_type: StateDictType
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
```