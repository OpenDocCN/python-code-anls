# `.\pytorch\torch\distributed\_composable\fsdp\_fsdp_api.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和类
from dataclasses import dataclass
from typing import Optional

import torch

# 定义一个名为MixedPrecisionPolicy的数据类，用于配置FSDP的混合精度设置
@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """
    This configures FSDP's mixed precision. Unlike autocast, this applies mixed
    precision at the module level, not op level, which means low-precision
    activations are saved for backward and high-to-low-precision casts are
    incurred only at module boundaries.

    FSDP works well with module-level mixed precision since it keeps the
    high-precision sharded parameters in memory anyway. In other words, FSDP
    does not require any extra memory to keep a high-precision copy of the
    parameters for the optimizer step.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for
            the unsharded parameter and hence the dtype for forward/backward
            computation and the parameter all-gather. If this is ``None``, then
            the unsharded parameter uses the original dtype. The optimizer step
            uses the sharded parameter in the original dtype. (Default:
            ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then the reduction
            uses the compute dtype. This can be used to run gradient reduction
            in full precision while using low precision for compute. If also
            gradient reduction is disabled via :meth:`set_requires_gradient_sync`,
            then FSDP will accumulate gradients using ``reduce_dtype``.
            (Default: ``None``)
        output_dtype (Optional[torch.dtype]): This specifies the dtype for
            casting floating-point forward outputs. This can be used to
            help implement cases where different modules have different mixed
            precision policies. (Default: ``None``)
        cast_forward_inputs (bool): This specifies whether FSDP should cast the
            forward's floating-point input tensors to ``param_dtype`` or not.
    """

    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True

    def __post_init__(self):
        # 在初始化后处理阶段，如果param_dtype与reduce_dtype相同，则将reduce_dtype设置为None，
        # 这是因为梯度计算使用param_dtype，如果reduce_dtype匹配，则不需要额外的类型转换
        if self.param_dtype == self.reduce_dtype:
            # 绕过数据类冻结检查，允许设置属性
            object.__setattr__(self, "reduce_dtype", None)


@dataclass
class OffloadPolicy:
    """This base class represents the policy of no offloading."""


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    """
    This offload policy offloads parameters, gradients, and optimizer states to
    # CPU. 在所有聚合之前，分片参数在主机到设备的复制。
    # 所有聚合后的参数根据 ``reshard_after_forward`` 进行释放。
    # 分片梯度在反向传播中在设备到主机的复制，并且优化器步骤在 CPU 上运行，使用 CPU 优化器状态。

    # 属性:
    #     pin_memory (bool): 是否固定分片参数和梯度内存。固定内存允许 H2D/D2H 复制而不阻塞 CPU，
    #         可以与计算重叠，但固定内存不能被其他进程使用。如果您的 CPU 内存不足，请将其设置为 ``False``。
    #         （默认: ``True``）
    pin_memory: bool = True
```