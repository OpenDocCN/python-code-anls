# `.\pytorch\torch\distributed\algorithms\_comm_hooks\default_hooks.py`

```py
# mypy: allow-untyped-defs
# 导入 functools 模块
import functools
# 导入 Optional 类型
from typing import Optional

# 导入 torch 库
import torch
# 导入 torch 分布式模块
import torch.distributed as dist


class DefaultState:
    r"""
    存储在通信钩子中执行默认通信算法所需的状态。

    Args:
        process_group (ProcessGroup): 要使用的进程组。
    """

    __slots__ = [
        "process_group",
        "world_size",
        "gradient_predivide_factor",
        "gradient_postdivide_factor",
    ]

    def __init__(self, process_group: dist.ProcessGroup):
        # 如果 process_group 为 None，则抛出 ValueError 异常
        if process_group is None:
            raise ValueError(f"Expected to pass in an explicit ProcessGroup to {self}.")
        self.process_group = process_group
        # 获取进程组的全局大小
        self.world_size = dist.get_world_size(process_group)
        # 设置两个因子 self.gradient_predivide_factor 和 self.gradient_postdivide_factor
        # 以避免下溢和上溢
        self.gradient_predivide_factor = self._get_gradient_predivide_factor(
            self.world_size
        )
        self.gradient_postdivide_factor = (
            self.world_size / self.gradient_predivide_factor
        )

    @staticmethod
    def _get_gradient_predivide_factor(world_size: int) -> float:
        # 计算梯度预分割因子
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)


class LowPrecisionState(DefaultState):
    r"""
    存储在通信钩子中以较低精度执行梯度通信所需的状态。

    通信钩子将梯度转换回指定的原始参数精度（默认为 torch.float32）。
    基于 DefaultState 类构建。

    Args:
        parameter_type (torch.dtype): 模型参数的精度。
        用于将梯度转换回参数精度的钩子所需参数。
    """

    __slots__ = [
        "parameter_type",
    ]

    def __init__(
        self,
        process_group,
        parameter_type=torch.float32,
    ):
        super().__init__(process_group)
        # 设置参数精度
        self.parameter_type = parameter_type


def _decompress(state: LowPrecisionState, grad: torch.Tensor):
    """
    将梯度转换回完整参数精度，以便进一步计算在完整精度下进行。
    """
    # 保存原始梯度数据
    orig_grad_data = grad.data
    # 将梯度数据转换为 state.parameter_type 精度
    grad.data = grad.data.to(state.parameter_type)
    device_type = ""
    try:
        # 获取设备类型
        if grad.device.type == "privateuse1":
            device_type = torch._C._get_privateuse1_backend_name()
        else:
            device_type = grad.device.type
        # 获取与设备类型对应的后端
        backend = getattr(torch, device_type)
    except AttributeError as e:
        raise AttributeError(
            f"Device {grad.device}  does not have a \
                corresponding backend registered as 'torch.device_type'."
        ) from e

    # 不要让这段内存在传输之后被重复使用。
    # 使用 record_stream 方法记录当前的 GPU 流，但由于类型提示不明确，忽略类型检查
    orig_grad_data.record_stream(backend.current_stream())  # type: ignore[arg-type]
# 实现 FSDP 通信钩子的 all_reduce 算法及梯度的必要预分和后分。
def allreduce_hook(state: DefaultState, grad: torch.Tensor):
    """
    Implement the FSDP communication hook for the `all_reduce` algorithm and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): 状态信息，配置预分和后分因子。
        grad (torch.Tensor): 需要在各个进程之间传播的本地批次梯度。
    """
    # 根据预分因子平均梯度。预分和后分因子的结合，确保了最终按照 world_size 平均，与 PyTorch DDP 保持一致。
    # 这是一个两步过程，以避免潜在的下溢和上溢。
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    # 使用 all_reduce 方法在进程组内对梯度进行汇总
    dist.all_reduce(grad, group=state.process_group)
    # 根据后分因子再次平均梯度。
    if state.gradient_postdivide_factor > 1:
        grad.div_(state.gradient_postdivide_factor)


# 实现 FSDP 通信钩子的 reduce_scatter 算法。
def reduce_scatter_hook(state: DefaultState, grad: torch.Tensor, output: torch.Tensor):
    """
    Implement the FSDP communication hook for the `reduce_scatter` algorithm.

    For sharded FSDP strategies and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): 状态信息，配置预分和后分因子。
        grad (torch.Tensor): 需要在各个进程之间传播的本地批次未分片梯度。
        output (torch.Tensor): 在 `reduce_scatter` 后存储单个梯度分片的张量。
    """
    # 根据预分因子平均梯度。
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    # 使用 reduce_scatter_tensor 方法将梯度进行分散汇总
    dist.reduce_scatter_tensor(output, grad, group=state.process_group)
    # 根据后分因子再次平均梯度的分片。
    if state.gradient_postdivide_factor > 1:
        output.div_(state.gradient_postdivide_factor)


# 实现低精度钩子函数，用于在 FSDP 策略中进行梯度压缩和解压缩。
def _low_precision_hook(
    prec: torch.dtype,
    state: LowPrecisionState,
    grad: torch.Tensor,
    output: torch.Tensor,
):
    """
    Implement a low precision hook for gradient compression and decompression in FSDP strategies.

    Args:
        prec (torch.dtype): 指定的低精度数据类型，如 torch.float16。
        state (LowPrecisionState): 状态信息，用于配置梯度精度变换和通信。
        grad (torch.Tensor): 需要压缩或解压的梯度张量。
        output (torch.Tensor): 压缩后的梯度张量，或者为 None，表示直接操作 grad。
    """
    # 如果梯度数据类型不是指定的精度类型，则转换为指定的精度类型
    if grad.dtype != prec:
        grad.data = grad.data.to(prec)
    # 如果输出张量不为空
    if output is not None:
        # 如果输出张量数据类型不是指定的精度类型，则转换为指定的精度类型
        if output.dtype != prec:
            output.data = output.data.to(prec)
        # 使用 reduce_scatter_hook 函数对梯度进行 reduce_scatter 操作
        reduce_scatter_hook(state, grad, output)
        # 对输出张量进行解压缩操作
        _decompress(state, output)
    else:
        # 使用 allreduce_hook 函数对梯度进行 all_reduce 操作
        allreduce_hook(state, grad)
        # 对梯度进行解压缩操作
        _decompress(state, grad)


# 实现 fp16 压缩钩子函数，用于简单的梯度压缩方法。
def fp16_compress_hook(
    state: LowPrecisionState, grad: torch.Tensor, output: Optional[torch.Tensor] = None
):
    """
    Implement FSDP communication hook for a simple gradient compression approach.
    Casts `grad` to half-precision floating-point format (`torch.float16`).

    It also averages gradients by `world_size` in two steps: first it pre-divides gradients by a
    `state.gradient_predivide_factor`, and after a communication step (`all_reduce` or `reduce_scatter`)
    gradients are averaged by a `state.gradient_postdivide_factor`.

    Args:
        state (LowPrecisionState): 状态信息，用于配置梯度压缩和通信。
        grad (torch.Tensor): 需要压缩的梯度张量。
        output (Optional[torch.Tensor]): 可选，压缩后的梯度张量或 None。
    """
    # 将梯度转换为半精度浮点数格式（torch.float16）
    grad = grad.to(torch.float16)
    # 如果输出张量不为空，则将其转换为半精度浮点数格式（torch.float16）
    if output is not None:
        output.data = output.data.to(torch.float16)
    # 使用 _low_precision_hook 函数进行梯度压缩和解压缩操作
    _low_precision_hook(torch.float16, state, grad, output)
    """
    对传入的梯度进行压缩后，将其转换回参数的精度。

    Args:
        state (LowPrecisionState): 状态信息，配置了预和后除因子，以及参数的精度。
        grad (torch.Tensor): 本地批次的梯度，需要在低精度下在各个进程间通信。
        output (torch.Tensor): 在 ``reduce_scatter`` 后存储梯度的单个分片。
    """
    # 使用 functools.partial 创建一个部分应用的函数，使用 torch.float16 作为参数类型
    fp16_hook = functools.partial(_low_precision_hook, torch.float16)
    # 返回经过 fp16_hook 处理的结果
    return fp16_hook(state, grad, output)
def bf16_compress_hook(
    state: LowPrecisionState, grad: torch.Tensor, output: Optional[torch.Tensor] = None
):
    r"""
    Implement FSDP communication hook for a simple gradient compression approach .
    Casts ``grad`` to half-precision floating-point format.

    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)
    gradients are averaged by a ``state.gradient_postdivide_factor``.
    Once post-division is done, compressed gradients are casted back to parameters' precision.

    Args:
        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    # 创建一个部分函数，用于执行低精度钩子函数，指定使用 bfloat16 格式
    bf16_hook = functools.partial(_low_precision_hook, torch.bfloat16)
    # 调用低精度钩子函数，并传递状态信息、梯度和输出张量
    return bf16_hook(state, grad, output)
```