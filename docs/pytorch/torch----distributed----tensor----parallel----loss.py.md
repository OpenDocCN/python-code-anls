# `.\pytorch\torch\distributed\tensor\parallel\loss.py`

```py
# 设置 mypy 不强制要求定义类型
# 代码版权声明，属于 Meta Platforms, Inc. 及其关联公司
import contextlib  # 导入上下文管理模块
from typing import cast, Dict, Optional, Tuple  # 导入类型提示相关模块

import torch  # 导入 PyTorch 模块
import torch._prims_common as utils  # 导入 PyTorch 内部常用函数模块
import torch.distributed._functional_collectives as funcol  # 导入分布式函数集合模块
import torch.distributed.distributed_c10d as c10d  # 导入分布式模块
from torch import Tensor  # 导入 Tensor 类型
from torch.distributed._tensor import DTensor, Replicate, Shard  # 导入分布式 Tensor 相关类
from torch.distributed._tensor.ops.embedding_ops import _MaskPartial  # 导入嵌入操作相关类
from torch.distributed._tensor.ops.math_ops import (  # 导入数学运算操作相关函数
    _skip_dim,
    Reduction,
    replicate_reduction_dims,
)
from torch.distributed._tensor.placement_types import (  # 导入分布式 Tensor 放置类型相关类
    DTensorSpec,
    Placement,
    TensorMeta,
)
from torch.distributed.device_mesh import DeviceMesh  # 导入设备网格模块

aten = torch.ops.aten  # 设置 aten 为 torch 操作的 aten 方法

__all__ = ["loss_parallel"]  # 设置模块的公共接口

@contextlib.contextmanager
def loss_parallel():
    """
    上下文管理器，启用损失并行计算，用于在类维度分片输入时进行高效的损失计算。
    目前仅支持交叉熵损失函数。

    在该上下文管理器中，可以像平常一样使用 :func:`~torch.nn.functional.cross_entropy` 或
    :class:`~torch.nn.CrossEntropyLoss`，对输入参数有如下假设。
    如果有反向传播 `backward()` 调用，也需要在该上下文管理器中进行。

    Args:
        input (:class:`DTensor`):
            输入的 logits，假设在类维度上分片。
        target (Union[:class:`torch.Tensor`, :class:`DTensor`]):
            必须是真实类别索引（当前不支持类概率）。
            假设在 `DeviceMesh` 上复制。
        weight (Union[:class:`torch.Tensor`, :class:`DTensor`], optional):
            如果给定，假设在 `DeviceMesh` 上复制。
        label_smoothing:
            目前不支持。

    Returns:
        一个复制的 :class:`DTensor`。

    Example:
        这里手动创建了一个分片的 DTensor 来展示使用方法。
        在实际中，通常是 TP 模块的输出。

        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import loss_parallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> device_mesh = init_device_mesh("cuda", (8,))
        >>> input = torch.randn(4, 16, device="cuda", requires_grad=True)
        >>> dist_input = distribute_tensor(input, device_mesh, placements=[Shard(1)])
        >>> target = torch.randint(16, (4,), device="cuda")
        >>> with loss_parallel():
        >>>     loss = F.cross_entropy(dist_input, target, reduction="mean")
        >>>     loss.backward()
        >>> ...
    """
    _enable_custom_loss_ops()  # 启用自定义损失操作

    yield  # 执行上下文管理器内部代码块

    _disable_custom_loss_ops()  # 禁用自定义损失操作

# 当前仅需支持一维 DeviceMesh；通常返回 mesh_dim 与 placements[mesh_dim].is_shard(dim)。
# 确定给定的放置信息列表是否只有一个元素，如果不是则引发 ValueError 异常
def _find_all_reduce_mesh_dim(placements: Tuple[Placement, ...], dim: int) -> int:
    if not len(placements) == 1:
        raise ValueError(
            "Currently loss_parallel() only supports input on one-dimensional DeviceMesh."
        )
    # 检查第一个放置信息是否与指定维度是分片的，如果不是则引发 ValueError 异常
    if not placements[0].is_shard(dim):
        raise ValueError(
            f"loss_parallel() should be enabled only when the input tensor is sharded on dimension {dim}."
        )
    # 返回零，表示没有找到其他维度需要处理
    return 0


# 将输入张量转换为 DTensor 类型，根据给定的放置信息和设备网格
def _cast_to_dtensor(
    tensor, placements: Tuple[Placement, ...], mesh: DeviceMesh
) -> DTensor:
    # 如果输入已经是 DTensor 类型且放置信息与指定的相同，则直接返回
    if isinstance(tensor, DTensor):
        if tensor.placements == placements:
            return tensor
        else:
            raise RuntimeError(f"Expected {placements} but got {tensor.placements}.")
    # 如果输入是 torch.Tensor 类型，则调用 DTensor 的 from_local 方法进行转换
    elif isinstance(tensor, torch.Tensor):
        return DTensor.from_local(
            tensor, device_mesh=mesh, placements=placements, run_check=False
        )
    # 如果输入类型不支持，则引发 TypeError 异常
    else:
        raise TypeError(f"Unsupported type {type(tensor)}")


# 从操作调用信息中获取张量元数据，返回 TensorMeta 对象
def _propagate_tensor_meta(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> TensorMeta:
    # 解析操作调用信息，获取操作信息对象
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    # 使用操作信息对象调用 sharding_propagator 的 _propagate_tensor_meta 方法获取张量元数据
    tensor_meta = DTensor._op_dispatcher.sharding_propagator._propagate_tensor_meta(
        op_info.schema
    )
    # 如果返回的是 TensorMeta 对象，则直接返回
    if isinstance(tensor_meta, TensorMeta):
        return tensor_meta
    # 如果返回的是元组，则返回元组的第一个元素，通常用于多个返回值的情况
    elif isinstance(tensor_meta, tuple):
        return tensor_meta[0]
    # 如果返回的类型不被预期，则引发 RuntimeError 异常
    else:
        raise RuntimeError(f"Unexpected tensor meta type: {type(tensor_meta)}.")


# 实现了对输入张量进行 log_softmax 操作的函数，使用 all_reduce 进行分布式计算
def _log_softmax(x, dim, half_to_float, mesh, mesh_dim):
    # 确保输入张量是连续的
    x = x.contiguous()
    # 如果需要将半精度转换为单精度，则检查输入张量的数据类型
    if half_to_float:
        assert x.dtype == torch.half
    # 确定用于计算的数据类型和结果的数据类型
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    # 将输入张量转换为指定的计算数据类型
    x = x.to(computation_dtype)
    # 如果张量元素数量为零，则直接使用原始张量进行计算
    if x.numel() == 0:
        shifted = x
    else:
        # 计算输入张量沿指定维度的最大值，并通过 all_reduce 在分布式环境中进行全局最大值计算
        x_max = torch.amax(x, dim, keepdim=True)
        x_max = funcol.all_reduce(
            x_max, reduceOp=c10d.ReduceOp.MAX.name, group=(mesh, mesh_dim)
        )
        shifted = x - x_max
    # 计算移位后的指数和，通过 all_reduce 在分布式环境中进行全局求和计算
    shifted_sumexp = torch.sum(torch.exp(shifted), dim, keepdim=True)
    shifted_sumexp = funcol.all_reduce(
        shifted_sumexp, reduceOp=c10d.ReduceOp.SUM.name, group=(mesh, mesh_dim)
    )
    # 计算移位后的对数和，用于计算最终的 log_softmax 结果
    shifted_logsumexp = torch.log(shifted_sumexp)
    # 计算最终的 log_softmax 结果
    result = shifted - shifted_logsumexp
    # 如果不需要将结果转换回原始数据类型，则将结果转换为指定的结果数据类型
    if not half_to_float:
        result = result.to(result_dtype)
    # 返回计算结果
    return result


# 对 log_softmax 操作的输入参数进行处理的函数，返回处理后的结果对象
def _log_softmax_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # 将第一个参数转换为 DTensor 类型
    x = cast(DTensor, args[0])
    # 获取 log_softmax 操作的维度参数
    dim = cast(int, args[1])
    # 获取是否将半精度转换为单精度的标志位
    half_to_float = cast(bool, args[2])

    # 从 DTensor 对象的规格信息中获取网格维度信息
    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, dim)
    # 调用函数 _propagate_tensor_meta，传入参数 op_call, args, kwargs，获取输出张量的元数据
    output_tensor_meta = _propagate_tensor_meta(op_call, args, kwargs)
    
    # 调用函数 _log_softmax，计算输入张量 x._local_tensor 的对数softmax值，并返回结果
    res = _log_softmax(x._local_tensor, dim, half_to_float, spec.mesh, mesh_dim)
    
    # 创建 DTensorSpec 对象，指定了张量的规格，包括网格信息、放置信息以及输出张量的元数据
    res_spec = DTensorSpec(
        spec.mesh,
        spec.placements,
        tensor_meta=output_tensor_meta,
    )
    
    # 创建 DTensor 对象，将计算得到的 res 张量、res_spec 规格以及 requires_grad 属性一起返回
    return DTensor(
        res,
        res_spec,
        requires_grad=res.requires_grad,
    )
# NOTE: 在 _nll_loss_and_log_softmax_backward 的解释中，_log_softmax_backward_handler 实际上不执行任何计算。
def _log_softmax_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # 从参数元组中获取梯度输出
    grad_output = cast(DTensor, args[0])
    # 从参数元组中获取输入的数据类型
    input_dtype = cast(torch.dtype, args[3])
    # 将梯度输出转换为指定的输入数据类型并返回
    return grad_output.to(input_dtype)


# NOTE: 实现遵循 torch._decomp.decomposition._nll_loss_forward，但插入了自定义的通信以执行分布式计算。
def _nll_loss_forward(
    x: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    local_weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    channel_dim_size: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> Tuple[Tensor, Tensor]:
    # 确定输入张量的维度数
    n_dims = x.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    # 定义用于处理权重的函数
    def _weight_view(weight: Tensor) -> Tensor:
        if n_dims > 1:
            shape = [
                1,
            ] * n_dims
            shape[channel_dim] = weight.shape[0]
            w = weight.view(shape)
        else:
            w = weight
        return w

    # 如果存在权重，则按本地权重对输入进行调整
    if weight is not None:
        w = _weight_view(weight)
        assert local_weight is not None
        local_w = _weight_view(local_weight)
        x = x * local_w
    
    # 在安全目标中将目标张量中等于忽略索引的位置替换为0
    safe_target = torch.where(target != ignore_index, target, 0)
    safe_target_ = safe_target.unsqueeze(channel_dim)

    # 下面的代码块是对以下代码的分布式版本：
    # result = -torch.gather(self, channel_dim, safe_target_).squeeze(channel_dim)
    partial_placement = _MaskPartial(logical_dim_size=channel_dim_size)
    safe_target_partial_ = partial_placement._partition_value(
        safe_target_, mesh, mesh_dim
    )
    result_partial = torch.gather(x, channel_dim, safe_target_partial_)
    # 执行全局归约操作
    result_reduced = partial_placement._reduce_value(result_partial, mesh, mesh_dim)
    result = -result_reduced.squeeze(channel_dim)

    # 将目标张量中等于忽略索引的位置的结果设置为0
    result = torch.where(target != ignore_index, result, 0)

    # 如果不进行缩减操作且输入张量的维度数大于1，则计算总权重
    if reduction == Reduction.NONE.value and n_dims > 1:
        total_weight = x.new_full((), 0.0)
        return result, total_weight

    # 如果存在权重，则根据通道维度扩展权重张量并计算总权重
    if weight is not None:
        new_shape = list(x.shape)
        new_shape[channel_dim] = -1
        w = w.expand(new_shape)
        wsum = torch.gather(w, channel_dim, safe_target_).squeeze(channel_dim)
        wsum = torch.where(target != ignore_index, wsum, 0)
        total_weight = wsum.sum()
    else:
        total_weight = (target != ignore_index).sum().to(x)

    # 如果缩减方式为 SUM，则将结果进行求和
    if reduction == Reduction.SUM.value:
        result = result.sum()
    # 如果缩减方式为 MEAN，则将结果求和后除以总权重得到均值
    elif reduction == Reduction.MEAN.value:
        result = result.sum() / total_weight

    # 返回最终的结果和总权重
    return result, total_weight


def _nll_loss_forward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],


    # 定义三个变量，分别用于表示操作调用、位置参数和关键字参数
    op_call: torch._ops.OpOverload,  # op_call 是一个类型为 torch._ops.OpOverload 的变量
    args: Tuple[object, ...],         # args 是一个包含任意数量对象的元组，类型为 Tuple[object, ...]
    kwargs: Dict[str, object],        # kwargs 是一个包含字符串键和任意对象值的字典，类型为 Dict[str, object]
# 定义函数_nll_loss_and_log_softmax_backward，用于计算交叉熵损失函数和对数softmax的反向传播
def _nll_loss_and_log_softmax_backward(
    grad_output: Tensor,  # 梯度输出，即上游梯度
    x: Tensor,            # 输入张量 x
    target: Tensor,       # 目标张量 target
    weight: Optional[Tensor],  # 权重张量，可选
    reduction: int,       # 减少方式，表示如何减少损失
    ignore_index: int,    # 忽略的索引，用于指定忽略特定类别的计算
) -> object:              # 返回类型为对象

    x = cast(DTensor, args[0])  # 将 args[0] 转换为 DTensor 类型，赋值给 x
    target = args[1]            # args[1] 赋值给 target
    weight = args[2]            # args[2] 赋值给 weight
    reduction = cast(int, args[3])  # 将 args[3] 转换为 int 类型，赋值给 reduction
    ignore_index = cast(int, args[4])  # 将 args[4] 转换为 int 类型，赋值给 ignore_index

    channel_dim = 1 if x.dim() >= 2 else 0  # 如果 x 的维度大于等于 2，则 channel_dim 等于 1，否则等于 0
    channel_dim_size = x.shape[channel_dim]  # 获取 x 在 channel_dim 维度上的大小
    spec = x._spec  # 获取 x 的规范化描述
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)  # 查找所有减少维度的网格维度

    # 检查用户输入：如果 target 和 weight 不是 DTensors，则将它们转换为 DTensors；
    # 如果它们是 DTensors，则检查它们是否具有所需的放置位置。
    target_placements = _skip_dim(
        replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
    )  # 根据 channel_dim 复制并减少维度，获取目标张量的放置位置
    all_replicate_placements = (Replicate(),) * spec.mesh.ndim  # 所有维度都复制的放置位置
    target = _cast_to_dtensor(target, target_placements, spec.mesh)  # 将目标张量转换为 DTensor
    local_weight = None  # 初始化本地权重为 None
    if weight is not None:  # 如果 weight 不为 None
        weight = _cast_to_dtensor(weight, all_replicate_placements, spec.mesh)  # 将权重张量转换为 DTensor
        # 对于本地计算，需要 _nll_loss_forward() 中的复制权重和分片本地权重两者。
        # 这里使用 DTensor API 生成 local_weight，而不会产生任何通信开销。
        sharded_placements = [
            Shard(0) if i == mesh_dim else Replicate() for i in range(spec.mesh.ndim)
        ]  # 根据 mesh_dim 创建分片放置位置
        local_weight = weight.redistribute(spec.mesh, sharded_placements)._local_tensor  # 重新分发权重并获取本地张量
        assert local_weight.shape[0] == x._local_tensor.shape[channel_dim]  # 断言本地权重的形状与 x 的 channel_dim 维度相符

    if reduction == Reduction.NONE.value:  # 如果减少方式为无
        output_placements = target_placements  # 输出放置位置为目标放置位置
    else:
        output_placements = all_replicate_placements  # 否则，输出放置位置为所有维度复制的放置位置

    # _propagate_tensor_meta 需要输入的张量为 DTensors
    args = list(args)  # 将参数 args 转换为列表
    args[1], args[2] = target, weight  # 将目标和权重赋值给 args 的第二和第三个位置
    output_tensor_meta = _propagate_tensor_meta(op_call, tuple(args), kwargs)  # 传播张量元信息

    # 调用 _nll_loss_forward 计算损失
    result, total_weight = _nll_loss_forward(
        x._local_tensor,  # x 的本地张量
        target._local_tensor,  # 目标的本地张量
        weight._local_tensor if weight is not None else None,  # 如果 weight 不为 None 则是权重的本地张量，否则为 None
        local_weight,  # 本地权重
        reduction,  # 减少方式
        ignore_index,  # 忽略的索引
        channel_dim_size,  # channel_dim 的大小
        spec.mesh,  # 网格
        mesh_dim,  # 网格维度
    )

    out_spec = DTensorSpec(spec.mesh, output_placements, tensor_meta=output_tensor_meta)  # 输出规范化描述

    # 返回结果和总权重
    return (
        DTensor(
            result,  # 结果
            out_spec,  # 输出规范化描述
            requires_grad=result.requires_grad,  # 是否需要梯度
        ),
        total_weight,  # 总权重
    )


# 注意：cross_entropy 的反向计算分为两个步骤：nll_loss 的反向和 log_softmax 的反向。
# 在损失并行计算中，这两个步骤融合为下面的函数（由 _nll_loss_backward_handler 调用），
# 以避免当目标包含类索引而非类概率时进行通信。
# 还要注意 _log_softmax_backward_handler 不执行计算。
# 该实现类似于 torch._decomp.decomposition 中的 _nll_loss_backward 和 _log_softmax_backward_data。
    ignore_index: int,  # 忽略的索引，应该是一个整数
    total_weight: Tensor,  # 总权重，类型为 Tensor（张量）
    channel_dim_size: int,  # 通道维度的大小，应该是一个整数
    mesh: DeviceMesh,  # 设备网格对象，可能是一个自定义的类或类型
    mesh_dim: int,  # 网格的维度，应该是一个整数
# 定义函数_nll_loss_backward_handler，用于处理负对数似然损失函数的反向传播
def _nll_loss_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # 从参数元组中获取对应的参数
    grad_output = cast(DTensor, args[0])  # 获取梯度输出
    x = cast(DTensor, args[1])  # 获取输入张量 x
    target = args[2]  # 获取目标张量 target
    weight = args[3]  # 获取权重张量 weight
    reduction = cast(int, args[4])  # 获取减少方式 reduction
    ignore_index = cast(int, args[5])  # 获取忽略索引 ignore_index
    total_weight = cast(Tensor, args[6])  # 获取总权重 total_weight
    
    # 确定通道维度的位置
    channel_dim = 1 if x.dim() >= 2 else 0
    
    # 根据指定的减少方式进行梯度输出的调整
    if reduction == Reduction.MEAN.value:
        grad_output = grad_output / total_weight
    
    # 将目标张量增加一个维度，以便与输入张量 x 的通道维度对齐
    target = target.unsqueeze(channel_dim)
    
    # 将目标张量中的忽略索引位置替换为 0，形成安全目标张量
    safe_target = torch.where(target != ignore_index, target, 0)
    
    # 创建与输入张量 x 相同形状的零张量，用于存储梯度输入
    grad_input = torch.zeros_like(x)
    
    # 下面的代码块是分布式版本的 torch.scatter 函数
    # grad_input = torch.scatter(grad_input, channel_dim, safe_target, -1.0)
    
    # 创建部分布局对象，用于处理逻辑维度大小为通道维度大小
    partial_placement = _MaskPartial(logical_dim_size=channel_dim_size)
    
    # 将安全目标张量展平并传递给部分布局对象的分区值方法
    safe_target = safe_target.squeeze(channel_dim).flatten()
    masked_safe_target = partial_placement._partition_value(safe_target, mesh, mesh_dim)
    
    # 断言部分布局对象的掩码缓冲数据不为空
    assert partial_placement.mask_buffer.data is not None
    
    # 计算梯度更新值
    grad_update = partial_placement.mask_buffer.data.float() - 1.0
    
    # 创建一维索引张量
    arange_1d = torch.arange(
        masked_safe_target.shape[0], device=masked_safe_target.device
    )
    
    # 根据输入张量 x 的维度情况，更新梯度输入张量 grad_input
    if x.dim() == 1:
        grad_input[masked_safe_target] = grad_update
    elif x.dim() == 2:
        grad_input[arange_1d, masked_safe_target] = grad_update
    else:
        grad_input_t = grad_input.transpose(channel_dim, -1)
        intermidate_shape = grad_input_t.shape
        grad_input_2d = grad_input_t.reshape(-1, x.shape[channel_dim])
        grad_input_2d[arange_1d, masked_safe_target] = grad_update
        grad_input = grad_input_2d.view(intermidate_shape).transpose(channel_dim, -1)
    
    # 如果梯度输入的维度大于梯度输出的维度且梯度输出的维度大于 0，则扩展梯度输出的维度
    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)
    
    # 如果权重张量不为空，则根据指定规则调整梯度输出
    if weight is not None:
        new_shape = [1 for _ in range(x.dim())]
        new_shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(new_shape)
        
        # 使用广播机制将权重应用到梯度输出上
        new_shape = list(x.shape)
        new_shape[channel_dim] = -1
        w = weight.expand(new_shape)
        w_target = torch.gather(w, channel_dim, target)
        grad_output = grad_output * w_target
    
    # 将梯度输出中目标索引位置不等于忽略索引的元素置为 0
    grad_output = torch.where(target != ignore_index, grad_output, 0)
    
    # 注意：为了避免额外的全局收集通信，在这里一起执行对 log_softmax 的反向传播计算
    # 返回最终的梯度输入 grad_input 乘以梯度输出 grad_output
    return (grad_input + torch.exp(x)) * grad_output
    # 获取张量 x 在指定通道维度上的大小
    channel_dim_size = x.shape[channel_dim]
    # 获取张量 x 的规格信息
    spec = x._spec
    # 查找所有减少操作的网格维度
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)
    
    # 如果目标和权重不是 DTensors，则将它们转换为 DTensors
    # 获取目标的放置信息，排除指定的通道维度
    target_placements = _skip_dim(
        replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
    )
    # 创建一个所有维度都复制的放置信息元组
    all_replicate_placements = (Replicate(),) * spec.mesh.ndim
    # 将目标转换为 DTensor
    target = _cast_to_dtensor(target, target_placements, spec.mesh)
    # 如果权重不为 None，则将权重也转换为 DTensor
    if weight is not None:
        weight = _cast_to_dtensor(weight, all_replicate_placements, spec.mesh)
    
    # 将输入张量转换为 DTensors 以便传递给 _propagate_tensor_meta 函数
    args = list(args)
    args[2], args[3] = target, weight
    args[6] = _cast_to_dtensor(total_weight, all_replicate_placements, spec.mesh)
    # 使用 op_call 和参数元组以及 kwargs 传播张量元数据
    output_tensor_meta = _propagate_tensor_meta(op_call, tuple(args), kwargs)
    
    # 调用 _nll_loss_and_log_softmax_backward 函数计算反向传播的结果
    result = _nll_loss_and_log_softmax_backward(
        grad_output._local_tensor,
        x._local_tensor,
        target._local_tensor,
        weight._local_tensor if weight is not None else None,
        reduction,
        ignore_index,
        total_weight,
        channel_dim_size,
        spec.mesh,
        mesh_dim,
    )
    
    # 输出规格与输入规格相同：在 mesh_dim 上按照 Shard(channel_dim) 进行分片
    out_spec = DTensorSpec(
        spec.mesh,
        spec.placements,
        tensor_meta=output_tensor_meta,
    )
    
    # 返回一个带有指定规格的 DTensor 对象
    return DTensor(
        result,
        out_spec,
        requires_grad=result.requires_grad,
    )
# 定义一个字典，将特定的 ATen 操作与对应的处理函数关联起来
customized_loss_ops = {
    aten._log_softmax.default: _log_softmax_handler,  # 将 _log_softmax 操作映射到 _log_softmax_handler 处理函数
    aten._log_softmax_backward_data.default: _log_softmax_backward_handler,  # 将 _log_softmax_backward_data 操作映射到 _log_softmax_backward_handler 处理函数
    aten.nll_loss_forward.default: _nll_loss_forward_handler,  # 将 nll_loss_forward 操作映射到 _nll_loss_forward_handler 处理函数
    aten.nll_loss2d_forward.default: _nll_loss_forward_handler,  # 将 nll_loss2d_forward 操作映射到 _nll_loss_forward_handler 处理函数
    aten.nll_loss_backward.default: _nll_loss_backward_handler,  # 将 nll_loss_backward 操作映射到 _nll_loss_backward_handler 处理函数
    aten.nll_loss2d_backward.default: _nll_loss_backward_handler,  # 将 nll_loss2d_backward 操作映射到 _nll_loss_backward_handler 处理函数
}

# 启用自定义损失函数操作的函数
def _enable_custom_loss_ops():
    # 更新 DTensor 类的操作调度器的自定义操作处理函数
    DTensor._op_dispatcher._custom_op_handlers.update(customized_loss_ops)

# 禁用自定义损失函数操作的函数
def _disable_custom_loss_ops():
    # 遍历所有自定义操作，从 DTensor 类的操作调度器的自定义操作处理函数中移除
    for custom_op in customized_loss_ops:
        DTensor._op_dispatcher._custom_op_handlers.pop(custom_op)
```