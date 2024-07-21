# `.\pytorch\torch\distributed\_spmd\experimental_ops.py`

```py
# 引入必要的模块和类型定义
from typing import cast, List, Optional, Sequence, Tuple

import torch
from torch.distributed._tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)

# 获取 torch.ops.aten 的别名，忽略 pyre 的类型检查
aten = torch.ops.aten  # pyre-ignore

# 注册属性规则，处理单目运算的情况，如负数、倒数、平方根等
@register_prop_rule(  # pyre-ignore
    [
        aten._foreach_neg.default,
        aten._foreach_reciprocal.default,
        aten._foreach_sqrt.default,
    ]
)
def _prop__foreach_unaop(op_schema: OpSchema) -> OutputSharding:
    # 获取第一个参数作为 self
    self = op_schema.args_schema[0]
    # 断言 self 是一个 DTensorSpec 的列表
    assert isinstance(self, list) and all(isinstance(s, DTensorSpec) for s in self)
    # 对于 sqrt，仅在 Replicate 和 Shard 张量上数学上正确
    # FIXME(@mrshenli): for sqrt, this is only mathematically correct for
    # Replicate and Shard tensor.
    return OutputSharding(output_spec=self)

# 注册属性规则，处理二元运算的情况，如加法、除法、乘法等（操作数为列表）
@register_prop_rule(  # pyre-ignore
    [
        aten._foreach_add.List,
        aten._foreach_div.List,
        aten._foreach_mul.List,
    ]
)
def _prop__foreach_binop_list(op_schema: OpSchema) -> OutputSharding:
    # 获取前两个参数作为 self 和 other
    self, other = op_schema.args_schema[:2]
    # 如果有第三个参数，将其视为 scalar
    scalar = None if len(op_schema.args_schema) < 3 else op_schema.args_schema[2]
    # 断言 self 和 other 都是 DTensorSpec 的列表
    assert isinstance(self, list) and all(
        isinstance(s, DTensorSpec) for s in self
    ), f"Expect a List[DTensorSpec] but got {self}"
    assert isinstance(other, list) and all(
        isinstance(o, DTensorSpec) for o in other
    ), f"Expect a List[DTensorSpec] but got {other}"
    # 断言 self 和 other 的长度必须相同
    assert len(self) == len(other), (
        "Two tensor lists must match in length, "
        f"but got {len(self)} and {len(other)}"
    )

    # 如果两个操作数的 DTensorSpec 不匹配，建议使用 self 的 DTensorSpec
    # 这会触发 allreduce，如果 other 是 partial 且 self 是 replicated
    if any(s != o for s, o in zip(self, other)):
        return OutputSharding(
            output_spec=None,
            redistribute_schema=OpSchema(
                op=op_schema.op,
                args_schema=(self, self, scalar) if scalar else (self, self),
                kwargs_schema=op_schema.kwargs_schema,
            ),
        )
    else:
        return OutputSharding(output_spec=self)

# 注册属性规则，处理二元运算的情况，如加法、除法、乘法等（其中一个操作数为标量）
@register_prop_rule(  # pyre-ignore
    [
        aten._foreach_add.Scalar,
        aten._foreach_div.Scalar,
        aten._foreach_mul.Scalar,
        aten._foreach_sub.Scalar,
    ]
)
def _prop__foreach_binop_scalar(op_schema: OpSchema) -> OutputSharding:
    # 获取前两个参数作为 self 和 scalar
    self, scalar = op_schema.args_schema
    # 断言 self 是 DTensorSpec 的列表，scalar 不是列表
    assert isinstance(self, list) and all(isinstance(s, DTensorSpec) for s in self)
    assert not isinstance(scalar, list)
    return OutputSharding(output_spec=self)
    [
        # 引用 PyTorch 中的 aten 模块，用于张量操作
        aten._foreach_addcdiv.Scalar,
        # 引用 PyTorch 中的 aten 模块，用于张量操作
        aten._foreach_addcmul.Scalar,
    ]
@register_prop_rule([aten._foreach_pow.ScalarAndTensor])  # 注册一个属性规则，应用于 aten._foreach_pow.ScalarAndTensor
def _prop__foreach_pow_scalar_and_tensor(op_schema: OpSchema):
    # 定义函数 _prop__foreach_pow_scalar_and_tensor，接受 OpSchema 类型的参数 op_schema
    scala, exponent = op_schema.args_schema
    # 从 op_schema 的参数模式中获取 scala 和 exponent

    assert isinstance(exponent, list) and all(
        isinstance(s, DTensorSpec) for s in exponent
    )
    # 断言 exponent 是一个列表，并且列表中的每个元素都是 DTensorSpec 类型

    return OutputSharding(output_spec=exponent)
    # 返回一个 OutputSharding 对象，其 output_spec 属性为 exponent


@register_prop_rule([aten._fused_adam.default])  # pyre-ignore
def _prop__fused_adam(op_schema: OpSchema):
    # 定义函数 _prop__fused_adam，接受 OpSchema 类型的参数 op_schema
    NT = 5
    # 设置常量 NT 为 5
    tesnor_list_args: Tuple[List[DTensorSpec]] = op_schema.args_schema[:NT]  # type: ignore[assignment]
    # 从 op_schema 的参数模式中获取前 NT 个参数，作为一个元组，每个参数都是 DTensorSpec 类型的列表

    assert all(isinstance(schema, list) for schema in tesnor_list_args)
    # 断言 tesnor_list_args 中的每个元素都是列表类型

    assert all(
        isinstance(s, DTensorSpec) for schema in tesnor_list_args for s in schema
    )
    # 断言 tesnor_list_args 中的每个 DTensorSpec 元素都是 DTensorSpec 类型

    tensor_schemas: Tuple[List[DTensorSpec]] = [  # type: ignore[assignment]
        schema for schema in tesnor_list_args if len(schema)
    ]
    # 创建一个元组 tensor_schemas，包含 tesnor_list_args 中所有非空列表的 DTensorSpec 元素

    assert all(len(s) == len(tensor_schemas[0]) for s in tensor_schemas), (
        "expect the same number of gradients and states, but got "
        f"{[len(s) for s in tensor_schemas]}."
    )
    # 断言 tensor_schemas 中的每个列表长度都与第一个列表相同，并输出错误信息如果不相同的话

    if any(any(t != ts[0] for t in ts) for ts in zip(*tensor_schemas)):
        # 如果存在任何一个梯度或状态与第一个列表中的不同
        new_schemas: Tuple[List[DTensorSpec]] = tuple(  # type: ignore[assignment]
            op_schema.args_schema[0] if len(s) else s for s in tesnor_list_args
        )
        # 创建一个新的元组 new_schemas，根据每个列表的长度决定是否从 op_schema 中取值

        return OutputSharding(
            output_spec=None,
            redistribute_schema=OpSchema(
                op=op_schema.op,
                args_schema=new_schemas + op_schema.args_schema[NT:],
                kwargs_schema=op_schema.kwargs_schema,
            ),
        )
        # 返回一个 OutputSharding 对象，其 redistribute_schema 属性为新的 OpSchema 对象
    else:
        return OutputSharding(output_spec=(op_schema.args_schema[0],) * NT)  # type: ignore[arg-type]
        # 返回一个 OutputSharding 对象，其 output_spec 属性为一个元组，包含 op_schema.args_schema[0] 重复 NT 次


@register_prop_rule(aten.nll_loss_forward.default)  # pyre-ignore
def _prop_nll_loss_forward(op_schema: OpSchema) -> OutputSharding:
    # 定义函数 _prop_nll_loss_forward，接受 OpSchema 类型的参数 op_schema，并返回 OutputSharding 类型的对象
    # 从操作模式的参数模式中获取 self 和 target
    self, target = op_schema.args_schema[:2]
    # 确保 self 是 DTensorSpec 类型
    assert isinstance(self, DTensorSpec)
    # 确保 target 是 DTensorSpec 类型
    assert isinstance(target, DTensorSpec)
    # 如果 self 和 target 的放置方式不同
    if self.placements != target.placements:
        # 自身和目标必须在放置方式上匹配，在数据并行使用情况下，应沿批处理维度分片。强制重新分发。
        
        # 需要创建一个新的 self，而不是返回 (target, target)，因为 target 和 self 可能在形状上不匹配。
        new_self = DTensorSpec(
            mesh=self.mesh,
            placements=target.placements,
            tensor_meta=self.tensor_meta,
        )
        # 返回一个 OutputSharding 对象，指定了输出规范为 None，重新分发模式为 OpSchema 对象
        return OutputSharding(
            output_spec=None,
            redistribute_schema=OpSchema(
                op=op_schema.op,
                args_schema=(new_self, target) + op_schema.args_schema[2:],
                kwargs_schema=op_schema.kwargs_schema,
            ),
        )
    else:
        # 如果 self 和 target 的放置方式相同
        
        # 返回一个 OutputSharding 对象，指定了输出规范为一个包含两个 DTensorSpec 的元组
        return OutputSharding(
            output_spec=(
                # 默认情况下，nll_loss_forward 进行缩减并返回一个标量张量，因此使用 _Partial 放置方式。
                DTensorSpec(mesh=self.mesh, placements=(_Partial(),)),
                # 第二个输出 total_weight 总是一个标量张量
                DTensorSpec(mesh=self.mesh, placements=(Replicate(),)),
            )
        )
# 注册属性规则到 nll_loss_backward 默认函数，忽略 Pyre 类型检查
@register_prop_rule(aten.nll_loss_backward.default)  # pyre-ignore
def _prop_nll_loss_backward(op_schema: OpSchema) -> OutputSharding:
    # 解构操作模式的参数模式，获取梯度输出和 self（应为 DTensorSpec 类型）
    grad_output, self = op_schema.args_schema[:2]
    assert isinstance(grad_output, DTensorSpec)
    assert isinstance(self, DTensorSpec)
    # 返回输出分片，使用 self 的规范作为输出规范
    return OutputSharding(output_spec=self)


# 注册属性规则到 stack 默认函数
@register_prop_rule(aten.stack.default)
def _prop_stack(op_schema: OpSchema) -> OutputSharding:
    # 解构操作模式的参数模式，获取 tensors 和 dim
    tensors = op_schema.args_schema[0]
    dim = 0 if len(op_schema.args_schema) == 1 else cast(int, op_schema.args_schema[1])
    assert (
        isinstance(tensors, list) and len(tensors) > 0
    ), "expect at least one tensor to stack"
    assert all(
        isinstance(t, DTensorSpec) for t in tensors
    ), f"expect a list of DTensorSpecs, but got {tensors}"
    assert all(
        t.shape == tensors[0].shape for t in tensors
    ), f"expect all tensors to have the same shape, but got {tensors}."
    # TODO: 在位置不匹配时提供 redistribute_schema
    assert all(
        t.placements == tensors[0].placements for t in tensors
    ), f"expect all tensors to have the same placements, but got {tensors}."
    assert all(
        not p.is_shard(dim) for p in tensors[0].placements
    ), "DTensor does not support stack on sharded dimension."
    
    # 返回输出分片，使用 tensors[0] 的 mesh 和重新排列后的 placements 作为输出规范
    return OutputSharding(
        output_spec=DTensorSpec(mesh=tensors[0].mesh, placements=tensors[0].placements)
    )


# 注册属性规则到 select.int 函数
@register_prop_rule(aten.select.int)
def _prop_select(op_schema: OpSchema) -> OutputSharding:
    # 解构操作模式的参数模式，获取 tensor 和 dim
    tensor, dim = op_schema.args_schema[:2]
    assert isinstance(tensor, DTensorSpec)
    assert isinstance(dim, int)
    placements: Sequence[Placement] = tensor.placements
    assert all(
        not p.is_shard(dim) for p in placements
    ), "DTensor does not support select on sharded dimension."

    # select 操作将移除一个维度，如果 Shard placements 的维度大于 dim，则将其减少 1
    new_placements: List[Placement] = []
    for p in placements:
        # 使用 isinstance 而不是 is_shard，以避免 mypy 报告关于访问 dim 属性的错误
        if isinstance(p, Shard) and p.dim > dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)

    # 返回输出分片，使用 tensor 的 mesh 和重新排列后的 new_placements 作为输出规范
    return OutputSharding(
        output_spec=DTensorSpec(mesh=tensor.mesh, placements=tuple(new_placements))
    )


# 注册属性规则到 native_layer_norm 默认函数，忽略 Pyre 类型检查
@register_prop_rule(aten.native_layer_norm.default)  # pyre-ignore
def _prop_native_layer_norm(op_schema: OpSchema) -> OutputSharding:
    # 解构操作模式的参数模式，获取 input, normalized_shape, weight, bias, eps
    input, normalized_shape, weight, bias, eps = op_schema.args_schema
    assert isinstance(input, DTensorSpec)
    assert isinstance(normalized_shape, (tuple, list))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all(isinstance(p, Replicate) for p in weight.placements)
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all(isinstance(p, Replicate) for p in bias.placements)
    
    # 返回输出分片，使用 input 的 mesh 和 placements 作为输出规范
    return OutputSharding(
        output_spec=DTensorSpec(mesh=input.mesh, placements=input.placements)
    )
    # 计算输入张量的批处理维度数
    batch_ndim = len(input.shape) - len(normalized_shape)
    
    # 断言：确保所有的输入放置（placements）都是复制（Replicate）或者分片（Shard）且分片维度小于批处理维度
    assert all(
        isinstance(p, Replicate) or (isinstance(p, Shard) and p.dim < batch_ndim,)
        for p in input.placements
    )
    
    # 创建统计规格对象（DTensorSpec），使用输入的网格（mesh）和放置（placements）
    stats_spec = DTensorSpec(
        mesh=input.mesh,
        placements=input.placements,
    )
    
    # 返回输出分片对象（OutputSharding），包含输入规格、统计规格和另一个统计规格
    return OutputSharding(output_spec=(input, stats_spec, stats_spec))
@register_prop_rule(aten.native_layer_norm_backward.default)  # pyre-ignore
def _prop_native_layer_norm_backward(op_schema: OpSchema) -> OutputSharding:
    (
        grad,
        input,
        normalized_shape,
        result1,
        result2,
        weight,
        bias,
        grad_input_mask,
    ) = op_schema.args_schema
    # 确保 grad 是 DTensorSpec 类型
    assert isinstance(grad, DTensorSpec)
    # 确保 grad_input_mask 是 list 或 tuple 类型
    assert isinstance(grad_input_mask, (list, tuple))
    if weight is not None:
        # 如果 weight 不为空，确保 weight 是 DTensorSpec 类型，并且所有的 placements 都是 Replicate 类型
        assert isinstance(weight, DTensorSpec)
        assert all(isinstance(s, Replicate) for s in weight.placements)
    if bias is not None:
        # 如果 bias 不为空，确保 bias 是 DTensorSpec 类型，并且所有的 placements 都是 Replicate 类型
        assert isinstance(bias, DTensorSpec)
        assert all(isinstance(s, Replicate) for s in bias.placements)
    # 确保 grad 中至少有一个 Shard 类型的 placement，且其 dim 属性为 0
    assert any(
        isinstance(s, Shard) and s.dim == 0 for s in grad.placements
    ), f"Got {grad.placements}"
    # 根据 weight 和 bias 的情况，创建对应的 weight_grad 和 bias_grad
    weight_grad = (
        DTensorSpec(
            mesh=weight.mesh,
            placements=tuple([_Partial()] * weight.mesh.ndim),
        )
        if weight
        else None
    )
    bias_grad = (
        DTensorSpec(
            mesh=bias.mesh,
            placements=tuple([_Partial()] * bias.mesh.ndim),
        )
        if bias
        else None
    )
    # 返回 OutputSharding 对象，包含 grad、weight_grad、bias_grad 的输出规范
    return OutputSharding(
        # 注意：以下的类型错误是合法的。这是因为 DTensor 目前不支持可选返回值。需要在 DTensor 仓库中修复。
        output_spec=(
            grad if grad_input_mask[0] else None,
            weight_grad if grad_input_mask[1] else None,
            bias_grad if grad_input_mask[2] else None,
        ),
    )


def _refine_sharding(
    op_schema: OpSchema, active_dim: Optional[int]
) -> Sequence[Placement]:
    """Considers 2 first inputs of op_schema as having same shape, and returns suggested placement for a pointwise operation."""
    # 将操作维度视为单例，以防止在其上进行分片
    # 然而，如果 active_dim 为 None，则意味着输入和输出的形状相同，我们将完全应用点对点规则。

    args_schema = []
    for s in op_schema.args_schema[:2]:
        # 确保 s 是 DTensorSpec 类型，并且其 tensor_meta 属性不为空
        assert isinstance(s, DTensorSpec) and s.tensor_meta is not None
        args_schema.append(
            DTensorSpec(
                mesh=s.mesh,  # type: ignore[attr-defined]
                placements=s.placements,  # type: ignore[attr-defined]
                tensor_meta=TensorMeta(
                    shape=torch.Size(
                        s.shape[0:active_dim] + (1,) + s.shape[active_dim + 1 :]
                    )
                    if active_dim is not None
                    else s.shape,
                    stride=s.tensor_meta.stride,
                    dtype=s.tensor_meta.dtype,
                ),
            )
        )

    op_schema = OpSchema(
        op=op_schema.op,
        args_schema=args_schema,  # type: ignore[arg-type]
        kwargs_schema={},
    )
    # 根据操作模式和线性性参数生成输出分片规则
    output_sharding = pointwise_rule(op_schema, linearity=False)
    
    # 如果输出规格存在
    if output_sharding.output_spec:
        # 断言输出规格是 DTensorSpec 类型
        assert isinstance(output_sharding.output_spec, DTensorSpec)
        # 返回输出规格的放置信息
        return output_sharding.output_spec.placements
    else:
        # 如果没有输出规格，则断言重新分配模式不为 None
        assert output_sharding.redistribute_schema is not None
        # 获取重新分配模式的参数模式中的第一个
        out_schema = output_sharding.redistribute_schema.args_schema[0]
        # 断言参数模式是 DTensorSpec 类型
        assert isinstance(out_schema, DTensorSpec)
        # 返回参数模式的放置信息的元组
        return tuple(out_schema.placements)
@register_prop_rule(aten.slice_scatter.default)  # pyre-ignore
def prop_slice_scatter(op_schema: OpSchema) -> OutputSharding:
    # 1. number of dimensions in input and src need to match.
    # 2. number of elements on all non-dim need to match between input and src.
    # 3. number of elements in src in dim need to match the slice size.
    # Given the above:
    # - We suggest for src to follow the sharding of input, except on the scatter dimension,
    #   where our best bet for now is to make them replicated as a fall-back.
    #   TODO: Ideally we'd like to make sure the output is re-sharded afterwards to keep input sharding.

    # 设置默认参数元组，如果参数不足则用默认值补充
    defaults = (None, None, 0, None, None, 1)
    # 从 op_schema 中解包参数
    input, src, dim, start, end, step = (
        op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    )
    # 断言确保 input 是 DTensorSpec 类型
    assert isinstance(input, DTensorSpec)
    # 断言确保 src 是 DTensorSpec 类型
    assert isinstance(src, DTensorSpec)
    # 断言确保 dim 是整数类型
    assert isinstance(dim, int)

    # 如果 dim 为负数，则转换为对应的正数索引
    if dim < 0:
        dim += input.ndim

    # 如果输入张量和源张量在操作维度上的形状相同，则直接返回输入的分片建议
    # 这相当于一个无操作，因此我们可以像对待点对点操作一样传播分片
    if input.shape[dim] == src.shape[dim]:
        assert start == 0
        assert end >= src.shape[dim]  # type: ignore[operator]
        dim = None

    # 应用点对点规则中实现的分片细化
    input_suggestion = list(_refine_sharding(op_schema, dim))
    # 在操作维度上应用例外 -- 禁止分片
    for i, p in enumerate(input_suggestion):
        if isinstance(p, Shard) and p.dim == dim:
            input_suggestion[i] = Replicate()
    input_suggestion = tuple(input_suggestion)  # type: ignore[assignment]

    # 如果我们的分片建议与输入的分片相同，并且源张量的分片也与输入的分片相同
    # 则输出的分片将与输入相同
    if input_suggestion == tuple(input.placements) and src.placements == tuple(
        input.placements
    ):
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=input.mesh,
                placements=input.placements,
            )
        )
    else:
        # 否则返回建议的输出分片规范
        return OutputSharding(
            output_spec=None,
            redistribute_schema=OpSchema(
                op=op_schema.op,
                args_schema=(
                    DTensorSpec(
                        mesh=input.mesh,
                        placements=input_suggestion,
                        tensor_meta=input.tensor_meta,
                    ),
                    DTensorSpec(
                        mesh=src.mesh,
                        placements=input_suggestion,
                        tensor_meta=src.tensor_meta,
                    ),
                )
                + op_schema.args_schema[2:],
                kwargs_schema=op_schema.kwargs_schema,
            ),
        )
```