# `.\pytorch\torch\distributed\_tensor\ops\common_rules.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
# 导入必要的类型提示和模块
from typing import cast, Dict, List, Optional, Tuple

import torch
from torch.distributed._tensor._op_schema import (
    _is_inplace_op,
    _is_out_variant_op,
    OpSchema,
    OutputSharding,
)
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta

# 定义替换字符串中指定位置字符的函数
def _replace_char_in_str(string: str, new_char: str, idx: int) -> str:
    return string[:idx] + new_char + string[idx + 1 :]

# 生成重新分片建议的函数
def _gen_reshard_suggestions(
    op_schema: OpSchema,
    input_dims: List[str],
    input_specs: Tuple[DTensorSpec, ...],
    dim_to_sharding: Dict[str, int],
    pending_sum: List[int],
) -> OutputSharding:
    # 初始化建议的参数规格列表
    suggested_arg_specs: List[DTensorSpec] = []
    # 遍历输入维度和对应的规格
    for input_dim, input_spec in zip(input_dims, input_specs):
        # 根据输入维度映射到分片值列表
        dim_map = [dim_to_sharding[dim] for dim in input_dim]
        # 创建新的张量规格，使用映射和待定求和
        suggested_arg_specs.append(
            DTensorSpec.from_dim_map(
                mesh=input_spec.mesh,
                dim_map=dim_map,
                sums=pending_sum,
                tensor_meta=input_spec.tensor_meta,
            )
        )
    # 基于操作模式和建议的参数规格生成新的操作模式
    suggested_schema = OpSchema(op_schema.op, tuple(suggested_arg_specs), {})
    # 在原始操作模式上执行就地重新包装模式建议
    suggested_schema._inplace_rewrap_schema_suggestion(op_schema)
    # 返回输出分片对象，其中重分配模式使用建议的操作模式
    return OutputSharding(
        None,
        redistribute_schema=suggested_schema,
    )

# einop_rule 函数定义
def einop_rule(
    equation: str,
    op_schema: OpSchema,
    *,
    linearity: bool = False,
    enforce_sharding: Optional[Dict[str, int]] = None,
) -> OutputSharding:
    """
    Propagate the sharding of inputs to output for ops whose data moves according to einsum notation.

    This is mostly borrowed from @zdevito's sharding simulator. Examples:
        mk,kn->mn - einsum
        ij,ij->ij - addition
        ij,j->ij - broadcasted addition
        ij->i - reduction
    Other ops could use this propagation algorithm when applied, note
    that einsum propagation only deal with list of specs (DTensor specs)
    as it only works on list of tensors!

    linearity in einop_rule means that the calling op `f` follows this rule:
        f(a + b) = f(a) + f(b)

    In this case we can propagate the partial sum, note that linearity in einop
    only applies to partial sum, not other operations like min/max (which are
    associative but not linear).
    """
    # 解析 einop 方程并提取参数规格
    inputs, outputs = equation.split("->")
    input_dims, output_dims = inputs.split(","), outputs.split(",")
    input_specs = op_schema.args_spec
    # 注意: 除非未来需要，否则仅支持单个输出
    output_dim = output_dims[0]

    # 初始化维度到分片值的映射字典和维度到大小的映射字典
    dim_to_sharding: Dict[str, int] = {}
    dim_to_size: Dict[str, int] = {}
    # 记录待定求和，键是网格维度，值是跨输入规格的待定求和计数器
    pending_sums_counter: Dict[int, int] = {}
    # 记录已见分片，键是分片值，值是描述字符串
    seen_shardings: Dict[int, str] = {}
    # 标志变量，指示是否需要重新分片操作的布尔值
    needs_reshard = False

    def merge_sharding(dim: str, a: int, b: int) -> int:
        # 合并输入的分片情况，如果可以合并的话，触发重新分片操作
        if a != b:
            if a == -1 or b == -1:
                # 将复制品重新分片以匹配已分片的部分
                nonlocal needs_reshard
                needs_reshard = True
                return a if a != -1 else b
            else:
                # TODO: 进一步正确地合并分片（例如，重新分片一个输入以复制）
                raise RuntimeError(
                    f"{equation}: dim {dim} sharded two different ways: {a} and {b}"
                )
        else:
            return a

    # 遍历输入的维度和规范
    for input_dim, input_spec in zip(input_dims, input_specs):
        # 处理部分和
        input_sums = input_spec.sums
        for sum_dim in input_sums:
            if sum_dim not in pending_sums_counter:
                seen_shardings[sum_dim] = "+"
            # 更新待处理和计数器，记录每个输入中的出现次数
            pending_sums_counter[sum_dim] = pending_sums_counter.get(sum_dim, 0) + 1

        # 遍历输入维度和规范的索引和对应关系
        for idx, (dim, mesh_dim) in enumerate(zip(input_dim, input_spec.dim_map)):
            if enforce_sharding and dim in enforce_sharding:
                if enforce_sharding[dim] != mesh_dim:
                    needs_reshard = True
                dim_to_sharding[dim] = enforce_sharding[dim]
                dim_to_size[dim] = input_spec.shape[idx]
            elif dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
                dim_to_size[dim] = input_spec.shape[idx]
            else:
                # 合并分片，更新维度到分片的映射
                dim_to_sharding[dim] = merge_sharding(
                    dim, dim_to_sharding[dim], mesh_dim
                )
                assert dim_to_size[dim] == input_spec.shape[idx]

            # 合并分片后，检查同一网格维度上是否存在多个分片
            merged_sharding_for_dim = dim_to_sharding[dim]
            if merged_sharding_for_dim != -1:
                if (
                    merged_sharding_for_dim in seen_shardings
                    and dim != seen_shardings[merged_sharding_for_dim]
                ):
                    needs_reshard = True
                    seen_shardings[merged_sharding_for_dim] += dim
                else:
                    seen_shardings[merged_sharding_for_dim] = dim

    # 如果有待处理和计数器且不是线性的情况下
    if pending_sums_counter and not linearity:
        # 返回重新分片建议，因为已经正确合并了分片，该重新分片建议是合理的
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, []
        )
    else:
        # 如果不是所有输入参数都是部分的，但支持线性操作，
        # 则无法进行分片传播，建议将所有输入在相应的网格维度上设为部分的
        # （所有输入在网格维度上都应该是部分的，以便在本地执行并延迟求和操作）
        for value in pending_sums_counter.values():
            if value != len(input_specs):
                needs_reshard = True

    # 遍历已经看到的分片信息，即网格维度和相应的维度集合
    for mesh_dim, dims in seen_shardings.items():
        if len(dims) > 1:
            # 发现在同一网格维度上有不同的输入维度在进行分片
            # 为了执行本地操作计算，我们需要根据一些简单的启发式方法重新分片输入
            # 现在我们简单地选择具有最小通信量的输入（即最小尺寸的输入）
            # TODO: 考虑采用更高级的启发式方法选择最佳的分片方式
            costs = []
            for d in dims:
                cost = 0
                for input_dim, input_spec in zip(input_dims, input_specs):
                    if (
                        d in input_dim
                        and input_spec.dim_map[input_dim.index(d)] == mesh_dim
                    ):
                        assert input_spec.tensor_meta is not None
                        global_shape = input_spec.tensor_meta.shape
                        local_shape = compute_local_shape(
                            global_shape, input_spec.mesh, input_spec.placements
                        )
                        cost += prod(local_shape) * input_spec.mesh.size(mesh_dim)
                costs.append(cost)
            d_to_keep_sharding = dims[costs.index(max(costs))]
            for d in dims:
                # 更新dim_to_sharding以保持具有最高通信量的维度的分片，
                # 并使其余维度复制
                if d != d_to_keep_sharding:
                    dim_to_sharding[d] = -1

    # 获取待处理的求和列表
    pending_sums = list(pending_sums_counter.keys())
    if needs_reshard:
        # 如果需要重新分片，返回重新分片建议
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, pending_sums
        )

    # 如果不需要重新分片，直接生成输出的分片
    output_dim_map = []
    output_shape = []
    for dim in output_dim:
        if dim == "1":
            # 找到输出维度中的单例维度，标记分片和形状
            output_dim_map.append(-1)
            output_shape.append(1)
        else:
            output_dim_map.append(dim_to_sharding[dim])
            output_shape.append(dim_to_size[dim])

    # XXX: 由于我们仍然需要进行中间形状计算，我们需要
    # 确保输入规格的第一个元素的张量元数据不为空
    assert input_specs[0].tensor_meta is not None
    # 创建一个张量元数据对象，包括输出形状、输入规格第一个元素的步幅和数据类型
    tensor_meta = TensorMeta(
        torch.Size(output_shape),                  # 使用输出形状创建张量大小对象
        input_specs[0].tensor_meta.stride,         # 使用输入规格第一个元素的步幅
        input_specs[0].tensor_meta.dtype,          # 使用输入规格第一个元素的数据类型
    )
    # 返回一个输出分片对象，其中包含使用维度映射、输出维度映射、待定求和和张量元数据
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_specs[0].mesh,                   # 输入规格第一个元素的网格
            output_dim_map,                        # 输出维度映射
            pending_sums,                          # 待定求和
            tensor_meta=tensor_meta,               # 张量元数据
        )
    )
def pointwise_rule(op_schema: OpSchema, linearity: bool = False) -> OutputSharding:
    """
    Propagate the sharding for pointwise operations.

    Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    """
    # 定义字母表用于生成维度字符
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    # 查找输入参数中的最大维度，以便处理广播情况
    input_specs = op_schema.args_spec
    max_dim = max(input.ndim for input in input_specs)
    
    # 初始化维度字符列表和单例维度计数器
    dimchars = []
    singleton_counter: List[int] = [0] * max_dim
    
    # 遍历输入参数
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        
        # 处理广播到共同形状的情况
        if len(input_specs) > 1:
            for i in range(max_dim):
                if i < start_dim:
                    # 将前导缺失维度字符视为单例维度
                    singleton_counter[i] += 1
                elif input.shape[i - start_dim] == 1:
                    # 将单例维度字符标记为特殊的 "1"
                    singleton_counter[i] += 1
                    p = _replace_char_in_str(p, "1", (i - start_dim))
        
        dimchars.append(p)
    
    # 初始化输出维度字符列表
    out_dimchars = alphabet[:max_dim]
    
    # 检查是否将所有输入的维度字符替换为单例维度
    for output_dim_idx in range(len(out_dimchars)):
        out_dimchar = out_dimchars[output_dim_idx]
        if singleton_counter[output_dim_idx] == len(input_specs):
            out_dimchars = _replace_char_in_str(out_dimchars, "1", output_dim_idx)
    
    # 构建格式字符串
    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"
    
    # 初始化强制执行分片的字典
    enforce_sharding: Dict[str, int] = {}
    
    # 根据操作类型设置强制执行分片的规则
    if _is_inplace_op(op_schema.op):
        # 就地操作应保留其写入的输入分片
        for out_dimchar, mesh_dim in zip(out_dimchars, input_specs[0].dim_map):
            enforce_sharding[out_dimchar] = mesh_dim
    elif _is_out_variant_op(op_schema.op):
        # 输出变体操作需要依据输出参数设置分片规则
        out_spec = cast(DTensorSpec, op_schema.kwargs_schema["out"])
        for out_dimchar, mesh_dim in zip(out_dimchars, out_spec.dim_map):
            enforce_sharding[out_dimchar] = mesh_dim
    
    # 调用 einop_rule 函数，传递格式字符串和其他参数，返回结果
    return einop_rule(
        fmt,
        op_schema,
        linearity=linearity,
        enforce_sharding=enforce_sharding,
    )
```