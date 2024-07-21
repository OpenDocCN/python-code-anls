# `.\pytorch\torch\distributed\_tensor\ops\matrix_ops.py`

```
# 导入 itertools 库，用于生成迭代器
import itertools
# 导入类型提示相关的模块
from typing import List, Optional

# 导入 PyTorch 库
import torch
# 导入分布式张量操作相关的模块和类
from torch.distributed._tensor._op_schema import OpSchema, OpStrategy, PlacementStrategy
from torch.distributed._tensor.ops.basic_strategy import gen_einsum_strategies
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    register_op_strategy,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)
# 导入设备网格相关的模块
from torch.distributed.device_mesh import DeviceMesh

# 使用 torch.ops.aten 别名为 aten
aten = torch.ops.aten

# 注册 aten.t.default 操作的转置策略
@register_op_strategy(aten.t.default)
def transpose_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 获取第一个参数自身的策略
    self_strategy = op_schema.args_schema[0]
    assert isinstance(self_strategy, OpStrategy)

    # 存储所有的转置策略
    transpose_strategies = []
    for input_strategy in self_strategy.strategies:
        input_spec = input_strategy.output_spec
        # 按照输入规范，转置 Shard 的放置策略
        output_placements = [
            Shard(1 - p.dim) if isinstance(p, Shard) else p
            for p in input_spec.placements
        ]
        # 创建新的放置策略
        transpose_strategy = PlacementStrategy(
            output_specs=DTensorSpec(
                mesh=input_strategy.output_spec.mesh,
                placements=tuple(output_placements),
            ),
            input_specs=(input_strategy.output_spec,),
        )
        # 将转置策略加入列表
        transpose_strategies.append(transpose_strategy)

    # 返回所有转置策略的 OpStrategy 对象
    return OpStrategy(strategies=transpose_strategies)


# 定义 _mm_like_strategy 函数，用于处理类似 mm 的操作策略
def _mm_like_strategy(
    mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # 获取操作的两个输入策略
    self_strategy, mat2_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    
    # 生成所有可能的 mm 策略
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    
    # 过滤掉无效的策略并关联成本
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        assert strtg.input_specs is not None
        self_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        # 检查是否可以对输入张量进行分片
        if is_tensor_shardable(self_strategy.shape, self_spec) and is_tensor_shardable(
            mat2_strategy.shape, mat2_spec
        ):
            redistribute_cost = [
                generate_redistribute_costs(self_strategy, self_spec),
                generate_redistribute_costs(mat2_strategy, mat2_spec),
            ]
            strtg.redistribute_cost = redistribute_cost
            filtered_strategies.append(strtg)

    # 更新 mm 策略对象的有效策略
    mm_strategy.strategies = filtered_strategies

    # 返回更新后的 mm 策略对象
    return mm_strategy


# 定义 _addmm_like_strategy 函数，用于处理类似 addmm 的操作策略
def _addmm_like_strategy(
    mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # 获取操作的三个输入策略
    self_strategy, mat1_strategy, mat2_strategy = op_schema.args_schema
    # 断言确保self_strategy是OpStrategy的实例
    assert isinstance(self_strategy, OpStrategy)
    # 断言确保mat1_strategy是OpStrategy的实例
    assert isinstance(mat1_strategy, OpStrategy)
    # 断言确保mat2_strategy是OpStrategy的实例
    assert isinstance(mat2_strategy, OpStrategy)
    
    # 获取self_strategy的形状
    self_shape = self_strategy.shape
    
    # 计算mm_out_shape，生成一个包含了所有可能的mm操作策略的形状
    mm_out_shape = torch.Size(
        [
            mat2_strategy.shape[-1] if i == len(mat1_strategy.shape) - 1 else dim_size
            for i, dim_size in enumerate(mat1_strategy.shape)
        ]
    )
    
    # 根据mm_equation和mesh生成所有可能的einsum操作策略
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    
    # 获取mm_strategy中的策略列表
    strategies = mm_strategy.strategies
    filtered_strategies = []
    
    # 遍历策略列表，过滤掉无效的策略并关联成本
    for strtg in strategies:
        # 确保strtg的输入规格不为空
        assert strtg.input_specs is not None
        # 获取mat1和mat2的规格
        mat1_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        out_spec = strtg.output_spec
    
        # 计算self参数的规格，考虑广播后的情况
        broadcast_dims_map = infer_broadcast_dims_map(mm_out_shape, self_shape)
        self_placements = map_placements_after_broadcast(
            out_spec.placements, mm_out_shape, broadcast_dims_map
        )
        self_spec = DTensorSpec(mesh=mesh, placements=self_placements)
    
        # 如果mat1和mat2的形状可分片，则更新输入规格以包含新的self规格
        if is_tensor_shardable(mat1_strategy.shape, mat1_spec) and is_tensor_shardable(
            mat2_strategy.shape, mat2_spec
        ):
            strtg.input_specs = (self_spec, mat1_spec, mat2_spec)
    
            # 关联成本信息
            redistribute_cost = [
                generate_redistribute_costs(self_strategy, self_spec),
                generate_redistribute_costs(mat1_strategy, mat1_spec),
                generate_redistribute_costs(mat2_strategy, mat2_spec),
            ]
            strtg.redistribute_cost = redistribute_cost
            filtered_strategies.append(strtg)
    
    # 更新mm_strategy的策略列表为过滤后的策略
    mm_strategy.strategies = filtered_strategies
    
    # 返回更新后的mm_strategy对象
    return mm_strategy
@register_op_strategy(aten.mm.default)
# 注册函数策略为aten.mm.default的操作策略
def mm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 返回使用_mm_like_strategy函数生成的策略，针对矩阵乘法"mk,kn->mn"
    return _mm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.addmm.default)
# 注册函数策略为aten.addmm.default的操作策略
def addmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 返回使用_addmm_like_strategy函数生成的策略，针对加法和矩阵乘法"mk,kn->mn"
    return _addmm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.bmm.default)
# 注册函数策略为aten.bmm.default的操作策略
def bmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 返回使用_mm_like_strategy函数生成的策略，针对批量矩阵乘法"bmk,bkn->bmn"
    return _mm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten.baddbmm.default)
# 注册函数策略为aten.baddbmm.default的操作策略
def baddmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 返回使用_addmm_like_strategy函数生成的策略，针对批量加法和矩阵乘法"bmk,bkn->bmn"
    return _addmm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten._scaled_dot_product_flash_attention.default)
# 注册函数策略为aten._scaled_dot_product_flash_attention.default的操作策略
def scaled_dot_product_flash_attention_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # 注意：目前我们仅支持一些简单的策略来支持张量并行处理
    # TODO: sdpa可能是我们探索分解分片传播的一个好候选，因为它涉及到矩阵乘法、逐点操作和归约操作。
    # 返回是否返回调试掩码，条件为参数模式的长度大于等于6并且第六个参数模式为真
    return_debug_mask = len(op_schema.args_schema) >= 6 and op_schema.args_schema[5]
    # 获取q输入策略作为q_input_strategy
    q_input_strategy = op_schema.args_schema[0]
    # 断言q/k/v具有相同的形状
    assert isinstance(q_input_strategy, OpStrategy)
    # 假设q/k/v具有相同的形状，赋值给qkv_shape
    qkv_shape = q_input_strategy.shape

    # 初始化所有网格维度策略为空列表
    all_mesh_dim_strategies = []
    # 遍历每个网格的维度
    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [outputs, inputs]
        # in the spda case, we have 3 valid tensor outputs and 3 tensor inputs
        # first we can always accept full replication for both inputs and outputs
        # 将全复制策略添加到单个网格维度策略列表中
        all_replicate: List[Placement] = [Replicate()] * 6
        single_mesh_dim_strategies.append(all_replicate)

        # second we can accept the sharding pattern of tensor parallelism, which
        # shard on the num of head dim
        # 定义按头部维度划分的策略
        qkv_sharding = Shard(1)  # num head dim
        output_sharding = Shard(1)  # num head dim
        logsumexp_sharding = Shard(1)  # num head dim
        if return_debug_mask:
            # 如果需要返回调试掩码，按头部维度划分调试掩码
            debug_attn_mask_sharding: Placement = Shard(1)  # num head dim
        else:
            # 否则，将调试掩码设为空（全复制）
            debug_attn_mask_sharding = Replicate()

        # 将按头部维度划分的策略组成列表
        num_heads_dim_sharding = [
            output_sharding,
            logsumexp_sharding,
            debug_attn_mask_sharding,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
        ]
        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        # Context Parallelism: shards on the sequence dim
        # 上下文并行性：在序列维度上进行划分
        single_mesh_dim_strategies.append(
            [
                Shard(2),  # output
                Shard(2),  # logsumexp
                Shard(2),  # debugattn
                Shard(2),  # q
                Shard(2),  # k
                Shard(2),  # v
            ]
        )

        # 将单个网格维度的策略列表添加到总策略列表中
        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    # 生成所有可能的策略组合
    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    # 初始化所有策略的列表
    all_strategies = []
    # 对于每个策略组合中的每一个组合，执行以下操作
    for strategy_comb in strategy_combs:
        # 创建一个空列表用于存储特定规格
        spec_list = []
        # 对于每个策略组合中的规格元组，创建 DTensorSpec 对象并添加到 spec_list 中
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        # 断言 spec_list 的长度为 6
        assert len(spec_list) == 6
        # 从 spec_list 中获取输入预期规格，即索引为 3 及之后的元素
        input_expected_specs = spec_list[3:]
        # 创建一个空列表 output_specs，用于存储输出规格
        output_specs: List[Optional[DTensorSpec]] = list(spec_list[:3])
        # 对 output_specs 进行修正，并为索引为 2 到 7 的位置插入 None 值，用于表示整数和空张量返回值
        for i in range(2, 8):
            output_specs.insert(i, None)
        
        # 如果所有的输入预期规格都可以进行张量分片
        if all(is_tensor_shardable(qkv_shape, spec) for spec in input_expected_specs):
            # 创建一个空列表 redistribute_cost，用于存储重新分布成本
            redistribute_cost = []
            # 遍历 input_expected_specs，根据 op_schema.args_schema 中的策略生成重新分布成本，并将其添加到 redistribute_cost 中
            for input_idx, spec in enumerate(input_expected_specs):
                qkv_strategy = op_schema.args_schema[input_idx]
                assert isinstance(qkv_strategy, OpStrategy)
                qkv_tensor_meta = qkv_strategy.strategies[0].output_spec.tensor_meta
                spec.tensor_meta = qkv_tensor_meta
                redistribute_cost.append(
                    generate_redistribute_costs(qkv_strategy, spec)
                )

            # 创建一个 PlacementStrategy 对象，并将其添加到 all_strategies 列表中
            strat = PlacementStrategy(
                output_specs=tuple(output_specs),
                input_specs=tuple(input_expected_specs),
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strat)

    # 返回 OpStrategy 对象，其中包含所有生成的策略
    return OpStrategy(all_strategies)
@register_op_strategy(aten._scaled_dot_product_flash_attention_backward.default)
def scaled_dot_product_flash_attention_backward_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # 获取输入 q 的策略，确保其为 OpStrategy 类型
    q_input_strategy = op_schema.args_schema[1]
    assert isinstance(q_input_strategy, OpStrategy)
    # 假设 q/k/v 具有相同的形状
    qkv_shape = q_input_strategy.shape

    # 查找所有输入张量的索引
    tensor_input_indices = [
        i
        for i, arg_spec in enumerate(op_schema.args_schema)
        if isinstance(arg_spec, OpStrategy)
    ]
    num_tensor_inputs = len(tensor_input_indices)

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list 存储 [输出, 输入] 的放置策略
        # 在 spda 反向传播中，有 3 个张量输出和 6 到 10 个张量输入
        # 首先，我们可以始终接受对输出和输入的完全复制
        all_replicate: List[Placement] = [Replicate()] * (3 + num_tensor_inputs)

        single_mesh_dim_strategies.append(all_replicate)

        # 其次，我们可以接受张量并行性的分片模式，即按头数进行分片
        grad_output_sharding = Shard(1)  # 头数维度
        qkv_sharding = Shard(1)  # 头数维度
        output_sharding = Shard(1)  # 头数维度
        logsumexp_sharding = Shard(1)  # 头数维度
        grad_qkv_sharding = Shard(1)  # 头数维度

        num_heads_dim_sharding: List[Placement] = [
            grad_qkv_sharding,
            grad_qkv_sharding,
            grad_qkv_sharding,
            grad_output_sharding,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
            output_sharding,
            logsumexp_sharding,
        ]
        # 在其余的张量输入上接受复制，可能是 cum_seq_q, cum_seq_k, philox_seed, philox_offset
        # 分别在索引 6, 7, 12, 13
        num_heads_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        # 上下文并行性：在序列维度上进行分片
        seq_dim_sharding: List[Placement] = [
            Shard(2),  # grad_q
            Shard(2),  # grad_k
            Shard(2),  # grad_v
            Shard(2),  # grad_output
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
            Shard(2),  # output
            Shard(2),  # logsumexp
        ]
        # 在其余的张量输入上接受复制，可能是 cum_seq_q, cum_seq_k, philox_seed, philox_offset
        # 分别在索引 6, 7, 12, 13
        seq_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
        single_mesh_dim_strategies.append(seq_dim_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    # 使用 itertools 生成所有可能的策略组合
    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    # 将所有策略组合收集到 all_strategies 中
    all_strategies = []
    for strategy_comb in strategy_combs:
        # 遍历策略组合列表中的每一个组合
        spec_list = []
        # 初始化空的规范列表
        for specs in zip(*strategy_comb):
            # 遍历当前策略组合中的每一个规范
            spec_list.append(DTensorSpec(mesh, tuple(specs)))
            # 创建 DTensorSpec 对象并添加到 spec_list 中

        assert len(spec_list) == 3 + num_tensor_inputs
        # 断言 spec_list 的长度为 3 加上输入张量的数量
        input_expected_specs = spec_list[3:]
        # 将 spec_list 中第四个元素及后续元素（即输入张量的规范）作为输入预期规范
        output_specs: List[Optional[DTensorSpec]] = list(spec_list[:3])
        # 将 spec_list 中前三个元素（即输出张量的规范）作为输出规范列表

        if all(
            is_tensor_shardable(qkv_shape, spec) for spec in input_expected_specs[:6]
        ):
            # 如果前六个输入预期规范都可分片
            # 只有当所有输入都可分片时才添加到策略列表中
            redistribute_cost = []
            # 初始化重新分布成本列表
            for input_idx, spec in enumerate(input_expected_specs):
                # 遍历输入预期规范列表
                qkv_strategy = op_schema.args_schema[tensor_input_indices[input_idx]]
                # 获取当前输入的策略
                assert isinstance(qkv_strategy, OpStrategy)
                # 断言当前策略是 OpStrategy 类型
                qkv_tensor_meta = qkv_strategy.strategies[0].output_spec.tensor_meta
                # 获取当前策略的第一个策略输出的张量元数据
                spec.tensor_meta = qkv_tensor_meta
                # 将当前规范的张量元数据设置为策略的张量元数据
                redistribute_cost.append(
                    generate_redistribute_costs(qkv_strategy, spec)
                )
                # 生成当前策略和规范的重新分布成本

            strat = PlacementStrategy(
                output_specs=tuple(output_specs),
                input_specs=tuple(input_expected_specs),
                redistribute_cost=redistribute_cost,
            )
            # 创建放置策略对象，包括输出规范、输入预期规范和重新分布成本
            all_strategies.append(strat)
            # 将策略添加到策略列表中

    return OpStrategy(all_strategies)
    # 返回 OpStrategy 对象，其中包含所有生成的放置策略
# 注册 constant_pad_nd.default 操作的策略函数，返回 OpStrategy 对象
@register_op_strategy(aten.constant_pad_nd.default)
def constant_pad_nd_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # TODO(d4l3k); 实现更正确的 constant_pad_nd 策略
    return OpStrategy(
        [
            PlacementStrategy(
                # 输出为完全复制的 DTensorSpec 规格
                output_specs=DTensorSpec(mesh, (Replicate(),)),
                # 输入为完全复制的 DTensorSpec 规格
                input_specs=(
                    DTensorSpec(mesh, (Replicate(),)),
                    DTensorSpec(mesh, (Replicate(),)),
                ),
                # 重新分配成本矩阵，全部为 [[1]]
                redistribute_cost=[[1]],
            )
        ]
    )


# 注册 _scaled_dot_product_efficient_attention.default 操作的策略函数
@register_op_strategy(aten._scaled_dot_product_efficient_attention.default)
def scaled_dot_product_efficient_attention_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # NOTE: 目前我们仅支持一些简单的策略来支持张量并行计算
    # 获取 q 输入的策略
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # 假设 q/k/v 具有相同的形状
    qkv_shape = q_input_strategy.shape
    # 检查是否有注意力偏置
    has_attn_bias = op_schema.args_schema[3] is not None
    # 计算 logsumexp 标志
    compute_log_sumexp = op_schema.args_schema[4]

    all_mesh_dim_strategies = []

    # 遍历设备网格的维度
    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # 位置列表存储 [输出, 输入] 的放置
        # 在 spda 案例中，我们有 2 个有效的张量输出和 3 或 4 个张量输入
        # 首先，我们始终接受输出和输入的完全复制
        all_replicate: List[Placement] = [Replicate()] * (5 + has_attn_bias)
        single_mesh_dim_strategies.append(all_replicate)

        # 其次，我们可以接受张量并行模式的分片模式，这在 heads 维度上分片
        qkv_sharding = Shard(1)
        output_sharding = Shard(1)
        if compute_log_sumexp:
            logsumexp_sharding: Placement = Shard(1)
        else:
            # 空的 logsumexp，完全复制
            logsumexp_sharding = Replicate()

        num_heads_dim_sharding = [
            output_sharding,
            logsumexp_sharding,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
        ]
        if has_attn_bias:
            num_heads_dim_sharding.append(Shard(1))
        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    # 生成所有策略组合的迭代器
    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    # 所有策略的列表
    all_strategies = []
    # 遍历给定的策略组合
    for strategy_comb in strategy_combs:
        # 初始化一个空的规格列表
        spec_list = []
        # 遍历每个策略组合中的规格，将其转换为 DTensorSpec 对象并添加到列表中
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        # 确保规格列表的长度符合预期，包括是否有注意力偏置
        assert len(spec_list) == (5 + has_attn_bias)
        # 确定输入和输出的规格列表
        input_expected_specs = spec_list[2:]
        output_specs: List[Optional[DTensorSpec]] = list(spec_list[:2])
        
        # 为标量张量返回的数值（例如 philox_seed 和 philox_offset）填充 None
        output_specs.extend([None, None])
        
        # 当所有输入都可以进行分片时才添加到策略列表中
        if all(is_tensor_shardable(qkv_shape, spec) for spec in input_expected_specs):
            # 初始化重分布成本列表
            redistribute_cost = []
            # 遍历输入期望的规格，生成重分布成本并添加到列表中
            for input_idx, spec in enumerate(input_expected_specs):
                qkv_strategy = op_schema.args_schema[input_idx]
                assert isinstance(qkv_strategy, OpStrategy)
                qkv_tensor_meta = qkv_strategy.strategies[0].output_spec.tensor_meta
                spec.tensor_meta = qkv_tensor_meta
                redistribute_cost.append(
                    generate_redistribute_costs(qkv_strategy, spec)
                )

            # 创建放置策略对象，并将其添加到策略列表中
            strat = PlacementStrategy(
                output_specs=tuple(output_specs),
                input_specs=tuple(input_expected_specs),
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strat)

    # 返回所有策略的操作策略对象
    return OpStrategy(all_strategies)
@register_op_strategy(aten._scaled_dot_product_efficient_attention_backward.default)
# 注册操作策略，使用默认的 scaled_dot_product_efficient_attention_backward 函数
def scaled_dot_product_efficient_attention_backward_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # 获取 q 输入的策略信息
    q_input_strategy = op_schema.args_schema[1]
    assert isinstance(q_input_strategy, OpStrategy)
    # 假设 q/k/v 具有相同的形状
    qkv_shape = q_input_strategy.shape
    # 检查是否存在 attention bias
    has_attn_bias = op_schema.args_schema[4] is not None

    # 获取所有是 OpStrategy 类型的张量输入的索引
    tensor_input_indices = [
        i
        for i, arg_spec in enumerate(op_schema.args_schema)
        if isinstance(arg_spec, OpStrategy)
    ]

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # 存储输出和输入的放置策略列表
        # 在 spda 反向情况下，有 4 个张量输出和 8 或 9 个张量输入
        # 注意：如果存在 attn_bias，则在 heads 维度上将 grad_bias 进行输出分片；
        #      否则 grad_bias 将为空，并且其 DTensorSpec 将被移除。
        # 首先，我们总是可以接受完全复制的输入和输出
        all_replicate: List[Placement] = [Replicate()] * (12 + has_attn_bias)

        single_mesh_dim_strategies.append(all_replicate)

        # 其次，我们接受张量并行化的分片模式，即在 heads 维度上进行分片
        grad_output_sharding = Shard(1)
        qkv_sharding = Shard(1)
        output_sharding = Shard(1)
        logsumexp_sharding = Shard(1)
        grad_qkv_sharding = Shard(1)
        grad_bias_sharding = Shard(1)

        num_heads_dim_sharding: List[Placement] = [
            grad_qkv_sharding,
            grad_qkv_sharding,
            grad_qkv_sharding,
            grad_bias_sharding,
            grad_output_sharding,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
            # 用于可选输入 attn_bias 的位置
            output_sharding,
            logsumexp_sharding,
        ]
        # 如果存在 attn_bias，则在 heads 维度上对 attn_bias 进行输入分片
        if has_attn_bias:
            num_heads_dim_sharding.insert(8, Shard(1))
        # 对其余的标量张量输入接受复制策略
        # 即 philox_seed 和 philox_offset
        num_heads_dim_sharding.extend([Replicate(), Replicate()])
        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    # 生成所有网格维度策略的组合
    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    # 遍历策略组合中的每一个组合
    for strategy_comb in strategy_combs:
        spec_list = []
        # 对每个策略组合中的规格进行处理
        for specs in zip(*strategy_comb):
            # 将每个组合转换为 DTensorSpec 对象并添加到 spec_list 中
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        # 断言 spec_list 的长度等于 12 加上是否有 attention bias 的结果
        assert len(spec_list) == (12 + has_attn_bias)

        # 确定输入期望规格，从第四个开始到末尾
        input_expected_specs = spec_list[4:]

        # 输出规格的列表，初始为前四个规格的 DTensorSpec，最后一个为 None 如果没有 attention bias
        output_specs: List[Optional[DTensorSpec]] = list(spec_list[:4])
        if not has_attn_bias:
            output_specs[-1] = None

        # 如果所有输入期望规格都可以进行分片，则添加到策略列表中
        if all(is_tensor_shardable(qkv_shape, spec) for spec in input_expected_specs):
            redistribute_cost = []
            # 遍历输入期望规格并生成重新分配成本
            for input_idx, spec in enumerate(input_expected_specs):
                qkv_strategy = op_schema.args_schema[tensor_input_indices[input_idx]]
                assert isinstance(qkv_strategy, OpStrategy)
                qkv_tensor_meta = qkv_strategy.strategies[0].output_spec.tensor_meta
                # 将 spec 的 tensor_meta 设置为 qkv_tensor_meta
                spec.tensor_meta = qkv_tensor_meta
                # 生成重新分配成本并添加到 redistribute_cost 列表中
                redistribute_cost.append(
                    generate_redistribute_costs(qkv_strategy, spec)
                )

            # 创建一个 PlacementStrategy 对象并将其添加到 all_strategies 列表中
            strat = PlacementStrategy(
                output_specs=tuple(output_specs),
                input_specs=tuple(input_expected_specs),
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strat)

    # 返回 OpStrategy 对象，包含所有生成的策略
    return OpStrategy(all_strategies)
```