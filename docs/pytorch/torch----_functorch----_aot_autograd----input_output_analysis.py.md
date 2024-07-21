# `.\pytorch\torch\_functorch\_aot_autograd\input_output_analysis.py`

```
"""
This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the following analyses are provided:
1. Refine the view and mutation metadata collected previously - removing duplicate
   inputs or mapping views to their bases.
2. We also analyze the function signature for export graphs.
"""

import itertools                                  # 导入 itertools 模块，用于高效的迭代操作
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch                                      # 导入 PyTorch 模块
import torch.utils._pytree as pytree              # 导入 PyTorch 内部模块 _pytree
from torch import Tensor                          # 导入 Tensor 类型
from torch._subclasses.functional_tensor import FunctionalTensor  # 导入 FunctionalTensor 类
from torch.fx.experimental.symbolic_shapes import is_concrete_int  # 导入 is_concrete_int 函数
from .. import config                             # 导入上层包的 config 模块
from .collect_metadata_analysis import coerce_tangent  # 导入本地包中的 collect_metadata_analysis 模块中的 coerce_tangent 函数
from .schemas import (
    BackwardSignature,                           # 导入 BackwardSignature 类型
    GraphSignature,                              # 导入 GraphSignature 类型
    InputAliasInfo,                              # 导入 InputAliasInfo 类型
    OutputAliasInfo,                             # 导入 OutputAliasInfo 类型
    OutputType,                                  # 导入 OutputType 类型
    ViewAndMutationMeta,                         # 导入 ViewAndMutationMeta 类型
)
from .utils import strict_zip                     # 导入本地包中的 strict_zip 函数

zip = strict_zip                                 # 将 strict_zip 函数赋值给 zip 变量


def remove_dupe_metadata(
    m: ViewAndMutationMeta,                       # 函数参数 m 的类型为 ViewAndMutationMeta
    keep_arg_mask: List[bool],                    # 函数参数 keep_arg_mask 的类型为布尔值列表
    add_dupe_map: List[int],                      # 函数参数 add_dupe_map 的类型为整数列表
) -> ViewAndMutationMeta:                        # 函数返回类型为 ViewAndMutationMeta
    assert len(m.input_info) == len(keep_arg_mask)  # 断言确保输入信息和 keep_arg_mask 的长度相同
    # Easy invariant: the first argument should never be a dupe (it will be kept)
    assert len(keep_arg_mask) > 0 and keep_arg_mask[0]  # 断言确保 keep_arg_mask 的长度大于 0 并且第一个元素为真

    # Filter dupe'd mutated inputs out of traced_tangents
    num_data_mutations = len([x for x in m.input_info if x.mutates_data])  # 统计 mutates_data 为真的输入数量
    other_traced_tangents = m.traced_tangents[num_data_mutations:]  # 获取除了数据变异之外的其他追踪切线
    inp_traced_tangents = m.traced_tangents[:num_data_mutations]  # 获取数据变异的追踪切线
    filtered_inp_traced_tangents = [
        # See Note [Tangents must be contiguous]
        x
        for i, x in enumerate(inp_traced_tangents)
        if keep_arg_mask[m.mutated_inp_runtime_indices[i]]  # 过滤掉不需要的变异输入追踪切线
    ]
    traced_tangents = filtered_inp_traced_tangents + other_traced_tangents  # 合并过滤后的追踪切线和其他追踪切线
    return ViewAndMutationMeta(
        input_info=[x for i, x in enumerate(m.input_info) if keep_arg_mask[i]],
        # 对于作为视图的输出，存储其来自于哪个输入的索引。需要更新这个索引以考虑移除的重复项。
        output_info=[
            OutputAliasInfo(
                output_type=o.output_type,
                raw_type=o.raw_type,
                dynamic_dims=o.dynamic_dims,
                base_idx=None if o.base_idx is None else add_dupe_map[o.base_idx],
                requires_grad=o.requires_grad,
                functional_tensor=o.functional_tensor,
            )
            for o in m.output_info
        ],
        num_intermediate_bases=m.num_intermediate_bases,
        keep_input_mutations=m.keep_input_mutations,
        traced_tangents=traced_tangents,
        # 我们保证不会到达这里，因为今天不支持带子类输入的重复项。
        subclass_inp_meta=[],  # 子类输入元数据为空列表
        subclass_fw_graph_out_meta=[],  # 子类前向图输出元数据为空列表
        subclass_tangent_meta=[],  # 子类切线元数据为空列表
        is_train=m.is_train,
    )
# 给定我们的 ViewAndMutation 元数据，这个函数构建一个新的元数据集，
# 在向函数中添加合成基础参数后。
# 这个函数的大部分工作是遍历所有与输入对应的元数据，
# 并使用我们的合成基础调用约定进行更新。
#
# 当 config.debug_assert 被设置时，我们会自动重新生成元数据，
# 并将其与此输出进行比较，以确保正确性。
#
# 除了更新后的元数据外，还返回需要在合成基础结尾处理中更新的输入索引列表。

def create_synthetic_base_metadata(
    m: ViewAndMutationMeta,
    # 将每个外部参数索引映射到其内部索引（如果此外部参数是从合成基础生成的，
    # 则会得到一个元组 (i, TensorMeta)，告诉你基础张量索引和视图元数据）
    synthetic_base_info: List[Union[int, Tuple[int, torch.Tensor]]],
    outer_args: List[Any],
    inner_args: List[Any],
) -> Tuple[ViewAndMutationMeta, List[int]]:
    # 将内部参数索引映射到外部参数索引的字典
    synthetic_base_to_indices: Dict[int, List[int]] = {}
    for inner_idx in range(len(inner_args)):
        # 当前基础参数的外部别名索引列表
        outer_aliased_indices_of_current_base_arg = [
            outer_idx
            for outer_idx, inner_idx_or_tuple in enumerate(synthetic_base_info)
            if (isinstance(inner_idx_or_tuple, int) and inner_idx_or_tuple == inner_idx)
            or (
                isinstance(inner_idx_or_tuple, tuple)
                and inner_idx_or_tuple[0] == inner_idx
            )
        ]
        synthetic_base_to_indices[inner_idx] = outer_aliased_indices_of_current_base_arg

    # 根据变异输入的 requires_grad 信息，
    # 生成相同变异输入的 requires_grad 信息，但在构建合成基础后。
    input_infos = []
    for outer_indices in synthetic_base_to_indices.values():
        # 对于每个 synthetic_base_to_indices 中的值，即一组索引列表，进行以下处理：

        # 检查是否所有的输入信息中的 is_leaf 属性要么全部为 True，要么全部为 False
        any_leaf = any(m.input_info[x].is_leaf for x in outer_indices)
        all_leaf = all(m.input_info[x].is_leaf for x in outer_indices)
        assert any_leaf == all_leaf

        # 如果 outer_indices 中的索引数量大于 1，则 mutates_data 设为 True，否则从第一个索引的 mutates_data 属性获取值
        mutates_data = (
            True
            if len(outer_indices) > 1
            else m.input_info[outer_indices[0]].mutates_data
        )

        # 如果 outer_indices 中的索引数量大于 1，则 mutates_metadata 设为 False，否则从第一个索引的 mutates_metadata 属性获取值
        mutates_metadata = (
            False
            if len(outer_indices) > 1
            else m.input_info[outer_indices[0]].mutates_metadata
        )

        # 检查 outer_indices 中是否有任何一个索引的 requires_grad 属性为 True
        requires_grad = any(m.input_info[x].requires_grad for x in outer_indices)

        # 检查 outer_indices 中是否所有索引的 mutations_hidden_from_autograd 属性为 True
        mutations_hidden_from_autograd = all(
            m.input_info[x].mutations_hidden_from_autograd for x in outer_indices
        )

        # 检查 outer_indices 中是否所有索引的 mutation_inductor_storage_resize 属性为 True
        mutation_inductor_storage_resize = all(
            m.input_info[x].mutation_inductor_storage_resize for x in outer_indices
        )

        # 根据收集的信息创建 InputAliasInfo 对象
        inpt_info = InputAliasInfo(
            # 如果 len(outer_indices) > 1，则说明该输入是一个 synthetic base。
            # 不变的规则是对于 aot autograd 的其余部分，只有当它们的别名之一发生数据变化时，synthetic base 才会出现。
            # 如果它们的别名之一发生元数据变化，它们将对 aot autograd 隐藏。
            mutates_data=mutates_data,
            mutates_metadata=mutates_metadata,
            mutations_hidden_from_autograd=all(
                m.input_info[x].mutations_hidden_from_autograd for x in outer_indices
            ),
            mutates_storage_metadata=False
            if len(outer_indices) > 1
            else m.input_info[outer_indices[0]].mutates_storage_metadata,
            mutations_under_no_grad_or_inference_mode=mutations_under_no_grad_or_inference_mode,
            mutation_inductor_storage_resize=mutation_inductor_storage_resize,
            is_leaf=any_leaf,
            requires_grad=requires_grad,
            keep_input_mutations=m.keep_input_mutations,
        )
        # 将创建的 InputAliasInfo 对象添加到 input_infos 列表中
        input_infos.append(inpt_info)

    # 找出满足以下条件的任何输入：
    # (1) 它们是 synthetic base 的一部分（因为它们是其他输入的别名，并且至少一个输入经历了数据变化）
    # (2) 它们经历了元数据变化
    outer_aliased_arg_idx_with_metadata_mutations = [
        outer_idx
        for outer_idx, inpt_info in enumerate(m.input_info)
        if inpt_info.mutates_metadata
        and not isinstance(synthetic_base_info[outer_idx], int)
    ]

    # 获取原始 requires grad 信息的输出，但不包括从变异输入获取的输出
    # 创建一个列表，用于存储输出别名信息的实例化对象
    input_metadata_output_info = [
        OutputAliasInfo(
            output_type=OutputType.alias_of_input,  # 设置输出类型为输入的别名
            raw_type=FunctionalTensor,  # 设置原始类型为FunctionalTensor
            dynamic_dims={  # 使用集合推导式确定动态维度的索引集合
                i
                for i, s in enumerate(outer_args[outer_idx].shape)
                if not is_concrete_int(s)
            },
            base_idx=synthetic_base_info[outer_idx][0],  # 设置基础索引为合成基础信息的第一个元素
            requires_grad=outer_args[outer_idx].requires_grad,  # 设置是否需要梯度
        )
        for outer_idx in outer_aliased_arg_idx_with_metadata_mutations  # 外部索引遍历
    ]

    # 初始化一个空列表，用于存储现有的输出信息
    existing_output_infos = []
    # 遍历现有模型输出信息列表
    for o in m.output_info:
        # 根据条件设置新的基础索引
        new_base_idx = (
            None
            if o.base_idx is None  # 如果原基础索引为空，则新基础索引也为空
            else (
                synthetic_base_info[o.base_idx]  # 否则根据原基础索引获取合成基础信息
                if isinstance(synthetic_base_info[o.base_idx], int)  # 如果合成基础信息是整数
                else synthetic_base_info[o.base_idx][0]  # 否则获取合成基础信息的第一个元素
            )
        )
        # 根据条件更新输出类型
        new_output_type = (
            OutputType.alias_of_input  # 如果原输出类型是输入，则新输出类型为输入的别名
            if o.output_type == OutputType.is_input and o.base_idx != new_base_idx  # 并且基础索引改变了
            else o.output_type  # 否则保持原输出类型不变
        )
        # 将更新后的信息添加到现有输出信息列表中
        existing_output_infos.append(
            OutputAliasInfo(
                output_type=new_output_type,  # 设置输出类型
                raw_type=o.raw_type,  # 设置原始类型
                dynamic_dims=o.dynamic_dims,  # 设置动态维度
                base_idx=new_base_idx,  # 设置基础索引
                requires_grad=o.requires_grad,  # 设置是否需要梯度
                functional_tensor=o.functional_tensor,  # 设置功能张量
            )
        )

    # 创建一个列表，用于存储内部变异的切线
    inner_mutated_tangents = [
        # 强制类型转换切线
        coerce_tangent(x)
        for inner_idx, x in enumerate(inner_args)  # 内部索引遍历
        if input_infos[inner_idx].mutates_data and input_infos[inner_idx].requires_grad  # 如果输入信息表明数据变异并且需要梯度
    ]

    # 合并现有输出信息和输入元数据输出信息，形成最终的输出信息列表
    output_info = existing_output_infos + input_metadata_output_info

    # 重新生成跟踪切线以包括变异输入，包括合成基础
    traced_tangents = (
        inner_mutated_tangents + m.traced_tangents[len(inner_mutated_tangents) :]  # 将内部变异的切线与现有模型的跟踪切线连接起来
    )

    # 返回元组，包含视图和变异元信息
    return (
        ViewAndMutationMeta(
            input_info=input_infos,  # 设置输入信息
            output_info=output_info,  # 设置输出信息
            num_intermediate_bases=m.num_intermediate_bases,  # 设置中间基数
            keep_input_mutations=m.keep_input_mutations,  # 设置是否保持输入变异
            traced_tangents=traced_tangents,  # 设置跟踪切线
            subclass_inp_meta=[],  # 子类输入元数据为空列表
            subclass_fw_graph_out_meta=[],  # 子类前向图输出元数据为空列表
            subclass_tangent_meta=[],  # 子类切线元数据为空列表
            is_train=m.is_train,  # 设置是否训练
        ),
        outer_aliased_arg_idx_with_metadata_mutations,  # 返回外部索引中带有元数据变异的参数索引
    )
# 获取张量 x 的最后一个内存地址
def _get_last_mem_address(x):
    out = x.storage_offset()  # 初始化为张量 x 的存储偏移量
    for size, stride in zip(x.size(), x.stride()):
        out += (size - 1) * stride  # 根据张量的大小和步长计算最后一个元素的内存地址
    return out  # 返回最后一个元素的内存地址


# 假设：x 和 y 共享存储空间，我们试图确定它们的内存是否完全不重叠，基于它们的大小、步长和存储偏移量
def _tensors_definitely_do_not_overlap(x, y):
    if x is y:
        return False  # 如果 x 和 y 是同一个张量对象，则它们肯定有重叠
    if x.numel() == 0 or y.numel() == 0:
        return True  # 如果任意一个张量的元素数为0，则它们肯定没有重叠

    # 确保 x 总是在左边
    if x.storage_offset() > y.storage_offset():
        x, y = y, x  # 如果 x 的存储偏移量大于 y 的存储偏移量，则交换它们的顺序

    # 短路情况：两个张量都是连续存储的情况下
    if x.is_contiguous() and y.is_contiguous():
        if x.storage_offset() + x.numel() > y.storage_offset():
            # 肯定重叠
            return False
        else:
            # 肯定不重叠
            return True

    # 短路情况：如果 x 的最后一个内存地址小于 y 的起始地址，则肯定不重叠
    x_last = _get_last_mem_address(x)
    if x_last < y.storage_offset():
        return True
    # 检查 x 和 y 是否都是二维张量，并且内存布局为非连续，且外部步长为 1
    # 这种情况适用于 shampoo 优化器。
    if x.dim() == 2 and y.dim() == 2 and x.stride(1) == 1 and y.stride(1) == 1:
        # 检查 x 和 y 的外部步长是否相同，即是否在内存中布局连续
        if x.stride(0) == y.stride(0):
            # 计算 y 的起始偏移相对于 x 的起始偏移的差值
            offset_delta = y.storage_offset() - x.storage_offset()
            # 如果偏移差小于 x 的第二维大小，则肯定有重叠
            if offset_delta < x.size(1):
                # 返回 False，表示有重叠
                return False
            # 计算 x 所覆盖的总元素数
            x_total_elems_covered = x.stride(0) * (x.size(0) - 1) + x.size(1)
            # 如果 x 的总覆盖元素数小于等于偏移差，肯定没有重叠
            if x_total_elems_covered <= offset_delta:
                # 返回 True，表示没有重叠
                return True
            # 计算偏移差对 x 的外部步长取模后的值
            offset_delta_mod = offset_delta % x.stride(0)
            # 如果 modded_offset 加上 y 的第二维大小小于等于 x 的外部步长，肯定没有重叠
            if offset_delta_mod + y.size(1) <= x.stride(0):
                return True
            else:
                return False
    # 如果不满足上述条件，返回 False，表示没有重叠
    return False
# 计算重叠输入的函数
def compute_overlapping_inputs(fwd_inputs, aliased_input_indices):
    # 获取允许动态形状的最大别名输入数量
    max_aliased_inps_w_dyn_shapes = (
        config._max_aliased_inputs_with_dynamic_shapes_enabled
    )
    definitely_error_on_dyn_shapes = False
    # 如果 JK（JustKnobs）未设置或为假，则遵循上述配置；如果为真，则始终在有别名且有动态形状的输入时报错
    if torch._inductor.config.is_fbcode():
        # 检查是否启用了特定的 JustKnobs 开关来禁用特定的动态形状与别名输入的组合
        definitely_error_on_dyn_shapes = torch._utils_internal.justknobs_check(
            "pytorch/dynamo:disable_aliased_inputs_with_mutation_and_dyn_shapes"
        )

    actual_aliased_indices = set()
    num_aliases = len(aliased_input_indices)
    # 如果别名数量大于等于 2，并且要么肯定在动态形状时报错，要么别名数量超过了允许的最大动态形状别名输入数量
    if num_aliases >= 2 and (
        definitely_error_on_dyn_shapes or num_aliases > max_aliased_inps_w_dyn_shapes
    ):
        dynamic_shape_indices = set()
        for j in range(num_aliases):
            j_ = aliased_input_indices[j]
            curr_inp = fwd_inputs[j_]
            # 检查当前输入是否具有任何符号整数，表示它的形状、步长或存储偏移是动态的
            if any(
                isinstance(x, torch.SymInt)
                for x in itertools.chain(
                    curr_inp.shape, curr_inp.stride(), [curr_inp.storage_offset()]
                )
            ):
                dynamic_shape_indices.add(j_)
        # 断言：没有动态形状的输入别名
        assert (
            len(dynamic_shape_indices) == 0
        ), f"""\
Encountered a graph where:
- {num_aliases} graph inputs all share the same storage (input indices: {str(aliased_input_indices)})
- at least one of these aliased inputs was mutated
- at least one of these inputs is being compiled with dynamic shapes (indices: {str(dynamic_shape_indices)})

Current limit: {str(max_aliased_inps_w_dyn_shapes)}
Killswitch enabled: {str(definitely_error_on_dyn_shapes)}

The most common way to run into this situation is when your model parameters are allocated as one giant buffer
and are all mutated by the optimizer, and some of your parameters end up getting compiled with dynamic shapes.

You can avoid this problem by marking your parameters so they explicitly do not participate in dynamic shapes,
by marking each dim of your parameter static:

torch._dynamo.mark_static(param, 0) # (1, 2, ... for every dimension on the parameter).

If you are running into this issue in a situation where your parameters are static but some other inputs
are aliased and mutated, and they should be dynamic, please file an issue.
"""
    # 检查所有可能的输入别名对，确认它们是否真正重叠
    for j in range(num_aliases):
        for i in range(j):
            j_ = aliased_input_indices[j]
            i_ = aliased_input_indices[i]
            if not _tensors_definitely_do_not_overlap(fwd_inputs[i_], fwd_inputs[j_]):
                actual_aliased_indices.add(i_)
                actual_aliased_indices.add(j_)
    # 返回实际重叠别名输入的索引集合
    return actual_aliased_indices


# 从图模块中获取所有输入节点的名称列表
def _graph_input_names(gm):
    return [node.name for node in gm.graph.find_nodes(op="placeholder")]


# 从图模块中获取所有输出节点的名称列表
def _graph_output_names(gm):
    output_node = next(iter(reversed(gm.graph.nodes)))
    # 断言输出节点的操作为 "output"，并且其参数列表长度为 1
    assert output_node.op == "output" and len(output_node.args) == 1
    # 获取输出节点的第一个参数（假设它是一个列表）
    return_args = output_node.args[0]
    # 返回列表中每个元素的 "name" 属性值（如果存在）
    return [getattr(return_arg, "name", None) for return_arg in return_args]
# 定义一个函数，用于创建图签名对象，输入参数如下：
def create_graph_signature(
    fx_g: torch.fx.GraphModule,  # 输入一个 Torch FX 图模块对象
    fw_metadata: ViewAndMutationMeta,  # 视图和变异元数据对象
    in_spec: pytree.TreeSpec,  # 输入规范，表示输入数据结构
    out_spec: pytree.TreeSpec,  # 输出规范，表示输出数据结构
    *,
    user_args_flat: List[Tensor],  # 扁平化的用户参数列表，包含张量对象
    params_and_buffers_flat: List[Tensor],  # 扁平化的参数和缓冲区列表，包含张量对象
    param_names: List[str],  # 参数名称列表
    buffer_names: List[str],  # 缓冲区名称列表
    trace_joint: bool,  # 布尔值，指示是否追踪联合
    num_user_fw_outs: Optional[int],  # 可选的整数，表示用户前向输出的数量
    loss_index: Optional[int],  # 可选的整数，表示损失索引
) -> GraphSignature:  # 函数返回一个图签名对象

    # 调用 _graph_input_names 函数，获取图输入的名称列表
    graph_input_names = _graph_input_names(fx_g)
    
    # 调用 _graph_output_names 函数，获取图输出的名称列表
    graph_output_names = _graph_output_names(fx_g)

    # 计算参数和缓冲区的总数
    num_params_buffers = len(param_names) + len(buffer_names)
    
    # 获取前向传播元数据中 token 的数量
    num_tokens = len(fw_metadata.tokens)
    
    # 计算用户参数的数量，根据图的限制条件，图输入的数量应为用户输入 + 参数 + 缓冲区 + token 数量
    num_user_args = len(graph_input_names) - num_params_buffers - num_tokens

    # 如果 trace_joint 为 True，则执行以下代码块
    if trace_joint:
        # 断言确保 num_user_fw_outs 不为 None
        assert num_user_fw_outs is not None
        
        # 计算前向传播输出的数量，考虑到用户前向输出和运行时输入索引的变异
        num_fw_outs = num_user_fw_outs + fw_metadata.num_mutated_inp_runtime_indices
        
        # 提取反向传播输出的名称
        backward_output_names = graph_output_names[num_fw_outs:]
        
        # 使用 itertools.count 创建一个计数器，用于生成梯度索引
        grad_index = itertools.count(0)
        
        # 创建梯度到参数的映射字典，仅对需要梯度计算的参数进行映射
        gradients_to_parameters = {
            backward_output_names[next(grad_index)]: param_names[i]
            for i, param in enumerate(params_and_buffers_flat)
            if param.requires_grad
        }
        
        # 创建梯度到用户输入的映射字典，仅对需要梯度计算的用户输入进行映射
        gradients_to_user_inputs = {
            backward_output_names[next(grad_index)]: graph_input_names[
                i + len(params_and_buffers_flat)
            ]
            for i, user_input in enumerate(user_args_flat)
            if user_input.requires_grad
        }
        
        # 断言确保梯度映射字典的总数与反向传播输出名称列表长度相等
        assert len(gradients_to_parameters) + len(gradients_to_user_inputs) == len(
            backward_output_names
        )

        # 创建反向传播签名对象，包括梯度到参数和梯度到用户输入的映射，以及损失输出的名称
        backward_signature = BackwardSignature(
            gradients_to_parameters,
            gradients_to_user_inputs,
            graph_output_names[loss_index],
        )
    else:
        # 如果 trace_joint 为 False，则将反向传播签名设为 None
        backward_signature = None
        
        # 计算用户前向输出的数量，考虑到变异输入运行时索引和 token 数量
        num_user_fw_outs = (
            len(graph_output_names)
            - fw_metadata.num_mutated_inp_runtime_indices
            - num_tokens
        )

    # 使用 GraphSignature 类的静态方法 from_tracing_metadata 创建并返回图签名对象
    return GraphSignature.from_tracing_metadata(
        in_spec=in_spec,
        out_spec=out_spec,
        graph_input_names=graph_input_names,
        graph_output_names=graph_output_names,
        view_mutation_metadata=fw_metadata,
        named_parameters=param_names,
        named_buffers=buffer_names,
        num_user_inputs=num_user_args,
        num_user_outputs=num_user_fw_outs,
        loss_index=loss_index,
        backward_signature=backward_signature,
    )
```