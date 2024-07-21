# `.\pytorch\torch\_functorch\_aot_autograd\subclass_utils.py`

```py
# mypy: allow-untyped-defs
"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""

from typing import Any, List, Optional, Tuple, Union

import torch.utils._pytree as pytree

from torch import Tensor
from torch._subclasses.fake_tensor import get_plain_tensors
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .schemas import MutationType, SubclassCreationMeta, ViewAndMutationMeta
from .utils import strict_zip

# 导入 strict_zip 函数并将其赋值给 zip
zip = strict_zip


def requires_subclass_dispatch(args, fw_metadata: ViewAndMutationMeta) -> bool:
    # 将输入参数 args 展开为列表
    args_flattened = pytree.arg_tree_leaves(*args)
    # 检查是否存在任何是 Tensor 类型的参数，并且是可追踪包装子类的
    any_subclass_args = any(
        is_traceable_wrapper_subclass(x)
        for x in args_flattened
        if isinstance(x, Tensor)
    )
    from torch._functorch._aot_autograd.schemas import SubclassCreationMeta

    # 检查任何输出是否为 SubclassCreationMeta 类型的元数据
    any_subclass_outputs = any(
        type(x) is SubclassCreationMeta for x in fw_metadata.subclass_fw_graph_out_meta
    )
    # 返回是否需要进行子类分发的布尔值结果
    return any_subclass_args or any_subclass_outputs


def create_subclass_metadata(a, start_idx):
    # 如果参数 a 不是可追踪包装子类，则返回 None 和增加后的起始索引
    if not is_traceable_wrapper_subclass(a):
        return None, start_idx + 1

    # 获取内部键和元数据
    inner_keys, metadata = a.__tensor_flatten__()
    new_start_idx = start_idx
    attrs = {}
    for key in inner_keys:
        # 递归创建子类元数据
        new_subclass_meta, new_start_idx = create_subclass_metadata(
            getattr(a, key), new_start_idx
        )
        attrs[key] = new_subclass_meta

    # 返回 SubclassCreationMeta 对象和更新后的起始索引
    return (
        SubclassCreationMeta(
            flat_tensor_start_idx=start_idx,
            arg_count=new_start_idx - start_idx,
            attrs=attrs,
            meta=metadata,
            outer_size=a.size(),
            outer_stride=a.stride(),
            original_subclass=a,
        ),
        new_start_idx,
    )


# 给定一个真实的 tensor 子类，返回一个 Plain tensor 类型的嵌套列表
def get_types_for_subclass(tensor_subclass):
    # 如果 tensor_subclass 不是可追踪包装子类，则返回一个包含 "Tensor" 的列表
    if not is_traceable_wrapper_subclass(tensor_subclass):
        return ["Tensor"]
    # 获取内部键和元数据
    inner_keys, _ = tensor_subclass.__tensor_flatten__()
    result = []
    for key in inner_keys:
        # 递归获取子类类型
        inner_tensor = getattr(tensor_subclass, key)
        result.extend(get_types_for_subclass(inner_tensor))
    # 返回结果列表
    return result


# 给定一个扁平化的参数列表，其中某些参数可能是 tensor 子类，
# 计算关于如何重建当前子类列表的元数据
def create_subclass_meta(
    curr_args: Union[List[Any], Tuple[Any, ...]]
) -> List[Union[int, SubclassCreationMeta]]:
    idx = 0
    infos: List[Union[int, SubclassCreationMeta]] = []
    # 返回包含信息的列表
    return infos
    # 遍历当前参数列表 `curr_args`
    for a in curr_args:
        # 检查参数 `a` 是否为 Tensor 类型并且是可追踪包装器的子类
        if isinstance(a, Tensor) and is_traceable_wrapper_subclass(a):
            # 记录当前索引位置到 `start_idx`
            start_idx = idx
            # 创建子类元数据 `subclass_meta`，并忽略返回的第二个值
            subclass_meta, _ = create_subclass_metadata(a, start_idx)
            # 将子类元数据添加到 `infos` 列表中
            infos.append(subclass_meta)
            # 设置 `cnt` 为子类元数据的参数数量
            cnt = subclass_meta.arg_count
        else:
            # 将当前索引位置 `idx` 添加到 `infos` 列表中
            infos.append(idx)
            # 设置 `cnt` 为 1，表示当前参数占用一个位置
            cnt = 1
        # 增加 `idx` 的值，以便处理下一个参数
        idx += cnt
    # 返回最终的 `infos` 列表，其中包含了每个参数的索引或者子类元数据
    return infos
# Output structure:
# - List[Tensor] if tracing an inference graph
# - Tuple[List[Tensor], List[Tensor]] if tracing a joint graph.
# This function effectively concats each inner list of subclass tensors
# into a (potentially longer) list of inner tensors.
#
# This function takes in a pytree of arguments and unwraps any tensor subclasses.
# Annoyingly, we can't use pytrees to perform the unwrapping, because unwrapping returns
# a list of tensors that we would then need to concat together.
# Instead, we specialize the logic for the inference vs. joint graph case.
# NOTE: this function is hot, since we unwrap tensor subclass inputs at runtime
def unwrap_tensor_subclasses(wrapped_args, *, is_joint_structure: bool):
    # Function to concatenate inner tensors from subclass tensors
    def concat_inner_tensors_from_subclasses(xs):
        xs_inner = []
        for x in xs:
            if is_traceable_wrapper_subclass(x):
                xs_inner.extend(get_plain_tensors(x))
            else:
                xs_inner.append(x)
        return xs_inner

    # Logic for handling joint structure vs. inference graph structure
    if is_joint_structure:
        assert isinstance(wrapped_args, tuple) and len(wrapped_args) == 2
        assert isinstance(wrapped_args[0], (tuple, list)) and isinstance(
            wrapped_args[1], (tuple, list)
        )
        # Unwrap tensors from each part of the joint structure
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args[0])
        unwrapped_args_tangents = concat_inner_tensors_from_subclasses(wrapped_args[1])
        unwrapped_args = (unwrapped_args_fw, unwrapped_args_tangents)
    else:
        assert isinstance(wrapped_args, (list, tuple))
        # Unwrap tensors from the single list structure
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args)
        unwrapped_args = unwrapped_args_fw
    return unwrapped_args


# Turns a flattened list of tensor arguments into (maybe) subclass tensors.
# This function is used both at trace time and runtime, so we have an is_runtime flag telling us which context we're in.
def wrap_tensor_subclasses(
    unwrapped_args: Union[Tuple[Any, ...], List[Any]],
    *,
    subclass_metas: List[Union[int, SubclassCreationMeta]],
    num_fw_outs_saved_for_bw: Optional[int] = None,
    is_runtime: bool = False,
) -> Tuple[Any, ...]:
    wrapped_args = []
    num_args_tallied = 0
    # Iterate over subclass metas to wrap tensors into subclass tensors
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, int):
            wrapped_args.append(unwrapped_args[subclass_meta])
            num_args_tallied += 1
        else:
            assert isinstance(subclass_meta, SubclassCreationMeta)
            wrapped_args.append(
                subclass_meta.creation_fn(unwrapped_args, is_runtime=is_runtime)
            )
            num_args_tallied += subclass_meta.arg_count

    # Note: [Partitioner handling for Subclasses, Part 2]
    # At the beginning of AOTAutograd, we collect metadata on the inputs and outputs of the user fw,
    # to figure out which inputs/outputs are subclasses, and how to reconstruct the subclasses after flattening them.
    #
    # When this function is called at runtime in the forward,
    # 如果 num_fw_outs_saved_for_bw 不为 None，则执行以下逻辑
    if num_fw_outs_saved_for_bw is not None:
        # 断言：确保 unwrapped_args 的长度等于 num_args_tallied + num_fw_outs_saved_for_bw
        assert len(unwrapped_args) == num_args_tallied + num_fw_outs_saved_for_bw, (
            f"Expected the number actual unwrapped-subclass outputs {len(unwrapped_args)} to equal "
            f"the number of args calculated from subclasses ({num_args_tallied}) plus the number of "
            f"additional activations saved for the backward pass ({num_fw_outs_saved_for_bw})"
        )
        # 获取 activations，这些是在 backward pass 中保存的额外激活值
        activations = unwrapped_args[num_args_tallied:]
        # 如果 wrapped_args 和 activations 都是元组，则返回它们的连接结果
        if isinstance(wrapped_args, tuple) and isinstance(activations, tuple):
            return wrapped_args + activations
        # 否则将 wrapped_args 和 activations 转换为列表，然后连接它们成为一个元组并返回
        return tuple(list(wrapped_args) + list(activations))
    else:
        # 如果 num_fw_outs_saved_for_bw 为 None，则执行以下逻辑
        # 断言：确保 unwrapped_args 的长度等于 num_args_tallied
        assert len(unwrapped_args) == num_args_tallied
        # 返回 wrapped_args 的元组形式
        return tuple(wrapped_args)
# 给定一组“密集”张量参数，此函数（可能）将它们包装成张量子类。
# 此函数仔细处理推断与联合情况：
# - 当 is_joint_structure 为 True 时，args 是 (primals, tangents)
# - 当 is_joint_structure 为 False 时，args 是 [*primals]
def wrap_tensor_subclasses_maybe_joint(
    unwrapped_args, *, is_joint_structure: bool, meta: ViewAndMutationMeta
) -> Union[Tuple[Any, ...], List[Any]]:
    # 由于此函数同时用于推断和联合图形，
    if is_joint_structure:
        assert isinstance(unwrapped_args, tuple) and len(unwrapped_args) == 2
        assert isinstance(unwrapped_args[0], (tuple, list)) and isinstance(
            unwrapped_args[1], (tuple, list)
        )
        # 解包 primals 和 tangents
        primals, tangents = unwrapped_args[0], unwrapped_args[1]
        # 包装 primals 和 tangents 的张量子类
        wrapped_primals = wrap_tensor_subclasses(
            primals, subclass_metas=meta.subclass_inp_meta
        )
        wrapped_tangents = wrap_tensor_subclasses(
            tangents, subclass_metas=meta.subclass_tangent_meta
        )
        return (wrapped_primals, wrapped_tangents)
    else:
        # 包装 unwrapped_args 的张量子类
        wrapped_args = wrap_tensor_subclasses(
            unwrapped_args, subclass_metas=meta.subclass_inp_meta
        )
        return wrapped_args


# TODO: UNUSED. delete?
# 为子类创建元数据
def create_metadata_for_subclass(meta: ViewAndMutationMeta) -> ViewAndMutationMeta:
    # 输入信息
    input_info = []
    for inp, subclass_meta in zip(meta.input_info, meta.subclass_inp_meta):
        num_inps = 1 if isinstance(subclass_meta, int) else subclass_meta.arg_count
        for _ in range(num_inps):
            input_info.append(inp)

    # 输出信息
    output_info = []
    subclass_out_meta_user_outs_only = meta.subclass_fw_graph_out_meta[
        meta.num_mutated_inp_runtime_indices :
    ]
    if meta.num_intermediate_bases > 0:
        subclass_out_meta_user_outs_only = subclass_out_meta_user_outs_only[
            : -meta.num_intermediate_bases
        ]
    # 断言检查
    assert len(meta.output_info) == len(subclass_out_meta_user_outs_only)
    # 假设输出信息对其内部张量共享
    for out, subclass_meta in zip(meta.output_info, subclass_out_meta_user_outs_only):
        num_outs = 1 if isinstance(subclass_meta, int) else subclass_meta.arg_count
        for _ in range(num_outs):
            output_info.append(out)

    # 有些不太正式，但实际上我们不关心这里的所有元数据。
    # 此元数据在自动求导和子类解糖之下使用，
    # 因此我们真正关心的只有像：
    # - 输入/输出数量（分区器所需）
    # - 输入突变（今天未使用，因为我们不处理子类内部的输入突变，尽管最终应该处理）
    # TODO: 添加一个测试用例，以确保我们在发生此类情况时报错，而不是得到静默的正确性
    num_intermediate_bases = None
    # 从 meta 中获取 keep_input_mutations 属性的值
    keep_input_mutations = meta.keep_input_mutations
    # 初始化 traced_tangents 变量为 None
    traced_tangents = None
    # 初始化 subclass_inp_meta 变量为 None
    subclass_inp_meta = None
    # 初始化 subclass_fw_graph_out_meta 变量为 None
    subclass_fw_graph_out_meta = None
    # 初始化 subclass_tangent_meta 变量为 None
    subclass_tangent_meta = None

    # 创建 ViewAndMutationMeta 的实例，并传入以下参数：
    #   - input_info: 输入信息
    #   - output_info: 输出信息
    #   - num_intermediate_bases: 中间基数的数量
    #   - keep_input_mutations: 是否保留输入的变化（从 meta 中获取）
    #   - traced_tangents: 追踪的切线信息（初始为 None）
    #   - subclass_inp_meta: 子类的输入元信息（初始为 None）
    #   - subclass_fw_graph_out_meta: 子类的前向图输出元信息（初始为 None）
    #   - subclass_tangent_meta: 子类的切线元信息（初始为 None）
    metadata = ViewAndMutationMeta(
        input_info=input_info,  # type: ignore[arg-type]
        output_info=output_info,  # type: ignore[arg-type]
        num_intermediate_bases=num_intermediate_bases,  # type: ignore[arg-type]
        keep_input_mutations=keep_input_mutations,  # type: ignore[arg-type]
        traced_tangents=traced_tangents,  # type: ignore[arg-type]
        subclass_inp_meta=subclass_inp_meta,  # type: ignore[arg-type]
        subclass_fw_graph_out_meta=subclass_fw_graph_out_meta,  # type: ignore[arg-type]
        subclass_tangent_meta=subclass_tangent_meta,  # type: ignore[arg-type]
    )
    # 返回创建的 metadata 实例作为函数的结果
    return metadata
# Note: [Recomputing subclass mutation handling]
#
# Generally, if a subclass requires grad, its components will not require grad.
# But for the purposes of tracking returned tensors, we should treat those component
# tensors as if they require grad.
#
# For example, if the subclass tensor requires grad and will be mutated in a way that
# requires us to handle the mutation outside of the graph, we need to return it
# from the forward graph. The inner_meta data won't consider the component tensors
# as if they need to be returned, because they don't require grad; but really, we
# should handle those tensors the same way we handle the subclass tensor itself; i.e.
# if we'd include the subclass tensor as part of the outputs, then we should also
# include the component tensors.
#
# To do this, we patch num_mutated_inp_runtime_indices below by expanding the inputs
# from the outer subclass tensors and propagating

updated_input_info = []
inner_idx = 0
if not fw_metadata.subclass_inp_meta:
    # Sometimes we don't have subclass info, e.g. synthetic_base codepaths
    return inner_metadata.mutated_inp_runtime_indices

assert len(fw_metadata.subclass_inp_meta) == len(fw_metadata.input_info)
for outer_idx, inp_meta in enumerate(fw_metadata.subclass_inp_meta):
    if isinstance(inp_meta, int):
        assert outer_idx < len(fw_metadata.input_info)
        if inner_metadata is not None:
            assert inner_idx < len(inner_metadata.input_info)
            assert (
                inner_metadata.input_info[inner_idx]
                == fw_metadata.input_info[outer_idx]
            )
        updated_input_info.append(fw_metadata.input_info[outer_idx])
        inner_idx += 1
    else:
        for _ in range(inp_meta.arg_count):
            updated_input_info.append(fw_metadata.input_info[outer_idx])
            inner_idx += 1

if inner_metadata is not None:
    assert len(inner_metadata.input_info) == len(updated_input_info)

# Return indices of mutated inputs that are flagged to be handled outside the graph
return [
    i
    for i, inp in enumerate(updated_input_info)
    if inp.mutation_type == MutationType.MUTATED_OUT_GRAPH
]
```