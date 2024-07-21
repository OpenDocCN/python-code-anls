# `.\pytorch\torch\_functorch\_aot_autograd\functional_utils.py`

```
"""
This file contains utilities related to functionalization in AOTAutograd:
1. converting to/from functional tensors
2. detecting Tensor mutations - both metadata and Tensor value
3. regenerating/replaying views from their base
4. checking if a graph is functional i.e. whether it contains any mutation ops
"""

# Import necessary modules and classes
from typing import Optional
import torch
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    transform_subclass,
)
from .. import config

# Initialize logger specific to AOTAutograd functionalization
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")


# Function to convert a tensor to its functional form
def to_fun(t):
    if isinstance(t, Tensor):
        # Check if the tensor is a subclass wrapper
        if is_traceable_wrapper_subclass(t):
            # See Note [Functionalization always runs last]
            # Recursively transform nested subclass wrappers to functional form
            out = transform_subclass(t, lambda _, inner_t: to_fun(inner_t))
            torch._mirror_autograd_meta_to(t, out)  # type: ignore[attr-defined]
            return out
        else:
            # Convert the tensor to a FunctionalTensor
            return FunctionalTensor.to_functional(t)
    else:
        return t


# Function to synchronize a functional tensor
def sync_functional_tensor(t):
    if is_traceable_wrapper_subclass(t):
        # Flatten the tensor and context attributes
        attrs, ctx = t.__tensor_flatten__()  # type: ignore[attr-defined]
        for attr in attrs:
            # Recursively synchronize nested attributes
            sync_functional_tensor(getattr(t, attr))
    else:
        # Synchronize the tensor
        torch._sync(t)


# Function to convert a functional tensor back to its original form
def from_fun(t):
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        # See Note [Functionalization always runs last]
        # Recursively transform nested subclass wrappers back to original form
        out = transform_subclass(t, lambda _, inner_t: from_fun(inner_t))
        torch._mirror_autograd_meta_to(t, out)  # type: ignore[attr-defined]
        return out

    if not isinstance(t, FunctionalTensor):
        # Quick sanity check assert
        if isinstance(t, torch.Tensor):
            assert not torch._is_functional_tensor(t)  # type: ignore[attr-defined]
        return t
    
    # Sync the functional tensor and convert it back to its element form
    sync_functional_tensor(t)
    return torch._from_functional_tensor(t.elem)


# Placeholder function definition
def is_fun(t):
    # TODO: Implement function to check if a tensor is functional
    pass
    # 检查变量 t 是否是 Tensor 类型，并且是可追踪包装器的子类
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        # 查看注释 [Functionalization always runs last]
        # 这意味着如果我们想要"功能化"一个子类，我们需要确保功能化的包装器
        # 放在最后面。
        # 在这里递归调用，以支持嵌套的包装器子类
        t_attrs, _ = t.__tensor_flatten__()  # type: ignore[attr-defined]
        # 获取所有 t_attrs 中的属性，并存入 t_inners 列表中
        t_inners = [getattr(t, attr) for attr in t_attrs]
        # 检查 t_inners 中是否存在至少一个是函数
        any_fun = any(is_fun(x) for x in t_inners)
        # 检查 t_inners 中是否所有元素都是函数
        all_fun = all(is_fun(x) for x in t_inners)
        # 断言任意函数的存在与所有元素都是函数的结果应该相等
        assert any_fun == all_fun
        # 返回是否存在任意函数的布尔值
        return any_fun

    # 如果 t 不是 Tensor 类型或者不是可追踪包装器的子类，则返回 t 是否是 FunctionalTensor 的实例
    return isinstance(t, FunctionalTensor)
# t here is either
# (1) A FunctionalTensor(_to_functional_tensor(FakeTensor))
# (2) A traceable tensor subclass that holds a FunctionalTensor
# (3) Not a tensor
def has_data_mutation(t):
    if is_traceable_wrapper_subclass(t):  # 如果 t 是可追踪的张量子类
        attrs, _ = t.__tensor_flatten__()  # 获取 t 的属性列表
        # 如果任何内部元素被更新，则张量子类已更新
        return any(has_data_mutation(getattr(t, attr)) for attr in attrs)
    else:
        if isinstance(t, torch.Tensor):  # 如果 t 是 torch.Tensor 类型
            assert isinstance(t, FunctionalTensor)  # 确保 t 是 FunctionalTensor 类型
            return torch._functionalize_has_data_mutation(t.elem)  # 调用 torch._functionalize_has_data_mutation 检查数据是否被突变
        return False  # 否则返回 False


def are_all_mutations_hidden_from_autograd(t):
    if is_traceable_wrapper_subclass(t):  # 如果 t 是可追踪的张量子类
        attrs, _ = t.__tensor_flatten__()  # 获取 t 的属性列表
        # 如果所有内部元素都是从自动求导中隐藏的突变，则它是一个从自动求导中隐藏的突变
        return all(
            are_all_mutations_hidden_from_autograd(getattr(t, attr)) for attr in attrs
        )
    elif isinstance(t, torch.Tensor):  # 如果 t 是 torch.Tensor 类型
        assert isinstance(t, FunctionalTensor)  # 确保 t 是 FunctionalTensor 类型
        return torch._functionalize_are_all_mutations_hidden_from_autograd(t.elem)  # 调用 torch._functionalize_are_all_mutations_hidden_from_autograd 检查是否所有突变都从自动求导中隐藏
    else:
        return False  # 否则返回 False


def are_all_mutations_under_no_grad_or_inference_mode(t):
    if is_traceable_wrapper_subclass(t):  # 如果 t 是可追踪的张量子类
        attrs, _ = t.__tensor_flatten__()  # 获取 t 的属性列表
        return all(
            are_all_mutations_under_no_grad_or_inference_mode(getattr(t, attr))
            for attr in attrs
        )
    else:
        assert isinstance(t, FunctionalTensor)  # 确保 t 是 FunctionalTensor 类型
        return torch._functionalize_are_all_mutations_under_no_grad_or_inference_mode(
            t.elem
        )


def was_inductor_storage_resized(t):
    if is_traceable_wrapper_subclass(t):  # 如果 t 是可追踪的张量子类
        attrs, _ = t.__tensor_flatten__()  # 获取 t 的属性列表
        if any(was_inductor_storage_resized(getattr(t, attr)) for attr in attrs):
            raise RuntimeError(
                f"storage resizing is not supported on tensor subclass: {type(t)}"
            )
    elif not isinstance(t, torch.Tensor):  # 如果 t 不是 torch.Tensor 类型
        return False  # 返回 False
    else:
        assert isinstance(t, FunctionalTensor)  # 确保 t 是 FunctionalTensor 类型
        return torch._functionalize_was_inductor_storage_resized(t.elem)


# f_arg here is either
# (1) A FunctionalTensor(_to_functional_tensor(FakeTensor))
# (2) A traceable tensor subclass that holds a FunctionalTensor
# (3) Not a tensor
# Assumption: arg promises to be the "original" tensor wrapped by f_arg
# Note: "storage mutations" coming from set_() are a type of metadata mutation. So:
# - check_only_storage_mutation=True: only return true if there was a storage mutation
# - check_only_storage_mutation=Flse: return true if there was any metadata mutation (including a storage mutation)
def has_metadata_mutation(f_arg, arg, *, check_only_storage_mutation: bool):
    # 检查是否为可追踪包装器的子类
    if is_traceable_wrapper_subclass(f_arg):
        # 调用 __tensor_flatten__ 方法获取属性列表 attrs 和未使用的变量 _
        attrs, _ = f_arg.__tensor_flatten__()
        # 如果张量的任何内部元素被更新，则张量子类已更新
        # 收集 f_arg 中每个属性的张量列表
        f_inner_ts = [getattr(f_arg, attr) for attr in attrs]
        # 收集 arg 中每个属性的张量列表
        inner_ts = [getattr(arg, attr) for attr in attrs]
        # 返回任何一对 f_inner_ts 和 inner_ts 中的元素满足 metadata 变异的检查
        return any(
            has_metadata_mutation(
                f_inner_t,
                inner_t,
                check_only_storage_mutation=check_only_storage_mutation,
            )
            for f_inner_t, inner_t in zip(f_inner_ts, inner_ts)
        )
        else:
            # 如果 f_arg 不是 torch.Tensor 类型，则断言 arg 也不是 torch.Tensor 类型
            if not isinstance(f_arg, torch.Tensor):
                assert not isinstance(arg, torch.Tensor)
                return False
            # 断言 f_arg 是 FunctionalTensor 类型，arg 是 FakeTensor 类型
            assert isinstance(f_arg, FunctionalTensor)
            assert isinstance(arg, FakeTensor)

            # 根据 FunctionalTensor 的元素创建新的 torch.Tensor
            arg_after = torch._from_functional_tensor(f_arg.elem)
            # 检查当前张量是否至少经历过一次 set_() 调用
            maybe_storage_changed = torch._functionalize_was_storage_changed(f_arg.elem)  # type: ignore[attr-defined]
            # 然而，多次 set_() 调用可以取消。因此，我们还检查张量的存储是否发生了变化。
            # 注意：如果一个输入经历了两次互相抵消的 set_() 调用，并且发生了数据变异，
            # 我们悲观地认为在这里需要 set_() 调用。理论上我们可以修复这个问题，
            # 但是希望这种情况不会在用户代码中出现，并且对于 fsdp 而言也不是必需的。
            if is_sparse_any(arg):
                # TODO: 添加对稀疏张量的支持到 functionalization
                same_storages = False
            else:
                # 比较两个张量的存储是否相同
                same_storages = StorageWeakRef(arg.untyped_storage()) == StorageWeakRef(
                    arg_after.untyped_storage()
                )
            # 检查是否存在存储元数据的变异
            has_storage_metadata_mutation = maybe_storage_changed and not same_storages
            if check_only_storage_mutation:
                return has_storage_metadata_mutation

            # 如果存在存储元数据的变异，则返回 True
            if has_storage_metadata_mutation:
                return True

            # 检查张量是否至少经历过一次元数据的变异
            maybe_metadata_mutated = torch._functionalize_has_metadata_mutation(f_arg.elem)  # type: ignore[attr-defined]
            # 如果当前张量没有经历过元数据的变异，则返回 False
            if not maybe_metadata_mutated:
                return False

            # 然而，多次元数据变异可以取消。因此，我们还检查张量的具体大小、步长和存储偏移是否有变化。
            same_sizes = arg.shape == arg_after.shape
            same_strides = arg.stride() == arg_after.stride()
            same_offsets = arg.storage_offset() == arg_after.storage_offset()
            # 如果张量的具体大小、步长和存储偏移没有完全相同，则认为发生了元数据变异
            has_metadata_mutation_ = maybe_metadata_mutated and not (
                same_sizes and same_strides and same_offsets
            )
            # 如果存在元数据的变异，则返回 True；否则返回 False
            return has_metadata_mutation_
# 从给定的基础张量生成别名张量，并根据需求调整梯度属性
def gen_alias_from_base(
    aliased_base_tensor,  # 给定的基础张量
    target_meta_tensor,   # 目标元数据张量
    target_requires_grad,  # 目标张量是否需要梯度
    target_functional_tensor: Optional[FunctionalTensorMetadataEq] = None,  # 可选的功能张量元数据
):
    # 调整输出张量的requires_grad属性，依赖于以下条件：
    # (i) 重建的输出(out)是否来自需要梯度的张量；
    # 和 (ii) 具体返回的输出是否需要梯度。
    def patch_requires_grad(out):
        if aliased_base_tensor.requires_grad and not target_requires_grad:
            out = out.detach()  # 如果基础张量需要梯度但目标不需要，则将输出张量分离
        elif not aliased_base_tensor.requires_grad and target_requires_grad:
            out.requires_grad_(True)  # 如果基础张量不需要梯度但目标需要，则设置输出张量需要梯度
        return out

    # 如果提供了目标功能张量，用于回放视图操作。
    #
    # 总结来说，我们利用 FunctionalTensorWrapper 保存了应用于自身的视图函数
    # （在功能化期间收集），以便在基础张量上重新播放这些（视图函数）。
    if (
        config.view_replay_for_aliased_outputs  # 配置中指示允许为别名输出重放视图操作
        and target_functional_tensor is not None  # 目标功能张量不为空
        and not torch._functionalize_is_symbolic(target_functional_tensor.tensor)  # 目标功能张量不是符号化的
    ):
        functional_tensor = target_functional_tensor.tensor

        # 应用视图元数据序列到基础张量，以重新构造输出
        out = torch._functionalize_apply_view_metas(
            functional_tensor, aliased_base_tensor
        )
        # 如果成功重新应用了 ViewMeta 序列，则应该没有更多问题。我们只需检查是否达到了目标形状并修正 requires_grad 标志。
        assert out.shape == target_meta_tensor.shape, (
            "incorrect out shape after application of ViewMeta sequence: "
            f"{tuple(out.shape)} (actual) vs {tuple(target_meta_tensor.shape)} (expected)"
        )
        return patch_requires_grad(out)  # 返回调整后的输出张量

    # 如果可能，尝试进行视图重放。
    # 如果无法进行视图重放，则回退到 .as_strided() 方法。
    # 如果目标元数据张量的基张量不为 None
    if target_meta_tensor._base is not None:
        # 我们希望基张量的重塑视图可能与视图的原始基张量形状不同。
        b = target_meta_tensor._base
        abt = aliased_base_tensor
        # 如果 aliased_base_tensor 和 b 不相等，或者它们的 size、stride、storage_offset 有任何一个不同，
        # 则需要调用 as_strided 进行重塑；由于 as_strided 的反向操作实现较差且较慢，应避免不必要的调用。
        if abt is not b and (
            abt.size() != b.size()
            or abt.stride() != b.stride()
            or abt.storage_offset() != b.storage_offset()
        ):
            # 使用 as_strided 方法重塑 aliased_base_tensor
            reshaped_base_tensor = aliased_base_tensor.as_strided(
                b.size(), b.stride(), b.storage_offset()
            )
        else:
            # 否则保持 aliased_base_tensor 不变
            reshaped_base_tensor = aliased_base_tensor
        # 使用 _view_func 方法对重塑后的基张量进行视图操作
        out = target_meta_tensor._view_func(reshaped_base_tensor)
        # 如果 out 不为 None 且其形状与目标元数据张量的形状相同，
        # 则调用 patch_requires_grad 函数处理 out
        if out is not None and out.shape == target_meta_tensor.shape:
            return patch_requires_grad(out)

    # 获取目标元数据张量的尺寸、步幅和存储偏移量
    size = target_meta_tensor.size()
    stride = target_meta_tensor.stride()
    storage_offset = target_meta_tensor.storage_offset()
    
    # 如果 aliased_base_tensor 是复数类型且目标元数据张量不是复数类型，
    # 则将 aliased_base_tensor 视图转换为实数视图并重塑
    if aliased_base_tensor.is_complex() and not target_meta_tensor.is_complex():
        aliased_out = torch.view_as_real(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    # 如果 aliased_base_tensor 不是复数类型且目标元数据张量是复数类型，
    # 则将 aliased_base_tensor 视图转换为复数视图并重塑
    elif not aliased_base_tensor.is_complex() and target_meta_tensor.is_complex():
        aliased_out = torch.view_as_complex(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    else:
        # 否则直接使用 aliased_base_tensor 的 as_strided 方法进行重塑
        aliased_out = aliased_base_tensor.as_strided(size, stride, storage_offset)
    
    # 对于输出与输入有别名的情况，需要检查是否需要更新 requires_grad 属性
    aliased_out = patch_requires_grad(aliased_out)
    
    # 对于输出与输入有别名的情况，需要检查是否需要更新数据类型
    # as_strided() 是最通用的视图，但不支持跨数据类型的视图转换
    if aliased_out.dtype != target_meta_tensor.dtype:
        aliased_out = aliased_out.view(target_meta_tensor.dtype)
    
    # 返回重塑后的张量 aliased_out
    return aliased_out
# 检查两个张量的元数据是否相同
def has_same_metadata(t1, t2):
    return (
        definitely_true(sym_eq(t1.size(), t2.size()))  # 检查张量大小是否相同
        and definitely_true(sym_eq(t1.stride(), t2.stride()))  # 检查张量步幅是否相同
        and definitely_true(t1.storage_offset() == t2.storage_offset())  # 检查张量存储偏移是否相同
        and t1.is_conj() == t2.is_conj()  # 检查张量是否为共轭
        and t1.is_neg() == t2.is_neg()  # 检查张量是否为负数
    )


# 用于比较经过所有 ViewMeta 操作后的元数据是否相同的 FunctionalTensorWrapper 的包装器
class FunctionalTensorMetadataEq:
    def __init__(self, tensor: torch.Tensor) -> None:
        assert torch._is_functional_tensor(tensor)
        self.tensor = tensor

    def __eq__(self, other: object) -> bool:
        # 如果 other 是 None，则可能表示我们无法重新创建 FunctionalTensorMetadataEq 的情况之一，
        # 例如当调用 create_synthetic_base_metadata 更新视图元数据时。
        if other is None:
            return True

        # 如果 other 不是 FunctionalTensorMetadataEq 的实例，则返回 Not Implemented
        if not isinstance(other, FunctionalTensorMetadataEq):
            return NotImplemented

        # 比较两个 FunctionalTensorMetadataEq 实例中的张量元数据是否相同
        return has_same_metadata(self.tensor, other.tensor)


# 检查输入张量是否被更新的函数
# new_arg 和 arg 可能是以下情况之一：
# (1) 都是 FakeTensor
# (2) 都是持有 FakeTensor 的可追踪张量子类
# 先决条件：这两个参数是从运行功能化后得到的“旧”和“新”输入。
# 当我们运行功能化并将输入包装成 FunctionalTensors 时，我们可以通过检查内部张量是否变化来检测输入是否被修改。
def was_tensor_updated(arg, new_arg):
    if is_traceable_wrapper_subclass(arg):  # 如果 arg 是可追踪包装器子类
        assert is_traceable_wrapper_subclass(new_arg)  # 确保 new_arg 也是可追踪包装器子类
        attrs, _ = arg.__tensor_flatten__()  # 获取 arg 的张量扁平化属性
        new_attrs, _ = new_arg.__tensor_flatten__()  # 获取 new_arg 的张量扁平化属性
        assert attrs == new_attrs  # 断言两者的属性相同
        # 如果张量子类的任何内部元素被更新，则该张量子类已被更新
        return any(
            was_tensor_updated(getattr(arg, attr), getattr(new_arg, attr))
            for attr in attrs
        )
    else:
        return arg is not new_arg  # 直接比较 arg 和 new_arg 是否相同，如果不同则表示被更新了


# 检查输入张量是否被更新的函数
# new_arg 和 arg 可能是以下情况之一：
# (1) 都是 FakeTensor
# (2) 都是持有 FakeTensor 的可追踪张量子类
# 先决条件：这两个参数是从运行功能化后得到的“旧”和“新”输入。
# 当我们运行功能化并将输入包装成 FunctionalTensors 时，我们可以通过检查内部张量是否变化来检测输入是否被修改。
# 检查是否更新了张量元数据
def was_tensor_metadata_updated(arg, new_arg):
    # 如果是可追踪包装器的子类，进行以下操作
    if is_traceable_wrapper_subclass(arg):
        # 断言新旧参数都是可追踪包装器的子类
        assert is_traceable_wrapper_subclass(new_arg)
        # 获取旧参数的张量属性及其展平版本
        attrs, _ = arg.__tensor_flatten__()
        # 获取新参数的张量属性及其展平版本
        new_attrs, _ = new_arg.__tensor_flatten__()
        # 断言旧参数和新参数的张量属性应相等
        assert attrs == new_attrs
        # 如果任何内部元素更新，则张量子类已更新
        return any(
            was_tensor_metadata_updated(getattr(arg, attr), getattr(new_arg, attr))
            for attr in attrs
        )
    else:
        # 如果旧参数和新参数不相同且它们的存储弱引用也不同，则返回 True
        return arg is not new_arg and StorageWeakRef(
            arg.untyped_storage()
        ) == StorageWeakRef(new_arg.untyped_storage())


# 检查函数式图中的变异操作数量，并返回
def assert_functional_graph(fx_g: torch.fx.Graph) -> int:
    # 存放占位符节点的集合
    placeholders = set()
    # 记录变异操作的计数器
    mutation_count = 0
    # 遍历图中的每个节点
    for n in fx_g.nodes:
        # 如果节点是占位符
        if n.op == "placeholder":
            placeholders.add(n)
        # 如果节点目标是 torch._ops.OpOverload 的实例
        if isinstance(n.target, torch._ops.OpOverload):
            # 如果目标是 copy_ 或 set_ 操作
            if n.target in [
                torch.ops.aten.copy_.default,
                torch.ops.aten.set_.source_Tensor,
            ]:
                suffix = True
                # 只能将 copy_/set_ 应用到输入中，且只能应用一次，这是为了避免 XLA 测试失败的一个临时修补
                if "set_buffer_donor_" not in str(n.args[0]):
                    # 断言节点的第一个参数在占位符集合中
                    assert (
                        n.args[0] in placeholders
                    ), f"n={str(n)}, n.args[0]={str(n.args[0])}, placeholders={str(placeholders)}, graph={str(fx_g)}"
                    # 从占位符集合中移除该参数
                    placeholders.remove(n.args[0])
                # 增加变异操作计数器
                mutation_count += 1
            else:
                # 断言节点目标的模式不可变，用于检查图是否完全是函数式的
                assert (
                    not n.target._schema.is_mutable
                ), f"aot_autograd expected to have an entirely functional graph, but found {n.format_node()}"
    # 返回变异操作的总数
    return mutation_count


# 传播输入变异的堆栈跟踪信息
def propagate_input_mutation_stacktraces(fx_g: torch.fx.Graph) -> None:
    # 存放占位符节点的集合
    placeholders = set()
    # 遍历图中的每个节点
    for n in fx_g.nodes:
        # 如果节点的操作类型是“placeholder”，将其添加到placeholders集合中
        if n.op == "placeholder":
            placeholders.add(n)
        # 如果节点的目标是torch._ops.OpOverload类型
        if isinstance(n.target, torch._ops.OpOverload):
            # 如果目标是torch.ops.aten.copy_.default
            if n.target is torch.ops.aten.copy_.default:
                # 只能将copy_操作应用于输入，并且只能应用一次
                # 检查第一个参数是否包含"set_buffer_donor_"，如果没有则断言第一个参数在placeholders集合中
                if "set_buffer_donor_" not in str(n.args[0]):
                    assert (
                        n.args[0] in placeholders
                    ), f"n={str(n)}, n.args[0]={str(n.args[0])}, placeholders={str(placeholders)}, graph={str(fx_g)}"
                    # 从placeholders集合中移除第一个参数
                    placeholders.remove(n.args[0])
                # 获取复制来源节点
                copy_from_node = n.args[1]
                # 前提条件：每个节点的meta中都有一个"stack_trace"字段，
                # 但是copy_()节点没有（因为我们在功能化期间手动添加了它们）。
                # 在这里我们手动传播它。
                if "stack_trace" in copy_from_node.meta:
                    n.meta["stack_trace"] = copy_from_node.meta["stack_trace"]
def _check_if_mutation_can_be_in_graph(
    keep_input_mutations: bool,
    mutates_data,
    mutates_metadata,
    mutations_hidden_from_autograd,
    mutations_under_no_grad_or_inference_mode,
    mutates_storage_metadata,
    mutation_inductor_storage_resize,
    requires_grad,
):
    # 根据 keep_input_mutations 参数决定是否允许输入的变化在计算图中存在
    if keep_input_mutations:
        # 计算 in_graph 变量，条件是数据或者存储元数据或者存储调整变异器中的一种，并且满足下列条件之一：
        # 不变更元数据且不需要梯度，或者变异对自动求导隐藏，或者处于无梯度或推理模式下的变异
        in_graph = (
            mutates_data or mutates_storage_metadata or mutation_inductor_storage_resize
        ) and (
            (not mutates_metadata and not requires_grad)
            or mutations_hidden_from_autograd
            or mutations_under_no_grad_or_inference_mode
        )
    else:
        # 如果不保留输入变化，直接将 in_graph 设置为 False
        in_graph = False

    # 查看注释 [set_() Input Mutations in AOTAutograd]
    # 如果存在 `set_()` 操作或者 `resize_()` 操作，要求所有变化必须在无梯度状态下，以便可以在运行时在图中使用该操作
    if mutation_inductor_storage_resize or mutates_storage_metadata:
        op_name = "resize_" if mutation_inductor_storage_resize else "set_"
        assert in_graph, f"""\
遇到了 {op_name} 操作在图输入上，但是输入有其他我们无法保留在图中的变异。目前不支持此操作。当前状态：
  keep_input_mutations={keep_input_mutations}
  mutates_data={mutates_data}
  mutates_metadata={mutates_metadata}
  mutations_hidden_from_autograd={mutations_hidden_from_autograd}
  mutations_under_no_grad_or_inference_mode={mutations_under_no_grad_or_inference_mode}
  mutation_inductor_storage_resize={mutation_inductor_storage_resize}
  requires_grad={requires_grad}"""
    
    # 返回计算得到的 in_graph 变量
    return in_graph
```