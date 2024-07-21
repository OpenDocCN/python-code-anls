# `.\pytorch\torch\_functorch\_aot_autograd\traced_function_transforms.py`

```
"""
This module is responsible for transforming functions to be traced into a form
that is easier for the downstream infra (e.g. Autograd, FX, AOTAutograd analysis)
to handle.

It does so by:
1. functionalization (including RNG functionalization)
2. creating a joint graph when required
3. transforming mutations into extra outputs
4. dispatching subclasses
"""

import warnings                                     # 导入警告模块
from contextlib import nullcontext                  # 导入上下文管理器 nullcontext
from functools import wraps                         # 导入 wraps 装饰器
from typing import Any, Callable, List, Tuple, Union # 导入类型提示相关的类
from unittest.mock import patch                    # 导入 patch 函数用于模拟测试

import torch                                        # 导入 PyTorch
import torch.fx.traceback as fx_traceback           # 导入 PyTorch FX 的 traceback 模块
import torch.utils._pytree as pytree                # 导入 PyTorch 的 _pytree 模块
from torch import Tensor                            # 导入 PyTorch 的 Tensor 类
from torch._decomp.decompositions_for_rng import PhiloxStateTracker   # 导入 PhiloxStateTracker 类
from torch._guards import detect_fake_mode          # 导入 detect_fake_mode 函数
from torch._prims_common import CUDARngStateHelper  # 导入 CUDARngStateHelper 类
from torch.fx.experimental.symbolic_shapes import ( # 导入符号形状相关函数和类
    definitely_false,
    PropagateUnbackedSymInts,
    sym_eq,
)
from torch.nn.utils import stateless               # 导入 stateless 函数

from .. import config                              # 导入相对路径的 config 模块
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata  # 导入元数据分析相关函数
from .functional_utils import (                    # 导入功能性工具函数
    from_fun,
    has_data_mutation,
    has_metadata_mutation,
    is_fun,
    sync_functional_tensor,
    to_fun,
    was_inductor_storage_resized,
)
from .logging_utils import setup_stacktrace_preservation_hooks  # 导入设置堆栈保留钩子的函数
from .schemas import (                             # 导入模式相关类和类型
    AOTConfig,
    MutationType,
    OutputType,
    SubclassMeta,
    SubclassTracingInfo,
    ViewAndMutationMeta,
)
from .subclass_utils import (                      # 导入子类相关的实用函数
    create_subclass_meta,
    requires_subclass_dispatch,
    unwrap_tensor_subclasses,
    wrap_tensor_subclasses_maybe_joint,
)
from .utils import maybe_to_fresh_input            # 导入 maybe_to_fresh_input 函数


# This function returns a new function that returns mutated inputs as outputs.
# if keep_data_input_mutations is set, then we assume that data-only mutations
# will be left in the graph, and we only return metadata-mutated inputs as outputs.
def fn_input_mutations_to_outputs(
    fn: Callable,
    meta: ViewAndMutationMeta,
    keep_data_input_mutations: bool,
) -> Any:
    @wraps(fn)
    def inner_fn(*args):
        outs = fn(*args)  # 调用传入的函数 fn，并获取其返回值
        assert len(meta.output_info) == len(outs)  # 断言输出信息的长度与实际输出的长度相等
        # The compiled fw will return mutated input tensors, *including* metadata-only mutation.
        # However, if keep_data_input_mutations is set, the compiled fw only needs to return metadata-mutated inputs.
        # (because data-only input mutations are handled directly in the compiled graph)
        # 编译后的前向传播将返回变异的输入张量，包括仅元数据变异。
        # 然而，如果设置了 keep_data_input_mutations，编译后的前向传播只需要返回元数据变异的输入。
        mutated_inputs_to_return = [
            x for (i, x) in enumerate(args) if i in meta.mutated_inp_runtime_indices
        ]
        return *mutated_inputs_to_return, *outs  # 返回变异的输入和所有的输出

    return inner_fn  # 返回经装饰的内部函数


# This function takes in a fn with external aliasing and mutation,
# and returns a new fn with no external aliasing and mutation,
# as needed for autograd.
# The main transformations are:
# - Return mutated inputs as extra outputs
# - Clone mutated inputs that require gradients,
# 函数 fn_prepped_for_autograd 将给定的函数 fn 包装，以便与 autograd 兼容。
# 它返回一个函数 inner_fn，该函数需要传入预变异的输入以便 autograd.grad 使用。
# 内部函数 inner_fn 的返回值包括：
# (1) 更新后的输出
# (2) 长度为 new_fn_outputs 的布尔掩码，用于指示 autograd.grad 哪些输出应该获取梯度。
def fn_prepped_for_autograd(
    fn: Callable,
    meta: ViewAndMutationMeta,
) -> Any:
    @wraps(fn)
    return inner_fn


# 给定一个函数 fn，计算其联合。
# 注意：fn 需要满足以下行为：
# (1) fn() 需要返回一个元组 (outs, mask)，其中 `mask` 告诉我们哪些输出应该有梯度。
#     我们不能自动知道这个信息，因为我们不希望盲目地为每个需要梯度的输出计算梯度。
#     具体来说，与输入别名的输出将不参与反向传播并获得梯度。
# (2) fn() 不能修改任何需要梯度的输入。
#     否则，当我们计算 autograd.grad() 时，将不会考虑这些输入的变化
#     （处理方式是确保通常会被变异的任何输入首先被克隆）。
def create_joint(fn: Callable, *, aot_config: AOTConfig) -> Any:
    # 定义一个内部函数 inner_fn，接受两个参数列表 primals 和 tangents
    def inner_fn(primals: List[Any], tangents: List[Any]):
        # 调用外部函数 fn 处理 primals，得到输出 outs 和 tangent_mask
        outs, tangent_mask = fn(*primals)
        # 断言 tangent_mask 的长度与 outs 的长度相等
        assert len(tangent_mask) == len(outs)
        
        # 从 outs 中筛选出需要计算梯度的输出
        outs_to_grad = [
            o for needs_tangent, o in zip(tangent_mask, outs) if needs_tangent
        ]
        # 断言需要计算梯度的输出个数与 tangents 的长度相等
        assert len(outs_to_grad) == len(tangents)

        # 获取需要计算梯度的输入 primals
        grad_primals = []
        inputs_needs_grads = []
        # 遍历 primals，判断是否是需要计算梯度的 Tensor 对象，并将结果存入 inputs_needs_grads
        # 注意这里不使用 primals 本身，避免在 autograd.grad() 中传递变异的输入
        for p in primals:
            is_grad_tensor = isinstance(p, Tensor) and p.requires_grad
            inputs_needs_grads.append(is_grad_tensor)
            if is_grad_tensor:
                grad_primals.append(p)

        # 获取需要计算梯度的输出 needed_outs 和对应的 tangents needed_tangents
        needed_outs = []
        needed_tangents = []
        for out, tangent in zip(outs_to_grad, tangents):
            # 判断输出 out 是否是需要计算梯度的 Tensor 对象
            if isinstance(out, Tensor) and out.requires_grad:
                # 处理形状不匹配的情况，确保 out 和 tangent 的形状一致
                needed_outs.append(
                    out
                    if not definitely_false(sym_eq(out.shape, tangent.shape))
                    else out.view(tangent.shape)
                )
                needed_tangents.append(tangent)

        # 设置用于保留堆栈跟踪的钩子函数
        setup_stacktrace_preservation_hooks([out.grad_fn for out in needed_outs])

        # 如果配置中启用了 functionalize_rng_ops，则标记反向传播的开始
        if config.functionalize_rng_ops:
            PhiloxStateTracker.mark_beginning_of_backward()

        backward_out: Tuple[Tensor, ...] = tuple()
        # 执行反向传播
        if grad_primals:
            with fx_traceback.preserve_node_meta():
                # 对于全图导出，假设不需要 tangents 的情况下始终导出一个联合图
                if aot_config.no_tangents:
                    assert len(needed_tangents) == 1 and needed_tangents[0].numel() == 1
                    # 计算梯度，不使用 tangents
                    backward_out = torch.autograd.grad(
                        needed_outs,
                        grad_primals,
                        allow_unused=True,
                    )
                else:
                    # 计算梯度，使用 tangents
                    backward_out = torch.autograd.grad(
                        needed_outs,
                        grad_primals,
                        grad_outputs=needed_tangents,
                        allow_unused=True,
                    )
        
        # 将 backward_out 转换为迭代器
        backward_out_iter = iter(backward_out)
        # 返回 outs 和按需计算梯度的结果列表
        return outs, [
            next(backward_out_iter) if i else None for i in inputs_needs_grads
        ]
    # 定义一个函数 inner_fn_with_anomaly，接受可变数量的参数
    def inner_fn_with_anomaly(*args):
        # 保留当前的 FX 调用堆栈信息，并忽略警告
        with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
            # 过滤特定警告消息："Anomaly Detection has been enabled."
            warnings.filterwarnings("ignore", "Anomaly Detection has been enabled.")
            # 启用 PyTorch 的异常检测，不检查 NaN 值
            with torch.autograd.detect_anomaly(check_nan=False):
                # 调用内部函数 inner_fn，并返回其结果
                return inner_fn(*args)

    # 返回函数 inner_fn_with_anomaly 作为结果
    return inner_fn_with_anomaly
# 创建一个包装器函数，将 rng 操作功能化，改变了联合图的调用约定。
# 在运行时，我们传递当前的种子和偏移量，这对用户是隐藏的。
def create_functionalized_rng_ops_wrapper(func, args, trace_joint=True) -> Any:
    # 检测是否处于模拟模式，如果是，则返回上下文管理器；否则返回空上下文管理器。
    fake_mode = detect_fake_mode()
    if fake_mode is None:
        fake_mode = nullcontext()

    # 覆盖获取 RNG 状态的函数，返回当前状态作为张量。
    def override_get_rng_state(device: Union[int, str, torch.device] = "cuda"):
        out = PhiloxStateTracker.get_state_as_tensor()
        return out

    # 覆盖设置 RNG 状态的函数，从张量设置状态。
    def override_set_rng_state(x, device: Union[int, str, torch.device] = "cuda"):
        PhiloxStateTracker.set_state_from_tensor(x)

    # 追加 RNG 偏移量的函数。
    def append_rng_offsets(args):
        if trace_joint:
            # 如果追踪联合图，则添加新的前向和后向 RNG 偏移量。
            return (
                (*args[0], PhiloxStateTracker.get_updated_fwd_offset()),
                (*args[1], PhiloxStateTracker.get_updated_bwd_offset()),
            )
        else:
            # 如果不追踪联合图，则添加新的前向 RNG 偏移量。
            return (*args, PhiloxStateTracker.get_updated_fwd_offset())

    # 追踪联合图的函数，处理原始值和切线，并添加种子和偏移量参数。
    def traced_joint(
        primals, tangents, fwd_seed, fwd_base_offset, bwd_seed, bwd_base_offset
    ):
        with patch("torch.cuda.get_rng_state", override_get_rng_state), patch(
            "torch.cuda.set_rng_state", override_set_rng_state
        ):
            return append_rng_offsets(func(primals, tangents))

    # 追踪前向传播的函数，处理原始值、种子和偏移量参数。
    def traced_forward(*primals_fwd_seed_fwd_base_offset):
        # 函数签名为 (*primals, seed, offset)，删除最后两个参数。
        with patch("torch.cuda.get_rng_state", override_get_rng_state), patch(
            "torch.cuda.set_rng_state", override_set_rng_state
        ):
            return append_rng_offsets(func(*primals_fwd_seed_fwd_base_offset[:-2]))

    # 如果追踪联合图，则获取当前种子和偏移量以设置追踪。
    if trace_joint:
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        bwd_seed, bwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        # 记录前向和后向状态，然后返回追踪联合图函数和其参数。
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        PhiloxStateTracker.record_state(bwd_seed, bwd_base_offset, "backward")
        return traced_joint, (
            *args,
            fwd_seed,
            fwd_base_offset,
            bwd_seed,
            bwd_base_offset,
        )
    else:
        # 如果不追踪联合图，则只获取前向种子和偏移量，记录前向状态。
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        # 返回追踪前向传播函数和其参数。
        return traced_forward, (*args, fwd_seed, fwd_base_offset)
# 创建功能化函数，用于跟踪使用make_fx()生成的最终函数，同时在aot_dispatch_autograd和aot_dispatch_base中使用。
# 前提条件：
# - fn对应于用户的fw函数
# - fn的参数已被展开，重复的参数已处理
# - 返回的函数中，“primals”参数 *包括* 合成的基础部分
# 此函数完成了对输入函数的功能化，并在函数末尾执行copy_()调用（如果设置了keep_input_mutations）。
# 返回的函数签名可以是：
# (1) "traced_fn(primals: List[Any])"，如果trace_joint为False
# (2) "traced_fn(primals: List[Any], tangents: List[Any])"，如果trace_joint为True
# 返回一个新的（功能化的）函数，以及更新后的调用它所需的参数。
def create_functionalized_fn(
    fn,
    args,
    *,
    meta: ViewAndMutationMeta,
    aot_config: AOTConfig,
    trace_joint: bool,
) -> Any:
    @wraps(fn)
    # 定义一个辅助函数，根据trace_joint是否为True来选择不同的函数_helper
    def joint_helper(primals, tangents):
        return _functionalized_f_helper(primals, tangents)

    # 根据trace_joint是否为True，选择使用joint_helper还是_functionalized_f_helper
    helper = joint_helper if trace_joint else _functionalized_f_helper
    
    # 如果config.functionalize_rng_ops为True，则设置rng操作的功能化包装器
    if config.functionalize_rng_ops:
        helper, args = create_functionalized_rng_ops_wrapper(helper, args, trace_joint)

    # 此外，作为输入额外传入tokens
    # 参见注释 [Side-Effectful Tokens in AOTAutograd]
    additional_token_inputs = [torch.tensor([])] * len(meta.tokens)
    if trace_joint:
        # 如果trace_joint为True，则更新args以包含additional_token_inputs和args[0]
        args = ([*additional_token_inputs, *args[0]], *args[1:])
    else:
        # 如果trace_joint为False，则更新args以包含additional_token_inputs和args
        args = [*additional_token_inputs, *args]

    # 返回helper函数和更新后的args
    return helper, args


# 给定一个操作Subclass -> Subclass的函数，返回一个操作Tensor -> Tensor的函数
# 同时返回：
# - 传递到该函数的新参数集（现在已经消除了tensor子类）
# - 此dense -> dense函数的更新ViewAndMutationMeta
# 其他重要参数包括：
# - flat_fn_maybe_joint: 当is_joint_structure=True时，这是联合fw-bw函数；
#                        当is_joint_structure=False时，这只是前向函数。
# - fw_only: 这始终是仅前向函数。
# 为什么我们需要这个？我们需要在新的dense -> dense函数上收集更新的ViewAndMutationMeta。
# 特别是，我们需要这样做以告知分区器有多少dense前向输出。
def aot_dispatch_subclass(
    flat_fn_maybe_joint,
    args: List[Any],  # 参数 args 是一个列表，可以包含任何类型的元素
    *,  # 这个逗号后面的参数是命名关键字参数
    is_joint_structure: bool,  # is_joint_structure 是一个布尔类型的参数，用于表示是否联合结构
    meta: ViewAndMutationMeta,  # meta 是一个 ViewAndMutationMeta 类型的参数，可能是用于视图和变异的元信息
    fw_only: Callable,  # fw_only 是一个可调用对象（函数），作为参数传递
# 如果不需要通过子类进行调度，则跳过逻辑
req_subclass_dispatch = requires_subclass_dispatch(args, meta)
if not req_subclass_dispatch:
    # 如果不需要子类调度，则直接返回一个SubclassTracingInfo对象
    return SubclassTracingInfo(
        plain_tensor_trace_fn=flat_fn_maybe_joint,
        plain_tensor_args=args,
        maybe_subclass_meta=None,
    )

# TODO: 添加子类保护逻辑（稍后处理）。

# 在这里发生了什么？我们需要计算关于联合（grad_inputs）输出的子类元数据。
# 麻烦的是：我们在追踪联合过程中不知道梯度输入元数据，因此我们稍后设置它，当我们在追踪联合（见下面的inner_fn()）时。
# 另一个选择是直接在联合上运行我们的run_functionalized_fw_and_collect_metadata()函数，但这会增加编译时间（通过联合增加另一个传递）。
subclass_meta = SubclassMeta()

def inner_fn(fn, args, *, use_trace_joint: bool):
    # 步骤1：如果需要，将张量输入包装为子类
    all_args = wrap_tensor_subclasses_maybe_joint(
        args, is_joint_structure=use_trace_joint, meta=meta
    )

    # 步骤2：调用内部函数，使用（可能是子类）的输入
    wrapped_outs = fn(*all_args)

    if use_trace_joint:
        # 见注释：[关于grad_inputs计算子类元数据]
        # 如果我们在追踪联合，则还需要在grad_inputs上存储子类信息。
        nonlocal subclass_meta
        assert isinstance(wrapped_outs, tuple) and len(wrapped_outs) == 2
        # 不需要前向输出，因为我们已经在它们上面有子类元数据了
        grad_inputs = wrapped_outs[1]
        subclass_meta.grad_input_metas = create_subclass_meta(grad_inputs)

    # 步骤3：将任何子类输出解包回密集张量
    unwrapped_outs = unwrap_tensor_subclasses(
        wrapped_outs, is_joint_structure=use_trace_joint
    )
    return unwrapped_outs

def joint_fn(primals, tangents):
    return inner_fn(flat_fn_maybe_joint, (primals, tangents), use_trace_joint=True)

def fw_fn(*primals):
    return inner_fn(flat_fn_maybe_joint, primals, use_trace_joint=False)

def metadata_fn(*primals):
    return inner_fn(fw_only, primals, use_trace_joint=False)

# 解包参数中的任何子类张量
args_unwrapped = unwrap_tensor_subclasses(
    args, is_joint_structure=is_joint_structure
)

if is_joint_structure:
    primals_unwrapped = args_unwrapped[0]
    fn_to_trace = joint_fn
else:
    primals_unwrapped = args_unwrapped
    fn_to_trace = fw_fn

# 注释：[子类的分区处理，第1部分]
# 分区处理器的工作方式是：
# (1) 我们传递一个包含联合前向/后向的单个图，其中图输出的数量对应于fw_outputs的数量 + grad_inputs的数量
# (2) 分区器接受一个参数，num_fwd_outputs，
    # 运行功能化的前向图并收集元数据
    meta_updated = run_functionalized_fw_and_collect_metadata(
        metadata_fn,
        keep_input_mutations=meta.keep_input_mutations,
        is_train=meta.is_train,
    )(*primals_unwrapped)

    # 更新子类的前向元数据
    subclass_meta.fw_metadata = meta_updated

    # 返回包含子类跟踪信息的对象
    return SubclassTracingInfo(
        plain_tensor_trace_fn=fn_to_trace,
        plain_tensor_args=args_unwrapped,
        maybe_subclass_meta=subclass_meta,
    )
# 创建一个可调用的函数 functional_call，它封装了 mod 参数，并接受任意位置和关键字参数
def create_functional_call(mod, params_spec, params_len, store_orig_mod=False):
    # Redundant with dynamo, but worth having in case this gets invoked elsewhere.
    # https://github.com/pytorch/pytorch/issues/103569

    # 定义一个内部函数 functional_call，用于执行模型的调用和处理输出
    def functional_call(*args, **kwargs):
        # 使用 stateless._reparametrize_module 上下文管理器重参数化 mod 模块，
        # 并将 args 的前 params_len 个参数展开为 params_spec 所指定的形式
        with stateless._reparametrize_module(
            mod, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
            # 如果 mod 是 torch.fx.GraphModule 类型
            if isinstance(mod, torch.fx.GraphModule):
                # 使用 fx_traceback.preserve_node_meta() 保存节点元数据，
                # 并使用 warnings.catch_warnings() 捕获警告信息
                with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
                    # 忽略特定警告信息 "Anomaly Detection has been enabled."
                    warnings.filterwarnings(
                        "ignore", "Anomaly Detection has been enabled."
                    )
                    # 启用 torch.autograd.detect_anomaly() 检测异常，
                    # 并通过 detect_fake_mode().epoch += 1 提示模型处于推理模式
                    out = PropagateUnbackedSymInts(mod).run(
                        *args[params_len:], **kwargs
                    )
            else:
                # 否则直接调用 mod 模块进行计算
                out = mod(*args[params_len:], **kwargs)

        # 如果输出不是元组或列表，则抛出 RuntimeError 异常
        if not isinstance(out, (tuple, list)):
            raise RuntimeError(
                "Graph output must be a tuple(). This is so that we can avoid "
                "pytree processing of the outputs. Please change the module to "
                "have tuple outputs or use aot_module instead."
            )
        # 返回计算结果 out
        return out

    # Note [Preserving the nn module stack metadata during export non-strict mode]
    # 在非严格导出模式下，这条路径目前仅用于保留 nn 模块堆栈元数据，
    # 我们无法依赖 dynamo 来在捕获的图中保留 nn 模块堆栈元数据。
    # 相反，我们将原始用户 nn 模块存储在这里，并依赖 make_fx 来获取此存储的模块，
    # 并用它来跟踪 nn 模块堆栈元数据
    if store_orig_mod and not hasattr(functional_call, "_orig_mod"):
        # 如果 store_orig_mod 为真且 functional_call 没有属性 _orig_mod
        # 则将 mod 存储在 functional_call 的 _orig_mod 属性中
        functional_call._orig_mod = mod  # type: ignore[attr-defined]

    # 返回定义的内部函数 functional_call
    return functional_call
```