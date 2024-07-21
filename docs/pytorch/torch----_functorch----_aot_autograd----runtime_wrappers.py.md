# `.\pytorch\torch\_functorch\_aot_autograd\runtime_wrappers.py`

```
"""
# mypy: allow-untyped-defs
"""
"""
This module defines runtime wrappers, which, based on previous analysis attempts to:
1. process the inputs and outputs
2. apply mutations
3. handle functionalized randomness
4. deduplicate inputs and consolidate views into their bases (see input_output_analysis)
"""

import collections  # 导入collections模块，用于处理集合数据类型
import pprint  # 导入pprint模块，用于漂亮地打印数据结构
from contextlib import nullcontext  # 从contextlib模块导入nullcontext，用于创建一个空的上下文管理器
from dataclasses import dataclass, field  # 从dataclasses模块导入dataclass和field装饰器，用于定义数据类和字段
from functools import wraps  # 导入functools模块的wraps装饰器，用于包装函数以保留原始函数的元数据
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入typing模块，定义类型提示

import torch  # 导入PyTorch库
import torch.utils.dlpack  # 导入PyTorch的dlpack模块
from torch import Tensor  # 从torch模块导入Tensor类
from torch._guards import (
    compile_context,  # 导入PyTorch的编译上下文管理器
    CompileContext,  # 导入PyTorch的编译上下文类
    detect_fake_mode,  # 导入PyTorch的检测伪模式函数
    DuplicateInputs,  # 导入PyTorch的重复输入异常类
    tracing,  # 导入PyTorch的跟踪装饰器
    TracingContext,  # 导入PyTorch的跟踪上下文类
)

from torch._prims_common import CUDARngStateHelper  # 导入PyTorch的CUDA随机数状态助手
from torch._subclasses import FakeTensor  # 导入PyTorch的FakeTensor子类
from torch.fx.experimental._backward_state import BackwardState  # 导入PyTorch的反向状态类
from torch.multiprocessing.reductions import StorageWeakRef  # 导入PyTorch的存储弱引用类
from torch.utils._python_dispatch import is_traceable_wrapper_subclass  # 导入PyTorch的可跟踪包装子类判断函数

from .. import config  # 导入上级目录的config模块
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata  # 从当前目录的collect_metadata_analysis模块导入函数

from .functional_utils import gen_alias_from_base  # 从当前目录的functional_utils模块导入函数
from .input_output_analysis import (
    compute_overlapping_inputs,  # 从当前目录的input_output_analysis模块导入函数
    create_synthetic_base_metadata,  # 从当前目录的input_output_analysis模块导入函数
    remove_dupe_metadata,  # 从当前目录的input_output_analysis模块导入函数
)

from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling  # 从当前目录的logging_utils模块导入函数
from .schemas import (
    AOTConfig,  # 从当前目录的schemas模块导入AOTConfig类
    InputAliasInfo,  # 从当前目录的schemas模块导入InputAliasInfo类
    MutationType,  # 从当前目录的schemas模块导入MutationType类
    OutputType,  # 从当前目录的schemas模块导入OutputType类
    SubclassCreationMeta,  # 从当前目录的schemas模块导入SubclassCreationMeta类
    SubclassMeta,  # 从当前目录的schemas模块导入SubclassMeta类
    TensorAlias,  # 从当前目录的schemas模块导入TensorAlias类
    ViewAndMutationMeta,  # 从当前目录的schemas模块导入ViewAndMutationMeta类
)

from .subclass_utils import (
    get_types_for_subclass,  # 从当前目录的subclass_utils模块导入函数
    requires_subclass_dispatch,  # 从当前目录的subclass_utils模块导入函数
    unwrap_tensor_subclasses,  # 从当前目录的subclass_utils模块导入函数
    wrap_tensor_subclasses,  # 从当前目录的subclass_utils模块导入函数
)

from .traced_function_transforms import aot_dispatch_subclass  # 从当前目录的traced_function_transforms模块导入函数

from .utils import (
    call_func_at_runtime_with_args,  # 从当前目录的utils模块导入函数
    make_boxed_func,  # 从当前目录的utils模块导入函数
    normalize_as_list,  # 从当前目录的utils模块导入函数
    partial_flatten_asdict,  # 从当前目录的utils模块导入函数
    strict_zip,  # 从当前目录的utils模块导入函数
)

zip = strict_zip  # 将strict_zip函数赋值给变量zip

class CompilerWrapper:
    """
    A wrapper around the inputs and outputs to the compiler_fn. We separate these into two parts:

    1. The prologue, which edits the input to the compiler_fn(flat_fn, flat_args, etc)
    2. The epilogue, which edits the outputs of the compiler_fn (compiled_fn, real arguments)

    Each wrapper below should be implemented as a CompilerWrapper, so that we can facilitate
    caching on the compiled output, and re-wrapping the output via epilogues.
    Extra metadata that is needed to compute pre or post compile can be passed in via attributes.
    """
    
    def pre_compile(
        self,
        flat_fn,  # 参数：flat_fn，传递给编译器函数的扁平化函数
        flat_args: List[Tensor],  # 参数：flat_args，传递给编译器函数的张量列表
        aot_config: AOTConfig,  # 参数：aot_config，AOT配置对象
        *,
        fw_metadata: ViewAndMutationMeta,  # 关键字参数：fw_metadata，视图和变异元数据对象
    ) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
        """
        Process the inputs to the compiler_fn. You can pass in extra metadata via kwargs.
        Args:
        flat_fn: The function to compile
        flat_args: Metadata from example inputs of the function to compile
        aot_config: AOTConfig passed in at compile time
        fw_metadata: ViewAndMutationMeta generated from flat_fn and flat_args
        """
        返回一个元组，包含编译函数flat_fn，输入函数flat_args的元数据列表，以及从flat_fn和flat_args生成的fw_metadata。
        可以通过kwargs传递额外的元数据信息。
        return flat_fn, flat_args, fw_metadata

    def post_compile(self, compiled_fn, aot_config, *, runtime_metadata) -> Callable:
        """
        Given an output of the compiler, wrap it with information received from prologue.
        Args:
        compiled_fn: Callable after calling compiler_fn
        aot_config: AOTConfig after calling prologue
        runtime_metadata: ViewAndMutationMeta after calling all wrappers's pre_compile steps.
        Example:

        def wrapped_compiled_fn(args):
            # do something with args, aot_config, fw_metadata
            return compiled_fn(args)

        return wrapped_compiled_fn
        """
        给定编译器的输出，使用从序言中接收到的信息对其进行包装。
        返回一个函数wrapped_compiled_fn，该函数在处理args时利用了compiled_fn、aot_config和fw_metadata。
        最终返回compiled_fn的调用结果。
        return compiled_fn
# 该函数创建的包装器处理所有运行时别名和变异的"结尾处理"逻辑。
# 它接受一个trace_joint标志，指示我们是否为前向推断图生成运行时结尾处理，
# 还是为autograd.Function.apply函数生成。这是因为在运行时处理这些情况时存在一些细微差异：
# - 在推断情况下处理resize_()，但在autograd情况下尚未完全处理。
# - autograd情况下，为输出别名输入插入TensorAlias包装器对象。
@dataclass
class RuntimeWrapper(CompilerWrapper):
    indices_of_inps_to_detach: List[int]
    trace_joint: bool
    disable_amp: bool

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        # 返回通过_create_runtime_wrapper函数创建的运行时包装器
        return _create_runtime_wrapper(
            compiled_fn,
            runtime_metadata=runtime_metadata,
            indices_of_inps_to_detach=self.indices_of_inps_to_detach,
            trace_joint=self.trace_joint,
            keep_input_mutations=aot_config.keep_inference_input_mutations,
            disable_amp=self.disable_amp,
        )


def _create_runtime_wrapper(
    compiled_fn,
    *,
    runtime_metadata: ViewAndMutationMeta,
    indices_of_inps_to_detach: List[int],
    trace_joint: bool,
    keep_input_mutations: bool,
    disable_amp: bool,
):
    # 如果compiled_fn没有"_boxed_call"属性，使用make_boxed_func函数包装compiled_fn
    if not hasattr(compiled_fn, "_boxed_call"):
        compiled_fn = make_boxed_func(compiled_fn)

    # 注释 [Inputs needed in runtime epilogue after list clearing]
    # 在Python函数中，无法在函数的作用域内释放函数的输入参数。一种解决方法是将输入参数包装在列表中，并在函数内部清除列表。
    # 这里实现为`call_func_at_runtime_with_args(..., steal_args=True)`。
    #
    # 这在编译的Autograd中是必需的，因为一些输入（激活）应该尽早释放。
    # 然而，我们不能盲目地清除整个列表，因为AOTAutograd可能需要在编译函数运行后访问一些图输入。
    # 主要有两种情况：
    #   (1) 输入变异：如果存在必须在图之外运行的输入变异，我们需要访问输入。
    #   (2) 输出别名：通常需要重新生成图输入别名的输出，在`autograd.Function`之外做这个操作需要访问相应的输入。
    epilogue_args_idx = []
    epilogue_args_idx.extend(runtime_metadata.mutated_inp_runtime_indices)
    num_tokens = len(runtime_metadata.tokens)
    # 遍历运行时元数据的输出信息列表
    for info in runtime_metadata.output_info:
        # 检查输出类型是否为输入的别名或者直接作为输入
        if (
            info.output_type == OutputType.alias_of_input
            or info.output_type == OutputType.is_input
        ):
            # 断言基础索引是整数类型
            assert isinstance(info.base_idx, int)
            # 将基础索引加上标记数目后的结果索引添加到 epilogue_args_idx 列表中
            epilogue_args_idx.append(info.base_idx + num_tokens)

    # 返回运行时包装器函数
    return runtime_wrapper
@dataclass
class FunctionalizedRngRuntimeWrapper(CompilerWrapper):
    # TODO: I would love to get rid of this argument, but it's
    # Wrapped pretty tightly around our aot_dispatch_autograd logic.
    # Specifically, tensors_saved_for_backwards_slice's value is both used for calculating indices
    # for setting placeholder strides(which is done before runtime, before this wrapper runs)
    # and for saving tensors for backward (which is done during runtime, after this wrapper runs)
    # So in aot_dispatch_autograd, this wrapper can't edit the set of outs without making one
    # of those two indices incorrect.
    return_new_outs: bool = True  # 默认情况下，设置为 True，表示返回新的输出

    def pre_compile(
        self,
        flat_fn,
        flat_args,
        aot_config,
        *,
        fw_metadata,
    ) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
        if config.functionalize_rng_ops:  # 如果配置启用了随机数生成操作的功能化
            # Update example inputs for the fw_compiler
            fake_mode = detect_fake_mode()  # 检测虚拟模式
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            flat_args.extend([seed, offset])  # 将种子和偏移量添加到参数列表中
            # We are not clearing flat_args here because
            # 1) There is a check in the debug compiler at the end
            # 2) It does not matter as these are fake tensors
        return flat_fn, flat_args, fw_metadata  # 返回更新后的函数、参数和元数据信息

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        @wraps(compiled_fn)
        def wrapper(runtime_args: List[Any]):
            if runtime_metadata.is_rng_op_functionalized:  # 如果随机数操作已经被功能化
                # Add the seed and offset to args
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                runtime_args.extend([seed, offset])  # 将种子和偏移量添加到运行时参数中
                out = compiled_fn(runtime_args)  # 调用编译后的函数
                out = self._functionalized_rng_runtime_epilogue(
                    runtime_metadata,
                    out,
                    # TODO: this won't be right for the backward when we convert the call_compiled_backward to use the wrapper
                    runtime_metadata.num_forward_returns,
                )  # 处理功能化随机数运行的尾声逻辑
                return out
            return compiled_fn(runtime_args)  # 否则直接调用编译后的函数

        return wrapper  # 返回装饰后的函数作为包装器

    # Calling convention: If we are running functionalized RNG, then outs consists
    # of (user_outs, rng_offset)
    def _functionalized_rng_runtime_epilogue(
        self,
        metadata: ViewAndMutationMeta,
        outs,
        offset_index,
    ):
        if metadata.is_rng_op_functionalized:  # 如果随机数操作已经被功能化
            assert metadata.num_outputs_rng_offset == 1  # 断言确保随机数偏移量的数量为1
            new_rng_offset = outs[offset_index]  # 获取新的随机数偏移量
            CUDARngStateHelper.set_new_offset(new_rng_offset)  # 设置新的偏移量
            if self.return_new_outs:  # 如果设置为返回新的输出
                user_outs = outs[:offset_index] + outs[offset_index + 1 :]  # 提取用户输出，去除随机数偏移量
                return user_outs  # 返回用户输出
            else:
                return outs  # 否则返回所有输出

        return outs  # 如果随机数操作未被功能化，则直接返回输出
# CompilerWrapper 的子类，用于处理 AOTDispatch 运行时逻辑
class FakifiedOutWrapper(CompilerWrapper):
    # 输出元数据列表，初始化为空列表
    out_metas: List[torch.Tensor] = field(default_factory=list)
    # TracingContext.fwd_output_strides 的跟踪输出步幅
    # 实际编译生成的结果
    fwd_output_strides: Optional[List[List[int]]] = None
    # 是否需要在编译后进行处理的标志，默认为 True
    needs_post_compile: bool = True

    # 准备编译前的预处理函数
    def pre_compile(
        self,
        fw_module,  # 必须是来自 aot_dispatch_*_graph 的 fw_module
        flat_args,
        aot_config,
        *,
        fw_metadata,
    ) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
        # 尝试获取当前的跟踪上下文
        tracing_context = torch._guards.TracingContext.try_get()
        # 如果存在跟踪上下文且需要伪造第一个调用
        if tracing_context and tracing_context.fakify_first_call:
            # 设置输出元数据为最后一个节点的第一个参数的值的元数据的列表
            self.out_metas = [
                n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])
            ]
        else:
            # 否则不需要在编译后处理
            self.needs_post_compile = False
        # 返回原始的 fw_module, flat_args 和 fw_metadata
        return fw_module, flat_args, fw_metadata

    # 使用感应器步幅计算输出元数据函数
    def _compute_output_meta_with_inductor_strides(self):
        # 将 out 设置为当前的输出元数据列表
        out = self.out_metas
        # 将 fwd_output_strides 设置为当前的跟踪输出步幅
        fwd_output_strides = self.fwd_output_strides
        # 如果跟踪输出步幅为空，则直接返回当前的输出元数据列表
        if not fwd_output_strides:
            return out

        # 导入静态已知真值函数 statically_known_true
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        # 遍历输出元数据列表
        for i in range(len(out)):
            # 如果当前元数据不是 Tensor 类型，则继续下一个循环
            if not isinstance(out[i], Tensor):
                continue
            # 如果所有的静态已知真值函数对应的两个步幅相等
            if all(
                statically_known_true(s1 == s2)
                for s1, s2 in zip(out[i].stride(), fwd_output_strides[i])
            ):
                continue
            # 将当前元数据重新设置为具有感应器步幅的 strides 版本
            out[i] = out[i].as_strided(out[i].shape, fwd_output_strides[i])
        # 返回更新后的输出元数据列表
        return out

    # 在编译后调用，设置前向输出步幅函数
    def set_fwd_output_strides(self, fwd_output_strides):
        # 设置类成员变量 fwd_output_strides 为给定的前向输出步幅
        self.fwd_output_strides = fwd_output_strides

    # 编译后的处理函数
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        # 如果需要在编译后处理
        if self.needs_post_compile:
            # 断言前向输出步幅不为空
            assert self.fwd_output_strides is not None
            # 计算具有感应器步幅的输出元数据
            fakified_out = self._compute_output_meta_with_inductor_strides()

            # 创建一个包装器函数
            @wraps(compiled_fn)
            def wrapper(runtime_args):
                nonlocal fakified_out
                # 如果 fakified_out 不为空
                if fakified_out is not None:
                    out = fakified_out
                    fakified_out = None
                    return out
                # 否则返回原始的编译函数的结果
                return compiled_fn(runtime_args)

            # 返回包装器函数
            return wrapper
        # 如果不需要伪造，直接返回原始的编译函数
        return compiled_fn


# 这个包装器处理张量子类的 AOTDispatch 运行时逻辑。
# 在运行时，我们有一个已编译函数，知道如何在 DenseTensor 的域上操作，输出也是 DenseTensor。
# 但用户可能传递了一些张量子类输入（或期望某些子类张量输出）。
# 该函数处理运行时的张量子类的包装和解包。
@dataclass
class AOTDispatchSubclassWrapper(CompilerWrapper):
    # 是否跟踪联合
    trace_joint: bool
    fw_only: Optional[Callable]  # 可选参数，用于指定仅在预编译阶段使用，不会被缓存

    maybe_subclass_meta: Optional[SubclassMeta]  # 可选参数，可能包含子类元数据信息

    num_fw_outs_saved_for_bw: Optional[int]  # 可选参数，用于指定前向输出被保存以供反向传播使用的数量

    def pre_compile(
        self,
        flat_fn,
        flat_args: List[Tensor],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ):
        # 调用 aot_dispatch_subclass 函数，根据传入参数生成新的扁平化函数和参数列表，并返回子类元数据
        (new_flat_fn, new_flat_args, subclass_meta) = aot_dispatch_subclass(
            flat_fn,
            flat_args,
            is_joint_structure=self.trace_joint,  # 根据 self.trace_joint 判断是否联合结构
            meta=fw_metadata,  # 使用传入的前向元数据
            fw_only=self.fw_only,  # 将 self.fw_only 作为参数传递给 aot_dispatch_subclass 函数，类型为忽略参数类型
        )
        self.maybe_subclass_meta = subclass_meta  # 将返回的子类元数据存储在类属性中
        return new_flat_fn, new_flat_args, fw_metadata  # 返回新的扁平化函数、参数列表和前向元数据

    def post_compile(
        self,
        compiled_fn,
        _aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        # 如果 maybe_subclass_meta 为 None，则直接返回编译后的函数 compiled_fn
        if self.maybe_subclass_meta is None:
            return compiled_fn

        # 从运行时元数据中获取子类前向图输出的元数据
        subclass_metas = runtime_metadata.subclass_fw_graph_out_meta

        @wraps(compiled_fn)
        def inner_fn(args: List[Any]):
            # 解包参数中的张量子类，并根据联合结构属性 self.trace_joint 进行判断
            unwrapped_args = unwrap_tensor_subclasses(
                args, is_joint_structure=self.trace_joint
            )
            args.clear()  # 清空原始参数列表

            # 预期：runtime_fn 是一个封装的函数
            # 调用编译后的函数 compiled_fn，并传入解包后的参数
            unwrapped_outs = compiled_fn(unwrapped_args)

            # 将编译后函数的输出再次包装成张量子类，并传入相关的子类元数据和前向输出保存数量信息
            wrapped_outs = wrap_tensor_subclasses(
                unwrapped_outs,
                subclass_metas=subclass_metas,
                num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
                is_runtime=True,
            )
            return wrapped_outs  # 返回包装后的输出结果

        # 将 inner_fn 标记为已封装调用
        inner_fn._boxed_call = True  # 类型为忽略的属性定义
        return inner_fn  # 返回封装后的函数 inner_fn
# MOTIVATION:
#
# When tracing functions for future execution, one must be careful not to pass
# in the same input tensor multiple times (e.g., f(x, x), as this can result
# in graphs that are ONLY valid if you later pass a new tensor in exactly the
# same way (e.g., f(y, y)).  (NB: we really mean duplicate; two distinct
# tensors that alias each other is a different situation that is covered by
# aot_dispatch_deduplicated_autograd). Here are two examples:
#
# (1) Suppose you have a function:
#
#   def f(x, y):
#       return x + y
#
# If you make_fx(f)(x, x), you will trace out:
#
#   def f(x, y):
#       return y + y
#
# Oops!
#
# (2) For most tensors x and y, you can compute f's gradient with respect to
# these to inputs by saying torch.autograd.grad(f(x, y), (x, y)).  However,
# if x is y, you will trace out a program that gets incorrect gradients:
#
#   >>> x = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + x, (x, x))
#   (tensor([2.]), tensor([2.]))
#
# In other words, the gradient is double-counted.  Deduplicating the arguments
# gives you an appropriate gradient:
#
#   >>> y = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + y, (x, y))
#   (tensor([1.]), tensor([1.]))
#
# HOW TO DEDUPLICATE:
#
# There are a few strategies, in order of preference:
#
# 1. For every duplicate argument to the function, detach it into
#    a separate leaf tensor, so that it is no longer duplicated.
#
#       PRO: The resulting compiled graph works for any configuration
#       of duplicated arguments.
#
#       CON: It does not (naively) work if you mutate the metadata of inputs:
#
#           def f(x, y):
#               x.transpose_(0, 1)
#               y.transpose_(0, 2)
#
#           x = torch.randn(2, 3, 4)
#           f(x, x)
#
#       The ordering of the transposes inside f dictates whether or not
#       you get [4, 2, 3] or [3, 4, 2].  This means that you cannot precompute
#       what metadata mutations should get applied to each input; you need to
#       assume they aren't duplicates (what we do today) or preserve
#       the original metadata mutations exactly in order, so that they work
#       for any duplicate configuration.
#
#       CON: It does not (naively) work if you mutate the data of inputs.
#       In particular, leaf tensors that require grad cannot be mutated,
#       this makes it impossible to differentiate with respect to the original
#       base.
#
# 2. For every duplicate argument to the function, remove it, so it is
#    no longer part of the "true" signature:
#
#       PRO: Implemented naively, it still works for metadata/data mutation.
#
#       CON: The resulting compiled graph is duplicate-specialized: it only
#       works if future calls duplicate arguments in exactly the same way.
#       Horribly, Dynamo doesn't guard on this at the moment.  But even if
#       it did, you could still end up recompiling a bunch of each duplicate.
#
# Our strategy is to do (1) if we can, and do (2) otherwise, erroring if
# Dynamo's guards are not enough.  In practice, this seems to cover
# everything.
#
# 使用dataclass装饰器定义AOTDedupeWrapper类，它是CompilerWrapper的子类，
# 包含以下成员变量和默认值：
#   - keep_arg_mask: 一个布尔值列表，用于标记是否保留对应位置的参数
#   - add_dupe_map: 一个整数列表，指示要添加的重复参数的位置
#   - old_input_metadata: 一个InputAliasInfo对象列表，保存旧的输入元数据信息
#   - needs_post_compile: 一个布尔值，表示是否需要在编译后执行后处理操作，默认为True
@dataclass
class AOTDedupeWrapper(CompilerWrapper):
    keep_arg_mask: List[bool] = field(default_factory=list)
    add_dupe_map: List[int] = field(default_factory=list)
    old_input_metadata: List[InputAliasInfo] = field(default_factory=list)
    needs_post_compile: bool = True

    # NB: Hot path, avoid set lookups here
    # TODO: Can avoid the zip here too, probably
    # 定义remove_dupe_args方法，用于移除重复参数
    def remove_dupe_args(self, args):
        # 通过列表推导式，根据keep_arg_mask标记保留参数
        return [t for t, keep in zip(args, self.keep_arg_mask) if keep]

    # 定义add_dupe_args方法，用于添加重复参数
    def add_dupe_args(self, args):
        # 根据add_dupe_map指示的位置，从args中选取对应的参数添加到结果列表中
        return [args[i] for i in self.add_dupe_map]

    # 定义pre_compile方法，用于预编译操作
    def pre_compile(
        self,
        flat_fn,
        flat_args: List[Tensor],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    # 定义post_compile方法，用于后编译操作
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
        ):
            如果不需要后续编译步骤，则直接返回编译后的函数

            @wraps(compiled_fn)
            定义一个装饰器函数，用于包装编译后的函数，并保留原始函数的元数据

            def wrapped_compiled_fn(args: List[Any]):
                去除重复参数后，调用原始编译函数
                deduped_args = self.remove_dupe_args(args)
                清空参数列表
                args.clear()
                返回编译后的函数，传入去重后的参数列表
                return compiled_fn(deduped_args)

            wrapped_compiled_fn._boxed_call = True  # type: ignore[attr-defined]
            设置一个标志，表示这是一个包装调用的函数

            # This can be uncommented when we properly guard for duplicates,
            # but right now we must not do it.
            如果没有正确处理重复参数的保护措施，则不应取消注释以下代码段
            # if not config.debug_assert:
            #     return wrapped_compiled_fn

            @wraps(wrapped_compiled_fn)
            定义一个装饰器函数，用于调试编译后的函数，并保留原始函数的元数据

            def debugged_compiled_fn(args):
                测试计算的移除/添加参数函数是否为反函数
                new_args = self.add_dupe_args(self.remove_dupe_args(args))
                创建一个空字典来记录参数的出现情况
                seen: Dict[Any, None] = {}
                遍历新旧参数列表，确保没有重复参数出现，并触发断言错误信息
                for i, (x, y) in enumerate(zip(new_args, args)):
                    seen[y] = None
                    assert x is y, format_guard_bug_msg(
                        aot_config,
                        f"{describe_input(i, aot_config)} would be a duplicate of "
                        f"{describe_input(self.add_dupe_map[i], aot_config)}",
                    )
                这仅在元数据同时对重复参数进行了修改时才会触发错误
                return wrapped_compiled_fn(args)

            debugged_compiled_fn._boxed_call = True  # type: ignore[attr-defined]
            设置一个标志，表示这是一个包装调试后的函数

            返回调试后的编译函数
            return debugged_compiled_fn
# 处理两个输入相互引用且其中一个输入发生了变化的情况的层级。
# 我们需要特别注意确保变化应用到图中其他引用上。
#
# 前提条件：AOTDedupWrapper 已经运行。
# （理论上，如果有重复的参数，这个函数可以工作。
# 然而，合成基础代码路径有点亚优化，在存在重复输入的情况下，会导致更频繁地触发该路径）。
@dataclass
class AOTSyntheticBaseWrapper(CompilerWrapper):
    # 当前，我们需要传递这个布尔值的唯一原因是因为
    # 合成基础代码在自动微分情况下禁止更多的情况比推断情况下多。
    trace_joint: bool  # TODO: refactor trace_joint
    needs_post_compile: bool = True
    aliased_arg_idx_with_metadata_mutations: List[int] = field(default_factory=list)

    def pre_compile(
        self,
        flat_fn,
        flat_args: List[Any],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ):
        # 在编译之前的处理步骤，接收平坦化后的函数、参数列表、AOT 配置以及前向元数据
        pass

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        # 在编译之后的处理步骤，接收编译后的函数、AOT 配置以及运行时元数据
        pass
        ):
            # 如果不需要后编译，则直接返回编译后的函数
            if not self.needs_post_compile:
                return compiled_fn

            # 判断是否为推断阶段（非联合跟踪）
            is_inference = not self.trace_joint

            # 包装编译后的函数，保留其元数据
            @wraps(compiled_fn)
            def wrapped_compiled_fn(args):
                # 合并视图输入参数和旧的输入信息，生成合成基的参数
                args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
                    args, self.old_input_info, is_inference=is_inference
                )
                # 断言确保合成基信息不为 None
                assert synthetic_base_info is not None

                # 提取带有元数据变化的别名参数
                aliased_args_w_metadata_mutations = [
                    args[i] for i in self.aliased_arg_idx_with_metadata_mutations
                ]
                # 计算带有元数据变化的别名参数的数量
                num_aliased_args_with_metadata_mutations = len(
                    aliased_args_w_metadata_mutations
                )

                # 清空原始参数列表
                args.clear()

                # 调用编译后的函数并获取输出
                outs = compiled_fn(args_with_synthetic_bases)

                # 如果存在带有元数据变化的别名参数
                if num_aliased_args_with_metadata_mutations > 0:
                    # 处理部分输入元数据的变化，仅处理转换为合成基的输入的元数据变化
                    # 例如：
                    # def f(a, b):
                    #     a.mul_(2)
                    #     b.t_(1, 0)
                    # f(x.view(2, 2), x.view(2, 2))
                    mutated_metadata_inps = outs[-num_aliased_args_with_metadata_mutations:]
                    user_outs = outs[:-num_aliased_args_with_metadata_mutations]
                    # 对每个带有元数据变化的别名参数和其对应的变化输入进行操作
                    for inp, mutated_inp in zip(
                        aliased_args_w_metadata_mutations, mutated_metadata_inps
                    ):
                        inp.as_strided_(
                            mutated_inp.size(),
                            mutated_inp.stride(),
                            mutated_inp.storage_offset(),
                        )
                    return user_outs

                # 返回处理后的输出
                return outs

            # 返回包装后的编译函数
            return wrapped_compiled_fn
# Note [Handling mutations on an input that aliases other inputs]
# The easiest example to show-case this edge case is here:
#
# def f(a, b):
#     a.mul_(2)
#     out = a + b
#     return out
# b = torch.ones(...)
# a = b.view(-1)
# f(a, b)
#
# In this situation, if a and b happened to be aliased, we need to trace something different!
# Suppose we had b = a.view(-1)
# (In this case, that means that `a._base is b`)
#
# We need to ensure that the aliasing relationship between a and b is preserved.
# We do that detecting the specific situation above (mutate an input that aliases another input),
# and when we do that, we create a synthetic base argument. Then inside of the traced forward,
# we regenerate a and b off of that base.
# The complete example of the transformed function looks like this:
#
# // The traced forward takes in a synthetic base, and regenerates the aliased inputs as views
# // We could consider getting view-replay support here to minimize as_strided_scatter ops in the graph
# def traced_forward(base):
#     a = base.as_strided(...)
#     b = base.as_strided(...)
#     a_updated = a.mul(2)
#     base_updated = torch.as_strided_scatter(base, a_updated, ...)
#     b_updated = base_updated.as_strided(...)
#     out = a_updated + b_updated
#     return a_updated, out
#
# def compiled_fn(a, b):
#     // we detect that a is the "differentiable base" here
#     base = a
#     // In other situations, we might do either:
#     // (1) a and b are both views off of some larger differentiable base
#     //     assert a._base is b._base and a._base is not None
#     //     base = a._base
#     // (2) a and b both don't require gradients. Create a base from the storage
#     //     assert a._base is None and b._base is None
#     //     base = torch.Tensor(a.storage())
#     a_updated, out = traced_forward(base)
#     a.copy_(a_updated)
#     return out
#
# This function:
# (1) Merges input views into a synthetic base argument, when any of those input views are mutated
# (2) Returns metadata telling the autograd.Function how to modify their arguments properly,
#     to respect the new calling convention.
#
# The calling convention is as follows.
# Any inputs that were originally views of one another get yanked, and replaced with a synthetic base.
# The argument list ordering goes [base1, ..., baseN], [arg1, ..., argN],
# Where the ordering of the bases is determined from the ordering of the original view args.
# baseA will come before baseB if the earliest original argument coming from baseA
# showed up earlier in the argument list than the earliest original argument coming from baseB.
#
# Example, given some tensors a, b, c, d
# call site:
#   f(a, c.view(-1), b.view(-1), b, c, d)
# Modified argument list:
#   c_base comes first because the first c view came earlier in arg list than the first b view
#   a and d still show up in the modified arg list, but b and c don't- they're regenerated from their bases
#   b_base = torch.Tensor(b.storage())
#   c_base = torch.Tensor(c.storage())
#   f(c_base, b_base, a, d)
def merge_view_inputs(
    fwd_inputs: List[Any],
    mutated_input_info: List[InputAliasInfo],
    *,
    # The autograd case currently has more restrictions than the inference case.
    is_inference: bool,
) -> Tuple[List[Any], Optional[List[Union[int, Tuple[int, torch.Tensor]]]]]:
    def _are_differentiable_views(view1, view2):
        if view1 is view2:
            return True
        if view1._base is None and view2._base is None:
            return False
        if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
            return True
        return False

    def _same_dtype_views(view1, view2):
        if view1.dtype != view2.dtype:
            return False
        if view1._base is not None and view1.dtype != view1._base.dtype:
            return False
        if view2._base is not None and view2.dtype != view2._base.dtype:
            return False
        return True

    # Assert that the length of forward inputs matches mutated input information
    assert len(fwd_inputs) == len(mutated_input_info)
    
    # Early return if there are no mutated inputs that alter data
    if not [info for info in mutated_input_info if info.mutates_data]:
        return fwd_inputs, None

    # Dictionary to map weak references to storage to their indices in the forward inputs list
    storage_ref_to_idx: Dict[StorageWeakRef, List[int]] = collections.defaultdict(list)
    base_args = []
    other_args = []
    
    # Iterate over forward inputs to categorize them into storage references or other arguments
    for i, inpt in enumerate(fwd_inputs):
        if isinstance(inpt, Tensor):
            storage_ref = StorageWeakRef(inpt.untyped_storage())
            storage_ref_to_idx[storage_ref].append(i)
        else:
            other_args.append(inpt)

    # Note [Synthetic Base Info Metadata]
    # This dictionary holds metadata about synthetic bases needed for inner calling convention
    # The metadata can be:
    # - another integer (index in the argument list from outer calling convention)
    # - idx, view_tensor (used to generate new output with view_tensor._view_func(old_args[idx]))
    inner_calling_convention_meta: Dict[int, Union[int, Tuple[int, torch.Tensor]]] = {}

    # If no synthetic bases are required, return the original forward inputs
    if len(base_args) == 0:
        assert len(other_args) == len(fwd_inputs)
        return fwd_inputs, None
    else:
        # 否则，返回：
        # (1) 根据更新后的调用约定返回新的参数：(synthetic_bases, other_args)
        # (2) 元数据，告诉功能化如何根据外部调用约定生成内部参数列表。
        #     我们将其后处理为列表，其中 meta[i] 告诉您关于内部调用约定中第 i 个参数的信息。
        args_to_functionalization = base_args + other_args
        # 创建一个映射，将参数映射回原始索引
        arg_to_old_idx_map = {arg: i for (i, arg) in enumerate(fwd_inputs)}
        # 更新内部调用约定的元数据
        for i, other_arg in enumerate(other_args):
            new_idx = len(base_args) + i
            old_idx = arg_to_old_idx_map[other_arg]
            inner_calling_convention_meta[old_idx] = new_idx
        # 后处理成一个列表
        post_processed_calling_convention_meta: List[
            Union[int, Tuple[int, torch.Tensor]]
        ] = [-1 for _ in range(len(inner_calling_convention_meta))]
        # 将内部调用约定的元数据填充到后处理列表中
        for k, v in inner_calling_convention_meta.items():
            post_processed_calling_convention_meta[k] = v
        # 快速断言：内部调用约定中的每个参数都应该有对应的值。
        for x in post_processed_calling_convention_meta:
            assert x != -1
        # 返回功能化后的参数和后处理的调用约定元数据
        return args_to_functionalization, post_processed_calling_convention_meta
# 数据类，用于保存自动求导的懒惰后向编译信息
@dataclass
class AutogradLazyBackwardCompileInfo:
    bw_module: Callable                    # 可调用对象，表示后向传播模块
    placeholder_list: List[Any]            # 占位符列表，用于保存任意类型的占位符
    saved_context: Optional[TracingContext] # 可选的追踪上下文，用于保存追踪上下文信息
    saved_compile_context: Optional[CompileContext] # 可选的编译上下文，用于保存编译上下文信息


# 这个类只是为了命名空间的目的而被包装在一个类中
# 不需要将其实际作为 CompilerWrapper，因为它与抽象的匹配不够清晰
class AOTDispatchAutograd:
    @staticmethod
    def _force_contiguous(x):
        if not isinstance(x, torch.Tensor):
            return x
        x = x.contiguous()  # 强制转换为连续张量
        if not is_traceable_wrapper_subclass(x):
            return x
        for attr in x.__tensor_flatten__()[0]:  # 获取张量展平后的属性列表
            elem = getattr(x, attr)
            if not elem.is_contiguous():    # 如果属性不是连续的，则强制转换为连续张量
                setattr(x, attr, elem.contiguous())
        return x

    # 查看注释 [切线必须是连续的，第二部分]
    @staticmethod
    def coerce_runtime_tangent(x, metadata):
        if not isinstance(x, torch.Tensor):
            return x
        if not is_traceable_wrapper_subclass(x):
            return x
        assert metadata is not None
        (_, expected_tangent_metadata) = metadata
        _, runtime_tangent_metadata = x.__tensor_flatten__()  # 获取张量展平后的运行时切线元数据
        if runtime_tangent_metadata == expected_tangent_metadata:
            return x
        if not hasattr(x, "__coerce_same_metadata_as_tangent__"):
            raise RuntimeError(
                f"""
在后向传播过程中，我们遇到了一个张量子类，我们错误地猜测了它的元数据。

预期元数据: {str(expected_tangent_metadata)}

运行时元数据: {str(runtime_tangent_metadata)}

形状: {str(x.shape)}
要解决此问题，您的张量子类必须实现双下划线方法 __force_to_same_metadata__。
"""
            )
        return x.__coerce_same_metadata_as_tangent__(expected_tangent_metadata)  # 转换为与切线元数据相同的元数据类型

    @staticmethod
    def post_compile(
        compiled_fw_func,  # 编译后的前向模块 + 包装器
        compiled_bw_func,  # 编译后的后向模块 + 包装器
        maybe_subclass_meta: Optional[SubclassMeta],  # 可选的子类元数据
        num_symints_saved_for_bw_: int,   # 为后向传播保存的符号整数的数量
        backward_state_indices: List[int],    # 后向状态索引列表
        disable_amp: bool,  # 是否禁用混合精度训练
        indices_of_inps_to_detach: List[int],  # 需要分离输入的索引列表
        lazy_backward_info: Optional[AutogradLazyBackwardCompileInfo],  # 可选的自动求导懒惰后向编译信息
        aot_config: AOTConfig,  # AOT（Ahead-of-Time）配置对象
        *,
        fw_metadata: ViewAndMutationMeta,  # 运行时元数据
        try_save_cache_entry: Optional[Callable],  # 编译后保存缓存条目的可选调用对象
    ):
        """
        我们错误地尝试使用不正确的子类元数据编译后向传播。
        如果您遇到此错误，请提交一个问题。
        预期的梯度输出类型: {str(CompiledFunction.metadata.output_types)}
        """


@dataclass
class DebugAssertWrapper(CompilerWrapper):
    flat_requires_grad: List[Optional[bool]] = field(default_factory=list)  # 用于调试断言的包装器，记录平坦化的梯度要求列表
    # 定义一个函数post_compile，用于在编译后对生成的函数进行调试
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        # 使用@wraps装饰器将debug_compiled_function函数的元数据与compiled_fn保持一致
        @wraps(compiled_fn)
        def debug_compiled_function(args: List[Any]):
            # TODO: 检查别名关系
            # TODO: 检查元数据突变的步幅
            # （注意：理想情况下，这些逻辑应该抽离出此函数，将这些调试检查移到那里）

            # 遍历参数列表args，逐个检查
            for i, a in enumerate(args):
                can_require_grad = self.flat_requires_grad[i]
                # 如果flat_requires_grad中对应索引的值为None，断言参数a不是Tensor对象
                if can_require_grad is None:
                    assert not isinstance(a, Tensor)
                # 否则，如果can_require_grad为False，则断言参数a的requires_grad属性为False
                elif not can_require_grad:
                    assert not a.requires_grad, format_guard_bug_msg(
                        aot_config,
                        f"{describe_input(i, aot_config)} would not require grad",
                    )

            # 调用编译后的函数compiled_fn并返回结果
            return compiled_fn(args)

        # 返回调试后的编译函数debug_compiled_function
        return debug_compiled_function
# 对给定的函数和参数运行一系列编译器包装器
# 在执行过程中修改编译器包装器
def pre_compile(
    wrappers: List[CompilerWrapper],
    flat_fn: Callable,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function and arguments.
    Mutates wrappers in place.
    """
    # 遍历每个编译器包装器
    for wrapper in wrappers:
        # 调用每个包装器的 pre_compile 方法，对给定的函数和参数进行处理
        flat_fn, flat_args, fw_metadata = wrapper.pre_compile(
            flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
        )
    # 返回处理后的函数、参数列表和元数据
    return flat_fn, flat_args, fw_metadata


# 对给定的已编译函数运行一系列编译后处理包装器
# 应在 pre_compile() 之后调用
def post_compile(
    wrappers: List[CompilerWrapper],
    compiled_fn: Callable,
    aot_config: AOTConfig,
    *,
    runtime_metadata: ViewAndMutationMeta,
) -> Tuple[Callable, ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function. Should be called after pre_compile()
    """
    # 反向遍历编译器包装器列表
    for wrapper in reversed(wrappers):
        # 调用每个包装器的 post_compile 方法，对已编译函数进行处理
        compiled_fn = wrapper.post_compile(
            compiled_fn, aot_config, runtime_metadata=runtime_metadata
        )
    # 返回处理后的已编译函数和运行时元数据
    return compiled_fn, runtime_metadata


# 使给定的 ViewAndMutationMeta 对象在运行时安全
# 修改了两个参数。允许 ViewAndMutationMeta 对象在 AOTAutogradCache 中安全缓存
def make_runtime_safe(
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: Optional[SubclassMeta],
):
    """
    Calls make_runtime_safe on all ViewAndMutationMetas.
    Modifies both arguments. Allows ViewAndMutationMetas to
    be safely cached in AOTAutogradCache.
    """
    # 调用 fw_metadata 对象的 make_runtime_safe 方法
    fw_metadata.make_runtime_safe()
    # 如果 maybe_subclass_meta 参数不为 None
    if maybe_subclass_meta is not None:
        # 调用 maybe_subclass_meta 对象的 fw_metadata 的 make_runtime_safe 方法
        maybe_subclass_meta.fw_metadata.make_runtime_safe()
```