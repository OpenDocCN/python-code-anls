# `.\pytorch\torch\_inductor\compile_fx.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import contextlib  # 上下文管理工具
import functools  # 函数工具
import itertools  # 迭代工具
import logging  # 日志记录
import os  # 操作系统功能
import sys  # 系统相关功能
import time  # 时间操作
import warnings  # 警告控制
from itertools import count  # 迭代工具中的计数器

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union  # 类型提示
from unittest import mock  # 单元测试的 mock 模块

import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
# 引入异步编译相关模块，用于预热 AsyncCompile 池

import torch.fx  # PyTorch 的特征图库
import torch.utils._pytree as pytree  # PyTree 相关实用工具

from functorch.compile import min_cut_rematerialization_partition  # 编译相关函数
from torch._dynamo import (
    compiled_autograd,  # 编译自动求导
    config as dynamo_config,  # Dynamo 配置
    logging as dynamo_logging,  # Dynamo 日志
    utils as dynamo_utils,  # Dynamo 实用工具
)
from torch._dynamo.utils import (
    counters,  # 计数器
    detect_fake_mode,  # 检测假模式
    flatten_graph_inputs,  # 平展图输入
    lazy_format_graph_code,  # 惰性格式化图代码
)
from torch._functorch import config as functorch_config  # Functorch 配置
from torch._functorch.aot_autograd import aot_export_module, make_boxed_func  # AOT 自动求导
from torch._inductor.codecache import (
    _StrideExprStr,  # 步幅表达式字符串
    code_hash,  # 代码哈希
    CompiledFxGraph,  # 编译的特征图
    FxGraphCache,  # 特征图缓存
)
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,  # 包装的设备索引
    get_placeholders,  # 获取占位符
    log_cudagraph_skip_and_bump_counter,  # 记录 cudagraph 跳过并增加计数器
)
from torch._inductor.debug import save_args_for_compile_fx_inner  # 调试工具：保存编译特征图内部参数
from torch._inductor.runtime.runtime_utils import cache_dir  # 运行时实用工具：缓存目录
from torch._inductor.utils import (
    BoxedBool,  # 包装的布尔值
    count_tangents,  # 计数切线
    fresh_inductor_cache,  # 新鲜的 Inductor 缓存
    should_assume_input_aligned,  # 是否假设输入对齐
    tensor_is_aligned,  # 张量是否对齐
)
from torch._logging import trace_structured  # 结构化跟踪日志
from torch._ops import OpOverload  # 运算符重载
from torch._subclasses.fake_tensor import FakeTensor  # 假张量
from torch._utils_internal import compile_time_strobelight_meta  # 编译时间的 strobelight 元信息
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymExprPrinter  # 符号形状相关实验特征图库
from torch.fx.passes.fake_tensor_prop import FakeTensorProp  # 假张量属性相关特征图库

from .._dynamo.backends.common import aot_autograd  # 动力学自动求导相关库
from ..fx._lazy_graph_module import _use_lazy_graph_module  # type: ignore[attr-defined] 惰性图模块相关库
from ..fx.graph import _PyTreeCodeGen  # 拼接树代码生成器相关库
from . import config, metrics  # 导入当前模块内的配置和度量

from .debug import DebugContext  # 调试上下文相关库
from .decomposition import select_decomp_table  # 选择分解表相关库
from .fx_passes.joint_graph import joint_graph_passes  # 联合图传递相关库
from .fx_passes.post_grad import post_grad_passes, view_to_reshape  # 后向传递和视图重塑相关库
from .fx_passes.pre_grad import pre_grad_passes  # 前向传递相关库
from .graph import GraphLowering  # 图降低相关库
from .ir import ExternKernelNode  # 外部内核节点
from .utils import (
    get_cloned_parameter_buffer_name,  # 获取克隆参数缓冲区名称
    has_incompatible_cudagraph_ops,  # 是否存在不兼容的 cudagraph 操作
    maybe_get_suppress_shape_guards_ctx,  # 可能获取抑制形状保护上下文
    output_node,  # 输出节点
)
from .virtualized import V  # 虚拟化相关类

if config.is_fbcode():
    from torch._inductor.fb.utils import log_optimus_to_scuba, time_and_log
else:
    # no-op decorator
    def time_and_log(attr: str):
        return dynamo_utils.identity

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")  # 获取性能提示日志记录器
post_grad_graphs_log = torch._logging.getArtifactLogger(__name__, "post_grad_graphs")  # 获取后向传递图日志记录器
ALIGNMENT = 16  # 设定对齐值为 16
# 获取扩展维度（即之前大小为1的维度 -> ?）
# 我们可以选择该维度中的一个元素并对其进行写入，
# 从而实现对输入张量该维度所有值的写入
def get_expanded_dims(t):
    if not isinstance(t, torch.Tensor):
        return None
    # 返回所有满足条件的维度列表，条件为该维度步长为0且大小不为1
    return [i for i in range(t.ndim) if t.stride(i) == 0 and t.size(i) != 1]


def index_expanded_dims(t: torch.Tensor, expanded_dims: List[int]) -> torch.Tensor:
    # 针对每个扩展维度，对输入张量进行切片操作，将该维度上除第一个元素外的元素都切除
    for expanded_dim in expanded_dims:
        t = torch.ops.aten.slice(t, expanded_dim, 0, 1)
    return t


def complex_memory_overlap(t: torch.Tensor) -> bool:
    # 如果 torch._debug_has_internal_overlap 认为该张量可能存在内存重叠，
    # 则进一步探索以确认是否属实。
    #
    # 调用 squeeze() 方法，以避免因大小为1的维度而导致误报。
    t = index_expanded_dims(t, get_expanded_dims(t)).squeeze()
    if torch._debug_has_internal_overlap(t) != 0:
        strides = t.stride()
        sizes = t.shape
        indices = list(range(len(strides)))
        # 根据步长排序索引，以便进一步检查内存重叠情况
        indices = [x for _, x in sorted(zip(strides, indices))]
        for i in range(len(strides)):
            prev_stride = 1 if i == 0 else strides[indices[i - 1]]
            prev_size = 1 if i == 0 else sizes[indices[i - 1]]
            # 检查当前维度的步长是否小于前一个维度总大小的步长
            if strides[indices[i]] < prev_stride * prev_size:
                return True
    return False


def get_static_input_idxs(num_fixed):
    # 如果我们正在内联 NNModules，则对于 cudagraphs 的目的，
    # 我们将所有 torch.nn.Parameters 视为静态的。
    # 与每次运行时普通输入不同，我们将重新记录 cudagraph（如果这些参数位置发生变化）。
    context = torch._guards.TracingContext.try_get()
    fixed = list(range(num_fixed))
    if not context or not context.fw_metadata:
        return fixed

    # 返回固定参数索引和当前 cudagraph 的静态参数索引的组合
    return fixed + context.fw_metadata.static_parameter_indices


@functools.lru_cache(None)
def _step_logger():
    # 返回与日志相关的步骤记录器
    return dynamo_logging.get_step_logger(log)


@functools.lru_cache(None)
def _warn_tf32_disabled():
    if (
        torch.cuda.is_available()
        and not torch.backends.cuda.matmul.allow_tf32
        and torch.cuda.get_device_capability() >= (8, 0)
    ):
        # 如果 TensorFloat32（float32 矩阵乘法的张量核心）可用但未启用，
        # 则发出警告，并建议设置 `torch.set_float32_matmul_precision('high')` 以获得更好性能。
        warnings.warn(
            "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. "
            "Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
        )


def _unlift_graph(mod, gm, graph_signature):
    from torch.export.unflatten import _assign_attr, _AttrKind

    state_dict = {}
    # 遍历模型中的每个命名参数，将其存储到 state_dict 中，
    # 并为每个参数分配属性，以便后续重构时使用
    for name, param in mod.named_parameters(remove_duplicate=False):
        state_dict[name] = param
        _assign_attr(
            param,
            gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )
    # 遍历模块的命名缓冲区，并将其名称和缓冲区本身存入状态字典
    for name, buffer in mod.named_buffers(remove_duplicate=False):
        state_dict[name] = buffer
        # 将缓冲区作为属性分配给模块，并标记为缓冲区类型
        _assign_attr(
            buffer,
            gm,
            name,
            attr_kind=_AttrKind.BUFFER,
        )

    # 在图中查找所有操作为“placeholder”的节点
    placeholder_nodes = gm.graph.find_nodes(op="placeholder")
    lifted_inputs = []

    # 在AOTI中，模块的参数和缓冲区不会被作为图的输入提升。
    # 因此，对缓冲区的突变会导致其初始值与Eager模式下不同。因此在此处克隆它们作为副本。
    # 对于参数，我们不进行克隆，尽管如果需要支持训练，将会需要这样做。
    for node in placeholder_nodes:
        node_name = node.name
        # 如果节点名在图签名中的输入到参数的映射中
        if node_name in graph_signature.inputs_to_parameters:
            parameter_name = graph_signature.inputs_to_parameters[node_name]
            lifted_inputs.append(parameter_name)
        # 如果节点名在图签名中的输入到缓冲区的映射中
        elif node_name in graph_signature.inputs_to_buffers:
            buffer_name = graph_signature.inputs_to_buffers[node_name]
            lifted_inputs.append(buffer_name)
            # 将克隆的缓冲区名称及其保持步幅的副本存入图元数据
            gm.meta[
                get_cloned_parameter_buffer_name(buffer_name)
            ] = clone_preserve_strides(state_dict[buffer_name])
        else:
            # 断言：节点名应该在图签名的用户输入中
            assert node_name in graph_signature.user_inputs
            lifted_inputs.append(None)

    # 导入_unlift模块中的_unlift函数
    from torch.export._unlift import _unlift

    # 获取图中最后一个节点的输出参数
    outputs = list(gm.graph.nodes)[-1].args[0]
    mutated_outputs = []
    buffer_mutations = graph_signature.buffers_to_mutate
    user_input_mutations = graph_signature.user_inputs_to_mutate
    output_tokens = graph_signature.output_tokens

    # 遍历输出节点，获取被突变的输出
    for idx, out in enumerate(outputs):
        value = None

        # 如果索引小于缓冲区突变量、用户输入突变量和输出标记的总和
        if idx < len(buffer_mutations) + len(user_input_mutations) + len(output_tokens):
            # 如果输出名称在缓冲区突变量中
            if out.name in buffer_mutations:
                value = buffer_mutations[out.name]
            # 如果输出名称在用户输入突变量中
            elif out.name in user_input_mutations:
                value = user_input_mutations[out.name]

        mutated_outputs.append(value)

    # 使用_unlift函数对图模型进行反提升操作，返回反提升后的图模型
    unlifted_gm = _unlift(
        gm,
        lifted_inputs,
        mutated_outputs,
        pytree.LeafSpec(),
        None,
        state_dict,
        {},
    )
    # 返回反提升后的图模型
    return unlifted_gm
def _get_subgraph_names(gm):
    # 遍历 gm.graph 中调用了 torch.ops.higher_order.cond 和 torch.ops.higher_order.while_loop 的节点
    for node in sorted(
        itertools.chain(
            gm.graph.find_nodes(op="call_function", target=torch.ops.higher_order.cond),
            gm.graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.while_loop
            ),
        )
    ):
        if node.target == torch.ops.higher_order.cond:
            # 获取条件为真和条件为假时的子图名称
            true_subgraph_name = node.args[1].name
            false_subgraph_name = node.args[2].name
            yield true_subgraph_name
            yield false_subgraph_name
        elif node.target == torch.ops.higher_order.while_loop:
            # 获取循环条件和循环体的子图名称
            cond_subgraph_name = node.args[0].name
            body_subgraph_name = node.args[1].name
            yield cond_subgraph_name
            yield body_subgraph_name


def _recursive_pre_grad_passes(gm, example_inputs):
    # 递归地对 gm 的子图应用预梯度传递操作
    for subgraph_name in _get_subgraph_names(gm):
        subgraph = getattr(gm, subgraph_name)
        # 在这里 example_inputs 被设为 None，因为没有递归示例输入
        new_subgraph = _recursive_pre_grad_passes(subgraph, example_inputs=None)
        setattr(gm, subgraph_name, new_subgraph)
    # 对 gm 应用预梯度传递操作
    return pre_grad_passes(gm, example_inputs)


def _recursive_joint_graph_passes(gm):
    # 递归地对 gm 的子图应用联合图传递操作
    for subgraph_name in _get_subgraph_names(gm):
        subgraph = getattr(gm, subgraph_name)
        _recursive_joint_graph_passes(subgraph)
    # 对 gm 应用联合图传递操作
    joint_graph_passes(gm)


def _recursive_post_grad_passes(gm, is_inference: bool = False):
    # 递归地对 gm 的子图应用后梯度传递操作
    for subgraph_name in _get_subgraph_names(gm):
        subgraph = getattr(gm, subgraph_name)
        _recursive_post_grad_passes(subgraph, is_inference)
    # 对 gm 应用后梯度传递操作
    post_grad_passes(gm, is_inference)


def split_const_gm(
    gm: torch.fx.GraphModule,
) -> Tuple[torch.fx.GraphModule, Dict[str, int]]:
    """
    This function takes an GraphModule input "gm".
    The gm will be split into 2 components,
      1) const_gm, which consists the subgraph of gm that can be constant folded.
      2) gm (being inplace modified,) which returns the graph after constant folding.

    const_output_index is a mapping of corresponding node name from gm to the
    output index of const_gm.
    Returns (const_gm, const_output_index)
    """
    from torch._inductor.constant_folding import (
        CONST_MODULE_TAG,
        META_TAG,
        MODULE_TAG,
        replace_node_with_constant,
        run_and_get_constant_graph,
    )

    # 运行常量折叠函数，得到可以常量折叠的子图 const_gm
    const_gm = run_and_get_constant_graph(gm)
    const_result = const_gm()

    # 构建一个字典 const_outputs，将 gm 中的节点名称映射到 const_gm 中的输出索引
    const_outputs = {
        x.name: idx for idx, x in enumerate(tuple(const_gm.graph.nodes)[-1].args[0])
    }

    to_erase_node = []
    to_replace_node = []
    const_output_index = {}
    for node in gm.graph.nodes:
        if node.name in const_outputs:
            # 如果节点在 const_outputs 中，则将其标记为待替换节点
            to_replace_node.append(node)
        elif node.meta[META_TAG] == CONST_MODULE_TAG:
            # 如果节点的元数据标记为 CONST_MODULE_TAG，则将其标记为待删除节点
            to_erase_node.append(node)
    # 遍历需要替换的节点列表
    for node in to_replace_node:
        # 创建新的常量名称，以原节点名称为基础
        new_const_name = "_FOLDED_CONST_" + node.name
        # 使用替换节点函数，将图中的节点替换为常量，并指定新常量的名称
        replace_node_with_constant(
            gm,
            node,
            const_result[const_outputs[node.name]],
            new_const_name,
        )
        # 记录新常量名称与原节点名称对应的索引
        const_output_index[new_const_name] = const_outputs[node.name]

    # 遍历需要删除的节点列表（反向遍历）
    for node in to_erase_node[::-1]:
        # 如果节点有用户（即有依赖它的节点）
        if node.users:
            # 验证节点的所有用户的元信息中包含指定的模块标签
            for n in node.users:
                assert n.meta[META_TAG] == MODULE_TAG, f"node: {node} user not empty."
        else:
            # 如果节点没有用户，则从图中删除该节点
            gm.graph.erase_node(node)

    # 重新编译图模型
    gm.recompile()

    # 返回经过常量替换后的图模型和常量输出索引
    return const_gm, const_output_index
# 检查是否应用 TF32 警告到给定的 Torch FX 图模块
def is_tf32_warning_applicable(gm: torch.fx.GraphModule):
    # 导入 Torch 的 ATen 操作命名空间
    aten = torch.ops.aten
    # 定义 TF32 相关的操作集合
    tf32_ops = {
        aten.mm.default,
        aten.addmm.default,
        aten.bmm.default,
        aten.baddbmm.default,
    }
    # 遍历图中的每个操作节点
    for target in tf32_ops:
        for node in gm.graph.find_nodes(op="call_function", target=target):
            # 检查节点是否关联到 Torch Tensor，且数据类型为 torch.float32，设备为 CUDA
            if (
                isinstance(node.meta.get("val", None), torch.Tensor)
                and node.meta["val"].dtype == torch.float32
                and node.meta["val"].device.type == "cuda"
            ):
                return True  # 如果符合条件，返回 True
    return False  # 如果没有符合条件的节点，返回 False


# 如果输入示例中存在 CUDA 设备，则根据配置可能禁用全面填充
def maybe_disable_comprehensive_padding(example_inputs: List[torch.Tensor]):
    """
    For CPU backend, enable comprehensive padding causes some unit tests
    fail due to changing number of generated kernels. Skip for now.
    """
    # 检查示例输入中是否存在 CUDA 设备的 Tensor
    has_cuda = any(
        t.device.type == "cuda" for t in example_inputs if isinstance(t, torch.Tensor)
    )

    # 如果配置为启用全面填充，并且没有 CUDA 设备，则记录跳过全面填充的信息并返回新的配置
    if config.comprehensive_padding and not has_cuda:
        perf_hint_log.info("Skip comprehensive padding on CPU")
        return config.patch(comprehensive_padding=False)
    else:
        return contextlib.nullcontext()  # 否则返回一个空的上下文管理器


# 如果无法从输入上下文中检测到伪造模式，则创建一个
def fake_tensor_prop(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    force_allow_non_fake_inputs: bool = False,
):
    """
    If we can not detect fake mode from the context of inputs, create one.

    The created fake mode will be returned.
    """
    # 检测输入中的伪造模式
    fake_mode = detect_fake_mode(example_inputs)
    # 如果没有检测到伪造模式，则创建一个新的伪造模式
    if not fake_mode:
        fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
    else:
        # 否则，根据需要允许非伪造输入，在上下文中创建或不创建一个上下文
        ctx = (
            contextlib.nullcontext()
            if not force_allow_non_fake_inputs
            else mock.patch.object(fake_mode, "allow_non_fake_inputs", True)
        )
        with ctx:  # type: ignore[attr-defined]
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
                *example_inputs
            )

    return fake_mode  # 返回创建或检测到的伪造模式


# 检查是否应该使用远程 FX 图缓存
def should_use_remote_fx_graph_cache():
    # 如果配置中定义了 FX 图远程缓存，则返回其值
    if config.fx_graph_remote_cache is not None:
        return config.fx_graph_remote_cache
    # 如果不是在 FB 代码中，则返回 False
    if not config.is_fbcode():
        return False
    # 如果使用 HIP 版本的 Torch，则返回 False
    if torch.version.hip is not None:
        return False

    try:
        from triton.fb.fb_memcache import MEMCACHE_VERSION
    except ModuleNotFoundError:
        return False

    # 返回是否满足 FX 图远程缓存版本的条件
    return MEMCACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:fx_graph_memcache_version"
    )


# 将配置字典传递回用户
def get_patched_config_dict(config_patches=None) -> Dict[str, Any]:
    # 使用配置修补器，获取修补后的配置字典副本
    with config.patch(config_patches):
        return config.get_config_copy()


# 如果配置允许，则使用新的缓存进行处理
def with_fresh_cache_if_config(fn):
    @functools.wraps(fn)
    # 函数装饰器的声明
    # 定义一个装饰器函数 wrapper，接受任意位置参数 (*args) 和关键字参数 (**kwargs)
    def wrapper(*args, **kwargs):
        # 检查配置中是否强制禁用缓存
        if config.force_disable_caches:
            # 如果禁用缓存，则不删除缓存目录，因为缓存需要在 compile_fx 调用之后继续存在。
            # 让临时目录位于默认的缓存目录下，这样更容易定位它们。
            with fresh_inductor_cache(dir=cache_dir(), delete=False):
                # 调用被装饰的函数 fn，并返回其结果
                return fn(*args, **kwargs)
        else:
            # 如果未禁用缓存，则直接调用被装饰的函数 fn，并返回其结果
            return fn(*args, **kwargs)

    # 返回装饰器函数 wrapper
    return wrapper
@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
@time_and_log(attr="compilation time (in seconds)")
# 需要为 compile_fx_inner 添加此装饰器，即使已经为 compile_fx 添加了一个装饰器。
# 原因是在 compile_fx 返回后，反向图的编译可能会发生，我们可能希望也使用 _LazyGraphModule
# 来编译反向图。
@_use_lazy_graph_module(dynamo_config.use_lazy_graph_module)
# 根据配置，如果需要，使用新的缓存环境
@with_fresh_cache_if_config
# 在动态图构建期间计时，fwd_only=False 表示包括前向和反向图
@dynamo_utils.dynamo_timed(phase_name="inductor_compile", fwd_only=False)
def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs: Optional[BoxedBool] = None,
    static_input_idxs: Optional[List[int]] = None,
    is_backward: bool = False,
    graph_id: Optional[int] = None,
    cpp_wrapper: bool = False,
    aot_mode: bool = False,
    is_inference: bool = False,
    boxed_forward_device_index: Optional[BoxedDeviceIndex] = None,
    user_visible_outputs: Optional[Dict[str, None]] = None,
    layout_opt: Optional[bool] = None,
    extern_node_serializer: Optional[Callable[[List[ExternKernelNode]], Any]] = None,
) -> Union[CompiledFxGraph, str]:
    """
    Inductor API that compiles a single graph.

    If you change the argument list for this function, make sure you
    also update the call to save_args_for_compile_fx_inner below accordingly.
    """
    if dynamo_utils.count_calls(gm.graph) == 0 and not aot_mode:
        # 触发 _LazyGraphModule 在返回前重新编译 forward 方法。
        from torch.fx._lazy_graph_module import _LazyGraphModule

        _LazyGraphModule.force_recompile(gm)
        return make_boxed_func(gm.forward)

    if static_input_idxs is None:
        static_input_idxs = []

    # 确保 gm.graph 的最后一个节点的第一个参数是元组或列表
    assert isinstance(
        next(iter(reversed(gm.graph.nodes))).args[0], (tuple, list)
    ), f"inductor can only compile FX graphs which return a tuple/list, but got {gm.graph}"

    if config.save_args:
        # 如果配置允许，保存参数以供后续编译使用
        save_args_for_compile_fx_inner(
            gm,
            example_inputs,
            cudagraphs=cudagraphs,
            static_input_idxs=static_input_idxs,
            is_backward=is_backward,
            graph_id=graph_id,
            cpp_wrapper=cpp_wrapper,
            aot_mode=aot_mode,
            is_inference=is_inference,
            boxed_forward_device_index=boxed_forward_device_index,
            user_visible_outputs=user_visible_outputs,
            layout_opt=layout_opt,
        )

    if cudagraphs is None:
        # 根据配置决定是否使用 cudagraphs
        cudagraphs = BoxedBool(config.triton.cudagraphs)

    # fx_codegen_and_compile 的输入参数
    # 任何影响代码生成的内容都应在此处列出，因此如果 fx_codegen_and_compile 的签名变化，
    # 这个字典也需要相应更新
    # 构建包含各种参数的字典，用于图形编译和代码生成
    graph_kwargs = {
        "cudagraphs": cudagraphs,                      # 是否启用 CUDAGraph
        "static_input_idxs": static_input_idxs,        # 静态输入索引列表
        "is_backward": is_backward,                    # 是否为反向模式
        "graph_id": graph_id,                          # 图形标识符
        "cpp_wrapper": cpp_wrapper,                    # C++ 封装器
        "aot_mode": aot_mode,                          # AOT 编译模式
        "is_inference": is_inference,                  # 是否为推断模式
        "user_visible_outputs": user_visible_outputs,  # 用户可见的输出
        "layout_opt": layout_opt,                      # 布局优化选项
        "extern_node_serializer": extern_node_serializer,  # 外部节点序列化器
    }

    start = time.time()  # 记录开始时间

    # 检查是否应该使用远程 FX 图形缓存
    fx_graph_remote_cache = should_use_remote_fx_graph_cache()
    # 获取需要检查输入索引的示例输入
    inputs_to_check = get_input_idxs_to_check(example_inputs, static_input_idxs)

    # 如果未强制禁用缓存，并且满足缓存条件，则尝试加载 FX 图形缓存
    if (
        not config.force_disable_caches
        and (config.fx_graph_cache or fx_graph_remote_cache)
        and not aot_mode
    ):
        # 标记静态输入张量以便后续处理
        for i, input in enumerate(example_inputs):
            if (
                isinstance(input, torch.Tensor)
                and input.device.type == "cuda"
                and i in static_input_idxs
            ):
                input._is_inductor_static = True  # 标记静态输入张量属性

        # 加载 FX 图形缓存，传入相关参数
        compiled_graph = FxGraphCache.load(
            fx_codegen_and_compile,
            gm,
            example_inputs,
            graph_kwargs,
            inputs_to_check,
            local=config.fx_graph_cache,
            remote=fx_graph_remote_cache,
        )
    else:
        # 否则，进行 FX 代码生成和编译
        compiled_graph = fx_codegen_and_compile(
            gm, example_inputs, **graph_kwargs  # type: ignore[arg-type]
        )

    # 记录 FX 代码生成和编译所花费的时间
    log.debug("FX codegen and compilation took %.3fs", time.time() - start)

    # 检查从感应器降级中禁用 cudagraph 的原因
    if cudagraphs and compiled_graph.disabled_cudagraphs_reason:
        if "cuda" in compiled_graph.device_types:
            # 如果 CUDA 在受限设备类型中，则记录跳过 cudagraph 的原因
            log_cudagraph_skip_and_bump_counter(
                f"skipping cudagraphs due to {compiled_graph.disabled_cudagraphs_reason}"
            )
        else:
            # 否则增加 cudagraph 跳过计数器
            counters["inductor"]["cudagraph_skips"] += 1
        BoxedBool.disable(cudagraphs)

    # 通过 TracingContext 将输出步幅返回给调用者
    context = torch._guards.TracingContext.try_get()
    if context is not None and context.output_strides is not None:
        assert len(context.output_strides) == 0
        shape_env = _shape_env_from_inputs(example_inputs)
        for exprs in compiled_graph.output_strides:
            if exprs is None:
                context.output_strides.append(None)
            else:
                # 将符号表达式转换为实际值，如果环境可用的话
                context.output_strides.append(
                    tuple(
                        (
                            shape_env.evaluate_symexpr(e)
                            if shape_env is not None
                            else int(e)
                        )
                        for e in exprs
                    )
                )

    if aot_mode:
        return compiled_graph

    # 在 AOT 模式下直接返回编译后的图形对象
    # cudagraphs 会自行处理输入的对齐
    # 如果cudagraphs为空（或者False），执行以下操作
    if not cudagraphs:
        # 根据输入索引对齐输入，返回一个新的可调用对象
        new_callable = align_inputs_from_check_idxs(
            compiled_graph.current_callable, inputs_to_check
        )
        # 如果新的可调用对象不等于当前编译图的当前可调用对象
        if new_callable is not compiled_graph.current_callable:
            # 更新当前编译图的当前可调用对象为新的可调用对象
            compiled_graph.current_callable = new_callable

    # 调用_step_logger函数，记录编译过程的信息
    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    # 设置编译后图的_boxed_call属性为True，以便aot autograd知道要将输入作为列表传入
    compiled_graph._boxed_call = True
    # 返回编译后的图对象compiled_graph
    return compiled_graph
# 应用装饰器以保留随机数生成器状态
@dynamo_utils.preserve_rng_state()
# 定义函数 fx_codegen_and_compile，用于生成和编译 Torch FX 图模块
def fx_codegen_and_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs: Optional[BoxedBool] = None,
    static_input_idxs: Optional[List[int]] = None,
    is_backward: bool = False,
    graph_id: Optional[int] = None,
    cpp_wrapper: bool = False,
    aot_mode: bool = False,
    is_inference: bool = False,
    # 使用带有 None 值的字典而不是集合，以保证迭代顺序是确定的
    user_visible_outputs: Optional[Dict[str, None]] = None,
    layout_opt: Optional[bool] = None,
    extern_node_serializer: Optional[Callable[[List[ExternKernelNode]], Any]] = None,
) -> Union[CompiledFxGraph, str]:
    # 如果适用，检查是否需要警告 TF32 禁用
    if is_tf32_warning_applicable(gm):
        _warn_tf32_disabled()

    # 提升 Python 解释器栈的最大深度，以适应大型/深层模型
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    # 记录编译步骤日志信息，包括编译方向（正向或反向）和图标识
    _step_logger()(
        logging.INFO,
        "torchinductor compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )
    # 调试输出 Torch FX 图模块及示例输入
    V.debug.fx_graph(gm, example_inputs)

    # TODO: 我们是否真的应该转储这个？这应该是与 AOT 结构化日志重复的...
    # trace_structured("inductor_input_graph", payload_fn=lambda: gm.print_readable(print_output=False))

    # 根据示例输入构建形状环境
    shape_env = _shape_env_from_inputs(example_inputs)

    # 将图中的视图操作转换为重塑操作，主要用于布局优化
    view_to_reshape(gm)

    # 在无梯度计算环境下运行 FakeTensorProp，因为假设 AOTAutograd 已经处理了自动求导
    with torch.no_grad():
        fake_mode = fake_tensor_prop(gm, example_inputs)

    # 模式匹配器可能不会在 node.meta["val"] 上保留步进信息
    # 如果将来我们依赖这些信息正确性，则需要修复
    # on node.meta["val"]. if in the future we rely on these being
    # correct we will need to fix.
    # 使用虚拟模式设置 V 对象，用于处理一些内存问题的情况
    with V.set_fake_mode(fake_mode):
        # 执行递归的后向梯度传递操作，可能用于推断过程中
        _recursive_post_grad_passes(gm, is_inference=is_inference)
        # 在调试模式下，打印转换后的图形及示例输入
        V.debug.fx_graph_transformed(gm, example_inputs)
        # 记录后向梯度操作后的图形代码，包括步长和设备信息，以彩色显示
        post_grad_graphs_log.debug(
            "%s",
            lazy_format_graph_code(
                "AFTER POST GRAD",
                gm,
                include_stride=True,
                include_device=True,
                colored=True,
            ),
        )
        # 跟踪结构化信息，记录后向梯度后的图形结构
        trace_structured(
            "inductor_post_grad_graph",
            payload_fn=lambda: gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )
        # 如果配置为 FB Code 环境，将优化信息记录到 Scuba
        if config.is_fbcode():
            log_optimus_to_scuba(
                extra_logging={"pt2_configs": str(get_patched_config_dict())}
            )

    # 使用虚拟模式设置 V 对象，并可能禁用全面填充，根据示例输入
    with V.set_fake_mode(fake_mode), maybe_disable_comprehensive_padding(
        example_inputs
    ):
        # 返回编译后的图形对象
        return compiled_graph
# 根据输入张量 x 创建一个克隆张量，保持相同的步长信息
def clone_preserve_strides(x: torch.Tensor):
    # 计算需要的缓冲区大小，以保持张量步长信息不变
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
    )
    # 使用 as_strided 创建步长信息保持的缓冲区，并进行克隆操作
    buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
    return torch.as_strided(buffer, x.size(), x.stride())

# 复制新输入张量列表中索引为 check_inputs_idxs 的张量，确保其对齐要求
def copy_misaligned_inputs(
    new_inputs: List[torch.Tensor], check_inputs_idxs: Sequence[int]
) -> None:
    for i in check_inputs_idxs:
        # 检查数据指针是否满足对齐要求，若不满足则进行克隆操作
        if new_inputs[i].data_ptr() % ALIGNMENT:
            new_inputs[i] = clone_preserve_strides(new_inputs[i])

# 获取需要检查的输入张量索引列表，用于在编译时生成可能需要复制的索引列表，以保持对齐要求
def get_input_idxs_to_check(
    inputs: Union[List[torch.Tensor], Sequence[int]],
    static_input_idxs: Sequence[int],
) -> Sequence[int]:
    """
    This function runs at compile time, and generates a list of indices for which we
    might need to do a copy to preserve alignment requirements.
    """
    ids_to_check = []

    for i, input in enumerate(inputs):
        if not isinstance(input, torch.Tensor):
            # 非张量不需要对齐
            continue
        if input.device.type != "cuda":
            # 目前仅关心 CUDA 张量的对齐
            continue
        with maybe_get_suppress_shape_guards_ctx():
            # 临时关闭形状保护，以防止 tensor_is_aligned 和 should_assume_input_aligned
            # 在输入的存储偏移上添加保护
            if i in static_input_idxs and tensor_is_aligned(input):
                continue
            if not should_assume_input_aligned(input):
                continue

        # 如果程序执行到这里，
        # (a) 我们的 Triton 代码假定输入已对齐
        # (b) 我们事先无法确定输入是否实际上已对齐。
        # 因此，在运行时，我们需要检查输入是否对齐（如果不对齐，则克隆它以使其对齐）。
        ids_to_check.append(i)

    return ids_to_check

# 从检查索引列表 inputs_to_check 对输入进行对齐处理
def align_inputs_from_check_idxs(
    model: Callable[[List[torch.Tensor]], Any], inputs_to_check: Sequence[int]
):
    if len(inputs_to_check) == 0:
        return model

    def run(new_inputs):
        # 复制不对齐的输入张量以保持对齐
        copy_misaligned_inputs(new_inputs, inputs_to_check)
        # 调用模型处理对齐后的输入
        return model(new_inputs)

    return run

# cudagraphify 函数用于将 Torch FX 图模块转换为 CUDA 图形表示
@dynamo_utils.dynamo_timed
def cudagraphify(
    model: torch.fx.GraphModule,
    inputs: List[torch.Tensor],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    stack_traces: List[Optional[str]],
    is_backward: bool,
    is_inference: bool,
    constants: Tuple[torch.Tensor, ...] = (),
    placeholders: Tuple[torch.fx.Node, ...] = (),
    mutated_input_idxs: Tuple[int, ...] = (),
):
    from torch._inductor.cudagraph_trees import (
        cudagraphify_impl as new_cudagraphify_impl,
    )

    cudagraphify_fn: Callable[..., Any]
    # 如果配置中指定了使用 Triton 的 cudagraph_trees
    if config.triton.cudagraph_trees:
        # 使用 functools.partial 创建一个新的 cudagraphify_fn 函数，
        # 该函数绑定了一些参数，其中包括设备索引、堆栈跟踪信息、是否为反向传播、
        # 是否为推理模式、常量、占位符、以及变异输入的索引。
        cudagraphify_fn = functools.partial(
            new_cudagraphify_impl,
            device_index=device_index,
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=constants,
            placeholders=placeholders,
            mutated_input_idxs=mutated_input_idxs,
        )
    else:
        # 如果不使用 cudagraph_trees，则直接将 cudagraphify_fn 设置为 cudagraphify_impl 函数。
        cudagraphify_fn = cudagraphify_impl

    # 如果输入中没有任何 FakeTensor 类型的输入，则直接调用 cudagraphify_fn 处理模型和输入。
    if not any(isinstance(inp, FakeTensor) for inp in inputs):
        return cudagraphify_fn(model, inputs, static_input_idxs)

    # 初始化 compiled_fn 为 None
    compiled_fn = None

    # 定义内部函数 run，接收新的输入 new_inputs
    def run(new_inputs):
        nonlocal compiled_fn
        # 如果 compiled_fn 还未初始化
        if compiled_fn is None:
            # 使用 preserve_rng_state 上下文管理器保存随机数生成器状态，
            # 并调用 cudagraphify_fn 生成编译后的函数
            with dynamo_utils.preserve_rng_state():
                compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
        # 返回编译后的函数对新输入 new_inputs 的计算结果
        return compiled_fn(new_inputs)

    # 返回内部定义的 run 函数，用于运行编译后的计算图
    return run
def remove_unaligned_input_idxs(
    inputs: Union[List[torch.Tensor], Sequence[int]],
    static_input_idxs: Sequence[int],
):
    """
    We require all inputs to be aligned, so introduce a copy for any
    that aren't.
    """
    # 创建一个空列表，用于存放与静态输入索引对齐的索引列表
    aligned_static_input_idxs = []
    # 遍历静态输入索引及对应的输入列表
    for idx, input in zip(static_input_idxs, inputs):
        # 检查输入是否为Tensor并且内存地址对齐
        if isinstance(input, torch.Tensor) and (input.data_ptr() % ALIGNMENT) == 0:
            # 如果对齐，将索引加入到对齐的静态输入索引列表中
            aligned_static_input_idxs.append(idx)
    # 如果对齐的静态输入索引数量不等于静态输入索引的总数，返回对齐的静态输入索引列表
    if len(aligned_static_input_idxs) != len(static_input_idxs):
        return aligned_static_input_idxs
    # 否则返回原始的静态输入索引列表
    return static_input_idxs


def static_input(x: torch.Tensor):
    """
    Copy and input while preserving strides
    """
    # 计算所需缓冲区大小，以保留原始Tensor的步幅
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
    )
    # 创建一个空缓冲区，与原始Tensor具有相同的数据类型和设备
    buffer = torch.empty(needed_size, dtype=x.dtype, device=x.device)
    # 使用as_strided方法创建具有相同大小和步幅的Tensor，并返回
    return torch.as_strided(buffer, x.size(), x.stride())


def index_expanded_dims_and_copy_(
    dst: torch.Tensor,
    src: torch.Tensor,
    expanded_dims: List[int],
):
    "Index into expanded dimensions of both dst and src then copy_"
    # 在扩展维度上索引目标Tensor和源Tensor，然后进行复制操作
    dst = index_expanded_dims(dst, expanded_dims)
    src = index_expanded_dims(src, expanded_dims)
    # 使用copy_方法将src的数据复制到dst中
    dst.copy_(src)


def cudagraphify_impl(
    model: torch.fx.GraphModule,
    inputs: List[torch.Tensor],
    static_input_idxs: Sequence[int] = (),
):
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    # 获取需要检查的输入索引列表
    check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
    # 移除不对齐的输入索引
    static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
    # 拷贝不对齐的输入
    copy_misaligned_inputs(inputs, check_input_idxs)

    # 断言确保输入是一个列表
    assert isinstance(inputs, list)

    # 获取每个输入的扩展维度列表，如果索引在静态输入索引列表中则为空列表
    inps_expanded_dims = [
        get_expanded_dims(x) if idx not in static_input_idxs else []
        for idx, x in enumerate(inputs)
    ]

    # 分配静态张量输入
    static_inputs = [
        x
        if not isinstance(x, torch.Tensor)
        else static_input(x)
        if idx not in static_input_idxs
        else x.detach()
        for idx, x in enumerate(inputs)
    ]

    # 复制新分配的输入值
    for idx, (x, expanded_dims) in enumerate(zip(inputs, inps_expanded_dims)):
        if isinstance(x, torch.Tensor) and idx not in static_input_idxs:
            index_expanded_dims_and_copy_(static_inputs[idx], x, expanded_dims)

    # 预热
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # 复制static_inputs因为它将在模型中被清除
    with torch.cuda.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # 记录
    graph = torch.cuda.CUDAGraph()
    # 使用 torch.cuda.graph 函数来创建一个 GPU 计算图，用指定的参数
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        # 使用模型对静态输入列表进行推理，得到静态输出
        static_outputs = model(list(static_inputs))
    # 如果静态输出不是列表或元组，将其转换为元组
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    # 如果配置中包含 size_asserts 选项
    if config.size_asserts:
        
        # 定义一个函数 run，用于对新输入进行处理并验证与静态输入的匹配关系
        def run(new_inputs):
            # 断言静态输入和新输入的长度相等
            assert len(static_inputs) == len(new_inputs)
            # 遍历静态输入、新输入和扩展维度的元组
            for idx, (dst, src, expanded_dims) in enumerate(
                zip(static_inputs, new_inputs, inps_expanded_dims)
            ):
                # 如果目标不是 torch.Tensor 类型，则跳过
                if not isinstance(dst, torch.Tensor):
                    pass
                # 如果索引在静态输入索引集合中
                elif idx in static_input_idxs:
                    # 断言目标张量的数据指针与源张量的数据指针相等
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    # 否则，使用索引和扩展维度来拷贝数据到目标张量
                    index_expanded_dims_and_copy_(dst, src, expanded_dims)
            # 清空新输入列表
            new_inputs.clear()
            # 回放 GPU 计算图
            graph.replay()
            # 返回静态输出
            return static_outputs

    # 如果配置中不包含 size_asserts 选项
    else:
        # 创建一个复制索引列表，包含不在静态输入索引集合中的索引
        copy_indices = [
            idx for idx in range(len(static_inputs)) if idx not in static_input_idxs
        ]

        # 定义一个函数 run，用于对新输入进行处理并执行数据拷贝操作
        def run(new_inputs):
            # 遍历复制索引列表
            for idx in copy_indices:
                # 获取对应索引处的扩展维度
                expanded_dims = inps_expanded_dims[idx]
                # 使用索引和扩展维度来拷贝数据到静态输入张量
                index_expanded_dims_and_copy_(
                    static_inputs[idx], new_inputs[idx], expanded_dims
                )
            # 清空新输入列表
            new_inputs.clear()
            # 回放 GPU 计算图
            graph.replay()
            # 返回静态输出
            return static_outputs

    # 返回根据检查输入索引对齐的输入结果
    return align_inputs_from_check_idxs(run, check_input_idxs)
# 编译 TorchScript 模型为 Ahead-of-Time (AOT) 编译库，并返回编译后的库路径
def compile_fx_aot(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    inner_compile: Callable[..., Any] = compile_fx_inner,
    config_patches: Optional[Dict[str, Any]] = None,
):
    # 如果未提供 config_patches 参数，则设置默认值 {"cpp_wrapper": True}
    config_patches: Dict[str, Any] = (
        {"cpp_wrapper": True}
        if config_patches is None
        else {**config_patches, "cpp_wrapper": True}
    )

    # 如果配置中没有指定 "aot_inductor.output_path"，则使用模型代码的哈希值作为输出路径
    if (
        "aot_inductor.output_path" not in config_patches
        and not config.aot_inductor.output_path
    ):
        config_patches = {
            **config_patches,
            "aot_inductor.output_path": code_hash(model_.code),
        }

    # 从 config_patches 中移除 "extern_node_serializer" 参数并获取其值
    extern_node_serializer = config_patches.pop("extern_node_serializer", None)

    # 设置 AOT 编译为 True 并调用 compile_fx 进行模型编译
    with V.set_aot_compilation(True):
        compiled_lib_path = compile_fx(
            model_,
            example_inputs_,
            inner_compile=functools.partial(
                inner_compile,
                aot_mode=True,
                extern_node_serializer=extern_node_serializer,
            ),
            config_patches=config_patches,
        )

        # 检查编译后的库路径是否存在
        assert os.path.exists(
            compiled_lib_path
        ), f"AOTInductor compiled library does not exist at {compiled_lib_path}"
        
        # 返回编译后的库路径
        return compiled_lib_path


# 用于生成图的计数器，从 0 开始
_graph_counter = count(0)


# 对 AOT 自动求导模型进行前向编译
def fw_compiler_freezing(
    aot_autograd_model: torch.fx.GraphModule,
    aot_example_inputs: List[torch.Tensor],
    dynamo_model: torch.fx.GraphModule,
    num_example_inputs: int,
    inner_compile: Callable[..., Any],
    cudagraphs: BoxedBool,
    graph_id: int,
    forward_device: BoxedDeviceIndex,
):
    # 导入模型冻结所需的函数
    from torch._inductor.freezing import convert_conv_weights_to_channels_last, freeze

    # 调用递归函数 _recursive_joint_graph_passes 处理 AOT 自动求导模型
    # 注意：此函数的具体实现细节未在提供的代码中显示
    _recursive_joint_graph_passes(aot_autograd_model)

    # 根据推理模式确定图的布局优化选项
    layout_opt = GraphLowering.decide_layout_opt(aot_autograd_model, is_inference=True)
    if layout_opt:
        # 确保 meta['val'] 被正确设置
        fake_tensor_prop(aot_autograd_model, aot_example_inputs, True)
        # 将卷积权重转换为 channels_last 格式
        convert_conv_weights_to_channels_last(aot_autograd_model)

    # 对动态模型进行冻结操作，返回优化后的模型及保留参数的索引
    opt_model, preserved_arg_indices = freeze(
        dynamo_model,
        aot_autograd_model,
        aot_example_inputs,  # type: ignore[arg-type]
    )

    # 更新 aot_example_inputs 为冻结后仍需保留的输入参数
    aot_example_inputs = [aot_example_inputs[ind] for ind in preserved_arg_indices]

    # 计算固定参数的数量
    num_fixed = len(preserved_arg_indices) - num_example_inputs

    # 检测是否处于伪装模式
    fake_mode = detect_fake_mode(aot_example_inputs)

    # 确保所有图输出均为用户可见
    *_, model_outputs_node = opt_model.graph.nodes
    model_outputs = model_outputs_node.args[0]
    user_visible_outputs = dict.fromkeys(
        n.name for n in model_outputs if isinstance(n, torch.fx.Node)
    )

    # 创建静态输入参数的索引列表
    static_input_idxs = list(range(num_fixed))
    # 常量参数将是真实张量，而非伪装张量
    tracing_context = torch._guards.TracingContext.try_get()
    # 如果跟踪上下文不为 None，则进行以下操作
    if tracing_context is not None:
        # 获取跟踪上下文中的扁平化参数列表
        params_flat = tracing_context.params_flat
        # 断言确保扁平化参数列表不为 None
        assert params_flat is not None
        # 遍历扁平化参数列表的索引
        for i in range(len(params_flat)):
            # 如果索引 i 不在保留参数索引列表中，则将其置为 None
            if i not in preserved_arg_indices:
                params_flat[i] = None

        # 如果跟踪上下文中存在前向元数据
        if tracing_context.fw_metadata:
            # 将静态参数索引添加到静态输入索引列表中
            static_input_idxs += tracing_context.fw_metadata.static_parameter_indices

    # 使用 mock.patch.object() 方法将 fake_mode 对象中的 allow_non_fake_inputs 属性设置为 True
    with mock.patch.object(fake_mode, "allow_non_fake_inputs", True):
        # 调用 inner_compile 函数进行优化编译
        optimized_function = inner_compile(
            opt_model,
            aot_example_inputs,
            static_input_idxs=static_input_idxs,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
            is_inference=True,
            boxed_forward_device_index=forward_device,
            layout_opt=layout_opt,
            user_visible_outputs=user_visible_outputs,
        )

    # 如果启用了 AOT 编译，则直接返回优化后的函数
    if V.aot_compilation is True:
        return optimized_function

    # 定义一个包装函数 wrapper，接受参数 args
    def wrapper(args):
        # 根据保留的参数索引从参数列表 args 中提取参数 args_new
        args_new = [args[i] for i in preserved_arg_indices]
        # 清空参数列表 args
        args.clear()
        # 调用优化后的函数 optimized_function，并传入 args_new 作为参数
        return optimized_function(args_new)

    # 将 wrapper 函数标记为支持 boxed 调用，用于类型提示
    wrapper._boxed_call = True  # type: ignore[attr-defined]

    # 返回包装函数 wrapper
    return wrapper
# 使用装饰器 @_use_lazy_graph_module，并传入 dynamo_config.use_lazy_graph_module 参数，延迟加载图模块
@_use_lazy_graph_module(dynamo_config.use_lazy_graph_module)
# 定义编译函数 compile_fx，接受以下参数:
#   - model_: torch.fx.GraphModule，表示待编译的FX图模块
#   - example_inputs_: List[torch.Tensor]，包含示例输入张量的列表
#   - inner_compile: Callable[..., Any]，默认为 compile_fx_inner，用于内部编译
#   - config_patches: Optional[Dict[str, Any]]，可选的配置补丁字典
#   - decompositions: Optional[Dict[OpOverload, Callable[..., Any]]]，可选的分解函数字典
def compile_fx(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    inner_compile: Callable[..., Any] = compile_fx_inner,
    config_patches: Optional[Dict[str, Any]] = None,
    decompositions: Optional[Dict[OpOverload, Callable[..., Any]]] = None,
):
    """Main entrypoint to a compile given FX graph"""
    # 如果存在 config_patches，则使用该配置补丁进行环境设置
    if config_patches:
        with config.patch(config_patches):
            # 递归调用 compile_fx，传入相同的参数和更新的 inner_compile
            return compile_fx(
                model_,
                example_inputs_,
                inner_compile=config.patch(config_patches)(inner_compile),
                decompositions=decompositions,
            )

    # 如果 config.cpp_wrapper 为真，执行以下代码块
    if config.cpp_wrapper:
        with config.patch(
            {
                "cpp_wrapper": False,
                # 对于 triton.autotune_at_compile_time，在 FBCode 中默认禁用，在 OSS 中默认启用
                "triton.autotune_at_compile_time": config.triton.autotune_at_compile_time
                if config.is_fbcode()
                else os.environ.get(
                    "TORCHINDUCTOR_TRITON_AUTOTUNE_AT_COMPILE_TIME", "1"
                )
                == "1",
                "triton.autotune_cublasLt": False,
                "triton.cudagraphs": False,
                "triton.store_cubin": True,
            }
        ), V.set_real_inputs(example_inputs_):
            # 设置 inputs_ 为 example_inputs_
            inputs_ = example_inputs_
            # 如果 model_ 是 torch.fx.GraphModule 类型，处理虚拟输入节点
            if isinstance(model_, torch.fx.GraphModule):
                # 提取所有操作为 "placeholder" 的节点的 meta 中的值作为 fake_inputs
                fake_inputs = [
                    node.meta.get("val")
                    for node in model_.graph.nodes
                    if node.op == "placeholder"
                ]
                # 如果所有的 fake_inputs 都不为 None，则进行设备验证
                if all(v is not None for v in fake_inputs):
                    # 验证设备匹配性，确保 fake_inputs 和 inputs_ 的设备一致
                    for idx, fi, i in zip(count(), fake_inputs, inputs_):
                        if fi.device != i.device:
                            raise ValueError(
                                f"Device mismatch between fake input and example input at position #{idx}: "
                                f"{fi.device} vs {i.device}. If the model was exported via torch.export(), "
                                "make sure torch.export() and torch.aot_compile() run on the same device."
                            )
                    # 将 inputs_ 更新为 fake_inputs
                    inputs_ = fake_inputs
            # 递归调用 compile_fx，传入更新的参数和 partial 函数 inner_compile
            return compile_fx(
                model_,
                inputs_,
                inner_compile=functools.partial(inner_compile, cpp_wrapper=True),
                decompositions=decompositions,
            )

    # 定义递归编译函数 recursive_compile_fx，使用 partial 函数固定 inner_compile 和 decompositions 参数
    recursive_compile_fx = functools.partial(
        compile_fx,
        inner_compile=inner_compile,
        decompositions=decompositions,
    )
    # 如果模型不返回元组，则转换成返回元组
    if not graph_returns_tuple(model_):
        return make_graph_return_tuple(
            model_,
            example_inputs_,
            recursive_compile_fx,
        )

    # 如果模型是 torch.fx.GraphModule 类型
    if isinstance(model_, torch.fx.GraphModule):
        # 如果模型的图是通过 dynamo.export() 导出的结果
        if isinstance(model_.graph._codegen, _PyTreeCodeGen):
            return handle_dynamo_export_graph(
                model_,
                example_inputs_,
                recursive_compile_fx,
            )

        # 对模型进行前向传播前的递归处理
        model_ = _recursive_pre_grad_passes(model_, example_inputs_)

    # 如果 example_inputs_ 中有任何元素是列表、元组或字典类型，则扁平化输入图
    if any(isinstance(x, (list, tuple, dict)) for x in example_inputs_):
        return flatten_graph_inputs(
            model_,
            example_inputs_,
            recursive_compile_fx,
        )

    # 断言：config._raise_error_for_testing 应为假
    assert not config._raise_error_for_testing

    # 计算 example_inputs_ 的数量
    num_example_inputs = len(example_inputs_)

    # 从 config.triton.cudagraphs 中获取 boxed 布尔值
    cudagraphs = BoxedBool(config.triton.cudagraphs)

    # 初始化 forward_device 为 None 的 boxed 设备索引
    forward_device = BoxedDeviceIndex(None)

    # 生成下一个图的唯一标识符
    graph_id = next(_graph_counter)

    # 如果 decompositions 不为 None，则使用它；否则选择默认的 decompositions 表
    decompositions = (
        decompositions if decompositions is not None else select_decomp_table()
    )

    # 定义一个被 dynamo_utils.dynamo_timed 装饰的函数 fw_compiler_base
    @dynamo_utils.dynamo_timed
    def fw_compiler_base(
        model: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        is_inference: bool,
    ):
        # 基于 is_inference 参数的函数调用
        fw_compiler = functools.partial(fw_compiler_base, is_inference=False)

    # 如果启用 freezing 并且梯度未开启
    if config.freezing and not torch.is_grad_enabled():
        # 定义一个推断编译器函数，用于冻结模型
        inference_compiler = functools.partial(
            fw_compiler_freezing,
            dynamo_model=model_,
            num_example_inputs=num_example_inputs,
            inner_compile=inner_compile,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
            forward_device=forward_device,
        )
    else:
        # 定义一个推断编译器函数，用于正常推断
        inference_compiler = functools.partial(fw_compiler_base, is_inference=True)

    # 定义一个分区函数 partition_fn，用于图的分区和重组
    def partition_fn(graph, joint_inputs, **kwargs):
        # 递归地对图进行联合处理
        _recursive_joint_graph_passes(graph)
        # 使用最小割重组分区图
        return min_cut_rematerialization_partition(
            graph, joint_inputs, **kwargs, compiler="inductor"
        )

    # 定义一个被 compile_time_strobelight_meta 装饰和 dynamo_utils.dynamo_timed 装饰的反向编译器函数 bw_compiler
    @compile_time_strobelight_meta(phase_name="bw_compiler")
    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        # 用户可见的输出节点
        user_visible_outputs = {}

        # 如果配置允许用户可见的反向输出
        if config.bw_outputs_user_visible:
            # 获取模型输出节点
            model_outputs_node = output_node(model)
            # 提取所有叶子节点
            model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
            # 创建用户可见输出的字典
            user_visible_outputs = dict.fromkeys(
                n.name for n in model_outputs if isinstance(n, torch.fx.Node)
            )
        
        # 计算固定的张量数量
        fixed = count_tangents(model)
        # 调用内部编译函数进行反向编译
        return inner_compile(
            model,
            example_inputs,
            static_input_idxs=list(range(fixed)),
            cudagraphs=cudagraphs,
            is_backward=True,
            graph_id=graph_id,
            boxed_forward_device_index=forward_device,
            user_visible_outputs=user_visible_outputs,
        )
    # 在调用 create_aot_dispatcher_function 之前/之后添加日志记录的TODO项
    # 在 torch._functorch/aot_autograd.py::aot_module_simplified::aot_function_simplified::new_func 中完成
    # 一旦 torchdynamo 合并到 pytorch 中

    # 检测是否存在 FakeTensorMode，根据 example_inputs_ 判断
    fake_mode = detect_fake_mode(example_inputs_) or torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True
    )

    # 获取或创建一个 TracingContext，使用 fake_mode 作为参数
    tracing_context = (
        torch._guards.TracingContext.try_get()
        or torch._guards.TracingContext(fake_mode)
    )

    # 如果 V.aot_compilation 为 True，则进行 AOT 编译
    if V.aot_compilation is True:
        # 使用 functorch_config.patch(unlift_effect_tokens=True) 上下文
        with functorch_config.patch(unlift_effect_tokens=True):
            # 对模型进行 AOT 导出，得到图模型 gm 和图签名 graph_signature
            gm, graph_signature = aot_export_module(
                model_,
                example_inputs_,
                trace_joint=False,
                decompositions=decompositions,
            )
        # 对图模型进行解卷积操作，得到 unlifted_gm
        unlifted_gm = _unlift_graph(model_, gm, graph_signature)

        # 如果模型元数据中包含 "dynamo_flat_name_to_original_fqn"，则将其复制到 unlifted_gm 中
        if "dynamo_flat_name_to_original_fqn" in model_.meta:
            unlifted_gm.meta["dynamo_flat_name_to_original_fqn"] = model_.meta[
                "dynamo_flat_name_to_original_fqn"
            ]

        # 禁用 amp（自动混合精度）
        disable_amp = torch._C._is_any_autocast_enabled()
        # 如果 amp 启用，则使用 torch._C._DisableAutocast 上下文
        context = torch._C._DisableAutocast if disable_amp else contextlib.nullcontext
        # 使用 fake_mode，禁用 compiled_autograd，禁用 amp 上下文，调用 inference_compiler 函数
        with V.set_fake_mode(fake_mode), compiled_autograd.disable(), context():
            return inference_compiler(unlifted_gm, example_inputs_)

    # 如果不是 AOT 编译，则执行普通的 autograd
    # 使用 fake_mode，tracing_context，禁用 compiled_autograd，使用 functorch_config.patch(unlift_effect_tokens=True) 上下文
    with V.set_fake_mode(fake_mode), torch._guards.tracing(
        tracing_context
    ), compiled_autograd.disable(), functorch_config.patch(unlift_effect_tokens=True):
        # 调用 aot_autograd 函数，传递各种编译器和参数
        return aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            inference_compiler=inference_compiler,
            decompositions=decompositions,
            partition_fn=partition_fn,
            keep_inference_input_mutations=True,
        )(model_, example_inputs_)
def _shape_env_from_inputs(inputs: List[torch.Tensor]):
    # 初始化 shape_env 为 None
    shape_env = None
    # 检测输入是否为伪造模式
    fake_mode = detect_fake_mode(inputs)

    # TODO(voz): It would be nice to enable this assert, but there are lots of tests that
    # pass in real inputs for now.
    # 如果输入列表不为空，希望启用断言，但目前有许多测试用例传入真实输入。
    # if len(inputs) > 0:
    # assert fake_mode is not None, breakpoint()

    # 如果检测到伪造模式，则直接返回其 shape_env
    if fake_mode is not None:
        return fake_mode.shape_env

    # 当没有伪造模式时，遍历输入列表，找到第一个 torch.SymInt 类型的输入，
    # 并返回其对应的 shape_env
    for input in inputs:
        if isinstance(input, torch.SymInt):
            return input.node.shape_env

    # TODO(voz): Should we always have one anyway?
    # 如果没有找到符合条件的输入，则返回 None
    return None


def graph_returns_tuple(gm: torch.fx.GraphModule):
    """True if a FX graph returns a tuple"""
    # 如果 gm 不是 torch.fx.GraphModule 类型，则默认返回 True，因为无法检查
    if not isinstance(gm, torch.fx.GraphModule):
        return True  # can't check this, assume true

    # 获取输出节点的返回值
    (rv,) = output_node(gm).args

    # 如果返回值是 list 或 tuple 类型，则返回 True
    if isinstance(rv, (list, tuple)):
        return True

    # 如果返回值是 torch.fx.node.Node 类型，并且具有多个返回值，
    # 并且所有返回值类型都是 "Tensor"，则返回 True
    if (
        isinstance(rv, torch.fx.node.Node)
        and hasattr(rv.target, "_schema")
        and len(rv.target._schema.returns) > 1
        and all(str(ret.type) == "Tensor" for ret in rv.target._schema.returns)
    ):
        return True

    # 否则返回 False
    return False


def make_graph_return_tuple(
    gm: torch.fx.GraphModule,
    inputs: List[torch.Tensor],
    compile_gm: Callable[..., Any],
):
    """
    Mutate gm so it returns a tuple.  This is only needed for graphs
    not created by torchdynamo that return non-tuples.
    """
    # 获取输出节点
    node = output_node(gm)
    # 解包返回值
    (rv,) = node.args
    # 对返回值进行扁平化处理
    rv, spec = pytree.tree_flatten(rv)
    # 在输出节点之前插入新的输出
    with gm.graph.inserting_before(node):
        gm.graph.output(rv)
    # 删除原输出节点
    gm.graph.erase_node(node)
    # 断言图现在返回一个元组
    assert graph_returns_tuple(gm)

    # 编译修改后的图模型
    compiled_fn = compile_gm(gm, inputs)

    @functools.wraps(compiled_fn)
    def wrapper(*args, **kwargs):
        return pytree.tree_unflatten(compiled_fn(*args, **kwargs), spec)

    # 返回包装后的函数
    return wrapper


def handle_dynamo_export_graph(
    gm: torch.fx.GraphModule,
    inputs: List[torch.Tensor],
    compile_gm: Callable[..., Any],
):
    """
    `torch._dynamo.export` embeds pytrees in the FX graph codegen object,
    convert that to a normal FX graph so inductor can compile it.
    """
    # 获取图模型的代码生成器对象
    codegen = gm.graph._codegen
    # 用新的代码生成器替换图模型中的原代码生成器
    gm.graph._codegen = torch.fx.graph.CodeGen()
    # 重新编译图模型
    gm.recompile()

    # 编译修改后的图模型
    compiled_fn = compile_gm(gm, codegen.process_inputs(*inputs))

    @functools.wraps(compiled_fn)
    def wrapper(*args):
        # 处理编译后的输出
        return codegen.process_outputs(compiled_fn(*codegen.process_inputs(*args)))

    # 返回包装后的函数
    return wrapper


def _check_triton_bf16_support(graph: GraphLowering) -> None:
    def warn_and_skip(device) -> None:
        from torch._dynamo.exc import SkipFrame

        # 获取设备的属性信息
        device_props = torch.cuda.get_device_properties(device)
        # 发出警告，指出设备不支持 bfloat16 编译
        warnings.warn(
            f"{device_props.name} does not support bfloat16 compilation natively, skipping"
        )
        # 抛出 SkipFrame 异常，表示不支持 BF16
        raise SkipFrame("BF16 is not supported")
    # 遍历计算图的输入节点
    for inp in graph.graph_inputs.values():
        # 获取输入节点的设备，如果不存在则使用默认"meta"设备
        device = getattr(inp, "get_device", lambda: torch.device("meta"))()
        # 如果设备不是CUDA或者数据类型不是torch.bfloat16，则跳过当前节点
        if device.type != "cuda" or inp.get_dtype() != torch.bfloat16:
            continue
        # 如果当前设备不支持torch.bfloat16（包括仿真），则返回
        if torch.cuda.is_bf16_supported(including_emulation=False):
            return
        # 发出警告并跳过当前帧
        warn_and_skip(device)

    # 遍历计算图的输出节点
    for out in graph.graph_outputs:
        # 获取输出节点的设备，如果不存在则使用默认"meta"设备
        device = getattr(out, "get_device", lambda: torch.device("meta"))()
        # 如果设备不是CUDA或者数据类型不是torch.bfloat16，则跳过当前节点
        if device.type != "cuda" or out.get_dtype() != torch.bfloat16:
            continue
        # 如果当前设备不支持torch.bfloat16（包括仿真），则返回
        if torch.cuda.is_bf16_supported(including_emulation=False):
            return
        # 发出警告并跳过当前帧
        warn_and_skip(device)
```