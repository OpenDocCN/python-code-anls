# `.\pytorch\torch\_inductor\fx_passes\post_grad.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import functools  # 导入 functools 模块，用于高阶函数操作
import itertools  # 导入 itertools 模块，用于迭代器操作
import logging  # 导入 logging 模块，用于日志记录
import operator  # 导入 operator 模块，提供了函数形式的标准运算符功能
from collections import Counter, defaultdict  # 导入 Counter 和 defaultdict 类，用于计数和默认字典操作
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, Union  # 导入类型相关的模块

import torch  # 导入 PyTorch 模块
import torch._inductor as inductor  # 导入 PyTorch 内部的 inductor 模块
import torch.utils._pytree as pytree  # 导入 PyTorch 内部的 _pytree 模块
from torch import fx  # 从 PyTorch 中导入 fx 模块
from torch._decomp import register_decomposition  # 从 PyTorch 中导入 register_decomposition 函数
from torch._dynamo.utils import counters, optimus_scuba_log  # 导入 PyTorch 内部的 counters 和 optimus_scuba_log 函数
from torch._inductor.virtualized import ops  # 导入 PyTorch 内部的 ops 模块

from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype  # 导入 PyTorch 内部的类型判断函数

from torch._utils_internal import upload_graph  # 从 PyTorch 内部导入 upload_graph 函数
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq  # 从 PyTorch.fx 中导入 symbolic_shapes 模块的函数
from torch.fx.passes.graph_transform_observer import GraphTransformObserver  # 从 PyTorch.fx.passes 中导入 GraphTransformObserver 类

from .. import config, ir, pattern_matcher  # 导入上级目录的 config、ir 和 pattern_matcher 模块
from ..codegen.common import BackendFeature, has_backend_feature  # 从上级目录中导入 BackendFeature 和 has_backend_feature 函数
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage  # 从上级目录中导入 FakeTensorUpdater 等函数

from ..lowering import lowerings as L  # 导入上级目录中 lowering 模块，并重命名为 L
from ..pattern_matcher import (  # 导入上级目录中 pattern_matcher 模块的多个函数和类
    _return_true,
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    filter_nodes,
    get_arg_value,
    get_mutation_region_id,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from ..utils import decode_device, is_pointwise_use  # 从上级目录中导入 decode_device 和 is_pointwise_use 函数
from ..virtualized import V  # 从上级目录中导入 V 类
from .ddp_fusion import fuse_ddp_communication  # 导入当前目录中的 fuse_ddp_communication 函数
from .group_batch_fusion import group_batch_fusion_passes, POST_GRAD_FUSIONS  # 导入当前目录中的 group_batch_fusion_passes 和 POST_GRAD_FUSIONS 常量
from .micro_pipeline_tp import patterns as micro_pipeline_tp_patterns  # 导入当前目录中的 patterns 别名为 micro_pipeline_tp_patterns
from .pre_grad import is_same_dict, save_inductor_dict  # 导入当前目录中的 is_same_dict 和 save_inductor_dict 函数
from .reinplace import reinplace_inplaceable_ops  # 导入当前目录中的 reinplace_inplaceable_ops 函数
from .split_cat import POST_GRAD_PATTERNS  # 导入当前目录中的 POST_GRAD_PATTERNS 常量

if TYPE_CHECKING:
    from sympy import Expr  # 如果是类型检查模式，导入 sympy 模块的 Expr 类型

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器
aten = torch.ops.aten  # 设置 torch.ops.aten 别名为 aten
prims = torch.ops.prims  # 设置 torch.ops.prims 别名为 prims

# First pass_patterns[0] are applied, then [1], then [2]
# 定义一系列的模式匹配器传递对象，按顺序依次应用于图形模块
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]


def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
    if config.dce:
        # 如果配置中启用了死代码消除，则执行死代码消除操作
        gm.graph.eliminate_dead_code()

    if is_inference and config.reorder_for_locality:
        # 如果处于推断模式，并且配置中启用了重排序以提高局部性，则执行重排序操作
        reorder_for_locality(gm.graph)

    fake_tensor_updater = FakeTensorUpdater(gm.graph)
    
    if config.post_grad_custom_pre_pass is not None:
        with GraphTransformObserver(
            gm, "post_grad_custom_pre_pass", config.trace.log_url_for_graph_xform
        ):
            # 如果配置中定义了自定义的后向传递预处理函数，则应用该函数到图形模块
            config.post_grad_custom_pre_pass(gm.graph)
    # 如果配置中存在模式匹配器，则执行以下操作
    if config.pattern_matcher:
        # 执行惰性初始化
        lazy_init()
        # 上传图形的先前重新编译后梯度前的日志
        optimus_scuba_log["before_recompile_post_grad"] = upload_graph(gm.graph)
        # 执行分组批次融合传递，不涉及梯度前处理
        group_batch_fusion_passes(gm.graph, pre_grad=False)
        # 从图形中移除无操作的操作符
        remove_noop_ops(gm.graph)
        # 对每个传递模式应用模式匹配器
        for patterns in pass_patterns:
            patterns.apply(gm.graph)  # type: ignore[arg-type]
        # 遍历配置中的后梯度融合选项
        for pass_name in config.post_grad_fusion_options:
            # 如果传递名称在后梯度融合中存在，则跳过
            if pass_name in POST_GRAD_FUSIONS:
                continue
            # 获取对应于传递名称的模式匹配器传递
            pattern_matcher_pass = POST_GRAD_PATTERNS[pass_name]
            # 保存变压器字典在变化之前的状态
            inductor_before_change = save_inductor_dict(
                [pattern_matcher_pass.pass_name]
            )
            # 应用模式匹配器传递到图形
            pattern_matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
            # 如果变压器计数器与变化前的不同，则上传图形
            if not is_same_dict(counters["inductor"], inductor_before_change):
                optimus_scuba_log[
                    f"{pattern_matcher_pass.pass_name}_post_grad"
                ] = upload_graph(gm.graph)

    # 如果配置中存在微管道 TP，则应用微管道 TP 模式到图形
    if config._micro_pipeline_tp:
        micro_pipeline_tp_patterns.apply(gm)

    # 如果配置中存在融合 DDP 通信，则执行融合 DDP 通信
    if config._fuse_ddp_communication:
        fuse_ddp_communication(
            gm.graph,
            config._fuse_ddp_communication_passes,
            config._fuse_ddp_bucket_size,
        )

    # 如果存在后梯度自定义后传递，则应用它到图形
    if config.post_grad_custom_post_pass is not None:
        with GraphTransformObserver(
            gm, "post_grad_custom_post_pass", config.trace.log_url_for_graph_xform
        ):
            config.post_grad_custom_post_pass(gm.graph)

    # 对图形进行稳定的拓扑排序
    stable_topological_sort(gm.graph)

    # 将构造函数移动到 CUDA 设备上
    move_constructors_to_cuda(gm.graph)

    # 更新虚假张量增量
    fake_tensor_updater.incremental_update()

    # 将不可替代操作符进行原地替换
    reinplace_inplaceable_ops(gm.graph)
    # 对自动功能化后的函数进行分解
    decompose_auto_functionalized(gm.graph)

    # 重新编译图形
    gm.recompile()
    # 上传图形的后重新编译梯度日志
    optimus_scuba_log["after_recompile_post_grad"] = upload_graph(gm.graph)
    # 对图形进行静态分析
    gm.graph.lint()
@init_once_fakemode
def lazy_init():
    # 初始化函数装饰器，用于延迟初始化
    if torch._C._has_mkldnn:
        # 如果 PyTorch 支持 MKLDNN，则导入相关模块和函数
        from . import decompose_mem_bound_mm  # noqa: F401
        from .mkldnn_fusion import _mkldnn_fusion_init

        # 调用 MKLDNN 相关初始化函数
        _mkldnn_fusion_init()


def reorder_for_locality(graph: torch.fx.Graph):
    def visit(other_node):
        # 如果节点是函数调用，但不是 operator.getitem，并且所有其用户节点都已经遍历过，
        # 并且节点所在的变异区域 ID 与当前节点相同，则将 other_node 插入到当前节点之前
        if (
            other_node.op == "call_function"
            and other_node.target != operator.getitem
            and all((n in seen_nodes) for n in other_node.users)
            and get_mutation_region_id(graph, node)
            == get_mutation_region_id(graph, other_node)
        ):
            node.prepend(other_node)

    seen_nodes = set()

    # 仅重新排序第一个 copy_ 出现之前的节点
    # 当输入发生变异时，functionalized 图中的 copy_ 会出现在末尾，而这种重新排序对变异不起作用
    first_copy = next(
        iter(graph.find_nodes(op="call_function", target=torch.ops.aten.copy_.default)),
        None,
    )
    past_mutating_epilogue = True if first_copy is None else False

    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        if not past_mutating_epilogue:
            past_mutating_epilogue = node is first_copy
            continue

        # 对节点的参数进行映射和访问
        torch.fx.map_arg((node.args, node.kwargs), visit)


def register_lowering_pattern(pattern, extra_check=_return_true, pass_number=1):
    """
    Register an aten to inductor IR replacement pattern
    """
    # 注册一个 aten 到 inductor IR 替换模式的下降模式
    return pattern_matcher.register_lowering_pattern(
        pattern, extra_check, pass_dict=pass_patterns[pass_number]
    )


################################################################################
# Actual patterns below this point.
# Priority of patterns is:
#   - later output nodes first
#   - order patterns are defined in
################################################################################


def is_valid_mm_plus_mm(match: Match):
    # 检查矩阵乘法的形状是否有效
    *b1, m1, k1 = match.kwargs["mat1"].meta.get("tensor_meta").shape
    *b2, k2, n1 = match.kwargs["mat2"].meta.get("tensor_meta").shape
    if k1 != k2:
        return False

    *b1, m2, k3 = match.kwargs["mat3"].meta.get("tensor_meta").shape
    *b2, k4, n2 = match.kwargs["mat4"].meta.get("tensor_meta").shape
    if k3 != k4:
        return False

    if m1 != m2 or n1 != n2:
        return False

    return True


def scatter_upon_const_tensor_extra_check(m):
    # 如果不优化 scatter_upon_const_tensor，则返回 False
    if not config.optimize_scatter_upon_const_tensor:
        return False
    full_shape = m.kwargs["shape"]
    selector = m.kwargs["selector"]
    dim = m.kwargs["dim"]
    if dim < 0:
        dim += len(full_shape)

    selector_ft = selector.meta["val"]
    assert selector_ft.dim() == len(full_shape)

    for idx, select_sz, full_sz in zip(
        itertools.count(), selector_ft.shape, full_shape
    ):
        # 对于选择器和完整形状进行检查
    ):
        # 如果索引等于维度，则继续下一次循环，跳过当前维度的处理
        if idx == dim:
            continue

        # TODO: the pattern can be updated to support the case that index tensor
        # is shorter. But that will need a more complex condition expression
        # especially for multi-dimensional tensors.
        # Skip it for now.
        # 如果索引张量是 fx.Node 类型，则更新 full_sz 为其元数据值
        if isinstance(full_sz, fx.Node):
            full_sz = full_sz.meta["val"]
        
        # 如果选择的尺寸小于完整尺寸，返回 False
        if select_sz < full_sz:
            return False

    # 实际上我们可以支持小于 1 的小尺寸。这会有点繁琐。
    # 例如，我们加载所有的索引值（不多），并将它们与张量中的位置进行比较，
    # 以决定返回什么值。
    return selector_ft.size(dim) == 1
@register_lowering_pattern(
    # 注册一个降低模式，匹配函数调用模式为 aten.scatter.value
    CallFunction(
        aten.scatter.value,
        # scatter 函数的第一个参数为 aten.full 的调用结果
        CallFunction(
            aten.full,
            KeywordArg("shape"),       # full 函数的 shape 参数
            KeywordArg("background_val"),  # full 函数的 background_val 参数
            dtype=KeywordArg("dtype"),  # full 函数的 dtype 参数
        ),
        KeywordArg("dim"),              # scatter 函数的 dim 参数
        KeywordArg("selector"),         # scatter 函数的 selector 参数
        KeywordArg("val"),              # scatter 函数的 val 参数（标量值）
    ),
    extra_check=scatter_upon_const_tensor_extra_check,  # 额外的检查函数
)
def scatter_upon_const_tensor(
    match: Match, shape, background_val, dtype, dim, selector, val
):
    """
    Match the pattern of full+scatter into a pointwise.

    TODO: Right now the scatter value must be a scalar. But we could support it
    when it is a tensor as well.
    """
    from torch._inductor import metrics

    metrics.num_matches_for_scatter_upon_const_tensor += 1

    selector_loader = selector.make_loader()

    def inner_fn(idx):
        # 创建 selector_idx，将 dim 维度设为 0
        selector_idx = list(idx)
        selector_idx[dim] = 0

        # 使用 selector_loader 获取对应的 selector
        selector = selector_loader(selector_idx)
        
        # 使用 ops.where 创建条件运算，根据 selector 决定返回 val 或 background_val
        return ops.where(
            selector == ops.index_expr(idx[dim], torch.int64),
            ops.constant(val, dtype),
            ops.constant(background_val, dtype),
        )

    # 创建并返回一个 Pointwise 对象，设备为 selector 的设备，数据类型为 dtype，
    # 内部函数为 inner_fn，范围为 shape
    return ir.Pointwise.create(
        device=selector.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=shape,
    )


@register_lowering_pattern(
    # 注册一个降低模式，匹配函数调用模式为 aten.add(aten.mm(...), aten.mm(...))
    CallFunction(
        aten.add,
        CallFunction(aten.mm, KeywordArg("mat1"), KeywordArg("mat2")),  # 第一个 mm 函数调用
        CallFunction(aten.mm, KeywordArg("mat3"), KeywordArg("mat4")),  # 第二个 mm 函数调用
    ),
    extra_check=is_valid_mm_plus_mm,  # 额外的检查函数
)
def mm_plus_mm(match: Match, mat1, mat2, mat3, mat4):
    # 调用 inductor.kernel.mm_plus_mm.tuned_mm_plus_mm 处理 mm+mm 的降低
    return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)


def cuda_and_enabled_mixed_mm(match):
    # 检查是否启用混合精度 mm 运算，并且输入张量是 CUDA 张量，且满足特定条件
    return (
        (config.use_mixed_mm or config.mixed_mm_choice != "default")
        and getattr(match.kwargs["mat1"].meta.get("val"), "is_cuda", False)
        and (
            match.kwargs["mat2_dtype"].itemsize
            > match.kwargs["mat2"].meta.get("val").dtype.itemsize
        )
        and has_backend_feature("cuda", BackendFeature.TRITON_TEMPLATES)
    )


def cuda_and_enabled_mixed_mm_and_not_int8(match):
    # 检查是否启用混合精度 mm 运算，并且输入张量是 CUDA 张量，且满足特定条件，但不是 torch.int8 类型
    return (
        cuda_and_enabled_mixed_mm(match)
        and getattr(match.kwargs["mat1"].meta.get("val"), "is_cuda", False)
        and getattr(match.kwargs["mat2"].meta.get("val"), "dtype", torch.int8)
        != torch.int8
    )  # bitshift numerics in triton and pytorch don't match for torch.int8


"""
    this is intended to be used to unpack a [K,N] int4 tensor from a [K/2, N] uint4x2 tensor
    (where the int4 and uint4x2 are represented with int8 and uint8 respectively)
    where every other row of the int4 is packed with the row above it as:
    uint4x2[k,n] = (8+int4[2*k,n])+(8+int4[2*k+1,n])<<4

    unpack formulas:
    int4[2*k,n]=(uint4x2[k,n] & 0xF) - 8
    int4[2*k+1,n]=(uint4x2[k,n] >> 4) - 8

    thus matching on unpack formula:
"""
    # 使用 torch 库进行矩阵乘法运算
    torch.mm(
        # 参数1: 第一个矩阵 mat1
        mat1,
        # 参数2: 将 mat2 进行按位操作后拼接的结果，再进行形状重塑和类型转换，并最终减去 8
        torch.cat(
            (
                # 将 mat2 的低4位与0xF进行按位与操作
                mat2 & 0xF,
                # 将 mat2 右移4位作为另一个拼接部分
                mat2 >> 4
            ),
            # 将拼接后的张量重塑为 mat2_mm_shape 形状
            1
        ).reshape(mat2_mm_shape).to(mat2_dtype).sub(8)
    )
    
    # 注意：虽然在 PyTorch 和 Triton 内核中解包公式是为了 uint8 的 mat2 设计的，但行为上
    # Triton 内核与 PyTorch 的公式在除了 torch.int8 类型外的所有数据类型上是匹配的，
    # 在 torch.int8 类型中，Triton 中的位运算数值与 PyTorch 中的不匹配。
"""
@register_lowering_pattern(
    CallFunction(
        aten.mm.default,  # 使用 torch.mm 的默认实现
        KeywordArg("mat1"),  # 第一个矩阵参数
        CallFunction(
            aten.sub.Tensor,  # 对第二个矩阵进行张量减法
            CallFunction(
                prims.convert_element_type.default,  # 转换第二个矩阵的元素类型
                CallFunction(
                    aten.reshape.default,  # 对张量进行重塑操作
                    CallFunction(
                        aten.cat.default,  # 沿指定维度拼接张量列表
                        ListOf(
                            CallFunction(
                                aten.bitwise_and.Scalar,  # 对第二个矩阵进行按位与运算
                                KeywordArg("mat2"),  # 第二个矩阵参数
                                0xF,  # 使用掩码 0xF 进行按位与
                            ),
                            True,  # 真值 (占位用)
                        ),
                        1,  # 沿着第一维度进行拼接
                    ),
                    KeywordArg("mat2_mm_shape"),  # 重塑后的张量形状
                ),
                KeywordArg("mat2_dtype"),  # 第二个矩阵的数据类型
            ),
            8,  # 使用常数 8 进行张量减法
        ),
    ),
    extra_check=cuda_and_enabled_mixed_mm_and_not_int8,  # 额外检查条件
)
def uint4x2_mixed_mm(match: Match, mat1, mat2, mat2_mm_shape, mat2_dtype):
    return inductor.kernel.unpack_mixed_mm.tuned_uint4x2_mixed_mm(
        mat1, mat2, mat2_mm_shape, mat2_dtype  # 调用特定函数处理混合乘法操作
    )


"""
torch.mm(mat1, mat2.to(mat2_dtype))  # 使用 torch.mm 函数进行矩阵乘法操作
"""


@register_lowering_pattern(
    CallFunction(
        aten.mm,  # 使用 torch.mm 函数
        KeywordArg("mat1"),  # 第一个矩阵参数
        CallFunction(
            prims.convert_element_type.default,  # 转换第二个矩阵的元素类型
            KeywordArg("mat2"),  # 第二个矩阵参数
            KeywordArg("mat2_dtype"),  # 第二个矩阵的数据类型
        ),
    ),
    extra_check=cuda_and_enabled_mixed_mm,  # 额外检查条件
)
def mixed_mm(match: Match, mat1, mat2, mat2_dtype):
    return inductor.kernel.mm.tuned_mixed_mm(mat1, mat2, mat2_dtype)  # 调用特定函数处理混合乘法操作


@register_graph_pattern(
    CallFunction(
        aten.cumsum.default,  # 使用 torch.cumsum 函数的默认实现
        CallFunction(
            torch.ops.aten.full.default,  # 使用 torch.full 函数的默认实现
            KeywordArg("shape"),  # 张量的形状参数
            KeywordArg("fill_value"),  # 填充值参数
            dtype=KeywordArg("dtype"),  # 数据类型参数
            layout=Ignored(),  # 忽略布局参数
            device=KeywordArg("device"),  # 设备参数
            pin_memory=False,  # 不启用 pin_memory
            _users=MULTIPLE,  # 多个用户
        ),
        KeywordArg("dim"),  # 累积求和的维度参数
        _users=MULTIPLE,  # 多个用户
    ),
    pass_dict=pass_patterns[1],  # 传递字典的特定模式
)
def pointless_cumsum_replacement(match: Match, shape, fill_value, device, dtype, dim):
    """基于 OPTForCausalLM 中的模式"""

    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        dtype = torch.int64  # 如果是整数或布尔型，将数据类型设置为 int64

    def repl(*shape):
        dim_size = shape[dim]  # 获取指定维度的尺寸
        idx = torch.arange(1, dim_size + 1, device=device, dtype=dtype)  # 创建索引张量

        inter_shape = [1] * len(shape)  # 初始化一个与输入张量相同长度的形状列表
        inter_shape[dim] = dim_size  # 设置指定维度的形状大小
        return (idx * fill_value).view(inter_shape).expand(shape)  # 返回填充后的张量

    match.nodes = [match.output_node()]  # 替换输出节点，而不是所有节点
"""
    # 使用 match 对象的 replace_by_example 方法，根据 repl 和 shape 参数进行替换操作
    match.replace_by_example(repl, list(shape))
def shape_of_mm(a, b):
    # 获取矩阵 a 的尺寸并返回其行数 m
    m, _ = a.get_size()
    # 获取矩阵 b 的尺寸并返回其列数 n
    _, n = b.get_size()
    return [m, n]


@register_lowering_pattern(
    CallFunction(aten.cat, ListOf(CallFunction(aten.mm, Arg(), Arg())), Arg()),
)
def cat_mm(match, inputs, dim):
    # 调用 cat_tuned_op 函数，处理 aten.mm 操作
    return cat_tuned_op(match, inputs, dim, op=L[aten.mm], shape_of=shape_of_mm)


@register_lowering_pattern(
    CallFunction(
        aten.cat, ListOf(CallFunction(aten.addmm, Arg(), Arg(), Arg())), Arg()
    ),
)
def cat_addmm(match, inputs, dim):
    def shape_of(bias, a, b):
        # 获取矩阵 a 的尺寸并返回其行数 m
        m, _ = a.get_size()
        # 获取矩阵 b 的尺寸并返回其列数 n
        _, n = b.get_size()
        return [m, n]

    # 调用 cat_tuned_op 函数，处理 aten.addmm 操作
    return cat_tuned_op(match, inputs, dim, op=L[aten.addmm], shape_of=shape_of)


def cat_tuned_op(match, inputs, dim, *, op, shape_of):
    """
    Memory planning to remove cat. We can't use the stock memory
    planner since autotuning matmuls needs to know the output layout.
    """
    if len(inputs) == 1:
        # 如果 inputs 中只有一个元素，则直接调用 op 函数
        return op(*inputs[0])

    # TODO(jansel): rewrite this as a bmm?
    if dim < 0:
        # 处理负数维度索引，将其转换为正数索引
        dim += len(shape_of(*inputs[0]))
    assert dim in (0, 1)
    notdim = 1 - dim

    new_size: Optional[Union[List[Expr], List[int]]] = None
    offsets_start = []
    offsets_end = []

    # compute output sizes
    for i in range(len(inputs)):
        # 获取每个输入的形状信息
        shape = shape_of(*inputs[i])
        if new_size is None:
            new_size = shape
        else:
            # 更新输出尺寸
            new_size[notdim] = V.graph.sizevars.guard_equals(
                shape[notdim], new_size[notdim]
            )
            new_size[dim] += shape[dim]
        offsets_start.append(new_size[dim] - shape[dim])
        offsets_end.append(new_size[dim])

    assert new_size is not None
    # 获取数据类型，使用 functools.reduce 和 itertools.chain.from_iterable 来推断
    dtype = functools.reduce(
        torch.promote_types,
        [x.get_dtype() for x in itertools.chain.from_iterable(inputs)],
    )
    device = inputs[0][0].get_device()
    # 创建 ConcatKernel 对象
    kernel = ir.ConcatKernel(
        name=None,
        layout=ir.FixedLayout(device, dtype, new_size),
        inputs=[],
    )
    kernel_tensor = ir.TensorBox.create(kernel)

    for i in range(len(inputs)):
        # 为每个输入创建切片视图，并执行 op 操作
        dst = ir.SliceView.create(kernel_tensor, dim, offsets_start[i], offsets_end[i])
        src = op(*inputs[i], layout=dst.get_layout()).data.data
        assert isinstance(src, (ir.ExternKernelOut, ir.TemplateBuffer))
        src.layout = ir.NonOwningLayout(dst)
        kernel.inputs.append(src)

    # 注册 kernel 并返回 kernel_tensor
    kernel.name = V.graph.register_buffer(kernel)
    kernel.inputs = ir.ConcatKernel.unwrap_storage(kernel.inputs)
    return kernel_tensor


_cat_1 = CallFunction(aten.cat, Arg(), 1, _users=2)


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        [
            _cat_1,
            CallFunction(
                aten.slice,
                _cat_1,
                1,
                0,
                KeywordArg("size"),
            ),
        ],
        1,
    )
)
def cat_slice_cat(match, cat_input, size, dim=1):
    """
    This is an example of a more complex pattern where cat_1 is used
    """
    # 这是一个复杂模式的示例，其中使用了 cat_1
    """
    # 如果尺寸（size）为正并且已知在first张量的指定维度（dim）内，则进行优化
    # 优化意味着将两次cat操作合并为一次
    if size >= 0 and V.graph.sizevars.statically_known_leq(size, first.get_size()[dim]):
        # 合并两次cat操作为一次cat操作
        return L[aten.cat](
            [
                first,                          # 第一个张量
                *rest,                          # 其余所有张量
                L[aten.slice](first, dim, 0, size),  # 对第一个张量进行切片操作
            ],
            dim,                              # 在指定维度dim上进行cat操作
        )
    else:
        # 不希望进入此情况，仅作为备用方案
        tmp = L[aten.cat](cat_input, dim)    # 在指定维度dim上进行cat操作
        return L[aten.cat](
            [
                tmp,                           # 第一个cat操作的结果
                L[aten.slice](tmp, dim, 0, size),  # 对第一个cat操作的结果进行切片操作
            ],
            dim,                              # 在指定维度dim上进行cat操作
        )
    """
# 判断是否是有效的 split_with_sizes 和 cat 操作的组合
def is_valid_splitwithsizes_cat(match):
    # 从匹配中筛选出所有 split_with_sizes 节点
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    # 从匹配中筛选出所有 cat 节点
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    # 从匹配中筛选出所有 getitem 节点
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    
    # 如果 split 或 cat 节点数量不等于 1，则返回 False
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    
    # 获取第一个 split 和 cat 节点
    split_node, cat_node = split_nodes[0], cat_nodes[0]
    
    # 检查 split 和 cat 的维度参数是否匹配
    if get_arg_value(split_node, 2, "dim") != get_arg_value(cat_node, 1, "dim"):
        return False
    
    # 获取所有 getitem 节点的第二个参数集合
    get_item_args = {
        get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes
    }
    # 断言 getitem 参数集合中不包含 None
    assert None not in get_item_args
    
    # 获取 split 节点的 split_sizes 参数
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    
    # 检查所有 split 的部分是否都包含在 cat 中
    if get_item_args != set(range(len(split_sizes))):
        return False
    
    # 检查 getitem 参数顺序是否与 cat 节点中使用的顺序相同
    cat_items_args_order = [
        get_arg_value(item_node, 1) for item_node in get_arg_value(cat_node, 0)
    ]
    if cat_items_args_order != list(range(len(split_sizes))):
        return False
    
    return True


# 检查两个节点是否具有相同的元数据
def same_meta(node1: torch.fx.Node, node2: torch.fx.Node):
    """True if two nodes have the same metadata"""
    val1 = node1.meta.get("val")
    val2 = node2.meta.get("val")
    return (
        val1 is not None
        and val2 is not None
        and statically_known_true(sym_eq(val1.size(), val2.size()))
        and val1.layout == val2.layout
        and val1.dtype == val2.dtype
        and val1.device == val2.device
        and (
            val1.layout != torch.strided
            or statically_known_true(sym_eq(val1.stride(), val2.stride()))
        )
    )


# 一个空的注册表
noop_registry: Dict[Any, Any] = {}


# 注册一个空操作的分解函数
def register_noop_decomp(targets, nop_arg=0):
    def register_fun(cond):
        # 在指定的注册表中注册分解函数
        register_decomposition(targets, registry=noop_registry, unsafe=True)(
            (cond, nop_arg)
        )
        return cond

    return register_fun


# 注册一个空的 slice 操作的分解函数
@register_noop_decomp(aten.slice)
def slice_noop(self, dim=0, start=None, end=None, step=1):
    if start is None or end is None:
        return False
    # 如果 slice 的参数符合特定条件，则返回 True
    if start == 0 and end >= 2**63 - 1 and step == 1:
        return True
    return False


# 注册一个空的 slice_scatter 操作的分解函数
@register_noop_decomp(aten.slice_scatter, 1)
def slice_scatter_noop(self, src, dim=0, start=None, end=None, step=1):
    if start is None:
        start = 0
    if end is None:
        end = 2**63 - 1
    # 如果 slice_scatter 的参数符合特定条件，则返回 True
    if start == 0 and end >= 2**63 - 1 and step == 1:
        return True
    return False


# 注册一个空的 repeat 操作的分解函数
@register_noop_decomp(aten.repeat)
def repeat_noop(self, repeats):
    # 如果所有的 repeats 都等于 1，则返回 True
    return all(r == 1 for r in repeats)


# 注册一个空的 constant_pad_nd 操作的分解函数
@register_noop_decomp(aten.constant_pad_nd)
def constant_pad_nd(x, padding, fill_value=0):
    # 如果所有的 padding 都等于 0，则返回 True
    return all(p == 0 for p in padding)


# 注册一个空的 convert_element_type 操作的分解函数
@register_noop_decomp(torch.ops.prims.convert_element_type)
def convert_element_type_noop(x, dtype: torch.dtype):
    # 检查张量 x 的数据类型是否与指定的 dtype 相同
    return x.dtype == dtype


@register_noop_decomp(torch.ops.prims.device_put)
def device_put_noop(x, device):
    # 检查张量 x 的设备是否与解码后的 device 相同
    return x.device == decode_device(device)


@register_noop_decomp([aten.ceil, aten.floor, aten.round, aten.trunc])
def int_noop(x):
    # 检查张量 x 的数据类型是否为整数类型
    return is_integer_dtype(x.dtype)


@register_noop_decomp([aten.pow])
def pow_noop(a, b):
    # 检查参数 b 是否为整数且等于 1
    return isinstance(b, int) and b == 1


@register_noop_decomp([aten.cat], lambda args: args[0][0])
def cat_noop(inputs, dim=0):
    # 检查输入的张量列表长度是否为 1
    return len(inputs) == 1


@register_noop_decomp(aten.view)
def view_noop(arg, size):
    # 检查张量 arg 的形状是否与指定的 size 相同
    return arg.shape == size


# 注意，我们总是检查相同的元数据，因此这些操作是安全的
@register_noop_decomp([aten.copy], nop_arg=1)
@register_noop_decomp([aten.alias, aten.clone])
def true_noop(*args, **kwargs):
    # 无条件返回 True，表示这些操作不会改变张量的内容
    return True


def remove_noop_ops(graph: torch.fx.Graph):
    """
    Removes both operations that are essentially aten.clone and operations that are essentially aten.alias from the graph.
    移除图中本质上是 aten.clone 和 aten.alias 操作的节点。
    """
    inputs = set()
    input_storages = set()
    output_storages = set()

    # 遍历图中的占位符节点，收集输入节点和输入存储
    for node in graph.find_nodes(op="placeholder"):
        inputs.add(node)
        input_storages.add(get_node_storage(node))

    # 获取输出节点（通常是最后一个节点）并收集输出存储
    output_node = next(iter(reversed(graph.nodes)))
    assert output_node.op == "output"
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        # 嵌套子图可能会有单一输出
        outputs = (outputs,)
    for out in outputs:
        if isinstance(out, torch.fx.Node):
            output_storages.add(get_node_storage(out))
    # 遍历图中的每个节点
    for node in graph.nodes:
        # 如果节点的目标在 noop_registry 中
        if node.target in noop_registry:
            # 从 noop_registry 中获取条件和源索引
            cond, src_index = noop_registry[node.target]
            # 如果源索引是整数，则从节点的参数中获取源
            if isinstance(src_index, int):
                src = node.args[src_index]
            else:
                # 否则，调用 src_index 函数获取源
                src = src_index(node.args)
            # 如果源不是 torch.fx.Node 类型，则继续下一个节点
            if not isinstance(src, torch.fx.Node):
                continue
            
            # 避免在输入和输出之间引入新的别名关系
            # 参见 fx_passes/README.md 了解为何这一步骤是必要的
            node_storage = get_node_storage(node)
            src_storage = get_node_storage(src)
            node_is_view = node_storage == src_storage
            
            # 如果节点和源的存储不同，并且节点存储在输出存储中，
            # 而源存储在输入存储或输出存储中，则继续下一个节点
            if (
                not node_is_view
                and node_storage in output_storages
                and (src_storage in input_storages or src_storage in output_storages)
            ):
                continue
            
            # 即使输入和输出预计会有别名关系，也不要使 "node is src" 为真
            if (
                node_is_view
                and node in output_node.args
                and (src in inputs or src in output_node.args)
            ):
                continue
            
            # 获取节点的虚拟参数和关键字参数
            is_valid, args, kwargs = get_fake_args_kwargs(node)
            # 如果不是有效的参数组合，则继续下一个节点
            if not is_valid:
                continue
            
            # 如果节点与源具有相同的元信息，并且满足条件函数 cond，则用源替换节点的所有使用
            if same_meta(node, src) and cond(*args, **kwargs):
                node.replace_all_uses_with(src)
                # 从图中删除节点
                graph.erase_node(node)
# 定义一个函数，用于分解自动功能化的图形
def decompose_auto_functionalized(graph):
    # 创建一个图案匹配器传递对象
    graph_pass = PatternMatcherPass()

    # 注册一个图形模式匹配器，匹配调用高阶自动功能化操作的函数
    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.auto_functionalized),
        pass_dict=graph_pass,
    )
    # 定义一个替换函数，用于替换匹配到的图形模式
    def replacement(match: Match, *args, **kwargs):
        # 导入自动功能化模块中的特定函数
        from torch._higher_order_ops.auto_functionalize import auto_functionalized_dense

        # 获取要克隆的张量列表
        only_clone_these_tensors = tuple(
            match.nodes[0].meta.get("only_clone_these_tensors", [])
        )

        # 将参数（args, kwargs）扁平化成一维数组
        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # 注意：我们将（args, kwargs）组合成扁平化参数以进行替换。
        # 这是因为 replace_by_example 使用 make_fx，它不支持跟踪带有 kwargs 的函数。
        def decomp(*flat_args):
            # 将扁平化参数重新构建成原始参数形式
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            # 调用自动功能化密集操作函数
            return auto_functionalized_dense(*args, only_clone_these_tensors, **kwargs)

        # 使用自定义的 decomp 函数进行替换
        match.replace_by_example(decomp, flat_args, run_dce=False)

    # 应用图案匹配器传递对象到图形
    graph_pass.apply(graph)

    # 查找所有调用 torch.ops.higher_order.auto_functionalized 的节点
    for node in graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized
    ):
        # 如果找到任何调用，抛出断言错误
        raise AssertionError("auto_functionalized was not removed")


# 注册一个降低模式匹配器，匹配调用 torch.aten.cat 的函数
@register_lowering_pattern(
    CallFunction(
        aten.cat,
        ListOf(
            CallFunction(
                operator.getitem,
                CallFunction(
                    aten.split_with_sizes,
                    KeywordArg("input_"),
                    Ignored(),
                    Ignored(),
                    _users=MULTIPLE,
                ),
                Ignored(),
            ),
        ),
        Ignored(),
    ),
    pass_number=2,
    extra_check=is_valid_splitwithsizes_cat,
)
# 定义一个替换函数，用于返回输入参数
def splitwithsizes_cat_replace(match, input_):
    return input_


# 定义一个函数，检查是否有效的 cat-split_with_sizes
def is_valid_cat_splitwithsizes(match):
    # 过滤所有 aten.cat 节点
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    # 过滤所有 aten.split_with_sizes 节点
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    
    # 如果找到的 split 节点数不是1或者 cat 节点数不是1，则返回 False
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    
    # 获取第一个 split 和 cat 节点
    split_node, cat_node = split_nodes[0], cat_nodes[0]

    # 如果 cat 节点有多于一个用户，则返回 False
    if len(cat_node.users) > 1:
        return False

    # 获取 split 和 cat 节点的维度参数，并比较它们是否相等
    dim = get_arg_value(split_node, 2, "dim")
    if dim != get_arg_value(cat_node, 1, "dim"):
        return False

    # 获取 cat 节点的输入张量列表和 split 节点的分割尺寸列表
    cat_inputs = list(get_arg_value(cat_node, 0))
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    
    # 如果 cat 输入张量数与 split 尺寸数不相等，则返回 False
    if len(cat_inputs) != len(split_sizes):
        return False

    # 逐一检查每个 cat 输入张量的尺寸是否与对应的 split 尺寸匹配
    for cat_input, split_size in zip(cat_inputs, split_sizes):
        # 如果 cat 输入张量的尺寸信息中没有 "val"，则返回 False
        if "val" not in cat_input.meta:
            return False
        # 获取 cat 输入张量在指定维度上的尺寸，并与对应的 split 尺寸比较
        cat_input_size = cat_input.meta["val"].size(dim)
        if cat_input_size != split_size:
            return False

    # 如果所有检查通过，则返回 True
    return True
# 注册一个降低模式，用于匹配对应的函数调用图模式
@register_lowering_pattern(
    # 匹配 aten.split_with_sizes 的函数调用
    CallFunction(
        aten.split_with_sizes,
        # 匹配 aten.cat 的函数调用，其中 "input_" 参数被关键字参数替代，其他参数被忽略
        CallFunction(
            aten.cat,
            KeywordArg("input_"),  # 关键字参数 "input_"
            Ignored(),             # 忽略的参数
            _users=MULTIPLE,       # 匹配多个用户
        ),
        Ignored(),  # 忽略的参数
        Ignored(),  # 忽略的参数
    ),
    pass_number=2,  # 传递次数为 2
    extra_check=is_valid_cat_splitwithsizes,  # 额外的检查函数
)

def cat_splitwithsizes_replace(match, input_):
    return input_


def view_to_reshape(gm):
    """
    将 GraphModule 中的 view 操作替换为 reshape 操作。
    """
    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.view.default
    ):
        nd.target = torch.ops.aten.reshape.default


def should_prefer_unfused_addmm(match):
    inp = match.kwargs["inp"]
    if not inp.meta["val"].is_cuda:
        return False

    output = match.output_node()
    return all(is_pointwise_use(use) for use in output.users)


@register_graph_pattern(
    # 匹配 aten.addmm 的函数调用，使用关键字参数 "inp"、位置参数
    CallFunction(aten.addmm, KeywordArg("inp"), Arg(), Arg()),
    pass_dict=pass_patterns[2],  # 使用 pass_patterns 中索引为 2 的字典
    extra_check=should_prefer_unfused_addmm,  # 额外的检查函数
)

def unfuse_bias_add_to_pointwise(match: Match, mat1, mat2, *, inp):
    def repl(inp, x1, x2):
        return x1 @ x2 + inp  # 替换函数的实现

    match.replace_by_example(repl, [inp, mat1, mat2])


def is_valid_addmm_fusion(match):
    mat1, mat2 = match.args
    inp = match.kwargs["inp"]

    if not (
        isinstance(inp, torch.fx.Node) and isinstance(inp.meta["val"], torch.Tensor)
    ):
        return False  # 输入不是张量

    in_shape = inp.meta["val"].shape
    mm_shape = mat1.meta["val"].shape[0], mat2.meta["val"].shape[1]
    matched = is_expandable_to(in_shape, mm_shape)
    if not matched:
        return False  # 形状不匹配

    return not should_prefer_unfused_addmm(match)


@register_graph_pattern(
    # 匹配 aten.add 函数调用，其中包含 aten.mm 的调用作为参数
    CallFunction(
        aten.add,
        CallFunction(aten.mm, Arg(), Arg()),  # 第一个参数为 aten.mm 的函数调用
        KeywordArg("inp"),  # 第二个参数为关键字参数 "inp"
    ),
    pass_dict=pass_patterns[2],  # 使用 pass_patterns 中索引为 2 的字典
    extra_check=is_valid_addmm_fusion,  # 额外的检查函数
)

@register_graph_pattern(
    # 匹配 aten.add 函数调用，其中包含 aten.mm 的调用作为参数
    CallFunction(
        aten.add,
        KeywordArg("inp"),  # 第一个参数为关键字参数 "inp"
        CallFunction(aten.mm, Arg(), Arg()),  # 第二个参数为 aten.mm 的函数调用
    ),
    pass_dict=pass_patterns[2],  # 使用 pass_patterns 中索引为 2 的字典
    extra_check=is_valid_addmm_fusion,  # 额外的检查函数
)

def addmm(match, mat1, mat2, *, inp):
    def repl(inp, mat1, mat2):
        return aten.addmm(inp, mat1, mat2)  # 替换函数的实现

    match.replace_by_example(repl, [inp, mat1, mat2])


def check_shape_cuda_and_fused_int_mm_mul_enabled(match):
    return (
        config.force_fuse_int_mm_with_mul
        and len(getattr(match.args[2].meta.get("val"), "shape", [])) == 2
        and getattr(match.args[2].meta.get("val"), "is_cuda", False)
    )


@register_lowering_pattern(
    # 注册一个降低模式，用于匹配 prims.convert_element_type.default 的函数调用
    CallFunction(
        prims.convert_element_type.default,
        CallFunction(
            aten.mul,
            CallFunction(
                aten._int_mm,
                Arg(),
                Arg(),
            ),
            Arg(),
        ),
        Arg(),
    ),
    check_shape_cuda_and_fused_int_mm_mul_enabled,  # 使用检查函数检查条件
)
@register_lowering_pattern(
    # 还有一些代码未提供，无法为其添加注释
    # 调用函数 CallFunction，并传入 aten.mul 作为第一个参数，表示执行乘法操作
    CallFunction(
        aten.mul,
        # 调用函数 CallFunction，并传入 aten._int_mm 作为第一个参数，执行整数矩阵乘法操作
        CallFunction(
            aten._int_mm,
            # Arg() 表示参数，这里作为 aten._int_mm 的第一个参数
            Arg(),
            # Arg() 表示参数，这里作为 aten._int_mm 的第二个参数
            Arg(),
        ),
        # Arg() 表示参数，作为 aten.mul 的第二个参数
        Arg(),
    ),
    # 调用函数 check_shape_cuda_and_fused_int_mm_mul_enabled，用于检查是否启用了 CUDA 和融合整数矩阵乘法操作
    check_shape_cuda_and_fused_int_mm_mul_enabled,
)
def fused_int_mm_mul(match: Match, mat1, mat2, mat3, out_dtype=None):
    # 调用外部库函数，执行整数矩阵乘法，并返回结果
    return inductor.kernel.mm.tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype)


class ConstructorMoverPass:
    def __init__(self, target: str, allow_outputs: bool = False) -> None:
        """
        Move constructors from cpu to the target_device.

        Sweeps through the module, looking for constructor nodes that can be moved
        to the target_device.

        A constructor node can be moved to the target_device iff all of its users
        can also be moved (tested by cannot_be_moved). Otherwise, all dependent
        constructor nodes won't be moved.

        - target: target device type
        - allow_outputs: allow outputs to be moved
        """
        # 初始化构造函数，设定目标设备类型和是否允许输出移动的标志
        self.target = target
        self.allow_outputs = allow_outputs

        # 断言目标类型是字符串，用于表示设备类型
        assert isinstance(target, str), (
            "target should be a string representing the device type. "
            f"Got: {type(target).__name__}"
        )

    def allow_cpu_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node that returns a tensor on the target device may have
        cpu tensors as input.
        """
        # 判断节点是否允许在目标设备上使用 CPU 张量作为输入
        return node.target in (
            torch.ops.aten.index.Tensor,
            torch.ops.aten.index_put.default,
            torch.ops.aten.index_put_.default,
            torch.ops.aten.copy.default,
            torch.ops.aten.copy_.default,
            torch.ops.aten.slice_scatter.default,
        )

    def cannot_be_moved(self, node: fx.Node) -> bool:
        """
        Returns whether a node can be moved to the target device.

        If this function returns False, it means that this node and all of its users
        won't be moved into the target device.
        """
        # 判断节点是否可以移动到目标设备
        if node.target == "output":
            return not self.allow_outputs

        if not (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace in ("prims", "aten")
        ):
            return True

        return False

    def get_node_device(self, node: fx.Node) -> Optional[torch.device]:
        """
        Get the device of a node.
        """
        # 获取节点所在的设备
        ten = node.meta.get("val")
        return None if not isinstance(ten, torch.Tensor) else ten.device

    def get_cpu_indeg_count(self, graph: fx.Graph) -> Dict[fx.Node, int]:
        """
        Get the number of cpu inputs to a node
        """
        # 统计每个节点的 CPU 输入数量
        cpu_indeg: Dict[fx.Node, int] = Counter()

        for node in graph.nodes:
            cpu_count = 0

            def add_cpu_inp(node):
                nonlocal cpu_count
                device = self.get_node_device(node)
                cpu_count += device is not None and device.type == "cpu"

            pytree.tree_map_only(fx.Node, add_cpu_inp, (node.args, node.kwargs))

            if cpu_count:
                cpu_indeg[node] = cpu_count

        return cpu_indeg
    # 定义一个方法 __call__，用于对图中的节点进行处理
    def __call__(self, graph: fx.Graph) -> None:
        # 初始化目标设备集合和构造器列表
        target_devices = set()
        constructors = []

        # 遍历图中的每个节点
        for node in graph.nodes:
            # 获取节点的设备信息
            device = self.get_node_device(node)
            # 如果设备存在且类型与目标设备相同，则加入目标设备集合
            if device and device.type == self.target:
                target_devices.add(device)

            # 如果节点的目标不是 torch._ops.OpOverload 类型或者不在命名空间 ("prims", "aten") 中，则继续下一个节点
            if not (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace in ("prims", "aten")
            ):
                continue

            # 如果节点的目标不是 fake_tensor._is_tensor_constructor，则继续下一个节点
            if not torch._subclasses.fake_tensor._is_tensor_constructor(node.target):
                continue

            # 如果节点的关键字参数中的设备不是 torch.device("cpu")，则继续下一个节点
            if not node.kwargs.get("device") == torch.device("cpu"):
                continue

            # 将符合条件的节点加入构造器列表
            constructors.append(node)

        # 如果没有符合条件的构造器或者目标设备数量不为 1，则直接返回
        if not constructors or len(target_devices) != 1:
            return

        # 找到可移动的构造器节点
        movable_constructors = self.find_movable_constructors(graph, constructors)

        # 遍历可移动的构造器节点
        for node in movable_constructors:
            # 复制节点的关键字参数
            kwargs = node.kwargs.copy()
            # 将设备参数设置为目标设备集合中的第一个设备
            kwargs["device"] = next(iter(target_devices))
            # 更新节点的关键字参数
            node.kwargs = kwargs

    # 定义一个方法 find_movable_constructors，用于找到可移动的构造器节点
    def find_movable_constructors(
        self, graph: fx.Graph, constructors: List[fx.Node]
    # 返回一个节点集合，表示从 CPU 构造函数开始，遍历图并测试所有下游使用是否可以安全地移动到 CPU。
    def find_safe_to_move_to_cpu(
        self,
        constructors: Set[fx.Node],
        graph: Graph,
    ) -> Set[fx.Node]:
        """
        Starting from the cpu constructors, iterate through the graph and test that all of their
        downstream uses can safely be moved to cpu.
        """
        
        # 获取每个节点的入度计数，用于 CPU 节点
        cpu_indeg: Dict[fx.Node, int] = self.get_cpu_indeg_count(graph)
    
        # 无法移动到 CUDA 的构造函数集合
        cannot_move_to_cuda: Set[fx.Node] = set()
    
        # 记录每个节点依赖的构造函数集合
        constructor_dependencies: Dict[fx.Node, Set[fx.Node]] = defaultdict(set)
    
        # 如果一个 CPU 节点依赖于两个不同的 CPU 构造函数，则任何一个无法移动到 CUDA，另一个也无法移动。
        # 在这种情况下，依赖于一个构造函数的节点也会依赖于另一个构造函数。
        equal_constructor_sets: Dict[fx.Node, Set[fx.Node]] = {
            c: {c} for c in constructors
        }
    
        # 函数：使两个依赖集合等价
        def make_dependencies_equivalent(
            set1: Set[fx.Node], set2: Set[fx.Node]
        ) -> Set[fx.Node]:
            # 可以使用并查集，但这里不值得增加复杂性
            set1.update(set2)
            for obj in set1:
                equal_constructor_sets[obj] = set1
            return set1
    
        # 初始将所有构造函数添加到队列中
        queue: List[fx.Node] = list(constructors)
    
        for c in queue:
            constructor_dependencies[c].add(c)
    
        # 开始遍历队列
        while queue:
            node = queue.pop()
            dependencies = constructor_dependencies[node]
    
            # 检查当前节点的每个用户节点
            for user in node.users:
                if self.cannot_be_moved(user):
                    # 如果用户节点无法移动到 CUDA，则当前节点的所有依赖节点都无法移动到 CUDA
                    cannot_move_to_cuda.update(dependencies)
                    break
    
                # 如果节点被用于一个接受多个设备并输出 CUDA 张量的操作，
                # 我们可以将其 CPU 输入转换为 CUDA 而不进行进一步的更改
                node_device = self.get_node_device(user)
                if (
                    self.allow_cpu_device(user)
                    and node_device
                    and node_device.type == self.target
                ):
                    del cpu_indeg[user]
                else:
                    # 否则，继续查看其下游使用
                    cpu_indeg[user] -= 1
                    if cpu_indeg[user] == 0:
                        del cpu_indeg[user]
                        queue.append(user)
    
                # 合并当前节点的依赖集合和用户节点的依赖集合
                unioned_set = make_dependencies_equivalent(
                    dependencies, constructor_dependencies[user]
                )
                constructor_dependencies[user] = unioned_set
    
        # 更新无法移动到 CUDA 的节点集合
        for node in cpu_indeg:
            if constructor_dependencies[node]:
                cannot_move_to_cuda.update(constructor_dependencies[node])
    
        # 更新所有无法移动到 CUDA 的构造函数集合
        all_cannot_move_to_cuda = cannot_move_to_cuda.copy()
        for constructor in cannot_move_to_cuda:
            all_cannot_move_to_cuda.update(equal_constructor_sets[constructor])
    
        # 返回所有可以安全移动到 CPU 的构造函数集合
        return set(constructors) - all_cannot_move_to_cuda
# 定义一个函数，将在图形中构造在 CPU 上的中间张量安全地移动到 CUDA 上
def move_constructors_to_cuda(graph: fx.Graph) -> None:
    """
    Moves intermediary tensors which are constructed on the cpu to cuda when safe
    """
    # 创建一个 ConstructorMoverPass 对象，使用 "cuda" 参数，将此对象应用到给定的图形上
    ConstructorMoverPass("cuda")(graph)
```