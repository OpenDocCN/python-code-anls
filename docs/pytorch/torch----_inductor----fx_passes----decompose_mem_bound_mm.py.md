# `.\pytorch\torch\_inductor\fx_passes\decompose_mem_bound_mm.py`

```py
# mypy: allow-untyped-defs
# 导入日志模块
import logging
# 导入类型注解相关模块
from typing import List

# 导入 PyTorch 相关模块
import torch
from torch import Tensor
from torch._dynamo.utils import counters

# 导入配置模块
from .. import config

# 导入模式匹配相关模块
from ..pattern_matcher import Arg, CallFunction, Match, register_graph_pattern
# 导入用于模式匹配的函数
from .split_cat import construct_pattern_matcher_pass

# 使用 torch.ops.aten 别名表示 torch 的 aten 操作
aten = torch.ops.aten
# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 定义用于矩阵分解的最小和最大维度常量
MIN_FIRST_DIMENSION_DECOMPOSITION = 10240
MAX_OTHER_DIMENSION_DECOMPOSITION = 32

# 根据配置设置最小和最大维度的分解阈值
min_first_dimension_decomposition = MIN_FIRST_DIMENSION_DECOMPOSITION
max_other_dimention_decomposition = MAX_OTHER_DIMENSION_DECOMPOSITION
if "decompose_mm_pass" in config.post_grad_fusion_options:
    # 如果配置中定义了特定的矩阵乘法分解选项，则更新阈值
    min_first_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("min_first_dimension_decomposition", MIN_FIRST_DIMENSION_DECOMPOSITION)
    max_other_dimention_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("max_other_dimention_decomposition", MAX_OTHER_DIMENSION_DECOMPOSITION)


# 检查两个张量是否在同一 CUDA 设备上
def check_device(a: Tensor, b: Tensor) -> bool:
    return a.is_cuda and b.is_cuda


# 给定一组 FX 节点，将其标记为实际输入
def realize_inputs(inputs: List[torch.fx.Node]):
    for inp in inputs:
        if isinstance(inp, torch.fx.node.Node):
            # 设置节点的元数据指示需要实际化为步幅
            inp.meta["inductor_realize_to_strides"] = True


# 判断是否应该对两个张量进行分解优化
def should_decompose_bmm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if not check_device(mat1, mat2):
        return False
    else:
        if len(mat1.shape) != 3 or len(mat2.shape) != 3:
            return False
        if mat1.shape[0] < min_first_dimension_decomposition:
            return False
        # 至少有两个维度的尺寸需小于 MAX_OTHER_DIMENSION_DECOMPOSITION
        if (mat1.shape[1] < max_other_dimention_decomposition) + (
            mat1.shape[2] < max_other_dimention_decomposition
        ) + (mat2.shape[2] < max_other_dimention_decomposition) < 2:
            return False
    return True


# 判断是否应该对两个矩阵进行分解优化
def should_decompose_mm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    return (
        check_device(mat1, mat2)
        and len(mat1.shape) == 2
        and len(mat2.shape) == 2
        and mat1.shape[0] >= min_first_dimension_decomposition
        and mat2.shape[0] < max_other_dimention_decomposition
        and mat2.shape[1] < max_other_dimention_decomposition
    )


# 检查 FX 节点的元数据是否有效
def is_node_meta_valid(node: torch.fx.Node):
    return "val" in node.meta


# 打印矩阵分解的模式匹配结果
def print_decompose_pattern(match: Match, inputs: List[torch.fx.Node]):
    node = match.nodes[-1]
    log.debug(
        "Decompose %s with input shape: %s",
        node.target,
        ", ".join(
            str(input.meta["val"].shape) if "val" in input.meta else "None"
            for input in inputs
        ),
    )
# 注册图模式，用于识别 torch.fx 中调用 torch 模块函数 aten.bmm 的模式
# 并应用名为 "decompose_mm_pass" 的模式匹配器传递参数
@register_graph_pattern(
    CallFunction(aten.bmm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_bmm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    # 定义替换函数 repl，用于分解 mat1 和 mat2 的乘积操作
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, :, None] * mat2[:, None, :, :], dim=-2)

    # 如果应该分解 bmm 操作，则执行以下代码块
    if should_decompose_bmm(mat1, mat2):
        # 增加统计计数器，记录分解 bmm 操作的次数
        counters["inductor"]["decompose_bmm"] += 1
        # 使用 repl 函数替换当前匹配的模式
        match.replace_by_example(repl, [mat1, mat2])
        # 打印分解的模式匹配信息
        print_decompose_pattern(match, [mat1, mat2])
        # 实现输入节点 mat1 和 mat2
        realize_inputs([mat1, mat2])
    return


# 注册图模式，用于识别 torch.fx 中调用 torch 模块函数 aten.addmm 的模式
# 并应用名为 "decompose_mm_pass" 的模式匹配器传递参数
@register_graph_pattern(
    CallFunction(aten.addmm, Arg(), Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_addmm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
    mat3: torch.fx.Node,
):
    # 定义替换函数 repl，用于分解 addmm 操作
    def repl(mat1, mat2, mat3):
        return torch.sum(mat2[:, :, None] * mat3[None, :, :], dim=-2) + mat1

    # 如果应该分解 mm 操作，则执行以下代码块
    if should_decompose_mm(mat2, mat3):
        # 增加统计计数器，记录分解 addmm 操作的次数
        counters["inductor"]["decompose_addmm"] += 1
        # 使用 repl 函数替换当前匹配的模式
        match.replace_by_example(repl, [mat1, mat2, mat3])
        # 打印分解的模式匹配信息
        print_decompose_pattern(match, [mat1, mat2, mat3])
        # 实现输入节点 mat1, mat2 和 mat3
        realize_inputs([mat1, mat2, mat3])
    return


# 注册图模式，用于识别 torch.fx 中调用 torch 模块函数 aten.mm 的模式
# 并应用名为 "decompose_mm_pass" 的模式匹配器传递参数
@register_graph_pattern(
    CallFunction(aten.mm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_mm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
):
    # 定义替换函数 repl，用于分解 mm 操作
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, None] * mat2[None, :, :], dim=-2)

    # 如果应该分解 mm 操作，则执行以下代码块
    if should_decompose_mm(mat1, mat2):
        # 增加统计计数器，记录分解 mm 操作的次数
        counters["inductor"]["decompose_mm"] += 1
        # 使用 repl 函数替换当前匹配的模式
        match.replace_by_example(repl, [mat1, mat2])
        # 打印分解的模式匹配信息
        print_decompose_pattern(match, [mat1, mat2])
        # 实现输入节点 mat1 和 mat2
        realize_inputs([mat1, mat2])
    return
```