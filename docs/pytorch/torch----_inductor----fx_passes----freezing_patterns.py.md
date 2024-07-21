# `.\pytorch\torch\_inductor\fx_passes\freezing_patterns.py`

```
# 设置类型检查工具允许未注释的函数定义
mypy: allow-untyped-defs

# 导入 functools 模块
import functools

# 导入 torch 模块
import torch

# 导入 torch._inductor.compile_fx 模块的 fake_tensor_prop 函数
from torch._inductor.compile_fx import fake_tensor_prop

# 导入 ..._dynamo.utils 模块的 counters 对象
from ..._dynamo.utils import counters

# 导入 .. 模块的 config 对象
from .. import config

# 导入 ..pattern_matcher 模块的各种函数和类
from ..pattern_matcher import (
    _return_true,
    CallFunction,
    fwd_only,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)

# 引用 torch.ops.aten 到 aten 变量
aten = torch.ops.aten

# 创建 pass_patterns 列表，包含三个 PatternMatcherPass 对象
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]

# 创建 binary_folding_pass 对象作为 PatternMatcherPass 实例
binary_folding_pass = PatternMatcherPass()

# 定义 freezing_passes 函数，接受 torch.fx.GraphModule 和 aot_example_inputs 两个参数
def freezing_passes(gm: torch.fx.GraphModule, aot_example_inputs):
    """
    Passes that are applied to the graph to freeze pass.
    """

    # 导入 ..freezing 模块的 constant_fold 函数
    from ..freezing import constant_fold

    # 调用 lazy_init 函数
    lazy_init()

    # 通过多轮二进制折叠来消除不必要的节点
    # 进行几轮二进制折叠以尽量消除不必要的节点，但可能需要一个好的方法来选择轮数。
    # 运行顺序为：conv+binary+binary。
    binary_folding = counters["inductor"]["binary_folding"]
    fake_tensor_prop(gm, aot_example_inputs, True)

    # 标记允许混合数据类型的卷积
    torch._inductor.fx_passes.binary_folding.mark_mixed_dtype_allowed_convs(gm)

    # 进行四轮常量折叠
    for _ in range(4):
        constant_fold(gm)
        # 确保所有节点的 meta['val'] 属性正确设置
        fake_tensor_prop(gm, aot_example_inputs, True)
        binary_folding_pass.apply(gm.graph)  # type: ignore[arg-type]
        # 如果没有进行二进制折叠，就不需要再运行该 pass 了。
        # TODO: 移除在整个模型上运行 fake_tensor_prop 的需要。
        if counters["inductor"]["binary_folding"] == binary_folding:
            break
        binary_folding = counters["inductor"]["binary_folding"]

    # 恢复被折叠卷积的原始精度
    torch._inductor.fx_passes.binary_folding.recover_original_precision_folded_convs(gm)

    # 再次进行常量折叠和 tensor 属性的伪推断
    constant_fold(gm)
    fake_tensor_prop(gm, aot_example_inputs, True)

    # 对 pass_patterns 中的每一个 PatternMatcherPass 对象应用到图上
    for pattern in pass_patterns:
        pattern.apply(gm.graph)  # type: ignore[arg-type]

    # 如果在 CPU 上进行权重打包，确保在执行时 layout_optimization 是开启的。
    if (
        torch._C._has_mkldnn
        and config.cpp.weight_prepack
        and config.layout_optimization
    ):
        # 导入 .mkldnn_fusion 模块的 _eliminate_duplicate_packed_nodes 函数
        from .mkldnn_fusion import _eliminate_duplicate_packed_nodes

        # 执行 _eliminate_duplicate_packed_nodes 函数
        _eliminate_duplicate_packed_nodes(gm)

    # 对图进行稳定的拓扑排序
    stable_topological_sort(gm.graph)

    # 重新编译图
    gm.recompile()

    # 对图进行静态检查
    gm.graph.lint()


# 装饰 lazy_init 函数，确保只初始化一次
@init_once_fakemode
def lazy_init():
    # 如果支持 MKLDNN 并且配置允许权重预打包
    if torch._C._has_mkldnn and config.cpp.weight_prepack:
        # 导入 .mkldnn_fusion 模块的 _mkldnn_weight_pack_init 函数
        from .mkldnn_fusion import _mkldnn_weight_pack_init

        # 执行 _mkldnn_weight_pack_init 函数
        _mkldnn_weight_pack_init()

    # 导入 .binary_folding 模块的 binary_folding_init 函数
    from .binary_folding import binary_folding_init

    # 执行 addmm_patterns_init 函数
    addmm_patterns_init()

    # 执行 binary_folding_init 函数
    binary_folding_init()


# 定义 register_freezing_graph_pattern 函数，用于注册冻结图模式
def register_freezing_graph_pattern(pattern, extra_check=_return_true, pass_number=0):
    # 调用函数 register_graph_pattern，注册图模式并返回结果
    return register_graph_pattern(
        pattern,                     # 第一个参数：图模式对象 pattern
        extra_check=extra_check,     # 关键字参数：额外的检查条件 extra_check
        pass_dict=pass_patterns[pass_number],  # 关键字参数：通过 pass_number 获取 pass_patterns 中的值
    )
# 注册二进制折叠模式的图模式，用于处理给定的模式
def register_binary_folding_pattern(pattern, extra_check=_return_true):
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        pass_dict=binary_folding_pass,
    )


# 使用 functools.lru_cache(None) 装饰器，初始化添加矩阵乘法模式
@functools.lru_cache(None)
def addmm_patterns_init():
    # 检查当前环境是否支持 CUDA，选择设备 "cuda" 或 "cpu"
    if torch.cuda.is_available():
        # 解决 https://github.com/pytorch/pytorch/issues/97894 的问题
        device = "cuda"
    else:
        device = "cpu"
    
    # 创建一个部分应用了 torch.empty 函数的 val 函数，指定设备和梯度不需求
    val = functools.partial(torch.empty, (10, 10), device=device, requires_grad=False)

    # 检查拼接权重的函数，验证输入权重的形状是否一致
    def check_concat_weights(match):
        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        equal_shape_inputs = [weight_inputs]

        if "b1" in match.kwargs:
            bias_inputs = ["b1", "b2"]
            if "b3" in match.kwargs:
                bias_inputs.append("b3")

            equal_shape_inputs.append(bias_inputs)

        for equal_shape_group in equal_shape_inputs:
            inps = [match.kwargs[name] for name in equal_shape_group]

            # 检查所有输入是否都是 "get_attr" 操作，并且形状相同
            if not all(
                inp.op == "get_attr"
                and inp.meta["val"].shape == inps[0].meta["val"].shape
                for inp in inps
            ):
                return False

        return True

    # 定义矩阵乘法融合模式和其替换函数
    def matmul_fuse_pattern(inp, w1, w2, w3):
        return (inp @ w1, inp @ w2, inp @ w3)

    def matmul_replacement(inp, w1, w2, w3):
        cat_t = torch.cat((w1, w2, w3), dim=1)
        mm = inp @ cat_t
        return mm.chunk(3, dim=1)

    # 注册矩阵乘法融合模式和替换函数
    register_replacement(
        matmul_fuse_pattern,
        matmul_replacement,
        [val(), val(), val(), val()],
        fwd_only,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3"),
    )

    # 定义另一个矩阵乘法融合模式和其替换函数
    def matmul_fuse_pattern_two(inp, w1, w2):
        return (inp @ w1, inp @ w2)

    def matmul_replacement_two(inp, w1, w2):
        cat_t = torch.cat((w1, w2), dim=1)
        mm = inp @ cat_t
        return mm.chunk(2, dim=1)

    # 注册另一个矩阵乘法融合模式和替换函数
    register_replacement(
        matmul_fuse_pattern_two,
        matmul_replacement_two,
        [val(), val(), val()],
        fwd_only,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2"),
    )

    # 定义 addmm 操作融合模式和其替换函数
    def addmm_fuse_pattern_second(inp, w1, w2, w3, b1, b2, b3):
        return (
            aten.addmm(b1, inp, w1),
            aten.addmm(b2, inp, w2),
            aten.addmm(b3, inp, w3),
        )

    def addmm_fuse_replacement_second(inp, w1, w2, w3, b1, b2, b3):
        cat_w = torch.cat((w1, w2, w3), dim=1)
        cat_b = torch.cat((b1, b2, b3))
        return aten.addmm(cat_b, inp, cat_w).chunk(3, dim=1)

    # 注册 addmm 操作融合模式和替换函数
    register_replacement(
        addmm_fuse_pattern_second,
        addmm_fuse_replacement_second,
        [val() for _ in range(7)],
        fwd_only,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3", "b1", "b2", "b3"),
    )


# 定义一个函数 same_dtype，暂时未提供其具体实现
def same_dtype(match):
    # 返回一个布尔值，判断 match 输出节点的第一个参数的元数据 "val" 的数据类型是否与 match 的关键字参数 "dtype" 相匹配
    return match.output_node().args[0].meta["val"].dtype == match.kwargs["dtype"]
# 注册一个图模式，用于识别和处理特定的函数调用图模式
@register_graph_pattern(
    # 匹配调用 torch.ops.prims.convert_element_type.default 函数
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        # 忽略第一个参数
        Ignored(),
        # 匹配关键字参数 "dtype"
        KeywordArg("dtype"),
    ),
    # 传递字典给 pass_dict 参数，使用 pass_patterns 列表的第一个模式
    pass_dict=pass_patterns[0],
    # 额外检查条件，使用 same_dtype 函数
    extra_check=same_dtype,
)
# 定义一个函数 unnecessary_dtype_convert，用于移除不必要的 dtype 转换操作，通常因为 Conv-Bn 折叠而留下
def unnecessary_dtype_convert(match: Match, **kwargs):
    """Remove unnecessary dtype conversion op, probably left as a result of Conv-Bn folding"""
    # 获取匹配对象的图
    graph = match.graph
    # 获取输出节点
    node = match.output_node()
    # 用第一个参数替换节点的所有使用
    node.replace_all_uses_with(node.args[0])
    # 从图中擦除该节点
    graph.erase_node(node)
```