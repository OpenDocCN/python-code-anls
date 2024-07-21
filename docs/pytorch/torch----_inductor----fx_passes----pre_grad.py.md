# `.\pytorch\torch\_inductor\fx_passes\pre_grad.py`

```
# 设置 mypy 以允许未类型化的定义
import copy  # 导入 copy 模块，用于复制对象
import itertools  # 导入 itertools 模块，用于迭代器操作
import logging  # 导入 logging 模块，用于日志记录
from typing import Dict, List, Optional  # 导入类型提示相关模块

import torch  # 导入 PyTorch 深度学习库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch._dynamo.utils import counters, detect_fake_mode, optimus_scuba_log  # 导入私有工具函数
from torch._utils_internal import upload_graph  # 导入内部工具函数
from torch.fx.experimental.optimization import (
    matches_module_pattern,  # 导入用于模块匹配的函数
    replace_node_module,  # 导入替换节点模块的函数
)
from torch.fx.passes.graph_transform_observer import GraphTransformObserver  # 导入图形转换观察器类
from torch.fx.passes.shape_prop import ShapeProp  # 导入形状属性类
from torch.nn import functional as F  # 导入神经网络的函数模块作为 F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights  # 导入融合卷积和批归一化的函数

from .. import config  # 导入相对于当前模块的配置

from ..fx_utils import matches_module_function_pattern  # 导入用于模块函数匹配的函数
from ..pattern_matcher import (
    init_once_fakemode,  # 导入初始化一次的伪模式函数
    PatternMatcherPass,  # 导入模式匹配器通行证类
    stable_topological_sort,  # 导入稳定的拓扑排序函数
)
from ..utils import is_cpu_device, pass_execution_and_save  # 导入工具函数：判断是否为 CPU 设备，执行和保存通行证

from .group_batch_fusion import group_batch_fusion_passes, PRE_GRAD_FUSIONS  # 导入批次融合组函数和预梯度融合
from .misc_patterns import numpy_compat_normalization  # 导入杂项模式：NumPy 兼容性归一化函数
from .split_cat import PRE_GRAD_PATTERNS  # 导入分割和连接模式

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

efficient_conv_bn_eval_pass = PatternMatcherPass(  # 创建模式匹配通行证：高效卷积批归一化评估
    prevent_match_across_mutations=True,  # 防止跨变异进行匹配
    pass_name="efficient_conv_bn_eval_pass",  # 设置通行证名称
)

fuse_split_linear_add_pass = PatternMatcherPass(  # 创建模式匹配通行证：融合分割线性加法
    prevent_match_across_mutations=True,  # 防止跨变异进行匹配
    pass_name="fuse_split_linear_add_pass",  # 设置通行证名称
)
fuse_chunk_squeeze_cat_pass = PatternMatcherPass(  # 创建模式匹配通行证：融合块挤压和连接
    prevent_match_across_mutations=True,  # 防止跨变异进行匹配
    pass_name="fuse_chunk_squeeze_cat_pass",  # 设置通行证名称
)
remove_reshape_pass = PatternMatcherPass(  # 创建模式匹配通行证：移除重塑
    prevent_match_across_mutations=True,  # 防止跨变异进行匹配
    pass_name="remove_reshape_pass",  # 设置通行证名称
)

# 基于 predispatch aten IR
normalization_pass_aten = PatternMatcherPass(prevent_match_across_mutations=True)  # 创建模式匹配通行证：标准化通行证
merge_splits_pass_aten = PatternMatcherPass(prevent_match_across_mutations=True)  # 创建模式匹配通行证：合并分割通行证
split_cat_pass_aten = PatternMatcherPass(prevent_match_across_mutations=True)  # 创建模式匹配通行证：分割连接通行证
unbind_stack_pass_aten = PatternMatcherPass(prevent_match_across_mutations=True)  # 创建模式匹配通行证：解绑栈通行证
merge_getitem_cat_pass_aten = PatternMatcherPass(prevent_match_across_mutations=True)  # 创建模式匹配通行证：合并 getitem 和连接通行证
merge_stack_tahn_unbind_pass_aten = PatternMatcherPass(  # 创建模式匹配通行证：合并栈、tanh 和解绑通行证
    prevent_match_across_mutations=True
)
mutate_cat_pass_aten = PatternMatcherPass(prevent_match_across_mutations=True)  # 创建模式匹配通行证：变异连接通行证
remove_split_with_size_one_pass_aten = PatternMatcherPass(  # 创建模式匹配通行证：移除大小为一的分割通行证
    prevent_match_across_mutations=True
)


def save_inductor_dict(pass_to_compare=None):
    if not pass_to_compare:
        pass_to_compare = list(config.pre_grad_fusion_options.keys()) + list(
            config.post_grad_fusion_options.keys()
        )
    return {p: dict(counters["inductor"]).get(p, 0) for p in pass_to_compare}
    # 返回一个字典，包含执行次数计数器中指定通行证的统计信息


def is_same_dict(inductor_dict, optimus_dict):
    for pass_name, count in optimus_dict.items():
        if count != dict(inductor_dict).get(pass_name, 0):
            return False
    return True
    # 检查两个字典是否相同，即 Optimus 优化器的通行证计数与执行次数计数器中的是否一致


def fuse_parallel_linear_pass(graph):
    return None
    # 对给定图形执行并行线性融合的通行证，当前未实现


def remove_split_ops(graph, shape_prop):
    return None
    # 从给定的图形中移除分割操作，当前未实现


pattern_matcher_passes_aten: List[PatternMatcherPass] = [
    # 定义模式匹配通行证列表，用于 aten 模块
    # 调用名为 `remove_split_with_size_one_pass_aten` 的函数或方法
    remove_split_with_size_one_pass_aten,
    
    # 调用名为 `merge_getitem_cat_pass_aten` 的函数或方法
    merge_getitem_cat_pass_aten,
    
    # 调用名为 `merge_stack_tahn_unbind_pass_aten` 的函数或方法
    merge_stack_tahn_unbind_pass_aten,
    
    # 调用名为 `merge_splits_pass_aten` 的函数或方法
    merge_splits_pass_aten,
    
    # 调用名为 `mutate_cat_pass_aten` 的函数或方法
    mutate_cat_pass_aten,
    
    # 调用名为 `split_cat_pass_aten` 的函数或方法
    split_cat_pass_aten,
    
    # 调用名为 `unbind_stack_pass_aten` 的函数或方法
    unbind_stack_pass_aten,
# 使用装饰器初始化一次性假模式
@init_once_fakemode
def lazy_init():
    # 导入效率卷积 BN 评估和分割类别模块
    from . import efficient_conv_bn_eval, split_cat  # noqa: F401  # noqa: F401

    # 如果配置为 FB 代码环境
    if config.is_fbcode():
        # 导入 FB 相关模块
        from . import fb  # type: ignore[attr-defined]  # noqa: F401


def pre_grad_passes(gm: torch.fx.GraphModule, example_inputs=None):
    """
    在输入的 Torch IR 图上应用 passes。

    警告：
    梯度前的 IR 并非功能性或标准化的，因此在此 IR 上编写 passes 较为困难。
    Passes 必须安全地处理别名和变异，并需要处理所有可能的参数模式。

    考虑将新的 passes 添加到 post_grad.py 或 joint_graph.py，这些文件在功能化和标准化之后。
    """
    # 如果配置指定了自定义梯度前 passes
    if config.pre_grad_custom_pass is not None:
        # 使用图转换观察器观察自定义 passes
        with GraphTransformObserver(
            gm, "pre_grad_custom_pass", config.trace.log_url_for_graph_xform
        ):
            # 在图上应用自定义 passes
            config.pre_grad_custom_pass(gm.graph)
    
    # 对图进行稳定的拓扑排序
    stable_topological_sort(gm.graph)

    # 导入量化模块，将量化 passes 应用于图
    from .quantization import quant_lift_up
    quant_lift_up(gm)

    # 对图进行 lint
    gm.graph.lint()

    # 重新编译图
    gm.recompile()

    # 上传图到优化 SCUBA 日志中
    optimus_scuba_log["after_recompile_pre_grad"] = upload_graph(gm.graph)

    # 如果配置了模式匹配器，并且有数值检查 passes
    if (
        config.pattern_matcher
        and hasattr(config, "fx_passes_numeric_check")
        and config.fx_passes_numeric_check.get("pre_grad", False)
        and example_inputs is not None
    ):
        # 导入数值工具模块，进行数值检查
        from .numeric_utils import numeric_check_if_enabled

        # 复制前的 FX passes
        gm_after_fx_passes = gm.__copy__()
        numeric_check_if_enabled(
            gm_before_fx_passes,  # type: ignore[possibly-undefined]
            gm_after_fx_passes,
            example_inputs,
            config.fx_passes_numeric_check.get("num_iterations", 1),
            config.fx_passes_numeric_check.get("precision", 1e-4),
        )

    # 返回处理后的图模块
    return gm


def fuse_fx(gm: torch.fx.GraphModule, example_inputs) -> torch.fx.GraphModule:
    # 检测输入是否为 CPU
    is_cpu = is_cpu_device(example_inputs)
    
    # 检测假模式
    # pyre-fixme[16]: Module `torch._dynamo.utils` has no attribute `detect_fake_mode`
    fake_mode = detect_fake_mode(example_inputs)

    # 将 sink_cat 应用于 pointwise 之后的图
    gm = sink_cat_after_pointwise(gm)

    # 如果配置允许置换融合，并且不是 CPU 环境
    if config.permute_fusion and not is_cpu:
        # 为线性置换融合，检查输入信息以识别并执行适当的置换/转置
        ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)

        # 使用图转换观察器观察线性置换融合
        with GraphTransformObserver(
            gm, "linear_permute_fusion", config.trace.log_url_for_graph_xform
        ):
            gm = linear_permute_fusion(gm)

        # 使用图转换观察器观察置换线性融合
        with GraphTransformObserver(
            gm, "permute_linear_fusion", config.trace.log_url_for_graph_xform
        ):
            gm = permute_linear_fusion(gm)

        # 使用图转换观察器观察置换矩阵乘法融合
        with GraphTransformObserver(
            gm, "permute_matmul_fusion", config.trace.log_url_for_graph_xform
        ):
            gm = permute_matmul_fusion(gm)

    # 确保自动梯度已禁用或不是 CPU 环境
    if torch.is_grad_enabled() or not is_cpu:
        return gm
    # 如果配置中设置了 freezing 标志为 True，则执行以下代码块
    if config.freezing:
        # 使用 GraphTransformObserver 监视 gm 对象，记录转换操作到日志中
        with GraphTransformObserver(
            gm, "remove_identity", config.trace.log_url_for_graph_xform
        ):
            # 对 gm 执行 remove_identity 转换操作，并更新 gm
            gm = remove_identity(gm)
        
        # 使用 GraphTransformObserver 监视 gm 对象，记录转换操作到日志中
        with GraphTransformObserver(
            gm, "fuse_conv_bn", config.trace.log_url_for_graph_xform
        ):
            # 对 gm 执行 fuse_conv_bn 转换操作，并更新 gm
            gm = fuse_conv_bn(gm)
    
    # 返回经过转换操作后的 gm 对象
    return gm
def fetch_attr(target: str, mod):
    # 将目标字符串按点号分割为列表
    target_atoms = target.split(".")
    # 初始化属性迭代器为给定模块对象
    attr_itr = mod
    # 遍历目标字符串中的每个属性
    for i, atom in enumerate(target_atoms):
        # 检查当前属性是否存在于当前迭代对象中
        if not hasattr(attr_itr, atom):
            # 如果不存在，抛出运行时异常，指示找不到指定的属性路径
            raise RuntimeError(
                f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
            )
        # 更新属性迭代器为当前属性的值
        attr_itr = getattr(attr_itr, atom)
    # 返回最终获取的属性值
    return attr_itr


def remove_identity(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Removes all identity layers from the module.
    """

    class IdentityRemover(torch.fx.Transformer):
        def call_module(self, target, args, kwargs):
            # 如果目标模块是 nn.Identity，直接返回其输入作为结果
            if isinstance(self.submodules[target], nn.Identity):
                assert len(args) == 1
                return args[0]
            else:
                # 否则，调用父类方法继续处理模块
                return super().call_module(target, args, kwargs)

    # 返回经过 IdentityRemover 处理后的 GraphModule 对象
    return IdentityRemover(gm).transform()


def fuse_conv_bn(gm: torch.fx.GraphModule, inplace=False) -> torch.fx.GraphModule:
    """
    Fuses Convolution/BN layers for inference purposes.
    """
    # 定义卷积层和对应的 BatchNorm 层的模式列表
    modules_patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    # 定义卷积层和对应的 Functional BatchNorm 的模式列表
    module_function_patterns = [
        (torch.nn.Conv1d, F.batch_norm),
        (torch.nn.Conv2d, F.batch_norm),
        (torch.nn.Conv3d, F.batch_norm),
    ]
    # 获取模块中所有命名模块的字典
    modules = dict(gm.named_modules())

    class ConvBNFusion:
        def __init__(
            self,
            bn_node,
            conv_module,
            bn_module=None,  # For BN Module
            bn_running_mean=None,  # For Functional BN
            bn_running_var=None,
            bn_eps=None,
            bn_weight=None,
            bn_bias=None,
        ):
            # 初始化 BatchNorm 节点列表、卷积模块和相关参数
            self.bn_nodes = [
                bn_node,
            ]
            self.conv_module = conv_module
            self.bn_module = bn_module
            self.bn_running_mean = bn_running_mean
            self.bn_running_var = bn_running_var
            self.bn_eps = bn_eps
            self.bn_weight = bn_weight
            self.bn_bias = bn_bias
            self.fusion_enabled = True

        def add_bn_node(self, bn_node):
            # 添加额外的 BatchNorm 节点到列表中
            self.bn_nodes.append(bn_node)

        def disable_fusion(self):
            # 禁用融合功能标志
            self.fusion_enabled = False

        def is_fusion_enabled(self):
            # 返回当前融合功能是否启用的标志
            return self.fusion_enabled

    # 初始化用于融合的字典，键为整数索引，值为 ConvBNFusion 实例
    conv_bn_to_fuse: Dict[int, ConvBNFusion] = {}
    # 遍历模块匹配模式列表
    for pattern in modules_patterns:
        # 清空用于存储卷积和批归一化融合信息的字典
        conv_bn_to_fuse.clear()
        # 遍历计算图中的每个节点
        for node in gm.graph.nodes:
            # 如果当前节点匹配给定模式且是卷积模块的输出节点
            if matches_module_pattern(pattern, node, modules):
                # 如果卷积模块的输出被多个节点使用，则跳过
                if len(node.args[0].users) > 1:
                    continue
                # 获取卷积和批归一化模块
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                # 检查卷积和批归一化模块是否都处于评估模式
                eval_mode = all(not n.training for n in [conv, bn])
                if not eval_mode:
                    continue
                # 如果批归一化模块未在追踪运行统计信息，则跳过
                if not bn.track_running_stats:
                    continue

                # 根据卷积模块的模块名进行哈希计算
                hash_id = hash(node.args[0].target)
                # 如果哈希值不存在于融合字典中，则创建新的融合对象
                if hash_id not in conv_bn_to_fuse:
                    conv_bn_to_fuse[hash_id] = ConvBNFusion(node, conv, bn)
                else:
                    # 如果批归一化模块相同，则进行融合操作
                    if bn == conv_bn_to_fuse[hash_id].bn_module:
                        conv_bn_to_fuse[hash_id].add_bn_node(node)
                    else:
                        # 如果卷积模块被不同的批归一化模块共享，则禁用融合
                        conv_bn_to_fuse[hash_id].disable_fusion()

        # 对于每个融合对象，检查是否启用了融合
        for conv_bn_fusion in conv_bn_to_fuse.values():
            if conv_bn_fusion.is_fusion_enabled():
                bn_nodes = conv_bn_fusion.bn_nodes
                conv = conv_bn_fusion.conv_module
                bn = conv_bn_fusion.bn_module

                # 执行卷积和批归一化的融合操作
                fused_conv = fuse_conv_bn_eval(conv, bn)
                # 替换所有批归一化节点的模块为融合后的卷积模块
                for bn_node in bn_nodes:
                    replace_node_module(bn_node.args[0], modules, fused_conv)
                    bn_node.replace_all_uses_with(bn_node.args[0])
                    # 在计算图中删除批归一化节点
                    gm.graph.erase_node(bn_node)

    # 执行计算图的静态分析
    gm.graph.lint()
    # 再次执行计算图的静态分析，确保稳定性
    gm.graph.lint()
    # 重新编译计算图以应用所有更改
    gm.recompile()

    # 返回更新后的计算图模型
    return gm
class NormalizedLinearNode:
    # 初始化方法，接受一个 torch.fx.Node 对象作为参数
    def __init__(self, node: torch.fx.Node) -> None:
        # 断言节点操作为调用函数
        assert node.op == "call_function"
        # 断言调用的目标函数在 torch.nn.functional.linear 中
        assert node.target in [torch.nn.functional.linear]
        # 将传入的节点对象赋值给实例变量
        self.node: torch.fx.Node = node

    # 获取输入节点的方法
    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]  # 返回第一个参数节点
        else:
            return self.node.kwargs["input"]  # 返回关键字参数中的 input 节点

    # 获取权重节点的方法
    def get_weight(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]  # 返回第二个参数节点
        else:
            return self.node.kwargs["weight"]  # 返回关键字参数中的 weight 节点

    # 获取偏置节点的方法
    def get_bias(self) -> torch.fx.Node:
        if len(self.node.args) > 2:
            return self.node.args[2]  # 返回第三个参数节点
        else:
            # 如果关键字参数中存在 bias，则返回其节点；否则返回 None
            return self.node.kwargs["bias"] if "bias" in self.node.kwargs else None


class NormalizedMatmulNode:
    # 初始化方法，接受一个 torch.fx.Node 对象作为参数
    def __init__(self, node: torch.fx.Node) -> None:
        # 断言节点操作为调用函数
        assert node.op == "call_function"
        # 断言调用的目标函数在 torch.bmm 或 torch.matmul 中
        assert node.target in [torch.bmm, torch.matmul]
        # 将传入的节点对象赋值给实例变量
        self.node: torch.fx.Node = node

    # 获取输入节点的方法
    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]  # 返回第一个参数节点
        else:
            return self.node.kwargs["input"]  # 返回关键字参数中的 input 节点

    # 获取其他节点的方法
    def get_other(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]  # 返回第二个参数节点
        else:
            return self.node.kwargs["other"]  # 返回关键字参数中的 other 节点


# 检查节点是否有正确的置换
def check_permute(node: torch.fx.Node) -> bool:
    # 获取节点的张量元数据的秩
    ranks = len(node.meta["tensor_meta"].shape)
    if len(node.args) > 3:
        # 如果节点参数个数大于3，根据参数来创建置换列表
        permutation = [node.args[i] % ranks for i in range(1, ranks + 1)]  # 计算参数的取模值
    elif (
        "permutation" in node.kwargs
        and node.kwargs["permutation"] is not None
        and len(node.kwargs["permutation"]) > 2
    ):
        # 如果关键字参数中有 permutation，并且其长度大于2，创建置换列表
        permutation = [i % ranks for i in node.kwargs["permutation"]]
    else:
        return False
    # 允许的置换列表，最后两个位置交换
    allowed_permutation = list(range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    # 返回节点的置换列表与允许的置换列表是否相等的比较结果
    return permutation == allowed_permutation


# 将 concat 操作移到逐点操作之后的方法
def sink_cat_after_pointwise(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # 获取节点的唯一用户
    def one_user(node):
        users = list(node.users)
        return users[0] if len(users) == 1 else None

    # 判断节点是否为 view 方法调用
    def is_view(node):
        view = {"view"}
        return node.op == "call_method" and node.target in view

    # 判断节点是否为逐点操作（pointwise unary）的方法调用
    def is_pointwise_unary(node):
        pointwise = {torch.relu, torch.tanh, "relu", "tanh"}
        return node.op in {"call_function", "call_method"} and node.target in pointwise

    # 获取模块的计算图
    g = module.graph
    # 遍历图中的所有节点
    for node in g.nodes:
        # 如果节点不是调用函数或目标不是 torch.cat，则继续下一个节点
        if node.op != "call_function" or node.target != torch.cat:
            continue
        
        # 将当前节点标记为可能是 torch.cat 的节点
        cat_or_view = node
        
        # 进入循环，直到找到不是视图的用户节点
        while True:
            # 获取当前节点的用户节点
            user = one_user(cat_or_view)
            # 如果没有用户或用户不是视图，则中断循环
            if not user or not is_view(user):
                break
            # 更新当前节点为用户节点
            cat_or_view = user
        
        # 如果找到用户节点且用户节点是逐点一元操作
        if user and is_pointwise_unary(user):
            # 在当前节点之前插入新节点
            with g.inserting_before(node):
                
                # 定义一个函数 cat_args，返回其参数 tensors 和维度 dim
                def cat_args(tensors, dim=0):
                    return tensors, dim
                
                # 调用 cat_args 函数并获取返回的 tensors 和 dim
                tensors, dim = cat_args(*node.args, **node.kwargs)
                
                # 创建一个新的关键字参数字典，排除掉名称为 "input" 的参数
                new_kwargs = {
                    name: val for name, val in user.kwargs.items() if name != "input"
                }
                
                # 根据 tensors 创建新的节点列表，每个节点使用用户节点的操作和目标
                new_tensors = [
                    g.create_node(user.op, user.target, args=(arg,), kwargs=new_kwargs)
                    for arg in tensors
                ]
                
                # 创建一个新的 torch.cat 节点，将新节点列表作为参数
                new_cat = g.create_node(
                    "call_function", torch.cat, args=(new_tensors, dim)
                )
                
                # 替换用户节点所有使用为 cat_or_view 节点
                user.replace_all_uses_with(cat_or_view)
                
                # 替换当前节点所有使用为新创建的 torch.cat 节点
                node.replace_all_uses_with(new_cat)
                
                # 删除用户节点和当前节点
                g.erase_node(user)
                g.erase_node(node)
    
    # 对图进行 lint 检查
    g.lint()
    
    # 重新编译模块
    module.recompile()
    
    # 返回重新编译后的模块
    return module
# 将输入的 torch.fx.GraphModule 对象进行线性变换和置换融合优化
def linear_permute_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # 遍历图中所有调用 permute 方法的节点
    for node in module.graph.find_nodes(op="call_method", target="permute"):
        # 检查 permute 节点是否符合优化条件
        if check_permute(node):
            # 确定 permute 方法的输入节点
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            # 检查输入节点是否是调用 torch.nn.functional.linear 方法
            if (
                input_node.op == "call_function"
                and input_node.target == torch.nn.functional.linear
            ):
                # 对输入节点进行规范化处理，获取其输入、权重和偏置
                normalized = NormalizedLinearNode(input_node)
                input = normalized.get_input()
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                # 在当前节点之前插入优化后的线性变换节点
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        linear_transpose, args=(input, weight, bias)
                    )
                    # 将原 permute 节点的所有使用替换为新的融合节点
                    node.replace_all_uses_with(fused_node)
                    # 移除原 permute 节点
                    module.graph.erase_node(node)
                    # 如果输入节点不再被使用，也移除该节点
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    # 对图进行静态分析，确保图的结构正确
    module.graph.lint()
    # 重新编译优化后的模块
    module.recompile()
    # 返回优化后的模块
    return module


# 执行线性变换的函数，计算 Y = X * W^T 或 Y = X * W^T + bias
def linear_transpose(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    # 如果没有偏置，执行权重和输入的转置矩阵乘法
    if bias is None:
        return torch.matmul(weight, input.transpose(-1, -2))
    # 否则执行加上偏置后的转置矩阵乘法
    return torch.matmul(weight, input.transpose(-1, -2)) + bias.unsqueeze(-1)


# 将输入的 torch.fx.GraphModule 对象进行置换和线性变换融合优化
def permute_linear_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # 遍历图中所有调用 torch.nn.functional.linear 方法的节点
    for node in module.graph.find_nodes(
        op="call_function", target=torch.nn.functional.linear
    ):
        # 确定 linear 方法的输入节点
        if len(node.args) > 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs["input"]
        # 检查输入节点是否是调用 permute 方法，并符合优化条件
        if (
            input_node.op == "call_method"
            and input_node.target == "permute"
            and check_permute(input_node)
        ):
            # 对节点进行规范化处理，获取其输入
            normalized = NormalizedLinearNode(node)
            if len(input_node.args) > 0:
                input = input_node.args[0]
            else:
                input = input_node.kwargs["input"]
            weight = normalized.get_weight()
            bias = normalized.get_bias()
            # 在当前节点之前插入优化后的线性转置节点
            with module.graph.inserting_before(node):
                fused_node = module.graph.call_function(
                    transpose_linear, args=(input, weight, bias)
                )
                # 将原 linear 节点的所有使用替换为新的融合节点
                node.replace_all_uses_with(fused_node)
                # 移除原 linear 节点
                module.graph.erase_node(node)
                # 如果输入节点不再被使用，也移除该节点
                if len(input_node.users) == 0:
                    module.graph.erase_node(input_node)

    # 对图进行静态分析，确保图的结构正确
    module.graph.lint()
    # 重新编译优化后的模块
    module.recompile()
    # 返回优化后的模块
    return module


def permute_matmul_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # 使用 itertools.chain 链接两个生成器，查找图中调用 torch.bmm 和 torch.matmul 的节点
    for node in itertools.chain(
        module.graph.find_nodes(op="call_function", target=torch.bmm),
        module.graph.find_nodes(op="call_function", target=torch.matmul),
    ):
        # 创建 NormalizedMatmulNode 实例，用于处理当前节点
        normalized = NormalizedMatmulNode(node)
        # 获取输入节点和另一个输入节点
        input_A_node = normalized.get_input()
        input_B_node = normalized.get_other()
        # 将 input_A_node 和 input_B_node 分别赋给 input_A 和 input_B
        input_A = input_A_node
        input_B = input_B_node
        # 初始化 Atrans 和 Btrans 为 False
        Atrans = Btrans = False
        
        # 检查 input_A_node 是否是 permute 方法调用且符合特定条件
        if (
            input_A_node.op == "call_method"
            and input_A_node.target == "permute"
            and check_permute(input_A_node)
        ):
            # 如果满足条件，标记 Atrans 为 True
            Atrans = True
            # 如果有位置参数，则将第一个参数赋给 input_A
            if len(input_A_node.args) > 0:
                input_A = input_A_node.args[0]  # type: ignore[assignment]
            else:
                # 否则，将 kwargs 中的 input 参数赋给 input_A
                input_A = input_A_node.kwargs["input"]  # type: ignore[assignment]

        # 检查 input_B_node 是否是 permute 方法调用且符合特定条件
        if (
            input_B_node.op == "call_method"
            and input_B_node.target == "permute"
            and check_permute(input_B_node)
        ):
            # 如果满足条件，标记 Btrans 为 True
            Btrans = True
            # 如果有位置参数，则将第一个参数赋给 input_B
            if len(input_B_node.args) > 0:
                input_B = input_B_node.args[0]  # type: ignore[assignment]
            else:
                # 否则，将 kwargs 中的 input 参数赋给 input_B
                input_B = input_B_node.kwargs["input"]  # type: ignore[assignment]

        # 如果 Atrans 或者 Btrans 任一为 True
        if Atrans or Btrans:
            # 在当前节点之前插入一个新节点，调用 transpose_matmul 函数
            with module.graph.inserting_before(node):
                fused_node = module.graph.call_function(
                    transpose_matmul,
                    args=(input_A, input_B, Atrans, Btrans),
                )
            # 用新节点替换当前节点的所有引用
            node.replace_all_uses_with(fused_node)
            # 移除当前节点
            module.graph.erase_node(node)
            # 如果 Atrans 为 True 且 input_A_node 没有其他用户引用，移除 input_A_node
            if Atrans and len(input_A_node.users) == 0:
                module.graph.erase_node(input_A_node)
            # 如果 Btrans 为 True 且 input_B_node 没有其他用户引用，移除 input_B_node
            if Btrans and len(input_B_node.users) == 0:
                module.graph.erase_node(input_B_node)

    # 对模型图进行 lint 检查
    module.graph.lint()
    # 重新编译模型
    module.recompile()
    # 返回更新后的模型
    return module
# X1 = X.permute(0, 2, 1)
# 对输入张量 X 进行维度置换，交换维度顺序为 (0, 2, 1)，即第1和第2维度互换
# 此操作后 X1 的形状与 X 的形状相同，但维度顺序不同

# Y1 = X1 * W1^T + bias1
# 使用置换后的输入 X1，与权重矩阵 W1 的转置进行矩阵乘法运算
# 如果没有偏置项 bias1，则结果为乘法运算的结果

# ---->
# 以上两行代码的优化版本，利用 PyTorch 的 transpose 方法直接在 matmul 中进行维度置换
# Y2 = X1.transpose(-1, -2) * W1^T + bias1

def transpose_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    if bias is None:
        # 如果没有偏置项，则返回输入张量按照最后两个维度转置后与权重矩阵转置的乘积
        return torch.matmul(input.transpose(-1, -2), weight.t())
    else:
        # 如果有偏置项，则返回输入张量按照最后两个维度转置后与权重矩阵转置的乘积，再加上偏置项
        return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def transpose_matmul(
    A: torch.Tensor, B: torch.Tensor, Atrans: bool, Btrans: bool
) -> torch.Tensor:
    if Atrans:
        # 如果 Atrans 为真，则对 A 进行最后两个维度的转置
        A = A.transpose(-1, -2)
    if Btrans:
        # 如果 Btrans 为真，则对 B 进行最后两个维度的转置
        B = B.transpose(-1, -2)
    # 返回经过可能的转置后的 A 与 B 的矩阵乘积结果
    return torch.matmul(A, B)
```