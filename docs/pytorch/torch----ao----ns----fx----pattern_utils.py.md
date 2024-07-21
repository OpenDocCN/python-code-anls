# `.\pytorch\torch\ao\ns\fx\pattern_utils.py`

```py
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的函数式接口模块
toq = torch.ops.quantized  # 导入 PyTorch 的量化操作

from torch.fx import GraphModule  # 从 torch.fx 中导入 GraphModule 类
from torch.fx.graph import Node  # 从 torch.fx.graph 中导入 Node 类

from torch.ao.quantization.backend_config import get_native_backend_config  # 从 torch.ao.quantization.backend_config 导入获取本地后端配置的函数
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers  # 从 torch.ao.quantization.fx.quantize_handler 导入获取量化处理程序模式的函数
from torch.ao.quantization.utils import getattr_from_fqn  # 从 torch.ao.quantization.utils 导入根据完全限定名获取属性的函数
from .ns_types import NSNodeTargetType  # 从当前目录下的 ns_types 模块中导入 NSNodeTargetType 类型
from torch.ao.quantization import (  # 从 torch.ao.quantization 导入以下模块：
    ObserverBase,  # 观察者基类
    FakeQuantizeBase,  # 伪量化基类
)

from typing import Dict, Tuple, Set, Callable, Any, Union, List  # 导入用于类型注解的模块


def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]],
) -> Set[Tuple[NSNodeTargetType, NSNodeTargetType]]:
    # TODO(future PR): allow customizations
    # TODO(future PR): reuse existing quantization mappings
    # TODO(future PR): add the rest of modules and ops here
    type_a_related_to_b: Set[Tuple[NSNodeTargetType, NSNodeTargetType]] = set()

    for s in base_name_to_sets_of_related_ops.values():
        s_list = list(s)
        # add every bidirectional pair
        for idx_0 in range(0, len(s_list)):
            for idx_1 in range(idx_0, len(s_list)):
                type_a_related_to_b.add((s_list[idx_0], s_list[idx_1]))
                type_a_related_to_b.add((s_list[idx_1], s_list[idx_0]))

    return type_a_related_to_b


NSFusionElType = Union[
    Callable,  # 可调用对象，如 call_function 或 call_module 类型，例如：F.linear 或 nn.Conv2d
    str,  # 方法名，例如："dequantize"
    Tuple[str, Any],  # 方法名和第一个参数的元组，例如：("to", torch.float16)
]
NSFusionType = Union[
    Tuple[NSFusionElType, NSFusionElType],  # 两个元素的元组，表示融合类型
    Tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType],  # 四个元素的元组，表示融合类型
]

def get_reversed_fusions() -> List[Tuple[NSFusionType, int]]:
    """
    Set of potential fusions, in reverse order.  The order is reversed
    to match how fusion patterns are defined in quantization code.

    Fusion format:
    ((fusion_op_0, fusion_op_1), base_op_idx)

    Where base_op_idx is the idx of the op we should use to match other related
    ops. Note: base_op_idx is specified in non-reverse order, i.e. a base_op_idx
    of 0 represents the first op in regular (non-reverse) order, 1 represents the
    second op, etc.
    """
    results: List[Tuple[NSFusionType, int]] = []

    # Possible syntaxes:
    # * single op: torch.nn.Conv2d
    # * multiple ops: (torch.nn.ReLU, torch.nn.Conv2d)
    # For fusions, we only care about patterns composed of multiple ops.
    # TODO(future PR): allow customizations from default patterns.
    all_quant_patterns = _get_pattern_to_quantize_handlers(get_native_backend_config())

    default_base_op_idx = 0
    # 遍历所有量化模式的键
    for quant_pattern in all_quant_patterns.keys():
        # TODO: 这是一个临时的解决方案，用于展开量化模式，以便与 ns 匹配函数配合使用，
        # 可能应该使用 torch.ao.quantization.fx.match_utils 中的 `_is_match` 来匹配这些模式

        # 如果量化模式是元组且长度为2，并且第二个元素也是元组且长度为2，则展开模式
        if isinstance(quant_pattern, tuple) and len(quant_pattern) == 2 and \
           isinstance(quant_pattern[1], tuple) and len(quant_pattern[1]) == 2:
            quant_pattern = (quant_pattern[0], quant_pattern[1][0], quant_pattern[1][1])

        # 只有包含多个操作的模式才会被视为融合模式，忽略只包含单个操作的模式
        # （它们会被匹配而不关心是否融合）
        if isinstance(quant_pattern, tuple):
            results.append((quant_pattern, default_base_op_idx))  # type: ignore[arg-type]

        # 对于每个模式，添加包含观察者和虚假量化器的额外模式
        # TODO（将来的 PR）：如果需要，实现对具有多个输出观察者的节点进行匹配
        for cls in (ObserverBase, FakeQuantizeBase):
            if isinstance(quant_pattern, tuple):
                new_pattern = (cls, *quant_pattern)
            else:
                new_pattern = (cls, quant_pattern)
            results.append((new_pattern, default_base_op_idx))  # type: ignore[arg-type]

    # 到此为止，results 包含类似以下形式的数值
    # [..., ((torch.nn.Relu, torch.nn.Conv2d), 0), ...]

    # 在量化融合映射中未指定用于匹配 fp16 模拟的模式。因此，现在在这里定义它们。
    fp16_em_base_op_idx = 1
    patterns_to_add = [
        # 线性-ReLU 的 fp16 模拟：
        # fp16_to_fp32 -> linear -> relu -> fp32_to_fp16
        ((("to", torch.float16), F.relu, F.linear, "dequantize"), fp16_em_base_op_idx,),
        # Conv-BN 融合（这发生在量化模式之外，因此在这里单独定义）
        ((nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
    ]

    # 将定义的模式添加到结果列表中
    for p in patterns_to_add:
        results.append(p)  # type: ignore[arg-type]
        results.append(((ObserverBase, *p[0]), p[1]))  # type: ignore[arg-type]
        results.append(((FakeQuantizeBase, *p[0]), p[1]))  # type: ignore[arg-type]

    # 返回最终的匹配结果列表
    return results
def end_node_matches_reversed_fusion(
    end_node: Node,
    reversed_fusion: NSFusionType,
    gm: GraphModule,
    seen_nodes: Set[Node],
) -> bool:
    """
    Returns true if a pattern ending with `end_node` matches
    the fusion pattern.
    """
    # 当前节点初始化为结束节点
    cur_node = end_node
    
    # 遍历逆转的融合模式列表
    for fusion_idx in range(len(reversed_fusion)):
        # 每个节点只能属于一个匹配模式
        if cur_node in seen_nodes:
            return False
        
        # 获取当前融合元素
        cur_fusion_el = reversed_fusion[fusion_idx]

        # 处理函数调用操作节点
        if cur_node.op == 'call_function':
            # 判断当前融合元素是否为函数
            fusion_el_is_fun = (not isinstance(cur_fusion_el, str)) and \
                (not isinstance(cur_fusion_el, type))
            if fusion_el_is_fun:
                # 检查目标函数是否匹配
                if cur_node.target != cur_fusion_el:
                    return False
                # 如果有参数且参数是节点类型，则将当前节点更新为第一个参数节点
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        # 处理模块调用操作节点
        elif cur_node.op == 'call_module':
            # 判断当前融合元素是否为模块类型
            fusion_el_is_mod = isinstance(cur_fusion_el, type)
            if fusion_el_is_mod:
                assert isinstance(cur_node.target, str)
                # 从图模块中获取目标模块对象
                target_mod = getattr_from_fqn(gm, cur_node.target)
                # 检查目标模块类型是否与当前融合元素匹配
                if not isinstance(target_mod, cur_fusion_el):
                    return False
                # 如果有参数且参数是节点类型，则将当前节点更新为第一个参数节点
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        # 处理方法调用操作节点
        elif cur_node.op == 'call_method':
            # 判断当前融合元素是否为方法名字符串或者带有两个元素的元组
            fusion_el_is_meth_with_second_arg = \
                isinstance(cur_fusion_el, tuple) and len(cur_fusion_el) == 2
            fusion_el_is_meth_without_args = isinstance(cur_fusion_el, str)
            if fusion_el_is_meth_without_args or fusion_el_is_meth_with_second_arg:
                if fusion_el_is_meth_without_args:
                    # 检查目标方法名是否匹配
                    if cur_node.target != cur_fusion_el:
                        return False
                else:
                    assert isinstance(cur_fusion_el, tuple)
                    # 检查目标方法名和第二个参数是否与当前融合元素匹配
                    if cur_node.target != cur_fusion_el[0]:
                        return False
                    elif len(cur_node.args) < 2:
                        return False
                    elif cur_node.args[1] != cur_fusion_el[1]:
                        return False

                # 如果有参数且参数是节点类型，则将当前节点更新为第一个参数节点
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False
        else:
            return False

    # 如果所有融合模式元素均匹配，则返回True
    return True
```