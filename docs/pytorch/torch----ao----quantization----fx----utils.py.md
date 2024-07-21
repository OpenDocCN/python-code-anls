# `.\pytorch\torch\ao\quantization\fx\utils.py`

```
# mypy: allow-untyped-defs
# 引入copy模块，用于对象复制操作
import copy
# 引入torch模块，主要用于神经网络相关操作
import torch
# 引入torch.nn模块，提供神经网络的构建块，如层、损失函数等
import torch.nn as nn
# 引入torch.ao.quantization模块，用于量化相关操作
from torch.ao.quantization import (
    QConfigAny,   # 量化配置的通用类型
    QuantType,    # 量化类型
)
# 引入torch.ao.quantization.backend_config模块，定义了带有约束的数据类型
from torch.ao.quantization.backend_config import (
    DTypeWithConstraints,   # 带有约束的数据类型
)
# 引入torch.ao.quantization.fake_quantize模块，提供伪量化相关的类和函数
from torch.ao.quantization.fake_quantize import (
    FakeQuantizeBase,             # 伪量化基类
    FixedQParamsFakeQuantize,     # 固定量化参数的伪量化类
)
# 引入torch.ao.quantization.observer模块，提供观察者类和函数
from torch.ao.quantization.observer import (
    FixedQParamsObserver,   # 固定量化参数的观察者类
    ObserverBase,           # 观察者基类
)
# 引入torch.ao.quantization.qconfig模块，定义了量化配置相关的函数和类
from torch.ao.quantization.qconfig import (
    float16_static_qconfig,   # 静态float16量化配置
    float16_dynamic_qconfig,  # 动态float16量化配置
    qconfig_equals,           # 判断两个量化配置是否相等的函数
)
# 引入torch.ao.quantization.stubs模块，提供了量化相关的存根类和函数
from torch.ao.quantization.stubs import DeQuantStub   # 反量化存根
# 引入torch.ao.quantization.utils模块，提供了一些用于量化的实用函数
from torch.ao.quantization.utils import (
    _assert_and_get_unique_device,        # 断言并获取唯一设备的函数
    activation_is_statically_quantized,   # 判断激活是否静态量化的函数
)
# 引入torch.ao.quantization.observer模块的_is_activation_post_process函数，用于判断激活是否后处理
from torch.ao.quantization.observer import _is_activation_post_process
# 引入torch.ao.quantization.qconfig_mapping模块，提供了量化配置映射类
from torch.ao.quantization.qconfig_mapping import QConfigMapping

# 引入torch.fx模块，用于功能图表示和变换
from torch.fx import GraphModule, map_arg
# 引入torch.fx.graph模块，定义了功能图相关的类和函数
from torch.fx.graph import (
    Graph,   # 功能图类
    Node,    # 图中节点类
)
# 引入.custom_config模块中的PrepareCustomConfig类，用于自定义配置的准备
from .custom_config import PrepareCustomConfig

# 导入_decomposed模块，用于注册量化分解操作
from ._decomposed import quantized_decomposed_lib  # noqa: F401

# 引入typing模块，提供类型相关的功能
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
# 引入dataclasses模块，用于定义数据类
from dataclasses import dataclass
# 引入collections模块中的namedtuple类，用于创建命名元组
from collections import namedtuple
# 引入operator模块，提供Python内置操作符的函数
import operator
# 引入warnings模块，用于警告相关操作

# TODO: revisit this list. Many helper methods shouldn't be public
# 模块公开的函数和类名列表
__all__ = [
    "all_node_args_except_first",
    "all_node_args_have_no_tensors",
    "assert_and_get_unique_device",
    "collect_producer_nodes",
    "create_getattr_from_value",
    "create_node_from_old_node_preserve_meta",
    "EMPTY_ARG_DICT",
    "get_custom_module_class_keys",
    "get_linear_prepack_op_for_dtype",
    "get_new_attr_name_with_prefix",
    "get_non_observable_arg_indexes_and_types",
    "get_qconv_prepack_op",
    "get_skipped_module_name_and_classes",
    "graph_module_from_producer_nodes",
    "maybe_get_next_module",
    "NodeInfo",
    "node_arg_is_bias",
    "node_arg_is_weight",
    "NON_OBSERVABLE_ARG_DICT",
    "NON_QUANTIZABLE_WEIGHT_OPS",
    "return_arg_list",
    "ObservedGraphModuleAttrs",
]

# 不可量化的权重操作集合
NON_QUANTIZABLE_WEIGHT_OPS = {torch.nn.functional.layer_norm, torch.nn.functional.group_norm, torch.nn.functional.instance_norm}

@dataclass
# 观察到的图模块属性数据类，用于存储各种图模块属性
class ObservedGraphModuleAttrs:
    node_name_to_qconfig: Dict[str, QConfigAny]                     # 节点名称到量化配置的映射
    node_name_to_scope: Dict[str, Tuple[str, type]]                # 节点名称到作用域名称和类型的映射
    prepare_custom_config: PrepareCustomConfig                      # 自定义配置的准备
    equalization_node_name_to_qconfig: Dict[str, Any]               # 均衡节点名称到量化配置的映射
    qconfig_mapping: QConfigMapping                                 # 量化配置映射
    is_qat: bool                                                    # 是否是量化感知训练
    observed_node_names: Set[str]                                   # 观察到的节点名称集合
    is_observed_standalone_module: bool = False                     # 是否是独立模块的观察
    standalone_module_input_quantized_idxs: Optional[List[int]] = None   # 独立模块输入量化索引列表
    standalone_module_output_quantized_idxs: Optional[List[int]] = None  # 独立模块输出量化索引列表

# 判断节点参数是否是权重的函数
def node_arg_is_weight(node: Node, arg: Any) -> bool:
    """Returns if node arg is weight"""
    weight_index = None   # 权重索引置空
    # 检查是否在节点的元数据中存在 "target_dtype_info" 键
    if "target_dtype_info" in node.meta:
        # 获取 "target_dtype_info" 中的 "weight_index" 值，若不存在则为 None
        weight_index = node.meta["target_dtype_info"].get("weight_index", None)
    
    # 如果 weight_index 不为 None，并且其值小于节点参数列表的长度，并且该位置的参数与给定的 arg 相同
    if weight_index is not None and weight_index < len(node.args) and node.args[weight_index] is arg:
        # 返回 True
        return True
    
    # 返回节点的关键字参数中 "weight" 的值是否与 arg 相同
    return node.kwargs.get("weight") is arg
# 检查节点参数是否为偏置
def node_arg_is_bias(node: Node, arg: Any) -> bool:
    """Returns if node arg is bias"""
    bias_index = None  # 初始化偏置索引为None
    if "target_dtype_info" in node.meta:  # 如果节点元数据中包含"target_dtype_info"
        bias_index = node.meta["target_dtype_info"].get("bias_index", None)  # 获取偏置索引
    if bias_index is not None and bias_index < len(node.args) and node.args[bias_index] is arg:
        # 如果偏置索引不为None且小于节点参数长度，并且节点参数中的值等于arg
        return True  # 返回True，表示节点参数是偏置
    return node.kwargs.get("bias") is arg  # 否则检查kwargs中的bias是否等于arg，返回结果

# 获取自定义模块类的键列表
def get_custom_module_class_keys(custom_module_mapping: Dict[QuantType, Dict[Type, Type]]) -> List[Any]:
    r""" Get all the unique custom module keys in the custom config dict
    e.g.
    Input:
    {
        QuantType.STATIC: {
            CustomModule1: ObservedCustomModule
        },
        QuantType.DYNAMIC: {
            CustomModule2: DynamicObservedCustomModule
        },
        QuantType.WEIGHT_ONLY: {
            CustomModule3: WeightOnlyObservedCustomModule
        },
    }

    Output:
    # extract the keys across all inner STATIC, DYNAMIC, and WEIGHT_ONLY dicts
    [CustomModule1, CustomModule2, CustomModule3]
    """
    # 使用set来去重
    float_custom_module_classes : Set[Any] = set()  # 初始化一个空的集合用于存放类的键
    for quant_mode in [QuantType.STATIC, QuantType.DYNAMIC, QuantType.WEIGHT_ONLY]:
        quant_mode_custom_module_config = custom_module_mapping.get(quant_mode, {})  # 获取特定量化模式的自定义模块配置
        quant_mode_custom_module_classes = set(quant_mode_custom_module_config.keys())  # 获取该模式下所有类的键
        float_custom_module_classes |= quant_mode_custom_module_classes  # 将这些键添加到总集合中
    return list(float_custom_module_classes)  # 将集合转换为列表并返回

# 根据dtype获取线性预打包操作
def get_linear_prepack_op_for_dtype(dtype):
    if dtype == torch.float16:
        return torch.ops.quantized.linear_prepack_fp16  # 如果dtype是torch.float16，返回相应的线性预打包操作
    elif dtype == torch.qint8:
        return torch.ops.quantized.linear_prepack  # 如果dtype是torch.qint8，返回相应的线性预打包操作
    else:
        raise Exception("can't get linear prepack op for dtype:", dtype)  # 如果dtype不是上述两种类型，抛出异常

# 根据卷积操作获取量化卷积预打包操作
def get_qconv_prepack_op(conv_op: Callable) -> Callable:
    prepack_ops = {
        torch.nn.functional.conv1d: torch.ops.quantized.conv1d_prepack,
        torch.nn.functional.conv2d: torch.ops.quantized.conv2d_prepack,
        torch.nn.functional.conv3d: torch.ops.quantized.conv3d_prepack,
        torch.nn.functional.conv_transpose1d: torch.ops.quantized.conv_transpose1d_prepack,
        torch.nn.functional.conv_transpose2d: torch.ops.quantized.conv_transpose2d_prepack,
        torch.nn.functional.conv_transpose3d: torch.ops.quantized.conv_transpose3d_prepack,
    }
    prepack_op = prepack_ops.get(conv_op, None)  # 根据卷积操作获取相应的预打包操作
    assert prepack_op, f"Didn't find prepack op for {conv_op}"  # 断言确保找到了预打包操作
    return prepack_op  # 返回预打包操作的函数

# 返回一个函数，该函数可以为给定前缀的模块获取新的属性名称
def get_new_attr_name_with_prefix(prefix: str) -> Callable:
    prefix = prefix.replace(".", "_")  # 将前缀中的点替换为下划线
    # 定义一个函数 get_new_attr_name，接受一个参数 module，类型为 torch.nn.Module
    def get_new_attr_name(module: torch.nn.Module):
        # 定义内部函数 get_attr_name，接受一个参数 i，返回一个以 prefix 开头加上 i 转换成字符串的字符串
        def get_attr_name(i: int):
            return prefix + str(i)
        
        # 初始化 i 为 0
        i = 0
        
        # 调用 get_attr_name 函数，将 i 作为参数，得到一个属性名 attr_name
        attr_name = get_attr_name(i)
        
        # 循环直到 module 没有属性名为 attr_name 为止
        while hasattr(module, attr_name):
            # i 自增 1
            i += 1
            # 再次调用 get_attr_name 函数，将更新后的 i 作为参数，重新获取属性名 attr_name
            attr_name = get_attr_name(i)
        
        # 返回找到的可用属性名 attr_name
        return attr_name
    # 返回定义的函数 get_new_attr_name 本身，作为一个函数对象的返回值
    return get_new_attr_name
def collect_producer_nodes(node: Node) -> Optional[List[Node]]:
    r''' Starting from a target node, trace back until we hit input or
    getattr node. This is used to extract the chain of operators
    starting from getattr to the target node, for example
    def forward(self, x):
      observed = self.observer(self.weight)
      return F.linear(x, observed)
    collect_producer_nodes(observed) will either return a list of nodes that
    produces the observed node or None if we can't extract a self contained
    graph without free variables(inputs of the forward function).
    '''
    nodes = [node]  # 初始化节点列表，起始节点是传入的参数节点
    frontier = [node]  # 初始化边界列表，起始节点也是传入的参数节点
    while frontier:
        node = frontier.pop()  # 从边界列表中弹出一个节点
        all_args = list(node.args) + list(node.kwargs.values())  # 获取当前节点的所有参数（包括位置参数和关键字参数）
        for arg in all_args:
            if not isinstance(arg, Node):
                continue
            if arg.op == 'placeholder':
                # 遇到输入节点，无法折叠图形
                return None  # 返回空，表示无法提取不含自由变量的自包含图形
            nodes.append(arg)  # 将参数节点加入节点列表
            if not (arg.op == 'call_function' and arg.target == getattr):
                frontier.append(arg)  # 如果参数节点不是调用getattr函数的结果节点，则将其加入边界列表
    return nodes  # 返回包含所有生产者节点的列表

def graph_module_from_producer_nodes(
        root: GraphModule, producer_nodes: List[Node]) -> GraphModule:
    r''' Construct a graph module from extracted producer nodes
    from `collect_producer_nodes` function
    Args:
      root: the root module for the original graph
      producer_nodes: a list of nodes we use to construct the graph
    Return:
      A graph module constructed from the producer nodes
    '''
    assert len(producer_nodes) > 0, 'list of producer nodes can not be empty'  # 断言确保生产者节点列表不为空
    # 由于我们是从节点到getattr进行追踪，因此反转节点顺序
    producer_nodes.reverse()
    graph = Graph()  # 创建新图形对象
    env: Dict[Any, Any] = {}  # 创建环境字典，用于映射节点到复制的节点对象

    def load_arg(a):
        return map_arg(a, lambda node: env[node])  # 加载参数的映射函数，通过环境字典获取复制的节点对象

    for producer_node in producer_nodes:
        env[producer_node] = graph.node_copy(producer_node, load_arg)  # 将复制的节点对象添加到环境字典中
    graph.output(load_arg(producer_nodes[-1]))  # 设置图形的输出为最后一个生产者节点的复制节点
    graph_module = GraphModule(root, graph)  # 创建基于根模块和图形的图形模块对象
    return graph_module  # 返回构建好的图形模块对象

# TODO: delete
def assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    return _assert_and_get_unique_device(module)  # 调用内部函数获取模块的唯一设备信息

def create_getattr_from_value(module: torch.nn.Module, graph: Graph, prefix: str, value: Any) -> Node:
    """
    Given a value of any type, creates a getattr node corresponding to the value and
    registers the value as a buffer to the module.
    """
    get_new_attr_name = get_new_attr_name_with_prefix(prefix)  # 根据前缀创建新属性名生成函数
    attr_name = get_new_attr_name(module)  # 根据模块和生成函数获取新属性名
    device = assert_and_get_unique_device(module)  # 获取模块的唯一设备
    new_value = value.clone().detach() if isinstance(value, torch.Tensor) \
        else torch.tensor(value, device=device)  # 根据值类型创建新的张量或张量副本
    module.register_buffer(attr_name, new_value)  # 将新值注册为模块的缓冲区
    # 创建包含值的getattr节点
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node
    # 如果缓存不为空且节点在缓存中，直接返回缓存中的结果
    if cache and node in cache:
        return cache[node]

    result = False  # 将被覆盖的初始值
    # 如果节点不是 Node 类型，则认为其参数全部为原语（没有张量），设置结果为 True
    if not isinstance(node, Node):
        result = True
    # 如果节点操作是 'placeholder'，则说明有张量，设置结果为 False
    elif node.op == 'placeholder':
        result = False
    # 如果节点操作是 'call_module'，并且目标模块是激活后处理模块，则递归检查其第一个参数
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        if _is_activation_post_process(modules[node.target]):
            result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
    # 如果节点操作是 'call_function' 并且目标函数是 operator.getitem，则递归检查其第一个参数
    elif node.op == 'call_function' and node.target is operator.getitem:
        result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
    # 如果节点操作是 'get_attr'，则说明有张量，设置结果为 False
    elif node.op == 'get_attr':
        result = False
    # 如果节点目标是 getattr 并且第二个参数是 ['ndim', 'shape'] 中的一个，则认为没有张量，设置结果为 True
    elif node.target is getattr and node.args[1] in ['ndim', 'shape']:
        # x1 = x0.ndim
        result = True
    # 如果节点操作是 'call_method' 并且目标方法是 'size'，则认为没有张量，设置结果为 True
    elif node.op == 'call_method' and node.target == 'size':
        # x1 = x0.size(0)
        result = True
    else:
        # 初始化 found_one_tensor 变量为 False，用于追踪是否找到张量
        found_one_tensor = False
        # 遍历 node 的参数列表
        for arg in node.args:
            # 如果参数是一个列表
            if isinstance(arg, list):
                # 遍历列表中的每个元素
                for list_el in arg:
                    # 如果列表元素是一个 Node 对象
                    if isinstance(list_el, Node):
                        # 递归调用函数检查该列表元素的所有参数是否都没有张量
                        this_list_el_args_have_no_tensors = \
                            all_node_args_have_no_tensors(list_el, modules, cache)
                        # 更新 found_one_tensor，如果该列表元素包含张量，则 found_one_tensor 变为 True
                        found_one_tensor = found_one_tensor or \
                            (not this_list_el_args_have_no_tensors)
                        # 如果已经找到了张量，则没有继续递归的必要
                        if found_one_tensor:
                            # 结果为 True，因为已经找到了张量
                            result = not found_one_tensor
                            # 如果 cache 不为空，则将结果缓存起来
                            if cache:
                                cache[node] = result
                            return result
            # 如果参数是一个整数，直接跳过
            elif isinstance(arg, int):
                pass
            else:
                # 如果参数是一个 Node 对象
                if isinstance(arg, Node):
                    # 递归调用函数检查该参数的所有参数是否都没有张量
                    this_arg_args_have_no_tensors = all_node_args_have_no_tensors(arg, modules, cache)
                    # 更新 found_one_tensor，如果该参数包含张量，则 found_one_tensor 变为 True
                    found_one_tensor = found_one_tensor or \
                        (not this_arg_args_have_no_tensors)
                    # 如果已经找到了张量，则没有继续递归的必要
                    if found_one_tensor:
                        # 结果为 True，因为已经找到了张量
                        result = not found_one_tensor
                        # 如果 cache 不为空，则将结果缓存起来
                        if cache:
                            cache[node] = result
                        return result
                else:
                    # 如果参数不是列表、整数或 Node 对象，则认为找到了张量
                    found_one_tensor = True
            # 结果为 found_one_tensor 的逻辑非，表示最终的结果
            result = not found_one_tensor
    # 如果 cache 不为空，则将最终结果缓存起来
    if cache:
        cache[node] = result
    return result
# 定义函数，返回除第一个参数外的所有节点参数索引列表
def all_node_args_except_first(node: Node) -> List[int]:
    return list(range(1, len(node.args)))

# 构造一个函数，接受一个节点作为参数，并返回适用于节点参数的参数索引列表
def return_arg_list(arg_indices: List[int]) -> Callable[[Node], List[int]]:
    def arg_indices_func(node: Node) -> List[int]:
        # 返回所有小于节点参数数量的参数索引列表
        return [i for i in arg_indices if i < len(node.args)]
    return arg_indices_func

# 定义一个命名元组，用于存储节点的操作和目标信息
NodeInfo = namedtuple("NodeInfo", "op target")

# 非可观察参数字典，标识了哪些节点索引是非张量的，以便正确传播，避免插入观察器导致错误
NON_OBSERVABLE_ARG_DICT: Dict[NodeInfo, Dict[Union[type, torch.dtype], Callable[[Node], List[int]]]] = {
    # 标识 "call_method" 操作和 "masked_fill" 目标的节点
    NodeInfo("call_method", "masked_fill") : {
        torch.bool: return_arg_list([1]),  # 返回参数索引为 1 的列表，仅适用于布尔类型
        float: return_arg_list([2])       # 返回参数索引为 2 的列表，仅适用于浮点数类型
    },
    # 标识 "call_method" 操作和其他目标的节点，对所有除第一个参数外的参数适用
    NodeInfo("call_method", "permute") : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", "repeat") : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", "reshape") : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", "size") : {
        int: return_arg_list([1])  # 返回参数索引为 1 的列表，仅适用于整数类型
    },
    NodeInfo("call_method", "transpose") : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", torch.transpose) : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", "unsqueeze") : {
        int: return_arg_list([1])  # 返回参数索引为 1 的列表，仅适用于整数类型
    },
    NodeInfo("call_method", "unsqueeze_") : {
        int: return_arg_list([1])  # 返回参数索引为 1 的列表，仅适用于整数类型
    },
    NodeInfo("call_method", torch.unsqueeze) : {
        int: return_arg_list([1])  # 返回参数索引为 1 的列表，仅适用于整数类型
    },
    NodeInfo("call_method", "view") : {
        int: all_node_args_except_first
    },
}

# 空参数字典，用于返回空的参数索引列表函数
EMPTY_ARG_DICT: Dict[Union[type, torch.dtype], Callable[[Node], List[int]]] = {}

def get_non_observable_arg_indexes_and_types(node: Node) -> Dict[Union[type, torch.dtype], Callable[[Node], List[int]]]:
    """
    返回一个字典，其中非浮点张量类型作为键，对应于一个函数，用于检索列表
    （该函数接受节点作为参数）
    """
    info = NodeInfo(node.op, node.target)
    # 获取节点信息对应的非可观察参数字典条目，如果不存在则返回空参数字典
    return NON_OBSERVABLE_ARG_DICT.get(info, EMPTY_ARG_DICT)

def maybe_get_next_module(
    node: Node,
    modules: Dict[str, nn.Module],
    target_module_type: Optional[Type[nn.Module]] = None,
    target_functional_type: Any = None,
) -> Optional[Node]:
    """
    获取符合需求的下一个模块（如果存在）
    
    Args:
        node: 需要查看其用户的节点
        target_module_type: 我们要检查的模块类型
        target_functional_type: 我们要检查的函数类型
    """
    # 遍历节点的用户列表中的每个用户
    for user in node.users.keys():
        # 检查用户操作是否为调用模块，并且目标模块类型不为空，且用户指向的模块是目标模块类型的实例
        if user.op == 'call_module' and target_module_type is not None and \
           isinstance(modules[str(user.target)], target_module_type):
            # 如果条件满足，返回当前用户对象
            return user
        # 如果用户操作为调用函数，并且目标函数类型不为空，且用户调用的函数正是目标函数类型
        elif (user.op == 'call_function' and target_functional_type is not None and
              user.target == target_functional_type):
            # 如果条件满足，返回当前用户对象
            return user

    # 如果没有符合条件的用户，返回None
    return None
# 根据给定的旧节点和量化图，创建一个新节点，并将旧节点的必要元数据复制到新节点中
def create_node_from_old_node_preserve_meta(
    quantized_graph: Graph,
    create_node_args: Tuple[Any, ...],
    old_node: Node,
) -> Node:
    """
    创建一个新节点 `new_node`，并从 `old_node` 中复制必要的元数据到新节点中。
    """
    # 使用给定参数在量化图中创建一个新节点
    new_node = quantized_graph.create_node(*create_node_args)
    # 将旧节点的调用堆栈信息复制给新节点
    new_node.stack_trace = old_node.stack_trace
    return new_node

# 获取跳过的模块名称和类名列表，用于量化配置
def get_skipped_module_name_and_classes(
        prepare_custom_config: PrepareCustomConfig,
        is_standalone_module: bool) -> Tuple[List[str], List[Type[Any]]]:
    """
    返回跳过的模块名称和类列表，以供量化配置使用。
    """
    # 复制非可追踪模块的名称列表和类列表
    skipped_module_names = copy.copy(prepare_custom_config.non_traceable_module_names)
    skipped_module_classes = copy.copy(prepare_custom_config.non_traceable_module_classes)
    if not is_standalone_module:
        # 如果不是独立模块，则将独立模块的名称和类加入到跳过列表中
        skipped_module_names += list(prepare_custom_config.standalone_module_names.keys())
        skipped_module_classes += list(prepare_custom_config.standalone_module_classes.keys())
        # 获取自定义模块类键，并加入到跳过的类列表中
        skipped_module_classes += get_custom_module_class_keys(prepare_custom_config.float_to_observed_mapping)

    return skipped_module_names, skipped_module_classes

# 判断节点是否为自定义模块的 LSTM 流程
def _is_custom_module_lstm(
        node: Node,
        named_modules: Dict[str, torch.nn.Module],
        qconfig: QConfigAny = None,
        # QuantizeHandler, but we cannot include the type here due to circular imports
        qhandler: Optional[Any] = None,
) -> bool:
    """
    判断节点是否属于自定义模块 LSTM 流程。
    """
    # 获取节点对应的模块
    mod = _get_module(node, named_modules)
    if qconfig is not None and qhandler is not None:
        # 断言 qhandler 是 QuantizeHandler 类型的实例
        assert isinstance(qhandler, torch.ao.quantization.fx.quantize_handler.QuantizeHandler)  # type: ignore[attr-defined]
        # 返回模块是否为 LSTM，并且激活是否静态量化，并且 qhandler 是否为自定义模块
        return isinstance(mod, torch.nn.LSTM) and \
            activation_is_statically_quantized(qconfig) and \
            qhandler.is_custom_module()
    else:
        # 返回模块是否为 LSTM
        return isinstance(mod, torch.ao.nn.quantizable.LSTM)

# 判断节点是否为自定义模块的 MultiheadAttention 流程
def _is_custom_module_mha(
        node: Node,
        named_modules: Dict[str, torch.nn.Module],
        qconfig: QConfigAny = None,
        # QuantizeHandler, but we cannot include the type here due to circular imports
        qhandler: Optional[Any] = None,
) -> bool:
    """
    判断节点是否属于自定义模块 MultiheadAttention 流程。
    """
    # 获取节点对应的模块
    mod = _get_module(node, named_modules)
    if qconfig is not None and qhandler is not None:
        # 断言 qhandler 是 QuantizeHandler 类型的实例
        assert isinstance(qhandler, torch.ao.quantization.fx.quantize_handler.QuantizeHandler)  # type: ignore[attr-defined]
        # 返回模块是否为 MultiheadAttention，并且激活是否静态量化，并且 qhandler 是否为自定义模块
        return isinstance(mod, torch.nn.MultiheadAttention) and \
            activation_is_statically_quantized(qconfig) and \
            qhandler.is_custom_module()
    else:
        # 返回模块是否为 MultiheadAttention
        return isinstance(mod, torch.ao.nn.quantizable.MultiheadAttention)

# 获取节点对应的模块，用于后续模块类型判断
def _get_module(node: Node, named_modules: Dict[str, torch.nn.Module]) -> Optional[torch.nn.Module]:
    """
    获取节点对应的模块。
    """
    # 返回节点名称在命名模块字典中对应的模块对象
    # 如果节点是调用模块节点，并且其目标在命名模块中存在，则返回该模块，否则返回 None
    if node.op == "call_module" and str(node.target) in named_modules:
        # 返回命名模块中目标节点对应的模块
        return named_modules[str(node.target)]
    else:
        # 如果条件不满足，返回 None
        return None
# 定义函数 `_insert_dequant_stub`，用于向模型中插入 `DeQuantStub` 并创建调用此 `DeQuantStub` 的节点
def _insert_dequant_stub(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Attach a `DeQuantStub` to the model and create a node that calls this
    `DeQuantStub` on the output of `node`, similar to how observers are inserted.
    """
    # 定义 `DeQuantStub` 的名称前缀
    prefix = "dequant_stub_"
    # 函数 `get_new_attr_name_with_prefix` 用于获取带有指定前缀的新属性名
    get_new_dequant_stub_name = get_new_attr_name_with_prefix(prefix)
    # 通过模型和前缀获取一个新的 `DeQuantStub` 名称
    dequant_stub_name = get_new_dequant_stub_name(model)
    # 创建一个 `DeQuantStub` 对象
    dequant_stub = DeQuantStub()
    # 将 `DeQuantStub` 对象设置为模型的属性
    setattr(model, dequant_stub_name, dequant_stub)
    # 将 `DeQuantStub` 对象添加到 `named_modules` 字典中
    named_modules[dequant_stub_name] = dequant_stub
    # 在节点 `node` 后插入新创建的 `DeQuantStub` 节点，并返回新节点
    with graph.inserting_after(node):
        return graph.call_module(dequant_stub_name, (node,))

# 定义函数 `_insert_dequant_stubs_for_custom_module_lstm_output`，用于在自定义模块 LSTM 的每个内部输出节点后插入 `DeQuantStub`
def _insert_dequant_stubs_for_custom_module_lstm_output(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Insert DeQuantStubs after each internal output node of custom module LSTM.

    Custom module LSTM outputs are nested tuples of the structure (output, (hidden0, hidden1)),
    Since we cannot dequantize a tuple as a whole, we must first break down the tuple into its
    components through `getitem`. This function transforms the graph as follows:

      (1) Split the LSTM node into (output, (hidden0, hidden1))
      (2) Insert a DeQuantStub after each internal node
      (3) Recombine the DeQuantStubs into the same structure as before
      (4) Reroute all consumers of the original LSTM node and its sub-nodes
          (e.g. lstm[0])

    Before:
                   lstm_output
                        |
                        v
                  original_user(s)
    After:
                   lstm_output
                  /           \\
                 /  (getitem)  \\
                /               \\
               v                 v
             output            hidden
               |               /   \\
         (DeQuantStub)        (getitem)
               |             /       \\
               v            v         v
           output_dq     hidden0    hidden1
               |            |         |
               |    (DeQuantStub) (DeQuantStub)
               |            |         |
               |            v         v
               |      hidden0_dq  hidden1_dq
               |            \\       /
               |              (tuple)
               |              \\   /
               |               v  v
               |             hidden_dq
               \\               /
                \\   (tuple)   /
                 v            v
                 lstm_output_dq
                       |
                       v
                original_user(s)

    For step (4), reroute all users of the original LSTM node(s) as follows:
      lstm_output -> lstm_output_dq
      lstm_output[0] -> output_dq
      lstm_output[1] -> hidden_dq
      lstm_output[1][0] -> hidden0_dq
      lstm_output[1][1] -> hidden1_dq
    """
    # 函数用于将 `DeQuantStub` 插入自定义模块 LSTM 的每个内部输出节点之后
    # LSTM 的输出是一个嵌套元组的结构 (output, (hidden0, hidden1))
    # 由于无法整体对元组进行去量化，必须先将元组分解为其组件，通过 `getitem` 方法
    # 此函数通过以下步骤转换图形结构：
    # (1) 将 LSTM 节点分解为 (output, (hidden0, hidden1))
    # (2) 在每个内部节点后插入 `DeQuantStub`
    # (3) 将 `DeQuantStub` 重新组合为与之前相同的结构
    # (4) 重新路由原始 LSTM 节点及其子节点的所有消费者
    # 例如，lstm[0]

    # 函数没有返回值，操作都是通过修改图形进行的
    markdown
    # 注释：
    
    ## (1) Split the LSTM node into (output, (hidden0, hidden1))
    在图中的LSTM节点分解为(output, (hidden0, hidden1))。
    
    ## (2) Insert a DeQuantStub after each internal node
    在每个内部节点后插入一个DeQuantStub。
    
    ## (3) Recombine the DeQuantStubs into the same structure as before
    重新组合DeQuantStubs，使其结构与之前相同。
    
    ## (4) Reroute all consumers of the original LSTM node and its sub-nodes
    重定向原始LSTM节点及其子节点的所有使用者。
def _maybe_get_custom_module_lstm_from_node_arg(
    arg: Node,
    named_modules: Dict[str, torch.nn.Module],
) -> Optional[Node]:
    """
    给定节点的参数，如果参数指向节点通过自定义模块 LSTM 的路径消费，则返回自定义模块 LSTM 节点，否则返回 None。

    用于确定节点是否是自定义模块 LSTM 的消费者，如果是，则跳过为该节点插入输入观察器。这是因为自定义模块 LSTM 生成量化输出，
    因此为自定义模块 LSTM 的消费者插入输入观察器将会不必要地再次量化输出。

    实际上，自定义模块 LSTM 输出一个元组 (output, (hidden0, hidden1))，每个内部节点附有 DeQuantStubs（见 `_insert_dequant_stubs_for_custom_module_lstm_output`）。

    此元组可以通过以下四种方式之一被消费：

      lstm -> getitem -> DeQuantStub -> consumer                       # 消费 lstm[0]
      lstm -> getitem -> getitem -> DeQuantStub -> tuple -> consumer   # 消费 lstm[1]
      lstm -> getitem -> getitem -> DeQuantStub -> consumer            # 消费 lstm[1][0] 或 lstm[1][1]
      lstm -> getitem -> DeQuantStub -> tuple -> consumer              # 消费 lstm

    因此，我们必须根据上述模式进行匹配，而不是简单地检查父节点，以确定该节点是否是自定义模块 LSTM 的消费者。
    """

    def match_dq(a):
        return isinstance(_get_module(a, named_modules), DeQuantStub)

    def match_lstm(a):
        return _is_custom_module_lstm(a, named_modules)

    def match_getitem(a):
        return a.op == "call_function" and a.target == operator.getitem

    def match_tuple(a):
        return a.op == "call_function" and a.target == tuple

    def _match_pattern(match_pattern: List[Callable]) -> Optional[Node]:
        """
        向上遍历图并逐个匹配参数。
        如果匹配成功，返回最后匹配的节点，否则返回 None。
        """
        a = arg
        for i, match in enumerate(match_pattern):
            if not match(a):
                return None
            # 匹配下一个参数，对于元组，参数是列表的元组，例如 ([dq_1, other_node],)
            if i < len(match_pattern) - 1:
                if match == match_tuple:
                    a = a.args[0][0]  # type: ignore[assignment,index]
                else:
                    a = a.args[0]  # type: ignore[assignment]
        return a

    all_match_patterns = [
        [match_dq, match_getitem, match_lstm],
        [match_tuple, match_dq, match_getitem, match_getitem, match_lstm],
        [match_dq, match_getitem, match_getitem, match_lstm],
        [match_tuple, match_dq, match_getitem, match_lstm],
    ]

    for p in all_match_patterns:
        matched_node = _match_pattern(p)
        if matched_node is not None:
            return matched_node
    return None
# 定义一个函数用于搜索图中特定模式，该模式是由连续的`tuple`调用函数节点和`getitem`调用函数节点组成的
def _reroute_tuple_getitem_pattern(graph: Graph):
    """
    Search for patterns where N consecutive `tuple` call_function nodes are followed by
    N consecutive `getitem` call_function nodes that are "reverses" of the `tuple` nodes.
    If we find this pattern, reroute the consumers of the last `getitem` to skip these
    N `tuple` and `getitem` nodes.

    Before:

        a   b     c
        |   \\   /
        \\   tuple
         \\   /
          tuple
            |
        getitem(1)
            |
        getitem(0)
            |
            d

    After:

        b
        |
        d
    """
    
    # 定义内部函数，用于递归地在图中查找符合条件的模式
    def find_patterns(
            node: Node,
            index_stack: List[int],
            current_pattern: List[Node],
            matched_patterns: List[List[Node]],
            seen: Set[Tuple[Node, Tuple[int, ...]]]):
        """
        Traverse the graph recursively to match for the N-tuple - N-getitem patterns,
        starting at the given node.

        We use a stack to keep track of the expected `getitem` indices, since these are
        reversed from the `tuple` indices. In the above example, the stack after
        (b -> tuple -> tuple) will be [0, 1], which will be popped by getitem(1) first
        and then by getitem(0).

        TODO: traverse upwards from the output and handle the case when tuple is not a
        separate node, e.g. graph.call_function(operator.getitem, args=(a, (b, c)))
        """
        # 如果 index_stack 已经为空，但是 current_pattern 不为空，表示找到了一个匹配的模式
        if len(index_stack) == 0 and len(current_pattern) > 0:
            matched_patterns.append(copy.copy(current_pattern))
            current_pattern.clear()

        # 避免重复处理相同节点
        state = (node, tuple(index_stack))
        if state in seen:
            return
        seen.add(state)

        # 遍历该节点的使用者，查找匹配的 tuple 和 getitem 节点
        for user in node.users:
            # 如果使用者是 call_function，且目标是 tuple
            if user.op == "call_function" and user.target == tuple:
                # 遍历 tuple 的参数，找到与当前节点匹配的参数索引
                for i, user_arg in enumerate(user.args[0]):  # type: ignore[arg-type]
                    if user_arg == node:
                        index_stack.append(i)
                        current_pattern.append(user)
                        find_patterns(user, index_stack, current_pattern, matched_patterns, seen)
            # 如果使用者是 call_function，且目标是 getitem
            elif user.op == "call_function" and user.target == operator.getitem:
                # 如果 index_stack 不为空
                if len(index_stack) > 0:
                    # 检查 getitem 的第二个参数是否与 index_stack 的最后一个元素匹配
                    if user.args[1] == index_stack[-1]:
                        index_stack.pop()
                        current_pattern.append(user)
                        find_patterns(user, index_stack, current_pattern, matched_patterns, seen)
        return matched_patterns

    # 初始化存储匹配模式和已经处理过的节点的集合
    matched_patterns: List[List[Node]] = []
    seen: Set[Tuple[Node, Tuple[int, ...]]] = set()  # (node, index_stack)
    
    # 遍历图中的每个节点，查找匹配的模式
    for node in graph.nodes:
        find_patterns(node, [], [], matched_patterns, seen)

    # 对于每个匹配的模式，重定向最后一个 getitem 节点的所有消费者到正确的输入
    # 遍历匹配到的模式列表中的每个模式
    for pattern in matched_patterns:
        # 获取当前模式的第一个元组节点和最后一个 getitem 节点
        first_tuple = pattern[0]
        last_getitem = pattern[-1]
        
        # 断言第一个元组节点是使用 tuple 函数调用生成的
        assert first_tuple.op == "call_function" and first_tuple.target == tuple
        # 断言最后一个 getitem 节点是使用 operator.getitem 函数调用生成的
        assert last_getitem.op == "call_function" and last_getitem.target == operator.getitem
        
        # 获取最后一个 getitem 节点的第二个参数，即索引值
        last_getitem_index = last_getitem.args[1]
        
        # 获取新的输入值，它来自于第一个元组节点的第一个参数中的特定索引位置
        new_input = first_tuple.args[0][last_getitem_index]  # type: ignore[index]
        
        # 遍历使用了最后一个 getitem 节点作为输入的所有用户
        for user in list(last_getitem.users.keys()):
            # 用新的输入值替换当前用户中的最后一个 getitem 节点
            user.replace_input_with(last_getitem, new_input)
def _get_observer_from_activation_post_process(
    activation_post_process: Union[ObserverBase, FakeQuantizeBase],
) -> ObserverBase:
    """
    If `activation_post_process` is an observer, return the observer.
    If `activation_post_process` is a fake quantize, return the internal observer.
    """
    # 如果 activation_post_process 是 ObserverBase 类型，则直接返回
    if isinstance(activation_post_process, ObserverBase):
        return activation_post_process
    else:
        # 否则，确认 activation_post_process 是 FakeQuantizeBase 类型
        assert isinstance(activation_post_process, FakeQuantizeBase)
        # 返回其内部的 activation_post_process 对象
        return activation_post_process.activation_post_process  # type: ignore[return-value]

def _qconfig_satisfies_dtype_config_constraints(
        qconfig: QConfigAny,
        dtype_with_constraints: DTypeWithConstraints,
        is_activation: bool = True) -> bool:
    """
    Return whether `qconfig` satisfies the following constraints from the backend,
    specified through the activation and weight DTypeWithConstraints.

        1. QConfig specified a quantization range that falls within the backend's, if any
        2. QConfig specified a min scale value that is >= the backend's, if any
        3. QConfig specified a FixedQParamsObserver or FixedQParamsFakeQuantize that has
           scale and zero point that match the backend's, if any

    If `is_activation` is True, we check `qconfig.activation`, else we check `qconfig.weight`.
    If `qconfig` or `dtype_with_constraints.dtype` is None, or the dtypes do not match, return True.
    """
    # TODO: log warnings only when the user enabled a debug flag
    
    # 如果 qconfig 或 dtype_with_constraints.dtype 为空，则返回 True
    if qconfig is None or dtype_with_constraints.dtype is None:
        return True

    # 根据 is_activation 的值选择要检查的 activation_post_process
    activation_post_process_ctr = qconfig.activation if is_activation else qconfig.weight
    debug_string = "activation" if is_activation else "weight"
    satisfies_constraints = True
    
    # 如果 activation_post_process_ctr 不为 None，则创建 activation_post_process 对象
    if activation_post_process_ctr is not None:
        activation_post_process = activation_post_process_ctr()
        assert _is_activation_post_process(activation_post_process)
        
        # 如果 activation_post_process 的 dtype 与 dtype_with_constraints 的 dtype 不匹配，直接返回 True
        if activation_post_process.dtype != dtype_with_constraints.dtype:
            return True
        
        # 检查 activation_post_process 是否满足 dtype 的配置约束
        satisfies_constraints = _activation_post_process_satisfies_dtype_config_constraints(
            activation_post_process, dtype_with_constraints, debug_string)
    
    return satisfies_constraints
```