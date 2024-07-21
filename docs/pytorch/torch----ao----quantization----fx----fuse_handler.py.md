# `.\pytorch\torch\ao\quantization\fx\fuse_handler.py`

```
# mypy: allow-untyped-defs
# 引入 torch 库
import torch
# 从 torch.ao.quantization.backend_config 模块中引入 BackendConfig 类
from torch.ao.quantization.backend_config import BackendConfig
# 从 torch.fx.graph 模块中引入 Node, Graph 类
from torch.fx.graph import Node, Graph
# 从 ..utils 模块中引入 _parent_name, NodePattern, Pattern 函数或类
from ..utils import _parent_name, NodePattern, Pattern
# 从 ..fuser_method_mappings 模块中引入 get_fuser_method_new 函数
from ..fuser_method_mappings import get_fuser_method_new
# 引入 ABC 抽象基类
from abc import ABC, abstractmethod
# 引入 Any, Callable, Dict, List, Union 类型
from typing import Any, Callable, Dict, List, Union
# 从 .custom_config 模块中引入 FuseCustomConfig 类
from .custom_config import FuseCustomConfig
# 从 .match_utils 模块中引入 MatchAllNode 类
from .match_utils import MatchAllNode
# 从 torch.nn.utils.parametrize 模块中引入 type_before_parametrizations 函数
from torch.nn.utils.parametrize import type_before_parametrizations

# __all__ 列表，指定模块中公开的接口
__all__ = [
    "DefaultFuseHandler",
    "FuseHandler",
]

# ----------------------------
# Fusion Pattern Registrations
# ----------------------------

# Base Pattern Handler
# 抽象基类 FuseHandler，用于处理融合模式
class FuseHandler(ABC):
    """ Base handler class for the fusion patterns
    """
    @abstractmethod
    def __init__(self, node: Node):
        pass

    @abstractmethod
    def fuse(self,
             load_arg: Callable,
             named_modules: Dict[str, torch.nn.Module],
             fused_graph: Graph,
             root_node: Node,
             extra_inputs: List[Any],
             matched_node_pattern: NodePattern,
             fuse_custom_config: FuseCustomConfig,
             fuser_method_mapping: Dict[Pattern, Union[torch.nn.Sequential, Callable]],
             is_qat: bool) -> Node:
        pass

# DefaultFuseHandler 类继承自 FuseHandler 抽象基类
class DefaultFuseHandler(FuseHandler):
    # 构造方法，初始化节点参数
    def __init__(
            self,
            node: Node):
        # 调用父类构造方法，传入节点参数
        super().__init__(node)  # type:ignore[safe-super]
    def fuse(self,
             load_arg: Callable,
             named_modules: Dict[str, torch.nn.Module],
             fused_graph: Graph,
             root_node: Node,
             extra_inputs: List[Any],
             matched_node_pattern: NodePattern,
             fuse_custom_config: FuseCustomConfig,
             fuser_method_mapping: Dict[Pattern, Union[torch.nn.Sequential, Callable]],
             is_qat: bool) -> Node:
        # 确保根节点的操作是调用模块，否则抛出异常
        assert root_node.op == "call_module", "Expecting module node to be a call_module Node"
        # 获取根节点对应的命名模块
        root_module = named_modules[str(root_node.target)]

        def get_modules(pattern):
            """ 给定节点模式，提取相应的模块
            例如：输入 (relu_node, (bn_node, conv_node))
                 输出 (relu_module, (bn_module, conv_module))
            """
            if isinstance(pattern, (tuple, list)):
                n, *args = pattern
                modules: List[torch.nn.Module] = []
                modules.append(get_modules(n))
                for a in args:
                    modules.append(get_modules(a))
                return tuple(modules)
            else:
                n = pattern
                if n.op == "call_module":
                    return named_modules[n.target]
                elif n.op == "call_function" and n.target == torch.nn.functional.relu:
                    relu = torch.nn.ReLU()
                    relu.training = root_module.training
                    return relu
                elif n.op == "call_function" or n.op == "call_method":
                    return n.target
                else:
                    return MatchAllNode

        # 根据匹配的节点模式获取对应的模块
        matched_modules = get_modules(matched_node_pattern)

        def get_matched_types(m):
            """ 获取匹配模块的类型 """
            if isinstance(m, tuple):
                return tuple(map(get_matched_types, m))
            if isinstance(m, torch.nn.Module):
                return type_before_parametrizations(m)
            return m

        # 获取匹配模块的类型
        matched_module_types = get_matched_types(matched_modules)
        # 获取模块的父名称和模块名称
        module_parent_name, module_name = _parent_name(root_node.target)
        # 获取融合方法
        fuser_method = get_fuser_method_new(matched_module_types, fuser_method_mapping)
        # 使用融合方法创建融合后的模块
        fused_module = fuser_method(is_qat, *matched_modules)
        # 将融合后的模块设置为命名模块的属性
        setattr(named_modules[module_parent_name], module_name, fused_module)
        extra_args = []
        # 加载额外输入参数
        for input in extra_inputs:
            extra_args.append(load_arg(input))
        # 复制根节点到融合图中，并加载参数
        node = fused_graph.node_copy(root_node, load_arg)
        args = list(node.args)
        args.extend(extra_args)
        node.args = tuple(args)
        # 返回融合后的节点
        return node
def _get_fusion_pattern_to_fuse_handler_cls(
        backend_config: BackendConfig) -> Dict[Pattern, Callable]:
    # 创建一个空字典，用于存储融合模式到处理器类的映射关系
    fusion_pattern_to_fuse_handlers: Dict[Pattern, Callable] = {}

    # 遍历后端配置对象中复杂格式模式到配置项的映射
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        # 检查配置项中的融合方法是否已定义
        if config.fuser_method is not None:
            # 如果有定义，将融合模式与默认融合处理器类关联起来
            fusion_pattern_to_fuse_handlers[pattern] = DefaultFuseHandler

    # 返回融合模式到处理器类的映射字典
    return fusion_pattern_to_fuse_handlers
```