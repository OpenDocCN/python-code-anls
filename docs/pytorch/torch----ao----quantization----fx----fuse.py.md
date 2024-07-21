# `.\pytorch\torch\ao\quantization\fx\fuse.py`

```
# mypy: allow-untyped-defs
# 导入需要的模块和函数
from torch.fx import (
    GraphModule,    # 导入GraphModule类
    Node,           # 导入Node类
    map_arg         # 导入map_arg函数
)
from torch.fx.graph import Graph  # 导入Graph类
from .match_utils import (        # 导入match_utils模块中的函数和类
    _is_match,                    # 导入_is_match函数
    MatchAllNode                  # 导入MatchAllNode类
)
from .pattern_utils import (      # 导入pattern_utils模块中的函数
    _sorted_patterns_dict         # 导入_sorted_patterns_dict函数
)

from ..backend_config import (    # 导入backend_config模块中的类和函数
    BackendConfig,                # 导入BackendConfig类
    get_native_backend_config     # 导入get_native_backend_config函数
)
from ..backend_config.utils import (  # 导入backend_config.utils模块中的函数
    get_fuser_method_mapping,         # 导入get_fuser_method_mapping函数
    get_fusion_pattern_to_root_node_getter,  # 导入get_fusion_pattern_to_root_node_getter函数
    get_fusion_pattern_to_extra_inputs_getter  # 导入get_fusion_pattern_to_extra_inputs_getter函数
)

from .custom_config import FuseCustomConfig  # 导入FuseCustomConfig类

from .fuse_handler import (  # 导入fuse_handler模块中的函数和类
    _get_fusion_pattern_to_fuse_handler_cls,  # 导入_get_fusion_pattern_to_fuse_handler_cls函数
    FuseHandler                              # 导入FuseHandler类
)

from typing import Any, Callable, Dict, List, Tuple, Union  # 导入类型提示模块中的类型
import warnings  # 导入警告模块

from torch.ao.quantization.utils import Pattern, NodePattern  # 导入模式和节点模式类


__all__ = [    # 声明模块中公开的所有符号
    "fuse",     # 公开fuse函数
    # TODO: 在未来应将此标记为私有
    # 目前由于某些原因在test_public_bindings中仍然需要
    "FuseHandler",  # 公开FuseHandler类
]


def fuse(  # 定义fuse函数，用于模型融合
    model: GraphModule,                                         # 输入参数：GraphModule类型的模型
    is_qat: bool,                                               # 输入参数：布尔值，指示是否是量化训练
    fuse_custom_config: Union[FuseCustomConfig, Dict[str, Any], None] = None,  # 输入参数：融合定制配置
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,        # 输入参数：后端配置
) -> GraphModule:  # 返回类型：GraphModule类型的模型
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()  # 如果未提供融合定制配置，则使用默认配置

    if isinstance(fuse_custom_config, dict):
        warnings.warn(
            "传递fuse_custom_config_dict到fuse已被弃用，并将在未来版本中不再支持。请传入FuseCustomConfig。",
            FutureWarning,
            stacklevel=2,
        )
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)  # 将字典类型的配置转换为FuseCustomConfig对象

    if isinstance(backend_config, dict):
        warnings.warn(
            "传递backend_config_dict到prepare已被弃用，并将在未来版本中不再支持。请传入BackendConfig。",
            FutureWarning,
            stacklevel=2,
        )
        backend_config = BackendConfig.from_dict(backend_config)  # 将字典类型的配置转换为BackendConfig对象

    named_modules = dict(model.named_modules())  # 获取模型中所有命名模块的字典形式

    if backend_config is None:
        backend_config = get_native_backend_config()  # 如果未提供后端配置，则使用本地后端配置

    # 获取融合模式到融合处理器类的映射，并按顺序排列
    fusion_pattern_to_fuse_handler_cls = _sorted_patterns_dict(_get_fusion_pattern_to_fuse_handler_cls(backend_config))
    fuser_method_mapping = get_fuser_method_mapping(backend_config)  # 获取融合方法映射
    fusion_pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)  # 获取融合模式到根节点获取器的映射
    fusion_pattern_to_extra_inputs_getter = get_fusion_pattern_to_extra_inputs_getter(backend_config)  # 获取融合模式到额外输入获取器的映射

    # 查找模型中的融合匹配对
    fusion_pairs = _find_matches(
        model, model.graph, fusion_pattern_to_fuse_handler_cls)
    # TODO: 将此修改为图中的原地更改，因为我们不再构造新的GraphModule了
    fused_graph = Graph()  # 创建一个新的图对象用于融合后的模型图
    env: Dict[Any, Any] = {}  # 创建一个空的环境字典，用于存储节点到数据的映射

    def load_arg(a):  # 定义load_arg函数，用于加载参数
        return map_arg(a, lambda node: env[node.name])  # 使用map_arg函数映射节点的名称到环境字典中的数据
    def default_root_node_getter(node_pattern):
        # 循环直到找到最后一个元素是 Node 类型的节点模式
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        # 返回最后一个 Node 类型的节点模式作为根节点
        return node_pattern[-1]

    # 遍历模型图中的每个节点
    for node in model.graph.nodes:
        # 从 fusion_pairs 字典中获取与当前节点名称对应的五元组，如果不存在则为 None
        maybe_last_node, pattern, matched_node_pattern, obj, node_to_subpattern = \
            fusion_pairs.get(node.name, (None, None, None, None, None))
        
        # 获取当前节点对应的子模式（如果存在的话）
        if node_to_subpattern is not None:
            node_subpattern = node_to_subpattern.get(node, None)
        else:
            node_subpattern = None
        
        # 如果 maybe_last_node 等于当前节点，则进行以下处理
        if maybe_last_node is node:
            # 确保 obj 不为 None
            assert obj is not None
            # 根据模式获取根节点获取器，如果模式未定义则使用默认的根节点获取器
            root_node_getter = fusion_pattern_to_root_node_getter.get(pattern, default_root_node_getter)
            # 通过根节点获取器获取根节点
            root_node = root_node_getter(matched_node_pattern)  # type: ignore[index]
            # 获取额外输入获取器（如果存在的话）
            extra_inputs_getter = fusion_pattern_to_extra_inputs_getter.get(pattern, None)
            extra_inputs = []
            # 如果额外输入获取器存在，则获取额外输入
            if extra_inputs_getter is not None:
                extra_inputs = extra_inputs_getter(matched_node_pattern)
            # TODO: 添加验证，确保 root_node 是一个模块，并且与配置中的 root_module 类型相同
            # 将融合后的节点 obj 加入环境中，调用其 fuse 方法
            env[node.name] = obj.fuse(
                load_arg, named_modules, fused_graph, root_node, extra_inputs, matched_node_pattern,  # type: ignore[arg-type]
                fuse_custom_config, fuser_method_mapping, is_qat)
        
        # 如果 maybe_last_node 是 None 或者 node_subpattern 是 MatchAllNode，则进行以下处理
        elif maybe_last_node is None or node_subpattern is MatchAllNode:
            # 将当前节点复制到融合后的图中，并加入环境中
            env[node.name] = fused_graph.node_copy(node, load_arg)
        
        # 如果节点在模式中匹配但不是根节点，则在此处移除
        # （这里的注释似乎是对整个块的总结，应该避免，只解释每行代码的作用）
    
    # 将模型重新封装为 GraphModule 类型，并返回
    model = GraphModule(model, fused_graph)
    return model
# 定义一个函数，用于在给定的图模块中查找匹配的子图，并返回匹配信息的字典
def _find_matches(
        root: GraphModule,  # 根节点，图模块的起始点
        graph: Graph,  # 图结构，包含待匹配的节点和边
        pattern_to_fuse_handler_cls: Dict[Pattern, Callable],  # 模式到融合处理器类的映射
) -> Dict[str, Tuple[Node, Pattern, NodePattern, FuseHandler, Dict[Node, Any]]]:
    modules = dict(root.named_modules())  # 将所有命名模块存储在字典中，方便按名称查找模块

    # node name -> (root_node, match_value)
    # 匹配映射表，将节点名称映射到节点、模式、节点模式、融合处理器和节点到子模式的字典元组
    match_map: Dict[
        str, Tuple[Node, Pattern, NodePattern, FuseHandler, Dict[Node, Any]]
    ] = {}

    # a map from node to the matched subpattern
    # 节点到匹配子模式的映射，用于存储每个节点匹配到的子模式
    node_to_subpattern: Dict[Node, Any] = {}

    # TODO: dedup with quantization matching function in match_utils.py
    # 定义一个匹配应用函数，用于处理模式匹配过程
    def apply_match(pattern, node, match, matched_node_pattern, node_to_subpattern):
        if isinstance(pattern, tuple):
            s, *args = pattern
            current_node_pattern: List[Node] = []
            apply_match(s, node, match, current_node_pattern, node_to_subpattern)
            for subpattern, arg in zip(args, node.args):
                apply_match(subpattern, arg, match, current_node_pattern, node_to_subpattern)
            matched_node_pattern.append(tuple(current_node_pattern))
        else:
            # the first pattern matches will take precedence
            # 第一个匹配到的模式将优先处理
            if node.name not in match_map:
                matched_node_pattern.append(node)
                # MatchAllNode here is actually MatchAllInputNode which should not
                # be added to match_map
                # 这里的 MatchAllNode 实际上是 MatchAllInputNode，不应该添加到 match_map 中
                if pattern is not MatchAllNode:
                    node_to_subpattern[node] = pattern
                    root_node, pattern, handler = match
                    match_map[node.name] = (
                        root_node, pattern, matched_node_pattern, handler, node_to_subpattern
                    )

    # 从图的末尾开始遍历每个节点
    for node in reversed(graph.nodes):
        if node.name not in match_map:
            # 对每个模式和对应的融合处理器类进行匹配
            for pattern, fuse_handler_cls in pattern_to_fuse_handler_cls.items():
                matched_node_pattern: List[Node] = []
                # 如果当前节点匹配到指定的模式
                if _is_match(modules, node, pattern):
                    # 应用匹配函数进行处理
                    apply_match(
                        pattern, node, (node, pattern, fuse_handler_cls(node)),
                        matched_node_pattern, node_to_subpattern
                    )
                    break  # 找到匹配后结束循环

    # 返回匹配映射表
    return match_map
```