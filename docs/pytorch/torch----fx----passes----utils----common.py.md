# `.\pytorch\torch\fx\passes\utils\common.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型声明
from typing import Dict, Tuple

# 从torch.fx._compatibility模块导入compatibility装饰器
from torch.fx._compatibility import compatibility
# 从torch.fx.graph模块中导入Graph类
from torch.fx.graph import Graph

# 从torch.fx.graph_module模块导入GraphModule类
from torch.fx.graph_module import GraphModule
# 从torch.fx.passes.utils.matcher_utils模块导入SubgraphMatcher类
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
# 从torch.nn模块导入Module类
from torch.nn import Module

# 设置模块的公共接口
__all__ = ["HolderModule", "lift_subgraph_as_module", "compare_graphs"]

# 定义一个不向后兼容的HolderModule类，继承自torch.nn.Module类
@compatibility(is_backward_compatible=False)
class HolderModule(Module):
    """
    HolderModule is used to copy all the attributes from original module to submodules
    that uses the attributes
    """

    # 初始化函数，接收一个字典d作为参数
    def __init__(self, d):
        super().__init__()
        # 遍历字典d中的键值对，将每个值作为子模块添加到当前模块中
        for k, v in d.items():
            self.add_module(k, v)

# 定义一个不向后兼容的函数lift_subgraph_as_module，返回一个GraphModule和一个字典
@compatibility(is_backward_compatible=False)
def lift_subgraph_as_module(
    gm: GraphModule,
    subgraph: Graph,
    comp_name: str = "",
    class_name: str = "GraphModule",
) -> Tuple[GraphModule, Dict[str, str]]:
    """
    Create a GraphModule for subgraph, which copies the necessary attributes from the original parent graph_module.

    Args:
        gm (GraphModule): parent graph module

        subgraph (Graph): a valid subgraph that contains copied nodes from the parent graph

        comp_name (str): name for the new component

        class_name (str): name for the submodule

    """

    # 创建一个HolderModule实例作为子模块
    submodule = HolderModule({})
    # 创建一个空字典orig_to_split_fqn_mapping，用于存储原始节点路径映射关系
    orig_to_split_fqn_mapping: Dict[str, str] = {}
    
    # 遍历子图中的每个节点
    for n in subgraph.nodes:
        # 如果节点的操作类型不是"call_module"或"get_attr"，则跳过
        if n.op not in ("call_module", "get_attr"):
            continue
        
        # 获取节点的目标属性名
        target = n.target
        assert isinstance(target, str)
        # 将目标属性名按"."拆分为多个部分
        target_name_parts = target.split(".")
        curr = submodule
        orig_gm = gm

        # 遍历目标属性名的每个部分，逐级添加HolderModule作为子模块
        for name in target_name_parts[:-1]:
            if not hasattr(curr, name):
                curr.add_module(name, HolderModule({}))

            curr = getattr(curr, name)
            orig_gm = getattr(orig_gm, name)

        # 获取目标属性名的最后一个部分作为叶子节点名
        leaf_node_name = target_name_parts[-1]
        # 获取原始模块中的叶子节点对象
        leaf_node = getattr(orig_gm, leaf_node_name)

        # 将原始节点路径映射关系记录到orig_to_split_fqn_mapping中
        orig_to_split_fqn_mapping[target] = f"{comp_name}.{target}"
        # 使用自定义的__setattr__魔法方法设置当前HolderModule实例的属性
        setattr(curr, leaf_node_name, leaf_node)

    # 返回一个新的GraphModule对象和原始节点路径映射字典
    return GraphModule(submodule, subgraph, class_name), orig_to_split_fqn_mapping

# 定义一个不向后兼容的函数compare_graphs，用于比较两个图是否相同
@compatibility(is_backward_compatible=False)
def compare_graphs(left: Graph, right: Graph) -> bool:
    """
    Return True if two graphs are identical, i.e they
        - have the same number of outputs in the same order
        - have the same number of inputs in the same order
        - have the same set of nodes, and identical connectivity
    """
    # 创建一个 SubgraphMatcher 对象，用于匹配两个图形 left 和 right，同时匹配输出和占位符
    matcher = SubgraphMatcher(left, match_output=True, match_placeholder=True)
    
    # 使用 matcher 对象匹配 right 图形中与 left 匹配的子图形，并返回所有匹配结果
    matches = matcher.match(right)
    
    # 如果匹配结果的数量大于0，则返回 True；否则返回 False
    return len(matches) > 0
```