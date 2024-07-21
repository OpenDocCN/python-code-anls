# `.\pytorch\torch\fx\passes\utils\matcher_with_name_node_map_utils.py`

```py
from typing import Dict, List, Tuple  # 导入必要的类型提示

from torch.fx import Graph, GraphModule, Node  # 导入 torch.fx 相关模块

from torch.fx._compatibility import compatibility  # 导入兼容性函数
from .matcher_utils import InternalMatch, SubgraphMatcher  # 导入本地模块中的匹配工具

__all__ = ["SubgraphMatcherWithNameNodeMap"]  # 设置模块的公开接口，只包含 SubgraphMatcherWithNameNodeMap 类


def _split_to_graph_and_name_node_map(
    gm: GraphModule,
) -> Tuple[GraphModule, Dict[str, Node]]:
    # 导入必要的图形信息处理模块
    from torch.fx.graph import _PyTreeInfo
    from torch.utils._pytree import tree_flatten, tree_unflatten

    name_node_map = {}  # 初始化一个空字典用于存储节点名称到节点对象的映射
    for n in gm.graph.nodes:  # 遍历图模块中的节点
        if n.op == "output":  # 如果节点操作为 "output"
            assert gm._out_spec is not None  # 断言输出规范不为空
            output = tree_unflatten(n.args[0], gm._out_spec)  # 根据输出规范解析输出
            assert isinstance(
                output, tuple
            ), "Expecting the pattern graph to return a tuple"  # 断言输出为元组
            assert (
                len(output) >= 2
            ), "Expecting the pattern graph to have at least two outputs"  # 断言输出至少包含两个元素
            *out, name_node_map = output  # 解包输出，获取除最后一个元素外的所有元素，并将最后一个元素赋给 name_node_map
            flattened, out_spec = tree_flatten(out)  # 将非字典部分扁平化
            assert isinstance(
                name_node_map, Dict
            ), "Expecting the input graph to have a dict output as the last element"  # 断言最后一个元素为字典
            n.args = (flattened,)  # 更新节点的参数为扁平化后的结果
            orig_pytree_info = gm._graph._codegen.pytree_info  # 获取原始的 PyTreeInfo 对象
            gm._graph._codegen.pytree_info = _PyTreeInfo(  # 更新 PyTreeInfo 对象
                orig_pytree_info.orig_args, orig_pytree_info.in_spec, out_spec
            )
    gm.recompile()  # 重新编译图模块
    return gm, name_node_map  # 返回更新后的图模块和节点名称映射字典


@compatibility(is_backward_compatible=False)
class SubgraphMatcherWithNameNodeMap(SubgraphMatcher):
    """扩展 SubgraphMatcher 类，支持通过节点名称查询匹配的子图节点，
    这要求模式具有特定的格式（在输出中返回额外的字典，
    字典的键为节点名称，值为模式图中的节点对象，详见示例）。

    与 SubgraphMatcher 的区别在于，它在初始化时需要传入 pattern_gm GraphModule，
    因为我们需要修改图形（这需要重新编译 GraphModule）。

    Example::
        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            return relu, {"conv": conv, "relu": relu}

        def target_graph(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu *= 2
            return relu

        pattern_gm = capture_pre_autograd_graph(pattern, example_inputs)
        target_gm = capture_pre_autograd_graph(target_graph, example_inputs)
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        matches = matcher.match(target_gm)
        for match in matches:
            match.name_node_map["conv"].meta["annotation"] = ...

    """

    def __init__(
        self,
        pattern_gm: GraphModule,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
        ignore_literals: bool = False,
        # 初始化方法，接受 pattern_gm 作为输入，并可选地设定匹配输出、匹配占位符、移除重叠匹配和忽略字面值参数

    ):
        super().__init__(
            pattern_gm,
            match_output=match_output,
            match_placeholder=match_placeholder,
            remove_overlapping_matches=remove_overlapping_matches,
            ignore_literals=ignore_literals,
        )
    def __init__(
        self,
        pattern_gm: GraphMatcher,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = False,
        ignore_literals: bool = False,
    ) -> None:
        # 将输入的图形模式对象分割为图形和名称节点映射，并将其存储在实例变量中
        pattern_gm, name_node_map = _split_to_graph_and_name_node_map(pattern_gm)
        self.name_node_map = name_node_map
        # 调用父类的初始化方法，传递相应的参数
        super().__init__(
            pattern_gm.graph,
            match_output,
            match_placeholder,
            remove_overlapping_matches,
            ignore_literals,
        )

    def match(self, graph: Graph) -> List[InternalMatch]:
        """返回的 InternalMatch 对象将会填充 name_node_map，其包含一个从节点名称 (str) 到目标节点的映射，
        例如 {"conv": target_conv_node, "relu": target_relu_node}

        这要求模式图形返回一个额外的输出，即节点名称到节点的映射，例如，而不是:
        ```
        def pattern(...):
            ...
            return relu
        ```py
        我们应该这样做:
        ```
        def pattern(...):
            ...
            return relu, {"conv": conv, "relu": relu}
        ```py 替代
        """
        # 调用父类的 match 方法，获取内部匹配结果列表
        internal_matches = super().match(graph)
        # 遍历每一个内部匹配结果
        for internal_match in internal_matches:
            # 遍历名称节点映射中的每一对键值对
            for k, n in self.name_node_map.items():
                # 将内部匹配结果中的 nodes_map 中对应节点的值赋给 name_node_map 中对应键的值
                internal_match.name_node_map[k] = internal_match.nodes_map[n]
        # 返回填充了 name_node_map 的内部匹配结果列表
        return internal_matches
```