# `.\pytorch\torch\fx\passes\split_utils.py`

```
# 用于允许未类型化的函数定义
mypy: allow-untyped-defs

# 引入必要的库和模块
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

# 引入 torch.fx 相关模块和函数
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import map_arg
from torch.fx.passes.utils import HolderModule, lift_subgraph_as_module

# 引入本地的 NodeList 工具
from .tools_common import NodeList

# 导出的模块成员列表
__all__ = ["getattr_recursive", "setattr_recursive", "Component", "split_by_tags"]


# 兼容性装饰器，用于支持向后兼容性
@compatibility(is_backward_compatible=False)
# 递归获取属性值的函数
def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


# 兼容性装饰器，用于支持向后兼容性
@compatibility(is_backward_compatible=False)
# 递归设置属性值的函数
def setattr_recursive(obj, attr, value):
    if "." not in attr:
        setattr(obj, attr, value)
    else:
        layer = attr.split(".")
        setattr_recursive(getattr(obj, layer[0]), ".".join(layer[1:]), value)


# 兼容性装饰器，用于支持向后兼容性
@compatibility(is_backward_compatible=False)
# 描述一个组件的数据结构，用于容纳稍后创建的子图
@dataclass
class Component:
    """
    A component serves as a container for a subgraph we want to create afterwards.
    """

    # 子图对象
    graph: torch.fx.Graph
    # 组件的顺序
    order: int
    # 组件的名称
    name: str

    # 存储在 `graph` 中的占位节点
    input_placeholders: List = field(default_factory=list)

    # 原始图中作为占位符的节点
    orig_inputs: List = field(default_factory=list)

    # 原始图中作为输出的节点
    orig_outputs: List = field(default_factory=list)

    # 将原始图中的 get_attr 节点映射到 `graph` 中的 get_attr 节点
    getattr_maps: Dict[torch.fx.Node, torch.fx.Node] = field(default_factory=dict)

    # 构造函数参数列表
    constructor_args: List[str] = field(default_factory=list)

    # torch.fx.GraphModule 对象，可选
    gm: Optional[torch.fx.GraphModule] = None


# 兼容性装饰器，用于支持向后兼容性
@compatibility(is_backward_compatible=False)
# 根据标签将 GraphModule 拆分成子模块的函数
def split_by_tags(
    gm: torch.fx.GraphModule,
    tags: List[str],
    return_fqn_mapping: bool = False,
    return_tuple: bool = False,
    GraphModuleCls: Type[torch.fx.GraphModule] = torch.fx.GraphModule,
) -> Union[torch.fx.GraphModule, Tuple[torch.fx.GraphModule, Dict[str, str]]]:
    """
    Splits a GraphModule using tags on its graph nodes. We honor the order of
    tags. For example, we have tags = ["a", "b", "c"], the function will create
    the initial submodules in the order of "a", "b", "c".

    To set a tag:
    gm.graph.nodes[idx].tag = "mytag"

    This will result in all nodes with the same tag being extracted and placed in their
    own submodule. For placeholder, output and get_attr node, the tag is ignored. placeholder
    and output nodes are created when needed while get_attr nodes get copied to submodules
    where they are used.

    Given the following module def:
    ...
    """
    # 函数的具体实现逻辑在此省略，可以参考文档和实际代码实现
    class SimpleModule(torch.nn.Module):
        # 定义简单的神经网络模块
        def __init__(self):
            super().__init__()
            # 初始化三个线性层
            self.linear1 = torch.nn.Linear(...)
            self.linear2 = torch.nn.Linear(...)
            self.linear3 = torch.nn.Linear(...)

        # 前向传播函数
        def forward(self, in1, in2):
            # 使用第一个线性层处理输入 in1
            r1 = self.linear1(in1)
            # 使用第二个线性层处理输入 in2
            r2 = self.linear2(in2)
            # 将两个处理结果拼接起来
            r3 = torch.cat([r1, r2])
            # 使用第三个线性层处理拼接后的结果并返回
            return self.linear3(r3)

    def flatten(x: torch.fx.node.Argument) -> NodeList:
        """
        将参数 x 中的节点存储到列表中并返回该列表。
        """
        r: NodeList = []
        # 调用 map_arg 函数将 x 中的节点追加到列表 r 中
        map_arg(x, r.append)
        return r

    # 从原始模块节点映射到创建的子模块节点的字典
    node_remapping: Dict[torch.fx.Node, torch.fx.Node] = {}

    # 从原始模块节点或创建的子模块节点映射到对应组件的字典
    node_to_component: Dict[torch.fx.Node, Component] = {}

    # 标签到对应组件的映射字典
    tag_to_component: Dict[str, Component] = {}

    # 所有组件的列表
    all_components: List[Component] = []

    # 将用于主图的节点字典
    used_in_main: Dict[torch.fx.Node, None] = {}

    # 分割后的主图
    main_g = torch.fx.Graph()

    # 从原始模块节点到分割后主图节点的映射字典
    main_remapping: Dict[torch.fx.Node, torch.fx.Node] = {}

    # 原始模块的输出节点
    output_node: Optional[torch.fx.Node] = None

    # 为每个标签创建一个组件，预期不会再创建其他组件
    for tag in tags:
        # 使用空的图和标签创建组件
        comp = Component(torch.fx.Graph(), len(all_components), f"{tag}")
        all_components.append(comp)
        tag_to_component[tag] = comp

    # 遍历原始图中的节点并处理它们。
    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 如果节点的操作为 "output"
        if node.op == "output":
            # 如果已经有输出节点存在，则抛出运行时错误
            if output_node is not None:
                raise RuntimeError("Multiple output nodes in graph!")
            output_node = node  # 将当前节点标记为输出节点
            continue  # 继续处理下一个节点

        # 原始图中的占位符节点被复制到主图中
        if node.op == "placeholder":
            main_remapping[node] = main_g.placeholder(node.name, type_expr=node.type)
            main_remapping[node].meta = copy.copy(node.meta)
            continue

        # get_attr 节点被忽略，因为我们不对它们进行标记
        # 而是直接复制到子模块中后续使用
        if node.op == "get_attr":
            continue

        # 处理可调用节点，即操作为 call_module、call_function 或 call_method 的节点
        assert hasattr(node, "tag")

        # 获取节点参数和关键字参数的所有组件
        upstream_components = [
            node_to_component[x]
            for x in flatten(node.args) + flatten(node.kwargs)
            if x.op not in {"placeholder", "get_attr"}
        ]

        comp = tag_to_component[node.tag]
        node_to_component[node] = comp  # 将节点映射到其对应的组件中

        # 计算上游组件的最大顺序
        mx = max((c.order for c in upstream_components), default=0)

        # 断言当前节点所属的组件顺序要高于其上游组件的最大顺序
        assert comp.order >= mx

        # 映射节点的输入到组件图中的节点
        def remap_func(x):
            # 如果输入是 get_attr 节点，复制到当前组件的图中
            # 返回当前组件图中的 get_attr 节点
            if x.op == "get_attr":
                if x not in comp.getattr_maps:
                    comp.getattr_maps[x] = comp.graph.get_attr(
                        x.target, type_expr=x.type
                    )
                return comp.getattr_maps[x]

            # 如果输入不是占位符，且已经被放入了组件中，则返回对应的组件中的节点
            if x.op != "placeholder" and node_to_component[x] == comp:
                return node_remapping[x]

            # 如果输入是占位符或者在其他组件中，则将其作为当前组件图中的占位符
            if x not in comp.orig_inputs:
                comp.orig_inputs.append(x)
                placeholder = comp.graph.placeholder(x.name, type_expr=x.type)
                placeholder.meta = copy.copy(x.meta)
                comp.input_placeholders.append(placeholder)
                used_in_main[x] = None

            return comp.input_placeholders[comp.orig_inputs.index(x)]

        # 复制节点到组件的图中，并使用 remap_func 进行输入映射
        n = comp.graph.node_copy(node, remap_func)
        n.tag = node.tag  # 设置节点的标记
        node_remapping[node] = n  # 更新节点映射
        node_to_component[n] = comp  # 更新节点到组件的映射
    # 如果输出节点为None，则抛出运行时错误，指示图没有输出节点
    if output_node is None:
        raise RuntimeError("Graph had no output node!")

    # 遍历展开输出节点的第一个参数
    for x in flatten(output_node.args[0]):
        if x.op == "get_attr":
            # 对于类型为"get_attr"且被输出节点使用的节点，不需要组件映射，
            # 只需确保在结果图中创建对应的节点。
            main_remapping[x] = main_g.get_attr(x.name, type_expr=x.type)
        else:
            # 所有被输出节点消费的组件结果应标记为在主图中使用
            used_in_main[x] = None

    # 如果节点在主图中被使用，则将其标记为所属组件的输出
    for n in used_in_main:
        if n.op != "placeholder":
            node_to_component[n].orig_outputs.append(n)

    # 现在为每个组件创建一个图模块
    orig_to_split_fqn_mapping: Dict[str, str] = {}
    for comp in all_components:
        # 获取组件的输出节点的映射
        outs = tuple(map(node_remapping.__getitem__, comp.orig_outputs))

        if return_tuple:
            # 如果返回元组，则输出组件的输出
            comp.graph.output(outs)
        else:
            # 处理FX输出节点的参数。如果只有一个输出，则输出节点参数如(output_single)，
            # 否则，如果有多个输出，则输出节点参数如((output_0, output_1, ...))。
            comp.graph.output(outs[0] if len(outs) == 1 else outs)

        # 将子图提升为模块，并返回组件的图模块和原始到分割全限定名映射
        comp.gm, comp_orig_to_split_fqn_mapping = lift_subgraph_as_module(
            gm, subgraph=comp.graph, comp_name=comp.name
        )
        orig_to_split_fqn_mapping.update(comp_orig_to_split_fqn_mapping)

        # 在主图中创建一个call_module节点
        main_node = main_g.call_module(
            comp.name,
            args=tuple(map(main_remapping.__getitem__, comp.orig_inputs)),
            kwargs=None,
        )

        if len(outs) == 1 and not return_tuple:
            # 如果只有一个输出且不返回元组，则将主图的映射设置为主节点
            main_remapping[comp.orig_outputs[0]] = main_node
        else:
            # 否则，为每个输出设置主图的映射
            for i, o in enumerate(comp.orig_outputs):
                # 使用Proxy记录getitem访问
                main_remapping[o] = torch.fx.Proxy(main_node)[i].node  # type: ignore[index]

    # 将主图中输出节点的参数映射为主映射
    main_g.output(map_arg(output_node.args[0], main_remapping.__getitem__))
    # 创建一个HolderModule，将所有组件的图模块作为参数
    main_root = HolderModule({comp.name: comp.gm for comp in all_components})
    # 将主图的_codegen设置为gm图的_codegen
    main_g._codegen = gm.graph._codegen

    # 如果输出节点直接消费原始图中的get_attr节点，则确保在新图中复制get_attr
    for x in flatten(output_node.args[0]):
        if x.op == "get_attr":
            setattr(main_root, x.name, getattr_recursive(gm, x.target))  # type: ignore[arg-type]

    # 创建GraphModuleCls实例作为结果，将主Root和主图作为参数
    result_gm = GraphModuleCls(main_root, main_g)
    if return_fqn_mapping:
        # 如果需要返回分割全限定名映射，则一并返回结果和映射
        return result_gm, orig_to_split_fqn_mapping

    # 否则，只返回结果
    return result_gm
```