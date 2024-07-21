# `.\pytorch\torch\fx\passes\split_module.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import inspect
from typing import Any, Callable, Dict, List, Optional, Set
from collections import OrderedDict
import logging

import torch
# 导入兼容性模块和类
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

# 指定模块的公开接口
__all__ = ["Partition", "split_module"]
# 获取当前模块的日志记录器
_LOGGER = logging.getLogger(__name__)

@compatibility(is_backward_compatible=True)
class Partition:
    def __init__(self, name: str):
        # 初始化分区对象，设置分区名称和相关属性
        self.name: str = name
        self.submod_name = f"submod_{name}"
        self.node_names: List[str] = []
        self.inputs: Dict[str, None] = {}
        self.outputs: Dict[str, None] = {}
        self.dependencies: Dict[str, None] = {}
        self.dependents: Dict[str, None] = {}
        # 创建一个空的 TorchFX 图对象
        self.graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
        # 环境变量字典，将 Node 对象映射到 Node 对象
        self.environment: Dict[Node, Node] = {}
        # 目标字典，用于存储任意类型的目标对象
        self.targets: Dict[str, Any] = {}

    def __repr__(self) -> str:
        # 返回分区对象的字符串表示，包括分区名称、节点名称列表等信息
        return (
            f"name: {self.name},\n"
            f" nodes: {self.node_names},\n"
            f" inputs: {self.inputs},\n"
            f" outputs: {self.outputs},\n"
            f" partitions depended on: {self.dependencies},\n"
            f" partition dependents: {self.dependents}"
        )


# 创建子图形模块的函数，将主图形模块拆分为多个子图形模块
@compatibility(is_backward_compatible=True)
def split_module(
    m: GraphModule,
    root_m: torch.nn.Module,
    split_callback: Callable[[Node], int],
    qualname_map: Optional[Dict[str, str]] = None,
    keep_original_order: Optional[bool] = False,
    keep_original_node_name: Optional[bool] = False,
):
    """
    创建子图形模块的函数，将主图形模块拆分为多个子图形模块

    Args:
        m (GraphModule): 要拆分的图形模块
        root_m (torch.nn.Module): 根 nn 模块。当前未使用。包含在此处是因为根 nn 模块通常通过
            torch.fx._symbolic_trace.symbolic_trace 进行转换（见下面的示例）
        split_callback (Callable[[Node], int]): 可调用函数，将给定的 Node 实例映射到数值分区标识符。
            split_module 将使用此函数作为策略，决定输出 Module 中的操作出现在哪些分区中。
        qualname_map: Optional[Dict[str, str]]: 可选输出参数，返回模块拆分后新目标名称与原始模块中旧目标名称的映射。
        keep_original_order: Optional[bool]: 是否保持 GraphModule 的原始顺序，或者使用新构建的 GraphModule 的拓扑顺序

    Returns:
        GraphModule: 拆分后的模块对象
    """
    # 定义一个函数 construct_graph，用于构建图形表示
    def construct_graph(
        node: Node,
        base_mod_env: Dict[str, Node],
        base_mod_attrs: Dict[str, torch.fx.graph_module.GraphModule],
        current_mod: torch.fx.graph_module.GraphModule,
        current_module_name: str,
    ):
        # 在当前模块环境中，查找并记录当前模块
        if current_module_name in base_mod_env:
            current_mod = base_mod_env[current_module_name]
        else:
            current_mod = base_mod_attrs[current_module_name]

        # 从当前节点中获取操作符
        getitem = node[0]

        # 通过获取的操作符来划分模块
        return current_mod, getitem
    ):
        # 如果节点操作为 "placeholder"
        if node.op == "placeholder":
            # 默认值为节点的第一个参数，如果参数个数大于零；否则为空签名
            default_value = (
                node.args[0] if len(node.args) > 0 else inspect.Signature.empty
            )
            # 如果需要保留原始节点名称
            if keep_original_node_name:
                # 如果有默认值，则将其作为参数；否则参数为空
                args = () if default_value is inspect.Signature.empty else (default_value,)
                # 在基础模块图中创建一个占位符节点，并将其添加到环境中
                base_mod_env[node.name] = base_mod_graph.create_node('placeholder', node.name, args=args, type_expr=node.type)
            else:
                # 使用基础模块图中的占位符函数创建一个占位符节点，并将其添加到环境中
                base_mod_env[node.name] = base_mod_graph.placeholder(
                    node.target, type_expr=node.type, default_value=default_value
                )
            # 复制节点的元数据到环境中的节点
            base_mod_env[node.name].meta = node.meta.copy()
        # 如果节点操作为 "get_attr"
        elif node.op == "get_attr":
            # 获取属性值并将其添加到环境中的节点
            base_mod_env[node.name] = base_mod_graph.get_attr(node.target)
            # 复制节点的元数据到环境中的节点
            base_mod_env[node.name].meta = node.meta.copy()
            # 获取属性值的实际对象
            attr_val = m
            # 遍历属性路径中的每个部分
            for atom in node.target.split("."):  # type: ignore[union-attr]
                # 如果当前对象没有该属性，则引发 AttributeError 异常
                if not hasattr(attr_val, atom):
                    raise AttributeError(f"Node target {node.target} not found!")
                # 获取当前属性的值作为下一个对象
                attr_val = getattr(attr_val, atom)
            # 在基础模块属性中存储属性路径及其值
            base_mod_attrs[node.target] = attr_val  # type: ignore[index]
        # 返回更新后的环境和属性字典
        return base_mod_env, base_mod_attrs

    import sympy

    # 用于存储分区的字典，键为分区名称，值为分区对象
    partitions: Dict[str, Partition] = {}
    # 用于存储原始节点的字典，键为节点名称，值为节点对象
    orig_nodes: Dict[str, Node] = {}
    # 用于将符号映射到节点的字典，键为 sympy 符号，值为节点对象
    symbol_to_node: Dict[sympy.Symbol, Node] = {}

    def record_cross_partition_use(
        def_node: Node, use_node: Optional[Node]
    ):  # noqa: B950
        # 导入符号形状模块中的自由符号函数
        from torch.fx.experimental.symbolic_shapes import free_symbols

        # 获取定义节点和使用节点所属的分区
        defined = getattr(def_node, "_fx_partition", None)
        used = getattr(use_node, "_fx_partition", None)
        # 如果定义节点和使用节点不属于同一个分区
        if defined != used:
            # 如果定义节点不为 None
            if defined is not None:
                # 获取定义节点所属的分区对象，并更新输出和依赖关系
                def_partition = partitions[defined]
                def_partition.outputs.setdefault(def_node.name)
                # 如果使用节点不为 None
                if used is not None:
                    def_partition.dependents.setdefault(used)

            # 如果使用节点不为 None
            if used is not None:
                # 获取使用节点所属的分区对象，并更新输入和依赖关系
                use_partition = partitions[used]
                use_partition.inputs.setdefault(def_node.name)
                # 如果定义节点的示例值不为 None
                if (def_val := def_node.meta.get("example_value")) is not None:
                    # 遍历定义节点示例值中的自由符号，并更新输入节点
                    for s in sorted(free_symbols(def_val), key=str):
                        use_partition.inputs.setdefault(symbol_to_node[s].name)
                # 更新使用节点的依赖关系
                if defined is not None:
                    use_partition.dependencies.setdefault(defined)

    def instantiate_node_partition_mapping(node):
        # 获取节点的分区名称
        partition_name = str(split_callback(node))

        # 添加节点到分区
        partition = partitions.get(partition_name)
        if partition is None:
            partitions[partition_name] = partition = Partition(partition_name)

        # 将节点名称添加到分区对象的节点名称列表中，并将分区名称赋给节点的 _fx_partition 属性
        partition.node_names.append(node.name)
        node._fx_partition = partition_name

    # 全局状态节点是通过其全局状态效应使所有下游节点"受污染"的节点。
    GLOBAL_STATE_NODES = [
        torch.amp._enter_autocast,
        torch.amp._exit_autocast,
        torch._C._set_grad_enabled
    ]


    # 全局状态节点列表，包括自动混合精度进入、退出以及梯度使能设置函数



    grad_regions: OrderedDict[Node, Set[int]] = OrderedDict()


    # 梯度区域：
    # ------------------------
    # 1. 第一个区域：无操作
    # 2. 后续区域：在开头插入梯度设置函数
    # 用于存储节点到分割回调索引集合的有序字典



    autocast_regions: OrderedDict[Node, Set[int]] = OrderedDict()
    autocast_exits: Dict[Node, Optional[Node]] = {}


    # 自动混合精度区域：
    # ------------------------
    # 1. 第一个区域：仅在末尾插入 _exit 函数
    # 2. 中间区域：在开头插入 _enter 函数，在末尾插入 _exit 函数
    # 3. 最后一个区域：仅在开头插入 _enter 函数
    # 按照自动混合精度实例化的顺序执行上述操作
    # 用于存储节点到分割回调索引集合的有序字典及自动混合精度退出映射字典



    active_grad = None
    active_autocasts = set()


    # 当前激活的梯度和自动混合精度设置变量



    for node in m.graph.nodes:


    # 遍历计算图中的节点



        if node.op in ["placeholder", "get_attr", "output"]:
            if (
                node.op == "placeholder" and
                (val := node.meta.get("example_value")) is not None and
                isinstance(val, torch.SymInt) and
                isinstance(val.node.expr, sympy.Symbol)
            ):
                symbol_to_node[val.node.expr] = node
            continue


        # 处理占位符、属性获取和输出节点
        # 如果节点是占位符且包含示例值，将其符号表达式映射到节点



        instantiate_node_partition_mapping(node)


        # 实例化节点分区映射



        if node.op == "call_function" and node.target in GLOBAL_STATE_NODES:
            if node.target == torch._C._set_grad_enabled:
                assert len(node.args) == 1
                assert isinstance(node.args[0], bool)
                active_grad = node
                grad_regions[active_grad] = set({split_callback(node)})
            elif node.target == torch.amp._enter_autocast:
                # Should all be python constants
                assert all(not isinstance(arg, Node) for arg in node.args)
                active_autocasts.add(node)
                autocast_regions[node] = set({split_callback(node)})
                autocast_exits[node] = None
            elif node.target == torch.amp._exit_autocast:
                assert len(node.args) == 1
                autocast_regions[node.args[0]].add(split_callback(node))
                active_autocasts.remove(node.args[0])
                autocast_exits[node.args[0]] = node


        # 处理调用函数节点，并且目标函数在全局状态节点列表中
        # 分别处理梯度使能设置、自动混合精度进入和退出函数



        if active_grad is not None:
            grad_regions[active_grad].add(split_callback(node))


        # 如果存在激活的梯度设置节点，则将当前节点的分割回调索引添加到梯度区域中



        for a in active_autocasts:
            autocast_regions[a].add(split_callback(node))


        # 对于所有激活的自动混合精度设置，将当前节点的分割回调索引添加到相应区域中



    assert all(v is not None for v in autocast_exits.values()), "autocast must exit"


    # 断言确保所有自动混合精度设置都有退出函数，否则抛出异常



    autocast_regions = {k: sorted(v) for k, v in autocast_regions.items()}
    grad_regions = {k: sorted(v) for k, v in grad_regions.items()}


    # 对自动混合精度区域和梯度区域中的分割回调索引进行排序



    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug("autocast_regions: %s", autocast_regions)
        _LOGGER.debug("grad_regions: %s", grad_regions)


    # 如果日志记录器启用了调试级别，记录自动混合精度区域和梯度区域的内容
    # 检查是否需要断言单调递增，根据 autocast_regions 和 grad_regions 的布尔值确定
    assert_monotonically_increasing = bool(autocast_regions) or bool(grad_regions)

    # 将节点分割为分区
    highest_partition = -1
    for node in m.graph.nodes:
        # 记录原始节点，使用节点名称作为键
        orig_nodes[node.name] = node

        # 如果节点的操作是占位符或者获取属性，则跳过
        if node.op in ["placeholder", "get_attr"]:
            continue
        # 如果节点的操作是输出，则对其参数应用记录跨分区使用的函数，并继续下一次循环
        if node.op == "output":
            torch.fx.graph.map_arg(
                node.args[0], lambda n: record_cross_partition_use(n, None)
            )
            continue

        # 如果需要断言单调递增，则调用分割回调函数获取分区号，并断言分区号单调递增
        if assert_monotonically_increasing:
            pid = split_callback(node)
            assert highest_partition <= pid, \
                ("autocast or set_grad_enabled require monotonically increasing partitions:"
                 f"highest: {highest_partition}, this node's: {pid}")
            highest_partition = pid

        # 对于不在全局状态节点列表中的节点，对其参数和关键字参数应用记录跨分区使用的函数
        if node.target not in GLOBAL_STATE_NODES:
            torch.fx.graph.map_arg(
                node.args, lambda def_node: record_cross_partition_use(def_node, node)
            )
            torch.fx.graph.map_arg(
                node.kwargs, lambda def_node: record_cross_partition_use(def_node, node)
            )  # noqa: B950

    # 按照初始分区顺序创建列表
    original_partition_order = list(partitions.keys())
    
    # 查找没有依赖关系的分区，并添加到根分区列表中
    root_partitions: List[str] = []
    for partition_name, partition in partitions.items():
        if not len(partition.dependencies):
            root_partitions.append(partition_name)

    # 检查分区之间是否存在循环依赖，并创建拓扑排序的分区顺序
    sorted_partitions: List[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].dependents:
            partitions[dependent].dependencies.pop(root_partition)
            if not partitions[dependent].dependencies:
                root_partitions.append(dependent)
    # 如果排序后的分区数量不等于总分区数量，则抛出运行时错误
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError("cycle exists between partitions!")

    # 进入预置代码段
    # 遍历自动类型转换和梯度区域映射列表
    for regions_mapping in [autocast_regions, grad_regions]:
        # 遍历映射中的每个节点及其对应的区域列表
        for node, regions in regions_mapping.items():
            # 确保每个区域列表至少有一个区域
            assert len(regions) > 0
            # 将第一个区域的环境映射到分区字典中的当前节点
            partitions[str(regions[0])].environment[node] = node
            # 对于剩余的区域，为每个区域创建新节点并进行环境映射
            for r in regions[1:]:
                # 获取当前区域对应的分区对象
                partition = partitions[str(r)]
                # 使用当前节点的属性创建新节点
                new_node = partition.graph.create_node(
                    op=node.op,
                    target=node.target,
                    args=tuple(arg for arg in node.args),
                    kwargs={},  # 空的关键字参数字典
                    type_expr=node.type,  # 节点的类型表达式
                )
                # 复制当前节点的元数据到新节点
                new_node.meta = node.meta.copy()  # 是否真的需要复制这个元数据？
                # 将当前节点映射到新节点的环境中
                partition.environment[node] = new_node

    # 向分区的输入中添加占位符
    for partition_name in sorted_partitions:
        # 获取当前分区对象
        partition = partitions[partition_name]
        # 遍历分区的输入节点
        for inp in partition.inputs:
            # 在分区的图中为输入节点创建占位符
            placeholder = partition.graph.placeholder(
                inp,
                type_expr=orig_nodes[inp].type,  # 输入节点的类型表达式
            )
            # 复制输入节点的元数据到占位符
            placeholder.meta = orig_nodes[inp].meta.copy()
            # 将输入节点映射到占位符的环境中
            partition.environment[orig_nodes[inp]] = placeholder

    # 转换节点并收集分区子模块的目标
    # 遍历模型图中的所有节点
    for node in m.graph.nodes:
        # 检查节点是否具有 "_fx_partition" 属性
        if hasattr(node, "_fx_partition"):
            # 获取节点所属的分区
            partition = partitions[node._fx_partition]

            # 将关键字参数和参数中旧的图节点替换为此子模块中新节点的引用
            environment = partition.environment
            gathered_args = torch.fx.graph.map_arg(node.args, lambda n: environment[n])
            gathered_kwargs = torch.fx.graph.map_arg(
                node.kwargs, lambda n: environment[n]
            )

            # 如果节点操作不是 "call_module" 或 "get_attr"
            if node.op not in ["call_module", "get_attr"]:
                target = node.target
            else:
                # 拆分目标路径成为单独的属性
                target_atoms = node.target.split(".")
                target_attr = m
                for atom in target_atoms:
                    # 检查目标属性是否存在
                    if not hasattr(target_attr, atom):
                        raise AttributeError(f"Operator target {node.target} not found!")
                    target_attr = getattr(target_attr, atom)
                # 将目标属性路径作为新的目标名称保存到分区的目标字典中
                target = "_".join(target_atoms)
                partition.targets[target] = target_attr

                # 如果存在映射关系，则填充从新限定名称到旧限定名称的映射
                if qualname_map is not None:
                    qualname = f"{partition.submod_name}.{target}"
                    qualname_map[qualname] = node.target

            # 确保 gathered_args 和 gathered_kwargs 的类型分别为元组和字典
            assert isinstance(gathered_args, tuple)
            assert isinstance(gathered_kwargs, dict)

            # 根据新的分区图创建一个新节点
            name = node.name if keep_original_node_name else None
            new_node = partition.graph.create_node(
                op=node.op,
                target=target,
                args=gathered_args,
                kwargs=gathered_kwargs,
                type_expr=node.type,
                name=name,
            )
            # 复制原始节点的元数据到新节点
            new_node.meta = node.meta.copy()

            # 将新节点添加到分区的环境中
            partition.environment[node] = new_node

    # 退出程序尾声
    for regions_mapping in [autocast_regions]:
        for node in reversed(regions_mapping):
            # 获取当前节点对应的区域列表
            regions = regions_mapping[node]
            assert len(regions) > 0

            # 遍历每个区域，除最后一个外的每个区域
            for r in regions[:-1]:
                partition = partitions[str(r)]
                exit_node = autocast_exits[node]
                assert exit_node is not None, "Missing exit node"

                # 在分区的图中创建一个新节点作为退出节点的替代
                new_node = partition.graph.create_node(
                    op=exit_node.op,
                    target=exit_node.target,
                    args=(partition.environment[node],),
                    kwargs={},
                    type_expr=exit_node.type,
                )
                # 复制退出节点的元数据到新节点
                new_node.meta = exit_node.meta.copy()

    # 原始模块环境字典，将节点名称映射到节点本身
    orig_mod_env: Dict[str, Node] = {}
    # 设置值以构造基础模块
    # 初始化空字典，用于存储基础模块环境中的节点映射
    base_mod_env: Dict[str, Node] = {}
    
    # 创建空的PyTorch FX图形对象，用于存储基础模块的计算图
    base_mod_graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
    
    # 初始化空字典，用于存储基础模块属性的映射
    base_mod_attrs: Dict[str, torch.fx.graph_module.GraphModule] = {}
    
    # 如果不保持原始顺序，对每个节点进行遍历，构建基础模块环境和属性
    if not keep_original_order:
        for node in m.graph.nodes:
            base_mod_env, base_mod_attrs = construct_graph(
                node, base_mod_env, base_mod_attrs
            )
    else:
        # 否则，遍历图形以构建节点名称到节点对象的原始模块环境映射
        for node in m.graph.nodes:
            orig_mod_env[node.name] = node
    
    # 执行以下操作，再次按拓扑顺序迭代分区：
    # 1) 通过设置相应的输出来完成子模块图形
    # 2) 为每个子模块构造GraphModule
    # 3) 通过按拓扑顺序或保持原始顺序指定的方式构造基础图形，发出对这些子模块的调用
    construct_order_partitions = (
        sorted_partitions if not keep_original_order else original_partition_order
    )
    
    # 初始化一个集合，用于记录已经构建的属性节点，避免重复构建
    already_constructed_attr_nodes = set()
    
    # 实际上，我们需要按照原始顺序插入占位符节点，否则图形签名将不正确
    original_order = [node for node in m.graph.nodes if node.op == "placeholder"]
    for partition_name in construct_order_partitions:
        partition = partitions[partition_name]

        # 设置正确的输出值
        output_vals = tuple(
            partition.environment[orig_nodes[name]] for name in partition.outputs
        )

        # 如果没有输出值，则跳过输出节点生成
        num_output_vals = len(output_vals)
        if num_output_vals == 1:
            partition.graph.output(output_vals[0])
        elif num_output_vals > 1:
            partition.graph.output(output_vals)

        if keep_original_order:
            # 首先获取此分区所需的属性节点
            orig_mod_attr_nodes: List[Node] = [
                orig_mod_env[key] for key in partition.inputs if key not in original_order
            ]

            # 根据原始顺序构建图形
            for node in original_order:
                if node in already_constructed_attr_nodes:
                    continue  # 已经将此属性添加到基础图中
                base_mod_env, based_mod_attrs = construct_graph(
                    node, base_mod_env, base_mod_attrs
                )
                already_constructed_attr_nodes.add(node)

            # 为此分区构建 GraphModule
            for node in orig_mod_attr_nodes:  # type: ignore[attr-defined]
                if node in already_constructed_attr_nodes:
                    continue
                base_mod_env, base_mod_attrs = construct_graph(
                    node, base_mod_env, base_mod_attrs
                )
                already_constructed_attr_nodes.add(node)

        # 将此分区的信息存入 base_mod_attrs 中
        base_mod_attrs[partition.submod_name] = torch.fx.graph_module.GraphModule(
            partition.targets, partition.graph
        )  # noqa: B950

        # 在基础图中调用此子模块
        output_val = base_mod_graph.call_module(
            partition.submod_name,
            tuple(base_mod_env[name] for name in partition.inputs),
        )

        num_outputs = len(partition.outputs)
        if num_outputs > 1:
            # 从子模块中解包多个返回值
            output_val_proxy = torch.fx.proxy.Proxy(output_val)
            for i, output_name in enumerate(partition.outputs):
                base_mod_env[output_name] = output_val_proxy[i].node  # type: ignore[index]
        elif num_outputs == 1:
            base_mod_env[next(iter(partition.outputs))] = output_val

    # 遍历主图中的节点
    for node in m.graph.nodes:
        if node.op == "output":
            # 将主图中输出节点的计算结果添加到 base_mod_graph 中
            base_mod_graph.output(
                torch.fx.graph.map_arg(node.args[0], lambda n: base_mod_env[n.name])
            )  # noqa: B950

    # 返回包含所有子模块和主图的 GraphModule
    return torch.fx.graph_module.GraphModule(base_mod_attrs, base_mod_graph)
```