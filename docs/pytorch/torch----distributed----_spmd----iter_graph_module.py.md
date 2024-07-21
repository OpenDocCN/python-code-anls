# `.\pytorch\torch\distributed\_spmd\iter_graph_module.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类型定义
import copy  # 导入深拷贝函数
import inspect  # 导入用于检查对象的结构和源代码的模块
import logging  # 导入日志记录模块
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type  # 导入类型提示相关的类和函数

import torch.nn as nn  # 导入PyTorch神经网络模块
from torch import fx  # 导入PyTorch的FX模块
from torch.distributed._spmd.graph_utils import (
    clone_subgraph,  # 导入克隆子图函数
    get_output,  # 导入获取输出函数
    is_leaf_subgraph,  # 导入判断是否为叶子子图函数
)
from torch.distributed._spmd.partial_lower import partial_lower  # 导入部分下降函数
from torch.fx.graph import _PyTreeCodeGen, PythonCode  # 导入FX图相关的代码生成器和Python代码类
from torch.fx.node import Argument  # 导入FX节点类Argument
from torch.profiler import record_function  # 导入记录函数性能的模块
from torch.utils import _pytree as pytree  # 导入PyTree相关的实用工具
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten  # 导入PyTree相关的操作函数


logger: logging.Logger = logging.getLogger("IterGraphModule")  # 设置logger对象用于记录日志信息


class IterGraph(fx.Graph):
    """``IterGraph``用于执行跨迭代优化的图形表示。

    ``IterGraph`` 跟踪三个图形：self（原始图形）、setup图和cleanup图。这三个图形应该是 ``fx.Graph`` 的相同副本。

    IterGraph子类化fx.Graph以重写构造优化时将使用的必要API，例如通信融合。IterGraph还提供了原本属于fx.Node的API，所有这些API都将有“node_”前缀。
    例如，``IterGraph.node_prepend`` 等同于 ``fx.Node.prepend``。请注意，所有优化必须使用这些API构建。
    """

    def __init__(
        self,
        orig_graph: fx.Graph,  # 原始图形对象
        setup_graph: fx.Graph,  # 设置图形对象
        cleanup_graph: fx.Graph,  # 清理图形对象
        owning_module: Optional[fx.GraphModule] = None,  # 拥有的模块对象，可选
        tracer_cls: Optional[Type["fx.Tracer"]] = None,  # 追踪器类，可选
        tracer_extras: Optional[Dict[str, Any]] = None,  # 追踪器额外参数，可选
    ):
        # 调用父类的构造函数，初始化基类的实例
        super().__init__(owning_module, tracer_cls, tracer_extras)

        # 复制原始图形并返回输出节点值
        output_vals = self.graph_copy(orig_graph, {}, return_output_node=True)
        # 如果我们执行 ``deepcopy(_codegen)`` 并且输入参数包含形如 Dict[torch.Tensor, Any] 的字典，
        # torch.fx._pytree.treen_flatten_spec 将无法展平该字典，
        # 因为 _input_spec 将保存字典的“keys”（不保存值）中的 torch.Tensor 会被复制。
        self._codegen = copy.deepcopy(orig_graph._codegen)
        assert isinstance(output_vals, tuple)
        output_val, old_output_val = output_vals
        # 调用基类的 output 方法，设置输出值和类型表达式
        super().output(output_val, type_expr=getattr(old_output_val, "type", None))

        # 设置图形的初始化和清理函数
        self.setup_graph = setup_graph
        self.cleanup_graph = cleanup_graph
        # 存储所有图形的元组，包括设置图和清理图以及父类的图形
        self._all_graphs: Tuple[fx.Graph, ...] = (
            self.setup_graph,
            self.cleanup_graph,
            cast(fx.Graph, super()),
        )

        # 初始化节点到设置和清理节点的映射字典
        self._setup_mapping: Dict[fx.Node, fx.Node] = {}
        self._cleanup_mapping: Dict[fx.Node, fx.Node] = {}
        # 设置跨迭代移动的冻结状态和跨迭代块计数
        self._freeze_cross_iter_movement = False
        self._cross_iter_block_count = 0

        # 遍历节点、设置图节点和清理图节点的三元组，建立节点到设置和清理节点的映射
        for node, setup_node, cleanup_node in zip(
            self.nodes, self.setup_graph.nodes, self.cleanup_graph.nodes
        ):
            self._setup_mapping[node] = setup_node
            self._cleanup_mapping[node] = cleanup_node

        # 额外输出数量初始化为 0
        self.num_extra_output = 0

    def _lookup_node(self, node: fx.Node, graph: fx.Graph) -> Optional[fx.Node]:
        # 根据所给图形查找节点在设置图或清理图中的对应节点，返回对应节点或 None
        if graph == self.setup_graph:
            return self._setup_mapping.get(node, None)
        elif graph == self.cleanup_graph:
            return self._cleanup_mapping.get(node, None)
        return node

    def _fx_graph_call(
        self, graph: fx.Graph, func: str, *args: Any, **kwargs: Any
    ) -> Any:
        # 获取有效的 fx.Graph 对象，如果传入的图形是 self，则转换为父类的 fx.Graph
        fx_graph: fx.Graph = graph if graph != self else cast(fx.Graph, super())
        # 调用 fx.Graph 对象的指定方法并返回结果
        return getattr(fx_graph, func)(*args, **kwargs)

    def _insert_context(self, func: str, node: fx.Node):
        # 定义内部类 _InsertPoint，用于管理插入点的上下文管理器
        class _InsertPoint:
            def __init__(self, insert_points: List[Any]):
                self.insert_points = insert_points

            def __enter__(self):
                pass

            def __exit__(self, type, value, tb):
                # 在退出上下文时，依次调用每个插入点的 __exit__ 方法
                for insert_point in self.insert_points:
                    insert_point.__exit__(type, value, tb)

        insert_points = []
        # 遍历所有图形，根据节点获取实际的设置或清理节点，并将其添加到插入点列表中
        for graph in self._all_graphs:
            if node:
                actual_node = self._lookup_node(node, graph)
                assert actual_node is not None, "Cannot handle None case now."
            else:
                actual_node = node
            insert_points.append(getattr(graph, func)(actual_node))

        # 返回 _InsertPoint 实例，该实例包含了所有插入点的上下文管理功能
        return _InsertPoint(insert_points)
    # 如果设置了冻结交叉迭代移动标志，调用父类方法插入节点后
    def inserting_after(self, node):
        if self._freeze_cross_iter_movement:
            return super().inserting_after(node)
        # 否则调用内部方法插入节点后，生成上下文
        return self._insert_context("inserting_after", node)

    # 如果设置了冻结交叉迭代移动标志，调用父类方法插入节点前
    def inserting_before(self, node):
        if self._freeze_cross_iter_movement:
            return super().inserting_before(node)
        # 否则调用内部方法插入节点前，生成上下文
        return self._insert_context("inserting_before", node)

    # 向子图中的节点列表传递正向输入，可能会在图中擦除节点
    def _forward_subgraph_inputs(
        self, subgraph: List[fx.Node], graph: fx.Graph, erase_node: bool
    ):
        pass  # 这个函数仅声明，实际实现需要根据具体逻辑填充

    # 向子图中的节点列表传递正向输入，可能会在图中增加额外的输入
    def _forward_inputs_to_subgraph(
        self, subgraph: List[fx.Node], graph: fx.Graph, extra_input: int
    ):
        pass  # 这个函数仅声明，实际实现需要根据具体逻辑填充
    ) -> None:
        """Create extra input nodes and forward the input nodes to the ``subgraph``.

        The external input nodes of ``subgraph`` (nodes that are not in ``subgraph``) will replaced by the newly
        created input nodes.
        """
        # 获取图中所有标记为"placeholder"的节点作为占位符列表
        placeholders = [node for node in graph.nodes if str(node.op) == "placeholder"]
        # 断言确保至少存在一个占位符节点
        assert placeholders, "No placeholders are found"
        # 在最后一个占位符节点之后插入额外的输入节点
        with self._fx_graph_call(graph, "inserting_after", placeholders[-1]):
            # 创建并反转顺序以便正确插入到图中的新输入节点列表
            new_input_nodes = list(
                reversed(
                    [
                        # 为每个新输入节点创建一个名称，名称中包含迭代块计数和索引
                        self._fx_graph_call(
                            graph,
                            "placeholder",
                            f"cross_iter_input_{self._cross_iter_block_count}_{i}",
                        )
                        for i in reversed(range(extra_input))
                    ]
                )
            )

        # 更新子图的输入节点，使用新创建的输入节点
        all_nodes = set(subgraph)
        new_input_index = 0
        for node in subgraph:
            node_inputs, spec = tree_flatten((node.args, node.kwargs))
            new_node_inputs = []
            for input_node in node_inputs:
                # 如果输入节点不是fx.Node类型或者在所有节点集合中，则保持不变
                if not isinstance(input_node, fx.Node) or input_node in all_nodes:
                    new_node_inputs.append(input_node)
                else:
                    # 否则，将其替换为新创建的输入节点
                    new_node_inputs.append(new_input_nodes[new_input_index])
                    new_input_index += 1
            # 更新节点的参数和关键字参数
            node.args, node.kwargs = tree_unflatten(new_node_inputs, spec)
        # 断言确保所有新创建的输入节点都已被使用
        assert new_input_index == len(
            new_input_nodes
        ), f"More inputs than needed {len(new_input_nodes)} > {new_input_index}"

        # 如果图的代码生成器是_PyTreeCodeGen类型并且in_spec不为None时，更新其in_spec
        if (
            isinstance(graph._codegen, _PyTreeCodeGen)
            and graph._codegen.pytree_info.in_spec is not None
        ):
            codegen = graph._codegen
            original_tree_in = tree_unflatten(placeholders, codegen.pytree_info.in_spec)
            # 将新创建的输入节点添加到原始输入节点列表中，并重新展平以获取更新后的in_spec
            _, in_spec = tree_flatten(tuple(list(original_tree_in) + new_input_nodes))
            codegen.pytree_info = codegen.pytree_info._replace(in_spec=in_spec)
            # 将新输入节点的名称添加到原始参数列表中
            for new_input in new_input_nodes:
                codegen.pytree_info.orig_args.append(new_input.name)
            codegen.pytree_info = codegen.pytree_info._replace(in_spec=in_spec)

    def move_to_next_iter_before(
        self, subgraph: List[fx.Node], target_node: fx.Node
    # 将给定的节点列表 nodes 移动到目标节点 target_node 之前
    def move_before(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
        # 遍历所有图形对象
        for graph in self._all_graphs:
            # 将节点列表中的每个节点通过 _lookup_node 方法转换为当前图形中的实际节点列表 actual_nodes
            actual_nodes = [self._lookup_node(node, graph) for node in nodes]
            # 将目标节点 target_node 通过 _lookup_node 方法转换为当前图形中的实际目标节点 actual_target_node
            actual_target_node = self._lookup_node(target_node, graph)
            # 确保实际目标节点不为空
            assert actual_target_node is not None
            # 将每个实际节点 actual_node 插入到实际目标节点 actual_target_node 前面
            for actual_node in actual_nodes:
                actual_target_node.prepend(actual_node)

    # 将给定的节点列表 nodes 移动到目标节点 target_node 之后
    def move_after(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
        # 遍历所有图形对象
        for graph in self._all_graphs:
            # 将节点列表中的每个节点通过 _lookup_node 方法转换为当前图形中的实际节点列表 actual_nodes
            actual_nodes = [self._lookup_node(node, graph) for node in nodes]
            # 将目标节点 target_node 通过 _lookup_node 方法转换为当前图形中的实际目标节点 actual_target_node
            actual_target_node = self._lookup_node(target_node, graph)
            # 将每个实际节点 actual_node 插入到实际目标节点 actual_target_node 后面
            for actual_node in actual_nodes:
                # 确保实际目标节点不为空
                assert actual_target_node is not None
                actual_target_node.append(actual_node)
                # 将当前节点 actual_node 设为新的实际目标节点，以便下一个节点插入到其后面
                actual_target_node = actual_node

    # 调用给定的函数 the_function，返回表示调用结果的节点
    def call_function(
        self,
        the_function: Callable[..., Any],
        args: Optional[Tuple[Argument, ...]] = None,
        kwargs: Optional[Dict[str, Argument]] = None,
        type_expr: Optional[Any] = None,
    ) -> fx.Node:
        # 如果设置了冻结交叉迭代移动，则调用父类的 call_function 方法
        if self._freeze_cross_iter_movement:
            return super().call_function(the_function, args, kwargs, type_expr)

        # 使用 tree_map 方法将 args 中的每个节点或参数转换为设置阶段图形 self.setup_graph 中的节点或参数
        setup_args = tree_map(
            lambda arg: self._lookup_node(arg, self.setup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            args,
        )
        # 使用 tree_map 方法将 kwargs 中的每个节点或参数转换为设置阶段图形 self.setup_graph 中的节点或参数
        setup_kwargs = tree_map(
            lambda arg: self._lookup_node(arg, self.setup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            kwargs,
        )
        # 使用 tree_map 方法将 args 中的每个节点或参数转换为清理阶段图形 self.cleanup_graph 中的节点或参数
        cleanup_args = tree_map(
            lambda arg: self._lookup_node(arg, self.cleanup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            args,
        )
        # 使用 tree_map 方法将 kwargs 中的每个节点或参数转换为清理阶段图形 self.cleanup_graph 中的节点或参数
        cleanup_kwargs = tree_map(
            lambda arg: self._lookup_node(arg, self.cleanup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            kwargs,
        )

        # 在设置阶段图形 self.setup_graph 中调用给定函数 the_function，并返回表示调用结果的节点 setup_node
        setup_node = self.setup_graph.call_function(
            the_function, setup_args, setup_kwargs, type_expr
        )
        # 在主图形中调用给定函数 the_function，并返回表示调用结果的节点 main_node
        main_node = super().call_function(the_function, args, kwargs, type_expr)
        # 在清理阶段图形 self.cleanup_graph 中调用给定函数 the_function，并返回表示调用结果的节点 cleanup_node
        cleanup_node = self.cleanup_graph.call_function(
            the_function, cleanup_args, cleanup_kwargs, type_expr
        )
        # 将 main_node 与其设置和清理阶段节点的映射关系存储在 _setup_mapping 和 _cleanup_mapping 中
        self._setup_mapping[main_node] = setup_node
        self._cleanup_mapping[main_node] = cleanup_node
        # 返回主节点 main_node
        return main_node

    # 从图形中删除给定的节点 to_erase
    def erase_node(self, to_erase: fx.Node) -> None:
        # 如果设置了冻结交叉迭代移动，则调用父类的 erase_node 方法
        if self._freeze_cross_iter_movement:
            return super().erase_node(to_erase)

        # 在设置阶段图形 self.setup_graph 中查找给定节点 to_erase 对应的节点 setup_node
        setup_node = self._lookup_node(to_erase, self.setup_graph)
        # 确保找到了设置阶段的节点 setup_node
        assert setup_node is not None, "setup_node is None"
        # 从设置阶段图形 self.setup_graph 中删除节点 setup_node
        self.setup_graph.erase_node(setup_node)
        # 在主图形中删除给定节点 to_erase
        super().erase_node(to_erase)
        # 在清理阶段图形 self.cleanup_graph 中查找给定节点 to_erase 对应的节点 cleanup_node
        cleanup_node = self._lookup_node(to_erase, self.cleanup_graph)
        # 从清理阶段图形 self.cleanup_graph 中删除节点 cleanup_node
        self.cleanup_graph.erase_node(cleanup_node)
    def placeholder(
        self,
        name: str,
        type_expr: Optional[Any] = None,
        default_value: Any = inspect.Signature.empty,
    ) -> fx.Node:
        # 如果冻结交叉迭代移动，则调用父类方法创建占位符
        if self._freeze_cross_iter_movement:
            return super().placeholder(name, type_expr, default_value)

        # 在主图中创建占位符节点
        main_placeholder = super().placeholder(name, type_expr, default_value)
        # 在设置图中创建占位符节点
        setup_placeholder = self.setup_graph.placeholder(name, type_expr, default_value)
        # 在清理图中创建占位符节点
        cleanup_placeholder = self.cleanup_graph.placeholder(
            name, type_expr, default_value
        )
        # 将主图中的占位符与设置图、清理图中的对应占位符进行映射
        self._setup_mapping[main_placeholder] = setup_placeholder
        self._cleanup_mapping[main_placeholder] = cleanup_placeholder
        return main_placeholder

    def output(self, result: Argument, type_expr: Optional[Any] = None) -> fx.Node:
        # 如果冻结交叉迭代移动，则调用父类方法创建输出节点
        if self._freeze_cross_iter_movement:
            return super().output(result, type_expr)

        # 在主图中创建输出节点
        main_output = super().output(result, type_expr)
        # 使用 tree_map 函数将结果映射到设置图中的对应节点
        setup_result = tree_map(
            lambda _result: self._lookup_node(_result, self.setup_graph)
            if isinstance(_result, fx.Node)
            else _result,
            result,
        )
        # 使用 tree_map 函数将结果映射到清理图中的对应节点
        cleanup_result = tree_map(
            lambda _result: self._lookup_node(_result, self.cleanup_graph)
            if isinstance(_result, fx.Node)
            else _result,
            result,
        )
        # 在设置图中输出映射后的结果
        self.setup_graph.output(setup_result, type_expr)
        # 在清理图中输出映射后的结果
        self.cleanup_graph.output(cleanup_result, type_expr)

        return main_output

    def lint(self) -> None:
        # 在设置图中进行 lint 操作
        self.setup_graph.lint()
        # 调用父类的 lint 方法
        super().lint()
        # 在清理图中进行 lint 操作
        self.cleanup_graph.lint()

    def node_prepend(self, target_node: fx.Node, node: fx.Node) -> None:
        """Prepend node to target_node."""
        # 如果冻结交叉迭代移动，则将节点插入到目标节点之前
        if self._freeze_cross_iter_movement:
            target_node.prepend(node)
            return

        # 遍历所有图，找到实际节点和目标节点，然后将节点插入到目标节点之前
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, "The node is None"
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None, "The target node is None"
            actual_target_node.prepend(actual_node)

    def node_append(self, target_node: fx.Node, node: fx.Node) -> None:
        """Append node to target_node."""
        # 如果冻结交叉迭代移动，则将节点追加到目标节点之后
        if self._freeze_cross_iter_movement:
            target_node.append(node)
            return

        # 遍历所有图，找到实际节点和目标节点，然后将节点追加到目标节点之后
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, f"The actual node is None, {node}."
            actual_target_node = self._lookup_node(target_node, graph)
            assert (
                actual_target_node is not None
            ), f"The actual target node is None, {target_node}."
            actual_target_node.append(actual_node)
    # 将给定节点设置为指定参数列表，如果设置了跨迭代移动冻结，则直接设置并返回
    def node_set_args(self, node: fx.Node, args: Tuple[Argument, ...]) -> None:
        if self._freeze_cross_iter_movement:
            node.args = args
            return

        # 在设置图中查找每个参数的对应节点，并构建与之对应的参数列表
        setup_args = tree_map_only(
            fx.Node, lambda _arg: self._lookup_node(_arg, self.setup_graph), args
        )
        # 查找并设置节点在设置图中的对应节点，并断言不为空
        setup_node = self._lookup_node(node, self.setup_graph)
        assert setup_node is not None
        setup_node.args = setup_args

        # 在清理图中查找每个参数的对应节点，并构建与之对应的参数列表
        cleanup_args = tree_map_only(
            fx.Node, lambda _arg: self._lookup_node(_arg, self.cleanup_graph), args
        )
        # 查找并设置节点在清理图中的对应节点，并断言不为空
        cleanup_node = self._lookup_node(node, self.cleanup_graph)
        assert cleanup_node is not None
        cleanup_node.args = cleanup_args

        # 最终设置节点本身的参数列表
        node.args = args

    # 将给定节点设置为指定关键字参数字典，如果设置了跨迭代移动冻结，则直接设置并返回
    def node_set_kwargs(self, node: fx.Node, kwargs: Dict[str, Argument]) -> None:
        if self._freeze_cross_iter_movement:
            node.kwargs = kwargs
            return

        # 在设置图中查找每个关键字参数的对应节点，并构建与之对应的关键字参数字典
        setup_kwargs = tree_map_only(
            fx.Node, lambda _arg: self._lookup_node(_arg, self.setup_graph), kwargs
        )
        # 查找并设置节点在设置图中的对应节点，并断言不为空
        setup_node = self._lookup_node(node, self.setup_graph)
        assert setup_node is not None
        setup_node.kwargs = setup_kwargs

        # 在清理图中查找每个关键字参数的对应节点，并构建与之对应的关键字参数字典
        cleanup_kwargs = tree_map_only(
            fx.Node, lambda _arg: self._lookup_node(_arg, self.cleanup_graph), kwargs
        )
        # 查找并设置节点在清理图中的对应节点，并断言不为空
        cleanup_node = self._lookup_node(node, self.cleanup_graph)
        assert cleanup_node is not None
        cleanup_node.kwargs = cleanup_kwargs

        # 最终设置节点本身的关键字参数字典
        node.kwargs = kwargs

    # 将节点在所有图中的使用替换为另一个节点，并可选择是否传播元信息
    def node_replace_all_uses_with(
        self,
        node: fx.Node,
        replace_with: fx.Node,
        delete_user_cb: Callable[[fx.Node], bool] = lambda user: True,
        *,
        propagate_meta=False,
    ) -> List[fx.Node]:
        # 遍历所有图，查找并替换节点以及其使用的节点
        for graph in self._all_graphs:
            # 在当前图中查找给定节点和替换节点的实际节点
            actual_node = self._lookup_node(node, graph)
            actual_replace_with = self._lookup_node(replace_with, graph)
            assert actual_node is not None
            # 替换给定节点在当前图中的所有使用节点，并返回结果
            ret = actual_node.replace_all_uses_with(
                actual_replace_with,
                delete_user_cb,
                propagate_meta=propagate_meta,
            )
        return ret  # type: ignore[possibly-undefined]

    # 为给定节点添加一个用户节点或对象到其用户字典中
    def node_add_user(self, node: fx.Node, user: Any) -> None:
        # 遍历所有图，为给定节点查找实际节点，并添加用户节点或对象到其用户字典中
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            if isinstance(user, fx.Node):
                actual_user_node = self._lookup_node(user, graph)
            else:
                actual_user_node = user
            assert actual_node is not None
            actual_node.users[actual_user_node] = None  # type: ignore[index]
    # 从所有图中删除给定节点的指定用户节点。
    def node_remove_user(self, node: fx.Node, user: Any) -> None:
        for graph in self._all_graphs:
            # 查找在当前图中的实际节点
            actual_node = self._lookup_node(node, graph)
            # 如果用户节点是 fx.Node 类型，则查找在当前图中的实际用户节点
            if isinstance(user, fx.Node):
                actual_user_node = self._lookup_node(user, graph)
            else:
                actual_user_node = user
            # 断言节点确实存在
            assert actual_node is not None
            # 删除实际节点的指定用户节点
            del actual_node.users[actual_user_node]  # type: ignore[arg-type]

    # 将没有使用者且操作类型不是 "output" 的节点加入 "__hold__" 用户
    def keep_unused_nodes(self) -> None:
        for node in self.nodes:
            if len(node.users) == 0 and str(node.op) != "output":
                self.node_add_user(node, "__hold__")

    # 为优化器功能化操作添加节点用户关系
    def functionalize_optim(self) -> None:
        # IterGraph 只能支持完整图（前向传播 + 反向传播 + 优化器）。由于优化器不是函数调用（是原地操作），
        # 此方法为优化器调用添加节点用户关系。这个方法对优化器有强烈的假设，并不一定总是有效。这个方法只是一个临时解决方案。

        # TODO: 在 DCE（Dead Code Elimination，死代码消除）移除后删除这个 API
        for node in reversed(self.nodes):
            if node.name.startswith("output"):
                output_node = node
            elif node.name.startswith(
                "_fused_adam_",
            ):
                optim_node = node
            elif node.name.startswith(
                "_foreach_add_",
            ):
                step_node = node
                # 为优化器节点添加输出节点作为用户
                self.node_add_user(optim_node, output_node)  # type: ignore[possibly-undefined]
                # 为步骤节点添加优化器节点作为用户
                self.node_add_user(step_node, optim_node)  # type: ignore[possibly-undefined]

    # 取消优化器功能化操作，移除节点用户关系
    def defunctionalize_optim(self) -> None:
        # TODO: 在 IterGraph 不再使用 DCE 后删除这个 API
        for graph in self._all_graphs:
            for node in reversed(graph.nodes):
                if node.name.startswith("output"):
                    output_node = node
                elif node.name.startswith(
                    "_fused_adam_",
                ):
                    optim_node = node
                elif node.name.startswith(
                    "_foreach_add_",
                ):
                    step_node = node
                    # 从优化器节点的用户中移除输出节点
                    optim_node.users.pop(output_node, None)  # type: ignore[possibly-undefined]
                    # 从步骤节点的用户中移除优化器节点
                    step_node.users.pop(optim_node, None)  # type: ignore[possibly-undefined]

    # 冻结跨迭代移动操作
    def freeze_cross_iter_movement(self) -> None:
        self._freeze_cross_iter_movement = True
class IterGraphModule(nn.Module):
    """``IterGraphModule`` provides the ability to do cross-iteration optimization.

    Given a ``fx.GraphModule``, main_gm, ``IterGraphModule`` internally
    duplicate it to 3 copies and redirect the ``forward`` request to a different
    ``fx.GraphModule`` based on the iteration count. This allows users to do
    graph optimizations that across iterations (e.g., moving collective wait in
    the backward to the forward of the next iteration).

    Note that users must call the APIs provided by ``IterGraphModule`` or
    ``IterGraph`` to rewrite the graph so that ``IterGraphModule`` can keep the
    data dependency for all 3 graphs.
    """

    def __init__(
        self,
        main_gm: fx.GraphModule,
        max_iters: int = -1,
        enable_inductor: bool = False,
    ) -> None:
        super().__init__()

        def _copy_gm(src: fx.GraphModule, graph: fx.Graph) -> fx.GraphModule:
            gm = fx.GraphModule(src, graph)
            gm.meta = getattr(graph, "meta", {})
            return gm

        # Copy the main graph module and its graph for setup, cleanup, and main processing
        self.setup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.cleanup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.main_gm = _copy_gm(
            main_gm,
            IterGraph(main_gm.graph, self.setup_gm.graph, self.cleanup_gm.graph),
        )

        # Initialize iteration control and state variables
        self._iter = 0
        self._max_iters = max_iters
        self._previous_output: Tuple[Any, ...] = tuple()
        self._num_extra_output = 0
        self._is_frozen = False
        self._enable_inductor = enable_inductor

    def finalize_setup(self) -> None:
        """Set up the internal states and also get the signal from users that what is the maximum iteration count.

        This method must be called before the forward() is called.
        """
        # Freeze the graph for cross-iteration optimization
        if not self._is_frozen:
            self.graph.freeze_cross_iter_movement()
            self._num_extra_output = self.graph.num_extra_output
            # Optionally apply partial lowering based on user configuration
            if self._enable_inductor:
                self.main_gm = partial_lower(self.main_gm)
            self._is_frozen = True

        # Reset iteration count
        self._iter = 0

    def _run(self, gm: fx.GraphModule, last_iter: bool, *args, **kwargs) -> Any:
        """Execute the given GraphModule with additional handling for cross-iteration optimization.

        Args:
        - gm (fx.GraphModule): The GraphModule to execute.
        - last_iter (bool): Indicates if this is the last iteration.
        - *args: Positional arguments to pass to gm.
        - **kwargs: Keyword arguments to pass to gm.

        Returns:
        - output (Any): The output from executing gm.
        """
        # Handle cross-iteration optimization if enabled
        if self._num_extra_output > 0:
            new_args = args + (self._previous_output)
            output = gm(*new_args, **kwargs)
            # Store and use previous outputs for subsequent iterations
            if not last_iter:
                assert len(output) == 2
                self._previous_output = tuple(output[-1])
                assert (
                    len(self._previous_output) > 0
                ), "There should be at least one extra output."
                output = output[0]
        else:
            # No cross-iteration optimization is done. Simply call the GraphModule.
            output = gm(*args, **kwargs)
        return output
    # 前进方法，用于执行下一步迭代操作
    def forward(self, *args: Any, last_iter: bool = False, **kwargs: Any) -> Any:
        # 增加迭代次数计数器
        self._iter += 1
        # 判断是否是最后一次迭代
        last_iter = last_iter or self._iter == self._max_iters
        if last_iter:
            # 若为最后一次迭代，使用清理图
            logger.info("Using the cleanup graph")
            gm = self.cleanup_gm
            profiler_string = "## IterGraphModule: Cleanup Graph ##"
            # 重置迭代计数器
            self._iter = 0
        elif self._iter == 1:
            # 若为第一次迭代，使用设置图
            logger.info("Using the setup graph")
            gm = self.setup_gm
            profiler_string = "## IterGraphModule: Setup Graph ##"
        else:
            # 其他情况使用主要图
            gm = self.main_gm
            if self._iter == 2:
                logger.info("Using the main graph")
                profiler_string = "## IterGraphModule -- Maybe Compiling ##"
            else:
                profiler_string = "## IterGraphModule ##"

        # 使用记录函数，记录当前操作的性能信息
        with record_function(profiler_string):
            return self._run(gm, last_iter, *args, **kwargs)

    @property
    def graph(self) -> IterGraph:
        # 返回主要图的图形属性
        return cast(IterGraph, self.main_gm.graph)

    def recompile(self) -> PythonCode:
        # 重新编译设置图和清理图，并返回主要图的重新编译结果
        self.setup_gm.recompile()
        self.cleanup_gm.recompile()
        return self.main_gm.recompile()

    def freeze_cross_iter_movement(self) -> None:
        # TODO: 移除此 API 一旦不再使用
        # 冻结跨迭代移动的图结构，并更新额外输出的数量
        self.graph.freeze_cross_iter_movement()
        self._num_extra_output = self.graph.num_extra_output

    def print_readable(self, print_output: bool = True) -> str:
        # 打印主要图的可读表示，并可选择是否输出到控制台
        return self.main_gm.print_readable(print_output)

    def print_all_graphs(self) -> None:
        # 打印所有三个 fx.Graph 的信息到日志
        logger.info("Printing the three fx.Graph:")
        logger.info("1. Setup fx.Graph:")
        logger.info("%s", self.setup_gm.graph)
        logger.info("2. Main fx.Graph:")
        logger.info("%s", self.main_gm.graph)
        logger.info("3. Cleanup fx.Graph:")
        logger.info("%s", self.cleanup_gm.graph)

    def print_all_graph_modules(self) -> None:
        # 打印所有三个 fx.GraphModule 的信息到日志
        logger.info("Printing the three fx gm:")
        logger.info("1. Setup fx.GraphModule:")
        logger.info("%s", self.setup_gm.print_readable(False))
        logger.info("2. Main fx.GraphModule:")
        logger.info("%s", self.main_gm.print_readable(False))
        logger.info("3. Cleanup fx.GraphModule:")
        logger.info("%s", self.cleanup_gm.print_readable(False))
```