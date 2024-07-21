# `.\pytorch\torch\contrib\_tensorboard_vis.py`

```py
# mypy: allow-untyped-defs
# 导入所需的模块和库
import time  # 导入时间模块
from collections import defaultdict  # 导入 defaultdict 类
from functools import partial  # 导入 partial 函数
from typing import DefaultDict  # 导入 DefaultDict 类型

import torch  # 导入 PyTorch 库


# Unfortunately it doesn't seem as if there was any way to get TensorBoard to do
# anything without having TF installed, and so this file has a hard dependency on it
# as well. It really is a debugging tool, so it doesn't matter.
try:
    # 尝试导入 TensorFlow 相关模块
    from tensorflow.core.util import event_pb2  # 导入 event_pb2 模块
    from tensorflow.core.framework import graph_pb2  # 导入 graph_pb2 模块
    from tensorflow.python.summary.writer.writer import FileWriter  # 导入 FileWriter 类
except ImportError:
    # 如果导入失败，抛出 ImportError 异常
    raise ImportError("TensorBoard visualization of GraphExecutors requires having "
                      "TensorFlow installed") from None


def dump_tensorboard_summary(graph_executor, logdir):
    # 使用 FileWriter 打开 logdir 路径，准备写入 TensorBoard 日志
    with FileWriter(logdir) as w:
        # 可视化图形执行器，得到 pb_graph
        pb_graph = visualize(graph_executor)
        # 创建事件对象 evt，包括当前时间和序列化后的图定义
        evt = event_pb2.Event(wall_time=time.time(), graph_def=pb_graph.SerializeToString())
        # 将事件添加到 FileWriter 中
        w.add_event(evt)


def visualize(graph, name_prefix='', pb_graph=None, executors_it=None):
    """Visualizes an independent graph, or a graph executor."""
    # value_map 用于映射值到名称
    value_map = {}
    # 如果未提供 pb_graph，则创建一个新的 GraphDef 对象
    pb_graph = pb_graph or graph_pb2.GraphDef()

    # 如果 graph 是 torch._C.GraphExecutorState 类型的实例
    if isinstance(graph, torch._C.GraphExecutorState):
        # 可视化图形执行器的状态
        visualize_graph_executor(graph, name_prefix, pb_graph,
                                 partial(visualize, pb_graph=pb_graph))
        # 返回最终的 pb_graph
        return pb_graph

    # 设置输入节点
    input_node = pb_graph.node.add(op='input', name=name_prefix + 'input')
    # 遍历 param_node 的输出，建立 value_map
    for i, value in enumerate(graph.param_node().outputs()):
        value_map[value.unique()] = name_prefix + 'input:' + str(i)

    # 递归地可视化图形
    visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it)

    # 设置输出节点
    return_node = pb_graph.node.add(op='output', name=name_prefix + 'output')
    # 遍历 return_node 的输入，设置输出节点的输入
    for value in graph.return_node().inputs():
        return_node.input.append(value_map[value.unique()])

    # 返回最终的 pb_graph
    return pb_graph


def visualize_graph_executor(state, name_prefix, pb_graph, inline_graph):
    """Append the state of a given GraphExecutor to the graph protobuf.

    Args:
        state (GraphExecutor or GraphExecutorState): GraphExecutor to display.
        name_prefix (str): Name prefix of the containing subgraph.
        pb_graph (GraphDef): graph to append to.
        inline_graph (Callable): a function that handles setting up a value_map,
            so that some graphs in here can be inlined. This is necessary, because
            this will simply be `visualize` for the top-level GraphExecutor,
            or `inline_graph` for all nested ones.

            The signature should look like (Graph, name_prefix) -> ().
            It will be called exactly once.

    The strategy is to embed all different configurations as independent subgraphs,
    while inlining the original graph as the one that actually produces the values.
    """
    # 追加给定 GraphExecutor 的状态到 graph protobuf 中
    pass  # 这里只是占位符，函数体暂时为空，具体实现可以在之后补充
    # 如果存在自动微分回退图，则可视化该图
    if state.autograd_fallback_graph is not None:
        visualize(graph=state.autograd_fallback_graph,
                  name_prefix=name_prefix + 'autograd_fallback/',
                  pb_graph=pb_graph,
                  executors_it=iter(state.autograd_fallback.executors()))

    # 遍历执行计划的每个条目
    for i, (arg_spec, plan) in enumerate(state.execution_plans.items()):
        # 创建一个断开连接的节点，用于保存这个追踪的输入类型信息。
        # 由于这个信息太详细，不适合包含在子图名称中。
        input_kinds = pb_graph.node.add(op='INPUT_KIND', name=subgraph_name)
        input_kinds.attr['inputs'].s = repr(arg_spec).encode('ascii')

        # 可视化执行计划的图形
        visualize(plan.graph, subgraph_name, pb_graph, iter(plan.code.executors()))

        # 如果存在梯度执行器，则将梯度显示为独立子图
        if plan.grad_executor is not None:
            grad_subgraph_name = subgraph_name + 'grad/'
            visualize(plan.grad_executor, grad_subgraph_name, pb_graph)

    # 返回以内联方式显示的原始图
    return inline_graph(state.graph, name_prefix + 'original/')
def visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it=None):
    """Recursive part of visualize (basically skips setting up the input and output nodes)."""
    # 定义内部函数inline_graph，用于递归地可视化子图
    def inline_graph(subgraph, name, node):
        # 创建递归调用所需的数值映射rec_value_map，将子图的输入值映射到当前图的数值映射中
        rec_value_map = {inp.unique(): value_map[val.unique()]
                         for inp, val in zip(subgraph.inputs(), node.inputs())}
        # 递归调用visualize_rec来处理子图
        visualize_rec(graph=subgraph,
                      value_map=rec_value_map,
                      name_prefix=name,
                      pb_graph=pb_graph)
        # 更新当前图的输出节点到数值映射中
        for out, val in zip(subgraph.outputs(), node.outputs()):
            value_map[val.unique()] = rec_value_map[out.unique()]

    # 使用默认字典来记录操作符的计数
    op_id_counter: DefaultDict[str, int] = defaultdict(int)

    # 返回给定节点的名称，其中包含操作类型和名称前缀
    def name_for(node):
        kind = node.kind()[node.kind().index('::') + 2:]
        op_id_counter[kind] += 1
        return kind, name_prefix + kind + '_' + str(op_id_counter[kind])

    # 添加融合组节点到图中
    def add_fusion_group(node):
        op, name = name_for(node)
        # 调用inline_graph函数来处理融合组的子图
        inline_graph(node.g('Subgraph'), name + '/', node)

    # 添加图执行器节点到图中
    def add_graph_executor(node):
        op, name = name_for(node)
        if executors_it is None:
            add_node(node)
        else:
            # 从executors_it中获取下一个图执行器，并可视化其子图
            ge = next(executors_it)
            visualize_graph_executor(ge, name + '/', pb_graph,
                                     partial(inline_graph, node=node))

    # 添加普通节点到图中
    def add_node(node):
        if node.kind() == 'prim::FusionGroup':
            return add_fusion_group(node)
        elif node.kind() == 'prim::GraphExecutor':
            return add_graph_executor(node)
        # 获取节点的操作类型和名称
        op, name = name_for(node)
        # 在pb_graph中添加节点，并将输入值映射到当前数值映射中
        pb_node = pb_graph.node.add(op=op, name=name)
        for value in node.inputs():
            pb_node.input.append(value_map[value.unique()])
        # TODO: 处理属性（暂未实现）
        for i, value in enumerate(node.outputs()):
            value_map[value.unique()] = name + ':' + str(i)

    # 遍历图中的每个节点，并添加到可视化图中
    for node in graph.nodes():
        add_node(node)
```