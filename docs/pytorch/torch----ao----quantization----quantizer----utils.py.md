# `.\pytorch\torch\ao\quantization\quantizer\utils.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型注释
from typing import List

# 导入用于检查节点是否是符号大小节点的函数
from torch.ao.quantization.pt2e.utils import _is_sym_size_node

# 导入量化注释类
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation
# 导入节点类
from torch.fx import Node


def _annotate_input_qspec_map(node: Node, input_node: Node, qspec):
    # 获取节点的量化注释，如果不存在则创建新的量化注释对象
    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    # 如果输入量化规格映射不存在，创建一个空字典
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}
    # 将输入节点与其量化规格映射添加到输入量化规格映射中
    quantization_annotation.input_qspec_map[input_node] = qspec
    # 更新节点的量化注释
    node.meta["quantization_annotation"] = quantization_annotation


def _annotate_output_qspec(node: Node, qspec):
    # 获取节点的量化注释，如果不存在则创建新的量化注释对象
    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    # 设置节点的输出量化规格
    quantization_annotation.output_qspec = qspec
    # 更新节点的量化注释
    node.meta["quantization_annotation"] = quantization_annotation


def _node_only_used_for_sym_size(node: Node, partition_nodes: List[Node]):
    """
    该实用函数用于处理在动态形状跟踪（dynami_shape=True）中出现线性模块模式中的符号整数节点情况。
    在这些情况下，我们需要区分仅用于提取某些维度值（和符号整数节点）的节点与激活节点。
    例如：
    graph(x, y, weight):
       size_0 = torch.ops.aten.sym_size([x], [0])
       size_1 = torch.ops.aten.sym_size([y], [1])
       view_size = size_0 * size_1
       size_3 = torch.ops.aten.sym_size([x], [2])
       vie_out = torch.ops.aten.view(x, [view_size, size_3])
       return mm(view_out, weight)
    在上面的示例中，y 节点并不是实际的输入。它仅存在以提取 size_1。
    """
    # 如果节点是符号大小节点，则返回True
    if _is_sym_size_node(node):
        return True

    # 检查节点的所有用户，确保它们要么不在分区节点中，要么是符号大小节点
    return all(
        ((user not in partition_nodes) or _is_sym_size_node(user))
        for user in node.users
    )


def _get_module_name_filter(module_name: str):
    """
    根据给定的模块名获取模块名称过滤器函数，该过滤器接受一个节点，并检查节点是否来自具有特定模块名称的模块

    例如：
        node: linear_op = call_function[...](...)  # 来自名称为 blocks.sub.linear1 的模块

    >> module_name_filter = _get_module_name_filter("blocks.sub")
    >> print(module_name_filter(node))
    True  # 基于完全限定名称 "blocks.sub.linear1"，节点来自 "blocks.sub"
    """
    # 定义一个函数 module_name_filter，接受一个 Node 类型的参数 n，返回一个布尔值
    def module_name_filter(n: Node) -> bool:
        # 从节点 n 的元数据中获取 nn_module_stack 字段，如果不存在则为空字典
        nn_module_stack = n.meta.get("nn_module_stack", {})

        # 定义内部函数 _normalize_path，用于规范化路径 n
        def _normalize_path(n):
            prefix = 0
            # TODO 这是非标准行为，当我们迁移至 capture_pre_autograd_graph 后应删除
            # 如果路径 n 以 "L['self']." 开头，则设置前缀长度为 len("L['self'].")
            if n.startswith("L['self']."):
                prefix = len("L['self'].")
            return n[prefix:]

        # 从 nn_module_stack 中获取所有键值对的第一个元素，经过 _normalize_path 规范化后组成 names 列表
        names = [_normalize_path(n) for n, _ in nn_module_stack.values()]

        # 返回 module_name 是否在 names 列表中的结果
        return module_name in names

    # 返回函数 module_name_filter 自身，而不是调用它的结果
    return module_name_filter
```