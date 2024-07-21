# `.\pytorch\torch\fx\passes\annotate_getitem_nodes.py`

```py
import operator  # 导入 operator 模块，用于操作符相关的函数

import torch  # 导入 PyTorch 模块


def annotate_getitem_nodes(graph: torch.fx.Graph) -> None:
    """
    Annotate the type of getitem nodes, inferred from the type of sequence node.
    If sequence node is not annotated with a type, do nothing.
    Currently support getitem nodes from Tuple, List, and NamedTuple sequence node.

    This is helpful since annotations on local names within function are lost during FX transforms.
    Adding back known type annotation for getitem nodes to improve jit scriptability.

    Args:
        graph (Graph): The graph to be annotated
    """
    for node in graph.nodes:  # 遍历图中的每个节点
        if node.target == operator.getitem:  # 判断节点是否为 getitem 操作符
            sequence_node, index_node = node.args  # 获取 getitem 节点的两个参数：sequence_node 和 index_node
            if not sequence_node.type:  # 如果 sequence_node 没有类型注释，则跳过此节点
                continue
            # container types
            if hasattr(sequence_node.type, "_name"):  # 如果 sequence_node 类型有 "_name" 属性
                parameterized_types = sequence_node.type.__args__  # 获取参数化类型
                if sequence_node.type._name == "Tuple":  # 如果是元组类型
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]  # 设置节点类型为元组的第一个参数类型
                    else:
                        assert len(parameterized_types) > index_node
                        node_type = parameterized_types[index_node]
                        node.type = node_type  # 设置节点类型为元组指定位置的类型
                elif sequence_node.type._name == "List":  # 如果是列表类型
                    assert len(parameterized_types) == 1
                    node.type = parameterized_types[0]  # 设置节点类型为列表的参数类型
            # NamedTuple type
            elif hasattr(sequence_node.type, "__annotations__"):  # 如果 sequence_node 类型有 "__annotations__" 属性
                if sequence_node.type == torch.Tensor:  # 如果是 Torch 张量类型，跳过
                    continue
                sequence_node_field_types = sequence_node.type.__annotations__  # 获取命名元组的字段类型字典
                field_name = sequence_node.type._fields[index_node]
                node.type = sequence_node_field_types[field_name]  # 设置节点类型为命名元组指定字段的类型
```