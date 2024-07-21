# `.\pytorch\torch\jit\_passes\_property_propagation.py`

```
"""
# 允许未标注的函数定义类型
Tools to help with tensor property propagation.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

# 导入必要的模块和类型定义
from typing import Any, List

import torch
from torch import TensorType  # 导入张量类型
from torch._C import Graph  # 导入图形对象


def apply_input_props_using_example(graph: Graph, example_input: List[Any]):
    """
    Applies properties for each tensor in the graph inputs
    using the example supplied.
    """
    # 获取图形对象的输入
    graph_inputs = list(graph.inputs())
    if len(graph_inputs) == 0:
        return

    # 对于方法，剥离掉 self 参数
    in_0 = graph_inputs[0]
    if isinstance(in_0.type(), torch._C.ClassType) and in_0.debugName() == "self":
        graph_inputs = graph_inputs[1:]

    # 检查图形输入和示例输入的数量是否一致
    if not len(graph_inputs) == len(example_input):
        raise RuntimeError(
            "Number of inputs in graph does not match number of inputs in the example"
        )

    # 遍历图形输入和示例输入，逐个检查类型匹配
    for i, (graph_i, example_i) in enumerate(zip(graph_inputs, example_input)):
        if example_i is None:
            continue  # 跳过类型检查

        # 检查示例输入和图形输入的类型是否一致
        if isinstance(example_i, torch.Tensor) != isinstance(
            graph_i.type(), TensorType
        ):
            raise RuntimeError(
                f"Input {i} does not match type of example", graph_i, example_i
            )

        # 如果示例输入是张量，则设置图形输入的类型
        if isinstance(example_i, torch.Tensor):
            graph_i.setType(TensorType.create_from_tensor(example_i))  # type: ignore[arg-type]
```