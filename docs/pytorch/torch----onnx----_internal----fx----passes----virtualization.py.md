# `.\pytorch\torch\onnx\_internal\fx\passes\virtualization.py`

```
# 设置 mypy 指令，允许未类型化的定义
# 导入未来版本的 annotations 特性
from __future__ import annotations

# 导入类型提示所需的模块
from typing import List, Optional, Tuple

# 导入 torch 库
import torch
# 导入 torch.fx 模块
import torch.fx

# 从 torch.onnx._internal 模块导入 _beartype
from torch.onnx._internal import _beartype
# 从 torch.onnx._internal.fx 模块导入 _pass
from torch.onnx._internal.fx import _pass

# 定义一个继承自 _pass.Transform 的类 MovePlaceholderToFront
class MovePlaceholderToFront(_pass.Transform):
    """This pass move all placeholder nodes to the front of the graph node list.

    In torch.fx.Graph, placeholder is a special assignment node. If it's not
    executed in the beginning, it could overwrite values computed by upstream
    nodes.
    """

    @_beartype.beartype
    # 定义一个方法 _run，接受任意参数并返回 torch.fx.GraphModule 对象
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        # 获取当前模块的图模块对象
        graph_module = self.module
        # 获取图模块的图对象
        graph = graph_module.graph
        # 初始化空列表 placeholders，用于存放占位符节点
        placeholders = []
        # 初始化 first_not_placeholder 为 None
        first_not_placeholder = None
        # 遍历图中的每个节点
        for node in graph.nodes:
            # 如果节点操作为 "placeholder"，将其添加到 placeholders 列表中
            if node.op == "placeholder":
                placeholders.append(node)
            # 如果 first_not_placeholder 仍为 None 并且节点操作不是 "placeholder"，将其赋值为当前节点
            if first_not_placeholder is None and node.op != "placeholder":
                first_not_placeholder = node
        # 如果没有找到非占位符节点，直接返回原始的 graph_module
        if first_not_placeholder is None:
            return graph_module
        # 将 placeholders 中的占位符节点插入到第一个非占位符节点之前
        for placeholder in placeholders:
            first_not_placeholder.prepend(placeholder)
        # 返回修改后的 graph_module
        return graph_module


# 定义一个继承自 _pass.Transform 的类 ReplaceGetAttrWithPlaceholder
class ReplaceGetAttrWithPlaceholder(_pass.Transform):
    """Replace get_attr with placeholder.

    The parameters and buffers accessed by the original get_attr are returned;
    they are useful when creating random inputs for the modified graph_module.
    """

    # 属性 _replaced_attrs，用于存放被替换的权重张量的元组或空值
    _replaced_attrs: Optional[Tuple[torch.Tensor, ...]]

    @property
    # 定义属性 replaced_attrs，返回被替换的权重张量的元组
    def replaced_attrs(self) -> Tuple[torch.Tensor, ...]:
        """The list of replaced weight tensors."""
        # 断言 _replaced_attrs 不为 None，否则抛出异常
        assert (
            self._replaced_attrs is not None
        ), "Must run ReplaceGetAttrWithPlaceholder first"
        return self._replaced_attrs

    @_beartype.beartype
    # 定义方法 _run，接受任意参数并返回 torch.fx.GraphModule 对象
    # 定义一个方法 _run，返回类型为 torch.fx.GraphModule
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        # 获取当前模块的图形模块
        graph_module = self.module
        # 获取图形模块的计算图
        graph = graph_module.graph
        # 初始化一个列表用于存储被替换的属性（tensor）
        replaced_attrs: List[torch.Tensor] = []

        # 遍历计算图中的每个节点
        for node in graph.nodes:
            # 如果节点操作是 "get_attr"
            if node.op == "get_attr":
                # 初始化一个变量用于存储被替换的属性，初始值为 None
                replaced_attr: Optional[torch.Tensor] = None

                # 尝试从 graph_module 中获取节点目标对应的参数
                try:
                    replaced_attr = graph_module.get_parameter(node.target)
                # 如果抛出 AttributeError 异常，说明该节点目标是 buffer 而不是 parameter
                except AttributeError:
                    # 从 graph_module 中获取节点目标对应的 buffer
                    replaced_attr = graph_module.get_buffer(node.target)

                # 将节点操作类型修改为 "placeholder"，使 get_attr 节点变为占位符节点
                node.op = "placeholder"

                # 将节点目标中的 "." 替换为 "_"，确保在占位符中的目标名是有效的 Python 标识符
                node.target = node.target.replace(".", "_")

                # 将节点的参数设为 None，以匹配原始 forward 方法的可选输入
                node.args = (None,)

                # 将被替换的属性添加到列表中
                replaced_attrs.append(replaced_attr)

        # 将替换后的属性元组化并赋值给实例变量 _replaced_attrs
        self._replaced_attrs = tuple(replaced_attrs)

        # 返回修改后的图形模块
        return graph_module
```