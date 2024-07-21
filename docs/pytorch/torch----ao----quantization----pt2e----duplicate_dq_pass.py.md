# `.\pytorch\torch\ao\quantization\pt2e\duplicate_dq_pass.py`

```
# 添加类型检查允许未类型化定义
# 导入日志记录模块
import logging
# 导入操作符模块
import operator

# 导入 PyTorch 库
import torch

# 从 PyTorch AO 量化模块中导入工具函数
from torch.ao.quantization.pt2e.utils import (
    _filter_sym_size_users,
    _is_valid_annotation,
)

# 从 PyTorch FX 模块中导入节点映射函数
from torch.fx.node import map_arg
# 从 PyTorch FX 通行证基础模块中导入通行证基类和通行证结果
from torch.fx.passes.infra.pass_base import PassBase, PassResult

# 获取当前模块的日志记录器实例
logger = logging.getLogger(__name__)
# 设置日志记录器的日志级别为警告
logger.setLevel(logging.WARNING)

# 声明模块的公开接口列表
__all__ = ["DuplicateDQPass"]

# 定义量化操作的列表
_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]

# 定义反量化操作的列表
_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]

# 定义函数 _maybe_duplicate_dq，用于可能复制 DQ 节点
def _maybe_duplicate_dq(
    gm: torch.fx.GraphModule, dq_node: torch.fx.Node, user: torch.fx.Node
):
    # 获取用户节点中的量化注释
    annotation = user.meta.get("quantization_annotation", None)
    # 如果注释无效，则直接返回
    if not _is_valid_annotation(annotation):
        return
    # 在图模块中插入新节点来复制 DQ 节点
    with gm.graph.inserting_after(dq_node):
        # 复制 DQ 节点生成新节点
        new_node = gm.graph.node_copy(dq_node)

        # 定义替换节点的函数
        def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
            if n == dq_node:
                return new_node
            else:
                return n

        # 使用映射函数替换用户节点的参数和关键字参数
        new_args = map_arg(user.args, maybe_replace_node)
        new_kwargs = map_arg(user.kwargs, maybe_replace_node)
        user.args = new_args
        user.kwargs = new_kwargs

# 定义通行证类 DuplicateDQPass，继承自 PassBase 类
class DuplicateDQPass(PassBase):
    # 遍历图中的每个节点
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            # 检查节点操作是否为函数调用且目标函数在_DEQUANTIZE_OPS中
            if node.op == "call_function" and node.target in _DEQUANTIZE_OPS:
                # 获取所有使用当前节点作为符号尺寸的节点
                dq_users = _filter_sym_size_users(node)
                # 如果符号尺寸使用节点数量小于等于1，则跳过
                if len(dq_users) <= 1:
                    continue
                # 避免为动态量化重复创建dq节点
                # 模式：choose_qparam - getitem - q - dq
                q_node = node.args[0]
                # 检查q节点是否为函数调用且目标函数在_QUANTIZE_OPS中
                if q_node.op == "call_function" and q_node.target in _QUANTIZE_OPS:
                    getitem_node = q_node.args[1]
                    # 检查getitem节点是否为torch.fx.node.Node类型且目标函数为getitem
                    if (
                        isinstance(getitem_node, torch.fx.node.Node)
                        and getitem_node.op == "call_function"
                        and getitem_node.target == operator.getitem
                    ):
                        choose_qparam_node = getitem_node.args[0]
                        # 检查choose_qparam节点是否为torch.fx.node.Node类型且目标函数为torch.ops.quantized_decomposed.choose_qparams.tensor
                        if (
                            isinstance(choose_qparam_node, torch.fx.node.Node)
                            and choose_qparam_node.op == "call_function"
                            and choose_qparam_node.target == torch.ops.quantized_decomposed.choose_qparams.tensor
                        ):
                            continue
                # 对于每个使用dq节点的用户节点，可能会复制dq节点
                for user in dq_users:
                    _maybe_duplicate_dq(graph_module, node, user)
        # 清除图中的死代码
        graph_module.graph.eliminate_dead_code()
        # 重新编译图模块
        graph_module.recompile()
        # 返回处理结果
        return PassResult(graph_module, True)
```