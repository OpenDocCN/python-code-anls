# `.\pytorch\torch\fx\passes\graph_manipulation.py`

```
# 指定允许未标记类型的函数和方法
# 导入必要的类型
from typing import Any, Dict, List, NamedTuple, Optional

# 导入 PyTorch 库
import torch
# 导入 Torch FX 兼容性模块
from torch.fx._compatibility import compatibility
# 导入 Torch FX 图形模块
from torch.fx.graph import Graph
# 导入 Torch FX 图形模块的模块化图形
from torch.fx.graph_module import GraphModule
# 导入 Torch FX 节点
from torch.fx.node import (
    map_arg,
    Node,
    Target,
)
# 导入 Torch FX 形状传播模块
from torch.fx.passes.shape_prop import ShapeProp

# 定义可导出的符号列表
__all__ = ['replace_target_nodes_with', 'size_bytes', 'get_size_of_all_nodes', 'get_tensor_meta',
           'get_size_of_node']

# 兼容性修饰器，标识不向后兼容
@compatibility(is_backward_compatible=False)
def replace_target_nodes_with(
    fx_module: GraphModule,
    old_op: str,
    old_target: Target,
    new_op: str,
    new_target: Target,
):
    """修改 fx_module.graph.nodes 中所有与指定操作码和目标匹配的节点，并更新它们以匹配新的操作码和目标"""
    # 创建一个新图形对象
    new_graph = Graph()
    # 创建值映射字典
    val_map: Dict[Node, Node] = {}
    # 遍历原图中的每个节点
    for node in fx_module.graph.nodes:
        # 如果节点的操作码和目标与旧操作码和目标匹配
        if node.op == old_op and node.target == old_target:
            # 映射节点的参数和关键字参数
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            # 使用新操作码和目标创建新节点，并更新值映射
            val_map[node] = new_graph.create_node(
                new_op, new_target, args, kwargs, node.name
            )
        else:
            # 复制节点到新图形，并更新值映射
            val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    # 更新 fx_module 的图形为新图形
    fx_module.graph = new_graph


# 定义命名元组 size_bytes
@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    output_size: int
    total_size: int


# 计算并更新所有节点的大小
@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(
    fx_module: GraphModule, args: Optional[List[torch.Tensor]] = None
) -> None:
    """给定一个 FX 图模块，更新每个节点的总大小（权重 + 偏置 + 输出）和输出大小（输出）。对于非模块节点，总大小即为输出大小。
    返回总大小"""
    if args is not None:
        # 对每个节点标记形状和数据类型（node.shape 和 node.dtype）
        ShapeProp(fx_module).propagate(*args)
    # 计算整个 FX 图的总大小
    total_size_of_graph = 0.0
    # 遍历图中的每个节点
    for node in fx_module.graph.nodes:
        if node.op == "output":
            break
        # 计算节点的大小并赋值给 node.size_bytes
        node.size_bytes = get_size_of_node(fx_module, node)
    return


# 获取节点的张量元数据
@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> Any:
    tensor_meta = node.meta.get("tensor_meta")

    if not tensor_meta:
        # 如果节点没有与之关联的张量元数据，则抛出运行时错误
        raise RuntimeError(
            f"Node {node} has no tensor metadata associated with it! "
            f"Check that shape propagation has run."
        )

    return tensor_meta


# 计算节点的大小
@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """给定具有 node.dtype 和 node.shape 的节点，返回其总大小和输出大小。
    总大小 = 权重 + 偏置 + 输出大小
    """
    # 总元素数
    total_num_of_elems = 0
    # 如果节点操作是调用模块
    if node.op == "call_module":
        # 从 fx_module 中获取所有命名子模块的字典
        submodule_dict = dict(fx_module.named_modules())
        # 获取目标节点对应的子模块
        submodule = submodule_dict[node.target]
        # 获取子模块的所有参数
        parameters = submodule.named_parameters()
        # 遍历子模块的每个参数（参数是命名元组）
        for name, p in parameters:
            # 计算所有参数元素的总数
            total_num_of_elems += p.numel()
    
    # 计算节点输出的元数据信息
    # node.shape 是该节点输出的形状
    tensor_meta = get_tensor_meta(node)
    # 计算节点输出的元素总数
    output_elem = tensor_meta.shape.numel()
    total_num_of_elems += output_elem
    
    # 假设输出是量化的，则元素大小为 qint8 或 quint8
    if tensor_meta.is_quantized:
        # 获取每个元素的字节大小
        size_per_elem_bytes = torch._empty_affine_quantized(
            [], dtype=tensor_meta.dtype
        ).element_size()
    else:
        # 获取每个元素的字节大小（未量化情况）
        size_per_elem_bytes = torch.tensor([], dtype=tensor_meta.dtype).element_size()
    
    # 计算所有元素的总字节数
    total_size = size_per_elem_bytes * total_num_of_elems
    # 计算输出元素的总字节数
    output_size = size_per_elem_bytes * output_elem
    
    # 返回计算出的字节大小
    return size_bytes(output_size, total_size)
```