# `.\pytorch\torch\distributed\_spmd\batch_dim_utils.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import Callable, Dict, List, Set

# 导入PyTorch相关库
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed._tensor.ops.view_ops import dim_maps, DimSpec, InputDim
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec

# 使用 torch.ops.aten 别名来引用 PyTorch 的 aten 操作
aten = torch.ops.aten

# 定义 BatchDimAnalyzer 类，用于分析图中每个张量/节点的批次维度
class BatchDimAnalyzer:
    """This class is used to analyze the batch dimension of each tensor/node in the graph.

    We need to know the batch dimension of each tensor/node so that we know
    exactly the sharding layout of intermediate tensors.

    We possibly should evaluate using symbolic shapes to track the batch dimension.
    We can experiment it later with dynamo integration (as dynamo have mark_dynamic
    API which allows marking batch dimension only) or try to use FakeTensorMode to
    mark the batch dimension. For now, let's just use the batch dimension of the first
    input tensor as the hint to track the batch dimension of all tensors/nodes in
    the graph.
    """

    def __init__(self, batch_dim: int = 0) -> None:
        # 初始化 BatchDimAnalyzer 对象，设置默认批次维度
        self.batch_dim = batch_dim

        # 用于存储每个节点（fx.Node）的批次维度映射
        self.batch_dim_map: Dict[fx.Node, int] = {}

        # 用于跟踪输入张量的批次维度大小
        self.batch_dim_size = -1

        # 定义操作符和对应的处理函数的映射关系
        self.dim_rule_map: Dict[torch._ops.OpOverload, Callable[..., torch.Tensor]] = {
            aten.squeeze.default: torch.squeeze,
            aten.squeeze.dim: torch.squeeze,
            aten.view.default: Tensor.view,
            aten.reshape.default: torch.reshape,
            aten._unsafe_view.default: Tensor.view,
            aten.unsqueeze.default: torch.unsqueeze,
            aten.expand.default: Tensor.expand,
            aten.permute.default: torch.permute,
            aten.repeat.default: Tensor.repeat,
            aten.transpose.int: torch.transpose,
        }

    def init_batch_dim_size(self, batch_dim_size: int) -> None:
        """Initialize batch dim size base on the first input batch size."""
        # 初始化批次维度大小，确保不重复初始化且与现有的大小一致
        if self.batch_dim_size != -1 and self.batch_dim_size != batch_dim_size:
            raise RuntimeError(
                f"batch dim size is already initialized! "
                f"Found new batch size: {batch_dim_size} not "
                f"matching existing batch dim size: {self.batch_dim_size}!"
            )
        self.batch_dim_size = batch_dim_size

    def set_batch_dim(self, node: fx.Node, batch_dim: int) -> None:
        """Set batch dimension for a specific node in the graph."""
        # 设置特定节点的批次维度映射
        self.batch_dim_map[node] = batch_dim

    def get_batch_dim(self, node: fx.Node) -> int:
        """Retrieve batch dimension for a specific node from the graph."""
        # 获取特定节点的批次维度映射，若未找到则引发运行时错误
        if node not in self.batch_dim_map:
            raise RuntimeError(f"batch dim analysis failed on node: {node}!")
        return self.batch_dim_map[node]
    # 定义一个方法，用于计算节点的活动规范，返回一个 DTensorSpec 对象
    def compute_act_spec(self, node: fx.Node, mesh: DeviceMesh) -> DTensorSpec:
        """Compute the batch dimension for the current node, then generate the sharding spec that shards on the batch dimension."""
        
        # 调用 compute_batch_dim 方法计算当前节点的批量维度
        node_batch_dim = self.compute_batch_dim(node)
        
        # 如果节点的批量维度为 -1，表示此激活是复制的
        if node_batch_dim == -1:
            # 创建一个 DTensorSpec 对象，指定在给定的 mesh 上复制激活
            act_spec = DTensorSpec(mesh=mesh, placements=(Replicate(),))
        
        # 如果节点的批量维度为 -2，表示此激活是部分的
        elif node_batch_dim == -2:
            # 创建一个 DTensorSpec 对象，指定在给定的 mesh 上部分激活
            act_spec = DTensorSpec(mesh=mesh, placements=(_Partial(),))
        
        # 如果节点的批量维度为其他数值，表示此激活是分片的
        else:
            # 创建一个 DTensorSpec 对象，指定在给定的 mesh 上按节点的批量维度进行分片
            act_spec = DTensorSpec(mesh=mesh, placements=(Shard(node_batch_dim),))
        
        # 返回生成的 DTensorSpec 对象，用于描述激活的分布方式
        return act_spec
```