# `.\pytorch\torch\fx\passes\backends\cudagraphs.py`

```py
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 FX 图分区所需的模块和类
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils import _pytree as pytree

# 导入 operator 模块
import operator

# 定义一个继承自 OperatorSupport 的类，支持 CUDA 图
class CudaGraphsSupport(OperatorSupport):
    # TODO: why is submodules passed here
    # 判断节点是否支持 CUDA 图运行
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # 如果节点的操作不在可调用操作列表中，返回 False
        if node.op not in CALLABLE_NODE_OPS:
            return False

        # 如果节点的目标操作为 torch.ops.aten.embedding_dense_backward.default，则返回 False
        if node.target in [torch.ops.aten.embedding_dense_backward.default]:
            return False

        # 如果节点的目标操作为 operator.getitem，则返回 True
        if node.target in [operator.getitem]:
            return True

        found_not_cuda = False

        # 定义一个函数，用于从 meta 中获取值
        def meta_fk(meta):
            return meta["val"] if "val" in meta else meta["fake_result"]

        # 定义一个函数，用于检查是否存在非 CUDA 设备的张量
        def find_not_cuda(t):
            nonlocal found_not_cuda
            if isinstance(t, torch.Tensor) and t.device.type != 'cuda':
                found_not_cuda = True

        # 遍历节点的所有输入节点，检查其中的 meta 数据是否包含非 CUDA 设备的张量
        for n in node.all_input_nodes:
            pytree.tree_map_(find_not_cuda, meta_fk(n.meta))

        # 检查当前节点的 meta 数据是否包含非 CUDA 设备的张量
        pytree.tree_map_(find_not_cuda, meta_fk(node.meta))

        # NB: factory function is accounted for because the result would be
        # cpu or cuda

        # 如果没有找到非 CUDA 设备的张量，则返回 True；否则返回 False
        return not found_not_cuda

# 将 FX 图分区为可以在 CUDA 图下有效运行的子 GraphModules
def partition_cudagraphs(gm, inputs):
    """
    Partition an FX graph into sub-GraphModules that can be validly run under
    CUDA graphs.  For a subgraph to be runnable under CUDA, all of the operations
    must involve CUDA tensors only.
    """
    # 传播假张量属性
    FakeTensorProp(gm).propagate(*inputs)
    # 创建 CudaGraphsSupport 对象，用于支持 CUDA 图操作
    supported_ops = CudaGraphsSupport()
    # TODO: single node partition may be wrong due to the pessimization
    # from copying in and out the data.  Check in benchmarks, perhaps
    # 使用能力基础的分区器进行图分区，允许单节点分区以提高性能
    partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
    # 提议分区
    partitions = partitioner.propose_partitions()
    # 融合分区以生成最终的融合图
    fused_graph = partitioner.fuse_partitions(partitions)
    # 返回融合后的图
    return fused_graph
```