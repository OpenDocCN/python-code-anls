# `.\pytorch\torch\_inductor\codegen\cuda_combined_scheduling.py`

```py
# mypy: allow-untyped-defs
# 导入必要的类型声明
from typing import Sequence, Union

# 导入调度器相关模块
from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
# 导入 CUDA 和 ROCm 的 C++ 调度模块
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .rocm.rocm_cpp_scheduling import ROCmCPPScheduling

# 导入 Triton 调度模块
from .triton import TritonScheduling

# CUDA 组合调度器，继承自 BaseScheduling
class CUDACombinedScheduling(BaseScheduling):
    """
    Scheduler for CUDA Kernels, which delegates calls as appropriate
    to the CUDA-C++ and Triton Schedulers, which both work for CUDA devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CUDA C++ code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Scheduler):
        super().__init__()
        # 初始化调度器和相关的 Triton, CUDA-C++ 和 ROCm-C++ 调度器
        self._scheduler = scheduler
        self._triton_scheduling = TritonScheduling(scheduler)
        self._cuda_cpp_scheduling = CUDACPPScheduling(scheduler)
        self._rocm_cpp_scheduling = ROCmCPPScheduling(scheduler)

    def get_backend_features(self, device):
        # 获取后端设备的特性，由 Triton 调度器处理
        return self._triton_scheduling.get_backend_features(device)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        # 选择节点的后端调度器，优先选择 CUDA-C++ 或 ROCm-C++ 调度器
        if self._cuda_cpp_scheduling.is_cuda_cpp_template(node):
            return self._cuda_cpp_scheduling
        if self._rocm_cpp_scheduling.is_rocm_cpp_template(node):
            return self._rocm_cpp_scheduling
        # 默认使用 Triton 调度器
        return self._triton_scheduling

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        # 判断是否可以垂直融合节点，优先使用 CUDA-C++ 调度器
        if self._cuda_cpp_scheduling.can_fuse_vertical(node1, node2):
            return True
        # 否则使用 Triton 调度器
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        # 判断是否可以水平融合节点，对于 CUDA-C++ 节点，当前总是返回 False
        for node in (node1, node2):
            if self._cuda_cpp_scheduling.is_cuda_cpp_template(node):
                return self._cuda_cpp_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
        # 否则使用 Triton 调度器判断
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

    def group_fn(self, sizes):
        # 使用 Triton 调度器进行分组函数操作
        return self._triton_scheduling.group_fn(sizes)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        # 实现模板代码生成，使用 Triton 调度器处理

        epilogue_schedulings: Sequence[Union[BaseScheduling, FusedSchedulerNode]],
    ):
        return self._triton_scheduling.codegen_template(
            template_node, epilogue_nodes, epilogue_schedulings
        )


注释：
    # 如果节点是 CUDA C++ 模板节点，则调用 CUDA C++ 调度器生成模板代码
    if self._cuda_cpp_scheduling.is_cuda_cpp_template(template_node):
        # 断言后续节点为空或者长度为0
        assert epilogue_nodes is None or len(epilogue_nodes) == 0
        return self._cuda_cpp_scheduling.codegen_template(
            template_node, epilogue_nodes
        )
    # 如果节点是 ROCm C++ 模板节点，则调用 ROCm C++ 调度器生成模板代码
    elif self._rocm_cpp_scheduling.is_rocm_cpp_template(template_node):
        # 断言后续节点为空或者长度为0
        assert epilogue_nodes is None or len(epilogue_nodes) == 0
        return self._rocm_cpp_scheduling.codegen_template(
            template_node, epilogue_nodes
        )
    # 否则调用 Triton 调度器生成模板代码
    else:
        return self._triton_scheduling.codegen_template(
            template_node, epilogue_nodes
        )

# 生成节点的代码
def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]):
    return self._triton_scheduling.codegen_node(node)

# 生成同步代码
def codegen_sync(self):
    return self._triton_scheduling.codegen_sync()

# 刷新操作
def flush(self):
    return self._triton_scheduling.flush()

# 生成循环代码
def codegen_foreach(self, *args, **kwargs):
    return self._triton_scheduling.codegen_foreach(*args, **kwargs)

# 对融合节点进行基准测试
def benchmark_fused_nodes(self, nodes):
    return self._triton_scheduling.benchmark_fused_nodes(nodes)

# 从节点生成内核代码
def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):
    return self._triton_scheduling.generate_kernel_code_from_nodes(
        nodes, benchmark_kernel
    )
```