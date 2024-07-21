# `.\pytorch\torch\jit\_fuser.py`

```
# mypy: allow-untyped-defs
import contextlib  # 导入上下文管理模块
from typing import List, Tuple  # 导入类型提示相关模块

import torch  # 导入 PyTorch 模块


@contextlib.contextmanager
def optimized_execution(should_optimize):
    """Context manager that controls whether the JIT's executor will run optimizations before executing a function."""
    stored_flag = torch._C._get_graph_executor_optimize()  # 获取当前的图执行器优化标志
    torch._C._set_graph_executor_optimize(should_optimize)  # 设置图执行器的优化标志为指定的值
    try:
        yield  # 执行上下文管理器的主体部分
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)  # 恢复图执行器的优化标志为原先保存的值


@contextlib.contextmanager
def fuser(name):
    """Context manager that facilitates switching between backend fusers.

    Valid names:
    * ``fuser0`` - enables only legacy fuser
    * ``fuser1`` - enables only NNC
    * ``fuser2`` - enables only nvFuser
    * ``fuser3`` - enables oneDNN Graph
    """
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()  # 获取当前 CPU 融合的状态
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()  # 获取当前 GPU 融合的状态
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()  # 获取当前张量表达式融合器的状态
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()  # 获取当前 nvFuser 融合器的状态
    old_llga_state = torch._C._jit_llga_enabled()  # 获取当前 LLGA 融合器的状态
    if name == "fuser0":  # 如果选择的是 legacy fuser
        torch._C._jit_override_can_fuse_on_cpu(True)  # 允许 CPU 融合
        torch._C._jit_override_can_fuse_on_gpu(True)  # 允许 GPU 融合
        torch._C._jit_set_texpr_fuser_enabled(False)  # 禁用张量表达式融合器
        torch._C._jit_set_nvfuser_enabled(False)  # 禁用 nvFuser 融合器
        torch._C._jit_set_llga_enabled(False)  # 禁用 LLGA 融合器
    elif name == "fuser1":  # 如果选择的是 NNC
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)  # 启用性能分析执行器
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)  # 获取当前的图执行器优化模式
        torch._C._jit_override_can_fuse_on_cpu(True)  # 允许 CPU 融合
        torch._C._jit_override_can_fuse_on_gpu(True)  # 允许 GPU 融合
        torch._C._jit_set_texpr_fuser_enabled(True)  # 启用张量表达式融合器
        torch._C._jit_set_nvfuser_enabled(False)  # 禁用 nvFuser 融合器
        torch._C._jit_set_llga_enabled(False)  # 禁用 LLGA 融合器
    elif name == "fuser2":  # 如果选择的是 nvFuser
        torch._C._jit_override_can_fuse_on_cpu(False)  # 禁用 CPU 融合
        torch._C._jit_override_can_fuse_on_gpu(False)  # 禁用 GPU 融合
        torch._C._jit_set_texpr_fuser_enabled(False)  # 禁用张量表达式融合器
        torch._C._jit_set_nvfuser_enabled(True)  # 启用 nvFuser 融合器
        torch._C._jit_set_llga_enabled(False)  # 禁用 LLGA 融合器
    elif name == "fuser3":  # 如果选择的是 oneDNN Graph
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)  # 启用性能分析执行器
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)  # 获取当前的图执行器优化模式
        torch._C._jit_override_can_fuse_on_cpu(True)  # 允许 CPU 融合
        torch._C._jit_override_can_fuse_on_gpu(False)  # 禁用 GPU 融合
        torch._C._jit_set_texpr_fuser_enabled(True)  # 启用张量表达式融合器
        torch._C._jit_set_nvfuser_enabled(False)  # 禁用 nvFuser 融合器
        torch._C._jit_set_llga_enabled(True)  # 启用 LLGA 融合器
    elif name == "none":  # 如果选择的是关闭 PyTorch 融合器
        torch._C._jit_override_can_fuse_on_cpu(False)  # 禁用 CPU 融合
        torch._C._jit_override_can_fuse_on_gpu(False)  # 禁用 GPU 融合
        torch._C._jit_set_texpr_fuser_enabled(False)  # 禁用张量表达式融合器
        torch._C._jit_set_nvfuser_enabled(False)  # 禁用 nvFuser 融合器
        torch._C._jit_set_llga_enabled(False)  # 禁用 LLGA 融合器
    else:
        raise Exception(f"unrecognized fuser option (name: {name})")  # 抛出异常，未识别的融合器选项
    try:
        yield  # 执行上下文管理器的主体部分
    ```
    # 最终执行块，无论是否发生异常都会执行
    finally:
        # 如果名称在 ["fuser1", "fuser3"] 中，表示是 NNC 或者 oneDNN 图
        if name in ["fuser1", "fuser3"]:
            # 恢复旧的性能分析执行器设置
            torch._C._jit_set_profiling_executor(old_profiling_executor)  # type: ignore[possibly-undefined]
            # 恢复旧的图执行器优化设置
            torch._C._get_graph_executor_optimize(old_profiling_mode)  # type: ignore[possibly-undefined]
        
        # 恢复之前的 CPU 融合设置
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        # 恢复之前的 GPU 融合设置
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        # 恢复之前的张量表达式融合器启用状态
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        # 恢复之前的 NV 融合器启用状态
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)
        # 恢复之前的 LLGA 融合器启用状态
        torch._C._jit_set_llga_enabled(old_llga_state)
# 获取最近优化过的图形对象，通常由 PyTorch 内部维护
last_executed_optimized_graph = torch._C._last_executed_optimized_graph

# 递归地查找包含不同可微图节点的节点，并将它们添加到 diff_node 列表中
def _get_differentiable_graph_node(node, diff_node):
    if node.kind() == "prim::DifferentiableGraph":
        diff_node.append(node)
    else:
        for block in node.blocks():
            for n in block.nodes():
                _get_differentiable_graph_node(n, diff_node)

# 返回给定对象的脚本方法的图形表示
def _graph_for(self, *args, **kwargs):
    return _script_method_graph_for(self, self, *args, **kwargs)

# 返回给定对象的脚本方法的图形表示，处理异常情况
def _script_method_graph_for(self, parent, *args, **kwargs):
    try:
        # 获取调试状态，包含执行计划
        dbs = parent.get_debug_state()
        eps = list(dbs.execution_plans.values())
        assert len(eps) == 1
        # 复制执行计划中的图形
        graph = eps[0].graph.copy()

        # 获取可微节点的前向状态
        fw_states = eps[0].code.differentiable_op_executor_states()
        diff_nodes: List[torch._C.Node] = []
        # 遍历图中的节点，找到所有可微节点并添加到 diff_nodes 列表中
        for n in graph.nodes():
            _get_differentiable_graph_node(n, diff_nodes)

        assert len(fw_states) == len(diff_nodes)
        # 将每个可微图形与其执行计划中的优化图形交换
        for n, state in zip(diff_nodes, fw_states):
            fw_execution_plans = list(state.execution_plans.values())
            # 只有在存在唯一执行计划时才更新子图形
            if len(fw_execution_plans) == 1:
                n.g_("Subgraph", fw_execution_plans[0].graph)

        return graph
    except Exception:
        # 异常处理：运行脚本方法并返回记录的最近优化图形作为后备方法
        self(*args, **kwargs)
        return last_executed_optimized_graph()

# 设置融合策略的类型和特化数量
def set_fusion_strategy(strategy: List[Tuple[str, int]]):
    """Set the type and number of specializations that can occur during fusion.

    Usage: provide a list of pairs (type, depth) where type is one of "STATIC" or "DYNAMIC"
    and depth is an integer.

    Behavior - static vs dynamic:
        In STATIC fusion, fused ops are compiled to have fixed input shapes. The shape is determined
        based on some initial profiling runs.
        In DYNAMIC fusion, fused ops are compiled to have variable input shapes, so that multiple
        shapes are possible.

    In both cases, we also recompile on new striding behavior, device, or dtype.

    Behavior - fallback functions & depth:
        When an input doesn't match the format required by the specialized compiled op, it will run
        a fallback function. Fallback functions are recursively be compiled and specialized based
        on the observed tensor shapes. Since compilation can be slow, the "depth" parameter is provided to
        limit the number of specializations that can be compiled, before giving up on recompiling and
        falling back to a completely un-fused, un-specialized implementation.
    """
    # 列表中的 (类型, 深度) 对控制特化的类型和数量
    # 例如：[("STATIC", 2), ("DYNAMIC", 2)] 表示前两个特化将使用静态融合，接下来两个特化将使用动态融合，
    # 任何不满足这四个选项的输入将运行未融合的实现

    # 注意：在未来，如果添加更多的融合后端，可能会有更细粒度的特定融合器的 API
    """
    设置融合策略，返回设置后的结果
    """
    return torch._C._jit_set_fusion_strategy(strategy)
```