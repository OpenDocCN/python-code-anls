# `.\pytorch\benchmarks\fastrnns\fuser.py`

```py
# 导入 torch 库，用于深度学习框架 PyTorch
import torch


# 设置混合器（fuser）和执行器（executor）的参数
def set_fuser(fuser_name, executor_name):
    # 确保 fuser_name 在指定的选项中
    assert fuser_name in ["te", "old", "none", "default"]
    
    # 根据 fuser_name 的不同选项进行不同设置
    if fuser_name == "te":
        # 启用分析执行器（profiling executor），启用图优化
        torch._C._jit_set_profiling_executor(True)
        torch._C._get_graph_executor_optimize(True)
        # 禁用 CPU 上的融合
        torch._C._jit_override_can_fuse_on_cpu(False)
        # 启用 GPU 上的融合
        torch._C._jit_override_can_fuse_on_gpu(True)
        # 启用张量表达式（tensor expression）融合器
        torch._C._jit_set_texpr_fuser_enabled(True)
    elif fuser_name == "old":
        # 禁用分析执行器，禁用图优化
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        # 启用 GPU 上的融合
        torch._C._jit_override_can_fuse_on_gpu(True)
        # 禁用张量表达式融合器
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == "none":
        # 禁用分析执行器，禁用图优化
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        # 禁用 GPU 上的融合和 CPU 上的融合
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        # 禁用张量表达式融合器
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == "default":
        # 默认情况下不做任何操作
        pass

    # 根据 executor_name 的不同选项进行不同设置
    # --executor 覆盖 --fuser 的设置
    if executor_name == "profiling":
        # 启用分析执行器，启用图优化
        torch._C._jit_set_profiling_executor(True)
        torch._C._get_graph_executor_optimize(True)
    elif executor_name == "simple":
        # 禁用图优化
        torch._C._get_graph_executor_optimize(False)
    elif executor_name == "legacy":
        # 禁用分析执行器，启用图优化
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(True)
    elif executor_name == "default":
        # 默认情况下不做任何操作
        pass
```