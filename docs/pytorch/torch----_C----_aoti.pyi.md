# `.\pytorch\torch\_C\_aoti.pyi`

```
# 从 ctypes 模块中导入 c_void_p 类型，用于处理指针操作

# 从 torch 模块中导入 Tensor 类型
from torch import Tensor

# 下面几行注释是对特定的 C++ 绑定文件的说明，通常用于告知代码的来源和用途

# 将一组 Tensor 转换为一组 c_void_p 指针的函数声明
def unsafe_alloc_void_ptrs_from_tensors(tensors: list[Tensor]) -> list[c_void_p]: ...

# 将单个 Tensor 转换为 c_void_p 指针的函数声明
def unsafe_alloc_void_ptr_from_tensor(tensor: Tensor) -> c_void_p: ...

# 从一组 c_void_p 指针中分配 Tensor 的函数声明
def alloc_tensors_by_stealing_from_void_ptrs(
    handles: list[c_void_p],
) -> list[Tensor]: ...

# 从单个 c_void_p 指针中分配 Tensor 的函数声明
def alloc_tensor_by_stealing_from_void_ptr(
    handle: c_void_p,
) -> Tensor: ...

# 定义 AOTIModelContainerRunnerCpu 类，用于运行 AOTI 模型容器的 CPU 版本
class AOTIModelContainerRunnerCpu: ...

# 定义 AOTIModelContainerRunnerCuda 类，用于运行 AOTI 模型容器的 CUDA 版本
class AOTIModelContainerRunnerCuda: ...
```