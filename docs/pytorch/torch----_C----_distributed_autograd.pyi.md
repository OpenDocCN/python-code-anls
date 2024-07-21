# `.\pytorch\torch\_C\_distributed_autograd.pyi`

```
# mypy: allow-untyped-defs
# 导入 torch 库，用于分布式自动求导功能
import torch

# 以下是分布式自动求导模块的定义，实现在 torch/csrc/distributed/autograd/init.cpp 中

# 分布式自动求导上下文类，定义了一系列方法用于管理上下文
class DistAutogradContext:
    # 返回上下文 ID 的方法声明，返回整数类型
    def _context_id(self) -> int: ...

    # 返回接收函数字典的方法声明，接收函数是一个从整数到任意类型的映射
    def _recv_functions(self) -> dict[int, Any]: ...

    # 返回发送函数字典的方法声明，发送函数是一个从整数到任意类型的映射
    def _send_functions(self) -> dict[int, Any]: ...

    # 返回已知 worker ID 集合的方法声明，这是一个整数集合
    def _known_worker_ids(self) -> set[int]: ...

# 创建新的分布式自动求导上下文的方法声明，返回 DistAutogradContext 类型
def _new_context() -> DistAutogradContext: ...

# 释放指定上下文 ID 的方法声明，无返回值
def _release_context(context_id: int) -> None: ...

# 获取最大上下文 ID 的方法声明，返回整数类型
def _get_max_id() -> int: ...

# 检查给定 worker ID 是否是有效上下文的方法声明，返回布尔类型
def _is_valid_context(worker_id: int) -> bool: ...

# 检索指定上下文 ID 的方法声明，返回 DistAutogradContext 类型
def _retrieve_context(context_id: int) -> DistAutogradContext: ...

# 返回当前上下文的方法声明，返回 DistAutogradContext 类型
def _current_context() -> DistAutogradContext: ...

# 初始化分布式自动求导功能，传入 worker ID，无返回值
def _init(worker_id: int) -> None: ...

# 获取调试信息的方法声明，返回字符串到字符串的字典
def _get_debug_info() -> dict[str, str]: ...

# 分布式自动求导的反向传播函数声明，接收上下文 ID、根张量列表、是否保留计算图的布尔参数，无返回值
def backward(
    context_id: int,
    roots: list[torch.Tensor],
    retain_graph=False,
) -> None: ...

# 获取梯度信息的方法声明，接收上下文 ID，返回从张量到张量的字典
def get_gradients(context_id: int) -> dict[torch.Tensor, torch.Tensor]: ...
```