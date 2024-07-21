# `.\pytorch\torch\_functorch\batch_norm_replacement.py`

```
# 添加类型检查允许未定义的函数
# 导入 PyTorch 的神经网络模块 nn
import torch.nn as nn
# 从 torch._functorch.utils 中导入 exposed_in 函数
from torch._functorch.utils import exposed_in

# 定义一个函数 batch_norm_without_running_stats，接收一个 nn.Module 类型的参数 module
def batch_norm_without_running_stats(module: nn.Module):
    # 检查 module 是否是 nn.modules.batchnorm._BatchNorm 类型，并且是否追踪运行统计信息
    if (
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ):
        # 如果是，则将 running_mean、running_var 和 num_batches_tracked 设置为 None
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        # 将 track_running_stats 设置为 False
        module.track_running_stats = False

# 在 "torch.func" 中注册的函数，用来替换所有 nn.BatchNorm 模块的统计信息
@exposed_in("torch.func")
def replace_all_batch_norm_modules_(root: nn.Module) -> nn.Module:
    """
    In place updates :attr:`root` by setting the ``running_mean`` and ``running_var`` to be None and
    setting track_running_stats to be False for any nn.BatchNorm module in :attr:`root`
    """
    # 对根模块调用 batch_norm_without_running_stats 函数，处理根模块自身
    batch_norm_without_running_stats(root)

    # 遍历 root 中的所有模块
    for obj in root.modules():
        # 对每个模块调用 batch_norm_without_running_stats 函数，处理所有子模块
        batch_norm_without_running_stats(obj)
    
    # 返回处理后的根模块
    return root
```