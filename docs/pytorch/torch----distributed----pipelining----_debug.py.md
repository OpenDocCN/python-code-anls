# `.\pytorch\torch\distributed\pipelining\_debug.py`

```py
# 添加类型提示允许未定义的函数
# 版权所有 Meta Platforms, Inc. 及其关联公司
# 导入 PyTorch 库
import torch


def friendly_debug_info(v):
    """
    辅助函数，友好地打印调试信息。
    
    Args:
        v: 输入的变量
    
    Returns:
        如果 v 是 torch.Tensor 类型，则返回包含形状、梯度和数据类型的字符串表示；
        否则返回 v 的字符串表示。
    """
    if isinstance(v, torch.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad}, dtype={v.dtype})"
    else:
        return str(v)


def map_debug_info(a):
    """
    辅助函数，将 `friendly_debug_info` 应用于 `a` 中的每个项。
    `a` 可能是列表、元组或字典。
    
    Args:
        a: 输入的列表、元组或字典
    
    Returns:
        将 `friendly_debug_info` 应用于 `a` 中每个项的结果。
    """
    return torch.fx.node.map_aggregate(a, friendly_debug_info)
```