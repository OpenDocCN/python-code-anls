# `.\pytorch\torch\utils\tensorboard\_convert_np.py`

```py
# mypy: allow-untyped-defs
"""This module converts objects into numpy array."""

# 导入 NumPy 库
import numpy as np

# 导入 PyTorch 库
import torch

# 定义函数，将对象转换为 NumPy 数组
def make_np(x):
    """
    Convert an object into numpy array.

    Args:
      x: An instance of torch tensor

    Returns:
        numpy.array: Numpy array
    """
    # 如果输入已经是 NumPy 数组，则直接返回
    if isinstance(x, np.ndarray):
        return x
    # 如果输入是标量，则转换为包含该标量的 NumPy 数组并返回
    if np.isscalar(x):
        return np.array([x])
    # 如果输入是 PyTorch 张量，则调用内部函数准备 PyTorch 张量并返回其 NumPy 表示
    if isinstance(x, torch.Tensor):
        return _prepare_pytorch(x)
    # 如果输入既不是 NumPy 数组也不是 PyTorch 张量，则抛出未实现的错误
    raise NotImplementedError(
        f"Got {type(x)}, but numpy array or torch tensor are expected."
    )

# 内部函数，准备 PyTorch 张量为 NumPy 数组
def _prepare_pytorch(x):
    # 如果 PyTorch 张量的数据类型是 bfloat16，则转换为 float16
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float16)
    # 将 PyTorch 张量从计算设备上分离，并转换为 CPU 上的 NumPy 数组
    x = x.detach().cpu().numpy()
    return x
```