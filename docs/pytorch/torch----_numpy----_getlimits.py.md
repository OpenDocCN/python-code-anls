# `.\pytorch\torch\_numpy\_getlimits.py`

```py
# 忽略 mypy 对类型错误的检查

# 导入 PyTorch 库
import torch

# 从当前目录中导入 _dtypes 模块
from . import _dtypes


# 定义函数 finfo，返回指定数据类型的浮点数信息
def finfo(dtyp):
    # 调用 _dtypes 模块的 dtype 函数获取对应的 PyTorch 数据类型
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    # 返回该 PyTorch 数据类型的浮点数信息
    return torch.finfo(torch_dtype)


# 定义函数 iinfo，返回指定数据类型的整数信息
def iinfo(dtyp):
    # 调用 _dtypes 模块的 dtype 函数获取对应的 PyTorch 数据类型
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    # 返回该 PyTorch 数据类型的整数信息
    return torch.iinfo(torch_dtype)
```