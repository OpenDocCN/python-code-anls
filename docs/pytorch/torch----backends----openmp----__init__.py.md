# `.\pytorch\torch\backends\openmp\__init__.py`

```py
# 引入名为 `torch` 的模块，用于与 PyTorch 进行交互
import torch

# 定义函数 `is_available()`，用于检查 PyTorch 是否支持 OpenMP
def is_available():
    r"""Return whether PyTorch is built with OpenMP support."""
    # 返回一个布尔值，指示 PyTorch 是否具备 OpenMP 支持的能力
    return torch._C.has_openmp
```