# `.\pytorch\torch\_library\__init__.py`

```py
# 导入PyTorch的自动求导库的C扩展模块
import torch._library.autograd
# 导入PyTorch的伪实现库的C扩展模块
import torch._library.fake_impl
# 导入PyTorch的简单注册表库的C扩展模块
import torch._library.simple_registry
# 导入PyTorch的工具库的C扩展模块
import torch._library.utils

# 从伪类注册表库中导入注册伪类的函数
from torch._library.fake_class_registry import register_fake_class
```