# `.\pytorch\functorch\__init__.py`

```
# 引入 torch 库，用于深度学习操作
import torch

# 从 torch._functorch.deprecated 模块中引入一系列功能函数
from torch._functorch.deprecated import (
    combine_state_for_ensemble,   # 用于将状态组合到集合中
    functionalize,                # 将函数转换为函数式模块
    grad,                         # 计算梯度
    grad_and_value,               # 计算梯度和值
    hessian,                      # 计算 Hessian 矩阵
    jacfwd,                       # 前向 Jacobian 向量乘积
    jacrev,                       # 反向 Jacobian 向量乘积
    jvp,                          # Jacobian 向量乘积
    make_functional,              # 创建函数式模块
    make_functional_with_buffers, # 创建带缓冲区的函数式模块
    vjp,                          # 倒向 Jacobian 向量乘积
    vmap,                         # 向量化映射
)

# 从 torch._functorch.make_functional 模块中引入 FunctionalModule 和 FunctionalModuleWithBuffers 类
from torch._functorch.make_functional import (
    FunctionalModule,             # 函数式模块
    FunctionalModuleWithBuffers,  # 带缓冲区的函数式模块
)

# 顶层 API。添加到顶层命名空间之前，请慎重考虑：
# - 私有辅助函数应放在 torch._functorch 中
# - 非常实验性的功能应放在 functorch.experimental 中
# - 与编译相关的内容应放在 functorch.compile 中

# 未曾文档化的功能
from torch._functorch.python_key import make_fx

# 设置 __version__ 变量为当前 torch 版本
__version__ = torch.__version__
```