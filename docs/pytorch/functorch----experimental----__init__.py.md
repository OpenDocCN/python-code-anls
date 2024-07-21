# `.\pytorch\functorch\experimental\__init__.py`

```py
# 导入functorch库中的functionalize函数，用于将PyTorch模块转换为可自动求导的函数
# 导入chunk_vmap函数，用于自动化地对PyTorch函数进行vmap操作
# 导入replace_all_batch_norm_modules_函数，用于替换模型中所有的批归一化模块
# 导入hessian、jacfwd和jvp函数，用于计算PyTorch模型的Hessian矩阵、正向Jacobian和Jacobian-向量积
from functorch import functionalize
from torch._functorch.apis import chunk_vmap
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import hessian, jacfwd, jvp
```