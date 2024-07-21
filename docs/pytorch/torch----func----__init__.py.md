# `.\pytorch\torch\func\__init__.py`

```
# 从torch._functorch.eager_transforms模块中导入以下函数和类
from torch._functorch.eager_transforms import (
    vjp,          # 导入反向变换的雅可比向量积函数
    jvp,          # 导入前向变换的雅可比向量积函数
    jacrev,       # 导入自动求导的反向雅可比矩阵函数
    jacfwd,       # 导入自动求导的前向雅可比矩阵函数
    hessian,      # 导入自动求导的黑塞矩阵函数
    functionalize,  # 导入函数化装饰器，用于函数转换
    linearize      # 导入线性化装饰器，用于模块线性化
)

# 从torch._functorch.apis模块中导入以下函数和类
from torch._functorch.apis import (
    grad,            # 导入自动求导的梯度计算函数
    grad_and_value   # 导入自动求导的梯度和值计算函数
)

# 从torch._functorch.functional_call模块中导入以下函数和类
from torch._functorch.functional_call import (
    functional_call,    # 导入函数调用功能，用于模块和函数的调用
    stack_module_state  # 导入堆叠模块状态的函数
)

# 从torch._functorch.batch_norm_replacement模块中导入替换所有批标准化模块的函数
from torch._functorch.batch_norm_replacement import (
    replace_all_batch_norm_modules_
)

# 从torch._functorch.apis模块中导入向量映射函数
from torch._functorch.apis import (
    vmap  # 导入向量映射函数，用于批处理向量化
)
```