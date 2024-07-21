# `.\pytorch\test\custom_operator\pointwise.py`

```py
# 从 model 模块导入 get_custom_op_library_path 函数
from model import get_custom_op_library_path

# 导入 torch 库
import torch

# 加载自定义操作库，使用 get_custom_op_library_path 函数获取库路径
torch.ops.load_library(get_custom_op_library_path())

# NB: cos 的 impl_abstract_pystub 实际上指定应该位于 my_custom_ops2 模块中。
# 定义 torch 自定义库中 "custom::cos" 的抽象实现函数，接受输入 x 并返回与 x 类型相同的空张量。
@torch.library.impl_abstract("custom::cos")
def cos_abstract(x):
    return torch.empty_like(x)

# NB: tan 没有 impl_abstract_pystub。
# 定义 torch 自定义库中 "custom::tan" 的抽象实现函数，接受输入 x 并返回与 x 类型相同的空张量。
@torch.library.impl_abstract("custom::tan")
def tan_abstract(x):
    return torch.empty_like(x)
```