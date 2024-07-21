# `.\pytorch\torch\jit\_await.py`

```
# 设置允许未类型化的函数定义，用于类型检查
# 导入PyTorch库
import torch
# 从torch._jit_internal模块导入_Await类
from torch._jit_internal import _Await
# 从torch.jit._builtins模块导入_register_builtin函数
from torch.jit._builtins import _register_builtin
# 从torch.utils模块导入set_module函数
from torch.utils import set_module

# 设置_Await类的模块路径为torch.jit
set_module(_Await, "torch.jit")

# 定义_awaitable函数，创建一个Await对象，用于在请求结果时调用指定的函数及参数
def _awaitable(func, *args, **kwargs):
    r"""Create Await object that will call specified functioni with specified args, when it is requested for the result."""
    # 调用torch._C._awaitable函数，返回一个Await对象
    return torch._C._awaitable(func, *args, **kwargs)

# 定义_awaitable_wait函数，请求等待执行结果，如果Await对象尚未完成，将立即调用func函数
def _awaitable_wait(aw):
    r"""Request await the result of execution, if Await is not completed yet, the func will be called immediately."""
    # 调用torch._C._awaitable_wait函数，等待Await对象的结果
    return torch._C._awaitable_wait(aw)

# 定义_awaitable_nowait函数，创建一个已完成的Await对象，使用指定的结果
def _awaitable_nowait(o):
    r"""Create completed Await with specified result."""
    # 调用torch._C._awaitable_nowait函数，创建一个已完成的Await对象
    return torch._C._awaitable_nowait(o)

# 将_awaitable_wait函数注册为prim::awaitable_wait内置函数
_register_builtin(_awaitable_wait, "prim::awaitable_wait")
# 将_awaitable_nowait函数注册为prim::awaitable_nowait内置函数
_register_builtin(_awaitable_nowait, "prim::awaitable_nowait")
```