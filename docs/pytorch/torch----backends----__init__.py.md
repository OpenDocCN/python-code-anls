# `.\pytorch\torch\backends\__init__.py`

```py
# 引入类型模块
import types
# 引入上下文管理器
from contextlib import contextmanager

# 在测试套件中禁止直接对 torch.backends.<cudnn|mkldnn>.enabled 等进行赋值的标志
# 避免在测试过程中忘记撤销更改的想法。
__allow_nonbracketed_mutation_flag = True

# 禁用全局标志的函数
def disable_global_flags():
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False

# 检查标志是否被冻结的函数
def flags_frozen():
    return not __allow_nonbracketed_mutation_flag

# 上下文管理器，允许非括号形式的突变
@contextmanager
def __allow_nonbracketed_mutation():
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old

# 上下文属性类，用于描述属性的获取和设置
class ContextProp:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    # 获取属性值
    def __get__(self, obj, objtype):
        return self.getter()

    # 设置属性值，如果标志未冻结则允许设置
    def __set__(self, obj, val):
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError(
                f"not allowed to set {obj.__name__} flags "
                "after disable_global_flags; please use flags() context manager instead"
            )

# 属性模块类，继承自 types.ModuleType
class PropModule(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    # 获取属性值
    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

# 从 torch.backends 导入各种模块
from torch.backends import (
    cpu as cpu,
    cuda as cuda,
    cudnn as cudnn,
    mha as mha,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    nnpack as nnpack,
    openmp as openmp,
    quantized as quantized,
)
```