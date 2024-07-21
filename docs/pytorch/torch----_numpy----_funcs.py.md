# `.\pytorch\torch\_numpy\_funcs.py`

```
# Ignore type errors from mypy type checker
# 忽略 mypy 类型检查器的类型错误

import inspect  # 导入 inspect 模块，用于获取对象信息
import itertools  # 导入 itertools 模块，用于创建迭代器的函数

from . import _funcs_impl, _reductions_impl  # 从当前包中导入 _funcs_impl 和 _reductions_impl 模块
from ._normalizations import normalizer  # 从当前包中导入 normalizer 函数

# _funcs_impl.py 包含了模仿 NumPy 同名函数的实现，
# 并且接受/返回 PyTorch 张量和数据类型。
# 这些函数也被类型注释。
# 从 _funcs_impl 中导入这些函数，并使用 @normalizer 装饰它们，该装饰器：
# - 将任何输入 `np.ndarray`, `torch._numpy.ndarray`, 列表的列表，Python 标量等转换为 `torch.Tensor`。
# - 将 NumPy 数据类型映射到 PyTorch 数据类型
# - 如果 `axis` 关键字参数的输入是 ndarray，则将其映射为元组
# - 实现了 `out=` 参数的语义
# - 将输出再次包装成 `torch._numpy.ndarrays`

def _public_functions(mod):
    def is_public_function(f):
        return inspect.isfunction(f) and not f.__name__.startswith("_")
    
    return inspect.getmembers(mod, is_public_function)

# 在下面的循环中填充 __all__
__all__ = []

# 使用参数正常化装饰实现者函数，并导出到顶级命名空间
for name, func in itertools.chain(
    _public_functions(_funcs_impl), _public_functions(_reductions_impl)
):
    if name in ["percentile", "quantile", "median"]:
        decorated = normalizer(func, promote_scalar_result=True)
    elif name == "einsum":
        # 手动进行规范化
        decorated = func
    else:
        decorated = normalizer(func)
    
    decorated.__qualname__ = name  # 设置装饰函数的限定名称
    decorated.__name__ = name  # 设置装饰函数的名称
    vars()[name] = decorated  # 将装饰后的函数赋值给对应的名称
    __all__.append(name)  # 将函数名称添加到 __all__ 列表中

"""
Vendored objects from numpy.lib.index_tricks
"""

class IndexExpression:
    """
    Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
    last revision: 1999-7-23

    Cosmetic changes by T. Oliphant 2001
    """
    
    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item

index_exp = IndexExpression(maketuple=True)  # 创建 IndexExpression 类的实例 index_exp，使其创建元组
s_ = IndexExpression(maketuple=False)  # 创建 IndexExpression 类的实例 s_，不要创建元组

__all__ += ["index_exp", "s_"]  # 将 "index_exp" 和 "s_" 添加到 __all__ 列表中
```