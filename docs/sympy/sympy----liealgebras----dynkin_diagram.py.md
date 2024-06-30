# `D:\src\scipysrc\sympy\sympy\liealgebras\dynkin_diagram.py`

```
# 导入 CartanType 类，该类用于处理李代数的 Cartan 类型
from .cartan_type import CartanType

# 定义 DynkinDiagram 函数，显示给定李代数的 Dynkin 图
def DynkinDiagram(t):
    """Display the Dynkin diagram of a given Lie algebra
    
    Works by generating the CartanType for the input, t, and then returning the
    Dynkin diagram method from the individual classes.
    
    Examples
    ========
    
    >>> from sympy.liealgebras.dynkin_diagram import DynkinDiagram
    >>> print(DynkinDiagram("A3"))
    0---0---0
    1   2   3
    
    >>> print(DynkinDiagram("B4"))
    0---0---0=>=0
    1   2   3   4
    
    """
    # 使用输入 t 创建 CartanType 对象，并调用其 dynkin_diagram 方法返回 Dynkin 图形式
    return CartanType(t).dynkin_diagram()
```