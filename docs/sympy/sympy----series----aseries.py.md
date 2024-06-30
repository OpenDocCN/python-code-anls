# `D:\src\scipysrc\sympy\sympy\series\aseries.py`

```
# 从 sympy 库的 core 模块中导入 sympify 函数
from sympy.core.sympify import sympify

# 定义名为 aseries 的函数，用于对表达式进行级数展开
def aseries(expr, x=None, n=6, bound=0, hir=False):
    """
    See the docstring of Expr.aseries() for complete details of this wrapper.
    参见 Expr.aseries() 的文档字符串，获取此函数的完整细节说明。

    """
    # 将输入的表达式 expr 转换为 sympy 的表达式对象
    expr = sympify(expr)
    # 调用表达式对象的 aseries 方法进行级数展开，并返回结果
    return expr.aseries(x, n, bound, hir)
```