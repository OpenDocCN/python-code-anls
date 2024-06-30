# `D:\src\scipysrc\sympy\sympy\concrete\__init__.py`

```
# 导入当前目录下的products模块中的product和Product类
from .products import product, Product
# 导入当前目录下的summations模块中的summation和Sum类
from .summations import summation, Sum
# 定义一个列表，指定了当前模块中可以导出的对象的名称
__all__ = [
    'product', 'Product',  # 导出product和Product
    'summation', 'Sum',    # 导出summation和Sum
]
```