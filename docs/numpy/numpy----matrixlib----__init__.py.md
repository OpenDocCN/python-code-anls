# `.\numpy\numpy\matrixlib\__init__.py`

```
"""Sub-package containing the matrix class and related functions.

"""
# 从当前包中导入 defmatrix 模块
from . import defmatrix
# 从 defmatrix 模块中导入所有公开的符号
from .defmatrix import *
# 将 defmatrix 模块中所有公开的符号添加到当前模块的 __all__ 列表中
__all__ = defmatrix.__all__

# 从 numpy._pytesttester 模块中导入 PytestTester 类
from numpy._pytesttester import PytestTester
# 创建一个 PytestTester 类的实例 test，并指定其名称为当前模块的名称
test = PytestTester(__name__)
# 删除当前作用域中的 PytestTester 类的引用，避免全局污染
del PytestTester
```