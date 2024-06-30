# `D:\src\scipysrc\sympy\sympy\logic\__init__.py`

```
# 从布尔代数相关模块中导入一系列函数和变量
from .boolalg import (to_cnf, to_dnf, to_nnf, And, Or, Not, Xor, Nand, Nor, Implies,
    Equivalent, ITE, POSform, SOPform, simplify_logic, bool_map, true, false,
    gateinputcount)
# 从推理模块中导入 satisfiable 函数
from .inference import satisfiable

# 定义 __all__ 变量，指定了在使用 `from package import *` 时应该导入的符号
__all__ = [
    'to_cnf', 'to_dnf', 'to_nnf', 'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
    'Implies', 'Equivalent', 'ITE', 'POSform', 'SOPform', 'simplify_logic',
    'bool_map', 'true', 'false', 'gateinputcount',

    'satisfiable',
]
```