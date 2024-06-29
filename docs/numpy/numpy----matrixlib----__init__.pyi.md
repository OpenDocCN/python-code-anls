# `D:\src\scipysrc\numpy\numpy\matrixlib\__init__.pyi`

```
# 从 numpy._pytesttester 模块中导入 PytestTester 类
from numpy._pytesttester import PytestTester

# 从 numpy 模块中导入 matrix 类并将其命名为 matrix
from numpy import (
    matrix as matrix,
)

# 从 numpy.matrixlib.defmatrix 模块中导入 bmat, mat, asmatrix 三个函数
from numpy.matrixlib.defmatrix import (
    bmat as bmat,
    mat as mat,
    asmatrix as asmatrix,
)

# 定义 __all__ 变量，表明模块中公开的所有符号，其类型为字符串列表
__all__: list[str]

# 定义 test 变量，类型为 PytestTester 类，用于执行 numpy 模块的测试
test: PytestTester
```