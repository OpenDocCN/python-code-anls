# `.\numpy\numpy\linalg\__init__.pyi`

```py
# 从 numpy.linalg._linalg 导入多个函数，重命名为相应的本地名称
from numpy.linalg._linalg import (
    matrix_power as matrix_power,  # 导入 matrix_power 函数并重命名为 matrix_power
    solve as solve,                # 导入 solve 函数并重命名为 solve
    tensorsolve as tensorsolve,    # 导入 tensorsolve 函数并重命名为 tensorsolve
    tensorinv as tensorinv,        # 导入 tensorinv 函数并重命名为 tensorinv
    inv as inv,                    # 导入 inv 函数并重命名为 inv
    cholesky as cholesky,          # 导入 cholesky 函数并重命名为 cholesky
    outer as outer,                # 导入 outer 函数并重命名为 outer
    eigvals as eigvals,            # 导入 eigvals 函数并重命名为 eigvals
    eigvalsh as eigvalsh,          # 导入 eigvalsh 函数并重命名为 eigvalsh
    pinv as pinv,                  # 导入 pinv 函数并重命名为 pinv
    slogdet as slogdet,            # 导入 slogdet 函数并重命名为 slogdet
    det as det,                    # 导入 det 函数并重命名为 det
    svd as svd,                    # 导入 svd 函数并重命名为 svd
    svdvals as svdvals,            # 导入 svdvals 函数并重命名为 svdvals
    eig as eig,                    # 导入 eig 函数并重命名为 eig
    eigh as eigh,                  # 导入 eigh 函数并重命名为 eigh
    lstsq as lstsq,                # 导入 lstsq 函数并重命名为 lstsq
    norm as norm,                  # 导入 norm 函数并重命名为 norm
    matrix_norm as matrix_norm,    # 导入 matrix_norm 函数并重命名为 matrix_norm
    vector_norm as vector_norm,    # 导入 vector_norm 函数并重命名为 vector_norm
    qr as qr,                      # 导入 qr 函数并重命名为 qr
    cond as cond,                  # 导入 cond 函数并重命名为 cond
    matrix_rank as matrix_rank,    # 导入 matrix_rank 函数并重命名为 matrix_rank
    multi_dot as multi_dot,        # 导入 multi_dot 函数并重命名为 multi_dot
    matmul as matmul,              # 导入 matmul 函数并重命名为 matmul
    trace as trace,                # 导入 trace 函数并重命名为 trace
    diagonal as diagonal,          # 导入 diagonal 函数并重命名为 diagonal
    cross as cross,                # 导入 cross 函数并重命名为 cross
)

# 从 numpy._core.fromnumeric 模块导入 matrix_transpose 函数并重命名为 matrix_transpose
from numpy._core.fromnumeric import (
    matrix_transpose as matrix_transpose
)

# 从 numpy._core.numeric 模块导入 tensordot 函数并重命名为 tensordot，vecdot 函数并重命名为 vecdot
from numpy._core.numeric import (
    tensordot as tensordot,
    vecdot as vecdot
)

# 从 numpy._pytesttester 模块导入 PytestTester 类
from numpy._pytesttester import PytestTester

# 定义 __all__ 变量，指示模块中应导出的所有公共名称列表为 list[str] 类型
__all__: list[str]

# 声明 test 变量为 PytestTester 类的一个实例
test: PytestTester

# 定义 LinAlgError 异常类，继承自内置 Exception 类
class LinAlgError(Exception): ...
```