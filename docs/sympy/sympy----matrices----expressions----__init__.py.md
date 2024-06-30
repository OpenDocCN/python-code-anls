# `D:\src\scipysrc\sympy\sympy\matrices\expressions\__init__.py`

```
# 一个处理矩阵表达式的模块

# 导入矩阵切片相关的模块和类
from .slice import MatrixSlice
# 导入块矩阵相关的模块和类，以及块矩阵的处理函数
from .blockmatrix import BlockMatrix, BlockDiagMatrix, block_collapse, blockcut
# 导入伴随矩阵相关的模块和类
from .companion import CompanionMatrix
# 导入函数矩阵相关的模块和类
from .funcmatrix import FunctionMatrix
# 导入逆矩阵相关的模块和类
from .inverse import Inverse
# 导入矩阵加法相关的模块和类
from .matadd import MatAdd
# 导入矩阵表达式相关的模块和类，包括矩阵符号和矩阵表达式
from .matexpr import MatrixExpr, MatrixSymbol, matrix_symbols
# 导入矩阵乘法相关的模块和类
from .matmul import MatMul
# 导入矩阵幂相关的模块和类
from .matpow import MatPow
# 导入矩阵迹相关的模块和类，以及迹函数
from .trace import Trace, trace
# 导入行列式相关的模块和类，以及行列式函数和永久函数
from .determinant import Determinant, det, Permanent, per
# 导入矩阵转置相关的模块和类
from .transpose import Transpose
# 导入伴随矩阵相关的模块和类
from .adjoint import Adjoint
# 导入Hadamard乘积相关的模块和类，包括Hadamard乘积函数和Hadamard幂函数
from .hadamard import hadamard_product, HadamardProduct, hadamard_power, HadamardPower
# 导入对角矩阵相关的模块和类，以及对角矩阵函数和向量对角化函数
from .diagonal import DiagonalMatrix, DiagonalOf, DiagMatrix, diagonalize_vector
# 导入点积相关的模块和类
from .dotproduct import DotProduct
# 导入Kronecker积相关的模块和类，以及Kronecker积的组合函数
from .kronecker import kronecker_product, KroneckerProduct, combine_kronecker
# 导入置换矩阵相关的模块和类，以及矩阵置换函数
from .permutation import PermutationMatrix, MatrixPermute
# 导入矩阵集合相关的模块和类
from .sets import MatrixSet
# 导入特殊矩阵相关的模块和类，包括零矩阵、单位矩阵和全1矩阵
from .special import ZeroMatrix, Identity, OneMatrix

# 所有需要导出的符号和类名列表
__all__ = [
    'MatrixSlice',  # 矩阵切片类

    'BlockMatrix', 'BlockDiagMatrix', 'block_collapse', 'blockcut',  # 块矩阵及其处理函数

    'FunctionMatrix',  # 函数矩阵类

    'CompanionMatrix',  # 伴随矩阵类

    'Inverse',  # 逆矩阵类

    'MatAdd',  # 矩阵加法类

    'Identity', 'MatrixExpr', 'MatrixSymbol', 'ZeroMatrix', 'OneMatrix',  # 标识、矩阵表达式和特殊矩阵类

    'matrix_symbols', 'MatrixSet',  # 矩阵符号集合类

    'MatMul',  # 矩阵乘法类

    'MatPow',  # 矩阵幂类

    'Trace', 'trace',  # 矩阵迹类和迹函数

    'Determinant', 'det',  # 行列式类和行列式函数

    'Transpose',  # 矩阵转置类

    'Adjoint',  # 伴随矩阵类

    'hadamard_product', 'HadamardProduct', 'hadamard_power', 'HadamardPower',  # Hadamard乘积和幂类及其函数

    'DiagonalMatrix', 'DiagonalOf', 'DiagMatrix', 'diagonalize_vector',  # 对角矩阵及其相关函数

    'DotProduct',  # 点积类

    'kronecker_product', 'KroneckerProduct', 'combine_kronecker',  # Kronecker积及其组合函数

    'PermutationMatrix', 'MatrixPermute',  # 置换矩阵及其置换函数

    'Permanent', 'per'  # 永久函数类和永久函数
]
```