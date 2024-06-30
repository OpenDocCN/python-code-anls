# `D:\src\scipysrc\sympy\sympy\matrices\__init__.py`

```
"""A module that handles matrices.

Includes functions for fast creating matrices like zero, one/eye, random
matrix, etc.
"""
# 导入异常类
from .exceptions import ShapeError, NonSquareMatrixError
# 导入矩阵种类类
from .kind import MatrixKind
# 导入密集矩阵相关函数
from .dense import (
    GramSchmidt, casoratian, diag, eye, hessian, jordan_cell,
    list2numpy, matrix2numpy, matrix_multiply_elementwise, ones,
    randMatrix, rot_axis1, rot_axis2, rot_axis3, rot_ccw_axis1,
    rot_ccw_axis2, rot_ccw_axis3, rot_givens,
    symarray, wronskian, zeros)
# 导入可变密集矩阵类
from .dense import MutableDenseMatrix
# 导入矩阵基类
from .matrixbase import DeferredVector, MatrixBase

# MutableMatrix别名指向MutableDenseMatrix
MutableMatrix = MutableDenseMatrix
# Matrix别名指向MutableMatrix
Matrix = MutableMatrix

# 导入可变稀疏矩阵类
from .sparse import MutableSparseMatrix
# 导入稀疏工具函数
from .sparsetools import banded
# 导入不可变密集矩阵类和不可变稀疏矩阵类
from .immutable import ImmutableDenseMatrix, ImmutableSparseMatrix

# ImmutableMatrix别名指向ImmutableDenseMatrix
ImmutableMatrix = ImmutableDenseMatrix
# SparseMatrix别名指向MutableSparseMatrix
SparseMatrix = MutableSparseMatrix

# 导入表达式类
from .expressions import (
    MatrixSlice, BlockDiagMatrix, BlockMatrix, FunctionMatrix, Identity,
    Inverse, MatAdd, MatMul, MatPow, MatrixExpr, MatrixSymbol, Trace,
    Transpose, ZeroMatrix, OneMatrix, blockcut, block_collapse, matrix_symbols, Adjoint,
    hadamard_product, HadamardProduct, HadamardPower, Determinant, det,
    diagonalize_vector, DiagMatrix, DiagonalMatrix, DiagonalOf, trace,
    DotProduct, kronecker_product, KroneckerProduct,
    PermutationMatrix, MatrixPermute, MatrixSet, Permanent, per)

# 导入矩阵运算工具函数
from .utilities import dotprodsimp

# __all__列表包含了模块的公共接口，便于使用者了解和导入
__all__ = [
    'ShapeError', 'NonSquareMatrixError', 'MatrixKind',

    'GramSchmidt', 'casoratian', 'diag', 'eye', 'hessian', 'jordan_cell',
    'list2numpy', 'matrix2numpy', 'matrix_multiply_elementwise', 'ones',
    'randMatrix', 'rot_axis1', 'rot_axis2', 'rot_axis3', 'symarray',
    'wronskian', 'zeros', 'rot_ccw_axis1', 'rot_ccw_axis2', 'rot_ccw_axis3',
    'rot_givens',

    'MutableDenseMatrix',

    'DeferredVector', 'MatrixBase',

    'Matrix', 'MutableMatrix',

    'MutableSparseMatrix',

    'banded',

    'ImmutableDenseMatrix', 'ImmutableSparseMatrix',

    'ImmutableMatrix', 'SparseMatrix',

    'MatrixSlice', 'BlockDiagMatrix', 'BlockMatrix', 'FunctionMatrix',
    'Identity', 'Inverse', 'MatAdd', 'MatMul', 'MatPow', 'MatrixExpr',
    'MatrixSymbol', 'Trace', 'Transpose', 'ZeroMatrix', 'OneMatrix',
    'blockcut', 'block_collapse', 'matrix_symbols', 'Adjoint',
    'hadamard_product', 'HadamardProduct', 'HadamardPower', 'Determinant',
    'det', 'diagonalize_vector', 'DiagMatrix', 'DiagonalMatrix',
    'DiagonalOf', 'trace', 'DotProduct', 'kronecker_product',
    'KroneckerProduct', 'PermutationMatrix', 'MatrixPermute', 'MatrixSet',
    'Permanent', 'per',

    'dotprodsimp',
]
```