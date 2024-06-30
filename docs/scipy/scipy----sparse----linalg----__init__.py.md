# `D:\src\scipysrc\scipy\scipy\sparse\linalg\__init__.py`

```
"""
Sparse linear algebra (:mod:`scipy.sparse.linalg`)
==================================================

.. currentmodule:: scipy.sparse.linalg

Abstract linear operators
-------------------------

.. autosummary::
   :toctree: generated/

   LinearOperator -- abstract representation of a linear operator
   aslinearoperator -- convert an object to an abstract linear operator

Matrix Operations
-----------------

.. autosummary::
   :toctree: generated/

   inv -- compute the sparse matrix inverse
   expm -- compute the sparse matrix exponential
   expm_multiply -- compute the product of a matrix exponential and a matrix
   matrix_power -- compute the matrix power by raising a matrix to an exponent

Matrix norms
------------

.. autosummary::
   :toctree: generated/

   norm -- Norm of a sparse matrix
   onenormest -- Estimate the 1-norm of a sparse matrix

Solving linear problems
-----------------------

Direct methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   spsolve -- Solve the sparse linear system Ax=b
   spsolve_triangular -- Solve sparse linear system Ax=b for a triangular A.
   factorized -- Pre-factorize matrix to a function solving a linear system
   MatrixRankWarning -- Warning on exactly singular matrices
   use_solver -- Select direct solver to use

Iterative methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   bicg -- Use BIConjugate Gradient iteration to solve Ax = b
   bicgstab -- Use BIConjugate Gradient STABilized iteration to solve Ax = b
   cg -- Use Conjugate Gradient iteration to solve Ax = b
   cgs -- Use Conjugate Gradient Squared iteration to solve Ax = b
   gmres -- Use Generalized Minimal RESidual iteration to solve Ax = b
   lgmres -- Solve a matrix equation using the LGMRES algorithm
   minres -- Use MINimum RESidual iteration to solve Ax = b
   qmr -- Use Quasi-Minimal Residual iteration to solve Ax = b
   gcrotmk -- Solve a matrix equation using the GCROT(m,k) algorithm
   tfqmr -- Use Transpose-Free Quasi-Minimal Residual iteration to solve Ax = b

Iterative methods for least-squares problems:

.. autosummary::
   :toctree: generated/

   lsqr -- Find the least-squares solution to a sparse linear equation system
   lsmr -- Find the least-squares solution to a sparse linear equation system

Matrix factorizations
---------------------

Eigenvalue problems:

.. autosummary::
   :toctree: generated/

   eigs -- Find k eigenvalues and eigenvectors of the square matrix A
   eigsh -- Find k eigenvalues and eigenvectors of a symmetric matrix
   lobpcg -- Solve symmetric partial eigenproblems with optional preconditioning

Singular values problems:

.. autosummary::
   :toctree: generated/

   svds -- Compute k singular values/vectors for a sparse matrix

The `svds` function supports the following solvers:

.. toctree::

    sparse.linalg.svds-arpack
    sparse.linalg.svds-lobpcg
    sparse.linalg.svds-propack

Complete or incomplete LU factorizations
"""
# 导入解决稀疏线性系统的函数和类
from ._isolve import *
# 导入解决稀疏矩阵特征值和特征向量的函数和类
from ._eigen import *
# 导入解决稀疏矩阵函数的函数和类
from ._matfuncs import *
# 导入求解稀疏矩阵1-范数的函数
from ._onenormest import *
# 导入稀疏矩阵的范数函数
from ._norm import *
# 导入稀疏矩阵指数函数的实现
from ._expm_multiply import *
# 导入特殊稀疏数组的函数
from ._special_sparse_arrays import *

# Deprecated namespaces, to be removed in v2.0.0
# 导入被弃用的命名空间，将在 v2.0.0 版本中移除
from . import isolve, dsolve, interface, eigen, matfuncs

# 将所有不以下划线开头的对象添加到 __all__ 列表中
__all__ = [s for s in dir() if not s.startswith('_')]

# 导入用于运行测试的 PytestTester 类
from scipy._lib._testutils import PytestTester
# 创建与当前模块相关的 PytestTester 实例 test
test = PytestTester(__name__)
# 删除 PytestTester 类的引用，确保它不会在模块中被访问
del PytestTester
```