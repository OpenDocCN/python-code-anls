# `D:\src\scipysrc\scipy\scipy\linalg\__init__.py`

```
"""
====================================
Linear algebra (:mod:`scipy.linalg`)
====================================

.. currentmodule:: scipy.linalg

.. toctree::
   :hidden:

   linalg.blas
   linalg.cython_blas
   linalg.cython_lapack
   linalg.interpolative
   linalg.lapack

Linear algebra functions.

.. eventually, we should replace the numpy.linalg HTML link with just `numpy.linalg`

.. seealso::

   `numpy.linalg <https://www.numpy.org/devdocs/reference/routines.linalg.html>`__
   for more linear algebra functions. Note that
   although `scipy.linalg` imports most of them, identically named
   functions from `scipy.linalg` may offer more or slightly differing
   functionality.


Basics
======

.. autosummary::
   :toctree: generated/

   inv - Find the inverse of a square matrix
   solve - Solve a linear system of equations
   solve_banded - Solve a banded linear system
   solveh_banded - Solve a Hermitian or symmetric banded system
   solve_circulant - Solve a circulant system
   solve_triangular - Solve a triangular matrix
   solve_toeplitz - Solve a toeplitz matrix
   matmul_toeplitz - Multiply a Toeplitz matrix with an array.
   det - Find the determinant of a square matrix
   norm - Matrix and vector norm
   lstsq - Solve a linear least-squares problem
   pinv - Pseudo-inverse (Moore-Penrose) using lstsq
   pinvh - Pseudo-inverse of hermitian matrix
   kron - Kronecker product of two arrays
   khatri_rao - Khatri-Rao product of two arrays
   orthogonal_procrustes - Solve an orthogonal Procrustes problem
   matrix_balance - Balance matrix entries with a similarity transformation
   subspace_angles - Compute the subspace angles between two matrices
   bandwidth - Return the lower and upper bandwidth of an array
   issymmetric - Check if a square 2D array is symmetric
   ishermitian - Check if a square 2D array is Hermitian
   LinAlgError
   LinAlgWarning

Eigenvalue Problems
===================

.. autosummary::
   :toctree: generated/

   eig - Find the eigenvalues and eigenvectors of a square matrix
   eigvals - Find just the eigenvalues of a square matrix
   eigh - Find the e-vals and e-vectors of a Hermitian or symmetric matrix
   eigvalsh - Find just the eigenvalues of a Hermitian or symmetric matrix
   eig_banded - Find the eigenvalues and eigenvectors of a banded matrix
   eigvals_banded - Find just the eigenvalues of a banded matrix
   eigh_tridiagonal - Find the eigenvalues and eigenvectors of a tridiagonal matrix
   eigvalsh_tridiagonal - Find just the eigenvalues of a tridiagonal matrix

Decompositions
==============

"""


注释：

# 这部分是一个文档字符串，用于描述 `scipy.linalg` 模块中线性代数相关的功能和结构
# 定义了当前模块为 `scipy.linalg`
# 导入其他子模块的目录结构
# 提供了 `scipy.linalg` 中线性代数函数的总体概览和链接
# 引用了 `numpy.linalg` 的链接，说明了它与 `scipy.linalg` 的关系


这段代码是一个文档的一部分，通常用于生成库或模块的说明文档。
lu - LU decomposition of a matrix
def lu(a, permute_l=False):
    # Perform LU decomposition of matrix `a`
    # permute_l: Whether to permute L matrix to be sorted
    return _umath_linalg.lu(a, permute_l)



lu_factor - LU decomposition returning unordered matrix and pivots
def lu_factor(a, overwrite_a=False, check_finite=True):
    # Compute LU decomposition of matrix `a` and return the result
    # overwrite_a: Whether to overwrite data in `a` (if possible)
    # check_finite: Whether to check that all elements of `a` are finite numbers
    return _umath_linalg.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)



lu_solve - Solve Ax=b using back substitution with output of lu_factor
def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    # Solve the linear equation system Ax = b using LU decomposition result `lu_and_piv`
    # trans: Type of system to solve
    # overwrite_b: Whether to overwrite data in `b` (if possible)
    # check_finite: Whether to check that all elements of `lu_and_piv` and `b` are finite numbers
    return _umath_linalg.lu_solve(lu_and_piv, b, trans=trans, overwrite_b=overwrite_b, check_finite=check_finite)



svd - Singular value decomposition of a matrix
def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd'):
    # Perform singular value decomposition of matrix `a`
    # full_matrices: Whether to compute full-sized U and V matrices
    # compute_uv: Whether to compute U and V matrices
    # overwrite_a: Whether to overwrite data in `a` (if possible)
    # check_finite: Whether to check that all elements of `a` are finite numbers
    # lapack_driver: Which LAPACK driver to use for computation
    return _umath_linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, overwrite_a=overwrite_a, check_finite=check_finite, lapack_driver=lapack_driver)



svdvals - Singular values of a matrix
def svdvals(a, overwrite_a=False, check_finite=True):
    # Compute singular values of matrix `a`
    # overwrite_a: Whether to overwrite data in `a` (if possible)
    # check_finite: Whether to check that all elements of `a` are finite numbers
    return _umath_linalg.svdvals(a, overwrite_a=overwrite_a, check_finite=check_finite)



diagsvd - Construct matrix of singular values from output of svd
def diagsvd(s, M, N):
    # Construct a matrix from singular values `s` for dimensions `M` by `N`
    return _umath_linalg.diagsvd(s, M, N)



orth - Construct orthonormal basis for the range of A using svd
def orth(A):
    # Construct orthonormal basis for the range of matrix `A`
    return _umath_linalg.orth(A)



null_space - Construct orthonormal basis for the null space of A using svd
def null_space(A, rcond=None):
    # Construct orthonormal basis for the null space of matrix `A`
    # rcond: Threshold for singular values below which values are considered zero
    return _umath_linalg.null_space(A, rcond=rcond)



ldl - LDL.T decomposition of a Hermitian or a symmetric matrix.
def ldl(A, lower=True, overwrite_a=False, check_finite=True):
    # Perform LDL.T decomposition of Hermitian or symmetric matrix `A`
    # lower: Whether to return L and D matrices in the lower triangular form
    # overwrite_a: Whether to overwrite data in `A` (if possible)
    # check_finite: Whether to check that all elements of `A` are finite numbers
    return _umath_linalg.ldl(A, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)



cholesky - Cholesky decomposition of a matrix
def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    # Perform Cholesky decomposition of matrix `a`
    # lower: Whether to return the lower triangular matrix
    # overwrite_a: Whether to overwrite data in `a` (if possible)
    # check_finite: Whether to check that all elements of `a` are finite numbers
    return _umath_linalg.cholesky(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)



cholesky_banded - Cholesky decomp. of a sym. or Hermitian banded matrix
def cholesky_banded(ab, overwrite_ab=False, check_finite=True):
    # Perform Cholesky decomposition of a symmetric or Hermitian banded matrix `ab`
    # overwrite_ab: Whether to overwrite data in `ab` (if possible)
    # check_finite: Whether to check that all elements of `ab` are finite numbers
    return _umath_linalg.cholesky_banded(ab, overwrite_ab=overwrite_ab, check_finite=check_finite)



cho_factor - Cholesky decomposition for use in solving a linear system
def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    # Perform Cholesky decomposition of matrix `a` for use in solving a linear system
    # lower: Whether to return the lower triangular matrix
    # overwrite_a: Whether to overwrite data in `a` (if possible)
    # check_finite: Whether to check that all elements of `a` are finite numbers
    return _umath_linalg.cho_factor(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)



cho_solve - Solve previously factored linear system
def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    # Solve the linear equation system Ax = b using previously factored Cholesky result `c_and_lower`
    # overwrite_b: Whether to overwrite data in `b` (if possible)
    # check_finite: Whether to check that all elements of `c_and_lower` and `b` are finite numbers
    return _umath_linalg.cho_solve(c_and_lower, b, overwrite_b=overwrite_b, check_finite=check_finite)



cho_solve_banded - Solve previously factored banded linear system
def cho_solve_banded(cb_and_lower, b, overwrite_b=False, check_finite=True):
    # Solve the linear equation system Ax = b using previously factored banded Cholesky result `cb_and_lower`
    # overwrite_b: Whether to overwrite data in `b` (if possible)
    # check_finite: Whether to check that all elements of `cb_and_lower` and `b` are finite numbers
    return _umath_linalg.cho_solve_banded(cb_and_lower, b, overwrite_b=overwrite_b, check_finite=check_finite)



polar - Compute the polar decomposition.
def polar(a, side='right', overwrite_a=False, check_finite=True):
    # Compute the polar decomposition of matrix `a`
    # side: Which side to form the polar decomposition ('right' or 'left')
    # overwrite_a: Whether to overwrite data in `a` (if possible)
    # check_finite: Whether to check that all elements of `a` are finite numbers
    return _umath_linalg.polar(a, side=side, overwrite_a=overwrite_a, check_finite=check_finite)



qr - QR decomposition of a matrix
def qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False, check_finite=True):
    # Compute QR decomposition of matrix `a`
    # overwrite_a: Whether to overwrite data in `a` (if possible)
    # lwork: Work array size
    # mode: Type of return ('full', 'r', 'economic', 'raw')
    # pivoting: Whether to perform column pivoting
    # check_finite: Whether to check that all elements of `a` are finite numbers
    return _umath_linalg.qr(a, overwrite_a=overwrite_a, lwork=lwork, mode=mode, pivoting=pivoting, check_finite=check_finite)



qr_multiply - QR decomposition and multiplication by Q
def qr_multiply(a1, a2, mode='left', transpose='notransp', conjugate_transpose='no_transp', overwrite_a=False, lwork=None, check_finite=True):
    # Compute QR decomposition of matrix `a1` and multiply by matrix `a2`
    # mode: Type of multiplication ('left' or 'right')
    # transpose: Type of transpose operation
    # conjugate_transpose: Type of conjugate transpose operation
    # overwrite_a: Whether to overwrite data in `a1` (if possible)
    # lwork: Work array size
    # check_finite: Whether to check that all elements of `a1`
# 导入模块和函数

from ._misc import *  # 导入 _misc 模块中的所有内容
from ._cythonized_array_utils import *  # 导入 _cythonized_array_utils 模块中的所有内容
from ._basic import *  # 导入 _basic 模块中的所有内容
from ._decomp import *  # 导入 _decomp 模块中的所有内容
from ._decomp_lu import *  # 导入 _decomp_lu 模块中的所有内容
from ._decomp_ldl import *  # 导入 _decomp_ldl 模块中的所有内容
from ._decomp_cholesky import *  # 导入 _decomp_cholesky 模块中的所有内容
from ._decomp_qr import *  # 导入 _decomp_qr 模块中的所有内容
from ._decomp_qz import *  # 导入 _decomp_qz 模块中的所有内容
from ._decomp_svd import *  # 导入 _decomp_svd 模块中的所有内容
from ._decomp_schur import *  # 导入 _decomp_schur 模块中的所有内容
from ._decomp_polar import *  # 导入 _decomp_polar 模块中的所有内容
from ._matfuncs import *  # 导入 _matfuncs 模块中的所有内容
from .blas import *  # 导入 blas 模块中的所有内容
from .lapack import *  # 导入 lapack 模块中的所有内容
from ._special_matrices import *  # 导入 _special_matrices 模块中的所有内容
from ._solvers import *  # 导入 _solvers 模块中的所有内容
from ._procrustes import *  # 导入 _procrustes 模块中的所有内容
from ._decomp_update import *  # 导入 _decomp_update 模块中的所有内容
from ._sketches import *  # 导入 _sketches 模块中的所有内容
from ._decomp_cossin import *  # 导入 _decomp_cossin 模块中的所有内容

# Deprecated namespaces, to be removed in v2.0.0
from . import (
    decomp, decomp_cholesky, decomp_lu, decomp_qr, decomp_svd, decomp_schur,
    basic, misc, special_matrices, matfuncs,
)  # 导入被标记为即将在 v2.0.0 版本中移除的一组模块和子模块

__all__ = [s for s in dir() if not s.startswith('_')]  # 设置导出所有非私有符号

from scipy._lib._testutils import PytestTester  # 导入 PytestTester 类
test = PytestTester(__name__)  # 创建 PytestTester 的实例 test，传入当前模块的名称作为参数
del PytestTester  # 删除 PytestTester 类的引用，以确保不会被模块外部访问
```