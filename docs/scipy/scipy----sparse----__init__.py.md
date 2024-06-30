# `D:\src\scipysrc\scipy\scipy\sparse\__init__.py`

```
"""
=====================================
Sparse matrices (:mod:`scipy.sparse`)
=====================================

.. currentmodule:: scipy.sparse

.. toctree::
   :hidden:

   sparse.csgraph
   sparse.linalg

SciPy 2-D sparse array package for numeric data.

.. note::

   This package is switching to an array interface, compatible with
   NumPy arrays, from the older matrix interface.  We recommend that
   you use the array objects (`bsr_array`, `coo_array`, etc.) for
   all new work.

   When using the array interface, please note that:

   - ``x * y`` no longer performs matrix multiplication, but
     element-wise multiplication (just like with NumPy arrays).  To
     make code work with both arrays and matrices, use ``x @ y`` for
     matrix multiplication.
   - Operations such as `sum`, that used to produce dense matrices, now
     produce arrays, whose multiplication behavior differs similarly.
   - Sparse arrays currently must be two-dimensional.  This also means
     that all *slicing* operations on these objects must produce
     two-dimensional results, or they will result in an error. This
     will be addressed in a future version.

   The construction utilities (`eye`, `kron`, `random`, `diags`, etc.)
   have not yet been ported, but their results can be wrapped into arrays::

     A = csr_array(eye(3))

Contents
========

Sparse array classes
--------------------

.. autosummary::
   :toctree: generated/

   bsr_array - Block Sparse Row array
   coo_array - A sparse array in COOrdinate format
   csc_array - Compressed Sparse Column array
   csr_array - Compressed Sparse Row array
   dia_array - Sparse array with DIAgonal storage
   dok_array - Dictionary Of Keys based sparse array
   lil_array - Row-based list of lists sparse array
   sparray - Sparse array base class

Sparse matrix classes
---------------------

.. autosummary::
   :toctree: generated/

   bsr_matrix - Block Sparse Row matrix
   coo_matrix - A sparse matrix in COOrdinate format
   csc_matrix - Compressed Sparse Column matrix
   csr_matrix - Compressed Sparse Row matrix
   dia_matrix - Sparse matrix with DIAgonal storage
   dok_matrix - Dictionary Of Keys based sparse matrix
   lil_matrix - Row-based list of lists sparse matrix
   spmatrix - Sparse matrix base class

Functions
---------

Building sparse arrays:

.. autosummary::
   :toctree: generated/

   diags_array - Return a sparse array from diagonals
   eye_array - Sparse MxN array whose k-th diagonal is all ones
   random_array - Random values in a given shape array
   block_array - Build a sparse array from sub-blocks

Building sparse matrices:
"""
# 生成一个稀疏的 MxN 矩阵，其第 k 条对角线全为 1
eye

# 生成一个稀疏的单位矩阵
identity

# 从给定的对角线返回一个稀疏矩阵
diags

# 从给定的对角线返回一个稀疏矩阵
spdiags

# 从稀疏子块构建一个稀疏矩阵
bmat

# 在给定形状的矩阵中生成随机值
random

# 在给定形状的矩阵中生成随机值（旧接口）
rand

# 计算两个稀疏矩阵的 Kronecker 乘积
kron

# 计算两个稀疏矩阵的 Kronecker 和
kronsum

# 构建一个块对角稀疏矩阵
block_diag

# 返回稀疏格式中矩阵的下三角部分
tril

# 返回稀疏格式中矩阵的上三角部分
triu

# 水平堆叠稀疏矩阵（按列）
hstack

# 垂直堆叠稀疏矩阵（按行）
vstack

# 保存稀疏矩阵或数组到以 `.npz` 格式的文件中
save_npz

# 从以 `.npz` 格式保存的文件中加载稀疏矩阵或数组
load_npz

# 在稀疏数组中查找非零元素的索引
find

# 检查一个对象是否是稀疏数组
issparse

# 检查一个对象是否是 CSR 格式的稀疏矩阵
isspmatrix_csr

# 检查一个对象是否是 CSC 格式的稀疏矩阵
isspmatrix_csc

# 检查一个对象是否是 BSR 格式的稀疏矩阵
isspmatrix_bsr

# 检查一个对象是否是 LIL 格式的稀疏矩阵
isspmatrix_lil

# 检查一个对象是否是 DOK 格式的稀疏矩阵
isspmatrix_dok

# 检查一个对象是否是 COO 格式的稀疏矩阵
isspmatrix_coo

# 检查一个对象是否是 DIA 格式的稀疏矩阵
isspmatrix_dia

# 稀疏图算法相关的压缩稀疏图例程
csgraph

# 稀疏线性代数相关的稀疏线性代数例程
linalg

# 稀疏效率警告的异常类
SparseEfficiencyWarning

# 稀疏警告的异常类
SparseWarning
# 导入警告模块，并重命名为 _warnings
import warnings as _warnings

# 导入稀疏矩阵相关的基础功能和各种格式的实现模块
from ._base import *
from ._csr import *
from ._csc import *
from ._lil import *
from ._dok import *
from ._coo import *
from ._dia import *
from ._bsr import *
from ._construct import *
from ._extract import *

# 原始代码由Travis Oliphant编写，由Ed Schofield、Robert Cimrman、
# Nathan Bell和Jake Vanderplas修改和扩展。
# 从._matrix模块中导入spmatrix类
from ._matrix import spmatrix
# 从._matrix_io模块中导入所有内容
from ._matrix_io import *

# 为了向后兼容v0.19版本
# 从csgraph模块导入所有内容
from . import csgraph

# 已弃用的命名空间，将在v2.0.0中移除
# 从以下模块导入所有内容：base, bsr, compressed, construct, coo, csc, csr, data, dia, dok, extract, lil, sparsetools, sputils
from . import (
    base, bsr, compressed, construct, coo, csc, csr, data, dia, dok, extract,
    lil, sparsetools, sputils
)

# 设置__all__列表，包含当前模块中所有非下划线开头的符号名称
__all__ = [s for s in dir() if not s.startswith('_')]

# 过滤掉关于numpy 1.15引入的np.matrix的PendingDeprecationWarning警告信息
msg = 'the matrix subclass is not the recommended way'
_warnings.filterwarnings('ignore', message=msg)

# 从scipy._lib._testutils模块导入PytestTester类，并为当前模块创建一个测试对象
test = PytestTester(__name__)
# 删除当前作用域中的PytestTester类引用，以避免对该类的进一步使用
del PytestTester
```