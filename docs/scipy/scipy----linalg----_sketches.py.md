# `D:\src\scipysrc\scipy\scipy\linalg\_sketches.py`

```
""" Sketching-based Matrix Computations """

# Author: Jordi Montes <jomsdev@gmail.com>
# August 28, 2017

# 导入必要的库
import numpy as np

# 导入随机数生成和矩阵处理的辅助函数
from scipy._lib._util import check_random_state, rng_integers
from scipy.sparse import csc_matrix

# 声明本模块中公开的函数和类
__all__ = ['clarkson_woodruff_transform']


def cwt_matrix(n_rows, n_columns, seed=None):
    r"""
    Generate a matrix S which represents a Clarkson-Woodruff transform.

    Given the desired size of matrix, the method returns a matrix S of size
    (n_rows, n_columns) where each column has all the entries set to 0
    except for one position which has been randomly set to +1 or -1 with
    equal probability.

    Parameters
    ----------
    n_rows : int
        Number of rows of S
    n_columns : int
        Number of columns of S
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    S : (n_rows, n_columns) csc_matrix
        The returned matrix has ``n_columns`` nonzero entries.

    Notes
    -----
    Given a matrix A, with probability at least 9/10,
    .. math:: \|SA\| = (1 \pm \epsilon)\|A\|
    Where the error epsilon is related to the size of S.
    """
    # 使用指定的种子创建随机数生成器
    rng = check_random_state(seed)
    # 在范围 [0, n_rows) 内生成 n_columns 个随机行索引
    rows = rng_integers(rng, 0, n_rows, n_columns)
    # 列索引，从 0 到 n_columns
    cols = np.arange(n_columns + 1)
    # 随机选择每列的符号，+1 或 -1
    signs = rng.choice([1, -1], n_columns)
    # 创建稀疏的 csc_matrix，表示 Clarkson-Woodruff 变换的矩阵 S
    S = csc_matrix((signs, rows, cols), shape=(n_rows, n_columns))
    return S


def clarkson_woodruff_transform(input_matrix, sketch_size, seed=None):
    r"""
    Applies a Clarkson-Woodruff Transform/sketch to the input matrix.

    Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A'`` of
    size (sketch_size, d) so that

    .. math:: \|Ax\| \approx \|A'x\|

    with high probability via the Clarkson-Woodruff Transform, otherwise
    known as the CountSketch matrix.

    Parameters
    ----------
    input_matrix : array_like
        Input matrix, of shape ``(n, d)``.
    sketch_size : int
        Number of rows for the sketch.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    A' : array_like
        Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.

    Notes
    -----
    To make the statement

    .. math:: \|Ax\| \approx \|A'x\|

    precise, observe the following result which is adapted from the
"""
    # 证明 [2]_ 中定理 14 的一种实现，使用马尔可夫不等式。如果我们有一个至少为
    # ``sketch_size=k`` 的草图大小，满足以下条件：
    #
    # .. math:: k \geq \frac{2}{\epsilon^2\delta}
    #
    # 那么对于任意固定向量 ``x``，
    #
    # .. math:: \|Ax\| = (1\pm\epsilon)\|A'x\|
    #
    # 其中概率至少为一减去 delta。
    #
    # 这个实现利用稀疏性：计算草图的时间与 ``A.nnz`` 成正比。格式为
    # ``scipy.sparse.csc_matrix`` 的数据 ``A`` 提供了稀疏输入的最快计算时间。

    >>> import numpy as np
    >>> from scipy import linalg
    >>> from scipy import sparse
    >>> rng = np.random.default_rng()
    >>> n_rows, n_columns, density, sketch_n_rows = 15000, 100, 0.01, 200
    >>> A = sparse.rand(n_rows, n_columns, density=density, format='csc')
    >>> B = sparse.rand(n_rows, n_columns, density=density, format='csr')
    >>> C = sparse.rand(n_rows, n_columns, density=density, format='coo')
    >>> D = rng.standard_normal((n_rows, n_columns))

    # 使用 linalg.clarkson_woodruff_transform 函数对矩阵 A 进行变换，生成最快的草图 SA
    >>> SA = linalg.clarkson_woodruff_transform(A, sketch_n_rows) # fastest
    # 使用 linalg.clarkson_woodruff_transform 函数对矩阵 B 进行变换，生成快速的草图 SB
    >>> SB = linalg.clarkson_woodruff_transform(B, sketch_n_rows) # fast
    # 使用 linalg.clarkson_woodruff_transform 函数对矩阵 C 进行变换，生成较慢的草图 SC
    >>> SC = linalg.clarkson_woodruff_transform(C, sketch_n_rows) # slower
    # 使用 linalg.clarkson_woodruff_transform 函数对矩阵 D 进行变换，生成最慢的草图 SD
    >>> SD = linalg.clarkson_woodruff_transform(D, sketch_n_rows) # slowest

    # 尽管如此，该方法在稠密输入上表现良好，只是相对较慢。

    # 参考文献
    # ----------
    # .. [1] Kenneth L. Clarkson 和 David P. Woodruff. Low rank approximation
    #        and regression in input sparsity time. In STOC, 2013.
    # .. [2] David P. Woodruff. Sketching as a tool for numerical linear algebra.
    #        In Foundations and Trends in Theoretical Computer Science, 2014.
    # 计算一个数值，看起来像是某种度量或结果
    122.83242365433877
    # 计算使用修改后的矩阵乘积与向量的差的2-范数
    >>> linalg.norm(A @ x_sketched - b)
    166.58473879945151
    
    """
    # 根据给定的参数生成一个稀疏矩阵 S，用于对输入矩阵进行草图(sketch)
    S = cwt_matrix(sketch_size, input_matrix.shape[0], seed)
    # 返回草图 S 与输入矩阵的乘积结果
    return S.dot(input_matrix)
```