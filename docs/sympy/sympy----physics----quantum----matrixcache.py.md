# `D:\src\scipysrc\sympy\sympy\physics\quantum\matrixcache.py`

```
"""A cache for storing small matrices in multiple formats."""

# 导入所需的模块和类
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.power import Pow
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix

# 导入量子物理相关的矩阵工具函数
from sympy.physics.quantum.matrixutils import (
    to_sympy, to_numpy, to_scipy_sparse
)


class MatrixCache:
    """A cache for small matrices in different formats.

    This class takes small matrices in the standard ``sympy.Matrix`` format,
    and then converts these to both ``numpy.matrix`` and
    ``scipy.sparse.csr_matrix`` matrices. These matrices are then stored for
    future recovery.
    """

    def __init__(self, dtype='complex'):
        # 初始化缓存字典和数据类型
        self._cache = {}
        self.dtype = dtype

    def cache_matrix(self, name, m):
        """Cache a matrix by its name.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        m : list of lists
            The raw matrix data as a SymPy Matrix.
        """
        try:
            # 尝试缓存为 SymPy 格式的矩阵
            self._sympy_matrix(name, m)
        except ImportError:
            pass
        try:
            # 尝试缓存为 NumPy 格式的矩阵
            self._numpy_matrix(name, m)
        except ImportError:
            pass
        try:
            # 尝试缓存为 SciPy 稀疏矩阵格式的矩阵
            self._scipy_sparse_matrix(name, m)
        except ImportError:
            pass

    def get_matrix(self, name, format):
        """Get a cached matrix by name and format.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        format : str
            The format desired ('sympy', 'numpy', 'scipy.sparse')
        """
        m = self._cache.get((name, format))
        if m is not None:
            return m
        # 如果未找到对应的矩阵，则抛出未实现错误
        raise NotImplementedError(
            'Matrix with name %s and format %s is not available.' %
            (name, format)
        )

    def _store_matrix(self, name, format, m):
        # 将矩阵存储到缓存中
        self._cache[(name, format)] = m

    def _sympy_matrix(self, name, m):
        # 将矩阵转换为 SymPy 格式并存储
        self._store_matrix(name, 'sympy', to_sympy(m))

    def _numpy_matrix(self, name, m):
        # 将矩阵转换为 NumPy 格式并存储
        m = to_numpy(m, dtype=self.dtype)
        self._store_matrix(name, 'numpy', m)

    def _scipy_sparse_matrix(self, name, m):
        # 将矩阵转换为 SciPy 稀疏矩阵格式并存储
        m = to_scipy_sparse(m, dtype=self.dtype)
        self._store_matrix(name, 'scipy.sparse', m)


# 计算 2 的负一半次方，并用 Pow 类表示
sqrt2_inv = Pow(2, Rational(-1, 2), evaluate=False)

# 创建一个 MatrixCache 对象作为矩阵缓存
matrix_cache = MatrixCache()

# 缓存常用的矩阵数据
matrix_cache.cache_matrix('eye2', Matrix([[1, 0], [0, 1]]))  # 单位矩阵
matrix_cache.cache_matrix('op11', Matrix([[0, 0], [0, 1]]))  # |1><1|
matrix_cache.cache_matrix('op00', Matrix([[1, 0], [0, 0]]))  # |0><0|
matrix_cache.cache_matrix('op10', Matrix([[0, 0], [1, 0]]))  # |1><0|
matrix_cache.cache_matrix('op01', Matrix([[0, 1], [0, 0]]))  # |0><1|
matrix_cache.cache_matrix('X', Matrix([[0, 1], [1, 0]]))     # Pauli X 矩阵
# 缓存矩阵 'Y'，对应矩阵 [[0, -I], [I, 0]]
matrix_cache.cache_matrix('Y', Matrix([[0, -I], [I, 0]]))

# 缓存矩阵 'Z'，对应矩阵 [[1, 0], [0, -1]]
matrix_cache.cache_matrix('Z', Matrix([[1, 0], [0, -1]]))

# 缓存矩阵 'S'，对应矩阵 [[1, 0], [0, I]]
matrix_cache.cache_matrix('S', Matrix([[1, 0], [0, I]]))

# 缓存矩阵 'T'，对应矩阵 [[1, 0], [0, exp(I*pi/4)]]
matrix_cache.cache_matrix('T', Matrix([[1, 0], [0, exp(I*pi/4)]]))

# 缓存矩阵 'H'，对应矩阵 sqrt2_inv * [[1, 1], [1, -1]]
matrix_cache.cache_matrix('H', sqrt2_inv * Matrix([[1, 1], [1, -1]]))

# 缓存矩阵 'Hsqrt2'，对应矩阵 [[1, 1], [1, -1]]
matrix_cache.cache_matrix('Hsqrt2', Matrix([[1, 1], [1, -1]]))

# 缓存矩阵 'SWAP'，对应矩阵 [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
matrix_cache.cache_matrix(
    'SWAP', Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))

# 缓存矩阵 'ZX'，对应矩阵 sqrt2_inv * [[1, 1], [1, -1]]
matrix_cache.cache_matrix('ZX', sqrt2_inv * Matrix([[1, 1], [1, -1]]))

# 缓存矩阵 'ZY'，对应矩阵 [[I, 0], [0, -I]]
matrix_cache.cache_matrix('ZY', Matrix([[I, 0], [0, -I]]))
```