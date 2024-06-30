# `D:\src\scipysrc\sympy\sympy\matrices\sparse.py`

```
from collections.abc import Callable  # 导入Callable类，用于类型提示和函数签名检查

from sympy.core.containers import Dict  # 导入Dict类，可能用于特定数据结构的字典
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入sympy_deprecation_warning，用于处理Sympy库的废弃警告
from sympy.utilities.iterables import is_sequence  # 导入is_sequence函数，用于检查对象是否为序列
from sympy.utilities.misc import as_int  # 导入as_int函数，用于将输入转换为整数

from .matrixbase import MatrixBase  # 从当前包导入MatrixBase类
from .repmatrix import MutableRepMatrix, RepMatrix  # 从当前包导入MutableRepMatrix和RepMatrix类

from .utilities import _iszero  # 从当前包导入_iszero函数

from .decompositions import (
    _liupc, _row_structure_symbolic_cholesky, _cholesky_sparse,
    _LDLdecomposition_sparse)  # 从当前包的decompositions模块导入多个函数

from .solvers import (
    _lower_triangular_solve_sparse, _upper_triangular_solve_sparse)  # 从当前包的solvers模块导入多个函数

class SparseRepMatrix(RepMatrix):
    """
    A sparse matrix (a matrix with a large number of zero elements).

    Examples
    ========

    >>> from sympy import SparseMatrix, ones
    >>> SparseMatrix(2, 2, range(4))
    Matrix([
    [0, 1],
    [2, 3]])
    >>> SparseMatrix(2, 2, {(1, 1): 2})
    Matrix([
    [0, 0],
    [0, 2]])

    A SparseMatrix can be instantiated from a ragged list of lists:

    >>> SparseMatrix([[1, 2, 3], [1, 2], [1]])
    Matrix([
    [1, 2, 3],
    [1, 2, 0],
    [1, 0, 0]])

    For safety, one may include the expected size and then an error
    will be raised if the indices of any element are out of range or
    (for a flat list) if the total number of elements does not match
    the expected shape:

    >>> SparseMatrix(2, 2, [1, 2])
    Traceback (most recent call last):
    ...
    ValueError: List length (2) != rows*columns (4)

    Here, an error is not raised because the list is not flat and no
    element is out of range:

    >>> SparseMatrix(2, 2, [[1, 2]])
    Matrix([
    [1, 2],
    [0, 0]])

    But adding another element to the first (and only) row will cause
    an error to be raised:

    >>> SparseMatrix(2, 2, [[1, 2, 3]])
    Traceback (most recent call last):
    ...
    ValueError: The location (0, 2) is out of designated range: (1, 1)

    To autosize the matrix, pass None for rows:

    >>> SparseMatrix(None, [[1, 2, 3]])
    Matrix([[1, 2, 3]])
    >>> SparseMatrix(None, {(1, 1): 1, (3, 3): 3})
    Matrix([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 3]])

    Values that are themselves a Matrix are automatically expanded:

    >>> SparseMatrix(4, 4, {(1, 1): ones(2)})
    Matrix([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]])

    A ValueError is raised if the expanding matrix tries to overwrite
    a different element already present:

    >>> SparseMatrix(3, 3, {(0, 0): ones(2), (1, 1): 2})
    Traceback (most recent call last):
    ...
    ValueError: collision at (1, 1)

    See Also
    ========
    DenseMatrix
    MutableSparseMatrix
    ImmutableSparseMatrix
    """

    @classmethod
    @property
    def _smat(self):
        sympy_deprecation_warning(
            """
            The private _smat attribute of SparseMatrix is deprecated. Use the
            .todok() method instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-private-matrix-attributes"
        )
        # 引发 SymPy 库的警告，提醒用户 _smat 属性已被废弃，建议使用 .todok() 方法代替
        return self.todok()

    def _eval_inverse(self, **kwargs):
        # 调用逆矩阵求解方法，支持自定义参数，如求解方法、零判定函数和尝试块对角化
        return self.inv(method=kwargs.get('method', 'LDL'),
                        iszerofunc=kwargs.get('iszerofunc', _iszero),
                        try_block_diag=kwargs.get('try_block_diag', False))

    def applyfunc(self, f):
        """Apply a function to each element of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> m = SparseMatrix(2, 2, lambda i, j: i*2+j)
        >>> m
        Matrix([
        [0, 1],
        [2, 3]])
        >>> m.applyfunc(lambda i: 2*i)
        Matrix([
        [0, 2],
        [4, 6]])

        """
        if not callable(f):
            raise TypeError("`f` must be callable.")

        # XXX: This only applies the function to the nonzero elements of the
        # matrix so is inconsistent with DenseMatrix.applyfunc e.g.
        #   zeros(2, 2).applyfunc(lambda x: x + 1)
        # 将函数 f 应用于矩阵的非零元素，构建非零元素经函数 f 处理后的字典
        dok = {}
        for k, v in self.todok().items():
            fv = f(v)
            if fv != 0:
                dok[k] = fv

        return self._new(self.rows, self.cols, dok)

    def as_immutable(self):
        """Returns an Immutable version of this Matrix."""
        from .immutable import ImmutableSparseMatrix
        # 返回当前稀疏矩阵的不可变版本
        return ImmutableSparseMatrix(self)

    def as_mutable(self):
        """Returns a mutable version of this matrix.

        Examples
        ========

        >>> from sympy import ImmutableMatrix
        >>> X = ImmutableMatrix([[1, 2], [3, 4]])
        >>> Y = X.as_mutable()
        >>> Y[1, 1] = 5 # Can set values in Y
        >>> Y
        Matrix([
        [1, 2],
        [3, 5]])
        """
        # 返回当前稀疏矩阵的可变版本
        return MutableSparseMatrix(self)

    def col_list(self):
        """Returns a column-sorted list of non-zero elements of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a=SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.CL
        [(0, 0, 1), (1, 0, 3), (0, 1, 2), (1, 1, 4)]

        See Also
        ========

        sympy.matrices.sparse.SparseMatrix.row_list
        """
        # 返回按列排序的非零元素列表，每个元素表示为 (行索引, 列索引, 元素值)
        return [tuple(k + (self[k],)) for k in sorted(self.todok().keys(), key=lambda k: list(reversed(k)))]

    def nnz(self):
        """Returns the number of non-zero elements in Matrix."""
        # 返回矩阵中非零元素的数量
        return len(self.todok())
    def row_list(self):
        """Returns a row-sorted list of non-zero elements of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a = SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.RL
        [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]

        See Also
        ========

        sympy.matrices.sparse.SparseMatrix.col_list
        """
        # 使用 todok() 方法获取非零元素的位置信息，并按行排序后生成列表
        return [tuple(k + (self[k],)) for k in
            sorted(self.todok().keys(), key=list)]

    def scalar_multiply(self, scalar):
        "Scalar element-wise multiplication"
        # 矩阵与标量的逐元素相乘操作
        return scalar * self

    def solve_least_squares(self, rhs, method='LDL'):
        """Return the least-square fit to the data.

        By default the cholesky_solve routine is used (method='CH'); other
        methods of matrix inversion can be used. To find out which are
        available, see the docstring of the .inv() method.

        Examples
        ========

        >>> from sympy import SparseMatrix, Matrix, ones
        >>> A = Matrix([1, 2, 3])
        >>> B = Matrix([2, 3, 4])
        >>> S = SparseMatrix(A.row_join(B))
        >>> S
        Matrix([
        [1, 2],
        [2, 3],
        [3, 4]])

        If each line of S represent coefficients of Ax + By
        and x and y are [2, 3] then S*xy is:

        >>> r = S*Matrix([2, 3]); r
        Matrix([
        [ 8],
        [13],
        [18]])

        But let's add 1 to the middle value and then solve for the
        least-squares value of xy:

        >>> xy = S.solve_least_squares(Matrix([8, 14, 18])); xy
        Matrix([
        [ 5/3],
        [10/3]])

        The error is given by S*xy - r:

        >>> S*xy - r
        Matrix([
        [1/3],
        [1/3],
        [1/3]])
        >>> _.norm().n(2)
        0.58

        If a different xy is used, the norm will be higher:

        >>> xy += ones(2, 1)/10
        >>> (S*xy - r).norm().n(2)
        1.5

        """
        # 转置自身，并使用指定方法求解最小二乘拟合
        t = self.T
        return (t*self).inv(method=method)*t*rhs

    def solve(self, rhs, method='LDL'):
        """Return solution to self*soln = rhs using given inversion method.

        For a list of possible inversion methods, see the .inv() docstring.
        """
        # 检查矩阵是否为方阵，根据不同情况选择求解方法
        if not self.is_square:
            if self.rows < self.cols:
                raise ValueError('Under-determined system.')
            elif self.rows > self.cols:
                raise ValueError('For over-determined system, M, having '
                    'more rows than columns, try M.solve_least_squares(rhs).')
        else:
            return self.inv(method=method).multiply(rhs)

    RL = property(row_list, None, None, "Alternate faster representation")
    CL = property(col_list, None, None, "Alternate faster representation")

    def liupc(self):
        # 调用 _liupc 函数，返回其计算结果
        return _liupc(self)

    def row_structure_symbolic_cholesky(self):
        # 调用 _row_structure_symbolic_cholesky 函数，返回其计算结果
        return _row_structure_symbolic_cholesky(self)
    # 将类中的稀疏矩阵的 Cholesky 分解方法委托给对应的函数处理
    def cholesky(self, hermitian=True):
        return _cholesky_sparse(self, hermitian=hermitian)

    # 将类中的稀疏矩阵的 LDL 分解方法委托给对应的函数处理
    def LDLdecomposition(self, hermitian=True):
        return _LDLdecomposition_sparse(self, hermitian=hermitian)

    # 将类中的稀疏矩阵的下三角线性方程求解方法委托给对应的函数处理
    def lower_triangular_solve(self, rhs):
        return _lower_triangular_solve_sparse(self, rhs)

    # 将类中的稀疏矩阵的上三角线性方程求解方法委托给对应的函数处理
    def upper_triangular_solve(self, rhs):
        return _upper_triangular_solve_sparse(self, rhs)

    # 将稀疏矩阵的 Cholesky 方法的文档字符串指向对应的稀疏版本的文档
    liupc.__doc__                           = _liupc.__doc__
    # 将稀疏矩阵的行结构符号 Cholesky 方法的文档字符串指向对应的稀疏版本的文档
    row_structure_symbolic_cholesky.__doc__ = _row_structure_symbolic_cholesky.__doc__
    # 将类中的 Cholesky 方法的文档字符串指向对应的稀疏版本的文档
    cholesky.__doc__                        = _cholesky_sparse.__doc__
    # 将类中的 LDL 分解方法的文档字符串指向对应的稀疏版本的文档
    LDLdecomposition.__doc__                = _LDLdecomposition_sparse.__doc__
    # 将类中的下三角线性方程求解方法的文档字符串指向自身的文档字符串
    lower_triangular_solve.__doc__          = lower_triangular_solve.__doc__
    # 将类中的上三角线性方程求解方法的文档字符串指向自身的文档字符串
    upper_triangular_solve.__doc__          = upper_triangular_solve.__doc__
# 定义一个名为 MutableSparseMatrix 的类，它继承自 SparseRepMatrix 和 MutableRepMatrix
class MutableSparseMatrix(SparseRepMatrix, MutableRepMatrix):

    # 类方法 _new，用于创建新的 MutableSparseMatrix 实例
    @classmethod
    def _new(cls, *args, **kwargs):
        # 调用 _handle_creation_inputs 处理输入参数，获取行数、列数和稀疏矩阵数据
        rows, cols, smat = cls._handle_creation_inputs(*args, **kwargs)

        # 将稀疏矩阵数据转换为 DomainMatrix 的表示形式
        rep = cls._smat_to_DomainMatrix(rows, cols, smat)

        # 使用 _fromrep 方法从 rep 创建一个新的 MutableSparseMatrix 实例并返回
        return cls._fromrep(rep)


# 将 MutableSparseMatrix 类赋值给 SparseMatrix，使得 SparseMatrix 成为 MutableSparseMatrix 的别名
SparseMatrix = MutableSparseMatrix
```