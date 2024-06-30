# `D:\src\scipysrc\sympy\sympy\matrices\matrices.py`

```
#
# A module consisting of deprecated matrix classes. New code should not be
# added here.
#

# 从 sympy.core.basic 导入 Basic 类
from sympy.core.basic import Basic
# 从 sympy.core.symbol 导入 Dummy 类
from sympy.core.symbol import Dummy

# 从当前目录下的 common.py 导入 MatrixCommon 类
from .common import MatrixCommon

# 从当前目录下的 exceptions.py 导入 NonSquareMatrixError 异常类
from .exceptions import NonSquareMatrixError

# 从当前目录下的 utilities.py 导入一些函数 _iszero, _is_zero_after_expand_mul, _simplify
from .utilities import _iszero, _is_zero_after_expand_mul, _simplify

# 从当前目录下的 determinant.py 导入以下函数：
# _find_reasonable_pivot, _find_reasonable_pivot_naive, _adjugate, _charpoly,
# _cofactor, _cofactor_matrix, _per, _det, _det_bareiss, _det_berkowitz,
# _det_bird, _det_laplace, _det_LU, _minor, _minor_submatrix
from .determinant import (
    _find_reasonable_pivot, _find_reasonable_pivot_naive,
    _adjugate, _charpoly, _cofactor, _cofactor_matrix, _per,
    _det, _det_bareiss, _det_berkowitz, _det_bird, _det_laplace, _det_LU,
    _minor, _minor_submatrix)

# 从当前目录下的 reductions.py 导入以下函数：_is_echelon, _echelon_form, _rank, _rref
from .reductions import _is_echelon, _echelon_form, _rank, _rref

# 从当前目录下的 subspaces.py 导入以下函数：_columnspace, _nullspace, _rowspace, _orthogonalize
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize

# 从当前目录下的 eigen.py 导入以下函数：
# _eigenvals, _eigenvects, _bidiagonalize, _bidiagonal_decomposition,
# _is_diagonalizable, _diagonalize, _is_positive_definite, _is_positive_semidefinite,
# _is_negative_definite, _is_negative_semidefinite, _is_indefinite, _jordan_form,
# _left_eigenvects, _singular_values
from .eigen import (
    _eigenvals, _eigenvects,
    _bidiagonalize, _bidiagonal_decomposition,
    _is_diagonalizable, _diagonalize,
    _is_positive_definite, _is_positive_semidefinite,
    _is_negative_definite, _is_negative_semidefinite, _is_indefinite,
    _jordan_form, _left_eigenvects, _singular_values)

# 从当前目录下的 matrixbase.py 导入 MatrixBase 类
# 这个类曾经在当前模块中定义，但已移至 sympy.matrices.matrixbase，这里为了向后兼容性而导入
from .matrixbase import MatrixBase

# 以下是一个特殊的声明，指定某些 doctest 需要 matplotlib 库
__doctest_requires__ = {
    ('MatrixEigen.is_indefinite',
     'MatrixEigen.is_negative_definite',
     'MatrixEigen.is_negative_semidefinite',
     'MatrixEigen.is_positive_definite',
     'MatrixEigen.is_positive_semidefinite'): ['matplotlib'],
}


class MatrixDeterminant(MatrixCommon):
    """Provides basic matrix determinant operations. Should not be instantiated
    directly. See ``determinant.py`` for their implementations."""

    # 计算行列式的 Bareiss 算法实现
    def _eval_det_bareiss(self, iszerofunc=_is_zero_after_expand_mul):
        return _det_bareiss(self, iszerofunc=iszerofunc)

    # 计算行列式的 Berkowitz 算法实现
    def _eval_det_berkowitz(self):
        return _det_berkowitz(self)

    # 计算行列式的 LU 分解算法实现
    def _eval_det_lu(self, iszerofunc=_iszero, simpfunc=None):
        return _det_LU(self, iszerofunc=iszerofunc, simpfunc=simpfunc)

    # 计算行列式的 Bird 算法实现
    def _eval_det_bird(self):
        return _det_bird(self)

    # 计算行列式的 Laplace 算法实现
    def _eval_det_laplace(self):
        return _det_laplace(self)

    # 计算表达式的行列式
    def _eval_determinant(self): # for expressions.determinant.Determinant
        return _det(self)

    # 计算伴随矩阵
    def adjugate(self, method="berkowitz"):
        return _adjugate(self, method=method)

    # 计算特征多项式
    def charpoly(self, x='lambda', simplify=_simplify):
        return _charpoly(self, x=x, simplify=simplify)

    # 计算余子式
    def cofactor(self, i, j, method="berkowitz"):
        return _cofactor(self, i, j, method=method)

    # 计算余子式矩阵
    def cofactor_matrix(self, method="berkowitz"):
        return _cofactor_matrix(self, method=method)

    # 计算行列式
    def det(self, method="bareiss", iszerofunc=None):
        return _det(self, method=method, iszerofunc=iszerofunc)

    # 计算行列式的迹
    def per(self):
        return _per(self)

    # 计算指定位置的子行列式
    def minor(self, i, j, method="berkowitz"):
        return _minor(self, i, j, method=method)
    # 定义一个方法 `minor_submatrix`，返回调用 `_minor_submatrix` 方法的结果
    def minor_submatrix(self, i, j):
        return _minor_submatrix(self, i, j)

    # 将 `_find_reasonable_pivot` 方法的文档字符串设置为自身的文档字符串
    _find_reasonable_pivot.__doc__       = _find_reasonable_pivot.__doc__
    # 将 `_find_reasonable_pivot_naive` 方法的文档字符串设置为自身的文档字符串
    _find_reasonable_pivot_naive.__doc__ = _find_reasonable_pivot_naive.__doc__
    # 将 `_eval_det_bareiss` 方法的文档字符串设置为 `_det_bareiss` 方法的文档字符串
    _eval_det_bareiss.__doc__            = _det_bareiss.__doc__
    # 将 `_eval_det_berkowitz` 方法的文档字符串设置为 `_det_berkowitz` 方法的文档字符串
    _eval_det_berkowitz.__doc__          = _det_berkowitz.__doc__
    # 将 `_eval_det_bird` 方法的文档字符串设置为 `_det_bird` 方法的文档字符串
    _eval_det_bird.__doc__            = _det_bird.__doc__
    # 将 `_eval_det_laplace` 方法的文档字符串设置为 `_det_laplace` 方法的文档字符串
    _eval_det_laplace.__doc__            = _det_laplace.__doc__
    # 将 `_eval_det_lu` 方法的文档字符串设置为 `_det_LU` 方法的文档字符串
    _eval_det_lu.__doc__                 = _det_LU.__doc__
    # 将 `_eval_determinant` 方法的文档字符串设置为 `_det` 方法的文档字符串
    _eval_determinant.__doc__            = _det.__doc__
    # 将 `adjugate` 方法的文档字符串设置为 `_adjugate` 方法的文档字符串
    adjugate.__doc__                     = _adjugate.__doc__
    # 将 `charpoly` 方法的文档字符串设置为 `_charpoly` 方法的文档字符串
    charpoly.__doc__                     = _charpoly.__doc__
    # 将 `cofactor` 方法的文档字符串设置为 `_cofactor` 方法的文档字符串
    cofactor.__doc__                     = _cofactor.__doc__
    # 将 `cofactor_matrix` 方法的文档字符串设置为 `_cofactor_matrix` 方法的文档字符串
    cofactor_matrix.__doc__              = _cofactor_matrix.__doc__
    # 将 `det` 方法的文档字符串设置为 `_det` 方法的文档字符串
    det.__doc__                          = _det.__doc__
    # 将 `per` 方法的文档字符串设置为 `_per` 方法的文档字符串
    per.__doc__                          = _per.__doc__
    # 将 `minor` 方法的文档字符串设置为 `_minor` 方法的文档字符串
    minor.__doc__                        = _minor.__doc__
    # 将 `minor_submatrix` 方法的文档字符串设置为 `_minor_submatrix` 方法的文档字符串
    minor_submatrix.__doc__              = _minor_submatrix.__doc__
class MatrixReductions(MatrixDeterminant):
    """Provides basic matrix row/column operations. Should not be instantiated
    directly. See ``reductions.py`` for some of their implementations."""

    # 返回矩阵的梯形形式，支持指定的零判定函数、简化选项和返回主元的选项
    def echelon_form(self, iszerofunc=_iszero, simplify=False, with_pivots=False):
        return _echelon_form(self, iszerofunc=iszerofunc, simplify=simplify,
                with_pivots=with_pivots)

    @property
    # 检查当前矩阵是否为梯形形式
    def is_echelon(self):
        return _is_echelon(self)

    # 返回矩阵的秩，支持指定的零判定函数和简化选项
    def rank(self, iszerofunc=_iszero, simplify=False):
        return _rank(self, iszerofunc=iszerofunc, simplify=simplify)

    # 返回增广矩阵的简化行阶梯形式和对应的 rhs 矩阵
    def rref_rhs(self, rhs):
        """Return reduced row-echelon form of matrix, matrix showing
        rhs after reduction steps. ``rhs`` must have the same number
        of rows as ``self``.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> r1, r2 = symbols('r1 r2')
        >>> Matrix([[1, 1], [2, 1]]).rref_rhs(Matrix([r1, r2]))
        (Matrix([
        [1, 0],
        [0, 1]]), Matrix([
        [ -r1 + r2],
        [2*r1 - r2]]))
        """
        # 返回增广矩阵的简化行阶梯形式，并分离出对应的 rhs 部分
        r, _ = _rref(self.hstack(self, self.eye(self.rows), rhs))
        return r[:, :self.cols], r[:, -rhs.cols:]

    # 返回矩阵的简化行阶梯形式
    def rref(self, iszerofunc=_iszero, simplify=False, pivots=True,
            normalize_last=True):
        return _rref(self, iszerofunc=iszerofunc, simplify=simplify,
            pivots=pivots, normalize_last=normalize_last)

    # 将方法的文档字符串指定为对应函数的文档字符串
    echelon_form.__doc__ = _echelon_form.__doc__
    is_echelon.__doc__   = _is_echelon.__doc__
    rank.__doc__         = _rank.__doc__
    rref.__doc__         = _rref.__doc__
    def _normalize_op_args(self, op, col, k, col1, col2, error_str="col"):
        """Validate the arguments for a row/column operation.  ``error_str``
        can be one of "row" or "col" depending on the arguments being parsed."""
        # 检查操作是否在合法操作列表中，如果不在则引发 ValueError 异常
        if op not in ["n->kn", "n<->m", "n->n+km"]:
            raise ValueError("Unknown {} operation '{}'. Valid col operations "
                             "are 'n->kn', 'n<->m', 'n->n+km'".format(error_str, op))

        # 根据 error_str 定义 self_cols
        self_cols = self.cols if error_str == 'col' else self.rows

        # 标准化和验证参数
        if op == "n->kn":
            # 如果 col 为 None，则使用 col1
            col = col if col is not None else col1
            # 检查 col、k 是否为 None
            if col is None or k is None:
                raise ValueError("For a {0} operation 'n->kn' you must provide the "
                                 "kwargs `{0}` and `k`".format(error_str))
            # 检查 col 是否在有效范围内
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))

        elif op == "n<->m":
            # 需要两个列进行交换。无论如何，收集并排除 None
            cols = {col, k, col1, col2}.difference([None])
            # 如果提供的列数量大于2，则可能是用户错误地留下了 `k`
            if len(cols) > 2:
                cols = {col, col1, col2}.difference([None])
            # 检查是否提供了两列
            if len(cols) != 2:
                raise ValueError("For a {0} operation 'n<->m' you must provide the "
                                 "kwargs `{0}1` and `{0}2`".format(error_str))
            col1, col2 = cols
            # 检查两列是否在有效范围内
            if not 0 <= col1 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col1))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        elif op == "n->n+km":
            # 如果 col 为 None，则使用 col1；如果 col2 为 None，则使用 col1
            col = col1 if col is None else col
            col2 = col1 if col2 is None else col2
            # 检查 col、col2、k 是否为 None
            if col is None or col2 is None or k is None:
                raise ValueError("For a {0} operation 'n->n+km' you must provide the "
                                 "kwargs `{0}`, `k`, and `{0}2`".format(error_str))
            # 检查 col 和 col2 是否不同
            if col == col2:
                raise ValueError("For a {0} operation 'n->n+km' `{0}` and `{0}2` must "
                                 "be different.".format(error_str))
            # 检查 col、col2 是否在有效范围内
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        else:
            # 如果操作不在有效操作列表中，则引发 ValueError 异常
            raise ValueError('invalid operation %s' % repr(op))

        # 返回验证后的参数
        return op, col, k, col1, col2
    # 定义一个函数，用于将列 `col` 乘以常数 `k`
    def _eval_col_op_multiply_col_by_const(self, col, k):
        # 定义一个函数 `entry(i, j)`，用于处理矩阵的每个元素
        def entry(i, j):
            # 如果列索引 `j` 等于 `col`，则返回该位置元素乘以常数 `k`
            if j == col:
                return k * self[i, j]
            # 否则返回原始矩阵中的元素
            return self[i, j]
        # 使用新的元素函数创建一个新的矩阵对象并返回
        return self._new(self.rows, self.cols, entry)

    # 定义一个函数，用于交换两列的位置
    def _eval_col_op_swap(self, col1, col2):
        # 定义一个函数 `entry(i, j)`，用于处理矩阵的每个元素
        def entry(i, j):
            # 如果列索引 `j` 等于 `col1`，返回原始矩阵中第 `col2` 列的元素
            if j == col1:
                return self[i, col2]
            # 如果列索引 `j` 等于 `col2`，返回原始矩阵中第 `col1` 列的元素
            elif j == col2:
                return self[i, col1]
            # 否则返回原始矩阵中的元素
            return self[i, j]
        # 使用新的元素函数创建一个新的矩阵对象并返回
        return self._new(self.rows, self.cols, entry)

    # 定义一个函数，用于将一列的倍数加到另一列上
    def _eval_col_op_add_multiple_to_other_col(self, col, k, col2):
        # 定义一个函数 `entry(i, j)`，用于处理矩阵的每个元素
        def entry(i, j):
            # 如果列索引 `j` 等于 `col`，返回原始矩阵中第 `col` 列的元素加上常数 `k` 倍的第 `col2` 列的元素
            if j == col:
                return self[i, j] + k * self[i, col2]
            # 否则返回原始矩阵中的元素
            return self[i, j]
        # 使用新的元素函数创建一个新的矩阵对象并返回
        return self._new(self.rows, self.cols, entry)

    # 定义一个函数，用于交换两行的位置
    def _eval_row_op_swap(self, row1, row2):
        # 定义一个函数 `entry(i, j)`，用于处理矩阵的每个元素
        def entry(i, j):
            # 如果行索引 `i` 等于 `row1`，返回原始矩阵中第 `row2` 行的元素
            if i == row1:
                return self[row2, j]
            # 如果行索引 `i` 等于 `row2`，返回原始矩阵中第 `row1` 行的元素
            elif i == row2:
                return self[row1, j]
            # 否则返回原始矩阵中的元素
            return self[i, j]
        # 使用新的元素函数创建一个新的矩阵对象并返回
        return self._new(self.rows, self.cols, entry)

    # 定义一个函数，用于将某一行乘以常数 `k`
    def _eval_row_op_multiply_row_by_const(self, row, k):
        # 定义一个函数 `entry(i, j)`，用于处理矩阵的每个元素
        def entry(i, j):
            # 如果行索引 `i` 等于 `row`，则返回该位置元素乘以常数 `k`
            if i == row:
                return k * self[i, j]
            # 否则返回原始矩阵中的元素
            return self[i, j]
        # 使用新的元素函数创建一个新的矩阵对象并返回
        return self._new(self.rows, self.cols, entry)

    # 定义一个函数，用于将某一行的倍数加到另一行上
    def _eval_row_op_add_multiple_to_other_row(self, row, k, row2):
        # 定义一个函数 `entry(i, j)`，用于处理矩阵的每个元素
        def entry(i, j):
            # 如果行索引 `i` 等于 `row`，返回原始矩阵中第 `row` 行的元素加上常数 `k` 倍的第 `row2` 行的元素
            if i == row:
                return self[i, j] + k * self[row2, j]
            # 否则返回原始矩阵中的元素
            return self[i, j]
        # 使用新的元素函数创建一个新的矩阵对象并返回
        return self._new(self.rows, self.cols, entry)

    # 定义一个函数，执行指定的列基本操作
    def elementary_col_op(self, op="n->kn", col=None, k=None, col1=None, col2=None):
        """Performs the elementary column operation `op`.

        `op` may be one of

            * ``"n->kn"`` (column n goes to k*n)
            * ``"n<->m"`` (swap column n and column m)
            * ``"n->n+km"`` (column n goes to column n + k*column m)

        Parameters
        ==========

        op : string; the elementary row operation
        col : the column to apply the column operation
        k : the multiple to apply in the column operation
        col1 : one column of a column swap
        col2 : second column of a column swap or column "m" in the column operation
               "n->n+km"
        """

        # 标准化操作参数
        op, col, k, col1, col2 = self._normalize_op_args(op, col, k, col1, col2, "col")

        # 根据不同的操作类型分发执行不同的列操作
        if op == "n->kn":
            return self._eval_col_op_multiply_col_by_const(col, k)
        if op == "n<->m":
            return self._eval_col_op_swap(col1, col2)
        if op == "n->n+km":
            return self._eval_col_op_add_multiple_to_other_col(col, k, col2)
    # 定义一个方法，用于执行初等行操作 `op`。

    # `op` 可以是以下操作之一：
    #   * "n->kn" (第 n 行乘以常数 k)
    #   * "n<->m" (交换第 n 行和第 m 行)
    #   * "n->n+km" (第 n 行加上 k 倍的第 m 行)

    # Parameters
    # ==========

    # op : string; 初等行操作的类型
    # row : 要应用行操作的行索引
    # k : 行操作中要应用的倍数
    # row1 : 行交换中的一行
    # row2 : 行交换中的另一行或者行操作 "n->n+km" 中的第二行（m）

    def elementary_row_op(self, op="n->kn", row=None, k=None, row1=None, row2=None):
        """Performs the elementary row operation `op`.

        `op` may be one of

            * ``"n->kn"`` (row n goes to k*n)
            * ``"n<->m"`` (swap row n and row m)
            * ``"n->n+km"`` (row n goes to row n + k*row m)

        Parameters
        ==========

        op : string; the elementary row operation
        row : the row to apply the row operation
        k : the multiple to apply in the row operation
        row1 : one row of a row swap
        row2 : second row of a row swap or row "m" in the row operation
               "n->n+km"
        """

        # 将参数标准化为内部使用的格式
        op, row, k, row1, row2 = self._normalize_op_args(op, row, k, row1, row2, "row")

        # 根据不同的操作类型分派到对应的方法执行
        if op == "n->kn":
            return self._eval_row_op_multiply_row_by_const(row, k)
        if op == "n<->m":
            return self._eval_row_op_swap(row1, row2)
        if op == "n->n+km":
            return self._eval_row_op_add_multiple_to_other_row(row, k, row2)
class MatrixSubspaces(MatrixReductions):
    """Provides methods relating to the fundamental subspaces of a matrix.
    Should not be instantiated directly. See ``subspaces.py`` for their
    implementations."""

    # 返回列空间的计算结果
    def columnspace(self, simplify=False):
        return _columnspace(self, simplify=simplify)

    # 返回零空间的计算结果
    def nullspace(self, simplify=False, iszerofunc=_iszero):
        return _nullspace(self, simplify=simplify, iszerofunc=iszerofunc)

    # 返回行空间的计算结果
    def rowspace(self, simplify=False):
        return _rowspace(self, simplify=simplify)

    # 提供给定向量集合的正交化处理结果
    def orthogonalize(cls, *vecs, **kwargs):
        return _orthogonalize(cls, *vecs, **kwargs)

    columnspace.__doc__   = _columnspace.__doc__   # 设置列空间方法的文档字符串
    nullspace.__doc__     = _nullspace.__doc__     # 设置零空间方法的文档字符串
    rowspace.__doc__      = _rowspace.__doc__      # 设置行空间方法的文档字符串
    orthogonalize.__doc__ = _orthogonalize.__doc__ # 设置正交化方法的文档字符串

    orthogonalize         = classmethod(orthogonalize)  # type:ignore


class MatrixEigen(MatrixSubspaces):
    """Provides basic matrix eigenvalue/vector operations.
    Should not be instantiated directly. See ``eigen.py`` for their
    implementations."""

    # 返回特征值的计算结果
    def eigenvals(self, error_when_incomplete=True, **flags):
        return _eigenvals(self, error_when_incomplete=error_when_incomplete, **flags)

    # 返回特征向量的计算结果
    def eigenvects(self, error_when_incomplete=True, iszerofunc=_iszero, **flags):
        return _eigenvects(self, error_when_incomplete=error_when_incomplete,
                iszerofunc=iszerofunc, **flags)

    # 检查矩阵是否可对角化，并返回结果
    def is_diagonalizable(self, reals_only=False, **kwargs):
        return _is_diagonalizable(self, reals_only=reals_only, **kwargs)

    # 对矩阵进行对角化处理，并返回结果
    def diagonalize(self, reals_only=False, sort=False, normalize=False):
        return _diagonalize(self, reals_only=reals_only, sort=sort,
                normalize=normalize)

    # 对矩阵进行双对角化处理，并返回结果
    def bidiagonalize(self, upper=True):
        return _bidiagonalize(self, upper=upper)

    # 对矩阵进行双对角分解，并返回结果
    def bidiagonal_decomposition(self, upper=True):
        return _bidiagonal_decomposition(self, upper=upper)

    # 返回矩阵是否正定的属性
    @property
    def is_positive_definite(self):
        return _is_positive_definite(self)

    # 返回矩阵是否半正定的属性
    @property
    def is_positive_semidefinite(self):
        return _is_positive_semidefinite(self)

    # 返回矩阵是否负定的属性
    @property
    def is_negative_definite(self):
        return _is_negative_definite(self)

    # 返回矩阵是否半负定的属性
    @property
    def is_negative_semidefinite(self):
        return _is_negative_semidefinite(self)

    # 返回矩阵是否不定的属性
    @property
    def is_indefinite(self):
        return _is_indefinite(self)

    # 计算矩阵的约当标准形，并返回结果
    def jordan_form(self, calc_transform=True, **kwargs):
        return _jordan_form(self, calc_transform=calc_transform, **kwargs)

    # 返回左特征向量的计算结果
    def left_eigenvects(self, **flags):
        return _left_eigenvects(self, **flags)

    # 返回矩阵的奇异值分解结果
    def singular_values(self):
        return _singular_values(self)

    eigenvals.__doc__ = _eigenvals.__doc__  # 设置特征值计算方法的文档字符串
    # 将每个函数的文档字符串设置为对应私有函数的文档字符串
    eigenvects.__doc__                 = _eigenvects.__doc__
    is_diagonalizable.__doc__          = _is_diagonalizable.__doc__
    diagonalize.__doc__                = _diagonalize.__doc__
    is_positive_definite.__doc__       = _is_positive_definite.__doc__
    is_positive_semidefinite.__doc__   = _is_positive_semidefinite.__doc__
    is_negative_definite.__doc__       = _is_negative_definite.__doc__
    is_negative_semidefinite.__doc__   = _is_negative_semidefinite.__doc__
    is_indefinite.__doc__              = _is_indefinite.__doc__
    jordan_form.__doc__                = _jordan_form.__doc__
    left_eigenvects.__doc__            = _left_eigenvects.__doc__
    singular_values.__doc__            = _singular_values.__doc__
    bidiagonalize.__doc__              = _bidiagonalize.__doc__
    bidiagonal_decomposition.__doc__   = _bidiagonal_decomposition.__doc__
# MatrixCalculus 类，继承自 MatrixCommon 类，提供与微积分相关的矩阵操作
class MatrixCalculus(MatrixCommon):
    """Provides calculus-related matrix operations."""

    def diff(self, *args, evaluate=True, **kwargs):
        """Calculate the derivative of each element in the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.diff(x)
        Matrix([
        [1, 0],
        [0, 0]])

        See Also
        ========

        integrate
        limit
        """
        # 导入 ArrayDerivative 类，用于计算数组导数
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        # 创建 ArrayDerivative 实例 deriv，用于计算矩阵的导数
        deriv = ArrayDerivative(self, *args, evaluate=evaluate)
        # 如果矩阵不是 Basic 类型且 evaluate=True，则将导数作为可变矩阵返回
        if not isinstance(self, Basic) and evaluate:
            return deriv.as_mutable()
        # 否则返回导数对象
        return deriv

    def _eval_derivative(self, arg):
        # 对矩阵中的每个元素应用 lambda 函数，计算其对参数 arg 的导数
        return self.applyfunc(lambda x: x.diff(arg))

    def integrate(self, *args, **kwargs):
        """Integrate each element of the matrix.  ``args`` will
        be passed to the ``integrate`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.integrate((x, ))
        Matrix([
        [x**2/2, x*y],
        [     x,   0]])
        >>> M.integrate((x, 0, 2))
        Matrix([
        [2, 2*y],
        [2,   0]])

        See Also
        ========

        limit
        diff
        """
        # 对矩阵中的每个元素应用 lambda 函数，进行积分操作，并传递参数 args 和 kwargs
        return self.applyfunc(lambda x: x.integrate(*args, **kwargs))
    def jacobian(self, X):
        """Calculates the Jacobian matrix (derivative of a vector-valued function).

        Parameters
        ==========

        ``self`` : vector of expressions representing functions f_i(x_1, ..., x_n).
        X : set of x_i's in order, it can be a list or a Matrix

        Both ``self`` and X can be a row or a column matrix in any order
        (i.e., jacobian() should always work).

        Examples
        ========

        >>> from sympy import sin, cos, Matrix
        >>> from sympy.abc import rho, phi
        >>> X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
        >>> Y = Matrix([rho, phi])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0]])
        >>> X = Matrix([rho*cos(phi), rho*sin(phi)])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)]])

        See Also
        ========

        hessian
        wronskian
        """
        if not isinstance(X, MatrixBase):
            # Convert X to a Matrix if it's not already
            X = self._new(X)
        
        # Determine the number of functions and variables
        if self.shape[0] == 1:
            m = self.shape[1]  # Number of functions
        elif self.shape[1] == 1:
            m = self.shape[0]  # Number of functions
        else:
            raise TypeError("``self`` must be a row or a column matrix")
        
        if X.shape[0] == 1:
            n = X.shape[1]  # Number of variables
        elif X.shape[1] == 1:
            n = X.shape[0]  # Number of variables
        else:
            raise TypeError("X must be a row or a column matrix")

        # m is the number of functions and n is the number of variables
        # computing the Jacobian is now easy:
        return self._new(m, n, lambda j, i: self[j].diff(X[i]))

    def limit(self, *args):
        """Calculate the limit of each element in the matrix.
        ``args`` will be passed to the ``limit`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.limit(x, 2)
        Matrix([
        [2, y],
        [1, 0]])

        See Also
        ========

        integrate
        diff
        """
        # Apply the limit function to each element of the matrix
        return self.applyfunc(lambda x: x.limit(*args))
# 定义了一个继承自 MatrixCommon 的类 MatrixDeprecated，用于存放已弃用的矩阵方法。
class MatrixDeprecated(MatrixCommon):
    """A class to house deprecated matrix methods."""

    # 使用 Berkowitz 方法计算特征多项式，将方法重定向到 charpoly 方法
    def berkowitz_charpoly(self, x=Dummy('lambda'), simplify=_simplify):
        return self.charpoly(x=x)

    # 使用 Berkowitz 方法计算行列式，将方法重定向到 det 方法
    def berkowitz_det(self):
        """Computes determinant using Berkowitz method.

        See Also
        ========

        det
        berkowitz
        """
        return self.det(method='berkowitz')

    # 使用 Berkowitz 方法计算矩阵的特征值，将方法重定向到 eigenvals 方法
    def berkowitz_eigenvals(self, **flags):
        """Computes eigenvalues of a Matrix using Berkowitz method.

        See Also
        ========

        berkowitz
        """
        return self.eigenvals(**flags)

    # 使用 Berkowitz 方法计算主子式，遍历每个 Berkowitz 多项式，并计算出主子式
    def berkowitz_minors(self):
        """Computes principal minors using Berkowitz method.

        See Also
        ========

        berkowitz
        """
        sign, minors = self.one, []

        # 对于每个 Berkowitz 多项式，计算主子式
        for poly in self.berkowitz():
            minors.append(sign * poly[-1])
            sign = -sign

        return tuple(minors)

    # 使用 Berkowitz 方法计算矩阵的 Berkowitz 多项式序列
    def berkowitz(self):
        from sympy.matrices import zeros
        berk = ((1,),)
        if not self:
            return berk

        if not self.is_square:
            raise NonSquareMatrixError()

        A, N = self, self.rows
        transforms = [0] * (N - 1)

        # 构建 Berkowitz 多项式序列的变换矩阵
        for n in range(N, 1, -1):
            T, k = zeros(n + 1, n), n - 1

            R, C = -A[k, :k], A[:k, k]
            A, a = A[:k, :k], -A[k, k]

            items = [C]

            for i in range(0, n - 2):
                items.append(A * items[i])

            for i, B in enumerate(items):
                items[i] = (R * B)[0, 0]

            items = [self.one, a] + items

            for i in range(n):
                T[i:, i] = items[:n - i + 1]

            transforms[k - 1] = T

        polys = [self._new([self.one, -A[0, 0]])]

        # 计算 Berkowitz 多项式序列
        for i, T in enumerate(transforms):
            polys.append(T * polys[i])

        return berk + tuple(map(tuple, polys))

    # 使用指定方法计算余子矩阵，将方法重定向到 cofactor_matrix 方法
    def cofactorMatrix(self, method="berkowitz"):
        return self.cofactor_matrix(method=method)

    # 使用 Bareiss 方法计算行列式，将方法重定向到 _det_bareiss 方法
    def det_bareis(self):
        return _det_bareiss(self)

    # 使用 LU 分解计算行列式，将方法重定向到 det 方法
    def det_LU_decomposition(self):
        """Compute matrix determinant using LU decomposition.


        Note that this method fails if the LU decomposition itself
        fails. In particular, if the matrix has no inverse this method
        will fail.

        TODO: Implement algorithm for sparse matrices (SFF),
        https://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps

        See Also
        ========


        det
        det_bareiss
        berkowitz_det
        """
        return self.det(method='lu')

    # 使用 Jordan 块计算矩阵的 Jordan 细胞，将方法重定向到 jordan_block 方法
    def jordan_cell(self, eigenval, n):
        return self.jordan_block(size=n, eigenvalue=eigenval)

    # 使用 Jordan 形式计算矩阵的 Jordan 细胞，返回 P 和对角块 J
    def jordan_cells(self, calc_transformation=True):
        P, J = self.jordan_form()
        return P, J.get_diag_blocks()

    # 使用指定方法计算主子式，将方法重定向到 minor 方法
    def minorEntry(self, i, j, method="berkowitz"):
        return self.minor(i, j, method=method)
    # 返回矩阵的(i, j)位置的子矩阵，即去掉第i行第j列后的矩阵
    def minorMatrix(self, i, j):
        return self.minor_submatrix(i, j)

    # 使用给定的置换逆向置换矩阵的行
    def permuteBkwd(self, perm):
        """Permute the rows of the matrix with the given permutation in reverse."""
        return self.permute_rows(perm, direction='backward')

    # 使用给定的置换正向置换矩阵的行
    def permuteFwd(self, perm):
        """Permute the rows of the matrix with the given permutation."""
        return self.permute_rows(perm, direction='forward')
```