# `D:\src\scipysrc\sympy\sympy\matrices\common.py`

```
"""
A module contining deprecated matrix mixin classes.

The classes in this module are deprecated and will be removed in a future
release. They are kept here for backwards compatibility in case downstream
code was subclassing them.

Importing anything else from this module is deprecated so anything here
should either not be used or should be imported from somewhere else.
"""

# 导入标准库中的数据结构 defaultdict
from collections import defaultdict
# 导入抽象基类 Iterable
from collections.abc import Iterable
# 导入函数用于检查对象是否为函数
from inspect import isfunction
# 导入 functools 中的 reduce 函数
from functools import reduce

# 导入 sympy 库中的 refine 函数
from sympy.assumptions.refine import refine
# 导入 sympy 库中的异常 SympifyError 和数学对象 Add
from sympy.core import SympifyError, Add
# 导入 sympy 库中的基本对象 Atom
from sympy.core.basic import Atom
# 导入 sympy 库中的装饰器 call_highest_priority
from sympy.core.decorators import call_highest_priority
# 导入 sympy 库中的逻辑操作函数 fuzzy_and 和 FuzzyBool
from sympy.core.logic import fuzzy_and, FuzzyBool
# 导入 sympy 库中的数值对象 Integer
from sympy.core.numbers import Integer
# 导入 sympy 库中的模运算对象 Mod
from sympy.core.mod import Mod
# 导入 sympy 库中的单例对象 S
from sympy.core.singleton import S
# 导入 sympy 库中的符号对象 Symbol
from sympy.core.symbol import Symbol
# 导入 sympy 库中的符号化函数 sympify
from sympy.core.sympify import sympify
# 导入 sympy 库中的复数操作函数 Abs, re, im
from sympy.functions.elementary.complexes import Abs, re, im
# 导入 sympy 库中的异常 sympy_deprecation_warning
from sympy.utilities.exceptions import sympy_deprecation_warning
# 从当前包中的 utilities 模块导入 _dotprodsimp 和 _simplify 函数
from .utilities import _dotprodsimp, _simplify
# 导入 sympy 中的多项式工具 Poly
from sympy.polys.polytools import Poly
# 导入 sympy 中的迭代工具 flatten 和 is_sequence
from sympy.utilities.iterables import flatten, is_sequence
# 导入 sympy 中的杂项工具 as_int 和 filldedent
from sympy.utilities.misc import as_int, filldedent
# 导入 sympy 中的张量数组对象 NDimArray
from sympy.tensor.array import NDimArray

# 从当前包中的 utilities 模块导入 _get_intermediate_simp_bool 函数
from .utilities import _get_intermediate_simp_bool


# 下面的异常类型曾经在本模块中定义，但已移动到 exceptions.py 中。
# 我们在这里重新导入它们，以确保向后兼容性，以防下游代码从这里导入它们。
from .exceptions import ( # noqa: F401
    MatrixError, ShapeError, NonSquareMatrixError, NonInvertibleMatrixError,
    NonPositiveDefiniteMatrixError
)


# 需要被移除的混合类名列表
_DEPRECATED_MIXINS = (
    'MatrixShaping',
    'MatrixSpecial',
    'MatrixProperties',
    'MatrixOperations',
    'MatrixArithmetic',
    'MatrixCommon',
    'MatrixDeterminant',
    'MatrixReductions',
    'MatrixSubspaces',
    'MatrixEigen',
    'MatrixCalculus',
    'MatrixDeprecated',
)


# _MatrixDeprecatedMeta 元类的定义
class _MatrixDeprecatedMeta(type):

    #
    # Override the default __instancecheck__ implementation to ensure that
    # e.g. isinstance(M, MatrixCommon) still works when M is one of the
    # matrix classes. Matrix no longer inherits from MatrixCommon so
    # isinstance(M, MatrixCommon) would now return False by default.
    #
    # There were lots of places in the codebase where this was being done
    # so it seems likely that downstream code may be doing it too. All use
    # of these mixins is deprecated though so we give a deprecation warning
    # unconditionally if they are being used with isinstance.
    #
    # Any code seeing this deprecation warning should be changed to use
    # isinstance(M, MatrixBase) instead which also works in previous versions
    # of SymPy.
    #
    # 定义一个特殊方法 __instancecheck__，用于检查对象是否为特定类的实例
    def __instancecheck__(cls, instance):

        # 引发符号计算库的弃用警告，提示某个类不应再使用 isinstance 进行判断
        sympy_deprecation_warning(
            f"""
            Checking whether an object is an instance of {cls.__name__} is
            deprecated.

            Use `isinstance(obj, Matrix)` instead of `isinstance(obj, {cls.__name__})`.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-matrix-mixins",
            stacklevel=3,
        )

        # 导入所需的矩阵基类和相关的矩阵混合类
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.matrices import (
            MatrixDeterminant,
            MatrixReductions,
            MatrixSubspaces,
            MatrixEigen,
            MatrixCalculus,
            MatrixDeprecated
        )

        # 所有的矩阵混合类的集合
        all_mixins = (
            MatrixRequired,
            MatrixShaping,
            MatrixSpecial,
            MatrixProperties,
            MatrixOperations,
            MatrixArithmetic,
            MatrixCommon,
            MatrixDeterminant,
            MatrixReductions,
            MatrixSubspaces,
            MatrixEigen,
            MatrixCalculus,
            MatrixDeprecated
        )

        # 如果当前类属于所有混合类中的一个，并且 instance 是 MatrixBase 类的实例，则返回 True
        if cls in all_mixins and isinstance(instance, MatrixBase):
            return True
        else:
            # 否则调用父类的 __instancecheck__ 方法进行进一步检查
            return super().__instancecheck__(instance)
class MatrixRequired(metaclass=_MatrixDeprecatedMeta):
    """Deprecated mixin class for making matrix classes."""

    # 类属性，表示矩阵的行数，初始为None
    rows = None  # type: int
    # 类属性，表示矩阵的列数，初始为None
    cols = None  # type: int
    # 类属性，用于简化矩阵操作，初始为None
    _simplify = None

    def __init_subclass__(cls, **kwargs):
        # 当子类被创建时执行的方法

        # 如果子类名不在_DEPRECATED_MIXINS中，则发出警告，表明子类正在继承此类或任何被弃用的mixin类
        # 我们不希望在创建弃用的mixin类本身时发出警告，只希望在它们被用作mixin类时发出警告
        # 否则，仅导入此模块将触发警告
        # 最终整个模块应该被弃用并移除，但在SymPy 1.13版本中，由于此模块是所有先前版本中引入矩阵异常类型的主要方式，因此现在删除它还为时过早。

        if cls.__name__ not in _DEPRECATED_MIXINS:
            sympy_deprecation_warning(
                f"""
                Inheriting from the Matrix mixin classes is deprecated.

                The class {cls.__name__} is subclassing a deprecated mixin.
                """,
                deprecated_since_version="1.13",
                active_deprecations_target="deprecated-matrix-mixins",
                stacklevel=3,
            )

        super().__init_subclass__(**kwargs)

    @classmethod
    def _new(cls, *args, **kwargs):
        """`_new`必须至少可调用为`_new(rows, cols, mat)`，其中`mat`是矩阵元素的扁平列表。"""
        # 抛出未实现错误，子类必须实现这个方法
        raise NotImplementedError("Subclasses must implement this.")

    def __eq__(self, other):
        # 抛出未实现错误，子类必须实现这个方法
        raise NotImplementedError("Subclasses must implement this.")

    def __getitem__(self, key):
        """实现__getitem__方法应接受整数，此时矩阵作为扁平列表索引，元组(i,j)，此时返回(i,j)条目，切片或混合元组(a,b)，其中a和b是任意组合的切片和整数。"""
        # 抛出未实现错误，子类必须实现这个方法
        raise NotImplementedError("Subclasses must implement this.")

    def __len__(self):
        """矩阵中条目的总数。"""
        # 抛出未实现错误，子类必须实现这个方法
        raise NotImplementedError("Subclasses must implement this.")

    @property
    def shape(self):
        # 抛出未实现错误，子类必须实现这个方法
        raise NotImplementedError("Subclasses must implement this.")


class MatrixShaping(MatrixRequired):
    """提供基本的矩阵形状操作和子矩阵提取功能"""

    def _eval_col_del(self, col):
        def entry(i, j):
            return self[i, j] if j < col else self[i, j + 1]
        # 返回一个新的矩阵，删除指定列后的结果
        return self._new(self.rows, self.cols - 1, entry)
    # 定义一个方法，用于在当前矩阵列的特定位置插入另一个矩阵
    def _eval_col_insert(self, pos, other):

        # 定义一个内部函数，用于根据给定的行和列索引返回对应位置的元素
        def entry(i, j):
            if j < pos:
                return self[i, j]  # 如果列索引小于插入位置pos，则返回当前矩阵的元素
            elif pos <= j < pos + other.cols:
                return other[i, j - pos]  # 如果列索引在插入位置pos和pos+other.cols之间，则返回插入矩阵的元素
            return self[i, j - other.cols]  # 否则返回当前矩阵中跳过插入矩阵后的元素

        # 调用_new方法，创建一个新矩阵，行数不变，列数为当前列数加上插入矩阵的列数，使用entry函数填充矩阵
        return self._new(self.rows, self.cols + other.cols, entry)

    # 定义一个方法，用于按列连接两个矩阵
    def _eval_col_join(self, other):
        rows = self.rows

        # 定义一个内部函数，根据给定的行和列索引返回对应位置的元素
        def entry(i, j):
            if i < rows:
                return self[i, j]  # 如果行索引小于当前矩阵的行数，则返回当前矩阵的元素
            return other[i - rows, j]  # 否则返回另一个矩阵中对应位置的元素

        # 调用classof函数创建相同类别的新矩阵，行数为两个矩阵行数之和，列数为当前矩阵列数，使用entry函数填充矩阵
        return classof(self, other)._new(self.rows + other.rows, self.cols, entry)

    # 定义一个方法，用于提取指定行和列的子矩阵
    def _eval_extract(self, rowsList, colsList):
        mat = list(self)
        cols = self.cols
        indices = (i * cols + j for i in rowsList for j in colsList)
        # 使用指定的行和列列表创建新矩阵，将矩阵元素按行主序存储在列表中
        return self._new(len(rowsList), len(colsList),
                         [mat[i] for i in indices])

    # 定义一个方法，用于获取当前矩阵的对角块
    def _eval_get_diag_blocks(self):
        sub_blocks = []

        # 定义一个递归函数，用于查找当前矩阵中的对角块并存储在sub_blocks列表中
        def recurse_sub_blocks(M):
            for i in range(1, M.shape[0] + 1):
                if i == 1:
                    to_the_right = M[0, i:]
                    to_the_bottom = M[i:, 0]
                else:
                    to_the_right = M[:i, i:]
                    to_the_bottom = M[i:, :i]
                # 如果当前对角块右侧或下侧有非零元素，则继续递归查找
                if any(to_the_right) or any(to_the_bottom):
                    continue
                # 将找到的对角块添加到sub_blocks列表中
                sub_blocks.append(M[:i, :i])
                # 如果对角块大小与当前矩阵不同，则递归查找剩余部分
                if M.shape != M[:i, :i].shape:
                    recurse_sub_blocks(M[i:, i:])
                return

        # 调用递归函数开始查找对角块
        recurse_sub_blocks(self)
        # 返回找到的所有对角块列表
        return sub_blocks

    # 定义一个方法，用于删除指定行的元素后形成新矩阵
    def _eval_row_del(self, row):
        # 定义一个内部函数，根据给定的行和列索引返回对应位置的元素
        def entry(i, j):
            return self[i, j] if i < row else self[i + 1, j]  # 如果行索引小于指定行row，则返回当前矩阵的元素，否则返回跳过指定行后的元素
        # 调用_new方法，创建一个新矩阵，行数减一，列数不变，使用entry函数填充矩阵
        return self._new(self.rows - 1, self.cols, entry)

    # 定义一个方法，用于在指定位置pos插入另一个矩阵形成新矩阵
    def _eval_row_insert(self, pos, other):
        entries = list(self)
        insert_pos = pos * self.cols
        # 在entries列表中的指定位置插入另一个矩阵的元素
        entries[insert_pos:insert_pos] = list(other)
        # 调用_new方法，创建一个新矩阵，行数为当前行数加上插入矩阵的行数，列数不变，使用entries填充矩阵
        return self._new(self.rows + other.rows, self.cols, entries)

    # 定义一个方法，用于按行连接两个矩阵
    def _eval_row_join(self, other):
        cols = self.cols

        # 定义一个内部函数，根据给定的行和列索引返回对应位置的元素
        def entry(i, j):
            if j < cols:
                return self[i, j]  # 如果列索引小于当前矩阵的列数，则返回当前矩阵的元素
            return other[i, j - cols]  # 否则返回另一个矩阵中对应位置的元素

        # 调用classof函数创建相同类别的新矩阵，行数不变，列数为两个矩阵列数之和，使用entry函数填充矩阵
        return classof(self, other)._new(self.rows, self.cols + other.cols,
                                         entry)

    # 定义一个方法，将当前矩阵转换为列表的形式
    def _eval_tolist(self):
        # 返回一个列表，包含当前矩阵每一行的列表形式
        return [list(self[i,:]) for i in range(self.rows)]

    # 定义一个方法，将当前矩阵转换为字典的形式
    def _eval_todok(self):
        dok = {}
        rows, cols = self.shape
        # 遍历当前矩阵的每个元素，如果不为零，则添加到字典dok中
        for i in range(rows):
            for j in range(cols):
                val = self[i, j]
                if val != self.zero:
                    dok[i, j] = val
        return dok

    # 定义一个方法，将当前矩阵转换为向量的形式
    def _eval_vec(self):
        rows = self.rows

        # 定义一个内部函数，根据给定的行索引返回当前矩阵中对应位置的元素
        def entry(n, _):
            # 将向量的索引n映射回矩阵的行列索引
            j = n // rows
            i = n - j * rows
            return self[i, j]

        # 调用_new方法，创建一个新矩阵，行数为当前矩阵元素个数，列数为1，使用entry函数填充矩阵
        return self._new(len(self), 1, entry)
    def _eval_vech(self, diagonal):
        # 获取矩阵的列数
        c = self.cols
        # 初始化空列表 v 用于存储结果
        v = []
        # 如果 diagonal 为 True，则生成对角线以上的元素
        if diagonal:
            for j in range(c):
                for i in range(j, c):
                    # 将矩阵中指定位置的元素加入列表 v 中
                    v.append(self[i, j])
        else:
            # 如果 diagonal 不为 True，则生成对角线以下的元素
            for j in range(c):
                for i in range(j + 1, c):
                    # 将矩阵中指定位置的元素加入列表 v 中
                    v.append(self[i, j])
        # 返回生成的新矩阵对象，包含列表 v 中的元素
        return self._new(len(v), 1, v)

    def col_del(self, col):
        """Delete the specified column."""
        # 如果 col 为负数，则转换为对应正数索引
        if col < 0:
            col += self.cols
        # 检查 col 是否在合法范围内
        if not 0 <= col < self.cols:
            raise IndexError("Column {} is out of range.".format(col))
        # 调用 _eval_col_del 方法删除指定列，并返回结果
        return self._eval_col_del(col)

    def col_insert(self, pos, other):
        """Insert one or more columns at the given column position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.col_insert(1, V)
        Matrix([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])

        See Also
        ========

        col
        row_insert
        """
        # 如果矩阵为空，则直接返回 other 对象
        if not self:
            return type(self)(other)

        # 将 pos 转换为整数索引
        pos = as_int(pos)

        # 如果 pos 为负数，则转换为对应正数索引
        if pos < 0:
            pos = self.cols + pos
        # 确保 pos 在合法范围内
        if pos < 0:
            pos = 0
        elif pos > self.cols:
            pos = self.cols

        # 检查矩阵行数与 other 的行数是否相同
        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))

        # 调用 _eval_col_insert 方法在指定位置 pos 插入 other 列，并返回结果
        return self._eval_col_insert(pos, other)

    def col_join(self, other):
        """Concatenates two matrices along self's last and other's first row.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.col_join(V)
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]])

        See Also
        ========

        col
        row_join
        """
        # 如果 self 是空矩阵且列数与 other 不同，则递归调用 col_join 方法
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        # 检查两个矩阵的列数是否相同
        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))
        
        # 调用 _eval_col_join 方法连接 other 到 self，返回结果
        return self._eval_col_join(other)

    def col(self, j):
        """Elementary column selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).col(0)
        Matrix([
        [1],
        [0]])

        See Also
        ========

        row
        col_del
        col_join
        col_insert
        """
        # 返回矩阵中第 j 列的所有元素
        return self[:, j]
    def extract(self, rowsList, colsList):
        r"""Return a submatrix by specifying a list of rows and columns.
        Negative indices can be given. All indices must be in the range
        $-n \le i < n$ where $n$ is the number of rows or columns.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(4, 3, range(12))
        >>> m
        Matrix([
        [0,  1,  2],
        [3,  4,  5],
        [6,  7,  8],
        [9, 10, 11]])
        >>> m.extract([0, 1, 3], [0, 1])
        Matrix([
        [0,  1],
        [3,  4],
        [9, 10]])

        Rows or columns can be repeated:

        >>> m.extract([0, 0, 1], [-1])
        Matrix([
        [2],
        [2],
        [5]])

        Every other row can be taken by using range to provide the indices:

        >>> m.extract(range(0, m.rows, 2), [-1])
        Matrix([
        [2],
        [8]])

        RowsList or colsList can also be a list of booleans, in which case
        the rows or columns corresponding to the True values will be selected:

        >>> m.extract([0, 1, 2, 3], [True, False, True])
        Matrix([
        [0,  2],
        [3,  5],
        [6,  8],
        [9, 11]])
        """

        # 检查输入参数是否为可迭代对象
        if not is_sequence(rowsList) or not is_sequence(colsList):
            raise TypeError("rowsList and colsList must be iterable")
        
        # 如果 rowsList 或 colsList 全为布尔值，则转换为索引列表
        if rowsList and all(isinstance(i, bool) for i in rowsList):
            rowsList = [index for index, item in enumerate(rowsList) if item]
        if colsList and all(isinstance(i, bool) for i in colsList):
            colsList = [index for index, item in enumerate(colsList) if item]

        # 确保所有索引在有效范围内
        rowsList = [a2idx(k, self.rows) for k in rowsList]
        colsList = [a2idx(k, self.cols) for k in colsList]

        # 调用 _eval_extract 方法获取子矩阵
        return self._eval_extract(rowsList, colsList)

    def get_diag_blocks(self):
        """Obtains the square sub-matrices on the main diagonal of a square matrix.

        Useful for inverting symbolic matrices or solving systems of
        linear equations which may be decoupled by having a block diagonal
        structure.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y, z
        >>> A = Matrix([[1, 3, 0, 0], [y, z*z, 0, 0], [0, 0, x, 0], [0, 0, 0, 0]])
        >>> a1, a2, a3 = A.get_diag_blocks()
        >>> a1
        Matrix([
        [1,    3],
        [y, z**2]])
        >>> a2
        Matrix([[x]])
        >>> a3
        Matrix([[0]])

        """
        # 调用 _eval_get_diag_blocks 方法获取主对角线上的方块矩阵
        return self._eval_get_diag_blocks()

    @classmethod
    def hstack(cls, *args):
        """Return a matrix formed by joining args horizontally (i.e.
        by repeated application of row_join).

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> Matrix.hstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]])
        """
        # 如果没有输入参数，则返回一个空的矩阵对象
        if len(args) == 0:
            return cls._new()

        # 取第一个参数的类型作为类对象
        kls = type(args[0])
        # 将所有输入的矩阵按行连接成一个新的矩阵
        return reduce(kls.row_join, args)

    def reshape(self, rows, cols):
        """Reshape the matrix. Total number of elements must remain the same.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 3, lambda i, j: 1)
        >>> m
        Matrix([
        [1, 1, 1],
        [1, 1, 1]])
        >>> m.reshape(1, 6)
        Matrix([[1, 1, 1, 1, 1, 1]])
        >>> m.reshape(3, 2)
        Matrix([
        [1, 1],
        [1, 1],
        [1, 1]])

        """
        # 检查重塑后的行列数是否与原矩阵元素总数相等
        if self.rows * self.cols != rows * cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        # 返回重塑后的新矩阵对象
        return self._new(rows, cols, lambda i, j: self[i * cols + j])

    def row_del(self, row):
        """Delete the specified row."""
        # 如果指定的行数为负数，将其转换为正数
        if row < 0:
            row += self.rows
        # 如果行数超出矩阵范围，抛出索引错误异常
        if not 0 <= row < self.rows:
            raise IndexError("Row {} is out of range.".format(row))

        # 调用内部方法删除指定行，并返回新矩阵对象
        return self._eval_row_del(row)

    def row_insert(self, pos, other):
        """Insert one or more rows at the given row position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.row_insert(1, V)
        Matrix([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]])

        See Also
        ========

        row
        col_insert
        """
        # 如果当前矩阵为空矩阵，则直接返回新的矩阵对象
        if not self:
            return self._new(other)

        # 将位置参数转换为整数
        pos = as_int(pos)

        # 如果位置为负数，根据矩阵行数进行调整
        if pos < 0:
            pos = self.rows + pos
        if pos < 0:
            pos = 0
        elif pos > self.rows:
            pos = self.rows

        # 检查要插入的矩阵行数是否与当前矩阵列数相同
        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))

        # 调用内部方法，在指定位置插入行，并返回新的矩阵对象
        return self._eval_row_insert(pos, other)
    def row_join(self, other):
        """Concatenates two matrices along self's last and rhs's first column

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.row_join(V)
        Matrix([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]])

        See Also
        ========

        row
        col_join
        """
        # 如果 self 的列数为 0 且行数不等于 other 的行数，则可以堆叠空矩阵 (见 #10770)
        if self.cols == 0 and self.rows != other.rows:
            return self._new(other.rows, 0, []).row_join(other)

        # 如果 self 和 other 的行数不相等，抛出形状错误异常
        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))
        
        # 调用 _eval_row_join 方法执行行连接操作
        return self._eval_row_join(other)

    def diagonal(self, k=0):
        """Returns the kth diagonal of self. The main diagonal
        corresponds to `k=0`; diagonals above and below correspond to
        `k > 0` and `k < 0`, respectively. The values of `self[i, j]`
        for which `j - i = k`, are returned in order of increasing
        `i + j`, starting with `i + j = |k|`.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(3, 3, lambda i, j: j - i); m
        Matrix([
        [ 0,  1, 2],
        [-1,  0, 1],
        [-2, -1, 0]])
        >>> _.diagonal()
        Matrix([[0, 0, 0]])
        >>> m.diagonal(1)
        Matrix([[1, 1]])
        >>> m.diagonal(-2)
        Matrix([[-2]])

        Even though the diagonal is returned as a Matrix, the element
        retrieval can be done with a single index:

        >>> Matrix.diag(1, 2, 3).diagonal()[1]  # instead of [0, 1]
        2

        See Also
        ========

        diag
        """
        rv = []
        k = as_int(k)
        r = 0 if k > 0 else -k
        c = 0 if r else k
        while True:
            if r == self.rows or c == self.cols:
                break
            # 将符合条件的对角线元素加入 rv 列表
            rv.append(self[r, c])
            r += 1
            c += 1
        # 如果 rv 为空，则抛出值错误异常
        if not rv:
            raise ValueError(filldedent('''
            The %s diagonal is out of range [%s, %s]''' % (
            k, 1 - self.rows, self.cols - 1)))
        # 返回一个新的 Matrix 对象，包含 rv 列表中的元素
        return self._new(1, len(rv), rv)

    def row(self, i):
        """Elementary row selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).row(0)
        Matrix([[1, 0]])

        See Also
        ========

        col
        row_del
        row_join
        row_insert
        """
        # 返回矩阵 self 的第 i 行
        return self[i, :]

    @property
    def shape(self):
        """The shape (dimensions) of the matrix as the 2-tuple (rows, cols).

        Examples
        ========

        >>> from sympy import zeros
        >>> M = zeros(2, 3)
        >>> M.shape
        (2, 3)
        >>> M.rows
        2
        >>> M.cols
        3
        """
        # 返回矩阵的行数和列数的元组形式
        return (self.rows, self.cols)
    def todok(self):
        """Return the matrix as dictionary of keys.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix.eye(3)
        >>> M.todok()
        {(0, 0): 1, (1, 1): 1, (2, 2): 1}
        """
        # 调用内部方法 _eval_todok() 来获取将矩阵转换为字典的结果
        return self._eval_todok()

    def tolist(self):
        """Return the Matrix as a nested Python list.

        Examples
        ========

        >>> from sympy import Matrix, ones
        >>> m = Matrix(3, 3, range(9))
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
        >>> m.tolist()
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> ones(3, 0).tolist()
        [[], [], []]

        When there are no rows then it will not be possible to tell how
        many columns were in the original matrix:

        >>> ones(0, 3).tolist()
        []

        """
        # 如果矩阵没有行，则返回空列表
        if not self.rows:
            return []
        # 如果矩阵没有列，则返回包含空列表的列表，数量等于行数
        if not self.cols:
            return [[] for i in range(self.rows)]
        # 调用内部方法 _eval_tolist() 来获取将矩阵转换为列表的结果
        return self._eval_tolist()

    def todod(M):
        """Returns matrix as dict of dicts containing non-zero elements of the Matrix

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1],[0, 3]])
        >>> A
        Matrix([
        [0, 1],
        [0, 3]])
        >>> A.todod()
        {0: {1: 1}, 1: {1: 3}}


        """
        # 初始化一个空字典，用于存储结果
        rowsdict = {}
        # 将矩阵 M 转换为列表
        Mlol = M.tolist()
        # 遍历转换后的列表，构建字典结构
        for i, Mi in enumerate(Mlol):
            # 从每一行 Mi 中筛选出非零元素，构建行的字典表示
            row = {j: Mij for j, Mij in enumerate(Mi) if Mij}
            # 如果该行字典不为空，则加入到结果字典中
            if row:
                rowsdict[i] = row
        return rowsdict

    def vec(self):
        """Return the Matrix converted into a one column matrix by stacking columns

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 3], [2, 4]])
        >>> m
        Matrix([
        [1, 3],
        [2, 4]])
        >>> m.vec()
        Matrix([
        [1],
        [2],
        [3],
        [4]])

        See Also
        ========

        vech
        """
        # 调用内部方法 _eval_vec() 来获取将矩阵转换为列向量的结果
        return self._eval_vec()
    def vech(self, diagonal=True, check_symmetry=True):
        """Reshapes the matrix into a column vector by stacking the
        elements in the lower triangle.

        Parameters
        ==========

        diagonal : bool, optional
            If ``True``, it includes the diagonal elements.

        check_symmetry : bool, optional
            If ``True``, it checks whether the matrix is symmetric.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 2], [2, 3]])
        >>> m
        Matrix([
        [1, 2],
        [2, 3]])
        >>> m.vech()
        Matrix([
        [1],
        [2],
        [3]])
        >>> m.vech(diagonal=False)
        Matrix([[2]])

        Notes
        =====

        This should work for symmetric matrices and ``vech`` can
        represent symmetric matrices in vector form with less size than
        ``vec``.

        See Also
        ========

        vec
        """
        # 检查是否为方阵，如果不是则抛出异常
        if not self.is_square:
            raise NonSquareMatrixError

        # 如果需要检查对称性并且矩阵不对称，则抛出异常
        if check_symmetry and not self.is_symmetric():
            raise ValueError("The matrix is not symmetric.")

        # 调用内部方法计算并返回 vech 表示的向量
        return self._eval_vech(diagonal)

    @classmethod
    def vstack(cls, *args):
        """Return a matrix formed by joining args vertically (i.e.
        by repeated application of col_join).

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> Matrix.vstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2]])
        """
        # 如果没有参数，则返回一个空矩阵
        if len(args) == 0:
            return cls._new()

        # 获取第一个参数的类，并通过 col_join 方法逐个连接参数
        kls = type(args[0])
        return reduce(kls.col_join, args)
    @classmethod
    def _eval_diag(cls, rows, cols, diag_dict):
        """Return a diagonal matrix based on given diagonal entries.

        Parameters
        ----------
        rows : int
            Number of rows in the matrix.
        cols : int
            Number of columns in the matrix.
        diag_dict : defaultdict
            Dictionary containing entries for the diagonal of the matrix.

        Returns
        -------
        MatrixSpecial
            Diagonal matrix with specified entries.
        """
        def entry(i, j):
            return diag_dict[(i, j)]
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_eye(cls, rows, cols):
        """Return an identity matrix.

        Parameters
        ----------
        rows : int
            Number of rows in the matrix.
        cols : int
            Number of columns in the matrix.

        Returns
        -------
        MatrixSpecial
            Identity matrix of size rows x cols.
        """
        vals = [cls.zero]*(rows*cols)
        vals[::cols+1] = [cls.one]*min(rows, cols)
        return cls._new(rows, cols, vals, copy=False)

    @classmethod
    def _eval_jordan_block(cls, size: int, eigenvalue, band='upper'):
        """Return a Jordan block matrix.

        Parameters
        ----------
        size : int
            Size of the Jordan block matrix (size x size).
        eigenvalue : object
            Value to fill along the main diagonal of the Jordan block.
        band : str, optional
            Determines if the matrix is upper or lower triangular ('upper' by default).

        Returns
        -------
        MatrixSpecial
            Jordan block matrix of specified size and band type.
        """
        if band == 'lower':
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif j + 1 == i:
                    return cls.one
                return cls.zero
        else:
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif i + 1 == j:
                    return cls.one
                return cls.zero
        return cls._new(size, size, entry)

    @classmethod
    def _eval_ones(cls, rows, cols):
        """Return a matrix filled with ones.

        Parameters
        ----------
        rows : int
            Number of rows in the matrix.
        cols : int
            Number of columns in the matrix.

        Returns
        -------
        MatrixSpecial
            Matrix filled with ones of size rows x cols.
        """
        def entry(i, j):
            return cls.one
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_zeros(cls, rows, cols):
        """Return a matrix filled with zeros.

        Parameters
        ----------
        rows : int
            Number of rows in the matrix.
        cols : int
            Number of columns in the matrix.

        Returns
        -------
        MatrixSpecial
            Matrix filled with zeros of size rows x cols.
        """
        return cls._new(rows, cols, [cls.zero]*(rows*cols), copy=False)

    @classmethod
    def _eval_wilkinson(cls, n):
        """Return Wilkinson matrices.

        Parameters
        ----------
        n : int
            Size of the Wilkinson matrices.

        Returns
        -------
        tuple
            Two Wilkinson matrices, wminus and wplus.
        """
        def entry(i, j):
            return cls.one if i + 1 == j else cls.zero

        D = cls._new(2*n + 1, 2*n + 1, entry)

        wminus = cls.diag(list(range(-n, n + 1)), unpack=True) + D + D.T
        wplus = abs(cls.diag(list(range(-n, n + 1)), unpack=True)) + D + D.T

        return wminus, wplus

    @classmethod
    def eye(cls, rows, cols=None, **kwargs):
        """Returns an identity matrix.

        Parameters
        ----------
        rows : int
            Number of rows in the matrix.
        cols : int, optional
            Number of columns in the matrix (if None, cols=rows).

        kwargs
        ------
        cls : class, optional
            Class of the returned matrix.

        Returns
        -------
        MatrixSpecial
            Identity matrix of size rows x cols.
        """
        if cols is None:
            cols = rows
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        klass = kwargs.get('cls', cls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_eye(rows, cols)
    def jordan_block(kls, size=None, eigenvalue=None, *, band='upper', **kwargs):
        """Returns a Jordan block
        
        Parameters
        ==========

        size : Integer, optional
            Specifies the shape of the Jordan block matrix.

        eigenvalue : Number or Symbol
            Specifies the value for the main diagonal of the matrix.

            .. note::
                The keyword ``eigenval`` is also specified as an alias
                of this keyword, but it is not recommended to use.

                We may deprecate the alias in later release.

        band : 'upper' or 'lower', optional
            Specifies the position of the off-diagonal to put `1` s on.

        cls : Matrix, optional
            Specifies the matrix class of the output form.

            If it is not specified, the class type where the method is
            being executed on will be returned.

        Returns
        =======

        Matrix
            A Jordan block matrix.

        Raises
        ======

        ValueError
            If insufficient arguments are given for matrix size
            specification, or no eigenvalue is given.

        Examples
        ========

        Creating a default Jordan block:

        >>> from sympy import Matrix
        >>> from sympy.abc import x
        >>> Matrix.jordan_block(4, x)
        Matrix([
        [x, 1, 0, 0],
        [0, x, 1, 0],
        [0, 0, x, 1],
        [0, 0, 0, x]])

        Creating an alternative Jordan block matrix where `1` is on
        lower off-diagonal:

        >>> Matrix.jordan_block(4, x, band='lower')
        Matrix([
        [x, 0, 0, 0],
        [1, x, 0, 0],
        [0, 1, x, 0],
        [0, 0, 1, x]])

        Creating a Jordan block with keyword arguments

        >>> Matrix.jordan_block(size=4, eigenvalue=x)
        Matrix([
        [x, 1, 0, 0],
        [0, x, 1, 0],
        [0, 0, x, 1],
        [0, 0, 0, x]])

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Jordan_matrix
        """
        klass = kwargs.pop('cls', kls)  # 获取关键字参数中的 cls，如果不存在则使用 kls

        eigenval = kwargs.get('eigenval', None)  # 获取关键字参数中的 eigenval，如果不存在则为 None
        if eigenvalue is None and eigenval is None:
            raise ValueError("Must supply an eigenvalue")  # 如果没有提供 eigenvalue 或 eigenval，则引发 ValueError
        elif eigenvalue != eigenval and None not in (eigenval, eigenvalue):
            raise ValueError(
                "Inconsistent values are given: 'eigenval'={}, "
                "'eigenvalue'={}".format(eigenval, eigenvalue))  # 如果 eigenvalue 和 eigenval 不一致，引发 ValueError
        else:
            if eigenval is not None:
                eigenvalue = eigenval  # 如果 eigenval 存在，则使用它来更新 eigenvalue

        if size is None:
            raise ValueError("Must supply a matrix size")  # 如果没有提供 size，则引发 ValueError

        size = as_int(size)  # 将 size 转换为整数类型
        return klass._eval_jordan_block(size, eigenvalue, band)  # 调用 klass 类型的 _eval_jordan_block 方法返回 Jordan 块矩阵

    @classmethod
    @classmethod
    def companion(kls, poly):
        """Returns a companion matrix of a polynomial.

        Examples
        ========

        >>> from sympy import Matrix, Poly, Symbol, symbols
        >>> x = Symbol('x')
        >>> c0, c1, c2, c3, c4 = symbols('c0:5')
        >>> p = Poly(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + x**5, x)
        >>> Matrix.companion(p)
        Matrix([
        [0, 0, 0, 0, -c0],
        [1, 0, 0, 0, -c1],
        [0, 1, 0, 0, -c2],
        [0, 0, 1, 0, -c3],
        [0, 0, 0, 1, -c4]])
        """
        # 将输入的多项式转化为 sympy 的 Poly 对象
        poly = kls._sympify(poly)
        # 检查输入的 poly 是否为 Poly 类型
        if not isinstance(poly, Poly):
            raise ValueError("{} must be a Poly instance.".format(poly))
        # 检查输入的多项式是否为首一多项式（leading coefficient 为 1）
        if not poly.is_monic:
            raise ValueError("{} must be a monic polynomial.".format(poly))
        # 检查输入的多项式是否为一元多项式
        if not poly.is_univariate:
            raise ValueError(
                "{} must be a univariate polynomial.".format(poly))

        # 获取多项式的次数（矩阵的大小）
        size = poly.degree()
        # 检查多项式的次数是否不小于 1
        if not size >= 1:
            raise ValueError(
                "{} must have degree not less than 1.".format(poly))

        # 获取多项式的系数列表
        coeffs = poly.all_coeffs()

        # 定义矩阵元素的函数
        def entry(i, j):
            # 如果 j 是 size-1，则返回对应位置的负最后一个系数
            if j == size - 1:
                return -coeffs[-1 - i]
            # 如果 i 是 j+1，则返回 kls 类的单位元素
            elif i == j + 1:
                return kls.one
            # 否则返回 kls 类的零元素
            return kls.zero
        
        # 使用定义的 entry 函数创建一个新的 companion 矩阵
        return kls._new(size, size, entry)
    def wilkinson(kls, n, **kwargs):
        """Returns two square Wilkinson Matrix of size 2*n + 1
        $W_{2n + 1}^-, W_{2n + 1}^+ =$ Wilkinson(n)

        Examples
        ========

        >>> from sympy import Matrix
        >>> wminus, wplus = Matrix.wilkinson(3)
        >>> wminus
        Matrix([
        [-3,  1,  0, 0, 0, 0, 0],
        [ 1, -2,  1, 0, 0, 0, 0],
        [ 0,  1, -1, 1, 0, 0, 0],
        [ 0,  0,  1, 0, 1, 0, 0],
        [ 0,  0,  0, 1, 1, 1, 0],
        [ 0,  0,  0, 0, 1, 2, 1],
        [ 0,  0,  0, 0, 0, 1, 3]])
        >>> wplus
        Matrix([
        [3, 1, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 1, 3]])

        References
        ==========

        .. [1] https://blogs.mathworks.com/cleve/2013/04/15/wilkinsons-matrices-2/
        .. [2] J. H. Wilkinson, The Algebraic Eigenvalue Problem, Claredon Press, Oxford, 1965, 662 pp.

        """
        # 获取参数中的类名，默认为给定的 kls
        klass = kwargs.get('cls', kls)
        # 将 n 转换为整数
        n = as_int(n)
        # 调用类的方法 _eval_wilkinson 返回 Wilkinsons 矩阵
        return klass._eval_wilkinson(n)
class MatrixProperties(MatrixRequired):
    """Provides basic properties of a matrix."""

    # 返回所有元素的原子
    def _eval_atoms(self, *types):
        result = set()
        for i in self:
            result.update(i.atoms(*types))
        return result

    # 返回所有自由符号
    def _eval_free_symbols(self):
        return set().union(*(i.free_symbols for i in self if i))

    # 判断是否存在给定模式
    def _eval_has(self, *patterns):
        return any(a.has(*patterns) for a in self)

    # 判断是否为反对称矩阵
    def _eval_is_anti_symmetric(self, simpfunc):
        if not all(simpfunc(self[i, j] + self[j, i]).is_zero for i in range(self.rows) for j in range(self.cols)):
            return False
        return True

    # 判断是否为对角矩阵
    def _eval_is_diagonal(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self[i, j]:
                    return False
        return True

    # 判断是否为埃尔米特矩阵
    # 注意名称以避免与通用SymPy例程冲突
    def _eval_is_matrix_hermitian(self, simpfunc):
        mat = self._new(self.rows, self.cols, lambda i, j: simpfunc(self[i, j] - self[j, i].conjugate()))
        return mat.is_zero_matrix

    # 判断是否为单位矩阵
    def _eval_is_Identity(self) -> FuzzyBool:
        def dirac(i, j):
            if i == j:
                return 1
            return 0

        return all(self[i, j] == dirac(i, j)
                   for i in range(self.rows)
                   for j in range(self.cols))

    # 判断是否为下Hessenberg矩阵
    def _eval_is_lower_hessenberg(self):
        return all(self[i, j].is_zero
                   for i in range(self.rows)
                   for j in range(i + 2, self.cols))

    # 判断是否为下三角矩阵
    def _eval_is_lower(self):
        return all(self[i, j].is_zero
                   for i in range(self.rows)
                   for j in range(i + 1, self.cols))

    # 判断是否为符号矩阵
    def _eval_is_symbolic(self):
        return self.has(Symbol)

    # 判断是否为对称矩阵
    def _eval_is_symmetric(self, simpfunc):
        mat = self._new(self.rows, self.cols, lambda i, j: simpfunc(self[i, j] - self[j, i]))
        return mat.is_zero_matrix

    # 判断是否为零矩阵
    def _eval_is_zero_matrix(self):
        if any(i.is_zero == False for i in self):
            return False
        if any(i.is_zero is None for i in self):
            return None
        return True

    # 判断是否为上Hessenberg矩阵
    def _eval_is_upper_hessenberg(self):
        return all(self[i, j].is_zero
                   for i in range(2, self.rows)
                   for j in range(min(self.cols, (i - 1))))

    # 返回所有非零元素
    def _eval_values(self):
        return [i for i in self if not i.is_zero]

    # 判断是否具有正对角线元素
    def _has_positive_diagonals(self):
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_positive for x in diagonal_entries)

    # 判断是否具有非负对角线元素
    def _has_nonnegative_diagonals(self):
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_nonnegative for x in diagonal_entries)
    def atoms(self, *types):
        """Returns the atoms that form the current object.

        Examples
        ========

        >>> from sympy.abc import x, y  # 导入符号变量 x, y
        >>> from sympy import Matrix  # 导入 Matrix 类
        >>> Matrix([[x]])  # 创建一个包含 x 的 Matrix 对象
        Matrix([[x]])
        >>> _.atoms()  # 调用 atoms 方法返回 Matrix 中的原子集合
        {x}
        >>> Matrix([[x, y], [y, x]])  # 创建一个包含 x 和 y 的 Matrix 对象
        Matrix([
        [x, y],
        [y, x]])
        >>> _.atoms()  # 调用 atoms 方法返回 Matrix 中的原子集合
        {x, y}
        """

        types = tuple(t if isinstance(t, type) else type(t) for t in types)  # 确保 types 中的每个元素都是类型对象
        if not types:
            types = (Atom,)  # 如果 types 为空，则默认为 Atom 类型
        return self._eval_atoms(*types)  # 调用内部方法 _eval_atoms 返回符合指定类型的原子集合

    @property
    def free_symbols(self):
        """Returns the free symbols within the matrix.

        Examples
        ========

        >>> from sympy.abc import x  # 导入符号变量 x
        >>> from sympy import Matrix  # 导入 Matrix 类
        >>> Matrix([[x], [1]]).free_symbols  # 创建一个包含 x 和 1 的 Matrix 对象，并返回其中的自由符号
        {x}
        """
        return self._eval_free_symbols()  # 调用内部方法 _eval_free_symbols 返回矩阵中的自由符号集合

    def has(self, *patterns):
        """Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import Matrix, SparseMatrix, Float  # 导入 Matrix、SparseMatrix 和 Float 类
        >>> from sympy.abc import x, y  # 导入符号变量 x, y
        >>> A = Matrix(((1, x), (0.2, 3)))  # 创建一个 Matrix 对象 A
        >>> B = SparseMatrix(((1, x), (0.2, 3)))  # 创建一个 SparseMatrix 对象 B
        >>> A.has(x)  # 测试 A 中是否包含符号变量 x
        True
        >>> A.has(y)  # 测试 A 中是否包含符号变量 y
        False
        >>> A.has(Float)  # 测试 A 中是否包含 Float 类型的元素
        True
        >>> B.has(x)  # 测试 B 中是否包含符号变量 x
        True
        >>> B.has(y)  # 测试 B 中是否包含符号变量 y
        False
        >>> B.has(Float)  # 测试 B 中是否包含 Float 类型的元素
        True
        """
        return self._eval_has(*patterns)  # 调用内部方法 _eval_has 测试是否有子表达式与给定模式匹配
    def is_anti_symmetric(self, simplify=True):
        """
        Check if matrix M is an antisymmetric matrix,
        that is, M is a square matrix with all M[i, j] == -M[j, i].

        When ``simplify=True`` (default), the sum M[i, j] + M[j, i] is
        simplified before testing to see if it is zero. By default,
        the SymPy simplify function is used. To use a custom function
        set simplify to a function that accepts a single argument which
        returns a simplified expression. To skip simplification, set
        simplify to False but note that although this will be faster,
        it may induce false negatives.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> m = Matrix(2, 2, [0, 1, -1, 0])
        >>> m
        Matrix([
        [ 0, 1],
        [-1, 0]])
        >>> m.is_anti_symmetric()
        True
        >>> x, y = symbols('x y')
        >>> m = Matrix(2, 3, [0, 0, x, -y, 0, 0])
        >>> m
        Matrix([
        [ 0, 0, x],
        [-y, 0, 0]])
        >>> m.is_anti_symmetric()
        False

        >>> from sympy.abc import x, y
        >>> m = Matrix(3, 3, [0, x**2 + 2*x + 1, y,
        ...                   -(x + 1)**2, 0, x*y,
        ...                   -y, -x*y, 0])

        Simplification of matrix elements is done by default so even
        though two elements which should be equal and opposite would not
        pass an equality test, the matrix is still reported as
        anti-symmetric:

        >>> m[0, 1] == -m[1, 0]
        False
        >>> m.is_anti_symmetric()
        True

        If ``simplify=False`` is used for the case when a Matrix is already
        simplified, this will speed things up. Here, we see that without
        simplification the matrix does not appear anti-symmetric:

        >>> print(m.is_anti_symmetric(simplify=False))
        None

        But if the matrix were already expanded, then it would appear
        anti-symmetric and simplification in the is_anti_symmetric routine
        is not needed:

        >>> m = m.expand()
        >>> m.is_anti_symmetric(simplify=False)
        True
        """

        # 接受自定义的简化函数
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _simplify if simplify else lambda x: x

        # 如果不是方阵，则返回 False
        if not self.is_square:
            return False

        # 调用内部方法来评估是否为反对称矩阵，使用指定的简化函数
        return self._eval_is_anti_symmetric(simpfunc)
    def is_diagonal(self):
        """
        检查矩阵是否为对角矩阵，
        即除了主对角线外的元素都为零的矩阵。

        Examples
        ========

        >>> from sympy import Matrix, diag
        >>> m = Matrix(2, 2, [1, 0, 0, 2])
        >>> m
        Matrix([
        [1, 0],
        [0, 2]])
        >>> m.is_diagonal()
        True

        >>> m = Matrix(2, 2, [1, 1, 0, 2])
        >>> m
        Matrix([
        [1, 1],
        [0, 2]])
        >>> m.is_diagonal()
        False

        >>> m = diag(1, 2, 3)
        >>> m
        Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])
        >>> m.is_diagonal()
        True

        See Also
        ========

        is_lower
        is_upper
        sympy.matrices.matrixbase.MatrixCommon.is_diagonalizable
        diagonalize
        """
        return self._eval_is_diagonal()

    @property
    def is_weakly_diagonally_dominant(self):
        r"""
        检验矩阵是否为行弱对角占优的。

        Explanation
        ===========

        一个 $n, n$ 矩阵 $A$ 如果满足行弱对角占优条件，则有

        .. math::
            \left|A_{i, i}\right| \ge \sum_{j = 0, j \neq i}^{n-1}
            \left|A_{i, j}\right| \quad {\text{for all }}
            i \in \{ 0, ..., n-1 \}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_weakly_diagonally_dominant
        True

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_weakly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_weakly_diagonally_dominant
        True

        Notes
        =====

        如果需要检验矩阵是否为列弱对角占优，可以在转置矩阵后进行测试。
        """
        if not self.is_square:
            return False

        rows, cols = self.shape

        def test_row(i):
            summation = self.zero
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            return (Abs(self[i, i]) - summation).is_nonnegative

        return fuzzy_and(test_row(i) for i in range(rows))
    def is_strongly_diagonally_dominant(self):
        r"""Tests if the matrix is row strongly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row strongly diagonally dominant if

        .. math::
            \left|A_{i, i}\right| > \sum_{j = 0, j \neq i}^{n-1}
            \left|A_{i, j}\right| \quad {\text{for all }}
            i \in \{ 0, ..., n-1 \}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_strongly_diagonally_dominant
        True

        Notes
        =====

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
        # 检查矩阵是否为方阵，如果不是则返回 False
        if not self.is_square:
            return False

        # 获取矩阵的行数和列数
        rows, cols = self.shape

        # 定义检测行是否强对角占优的函数
        def test_row(i):
            summation = self.zero
            # 遍历当前行的所有列，计算除对角线元素外绝对值的和
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            # 返回是否满足强对角占优条件的布尔值
            return (Abs(self[i, i]) - summation).is_positive

        # 对所有行应用检测函数，返回是否所有行都满足强对角占优条件
        return fuzzy_and(test_row(i) for i in range(rows))

    @property
    def is_hermitian(self):
        """Checks if the matrix is Hermitian.

        In a Hermitian matrix element i,j is the complex conjugate of
        element j,i.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy import I
        >>> from sympy.abc import x
        >>> a = Matrix([[1, I], [-I, 1]])
        >>> a
        Matrix([
        [ 1, I],
        [-I, 1]])
        >>> a.is_hermitian
        True
        >>> a[0, 0] = 2*I
        >>> a.is_hermitian
        False
        >>> a[0, 0] = x
        >>> a.is_hermitian
        >>> a[0, 1] = a[1, 0]*I
        >>> a.is_hermitian
        False
        """
        # 检查矩阵是否为方阵，如果不是则返回 False
        if not self.is_square:
            return False

        # 调用内部方法检查矩阵是否为 Hermitian
        return self._eval_is_matrix_hermitian(_simplify)

    @property
    def is_Identity(self) -> FuzzyBool:
        # 检查矩阵是否为方阵，如果不是则返回 False
        if not self.is_square:
            return False
        # 调用内部方法检查矩阵是否为单位矩阵
        return self._eval_is_Identity()

    @property
    def is_lower_hessenberg(self):
        r"""Checks if the matrix is in the lower-Hessenberg form.

        The lower Hessenberg matrix has zero entries
        above the first superdiagonal.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 2, 0, 0], [5, 2, 3, 0], [3, 4, 3, 7], [5, 6, 1, 1]])
        >>> a
        Matrix([
        [1, 2, 0, 0],
        [5, 2, 3, 0],
        [3, 4, 3, 7],
        [5, 6, 1, 1]])
        >>> a.is_lower_hessenberg
        True

        See Also
        ========

        is_upper_hessenberg
        is_lower
        """
        # 调用内部方法进行实际的下 Hessenberg 形式判断
        return self._eval_is_lower_hessenberg()

    @property
    def is_lower(self):
        """Check if matrix is a lower triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_lower
        True

        >>> m = Matrix(4, 3, [0, 0, 0, 2, 0, 0, 1, 4, 0, 6, 6, 5])
        >>> m
        Matrix([
        [0, 0, 0],
        [2, 0, 0],
        [1, 4, 0],
        [6, 6, 5]])
        >>> m.is_lower
        True

        >>> from sympy.abc import x, y
        >>> m = Matrix(2, 2, [x**2 + y, y**2 + x, 0, x + y])
        >>> m
        Matrix([
        [x**2 + y, x + y**2],
        [       0,    x + y]])
        >>> m.is_lower
        False

        See Also
        ========

        is_upper
        is_diagonal
        is_lower_hessenberg
        """
        # 调用内部方法进行实际的下三角矩阵判断
        return self._eval_is_lower()

    @property
    def is_square(self):
        """Checks if a matrix is square.

        A matrix is square if the number of rows equals the number of columns.
        The empty matrix is square by definition, since the number of rows and
        the number of columns are both zero.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> b = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> c = Matrix([])
        >>> a.is_square
        False
        >>> b.is_square
        True
        >>> c.is_square
        True
        """
        # 返回行数和列数是否相等的布尔值判断结果
        return self.rows == self.cols

    def is_symbolic(self):
        """Checks if any elements contain Symbols.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.is_symbolic()
        True

        """
        # 调用内部方法判断矩阵中是否包含符号变量
        return self._eval_is_symbolic()
    def is_symmetric(self, simplify=True):
        """
        Check if matrix is symmetric matrix,
        that is square matrix and is equal to its transpose.

        By default, simplifications occur before testing symmetry.
        They can be skipped using 'simplify=False'; while speeding things a bit,
        this may however induce false negatives.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [0, 1, 1, 2])
        >>> m
        Matrix([
        [0, 1],
        [1, 2]])
        >>> m.is_symmetric()
        True

        >>> m = Matrix(2, 2, [0, 1, 2, 0])
        >>> m
        Matrix([
        [0, 1],
        [2, 0]])
        >>> m.is_symmetric()
        False

        >>> m = Matrix(2, 3, [0, 0, 0, 0, 0, 0])
        >>> m
        Matrix([
        [0, 0, 0],
        [0, 0, 0]])
        >>> m.is_symmetric()
        False

        >>> from sympy.abc import x, y
        >>> m = Matrix(3, 3, [1, x**2 + 2*x + 1, y, (x + 1)**2, 2, 0, y, 0, 3])
        >>> m
        Matrix([
        [         1, x**2 + 2*x + 1, y],
        [(x + 1)**2,              2, 0],
        [         y,              0, 3]])
        >>> m.is_symmetric()
        True

        If the matrix is already simplified, you may speed-up is_symmetric()
        test by using 'simplify=False'.

        >>> bool(m.is_symmetric(simplify=False))
        False
        >>> m1 = m.expand()
        >>> m1.is_symmetric(simplify=False)
        True
        """
        # 将 simplify 参数作为函数来使用，如果不是函数则使用 _simplify 函数或者直接返回参数
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _simplify if simplify else lambda x: x

        # 如果矩阵不是方阵，则直接返回 False
        if not self.is_square:
            return False

        # 调用内部方法进行对称性检测，使用传入的简化函数
        return self._eval_is_symmetric(simpfunc)

    @property
    def is_upper_hessenberg(self):
        """
        Checks if the matrix is the upper-Hessenberg form.

        The upper hessenberg matrix has zero entries
        below the first subdiagonal.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 4, 2, 3], [3, 4, 1, 7], [0, 2, 3, 4], [0, 0, 1, 3]])
        >>> a
        Matrix([
        [1, 4, 2, 3],
        [3, 4, 1, 7],
        [0, 2, 3, 4],
        [0, 0, 1, 3]])
        >>> a.is_upper_hessenberg
        True

        See Also
        ========

        is_lower_hessenberg
        is_upper
        """
        # 调用内部方法进行上Hessenberg形式的检测
        return self._eval_is_upper_hessenberg()
    # 检查矩阵是否为上三角矩阵。即使矩阵不是方阵，也可以返回 True。
    def is_upper(self):
        """Check if matrix is an upper triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_upper
        True

        >>> m = Matrix(4, 3, [5, 1, 9, 0, 4, 6, 0, 0, 5, 0, 0, 0])
        >>> m
        Matrix([
        [5, 1, 9],
        [0, 4, 6],
        [0, 0, 5],
        [0, 0, 0]])
        >>> m.is_upper
        True

        >>> m = Matrix(2, 3, [4, 2, 5, 6, 1, 1])
        >>> m
        Matrix([
        [4, 2, 5],
        [6, 1, 1]])
        >>> m.is_upper
        False

        See Also
        ========

        is_lower
        is_diagonal
        is_upper_hessenberg
        """
        # 使用列表推导式检查每个非零元素是否在对角线下方
        return all(self[i, j].is_zero
                   for i in range(1, self.rows)
                   for j in range(min(i, self.cols)))

    @property
    def is_zero_matrix(self):
        """Checks if a matrix is a zero matrix.

        A matrix is zero if every element is zero.  A matrix need not be square
        to be considered zero.  The empty matrix is zero by the principle of
        vacuous truth.  For a matrix that may or may not be zero (e.g.
        contains a symbol), this will be None

        Examples
        ========

        >>> from sympy import Matrix, zeros
        >>> from sympy.abc import x
        >>> a = Matrix([[0, 0], [0, 0]])
        >>> b = zeros(3, 4)
        >>> c = Matrix([[0, 1], [0, 0]])
        >>> d = Matrix([])
        >>> e = Matrix([[x, 0], [0, 0]])
        >>> a.is_zero_matrix
        True
        >>> b.is_zero_matrix
        True
        >>> c.is_zero_matrix
        False
        >>> d.is_zero_matrix
        True
        >>> e.is_zero_matrix
        """
        # 调用内部方法判断矩阵是否为零矩阵
        return self._eval_is_zero_matrix()

    def values(self):
        """Return non-zero values of self."""
        # 返回矩阵中所有非零元素的值
        return self._eval_values()
class MatrixOperations(MatrixRequired):
    """Provides basic matrix shape and elementwise
    operations.  Should not be instantiated directly."""

    def _eval_adjoint(self):
        # 返回当前矩阵的共轭转置
        return self.transpose().conjugate()

    def _eval_applyfunc(self, f):
        # 对矩阵的每个元素应用函数 f，并返回新的矩阵
        out = self._new(self.rows, self.cols, [f(x) for x in self])
        return out

    def _eval_as_real_imag(self):  # type: ignore
        # 返回矩阵每个元素的实部和虚部构成的元组
        return (self.applyfunc(re), self.applyfunc(im))

    def _eval_conjugate(self):
        # 对矩阵的每个元素取共轭，并返回新的矩阵
        return self.applyfunc(lambda x: x.conjugate())

    def _eval_permute_cols(self, perm):
        # 将列的排列应用到矩阵中
        mapping = list(perm)

        def entry(i, j):
            return self[i, mapping[j]]

        return self._new(self.rows, self.cols, entry)

    def _eval_permute_rows(self, perm):
        # 将行的排列应用到矩阵中
        mapping = list(perm)

        def entry(i, j):
            return self[mapping[i], j]

        return self._new(self.rows, self.cols, entry)

    def _eval_trace(self):
        # 计算矩阵的迹（主对角线元素的和）
        return sum(self[i, i] for i in range(self.rows))

    def _eval_transpose(self):
        # 返回矩阵的转置
        return self._new(self.cols, self.rows, lambda i, j: self[j, i])

    def adjoint(self):
        """Conjugate transpose or Hermitian conjugation."""
        # 调用 _eval_adjoint 方法，返回共轭转置矩阵
        return self._eval_adjoint()

    def applyfunc(self, f):
        """Apply a function to each element of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, lambda i, j: i*2+j)
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

        # 调用 _eval_applyfunc 方法，返回应用函数后的新矩阵
        return self._eval_applyfunc(f)

    def as_real_imag(self, deep=True, **hints):
        """Returns a tuple containing the (real, imaginary) part of matrix."""
        # XXX: Ignoring deep and hints...
        # 调用 _eval_as_real_imag 方法，返回矩阵每个元素的实部和虚部构成的元组
        return self._eval_as_real_imag()

    def conjugate(self):
        """Return the by-element conjugation.

        Examples
        ========

        >>> from sympy import SparseMatrix, I
        >>> a = SparseMatrix(((1, 2 + I), (3, 4), (I, -I)))
        >>> a
        Matrix([
        [1, 2 + I],
        [3,     4],
        [I,    -I]])
        >>> a.C
        Matrix([
        [ 1, 2 - I],
        [ 3,     4],
        [-I,     I]])

        See Also
        ========

        transpose: Matrix transposition
        H: Hermite conjugation
        sympy.matrices.matrixbase.MatrixBase.D: Dirac conjugation
        """
        # 调用 _eval_conjugate 方法，返回矩阵每个元素的共轭构成的新矩阵
        return self._eval_conjugate()

    def doit(self, **hints):
        # 对矩阵中的每个元素应用 `doit` 方法，并返回新的矩阵
        return self.applyfunc(lambda x: x.doit(**hints))
    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """Apply evalf() to each element of self."""
        # 设置选项字典，用于传递给 evalf 方法的参数
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        # 对于 self 中的每个元素，调用 evalf 方法，并传递选项字典作为参数
        return self.applyfunc(lambda i: i.evalf(n, **options))

    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True,
               mul=True, log=True, multinomial=True, basic=True, **hints):
        """Apply core.function.expand to each entry of the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import Matrix
        >>> Matrix(1, 1, [x*(x+1)])
        Matrix([[x*(x + 1)]])
        >>> _.expand()
        Matrix([[x**2 + x]])

        """
        # 对于矩阵中的每个元素，调用 expand 方法，并传递相应的参数和 hints
        return self.applyfunc(lambda x: x.expand(
            deep, modulus, power_base, power_exp, mul, log, multinomial, basic,
            **hints))

    @property
    def H(self):
        """Return Hermite conjugate.

        Examples
        ========

        >>> from sympy import Matrix, I
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m
        Matrix([
        [    0],
        [1 + I],
        [    2],
        [    3]])
        >>> m.H
        Matrix([[0, 1 - I, 2, 3]])

        See Also
        ========

        conjugate: By-element conjugation
        sympy.matrices.matrixbase.MatrixBase.D: Dirac conjugation
        """
        # 返回自身的转置的共轭转置，即 Hermite 共轭
        return self.T.C

    def permute_cols(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='cols', direction=direction)``

        See Also
        ========

        permute
        """
        # 列置换的别名，调用 self.permute 方法，方向为 'cols'
        return self.permute(swaps, orientation='cols', direction=direction)

    def permute_rows(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='rows', direction=direction)``

        See Also
        ========

        permute
        """
        # 行置换的别名，调用 self.permute 方法，方向为 'rows'
        return self.permute(swaps, orientation='rows', direction=direction)

    def refine(self, assumptions=True):
        """Apply refine to each element of the matrix.

        Examples
        ========

        >>> from sympy import Symbol, Matrix, Abs, sqrt, Q
        >>> x = Symbol('x')
        >>> Matrix([[Abs(x)**2, sqrt(x**2)],[sqrt(x**2), Abs(x)**2]])
        Matrix([
        [ Abs(x)**2, sqrt(x**2)],
        [sqrt(x**2),  Abs(x)**2]])
        >>> _.refine(Q.real(x))
        Matrix([
        [  x**2, Abs(x)],
        [Abs(x),   x**2]])

        """
        # 对矩阵中的每个元素应用 refine 方法，并传递给定的假设参数
        return self.applyfunc(lambda x: refine(x, assumptions))
    def replace(self, F, G, map=False, simultaneous=True, exact=None):
        """Replaces Function F in Matrix entries with Function G.

        Examples
        ========

        >>> from sympy import symbols, Function, Matrix
        >>> F, G = symbols('F, G', cls=Function)
        >>> M = Matrix(2, 2, lambda i, j: F(i+j)) ; M
        Matrix([
        [F(0), F(1)],
        [F(1), F(2)]])
        >>> N = M.replace(F,G)
        >>> N
        Matrix([
        [G(0), G(1)],
        [G(1), G(2)]])
        """
        # 应用函数替换，将矩阵中的函数 F 替换为函数 G
        return self.applyfunc(
            lambda x: x.replace(F, G, map=map, simultaneous=simultaneous, exact=exact))

    def rot90(self, k=1):
        """Rotates Matrix by 90 degrees

        Parameters
        ==========

        k : int
            Specifies how many times the matrix is rotated by 90 degrees
            (clockwise when positive, counter-clockwise when negative).

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> A = Matrix(2, 2, symbols('a:d'))
        >>> A
        Matrix([
        [a, b],
        [c, d]])

        Rotating the matrix clockwise one time:

        >>> A.rot90(1)
        Matrix([
        [c, a],
        [d, b]])

        Rotating the matrix anticlockwise two times:

        >>> A.rot90(-2)
        Matrix([
        [d, c],
        [b, a]])
        """

        # 计算旋转次数的模 4 值，根据不同的模值进行相应的旋转操作
        mod = k%4
        if mod == 0:
            return self
        if mod == 1:
            return self[::-1, ::].T  # 顺时针旋转90度
        if mod == 2:
            return self[::-1, ::-1]  # 顺时针旋转180度
        if mod == 3:
            return self[::, ::-1].T  # 顺时针旋转270度

    def simplify(self, **kwargs):
        """Apply simplify to each element of the matrix.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, sin, cos
        >>> SparseMatrix(1, 1, [x*sin(y)**2 + x*cos(y)**2])
        Matrix([[x*sin(y)**2 + x*cos(y)**2]])
        >>> _.simplify()
        Matrix([[x]])
        """
        # 对矩阵的每个元素应用简化操作
        return self.applyfunc(lambda x: x.simplify(**kwargs))

    def subs(self, *args, **kwargs):  # should mirror core.basic.subs
        """Return a new matrix with subs applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.subs(x, y)
        Matrix([[y]])
        >>> Matrix(_).subs(y, x)
        Matrix([[x]])
        """

        # 如果只传入一个参数且为非字典、非集合、可迭代对象，则转换为列表
        if len(args) == 1 and not isinstance(args[0], (dict, set)) and iter(args[0]) and not is_sequence(args[0]):
            args = (list(args[0]),)

        # 对矩阵的每个元素应用替换操作
        return self.applyfunc(lambda x: x.subs(*args, **kwargs))
    def trace(self):
        """
        Returns the trace of a square matrix i.e. the sum of the
        diagonal elements.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.trace()
        5

        """
        # 检查矩阵是否为方阵，若不是则抛出异常
        if self.rows != self.cols:
            raise NonSquareMatrixError()
        # 调用内部方法计算矩阵的迹并返回结果
        return self._eval_trace()

    def transpose(self):
        """
        Returns the transpose of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.transpose()
        Matrix([
        [1, 3],
        [2, 4]])

        >>> from sympy import Matrix, I
        >>> m=Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m.transpose()
        Matrix([
        [    1, 3],
        [2 + I, 4]])
        >>> m.T == m.transpose()
        True

        See Also
        ========

        conjugate: By-element conjugation

        """
        # 调用内部方法返回矩阵的转置
        return self._eval_transpose()

    @property
    def T(self):
        '''Matrix transposition'''
        # 返回矩阵的转置，与 transpose 方法等价
        return self.transpose()

    @property
    def C(self):
        '''By-element conjugation'''
        # 返回矩阵的逐元素共轭
        return self.conjugate()

    def n(self, *args, **kwargs):
        """Apply evalf() to each element of self."""
        # 对矩阵的每个元素应用 evalf() 方法，并返回结果
        return self.evalf(*args, **kwargs)

    def xreplace(self, rule):  # should mirror core.basic.xreplace
        """Return a new matrix with xreplace applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.xreplace({x: y})
        Matrix([[y]])
        >>> Matrix(_).xreplace({y: x})
        Matrix([[x]])
        """
        # 对每个矩阵元素应用 xreplace 规则并返回新的矩阵
        return self.applyfunc(lambda x: x.xreplace(rule))

    def _eval_simplify(self, **kwargs):
        # XXX: We can't use self.simplify here as mutable subclasses will
        # override simplify and have it return None
        # 调用 MatrixOperations 中的 simplify 方法简化矩阵并返回结果
        return MatrixOperations.simplify(self, **kwargs)

    def _eval_trigsimp(self, **opts):
        from sympy.simplify.trigsimp import trigsimp
        # 对矩阵的每个元素应用 trigsimp 方法简化三角函数表达式并返回新的矩阵
        return self.applyfunc(lambda x: trigsimp(x, **opts))
    def upper_triangular(self, k=0):
        """
        Return the elements on and above the kth diagonal of a matrix.
        If k is not specified then simply returns upper-triangular portion
        of a matrix

        Examples
        ========

        >>> from sympy import ones
        >>> A = ones(4)
        >>> A.upper_triangular()
        Matrix([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1]])

        >>> A.upper_triangular(2)
        Matrix([
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])

        >>> A.upper_triangular(-1)
        Matrix([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1]])

        """

        def entry(i, j):
            # 返回矩阵的元素 self[i, j]，如果 i + k 小于等于 j，否则返回零元素 self.zero
            return self[i, j] if i + k <= j else self.zero

        # 使用 entry 函数构造新的矩阵，并返回结果
        return self._new(self.rows, self.cols, entry)


    def lower_triangular(self, k=0):
        """
        Return the elements on and below the kth diagonal of a matrix.
        If k is not specified then simply returns lower-triangular portion
        of a matrix

        Examples
        ========

        >>> from sympy import ones
        >>> A = ones(4)
        >>> A.lower_triangular()
        Matrix([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]])

        >>> A.lower_triangular(-2)
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0]])

        >>> A.lower_triangular(1)
        Matrix([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1]])

        """

        def entry(i, j):
            # 返回矩阵的元素 self[i, j]，如果 i + k 大于等于 j，否则返回零元素 self.zero
            return self[i, j] if i + k >= j else self.zero

        # 使用 entry 函数构造新的矩阵，并返回结果
        return self._new(self.rows, self.cols, entry)
class MatrixArithmetic(MatrixRequired):
    """提供基本的矩阵算术操作。
    不应直接实例化。"""

    _op_priority = 10.01  # 设置操作优先级为10.01

    def _eval_Abs(self):
        # 返回一个新矩阵，其元素为原矩阵各元素的绝对值
        return self._new(self.rows, self.cols, lambda i, j: Abs(self[i, j]))

    def _eval_add(self, other):
        # 返回一个新矩阵，其元素为两个矩阵对应位置元素的和
        return self._new(self.rows, self.cols,
                         lambda i, j: self[i, j] + other[i, j])

    def _eval_matrix_mul(self, other):
        # 定义矩阵乘法的处理函数
        def entry(i, j):
            # 计算第(i, j)位置的乘积
            vec = [self[i,k]*other[k,j] for k in range(self.cols)]
            try:
                return Add(*vec)  # 尝试将所有乘积相加
            except (TypeError, SympifyError):
                # 处理加法操作失败的情况
                # 一些矩阵可能不支持 `sum` 或 `Add` 操作
                # 返回一个通过 reduce 函数累加的结果
                return reduce(lambda a, b: a + b, vec)

        return self._new(self.rows, other.cols, entry)  # 返回新矩阵

    def _eval_matrix_mul_elementwise(self, other):
        # 返回一个新矩阵，其元素为两个矩阵对应位置元素的乘积
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other[i,j])

    def _eval_matrix_rmul(self, other):
        # 定义右乘矩阵的处理函数
        def entry(i, j):
            # 计算第(i, j)位置的乘积
            return sum(other[i,k]*self[k,j] for k in range(other.cols))

        return self._new(other.rows, self.cols, entry)  # 返回新矩阵

    def _eval_pow_by_recursion(self, num):
        # 使用递归方式计算矩阵的幂
        if num == 1:
            return self

        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion(num - 1)
        else:
            a = b = self._eval_pow_by_recursion(num // 2)

        return a.multiply(b)  # 返回乘积矩阵

    def _eval_pow_by_cayley(self, exp):
        # 使用 Cayley-Hamilton 定理计算矩阵的幂
        from sympy.discrete.recurrences import linrec_coeffs
        row = self.shape[0]
        p = self.charpoly()

        coeffs = (-p).all_coeffs()[1:]
        coeffs = linrec_coeffs(coeffs, exp)
        new_mat = self.eye(row)
        ans = self.zeros(row)

        for i in range(row):
            ans += coeffs[i]*new_mat
            new_mat *= self

        return ans  # 返回幂矩阵

    def _eval_pow_by_recursion_dotprodsimp(self, num, prevsimp=None):
        # 使用带简化的递归方式计算矩阵的幂
        if prevsimp is None:
            prevsimp = [True]*len(self)

        if num == 1:
            return self

        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion_dotprodsimp(num - 1,
                    prevsimp=prevsimp)
        else:
            a = b = self._eval_pow_by_recursion_dotprodsimp(num // 2,
                    prevsimp=prevsimp)

        m     = a.multiply(b, dotprodsimp=False)
        lenm  = len(m)
        elems = [None]*lenm

        for i in range(lenm):
            if prevsimp[i]:
                elems[i], prevsimp[i] = _dotprodsimp(m[i], withsimp=True)
            else:
                elems[i] = m[i]

        return m._new(m.rows, m.cols, elems)  # 返回新矩阵

    def _eval_scalar_mul(self, other):
        # 返回一个新矩阵，其元素为原矩阵各元素乘以标量 other
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other)

    def _eval_scalar_rmul(self, other):
        # 返回一个新矩阵，其元素为标量 other 乘以原矩阵各元素
        return self._new(self.rows, self.cols, lambda i, j: other*self[i,j])
    def _eval_Mod(self, other):
        # 返回一个新的矩阵，其元素为 self 和 other 逐元素进行 Mod 运算的结果
        return self._new(self.rows, self.cols, lambda i, j: Mod(self[i, j], other))

    # Python arithmetic functions
    def __abs__(self):
        """Returns a new matrix with entry-wise absolute values."""
        # 返回一个新的矩阵，其元素为 self 矩阵每个元素的绝对值
        return self._eval_Abs()

    @call_highest_priority('__radd__')
    def __add__(self, other):
        """Return self + other, raising ShapeError if shapes do not match."""
        if isinstance(other, NDimArray): # Matrix and array addition is currently not implemented
            # 如果 other 是 NDimArray 类型，则返回 Not Implemented，表示矩阵和数组的加法尚未实现
            return NotImplemented
        other = _matrixify(other)
        # 对于像矩阵的对象，可以有形状属性。这是我们的第一个完整性检查。
        if hasattr(other, 'shape'):
            if self.shape != other.shape:
                # 如果 self 和 other 的形状不匹配，则抛出 ShapeError 异常
                raise ShapeError("Matrix size mismatch: %s + %s" % (
                    self.shape, other.shape))

        # 对于真实的 SymPy 矩阵，会调用它们类的对应例程
        if getattr(other, 'is_Matrix', False):
            # 调用优先级最高的类的 _eval_add 方法
            a, b = self, other
            if a.__class__ != classof(a, b):
                b, a = a, b
            return a._eval_add(b)
        # 类似矩阵的对象可以直接传递给 CommonMatrix 的例程。
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_add(self, other)

        # 如果不是上述类型的对象，则抛出 TypeError 异常
        raise TypeError('cannot add %s and %s' % (type(self), type(other)))

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        # 返回 self 与 other 的真除法结果，等价于 self * (self.one / other)
        return self * (self.one / other)

    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        other = _matrixify(other)
        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
            # 如果 other 既不是 Matrix 类型也不是 MatrixLike 类型，则返回 Not Implemented
            return NotImplemented

        # 返回 self 和 other 矩阵相乘的结果
        return self.__mul__(other)

    def __mod__(self, other):
        # 返回一个新矩阵，其元素为 self 中每个元素对 other 取模的结果
        return self.applyfunc(lambda x: x % other)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """Return self*other where other is either a scalar or a matrix
        of compatible dimensions.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> 2*A == A*2 == Matrix([[2, 4, 6], [8, 10, 12]])
        True
        >>> B = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> A*B
        Matrix([
        [30, 36, 42],
        [66, 81, 96]])
        >>> B*A
        Traceback (most recent call last):
        ...
        ShapeError: Matrices size mismatch.
        >>>

        See Also
        ========

        matrix_multiply_elementwise
        """
        # 返回 self 和 other 矩阵相乘的结果，other 可以是标量或者兼容维度的矩阵
        return self.multiply(other)
    def multiply(self, other, dotprodsimp=None):
        """Same as __mul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """

        # 根据 dotprodsimp 参数获取是否进行中间项代数简化的布尔值，默认为 False
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)

        # 将 other 转换为矩阵形式
        other = _matrixify(other)

        # 如果 other 是类似矩阵的对象，进行形状检查
        if (hasattr(other, 'shape') and len(other.shape) == 2 and
            (getattr(other, 'is_Matrix', True) or
             getattr(other, 'is_MatrixLike', True))):
            # 检查矩阵乘法是否合法，即列数和行数匹配
            if self.shape[1] != other.shape[0]:
                raise ShapeError("Matrix size mismatch: %s * %s." % (
                    self.shape, other.shape))

        # 如果 other 是 SymPy 的矩阵对象，调用 _eval_matrix_mul 方法进行乘法运算
        if getattr(other, 'is_Matrix', False):
            m = self._eval_matrix_mul(other)
            # 如果指定了 dotprodsimp 为 True，则对结果进行点乘简化处理
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])
            return m

        # 如果 other 是类似矩阵的对象，调用 MatrixArithmetic._eval_matrix_mul 方法进行乘法运算
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_mul(self, other)

        # 如果 other 不可迭代，则进行标量乘法
        if not isinstance(other, Iterable):
            try:
                return self._eval_scalar_mul(other)
            except TypeError:
                pass

        # 如果以上情况都不符合，则返回 NotImplemented
        return NotImplemented
    # 实现右乘（右侧运算符为@）操作的特殊方法，用于矩阵与其他对象的乘法
    def __rmatmul__(self, other):
        # 将other转换为矩阵形式
        other = _matrixify(other)
        # 如果other既不是Matrix也不是MatrixLike类型，则返回NotImplemented
        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
            return NotImplemented

        # 调用自身的右乘方法__rmul__
        return self.__rmul__(other)

    # 使用装饰器设置优先级调用'__mul__'方法
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        # 调用rmultiply方法进行右乘操作
        return self.rmultiply(other)

    # 实现矩阵的右乘操作，可选是否进行代数简化
    def rmultiply(self, other, dotprodsimp=None):
        """Same as __rmul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """
        # 根据dotprodsimp参数获取是否进行中间项简化的布尔值
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        # 将other转换为矩阵形式
        other = _matrixify(other)
        
        # 如果other具有'shape'属性且为二维矩阵形式，且被识别为Matrix或MatrixLike类型
        if (hasattr(other, 'shape') and len(other.shape) == 2 and
            (getattr(other, 'is_Matrix', True) or
             getattr(other, 'is_MatrixLike', True))):
            # 检查矩阵尺寸是否匹配，若不匹配则引发ShapeError异常
            if self.shape[0] != other.shape[1]:
                raise ShapeError("Matrix size mismatch.")

        # 如果other被识别为Matrix类型，则调用_eval_matrix_rmul方法处理乘法
        if getattr(other, 'is_Matrix', False):
            m = self._eval_matrix_rmul(other)
            # 若isimpbool为True，则对结果应用代数简化
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])
            return m
        
        # 如果other被识别为MatrixLike类型，则直接调用MatrixArithmetic._eval_matrix_rmul方法处理
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_rmul(self, other)

        # 若other不可迭代，则视为标量乘法
        if not isinstance(other, Iterable):
            try:
                return self._eval_scalar_rmul(other)
            except TypeError:
                pass

        # 如果以上情况均不符合，则返回NotImplemented
        return NotImplemented

    # 使用装饰器设置优先级调用'__sub__'方法
    @call_highest_priority('__sub__')
    def __rsub__(self, a):
        # 实现右侧减法操作，即-a + self
        return (-self) + a

    # 使用装饰器设置优先级调用'__rsub__'方法
    @call_highest_priority('__rsub__')
    def __sub__(self, a):
        # 实现减法操作，即self + (-a)
        return self + (-a)
class MatrixCommon(MatrixArithmetic, MatrixOperations, MatrixProperties,
                  MatrixSpecial, MatrixShaping):
    """All common matrix operations including basic arithmetic, shaping,
    and special matrices like `zeros`, and `eye`."""
    _diff_wrt = True  # type: bool


class _MinimalMatrix:
    """Class providing the minimum functionality
    for a matrix-like object and implementing every method
    required for a `MatrixRequired`.  This class does not have everything
    needed to become a full-fledged SymPy object, but it will satisfy the
    requirements of anything inheriting from `MatrixRequired`.  If you wish
    to make a specialized matrix type, make sure to implement these
    methods and properties with the exception of `__init__` and `__repr__`
    which are included for convenience."""

    is_MatrixLike = True  # Indicates that this class behaves like a matrix
    _sympify = staticmethod(sympify)  # Method to convert objects to SymPy expressions
    _class_priority = 3  # Priority level of this class in the SymPy class hierarchy
    zero = S.Zero  # Symbolic zero value
    one = S.One  # Symbolic one value

    is_Matrix = True  # Indicates that this is a matrix
    is_MatrixExpr = False  # Indicates that this is not a matrix expression

    @classmethod
    def _new(cls, *args, **kwargs):
        """Class method to create a new instance of this class."""
        return cls(*args, **kwargs)

    def __init__(self, rows, cols=None, mat=None, copy=False):
        """Initialize the minimal matrix object with specified dimensions and data.

        Args:
            rows (int): Number of rows in the matrix.
            cols (int or None): Number of columns in the matrix. Defaults to None.
            mat (list or callable or None): Matrix data as a list of lists or a callable function.
                Defaults to None.
            copy (bool): Whether to copy the matrix data. Defaults to False.
        """
        if isfunction(mat):
            # if we passed in a function, use that to populate the indices
            mat = [mat(i, j) for i in range(rows) for j in range(cols)]
        if cols is None and mat is None:
            mat = rows
        rows, cols = getattr(mat, 'shape', (rows, cols))
        try:
            # if we passed in a list of lists, flatten it and set the size
            if cols is None and mat is None:
                mat = rows
            cols = len(mat[0])
            rows = len(mat)
            mat = [x for l in mat for x in l]
        except (IndexError, TypeError):
            pass
        self.mat = tuple(self._sympify(x) for x in mat)  # Convert matrix elements to SymPy expressions
        self.rows, self.cols = rows, cols  # Set the rows and columns attributes
        if self.rows is None or self.cols is None:
            raise NotImplementedError("Cannot initialize matrix with given parameters")
    # 定义类的特殊方法，使得对象可以通过索引访问元素
    def __getitem__(self, key):
        # 内部函数：规范化行和列的切片，确保它们不含有`None`，将整数转换为长度为1的切片
        def _normalize_slices(row_slice, col_slice):
            """Ensure that row_slice and col_slice do not have
            `None` in their arguments.  Any integers are converted
            to slices of length 1"""
            if not isinstance(row_slice, slice):
                row_slice = slice(row_slice, row_slice + 1, None)
            # 确保行切片在有效范围内
            row_slice = slice(*row_slice.indices(self.rows))

            if not isinstance(col_slice, slice):
                col_slice = slice(col_slice, col_slice + 1, None)
            # 确保列切片在有效范围内
            col_slice = slice(*col_slice.indices(self.cols))

            return (row_slice, col_slice)

        # 内部函数：将矩阵坐标(i, j)转换为_mat中对应的索引
        def _coord_to_index(i, j):
            """Return the index in _mat corresponding
            to the (i,j) position in the matrix. """
            return i * self.cols + j

        # 如果key是元组
        if isinstance(key, tuple):
            i, j = key
            # 如果i或j是切片对象，则规范化这些切片
            if isinstance(i, slice) or isinstance(j, slice):
                # 确保行列切片不含有`None`，并展开切片以适应有效范围
                i, j = _normalize_slices(i, j)

                # 获取行和列的列表
                rowsList, colsList = list(range(self.rows))[i], \
                                     list(range(self.cols))[j]
                # 生成索引生成器，用于遍历行和列的组合
                indices = (i * self.cols + j for i in rowsList for j in
                           colsList)
                # 根据生成的索引生成新的_MinimalMatrix对象
                return self._new(len(rowsList), len(colsList),
                                 [self.mat[i] for i in indices])

            # 如果key是由整数组成的元组，将其转换为数组索引
            key = _coord_to_index(i, j)
        
        # 返回_mat中对应索引的元素
        return self.mat[key]

    # 定义类的特殊方法，用于比较两个对象是否相等
    def __eq__(self, other):
        try:
            classof(self, other)
        except TypeError:
            return False
        # 检查对象形状和数据是否相等
        return (
            self.shape == other.shape and list(self) == list(other))

    # 定义类的特殊方法，返回矩阵元素总数
    def __len__(self):
        return self.rows*self.cols

    # 定义类的特殊方法，返回对象的字符串表示形式
    def __repr__(self):
        return "_MinimalMatrix({}, {}, {})".format(self.rows, self.cols,
                                                   self.mat)

    # 定义属性方法，返回对象的形状
    @property
    def shape(self):
        return (self.rows, self.cols)
class _CastableMatrix: # this is needed here ONLY FOR TESTS.
    # 这是一个用于测试的类，没有其他实际用途。
    def as_mutable(self):
        return self

    def as_immutable(self):
        return self


class _MatrixWrapper:
    """Wrapper class providing the minimum functionality for a matrix-like
    object: .rows, .cols, .shape, indexability, and iterability. CommonMatrix
    math operations should work on matrix-like objects. This one is intended for
    matrix-like objects which use the same indexing format as SymPy with respect
    to returning matrix elements instead of rows for non-tuple indexes.
    """
    
    is_Matrix     = False # needs to be here because of __getattr__
    is_MatrixLike = True

    def __init__(self, mat, shape):
        # 初始化函数，接收一个矩阵及其形状作为参数
        self.mat = mat
        self.shape = shape
        self.rows, self.cols = shape

    def __getitem__(self, key):
        # 索引操作符重载，支持不同类型的索引格式
        if isinstance(key, tuple):
            return sympify(self.mat.__getitem__(key))

        return sympify(self.mat.__getitem__((key // self.rows, key % self.cols)))

    def __iter__(self): # supports numpy.matrix and numpy.array
        # 迭代器方法，支持对矩阵元素的迭代访问
        mat = self.mat
        cols = self.cols

        return iter(sympify(mat[r, c]) for r in range(self.rows) for c in range(cols))


def _matrixify(mat):
    """If `mat` is a Matrix or is matrix-like,
    return a Matrix or MatrixWrapper object.  Otherwise
    `mat` is passed through without modification."""
    
    if getattr(mat, 'is_Matrix', False) or getattr(mat, 'is_MatrixLike', False):
        return mat

    if not(getattr(mat, 'is_Matrix', True) or getattr(mat, 'is_MatrixLike', True)):
        return mat

    shape = None

    if hasattr(mat, 'shape'): # numpy, scipy.sparse
        if len(mat.shape) == 2:
            shape = mat.shape
    elif hasattr(mat, 'rows') and hasattr(mat, 'cols'): # mpmath
        shape = (mat.rows, mat.cols)

    if shape:
        return _MatrixWrapper(mat, shape)

    return mat


def a2idx(j, n=None):
    """Return integer after making positive and validating against n."""
    # 将索引 j 转换为整数并验证其合法性，如果需要，与 n 进行比较
    if not isinstance(j, int):
        jindex = getattr(j, '__index__', None)
        if jindex is not None:
            j = jindex()
        else:
            raise IndexError("Invalid index a[%r]" % (j,))
    if n is not None:
        if j < 0:
            j += n
        if not (j >= 0 and j < n):
            raise IndexError("Index out of range: a[%s]" % (j,))
    return int(j)


def classof(A, B):
    """
    Get the type of the result when combining matrices of different types.

    Currently the strategy is that immutability is contagious.

    Examples
    ========

    >>> from sympy import Matrix, ImmutableMatrix
    >>> from sympy.matrices.matrixbase import classof
    >>> M = Matrix([[1, 2], [3, 4]]) # a Mutable Matrix
    >>> IM = ImmutableMatrix([[1, 2], [3, 4]])
    >>> classof(M, IM)
    <class 'sympy.matrices.immutable.ImmutableDenseMatrix'>
    """
    priority_A = getattr(A, '_class_priority', None)
    priority_B = getattr(B, '_class_priority', None)
    # 检查 priority_A 和 priority_B 是否都不为 None
    if None not in (priority_A, priority_B):
        # 如果 A 的类优先级大于 B 的类优先级，则返回 A 的类对象
        if A._class_priority > B._class_priority:
            return A.__class__
        else:
            # 否则返回 B 的类对象
            return B.__class__

    try:
        # 尝试导入 numpy 库
        import numpy
    except ImportError:
        # 如果导入失败，则忽略该错误，继续执行下面的代码
        pass
    else:
        # 如果 A 是 numpy 数组，则返回 B 的类对象
        if isinstance(A, numpy.ndarray):
            return B.__class__
        # 如果 B 是 numpy 数组，则返回 A 的类对象
        if isinstance(B, numpy.ndarray):
            return A.__class__

    # 如果以上条件都不满足，则抛出 TypeError 异常，指示类不兼容
    raise TypeError("Incompatible classes %s, %s" % (A.__class__, B.__class__))
```