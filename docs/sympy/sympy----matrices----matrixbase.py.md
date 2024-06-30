# `D:\src\scipysrc\sympy\sympy\matrices\matrixbase.py`

```
# 导入 defaultdict 类，创建一个可以设置默认值的字典
from collections import defaultdict
# 导入 Iterable 抽象基类，用于判断对象是否可迭代
from collections.abc import Iterable
# 导入 isfunction 函数，用于检查对象是否为函数
from inspect import isfunction
# 导入 reduce 函数，用于对可迭代对象进行累积计算
from functools import reduce

# 从 sympy.assumptions.refine 模块导入 refine 函数
from sympy.assumptions.refine import refine
# 从 sympy.core 模块导入 SympifyError 和 Add 类
from sympy.core import SympifyError, Add
# 从 sympy.core.basic 模块导入 Atom 和 Basic 类
from sympy.core.basic import Atom, Basic
# 从 sympy.core.kind 模块导入 UndefinedKind 类
from sympy.core.kind import UndefinedKind
# 从 sympy.core.numbers 模块导入 Integer 类
from sympy.core.numbers import Integer
# 从 sympy.core.mod 模块导入 Mod 类
from sympy.core.mod import Mod
# 从 sympy.core.symbol 模块导入 Symbol 和 Dummy 类
from sympy.core.symbol import Symbol, Dummy
# 从 sympy.core.sympify 模块导入 sympify 和 _sympify 函数
from sympy.core.sympify import sympify, _sympify
# 从 sympy.core.function 模块导入 diff 函数
from sympy.core.function import diff
# 从 sympy.polys 模块导入 cancel 函数
from sympy.polys import cancel
# 从 sympy.functions.elementary.complexes 模块导入 Abs、re、im 函数
from sympy.functions.elementary.complexes import Abs, re, im
# 从 sympy.printing 模块导入 sstr 函数
from sympy.printing import sstr
# 从 sympy.functions.elementary.miscellaneous 模块导入 Max、Min、sqrt 函数
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
# 从 sympy.functions.special.tensor_functions 模块导入 KroneckerDelta、LeviCivita 函数
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
# 从 sympy.core.singleton 模块导入 S 单例
from sympy.core.singleton import S
# 从 sympy.printing.defaults 模块导入 Printable 类
from sympy.printing.defaults import Printable
# 从 sympy.printing.str 模块导入 StrPrinter 类
from sympy.printing.str import StrPrinter
# 从 sympy.functions.elementary.exponential 模块导入 exp、log 函数
from sympy.functions.elementary.exponential import exp, log
# 从 sympy.functions.combinatorial.factorials 模块导入 binomial、factorial 函数
from sympy.functions.combinatorial.factorials import binomial, factorial

# 导入 mpmath 库并重命名为 mp
import mpmath as mp
# 从 collections.abc 模块导入 Callable 抽象基类
from collections.abc import Callable
# 从 sympy.utilities.iterables 模块导入 reshape 函数
from sympy.utilities.iterables import reshape
# 从 sympy.core.expr 模块导入 Expr 类
from sympy.core.expr import Expr
# 从 sympy.core.power 模块导入 Pow 类
from sympy.core.power import Pow
# 从 sympy.core.symbol 模块导入 uniquely_named_symbol 函数
from sympy.core.symbol import uniquely_named_symbol

# 从当前包的 utilities 模块导入 _dotprodsimp 和 _utilities_simplify 函数
from .utilities import _dotprodsimp, _simplify as _utilities_simplify
# 从 sympy.polys.polytools 模块导入 Poly 类
from sympy.polys.polytools import Poly
# 从 sympy.utilities.iterables 模块导入 flatten 和 is_sequence 函数
from sympy.utilities.iterables import flatten, is_sequence
# 从 sympy.utilities.misc 模块导入 as_int 和 filldedent 函数
from sympy.utilities.misc import as_int, filldedent
# 从 sympy.core.decorators 模块导入 call_highest_priority 函数
from sympy.core.decorators import call_highest_priority
# 从 sympy.core.logic 模块导入 fuzzy_and 和 FuzzyBool 类
from sympy.core.logic import fuzzy_and, FuzzyBool
# 从 sympy.tensor.array 模块导入 NDimArray 类
from sympy.tensor.array import NDimArray
# 从 sympy.utilities.iterables 模块导入 NotIterable 类
from sympy.utilities.iterables import NotIterable

# 从当前包的 utilities 模块导入 _get_intermediate_simp_bool 函数
from .utilities import _get_intermediate_simp_bool

# 从当前包的 kind 模块导入 MatrixKind 类
from .kind import MatrixKind

# 从当前包的 exceptions 模块导入 MatrixError、ShapeError、NonSquareMatrixError、NonInvertibleMatrixError 异常类
from .exceptions import (
    MatrixError, ShapeError, NonSquareMatrixError, NonInvertibleMatrixError,
)

# 从当前包的 utilities 模块导入 _iszero 和 _is_zero_after_expand_mul 函数
from .utilities import _iszero, _is_zero_after_expand_mul

# 从当前包的 determinant 模块导入一系列函数
from .determinant import (
    _find_reasonable_pivot, _find_reasonable_pivot_naive,
    _adjugate, _charpoly, _cofactor, _cofactor_matrix, _per,
    _det, _det_bareiss, _det_berkowitz, _det_bird, _det_laplace, _det_LU,
    _minor, _minor_submatrix)

# 从当前包的 reductions 模块导入一系列函数
from .reductions import _is_echelon, _echelon_form, _rank, _rref

# 从当前包的 solvers 模块导入一系列函数
from .solvers import (
    _diagonal_solve, _lower_triangular_solve, _upper_triangular_solve,
    _cholesky_solve, _LDLsolve, _LUsolve, _QRsolve, _gauss_jordan_solve,
    _pinv_solve, _cramer_solve, _solve, _solve_least_squares)

# 从当前包的 inverse 模块导入一系列函数
from .inverse import (
    _pinv, _inv_ADJ, _inv_GE, _inv_LU, _inv_CH, _inv_LDL, _inv_QR,
    _inv, _inv_block)

# 从当前包的 subspaces 模块导入一系列函数
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize

# 从当前包的 eigen 模块导入一系列函数
from .eigen import (
    _eigenvals, _eigenvects,
    _bidiagonalize, _bidiagonal_decomposition,
    _is_diagonalizable, _diagonalize,
    _is_positive_definite, _is_positive_semidefinite,
    _is_negative_definite, _is_negative_semidefinite, _is_indefinite,
    _jordan_form, _left_eigenvects, _singular_values)

# 从当前包的 decompositions 模块导入未完成的内容
from .decompositions import (
    # 导入一些线性代数相关的函数和类，用于矩阵分解和计算
    _rank_decomposition, _cholesky, _LDLdecomposition,
    _LUdecomposition, _LUdecomposition_Simple, _LUdecompositionFF,
    _singular_value_decomposition, _QRdecomposition, _upper_hessenberg_decomposition)
# 导入所需函数和类（从自定义模块.graph中导入）
from .graph import (
    _connected_components, _connected_components_decomposition,
    _strongly_connected_components, _strongly_connected_components_decomposition)

# 指定此模块在进行文档测试时所需的依赖
__doctest_requires__ = {
    ('MatrixBase.is_indefinite',
     'MatrixBase.is_positive_definite',
     'MatrixBase.is_positive_semidefinite',
     'MatrixBase.is_negative_definite',
     'MatrixBase.is_negative_semidefinite'): ['matplotlib'],
}

# 定义矩阵基类MatrixBase，继承自Printable类
class MatrixBase(Printable):
    """All common matrix operations including basic arithmetic, shaping,
    and special matrices like `zeros`, and `eye`."""

    _op_priority = 10.01  # 运算优先级

    # 为了与numpy兼容性而添加的属性
    __array_priority__ = 11

    is_Matrix = True  # 类型标记为矩阵
    _class_priority = 3  # 类的优先级
    _sympify = staticmethod(sympify)  # 静态方法，用于将对象转换为SymPy表达式
    zero = S.Zero  # 表示零的常量
    one = S.One  # 表示一的常量

    _diff_wrt = True  # 声明矩阵可微分
    rows = None  # 矩阵的行数
    cols = None  # 矩阵的列数
    _simplify = None  # 简化矩阵的选项

    @classmethod
    def _new(cls, *args, **kwargs):
        """`_new`方法必须最少支持以下调用：
        `_new(rows, cols, mat)`，其中`mat`是矩阵元素的扁平列表。"""
        raise NotImplementedError("Subclasses must implement this.")

    def __eq__(self, other):
        """判断两个矩阵是否相等的方法"""
        raise NotImplementedError("Subclasses must implement this.")

    def __getitem__(self, key):
        """获取矩阵元素的方法。
        支持整数索引、元组索引 (i, j)、切片索引或混合索引 (a, b)，
        其中a和b可以是任意组合的整数和切片。"""
        raise NotImplementedError("Subclasses must implement this.")

    @property
    def shape(self):
        """返回矩阵的形状（行数和列数）作为一个二元组。

        示例
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
        return (self.rows, self.cols)

    def _eval_col_del(self, col):
        """删除矩阵指定列的方法。

        Args:
            col (int): 要删除的列的索引。

        Returns:
            MatrixBase: 删除指定列后的新矩阵。
        """
        def entry(i, j):
            return self[i, j] if j < col else self[i, j + 1]
        return self._new(self.rows, self.cols - 1, entry)

    def _eval_col_insert(self, pos, other):
        """插入另一个矩阵到指定位置的方法。

        Args:
            pos (int): 插入的位置索引。
            other (MatrixBase): 要插入的另一个矩阵。

        Returns:
            MatrixBase: 插入后的新矩阵。
        """
        def entry(i, j):
            if j < pos:
                return self[i, j]
            elif pos <= j < pos + other.cols:
                return other[i, j - pos]
            return self[i, j - other.cols]

        return self._new(self.rows, self.cols + other.cols, entry)

    def _eval_col_join(self, other):
        """连接另一个矩阵到当前矩阵列的末尾的方法。

        Args:
            other (MatrixBase): 要连接的另一个矩阵。

        Returns:
            MatrixBase: 连接后的新矩阵。
        """
        rows = self.rows

        def entry(i, j):
            if i < rows:
                return self[i, j]
            return other[i - rows, j]

        return classof(self, other)._new(self.rows + other.rows, self.cols,
                                         entry)
    # 定义一个方法 `_eval_extract`，用于从当前对象中提取指定行列的子矩阵
    def _eval_extract(self, rowsList, colsList):
        # 将当前对象转换为列表形式
        mat = list(self)
        # 获取当前对象的列数
        cols = self.cols
        # 生成行列索引的迭代器，用于定位要提取的元素位置
        indices = (i * cols + j for i in rowsList for j in colsList)
        # 返回新的对象，其内容为提取出的子矩阵
        return self._new(len(rowsList), len(colsList),
                         [mat[i] for i in indices])
    
    # 定义一个方法 `_eval_get_diag_blocks`，用于获取当前对象的对角块
    def _eval_get_diag_blocks(self):
        # 初始化子块列表
        sub_blocks = []
    
        # 定义递归函数，用于查找并添加子块
        def recurse_sub_blocks(M):
            # 遍历 M 的行
            for i in range(1, M.shape[0] + 1):
                # 根据当前行 i 划分子块
                if i == 1:
                    to_the_right = M[0, i:]  # 右侧的元素
                    to_the_bottom = M[i:, 0]  # 底部的元素
                else:
                    to_the_right = M[:i, i:]  # 右侧的元素
                    to_the_bottom = M[i:, :i]  # 底部的元素
                
                # 如果右侧或底部有非零元素，则继续下一个循环
                if any(to_the_right) or any(to_the_bottom):
                    continue
                
                # 将当前子块添加到子块列表中
                sub_blocks.append(M[:i, :i])
                
                # 如果当前子块不是 M 的完整大小，则递归查找更小的子块
                if M.shape != M[:i, :i].shape:
                    recurse_sub_blocks(M[i:, i:])
                return
    
        # 调用递归函数开始查找子块
        recurse_sub_blocks(self)
        # 返回找到的所有子块
        return sub_blocks
    
    # 定义一个方法 `_eval_row_del`，用于删除当前对象的指定行
    def _eval_row_del(self, row):
        # 定义一个内部函数 entry，用于获取指定位置的元素
        def entry(i, j):
            # 如果行号 i 小于指定的行号 row，则返回当前对象的对应元素
            return self[i, j] if i < row else self[i + 1, j]
        # 返回一个新的对象，其内容为删除指定行后的矩阵
        return self._new(self.rows - 1, self.cols, entry)
    
    # 定义一个方法 `_eval_row_insert`，用于在当前对象的指定位置插入另一个对象的行
    def _eval_row_insert(self, pos, other):
        # 将当前对象转换为列表形式
        entries = list(self)
        # 计算插入位置的索引
        insert_pos = pos * self.cols
        # 在 entries 列表中插入另一个对象 other 的内容
        entries[insert_pos:insert_pos] = list(other)
        # 返回一个新的对象，其内容为插入行后的矩阵
        return self._new(self.rows + other.rows, self.cols, entries)
    
    # 定义一个方法 `_eval_row_join`，用于将当前对象与另一个对象按行连接
    def _eval_row_join(self, other):
        # 获取当前对象的列数
        cols = self.cols
    
        # 定义一个内部函数 entry，用于获取指定位置的元素
        def entry(i, j):
            # 如果列号 j 小于当前对象的列数 cols，则返回当前对象的对应元素
            if j < cols:
                return self[i, j]
            # 否则返回另一个对象的对应元素
            return other[i, j - cols]
    
        # 返回一个新的对象，其内容为按行连接后的矩阵
        return classof(self, other)._new(self.rows, self.cols + other.cols,
                                         entry)
    
    # 定义一个方法 `_eval_tolist`，将当前对象转换为列表的形式
    def _eval_tolist(self):
        # 返回一个列表，其中包含当前对象每行的列表形式
        return [list(self[i,:]) for i in range(self.rows)]
    
    # 定义一个方法 `_eval_todok`，将当前对象转换为字典的形式
    def _eval_todok(self):
        # 初始化一个空的字典
        dok = {}
        # 获取当前对象的行数和列数
        rows, cols = self.shape
        # 遍历当前对象的每个元素
        for i in range(rows):
            for j in range(cols):
                val = self[i, j]
                # 如果元素不为零，则将其添加到字典中
                if val != self.zero:
                    dok[i, j] = val
        # 返回转换后的字典
        return dok
    
    # 定义一个类方法 `_eval_from_dok`，用于从字典中恢复对象
    @classmethod
    def _eval_from_dok(cls, rows, cols, dok):
        # 初始化一个列表，用于存放对象的每个元素
        out_flat = [cls.zero] * (rows * cols)
        # 遍历字典中的每个键值对，将元素放入列表中
        for (i, j), val in dok.items():
            out_flat[i * cols + j] = val
        # 返回一个新的对象，其内容为从字典中恢复的矩阵
        return cls._new(rows, cols, out_flat)
    
    # 定义一个方法 `_eval_vec`，用于获取当前对象的列向量形式
    def _eval_vec(self):
        # 获取当前对象的行数
        rows = self.rows
    
        # 定义一个内部函数 entry，用于获取指定位置的元素
        def entry(n, _):
            # 计算元素在当前对象中的行列索引
            j = n // rows
            i = n - j * rows
            # 返回当前对象中指定位置的元素
            return self[i, j]
    
        # 返回一个新的对象，其内容为当前对象的列向量形式
        return self._new(len(self), 1, entry)
    
    # 定义一个方法 `_eval_vech`，用于获取当前对象的上三角或下三角向量形式
    def _eval_vech(self, diagonal):
        # 获取当前对象的列数
        c = self.cols
        # 初始化一个空的列表，用于存放向量中的元素
        v = []
        # 根据 diagonal 参数判断是上三角还是下三角
        if diagonal:
            # 遍历上三角区域的元素，并添加到列表中
            for j in range(c):
                for i in range(j, c):
                    v.append(self[i, j])
        else:
            # 遍历下三角区域的元素，并添加到列表中
            for j in range(c):
                for i in range(j + 1, c):
                    v.append(self[i, j])
        # 返回一个新的对象，其内容为获取的向量形式
        return self._new(len(v), 1, v)
    def col_del(self, col):
        """Delete the specified column."""
        # 如果列索引为负数，则转换为相对于末尾的正数索引
        if col < 0:
            col += self.cols
        # 检查列索引是否在有效范围内
        if not 0 <= col < self.cols:
            raise IndexError("Column {} is out of range.".format(col))
        # 调用内部方法执行列删除操作
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
        # 如果当前矩阵为空，则直接返回插入的列矩阵作为新矩阵
        if not self:
            return type(self)(other)

        # 将插入位置转换为整数
        pos = as_int(pos)

        # 处理负数位置索引，使其相对于末尾正数索引
        if pos < 0:
            pos = self.cols + pos
        # 如果位置索引小于0，则调整为0
        if pos < 0:
            pos = 0
        # 如果位置索引大于当前列数，则调整为当前列数
        elif pos > self.cols:
            pos = self.cols

        # 检查要插入的列矩阵与当前矩阵行数是否相等
        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))

        # 调用内部方法执行列插入操作
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
        # 如果当前矩阵行数为0且列数不等于要连接的矩阵列数，则递归调用以处理空矩阵问题
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        # 检查要连接的两个矩阵列数是否相等
        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))
        
        # 调用内部方法执行列连接操作
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
        # 返回当前矩阵的指定列
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

        if not is_sequence(rowsList) or not is_sequence(colsList):
            raise TypeError("rowsList and colsList must be iterable")
        # ensure rowsList and colsList are lists of integers
        if rowsList and all(isinstance(i, bool) for i in rowsList):
            # Convert boolean list to list of indices where True
            rowsList = [index for index, item in enumerate(rowsList) if item]
        if colsList and all(isinstance(i, bool) for i in colsList):
            # Convert boolean list to list of indices where True
            colsList = [index for index, item in enumerate(colsList) if item]

        # ensure everything is in range
        # Map each index in rowsList to its valid range index
        rowsList = [a2idx(k, self.rows) for k in rowsList]
        # Map each index in colsList to its valid range index
        colsList = [a2idx(k, self.cols) for k in colsList]

        # Return the submatrix extracted based on validated rowsList and colsList
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
        # Delegate to the private method for evaluation and return its result
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
        # 如果没有输入参数，则返回空矩阵
        if len(args) == 0:
            return cls._new()

        # 获取第一个参数的类型
        kls = type(args[0])
        # 将所有参数按行连接成一个新的矩阵
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
        # 如果重塑后元素数量不同，则抛出 ValueError 异常
        if self.rows * self.cols != rows * cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        # 返回一个新的矩阵，按指定的行列数重塑
        return self._new(rows, cols, lambda i, j: self[i * cols + j])

    def row_del(self, row):
        """Delete the specified row."""
        # 如果行数为负数，则从末尾计数
        if row < 0:
            row += self.rows
        # 如果行数超出矩阵范围，则抛出 IndexError 异常
        if not 0 <= row < self.rows:
            raise IndexError("Row {} is out of range.".format(row))

        # 返回删除指定行后的新矩阵
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
        # 如果当前矩阵为空，则直接返回插入的新行或矩阵
        if not self:
            return self._new(other)

        # 将插入位置转换为整数
        pos = as_int(pos)

        # 如果插入位置为负数，则从末尾计数
        if pos < 0:
            pos = self.rows + pos
        if pos < 0:
            pos = 0
        # 如果插入位置超出矩阵范围，则置为最末尾
        elif pos > self.rows:
            pos = self.rows

        # 如果插入行的列数与当前矩阵不符，则抛出 ShapeError 异常
        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))

        # 返回在指定位置插入新行或矩阵后的新矩阵
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
        # 检查是否当前矩阵列数为0，允许空矩阵与其他矩阵连接
        if self.cols == 0 and self.rows != other.rows:
            return self._new(other.rows, 0, []).row_join(other)

        # 检查两个矩阵行数是否相同，不同则抛出形状错误
        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))
        
        # 调用实际的行连接计算方法
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
        
        # 逐行逐列遍历矩阵，提取对应主对角线元素
        while True:
            if r == self.rows or c == self.cols:
                break
            rv.append(self[r, c])
            r += 1
            c += 1
        
        # 如果未找到对应对角线元素，抛出值错误
        if not rv:
            raise ValueError(filldedent('''
            The %s diagonal is out of range [%s, %s]''' % (
            k, 1 - self.rows, self.cols - 1)))
        
        # 返回新的对角线矩阵
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
        # 返回指定索引的整行
        return self[i, :]

    def todok(self):
        """Return the matrix as dictionary of keys.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix.eye(3)
        >>> M.todok()
        {(0, 0): 1, (1, 1): 1, (2, 2): 1}
        """
        # 调用实际的转换为 dok 格式的方法
        return self._eval_todok()

    @classmethod
    def from_dok(cls, rows, cols, dok):
        """Create a matrix from a dictionary of keys.

        Examples
        ========

        >>> from sympy import Matrix
        >>> d = {(0, 0): 1, (1, 2): 3, (2, 1): 4}
        >>> Matrix.from_dok(3, 3, d)
        Matrix([
        [1, 0, 0],
        [0, 0, 3],
        [0, 4, 0]])

        将输入的字典转换成符号运算所需的格式，并调用 _eval_from_dok 方法生成矩阵对象。
        """
        dok = {ij: cls._sympify(val) for ij, val in dok.items()}
        return cls._eval_from_dok(rows, cols, dok)

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

        当矩阵没有行时，返回空列表；当矩阵没有列时，返回包含空列表的列表。
        """
        if not self.rows:
            return []
        if not self.cols:
            return [[] for i in range(self.rows)]
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

        将矩阵转换为一个包含非零元素的字典的字典格式。
        """
        rowsdict = {}
        Mlol = M.tolist()
        for i, Mi in enumerate(Mlol):
            row = {j: Mij for j, Mij in enumerate(Mi) if Mij}
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

        将矩阵按列堆叠成一个列向量。
        """
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
        # 检查矩阵是否为方阵，否则抛出异常
        if not self.is_square:
            raise NonSquareMatrixError

        # 如果需要检查对称性，并且矩阵不对称，则抛出异常
        if check_symmetry and not self.is_symmetric():
            raise ValueError("The matrix is not symmetric.")

        # 调用 _eval_vech 方法，返回变形后的列向量
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
        # 如果没有参数传入，则返回一个空的矩阵
        if len(args) == 0:
            return cls._new()

        # 获取第一个参数的类，作为创建新矩阵的类
        kls = type(args[0])
        # 使用 reduce 函数，逐个纵向连接所有参数形成新矩阵
        return reduce(kls.col_join, args)

    @classmethod
    def _eval_diag(cls, rows, cols, diag_dict):
        """diag_dict is a defaultdict containing
        all the entries of the diagonal matrix."""
        # 定义一个函数 entry，用于返回对角线元素的值
        def entry(i, j):
            return diag_dict[(i, j)]
        # 调用 _new 方法创建一个新的对角线矩阵
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_eye(cls, rows, cols):
        # 初始化一个列表 vals，表示矩阵的所有元素，默认为零
        vals = [cls.zero]*(rows*cols)
        # 将对角线位置上的元素设为 1
        vals[::cols+1] = [cls.one]*min(rows, cols)
        # 调用 _new 方法创建一个单位矩阵
        return cls._new(rows, cols, vals, copy=False)

    @classmethod
    def _eval_jordan_block(cls, size: int, eigenvalue, band='upper'):
        # 根据 band 参数的值选择不同类型的 Jordan 块
        if band == 'lower':
            # 定义一个函数 entry，返回下三角 Jordan 块的元素值
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif j + 1 == i:
                    return cls.one
                return cls.zero
        else:
            # 定义一个函数 entry，返回上三角 Jordan 块的元素值
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif i + 1 == j:
                    return cls.one
                return cls.zero
        # 调用 _new 方法创建指定类型的 Jordan 块
        return cls._new(size, size, entry)

    @classmethod
    def _eval_ones(cls, rows, cols):
        # 定义一个函数 entry，返回矩阵所有元素值为 1
        def entry(i, j):
            return cls.one
        # 调用 _new 方法创建一个所有元素为 1 的矩阵
        return cls._new(rows, cols, entry)

    @classmethod
    # 返回一个指定大小的零矩阵
    def _eval_zeros(cls, rows, cols):
        return cls._new(rows, cols, [cls.zero]*(rows*cols), copy=False)

    @classmethod
    # 返回威尔金森矩阵的两个版本：wminus 和 wplus
    def _eval_wilkinson(cls, n):
        # 定义一个函数用于生成威尔金森矩阵的每个元素
        def entry(i, j):
            return cls.one if i + 1 == j else cls.zero

        # 构造一个威尔金森矩阵 D
        D = cls._new(2*n + 1, 2*n + 1, entry)

        # 构造 wminus 和 wplus 矩阵
        wminus = cls.diag(list(range(-n, n + 1)), unpack=True) + D + D.T
        wplus = abs(cls.diag(list(range(-n, n + 1)), unpack=True)) + D + D.T

        return wminus, wplus

    @classmethod
    # 返回一个单位矩阵
    def eye(kls, rows, cols=None, **kwargs):
        """Returns an identity matrix.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        # 如果 cols 未指定，与 rows 相同
        if cols is None:
            cols = rows
        # 如果 rows 或 cols 小于零，抛出 ValueError
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        # 获取关键字参数中的类对象，或使用默认的 kls
        klass = kwargs.get('cls', kls)
        # 将 rows 和 cols 转换为整数
        rows, cols = as_int(rows), as_int(cols)

        # 调用 _eval_eye 方法创建单位矩阵并返回
        return klass._eval_eye(rows, cols)

    @classmethod
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
        # 从 kwargs 中弹出 'cls'，默认使用 kls
        klass = kwargs.pop('cls', kls)

        # 获取 'eigenval' 的值作为 eigenval 变量
        eigenval = kwargs.get('eigenval', None)
        
        # 如果 eigenvalue 和 eigenval 都为 None，则抛出 ValueError
        if eigenvalue is None and eigenval is None:
            raise ValueError("Must supply an eigenvalue")
        # 如果 eigenvalue 和 eigenval 不一致且都不为 None，则抛出 ValueError
        elif eigenvalue != eigenval and None not in (eigenval, eigenvalue):
            raise ValueError(
                "Inconsistent values are given: 'eigenval'={}, "
                "'eigenvalue'={}".format(eigenval, eigenvalue))
        else:
            # 如果 eigenval 不为 None，则将其赋值给 eigenvalue
            if eigenval is not None:
                eigenvalue = eigenval

        # 如果 size 为 None，则抛出 ValueError
        if size is None:
            raise ValueError("Must supply a matrix size")

        # 将 size 转换为整数
        size = as_int(size)
        
        # 调用类的内部方法 _eval_jordan_block 生成 Jordan 块矩阵，并返回结果
        return klass._eval_jordan_block(size, eigenvalue, band)
    @classmethod
    def ones(kls, rows, cols=None, **kwargs):
        """Returns a matrix of ones.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        # 如果未指定 cols，则设为与 rows 相同
        if cols is None:
            cols = rows
        # 获取关键字参数中的 cls，用于确定返回矩阵的类
        klass = kwargs.get('cls', kls)
        # 将 rows 和 cols 转换为整数类型
        rows, cols = as_int(rows), as_int(cols)

        # 调用类方法 _eval_ones 来生成一个全为 1 的矩阵
        return klass._eval_ones(rows, cols)

    @classmethod
    def zeros(kls, rows, cols=None, **kwargs):
        """Returns a matrix of zeros.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        # 如果未指定 cols，则设为与 rows 相同
        if cols is None:
            cols = rows
        # 检查行和列数是否为负数，如果是则抛出 ValueError 异常
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        # 获取关键字参数中的 cls，用于确定返回矩阵的类
        klass = kwargs.get('cls', kls)
        # 将 rows 和 cols 转换为整数类型
        rows, cols = as_int(rows), as_int(cols)

        # 调用类方法 _eval_zeros 来生成一个全为 0 的矩阵
        return klass._eval_zeros(rows, cols)

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
        # 将 poly 转换为符号表达式
        poly = kls._sympify(poly)
        # 检查 poly 是否为多项式类型
        if not isinstance(poly, Poly):
            raise ValueError("{} must be a Poly instance.".format(poly))
        # 检查 poly 是否为首一多项式
        if not poly.is_monic:
            raise ValueError("{} must be a monic polynomial.".format(poly))
        # 检查 poly 是否为单变量多项式
        if not poly.is_univariate:
            raise ValueError(
                "{} must be a univariate polynomial.".format(poly))

        # 获取多项式的阶数
        size = poly.degree()
        # 检查多项式阶数是否至少为 1
        if not size >= 1:
            raise ValueError(
                "{} must have degree not less than 1.".format(poly))

        # 获取多项式的所有系数
        coeffs = poly.all_coeffs()

        # 定义矩阵元素的计算方法
        def entry(i, j):
            if j == size - 1:
                return -coeffs[-1 - i]
            elif i == j + 1:
                return kls.one
            return kls.zero
        
        # 使用类方法 _new 创建一个新的 companion matrix
        return kls._new(size, size, entry)
    # 定义一个函数 wilkinson，用于生成大小为 2*n + 1 的两个 Wilkinson 矩阵
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
        # 从 kwargs 中获取 'cls'，默认值为 kls
        klass = kwargs.get('cls', kls)
        # 将 n 转换为整数类型
        n = as_int(n)
        # 调用 klass 对象的 _eval_wilkinson 方法并返回其结果
        return klass._eval_wilkinson(n)
    
    # The RepMatrix subclass uses more efficient sparse implementations of
    # _eval_iter_values and other things.
    
    # 定义 _eval_iter_values 方法，返回非零元素的迭代器
    def _eval_iter_values(self):
        return (i for i in self if i is not S.Zero)
    
    # 定义 _eval_values 方法，返回 _eval_iter_values 方法的结果列表
    def _eval_values(self):
        return list(self.iter_values())
    
    # 定义 _eval_iter_items 方法，返回矩阵的非零项及其位置的迭代器
    def _eval_iter_items(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j]:
                    yield (i, j), self[i, j]
    
    # 定义 _eval_atoms 方法，返回矩阵的原子元素集合
    def _eval_atoms(self, *types):
        # 获取矩阵的所有值
        values = self.values()
        # 如果矩阵中的非零元素数量小于总元素数，并且 S.Zero 是 types 的实例
        if len(values) < self.rows * self.cols and isinstance(S.Zero, types):
            s = {S.Zero}
        else:
            s = set()
        # 返回所有值的原子元素的集合
        return s.union(*[v.atoms(*types) for v in values])
    
    # 定义 _eval_free_symbols 方法，返回矩阵中所有元素的自由符号集合
    def _eval_free_symbols(self):
        return set().union(*(i.free_symbols for i in set(self.values())))
    
    # 定义 _eval_has 方法，判断矩阵是否包含指定模式的元素
    def _eval_has(self, *patterns):
        return any(a.has(*patterns) for a in self.iter_values())
    
    # 定义 _eval_is_symbolic 方法，判断矩阵是否包含符号变量 Symbol
    def _eval_is_symbolic(self):
        return self.has(Symbol)
    
    # _eval_is_hermitian is called by some general SymPy
    # routines and has a different *args signature.  Make
    # sure the names don't clash by adding `_matrix_` in name.
    # 定义 _eval_is_matrix_hermitian 方法，检查矩阵是否是厄米特矩阵
    def _eval_is_matrix_hermitian(self, simpfunc):
        herm = lambda i, j: simpfunc(self[i, j] - self[j, i].conjugate()).is_zero
        return fuzzy_and(herm(i, j) for (i, j), v in self.iter_items())
    
    # 定义 _eval_is_zero_matrix 方法，判断矩阵是否为零矩阵
    def _eval_is_zero_matrix(self):
        return fuzzy_and(v.is_zero for v in self.iter_values())
    
    # 定义 _eval_is_Identity 方法，判断矩阵是否为单位矩阵
    def _eval_is_Identity(self) -> FuzzyBool:
        one = self.one
        zero = self.zero
        ident = lambda i, j, v: v is one if i == j else v is zero
        return all(ident(i, j, v) for (i, j), v in self.iter_items())
    
    # 定义 _eval_is_diagonal 方法，判断矩阵是否为对角矩阵
    def _eval_is_diagonal(self):
        return fuzzy_and(v.is_zero for (i, j), v in self.iter_items() if i != j)
    def _eval_is_lower(self):
        # 检查矩阵是否为下三角矩阵
        return all(v.is_zero for (i, j), v in self.iter_items() if i < j)

    def _eval_is_upper(self):
        # 检查矩阵是否为上三角矩阵
        return all(v.is_zero for (i, j), v in self.iter_items() if i > j)

    def _eval_is_lower_hessenberg(self):
        # 检查矩阵是否为下 Hessenberg 矩阵
        return all(v.is_zero for (i, j), v in self.iter_items() if i + 1 < j)

    def _eval_is_upper_hessenberg(self):
        # 检查矩阵是否为上 Hessenberg 矩阵
        return all(v.is_zero for (i, j), v in self.iter_items() if i > j + 1)

    def _eval_is_symmetric(self, simpfunc):
        # 使用给定的简化函数检查矩阵是否对称
        sym = lambda i, j: simpfunc(self[i, j] - self[j, i]).is_zero
        return fuzzy_and(sym(i, j) for (i, j), v in self.iter_items())

    def _eval_is_anti_symmetric(self, simpfunc):
        # 使用给定的简化函数检查矩阵是否反对称
        anti = lambda i, j: simpfunc(self[i, j] + self[j, i]).is_zero
        return fuzzy_and(anti(i, j) for (i, j), v in self.iter_items())

    def _has_positive_diagonals(self):
        # 检查矩阵对角线上的元素是否都为正数
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_positive for x in diagonal_entries)

    def _has_nonnegative_diagonals(self):
        # 检查矩阵对角线上的元素是否都为非负数
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_nonnegative for x in diagonal_entries)

    def atoms(self, *types):
        """Returns the atoms that form the current object.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import Matrix
        >>> Matrix([[x]])
        Matrix([[x]])
        >>> _.atoms()
        {x}
        >>> Matrix([[x, y], [y, x]])
        Matrix([
        [x, y],
        [y, x]])
        >>> _.atoms()
        {x, y}
        """
        types = tuple(t if isinstance(t, type) else type(t) for t in types)
        if not types:
            types = (Atom,)
        return self._eval_atoms(*types)

    @property
    def free_symbols(self):
        """Returns the free symbols within the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import Matrix
        >>> Matrix([[x], [1]]).free_symbols
        {x}
        """
        return self._eval_free_symbols()

    def has(self, *patterns):
        """Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import Matrix, SparseMatrix, Float
        >>> from sympy.abc import x, y
        >>> A = Matrix(((1, x), (0.2, 3)))
        >>> B = SparseMatrix(((1, x), (0.2, 3)))
        >>> A.has(x)
        True
        >>> A.has(y)
        False
        >>> A.has(Float)
        True
        >>> B.has(x)
        True
        >>> B.has(y)
        False
        >>> B.has(Float)
        True
        """
        return self._eval_has(*patterns)
    # 检查矩阵是否为反对称矩阵的方法
    def is_anti_symmetric(self, simplify=True):
        """Check if matrix M is an antisymmetric matrix,
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
        # 接受自定义简化函数或者默认的 SymPy 简化函数
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _utilities_simplify if simplify else lambda x: x

        # 如果矩阵不是方阵，则直接返回 False
        if not self.is_square:
            return False
        # 调用内部方法来判断是否为反对称矩阵，传入简化函数
        return self._eval_is_anti_symmetric(simpfunc)
    def is_diagonal(self):
        """Check if matrix is diagonal,
        that is matrix in which the entries outside the main diagonal are all zero.

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
        sympy.matrices.matrixbase.MatrixBase.is_diagonalizable
        diagonalize
        """
        # 调用内部方法 _eval_is_diagonal() 进行实际的对角线矩阵判断
        return self._eval_is_diagonal()

    @property
    def is_weakly_diagonally_dominant(self):
        r"""Tests if the matrix is row weakly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row weakly diagonally dominant if

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

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
        # 如果矩阵不是方阵，则直接返回 False
        if not self.is_square:
            return False

        rows, cols = self.shape

        def test_row(i):
            summation = self.zero
            # 计算当前行除对角线元素外的绝对值之和
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            # 判断是否满足行弱对角线支配条件
            return (Abs(self[i, i]) - summation).is_nonnegative

        # 对所有行应用 test_row 函数，并使用 fuzzy_and 进行模糊逻辑与操作
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
        # 检查矩阵是否为方阵，如果不是，则不可能是行强对角占优的
        if not self.is_square:
            return False

        # 获取矩阵的行数和列数
        rows, cols = self.shape

        # 定义测试函数，检查第 i 行是否满足行强对角占优条件
        def test_row(i):
            summation = self.zero
            # 对第 i 行的每一列元素进行求和，除去对角线上的元素
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            # 检查是否满足行强对角占优的条件
            return (Abs(self[i, i]) - summation).is_positive

        # 对所有行应用测试函数，使用模糊逻辑与运算符
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
        # 检查矩阵是否为方阵，如果不是，则不可能是埃尔米特矩阵
        if not self.is_square:
            return False

        # 调用内部方法检查矩阵是否为埃尔米特矩阵
        return self._eval_is_matrix_hermitian(_utilities_simplify)

    @property
    def is_Identity(self) -> FuzzyBool:
        # 检查矩阵是否为方阵，如果不是，则不可能是单位矩阵
        if not self.is_square:
            return False
        # 调用内部方法检查矩阵是否为单位矩阵
        return self._eval_is_Identity()
    def is_lower_hessenberg(self):
        r"""Checks if the matrix is in the lower-Hessenberg form.

        The lower hessenberg matrix has zero entries
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
        # 返回行数和列数是否相等来判断矩阵是否为方阵
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
        # 调用内部方法检查矩阵是否包含符号变量
        return self._eval_is_symbolic()
    def is_symmetric(self, simplify=True):
        """Check if matrix is symmetric matrix,
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
        # 定义简化函数，默认为输入的 simplify 参数，如果 simplify 不是函数则使用默认简化函数
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _utilities_simplify if simplify else lambda x: x

        # 如果矩阵不是方阵，则不可能是对称矩阵，返回 False
        if not self.is_square:
            return False

        # 调用内部方法 _eval_is_symmetric() 进行实际的对称性检查，传入简化函数
        return self._eval_is_symmetric(simpfunc)

    @property
    def is_upper_hessenberg(self):
        """Checks if the matrix is the upper-Hessenberg form.

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
        # 调用内部方法 _eval_is_upper_hessenberg() 进行实际的上Hessenberg形式检查
        return self._eval_is_upper_hessenberg()
    def is_upper(self):
        """检查矩阵是否为上三角矩阵。即使矩阵不是方阵，也可以返回True。

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
        return self._eval_is_upper()

    @property
    def is_zero_matrix(self):
        """检查矩阵是否为零矩阵。

        如果矩阵的每个元素都是零，则该矩阵被认为是零矩阵。矩阵不需要是方阵。空矩阵也被视为零矩阵，
        根据真空真理原理。对于可能或可能不是零矩阵（例如包含符号的矩阵），返回None。

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
        return self._eval_is_zero_matrix()

    def values(self):
        """返回矩阵中非零值的列表。

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> m.values()
        [1, 2, 3]

        See Also
        ========

        iter_values
        tolist
        flat
        """
        return self._eval_values()

    def iter_values(self):
        """迭代返回矩阵中非零值。

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> list(m.iter_values())
        [1, 2, 3]

        See Also
        ========

        values
        """
        return self._eval_iter_values()

    def iter_items(self):
        """迭代返回矩阵中非零项的索引和值。

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> list(m.iter_items())
        [((0, 1), 1), ((1, 0), 2), ((1, 1), 3)]

        See Also
        ========

        iter_values
        todok
        """
        return self._eval_iter_items()

    def _eval_adjoint(self):
        """返回矩阵的共轭转置矩阵。"""
        return self.transpose().conjugate()
    def _eval_applyfunc(self, f):
        # 获取矩阵的列数
        cols = self.cols
        # 计算矩阵的总元素个数
        size = self.rows*self.cols

        # 将稀疏矩阵转换为字典形式
        dok = self.todok()
        # 构建数值映射字典，对每个值应用函数 f
        valmap = {v: f(v) for v in dok.values()}

        # 如果非零元素个数少于矩阵总元素个数，并且 f(0) 不是零，则使用扁平化输出
        if len(dok) < size and ((fzero := f(S.Zero)) is not S.Zero):
            out_flat = [fzero]*size
            # 将字典中的值映射到扁平化输出中的位置
            for (i, j), v in dok.items():
                out_flat[i*cols + j] = valmap[v]
            out = self._new(self.rows, self.cols, out_flat)
        else:
            # 否则，根据映射后的字典 fdok 构建新的稀疏矩阵
            fdok = {ij: valmap[v] for ij, v in dok.items()}
            out = self.from_dok(self.rows, self.cols, fdok)

        # 返回应用函数后的结果矩阵
        return out

    def _eval_as_real_imag(self):  # type: ignore
        # 分别对矩阵的实部和虚部应用函数 re 和 im
        return (self.applyfunc(re), self.applyfunc(im))

    def _eval_conjugate(self):
        # 对矩阵的每个元素应用共轭函数
        return self.applyfunc(lambda x: x.conjugate())

    def _eval_permute_cols(self, perm):
        # 对列进行排列，使用给定的排列 perm
        # 创建一个函数 entry，用于获取排列后矩阵的元素
        mapping = list(perm)

        def entry(i, j):
            return self[i, mapping[j]]

        # 返回排列后的新矩阵
        return self._new(self.rows, self.cols, entry)

    def _eval_permute_rows(self, perm):
        # 对行进行排列，使用给定的排列 perm
        # 创建一个函数 entry，用于获取排列后矩阵的元素
        mapping = list(perm)

        def entry(i, j):
            return self[mapping[i], j]

        # 返回排列后的新矩阵
        return self._new(self.rows, self.cols, entry)

    def _eval_trace(self):
        # 计算矩阵的迹，即主对角线上元素之和
        return sum(self[i, i] for i in range(self.rows))

    def _eval_transpose(self):
        # 返回矩阵的转置矩阵
        return self._new(self.cols, self.rows, lambda i, j: self[j, i])

    def adjoint(self):
        """共轭转置或厄米共轭."""
        # 返回矩阵的共轭转置
        return self._eval_adjoint()

    def applyfunc(self, f):
        """对矩阵的每个元素应用给定的函数 f."""
        # 检查函数 f 是否可调用，若不可调用则引发 TypeError
        if not callable(f):
            raise TypeError("`f` must be callable.")

        # 返回应用函数后的结果矩阵
        return self._eval_applyfunc(f)

    def as_real_imag(self, deep=True, **hints):
        """返回矩阵的实部和虚部构成的元组."""
        # 忽略参数 deep 和 hints，直接返回实部和虚部的计算结果
        return self._eval_as_real_imag()

    def conjugate(self):
        """返回矩阵的逐元素共轭."""
        # 返回矩阵的逐元素共轭结果
        return self._eval_conjugate()

    def doit(self, **hints):
        # 对矩阵的每个元素应用 doit 方法
        return self.applyfunc(lambda x: x.doit(**hints))
    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """Apply evalf() to each element of self."""
        # 定义选项字典，用于传递给 evalf 方法的参数
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        # 对 self 中的每个元素应用 evalf 方法，并返回结果
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
        # 对矩阵中的每个元素应用 expand 方法，返回扩展后的矩阵
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
        # 返回矩阵的 Hermite 共轭
        return self.T.C

    def permute_cols(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='cols', direction=direction)``

        See Also
        ========

        permute
        """
        # 对列进行置换，是 self.permute 的简写形式，返回置换后的矩阵
        return self.permute(swaps, orientation='cols', direction=direction)

    def permute_rows(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='rows', direction=direction)``

        See Also
        ========

        permute
        """
        # 对行进行置换，是 self.permute 的简写形式，返回置换后的矩阵
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
        # 对矩阵中的每个元素应用 refine 方法，返回精化后的矩阵
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
        kwargs = {'map': map, 'simultaneous': simultaneous, 'exact': exact}

        # 如果 map 参数为真，执行替换操作
        if map:
            # 创建空字典 d 用于收集替换的结果
            d = {}
            
            # 定义一个函数 func 用于替换 Matrix 中每个元素的 F 函数为 G 函数
            def func(eij):
                # 对每个元素执行 F 到 G 的替换，并将详细的替换结果更新到 d 中
                eij, dij = eij.replace(F, G, **kwargs)
                d.update(dij)
                return eij

            # 对当前的 Matrix 对象应用 func 函数，得到新的 Matrix 对象 M
            M = self.applyfunc(func)
            # 返回替换后的 Matrix 对象 M 和收集的替换结果字典 d
            return M, d

        else:
            # 如果 map 参数为假，直接对每个元素执行 F 到 G 的替换操作，并返回结果
            return self.applyfunc(lambda i: i.replace(F, G, **kwargs))


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

        # 计算 k 对 4 取模，得到实际的旋转次数
        mod = k % 4
        # 根据 mod 的不同取值执行不同的旋转操作并返回结果
        if mod == 0:
            return self
        if mod == 1:
            return self[::-1, ::].T
        if mod == 2:
            return self[::-1, ::-1]
        if mod == 3:
            return self[::, ::-1].T


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
        # 对 Matrix 中的每个元素应用 simplify 方法，并返回结果
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

        # 如果只有一个参数且不是字典或集合，而是迭代器且不是序列，则将其转换为列表
        if len(args) == 1 and not isinstance(args[0], (dict, set)) and iter(args[0]) and not is_sequence(args[0]):
            args = (list(args[0]),)

        # 对 Matrix 中的每个元素应用 subs 方法，并返回结果
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
        # 检查矩阵是否为方阵，如果不是则抛出异常
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
        # 使用 transpose 方法作为属性 T 的别名
        return self.transpose()

    @property
    def C(self):
        '''By-element conjugation'''
        # 返回按元素共轭的矩阵，调用 conjugate 方法
        return self.conjugate()

    def n(self, *args, **kwargs):
        """Apply evalf() to each element of self."""
        # 对矩阵中的每个元素应用 evalf() 方法，并返回结果
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
        # 对矩阵中的每个元素应用 xreplace 规则，并返回新的矩阵
        return self.applyfunc(lambda x: x.xreplace(rule))

    def _eval_simplify(self, **kwargs):
        # XXX: We can't use self.simplify here as mutable subclasses will
        # override simplify and have it return None
        # 对矩阵中的每个元素应用 simplify 方法，并返回简化后的矩阵
        return self.applyfunc(lambda x: x.simplify(**kwargs))

    def _eval_trigsimp(self, **opts):
        from sympy.simplify.trigsimp import trigsimp
        # 对矩阵中的每个元素应用 trigsimp 方法进行三角函数简化，并返回结果
        return self.applyfunc(lambda x: trigsimp(x, **opts))
    def upper_triangular(self, k=0):
        """Return the elements on and above the kth diagonal of a matrix.
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

        # 定义一个内部函数，用于生成上三角矩阵的每个元素
        def entry(i, j):
            return self[i, j] if i + k <= j else self.zero

        # 调用内部函数生成新的矩阵对象并返回
        return self._new(self.rows, self.cols, entry)

    def lower_triangular(self, k=0):
        """Return the elements on and below the kth diagonal of a matrix.
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

        # 定义一个内部函数，用于生成下三角矩阵的每个元素
        def entry(i, j):
            return self[i, j] if i + k >= j else self.zero

        # 调用内部函数生成新的矩阵对象并返回
        return self._new(self.rows, self.cols, entry)

    def _eval_Abs(self):
        # 返回一个新的矩阵对象，其每个元素为原矩阵元素的绝对值
        return self._new(self.rows, self.cols, lambda i, j: Abs(self[i, j]))

    def _eval_add(self, other):
        # 返回一个新的矩阵对象，其每个元素为原矩阵和另一个矩阵对应元素的和
        return self._new(self.rows, self.cols,
                         lambda i, j: self[i, j] + other[i, j])

    def _eval_matrix_mul(self, other):
        # 定义一个内部函数，用于计算两个矩阵的乘积
        def entry(i, j):
            # 使用列表推导式生成第 i 行第 j 列元素的计算表达式
            vec = [self[i,k]*other[k,j] for k in range(self.cols)]
            try:
                # 尝试将 vec 列表中的所有元素相加，返回相加后的结果
                return Add(*vec)
            except (TypeError, SympifyError):
                # 如果出现错误，说明某些矩阵无法使用 `sum` 或 `Add` 相加
                # 回退到一种安全的乘法计算方式
                return reduce(lambda a, b: a + b, vec)

        # 返回一个新的矩阵对象，其每个元素为两个矩阵乘积的结果
        return self._new(self.rows, other.cols, entry)

    def _eval_matrix_mul_elementwise(self, other):
        # 返回一个新的矩阵对象，其每个元素为原矩阵和另一个矩阵对应元素的乘积
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other[i,j])

    def _eval_matrix_rmul(self, other):
        # 定义一个内部函数，用于计算另一个矩阵与自身的乘积
        def entry(i, j):
            # 使用列表推导式生成第 i 行第 j 列元素的计算表达式
            return sum(other[i,k]*self[k,j] for k in range(other.cols))

        # 返回一个新的矩阵对象，其每个元素为另一个矩阵与自身乘积的结果
        return self._new(other.rows, self.cols, entry)
    # 使用递归方式计算幂运算
    def _eval_pow_by_recursion(self, num):
        # 如果幂为1，直接返回自身
        if num == 1:
            return self

        # 如果幂为奇数，分解为 self 和 self^(num-1)
        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion(num - 1)
        else:
            # 如果幂为偶数，self^(num/2)
            a = b = self._eval_pow_by_recursion(num // 2)

        # 返回 self 的乘积
        return a.multiply(b)

    # 使用 Cayley-Hamilton 定理进行幂运算
    def _eval_pow_by_cayley(self, exp):
        from sympy.discrete.recurrences import linrec_coeffs
        row = self.shape[0]
        p = self.charpoly()

        # 获取特征多项式的系数
        coeffs = (-p).all_coeffs()[1:]
        # 计算线性递推系数
        coeffs = linrec_coeffs(coeffs, exp)
        new_mat = self.eye(row)
        ans = self.zeros(row)

        # 使用递推关系求解幂
        for i in range(row):
            ans += coeffs[i]*new_mat
            new_mat *= self

        return ans

    # 使用递归方式计算幂运算，并进行优化 dotprodsimp
    def _eval_pow_by_recursion_dotprodsimp(self, num, prevsimp=None):
        # 初始化 prevsimp
        if prevsimp is None:
            prevsimp = [True]*len(self)

        # 如果幂为1，直接返回自身
        if num == 1:
            return self

        # 如果幂为奇数，分解为 self 和 self^(num-1)
        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion_dotprodsimp(num - 1,
                    prevsimp=prevsimp)
        else:
            # 如果幂为偶数，self^(num/2)
            a = b = self._eval_pow_by_recursion_dotprodsimp(num // 2,
                    prevsimp=prevsimp)

        # 计算矩阵乘积
        m = a.multiply(b, dotprodsimp=False)
        lenm = len(m)
        elems = [None]*lenm

        # 对每一行进行 dotprodsimp 优化
        for i in range(lenm):
            if prevsimp[i]:
                elems[i], prevsimp[i] = _dotprodsimp(m[i], withsimp=True)
            else:
                elems[i] = m[i]

        # 返回优化后的结果矩阵
        return m._new(m.rows, m.cols, elems)

    # 矩阵与标量的乘法
    def _eval_scalar_mul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other)

    # 标量与矩阵的乘法
    def _eval_scalar_rmul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: other*self[i,j])

    # 矩阵元素对另一个数取模
    def _eval_Mod(self, other):
        return self._new(self.rows, self.cols, lambda i, j: Mod(self[i, j], other))

    # Python 算术函数
    def __abs__(self):
        """返回一个新矩阵，其中每个元素为绝对值。"""
        return self._eval_Abs()

    @call_highest_priority('__radd__')
    def __add__(self, other):
        """返回 self + other，如果形状不匹配则引发 ShapeError 异常。"""

        # 强制转换操作数
        other, T = _coerce_operand(self, other)

        # 如果转换后的类型不是矩阵，返回 NotImplemented
        if T != "is_matrix":
            return NotImplemented

        # 如果形状不匹配，引发异常
        if self.shape != other.shape:
            raise ShapeError(f"矩阵大小不匹配: {self.shape} + {other.shape}.")

        # 统一矩阵类型
        a, b = self, other
        if a.__class__ != classof(a, b):
            b, a = a, b

        # 返回相加后的结果
        return a._eval_add(b)

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        # 实现矩阵除法运算
        return self * (self.one / other)

    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        # 实现矩阵乘法运算
        self, other, T = _unify_with_other(self, other)

        # 如果不是矩阵类型，返回 NotImplemented
        if T != "is_matrix":
            return NotImplemented

        # 返回乘法运算结果
        return self.__mul__(other)

    def __mod__(self, other):
        # 对矩阵中每个元素取模
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

        # 调用 multiply 方法执行矩阵乘法
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

        # 获取中间简化的布尔值
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)

        # 统一 self 和 other 为相同类型，并返回转换后的对象
        self, other, T = _unify_with_other(self, other)

        # 如果 T 表示可能是标量
        if T == "possible_scalar":
            try:
                # 尝试执行标量乘法
                return self._eval_scalar_mul(other)
            except TypeError:
                return NotImplemented

        # 如果 T 表示是矩阵
        elif T == "is_matrix":

            # 检查矩阵的形状是否兼容
            if self.shape[1] != other.shape[0]:
                raise ShapeError(f"Matrix size mismatch: {self.shape} * {other.shape}.")

            # 执行矩阵乘法
            m = self._eval_matrix_mul(other)

            # 如果需要中间项简化，则应用
            if isimpbool:
                m = m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])

            return m

        else:
            return NotImplemented

    def multiply_elementwise(self, other):
        """Return the Hadamard product (elementwise product) of A and B

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
        >>> A.multiply_elementwise(B)
        Matrix([
        [  0, 10, 200],
        [300, 40,   5]])

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.cross
        sympy.matrices.matrixbase.MatrixBase.dot
        multiply
        """

        # 检查矩阵形状是否匹配，执行元素级乘法
        if self.shape != other.shape:
            raise ShapeError("Matrix shapes must agree {} != {}".format(self.shape, other.shape))

        return self._eval_matrix_mul_elementwise(other)

    def __neg__(self):
        # 返回自身乘以 -1 的结果
        return self._eval_scalar_mul(-1)

    @call_highest_priority('__rpow__')
    def __pow__(self, exp):
        """Return self**exp a scalar or symbol."""

        # 调用 pow 方法执行幂运算
        return self.pow(exp)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        # 返回 self + other 的结果
        return self + other

    @call_highest_priority('__matmul__')
    # 定义特殊的矩阵乘法反向运算符，用于支持与其他对象的乘法操作
    def __rmatmul__(self, other):
        # 统一自身与其他对象的类型和属性
        self, other, T = _unify_with_other(self, other)

        # 若统一结果不是矩阵，则返回 Not Implemented
        if T != "is_matrix":
            return NotImplemented

        # 调用自身的右乘方法来执行矩阵乘法的反向运算
        return self.__rmul__(other)

    # 使用装饰器调用具有最高优先级的 '__mul__' 方法
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        # 调用 rmultiply 方法执行矩阵乘法
        return self.rmultiply(other)

    # 执行矩阵乘法，支持可选的代数简化
    def rmultiply(self, other, dotprodsimp=None):
        """Same as __rmul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """
        # 获取是否进行中间结果简化的布尔值
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        # 统一自身与其他对象的类型和属性
        self, other, T = _unify_with_other(self, other)

        # 如果其他对象是可能的标量，尝试执行标量乘法
        if T == "possible_scalar":
            try:
                return self._eval_scalar_rmul(other)
            except TypeError:
                return NotImplemented

        # 如果其他对象是矩阵
        elif T == "is_matrix":
            # 检查矩阵尺寸是否匹配
            if self.shape[0] != other.shape[1]:
                raise ShapeError("Matrix size mismatch.")

            # 执行矩阵乘法
            m = self._eval_matrix_rmul(other)

            # 如果开启了简化，则对结果进行代数简化
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])

            # 返回乘法结果
            return m

        else:
            return NotImplemented

    # 使用装饰器调用具有最高优先级的 '__sub__' 方法
    @call_highest_priority('__sub__')
    def __rsub__(self, a):
        # 执行右减操作，等价于 (-self) + a
        return (-self) + a

    # 使用装饰器调用具有最高优先级的 '__rsub__' 方法
    @call_highest_priority('__rsub__')
    def __sub__(self, a):
        # 执行减法操作，等价于 self + (-a)
        return self + (-a)

    # 计算矩阵的行列式，使用 Bareiss 方法
    def _eval_det_bareiss(self, iszerofunc=_is_zero_after_expand_mul):
        return _det_bareiss(self, iszerofunc=iszerofunc)

    # 计算矩阵的行列式，使用 Berkowitz 方法
    def _eval_det_berkowitz(self):
        return _det_berkowitz(self)

    # 计算矩阵的行列式，使用 LU 分解方法
    def _eval_det_lu(self, iszerofunc=_iszero, simpfunc=None):
        return _det_LU(self, iszerofunc=iszerofunc, simpfunc=simpfunc)

    # 计算矩阵的行列式，使用 Bird 方法
    def _eval_det_bird(self):
        return _det_bird(self)

    # 计算矩阵的行列式，使用 Laplace 方法
    def _eval_det_laplace(self):
        return _det_laplace(self)

    # 计算表达式的行列式，用于 expressions.determinant.Determinant 类
    def _eval_determinant(self):
        return _det(self)

    # 求矩阵的伴随矩阵，支持多种方法，默认使用 Berkowitz 方法
    def adjugate(self, method="berkowitz"):
        return _adjugate(self, method=method)

    # 计算矩阵的特征多项式，默认使用 lambda 作为变量，并可选简化方法
    def charpoly(self, x='lambda', simplify=_utilities_simplify):
        return _charpoly(self, x=x, simplify=simplify)

    # 计算矩阵的余子式，支持多种方法，默认使用 Berkowitz 方法
    def cofactor(self, i, j, method="berkowitz"):
        return _cofactor(self, i, j, method=method)

    # 计算矩阵的余子式矩阵，支持多种方法，默认使用 Berkowitz 方法
    def cofactor_matrix(self, method="berkowitz"):
        return _cofactor_matrix(self, method=method)

    # 计算矩阵的行列式，支持多种方法，默认使用 Bareiss 方法
    def det(self, method="bareiss", iszerofunc=None):
        return _det(self, method=method, iszerofunc=iszerofunc)

    # 计算矩阵的行列式
    def per(self):
        return _per(self)

    # 计算矩阵的指定元素的余子式，支持多种方法，默认使用 Berkowitz 方法
    def minor(self, i, j, method="berkowitz"):
        return _minor(self, i, j, method=method)

    # 计算矩阵的指定元素的余子式子矩阵
    def minor_submatrix(self, i, j):
        return _minor_submatrix(self, i, j)

    # 将 _find_reasonable_pivot 方法的文档字符串设置为与 _find_reasonable_pivot.__doc__ 相同
    _find_reasonable_pivot.__doc__ = _find_reasonable_pivot.__doc__
    # 将函数的文档字符串设置为对应函数的文档字符串，用于文档生成和帮助信息显示
    _find_reasonable_pivot_naive.__doc__ = _find_reasonable_pivot_naive.__doc__
    _eval_det_bareiss.__doc__            = _det_bareiss.__doc__
    _eval_det_berkowitz.__doc__          = _det_berkowitz.__doc__
    _eval_det_bird.__doc__               = _det_bird.__doc__
    _eval_det_laplace.__doc__            = _det_laplace.__doc__
    _eval_det_lu.__doc__                 = _det_LU.__doc__
    _eval_determinant.__doc__            = _det.__doc__
    adjugate.__doc__                     = _adjugate.__doc__
    charpoly.__doc__                     = _charpoly.__doc__
    cofactor.__doc__                     = _cofactor.__doc__
    cofactor_matrix.__doc__              = _cofactor_matrix.__doc__
    det.__doc__                          = _det.__doc__
    per.__doc__                          = _per.__doc__
    minor.__doc__                        = _minor.__doc__
    minor_submatrix.__doc__              = _minor_submatrix.__doc__

    # 定义矩阵对象的方法，返回矩阵的梯形形式（行简化阶梯形式）
    def echelon_form(self, iszerofunc=_iszero, simplify=False, with_pivots=False):
        return _echelon_form(self, iszerofunc=iszerofunc, simplify=simplify,
                with_pivots=with_pivots)

    # 属性装饰器，用于检查矩阵是否处于梯形形式
    @property
    def is_echelon(self):
        return _is_echelon(self)

    # 返回矩阵的秩，即线性无关的行或列的最大数量
    def rank(self, iszerofunc=_iszero, simplify=False):
        return _rank(self, iszerofunc=iszerofunc, simplify=simplify)

    # 返回矩阵与右侧向量或矩阵的增广矩阵的简化行阶梯形式及其变换结果
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
        # 将增广矩阵传入 `_rref` 函数进行处理，并返回结果
        r, _ = _rref(self.hstack(self, self.eye(self.rows), rhs))
        return r[:, :self.cols], r[:, -rhs.cols:]

    # 返回矩阵的简化行阶梯形式
    def rref(self, iszerofunc=_iszero, simplify=False, pivots=True,
            normalize_last=True):
        return _rref(self, iszerofunc=iszerofunc, simplify=simplify,
            pivots=pivots, normalize_last=normalize_last)

    # 将方法的文档字符串设置为对应函数的文档字符串，用于文档生成和帮助信息显示
    echelon_form.__doc__ = _echelon_form.__doc__
    is_echelon.__doc__   = _is_echelon.__doc__
    rank.__doc__         = _rank.__doc__
    rref.__doc__         = _rref.__doc__
    def _normalize_op_args(self, op, col, k, col1, col2, error_str="col"):
        """Validate the arguments for a row/column operation.  ``error_str``
        can be one of "row" or "col" depending on the arguments being parsed."""
        # 检查操作符是否在支持的列表中，如果不在则抛出数值错误异常
        if op not in ["n->kn", "n<->m", "n->n+km"]:
            raise ValueError("Unknown {} operation '{}'. Valid col operations "
                             "are 'n->kn', 'n<->m', 'n->n+km'".format(error_str, op))

        # 根据 error_str 定义 self_cols，可以是 self.cols 或 self.rows
        self_cols = self.cols if error_str == 'col' else self.rows

        # 根据不同的操作符进行参数的规范化和验证
        if op == "n->kn":
            # 如果 col 为 None，则使用 col1
            col = col if col is not None else col1
            # 如果 col 或 k 为 None，则抛出数值错误异常
            if col is None or k is None:
                raise ValueError("For a {0} operation 'n->kn' you must provide the "
                                 "kwargs `{0}` and `k`".format(error_str))
            # 检查 col 是否在合法范围内
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))

        elif op == "n<->m":
            # 需要两个列进行交换。收集指定的列，并移除 None
            cols = {col, k, col1, col2}.difference([None])
            # 如果列数大于 2，则可能用户误留下了 `k`
            if len(cols) > 2:
                cols = {col, col1, col2}.difference([None])
            # 如果列数不等于 2，则抛出数值错误异常
            if len(cols) != 2:
                raise ValueError("For a {0} operation 'n<->m' you must provide the "
                                 "kwargs `{0}1` and `{0}2`".format(error_str))
            col1, col2 = cols
            # 检查 col1 和 col2 是否在合法范围内
            if not 0 <= col1 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col1))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        elif op == "n->n+km":
            # 如果 col 为 None，则使用 col1；如果 col2 为 None，则使用 col1
            col = col1 if col is None else col
            col2 = col1 if col2 is None else col2
            # 如果 col、col2 或 k 为 None，则抛出数值错误异常
            if col is None or col2 is None or k is None:
                raise ValueError("For a {0} operation 'n->n+km' you must provide the "
                                 "kwargs `{0}`, `k`, and `{0}2`".format(error_str))
            # 检查 col 和 col2 是否相同，必须不同
            if col == col2:
                raise ValueError("For a {0} operation 'n->n+km' `{0}` and `{0}2` must "
                                 "be different.".format(error_str))
            # 检查 col 和 col2 是否在合法范围内
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        else:
            # 如果操作符无效，则抛出数值错误异常
            raise ValueError('invalid operation %s' % repr(op))

        # 返回规范化后的操作参数
        return op, col, k, col1, col2
    def _eval_col_op_multiply_col_by_const(self, col, k):
        # 定义一个内部函数 `entry(i, j)`，用于处理矩阵元素操作
        def entry(i, j):
            # 如果当前处理的列 `j` 是目标列 `col`，则返回乘以常数 `k` 的结果
            if j == col:
                return k * self[i, j]
            # 否则返回矩阵中原始的元素值
            return self[i, j]
        # 返回一个新的矩阵对象，应用了列乘以常数的操作
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_swap(self, col1, col2):
        # 定义一个内部函数 `entry(i, j)`，用于处理矩阵元素操作
        def entry(i, j):
            # 如果当前处理的列 `j` 是列 `col1`，则返回与列 `col2` 对应的元素值
            if j == col1:
                return self[i, col2]
            # 如果当前处理的列 `j` 是列 `col2`，则返回与列 `col1` 对应的元素值
            elif j == col2:
                return self[i, col1]
            # 否则返回矩阵中原始的元素值
            return self[i, j]
        # 返回一个新的矩阵对象，应用了列交换的操作
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_add_multiple_to_other_col(self, col, k, col2):
        # 定义一个内部函数 `entry(i, j)`，用于处理矩阵元素操作
        def entry(i, j):
            # 如果当前处理的列 `j` 是目标列 `col`，则返回原始值加上常数 `k` 乘以列 `col2` 对应的值
            if j == col:
                return self[i, j] + k * self[i, col2]
            # 否则返回矩阵中原始的元素值
            return self[i, j]
        # 返回一个新的矩阵对象，应用了列加倍并加到另一列的操作
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_swap(self, row1, row2):
        # 定义一个内部函数 `entry(i, j)`，用于处理矩阵元素操作
        def entry(i, j):
            # 如果当前处理的行 `i` 是行 `row1`，则返回与行 `row2` 对应的列 `j` 的元素值
            if i == row1:
                return self[row2, j]
            # 如果当前处理的行 `i` 是行 `row2`，则返回与行 `row1` 对应的列 `j` 的元素值
            elif i == row2:
                return self[row1, j]
            # 否则返回矩阵中原始的元素值
            return self[i, j]
        # 返回一个新的矩阵对象，应用了行交换的操作
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_multiply_row_by_const(self, row, k):
        # 定义一个内部函数 `entry(i, j)`，用于处理矩阵元素操作
        def entry(i, j):
            # 如果当前处理的行 `i` 是目标行 `row`，则返回乘以常数 `k` 的结果
            if i == row:
                return k * self[i, j]
            # 否则返回矩阵中原始的元素值
            return self[i, j]
        # 返回一个新的矩阵对象，应用了行乘以常数的操作
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_add_multiple_to_other_row(self, row, k, row2):
        # 定义一个内部函数 `entry(i, j)`，用于处理矩阵元素操作
        def entry(i, j):
            # 如果当前处理的行 `i` 是目标行 `row`，则返回原始值加上常数 `k` 乘以行 `row2` 对应的值
            if i == row:
                return self[i, j] + k * self[row2, j]
            # 否则返回矩阵中原始的元素值
            return self[i, j]
        # 返回一个新的矩阵对象，应用了行加倍并加到另一行的操作
        return self._new(self.rows, self.cols, entry)

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

        # 规范化操作参数
        op, col, k, col1, col2 = self._normalize_op_args(op, col, k, col1, col2, "col")

        # 根据规范化后的操作类型分发到相应的列操作函数
        if op == "n->kn":
            return self._eval_col_op_multiply_col_by_const(col, k)
        if op == "n<->m":
            return self._eval_col_op_swap(col1, col2)
        if op == "n->n+km":
            return self._eval_col_op_add_multiple_to_other_col(col, k, col2)
    def elementary_row_op(self, op="n->kn", row=None, k=None, row1=None, row2=None):
        """Performs elementary row operations on a matrix.

        `op` specifies the type of operation:
            * ``"n->kn"``: Multiply row `n` by scalar `k`.
            * ``"n<->m"``: Swap rows `n` and `m`.
            * ``"n->n+km"``: Add `k` times row `m` to row `n`.

        Parameters
        ==========

        op : str
            The type of elementary row operation to perform.
        row : int or None
            The index of the row to which the operation applies.
        k : int or None
            The scalar multiplier for the row operation.
        row1 : int or None
            The index of one row in case of a row swap or the row for operation "n->n+km".
        row2 : int or None
            The index of another row in case of a row swap or the row `m` for operation "n->n+km".
        """

        op, row, k, row1, row2 = self._normalize_op_args(op, row, k, row1, row2, "row")

        # Dispatch to the corresponding method based on the operation type
        if op == "n->kn":
            return self._eval_row_op_multiply_row_by_const(row, k)
        if op == "n<->m":
            return self._eval_row_op_swap(row1, row2)
        if op == "n->n+km":
            return self._eval_row_op_add_multiple_to_other_row(row, k, row2)

    def columnspace(self, simplify=False):
        """Returns the column space (span of columns) of the matrix."""
        return _columnspace(self, simplify=simplify)

    def nullspace(self, simplify=False, iszerofunc=_iszero):
        """Returns the null space (kernel) of the matrix."""
        return _nullspace(self, simplify=simplify, iszerofunc=iszerofunc)

    def rowspace(self, simplify=False):
        """Returns the row space (span of rows) of the matrix."""
        return _rowspace(self, simplify=simplify)

    # This method is initially defined as a normal method and later converted to classmethod
    # to allow assignment of __doc__ due to restrictions in Python 3.6.
    def orthogonalize(cls, *vecs, **kwargs):
        """Orthogonalizes the given vectors."""
        return _orthogonalize(cls, *vecs, **kwargs)

    columnspace.__doc__   = _columnspace.__doc__
    nullspace.__doc__     = _nullspace.__doc__
    rowspace.__doc__      = _rowspace.__doc__
    orthogonalize.__doc__ = _orthogonalize.__doc__

    orthogonalize         = classmethod(orthogonalize)  # type:ignore

    def eigenvals(self, error_when_incomplete=True, **flags):
        """Computes eigenvalues of the matrix."""
        return _eigenvals(self, error_when_incomplete=error_when_incomplete, **flags)

    def eigenvects(self, error_when_incomplete=True, iszerofunc=_iszero, **flags):
        """Computes eigenvalues and eigenvectors of the matrix."""
        return _eigenvects(self, error_when_incomplete=error_when_incomplete,
                iszerofunc=iszerofunc, **flags)

    def is_diagonalizable(self, reals_only=False, **kwargs):
        """Checks if the matrix is diagonalizable."""
        return _is_diagonalizable(self, reals_only=reals_only, **kwargs)

    def diagonalize(self, reals_only=False, sort=False, normalize=False):
        """Diagonalizes the matrix."""
        return _diagonalize(self, reals_only=reals_only, sort=sort,
                normalize=normalize)

    def bidiagonalize(self, upper=True):
        """Bidiagonalizes the matrix."""
        return _bidiagonalize(self, upper=upper)

    def bidiagonal_decomposition(self, upper=True):
        """Computes the bidiagonal decomposition of the matrix."""
        return _bidiagonal_decomposition(self, upper=upper)

    @property
    def is_positive_definite(self):
        """Checks if the matrix is positive definite."""
        return _is_positive_definite(self)
    # 检查矩阵是否半正定，并返回结果
    def is_positive_semidefinite(self):
        return _is_positive_semidefinite(self)

    # 属性装饰器，检查矩阵是否负定，并返回结果
    @property
    def is_negative_definite(self):
        return _is_negative_definite(self)

    # 属性装饰器，检查矩阵是否半负定，并返回结果
    @property
    def is_negative_semidefinite(self):
        return _is_negative_semidefinite(self)

    # 属性装饰器，检查矩阵是否不定，并返回结果
    @property
    def is_indefinite(self):
        return _is_indefinite(self)

    # 计算矩阵的乔丹标准型，可选择计算变换，默认情况下返回结果
    def jordan_form(self, calc_transform=True, **kwargs):
        return _jordan_form(self, calc_transform=calc_transform, **kwargs)

    # 计算矩阵的左特征向量
    def left_eigenvects(self, **flags):
        return _left_eigenvects(self, **flags)

    # 计算矩阵的奇异值
    def singular_values(self):
        return _singular_values(self)

    # 将函数文档设置为相应函数的文档
    eigenvals.__doc__                  = _eigenvals.__doc__
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

    # 计算矩阵每个元素的导数
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
        # 导入数组导数处理模块，并计算导数
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        deriv = ArrayDerivative(self, *args, evaluate=evaluate)
        # 如果矩阵不是基本对象并且要求评估结果，则返回可变矩阵
        if not isinstance(self, Basic) and evaluate:
            return deriv.as_mutable()
        return deriv

    # 计算矩阵对某个参数的导数
    def _eval_derivative(self, arg):
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
        # 对矩阵中的每个元素进行积分，将参数传递给积分函数
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
        from sympy.matrices.matrixbase import MatrixBase
        # 如果X不是MatrixBase类型，则将其转换为Matrix
        if not isinstance(X, MatrixBase):
            X = self._new(X)
        
        # 确保self和X都可以是行或列矩阵，否则抛出TypeError
        if self.shape[0] == 1:
            m = self.shape[1]
        elif self.shape[1] == 1:
            m = self.shape[0]
        else:
            raise TypeError("``self`` must be a row or a column matrix")
        
        if X.shape[0] == 1:
            n = X.shape[1]
        elif X.shape[1] == 1:
            n = X.shape[0]
        else:
            raise TypeError("X must be a row or a column matrix")

        # m是函数数量，n是变量数量
        # 计算Jacobian矩阵
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
        # 使用 applyfunc 方法对矩阵中每个元素应用 limit 函数，并传递 args 参数
        return self.applyfunc(lambda x: x.limit(*args))

    def berkowitz_charpoly(self, x=Dummy('lambda'), simplify=_utilities_simplify):
        # 调用 charpoly 方法，返回特征多项式
        return self.charpoly(x=x)

    def berkowitz_det(self):
        """Computes determinant using Berkowitz method.

        See Also
        ========

        det
        """
        # 调用 Matrix 类的 det 方法，使用 Berkowitz 方法计算行列式
        return self.det(method='berkowitz')

    def berkowitz_eigenvals(self, **flags):
        """Computes eigenvalues of a Matrix using Berkowitz method."""
        # 调用 eigenvals 方法，使用 Berkowitz 方法计算矩阵的特征值
        return self.eigenvals(**flags)

    def berkowitz_minors(self):
        """Computes principal minors using Berkowitz method."""
        # 使用 Berkowitz 方法计算主子式
        sign, minors = self.one, []

        for poly in self.berkowitz():
            minors.append(sign * poly[-1])
            sign = -sign

        return tuple(minors)

    def berkowitz(self):
        from sympy.matrices import zeros
        berk = ((1,),)
        if not self:
            return berk

        if not self.is_square:
            raise NonSquareMatrixError()

        A, N = self, self.rows
        transforms = [0] * (N - 1)

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

        for i, T in enumerate(transforms):
            polys.append(T * polys[i])

        return berk + tuple(map(tuple, polys))

    def cofactorMatrix(self, method="berkowitz"):
        # 调用 cofactor_matrix 方法，计算伴随矩阵，默认使用 Berkowitz 方法
        return self.cofactor_matrix(method=method)

    def det_bareis(self):
        # 调用 _det_bareiss 方法，计算行列式
        return _det_bareiss(self)

    def det_LU_decomposition(self):
        """Compute matrix determinant using LU decomposition.


        Note that this method fails if the LU decomposition itself
        fails. In particular, if the matrix has no inverse this method
        will fail.

        TODO: Implement algorithm for sparse matrices (SFF),
        http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.

        See Also
        ========


        det
        berkowitz_det
        """
        # 调用 Matrix 类的 det 方法，使用 LU 分解计算行列式
        return self.det(method='lu')

    def jordan_cell(self, eigenval, n):
        # 调用 jordan_block 方法，返回特定特征值和大小的 Jordan 块
        return self.jordan_block(size=n, eigenvalue=eigenval)
    # 返回矩阵的乔丹细胞（Jordan cells）形式
    def jordan_cells(self, calc_transformation=True):
        # 获得矩阵的乔丹形式 P 和 J
        P, J = self.jordan_form()
        # 返回乔丹变换矩阵 P 和对角块 J
        return P, J.get_diag_blocks()

    # 返回矩阵的（i, j）次子式，可以选择不同的计算方法，默认为 "berkowitz"
    def minorEntry(self, i, j, method="berkowitz"):
        return self.minor(i, j, method=method)

    # 返回矩阵的（i, j）次子矩阵
    def minorMatrix(self, i, j):
        return self.minor_submatrix(i, j)

    # 使用给定的排列反向置换矩阵的行
    def permuteBkwd(self, perm):
        return self.permute_rows(perm, direction='backward')

    # 使用给定的排列正向置换矩阵的行
    def permuteFwd(self, perm):
        return self.permute_rows(perm, direction='forward')

    @property
    # 返回矩阵的类型，基于元素类型
    def kind(self) -> MatrixKind:
        # 获取矩阵中所有元素的类型集合
        elem_kinds = {e.kind for e in self.flat()}
        # 如果只有一个元素类型
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            # 如果存在多种类型，则归类为未定义类型
            elemkind = UndefinedKind
        # 返回矩阵类型对象
        return MatrixKind(elemkind)

    # 返回矩阵的扁平化列表，包含矩阵中所有元素
    def flat(self):
        """
        Returns a flat list of all elements in the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 2], [3, 4]])
        >>> m.flat()
        [0, 2, 3, 4]

        See Also
        ========

        tolist
        values
        """
        return [self[i, j] for i in range(self.rows) for j in range(self.cols)]

    # 将矩阵转换为 ndarray 类型的数组
    def __array__(self, dtype=object, copy=None):
        # 如果指定不复制，则抛出错误，因为无法实现从矩阵到 ndarray 的不复制转换
        if copy is not None and not copy:
            raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
        # 导入 matrix2numpy 函数，将矩阵转换为 ndarray
        from .dense import matrix2numpy
        return matrix2numpy(self, dtype=dtype)

    # 返回矩阵的元素个数，用于实现 bool(Matrix()) == False
    def __len__(self):
        """Return the number of elements of ``self``.

        Implemented mainly so bool(Matrix()) == False.
        """
        return self.rows * self.cols
    def __str__(self):
        # 如果矩阵的形状包含零维度，则返回特定格式的字符串表示
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        # 否则返回普通格式的字符串表示，转换为列表形式
        return "Matrix(%s)" % str(self.tolist())

    def _format_str(self, printer=None):
        # 如果没有指定打印器，默认使用 StrPrinter
        if not printer:
            printer = StrPrinter()
        # 处理零维度的情况
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        # 处理只有一行的情况，返回格式化后的字符串表示
        if self.rows == 1:
            return "Matrix([%s])" % self.table(printer, rowsep=',\n')
        # 对于多行多列的矩阵，返回格式化后的字符串表示
        return "Matrix([\n%s])" % self.table(printer, rowsep=',\n')
    def irregular(cls, ntop, *matrices, **kwargs):
        """Return a matrix filled by the given matrices which
        are listed in order of appearance from left to right, top to
        bottom as they first appear in the matrix. They must fill the
        matrix completely.

        Examples
        ========

        >>> from sympy import ones, Matrix
        >>> Matrix.irregular(3, ones(2,1), ones(3,3)*2, ones(2,2)*3,
        ...   ones(1,1)*4, ones(2,2)*5, ones(1,2)*6, ones(1,2)*7)
        Matrix([
          [1, 2, 2, 2, 3, 3],
          [1, 2, 2, 2, 3, 3],
          [4, 2, 2, 2, 5, 5],
          [6, 6, 7, 7, 5, 5]])
        """
        ntop = as_int(ntop)
        # 将传入的 matrices 转换为明确的矩阵对象
        b = [i.as_explicit() if hasattr(i, 'as_explicit') else i
            for i in matrices]
        # 创建一个队列来追踪矩阵的顺序
        q = list(range(len(b)))
        # 记录每个矩阵的行数
        dat = [i.rows for i in b]
        # 从队列中取出前 ntop 个矩阵作为活跃矩阵
        active = [q.pop(0) for _ in range(ntop)]
        # 计算最终矩阵的列数
        cols = sum(b[i].cols for i in active)
        rows = []
        while any(dat):
            r = []
            # 遍历活跃矩阵
            for a, j in enumerate(active):
                # 将活跃矩阵的行依次加入结果行中
                r.extend(b[j][-dat[j], :])
                dat[j] -= 1
                if dat[j] == 0 and q:
                    active[a] = q.pop(0)
            # 检查结果行的长度是否与预期的列数相符
            if len(r) != cols:
                raise ValueError(filldedent('''
                    Matrices provided do not appear to fill
                    the space completely.'''))
            rows.append(r)
        # 返回新创建的不规则矩阵对象
        return cls._new(rows)

    @classmethod
    def _handle_ndarray(cls, arg):
        # 处理 NumPy 数组或矩阵，或实现了 __array__ 方法的其他对象
        # 先将其转换为 numpy.array()，然后再转换为 Python 列表
        arr = arg.__array__()
        if len(arr.shape) == 2:
            rows, cols = arr.shape[0], arr.shape[1]
            flat_list = [cls._sympify(i) for i in arr.ravel()]
            return rows, cols, flat_list
        elif len(arr.shape) == 1:
            flat_list = [cls._sympify(i) for i in arr]
            return arr.shape[0], 1, flat_list
        else:
            # 抛出未实现的错误，SymPy 仅支持 1D 和 2D 矩阵
            raise NotImplementedError(
                "SymPy supports just 1D and 2D matrices")
    def _setitem(self, key, value):
        """Helper to set value at location given by key.

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        To replace row r you assign to position r*m where m
        is the number of columns:

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        And to replace column c you can assign to position c:

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        from .dense import Matrix  # 导入密集矩阵类 Matrix

        is_slice = isinstance(key, slice)  # 检查 key 是否为切片对象
        i, j = key = self.key2ij(key)  # 解析 key 为行列索引 i, j
        is_mat = isinstance(value, MatrixBase)  # 检查 value 是否为 MatrixBase 类型的对象
        if isinstance(i, slice) or isinstance(j, slice):  # 如果 i 或 j 是切片对象
            if is_mat:  # 如果 value 是 MatrixBase 类型的对象
                self.copyin_matrix(key, value)  # 将矩阵 value 复制到指定位置
                return
            if not isinstance(value, Expr) and is_sequence(value):  # 如果 value 不是表达式且是可迭代序列
                self.copyin_list(key, value)  # 将列表 value 复制到指定位置
                return
            raise ValueError('unexpected value: %s' % value)  # 抛出值错误异常，显示不期望的值
        else:
            if (not is_mat and
                    not isinstance(value, Basic) and is_sequence(value)):  # 如果 value 不是 MatrixBase、Basic 类型且是可迭代序列
                value = Matrix(value)  # 将 value 转换为 Matrix 对象
                is_mat = True
            if is_mat:  # 如果 value 是 MatrixBase 类型的对象
                if is_slice:  # 如果 key 是切片对象
                    key = (slice(*divmod(i, self.cols)),  # 计算行索引切片
                           slice(*divmod(j, self.cols)))  # 计算列索引切片
                else:
                    key = (slice(i, i + value.rows),  # 计算行索引切片
                           slice(j, j + value.cols))  # 计算列索引切片
                self.copyin_matrix(key, value)  # 将矩阵 value 复制到指定位置
            else:
                return i, j, self._sympify(value)  # 返回行索引 i、列索引 j、值的符号化表示
            return

    def add(self, b):
        """Return self + b."""
        return self + b  # 返回 self 与 b 相加的结果

    def condition_number(self):
        """Returns the condition number of a matrix.

        This is the maximum singular value divided by the minimum singular value

        Examples
        ========

        >>> from sympy import Matrix, S
        >>> A = Matrix([[1, 0, 0], [0, 10, 0], [0, 0, S.One/10]])
        >>> A.condition_number()
        100

        See Also
        ========

        singular_values
        """

        if not self:  # 如果矩阵 self 是空的
            return self.zero  # 返回矩阵 self 的零元素
        singularvalues = self.singular_values()  # 计算矩阵 self 的奇异值
        return Max(*singularvalues) / Min(*singularvalues)  # 返回奇异值的最大值除以最小值作为条件数
    def copy(self):
        """
        返回矩阵的副本。

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.copy()
        Matrix([
        [1, 2],
        [3, 4]])

        """
        # 调用 _new 方法创建当前矩阵的副本，使用 self.flat() 获取扁平化后的元素列表作为参数
        return self._new(self.rows, self.cols, self.flat())

    def cross(self, b):
        r"""
        返回矩阵 self 与 b 的叉乘结果，允许维度不匹配：如果两者都有3个元素，返回与 self 相同类型和形状的矩阵。
        如果 b 的形状与 self 相同，则叉乘的常见性质（如 `a \times b = - b \times a`）成立。

        Parameters
        ==========
            b : 3x1 或 1x3 的矩阵

        See Also
        ========

        dot
        hat
        vee
        multiply
        multiply_elementwise
        """
        # 导入 MatrixExpr 类型，确保 b 是 MatrixBase 或 MatrixExpr 的实例
        from sympy.matrices.expressions.matexpr import MatrixExpr

        if not isinstance(b, (MatrixBase, MatrixExpr)):
            # 如果 b 不是 MatrixBase 或 MatrixExpr 的实例，则引发类型错误
            raise TypeError(
                "{} must be a Matrix, not {}.".format(b, type(b)))

        if not (self.rows * self.cols == b.rows * b.cols == 3):
            # 如果 self 和 b 的行列数乘积不等于3，引发形状错误
            raise ShapeError("Dimensions incorrect for cross product: %s x %s" %
                             ((self.rows, self.cols), (b.rows, b.cols)))
        else:
            # 计算并返回叉乘结果的新矩阵
            return self._new(self.rows, self.cols, (
                (self[1] * b[2] - self[2] * b[1]),
                (self[2] * b[0] - self[0] * b[2]),
                (self[0] * b[1] - self[1] * b[0])))

    def hat(self):
        r"""
        返回表示叉乘的反对称矩阵，使得 ``self.hat() * b`` 等同于 ``self.cross(b)``。

        Examples
        ========

        调用 ``hat`` 方法从一个 3x1 的矩阵创建一个反对称的 3x3 矩阵：

        >>> from sympy import Matrix
        >>> a = Matrix([1, 2, 3])
        >>> a.hat()
        Matrix([
        [ 0, -3,  2],
        [ 3,  0, -1],
        [-2,  1,  0]])

        将其与另一个 3x1 矩阵相乘计算叉乘结果：

        >>> b = Matrix([3, 2, 1])
        >>> a.hat() * b
        Matrix([
        [-4],
        [ 8],
        [-4]])

        这等同于调用 ``cross`` 方法得到的结果：

        >>> a.cross(b)
        Matrix([
        [-4],
        [ 8],
        [-4]])

        See Also
        ========

        dot
        cross
        vee
        multiply
        multiply_elementwise
        """

        if self.shape != (3, 1):
            # 如果 self 的形状不是 (3, 1)，引发形状错误
            raise ShapeError("Dimensions incorrect, expected (3, 1), got " +
                             str(self.shape))
        else:
            # 获取 self 的元素 x, y, z，并返回叉乘的反对称矩阵
            x, y, z = self
            return self._new(3, 3, (
                 0, -z,  y,
                 z,  0, -x,
                -y,  x,  0))
    def vee(self):
        r"""
        Return a 3x1 vector from a skew-symmetric matrix representing the cross product,
        so that ``self * b`` is equivalent to  ``self.vee().cross(b)``.

        Examples
        ========

        Calling ``vee`` creates a vector from a skew-symmetric Matrix:

        >>> from sympy import Matrix
        >>> A = Matrix([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        >>> a = A.vee()
        >>> a
        Matrix([
        [1],
        [2],
        [3]])

        Calculating the matrix product of the original matrix with a vector
        is equivalent to a cross product:

        >>> b = Matrix([3, 2, 1])
        >>> A * b
        Matrix([
        [-4],
        [ 8],
        [-4]])

        >>> a.cross(b)
        Matrix([
        [-4],
        [ 8],
        [-4]])

        ``vee`` can also be used to retrieve angular velocity expressions.
        Defining a rotation matrix:

        >>> from sympy import rot_ccw_axis3, trigsimp
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> theta = dynamicsymbols('theta')
        >>> R = rot_ccw_axis3(theta)
        >>> R
        Matrix([
        [cos(theta(t)), -sin(theta(t)), 0],
        [sin(theta(t)),  cos(theta(t)), 0],
        [            0,              0, 1]])

        We can retrieve the angular velocity:

        >>> Omega = R.T * R.diff()
        >>> Omega = trigsimp(Omega)
        >>> Omega.vee()
        Matrix([
        [                      0],
        [                      0],
        [Derivative(theta(t), t)]])

        See Also
        ========

        dot
        cross
        hat
        multiply
        multiply_elementwise
        """

        # 检查矩阵形状是否为 (3, 3)
        if self.shape != (3, 3):
            raise ShapeError("Dimensions incorrect, expected (3, 3), got " +
                             str(self.shape))
        # 检查矩阵是否为反对称矩阵（即 skew-symmetric）
        elif not self.is_anti_symmetric():
            raise ValueError("Matrix is not skew-symmetric")
        else:
            # 根据 skew-symmetric 矩阵生成对应的 3x1 向量
            return self._new(3, 1, (
                 self[2, 1],  # 第一个分量
                 self[0, 2],  # 第二个分量
                 self[1, 0]))  # 第三个分量

    @property
    def D(self):
        """Return Dirac conjugate (if ``self.rows == 4``).

        Examples
        ========

        >>> from sympy import Matrix, I, eye
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m.D
        Matrix([[0, 1 - I, -2, -3]])
        >>> m = (eye(4) + I*eye(4))
        >>> m[0, 3] = 2
        >>> m.D
        Matrix([
        [1 - I,     0,      0,      0],
        [    0, 1 - I,      0,      0],
        [    0,     0, -1 + I,      0],
        [    2,     0,      0, -1 + I]])

        If the matrix does not have 4 rows an AttributeError will be raised
        because this property is only defined for matrices with 4 rows.

        >>> Matrix(eye(2)).D
        Traceback (most recent call last):
        ...
        AttributeError: Matrix has no attribute D.

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.conjugate: By-element conjugation
        sympy.matrices.matrixbase.MatrixBase.H: Hermite conjugation
        """
        from sympy.physics.matrices import mgamma
        # 检查矩阵的行数是否为4，如果不是则抛出 AttributeError 异常
        if self.rows != 4:
            raise AttributeError
        # 返回 Hermite 共轭乘以 mgamma(0) 的结果
        return self.H * mgamma(0)

    def dual(self):
        """Returns the dual of a matrix.

        A dual of a matrix is:

        ``(1/2)*levicivita(i, j, k, l)*M(k, l)`` summed over indices `k` and `l`

        Since the levicivita method is anti_symmetric for any pairwise
        exchange of indices, the dual of a symmetric matrix is the zero
        matrix. Strictly speaking the dual defined here assumes that the
        'matrix' `M` is a contravariant anti_symmetric second rank tensor,
        so that the dual is a covariant second rank tensor.

        """
        from sympy.matrices import zeros
        # 复制当前矩阵和行数
        M, n = self[:, :], self.rows
        # 创建一个全零矩阵作为工作空间
        work = zeros(n)
        # 如果矩阵是对称的，直接返回全零矩阵
        if self.is_symmetric():
            return work

        # 计算非对称情况下的对偶矩阵
        for i in range(1, n):
            for j in range(1, n):
                acum = 0
                for k in range(1, n):
                    # 根据 LeviCivita 符号计算累积和
                    acum += LeviCivita(i, j, 0, k) * M[0, k]
                work[i, j] = acum
                work[j, i] = -acum

        for l in range(1, n):
            acum = 0
            for a in range(1, n):
                for b in range(1, n):
                    # 根据 LeviCivita 符号计算累积和
                    acum += LeviCivita(0, l, a, b) * M[a, b]
            acum /= 2
            work[0, l] = -acum
            work[l, 0] = acum

        return work
    def _eval_matrix_exp_jblock(self):
        """A helper function to compute an exponential of a Jordan block
        matrix

        Examples
        ========

        >>> from sympy import Symbol, Matrix
        >>> l = Symbol('lamda')

        A trivial example of 1*1 Jordan block:

        >>> m = Matrix.jordan_block(1, l)
        >>> m._eval_matrix_exp_jblock()
        Matrix([[exp(lamda)]])

        An example of 3*3 Jordan block:

        >>> m = Matrix.jordan_block(3, l)
        >>> m._eval_matrix_exp_jblock()
        Matrix([
        [exp(lamda), exp(lamda), exp(lamda)/2],
        [         0, exp(lamda),   exp(lamda)],
        [         0,          0,   exp(lamda)]])

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Matrix_function#Jordan_decomposition
        """
        # 获取矩阵的行数作为 Jordan 块的大小
        size = self.rows
        # 获取 Jordan 块的左上角元素
        l = self[0, 0]
        # 计算 Jordan 块左上角元素的指数函数值
        exp_l = exp(l)

        # 计算 Jordan 块的指数函数系数
        bands = {i: exp_l / factorial(i) for i in range(size)}

        # 导入 banded 函数用于处理带状矩阵
        from .sparsetools import banded
        # 返回带状矩阵形式的结果
        return self.__class__(banded(size, bands))


    def exp(self):
        """Return the exponential of a square matrix.

        Examples
        ========

        >>> from sympy import Symbol, Matrix

        >>> t = Symbol('t')
        >>> m = Matrix([[0, 1], [-1, 0]]) * t
        >>> m.exp()
        Matrix([
        [    exp(I*t)/2 + exp(-I*t)/2, -I*exp(I*t)/2 + I*exp(-I*t)/2],
        [I*exp(I*t)/2 - I*exp(-I*t)/2,      exp(I*t)/2 + exp(-I*t)/2]])

        """
        # 检查矩阵是否为方阵，如果不是则引发异常
        if not self.is_square:
            raise NonSquareMatrixError(
                "Exponentiation is valid only for square matrices")
        
        try:
            # 获取矩阵的 Jordan 形式
            P, J = self.jordan_form()
            # 获取对角块列表
            cells = J.get_diag_blocks()
        except MatrixError:
            # 如果无法计算 Jordan 形式，则引发异常
            raise NotImplementedError(
                "Exponentiation is implemented only for matrices for which the Jordan normal form can be computed")

        # 对每个对角块计算其指数函数
        blocks = [cell._eval_matrix_exp_jblock() for cell in cells]
        # 导入 diag 函数用于创建对角矩阵
        from sympy.matrices import diag
        eJ = diag(*blocks)
        
        # 计算指数函数矩阵的乘积
        ret = P.multiply(eJ, dotprodsimp=None).multiply(P.inv(), dotprodsimp=None)
        
        # 如果矩阵所有元素都是实数，则返回实数类型的矩阵
        if all(value.is_real for value in self.values()):
            return type(self)(re(ret))
        else:
            return type(self)(ret)
    def _eval_matrix_log_jblock(self):
        """Helper function to compute logarithm of a jordan block.

        Examples
        ========

        >>> from sympy import Symbol, Matrix
        >>> l = Symbol('lamda')

        A trivial example of 1*1 Jordan block:

        >>> m = Matrix.jordan_block(1, l)
        >>> m._eval_matrix_log_jblock()
        Matrix([[log(lamda)]])

        An example of 3*3 Jordan block:

        >>> m = Matrix.jordan_block(3, l)
        >>> m._eval_matrix_log_jblock()
        Matrix([
        [log(lamda),    1/lamda, -1/(2*lamda**2)],
        [         0, log(lamda),         1/lamda],
        [         0,          0,      log(lamda)]])
        """

        # 获取矩阵行数（或列数，因为这是方阵）
        size = self.rows
        # 获取Jordan块的特征值lamda，通常是矩阵中第一个元素
        l = self[0, 0]

        # 如果特征值为零，无法计算其对数或倒数，抛出异常
        if l.is_zero:
            raise MatrixError(
                'Could not take logarithm or reciprocal for the given '
                'eigenvalue {}'.format(l))

        # 初始化一个字典用于存储带状矩阵的带
        bands = {0: log(l)}
        # 计算带状矩阵的每一带的值
        for i in range(1, size):
            bands[i] = -((-l) ** -i) / i

        # 导入带状矩阵处理工具
        from .sparsetools import banded
        # 返回一个新的矩阵，其中的数据由带状矩阵函数生成
        return self.__class__(banded(size, bands))
    def log(self, simplify=cancel):
        """Return the logarithm of a square matrix.

        Parameters
        ==========

        simplify : function, bool
            The function to simplify the result with.

            Default is ``cancel``, which is effective to reduce the
            expression growing for taking reciprocals and inverses for
            symbolic matrices.

        Examples
        ========

        >>> from sympy import S, Matrix

        Examples for positive-definite matrices:

        >>> m = Matrix([[1, 1], [0, 1]])
        >>> m.log()
        Matrix([
        [0, 1],
        [0, 0]])

        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])
        >>> m.log()
        Matrix([
        [     0, log(2)],
        [log(2),      0]])

        Examples for non positive-definite matrices:

        >>> m = Matrix([[S(3)/4, S(5)/4], [S(5)/4, S(3)/4]])
        >>> m.log()
        Matrix([
        [         I*pi/2, log(2) - I*pi/2],
        [log(2) - I*pi/2,          I*pi/2]])

        >>> m = Matrix(
        ...     [[0, 0, 0, 1],
        ...      [0, 0, 1, 0],
        ...      [0, 1, 0, 0],
        ...      [1, 0, 0, 0]])
        >>> m.log()
        Matrix([
        [ I*pi/2,       0,       0, -I*pi/2],
        [      0,  I*pi/2, -I*pi/2,       0],
        [      0, -I*pi/2,  I*pi/2,       0],
        [-I*pi/2,       0,       0,  I*pi/2]])

        """
        # 检查矩阵是否为方阵，若不是则抛出异常
        if not self.is_square:
            raise NonSquareMatrixError(
                "Logarithm is valid only for square matrices")

        try:
            # 尝试计算简化后的约当形式
            if simplify:
                # 使用给定的简化函数计算约当形式
                P, J = simplify(self).jordan_form()
            else:
                # 计算原始的约当形式
                P, J = self.jordan_form()

            # 获取约当块
            cells = J.get_diag_blocks()
        except MatrixError:
            # 若无法计算约当形式则抛出未实现异常
            raise NotImplementedError(
                "Logarithm is implemented only for matrices for which "
                "the Jordan normal form can be computed")

        # 计算每个约当块的对数
        blocks = [
            cell._eval_matrix_log_jblock()
            for cell in cells]
        from sympy.matrices import diag
        # 构造对角矩阵
        eJ = diag(*blocks)

        if simplify:
            # 简化计算结果
            ret = simplify(P * eJ * simplify(P.inv()))
            # 将结果封装成当前类的实例
            ret = self.__class__(ret)
        else:
            # 直接计算结果
            ret = P * eJ * P.inv()

        return ret
    def is_nilpotent(self):
        """检查矩阵是否幂零。

        如果存在整数 k，使得矩阵 B 的 k 次幂是零矩阵，则矩阵 B 是幂零的。

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        >>> a.is_nilpotent()
        True

        >>> a = Matrix([[1, 0, 1], [1, 0, 0], [1, 1, 0]])
        >>> a.is_nilpotent()
        False
        """
        # 检查矩阵是否为空矩阵
        if not self:
            return True
        # 检查矩阵是否为方阵
        if not self.is_square:
            raise NonSquareMatrixError(
                "Nilpotency is valid only for square matrices")
        # 创建一个唯一命名的符号 x，用于计算特征多项式
        x = uniquely_named_symbol('x', self, modify=lambda s: '_' + s)
        # 计算矩阵的特征多项式
        p = self.charpoly(x)
        # 如果特征多项式的第一个参数是 x 的矩阵行数次幂，则返回 True
        if p.args[0] == x ** self.rows:
            return True
        return False

    def key2bounds(self, keys):
        """将具有混合类型键（整数和切片）的键转换为范围的元组，并在任何索引超出
        ``self`` 范围时引发错误。

        See Also
        ========

        key2ij
        """
        # 判断是否存在切片类型的键
        islice, jslice = [isinstance(k, slice) for k in keys]
        # 处理行索引
        if islice:
            if not self.rows:
                rlo = rhi = 0
            else:
                rlo, rhi = keys[0].indices(self.rows)[:2]
        else:
            rlo = a2idx(keys[0], self.rows)
            rhi = rlo + 1
        # 处理列索引
        if jslice:
            if not self.cols:
                clo = chi = 0
            else:
                clo, chi = keys[1].indices(self.cols)[:2]
        else:
            clo = a2idx(keys[1], self.cols)
            chi = clo + 1
        return rlo, rhi, clo, chi

    def key2ij(self, key):
        """将键转换为规范形式，将整数或可索引项转换为 ``self`` 范围内的有效整数，
        或者返回未更改的切片。

        See Also
        ========

        key2bounds
        """
        # 如果键是序列
        if is_sequence(key):
            # 确保键的长度为 2
            if not len(key) == 2:
                raise TypeError('key must be a sequence of length 2')
            # 将每个键转换为有效整数索引或保持切片不变
            return [a2idx(i, n) if not isinstance(i, slice) else i
                    for i, n in zip(key, self.shape)]
        # 如果键是切片
        elif isinstance(key, slice):
            return key.indices(len(self))[:2]
        # 否则，将单个键转换为有效整数索引
        else:
            return divmod(a2idx(key, len(self)), self.cols)
    def normalized(self, iszerofunc=_iszero):
        """Return the normalized version of ``self``.

        Parameters
        ==========

        iszerofunc : Function, optional
            A function to determine whether ``self`` is a zero vector.
            The default ``_iszero`` tests to see if each element is
            exactly zero.

        Returns
        =======

        Matrix
            Normalized vector form of ``self``.
            It has the same length as a unit vector. However, a zero vector
            will be returned for a vector with norm 0.

        Raises
        ======

        ShapeError
            If the matrix is not in a vector form.

        See Also
        ========

        norm
        """
        # 检查矩阵是否是向量形式，如果不是则引发形状错误异常
        if self.rows != 1 and self.cols != 1:
            raise ShapeError("A Matrix must be a vector to normalize.")
        # 计算矩阵的范数
        norm = self.norm()
        # 根据给定的 iszerofunc 函数判断矩阵是否为零向量
        if iszerofunc(norm):
            # 若为零向量则返回一个相同形状的零矩阵
            out = self.zeros(self.rows, self.cols)
        else:
            # 否则将矩阵中每个元素除以其范数，得到归一化后的矩阵
            out = self.applyfunc(lambda i: i / norm)
        return out

    def print_nonzero(self, symb="X"):
        """Shows location of non-zero entries for fast shape lookup.

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> m = Matrix(2, 3, lambda i, j: i*3+j)
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5]])
        >>> m.print_nonzero()
        [ XX]
        [XXX]
        >>> m = eye(4)
        >>> m.print_nonzero("x")
        [x   ]
        [ x  ]
        [  x ]
        [   x]

        """
        # 初始化一个空列表 s，用于存储每行的非零元素位置信息
        s = []
        # 遍历矩阵的每一行
        for i in range(self.rows):
            line = []
            # 遍历矩阵的每一列
            for j in range(self.cols):
                # 如果当前元素为零，则添加空格字符到 line 列表中
                if self[i, j] == 0:
                    line.append(" ")
                else:
                    # 否则添加指定符号（默认为 "X"）到 line 列表中
                    line.append(str(symb))
            # 将 line 列表转换为字符串，添加到 s 列表中
            s.append("[%s]" % ''.join(line))
        # 将 s 列表中的所有字符串以换行符连接并打印输出
        print('\n'.join(s))

    def project(self, v):
        """Return the projection of ``self`` onto the line containing ``v``.

        Examples
        ========

        >>> from sympy import Matrix, S, sqrt
        >>> V = Matrix([sqrt(3)/2, S.Half])
        >>> x = Matrix([[1, 0]])
        >>> V.project(x)
        Matrix([[sqrt(3)/2, 0]])
        >>> V.project(-x)
        Matrix([[sqrt(3)/2, 0]])
        """
        # 计算向量 self 在向量 v 所在直线上的投影
        return v * (self.dot(v) / v.dot(v))
    def rank_decomposition(self, iszerofunc=_iszero, simplify=False):
        # 使用 _rank_decomposition 函数对当前矩阵进行秩分解
        return _rank_decomposition(self, iszerofunc=iszerofunc,
                simplify=simplify)

    def cholesky(self, hermitian=True):
        # 引发 NotImplementedError 异常，提示该函数在 DenseMatrix 或 SparseMatrix 中实现
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LDLdecomposition(self, hermitian=True):
        # 引发 NotImplementedError 异常，提示该函数在 DenseMatrix 或 SparseMatrix 中实现
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')
    # 对象方法：LU 分解，返回调用 _LUdecomposition 函数的结果
    def LUdecomposition(self, iszerofunc=_iszero, simpfunc=None,
            rankcheck=False):
        return _LUdecomposition(self, iszerofunc=iszerofunc, simpfunc=simpfunc,
                rankcheck=rankcheck)

    # 对象方法：简化的 LU 分解，返回调用 _LUdecomposition_Simple 函数的结果
    def LUdecomposition_Simple(self, iszerofunc=_iszero, simpfunc=None,
            rankcheck=False):
        return _LUdecomposition_Simple(self, iszerofunc=iszerofunc,
                simpfunc=simpfunc, rankcheck=rankcheck)

    # 对象方法：完全 LU 分解，返回调用 _LUdecompositionFF 函数的结果
    def LUdecompositionFF(self):
        return _LUdecompositionFF(self)

    # 对象方法：奇异值分解，返回调用 _singular_value_decomposition 函数的结果
    def singular_value_decomposition(self):
        return _singular_value_decomposition(self)

    # 对象方法：QR 分解，返回调用 _QRdecomposition 函数的结果
    def QRdecomposition(self):
        return _QRdecomposition(self)

    # 对象方法：上 Hessenberg 分解，返回调用 _upper_hessenberg_decomposition 函数的结果
    def upper_hessenberg_decomposition(self):
        return _upper_hessenberg_decomposition(self)

    # 对象方法：对角线求解，返回调用 _diagonal_solve 函数的结果
    def diagonal_solve(self, rhs):
        return _diagonal_solve(self, rhs)

    # 对象方法：下三角求解，抛出未实现错误信息
    def lower_triangular_solve(self, rhs):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    # 对象方法：上三角求解，抛出未实现错误信息
    def upper_triangular_solve(self, rhs):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    # 对象方法：Cholesky 分解求解，返回调用 _cholesky_solve 函数的结果
    def cholesky_solve(self, rhs):
        return _cholesky_solve(self, rhs)

    # 对象方法：LDL 分解求解，返回调用 _LDLsolve 函数的结果
    def LDLsolve(self, rhs):
        return _LDLsolve(self, rhs)

    # 对象方法：LU 分解求解，返回调用 _LUsolve 函数的结果
    def LUsolve(self, rhs, iszerofunc=_iszero):
        return _LUsolve(self, rhs, iszerofunc=iszerofunc)

    # 对象方法：QR 分解求解，返回调用 _QRsolve 函数的结果
    def QRsolve(self, b):
        return _QRsolve(self, b)

    # 对象方法：高斯-约旦消元法求解，返回调用 _gauss_jordan_solve 函数的结果
    def gauss_jordan_solve(self, B, freevar=False):
        return _gauss_jordan_solve(self, B, freevar=freevar)

    # 对象方法：伪逆求解，返回调用 _pinv_solve 函数的结果
    def pinv_solve(self, B, arbitrary_matrix=None):
        return _pinv_solve(self, B, arbitrary_matrix=arbitrary_matrix)

    # 对象方法：克拉默法则求解，返回调用 _cramer_solve 函数的结果
    def cramer_solve(self, rhs, det_method="laplace"):
        return _cramer_solve(self, rhs, det_method=det_method)

    # 对象方法：一般求解方法，返回调用 _solve 函数的结果
    def solve(self, rhs, method='GJ'):
        return _solve(self, rhs, method=method)

    # 对象方法：最小二乘法求解，返回调用 _solve_least_squares 函数的结果
    def solve_least_squares(self, rhs, method='CH'):
        return _solve_least_squares(self, rhs, method=method)

    # 对象方法：伪逆矩阵求解，返回调用 _pinv 函数的结果
    def pinv(self, method='RD'):
        return _pinv(self, method=method)

    # 对象方法：求逆矩阵 ADJ 方法，返回调用 _inv_ADJ 函数的结果
    def inverse_ADJ(self, iszerofunc=_iszero):
        return _inv_ADJ(self, iszerofunc=iszerofunc)

    # 对象方法：求逆矩阵块方法，返回调用 _inv_block 函数的结果
    def inverse_BLOCK(self, iszerofunc=_iszero):
        return _inv_block(self, iszerofunc=iszerofunc)

    # 对象方法：求逆矩阵 GE 方法，返回调用 _inv_GE 函数的结果
    def inverse_GE(self, iszerofunc=_iszero):
        return _inv_GE(self, iszerofunc=iszerofunc)

    # 对象方法：求逆矩阵 LU 方法，返回调用 _inv_LU 函数的结果
    def inverse_LU(self, iszerofunc=_iszero):
        return _inv_LU(self, iszerofunc=iszerofunc)

    # 对象方法：求逆矩阵 Cholesky 方法，返回调用 _inv_CH 函数的结果
    def inverse_CH(self, iszerofunc=_iszero):
        return _inv_CH(self, iszerofunc=iszerofunc)

    # 对象方法：求逆矩阵 LDL 方法，返回调用 _inv_LDL 函数的结果
    def inverse_LDL(self, iszerofunc=_iszero):
        return _inv_LDL(self, iszerofunc=iszerofunc)

    # 对象方法：求逆矩阵 QR 方法，返回调用 _inv_QR 函数的结果
    def inverse_QR(self, iszerofunc=_iszero):
        return _inv_QR(self, iszerofunc=iszerofunc)

    # 对象方法：求逆矩阵通用方法，返回调用 _inv 函数的结果
    def inv(self, method=None, iszerofunc=_iszero, try_block_diag=False):
        return _inv(self, method=method, iszerofunc=iszerofunc,
                try_block_diag=try_block_diag)
    # 返回基于当前对象的连通分量
    def connected_components(self):
        return _connected_components(self)

    # 返回基于当前对象的连通分量分解
    def connected_components_decomposition(self):
        return _connected_components_decomposition(self)

    # 返回基于当前对象的强连通分量
    def strongly_connected_components(self):
        return _strongly_connected_components(self)

    # 返回基于当前对象的强连通分量分解，可以选择是否使用降阶方法
    def strongly_connected_components_decomposition(self, lower=True):
        return _strongly_connected_components_decomposition(self, lower=lower)

    # 将 Basic 类的 _sage_ 属性赋值给当前对象的 _sage_ 属性
    _sage_ = Basic._sage_

    # 将 _rank_decomposition 的文档字符串赋值给 rank_decomposition 方法的文档字符串
    rank_decomposition.__doc__     = _rank_decomposition.__doc__
    # 将 _cholesky 的文档字符串赋值给 cholesky 方法的文档字符串
    cholesky.__doc__               = _cholesky.__doc__
    # 将 _LDLdecomposition 的文档字符串赋值给 LDLdecomposition 方法的文档字符串
    LDLdecomposition.__doc__       = _LDLdecomposition.__doc__
    # 将 _LUdecomposition 的文档字符串赋值给 LUdecomposition 方法的文档字符串
    LUdecomposition.__doc__        = _LUdecomposition.__doc__
    # 将 _LUdecomposition_Simple 的文档字符串赋值给 LUdecomposition_Simple 方法的文档字符串
    LUdecomposition_Simple.__doc__ = _LUdecomposition_Simple.__doc__
    # 将 _LUdecompositionFF 的文档字符串赋值给 LUdecompositionFF 方法的文档字符串
    LUdecompositionFF.__doc__      = _LUdecompositionFF.__doc__
    # 将 _singular_value_decomposition 的文档字符串赋值给 singular_value_decomposition 方法的文档字符串
    singular_value_decomposition.__doc__ = _singular_value_decomposition.__doc__
    # 将 _QRdecomposition 的文档字符串赋值给 QRdecomposition 方法的文档字符串
    QRdecomposition.__doc__        = _QRdecomposition.__doc__
    # 将 _upper_hessenberg_decomposition 的文档字符串赋值给 upper_hessenberg_decomposition 方法的文档字符串
    upper_hessenberg_decomposition.__doc__ = _upper_hessenberg_decomposition.__doc__

    # 将 _diagonal_solve 的文档字符串赋值给 diagonal_solve 方法的文档字符串
    diagonal_solve.__doc__         = _diagonal_solve.__doc__
    # 将 _lower_triangular_solve 的文档字符串赋值给 lower_triangular_solve 方法的文档字符串
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    # 将 _upper_triangular_solve 的文档字符串赋值给 upper_triangular_solve 方法的文档字符串
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__
    # 将 _cholesky_solve 的文档字符串赋值给 cholesky_solve 方法的文档字符串
    cholesky_solve.__doc__         = _cholesky_solve.__doc__
    # 将 _LDLsolve 的文档字符串赋值给 LDLsolve 方法的文档字符串
    LDLsolve.__doc__               = _LDLsolve.__doc__
    # 将 _LUsolve 的文档字符串赋值给 LUsolve 方法的文档字符串
    LUsolve.__doc__                = _LUsolve.__doc__
    # 将 _QRsolve 的文档字符串赋值给 QRsolve 方法的文档字符串
    QRsolve.__doc__                = _QRsolve.__doc__
    # 将 _gauss_jordan_solve 的文档字符串赋值给 gauss_jordan_solve 方法的文档字符串
    gauss_jordan_solve.__doc__     = _gauss_jordan_solve.__doc__
    # 将 _pinv_solve 的文档字符串赋值给 pinv_solve 方法的文档字符串
    pinv_solve.__doc__             = _pinv_solve.__doc__
    # 将 _cramer_solve 的文档字符串赋值给 cramer_solve 方法的文档字符串
    cramer_solve.__doc__           = _cramer_solve.__doc__
    # 将 _solve 的文档字符串赋值给 solve 方法的文档字符串
    solve.__doc__                  = _solve.__doc__
    # 将 _solve_least_squares 的文档字符串赋值给 solve_least_squares 方法的文档字符串
    solve_least_squares.__doc__    = _solve_least_squares.__doc__

    # 将 _pinv 的文档字符串赋值给 pinv 方法的文档字符串
    pinv.__doc__                   = _pinv.__doc__
    # 将 _inv_ADJ 的文档字符串赋值给 inverse_ADJ 方法的文档字符串
    inverse_ADJ.__doc__            = _inv_ADJ.__doc__
    # 将 _inv_GE 的文档字符串赋值给 inverse_GE 方法的文档字符串
    inverse_GE.__doc__             = _inv_GE.__doc__
    # 将 _inv_LU 的文档字符串赋值给 inverse_LU 方法的文档字符串
    inverse_LU.__doc__             = _inv_LU.__doc__
    # 将 _inv_CH 的文档字符串赋值给 inverse_CH 方法的文档字符串
    inverse_CH.__doc__             = _inv_CH.__doc__
    # 将 _inv_LDL 的文档字符串赋值给 inverse_LDL 方法的文档字符串
    inverse_LDL.__doc__            = _inv_LDL.__doc__
    # 将 _inv_QR 的文档字符串赋值给 inverse_QR 方法的文档字符串
    inverse_QR.__doc__             = _inv_QR.__doc__
    # 将 _inv_block 的文档字符串赋值给 inverse_BLOCK 方法的文档字符串
    inverse_BLOCK.__doc__          = _inv_block.__doc__
    # 将 _inv 的文档字符串赋值给 inv 方法的文档字符串
    inv.__doc__                    = _inv.__doc__

    # 将 _connected_components 的文档字符串赋值给 connected_components 方法的文档字符串
    connected_components.__doc__   = _connected_components.__doc__
    # 将 _connected_components_decomposition 的文档字符串赋值给 connected_components_decomposition 方法的文档字符串
    connected_components_decomposition.__doc__ = \
        _connected_components_decomposition.__doc__
    # 将 _strongly_connected_components 的文档字符串赋值给 strongly_connected_components 方法的文档字符串
    strongly_connected_components.__doc__   = \
        _strongly_connected_components.__doc__
    # 将 _strongly_connected_components_decomposition 的文档字符串赋值给 strongly_connected_components_decomposition 方法的文档字符串
    strongly_connected_components_decomposition.__doc__ = \
        _strongly_connected_components_decomposition.__doc__
# 将矩阵 mat 转换为类型 typ 的矩阵对象
def _convert_matrix(typ, mat):
    from sympy.matrices.matrixbase import MatrixBase
    # 检查 mat 是否具有 is_Matrix 属性，但不是 MatrixBase 的实例
    if getattr(mat, "is_Matrix", False) and not isinstance(mat, MatrixBase):
        # 这段代码用于在 Matrix 和冗余的矩阵混合类型（如 _MinimalMatrix 等）之间进行交互操作。
        # 如果有人使用这些类型，则保证其继续工作。实际上，_MinimalMatrix 等应该被弃用和移除。
        return typ(*mat.shape, list(mat))
    else:
        return typ(mat)


# 检查对象 other 是否具有形状（shape）属性，且为二维元组
def _has_matrix_shape(other):
    shape = getattr(other, 'shape', None)
    if shape is None:
        return False
    return isinstance(shape, tuple) and len(shape) == 2


# 检查对象 other 是否具有 rows 和 cols 属性
def _has_rows_cols(other):
    return hasattr(other, 'rows') and hasattr(other, 'cols')


# 将 other 转换为一个矩阵，或检查是否可能为标量值
def _coerce_operand(self, other):
    """Convert other to a Matrix, or check for possible scalar."""
    INVALID = None, 'invalid_type'

    # 禁止混合 Matrix 和 NDimArray
    if isinstance(other, NDimArray):
        return INVALID

    is_Matrix = getattr(other, 'is_Matrix', None)

    # 如果 other 已经是 Matrix，则直接返回
    if is_Matrix:
        return other, 'is_matrix'

    # 尝试转换 numpy 数组、mpmath 矩阵等
    if is_Matrix is None:
        if _has_matrix_shape(other) or _has_rows_cols(other):
            return _convert_matrix(type(self), other), 'is_matrix'

    # 如果 other 不是可迭代对象，则可能是标量
    if not isinstance(other, Iterable):
        return other, 'possible_scalar'

    return INVALID


# 确定结合不同类型矩阵 A 和 B 时的结果类型
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
    if None not in (priority_A, priority_B):
        if A._class_priority > B._class_priority:
            return A.__class__
        else:
            return B.__class__

    try:
        import numpy
    except ImportError:
        pass
    else:
        if isinstance(A, numpy.ndarray):
            return B.__class__
        if isinstance(B, numpy.ndarray):
            return A.__class__

    raise TypeError("Incompatible classes %s, %s" % (A.__class__, B.__class__))


# 将 self 和 other 统一为同一种矩阵类型，或检查是否为标量
def _unify_with_other(self, other):
    """Unify self and other into a single matrix type, or check for scalar."""
    other, T = _coerce_operand(self, other)
    # 如果操作类型为 "is_matrix"
    if T == "is_matrix":
        # 获取对象 self 和 other 的类别
        typ = classof(self, other)
        # 如果 self 的类别不等于当前对象的类别
        if typ != self.__class__:
            # 将 self 转换为 typ 类别的矩阵
            self = _convert_matrix(typ, self)
        # 如果 other 的类别不等于当前对象的类别
        if typ != other.__class__:
            # 将 other 转换为 typ 类别的矩阵
            other = _convert_matrix(typ, other)
    
    # 返回经过处理的 self, other 对象和操作类型 T
    return self, other, T
# 定义函数 a2idx，用于处理索引转换和验证
def a2idx(j, n=None):
    """Return integer after making positive and validating against n."""
    # 如果 j 不是整数，则尝试获取其 __index__ 方法返回的整数值
    if not isinstance(j, int):
        jindex = getattr(j, '__index__', None)
        if jindex is not None:
            j = jindex()
        else:
            raise IndexError("Invalid index a[%r]" % (j,))
    
    # 如果传入了 n，确保 j 为非负数，并且在合法范围内
    if n is not None:
        if j < 0:
            j += n
        if not (j >= 0 and j < n):
            raise IndexError("Index out of range: a[%s]" % (j,))
    
    # 返回转换为整数后的 j
    return int(j)


class DeferredVector(Symbol, NotIterable):
    """A vector whose components are deferred (e.g. for use with lambdify).

    Examples
    ========

    >>> from sympy import DeferredVector, lambdify
    >>> X = DeferredVector( 'X' )
    >>> X
    X
    >>> expr = (X[0] + 2, X[2] + 3)
    >>> func = lambdify( X, expr)
    >>> func( [1, 2, 3] )
    (3, 6)
    """

    # 定义 __getitem__ 方法，处理向量的索引访问
    def __getitem__(self, i):
        # 将 -0 视为 0
        if i == -0:
            i = 0
        # 如果索引 i 小于 0，则抛出索引超出范围的异常
        if i < 0:
            raise IndexError('DeferredVector index out of range')
        # 构造分量名称，并返回对应的符号对象
        component_name = '%s[%d]' % (self.name, i)
        return Symbol(component_name)

    # 定义 __str__ 方法，返回 DeferredVector 对象的字符串表示
    def __str__(self):
        return sstr(self)

    # 定义 __repr__ 方法，返回 DeferredVector 对象的详细表示
    def __repr__(self):
        return "DeferredVector('%s')" % self.name
```