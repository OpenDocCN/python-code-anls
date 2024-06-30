# `D:\src\scipysrc\sympy\sympy\matrices\immutable.py`

```
# 从 mpmath.matrices.matrices 模块导入 _matrix 类
from mpmath.matrices.matrices import _matrix

# 从 sympy.core 模块导入 Basic, Dict, Tuple 类
from sympy.core import Basic, Dict, Tuple
# 从 sympy.core.numbers 模块导入 Integer 类
from sympy.core.numbers import Integer
# 从 sympy.core.cache 模块导入 cacheit 函数
from sympy.core.cache import cacheit
# 从 sympy.core.sympify 模块导入 sympify_converter, _sympify 函数
from sympy.core.sympify import _sympy_converter as sympify_converter, _sympify
# 从 sympy.matrices.dense 模块导入 DenseMatrix 类
from sympy.matrices.dense import DenseMatrix
# 从 sympy.matrices.expressions 模块导入 MatrixExpr 类
from sympy.matrices.expressions import MatrixExpr
# 从 sympy.matrices.matrixbase 模块导入 MatrixBase 类
from sympy.matrices.matrixbase import MatrixBase
# 从 sympy.matrices.repmatrix 模块导入 RepMatrix 类
from sympy.matrices.repmatrix import RepMatrix
# 从 sympy.matrices.sparse 模块导入 SparseRepMatrix 类
from sympy.matrices.sparse import SparseRepMatrix
# 从 sympy.multipledispatch 模块导入 dispatch 装饰器
from sympy.multipledispatch import dispatch


def sympify_matrix(arg):
    return arg.as_immutable()

# 将 MatrixBase 类型的对象转换为不可变形式
sympify_converter[MatrixBase] = sympify_matrix


def sympify_mpmath_matrix(arg):
    # 将 mpmath.matrices.matrices._matrix 类型的对象转换为不可变的 DenseMatrix 对象
    mat = [_sympify(x) for x in arg]
    return ImmutableDenseMatrix(arg.rows, arg.cols, mat)

# 将 mpmath.matrices.matrices._matrix 类型的对象转换为不可变的 DenseMatrix 对象
sympify_converter[_matrix] = sympify_mpmath_matrix


class ImmutableRepMatrix(RepMatrix, MatrixExpr): # type: ignore
    """Immutable matrix based on RepMatrix

    Uses DomainMatrix as the internal representation.
    """

    # 这是 RepMatrix 的子类，添加/覆盖了一些方法，使实例成为 Basic 和不可变的。ImmutableRepMatrix 是 ImmutableDenseMatrix 和 ImmutableSparseMatrix 的超类。

    def __new__(cls, *args, **kwargs):
        return cls._new(*args, **kwargs)

    # 设置 __hash__ 方法使用 MatrixExpr 的 __hash__ 方法
    __hash__ = MatrixExpr.__hash__

    # 返回对象本身的拷贝
    def copy(self):
        return self

    # 返回矩阵的列数
    @property
    def cols(self):
        return self._cols

    # 返回矩阵的行数
    @property
    def rows(self):
        return self._rows

    # 返回矩阵的形状 (行数, 列数)
    @property
    def shape(self):
        return self._rows, self._cols

    # 返回对象本身，表示它是不可变的
    def as_immutable(self):
        return self

    # 获取矩阵的元素 (i, j)，使用 __getitem__ 方法
    def _entry(self, i, j, **kwargs):
        return self[i, j]

    # 禁止设置对象的元素，抛出 TypeError
    def __setitem__(self, *args):
        raise TypeError("Cannot set values of {}".format(self.__class__))

    # 检查矩阵是否可对角化，可缓存结果
    def is_diagonalizable(self, reals_only=False, **kwargs):
        return super().is_diagonalizable(
            reals_only=reals_only, **kwargs)

    # 将 is_diagonalizable 方法标记为可缓存
    is_diagonalizable.__doc__ = SparseRepMatrix.is_diagonalizable.__doc__
    is_diagonalizable = cacheit(is_diagonalizable)

    # 对矩阵应用分析函数 f(x)，返回不可变的结果
    def analytic_func(self, f, x):
        return self.as_mutable().analytic_func(f, x).as_immutable()


class ImmutableDenseMatrix(DenseMatrix, ImmutableRepMatrix):  # type: ignore
    """Create an immutable version of a matrix.

    Examples
    ========

    >>> from sympy import eye, ImmutableMatrix
    >>> ImmutableMatrix(eye(3))
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> _[0, 0] = 42
    Traceback (most recent call last):
    ...
    TypeError: Cannot set values of ImmutableDenseMatrix
    """

    # MatrixExpr 被设置为 NotIterable，但我们希望显式矩阵是可迭代的
    _iterable = True
    _class_priority = 8
    _op_priority = 10.001

    @classmethod
    # 定义一个类方法 `_new`，用于创建新的实例
    def _new(cls, *args, **kwargs):
        # 如果参数个数为1且第一个参数是 ImmutableDenseMatrix 类型，则直接返回该实例
        if len(args) == 1 and isinstance(args[0], ImmutableDenseMatrix):
            return args[0]
        
        # 如果传入的关键字参数中存在 'copy'，且其值为 False，则执行以下逻辑
        if kwargs.get('copy', True) is False:
            # 如果参数个数不等于3，则抛出类型错误异常
            if len(args) != 3:
                raise TypeError("'copy=False' requires a matrix be initialized as rows,cols,[list]")
            # 将参数分解为行数 rows，列数 cols，和扁平列表 flat_list
            rows, cols, flat_list = args
        else:
            # 否则，使用类方法 `_handle_creation_inputs` 处理输入参数并获取 rows, cols, flat_list
            rows, cols, flat_list = cls._handle_creation_inputs(*args, **kwargs)
            # 创建 flat_list 的浅拷贝
            flat_list = list(flat_list)  # create a shallow copy
        
        # 将 flat_list 转换为 DomainMatrix 表示形式 rep
        rep = cls._flat_list_to_DomainMatrix(rows, cols, flat_list)

        # 调用类方法 `_fromrep` 根据 rep 创建并返回一个新的实例
        return cls._fromrep(rep)

    @classmethod
    # 定义一个类方法 `_fromrep`，用于从给定的表示 rep 创建实例
    def _fromrep(cls, rep):
        # 获取 rep 的行数和列数
        rows, cols = rep.shape
        # 将 rep 转换为 sympy 形式的扁平列表 flat_list
        flat_list = rep.to_sympy().to_list_flat()
        # 使用 Basic 类的 __new__ 方法创建一个新的对象 obj
        obj = Basic.__new__(cls,
            Integer(rows),
            Integer(cols),
            Tuple(*flat_list, sympify=False))
        # 设置 obj 的行数、列数和 rep 属性
        obj._rows = rows
        obj._cols = cols
        obj._rep = rep
        # 返回创建的对象 obj
        return obj
# 将ImmutableDenseMatrix重命名为ImmutableMatrix，作为别名
ImmutableMatrix = ImmutableDenseMatrix

# 创建一个ImmutableSparseMatrix类，继承自SparseRepMatrix和ImmutableRepMatrix，忽略类型检查
class ImmutableSparseMatrix(SparseRepMatrix, ImmutableRepMatrix):  # type:ignore
    """创建一个稀疏矩阵的不可变版本。

    Examples
    ========

    >>> from sympy import eye, ImmutableSparseMatrix
    >>> ImmutableSparseMatrix(1, 1, {})
    Matrix([[0]])
    >>> ImmutableSparseMatrix(eye(3))
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> _[0, 0] = 42
    Traceback (most recent call last):
    ...
    TypeError: Cannot set values of ImmutableSparseMatrix
    >>> _.shape
    (3, 3)
    """
    is_Matrix = True  # 表示这是一个矩阵类
    _class_priority = 9  # 类优先级设定为9，用于比较和排序

    @classmethod
    def _new(cls, *args, **kwargs):
        # 处理创建输入，返回行数、列数和稀疏矩阵表示
        rows, cols, smat = cls._handle_creation_inputs(*args, **kwargs)

        # 将稀疏矩阵表示转换为领域矩阵
        rep = cls._smat_to_DomainMatrix(rows, cols, smat)

        # 使用领域矩阵创建新的ImmutableSparseMatrix对象
        return cls._fromrep(rep)

    @classmethod
    def _fromrep(cls, rep):
        # 从领域矩阵表示中获取行数和列数
        rows, cols = rep.shape
        # 将领域矩阵表示转换为SymPy的dok_matrix
        smat = rep.to_sympy().to_dok()
        # 使用基础类Basic创建新的ImmutableSparseMatrix对象
        obj = Basic.__new__(cls, Integer(rows), Integer(cols), Dict(smat))
        obj._rows = rows  # 设置对象的行数属性
        obj._cols = cols  # 设置对象的列数属性
        obj._rep = rep  # 设置对象的领域矩阵表示属性
        return obj


@dispatch(ImmutableDenseMatrix, ImmutableDenseMatrix)
def _eval_is_eq(lhs, rhs): # noqa:F811
    """用于比较两个ImmutableDenseMatrix矩阵是否相等的辅助方法。

    Relational会自动将矩阵转换为ImmutableDenseMatrix实例，因此这个方法仅适用于这里。
    如果矩阵明确相同则返回True，如果明确不同则返回False，如果不确定则返回None（例如，它们包含符号）。
    返回None会触发默认的等式处理方式。

    """
    if lhs.shape != rhs.shape:  # 检查矩阵形状是否相同
        return False
    return (lhs - rhs).is_zero_matrix  # 返回矩阵差是否为零矩阵
```