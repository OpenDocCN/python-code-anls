# `D:\src\scipysrc\sympy\sympy\matrices\expressions\matadd.py`

```
# 导入 functools 库中的 reduce 函数和 operator 模块
from functools import reduce
import operator

# 导入 sympy 库中的相关模块和函数
from sympy.core import Basic, sympify
from sympy.core.add import add, Add, _could_extract_minus_sign
from sympy.core.sorting import default_sort_key
from sympy.functions import adjoint
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.expressions.transpose import transpose
from sympy.strategies import (rm_id, unpack, flatten, sort, condition,
    exhaust, do_one, glom)
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix, GenericZeroMatrix
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning

# 定义一个矩阵加法类 MatAdd，继承自 MatrixExpr 和 Add
# XXX: MatAdd 应该可能不直接从 Add 继承
class MatAdd(MatrixExpr, Add):
    """A Sum of Matrix Expressions

    MatAdd inherits from and operates like SymPy Add

    Examples
    ========

    >>> from sympy import MatAdd, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> C = MatrixSymbol('C', 5, 5)
    >>> MatAdd(A, B, C)
    A + B + C
    """
    # 标识这是一个 MatAdd 类型的对象
    is_MatAdd = True

    # 定义 MatAdd 类的单位元为 GenericZeroMatrix
    identity = GenericZeroMatrix()

    # 构造函数，接受一系列参数并创建 MatAdd 对象
    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        # 如果没有参数传入，则返回单位元素
        if not args:
            return cls.identity

        # 从参数列表中移除单位元素，以避免在构造函数中引发 TypeError
        args = list(filter(lambda i: cls.identity != i, args))
        # 如果 _sympify 为 True，则将参数转换为 SymPy 对象
        if _sympify:
            args = list(map(sympify, args))

        # 检查所有参数是否都是 MatrixExpr 类型，如果不是则引发 TypeError
        if not all(isinstance(arg, MatrixExpr) for arg in args):
            raise TypeError("Mix of Matrix and Scalar symbols")

        # 创建一个基本对象 obj
        obj = Basic.__new__(cls, *args)

        # 如果传入了 check 参数，则发出 sympy_deprecation_warning 警告
        if check is not None:
            sympy_deprecation_warning(
                "Passing check to MatAdd is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        # 如果 check 不为 False，则调用 validate 函数验证参数
        if check is not False:
            validate(*args)

        # 如果 evaluate 为 True，则调用 _evaluate 方法对对象进行评估
        if evaluate:
            obj = cls._evaluate(obj)

        return obj

    # 类方法，用于评估表达式
    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)

    # 返回 MatAdd 对象的形状
    @property
    def shape(self):
        return self.args[0].shape

    # 检查是否可以提取负号
    def could_extract_minus_sign(self):
        return _could_extract_minus_sign(self)

    # 展开 MatAdd 对象
    def expand(self, **kwargs):
        expanded = super(MatAdd, self).expand(**kwargs)
        return self._evaluate(expanded)

    # 返回矩阵中指定位置 (i, j) 的条目
    def _entry(self, i, j, **kwargs):
        return Add(*[arg._entry(i, j, **kwargs) for arg in self.args])

    # 返回 MatAdd 对象的转置
    def _eval_transpose(self):
        return MatAdd(*[transpose(arg) for arg in self.args]).doit()

    # 返回 MatAdd 对象的伴随矩阵
    def _eval_adjoint(self):
        return MatAdd(*[adjoint(arg) for arg in self.args]).doit()
    # 计算表达式的迹（trace），迹是矩阵对角线上元素的和
    def _eval_trace(self):
        # 导入跟踪（trace）函数，位于当前包中的 trace 模块中
        from .trace import trace
        # 对表达式中的每个参数调用 trace 函数，并将结果作为 Add 对象的参数传入，然后计算结果
        return Add(*[trace(arg) for arg in self.args]).doit()

    # 执行操作的规范化（canonicalization），并返回结果
    def doit(self, **hints):
        # 获取 hints 参数中的 deep 值，默认为 True
        deep = hints.get('deep', True)
        # 如果 deep 为 True，则对每个参数递归调用 doit 方法；否则直接使用当前对象的参数列表
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        # 创建 MatAdd 对象，并对其进行规范化操作，返回规范化后的结果
        return canonicalize(MatAdd(*args))

    # 计算关于变量 x 的导数矩阵的每行，并将结果合并为一个列表返回
    def _eval_derivative_matrix_lines(self, x):
        # 对当前对象的每个参数调用 _eval_derivative_matrix_lines 方法，得到一个列表的列表
        add_lines = [arg._eval_derivative_matrix_lines(x) for arg in self.args]
        # 将嵌套的列表展开为单层列表，并返回结果
        return [j for i in add_lines for j in i]
# 定义一个函数式注册器，注册 Add 和 MatAdd 类的处理程序
add.register_handlerclass((Add, MatAdd), MatAdd)

# 定义一个 lambda 函数，用于提取参数的系数
factor_of = lambda arg: arg.as_coeff_mmul()[0]

# 定义一个 lambda 函数，用于解包参数的矩阵部分
matrix_of = lambda arg: unpack(arg.as_coeff_mmul()[1])

# 定义一个函数 combine，根据给定的计数和矩阵返回计算结果
def combine(cnt, mat):
    if cnt == 1:
        return mat
    else:
        return cnt * mat

# 定义函数 merge_explicit，用于合并显式的 MatrixBase 参数
def merge_explicit(matadd):
    """ Merge explicit MatrixBase arguments

    Examples
    ========

    >>> from sympy import MatrixSymbol, eye, Matrix, MatAdd, pprint
    >>> from sympy.matrices.expressions.matadd import merge_explicit
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = eye(2)
    >>> C = Matrix([[1, 2], [3, 4]])
    >>> X = MatAdd(A, B, C)
    >>> pprint(X)
        [1  0]   [1  2]
    A + [    ] + [    ]
        [0  1]   [3  4]
    >>> pprint(merge_explicit(X))
        [2  2]
    A + [    ]
        [3  5]
    """
    # 使用 sift 函数将 matadd.args 中的 MatrixBase 类型参数分组
    groups = sift(matadd.args, lambda arg: isinstance(arg, MatrixBase))
    # 如果 MatrixBase 参数的数量大于 1，则将它们合并成一个 MatAdd 对象返回
    if len(groups[True]) > 1:
        return MatAdd(*(groups[False] + [reduce(operator.add, groups[True])]))
    else:
        return matadd

# 定义规则列表，用于规范化处理 MatAdd 类型的表达式
rules = (
    rm_id(lambda x: x == 0 or isinstance(x, ZeroMatrix)),  # 移除值为零或 ZeroMatrix 类型的元素
    unpack,  # 解包
    flatten,  # 扁平化
    glom(matrix_of, factor_of, combine),  # 组合函数处理
    merge_explicit,  # 合并显式参数
    sort(default_sort_key)  # 使用默认排序键进行排序
)

# 定义函数 canonicalize，将规则应用于条件为 MatAdd 类型的表达式，进行归一化处理
canonicalize = exhaust(condition(lambda x: isinstance(x, MatAdd),
                                 do_one(*rules)))
```