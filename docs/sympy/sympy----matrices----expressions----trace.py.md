# `D:\src\scipysrc\sympy\sympy\matrices\expressions\trace.py`

```
`
# 导入必要的类和函数
from sympy.core.basic import Basic
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.exceptions import NonSquareMatrixError


class Trace(Expr):
    """Matrix Trace

    表示矩阵表达式的迹。

    Examples
    ========

    >>> from sympy import MatrixSymbol, Trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Trace(A)
    Trace(A)
    >>> Trace(eye(3))
    Trace(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> Trace(eye(3)).simplify()
    3
    """
    
    # 设置类属性
    is_Trace = True  # 表示这是一个 Trace 类型的表达式
    is_commutative = True  # 表示可以进行交换运算

    def __new__(cls, mat):
        # 将输入矩阵 mat 转换为 sympy 中的表达式
        mat = sympify(mat)

        # 检查输入是否为矩阵
        if not mat.is_Matrix:
            raise TypeError("input to Trace, %s, is not a matrix" % str(mat))

        # 检查矩阵是否为方阵
        if mat.is_square is False:
            raise NonSquareMatrixError("Trace of a non-square matrix")

        # 使用 Basic 类的构造函数创建一个新的 Trace 对象
        return Basic.__new__(cls, mat)

    def _eval_transpose(self):
        # 迹的转置操作，迹不受转置影响，直接返回自身
        return self

    def _eval_derivative(self, v):
        # 导数运算
        from sympy.concrete.summations import Sum
        from .matexpr import MatrixElement

        if isinstance(v, MatrixElement):
            # 如果 v 是 MatrixElement 类型，则使用 Sum 进行重写并对其求导
            return self.rewrite(Sum).diff(v)

        expr = self.doit()
        if isinstance(expr, Trace):
            # 避免无限循环，如果表达式是迹本身，则抛出未实现的错误
            raise NotImplementedError

        # 否则，对表达式进行求导
        return expr._eval_derivative(v)
    # 计算导数矩阵的行表达式，返回一个列表
    def _eval_derivative_matrix_lines(self, x):
        # 导入必要的模块和类
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayContraction
        # 递归调用第一个参数对象的_eval_derivative_matrix_lines方法，获取结果
        r = self.args[0]._eval_derivative_matrix_lines(x)
        # 遍历结果列表中的每一个元素lr
        for lr in r:
            # 如果lr对象的higher属性为1，执行以下操作
            if lr.higher == 1:
                # 将lr.higher属性替换为一个新的表达式构造器对象，用于执行ArrayContraction操作
                lr.higher = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                lr._lines[0],
                                lr._lines[1],
                            ]
                        ),
                        (1, 3),
                    ],
                    validator=ArrayContraction._validate
                )
            else:
                # 如果lr对象不是矩阵行：
                # 将lr.higher属性替换为一个新的表达式构造器对象，用于执行ArrayContraction操作
                lr.higher = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                lr._lines[0],
                                lr._lines[1],
                                lr.higher,
                            ]
                        ),
                        (1, 3), (0, 2)
                    ]
                )
            # 将lr._lines属性设置为[S.One, S.One]
            lr._lines = [S.One, S.One]
            # 设置lr._first_pointer_parent和lr._second_pointer_parent属性为lr._lines
            lr._first_pointer_parent = lr._lines
            lr._second_pointer_parent = lr._lines
            # 设置lr._first_pointer_index和lr._second_pointer_index属性分别为0和1
            lr._first_pointer_index = 0
            lr._second_pointer_index = 1
        # 返回处理后的结果列表r
        return r

    # 返回对象的第一个参数arg
    @property
    def arg(self):
        return self.args[0]

    # 执行对象的运算，根据hints参数的设置决定是否深度处理
    def doit(self, **hints):
        # 如果hints中指定deep为True，执行以下操作
        if hints.get('deep', True):
            # 对对象的第一个参数调用doit方法，获取结果
            arg = self.arg.doit(**hints)
            # 调用结果对象的_eval_trace方法，获取结果
            result = arg._eval_trace()
            # 如果结果不为None，直接返回结果
            if result is not None:
                return result
            else:
                # 如果结果为None，返回一个Trace对象，参数为arg
                return Trace(arg)
        else:
            # 如果hints中指定deep不为True，执行以下操作
            # 如果对象的第一个参数是MatrixBase类型，返回其trace
            if isinstance(self.arg, MatrixBase):
                return trace(self.arg)
            else:
                # 否则返回一个Trace对象，参数为arg
                return Trace(self.arg)

    # 返回对象参数arg的显式表达式的Trace
    def as_explicit(self):
        return Trace(self.arg.as_explicit()).doit()
    # 对矩阵乘积的迹进行标准化处理。利用转置和迹的循环性质，确保矩阵乘积的参数被排序，
    # 并且第一个参数不是一个转置。
    def _normalize(self):
        # 导入需要的类和函数
        from sympy.matrices.expressions.matmul import MatMul
        from sympy.matrices.expressions.transpose import Transpose
        # 获取迹的参数
        trace_arg = self.arg
        # 如果迹的参数是一个 MatMul 对象
        if isinstance(trace_arg, MatMul):

            # 定义一个函数，用于获取排序键值
            def get_arg_key(x):
                a = trace_arg.args[x]
                # 如果是一个转置，则取其参数
                if isinstance(a, Transpose):
                    a = a.arg
                return default_sort_key(a)

            # 找到参数列表中排序键值最小的索引
            indmin = min(range(len(trace_arg.args)), key=get_arg_key)
            # 如果最小键值对应的参数是一个转置
            if isinstance(trace_arg.args[indmin], Transpose):
                # 对整个表达式进行转置并求值
                trace_arg = Transpose(trace_arg).doit()
                # 重新计算排序键值最小的索引
                indmin = min(range(len(trace_arg.args)), key=lambda x: default_sort_key(trace_arg.args[x]))
            # 重新排列参数列表，确保从最小键值索引开始，保持循环性质
            trace_arg = MatMul.fromiter(trace_arg.args[indmin:] + trace_arg.args[:indmin])
            # 返回一个新的 Trace 对象，其参数为重新排列后的矩阵乘积
            return Trace(trace_arg)
        # 如果迹的参数不是一个 MatMul 对象，则返回自身
        return self

    # 将迹表达式重写为求和表达式
    def _eval_rewrite_as_Sum(self, expr, **kwargs):
        # 导入需要的类和函数
        from sympy.concrete.summations import Sum
        # 使用唯一命名的符号 'i'，在表达式中创建一个求和
        i = uniquely_named_symbol('i', [expr])
        # 构建求和表达式，对迹的参数进行求和，范围为 0 到 self.arg.rows - 1
        s = Sum(self.arg[i, i], (i, 0, self.arg.rows - 1))
        # 求和表达式求值
        return s.doit()
# 定义一个函数用于计算矩阵的迹（对角线元素之和）
def trace(expr):
    """Trace of a Matrix.  Sum of the diagonal elements.
    
    Examples
    ========
    
    >>> from sympy import trace, Symbol, MatrixSymbol, eye
    >>> n = Symbol('n')
    >>> X = MatrixSymbol('X', n, n)  # A square matrix
    >>> trace(2*X)
    2*Trace(X)
    >>> trace(eye(3))
    3
    """
    # 调用 SymPy 中的 Trace 对象，并计算其值
    return Trace(expr).doit()
```