# `D:\src\scipysrc\sympy\sympy\stats\symbolic_multivariate_probability.py`

```
# 导入 itertools 模块，用于生成迭代器的函数
import itertools

# 导入 sympy 核心模块中的特定类和函数
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.singleton import S

# 导入 sympy 矩阵模块中的异常类
from sympy.matrices.exceptions import ShapeError

# 导入 sympy 矩阵表达式相关的类和函数
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import ZeroMatrix

# 导入 sympy 统计模块中的随机变量相关类和函数
from sympy.stats.rv import RandomSymbol, is_random

# 导入 sympy 核心模块中的函数，用于将输入转换为 sympy 表达式
from sympy.core.sympify import _sympify

# 导入 sympy 统计模块中的概率相关类
from sympy.stats.symbolic_probability import Variance, Covariance, Expectation


class ExpectationMatrix(Expectation, MatrixExpr):
    """
    Expectation of a random matrix expression.

    Examples
    ========

    >>> from sympy.stats import ExpectationMatrix, Normal
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> ExpectationMatrix(X)
    ExpectationMatrix(X)
    >>> ExpectationMatrix(A*X).shape
    (k, 1)

    To expand the expectation in its expression, use ``expand()``:

    >>> ExpectationMatrix(A*X + B*Y).expand()
    A*ExpectationMatrix(X) + B*ExpectationMatrix(Y)
    >>> ExpectationMatrix((X + Y)*(X - Y).T).expand()
    ExpectationMatrix(X*X.T) - ExpectationMatrix(X*Y.T) + ExpectationMatrix(Y*X.T) - ExpectationMatrix(Y*Y.T)

    To evaluate the ``ExpectationMatrix``, use ``doit()``:

    >>> N11, N12 = Normal('N11', 11, 1), Normal('N12', 12, 1)
    >>> N21, N22 = Normal('N21', 21, 1), Normal('N22', 22, 1)
    >>> M11, M12 = Normal('M11', 1, 1), Normal('M12', 2, 1)
    >>> M21, M22 = Normal('M21', 3, 1), Normal('M22', 4, 1)
    >>> x1 = Matrix([[N11, N12], [N21, N22]])
    >>> x2 = Matrix([[M11, M12], [M21, M22]])
    >>> ExpectationMatrix(x1 + x2).doit()
    Matrix([
    [12, 14],
    [24, 26]])

    """

    def __new__(cls, expr, condition=None):
        # 将输入表达式转换为 sympy 表达式
        expr = _sympify(expr)

        # 如果未提供条件，则检查表达式是否是随机变量；如果不是，则返回原表达式
        if condition is None:
            if not is_random(expr):
                return expr
            # 创建一个新的 ExpectationMatrix 对象
            obj = Expr.__new__(cls, expr)
        else:
            # 将条件转换为 sympy 表达式
            condition = _sympify(condition)
            obj = Expr.__new__(cls, expr, condition)

        # 设置对象的形状属性为表达式的形状
        obj._shape = expr.shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        # 返回对象的形状属性
        return self._shape
    # 定义一个方法 `expand`，接受关键字参数 `hints`
    def expand(self, **hints):
        # 获取表达式对象，通常是函数参数列表中的第一个表达式
        expr = self.args[0]
        # 获取条件对象，可能是类内部的私有属性 `_condition`
        condition = self._condition
        
        # 如果表达式不是随机变量，则直接返回表达式本身
        if not is_random(expr):
            return expr

        # 如果表达式是加法表达式（Add 类的实例）
        if isinstance(expr, Add):
            # 对加法表达式中的每个项进行期望值操作的展开，并返回新的加法表达式
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expr.args)

        # 对表达式进行展开操作，返回展开后的表达式
        expand_expr = _expand(expr)
        
        # 如果展开后的表达式是加法表达式
        if isinstance(expand_expr, Add):
            # 对加法表达式中的每个项进行期望值操作的展开，并返回新的加法表达式
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expand_expr.args)

        # 如果表达式是乘法表达式（Mul 或 MatMul 类的实例）
        elif isinstance(expr, (Mul, MatMul)):
            rv = []      # 存储随机变量的列表
            nonrv = []   # 存储非随机变量的列表
            postnon = [] # 存储后续非随机变量的列表

            # 遍历乘法表达式中的每个项
            for a in expr.args:
                if is_random(a):     # 如果当前项是随机变量
                    if rv:
                        rv.extend(postnon)  # 如果已有随机变量，将后续非随机变量加入到随机变量列表
                    else:
                        nonrv.extend(postnon)  # 否则加入到非随机变量列表
                    postnon = []  # 重置后续非随机变量列表
                    rv.append(a)  # 将当前随机变量加入到随机变量列表
                elif a.is_Matrix:   # 如果当前项是矩阵
                    postnon.append(a)  # 将其加入到后续非随机变量列表
                else:
                    nonrv.append(a)  # 否则加入到非随机变量列表

            # 如果没有剩余的非随机变量，为避免无限循环（MatMul 可能再次调用 .doit()），直接返回自身
            if len(nonrv) == 0:
                return self
            # 返回由剩余非随机变量乘积、随机变量乘积的期望值、以及剩余非随机变量乘积构成的新乘积表达式
            return Mul.fromiter(nonrv)*Expectation(Mul.fromiter(rv),
                    condition=condition)*Mul.fromiter(postnon)

        return self  # 默认情况下返回自身，未做任何改变
class VarianceMatrix(Variance, MatrixExpr):
    """
    Variance of a random matrix probability expression. Also known as
    Covariance matrix, auto-covariance matrix, dispersion matrix,
    or variance-covariance matrix.

    Examples
    ========

    >>> from sympy.stats import VarianceMatrix
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> VarianceMatrix(X)
    VarianceMatrix(X)
    >>> VarianceMatrix(X).shape
    (k, k)

    To expand the variance in its expression, use ``expand()``:

    >>> VarianceMatrix(A*X).expand()
    A*VarianceMatrix(X)*A.T
    >>> VarianceMatrix(A*X + B*Y).expand()
    2*A*CrossCovarianceMatrix(X, Y)*B.T + A*VarianceMatrix(X)*A.T + B*VarianceMatrix(Y)*B.T
    """
    def __new__(cls, arg, condition=None):
        # 将参数符号化
        arg = _sympify(arg)

        # 检查表达式是否为向量
        if 1 not in arg.shape:
            raise ShapeError("Expression is not a vector")

        # 根据参数形状设置矩阵形状
        shape = (arg.shape[0], arg.shape[0]) if arg.shape[1] == 1 else (arg.shape[1], arg.shape[1])

        # 根据是否有条件，创建新对象
        if condition:
            obj = Expr.__new__(cls, arg, condition)
        else:
            obj = Expr.__new__(cls, arg)

        # 设置对象的形状和条件属性
        obj._shape = shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        # 返回对象的形状
        return self._shape

    def expand(self, **hints):
        # 获取参数和条件
        arg = self.args[0]
        condition = self._condition

        # 如果参数不是随机变量，则返回零矩阵
        if not is_random(arg):
            return ZeroMatrix(*self.shape)

        # 如果参数是随机符号，则直接返回对象自身
        if isinstance(arg, RandomSymbol):
            return self
        elif isinstance(arg, Add):
            rv = []
            # 将参数中的随机变量收集起来
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
            # 计算各个随机变量的方差
            variances = Add(*(Variance(xv, condition).expand() for xv in rv))
            # 计算各个随机变量之间的协方差
            map_to_covar = lambda x: 2*Covariance(*x, condition=condition).expand()
            covariances = Add(*map(map_to_covar, itertools.combinations(rv, 2)))
            # 返回扩展后的表达式
            return variances + covariances
        elif isinstance(arg, (Mul, MatMul)):
            nonrv = []
            rv = []
            # 将参数中的随机变量和非随机变量分开
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a)
            # 如果没有随机变量，则返回零矩阵
            if len(rv) == 0:
                return ZeroMatrix(*self.shape)
            # 对于全是随机变量的情况，避免可能的无限循环
            if len(nonrv) == 0:
                return self
            # 目前尚未实现多个矩阵乘积的方差
            if len(rv) > 1:
                return self
            # 返回扩展后的表达式
            return Mul.fromiter(nonrv)*Variance(Mul.fromiter(rv),
                            condition)*(Mul.fromiter(nonrv)).transpose()

        # 如果表达式中包含随机符号，则返回对象自身
        return self
    Covariance of a random matrix probability expression.

    Examples
    ========

    >>> from sympy.stats import CrossCovarianceMatrix  # 导入交叉协方差矩阵类
    >>> from sympy.stats.rv import RandomMatrixSymbol  # 导入随机矩阵符号类
    >>> from sympy import symbols, MatrixSymbol  # 导入符号变量类和矩阵符号类
    >>> k = symbols("k")  # 创建符号变量 k
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)  # 创建矩阵符号 A 和 B，形状为 k x k
    >>> C, D = MatrixSymbol("C", k, k), MatrixSymbol("D", k, k)  # 创建矩阵符号 C 和 D，形状为 k x k
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)  # 创建随机矩阵符号 X 和 Y，形状为 k x 1
    >>> Z, W = RandomMatrixSymbol("Z", k, 1), RandomMatrixSymbol("W", k, 1)  # 创建随机矩阵符号 Z 和 W，形状为 k x 1
    >>> CrossCovarianceMatrix(X, Y)  # 创建 X 和 Y 的交叉协方差矩阵对象
    CrossCovarianceMatrix(X, Y)
    >>> CrossCovarianceMatrix(X, Y).shape  # 获取交叉协方差矩阵对象的形状
    (k, k)

    To expand the covariance in its expression, use ``expand()``:

    >>> CrossCovarianceMatrix(X + Y, Z).expand()  # 对 X+Y 和 Z 的交叉协方差矩阵对象进行展开操作
    CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z)
    >>> CrossCovarianceMatrix(A*X, Y).expand()  # 对 A*X 和 Y 的交叉协方差矩阵对象进行展开操作
    A*CrossCovarianceMatrix(X, Y)
    >>> CrossCovarianceMatrix(A*X, B.T*Y).expand()  # 对 A*X 和 B^T*Y 的交叉协方差矩阵对象进行展开操作
    A*CrossCovarianceMatrix(X, Y)*B
    >>> CrossCovarianceMatrix(A*X + B*Y, C.T*Z + D.T*W).expand()  # 对 A*X + B*Y 和 C^T*Z + D^T*W 的交叉协方差矩阵对象进行展开操作
    A*CrossCovarianceMatrix(X, W)*D + A*CrossCovarianceMatrix(X, Z)*C + B*CrossCovarianceMatrix(Y, W)*D + B*CrossCovarianceMatrix(Y, Z)*C

    """
    def __new__(cls, arg1, arg2, condition=None):
        arg1 = _sympify(arg1)  # 对 arg1 进行符号化处理
        arg2 = _sympify(arg2)  # 对 arg2 进行符号化处理

        if (1 not in arg1.shape) or (1 not in arg2.shape) or (arg1.shape[1] != arg2.shape[1]):
            raise ShapeError("Expression is not a vector")  # 如果 arg1 或 arg2 不是向量，抛出形状错误异常

        shape = (arg1.shape[0], arg2.shape[0]) if arg1.shape[1] == 1 and arg2.shape[1] == 1 \
                    else (1, 1)  # 根据 arg1 和 arg2 的形状确定交叉协方差矩阵的形状

        if condition:
            obj = Expr.__new__(cls, arg1, arg2, condition)  # 如果存在条件，创建包含条件的表达式对象
        else:
            obj = Expr.__new__(cls, arg1, arg2)  # 否则，创建不包含条件的表达式对象

        obj._shape = shape  # 将形状信息存储在对象中
        obj._condition = condition  # 存储条件信息在对象中
        return obj

    @property
    def shape(self):
        return self._shape  # 返回对象的形状属性

    def expand(self, **hints):
        arg1 = self.args[0]  # 获取第一个参数
        arg2 = self.args[1]  # 获取第二个参数
        condition = self._condition  # 获取条件

        if arg1 == arg2:
            return VarianceMatrix(arg1, condition).expand()  # 如果 arg1 等于 arg2，则返回 arg1 的方差矩阵展开结果

        if not is_random(arg1) or not is_random(arg2):
            return ZeroMatrix(*self.shape)  # 如果 arg1 或 arg2 不是随机符号，则返回形状对应的零矩阵

        if isinstance(arg1, RandomSymbol) and isinstance(arg2, RandomSymbol):
            return CrossCovarianceMatrix(arg1, arg2, condition)  # 如果 arg1 和 arg2 都是随机符号，则返回它们的交叉协方差矩阵对象

        coeff_rv_list1 = self._expand_single_argument(arg1.expand())  # 对 arg1 展开并获取系数随机变量列表
        coeff_rv_list2 = self._expand_single_argument(arg2.expand())  # 对 arg2 展开并获取系数随机变量列表

        addends = [a*CrossCovarianceMatrix(r1, r2, condition=condition)*b.transpose()
                   for (a, r1) in coeff_rv_list1 for (b, r2) in coeff_rv_list2]  # 计算所有可能的交叉项
        return Add.fromiter(addends)  # 返回所有交叉项的和

    @classmethod
    # 对单个参数进行扩展的方法，用于返回 (系数, 随机符号) 的列表：
    def _expand_single_argument(cls, expr):
        # 如果参数是 RandomSymbol 类型，则返回包含 (1, expr) 的列表
        if isinstance(expr, RandomSymbol):
            return [(S.One, expr)]
        # 如果参数是 Add 类型
        elif isinstance(expr, Add):
            outval = []
            # 遍历 Add 类型表达式的每个子表达式
            for a in expr.args:
                # 如果子表达式是 Mul 或 MatMul 类型
                if isinstance(a, (Mul, MatMul)):
                    # 调用 _get_mul_nonrv_rv_tuple 方法处理 Mul 或 MatMul 类型的表达式
                    outval.append(cls._get_mul_nonrv_rv_tuple(a))
                # 如果子表达式是随机变量
                elif is_random(a):
                    # 将 (1, a) 添加到结果列表中
                    outval.append((S.One, a))

            return outval
        # 如果参数是 Mul 或 MatMul 类型
        elif isinstance(expr, (Mul, MatMul)):
            # 返回调用 _get_mul_nonrv_rv_tuple 方法处理后的结果
            return [cls._get_mul_nonrv_rv_tuple(expr)]
        # 如果参数是随机变量
        elif is_random(expr):
            # 返回包含 (1, expr) 的列表
            return [(S.One, expr)]

    @classmethod
    def _get_mul_nonrv_rv_tuple(cls, m):
        # 初始化随机变量和非随机变量的列表
        rv = []
        nonrv = []
        # 遍历 Mul 或 MatMul 类型表达式的每个子表达式
        for a in m.args:
            # 如果子表达式是随机变量
            if is_random(a):
                # 将随机变量添加到 rv 列表中
                rv.append(a)
            else:
                # 将非随机变量添加到 nonrv 列表中
                nonrv.append(a)
        # 返回元组，第一个元素为非随机变量构成的乘积，第二个元素为随机变量构成的乘积
        return (Mul.fromiter(nonrv), Mul.fromiter(rv))
```