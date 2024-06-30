# `D:\src\scipysrc\sympy\sympy\matrices\expressions\applyfunc.py`

```
# 导入符号计算库中所需的模块和类
from sympy.core.expr import ExprBuilder
from sympy.core.function import (Function, FunctionClass, Lambda)
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.matrices.matrixbase import MatrixBase

# 定义一个继承自 MatrixExpr 的类 ElementwiseApplyFunction
class ElementwiseApplyFunction(MatrixExpr):
    r"""
    Apply function to a matrix elementwise without evaluating.

    Examples
    ========

    It can be created by calling ``.applyfunc(<function>)`` on a matrix
    expression:

    >>> from sympy import MatrixSymbol
    >>> from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    >>> from sympy import exp
    >>> X = MatrixSymbol("X", 3, 3)
    >>> X.applyfunc(exp)
    Lambda(_d, exp(_d)).(X)

    Otherwise using the class constructor:

    >>> from sympy import eye
    >>> expr = ElementwiseApplyFunction(exp, eye(3))
    >>> expr
    Lambda(_d, exp(_d)).(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> expr.doit()
    Matrix([
    [E, 1, 1],
    [1, E, 1],
    [1, 1, E]])

    Notice the difference with the real mathematical functions:

    >>> exp(eye(3))
    Matrix([
    [E, 0, 0],
    [0, E, 0],
    [0, 0, E]])
    """

    # 定义类的构造函数，接受一个函数和一个表达式作为参数
    def __new__(cls, function, expr):
        # 将表达式转换为符号对象
        expr = _sympify(expr)
        # 检查表达式是否为矩阵
        if not expr.is_Matrix:
            raise ValueError("{} must be a matrix instance.".format(expr))

        # 如果表达式是 1x1 矩阵
        if expr.shape == (1, 1):
            # 检查函数是否返回一个矩阵，如果是，则直接应用函数而不创建 ElementwiseApplyFunc 对象
            ret = function(expr)
            if isinstance(ret, MatrixExpr):
                return ret

        # 检查传入的函数是否为 FunctionClass 或 Lambda 类型
        if not isinstance(function, (FunctionClass, Lambda)):
            d = Dummy('d')
            function = Lambda(d, function(d))

        # 将函数转换为符号对象
        function = sympify(function)
        # 再次检查函数类型是否为 FunctionClass 或 Lambda
        if not isinstance(function, (FunctionClass, Lambda)):
            raise ValueError(
                "{} should be compatible with SymPy function classes."
                .format(function))

        # 检查函数是否接受一个参数
        if 1 not in function.nargs:
            raise ValueError(
                '{} should be able to accept 1 arguments.'.format(function))

        # 如果函数不是 Lambda 类型，则创建一个 Lambda 对象
        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))

        # 调用 MatrixExpr 的构造函数创建对象
        obj = MatrixExpr.__new__(cls, function, expr)
        return obj

    # 定义 function 属性，返回第一个参数，即应用的函数
    @property
    def function(self):
        return self.args[0]

    # 定义 expr 属性，返回第二个参数，即矩阵表达式
    @property
    def expr(self):
        return self.args[1]

    # 定义 shape 属性，返回表达式的形状
    @property
    def shape(self):
        return self.expr.shape
    # 执行函数，处理参数 hints 作为深度参数
    def doit(self, **hints):
        # 从 hints 中获取 deep 参数，默认为 True
        deep = hints.get("deep", True)
        # 复制表达式到本地变量 expr
        expr = self.expr
        # 如果 deep 为 True，则递归调用 doit 方法获取表达式的结果
        if deep:
            expr = expr.doit(**hints)
        # 复制函数到本地变量 function
        function = self.function
        # 如果 function 是 Lambda 类型且是 identity 函数，则返回表达式
        if isinstance(function, Lambda) and function.is_identity:
            # 这是包含 identity 函数的 Lambda
            return expr
        # 如果 expr 是 MatrixBase 类型，则应用 function 到每个元素
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(self.function)
        # 如果 expr 是 ElementwiseApplyFunction 类型
        elif isinstance(expr, ElementwiseApplyFunction):
            # 创建新的 ElementwiseApplyFunction 对象，应用 self.function 到 expr.function(x)
            return ElementwiseApplyFunction(
                lambda x: self.function(expr.function(x)),
                expr.expr
            ).doit(**hints)
        else:
            # 其他情况返回自身对象
            return self

    # 计算并返回表达式的 (i, j) 位置的函数应用结果
    def _entry(self, i, j, **kwargs):
        return self.function(self.expr._entry(i, j, **kwargs))

    # 获取函数的一阶导数
    def _get_function_fdiff(self):
        # 创建虚拟符号 d
        d = Dummy("d")
        # 对虚拟符号 d 应用 self.function 函数
        function = self.function(d)
        # 计算函数对 d 的导数
        fdiff = function.diff(d)
        # 如果导数结果是 Function 类型，则将 fdiff 设为其类型
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            # 否则创建 Lambda 函数表示导数
            fdiff = Lambda(d, fdiff)
        return fdiff

    # 计算并返回关于变量 x 的导数结果
    def _eval_derivative(self, x):
        # 计算表达式关于变量 x 的导数
        dexpr = self.expr.diff(x)
        # 获取函数的一阶导数
        fdiff = self._get_function_fdiff()
        # 计算哈达玛积，应用 fdiff 到 self.expr 的每个元素
        return hadamard_product(
            dexpr,
            ElementwiseApplyFunction(fdiff, self.expr)
        )
    # 定义一个方法来计算导数矩阵的表达式行
    def _eval_derivative_matrix_lines(self, x):
        # 导入必要的模块和类
        from sympy.matrices.expressions.special import Identity
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct

        # 获取函数的导数
        fdiff = self._get_function_fdiff()
        # 计算当前表达式的导数矩阵行
        lr = self.expr._eval_derivative_matrix_lines(x)
        # 对函数进行逐元素应用
        ewdiff = ElementwiseApplyFunction(fdiff, self.expr)
        
        # 如果向量是一维的情况
        if 1 in x.shape:
            # 判断是否是列向量
            iscolumn = self.shape[1] == 1
            # 对每一个导数行对象进行处理
            for i in lr:
                if iscolumn:
                    # 第一个指针指向当前导数行的第一个指针
                    ptr1 = i.first_pointer
                    # 第二个指针为列数为1的单位矩阵
                    ptr2 = Identity(self.shape[1])
                else:
                    # 第一个指针为行数为1的单位矩阵
                    ptr1 = Identity(self.shape[0])
                    # 第二个指针指向当前导数行的第二个指针
                    ptr2 = i.second_pointer

                # 构建表达式，对角阵对应的张量积
                subexpr = ExprBuilder(
                    ArrayDiagonal,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                ewdiff,
                                ptr1,
                                ptr2,
                            ]
                        ),
                        (0, 2) if iscolumn else (1, 4)
                    ],
                    validator=ArrayDiagonal._validate
                )
                # 将构建好的表达式赋给当前导数行对象的行列表
                i._lines = [subexpr]
                # 设置第一个指针的父级对象和索引
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                # 设置第二个指针的父级对象和索引
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 2
        else:
            # 矩阵的情况
            for i in lr:
                # 第一个指针指向当前导数行的第一个指针
                ptr1 = i.first_pointer
                # 第二个指针指向当前导数行的第二个指针
                ptr2 = i.second_pointer
                # 新的第一个指针为第一个指针列数的单位矩阵
                newptr1 = Identity(ptr1.shape[1])
                # 新的第二个指针为第二个指针列数的单位矩阵
                newptr2 = Identity(ptr2.shape[1])
                # 构建表达式，收缩的张量积
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [ptr1, newptr1, ewdiff, ptr2, newptr2]
                        ),
                        (1, 2, 4),
                        (5, 7, 8),
                    ],
                    validator=ArrayContraction._validate
                )
                # 设置第一个指针的父级对象和索引
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                # 设置第二个指针的父级对象和索引
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                # 将构建好的表达式赋给当前导数行对象的行列表
                i._lines = [subexpr]
        
        # 返回计算好的导数矩阵行对象列表
        return lr

    # 定义一个方法来计算当前对象的转置
    def _eval_transpose(self):
        # 导入转置相关的模块和类
        from sympy.matrices.expressions.transpose import Transpose
        # 返回当前对象应用转置操作后的结果
        return self.func(self.function, Transpose(self.expr).doit())
```