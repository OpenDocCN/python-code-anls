# `D:\src\scipysrc\sympy\sympy\tensor\functions.py`

```
# 导入必要的模块和函数
from collections.abc import Iterable  # 从 collections.abc 模块导入 Iterable 类
from functools import singledispatch  # 从 functools 模块导入 singledispatch 函数

# 导入 SymPy 相关模块和类
from sympy.core.expr import Expr  # 从 sympy.core.expr 模块导入 Expr 类
from sympy.core.mul import Mul  # 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.singleton import S  # 从 sympy.core.singleton 模块导入 S 对象
from sympy.core.sympify import sympify  # 从 sympy.core.sympify 模块导入 sympify 函数
from sympy.core.parameters import global_parameters  # 从 sympy.core.parameters 模块导入 global_parameters 对象

# 定义 TensorProduct 类，继承自 Expr 类
class TensorProduct(Expr):
    """
    Generic class for tensor products.
    """
    is_number = False  # 类属性，表示该类不是数值类型

    # 构造函数
    def __new__(cls, *args, **kwargs):
        # 导入必要的类和函数
        from sympy.tensor.array import NDimArray, tensorproduct, Array  # 导入数组相关类和函数
        from sympy.matrices.expressions.matexpr import MatrixExpr  # 导入矩阵表达式相关类
        from sympy.matrices.matrixbase import MatrixBase  # 导入矩阵基类
        from sympy.strategies import flatten  # 导入 flatten 函数

        # 将 args 中的每个元素转换为 SymPy 表达式
        args = [sympify(arg) for arg in args]
        # 获取 evaluate 参数，如果 kwargs 中没有则使用全局参数中的设置
        evaluate = kwargs.get("evaluate", global_parameters.evaluate)

        # 如果 evaluate 为 False，则只创建一个 Expr 对象并返回
        if not evaluate:
            obj = Expr.__new__(cls, *args)
            return obj

        arrays = []  # 存储数组对象的列表
        other = []   # 存储其他对象的列表
        scalar = S.One  # 标量初始化为 1

        # 遍历 args 中的每个元素
        for arg in args:
            # 如果 arg 是 Iterable、MatrixBase 或 NDimArray 类型之一，转换为 Array 对象并添加到 arrays 中
            if isinstance(arg, (Iterable, MatrixBase, NDimArray)):
                arrays.append(Array(arg))
            # 如果 arg 是 MatrixExpr 类型，直接添加到 other 列表中
            elif isinstance(arg, (MatrixExpr,)):
                other.append(arg)
            else:
                scalar *= arg  # 否则，将 arg 视为标量，与之前的标量相乘得到新的标量值

        coeff = scalar * tensorproduct(*arrays)  # 计算标量乘以数组的张量积
        # 如果 other 列表为空，则返回 coeff
        if len(other) == 0:
            return coeff
        # 如果 coeff 不等于 1，则将 coeff 与 other 列表合并为新的参数列表 newargs
        if coeff != 1:
            newargs = [coeff] + other
        else:
            newargs = other
        obj = Expr.__new__(cls, *newargs, **kwargs)  # 创建新的 Expr 对象
        return flatten(obj)  # 展平生成的对象并返回

    def rank(self):
        # 返回张量积的维度数，即所有参数的维度数之和
        return len(self.shape)

    def _get_args_shapes(self):
        # 获取所有参数对象的形状列表，如果参数对象有 shape 属性则直接获取，否则转换为 Array 对象再获取其 shape
        from sympy.tensor.array import Array
        return [i.shape if hasattr(i, "shape") else Array(i).shape for i in self.args]

    @property
    def shape(self):
        # 返回张量积对象的形状，为所有参数对象形状的拼接
        shape_list = self._get_args_shapes()
        return sum(shape_list, ())

    def __getitem__(self, index):
        # 获取张量积对象的索引元素
        index = iter(index)
        return Mul.fromiter(
            # 逐个获取每个参数对象的索引元素并返回乘积
            arg.__getitem__(tuple(next(index) for i in shp))
            for arg, shp in zip(self.args, self._get_args_shapes())
        )


@singledispatch
def shape(expr):
    """
    Return the shape of the *expr* as a tuple. *expr* should represent
    suitable object such as matrix or array.

    Parameters
    ==========

    expr : SymPy object having ``MatrixKind`` or ``ArrayKind``.

    Raises
    ======

    NoShapeError : Raised when object with wrong kind is passed.

    Examples
    ========

    This function returns the shape of any object representing matrix or array.

    >>> from sympy import shape, Array, ImmutableDenseMatrix, Integral
    >>> from sympy.abc import x
    >>> A = Array([1, 2])
    >>> shape(A)
    (2,)
    >>> shape(Integral(A, x))
    (2,)
    >>> M = ImmutableDenseMatrix([1, 2])
    >>> shape(M)
    (2, 1)
    >>> shape(Integral(M, x))
    (2, 1)

    You can support new type by dispatching.

    >>> from sympy import Expr
    >>> class NewExpr(Expr):
    ...     pass
    >>> @shape.register(NewExpr)
    ... def _(expr):
    """
    # 返回 *expr* 对象的形状作为元组，*expr* 应该代表矩阵或数组等适当的对象
    pass  # 该函数仅包含文档字符串，具体实现在实际调用时根据对象类型分发执行
    # 如果表达式适合获取形状信息，则返回其形状
    return shape(expr.args[0])

>>> shape(NewExpr(M))
(2, 1)

如果传入的表达式不适合，将引发 ``NoShapeError()`` 异常。

>>> shape(Integral(x, x))
Traceback (most recent call last):
  ...
sympy.tensor.functions.NoShapeError: shape() called on non-array object: Integral(x, x)

Notes
=====

数组类（如 ``Matrix`` 或 ``NDimArray``）具有 ``shape`` 属性，用于返回其形状，
但不能用于包含数组的非数组类。此函数返回任何已注册为数组表示的对象的形状信息。

"""
# 如果表达式具有 "shape" 属性，则返回其形状信息
if hasattr(expr, "shape"):
    return expr.shape
# 如果表达式没有 "shape" 属性，或者其类型未注册到 "shape()" 函数中，则引发 NoShapeError 异常
raise NoShapeError(
    "%s does not have shape, or its type is not registered to shape()." % expr)
# 定义一个自定义异常类 NoShapeError，继承自 Exception 类
class NoShapeError(Exception):
    """
    Raised when ``shape()`` is called on non-array object.

    This error can be imported from ``sympy.tensor.functions``.

    Examples
    ========

    >>> from sympy import shape
    >>> from sympy.abc import x
    >>> shape(x)
    Traceback (most recent call last):
      ...
    sympy.tensor.functions.NoShapeError: shape() called on non-array object: x
    """
    # pass 语句，用于占位，表示类体为空，没有额外定义
    pass
```