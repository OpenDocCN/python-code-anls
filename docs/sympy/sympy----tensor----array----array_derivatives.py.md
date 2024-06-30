# `D:\src\scipysrc\sympy\sympy\tensor\array\array_derivatives.py`

```
# 导入 __future__ 模块中的 annotations 功能，使得类型提示中的类型可以是字符串表示的类名
from __future__ import annotations

# 导入 SymPy 库中的表达式类
from sympy.core.expr import Expr
# 导入 SymPy 库中的求导类
from sympy.core.function import Derivative
# 导入 SymPy 库中的整数类
from sympy.core.numbers import Integer
# 导入 SymPy 库中的矩阵基类
from sympy.matrices.matrixbase import MatrixBase
# 导入当前包中的 NDimArray 类
from .ndim_array import NDimArray
# 导入当前包中的 derive_by_array 函数
from .arrayop import derive_by_array
# 导入 SymPy 库中的矩阵表达式类
from sympy.matrices.expressions.matexpr import MatrixExpr
# 导入 SymPy 库中的特殊矩阵类
from sympy.matrices.expressions.special import ZeroMatrix
# 导入 SymPy 库中的矩阵求导函数
from sympy.matrices.expressions.matexpr import _matrix_derivative

# 定义 ArrayDerivative 类，继承自 Derivative 类
class ArrayDerivative(Derivative):

    # 标记该类不是标量（scalar）
    is_scalar = False

    # 定义构造方法，重载 Derivative 类的构造方法
    def __new__(cls, expr, *variables, **kwargs):
        # 调用父类的构造方法创建对象
        obj = super().__new__(cls, expr, *variables, **kwargs)
        # 如果创建的对象属于 ArrayDerivative 类
        if isinstance(obj, ArrayDerivative):
            # 获取对象的形状并赋值给 _shape 属性
            obj._shape = obj._get_shape()
        # 返回创建的对象
        return obj

    # 获取对象的形状的方法
    def _get_shape(self):
        # 初始化形状为空元组
        shape = ()
        # 遍历每个变量及其出现次数
        for v, count in self.variable_count:
            # 如果变量 v 有 shape 属性
            if hasattr(v, "shape"):
                # 将变量 v 的形状 count 次添加到 shape 中
                for i in range(count):
                    shape += v.shape
        # 如果表达式 expr 有 shape 属性，则将其形状添加到 shape 中
        if hasattr(self.expr, "shape"):
            shape += self.expr.shape
        # 返回计算得到的形状
        return shape

    # 定义 shape 属性，返回对象的形状
    @property
    def shape(self):
        return self._shape

    # 类方法：根据给定表达式返回与其形状相同的零矩阵
    @classmethod
    def _get_zero_with_shape_like(cls, expr):
        # 如果 expr 是 MatrixBase 或 NDimArray 类型，则返回形状相同的零矩阵
        if isinstance(expr, (MatrixBase, NDimArray)):
            return expr.zeros(*expr.shape)
        # 如果 expr 是 MatrixExpr 类型，则返回形状相同的零矩阵
        elif isinstance(expr, MatrixExpr):
            return ZeroMatrix(*expr.shape)
        # 否则，抛出运行时错误，无法确定数组导数的形状
        else:
            raise RuntimeError("Unable to determine shape of array-derivative.")

    # 静态方法：计算标量关于矩阵的导数
    @staticmethod
    def _call_derive_scalar_by_matrix(expr: Expr, v: MatrixBase) -> Expr:
        return v.applyfunc(lambda x: expr.diff(x))

    # 静态方法：计算标量关于矩阵表达式的导数
    @staticmethod
    def _call_derive_scalar_by_matexpr(expr: Expr, v: MatrixExpr) -> Expr:
        # 如果表达式 expr 中包含 v，则调用 _matrix_derivative 计算导数
        if expr.has(v):
            return _matrix_derivative(expr, v)
        # 否则，返回与 v 相同形状的零矩阵
        else:
            return ZeroMatrix(*v.shape)

    # 静态方法：计算标量关于 NDimArray 类型的数组的导数
    @staticmethod
    def _call_derive_scalar_by_array(expr: Expr, v: NDimArray) -> Expr:
        return v.applyfunc(lambda x: expr.diff(x))

    # 静态方法：计算矩阵关于标量的导数
    @staticmethod
    def _call_derive_matrix_by_scalar(expr: MatrixBase, v: Expr) -> Expr:
        return _matrix_derivative(expr, v)

    # 静态方法：计算矩阵表达式关于标量的导数
    @staticmethod
    def _call_derive_matexpr_by_scalar(expr: MatrixExpr, v: Expr) -> Expr:
        return expr._eval_derivative(v)

    # 静态方法：计算 NDimArray 类型数组关于标量的导数
    @staticmethod
    def _call_derive_array_by_scalar(expr: NDimArray, v: Expr) -> Expr:
        return expr.applyfunc(lambda x: x.diff(v))

    # 静态方法：默认情况下计算表达式关于标量的导数
    @staticmethod
    def _call_derive_default(expr: Expr, v: Expr) -> Expr | None:
        # 如果表达式 expr 中包含 v，则调用 _matrix_derivative 计算导数
        if expr.has(v):
            return _matrix_derivative(expr, v)
        # 否则，返回 None
        else:
            return None

    # 类方法
    @classmethod
    # 定义一个类方法，用于对表达式求导数 `n` 次。如果当前对象没有重写 `_eval_derivative_n_times` 方法，
    # 则在 `Basic` 类中的默认实现将调用 `_eval_derivative` 方法的循环执行。

    if not isinstance(count, (int, Integer)) or ((count <= 0) == True):
        # 如果 `count` 不是整数或者不是正整数，则返回 None
        return None

    # TODO: 可以使用多重分派来实现这个功能:
    if expr.is_scalar:
        # 如果 `expr` 是标量
        if isinstance(v, MatrixBase):
            # 如果 `v` 是 `MatrixBase` 类型的对象
            result = cls._call_derive_scalar_by_matrix(expr, v)
        elif isinstance(v, MatrixExpr):
            # 如果 `v` 是 `MatrixExpr` 类型的对象
            result = cls._call_derive_scalar_by_matexpr(expr, v)
        elif isinstance(v, NDimArray):
            # 如果 `v` 是 `NDimArray` 类型的对象
            result = cls._call_derive_scalar_by_array(expr, v)
        elif v.is_scalar:
            # 如果 `v` 是标量，这种情况有特殊处理
            return super()._dispatch_eval_derivative_n_times(expr, v, count)
        else:
            # 其他情况返回 None
            return None
    elif v.is_scalar:
        # 如果 `v` 是标量
        if isinstance(expr, MatrixBase):
            # 如果 `expr` 是 `MatrixBase` 类型的对象
            result = cls._call_derive_matrix_by_scalar(expr, v)
        elif isinstance(expr, MatrixExpr):
            # 如果 `expr` 是 `MatrixExpr` 类型的对象
            result = cls._call_derive_matexpr_by_scalar(expr, v)
        elif isinstance(expr, NDimArray):
            # 如果 `expr` 是 `NDimArray` 类型的对象
            result = cls._call_derive_array_by_scalar(expr, v)
        else:
            # 其他情况返回 None
            return None
    else:
        # 如果 `expr` 和 `v` 都是某种数组/矩阵类型
        if isinstance(expr, MatrixBase) or isinstance(v, MatrixBase):
            # 如果 `expr` 或者 `v` 是 `MatrixBase` 类型的对象
            result = derive_by_array(expr, v)
        elif isinstance(expr, MatrixExpr) and isinstance(v, MatrixExpr):
            # 如果 `expr` 和 `v` 都是 `MatrixExpr` 类型的对象
            result = cls._call_derive_default(expr, v)
        elif isinstance(expr, MatrixExpr) or isinstance(v, MatrixExpr):
            # 如果一个是符号矩阵表达式而另一个不是，则不进行评估
            return None
        else:
            # 其他情况通过数组求导数
            result = derive_by_array(expr, v)
    
    # 如果结果是 None，则返回 None
    if result is None:
        return None
    
    # 如果 `count` 等于 1，则直接返回结果
    if count == 1:
        return result
    else:
        # 否则递归调用 `_dispatch_eval_derivative_n_times` 函数
        return cls._dispatch_eval_derivative_n_times(result, v, count - 1)
```