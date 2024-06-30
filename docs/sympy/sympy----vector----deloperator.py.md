# `D:\src\scipysrc\sympy\sympy\vector\deloperator.py`

```
from sympy.core import Basic  # 导入 SymPy 核心库中的 Basic 类
from sympy.vector.operators import gradient, divergence, curl  # 导入梯度、散度、旋度操作

class Del(Basic):
    """
    Represents the vector differential operator, usually represented in
    mathematical expressions as the 'nabla' symbol.
    表示向量微分操作符，通常在数学表达式中表示为 'nabla' 符号。
    """

    def __new__(cls):
        obj = super().__new__(cls)  # 创建 Del 类的新实例
        obj._name = "delop"  # 设置对象的名称属性为 "delop"
        return obj

    def gradient(self, scalar_field, doit=False):
        """
        Returns the gradient of the given scalar field, as a
        Vector instance.

        Parameters
        ==========

        scalar_field : SymPy expression
            The scalar field to calculate the gradient of.
            要计算梯度的标量场。

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances
            如果为 True，则在每个组件上调用 .doit() 后返回结果。否则，返回的表达式包含 Derivative 实例。

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> delop.gradient(9)
        0
        >>> delop(C.x*C.y*C.z).doit()
        C.y*C.z*C.i + C.x*C.z*C.j + C.x*C.y*C.k

        """
        return gradient(scalar_field, doit=doit)  # 调用梯度操作函数，计算标量场的梯度

    __call__ = gradient
    __call__.__doc__ = gradient.__doc__  # 设置 __call__ 方法的文档字符串为 gradient 方法的文档字符串

    def dot(self, vect, doit=False):
        """
        Represents the dot product between this operator and a given
        vector - equal to the divergence of the vector field.

        Parameters
        ==========

        vect : Vector
            The vector whose divergence is to be calculated.
            要计算散度的向量。

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances
            如果为 True，则在每个组件上调用 .doit() 后返回结果。否则，返回的表达式包含 Derivative 实例。

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> delop = Del()
        >>> C = CoordSys3D('C')
        >>> delop.dot(C.x*C.i)
        Derivative(C.x, C.x)
        >>> v = C.x*C.y*C.z * (C.i + C.j + C.k)
        >>> (delop & v).doit()
        C.x*C.y + C.x*C.z + C.y*C.z

        """
        return divergence(vect, doit=doit)  # 调用散度操作函数，计算向量的散度

    __and__ = dot
    __and__.__doc__ = dot.__doc__  # 设置 __and__ 方法的文档字符串为 dot 方法的文档字符串
    def cross(self, vect, doit=False):
        """
        Represents the cross product between this operator and a given
        vector - equal to the curl of the vector field.

        Parameters
        ==========

        vect : Vector
            The vector whose curl is to be calculated.

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> v = C.x*C.y*C.z * (C.i + C.j + C.k)
        >>> delop.cross(v, doit=True)
        (-C.x*C.y + C.x*C.z)*C.i + (C.x*C.y - C.y*C.z)*C.j +
            (-C.x*C.z + C.y*C.z)*C.k
        >>> (delop ^ C.i).doit()
        0

        """
        # 返回调用 curl 函数计算的结果
        return curl(vect, doit=doit)

    # 定义运算符 ^ 的别名为 cross 函数
    __xor__ = cross
    # 将运算符 ^ 的文档字符串设置为 cross 函数的文档字符串
    __xor__.__doc__ = cross.__doc__

    def _sympystr(self, printer):
        # 返回该对象的名称字符串表示
        return self._name
```