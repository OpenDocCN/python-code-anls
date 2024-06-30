# `D:\src\scipysrc\sympy\sympy\vector\dyadic.py`

```
from __future__ import annotations

from sympy.vector.basisdependent import (BasisDependent, BasisDependentAdd,
                                         BasisDependentMul, BasisDependentZero)
# Import specific classes from sympy.core and sympy.matrices modules
from sympy.core import S, Pow
from sympy.core.expr import AtomicExpr
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
import sympy.vector

# Define a class representing a Dyadic object, inheriting from BasisDependent
class Dyadic(BasisDependent):
    """
    Super class for all Dyadic-classes.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dyadic_tensor
    .. [2] Kane, T., Levinson, D. Dynamics Theory and Applications. 1985
           McGraw-Hill

    """

    # Operator priority for operations involving Dyadic objects
    _op_priority = 13.0

    # Type hints for attributes related to Dyadic expressions
    _expr_type: type[Dyadic]
    _mul_func: type[Dyadic]
    _add_func: type[Dyadic]
    _zero_func: type[Dyadic]
    _base_func: type[Dyadic]

    # Define a zero element for Dyadics
    zero: DyadicZero

    @property
    def components(self):
        """
        Returns the components of this dyadic in the form of a
        Python dictionary mapping BaseDyadic instances to the
        corresponding measure numbers.

        """
        # The '_components' attribute is defined according to the
        # subclass of Dyadic the instance belongs to.
        return self._components

    def dot(self, other):
        """
        Returns the dot product(also called inner product) of this
        Dyadic, with another Dyadic or Vector.
        If 'other' is a Dyadic, this returns a Dyadic. Else, it returns
        a Vector (unless an error is encountered).

        Parameters
        ==========

        other : Dyadic/Vector
            The other Dyadic or Vector to take the inner product with

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> D1 = N.i.outer(N.j)
        >>> D2 = N.j.outer(N.j)
        >>> D1.dot(D2)
        (N.i|N.j)
        >>> D1.dot(N.j)
        N.i

        """

        # Alias for sympy.vector.Vector
        Vector = sympy.vector.Vector

        # Check the type of 'other' and perform appropriate operations
        if isinstance(other, BasisDependentZero):
            # If 'other' is zero, return the zero vector
            return Vector.zero
        elif isinstance(other, Vector):
            # If 'other' is a Vector, perform dot product operation
            outvec = Vector.zero
            for k, v in self.components.items():
                # Calculate dot product between vectors and accumulate
                vect_dot = k.args[1].dot(other)
                outvec += vect_dot * v * k.args[0]
            return outvec
        elif isinstance(other, Dyadic):
            # If 'other' is a Dyadic, perform dyadic dot product
            outdyad = Dyadic.zero
            for k1, v1 in self.components.items():
                for k2, v2 in other.components.items():
                    # Calculate dot products and outer products
                    vect_dot = k1.args[1].dot(k2.args[0])
                    outer_product = k1.args[0].outer(k2.args[1])
                    outdyad += vect_dot * v1 * v2 * outer_product
            return outdyad
        else:
            # Raise TypeError if inner product is not defined for given types
            raise TypeError("Inner product is not defined for " +
                            str(type(other)) + " and Dyadics.")

    # Define '__and__' operator to perform dot product, with documentation alias
    def __and__(self, other):
        return self.dot(other)

    # Link '__and__' documentation to 'dot' method documentation
    __and__.__doc__ = dot.__doc__
    def cross(self, other):
        """
        返回这个二阶张量与一个向量的叉乘结果，作为一个向量实例。

        Parameters
        ==========

        other : Vector
            要与此二阶张量进行叉乘的向量

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> d = N.i.outer(N.i)
        >>> d.cross(N.j)
        (N.i|N.k)

        """

        Vector = sympy.vector.Vector  # 导入向量类
        if other == Vector.zero:  # 如果传入的向量是零向量，返回零二阶张量
            return Dyadic.zero
        elif isinstance(other, Vector):  # 如果传入的是向量实例
            outdyad = Dyadic.zero  # 初始化输出的二阶张量为零
            for k, v in self.components.items():  # 遍历当前二阶张量的分量
                cross_product = k.args[1].cross(other)  # 计算当前分量的第二个元素与传入向量的叉乘
                outer = k.args[0].outer(cross_product)  # 计算叉乘结果与当前分量的第一个元素的外积
                outdyad += v * outer  # 累加到输出的二阶张量中
            return outdyad  # 返回计算结果
        else:
            raise TypeError(str(type(other)) + " not supported for " +
                            "cross with dyadics")  # 如果传入的不是向量实例，抛出类型错误异常

    def __xor__(self, other):
        return self.cross(other)

    __xor__.__doc__ = cross.__doc__  # 设置 __xor__ 方法的文档字符串与 cross 方法的一致

    def to_matrix(self, system, second_system=None):
        """
        返回二阶张量在一个或两个坐标系下的矩阵形式。

        Parameters
        ==========

        system : CoordSys3D
            矩阵的行和列对应的坐标系。如果提供了第二个坐标系，只对应于矩阵的行。
        second_system : CoordSys3D, optional, default=None
            矩阵的列对应的坐标系。

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> v = N.i + 2*N.j
        >>> d = v.outer(N.i)
        >>> d.to_matrix(N)
        Matrix([
        [1, 0, 0],
        [2, 0, 0],
        [0, 0, 0]])
        >>> from sympy import Symbol
        >>> q = Symbol('q')
        >>> P = N.orient_new_axis('P', q, N.k)
        >>> d.to_matrix(N, P)
        Matrix([
        [  cos(q),   -sin(q), 0],
        [2*cos(q), -2*sin(q), 0],
        [       0,         0, 0]])

        """

        if second_system is None:
            second_system = system

        return Matrix([i.dot(self).dot(j) for i in system for j in
                       second_system]).reshape(3, 3)

    def _div_helper(one, other):
        """ 用于涉及二阶张量的除法的辅助函数 """
        if isinstance(one, Dyadic) and isinstance(other, Dyadic):
            raise TypeError("不能对两个二阶张量进行除法")
        elif isinstance(one, Dyadic):
            return DyadicMul(one, Pow(other, S.NegativeOne))
        else:
            raise TypeError("不能除以一个二阶张量")
# 定义一个基本的双线性张量组件类，继承自 Dyadic 和 AtomicExpr 类
class BaseDyadic(Dyadic, AtomicExpr):
    """
    Class to denote a base dyadic tensor component.
    """

    # 实现 __new__ 方法用于创建新对象
    def __new__(cls, vector1, vector2):
        # 导入必要的符号计算库中的向量和基础向量类
        Vector = sympy.vector.Vector
        BaseVector = sympy.vector.BaseVector
        VectorZero = sympy.vector.VectorZero
        
        # 验证传入的参数是否是 BaseVector 或 VectorZero 类的实例
        if not isinstance(vector1, (BaseVector, VectorZero)) or \
                not isinstance(vector2, (BaseVector, VectorZero)):
            # 如果不是，则抛出类型错误异常
            raise TypeError("BaseDyadic cannot be composed of non-base " +
                            "vectors")
        
        # 处理特殊情况：如果任一向量为零向量，则返回零双线性张量
        elif vector1 == Vector.zero or vector2 == Vector.zero:
            return Dyadic.zero
        
        # 如果参数验证通过，创建新对象实例
        obj = super().__new__(cls, vector1, vector2)
        
        # 设置对象的属性
        obj._base_instance = obj
        obj._measure_number = 1
        obj._components = {obj: S.One}
        obj._sys = vector1._sys
        obj._pretty_form = ('(' + vector1._pretty_form + '|' +
                             vector2._pretty_form + ')')
        obj._latex_form = (r'\left(' + vector1._latex_form + r"{\middle|}" +
                           vector2._latex_form + r'\right)')
        
        return obj

    # 返回对象的字符串表示形式
    def _sympystr(self, printer):
        return "({}|{})".format(
            printer._print(self.args[0]), printer._print(self.args[1]))

    # 返回对象的符号表示形式
    def _sympyrepr(self, printer):
        return "BaseDyadic({}, {})".format(
            printer._print(self.args[0]), printer._print(self.args[1]))


# 定义用于标量和 BaseDyadics 乘积的类，继承自 BasisDependentMul 和 Dyadic 类
class DyadicMul(BasisDependentMul, Dyadic):
    """ Products of scalars and BaseDyadics """

    # 实现 __new__ 方法用于创建新对象
    def __new__(cls, *args, **options):
        # 调用 BasisDependentMul 的 __new__ 方法创建新对象
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    # 定义 base_dyadic 属性，返回与乘积相关的 BaseDyadic 对象
    @property
    def base_dyadic(self):
        """ The BaseDyadic involved in the product. """
        return self._base_instance

    # 定义 measure_number 属性，返回与定义 DyadicMul 相关的标量表达式
    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this DyadicMul.
        """
        return self._measure_number


# 定义用于保存双线性和的类，继承自 BasisDependentAdd 和 Dyadic 类
class DyadicAdd(BasisDependentAdd, Dyadic):
    """ Class to hold dyadic sums """

    # 实现 __new__ 方法用于创建新对象
    def __new__(cls, *args, **options):
        # 调用 BasisDependentAdd 的 __new__ 方法创建新对象
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    # 重写 _sympystr 方法，返回对象的字符串表示形式
    def _sympystr(self, printer):
        items = list(self.components.items())
        items.sort(key=lambda x: x[0].__str__())
        return " + ".join(printer._print(k * v) for k, v in items)


# 定义用于表示零双线性和的类，继承自 BasisDependentZero 和 Dyadic 类
class DyadicZero(BasisDependentZero, Dyadic):
    """
    Class to denote a zero dyadic
    """

    # 设置操作优先级、漂亮形式和 LaTeX 表示形式
    _op_priority = 13.1
    _pretty_form = '(0|0)'
    _latex_form = r'(\mathbf{\hat{0}}|\mathbf{\hat{0}})'

    # 实现 __new__ 方法用于创建新对象
    def __new__(cls):
        # 调用 BasisDependentZero 的 __new__ 方法创建新对象
        obj = BasisDependentZero.__new__(cls)
        return obj


# 设置 Dyadic 类的表达式类型、乘法函数、加法函数、零函数和基函数
Dyadic._expr_type = Dyadic
Dyadic._mul_func = DyadicMul
Dyadic._add_func = DyadicAdd
Dyadic._zero_func = DyadicZero
# 创建 DyadicZero 的实例并赋给 Dyadic 类的 zero 属性
Dyadic.zero = DyadicZero()
```