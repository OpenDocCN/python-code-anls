# `D:\src\scipysrc\sympy\sympy\vector\vector.py`

```
# 导入未来的注解支持
from __future__ import annotations
# 导入 itertools 模块的 product 函数，用于生成迭代器的笛卡尔积
from itertools import product

# 导入 SymPy 核心模块中的各类和函数
from sympy.core import Add, Basic
# 导入 SymPy 核心模块中的假设知识库
from sympy.core.assumptions import StdFactKB
# 导入 SymPy 核心模块中的表达式类和原子表达式类
from sympy.core.expr import AtomicExpr, Expr
# 导入 SymPy 核心模块中的幂函数类
from sympy.core.power import Pow
# 导入 SymPy 核心模块中的单例类
from sympy.core.singleton import S
# 导入 SymPy 核心模块中的默认排序键函数
from sympy.core.sorting import default_sort_key
# 导入 SymPy 核心模块中的 sympify 函数，用于将字符串转换为 SymPy 表达式
from sympy.core.sympify import sympify
# 导入 SymPy 元素函数中的平方根函数
from sympy.functions.elementary.miscellaneous import sqrt
# 导入 SymPy 矩阵模块中的不可变密集矩阵类，并重命名为 Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
# 导入 SymPy 向量模块中的基于基向量的类
from sympy.vector.basisdependent import (BasisDependentZero,
    BasisDependent, BasisDependentMul, BasisDependentAdd)
# 导入 SymPy 向量模块中的直角坐标系类
from sympy.vector.coordsysrect import CoordSys3D
# 导入 SymPy 向量模块中的二阶张量类及其基类
from sympy.vector.dyadic import Dyadic, BaseDyadic, DyadicAdd
# 导入 SymPy 向量模块中的向量类型类
from sympy.vector.kind import VectorKind

# 定义一个名为 Vector 的类，继承自 BasisDependent 类
class Vector(BasisDependent):
    """
    Super class for all Vector classes.
    Ideally, neither this class nor any of its subclasses should be
    instantiated by the user.
    """

    # 设置类属性
    is_scalar = False  # 表示这不是一个标量
    is_Vector = True   # 表示这是一个向量
    _op_priority = 12.0  # 运算优先级设为 12.0

    # 类型注解
    _expr_type: type[Vector]
    _mul_func: type[Vector]
    _add_func: type[Vector]
    _zero_func: type[Vector]
    _base_func: type[Vector]
    zero: VectorZero

    # 创建一个 VectorKind 的实例作为 kind 属性，默认为空
    kind: VectorKind = VectorKind()

    # 定义 components 属性，返回向量的各个基向量分量的字典形式
    @property
    def components(self):
        """
        Returns the components of this vector in the form of a
        Python dictionary mapping BaseVector instances to the
        corresponding measure numbers.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.components
        {C.i: 3, C.j: 4, C.k: 5}

        """
        # '_components' 属性根据实例所属的 Vector 的子类进行定义
        return self._components

    # 定义 magnitude 方法，返回向量的大小（模）
    def magnitude(self):
        """
        Returns the magnitude of this vector.
        """
        return sqrt(self & self)

    # 定义 normalize 方法，返回向量的单位向量（归一化向量）
    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        return self / self.magnitude()
    def dot(self, other):
        """
        Returns the dot product of this Vector, either with another
        Vector, or a Dyadic, or a Del operator.
        If 'other' is a Vector, returns the dot product scalar (SymPy
        expression).
        If 'other' is a Dyadic, the dot product is returned as a Vector.
        If 'other' is an instance of Del, returns the directional
        derivative operator as a Python function. If this function is
        applied to a scalar expression, it returns the directional
        derivative of the scalar field wrt this Vector.

        Parameters
        ==========

        other: Vector/Dyadic/Del
            The Vector or Dyadic we are dotting with, or a Del operator .

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> C.i.dot(C.j)
        0
        >>> C.i & C.i
        1
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.dot(C.k)
        5
        >>> (C.i & delop)(C.x*C.y*C.z)
        C.y*C.z
        >>> d = C.i.outer(C.i)
        >>> C.i.dot(d)
        C.i

        """

        # Check special cases
        if isinstance(other, Dyadic):
            # Handle dot product with a Dyadic
            if isinstance(self, VectorZero):
                return Vector.zero
            outvec = Vector.zero
            for k, v in other.components.items():
                # Compute dot product of each component of the Dyadic with self
                vect_dot = k.args[0].dot(self)
                outvec += vect_dot * v * k.args[1]
            return outvec
        from sympy.vector.deloperator import Del
        if not isinstance(other, (Del, Vector)):
            # Raise TypeError if 'other' is neither Del nor Vector
            raise TypeError(str(other) + " is not a vector, dyadic or " +
                            "del operator")

        # Check if the other is a del operator
        if isinstance(other, Del):
            # Return a directional derivative function if 'other' is Del
            def directional_derivative(field):
                from sympy.vector.functions import directional_derivative
                return directional_derivative(field, self)
            return directional_derivative

        # If 'other' is Vector, delegate to dot function
        return dot(self, other)

    def __and__(self, other):
        """
        Override the '&' operator to perform dot product using '__and__'.

        Parameters
        ==========

        other: Vector/Dyadic/Del
            The Vector or Dyadic we are dotting with, or a Del operator.

        Returns
        =======

        dot product result based on the type of 'other'.

        """
        return self.dot(other)

    __and__.__doc__ = dot.__doc__
    def cross(self, other):
        """
        返回该向量与另一个向量或二阶张量的叉乘结果。
        如果 'other' 是向量，则结果是一个向量；如果 'other' 是二阶张量，则返回一个二阶张量。

        Parameters
        ==========

        other: Vector/Dyadic
            要进行叉乘的向量或二阶张量。

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> C.i.cross(C.j)
        C.k
        >>> C.i ^ C.i
        0
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v ^ C.i
        5*C.j + (-4)*C.k
        >>> d = C.i.outer(C.i)
        >>> C.j.cross(d)
        (-1)*(C.k|C.i)

        """

        # 检查特殊情况
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Dyadic.zero
            outdyad = Dyadic.zero
            for k, v in other.components.items():
                cross_product = self.cross(k.args[0])
                outer = cross_product.outer(k.args[1])
                outdyad += v * outer
            return outdyad

        # 调用cross函数来进行叉乘计算
        return cross(self, other)

    def __xor__(self, other):
        """
        将'^'操作符重载为叉乘操作。
        """
        return self.cross(other)

    __xor__.__doc__ = cross.__doc__

    def outer(self, other):
        """
        返回该向量与另一个向量的外积，以二阶张量的形式返回。

        Parameters
        ==========

        other : Vector
            要进行外积计算的向量。

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N.i.outer(N.j)
        (N.i|N.j)

        """

        # 处理特殊情况
        if not isinstance(other, Vector):
            raise TypeError("Invalid operand for outer product")
        elif (isinstance(self, VectorZero) or
                isinstance(other, VectorZero)):
            return Dyadic.zero

        # 遍历两个向量的分量，生成所需的二阶张量实例
        args = [(v1 * v2) * BaseDyadic(k1, k2) for (k1, v1), (k2, v2)
                in product(self.components.items(), other.components.items())]

        return DyadicAdd(*args)
    def projection(self, other, scalar=False):
        """
        Returns the vector or scalar projection of the 'other' on 'self'.

        Examples
        ========

        >>> from sympy.vector.coordsysrect import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> i, j, k = C.base_vectors()
        >>> v1 = i + j + k
        >>> v2 = 3*i + 4*j
        >>> v1.projection(v2)
        7/3*C.i + 7/3*C.j + 7/3*C.k
        >>> v1.projection(v2, scalar=True)
        7/3

        """
        # 如果自身向量为零向量，则根据标量参数返回零或零向量
        if self.equals(Vector.zero):
            return S.Zero if scalar else Vector.zero

        # 如果计算标量投影，则返回内积除以自身向量的内积
        if scalar:
            return self.dot(other) / self.dot(self)
        else:
            # 否则返回向量投影，即内积除以自身向量的内积再乘以自身向量
            return self.dot(other) / self.dot(self) * self

    @property
    def _projections(self):
        """
        Returns the components of this vector but the output includes
        also zero values components.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Vector
        >>> C = CoordSys3D('C')
        >>> v1 = 3*C.i + 4*C.j + 5*C.k
        >>> v1._projections
        (3, 4, 5)
        >>> v2 = C.x*C.y*C.z*C.i
        >>> v2._projections
        (C.x*C.y*C.z, 0, 0)
        >>> v3 = Vector.zero
        >>> v3._projections
        (0, 0, 0)
        """

        # 导入必要的模块
        from sympy.vector.operators import _get_coord_systems
        # 如果向量是零向量，则返回全为零的元组
        if isinstance(self, VectorZero):
            return (S.Zero, S.Zero, S.Zero)
        # 获取坐标系的基向量并计算向量与每个基向量的内积，返回结果作为元组
        base_vec = next(iter(_get_coord_systems(self))).base_vectors()
        return tuple([self.dot(i) for i in base_vec])

    def __or__(self, other):
        """
        Overrides the bitwise OR operator to perform outer product.

        Returns
        =======
        Vector
            The outer product of self with other.

        """
        return self.outer(other)

    __or__.__doc__ = outer.__doc__

    def to_matrix(self, system):
        """
        Returns the matrix form of this vector with respect to the
        specified coordinate system.

        Parameters
        ==========

        system : CoordSys3D
            The system wrt which the matrix form is to be computed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> from sympy.abc import a, b, c
        >>> v = a*C.i + b*C.j + c*C.k
        >>> v.to_matrix(C)
        Matrix([
        [a],
        [b],
        [c]])

        """

        # 返回向量相对于指定坐标系的矩阵形式，其中每个元素是向量与坐标系基向量的内积
        return Matrix([self.dot(unit_vec) for unit_vec in
                       system.base_vectors()])
    def separate(self):
        """
        The constituents of this vector in different coordinate systems,
        as per its definition.

        Returns a dict mapping each CoordSys3D to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> R1 = CoordSys3D('R1')
        >>> R2 = CoordSys3D('R2')
        >>> v = R1.i + R2.i
        >>> v.separate() == {R1: R1.i, R2: R2.i}
        True

        """

        # 初始化一个空字典来存储不同坐标系下的向量部分
        parts = {}
        # 遍历当前向量对象的成分字典
        for vect, measure in self.components.items():
            # 将当前向量的部分加到对应坐标系的已有向量上
            parts[vect.system] = (parts.get(vect.system, Vector.zero) +
                                  vect * measure)
        # 返回各个坐标系对应的向量部分的字典
        return parts

    def _div_helper(one, other):
        """ Helper for division involving vectors. """
        # 如果其中一个参数是向量而另一个不是，则抛出类型错误
        if isinstance(one, Vector) and isinstance(other, Vector):
            raise TypeError("Cannot divide two vectors")
        # 如果被除数是向量且除数为零，则抛出值错误
        elif isinstance(one, Vector):
            if other == S.Zero:
                raise ValueError("Cannot divide a vector by zero")
            # 返回向量乘以除数的倒数的结果
            return VectorMul(one, Pow(other, S.NegativeOne))
        else:
            # 其他情况视为无效的向量相关的除法，抛出类型错误
            raise TypeError("Invalid division involving a vector")
# The following is adapted from the matrices.expressions.matexpr file

def get_postprocessor(cls):
    # 定义一个内部函数 _postprocessor，用于处理表达式
    def _postprocessor(expr):
        # 根据传入的类别 cls 确定向量类别为 VectorAdd
        vec_class = {Add: VectorAdd}[cls]
        # 初始化空列表用于存储向量对象
        vectors = []
        # 遍历表达式中的项
        for term in expr.args:
            # 检查每个项是否属于 VectorKind 类型
            if isinstance(term.kind, VectorKind):
                vectors.append(term)

        # 如果 vec_class 是 VectorAdd 类型，则返回向量加法的运算结果
        if vec_class == VectorAdd:
            return VectorAdd(*vectors).doit(deep=False)
    return _postprocessor


# 将 get_postprocessor(Add) 函数作为后处理器映射到 Basic._constructor_postprocessor_mapping[Vector] 的 "Add" 键下
Basic._constructor_postprocessor_mapping[Vector] = {
    "Add": [get_postprocessor(Add)],
}

class BaseVector(Vector, AtomicExpr):
    """
    Class to denote a base vector.
    """

    def __new__(cls, index, system, pretty_str=None, latex_str=None):
        # 如果 pretty_str 为 None，则设置默认的漂亮字符串
        if pretty_str is None:
            pretty_str = "x{}".format(index)
        # 如果 latex_str 为 None，则设置默认的 LaTeX 字符串
        if latex_str is None:
            latex_str = "x_{}".format(index)
        pretty_str = str(pretty_str)
        latex_str = str(latex_str)
        # 验证索引参数是否在 0 到 2 的范围内
        if index not in range(0, 3):
            raise ValueError("index must be 0, 1 or 2")
        # 验证 system 参数是否为 CoordSys3D 类型
        if not isinstance(system, CoordSys3D):
            raise TypeError("system should be a CoordSys3D")
        # 根据参数创建一个新的对象
        name = system._vector_names[index]
        obj = super().__new__(cls, S(index), system)
        # 分配重要的属性
        obj._base_instance = obj
        obj._components = {obj: S.One}
        obj._measure_number = S.One
        obj._name = system._name + '.' + name
        obj._pretty_form = '' + pretty_str
        obj._latex_form = latex_str
        obj._system = system
        # 设置用于打印目的的 _id
        obj._id = (index, system)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)

        # 此属性用于将向量重新表达为涉及其定义的一个系统之一
        # 适用于 VectorMul 和 VectorAdd 类
        obj._sys = system

        return obj

    @property
    def system(self):
        return self._system

    def _sympystr(self, printer):
        # 返回向量的名称字符串表示
        return self._name

    def _sympyrepr(self, printer):
        # 返回向量的符号表示形式
        index, system = self._id
        return printer._print(system) + '.' + system._vector_names[index]

    @property
    def free_symbols(self):
        # 返回向量自由符号集合
        return {self}


class VectorAdd(BasisDependentAdd, Vector):
    """
    Class to denote sum of Vector instances.
    """

    def __new__(cls, *args, **options):
        # 创建一个新的 VectorAdd 实例
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    def _sympystr(self, printer):
        # 返回向量加法运算的字符串表示
        ret_str = ''
        items = list(self.separate().items())
        items.sort(key=lambda x: x[0].__str__())
        for system, vect in items:
            base_vects = system.base_vectors()
            for x in base_vects:
                if x in vect.components:
                    temp_vect = self.components[x] * x
                    ret_str += printer._print(temp_vect) + " + "
        return ret_str[:-3]
class VectorMul(BasisDependentMul, Vector):
    """
    Class to denote products of scalars and BaseVectors.
    表示标量和基向量的乘积的类。
    """

    def __new__(cls, *args, **options):
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    @property
    def base_vector(self):
        """ The BaseVector involved in the product. """
        # 返回参与乘积的基向量
        return self._base_instance

    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this VectorMul.
        与定义该 VectorMul 的标量表达式相关联。
        """
        return self._measure_number


class VectorZero(BasisDependentZero, Vector):
    """
    Class to denote a zero vector
    表示零向量的类。
    """

    _op_priority = 12.1
    _pretty_form = '0'
    _latex_form = r'\mathbf{\hat{0}}'

    def __new__(cls):
        obj = BasisDependentZero.__new__(cls)
        return obj


class Cross(Vector):
    """
    Represents unevaluated Cross product.
    表示未求值的叉乘。

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> Cross(v1, v2)
    Cross(R.i + R.j + R.k, R.x*R.i + R.y*R.j + R.z*R.k)
    >>> Cross(v1, v2).doit()
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        if default_sort_key(expr1) > default_sort_key(expr2):
            return -Cross(expr2, expr1)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        return cross(self._expr1, self._expr2)


class Dot(Expr):
    """
    Represents unevaluated Dot product.
    表示未求值的点乘。

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Dot
    >>> from sympy import symbols
    >>> R = CoordSys3D('R')
    >>> a, b, c = symbols('a b c')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = a * R.i + b * R.j + c * R.k
    >>> Dot(v1, v2)
    Dot(R.i + R.j + R.k, a*R.i + b*R.j + c*R.k)
    >>> Dot(v1, v2).doit()
    a + b + c

    """

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        expr1, expr2 = sorted([expr1, expr2], key=default_sort_key)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        return dot(self._expr1, self._expr2)


def cross(vect1, vect2):
    """
    Returns cross product of two vectors.
    返回两个向量的叉乘。

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> cross(v1, v2)
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """
    if isinstance(vect1, Add):
        return VectorAdd.fromiter(cross(i, vect2) for i in vect1.args)
    # 如果 vect2 是 Add 类型的对象，则返回向量加法对象 VectorAdd
    if isinstance(vect2, Add):
        return VectorAdd.fromiter(cross(vect1, i) for i in vect2.args)
    
    # 如果 vect1 和 vect2 都是 BaseVector 类型的对象
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        # 如果它们所属的坐标系相同
        if vect1._sys == vect2._sys:
            # 分别获取两个向量的第一个坐标轴编号
            n1 = vect1.args[0]
            n2 = vect2.args[0]
            # 如果两个向量的第一个坐标轴编号相同，则返回零向量
            if n1 == n2:
                return Vector.zero
            # 找到第三个坐标轴编号（不在 n1 和 n2 中的）
            n3 = ({0, 1, 2}.difference({n1, n2})).pop()
            # 根据坐标轴的顺序确定符号
            sign = 1 if ((n1 + 1) % 3 == n2) else -1
            # 返回对应的基向量
            return sign * vect1._sys.base_vectors()[n3]
        
        # 导入 express 函数，并尝试用其表达式来计算向量
        from .functions import express
        try:
            v = express(vect1, vect2._sys)
        except ValueError:
            # 如果 express 函数抛出 ValueError，则返回叉乘对象 Cross
            return Cross(vect1, vect2)
        else:
            # 否则，返回 vect1 和 vect2 的叉乘结果
            return cross(v, vect2)
    
    # 如果 vect1 或 vect2 是 VectorZero 类型的对象，则返回零向量
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return Vector.zero
    
    # 如果 vect1 是 VectorMul 类型的对象
    if isinstance(vect1, VectorMul):
        # 获取 vect1 的第一个成分向量和乘数
        v1, m1 = next(iter(vect1.components.items()))
        # 返回 v1 乘以与 vect2 的叉乘结果
        return m1 * cross(v1, vect2)
    
    # 如果 vect2 是 VectorMul 类型的对象
    if isinstance(vect2, VectorMul):
        # 获取 vect2 的第一个成分向量和乘数
        v2, m2 = next(iter(vect2.components.items()))
        # 返回 m2 倍的 vect1 和 v2 的叉乘结果
        return m2 * cross(vect1, v2)
    
    # 如果以上情况都不符合，则返回 vect1 和 vect2 的叉乘对象 Cross
    return Cross(vect1, vect2)
# 定义一个函数，计算两个向量的点积。

def dot(vect1, vect2):
    """
    Returns dot product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import dot
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> dot(v1, v2)
    R.x + R.y + R.z

    """

    # 检查 vect1 是否是加法表达式，递归计算每个项和 vect2 的点积
    if isinstance(vect1, Add):
        return Add.fromiter(dot(i, vect2) for i in vect1.args)
    
    # 检查 vect2 是否是加法表达式，递归计算 vect1 和每个项的点积
    if isinstance(vect2, Add):
        return Add.fromiter(dot(vect1, i) for i in vect2.args)
    
    # 如果 vect1 和 vect2 都是基本向量，并且属于同一个坐标系，则返回相应的值（1 或 0）
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            return S.One if vect1 == vect2 else S.Zero
        from .functions import express
        try:
            # 尝试将 vect2 表达为 vect1 所在坐标系中的表达式
            v = express(vect2, vect1._sys)
        except ValueError:
            return Dot(vect1, vect2)
        else:
            return dot(vect1, v)
    
    # 如果 vect1 或 vect2 是零向量，则返回零
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return S.Zero
    
    # 如果 vect1 是向量乘积，则分别取出向量和系数进行计算
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1 * dot(v1, vect2)
    
    # 如果 vect2 是向量乘积，则分别取出向量和系数进行计算
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2 * dot(vect1, v2)

    # 如果以上情况都不符合，则返回标准点积的表达式
    return Dot(vect1, vect2)


# 设置向量类型的表达方式和乘法、加法、零向量、基本向量的函数
Vector._expr_type = Vector
Vector._mul_func = VectorMul
Vector._add_func = VectorAdd
Vector._zero_func = VectorZero
Vector._base_func = BaseVector

# 创建一个零向量对象
Vector.zero = VectorZero()
```