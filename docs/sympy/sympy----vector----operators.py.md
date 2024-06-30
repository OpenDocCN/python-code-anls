# `D:\src\scipysrc\sympy\sympy\vector\operators.py`

```
import collections  # 导入collections模块，用于创建默认字典
from sympy.core.expr import Expr  # 导入Expr类，表示SymPy表达式的基类
from sympy.core import sympify, S, preorder_traversal  # 导入sympify函数、S对象和preorder_traversal函数
from sympy.vector.coordsysrect import CoordSys3D  # 导入CoordSys3D类，表示三维坐标系
from sympy.vector.vector import Vector, VectorMul, VectorAdd, Cross, Dot  # 导入向量相关的类和操作
from sympy.core.function import Derivative  # 导入Derivative类，表示SymPy中的导数
from sympy.core.add import Add  # 导入Add类，表示SymPy中的加法表达式
from sympy.core.mul import Mul  # 导入Mul类，表示SymPy中的乘法表达式


def _get_coord_systems(expr):
    # 使用前序遍历获取表达式中所有的CoordSys3D对象集合
    g = preorder_traversal(expr)
    ret = set()
    for i in g:
        if isinstance(i, CoordSys3D):
            ret.add(i)
            g.skip()  # 跳过当前CoordSys3D对象的子节点遍历
    return frozenset(ret)


def _split_mul_args_wrt_coordsys(expr):
    # 创建一个默认字典，以CoordSys3D对象集合作为键，默认值为S.One
    d = collections.defaultdict(lambda: S.One)
    for i in expr.args:
        # 对于表达式中的每个参数，将其对应的CoordSys3D对象集合作为键，乘到对应的值上
        d[_get_coord_systems(i)] *= i
    # 将字典中的值转换为列表并返回
    return list(d.values())


class Gradient(Expr):
    """
    Represents unevaluated Gradient.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Gradient
    >>> R = CoordSys3D('R')
    >>> s = R.x*R.y*R.z
    >>> Gradient(s)
    Gradient(R.x*R.y*R.z)

    """

    def __new__(cls, expr):
        expr = sympify(expr)  # 将输入的表达式转换为SymPy表达式
        obj = Expr.__new__(cls, expr)  # 调用父类Expr的构造方法创建对象
        obj._expr = expr  # 将表达式存储在对象属性_expr中
        return obj

    def doit(self, **hints):
        return gradient(self._expr, doit=True)  # 调用gradient函数计算梯度


class Divergence(Expr):
    """
    Represents unevaluated Divergence.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Divergence
    >>> R = CoordSys3D('R')
    >>> v = R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k
    >>> Divergence(v)
    Divergence(R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k)

    """

    def __new__(cls, expr):
        expr = sympify(expr)  # 将输入的表达式转换为SymPy表达式
        obj = Expr.__new__(cls, expr)  # 调用父类Expr的构造方法创建对象
        obj._expr = expr  # 将表达式存储在对象属性_expr中
        return obj

    def doit(self, **hints):
        return divergence(self._expr, doit=True)  # 调用divergence函数计算散度


class Curl(Expr):
    """
    Represents unevaluated Curl.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Curl
    >>> R = CoordSys3D('R')
    >>> v = R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k
    >>> Curl(v)
    Curl(R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k)

    """

    def __new__(cls, expr):
        expr = sympify(expr)  # 将输入的表达式转换为SymPy表达式
        obj = Expr.__new__(cls, expr)  # 调用父类Expr的构造方法创建对象
        obj._expr = expr  # 将表达式存储在对象属性_expr中
        return obj

    def doit(self, **hints):
        return curl(self._expr, doit=True)  # 调用curl函数计算旋度


def curl(vect, doit=True):
    """
    Returns the curl of a vector field computed wrt the base scalars
    of the given coordinate system.

    Parameters
    ==========

    vect : Vector
        The vector operand

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, curl
    >>> R = CoordSys3D('R')
    >>> v1 = R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k
    >>> curl(v1)
    0
    >>> v2 = R.x*R.y*R.z*R.i
    >>> curl(v2)
    R.x*R.y*R.j + (-R.x*R.z)*R.k

    """

    coord_sys = _get_coord_systems(vect)  # 获取向量vect涉及的坐标系集合
    # 如果坐标系列表为空，则返回零向量
    if len(coord_sys) == 0:
        return Vector.zero
    # 如果坐标系列表只包含一个坐标系，进行以下操作
    elif len(coord_sys) == 1:
        # 获取唯一的坐标系对象
        coord_sys = next(iter(coord_sys))
        # 获取坐标系的基向量 i, j, k
        i, j, k = coord_sys.base_vectors()
        # 获取坐标系的基标量 x, y, z
        x, y, z = coord_sys.base_scalars()
        # 获取坐标系的拉姆系数 h1, h2, h3
        h1, h2, h3 = coord_sys.lame_coefficients()
        # 计算向量 vect 在基向量 i, j, k 上的分量
        vectx = vect.dot(i)
        vecty = vect.dot(j)
        vectz = vect.dot(k)
        # 初始化输出向量为零向量
        outvec = Vector.zero
        # 计算输出向量的每个分量，根据给定公式进行计算
        outvec += (Derivative(vectz * h3, y) - Derivative(vecty * h2, z)) * i / (h2 * h3)
        outvec += (Derivative(vectx * h1, z) - Derivative(vectz * h3, x)) * j / (h1 * h3)
        outvec += (Derivative(vecty * h2, x) - Derivative(vectx * h1, y)) * k / (h2 * h1)

        # 如果需要进行求值，则调用 do-it 方法对输出向量进行求值
        if doit:
            return outvec.doit()
        # 否则直接返回输出向量
        return outvec
    # 如果坐标系列表包含多个坐标系，则根据不同情况进行处理
    else:
        # 如果向量是加法或向量加法的实例，从 sympy.vector 中引入 express 方法，并尝试转换向量表达式
        if isinstance(vect, (Add, VectorAdd)):
            from sympy.vector import express
            try:
                # 获取坐标系列表中的第一个坐标系
                cs = next(iter(coord_sys))
                # 对向量中的每个参数应用 express 方法
                args = [express(i, cs, variables=True) for i in vect.args]
            except ValueError:
                # 如果出现 ValueError，则直接使用原始参数
                args = vect.args
            # 返回由 curl 函数对每个参数计算得到的向量加法结果
            return VectorAdd.fromiter(curl(i, doit=doit) for i in args)
        # 如果向量是乘法或向量乘法的实例
        elif isinstance(vect, (Mul, VectorMul)):
            # 从 vect.args 中获取向量和标量部分
            vector = [i for i in vect.args if isinstance(i, (Vector, Cross, Gradient))][0]
            scalar = Mul.fromiter(i for i in vect.args if not isinstance(i, (Vector, Cross, Gradient)))
            # 计算交叉积的结果，然后与标量乘以 curl 函数的结果相加
            res = Cross(gradient(scalar), vector).doit() + scalar * curl(vector, doit=doit)
            # 如果需要求值，则对结果应用 do-it 方法
            if doit:
                return res.doit()
            # 否则直接返回结果
            return res
        # 如果向量是交叉积、旋度或梯度的实例，则直接返回其 curl 结果
        elif isinstance(vect, (Cross, Curl, Gradient)):
            return Curl(vect)
        else:
            # 如果向量类型不在预期范围内，则引发异常
            raise Curl(vect)
# 定义函数 divergence，计算向量场的散度，相对于给定坐标系的基标量
def divergence(vect, doit=True):
    # 获取包含向量操作的坐标系
    coord_sys = _get_coord_systems(vect)
    # 如果没有坐标系，则返回零
    if len(coord_sys) == 0:
        return S.Zero
    # 如果只有一个坐标系
    elif len(coord_sys) == 1:
        # 如果 vect 是 Cross、Curl 或 Gradient 类的实例，返回其散度
        if isinstance(vect, (Cross, Curl, Gradient)):
            return Divergence(vect)
        # 从多个坐标系中选择一个（这里选择第一个）
        coord_sys = next(iter(coord_sys))
        # 获取基向量和基标量
        i, j, k = coord_sys.base_vectors()
        x, y, z = coord_sys.base_scalars()
        h1, h2, h3 = coord_sys.lame_coefficients()
        # 计算向量的 x、y、z 分量的偏导数条件
        vx = _diff_conditional(vect.dot(i), x, h2, h3) \
             / (h1 * h2 * h3)
        vy = _diff_conditional(vect.dot(j), y, h3, h1) \
             / (h1 * h2 * h3)
        vz = _diff_conditional(vect.dot(k), z, h1, h2) \
             / (h1 * h2 * h3)
        # 计算散度结果
        res = vx + vy + vz
        # 如果需执行计算，则调用 .doit() 方法
        if doit:
            return res.doit()
        return res
    # 如果有多个坐标系
    else:
        # 如果 vect 是 Add 或 VectorAdd 类的实例，返回其各个向量散度之和
        if isinstance(vect, (Add, VectorAdd)):
            return Add.fromiter(divergence(i, doit=doit) for i in vect.args)
        # 如果 vect 是 Mul 或 VectorMul 类的实例
        elif isinstance(vect, (Mul, VectorMul)):
            # 找到其中的向量和标量
            vector = [i for i in vect.args if isinstance(i, (Vector, Cross, Gradient))][0]
            scalar = Mul.fromiter(i for i in vect.args if not isinstance(i, (Vector, Cross, Gradient)))
            # 计算 Dot(vector, gradient(scalar)) + scalar*divergence(vector)
            res = Dot(vector, gradient(scalar)) + scalar*divergence(vector, doit=doit)
            # 如果需执行计算，则调用 .doit() 方法
            if doit:
                return res.doit()
            return res
        # 如果 vect 是 Cross、Curl 或 Gradient 类的实例，返回其散度
        elif isinstance(vect, (Cross, Curl, Gradient)):
            return Divergence(vect)
        else:
            # 若不属于以上任何类型，则引发异常
            raise Divergence(vect)


# 定义函数 gradient，计算标量场的梯度，相对于给定坐标系的基标量
def gradient(scalar_field, doit=True):
    # 获取包含标量场的坐标系
    coord_sys = _get_coord_systems(scalar_field)
    # 如果坐标系列表为空，则返回零向量
    if len(coord_sys) == 0:
        return Vector.zero
    # 如果坐标系列表只有一个元素
    elif len(coord_sys) == 1:
        # 将坐标系列表的唯一元素赋值给变量 coord_sys
        coord_sys = next(iter(coord_sys))
        # 获取坐标系的拉姆系数
        h1, h2, h3 = coord_sys.lame_coefficients()
        # 获取坐标系的基向量
        i, j, k = coord_sys.base_vectors()
        # 获取坐标系的基标量
        x, y, z = coord_sys.base_scalars()
        # 计算标量场在 x 方向的导数，并除以 h1
        vx = Derivative(scalar_field, x) / h1
        # 计算标量场在 y 方向的导数，并除以 h2
        vy = Derivative(scalar_field, y) / h2
        # 计算标量场在 z 方向的导数，并除以 h3
        vz = Derivative(scalar_field, z) / h3

        # 如果需要执行 doit 操作
        if doit:
            # 返回向量场的求和结果，并执行 doit 操作
            return (vx * i + vy * j + vz * k).doit()
        # 否则，返回未执行 doit 操作的向量场求和结果
        return vx * i + vy * j + vz * k
    # 如果坐标系列表有多个元素
    else:
        # 如果标量场是加法或者向量加法
        if isinstance(scalar_field, (Add, VectorAdd)):
            # 返回从迭代中生成的梯度的向量加法
            return VectorAdd.fromiter(gradient(i) for i in scalar_field.args)
        # 如果标量场是乘法或者向量乘法
        if isinstance(scalar_field, (Mul, VectorMul)):
            # 将标量场按照坐标系拆分
            s = _split_mul_args_wrt_coordsys(scalar_field)
            # 返回从迭代中生成的乘以标量场的梯度的向量加法
            return VectorAdd.fromiter(scalar_field / i * gradient(i) for i in s)
        # 返回标量场的梯度
        return Gradient(scalar_field)
# Laplacian 类，表示未求值的拉普拉斯算子
class Laplacian(Expr):
    """
    Represents unevaluated Laplacian.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Laplacian
    >>> R = CoordSys3D('R')
    >>> v = 3*R.x**3*R.y**2*R.z**3
    >>> Laplacian(v)
    Laplacian(3*R.x**3*R.y**2*R.z**3)

    """

    # __new__ 方法，用于创建新的 Laplacian 对象
    def __new__(cls, expr):
        # 将输入的表达式转换为 SymPy 表达式
        expr = sympify(expr)
        # 调用父类 Expr 的 __new__ 方法创建对象
        obj = Expr.__new__(cls, expr)
        # 存储表达式
        obj._expr = expr
        return obj

    # doit 方法，执行拉普拉斯操作
    def doit(self, **hints):
        # 导入 laplacian 函数
        from sympy.vector.functions import laplacian
        # 调用 laplacian 函数对存储的表达式进行拉普拉斯操作
        return laplacian(self._expr)


# _diff_conditional 函数，条件性地对表达式进行偏导数计算
def _diff_conditional(expr, base_scalar, coeff_1, coeff_2):
    """
    First re-expresses expr in the system that base_scalar belongs to.
    If base_scalar appears in the re-expressed form, differentiates
    it wrt base_scalar.
    Else, returns 0
    """
    # 导入 express 函数
    from sympy.vector.functions import express
    # 在 base_scalar 所属的坐标系中重新表达表达式 expr，并允许变量
    new_expr = express(expr, base_scalar.system, variables=True)
    # 构建参数 arg，是 coeff_1 * coeff_2 * new_expr
    arg = coeff_1 * coeff_2 * new_expr
    # 如果 arg 存在，返回关于 base_scalar 的导数 Derivative(arg, base_scalar)，否则返回零 S.Zero
    return Derivative(arg, base_scalar) if arg else S.Zero
```