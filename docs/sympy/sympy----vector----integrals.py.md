# `D:\src\scipysrc\sympy\sympy\vector\integrals.py`

```
# 导入 sympy 库中需要的模块和函数
from sympy.core import Basic, diff
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.matrices import Matrix
from sympy.integrals import Integral, integrate
from sympy.geometry.entity import GeometryEntity
from sympy.simplify.simplify import simplify
from sympy.utilities.iterables import topological_sort
from sympy.vector import (CoordSys3D, Vector, ParametricRegion,
                        parametric_region_list, ImplicitRegion)
from sympy.vector.operators import _get_coord_systems

# 定义 ParametricIntegral 类，用于表示在参数化区域上的标量或矢量场积分
class ParametricIntegral(Basic):
    """
    Represents integral of a scalar or vector field
    over a Parametric Region

    Examples
    ========

    >>> from sympy import cos, sin, pi
    >>> from sympy.vector import CoordSys3D, ParametricRegion, ParametricIntegral
    >>> from sympy.abc import r, t, theta, phi

    >>> C = CoordSys3D('C')
    定义一个三维坐标系 C

    >>> curve = ParametricRegion((3*t - 2, t + 1), (t, 1, 2))
    定义一个参数化曲线 ParametricRegion 对象 curve

    >>> ParametricIntegral(C.x, curve)
    计算在 curve 上对 C.x 的参数化积分结果

    >>> length = ParametricIntegral(1, curve)
    计算在 curve 上对标量 1 的参数化积分结果，即曲线的长度

    >>> semisphere = ParametricRegion((2*sin(phi)*cos(theta), 2*sin(phi)*sin(theta), 2*cos(phi)),\
                            (theta, 0, 2*pi), (phi, 0, pi/2))
    定义一个半球面的参数化区域 ParametricRegion 对象 semisphere

    >>> ParametricIntegral(C.z, semisphere)
    计算在 semisphere 上对 C.z 的参数化积分结果

    >>> ParametricIntegral(C.j + C.k, ParametricRegion((r*cos(theta), r*sin(theta)), r, theta))
    计算在参数化曲线 ParametricRegion((r*cos(theta), r*sin(theta)), r, theta) 上对 C.j + C.k 的参数化积分结果
    """
    # 定义一个新的类方法 `__new__`，用于创建类的新实例
    def __new__(cls, field, parametricregion):

        # 获取场 `field` 的坐标系集合
        coord_set = _get_coord_systems(field)

        # 如果坐标系集合为空，则使用默认的三维坐标系 'C'
        if len(coord_set) == 0:
            coord_sys = CoordSys3D('C')
        # 如果坐标系集合大于1个，则抛出值错误异常
        elif len(coord_set) > 1:
            raise ValueError
        # 否则，选择集合中的第一个坐标系作为当前坐标系
        else:
            coord_sys = next(iter(coord_set))

        # 如果 parametricregion 的维度为0，则返回零标量 S.Zero
        if parametricregion.dimensions == 0:
            return S.Zero

        # 获取当前坐标系的基向量和基标量
        base_vectors = coord_sys.base_vectors()
        base_scalars = coord_sys.base_scalars()

        # 将 parametricfield 初始化为 field
        parametricfield = field

        # 初始化位置矢量 r 为零向量 Vector.zero
        r = Vector.zero
        # 遍历 parametricregion 的定义，计算位置矢量 r
        for i in range(len(parametricregion.definition)):
            r += base_vectors[i] * parametricregion.definition[i]

        # 如果坐标系集合不为空，替换 parametricfield 中的基标量为 parametricregion 的定义值
        if len(coord_set) != 0:
            for i in range(len(parametricregion.definition)):
                parametricfield = parametricfield.subs(base_scalars[i], parametricregion.definition[i])

        # 如果 parametricregion 的维度为1
        if parametricregion.dimensions == 1:
            # 获取参数 parameter
            parameter = parametricregion.parameters[0]

            # 计算位置矢量 r 对 parameter 的导数 r_diff
            r_diff = diff(r, parameter)
            # 获取 parameter 的上下限
            lower, upper = parametricregion.limits[parameter][0], parametricregion.limits[parameter][1]

            # 如果 parametricfield 是向量，则计算 r_diff 与 parametricfield 的点积 integrand
            if isinstance(parametricfield, Vector):
                integrand = simplify(r_diff.dot(parametricfield))
            # 否则，计算 r_diff 的大小与 parametricfield 的乘积 integrand
            else:
                integrand = simplify(r_diff.magnitude() * parametricfield)

            # 对 integrand 进行积分，参数为 (parameter, lower, upper)
            result = integrate(integrand, (parameter, lower, upper))

        # 如果 parametricregion 的维度为2
        elif parametricregion.dimensions == 2:
            # 获取参数 u 和 v
            u, v = cls._bounds_case(parametricregion.parameters, parametricregion.limits)

            # 计算位置矢量 r 对 u 和 v 的偏导数 r_u 和 r_v
            r_u = diff(r, u)
            r_v = diff(r, v)
            # 计算法向量 normal_vector
            normal_vector = simplify(r_u.cross(r_v))

            # 如果 parametricfield 是向量，则计算 parametricfield 与 normal_vector 的点积 integrand
            if isinstance(parametricfield, Vector):
                integrand = parametricfield.dot(normal_vector)
            # 否则，计算 parametricfield 与 normal_vector 的大小的乘积 integrand
            else:
                integrand = parametricfield * normal_vector.magnitude()

            # 简化 integrand
            integrand = simplify(integrand)

            # 获取参数 u 和 v 的上下限
            lower_u, upper_u = parametricregion.limits[u][0], parametricregion.limits[u][1]
            lower_v, upper_v = parametricregion.limits[v][0], parametricregion.limits[v][1]

            # 对 integrand 进行双重积分，参数为 ((u, lower_u, upper_u), (v, lower_v, upper_v))
            result = integrate(integrand, (u, lower_u, upper_u), (v, lower_v, upper_v))

        # 如果 parametricregion 的维度大于2
        else:
            # 获取与 parametricregion 相关的变量集合
            variables = cls._bounds_case(parametricregion.parameters, parametricregion.limits)
            # 计算参数区域的雅可比行列式系数 coeff
            coeff = Matrix(parametricregion.definition).jacobian(variables).det()
            # 计算 integrand
            integrand = simplify(parametricfield * coeff)

            # 构建积分的限制列表 l
            l = [(var, parametricregion.limits[var][0], parametricregion.limits[var][1]) for var in variables]
            # 对 integrand 进行多重积分，参数为 *l
            result = integrate(integrand, *l)

        # 如果 result 不是积分对象 Integral，则直接返回结果
        if not isinstance(result, Integral):
            return result
        # 否则，调用父类的 `__new__` 方法创建类的新实例
        else:
            return super().__new__(cls, field, parametricregion)

    # 定义一个类方法 `@classmethod`
    @classmethod
    # 定义类方法 `_bounds_case`，接收参数 `cls`, `parameters`, `limits`
    def _bounds_case(cls, parameters, limits):

        # 获取限制条件的键列表
        V = list(limits.keys())
        # 初始化空的边列表
        E = []

        # 遍历参数列表 V
        for p in V:
            # 获取当前参数 p 的下限和上限
            lower_p = limits[p][0]
            upper_p = limits[p][1]

            # 获取下限和上限的原子表达式集合
            lower_p = lower_p.atoms()
            upper_p = upper_p.atoms()

            # 将满足条件的边 (p, q) 添加到边列表 E 中，其中 q 是 V 中的另一个参数，且 p != q
            E.extend((p, q) for q in V if p != q and
                     (lower_p.issuperset({q}) or upper_p.issuperset({q})))

        # 如果边列表 E 为空，则返回参数列表 parameters
        if not E:
            return parameters
        else:
            # 否则，对顶点集合 V 和边集合 E 进行拓扑排序，并返回排序结果
            return topological_sort((V, E), key=default_sort_key)

    # 定义属性方法 field，返回对象的第一个参数
    @property
    def field(self):
        return self.args[0]

    # 定义属性方法 parametricregion，返回对象的第二个参数
    @property
    def parametricregion(self):
        return self.args[1]
# 定义函数，用于计算向量/标量场在区域或一组参数上的积分
def vector_integrate(field, *region):
    """
    Compute the integral of a vector/scalar field
    over a a region or a set of parameters.

    Examples
    ========
    >>> from sympy.vector import CoordSys3D, ParametricRegion, vector_integrate
    >>> from sympy.abc import x, y, t
    >>> C = CoordSys3D('C')

    >>> region = ParametricRegion((t, t**2), (t, 1, 5))
    >>> vector_integrate(C.x*C.i, region)
    12

    Integrals over some objects of geometry module can also be calculated.

    >>> from sympy.geometry import Point, Circle, Triangle
    >>> c = Circle(Point(0, 2), 5)
    >>> vector_integrate(C.x**2 + C.y**2, c)
    290*pi
    >>> triangle = Triangle(Point(-2, 3), Point(2, 3), Point(0, 5))
    >>> vector_integrate(3*C.x**2*C.y*C.i + C.j, triangle)
    -8

    Integrals over some simple implicit regions can be computed. But in most cases,
    it takes too long to compute over them. This is due to the expressions of parametric
    representation becoming large.

    >>> from sympy.vector import ImplicitRegion
    >>> c2 = ImplicitRegion((x, y), (x - 2)**2 + (y - 1)**2 - 9)
    >>> vector_integrate(1, c2)
    6*pi

    Integral of fields with respect to base scalars:

    >>> vector_integrate(12*C.y**3, (C.y, 1, 3))
    240
    >>> vector_integrate(C.x**2*C.z, C.x)
    C.x**3*C.z/3
    >>> vector_integrate(C.x*C.i - C.y*C.k, C.x)
    (Integral(C.x, C.x))*C.i + (Integral(-C.y, C.x))*C.k
    >>> _.doit()
    C.x**2/2*C.i + (-C.x*C.y)*C.k

    """

    # 如果只有一个参数，且该参数是 ParametricRegion 类型，则返回 ParametricIntegral 对象
    if len(region) == 1:
        if isinstance(region[0], ParametricRegion):
            return ParametricIntegral(field, region[0])

        # 如果参数是 ImplicitRegion 类型，则转换为 ParametricRegion 并递归调用 vector_integrate 函数
        if isinstance(region[0], ImplicitRegion):
            region = parametric_region_list(region[0])[0]
            return vector_integrate(field, region)

        # 如果参数是 GeometryEntity 类型，则获取其对应的 ParametricRegion 列表，分别计算积分并返回总和
        if isinstance(region[0], GeometryEntity):
            regions_list = parametric_region_list(region[0])

            result = 0
            for reg in regions_list:
                result += vector_integrate(field, reg)
            return result

    # 如果有多个参数或参数不符合以上类型，则调用 integrate 函数进行普通积分计算
    return integrate(field, *region)
```