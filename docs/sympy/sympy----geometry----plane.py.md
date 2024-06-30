# `D:\src\scipysrc\sympy\sympy\geometry\plane.py`

```
# 导入所需模块和类
from sympy.core import Dummy, Rational, S, Symbol  # 导入符号计算相关模块
from sympy.core.symbol import _symbol  # 导入符号类
from sympy.functions.elementary.trigonometric import cos, sin, acos, asin, sqrt  # 导入三角函数和平方根函数
from .entity import GeometryEntity  # 导入几何实体类
from .line import (Line, Ray, Segment, Line3D, LinearEntity, LinearEntity3D,  # 导入线段和直线类
                   Ray3D, Segment3D)
from .point import Point, Point3D  # 导入点和三维点类
from sympy.matrices import Matrix  # 导入矩阵类
from sympy.polys.polytools import cancel  # 导入多项式工具类中的取消函数
from sympy.solvers import solve, linsolve  # 导入解方程和线性方程组解的函数
from sympy.utilities.iterables import uniq, is_sequence  # 导入去重和判断是否为序列的工具函数
from sympy.utilities.misc import filldedent, func_name, Undecidable  # 导入填充缩进文本和函数名的工具函数

from mpmath.libmp.libmpf import prec_to_dps  # 导入精度到小数位数的转换函数

import random  # 导入随机数生成模块

# 创建四个虚拟符号对象，用于后续的符号计算
x, y, z, t = [Dummy('plane_dummy') for i in range(4)]


class Plane(GeometryEntity):
    """
    A plane is a flat, two-dimensional surface. A plane is the two-dimensional
    analogue of a point (zero-dimensions), a line (one-dimension) and a solid
    (three-dimensions). A plane can generally be constructed by two types of
    inputs. They are:
    - three non-collinear points
    - a point and the plane's normal vector

    Attributes
    ==========

    p1
        第一个点
    normal_vector
        法向量

    Examples
    ========

    >>> from sympy import Plane, Point3D
    >>> Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
    Plane(Point3D(1, 1, 1), (-1, 2, -1))
    >>> Plane((1, 1, 1), (2, 3, 4), (2, 2, 2))
    Plane(Point3D(1, 1, 1), (-1, 2, -1))
    >>> Plane(Point3D(1, 1, 1), normal_vector=(1,4,7))
    Plane(Point3D(1, 1, 1), (1, 4, 7))

    """
    def __new__(cls, p1, a=None, b=None, **kwargs):
        # 将输入的第一个点转换为三维点对象
        p1 = Point3D(p1, dim=3)
        if a and b:
            # 将输入的第二个点和第三个点转换为三维点对象
            p2 = Point(a, dim=3)
            p3 = Point(b, dim=3)
            # 检查三个点是否共线，如果共线则引发值错误
            if Point3D.are_collinear(p1, p2, p3):
                raise ValueError('Enter three non-collinear points')
            # 计算第一个点到第二个点和第三个点的方向比率
            a = p1.direction_ratio(p2)
            b = p1.direction_ratio(p3)
            # 计算法向量，使用矩阵计算叉乘得到
            normal_vector = tuple(Matrix(a).cross(Matrix(b)))
        else:
            # 如果没有给定第二个点和第三个点，检查是否提供了法向量
            a = kwargs.pop('normal_vector', a)
            evaluate = kwargs.get('evaluate', True)
            if is_sequence(a) and len(a) == 3:
                # 如果提供了长度为3的序列作为法向量，将其转换为三维点的参数
                normal_vector = Point3D(a).args if evaluate else a
            else:
                # 如果不符合上述条件，引发值错误
                raise ValueError(filldedent('''
                    Either provide 3 3D points or a point with a
                    normal vector expressed as a sequence of length 3'''))
            # 检查法向量是否全为零，如果是则引发值错误
            if all(coord.is_zero for coord in normal_vector):
                raise ValueError('Normal vector cannot be zero vector')
        # 使用父类的构造函数创建几何实体对象并返回
        return GeometryEntity.__new__(cls, p1, normal_vector, **kwargs)
    def __contains__(self, o):
        # 计算方程的结果
        k = self.equation(x, y, z)
        # 如果 o 是 LinearEntity 或者 LinearEntity3D 类型
        if isinstance(o, (LinearEntity, LinearEntity3D)):
            # 获取 o 的任意点
            d = Point3D(o.arbitrary_point(t))
            # 将点的坐标代入方程，检查结果是否为零
            e = k.subs([(x, d.x), (y, d.y), (z, d.z)])
            return e.equals(0)
        # 如果 o 是 Point 类型，尝试将其转换为 Point3D 类型
        try:
            o = Point(o, dim=3, strict=True)
            # 将点的坐标代入方程，检查结果是否为零
            d = k.xreplace(dict(zip((x, y, z), o.args)))
            return d.equals(0)
        except TypeError:
            # 如果转换失败，返回 False
            return False

    def _eval_evalf(self, prec=15, **options):
        # 获取参数 pt 和 tup
        pt, tup = self.args
        # 将给定的精度转换为小数点位数
        dps = prec_to_dps(prec)
        # 对 pt 进行数值计算
        pt = pt.evalf(n=dps, **options)
        # 对 tup 中的每个元素进行数值计算
        tup = tuple([i.evalf(n=dps, **options) for i in tup])
        # 返回计算后的对象，不进行评估
        return self.func(pt, normal_vector=tup, evaluate=False)

    def angle_between(self, o):
        """Angle between the plane and other geometric entity.

        Parameters
        ==========

        LinearEntity3D, Plane.

        Returns
        =======

        angle : angle in radians

        Notes
        =====

        This method accepts only 3D entities as it's parameter, but if you want
        to calculate the angle between a 2D entity and a plane you should
        first convert to a 3D entity by projecting onto a desired plane and
        then proceed to calculate the angle.

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 2, 2), normal_vector=(1, 2, 3))
        >>> b = Line3D(Point3D(1, 3, 4), Point3D(2, 2, 2))
        >>> a.angle_between(b)
        -asin(sqrt(21)/6)

        """
        # 如果 o 是 LinearEntity3D 类型
        if isinstance(o, LinearEntity3D):
            # 计算两个向量的点积
            a = Matrix(self.normal_vector)
            b = Matrix(o.direction_ratio)
            c = a.dot(b)
            # 计算向量的长度
            d = sqrt(sum(i**2 for i in self.normal_vector))
            e = sqrt(sum(i**2 for i in o.direction_ratio))
            # 计算角度并返回
            return asin(c/(d*e))
        # 如果 o 是 Plane 类型
        if isinstance(o, Plane):
            # 计算两个法向量的点积
            a = Matrix(self.normal_vector)
            b = Matrix(o.normal_vector)
            c = a.dot(b)
            # 计算法向量的长度
            d = sqrt(sum(i**2 for i in self.normal_vector))
            e = sqrt(sum(i**2 for i in o.normal_vector))
            # 计算角度并返回
            return acos(c/(d*e))
    def arbitrary_point(self, u=None, v=None):
        """ 返回平面上的任意点。如果给定两个参数，点在整个平面上变化。
        如果给定1个或没有参数，返回一个带有一个参数的点，该参数在0到2*pi范围内变化，
        绕平面的p1点形成半径为1的圆。

        Examples
        ========

        >>> from sympy import Plane, Ray
        >>> from sympy.abc import u, v, t, r
        >>> p = Plane((1, 1, 1), normal_vector=(1, 0, 0))
        >>> p.arbitrary_point(u, v)
        Point3D(1, u + 1, v + 1)
        >>> p.arbitrary_point(t)
        Point3D(1, cos(t) + 1, sin(t) + 1)

        当任意给定u和v的值时，点可以在平面上任意移动，
        而单参数点可以用来构建一条射线，其任意点可以在角度t和半径r处从p.p1定位：

        >>> Ray(p.p1, _).arbitrary_point(r)
        Point3D(1, r*cos(t) + 1, r*sin(t) + 1)

        Returns
        =======

        Point3D

        """
        circle = v is None
        # 如果只有一个参数，将u定义为实数符号t
        if circle:
            u = _symbol(u or 't', real=True)
        else:
            # 如果有两个参数，分别定义u和v为实数符号u和v
            u = _symbol(u or 'u', real=True)
            v = _symbol(v or 'v', real=True)
        x, y, z = self.normal_vector
        a, b, c = self.p1.args
        # x1, y1, z1 是与平面平行且非零的向量
        if x.is_zero and y.is_zero:
            x1, y1, z1 = S.One, S.Zero, S.Zero
        else:
            x1, y1, z1 = -y, x, S.Zero
        # x2, y2, z2 也是与平面平行的向量，并且与x1, y1, z1正交
        x2, y2, z2 = tuple(Matrix((x, y, z)).cross(Matrix((x1, y1, z1))))
        # 如果是圆形参数，归一化向量x1, y1, z1和x2, y2, z2
        if circle:
            x1, y1, z1 = (w/sqrt(x1**2 + y1**2 + z1**2) for w in (x1, y1, z1))
            x2, y2, z2 = (w/sqrt(x2**2 + y2**2 + z2**2) for w in (x2, y2, z2))
            # 构造圆形参数的点
            p = Point3D(a + x1*cos(u) + x2*sin(u), \
                        b + y1*cos(u) + y2*sin(u), \
                        c + z1*cos(u) + z2*sin(u))
        else:
            # 构造给定u和v参数的点
            p = Point3D(a + x1*u + x2*v, b + y1*u + y2*v, c + z1*u + z2*v)
        return p
    def are_concurrent(*planes):
        """判断一组平面是否共线。

        当两个或多个平面的交线为公共直线时，它们称为共线。

        Parameters
        ==========

        planes: list
            包含待检查平面的列表

        Returns
        =======

        Boolean
            如果平面共线返回 True，否则返回 False

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(5, 0, 0), normal_vector=(1, -1, 1))
        >>> b = Plane(Point3D(0, -2, 0), normal_vector=(3, 1, 1))
        >>> c = Plane(Point3D(0, -1, 0), normal_vector=(5, -1, 9))
        >>> Plane.are_concurrent(a, b)
        True
        >>> Plane.are_concurrent(a, b, c)
        False

        """
        planes = list(uniq(planes))  # 去除重复的平面对象
        for i in planes:
            if not isinstance(i, Plane):
                raise ValueError('All objects should be Planes but got %s' % i.func)
                # 如果不是平面对象则引发 ValueError 异常
        if len(planes) < 2:
            return False  # 如果平面数量小于 2，则返回 False
        planes = list(planes)
        first = planes.pop(0)  # 弹出第一个平面对象
        sol = first.intersection(planes[0])  # 计算第一个平面与第二个平面的交线
        if sol == []:
            return False  # 如果交线为空列表，则返回 False
        else:
            line = sol[0]  # 获取第一个交线
            for i in planes[1:]:
                l = first.intersection(i)  # 计算第一个平面与当前平面的交线
                if not l or l[0] not in line:
                    return False  # 如果交线不存在或者不在之前的线上，则返回 False
            return True  # 如果所有平面的交线都在同一条线上，则返回 True


    def distance(self, o):
        """计算平面与另一个几何实体的距离。

        Parameters
        ==========

        o: Point3D, LinearEntity3D, Plane
            另一个几何实体，可以是点、直线或平面

        Returns
        =======

        distance
            返回平面与另一个几何实体的距离

        Notes
        =====

        该方法只接受三维实体作为参数。如果需要计算二维实体与平面的距离，
        需要先投影到所需的平面上，然后再进行距离计算。

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))
        >>> b = Point3D(1, 2, 3)
        >>> a.distance(b)
        sqrt(3)
        >>> c = Line3D(Point3D(2, 3, 1), Point3D(1, 2, 2))
        >>> a.distance(c)
        0

        """
        if self.intersection(o) != []:
            return S.Zero

        if isinstance(o, (Segment3D, Ray3D)):
            a, b = o.p1, o.p2
            pi, = self.intersection(Line3D(a, b))
            if pi in o:
                return self.distance(pi)
            elif a in Segment3D(pi, b):
                return self.distance(a)
            else:
                assert isinstance(o, Segment3D) is True
                return self.distance(b)

        # 处理 `Point3D`, `LinearEntity3D`, `Plane`
        a = o if isinstance(o, Point3D) else o.p1
        n = Point3D(self.normal_vector).unit
        d = (a - self.p1).dot(n)
        return abs(d)
    def equals(self, o):
        """
        Returns True if self and o are the same mathematical entities.

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))
        >>> b = Plane(Point3D(1, 2, 3), normal_vector=(2, 2, 2))
        >>> c = Plane(Point3D(1, 2, 3), normal_vector=(-1, 4, 6))
        >>> a.equals(a)
        True
        >>> a.equals(b)
        True
        >>> a.equals(c)
        False
        """
        # 检查 o 是否为 Plane 类的实例
        if isinstance(o, Plane):
            # 获取当前对象和 o 的方程式
            a = self.equation()
            b = o.equation()
            # 检查方程式是否可以化简为常数
            return cancel(a/b).is_constant()
        else:
            # 如果 o 不是 Plane 类的实例，则返回 False
            return False


    def equation(self, x=None, y=None, z=None):
        """The equation of the Plane.

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 2), Point3D(2, 4, 7), Point3D(3, 5, 1))
        >>> a.equation()
        -23*x + 11*y - 2*z + 16
        >>> a = Plane(Point3D(1, 4, 2), normal_vector=(6, 6, 6))
        >>> a.equation()
        6*x + 6*y + 6*z - 42

        """
        # 如果没有提供 x, y, z 值，则创建符号变量 x, y, z
        x, y, z = [i if i else Symbol(j, real=True) for i, j in zip((x, y, z), 'xyz')]
        # 创建 Point3D 对象 a，表示点 (x, y, z)
        a = Point3D(x, y, z)
        # 计算点 a 到 self.p1 的方向比例
        b = self.p1.direction_ratio(a)
        # 获取平面的法向量
        c = self.normal_vector
        # 返回平面方程的表达式，使用点乘计算
        return (sum(i*j for i, j in zip(b, c)))
    # 定义一个方法用于计算当前几何实体与另一个几何实体的交集
    def intersection(self, o):
        """ The intersection with other geometrical entity.

        Parameters
        ==========

        Point, Point3D, LinearEntity, LinearEntity3D, Plane
            可以接受的参数类型，表示可以与当前几何实体进行交集计算的对象类型

        Returns
        =======

        List
            返回一个包含交集结果的列表，可能包括 Point3D 或 Line3D 对象

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))
        >>> b = Point3D(1, 2, 3)
        >>> a.intersection(b)
        [Point3D(1, 2, 3)]
        >>> c = Line3D(Point3D(1, 4, 7), Point3D(2, 2, 2))
        >>> a.intersection(c)
        [Point3D(2, 2, 2)]
        >>> d = Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))
        >>> e = Plane(Point3D(2, 0, 0), normal_vector=(3, 4, -3))
        >>> d.intersection(e)
        [Line3D(Point3D(78/23, -24/23, 0), Point3D(147/23, 321/23, 23))]

        """
        # 检查 o 是否为 GeometryEntity 的实例，如果不是，则将其转换为 Point 对象（三维）
        if not isinstance(o, GeometryEntity):
            o = Point(o, dim=3)
        
        # 如果 o 是 Point 对象
        if isinstance(o, Point):
            # 如果 o 在当前对象中，则返回包含 o 的列表，否则返回空列表
            if o in self:
                return [o]
            else:
                return []
        
        # 如果 o 是 LinearEntity 或 LinearEntity3D 对象
        if isinstance(o, (LinearEntity, LinearEntity3D)):
            # 将 o 重新投影为三维对象
            p1, p2 = o.p1, o.p2
            if isinstance(o, Segment):
                o = Segment3D(p1, p2)
            elif isinstance(o, Ray):
                o = Ray3D(p1, p2)
            elif isinstance(o, Line):
                o = Line3D(p1, p2)
            else:
                raise ValueError('unhandled linear entity: %s' % o.func)
            
            # 如果 o 在当前对象中，则返回包含 o 的列表，否则返回空列表
            if o in self:
                return [o]
            else:
                # 计算与平面相交的情况
                a = Point3D(o.arbitrary_point(t))
                p1, n = self.p1, Point3D(self.normal_vector)
                
                # 解方程 (a - p1)·n = 0，找出交点参数 t
                c = solve((a - p1).dot(n), t)
                if not c:
                    return []
                else:
                    # 筛选出实数解
                    c = [i for i in c if i.is_real is not False]
                    if len(c) > 1:
                        c = [i for i in c if i.is_real]
                    if len(c) != 1:
                        raise Undecidable("not sure which point is real")
                    
                    # 计算实际交点 p，并检查它是否在 o 中
                    p = a.subs(t, c[0])
                    if p not in o:
                        return []  # 例如，线段可能与平面不相交
                    return [p]
        
        # 如果 o 是 Plane 对象
        if isinstance(o, Plane):
            # 如果当前平面与 o 平面相等，则返回包含当前平面的列表
            if self.equals(o):
                return [self]
            # 如果当前平面与 o 平行，则返回空列表
            if self.is_parallel(o):
                return []
            else:
                # 否则计算两平面的交线
                x, y, z = map(Dummy, 'xyz')
                a, b = Matrix([self.normal_vector]), Matrix([o.normal_vector])
                c = list(a.cross(b))
                d = self.equation(x, y, z)
                e = o.equation(x, y, z)
                result = list(linsolve([d, e], x, y, z))[0]
                for i in (x, y, z): result = result.subs(i, 0)
                return [Line3D(Point3D(result), direction_ratio=c)]
    def is_coplanar(self, o):
        """ Returns True if `o` is coplanar with self, else False.

        Examples
        ========

        >>> from sympy import Plane
        >>> o = (0, 0, 0)
        >>> p = Plane(o, (1, 1, 1))
        >>> p2 = Plane(o, (2, 2, 2))
        >>> p == p2
        False
        >>> p.is_coplanar(p2)
        True
        """
        # 如果 `o` 是一个 Plane 对象
        if isinstance(o, Plane):
            # 检查两个平面方程是否通过取消变量 x, y, z 后相等，即判断是否共面
            return not cancel(self.equation(x, y, z)/o.equation(x, y, z)).has(x, y, z)
        # 如果 `o` 是一个 Point3D 对象
        if isinstance(o, Point3D):
            # 检查点是否在当前平面上
            return o in self
        # 如果 `o` 是一个 LinearEntity3D 对象
        elif isinstance(o, LinearEntity3D):
            # 检查所有点是否在当前平面上
            return all(i in self for i in self)
        # 如果 `o` 是一个 GeometryEntity 对象（现在应该只处理二维对象）
        elif isinstance(o, GeometryEntity):  # XXX should only be handling 2D objects now
            # 检查法向量的前两个分量是否为零，以判断是否共面
            return all(i == 0 for i in self.normal_vector[:2])


    def is_parallel(self, l):
        """Is the given geometric entity parallel to the plane?

        Parameters
        ==========

        LinearEntity3D or Plane

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> b = Plane(Point3D(3,1,3), normal_vector=(4, 8, 12))
        >>> a.is_parallel(b)
        True

        """
        # 如果 `l` 是一个 LinearEntity3D 对象
        if isinstance(l, LinearEntity3D):
            # 获取直线的方向比例和平面的法向量，并检查它们的点积是否为零，判断是否平行
            a = l.direction_ratio
            b = self.normal_vector
            return sum(i*j for i, j in zip(a, b)) == 0
        # 如果 `l` 是一个 Plane 对象
        if isinstance(l, Plane):
            # 将法向量转换为矩阵形式，并检查它们的叉乘是否为零矩阵，判断是否平行
            a = Matrix(l.normal_vector)
            b = Matrix(self.normal_vector)
            return bool(a.cross(b).is_zero_matrix)


    def is_perpendicular(self, l):
        """Is the given geometric entity perpendicualar to the given plane?

        Parameters
        ==========

        LinearEntity3D or Plane

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> b = Plane(Point3D(2, 2, 2), normal_vector=(-1, 2, -1))
        >>> a.is_perpendicular(b)
        True

        """
        # 如果 `l` 是一个 LinearEntity3D 对象
        if isinstance(l, LinearEntity3D):
            # 将直线的方向比例转换为矩阵形式，并与平面的法向量做叉乘，检查结果是否为零矩阵，判断是否垂直
            a = Matrix(l.direction_ratio)
            b = Matrix(self.normal_vector)
            if a.cross(b).is_zero_matrix:
                return True
            else:
                return False
        # 如果 `l` 是一个 Plane 对象
        elif isinstance(l, Plane):
           # 将法向量转换为矩阵形式，并检查它们的点积是否为零，判断是否垂直
           a = Matrix(l.normal_vector)
           b = Matrix(self.normal_vector)
           if a.dot(b) == 0:
               return True
           else:
               return False
        else:
            return False
    def normal_vector(self):
        """返回给定平面的法向量。

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
        >>> a.normal_vector
        (-1, 2, -1)
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 4, 7))
        >>> a.normal_vector
        (1, 4, 7)

        """
        return self.args[1]

    @property
    def p1(self):
        """平面的唯一定义点。其他点可以通过 arbitrary_point 方法获取。

        See Also
        ========

        sympy.geometry.point.Point3D

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
        >>> a.p1
        Point3D(1, 1, 1)

        """
        return self.args[0]

    def parallel_plane(self, pt):
        """
        返回与给定平面平行并且通过点 pt 的平面。

        Parameters
        ==========

        pt: Point3D

        Returns
        =======

        Plane

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1, 4, 6), normal_vector=(2, 4, 6))
        >>> a.parallel_plane(Point3D(2, 3, 5))
        Plane(Point3D(2, 3, 5), (2, 4, 6))

        """
        a = self.normal_vector
        return Plane(pt, normal_vector=a)

    def perpendicular_line(self, pt):
        """返回与给定平面垂直的直线。

        Parameters
        ==========

        pt: Point3D

        Returns
        =======

        Line3D

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> a.perpendicular_line(Point3D(9, 8, 7))
        Line3D(Point3D(9, 8, 7), Point3D(11, 12, 13))

        """
        a = self.normal_vector
        return Line3D(pt, direction_ratio=a)
    def perpendicular_plane(self, *pts):
        """
        返回通过给定点的垂直平面。如果点之间的方向比例与平面的法向量相同，
        为了从无限可能的平面中选择一个，将在z轴（或者如果法向量已经与z轴平行，则在y轴上）上选择第三个点。
        如果给出少于两个点，则补充如下：如果没有给出点，则使用self.p1作为pt1；
        如果没有给出第二个点，则它将是通过pt1的点，沿着与z轴平行的线（如果法向量不是z轴，则沿着与y轴平行的线）。

        Parameters
        ==========

        pts: 0, 1 or 2 Point3D
            给定的点，可以是0个，1个或2个Point3D对象

        Returns
        =======

        Plane
            返回一个Plane对象，表示通过给定点的垂直平面

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a, b = Point3D(0, 0, 0), Point3D(0, 1, 0)
        >>> Z = (0, 0, 1)
        >>> p = Plane(a, normal_vector=Z)
        >>> p.perpendicular_plane(a, b)
        Plane(Point3D(0, 0, 0), (1, 0, 0))
        """
        if len(pts) > 2:
            raise ValueError('No more than 2 pts should be provided.')

        pts = list(pts)
        if len(pts) == 0:
            pts.append(self.p1)
        if len(pts) == 1:
            x, y, z = self.normal_vector
            if x == y == 0:
                dir = (0, 1, 0)
            else:
                dir = (0, 0, 1)
            pts.append(pts[0] + Point3D(*dir))

        p1, p2 = [Point(i, dim=3) for i in pts]
        l = Line3D(p1, p2)
        n = Line3D(p1, direction_ratio=self.normal_vector)
        if l in n:  # XXX should an error be raised instead?
            # 有无限多个垂直平面；
            x, y, z = self.normal_vector
            if x == y == 0:
                # 法向量是z轴，因此选择y轴上的一个点
                p3 = Point3D(0, 1, 0)  # 情况1
            else:
                # 否则选择z轴上的一个点
                p3 = Point3D(0, 0, 1)  # 情况2
            # 如果该点已经给出，稍微移动它
            if p3 in l:
                p3 *= 2  # 情况3
        else:
            p3 = p1 + Point3D(*self.normal_vector)  # 情况4
        return Plane(p1, p2, p3)
    def projection_line(self, line):
        """Project the given line onto the plane through the normal plane
        containing the line.

        Parameters
        ==========

        line : LinearEntity or LinearEntity3D
            The line to be projected onto the plane.

        Returns
        =======

        Point3D, Line3D, Ray3D or Segment3D
            Returns a geometric object based on the type of the input line and its projection.

        Notes
        =====

        This method projects a line onto a plane defined by its normal vector. It handles
        interaction between 2D and 3D lines (segments, rays) by converting 2D lines to
        3D through projection onto the plane.

        Examples
        ========

        >>> from sympy import Plane, Line, Line3D, Point3D
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))
        >>> b = Line(Point3D(1, 1), Point3D(2, 2))
        >>> a.projection_line(b)
        Line3D(Point3D(4/3, 4/3, 1/3), Point3D(5/3, 5/3, -1/3))
        >>> c = Line3D(Point3D(1, 1, 1), Point3D(2, 2, 2))
        >>> a.projection_line(c)
        Point3D(1, 1, 1)

        """
        # Check if the input line is of the correct type (either LinearEntity or LinearEntity3D)
        if not isinstance(line, (LinearEntity, LinearEntity3D)):
            raise NotImplementedError('Enter a linear entity only')
        
        # Project the endpoints of the line onto the plane and get their projections
        a, b = self.projection(line.p1), self.projection(line.p2)
        
        # If both projections are the same point, return that point
        if a == b:
            return a
        
        # Determine the type of line and return the appropriate geometric object
        if isinstance(line, (Line, Line3D)):
            return Line3D(a, b)
        if isinstance(line, (Ray, Ray3D)):
            return Ray3D(a, b)
        if isinstance(line, (Segment, Segment3D)):
            return Segment3D(a, b)

    def projection(self, pt):
        """Project the given point onto the plane along the plane normal.

        Parameters
        ==========

        pt : Point or Point3D
            The point to be projected onto the plane.

        Returns
        =======

        Point3D
            Returns the projected point on the plane.

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> A = Plane(Point3D(1, 1, 2), normal_vector=(1, 1, 1))

        The projection is along the normal vector direction, not the z
        axis, so (1, 1) does not project to (1, 1, 2) on the plane A:

        >>> b = Point3D(1, 1)
        >>> A.projection(b)
        Point3D(5/3, 5/3, 2/3)
        >>> _ in A
        True

        But the point (1, 1, 2) projects to (1, 1) on the XY-plane:

        >>> XY = Plane((0, 0, 0), (0, 0, 1))
        >>> XY.projection((1, 1, 2))
        Point3D(1, 1, 0)
        """
        # Convert the given point to Point3D if it's not already
        rv = Point(pt, dim=3)
        
        # Check if the projected point lies on the plane, if so return it
        if rv in self:
            return rv
        
        # Project the point onto the plane using its normal vector
        return self.intersection(Line3D(rv, rv + Point3D(self.normal_vector)))[0]
    def random_point(self, seed=None):
        """ 
        返回平面上的一个随机点。

        Parameters
        ==========

        seed : int or None, optional
            随机数种子。如果指定，将用于生成随机数；如果未指定，将使用系统默认的随机数生成器。

        Returns
        =======

        Point3D
            返回一个在平面上的随机点，作为 Point3D 对象。

        Examples
        ========

        >>> from sympy import Plane
        >>> p = Plane((1, 0, 0), normal_vector=(0, 1, 0))
        >>> r = p.random_point(seed=42)  # 可选地使用种子值
        >>> r.n(3)
        Point3D(2.29, 0, -1.35)

        随机点可以移动以落在以 p1 为中心、半径为1的圆上：

        >>> c = p.p1 + (r - p.p1).unit
        >>> c.distance(p.p1).equals(1)
        True
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        params = {
            x: 2*Rational(rng.gauss(0, 1)) - 1,  # 生成 x 坐标的随机参数
            y: 2*Rational(rng.gauss(0, 1)) - 1}  # 生成 y 坐标的随机参数
        return self.arbitrary_point(x, y).subs(params)

    def parameter_value(self, other, u, v=None):
        """ 
        返回与给定点对应的参数值或参数值字典。

        Parameters
        ==========

        other : GeometryEntity
            给定的点或几何实体。

        u : Symbol
            第一个参数符号。

        v : Symbol, optional
            第二个参数符号（如果需要返回两个参数值的字典时需要提供）。

        Returns
        =======

        dict
            如果只有一个参数，返回形如 {u: value} 的字典；如果有两个参数，返回形如 {u: uvalue, v: vvalue} 的字典。

        Raises
        ======

        ValueError
            如果给定的 other 不是 Point 类型或几何实体，或者参数不正确时抛出异常。

        Examples
        ========

        >>> from sympy import pi, Plane
        >>> from sympy.abc import t, u, v
        >>> p = Plane((2, 0, 0), (0, 0, 1), (0, 1, 0))

        默认情况下，返回的参数值定义了一个距离平面 p1 1单位距离的点，并与给定点在一条直线上：

        >>> on_circle = p.arbitrary_point(t).subs(t, pi/4)
        >>> on_circle.distance(p.p1)
        1
        >>> p.parameter_value(on_circle, t)
        {t: pi/4}

        将点从 p1 移动两倍距离，参数值不变：

        >>> off_circle = p.p1 + (on_circle - p.p1)*2
        >>> off_circle.distance(p.p1)
        2
        >>> p.parameter_value(off_circle, t)
        {t: pi/4}

        如果需要返回两个参数值的字典，可以提供两个参数符号 u 和 v：

        >>> p.parameter_value(on_circle, u, v)
        {u: sqrt(10)/10, v: sqrt(10)/30}
        >>> p.parameter_value(off_circle, u, v)
        {u: sqrt(10)/5, v: sqrt(10)/15}
        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other, Point):
            raise ValueError("other must be a point")
        if other == self.p1:
            return other
        if isinstance(u, Symbol) and v is None:
            delta = self.arbitrary_point(u) - self.p1
            eq = delta - (other - self.p1).unit
            sol = solve(eq, u, dict=True)
        elif isinstance(u, Symbol) and isinstance(v, Symbol):
            pt = self.arbitrary_point(u, v)
            sol = solve(pt - other, (u, v), dict=True)
        else:
            raise ValueError('expecting 1 or 2 symbols')
        if not sol:
            raise ValueError("Given point is not on %s" % func_name(self))
        return sol[0]  # 返回参数值字典中的第一个解，例如 {t: tval} 或 {u: uval, v: vval}
    # 定义一个方法 `ambient_dimension`，属于当前类的实例方法
    def ambient_dimension(self):
        # 返回属性 `p1` 的 `ambient_dimension` 属性值
        return self.p1.ambient_dimension
```