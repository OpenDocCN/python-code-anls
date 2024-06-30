# `D:\src\scipysrc\sympy\sympy\geometry\point.py`

```
# 引入警告模块，用于处理可能出现的警告信息
import warnings

# 导入 sympy 核心模块中的相关子模块和类
from sympy.core import S, sympify, Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.numbers import Float
from sympy.core.parameters import global_parameters
from sympy.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import Matrix
from sympy.matrices.expressions import Transpose
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable

# 导入自定义的实体类 GeometryEntity
from .entity import GeometryEntity

# 导入 mpmath 库中的 libmpf 模块，用于精度计算
from mpmath.libmp.libmpf import prec_to_dps


class Point(GeometryEntity):
    """A point in a n-dimensional Euclidean space.

    Parameters
    ==========

    coords : sequence of n-coordinate values. In the special
        case where n=2 or 3, a Point2D or Point3D will be created
        as appropriate.
    evaluate : if `True` (default), all floats are turn into
        exact types.
    dim : number of coordinates the point should have.  If coordinates
        are unspecified, they are padded with zeros.
    on_morph : indicates what should happen when the number of
        coordinates of a point need to be changed by adding or
        removing zeros.  Possible values are `'warn'`, `'error'`, or
        `ignore` (default).  No warning or error is given when `*args`
        is empty and `dim` is given. An error is always raised when
        trying to remove nonzero coordinates.


    Attributes
    ==========

    length
    origin: A `Point` representing the origin of the
        appropriately-dimensioned space.

    Raises
    ======

    TypeError : When instantiating with anything but a Point or sequence
    ValueError : when instantiating with a sequence with length < 2 or
        when trying to reduce dimensions if keyword `on_morph='error'` is
        set.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy import Point
    >>> from sympy.abc import x
    >>> Point(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point([1, 2])
    Point2D(1, 2)
    >>> Point(0, x)
    Point2D(0, x)
    >>> Point(dim=4)
    Point(0, 0, 0, 0)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    # 类属性，指示这是一个 Point 类
    is_Point = True
    def __new__(cls, *args, **kwargs):
        # 获取 evaluate 参数，如果不存在则使用全局参数中的 evaluate
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        # 获取 on_morph 参数，如果不存在则默认为 'ignore'
        on_morph = kwargs.get('on_morph', 'ignore')

        # 将传入的坐标参数解包为 coords
        coords = args[0] if len(args) == 1 else args

        # 如果 coords 是 Point 实例，则不重新评估其坐标
        if isinstance(coords, Point):
            # 即使我们改变点的维度，也不重新评估其坐标
            evaluate = False
            if len(coords) == kwargs.get('dim', len(coords)):
                return coords

        # 如果 coords 不是序列，则抛出类型错误
        if not is_sequence(coords):
            raise TypeError(filldedent('''
                Expecting sequence of coordinates, not `{}`'''
                                       .format(func_name(coords))))
        
        # 如果 coords 的长度为 0，并且提供了 'dim' 关键字参数，则初始化为全零坐标
        if len(coords) == 0 and kwargs.get('dim', None):
            coords = (S.Zero,)*kwargs.get('dim')

        # 将 coords 转换为 Tuple 类型
        coords = Tuple(*coords)
        # 获取或设定 dim 参数
        dim = kwargs.get('dim', len(coords))

        # 如果 coords 长度小于 2，则抛出值错误
        if len(coords) < 2:
            raise ValueError(filldedent('''
                Point requires 2 or more coordinates or
                keyword `dim` > 1.'''))
        
        # 如果 coords 的长度与 dim 不一致，则根据 on_morph 参数处理
        if len(coords) != dim:
            message = ("Dimension of {} needs to be changed "
                       "from {} to {}.").format(coords, len(coords), dim)
            if on_morph == 'ignore':
                pass
            elif on_morph == "error":
                raise ValueError(message)
            elif on_morph == 'warn':
                warnings.warn(message, stacklevel=2)
            else:
                raise ValueError(filldedent('''
                        on_morph value should be 'error',
                        'warn' or 'ignore'.'''))

        # 如果 coords[dim:] 中存在非零值，则抛出值错误
        if any(coords[dim:]):
            raise ValueError('Nonzero coordinates cannot be removed.')

        # 如果 coords 中存在非零实部非零的复数，则抛出值错误
        if any(a.is_number and im(a).is_zero is False for a in coords):
            raise ValueError('Imaginary coordinates are not permitted.')

        # 如果 coords 中不全是 SymPy 表达式，则抛出类型错误
        if not all(isinstance(a, Expr) for a in coords):
            raise TypeError('Coordinates must be valid SymPy expressions.')

        # 根据 dim 合适地用零填充 coords
        coords = coords[:dim] + (S.Zero,)*(dim - len(coords))

        # 如果 evaluate 为 True，则将 Float 类型的坐标替换为有理数，并简化表达式
        if evaluate:
            coords = coords.xreplace({
                f: simplify(nsimplify(f, rational=True))
                 for f in coords.atoms(Float)})

        # 根据 coords 的长度返回 2D 或 3D 的点实例
        if len(coords) == 2:
            kwargs['_nocheck'] = True
            return Point2D(*coords, **kwargs)
        elif len(coords) == 3:
            kwargs['_nocheck'] = True
            return Point3D(*coords, **kwargs)

        # 默认返回通用的 Point 实例
        return GeometryEntity.__new__(cls, *coords)
    def __abs__(self):
        """Returns the distance between this point and the origin."""
        # 创建一个与当前点相同维度、坐标全为零的原点 Point 对象
        origin = Point([0]*len(self))
        # 调用 Point 类的 distance 静态方法计算当前点与原点之间的距离，并返回结果
        return Point.distance(origin, self)

    def __add__(self, other):
        """Add other to self by incrementing self's coordinates by
        those of other.

        Notes
        =====

        >>> from sympy import Point

        When sequences of coordinates are passed to Point methods, they
        are converted to a Point internally. This __add__ method does
        not do that so if floating point values are used, a floating
        point result (in terms of SymPy Floats) will be returned.

        >>> Point(1, 2) + (.1, .2)
        Point2D(1.1, 2.2)

        If this is not desired, the `translate` method can be used or
        another Point can be added:

        >>> Point(1, 2).translate(.1, .2)
        Point2D(11/10, 11/5)
        >>> Point(1, 2) + Point(.1, .2)
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.translate

        """
        try:
            # 尝试将 self 和 other 规范化为 Point 对象，用于计算坐标的逐元素相加
            s, o = Point._normalize_dimension(self, Point(other, evaluate=False))
        except TypeError:
            # 如果类型错误，抛出 GeometryError 异常，说明无法将 other 添加到 Point 对象上
            raise GeometryError("Don't know how to add {} and a Point object".format(other))

        # 对坐标逐元素相加，并简化表达式
        coords = [simplify(a + b) for a, b in zip(s, o)]
        # 返回一个新的 Point 对象，其坐标为相加后的结果
        return Point(coords, evaluate=False)

    def __contains__(self, item):
        # 检查 item 是否在 self.args 中
        return item in self.args

    def __truediv__(self, divisor):
        """Divide point's coordinates by a factor."""
        # 将 divisor 转换为 Sympy 表达式
        divisor = sympify(divisor)
        # 对当前点的坐标逐元素进行除法运算，并简化结果
        coords = [simplify(x/divisor) for x in self.args]
        # 返回一个新的 Point 对象，其坐标为除法运算后的结果
        return Point(coords, evaluate=False)

    def __eq__(self, other):
        # 检查是否与另一个 Point 对象 other 相等，包括坐标维度和值
        if not isinstance(other, Point) or len(self.args) != len(other.args):
            return False
        return self.args == other.args

    def __getitem__(self, key):
        # 获取 self.args 中索引为 key 的元素
        return self.args[key]

    def __hash__(self):
        # 返回当前 Point 对象的哈希值
        return hash(self.args)

    def __iter__(self):
        # 返回一个迭代器，用于迭代当前 Point 对象的坐标
        return self.args.__iter__()

    def __len__(self):
        # 返回当前 Point 对象的坐标维度
        return len(self.args)

    def __mul__(self, factor):
        """Multiply point's coordinates by a factor.

        Notes
        =====

        >>> from sympy import Point

        When multiplying a Point by a floating point number,
        the coordinates of the Point will be changed to Floats:

        >>> Point(1, 2)*0.1
        Point2D(0.1, 0.2)

        If this is not desired, the `scale` method can be used or
        else only multiply or divide by integers:

        >>> Point(1, 2).scale(1.1, 1.1)
        Point2D(11/10, 11/5)
        >>> Point(1, 2)*11/10
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.scale
        """
        # 将 factor 转换为 Sympy 表达式
        factor = sympify(factor)
        # 对当前点的坐标逐元素进行乘法运算，并简化结果
        coords = [simplify(x*factor) for x in self.args]
        # 返回一个新的 Point 对象，其坐标为乘法运算后的结果
        return Point(coords, evaluate=False)

    def __rmul__(self, factor):
        """Multiply a factor by point's coordinates."""
        # 调用 __mul__ 方法实现反向乘法，即 factor * self
        return self.__mul__(factor)
    def __neg__(self):
        """Negate the point."""
        # 对点进行取反操作，即所有坐标取负值
        coords = [-x for x in self.args]
        return Point(coords, evaluate=False)

    def __sub__(self, other):
        """Subtract two points, or subtract a factor from this point's
        coordinates."""
        # 实现点的减法操作，支持点与点的减法，或者点与一个因子的减法
        return self + [-x for x in other]

    @classmethod
    def _normalize_dimension(cls, *points, **kwargs):
        """Ensure that points have the same dimension.
        By default `on_morph='warn'` is passed to the
        `Point` constructor."""
        # 如果存在类的环境维度属性，则使用它
        dim = getattr(cls, '_ambient_dimension', None)
        # 如果在参数中指定了维度，则覆盖默认值
        dim = kwargs.get('dim', dim)
        # 如果没有指定维度，则使用最高维度的点
        if dim is None:
            dim = max(i.ambient_dimension for i in points)
        # 如果所有点的维度都相同，则直接返回点的列表
        if all(i.ambient_dimension == dim for i in points):
            return list(points)
        # 否则，将所有点标准化到同一维度
        kwargs['dim'] = dim
        kwargs['on_morph'] = kwargs.get('on_morph', 'warn')
        return [Point(i, **kwargs) for i in points]

    @staticmethod
    def affine_rank(*args):
        """The affine rank of a set of points is the dimension
        of the smallest affine space containing all the points.
        For example, if the points lie on a line (and are not all
        the same) their affine rank is 1.  If the points lie on a plane
        but not a line, their affine rank is 2.  By convention, the empty
        set has affine rank -1."""

        if len(args) == 0:
            return -1
        # 确保参数是真正的点，并将每个点转换到原点
        points = Point._normalize_dimension(*[Point(i) for i in args])
        origin = points[0]
        # 将除了原点以外的每个点移到原点
        points = [i - origin for i in points[1:]]

        m = Matrix([i.args for i in points])
        # XXX 脆弱的实现方式 -- 有没有更好的方法？
        return m.rank(iszerofunc = lambda x:
            abs(x.n(2)) < 1e-12 if x.is_number else x.is_zero)

    @property
    def ambient_dimension(self):
        """Number of components this point has."""
        # 返回该点的组件数，即环境维度
        return getattr(self, '_ambient_dimension', len(self))

    @classmethod
    def are_coplanar(cls, *points):
        """Return True if there exists a plane in which all the points
        lie.  A trivial True value is returned if `len(points) < 3` or
        all Points are 2-dimensional.

        Parameters
        ==========

        A set of points

        Raises
        ======

        ValueError : if less than 3 unique points are given

        Returns
        =======

        boolean

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 2)
        >>> p2 = Point3D(2, 7, 2)
        >>> p3 = Point3D(0, 0, 2)
        >>> p4 = Point3D(1, 1, 2)
        >>> Point3D.are_coplanar(p1, p2, p3, p4)
        True
        >>> p5 = Point3D(0, 1, 3)
        >>> Point3D.are_coplanar(p1, p2, p3, p5)
        False

        """
        # 如果点的数量少于等于1个，直接返回True（即平面上至少需要3个点）
        if len(points) <= 1:
            return True

        # 将传入的点参数进行维度规范化处理，确保所有点的维度一致
        points = cls._normalize_dimension(*[Point(i) for i in points])
        # 如果第一个点是二维的，直接返回True（因为所有点都在一个平面上）
        if points[0].ambient_dimension == 2:
            return True
        # 对点列表进行去重处理
        points = list(uniq(points))
        # 使用 affine_rank 方法判断点集的仿射秩是否小于等于2，从而确定是否共面
        return Point.affine_rank(*points) <= 2

    def distance(self, other):
        """The Euclidean distance between self and another GeometricEntity.

        Returns
        =======

        distance : number or symbolic expression.

        Raises
        ======

        TypeError : if other is not recognized as a GeometricEntity or is a
                    GeometricEntity for which distance is not defined.

        See Also
        ========

        sympy.geometry.line.Segment.length
        sympy.geometry.point.Point.taxicab_distance

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> l = Line((3, 1), (2, 2))
        >>> p1.distance(p2)
        5
        >>> p1.distance(l)
        sqrt(2)

        The computed distance may be symbolic, too:

        >>> from sympy.abc import x, y
        >>> p3 = Point(x, y)
        >>> p3.distance((0, 0))
        sqrt(x**2 + y**2)

        """
        # 如果 other 不是 GeometricEntity 的实例，则尝试将其转换为 Point 对象
        if not isinstance(other, GeometryEntity):
            try:
                other = Point(other, dim=self.ambient_dimension)
            except TypeError:
                raise TypeError("not recognized as a GeometricEntity: %s" % type(other))
        # 如果 other 是 Point 类型，则对 self 和 other 进行维度规范化处理，并计算它们之间的欧几里得距离
        if isinstance(other, Point):
            s, p = Point._normalize_dimension(self, Point(other))
            return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
        # 否则，尝试调用 other 对象的 distance 方法
        distance = getattr(other, 'distance', None)
        if distance is None:
            raise TypeError("distance between Point and %s is not defined" % type(other))
        return distance(self)

    def dot(self, p):
        """Return dot product of self with another Point."""
        # 如果 p 不是序列类型，则将其转换为 Point 对象
        if not is_sequence(p):
            p = Point(p)  # raise the error via Point
        # 计算 self 和 p 之间的点积
        return Add(*(a*b for a, b in zip(self, p)))
    def equals(self, other):
        """Returns whether the coordinates of self and other agree."""
        # 检查是否为 Point 类型的对象，并且长度相同
        if not isinstance(other, Point) or len(self) != len(other):
            return False
        # 使用 zip 函数逐个比较对应位置的坐标是否相等
        return all(a.equals(b) for a, b in zip(self, other))

    def _eval_evalf(self, prec=15, **options):
        """Evaluate the coordinates of the point.

        This method will, where possible, create and return a new Point
        where the coordinates are evaluated as floating point numbers to
        the precision indicated (default=15).

        Parameters
        ==========

        prec : int
            Desired precision for floating point evaluation.

        Returns
        =======

        point : Point
            A new Point object with evaluated coordinates.

        Examples
        ========

        >>> from sympy import Point, Rational
        >>> p1 = Point(Rational(1, 2), Rational(3, 2))
        >>> p1
        Point2D(1/2, 3/2)
        >>> p1.evalf()
        Point2D(0.5, 1.5)

        """
        # 根据指定精度计算每个坐标的浮点数值
        dps = prec_to_dps(prec)
        coords = [x.evalf(n=dps, **options) for x in self.args]
        # 返回一个新的 Point 对象，其坐标已经被评估为浮点数
        return Point(*coords, evaluate=False)

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : GeometryEntity or sequence of coordinates
            The other entity or coordinates to intersect with.

        Returns
        =======

        intersection : list of Points
            List containing Points representing the intersection.

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point2D(0, 0)]

        """
        # 如果 other 不是 GeometryEntity 类型，则转换为 Point 对象
        if not isinstance(other, GeometryEntity):
            other = Point(other)
        # 如果 other 是 Point 类型
        if isinstance(other, Point):
            # 如果两个点相等，返回包含当前点的列表
            if self == other:
                return [self]
            # 根据维度规范化两个点的顺序
            p1, p2 = Point._normalize_dimension(self, other)
            # 如果规范化后的点仍然相同，返回包含当前点的列表
            if p1 == self and p1 == p2:
                return [self]
            # 否则返回空列表，表示没有交点
            return []
        # 对于其他 GeometryEntity 类型，调用其 intersection 方法
        return other.intersection(self)
    def is_collinear(self, *args):
        """判断点集中是否存在共线的点集合。

        返回 True 如果存在包含 self 和 points 的直线，否则返回 False。
        如果没有给定点，则返回一个显而易见的 True 值。

        Parameters
        ==========

        args : sequence of Points
            传入的点集合

        Returns
        =======

        is_collinear : boolean
            如果点集中的点共线，则返回 True；否则返回 False。

        See Also
        ========

        sympy.geometry.line.Line
            SymPy 中处理线的对象

        Examples
        ========

        >>> from sympy import Point
        >>> from sympy.abc import x
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> p3, p4, p5 = Point(2, 2), Point(x, x), Point(1, 2)
        >>> Point.is_collinear(p1, p2, p3, p4)
        True
        >>> Point.is_collinear(p1, p2, p3, p5)
        False

        """
        points = (self,) + args  # 将 self 和传入的其他点集合合并成一个点集
        points = Point._normalize_dimension(*[Point(i) for i in points])  # 标准化点集的维度
        points = list(uniq(points))  # 去除重复的点
        return Point.affine_rank(*points) <= 1  # 判断点集的仿射秩是否小于等于1，即是否共线

    def is_concyclic(self, *args):
        """判断点集是否共圆。

        如果点集中的点共圆则返回 True，否则返回 False。
        如果少于两个其他点，则返回一个显而易见的 True 值。

        Parameters
        ==========

        args : sequence of Points
            传入的点集合

        Returns
        =======

        is_concyclic : boolean
            如果点集中的点共圆，则返回 True；否则返回 False。

        Examples
        ========

        >>> from sympy import Point

        定义四个位于单位圆上的点：

        >>> p1, p2, p3, p4 = Point(1, 0), (0, 1), (-1, 0), (0, -1)

        >>> p1.is_concyclic() == p1.is_concyclic(p2, p3, p4) == True
        True

        定义一个不在该圆上的点：

        >>> p = Point(1, 1)

        >>> p.is_concyclic(p1, p2, p3)
        False

        """
        points = (self,) + args  # 将 self 和传入的其他点集合合并成一个点集
        points = Point._normalize_dimension(*[Point(i) for i in points])  # 标准化点集的维度
        points = list(uniq(points))  # 去除重复的点
        if not Point.affine_rank(*points) <= 2:
            return False  # 如果点集的仿射秩大于2，则不可能共圆，返回 False
        origin = points[0]
        points = [p - origin for p in points]  # 将所有点都以第一个点为原点平移
        # 如果点集共圆，则它们必须共面，并且存在一个点 c，使得所有的 ||p_i-c|| == ||p_j-c||。
        # 重新排列这个方程得到以下条件：矩阵 `mat` 的最后一列不应该有主元。
        mat = Matrix([list(i) + [i.dot(i)] for i in points])
        rref, pivots = mat.rref()
        if len(origin) not in pivots:
            return True  # 如果原点所在列不是主元列，则点集共圆，返回 True
        return False  # 否则返回 False

    @property
    def is_nonzero(self):
        """如果任何坐标不为零则返回 True，如果每个坐标都为零则返回 False，
        如果无法确定则返回 None。"""
        is_zero = self.is_zero
        if is_zero is None:
            return None  # 如果无法确定是否为零，则返回 None
        return not is_zero  # 如果确定不是全零，则返回 True；否则返回 False
    def is_scalar_multiple(self, p):
        """Returns whether each coordinate of `self` is a scalar
        multiple of the corresponding coordinate in point p.
        """
        # 将 self 和 p 规范化为同一维度的 Point 对象 s 和 o
        s, o = Point._normalize_dimension(self, Point(p))
        
        # 优化 2 维点的情况，因为这种情况经常发生
        if s.ambient_dimension == 2:
            # 分解 s 和 o 的坐标
            (x1, y1), (x2, y2) = s.args, o.args
            # 计算叉乘结果是否为零，以确定是否为标量倍数关系
            rv = (x1*y2 - x2*y1).equals(0)
            if rv is None:
                # 如果无法确定，则抛出 Undecidable 异常
                raise Undecidable(filldedent(
                    '''Cannot determine if %s is a scalar multiple of
                    %s''' % (s, o)))
        
        # 使用矩阵来判断是否线性相关，从而确定是否为标量倍数关系
        m = Matrix([s.args, o.args])
        return m.rank() < 2

    @property
    def is_zero(self):
        """True if every coordinate is zero, False if any coordinate is not zero,
        and None if it cannot be determined."""
        # 检查每个坐标是否为零
        nonzero = [x.is_nonzero for x in self.args]
        if any(nonzero):
            return False
        if any(x is None for x in nonzero):
            return None
        return True

    @property
    def length(self):
        """
        Treating a Point as a Line, this returns 0 for the length of a Point.

        Examples
        ========

        >>> from sympy import Point
        >>> p = Point(0, 1)
        >>> p.length
        0
        """
        # 点视作线，长度为 0
        return S.Zero

    def midpoint(self, p):
        """The midpoint between self and point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        midpoint : Point

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(13, 5)
        >>> p1.midpoint(p2)
        Point2D(7, 3)

        """
        # 计算 self 和 p 的中点
        s, p = Point._normalize_dimension(self, Point(p))
        return Point([simplify((a + b)*S.Half) for a, b in zip(s, p)])

    @property
    def origin(self):
        """A point of all zeros of the same ambient dimension
        as the current point"""
        # 返回与当前点具有相同维度的全零点
        return Point([0]*len(self), evaluate=False)
    def orthogonal_direction(self):
        """Returns a non-zero point that is orthogonal to the
        line containing `self` and the origin.

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2, 3)
        >>> a.orthogonal_direction
        Point3D(-2, 1, 0)
        >>> b = _
        >>> Line(b, b.origin).is_perpendicular(Line(a, a.origin))
        True
        """
        # 获取点的维度
        dim = self.ambient_dimension
        # 如果第一个坐标为零，返回一个具有1的坐标和其余为零的点
        if self[0].is_zero:
            return Point([1] + (dim - 1)*[0])
        # 如果第二个坐标为零，返回一个具有0和1的坐标以及其余为零的点
        if self[1].is_zero:
            return Point([0, 1] + (dim - 2)*[0])
        # 如果前两个坐标都不为零，创建一个非零的正交向量，通过交换它们、取负值和填充零
        return Point([-self[1], self[0]] + (dim - 2)*[0])

    @staticmethod
    def project(a, b):
        """Project the point `a` onto the line between the origin
        and point `b` along the normal direction.

        Parameters
        ==========

        a : Point
        b : Point

        Returns
        =======

        p : Point

        See Also
        ========

        sympy.geometry.line.LinearEntity.projection

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2)
        >>> b = Point(2, 5)
        >>> z = a.origin
        >>> p = Point.project(a, b)
        >>> Line(p, a).is_perpendicular(Line(p, b))
        True
        >>> Point.is_collinear(z, p, b)
        True
        """
        # 将点 `a` 和 `b` 统一为相同的维度
        a, b = Point._normalize_dimension(Point(a), Point(b))
        # 如果点 `b` 是零向量，抛出异常
        if b.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        # 返回点 `a` 在由原点和点 `b` 确定的直线上的投影点
        return b * (a.dot(b) / b.dot(b))

    def taxicab_distance(self, p):
        """The Taxicab Distance from self to point p.

        Returns the sum of the horizontal and vertical distances to point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        taxicab_distance : The sum of the horizontal
        and vertical distances to point p.

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> p1.taxicab_distance(p2)
        7

        """
        # 规范化点 `self` 和点 `p` 的维度
        s, p = Point._normalize_dimension(self, Point(p))
        # 返回点 `self` 到点 `p` 的曼哈顿距离（水平和垂直距离之和）
        return Add(*(abs(a - b) for a, b in zip(s, p)))
    # 计算从当前点 self 到点 p 的坎贝拉距离

    def canberra_distance(self, p):
        """The Canberra Distance from self to point p.

        Returns the weighted sum of horizontal and vertical distances to
        point p.

        Parameters
        ==========

        p : Point
            要计算距离的目标点

        Returns
        =======

        canberra_distance : float
            加权的水平和垂直距离到点 p 的和。权重是坐标绝对值的和。

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(3, 3)
        >>> p1.canberra_distance(p2)
        1
        >>> p1, p2 = Point(0, 0), Point(3, 3)
        >>> p1.canberra_distance(p2)
        2

        Raises
        ======

        ValueError
            当两个向量都为零时抛出异常。

        See Also
        ========

        sympy.geometry.point.Point.distance
            参考距离计算方法

        """

        # 标准化维度以便进行距离计算
        s, p = Point._normalize_dimension(self, Point(p))
        
        # 如果当前点和目标点都是零向量，则无法计算距离，抛出异常
        if self.is_zero and p.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        
        # 计算坎贝拉距离的加权和
        return Add(*((abs(a - b)/(abs(a) + abs(b))) for a, b in zip(s, p)))

    @property
    def unit(self):
        """Return the Point that is in the same direction as `self`
        and a distance of 1 from the origin"""
        
        # 返回与当前点方向相同且距离原点为1的单位向量
        return self / abs(self)
class Point2D(Point):
    """A point in a 2-dimensional Euclidean space.

    Parameters
    ==========

    coords
        A sequence of 2 coordinate values.

    Attributes
    ==========

    x
    y
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When trying to create a point with more than two dimensions.
        When `intersection` is called with object other than a Point.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy import Point2D
    >>> from sympy.abc import x
    >>> Point2D(1, 2)
    Point2D(1, 2)
    >>> Point2D([1, 2])
    Point2D(1, 2)
    >>> Point2D(0, x)
    Point2D(0, x)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point2D(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point2D(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    _ambient_dimension = 2  # 设置环境维度为2

    def __new__(cls, *args, _nocheck=False, **kwargs):
        # 如果不禁用检查
        if not _nocheck:
            # 设置关键字参数维度为2，并调用基类Point的构造函数
            kwargs['dim'] = 2
            args = Point(*args, **kwargs)
        # 使用GeometryEntity的构造函数创建新对象
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        # 检查是否包含给定项
        return item == self

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """
        # 返回边界框的坐标范围元组
        return (self.x, self.y, self.x, self.y)

    def rotate(self, angle, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        See Also
        ========

        translate, scale

        Examples
        ========

        >>> from sympy import Point2D, pi
        >>> t = Point2D(1, 0)
        >>> t.rotate(pi/2)
        Point2D(0, 1)
        >>> t.rotate(pi/2, (2, 0))
        Point2D(2, -1)

        """
        # 计算旋转角度的余弦和正弦值
        c = cos(angle)
        s = sin(angle)

        rv = self
        # 如果指定了旋转点pt
        if pt is not None:
            # 将pt转换为Point对象，并将rv向量与pt做减法
            pt = Point(pt, dim=2)
            rv -= pt
        x, y = rv.args
        # 应用旋转变换
        rv = Point(c*x - s*y, s*x + c*y)
        # 如果指定了旋转点pt，则将rv向量与pt做加法
        if pt is not None:
            rv += pt
        # 返回旋转后的Point对象
        return rv

    def scale(self, x=1, y=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        rotate, translate

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(1, 1)
        >>> t.scale(2)
        Point2D(2, 1)
        >>> t.scale(2, 2)
        Point2D(2, 2)

        """
        # 如果指定了缩放点pt
        if pt:
            # 将pt转换为Point对象，并对点进行平移和缩放
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        # 否则，直接缩放当前点的坐标
        return Point(self.x*x, self.y*y)
    def transform(self, matrix):
        """
        Return the point after applying the transformation described
        by the 3x3 Matrix, ``matrix``.

        See Also
        ========
        sympy.geometry.point.Point2D.rotate
        sympy.geometry.point.Point2D.scale
        sympy.geometry.point.Point2D.translate
        """
        # 检查输入的矩阵是否为3x3的矩阵
        if not (matrix.is_Matrix and matrix.shape == (3, 3)):
            raise ValueError("matrix must be a 3x3 matrix")
        # 提取当前点的坐标
        x, y = self.args
        # 应用矩阵变换到点上，并返回变换后的新点
        return Point(*(Matrix(1, 3, [x, y, 1])*matrix).tolist()[0][:2])

    def translate(self, x=0, y=0):
        """
        Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========
        sympy.geometry.point.Point2D.rotate, scale

        Examples
        ========
        >>> from sympy import Point2D
        >>> t = Point2D(0, 1)
        >>> t.translate(2)
        Point2D(2, 1)
        >>> t.translate(2, 2)
        Point2D(2, 3)
        >>> t + Point2D(2, 2)
        Point2D(2, 3)
        """
        # 返回经过平移后的新点
        return Point(self.x + x, self.y + y)

    @property
    def coordinates(self):
        """
        Returns the two coordinates of the Point.

        Examples
        ========
        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.coordinates
        (0, 1)
        """
        # 返回当前点的坐标元组
        return self.args

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========
        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.x
        0
        """
        # 返回当前点的 X 坐标
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========
        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.y
        1
        """
        # 返回当前点的 Y 坐标
        return self.args[1]
class Point3D(Point):
    """A point in a 3-dimensional Euclidean space.

    Parameters
    ==========

    coords
        A sequence of 3 coordinate values.

    Attributes
    ==========

    x
    y
    z
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When `intersection` is called with object other than a Point.

    Examples
    ========

    >>> from sympy import Point3D
    >>> from sympy.abc import x
    >>> Point3D(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point3D([1, 2, 3])
    Point3D(1, 2, 3)
    >>> Point3D(0, x, 3)
    Point3D(0, x, 3)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point3D(0.5, 0.25, 2)
    Point3D(1/2, 1/4, 2)
    >>> Point3D(0.5, 0.25, 3, evaluate=False)
    Point3D(0.5, 0.25, 3)

    """

    _ambient_dimension = 3

    def __new__(cls, *args, _nocheck=False, **kwargs):
        # 如果 _nocheck 为 False，则将维度设置为 3，然后调用 Point 的构造函数
        if not _nocheck:
            kwargs['dim'] = 3
            args = Point(*args, **kwargs)
        # 调用父类 GeometryEntity 的构造函数创建新的实例
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        # 判断 item 是否与当前点相等
        return item == self

    @staticmethod
    def are_collinear(*points):
        """Is a sequence of points collinear?

        Test whether or not a set of points are collinear. Returns True if
        the set of points are collinear, or False otherwise.

        Parameters
        ==========

        points : sequence of Point

        Returns
        =======

        are_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line3D

        Examples
        ========

        >>> from sympy import Point3D
        >>> from sympy.abc import x
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(1, 1, 1)
        >>> p3, p4, p5 = Point3D(2, 2, 2), Point3D(x, x, x), Point3D(1, 2, 6)
        >>> Point3D.are_collinear(p1, p2, p3, p4)
        True
        >>> Point3D.are_collinear(p1, p2, p3, p5)
        False
        """
        # 调用 Point 类的静态方法判断给定的点序列是否共线
        return Point.is_collinear(*points)

    def direction_cosine(self, point):
        """
        Gives the direction cosine between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_cosine(Point3D(2, 3, 5))
        [sqrt(6)/6, sqrt(6)/6, sqrt(6)/3]
        """
        # 计算当前点到给定点的方向余弦
        a = self.direction_ratio(point)
        b = sqrt(Add(*(i**2 for i in a)))
        return [(point.x - self.x) / b, (point.y - self.y) / b,
                (point.z - self.z) / b]
    def direction_ratio(self, point):
        """
        Gives the direction ratio between 2 points

        Parameters
        ==========

        point : Point3D
            The other point to compute direction ratio with.

        Returns
        =======

        list
            A list containing the direction ratios along x, y, and z axes.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_ratio(Point3D(2, 3, 5))
        [1, 1, 2]
        """
        # Compute direction ratios by subtracting coordinates of self from point
        return [(point.x - self.x),(point.y - self.y),(point.z - self.z)]

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : GeometryEntity or sequence of coordinates
            The other geometry entity or coordinates to find intersection with.

        Returns
        =======

        intersection : list of Points
            List containing the intersection points, which may be empty.

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point3D(0, 0, 0)]

        """
        # Check if other is not a GeometryEntity, convert to Point3D
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=3)
        # If other is a Point3D and equal to self, return self as intersection
        if isinstance(other, Point3D):
            if self == other:
                return [self]
            return []  # Otherwise return an empty list
        # Call intersection method of other with self as argument
        return other.intersection(self)

    def scale(self, x=1, y=1, z=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        translate

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(1, 1, 1)
        >>> t.scale(2)
        Point3D(2, 1, 1)
        >>> t.scale(2, 2)
        Point3D(2, 2, 1)

        """
        # If pt is provided, translate the point, scale, and then translate back
        if pt:
            pt = Point3D(pt)
            return self.translate(*(-pt).args).scale(x, y, z).translate(*pt.args)
        # Otherwise, directly scale the coordinates of the point
        return Point3D(self.x*x, self.y*y, self.z*z)

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 4x4 Matrix, ``matrix``.

        See Also
        ========
        sympy.geometry.point.Point3D.scale
        sympy.geometry.point.Point3D.translate
        """
        # Check if matrix is a 4x4 Matrix
        if not (matrix.is_Matrix and matrix.shape == (4, 4)):
            raise ValueError("matrix must be a 4x4 matrix")
        # Extract coordinates from self
        x, y, z = self.args
        # Transpose the matrix
        m = Transpose(matrix)
        # Apply transformation and return the transformed Point3D
        return Point3D(*(Matrix(1, 4, [x, y, z, 1])*m).tolist()[0][:3])
    def translate(self, x=0, y=0, z=0):
        """
        Shift the Point by adding x, y, and z to the coordinates of the Point.

        Parameters
        ==========
        x : int or float, optional
            Amount to shift in the x-direction (default is 0).
        y : int or float, optional
            Amount to shift in the y-direction (default is 0).
        z : int or float, optional
            Amount to shift in the z-direction (default is 0).

        Returns
        =======
        Point3D
            A new Point3D object shifted by x, y, and z.

        See Also
        ========
        scale

        Examples
        ========
        >>> from sympy import Point3D
        >>> t = Point3D(0, 1, 1)
        >>> t.translate(2)
        Point3D(2, 1, 1)
        >>> t.translate(2, 2)
        Point3D(2, 3, 1)
        >>> t + Point3D(2, 2, 2)
        Point3D(2, 3, 3)
        """
        return Point3D(self.x + x, self.y + y, self.z + z)

    @property
    def coordinates(self):
        """
        Returns the three coordinates of the Point as a tuple.

        Returns
        =======
        tuple
            A tuple containing the x, y, and z coordinates of the Point.

        Examples
        ========
        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.coordinates
        (0, 1, 2)
        """
        return self.args

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Returns
        =======
        int or float
            The X coordinate of the Point.

        Examples
        ========
        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 3)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Returns
        =======
        int or float
            The Y coordinate of the Point.

        Examples
        ========
        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.y
        1
        """
        return self.args[1]

    @property
    def z(self):
        """
        Returns the Z coordinate of the Point.

        Returns
        =======
        int or float
            The Z coordinate of the Point.

        Examples
        ========
        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 1)
        >>> p.z
        1
        """
        return self.args[2]
```