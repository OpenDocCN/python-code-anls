# `D:\src\scipysrc\sympy\sympy\geometry\curve.py`

```
# 导入所需的符号和函数
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import diff
from sympy.core.containers import Tuple
from sympy.core.symbol import _symbol
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point
from sympy.integrals import integrate
from sympy.matrices import Matrix, rot_axis3
from sympy.utilities.iterables import is_sequence

# 导入特定于多精度数学计算的函数
from mpmath.libmp.libmpf import prec_to_dps


class Curve(GeometrySet):
    """2维欧几里德空间中的曲线。

    包含
    ========
    Curve

    """

    def __new__(cls, function, limits):
        # 检查函数参数是否正确，并要求为二元组
        if not is_sequence(function) or len(function) != 2:
            raise ValueError("Function argument should be (x(t), y(t)) "
                "but got %s" % str(function))
        # 检查限制参数是否正确，并要求为三元组
        if not is_sequence(limits) or len(limits) != 3:
            raise ValueError("Limit argument should be (t, tmin, tmax) "
                "but got %s" % str(limits))

        # 创建 GeometryEntity 的新实例
        return GeometryEntity.__new__(cls, Tuple(*function), Tuple(*limits))

    def __call__(self, f):
        # 将参数替换为指定的值并返回结果
        return self.subs(self.parameter, f)

    def _eval_subs(self, old, new):
        # 如果替换的是参数，返回替换后的 Point 对象
        if old == self.parameter:
            return Point(*[f.subs(old, new) for f in self.functions])

    def _eval_evalf(self, prec=15, **options):
        # 获取参数和限制
        f, (t, a, b) = self.args
        # 将精度转换为小数点位数
        dps = prec_to_dps(prec)
        # 对函数和限制进行数值评估
        f = tuple([i.evalf(n=dps, **options) for i in f])
        a, b = [i.evalf(n=dps, **options) for i in (a, b)]
        # 返回更新后的曲线对象
        return self.func(f, (t, a, b))
    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the curve.

        Parameters
        ==========

        parameter : str or Symbol, optional
            Default value is 't'.
            The Curve's parameter is selected with None or self.parameter
            otherwise the provided symbol is used.

        Returns
        =======

        Point :
            Returns a point in parametric form.

        Raises
        ======

        ValueError
            When `parameter` already appears in the functions.

        Examples
        ========

        >>> from sympy import Curve, Symbol
        >>> from sympy.abc import s
        >>> C = Curve([2*s, s**2], (s, 0, 2))
        >>> C.arbitrary_point()
        Point2D(2*t, t**2)
        >>> C.arbitrary_point(C.parameter)
        Point2D(2*s, s**2)
        >>> C.arbitrary_point(None)
        Point2D(2*s, s**2)
        >>> C.arbitrary_point(Symbol('a'))
        Point2D(2*a, a**2)

        See Also
        ========

        sympy.geometry.point.Point

        """
        # 如果参数为 None，则返回曲线函数中的参数点
        if parameter is None:
            return Point(*self.functions)
        
        # 将提供的参数转换为符号对象
        tnew = _symbol(parameter, self.parameter, real=True)
        t = self.parameter
        # 如果新参数的名称与现有自由符号中的任何名称冲突，则引发 ValueError
        if (tnew.name != t.name and
                tnew.name in (f.name for f in self.free_symbols)):
            raise ValueError('Symbol %s already appears in object '
                'and cannot be used as a parameter.' % tnew.name)
        # 返回曲线在新参数下的点
        return Point(*[w.subs(t, tnew) for w in self.functions])

    @property
    def free_symbols(self):
        """Return a set of symbols other than the bound symbols used to
        parametrically define the Curve.

        Returns
        =======

        set :
            Set of all non-parameterized symbols.

        Examples
        ========

        >>> from sympy.abc import t, a
        >>> from sympy import Curve
        >>> Curve((t, t**2), (t, 0, 2)).free_symbols
        set()
        >>> Curve((t, t**2), (t, a, 2)).free_symbols
        {a}

        """
        # 初始化一个空集合，用于存放非参数化符号
        free = set()
        # 将曲线函数以及限制范围中的每个元素的自由符号加入集合
        for a in self.functions + self.limits[1:]:
            free |= a.free_symbols
        # 从自由符号集合中排除当前参数
        free = free.difference({self.parameter})
        return free

    @property
    def ambient_dimension(self):
        """The dimension of the curve.

        Returns
        =======

        int :
            the dimension of curve.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve((t, t**2), (t, 0, 2))
        >>> C.ambient_dimension
        2

        """

        # 返回曲线函数的长度，即曲线的维度
        return len(self.args[0])
    # 定义一个方法 functions，用于返回参数化坐标函数列表
    def functions(self):
        """The functions specifying the curve.

        Returns
        =======

        functions :
            list of parameterized coordinate functions.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve((t, t**2), (t, 0, 2))
        >>> C.functions
        (t, t**2)

        See Also
        ========

        parameter

        """
        return self.args[0]

    @property
    # 定义属性 limits，返回曲线的参数限制元组
    def limits(self):
        """The limits for the curve.

        Returns
        =======

        limits : tuple
            Contains parameter and lower and upper limits.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve([t, t**3], (t, -2, 2))
        >>> C.limits
        (t, -2, 2)

        See Also
        ========

        plot_interval

        """
        return self.args[1]

    @property
    # 定义属性 parameter，返回曲线函数的变量符号
    def parameter(self):
        """The curve function variable.

        Returns
        =======

        Symbol :
            returns a bound symbol.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve([t, t**2], (t, 0, 2))
        >>> C.parameter
        t

        See Also
        ========

        functions

        """
        return self.args[1][0]

    @property
    # 定义属性 length，返回曲线的长度
    def length(self):
        """The curve length.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import t
        >>> Curve((t, t), (t, 0, 1)).length
        sqrt(2)

        """
        # 计算曲线的积分被积函数，是曲线函数在参数范围内的导数的平方和开方
        integrand = sqrt(sum(diff(func, self.limits[0])**2 for func in self.functions))
        # 返回曲线的长度，即被积函数在参数范围内的积分结果
        return integrate(integrand, self.limits)

    # 定义一个方法 plot_interval，返回默认几何绘图曲线的绘图区间
    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the curve.

        Parameters
        ==========

        parameter : str or Symbol, optional
            Default value is 't';
            otherwise the provided symbol is used.

        Returns
        =======

        List :
            the plot interval as below:
                [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Curve, sin
        >>> from sympy.abc import x, s
        >>> Curve((x, sin(x)), (x, 1, 2)).plot_interval()
        [t, 1, 2]
        >>> Curve((x, sin(x)), (x, 1, 2)).plot_interval(s)
        [s, 1, 2]

        See Also
        ========

        limits : Returns limits of the parameter interval

        """
        # 将参数转换为符号对象，用于绘图区间的标识
        t = _symbol(parameter, self.parameter, real=True)
        # 返回绘图区间，包括参数符号及其上下界
        return [t] + list(self.limits[1:])
    def rotate(self, angle=0, pt=None):
        """Rotate the curve around a specified point or the origin by a given angle.

        Parameters
        ==========

        angle :
            Angle of rotation in radians. Default is 0.

        pt : Point
            Point around which the curve will be rotated. If None, rotation occurs around the origin.

        Returns
        =======

        Curve :
            Returns the rotated curve.

        Examples
        ========

        >>> from sympy import Curve, pi
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).rotate(pi/2)
        Curve((-x, x), (x, 0, 1))

        """
        # If pt is provided, create a Point object and negate it
        if pt:
            pt = -Point(pt, dim=2)
        else:
            # Otherwise, use the origin (0, 0) as the rotation point
            pt = Point(0,0)
        # Translate the curve by the negative of pt's arguments
        rv = self.translate(*pt.args)
        # Get the list of functions defining the curve
        f = list(rv.functions)
        # Append 0 to the list of functions (to adjust matrix dimensions)
        f.append(0)
        # Convert the list of functions to a Matrix with 1 row and 3 columns
        f = Matrix(1, 3, f)
        # Rotate the curve using rot_axis3 function by angle
        f *= rot_axis3(angle)
        # Reconstruct the curve with the rotated coordinates
        rv = self.func(f[0, :2].tolist()[0], self.limits)
        # Translate the curve back by the negative of pt's arguments
        pt = -pt
        return rv.translate(*pt.args)

    def scale(self, x=1, y=1, pt=None):
        """Scale the curve by the given factors optionally around a specified point.

        Returns
        =======

        Curve :
            Returns the scaled curve.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).scale(2)
        Curve((2*x, x), (x, 0, 1))

        """
        # If pt is provided, create a Point object and negate it
        if pt:
            pt = Point(pt, dim=2)
            # Translate the curve by the negative of pt's arguments,
            # then scale it by x and y factors, and translate it back by pt's arguments
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        # Otherwise, directly scale the curve by x and y factors
        fx, fy = self.functions
        return self.func((fx*x, fy*y), self.limits)

    def translate(self, x=0, y=0):
        """Translate the Curve by (x, y).

        Returns
        =======

        Curve :
            Returns a translated curve.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).translate(1, 2)
        Curve((x + 1, x + 2), (x, 0, 1))

        """
        # Get the functions defining the curve
        fx, fy = self.functions
        # Translate the curve by adding x to fx and y to fy
        return self.func((fx + x, fy + y), self.limits)
```