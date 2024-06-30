# `D:\src\scipysrc\sympy\sympy\vector\implicitregion.py`

```
# 从 sympy 库中导入 Rational 类，用于处理有理数
from sympy.core.numbers import Rational
# 从 sympy 库中导入 S 单例，表示全局的 S 符号
from sympy.core.singleton import S
# 从 sympy 库中导入 symbols 函数，用于创建符号变量
from sympy.core.symbol import symbols
# 从 sympy 库中导入 sign 函数，用于返回数值的符号
from sympy.functions.elementary.complexes import sign
# 从 sympy 库中导入 sqrt 函数，用于计算平方根
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy 库中导入 gcd 函数，用于计算多项式的最大公约数
from sympy.polys.polytools import gcd
# 从 sympy 库中导入 Complement 类，用于集合的补集操作
from sympy.sets.sets import Complement
# 从 sympy 库中导入 Basic, Tuple, diff, expand, Eq, Integer 类
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
# 从 sympy 库中导入 ordered 函数，用于排序对象
from sympy.core.sorting import ordered
# 从 sympy 库中导入 _symbol 函数，用于创建符号变量的内部表示
from sympy.core.symbol import _symbol
# 从 sympy 库中导入 solveset, nonlinsolve, diophantine 函数，用于求解方程和不定方程
from sympy.solvers import solveset, nonlinsolve, diophantine
# 从 sympy 库中导入 total_degree 函数，用于计算多项式的总次数
from sympy.polys import total_degree
# 从 sympy 库中导入 Point 类，用于表示几何空间中的点
from sympy.geometry import Point
# 从 sympy 库中导入 core 模块，包含与整数论相关的函数和类
from sympy.ntheory.factor_ import core

# 定义 ImplicitRegion 类，继承自 Basic 类
class ImplicitRegion(Basic):
    """
    Represents an implicit region in space.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z, t
    >>> from sympy.vector import ImplicitRegion

    >>> ImplicitRegion((x, y), x**2 + y**2 - 4)
    ImplicitRegion((x, y), x**2 + y**2 - 4)
    >>> ImplicitRegion((x, y), Eq(y*x, 1))
    ImplicitRegion((x, y), x*y - 1)

    >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
    >>> parabola.degree
    2
    >>> parabola.equation
    -4*x + y**2
    >>> parabola.rational_parametrization(t)
    (4/t**2, 4/t)

    >>> r = ImplicitRegion((x, y, z), Eq(z, x**2 + y**2))
    >>> r.variables
    (x, y, z)
    >>> r.singular_points()
    EmptySet
    >>> r.regular_point()
    (-10, -10, 200)

    Parameters
    ==========

    variables : tuple to map variables in implicit equation to base scalars.

    equation : An expression or Eq denoting the implicit equation of the region.

    """
    # 定义类构造函数 __new__
    def __new__(cls, variables, equation):
        # 如果 variables 不是 Tuple 类型，则转换为 Tuple 类型
        if not isinstance(variables, Tuple):
            variables = Tuple(*variables)

        # 如果 equation 是 Eq 类型的对象，则转换为等式的左侧减去右侧
        if isinstance(equation, Eq):
            equation = equation.lhs - equation.rhs

        # 调用父类 Basic 的构造函数，创建 ImplicitRegion 的实例
        return super().__new__(cls, variables, equation)

    # 返回 variables 属性，即表示变量的元组
    @property
    def variables(self):
        return self.args[0]

    # 返回 equation 属性，即表示隐式方程的表达式
    @property
    def equation(self):
        return self.args[1]

    # 返回隐式方程的总次数，由 sympy.polys.total_degree 函数计算
    @property
    def degree(self):
        return total_degree(self.equation)
    # 返回隐式区域上的一个点
    def regular_point(self):
        """
        Returns a point on the implicit region.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> circle = ImplicitRegion((x, y), (x + 2)**2 + (y - 3)**2 - 16)
        >>> circle.regular_point()
        (-2, -1)
        >>> parabola = ImplicitRegion((x, y), x**2 - 4*y)
        >>> parabola.regular_point()
        (0, 0)
        >>> r = ImplicitRegion((x, y, z), (x + y + z)**4)
        >>> r.regular_point()
        (-10, -10, 20)

        References
        ==========

        - Erik Hillgarter, "Rational Points on Conics", Diploma Thesis, RISC-Linz,
          J. Kepler Universitat Linz, 1996. Available:
          https://www3.risc.jku.at/publications/download/risc_1355/Rational%20Points%20on%20Conics.pdf

        """
        # 获取隐式方程
        equation = self.equation

        # 如果只有一个变量
        if len(self.variables) == 1:
            # 求解方程，返回结果的第一个元素作为元组
            return (list(solveset(equation, self.variables[0], domain=S.Reals))[0],)
        
        # 如果有两个变量
        elif len(self.variables) == 2:

            # 如果是二次方程
            if self.degree == 2:
                # 获取圆锥曲线的系数
                coeffs = a, b, c, d, e, f = conic_coeff(self.variables, equation)

                # 检查是否为抛物线
                if b**2 == 4*a*c:
                    # 计算抛物线的正则点
                    x_reg, y_reg = self._regular_point_parabola(*coeffs)
                else:
                    # 否则计算椭圆的正则点
                    x_reg, y_reg = self._regular_point_ellipse(*coeffs)
                return x_reg, y_reg
        
        # 如果有三个变量
        if len(self.variables) == 3:
            x, y, z = self.variables

            # 在指定范围内搜索合适的点
            for x_reg in range(-10, 10):
                for y_reg in range(-10, 10):
                    # 检查该点是否满足方程的条件
                    if not solveset(equation.subs({x: x_reg, y: y_reg}), self.variables[2], domain=S.Reals).is_empty:
                        # 返回满足条件的点及第三个变量的解
                        return (x_reg, y_reg, list(solveset(equation.subs({x: x_reg, y: y_reg})))[0])

        # 如果存在奇点
        if len(self.singular_points()) != 0:
            # 返回奇点列表的第一个元素
            return list[self.singular_points()][0]

        # 如果以上条件都不满足，则抛出未实现错误
        raise NotImplementedError()

    def _regular_point_parabola(self, a, b, c, d, e, f):
        # 检查是否存在有理点
        ok = (a, d) != (0, 0) and (c, e) != (0, 0) and b**2 == 4*a*c and (a, c) != (0, 0)

        # 如果不存在有理点，则抛出值错误
        if not ok:
            raise ValueError("Rational Point on the conic does not exist")

        # 如果 a 不为零
        if a != 0:
            # 计算参数变换后的值
            d_dash, f_dash = (4*a*e - 2*b*d, 4*a*f - d**2)
            if d_dash != 0:
                y_reg = -f_dash/d_dash
                x_reg = -(d + b*y_reg)/(2*a)
            else:
                ok = False
        # 如果 c 不为零
        elif c != 0:
            # 计算参数变换后的值
            d_dash, f_dash = (4*c*d - 2*b*e, 4*c*f - e**2)
            if d_dash != 0:
                x_reg = -f_dash/d_dash
                y_reg = -(e + b*x_reg)/(2*c)
            else:
                ok = False

        # 如果存在有理点，则返回计算结果
        if ok:
            return x_reg, y_reg
        else:
            # 否则抛出值错误
            raise ValueError("Rational Point on the conic does not exist")
    def singular_points(self):
        """
        返回区域的奇点集合。

        奇点是区域上所有偏导数都为零的点。

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y), (y-1)**2 -x**3 + 2*x**2 -x)
        >>> I.singular_points()
        {(1, 1)}

        """
        # 初始化方程列表，包含当前对象的方程
        eq_list = [self.equation]
        # 遍历所有变量，计算每个变量对方程的偏导数，并添加到方程列表中
        for var in self.variables:
            eq_list += [diff(self.equation, var)]

        # 使用非线性求解器求解方程组，返回奇点集合
        return nonlinsolve(eq_list, list(self.variables))

    def multiplicity(self, point):
        """
        返回区域上奇点的重数。

        区域的奇点 (x,y) 的重数为 m，如果在该点所有到 m-1 阶的偏导数都为零。

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y, z), x**2 + y**3 - z**4)
        >>> I.singular_points()
        {(0, 0, 0)}
        >>> I.multiplicity((0, 0, 0))
        2

        """
        # 如果 point 是 Point 对象，则获取其参数
        if isinstance(point, Point):
            point = point.args

        # 修改方程以考虑给定点的偏移
        modified_eq = self.equation
        for i, var in enumerate(self.variables):
            modified_eq = modified_eq.subs(var, var + point[i])
        modified_eq = expand(modified_eq)

        # 如果方程包含多个项，则计算最小总次数的项作为重数 m
        if len(modified_eq.args) != 0:
            terms = modified_eq.args
            m = min(total_degree(term) for term in terms)
        else:
            terms = modified_eq
            m = total_degree(terms)

        # 返回奇点的重数
        return m
# 计算二次曲线方程的系数
def conic_coeff(variables, equation):
    # 检查方程的总次数是否为2，如果不是则引发值错误异常
    if total_degree(equation) != 2:
        raise ValueError()
    
    # 提取变量 x 和 y
    x = variables[0]
    y = variables[1]

    # 将方程式进行展开
    equation = expand(equation)
    
    # 提取二次项系数
    a = equation.coeff(x**2)
    
    # 提取混合项系数
    b = equation.coeff(x*y)
    
    # 提取二次项系数
    c = equation.coeff(y**2)
    
    # 提取一次项系数 d，其为 x 的系数并且 y 的系数为 0
    d = equation.coeff(x, 1).coeff(y, 0)
    
    # 提取一次项系数 e，其为 y 的系数并且 x 的系数为 0
    e = equation.coeff(y, 1).coeff(x, 0)
    
    # 提取常数项系数
    f = equation.coeff(x, 0).coeff(y, 0)
    
    # 返回计算得到的系数 a, b, c, d, e, f
    return a, b, c, d, e, f
```