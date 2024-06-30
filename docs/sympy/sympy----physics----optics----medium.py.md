# `D:\src\scipysrc\sympy\sympy\physics\optics\medium.py`

```
"""
**Contains**

* Medium
"""
# 导入必要的模块和符号
from sympy.physics.units import second, meter, kilogram, ampere
# 导入所有的Medium类
__all__ = ['Medium']

# 导入必要的SymPy类和函数
from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import speed_of_light, u0, e0

# 将光速c转换为米每秒
c = speed_of_light.convert_to(meter/second)
# 将真空电容率e0转换为安培的平方秒的平方毫千克每立方米
_e0mksa = e0.convert_to(ampere**2*second**4/(kilogram*meter**3))
# 将真空磁导率u0转换为米毫千克每安培的平方秒的平方
_u0mksa = u0.convert_to(meter*kilogram/(ampere**2*second**2))

# 定义光学介质类Medium，继承自Basic类
class Medium(Basic):

    """
    This class represents an optical medium. The prime reason to implement this is
    to facilitate refraction, Fermat's principle, etc.

    Explanation
    ===========

    An optical medium is a material through which electromagnetic waves propagate.
    The permittivity and permeability of the medium define how electromagnetic
    waves propagate in it.


    Parameters
    ==========

    name: string
        The display name of the Medium.

    permittivity: Sympifyable
        Electric permittivity of the space.

    permeability: Sympifyable
        Magnetic permeability of the space.

    n: Sympifyable
        Index of refraction of the medium.


    Examples
    ========

    >>> from sympy.abc import epsilon, mu
    >>> from sympy.physics.optics import Medium
    >>> m1 = Medium('m1')
    >>> m2 = Medium('m2', epsilon, mu)
    >>> m1.intrinsic_impedance
    149896229*pi*kilogram*meter**2/(1250000*ampere**2*second**3)
    >>> m2.refractive_index
    299792458*meter*sqrt(epsilon*mu)/second


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Optical_medium

    """
    # 定义类的构造方法，初始化介质的名称及其特性（介电常数、磁导率、折射率）
    def __new__(cls, name, permittivity=None, permeability=None, n=None):
        # 如果名称不是字符串类型，则转换为字符串
        if not isinstance(name, Str):
            name = Str(name)

        # 将介电常数、磁导率和折射率符号化（如果它们不是None的话）
        permittivity = _sympify(permittivity) if permittivity is not None else permittivity
        permeability = _sympify(permeability) if permeability is not None else permeability
        n = _sympify(n) if n is not None else n

        # 如果指定了折射率n
        if n is not None:
            # 如果指定了介电常数而未指定磁导率
            if permittivity is not None and permeability is None:
                # 根据折射率和介电常数计算磁导率
                permeability = n**2/(c**2*permittivity)
                # 返回介质对象MediumPP
                return MediumPP(name, permittivity, permeability)
            # 如果指定了磁导率而未指定介电常数
            elif permeability is not None and permittivity is None:
                # 根据折射率和磁导率计算介电常数
                permittivity = n**2/(c**2*permeability)
                # 返回介质对象MediumPP
                return MediumPP(name, permittivity, permeability)
            # 如果同时指定了介电常数和磁导率
            elif permittivity is not None and permittivity is not None:
                # 抛出异常，不允许同时指定介电常数、磁导率和折射率
                raise ValueError("Specifying all of permittivity, permeability, and n is not allowed")
            else:
                # 否则，返回介质对象MediumN，只包含名称和折射率
                return MediumN(name, n)
        # 如果未指定折射率n
        elif permittivity is not None and permeability is not None:
            # 返回介质对象MediumPP，包含名称、介电常数和磁导率
            return MediumPP(name, permittivity, permeability)
        # 如果介电常数和磁导率均未指定
        elif permittivity is None and permeability is None:
            # 返回介质对象MediumPP，使用标准的介电常数_e0mksa和磁导率_u0mksa
            return MediumPP(name, _e0mksa, _u0mksa)
        else:
            # 否则，抛出异常，参数未充分指定。应指定折射率n或者任意两个介电常数、磁导率和折射率
            raise ValueError("Arguments are underspecified. Either specify n or any two of permittivity, "
                             "permeability, and n")

    @property
    def name(self):
        # 返回介质对象的名称
        return self.args[0]

    @property
    def speed(self):
        """
        返回介质中电磁波的传播速度。

        示例
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.speed
        299792458*meter/second
        >>> m2 = Medium('m2', n=1)
        >>> m.speed == m2.speed
        True

        """
        # 返回光速c除以介质的折射率n，计算电磁波的传播速度
        return c / self.n

    @property
    def refractive_index(self):
        """
        返回介质的折射率。

        示例
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.refractive_index
        1

        """
        # 返回光速c除以介质的传播速度，计算介质的折射率
        return (c/self.speed)
class MediumN(Medium):

    """
    Represents an optical medium for which only the refractive index is known.
    Useful for simple ray optics.

    This class should never be instantiated directly.
    Instead it should be instantiated indirectly by instantiating Medium with
    only n specified.

    Examples
    ========
    >>> from sympy.physics.optics import Medium
    >>> m = Medium('m', n=2)
    >>> m
    MediumN(Str('m'), 2)
    """

    def __new__(cls, name, n):
        # 调用父类的 __new__ 方法来创建 MediumN 对象
        obj = super(Medium, cls).__new__(cls, name, n)
        return obj


class MediumPP(Medium):
    """
    Represents an optical medium for which the permittivity and permeability are known.

    This class should never be instantiated directly. Instead it should be
    instantiated indirectly by instantiating Medium with any two of
    permittivity, permeability, and n specified, or by not specifying any
    of permittivity, permeability, or n, in which case default values for
    permittivity and permeability will be used.

    Examples
    ========
    >>> from sympy.physics.optics import Medium
    >>> from sympy.abc import epsilon, mu
    >>> m1 = Medium('m1', permittivity=epsilon, permeability=mu)
    >>> m1
    MediumPP(Str('m1'), epsilon, mu)
    >>> m2 = Medium('m2')
    >>> m2
    MediumPP(Str('m2'), 625000*ampere**2*second**4/(22468879468420441*pi*kilogram*meter**3), pi*kilogram*meter/(2500000*ampere**2*second**2))
    """

    def __new__(cls, name, permittivity, permeability):
        # 调用父类的 __new__ 方法来创建 MediumPP 对象
        obj = super(Medium, cls).__new__(cls, name, permittivity, permeability)
        return obj

    @property
    def intrinsic_impedance(self):
        """
        Returns intrinsic impedance of the medium.

        Explanation
        ===========

        The intrinsic impedance of a medium is the ratio of the
        transverse components of the electric and magnetic fields
        of the electromagnetic wave travelling in the medium.
        In a region with no electrical conductivity it simplifies
        to the square root of ratio of magnetic permeability to
        electric permittivity.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.intrinsic_impedance
        149896229*pi*kilogram*meter**2/(1250000*ampere**2*second**3)

        """
        # 返回介质的固有阻抗，即磁导率与电容率的比值的平方根
        return sqrt(self.permeability / self.permittivity)

    @property
    def permittivity(self):
        """
        Returns electric permittivity of the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.permittivity
        625000*ampere**2*second**4/(22468879468420441*pi*kilogram*meter**3)

        """
        # 返回介质的电容率
        return self.args[1]

    @property
    def permeability(self):
        """
        返回介质的磁导率。

        示例
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.permeability
        pi*kilogram*meter/(2500000*ampere**2*second**2)

        """
        # 返回当前对象的第三个参数，即介质的磁导率
        return self.args[2]

    @property
    def n(self):
        # 返回光速 c 乘以介质的电容率和磁导率的乘积的平方根
        return c*sqrt(self.permittivity*self.permeability)
```