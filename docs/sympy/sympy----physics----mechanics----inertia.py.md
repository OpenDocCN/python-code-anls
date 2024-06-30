# `D:\src\scipysrc\sympy\sympy\physics\mechanics\inertia.py`

```
# 导入 sympify 函数，用于将输入值转换为 Sympy 符号表达式
from sympy import sympify
# 导入 Point、Dyadic、ReferenceFrame 和 outer 函数
from sympy.physics.vector import Point, Dyadic, ReferenceFrame, outer
# 导入 namedtuple 类型
from collections import namedtuple

# 定义模块中公开的类和函数名列表
__all__ = ['inertia', 'inertia_of_point_mass', 'Inertia']


def inertia(frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0):
    """Simple way to create inertia Dyadic object.

    Explanation
    ===========

    Creates an inertia Dyadic based on the given tensor values and a body-fixed
    reference frame.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame the inertia is defined in.
    ixx : Sympifyable
        The xx element in the inertia dyadic.
    iyy : Sympifyable
        The yy element in the inertia dyadic.
    izz : Sympifyable
        The zz element in the inertia dyadic.
    ixy : Sympifyable
        The xy element in the inertia dyadic.
    iyz : Sympifyable
        The yz element in the inertia dyadic.
    izx : Sympifyable
        The zx element in the inertia dyadic.

    Examples
    ========

    >>> from sympy.physics.mechanics import ReferenceFrame, inertia
    >>> N = ReferenceFrame('N')
    >>> inertia(N, 1, 2, 3)
    (N.x|N.x) + 2*(N.y|N.y) + 3*(N.z|N.z)

    """

    # 检查 frame 是否为 ReferenceFrame 类型，否则抛出类型错误
    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Need to define the inertia in a frame')
    # 使用 sympify 将输入的值转换为符号表达式
    ixx, iyy, izz = sympify(ixx), sympify(iyy), sympify(izz)
    ixy, iyz, izx = sympify(ixy), sympify(iyz), sympify(izx)
    # 构建惯性 Dyadic 对象并返回
    return (ixx*outer(frame.x, frame.x) + ixy*outer(frame.x, frame.y) +
            izx*outer(frame.x, frame.z) + ixy*outer(frame.y, frame.x) +
            iyy*outer(frame.y, frame.y) + iyz*outer(frame.y, frame.z) +
            izx*outer(frame.z, frame.x) + iyz*outer(frame.z, frame.y) +
            izz*outer(frame.z, frame.z))


def inertia_of_point_mass(mass, pos_vec, frame):
    """Inertia dyadic of a point mass relative to point O.

    Parameters
    ==========

    mass : Sympifyable
        Mass of the point mass
    pos_vec : Vector
        Position from point O to point mass
    frame : ReferenceFrame
        Reference frame to express the dyadic in

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import ReferenceFrame, inertia_of_point_mass
    >>> N = ReferenceFrame('N')
    >>> r, m = symbols('r m')
    >>> px = r * N.x
    >>> inertia_of_point_mass(m, px, N)
    m*r**2*(N.y|N.y) + m*r**2*(N.z|N.z)

    """

    # 计算相对于点 O 的点质量的惯性 Dyadic
    return mass*(
        (outer(frame.x, frame.x) +
         outer(frame.y, frame.y) +
         outer(frame.z, frame.z)) *
        (pos_vec.dot(pos_vec)) - outer(pos_vec, pos_vec))


class Inertia(namedtuple('Inertia', ['dyadic', 'point'])):
    """Inertia object consisting of a Dyadic and a Point of reference.

    Explanation
    ===========

    This is a simple class to store the Point and Dyadic, belonging to an
    inertia.

    Attributes
    ==========

    dyadic : Dyadic
        The dyadic of the inertia.
    point : Point
        The reference point of the inertia.

    Examples
    ========

    """
    >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
    # 导入 Sympy 中的力学模块，包括参考系、点和惯性对象的定义

    >>> N = ReferenceFrame('N')
    # 创建一个名为 N 的参考系对象

    >>> Po = Point('Po')
    # 创建一个名为 Po 的点对象

    >>> Inertia(N.x.outer(N.x) + N.y.outer(N.y) + N.z.outer(N.z), Po)
    # 创建一个惯性对象，使用给定的 dyadic 和参考点 Po

    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    In the example above the Dyadic was created manually, one can however also
    use the ``inertia`` function for this or the class method ``from_tensor`` as
    shown below.

    >>> Inertia.from_inertia_scalars(Po, N, 1, 1, 1)
    # 使用标量张量值快速创建一个惯性对象，传入参考点 Po、参考系 N 和对应的惯性张量值

    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    """
    def __new__(cls, dyadic, point):
        # 创建新的实例方法，用于初始化惯性对象

        # Switch order if given in the wrong order
        if isinstance(dyadic, Point) and isinstance(point, Dyadic):
            # 如果参数顺序错误，交换它们的顺序
            point, dyadic = dyadic, point
        if not isinstance(point, Point):
            # 如果参考点不是 Point 类型，则引发类型错误异常
            raise TypeError('Reference point should be of type Point')
        if not isinstance(dyadic, Dyadic):
            # 如果惯性张量不是 Dyadic 类型，则引发类型错误异常
            raise TypeError('Inertia value should be expressed as a Dyadic')
        return super().__new__(cls, dyadic, point)

    @classmethod
    def from_inertia_scalars(cls, point, frame, ixx, iyy, izz, ixy=0, iyz=0,
                             izx=0):
        """Simple way to create an Inertia object based on the tensor values.

        Explanation
        ===========

        This class method uses the :func`~.inertia` to create the Dyadic based
        on the tensor values.

        Parameters
        ==========

        point : Point
            The reference point of the inertia.
        frame : ReferenceFrame
            The frame the inertia is defined in.
        ixx : Sympifyable
            The xx element in the inertia dyadic.
        iyy : Sympifyable
            The yy element in the inertia dyadic.
        izz : Sympifyable
            The zz element in the inertia dyadic.
        ixy : Sympifyable
            The xy element in the inertia dyadic.
        iyz : Sympifyable
            The yz element in the inertia dyadic.
        izx : Sympifyable
            The zx element in the inertia dyadic.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
        >>> ixx, iyy, izz, ixy, iyz, izx = symbols('ixx iyy izz ixy iyz izx')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> I = Inertia.from_inertia_scalars(P, N, ixx, iyy, izz, ixy, iyz, izx)

        The tensor values can easily be seen when converting the dyadic to a
        matrix.

        >>> I.dyadic.to_matrix(N)
        Matrix([
        [ixx, ixy, izx],
        [ixy, iyy, iyz],
        [izx, iyz, izz]])

        """
        return cls(inertia(frame, ixx, iyy, izz, ixy, iyz, izx), point)

    def __add__(self, other):
        # 重载加法操作，禁止 Inertia 对象之间的相加操作，抛出类型错误异常
        raise TypeError(f"unsupported operand type(s) for +: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")
    # 定义对象的乘法运算符重载方法
    def __mul__(self, other):
        # 抛出类型错误异常，说明不支持该操作数类型的乘法运算
        raise TypeError(f"unsupported operand type(s) for *: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    # 定义对象的右加法运算符重载方法，与加法运算符方法相同
    __radd__ = __add__

    # 定义对象的右乘法运算符重载方法，与乘法运算符方法相同
    __rmul__ = __mul__
```