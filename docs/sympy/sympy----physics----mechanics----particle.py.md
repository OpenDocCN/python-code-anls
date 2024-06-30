# `D:\src\scipysrc\sympy\sympy\physics\mechanics\particle.py`

```
# 导入 sympy 库中的 S 对象，用于处理符号表达式
from sympy import S
# 从 sympy.physics.vector 中导入交叉乘积 cross 和点乘 dot 函数
from sympy.physics.vector import cross, dot
# 从 sympy.physics.mechanics.body_base 中导入 BodyBase 类
from sympy.physics.mechanics.body_base import BodyBase
# 从 sympy.physics.mechanics.inertia 中导入 inertia_of_point_mass 函数
from sympy.physics.mechanics.inertia import inertia_of_point_mass
# 从 sympy.utilities.exceptions 中导入 sympy_deprecation_warning 函数
from sympy.utilities.exceptions import sympy_deprecation_warning

# 定义模块中公开的内容
__all__ = ['Particle']

# 定义粒子类 Particle，继承自 BodyBase 类
class Particle(BodyBase):
    """A particle.

    Explanation
    ===========

    Particles have a non-zero mass and lack spatial extension; they take up no
    space.

    Values need to be supplied on initialization, but can be changed later.

    Parameters
    ==========

    name : str
        Name of particle
    point : Point
        A physics/mechanics Point which represents the position, velocity, and
        acceleration of this Particle
    mass : Sympifyable
        A SymPy expression representing the Particle's mass
    potential_energy : Sympifyable
        The potential energy of the Particle.

    Examples
    ========

    >>> from sympy.physics.mechanics import Particle, Point
    >>> from sympy import Symbol
    >>> po = Point('po')
    >>> m = Symbol('m')
    >>> pa = Particle('pa', po, m)
    >>> # Or you could change these later
    >>> pa.mass = m
    >>> pa.point = po

    """

    # 类属性 point 被设置为 BodyBase 类的质心属性
    point = BodyBase.masscenter

    # 构造函数，初始化粒子的名称、位置点和质量
    def __init__(self, name, point=None, mass=None):
        # 调用父类 BodyBase 的构造函数
        super().__init__(name, point, mass)

    # 计算粒子在给定参考系中的线动量
    def linear_momentum(self, frame):
        """Linear momentum of the particle.

        Explanation
        ===========

        The linear momentum L, of a particle P, with respect to frame N is
        given by:

        L = m * v

        where m is the mass of the particle, and v is the velocity of the
        particle in the frame N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v = dynamicsymbols('m v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> A = Particle('A', P, m)
        >>> P.set_vel(N, v * N.x)
        >>> A.linear_momentum(N)
        m*v*N.x

        """

        # 返回粒子的质量乘以其在给定参考系中的速度
        return self.mass * self.point.vel(frame)
    def angular_momentum(self, point, frame):
        """Angular momentum of the particle about the point.

        Explanation
        ===========

        The angular momentum H, about some point O of a particle, P, is given
        by:

        ``H = cross(r, m * v)``

        where r is the position vector from point O to the particle P, m is
        the mass of the particle, and v is the velocity of the particle in
        the inertial frame, N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum of the particle is desired.

        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r = dynamicsymbols('m v r')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> A = O.locatenew('A', r * N.x)
        >>> P = Particle('P', A, m)
        >>> P.point.set_vel(N, v * N.y)
        >>> P.angular_momentum(O, N)
        m*r*v*N.z

        """

        # 计算并返回关于给定点的粒子的角动量
        return cross(self.point.pos_from(point),
                     self.mass * self.point.vel(frame))

    def kinetic_energy(self, frame):
        """Kinetic energy of the particle.

        Explanation
        ===========

        The kinetic energy, T, of a particle, P, is given by:

        ``T = 1/2 (dot(m * v, v))``

        where m is the mass of particle P, and v is the velocity of the
        particle in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The Particle's velocity is typically defined with respect to
            an inertial frame but any relevant frame in which the velocity is
            known can be supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy import symbols
        >>> m, v, r = symbols('m v r')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> P = Particle('P', O, m)
        >>> P.point.set_vel(N, v * N.y)
        >>> P.kinetic_energy(N)
        m*v**2/2

        """

        # 计算并返回粒子的动能
        return S.Half * self.mass * dot(self.point.vel(frame),
                                        self.point.vel(frame))

    def set_potential_energy(self, scalar):
        sympy_deprecation_warning(
            """
        This function is deprecated and will be removed in future versions
        of SymPy. Please use the `potential_energy` attribute instead.
            """
        )
        self._deprecated_print(
            "The sympy.physics.mechanics.Particle.set_potential_energy()"
            "method is deprecated. Instead use",

            P.potential_energy = scalar
            """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-set-potential-energy",
        )

将消息打印为已弃用的警告，通知用户该方法已经过时，提供了替代方法和相关信息。


        self.potential_energy = scalar

设置粒子的势能为给定的标量值。


    def parallel_axis(self, point, frame):
        """
        Returns an inertia dyadic of the particle with respect to another
        point and frame.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the particle expressed about the provided
            point and frame.
        """
        return inertia_of_point_mass(self.mass, self.point.pos_from(point),
                                     frame)

计算粒子相对于另一个点和参考系的惯性迪亚德（Inertia Dyadic），用于描述粒子的惯性特性。
```