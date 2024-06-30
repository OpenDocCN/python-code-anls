# `D:\src\scipysrc\sympy\sympy\vector\orienters.py`

```
from sympy.core.basic import Basic
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, rot_axis1, rot_axis2, rot_axis3)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.core.cache import cacheit
from sympy.core.symbol import Str
import sympy.vector

class Orienter(Basic):
    """
    Super-class for all orienter classes.
    """

    def rotation_matrix(self):
        """
        The rotation matrix corresponding to this orienter
        instance.
        """
        return self._parent_orient


class AxisOrienter(Orienter):
    """
    Class to denote an axis orienter.
    """

    def __new__(cls, angle, axis):
        if not isinstance(axis, sympy.vector.Vector):
            raise TypeError("axis should be a Vector")
        angle = sympify(angle)

        obj = super().__new__(cls, angle, axis)
        obj._angle = angle
        obj._axis = axis

        return obj

    def __init__(self, angle, axis):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> from sympy.vector import AxisOrienter
        >>> orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> B = N.orient_new('B', (orienter, ))

        """
        # Dummy initializer for docstrings
        pass

    @cacheit
    def rotation_matrix(self, system):
        """
        The rotation matrix corresponding to this orienter
        instance.

        Parameters
        ==========

        system : CoordSys3D
            The coordinate system wrt which the rotation matrix
            is to be computed
        """
        # 将轴向量表达为给定坐标系的矩阵，并归一化
        axis = sympy.vector.express(self.axis, system).normalize()
        axis = axis.to_matrix(system)
        # 获取角度
        theta = self.angle
        # 计算旋转矩阵
        parent_orient = ((eye(3) - axis * axis.T) * cos(theta) +
                         Matrix([[0, -axis[2], axis[1]],
                                 [axis[2], 0, -axis[0]],
                                 [-axis[1], axis[0], 0]]) * sin(theta) +
                         axis * axis.T)
        parent_orient = parent_orient.T
        return parent_orient

    @property
    def angle(self):
        return self._angle

    @property
    def axis(self):
        return self._axis


class ThreeAngleOrienter(Orienter):
    """
    Super-class for Body and Space orienters.
    """
    # 定义一个特殊方法 __new__，用于创建新对象，接收四个参数：角度1、角度2、角度3、旋转顺序
    def __new__(cls, angle1, angle2, angle3, rot_order):
        # 如果 rot_order 是 Str 类型，则转换为其名称
        if isinstance(rot_order, Str):
            rot_order = rot_order.name

        # 定义可接受的旋转顺序组合
        approved_orders = ('123', '231', '312', '132', '213',
                           '321', '121', '131', '212', '232',
                           '313', '323', '')
        
        # 保存原始的 rot_order 值
        original_rot_order = rot_order
        
        # 将 rot_order 转换为大写字符串形式
        rot_order = str(rot_order).upper()
        
        # 检查 rot_order 的长度是否为 3
        if not (len(rot_order) == 3):
            raise TypeError('rot_order should be a str of length 3')
        
        # 将 rot_order 中的 'X' 替换为 '1'，'Y' 替换为 '2'，'Z' 替换为 '3'
        rot_order = [i.replace('X', '1') for i in rot_order]
        rot_order = [i.replace('Y', '2') for i in rot_order]
        rot_order = [i.replace('Z', '3') for i in rot_order]
        
        # 将列表转换为字符串
        rot_order = ''.join(rot_order)
        
        # 检查处理后的 rot_order 是否在可接受的旋转顺序中
        if rot_order not in approved_orders:
            raise TypeError('Invalid rot_type parameter')
        
        # 提取每个角度的整数值
        a1 = int(rot_order[0])
        a2 = int(rot_order[1])
        a3 = int(rot_order[2])
        
        # 将 angle1、angle2、angle3 转换为符号表达式
        angle1 = sympify(angle1)
        angle2 = sympify(angle2)
        angle3 = sympify(angle3)
        
        # 根据 _in_order 的值选择父级方向的旋转顺序
        if cls._in_order:
            parent_orient = (_rot(a1, angle1) *
                             _rot(a2, angle2) *
                             _rot(a3, angle3))
        else:
            parent_orient = (_rot(a3, angle3) *
                             _rot(a2, angle2) *
                             _rot(a1, angle1))
        
        # 获取 parent_orient 的转置
        parent_orient = parent_orient.T
        
        # 调用父类的 __new__ 方法创建新对象
        obj = super().__new__(
            cls, angle1, angle2, angle3, Str(rot_order))
        
        # 初始化对象的属性
        obj._angle1 = angle1
        obj._angle2 = angle2
        obj._angle3 = angle3
        obj._rot_order = original_rot_order
        obj._parent_orient = parent_orient
        
        # 返回创建的对象
        return obj

    # 定义 angle1 的属性方法
    @property
    def angle1(self):
        return self._angle1

    # 定义 angle2 的属性方法
    @property
    def angle2(self):
        return self._angle2

    # 定义 angle3 的属性方法
    @property
    def angle3(self):
        return self._angle3

    # 定义 rot_order 的属性方法
    @property
    def rot_order(self):
        return self._rot_order
# 定义一个名为 BodyOrienter 的类，继承自 ThreeAngleOrienter 类
class BodyOrienter(ThreeAngleOrienter):
    """
    Class to denote a body-orienter.
    """

    # 类级别的属性，用于指示是否按照特定顺序进行旋转
    _in_order = True

    # 类的特殊方法 __new__，用于创建一个新的对象实例
    def __new__(cls, angle1, angle2, angle3, rot_order):
        # 调用父类 ThreeAngleOrienter 的 __new__ 方法创建对象
        obj = ThreeAngleOrienter.__new__(cls, angle1, angle2, angle3,
                                         rot_order)
        return obj

    # 类的初始化方法 __init__，用于初始化对象的属性和文档字符串
    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, BodyOrienter
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')
        >>> D = N.orient_new('D', (body_orienter, ))

        is same as

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter1 = AxisOrienter(q1, N.i)
        >>> D = N.orient_new('D', (axis_orienter1, ))
        >>> axis_orienter2 = AxisOrienter(q2, D.j)
        >>> D = D.orient_new('D', (axis_orienter2, ))
        >>> axis_orienter3 = AxisOrienter(q3, D.k)
        >>> D = D.orient_new('D', (axis_orienter3, ))

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> body_orienter1 = BodyOrienter(q1, q2, q3, '123')
        >>> body_orienter2 = BodyOrienter(q1, q2, 0, 'ZXZ')
        >>> body_orienter3 = BodyOrienter(0, 0, 0, 'XYX')

        """
        # 空白的初始化方法，用于文档字符串的占位符
        pass


# 定义一个名为 SpaceOrienter 的类，继承自 ThreeAngleOrienter 类
class SpaceOrienter(ThreeAngleOrienter):
    """
    Class to denote a space-orienter.
    """

    # 类级别的属性，用于指示是否按照特定顺序进行旋转
    _in_order = False

    # 类的特殊方法 __new__，用于创建一个新的对象实例
    def __new__(cls, angle1, angle2, angle3, rot_order):
        # 调用父类 ThreeAngleOrienter 的 __new__ 方法创建对象
        obj = ThreeAngleOrienter.__new__(cls, angle1, angle2, angle3,
                                         rot_order)
        return obj
    # 定义类的初始化方法，用于描述空间旋转操作，与Body rotation类似，但旋转顺序相反。

    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        See Also
        ========

        BodyOrienter : Orienter to orient systems wrt Euler angles.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, SpaceOrienter
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')
        >>> D = N.orient_new('D', (space_orienter, ))

        is same as

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter1 = AxisOrienter(q1, N.i)
        >>> B = N.orient_new('B', (axis_orienter1, ))
        >>> axis_orienter2 = AxisOrienter(q2, N.j)
        >>> C = B.orient_new('C', (axis_orienter2, ))
        >>> axis_orienter3 = AxisOrienter(q3, N.k)
        >>> D = C.orient_new('C', (axis_orienter3, ))

        """
        # Dummy initializer for docstrings
        # 占位的初始化方法，仅用于文档字符串展示目的，实际无功能。
        pass
class QuaternionOrienter(Orienter):
    """
    Class to denote a quaternion-orienter.
    """

    def __new__(cls, q0, q1, q2, q3):
        q0 = sympify(q0)  # 将参数 q0 转换为 Sympy 表达式
        q1 = sympify(q1)  # 将参数 q1 转换为 Sympy 表达式
        q2 = sympify(q2)  # 将参数 q2 转换为 Sympy 表达式
        q3 = sympify(q3)  # 将参数 q3 转换为 Sympy 表达式
        # 计算父方向的四元数表示矩阵
        parent_orient = (Matrix([[q0 ** 2 + q1 ** 2 - q2 ** 2 -
                                  q3 ** 2,
                                  2 * (q1 * q2 - q0 * q3),
                                  2 * (q0 * q2 + q1 * q3)],
                                 [2 * (q1 * q2 + q0 * q3),
                                  q0 ** 2 - q1 ** 2 +
                                  q2 ** 2 - q3 ** 2,
                                  2 * (q2 * q3 - q0 * q1)],
                                 [2 * (q1 * q3 - q0 * q2),
                                  2 * (q0 * q1 + q2 * q3),
                                  q0 ** 2 - q1 ** 2 -
                                  q2 ** 2 + q3 ** 2]]))
        parent_orient = parent_orient.T  # 转置父方向的四元数表示矩阵

        obj = super().__new__(cls, q0, q1, q2, q3)  # 调用父类的构造函数创建对象
        obj._q0 = q0  # 设置对象的 q0 属性
        obj._q1 = q1  # 设置对象的 q1 属性
        obj._q2 = q2  # 设置对象的 q2 属性
        obj._q3 = q3  # 设置对象的 q3 属性
        obj._parent_orient = parent_orient  # 设置对象的父方向属性

        return obj  # 返回创建的对象

    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> from sympy.vector import QuaternionOrienter
        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)
        >>> B = N.orient_new('B', (q_orienter, ))

        """
        # Dummy initializer for docstrings
        pass

    @property
    def q0(self):
        return self._q0  # 返回对象的 q0 属性

    @property
    def q1(self):
        return self._q1  # 返回对象的 q1 属性

    @property
    def q2(self):
        return self._q2  # 返回对象的 q2 属性

    @property
    def q3(self):
        return self._q3  # 返回对象的 q3 属性


def _rot(axis, angle):
    """DCM for simple axis 1, 2 or 3 rotations. """
    if axis == 1:
        return Matrix(rot_axis1(angle).T)  # 返回绕轴 1 旋转角度的方向余弦矩阵
    elif axis == 2:
        return Matrix(rot_axis2(angle).T)  # 返回绕轴 2 旋转角度的方向余弦矩阵
    elif axis == 3:
        return Matrix(rot_axis3(angle).T)  # 返回绕轴 3 旋转角度的方向余弦矩阵
```