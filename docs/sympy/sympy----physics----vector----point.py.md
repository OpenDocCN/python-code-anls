# `D:\src\scipysrc\sympy\sympy\physics\vector\point.py`

```
# 从当前目录下的 vector 模块中导入 Vector 和 _check_vector
# 从 frame 模块中导入 _check_frame
from .vector import Vector, _check_vector
from .frame import _check_frame
# 导入警告模块中的 warn 函数
from warnings import warn
# 从 sympy.utilities.misc 模块中导入 filldedent 函数
from sympy.utilities.misc import filldedent

# 定义 __all__ 列表，指定 Point 类作为公开接口
__all__ = ['Point']

# 定义 Point 类，表示动态系统中的一个点
class Point:
    """This object represents a point in a dynamic system.

    It stores the: position, velocity, and acceleration of a point.
    The position is a vector defined as the vector distance from a parent
    point to this point.

    Parameters
    ==========

    name : string
        The display name of the Point

    Examples
    ========

    >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> P = Point('P')
    >>> u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    >>> O.set_vel(N, u1 * N.x + u2 * N.y + u3 * N.z)
    >>> O.acc(N)
    u1'*N.x + u2'*N.y + u3'*N.z

    ``symbols()`` can be used to create multiple Points in a single step, for
    example:

    >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import symbols
    >>> N = ReferenceFrame('N')
    >>> u1, u2 = dynamicsymbols('u1 u2')
    >>> A, B = symbols('A B', cls=Point)
    >>> type(A)
    <class 'sympy.physics.vector.point.Point'>
    >>> A.set_vel(N, u1 * N.x + u2 * N.y)
    >>> B.set_vel(N, u2 * N.x + u1 * N.y)
    >>> A.acc(N) - B.acc(N)
    (u1' - u2')*N.x + (-u1' + u2')*N.y

    """

    def __init__(self, name):
        """Initialization of a Point object. """
        # 设置点的名称
        self.name = name
        # 初始化位置、速度和加速度字典
        self._pos_dict = {}
        self._vel_dict = {}
        self._acc_dict = {}
        # 将这些字典放入列表中
        self._pdlist = [self._pos_dict, self._vel_dict, self._acc_dict]

    def __str__(self):
        # 返回点的名称
        return self.name

    __repr__ = __str__

    def _check_point(self, other):
        # 检查另一个对象是否是 Point 类型，如果不是则抛出 TypeError 异常
        if not isinstance(other, Point):
            raise TypeError('A Point must be supplied')
    def _pdict_list(self, other, num):
        """Returns a list of points that gives the shortest path with respect
        to position, velocity, or acceleration from this point to the provided
        point.

        Parameters
        ==========
        other : Point
            A point that may be related to this point by position, velocity, or
            acceleration.
        num : integer
            0 for searching the position tree, 1 for searching the velocity
            tree, and 2 for searching the acceleration tree.

        Returns
        =======
        list of Points
            A sequence of points from self to other.

        Notes
        =====

        It is not clear if num = 1 or num = 2 actually works because the keys
        to ``_vel_dict`` and ``_acc_dict`` are :class:`ReferenceFrame` objects
        which do not have the ``_pdlist`` attribute.

        """
        # Initialize a list containing a list with only the starting point self
        outlist = [[self]]
        # Initialize an empty list to hold the previous version of outlist
        oldlist = [[]]
        
        # Continue the loop until outlist no longer changes
        while outlist != oldlist:
            # Update oldlist to be a copy of outlist
            oldlist = outlist[:]
            # Iterate through each list of points in outlist
            for v in outlist:
                # Retrieve keys (points) from _pdlist[num] attribute of the last point in v
                templist = v[-1]._pdlist[num].keys()
                # Iterate through each key (point) v2 in templist
                for v2 in templist:
                    # Check if v does not already contain v2
                    if not v.__contains__(v2):
                        # Create a new list littletemplist by appending v2 to v
                        littletemplist = v + [v2]
                        # Check if outlist does not already contain littletemplist
                        if not outlist.__contains__(littletemplist):
                            # Add littletemplist to outlist
                            outlist.append(littletemplist)
        
        # Iterate through each list of points v in oldlist
        for v in oldlist:
            # Check if the last point in v is not equal to the provided point other
            if v[-1] != other:
                # Remove v from outlist
                outlist.remove(v)
        
        # Sort outlist based on the length of each list of points
        outlist.sort(key=len)
        
        # Check if outlist is not empty
        if len(outlist) != 0:
            # Return the shortest path found (the first list of points in outlist)
            return outlist[0]
        # Raise a ValueError if no connecting path is found
        raise ValueError('No Connecting Path found between ' + other.name +
                         ' and ' + self.name)
    def a1pt_theory(self, otherpoint, outframe, interframe):
        """Sets the acceleration of this point with the 1-point theory.

        The 1-point theory for point acceleration looks like this:

        ^N a^P = ^B a^P + ^N a^O + ^N alpha^B x r^OP + ^N omega^B x (^N omega^B
        x r^OP) + 2 ^N omega^B x ^B v^P

        where O is a point fixed in B, P is a point moving in B, and B is
        rotating in frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 1-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's acceleration defined in (N)
        interframe : ReferenceFrame
            The intermediate frame in this calculation (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> from sympy.physics.vector import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> q2 = dynamicsymbols('q2')
        >>> qd = dynamicsymbols('q', 1)
        >>> q2d = dynamicsymbols('q2', 1)
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.set_ang_vel(N, 5 * B.y)
        >>> O = Point('O')
        >>> P = O.locatenew('P', q * B.x + q2 * B.y)
        >>> P.set_vel(B, qd * B.x + q2d * B.y)
        >>> O.set_vel(N, 0)
        >>> P.a1pt_theory(O, N, B)
        (-25*q + q'')*B.x + q2''*B.y - 10*q'*B.z

        """

        # 检查输出参考系的有效性
        _check_frame(outframe)
        # 检查中间参考系的有效性
        _check_frame(interframe)
        # 检查第一个点的有效性
        self._check_point(otherpoint)
        # 计算该点与其他点之间的位置矢量
        dist = self.pos_from(otherpoint)
        # 获取该点在中间参考系中的速度
        v = self.vel(interframe)
        # 获取第一个点在输出参考系中的加速度
        a1 = otherpoint.acc(outframe)
        # 获取该点在中间参考系中的加速度
        a2 = self.acc(interframe)
        # 获取中间参考系相对于输出参考系的角速度
        omega = interframe.ang_vel_in(outframe)
        # 获取中间参考系相对于输出参考系的角加速度
        alpha = interframe.ang_acc_in(outframe)
        # 设置该点在输出参考系中的加速度，根据1-point theory公式计算
        self.set_acc(outframe, a2 + 2 * (omega.cross(v)) + a1 +
                     (alpha.cross(dist)) + (omega.cross(omega.cross(dist))))
        # 返回该点在输出参考系中的加速度
        return self.acc(outframe)
    def a2pt_theory(self, otherpoint, outframe, fixedframe):
        """Sets the acceleration of this point with the 2-point theory.

        The 2-point theory for point acceleration looks like this:

        ^N a^P = ^N a^O + ^N alpha^B x r^OP + ^N omega^B x (^N omega^B x r^OP)

        where O and P are both points fixed in frame B, which is rotating in
        frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 2-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's acceleration defined in (N)
        fixedframe : ReferenceFrame
            The frame in which both points are fixed (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> N = ReferenceFrame('N')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> O = Point('O')
        >>> P = O.locatenew('P', 10 * B.x)
        >>> O.set_vel(N, 5 * N.x)
        >>> P.a2pt_theory(O, N, B)
        - 10*q'**2*B.x + 10*q''*B.y

        """

        # 检查输出参考系是否有效
        _check_frame(outframe)
        # 检查固定参考系是否有效
        _check_frame(fixedframe)
        # 检查第一个点是否有效
        self._check_point(otherpoint)
        # 计算两点之间的位置向量
        dist = self.pos_from(otherpoint)
        # 获取第一个点的加速度
        a = otherpoint.acc(outframe)
        # 获取固定参考系的角速度
        omega = fixedframe.ang_vel_in(outframe)
        # 获取固定参考系的角加速度
        alpha = fixedframe.ang_acc_in(outframe)
        # 设置该点在输出参考系中的加速度
        self.set_acc(outframe, a + (alpha.cross(dist)) +
                     (omega.cross(omega.cross(dist))))
        # 返回该点在输出参考系中的加速度
        return self.acc(outframe)

    def acc(self, frame):
        """The acceleration Vector of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which the returned acceleration vector will be defined
            in.

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_acc(N, 10 * N.x)
        >>> p1.acc(N)
        10*N.x

        """

        # 检查给定的参考系是否有效
        _check_frame(frame)
        # 如果加速度字典中不存在该参考系的加速度向量，则计算其加速度向量
        if not (frame in self._acc_dict):
            # 如果该点在该参考系的速度不为零，则返回其加速度的导数
            if self.vel(frame) != 0:
                return (self._vel_dict[frame]).dt(frame)
            else:
                # 否则返回零向量
                return Vector(0)
        # 返回已存储的加速度向量
        return self._acc_dict[frame]
    def locatenew(self, name, value):
        """
        创建一个新的点，其位置相对于当前点定义。

        Parameters
        ==========

        name : str
            新点的名称
        value : Vector
            相对于当前点的新点位置向量

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, Point
        >>> N = ReferenceFrame('N')
        >>> P1 = Point('P1')
        >>> P2 = P1.locatenew('P2', 10 * N.x)

        """

        if not isinstance(name, str):
            raise TypeError('Must supply a valid name')  # 如果名称不是字符串，则抛出类型错误异常
        if value == 0:
            value = Vector(0)  # 如果值为零，则设置为零向量
        value = _check_vector(value)  # 检查并确保值是向量类型
        p = Point(name)  # 创建一个新的点对象
        p.set_pos(self, value)  # 设置新点相对于当前点的位置
        self.set_pos(p, -value)  # 设置当前点相对于新点的位置
        return p  # 返回新创建的点对象

    def pos_from(self, otherpoint):
        """
        返回当前点与另一点之间的向量距离。

        Parameters
        ==========

        otherpoint : Point
            另一个点，用于计算与当前点的距离向量

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')
        >>> p1.set_pos(p2, 10 * N.x)
        >>> p1.pos_from(p2)
        10*N.x

        """

        outvec = Vector(0)  # 初始化输出向量为零向量
        plist = self._pdict_list(otherpoint, 0)  # 获取连接两个点的路径列表
        for i in range(len(plist) - 1):
            outvec += plist[i]._pos_dict[plist[i + 1]]  # 计算两点之间每一段路径的向量并累加
        return outvec  # 返回两点之间的总向量距离

    def set_acc(self, frame, value):
        """
        设置该点在参考坐标系中的加速度。

        Parameters
        ==========

        frame : ReferenceFrame
            加速度定义所在的参考坐标系
        value : Vector
            该点在参考坐标系中的加速度向量值

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_acc(N, 10 * N.x)
        >>> p1.acc(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)  # 如果值为零，则设置为零向量
        value = _check_vector(value)  # 检查并确保值是向量类型
        _check_frame(frame)  # 检查并确保参考坐标系是有效的
        self._acc_dict.update({frame: value})  # 更新加速度字典，将加速度值与参考坐标系关联起来
    def set_pos(self, otherpoint, value):
        """Used to set the position of this point w.r.t. another point.

        Parameters
        ==========

        otherpoint : Point
            The other point which this point's location is defined relative to
        value : Vector
            The vector which defines the location of this point

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')
        >>> p1.set_pos(p2, 10 * N.x)
        >>> p1.pos_from(p2)
        10*N.x

        """

        # 如果给定的值为零向量，则转换为零向量对象
        if value == 0:
            value = Vector(0)
        # 确保值是一个合法的向量对象
        value = _check_vector(value)
        # 检查 otherpoint 是否是合法的 Point 对象
        self._check_point(otherpoint)
        # 更新当前点与其他点之间的位置关系字典
        self._pos_dict.update({otherpoint: value})
        # 同时更新其他点与当前点之间的位置关系字典，保持位置对称性
        otherpoint._pos_dict.update({self: -value})

    def set_vel(self, frame, value):
        """Sets the velocity Vector of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which this point's velocity is defined
        value : Vector
            The vector value of this point's velocity in the frame

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_vel(N, 10 * N.x)
        >>> p1.vel(N)
        10*N.x

        """

        # 如果给定的值为零向量，则转换为零向量对象
        if value == 0:
            value = Vector(0)
        # 确保值是一个合法的向量对象
        value = _check_vector(value)
        # 检查 frame 是否是合法的 ReferenceFrame 对象
        _check_frame(frame)
        # 更新当前点在指定参考系中的速度字典
        self._vel_dict.update({frame: value})
    def v1pt_theory(self, otherpoint, outframe, interframe):
        """
        Sets the velocity of this point using the 1-point theory.

        The 1-point theory for point velocity looks like this:

        ^N v^P = ^B v^P + ^N v^O + ^N omega^B x r^OP

        where O is a point fixed in B, P is a point moving in B, and B is
        rotating in frame N.

        Parameters
        ==========

        otherpoint : Point
            The point O in the 1-point theory, fixed in frame B.
        outframe : ReferenceFrame
            The frame N where the velocity of this point is defined.
        interframe : ReferenceFrame
            The intermediate frame B where point P is moving.

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> from sympy.physics.vector import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> q2 = dynamicsymbols('q2')
        >>> qd = dynamicsymbols('q', 1)
        >>> q2d = dynamicsymbols('q2', 1)
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.set_ang_vel(N, 5 * B.y)
        >>> O = Point('O')
        >>> P = O.locatenew('P', q * B.x + q2 * B.y)
        >>> P.set_vel(B, qd * B.x + q2d * B.y)
        >>> O.set_vel(N, 0)
        >>> P.v1pt_theory(O, N, B)
        q'*B.x + q2'*B.y - 5*q*B.z

        """

        _check_frame(outframe)  # Validate that outframe is a ReferenceFrame object
        _check_frame(interframe)  # Validate that interframe is a ReferenceFrame object
        self._check_point(otherpoint)  # Ensure otherpoint is a Point object related to the current point
        dist = self.pos_from(otherpoint)  # Calculate the position vector from otherpoint to self
        v1 = self.vel(interframe)  # Calculate the velocity of self in the intermediate frame B
        v2 = otherpoint.vel(outframe)  # Calculate the velocity of otherpoint in the frame N
        omega = interframe.ang_vel_in(outframe)  # Determine the angular velocity of frame B in frame N
        # Set the velocity of self in frame N using the 1-point theory equation
        self.set_vel(outframe, v1 + v2 + (omega.cross(dist)))
        # Return the velocity of self in frame N
        return self.vel(outframe)
    # 使用 2 点理论设置此点的速度。
    def v2pt_theory(self, otherpoint, outframe, fixedframe):
        """Sets the velocity of this point with the 2-point theory.

        The 2-point theory for point velocity looks like this:

        ^N v^P = ^N v^O + ^N omega^B x r^OP

        where O and P are both points fixed in frame B, which is rotating in
        frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 2-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's velocity defined in (N)
        fixedframe : ReferenceFrame
            The frame in which both points are fixed (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> N = ReferenceFrame('N')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> O = Point('O')
        >>> P = O.locatenew('P', 10 * B.x)
        >>> O.set_vel(N, 5 * N.x)
        >>> P.v2pt_theory(O, N, B)
        5*N.x + 10*q'*B.y

        """

        _check_frame(outframe)  # 检查输出参考系的有效性
        _check_frame(fixedframe)  # 检查固定参考系的有效性
        self._check_point(otherpoint)  # 检查输入的其他点是否有效
        dist = self.pos_from(otherpoint)  # 计算此点与其他点之间的位移向量
        v = otherpoint.vel(outframe)  # 获取其他点在输出参考系中的速度向量
        omega = fixedframe.ang_vel_in(outframe)  # 获取固定参考系相对于输出参考系的角速度向量
        self.set_vel(outframe, v + (omega.cross(dist)))  # 设置此点在输出参考系中的速度
        return self.vel(outframe)  # 返回此点在输出参考系中的速度向量

    # 返回此点在给定参考系中对于一个或多个广义速度的部分速度向量
    def partial_velocity(self, frame, *gen_speeds):
        """Returns the partial velocities of the linear velocity vector of this
        point in the given frame with respect to one or more provided
        generalized speeds.

        Parameters
        ==========
        frame : ReferenceFrame
            The frame with which the velocity is defined in.
        gen_speeds : functions of time
            The generalized speeds.

        Returns
        =======
        partial_velocities : tuple of Vector
            The partial velocity vectors corresponding to the provided
            generalized speeds.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, Point
        >>> from sympy.physics.vector import dynamicsymbols
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> p = Point('p')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> p.set_vel(N, u1 * N.x + u2 * A.y)
        >>> p.partial_velocity(N, u1)
        N.x
        >>> p.partial_velocity(N, u1, u2)
        (N.x, A.y)

        """

        from sympy.physics.vector.functions import partial_velocity

        vel = self.vel(frame)  # 获取此点在给定参考系中的速度向量
        partials = partial_velocity([vel], gen_speeds, frame)[0]  # 计算部分速度向量

        if len(partials) == 1:
            return partials[0]
        else:
            return tuple(partials)
```