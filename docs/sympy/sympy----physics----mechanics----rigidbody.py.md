# `D:\src\scipysrc\sympy\sympy\physics\mechanics\rigidbody.py`

```
    def inertia(self):
        """The inertia of the rigid body."""
        return self._inertia

    @inertia.setter
    def inertia(self, I):
        if not isinstance(I, tuple) or len(I) != 2 or not isinstance(I[0], Dyadic) or not isinstance(I[1], Point):
            raise TypeError("Inertia must be a tuple of (Dyadic, Point).")
        self._inertia = I
    def x(self):
        """
        The basis Vector for the body, in the x direction.
        """
        return self.frame.x

    @property
    def y(self):
        """
        The basis Vector for the body, in the y direction.
        """
        return self.frame.y

    @property
    def z(self):
        """
        The basis Vector for the body, in the z direction.
        """
        return self.frame.z

    @property
    def inertia(self):
        """
        The body's inertia about a point; stored as (Dyadic, Point).
        """
        return self._inertia

    @inertia.setter
    def inertia(self, I):
        """
        Setter for the body's inertia.

        Parameters
        ==========
        I : tuple
            A tuple containing a Dyadic object and a Point object.

        Raises
        ======
        TypeError
            If the input I does not match the expected format.

        Explanation
        ===========
        Updates the body's inertia and computes the central inertia.

        """
        # check if I is of the form (Dyadic, Point)
        if len(I) != 2 or not isinstance(I[0], Dyadic) or not isinstance(I[1], Point):
            raise TypeError("RigidBody inertia must be a tuple of the form (Dyadic, Point).")

        self._inertia = Inertia(I[0], I[1])
        # Compute central inertia I_S/S* using the formula I_S/S* = I_S/O - I_S*/O
        I_Ss_O = inertia_of_point_mass(self.mass,
                                       self.masscenter.pos_from(I[1]),
                                       self.frame)
        self._central_inertia = I[0] - I_Ss_O

    @property
    def central_inertia(self):
        """
        The body's central inertia dyadic.
        """
        return self._central_inertia

    @central_inertia.setter
    def central_inertia(self, I):
        """
        Setter for the body's central inertia.

        Parameters
        ==========
        I : Dyadic
            The central inertia dyadic.

        Raises
        ======
        TypeError
            If I is not a Dyadic object.

        """
        if not isinstance(I, Dyadic):
            raise TypeError("RigidBody inertia must be a Dyadic object.")
        self.inertia = Inertia(I, self.masscenter)

    def linear_momentum(self, frame):
        """
        Linear momentum of the rigid body.

        Parameters
        ==========
        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Explanation
        ===========
        Calculates and returns the linear momentum L of a rigid body B in frame N,
        given by L = m * v, where m is the mass and v is the velocity of the
        mass center of B in the frame N.

        Examples
        ========
        Provides an example usage of calculating linear momentum.

        """
        return self.mass * self.masscenter.vel(frame)
    def angular_momentum(self, point, frame):
        """
        Returns the angular momentum of the rigid body about a point in the
        given frame.

        Explanation
        ===========

        The angular momentum H of a rigid body B about some point O in a frame N
        is given by:

        ``H = dot(I, w) + cross(r, m * v)``

        where I and m are the central inertia dyadic and mass of rigid body B, w
        is the angular velocity of body B in the frame N, r is the position
        vector from point O to the mass center of B, and v is the velocity of
        the mass center in the frame N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum is desired.
        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r, omega = dynamicsymbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, 1 * N.x)
        >>> I = outer(b.x, b.x)
        >>> B = RigidBody('B', P, b, m, (I, P))
        >>> B.angular_momentum(P, N)
        omega*b.x

        """
        # 获取中心惯性矩阵
        I = self.central_inertia
        # 获取刚体相对于给定参考系的角速度
        w = self.frame.ang_vel_in(frame)
        # 获取刚体的质量
        m = self.mass
        # 获取从给定点到质心的位置矢量
        r = self.masscenter.pos_from(point)
        # 获取质心在给定参考系中的速度
        v = self.masscenter.vel(frame)

        # 计算并返回角动量，根据角速度、位置矢量和速度
        return I.dot(w) + r.cross(m * v)
        """设置刚体的势能。

        Explanation
        ===========

        势能U可以表示为刚体B的位置与其他参考点之间的势能差的总和。

        Parameters
        ==========

        scalar : scalar
            标量值，表示刚体的势能。

        Examples
        ========

        >>> scalar = 10
        >>> B.set_potential_energy(scalar)
        10

        """
        sympy_deprecation_warning(
            """
            警告：这个方法即将被弃用，请使用新的势能计算方法。
            """
        )
    The sympy.physics.mechanics.RigidBody.set_potential_energy()
    method is deprecated. Instead use

        B.potential_energy = scalar
                """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-set-potential-energy",
        )
        self.potential_energy = scalar


        注释：
        # sympy.physics.mechanics.RigidBody.set_potential_energy() 方法已经被弃用。现在应该使用下面的方法来设置势能：
        # 设置刚体的势能为标量值 scalar
        B.potential_energy = scalar


    def parallel_axis(self, point, frame=None):
        """Returns the inertia dyadic of the body with respect to another point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        """
        if frame is None:
            frame = self.frame
        return self.central_inertia + inertia_of_point_mass(
            self.mass, self.masscenter.pos_from(point), frame)


        注释：
        # 定义函数 parallel_axis(self, point, frame=None)，返回刚体相对于另一点的惯性迪亚德矩阵。
        
        # 参数：
        # point : sympy.physics.vector.Point
        #     惯性迪亚德矩阵相对于的点。
        # frame : sympy.physics.vector.ReferenceFrame
        #     用于构建迪亚德矩阵的参考坐标系。
        
        # 返回值：
        # inertia : sympy.physics.vector.Dyadic
        #     表示刚体相对于提供点的惯性迪亚德矩阵。
        
        如果未提供参考坐标系，则默认使用 self.frame。
        返回 self.central_inertia 加上使用 inertia_of_point_mass 计算的质量、质心到 point 的位置向量和参考坐标系 frame。
```