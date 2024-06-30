# `D:\src\scipysrc\sympy\sympy\physics\vector\frame.py`

```
# 导入必要的库和模块
from sympy import (diff, expand, sin, cos, sympify, eye, zeros,
                                ImmutableMatrix as Matrix, MatrixBase)
from sympy.core.symbol import Symbol  # 导入符号类
from sympy.simplify.trigsimp import trigsimp  # 导入三角函数简化工具
from sympy.physics.vector.vector import Vector, _check_vector  # 导入向量和向量检查函数
from sympy.utilities.misc import translate  # 导入翻译函数

from warnings import warn  # 导入警告模块

__all__ = ['CoordinateSym', 'ReferenceFrame']  # 模块的公开接口，包含 CoordinateSym 和 ReferenceFrame 类

class CoordinateSym(Symbol):
    """
    A coordinate symbol/base scalar associated wrt a Reference Frame.

    Ideally, users should not instantiate this class. Instances of
    this class must only be accessed through the corresponding frame
    as 'frame[index]'.

    CoordinateSyms having the same frame and index parameters are equal
    (even though they may be instantiated separately).

    Parameters
    ==========

    name : string
        The display name of the CoordinateSym

    frame : ReferenceFrame
        The reference frame this base scalar belongs to

    index : 0, 1 or 2
        The index of the dimension denoted by this coordinate variable

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, CoordinateSym
    >>> A = ReferenceFrame('A')
    >>> A[1]
    A_y
    >>> type(A[0])
    <class 'sympy.physics.vector.frame.CoordinateSym'>
    >>> a_y = CoordinateSym('a_y', A, 1)
    >>> a_y == A[1]
    True

    """

    def __new__(cls, name, frame, index):
        # We can't use the cached Symbol.__new__ because this class depends on
        # frame and index, which are not passed to Symbol.__xnew__.
        assumptions = {}
        super()._sanitize(assumptions, cls)  # 调用父类的符号初始化方法，确保符号类的一致性
        obj = super().__xnew__(cls, name, **assumptions)  # 创建新的 CoordinateSym 对象
        _check_frame(frame)  # 检查参考系对象的合法性
        if index not in range(0, 3):
            raise ValueError("Invalid index specified")  # 如果索引不在 0 到 2 的范围内，抛出异常
        obj._id = (frame, index)  # 设置 CoordinateSym 对象的参考系和索引属性
        return obj

    def __getnewargs_ex__(self):
        return (self.name, *self._id), {}  # 返回用于创建对象的参数元组，包括名称和参考系索引

    @property
    def frame(self):
        return self._id[0]  # 返回 CoordinateSym 对象的参考系属性

    def __eq__(self, other):
        # Check if the other object is a CoordinateSym of the same frame and
        # same index
        if isinstance(other, CoordinateSym):
            if other._id == self._id:
                return True  # 检查是否两个 CoordinateSym 对象具有相同的参考系和索引
        return False

    def __ne__(self, other):
        return not self == other  # 检查两个 CoordinateSym 对象是否不相等

    def __hash__(self):
        return (self._id[0].__hash__(), self._id[1]).__hash__()  # 返回 CoordinateSym 对象的哈希值

class ReferenceFrame:
    """A reference frame in classical mechanics.

    ReferenceFrame is a class used to represent a reference frame in classical
    mechanics. It has a standard basis of three unit vectors in the frame's
    x, y, and z directions.

    It also can have a rotation relative to a parent frame; this rotation is
    defined by a direction cosine matrix relating this frame's basis vectors to
    the parent frame's basis vectors.  It can also have an angular velocity
    vector, defined in another frame.

    """
    _count = 0  # 计数器，用于跟踪创建的 ReferenceFrame 实例数量
    # 定义特殊方法，用于通过索引访问对象的属性或基向量

    def __getitem__(self, ind):
        """
        Returns basis vector for the provided index, if the index is a string.

        If the index is a number, returns the coordinate variable correspon-
        -ding to that index.
        """
        # 如果索引不是字符串类型
        if not isinstance(ind, str):
            # 如果索引小于3，返回对应的变量列表中的元素
            if ind < 3:
                return self.varlist[ind]
            else:
                # 抛出数值错误异常，索引无效
                raise ValueError("Invalid index provided")
        
        # 如果索引是字符串类型
        # 检查索引是否是预定义的索引之一，返回对应的基向量
        if self.indices[0] == ind:
            return self.x
        if self.indices[1] == ind:
            return self.y
        if self.indices[2] == ind:
            return self.z
        else:
            # 如果索引未定义，抛出值错误异常
            raise ValueError('Not a defined index')

    # 定义特殊方法，使对象可迭代
    def __iter__(self):
        # 返回基向量的迭代器
        return iter([self.x, self.y, self.z])

    # 定义特殊方法，返回对象的字符串表示
    def __str__(self):
        """Returns the name of the frame. """
        # 返回框架的名称作为字符串
        return self.name

    # 使用 __str__ 方法作为 __repr__ 方法的实现
    __repr__ = __str__
    def _dict_list(self, other, num):
        """
        Returns an inclusive list of reference frames that connect this
        reference frame to the provided reference frame.

        Parameters
        ==========
        other : ReferenceFrame
            The other reference frame to look for a connecting relationship to.
        num : integer
            ``0``, ``1``, and ``2`` will look for orientation, angular
            velocity, and angular acceleration relationships between the two
            frames, respectively.

        Returns
        =======
        list
            Inclusive list of reference frames that connect this reference
            frame to the other reference frame.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> C = ReferenceFrame('C')
        >>> D = ReferenceFrame('D')
        >>> B.orient_axis(A, A.x, 1.0)
        >>> C.orient_axis(B, B.x, 1.0)
        >>> D.orient_axis(C, C.x, 1.0)
        >>> D._dict_list(A, 0)
        [D, C, B, A]

        Raises
        ======

        ValueError
            When no path is found between the two reference frames or ``num``
            is an incorrect value.

        """

        # Dictionary mapping `num` values to connecting relationship types
        connect_type = {0: 'orientation',
                        1: 'angular velocity',
                        2: 'angular acceleration'}

        # Validate the `num` parameter
        if num not in connect_type.keys():
            raise ValueError('Valid values for num are 0, 1, or 2.')

        # Initialize the list of possible connecting paths
        possible_connecting_paths = [[self]]
        oldlist = [[]]

        # Iterate until no new paths are found
        while possible_connecting_paths != oldlist:
            oldlist = possible_connecting_paths[:]  # make a copy
            for frame_list in possible_connecting_paths:
                # Get adjacent frames based on the specified `num`
                frames_adjacent_to_last = frame_list[-1]._dlist[num].keys()
                for adjacent_frame in frames_adjacent_to_last:
                    if adjacent_frame not in frame_list:
                        # Extend the connecting path
                        connecting_path = frame_list + [adjacent_frame]
                        if connecting_path not in possible_connecting_paths:
                            possible_connecting_paths.append(connecting_path)

        # Filter out paths that don't end with `other` and sort by length
        for connecting_path in oldlist:
            if connecting_path[-1] != other:
                possible_connecting_paths.remove(connecting_path)
        possible_connecting_paths.sort(key=len)

        # Return the shortest connecting path if found
        if len(possible_connecting_paths) != 0:
            return possible_connecting_paths[0]

        # Raise an error if no connecting path is found
        msg = 'No connecting {} path found between {} and {}.'
        raise ValueError(msg.format(connect_type[num], self.name, other.name))
    def _w_diff_dcm(self, otherframe):
        """
        计算参考框架的角速度，通过对方向余弦矩阵进行时间微分得到。
        """
        # 导入动态符号
        from sympy.physics.vector.functions import dynamicsymbols
        # 计算从 otherframe 到 self 的方向余弦矩阵的时间微分
        dcm2diff = otherframe.dcm(self)
        # 对时间进行微分
        diffed = dcm2diff.diff(dynamicsymbols._t)
        # 计算角速度矩阵
        angvelmat = diffed * dcm2diff.T
        # 对角速度矩阵中的元素进行三角函数简化
        w1 = trigsimp(expand(angvelmat[7]), recursive=True)
        w2 = trigsimp(expand(angvelmat[2]), recursive=True)
        w3 = trigsimp(expand(angvelmat[3]), recursive=True)
        # 返回以向量形式表示的角速度和 otherframe 的元组
        return Vector([(Matrix([w1, w2, w3]), otherframe)])

    def variable_map(self, otherframe):
        """
        返回一个字典，将该参考框架的坐标变量表达为另一个框架的变量。

        如果 Vector.simp 为 True，则返回映射值的简化版本。否则，返回未简化的值。

        参数
        ==========

        otherframe : ReferenceFrame
            要映射变量的另一个参考框架

        示例
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> A = ReferenceFrame('A')
        >>> q = dynamicsymbols('q')
        >>> B = A.orientnew('B', 'Axis', [q, A.z])
        >>> A.variable_map(B)
        {A_x: B_x*cos(q(t)) - B_y*sin(q(t)), A_y: B_x*sin(q(t)) + B_y*cos(q(t)), A_z: B_z}

        """

        # 检查框架是否有效
        _check_frame(otherframe)
        # 如果已经计算过映射并且简化设置相同，则直接返回存储的映射结果
        if (otherframe, Vector.simp) in self._var_dict:
            return self._var_dict[(otherframe, Vector.simp)]
        else:
            # 计算方向余弦矩阵乘以 otherframe 的坐标变量矩阵
            vars_matrix = self.dcm(otherframe) * Matrix(otherframe.varlist)
            mapping = {}
            # 遍历 self 的坐标变量，并根据 Vector.simp 的设置选择是否简化表达式
            for i, x in enumerate(self):
                if Vector.simp:
                    mapping[self.varlist[i]] = trigsimp(vars_matrix[i], method='fu')
                else:
                    mapping[self.varlist[i]] = vars_matrix[i]
            # 缓存映射结果以备后续使用
            self._var_dict[(otherframe, Vector.simp)] = mapping
            return mapping

    def ang_acc_in(self, otherframe):
        """
        返回参考框架的角加速度向量。

        实际上返回向量:

        ``N_alpha_B``

        其中 B 是 self，N 是 otherframe。

        参数
        ==========

        otherframe : ReferenceFrame
            角加速度所在的参考框架

        示例
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_acc(N, V)
        >>> A.ang_acc_in(N)
        10*N.x

        """

        # 检查框架是否有效
        _check_frame(otherframe)
        # 如果已经计算过该框架的角加速度，则直接返回缓存的结果
        if otherframe in self._ang_acc_dict:
            return self._ang_acc_dict[otherframe]
        else:
            # 否则返回 self 在 otherframe 中的角速度的时间导数
            return self.ang_vel_in(otherframe).dt(otherframe)
    def ang_vel_in(self, otherframe):
        """
        Returns the angular velocity Vector of the ReferenceFrame.

        Effectively returns the Vector:

        ^N omega ^B

        which represent the angular velocity of B in N, where B is self, and
        N is otherframe.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The ReferenceFrame in which the angular velocity is returned.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_vel(N, V)
        >>> A.ang_vel_in(N)
        10*N.x
        """

        # Check if otherframe is a valid ReferenceFrame object
        _check_frame(otherframe)

        # Get a list of frames between self and otherframe, including both
        flist = self._dict_list(otherframe, 1)

        # Initialize the output vector
        outvec = Vector(0)

        # Iterate over the frames in flist to accumulate angular velocities
        for i in range(len(flist) - 1):
            outvec += flist[i]._ang_vel_dict[flist[i + 1]]

        # Return the accumulated angular velocity vector
        return outvec
    def dcm(self, otherframe):
        r"""Returns the direction cosine matrix of this reference frame
        relative to the provided reference frame.

        The returned matrix can be used to express the orthogonal unit vectors
        of this frame in terms of the orthogonal unit vectors of
        ``otherframe``.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The reference frame which the direction cosine matrix of this frame
            is formed relative to.

        Examples
        ========

        The following example rotates the reference frame A relative to N by a
        simple rotation and then calculates the direction cosine matrix of N
        relative to A.

        >>> from sympy import symbols, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, q1, N.x)
        >>> N.dcm(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The second row of the above direction cosine matrix represents the
        ``N.y`` unit vector in N expressed in A. Like so:

        >>> Ny = 0*A.x + cos(q1)*A.y - sin(q1)*A.z

        Thus, expressing ``N.y`` in A should return the same result:

        >>> N.y.express(A)
        cos(q1)*A.y - sin(q1)*A.z

        Notes
        =====

        It is important to know what form of the direction cosine matrix is
        returned. If ``B.dcm(A)`` is called, it means the "direction cosine
        matrix of B rotated relative to A". This is the matrix
        :math:`{}^B\mathbf{C}^A` shown in the following relationship:

        .. math::

           \begin{bmatrix}
             \hat{\mathbf{b}}_1 \\
             \hat{\mathbf{b}}_2 \\
             \hat{\mathbf{b}}_3
           \end{bmatrix}
           =
           {}^B\mathbf{C}^A
           \begin{bmatrix}
             \hat{\mathbf{a}}_1 \\
             \hat{\mathbf{a}}_2 \\
             \hat{\mathbf{a}}_3
           \end{bmatrix}.

        :math:`{}^B\mathbf{C}^A` is the matrix that expresses the B unit
        vectors in terms of the A unit vectors.

        """

        # Check if the provided otherframe is valid
        _check_frame(otherframe)

        # Check if the direction cosine matrix with respect to `otherframe` is already cached
        if otherframe in self._dcm_cache:
            # If cached, return the cached matrix to avoid recomputation
            return self._dcm_cache[otherframe]

        # Calculate the list of transformation matrices needed to transform to `otherframe`
        flist = self._dict_list(otherframe, 0)

        # Initialize the output direction cosine matrix as identity matrix
        outdcm = eye(3)

        # Multiply transformation matrices to get the final direction cosine matrix
        for i in range(len(flist) - 1):
            outdcm = outdcm * flist[i]._dcm_dict[flist[i + 1]]

        # Cache the computed direction cosine matrix for faster future retrieval
        self._dcm_cache[otherframe] = outdcm

        # Store the transpose of the computed matrix in the otherframe's cache for consistency
        otherframe._dcm_cache[self] = outdcm.T

        # Return the computed direction cosine matrix
        return outdcm
    # 定义一个方法 `_dcm`，用于处理两个帧之间的方向余弦矩阵 (DCM) 关系
    def _dcm(self, parent, parent_orient):
        # 如果 `parent` 已经在当前帧的 `_dcm_cache` 中存在
        # 更新 `parent` 的 `_dcm_dict`，并覆盖当前帧的 `_dcm_dict` 和 `_dcm_cache`
        # 以新的 DCM 关系来更新
        frames = self._dcm_cache.keys()

        # 用于记录需要删除的 `_dcm_dict` 和 `_dcm_cache` 中的帧
        dcm_dict_del = []
        dcm_cache_del = []

        if parent in frames:
            # 遍历当前帧的所有帧
            for frame in frames:
                # 如果当前帧在自身的 `_dcm_dict` 中存在
                if frame in self._dcm_dict:
                    dcm_dict_del += [frame]  # 记录需要删除的 `_dcm_dict` 中的帧
                dcm_cache_del += [frame]  # 记录需要删除的 `_dcm_cache` 中的帧

            # 重置当前帧的 `_dcm_dict`，并从所有链接的帧的 `_dcm_caches` 中移除当前帧
            for frame in dcm_dict_del:
                del frame._dcm_dict[self]

            # 清空当前帧的 `_dcm_cache`
            for frame in dcm_cache_del:
                del frame._dcm_cache[self]

            # 重置当前帧的 `_dcm_dict`
            self._dcm_dict = self._dlist[0] = {}

            # 清空当前帧的 `_dcm_cache`
            self._dcm_cache = {}

        else:
            # 检查是否存在循环引用，如果有则发出警告
            visited = []
            queue = list(frames)
            cont = True  # 控制循环的标志位
            while queue and cont:
                node = queue.pop(0)
                if node not in visited:
                    visited.append(node)
                    neighbors = node._dcm_dict.keys()
                    for neighbor in neighbors:
                        if neighbor == parent:
                            # 发出警告，指出帧之间存在循环引用
                            warn('Loops are defined among the orientation of '
                                 'frames. This is likely not desired and may '
                                 'cause errors in your calculations.')
                            cont = False  # 中断循环
                            break
                        queue.append(neighbor)

        # 将父帧 `parent` 的方向余弦矩阵 (DCM) 关系添加到当前帧的 `_dcm_dict` 中
        self._dcm_dict.update({parent: parent_orient.T})

        # 将当前帧的方向余弦矩阵 (DCM) 关系添加到父帧 `parent` 的 `_dcm_dict` 中
        parent._dcm_dict.update({self: parent_orient})

        # 更新当前帧的 `_dcm_cache`，添加父帧 `parent` 的方向余弦矩阵 (DCM) 关系
        self._dcm_cache.update({parent: parent_orient.T})

        # 更新父帧 `parent` 的 `_dcm_cache`，添加当前帧的方向余弦矩阵 (DCM) 关系
        parent._dcm_cache.update({self: parent_orient})
    def orient_axis(self, parent, axis, angle):
        """Sets the orientation of this reference frame with respect to a
        parent reference frame by rotating through an angle about an axis fixed
        in the parent reference frame.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        axis : Vector
            Vector fixed in the parent frame about about which this frame is
            rotated. It need not be a unit vector and the rotation follows the
            right hand rule.
        angle : sympifiable
            Angle in radians by which it the frame is to be rotated.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.orient_axis(N, N.x, q1)

        The ``orient_axis()`` method generates a direction cosine matrix and
        its transpose which defines the orientation of B relative to N and vice
        versa. Once orient is called, ``dcm()`` outputs the appropriate
        direction cosine matrix:

        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])
        >>> N.dcm(B)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The following two lines show that the sense of the rotation can be
        defined by negating the vector direction or the angle. Both lines
        produce the same result.

        >>> B.orient_axis(N, -N.x, q1)
        >>> B.orient_axis(N, N.x, -q1)

        """

        from sympy.physics.vector.functions import dynamicsymbols  # 导入动力学符号函数库

        _check_frame(parent)  # 检查父参考系的有效性

        if not isinstance(axis, Vector) and isinstance(angle, Vector):
            axis, angle = angle, axis

        axis = _check_vector(axis)  # 检查并修正向量轴
        theta = sympify(angle)  # 将角度符号化

        if not axis.dt(parent) == 0:
            raise ValueError('Axis cannot be time-varying.')  # 若轴向不是时间无关，则引发数值错误

        unit_axis = axis.express(parent).normalize()  # 在父参考系中表达并归一化轴向
        unit_col = unit_axis.args[0][0]  # 获取单位轴向的第一个参数
        parent_orient_axis = (
            (eye(3) - unit_col * unit_col.T) * cos(theta) +  # 计算父参考系中的方向余弦矩阵
            Matrix([[0, -unit_col[2], unit_col[1]],         # 使用单位轴向的矢量进行计算
                    [unit_col[2], 0, -unit_col[0]],
                    [-unit_col[1], unit_col[0], 0]]) *
            sin(theta) + unit_col * unit_col.T)

        self._dcm(parent, parent_orient_axis)  # 更新当前参考系的方向余弦矩阵

        thetad = (theta).diff(dynamicsymbols._t)  # 计算角速度
        wvec = thetad * axis.express(parent).normalize()  # 计算角速度向量
        self._ang_vel_dict.update({parent: wvec})  # 更新当前参考系的角速度字典
        parent._ang_vel_dict.update({self: -wvec})  # 更新父参考系的角速度字典
        self._var_dict = {}  # 清空当前参考系的变量字典
    def orient_explicit(self, parent, dcm):
        """
        Sets the orientation of this reference frame relative to another (parent) reference frame
        using a direction cosine matrix that describes the rotation from the parent to the child.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        dcm : Matrix, shape(3, 3)
            Direction cosine matrix that specifies the relative rotation
            between the two reference frames.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols, Matrix, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> N = ReferenceFrame('N')

        A simple rotation of ``A`` relative to ``N`` about ``N.x`` is defined
        by the following direction cosine matrix:

        >>> dcm = Matrix([[1, 0, 0],
        ...               [0, cos(q1), -sin(q1)],
        ...               [0, sin(q1), cos(q1)]])
        >>> A.orient_explicit(N, dcm)
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        This is equivalent to using ``orient_axis()``:

        >>> B.orient_axis(N, N.x, q1)
        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        **Note carefully that** ``N.dcm(B)`` **(the transpose) would be passed
        into** ``orient_explicit()`` **for** ``A.dcm(N)`` **to match**
        ``B.dcm(N)``:

        >>> A.orient_explicit(N, N.dcm(B))
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        """
        # 检查父参考系是否符合规范
        _check_frame(parent)
        # 确保 dcm 是 SymPy 中的 Matrix 类型对象
        if not isinstance(dcm, MatrixBase):
            raise TypeError("Amounts must be a SymPy Matrix type object.")

        # 调用内部方法 orient_dcm 将转置后的 dcm 传递给自身的 orient_dcm 方法
        self.orient_dcm(parent, dcm.T)
    def orient_dcm(self, parent, dcm):
        """Sets the orientation of this reference frame relative to another (parent) reference frame
        using a direction cosine matrix that describes the rotation from the child to the parent.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        dcm : Matrix, shape(3, 3)
            Direction cosine matrix that specifies the relative rotation
            between the two reference frames.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols, Matrix, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> N = ReferenceFrame('N')

        A simple rotation of ``A`` relative to ``N`` about ``N.x`` is defined
        by the following direction cosine matrix:

        >>> dcm = Matrix([[1, 0, 0],
        ...               [0,  cos(q1), sin(q1)],
        ...               [0, -sin(q1), cos(q1)]])
        >>> A.orient_dcm(N, dcm)
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        This is equivalent to using ``orient_axis()``:

        >>> B.orient_axis(N, N.x, q1)
        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        """

        _check_frame(parent)
        # 检查 dcm 是否为 SymPy 的 MatrixBase 类型对象
        if not isinstance(dcm, MatrixBase):
            raise TypeError("Amounts must be a SymPy Matrix type object.")

        # 将当前参考系相对于父参考系的方向余弦矩阵应用到当前参考系中
        self._dcm(parent, dcm.T)

        # 计算相对角速度向量并更新到当前参考系的角速度字典中
        wvec = self._w_diff_dcm(parent)
        self._ang_vel_dict.update({parent: wvec})
        # 更新父参考系的角速度字典，以反向存储当前参考系的角速度向量
        parent._ang_vel_dict.update({self: -wvec})
        # 清空当前参考系的变量字典
        self._var_dict = {}

    def _rot(self, axis, angle):
        """DCM for simple axis 1,2,or 3 rotations."""
        if axis == 1:
            # 返回绕 x 轴旋转给定角度的方向余弦矩阵
            return Matrix([[1, 0, 0],
                           [0, cos(angle), -sin(angle)],
                           [0, sin(angle), cos(angle)]])
        elif axis == 2:
            # 返回绕 y 轴旋转给定角度的方向余弦矩阵
            return Matrix([[cos(angle), 0, sin(angle)],
                           [0, 1, 0],
                           [-sin(angle), 0, cos(angle)]])
        elif axis == 3:
            # 返回绕 z 轴旋转给定角度的方向余弦矩阵
            return Matrix([[cos(angle), -sin(angle), 0],
                           [sin(angle), cos(angle), 0],
                           [0, 0, 1]])
    def _parse_consecutive_rotations(self, angles, rotation_order):
        """Helper for orient_body_fixed and orient_space_fixed.

        Parameters
        ==========
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations. The order can be specified by the strings
            ``'XZX'``, ``'131'``, or the integer ``131``. There are 12 unique
            valid rotation orders.

        Returns
        =======

        amounts : list
            List of sympifiables corresponding to the rotation angles.
        rot_order : list
            List of integers corresponding to the axis of rotation.
        rot_matrices : list
            List of DCM around the given axis with corresponding magnitude.

        """
        amounts = list(angles)  # 将输入的角度转换为列表形式
        for i, v in enumerate(amounts):
            if not isinstance(v, Vector):
                amounts[i] = sympify(v)  # 使用sympify处理每个角度，确保其为符号表达式

        approved_orders = ('123', '231', '312', '132', '213', '321', '121',
                           '131', '212', '232', '313', '323', '')
        # 确保旋转顺序字符串转换为标准形式 'XYZ' => '123'
        rot_order = translate(str(rotation_order), 'XYZxyz', '123123')
        if rot_order not in approved_orders:
            raise TypeError('The rotation order is not a valid order.')  # 如果旋转顺序不在允许的列表中，则抛出错误

        rot_order = [int(r) for r in rot_order]  # 将旋转顺序转换为整数列表
        if not (len(amounts) == 3 & len(rot_order) == 3):
            raise TypeError('Body orientation takes 3 values & 3 orders')  # 如果角度数量或旋转顺序数量不等于3，则抛出错误
        rot_matrices = [self._rot(order, amount)
                        for (order, amount) in zip(rot_order, amounts)]  # 使用给定的旋转顺序和角度创建旋转矩阵列表
        return amounts, rot_order, rot_matrices  # 返回处理后的角度列表、旋转顺序列表和旋转矩阵列表

    def set_ang_acc(self, otherframe, value):
        """Define the angular acceleration Vector in a ReferenceFrame.

        Defines the angular acceleration of this ReferenceFrame, in another.
        Angular acceleration can be defined with respect to multiple different
        ReferenceFrames. Care must be taken to not create loops which are
        inconsistent.

        Parameters
        ==========

        otherframe : ReferenceFrame
            A ReferenceFrame to define the angular acceleration in
        value : Vector
            The Vector representing angular acceleration

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_acc(N, V)
        >>> A.ang_acc_in(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)  # 如果值为零，则将其定义为零向量
        value = _check_vector(value)  # 确保值是一个向量
        _check_frame(otherframe)  # 检查帧对象的有效性
        self._ang_acc_dict.update({otherframe: value})  # 将另一个帧对象与其对应的角加速度向量更新到当前帧对象的角加速度字典中
        otherframe._ang_acc_dict.update({self: -value})  # 同时将当前帧对象与另一个帧对象的相反角加速度向量更新到另一个帧对象的角加速度字典中
    def set_ang_vel(self, otherframe, value):
        """Define the angular velocity vector in a ReferenceFrame.

        Defines the angular velocity of this ReferenceFrame, in another.
        Angular velocity can be defined with respect to multiple different
        ReferenceFrames. Care must be taken to not create loops which are
        inconsistent.

        Parameters
        ==========

        otherframe : ReferenceFrame
            A ReferenceFrame to define the angular velocity in
        value : Vector
            The Vector representing angular velocity

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_vel(N, V)
        >>> A.ang_vel_in(N)
        10*N.x

        """

        # 如果传入的角速度值为零，将其设为零向量
        if value == 0:
            value = Vector(0)
        # 确保传入的值是一个向量对象
        value = _check_vector(value)
        # 确保传入的参考系对象是有效的
        _check_frame(otherframe)
        # 更新当前参考系对象的角速度字典，以指定参考系为键，角速度向量为值
        self._ang_vel_dict.update({otherframe: value})
        # 同时更新传入参考系对象的角速度字典，以当前参考系为键，角速度向量的反向为值
        otherframe._ang_vel_dict.update({self: -value})

    @property
    def x(self):
        """The basis Vector for the ReferenceFrame, in the x direction. """
        return self._x

    @property
    def y(self):
        """The basis Vector for the ReferenceFrame, in the y direction. """
        return self._y

    @property
    def z(self):
        """The basis Vector for the ReferenceFrame, in the z direction. """
        return self._z

    @property
    def xx(self):
        """Unit dyad of basis Vectors x and x for the ReferenceFrame."""
        return Vector.outer(self.x, self.x)

    @property
    def xy(self):
        """Unit dyad of basis Vectors x and y for the ReferenceFrame."""
        return Vector.outer(self.x, self.y)

    @property
    def xz(self):
        """Unit dyad of basis Vectors x and z for the ReferenceFrame."""
        return Vector.outer(self.x, self.z)

    @property
    def yx(self):
        """Unit dyad of basis Vectors y and x for the ReferenceFrame."""
        return Vector.outer(self.y, self.x)

    @property
    def yy(self):
        """Unit dyad of basis Vectors y and y for the ReferenceFrame."""
        return Vector.outer(self.y, self.y)

    @property
    def yz(self):
        """Unit dyad of basis Vectors y and z for the ReferenceFrame."""
        return Vector.outer(self.y, self.z)

    @property
    def zx(self):
        """Unit dyad of basis Vectors z and x for the ReferenceFrame."""
        return Vector.outer(self.z, self.x)

    @property
    def zy(self):
        """Unit dyad of basis Vectors z and y for the ReferenceFrame."""
        return Vector.outer(self.z, self.y)

    @property
    def zz(self):
        """Unit dyad of basis Vectors z and z for the ReferenceFrame."""
        return Vector.outer(self.z, self.z)

    @property
    def u(self):
        """Unit dyadic for the ReferenceFrame."""
        return self.xx + self.yy + self.zz
    def partial_velocity(self, frame, *gen_speeds):
        """
        Returns the partial angular velocities of this frame in the given
        frame with respect to one or more provided generalized speeds.

        Parameters
        ==========
        frame : ReferenceFrame
            The frame with which the angular velocity is defined in.
        gen_speeds : functions of time
            The generalized speeds.

        Returns
        =======
        partial_velocities : tuple of Vector
            The partial angular velocity vectors corresponding to the provided
            generalized speeds.

        Examples
        ========
        
        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> A.set_ang_vel(N, u1 * A.x + u2 * N.y)
        >>> A.partial_velocity(N, u1)
        A.x
        >>> A.partial_velocity(N, u1, u2)
        (A.x, N.y)

        """

        from sympy.physics.vector.functions import partial_velocity
        
        # 获取该参考框架相对于给定广义速度的部分角速度
        vel = self.ang_vel_in(frame)
        # 计算部分角速度向量
        partials = partial_velocity([vel], gen_speeds, frame)[0]

        # 如果只有一个部分角速度，则返回单个向量
        if len(partials) == 1:
            return partials[0]
        else:
            # 否则返回多个部分角速度向量的元组
            return tuple(partials)
# 定义一个函数 `_check_frame`，用于检查参数 `other` 是否为 `ReferenceFrame` 类的实例
def _check_frame(other):
    # 从当前目录下的 `vector` 模块中导入 `VectorTypeError` 类
    from .vector import VectorTypeError
    # 如果 `other` 不是 `ReferenceFrame` 类的实例，则抛出 `VectorTypeError` 异常
    if not isinstance(other, ReferenceFrame):
        raise VectorTypeError(other, ReferenceFrame('A'))
```