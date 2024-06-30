# `D:\src\scipysrc\sympy\sympy\algebras\quaternion.py`

```
# 从 sympy 库中导入有理数 Rational
from sympy.core.numbers import Rational
# 从 sympy 库中导入单例 S
from sympy.core.singleton import S
# 从 sympy 库中导入关系运算 is_eq
from sympy.core.relational import is_eq
# 从 sympy 库中导入复数函数（如共轭、实部、虚部、符号等）
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
# 从 sympy 库中导入指数函数（如指数、对数）
from sympy.functions.elementary.exponential import (exp, log as ln)
# 从 sympy 库中导入杂项函数（如平方根）
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy 库中导入三角函数（如反余弦、反正弦、反正切）
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
# 从 sympy 库中导入三角函数（如余弦、正弦）
from sympy.functions.elementary.trigonometric import (cos, sin)
# 从 sympy 库中导入简化三角表达式的函数 trigsimp
from sympy.simplify.trigsimp import trigsimp
# 从 sympy 库中导入积分函数 integrate
from sympy.integrals.integrals import integrate
# 从 sympy 库中导入密集矩阵类 MutableDenseMatrix
from sympy.matrices.dense import MutableDenseMatrix as Matrix
# 从 sympy 库中导入符号表达式转换函数 sympify 和私有符号表达式转换函数 _sympify
from sympy.core.sympify import sympify, _sympify
# 从 sympy 库中导入表达式基类 Expr
from sympy.core.expr import Expr
# 从 sympy 库中导入逻辑运算函数（如模糊非、模糊或）
from sympy.core.logic import fuzzy_not, fuzzy_or
# 从 sympy 库中导入转换为整数的辅助函数 as_int

# 从 mpmath 库中导入转换精度到小数位数的函数 prec_to_dps
from mpmath.libmp.libmpf import prec_to_dps
    _op_priority = 11.0
    # 定义运算符优先级为 11.0

    is_commutative = False
    # 四元数不满足交换律，因此设置为 False

    def __new__(cls, a=0, b=0, c=0, d=0, real_field=True, norm=None):
        a, b, c, d = map(sympify, (a, b, c, d))
        # 将参数 a, b, c, d 转换为 sympy 对象

        if any(i.is_commutative is False for i in [a, b, c, d]):
            raise ValueError("arguments have to be commutative")
        # 如果任何一个参数不满足交换律，则抛出异常

        obj = super().__new__(cls, a, b, c, d)
        # 调用父类的构造方法创建对象

        obj._real_field = real_field
        # 设置是否为实数域的标志

        obj.set_norm(norm)
        # 设置四元数的范数

        return obj
        # 返回创建的四元数对象

    def set_norm(self, norm):
        """Sets norm of an already instantiated quaternion.

        Parameters
        ==========

        norm : None or number
            Pre-defined quaternion norm. If a value is given, Quaternion.norm
            returns this pre-defined value instead of calculating the norm

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion(a, b, c, d)
        >>> q.norm()
        sqrt(a**2 + b**2 + c**2 + d**2)

        Setting the norm:

        >>> q.set_norm(1)
        >>> q.norm()
        1

        Removing set norm:

        >>> q.set_norm(None)
        >>> q.norm()
        sqrt(a**2 + b**2 + c**2 + d**2)

        """
        norm = sympify(norm)
        # 将 norm 转换为 sympy 对象

        _check_norm(self.args, norm)
        # 调用 _check_norm 函数检查范数的合法性

        self._norm = norm
        # 设置四元数对象的范数属性

    @property
    def a(self):
        return self.args[0]
        # 返回四元数的第一个分量 a

    @property
    def b(self):
        return self.args[1]
        # 返回四元数的第二个分量 b

    @property
    def c(self):
        return self.args[2]
        # 返回四元数的第三个分量 c

    @property
    def d(self):
        return self.args[3]
        # 返回四元数的第四个分量 d

    @property
    def real_field(self):
        return self._real_field
        # 返回四元数是否在实数域内的标志
    # 返回一个 4x4 矩阵，表示从左侧进行 Hamilton 乘积的等效矩阵
    # 当将四元数元素视为列向量时，这个函数非常有用
    # 给定四元数 $q = a + bi + cj + dk$，其中 a, b, c 和 d 是实数，
    # 从左侧的乘积矩阵定义如下：
    #
    # M  =  \begin{bmatrix} a  &-b  &-c  &-d \\
    #                       b  & a  &-d  & c \\
    #                       c  & d  & a  &-b \\
    #                       d  &-c  & b  & a \end{bmatrix}
    #
    # 示例：
    # >>> from sympy import Quaternion
    # >>> from sympy.abc import a, b, c, d
    # >>> q1 = Quaternion(1, 0, 0, 1)
    # >>> q2 = Quaternion(a, b, c, d)
    # >>> q1.product_matrix_left
    # Matrix([
    # [1, 0,  0, -1],
    # [0, 1, -1,  0],
    # [0, 1,  1,  0],
    # [1, 0,  0,  1]])
    #
    # >>> q1.product_matrix_left * q2.to_Matrix()
    # Matrix([
    # [a - d],
    # [b - c],
    # [b + c],
    # [a + d]])
    #
    # 这等价于：
    # >>> (q1 * q2).to_Matrix()
    # Matrix([
    # [a - d],
    # [b - c],
    # [b + c],
    # [a + d]])
    def product_matrix_left(self):
        return Matrix([
                [self.a, -self.b, -self.c, -self.d],
                [self.b, self.a, -self.d, self.c],
                [self.c, self.d, self.a, -self.b],
                [self.d, -self.c, self.b, self.a]])
    def product_matrix_right(self):
        r"""Returns 4 x 4 Matrix equivalent to a Hamilton product from the
        right. This can be useful when treating quaternion elements as column
        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d
        are real numbers, the product matrix from the left is:

        .. math::

            M  =  \begin{bmatrix} a  &-b  &-c  &-d \\
                                  b  & a  & d  &-c \\
                                  c  &-d  & a  & b \\
                                  d  & c  &-b  & a \end{bmatrix}

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q1 = Quaternion(a, b, c, d)
        >>> q2 = Quaternion(1, 0, 0, 1)
        >>> q2.product_matrix_right
        Matrix([
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [1, 0, 0, 1]])

        Note the switched arguments: the matrix represents the quaternion on
        the right, but is still considered as a matrix multiplication from the
        left.

        >>> q2.product_matrix_right * q1.to_Matrix()
        Matrix([
        [ a - d],
        [ b + c],
        [-b + c],
        [ a + d]])

        This is equivalent to:

        >>> (q1 * q2).to_Matrix()
        Matrix([
        [ a - d],
        [ b + c],
        [-b + c],
        [ a + d]])

        """
        # 返回一个 4x4 的矩阵，表示从右侧进行哈密尔顿积的矩阵形式
        return Matrix([
                [self.a, -self.b, -self.c, -self.d],
                [self.b, self.a, self.d, -self.c],
                [self.c, -self.d, self.a, self.b],
                [self.d, self.c, -self.b, self.a]])

    def to_Matrix(self, vector_only=False):
        """Returns elements of quaternion as a column vector.
        By default, a ``Matrix`` of length 4 is returned, with the real part as the
        first element.
        If ``vector_only`` is ``True``, returns only imaginary part as a Matrix of
        length 3.

        Parameters
        ==========

        vector_only : bool
            If True, only imaginary part is returned.
            Default value: False

        Returns
        =======

        Matrix
            A column vector constructed by the elements of the quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion(a, b, c, d)
        >>> q
        a + b*i + c*j + d*k

        >>> q.to_Matrix()
        Matrix([
        [a],
        [b],
        [c],
        [d]])

        >>> q.to_Matrix(vector_only=True)
        Matrix([
        [b],
        [c],
        [d]])

        """
        # 如果 vector_only 为 True，则返回一个只包含虚部的长度为 3 的矩阵
        if vector_only:
            return Matrix(self.args[1:])
        else:
            # 否则返回一个包含全部四个元素的列向量矩阵
            return Matrix(self.args)
    def from_Matrix(cls, elements):
        """
        Returns quaternion from elements of a column vector.
        If vector_only is True, returns only imaginary part as a Matrix of
        length 3.

        Parameters
        ==========

        elements : Matrix, list or tuple of length 3 or 4. If length is 3,
            assume real part is zero.
            Default value: False

        Returns
        =======

        Quaternion
            A quaternion created from the input elements.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion.from_Matrix([a, b, c, d])
        >>> q
        a + b*i + c*j + d*k

        >>> q = Quaternion.from_Matrix([b, c, d])
        >>> q
        0 + b*i + c*j + d*k

        """
        # 获取输入元素列表的长度
        length = len(elements)
        # 如果长度既不是3也不是4，抛出值错误异常
        if length != 3 and length != 4:
            raise ValueError("Input elements must have length 3 or 4, got {} "
                             "elements".format(length))

        # 如果长度为3，返回一个以0为实部的四元数
        if length == 3:
            return Quaternion(0, *elements)
        # 否则，返回一个由输入元素构成的四元数
        else:
            return Quaternion(*elements)

    @classmethod
    def from_euler(cls, angles, seq):
        """Returns quaternion equivalent to rotation represented by the Euler
        angles, in the sequence defined by ``seq``.

        Parameters
        ==========

        angles : list, tuple or Matrix of 3 numbers
            The Euler angles (in radians).
        seq : string of length 3
            Represents the sequence of rotations.
            For extrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For intrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the Euler angles
            in the given sequence.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi
        >>> q = Quaternion.from_euler([pi/2, 0, 0], 'xyz')
        >>> q
        sqrt(2)/2 + sqrt(2)/2*i + 0*j + 0*k

        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'zyz')
        >>> q
        0 + (-sqrt(2)/2)*i + 0*j + sqrt(2)/2*k

        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'ZYZ')
        >>> q
        0 + sqrt(2)/2*i + 0*j + sqrt(2)/2*k

        """

        # 检查输入的角度列表是否为三个数值，否则引发 ValueError
        if len(angles) != 3:
            raise ValueError("3 angles must be given.")

        # 判断是否为外固定轴旋转序列
        extrinsic = _is_extrinsic(seq)
        
        # 解析旋转序列，获取对应的基向量
        i, j, k = seq.lower()
        
        # 获取基本单位向量
        ei = [1 if n == i else 0 for n in 'xyz']
        ej = [1 if n == j else 0 for n in 'xyz']
        ek = [1 if n == k else 0 for n in 'xyz']

        # 根据角度分别计算三个基向量对应的四元数
        qi = cls.from_axis_angle(ei, angles[0])
        qj = cls.from_axis_angle(ej, angles[1])
        qk = cls.from_axis_angle(ek, angles[2])

        # 根据是否为外固定轴旋转序列，计算最终的四元数
        if extrinsic:
            return trigsimp(qk * qj * qi)
        else:
            return trigsimp(qi * qj * qk)

    @classmethod


这段代码定义了一个类方法 `from_euler`，用于根据欧拉角计算旋转的四元数。
    def from_axis_angle(cls, vector, angle):
        """Returns a rotation quaternion given the axis and the angle of rotation.

        Parameters
        ==========

        vector : tuple of three numbers
            The vector representation of the given axis.
        angle : number
            The angle by which axis is rotated (in radians).

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the given axis and the angle of rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi, sqrt
        >>> q = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), 2*pi/3)
        >>> q
        1/2 + 1/2*i + 1/2*j + 1/2*k

        """
        (x, y, z) = vector   # 解构向量元组，获取向量的 x, y, z 分量
        norm = sqrt(x**2 + y**2 + z**2)   # 计算向量的模长
        (x, y, z) = (x / norm, y / norm, z / norm)   # 将向量归一化
        s = sin(angle * S.Half)   # 计算角度的半角正弦值
        a = cos(angle * S.Half)   # 计算角度的半角余弦值
        b = x * s   # 计算四元数的 b 分量
        c = y * s   # 计算四元数的 c 分量
        d = z * s   # 计算四元数的 d 分量

        # note that this quaternion is already normalized by construction:
        # c^2 + (s*x)^2 + (s*y)^2 + (s*z)^2 = c^2 + s^2*(x^2 + y^2 + z^2) = c^2 + s^2 * 1 = c^2 + s^2 = 1
        # so, what we return is a normalized quaternion
        # 此四元数在构造时已经归一化：
        # c^2 + (s*x)^2 + (s*y)^2 + (s*z)^2 = c^2 + s^2*(x^2 + y^2 + z^2) = c^2 + s^2 * 1 = c^2 + s^2 = 1
        # 因此，我们返回的是一个归一化的四元数

        return cls(a, b, c, d)

    @classmethod
    def from_rotation_matrix(cls, M):
        """Returns the equivalent quaternion of a matrix. The quaternion will be normalized
        only if the matrix is special orthogonal (orthogonal and det(M) = 1).

        Parameters
        ==========

        M : Matrix
            Input matrix to be converted to equivalent quaternion. M must be special
            orthogonal (orthogonal and det(M) = 1) for the quaternion to be normalized.

        Returns
        =======

        Quaternion
            The quaternion equivalent to given matrix.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Matrix, symbols, cos, sin, trigsimp
        >>> x = symbols('x')
        >>> M = Matrix([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])
        >>> q = trigsimp(Quaternion.from_rotation_matrix(M))
        >>> q
        sqrt(2)*sqrt(cos(x) + 1)/2 + 0*i + 0*j + sqrt(2 - 2*cos(x))*sign(sin(x))/2*k

        """

        absQ = M.det()**Rational(1, 3)   # 计算矩阵的行列式的三分之一次方

        a = sqrt(absQ + M[0, 0] + M[1, 1] + M[2, 2]) / 2   # 计算四元数的 a 分量
        b = sqrt(absQ + M[0, 0] - M[1, 1] - M[2, 2]) / 2   # 计算四元数的 b 分量
        c = sqrt(absQ - M[0, 0] + M[1, 1] - M[2, 2]) / 2   # 计算四元数的 c 分量
        d = sqrt(absQ - M[0, 0] - M[1, 1] + M[2, 2]) / 2   # 计算四元数的 d 分量

        b = b * sign(M[2, 1] - M[1, 2])   # 调整四元数的 b 分量符号
        c = c * sign(M[0, 2] - M[2, 0])   # 调整四元数的 c 分量符号
        d = d * sign(M[1, 0] - M[0, 1])   # 调整四元数的 d 分量符号

        return Quaternion(a, b, c, d)

    def __add__(self, other):
        return self.add(other)   # 调用实例方法 add 处理四元数的加法

    def __radd__(self, other):
        return self.add(other)   # 调用实例方法 add 处理四元数的反向加法

    def __sub__(self, other):
        return self.add(other*-1)   # 调用实例方法 add 处理四元数的减法

    def __mul__(self, other):
        return self._generic_mul(self, _sympify(other))   # 调用实例方法 _generic_mul 处理四元数的乘法
    # 定义魔术方法 __rmul__，支持右乘操作
    def __rmul__(self, other):
        return self._generic_mul(_sympify(other), self)

    # 定义魔术方法 __pow__，支持幂运算操作
    def __pow__(self, p):
        return self.pow(p)

    # 定义魔术方法 __neg__，支持取负操作
    def __neg__(self):
        return Quaternion(-self.a, -self.b, -self.c, -self.d)

    # 定义魔术方法 __truediv__，支持真除操作
    def __truediv__(self, other):
        return self * sympify(other)**-1

    # 定义魔术方法 __rtruediv__，支持右真除操作
    def __rtruediv__(self, other):
        return sympify(other) * self**-1

    # 定义私有方法 _eval_Integral，用于计算积分
    def _eval_Integral(self, *args):
        return self.integrate(*args)

    # 定义 diff 方法，用于计算导数
    def diff(self, *symbols, **kwargs):
        kwargs.setdefault('evaluate', True)
        return self.func(*[a.diff(*symbols, **kwargs) for a  in self.args])

    # 定义 add 方法，实现四元数的加法
    def add(self, other):
        """Adds quaternions.

        Parameters
        ==========

        other : Quaternion
            要加到当前（self）四元数上的四元数。

        Returns
        =======

        Quaternion
            加法操作后的结果四元数

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.add(q2)
        6 + 8*i + 10*j + 12*k
        >>> q1 + 5
        6 + 2*i + 3*j + 4*k
        >>> x = symbols('x', real = True)
        >>> q1.add(x)
        (x + 1) + 2*i + 3*j + 4*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.add(2 + 3*I)
        (5 + 7*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

        """
        q1 = self
        q2 = sympify(other)

        # 如果 q2 是数值或 SymPy 表达式而不是四元数
        if not isinstance(q2, Quaternion):
            # 如果 q1 是实数域且 q2 是复数
            if q1.real_field and q2.is_complex:
                return Quaternion(re(q2) + q1.a, im(q2) + q1.b, q1.c, q1.d)
            # 如果 q2 是可交换的普通表达式
            elif q2.is_commutative:
                return Quaternion(q1.a + q2, q1.b, q1.c, q1.d)
            else:
                raise ValueError("Only commutative expressions can be added with a Quaternion.")

        # 返回两个四元数相加的结果
        return Quaternion(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d + q2.d)
    # 定义一个方法用于乘法运算，用于计算四元数的乘法结果
    def mul(self, other):
        """Multiplies quaternions.

        Parameters
        ==========

        other : Quaternion or symbol
            The quaternion to multiply to current (self) quaternion.
            要乘到当前四元数（self）的四元数或符号。

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying self with other
            与其他四元数相乘后得到的结果四元数。

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.mul(q2)
        (-60) + 12*i + 30*j + 24*k
        >>> q1.mul(2)
        2 + 4*i + 6*j + 8*k
        >>> x = symbols('x', real = True)
        >>> q1.mul(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.mul(2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        # 调用内部方法 _generic_mul 对当前四元数和参数进行乘法运算
        return self._generic_mul(self, _sympify(other))
    def _generic_mul(q1, q2):
        """Generic multiplication.

        Parameters
        ==========

        q1 : Quaternion or symbol
            The first quaternion or symbol to multiply.
        q2 : Quaternion or symbol
            The second quaternion or symbol to multiply.

        It is important to note that if neither q1 nor q2 is a Quaternion,
        this function simply returns q1 * q2.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying q1 and q2.

        Raises
        ======

        ValueError
            Raised if non-commutative expressions are multiplied with a Quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Symbol, S
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> Quaternion._generic_mul(q1, q2)
        (-60) + 12*i + 30*j + 24*k
        >>> Quaternion._generic_mul(q1, S(2))
        2 + 4*i + 6*j + 8*k
        >>> x = Symbol('x', real=True)
        >>> Quaternion._generic_mul(q1, x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields:

        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field=False)
        >>> Quaternion._generic_mul(q3, 2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        # None is a Quaternion:
        if not isinstance(q1, Quaternion) and not isinstance(q2, Quaternion):
            return q1 * q2

        # If q1 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q1, Quaternion):
            if q2.real_field and q1.is_complex:
                return Quaternion(re(q1), im(q1), 0, 0) * q2
            elif q1.is_commutative:
                return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If q2 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return q1 * Quaternion(re(q2), im(q2), 0, 0)
            elif q2.is_commutative:
                return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If any of the quaternions has a fixed norm, pre-compute norm
        if q1._norm is None and q2._norm is None:
            norm = None
        else:
            norm = q1.norm() * q2.norm()

        return Quaternion(-q1.b*q2.b - q1.c*q2.c - q1.d*q2.d + q1.a*q2.a,
                          q1.b*q2.a + q1.c*q2.d - q1.d*q2.c + q1.a*q2.b,
                          -q1.b*q2.d + q1.c*q2.a + q1.d*q2.b + q1.a*q2.c,
                          q1.b*q2.c - q1.c*q2.b + q1.d*q2.a + q1.a * q2.d,
                          norm=norm)


    def _eval_conjugate(self):
        """Returns the conjugate of the quaternion.

        Returns
        =======

        Quaternion
            The conjugate quaternion, with negated imaginary components.

        """
        q = self
        return Quaternion(q.a, -q.b, -q.c, -q.d, norm=q._norm)
    def norm(self):
        """Returns the norm of the quaternion."""
        # 如果尚未计算过范数，则计算并返回四元数的范数
        if self._norm is None:  # check if norm is pre-defined
            q = self
            # trigsimp 用于简化 sin(x)^2 + cos(x)^2（当使用 from_axis_angle 时会出现这些项）
            return sqrt(trigsimp(q.a**2 + q.b**2 + q.c**2 + q.d**2))

        # 如果已经预定义了范数，则直接返回预定义的范数
        return self._norm

    def normalize(self):
        """Returns the normalized form of the quaternion."""
        q = self
        # 返回四元数的单位化形式
        return q * (1/q.norm())

    def inverse(self):
        """Returns the inverse of the quaternion."""
        q = self
        # 如果四元数的范数为零，则无法计算其逆
        if not q.norm():
            raise ValueError("Cannot compute inverse for a quaternion with zero norm")
        # 返回四元数的逆
        return conjugate(q) * (1/q.norm()**2)

    def pow(self, p):
        """Finds the pth power of the quaternion.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            Returns the p-th power of the current quaternion.
            Returns the inverse if p = -1.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow(4)
        668 + (-224)*i + (-336)*j + (-448)*k

        """
        try:
            q, p = self, as_int(p)
        except ValueError:
            return NotImplemented

        # 如果指数 p 为负数，先计算四元数的逆，再计算其绝对值的指数
        if p < 0:
            q, p = q.inverse(), -p

        # 如果指数 p 为 1，直接返回四元数本身
        if p == 1:
            return q

        # 初始化结果为单位四元数
        res = Quaternion(1, 0, 0, 0)
        # 使用快速幂算法计算四元数的 p 次幂
        while p > 0:
            if p & 1:  # 如果 p 是奇数，乘以当前的 q
                res *= q
            q *= q  # 平方 q
            p >>= 1  # p 右移一位，相当于 p // 2

        return res

    def exp(self):
        """Returns the exponential of $q$, given by $e^q$.

        Returns
        =======

        Quaternion
            The exponential of the quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.exp()
        E*cos(sqrt(29))
        + 2*sqrt(29)*E*sin(sqrt(29))/29*i
        + 3*sqrt(29)*E*sin(sqrt(29))/29*j
        + 4*sqrt(29)*E*sin(sqrt(29))/29*k

        """
        # 计算四元数的指数函数 exp(q) = e^a(cos||v|| + v/||v||*sin||v||)
        q = self
        # 计算四元数向量部分的范数
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        # 分别计算指数函数的实部和虚部
        a = exp(q.a) * cos(vector_norm)
        b = exp(q.a) * sin(vector_norm) * q.b / vector_norm
        c = exp(q.a) * sin(vector_norm) * q.c / vector_norm
        d = exp(q.a) * sin(vector_norm) * q.d / vector_norm

        # 返回结果四元数
        return Quaternion(a, b, c, d)
    def log(self):
        r"""Returns the logarithm of the quaternion, given by $\log q$.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.log()
        log(sqrt(30))
        + 2*sqrt(29)*acos(sqrt(30)/30)/29*i
        + 3*sqrt(29)*acos(sqrt(30)/30)/29*j
        + 4*sqrt(29)*acos(sqrt(30)/30)/29*k

        """
        # 获取当前实例的四元数对象
        q = self
        # 计算向量的范数
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        # 计算四元数的范数
        q_norm = q.norm()
        # 计算标量部分的自然对数
        a = ln(q_norm)
        # 计算向量部分的每个分量
        b = q.b * acos(q.a / q_norm) / vector_norm
        c = q.c * acos(q.a / q_norm) / vector_norm
        d = q.d * acos(q.a / q_norm) / vector_norm

        # 返回新的四元数对象，表示对当前四元数的对数运算
        return Quaternion(a, b, c, d)

    def _eval_subs(self, *args):
        # 对每个元素执行符号代换
        elements = [i.subs(*args) for i in self.args]
        # 获取当前实例的范数对象
        norm = self._norm
        # 如果范数存在，则对其进行符号代换
        if norm is not None:
            norm = norm.subs(*args)
        # 检查替换后的元素和范数是否满足条件
        _check_norm(elements, norm)
        # 返回带有替换后元素和范数的新的四元数对象
        return Quaternion(*elements, norm=norm)

    def _eval_evalf(self, prec):
        """Returns the floating point approximations (decimal numbers) of the quaternion.

        Returns
        =======

        Quaternion
            Floating point approximations of quaternion(self)

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import sqrt
        >>> q = Quaternion(1/sqrt(1), 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))
        >>> q.evalf()
        1.00000000000000
        + 0.707106781186547*i
        + 0.577350269189626*j
        + 0.500000000000000*k

        """
        # 将每个参数的浮点数近似值返回为四元数对象
        nprec = prec_to_dps(prec)
        return Quaternion(*[arg.evalf(n=nprec) for arg in self.args])

    def pow_cos_sin(self, p):
        """Computes the pth power in the cos-sin form.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            The p-th power in the cos-sin form.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow_cos_sin(4)
        900*cos(4*acos(sqrt(30)/30))
        + 1800*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*i
        + 2700*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*j
        + 3600*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*k

        """
        # 将四元数表示为单位轴和角度的形式
        (v, angle) = self.to_axis_angle()
        # 计算新的四元数对象，表示四元数的p次幂
        q2 = Quaternion.from_axis_angle(v, p * angle)
        # 返回结果四元数对象
        return q2 * (self.norm()**p)
    def integrate(self, *args):
        """Computes integration of quaternion.

        Returns
        =======
        Quaternion
            Integration of the quaternion(self) with the given variable.

        Examples
        =======
        Indefinite Integral of quaternion :

        >>> from sympy import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Definite integral of quaternion :

        >>> from sympy import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate((x, 1, 5))
        4 + 8*i + 12*j + 16*k

        """
        # TODO: is this expression correct?
        # 对四元数进行积分，返回积分后的四元数
        return Quaternion(integrate(self.a, *args), integrate(self.b, *args),
                          integrate(self.c, *args), integrate(self.d, *args))

    @staticmethod
    def rotate_point(pin, r):
        """Returns the coordinates of the point pin (a 3 tuple) after rotation.

        Parameters
        ==========
        pin : tuple
            A 3-element tuple of coordinates of a point which needs to be
            rotated.
        r : Quaternion or tuple
            Axis and angle of rotation.

            It's important to note that when r is a tuple, it must be of the form
            (axis, angle)

        Returns
        =======
        tuple
            The coordinates of the point after rotation.

        Examples
        =======
        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), q))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), (axis, angle)))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)

        """
        if isinstance(r, tuple):
            # 如果 r 是形如 (向量, 角度) 的元组
            q = Quaternion.from_axis_angle(r[0], r[1])
        else:
            # 如果 r 是一个四元数
            q = r.normalize()
        # 对点进行四元数旋转，并返回旋转后的坐标
        pout = q * Quaternion(0, pin[0], pin[1], pin[2]) * conjugate(q)
        return (pout.b, pout.c, pout.d)
    def to_axis_angle(self):
        """Returns the axis and angle of rotation of a quaternion.
        
        Returns
        =======
        
        tuple
            Tuple of (axis, angle)
        
        Examples
        ========
        
        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> axis
        (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)
        >>> angle
        2*pi/3
        
        """
        q = self  # 将当前对象赋给变量 q

        # 如果四元数的实部为负数，则取其相反数
        if q.a.is_negative:
            q = q * -1
        
        q = q.normalize()  # 规范化四元数
        
        angle = trigsimp(2 * acos(q.a))  # 计算旋转角度
        
        # 由于四元数已经被规范化，q.a 应小于 1
        s = sqrt(1 - q.a*q.a)  # 计算 sin(θ/2) 的值，其中 θ 是旋转角度

        # 计算旋转轴的三个分量
        x = trigsimp(q.b / s)
        y = trigsimp(q.c / s)
        z = trigsimp(q.d / s)

        v = (x, y, z)  # 将旋转轴的三个分量组成一个元组
        t = (v, angle)  # 将旋转轴和角度组成一个元组

        return t  # 返回包含旋转轴和角度的元组
    def to_rotation_matrix(self, v=None, homogeneous=True):
        """Returns the equivalent rotation transformation matrix of the quaternion
        which represents rotation about the origin if ``v`` is not passed.

        Parameters
        ==========

        v : tuple or None
            Default value: None
            Specifies the point around which rotation occurs if provided.
        homogeneous : bool
            When True, gives an expression potentially more efficient for
            symbolic calculations but less so for direct evaluation. Both
            formulas are mathematically equivalent.
            Default value: True

        Returns
        =======

        Matrix
            Returns a 3x3 or 4x4 rotation transformation matrix depending on
            whether ``v`` is None or not.

        Examples
        ========

        >>> from sympy import Quaternion, symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(q.to_rotation_matrix())
        Matrix([
        [cos(x), -sin(x), 0],
        [sin(x),  cos(x), 0],
        [     0,       0, 1]])

        Generates a 4x4 transformation matrix (used for rotation about a point
        other than the origin) if the point(v) is passed as an argument.
        """

        q = self
        s = q.norm()**-2

        # Calculate matrix elements based on the quaternion components
        if homogeneous:
            # Homogeneous transformation matrix
            m00 = s*(q.a**2 + q.b**2 - q.c**2 - q.d**2)
            m11 = s*(q.a**2 - q.b**2 + q.c**2 - q.d**2)
            m22 = s*(q.a**2 - q.b**2 - q.c**2 + q.d**2)
        else:
            # Non-homogeneous transformation matrix
            m00 = 1 - 2*s*(q.c**2 + q.d**2)
            m11 = 1 - 2*s*(q.b**2 + q.d**2)
            m22 = 1 - 2*s*(q.b**2 + q.c**2)

        m01 = 2*s*(q.b*q.c - q.d*q.a)
        m02 = 2*s*(q.b*q.d + q.c*q.a)

        m10 = 2*s*(q.b*q.c + q.d*q.a)
        m12 = 2*s*(q.c*q.d - q.b*q.a)

        m20 = 2*s*(q.b*q.d - q.c*q.a)
        m21 = 2*s*(q.c*q.d + q.b*q.a)

        if not v:
            # Return a 3x3 rotation matrix if v is None (rotation about origin)
            return Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
        else:
            # Compute a 4x4 transformation matrix for rotation about point v
            (x, y, z) = v

            m03 = x - x*m00 - y*m01 - z*m02
            m13 = y - x*m10 - y*m11 - z*m12
            m23 = z - x*m20 - y*m21 - z*m22
            m30 = m31 = m32 = 0
            m33 = 1

            return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13],
                          [m20, m21, m22, m23], [m30, m31, m32, m33]])

    def scalar_part(self):
        r"""Returns scalar part($\mathbf{S}(q)$) of the quaternion q.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{S}(q) = a$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(4, 8, 13, 12)
        >>> q.scalar_part()
        4

        """

        return self.a
    def vector_part(self):
        r"""
        Returns $\mathbf{V}(q)$, the vector part of the quaternion $q$.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{V}(q) = bi + cj + dk$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.vector_part()
        0 + 1*i + 1*j + 1*k

        >>> q = Quaternion(4, 8, 13, 12)
        >>> q.vector_part()
        0 + 8*i + 13*j + 12*k

        """

        # 返回一个新的四元数对象，代表输入四元数的向量部分
        return Quaternion(0, self.b, self.c, self.d)

    def axis(self):
        r"""
        Returns $\mathbf{Ax}(q)$, the axis of the quaternion $q$.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{Ax}(q)$ i.e., the versor of the vector part of that quaternion
        equal to $\mathbf{U}[\mathbf{V}(q)]$.
        The axis is always an imaginary unit with square equal to $-1 + 0i + 0j + 0k$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.axis()
        0 + sqrt(3)/3*i + sqrt(3)/3*j + sqrt(3)/3*k

        See Also
        ========

        vector_part

        """
        # 计算四元数的向量部分并归一化，返回一个新的四元数对象代表轴
        axis = self.vector_part().normalize()
        return Quaternion(0, axis.b, axis.c, axis.d)

    def is_pure(self):
        """
        Returns true if the quaternion is pure, false if the quaternion is not pure
        or returns none if it is unknown.

        Explanation
        ===========

        A pure quaternion (also a vector quaternion) is a quaternion with scalar
        part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 8, 13, 12)
        >>> q.is_pure()
        True

        See Also
        ========
        scalar_part

        """

        # 检查四元数的标量部分是否为零，返回布尔值
        return self.a.is_zero

    def is_zero_quaternion(self):
        """
        Returns true if the quaternion is a zero quaternion or false if it is not a zero quaternion
        and None if the value is unknown.

        Explanation
        ===========

        A zero quaternion is a quaternion with both scalar part and
        vector part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 0, 0, 0)
        >>> q.is_zero_quaternion()
        False

        >>> q = Quaternion(0, 0, 0, 0)
        >>> q.is_zero_quaternion()
        True

        See Also
        ========
        scalar_part
        vector_part

        """

        # 检查四元数的范数是否为零，返回布尔值
        return self.norm().is_zero
    def angle(self):
        r"""
        Returns the angle of the quaternion measured in the real-axis plane.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$ where $a$, $b$, $c$ and $d$
        are real numbers, returns the angle of the quaternion given by

        .. math::
            \theta := 2 \operatorname{atan_2}\left(\sqrt{b^2 + c^2 + d^2}, {a}\right)

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 4, 4, 4)
        >>> q.angle()
        2*atan(4*sqrt(3))

        """

        # 计算并返回四元数在实轴平面上的角度
        return 2 * atan2(self.vector_part().norm(), self.scalar_part())


    def arc_coplanar(self, other):
        """
        Returns True if the transformation arcs represented by the input quaternions happen in the same plane.

        Explanation
        ===========

        Two quaternions are said to be coplanar (in this arc sense) when their axes are parallel.
        The plane of a quaternion is the one normal to its axis.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the planes of the two quaternions are the same, apart from its orientation/sign.
        False : if the planes of the two quaternions are not the same, apart from its orientation/sign.
        None : if plane of either of the quaternion is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(1, 4, 4, 4)
        >>> q2 = Quaternion(3, 8, 8, 8)
        >>> Quaternion.arc_coplanar(q1, q2)
        True

        >>> q1 = Quaternion(2, 8, 13, 12)
        >>> Quaternion.arc_coplanar(q1, q2)
        False

        See Also
        ========

        vector_coplanar
        is_pure

        """
        # 检查两个四元数的转换弧是否在同一平面上
        if (self.is_zero_quaternion()) or (other.is_zero_quaternion()):
            raise ValueError('Neither of the given quaternions can be 0')

        # 使用模糊逻辑判断两个四元数的轴是否平行
        return fuzzy_or([(self.axis() - other.axis()).is_zero_quaternion(), (self.axis() + other.axis()).is_zero_quaternion()])

    @classmethod
    def vector_coplanar(cls, q1, q2, q3):
        r"""
        Returns True if the axis of the pure quaternions seen as 3D vectors
        ``q1``, ``q2``, and ``q3`` are coplanar.

        Explanation
        ===========

        Three pure quaternions are vector coplanar if the quaternions seen as 3D vectors are coplanar.

        Parameters
        ==========

        q1
            A pure Quaternion.
        q2
            A pure Quaternion.
        q3
            A pure Quaternion.

        Returns
        =======

        True : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar.
        False : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are not coplanar.
        None : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(0, 4, 4, 4)
        >>> q2 = Quaternion(0, 8, 8, 8)
        >>> q3 = Quaternion(0, 24, 24, 24)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        True

        >>> q1 = Quaternion(0, 8, 16, 8)
        >>> q2 = Quaternion(0, 8, 3, 12)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        False

        See Also
        ========

        axis
        is_pure

        """

        # Check if any of the input quaternions are not pure
        if fuzzy_not(q1.is_pure()) or fuzzy_not(q2.is_pure()) or fuzzy_not(q3.is_pure()):
            raise ValueError('The given quaternions must be pure')

        # Compute the determinant of the matrix formed by the vector parts of q1, q2, and q3
        M = Matrix([[q1.b, q1.c, q1.d], [q2.b, q2.c, q2.d], [q3.b, q3.c, q3.d]]).det()
        # Check if the determinant is zero, indicating coplanarity
        return M.is_zero

    def parallel(self, other):
        """
        Returns True if the two pure quaternions seen as 3D vectors are parallel.

        Explanation
        ===========

        Two pure quaternions are called parallel when their vector product is commutative which
        implies that the quaternions seen as 3D vectors have the same direction.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the two pure quaternions seen as 3D vectors are parallel.
        False : if the two pure quaternions seen as 3D vectors are not parallel.
        None : if the two pure quaternions seen as 3D vectors are parallel is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 4, 4, 4)
        >>> q1 = Quaternion(0, 8, 8, 8)
        >>> q.parallel(q1)
        True

        >>> q1 = Quaternion(0, 8, 13, 12)
        >>> q.parallel(q1)
        False

        """

        # Check if self and other are pure quaternions
        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The provided quaternions must be pure')

        # Check if the quaternion product (self * other - other * self) is zero quaternion
        return (self * other - other * self).is_zero_quaternion()
    def orthogonal(self, other):
        """
        Returns the orthogonality of two quaternions.

        Explanation
        ===========

        Two pure quaternions are called orthogonal when their product is anti-commutative.

        Parameters
        ==========

        other : a Quaternion
            Another quaternion object to check orthogonality with.

        Returns
        =======

        True : if the two pure quaternions seen as 3D vectors are orthogonal.
        False : if the two pure quaternions seen as 3D vectors are not orthogonal.
        None : if the orthogonality status is indeterminate.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 4, 4, 4)
        >>> q1 = Quaternion(0, 8, 8, 8)
        >>> q.orthogonal(q1)
        False

        >>> q1 = Quaternion(0, 2, 2, 0)
        >>> q = Quaternion(0, 2, -2, 0)
        >>> q.orthogonal(q1)
        True

        """

        # Check if either quaternion is not pure (contains scalar part)
        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The given quaternions must be pure')

        # Evaluate orthogonality using the anti-commutativity property of quaternions
        return (self*other + other*self).is_zero_quaternion()

    def index_vector(self):
        r"""
        Returns the index vector of the quaternion.

        Explanation
        ===========

        The index vector is given by $\mathbf{T}(q)$, the norm (or magnitude) of
        the quaternion $q$, multiplied by $\mathbf{Ax}(q)$, the axis of $q$.

        Returns
        =======

        Quaternion: representing index vector of the provided quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.index_vector()
        0 + 4*sqrt(10)/3*i + 2*sqrt(10)/3*j + 4*sqrt(10)/3*k

        See Also
        ========

        axis
        norm

        """

        # Compute and return the index vector using quaternion properties
        return self.norm() * self.axis()

    def mensor(self):
        """
        Returns the natural logarithm of the norm(magnitude) of the quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.mensor()
        log(2*sqrt(10))
        >>> q.norm()
        2*sqrt(10)

        See Also
        ========

        norm

        """

        # Calculate and return the natural logarithm of the quaternion's norm
        return ln(self.norm())
```