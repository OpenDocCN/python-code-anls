# `D:\src\scipysrc\sympy\sympy\physics\vector\vector.py`

```
# 导入必要的模块和函数，从sympy库中导入多个具体函数和类
from sympy import (S, sympify, expand, sqrt, Add, zeros, acos,
                   ImmutableMatrix as Matrix, simplify)
# 从sympy.simplify.trigsimp模块中导入trigsimp函数
from sympy.simplify.trigsimp import trigsimp
# 从sympy.printing.defaults模块中导入Printable类
from sympy.printing.defaults import Printable
# 从sympy.utilities.misc模块中导入filldedent函数
from sympy.utilities.misc import filldedent
# 从sympy.core.evalf模块中导入EvalfMixin类
from sympy.core.evalf import EvalfMixin
# 从mpmath.libmp.libmpf模块中导入prec_to_dps函数

from mpmath.libmp.libmpf import prec_to_dps

# 定义此模块中可导出的所有内容，只有Vector类是可导出的
__all__ = ['Vector']


# 定义Vector类，它继承了Printable类和EvalfMixin类
class Vector(Printable, EvalfMixin):
    """The class used to define vectors.

    It along with ReferenceFrame are the building blocks of describing a
    classical mechanics system in PyDy and sympy.physics.vector.

    Attributes
    ==========

    simp : Boolean
        Let certain methods use trigsimp on their outputs

    """

    # 类属性simp，用于控制是否在某些方法中使用trigsimp函数简化输出
    simp = False
    # 类属性is_number，标记此类实例是否为数值
    is_number = False

    # Vector类的初始化方法，接受一个inlist参数
    def __init__(self, inlist):
        """This is the constructor for the Vector class. You should not be
        calling this, it should only be used by other functions. You should be
        treating Vectors like you would with if you were doing the math by
        hand, and getting the first 3 from the standard basis vectors from a
        ReferenceFrame.

        The only exception is to create a zero vector:
        zv = Vector(0)

        """

        # 初始化实例属性args为一个空列表
        self.args = []
        # 如果inlist为0，则将inlist设为空列表
        if inlist == 0:
            inlist = []
        # 如果inlist为字典类型，则直接赋值给d
        if isinstance(inlist, dict):
            d = inlist
        else:
            # 否则创建一个空字典d，遍历inlist中的元素进行处理
            d = {}
            for inp in inlist:
                if inp[1] in d:
                    d[inp[1]] += inp[0]
                else:
                    d[inp[1]] = inp[0]

        # 将处理后的键值对加入到self.args中，每个元素是一个元组(v, k)
        for k, v in d.items():
            if v != Matrix([0, 0, 0]):
                self.args.append((v, k))

    # func属性的getter方法，返回Vector类本身
    @property
    def func(self):
        """Returns the class Vector. """
        return Vector

    # 定义Vector类的哈希方法，返回args元组的哈希值
    def __hash__(self):
        return hash(tuple(self.args))

    # 定义Vector类的加法运算符重载方法
    def __add__(self, other):
        """The add operator for Vector. """
        # 如果other为0，则返回self，即返回原向量
        if other == 0:
            return self
        # 否则调用_check_vector函数检查other是否为Vector类的实例
        other = _check_vector(other)
        # 返回一个新的Vector实例，其args属性为self.args与other.args的合并
        return Vector(self.args + other.args)
    def dot(self, other):
        """Calculate the dot product of two vectors.

        Returns a scalar, which is the dot product of the current Vector instance and another Vector instance.

        Parameters
        ==========

        other : Vector
            The Vector object with which the dot product is calculated.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dot
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> dot(N.x, N.x)
        1
        >>> dot(N.x, N.y)
        0
        >>> A = N.orientnew('A', 'Axis', [q1, N.x])
        >>> dot(N.y, A.y)
        cos(q1)

        """

        # Import necessary components for Dyadic handling
        from sympy.physics.vector.dyadic import Dyadic, _check_dyadic

        # Check if 'other' is a Dyadic object, and convert it if necessary
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Vector(0)
            # Iterate over components of the Dyadic object
            for v in other.args:
                # Calculate contribution to dot product from each component
                ol += v[0] * v[2] * (v[1].dot(self))
            return ol

        # If 'other' is not a Dyadic object, proceed with vector dot product calculation
        other = _check_vector(other)
        out = S.Zero
        # Iterate over components of the current vector
        for v1 in self.args:
            # Iterate over components of 'other' vector
            for v2 in other.args:
                # Compute dot product contribution and accumulate
                out += ((v2[0].T) * (v2[1].dcm(v1[1])) * (v1[0]))[0]

        # Simplify the result if Vector.simp is set to True
        if Vector.simp:
            return trigsimp(out, recursive=True)
        else:
            return out

    def __truediv__(self, other):
        """Divides the Vector by a sympifyable expression.

        This method divides each component of the Vector by the given scalar expression.

        Parameters
        ==========

        other : Sympifyable
            The scalar expression by which to divide the Vector.

        """

        # Use multiplication method to divide by the reciprocal of 'other'
        return self.__mul__(S.One / other)

    def __eq__(self, other):
        """Checks equality between Vectors.

        Tests for equality between the current Vector and 'other'. Handles zero Vectors and checks all components.

        Parameters
        ==========

        other : Vector
            The Vector to compare with.

        """

        # If 'other' is zero, treat it as an empty Vector
        if other == 0:
            other = Vector(0)

        try:
            other = _check_vector(other)
        except TypeError:
            return False

        # Compare component-wise equality of Vectors
        if (self.args == []) and (other.args == []):
            return True
        elif (self.args == []) or (other.args == []):
            return False

        frame = self.args[0][1]
        # Iterate over base vectors of the reference frame and check equality condition
        for v in frame:
            if expand((self - other).dot(v)) != 0:
                return False
        return True

    def __mul__(self, other):
        """Multiplies the Vector by a sympifyable expression.

        Multiplies each component of the Vector by the given scalar expression.

        Parameters
        ==========

        other : Sympifyable
            The scalar expression to multiply the Vector by.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> b = Symbol('b')
        >>> V = 10 * b * N.x
        >>> print(V)
        10*b*N.x

        """

        # Create a new list of vector components
        newlist = list(self.args)
        other = sympify(other)
        # Multiply each component by 'other'
        for i, v in enumerate(newlist):
            newlist[i] = (other * newlist[i][0], newlist[i][1])
        return Vector(newlist)

    def __neg__(self):
        """Negates the Vector.

        Negates each component of the Vector.

        """
        # Multiply each component of the Vector by -1
        return self * -1
    # 外积操作，计算两个向量的外积，返回一个 Dyadic 对象
    def outer(self, other):
        """Outer product between two Vectors.

        A rank increasing operation, which returns a Dyadic from two Vectors

        Parameters
        ==========

        other : Vector
            The Vector to take the outer product with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> outer(N.x, N.x)
        (N.x|N.x)

        """

        from sympy.physics.vector.dyadic import Dyadic
        other = _check_vector(other)  # 确保 other 是一个有效的向量对象
        ol = Dyadic(0)  # 初始化一个 Dyadic 对象 ol
        for v in self.args:
            for v2 in other.args:
                # 外积公式的展开，将每个分量相乘，并加入到 Dyadic 对象 ol 中
                ol += Dyadic([(v[0][0] * v2[0][0], v[1].x, v2[1].x)])  # 外积的 x 分量
                ol += Dyadic([(v[0][0] * v2[0][1], v[1].x, v2[1].y)])  # 外积的 y 分量
                ol += Dyadic([(v[0][0] * v2[0][2], v[1].x, v2[1].z)])  # 外积的 z 分量
                ol += Dyadic([(v[0][1] * v2[0][0], v[1].y, v2[1].x)])  # 外积的 x 分量
                ol += Dyadic([(v[0][1] * v2[0][1], v[1].y, v2[1].y)])  # 外积的 y 分量
                ol += Dyadic([(v[0][1] * v2[0][2], v[1].y, v2[1].z)])  # 外积的 z 分量
                ol += Dyadic([(v[0][2] * v2[0][0], v[1].z, v2[1].x)])  # 外积的 x 分量
                ol += Dyadic([(v[0][2] * v2[0][1], v[1].z, v2[1].y)])  # 外积的 y 分量
                ol += Dyadic([(v[0][2] * v2[0][2], v[1].z, v2[1].z)])  # 外积的 z 分量
        return ol  # 返回计算得到的 Dyadic 对象

    # LaTeX 输出方法
    def _latex(self, printer):
        """Latex Printing method. """

        ar = self.args  # 缩短 ar 变量名称，方便后续使用
        if len(ar) == 0:
            return str(0)  # 如果参数列表为空，返回字符串 '0'
        ol = []  # 输出列表，用于构建最终的 LaTeX 字符串
        for i, v in enumerate(ar):
            for j in 0, 1, 2:
                # 如果基向量的系数为 1，则省略 1
                if ar[i][0][j] == 1:
                    ol.append(' + ' + ar[i][1].latex_vecs[j])
                # 如果基向量的系数为 -1，则省略 1
                elif ar[i][0][j] == -1:
                    ol.append(' - ' + ar[i][1].latex_vecs[j])
                elif ar[i][0][j] != 0:
                    # 如果基向量的系数既不是 1 也不是 -1，则构建带括号的字符串以提高可读性
                    arg_str = printer._print(ar[i][0][j])
                    if isinstance(ar[i][0][j], Add):
                        arg_str = "(%s)" % arg_str
                    if arg_str[0] == '-':
                        arg_str = arg_str[1:]
                        str_start = ' - '
                    else:
                        str_start = ' + '
                    ol.append(str_start + arg_str + ar[i][1].latex_vecs[j])
        outstr = ''.join(ol)  # 将输出列表中的所有元素连接成一个字符串
        if outstr.startswith(' + '):
            outstr = outstr[3:]  # 如果字符串以 ' + ' 开头，则去掉前面的 ' + '
        elif outstr.startswith(' '):
            outstr = outstr[1:]  # 如果字符串以空格开头，则去掉第一个空格
        return outstr  # 返回构建好的 LaTeX 字符串
    def _pretty(self, printer):
        """Pretty Printing method. """

        # 导入 SymPy 的 prettyForm 类
        from sympy.printing.pretty.stringpict import prettyForm

        # 初始化一个空列表，用于存储打印结果的条目
        terms = []

        # 定义一个内部函数 juxtapose，用于打印两个对象并按指定格式连接它们
        def juxtapose(a, b):
            # 使用打印机对象打印 a 和 b
            pa = printer._print(a)
            pb = printer._print(b)
            # 如果 a 是加法对象，则给 a 加上括号
            if a.is_Add:
                pa = prettyForm(*pa.parens())
            # 返回用指定分隔符连接的打印结果
            return printer._print_seq([pa, pb], delimiter=' ')

        # 遍历 self.args 中的每对 M 和 N
        for M, N in self.args:
            # 遍历每个索引 i（0 到 2）
            for i in range(3):
                # 如果 M[i] 等于 0，则跳过当前循环
                if M[i] == 0:
                    continue
                # 如果 M[i] 等于 1，则将 N.pretty_vecs[i] 加入 terms 列表
                elif M[i] == 1:
                    terms.append(prettyForm(N.pretty_vecs[i]))
                # 如果 M[i] 等于 -1，则将 "-1" * N.pretty_vecs[i] 加入 terms 列表
                elif M[i] == -1:
                    terms.append(prettyForm("-1") * prettyForm(N.pretty_vecs[i]))
                # 否则，使用 juxtapose 函数将 M[i] 和 N.pretty_vecs[i] 打印，并加入 terms 列表
                else:
                    terms.append(juxtapose(M[i], N.pretty_vecs[i]))

        # 返回将所有 terms 中的打印结果连接起来的 prettyForm 对象
        return prettyForm.__add__(*terms)

    def __rsub__(self, other):
        """The reflected subtraction operator. """

        # 返回反向加法操作的结果
        return (-1 * self) + other

    def _sympystr(self, printer, order=True):
        """Printing method. """

        # 如果不需要排序或只有一个参数，则直接将参数列表转换为列表 ar
        if not order or len(self.args) == 1:
            ar = list(self.args)
        # 如果没有参数，则返回打印机对象打印的数字 0
        elif len(self.args) == 0:
            return printer._print(0)
        else:
            # 否则，创建一个字典 d，将参数列表中的键值对按照索引排序
            d = {v[1]: v[0] for v in self.args}
            keys = sorted(d.keys(), key=lambda x: x.index)
            ar = []
            # 遍历排序后的键，并将其添加到 ar 列表中
            for key in keys:
                ar.append((d[key], key))

        # 初始化一个空列表 ol，用于存储要连接成字符串的输出
        ol = []

        # 遍历参数列表 ar 中的每个元素 v
        for i, v in enumerate(ar):
            # 遍历向量的每个索引 j（0 到 2）
            for j in 0, 1, 2:
                # 如果基向量的系数为 1，则跳过 " + " 并添加基向量的字符串表示
                if ar[i][0][j] == 1:
                    ol.append(' + ' + ar[i][1].str_vecs[j])
                # 如果基向量的系数为 -1，则跳过 " - " 并添加基向量的字符串表示
                elif ar[i][0][j] == -1:
                    ol.append(' - ' + ar[i][1].str_vecs[j])
                # 如果基向量的系数不为 0
                elif ar[i][0][j] != 0:
                    # 使用打印机对象打印基向量的系数
                    arg_str = printer._print(ar[i][0][j])
                    # 如果系数是加法对象，则在打印结果外围加上括号
                    if isinstance(ar[i][0][j], Add):
                        arg_str = "(%s)" % arg_str
                    # 如果打印结果以 "-" 开头，则去掉 "-"，并添加 " - "
                    if arg_str[0] == '-':
                        arg_str = arg_str[1:]
                        str_start = ' - '
                    # 否则，添加 " + "
                    else:
                        str_start = ' + '
                    # 将系数、乘号和基向量的字符串表示添加到 ol 列表
                    ol.append(str_start + arg_str + '*' + ar[i][1].str_vecs[j])

        # 将 ol 列表中的所有字符串连接起来
        outstr = ''.join(ol)

        # 如果字符串以 " + " 开头，则去掉前面的 " + "
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        # 如果字符串以 " " 开头，则去掉前面的空格
        elif outstr.startswith(' '):
            outstr = outstr[1:]

        # 返回连接好的字符串表示
        return outstr

    def __sub__(self, other):
        """The subtraction operator. """

        # 返回加法操作 self + (-1 * other) 的结果
        return self.__add__(other * -1)
    def cross(self, other):
        """The cross product operator for two Vectors.

        Returns a Vector, expressed in the same ReferenceFrames as self.

        Parameters
        ==========

        other : Vector
            The Vector which we are crossing with

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame, cross
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> cross(N.x, N.y)
        N.z
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, q1, N.x)
        >>> cross(A.x, N.y)
        N.z
        >>> cross(N.y, A.x)
        - sin(q1)*A.y - cos(q1)*A.z

        """

        from sympy.physics.vector.dyadic import Dyadic, _check_dyadic
        # 检查 other 是否为 Dyadic 类型，如果是，则转换为 Dyadic 对象
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Dyadic(0)
            # 对 Dyadic 对象进行处理，计算交叉乘积
            for i, v in enumerate(other.args):
                # 计算交叉乘积的结果，并累加到 ol 中
                ol += v[0] * ((self.cross(v[1])).outer(v[2]))
            return ol
        # 如果 other 不是 Dyadic 类型，则将其视为 Vector 类型
        other = _check_vector(other)
        # 如果 other 没有成分，则返回零向量
        if other.args == []:
            return Vector(0)

        def _det(mat):
            """This is needed as a little method for to find the determinant
            of a list in python; needs to work for a 3x3 list.
            SymPy's Matrix will not take in Vector, so need a custom function.
            You should not be calling this.

            """
            # 计算给定 3x3 矩阵的行列式
            return (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
                    + mat[0][1] * (mat[1][2] * mat[2][0] - mat[1][0] *
                    mat[2][2]) + mat[0][2] * (mat[1][0] * mat[2][1] -
                    mat[1][1] * mat[2][0]))

        outlist = []
        ar = other.args  # For brevity
        # 对 other 中的每个向量成分进行处理
        for i, v in enumerate(ar):
            tempx = v[1].x
            tempy = v[1].y
            tempz = v[1].z
            # 构造一个 3x3 的矩阵并计算其行列式，将结果添加到 outlist 中
            tempm = ([[tempx, tempy, tempz],
                      [self.dot(tempx), self.dot(tempy), self.dot(tempz)],
                      [Vector([ar[i]]).dot(tempx), Vector([ar[i]]).dot(tempy),
                       Vector([ar[i]]).dot(tempz)]])
            outlist += _det(tempm).args
        return Vector(outlist)

    __radd__ = __add__  # 右加法的运算符重载，与 __add__ 方法等效
    __rmul__ = __mul__  # 右乘法的运算符重载，与 __mul__ 方法等效

    def separate(self):
        """
        The constituents of this vector in different reference frames,
        as per its definition.

        Returns a dict mapping each ReferenceFrame to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> R1 = ReferenceFrame('R1')
        >>> R2 = ReferenceFrame('R2')
        >>> v = R1.x + R2.x
        >>> v.separate() == {R1: R1.x, R2: R2.x}
        True

        """

        components = {}
        # 将向量中每个成分映射到对应的参考系，存储在 components 字典中
        for x in self.args:
            components[x[1]] = Vector([x])
        return components

    def __and__(self, other):
        return self.dot(other)
    __and__.__doc__ = dot.__doc__  # 将 __and__ 方法的文档字符串设置为 dot 方法的文档字符串
    __rand__ = __and__  # 右与运算符重载，与 __and__ 方法等效
    def __xor__(self, other):
        # 定义 XOR 运算符，调用 cross 方法
        return self.cross(other)
    # 设置 XOR 方法的文档字符串为 cross 方法的文档字符串
    __xor__.__doc__ = cross.__doc__

    def __or__(self, other):
        # 定义 OR 运算符，调用 outer 方法
        return self.outer(other)
    # 设置 OR 方法的文档字符串为 outer 方法的文档字符串
    __or__.__doc__ = outer.__doc__

    def diff(self, var, frame, var_in_dcm=True):
        """Returns the partial derivative of the vector with respect to a
        variable in the provided reference frame.

        Parameters
        ==========
        var : Symbol
            What the partial derivative is taken with respect to.
        frame : ReferenceFrame
            The reference frame that the partial derivative is taken in.
        var_in_dcm : boolean
            If true, the differentiation algorithm assumes that the variable
            may be present in any of the direction cosine matrices that relate
            the frame to the frames of any component of the vector. But if it
            is known that the variable is not present in the direction cosine
            matrices, false can be set to skip full reexpression in the desired
            frame.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.vector import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> t = Symbol('t')
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.diff(t, N)
        - sin(q1)*q1'*N.x - cos(q1)*q1'*N.z
        >>> A.x.diff(t, N).express(A).simplify()
        - q1'*A.z
        >>> B = ReferenceFrame('B')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> v = u1 * A.x + u2 * B.y
        >>> v.diff(u2, N, var_in_dcm=False)
        B.y

        """

        from sympy.physics.vector.frame import _check_frame

        # 检查参考系是否为合法的 ReferenceFrame 对象
        _check_frame(frame)
        # 将 var 转换为 SymPy 的符号对象
        var = sympify(var)

        inlist = []

        for vector_component in self.args:
            measure_number = vector_component[0]
            component_frame = vector_component[1]
            if component_frame == frame:
                # 如果组件的参考系与给定参考系相同，直接加入对应的偏导数
                inlist += [(measure_number.diff(var), frame)]
            else:
                # 如果组件的参考系与给定参考系不同
                # 检查是否需要在方向余弦矩阵中重新表达
                if not var_in_dcm or (frame.dcm(component_frame).diff(var) ==
                                      zeros(3, 3)):
                    # 如果不需要在方向余弦矩阵中重新表达，则直接加入对应的偏导数
                    inlist += [(measure_number.diff(var), component_frame)]
                else:
                    # 否则，在所需参考系中重新表达向量组分
                    reexp_vec_comp = Vector([vector_component]).express(frame)
                    deriv = reexp_vec_comp.args[0][0].diff(var)
                    inlist += Vector([(deriv, frame)]).args

        # 返回包含偏导数信息的新的向量对象
        return Vector(inlist)
    def express(self, otherframe, variables=False):
        """
        Returns a Vector equivalent to this one, expressed in otherframe.
        Uses the global express method.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame for this Vector to be described in

        variables : boolean
            If True, the coordinate symbols(if present) in this Vector
            are re-expressed in terms otherframe

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.express(N)
        cos(q1)*N.x - sin(q1)*N.z

        """
        # 调用全局的 express 方法，返回在 otherframe 中描述的等效向量
        from sympy.physics.vector import express
        return express(self, otherframe, variables=variables)

    def to_matrix(self, reference_frame):
        """Returns the matrix form of the vector with respect to the given
        frame.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows of the matrix correspond to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,1)
            The matrix that gives the 1D vector.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> a, b, c = symbols('a, b, c')
        >>> N = ReferenceFrame('N')
        >>> vector = a * N.x + b * N.y + c * N.z
        >>> vector.to_matrix(N)
        Matrix([
        [a],
        [b],
        [c]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> vector.to_matrix(A)
        Matrix([
        [                         a],
        [ b*cos(beta) + c*sin(beta)],
        [-b*sin(beta) + c*cos(beta)]])

        """
        # 返回向量相对于给定参考系的矩阵形式
        return Matrix([self.dot(unit_vec) for unit_vec in
                       reference_frame]).reshape(3, 1)

    def doit(self, **hints):
        """Calls .doit() on each term in the Vector"""
        # 对向量中的每一项调用 .doit() 方法，并返回处理后的向量
        d = {}
        for v in self.args:
            d[v[1]] = v[0].applyfunc(lambda x: x.doit(**hints))
        return Vector(d)

    def dt(self, otherframe):
        """
        Returns a Vector which is the time derivative of
        the self Vector, taken in frame otherframe.

        Calls the global time_derivative method

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame to calculate the time derivative in

        """
        # 调用全局的 time_derivative 方法，返回在 otherframe 中的时间导数向量
        from sympy.physics.vector import time_derivative
        return time_derivative(self, otherframe)

    def simplify(self):
        """Returns a simplified Vector."""
        # 返回简化后的向量
        d = {}
        for v in self.args:
            d[v[1]] = simplify(v[0])
        return Vector(d)
    def subs(self, *args, **kwargs):
        """Substitute values into the components of the Vector.

        This method replaces symbols with given values in the components
        of the Vector.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = N.x * s
        >>> a.subs({s: 2})
        2*N.x

        """

        # Initialize an empty dictionary to store substitutions
        d = {}
        # Iterate over the components of the vector
        for v in self.args:
            # Perform substitution on each component
            d[v[1]] = v[0].subs(*args, **kwargs)
        # Return a new Vector object with substituted components
        return Vector(d)

    def magnitude(self):
        """Returns the magnitude (Euclidean norm) of the Vector.

        Warnings
        ========

        Python ignores the leading negative sign so that might
        give wrong results.
        ``-A.x.magnitude()`` would be treated as ``-(A.x.magnitude())``,
        instead of ``(-A.x).magnitude()``.

        """
        # Calculate and return the square root of the dot product of the Vector with itself
        return sqrt(self.dot(self))

    def normalize(self):
        """Returns a Vector of magnitude 1, codirectional with self."""
        # Create a normalized Vector by dividing each component by its magnitude
        return Vector(self.args + []) / self.magnitude()

    def applyfunc(self, f):
        """Apply a function to each component of the Vector."""
        # Ensure the input function is callable
        if not callable(f):
            raise TypeError("`f` must be callable.")

        # Initialize an empty dictionary to store results of applying `f` to components
        d = {}
        # Iterate over the components of the Vector
        for v in self.args:
            # Apply function `f` to the component and store the result
            d[v[1]] = v[0].applyfunc(f)
        # Return a new Vector object with components transformed by `f`
        return Vector(d)

    def angle_between(self, vec):
        """
        Returns the smallest angle between this Vector and Vector 'vec'.

        Parameter
        =========
        vec : Vector
            The Vector between which angle is needed.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame("A")
        >>> v1 = A.x
        >>> v2 = A.y
        >>> v1.angle_between(v2)
        pi/2

        >>> v3 = A.x + A.y + A.z
        >>> v1.angle_between(v3)
        acos(sqrt(3)/3)

        Warnings
        ========

        Python ignores the leading negative sign so that might give wrong
        results. ``-A.x.angle_between()`` would be treated as
        ``-(A.x.angle_between())``, instead of ``(-A.x).angle_between()``.

        """
        # Normalize both vectors
        vec1 = self.normalize()
        vec2 = vec.normalize()
        # Calculate and return the angle between the normalized vectors
        angle = acos(vec1.dot(vec2))
        return angle

    def free_symbols(self, reference_frame):
        """Returns the free symbols in the measure numbers of the Vector
        expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free symbols of the given
            Vector is to be determined.

        Returns
        =======
        set of Symbol
            set of symbols present in the measure numbers of
            `reference_frame`.

        """
        # Convert the Vector into a matrix representation in the given reference frame
        # and return the free symbols in its measure numbers
        return self.to_matrix(reference_frame).free_symbols
    def free_dynamicsymbols(self, reference_frame):
        """Returns the free dynamic symbols (functions of time `t`) in the
        measure numbers of the vector expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free dynamic symbols of the
            given vector are to be determined.

        Returns
        =======
        set
            Set of functions of time `t`, e.g.
            `Function('f')(me.dynamicsymbols._t)`.

        """
        # 从 sympy.physics.mechanics.functions 中导入 find_dynamicsymbols 函数
        # 由于可能会存在循环依赖问题，应将 find_dynamicsymbols 移至 physics.vector.functions 中
        from sympy.physics.mechanics.functions import find_dynamicsymbols

        # 调用 find_dynamicsymbols 函数，返回在给定参考系中表示的向量的自由动态符号（时间函数）
        return find_dynamicsymbols(self, reference_frame=reference_frame)

    def _eval_evalf(self, prec):
        # 如果参数列表为空，则直接返回自身
        if not self.args:
            return self
        new_args = []
        # 将精度 prec 转换为小数点位数 dps
        dps = prec_to_dps(prec)
        # 遍历 self.args 中的每对 mat 和 frame
        for mat, frame in self.args:
            # 对 mat 中的每个元素应用 evalf 函数，使用给定的小数点位数 dps
            new_args.append([mat.evalf(n=dps), frame])
        # 返回新的 Vector 对象，其中每个 mat 都已经被 evalf 处理过
        return Vector(new_args)

    def xreplace(self, rule):
        """Replace occurrences of objects within the measure numbers of the
        vector.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Vector
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame('A')
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * A.x).xreplace({x: pi})
        (pi*y + 1)*A.x
        >>> ((1 + x*y) * A.x).xreplace({x: pi, y: 2})
        (1 + 2*pi)*A.x

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * A.x).xreplace({x*y: pi})
        (z + pi)*A.x
        >>> ((x*y*z) * A.x).xreplace({x*y: pi})
        x*y*z*A.x

        """

        new_args = []
        # 遍历 self.args 中的每对 mat 和 frame
        for mat, frame in self.args:
            # 对 mat 中的每个元素应用 xreplace 函数，使用给定的替换规则 rule
            mat = mat.xreplace(rule)
            new_args.append([mat, frame])
        # 返回新的 Vector 对象，其中每个 mat 都已经应用了 xreplace 函数
        return Vector(new_args)
# 定义一个自定义异常类 VectorTypeError，继承自 TypeError
class VectorTypeError(TypeError):

    # 初始化方法，接受参数 other 和 want
    def __init__(self, other, want):
        # 构造异常信息的消息内容，描述期望接收的类型和实际接收的对象的类型
        msg = filldedent("Expected an instance of %s, but received object "
                         "'%s' of %s." % (type(want), other, type(other)))
        # 调用父类的初始化方法，传入消息内容
        super().__init__(msg)


# 定义一个辅助函数 _check_vector，用于检查参数 other 是否是 Vector 类的实例
def _check_vector(other):
    # 如果 other 不是 Vector 类的实例，则抛出 TypeError 异常
    if not isinstance(other, Vector):
        raise TypeError('A Vector must be supplied')
    # 如果是 Vector 类的实例，则直接返回 other
    return other
```