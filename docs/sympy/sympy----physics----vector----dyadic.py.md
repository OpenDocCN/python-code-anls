# `D:\src\scipysrc\sympy\sympy\physics\vector\dyadic.py`

```
# 从 sympy 库中导入 sympify, Add, Matrix 等必要的类和函数
from sympy import sympify, Add, ImmutableMatrix as Matrix
# 从 sympy.core.evalf 中导入 EvalfMixin 类
from sympy.core.evalf import EvalfMixin
# 从 sympy.printing.defaults 中导入 Printable 类
from sympy.printing.defaults import Printable
# 从 mpmath.libmp.libmpf 中导入 prec_to_dps 函数
from mpmath.libmp.libmpf import prec_to_dps

# 定义 __all__ 列表，用于模块级导入控制
__all__ = ['Dyadic']

# 定义 Dyadic 类，继承 Printable 和 EvalfMixin 类
class Dyadic(Printable, EvalfMixin):
    """A Dyadic object.

    See:
    https://en.wikipedia.org/wiki/Dyadic_tensor
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    A more powerful way to represent a rigid body's inertia. While it is more
    complex, by choosing Dyadic components to be in body fixed basis vectors,
    the resulting matrix is equivalent to the inertia tensor.

    """

    is_number = False  # 设置类属性 is_number 为 False，表示 Dyadic 对象不是数值

    def __init__(self, inlist):
        """
        Just like Vector's init, you should not call this unless creating a
        zero dyadic.

        zd = Dyadic(0)

        Stores a Dyadic as a list of lists; the inner list has the measure
        number and the two unit vectors; the outerlist holds each unique
        unit vector pair.

        """

        self.args = []  # 初始化实例变量 args 为空列表
        if inlist == 0:
            inlist = []  # 如果传入参数 inlist 为 0，则将其设为空列表
        while len(inlist) != 0:
            added = 0
            for i, v in enumerate(self.args):
                if ((str(inlist[0][1]) == str(self.args[i][1])) and
                        (str(inlist[0][2]) == str(self.args[i][2]))):
                    # 如果 inlist 的第一个元素的第二个和第三个元素与 args 中的对应元素相同
                    self.args[i] = (self.args[i][0] + inlist[0][0],  # 将两者的第一个元素相加
                                    inlist[0][1], inlist[0][2])  # 更新 args 中的对应元素
                    inlist.remove(inlist[0])  # 移除已处理的 inlist 的第一个元素
                    added = 1  # 设置标志位 added 为 1
                    break
            if added != 1:
                self.args.append(inlist[0])  # 如果未添加，则将 inlist 的第一个元素添加到 args
                inlist.remove(inlist[0])  # 移除已处理的 inlist 的第一个元素
        i = 0
        # 清除 args 中数值为 0 的元素
        while i < len(self.args):
            if ((self.args[i][0] == 0) | (self.args[i][1] == 0) |
                    (self.args[i][2] == 0)):
                self.args.remove(self.args[i])
                i -= 1
            i += 1

    @property
    def func(self):
        """Returns the class Dyadic. """
        return Dyadic  # 返回 Dyadic 类本身作为 func 属性的值

    def __add__(self, other):
        """The add operator for Dyadic. """
        other = _check_dyadic(other)  # 检查并获取正确的 Dyadic 对象
        return Dyadic(self.args + other.args)  # 返回两个 Dyadic 对象相加后的新 Dyadic 对象

    __radd__ = __add__  # 右加法运算符与左加法运算符相同

    def __mul__(self, other):
        """Multiplies the Dyadic by a sympifyable expression.

        Parameters
        ==========

        other : Sympafiable
            The scalar to multiply this Dyadic with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> 5 * d
        5*(N.x|N.x)

        """
        newlist = list(self.args)  # 复制 self.args 到 newlist
        other = sympify(other)  # 将 other 转换为 sympy 可识别的表达式
        for i, v in enumerate(newlist):
            newlist[i] = (other * newlist[i][0], newlist[i][1],
                          newlist[i][2])  # 将 newlist 中每个元素的第一个元素乘以 other
        return Dyadic(newlist)  # 返回乘以 other 后的新 Dyadic 对象

    __rmul__ = __mul__  # 右乘法运算符与左乘法运算符相同
    # 内积运算符，用于 Dyadic 和 Dyadic 或 Vector 的内积计算

    # 参数：
    # other : Dyadic 或 Vector
    #     用于计算内积的另一个 Dyadic 或 Vector 对象

    # 示例：
    # >>> from sympy.physics.vector import ReferenceFrame, outer
    # >>> N = ReferenceFrame('N')
    # >>> D1 = outer(N.x, N.y)
    # >>> D2 = outer(N.y, N.y)
    # >>> D1.dot(D2)
    # (N.x|N.y)
    # >>> D1.dot(N.y)
    # N.x
    def dot(self, other):
        from sympy.physics.vector.vector import Vector, _check_vector

        # 如果 other 是 Dyadic 类型，则转换为 Dyadic 对象
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Dyadic(0)
            # 遍历 self 和 other 的成员
            for v in self.args:
                for v2 in other.args:
                    # 计算内积的部分和，将结果累加到 ol 中
                    ol += v[0] * v2[0] * (v[2].dot(v2[1])) * (v[1].outer(v2[2]))
        else:
            # 如果 other 是 Vector 类型，则转换为 Vector 对象
            other = _check_vector(other)
            ol = Vector(0)
            # 遍历 self 的成员
            for v in self.args:
                # 计算内积的部分和，将结果累加到 ol 中
                ol += v[0] * v[1] * (v[2].dot(other))
        # 返回计算结果
        return ol

    # NOTE : 支持未公开的 Dyadic & Dyadic，Dyadic & Vector 表示法
    __and__ = dot

    # 将 Dyadic 对象按 sympifyable 表达式进行除法运算
    def __truediv__(self, other):
        return self.__mul__(1 / other)

    # 测试是否相等
    def __eq__(self, other):
        # 如果 other 是 0，则转换为 Dyadic 对象
        if other == 0:
            other = Dyadic(0)
        other = _check_dyadic(other)
        # 比较 self 和 other 的成员集合是否相等
        if (self.args == []) and (other.args == []):
            return True
        elif (self.args == []) or (other.args == []):
            return False
        return set(self.args) == set(other.args)

    # 测试是否不相等
    def __ne__(self, other):
        return not self == other

    # 返回 Dyadic 对象的负值
    def __neg__(self):
        return self * -1
    # 定义一个方法，用于生成 LaTeX 表示
    def _latex(self, printer):
        ar = self.args  # 将 self.args 赋值给 ar，简化操作
        # 如果 ar 为空列表，则返回字符串 '0'
        if len(ar) == 0:
            return str(0)
        ol = []  # 创建一个空列表，用于存储输出的每一部分，最终将其连接成一个字符串
        # 遍历 ar 列表中的元素，i 是索引，v 是元素
        for i, v in enumerate(ar):
            # 如果 dyadic 的系数为 1，则省略 '1'，直接连接下一个项
            if ar[i][0] == 1:
                ol.append(' + ' + printer._print(ar[i][1]) + r"\otimes " +
                          printer._print(ar[i][2]))
            # 如果 dyadic 的系数为 -1，则省略 '-1'，直接连接下一个项
            elif ar[i][0] == -1:
                ol.append(' - ' +
                          printer._print(ar[i][1]) +
                          r"\otimes " +
                          printer._print(ar[i][2]))
            # 如果 dyadic 的系数既不是 1 也不是 -1，
            # 可能将其用括号括起来以增加可读性
            elif ar[i][0] != 0:
                arg_str = printer._print(ar[i][0])
                if isinstance(ar[i][0], Add):
                    arg_str = '(%s)' % arg_str
                if arg_str.startswith('-'):
                    arg_str = arg_str[1:]
                    str_start = ' - '
                else:
                    str_start = ' + '
                ol.append(str_start + arg_str + printer._print(ar[i][1]) +
                          r"\otimes " + printer._print(ar[i][2]))
        outstr = ''.join(ol)  # 将列表 ol 中的所有字符串连接成一个字符串
        if outstr.startswith(' + '):
            outstr = outstr[3:]  # 如果字符串以 ' + ' 开头，则去除前面的 '+ '
        elif outstr.startswith(' '):
            outstr = outstr[1:]  # 如果字符串以空格开头，则去除第一个字符空格
        return outstr  # 返回生成的 LaTeX 字符串
    # 定义一个内部方法 `_pretty`，接受一个打印机对象 `printer` 作为参数
    def _pretty(self, printer):
        # 将当前对象保存在变量 e 中
        e = self

        # 定义一个内部类 Fake
        class Fake:
            # 设定 baseline 属性为 0
            baseline = 0

            # 定义 render 方法，用于渲染输出
            def render(self, *args, **kwargs):
                # 缩短代码中的 e.args，保存在 ar 变量中
                ar = e.args
                # 缩短代码中的 printer，保存在 mpp 变量中
                mpp = printer

                # 如果参数列表 ar 的长度为 0，则返回字符串 "0"
                if len(ar) == 0:
                    return str(0)

                # 如果 printer 使用 Unicode，则设置 bar 为 "⭕"，否则为 "|"
                bar = "\N{CIRCLED TIMES}" if printer._use_unicode else "|"

                # 创建一个空列表 ol，用于存储最终输出的各部分字符串
                ol = []

                # 遍历参数列表 ar 中的元素，i 是索引，v 是值
                for i, v in enumerate(ar):
                    # 如果 dyadic 的系数 ar[i][0] 为 1，则跳过系数为 1
                    if ar[i][0] == 1:
                        ol.extend([" + ",  # 添加 " + "
                                  mpp.doprint(ar[i][1]),  # 打印 ar[i][1] 的字符串表示
                                  bar,  # 添加分隔符 bar
                                  mpp.doprint(ar[i][2])])  # 打印 ar[i][2] 的字符串表示

                    # 如果 dyadic 的系数 ar[i][0] 为 -1，则跳过系数为 -1
                    elif ar[i][0] == -1:
                        ol.extend([" - ",  # 添加 " - "
                                  mpp.doprint(ar[i][1]),  # 打印 ar[i][1] 的字符串表示
                                  bar,  # 添加分隔符 bar
                                  mpp.doprint(ar[i][2])])  # 打印 ar[i][2] 的字符串表示

                    # 如果 dyadic 的系数 ar[i][0] 不为 0
                    elif ar[i][0] != 0:
                        # 如果系数是 Add 类型，则在打印时可能需要加上括号以提高可读性
                        if isinstance(ar[i][0], Add):
                            arg_str = mpp._print(
                                ar[i][0]).parens()[0]  # 使用 printer 打印并添加括号
                        else:
                            arg_str = mpp.doprint(ar[i][0])  # 打印 ar[i][0] 的字符串表示

                        # 如果打印的字符串以 "-" 开头，则去掉负号，并设置 str_start 为 " - "
                        if arg_str.startswith("-"):
                            arg_str = arg_str[1:]
                            str_start = " - "
                        else:
                            str_start = " + "  # 否则设置 str_start 为 " + "

                        # 将拼接好的字符串依次加入 ol 列表中
                        ol.extend([str_start, arg_str, " ",  # 添加符号和系数字符串
                                  mpp.doprint(ar[i][1]),  # 打印 ar[i][1] 的字符串表示
                                  bar,  # 添加分隔符 bar
                                  mpp.doprint(ar[i][2])])  # 打印 ar[i][2] 的字符串表示

                # 将列表 ol 中的所有部分连接成一个字符串 outstr
                outstr = "".join(ol)

                # 如果 outstr 以 " + " 开头，则去掉这部分
                if outstr.startswith(" + "):
                    outstr = outstr[3:]
                # 如果 outstr 以 " " 开头，则去掉这部分
                elif outstr.startswith(" "):
                    outstr = outstr[1:]

                # 返回拼接好的输出字符串 outstr
                return outstr

        # 返回 Fake 类的实例化对象
        return Fake()

    # 定义反向减法运算符方法 __rsub__
    def __rsub__(self, other):
        # 返回当前对象乘以 -1 后再加上 other 对象的结果
        return (-1 * self) + other
    # 定义一个方法，用于将对象打印成字符串形式
    def _sympystr(self, printer):
        """Printing method. """
        # 简化参数引用
        ar = self.args  # just to shorten things
        # 如果参数列表为空，则打印零
        if len(ar) == 0:
            return printer._print(0)
        # 初始化输出列表，用于将各部分连接成字符串
        ol = []  # output list, to be concatenated to a string
        # 遍历参数列表
        for i, v in enumerate(ar):
            # 如果二项式的系数为1，省略显示1
            if ar[i][0] == 1:
                ol.append(' + (' + printer._print(ar[i][1]) + '|' +
                          printer._print(ar[i][2]) + ')')
            # 如果二项式的系数为-1，省略显示-1
            elif ar[i][0] == -1:
                ol.append(' - (' + printer._print(ar[i][1]) + '|' +
                          printer._print(ar[i][2]) + ')')
            # 如果二项式的系数不是1或-1，可能将系数放入括号中以提高可读性
            elif ar[i][0] != 0:
                arg_str = printer._print(ar[i][0])
                # 如果系数是一个加法表达式，则将其用括号括起来
                if isinstance(ar[i][0], Add):
                    arg_str = "(%s)" % arg_str
                # 根据系数正负决定显示的格式
                if arg_str[0] == '-':
                    arg_str = arg_str[1:]
                    str_start = ' - '
                else:
                    str_start = ' + '
                # 将格式化后的二项式添加到输出列表中
                ol.append(str_start + arg_str + '*(' +
                          printer._print(ar[i][1]) +
                          '|' + printer._print(ar[i][2]) + ')')
        # 将所有部分连接成最终的输出字符串
        outstr = ''.join(ol)
        # 移除开头可能多余的加号或空格
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        # 返回最终的输出字符串
        return outstr

    # 定义减法运算符的重载方法，实现通过加法和乘法实现减法操作
    def __sub__(self, other):
        """The subtraction operator. """
        # 调用加法运算符的重载方法，乘以-1实现减法操作
        return self.__add__(other * -1)

    # 定义向量叉乘操作的方法，返回一个叉乘后的二项式结果
    def cross(self, other):
        """Returns the dyadic resulting from the dyadic vector cross product:
        Dyadic x Vector.

        Parameters
        ==========
        other : Vector
            Vector to cross with.

        Examples
        ========
        >>> from sympy.physics.vector import ReferenceFrame, outer, cross
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> cross(d, N.y)
        (N.x|N.z)

        """
        # 导入内部函数进行向量检查
        from sympy.physics.vector.vector import _check_vector
        # 检查并转换向量参数
        other = _check_vector(other)
        # 初始化结果二项式为0
        ol = Dyadic(0)
        # 遍历当前对象的参数列表
        for v in self.args:
            # 计算每个二项式的乘积，实现叉乘操作
            ol += v[0] * (v[1].outer((v[2].cross(other))))
        # 返回叉乘操作后得到的二项式结果
        return ol

    # 注意：支持非公开的 Dyadic ^ Vector 符号
    __xor__ = cross
    # 将当前的双分量对象在不同参考框架中表达

    """
    Expresses this Dyadic in alternate frame(s)

    The first frame is the list side expression, the second frame is the
    right side; if Dyadic is in form A.x|B.y, you can express it in two
    different frames. If no second frame is given, the Dyadic is
    expressed in only one frame.

    Calls the global express function

    Parameters
    ==========

    frame1 : ReferenceFrame
        The frame to express the left side of the Dyadic in
    frame2 : ReferenceFrame, optional
        If provided, the frame to express the right side of the Dyadic in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> B = N.orientnew('B', 'Axis', [q, N.z])
    >>> d = outer(N.x, N.x)
    >>> d.express(B, N)
    cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)

    """
    # 导入表达函数
    from sympy.physics.vector.functions import express
    # 调用全局的表达函数来表达当前的双分量对象
    return express(self, frame1, frame2)
    def to_matrix(self, reference_frame, second_reference_frame=None):
        """Returns the matrix form of the dyadic with respect to one or two
        reference frames.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows and columns of the matrix
            correspond to. If a second reference frame is provided, this
            only corresponds to the rows of the matrix.
        second_reference_frame : ReferenceFrame, optional, default=None
            The reference frame that the columns of the matrix correspond
            to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,3)
            The matrix that gives the 2D tensor form.

        Examples
        ========

        >>> from sympy import symbols, trigsimp
        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy.physics.mechanics import inertia
        >>> Ixx, Iyy, Izz, Ixy, Iyz, Ixz = symbols('Ixx, Iyy, Izz, Ixy, Iyz, Ixz')
        >>> N = ReferenceFrame('N')
        >>> inertia_dyadic = inertia(N, Ixx, Iyy, Izz, Ixy, Iyz, Ixz)
        >>> inertia_dyadic.to_matrix(N)
        Matrix([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> trigsimp(inertia_dyadic.to_matrix(A))
        Matrix([
        [                           Ixx,                                           Ixy*cos(beta) + Ixz*sin(beta),                                           -Ixy*sin(beta) + Ixz*cos(beta)],
        [ Ixy*cos(beta) + Ixz*sin(beta), Iyy*cos(2*beta)/2 + Iyy/2 + Iyz*sin(2*beta) - Izz*cos(2*beta)/2 + Izz/2,                 -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2],
        [-Ixy*sin(beta) + Ixz*cos(beta),                -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2, -Iyy*cos(2*beta)/2 + Iyy/2 - Iyz*sin(2*beta) + Izz*cos(2*beta)/2 + Izz/2]])

        """

        # 如果未提供第二个参考系，则默认使用第一个参考系
        if second_reference_frame is None:
            second_reference_frame = reference_frame

        # 返回一个3x3的矩阵，该矩阵表示二维张量形式
        return Matrix([i.dot(self).dot(j) for i in reference_frame for j in
                      second_reference_frame]).reshape(3, 3)

    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""

        # 对 Dyadic 对象中的每个项调用 .doit() 方法，并返回结果的总和
        return sum([Dyadic([(v[0].doit(**hints), v[1], v[2])])
                    for v in self.args], Dyadic(0))
    def dt(self, frame):
        """Take the time derivative of this Dyadic in a frame.

        This function calls the global time_derivative method

        Parameters
        ==========

        frame : ReferenceFrame
            The frame to take the time derivative in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.dt(B)
        - q'*(N.y|N.x) - q'*(N.x|N.y)

        """
        # 导入时间导数函数
        from sympy.physics.vector.functions import time_derivative
        # 调用全局的时间导数方法，对当前的 Dyadic 对象在给定参考系中进行时间导数运算
        return time_derivative(self, frame)

    def simplify(self):
        """Returns a simplified Dyadic."""
        # 初始化一个全零的 Dyadic 对象
        out = Dyadic(0)
        # 遍历当前 Dyadic 对象中的每个项，对每个项进行简化，并加到输出对象中
        for v in self.args:
            out += Dyadic([(v[0].simplify(), v[1], v[2])])
        return out

    def subs(self, *args, **kwargs):
        """Substitution on the Dyadic.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = s*(N.x|N.x)
        >>> a.subs({s: 2})
        2*(N.x|N.x)

        """
        # 对当前 Dyadic 对象中的每个项进行符号替换，返回替换后的新 Dyadic 对象
        return sum([Dyadic([(v[0].subs(*args, **kwargs), v[1], v[2])])
                    for v in self.args], Dyadic(0))

    def applyfunc(self, f):
        """Apply a function to each component of a Dyadic."""
        # 检查传入的函数是否可调用，若不可调用则抛出 TypeError 异常
        if not callable(f):
            raise TypeError("`f` must be callable.")
        
        # 初始化一个全零的 Dyadic 对象
        out = Dyadic(0)
        # 对当前 Dyadic 对象中的每个项应用传入的函数 f，并将结果加到输出对象中
        for a, b, c in self.args:
            out += f(a) * (b.outer(c))
        return out

    def _eval_evalf(self, prec):
        # 若当前 Dyadic 对象没有任何项，则直接返回自身
        if not self.args:
            return self
        
        # 将精度 prec 转换为有效数字位数
        dps = prec_to_dps(prec)
        # 初始化一个空列表，用于存储计算后的新参数
        new_args = []
        # 对当前 Dyadic 对象中的每个项进行处理
        for inlist in self.args:
            # 将每个项转换为列表，并更新第一个元素为其 evalf 计算后的值
            new_inlist = list(inlist)
            new_inlist[0] = inlist[0].evalf(n=dps)
            new_args.append(tuple(new_inlist))
        # 返回一个新的 Dyadic 对象，其参数为计算后的 new_args
        return Dyadic(new_args)
    # 定义一个方法 xreplace，用于在 Dyadic 对象的测量数中替换对象的出现
    def xreplace(self, rule):
        """
        Replace occurrences of objects within the measure numbers of the
        Dyadic.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Dyadic
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D = outer(N.x, N.x)
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * D).xreplace({x: pi})
        (pi*y + 1)*(N.x|N.x)
        >>> ((1 + x*y) * D).xreplace({x: pi, y: 2})
        (1 + 2*pi)*(N.x|N.x)

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * D).xreplace({x*y: pi})
        (z + pi)*(N.x|N.x)
        >>> ((x*y*z) * D).xreplace({x*y: pi})
        x*y*z*(N.x|N.x)

        """

        # 创建一个空列表，用于存放替换后的参数
        new_args = []
        # 遍历 self.args 中的每个列表
        for inlist in self.args:
            # 将每个列表转换为可变列表
            new_inlist = list(inlist)
            # 对列表中第一个元素调用 xreplace 方法，使用给定的替换规则
            new_inlist[0] = new_inlist[0].xreplace(rule)
            # 将处理后的列表转换为元组，并添加到新参数列表中
            new_args.append(tuple(new_inlist))
        # 返回一个新的 Dyadic 对象，其中的参数已经进行了替换
        return Dyadic(new_args)
# 定义一个函数 _check_dyadic，用于检查传入的参数是否为 Dyadic 类型
def _check_dyadic(other):
    # 如果传入的参数不是 Dyadic 类型，则抛出 TypeError 异常
    if not isinstance(other, Dyadic):
        raise TypeError('A Dyadic must be supplied')
    # 如果参数是 Dyadic 类型，则直接返回该参数
    return other
```