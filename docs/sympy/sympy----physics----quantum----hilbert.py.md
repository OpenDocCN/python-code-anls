# `D:\src\scipysrc\sympy\sympy\physics\quantum\hilbert.py`

```
# 导入必要的模块和类
from functools import reduce

from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError

# 定义模块中可以导出的类名列表
__all__ = [
    'HilbertSpaceError',
    'HilbertSpace',
    'TensorProductHilbertSpace',
    'TensorPowerHilbertSpace',
    'DirectSumHilbertSpace',
    'ComplexSpace',
    'L2',
    'FockSpace'
]

#-----------------------------------------------------------------------------
# 主要对象
#-----------------------------------------------------------------------------


class HilbertSpaceError(QuantumError):
    pass

#-----------------------------------------------------------------------------
# 主要对象
#-----------------------------------------------------------------------------


class HilbertSpace(Basic):
    """量子力学中的抽象希尔伯特空间。

    简言之，希尔伯特空间是一个包含内积的完备抽象向量空间 [1]_。

    示例
    ========

    >>> from sympy.physics.quantum.hilbert import HilbertSpace
    >>> hs = HilbertSpace()
    >>> hs
    H

    参考文献
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space
    """

    def __new__(cls):
        # 创建一个新的实例
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        """返回空间的希尔伯特维度。"""
        raise NotImplementedError('This Hilbert space has no dimension.')

    def __add__(self, other):
        """定义希尔伯特空间的加法操作。"""
        return DirectSumHilbertSpace(self, other)

    def __radd__(self, other):
        """定义希尔伯特空间的反向加法操作。"""
        return DirectSumHilbertSpace(other, self)

    def __mul__(self, other):
        """定义希尔伯特空间的乘法操作。"""
        return TensorProductHilbertSpace(self, other)

    def __rmul__(self, other):
        """定义希尔伯特空间的反向乘法操作。"""
        return TensorProductHilbertSpace(other, self)

    def __pow__(self, other, mod=None):
        """定义希尔伯特空间的乘幂操作。"""
        if mod is not None:
            raise ValueError('The third argument to __pow__ is not supported \
            for Hilbert spaces.')
        return TensorPowerHilbertSpace(self, other)

    def __contains__(self, other):
        """判断操作符或状态是否在该希尔伯特空间中。

        通过比较希尔伯特空间的类来判断，而不是实例。这样做是为了允许具有符号维度的希尔伯特空间。
        """
        if other.hilbert_space.__class__ == self.__class__:
            return True
        else:
            return False

    def _sympystr(self, printer, *args):
        """返回希尔伯特空间的字符串表示。"""
        return 'H'

    def _pretty(self, printer, *args):
        """返回希尔伯特空间的美观打印形式。"""
        ustr = '\N{LATIN CAPITAL LETTER H}'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        """返回希尔伯特空间的 LaTeX 表示。"""
        return r'\mathcal{H}'


class ComplexSpace(HilbertSpace):
    """复向量的有限维希尔伯特空间。
    The elements of this Hilbert space are n-dimensional complex valued
    vectors with the usual inner product that takes the complex conjugate
    of the vector on the right.

    A classic example of this type of Hilbert space is spin-1/2, which is
    ``ComplexSpace(2)``. Generalizing to spin-s, the space is
    ``ComplexSpace(2*s+1)``.  Quantum computing with N qubits is done with the
    direct product space ``ComplexSpace(2)**N``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.hilbert import ComplexSpace
    >>> c1 = ComplexSpace(2)
    >>> c1
    C(2)
    >>> c1.dimension
    2

    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> c2
    C(n)
    >>> c2.dimension
    n

    """

    # 定义一个 ComplexSpace 类，表示一个复数域的 Hilbert 空间
    def __new__(cls, dimension):
        # 将 dimension 转换为 Sympy 的表达式
        dimension = sympify(dimension)
        # 调用 eval 方法，返回相应的结果对象
        r = cls.eval(dimension)
        # 如果返回结果是 Basic 类型，则直接返回
        if isinstance(r, Basic):
            return r
        # 否则创建新的 ComplexSpace 对象
        obj = Basic.__new__(cls, dimension)
        return obj

    @classmethod
    # 静态方法，用于评估给定维度的合法性
    def eval(cls, dimension):
        # 检查维度是否只包含一个原子项
        if len(dimension.atoms()) == 1:
            # 如果是整数且大于 0，或者是无穷大，或者是符号，则合法
            if not (dimension.is_Integer and dimension > 0 or dimension is S.Infinity
            or dimension.is_Symbol):
                # 抛出类型错误，说明 ComplexSpace 的维度必须是正整数、无穷大或符号
                raise TypeError('The dimension of a ComplexSpace can only'
                                'be a positive integer, oo, or a Symbol: %r'
                                % dimension)
        else:
            # 如果维度包含多个原子项，每一项必须是整数、无穷大或符号
            for dim in dimension.atoms():
                if not (dim.is_Integer or dim is S.Infinity or dim.is_Symbol):
                    # 抛出类型错误，说明 ComplexSpace 的维度只能包含整数、无穷大或符号
                    raise TypeError('The dimension of a ComplexSpace can only'
                                    ' contain integers, oo, or a Symbol: %r'
                                    % dim)

    # 维度属性的 getter 方法，返回 ComplexSpace 对象的维度
    @property
    def dimension(self):
        return self.args[0]

    # 返回 SymPy 表示形式的私有方法
    def _sympyrepr(self, printer, *args):
        return "%s(%s)" % (self.__class__.__name__,
                           printer._print(self.dimension, *args))

    # 返回 SymPy 字符串表示形式的私有方法
    def _sympystr(self, printer, *args):
        return "C(%s)" % printer._print(self.dimension, *args)

    # 返回漂亮打印形式的私有方法
    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER C}'
        pform_exp = printer._print(self.dimension, *args)
        pform_base = prettyForm(ustr)
        return pform_base**pform_exp

    # 返回 LaTeX 表示形式的私有方法
    def _latex(self, printer, *args):
        return r'\mathcal{C}^{%s}' % printer._print(self.dimension, *args)
class L2(HilbertSpace):
    """The Hilbert space of square integrable functions on an interval.

    An L2 object takes in a single SymPy Interval argument which represents
    the interval its functions (vectors) are defined on.

    Examples
    ========

    >>> from sympy import Interval, oo
    >>> from sympy.physics.quantum.hilbert import L2
    >>> hs = L2(Interval(0,oo))
    >>> hs
    L2(Interval(0, oo))
    >>> hs.dimension
    oo
    >>> hs.interval
    Interval(0, oo)

    """

    def __new__(cls, interval):
        # 检查输入是否为 SymPy 的 Interval 实例，否则引发类型错误
        if not isinstance(interval, Interval):
            raise TypeError('L2 interval must be an Interval instance: %r'
            % interval)
        # 创建 L2 对象的实例
        obj = Basic.__new__(cls, interval)
        return obj

    @property
    def dimension(self):
        # 返回空间维度为无穷大
        return S.Infinity

    @property
    def interval(self):
        # 返回此 L2 空间的定义区间
        return self.args[0]

    def _sympyrepr(self, printer, *args):
        # 返回用于在 SymPy 中表示该对象的字符串形式
        return "L2(%s)" % printer._print(self.interval, *args)

    def _sympystr(self, printer, *args):
        # 返回用于在 SymPy 中打印该对象的字符串形式
        return "L2(%s)" % printer._print(self.interval, *args)

    def _pretty(self, printer, *args):
        # 返回美观打印格式
        pform_exp = prettyForm('2')
        pform_base = prettyForm('L')
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        # 返回 LaTeX 格式的表示，显示 L2 空间及其区间
        interval = printer._print(self.interval, *args)
        return r'{\mathcal{L}^2}\left( %s \right)' % interval


class FockSpace(HilbertSpace):
    """The Hilbert space for second quantization.

    Technically, this Hilbert space is a infinite direct sum of direct
    products of single particle Hilbert spaces [1]_. This is a mess, so we have
    a class to represent it directly.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import FockSpace
    >>> hs = FockSpace()
    >>> hs
    F
    >>> hs.dimension
    oo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fock_space
    """

    def __new__(cls):
        # 创建 FockSpace 对象的实例
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        # 返回空间维度为无穷大
        return S.Infinity

    def _sympyrepr(self, printer, *args):
        # 返回用于在 SymPy 中表示该对象的字符串形式
        return "FockSpace()"

    def _sympystr(self, printer, *args):
        # 返回用于在 SymPy 中打印该对象的字符串形式
        return "F"

    def _pretty(self, printer, *args):
        # 返回美观打印格式
        ustr = '\N{LATIN CAPITAL LETTER F}'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        # 返回 LaTeX 格式的表示，显示 Fock 空间
        return r'\mathcal{F}'


class TensorProductHilbertSpace(HilbertSpace):
    """A tensor product of Hilbert spaces [1]_.

    The tensor product between Hilbert spaces is represented by the
    operator ``*`` Products of the same Hilbert space will be combined into
    tensor powers.

    A ``TensorProductHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. In addition, multiplication of
    ``HilbertSpace`` objects will automatically return this tensor product
    object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace

    """
    >>> from sympy import symbols

    >>> c = ComplexSpace(2)  # 创建一个复数空间，维度为2
    >>> f = FockSpace()  # 创建一个Fock空间
    >>> hs = c*f  # 计算复数空间和Fock空间的张量积
    >>> hs  # 打印张量积结果
    C(2)*F
    >>> hs.dimension  # 计算张量积空间的维度
    oo  # 返回无穷大，因为Fock空间的维度是无穷大
    >>> hs.spaces  # 返回张量积中的空间组成元组
    (C(2), F)

    >>> c1 = ComplexSpace(2)  # 创建另一个复数空间，维度为2
    >>> n = symbols('n')  # 创建一个符号变量 n
    >>> c2 = ComplexSpace(n)  # 创建一个以 n 为维度的复数空间
    >>> hs = c1*c2  # 计算两个复数空间的张量积
    >>> hs  # 打印张量积结果
    C(2)*C(n)
    >>> hs.dimension  # 计算张量积空间的维度
    2*n  # 返回维度的表达式，这里是 2*n

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)  # 调用类方法 eval 处理参数并返回结果
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, *args)  # 调用基类 Basic 的构造方法创建新对象
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct product."""
        new_args = []
        recall = False
        # flatten arguments 将参数扁平化
        for arg in args:
            if isinstance(arg, TensorProductHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, (HilbertSpace, TensorPowerHilbertSpace)):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be multiplied by \
                other Hilbert spaces: %r' % arg)
        # combine like arguments into direct powers 将类似的参数组合成直接乘幂
        comb_args = []
        prev_arg = None
        for new_arg in new_args:
            if prev_arg is not None:
                if isinstance(new_arg, TensorPowerHilbertSpace) and \
                    isinstance(prev_arg, TensorPowerHilbertSpace) and \
                        new_arg.base == prev_arg.base:
                    prev_arg = new_arg.base**(new_arg.exp + prev_arg.exp)
                elif isinstance(new_arg, TensorPowerHilbertSpace) and \
                        new_arg.base == prev_arg:
                    prev_arg = prev_arg**(new_arg.exp + 1)
                elif isinstance(prev_arg, TensorPowerHilbertSpace) and \
                        new_arg == prev_arg.base:
                    prev_arg = new_arg**(prev_arg.exp + 1)
                elif new_arg == prev_arg:
                    prev_arg = new_arg**2
                else:
                    comb_args.append(prev_arg)
                    prev_arg = new_arg
            elif prev_arg is None:
                prev_arg = new_arg
        comb_args.append(prev_arg)
        if recall:
            return TensorProductHilbertSpace(*comb_args)  # 如果需要回调，返回张量积空间对象
        elif len(comb_args) == 1:
            return TensorPowerHilbertSpace(comb_args[0].base, comb_args[0].exp)  # 如果只有一个参数，返回乘幂空间对象
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]  # 获取每个参数的维度
        if S.Infinity in arg_list:
            return S.Infinity  # 如果有一个参数的维度是无穷大，则返回无穷大
        else:
            return reduce(lambda x, y: x*y, arg_list)  # 否则返回所有参数维度的乘积

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this tensor product."""
        return self.args  # 返回张量积空间中的所有参数组成的元组
    # 定义一个方法用于打印空间对象的表示形式，使用给定的打印器和参数
    def _spaces_printer(self, printer, *args):
        # 初始化一个空列表，用于存储空间对象的字符串表示
        spaces_strs = []
        # 遍历当前对象的参数列表
        for arg in self.args:
            # 调用打印器的打印方法，获取参数的字符串表示
            s = printer._print(arg, *args)
            # 如果参数是 DirectSumHilbertSpace 类型，将其用括号括起来
            if isinstance(arg, DirectSumHilbertSpace):
                s = '(%s)' % s
            # 将参数的字符串表示添加到列表中
            spaces_strs.append(s)
        # 返回所有空间对象的字符串表示的列表
        return spaces_strs

    # 定义一个方法用于返回 TensorProductHilbertSpace 对象的 SymPy 表示形式
    def _sympyrepr(self, printer, *args):
        # 调用前面定义的方法，获取空间对象的字符串表示的列表
        spaces_reprs = self._spaces_printer(printer, *args)
        # 使用字符串连接方式构建 TensorProductHilbertSpace 的 SymPy 表示形式
        return "TensorProductHilbertSpace(%s)" % ','.join(spaces_reprs)

    # 定义一个方法用于返回 TensorProductHilbertSpace 对象的字符串表示形式
    def _sympystr(self, printer, *args):
        # 调用前面定义的方法，获取空间对象的字符串表示的列表
        spaces_strs = self._spaces_printer(printer, *args)
        # 使用 '*' 符号连接所有空间对象的字符串表示，形成字符串表示形式
        return '*'.join(spaces_strs)

    # 定义一个方法用于返回 TensorProductHilbertSpace 对象的漂亮打印形式
    def _pretty(self, printer, *args):
        # 获取参数列表的长度
        length = len(self.args)
        # 利用打印器获取当前对象的空字符串表示
        pform = printer._print('', *args)
        # 遍历参数列表
        for i in range(length):
            # 获取当前参数的打印形式
            next_pform = printer._print(self.args[i], *args)
            # 如果参数是 DirectSumHilbertSpace 或 TensorProductHilbertSpace 类型，用漂亮打印形式包裹
            if isinstance(self.args[i], (DirectSumHilbertSpace, TensorProductHilbertSpace)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            # 将当前参数的打印形式添加到 pform 的右侧
            pform = prettyForm(*pform.right(next_pform))
            # 如果不是最后一个参数，根据打印器的设置添加适当的分隔符
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right(' ' + '\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right(' x '))
        # 返回最终的漂亮打印形式
        return pform

    # 定义一个方法用于返回 TensorProductHilbertSpace 对象的 LaTeX 表示形式
    def _latex(self, printer, *args):
        # 获取参数列表的长度
        length = len(self.args)
        # 初始化一个空字符串 s
        s = ''
        # 遍历参数列表
        for i in range(length):
            # 获取当前参数的 LaTeX 表示形式
            arg_s = printer._print(self.args[i], *args)
            # 如果参数是 DirectSumHilbertSpace 或 TensorProductHilbertSpace 类型，用 LaTeX 格式括起来
            if isinstance(self.args[i], (DirectSumHilbertSpace, TensorProductHilbertSpace)):
                arg_s = r'\left(%s\right)' % arg_s
            # 将当前参数的 LaTeX 表示形式添加到 s 中
            s = s + arg_s
            # 如果不是最后一个参数，添加 \otimes 分隔符
            if i != length - 1:
                s = s + r'\otimes '
        # 返回最终的 LaTeX 表示形式字符串
        return s
class DirectSumHilbertSpace(HilbertSpace):
    """A direct sum of Hilbert spaces [1]_.

    This class represents the direct sum of multiple Hilbert spaces.

    This class uses the ``+`` operator to represent direct sums between
    different Hilbert spaces.

    A ``DirectSumHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. Also, addition of
    ``HilbertSpace`` objects will automatically return a direct sum object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c+f
    >>> hs
    C(2)+F
    >>> hs.dimension
    oo
    >>> list(hs.spaces)
    [C(2), F]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Direct_sums
    """

    def __new__(cls, *args):
        # Evaluate the arguments and return the result if it's already a Basic object
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        # Otherwise, create a new instance of the class with the given arguments
        obj = Basic.__new__(cls, *args)
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct sum."""
        new_args = []
        recall = False
        # Flatten arguments: check each argument and process accordingly
        for arg in args:
            if isinstance(arg, DirectSumHilbertSpace):
                # If argument is a DirectSumHilbertSpace, extend its arguments to new_args
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, HilbertSpace):
                # If argument is a HilbertSpace, append it to new_args
                new_args.append(arg)
            else:
                # Raise an error for unsupported types
                raise TypeError('Hilbert spaces can only be summed with other Hilbert spaces: %r' % arg)
        # If recall flag is set, recursively create a new DirectSumHilbertSpace with updated arguments
        if recall:
            return DirectSumHilbertSpace(*new_args)
        else:
            return None

    @property
    def dimension(self):
        # Calculate the dimension of the direct sum by summing dimensions of each component
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x + y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this direct sum."""
        return self.args

    def _sympyrepr(self, printer, *args):
        # Representation of the object suitable for SymPy's internal representation
        spaces_reprs = [printer._print(arg, *args) for arg in self.args]
        return "DirectSumHilbertSpace(%s)" % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        # String representation of the object
        spaces_strs = [printer._print(arg, *args) for arg in self.args]
        return '+'.join(spaces_strs)
    # 定义一个方法 `_pretty`，用于美化打印输出
    def _pretty(self, printer, *args):
        # 计算参数的长度
        length = len(self.args)
        # 使用打印机对象打印空字符串和参数
        pform = printer._print('', *args)
        # 遍历参数列表
        for i in range(length):
            # 使用打印机对象打印当前参数
            next_pform = printer._print(self.args[i], *args)
            # 如果当前参数是 DirectSumHilbertSpace 或 TensorProductHilbertSpace 类的实例
            if isinstance(self.args[i], (DirectSumHilbertSpace, TensorProductHilbertSpace)):
                # 将打印结果用括号包围
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            # 将当前打印结果连接到主打印表单的右侧
            pform = prettyForm(*pform.right(next_pform))
            # 如果不是最后一个参数
            if i != length - 1:
                # 如果打印机使用 Unicode
                if printer._use_unicode:
                    # 在主打印表单的右侧连接一个圆圈加号符号
                    pform = prettyForm(*pform.right(' \N{CIRCLED PLUS} '))
                else:
                    # 否则，在主打印表单的右侧连接一个加号符号
                    pform = prettyForm(*pform.right(' + '))
        # 返回最终的美化打印表单
        return pform

    # 定义一个方法 `_latex`，用于生成 LaTeX 格式的输出
    def _latex(self, printer, *args):
        # 计算参数的长度
        length = len(self.args)
        # 初始化空字符串
        s = ''
        # 遍历参数列表
        for i in range(length):
            # 使用打印机对象打印当前参数
            arg_s = printer._print(self.args[i], *args)
            # 如果当前参数是 DirectSumHilbertSpace 或 TensorProductHilbertSpace 类的实例
            if isinstance(self.args[i], (DirectSumHilbertSpace, TensorProductHilbertSpace)):
                # 在参数字符串外面加上 LaTeX 的左右括号
                arg_s = r'\left(%s\right)' % arg_s
            # 将当前参数字符串连接到总字符串 s 上
            s = s + arg_s
            # 如果不是最后一个参数
            if i != length - 1:
                # 在 s 上连接 LaTeX 的直和符号
                s = s + r'\oplus '
        # 返回最终的 LaTeX 字符串表示
        return s
class TensorPowerHilbertSpace(HilbertSpace):
    """An exponentiated Hilbert space [1]_.

    Tensor powers (repeated tensor products) are represented by the
    operator ``**`` Identical Hilbert spaces that are multiplied together
    will be automatically combined into a single tensor power object.

    Any Hilbert space, product, or sum may be raised to a tensor power. The
    ``TensorPowerHilbertSpace`` takes two arguments: the Hilbert space; and the
    tensor power (number).

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> n = symbols('n')
    >>> c = ComplexSpace(2)
    >>> hs = c**n
    >>> hs
    C(2)**n
    >>> hs.dimension
    2**n

    >>> c = ComplexSpace(2)
    >>> c*c
    C(2)**2
    >>> f = FockSpace()
    >>> c*f*f
    C(2)*F**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        return Basic.__new__(cls, *r)

    @classmethod
    def eval(cls, args):
        new_args = args[0], sympify(args[1])
        exp = new_args[1]

        # simplify hs**1 -> hs
        if exp is S.One:
            return args[0]
        # simplify hs**0 -> 1
        if exp is S.Zero:
            return S.One
        # check (and allow) for hs**(x+42+y...) case
        if len(exp.atoms()) == 1:
            if not (exp.is_Integer and exp >= 0 or exp.is_Symbol):
                raise ValueError('Hilbert spaces can only be raised to \
                positive integers or Symbols: %r' % exp)
        else:
            for power in exp.atoms():
                if not (power.is_Integer or power.is_Symbol):
                    raise ValueError('Tensor powers can only contain integers \
                    or Symbols: %r' % power)
        return new_args

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

    @property
    def dimension(self):
        # Calculate the dimension of the tensor power Hilbert space
        if self.base.dimension is S.Infinity:
            return S.Infinity
        else:
            return self.base.dimension**self.exp

    def _sympyrepr(self, printer, *args):
        # Representation of the object in SymPy's internal format
        return "TensorPowerHilbertSpace(%s,%s)" % (printer._print(self.base,
        *args), printer._print(self.exp, *args))

    def _sympystr(self, printer, *args):
        # String representation of the object for printing
        return "%s**%s" % (printer._print(self.base, *args),
        printer._print(self.exp, *args))

    def _pretty(self, printer, *args):
        # Pretty-printing representation of the object
        pform_exp = printer._print(self.exp, *args)
        if printer._use_unicode:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('\N{N-ARY CIRCLED TIMES OPERATOR}')))
        else:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('x')))
        pform_base = printer._print(self.base, *args)
        return pform_base**pform_exp
    # 定义一个方法 `_latex`，用于生成 LaTeX 格式的表示
    def _latex(self, printer, *args):
        # 将 self.base 对象转换为 LaTeX 格式的字符串表示
        base = printer._print(self.base, *args)
        # 将 self.exp 对象转换为 LaTeX 格式的字符串表示
        exp = printer._print(self.exp, *args)
        # 返回格式化后的 LaTeX 字符串，表示为 base 的 exp 次幂
        return r'{%s}^{\otimes %s}' % (base, exp)
```