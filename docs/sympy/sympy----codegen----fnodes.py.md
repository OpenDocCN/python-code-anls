# `D:\src\scipysrc\sympy\sympy\codegen\fnodes.py`

```
"""
AST nodes specific to Fortran.

The functions defined in this module allow the user to express functions such as ``dsign``
as a SymPy function for symbolic manipulation.
"""

# 导入需要的类和函数
from sympy.codegen.ast import (
    Attribute, CodeBlock, FunctionCall, Node, none, String,
    Token, _mk_Tuple, Variable
)
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable

# 定义 Fortran 中常用的属性
pure = Attribute('pure')
elemental = Attribute('elemental')  # (all elemental procedures are also pure)

intent_in = Attribute('intent_in')
intent_out = Attribute('intent_out')
intent_inout = Attribute('intent_inout')

allocatable = Attribute('allocatable')

# 定义 Program 类，表示 Fortran 中的 'program' 块
class Program(Token):
    """ Represents a 'program' block in Fortran.

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy.codegen.fnodes import Program
    >>> prog = Program('myprogram', [Print([42])])
    >>> from sympy import fcode
    >>> print(fcode(prog, source_format='free'))
    program myprogram
        print *, 42
    end program

    """
    __slots__ = _fields = ('name', 'body')
    _construct_name = String
    _construct_body = staticmethod(lambda body: CodeBlock(*body))

# 定义 use_rename 类，表示 Fortran 中 use 语句中的重命名
class use_rename(Token):
    """ Represents a renaming in a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use_rename, use
    >>> from sympy import fcode
    >>> ren = use_rename("thingy", "convolution2d")
    >>> print(fcode(ren, source_format='free'))
    thingy => convolution2d
    >>> full = use('signallib', only=['snr', ren])
    >>> print(fcode(full, source_format='free'))
    use signallib, only: snr, thingy => convolution2d

    """
    __slots__ = _fields = ('local', 'original')
    _construct_local = String
    _construct_original = String

# 定义 _name 函数，用于获取对象的名称
def _name(arg):
    if hasattr(arg, 'name'):
        return arg.name
    else:
        return String(arg)

# 定义 use 类，表示 Fortran 中的 use 语句
class use(Token):
    """ Represents a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use
    >>> from sympy import fcode
    >>> fcode(use('signallib'), source_format='free')
    'use signallib'
    >>> fcode(use('signallib', [('metric', 'snr')]), source_format='free')
    'use signallib, metric => snr'
    >>> fcode(use('signallib', only=['snr', 'convolution2d']), source_format='free')
    'use signallib, only: snr, convolution2d'

    """
    __slots__ = _fields = ('namespace', 'rename', 'only')
    defaults = {'rename': none, 'only': none}
    _construct_namespace = staticmethod(_name)
    _construct_rename = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else use_rename(*arg) for arg in args]))
    # 定义一个静态方法，用于构造只含特定类型元素的元组
    _construct_only = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else _name(arg) for arg in args]))
class Module(Token):
    """ Represents a module in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Module
    >>> from sympy import fcode
    >>> print(fcode(Module('signallib', ['implicit none'], []), source_format='free'))
    module signallib
    implicit none
    <BLANKLINE>
    contains
    <BLANKLINE>
    <BLANKLINE>
    end module

    """
    # 模块类，表示Fortran中的模块

    __slots__ = _fields = ('name', 'declarations', 'definitions')
    # 限制实例可以拥有的属性，包括模块名称、声明和定义

    defaults = {'declarations': Tuple()}
    # 默认情况下，模块的声明为空元组

    _construct_name = String
    # 构造模块名称的类型为字符串

    @classmethod
    def _construct_declarations(cls, args):
        # 类方法，用于构造声明部分
        args = [Str(arg) if isinstance(arg, str) else arg for arg in args]
        # 将参数列表中的每个参数转换为字符串类型（如果是字符串）
        return CodeBlock(*args)
        # 返回一个代码块对象，包含所有声明内容

    _construct_definitions = staticmethod(lambda arg: CodeBlock(*arg))
    # 静态方法，用于构造模块的定义部分，将给定参数转换为代码块对象


class Subroutine(Node):
    """ Represents a subroutine in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import Print
    >>> from sympy.codegen.fnodes import Subroutine
    >>> x, y = symbols('x y', real=True)
    >>> sub = Subroutine('mysub', [x, y], [Print([x**2 + y**2, x*y])])
    >>> print(fcode(sub, source_format='free', standard=2003))
    subroutine mysub(x, y)
    real*8 :: x
    real*8 :: y
    print *, x**2 + y**2, x*y
    end subroutine

    """
    # 表示Fortran中的子程序

    __slots__ = ('name', 'parameters', 'body')
    # 限制实例可以拥有的属性，包括子程序名称、参数和主体

    _fields = __slots__ + Node._fields
    # 属性包括子程序自身属性以及继承自Node的属性

    _construct_name = String
    # 构造子程序名称的类型为字符串

    _construct_parameters = staticmethod(lambda params: Tuple(*map(Variable.deduced, params)))
    # 静态方法，用于构造子程序的参数列表，将参数映射为推断类型的变量

    @classmethod
    def _construct_body(cls, itr):
        # 类方法，用于构造子程序的主体部分
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)
        # 如果输入是代码块，则直接返回，否则将所有输入作为代码块内容返回


class SubroutineCall(Token):
    """ Represents a call to a subroutine in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import SubroutineCall
    >>> from sympy import fcode
    >>> fcode(SubroutineCall('mysub', 'x y'.split()))
    '       call mysub(x, y)'

    """
    # 表示Fortran中对子程序的调用

    __slots__ = _fields = ('name', 'subroutine_args')
    # 限制实例可以拥有的属性，包括子程序名称和参数

    _construct_name = staticmethod(_name)
    # 静态方法，用于构造子程序名称

    _construct_subroutine_args = staticmethod(_mk_Tuple)
    # 静态方法，用于构造子程序的参数列表


class Do(Token):
    """ Represents a Do loop in in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import aug_assign, Print
    >>> from sympy.codegen.fnodes import Do
    >>> i, n = symbols('i n', integer=True)
    >>> r = symbols('r', real=True)
    >>> body = [aug_assign(r, '+', 1/i), Print([i, r])]
    >>> do1 = Do(body, i, 1, n)
    >>> print(fcode(do1, source_format='free'))
    do i = 1, n
        r = r + 1d0/i
        print *, i, r
    end do
    >>> do2 = Do(body, i, 1, n, 2)
    >>> print(fcode(do2, source_format='free'))
    do i = 1, n, 2
        r = r + 1d0/i
        print *, i, r
    end do

    """
    # 表示Fortran中的Do循环

    __slots__ = _fields = ('body', 'counter', 'first', 'last', 'step', 'concurrent')
    # 限制实例可以拥有的属性，包括循环体、计数器、起始、结束、步长和并发标志

    defaults = {'step': Integer(1), 'concurrent': false}
    # 默认情况下，步长为1，不是并发循环

    _construct_body = staticmethod(lambda body: CodeBlock(*body))
    # 静态方法，用于构造循环的主体部分，将输入的循环体列表转换为代码块对象
    # 使用 sympify 函数作为静态方法来构造计数器
    _construct_counter = staticmethod(sympify)
    # 使用 sympify 函数作为静态方法来构造第一个元素
    _construct_first = staticmethod(sympify)
    # 使用 sympify 函数作为静态方法来构造最后一个元素
    _construct_last = staticmethod(sympify)
    # 使用 sympify 函数作为静态方法来构造步长
    _construct_step = staticmethod(sympify)
    # 使用 lambda 表达式作为静态方法来构造并发状态，如果参数为真则返回 true，否则返回 false
    _construct_concurrent = staticmethod(lambda arg: true if arg else false)
class ArrayConstructor(Token):
    """ Represents an array constructor.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import ArrayConstructor
    >>> ac = ArrayConstructor([1, 2, 3])
    >>> fcode(ac, standard=95, source_format='free')
    '(/1, 2, 3/)'
    >>> fcode(ac, standard=2003, source_format='free')
    '[1, 2, 3]'

    """
    __slots__ = _fields = ('elements',)
    _construct_elements = staticmethod(_mk_Tuple)


class ImpliedDoLoop(Token):
    """ Represents an implied do loop in Fortran.

    Examples
    ========

    >>> from sympy import Symbol, fcode
    >>> from sympy.codegen.fnodes import ImpliedDoLoop, ArrayConstructor
    >>> i = Symbol('i', integer=True)
    >>> idl = ImpliedDoLoop(i**3, i, -3, 3, 2)  # -27, -1, 1, 27
    >>> ac = ArrayConstructor([-28, idl, 28]) # -28, -27, -1, 1, 27, 28
    >>> fcode(ac, standard=2003, source_format='free')
    '[-28, (i**3, i = -3, 3, 2), 28]'

    """
    __slots__ = _fields = ('expr', 'counter', 'first', 'last', 'step')
    defaults = {'step': Integer(1)}
    _construct_expr = staticmethod(sympify)
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)


class Extent(Basic):
    """ Represents a dimension extent.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Extent
    >>> e = Extent(-3, 3)  # -3, -2, -1, 0, 1, 2, 3
    >>> from sympy import fcode
    >>> fcode(e, source_format='free')
    '-3:3'
    >>> from sympy.codegen.ast import Variable, real
    >>> from sympy.codegen.fnodes import dimension, intent_out
    >>> dim = dimension(e, e)
    >>> arr = Variable('x', real, attrs=[dim, intent_out])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'real*8, dimension(-3:3, -3:3), intent(out) :: x'

    """
    def __new__(cls, *args):
        if len(args) == 2:
            low, high = args
            return Basic.__new__(cls, sympify(low), sympify(high))
        elif len(args) == 0 or (len(args) == 1 and args[0] in (':', None)):
            return Basic.__new__(cls)  # assumed shape
        else:
            raise ValueError("Expected 0 or 2 args (or one argument == None or ':')")

    def _sympystr(self, printer):
        if len(self.args) == 0:
            return ':'
        return ":".join(str(arg) for arg in self.args)

assumed_extent = Extent() # or Extent(':'), Extent(None)


def dimension(*args):
    """ Creates a 'dimension' Attribute with (up to 7) extents.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import dimension, intent_in
    >>> dim = dimension('2', ':')  # 2 rows, runtime determined number of columns
    >>> from sympy.codegen.ast import Variable, integer
    >>> arr = Variable('a', integer, attrs=[dim, intent_in])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'integer :: a(2,:)'

    """
    """
    # 检查参数列表长度，如果超过7个维度则抛出异常
    if len(args) > 7:
        raise ValueError("Fortran only supports up to 7 dimensional arrays")
    
    # 初始化一个空列表来存储参数
    parameters = []
    
    # 遍历传入的参数列表
    for arg in args:
        # 如果参数是 Extent 类型，则直接添加到参数列表中
        if isinstance(arg, Extent):
            parameters.append(arg)
        # 如果参数是字符串类型
        elif isinstance(arg, str):
            # 如果是 ':'，则表示一个未指定大小的维度，使用 Extent() 对象代表
            if arg == ':':
                parameters.append(Extent())
            # 否则，将字符串包装成 String 对象后添加到参数列表
            else:
                parameters.append(String(arg))
        # 如果参数是可迭代的，将其解包成 Extent 对象后添加到参数列表
        elif iterable(arg):
            parameters.append(Extent(*arg))
        # 其他情况，将参数转换成符号表达式后添加到参数列表
        else:
            parameters.append(sympify(arg))
    
    # 如果参数列表为空，则抛出异常
    if len(args) == 0:
        raise ValueError("Need at least one dimension")
    
    # 返回一个 Attribute 对象，表示维度相关的属性
    return Attribute('dimension', parameters)
    ```
# 假设数组大小为 "*"
assumed_size = dimension('*')

def array(symbol, dim, intent=None, *, attrs=(), value=None, type=None):
    """ Convenience function for creating a Variable instance for a Fortran array.

    Parameters
    ==========

    symbol : symbol
        变量名
    dim : Attribute or iterable
        如果 dim 是一个 `Attribute`，它的名称必须是 'dimension'。如果不是 `Attribute`，则传递给 `dimension` 函数作为 `*dim`
        数组的维度描述，可以是 `Attribute` 类型或者可迭代对象
    intent : str
        参数意图，可以是 'in', 'out', 'inout' 或 None
    \\*\\*kwargs:
        关键字参数用于 `Variable` 类型和值 ('type' 和 'value')

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.ast import integer, real
    >>> from sympy.codegen.fnodes import array
    >>> arr = array('a', '*', 'in', type=integer)
    >>> print(fcode(arr.as_Declaration(), source_format='free', standard=2003))
    integer*4, dimension(*), intent(in) :: a
    >>> x = array('x', [3, ':', ':'], intent='out', type=real)
    >>> print(fcode(x.as_Declaration(value=1), source_format='free', standard=2003))
    real*8, dimension(3, :, :), intent(out) :: x = 1

    """
    if isinstance(dim, Attribute):
        if str(dim.name) != 'dimension':
            raise ValueError("Got an unexpected Attribute argument as dim: %s" % str(dim))
    else:
        dim = dimension(*dim)

    # 将维度属性添加到属性列表中
    attrs = list(attrs) + [dim]
    if intent is not None:
        # 如果指定了参数意图，则添加到属性列表中
        if intent not in (intent_in, intent_out, intent_inout):
            intent = {'in': intent_in, 'out': intent_out, 'inout': intent_inout}[intent]
        attrs.append(intent)
    # 如果没有指定类型，则根据值和属性自动推断变量类型
    if type is None:
        return Variable.deduced(symbol, value=value, attrs=attrs)
    else:
        # 否则，显式指定变量类型和其他属性创建变量
        return Variable(symbol, type, value=value, attrs=attrs)

def _printable(arg):
    return String(arg) if isinstance(arg, str) else sympify(arg)


def allocated(array):
    """ Creates an AST node for a function call to Fortran's "allocated(...)"

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import allocated
    >>> alloc = allocated('x')
    >>> fcode(alloc, source_format='free')
    'allocated(x)'

    """
    # 创建调用 Fortran 中 "allocated(...)" 函数的 AST 节点
    return FunctionCall('allocated', [_printable(array)])


def lbound(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "lbound(...)"

    Parameters
    ==========

    array : Symbol or String
        数组的符号或字符串表示
    dim : expr
        维度表达式
    kind : expr
        类型表达式

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import lbound
    >>> lb = lbound('arr', dim=2)
    >>> fcode(lb, source_format='free')
    'lbound(arr, 2)'

    """
    # 创建调用 Fortran 中 "lbound(...)" 函数的 AST 节点
    return FunctionCall(
        'lbound',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def ubound(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "ubound(...)"

    Parameters
    ==========

    array : Symbol or String
        数组的符号或字符串表示
    dim : expr
        维度表达式
    kind : expr
        类型表达式

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import ubound
    >>> ub = ubound('arr', dim=1, kind=4)
    >>> fcode(ub, source_format='free')
    'ubound(arr, 1, 4)'

    """
    # 创建调用 Fortran 中 "ubound(...)" 函数的 AST 节点
    return FunctionCall(
        'ubound',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )
# 定义一个函数 shape，用于创建表示 Fortran 的 "shape(...)" 函数调用的 AST 节点
def shape(source, kind=None):
    # 返回一个 FunctionCall 对象，表示函数调用 "shape"
    return FunctionCall(
        'shape',
        [_printable(source)] +  # 将 source 转换为可打印的对象，并作为函数参数
        ([_printable(kind)] if kind else [])  # 如果提供了 kind 参数，则将其转换为可打印的对象并作为额外的参数
    )


# 定义一个函数 size，用于创建表示 Fortran 的 "size(...)" 函数调用的 AST 节点
def size(array, dim=None, kind=None):
    # 返回一个 FunctionCall 对象，表示函数调用 "size"
    return FunctionCall(
        'size',
        [_printable(array)] +  # 将 array 转换为可打印的对象，并作为函数参数
        ([_printable(dim)] if dim else []) +  # 如果提供了 dim 参数，则将其转换为可打印的对象并作为额外的参数
        ([_printable(kind)] if kind else [])  # 如果提供了 kind 参数，则将其转换为可打印的对象并作为额外的参数
    )


# 定义一个函数 reshape，用于创建表示 Fortran 的 "reshape(...)" 函数调用的 AST 节点
def reshape(source, shape, pad=None, order=None):
    # 返回一个 FunctionCall 对象，表示函数调用 "reshape"
    return FunctionCall(
        'reshape',
        [_printable(source), _printable(shape)] +  # 将 source 和 shape 转换为可打印的对象，并作为函数参数
        ([_printable(pad)] if pad else []) +  # 如果提供了 pad 参数，则将其转换为可打印的对象并作为额外的参数
        ([_printable(order)] if pad else [])  # 如果提供了 order 参数，则将其转换为可打印的对象并作为额外的参数
    )


# 定义一个函数 bind_C，用于创建带有名称的 "bind_C" 属性
def bind_C(name=None):
    # 返回一个 Attribute 对象，表示具有名称为 "bind_C" 的属性
    return Attribute('bind_C', [String(name)] if name else [])


# 定义一个类 GoTo，表示 Fortran 中的 goto 语句
class GoTo(Token):
    """ Represents a goto statement in Fortran

    Examples
    ========

    >>> from sympy.codegen.fnodes import GoTo
    >>> go = GoTo([10, 20, 30], 'i')
    >>> from sympy import fcode
    >>> fcode(go, source_format='free')
    'go to (10, 20, 30), i'

    """
    __slots__ = _fields = ('labels', 'expr')
    defaults = {'expr': none}  # 默认的 expr 属性为 none
    # 将 _mk_Tuple 方法作为静态方法 _construct_labels
    _construct_labels = staticmethod(_mk_Tuple)
    # 将 sympify 方法作为静态方法 _construct_expr
    _construct_expr = staticmethod(sympify)
class FortranReturn(Token):
    """ AST node explicitly mapped to a Fortran "return".

    Explanation
    ===========
    
    Because a return statement in Fortran is different from C, and
    in order to aid reuse of our codegen ASTs, the ordinary
    ``.codegen.ast.Return`` is interpreted as assignment to
    the result variable of the function. If one, for some reason, needs
    to generate a Fortran RETURN statement, this node should be used.

    Examples
    ========

    >>> from sympy.codegen.fnodes import FortranReturn
    >>> from sympy import fcode
    >>> fcode(FortranReturn('x'))
    '       return x'

    """
    __slots__ = _fields = ('return_value',)
    defaults = {'return_value': none}
    _construct_return_value = staticmethod(sympify)


class FFunction(Function):
    _required_standard = 77

    def _fcode(self, printer):
        name = self.__class__.__name__
        if printer._settings['standard'] < self._required_standard:
            raise NotImplementedError("%s requires Fortran %d or newer" %
                                      (name, self._required_standard))
        return '{}({})'.format(name, ', '.join(map(printer._print, self.args)))


class F95Function(FFunction):
    _required_standard = 95


class isign(FFunction):
    """ Fortran sign intrinsic for integer arguments. """
    nargs = 2


class dsign(FFunction):
    """ Fortran sign intrinsic for double precision arguments. """
    nargs = 2


class cmplx(FFunction):
    """ Fortran complex conversion function. """
    nargs = 2  # may be extended to (2, 3) at a later point


class kind(FFunction):
    """ Fortran kind function. """
    nargs = 1


class merge(F95Function):
    """ Fortran merge function """
    nargs = 3


class _literal(Float):
    _token = None  # type: str
    _decimals = None  # type: int

    def _fcode(self, printer, *args, **kwargs):
        mantissa, sgnd_ex = ('%.{}e'.format(self._decimals) % self).split('e')
        mantissa = mantissa.strip('0').rstrip('.')
        ex_sgn, ex_num = sgnd_ex[0], sgnd_ex[1:].lstrip('0')
        ex_sgn = '' if ex_sgn == '+' else ex_sgn
        return (mantissa or '0') + self._token + ex_sgn + (ex_num or '0')


class literal_sp(_literal):
    """ Fortran single precision real literal """
    _token = 'e'
    _decimals = 9


class literal_dp(_literal):
    """ Fortran double precision real literal """
    _token = 'd'
    _decimals = 17


class sum_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)


class product_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)
```