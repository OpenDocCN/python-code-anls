# `D:\src\scipysrc\sympy\sympy\parsing\sym_expr.py`

```
# 从 sympy.printing 模块中导入 pycode, ccode, fcode 函数，用于生成不同语言的代码表示
from sympy.printing import pycode, ccode, fcode
# 从 sympy.external 模块中导入 import_module 函数，用于动态导入模块
from sympy.external import import_module
# 从 sympy.utilities.decorator 模块中导入 doctest_depends_on 装饰器函数
from sympy.utilities.decorator import doctest_depends_on

# 尝试导入 'lfortran' 模块，如果成功则从 sympy.parsing.fortran.fortran_parser 导入 src_to_sympy 函数
lfortran = import_module('lfortran')
# 尝试导入 'clang.cindex' 模块，如果成功则从 sympy.parsing.c.c_parser 导入 parse_c 函数
cin = import_module('clang.cindex', import_kwargs={'fromlist': ['cindex']})

# 如果成功导入了 lfortran 模块，则从 sympy.parsing.fortran.fortran_parser 中导入 src_to_sympy 函数
if lfortran:
    from sympy.parsing.fortran.fortran_parser import src_to_sympy
# 如果成功导入了 cin 模块，则从 sympy.parsing.c.c_parser 中导入 parse_c 函数
if cin:
    from sympy.parsing.c.c_parser import parse_c

# 使用 doctest_depends_on 装饰器声明 SymPyExpression 类依赖于 'lfortran' 和 'clang.cindex' 模块
@doctest_depends_on(modules=['lfortran', 'clang.cindex'])
class SymPyExpression:  # type: ignore
    """Class to store and handle SymPy expressions

    This class will hold SymPy Expressions and handle the API for the
    conversion to and from different languages.

    It works with the C and the Fortran Parser to generate SymPy expressions
    which are stored here and which can be converted to multiple language's
    source code.

    Notes
    =====

    The module and its API are currently under development and experimental
    and can be changed during development.

    The Fortran parser does not support numeric assignments, so all the
    variables have been Initialized to zero.

    The module also depends on external dependencies:

    - LFortran which is required to use the Fortran parser
    - Clang which is required for the C parser

    Examples
    ========

    Example of parsing C code:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... int a,b;
    ... float c = 2, d =4;
    ... '''
    >>> a = SymPyExpression(src, 'c')
    >>> a.return_expr()
    [Declaration(Variable(a, type=intc)),
    Declaration(Variable(b, type=intc)),
    Declaration(Variable(c, type=float32, value=2.0)),
    Declaration(Variable(d, type=float32, value=4.0))]

    An example of variable definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src2 = '''
    ... integer :: a, b, c, d
    ... real :: p, q, r, s
    ... '''
    >>> p = SymPyExpression()
    >>> p.convert_to_expr(src2, 'f')
    >>> p.convert_to_c()
    ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0']

    An example of Assignment:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src3 = '''
    ... integer :: a, b, c, d, e
    ... d = a + b - c
    ... e = b * d + c * e / a
    ... '''
    >>> p = SymPyExpression(src3, 'f')
    >>> p.convert_to_python()
    ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'e = 0', 'd = a + b - c', 'e = b*d + c*e/a']

    An example of function definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... integer function f(a,b)
    ... integer, intent(in) :: a, b
    ... integer :: r
    ... end function
    ... '''
    >>> a = SymPyExpression(src, 'f')
    >>> a.convert_to_python()
    ['def f(a, b):\\n   f = 0\\n    r = 0\\n    return f']
    """
    def __init__(self, source_code = None, mode = None):
        """Constructor for SymPyExpression class"""
        super().__init__()
        # 如果没有指定 mode 或者 source_code，初始化表达式列表为空
        if not(mode or source_code):
            self._expr = []
        elif mode:
            if source_code:
                # 如果指定了 mode 和 source_code，则根据 mode 进行解析
                if mode.lower() == 'f':
                    # 如果 mode 是 'f'，并且 lfortran 已安装，则将源码转换为 SymPy 表达式
                    if lfortran:
                        self._expr = src_to_sympy(source_code)
                    else:
                        # 如果 lfortran 没有安装，则抛出 ImportError 异常
                        raise ImportError("LFortran is not installed, cannot parse Fortran code")
                elif mode.lower() == 'c':
                    # 如果 mode 是 'c'，并且 cin 已安装，则将源码解析为 SymPy 表达式
                    if cin:
                        self._expr = parse_c(source_code)
                    else:
                        # 如果 cin 没有安装，则抛出 ImportError 异常
                        raise ImportError("Clang is not installed, cannot parse C code")
                else:
                    # 如果 mode 不是 'f' 也不是 'c'，则抛出 NotImplementedError 异常
                    raise NotImplementedError(
                        'Parser for specified language is not implemented'
                    )
            else:
                # 如果没有提供 source_code，则抛出 ValueError 异常
                raise ValueError('Source code not present')
        else:
            # 如果没有指定 mode，则抛出 ValueError 异常
            raise ValueError('Please specify a mode for conversion')

    def convert_to_expr(self, src_code, mode):
        """Converts the given source code to SymPy Expressions

        Attributes
        ==========

        src_code : String
            the source code or filename of the source code that is to be
            converted

        mode: String
            the mode to determine which parser is to be used according to
            the language of the source code
            f or F for Fortran
            c or C for C/C++

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b) result(r)
        ... integer, intent(in) :: a, b
        ... integer :: x
        ... r = a + b -x
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(r, type=integer, value=0)),
        Declaration(Variable(x, type=integer, value=0)),
        Assignment(Variable(r), a + b - x),
        Return(Variable(r))
        ))]

        """
        # 根据 mode 将源码转换为 SymPy 表达式
        if mode.lower() == 'f':
            # 如果 mode 是 'f'，并且 lfortran 已安装，则将源码转换为 SymPy 表达式
            if lfortran:
                self._expr = src_to_sympy(src_code)
            else:
                # 如果 lfortran 没有安装，则抛出 ImportError 异常
                raise ImportError("LFortran is not installed, cannot parse Fortran code")
        elif mode.lower() == 'c':
            # 如果 mode 是 'c'，并且 cin 已安装，则将源码解析为 SymPy 表达式
            if cin:
                self._expr = parse_c(src_code)
            else:
                # 如果 cin 没有安装，则抛出 ImportError 异常
                raise ImportError("Clang is not installed, cannot parse C code")
        else:
            # 如果 mode 不是 'f' 也不是 'c'，则抛出 NotImplementedError 异常
            raise NotImplementedError(
                "Parser for specified language has not been implemented"
            )
    def convert_to_python(self):
        """Returns a list with Python code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression(src2, 'f')
        >>> p.convert_to_python()
        ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'p = 0.0', 'q = 0.0', 'r = 0.0', 's = 0.0', 'c = a/b', 'd = c/a', 's = p/q', 'r = q/p']

        """
        # 初始化一个空列表用于存储转换后的Python代码
        self._pycode = []
        # 遍历表达式列表，对每个表达式生成对应的Python代码，并添加到_pycode列表中
        for iter in self._expr:
            self._pycode.append(pycode(iter))
        # 返回生成的Python代码列表
        return self._pycode

    def convert_to_c(self):
        """Returns a list with the c source code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src2, 'f')
        >>> p.convert_to_c()
        ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0', 'c = a/b;', 'd = c/a;', 's = p/q;', 'r = q/p;']

        """
        # 初始化一个空列表用于存储转换后的C语言代码
        self._ccode = []
        # 遍历表达式列表，对每个表达式生成对应的C语言代码，并添加到_ccode列表中
        for iter in self._expr:
            self._ccode.append(ccode(iter))
        # 返回生成的C语言代码列表
        return self._ccode

    def convert_to_fortran(self):
        """Returns a list with the fortran source code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression(src2, 'f')
        >>> p.convert_to_fortran()
        ['      integer*4 a', '      integer*4 b', '      integer*4 c', '      integer*4 d', '      real*8 p', '      real*8 q', '      real*8 r', '      real*8 s', '      c = a/b', '      d = c/a', '      s = p/q', '      r = q/p']

        """
        # 初始化一个空列表用于存储转换后的Fortran代码
        self._fcode = []
        # 遍历表达式列表，对每个表达式生成对应的Fortran代码，并添加到_fcode列表中
        for iter in self._expr:
            self._fcode.append(fcode(iter))
        # 返回生成的Fortran代码列表
        return self._fcode
    def return_expr(self):
        """Returns the expression list
        
        返回表达式列表

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b)
        ... integer, intent(in) :: a, b
        ... integer :: r
        ... r = a+b
        ... f = r
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(f, type=integer, value=0)),
        Declaration(Variable(r, type=integer, value=0)),
        Assignment(Variable(f), Variable(r)),
        Return(Variable(f))
        ))]

        """
        # 返回存储在对象属性中的表达式列表
        return self._expr
```