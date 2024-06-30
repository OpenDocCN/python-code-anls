# `D:\src\scipysrc\sympy\sympy\printing\jscode.py`

```
"""
Javascript code printer

The JavascriptCodePrinter converts single SymPy expressions into single
Javascript expressions, using the functions defined in the Javascript
Math object where possible.

"""

# 引入未来的注解支持
from __future__ import annotations
# 引入类型提示中的 Any 类型
from typing import Any

# 引入 SymPy 核心模块
from sympy.core import S
# 引入 SymPy 核心模块中的数值比较函数 equal_valued
from sympy.core.numbers import equal_valued
# 引入 SymPy 中的代码打印器基类 CodePrinter
from sympy.printing.codeprinter import CodePrinter
# 引入 SymPy 中的操作符优先级模块
from sympy.printing.precedence import precedence, PRECEDENCE


# 将 SymPy 函数映射到（参数条件，对应的 JavaScript 函数）的字典
# 用于 JavascriptCodePrinter._print_Function(self) 方法
known_functions = {
    'Abs': 'Math.abs',
    'acos': 'Math.acos',
    'acosh': 'Math.acosh',
    'asin': 'Math.asin',
    'asinh': 'Math.asinh',
    'atan': 'Math.atan',
    'atan2': 'Math.atan2',
    'atanh': 'Math.atanh',
    'ceiling': 'Math.ceil',
    'cos': 'Math.cos',
    'cosh': 'Math.cosh',
    'exp': 'Math.exp',
    'floor': 'Math.floor',
    'log': 'Math.log',
    'Max': 'Math.max',
    'Min': 'Math.min',
    'sign': 'Math.sign',
    'sin': 'Math.sin',
    'sinh': 'Math.sinh',
    'tan': 'Math.tan',
    'tanh': 'Math.tanh',
}


# 定义一个 JavaScript 代码打印器类 JavascriptCodePrinter，继承自 CodePrinter
class JavascriptCodePrinter(CodePrinter):
    """"A Printer to convert Python expressions to strings of JavaScript code
    """
    # 打印方法名称
    printmethod = '_javascript'
    # 语言类型为 JavaScript
    language = 'JavaScript'

    # 默认设置，包括精度、用户自定义函数和合同等
    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
    })

    # 初始化方法，接受一个 settings 参数
    def __init__(self, settings={}):
        # 调用父类 CodePrinter 的初始化方法
        CodePrinter.__init__(self, settings)
        # 将已知的函数从 known_functions 复制到 self.known_functions 中
        self.known_functions = dict(known_functions)
        # 获取用户自定义函数列表
        userfuncs = settings.get('user_functions', {})
        # 将用户自定义函数更新到 self.known_functions 中
        self.known_functions.update(userfuncs)

    # 评估索引位置的方法，返回评估结果乘以 5
    def _rate_index_position(self, p):
        return p * 5

    # 获取语句字符串的方法，在代码字符串末尾添加分号
    def _get_statement(self, codestring):
        return "%s;" % codestring

    # 获取注释字符串的方法，返回以双斜杠开头的注释
    def _get_comment(self, text):
        return "// {}".format(text)

    # 声明数值常量的方法，返回以 var 声明的 JavaScript 格式
    def _declare_number_const(self, name, value):
        return "var {} = {};".format(name, value.evalf(self._settings['precision']))

    # 格式化代码的方法，将代码块进行缩进
    def _format_code(self, lines):
        return self.indent_code(lines)

    # 遍历矩阵索引的方法，返回矩阵的行和列的组合生成器
    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    # 获取循环开头和结束的方法，根据索引生成对应的 JavaScript 循环语句
    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (var %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){"
        for i in indices:
            # JavaScript 数组从 0 开始，到 dimension-1 结束
            open_lines.append(loopstart % {
                'varble': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines
    # 定义一个方法，用于打印指数表达式
    def _print_Pow(self, expr):
        # 获取表达式的优先级
        PREC = precedence(expr)
        # 如果指数为 -1，返回其倒数形式
        if equal_valued(expr.exp, -1):
            return '1/%s' % (self.parenthesize(expr.base, PREC))
        # 如果指数为 0.5，返回其平方根形式
        elif equal_valued(expr.exp, 0.5):
            return 'Math.sqrt(%s)' % self._print(expr.base)
        # 如果指数为 1/3，返回其立方根形式
        elif expr.exp == S.One/3:
            return 'Math.cbrt(%s)' % self._print(expr.base)
        # 对于其他情况，返回其幂函数形式
        else:
            return 'Math.pow(%s, %s)' % (self._print(expr.base),
                                         self._print(expr.exp))

    # 定义一个方法，用于打印有理数表达式
    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d/%d' % (p, q)

    # 定义一个方法，用于打印取模表达式
    def _print_Mod(self, expr):
        # 获取表达式的参数
        num, den = expr.args
        # 获取表达式的优先级
        PREC = precedence(expr)
        # 对分子和分母进行括号处理
        snum, sden = [self.parenthesize(arg, PREC) for arg in expr.args]
        # 在 JavaScript 中，% 表示余数（与分子相同的符号），而不是模运算（与分母相同的符号）
        # 因此，只有当分子和分母具有相同的符号时，% 才能正确表示模运算
        if (num.is_nonnegative and den.is_nonnegative or
            num.is_nonpositive and den.is_nonpositive):
            return f"{snum} % {sden}"
        # 否则，返回调整后的模运算表达式
        return f"(({snum} % {sden}) + {sden}) % {sden}"

    # 定义一个方法，用于打印关系表达式
    def _print_Relational(self, expr):
        # 获取左右操作数的打印形式
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        # 获取关系运算符
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    # 定义一个方法，用于打印索引表达式
    def _print_Indexed(self, expr):
        # 计算一维数组的索引
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        # 逆序计算索引元素
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

    # 定义一个方法，用于打印索引标签
    def _print_Idx(self, expr):
        return self._print(expr.label)

    # 定义一个方法，用于打印自然对数的底 e
    def _print_Exp1(self, expr):
        return "Math.E"

    # 定义一个方法，用于打印数学常数 π
    def _print_Pi(self, expr):
        return 'Math.PI'

    # 定义一个方法，用于打印正无穷大
    def _print_Infinity(self, expr):
        return 'Number.POSITIVE_INFINITY'

    # 定义一个方法，用于打印负无穷大
    def _print_NegativeInfinity(self, expr):
        return 'Number.NEGATIVE_INFINITY'
    # 定义一个方法用于打印 Piecewise 表达式的代码表示
    def _print_Piecewise(self, expr):
        # 导入 Assignment 类型用于检查表达式中是否含有赋值语句
        from sympy.codegen.ast import Assignment
        # 检查最后一个条件是否为 True，否则会导致生成的函数在某些条件下无法返回结果
        if expr.args[-1].cond != True:
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        # 初始化空列表，用于存储生成的代码行
        lines = []
        # 如果表达式中含有赋值语句
        if expr.has(Assignment):
            # 遍历 Piecewise 表达式的每个分支
            for i, (e, c) in enumerate(expr.args):
                # 第一个条件分支
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                # 最后一个条件分支且条件为 True
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                # 中间的条件分支
                else:
                    lines.append("else if (%s) {" % self._print(c))
                # 打印当前分支的表达式部分
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            # 将所有生成的代码行拼接成一个字符串并返回
            return "\n".join(lines)
        else:
            # 如果 Piecewise 表达式被用在表达式中，需要进行内联操作符处理
            # 这种方式的缺点是，对于跨越多行的语句（如矩阵或索引表达式），内联操作符不起作用
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    # 定义一个方法用于打印矩阵元素的代码表示
    def _print_MatrixElement(self, expr):
        # 格式化输出矩阵元素的字符串表示，包括其所在矩阵的父级及其在矩阵中的索引
        return "{}[{}]".format(self.parenthesize(expr.parent,
            PRECEDENCE["Atom"], strict=True),
            expr.j + expr.i*expr.parent.shape[1])

    # 定义一个方法用于缩进给定的代码字符串或代码行列表
    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # 如果输入是字符串，则先将其按行分割，并递归调用本方法处理
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        # 定义缩进所用的制表符
        tab = "   "
        # 定义增加和减少缩进级别的标记
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        # 去除代码每行开头的空格和制表符
        code = [ line.lstrip(' \t') for line in code ]

        # 判断每行代码是否增加或减少缩进级别
        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

        # 初始化一个空列表用于存储格式化后的代码
        pretty = []
        level = 0
        # 遍历每行代码并根据增加和减少的缩进级别进行格式化
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        # 返回格式化后的代码列表
        return pretty
# 定义一个函数，用于将 SymPy 表达式转换为 JavaScript 代码的字符串
def jscode(expr, assign_to=None, **settings):
    """Converts an expr to a string of javascript code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, js_function_string)]. See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import jscode, symbols, Rational, sin, ceiling, Abs
    >>> x, tau = symbols("x, tau")
    >>> jscode((2*tau)**Rational(7, 2))
    '8*Math.sqrt(2)*Math.pow(tau, 7/2)'
    >>> jscode(sin(x), assign_to="s")
    's = Math.sin(x);'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    js_function_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")]
    ... }
    >>> jscode(Abs(x) + ceiling(x), user_functions=custom_functions)
    'fabs(x) + CEIL(x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(jscode(expr, tau))
    if (x > 0) {
       tau = x + 1;
    }
    else {
       tau = x;
    }

    Support for loops is provided through ``Indexed`` types. With
    """
    # 函数主体未提供，因此这里只是占位符注释，不包含任何实际代码
    pass
    """
    ``contract=True`` 这些表达式将被转换为循环，而
    ``contract=False`` 只会打印应该循环的赋值表达式：

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    设置索引数组的长度
    >>> y = IndexedBase('y', shape=(len_y,))
    创建一个 IndexedBase 对象 'y'，形状为 (len_y,)
    >>> t = IndexedBase('t', shape=(len_y,))
    创建一个 IndexedBase 对象 't'，形状为 (len_y,)
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    创建一个 IndexedBase 对象 'Dy'，形状为 (len_y-1,)
    >>> i = Idx('i', len_y-1)
    创建一个 Idx 对象 'i'，范围为 0 到 len_y-2
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    创建一个等式对象，将 (y[i+1]-y[i])/(t[i+1]-t[i]) 赋值给 Dy[i]
    >>> jscode(e.rhs, assign_to=e.lhs, contract=False)
    将等式右侧转换为 JavaScript 代码，并赋值给左侧，不使用循环

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    矩阵也是支持的，但必须为相同维度的 ``MatrixSymbol`` 提供给 ``assign_to``。
    注意，任何可以正常生成的表达式也可以存在于矩阵中：

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    创建一个矩阵对象 mat，包含多种表达式
    >>> A = MatrixSymbol('A', 3, 1)
    创建一个 MatrixSymbol 对象 'A'，形状为 3x1
    >>> print(jscode(mat, A))
    将矩阵 mat 转换为 JavaScript 代码，赋值给矩阵符号 A

    """

    return JavascriptCodePrinter(settings).doprint(expr, assign_to)
# 定义一个函数 print_jscode，用于打印给定表达式的 JavaScript 表示形式
def print_jscode(expr, **settings):
    """Prints the Javascript representation of the given expression.

       See jscode for the meaning of the optional arguments.
    """
    # 调用 jscode 函数生成表达式的 JavaScript 代码，并打印输出
    print(jscode(expr, **settings))
```