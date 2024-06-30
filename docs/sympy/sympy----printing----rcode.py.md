# `D:\src\scipysrc\sympy\sympy\printing\rcode.py`

```
"""
R code printer

The RCodePrinter converts single SymPy expressions into single R expressions,
using the functions defined in math.h where possible.



"""

# 导入必要的模块和库
from __future__ import annotations
from typing import Any

# 导入 SymPy 中需要的模块和函数
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range

# dictionary mapping SymPy function to (argument_conditions, C_function).
# Used in RCodePrinter._print_Function(self)
# 定义已知 SymPy 函数和它们对应的 R 语言函数
known_functions = {
    #"Abs": [(lambda x: not x.is_integer, "fabs")],
    "Abs": "abs",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "erf": "erf",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "floor": "floor",
    "ceiling": "ceiling",
    "sign": "sign",
    "Max": "max",
    "Min": "min",
    "factorial": "factorial",
    "gamma": "gamma",
    "digamma": "digamma",
    "trigamma": "trigamma",
    "beta": "beta",
    "sqrt": "sqrt",  # To enable automatic rewrite
}

# These are the core reserved words in the R language. Taken from:
# https://cran.r-project.org/doc/manuals/r-release/R-lang.html#Reserved-words
# R 语言的保留关键字列表
reserved_words = ['if',
                  'else',
                  'repeat',
                  'while',
                  'function',
                  'for',
                  'in',
                  'next',
                  'break',
                  'TRUE',
                  'FALSE',
                  'NULL',
                  'Inf',
                  'NaN',
                  'NA',
                  'NA_integer_',
                  'NA_real_',
                  'NA_complex_',
                  'NA_character_',
                  'volatile']

# 定义 RCodePrinter 类，用于将 SymPy 表达式转换为 R 语言代码字符串
class RCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of R code"""
    printmethod = "_rcode"
    language = "R"

    # 默认设置，包括精度、用户定义函数、是否合并、解引用等
    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 15,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
    })

    # R 语言中的运算符映射
    _operators = {
       'and': '&',
        'or': '|',
       'not': '!',
    }

    # 关系运算符的映射
    _relationals: dict[str, str] = {}

    # 初始化方法，设置用户定义函数和解引用等参数
    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)

    # 打分函数索引位置的计算方法
    def _rate_index_position(self, p):
        return p*5

    # 获取语句的方法，返回带分号的代码字符串
    def _get_statement(self, codestring):
        return "%s;" % codestring

    # 获取注释的方法，返回以双斜杠开头的注释字符串
    def _get_comment(self, text):
        return "// {}".format(text)
    # 定义一个方法来声明数值常量，返回格式化的赋值语句
    def _declare_number_const(self, name, value):
        return "{} = {};".format(name, value)
    
    # 定义一个方法来格式化代码，调用另一个方法来缩进代码行
    def _format_code(self, lines):
        return self.indent_code(lines)
    
    # 定义一个方法来遍历矩阵的索引，返回一个生成器，生成所有可能的索引组合
    def _traverse_matrix_indices(self, mat):
        # 获取矩阵的行数和列数
        rows, cols = mat.shape
        # 使用生成器表达式返回所有的索引组合
        return ((i, j) for i in range(rows) for j in range(cols))
    
    # 定义一个方法来获取循环的开头和结尾代码行，输入参数是索引的集合
    def _get_loop_opening_ending(self, indices):
        """Returns a tuple (open_lines, close_lines) containing lists of codelines
        """
        # 初始化两个空列表，用于存放循环的开头和结尾代码行
        open_lines = []
        close_lines = []
        # 循环遍历索引集合
        loopstart = "for (%(var)s in %(start)s:%(end)s){"
        for i in indices:
            # R 数组从1开始，到维度结束
            # 根据索引的标签、起始和结束位置生成循环的开头代码行，并添加到 open_lines 列表中
            open_lines.append(loopstart % {
                'var': self._print(i.label),  # 打印索引的标签
                'start': self._print(i.lower+1),  # 打印索引的起始位置加1
                'end': self._print(i.upper + 1)})  # 打印索引的结束位置加1
            # 生成循环的结尾代码行，并添加到 close_lines 列表中
            close_lines.append("}")
        # 返回包含开头和结尾代码行列表的元组
        return open_lines, close_lines
    
    # 定义一个方法来打印幂运算的表达式，根据幂运算的底数和指数，返回相应的字符串表示
    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)  # 如果幂函数在已知函数中，调用打印函数进行打印
        PREC = precedence(expr)  # 获取表达式的优先级
        # 根据指数是否为-1或0.5，返回不同格式的幂运算表达式
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))  # 返回倒数表达式
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)  # 返回平方根表达式
        else:
            return '%s^%s' % (self.parenthesize(expr.base, PREC),
                                 self.parenthesize(expr.exp, PREC))  # 返回一般的幂运算表达式
    
    # 定义一个方法来打印有理数表达式，根据分子和分母，返回相应的字符串表示
    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d.0/%d.0' % (p, q)  # 返回有理数的字符串表示
    
    # 定义一个方法来打印索引表达式，根据索引的列表，返回相应的字符串表示
    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]  # 打印所有索引的字符串表示
        return "%s[%s]" % (self._print(expr.base.label), ", ".join(inds))  # 返回带索引的基本表达式的字符串表示
    
    # 定义一个方法来打印索引标签表达式，返回其字符串表示
    def _print_Idx(self, expr):
        return self._print(expr.label)  # 返回索引标签的字符串表示
    
    # 定义一个方法来打印自然对数的底数表达式，返回其字符串表示
    def _print_Exp1(self, expr):
        return "exp(1)"  # 返回自然对数的底数的字符串表示
    
    # 定义一个方法来打印圆周率表达式，返回其字符串表示
    def _print_Pi(self, expr):
        return 'pi'  # 返回圆周率的字符串表示
    
    # 定义一个方法来打印正无穷大表达式，返回其字符串表示
    def _print_Infinity(self, expr):
        return 'Inf'  # 返回正无穷大的字符串表示
    
    # 定义一个方法来打印负无穷大表达式，返回其字符串表示
    def _print_NegativeInfinity(self, expr):
        return '-Inf'  # 返回负无穷大的字符串表示
    def _print_Assignment(self, expr):
        # 导入需要的模块和类
        from sympy.codegen.ast import Assignment
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase

        # 提取表达式的左右两侧
        lhs = expr.lhs
        rhs = expr.rhs

        # 特殊情况：处理多行赋值的情况（目前被注释掉的代码）
        #if isinstance(expr.rhs, Piecewise):
        #    from sympy.functions.elementary.piecewise import Piecewise
        #    expressions = []
        #    conditions = []
        #    for (e, c) in rhs.args:
        #        expressions.append(Assignment(lhs, e))
        #        conditions.append(c)
        #    temp = Piecewise(*zip(expressions, conditions))
        #    return self._print(temp)

        # 如果左侧是 MatrixSymbol 类型
        if isinstance(lhs, MatrixSymbol):
            # 遍历矩阵的索引并形成 Assignment 对象，每个对象进行打印
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return "\n".join(lines)
        
        # 如果启用了 contract 选项并且左右侧至少有一个包含 IndexedBase
        elif self._settings["contract"] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            # 检查是否需要进行循环打印
            return self._doprint_loops(rhs, lhs)
        
        # 默认情况下打印左右侧的赋值语句
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def _print_Piecewise(self, expr):
        # 这个方法仅处理内联的 if 构造
        # 最顶层的 Piecewise 在 doprint() 中处理
        if expr.args[-1].cond == True:
            last_line = "%s" % self._print(expr.args[-1].expr)
        else:
            last_line = "ifelse(%s,%s,NA)" % (self._print(expr.args[-1].cond), self._print(expr.args[-1].expr))
        code = last_line
        for e, c in reversed(expr.args[:-1]):
            code = "ifelse(%s,%s," % (self._print(c), self._print(e)) + code + ")"
        return code

    def _print_ITE(self, expr):
        # 导入 Piecewise 后重写并打印
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_MatrixElement(self, expr):
        # 打印矩阵元素的表达式
        return "{}[{}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
            strict=True), expr.j + expr.i * expr.parent.shape[1])

    def _print_Symbol(self, expr):
        # 打印符号名称，如果需要解引用，则打印带星号的名称
        name = super()._print_Symbol(expr)
        if expr in self._dereference:
            return '(*{})'.format(name)
        else:
            return name
    # 打印关系表达式，将左操作数、操作符和右操作数格式化为字符串返回
    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)  # 获取左操作数的打印形式
        rhs_code = self._print(expr.rhs)  # 获取右操作数的打印形式
        op = expr.rel_op  # 获取关系操作符
        return "{} {} {}".format(lhs_code, op, rhs_code)  # 返回格式化后的关系表达式字符串

    # 打印增强赋值表达式，将左操作数、操作符和右操作数格式化为字符串返回
    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)  # 获取左操作数的打印形式
        op = expr.op  # 获取增强赋值操作符
        rhs_code = self._print(expr.rhs)  # 获取右操作数的打印形式
        return "{} {} {};".format(lhs_code, op, rhs_code)  # 返回格式化后的增强赋值表达式字符串

    # 打印 for 循环表达式，包括目标变量、迭代器和循环体
    def _print_For(self, expr):
        target = self._print(expr.target)  # 获取目标变量的打印形式
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args  # 获取 Range 迭代器的起始值、终止值和步长
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)  # 获取循环体的打印形式
        return 'for({target} in seq(from={start}, to={stop}, by={step}){{\n{body}\n}}'.format(
            target=target, start=start, stop=stop-1, step=step, body=body)
        # 返回格式化后的 for 循环表达式字符串，使用 seq 函数进行范围迭代

    # 缩进代码的函数，接受代码字符串或代码行列表作为输入
    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))  # 递归调用处理代码行列表
            return ''.join(code_lines)  # 返回连接后的字符串形式代码

        tab = "   "  # 定义缩进的空格数
        inc_token = ('{', '(', '{\n', '(\n')  # 增加缩进的标记
        dec_token = ('}', ')')  # 减少缩进的标记

        code = [ line.lstrip(' \t') for line in code ]  # 去除每行代码的前导空格和制表符

        # 计算每行代码是否增加或减少缩进的标志位
        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

        pretty = []  # 初始化格式化后的代码列表
        level = 0  # 初始化缩进级别
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)  # 如果是空行，则直接添加到格式化列表
                continue
            level -= decrease[n]  # 根据减少标志降低缩进级别
            pretty.append("%s%s" % (tab*level, line))  # 添加带有正确缩进的代码行
            level += increase[n]  # 根据增加标志增加缩进级别
        return pretty  # 返回格式化后的代码列表
# 将 SymPy 表达式转换为 R 代码字符串
def rcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of r code

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
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired R string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        rfunction_string)] or [(argument_test, rfunction_formater)]. See below
        for examples.
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

    >>> from sympy import rcode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rcode((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau^(7.0/2.0)'
    >>> rcode(sin(x), assign_to="s")
    's = sin(x);'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> rcode(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    or if the R-function takes a subset of the original arguments:

    >>> rcode(2**x + 3**x, user_functions={'Pow': [
    ...   (lambda b, e: b == 2, lambda b, e: 'exp2(%s)' % e),
    ...   (lambda b, e: b != 2, 'pow')]})
    'exp2(x) + pow(3, x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    """

    # 如果指定了 assign_to 参数，则生成带赋值语句的 R 代码字符串
    if assign_to:
        code_str = f"{assign_to} = "
    else:
        code_str = ""

    # 获取 human 参数，默认为 True，控制返回值类型
    human = settings.get('human', True)

    # 通过 contract 参数控制是否生成索引的循环
    contract = settings.get('contract', True)

    # 读取 precision 参数，设置数字精度，默认为 15
    precision = settings.get('precision', 15)

    # 获取 user_functions 参数，用于自定义函数映射
    user_functions = settings.get('user_functions', {})

    # 生成 R 代码的基础字符串表示
    code_str += "TODO: Generate R code here"

    # 根据 human 参数返回不同类型的结果
    if human:
        return code_str
    else:
        return ("TODO: symbols_to_declare", "TODO: not_supported_functions", code_str)
    """
    Generate and print code for given SymPy expression 'expr'.

    This function converts a SymPy expression into a string representation
    of code, which can be assigned to a variable or used in various contexts.

    Parameters:
    expr : sympy.Basic
        The SymPy expression to be converted into code.
    assign_to : str or sympy.Basic, optional
        Variable or expression to assign the generated code to.
    contract : bool, optional
        If True, generates code that supports loops (Indexed types).
        If False, generates straightforward assignment expressions.

    Returns:
    str
        String representation of the generated code.

    Examples:
    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(rcode(expr, assign_to=tau))
    tau = ifelse(x > 0,x + 1,x);

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> rcode(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(rcode(mat, A))
    A[0] = x^2;
    A[1] = ifelse(x > 0,x + 1,x);
    A[2] = sin(x);
    """

    return RCodePrinter(settings).doprint(expr, assign_to)
# 定义一个函数 print_rcode，用于打印给定表达式的 R 语言表示形式
def print_rcode(expr, **settings):
    """Prints R representation of the given expression."""
    # 调用 rcode 函数生成给定表达式的 R 语言表示形式，并打印出来
    print(rcode(expr, **settings))
```