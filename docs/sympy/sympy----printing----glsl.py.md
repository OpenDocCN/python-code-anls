# `D:\src\scipysrc\sympy\sympy\printing\glsl.py`

```
from __future__ import annotations
# 导入用于支持类型注解的模块（Python 3.7及以后版本默认支持）

from sympy.core import Basic, S
# 导入 Sympy 的核心基础类 Basic 和 S

from sympy.core.function import Lambda
# 导入 Sympy 的 Lambda 函数支持

from sympy.core.numbers import equal_valued
# 导入 Sympy 的数值相等比较支持

from sympy.printing.codeprinter import CodePrinter
# 导入 Sympy 的代码打印器基类 CodePrinter

from sympy.printing.precedence import precedence
# 导入 Sympy 的运算符优先级支持

from functools import reduce
# 导入 functools 模块的 reduce 函数

known_functions = {
    'Abs': 'abs',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'acos': 'acos',
    'asin': 'asin',
    'atan': 'atan',
    'atan2': 'atan',
    'ceiling': 'ceil',
    'floor': 'floor',
    'sign': 'sign',
    'exp': 'exp',
    'log': 'log',
    'add': 'add',
    'sub': 'sub',
    'mul': 'mul',
    'pow': 'pow'
}
# 已知的函数映射表，将 Sympy 中的函数名映射为 GLSL 中对应的函数名

class GLSLPrinter(CodePrinter):
    """
    Rudimentary, generic GLSL printing tools.

    Additional settings:
    'use_operators': Boolean (should the printer use operators for +,-,*, or functions?)
    """
    _not_supported: set[Basic] = set()
    # 设置一个不支持的基础类集合，默认为空集合

    printmethod = "_glsl"
    # 打印方法名称为 "_glsl"

    language = "GLSL"
    # 设定代码输出语言为 GLSL

    _default_settings = dict(CodePrinter._default_settings, **{
        'use_operators': True,
        'zero': 0,
        'mat_nested': False,
        'mat_separator': ',\n',
        'mat_transpose': False,
        'array_type': 'float',
        'glsl_types': True,

        'precision': 9,
        'user_functions': {},
        'contract': True,
    })
    # 默认的代码打印设置，继承自 CodePrinter 的默认设置，并添加了 GLSL 打印器特有的设置选项

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        # 调用 CodePrinter 的初始化方法来初始化基础设置
        self.known_functions = dict(known_functions)
        # 使用预定义的函数映射表初始化 GLSLPrinter 的已知函数映射
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        # 更新已知函数映射，允许用户自定义添加额外的函数映射

    def _rate_index_position(self, p):
        return p*5
    # 定义一个方法，用于计算索引位置的评分，返回输入参数 p 的五倍值

    def _get_statement(self, codestring):
        return "%s;" % codestring
    # 定义一个方法，用于生成一个语句，输入参数 codestring 将被格式化为一个带分号的语句字符串

    def _get_comment(self, text):
        return "// {}".format(text)
    # 定义一个方法，用于生成注释文本的 GLSL 格式，格式为双斜杠加上注释内容

    def _declare_number_const(self, name, value):
        return "float {} = {};".format(name, value)
    # 定义一个方法，用于声明一个数值常量，在 GLSL 中声明一个浮点数常量名为 name，值为 value

    def _format_code(self, lines):
        return self.indent_code(lines)
    # 定义一个方法，用于格式化代码，调用 indent_code 方法缩进代码

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)
        # 如果输入的是字符串，则将其拆分成行并递归调用 indent_code 处理

        tab = "   "
        # 设置缩进字符串为四个空格

        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')
        # 设置增加和减少缩进的标记

        code = [line.lstrip(' \t') for line in code]
        # 对代码每行进行去除左侧空格和制表符的处理

        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token))) for line in code]
        # 根据行末尾和开头的标记确定每行是否增加或减少缩进的情况

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        # 遍历代码的每一行，根据增减缩进的情况进行适当的缩进处理，并生成格式化后的代码列表

        return pretty
    # 返回格式化后的代码列表或字符串
    # 打印 MatrixBase 类的矩阵对象 mat
    def _print_MatrixBase(self, mat):
        # 获取矩阵分隔符设置
        mat_separator = self._settings['mat_separator']
        # 获取矩阵转置设置
        mat_transpose = self._settings['mat_transpose']
        # 检查是否是列向量
        column_vector = (mat.rows == 1) if mat_transpose else (mat.cols == 1)
        # 根据转置需求选择适当的矩阵 A
        A = mat.transpose() if mat_transpose != column_vector else mat

        # 获取 GLSL 类型设置
        glsl_types = self._settings['glsl_types']
        # 获取数组类型设置
        array_type = self._settings['array_type']
        # 计算数组大小
        array_size = A.cols * A.rows
        # 构造数组类型及大小的字符串
        array_constructor = "{}[{}]".format(array_type, array_size)

        # 根据矩阵列数进行不同的输出
        if A.cols == 1:
            return self._print(A[0]);  # 打印矩阵的第一列
        # 如果矩阵行数和列数均不大于4，并且支持 GLSL 类型
        if A.rows <= 4 and A.cols <= 4 and glsl_types:
            if A.rows == 1:
                return "vec{}{}".format(
                    A.cols, A.table(self, rowstart='(', rowend=')')
                )  # 输出 GLSL 的向量表示
            elif A.rows == A.cols:
                return "mat{}({})".format(
                    A.rows, A.table(self, rowsep=', ', rowstart='', rowend='')
                )  # 输出 GLSL 的方阵表示
            else:
                return "mat{}x{}({})".format(
                    A.cols, A.rows,
                    A.table(self, rowsep=', ', rowstart='', rowend='')
                )  # 输出 GLSL 的非方阵表示
        elif S.One in A.shape:  # 如果矩阵的形状中包含单位元素
            return "{}({})".format(
                array_constructor,
                A.table(self, rowsep=mat_separator, rowstart='', rowend='')
            )  # 输出数组表示，用于一维化矩阵
        elif not self._settings['mat_nested']:  # 如果不支持嵌套矩阵
            return "{}(\n{}\n) /* a {}x{} matrix */".format(
                array_constructor,
                A.table(self, rowsep=mat_separator, rowstart='', rowend=''),
                A.rows, A.cols
            )  # 输出简单的矩阵表示，带有注释
        elif self._settings['mat_nested']:  # 如果支持嵌套矩阵
            return "{}[{}][{}](\n{}\n)".format(
                array_type, A.rows, A.cols,
                A.table(self, rowsep=mat_separator, rowstart='float[](', rowend=')')
            )  # 输出嵌套数组表示，用于 GLSL 中的嵌套矩阵

    # 打印 SparseRepMatrix 类的稀疏矩阵对象 mat
    def _print_SparseRepMatrix(self, mat):
        # 禁止打印稀疏矩阵，返回不支持的提示
        return self._print_not_supported(mat)

    # 遍历矩阵的索引，返回索引生成器
    def _traverse_matrix_indices(self, mat):
        # 获取矩阵转置设置
        mat_transpose = self._settings['mat_transpose']
        # 根据转置设置确定行列数
        if mat_transpose:
            rows, cols = mat.shape
        else:
            cols, rows = mat.shape
        # 生成所有可能的矩阵元素索引
        return ((i, j) for i in range(cols) for j in range(rows))

    # 打印 MatrixElement 对象 expr
    def _print_MatrixElement(self, expr):
        # 获取矩阵嵌套设置
        nest = self._settings['mat_nested']
        # 获取 GLSL 类型设置
        glsl_types = self._settings['glsl_types']
        # 获取矩阵转置设置
        mat_transpose = self._settings['mat_transpose']
        # 根据转置设置确定行列数及元素位置
        if mat_transpose:
            cols, rows = expr.parent.shape
            i, j = expr.j, expr.i
        else:
            rows, cols = expr.parent.shape
            i, j = expr.i, expr.j
        # 打印矩阵表达式的父对象
        pnt = self._print(expr.parent)
        # 如果支持 GLSL 类型并且矩阵小于等于4x4，或者支持嵌套
        if glsl_types and ((rows <= 4 and cols <= 4) or nest):
            return "{}[{}][{}]".format(pnt, i, j)  # 输出 GLSL 风格的元素索引表示
        else:
            return "{}[{}]".format(pnt, i + j * rows)  # 输出普通的元素索引表示
    # 将表达式列表中的每个元素打印为字符串，并用逗号分隔连接成一个字符串
    l = ', '.join(self._print(item) for item in expr)
    # 获取当前设置中的 GLSL 类型列表
    glsl_types = self._settings['glsl_types']
    # 获取当前设置中的数组类型
    array_type = self._settings['array_type']
    # 计算表达式列表的长度，即数组的大小
    array_size = len(expr)
    # 根据数组类型和大小构造数组的声明字符串
    array_constructor = '{}[{}]'.format(array_type, array_size)

    # 如果数组大小小于等于4并且设置了 GLSL 类型
    if array_size <= 4 and glsl_types:
        # 返回一个 GLSL 向量的构造字符串
        return 'vec{}({})'.format(array_size, l)
    else:
        # 返回通用的数组构造字符串
        return '{}({})'.format(array_constructor, l)

# 将 _print_list 方法赋值给 _print_tuple 和 _print_Tuple 方法，作为它们的别名
_print_tuple = _print_list
_print_Tuple = _print_list

    # 根据索引列表生成循环的开头和结尾部分的代码行
def _get_loop_opening_ending(self, indices):
    open_lines = []  # 存储循环开头的代码行列表
    close_lines = []  # 存储循环结尾的代码行列表
    loopstart = "for (int %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){"
    # 遍历索引列表，为每个索引生成对应的循环开头和结尾代码行
    for i in indices:
        # GLSL 数组从0开始，到维度-1结束
        open_lines.append(loopstart % {
            'varble': self._print(i.label),  # 替换循环变量
            'start': self._print(i.lower),   # 替换循环起始值
            'end': self._print(i.upper + 1)})  # 替换循环结束值
        close_lines.append("}")  # 添加循环结尾的闭合大括号
    return open_lines, close_lines  # 返回循环的开头和结尾代码行列表

    # 根据函数名称和参数打印函数调用表达式
def _print_Function_with_args(self, func, func_args):
    # 如果函数在已知函数列表中
    if func in self.known_functions:
        cond_func = self.known_functions[func]
        func = None
        # 如果条件函数是字符串，则将其作为函数名
        if isinstance(cond_func, str):
            func = cond_func
        else:
            # 否则，根据条件查找适合的函数
            for cond, func in cond_func:
                if cond(func_args):
                    break
        # 如果找到适合的函数，则尝试调用它，并返回结果字符串
        if func is not None:
            try:
                return func(*[self.parenthesize(item, 0) for item in func_args])
            except TypeError:
                return '{}({})'.format(func, self.stringify(func_args, ", "))
    # 如果函数是 Lambda 类型的函数
    elif isinstance(func, Lambda):
        # 内联函数，直接打印其调用表达式
        return self._print(func(*func_args))
    else:
        # 如果函数不在已知函数列表中，返回不支持打印的提示
        return self._print_not_supported(func)
    def _print_Piecewise(self, expr):
        # 导入赋值表达式类
        from sympy.codegen.ast import Assignment
        # 检查 Piecewise 表达式的最后一个条件是否为 True，确保函数有返回结果
        if expr.args[-1].cond != True:
            # 如果最后一个条件不是 True，则抛出数值错误
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        
        # 初始化代码行列表
        lines = []
        
        # 如果表达式中包含赋值表达式
        if expr.has(Assignment):
            # 遍历 Piecewise 表达式的每个分支
            for i, (e, c) in enumerate(expr.args):
                # 第一个分支使用 if
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                # 最后一个分支且条件为 True 使用 else
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                # 其他分支使用 else if
                else:
                    lines.append("else if (%s) {" % self._print(c))
                
                # 打印表达式的代码
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            
            # 返回拼接后的代码段
            return "\n".join(lines)
        
        else:
            # 如果 Piecewise 表达式用于一个表达式中，需要进行内联运算符处理
            # 这种方式的缺点是对于跨多行的语句（如矩阵或索引表达式），内联运算符将不起作用
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c),
                                               self._print(e))
                       for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            
            # 返回内联运算符处理后的代码段
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_Idx(self, expr):
        # 打印索引对象的标签
        return self._print(expr.label)

    def _print_Indexed(self, expr):
        # 计算一维数组的索引
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        
        # 反向遍历索引维度
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        
        # 返回格式化后的索引字符串
        return "{}[{}]".format(
            self._print(expr.base.label),
            self._print(elem)
        )

    def _print_Pow(self, expr):
        # 设置操作符的优先级
        PREC = precedence(expr)
        
        # 处理指数为 -1 的情况
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        # 处理指数为 0.5 的情况
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            try:
                # 尝试将指数转换为浮点数打印
                e = self._print(float(expr.exp))
            except TypeError:
                # 如果转换失败，则直接打印指数表达式
                e = self._print(expr.exp)
            
            # 使用函数和参数打印 pow 函数调用
            return self._print_Function_with_args('pow', (
                self._print(expr.base),
                e
            ))

    def _print_int(self, expr):
        # 打印整数对象的浮点表示
        return str(float(expr))

    def _print_Rational(self, expr):
        # 打印有理数对象的浮点表示
        return "{}.0/{}.0".format(expr.p, expr.q)
    # 打印关系表达式，格式为左操作数、关系运算符、右操作数的字符串
    def _print_Relational(self, expr):
        # 获取左操作数的打印形式
        lhs_code = self._print(expr.lhs)
        # 获取右操作数的打印形式
        rhs_code = self._print(expr.rhs)
        # 获取关系运算符
        op = expr.rel_op
        # 返回格式化后的字符串，包括左操作数、运算符和右操作数
        return "{} {} {}".format(lhs_code, op, rhs_code)

    # 打印加法表达式
    def _print_Add(self, expr, order=None):
        # 如果设置允许使用操作符，则调用父类的打印方法
        if self._settings['use_operators']:
            return CodePrinter._print_Add(self, expr, order=order)

        # 获取表达式中的有序项
        terms = expr.as_ordered_terms()

        # 定义函数，将项分成负数项和正数项
        def partition(p,l):
            return reduce(lambda x, y: (x[0]+[y], x[1]) if p(y) else (x[0], x[1]+[y]), l,  ([], []))
        
        # 定义函数，用于打印加法函数
        def add(a,b):
            return self._print_Function_with_args('add', (a, b))
        
        # 将项按照是否可能提取负号分成负数项和正数项
        neg, pos = partition(lambda arg: arg.could_extract_minus_sign(), terms)

        # 如果存在正数项
        if pos:
            # 将正数项相加，得到字符串表示
            s = pos = reduce(lambda a,b: add(a,b), (self._print(t) for t in pos))
        else:
            # 如果没有正数项，则打印零
            s = pos = self._print(self._settings['zero'])

        # 如果存在负数项
        if neg:
            # 将负数项的绝对值相加，得到字符串表示
            neg = reduce(lambda a,b: add(a,b), (self._print(-n) for n in neg))
            # 将负数项的和从正数项中减去，得到最终的加法表达式字符串表示
            s = self._print_Function_with_args('sub', (pos,neg))

        # 返回最终的加法表达式字符串
        return s

    # 打印乘法表达式
    def _print_Mul(self, expr, **kwargs):
        # 如果设置允许使用操作符，则调用父类的打印方法
        if self._settings['use_operators']:
            return CodePrinter._print_Mul(self, expr, **kwargs)
        
        # 获取表达式中的有序因子
        terms = expr.as_ordered_factors()

        # 定义函数，用于打印乘法函数
        def mul(a,b):
            return self._print_Function_with_args('mul', (a,b))

        # 将所有因子相乘，得到字符串表示
        s = reduce(lambda a,b: mul(a,b), (self._print(t) for t in terms))

        # 返回乘法表达式的字符串表示
        return s
# 将 SymPy 表达式转换为 GLSL 代码字符串的函数
def glsl_code(expr, assign_to=None, **settings):
    """Converts an expr to a string of GLSL code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used for naming the variable or variables
        to which the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol`` or ``Indexed`` type object. In cases where ``expr``
        would be printed as an array, a list of string or ``Symbol`` objects
        can also be passed.

        This is helpful in case of line-wrapping, or for expressions that
        generate multi-line statements.  It can also be used to spread an array-like
        expression into multiple assignments.
    use_operators: bool, optional
        If set to False, then *,/,+,- operators will be replaced with functions
        mul, add, and sub, which must be implemented by the user, e.g. for
        implementing non-standard rings or emulated quad/octal precision.
        [default=True]
    glsl_types: bool, optional
        Set this argument to ``False`` in order to avoid using the ``vec`` and ``mat``
        types.  The printer will instead use arrays (or nested arrays).
        [default=True]
    mat_nested: bool, optional
        GLSL version 4.3 and above support nested arrays (arrays of arrays).  Set this to ``True``
        to render matrices as nested arrays.
        [default=False]
    mat_separator: str, optional
        By default, matrices are rendered with newlines using this separator,
        making them easier to read, but less compact.  By removing the newline
        this option can be used to make them more vertically compact.
        [default=',\n']
    mat_transpose: bool, optional
        GLSL's matrix multiplication implementation assumes column-major indexing.
        By default, this printer ignores that convention. Setting this option to
        ``True`` transposes all matrix output.
        [default=False]
    array_type: str, optional
        The GLSL array constructor type.
        [default='float']
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
    """
    # 定义函数参数 `contract`，可选，默认为 True。当为 True 时，假定 `Indexed` 实例遵循张量收缩规则，并生成相应的嵌套索引循环。设置为 False 时，不生成循环，用户需在代码中提供索引的值。
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    
    # 示例代码部分
    
    Examples
    ========
    
    # 引入 sympy 库中的 glsl_code 函数以及其他必要的符号和函数
    >>> from sympy import glsl_code, symbols, Rational, sin, ceiling, Abs
    >>> x, tau = symbols("x, tau")
    
    # 调用 glsl_code 函数计算并输出表达式 (2*tau)**Rational(7, 2) 的 GLSL 代码表示
    >>> glsl_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*pow(tau, 3.5)'
    
    # 调用 glsl_code 函数计算并输出 sin(x) 的 GLSL 代码表示，并赋值给变量 "float y"
    >>> glsl_code(sin(x), assign_to="float y")
    'float y = sin(x);'
    
    # 支持多种 GLSL 类型的转换，例如将向量转换为 GLSL 表示
    >>> from sympy import Matrix, glsl_code
    >>> glsl_code(Matrix([1,2,3]))
    'vec3(1, 2, 3)'
    
    # 将矩阵转换为 GLSL 表示
    >>> glsl_code(Matrix([[1, 2],[3, 4]]))
    'mat2(1, 2, 3, 4)'
    
    # 设置 mat_transpose=True 可以切换到列主索引，输出 GLSL 表示
    >>> glsl_code(Matrix([[1, 2],[3, 4]]), mat_transpose=True)
    'mat2(1, 3, 2, 4)'
    
    # 默认情况下，较大的矩阵会折叠成 float 数组的形式输出
    >>> print(glsl_code( Matrix([[1,2,3,4,5],[6,7,8,9,10]]) ))
    float[10](
       1, 2, 3, 4,  5,
       6, 7, 8, 9, 10
    ) /* a 2x5 matrix */
    
    # 可以通过 array_type 参数控制 GLSL 数组的构造类型
    >>> glsl_code(Matrix([1,2,3,4,5]), array_type='int')
    'int[5](1, 2, 3, 4, 5)'
    
    # 将字符串列表或符号列表传递给 assign_to 参数，可以生成多行赋值语句
    >>> x_struct_members = symbols('x.a x.b x.c x.d')
    >>> print(glsl_code(Matrix([1,2,3,4]), assign_to=x_struct_members))
    x.a = 1;
    x.b = 2;
    x.c = 3;
    x.d = 4;
    
    # 通过设置 mat_nested=True 可以输出嵌套的浮点数组表示，适用于 GLSL 4.3 及以上版本
    >>> mat = Matrix([
    ... [ 0,  1,  2],
    ... [ 3,  4,  5],
    ... [ 6,  7,  8],
    ... [ 9, 10, 11],
    ... [12, 13, 14]])
    >>> print(glsl_code( mat, mat_nested=True ))
    float[5][3](
       float[]( 0,  1,  2),
       float[]( 3,  4,  5),
       float[]( 6,  7,  8),
       float[]( 9, 10, 11),
       float[](12, 13, 14)
    )
    
    # 可以通过 user_functions 参数为特定类型定义自定义打印方式，以字典或元组列表的形式传递
    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    # 返回转换为 GLSL（OpenGL Shading Language）代码的表达式字符串
    return GLSLPrinter(settings).doprint(expr,assign_to)
# 定义一个函数 print_glsl，用于打印给定表达式的 GLSL 表示。
# **settings 是一个关键字参数，用于传递给 glsl_code 函数的额外设置。
def print_glsl(expr, **settings):
    """Prints the GLSL representation of the given expression.

       See GLSLPrinter init function for settings.
    """
    # 调用 glsl_code 函数生成给定表达式的 GLSL 代码，并将其打印出来。
    print(glsl_code(expr, **settings))
```