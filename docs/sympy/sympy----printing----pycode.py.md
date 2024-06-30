# `D:\src\scipysrc\sympy\sympy\printing\pycode.py`

```
"""
Python code printers

This module contains Python code printers for plain Python as well as NumPy & SciPy enabled code.
"""
from collections import defaultdict  # 导入 defaultdict 类
from itertools import chain  # 导入 chain 函数
from sympy.core import S  # 导入 S 对象
from sympy.core.mod import Mod  # 导入 Mod 对象
from .precedence import precedence  # 从当前包中导入 precedence 模块
from .codeprinter import CodePrinter  # 从当前包中导入 CodePrinter 类

_kw = {
    'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif',
    'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
    'is', 'lambda', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
    'with', 'yield', 'None', 'False', 'nonlocal', 'True'
}  # Python 中的保留关键字集合

_known_functions = {
    'Abs': 'abs',
    'Min': 'min',
    'Max': 'max',
}  # 已知函数及其在 Python 中的对应函数名映射

_known_functions_math = {
    'acos': 'acos',
    'acosh': 'acosh',
    'asin': 'asin',
    'asinh': 'asinh',
    'atan': 'atan',
    'atan2': 'atan2',
    'atanh': 'atanh',
    'ceiling': 'ceil',
    'cos': 'cos',
    'cosh': 'cosh',
    'erf': 'erf',
    'erfc': 'erfc',
    'exp': 'exp',
    'expm1': 'expm1',
    'factorial': 'factorial',
    'floor': 'floor',
    'gamma': 'gamma',
    'hypot': 'hypot',
    'isnan': 'isnan',
    'loggamma': 'lgamma',
    'log': 'log',
    'ln': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'sin': 'sin',
    'sinh': 'sinh',
    'Sqrt': 'sqrt',
    'tan': 'tan',
    'tanh': 'tanh'
}  # 数学函数及其在 Python math 模块中的对应函数名映射

_known_constants_math = {
    'Exp1': 'e',
    'Pi': 'pi',
    'E': 'e',
    'Infinity': 'inf',
    'NaN': 'nan',
    'ComplexInfinity': 'nan'
}  # 数学常数及其在 Python math 模块中的对应名称映射

def _print_known_func(self, expr):
    """
    根据表达式对象的类名查找已知函数并生成其对应的 Python 代码表示形式
    """
    known = self.known_functions[expr.__class__.__name__]
    return '{name}({args})'.format(name=self._module_format(known),
                                   args=', '.join((self._print(arg) for arg in expr.args)))

def _print_known_const(self, expr):
    """
    根据表达式对象查找已知常数并生成其对应的 Python 代码表示形式
    """
    known = self.known_constants[expr.__class__.__name__]
    return self._module_format(known)

class AbstractPythonCodePrinter(CodePrinter):
    """
    Python 代码打印器的抽象基类，继承自 CodePrinter 类
    """
    printmethod = "_pythoncode"
    language = "Python"
    reserved_words = _kw
    modules = None  # 在 __init__ 方法中初始化为集合
    tab = '    '  # 缩进字符串为四个空格
    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'math.' + v) for k, v in _known_functions_math.items()]
    ))  # 合并已知函数映射及数学函数映射到 _kf 字典中
    _kc = {k: 'math.'+v for k, v in _known_constants_math.items()}  # 将数学常数映射到 _kc 字典中
    _operators = {'and': 'and', 'or': 'or', 'not': 'not'}  # 逻辑运算符映射
    _default_settings = dict(
        CodePrinter._default_settings,
        user_functions={},
        precision=17,
        inline=True,
        fully_qualified_modules=True,
        contract=False,
        standard='python3',
    )  # 默认设置字典，包含代码打印器的各种设置选项
    def __init__(self, settings=None):
        super().__init__(settings)

        # Python standard handler
        std = self._settings['standard']
        # 如果没有指定标准版本，则默认使用当前 Python 版本
        if std is None:
            import sys
            std = 'python{}'.format(sys.version_info.major)
        # 如果标准版本不是 Python 3，则抛出数值错误
        if std != 'python3':
            raise ValueError('Only Python 3 is supported.')
        self.standard = std

        self.module_imports = defaultdict(set)

        # Known functions and constants handler
        # 初始化已知函数和常量处理器，包含内置和用户提供的函数和常量
        self.known_functions = dict(self._kf, **(settings or {}).get(
            'user_functions', {}))
        self.known_constants = dict(self._kc, **(settings or {}).get(
            'user_constants', {}))

    def _declare_number_const(self, name, value):
        # 声明数值常量
        return "%s = %s" % (name, value)

    def _module_format(self, fqn, register=True):
        # 格式化模块名称，如果需要注册则添加到模块导入列表中
        parts = fqn.split('.')
        if register and len(parts) > 1:
            self.module_imports['.'.join(parts[:-1])].add(parts[-1])

        # 如果设置要求完全限定模块名称，则返回完全限定的模块名称
        if self._settings['fully_qualified_modules']:
            return fqn
        else:
            # 否则返回最后一部分的模块名称
            return fqn.split('(')[0].split('[')[0].split('.')[-1]

    def _format_code(self, lines):
        # 格式化代码行
        return lines

    def _get_statement(self, codestring):
        # 获取语句字符串
        return "{}".format(codestring)

    def _get_comment(self, text):
        # 获取注释文本
        return "  # {}".format(text)

    def _expand_fold_binary_op(self, op, args):
        """
        This method expands a fold on binary operations.

        ``functools.reduce`` is an example of a folded operation.

        For example, the expression

        `A + B + C + D`

        is folded into

        `((A + B) + C) + D`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            # 展开二进制操作的折叠
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_fold_binary_op(op, args[:-1]),
                self._print(args[-1]),
            )

    def _expand_reduce_binary_op(self, op, args):
        """
        This method expands a reduction on binary operations.

        Notice: this is NOT the same as ``functools.reduce``.

        For example, the expression

        `A + B + C + D`

        is reduced into:

        `(A + B) + (C + D)`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            N = len(args)
            Nhalf = N // 2
            # 展开二进制操作的缩减
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_reduce_binary_op(op, args[:Nhalf]),
                self._expand_reduce_binary_op(op, args[Nhalf:]),
            )

    def _print_NaN(self, expr):
        # 打印 NaN（Not a Number）
        return "float('nan')"

    def _print_Infinity(self, expr):
        # 打印正无穷
        return "float('inf')"

    def _print_NegativeInfinity(self, expr):
        # 打印负无穷
        return "float('-inf')"

    def _print_ComplexInfinity(self, expr):
        # 打印复杂无穷，实际上是打印 NaN
        return self._print_NaN(expr)
    # 定义一个函数 _print_Mod，用于打印取模表达式的字符串表示
    def _print_Mod(self, expr):
        # 获取表达式的优先级
        PREC = precedence(expr)
        # 返回格式化后的取模表达式字符串
        return ('{} % {}'.format(*(self.parenthesize(x, PREC) for x in expr.args)))

    # 定义一个函数 _print_Piecewise，用于打印分段函数的字符串表示
    def _print_Piecewise(self, expr):
        # 初始化结果列表和计数器
        result = []
        i = 0
        # 遍历分段函数的每个分段
        for arg in expr.args:
            e = arg.expr  # 获取分段的表达式部分
            c = arg.cond  # 获取分段的条件部分
            # 第一个分段的处理
            if i == 0:
                result.append('(')
            result.append('(')
            result.append(self._print(e))  # 打印分段的表达式
            result.append(')')
            result.append(' if ')
            result.append(self._print(c))  # 打印分段的条件
            result.append(' else ')
            i += 1
        result = result[:-1]  # 去除最后一个多余的空格
        # 处理最后一个分段
        if result[-1] == 'True':
            result = result[:-2]
            result.append(')')
        else:
            result.append(' else None)')
        return ''.join(result)

    # 定义一个函数 _print_Relational，用于打印关系运算表达式的字符串表示
    def _print_Relational(self, expr):
        # 定义关系运算符的字符串映射
        op = {
            '==': 'equal',
            '!=': 'not_equal',
            '<': 'less',
            '<=': 'less_equal',
            '>': 'greater',
            '>=': 'greater_equal',
        }
        # 如果表达式的关系运算符在映射中，则格式化打印
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)  # 打印左操作数
            rhs = self._print(expr.rhs)  # 打印右操作数
            return '({lhs} {op} {rhs})'.format(op=expr.rel_op, lhs=lhs, rhs=rhs)
        # 否则调用父类的默认打印方法
        return super()._print_Relational(expr)

    # 定义一个函数 _print_ITE，用于打印条件表达式的字符串表示
    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        # 将条件表达式转换为分段函数后再打印
        return self._print(expr.rewrite(Piecewise))

    # 定义一个函数 _print_Sum，用于打印求和表达式的字符串表示
    def _print_Sum(self, expr):
        # 构建循环语句字符串
        loops = (
            'for {i} in range({a}, {b}+1)'.format(
                i=self._print(i),
                a=self._print(a),
                b=self._print(b))
            for i, a, b in expr.limits)
        # 返回格式化后的求和表达式字符串
        return '(builtins.sum({function} {loops}))'.format(
            function=self._print(expr.function),
            loops=' '.join(loops))

    # 定义一个函数 _print_ImaginaryUnit，用于打印虚数单位的字符串表示
    def _print_ImaginaryUnit(self, expr):
        # 直接返回虚数单位的字符串表示
        return '1j'

    # 定义一个函数 _print_KroneckerDelta，用于打印克罗内克 δ 函数的字符串表示
    def _print_KroneckerDelta(self, expr):
        a, b = expr.args
        # 返回克罗内克 δ 函数的格式化字符串表示
        return '(1 if {a} == {b} else 0)'.format(
            a=self._print(a),
            b=self._print(b)
        )

    # 定义一个函数 _print_MatrixBase，用于打印矩阵基类的字符串表示
    def _print_MatrixBase(self, expr):
        name = expr.__class__.__name__
        func = self.known_functions.get(name, name)
        # 返回格式化后的矩阵基类的字符串表示
        return "%s(%s)" % (func, self._print(expr.tolist()))

    # 将多个矩阵类型的打印方法统一为调用 _print_MatrixBase 方法
    _print_SparseRepMatrix = \
        _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        lambda self, expr: self._print_MatrixBase(expr)

    # 定义一个函数 _indent_codestring，用于对代码字符串进行缩进处理
    def _indent_codestring(self, codestring):
        # 将输入的代码字符串按行分割并在每行前面加上缩进符号
        return '\n'.join([self.tab + line for line in codestring.split('\n')])
    # 定义打印函数定义的方法，将函数定义对象转换为字符串表示
    def _print_FunctionDefinition(self, fd):
        # 生成函数体部分的字符串表示，包括每个参数
        body = '\n'.join((self._print(arg) for arg in fd.body))
        # 返回格式化后的函数定义字符串
        return "def {name}({parameters}):\n{body}".format(
            name=self._print(fd.name),  # 函数名
            parameters=', '.join([self._print(var.symbol) for var in fd.parameters]),  # 函数参数
            body=self._indent_codestring(body)  # 缩进后的函数体
        )
    
    # 定义打印 while 循环的方法，将循环对象转换为字符串表示
    def _print_While(self, whl):
        # 生成循环体部分的字符串表示
        body = '\n'.join((self._print(arg) for arg in whl.body))
        # 返回格式化后的 while 循环字符串
        return "while {cond}:\n{body}".format(
            cond=self._print(whl.condition),  # 循环条件
            body=self._indent_codestring(body)  # 缩进后的循环体
        )
    
    # 定义打印声明语句的方法，将声明对象转换为字符串表示
    def _print_Declaration(self, decl):
        # 返回格式化后的声明语句字符串
        return '%s = %s' % (
            self._print(decl.variable.symbol),  # 变量名
            self._print(decl.variable.value)  # 变量值
        )
    
    # 定义打印 break 语句的方法，将 break 对象转换为字符串表示
    def _print_BreakToken(self, bt):
        # 返回字符串表示的 break
        return 'break'
    
    # 定义打印 return 语句的方法，将 return 对象转换为字符串表示
    def _print_Return(self, ret):
        arg, = ret.args
        # 返回格式化后的 return 语句字符串
        return 'return %s' % self._print(arg)
    
    # 定义打印 raise 语句的方法，将 raise 对象转换为字符串表示
    def _print_Raise(self, rs):
        arg, = rs.args
        # 返回格式化后的 raise 语句字符串
        return 'raise %s' % self._print(arg)
    
    # 定义打印 RuntimeError 异常的方法，将异常对象转换为字符串表示
    def _print_RuntimeError_(self, re):
        message, = re.args
        # 返回格式化后的 RuntimeError 异常字符串
        return "RuntimeError(%s)" % self._print(message)
    
    # 定义打印 print 语句的方法，将 print 对象转换为字符串表示
    def _print_Print(self, prnt):
        # 生成打印参数部分的字符串表示
        print_args = ', '.join((self._print(arg) for arg in prnt.print_args))
        # 导入 none 对象
        from sympy.codegen.ast import none
        # 如果格式字符串不是 none，则添加格式化字符串
        if prnt.format_string != none:
            print_args = '{} % ({}), end=""'.format(
                self._print(prnt.format_string),  # 格式化字符串
                print_args  # 打印参数
            )
        # 如果文件不是 None，则添加文件参数
        if prnt.file != None:  # 必须是 '!= None'，不能是 'is not None'
            print_args += ', file=%s' % self._print(prnt.file)  # 打印文件
        # 返回格式化后的 print 语句字符串
        return 'print(%s)' % print_args
    
    # 定义打印 Stream 对象的方法，将流对象转换为字符串表示
    def _print_Stream(self, strm):
        # 如果流的名称是 stdout，则返回 sys.stdout 的字符串表示
        if str(strm.name) == 'stdout':
            return self._module_format('sys.stdout')
        # 如果流的名称是 stderr，则返回 sys.stderr 的字符串表示
        elif str(strm.name) == 'stderr':
            return self._module_format('sys.stderr')
        else:
            # 否则返回流名称的字符串表示
            return self._print(strm.name)
    
    # 定义打印 NoneToken 对象的方法，将 NoneToken 转换为字符串表示
    def _print_NoneToken(self, arg):
        # 返回字符串表示的 None
        return 'None'
    # 打印辅助函数，用于处理 ``Pow`` 操作

    # 根据表达式的优先级获取 PREC 值
    PREC = precedence(expr)

    # 如果指数为 S.Half 并且 rational 参数为 False
    if expr.exp == S.Half and not rational:
        # 使用指定的 sqrt 函数格式化为模块格式
        func = self._module_format(sqrt)
        # 打印基数表达式
        arg = self._print(expr.base)
        return '{func}({arg})'.format(func=func, arg=arg)

    # 如果表达式是可交换的并且 rational 参数为 False
    if expr.is_commutative and not rational:
        # 如果指数为负的 S.Half
        if -expr.exp is S.Half:
            # 使用指定的 sqrt 函数格式化为模块格式
            func = self._module_format(sqrt)
            # 打印数值 '1'
            num = self._print(S.One)
            # 打印基数表达式
            arg = self._print(expr.base)
            return f"{num}/{func}({arg})"
        # 如果指数为负的 S.NegativeOne
        if expr.exp is S.NegativeOne:
            # 打印数值 '1'
            num = self._print(S.One)
            # 打印基数表达式
            arg = self.parenthesize(expr.base, PREC, strict=False)
            return f"{num}/{arg}"

    # 打印基数表达式
    base_str = self.parenthesize(expr.base, PREC, strict=False)
    # 打印指数表达式
    exp_str = self.parenthesize(expr.exp, PREC, strict=False)
    return "{}**{}".format(base_str, exp_str)
class ArrayPrinter:

    def _arrayify(self, indexed):
        # 导入将索引转换为数组的函数
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        try:
            # 尝试将索引转换为数组表示
            return convert_indexed_to_array(indexed)
        except Exception:
            # 转换失败则返回原始索引
            return indexed

    def _get_einsum_string(self, subranks, contraction_indices):
        # 获取用于 Einstein 求和符号的字母生成器
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ""
        counter = 0
        # 构建用于收缩的索引字典
        d = {j: min(i) for i in contraction_indices for j in i}
        indices = []
        # 遍历每个子秩并生成对应的索引列表
        for rank_arg in subranks:
            lindices = []
            for i in range(rank_arg):
                if counter in d:
                    lindices.append(d[counter])
                else:
                    lindices.append(counter)
                counter += 1
            indices.append(lindices)
        mapping = {}
        letters_free = []
        letters_dum = []
        # 为每个索引生成对应的字母映射
        for i in indices:
            for j in i:
                if j not in mapping:
                    l = next(letters)
                    mapping[j] = l
                else:
                    l = mapping[j]
                contraction_string += l
                if j in d:
                    if l not in letters_dum:
                        letters_dum.append(l)
                else:
                    letters_free.append(l)
            contraction_string += ","
        contraction_string = contraction_string[:-1]  # 去除最后一个逗号
        return contraction_string, letters_free, letters_dum

    def _get_letter_generator_for_einsum(self):
        # 生成用于 Einstein 求和符号的字母生成器
        for i in range(97, 123):  # 小写字母 a-z
            yield chr(i)
        for i in range(65, 91):   # 大写字母 A-Z
            yield chr(i)
        # 如果超出了字母范围，引发 ValueError 异常
        raise ValueError("out of letters")

    def _print_ArrayTensorProduct(self, expr):
        # 获取用于 Einstein 求和符号的字母生成器
        letters = self._get_letter_generator_for_einsum()
        # 构建表示收缩的字符串
        contraction_string = ",".join(["".join([next(letters) for j in range(i)]) for i in expr.subranks])
        # 返回格式化后的字符串表示
        return '%s("%s", %s)' % (
                self._module_format(self._module + "." + self._einsum),
                contraction_string,
                ", ".join([self._print(arg) for arg in expr.args])
        )
    def _print_ArrayContraction(self, expr):
        # 导入所需模块
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        # 获取表达式的基础部分和收缩指标
        base = expr.expr
        contraction_indices = expr.contraction_indices

        # 如果基础部分是 ArrayTensorProduct 类型
        if isinstance(base, ArrayTensorProduct):
            # 将所有参数打印成字符串并用逗号分隔
            elems = ",".join(["%s" % (self._print(arg)) for arg in base.args])
            # 获取子秩
            ranks = base.subranks
        else:
            # 否则，将基础部分打印成字符串
            elems = self._print(base)
            # 获取基础部分的秩
            ranks = [len(base.shape)]

        # 获取收缩字符串、自由指标、虚拟指标的字符串表示
        contraction_string, letters_free, letters_dum = self._get_einsum_string(ranks, contraction_indices)

        # 如果没有收缩指标，直接返回基础部分的打印结果
        if not contraction_indices:
            return self._print(base)

        # 如果基础部分是 ArrayTensorProduct 类型，重新计算 elems
        if isinstance(base, ArrayTensorProduct):
            elems = ",".join(["%s" % (self._print(arg)) for arg in base.args])
        else:
            elems = self._print(base)

        # 返回格式化的字符串表示
        return "%s(\"%s\", %s)" % (
            self._module_format(self._module + "." + self._einsum),
            "{}->{}".format(contraction_string, "".join(sorted(letters_free))),
            elems,
        )

    def _print_ArrayDiagonal(self, expr):
        # 导入所需模块
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        # 获取对角线指标
        diagonal_indices = list(expr.diagonal_indices)

        # 如果表达式的基础部分是 ArrayTensorProduct 类型
        if isinstance(expr.expr, ArrayTensorProduct):
            subranks = expr.expr.subranks
            elems = expr.expr.args
        else:
            subranks = expr.subranks
            elems = [expr.expr]

        # 获取对角线字符串、自由指标、虚拟指标的字符串表示
        diagonal_string, letters_free, letters_dum = self._get_einsum_string(subranks, diagonal_indices)

        # 将 elems 中的每个元素打印成字符串
        elems = [self._print(i) for i in elems]

        # 返回格式化的字符串表示
        return '%s("%s", %s)' % (
            self._module_format(self._module + "." + self._einsum),
            "{}->{}".format(diagonal_string, "".join(letters_free + letters_dum)),
            ", ".join(elems)
        )

    def _print_PermuteDims(self, expr):
        # 返回格式化的字符串表示，包括模块、表达式、排列形式
        return "%s(%s, %s)" % (
            self._module_format(self._module + "." + self._transpose),
            self._print(expr.expr),
            self._print(expr.permutation.array_form),
        )

    def _print_ArrayAdd(self, expr):
        # 调用二元操作的扩展和折叠方法，并返回结果
        return self._expand_fold_binary_op(self._module + "." + self._add, expr.args)

    def _print_OneArray(self, expr):
        # 返回格式化的字符串表示，包括模块、包含元组的一阵
        return "%s((%s,))" % (
            self._module_format(self._module + "." + self._ones),
            ','.join(map(self._print, expr.args))
        )

    def _print_ZeroArray(self, expr):
        # 返回格式化的字符串表示，包括模块、包含元组的零阵
        return "%s((%s,))" % (
            self._module_format(self._module + "." + self._zeros),
            ','.join(map(self._print, expr.args))
        )

    def _print_Assignment(self, expr):
        # 返回赋值表达式的格式化字符串表示，包括左值和右值
        lhs = self._print(self._arrayify(expr.lhs))
        rhs = self._print(self._arrayify(expr.rhs))
        return "%s = %s" % (lhs, rhs)

    def _print_IndexedBase(self, expr):
        # 调用打印 ArraySymbol 的方法并返回结果
        return self._print_ArraySymbol(expr)
class PythonCodePrinter(AbstractPythonCodePrinter):
    # PythonCodePrinter 类，继承自 AbstractPythonCodePrinter

    def _print_sign(self, e):
        # 打印 sign 函数的表达式
        return '(0.0 if {e} == 0 else {f}(1, {e}))'.format(
            f=self._module_format('math.copysign'), e=self._print(e.args[0]))
        # 返回一个字符串，根据 e 参数的值打印不同的数学函数表达式

    def _print_Not(self, expr):
        # 打印逻辑非操作符（not）
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)
        # 返回包含逻辑非操作符的字符串，优先级由 expr 的优先级决定

    def _print_IndexedBase(self, expr):
        # 打印 IndexedBase 对象
        return expr.name
        # 返回 IndexedBase 对象的名称

    def _print_Indexed(self, expr):
        # 打印 Indexed 对象
        base = expr.args[0]
        index = expr.args[1:]
        return "{}[{}]".format(str(base), ", ".join([self._print(ind) for ind in index]))
        # 返回包含索引表达式的字符串，格式为 base[index1, index2, ...]

    def _print_Pow(self, expr, rational=False):
        # 打印幂操作符（指数）
        return self._hprint_Pow(expr, rational=rational)
        # 返回幂操作符的打印字符串

    def _print_Rational(self, expr):
        # 打印有理数表达式
        return '{}/{}'.format(expr.p, expr.q)
        # 返回有理数的字符串表示，格式为 p/q

    def _print_Half(self, expr):
        # 打印 Half 对象
        return self._print_Rational(expr)
        # 返回 Half 对象的有理数表示

    def _print_frac(self, expr):
        # 打印 frac 函数的表达式
        return self._print_Mod(Mod(expr.args[0], 1))
        # 返回 Mod 函数对应的表达式字符串

    def _print_Symbol(self, expr):
        # 打印符号表达式
        name = super()._print_Symbol(expr)

        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = ('This expression includes the symbol "{}" which is a '
                       'reserved keyword in this language.')
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        elif '{' in name:   # Remove curly braces from subscripted variables
            return name.replace('{', '').replace('}', '')
        else:
            return name
        # 处理符号表达式的打印，检查是否为保留字，处理下标变量的花括号

    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported
    # 定义不支持打印的特定函数，如 lowergamma, uppergamma, fresnelc, fresnels


for k in PythonCodePrinter._kf:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_func)
    # 动态设置 PythonCodePrinter 类的打印函数，使用 _print_known_func 函数

for k in _known_constants_math:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_const)
    # 动态设置 PythonCodePrinter 类的打印常数，使用 _print_known_const 函数


def pycode(expr, **settings):
    """ Converts an expr to a string of Python code

    Parameters
    ==========

    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    standard : str or None, optional
        Only 'python3' (default) is supported.
        This parameter may be removed in the future.

    Examples
    ========

    >>> from sympy import pycode, tan, Symbol
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'

    """
    return PythonCodePrinter(settings).doprint(expr)
    # 将 SymPy 表达式转换为 Python 代码字符串的函数


_not_in_mpmath = 'log1p log2'.split()
_in_mpmath = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_mpmath]
_known_functions_mpmath = dict(_in_mpmath, **{
    'beta': 'beta',
    'frac': 'frac',
    'fresnelc': 'fresnelc',
    'fresnels': 'fresnels',
    'sign': 'sign',
    'loggamma': 'loggamma',
    'hyper': 'hyper',
})
# 定义 _known_functions_mpmath 字典，包含与数学函数库 mpmath 相关的已知函数名称和对应的字符串
    'meijerg': 'meijerg',   # 字典条目：键为 'meijerg'，值为 'meijerg'
    'besselj': 'besselj',   # 字典条目：键为 'besselj'，值为 'besselj'
    'bessely': 'bessely',   # 字典条目：键为 'bessely'，值为 'bessely'
    'besseli': 'besseli',   # 字典条目：键为 'besseli'，值为 'besseli'
    'besselk': 'besselk',   # 字典条目：键为 'besselk'，值为 'besselk'
# 定义一个字典，映射 SymPy 中定义的数学常数到 mpmath 中对应的名称
_known_constants_mpmath = {
    'Exp1': 'e',                    # 自然对数的底 e
    'Pi': 'pi',                     # 圆周率 π
    'GoldenRatio': 'phi',           # 黄金比例 φ
    'EulerGamma': 'euler',          # 欧拉常数 γ
    'Catalan': 'catalan',           # Catalan's constant
    'NaN': 'nan',                   # 非数值（Not a Number）
    'Infinity': 'inf',              # 正无穷
    'NegativeInfinity': 'ninf'      # 负无穷
}

def _unpack_integral_limits(integral_expr):
    """ helper function for _print_Integral that
        - accepts an Integral expression
        - returns a tuple of
           - a list variables of integration
           - a list of tuples of the upper and lower limits of integration
    """
    integration_vars = []   # 存储积分变量的列表
    limits = []             # 存储积分上下限的元组列表
    for integration_range in integral_expr.limits:
        if len(integration_range) == 3:
            integration_var, lower_limit, upper_limit = integration_range  # 解包积分表达式的上下限
        else:
            raise NotImplementedError("Only definite integrals are supported")  # 报告仅支持定积分
        integration_vars.append(integration_var)     # 将积分变量添加到列表中
        limits.append((lower_limit, upper_limit))    # 将上下限元组添加到列表中
    return integration_vars, limits

class MpmathPrinter(PythonCodePrinter):
    """
    Lambda printer for mpmath which maintains precision for floats
    """
    printmethod = "_mpmathcode"   # 打印方法的名称

    language = "Python with mpmath"  # 使用 mpmath 的 Python 语言

    # _kf 字典合并了 _known_functions 和 _known_functions_mpmath，映射到 mpmath 函数
    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'mpmath.' + v) for k, v in _known_functions_mpmath.items()]
    ))
    
    # _kc 字典，将 _known_constants_mpmath 的常数映射到 mpmath 常数
    _kc = {k: 'mpmath.'+v for k, v in _known_constants_mpmath.items()}

    def _print_Float(self, e):
        # XXX: This does not handle setting mpmath.mp.dps. It is assumed that
        # the caller of the lambdified function will have set it to sufficient
        # precision to match the Floats in the expression.

        # Remove 'mpz' if gmpy is installed.
        args = str(tuple(map(int, e._mpf_)))  # 将浮点数转换为整数元组
        return '{func}({args})'.format(func=self._module_format('mpmath.mpf'), args=args)

    def _print_Rational(self, e):
        # 打印有理数表达式
        return "{func}({p})/{func}({q})".format(
            func=self._module_format('mpmath.mpf'),
            q=self._print(e.q),
            p=self._print(e.p)
        )

    def _print_Half(self, e):
        # 打印分数表达式
        return self._print_Rational(e)

    def _print_uppergamma(self, e):
        # 打印上不完全伽马函数表达式
        return "{}({}, {}, {})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]),
            self._module_format('mpmath.inf'))

    def _print_lowergamma(self, e):
        # 打印下不完全伽马函数表达式
        return "{}({}, 0, {})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]))

    def _print_log2(self, e):
        # 打印以 2 为底的对数表达式
        return '{0}({1})/{0}(2)'.format(
            self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_log1p(self, e):
        # 打印 log(1+x) 函数表达式
        return '{}({})'.format(
            self._module_format('mpmath.log1p'), self._print(e.args[0]))

    def _print_Pow(self, expr, rational=False):
        # 打印幂函数表达式
        return self._hprint_Pow(expr, rational=rational, sqrt='mpmath.sqrt')
    # 定义一个私有方法 `_print_Integral`，用于打印积分表达式
    def _print_Integral(self, e):
        # 解析积分变量和积分上下限
        integration_vars, limits = _unpack_integral_limits(e)
        
        # 格式化返回一个字符串，表示积分表达式
        return "{}(lambda {}: {}, {})".format(
                self._module_format("mpmath.quad"),  # 使用 mpmath.quad 模块进行积分计算
                ", ".join(map(self._print, integration_vars)),  # 将积分变量转换为打印字符串
                self._print(e.args[0]),  # 打印积分被积函数
                ", ".join("(%s, %s)" % tuple(map(self._print, l)) for l in limits))  # 打印积分上下限
# 遍历 MpmathPrinter 类的 _kf 属性中的每个键值
for k in MpmathPrinter._kf:
    # 动态设置 MpmathPrinter 类的方法名，格式为 _print_<k>，并将其指向 _print_known_func 函数
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_func)

# 遍历 _known_constants_mpmath 列表中的每个元素
for k in _known_constants_mpmath:
    # 动态设置 MpmathPrinter 类的方法名，格式为 _print_<k>，并将其指向 _print_known_const 函数
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_const)


# 定义 SymPyPrinter 类，继承自 AbstractPythonCodePrinter 类
class SymPyPrinter(AbstractPythonCodePrinter):

    # 设定语言属性为 "Python with SymPy"
    language = "Python with SymPy"

    # 定义默认设置字典，包括 AbstractPythonCodePrinter 类的默认设置，并添加 strict=False 设置
    _default_settings = dict(
        AbstractPythonCodePrinter._default_settings,
        strict=False   # 任何类名都被定义为我们在 SymPyPrinter 中的目标。
    )

    # 定义 _print_Function 方法，用于打印 Function 类型的表达式
    def _print_Function(self, expr):
        # 获取函数的模块名，如果不存在则为空字符串
        mod = expr.func.__module__ or ''
        # 返回格式化后的函数调用字符串，包括模块名和函数名，以及参数列表
        return '%s(%s)' % (self._module_format(mod + ('.' if mod else '') + expr.func.__name__),
                           ', '.join((self._print(arg) for arg in expr.args)))

    # 定义 _print_Pow 方法，用于打印 Pow 类型的表达式
    def _print_Pow(self, expr, rational=False):
        # 调用 _hprint_Pow 方法，返回 Pow 类型表达式的字符串表示
        return self._hprint_Pow(expr, rational=rational, sqrt='sympy.sqrt')
```