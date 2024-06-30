# `D:\src\scipysrc\sympy\sympy\printing\fortran.py`

```
"""
Fortran code printer

The FCodePrinter converts single SymPy expressions into single Fortran
expressions, using the functions defined in the Fortran 77 standard where
possible. Some useful pointers to Fortran can be found on wikipedia:

https://en.wikipedia.org/wiki/Fortran

Most of the code below is based on the "Professional Programmer\'s Guide to
Fortran77" by Clive G. Page:

https://www.star.le.ac.uk/~cgp/prof77.html

Fortran is a case-insensitive language. This might cause trouble because
SymPy is case sensitive. So, fcode adds underscores to variable names when
it is necessary to make them different for Fortran.
"""

from __future__ import annotations
from typing import Any

from collections import defaultdict
from itertools import chain
import string

from sympy.codegen.ast import (
    Assignment, Declaration, Pointer, value_const,
    float32, float64, float80, complex64, complex128, int8, int16, int32,
    int64, intc, real, integer,  bool_, complex_, none, stderr, stdout
)
from sympy.codegen.fnodes import (
    allocatable, isign, dsign, cmplx, merge, literal_dp, elemental, pure,
    intent_in, intent_out, intent_inout
)
from sympy.core import S, Add, N, Float, Symbol
from sympy.core.function import Function
from sympy.core.numbers import equal_valued
from sympy.core.relational import Eq
from sympy.sets import Range
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.printing.printer import printer_context

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import fcode, print_fcode # noqa:F401

known_functions = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "log": "log",
    "exp": "exp",
    "erf": "erf",
    "Abs": "abs",
    "conjugate": "conjg",
    "Max": "max",
    "Min": "min",
}

# Define a subclass of CodePrinter tailored for Fortran code generation
class FCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Fortran code"""
    printmethod = "_fcode"  # Method used for printing Fortran code
    language = "Fortran"  # Language identifier for the printer

    # Type aliases mapping SymPy types to Fortran types
    type_aliases = {
        integer: int32,
        real: float64,
        complex_: complex128,
    }

    # Type mappings specifying Fortran type representations
    type_mappings = {
        intc: 'integer(c_int)',
        float32: 'real*4',  # real(kind(0.e0))
        float64: 'real*8',  # real(kind(0.d0))
        float80: 'real*10', # real(kind(????))
        complex64: 'complex*8',
        complex128: 'complex*16',
        int8: 'integer*1',
        int16: 'integer*2',
        int32: 'integer*4',
        int64: 'integer*8',
        bool_: 'logical'
    }

    # Type modules specifying additional modules for specific types
    type_modules = {
        intc: {'iso_c_binding': 'c_int'}
    }
    # 默认设置字典，包含代码打印器的默认设置，并添加额外设置
    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'source_format': 'fixed',
        'contract': True,
        'standard': 77,
        'name_mangling': True,
    })

    # 运算符映射表，将逻辑运算符映射到Fortran中的等效字符串
    _operators = {
        'and': '.and.',
        'or': '.or.',
        'xor': '.neqv.',
        'equivalent': '.eqv.',
        'not': '.not. ',
    }

    # 关系运算符映射表，将'!='映射为'/='
    _relationals = {
        '!=': '/=',
    }

    # 类初始化方法，设置对象的初始状态
    def __init__(self, settings=None):
        if not settings:
            settings = {}
        # 显示已经混淆的符号的字典，显示所有的单词映射
        self.mangled_symbols = {}
        # 已使用的名称列表
        self.used_name = []
        # 类型别名字典，包含默认别名和用户提供的别名
        self.type_aliases = dict(chain(self.type_aliases.items(),
                                       settings.pop('type_aliases', {}).items()))
        # 类型映射字典，包含默认映射和用户提供的映射
        self.type_mappings = dict(chain(self.type_mappings.items(),
                                        settings.pop('type_mappings', {}).items()))
        # 调用父类的初始化方法，传入设置参数
        super().__init__(settings)
        # 已知函数的字典，包含默认已知函数和用户提供的函数
        self.known_functions = dict(known_functions)
        # 获取用户自定义的函数，并更新到已知函数字典中
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        # Fortran标准集合
        standards = {66, 77, 90, 95, 2003, 2008}
        # 检查设置中的Fortran标准是否在已知标准集合中
        if self._settings['standard'] not in standards:
            raise ValueError("Unknown Fortran standard: %s" % self._settings[
                             'standard'])
        # 模块使用情况的默认字典，例如: use iso_c_binding, only: c_int
        self.module_uses = defaultdict(set)

    # 属性方法，根据源代码格式返回前导空白字符串的字典
    @property
    def _lead(self):
        if self._settings['source_format'] == 'fixed':
            return {'code': "      ", 'cont': "     @ ", 'comment': "C     "}
        elif self._settings['source_format'] == 'free':
            return {'code': "", 'cont': "      ", 'comment': "! "}
        else:
            raise ValueError("Unknown source format: %s" % self._settings['source_format'])

    # 打印符号表达式的方法，根据名称混淆设置处理符号
    def _print_Symbol(self, expr):
        if self._settings['name_mangling'] == True:
            if expr not in self.mangled_symbols:
                name = expr.name
                while name.lower() in self.used_name:
                    name += '_'
                self.used_name.append(name.lower())
                if name == expr.name:
                    self.mangled_symbols[expr] = expr
                else:
                    self.mangled_symbols[expr] = Symbol(name)

            expr = expr.xreplace(self.mangled_symbols)

        name = super()._print_Symbol(expr)
        return name

    # 计算索引位置评分的方法，根据位置p返回负值乘以5
    def _rate_index_position(self, p):
        return -p*5

    # 获取语句字符串的方法，简单返回输入的代码字符串
    def _get_statement(self, codestring):
        return codestring

    # 获取注释字符串的方法，格式化输入的文本为Fortran风格的注释
    def _get_comment(self, text):
        return "! {}".format(text)

    # 声明数值常量的方法，返回参数声明的Fortran字符串
    def _declare_number_const(self, name, value):
        return "parameter ({} = {})".format(name, self._print(value))
    # 将表达式添加到 _number_symbols 集合中，并返回表达式的字符串表示
    def _print_NumberSymbol(self, expr):
        # 如果 Number symbol 在此处或者通过 _printmethod 实现的方法中未实现，
        # 则注册并计算其值
        self._number_symbols.add((expr, Float(expr.evalf(self._settings['precision']))))
        return str(expr)

    # 格式化代码行，先缩进再用 Fortran 风格包装
    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    # 遍历矩阵的索引，返回一个生成器，按行列顺序遍历
    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    # 获取循环的开头和结尾行，根据给定的索引信息
    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # Fortran 数组从 1 开始，到维度结束
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("do %s = %s, %s" % (var, start, stop))
            close_lines.append("end do")
        return open_lines, close_lines

    # 打印符号表达式的符号
    def _print_sign(self, expr):
        from sympy.functions.elementary.complexes import Abs
        arg, = expr.args
        if arg.is_integer:
            new_expr = merge(0, isign(1, arg), Eq(arg, 0))
        elif (arg.is_complex or arg.is_infinite):
            new_expr = merge(cmplx(literal_dp(0), literal_dp(0)), arg/Abs(arg), Eq(Abs(arg), literal_dp(0)))
        else:
            new_expr = merge(literal_dp(0), dsign(literal_dp(1), arg), Eq(arg, literal_dp(0)))
        return self._print(new_expr)
    # 定义一个方法，用于将 Piecewise 表达式打印成字符串形式
    def _print_Piecewise(self, expr):
        # 检查 Piecewise 表达式的最后一个条件是否为 True
        if expr.args[-1].cond != True:
            # 如果最后一个条件不是 True，则抛出数值错误
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        # 初始化空列表，用于存储生成的代码行
        lines = []
        # 如果表达式中包含赋值语句
        if expr.has(Assignment):
            # 遍历 Piecewise 表达式的每对 (e, c)
            for i, (e, c) in enumerate(expr.args):
                # 第一条条件语句
                if i == 0:
                    lines.append("if (%s) then" % self._print(c))
                # 最后一条条件语句且条件为 True
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                # 中间的条件语句
                else:
                    lines.append("else if (%s) then" % self._print(c))
                # 添加表达式 e 的打印结果作为代码行
                lines.append(self._print(e))
            # 添加结束 if 结构的语句
            lines.append("end if")
            # 将所有代码行组合成一个字符串返回
            return "\n".join(lines)
        # 如果 Fortran 标准版本 >= 95
        elif self._settings["standard"] >= 95:
            # 只支持 F95 及更新版本：
            # 在表达式中使用 Piecewise，需要进行内联运算符处理
            # 这种方法的缺点是，对于跨越多行的语句（如矩阵或索引表达式），内联运算符不起作用
            pattern = "merge({T}, {F}, {COND})"
            # 打印最后一个表达式的代码
            code = self._print(expr.args[-1].expr)
            # 处理除最后一个以外的所有 (e, c) 对
            terms = list(expr.args[:-1])
            while terms:
                e, c = terms.pop()
                # 打印表达式 e 和条件 c
                expr = self._print(e)
                cond = self._print(c)
                # 使用 pattern 格式化字符串生成代码
                code = pattern.format(T=expr, F=code, COND=cond)
            # 返回生成的代码
            return code
        else:
            # 在不支持 F95 之前的标准中，不支持使用内联运算符处理 Piecewise
            raise NotImplementedError("Using Piecewise as an expression using "
                                      "inline operators is not supported in "
                                      "standards earlier than Fortran95.")

    # 打印 MatrixElement 对象的字符串表示
    def _print_MatrixElement(self, expr):
        return "{}({}, {})".format(self.parenthesize(expr.parent,
                PRECEDENCE["Atom"], strict=True), expr.i + 1, expr.j + 1)
    def _print_Add(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # collect the purely real and purely imaginary parts:
        pure_real = []
        pure_imaginary = []
        mixed = []
        for arg in expr.args:
            if arg.is_number and arg.is_real:
                pure_real.append(arg)
            elif arg.is_number and arg.is_imaginary:
                pure_imaginary.append(arg)
            else:
                mixed.append(arg)
        if pure_imaginary:
            if mixed:
                # Determine operator precedence
                PREC = precedence(expr)
                # Combine the mixed terms
                term = Add(*mixed)
                t = self._print(term)
                # Handle negative signs
                if t.startswith('-'):
                    sign = "-"
                    t = t[1:]
                else:
                    sign = "+"
                # Ensure correct parentheses around terms with lower precedence
                if precedence(term) < PREC:
                    t = "(%s)" % t

                # Format the final output for complex numbers in Fortran style
                return "cmplx(%s,%s) %s %s" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                    sign, t,
                )
            else:
                # Format the output for purely imaginary terms
                return "cmplx(%s,%s)" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                )
        else:
            # If no purely imaginary parts, defer to the base class method
            return CodePrinter._print_Add(self, expr)

    def _print_Function(self, expr):
        # All constant function args are evaluated as floats
        prec =  self._settings['precision']
        # Evaluate function arguments as floats
        args = [N(a, prec) for a in expr.args]
        eval_expr = expr.func(*args)
        # If evaluated expression is not a Function object, print it
        if not isinstance(eval_expr, Function):
            return self._print(eval_expr)
        else:
            # Otherwise, recurse with the evaluated function expression
            return CodePrinter._print_Function(self, expr.func(*args))

    def _print_Mod(self, expr):
        # NOTE : Fortran has the functions mod() and modulo(). modulo() behaves
        # the same wrt to the sign of the arguments as Python and SymPy's
        # modulus computations (% and Mod()) but is not available in Fortran 66
        # or Fortran 77, thus we raise an error.
        if self._settings['standard'] in [66, 77]:
            # Raise NotImplementedError for incompatible Fortran standards
            msg = ("Python % operator and SymPy's Mod() function are not "
                   "supported by Fortran 66 or 77 standards.")
            raise NotImplementedError(msg)
        else:
            # Print Fortran-style modulo function for other standards
            x, y = expr.args
            return "      modulo({}, {})".format(self._print(x), self._print(y))

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # Return Fortran-style representation of imaginary unit
        return "cmplx(0,1)"

    def _print_int(self, expr):
        # Convert integer to string representation
        return str(expr)

    def _print_Mul(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # Check if expression is a number and imaginary, then format accordingly
        if expr.is_number and expr.is_imaginary:
            return "cmplx(0,%s)" % (
                self._print(-S.ImaginaryUnit*expr)
            )
        else:
            # Default behavior to base class method for other cases
            return CodePrinter._print_Mul(self, expr)
    # 打印幂表达式，根据表达式优先级选择是否加括号
    def _print_Pow(self, expr):
        PREC = precedence(expr)
        # 如果指数为 -1，打印为分数形式
        if equal_valued(expr.exp, -1):
            return '%s/%s' % (
                self._print(literal_dp(1)),  # 打印分子
                self.parenthesize(expr.base, PREC)  # 打印加括号的基数表达式
            )
        # 如果指数为 0.5，打印平方根形式
        elif equal_valued(expr.exp, 0.5):
            if expr.base.is_integer:
                # Fortran 的内置 sqrt() 函数不接受整数参数
                if expr.base.is_Number:
                    return 'sqrt(%s.0d0)' % self._print(expr.base)  # 打印整数的平方根
                else:
                    return 'sqrt(dble(%s))' % self._print(expr.base)  # 打印浮点数的平方根
            else:
                return 'sqrt(%s)' % self._print(expr.base)  # 打印一般表达式的平方根
        else:
            return CodePrinter._print_Pow(self, expr)  # 其他情况调用父类的打印函数

    # 打印有理数表达式
    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return "%d.0d0/%d.0d0" % (p, q)  # 格式化打印有理数

    # 打印浮点数表达式
    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)  # 调用父类的打印函数获取打印结果
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])  # 处理科学计数法形式
        return "%sd0" % printed  # 打印普通浮点数形式

    # 打印关系表达式
    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)  # 打印左表达式
        rhs_code = self._print(expr.rhs)  # 打印右表达式
        op = expr.rel_op
        op = op if op not in self._relationals else self._relationals[op]  # 处理关系操作符
        return "{} {} {}".format(lhs_code, op, rhs_code)  # 格式化打印关系表达式

    # 打印索引表达式
    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]  # 打印所有索引
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))  # 格式化打印索引表达式

    # 打印索引标签
    def _print_Idx(self, expr):
        return self._print(expr.label)  # 直接打印索引标签

    # 打印增强赋值语句
    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)  # 打印左表达式
        rhs_code = self._print(expr.rhs)  # 打印右表达式
        return self._get_statement("{0} = {0} {1} {2}".format(
            self._print(lhs_code), self._print(expr.binop), self._print(rhs_code)))  # 格式化打印增强赋值语句

    # 打印求和表达式
    def _print_sum_(self, sm):
        params = self._print(sm.array)  # 打印求和数组表达式
        if sm.dim != None:  # 必须使用 '!= None'，不能用 'is not None'
            params += ', ' + self._print(sm.dim)  # 打印维度参数
        if sm.mask != None:  # 必须使用 '!= None'，不能用 'is not None'
            params += ', mask=' + self._print(sm.mask)  # 打印掩码参数
        return '%s(%s)' % (sm.__class__.__name__.rstrip('_'), params)  # 格式化打印求和表达式

    # 打印乘积表达式，与求和表达式相同
    def _print_product_(self, prod):
        return self._print_sum_(prod)  # 直接调用求和表达式的打印函数

    # 打印 Do 循环语句
    def _print_Do(self, do):
        excl = ['concurrent']  # 排除参数列表
        if do.step == 1:
            excl.append('step')
            step = ''
        else:
            step = ', {step}'  # 如果步长不为1，则设置步长参数

        return (
            'do {concurrent}{counter} = {first}, {last}'+step+'\n'  # 格式化打印 Do 循环语句
            '{body}\n'  # 打印循环体
            'end do\n'  # 打印结束 do 语句
        ).format(
            concurrent='concurrent ' if do.concurrent else '',  # 如果是并行循环则加入 concurrent 关键字
            **do.kwargs(apply=lambda arg: self._print(arg), exclude=excl)  # 格式化其他参数
        )
    # 打印隐式 DO 循环的字符串表示形式
    def _print_ImpliedDoLoop(self, idl):
        # 确定步长字符串，如果步长为1则为空字符串，否则为 ', {step}'
        step = '' if idl.step == 1 else ', {step}'
        # 格式化输出隐式 DO 循环的字符串表示
        return ('({expr}, {counter} = {first}, {last}'+step+')').format(
            **idl.kwargs(apply=lambda arg: self._print(arg))
        )

    # 打印 FOR 循环的字符串表示形式
    def _print_For(self, expr):
        # 获取循环目标的字符串表示
        target = self._print(expr.target)
        # 如果循环的可迭代对象是 Range 类型
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            # 抛出未实现错误，目前仅支持 Range 类型的可迭代对象
            raise NotImplementedError("Only iterable currently supported is Range")
        # 获取循环体的字符串表示
        body = self._print(expr.body)
        # 返回 FOR 循环的完整字符串表示形式
        return ('do {target} = {start}, {stop}, {step}\n'
                '{body}\n'
                'end do').format(target=target, start=start, stop=stop - 1,
                        step=step, body=body)

    # 打印类型声明的字符串表示形式
    def _print_Type(self, type_):
        # 获取类型别名或默认类型名称
        type_ = self.type_aliases.get(type_, type_)
        # 获取类型映射的字符串表示
        type_str = self.type_mappings.get(type_, type_.name)
        # 获取类型所使用的模块信息
        module_uses = self.type_modules.get(type_)
        # 如果有模块使用信息，则添加到模块使用字典中
        if module_uses:
            for k, v in module_uses:
                self.module_uses[k].add(v)
        # 返回类型的字符串表示形式
        return type_str

    # 打印元素访问表达式的字符串表示形式
    def _print_Element(self, elem):
        # 返回元素访问表达式的字符串表示形式，包括符号和索引
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.symbol),
            idxs=', '.join((self._print(arg) for arg in elem.indices))
        )

    # 打印范围表达式的字符串表示形式
    def _print_Extent(self, ext):
        # 直接返回范围表达式的字符串表示形式
        return str(ext)

    # 打印声明表达式的字符串表示形式
    def _print_Declaration(self, expr):
        # 获取变量、值、维度等信息
        var = expr.variable
        val = var.value
        dim = var.attr_params('dimension')
        intents = [intent in var.attrs for intent in (intent_in, intent_out, intent_inout)]
        # 确定意图标识
        if intents.count(True) == 0:
            intent = ''
        elif intents.count(True) == 1:
            intent = ', intent(%s)' % ['in', 'out', 'inout'][intents.index(True)]
        else:
            # 如果多个意图标识被指定，则抛出值错误异常
            raise ValueError("Multiple intents specified for %s" % self)
        
        # 如果变量是指针类型，抛出未实现错误
        if isinstance(var, Pointer):
            raise NotImplementedError("Pointers are not available by default in Fortran.")
        
        # 根据标准版本生成声明语句
        if self._settings["standard"] >= 90:
            result = '{t}{vc}{dim}{intent}{alloc} :: {s}'.format(
                t=self._print(var.type),
                vc=', parameter' if value_const in var.attrs else '',
                dim=', dimension(%s)' % ', '.join((self._print(arg) for arg in dim)) if dim else '',
                intent=intent,
                alloc=', allocatable' if allocatable in var.attrs else '',
                s=self._print(var.symbol)
            )
            # 如果有初始值，则添加到声明语句中
            if val != None: # Must be "!= None", cannot be "is not None"
                result += ' = %s' % self._print(val)
        else:
            # 对于 F77 标准，处理初始化值或参数声明语句
            if value_const in var.attrs or val:
                raise NotImplementedError("F77 init./parameter statem. req. multiple lines.")
            result = ' '.join((self._print(arg) for arg in [var.type, var.symbol]))

        # 返回最终的声明语句结果
        return result

    # 打印无穷大的字符串表示形式
    def _print_Infinity(self, expr):
        # 返回表示无穷大的字符串表达式
        return '(huge(%s) + 1)' % self._print(literal_dp(0))
    # 定义一个方法用于打印带有条件和主体的 do while 循环结构
    def _print_While(self, expr):
        return 'do while ({condition})\n{body}\nend do'.format(**expr.kwargs(
            apply=lambda arg: self._print(arg)))
    
    # 定义一个方法返回布尔真值 '.true.'
    def _print_BooleanTrue(self, expr):
        return '.true.'
    
    # 定义一个方法返回布尔假值 '.false.'
    def _print_BooleanFalse(self, expr):
        return '.false.'
    
    # 定义一个方法用于在代码行中添加指定数量的前导空格或注释标记
    def _pad_leading_columns(self, lines):
        result = []
        for line in lines:
            # 如果行以 '!' 开头，将其除去首部的 '!'，并在前面加上注释前导标记
            if line.startswith('!'):
                result.append(self._lead['comment'] + line[1:].lstrip())
            else:
                # 否则，在行首添加代码前导标记
                result.append(self._lead['code'] + line)
        return result
    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (without \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # 定义用于判断字符类型的集合
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            """Find the split point in a code line

               Arguments:
                 line    --  the line to split
                 endpos  --  the end position to consider for splitting

               Returns the position to split the line.
            """
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            # 判断是否应该在此处分割字符串
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos
        
        # 初始化结果列表
        result = []
        # 根据设置的源代码格式确定行末是否需要追加符号
        if self._settings['source_format'] == 'free':
            trailing = ' &'
        else:
            trailing = ''
        
        # 遍历输入的每一行
        for line in lines:
            if line.startswith(self._lead['comment']):
                # 处理注释行
                if len(line) > 72:
                    # 如果注释行超过72字符，进行分割
                    pos = line.rfind(" ", 6, 72)
                    if pos == -1:
                        pos = 72
                    hunk = line[:pos]
                    line = line[pos:].lstrip()
                    result.append(hunk)
                    while line:
                        pos = line.rfind(" ", 0, 66)
                        if pos == -1 or len(line) < 66:
                            pos = 66
                        hunk = line[:pos]
                        line = line[pos:].lstrip()
                        result.append("%s%s" % (self._lead['comment'], hunk))
                else:
                    result.append(line)
            elif line.startswith(self._lead['code']):
                # 处理代码行
                pos = split_pos_code(line, 72)
                hunk = line[:pos].rstrip()
                line = line[pos:].lstrip()
                if line:
                    hunk += trailing
                result.append(hunk)
                while line:
                    pos = split_pos_code(line, 65)
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                    if line:
                        hunk += trailing
                    result.append("%s%s" % (self._lead['cont'], hunk))
            else:
                result.append(line)
        
        # 返回处理后的结果列表
        return result
    def indent_code(self, code):
        """对给定的代码字符串或代码行列表进行缩进处理"""
        if isinstance(code, str):
            # 如果输入是字符串，将其拆分为行，并递归调用自身处理
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        # 检查是否使用自由格式（free form）
        free = self._settings['source_format'] == 'free'
        # 去除每行代码的前导空格和制表符
        code = [ line.lstrip(' \t') for line in code ]

        # 定义增加缩进的关键字
        inc_keyword = ('do ', 'if(', 'if ', 'do\n', 'else', 'program', 'interface')
        # 定义减少缩进的关键字
        dec_keyword = ('end do', 'enddo', 'end if', 'endif', 'else', 'end program', 'end interface')

        # 对于每行代码，检查是否包含增加或减少缩进的关键字
        increase = [ int(any(map(line.startswith, inc_keyword)))
                     for line in code ]
        decrease = [ int(any(map(line.startswith, dec_keyword)))
                     for line in code ]
        # 检查每行代码是否以续行符结束
        continuation = [ int(any(map(line.endswith, ['&', '&\n'])))
                         for line in code ]

        # 初始化缩进级别、续行填充、制表符宽度和新代码列表
        level = 0
        cont_padding = 0
        tabwidth = 3
        new_code = []
        for i, line in enumerate(code):
            if line in ('', '\n'):
                # 空行直接添加到新代码列表中
                new_code.append(line)
                continue
            # 根据减少缩进的关键字调整当前级别
            level -= decrease[i]

            # 根据是否为自由格式代码，计算当前行的填充空格
            if free:
                padding = " "*(level*tabwidth + cont_padding)
            else:
                padding = " "*level*tabwidth

            # 添加适当的填充空格后的行到新代码列表中
            line = "%s%s" % (padding, line)
            if not free:
                # 如果不是自由格式，再次处理缩进
                line = self._pad_leading_columns([line])[0]

            new_code.append(line)

            # 根据当前行是否有续行符，调整续行填充
            if continuation[i]:
                cont_padding = 2*tabwidth
            else:
                cont_padding = 0
            # 根据增加缩进的关键字调整当前级别
            level += increase[i]

        # 如果不是自由格式，使用特定方法对Fortran代码进行包装处理
        if not free:
            return self._wrap_fortran(new_code)
        return new_code

    def _print_GoTo(self, goto):
        """打印给定的goto语句对象"""
        if goto.expr:  # 如果是计算型goto语句
            return "go to ({labels}), {expr}".format(
                labels=', '.join((self._print(arg) for arg in goto.labels)),
                expr=self._print(goto.expr)
            )
        else:
            # 否则为简单的goto语句
            lbl, = goto.labels
            return "go to %s" % self._print(lbl)

    def _print_Program(self, prog):
        """打印给定的program对象"""
        return (
            "program {name}\n"
            "{body}\n"
            "end program\n"
        ).format(**prog.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Module(self, mod):
        """打印给定的module对象"""
        return (
            "module {name}\n"
            "{declarations}\n"
            "\ncontains\n\n"
            "{definitions}\n"
            "end module\n"
        ).format(**mod.kwargs(apply=lambda arg: self._print(arg)))
    def _print_Stream(self, strm):
        # 检查流对象是否为 stdout 并且设置中的标准大于等于 2003 年
        if strm.name == 'stdout' and self._settings["standard"] >= 2003:
            # 添加 ISO_C_BINDING 模块的使用，并映射为 input_unit
            self.module_uses['iso_c_binding'].add('stdint=>input_unit')
            # 返回字符串 'input_unit'
            return 'input_unit'
        # 检查流对象是否为 stderr 并且设置中的标准大于等于 2003 年
        elif strm.name == 'stderr' and self._settings["standard"] >= 2003:
            # 添加 ISO_C_BINDING 模块的使用，并映射为 error_unit
            self.module_uses['iso_c_binding'].add('stdint=>error_unit')
            # 返回字符串 'error_unit'
            return 'error_unit'
        else:
            # 如果流对象是 stdout，返回 '*'
            if strm.name == 'stdout':
                return '*'
            # 否则返回流对象的名称
            else:
                return strm.name

    def _print_Print(self, ps):
        # 检查格式字符串是否为 None
        if ps.format_string == none: # Must be '!= None', cannot be 'is not None'
            # 设置默认模板为 "print {fmt}, {iolist}"，格式为 '*'
            template = "print {fmt}, {iolist}"
            fmt = '*'
        else:
            # 根据输出流选择模板和格式字符串
            template = 'write(%(out)s, fmt="{fmt}", advance="no"), {iolist}' % {
                'out': {stderr: '0', stdout: '6'}.get(ps.file, '*')
            }
            # 调用 _print 方法处理格式字符串
            fmt = self._print(ps.format_string)
        # 返回格式化后的字符串
        return template.format(fmt=fmt, iolist=', '.join(
            (self._print(arg) for arg in ps.print_args)))

    def _print_Return(self, rs):
        # 获取返回参数
        arg, = rs.args
        # 返回格式化的返回语句
        return "{result_name} = {arg}".format(
            result_name=self._context.get('result_name', 'sympy_result'),
            arg=self._print(arg)
        )

    def _print_FortranReturn(self, frs):
        # 获取 Fortran 返回参数
        arg, = frs.args
        # 如果有返回参数，返回带参数的 return 语句，否则返回简单的 return
        if arg:
            return 'return %s' % self._print(arg)
        else:
            return 'return'

    def _head(self, entity, fp, **kwargs):
        # 获取绑定到 C 的参数
        bind_C_params = fp.attr_params('bind_C')
        # 如果没有绑定参数，设置 bind 为空字符串
        if bind_C_params is None:
            bind = ''
        else:
            # 否则，设置 bind 为绑定到 C 的参数名或默认为 ' bind(C)'
            bind = ' bind(C, name="%s")' % bind_C_params[0] if bind_C_params else ' bind(C)'
        # 获取结果名称
        result_name = self._settings.get('result_name', None)
        # 返回格式化后的实体头部字符串和参数声明
        return (
            "{entity}{name}({arg_names}){result}{bind}\n"
            "{arg_declarations}"
        ).format(
            entity=entity,
            name=self._print(fp.name),
            arg_names=', '.join([self._print(arg.symbol) for arg in fp.parameters]),
            result=(' result(%s)' % result_name) if result_name else '',
            bind=bind,
            arg_declarations='\n'.join((self._print(Declaration(arg)) for arg in fp.parameters))
        )

    def _print_FunctionPrototype(self, fp):
        # 获取函数原型的实体部分
        entity = "{} function ".format(self._print(fp.return_type))
        # 返回格式化后的函数原型字符串
        return (
            "interface\n"
            "{function_head}\n"
            "end function\n"
            "end interface"
        ).format(function_head=self._head(entity, fp))
    # 定义一个方法，用于打印函数定义
    def _print_FunctionDefinition(self, fd):
        # 检查函数是否标记为elemental
        if elemental in fd.attrs:
            prefix = 'elemental '
        # 检查函数是否标记为pure
        elif pure in fd.attrs:
            prefix = 'pure '
        else:
            prefix = ''

        # 构造函数头部的实体部分，包括返回类型和函数关键字
        entity = "{} function ".format(self._print(fd.return_type))
        
        # 使用打印上下文，打印函数名和结果
        with printer_context(self, result_name=fd.name):
            # 返回格式化的函数定义字符串，包括前缀、函数头部和函数体
            return (
                "{prefix}{function_head}\n"
                "{body}\n"
                "end function\n"
            ).format(
                prefix=prefix,
                function_head=self._head(entity, fd),
                body=self._print(fd.body)
            )

    # 定义一个方法，用于打印子程序定义
    def _print_Subroutine(self, sub):
        # 返回格式化的子程序定义字符串，包括子程序头部和子程序体
        return (
            '{subroutine_head}\n'
            '{body}\n'
            'end subroutine\n'
        ).format(
            subroutine_head=self._head('subroutine ', sub),
            body=self._print(sub.body)
        )

    # 定义一个方法，用于打印子程序调用
    def _print_SubroutineCall(self, scall):
        # 返回格式化的子程序调用字符串，包括子程序名和参数列表
        return 'call {name}({args})'.format(
            name=self._print(scall.name),
            args=', '.join((self._print(arg) for arg in scall.subroutine_args))
        )

    # 定义一个方法，用于打印use语句中的重命名
    def _print_use_rename(self, rnm):
        # 返回格式化的重命名字符串，包括原名和新名
        return "%s => %s" % tuple((self._print(arg) for arg in rnm.args))

    # 定义一个方法，用于打印use语句
    def _print_use(self, use):
        # 构造use语句的基本形式
        result = 'use %s' % self._print(use.namespace)
        # 如果存在重命名，则添加重命名部分
        if use.rename != None: # Must be '!= None', cannot be 'is not None'
            result += ', ' + ', '.join([self._print(rnm) for rnm in use.rename])
        # 如果存在only部分，则添加only限定
        if use.only != None: # Must be '!= None', cannot be 'is not None'
            result += ', only: ' + ', '.join([self._print(nly) for nly in use.only])
        return result

    # 定义一个方法，用于打印BreakToken（中断语句）
    def _print_BreakToken(self, _):
        return 'exit'

    # 定义一个方法，用于打印ContinueToken（继续语句）
    def _print_ContinueToken(self, _):
        return 'cycle'

    # 定义一个方法，用于打印数组构造器
    def _print_ArrayConstructor(self, ac):
        # 根据标准版本选择不同的数组构造器格式
        fmtstr = "[%s]" if self._settings["standard"] >= 2003 else '(/%s/)'
        # 返回格式化的数组构造器字符串，包括所有元素
        return fmtstr % ', '.join((self._print(arg) for arg in ac.elements))

    # 定义一个方法，用于打印数组元素
    def _print_ArrayElement(self, elem):
        # 返回格式化的数组元素字符串，包括数组名和索引列表
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.name),
            idxs=', '.join((self._print(arg) for arg in elem.indices))
        )
```