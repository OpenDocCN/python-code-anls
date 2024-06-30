# `D:\src\scipysrc\sympy\sympy\printing\c.py`

```
# 导入未来版本的注解功能，用于类型提示
from __future__ import annotations

# 导入 Any 类型，用于灵活类型提示
from typing import Any

# 导入函数装饰器
from functools import wraps

# 导入 itertools 中的链式迭代器
from itertools import chain

# 导入 SymPy 的核心模块和数值模块
from sympy.core import S
from sympy.core.numbers import equal_valued, Float

# 导入 SymPy 中的代码生成抽象语法树模块
from sympy.codegen.ast import (
    Assignment, Pointer, Variable, Declaration, Type,
    real, complex_, integer, bool_, float32, float64, float80,
    complex64, complex128, intc, value_const, pointer_const,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64, untyped,
    none
)

# 导入 SymPy 的代码打印模块和相关打印需求函数
from sympy.printing.codeprinter import CodePrinter, requires

# 导入 SymPy 的运算符优先级和打印优先级
from sympy.printing.precedence import precedence, PRECEDENCE

# 导入 SymPy 中的范围集合
from sympy.sets.fancysets import Range

# 导入 sympy.printing.codeprinter 模块中的 ccode 和 print_ccode 函数
from sympy.printing.codeprinter import ccode, print_ccode  # noqa:F401

# 在 C89CodePrinter._print_Function(self) 中使用的已知 SymPy 函数映射到其 C 函数名的字典
known_functions_C89 = {
    "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "floor": "floor",
    "ceiling": "ceil",
    "sqrt": "sqrt",  # 用于启用自动重写
}

# 在 C99 标准中新增的已知 SymPy 函数映射到其 C 函数名的字典
known_functions_C99 = dict(known_functions_C89, **{
    'exp2': 'exp2',
    'expm1': 'expm1',
    'log10': 'log10',
    'log2': 'log2',
    'log1p': 'log1p',
    'Cbrt': 'cbrt',
    'hypot': 'hypot',
    'fma': 'fma',
    'loggamma': 'lgamma',
    'erfc': 'erfc',
    'Max': 'fmax',
    'Min': 'fmin',
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "erf": "erf",
    "gamma": "tgamma",
})

# C 语言中的核心保留字列表，来源于 https://en.cppreference.com/w/c/keyword
reserved_words = [
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int',
    'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
    'struct', 'entry',  # 从未标准化，但我们仍将其保留在这里
    'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'
]

# 在 C99 标准中新增的保留字列表
reserved_words_c99 = ['inline', 'restrict']

def get_math_macros():
    """ 返回包含来自 math.h/cmath 的数学相关宏定义的字典

    注意，这些宏定义并不严格符合 C/C++ 标准。
    """
    # 导入必要的 SymPy 函数和类
    from sympy.codegen.cfunctions import log2, Sqrt
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.miscellaneous import sqrt

    # 返回一个字典，将 SymPy 表达式映射到对应的宏名称字符串
    return {
        S.Exp1: 'M_E',          # 自然对数的底 e 对应的宏名称
        log2(S.Exp1): 'M_LOG2E',# 以 2 为底 e 的对数对应的宏名称
        1/log(2): 'M_LOG2E',     # 1 除以以 e 为底 2 的对数对应的宏名称
        log(2): 'M_LN2',         # 以 2 为底的自然对数对应的宏名称
        log(10): 'M_LN10',       # 以 10 为底的自然对数对应的宏名称
        S.Pi: 'M_PI',            # 圆周率 π 对应的宏名称
        S.Pi/2: 'M_PI_2',        # π/2 对应的宏名称
        S.Pi/4: 'M_PI_4',        # π/4 对应的宏名称
        1/S.Pi: 'M_1_PI',        # 1/π 对应的宏名称
        2/S.Pi: 'M_2_PI',        # 2/π 对应的宏名称
        2/sqrt(S.Pi): 'M_2_SQRTPI',  # 2/√π 对应的宏名称
        2/Sqrt(S.Pi): 'M_2_SQRTPI',  # 2/√π 对应的宏名称（使用大写 Sqrt）
        sqrt(2): 'M_SQRT2',      # √2 对应的宏名称
        Sqrt(2): 'M_SQRT2',      # √2 对应的宏名称（使用大写 Sqrt）
        1/sqrt(2): 'M_SQRT1_2',  # 1/√2 对应的宏名称
        1/Sqrt(2): 'M_SQRT1_2'   # 1/√2 对应的宏名称（使用大写 Sqrt）
    }
def _as_macro_if_defined(meth):
    """ Decorator for printer methods

    When a Printer's method is decorated using this decorator the expressions printed
    will first be looked for in the attribute ``math_macros``, and if present it will
    print the macro name in ``math_macros`` followed by a type suffix for the type
    ``real``. e.g. printing ``sympy.pi`` would print ``M_PIl`` if real is mapped to float80.

    """
    @wraps(meth)
    def _meth_wrapper(self, expr, **kwargs):
        # 如果表达式在 math_macros 属性中存在，则打印 math_macros 中对应的宏名称，以及针对 real 类型的类型后缀
        if expr in self.math_macros:
            return '%s%s' % (self.math_macros[expr], self._get_math_macro_suffix(real))
        else:
            # 否则调用原始的 meth 方法进行处理
            return meth(self, expr, **kwargs)

    return _meth_wrapper


class C89CodePrinter(CodePrinter):
    """A printer to convert Python expressions to strings of C code"""
    printmethod = "_ccode"  # 打印方法名称
    language = "C"  # 代码语言设为 C
    standard = "C89"  # 使用 C89 标准
    reserved_words = set(reserved_words)  # 设置保留字集合

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,  # 精度设置为 17
        'user_functions': {},  # 用户函数为空字典
        'contract': True,  # 启用合同（contract）
        'dereference': set(),  # 解引用设为空集合
        'error_on_reserved': False,  # 遇到保留字不报错
    })

    type_aliases = {
        real: float64,  # 类型别名定义，real 映射到 float64
        complex_: complex128,  # complex_ 映射到 complex128
        integer: intc  # integer 映射到 intc
    }

    type_mappings: dict[Type, Any] = {
        real: 'double',  # real 类型映射到 C 中的 double
        intc: 'int',  # intc 类型映射到 C 中的 int
        float32: 'float',  # float32 类型映射到 C 中的 float
        float64: 'double',  # float64 类型映射到 C 中的 double
        integer: 'int',  # integer 类型映射到 C 中的 int
        bool_: 'bool',  # bool_ 类型映射到 C 中的 bool
        int8: 'int8_t',  # int8 类型映射到 C 中的 int8_t
        int16: 'int16_t',  # int16 类型映射到 C 中的 int16_t
        int32: 'int32_t',  # int32 类型映射到 C 中的 int32_t
        int64: 'int64_t',  # int64 类型映射到 C 中的 int64_t
        uint8: 'int8_t',  # uint8 类型映射到 C 中的 int8_t
        uint16: 'int16_t',  # uint16 类型映射到 C 中的 int16_t
        uint32: 'int32_t',  # uint32 类型映射到 C 中的 int32_t
        uint64: 'int64_t',  # uint64 类型映射到 C 中的 int64_t
    }

    type_headers = {
        bool_: {'stdbool.h'},  # bool 类型所需包含 stdbool.h
        int8: {'stdint.h'},  # int8 类型所需包含 stdint.h
        int16: {'stdint.h'},  # int16 类型所需包含 stdint.h
        int32: {'stdint.h'},  # int32 类型所需包含 stdint.h
        int64: {'stdint.h'},  # int64 类型所需包含 stdint.h
        uint8: {'stdint.h'},  # uint8 类型所需包含 stdint.h
        uint16: {'stdint.h'},  # uint16 类型所需包含 stdint.h
        uint32: {'stdint.h'},  # uint32 类型所需包含 stdint.h
        uint64: {'stdint.h'},  # uint64 类型所需包含 stdint.h
    }

    # 定义在使用某种类型时需要定义的宏
    type_macros: dict[Type, tuple[str, ...]] = {}

    type_func_suffixes = {
        float32: 'f',  # float32 类型的函数后缀为 'f'
        float64: '',  # float64 类型的函数后缀为空字符串
        float80: 'l'  # float80 类型的函数后缀为 'l'
    }

    type_literal_suffixes = {
        float32: 'F',  # float32 类型的字面量后缀为 'F'
        float64: '',  # float64 类型的字面量后缀为空字符串
        float80: 'L'  # float80 类型的字面量后缀为 'L'
    }

    type_math_macro_suffixes = {
        float80: 'l'  # float80 类型的数学宏后缀为 'l'
    }

    math_macros = None  # 数学宏初始化为空

    _ns = ''  # 命名空间，C++ 中使用 'std::'
    _kf: dict[str, Any] = known_functions_C89  # 已知函数字典复制
    # 初始化方法，接受一个设置字典作为参数，默认为空字典
    def __init__(self, settings=None):
        # 如果没有传入设置字典，则设为一个空字典
        settings = settings or {}
        # 如果当前对象的数学宏为空，则从设置中获取数学宏，或者使用默认的数学宏
        if self.math_macros is None:
            self.math_macros = settings.pop('math_macros', get_math_macros())
        # 更新类型别名字典，将设置中的类型别名合并进去
        self.type_aliases = dict(chain(self.type_aliases.items(),
                                       settings.pop('type_aliases', {}).items()))
        # 更新类型映射字典，将设置中的类型映射合并进去
        self.type_mappings = dict(chain(self.type_mappings.items(),
                                        settings.pop('type_mappings', {}).items()))
        # 更新类型头文件字典，将设置中的类型头文件合并进去
        self.type_headers = dict(chain(self.type_headers.items(),
                                       settings.pop('type_headers', {}).items()))
        # 更新类型宏字典，将设置中的类型宏合并进去
        self.type_macros = dict(chain(self.type_macros.items(),
                                      settings.pop('type_macros', {}).items()))
        # 更新类型函数后缀字典，将设置中的类型函数后缀合并进去
        self.type_func_suffixes = dict(chain(self.type_func_suffixes.items(),
                                             settings.pop('type_func_suffixes', {}).items()))
        # 更新类型字面量后缀字典，将设置中的类型字面量后缀合并进去
        self.type_literal_suffixes = dict(chain(self.type_literal_suffixes.items(),
                                                settings.pop('type_literal_suffixes', {}).items()))
        # 更新数学宏后缀字典，将设置中的数学宏后缀合并进去
        self.type_math_macro_suffixes = dict(chain(self.type_math_macro_suffixes.items(),
                                                   settings.pop('type_math_macro_suffixes', {}).items()))
        # 调用父类的初始化方法，传入更新后的设置字典
        super().__init__(settings)
        # 将用户定义的函数合并到已知函数字典中
        self.known_functions = dict(self._kf, **settings.get('user_functions', {}))
        # 设置解引用集合，从设置中获取或默认为空列表
        self._dereference = set(settings.get('dereference', []))
        # 初始化头文件集合
        self.headers = set()
        # 初始化库集合
        self.libraries = set()
        # 初始化宏集合
        self.macros = set()
    # 打印 Pow 表达式的字符串表示
    def _print_Pow(self, expr):
        # 如果 "Pow" 在已知函数中，则调用 _print_Function 处理
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        # 获取表达式的优先级
        PREC = precedence(expr)
        # 获取实数的函数后缀
        suffix = self._get_func_suffix(real)
        # 如果指数等于 -1，则返回倒数的表达式
        if equal_valued(expr.exp, -1):
            return '%s/%s' % (self._print_Float(Float(1.0)), self.parenthesize(expr.base, PREC))
        # 如果指数等于 0.5，则返回开平方根的表达式
        elif equal_valued(expr.exp, 0.5):
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        # 如果指数是 1/3，并且不是按照 C89 标准，则返回立方根的表达式
        elif expr.exp == S.One/3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        # 其他情况下返回幂函数的表达式
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base),
                                   self._print(expr.exp))

    # 打印 Mod 表达式的字符串表示
    def _print_Mod(self, expr):
        # 获取 Mod 表达式的分子和分母
        num, den = expr.args
        # 如果都是整数，则处理 Mod 表达式
        if num.is_integer and den.is_integer:
            # 获取表达式的优先级
            PREC = precedence(expr)
            # 将分子和分母分别括在合适的优先级中
            snum, sden = [self.parenthesize(arg, PREC) for arg in expr.args]
            # 根据 C 语言中 % 的特性，计算并返回 Mod 或者 Modulo 表达式
            if (num.is_nonnegative and den.is_nonnegative or
                num.is_nonpositive and den.is_nonpositive):
                return f"{snum} % {sden}"
            return f"(({snum} % {sden}) + {sden}) % {sden}"
        # 如果不能保证是整数，则调用 _print_math_func 处理
        return self._print_math_func(expr, known='fmod')

    # 打印 Rational 表达式的字符串表示
    def _print_Rational(self, expr):
        # 获取 Rational 表达式的分子和分母
        p, q = int(expr.p), int(expr.q)
        # 获取实数的字面后缀
        suffix = self._get_literal_suffix(real)
        # 返回格式化后的有理数字符串表示
        return '%d.0%s/%d.0%s' % (p, suffix, q, suffix)

    # 打印 Indexed 表达式的字符串表示
    def _print_Indexed(self, expr):
        # 计算一维数组的索引
        offset = getattr(expr.base, 'offset', S.Zero)
        strides = getattr(expr.base, 'strides', None)
        indices = expr.indices

        # 如果没有指定步幅或者步幅是字符串，则根据情况设置默认值
        if strides is None or isinstance(strides, str):
            dims = expr.shape
            shift = S.One
            temp = ()
            # 根据步幅类型选择遍历顺序
            if strides == 'C' or strides is None:
                traversal = reversed(range(expr.rank))
                indices = indices[::-1]
            elif strides == 'F':
                traversal = range(expr.rank)

            # 计算有效步幅
            for i in traversal:
                temp += (shift,)
                shift *= dims[i]
            strides = temp

        # 计算扁平化索引并返回索引表达式的字符串表示
        flat_index = sum(x[0]*x[1] for x in zip(indices, strides)) + offset
        return "%s[%s]" % (self._print(expr.base.label),
                           self._print(flat_index))

    # 打印 Idx 表达式的字符串表示
    def _print_Idx(self, expr):
        # 返回 Idx 表达式的标签字符串表示
        return self._print(expr.label)

    # 如果定义了宏，则作为宏处理 NumberSymbol 表达式的字符串表示
    @_as_macro_if_defined
    def _print_NumberSymbol(self, expr):
        return super()._print_NumberSymbol(expr)

    # 打印 Infinity 表达式的字符串表示
    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    # 打印 NegativeInfinity 表达式的字符串表示
    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'
    # 定义一个方法用于打印 Piecewise 表达式
    def _print_Piecewise(self, expr):
        # 检查 Piecewise 表达式的最后一个条件是否为 True
        if expr.args[-1].cond != True:
            # 如果最后一个条件不是 True，则抛出 ValueError
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        
        # 初始化代码行列表
        lines = []
        
        # 如果表达式中包含赋值语句
        if expr.has(Assignment):
            # 遍历 Piecewise 表达式的每个分支
            for i, (e, c) in enumerate(expr.args):
                # 第一个分支
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                # 最后一个分支且条件为 True
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                # 其他分支
                else:
                    lines.append("else if (%s) {" % self._print(c))
                
                # 打印当前分支的代码
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            
            # 返回整合后的代码段
            return "\n".join(lines)
        
        # 如果 Piecewise 表达式中没有赋值语句
        else:
            # 在表达式中嵌入条件运算符（三元运算符）
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c),
                                               self._print(e))
                       for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            # 返回嵌入条件运算符的代码段
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    # 定义一个方法用于打印 ITE（If-Then-Else）表达式
    def _print_ITE(self, expr):
        # 导入 Piecewise 函数
        from sympy.functions import Piecewise
        # 将 ITE 表达式重写为 Piecewise 表达式后打印
        return self._print(expr.rewrite(Piecewise, deep=False))

    # 定义一个方法用于打印矩阵元素
    def _print_MatrixElement(self, expr):
        # 格式化输出矩阵元素的字符串表示
        return "{}[{}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
                                                 strict=True), expr.j + expr.i*expr.parent.shape[1])

    # 定义一个方法用于打印符号（变量）
    def _print_Symbol(self, expr):
        # 调用父类方法获取符号的名称
        name = super()._print_Symbol(expr)
        # 如果符号在解引用设置中
        if expr in self._settings['dereference']:
            # 返回带有解引用符号的格式化字符串
            return '(*{})'.format(name)
        else:
            # 返回普通符号的名称
            return name

    # 定义一个方法用于打印关系表达式（比较表达式）
    def _print_Relational(self, expr):
        # 打印左操作数的代码
        lhs_code = self._print(expr.lhs)
        # 打印右操作数的代码
        rhs_code = self._print(expr.rhs)
        # 获取比较运算符
        op = expr.rel_op
        # 返回格式化的比较表达式的代码
        return "{} {} {}".format(lhs_code, op, rhs_code)

    # 定义一个方法用于打印 for 循环语句
    def _print_For(self, expr):
        # 打印目标变量的代码
        target = self._print(expr.target)
        # 如果循环的可迭代对象是一个 Range 对象
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            # 目前仅支持 Range 类型的可迭代对象
            raise NotImplementedError("Only iterable currently supported is Range")
        
        # 打印循环体的代码
        body = self._print(expr.body)
        
        # 返回格式化后的 for 循环语句的代码
        return ('for ({target} = {start}; {target} < {stop}; {target} += '
                '{step}) {{\n{body}\n}}').format(target=target, start=start,
                                                 stop=stop, step=step, body=body)
    # 定义一个私有方法，用于打印表达式中大于0返回1，小于0返回-1，等于0返回0的表达式
    def _print_sign(self, func):
        return '((({0}) > 0) - (({0}) < 0))'.format(self._print(func.args[0]))

    # 定义一个私有方法，用于打印表达式中的Max函数，如果Max函数已知，则直接调用_print_Function方法进行打印
    # 否则定义内部方法inner_print_max，递归地计算表达式中的最大值，以优化处理大量参数时的性能
    def _print_Max(self, expr):
        if "Max" in self.known_functions:
            return self._print_Function(expr)
        def inner_print_max(args):
            if len(args) == 1:
                return self._print(args[0])
            half = len(args) // 2
            return "((%(a)s > %(b)s) ? %(a)s : %(b)s)" % {
                'a': inner_print_max(args[:half]),
                'b': inner_print_max(args[half:])
            }
        return inner_print_max(expr.args)

    # 定义一个私有方法，用于打印表达式中的Min函数，如果Min函数已知，则直接调用_print_Function方法进行打印
    # 否则定义内部方法inner_print_min，递归地计算表达式中的最小值，以优化处理大量参数时的性能
    def _print_Min(self, expr):
        if "Min" in self.known_functions:
            return self._print_Function(expr)
        def inner_print_min(args):
            if len(args) == 1:
                return self._print(args[0])
            half = len(args) // 2
            return "((%(a)s < %(b)s) ? %(a)s : %(b)s)" % {
                'a': inner_print_min(args[:half]),
                'b': inner_print_min(args[half:])
            }
        return inner_print_min(expr.args)

    # 定义一个方法，用于给定的代码字符串或代码行列表进行缩进处理，保持代码结构的可读性
    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        # 去除每行代码的开头空格和制表符
        code = [line.lstrip(' \t') for line in code]

        # 判断每行代码是否增加或减少缩进
        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token))) for line in code]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty

    # 根据给定的类型获取函数后缀
    def _get_func_suffix(self, type_):
        return self.type_func_suffixes[self.type_aliases.get(type_, type_)]

    # 根据给定的类型获取字面值后缀
    def _get_literal_suffix(self, type_):
        return self.type_literal_suffixes[self.type_aliases.get(type_, type_)]

    # 根据给定的类型获取数学宏后缀
    def _get_math_macro_suffix(self, type_):
        alias = self.type_aliases.get(type_, type_)
        dflt = self.type_math_macro_suffixes.get(alias, '')
        return self.type_math_macro_suffixes.get(type_, dflt)

    # 打印表达式中的Tuple类型，返回用逗号分隔的元素列表，以大括号括起来
    def _print_Tuple(self, expr):
        return '{'+', '.join(self._print(e) for e in expr)+'}'

    # 将_List方法指向_Tuple方法，表示打印List类型和Tuple类型的输出方式一致
    _print_List = _print_Tuple
    # 更新对象的头部信息，使用给定类型的头部信息更新当前对象的头部信息
    def _print_Type(self, type_):
        self.headers.update(self.type_headers.get(type_, set()))
        # 更新对象的宏信息，使用给定类型的宏信息更新当前对象的宏信息
        self.macros.update(self.type_macros.get(type_, set()))
        # 返回给定类型的打印结果
        return self._print(self.type_mappings.get(type_, type_.name))

    # 打印声明表达式
    def _print_Declaration(self, decl):
        # 导入 sympy.codegen.cnodes 中的 restrict
        from sympy.codegen.cnodes import restrict
        # 获取声明中的变量和值
        var = decl.variable
        val = var.value
        # 如果变量类型为 untyped，则抛出错误，因为 C 不支持未指定类型的变量
        if var.type == untyped:
            raise ValueError("C does not support untyped variables")

        # 根据变量类型不同进行格式化输出
        if isinstance(var, Pointer):
            result = '{vc}{t} *{pc} {r}{s}'.format(
                vc='const ' if value_const in var.attrs else '',
                t=self._print(var.type),
                pc=' const' if pointer_const in var.attrs else '',
                r='restrict ' if restrict in var.attrs else '',
                s=self._print(var.symbol)
            )
        elif isinstance(var, Variable):
            result = '{vc}{t} {s}'.format(
                vc='const ' if value_const in var.attrs else '',
                t=self._print(var.type),
                s=self._print(var.symbol)
            )
        else:
            raise NotImplementedError("Unknown type of var: %s" % type(var))
        
        # 如果变量有值，则追加赋值表达式
        if val != None: # Must be "!= None", cannot be "is not None"
            result += ' = %s' % self._print(val)
        return result

    # 打印浮点数表达式
    def _print_Float(self, flt):
        # 获取浮点数的类型别名
        type_ = self.type_aliases.get(real, real)
        # 更新对象的宏信息，使用给定类型的宏信息更新当前对象的宏信息
        self.macros.update(self.type_macros.get(type_, set()))
        # 获取浮点数字面量的后缀
        suffix = self._get_literal_suffix(type_)
        # 对浮点数进行格式化输出
        num = str(flt.evalf(type_.decimal_dig))
        if 'e' not in num and '.' not in num:
            num += '.0'
        num_parts = num.split('e')
        num_parts[0] = num_parts[0].rstrip('0')
        if num_parts[0].endswith('.'):
            num_parts[0] += '0'
        return 'e'.join(num_parts) + suffix

    # 打印布尔值为 true 的表达式
    @requires(headers={'stdbool.h'})
    def _print_BooleanTrue(self, expr):
        return 'true'

    # 打印布尔值为 false 的表达式
    @requires(headers={'stdbool.h'})
    def _print_BooleanFalse(self, expr):
        return 'false'

    # 打印元素表达式
    def _print_Element(self, elem):
        # 如果元素的步长为 None，则抛出错误，因为给定偏移时预期有步长
        if elem.strides == None: # Must be "== None", cannot be "is None"
            if elem.offset != None: # Must be "!= None", cannot be "is not None"
                raise ValueError("Expected strides when offset is given")
            # 将索引列表转换为字符串格式
            idxs = ']['.join((self._print(arg) for arg in elem.indices))
        else:
            # 计算全局索引
            global_idx = sum(i*s for i, s in zip(elem.indices, elem.strides))
            if elem.offset != None: # Must be "!= None", cannot be "is not None"
                global_idx += elem.offset
            # 将全局索引转换为字符串格式
            idxs = self._print(global_idx)

        # 返回格式化后的元素表示
        return "{symb}[{idxs}]".format(
            symb=self._print(elem.symbol),
            idxs=idxs
        )

    # 打印代码块表达式
    def _print_CodeBlock(self, expr):
        """ Elements of code blocks printed as statements. """
        # 将代码块中的每个元素转换为语句，并用换行符连接起来
        return '\n'.join([self._get_statement(self._print(i)) for i in expr.args])
    # 返回一个格式化的字符串，表示一个 while 循环结构，包括条件和循环体
    def _print_While(self, expr):
        return 'while ({condition}) {{\n{body}\n}}'.format(**expr.kwargs(
            apply=lambda arg: self._print(arg)))

    # 返回一个格式化的字符串，表示一个代码块的作用域，包括其中的代码内容
    def _print_Scope(self, expr):
        return '{\n%s\n}' % self._print_CodeBlock(expr.body)

    # 根据打印表达式生成对应的 printf 或 fprintf 格式的字符串
    @requires(headers={'stdio.h'})
    def _print_Print(self, expr):
        if expr.file == none:
            template = 'printf({fmt}, {pargs})'
        else:
            template = 'fprintf(%(out)s, {fmt}, {pargs})' % {
                'out': self._print(expr.file)
            }
        return template.format(
            fmt="%s\n" if expr.format_string == none else self._print(expr.format_string),
            pargs=', '.join((self._print(arg) for arg in expr.print_args))
        )

    # 返回流的名称的字符串表示
    def _print_Stream(self, strm):
        return strm.name

    # 返回函数原型的字符串表示，包括返回类型、函数名和参数列表
    def _print_FunctionPrototype(self, expr):
        pars = ', '.join((self._print(Declaration(arg)) for arg in expr.parameters))
        return "%s %s(%s)" % (
            tuple((self._print(arg) for arg in (expr.return_type, expr.name))) + (pars,)
        )

    # 返回函数定义的字符串表示，包括函数原型和函数体的作用域
    def _print_FunctionDefinition(self, expr):
        return "%s%s" % (self._print_FunctionPrototype(expr),
                         self._print_Scope(expr))

    # 返回 return 语句的字符串表示，包括返回的表达式
    def _print_Return(self, expr):
        arg, = expr.args
        return 'return %s' % self._print(arg)

    # 返回逗号运算符的字符串表示，包括逗号分隔的多个表达式
    def _print_CommaOperator(self, expr):
        return '(%s)' % ', '.join((self._print(arg) for arg in expr.args))

    # 返回标签（label）的字符串表示，包括标签名和可能的代码块
    def _print_Label(self, expr):
        if expr.body == none:
            return '%s:' % str(expr.name)
        if len(expr.body.args) == 1:
            return '%s:\n%s' % (str(expr.name), self._print_CodeBlock(expr.body))
        return '%s:\n{\n%s\n}' % (str(expr.name), self._print_CodeBlock(expr.body))

    # 返回跳转语句（goto）的字符串表示，包括跳转的标签名
    def _print_goto(self, expr):
        return 'goto %s' % expr.label.name

    # 返回前置递增运算符的字符串表示，包括递增的表达式
    def _print_PreIncrement(self, expr):
        arg, = expr.args
        return '++(%s)' % self._print(arg)

    # 返回后置递增运算符的字符串表示，包括表达式和递增操作
    def _print_PostIncrement(self, expr):
        arg, = expr.args
        return '(%s)++' % self._print(arg)

    # 返回前置递减运算符的字符串表示，包括递减的表达式
    def _print_PreDecrement(self, expr):
        arg, = expr.args
        return '--(%s)' % self._print(arg)

    # 返回后置递减运算符的字符串表示，包括表达式和递减操作
    def _print_PostDecrement(self, expr):
        arg, = expr.args
        return '(%s)--' % self._print(arg)

    # 返回结构体（struct）或联合体（union）的字符串表示，包括关键字、名称和声明列表
    def _print_struct(self, expr):
        return "%(keyword)s %(name)s {\n%(lines)s}" % {
            "keyword": expr.__class__.__name__, "name": expr.name, "lines": ';\n'.join(
                [self._print(decl) for decl in expr.declarations] + [''])
        }

    # 返回 break 语句的字符串表示
    def _print_BreakToken(self, _):
        return 'break'

    # 返回 continue 语句的字符串表示
    def _print_ContinueToken(self, _):
        return 'continue'

    # 将 _print_union 定义为 _print_struct 的别名
    _print_union = _print_struct
# 定义一个 C99CodePrinter 类，继承自 C89CodePrinter 类
class C99CodePrinter(C89CodePrinter):
    # 设定标准为 'C99'
    standard = 'C99'
    # 将 C89 中的保留字集合与 C99 中的合并为一个集合
    reserved_words = set(reserved_words + reserved_words_c99)
    # 定义类型映射字典，包括复数类型的映射
    type_mappings=dict(chain(C89CodePrinter.type_mappings.items(), {
        complex64: 'float complex',
        complex128: 'double complex',
    }.items()))
    # 定义类型的头文件字典，包括复数类型的头文件
    type_headers = dict(chain(C89CodePrinter.type_headers.items(), {
        complex64: {'complex.h'},
        complex128: {'complex.h'}
    }.items()))

    # 将 known_functions_C99 字典复制给 _kf 变量，作为已知函数的字典
    _kf: dict[str, Any] = known_functions_C99

    # 定义包含 'f' 和 'l' 后缀版本的函数列表
    _prec_funcs = ('fabs fmod remainder remquo fma fmax fmin fdim nan exp exp2'
                   ' expm1 log log10 log2 log1p pow sqrt cbrt hypot sin cos tan'
                   ' asin acos atan atan2 sinh cosh tanh asinh acosh atanh erf'
                   ' erfc tgamma lgamma ceil floor trunc round nearbyint rint'
                   ' frexp ldexp modf scalbn ilogb logb nextafter copysign').split()

    # 定义打印 Infinity 的方法
    def _print_Infinity(self, expr):
        return 'INFINITY'

    # 定义打印 NegativeInfinity 的方法
    def _print_NegativeInfinity(self, expr):
        return '-INFINITY'

    # 定义打印 NaN 的方法
    def _print_NaN(self, expr):
        return 'NAN'

    # 定义打印数学函数的方法，具有条件的宏定义和头文件声明
    @requires(headers={'math.h'}, libraries={'m'})
    @_as_macro_if_defined
    def _print_math_func(self, expr, nest=False, known=None):
        # 如果 known 为 None，则使用类名从 known_functions 中获取已知函数
        if known is None:
            known = self.known_functions[expr.__class__.__name__]
        # 如果 known 不是字符串，则遍历已知函数列表，匹配符合条件的函数名
        if not isinstance(known, str):
            for cb, name in known:
                if cb(*expr.args):
                    known = name
                    break
            else:
                raise ValueError("No matching printer")
        # 尝试执行 known 函数，若有 TypeError 则根据实际情况获取函数后缀
        try:
            return known(self, *expr.args)
        except TypeError:
            suffix = self._get_func_suffix(real) if self._ns + known in self._prec_funcs else ''

        # 如果 nest 为 True，递归打印表达式的参数
        if nest:
            args = self._print(expr.args[0])
            if len(expr.args) > 1:
                paren_pile = ''
                for curr_arg in expr.args[1:-1]:
                    paren_pile += ')'
                    args += ', {ns}{name}{suffix}({next}'.format(
                        ns=self._ns,
                        name=known,
                        suffix=suffix,
                        next = self._print(curr_arg)
                    )
                args += ', %s%s' % (
                    self._print(expr.func(expr.args[-1])),
                    paren_pile
                )
        # 如果 nest 为 False，直接打印参数列表
        else:
            args = ', '.join((self._print(arg) for arg in expr.args))
        # 返回格式化后的函数调用字符串
        return '{ns}{name}{suffix}({args})'.format(
            ns=self._ns,
            name=known,
            suffix=suffix,
            args=args
        )

    # 定义打印 Max 函数的方法，调用 _print_math_func 方法
    def _print_Max(self, expr):
        return self._print_math_func(expr, nest=True)

    # 定义打印 Min 函数的方法，调用 _print_math_func 方法
    def _print_Min(self, expr):
        return self._print_math_func(expr, nest=True)
    # 定义一个方法用于生成循环的开头和结尾代码，接受一个索引列表作为参数
    def _get_loop_opening_ending(self, indices):
        # 存储循环开始的代码行列表
        open_lines = []
        # 存储循环结束的代码行列表
        close_lines = []
        # 定义循环的起始语句模板，使用 C99 风格的 for 循环
        loopstart = "for (int %(var)s=%(start)s; %(var)s<%(end)s; %(var)s++){"  # C99
        # 对于每个索引对象
        for i in indices:
            # 向开头代码列表中添加生成的循环起始语句
            open_lines.append(loopstart % {
                'var': self._print(i.label),    # 使用 self._print 方法打印索引变量的名称
                'start': self._print(i.lower),  # 打印索引的下界
                'end': self._print(i.upper + 1)})  # 打印索引的上界加一，构成循环条件
            # 向结束代码列表中添加循环结束符号
            close_lines.append("}")
        # 返回生成的开头和结尾代码行列表
        return open_lines, close_lines
# 遍历由数学函数名称组成的字符串列表，为每个函数名称设置对应的打印方法
for k in ('Abs Sqrt exp exp2 expm1 log log10 log2 log1p Cbrt hypot fma'
          ' loggamma sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh '
          'atanh erf erfc loggamma gamma ceiling floor').split():
    # 使用 setattr 方法为 C99CodePrinter 类动态设置方法名，格式为 _print_函数名，映射到 _print_math_func 方法
    setattr(C99CodePrinter, '_print_%s' % k, C99CodePrinter._print_math_func)

# 定义 C11CodePrinter 类，继承自 C99CodePrinter 类
class C11CodePrinter(C99CodePrinter):

    # 声明装饰器，指定函数依赖的头文件为 'stdalign.h'
    @requires(headers={'stdalign.h'})
    # 定义 _print_alignof 方法，用于打印 alignof 表达式
    def _print_alignof(self, expr):
        # 从表达式的参数中获取参数值
        arg, = expr.args
        # 返回格式化后的 alignof 表达式字符串
        return 'alignof(%s)' % self._print(arg)

# 定义 c_code_printers 字典，将字符串 'c89', 'c99', 'c11' 映射到对应的打印器类
c_code_printers = {
    'c89': C89CodePrinter,
    'c99': C99CodePrinter,
    'c11': C11CodePrinter
}
```