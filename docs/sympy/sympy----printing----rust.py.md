# `D:\src\scipysrc\sympy\sympy\printing\rust.py`

```
"""
Rust code printer

The `RustCodePrinter` converts SymPy expressions into Rust expressions.

A complete code generator, which uses `rust_code` extensively, can be found
in `sympy.utilities.codegen`. The `codegen` module can be used to generate
complete source code files.

"""

# Possible Improvement
#
# * make sure we follow Rust Style Guidelines_
# * make use of pattern matching
# * better support for reference
# * generate generic code and use trait to make sure they have specific methods
# * use crates_ to get more math support
#     - num_
#         + BigInt_, BigUint_
#         + Complex_
#         + Rational64_, Rational32_, BigRational_
#
# .. _crates: https://crates.io/
# .. _Guidelines: https://github.com/rust-lang/rust/tree/master/src/doc/style
# .. _num: http://rust-num.github.io/num/num/
# .. _BigInt: http://rust-num.github.io/num/num/bigint/struct.BigInt.html
# .. _BigUint: http://rust-num.github.io/num/num/bigint/struct.BigUint.html
# .. _Complex: http://rust-num.github.io/num/num/complex/struct.Complex.html
# .. _Rational32: http://rust-num.github.io/num/num/rational/type.Rational32.html
# .. _Rational64: http://rust-num.github.io/num/num/rational/type.Rational64.html
# .. _BigRational: http://rust-num.github.io/num/num/rational/type.BigRational.html

from __future__ import annotations
from typing import Any

from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter

# Rust's methods for integer and float can be found at here :
#
# * `Rust - Primitive Type f64 <https://doc.rust-lang.org/std/primitive.f64.html>`_
# * `Rust - Primitive Type i64 <https://doc.rust-lang.org/std/primitive.i64.html>`_

# Function Style :
#
# 1. args[0].func(args[1:]), method with arguments
# 2. args[0].func(), method without arguments
# 3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())
# 4. func(args), function with arguments

# dictionary mapping SymPy function to (argument_conditions, Rust_function).
# Used in RustCodePrinter._print_Function(self)

# f64 method in Rust
known_functions = {
    # "": "is_nan",
    # "": "is_infinite",
    # "": "is_finite",
    # "": "is_normal",
    # "": "classify",
    "floor": "floor",  # Maps SymPy's `floor` function to Rust's `floor` method
    "ceiling": "ceil",  # Maps SymPy's `ceiling` function to Rust's `ceil` method
    # "": "round",
    # "": "trunc",
    # "": "fract",
    "Abs": "abs",  # Maps SymPy's `Abs` function to Rust's `abs` method
    "sign": "signum",  # Maps SymPy's `sign` function to Rust's `signum` method
    # "": "is_sign_positive",
    # "": "is_sign_negative",
    # "": "mul_add",
    "Pow": [(lambda base, exp: equal_valued(exp, -1), "recip", 2),   # Define operation for power with exponent -1, equivalent to reciprocal (1.0/x)
            (lambda base, exp: equal_valued(exp, 0.5), "sqrt", 2),   # Define operation for square root (x ** 0.5)
            (lambda base, exp: equal_valued(exp, -0.5), "sqrt().recip", 2),   # Define operation for reciprocal of square root (1/(x ** 0.5))
            (lambda base, exp: exp == Rational(1, 3), "cbrt", 2),    # Define operation for cube root (x ** (1/3))
            (lambda base, exp: equal_valued(base, 2), "exp2", 3),    # Define operation for exponential with base 2 (2 ** x)
            (lambda base, exp: exp.is_integer, "powi", 1),           # Define integer exponentiation operation (x ** y, for integer y)
            (lambda base, exp: not exp.is_integer, "powf", 1)],      # Define floating-point exponentiation operation (x ** y, for non-integer y)
    "exp": [(lambda exp: True, "exp", 2)],   # Define exponential function (e ** x)
    "log": "ln",    # Alias for natural logarithm (ln)
    # "": "log",          # Placeholder for logarithm with unspecified base (number.log(base))
    # "": "log2",         # Placeholder for base-2 logarithm (log base 2)
    # "": "log10",        # Placeholder for base-10 logarithm (log base 10)
    # "": "to_degrees",   # Placeholder for converting radians to degrees
    # "": "to_radians",   # Placeholder for converting degrees to radians
    "Max": "max",    # Alias for maximum value calculation
    "Min": "min",    # Alias for minimum value calculation
    # "": "hypot",        # Placeholder for computing hypotenuse (sqrt(x**2 + y**2))
    "sin": "sin",    # Sine function
    "cos": "cos",    # Cosine function
    "tan": "tan",    # Tangent function
    "asin": "asin",  # Inverse sine function (arcsin)
    "acos": "acos",  # Inverse cosine function (arccos)
    "atan": "atan",  # Inverse tangent function (arctan)
    "atan2": "atan2",    # Two-argument inverse tangent function (arctan2)
    # "": "sin_cos",      # Placeholder for simultaneous computation of sine and cosine
    # "": "exp_m1",       # Placeholder for exponential minus one (e ** x - 1)
    # "": "ln_1p",        # Placeholder for natural logarithm of (1 + x)
    "sinh": "sinh",  # Hyperbolic sine function
    "cosh": "cosh",  # Hyperbolic cosine function
    "tanh": "tanh",  # Hyperbolic tangent function
    "asinh": "asinh",    # Inverse hyperbolic sine function (arcsinh)
    "acosh": "acosh",    # Inverse hyperbolic cosine function (arccosh)
    "atanh": "atanh",    # Inverse hyperbolic tangent function (arctanh)
    "sqrt": "sqrt",  # Square root function, enabled for automatic rewrites
}

# i64 method in Rust
# known_functions_i64 是一个字典，包含了各种与 i64 类型相关的方法及其名称
known_functions_i64 = {
    "": "min_value",           # 返回 i64 类型的最小值
    "": "max_value",           # 返回 i64 类型的最大值
    "": "from_str_radix",      # 将字符串解析为指定基数的整数
    "": "count_ones",          # 返回该整数二进制表示中的 1 的数量
    "": "count_zeros",         # 返回该整数二进制表示中的 0 的数量
    "": "leading_zeros",       # 返回该整数二进制表示中前导 0 的数量
    "": "trainling_zeros",     # 返回该整数二进制表示中末尾 0 的数量（可能是拼写错误，应为 trailing_zeros）
    "": "rotate_left",         # 将该整数左旋指定的位数
    "": "rotate_right",        # 将该整数右旋指定的位数
    "": "swap_bytes",          # 交换该整数的字节顺序
    "": "from_be",             # 将大端格式的字节转换为整数
    "": "from_le",             # 将小端格式的字节转换为整数
    "": "to_be",               # 将整数转换为大端格式的字节
    "": "to_le",               # 将整数转换为小端格式的字节
    "": "checked_add",         # 执行加法并检查是否溢出
    "": "checked_sub",         # 执行减法并检查是否溢出
    "": "checked_mul",         # 执行乘法并检查是否溢出
    "": "checked_div",         # 执行除法并检查是否溢出
    "": "checked_rem",         # 执行取余并检查是否溢出
    "": "checked_neg",         # 执行取负并检查是否溢出
    "": "checked_shl",         # 执行左移并检查是否溢出
    "": "checked_shr",         # 执行右移并检查是否溢出
    "": "checked_abs",         # 执行取绝对值并检查是否溢出
    "": "saturating_add",      # 执行饱和加法
    "": "saturating_sub",      # 执行饱和减法
    "": "saturating_mul",      # 执行饱和乘法
    "": "wrapping_add",        # 执行包装（不检查）加法
    "": "wrapping_sub",        # 执行包装（不检查）减法
    "": "wrapping_mul",        # 执行包装（不检查）乘法
    "": "wrapping_div",        # 执行包装（不检查）除法
    "": "wrapping_rem",        # 执行包装（不检查）取余
    "": "wrapping_neg",        # 执行包装（不检查）取负
    "": "wrapping_shl",        # 执行包装（不检查）左移
    "": "wrapping_shr",        # 执行包装（不检查）右移
    "": "wrapping_abs",        # 执行包装（不检查）取绝对值
    "": "overflowing_add",     # 执行溢出（不检查）加法
    "": "overflowing_sub",     # 执行溢出（不检查）减法
    "": "overflowing_mul",     # 执行溢出（不检查）乘法
    "": "overflowing_div",     # 执行溢出（不检查）除法
    "": "overflowing_rem",     # 执行溢出（不检查）取余
    "": "overflowing_neg",     # 执行溢出（不检查）取负
    "": "overflowing_shl",     # 执行溢出（不检查）左移
    "": "overflowing_shr",     # 执行溢出（不检查）右移
    "": "overflowing_abs",     # 执行溢出（不检查）取绝对值
    "Pow": "pow",               # 计算该整数的指定次幂
    "Abs": "abs",               # 返回该整数的绝对值
    "sign": "signum",           # 返回该整数的符号
    "": "is_positive",         # 检查该整数是否为正数
    "": "is_negnative",        # 检查该整数是否为负数
}

# Rust 语言的保留关键字列表，用于语法分析和词法分析
reserved_words = ['abstract',     # 抽象
                  'alignof',     # 对齐
                  'as',          # 作为
                  'become',      # 变为
                  'box',         # 盒子
                  'break',       # 中断
                  'const',       # 常量
                  'continue',    # 继续
                  'crate',       # 创建
                  'do',          # 执行
                  'else',        # 否则
                  'enum',        # 枚举
                  'extern',      # 外部
                  'false',       # 假
                  'final',       # 最终
                  'fn',          # 函数
                  'for',         # 对于
                  'if',          # 如果
                  'impl',        # 实现
                  'in',          # 在
                  'let',         # 让
                  'loop',        # 循环
                  'macro',       # 宏
                  'match',       # 匹配
                  'mod',         # 模块
                  'move',        # 移动
                  'mut',         # 可变
                  'offsetof',    # 偏移量
                  'override',    # 覆盖
                  'priv',        # 私有
                  'proc',        # 过程
                  'pub',         # 公共
                  'pure',        # 纯
                  'ref',         # 引用
                  'return',      # 返回
                  'Self',        # 自身
                  'self',        # 自身
                  'sizeof',      # 大小
                  'static',      # 静态
                  'struct',      # 结构
                  'super',       # 超级
                  'trait',       # 特征
                  'true',        # 真
                  'type',        # 类型
                  'typeof',      # 类型的类型
                  'unsafe',      # 不安全
                  'unsized',     # 非大小限定的
                  'use',         # 使用
                  'virtual',     # 虚拟的
                  'where',       # 在哪里
                  'while',       # 当
                  'yield']       # 产出

class RustCodePrinter(CodePrinter):
    """一个将 SymPy 表达式转换为 Rust 代码字符串的打印机"""
    printmethod = "_rust_code"
    # 设置变量 language 为 "Rust"
    language = "Rust"

    # 定义默认设置 _default_settings 为字典，继承 CodePrinter._default_settings 并添加额外设置
    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
    })

    # 初始化函数，接受一个 settings 字典作为参数
    def __init__(self, settings={}):
        # 调用父类 CodePrinter 的初始化方法，传入 settings
        CodePrinter.__init__(self, settings)
        # 复制 known_functions 字典
        self.known_functions = dict(known_functions)
        # 从 settings 中获取 'user_functions' 键对应的值，默认为空字典，更新到 known_functions
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        # 从 settings 中获取 'dereference' 键对应的集合，转换为 set，赋值给 self._dereference
        self._dereference = set(settings.get('dereference', []))
        # 设置 reserved_words 为保留关键字集合
        self.reserved_words = set(reserved_words)

    # 计算索引位置评分的私有方法，参数 p 乘以 5
    def _rate_index_position(self, p):
        return p*5

    # 根据传入的 codestring 返回一个带分号的语句
    def _get_statement(self, codestring):
        return "%s;" % codestring

    # 根据传入的 text 返回一个以双斜杠开头的注释语句
    def _get_comment(self, text):
        return "// %s" % text

    # 根据名称和值声明一个浮点数常量的方法
    def _declare_number_const(self, name, value):
        return "const %s: f64 = %s;" % (name, value)

    # 格式化代码的私有方法，调用 indent_code 方法
    def _format_code(self, lines):
        return self.indent_code(lines)

    # 遍历矩阵的索引，返回每个索引的元组生成器
    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    # 获取循环开始和结束的方法，返回开启和关闭循环的行列表
    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for %(var)s in %(start)s..%(end)s {"
        for i in indices:
            # Rust 数组从 0 开始到维度-1结束
            open_lines.append(loopstart % {
                'var': self._print(i),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    # 打印调用者变量的方法，根据表达式的参数数量和是否为数字返回不同格式的字符串
    def _print_caller_var(self, expr):
        if len(expr.args) > 1:
            # 对于类似 `sin(x + y + z)` 这样的表达式，确保能够得到 '(x + y + z).sin()'
            # 而不是 'x + y + z.sin()'
            return '(' + self._print(expr) + ')'
        elif expr.is_number:
            return self._print(expr, _type=True)
        else:
            return self._print(expr)
    # 定义一个方法，用于打印 `Function` 类型的表达式
    def _print_Function(self, expr):
        """
        basic function for printing `Function`

        Function Style :

        1. args[0].func(args[1:]), method with arguments
        2. args[0].func(), method without arguments
        3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())
        4. func(args), function with arguments
        """

        # 检查表达式的函数名是否在已知函数列表中
        if expr.func.__name__ in self.known_functions:
            # 获取对应的条件函数列表或函数名
            cond_func = self.known_functions[expr.func.__name__]
            func = None
            style = 1
            # 如果条件函数是字符串，则直接作为函数名
            if isinstance(cond_func, str):
                func = cond_func
            else:
                # 否则遍历条件函数列表，找到匹配的条件并获取对应的函数名和样式
                for cond, func, style in cond_func:
                    if cond(*expr.args):
                        break
            # 如果找到对应的函数名
            if func is not None:
                # 根据样式生成打印字符串
                if style == 1:
                    ret = "%(var)s.%(method)s(%(args)s)" % {
                        'var': self._print_caller_var(expr.args[0]),
                        'method': func,
                        'args': self.stringify(expr.args[1:], ", ") if len(expr.args) > 1 else ''
                    }
                elif style == 2:
                    ret = "%(var)s.%(method)s()" % {
                        'var': self._print_caller_var(expr.args[0]),
                        'method': func,
                    }
                elif style == 3:
                    ret = "%(var)s.%(method)s()" % {
                        'var': self._print_caller_var(expr.args[1]),
                        'method': func,
                    }
                else:
                    ret = "%(func)s(%(args)s)" % {
                        'func': func,
                        'args': self.stringify(expr.args, ", "),
                    }
                return ret
        # 如果表达式的函数名在未知函数列表中，并且表达式具有 `_imp_` 属性且 `_imp_` 属性是 Lambda 类型
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            # 返回内联函数的打印结果
            return self._print(expr._imp_(*expr.args))
        # 如果表达式的函数名在可重写函数列表中
        elif expr.func.__name__ in self._rewriteable_functions:
            # 简单重写为支持的函数
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all(self._can_print(f) for f in required_fs):
                return self._print(expr.rewrite(target_f))
        else:
            # 否则返回不支持打印的结果
            return self._print_not_supported(expr)

    # 打印 `Pow` 类型的表达式
    def _print_Pow(self, expr):
        if expr.base.is_integer and not expr.exp.is_integer:
            # 如果底数是整数且指数不是整数，则转换为 Float 类型并打印
            expr = type(expr)(Float(expr.base), expr.exp)
            return self._print(expr)
        # 否则调用打印 Function 方法
        return self._print_Function(expr)

    # 打印 `Float` 类型的表达式
    def _print_Float(self, expr, _type=False):
        # 调用父类的打印方法获取字符串表示
        ret = super()._print_Float(expr)
        if _type:
            return ret + '_f64'  # 如果有类型标志，添加后缀
        else:
            return ret

    # 打印 `Integer` 类型的表达式
    def _print_Integer(self, expr, _type=False):
        # 调用父类的打印方法获取字符串表示
        ret = super()._print_Integer(expr)
        if _type:
            return ret + '_i32'  # 如果有类型标志，添加后缀
        else:
            return ret
    def _print_Rational(self, expr):
        # 将有理数表达式转换为指定格式的字符串
        p, q = int(expr.p), int(expr.q)
        return '%d_f64/%d.0' % (p, q)

    def _print_Relational(self, expr):
        # 将关系表达式转换为字符串表示形式
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        # 计算一维数组的索引
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

    def _print_Idx(self, expr):
        # 返回索引标签的名称
        return expr.label.name

    def _print_Dummy(self, expr):
        # 返回虚拟变量的名称
        return expr.name

    def _print_Exp1(self, expr, _type=False):
        # 返回常数 e 的表示形式
        return "E"

    def _print_Pi(self, expr, _type=False):
        # 返回常数 π 的表示形式
        return 'PI'

    def _print_Infinity(self, expr, _type=False):
        # 返回正无穷大的表示形式
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr, _type=False):
        # 返回负无穷大的表示形式
        return 'NEG_INFINITY'

    def _print_BooleanTrue(self, expr, _type=False):
        # 返回布尔值 True 的表示形式
        return "true"

    def _print_BooleanFalse(self, expr, _type=False):
        # 返回布尔值 False 的表示形式
        return "false"

    def _print_bool(self, expr, _type=False):
        # 将布尔表达式转换为小写字符串形式
        return str(expr).lower()

    def _print_NaN(self, expr, _type=False):
        # 返回非数值（NaN）的表示形式
        return "NAN"

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # 检查最后一个条件是否为 True，确保生成的函数在某些条件下可以返回结果
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []

        for i, (e, c) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s) {" % self._print(c))
            elif i == len(expr.args) - 1 and c == True:
                lines[-1] += " else {"
            else:
                lines[-1] += " else if (%s) {" % self._print(c)
            code0 = self._print(e)
            lines.append(code0)
            lines.append("}")

        if self._settings['inline']:
            return " ".join(lines)
        else:
            return "\n".join(lines)

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        # 将 ITE（if-then-else）表达式转换为 Piecewise 形式再进行打印
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixBase(self, A):
        if A.cols == 1:
            # 如果矩阵只有一列，则返回其元素列表的字符串表示
            return "[%s]" % ", ".join(self._print(a) for a in A)
        else:
            # 目前不支持完整的矩阵功能，需要 Rust 中的 Crates 支持
            raise ValueError("Full Matrix Support in Rust need Crates (https://crates.io/keywords/matrix).")
    # 禁止将稀疏矩阵转换为密集矩阵
    def _print_SparseRepMatrix(self, mat):
        # 调用方法来显示不支持的信息
        return self._print_not_supported(mat)

    # 根据矩阵元素对象打印其字符串表示
    def _print_MatrixElement(self, expr):
        return "%s[%s]" % (expr.parent,
                           expr.j + expr.i*expr.parent.shape[1])

    # 根据符号对象打印其字符串表示
    def _print_Symbol(self, expr):
        # 调用超类方法获取符号的基本名称
        name = super()._print_Symbol(expr)

        # 如果符号在解引用表中，则添加指针解引用符号，否则直接返回符号名称
        if expr in self._dereference:
            return '(*%s)' % name
        else:
            return name

    # 根据赋值表达式对象打印其字符串表示
    def _print_Assignment(self, expr):
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs

        # 如果设置允许合并（contract）且左侧或右侧包含 IndexedBase，则需要打印相应的循环
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # 检查是否需要循环，并打印所需的循环结构
            return self._doprint_loops(rhs, lhs)
        else:
            # 否则，分别打印左侧和右侧表达式，并以赋值语句形式返回
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    # 缩进代码块或代码行列表
    def indent_code(self, code):
        """接受代码字符串或代码行列表"""

        # 如果输入是字符串，则递归调用本方法以处理代码行列表
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "    "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        # 去除每行代码的左侧空白和制表符
        code = [line.lstrip(' \t') for line in code]

        # 确定每行代码是否需要增加或减少缩进
        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token)))
                    for line in code]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab * level, line))
            level += increase[n]
        return pretty
# 定义一个函数，将 SymPy 表达式转换为 Rust 代码字符串
def rust_code(expr, assign_to=None, **settings):
    """Converts an expr to a string of Rust code

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
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)].  See below for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
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

    >>> from sympy import rust_code, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rust_code((2*tau)**Rational(7, 2))
    '8*1.4142135623731*tau.powf(7_f64/2.0)'
    >>> rust_code(sin(x), assign_to="s")
    's = x.sin();'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs", 4),
    ...           (lambda x: x.is_integer, "ABS", 4)],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> rust_code(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    '(fabs(x) + x.CEIL()).f()'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    """
    # 使用 RustCodePrinter 将 sympy 表达式转换为 Rust 代码字符串
    return RustCodePrinter(settings).doprint(expr, assign_to)
# 定义一个函数，用于打印给定表达式的 Rust 代码表示
def print_rust_code(expr, **settings):
    """Prints Rust representation of the given expression."""
    # 调用 rust_code 函数生成给定表达式的 Rust 代码表示，并打印出来
    print(rust_code(expr, **settings))
```