# `D:\src\scipysrc\sympy\sympy\printing\octave.py`

```
"""
Octave (and Matlab) code printer

The `OctaveCodePrinter` converts SymPy expressions into Octave expressions.
It uses a subset of the Octave language for Matlab compatibility.

A complete code generator, which uses `octave_code` extensively, can be found
in `sympy.utilities.codegen`.  The `codegen` module can be used to generate
complete source code files.

"""

# 导入必要的模块和类
from __future__ import annotations
from typing import Any

from sympy.core import Mul, Pow, S, Rational  # 导入 SymPy 的核心类
from sympy.core.mul import _keep_coeff  # 导入 SymPy 的乘法相关函数
from sympy.core.numbers import equal_valued  # 导入 SymPy 的数值比较函数
from sympy.printing.codeprinter import CodePrinter  # 导入 SymPy 的代码打印器
from sympy.printing.precedence import precedence, PRECEDENCE  # 导入 SymPy 的操作符优先级定义
from re import search  # 导入正则表达式搜索函数

# List of known functions.  First, those that have the same name in
# SymPy and Octave.   This is almost certainly incomplete!
# 已知在 SymPy 和 Octave 中名称相同的函数列表，可能不完整
known_fcns_src1 = ["sin", "cos", "tan", "cot", "sec", "csc",
                   "asin", "acos", "acot", "atan", "atan2", "asec", "acsc",
                   "sinh", "cosh", "tanh", "coth", "csch", "sech",
                   "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
                   "erfc", "erfi", "erf", "erfinv", "erfcinv",
                   "besseli", "besselj", "besselk", "bessely",
                   "bernoulli", "beta", "euler", "exp", "factorial", "floor",
                   "fresnelc", "fresnels", "gamma", "harmonic", "log",
                   "polylog", "sign", "zeta", "legendre"]

# These functions have different names ("SymPy": "Octave"), more
# generally a mapping to (argument_conditions, octave_function).
# 这些函数在 SymPy 和 Octave 中有不同的名称映射关系
known_fcns_src2 = {
    "Abs": "abs",
    "arg": "angle",  # arg/angle 在 Octave 中可用，但在 Matlab 中只能用 angle
    "binomial": "bincoeff",
    "ceiling": "ceil",
    "chebyshevu": "chebyshevU",
    "chebyshevt": "chebyshevT",
    "Chi": "coshint",
    "Ci": "cosint",
    "conjugate": "conj",
    "DiracDelta": "dirac",
    "Heaviside": "heaviside",
    "im": "imag",
    "laguerre": "laguerreL",
    "LambertW": "lambertw",
    "li": "logint",
    "loggamma": "gammaln",
    "Max": "max",
    "Min": "min",
    "Mod": "mod",
    "polygamma": "psi",
    "re": "real",
    "RisingFactorial": "pochhammer",
    "Shi": "sinhint",
    "Si": "sinint",
}


class OctaveCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Octave/Matlab code.
    """
    printmethod = "_octave"
    language = "Octave"

    _operators = {
        'and': '&',
        'or': '|',
        'not': '~',
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,  # 设置默认精度为 17
        'user_functions': {},  # 用户自定义函数的字典，初始为空
        'contract': True,  # 表示是否压缩（contract）张量表示为循环（True）或只是赋值（False）
        'inline': True,  # 是否内联函数（True）
    })
    # Note: contract is for expressing tensors as loops (if True), or just
    # assignment (if False).  FIXME: this should be looked a more carefully
    # for Octave.
    # 注意：contract 用于将张量表示为循环（True）或仅赋值（False），此处应更加仔细地检查对 Octave 的影响。
    # 初始化函数，接受一个包含设置的字典作为参数
    def __init__(self, settings={}):
        # 调用父类的初始化方法，传入设置字典
        super().__init__(settings)
        # 将已知函数名和函数源自 src1 组成字典并赋值给 known_functions
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        # 更新 known_functions 字典，将 src2 中的已知函数也加入其中
        self.known_functions.update(dict(known_fcns_src2))
        # 从设置中获取用户自定义函数字典，更新到 known_functions 中
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)


    # 计算索引位置的评分，简单地将位置乘以 5
    def _rate_index_position(self, p):
        return p*5


    # 返回一个给定代码字符串的语句格式，以分号结尾
    def _get_statement(self, codestring):
        return "%s;" % codestring


    # 根据给定的文本返回一个格式化的注释字符串
    def _get_comment(self, text):
        return "% {}".format(text)


    # 返回声明数值常量的字符串，格式为 "name = value;"
    def _declare_number_const(self, name, value):
        return "{} = {};".format(name, value)


    # 格式化给定的代码行列表，调用 indent_code 方法来处理缩进
    def _format_code(self, lines):
        return self.indent_code(lines)


    # 遍历矩阵的索引，以 Octave 的 Fortran 顺序（列优先）返回所有索引的生成器
    def _traverse_matrix_indices(self, mat):
        # 获取矩阵的行数和列数
        rows, cols = mat.shape
        # 使用生成器表达式依次生成所有行列组合的索引
        return ((i, j) for j in range(cols) for i in range(rows))


    # 根据给定的索引列表生成循环的开头和结尾行
    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # 根据 Octave 数组从 1 开始的特性，格式化循环变量、起始和结束值
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            # 构建循环的开头行，并添加到 open_lines 列表中
            open_lines.append("for %s = %s:%s" % (var, start, stop))
            # 添加循环的结束行 "end" 到 close_lines 列表中
            close_lines.append("end")
        # 返回生成的循环开头行和结束行列表
        return open_lines, close_lines
    def _print_Mul(self, expr):
        # 在 Octave 中美观地打印复数
        if (expr.is_number and expr.is_imaginary and
                (S.ImaginaryUnit*expr).is_Integer):
            # 如果表达式是一个整数倍的虚数单位，打印其格式化字符串
            return "%si" % self._print(-S.ImaginaryUnit*expr)

        # 从 str.py 中借鉴而来
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            # 如果系数 c 小于 0，保留系数并改变符号
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # 分子中的项
        b = []  # 分母中的项（如果有的话）

        pow_paren = []  # 将收集所有指数为 -1 且具有多个基数元素的幂

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # 如果 order 不是 'old' 或 'none'，则按顺序排列因子
            args = Mul.make_args(expr)

        # 收集分子和分母的项
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                # 如果项是交换的、是幂，并且指数是负数有理数
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # 避免像 #14160 这样的情况
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                # 如果项是有理数且不是无穷大
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # 对于指数为 -1 并且具有多个符号的幂，添加括号
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # 以下是与 str.py 不同的处理 "*" 和 ".*"
        def multjoin(a, a_str):
            # 假设常数将首先出现
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = r + mulsym + a_str[i]
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = '/' if all(bi.is_number for bi in b) else './'
            return (sign + multjoin(a, a_str) +
                    divsym + "(%s)" % multjoin(b, b_str))
    # 将关系表达式打印为字符串，包括左右操作数和关系运算符
    def _print_Relational(self, expr):
        # 打印左操作数的代码表示
        lhs_code = self._print(expr.lhs)
        # 打印右操作数的代码表示
        rhs_code = self._print(expr.rhs)
        # 获取关系运算符
        op = expr.rel_op
        # 返回格式化后的关系表达式字符串
        return "{} {} {}".format(lhs_code, op, rhs_code)

    # 将幂运算表达式打印为字符串
    def _print_Pow(self, expr):
        # 根据参数类型确定幂运算符号
        powsymbol = '^' if all(x.is_number for x in expr.args) else '.^'

        # 获取表达式的优先级
        PREC = precedence(expr)

        # 如果指数为 0.5，则打印平方根函数表达式
        if equal_valued(expr.exp, 0.5):
            return "sqrt(%s)" % self._print(expr.base)

        # 如果表达式是可交换的
        if expr.is_commutative:
            # 如果指数为 -0.5，则打印倒数除法表达式
            if equal_valued(expr.exp, -0.5):
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "sqrt(%s)" % self._print(expr.base)
            # 如果指数为 -1，则打印倒数表达式
            if equal_valued(expr.exp, -1):
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "%s" % self.parenthesize(expr.base, PREC)

        # 默认情况下，返回幂运算表达式字符串
        return '%s%s%s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))


    # 将矩阵幂运算表达式打印为字符串
    def _print_MatPow(self, expr):
        # 获取表达式的优先级
        PREC = precedence(expr)
        # 返回格式化后的矩阵幂运算表达式字符串
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    # 将矩阵求解表达式打印为字符串
    def _print_MatrixSolve(self, expr):
        # 获取表达式的优先级
        PREC = precedence(expr)
        # 返回格式化后的矩阵求解表达式字符串
        return "%s \\ %s" % (self.parenthesize(expr.matrix, PREC),
                             self.parenthesize(expr.vector, PREC))

    # 将 Pi 表达式打印为字符串
    def _print_Pi(self, expr):
        return 'pi'

    # 将虚数单位表达式打印为字符串
    def _print_ImaginaryUnit(self, expr):
        return "1i"

    # 将自然对数底 e 表达式打印为字符串
    def _print_Exp1(self, expr):
        return "exp(1)"

    # 将黄金比例表达式打印为字符串
    def _print_GoldenRatio(self, expr):
        # 返回黄金比例的数值表达式字符串
        # 注意：这里存在一个待改进的问题，例如用于 octave_code(2*GoldenRatio) 的情况
        # return self._print((1+sqrt(S(5)))/2)
        return "(1+sqrt(5))/2"
    # 定义一个方法用于打印赋值表达式，包含了对 sympy 库的导入
    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment  # 导入赋值表达式类
        from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数类
        from sympy.tensor.indexed import IndexedBase  # 导入索引基类

        lhs = expr.lhs  # 获取赋值表达式的左侧
        rhs = expr.rhs  # 获取赋值表达式的右侧

        # 如果不是内联打印，并且右侧表达式是 Piecewise 类型
        if not self._settings["inline"] and isinstance(expr.rhs, Piecewise):
            # 将 Piecewise 的每个分支转换为赋值表达式，以便继续打印
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)

        # 如果启用了紧凑模式，并且左侧或右侧包含 IndexedBase
        if self._settings["contract"] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            # 检查是否需要打印循环，如果是则执行打印循环的函数
            return self._doprint_loops(rhs, lhs)
        else:
            # 否则，分别打印左侧和右侧的代码，并返回赋值语句字符串
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))


    # 打印 Infinity 符号的字符串表示
    def _print_Infinity(self, expr):
        return 'inf'


    # 打印 NegativeInfinity 符号的字符串表示
    def _print_NegativeInfinity(self, expr):
        return '-inf'


    # 打印 NaN 符号的字符串表示
    def _print_NaN(self, expr):
        return 'NaN'


    # 打印列表表达式的字符串表示
    def _print_list(self, expr):
        return '{' + ', '.join(self._print(a) for a in expr) + '}'


    # _print_tuple 和 _print_List 共用 _print_list 方法
    _print_tuple = _print_list
    _print_Tuple = _print_list
    _print_List = _print_list


    # 打印 BooleanTrue 符号的字符串表示
    def _print_BooleanTrue(self, expr):
        return "true"


    # 打印 BooleanFalse 符号的字符串表示
    def _print_BooleanFalse(self, expr):
        return "false"


    # 打印布尔表达式的字符串表示
    def _print_bool(self, expr):
        return str(expr).lower()


    # 打印 MatrixBase 对象的字符串表示
    def _print_MatrixBase(self, A):
        # 处理零维情况
        if (A.rows, A.cols) == (0, 0):
            return '[]'
        # 如果有任一维度为零，则返回对应维度的零矩阵表示
        elif S.Zero in A.shape:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        # 如果是 1x1 矩阵，则在 Octave 中返回标量的字符串表示
        elif (A.rows, A.cols) == (1, 1):
            return self._print(A[0, 0])
        # 否则，按行打印矩阵的元素，返回字符串表示
        return "[%s]" % "; ".join(" ".join([self._print(a) for a in A[r, :]])
                                  for r in range(A.rows))


    # 打印 SparseRepMatrix 对象的字符串表示
    def _print_SparseRepMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list()  # 获取稀疏矩阵的列列表
        # 生成稀疏矩阵的行、列和值的字符串表示
        I = Matrix([[k[0] + 1 for k in L]])
        J = Matrix([[k[1] + 1 for k in L]])
        AIJ = Matrix([[k[2] for k in L]])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                              self._print(AIJ), A.rows, A.cols)
    # 返回带有表达式的矩阵元素位置，格式为 (i+1, j+1)，并使用其父对象的括号表示
    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '(%s, %s)' % (expr.i + 1, expr.j + 1)


    # 返回矩阵切片的字符串表示，包括行和列的切片信息
    def _print_MatrixSlice(self, expr):
        def strslice(x, lim):
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        # 返回矩阵切片的完整字符串表示，包括行和列的切片
        return (self._print(expr.parent) + '(' +
                strslice(expr.rowslice, expr.parent.shape[0]) + ', ' +
                strslice(expr.colslice, expr.parent.shape[1]) + ')')


    # 返回索引对象的字符串表示，包括基础标签和所有索引的列表
    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))


    # 返回索引对象的标签的字符串表示
    def _print_Idx(self, expr):
        return self._print(expr.label)


    # 返回 Kronecker δ 函数的字符串表示，使用括号保护参数
    def _print_KroneckerDelta(self, expr):
        prec = PRECEDENCE["Pow"]
        return "double(%s == %s)" % tuple(self.parenthesize(x, prec)
                                          for x in expr.args)

    # 返回哈达玛积的字符串表示，使用.*连接所有参数的字符串表示
    def _print_HadamardProduct(self, expr):
        return '.*'.join([self.parenthesize(arg, precedence(expr))
                          for arg in expr.args])

    # 返回哈达玛幂的字符串表示，使用.**连接基数和指数的字符串表示
    def _print_HadamardPower(self, expr):
        PREC = precedence(expr)
        return '.**'.join([
            self.parenthesize(expr.base, PREC),
            self.parenthesize(expr.exp, PREC)
            ])

    # 返回单位矩阵的字符串表示，根据形状选择 eye 函数的参数
    def _print_Identity(self, expr):
        shape = expr.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            shape = [shape[0]]
        s = ", ".join(self._print(n) for n in shape)
        return "eye(" + s + ")"

    # 返回 lowergamma 函数的字符串表示，参数按格式化字符串给出
    def _print_lowergamma(self, expr):
        # Octave 实现的正则化不完全伽玛函数
        return "(gammainc({1}, {0}).*gamma({0}))".format(
            self._print(expr.args[0]), self._print(expr.args[1]))

    # 返回 uppergamma 函数的字符串表示，参数按格式化字符串给出
    def _print_uppergamma(self, expr):
        return "(gammainc({1}, {0}, 'upper').*gamma({0}))".format(
            self._print(expr.args[0]), self._print(expr.args[1]))

    # 返回 sinc 函数的字符串表示，注意除以 pi 是因为 Octave 实现的是标准化 sinc 函数
    def _print_sinc(self, expr):
        return "sinc(%s)" % self._print(expr.args[0]/S.Pi)

    # 返回 hankel1 函数的字符串表示，参数按格式化字符串给出
    def _print_hankel1(self, expr):
        return "besselh(%s, 1, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))

    # 返回 hankel2 函数的字符串表示，参数按格式化字符串给出
    def _print_hankel2(self, expr):
        return "besselh(%s, 2, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))

    # 注意：截至 2015 年，Octave 不支持球形贝塞尔函数
    # 定义一个方法，用于打印 sympy 表达式中的贝塞尔函数 Jn
    def _print_jn(self, expr):
        # 导入必要的数学函数
        from sympy.functions import sqrt, besselj
        # 从表达式中获取参数 x
        x = expr.argument
        # 计算修改后的贝塞尔函数表达式
        expr2 = sqrt(S.Pi/(2*x)) * besselj(expr.order + S.Half, x)
        # 返回格式化后的字符串表达式
        return self._print(expr2)


    # 定义一个方法，用于打印 sympy 表达式中的贝塞尔函数 Yn
    def _print_yn(self, expr):
        # 导入必要的数学函数
        from sympy.functions import sqrt, bessely
        # 从表达式中获取参数 x
        x = expr.argument
        # 计算修改后的贝塞尔函数表达式
        expr2 = sqrt(S.Pi/(2*x)) * bessely(expr.order + S.Half, x)
        # 返回格式化后的字符串表达式
        return self._print(expr2)


    # 定义一个方法，用于打印 sympy 表达式中的 Airy 函数 Ai
    def _print_airyai(self, expr):
        # 返回格式化后的字符串表达式
        return "airy(0, %s)" % self._print(expr.args[0])


    # 定义一个方法，用于打印 sympy 表达式中的 Airy 函数 Ai'
    def _print_airyaiprime(self, expr):
        # 返回格式化后的字符串表达式
        return "airy(1, %s)" % self._print(expr.args[0])


    # 定义一个方法，用于打印 sympy 表达式中的 Airy 函数 Bi
    def _print_airybi(self, expr):
        # 返回格式化后的字符串表达式
        return "airy(2, %s)" % self._print(expr.args[0])


    # 定义一个方法，用于打印 sympy 表达式中的 Airy 函数 Bi'
    def _print_airybiprime(self, expr):
        # 返回格式化后的字符串表达式
        return "airy(3, %s)" % self._print(expr.args[0])


    # 定义一个方法，用于打印 sympy 表达式中的指数积分函数
    def _print_expint(self, expr):
        # 从表达式中获取参数 mu 和 x
        mu, x = expr.args
        # 如果 mu 不等于 1，则调用未支持的表达式处理方法
        if mu != 1:
            return self._print_not_supported(expr)
        # 返回格式化后的字符串表达式
        return "expint(%s)" % self._print(x)


    # 定义一个方法，用于处理只有一个或两个参数反转顺序的 sympy 表达式
    def _one_or_two_reversed_args(self, expr):
        # 断言表达式的参数数量不超过 2
        assert len(expr.args) <= 2
        # 返回格式化后的字符串表达式
        return '{name}({args})'.format(
            name=self.known_functions[expr.__class__.__name__],
            args=", ".join([self._print(x) for x in reversed(expr.args)])
        )


    # 将 DiracDelta 和 LambertW 方法映射到 _one_or_two_reversed_args 方法
    _print_DiracDelta = _print_LambertW = _one_or_two_reversed_args


    # 定义一个方法，用于处理嵌套的二元数学函数
    def _nested_binary_math_func(self, expr):
        # 返回格式化后的字符串表达式
        return '{name}({arg1}, {arg2})'.format(
            name=self.known_functions[expr.__class__.__name__],
            arg1=self._print(expr.args[0]),
            arg2=self._print(expr.func(*expr.args[1:]))
        )


    # 将 Max 和 Min 方法映射到 _nested_binary_math_func 方法
    _print_Max = _print_Min = _nested_binary_math_func
    # 定义一个方法用于打印 Piecewise 表达式
    def _print_Piecewise(self, expr):
        # 检查最后一个条件是否为 True，否则可能导致生成的函数无法返回结果
        if expr.args[-1].cond != True:
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        
        # 初始化空列表用于存储生成的代码行
        lines = []
        
        # 如果设置为内联模式
        if self._settings["inline"]:
            # 将每个 (cond, expr) 对表达为嵌套的 Horner 形式：
            #   (condition) .* (expr) + (not cond) .* (<others>)
            # 结果为多个语句的表达式在这里不适用。
            ecpairs = ["({0}).*({1}) + (~({0})).*(".format
                       (self._print(c), self._print(e))
                       for e, c in expr.args[:-1]]
            elast = "%s" % self._print(expr.args[-1].expr)
            pw = " ...\n".join(ecpairs) + elast + ")"*len(ecpairs)
            # 注意：为了 2*pw，当前需要这些外部括号。在需要时更好地教导 parenthesize() 来为我们完成这个工作！
            return "(" + pw + ")"
        else:
            # 遍历每个 (expr, cond) 对
            for i, (e, c) in enumerate(expr.args):
                # 如果是第一个条件
                if i == 0:
                    lines.append("if (%s)" % self._print(c))
                # 如果是最后一个条件，并且条件为 True
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("elseif (%s)" % self._print(c))
                
                # 打印当前表达式
                code0 = self._print(e)
                lines.append(code0)
                
                # 如果是最后一个条件
                if i == len(expr.args) - 1:
                    lines.append("end")
            
            # 将所有行连接成一个字符串返回
            return "\n".join(lines)


    # 定义一个方法用于打印 zeta 函数表达式
    def _print_zeta(self, expr):
        # 如果参数个数为 1，返回 zeta 函数的打印形式
        if len(expr.args) == 1:
            return "zeta(%s)" % self._print(expr.args[0])
        else:
            # 对于 Matlab 的两个参数的 zeta 函数，不支持 SymPy 的等价功能
            return self._print_not_supported(expr)
    # 定义一个方法用于缩进代码，接受一个代码字符串或代码行列表作为参数
    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # 如果传入的是字符串类型的代码，则将其按行分割后递归调用自身，并拼接返回
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        # 定义一个制表符字符串
        tab = "  "
        # 定义增加缩进的正则表达式模式列表
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        # 定义减少缩进的正则表达式模式列表
        dec_regex = ('^end$', '^elseif ', '^else$')

        # 去除每行代码开头的空格或制表符
        code = [ line.lstrip(' \t') for line in code ]

        # 生成一个增加缩进的标志列表，检查每行代码是否匹配增加缩进的正则表达式
        increase = [ int(any(search(re, line) for re in inc_regex))
                     for line in code ]
        # 生成一个减少缩进的标志列表，检查每行代码是否匹配减少缩进的正则表达式
        decrease = [ int(any(search(re, line) for re in dec_regex))
                     for line in code ]

        # 初始化一个空列表用于存储格式化后的代码
        pretty = []
        # 初始化缩进级别为0
        level = 0
        # 遍历代码的每一行及其索引
        for n, line in enumerate(code):
            # 如果是空行或者只包含换行符的行，则直接添加到格式化代码列表中
            if line in ('', '\n'):
                pretty.append(line)
                continue
            # 根据减少缩进标志，减少当前缩进级别
            level -= decrease[n]
            # 添加经过适当缩进后的代码行到格式化代码列表中
            pretty.append("%s%s" % (tab*level, line))
            # 根据增加缩进标志，增加当前缩进级别
            level += increase[n]
        
        # 返回格式化后的代码列表
        return pretty
# 定义一个函数 octave_code，用于将 SymPy 表达式转换为 Octave（或 Matlab）代码字符串
def octave_code(expr, assign_to=None, **settings):
    r"""Converts `expr` to a string of Octave (or Matlab) code.

    The string uses a subset of the Octave language for Matlab compatibility.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    Examples
    ========

    >>> from sympy import octave_code, symbols, sin, pi
    >>> x = symbols('x')
    >>> octave_code(sin(x).series(x).removeO())
    'x.^5/120 - x.^3/6 + x'

    >>> from sympy import Rational, ceiling
    >>> x, y, tau = symbols("x, y, tau")
    >>> octave_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau.^(7/2)'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its very common in Octave to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> octave_code(sin(pi*x*y), assign_to="s")
    's = sin(pi*x.*y);'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> octave_code(3*pi*A**3)
    '(3*pi)*A^3'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify

    """
    """
    This function generates Octave code from a SymPy expression.

    The Octave code can include scalar symbols and matrices. When using matrices,
    the assignment can be specified either as a string or as a MatrixSymbol, but
    dimensions must align when using MatrixSymbol.

    Piecewise expressions are handled using logical masking by default. If you
    prefer if-else conditionals, pass inline=False, but ensure a default term
    "(expr, True)" exists to prevent potential evaluation errors.

    Any expression, including Piecewise, can exist inside a Matrix.

    Custom printing for specific types is supported by passing a dictionary to
    "user_functions". Each entry maps a type to either a function name or a list
    of tuples specifying conditions and corresponding function strings.

    Loops are supported through Indexed types. Set "contract=True" to generate
    looped expressions; otherwise, the assignment expression is printed.

    Parameters:
    expr : SymPy expression
        The expression to be converted to Octave code.
    assign_to : str or MatrixSymbol, optional
        Name to assign the generated code to.
    user_functions : dict, optional
        Dictionary mapping type names to custom Octave function names or lists
        of tuples for conditional function mapping.

    Returns:
    str
        Octave code representing the input SymPy expression.

    Examples:
    >>> octave_code(x**2*y*A**3)
    '(x.^2.*y)*A^3'

    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 sin(x) ceil(x)];'

    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> octave_code(pw, assign_to=tau)
    'tau = ((x > 0).*(x + 1) + (~(x > 0)).*(x));'

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 ((x > 0).*(x + 1) + (~(x > 0)).*(x)) sin(x)];'

    >>> custom_functions = {
    ...   "f": "existing_octave_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> octave_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    'existing_octave_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])'

    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> octave_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy(i) = (y(i + 1) - y(i))./(t(i + 1) - t(i));'
    """
    return OctaveCodePrinter(settings).doprint(expr, assign_to)
# 定义一个函数，用于打印给定表达式的 Octave（或 Matlab）表示形式
def print_octave_code(expr, **settings):
    """Prints the Octave (or Matlab) representation of the given expression.

    See `octave_code` for the meaning of the optional arguments.
    """
    # 调用 octave_code 函数生成表达式的 Octave 表示形式，并打印出来
    print(octave_code(expr, **settings))
```