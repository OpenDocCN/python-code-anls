# `D:\src\scipysrc\sympy\sympy\printing\maple.py`

```
"""
Maple code printer

The MapleCodePrinter converts single SymPy expressions into single
Maple expressions, using the functions defined in the Maple objects where possible.


FIXME: This module is still under actively developed. Some functions may be not completed.
"""

from sympy.core import S                                      # 导入 SymPy 的核心模块 S
from sympy.core.numbers import Integer, IntegerConstant, equal_valued   # 导入 SymPy 的数值相关模块
from sympy.printing.codeprinter import CodePrinter             # 导入 SymPy 的代码打印器
from sympy.printing.precedence import precedence, PRECEDENCE  # 导入 SymPy 的打印优先级相关模块

import sympy                                                  # 导入 sympy 库（此行不需要注释，因为仅为导入）

_known_func_same_name = (
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'sech',
    'csch', 'coth', 'exp', 'floor', 'factorial', 'bernoulli',  'euler',
    'fibonacci', 'gcd', 'lcm', 'conjugate', 'Ci', 'Chi', 'Ei', 'Li', 'Si', 'Shi',
    'erf', 'erfc', 'harmonic', 'LambertW',
    'sqrt', # For automatic rewrites
)

known_functions = {
    # SymPy -> Maple
    'Abs': 'abs',                  # 绝对值函数
    'log': 'ln',                   # 对数函数
    'asin': 'arcsin',              # 反正弦函数
    'acos': 'arccos',              # 反余弦函数
    'atan': 'arctan',              # 反正切函数
    'asec': 'arcsec',              # 反正割函数
    'acsc': 'arccsc',              # 反余割函数
    'acot': 'arccot',              # 反余切函数
    'asinh': 'arcsinh',            # 反双曲正弦函数
    'acosh': 'arccosh',            # 反双曲余弦函数
    'atanh': 'arctanh',            # 反双曲正切函数
    'asech': 'arcsech',            # 反双曲正割函数
    'acsch': 'arccsch',            # 反双曲余割函数
    'acoth': 'arccoth',            # 反双曲余切函数
    'ceiling': 'ceil',             # 天花板函数
    'Max' : 'max',                 # 最大值函数
    'Min' : 'min',                 # 最小值函数

    'factorial2': 'doublefactorial',   # 双阶乘函数
    'RisingFactorial': 'pochhammer',   # 上升阶乘幂函数
    'besseli': 'BesselI',              # 贝塞尔函数I
    'besselj': 'BesselJ',              # 贝塞尔函数J
    'besselk': 'BesselK',              # 贝塞尔函数K
    'bessely': 'BesselY',              # 贝塞尔函数Y
    'hankelh1': 'HankelH1',            # 汉克尔函数H1
    'hankelh2': 'HankelH2',            # 汉克尔函数H2
    'airyai': 'AiryAi',                # 艾里函数Ai
    'airybi': 'AiryBi',                # 艾里函数Bi
    'appellf1': 'AppellF1',            # 阿贝尔函数F1
    'fresnelc': 'FresnelC',            # 菲涅尔余弦积分函数
    'fresnels': 'FresnelS',            # 菲涅尔正弦积分函数
    'lerchphi' : 'LerchPhi',           # 勒让德函数Phi
}

for _func in _known_func_same_name:
    known_functions[_func] = _func      # 将已知函数列表中的函数名映射为其本身，以便 SymPy -> Maple 的转换

number_symbols = {
    # SymPy -> Maple
    S.Pi: 'Pi',                             # 圆周率符号
    S.Exp1: 'exp(1)',                        # 自然对数的底 e
    S.Catalan: 'Catalan',                    # Catalan 常数
    S.EulerGamma: 'gamma',                   # 欧拉-马歇罗尼常数
    S.GoldenRatio: '(1/2 + (1/2)*sqrt(5))'   # 黄金比例
}

spec_relational_ops = {
    # SymPy -> Maple
    '==': '=',      # 等于运算符
    '!=': '<>',     # 不等于运算符
}

not_supported_symbol = [
    S.ComplexInfinity   # 复无穷符号，在 Maple 中不支持
]

class MapleCodePrinter(CodePrinter):
    """
    Printer which converts a SymPy expression into a maple code.
    """
    printmethod = "_maple"   # 打印方法名称
    language = "maple"       # 打印目标语言名称

    _operators = {
        'and': 'and',       # 逻辑与运算符
        'or': 'or',         # 逻辑或运算符
        'not': 'not ',      # 逻辑非运算符
    }

    _default_settings = dict(CodePrinter._default_settings, **{
        'inline': True,                                 # 默认设置为内联模式
        'allow_unknown_functions': True,                # 允许使用未知函数
    })

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        super().__init__(settings)
        self.known_functions = dict(known_functions)    # 初始化已知函数映射表
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)          # 更新用户定义的函数映射表

    def _get_statement(self, codestring):
        return "%s;" % codestring                      # 生成带有分号的语句字符串

    def _get_comment(self, text):
        return "# {}".format(text)                     # 生成注释字符串
    def _declare_number_const(self, name, value):
        # 返回一个字符串，格式为 "{name} := {value}"，其中 value 是根据给定精度计算后的数值
        return "{} := {};".format(name,
                                    value.evalf(self._settings['precision']))

    def _format_code(self, lines):
        # 直接返回传入的 lines 参数，不做任何格式化处理
        return lines

    def _print_tuple(self, expr):
        # 将元组 expr 转换为列表，并调用 _print 方法打印列表的字符串表示
        return self._print(list(expr))

    def _print_Tuple(self, expr):
        # 将元组 expr 转换为列表，并调用 _print 方法打印列表的字符串表示
        return self._print(list(expr))

    def _print_Assignment(self, expr):
        # 将赋值表达式 expr 拆分为左值和右值，并返回格式化后的赋值字符串
        lhs = self._print(expr.lhs)  # 打印左值
        rhs = self._print(expr.rhs)  # 打印右值
        return "{lhs} := {rhs}".format(lhs=lhs, rhs=rhs)

    def _print_Pow(self, expr, **kwargs):
        # 根据幂指数的不同情况返回对应的字符串表示
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1/%s' % (self.parenthesize(expr.base, PREC))  # 处理幂指数为 -1 的情况
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)  # 处理幂指数为 0.5 的情况
        elif equal_valued(expr.exp, -0.5):
            return '1/sqrt(%s)' % self._print(expr.base)  # 处理幂指数为 -0.5 的情况
        else:
            return '{base}^{exp}'.format(  # 默认情况下返回幂指数表达式
                base=self.parenthesize(expr.base, PREC),
                exp=self.parenthesize(expr.exp, PREC))

    def _print_Piecewise(self, expr):
        # 打印 Piecewise 表达式的字符串表示
        if (expr.args[-1].cond is not True) and (expr.args[-1].cond != S.BooleanTrue):
            # 检查最后一个条件是否为 True，否则抛出异常
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        _coup_list = [
            ("{c}, {e}".format(c=self._print(c),
                               e=self._print(e)) if c is not True and c is not S.BooleanTrue else "{e}".format(
                e=self._print(e)))
            for e, c in expr.args]  # 遍历 Piecewise 表达式的每个分支，并打印条件和结果
        _inbrace = ', '.join(_coup_list)  # 将所有分支组成一个字符串
        return 'piecewise({_inbrace})'.format(_inbrace=_inbrace)  # 返回格式化后的 Piecewise 表达式字符串

    def _print_Rational(self, expr):
        # 打印有理数表达式的字符串表示
        p, q = int(expr.p), int(expr.q)
        return "{p}/{q}".format(p=str(p), q=str(q))  # 返回有理数的分子和分母的字符串表示

    def _print_Relational(self, expr):
        # 打印关系表达式的字符串表示
        PREC=precedence(expr)
        lhs_code = self.parenthesize(expr.lhs, PREC)  # 打印左操作数
        rhs_code = self.parenthesize(expr.rhs, PREC)  # 打印右操作数
        op = expr.rel_op
        if op in spec_relational_ops:
            op = spec_relational_ops[op]  # 替换特殊的关系运算符
        return "{lhs} {rel_op} {rhs}".format(lhs=lhs_code, rel_op=op, rhs=rhs_code)  # 返回格式化后的关系表达式字符串

    def _print_NumberSymbol(self, expr):
        # 打印数值符号表达式的字符串表示
        return number_symbols[expr]  # 返回数值符号的字符串表示

    def _print_NegativeInfinity(self, expr):
        # 打印负无穷大的字符串表示
        return '-infinity'

    def _print_Infinity(self, expr):
        # 打印正无穷大的字符串表示
        return 'infinity'

    def _print_Idx(self, expr):
        # 打印索引表达式的字符串表示
        return self._print(expr.label)  # 返回索引的字符串表示

    def _print_BooleanTrue(self, expr):
        # 打印布尔真值的字符串表示
        return "true"

    def _print_BooleanFalse(self, expr):
        # 打印布尔假值的字符串表示
        return "false"
    # 返回布尔表达式的字符串表示：如果表达式为真返回 'true'，否则返回 'false'
    def _print_bool(self, expr):
        return 'true' if expr else 'false'

    # 返回 NaN 的字符串表示：始终返回 'undefined'
    def _print_NaN(self, expr):
        return 'undefined'

    # 返回矩阵的字符串表示：根据稀疏矩阵选项确定存储方式
    def _get_matrix(self, expr, sparse=False):
        # 如果矩阵的形状包含零，则返回空矩阵的字符串表示
        if S.Zero in expr.shape:
            _strM = 'Matrix([], storage = {storage})'.format(
                storage='sparse' if sparse else 'rectangular')
        else:
            # 否则返回具有数据列表的矩阵的字符串表示
            _strM = 'Matrix({list}, storage = {storage})'.format(
                list=self._print(expr.tolist()),
                storage='sparse' if sparse else 'rectangular')
        return _strM

    # 返回矩阵元素的字符串表示：格式为 "矩阵名[i+1, j+1]"
    def _print_MatrixElement(self, expr):
        return "{parent}[{i_maple}, {j_maple}]".format(
            parent=self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True),
            i_maple=self._print(expr.i + 1),
            j_maple=self._print(expr.j + 1))

    # 返回矩阵基类的字符串表示：默认返回非稀疏矩阵的字符串表示
    def _print_MatrixBase(self, expr):
        return self._get_matrix(expr, sparse=False)

    # 返回稀疏矩阵的字符串表示
    def _print_SparseRepMatrix(self, expr):
        return self._get_matrix(expr, sparse=True)

    # 返回单位矩阵的字符串表示：如果行数是整数，则返回对应大小的稀疏矩阵的字符串表示
    def _print_Identity(self, expr):
        if isinstance(expr.rows, (Integer, IntegerConstant)):
            return self._print(sympy.SparseMatrix(expr))
        else:
            return "Matrix({var_size}, shape = identity)".format(var_size=self._print(expr.rows))

    # 返回矩阵乘法的字符串表示
    def _print_MatMul(self, expr):
        PREC=precedence(expr)
        _fact_list = list(expr.args)
        _const = None
        if not isinstance(_fact_list[0], (sympy.MatrixBase, sympy.MatrixExpr,
                                          sympy.MatrixSlice, sympy.MatrixSymbol)):
            _const, _fact_list = _fact_list[0], _fact_list[1:]

        if _const is None or _const == 1:
            return '.'.join(self.parenthesize(_m, PREC) for _m in _fact_list)
        else:
            return '{c}*{m}'.format(c=_const, m='.'.join(self.parenthesize(_m, PREC) for _m in _fact_list))

    # 返回矩阵幂的字符串表示
    def _print_MatPow(self, expr):
        # 此函数需要在Maple中使用线性代数函数
        return 'MatrixPower({A}, {n})'.format(A=self._print(expr.base), n=self._print(expr.exp))

    # 返回Hadamard积的字符串表示
    def _print_HadamardProduct(self, expr):
        PREC = precedence(expr)
        _fact_list = list(expr.args)
        return '*'.join(self.parenthesize(_m, PREC) for _m in _fact_list)

    # 返回导数运算的字符串表示
    def _print_Derivative(self, expr):
        _f, (_var, _order) = expr.args

        if _order != 1:
            _second_arg = '{var}${order}'.format(var=self._print(_var),
                                                 order=self._print(_order))
        else:
            _second_arg = '{var}'.format(var=self._print(_var))
        return 'diff({func_expr}, {sec_arg})'.format(func_expr=self._print(_f), sec_arg=_second_arg)
# 将 SymPy 表达式 `expr` 转换为 Maple 代码字符串的函数
def maple_code(expr, assign_to=None, **settings):
    r"""Converts ``expr`` to a string of Maple code.

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

    """
    # 创建一个 MapleCodePrinter 对象，使用指定的设置打印 Maple 代码
    return MapleCodePrinter(settings).doprint(expr, assign_to)


# 打印给定表达式的 Maple 表示
def print_maple_code(expr, **settings):
    """Prints the Maple representation of the given expression.

    See :func:`maple_code` for the meaning of the optional arguments.

    Examples
    ========

    >>> from sympy import print_maple_code, symbols
    >>> x, y = symbols('x y')
    >>> print_maple_code(x, assign_to=y)
    y := x
    """
    # 调用 maple_code 函数生成 Maple 代码，并打印结果
    print(maple_code(expr, **settings))
```