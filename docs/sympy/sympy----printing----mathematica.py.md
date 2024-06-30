# `D:\src\scipysrc\sympy\sympy\printing\mathematica.py`

```
"""
Mathematica code printer
"""

# 引入未来的注释导入模块，用于支持类型注解
from __future__ import annotations
# 引入类型提示中的Any类型
from typing import Any

# 从sympy.core模块导入基本对象、表达式和浮点数
from sympy.core import Basic, Expr, Float
# 从sympy.core.sorting模块导入默认排序键函数
from sympy.core.sorting import default_sort_key

# 从sympy.printing.codeprinter模块导入代码打印器类
from sympy.printing.codeprinter import CodePrinter
# 从sympy.printing.precedence模块导入优先级函数
from sympy.printing.precedence import precedence

# 在MCodePrinter._print_Function(self)方法中使用的已知函数和对应的打印名称
known_functions = {
    "exp": [(lambda x: True, "Exp")],  # 指数函数
    "log": [(lambda x: True, "Log")],  # 对数函数
    "sin": [(lambda x: True, "Sin")],  # 正弦函数
    "cos": [(lambda x: True, "Cos")],  # 余弦函数
    "tan": [(lambda x: True, "Tan")],  # 正切函数
    "cot": [(lambda x: True, "Cot")],  # 余切函数
    "sec": [(lambda x: True, "Sec")],  # 正割函数
    "csc": [(lambda x: True, "Csc")],  # 余割函数
    "asin": [(lambda x: True, "ArcSin")],  # 反正弦函数
    "acos": [(lambda x: True, "ArcCos")],  # 反余弦函数
    "atan": [(lambda x: True, "ArcTan")],  # 反正切函数
    "acot": [(lambda x: True, "ArcCot")],  # 反余切函数
    "asec": [(lambda x: True, "ArcSec")],  # 反正割函数
    "acsc": [(lambda x: True, "ArcCsc")],  # 反余割函数
    "atan2": [(lambda *x: True, "ArcTan")],  # 二参数反正切函数
    "sinh": [(lambda x: True, "Sinh")],  # 双曲正弦函数
    "cosh": [(lambda x: True, "Cosh")],  # 双曲余弦函数
    "tanh": [(lambda x: True, "Tanh")],  # 双曲正切函数
    "coth": [(lambda x: True, "Coth")],  # 双曲余切函数
    "sech": [(lambda x: True, "Sech")],  # 双曲正割函数
    "csch": [(lambda x: True, "Csch")],  # 双曲余割函数
    "asinh": [(lambda x: True, "ArcSinh")],  # 反双曲正弦函数
    "acosh": [(lambda x: True, "ArcCosh")],  # 反双曲余弦函数
    "atanh": [(lambda x: True, "ArcTanh")],  # 反双曲正切函数
    "acoth": [(lambda x: True, "ArcCoth")],  # 反双曲余切函数
    "asech": [(lambda x: True, "ArcSech")],  # 反双曲正割函数
    "acsch": [(lambda x: True, "ArcCsch")],  # 反双曲余割函数
    "sinc": [(lambda x: True, "Sinc")],  # 同步正弦函数
    "conjugate": [(lambda x: True, "Conjugate")],  # 复共轭函数
    "Max": [(lambda *x: True, "Max")],  # 最大值函数
    "Min": [(lambda *x: True, "Min")],  # 最小值函数
    "erf": [(lambda x: True, "Erf")],  # 误差函数
    "erf2": [(lambda *x: True, "Erf")],  # 第二种误差函数
    "erfc": [(lambda x: True, "Erfc")],  # 余误差函数
    "erfi": [(lambda x: True, "Erfi")],  # 超越误差函数
    "erfinv": [(lambda x: True, "InverseErf")],  # 误差函数的反函数
    "erfcinv": [(lambda x: True, "InverseErfc")],  # 余误差函数的反函数
    "erf2inv": [(lambda *x: True, "InverseErf")],  # 第二种误差函数的反函数
    "expint": [(lambda *x: True, "ExpIntegralE")],  # 指数积分E函数
    "Ei": [(lambda x: True, "ExpIntegralEi")],  # 指数积分Ei函数
    "fresnelc": [(lambda x: True, "FresnelC")],  # 弗雷内尔余弦积分函数
    "fresnels": [(lambda x: True, "FresnelS")],  # 弗雷内尔正弦积分函数
    "gamma": [(lambda x: True, "Gamma")],  # Γ函数
    "uppergamma": [(lambda *x: True, "Gamma")],  # 上不完全Γ函数
    "polygamma": [(lambda *x: True, "PolyGamma")],  # 多Γ函数
    "loggamma": [(lambda x: True, "LogGamma")],  # 对数Γ函数
    "beta": [(lambda *x: True, "Beta")],  # β函数
    "Ci": [(lambda x: True, "CosIntegral")],  # 余弦积分函数
    "Si": [(lambda x: True, "SinIntegral")],  # 正弦积分函数
    "Chi": [(lambda x: True, "CoshIntegral")],  # 双曲余弦积分函数
    "Shi": [(lambda x: True, "SinhIntegral")],  # 双曲正弦积分函数
    "li": [(lambda x: True, "LogIntegral")],  # 对数积分函数
    "factorial": [(lambda x: True, "Factorial")],  # 阶乘函数
    "factorial2": [(lambda x: True, "Factorial2")],  # 双阶乘函数
    "subfactorial": [(lambda x: True, "Subfactorial")],  # 非负整数的排列函数
    "catalan": [(lambda x: True, "CatalanNumber")],  # 卡塔兰数函数
    "harmonic": [(lambda *x: True, "HarmonicNumber")],  # 谐波数函数
    "lucas": [(lambda x: True, "LucasL")],  # 卢卡斯数函数
    "RisingFactorial": [(lambda *x: True, "Pochhammer")],  # 升幂多项式函数
    # 定义了一系列数学函数的映射关系，每个函数名对应一个列表，列表包含一个 lambda 函数和一个字符串
    "FallingFactorial": [(lambda *x: True, "FactorialPower")],
    "laguerre": [(lambda *x: True, "LaguerreL")],
    "assoc_laguerre": [(lambda *x: True, "LaguerreL")],
    "hermite": [(lambda *x: True, "HermiteH")],
    "jacobi": [(lambda *x: True, "JacobiP")],
    "gegenbauer": [(lambda *x: True, "GegenbauerC")],
    "chebyshevt": [(lambda *x: True, "ChebyshevT")],
    "chebyshevu": [(lambda *x: True, "ChebyshevU")],
    "legendre": [(lambda *x: True, "LegendreP")],
    "assoc_legendre": [(lambda *x: True, "LegendreP")],
    "mathieuc": [(lambda *x: True, "MathieuC")],
    "mathieus": [(lambda *x: True, "MathieuS")],
    "mathieucprime": [(lambda *x: True, "MathieuCPrime")],
    "mathieusprime": [(lambda *x: True, "MathieuSPrime")],
    "stieltjes": [(lambda x: True, "StieltjesGamma")],
    "elliptic_e": [(lambda *x: True, "EllipticE")],
    "elliptic_f": [(lambda *x: True, "EllipticE")],
    "elliptic_k": [(lambda x: True, "EllipticK")],
    "elliptic_pi": [(lambda *x: True, "EllipticPi")],
    "zeta": [(lambda *x: True, "Zeta")],
    "dirichlet_eta": [(lambda x: True, "DirichletEta")],
    "riemann_xi": [(lambda x: True, "RiemannXi")],
    "besseli": [(lambda *x: True, "BesselI")],
    "besselj": [(lambda *x: True, "BesselJ")],
    "besselk": [(lambda *x: True, "BesselK")],
    "bessely": [(lambda *x: True, "BesselY")],
    "hankel1": [(lambda *x: True, "HankelH1")],
    "hankel2": [(lambda *x: True, "HankelH2")],
    "airyai": [(lambda x: True, "AiryAi")],
    "airybi": [(lambda x: True, "AiryBi")],
    "airyaiprime": [(lambda x: True, "AiryAiPrime")],
    "airybiprime": [(lambda x: True, "AiryBiPrime")],
    "polylog": [(lambda *x: True, "PolyLog")],
    "lerchphi": [(lambda *x: True, "LerchPhi")],
    "gcd": [(lambda *x: True, "GCD")],
    "lcm": [(lambda *x: True, "LCM")],
    "jn": [(lambda *x: True, "SphericalBesselJ")],
    "yn": [(lambda *x: True, "SphericalBesselY")],
    "hyper": [(lambda *x: True, "HypergeometricPFQ")],
    "meijerg": [(lambda *x: True, "MeijerG")],
    "appellf1": [(lambda *x: True, "AppellF1")],
    "DiracDelta": [(lambda x: True, "DiracDelta")],
    "Heaviside": [(lambda x: True, "HeavisideTheta")],
    "KroneckerDelta": [(lambda *x: True, "KroneckerDelta")],
    "sqrt": [(lambda x: True, "Sqrt")],  # 用于自动重写
}

# 定义一个名为 MCodePrinter 的类，继承自 CodePrinter 类
class MCodePrinter(CodePrinter):
    """A printer to convert Python expressions to
    strings of the Wolfram's Mathematica code
    """
    
    # 类变量，用于指定打印方法和语言类型
    printmethod = "_mcode"
    language = "Wolfram Language"

    # 默认设置，包括精度和用户自定义函数等
    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 15,
        'user_functions': {},
    })

    # 存储数学表达式中的数字符号
    _number_symbols: set[tuple[Expr, Float]] = set()
    # 存储不支持的表达式
    _not_supported: set[Basic] = set()

    def __init__(self, settings={}):
        """Register function mappings supplied by user"""
        # 调用父类的构造函数
        CodePrinter.__init__(self, settings)
        # 初始化已知函数的字典
        self.known_functions = dict(known_functions)
        # 复制用户提供的函数映射
        userfuncs = settings.get('user_functions', {}).copy()
        # 对于每个用户函数，如果其值不是列表，则创建一个仅接受任意参数的 lambda 函数
        for k, v in userfuncs.items():
            if not isinstance(v, list):
                userfuncs[k] = [(lambda *x: True, v)]
        # 更新已知函数字典
        self.known_functions.update(userfuncs)

    def _format_code(self, lines):
        return lines

    def _print_Pow(self, expr):
        # 获取幂运算的优先级
        PREC = precedence(expr)
        # 格式化幂运算表达式
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    def _print_Mul(self, expr):
        # 获取乘法运算的优先级
        PREC = precedence(expr)
        # 将表达式中的可组合项 (c, nc) 分离
        c, nc = expr.args_cnc()
        # 调用父类的打印乘法运算方法
        res = super()._print_Mul(expr.func(*c))
        # 如果存在非可组合项，则将它们用 '**' 连接后添加到结果中
        if nc:
            res += '*'
            res += '**'.join(self.parenthesize(a, PREC) for a in nc)
        return res

    def _print_Relational(self, expr):
        # 打印关系表达式，包括左操作数、关系操作符和右操作数
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    # 下面是一些基本数值的打印方法

    def _print_Zero(self, expr):
        return '0'

    def _print_One(self, expr):
        return '1'

    def _print_NegativeOne(self, expr):
        return '-1'

    def _print_Half(self, expr):
        return '1/2'

    def _print_ImaginaryUnit(self, expr):
        return 'I'

    # 下面是一些特殊数值的打印方法

    def _print_Infinity(self, expr):
        return 'Infinity'

    def _print_NegativeInfinity(self, expr):
        return '-Infinity'

    def _print_ComplexInfinity(self, expr):
        return 'ComplexInfinity'

    def _print_NaN(self, expr):
        return 'Indeterminate'

    # 下面是一些数学常数的打印方法

    def _print_Exp1(self, expr):
        return 'E'

    def _print_Pi(self, expr):
        return 'Pi'

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_TribonacciConstant(self, expr):
        # 获取扩展后的特定数学常数表达式，并根据其优先级格式化输出
        expanded = expr.expand(func=True)
        PREC = precedence(expr)
        return self.parenthesize(expanded, PREC)

    def _print_EulerGamma(self, expr):
        return 'EulerGamma'

    def _print_Catalan(self, expr):
        return 'Catalan'

    # 下面是一些集合和矩阵的打印方法

    def _print_list(self, expr):
        # 打印列表表达式
        return '{' + ', '.join(self.doprint(a) for a in expr) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _print_ImmutableDenseMatrix(self, expr):
        # 打印不可变稠密矩阵表达式
        return self.doprint(expr.tolist())
    def _print_ImmutableSparseMatrix(self, expr):
        # 定义内部函数，用于打印每个非零元素的位置和值的规则
        def print_rule(pos, val):
            return '{} -> {}'.format(
            self.doprint((pos[0]+1, pos[1]+1)), self.doprint(val))

        # 定义内部函数，打印稀疏矩阵的数据部分
        def print_data():
            # 获取稀疏矩阵的非零元素并按默认排序键排序
            items = sorted(expr.todok().items(), key=default_sort_key)
            # 将非零元素格式化为 Mathematica 风格的字符串
            return '{' + \
                ', '.join(print_rule(k, v) for k, v in items) + \
                '}'

        # 定义内部函数，打印稀疏矩阵的维度信息
        def print_dims():
            return self.doprint(expr.shape)

        # 返回稀疏矩阵的 Mathematica 表示字符串
        return 'SparseArray[{}, {}]'.format(print_data(), print_dims())

    def _print_ImmutableDenseNDimArray(self, expr):
        # 直接打印密集多维数组的列表形式
        return self.doprint(expr.tolist())

    def _print_ImmutableSparseNDimArray(self, expr):
        # 定义内部函数，用于打印字符串列表
        def print_string_list(string_list):
            return '{' + ', '.join(a for a in string_list) + '}'

        # 定义内部函数，将 Python 风格的索引转换为 Mathematica 风格的索引
        def to_mathematica_index(*args):
            """Helper function to change Python style indexing to
            Pathematica indexing.

            Python indexing (0, 1 ... n-1)
            -> Mathematica indexing (1, 2 ... n)
            """
            return tuple(i + 1 for i in args)

        # 定义内部函数，用于打印每个规则的字符串
        def print_rule(pos, val):
            """Helper function to print a rule of Mathematica"""
            return '{} -> {}'.format(self.doprint(pos), self.doprint(val))

        # 定义内部函数，打印稀疏多维数组的数据部分
        def print_data():
            """Helper function to print data part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html

            ``data`` must be formatted with rule.
            """
            # 获取稀疏多维数组的非零元素并按索引排序，生成 Mathematica 风格的字符串列表
            return print_string_list(
                [print_rule(
                    to_mathematica_index(*(expr._get_tuple_index(key))),
                    value)
                for key, value in sorted(expr._sparse_array.items())]
            )

        # 定义内部函数，打印稀疏多维数组的维度信息
        def print_dims():
            """Helper function to print dimensions part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html
            """
            return self.doprint(expr.shape)

        # 返回稀疏多维数组的 Mathematica 表示字符串
        return 'SparseArray[{}, {}]'.format(print_data(), print_dims())
    # 定义一个方法来打印表达式中的函数，根据表达式的类型和已知函数列表进行处理
    def _print_Function(self, expr):
        # 如果函数名在已知函数列表中
        if expr.func.__name__ in self.known_functions:
            # 获取条件-方法映射列表
            cond_mfunc = self.known_functions[expr.func.__name__]
            # 遍历条件-方法映射
            for cond, mfunc in cond_mfunc:
                # 如果表达式符合条件
                if cond(*expr.args):
                    # 返回格式化后的方法调用字符串
                    return "%s[%s]" % (mfunc, self.stringify(expr.args, ", "))
        # 如果函数名在可重写函数列表中
        elif expr.func.__name__ in self._rewriteable_functions:
            # 简单地重写为支持的函数
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            # 如果目标函数及其所需函数都可以打印
            if self._can_print(target_f) and all(self._can_print(f) for f in required_fs):
                # 返回重写后的表达式打印结果
                return self._print(expr.rewrite(target_f))
        # 否则返回函数名及其参数的字符串表示
        return expr.func.__name__ + "[%s]" % self.stringify(expr.args, ", ")

    # 将 _print_Function 方法绑定到 _print_MinMaxBase 上
    _print_MinMaxBase = _print_Function

    # 定义一个方法来打印 LambertW 函数表达式
    def _print_LambertW(self, expr):
        # 如果参数个数为1
        if len(expr.args) == 1:
            # 返回单参数情况下的打印格式
            return "ProductLog[{}]".format(self._print(expr.args[0]))
        # 否则返回双参数情况下的打印格式
        return "ProductLog[{}, {}]".format(
            self._print(expr.args[1]), self._print(expr.args[0]))

    # 定义一个方法来打印 Integral 积分表达式
    def _print_Integral(self, expr):
        # 如果只有一个积分变量且无积分上下限
        if len(expr.variables) == 1 and not expr.limits[0][1:]:
            # 构造积分表达式的参数列表
            args = [expr.args[0], expr.variables[0]]
        else:
            # 否则使用原始参数列表
            args = expr.args
        # 返回积分表达式的打印格式
        return "Hold[Integrate[" + ', '.join(self.doprint(a) for a in args) + "]]"

    # 定义一个方法来打印 Sum 求和表达式
    def _print_Sum(self, expr):
        # 返回求和表达式的打印格式
        return "Hold[Sum[" + ', '.join(self.doprint(a) for a in expr.args) + "]]"

    # 定义一个方法来打印 Derivative 导数表达式
    def _print_Derivative(self, expr):
        # 提取导数表达式及其变量
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        # 返回导数表达式的打印格式
        return "Hold[D[" + ', '.join(self.doprint(a) for a in [dexpr] + dvars) + "]]"

    # 定义一个方法，根据文本生成一个注释字符串
    def _get_comment(self, text):
        return "(* {} *)".format(text)
# 定义一个函数 mathematica_code，用于将表达式转换为 Wolfram Mathematica 代码的字符串
def mathematica_code(expr, **settings):
    # 文档字符串，描述了函数的作用和使用示例
    r"""Converts an expr to a string of the Wolfram Mathematica code

    Examples
    ========

    >>> from sympy import mathematica_code as mcode, symbols, sin
    >>> x = symbols('x')
    >>> mcode(sin(x).series(x).removeO())
    '(1/120)*x^5 - 1/6*x^3 + x'
    """
    # 调用 MCodePrinter 类的 doprint 方法，将输入的表达式 expr 转换为 Mathematica 代码的字符串并返回
    return MCodePrinter(settings).doprint(expr)
```