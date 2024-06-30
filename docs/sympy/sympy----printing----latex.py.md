# `D:\src\scipysrc\sympy\sympy\printing\latex.py`

```
"""
A Printer which converts an expression into its LaTeX equivalent.
"""
# 导入必要的模块和类
from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import itertools

# 导入 SymPy 相关模块和类
from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse

# sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE

# 导入 mpmath 库中的特定函数
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str

# 导入 SymPy 中的实用工具函数
from sympy.utilities.iterables import has_variety, sift

# 导入正则表达式模块
import re

# 如果正在进行类型检查，则导入特定类型
if TYPE_CHECKING:
    from sympy.tensor.array import NDimArray
    from sympy.vector.basisdependent import BasisDependent

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at
# https://docs.mathjax.org/en/latest/tex.html#supported-latex-commands
# This variable only contains those functions which SymPy uses.
# 定义可以直接在 LaTeX 和 MathJax 中使用的函数列表
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                            'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec',
                            'csc', 'cot', 'coth', 're', 'im', 'frac', 'root',
                            'arg',
                            ]

# 定义希腊字母的 LaTeX 对照表
tex_greek_dictionary = {
    'Alpha': r'\mathrm{A}',
    'Beta': r'\mathrm{B}',
    'Gamma': r'\Gamma',
    'Delta': r'\Delta',
    'Epsilon': r'\mathrm{E}',
    'Zeta': r'\mathrm{Z}',
    'Eta': r'\mathrm{H}',
    'Theta': r'\Theta',
    'Iota': r'\mathrm{I}',
    'Kappa': r'\mathrm{K}',
    'Lambda': r'\Lambda',
    'Mu': r'\mathrm{M}',
    'Nu': r'\mathrm{N}',
    'Xi': r'\Xi',
    'omicron': 'o',
    'Omicron': r'\mathrm{O}',
    'Pi': r'\Pi',
    'Rho': r'\mathrm{P}',
    'Sigma': r'\Sigma',
    'Tau': r'\mathrm{T}',
    'Upsilon': r'\Upsilon',
    'Phi': r'\Phi',
    'Chi': r'\mathrm{X}',
    'Psi': r'\Psi',
    'Omega': r'\Omega',
    'lamda': r'\lambda',
    'Lamda': r'\Lambda',
    'khi': r'\chi',
    'Khi': r'\mathrm{X}',
    'varepsilon': r'\varepsilon',
    'varkappa': r'\varkappa',
    'varphi': r'\varphi',
    'varpi': r'\varpi',
    'varrho': r'\varrho',
    'varsigma': r'\varsigma',
    'vartheta': r'\vartheta',
}

# 定义其他符号集合
other_symbols = {'aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar',
                 'hslash', 'mho', 'wp'}

# 定义变量名修饰符字典
modifier_dict: dict[str, Callable[[str], str]] = {
    # Accents
    'mathring': lambda s: r'\mathring{'+s+r'}',
    'ddddot': lambda s: r'\ddddot{'+s+r'}',
    'dddot': lambda s: r'\dddot{'+s+r'}',
}
    # 定义 LaTeX 中常用的数学符号和格式化函数
    'ddot': lambda s: r'\ddot{'+s+r'}',   # 返回 s 的双点标记形式
    'dot': lambda s: r'\dot{'+s+r'}',     # 返回 s 的点标记形式
    'check': lambda s: r'\check{'+s+r'}', # 返回 s 的对勾标记形式
    'breve': lambda s: r'\breve{'+s+r'}', # 返回 s 的短弧标记形式
    'acute': lambda s: r'\acute{'+s+r'}', # 返回 s 的锐音标记形式
    'grave': lambda s: r'\grave{'+s+r'}', # 返回 s 的重音标记形式
    'tilde': lambda s: r'\tilde{'+s+r'}', # 返回 s 的波浪线标记形式
    'hat': lambda s: r'\hat{'+s+r'}',     # 返回 s 的帽子标记形式
    'bar': lambda s: r'\bar{'+s+r'}',     # 返回 s 的长线标记形式
    'vec': lambda s: r'\vec{'+s+r'}',     # 返回 s 的向量标记形式
    'prime': lambda s: "{"+s+"}'",         # 返回 s 的一阶导数标记形式
    'prm': lambda s: "{"+s+"}'",           # 返回 s 的一阶导数标记形式

    # 符号风格
    'bold': lambda s: r'\boldsymbol{'+s+r'}', # 返回 s 的粗体符号形式
    'bm': lambda s: r'\boldsymbol{'+s+r'}',   # 返回 s 的粗体符号形式
    'cal': lambda s: r'\mathcal{'+s+r'}',     # 返回 s 的花体符号形式
    'scr': lambda s: r'\mathscr{'+s+r'}',     # 返回 s 的手写体符号形式
    'frak': lambda s: r'\mathfrak{'+s+r'}',   # 返回 s 的黑体符号形式

    # 括号
    'norm': lambda s: r'\left\|{'+s+r'}\right\|',       # 返回 s 的范数形式
    'avg': lambda s: r'\left\langle{'+s+r'}\right\rangle',  # 返回 s 的平均值形式
    'abs': lambda s: r'\left|{'+s+r'}\right|',           # 返回 s 的绝对值形式
    'mag': lambda s: r'\left|{'+s+r'}\right|',           # 返回 s 的大小形式
}

# 定义一个空的代码块，用于示例目的。在实际代码中，此处可能有更多的代码。

greek_letters_set = frozenset(greeks)

# 将 greeks 转换为不可变集合，并赋值给 greek_letters_set

_between_two_numbers_p = (
    re.compile(r'[0-9][} ]*$'),  # 匹配以数字结尾的字符
    re.compile(r'(\d|\\frac{\d+}{\d+})'),  # 匹配数字或 LaTeX 分数表达式
)

# _between_two_numbers_p 是一个包含两个正则表达式对象的元组：
#   - 第一个正则表达式用于搜索以数字结尾的字符序列
#   - 第二个正则表达式用于匹配数字或 LaTeX 分数表达式

def latex_escape(s: str) -> str:
    """
    Escape a string such that latex interprets it as plaintext.

    We cannot use verbatim easily with mathjax, so escaping is easier.
    Rules from https://tex.stackexchange.com/a/34586/41112.
    """
    s = s.replace('\\', r'\textbackslash')
    # 将反斜杠转义为 LaTeX 中的文本反斜杠
    for c in '&%$#_{}':
        s = s.replace(c, '\\' + c)
    # 对 &%$#_{} 进行转义，以便 LaTeX 正确解释
    s = s.replace('~', r'\textasciitilde')
    # 将波浪号转义为 LaTeX 中的波浪线
    s = s.replace('^', r'\textasciicircum')
    # 将插入符号转义为 LaTeX 中的插入符
    return s
    # 返回转义后的字符串

class LatexPrinter(Printer):
    printmethod = "_latex"

    _default_settings: dict[str, Any] = {
        "full_prec": False,
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "itex": False,
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_str": None,
        "mode": "plain",
        "mul_symbol": None,
        "order": None,
        "symbol_names": {},
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "gothic_re_im": False,
        "decimal_separator": "period",
        "perm_cyclic": True,
        "parenthesize_super": True,
        "min": None,
        "max": None,
        "diff_operator": "d",
        "adjoint_style": "dagger",
    }

# LatexPrinter 类，继承自 Printer 类，定义了 _latex 打印方法。
# _default_settings 是 LatexPrinter 类的类属性，包含了一系列默认设置。
# 这些设置用于控制 LaTeX 打印输出的格式和样式。
    # 初始化函数，接受一个可选的设置参数 `settings`
    def __init__(self, settings=None):
        # 调用父类 Printer 的初始化函数，传入设置参数 `settings`
        Printer.__init__(self, settings)

        # 检查设置参数中是否包含 'mode' 键
        if 'mode' in self._settings:
            # 如果包含 'mode' 键，则定义有效的模式列表
            valid_modes = ['inline', 'plain', 'equation', 'equation*']
            # 如果设置中的 'mode' 不在有效模式列表中，则抛出数值错误异常
            if self._settings['mode'] not in valid_modes:
                raise ValueError("'mode' must be one of 'inline', 'plain', "
                                 "'equation' or 'equation*'")

        # 如果设置中的 'fold_short_frac' 是 None 并且 'mode' 是 'inline'，则将 'fold_short_frac' 设置为 True
        if self._settings['fold_short_frac'] is None and \
                self._settings['mode'] == 'inline':
            self._settings['fold_short_frac'] = True

        # 定义乘号符号表，根据设置中的 'mul_symbol' 来选择 LaTeX 表示
        mul_symbol_table = {
            None: r" ",
            "ldot": r" \,.\, ",
            "dot": r" \cdot ",
            "times": r" \times "
        }
        try:
            # 根据设置中的 'mul_symbol' 选择对应的 LaTeX 表示，并存储在 'mul_symbol_latex' 中
            self._settings['mul_symbol_latex'] = \
                mul_symbol_table[self._settings['mul_symbol']]
        except KeyError:
            # 如果 'mul_symbol' 不在表中，则直接使用原始设置值
            self._settings['mul_symbol_latex'] = \
                self._settings['mul_symbol']
        
        try:
            # 尝试根据 'mul_symbol' 选择对应的 LaTeX 表示，并存储在 'mul_symbol_latex_numbers' 中
            self._settings['mul_symbol_latex_numbers'] = \
                mul_symbol_table[self._settings['mul_symbol'] or 'dot']
        except KeyError:
            # 如果 'mul_symbol' 为非预期值，则根据情况选择默认的 'dot' 表示
            if (self._settings['mul_symbol'].strip() in
                    ['', ' ', '\\', '\\,', '\\:', '\\;', '\\quad']):
                self._settings['mul_symbol_latex_numbers'] = \
                    mul_symbol_table['dot']
            else:
                self._settings['mul_symbol_latex_numbers'] = \
                    self._settings['mul_symbol']

        # 初始化括号字典，定义左右括号对应关系
        self._delim_dict = {'(': ')', '[': ']'}

        # 定义虚数单位表，根据设置中的 'imaginary_unit' 选择 LaTeX 表示
        imaginary_unit_table = {
            None: r"i",
            "i": r"i",
            "ri": r"\mathrm{i}",
            "ti": r"\text{i}",
            "j": r"j",
            "rj": r"\mathrm{j}",
            "tj": r"\text{j}",
        }
        # 根据设置中的 'imaginary_unit' 选择对应的 LaTeX 表示，并存储在 'imaginary_unit_latex' 中
        imag_unit = self._settings['imaginary_unit']
        self._settings['imaginary_unit_latex'] = imaginary_unit_table.get(imag_unit, imag_unit)

        # 定义微分操作符表，根据设置中的 'diff_operator' 选择 LaTeX 表示
        diff_operator_table = {
            None: r"d",
            "d": r"d",
            "rd": r"\mathrm{d}",
            "td": r"\text{d}",
        }
        # 根据设置中的 'diff_operator' 选择对应的 LaTeX 表示，并存储在 'diff_operator_latex' 中
        diff_operator = self._settings['diff_operator']
        self._settings["diff_operator_latex"] = diff_operator_table.get(diff_operator, diff_operator)

    # 定义一个私有方法，用于在给定的字符串 `s` 周围添加 LaTeX 格式的左右括号，并返回处理后的字符串
    def _add_parens(self, s) -> str:
        return r"\left({}\right)".format(s)

    # TODO: 将此方法与上面的方法合并，这将需要大量的测试更改
    # 定义一个私有方法，用于在给定的字符串 `s` 周围添加 LaTeX 格式的左右括号，并返回处理后的字符串
    def _add_parens_lspace(self, s) -> str:
        return r"\left( {}\right)".format(s)

    # 定义一个公共方法，用于根据给定的 `item`、运算优先级 `level`、是否为负数 `is_neg` 和是否严格 `strict`，添加适当的括号并返回字符串表示
    def parenthesize(self, item, level, is_neg=False, strict=False) -> str:
        # 计算 `item` 的传统运算优先级
        prec_val = precedence_traditional(item)
        # 如果是负数且严格模式，则在其周围添加括号并返回处理后的字符串
        if is_neg and strict:
            return self._add_parens(self._print(item))

        # 如果 `item` 的运算优先级低于给定的 `level`，或者不严格模式且优先级小于等于 `level`，则添加括号并返回处理后的字符串
        if (prec_val < level) or ((not strict) and prec_val <= level):
            return self._add_parens(self._print(item))
        else:
            # 否则直接返回 `item` 的字符串表示
            return self._print(item)
    # 定义一个方法，用于处理字符串 s 中的上标符号
    def parenthesize_super(self, s):
        """
        Protect superscripts in s

        If the parenthesize_super option is set, protect with parentheses, else
        wrap in braces.
        """
        # 检查字符串 s 是否包含上标符号 "^"
        if "^" in s:
            # 如果设置中指定了 parenthesize_super 选项，则用括号保护上标
            if self._settings['parenthesize_super']:
                return self._add_parens(s)  # 使用括号保护上标
            else:
                return "{{{}}}".format(s)  # 使用大括号保护上标
        return s  # 返回处理后的字符串 s，如果没有上标符号则直接返回原始字符串

    # 定义一个方法，用于将表达式 expr 转换为 LaTeX 格式的字符串
    def doprint(self, expr) -> str:
        # 调用 Printer 类的 doprint 方法生成 LaTeX 格式的字符串
        tex = Printer.doprint(self, expr)

        # 根据设置的 mode 属性，选择不同的输出方式
        if self._settings['mode'] == 'plain':
            return tex  # 返回普通文本模式的 LaTeX 字符串
        elif self._settings['mode'] == 'inline':
            return r"$%s$" % tex  # 返回内联数学模式的 LaTeX 字符串
        elif self._settings['itex']:
            return r"$$%s$$" % tex  # 返回 iTex 数学环境的 LaTeX 字符串
        else:
            env_str = self._settings['mode']
            return r"\begin{%s}%s\end{%s}" % (env_str, tex, env_str)
            # 返回以指定 mode 属性为环境的 LaTeX 字符串

    # 定义一个方法，判断表达式 expr 是否需要在打印时用括号包裹
    def _needs_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        # 判断表达式是否需要括号
        return not ((expr.is_Integer and expr.is_nonnegative)
                    or (expr.is_Atom and (expr is not S.NegativeOne
                                          and expr.is_Rational is False)))
        # 返回判断结果，表达式需要括号返回 True，否则返回 False

    # 定义一个方法，判断表达式 expr 是否需要在作为函数参数时用括号包裹
    def _needs_function_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        passed as an argument to a function, False otherwise. This is a more
        liberal version of _needs_brackets, in that many expressions which need
        to be wrapped in brackets when added/subtracted/raised to a power do
        not need them when passed to a function. Such an example is a*b.
        """
        # 如果表达式不需要打印时用括号，则不需要函数调用时用括号
        if not self._needs_brackets(expr):
            return False
        else:
            # 对于乘法表达式，如果需要用括号包裹则返回 True
            if expr.is_Mul and not self._mul_is_clean(expr):
                return True
            # 对于指数表达式，如果需要用括号包裹则返回 True
            elif expr.is_Pow and not self._pow_is_clean(expr):
                return True
            # 对于加法表达式或函数表达式，总是返回 True
            elif expr.is_Add or expr.is_Function:
                return True
            else:
                return False
            # 其他情况返回 False
    def _needs_mul_brackets(self, expr, first=False, last=False) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of a Mul, False otherwise. This is True for Add,
        but also for some container objects that would not need brackets
        when appearing last in a Mul, e.g. an Integral. ``last=True``
        specifies that this expr is the last to appear in a Mul.
        ``first=True`` specifies that this expr is the first to appear in
        a Mul.
        """
        # 导入必要的模块和类
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral

        # 如果表达式是乘法表达式
        if expr.is_Mul:
            # 如果不是第一个表达式且可能提取负号，则需要括号
            if not first and expr.could_extract_minus_sign():
                return True
        # 如果表达式的传统优先级低于乘法优先级，需要括号
        elif precedence_traditional(expr) < PRECEDENCE["Mul"]:
            return True
        # 如果表达式是关系表达式，需要括号
        elif expr.is_Relational:
            return True
        # 如果表达式是分段函数，需要括号
        if expr.is_Piecewise:
            return True
        # 如果表达式包含 Mod 等特定元素，需要括号
        if any(expr.has(x) for x in (Mod,)):
            return True
        # 如果不是最后一个表达式且包含积分、乘积、求和等，需要括号
        if (not last and
                any(expr.has(x) for x in (Integral, Product, Sum))):
            return True

        # 默认情况下不需要括号
        return False

    def _needs_add_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of an Add, False otherwise.  This is False for most
        things.
        """
        # 如果表达式是关系表达式，需要括号
        if expr.is_Relational:
            return True
        # 如果表达式包含 Mod 等特定元素，需要括号
        if any(expr.has(x) for x in (Mod,)):
            return True
        # 如果表达式是加法表达式，需要括号
        if expr.is_Add:
            return True
        # 默认情况下不需要括号
        return False

    def _mul_is_clean(self, expr) -> bool:
        """
        Returns True if the expression is clean (does not need brackets)
        as part of a Mul, False otherwise.
        """
        # 遍历表达式的所有参数
        for arg in expr.args:
            # 如果参数是函数，则需要括号
            if arg.is_Function:
                return False
        # 默认情况下表达式不需要括号
        return True

    def _pow_is_clean(self, expr) -> bool:
        """
        Returns True if the expression is clean (does not need brackets)
        as part of a power operation, False otherwise.
        """
        # 判断指数运算中基数是否需要括号
        return not self._needs_brackets(expr.base)

    def _do_exponent(self, expr: str, exp):
        """
        Returns the LaTeX representation of an exponentiation with or
        without brackets around the base expression.
        """
        # 如果指数不为 None，则返回带括号的指数运算的 LaTeX 表示
        if exp is not None:
            return r"\left(%s\right)^{%s}" % (expr, exp)
        # 否则返回不带括号的表达式
        else:
            return expr

    def _print_Basic(self, expr):
        """
        Returns the LaTeX representation of a basic expression, handling
        super/subscripts and function arguments.
        """
        # 处理类名，处理带上标和下标的符号
        name = self._deal_with_super_sub(expr.__class__.__name__)
        # 如果表达式有参数
        if expr.args:
            # 递归打印每个参数，并使用 LaTeX 表示函数
            ls = [self._print(o) for o in expr.args]
            s = r"\operatorname{{{}}}\left({}\right)"
            return s.format(name, ", ".join(ls))
        else:
            # 如果表达式没有参数，直接返回名称的文本表示
            return r"\text{{{}}}".format(name)

    def _print_bool(self, e: bool | BooleanTrue | BooleanFalse):
        """
        Returns the LaTeX representation of a boolean expression.
        """
        # 返回布尔值的文本表示
        return r"\text{%s}" % e

    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        """
        Returns the LaTeX representation of NoneType.
        """
        # 返回 NoneType 的文本表示
        return r"\text{%s}" % e
    # 将表达式按照给定的顺序转换为有序项列表
    terms = self._as_ordered_terms(expr, order=order)

    # 初始化空的 TeX 字符串
    tex = ""
    # 遍历所有项
    for i, term in enumerate(terms):
        # 第一项无需符号
        if i == 0:
            pass
        # 对于后续项，根据是否可以提取负号来添加 "+/-"
        elif term.could_extract_minus_sign():
            tex += " - "
            # 如果可以提取负号，将 term 取负
            term = -term
        else:
            tex += " + "
        # 将项转换为 TeX 格式
        term_tex = self._print(term)
        # 如果需要添加括号，则在 term_tex 外围添加左右括号
        if self._needs_add_brackets(term):
            term_tex = r"\left(%s\right)" % term_tex
        # 将转换后的项添加到 TeX 字符串中
        tex += term_tex

    # 返回生成的 TeX 字符串
    return tex

    # 将置换表达式 expr 转换为其循环表示的 TeX 格式
    from sympy.combinatorics.permutations import Permutation
    if expr.size == 0:
        return r"\left( \right)"
    expr = Permutation(expr)
    expr_perm = expr.cyclic_form
    siz = expr.size
    if expr.array_form[-1] == siz - 1:
        expr_perm = expr_perm + [[siz - 1]]
    term_tex = ''
    for i in expr_perm:
        term_tex += str(i).replace(',', r"\;")
    term_tex = term_tex.replace('[', r"\left( ")
    term_tex = term_tex.replace(']', r"\right)")
    return term_tex

    # 将置换表达式 expr 转换为其矩阵表示的 TeX 格式
    from sympy.combinatorics.permutations import Permutation
    from sympy.utilities.exceptions import sympy_deprecation_warning

    # 获取设置中的置换循环打印选项并显示弃用警告
    perm_cyclic = Permutation.print_cyclic
    if perm_cyclic is not None:
        sympy_deprecation_warning(
            f"""
            Setting Permutation.print_cyclic is deprecated. Instead use
            init_printing(perm_cyclic={perm_cyclic}).
            """,
            deprecated_since_version="1.6",
            active_deprecations_target="deprecated-permutation-print_cyclic",
            stacklevel=8,
        )
    else:
        perm_cyclic = self._settings.get("perm_cyclic", True)

    # 如果设置为打印置换循环，则调用 _print_Cycle 方法
    if perm_cyclic:
        return self._print_Cycle(expr)

    # 如果置换大小为 0，则返回空的括号
    if expr.size == 0:
        return r"\left( \right)"

    # 分别转换置换的上下标为 TeX 格式
    lower = [self._print(arg) for arg in expr.array_form]
    upper = [self._print(arg) for arg in range(len(lower))]

    # 构建矩阵字符串
    row1 = " & ".join(upper)
    row2 = " & ".join(lower)
    mat = r" \\ ".join((row1, row2))
    return r"\begin{pmatrix} %s \end{pmatrix}" % mat

    # 将应用的置换 expr 转换为 TeX 格式
    perm, var = expr.args
    return r"\sigma_{%s}(%s)" % (self._print(perm), self._print(var))
    def _print_Float(self, expr):
        # 基于 StrPrinter 中的方法实现
        # 根据表达式的精度确定小数点后的位数
        dps = prec_to_dps(expr._prec)
        # 根据设置决定是否去除末尾的零
        strip = False if self._settings['full_prec'] else True
        # 如果设置中有定义最小值和最大值，则设置为这些值
        low = self._settings["min"] if "min" in self._settings else None
        high = self._settings["max"] if "max" in self._settings else None
        # 将 expr._mpf_ 转换成字符串表示的实数
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)

        # 必须始终使用乘号符号（例如 2.5 10^{20} 看起来奇怪）
        # 因此我们使用数字分隔符号
        separator = self._settings['mul_symbol_latex_numbers']

        # 如果字符串中包含 'e'，表示使用科学计数法
        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            # 科学计数法指数部分处理
            if exp[0] == '+':
                exp = exp[1:]
            # 如果设置使用逗号作为小数点分隔符，则替换点为逗号
            if self._settings['decimal_separator'] == 'comma':
                mant = mant.replace('.','{,}')

            return r"%s%s10^{%s}" % (mant, separator, exp)
        # 如果结果为正无穷，返回 LaTeX 表示
        elif str_real == "+inf":
            return r"\infty"
        # 如果结果为负无穷，返回 LaTeX 表示
        elif str_real == "-inf":
            return r"- \infty"
        else:
            # 如果设置使用逗号作为小数点分隔符，则替换点为逗号
            if self._settings['decimal_separator'] == 'comma':
                str_real = str_real.replace('.','{,}')
            return str_real

    def _print_Cross(self, expr):
        # 获取表达式中的两个向量并返回交叉乘积的 LaTeX 表示
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \times %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Curl(self, expr):
        # 获取表达式中的向量并返回旋度的 LaTeX 表示
        vec = expr._expr
        return r"\nabla\times %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Divergence(self, expr):
        # 获取表达式中的向量并返回散度的 LaTeX 表示
        vec = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Dot(self, expr):
        # 获取表达式中的两个向量并返回点积的 LaTeX 表示
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \cdot %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                 self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Gradient(self, expr):
        # 获取表达式中的函数并返回梯度的 LaTeX 表示
        func = expr._expr
        return r"\nabla %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Laplacian(self, expr):
        # 获取表达式中的函数并返回拉普拉斯算子的 LaTeX 表示
        func = expr._expr
        return r"\Delta %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_AlgebraicNumber(self, expr):
        # 如果代数数被别名化，则打印其多项式表达式；否则打印其原始表达式
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_PrimeIdeal(self, expr):
        # 打印素数理想的 LaTeX 表示，包括是否为惰性素数理想的检查
        p = self._print(expr.p)
        if expr.is_inert:
            return rf'\left({p}\right)'
        alpha = self._print(expr.alpha.as_expr())
        return rf'\left({p}, {alpha}\right)'
    def _print_Pow(self, expr: Pow):
        # 处理 x**Rational(1,n) 的特殊情况
        if expr.exp.is_Rational:
            p: int = expr.exp.p  # 获取分子
            q: int = expr.exp.q  # 获取分母
            if abs(p) == 1 and q != 1 and self._settings['root_notation']:
                base = self._print(expr.base)
                if q == 2:
                    tex = r"\sqrt{%s}" % base  # 平方根表示
                elif self._settings['itex']:
                    tex = r"\root{%d}{%s}" % (q, base)  # iTex 格式根号表示
                else:
                    tex = r"\sqrt[%d]{%s}" % (q, base)  # 根号表示
                if expr.exp.is_negative:
                    return r"\frac{1}{%s}" % tex  # 负指数处理
                else:
                    return tex  # 正指数处理
            elif self._settings['fold_frac_powers'] and q != 1:
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                # issue #12886: 对于上标的幂次，添加括号
                if expr.base.is_Symbol:
                    base = self.parenthesize_super(base)
                if expr.base.is_Function:
                    return self._print(expr.base, exp="%s/%s" % (p, q))  # 函数作为底数
                return r"%s^{%s/%s}" % (base, p, q)  # 一般幂次表示
            elif expr.exp.is_negative and expr.base.is_commutative:
                # 对于特殊情况 1^(-x)，issue 9216
                if expr.base == 1:
                    return r"%s^{%s}" % (expr.base, expr.exp)
                # 对于 (1/x)^(-y) 和 (-1/-x)^(-y)，issue 20252
                if expr.base.is_Rational:
                    base_p: int = expr.base.p  # 获取分子
                    base_q: int = expr.base.q  # 获取分母
                    if base_p * base_q == abs(base_q):
                        if expr.exp == -1:
                            return r"\frac{1}{\frac{%s}{%s}}" % (base_p, base_q)
                        else:
                            return r"\frac{1}{(\frac{%s}{%s})^{%s}}" % (base_p, base_q, abs(expr.exp))
                # 类似 1/x 的情况
                return self._print_Mul(expr)
        if expr.base.is_Function:
            return self._print(expr.base, exp=self._print(expr.exp))  # 函数作为底数
        tex = r"%s^{%s}"  # 标准的幂次表示
        return self._helper_print_standard_power(expr, tex)
    # 打印标准幂表达式的辅助函数，返回字符串形式的TeX表达式
    def _helper_print_standard_power(self, expr, template: str) -> str:
        # 获取指数部分的打印形式
        exp = self._print(expr.exp)
        # 在幂运算的底数周围添加括号，解决问题＃12886：对于幂次幂，加上括号
        base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
        # 如果底数是符号，则在超级脚本周围添加括号
        if expr.base.is_Symbol:
            base = self.parenthesize_super(base)
        # 如果底数是浮点数，则将其格式化为TeX形式
        elif expr.base.is_Float:
            base = r"{%s}" % base
        # 如果底数是导数实例，并且符合特定格式，则不在其周围使用括号
        elif (isinstance(expr.base, Derivative)
            and base.startswith(r'\left(')
            and re.match(r'\\left\(\\d?d?dot', base)
            and base.endswith(r'\right)')):
            base = base[6: -7]  # 移除最外层的添加的括号
        # 使用模板字符串格式化TeX表达式，返回结果
        return template % (base, exp)

    # 打印UnevaluatedExpr表达式的辅助函数，返回其参数的打印形式
    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    # 打印Sum表达式的辅助函数，返回其TeX表达式
    def _print_Sum(self, expr):
        # 如果表达式只有一个限制条件
        if len(expr.limits) == 1:
            tex = r"\sum_{%s=%s}^{%s} " % \
                tuple([self._print(i) for i in expr.limits[0]])
        else:
            # 格式化不等式限制条件
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            # 使用换行符连接多个不等式条件
            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        # 如果函数部分是Add类型，则在周围加上括号
        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex

    # 打印Product表达式的辅助函数，返回其TeX表达式
    def _print_Product(self, expr):
        # 如果表达式只有一个限制条件
        if len(expr.limits) == 1:
            tex = r"\prod_{%s=%s}^{%s} " % \
                tuple([self._print(i) for i in expr.limits[0]])
        else:
            # 格式化不等式限制条件
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            # 使用换行符连接多个不等式条件
            tex = r"\prod_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        # 如果函数部分是Add类型，则在周围加上括号
        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
    # 定义一个方法用于打印 BasisDependent 类型的表达式
    def _print_BasisDependent(self, expr: 'BasisDependent'):
        # 导入 sympy.vector 中的 Vector 类
        from sympy.vector import Vector

        # 初始化一个空列表 o1，用于存储最终输出的 LaTeX 表达式
        o1: list[str] = []

        # 如果表达式 expr 等于其零向量 zero，则直接返回零向量的 LaTeX 表示形式
        if expr == expr.zero:
            return expr.zero._latex_form
        
        # 如果 expr 是 Vector 类型的实例
        if isinstance(expr, Vector):
            # 对 expr 进行分离，得到其组成的项 items
            items = expr.separate().items()
        else:
            # 否则，将 expr 包装为只包含一个项 (0, expr) 的列表
            items = [(0, expr)]

        # 遍历 items 中的每个系统 system 和向量 vect
        for system, vect in items:
            # 获取向量 vect 的分量 items，并转换为列表
            inneritems = list(vect.components.items())
            # 根据分量的键值进行排序
            inneritems.sort(key=lambda x: x[0].__str__())
            # 遍历内部项中的每对键值对 k 和 v
            for k, v in inneritems:
                # 根据分量 v 的值 v 的 LaTeX 表示形式 k._latex_form，添加到 o1 中
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    # 如果 v 不等于 1 或 -1，则构造带括号的 v 的 LaTeX 表示形式
                    arg_str = r'\left(' + self._print(v) + r'\right)'
                    o1.append(' + ' + arg_str + k._latex_form)

        # 将 o1 中的所有项连接成一个字符串 outstr
        outstr = (''.join(o1))
        
        # 如果 outstr 的第一个字符不是减号，则去掉前面的三个字符（'+ '）
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            # 否则，去掉前面的一个字符（' -'）
            outstr = outstr[1:]
        
        # 返回处理后的字符串 outstr
        return outstr

    # 定义一个方法用于打印 Indexed 类型的表达式
    def _print_Indexed(self, expr):
        # 打印 expr 的基础部分的 LaTeX 表示形式
        tex_base = self._print(expr.base)
        # 构造带有索引的 LaTeX 表示形式，使用大括号和逗号分隔
        tex = '{'+tex_base+'}'+'_{%s}' % ','.join(
            map(self._print, expr.indices))
        # 返回构造的 LaTeX 表示形式 tex
        return tex

    # 定义一个方法用于打印 IndexedBase 类型的表达式
    def _print_IndexedBase(self, expr):
        # 打印 expr 的标签部分的 LaTeX 表示形式
        return self._print(expr.label)

    # 定义一个方法用于打印 Idx 类型的表达式
    def _print_Idx(self, expr):
        # 打印 expr 的标签部分的 LaTeX 表示形式
        label = self._print(expr.label)
        
        # 如果 expr 有上限 upper
        if expr.upper is not None:
            # 打印 expr 的上限部分的 LaTeX 表示形式
            upper = self._print(expr.upper)
            
            # 如果 expr 有下限 lower
            if expr.lower is not None:
                # 打印 expr 的下限部分的 LaTeX 表示形式
                lower = self._print(expr.lower)
            else:
                # 否则，使用零作为下限的 LaTeX 表示形式
                lower = self._print(S.Zero)
            
            # 构造区间的 LaTeX 表示形式，使用大括号和省略号表示范围
            interval = '{lower}\\mathrel{{..}}\\nobreak {upper}'.format(
                    lower = lower, upper = upper)
            
            # 返回带有标签和区间的 LaTeX 表示形式
            return '{{{label}}}_{{{interval}}}'.format(
                label = label, interval = interval)
        
        # 如果没有定义边界，直接返回标签的 LaTeX 表示形式
        return label
    # 打印导数的 LaTeX 表示
    def _print_Derivative(self, expr):
        # 检查表达式是否需要偏导数符号
        if requires_partial(expr.expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = self._settings["diff_operator_latex"]

        # 初始化 LaTeX 字符串和维度计数
        tex = ""
        dim = 0

        # 反向遍历变量计数列表
        for x, num in reversed(expr.variable_count):
            dim += num
            # 根据变量数目添加对应的偏导数或求导数符号
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol,
                                        self.parenthesize_super(self._print(x)),
                                        self._print(num))

        # 根据维度数选择合适的分数形式
        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, self._print(dim), tex)

        # 检查表达式是否有子表达式可能提取负号
        if any(i.could_extract_minus_sign() for i in expr.args):
            # 返回包含负号的表达式
            return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=True,
                                                  strict=True))

        # 返回正常的表达式
        return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=False,
                                                  strict=True))

    # 打印替换表达式的 LaTeX 表示
    def _print_Subs(self, subs):
        expr, old, new = subs.args
        # 打印主表达式的 LaTeX 表示
        latex_expr = self._print(expr)
        # 生成旧变量和新变量的 LaTeX 表示
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        # 使用换行符连接每对旧变量和新变量的 LaTeX 表示
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        # 返回带有替换条件的表达式的 LaTeX 表示
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr,
                                                         latex_subs)
    # 定义一个方法用于打印积分表达式的 LaTeX 格式字符串
    def _print_Integral(self, expr):
        tex, symbols = "", []  # 初始化 LaTeX 字符串和符号列表

        # 获取微分符号的 LaTeX 格式
        diff_symbol = self._settings["diff_operator_latex"]

        # 如果积分限制条件不超过4个，并且每个限制条件只有一个符号
        if len(expr.limits) <= 4 and all(len(lim) == 1 for lim in expr.limits):
            # 构造积分符号的 LaTeX 字符串
            tex = r"\i" + "i"*(len(expr.limits) - 1) + "nt"
            # 构造每个符号的 LaTeX 字符串并添加到符号列表中
            symbols = [r"\, %s%s" % (diff_symbol, self._print(symbol[0]))
                       for symbol in expr.limits]

        else:
            # 反向遍历积分限制条件
            for lim in reversed(expr.limits):
                symbol = lim[0]
                tex += r"\int"  # 添加积分符号到 LaTeX 字符串

                # 处理多种限制条件的情况
                if len(lim) > 1:
                    # 根据限制条件的数量添加上下标
                    if self._settings['mode'] != 'inline' \
                            and not self._settings['itex']:
                        tex += r"\limits"  # 非内联模式添加 limits 命令

                    if len(lim) == 3:
                        tex += "_{%s}^{%s}" % (self._print(lim[1]),
                                               self._print(lim[2]))
                    if len(lim) == 2:
                        tex += "^{%s}" % (self._print(lim[1]))

                # 构造每个符号的 LaTeX 字符串并添加到符号列表开头
                symbols.insert(0, r"\, %s%s" % (diff_symbol, self._print(symbol)))

        # 返回格式化后的积分表达式的 LaTeX 字符串
        return r"%s %s%s" % (tex, self.parenthesize(expr.function,
                                                    PRECEDENCE["Mul"],
                                                    is_neg=any(i.could_extract_minus_sign() for i in expr.args),
                                                    strict=True),
                             "".join(symbols))

    # 定义一个方法用于打印极限表达式的 LaTeX 格式字符串
    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args  # 解析表达式中的参数

        # 构造极限符号的 LaTeX 字符串
        tex = r"\lim_{%s \to " % self._print(z)

        # 处理极限的方向
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), self._print(dir))

        # 如果表达式是关联操作（例如乘法、加法等），使用括号包裹
        if isinstance(e, AssocOp):
            return r"%s\left(%s\right)" % (tex, self._print(e))
        else:
            return r"%s %s" % (tex, self._print(e))
    # 将函数名转换为 LaTeX 格式的字符串表示
    def _hprint_Function(self, func: str) -> str:
        r'''
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, excluding sub- and superscripts, just use that letter
          - if it is a longer name, then put \operatorname{} around it and be
            mindful of undercores in the name
        '''
        # 处理函数名中的上标和下标
        func = self._deal_with_super_sub(func)
        
        # 查找上标和下标的位置
        superscriptidx = func.find("^")
        subscriptidx = func.find("_")
        
        # 根据函数名的不同情况进行处理
        if func in accepted_latex_functions:
            # 如果是已识别的 LaTeX 函数名，则使用相应的 LaTeX 命令
            name = r"\%s" % func
        elif len(func) == 1 or func.startswith('\\') or subscriptidx == 1 or superscriptidx == 1:
            # 如果函数名只有一个字符，或者以反斜杠开头，或者包含上标或下标，则直接使用该字符
            name = func
        else:
            # 处理较长的函数名，使用 \operatorname{} 包裹，并注意名称中的下划线
            if superscriptidx > 0 and subscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:min(subscriptidx,superscriptidx)],
                    func[min(subscriptidx,superscriptidx):])
            elif superscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:superscriptidx],
                    func[superscriptidx:])
            elif subscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:subscriptidx],
                    func[subscriptidx:])
            else:
                name = r"\operatorname{%s}" % func
        
        return name

    # 将未定义的函数表达式转换为 LaTeX 格式
    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    # 将逐元素应用函数表达式转换为 LaTeX 格式
    def _print_ElementwiseApplyFunction(self, expr):
        return r"{%s}_{\circ}\left({%s}\right)" % (
            self._print(expr.function),
            self._print(expr.expr),
        )

    # 返回特殊函数类和它们对应的 LaTeX 表示的映射
    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: r'\delta',
                gamma:  r'\Gamma',
                lowergamma: r'\gamma',
                beta: r'\operatorname{B}',
                DiracDelta: r'\delta',
                Chi: r'\operatorname{Chi}'}

    # 将特殊函数类的实例转换为 LaTeX 格式
    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                return self._special_function_classes[cls]
        # 如果不是特殊函数类的实例，则使用通用函数名转换函数
        return self._hprint_Function(str(expr))

    # 将 Lambda 表达式转换为 LaTeX 格式
    def _print_Lambda(self, expr):
        symbols, expr = expr.args

        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(tuple(symbols))

        # 格式化 Lambda 表达式的 LaTeX 表示
        tex = r"\left( %s \mapsto %s \right)" % (symbols, self._print(expr))

        return tex
    # 定义一个打印恒等函数的方法，返回 LaTeX 表示
    def _print_IdentityFunction(self, expr):
        return r"\left( x \mapsto x \right)"

    # 定义一个打印可变参数函数的方法，返回 LaTeX 表示
    def _hprint_variadic_function(self, expr, exp=None) -> str:
        # 对表达式的参数按照默认排序关键字排序
        args = sorted(expr.args, key=default_sort_key)
        # 将每个参数打印成 LaTeX 表示，并组成列表
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        # 构建包含函数名和参数的 LaTeX 表示
        tex = r"\%s\left(%s\right)" % (str(expr.func).lower(),
                                       ", ".join(texargs))
        # 如果指定了指数，添加上指数
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 将 _print_Min 和 _print_Max 方法指向 _hprint_variadic_function，因为它们的实现相同
    _print_Min = _print_Max = _hprint_variadic_function

    # 定义打印 floor 函数的方法，返回 LaTeX 表示
    def _print_floor(self, expr, exp=None):
        # 构建 floor 函数的 LaTeX 表示
        tex = r"\left\lfloor{%s}\right\rfloor" % self._print(expr.args[0])

        # 如果指定了指数，添加上指数
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 定义打印 ceiling 函数的方法，返回 LaTeX 表示
    def _print_ceiling(self, expr, exp=None):
        # 构建 ceiling 函数的 LaTeX 表示
        tex = r"\left\lceil{%s}\right\rceil" % self._print(expr.args[0])

        # 如果指定了指数，添加上指数
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 定义打印 log 函数的方法，返回 LaTeX 表示
    def _print_log(self, expr, exp=None):
        # 根据设置选择是使用 log 还是 ln 符号
        if not self._settings["ln_notation"]:
            tex = r"\log{\left(%s \right)}" % self._print(expr.args[0])
        else:
            tex = r"\ln{\left(%s \right)}" % self._print(expr.args[0])

        # 如果指定了指数，添加上指数
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 定义打印绝对值函数的方法，返回 LaTeX 表示
    def _print_Abs(self, expr, exp=None):
        # 构建绝对值函数的 LaTeX 表示
        tex = r"\left|{%s}\right|" % self._print(expr.args[0])

        # 如果指定了指数，添加上指数
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 定义打印实部函数的方法，返回 LaTeX 表示
    def _print_re(self, expr, exp=None):
        # 根据设置选择使用不同的实部表示方式
        if self._settings['gothic_re_im']:
            tex = r"\Re{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{re}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        # 调用内部方法处理指数，并返回结果
        return self._do_exponent(tex, exp)

    # 定义打印虚部函数的方法，返回 LaTeX 表示
    def _print_im(self, expr, exp=None):
        # 根据设置选择使用不同的虚部表示方式
        if self._settings['gothic_re_im']:
            tex = r"\Im{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{im}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        # 调用内部方法处理指数，并返回结果
        return self._do_exponent(tex, exp)

    # 定义打印逻辑非函数的方法，返回 LaTeX 表示
    def _print_Not(self, e):
        # 导入需要用到的逻辑运算符
        from sympy.logic.boolalg import (Equivalent, Implies)

        # 根据表达式的类型选择适当的逻辑非表示方式
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg \left(%s\right)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])
    # 定义一个方法，用于打印逻辑运算表达式，例如与运算
    def _print_LogOp(self, args, char):
        # 取第一个参数
        arg = args[0]
        # 如果第一个参数是布尔类型且不是否定形式，则加上括号
        if arg.is_Boolean and not arg.is_Not:
            tex = r"\left(%s\right)" % self._print(arg)
        else:
            tex = r"%s" % self._print(arg)

        # 遍历其余参数
        for arg in args[1:]:
            # 如果参数是布尔类型且不是否定形式，则加上运算符和括号
            if arg.is_Boolean and not arg.is_Not:
                tex += r" %s \left(%s\right)" % (char, self._print(arg))
            else:
                tex += r" %s %s" % (char, self._print(arg))

        return tex

    # 打印与运算表达式
    def _print_And(self, e):
        # 对表达式参数进行排序
        args = sorted(e.args, key=default_sort_key)
        # 调用打印逻辑运算的方法，并传入与运算符号
        return self._print_LogOp(args, r"\wedge")

    # 打印或运算表达式
    def _print_Or(self, e):
        # 对表达式参数进行排序
        args = sorted(e.args, key=default_sort_key)
        # 调用打印逻辑运算的方法，并传入或运算符号
        return self._print_LogOp(args, r"\vee")

    # 打印异或运算表达式
    def _print_Xor(self, e):
        # 对表达式参数进行排序
        args = sorted(e.args, key=default_sort_key)
        # 调用打印逻辑运算的方法，并传入异或运算符号
        return self._print_LogOp(args, r"\veebar")

    # 打印蕴含运算表达式
    def _print_Implies(self, e, altchar=None):
        # 调用打印逻辑运算的方法，并传入蕴含符号
        return self._print_LogOp(e.args, altchar or r"\Rightarrow")

    # 打印等价运算表达式
    def _print_Equivalent(self, e, altchar=None):
        # 对表达式参数进行排序
        args = sorted(e.args, key=default_sort_key)
        # 调用打印逻辑运算的方法，并传入等价符号
        return self._print_LogOp(args, altchar or r"\Leftrightarrow")

    # 打印复共轭运算表达式
    def _print_conjugate(self, expr, exp=None):
        # 构造复共轭运算的 LaTeX 表示
        tex = r"\overline{%s}" % self._print(expr.args[0])

        # 如果有指数，则添加指数
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 打印极坐标提升运算表达式
    def _print_polar_lift(self, expr, exp=None):
        # 构造极坐标提升运算的 LaTeX 表示
        func = r"\operatorname{polar\_lift}"
        arg = r"{\left(%s \right)}" % self._print(expr.args[0])

        # 如果有指数，则添加指数
        if exp is not None:
            return r"%s^{%s}%s" % (func, exp, arg)
        else:
            return r"%s%s" % (func, arg)

    # 打印指数底数 e 的运算表达式
    def _print_ExpBase(self, expr, exp=None):
        # 构造指数底数 e 的运算的 LaTeX 表示
        tex = r"e^{%s}" % self._print(expr.args[0])
        return self._do_exponent(tex, exp)

    # 打印自然指数 e
    def _print_Exp1(self, expr, exp=None):
        return "e"

    # 打印椭圆函数 K
    def _print_elliptic_k(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        # 如果有指数，则添加指数
        if exp is not None:
            return r"K^{%s}%s" % (exp, tex)
        else:
            return r"K%s" % tex

    # 打印椭圆函数 F
    def _print_elliptic_f(self, expr, exp=None):
        tex = r"\left(%s\middle| %s\right)" % \
            (self._print(expr.args[0]), self._print(expr.args[1]))
        # 如果有指数，则添加指数
        if exp is not None:
            return r"F^{%s}%s" % (exp, tex)
        else:
            return r"F%s" % tex

    # 打印椭圆函数 E
    def _print_elliptic_e(self, expr, exp=None):
        # 构造椭圆函数 E 的 LaTeX 表示
        if len(expr.args) == 2:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        # 如果有指数，则添加指数
        if exp is not None:
            return r"E^{%s}%s" % (exp, tex)
        else:
            return r"E%s" % tex
    # 打印椭圆积分函数 \Pi 的 LaTeX 表示
    def _print_elliptic_pi(self, expr, exp=None):
        # 检查参数个数，选择不同的 LaTeX 格式
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]),
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        # 如果有指数，添加到 \Pi 的 LaTeX 表示中
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex

    # 打印贝塔函数 \operatorname{B} 的 LaTeX 表示
    def _print_beta(self, expr, exp=None):
        x = expr.args[0]
        # 处理未求值的单参数贝塔函数情况
        y = expr.args[0] if len(expr.args) == 1 else expr.args[1]
        tex = rf"\left({x}, {y}\right)"

        # 如果有指数，添加到 \operatorname{B} 的 LaTeX 表示中
        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    # 打印正规化贝塔不完全积分函数 \operatorname{I} 的 LaTeX 表示
    def _print_betainc_regularized(self, expr, exp=None):
        # 调用 _print_betainc 方法生成 LaTeX 表示
        return self._print_betainc(expr, exp, operator='I')

    # 打印上半伽马函数 \Gamma 的 LaTeX 表示
    def _print_uppergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        # 如果有指数，添加到 \Gamma 的 LaTeX 表示中
        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

    # 打印下半伽马函数 \gamma 的 LaTeX 表示
    def _print_lowergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        # 如果有指数，添加到 \gamma 的 LaTeX 表示中
        if exp is not None:
            return r"\gamma^{%s}%s" % (exp, tex)
        else:
            return r"\gamma%s" % tex

    # 打印单参数函数的通用方法 \operatorname{func} 的 LaTeX 表示
    def _hprint_one_arg_func(self, expr, exp=None) -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        # 如果有指数，添加到函数名的 LaTeX 表示中
        if exp is not None:
            return r"%s^{%s}%s" % (self._print(expr.func), exp, tex)
        else:
            return r"%s%s" % (self._print(expr.func), tex)

    # 打印卡伦特函数 \operatorname{Chi} 的 LaTeX 表示
    def _print_Chi(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        # 如果有指数，添加到 \operatorname{Chi} 的 LaTeX 表示中
        if exp is not None:
            return r"\operatorname{Chi}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{Chi}%s" % tex

    # 打印指数积分函数 \operatorname{E} 的 LaTeX 表示
    def _print_expint(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[1])
        nu = self._print(expr.args[0])

        # 如果有指数，添加到 \operatorname{E} 的 LaTeX 表示中
        if exp is not None:
            return r"\operatorname{E}_{%s}^{%s}%s" % (nu, exp, tex)
        else:
            return r"\operatorname{E}_{%s}%s" % (nu, tex)
    # 定义一个方法来打印 Fresnel S 函数的 LaTeX 表示
    def _print_fresnels(self, expr, exp=None):
        # 将表达式的第一个参数转换成 LaTeX 表示，并加上括号
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        # 如果指定了指数 exp，则返回带指数的表达式
        if exp is not None:
            return r"S^{%s}%s" % (exp, tex)
        else:
            return r"S%s" % tex

    # 定义一个方法来打印 Fresnel C 函数的 LaTeX 表示
    def _print_fresnelc(self, expr, exp=None):
        # 将表达式的第一个参数转换成 LaTeX 表示，并加上括号
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        # 如果指定了指数 exp，则返回带指数的表达式
        if exp is not None:
            return r"C^{%s}%s" % (exp, tex)
        else:
            return r"C%s" % tex

    # 定义一个方法来打印子阶乘（subfactorial）函数的 LaTeX 表示
    def _print_subfactorial(self, expr, exp=None):
        # 将表达式的第一个参数转换成子阶乘函数的 LaTeX 表示，并加上括号
        tex = r"!%s" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        # 如果指定了指数 exp，则返回带指数的表达式
        if exp is not None:
            return r"\left(%s\right)^{%s}" % (tex, exp)
        else:
            return tex

    # 定义一个方法来打印阶乘（factorial）函数的 LaTeX 表示
    def _print_factorial(self, expr, exp=None):
        # 将表达式的第一个参数转换成阶乘函数的 LaTeX 表示，并加上括号
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        # 如果指定了指数 exp，则返回带指数的表达式
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 定义一个方法来打印双阶乘（double factorial）函数的 LaTeX 表示
    def _print_factorial2(self, expr, exp=None):
        # 将表达式的第一个参数转换成双阶乘函数的 LaTeX 表示，并加上括号
        tex = r"%s!!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        # 如果指定了指数 exp，则返回带指数的表达式
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 定义一个方法来打印二项式系数（binomial coefficient）的 LaTeX 表示
    def _print_binomial(self, expr, exp=None):
        # 将表达式的两个参数转换成二项式系数的 LaTeX 表示
        tex = r"{\binom{%s}{%s}}" % (self._print(expr.args[0]),
                                     self._print(expr.args[1]))

        # 如果指定了指数 exp，则返回带指数的表达式
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    # 定义一个方法来打印上升阶乘（rising factorial）函数的 LaTeX 表示
    def _print_RisingFactorial(self, expr, exp=None):
        # 获取函数的两个参数
        n, k = expr.args
        # 将第一个参数转换成 LaTeX 表示，并加上括号
        base = r"%s" % self.parenthesize(n, PRECEDENCE['Func'])

        # 构建上升阶乘函数的 LaTeX 表示
        tex = r"{%s}^{\left(%s\right)}" % (base, self._print(k))

        # 调用辅助方法处理可能存在的指数
        return self._do_exponent(tex, exp)

    # 定义一个方法来打印下降阶乘（falling factorial）函数的 LaTeX 表示
    def _print_FallingFactorial(self, expr, exp=None):
        # 获取函数的两个参数
        n, k = expr.args
        # 将第二个参数转换成 LaTeX 表示，并加上括号
        sub = r"%s" % self.parenthesize(k, PRECEDENCE['Func'])

        # 构建下降阶乘函数的 LaTeX 表示
        tex = r"{\left(%s\right)}_{%s}" % (self._print(n), sub)

        # 调用辅助方法处理可能存在的指数
        return self._do_exponent(tex, exp)

    # 定义一个私有方法来打印贝塞尔函数基类的 LaTeX 表示
    def _hprint_BesselBase(self, expr, exp, sym: str) -> str:
        # 使用给定的符号生成 LaTeX 表示
        tex = r"%s" % (sym)

        # 检查是否需要添加指数
        need_exp = False
        if exp is not None:
            if tex.find('^') == -1:
                tex = r"%s^{%s}" % (tex, exp)
            else:
                need_exp = True

        # 构建贝塞尔函数的完整 LaTeX 表示
        tex = r"%s_{%s}\left(%s\right)" % (tex, self._print(expr.order),
                                           self._print(expr.argument))

        # 如果之前判断需要添加指数，则调用辅助方法添加指数
        if need_exp:
            tex = self._do_exponent(tex, exp)
        return tex

    # 定义一个私有方法来打印向量的 LaTeX 表示
    def _hprint_vec(self, vec) -> str:
        # 如果向量为空，直接返回空字符串
        if not vec:
            return ""

        # 构建向量的 LaTeX 表示，包括逗号分隔的元素
        s = ""
        for i in vec[:-1]:
            s += "%s, " % self._print(i)
        s += self._print(vec[-1])
        return s

    # 定义一个方法来打印贝塞尔函数 J 的 LaTeX 表示
    def _print_besselj(self, expr, exp=None):
        # 调用私有方法生成贝塞尔函数 J 的 LaTeX 表示
        return self._hprint_BesselBase(expr, exp, 'J')

    # 定义一个方法来打印贝塞尔函数 I 的 LaTeX 表示
    def _print_besseli(self, expr, exp=None):
        # 调用私有方法生成贝塞尔函数 I 的 LaTeX 表示
        return self._hprint_BesselBase(expr, exp, 'I')

    # 定义一个方法来打印贝塞尔函数 K 的 LaTeX 表示
    def _print_besselk(self, expr, exp=None):
        # 调用私有方法生成贝塞尔函数 K 的 LaTeX 表示
        return self._hprint_BesselBase(expr, exp, 'K')
    # 使用 BesselBase 类的打印方法打印第一类和第二类贝塞尔函数的表达式
    def _print_bessely(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'Y')

    # 使用 BesselBase 类的打印方法打印贝塞尔函数的第二类 Neumann 函数的表达式
    def _print_yn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'y')

    # 使用 BesselBase 类的打印方法打印贝塞尔函数的第一类 Bessel 函数的表达式
    def _print_jn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'j')

    # 使用 BesselBase 类的打印方法打印第一类 Hankel 函数的表达式
    def _print_hankel1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(1)}')

    # 使用 BesselBase 类的打印方法打印第二类 Hankel 函数的表达式
    def _print_hankel2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(2)}')

    # 使用 BesselBase 类的打印方法打印第一类修改 Bessel 函数的表达式
    def _print_hn1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(1)}')

    # 使用 BesselBase 类的打印方法打印第二类修改 Bessel 函数的表达式
    def _print_hn2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(2)}')

    # 使用 Airy 类的打印方法打印 Airy 函数的表达式，包括指数和特定的记号
    def _hprint_airy(self, expr, exp=None, notation="") -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s%s" % (notation, tex)

    # 使用 Airy 类的打印方法打印 Airy 函数的导数的表达式，包括指数和特定的记号
    def _hprint_airy_prime(self, expr, exp=None, notation="") -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"{%s^\prime}^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s^\prime%s" % (notation, tex)

    # 使用 Airy 类的打印方法打印 Airy 函数的第一类表达式
    def _print_airyai(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Ai')

    # 使用 Airy 类的打印方法打印 Airy 函数的第二类表达式
    def _print_airybi(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Bi')

    # 使用 Airy 类的打印方法打印 Airy 函数的第一类导数表达式
    def _print_airyaiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Ai')

    # 使用 Airy 类的打印方法打印 Airy 函数的第二类导数表达式
    def _print_airybiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Bi')

    # 使用特定的超几何函数类的打印方法打印超几何函数的表达式，包括指数和相关参数
    def _print_hyper(self, expr, exp=None):
        tex = r"{{}_{%s}F_{%s}\left(\begin{matrix} %s \\ %s \end{matrix}" \
              r"\middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._hprint_vec(expr.ap), self._hprint_vec(expr.bq),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, exp)
        return tex

    # 使用 MeijerG 类的打印方法打印 Meijer G 函数的表达式，包括指数和相关参数
    def _print_meijerg(self, expr, exp=None):
        tex = r"{G_{%s, %s}^{%s, %s}\left(\begin{matrix} %s & %s \\" \
              r"%s & %s \end{matrix} \middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._print(len(expr.bm)), self._print(len(expr.an)),
              self._hprint_vec(expr.an), self._hprint_vec(expr.aother),
              self._hprint_vec(expr.bm), self._hprint_vec(expr.bother),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, exp)
        return tex

    # 使用 Dirichlet_eta 类的打印方法打印 Dirichlet eta 函数的表达式，包括指数和相关参数
    def _print_dirichlet_eta(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\eta^{%s}%s" % (exp, tex)
        return r"\eta%s" % tex
    # 打印 Zeta 函数表达式的 LaTeX 代码
    def _print_zeta(self, expr, exp=None):
        # 如果表达式参数长度为2，生成带括号的表达式
        if len(expr.args) == 2:
            tex = r"\left(%s, %s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        # 如果指数参数不为空，添加 Zeta 函数的指数部分
        if exp is not None:
            return r"\zeta^{%s}%s" % (exp, tex)
        # 否则，只返回 Zeta 函数的表达式
        return r"\zeta%s" % tex

    # 打印 Stieltjes gamma 函数表达式的 LaTeX 代码
    def _print_stieltjes(self, expr, exp=None):
        # 如果表达式参数长度为2，生成带下标的表达式
        if len(expr.args) == 2:
            tex = r"_{%s}\left(%s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"_{%s}" % self._print(expr.args[0])
        # 如果指数参数不为空，添加 Stieltjes gamma 函数的指数部分
        if exp is not None:
            return r"\gamma%s^{%s}" % (tex, exp)
        # 否则，只返回 Stieltjes gamma 函数的表达式
        return r"\gamma%s" % tex

    # 打印 Lerch Phi 函数表达式的 LaTeX 代码
    def _print_lerchphi(self, expr, exp=None):
        # 生成带括号的 Lerch Phi 函数表达式
        tex = r"\left(%s, %s, %s\right)" % tuple(map(self._print, expr.args))
        # 如果指数参数为空，返回 Lerch Phi 函数的表达式
        if exp is None:
            return r"\Phi%s" % tex
        # 否则，添加 Lerch Phi 函数的指数部分
        return r"\Phi^{%s}%s" % (exp, tex)

    # 打印 Polylogarithm 函数表达式的 LaTeX 代码
    def _print_polylog(self, expr, exp=None):
        # 获取 Polylogarithm 函数的参数
        s, z = map(self._print, expr.args)
        tex = r"\left(%s\right)" % z
        # 如果指数参数为空，返回 Polylogarithm 函数的表达式
        if exp is None:
            return r"\operatorname{Li}_{%s}%s" % (s, tex)
        # 否则，添加 Polylogarithm 函数的指数部分
        return r"\operatorname{Li}_{%s}^{%s}%s" % (s, exp, tex)

    # 打印 Jacobi 多项式表达式的 LaTeX 代码
    def _print_jacobi(self, expr, exp=None):
        # 获取 Jacobi 多项式的参数
        n, a, b, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s,%s\right)}\left(%s\right)" % (n, a, b, x)
        # 如果指数参数不为空，添加 Jacobi 多项式的指数部分
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 打印 Gegenbauer 函数表达式的 LaTeX 代码
    def _print_gegenbauer(self, expr, exp=None):
        # 获取 Gegenbauer 函数的参数
        n, a, x = map(self._print, expr.args)
        tex = r"C_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        # 如果指数参数不为空，添加 Gegenbauer 函数的指数部分
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 打印 Chebyshev T 函数表达式的 LaTeX 代码
    def _print_chebyshevt(self, expr, exp=None):
        # 获取 Chebyshev T 函数的参数
        n, x = map(self._print, expr.args)
        tex = r"T_{%s}\left(%s\right)" % (n, x)
        # 如果指数参数不为空，添加 Chebyshev T 函数的指数部分
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 打印 Chebyshev U 函数表达式的 LaTeX 代码
    def _print_chebyshevu(self, expr, exp=None):
        # 获取 Chebyshev U 函数的参数
        n, x = map(self._print, expr.args)
        tex = r"U_{%s}\left(%s\right)" % (n, x)
        # 如果指数参数不为空，添加 Chebyshev U 函数的指数部分
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 打印 Legendre 多项式表达式的 LaTeX 代码
    def _print_legendre(self, expr, exp=None):
        # 获取 Legendre 多项式的参数
        n, x = map(self._print, expr.args)
        tex = r"P_{%s}\left(%s\right)" % (n, x)
        # 如果指数参数不为空，添加 Legendre 多项式的指数部分
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 打印 Associated Legendre 函数表达式的 LaTeX 代码
    def _print_assoc_legendre(self, expr, exp=None):
        # 获取 Associated Legendre 函数的参数
        n, a, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        # 如果指数参数不为空，添加 Associated Legendre 函数的指数部分
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 打印 Hermite 多项式表达式的 LaTeX 代码
    def _print_hermite(self, expr, exp=None):
        # 获取 Hermite 多项式的参数
        n, x = map(self._print, expr.args)
        tex = r"H_{%s}\left(%s\right)" % (n, x)
        # 如果指数参数不为空，添加 Hermite 多项式的指数部分
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex
    # 将 Laguerre 多项式 L_n(x) 转换为 LaTeX 格式
    def _print_laguerre(self, expr, exp=None):
        # 提取表达式中的参数 n 和 x，并转换为 LaTeX 格式
        n, x = map(self._print, expr.args)
        # 构建 LaTeX 表示式 L_n(x)
        tex = r"L_{%s}\left(%s\right)" % (n, x)
        # 如果指定了指数 exp，则添加到表达式中
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 将关联 Laguerre 多项式 L_n^(a)(x) 转换为 LaTeX 格式
    def _print_assoc_laguerre(self, expr, exp=None):
        # 提取表达式中的参数 n, a 和 x，并转换为 LaTeX 格式
        n, a, x = map(self._print, expr.args)
        # 构建 LaTeX 表示式 L_n^(a)(x)
        tex = r"L_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        # 如果指定了指数 exp，则添加到表达式中
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 将球谐函数 Y_n^m(theta, phi) 转换为 LaTeX 格式
    def _print_Ynm(self, expr, exp=None):
        # 提取表达式中的参数 n, m, theta 和 phi，并转换为 LaTeX 格式
        n, m, theta, phi = map(self._print, expr.args)
        # 构建 LaTeX 表示式 Y_n^m(theta, phi)
        tex = r"Y_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        # 如果指定了指数 exp，则添加到表达式中
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 将旋转球谐函数 Z_n^m(theta, phi) 转换为 LaTeX 格式
    def _print_Znm(self, expr, exp=None):
        # 提取表达式中的参数 n, m, theta 和 phi，并转换为 LaTeX 格式
        n, m, theta, phi = map(self._print, expr.args)
        # 构建 LaTeX 表示式 Z_n^m(theta, phi)
        tex = r"Z_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        # 如果指定了指数 exp，则添加到表达式中
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    # 将 Mathieu 函数 C(a, q, z) 转换为 LaTeX 格式
    def __print_mathieu_functions(self, character, args, prime=False, exp=None):
        # 提取 Mathieu 函数的参数 a, q 和 z，并转换为 LaTeX 格式
        a, q, z = map(self._print, args)
        # 如果需要，添加 Mathieu 函数的一阶导数符号
        sup = r"^{\prime}" if prime else ""
        # 如果指定了指数 exp，则添加到表达式中
        exp = "" if not exp else "^{%s}" % exp
        # 构建 LaTeX 表示式 character^sup(a, q, z)exp
        return r"%s%s\left(%s, %s, %s\right)%s" % (character, sup, a, q, z, exp)

    # 将 Mathieu 函数 C_n(exp) 转换为 LaTeX 格式
    def _print_mathieuc(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, exp=exp)

    # 将 Mathieu 函数 S_n(exp) 转换为 LaTeX 格式
    def _print_mathieus(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, exp=exp)

    # 将 Mathieu 函数 C_n'(exp) 转换为 LaTeX 格式
    def _print_mathieucprime(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, prime=True, exp=exp)

    # 将 Mathieu 函数 S_n'(exp) 转换为 LaTeX 格式
    def _print_mathieusprime(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, prime=True, exp=exp)

    # 将有理数表达式转换为 LaTeX 格式
    def _print_Rational(self, expr):
        # 如果分母不为1，则构建带分数线的 LaTeX 表示式
        if expr.q != 1:
            sign = ""
            p = expr.p
            # 如果分子为负数，则添加负号
            if expr.p < 0:
                sign = "- "
                p = -p
            # 如果设置中折叠短分数，则使用紧凑形式
            if self._settings['fold_short_frac']:
                return r"%s%d / %d" % (sign, p, expr.q)
            # 否则使用分数形式
            return r"%s\frac{%d}{%d}" % (sign, p, expr.q)
        else:
            # 如果分母为1，则直接输出分子
            return self._print(expr.p)

    # 将阶符号表达式转换为 LaTeX 格式
    def _print_Order(self, expr):
        # 转换表达式部分并初始化输出字符串
        s = self._print(expr.expr)
        # 如果存在指定的点或变量数量超过1，则添加附加信息
        if expr.point and any(p != S.Zero for p in expr.point) or \
           len(expr.variables) > 1:
            s += '; '
            # 如果变量数量大于1，则输出所有变量
            if len(expr.variables) > 1:
                s += self._print(expr.variables)
            # 否则输出单个变量
            elif expr.variables:
                s += self._print(expr.variables[0])
            # 输出箭头符号指示
            s += r'\rightarrow '
            # 如果点数量大于1，则输出所有点
            if len(expr.point) > 1:
                s += self._print(expr.point)
            # 否则输出单个点
            else:
                s += self._print(expr.point[0])
        # 构建 LaTeX 表示式 O(s)
        return r"O\left(%s\right)" % s
    # 打印符号表达式的名称，支持不同的样式（默认为普通样式）
    def _print_Symbol(self, expr: Symbol, style='plain'):
        # 尝试从设置中获取符号表达式的名称
        name: str = self._settings['symbol_names'].get(expr)
        # 如果找到名称，则直接返回
        if name is not None:
            return name

        # 否则，调用处理上下标的函数来处理符号表达式的名称
        return self._deal_with_super_sub(expr.name, style=style)

    # 将打印随机符号的函数别名为打印符号的函数
    _print_RandomSymbol = _print_Symbol

    # 处理包含上下标的字符串，根据样式进行处理并返回处理后的字符串
    def _deal_with_super_sub(self, string: str, style='plain') -> str:
        # 如果字符串中包含 '{'，则表明没有上下标
        if '{' in string:
            name, supers, subs = string, [], []
        else:
            # 否则，调用函数分离出上下标
            name, supers, subs = split_super_sub(string)

            # 对名称、上标、下标进行翻译处理
            name = translate(name)
            supers = [translate(sup) for sup in supers]
            subs = [translate(sub) for sub in subs]

        # 根据样式应用修饰，将上标和下标附加到名称后面
        if style == 'bold':
            name = "\\mathbf{{{}}}".format(name)

        # 拼接名称、上标和下标成为最终的字符串
        if supers:
            name += "^{%s}" % " ".join(supers)
        if subs:
            name += "_{%s}" % " ".join(subs)

        return name

    # 打印关系表达式，根据设置返回正确的 TeX 表示
    def _print_Relational(self, expr):
        # 根据设置选择 TeX 表示的大于和小于符号
        if self._settings['itex']:
            gt = r"\gt"
            lt = r"\lt"
        else:
            gt = ">"
            lt = "<"

        # 定义字符映射关系
        charmap = {
            "==": "=",
            ">": gt,
            "<": lt,
            ">=": r"\geq",
            "<=": r"\leq",
            "!=": r"\neq",
        }

        # 返回格式化后的表达式字符串
        return "%s %s %s" % (self._print(expr.lhs),
                             charmap[expr.rel_op], self._print(expr.rhs))

    # 打印分段函数表达式的 TeX 表示
    def _print_Piecewise(self, expr):
        # 将分段函数的每个条件和表达式组合成字符串对
        ecpairs = [r"%s & \text{for}\: %s" % (self._print(e), self._print(c))
                   for e, c in expr.args[:-1]]
        # 处理最后一个条件，如果是 true，则表示无条件的表达式
        if expr.args[-1].cond == true:
            ecpairs.append(r"%s & \text{otherwise}" %
                           self._print(expr.args[-1].expr))
        else:
            ecpairs.append(r"%s & \text{for}\: %s" %
                           (self._print(expr.args[-1].expr),
                            self._print(expr.args[-1].cond)))
        # 构建分段函数的 TeX 表示
        tex = r"\begin{cases} %s \end{cases}"
        return tex % r" \\".join(ecpairs)

    # 打印矩阵内容的 TeX 表示
    def _print_matrix_contents(self, expr):
        # 初始化行列表
        lines = []

        # 遍历矩阵的每一行
        for line in range(expr.rows):  # horrible, should be 'rows'
            # 将每行中的每个元素打印为字符串，并用 '&' 连接
            lines.append(" & ".join([self._print(i) for i in expr[line, :]]))

        # 确定矩阵的显示模式（mat_str），如果未设置则根据模式选择适当的默认值
        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.cols <= 10) is True:
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'

        # 根据矩阵显示模式生成最终的 TeX 字符串
        out_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        out_str = out_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            out_str = out_str.replace('%s', '{' + 'c'*expr.cols + '}%s')
        return out_str % r"\\".join(lines)
    # 打印 MatrixBase 表达式的字符串表示
    def _print_MatrixBase(self, expr):
        # 调用内部方法获取矩阵内容的字符串表示
        out_str = self._print_matrix_contents(expr)
        # 如果设置中包含矩阵分隔符
        if self._settings['mat_delim']:
            # 获取左右分隔符
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            # 添加左右分隔符到输出字符串两侧
            out_str = r'\left' + left_delim + out_str + \
                      r'\right' + right_delim
        # 返回最终的输出字符串
        return out_str

    # 打印 MatrixElement 表达式的字符串表示
    def _print_MatrixElement(self, expr):
        # 获取包围矩阵元素的部分
        matrix_part = self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True)
        # 获取索引部分的字符串表示
        index_part = f"{self._print(expr.i)},{self._print(expr.j)}"
        # 返回格式化后的字符串表示
        return f"{{{matrix_part}}}_{{{index_part}}}"

    # 打印 MatrixSlice 表达式的字符串表示
    def _print_MatrixSlice(self, expr):
        # 定义内部函数，用于生成 LaTeX 切片表示
        def latexslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = None
            if x[1] == dim:
                x[1] = None
            # 生成每个维度的字符串表示，使用':'分隔
            return ':'.join(self._print(xi) if xi is not None else '' for xi in x)
        # 构建最终的 LaTeX 表达式字符串
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + r'\left[' +
                latexslice(expr.rowslice, expr.parent.rows) + ', ' +
                latexslice(expr.colslice, expr.parent.cols) + r'\right]')

    # 打印 BlockMatrix 表达式的字符串表示
    def _print_BlockMatrix(self, expr):
        # 直接返回 BlockMatrix 对象的打印字符串
        return self._print(expr.blocks)

    # 打印 Transpose 表达式的字符串表示
    def _print_Transpose(self, expr):
        # 获取 Transpose 的参数矩阵
        mat = expr.arg
        # 导入必要的类
        from sympy.matrices import MatrixSymbol, BlockMatrix
        # 如果参数不是 MatrixSymbol、BlockMatrix 或者是 MatrixExpr
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            # 返回带括号的参数矩阵的转置
            return r"\left(%s\right)^{T}" % self._print(mat)
        else:
            # 对参数矩阵进行括号化处理，并添加转置符号
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return r"\left(%s\right)^{T}" % s
            else:
                return "%s^{T}" % s

    # 打印 Trace 表达式的字符串表示
    def _print_Trace(self, expr):
        # 获取 Trace 的参数矩阵
        mat = expr.arg
        # 返回带迹号的参数矩阵的字符串表示
        return r"\operatorname{tr}\left(%s \right)" % self._print(mat)

    # 打印 Adjoint 表达式的字符串表示
    def _print_Adjoint(self, expr):
        # 定义不同风格的共轭转置符号
        style_to_latex = {
            "dagger"   : r"\dagger",
            "star"     : r"\ast",
            "hermitian": r"\mathsf{H}"
        }
        # 获取当前设置中的共轭转置风格
        adjoint_style = style_to_latex.get(self._settings["adjoint_style"], r"\dagger")
        # 获取 Adjoint 的参数矩阵
        mat = expr.arg
        # 导入必要的类
        from sympy.matrices import MatrixSymbol, BlockMatrix
        # 如果参数不是 MatrixSymbol、BlockMatrix 或者是 MatrixExpr
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            # 返回带括号的参数矩阵的共轭转置表示
            return r"\left(%s\right)^{%s}" % (self._print(mat), adjoint_style)
        else:
            # 对参数矩阵进行括号化处理，并添加共轭转置符号
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return r"\left(%s\right)^{%s}" % (s, adjoint_style)
            else:
                return r"%s^{%s}" % (s, adjoint_style)
    # 定义一个方法，用于打印 MatMul 对象的表达式
    def _print_MatMul(self, expr):
        # 导入 sympy 中的 MatMul 类
        from sympy import MatMul
        
        # 定义一个 lambda 函数用于给表达式添加括号，
        # 对于 Mul 对象而言，只有在不是 MatMul 类型时才添加括号
        parens = lambda x: self._print(x) if isinstance(x, Mul) and not isinstance(x, MatMul) else \
            self.parenthesize(x, precedence_traditional(expr), False)
        
        # 获取表达式的所有参数
        args = list(expr.args)
        
        # 如果表达式能够提取负号
        if expr.could_extract_minus_sign():
            # 如果第一个参数是 -1，则从参数列表中移除第一个参数
            if args[0] == -1:
                args = args[1:]
            else:
                # 否则将第一个参数变为负数
                args[0] = -args[0]
            
            # 返回带有负号的参数表达式
            return '- ' + ' '.join(map(parens, args))
        else:
            # 返回正常的参数表达式
            return ' '.join(map(parens, args))

    # 定义一个方法，用于打印 Determinant 对象的表达式
    def _print_Determinant(self, expr):
        # 获取表达式的矩阵参数
        mat = expr.arg
        
        # 如果参数是一个矩阵表达式
        if mat.is_MatrixExpr:
            # 导入 BlockMatrix 类
            from sympy.matrices.expressions.blockmatrix import BlockMatrix
            
            # 如果矩阵是 BlockMatrix 类型，则打印其矩阵内容
            if isinstance(mat, BlockMatrix):
                return r"\left|{%s}\right|" % self._print_matrix_contents(mat.blocks)
            
            # 否则直接打印矩阵表达式
            return r"\left|{%s}\right|" % self._print(mat)
        
        # 如果参数不是矩阵表达式，则直接打印其内容
        return r"\left|{%s}\right|" % self._print_matrix_contents(mat)

    # 定义一个方法，用于打印 Mod 对象的表达式
    def _print_Mod(self, expr, exp=None):
        # 如果指定了指数参数
        if exp is not None:
            # 返回带有括号的 Mod 表达式及其指数
            return r'\left(%s \bmod %s\right)^{%s}' % \
                (self.parenthesize(expr.args[0], PRECEDENCE['Mul'], strict=True),
                 self.parenthesize(expr.args[1], PRECEDENCE['Mul'], strict=True),
                 exp)
        
        # 如果没有指定指数参数，则返回普通的 Mod 表达式
        return r'%s \bmod %s' % (self.parenthesize(expr.args[0], PRECEDENCE['Mul'], strict=True),
                                 self.parenthesize(expr.args[1], PRECEDENCE['Mul'], strict=True))

    # 定义一个方法，用于打印 HadamardProduct 对象的表达式
    def _print_HadamardProduct(self, expr):
        # 获取表达式的所有参数
        args = expr.args
        
        # 定义优先级和括号化方法
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize
        
        # 返回用圆点表示的 Hadamard 乘积表达式
        return r' \circ '.join((parens(arg, prec, strict=True) for arg in args))

    # 定义一个方法，用于打印 HadamardPower 对象的表达式
    def _print_HadamardPower(self, expr):
        # 如果指数的传统优先级小于 Mul 的优先级
        if precedence_traditional(expr.exp) < PRECEDENCE["Mul"]:
            template = r"%s^{\circ \left({%s}\right)}"
        else:
            template = r"%s^{\circ {%s}}"
        
        # 调用辅助方法打印标准的幂次方表达式
        return self._helper_print_standard_power(expr, template)

    # 定义一个方法，用于打印 KroneckerProduct 对象的表达式
    def _print_KroneckerProduct(self, expr):
        # 获取表达式的所有参数
        args = expr.args
        
        # 定义优先级和括号化方法
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize
        
        # 返回用 Kronecker 乘积符号表示的表达式
        return r' \otimes '.join((parens(arg, prec, strict=True) for arg in args))
    # 打印矩阵乘幂表达式
    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        # 检查 base 是否为 MatrixSymbol 类型且是 MatrixExpr 类型
        if not isinstance(base, MatrixSymbol) and base.is_MatrixExpr:
            # 如果不是 MatrixSymbol 但是是 MatrixExpr，则返回格式化的矩阵乘幂表达式
            return "\\left(%s\\right)^{%s}" % (self._print(base),
                                              self._print(exp))
        else:
            base_str = self._print(base)
            # 如果 base_str 中包含 '^'，则添加外层括号并打印矩阵乘幂表达式
            if '^' in base_str:
                return r"\left(%s\right)^{%s}" % (base_str, self._print(exp))
            else:
                # 否则直接打印矩阵乘幂表达式
                return "%s^{%s}" % (base_str, self._print(exp))
    
    # 打印矩阵符号
    def _print_MatrixSymbol(self, expr):
        return self._print_Symbol(expr, style=self._settings[
            'mat_symbol_style'])
    
    # 打印零矩阵
    def _print_ZeroMatrix(self, Z):
        # 如果设置为 plain 模式，则返回 "0"，否则返回 LaTeX 格式的零矩阵
        return "0" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{0}"
    
    # 打印单位矩阵
    def _print_OneMatrix(self, O):
        # 如果设置为 plain 模式，则返回 "1"，否则返回 LaTeX 格式的单位矩阵
        return "1" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{1}"
    
    # 打印单位矩阵（Identity）
    def _print_Identity(self, I):
        # 如果设置为 plain 模式，则返回 LaTeX 格式的黑板粗体 I，否则返回 LaTeX 格式的单位矩阵
        return r"\mathbb{I}" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{I}"
    
    # 打印排列矩阵（PermutationMatrix）
    def _print_PermutationMatrix(self, P):
        perm_str = self._print(P.args[0])
        # 返回格式化的排列矩阵表达式
        return "P_{%s}" % perm_str
    # 定义一个方法用于打印 N 维数组表达式
    def _print_NDimArray(self, expr: NDimArray):
        
        # 如果数组的秩为 0，直接打印表达式的值
        if expr.rank() == 0:
            return self._print(expr[()])
        
        # 获取矩阵字符串表示，如果未设置，则根据模式和数组大小选择默认值
        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.rank() == 0) or (expr.shape[-1] <= 10):
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'
        
        # 构建块字符串模板，并根据数组类型调整
        block_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        block_str = block_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            block_str = block_str.replace('%s', '{' + 'c'*expr.shape[0] + '}%s')
        
        # 如果设置了矩阵分隔符，则添加左右分隔符
        if self._settings['mat_delim']:
            left_delim: str = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            block_str = r'\left' + left_delim + block_str + \
                        r'\right' + right_delim
        
        # 如果数组秩为 0，直接返回块字符串
        if expr.rank() == 0:
            return block_str % ""
        
        # 初始化用于存储不同层级字符串表示的列表
        level_str: list[list[str]] = [[] for i in range(expr.rank() + 1)]
        shape_ranges = [list(range(i)) for i in expr.shape]
        
        # 使用 itertools.product 生成所有可能的索引组合
        for outer_i in itertools.product(*shape_ranges):
            # 将每个元素的打印结果添加到对应层级的列表中
            level_str[-1].append(self._print(expr[outer_i]))
            even = True
            # 逆序处理每个维度，组合成块字符串
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(
                        r" & ".join(level_str[back_outer_i+1]))
                else:
                    level_str[back_outer_i].append(
                        block_str % (r"\\".join(level_str[back_outer_i+1])))
                    if len(level_str[back_outer_i+1]) == 1:
                        level_str[back_outer_i][-1] = r"\left[" + \
                            level_str[back_outer_i][-1] + r"\right]"
                even = not even
                level_str[back_outer_i+1] = []
        
        # 最终得到的字符串表示即为最外层的第一个元素
        out_str = level_str[0][0]
        
        # 如果数组的秩为奇数，使用块字符串包裹输出字符串
        if expr.rank() % 2 == 1:
            out_str = block_str % out_str
        
        # 返回最终的输出字符串
        return out_str
    # 打印张量索引信息
    def _printer_tensor_indices(self, name, indices, index_map: dict):
        # 将张量名称打印为字符串
        out_str = self._print(name)
        # 初始化上一个张量阶数和前一个映射
        last_valence = None
        prev_map = None
        # 遍历所有的索引
        for index in indices:
            # 获取当前索引的阶数
            new_valence = index.is_up
            # 如果当前索引在索引映射中，或者前一个映射存在，并且与上一个张量阶数相同，则添加逗号
            if ((index in index_map) or prev_map) and \
                    last_valence == new_valence:
                out_str += ","
            # 如果上一个张量阶数与当前索引的阶数不同
            if last_valence != new_valence:
                # 关闭当前张量阶数的打印
                if last_valence is not None:
                    out_str += "}"
                # 根据当前索引的阶数，添加相应的开始打印标记
                if index.is_up:
                    out_str += "{}^{"
                else:
                    out_str += "{}_{"
            # 将索引的第一个参数打印为字符串
            out_str += self._print(index.args[0])
            # 如果当前索引在索引映射中，添加等号和映射后的值打印
            if index in index_map:
                out_str += "="
                out_str += self._print(index_map[index])
                prev_map = True
            else:
                prev_map = False
            # 更新上一个张量阶数为当前阶数
            last_valence = new_valence
        # 关闭最后一个张量阶数的打印
        if last_valence is not None:
            out_str += "}"
        # 返回所有打印的字符串
        return out_str

    # 打印张量表达式
    def _print_Tensor(self, expr):
        # 获取张量的名称、索引和空的索引映射，然后调用打印张量索引信息函数
        name = expr.args[0].args[0]
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices, {})

    # 打印张量元素表达式
    def _print_TensorElement(self, expr):
        # 获取张量元素的名称、索引和索引映射，然后调用打印张量索引信息函数
        name = expr.expr.args[0].args[0]
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    # 打印张量乘积表达式
    def _print_TensMul(self, expr):
        # 打印类似 "A(a)", "3*A(a)", "(1+x)*A(a)" 的表达式
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    # 打印张量加法表达式
    def _print_TensAdd(self, expr):
        # 打印张量加法表达式，并确保表达式以合适的方式显示负号
        a = []
        args = expr.args
        for x in args:
            a.append(self.parenthesize(x, precedence(expr)))
        a.sort()
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    # 打印张量索引
    def _print_TensorIndex(self, expr):
        # 返回形如 "{}^{%s}{%s}" 或 "{}_{%s}{%s}" 的字符串，根据索引的上下标决定
        return "{}%s{%s}" % (
            "^" if expr.is_up else "_",
            self._print(expr.args[0])
        )

    # 打印偏微分表达式
    def _print_PartialDerivative(self, expr):
        # 根据表达式的变量数目，打印偏微分的 LaTeX 表示
        if len(expr.variables) == 1:
            return r"\frac{\partial}{\partial {%s}}{%s}" % (
                self._print(expr.variables[0]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )
        else:
            return r"\frac{\partial^{%s}}{%s}{%s}" % (
                len(expr.variables),
                " ".join([r"\partial {%s}" % self._print(i) for i in expr.variables]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )

    # 打印数组符号表达式
    def _print_ArraySymbol(self, expr):
        # 返回数组符号的名称字符串表示
        return self._print(expr.name)

    # 打印数组元素表达式
    def _print_ArrayElement(self, expr):
        # 返回数组元素的字符串表示，形如 "{{%s}_{%s}}"
        return "{{%s}_{%s}}" % (
            self.parenthesize(expr.name, PRECEDENCE["Func"], True),
            ", ".join([f"{self._print(i)}" for i in expr.indices]))
    def _print_UniversalSet(self, expr):
        # 返回数学符号表示通用集合
        return r"\mathbb{U}"

    def _print_frac(self, expr, exp=None):
        # 根据指定的指数打印分数表达式
        if exp is None:
            return r"\operatorname{frac}{\left(%s\right)}" % self._print(expr.args[0])
        else:
            return r"\operatorname{frac}{\left(%s\right)}^{%s}" % (
                    self._print(expr.args[0]), exp)

    def _print_tuple(self, expr):
        # 根据设置的小数分隔符打印元组表达式
        if self._settings['decimal_separator'] == 'comma':
            sep = ";"
        elif self._settings['decimal_separator'] == 'period':
            sep = ","
        else:
            raise ValueError('Unknown Decimal Separator')

        if len(expr) == 1:
            # 对于单个元素的元组，需要添加尾随分隔符
            return self._add_parens_lspace(self._print(expr[0]) + sep)
        else:
            # 对于多个元素的元组，用分隔符连接所有元素
            return self._add_parens_lspace(
                (sep + r" \  ").join([self._print(i) for i in expr]))

    def _print_TensorProduct(self, expr):
        # 打印张量积表达式
        elements = [self._print(a) for a in expr.args]
        return r' \otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        # 打印外积表达式
        elements = [self._print(a) for a in expr.args]
        return r' \wedge '.join(elements)

    def _print_Tuple(self, expr):
        # 调用打印元组的方法
        return self._print_tuple(expr)

    def _print_list(self, expr):
        # 根据设置的小数分隔符打印列表表达式
        if self._settings['decimal_separator'] == 'comma':
            return r"\left[ %s\right]" % \
                r"; \  ".join([self._print(i) for i in expr])
        elif self._settings['decimal_separator'] == 'period':
            return r"\left[ %s\right]" % \
                r", \  ".join([self._print(i) for i in expr])
        else:
            raise ValueError('Unknown Decimal Separator')

    def _print_dict(self, d):
        # 打印字典表达式，按键排序
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left\{ %s\right\}" % r", \  ".join(items)

    def _print_Dict(self, expr):
        # 调用打印字典的方法
        return self._print_dict(expr)

    def _print_DiracDelta(self, expr, exp=None):
        # 打印狄拉克 delta 函数表达式
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = r"\delta\left(%s\right)" % self._print(expr.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left( %s \right)" % (
                self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    def _print_SingularityFunction(self, expr, exp=None):
        # 打印奇异函数表达式
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = r"{\left\langle %s \right\rangle}^{%s}" % (shift, power)
        if exp is not None:
            tex = r"{\left({\langle %s \rangle}^{%s}\right)}^{%s}" % (shift, power, exp)
        return tex
    # 输出 Heaviside 函数的 LaTeX 表示
    def _print_Heaviside(self, expr, exp=None):
        # 将表达式中的参数转换为字符串，用逗号分隔
        pargs = ', '.join(self._print(arg) for arg in expr.pargs)
        # 构建 Heaviside 函数的 LaTeX 表示
        tex = r"\theta\left(%s\right)" % pargs
        # 如果有指数部分，则添加指数表示
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    # 输出 Kronecker δ 函数的 LaTeX 表示
    def _print_KroneckerDelta(self, expr, exp=None):
        # 获取 Kronecker δ 函数的参数 i 和 j 的打印表示
        i = self._print(expr.args[0])
        j = self._print(expr.args[1])
        # 根据参数是否为原子表示选择合适的 LaTeX 格式
        if expr.args[0].is_Atom and expr.args[1].is_Atom:
            tex = r'\delta_{%s %s}' % (i, j)
        else:
            tex = r'\delta_{%s, %s}' % (i, j)
        # 如果有指数部分，则添加指数表示
        if exp is not None:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    # 输出 Levi-Civita 符号的 LaTeX 表示
    def _print_LeviCivita(self, expr, exp=None):
        # 将 Levi-Civita 符号的参数转换为字符串列表
        indices = map(self._print, expr.args)
        # 根据参数是否全部为原子表示选择合适的 LaTeX 格式
        if all(x.is_Atom for x in expr.args):
            tex = r'\varepsilon_{%s}' % " ".join(indices)
        else:
            tex = r'\varepsilon_{%s}' % ", ".join(indices)
        # 如果有指数部分，则添加指数表示
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    # 输出随机变量的定义域的 LaTeX 表示
    def _print_RandomDomain(self, d):
        # 根据定义域的类型输出对应的 LaTeX 表示
        if hasattr(d, 'as_boolean'):
            return '\\text{Domain: }' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('\\text{Domain: }' + self._print(d.symbols) + ' \\in ' +
                    self._print(d.set))
        elif hasattr(d, 'symbols'):
            return '\\text{Domain on }' + self._print(d.symbols)
        else:
            return self._print(None)

    # 输出有限集合的 LaTeX 表示
    def _print_FiniteSet(self, s):
        # 对有限集合的元素按默认排序键排序
        items = sorted(s.args, key=default_sort_key)
        # 调用 _print_set 方法输出集合的 LaTeX 表示
        return self._print_set(items)

    # 输出集合的 LaTeX 表示
    def _print_set(self, s):
        # 对集合按默认排序键排序
        items = sorted(s, key=default_sort_key)
        # 根据设置的小数分隔符选择适当的格式输出集合的元素
        if self._settings['decimal_separator'] == 'comma':
            items = "; ".join(map(self._print, items))
        elif self._settings['decimal_separator'] == 'period':
            items = ", ".join(map(self._print, items))
        else:
            raise ValueError('Unknown Decimal Separator')
        # 返回包含集合元素的 LaTeX 表示
        return r"\left\{%s\right\}" % items

    # 输出 frozenset 的 LaTeX 表示，与 _print_set 方法相同
    _print_frozenset = _print_set
    def _print_Range(self, s):
        def _print_symbolic_range():
            # 定义一个处理无法解析的符号范围的内部函数
            if s.args[0] == 0:
                # 如果范围起始值为0
                if s.args[2] == 1:
                    # 如果步长为1
                    cont = self._print(s.args[1])
                else:
                    # 否则，将所有参数打印成字符串并以逗号分隔
                    cont = ", ".join(self._print(arg) for arg in s.args)
            else:
                # 如果起始值不为0
                if s.args[2] == 1:
                    # 如果步长为1，打印起始值和结束值
                    cont = ", ".join(self._print(arg) for arg in s.args[:2])
                else:
                    # 否则，将所有参数打印成字符串并以逗号分隔
                    cont = ", ".join(self._print(arg) for arg in s.args)

            return(f"\\text{{Range}}\\left({cont}\\right)")

        dots = object()

        if s.start.is_infinite and s.stop.is_infinite:
            # 如果起始和结束值都是无穷大
            if s.step.is_positive:
                # 如果步长为正数
                printset = dots, -1, 0, 1, dots
            else:
                # 如果步长不是正数，以逆序打印
                printset = dots, 1, 0, -1, dots
        elif s.start.is_infinite:
            # 如果起始值为无穷大
            printset = dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite:
            # 如果结束值为无穷大
            it = iter(s)
            printset = next(it), next(it), dots
        elif s.is_empty is not None:
            # 如果范围为空
            if (s.size < 4) == True:
                # 如果范围的大小小于4
                printset = tuple(s)
            elif s.is_iterable:
                # 如果范围可迭代
                it = iter(s)
                printset = next(it), next(it), dots, s[-1]
            else:
                # 否则，调用内部函数处理无法解析的符号范围
                return _print_symbolic_range()
        else:
            # 如果以上条件都不符合，调用内部函数处理无法解析的符号范围
            return _print_symbolic_range()
        # 返回打印出的范围集合字符串
        return (r"\left\{" +
                r", ".join(self._print(el) if el is not dots else r'\ldots' for el in printset) +
                r"\right\}")

    def __print_number_polynomial(self, expr, letter, exp=None):
        # 打印数字多项式的 LaTeX 表达式
        if len(expr.args) == 2:
            # 如果表达式参数长度为2
            if exp is not None:
                # 如果指数参数不为空，返回带指数的多项式 LaTeX 表达式
                return r"%s_{%s}^{%s}\left(%s\right)" % (letter,
                            self._print(expr.args[0]), exp,
                            self._print(expr.args[1]))
            # 否则，返回不带指数的多项式 LaTeX 表达式
            return r"%s_{%s}\left(%s\right)" % (letter,
                        self._print(expr.args[0]), self._print(expr.args[1]))

        # 对于其他情况，返回数字多项式的 LaTeX 表达式
        tex = r"%s_{%s}" % (letter, self._print(expr.args[0]))
        if exp is not None:
            # 如果指数参数不为空，添加指数到 LaTeX 表达式中
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_bernoulli(self, expr, exp=None):
        # 打印伯努利数的 LaTeX 表达式
        return self.__print_number_polynomial(expr, "B", exp)

    def _print_genocchi(self, expr, exp=None):
        # 打印赫诺奇数的 LaTeX 表达式
        return self.__print_number_polynomial(expr, "G", exp)

    def _print_bell(self, expr, exp=None):
        # 打印贝尔多项式的 LaTeX 表达式
        if len(expr.args) == 3:
            # 如果表达式参数长度为3
            tex1 = r"B_{%s, %s}" % (self._print(expr.args[0]),
                                self._print(expr.args[1]))
            tex2 = r"\left(%s\right)" % r", ".join(self._print(el) for
                                               el in expr.args[2])
            if exp is not None:
                # 如果指数参数不为空，返回带指数的贝尔多项式 LaTeX 表达式
                tex = r"%s^{%s}%s" % (tex1, exp, tex2)
            else:
                # 否则，返回不带指数的贝尔多项式 LaTeX 表达式
                tex = tex1 + tex2
            return tex
        # 对于其他情况，返回数字多项式的 LaTeX 表达式
        return self.__print_number_polynomial(expr, "B", exp)
    def _print_fibonacci(self, expr, exp=None):
        # 调用私有方法 __print_number_polynomial，用于打印斐波那契数
        return self.__print_number_polynomial(expr, "F", exp)

    def _print_lucas(self, expr, exp=None):
        # 构造 LaTeX 表示的 Lucas 数，根据指数 exp 和参数表达式 expr
        tex = r"L_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_tribonacci(self, expr, exp=None):
        # 调用私有方法 __print_number_polynomial，用于打印 Tribonacci 数
        return self.__print_number_polynomial(expr, "T", exp)

    def _print_mobius(self, expr, exp=None):
        # 打印莫比乌斯函数，根据指数 exp 和参数表达式 expr
        if exp is None:
            return r'\mu\left(%s\right)' % self._print(expr.args[0])
        return r'\mu^{%s}\left(%s\right)' % (exp, self._print(expr.args[0]))

    def _print_SeqFormula(self, s):
        # 打印数列公式，根据数列 s 的不同特性进行格式化输出
        dots = object()
        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            # 如果起始或结束有自由符号，则格式化为带限制条件的数列
            return r"\left\{%s\right\}_{%s=%s}^{%s}" % (
                self._print(s.formula),
                self._print(s.variables[0]),
                self._print(s.start),
                self._print(s.stop)
            )
        if s.start is S.NegativeInfinity:
            # 如果起始为负无穷，则输出从结尾向前推算的有限项
            stop = s.stop
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2),
                        s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            # 如果结束为正无穷或数列长度大于4，则输出前4项加省略号
            printset = s[:4]
            printset.append(dots)
        else:
            # 否则输出数列的所有元素
            printset = tuple(s)

        return (r"\left[" +
                r", ".join(self._print(el) if el is not dots else r'\ldots' for el in printset) +
                r"\right]")

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_Interval(self, i):
        # 打印数学区间，根据区间的起始、结束值及开闭性进行格式化输出
        if i.start == i.end:
            return r"\left\{%s\right\}" % self._print(i.start)
        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return r"\left%s%s, %s\right%s" % \
                   (left, self._print(i.start), self._print(i.end), right)

    def _print_AccumulationBounds(self, i):
        # 打印积分区间的界限，用尖括号表示
        return r"\left\langle %s, %s\right\rangle" % \
                (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        # 打印集合并操作，用 \cup 符号连接各子集
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cup ".join(args_str)

    def _print_Complement(self, u):
        # 打印集合补操作，用 \setminus 符号连接各子集
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \setminus ".join(args_str)

    def _print_Intersection(self, u):
        # 打印集合交操作，用 \cap 符号连接各子集
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cap ".join(args_str)

    def _print_SymmetricDifference(self, u):
        # 打印集合对称差操作，用 \triangle 符号连接各子集
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \triangle ".join(args_str)
    def _print_ProductSet(self, p):
        # 确定打印优先级
        prec = precedence_traditional(p)
        # 如果集合 p 中至少有一个元素且没有多样性，则返回第一个集合的打印形式加上指数
        if len(p.sets) >= 1 and not has_variety(p.sets):
            return self.parenthesize(p.sets[0], prec) + "^{%d}" % len(p.sets)
        # 否则，以 \times 连接所有集合的打印形式
        return r" \times ".join(
            self.parenthesize(set, prec) for set in p.sets)

    def _print_EmptySet(self, e):
        # 返回空集的 LaTeX 表示
        return r"\emptyset"

    def _print_Naturals(self, n):
        # 返回自然数集的 LaTeX 表示
        return r"\mathbb{N}"

    def _print_Naturals0(self, n):
        # 返回包括零的自然数集的 LaTeX 表示
        return r"\mathbb{N}_0"

    def _print_Integers(self, i):
        # 返回整数集的 LaTeX 表示
        return r"\mathbb{Z}"

    def _print_Rationals(self, i):
        # 返回有理数集的 LaTeX 表示
        return r"\mathbb{Q}"

    def _print_Reals(self, i):
        # 返回实数集的 LaTeX 表示
        return r"\mathbb{R}"

    def _print_Complexes(self, i):
        # 返回复数集的 LaTeX 表示
        return r"\mathbb{C}"

    def _print_ImageSet(self, s):
        # 获取映射表达式和签名
        expr = s.lamda.expr
        sig = s.lamda.signature
        # 构建形如 {expr | x1 \in S1, x2 \in S2, ...} 的集合表达式
        xys = ((self._print(x), self._print(y)) for x, y in zip(sig, s.base_sets))
        xinys = r", ".join(r"%s \in %s" % xy for xy in xys)
        return r"\left\{%s\; \middle|\; %s\right\}" % (self._print(expr), xinys)

    def _print_ConditionSet(self, s):
        # 获取符号集合的打印形式
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        # 如果基础集合是 UniversalSet，则返回形如 {vars | condition} 的集合表示
        if s.base_set is S.UniversalSet:
            return r"\left\{%s\; \middle|\; %s \right\}" % \
                (vars_print, self._print(s.condition))
        # 否则返回形如 {vars | vars in base_set ∧ condition} 的集合表示
        return r"\left\{%s\; \middle|\; %s \in %s \wedge %s \right\}" % (
            vars_print,
            vars_print,
            self._print(s.base_set),
            self._print(s.condition))

    def _print_PowerSet(self, expr):
        # 获取参数的打印形式
        arg_print = self._print(expr.args[0])
        # 返回形如 \mathcal{P}(arg_print) 的幂集的 LaTeX 表示
        return r"\mathcal{{P}}\left({}\right)".format(arg_print)

    def _print_ComplexRegion(self, s):
        # 获取变量的打印形式
        vars_print = ', '.join([self._print(var) for var in s.variables])
        # 返回形如 {expr | vars in sets} 的复数区域的 LaTeX 表示
        return r"\left\{%s\; \middle|\; %s \in %s \right\}" % (
            self._print(s.expr),
            vars_print,
            self._print(s.sets))

    def _print_Contains(self, e):
        # 返回形如 expr in set 的成员关系的 LaTeX 表示
        return r"%s \in %s" % tuple(self._print(a) for a in e.args)

    def _print_FourierSeries(self, s):
        # 如果 Fourier 系数都是零，则只打印常数项
        if s.an.formula is S.Zero and s.bn.formula is S.Zero:
            return self._print(s.a0)
        # 否则打印 Fourier 系数的截断形式
        return self._print_Add(s.truncate()) + r' + \ldots'

    def _print_FormalPowerSeries(self, s):
        # 打印形式幂级数的无限和
        return self._print_Add(s.infinite)

    def _print_FiniteField(self, expr):
        # 返回有限域的 LaTeX 表示
        return r"\mathbb{F}_{%s}" % expr.mod

    def _print_IntegerRing(self, expr):
        # 返回整数环的 LaTeX 表示
        return r"\mathbb{Z}"

    def _print_RationalField(self, expr):
        # 返回有理数域的 LaTeX 表示
        return r"\mathbb{Q}"

    def _print_RealField(self, expr):
        # 返回实数域的 LaTeX 表示
        return r"\mathbb{R}"

    def _print_ComplexField(self, expr):
        # 返回复数域的 LaTeX 表示
        return r"\mathbb{C}"

    def _print_PolynomialRing(self, expr):
        # 获取定义域和符号的打印形式
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        # 返回形如 domain[symbols] 的多项式环的 LaTeX 表示
        return r"%s\left[%s\right]" % (domain, symbols)
    # 定义一个方法用于打印分式域字段表达式的 LaTeX 代码
    def _print_FractionField(self, expr):
        # 打印分式域的域信息
        domain = self._print(expr.domain)
        # 将符号列表转换为逗号分隔的字符串，并打印每个符号的 LaTeX 代码
        symbols = ", ".join(map(self._print, expr.symbols))
        # 返回格式化后的 LaTeX 代码，包含域信息和符号列表
        return r"%s\left(%s\right)" % (domain, symbols)

    # 定义一个方法用于打印多项式环的基本表达式的 LaTeX 代码
    def _print_PolynomialRingBase(self, expr):
        # 打印多项式环的域信息
        domain = self._print(expr.domain)
        # 将符号列表转换为逗号分隔的字符串，并打印每个符号的 LaTeX 代码
        symbols = ", ".join(map(self._print, expr.symbols))
        # 初始化一个空字符串表示逆元素（如果不是多项式）
        inv = ""
        # 如果不是多项式，则设置逆元素的 LaTeX 代码
        if not expr.is_Poly:
            inv = r"S_<^{-1}"
        # 返回格式化后的 LaTeX 代码，包含逆元素、域信息和符号列表
        return r"%s%s\left[%s\right]" % (inv, domain, symbols)

    # 定义一个方法用于打印多项式的 LaTeX 代码
    def _print_Poly(self, poly):
        # 获取多项式类名
        cls = poly.__class__.__name__
        # 初始化一个空列表用于存储多项式的每一项的 LaTeX 表示
        terms = []
        # 遍历多项式的每一项和系数
        for monom, coeff in poly.terms():
            # 初始化空字符串表示单项式的 LaTeX 表示
            s_monom = ''
            # 遍历单项式的每个指数和对应的变量
            for i, exp in enumerate(monom):
                # 如果指数大于0，则打印变量的 LaTeX 表示
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))

            # 如果系数是加法表达式，则打印加括号
            if coeff.is_Add:
                if s_monom:
                    s_coeff = r"\left(%s\right)" % self._print(coeff)
                else:
                    s_coeff = self._print(coeff)
            else:
                # 如果不是加法表达式，则直接打印系数
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            # 如果没有单项式，则打印系数
            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + " " + s_monom

            # 如果项以减号开头，则添加到项列表
            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        # 如果项列表的第一个元素是加号或减号，则弹出并设置修饰符
        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        # 将项列表合并为一个表达式字符串
        expr = ' '.join(terms)
        # 获取每个变量的 LaTeX 表示列表
        gens = list(map(self._print, poly.gens))
        # 获取多项式的域信息的 LaTeX 表示
        domain = "domain=%s" % self._print(poly.get_domain())

        # 将表达式、变量列表和域信息格式化为一个字符串表示多项式的 LaTeX 代码
        args = ", ".join([expr] + gens + [domain])
        if cls in accepted_latex_functions:
            tex = r"\%s {\left(%s \right)}" % (cls, args)
        else:
            tex = r"\operatorname{%s}{\left( %s \right)}" % (cls, args)

        # 返回多项式的 LaTeX 表示
        return tex

    # 定义一个方法用于打印复数根的 LaTeX 代码
    def _print_ComplexRootOf(self, root):
        # 获取根对象的类名
        cls = root.__class__.__name__
        # 如果是复数根对象，则更改类名为 CRootOf
        if cls == "ComplexRootOf":
            cls = "CRootOf"
        # 获取根表达式的 LaTeX 表示
        expr = self._print(root.expr)
        # 获取根的索引
        index = root.index
        # 根据类名是否在接受的 LaTeX 函数列表中，返回格式化后的 LaTeX 表示
        if cls in accepted_latex_functions:
            return r"\%s {\left(%s, %d\right)}" % (cls, expr, index)
        else:
            return r"\operatorname{%s} {\left(%s, %d\right)}" % (cls, expr,
                                                                 index)
    # 打印表达式的根和求和
    def _print_RootSum(self, expr):
        # 获取表达式的类名
        cls = expr.__class__.__name__
        # 将表达式的 expr 属性打印并存入 args 列表中
        args = [self._print(expr.expr)]

        # 如果表达式的 fun 属性不是 S.IdentityFunction，则打印 fun 属性并添加到 args 列表中
        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        # 如果表达式的类名在 accepted_latex_functions 中，则返回相应的 LaTeX 格式化字符串
        if cls in accepted_latex_functions:
            return r"\%s {\left(%s\right)}" % (cls, ", ".join(args))
        else:
            return r"\operatorname{%s} {\left(%s\right)}" % (cls,
                                                             ", ".join(args))

    # 打印 OrdinalOmega 表达式
    def _print_OrdinalOmega(self, expr):
        return r"\omega"

    # 打印 OmegaPower 表达式
    def _print_OmegaPower(self, expr):
        exp, mul = expr.args
        # 根据指数和系数的不同情况返回相应的 LaTeX 格式化字符串
        if mul != 1:
            if exp != 1:
                return r"{} \omega^{{{}}}".format(mul, exp)
            else:
                return r"{} \omega".format(mul)
        else:
            if exp != 1:
                return r"\omega^{{{}}}".format(exp)
            else:
                return r"\omega"

    # 打印 Ordinal 表达式
    def _print_Ordinal(self, expr):
        # 打印所有参数的表达式并连接成字符串，使用加号连接
        return " + ".join([self._print(arg) for arg in expr.args])

    # 打印 PolyElement 对象
    def _print_PolyElement(self, poly):
        # 获取乘法符号的 LaTeX 表示形式
        mul_symbol = self._settings['mul_symbol_latex']
        # 使用 poly 对象的 str 方法打印 LaTeX 格式的多项式表示
        return poly.str(self, PRECEDENCE, "{%s}^{%d}", mul_symbol)

    # 打印 FracElement 对象
    def _print_FracElement(self, frac):
        # 如果分母为 1，则直接打印分子部分
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            # 否则打印分数的 LaTeX 格式表示
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return r"\frac{%s}{%s}" % (numer, denom)

    # 打印 euler 函数
    def _print_euler(self, expr, exp=None):
        m, x = (expr.args[0], None) if len(expr.args) == 1 else expr.args
        tex = r"E_{%s}" % self._print(m)
        # 根据指数和参数的存在情况，构造相应的 LaTeX 格式化字符串
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        if x is not None:
            tex = r"%s\left(%s\right)" % (tex, self._print(x))
        return tex

    # 打印 catalan 函数
    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        # 根据指数的存在情况，构造相应的 LaTeX 格式化字符串
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    # 打印 UnifiedTransform 对象
    def _print_UnifiedTransform(self, expr, s, inverse=False):
        # 根据是否是反变换，构造相应的 LaTeX 格式化字符串
        return r"\mathcal{{{}}}{}_{{{}}}\left[{}\right]\left({}\right)".format(s, '^{-1}' if inverse else '', self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    # 打印 MellinTransform 函数
    def _print_MellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M')

    # 打印 InverseMellinTransform 函数
    def _print_InverseMellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M', True)

    # 打印 LaplaceTransform 函数
    def _print_LaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L')

    # 打印 InverseLaplaceTransform 函数
    def _print_InverseLaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L', True)

    # 打印 FourierTransform 函数
    def _print_FourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F')

    # 打印 InverseFourierTransform 函数
    def _print_InverseFourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F', True)

    # 打印 SineTransform 函数
    def _print_SineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN')
    # 返回一个表达式的反正弦变换的打印形式
    def _print_InverseSineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN', True)

    # 返回一个表达式的余弦变换的打印形式
    def _print_CosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS')

    # 返回一个表达式的反余弦变换的打印形式
    def _print_InverseCosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS', True)

    # 打印一个多项式对象的字符串表示
    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        # 如果无法转换为 SymPy 对象，则打印多项式的字符串表示
        return self._print(repr(p))

    # 打印一个有理函数对象的字符串表示
    def _print_DMF(self, p):
        return self._print_DMP(p)

    # 打印一个对象的名称为符号的字符串表示
    def _print_Object(self, object):
        return self._print(Symbol(object.name))

    # 打印 Lambert W 函数的字符串表示
    def _print_LambertW(self, expr, exp=None):
        arg0 = self._print(expr.args[0])
        exp = r"^{%s}" % (exp,) if exp is not None else ""
        if len(expr.args) == 1:
            result = r"W%s\left(%s\right)" % (exp, arg0)
        else:
            arg1 = self._print(expr.args[1])
            result = "W{0}_{{{1}}}\\left({2}\\right)".format(exp, arg1, arg0)
        return result

    # 打印期望值操作的字符串表示
    def _print_Expectation(self, expr):
        return r"\operatorname{{E}}\left[{}\right]".format(self._print(expr.args[0]))

    # 打印方差操作的字符串表示
    def _print_Variance(self, expr):
        return r"\operatorname{{Var}}\left({}\right)".format(self._print(expr.args[0]))

    # 打印协方差操作的字符串表示
    def _print_Covariance(self, expr):
        return r"\operatorname{{Cov}}\left({}\right)".format(", ".join(self._print(arg) for arg in expr.args))

    # 打印概率操作的字符串表示
    def _print_Probability(self, expr):
        return r"\operatorname{{P}}\left({}\right)".format(self._print(expr.args[0]))

    # 打印态射对象的字符串表示
    def _print_Morphism(self, morphism):
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        return "%s\\rightarrow %s" % (domain, codomain)

    # 打印传递函数对象的字符串表示
    def _print_TransferFunction(self, expr):
        num, den = self._print(expr.num), self._print(expr.den)
        return r"\frac{%s}{%s}" % (num, den)

    # 打印级数对象的字符串表示
    def _print_Series(self, expr):
        args = list(expr.args)
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                            False)
        return ' '.join(map(parens, args))

    # 打印 MIMO 级数对象的字符串表示
    def _print_MIMOSeries(self, expr):
        from sympy.physics.control.lti import MIMOParallel
        args = list(expr.args)[::-1]
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                             False) if isinstance(x, MIMOParallel) else self._print(x)
        return r"\cdot".join(map(parens, args))

    # 打印并联操作的字符串表示
    def _print_Parallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    # 打印 MIMO 并联操作的字符串表示
    def _print_MIMOParallel(self, expr):
        return ' + '.join(map(self._print, expr.args))
    # 打印符合给定表达式的反馈函数
    def _print_Feedback(self, expr):
        # 从 sympy.physics.control 导入 TransferFunction 和 Series 类
        from sympy.physics.control import TransferFunction, Series

        # 获取表达式的分子和一个默认的传输函数对象
        num, tf = expr.sys1, TransferFunction(1, 1, expr.var)
        # 如果分子是 Series 类型，则转换为参数列表；否则将其放入列表中
        num_arg_list = list(num.args) if isinstance(num, Series) else [num]
        # 如果表达式的 sys2 是 Series 类型，则转换为参数列表；否则将其放入列表中
        den_arg_list = list(expr.sys2.args) if isinstance(expr.sys2, Series) else [expr.sys2]
        # 设置传输函数作为分母的第一个术语
        den_term_1 = tf

        # 根据分子和表达式的类型设置第二个分母术语
        if isinstance(num, Series) and isinstance(expr.sys2, Series):
            den_term_2 = Series(*num_arg_list, *den_arg_list)
        elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
            if expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = tf, Series(*num_arg_list, expr.sys2)
        elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            else:
                den_term_2 = Series(num, *den_arg_list)
        else:
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            elif expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = Series(*num_arg_list, *den_arg_list)

        # 打印分子
        numer = self._print(num)
        # 打印第一个分母术语
        denom_1 = self._print(den_term_1)
        # 打印第二个分母术语
        denom_2 = self._print(den_term_2)
        # 根据符号的正负设置符号字符串
        _sign = "+" if expr.sign == -1 else "-"

        # 返回 LaTeX 格式的表达式字符串
        return r"\frac{%s}{%s %s %s}" % (numer, denom_1, _sign, denom_2)

    # 打印符合给定 MIMO 反馈表达式的字符串
    def _print_MIMOFeedback(self, expr):
        # 从 sympy.physics.control 导入 MIMOSeries 类
        from sympy.physics.control import MIMOSeries
        # 打印 MIMOSeries 对象
        inv_mat = self._print(MIMOSeries(expr.sys2, expr.sys1))
        # 打印表达式的 sys1
        sys1 = self._print(expr.sys1)
        # 根据符号的正负设置符号字符串
        _sign = "+" if expr.sign == -1 else "-"
        # 返回 LaTeX 格式的表达式字符串
        return r"\left(I_{\tau} %s %s\right)^{-1} \cdot %s" % (_sign, inv_mat, sys1)

    # 打印符合给定 TransferFunctionMatrix 表达式的字符串
    def _print_TransferFunctionMatrix(self, expr):
        # 打印表达式中的矩阵
        mat = self._print(expr._expr_mat)
        # 返回 LaTeX 格式的表达式字符串
        return r"%s_\tau" % mat

    # 打印符合给定 DFT 或 IDFT 表达式的字符串
    def _print_DFT(self, expr):
        # 返回以 LaTeX 格式表示的 DFT 或 IDFT 的字符串
        return r"\text{{{}}}_{{{}}}".format(expr.__class__.__name__, expr.n)
    _print_IDFT = _print_DFT  # 将 _print_IDFT 定义为 _print_DFT 的别名

    # 打印符合给定 NamedMorphism 表达式的字符串
    def _print_NamedMorphism(self, morphism):
        # 打印 morphism.name 符号的美观表示
        pretty_name = self._print(Symbol(morphism.name))
        # 打印 NamedMorphism 的美观表示
        pretty_morphism = self._print_Morphism(morphism)
        # 返回字符串形式的 NamedMorphism
        return "%s:%s" % (pretty_name, pretty_morphism)

    # 打印符合给定 IdentityMorphism 表达式的字符串
    def _print_IdentityMorphism(self, morphism):
        # 从 sympy.categories 导入 NamedMorphism 类
        from sympy.categories import NamedMorphism
        # 打印 IdentityMorphism 对象
        return self._print_NamedMorphism(NamedMorphism(
            morphism.domain, morphism.codomain, "id"))
    # 输出组合态射的字符串表示，将所有组件的名称倒序连接起来作为组合态射的名称
    def _print_CompositeMorphism(self, morphism):
        # 收集所有组件的名称列表，通过打印符号将名称转换为字符串
        component_names_list = [self._print(Symbol(component.name)) for
                                component in morphism.components]
        # 将列表倒序排列，并用 "\\circ " 连接各个组件名称，最后加上冒号
        component_names_list.reverse()
        component_names = "\\circ ".join(component_names_list) + ":"

        # 调用打印函数打印整个态射的美观表示
        pretty_morphism = self._print_Morphism(morphism)
        # 返回组合态射的完整字符串表示
        return component_names + pretty_morphism

    # 打印范畴对象的 LaTeX 表示
    def _print_Category(self, morphism):
        # 使用 LaTeX 表示形式包围符号的名称
        return r"\mathbf{{{}}}".format(self._print(Symbol(morphism.name)))

    # 打印图表对象的 LaTeX 表示
    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # 如果图表没有前提，表示为空图表
            return self._print(S.EmptySet)

        # 打印图表的前提部分
        latex_result = self._print(diagram.premises)
        if diagram.conclusions:
            # 如果图表有结论，添加结论的 LaTeX 表示
            latex_result += "\\Longrightarrow %s" % \
                            self._print(diagram.conclusions)

        # 返回完整的图表的 LaTeX 表示
        return latex_result

    # 打印图表格对象的 LaTeX 表示
    def _print_DiagramGrid(self, grid):
        # 构建 LaTeX 表示的表格头
        latex_result = "\\begin{array}{%s}\n" % ("c" * grid.width)

        # 遍历表格每个单元格
        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    # 如果单元格有内容，将内容转换为 LaTeX 表示并添加到结果中
                    latex_result += latex(grid[i, j])
                latex_result += " "
                if j != grid.width - 1:
                    latex_result += "& "

            # 如果不是最后一行，添加换行符
            if i != grid.height - 1:
                latex_result += "\\\\"
            latex_result += "\n"

        # 添加表格尾部的 LaTeX 表示
        latex_result += "\\end{array}\n"
        # 返回完整的表格的 LaTeX 表示
        return latex_result

    # 打印自由模的 LaTeX 表示
    def _print_FreeModule(self, M):
        # 返回自由模的 LaTeX 表示，包括环和秩的打印结果
        return '{{{}}}^{{{}}}'.format(self._print(M.ring), self._print(M.rank))

    # 打印自由模元素的 LaTeX 表示
    def _print_FreeModuleElement(self, m):
        # 打印为行向量，将自由模元素的各个部分连接起来
        return r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in m))

    # 打印子模的 LaTeX 表示
    def _print_SubModule(self, m):
        # 将子模的生成元转换为 LaTeX 格式的表示
        gens = [[self._print(m.ring.to_sympy(x)) for x in g] for g in m.gens]
        curly = lambda o: r"{" + o + r"}"
        square = lambda o: r"\left[ " + o + r" \right]"
        # 构建生成元的 LaTeX 表示字符串
        gens_latex = ",".join(curly(square(",".join(curly(x) for x in g))) for g in gens)
        # 返回完整的子模的 LaTeX 表示
        return r"\left\langle {} \right\rangle".format(gens_latex)

    # 打印子商模的 LaTeX 表示
    def _print_SubQuotientModule(self, m):
        # 将子商模的生成元转换为 LaTeX 格式的表示
        gens_latex = ",".join(["{" + self._print(g) + "}" for g in m.gens])
        # 返回完整的子商模的 LaTeX 表示
        return r"\left\langle {} \right\rangle".format(gens_latex)

    # 打印模实现的理想的 LaTeX 表示
    def _print_ModuleImplementedIdeal(self, m):
        # 将模实现的理想的生成元转换为 LaTeX 格式的表示
        gens = [m.ring.to_sympy(x) for [x] in m._module.gens]
        gens_latex = ",".join('{' + self._print(x) + '}' for x in gens)
        # 返回完整的模实现的理想的 LaTeX 表示
        return r"\left\langle {} \right\rangle".format(gens_latex)
    # 输出一个表示四元数的 LaTeX 字符串，形式如 `Quaternion( ... )`，根据表达式中的参数生成字符串
    def _print_Quaternion(self, expr):
        # 对表达式中的每个参数进行括号化处理，严格按照乘法运算符的优先级
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True)
             for i in expr.args]
        # 将参数组装成表示四元数的字符串形式，如 `s[0] + s[1] i + s[2] j + s[3] k`
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        # 返回用加号连接的四元数字符串
        return " + ".join(a)

    # 输出一个表示商环的 LaTeX 字符串，形式为 `\frac{R}{I}`，其中 `R` 和 `I` 分别是商环和基理想的打印形式
    def _print_QuotientRing(self, R):
        # 生成用于表示分数形式的 LaTeX 字符串，使用商环 `R` 和基理想 `R.base_ideal` 的打印形式
        return r"\frac{{{}}}{{{}}}".format(self._print(R.ring),
                 self._print(R.base_ideal))

    # 输出一个表示商环元素的 LaTeX 字符串，形式为 `{x.ring.to_sympy(x)} + {x.ring.base_ideal}`
    def _print_QuotientRingElement(self, x):
        # 获取商环元素 `x` 的 LaTeX 表示，包括环 `x.ring` 转换为 sympy 后的表达式和基理想的打印形式
        x_latex = self._print(x.ring.to_sympy(x))
        return r"{{{}}} + {{{}}}".format(x_latex,
                 self._print(x.ring.base_ideal))

    # 输出一个表示商模的 LaTeX 字符串，形式为 `[{m.module.ring.to_sympy(x)}] + {self._print(m.module.killed_module)}`
    def _print_QuotientModuleElement(self, m):
        # 获取商模元素 `m` 的数据列表的 sympy 表示，然后生成表示商模元素的 LaTeX 字符串
        data = [m.module.ring.to_sympy(x) for x in m.data]
        data_latex = r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in data))
        return r"{{{}}} + {{{}}}".format(data_latex,
                 self._print(m.module.killed_module))

    # 输出一个表示商模的 LaTeX 字符串，形式为 `\frac{M.base}{M.killed_module}`，其中 `M` 是商模对象
    def _print_QuotientModule(self, M):
        # 生成用于表示分数形式的 LaTeX 字符串，使用商模 `M` 的基础模和被杀死的模的打印形式
        return r"\frac{{{}}}{{{}}}".format(self._print(M.base),
                 self._print(M.killed_module))

    # 输出一个表示矩阵同态的 LaTeX 字符串，形式为 `{self._print(h._sympy_matrix())} : {self._print(h.domain)} \to {self._print(h.codomain)}`
    def _print_MatrixHomomorphism(self, h):
        # 生成表示矩阵同态的 LaTeX 字符串，包括矩阵 `h._sympy_matrix()` 的打印形式和定义域、值域的打印形式
        return r"{{{}}} : {{{}}} \to {{{}}}".format(self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    # 输出一个表示流形的 LaTeX 字符串，形式为 `\text{name}^{supers}_{subs}`，其中 name, supers, subs 是流形的名称及其上下标
    def _print_Manifold(self, manifold):
        # 获取流形的名称，并处理其中的上标和下标，最后生成用于表示流形的 LaTeX 字符串
        string = manifold.name.name
        if '{' in string:
            name, supers, subs = string, [], []
        else:
            name, supers, subs = split_super_sub(string)

            name = translate(name)
            supers = [translate(sup) for sup in supers]
            subs = [translate(sub) for sub in subs]

        name = r'\text{%s}' % name
        if supers:
            name += "^{%s}" % " ".join(supers)
        if subs:
            name += "_{%s}" % " ".join(subs)

        return name

    # 输出一个表示补丁的 LaTeX 字符串，形式为 `\text{patch.name}_{patch.manifold}`
    def _print_Patch(self, patch):
        # 生成表示补丁的 LaTeX 字符串，包括补丁的名称和所属流形的打印形式
        return r'\text{%s}_{%s}' % (self._print(patch.name), self._print(patch.manifold))

    # 输出一个表示坐标系的 LaTeX 字符串，形式为 `\text{coordsys.name}^{\text{coordsys.patch.name}}_{coordsys.manifold}`
    def _print_CoordSystem(self, coordsys):
        # 生成表示坐标系的 LaTeX 字符串，包括坐标系名称、所属补丁和流形的打印形式
        return r'\text{%s}^{\text{%s}}_{%s}' % (
            self._print(coordsys.name), self._print(coordsys.patch.name), self._print(coordsys.manifold)
        )

    # 输出一个表示协变导数算子的 LaTeX 字符串，形式为 `\mathbb{\nabla}_{cvd._wrt}`
    def _print_CovarDerivativeOp(self, cvd):
        # 生成表示协变导数算子的 LaTeX 字符串，包括相对于哪个对象求导的打印形式
        return r'\mathbb{\nabla}_{%s}' % self._print(cvd._wrt)

    # 输出一个表示基础标量场的 LaTeX 字符串，形式为 `\mathbf{Symbol(string)}`
    def _print_BaseScalarField(self, field):
        # 获取基础标量场的名称并生成对应的 LaTeX 字符串
        string = field._coord_sys.symbols[field._index].name
        return r'\mathbf{{{}}}'.format(self._print(Symbol(string)))

    # 输出一个表示基础向量场的 LaTeX 字符串，形式为 `\partial_{Symbol(string)}`
    def _print_BaseVectorField(self, field):
        # 获取基础向量场的名称并生成对应的 LaTeX 字符串
        string = field._coord_sys.symbols[field._index].name
        return r'\partial_{{{}}}'.format(self._print(Symbol(string)))
    # 定义一个方法，用于打印表示微分形式的 LaTeX 代码
    def _print_Differential(self, diff):
        # 获取微分形式的字段
        field = diff._form_field
        # 检查字段是否具有坐标系属性
        if hasattr(field, '_coord_sys'):
            # 获取坐标系中的符号名称，并格式化成 LaTeX 字符串
            string = field._coord_sys.symbols[field._index].name
            return r'\operatorname{{d}}{}'.format(self._print(Symbol(string)))
        else:
            # 否则，直接打印字段的 LaTeX 表达式，并包裹在 \operatorname{d}() 中
            string = self._print(field)
            return r'\operatorname{{d}}\left({}\right)'.format(string)

    # 定义一个方法，用于打印表示矩阵迹的 LaTeX 代码
    def _print_Tr(self, p):
        # TODO: 处理指标
        # 获取参数列表中第一个参数的 LaTeX 表达式
        contents = self._print(p.args[0])
        return r'\operatorname{{tr}}\left({}\right)'.format(contents)

    # 定义一个方法，用于打印 Euler 函数的 LaTeX 代码
    def _print_totient(self, expr, exp=None):
        # 如果给定了指数，则打印 Euler 函数的带指数的格式化表达式
        if exp is not None:
            return r'\left(\phi\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        # 否则，打印 Euler 函数的普通格式化表达式
        return r'\phi\left(%s\right)' % self._print(expr.args[0])

    # 定义一个方法，用于打印简化 Euler 函数的 LaTeX 代码
    def _print_reduced_totient(self, expr, exp=None):
        # 如果给定了指数，则打印简化 Euler 函数的带指数的格式化表达式
        if exp is not None:
            return r'\left(\lambda\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        # 否则，打印简化 Euler 函数的普通格式化表达式
        return r'\lambda\left(%s\right)' % self._print(expr.args[0])

    # 定义一个方法，用于打印约数和函数的 LaTeX 代码
    def _print_divisor_sigma(self, expr, exp=None):
        # 如果参数列表长度为2，则打印带有两个参数的格式化表达式
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            # 否则，打印带有单个参数的格式化表达式
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        # 如果给定了指数，则打印带有指数的约数和函数的格式化表达式
        if exp is not None:
            return r"\sigma^{%s}%s" % (exp, tex)
        # 否则，打印不带指数的约数和函数的格式化表达式
        return r"\sigma%s" % tex

    # 定义一个方法，用于打印乘积约数和函数的 LaTeX 代码
    def _print_udivisor_sigma(self, expr, exp=None):
        # 如果参数列表长度为2，则打印带有两个参数的格式化表达式
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            # 否则，打印带有单个参数的格式化表达式
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        # 如果给定了指数，则打印带有指数的乘积约数和函数的格式化表达式
        if exp is not None:
            return r"\sigma^*^{%s}%s" % (exp, tex)
        # 否则，打印不带指数的乘积约数和函数的格式化表达式
        return r"\sigma^*%s" % tex

    # 定义一个方法，用于打印素因子计数函数的 LaTeX 代码
    def _print_primenu(self, expr, exp=None):
        # 如果给定了指数，则打印素因子计数函数的带指数的格式化表达式
        if exp is not None:
            return r'\left(\nu\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        # 否则，打印素因子计数函数的普通格式化表达式
        return r'\nu\left(%s\right)' % self._print(expr.args[0])

    # 定义一个方法，用于打印素因子幂次函数的 LaTeX 代码
    def _print_primeomega(self, expr, exp=None):
        # 如果给定了指数，则打印素因子幂次函数的带指数的格式化表达式
        if exp is not None:
            return r'\left(\Omega\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        # 否则，打印素因子幂次函数的普通格式化表达式
        return r'\Omega\left(%s\right)' % self._print(expr.args[0])

    # 定义一个方法，用于打印字符串的 LaTeX 代码
    def _print_Str(self, s):
        return str(s.name)

    # 定义一个方法，用于打印浮点数的 LaTeX 代码
    def _print_float(self, expr):
        return self._print(Float(expr))

    # 定义一个方法，用于打印整数的 LaTeX 代码
    def _print_int(self, expr):
        return str(expr)

    # 定义一个方法，用于打印任意精度整数的 LaTeX 代码
    def _print_mpz(self, expr):
        return str(expr)

    # 定义一个方法，用于打印任意精度有理数的 LaTeX 代码
    def _print_mpq(self, expr):
        return str(expr)

    # 定义一个方法，用于打印有理数的 LaTeX 代码
    def _print_fmpz(self, expr):
        return str(expr)

    # 定义一个方法，用于打印有理数的 LaTeX 代码
    def _print_fmpq(self, expr):
        return str(expr)

    # 定义一个方法，用于打印谓词的 LaTeX 代码
    def _print_Predicate(self, expr):
        # 打印带有转义处理的谓词名称的格式化表达式
        return r"\operatorname{{Q}}_{{\text{{{}}}}}".format(latex_escape(str(expr.name)))
    # 定义一个方法，用于打印应用的谓词表达式
    def _print_AppliedPredicate(self, expr):
        # 提取表达式的谓词部分
        pred = expr.function
        # 提取表达式的参数部分
        args = expr.arguments
        # 将谓词部分转换为 LaTeX 格式的字符串
        pred_latex = self._print(pred)
        # 将参数部分转换为逗号分隔的 LaTeX 格式字符串列表，各个参数由逗号分隔
        args_latex = ', '.join([self._print(a) for a in args])
        # 返回格式化后的字符串，包含谓词和参数
        return '%s(%s)' % (pred_latex, args_latex)

    # 定义一个空的打印方法，用于处理默认的打印情况
    def emptyPrinter(self, expr):
        # 调用父类的打印方法，获取默认的字符串表示
        s = super().emptyPrinter(expr)
        # 返回带有数学特征的 monospace 格式的 LaTeX 转义字符串
        return r"\mathtt{\text{%s}}" % latex_escape(s)
def translate(s: str) -> str:
    r'''
    Check for a modifier ending the string.  If present, convert the
    modifier to latex and translate the rest recursively.

    Given a description of a Greek letter or other special character,
    return the appropriate latex.

    Let everything else pass as given.

    >>> from sympy.printing.latex import translate
    >>> translate('alphahatdotprime')
    "{\\dot{\\hat{\\alpha}}}'"
    '''
    # 查找字符串 s 是否在 tex_greek_dictionary 中
    tex = tex_greek_dictionary.get(s)
    if tex:
        return tex
    # 如果 s 是小写希腊字母集合中的元素，返回带反斜杠的小写形式
    elif s.lower() in greek_letters_set:
        return "\\" + s.lower()
    # 如果 s 是其他符号集合中的元素，返回带反斜杠的原始形式
    elif s in other_symbols:
        return "\\" + s
    else:
        # 处理可能存在的修饰符并递归调用 translate 函数
        for key in sorted(modifier_dict.keys(), key=len, reverse=True):
            # 如果 s 的小写形式以修饰符结尾且长度大于修饰符本身，则递归调用 translate 函数
            if s.lower().endswith(key) and len(s) > len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        # 若以上条件均不满足，返回原始字符串 s
        return s


@print_function(LatexPrinter)
def latex(expr, **settings):
    r"""Convert the given expression to LaTeX string representation.

    Parameters
    ==========
    full_prec: boolean, optional
        If set to True, a floating point number is printed with full precision.
    fold_frac_powers : boolean, optional
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
    fold_func_brackets : boolean, optional
        Fold function brackets where applicable.
    fold_short_frac : boolean, optional
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.
    inv_trig_style : string, optional
        How inverse trig functions should be displayed. Can be one of
        ``'abbreviated'``, ``'full'``, or ``'power'``. Defaults to
        ``'abbreviated'``.
    itex : boolean, optional
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.
    ln_notation : boolean, optional
        If set to ``True``, ``\ln`` is used instead of default ``\log``.
    long_frac_ratio : float or None, optional
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.
    mat_delim : string, optional
        The delimiter to wrap around matrices. Can be one of ``'['``, ``'('``,
        or the empty string ``''``. Defaults to ``'['``.
    mat_str : string, optional
        Which matrix environment string to emit. ``'smallmatrix'``,
        ``'matrix'``, ``'array'``, etc. Defaults to ``'smallmatrix'`` for
        inline mode, ``'matrix'`` for matrices of no more than 10 columns, and
        ``'array'`` otherwise.
    '''
    mode: string, optional
        Specifies how the generated code will be delimited. ``mode`` can be one
        of ``'plain'``, ``'inline'``, ``'equation'`` or ``'equation*'``.  If
        ``mode`` is set to ``'plain'``, then the resulting code will not be
        delimited at all (this is the default). If ``mode`` is set to
        ``'inline'`` then inline LaTeX ``$...$`` will be used. If ``mode`` is
        set to ``'equation'`` or ``'equation*'``, the resulting code will be
        enclosed in the ``equation`` or ``equation*`` environment (remember to
        import ``amsmath`` for ``equation*``), unless the ``itex`` option is
        set. In the latter case, the ``$$...$$`` syntax is used.
    mul_symbol : string or None, optional
        The symbol to use for multiplication. Can be one of ``None``,
        ``'ldot'``, ``'dot'``, or ``'times'``.
    order: string, optional
        Any of the supported monomial orderings (currently ``'lex'``,
        ``'grlex'``, or ``'grevlex'``), ``'old'``, and ``'none'``. This
        parameter does nothing for `~.Mul` objects. Setting order to ``'old'``
        uses the compatibility ordering for ``~.Add`` defined in Printer. For
        very large expressions, set the ``order`` keyword to ``'none'`` if
        speed is a concern.
    symbol_names : dictionary of strings mapped to symbols, optional
        Dictionary of symbols and the custom strings they should be emitted as.
    root_notation : boolean, optional
        If set to ``False``, exponents of the form 1/n are printed in fractonal
        form. Default is ``True``, to print exponent in root form.
    mat_symbol_style : string, optional
        Can be either ``'plain'`` (default) or ``'bold'``. If set to
        ``'bold'``, a `~.MatrixSymbol` A will be printed as ``\mathbf{A}``,
        otherwise as ``A``.
    imaginary_unit : string, optional
        String to use for the imaginary unit. Defined options are ``'i'``
        (default) and ``'j'``. Adding ``r`` or ``t`` in front gives ``\mathrm``
        or ``\text``, so ``'ri'`` leads to ``\mathrm{i}`` which gives
        `\mathrm{i}`.
    gothic_re_im : boolean, optional
        If set to ``True``, `\Re` and `\Im` is used for ``re`` and ``im``, respectively.
        The default is ``False`` leading to `\operatorname{re}` and `\operatorname{im}`.
    decimal_separator : string, optional
        Specifies what separator to use to separate the whole and fractional parts of a
        floating point number as in `2.5` for the default, ``period`` or `2{,}5`
        when ``comma`` is specified. Lists, sets, and tuple are printed with semicolon
        separating the elements when ``comma`` is chosen. For example, [1; 2; 3] when
        ``comma`` is chosen and [1,2,3] for when ``period`` is chosen.
    # 定义一个布尔型参数，指定是否在指数为超标时加括号
    parenthesize_super : boolean, optional
        If set to ``False``, superscripted expressions will not be parenthesized when
        powered. Default is ``True``, which parenthesizes the expression when powered.
    # 定义一个整数或 None 类型的可选参数，设定打印浮点数时指数的下限
    min: Integer or None, optional
        Sets the lower bound for the exponent to print floating point numbers in
        fixed-point format.
    # 定义一个整数或 None 类型的可选参数，设定打印浮点数时指数的上限
    max: Integer or None, optional
        Sets the upper bound for the exponent to print floating point numbers in
        fixed-point format.
    # 定义一个字符串型可选参数，指定微分操作符的字符串形式，默认为 `'d'`，以斜体形式打印
    diff_operator: string, optional
        String to use for differential operator. Default is ``'d'``, to print in italic
        form. ``'rd'``, ``'td'`` are shortcuts for ``\mathrm{d}`` and ``\text{d}``.
    # 定义一个字符串型可选参数，指定共轭转置符号的字符串形式，默认为 `'dagger'`
    # 可选项包括 `'dagger'`（默认）、`'star'` 和 `'hermitian'`
    adjoint_style: string, optional
        String to use for the adjoint symbol. Defined options are ``'dagger'``
        (default),``'star'``, and ``'hermitian'``.

    # 注释部分
    Notes
    =====

    # 提示不使用 print 语句打印时，latex 命令会产生双反斜杠，因为 Python 在字符串中使用双反斜杠转义单个反斜杠
    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    # 示例部分
    Examples
    ========

    # 导入 sympy 库中的 latex 函数和 Rational 类
    >>> from sympy import latex, Rational
    # 导入 sympy.abc 模块中的 tau 符号
    >>> from sympy.abc import tau

    # 打印 (2*tau)**(7/2) 的 LaTeX 表达式
    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
    # 打印并输出 (2*tau)**(7/2) 的 LaTeX 表达式
    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    # 基本用法示例
    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    # 使用 'mode' 和 'itex' 选项
    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    # 分数选项示例
    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}
    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left(\frac{7}{2} \right)}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets=True))
    # 左侧内容为 LaTeX 表达式，表示二元运算：$ \left(2 \tau\right)^{\sin {\frac{7}{2}}} $
    >>> print(latex(3*x**2/y))
    # 输出 LaTeX 表达式以显示分数形式的结果：$ \frac{3 x^{2}}{y} $
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    # 输出 LaTeX 表达式，并指定长分数形式的比率为 2：$ \frac{\int r\, dr}{2 \pi} $
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    # 输出 LaTeX 表达式，并将长分数形式关闭：$ \frac{1}{2 \pi} \int r\, dr $

    # 乘法选项的示例，输出 LaTeX 表达式，使用 "times" 作为乘号：$ \left(2 \times \tau\right)^{\sin{\left(\frac{7}{2} \right)}} $
    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))

    # 三角函数选项的示例，输出 LaTeX 表达式，显示反正弦函数：$ \operatorname{asin}{\left(\frac{7}{2} \right)} $
    >>> print(latex(asin(Rational(7,2))))
    # 输出 LaTeX 表达式，指定完整的反正弦函数样式：$ \arcsin{\left(\frac{7}{2} \right)} $
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    # 输出 LaTeX 表达式，指定幂的反正弦函数样式：$ \sin^{-1}{\left(\frac{7}{2} \right)} $
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))

    # 矩阵选项的示例，输出 LaTeX 表达式，创建 2x1 矩阵：$ \left[\begin{matrix}x\\y\end{matrix}\right] $
    >>> print(latex(Matrix(2, 1, [x, y])))
    # 输出 LaTeX 表达式，指定矩阵为 array 类型：$ \left[\begin{array}{c}x\\y\end{array}\right] $
    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    # 输出 LaTeX 表达式，指定矩阵定界符为圆括号：$ \left(\begin{matrix}x\\y\end{matrix}\right) $
    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))

    # 自定义符号的打印示例，输出 LaTeX 表达式，将符号 x 显示为 x_i：$ x_i^{2} $
    >>> print(latex(x**2, symbol_names={x: 'x_i'}))

    # 对数函数的示例，输出 LaTeX 表达式，显示普通对数：$ \log{\left(10 \right)} $
    >>> print(latex(log(10)))
    # 输出 LaTeX 表达式，显示自然对数：$ \ln{\left(10 \right)} $
    >>> print(latex(log(10), ln_notation=True))

    # latex() 支持的内置容器类型示例，输出 LaTeX 表达式，显示内联模式的列表：$ \left[ 2 / x, \  y\right] $
    >>> print(latex([2/x, y], mode='inline'))

    # 不支持的类型将作为单行等宽文本渲染示例，输出 LaTeX 表达式，显示整数类型：$ \mathtt{\text{<class 'int'>}} $
    >>> print(latex(int))
    # 输出 LaTeX 表达式，显示单行等宽文本字符串：$ \mathtt{\text{plain \% text}} $
    >>> print(latex("plain % text"))

    # 查看更多关于如何自定义类型渲染的示例链接
    See :ref:`printer_method_example` for an example of how to override
    this behavior for your own types by implementing ``_latex``.

    # 版本更新说明，不再将不支持的类型的 str 表示作为有效的 LaTeX 处理
    .. versionchanged:: 1.7.0
    """
    return LatexPrinter(settings).doprint(expr)
def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression. Takes the same
    settings as ``latex()``."""
    
    # 调用 latex() 函数生成表达式的 LaTeX 表示，并打印出来
    print(latex(expr, **settings))


def multiline_latex(lhs, rhs, terms_per_line=1, environment="align*", use_dots=False, **settings):
    r"""
    This function generates a LaTeX equation with a multiline right-hand side
    in an ``align*``, ``eqnarray`` or ``IEEEeqnarray`` environment.

    Parameters
    ==========

    lhs : Expr
        Left-hand side of equation

    rhs : Expr
        Right-hand side of equation

    terms_per_line : integer, optional
        Number of terms per line to print. Default is 1.

    environment : "string", optional
        Which LaTeX environment to use for the output. Options are "align*"
        (default), "eqnarray", and "IEEEeqnarray".

    use_dots : boolean, optional
        If ``True``, ``\\dots`` is added to the end of each line. Default is ``False``.

    Examples
    ========

    >>> from sympy import multiline_latex, symbols, sin, cos, exp, log, I
    >>> x, y, alpha = symbols('x y alpha')
    >>> expr = sin(alpha*y) + exp(I*alpha) - cos(log(y))
    >>> print(multiline_latex(x, expr))
    \begin{align*}
    x = & e^{i \alpha} \\
    & + \sin{\left(\alpha y \right)} \\
    & - \cos{\left(\log{\left(y \right)} \right)}
    \end{align*}

    Using at most two terms per line:
    >>> print(multiline_latex(x, expr, 2))
    \begin{align*}
    x = & e^{i \alpha} + \sin{\left(\alpha y \right)} \\
    & - \cos{\left(\log{\left(y \right)} \right)}
    \end{align*}

    Using ``eqnarray`` and dots:
    >>> print(multiline_latex(x, expr, terms_per_line=2, environment="eqnarray", use_dots=True))
    \begin{eqnarray}
    x & = & e^{i \alpha} + \sin{\left(\alpha y \right)} \dots\nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)}
    \end{eqnarray}

    Using ``IEEEeqnarray``:
    >>> print(multiline_latex(x, expr, environment="IEEEeqnarray"))
    \begin{IEEEeqnarray}{rCl}
    x & = & e^{i \alpha} \nonumber\\
    & & + \sin{\left(\alpha y \right)} \nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)}
    \end{IEEEeqnarray}

    Notes
    =====

    All optional parameters from ``latex`` can also be used.

    """

    # 根据参数 environment 选择不同的 LaTeX 环境和相应的格式设置
    l = LatexPrinter(**settings)
    if environment == "eqnarray":
        result = r'\begin{eqnarray}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{eqnarray}'
        doubleet = True
    elif environment == "IEEEeqnarray":
        result = r'\begin{IEEEeqnarray}{rCl}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{IEEEeqnarray}'
        doubleet = True
    elif environment == "align*":
        result = r'\begin{align*}' + '\n'
        first_term = '= &'
        nonumber = ''
        end_term =  '\n\\end{align*}'
        doubleet = False
    else:
        raise ValueError("Unknown environment: {}".format(environment))
    # 初始化空字符串用于存储省略号或者空白
    dots = ''
    # 如果 use_dots 为真，则 dots 被赋值为 '\dots'
    if use_dots:
        dots = r'\dots'
    # 获取右手边表达式的有序项
    terms = rhs.as_ordered_terms()
    # 获取项的数量
    n_terms = len(terms)
    # 初始化项计数器
    term_count = 1
    # 遍历每个项
    for i in range(n_terms):
        # 获取当前项
        term = terms[i]
        # 初始化项开始和结束字符串
        term_start = ''
        term_end = ''
        # 默认符号为 '+'
        sign = '+'
        # 如果超过每行允许的项数
        if term_count > terms_per_line:
            # 如果 doubleet 为真，则 term_start 为 '& & '，否则为 '& '
            if doubleet:
                term_start = '& & '
            else:
                term_start = '& '
            # 重置项计数器
            term_count = 1
        # 如果当前项正好是每行允许的最后一项
        if term_count == terms_per_line:
            # 行末处理
            if i < n_terms - 1:
                # 还有剩余的项
                term_end = dots + nonumber + r'\\' + '\n'
            else:
                term_end = ''

        # 如果当前项的第一个因子是 -1，则转换为负数形式
        if term.as_ordered_factors()[0] == -1:
            term = -1 * term
            sign = r'-'
        # 如果是第一项
        if i == 0:  # 开始
            # 如果符号为 '+'，则 sign 留空
            if sign == '+':
                sign = ''
            # 构建结果字符串的第一部分
            result += r'{:s} {:s}{:s} {:s} {:s}'.format(l.doprint(lhs),
                        first_term, sign, l.doprint(term), term_end)
        else:
            # 构建结果字符串的其余部分
            result += r'{:s}{:s} {:s} {:s}'.format(term_start, sign,
                        l.doprint(term), term_end)
        # 增加项计数器
        term_count += 1
    # 添加结束项
    result += end_term
    # 返回最终结果
    return result
```