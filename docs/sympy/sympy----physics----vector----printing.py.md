# `D:\src\scipysrc\sympy\sympy\physics\vector\printing.py`

```
# 导入 Sympython
# 导入 Derivative 类，用于求导数的表示
# 导入 UndefinedFunction 和 AppliedUndef 类，用于处理未定义的函数和应用的未定义函数
# 导入 Symbol 类，用于符号表示
# 导入 init_printing 函数，用于初始化打印设置
# 导入 LatexPrinter 类，用于 LaTeX 格式的打印
# 导入 PrettyPrinter 类，用于漂亮的打印
# 导入 center_accent 函数，用于在漂亮打印中居中显示符号
# 导入 StrPrinter 类，用于字符串表示
# 导入 PRECEDENCE，用于确定打印操作符的优先级
from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE

# 设置 __all__ 列表，用于模块导入时指定可导入的名称
__all__ = ['vprint', 'vsstrrepr', 'vsprint', 'vpprint', 'vlatex',
           'init_vprinting']

# 定义 VectorStrPrinter 类，继承自 StrPrinter 类，用于向量表达式的字符串打印
class VectorStrPrinter(StrPrinter):
    """String Printer for vector expressions. """

    # 重写 _print_Derivative 方法，处理导数表达式的打印
    def _print_Derivative(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        # 如果表达式的变量中包含 t，且第一个参数是未定义的函数
        if (bool(sum(i == t for i in e.variables)) &
                isinstance(type(e.args[0]), UndefinedFunction)):
            ol = str(e.args[0].func)
            for i, v in enumerate(e.variables):
                ol += dynamicsymbols._str
            return ol
        else:
            return StrPrinter().doprint(e)

    # 重写 _print_Function 方法，处理函数表达式的打印
    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        # 如果表达式是未定义的函数
        if isinstance(type(e), UndefinedFunction):
            return StrPrinter().doprint(e).replace("(%s)" % t, '')
        return e.func.__name__ + "(%s)" % self.stringify(e.args, ", ")

# 定义 VectorStrReprPrinter 类，继承自 VectorStrPrinter 类，用于向量表达式的字符串表示打印
class VectorStrReprPrinter(VectorStrPrinter):
    """String repr printer for vector expressions."""
    # 重写 _print_str 方法，处理字符串的表示打印
    def _print_str(self, s):
        return repr(s)

# 定义 VectorLatexPrinter 类，继承自 LatexPrinter 类，用于向量表达式的 LaTeX 格式打印
class VectorLatexPrinter(LatexPrinter):
    """Latex Printer for vector expressions. """

    # 重写 _print_Function 方法，处理函数表达式的 LaTeX 格式打印
    def _print_Function(self, expr, exp=None):
        from sympy.physics.vector.functions import dynamicsymbols
        func = expr.func.__name__
        t = dynamicsymbols._t

        if (hasattr(self, '_print_' + func) and not
            isinstance(type(expr), UndefinedFunction)):
            return getattr(self, '_print_' + func)(expr, exp)
        elif isinstance(type(expr), UndefinedFunction) and (expr.args == (t,)):
            # 将这种函数视为符号处理
            expr = Symbol(func)
            if exp is not None:
                base = self.parenthesize(expr, PRECEDENCE['Pow'])
                base = self.parenthesize_super(base)
                return r"%s^{%s}" % (base, exp)
            else:
                return super()._print(expr)
        else:
            return super()._print_Function(expr, exp)
    # 定义一个方法用于打印导数表达式，接受一个 sympy 的 Derivative 对象作为参数
    def _print_Derivative(self, der_expr):
        # 导数表达式求值，确保表达式处于正确的形式
        der_expr = der_expr.doit()
        
        # 如果导数表达式不是 Derivative 对象，则直接打印其内容
        if not isinstance(der_expr, Derivative):
            return r"\left(%s\right)" % self.doprint(der_expr)

        # 导数表达式是 Derivative 对象，检查其是否为动力学符号
        t = dynamicsymbols._t
        expr = der_expr.expr
        red = expr.atoms(AppliedUndef)
        syms = der_expr.variables
        
        # 检查导数表达式中是否含有应用未定义符号，且未绑定到时间 t 上
        test1 = not all(True for i in red if i.free_symbols == {t})
        # 检查所有的变量是否都是时间 t
        test2 = not all(t == i for i in syms)
        
        # 如果任何一个检查不通过，则调用父类的打印方法处理
        if test1 or test2:
            return super()._print_Derivative(der_expr)

        # 检查完成，确定导数的阶数
        dots = len(syms)
        # 打印函数表达式的基本部分
        base = self._print_Function(expr)
        # 按照导数阶数添加相应的点数符号
        base_split = base.split('_', 1)
        base = base_split[0]
        if dots == 1:
            base = r"\dot{%s}" % base
        elif dots == 2:
            base = r"\ddot{%s}" % base
        elif dots == 3:
            base = r"\dddot{%s}" % base
        elif dots == 4:
            base = r"\ddddot{%s}" % base
        else:  # 如果阶数超过 4，则回退到标准打印处理
            return super()._print_Derivative(der_expr)
        
        # 如果原函数表达式含有下划线，则保留下划线后的部分
        if len(base_split) != 1:
            base += '_' + base_split[1]
        
        # 返回添加点数符号后的导数表达式字符串
        return base
class VectorPrettyPrinter(PrettyPrinter):
    """Pretty Printer for vectorialexpressions. """

    def _print_Derivative(self, deriv):
        # 导入动态符号函数
        from sympy.physics.vector.functions import dynamicsymbols
        # XXX use U('PARTIAL DIFFERENTIAL') here ?
        # 获取时间符号
        t = dynamicsymbols._t
        # 初始化导数级数
        dot_i = 0
        # 获取导数变量列表，并反转顺序
        syms = list(reversed(deriv.variables))

        # 循环处理导数变量
        while len(syms) > 0:
            # 如果当前变量为时间符号，则移除，并增加dot_i计数
            if syms[-1] == t:
                syms.pop()
                dot_i += 1
            else:
                # 如果不是时间符号，则调用父类的打印函数处理
                return super()._print_Derivative(deriv)

        # 检查表达式是否为未定义函数，并且参数只包含时间符号
        if not (isinstance(type(deriv.expr), UndefinedFunction) and
                (deriv.expr.args == (t,))):
            return super()._print_Derivative(deriv)
        else:
            # 打印函数表达式
            pform = self._print_Function(deriv.expr)

        # 如果pform的picture长度大于1，则调用父类的打印函数处理
        if len(pform.picture) > 1:
            return super()._print_Derivative(deriv)

        # 处理特殊符号，用于表示高阶导数
        if dot_i >= 5:
            return super()._print_Derivative(deriv)

        # 处理特殊符号的显示方式
        dots = {0: "",
                1: "\N{COMBINING DOT ABOVE}",
                2: "\N{COMBINING DIAERESIS}",
                3: "\N{COMBINING THREE DOTS ABOVE}",
                4: "\N{COMBINING FOUR DOTS ABOVE}"}

        # 获取pform对象的字典表示
        d = pform.__dict__
        # 如果不使用Unicode，则计算所需的撇号数量并添加到输出中
        if not self._use_unicode:
            apostrophes = ""
            for i in range(0, dot_i):
                apostrophes += "'"
            d['picture'][0] += apostrophes + "(t)"
        else:
            # 使用Unicode时，通过中央重音函数修改显示样式
            d['picture'] = [center_accent(d['picture'][0], dots[dot_i])]
        return pform

    def _print_Function(self, e):
        # 导入动态符号函数
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        # XXX works only for applied functions
        # 获取函数和参数
        func = e.func
        args = e.args
        func_name = func.__name__
        # 打印符号
        pform = self._print_Symbol(Symbol(func_name))
        # 如果这个函数是t的未定义函数，则可能是一个动态符号，跳过(t)的显示
        if not (isinstance(func, UndefinedFunction) and (args == (t,))):
            return super()._print_Function(e)
        return pform


def vprint(expr, **settings):
    r"""Function for printing of expressions generated in the
    sympy.physics vector package.

    Extends SymPy's StrPrinter, takes the same setting accepted by SymPy's
    :func:`~.sstr`, and is equivalent to ``print(sstr(foo))``.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    """
    # 导入 SymPy 中的符号动力学相关模块
    from sympy.physics.vector import vprint, dynamicsymbols
    # 创建一个动力学符号 u1，表示为 u1(t)
    u1 = dynamicsymbols('u1')
    # 打印动力学符号 u1，显示为 u1(t)
    print(u1)
    # 使用 vprint 函数打印动力学符号 u1，不包含时间变量
    vprint(u1)
    
    """
    
    outstr = vsprint(expr, **settings)
    
    # 导入内建模块 builtins
    import builtins
    # 如果 outstr 不是字符串 'None'，则执行以下操作
    if (outstr != 'None'):
        # 将 outstr 存储到内建变量 _ 中
        builtins._ = outstr
        # 打印 outstr 的内容
        print(outstr)
def vsstrrepr(expr, **settings):
    """Function for displaying expression representation's with vector
    printing enabled.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstrrepr().

    """
    # 创建一个 VectorStrReprPrinter 对象，用于打印 SymPy 表达式的字符串表示
    p = VectorStrReprPrinter(settings)
    # 调用 VectorStrReprPrinter 的 doprint 方法，返回表达式的字符串表示
    return p.doprint(expr)


def vsprint(expr, **settings):
    r"""Function for displaying expressions generated in the
    sympy.physics vector package.

    Returns the output of vprint() as a string.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    >>> from sympy.physics.vector import vsprint, dynamicsymbols
    >>> u1, u2 = dynamicsymbols('u1 u2')
    >>> u2d = dynamicsymbols('u2', level=1)
    >>> print("%s = %s" % (u1, u2 + u2d))
    u1(t) = u2(t) + Derivative(u2(t), t)
    >>> print("%s = %s" % (vsprint(u1), vsprint(u2 + u2d)))
    u1 = u2 + u2'

    """
    # 创建一个 VectorStrPrinter 对象，用于打印 sympy.physics.vector 包中的表达式
    string_printer = VectorStrPrinter(settings)
    # 调用 VectorStrPrinter 的 doprint 方法，返回表达式的字符串表示
    return string_printer.doprint(expr)


def vpprint(expr, **settings):
    r"""Function for pretty printing of expressions generated in the
    sympy.physics vector package.

    Mainly used for expressions not inside a vector; the output of running
    scripts and generating equations of motion. Takes the same options as
    SymPy's :func:`~.pretty_print`; see that function for more information.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to pretty print
    settings : args
        Same as those accepted by SymPy's pretty_print.

    """
    # 创建一个 VectorPrettyPrinter 对象，用于漂亮打印 sympy.physics.vector 包中的表达式
    pp = VectorPrettyPrinter(settings)

    # Note that this is copied from sympy.printing.pretty.pretty_print:

    # XXX: this is an ugly hack, but at least it works
    # 获取当前设置下是否使用 Unicode
    use_unicode = pp._settings['use_unicode']
    from sympy.printing.pretty.pretty_symbology import pretty_use_unicode
    # 设置是否使用 Unicode，返回原始设置状态
    uflag = pretty_use_unicode(use_unicode)

    try:
        # 调用 VectorPrettyPrinter 的 doprint 方法，返回表达式的漂亮打印字符串
        return pp.doprint(expr)
    finally:
        # 恢复原始的 Unicode 使用设置
        pretty_use_unicode(uflag)


def vlatex(expr, **settings):
    r"""Function for printing latex representation of sympy.physics.vector
    objects.

    For latex representation of Vectors, Dyadics, and dynamicsymbols. Takes the
    same options as SymPy's :func:`~.latex`; see that function for more
    information;

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to represent in LaTeX form
    settings : args
        Same as latex()

    Examples
    ========

    >>> from sympy.physics.vector import vlatex, ReferenceFrame, dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q1, q2 = dynamicsymbols('q1 q2')
    >>> q1d, q2d = dynamicsymbols('q1 q2', 1)
    >>> q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    >>> vlatex(N.x + N.y)
    '\\mathbf{\\hat{n}_x} + \\mathbf{\\hat{n}_y}'
    >>> vlatex(q1 + q2)
    'q_{1} + q_{2}'

    """
    # 创建一个 VectorLatexPrinter 对象，用于打印 sympy.physics.vector 包中表达式的 LaTeX 表示
    latex_printer = VectorLatexPrinter(settings)
    # 调用 VectorLatexPrinter 的 doprint 方法，返回表达式的 LaTeX 字符串表示
    return latex_printer.doprint(expr)
    # 使用 vlatex 函数将表达式 q1d 转换为 LaTeX 格式的字符串 '\\dot{q}_{1}'
    >>> vlatex(q1d)
    
    # 使用 vlatex 函数将表达式 q1 * q2d 转换为 LaTeX 格式的字符串 'q_{1} \\dot{q}_{2}'
    >>> vlatex(q1 * q2d)
    
    # 使用 vlatex 函数将表达式 q1dd * q1 / q1d 转换为 LaTeX 格式的字符串 '\\frac{q_{1} \\ddot{q}_{1}}{\\dot{q}_{1}}'
    >>> vlatex(q1dd * q1 / q1d)
    
    """
    # 创建一个 VectorLatexPrinter 对象，使用指定的 settings 参数
    latex_printer = VectorLatexPrinter(settings)
    
    # 调用 VectorLatexPrinter 对象的 doprint 方法，将输入的 expr 参数转换为 LaTeX 格式的字符串，并返回结果
    return latex_printer.doprint(expr)
def init_vprinting(**kwargs):
    """Initializes time derivative printing for all SymPy objects, i.e. any
    functions of time will be displayed in a more compact notation. The main
    benefit of this is for printing of time derivatives; instead of
    displaying as ``Derivative(f(t),t)``, it will display ``f'``. This is
    only actually needed for when derivatives are present and are not in a
    physics.vector.Vector or physics.vector.Dyadic object. This function is a
    light wrapper to :func:`~.init_printing`. Any keyword
    arguments for it are valid here.

    {0}

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> t, x = symbols('t, x')
    >>> omega = Function('omega')
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    Derivative(omega(t), t)

    Now use the string printer:

    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    omega'

    """
    # 设置字符串打印函数为 vsstrrepr
    kwargs['str_printer'] = vsstrrepr
    # 设置美化打印函数为 vpprint
    kwargs['pretty_printer'] = vpprint
    # 设置 LaTeX 打印函数为 vlatex
    kwargs['latex_printer'] = vlatex
    # 调用 init_printing 函数，传入所有关键字参数
    init_printing(**kwargs)


# 从 init_printing 函数的文档字符串中提取参数说明部分
params = init_printing.__doc__.split('Examples\n    ========')[0]  # type: ignore
# 格式化 init_vprinting 函数的文档字符串，插入参数说明
init_vprinting.__doc__ = init_vprinting.__doc__.format(params)  # type: ignore
```