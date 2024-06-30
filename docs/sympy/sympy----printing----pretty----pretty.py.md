# `D:\src\scipysrc\sympy\sympy\printing\pretty\pretty.py`

```
import itertools  # 导入 itertools 模块，用于高效的迭代工具

from sympy.core import S  # 导入 sympy 核心模块的 S 对象
from sympy.core.add import Add  # 导入 sympy 核心模块的 Add 类
from sympy.core.containers import Tuple  # 导入 sympy 核心模块的 Tuple 类
from sympy.core.function import Function  # 导入 sympy 核心模块的 Function 类
from sympy.core.mul import Mul  # 导入 sympy 核心模块的 Mul 类
from sympy.core.numbers import Number, Rational  # 导入 sympy 核心模块的 Number 和 Rational 类
from sympy.core.power import Pow  # 导入 sympy 核心模块的 Pow 类
from sympy.core.sorting import default_sort_key  # 导入 sympy 核心模块的 default_sort_key 函数
from sympy.core.symbol import Symbol  # 导入 sympy 核心模块的 Symbol 类
from sympy.core.sympify import SympifyError  # 导入 sympy 核心模块的 SympifyError 异常
from sympy.printing.conventions import requires_partial  # 导入 sympy 打印模块的 requires_partial 函数
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional  # 导入 sympy 打印模块的几个优先级相关变量和函数
from sympy.printing.printer import Printer, print_function  # 导入 sympy 打印模块的 Printer 类和 print_function 函数
from sympy.printing.str import sstr  # 导入 sympy 打印模块的 sstr 函数
from sympy.utilities.iterables import has_variety  # 导入 sympy 工具模块的 has_variety 函数
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入 sympy 工具模块的 sympy_deprecation_warning 异常

from sympy.printing.pretty.stringpict import prettyForm, stringPict  # 导入 sympy 漂亮打印模块的 prettyForm 和 stringPict 类
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \  # 导入 sympy 漂亮打印模块的几个符号相关变量和函数
    xsym, pretty_symbol, pretty_atom, pretty_use_unicode, greek_unicode, U, \
    pretty_try_use_unicode, annotated, is_subscriptable_in_unicode, center_pad,  root as nth_root

# 重命名以供外部使用
pprint_use_unicode = pretty_use_unicode  # 将 pretty_use_unicode 函数重命名为 pprint_use_unicode
pprint_try_use_unicode = pretty_try_use_unicode  # 将 pretty_try_use_unicode 函数重命名为 pprint_try_use_unicode


class PrettyPrinter(Printer):
    """Printer, which converts an expression into 2D ASCII-art figure."""

    printmethod = "_pretty"  # 打印方法设定为 "_pretty"

    _default_settings = {
        "order": None,  # 排序设定为 None
        "full_prec": "auto",  # 全精度设定为 "auto"
        "use_unicode": None,  # 使用 Unicode 设定为 None
        "wrap_line": True,  # 是否换行设定为 True
        "num_columns": None,  # 列数设定为 None
        "use_unicode_sqrt_char": True,  # 使用 Unicode 平方根字符设定为 True
        "root_notation": True,  # 使用根号符号设定为 True
        "mat_symbol_style": "plain",  # 矩阵符号风格设定为 "plain"
        "imaginary_unit": "i",  # 虚数单位设定为 "i"
        "perm_cyclic": True  # 循环置换设定为 True
    }

    def __init__(self, settings=None):
        """初始化函数，设定打印器的设置"""
        Printer.__init__(self, settings)

        if not isinstance(self._settings['imaginary_unit'], str):
            raise TypeError("'imaginary_unit' must a string, not {}".format(self._settings['imaginary_unit']))
        elif self._settings['imaginary_unit'] not in ("i", "j"):
            raise ValueError("'imaginary_unit' must be either 'i' or 'j', not '{}'".format(self._settings['imaginary_unit']))

    def emptyPrinter(self, expr):
        """返回表达式的字符串形式的 prettyForm 对象"""
        return prettyForm(str(expr))

    @property
    def _use_unicode(self):
        """判断是否使用 Unicode"""
        if self._settings['use_unicode']:
            return True
        else:
            return pretty_use_unicode()

    def doprint(self, expr):
        """打印表达式的函数"""
        return self._print(expr).render(**self._settings)

    # 空操作，确保 _print(stringPict) 返回相同的对象
    def _print_stringPict(self, e):
        return e

    def _print_basestring(self, e):
        """打印字符串的函数"""
        return prettyForm(e)

    def _print_atan2(self, e):
        """打印 atan2 函数的函数"""
        pform = prettyForm(*self._print_seq(e.args).parens())
        pform = prettyForm(*pform.left('atan2'))
        return pform

    def _print_Symbol(self, e, bold_name=False):
        """打印符号的函数"""
        symb = pretty_symbol(e.name, bold_name)
        return prettyForm(symb)
    _print_RandomSymbol = _print_Symbol  # _print_RandomSymbol 与 _print_Symbol 功能相同，用于打印随机符号
    # 用于打印矩阵符号表达式，根据设置决定是否使用粗体样式
    def _print_MatrixSymbol(self, e):
        return self._print_Symbol(e, self._settings['mat_symbol_style'] == "bold")

    # 用于打印浮点数表达式
    def _print_Float(self, e):
        # 根据 self._print_level 和 self._settings["full_prec"] 决定是否使用全精度打印
        full_prec = self._settings["full_prec"]
        if full_prec == "auto":
            full_prec = self._print_level == 1
        return prettyForm(sstr(e, full_prec=full_prec))

    # 用于打印向量的叉乘表达式
    def _print_Cross(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        # 将向量表达式转换为美观格式
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('MULTIPLICATION SIGN'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform

    # 用于打印向量的旋度（curl）表达式
    def _print_Curl(self, e):
        vec = e._expr
        # 将向量表达式转换为美观格式
        pform = self._print(vec)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('MULTIPLICATION SIGN'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    # 用于打印向量的散度（divergence）表达式
    def _print_Divergence(self, e):
        vec = e._expr
        # 将向量表达式转换为美观格式
        pform = self._print(vec)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    # 用于打印向量的点乘表达式
    def _print_Dot(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        # 将向量表达式转换为美观格式
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform

    # 用于打印函数的梯度（gradient）表达式
    def _print_Gradient(self, e):
        func = e._expr
        # 将函数表达式转换为美观格式
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    # 用于打印函数的拉普拉斯算子（Laplacian）表达式
    def _print_Laplacian(self, e):
        func = e._expr
        # 将函数表达式转换为美观格式
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('INCREMENT'))))
        return pform

    # 用于打印单一原子表达式
    def _print_Atom(self, e):
        try:
            # 打印像 Exp1 或 Pi 这样的原子表达式
            return prettyForm(pretty_atom(e.__class__.__name__, printer=self))
        except KeyError:
            return self.emptyPrinter(e)

    # Infinity 继承自 Number，所以需要覆盖打印顺序
    _print_Infinity = _print_Atom
    _print_NegativeInfinity = _print_Atom
    # 将打印函数的别名指向打印原子表达式的函数
    _print_EmptySet = _print_Atom
    _print_Naturals = _print_Atom
    _print_Naturals0 = _print_Atom
    _print_Integers = _print_Atom
    _print_Rationals = _print_Atom
    _print_Complexes = _print_Atom

    # 将打印函数的别名指向打印原子序列表达式的函数
    _print_EmptySequence = _print_Atom

    # 定义打印实数的函数，根据使用Unicode与否选择合适的打印方法
    def _print_Reals(self, e):
        if self._use_unicode:
            return self._print_Atom(e)
        else:
            # 构建一个包含无穷符号的列表
            inf_list = ['-oo', 'oo']
            # 使用特定的符号打印这个列表，加上括号
            return self._print_seq(inf_list, '(', ')')

    # 定义打印子阶乘的函数
    def _print_subfactorial(self, e):
        x = e.args[0]
        # 打印参数 x
        pform = self._print(x)
        # 如果 x 不是非负整数或符号，则给打印结果加上括号
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        # 在打印结果左边加上感叹号
        pform = prettyForm(*pform.left('!'))
        return pform

    # 定义打印阶乘的函数
    def _print_factorial(self, e):
        x = e.args[0]
        # 打印参数 x
        pform = self._print(x)
        # 如果 x 不是非负整数或符号，则给打印结果加上括号
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        # 在打印结果右边加上感叹号
        pform = prettyForm(*pform.right('!'))
        return pform

    # 定义打印双阶乘的函数
    def _print_factorial2(self, e):
        x = e.args[0]
        # 打印参数 x
        pform = self._print(x)
        # 如果 x 不是非负整数或符号，则给打印结果加上括号
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        # 在打印结果右边加上双感叹号
        pform = prettyForm(*pform.right('!!'))
        return pform

    # 定义打印二项式系数的函数
    def _print_binomial(self, e):
        n, k = e.args
        # 打印 n 和 k
        n_pform = self._print(n)
        k_pform = self._print(k)
        # 创建一个用空格填充的线，长度为 n_pform 和 k_pform 中宽度较大的那个
        bar = ' '*max(n_pform.width(), k_pform.width())
        # 将 k_pform 放在 bar 上方，再将 n_pform 放在上面，最后用括号括起来
        pform = prettyForm(*k_pform.above(bar))
        pform = prettyForm(*pform.above(n_pform))
        pform = prettyForm(*pform.parens('(', ')'))
        # 调整基线位置
        pform.baseline = (pform.baseline + 1)//2
        return pform

    # 定义打印关系表达式的函数
    def _print_Relational(self, e):
        # 获取关系运算符
        op = prettyForm(' ' + xsym(e.rel_op) + ' ')
        # 打印左右操作数
        l = self._print(e.lhs)
        r = self._print(e.rhs)
        # 将左操作数、运算符和右操作数组合成一个字符串图像对象，使用括号绑定
        pform = prettyForm(*stringPict.next(l, op, r), binding=prettyForm.OPEN)
        return pform

    # 定义打印逻辑非表达式的函数
    def _print_Not(self, e):
        from sympy.logic.boolalg import (Equivalent, Implies)
        if self._use_unicode:
            arg = e.args[0]
            # 打印参数 arg
            pform = self._print(arg)
            # 如果 arg 是等价式，使用 NotEquiv 替换打印结果
            if isinstance(arg, Equivalent):
                return self._print_Equivalent(arg, altchar=pretty_atom('NotEquiv'))
            # 如果 arg 是蕴含式，使用 NotArrow 替换打印结果
            if isinstance(arg, Implies):
                return self._print_Implies(arg, altchar=pretty_atom('NotArrow'))
            # 如果 arg 是布尔类型且不是 Not，给打印结果加上括号
            if arg.is_Boolean and not arg.is_Not:
                pform = prettyForm(*pform.parens())
            # 在打印结果左边加上 Not 符号
            return prettyForm(*pform.left(pretty_atom('Not')))
        else:
            # 使用默认的打印函数打印表达式 e
            return self._print_Function(e)
    # 定义一个方法，用于打印布尔运算表达式的形式化表示
    def __print_Boolean(self, e, char, sort=True):
        # 获取表达式的参数
        args = e.args
        # 如果指定了排序，则按照默认排序键对参数进行排序
        if sort:
            args = sorted(e.args, key=default_sort_key)
        # 取第一个参数
        arg = args[0]
        # 对第一个参数进行打印
        pform = self._print(arg)

        # 如果第一个参数是布尔类型且不是逻辑非，将其形式化表示加上括号
        if arg.is_Boolean and not arg.is_Not:
            pform = prettyForm(*pform.parens())

        # 对剩余参数进行处理
        for arg in args[1:]:
            # 打印每个参数
            pform_arg = self._print(arg)

            # 如果参数是布尔类型且不是逻辑非，将其形式化表示加上括号
            if arg.is_Boolean and not arg.is_Not:
                pform_arg = prettyForm(*pform_arg.parens())

            # 将运算符添加到形式化表示的右侧
            pform = prettyForm(*pform.right(' %s ' % char))
            # 将参数的形式化表示添加到当前形式化表示的右侧
            pform = prettyForm(*pform.right(pform_arg))

        # 返回最终形式化表示
        return pform

    # 打印 And 运算符的形式化表示
    def _print_And(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('And'))
        else:
            return self._print_Function(e, sort=True)

    # 打印 Or 运算符的形式化表示
    def _print_Or(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Or'))
        else:
            return self._print_Function(e, sort=True)

    # 打印 Xor 运算符的形式化表示
    def _print_Xor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom("Xor"))
        else:
            return self._print_Function(e, sort=True)

    # 打印 Nand 运算符的形式化表示
    def _print_Nand(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Nand'))
        else:
            return self._print_Function(e, sort=True)

    # 打印 Nor 运算符的形式化表示
    def _print_Nor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Nor'))
        else:
            return self._print_Function(e, sort=True)

    # 打印 Implies 运算符的形式化表示
    def _print_Implies(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or pretty_atom('Arrow'), sort=False)
        else:
            return self._print_Function(e)

    # 打印 Equivalent 运算符的形式化表示
    def _print_Equivalent(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or pretty_atom('Equiv'))
        else:
            return self._print_Function(e, sort=True)

    # 打印共轭运算的形式化表示
    def _print_conjugate(self, e):
        # 打印参数的形式化表示
        pform = self._print(e.args[0])
        # 在参数的形式化表示上方添加下划线
        return prettyForm( *pform.above( hobj('_', pform.width())) )

    # 打印绝对值运算的形式化表示
    def _print_Abs(self, e):
        # 打印参数的形式化表示，同时用竖线包围
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens('|', '|'))
        return pform

    # 打印向下取整运算的形式化表示
    def _print_floor(self, e):
        if self._use_unicode:
            # 打印参数的形式化表示，同时用左右地板符号包围
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens('lfloor', 'rfloor'))
            return pform
        else:
            return self._print_Function(e)

    # 打印向上取整运算的形式化表示
    def _print_ceiling(self, e):
        if self._use_unicode:
            # 打印参数的形式化表示，同时用左右天花板符号包围
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens('lceil', 'rceil'))
            return pform
        else:
            return self._print_Function(e)
    # 打印导数的方法，生成漂亮的数学表达式
    def _print_Derivative(self, deriv):
        # 检查导数表达式是否需要局部导数符号，并且是否使用Unicode
        if requires_partial(deriv.expr) and self._use_unicode:
            deriv_symbol = U('PARTIAL DIFFERENTIAL')
        else:
            deriv_symbol = r'd'
        x = None
        count_total_deriv = 0

        # 对导数变量计数列表进行逆序遍历
        for sym, num in reversed(deriv.variable_count):
            # 打印当前变量符号的字符串表示
            s = self._print(sym)
            # 将导数符号与变量符号左对齐形成漂亮的表达式
            ds = prettyForm(*s.left(deriv_symbol))
            count_total_deriv += num

            # 如果变量个数不是整数或大于1，则将数量作为指数添加到导数符号上
            if (not num.is_Integer) or (num > 1):
                ds = ds**prettyForm(str(num))

            # 如果是第一个变量，直接赋值给x；否则右对齐形成链式导数表达式
            if x is None:
                x = ds
            else:
                x = prettyForm(*x.right(' '))
                x = prettyForm(*x.right(ds))

        # 对表达式本身进行漂亮的打印
        f = prettyForm(
            binding=prettyForm.FUNC, *self._print(deriv.expr).parens())

        # 创建导数符号的漂亮表达式
        pform = prettyForm(deriv_symbol)

        # 如果总导数个数大于1，则将总数作为指数添加到导数符号上
        if (count_total_deriv > 1) != False:
            pform = pform**prettyForm(str(count_total_deriv))

        # 将导数符号放在表达式的上方，并调整基线
        pform = prettyForm(*pform.below(stringPict.LINE, x))
        pform.baseline = pform.baseline + 1
        # 将表达式放在导数符号下方，并标记为乘法绑定
        pform = prettyForm(*stringPict.next(pform, f))
        pform.binding = prettyForm.MUL

        # 返回最终的漂亮表达式
        return pform

    # 打印循环置换的方法
    def _print_Cycle(self, dc):
        from sympy.combinatorics.permutations import Permutation, Cycle
        # 如果是空置换，返回一个空的漂亮表达式
        if dc == Cycle():
            cyc = stringPict('')
            return prettyForm(*cyc.parens())

        # 获取循环置换的列表表示
        dc_list = Permutation(dc.list()).cyclic_form
        # 如果是恒等置换，返回大小减1的漂亮表达式
        if dc_list == []:
            cyc = self._print(dc.size - 1)
            return prettyForm(*cyc.parens())

        # 初始化空的漂亮表达式
        cyc = stringPict('')
        # 遍历循环置换列表，生成对应的漂亮表达式
        for i in dc_list:
            l = self._print(str(tuple(i)).replace(',', ''))
            cyc = prettyForm(*cyc.right(l))
        return cyc

    # 打印排列的方法
    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle

        # 获取全局设置中的置换打印格式
        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            # 如果已设置置换打印格式，发出警告信息
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=7,
            )
        else:
            # 否则使用默认的置换打印格式
            perm_cyclic = self._settings.get("perm_cyclic", True)

        # 如果需要循环打印置换
        if perm_cyclic:
            return self._print_Cycle(Cycle(expr))

        # 获取排列的下、上标形式
        lower = expr.array_form
        upper = list(range(len(lower)))

        # 初始化结果表达式
        result = stringPict('')
        first = True
        # 遍历上、下标形式，生成漂亮的排列表达式
        for u, l in zip(upper, lower):
            s1 = self._print(u)
            s2 = self._print(l)
            col = prettyForm(*s1.below(s2))
            if first:
                first = False
            else:
                col = prettyForm(*col.left(" "))
            result = prettyForm(*result.right(col))
        return prettyForm(*result.parens())
    # 定义一个方法来打印积分表达式，接受一个积分对象作为参数
    def _print_Integral(self, integral):
        # 获取积分函数
        f = integral.function

        # 如果参数包含多个项的加法，则添加括号并创建一个漂亮的参数形式
        prettyF = self._print(f)
        # XXX 通用化括号
        if f.is_Add:
            prettyF = prettyForm(*prettyF.parens())

        # dx dy dz ...
        arg = prettyF
        for x in integral.limits:
            # 获取积分变量并创建漂亮的参数形式
            prettyArg = self._print(x[0])
            # XXX qparens (如果需要括号则添加括号)
            if prettyArg.width() > 1:
                prettyArg = prettyForm(*prettyArg.parens())

            arg = prettyForm(*arg.right(' d', prettyArg))

        # \int \int \int ...
        firstterm = True
        s = None
        for lim in integral.limits:
            # 基于参数高度创建积分符号
            h = arg.height()
            H = h + 2

            # XXX hack!
            ascii_mode = not self._use_unicode
            if ascii_mode:
                H += 2

            vint = vobj('int', H)

            # 构造包含积分符号和参数的漂亮形式
            pform = prettyForm(vint)
            pform.baseline = arg.baseline + (
                H - h)//2    # 覆盖整个参数

            if len(lim) > 1:
                # 如果是定积分，则创建端点的漂亮形式。不打印空端点。
                if len(lim) == 2:
                    prettyA = prettyForm("")
                    prettyB = self._print(lim[1])
                if len(lim) == 3:
                    prettyA = self._print(lim[1])
                    prettyB = self._print(lim[2])

                if ascii_mode:  # XXX hack
                    # 添加间距以便更容易地将端点与正确的积分符号对应起来
                    spc = max(1, 3 - prettyB.width())
                    prettyB = prettyForm(*prettyB.left(' ' * spc))

                    spc = max(1, 4 - prettyA.width())
                    prettyA = prettyForm(*prettyA.right(' ' * spc))

                pform = prettyForm(*pform.above(prettyB))
                pform = prettyForm(*pform.below(prettyA))

            if not ascii_mode:  # XXX hack
                pform = prettyForm(*pform.right(' '))

            if firstterm:
                s = pform   # 第一项
                firstterm = False
            else:
                s = prettyForm(*s.left(pform))

        pform = prettyForm(*arg.left(s))
        pform.binding = prettyForm.MUL
        return pform
    # 定义一个方法来打印乘积表达式，接受一个表达式参数expr
    def _print_Product(self, expr):
        # 获取表达式中的项
        func = expr.term
        # 使用内部方法打印该项，并得到其美化后的字符串形式
        pretty_func = self._print(func)

        # 设置水平线、角线和垂直线字符
        horizontal_chr = xobj('_', 1)
        corner_chr = xobj('_', 1)
        vertical_chr = xobj('|', 1)

        # 如果使用Unicode，则使用相应的角线字符
        if self._use_unicode:
            horizontal_chr = xobj('-', 1)
            corner_chr = xobj('UpTack', 1)

        # 计算项的高度
        func_height = pretty_func.height()

        # 初始化变量
        first = True  # 是否是第一个限制条件
        max_upper = 0  # 记录最大上限高度
        sign_height = 0  # 记录符号的高度

        # 遍历表达式的限制条件
        for lim in expr.limits:
            # 使用内部方法打印上下限，并得到其美化后的字符串形式
            pretty_lower, pretty_upper = self.__print_SumProduct_Limits(lim)

            # 计算符号框架的宽度
            width = (func_height + 2) * 5 // 3 - 2
            # 构建符号的线条
            sign_lines = [horizontal_chr + corner_chr + (horizontal_chr * (width-2)) + corner_chr + horizontal_chr]
            for _ in range(func_height + 1):
                sign_lines.append(' ' + vertical_chr + (' ' * (width-2)) + vertical_chr + ' ')

            # 创建符号的字符串形式
            pretty_sign = stringPict('')
            pretty_sign = prettyForm(*pretty_sign.stack(*sign_lines))

            # 更新最大上限高度
            max_upper = max(max_upper, pretty_upper.height())

            # 如果是第一个限制条件，记录符号的高度
            if first:
                sign_height = pretty_sign.height()

            # 将符号放置在上下限之上和之下
            pretty_sign = prettyForm(*pretty_sign.above(pretty_upper))
            pretty_sign = prettyForm(*pretty_sign.below(pretty_lower))

            # 如果是第一个限制条件，调整函数的基线位置
            if first:
                pretty_func.baseline = 0
                first = False

            # 更新符号的高度，并添加填充以对齐函数
            height = pretty_sign.height()
            padding = stringPict('')
            padding = prettyForm(*padding.stack(*[' ']*(height - 1)))
            pretty_sign = prettyForm(*pretty_sign.right(padding))

            # 将函数放置在符号的右侧
            pretty_func = prettyForm(*pretty_sign.right(pretty_func))

        # 设置函数的基线位置为最大上限高度加上符号高度的一半
        pretty_func.baseline = max_upper + sign_height//2
        # 设置函数的绑定为乘法
        pretty_func.binding = prettyForm.MUL
        # 返回美化后的函数表达式
        return pretty_func

    # 内部方法，用于打印求和乘积的上下限
    def __print_SumProduct_Limits(self, lim):
        # 定义一个辅助函数来打印起始符号
        def print_start(lhs, rhs):
            # 使用等号打印操作符
            op = prettyForm(' ' + xsym("==") + ' ')
            # 打印左右两侧的表达式
            l = self._print(lhs)
            r = self._print(rhs)
            # 将打印结果组合成字符串形式
            pform = prettyForm(*stringPict.next(l, op, r))
            return pform

        # 使用内部方法打印上限，并得到其美化后的字符串形式
        prettyUpper = self._print(lim[2])
        # 使用辅助函数打印下限，并得到其美化后的字符串形式
        prettyLower = print_start(lim[0], lim[1])
        # 返回美化后的下限和上限
        return prettyLower, prettyUpper
    # 定义一个方法用于打印极限表达式，接受一个参数 l
    def _print_Limit(self, l):
        # 解构 l 的参数 e, z, z0, dir
        e, z, z0, dir = l.args

        # 打印表达式 e 并处理其优先级小于等于乘法的情况下加上括号
        E = self._print(e)
        if precedence(e) <= PRECEDENCE["Mul"]:
            E = prettyForm(*E.parens('(', ')'))

        # 创建一个表示极限符号的 prettyForm 对象
        Lim = prettyForm('lim')

        # 打印变量 z，并根据使用的语言设定箭头的显示方式
        LimArg = self._print(z)
        if self._use_unicode:
            LimArg = prettyForm(*LimArg.right(f"{xobj('-', 1)}{pretty_atom('Arrow')}"))
        else:
            LimArg = prettyForm(*LimArg.right('->'))
        LimArg = prettyForm(*LimArg.right(self._print(z0)))

        # 处理极限方向的显示，根据 dir 的值决定是否显示正负号
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            dir = ""
        else:
            if self._use_unicode:
                dir = pretty_atom('SuperscriptPlus') if str(dir) == "+" else pretty_atom('SuperscriptMinus')
        
        # 将极限方向打印到 LimArg 中
        LimArg = prettyForm(*LimArg.right(self._print(dir)))

        # 构建完整的极限表达式
        Lim = prettyForm(*Lim.below(LimArg))
        Lim = prettyForm(*Lim.right(E), binding=prettyForm.MUL)

        # 返回最终的极限表达式
        return Lim


    # 打印矩阵内容的方法
    def _print_matrix_contents(self, e):
        """
        This method factors out what is essentially grid printing.
        """
        M = e   # matrix
        Ms = {}  # i,j -> pretty(M[i,j])

        # 遍历矩阵 M 的所有元素并打印
        for i in range(M.rows):
            for j in range(M.cols):
                Ms[i, j] = self._print(M[i, j])

        # 设定水平和垂直分隔符的大小
        hsep = 2
        vsep = 1

        # 计算每列的最大宽度
        maxw = [-1] * M.cols
        for j in range(M.cols):
            maxw[j] = max([Ms[i, j].width() for i in range(M.rows)] or [0])

        # 初始化绘制结果
        D = None

        # 逐行构建矩阵的绘制结果
        for i in range(M.rows):
            D_row = None
            for j in range(M.cols):
                s = Ms[i, j]

                # 调整 s 的宽度到 maxw[j] 大小
                assert s.width() <= maxw[j]

                # 水平居中 s，向右偏移 0.5
                left, right = center_pad(s.width(), maxw[j])
                s = prettyForm(*s.right(right))
                s = prettyForm(*s.left(left))

                # 如果是该行的第一个盒子，则直接赋值给 D_row
                if D_row is None:
                    D_row = s
                    continue

                # 添加水平间隔
                D_row = prettyForm(*D_row.right(' ' * hsep))
                D_row = prettyForm(*D_row.right(s))

            # 如果是第一行，则直接赋值给 D
            if D is None:
                D = D_row
                continue

            # 添加垂直间隔
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))

            # 将当前行添加到 D 中
            D = prettyForm(*D.below(D_row))

        # 如果 D 仍为 None，则说明矩阵为空，返回空字符串的 prettyForm
        if D is None:
            D = prettyForm('')  # Empty Matrix

        # 返回绘制结果 D
        return D
    # 定义一个方法，用于打印基础矩阵表达式，包含左右括号定制功能
    def _print_MatrixBase(self, e, lparens='[', rparens=']'):
        # 获取矩阵内容的显示形式
        D = self._print_matrix_contents(e)
        # 设置基线为高度的一半
        D.baseline = D.height() // 2
        # 使用定制的左右括号包围矩阵内容并返回
        D = prettyForm(*D.parens(lparens, rparens))
        return D

    # 定义一个方法，用于打印行列式表达式
    def _print_Determinant(self, e):
        # 获取行列式的参数
        mat = e.arg
        # 如果参数是矩阵表达式
        if mat.is_MatrixExpr:
            # 导入块矩阵类
            from sympy.matrices.expressions.blockmatrix import BlockMatrix
            # 如果是块矩阵，使用特定左右竖线打印
            if isinstance(mat, BlockMatrix):
                return self._print_MatrixBase(mat.blocks, lparens='|', rparens='|')
            # 否则打印普通矩阵表达式
            D = self._print(mat)
            D.baseline = D.height() // 2
            return prettyForm(*D.parens('|', '|'))
        else:
            # 如果参数不是矩阵表达式，则打印基础矩阵表达式
            return self._print_MatrixBase(mat, lparens='|', rparens='|')

    # 定义一个方法，用于打印张量积表达式
    def _print_TensorProduct(self, expr):
        # 根据是否使用 Unicode 设置张量积符号
        if self._use_unicode:
            circled_times = "\u2297"
        else:
            circled_times = ".*"
        # 调用通用打印序列的方法，传入参数表达式和定制符号
        return self._print_seq(expr.args, None, None, circled_times,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    # 定义一个方法，用于打印楔积表达式
    def _print_WedgeProduct(self, expr):
        # 根据是否使用 Unicode 设置楔积符号
        if self._use_unicode:
            wedge_symbol = "\u2227"
        else:
            wedge_symbol = '/\\'
        # 调用通用打印序列的方法，传入参数表达式和定制符号
        return self._print_seq(expr.args, None, None, wedge_symbol,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    # 定义一个方法，用于打印迹运算表达式
    def _print_Trace(self, e):
        # 获取迹运算参数的打印形式
        D = self._print(e.arg)
        # 使用圆括号包围打印形式
        D = prettyForm(*D.parens('(',')'))
        # 设置基线为高度的一半
        D.baseline = D.height() // 2
        # 在左侧添加 'tr' 字符串
        D = prettyForm(*D.left('\n' * (0) + 'tr'))
        return D

    # 定义一个方法，用于打印矩阵元素表达式
    def _print_MatrixElement(self, expr):
        # 导入矩阵符号类
        from sympy.matrices import MatrixSymbol
        # 如果父级对象是矩阵符号并且索引是数字
        if (isinstance(expr.parent, MatrixSymbol)
                and expr.i.is_number and expr.j.is_number):
            # 打印符号名加上索引组成的新符号
            return self._print(
                    Symbol(expr.parent.name + '_%d%d' % (expr.i, expr.j)))
        else:
            # 否则打印父级对象的打印形式，并设置为函数类型
            prettyFunc = self._print(expr.parent)
            prettyFunc = prettyForm(*prettyFunc.parens())
            # 打印索引元组并加上左右方括号
            prettyIndices = self._print_seq((expr.i, expr.j), delimiter=', '
                    ).parens(left='[', right=']')[0]
            # 创建一个带有函数绑定的打印形式对象
            pform = prettyForm(binding=prettyForm.FUNC,
                    *stringPict.next(prettyFunc, prettyIndices))

            # 存储 pform 的部分以便在需要时重新组装，例如在指数化时
            pform.prettyFunc = prettyFunc
            pform.prettyArgs = prettyIndices

            return pform
    # 定义一个方法来打印矩阵切片的表达式
    def _print_MatrixSlice(self, m):
        # XXX 只对应用函数有效
        from sympy.matrices import MatrixSymbol
        # 获取父矩阵的漂亮打印形式
        prettyFunc = self._print(m.parent)
        # 如果父矩阵不是MatrixSymbol类型，则将其漂亮打印形式用括号包裹
        if not isinstance(m.parent, MatrixSymbol):
            prettyFunc = prettyForm(*prettyFunc.parens())
        # 定义一个内部方法来处理切片的漂亮打印形式
        def ppslice(x, dim):
            x = list(x)
            # 如果步长为1，则删除步长的显示
            if x[2] == 1:
                del x[2]
            # 如果起始索引为0，则将其置为空
            if x[0] == 0:
                x[0] = ''
            # 如果结束索引与维度相同，则将其置为空
            if x[1] == dim:
                x[1] = ''
            # 使用逗号分隔来构造切片的漂亮打印形式
            return prettyForm(*self._print_seq(x, delimiter=':'))
        # 获取行切片和列切片的漂亮打印形式，用方括号括起来
        prettyArgs = self._print_seq((ppslice(m.rowslice, m.parent.rows),
            ppslice(m.colslice, m.parent.cols)), delimiter=', ').parens(left='[', right=']')[0]
        # 构造一个漂亮的函数表达式，包含函数名和参数
        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))
        # 存储pform的部分，以便在需要时重新组装，例如在求幂运算时
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs
        # 返回最终的漂亮打印形式
        return pform

    # 定义一个方法来打印矩阵的转置
    def _print_Transpose(self, expr):
        # 获取表达式的矩阵部分
        mat = expr.arg
        # 获取矩阵的漂亮打印形式
        pform = self._print(mat)
        from sympy.matrices import MatrixSymbol, BlockMatrix
        # 如果矩阵不是MatrixSymbol或BlockMatrix类型，且是MatrixExpr类型，则用括号包裹其漂亮打印形式
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            pform = prettyForm(*pform.parens())
        # 添加转置符号（T）到漂亮打印形式上
        pform = pform**(prettyForm('T'))
        # 返回最终的漂亮打印形式
        return pform

    # 定义一个方法来打印矩阵的伴随
    def _print_Adjoint(self, expr):
        # 获取表达式的矩阵部分
        mat = expr.arg
        # 获取矩阵的漂亮打印形式
        pform = self._print(mat)
        # 根据使用unicode的情况选择不同的伴随符号
        if self._use_unicode:
            dag = prettyForm(pretty_atom('Dagger'))
        else:
            dag = prettyForm('+')
        from sympy.matrices import MatrixSymbol, BlockMatrix
        # 如果矩阵不是MatrixSymbol或BlockMatrix类型，且是MatrixExpr类型，则用括号包裹其漂亮打印形式
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            pform = prettyForm(*pform.parens())
        # 添加伴随符号到漂亮打印形式上
        pform = pform**dag
        # 返回最终的漂亮打印形式
        return pform

    # 定义一个方法来打印块矩阵
    def _print_BlockMatrix(self, B):
        # 如果块矩阵只有一个块，则打印其唯一的块
        if B.blocks.shape == (1, 1):
            return self._print(B.blocks[0, 0])
        # 否则打印整个块矩阵
        return self._print(B.blocks)

    # 定义一个方法来打印矩阵加法
    def _print_MatAdd(self, expr):
        s = None
        # 遍历表达式的每一个项
        for item in expr.args:
            # 获取项的漂亮打印形式
            pform = self._print(item)
            # 如果是第一个元素，则直接赋值给s
            if s is None:
                s = pform     # 第一个元素
            else:
                # 获取项的系数
                coeff = item.as_coeff_mmul()[0]
                # 如果系数可能有负号，则在s后面加空格
                if S(coeff).could_extract_minus_sign():
                    s = prettyForm(*stringPict.next(s, ' '))
                    pform = self._print(item)
                else:
                    s = prettyForm(*stringPict.next(s, ' + '))
                # 将当前项的漂亮打印形式加到s后面
                s = prettyForm(*stringPict.next(s, pform))
        # 返回最终的漂亮打印形式
        return s
    # 定义一个方法，用于打印 MatMul 表达式
    def _print_MatMul(self, expr):
        # 将表达式的参数转换为列表
        args = list(expr.args)
        # 导入 HadamardProduct 和 KroneckerProduct 类
        from sympy.matrices.expressions.hadamard import HadamardProduct
        from sympy.matrices.expressions.kronecker import KroneckerProduct
        from sympy.matrices.expressions.matadd import MatAdd
        # 遍历参数列表
        for i, a in enumerate(args):
            # 如果参数是 Add、MatAdd、HadamardProduct 或 KroneckerProduct 的实例，并且参数数量大于 1
            if (isinstance(a, (Add, MatAdd, HadamardProduct, KroneckerProduct))
                    and len(expr.args) > 1):
                # 将参数 a 转换为可视化形式，并添加括号
                args[i] = prettyForm(*self._print(a).parens())
            else:
                # 否则，直接将参数转换为可视化形式
                args[i] = self._print(a)

        # 返回参数的乘积形式
        return prettyForm.__mul__(*args)

    # 定义一个方法，用于打印 Identity 表达式
    def _print_Identity(self, expr):
        # 如果设置使用 Unicode，则返回可视化的 IdentityMatrix
        if self._use_unicode:
            return prettyForm(pretty_atom('IdentityMatrix'))
        else:
            # 否则，返回简单的 'I'
            return prettyForm('I')

    # 定义一个方法，用于打印 ZeroMatrix 表达式
    def _print_ZeroMatrix(self, expr):
        # 如果设置使用 Unicode，则返回可视化的 ZeroMatrix
        if self._use_unicode:
            return prettyForm(pretty_atom('ZeroMatrix'))
        else:
            # 否则，返回简单的 '0'
            return prettyForm('0')

    # 定义一个方法，用于打印 OneMatrix 表达式
    def _print_OneMatrix(self, expr):
        # 如果设置使用 Unicode，则返回可视化的 OneMatrix
        if self._use_unicode:
            return prettyForm(pretty_atom("OneMatrix"))
        else:
            # 否则，返回简单的 '1'
            return prettyForm('1')

    # 定义一个方法，用于打印 DotProduct 表达式
    def _print_DotProduct(self, expr):
        # 将表达式的参数转换为列表
        args = list(expr.args)

        # 遍历参数列表，并将每个参数转换为可视化形式
        for i, a in enumerate(args):
            args[i] = self._print(a)

        # 返回参数的乘积形式
        return prettyForm.__mul__(*args)

    # 定义一个方法，用于打印 MatPow 表达式
    def _print_MatPow(self, expr):
        # 获取基底的可视化形式
        pform = self._print(expr.base)
        # 导入 MatrixSymbol 类
        from sympy.matrices import MatrixSymbol
        # 如果基底不是 MatrixSymbol 类型且是 MatrixExpr 类型，则将其括号化
        if not isinstance(expr.base, MatrixSymbol) and expr.base.is_MatrixExpr:
            pform = prettyForm(*pform.parens())
        # 返回基底的指数形式
        pform = pform**(self._print(expr.exp))
        return pform

    # 定义一个方法，用于打印 HadamardProduct 表达式
    def _print_HadamardProduct(self, expr):
        # 导入 HadamardProduct、MatAdd 和 MatMul 类
        from sympy.matrices.expressions.hadamard import HadamardProduct
        from sympy.matrices.expressions.matadd import MatAdd
        from sympy.matrices.expressions.matmul import MatMul
        # 如果设置使用 Unicode，则设置分隔符为 'Ring'
        if self._use_unicode:
            delim = pretty_atom('Ring')
        else:
            # 否则，设置分隔符为 '.*'
            delim = '.*'
        # 返回参数序列的可视化形式，根据类型添加括号
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul, HadamardProduct)))

    # 定义一个方法，用于打印 HadamardPower 表达式
    def _print_HadamardPower(self, expr):
        # 如果设置使用 Unicode，则设置循环符号为 'Ring'
        if self._use_unicode:
            circ = pretty_atom('Ring')
        else:
            # 否则，设置循环符号为 '.'
            circ = self._print('.')
        # 获取基底和指数的可视化形式
        pretty_base = self._print(expr.base)
        pretty_exp = self._print(expr.exp)
        # 如果指数的优先级小于 PRECEDENCE["Mul"]，则在其周围添加括号
        if precedence(expr.exp) < PRECEDENCE["Mul"]:
            pretty_exp = prettyForm(*pretty_exp.parens())
        # 创建循环的可视化形式，并返回基底的指数形式
        pretty_circ_exp = prettyForm(
            binding=prettyForm.LINE,
            *stringPict.next(circ, pretty_exp)
        )
        return pretty_base**pretty_circ_exp
    # 打印 Kronecker 乘积表达式的函数
    def _print_KroneckerProduct(self, expr):
        # 导入所需的模块
        from sympy.matrices.expressions.matadd import MatAdd
        from sympy.matrices.expressions.matmul import MatMul
        # 根据是否使用 Unicode 设置分隔符
        if self._use_unicode:
            delim = f" {pretty_atom('TensorProduct')} "
        else:
            delim = ' x '
        # 调用打印序列函数来处理表达式中的参数
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul)))

    # 打印函数矩阵的函数
    def _print_FunctionMatrix(self, X):
        # 打印 X.lamda.expr 的表达式
        D = self._print(X.lamda.expr)
        # 使用美化形式处理结果并添加方括号
        D = prettyForm(*D.parens('[', ']'))
        return D

    # 打印传递函数的函数
    def _print_TransferFunction(self, expr):
        # 如果分子部分不等于 1
        if not expr.num == 1:
            num, den = expr.num, expr.den
            # 计算传递函数的乘积形式
            res = Mul(num, Pow(den, -1, evaluate=False), evaluate=False)
            return self._print_Mul(res)
        else:
            # 打印 1 除以传递函数的分母部分
            return self._print(1)/self._print(expr.den)

    # 打印级数的函数
    def _print_Series(self, expr):
        args = list(expr.args)
        # 对表达式中的每个参数进行打印和美化处理
        for i, a in enumerate(expr.args):
            args[i] = prettyForm(*self._print(a).parens())
        return prettyForm.__mul__(*args)

    # 打印 MIMO 级数的函数
    def _print_MIMOSeries(self, expr):
        # 导入 MIMOParallel 类
        from sympy.physics.control.lti import MIMOParallel
        args = list(expr.args)
        pretty_args = []
        # 反向处理参数列表
        for a in reversed(args):
            # 如果参数是 MIMOParallel 类型并且参数数量大于 1
            if (isinstance(a, MIMOParallel) and len(expr.args) > 1):
                expression = self._print(a)
                expression.baseline = expression.height()//2
                pretty_args.append(prettyForm(*expression.parens()))
            else:
                expression = self._print(a)
                expression.baseline = expression.height()//2
                pretty_args.append(expression)
        return prettyForm.__mul__(*pretty_args)

    # 打印并行表达式的函数
    def _print_Parallel(self, expr):
        s = None
        # 对表达式中的每个元素进行处理
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # 第一个元素
            else:
                s = prettyForm(*stringPict.next(s))
                s.baseline = s.height()//2
                s = prettyForm(*stringPict.next(s, ' + '))
                s = prettyForm(*stringPict.next(s, pform))
        return s

    # 打印 MIMO 并行表达式的函数
    def _print_MIMOParallel(self, expr):
        # 导入 TransferFunctionMatrix 类
        from sympy.physics.control.lti import TransferFunctionMatrix
        s = None
        # 对表达式中的每个元素进行处理
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # 第一个元素
            else:
                s = prettyForm(*stringPict.next(s))
                s.baseline = s.height()//2
                s = prettyForm(*stringPict.next(s, ' + '))
                # 如果当前元素是 TransferFunctionMatrix 类型，则调整基线位置
                if isinstance(item, TransferFunctionMatrix):
                    s.baseline = s.height() - 1
                s = prettyForm(*stringPict.next(s, pform))
            # s.baseline = s.height()//2
        return s
    # 打印单输入单输出控制系统的反馈表达式
    def _print_Feedback(self, expr):
        # 导入所需模块
        from sympy.physics.control import TransferFunction, Series

        # 提取系统的分子和分母
        num, tf = expr.sys1, TransferFunction(1, 1, expr.var)

        # 如果分子是 Series 类型，则转换为列表，否则保持单个元素列表
        num_arg_list = list(num.args) if isinstance(num, Series) else [num]
        # 如果分母是 Series 类型，则转换为列表，否则保持单个元素列表
        den_arg_list = list(expr.sys2.args) if isinstance(expr.sys2, Series) else [expr.sys2]

        # 根据不同情况构造 Series 对象 den
        if isinstance(num, Series) and isinstance(expr.sys2, Series):
            den = Series(*num_arg_list, *den_arg_list)
        elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
            if expr.sys2 == tf:
                den = Series(*num_arg_list)
            else:
                den = Series(*num_arg_list, expr.sys2)
        elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
            if num == tf:
                den = Series(*den_arg_list)
            else:
                den = Series(num, *den_arg_list)
        else:
            if num == tf:
                den = Series(*den_arg_list)
            elif expr.sys2 == tf:
                den = Series(*num_arg_list)
            else:
                den = Series(*num_arg_list, *den_arg_list)

        # 构造分母的打印形式
        denom = prettyForm(*stringPict.next(self._print(tf)))
        denom.baseline = denom.height()//2
        # 根据符号选择连接符号是加号还是减号
        denom = prettyForm(*stringPict.next(denom, ' + ')) if expr.sign == -1 \
            else prettyForm(*stringPict.next(denom, ' - '))
        # 添加 den 的打印形式到分母
        denom = prettyForm(*stringPict.next(denom, self._print(den)))

        # 返回分子除以分母的打印形式
        return self._print(num)/denom

    # 打印多输入多输出控制系统的反馈表达式
    def _print_MIMOFeedback(self, expr):
        # 导入所需模块
        from sympy.physics.control import MIMOSeries, TransferFunctionMatrix

        # 打印 MIMO 系统的反馈表达式
        inv_mat = self._print(MIMOSeries(expr.sys2, expr.sys1))
        plant = self._print(expr.sys1)
        # 初始化反馈表达式
        _feedback = prettyForm(*stringPict.next(inv_mat))
        # 根据符号选择 "I + " 或 "I - " 作为开头
        _feedback = prettyForm(*stringPict.right("I + ", _feedback)) if expr.sign == -1 \
            else prettyForm(*stringPict.right("I - ", _feedback))
        # 将反馈表达式包裹在括号内
        _feedback = prettyForm(*stringPict.parens(_feedback))
        _feedback.baseline = 0
        # 添加 "-1 " 到反馈表达式的右侧
        _feedback = prettyForm(*stringPict.right(_feedback, '-1 '))
        _feedback.baseline = _feedback.height()//2
        _feedback = prettyForm.__mul__(_feedback, prettyForm(" "))
        # 如果系统是 TransferFunctionMatrix，则将 baseline 设置为合适的高度
        if isinstance(expr.sys1, TransferFunctionMatrix):
            _feedback.baseline = _feedback.height() - 1
        # 将 plant 添加到反馈表达式的右侧
        _feedback = prettyForm(*stringPict.next(_feedback, plant))
        # 返回最终的反馈表达式
        return _feedback

    # 打印传递函数矩阵的表达式
    def _print_TransferFunctionMatrix(self, expr):
        # 打印表达式的矩阵部分
        mat = self._print(expr._expr_mat)
        mat.baseline = mat.height() - 1
        # 添加下标到矩阵的右侧
        subscript = greek_unicode['tau'] if self._use_unicode else r'{t}'
        mat = prettyForm(*mat.right(subscript))
        # 返回最终的打印形式
        return mat

    # 打印状态空间表达式
    def _print_StateSpace(self, expr):
        # 导入所需模块
        from sympy.matrices.expressions.blockmatrix import BlockMatrix
        # 分别获取状态空间表达式的 A, B, C, D 矩阵
        A = expr._A
        B = expr._B
        C = expr._C
        D = expr._D
        # 构造块矩阵对象 mat
        mat = BlockMatrix([[A, B], [C, D]])
        # 返回块矩阵的打印形式
        return self._print(mat.blocks)
    # 定义一个方法用于打印 N 维数组的表示形式
    def _print_NDimArray(self, expr):
        # 导入不可变矩阵的定义
        from sympy.matrices.immutable import ImmutableMatrix

        # 如果表达式的秩为 0，直接返回其标量值的打印结果
        if expr.rank() == 0:
            return self._print(expr[()])

        # 初始化用于存储每个维度的打印字符串的列表
        level_str = [[]] + [[] for i in range(expr.rank())]
        # 生成每个维度的索引范围
        shape_ranges = [list(range(i)) for i in expr.shape]
        # 定义一个函数 mat，用于将列表转换为不可变矩阵
        mat = lambda x: ImmutableMatrix(x, evaluate=False)
        
        # 使用 itertools 模块的 product 方法生成所有可能的索引组合
        for outer_i in itertools.product(*shape_ranges):
            # 将每个索引组合对应位置的元素添加到最内层的打印字符串中
            level_str[-1].append(expr[outer_i])
            even = True
            # 逆序遍历每个维度，进行打印字符串的构建
            for back_outer_i in range(expr.rank()-1, -1, -1):
                # 如果某个维度的打印字符串长度小于其长度，退出循环
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                # 根据 even 变量的值选择添加列表或矩阵到当前维度的打印字符串
                if even:
                    level_str[back_outer_i].append(level_str[back_outer_i+1])
                else:
                    level_str[back_outer_i].append(mat(
                        level_str[back_outer_i+1]))
                    # 如果该维度的打印字符串长度为 1，将其包装成矩阵形式
                    if len(level_str[back_outer_i + 1]) == 1:
                        level_str[back_outer_i][-1] = mat(
                            [[level_str[back_outer_i][-1]]])
                even = not even
                # 清空下一个维度的打印字符串列表
                level_str[back_outer_i+1] = []

        # 获取最终的打印结果
        out_expr = level_str[0][0]
        # 如果表达式的秩为奇数，将最终结果包装成矩阵形式
        if expr.rank() % 2 == 1:
            out_expr = mat([out_expr])

        # 返回最终的打印结果
        return self._print(out_expr)

    # 定义一个方法用于打印张量的表示形式
    def _print_Tensor(self, expr):
        # 获取张量的名称和索引
        name = expr.args[0].name
        indices = expr.get_indices()
        # 调用打印张量索引的方法，并返回打印结果
        return self._printer_tensor_indices(name, indices)

    # 定义一个方法用于打印张量的索引
    def _printer_tensor_indices(self, name, indices, index_map={}):
        # 创建中心部分的字符串画布
        center = stringPict(name)
        # 创建顶部和底部的空白字符串画布
        top = stringPict(" "*center.width())
        bot = stringPict(" "*center.width())

        # 初始化变量
        last_valence = None
        prev_map = None

        # 遍历所有的索引
        for index in indices:
            # 获取索引对应的字符串画布
            indpic = self._print(index.args[0])
            # 如果索引在索引映射中，添加等号和映射后的值到画布
            if ((index in index_map) or prev_map) and last_valence == index.is_up:
                if index.is_up:
                    top = prettyForm(*stringPict.next(top, ","))
                else:
                    bot = prettyForm(*stringPict.next(bot, ","))
            if index in index_map:
                indpic = prettyForm(*stringPict.next(indpic, "="))
                indpic = prettyForm(*stringPict.next(indpic, self._print(index_map[index])))
                prev_map = True
            else:
                prev_map = False
            # 根据索引的方向更新顶部、中心和底部的字符串画布
            if index.is_up:
                top = stringPict(*top.right(indpic))
                center = stringPict(*center.right(" "*indpic.width()))
                bot = stringPict(*bot.right(" "*indpic.width()))
            else:
                bot = stringPict(*bot.right(indpic))
                center = stringPict(*center.right(" "*indpic.width()))
                top = stringPict(*top.right(" "*indpic.width()))
            last_valence = index.is_up

        # 组合最终的字符串画布
        pict = prettyForm(*center.above(top))
        pict = prettyForm(*pict.below(bot))
        # 返回最终的字符串画布
        return pict
    # 打印 TensorElement 对象的表达式
    def _print_TensorElement(self, expr):
        # 提取表达式的名称
        name = expr.expr.args[0].name
        # 获取表达式的索引
        indices = expr.expr.get_indices()
        # 获取索引映射
        index_map = expr.index_map
        # 调用特定方法打印张量及其索引
        return self._printer_tensor_indices(name, indices, index_map)

    # 打印 TensMul 对象的表达式
    def _print_TensMul(self, expr):
        # 获取符号和参数
        sign, args = expr._get_args_for_traditional_printer()
        # 打印每个参数
        args = [
            prettyForm(*self._print(i).parens()) if
            precedence_traditional(i) < PRECEDENCE["Mul"] else self._print(i)
            for i in args
        ]
        # 创建乘法形式
        pform = prettyForm.__mul__(*args)
        # 如果有符号，添加符号到左侧
        if sign:
            return prettyForm(*pform.left(sign))
        else:
            return pform

    # 打印 TensAdd 对象的表达式
    def _print_TensAdd(self, expr):
        # 打印每个参数
        args = [
            prettyForm(*self._print(i).parens()) if
            precedence_traditional(i) < PRECEDENCE["Mul"] else self._print(i)
            for i in expr.args
        ]
        # 创建加法形式
        return prettyForm.__add__(*args)

    # 打印 TensorIndex 对象的表达式
    def _print_TensorIndex(self, expr):
        # 获取符号
        sym = expr.args[0]
        # 如果不是上标，取其相反数
        if not expr.is_up:
            sym = -sym
        # 打印符号
        return self._print(sym)

    # 打印 PartialDerivative 对象的表达式
    def _print_PartialDerivative(self, deriv):
        # 根据设置选择不同的偏导数符号
        if self._use_unicode:
            deriv_symbol = U('PARTIAL DIFFERENTIAL')
        else:
            deriv_symbol = r'd'
        # 初始化变量为 None
        x = None

        # 遍历反向的偏导变量
        for variable in reversed(deriv.variables):
            # 打印变量
            s = self._print(variable)
            # 将偏导数符号添加到左侧
            ds = prettyForm(*s.left(deriv_symbol))

            # 如果 x 为空，直接赋值为 ds
            if x is None:
                x = ds
            else:
                # 添加空格和 ds 到 x 的右侧
                x = prettyForm(*x.right(' '))
                x = prettyForm(*x.right(ds))

        # 打印表达式
        f = prettyForm(
            binding=prettyForm.FUNC, *self._print(deriv.expr).parens())

        # 初始化偏导数形式
        pform = prettyForm(deriv_symbol)

        # 如果偏导数变量大于 1，添加指数
        if len(deriv.variables) > 1:
            pform = pform**self._print(len(deriv.variables))

        # 在 x 下方添加分隔线
        pform = prettyForm(*pform.below(stringPict.LINE, x))
        # 调整基线
        pform.baseline = pform.baseline + 1
        # 在 f 的下方添加 pform
        pform = prettyForm(*stringPict.next(pform, f))
        # 设置绑定为乘法
        pform.binding = prettyForm.MUL

        return pform
    # 打印分段函数表达式的格式化输出
    def _print_Piecewise(self, pexpr):
        # 初始化空字典 P，用于存储格式化后的表达式和条件
        P = {}
        # 遍历分段函数表达式的每个分段
        for n, ec in enumerate(pexpr.args):
            # 将分段表达式格式化输出并存储在 P 中
            P[n, 0] = self._print(ec.expr)
            # 如果条件为真（True），则使用 'otherwise' 作为条件的格式化输出，否则格式化输出条件
            if ec.cond == True:
                P[n, 1] = prettyForm('otherwise')
            else:
                P[n, 1] = prettyForm(
                    *prettyForm('for ').right(self._print(ec.cond)))
        
        # 设置水平和垂直分隔符的间距
        hsep = 2
        vsep = 1
        len_args = len(pexpr.args)

        # 计算每列的最大宽度
        maxw = [max(P[i, j].width() for i in range(len_args))
                for j in range(2)]

        # FIXME: 重构此代码和矩阵，使其进入某些表格环境中。
        # 绘制结果
        D = None

        # 遍历处理每行和每列
        for i in range(len_args):
            D_row = None
            for j in range(2):
                p = P[i, j]
                # 断言格式化输出的宽度不超过对应列的最大宽度
                assert p.width() <= maxw[j]

                # 计算左右两侧的空白填充量
                wdelta = maxw[j] - p.width()
                wleft = wdelta // 2
                wright = wdelta - wleft

                # 右侧填充空白
                p = prettyForm(*p.right(' '*wright))
                # 左侧填充空白
                p = prettyForm(*p.left(' '*wleft))

                # 如果是第一行，则直接赋值为当前行
                if D_row is None:
                    D_row = p
                    continue

                # 添加水平间隔符
                D_row = prettyForm(*D_row.right(' '*hsep))  # h-spacer
                # 将当前格式化输出添加到当前行
                D_row = prettyForm(*D_row.right(p))
            
            # 如果是第一行，则直接赋值为当前行
            if D is None:
                D = D_row       # first row in a picture
                continue

            # 添加垂直间隔符
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))

            # 将当前行添加到总体输出 D 中
            D = prettyForm(*D.below(D_row))

        # 添加花括号和空格
        D = prettyForm(*D.parens('{', ''))
        # 设置基线为高度的一半
        D.baseline = D.height()//2
        D.binding = prettyForm.OPEN
        # 返回格式化后的输出
        return D

    # 打印 if-then-else（ITE）表达式的格式化输出
    def _print_ITE(self, ite):
        # 导入分段函数类
        from sympy.functions.elementary.piecewise import Piecewise
        # 调用分段函数的重写方法并返回其格式化输出
        return self._print(ite.rewrite(Piecewise))

    # 横向打印向量的格式化输出
    def _hprint_vec(self, v):
        D = None

        # 遍历向量中的每个元素并格式化输出
        for a in v:
            p = a
            # 如果是第一个元素，则直接赋值为当前元素
            if D is None:
                D = p
            else:
                # 添加逗号和空格分隔符，并将当前元素追加到输出结果中
                D = prettyForm(*D.right(', '))
                D = prettyForm(*D.right(p))
        
        # 如果没有元素，则返回空格字符串
        if D is None:
            D = stringPict(' ')

        # 返回格式化后的向量输出
        return D

    # 横向打印两个表达式的格式化输出，并添加垂直分隔符
    def _hprint_vseparator(self, p1, p2, left=None, right=None, delimiter='', ifascii_nougly=False):
        # 如果需要 ASCII 输出且不使用 Unicode，则调用特定的序列打印方法
        if ifascii_nougly and not self._use_unicode:
            return self._print_seq((p1, '|', p2), left=left, right=right,
                                   delimiter=delimiter, ifascii_nougly=True)
        
        # 否则，调用通用的序列打印方法并添加垂直分隔符
        tmp = self._print_seq((p1, p2,), left=left, right=right, delimiter=delimiter)
        sep = stringPict(vobj('|', tmp.height()), baseline=tmp.baseline)
        return self._print_seq((p1, sep, p2), left=left, right=right,
                               delimiter=delimiter)
    # 定义一个方法 `_print_hyper`，用于打印超函环境表达式 `e`
    def _print_hyper(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment
        # 获取 `e` 中属性 `ap` 的每个元素的打印形式
        ap = [self._print(a) for a in e.ap]
        # 获取 `e` 中属性 `bq` 的每个元素的打印形式
        bq = [self._print(b) for b in e.bq]

        # 打印 `e` 的主参数 `argument` 的表达式形式，并设置其基线为高度的一半
        P = self._print(e.argument)
        P.baseline = P.height() // 2

        # 绘制结果 - 首先创建 `ap` 和 `bq` 向量
        D = None
        for v in [ap, bq]:
            # 将每个向量 `v` 转换为水平打印形式
            D_row = self._hprint_vec(v)
            if D is None:
                D = D_row       # 第一行在图片中
            else:
                D = prettyForm(*D.below(' '))
                D = prettyForm(*D.below(D_row))

        # 确保参数 `z` 在垂直方向上居中
        D.baseline = D.height() // 2

        # 插入水平分隔线
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # 插入分隔符 `|`
        D = self._hprint_vseparator(D, P)

        # 添加括号
        D = prettyForm(*D.parens('(', ')'))

        # 创建符号 `F`
        above = D.height() // 2 - 1
        below = D.height() - above - 1

        # annotated 函数返回的结果：sz, t, b, add, img，用于创建 `F` 的打印形式
        sz, t, b, add, img = annotated('F')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)
        add = (sz + 1) // 2

        # 在 `F` 的左侧添加 `len(e.ap)` 的打印形式
        F = prettyForm(*F.left(self._print(len(e.ap))))
        # 在 `F` 的右侧添加 `len(e.bq)` 的打印形式
        F = prettyForm(*F.right(self._print(len(e.bq))))
        F.baseline = above + add

        # 将 `F` 放置在 `D` 的右侧，与空格分隔
        D = prettyForm(*F.right(' ', D))

        # 返回最终的打印形式 `D`
        return D
    def _print_meijerg(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment
        # 定义一个空字典v，用于存储元素e的打印结果
        v = {}
        # 将元素e的an属性中的每个元素打印后存储在v[(0, 0)]中
        v[(0, 0)] = [self._print(a) for a in e.an]
        # 将元素e的aother属性中的每个元素打印后存储在v[(0, 1)]中
        v[(0, 1)] = [self._print(a) for a in e.aother]
        # 将元素e的bm属性中的每个元素打印后存储在v[(1, 0)]中
        v[(1, 0)] = [self._print(b) for b in e.bm]
        # 将元素e的bother属性中的每个元素打印后存储在v[(1, 1)]中
        v[(1, 1)] = [self._print(b) for b in e.bother]

        # 打印元素e的argument属性，并设置基线为其高度的一半
        P = self._print(e.argument)
        P.baseline = P.height() // 2

        # 定义一个空字典vp，用于存储v中每个键对应的水平打印结果
        vp = {}
        # 遍历v中的键
        for idx in v:
            # 将v[idx]中的打印结果进行水平打印，并存储在vp[idx]中
            vp[idx] = self._hprint_vec(v[idx])

        # 对于每个列（i取值0和1）
        for i in range(2):
            # 计算每列中最宽的vp[(0, i)]和vp[(1, i)]的宽度
            maxw = max(vp[(0, i)].width(), vp[(1, i)].width())
            # 对于每个行（j取值0和1）
            for j in range(2):
                # 获取vp[(j, i)]的打印结果
                s = vp[(j, i)]
                # 计算左侧和右侧的空格数，使得s在最宽为maxw的区域内居中
                left = (maxw - s.width()) // 2
                right = maxw - left - s.width()
                # 在s左侧和右侧添加空格，重新赋值给s
                s = prettyForm(*s.left(' ' * left))
                s = prettyForm(*s.right(' ' * right))
                # 更新vp[(j, i)]为居中后的s
                vp[(j, i)] = s

        # 创建第一行的prettyForm对象D1，包括vp[(0, 0)]右侧2个空格和vp[(0, 1)]
        D1 = prettyForm(*vp[(0, 0)].right('  ', vp[(0, 1)]))
        # 在D1下方添加一个空行
        D1 = prettyForm(*D1.below(' '))
        # 创建第二行的prettyForm对象D2，包括vp[(1, 0)]右侧2个空格和vp[(1, 1)]
        D2 = prettyForm(*vp[(1, 0)].right('  ', vp[(1, 1)]))
        # 将D2放在D1的下方，形成一个新的prettyForm对象D
        D = prettyForm(*D1.below(D2))

        # 确保参数'z'在垂直方向上居中
        D.baseline = D.height() // 2

        # 在P的左侧插入一个空格，形成新的prettyForm对象P
        P = prettyForm(*P.left(' '))
        # 在D的右侧插入一个空格，形成新的prettyForm对象D
        D = prettyForm(*D.right(' '))

        # 在D和P之间插入分隔符`|`
        D = self._hprint_vseparator(D, P)

        # 给D两侧添加括号
        D = prettyForm(*D.parens('(', ')'))

        # 创建符号G的prettyForm对象F
        above = D.height() // 2 - 1
        below = D.height() - above - 1
        sz, t, b, add, img = annotated('G')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)

        # 分别打印e.ap、e.bq、e.bm、e.an的长度，并将结果存储在pp、pq、pm、pn中
        pp = self._print(len(e.ap))
        pq = self._print(len(e.bq))
        pm = self._print(len(e.bm))
        pn = self._print(len(e.an))

        # 调整pp和pm的宽度，使它们相等
        def adjust(p1, p2):
            diff = p1.width() - p2.width()
            if diff == 0:
                return p1, p2
            elif diff > 0:
                return p1, prettyForm(*p2.left(' ' * diff))
            else:
                return prettyForm(*p1.left(' ' * -diff)), p2
        pp, pm = adjust(pp, pm)
        # 调整pq和pn的宽度，使它们相等
        pq, pn = adjust(pq, pn)
        # 将pm和pn右侧添加逗号后组合成新的prettyForm对象pu
        pu = prettyForm(*pm.right(', ', pn))
        # 将pp和pq右侧添加逗号后组合成新的prettyForm对象pl
        pl = prettyForm(*pp.right(', ', pq))

        # 计算F的基线位置和D上方填充的行数，将pu放置在其下方，形成新的prettyForm对象p
        ht = F.baseline - above - 2
        if ht > 0:
            pu = prettyForm(*pu.below('\n' * ht))
        p = prettyForm(*pu.below(pl))

        # 调整F的基线位置，并将p放在其右侧，形成新的prettyForm对象F
        F.baseline = above
        F = prettyForm(*F.right(p))

        # 调整F的基线位置，并将D放在其右侧，形成新的prettyForm对象D
        F.baseline = above + add
        D = prettyForm(*F.right(' ', D))

        # 返回最终的prettyForm对象D
        return D

    def _print_ExpBase(self, e):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        # 创建一个基本指数表达式的prettyForm对象base，用于打印'e'
        base = prettyForm(pretty_atom('Exp1', 'e'))
        # 返回base的幂指数为e.args[0]的prettyForm对象
        return base ** self._print(e.args[0])

    def _print_Exp1(self, e):
        # 返回一个带有'Exp1'标签的prettyForm对象，表示指数1
        return prettyForm(pretty_atom('Exp1', 'e'))
    # 定义一个方法来打印数学函数表达式，支持排序和自定义函数名
    def _print_Function(self, e, sort=False, func_name=None, left='(', right=')'):
        # 如果需要，可以通过 func_name 参数提供自定义函数名
        # XXX 只适用于应用函数
        return self._helper_print_function(e.func, e.args, sort=sort, func_name=func_name, left=left, right=right)

    # 打印数学表达式，指定函数名为 'C'
    def _print_mathieuc(self, e):
        return self._print_Function(e, func_name='C')

    # 打印数学表达式，指定函数名为 'S'
    def _print_mathieus(self, e):
        return self._print_Function(e, func_name='S')

    # 打印数学表达式，指定函数名为 "C'"
    def _print_mathieucprime(self, e):
        return self._print_Function(e, func_name="C'")

    # 打印数学表达式，指定函数名为 "S'"
    def _print_mathieusprime(self, e):
        return self._print_Function(e, func_name="S'")

    # 辅助方法：打印函数表达式的详细信息，支持排序、自定义函数名、分隔符等
    def _helper_print_function(self, func, args, sort=False, func_name=None, delimiter=', ', elementwise=False, left='(', right=')'):
        # 如果需要排序参数
        if sort:
            args = sorted(args, key=default_sort_key)

        # 如果未指定函数名但是函数对象有 __name__ 属性，则使用函数对象的名称
        if not func_name and hasattr(func, "__name__"):
            func_name = func.__name__

        # 如果有指定函数名，则打印函数名
        if func_name:
            prettyFunc = self._print(Symbol(func_name))
        else:
            prettyFunc = prettyForm(*self._print(func).parens())

        # 如果需要元素级别的打印
        if elementwise:
            if self._use_unicode:
                circ = pretty_atom('Modifier Letter Low Ring')
            else:
                circ = '.'
            circ = self._print(circ)
            prettyFunc = prettyForm(
                binding=prettyForm.LINE,
                *stringPict.next(prettyFunc, circ)
            )

        # 打印参数列表，并使用指定的分隔符、左右括号
        prettyArgs = prettyForm(*self._print_seq(args, delimiter=delimiter).parens(
                                                 left=left, right=right))

        # 组合最终的打印结果
        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # 存储 pform 的部分以便后续重组（比如在指数运算时）
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform

    # 打印应用在每个元素上的函数表达式
    def _print_ElementwiseApplyFunction(self, e):
        func = e.function
        arg = e.expr
        args = [arg]
        return self._helper_print_function(func, args, delimiter="", elementwise=True)
    # 返回特殊函数类及其对应的打印样式
    def _special_function_classes(self):
        # 导入所需的特殊函数类
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.zeta_functions import lerchphi
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        # 返回特殊函数类及其对应的打印样式的字典
        return {KroneckerDelta: [greek_unicode['delta'], 'delta'],
                gamma: [greek_unicode['Gamma'], 'Gamma'],
                lerchphi: [greek_unicode['Phi'], 'lerchphi'],
                lowergamma: [greek_unicode['gamma'], 'gamma'],
                beta: [greek_unicode['Beta'], 'B'],
                DiracDelta: [greek_unicode['delta'], 'delta'],
                Chi: ['Chi', 'Chi']}

    # 打印特定函数类的表达式
    def _print_FunctionClass(self, expr):
        # 遍历所有特殊函数类
        for cls in self._special_function_classes:
            # 如果给定表达式是特殊函数类的子类且名称匹配
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                # 如果使用 Unicode，返回特殊函数类对应的 Unicode 样式
                if self._use_unicode:
                    return prettyForm(self._special_function_classes[cls][0])
                else:
                    # 否则返回特殊函数类对应的非 Unicode 样式
                    return prettyForm(self._special_function_classes[cls][1])
        # 如果未找到匹配的特殊函数类，则返回表达式名称的打印形式
        func_name = expr.__name__
        return prettyForm(pretty_symbol(func_name))

    # 打印 GeometryEntity 类的表达式
    def _print_GeometryEntity(self, expr):
        # GeometryEntity 基于 Tuple，但不应像 Tuple 一样打印
        return self.emptyPrinter(expr)

    # 打印 polylog 函数的表达式
    def _print_polylog(self, e):
        # 获取 polylog 函数的下标
        subscript = self._print(e.args[0])
        # 如果使用 Unicode 且下标支持 Unicode 样式，则打印为 Li_下标
        if self._use_unicode and is_subscriptable_in_unicode(subscript):
            return self._print_Function(Function('Li_%s' % subscript)(e.args[1]))
        # 否则直接打印 polylog 函数
        return self._print_Function(e)

    # 打印 lerchphi 函数的表达式
    def _print_lerchphi(self, e):
        # 获取函数名，根据使用 Unicode 决定返回的函数名样式
        func_name = greek_unicode['Phi'] if self._use_unicode else 'lerchphi'
        return self._print_Function(e, func_name=func_name)

    # 打印 dirichlet_eta 函数的表达式
    def _print_dirichlet_eta(self, e):
        # 获取函数名，根据使用 Unicode 决定返回的函数名样式
        func_name = greek_unicode['eta'] if self._use_unicode else 'dirichlet_eta'
        return self._print_Function(e, func_name=func_name)

    # 打印 Heaviside 函数的表达式
    def _print_Heaviside(self, e):
        # 获取函数名，根据使用 Unicode 决定返回的函数名样式
        func_name = greek_unicode['theta'] if self._use_unicode else 'Heaviside'
        # 如果第二个参数是 S.Half，则特殊处理打印格式
        if e.args[1] is S.Half:
            pform = prettyForm(*self._print(e.args[0]).parens())
            pform = prettyForm(*pform.left(func_name))
            return pform
        else:
            # 否则按普通函数处理打印格式
            return self._print_Function(e, func_name=func_name)

    # 打印 fresnels 函数的表达式
    def _print_fresnels(self, e):
        return self._print_Function(e, func_name="S")

    # 打印 fresnelc 函数的表达式
    def _print_fresnelc(self, e):
        return self._print_Function(e, func_name="C")

    # 打印 airyai 函数的表达式
    def _print_airyai(self, e):
        return self._print_Function(e, func_name="Ai")

    # 打印 airybi 函数的表达式
    def _print_airybi(self, e):
        return self._print_Function(e, func_name="Bi")

    # 打印 airyaiprime 函数的表达式
    def _print_airyaiprime(self, e):
        return self._print_Function(e, func_name="Ai'")
    # 打印 Bi' 函数
    def _print_airybiprime(self, e):
        return self._print_Function(e, func_name="Bi'")

    # 打印 LambertW 函数
    def _print_LambertW(self, e):
        return self._print_Function(e, func_name="W")

    # 打印 Covariance 函数
    def _print_Covariance(self, e):
        return self._print_Function(e, func_name="Cov")

    # 打印 Variance 函数
    def _print_Variance(self, e):
        return self._print_Function(e, func_name="Var")

    # 打印 Probability 函数
    def _print_Probability(self, e):
        return self._print_Function(e, func_name="P")

    # 打印 Expectation 函数，定制左右分隔符为 '[' 和 ']'
    def _print_Expectation(self, e):
        return self._print_Function(e, func_name="E", left='[', right=']')

    # 打印 Lambda 函数
    def _print_Lambda(self, e):
        expr = e.expr
        sig = e.signature
        if self._use_unicode:
            arrow = f" {pretty_atom('ArrowFromBar')} "
        else:
            arrow = " -> "
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        var_form = self._print(sig)
        # 格式化 Lambda 表达式打印
        return prettyForm(*stringPict.next(var_form, arrow, self._print(expr)), binding=8)

    # 打印 Order 函数
    def _print_Order(self, expr):
        pform = self._print(expr.expr)
        if (expr.point and any(p != S.Zero for p in expr.point)) or \
           len(expr.variables) > 1:
            pform = prettyForm(*pform.right("; "))
            if len(expr.variables) > 1:
                pform = prettyForm(*pform.right(self._print(expr.variables)))
            elif len(expr.variables):
                pform = prettyForm(*pform.right(self._print(expr.variables[0])))
            if self._use_unicode:
                pform = prettyForm(*pform.right(f" {pretty_atom('Arrow')} "))
            else:
                pform = prettyForm(*pform.right(" -> "))
            if len(expr.point) > 1:
                pform = prettyForm(*pform.right(self._print(expr.point)))
            else:
                pform = prettyForm(*pform.right(self._print(expr.point[0])))
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left("O"))
        return pform

    # 打印 SingularityFunction 函数
    def _print_SingularityFunction(self, e):
        if self._use_unicode:
            shift = self._print(e.args[0]-e.args[1])
            n = self._print(e.args[2])
            base = prettyForm("<")
            base = prettyForm(*base.right(shift))
            base = prettyForm(*base.right(">"))
            pform = base**n
            return pform
        else:
            n = self._print(e.args[2])
            shift = self._print(e.args[0]-e.args[1])
            base = self._print_seq(shift, "<", ">", ' ')
            return base**n

    # 打印 Beta 函数，根据使用情况选择希腊字母符号或字母 'B'
    def _print_beta(self, e):
        func_name = greek_unicode['Beta'] if self._use_unicode else 'B'
        return self._print_Function(e, func_name=func_name)

    # 打印 betainc 函数，使用名称 "B'"
    def _print_betainc(self, e):
        func_name = "B'"
        return self._print_Function(e, func_name=func_name)

    # 打印 betainc_regularized 函数，使用名称 'I'
    def _print_betainc_regularized(self, e):
        func_name = 'I'
        return self._print_Function(e, func_name=func_name)
    # 打印 Gamma 函数，根据设置决定使用希腊字母或文本表示
    def _print_gamma(self, e):
        func_name = greek_unicode['Gamma'] if self._use_unicode else 'Gamma'
        # 调用通用的打印函数来输出 Gamma 函数
        return self._print_Function(e, func_name=func_name)

    # 打印 UpperGamma 函数，根据设置决定使用希腊字母或文本表示
    def _print_uppergamma(self, e):
        func_name = greek_unicode['Gamma'] if self._use_unicode else 'Gamma'
        # 调用通用的打印函数来输出 UpperGamma 函数
        return self._print_Function(e, func_name=func_name)

    # 打印 LowerGamma 函数，根据设置决定使用希腊字母或文本表示
    def _print_lowergamma(self, e):
        func_name = greek_unicode['gamma'] if self._use_unicode else 'lowergamma'
        # 调用通用的打印函数来输出 LowerGamma 函数
        return self._print_Function(e, func_name=func_name)

    # 打印 DiracDelta 函数，根据设置决定使用希腊字母或文本表示
    def _print_DiracDelta(self, e):
        if self._use_unicode:
            # 如果使用 Unicode，处理特殊格式化，包括 Delta 符号和参数
            if len(e.args) == 2:
                a = prettyForm(greek_unicode['delta'])
                b = self._print(e.args[1])
                b = prettyForm(*b.parens())
                c = self._print(e.args[0])
                c = prettyForm(*c.parens())
                pform = a**b
                pform = prettyForm(*pform.right(' '))
                pform = prettyForm(*pform.right(c))
                return pform
            # 单一参数情况下，只输出 Delta 符号和其参数
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens())
            pform = prettyForm(*pform.left(greek_unicode['delta']))
            return pform
        else:
            # 如果不使用 Unicode，调用通用的打印函数来输出 DiracDelta 函数
            return self._print_Function(e)

    # 打印 expint 函数，根据设置决定使用希腊字母或文本表示
    def _print_expint(self, e):
        subscript = self._print(e.args[0])
        if self._use_unicode and is_subscriptable_in_unicode(subscript):
            # 使用 Unicode 表示时，处理指数部分的格式化
            return self._print_Function(Function('E_%s' % subscript)(e.args[1]))
        # 调用通用的打印函数来输出 expint 函数
        return self._print_Function(e)

    # 打印 Chi 函数，以避免希腊字母 Chi 被误认为希腊字母
    def _print_Chi(self, e):
        prettyFunc = prettyForm("Chi")
        prettyArgs = prettyForm(*self._print_seq(e.args).parens())
        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))
        # 存储 pform 的部分以便稍后重新组装，例如在上方加指数时
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs
        return pform

    # 打印 elliptic_e 函数，输出以 E 为开头的特定格式化
    def _print_elliptic_e(self, e):
        pforma0 = self._print(e.args[0])
        if len(e.args) == 1:
            pform = pforma0
        else:
            pforma1 = self._print(e.args[1])
            pform = self._hprint_vseparator(pforma0, pforma1)
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('E'))
        return pform

    # 打印 elliptic_k 函数，输出以 K 为开头的特定格式化
    def _print_elliptic_k(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('K'))
        return pform

    # 打印 elliptic_f 函数，输出以 F 为开头的特定格式化
    def _print_elliptic_f(self, e):
        pforma0 = self._print(e.args[0])
        pforma1 = self._print(e.args[1])
        pform = self._hprint_vseparator(pforma0, pforma1)
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('F'))
        return pform
    # 打印椭圆函数 Pi 表示符号，根据是否使用 Unicode 设置名称
    def _print_elliptic_pi(self, e):
        # 根据是否使用 Unicode 设置 π 符号的名称
        name = greek_unicode['Pi'] if self._use_unicode else 'Pi'
        # 打印第一个参数
        pforma0 = self._print(e.args[0])
        # 打印第二个参数
        pforma1 = self._print(e.args[1])
        # 如果参数个数为2，则使用水平打印函数连接 pforma0 和 pforma1
        if len(e.args) == 2:
            pform = self._hprint_vseparator(pforma0, pforma1)
        else:
            # 否则，打印第三个参数，并用水平打印函数连接 pforma1 和 pforma2
            pforma2 = self._print(e.args[2])
            pforma = self._hprint_vseparator(pforma1, pforma2, ifascii_nougly=False)
            # 格式化 pforma 为漂亮的形式，左侧加上 pforma0
            pforma = prettyForm(*pforma.left('; '))
            pform = prettyForm(*pforma.left(pforma0))
        # 给 pform 加上括号
        pform = prettyForm(*pform.parens())
        # 在 pform 左侧加上 π 符号
        pform = prettyForm(*pform.left(name))
        # 返回格式化后的结果
        return pform

    # 打印黄金比例符号
    def _print_GoldenRatio(self, expr):
        # 如果使用 Unicode，则返回漂亮形式的 φ 符号
        if self._use_unicode:
            return prettyForm(pretty_symbol('phi'))
        # 否则返回 GoldenRatio 符号的打印形式
        return self._print(Symbol("GoldenRatio"))

    # 打印欧拉常数符号
    def _print_EulerGamma(self, expr):
        # 如果使用 Unicode，则返回漂亮形式的 γ 符号
        if self._use_unicode:
            return prettyForm(pretty_symbol('gamma'))
        # 否则返回 EulerGamma 符号的打印形式
        return self._print(Symbol("EulerGamma"))

    # 打印卡塔兰常数符号
    def _print_Catalan(self, expr):
        # 返回卡塔兰常数符号 G 的打印形式
        return self._print(Symbol("G"))

    # 打印取模运算符号
    def _print_Mod(self, expr):
        # 打印第一个参数
        pform = self._print(expr.args[0])
        # 如果 pform 的绑定优先级大于乘法，则给它加上括号
        if pform.binding > prettyForm.MUL:
            pform = prettyForm(*pform.parens())
        # 在 pform 的右侧加上 ' mod ' 字符串
        pform = prettyForm(*pform.right(' mod '))
        # 在 pform 的右侧加上第二个参数的打印形式
        pform = prettyForm(*pform.right(self._print(expr.args[1])))
        # 设置 pform 的绑定为开放
        pform.binding = prettyForm.OPEN
        # 返回格式化后的结果
        return pform
    def _print_Add(self, expr, order=None):
        # 将表达式转换为有序的项列表
        terms = self._as_ordered_terms(expr, order=order)
        # 用于存储每个项的打印形式和索引
        pforms, indices = [], []

        def pretty_negative(pform, index):
            """在漂亮的形式前面添加负号。"""
            # TODO: 将这段代码移到prettyForm中
            # 如果是第一个项，并且高度大于1，则前面加上' - '
            if index == 0:
                if pform.height() > 1:
                    pform_neg = '- '
                else:
                    pform_neg = '-'
            else:
                pform_neg = ' - '

            # 根据绑定和类型创建字符串图片对象
            if (pform.binding > prettyForm.NEG
                or pform.binding == prettyForm.ADD):
                p = stringPict(*pform.parens())
            else:
                p = pform
            # 连接负号和形式对象
            p = stringPict.next(pform_neg, p)
            # 降低绑定到NEG级别，即使它原来更高，以避免打印错误
            return prettyForm(binding=prettyForm.NEG, *p)

        # 遍历所有项并处理
        for i, term in enumerate(terms):
            # 如果是乘法项且可以提取负号
            if term.is_Mul and term.could_extract_minus_sign():
                # 提取系数和其他部分
                coeff, other = term.as_coeff_mul(rational=False)
                if coeff == -1:
                    negterm = Mul(*other, evaluate=False)
                else:
                    negterm = Mul(-coeff, *other, evaluate=False)
                # 获取打印形式并添加负号
                pform = self._print(negterm)
                pforms.append(pretty_negative(pform, i))
            # 如果是有理数且分母大于1
            elif term.is_Rational and term.q > 1:
                pforms.append(None)  # 占位，表示这个位置为空
                indices.append(i)  # 记录这个位置的索引
            # 如果是负数
            elif term.is_Number and term < 0:
                # 获取其正数的打印形式，并添加负号
                pform = self._print(-term)
                pforms.append(pretty_negative(pform, i))
            # 如果是关系运算符
            elif term.is_Relational:
                # 获取打印形式并加上括号
                pforms.append(prettyForm(*self._print(term).parens()))
            else:
                # 其他情况直接获取打印形式
                pforms.append(self._print(term))

        # 如果有需要调整的有理数项
        if indices:
            large = True

            # 判断是否有大型项需要调整
            for pform in pforms:
                if pform is not None and pform.height() > 1:
                    break
            else:
                large = False

            # 对需要调整的项进行处理
            for i in indices:
                term, negative = terms[i], False

                # 如果是负数，转为正数处理
                if term < 0:
                    term, negative = -term, True

                # 根据大小选择打印形式
                if large:
                    pform = prettyForm(str(term.p))/prettyForm(str(term.q))
                else:
                    pform = self._print(term)

                # 如果是负数，则添加负号
                if negative:
                    pform = pretty_negative(pform, i)

                # 更新处理后的打印形式
                pforms[i] = pform

        # 返回所有项组成的总和的打印形式
        return prettyForm.__add__(*pforms)
    # 定义一个方法来打印乘积表达式
    def _print_Mul(self, product):
        # 导入 Quantity 类
        from sympy.physics.units import Quantity

        # 检查是否为未评估的 Mul。在这种情况下，我们需要确保恒等式可见，
        # 多个 Rational 因子不被合并等等，因此显示一个直接的形式，完全保留所有参数及其顺序。
        args = product.args
        if args[0] is S.One or any(isinstance(arg, Number) for arg in args[1:]):
            # 将参数映射为其打印形式
            strargs = list(map(self._print, args))
            # XXX: 这是一个解决方案，用于解决 prettyForm.__mul__ 吸收参数中前导的 -1 的问题。可能更好的解决方法是在 prettyForm.__mul__ 中进行修复。
            negone = strargs[0] == '-1'
            if negone:
                strargs[0] = prettyForm('1', 0, 0)
            # 创建 prettyForm 对象来表示乘积
            obj = prettyForm.__mul__(*strargs)
            if negone:
                obj = prettyForm('-' + obj.s, obj.baseline, obj.binding)
            return obj

        # 初始化空列表来存储分子和分母的项
        a = []  # 分子中的项
        b = []  # 分母中的项（如果有的话）

        # 根据 self.order 对 product 进行排序
        if self.order not in ('old', 'none'):
            args = product.as_ordered_factors()
        else:
            args = list(product.args)

        # 根据是否为 Quantity 或者其基数为 Quantity 的 Pow 对 args 进行排序
        args = sorted(args, key=lambda x: isinstance(x, Quantity) or
                     (isinstance(x, Pow) and isinstance(x.base, Quantity)))

        # 将项分类为分子或分母
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                # 处理形如 x**(-n) 的项
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                # 处理有理数项
                if item.p != 1:
                    a.append( Rational(item.p) )
                if item.q != 1:
                    b.append( Rational(item.q) )
            else:
                # 默认为分子项
                a.append(item)

        # 将每个项转换为其打印形式
        a = [self._print(ai) for ai in a]
        b = [self._print(bi) for bi in b]

        # 构造一个 prettyForm 对象来表示乘积
        if len(b) == 0:
            return prettyForm.__mul__(*a)
        else:
            if len(a) == 0:
                a.append( self._print(S.One) )
            return prettyForm.__mul__(*a)/prettyForm.__mul__(*b)

    # _print_Pow 的辅助函数，用于打印 x**(1/n)
    # 定义一个方法用于打印以给定底数和指数的幂
    def _print_nth_root(self, base, root):
        # 对底数进行漂亮打印
        bpretty = self._print(base)

        # 在非常简单的情况下，使用单字符的根号符号
        if (self._settings['use_unicode_sqrt_char'] and self._use_unicode
            and root == 2 and bpretty.height() == 1
            and (bpretty.width() == 1
                 or (base.is_Integer and base.is_nonnegative))):
            return prettyForm(*bpretty.left(nth_root[2]))

        # 构造根号符号，以 \/ 形状开始
        _zZ = xobj('/', 1)
        rootsign = xobj('\\', 1) + _zZ
        # 构造放在根号上的数字
        rpretty = self._print(root)
        # 如果根号不是单行，则返回幂运算的打印形式
        if rpretty.height() != 1:
            return self._print(base)**self._print(1/root)
        # 如果指数是二分之一，则根号上不应出现数字
        exp = '' if root == 2 else str(rpretty).ljust(2)
        if len(exp) > 2:
            rootsign = ' '*(len(exp) - 2) + rootsign
        # 堆叠指数
        rootsign = stringPict(exp + '\n' + rootsign)
        rootsign.baseline = 0
        # 对角线：长度比底数的高度小一
        linelength = bpretty.height() - 1
        diagonal = stringPict('\n'.join(
            ' '*(linelength - i - 1) + _zZ + ' '*i
            for i in range(linelength)
        ))
        # 将基线放在最低线的下方：与指数相邻
        diagonal.baseline = linelength - 1
        # 创建根号符号
        rootsign = prettyForm(*rootsign.right(diagonal))
        # 设置基线以匹配内容以修正高度，但如果 bpretty 的高度为一，则根号必须再高一点
        rootsign.baseline = max(1, bpretty.baseline)
        # 构建结果
        s = prettyForm(hobj('_', 2 + bpretty.width()))
        s = prettyForm(*bpretty.above(s))
        s = prettyForm(*s.left(rootsign))
        return s

    # 打印指数表达式
    def _print_Pow(self, power):
        # 导入需要的函数
        from sympy.simplify.simplify import fraction
        # 将指数分解为底数和指数部分
        b, e = power.as_base_exp()
        # 如果指数是可交换的
        if power.is_commutative:
            # 如果指数是 -1
            if e is S.NegativeOne:
                return prettyForm("1")/self._print(b)
            # 将指数 e 分解为分数
            n, d = fraction(e)
            # 如果指数是有理数且小于 0，则返回根号的打印形式
            if n is S.One and d.is_Atom and not e.is_Integer and (e.is_Rational or d.is_Symbol) \
                    and self._settings['root_notation']:
                return self._print_nth_root(b, d)
            # 如果指数是负有理数，则返回倒数的打印形式
            if e.is_Rational and e < 0:
                return prettyForm("1")/self._print(Pow(b, -e, evaluate=False))

        # 如果底数是关系式，则返回打印底数的括号形式，并对其做幂运算
        if b.is_Relational:
            return prettyForm(*self._print(b).parens()).__pow__(self._print(e))

        # 返回底数的打印形式的指数运算
        return self._print(b)**self._print(e)

    # 打印未评估的表达式
    def _print_UnevaluatedExpr(self, expr):
        # 返回未评估表达式中的第一个参数的打印形式
        return self._print(expr.args[0])
    # 打印有理数表达式的分子和分母，根据情况返回格式化的 prettyForm 对象
    def __print_numer_denom(self, p, q):
        if q == 1:
            # 如果分母为1，根据分子正负返回格式化的 prettyForm 对象
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)
            else:
                return prettyForm(str(p))
        elif abs(p) >= 10 and abs(q) >= 10:
            # 如果分子和分母的绝对值都大于等于10，打印较大的分数形式
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)/prettyForm(str(q))
                # 旧的打印方法:
                # pform = prettyForm(str(-p))/prettyForm(str(q))
                # return prettyForm(binding=prettyForm.NEG, *pform.left('- '))
            else:
                return prettyForm(str(p))/prettyForm(str(q))
        else:
            # 其他情况返回空值
            return None

    # 打印有理数表达式的格式化输出，如果无法处理则返回空打印
    def _print_Rational(self, expr):
        result = self.__print_numer_denom(expr.p, expr.q)

        if result is not None:
            return result
        else:
            return self.emptyPrinter(expr)

    # 打印分数表达式的格式化输出，如果无法处理则返回空打印
    def _print_Fraction(self, expr):
        result = self.__print_numer_denom(expr.numerator, expr.denominator)

        if result is not None:
            return result
        else:
            return self.emptyPrinter(expr)

    # 打印乘积集合的格式化输出
    def _print_ProductSet(self, p):
        if len(p.sets) >= 1 and not has_variety(p.sets):
            # 如果集合数量大于等于1且所有集合没有多样性，则打印形如 A^B 的格式
            return self._print(p.sets[0]) ** self._print(len(p.sets))
        else:
            # 否则打印乘法符号分隔的集合
            prod_char = pretty_atom('Multiplication') if self._use_unicode else 'x'
            return self._print_seq(p.sets, None, None, ' %s ' % prod_char,
                                   parenthesize=lambda set: set.is_Union or
                                   set.is_Intersection or set.is_ProductSet)

    # 打印有限集合的格式化输出
    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_seq(items, '{', '}', ', ' )

    # 打印范围集合的格式化输出
    def _print_Range(self, s):
        if self._use_unicode:
            dots = pretty_atom('Dots')
        else:
            dots = '...'

        if s.start.is_infinite and s.stop.is_infinite:
            # 如果起始和结束都是无穷，根据步长打印相应形式
            if s.step.is_positive:
                printset = dots, -1, 0, 1, dots
            else:
                printset = dots, 1, 0, -1, dots
        elif s.start.is_infinite:
            # 如果起始是无穷，打印从最后元素到步长的形式
            printset = dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite:
            # 如果结束是无穷，打印从起始到步长的形式
            it = iter(s)
            printset = next(it), next(it), dots
        elif len(s) > 4:
            # 如果集合长度大于4，打印部分元素和省略号的形式
            it = iter(s)
            printset = next(it), next(it), dots, s[-1]
        else:
            # 其他情况直接打印集合元素
            printset = tuple(s)

        return self._print_seq(printset, '{', '}', ', ' )

    # 打印区间的格式化输出
    def _print_Interval(self, i):
        if i.start == i.end:
            # 如果区间起始等于结束，打印单个元素的形式
            return self._print_seq(i.args[:1], '{', '}')

        else:
            # 否则根据左右开闭状态打印相应形式
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return self._print_seq(i.args[:2], left, right)
    # 打印累积边界符号 '<' 和 '>'
    def _print_AccumulationBounds(self, i):
        left = '<'
        right = '>'
        
        # 调用 self._print_seq 方法打印 i.args 的前两个元素，使用 '<' 和 '>' 作为左右边界符
        return self._print_seq(i.args[:2], left, right)

    # 打印 Intersection 对象
    def _print_Intersection(self, u):
        # 设置 Intersection 对象的分隔符为 ' Intersection(n) '，其中 n 可能是 pretty_atom 函数的结果
        delimiter = ' %s ' % pretty_atom('Intersection', 'n')
        
        # 调用 self._print_seq 方法打印 u.args，不使用额外的左右边界符，使用 delimiter 作为分隔符
        return self._print_seq(u.args, None, None, delimiter,
                               # 根据条件使用括号将 set 参数进行括起来
                               parenthesize=lambda set: set.is_ProductSet or
                               set.is_Union or set.is_Complement)

    # 打印 Union 对象
    def _print_Union(self, u):
        # 设置 Union 对象的分隔符为 ' Union(U) '，其中 U 可能是 pretty_atom 函数的结果
        union_delimiter = ' %s ' % pretty_atom('Union', 'U')
        
        # 调用 self._print_seq 方法打印 u.args，不使用额外的左右边界符，使用 union_delimiter 作为分隔符
        return self._print_seq(u.args, None, None, union_delimiter,
                               # 根据条件使用括号将 set 参数进行括起来
                               parenthesize=lambda set: set.is_ProductSet or
                               set.is_Intersection or set.is_Complement)

    # 打印 SymmetricDifference 对象
    def _print_SymmetricDifference(self, u):
        # 如果不使用 Unicode，则抛出未实现的错误信息
        if not self._use_unicode:
            raise NotImplementedError("ASCII pretty printing of SymmetricDifference is not implemented")
        
        # 设置 SymmetricDifference 对象的分隔符为 ' SymmetricDifference '
        sym_delimeter = ' %s ' % pretty_atom('SymmetricDifference')
        
        # 调用 self._print_seq 方法打印 u.args，不使用额外的左右边界符，使用 sym_delimeter 作为分隔符
        return self._print_seq(u.args, None, None, sym_delimeter)

    # 打印 Complement 对象
    def _print_Complement(self, u):
        # 设置 Complement 对象的分隔符为 ' \ '
        delimiter = r' \ '
        
        # 调用 self._print_seq 方法打印 u.args，不使用额外的左右边界符，使用 delimiter 作为分隔符
        return self._print_seq(u.args, None, None, delimiter,
             # 根据条件使用括号将 set 参数进行括起来
             parenthesize=lambda set: set.is_ProductSet or set.is_Intersection
                               or set.is_Union)

    # 打印 ImageSet 对象
    def _print_ImageSet(self, ts):
        # 如果使用 Unicode，则设置 inn 为 "SmallElementOf"，否则设置为 'in'
        if self._use_unicode:
            inn = pretty_atom("SmallElementOf")
        else:
            inn = 'in'
        
        # 获取 ImageSet 对象的 lambda 表达式 fun、基础集合 sets、签名 signature 和表达式 expr
        fun = ts.lamda
        sets = ts.base_sets
        signature = fun.signature
        expr = self._print(fun.expr)

        # TODO: 略过注释
        if len(signature) == 1:
            # 如果签名长度为1，将 signature[0]、inn 和 sets[0] 组成序列 S
            S = self._print_seq((signature[0], inn, sets[0]),
                                delimiter=' ')
            # 调用 self._hprint_vseparator 方法，以 expr 和 S 为参数，打印左右括号为 '{' 和 '}'，使用空格作为分隔符
            return self._hprint_vseparator(expr, S,
                                           left='{', right='}',
                                           ifascii_nougly=True, delimiter=' ')
        else:
            # 如果签名长度不为1，构建包含变量和集合的参数 pargs
            pargs = tuple(j for var, setv in zip(signature, sets) for j in
                          (var, ' ', inn, ' ', setv, ", "))
            # 调用 self._print_seq 方法打印参数 pargs，不使用额外的左右边界符
            S = self._print_seq(pargs[:-1], delimiter='')
            # 调用 self._hprint_vseparator 方法，以 expr 和 S 为参数，打印左右括号为 '{' 和 '}'，使用空格作为分隔符
            return self._hprint_vseparator(expr, S,
                                           left='{', right='}',
                                           ifascii_nougly=True, delimiter=' ')
    def _print_ConditionSet(self, ts):
        # 根据 self._use_unicode 的值选择合适的字符串表示
        if self._use_unicode:
            # 将 'SmallElementOf' 转换为美观的表示形式
            inn = pretty_atom('SmallElementOf')
            # 使用 pretty_atom 函数来避免覆盖 Python 关键字 'and'
            _and = pretty_atom('And')
        else:
            # 当不使用 Unicode 时，直接使用原始字符串
            inn = 'in'
            _and = 'and'

        # 打印序列中的符号变量
        variables = self._print_seq(Tuple(ts.sym))
        # 获取条件表达式（如果可用），否则打印条件本身
        as_expr = getattr(ts.condition, 'as_expr', None)
        if as_expr is not None:
            cond = self._print(ts.condition.as_expr())
        else:
            cond = self._print(ts.condition)
            if self._use_unicode:
                # 为 Unicode 情况下打印条件添加额外的美化处理
                cond = self._print(cond)
                cond = prettyForm(*cond.parens())

        # 如果 base_set 是全集 UniversalSet，则返回特定格式的字符串
        if ts.base_set is S.UniversalSet:
            return self._hprint_vseparator(variables, cond, left="{",
                                           right="}", ifascii_nougly=True,
                                           delimiter=' ')

        # 否则打印变量、条件、基础集合和 'and' 连接符
        base = self._print(ts.base_set)
        C = self._print_seq((variables, inn, base, _and, cond),
                            delimiter=' ')
        return self._hprint_vseparator(variables, C, left="{", right="}",
                                       ifascii_nougly=True, delimiter=' ')

    def _print_ComplexRegion(self, ts):
        # 根据 self._use_unicode 的值选择合适的字符串表示
        if self._use_unicode:
            inn = pretty_atom('SmallElementOf')
        else:
            inn = 'in'

        # 打印序列中的变量
        variables = self._print_seq(ts.variables)
        # 打印表达式和集合
        expr = self._print(ts.expr)
        prodsets = self._print(ts.sets)

        # 打印变量、in/not in、产品集
        C = self._print_seq((variables, inn, prodsets),
                            delimiter=' ')
        return self._hprint_vseparator(expr, C, left="{", right="}",
                                       ifascii_nougly=True, delimiter=' ')

    def _print_Contains(self, e):
        # 获取元素和集合参数
        var, set = e.args
        if self._use_unicode:
            # 使用 pretty_atom 函数美化 ElementOf 表示
            el = f" {pretty_atom('ElementOf')} "
            # 返回美化后的打印结果
            return prettyForm(*stringPict.next(self._print(var),
                                               el, self._print(set)), binding=8)
        else:
            # 返回原始的打印结果字符串
            return prettyForm(sstr(e))

    def _print_FourierSeries(self, s):
        # 如果 Fourier 级数的一阶和二阶项均为零，则只打印 a0
        if s.an.formula is S.Zero and s.bn.formula is S.Zero:
            return self._print(s.a0)
        if self._use_unicode:
            # 美化表示 dots
            dots = pretty_atom('Dots')
        else:
            dots = '...'
        # 返回截断的 Fourier 级数和 dots 的打印结果
        return self._print_Add(s.truncate()) + self._print(dots)

    def _print_FormalPowerSeries(self, s):
        # 打印无限级数
        return self._print_Add(s.infinite)

    def _print_SetExpr(self, se):
        # 打印 SetExpr 对象，格式化输出
        pretty_set = prettyForm(*self._print(se.set).parens())
        pretty_name = self._print(Symbol("SetExpr"))
        return prettyForm(*pretty_name.right(pretty_set))
    # 打印序列的特定格式表达式
    def _print_SeqFormula(self, s):
        # 根据 self._use_unicode 决定使用不同的 dots 符号
        if self._use_unicode:
            dots = pretty_atom('Dots')
        else:
            dots = '...'

        # 检查序列起始和结束是否有符号变量，如果有则抛出未实现异常
        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            raise NotImplementedError("Pretty printing of sequences with symbolic bound not implemented")

        # 根据序列的起始值选择打印的元素集合
        if s.start is S.NegativeInfinity:
            stop = s.stop
            # 根据序列的长度选择打印的元素集合
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2),
                        s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            # 如果序列长度大于4或结束为无穷，则选择部分元素加省略号
            printset = s[:4]
            printset.append(dots)
            printset = tuple(printset)
        else:
            # 否则选择全部序列元素打印
            printset = tuple(s)
        return self._print_list(printset)

    # 下面三个函数功能相同，都调用了 _print_SeqFormula 函数来打印序列表达式
    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    # 打印序列或者集合
    def _print_seq(self, seq, left=None, right=None, delimiter=', ',
            parenthesize=lambda x: False, ifascii_nougly=True):
        pforms = []
        # 遍历序列中的每个元素，并进行打印
        for item in seq:
            pform = self._print(item)
            # 如果需要对元素进行括号化，则进行处理
            if parenthesize(item):
                pform = prettyForm(*pform.parens())
            if pforms:
                pforms.append(delimiter)
            pforms.append(pform)

        # 如果序列为空，则返回一个空字符串
        if not pforms:
            s = stringPict('')
        else:
            # 否则将所有打印的元素拼接成一个 prettyForm 对象
            s = prettyForm(*stringPict.next(*pforms))

        # 对打印结果进行括号化处理
        s = prettyForm(*s.parens(left, right, ifascii_nougly=ifascii_nougly))
        return s

    # 将一系列参数用指定的分隔符连接起来打印
    def join(self, delimiter, args):
        pform = None
        # 遍历参数列表，并逐个连接打印
        for arg in args:
            if pform is None:
                pform = arg
            else:
                pform = prettyForm(*pform.right(delimiter))
                pform = prettyForm(*pform.right(arg))

        # 如果没有参数，则返回空的 prettyForm 对象
        if pform is None:
            return prettyForm("")
        else:
            return pform

    # 打印列表
    def _print_list(self, l):
        return self._print_seq(l, '[', ']')

    # 打印元组
    def _print_tuple(self, t):
        # 如果元组只有一个元素，则打印时添加逗号
        if len(t) == 1:
            ptuple = prettyForm(*stringPict.next(self._print(t[0]), ','))
            return prettyForm(*ptuple.parens('(', ')', ifascii_nougly=True))
        else:
            # 否则按照一般的元组打印格式处理
            return self._print_seq(t, '(', ')')

    # 打印字典
    def _print_dict(self, d):
        # 对字典的键按照默认排序进行排序
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        # 遍历排序后的键值对，并打印成特定格式
        for k in keys:
            K = self._print(k)
            V = self._print(d[k])
            s = prettyForm(*stringPict.next(K, ': ', V))

            items.append(s)

        # 将打印结果按照字典格式输出
        return self._print_seq(items, '{', '}')

    # 打印大写字典
    def _print_Dict(self, d):
        return self._print_dict(d)

    # 打印集合
    def _print_set(self, s):
        # 如果集合为空，则打印空集合
        if not s:
            return prettyForm('set()')
        # 否则对集合元素按照默认排序进行排序
        items = sorted(s, key=default_sort_key)
        # 打印排序后的集合元素
        pretty = self._print_seq(items)
        # 将打印结果按照集合格式输出
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        return pretty
    # 打印 frozenset 类型的对象
    def _print_frozenset(self, s):
        # 如果 s 是空集合，则返回漂亮格式化的 frozenset()
        if not s:
            return prettyForm('frozenset()')
        # 对集合 s 进行排序，并使用默认排序键进行排序
        items = sorted(s, key=default_sort_key)
        # 调用 self._print_seq 方法将排序后的集合项转换为漂亮格式化的序列形式
        pretty = self._print_seq(items)
        # 将格式化后的序列用大括号括起来，并生成漂亮格式
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        # 将格式化后的序列用圆括号括起来，并生成漂亮格式
        pretty = prettyForm(*pretty.parens('(', ')', ifascii_nougly=True))
        # 将对象类型的名称和漂亮格式化的内容合并成最终的漂亮格式
        pretty = prettyForm(*stringPict.next(type(s).__name__, pretty))
        # 返回最终生成的漂亮格式
        return pretty

    # 打印 UniversalSet 类型的对象
    def _print_UniversalSet(self, s):
        # 如果设置了使用 Unicode，则返回漂亮格式化的 'Universe'
        if self._use_unicode:
            return prettyForm(pretty_atom('Universe'))
        # 否则，返回漂亮格式化的 'UniversalSet'
        else:
            return prettyForm('UniversalSet')

    # 打印 PolyRing 类型的对象
    def _print_PolyRing(self, ring):
        # 返回 ring 对象的字符串形式的漂亮格式化
        return prettyForm(sstr(ring))

    # 打印 FracField 类型的对象
    def _print_FracField(self, field):
        # 返回 field 对象的字符串形式的漂亮格式化
        return prettyForm(sstr(field))

    # 打印 FreeGroupElement 类型的对象
    def _print_FreeGroupElement(self, elm):
        # 返回 FreeGroupElement 对象的字符串形式的漂亮格式化
        return prettyForm(str(elm))

    # 打印 PolyElement 类型的对象
    def _print_PolyElement(self, poly):
        # 返回 poly 对象的字符串形式的漂亮格式化
        return prettyForm(sstr(poly))

    # 打印 FracElement 类型的对象
    def _print_FracElement(self, frac):
        # 返回 frac 对象的字符串形式的漂亮格式化
        return prettyForm(sstr(frac))

    # 打印 AlgebraicNumber 类型的对象
    def _print_AlgebraicNumber(self, expr):
        # 如果 AlgebraicNumber 对象被别名化，则打印其多项式表达式的漂亮格式
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        # 否则，打印其表达式的漂亮格式
        else:
            return self._print(expr.as_expr())

    # 打印 ComplexRootOf 类型的对象
    def _print_ComplexRootOf(self, expr):
        # 将表达式 expr.expr 和指数 expr.index 以 lex 顺序打印，用括号括起来
        args = [self._print_Add(expr.expr, order='lex'), expr.index]
        pform = prettyForm(*self._print_seq(args).parens())
        # 在左侧添加 'CRootOf' 标签，并生成漂亮格式
        pform = prettyForm(*pform.left('CRootOf'))
        # 返回最终生成的漂亮格式
        return pform

    # 打印 RootSum 类型的对象
    def _print_RootSum(self, expr):
        # 将表达式 expr.expr 以 lex 顺序打印，用括号括起来
        args = [self._print_Add(expr.expr, order='lex')]

        # 如果 expr.fun 不是恒等函数，则打印其函数表达式的漂亮格式
        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        # 将所有参数用括号括起来，并生成漂亮格式
        pform = prettyForm(*self._print_seq(args).parens())
        # 在左侧添加 'RootSum' 标签，并生成漂亮格式
        pform = prettyForm(*pform.left('RootSum'))

        # 返回最终生成的漂亮格式
        return pform

    # 打印 FiniteField 类型的对象
    def _print_FiniteField(self, expr):
        # 根据设置的使用 Unicode，选择格式化字符串的形式
        if self._use_unicode:
            form = f"{pretty_atom('Integers')}_%d"
        else:
            form = 'GF(%d)'

        # 返回按照格式化字符串 form 格式化后的漂亮格式
        return prettyForm(pretty_symbol(form % expr.mod))

    # 打印 IntegerRing 类型的对象
    def _print_IntegerRing(self, expr):
        # 根据设置的使用 Unicode，选择漂亮格式化的形式
        if self._use_unicode:
            return prettyForm(pretty_atom('Integers'))
        else:
            return prettyForm('ZZ')

    # 打印 RationalField 类型的对象
    def _print_RationalField(self, expr):
        # 根据设置的使用 Unicode，选择漂亮格式化的形式
        if self._use_unicode:
            return prettyForm(pretty_atom('Rationals'))
        else:
            return prettyForm('QQ')

    # 打印 RealField 类型的对象
    def _print_RealField(self, domain):
        # 根据设置的使用 Unicode，选择前缀的漂亮格式化形式
        if self._use_unicode:
            prefix = pretty_atom("Reals")
        else:
            prefix = 'RR'

        # 如果 domain 具有默认精度，则返回前缀的漂亮格式化
        if domain.has_default_precision:
            return prettyForm(prefix)
        # 否则，将精度信息附加到前缀后，并返回格式化的漂亮格式
        else:
            return self._print(pretty_symbol(prefix + "_" + str(domain.precision)))
    # 打印复数域表达式的格式化输出
    def _print_ComplexField(self, domain):
        # 根据 self._use_unicode 决定前缀
        if self._use_unicode:
            prefix = pretty_atom('Complexes')
        else:
            prefix = 'CC'

        # 如果 domain 有默认精度，则返回前缀的美化形式
        if domain.has_default_precision:
            return prettyForm(prefix)
        else:
            # 否则返回前缀加上精度的符号的美化输出
            return self._print(pretty_symbol(prefix + "_" + str(domain.precision)))

    # 打印多项式环表达式的格式化输出
    def _print_PolynomialRing(self, expr):
        args = list(expr.symbols)

        # 如果表达式的顺序不是默认的，则添加顺序的美化形式
        if not expr.order.is_default:
            order = prettyForm(*prettyForm("order=").right(self._print(expr.order)))
            args.append(order)

        # 组成参数列表的美化输出，左侧添加域的美化输出
        pform = self._print_seq(args, '[', ']')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    # 打印分式域表达式的格式化输出
    def _print_FractionField(self, expr):
        args = list(expr.symbols)

        # 如果表达式的顺序不是默认的，则添加顺序的美化形式
        if not expr.order.is_default:
            order = prettyForm(*prettyForm("order=").right(self._print(expr.order)))
            args.append(order)

        # 组成参数列表的美化输出，左侧添加域的美化输出
        pform = self._print_seq(args, '(', ')')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    # 打印多项式环基类表达式的格式化输出
    def _print_PolynomialRingBase(self, expr):
        g = expr.symbols
        # 如果表达式的顺序不是默认的，则添加顺序的美化形式
        if str(expr.order) != str(expr.default_order):
            g = g + ("order=" + str(expr.order),)
        # 组成参数列表的美化输出，左侧添加域的美化输出
        pform = self._print_seq(g, '[', ']')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    # 打印格罗布纳基表达式的格式化输出
    def _print_GroebnerBasis(self, basis):
        # 对于基的每个表达式，调用 self._print_Add 方法，并按顺序美化输出
        exprs = [ self._print_Add(arg, order=basis.order)
                  for arg in basis.exprs ]
        exprs = prettyForm(*self.join(", ", exprs).parens(left="[", right="]"))

        # 对于基的每个生成元，调用 self._print 方法，然后按顺序美化输出
        gens = [ self._print(gen) for gen in basis.gens ]

        # 创建域和顺序的美化输出
        domain = prettyForm(
            *prettyForm("domain=").right(self._print(basis.domain)))
        order = prettyForm(
            *prettyForm("order=").right(self._print(basis.order)))

        # 组成所有输出的美化形式，左侧添加基类的名称
        pform = self.join(", ", [exprs] + gens + [domain, order])
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left(basis.__class__.__name__))

        return pform

    # 打印替换表达式的格式化输出
    def _print_Subs(self, e):
        # 打印表达式 e.expr 的格式化输出，加上括号
        pform = self._print(e.expr)
        pform = prettyForm(*pform.parens())

        # 计算 pform 的高度，如果高度大于 1 则为其高度，否则为 2
        h = pform.height() if pform.height() > 1 else 2
        # 创建垂直线对象，并与 pform 对齐
        rvert = stringPict(vobj('|', h), baseline=pform.baseline)
        pform = prettyForm(*pform.right(rvert))

        # 设置基线为 b，计算新的替换表达式的美化输出
        b = pform.baseline
        pform.baseline = pform.height() - 1
        # 为每个变量 v[0] 和其点 v[1] 创建格式化输出，按顺序连接
        pform = prettyForm(*pform.right(self._print_seq([
            self._print_seq((self._print(v[0]), xsym('=='), self._print(v[1])),
                delimiter='') for v in zip(e.variables, e.point) ])))

        # 恢复基线为 b
        pform.baseline = b
        return pform
    # 打印数学函数的表示形式，包括函数名称和参数
    def _print_number_function(self, e, name):
        # 创建一个表示形式对象，用于存储函数的打印格式
        pform = prettyForm(name)
        # 打印函数的第一个参数
        arg = self._print(e.args[0])
        # 创建一个与参数宽度相匹配的空白行对象
        pform_arg = prettyForm(" "*arg.width())
        # 将参数放在打印格式对象下方
        pform_arg = prettyForm(*pform_arg.below(arg))
        # 将参数形式化对象放在函数形式对象的右侧
        pform = prettyForm(*pform.right(pform_arg))
        # 如果函数有多个参数
        if len(e.args) == 1:
            return pform
        # 如果有两个参数，m 和 x
        m, x = e.args
        # TODO: 从 _print_Function 复制粘贴：是否有更好的方法？
        # 创建函数形式对象
        prettyFunc = pform
        # 将参数序列打印为括号形式对象
        prettyArgs = prettyForm(*self._print_seq([x]).parens())
        # 创建一个新的打印格式对象，包含函数和参数
        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))
        # 将函数形式对象和参数形式对象存储在打印格式对象中
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs
        # 返回打印格式对象
        return pform

    # 打印 Euler 函数的表示形式
    def _print_euler(self, e):
        return self._print_number_function(e, "E")

    # 打印 Catalan 函数的表示形式
    def _print_catalan(self, e):
        return self._print_number_function(e, "C")

    # 打印 Bernoulli 函数的表示形式
    def _print_bernoulli(self, e):
        return self._print_number_function(e, "B")

    # 将 _print_bell 与 _print_bernoulli 相同
    _print_bell = _print_bernoulli

    # 打印 Lucas 函数的表示形式
    def _print_lucas(self, e):
        return self._print_number_function(e, "L")

    # 打印 Fibonacci 函数的表示形式
    def _print_fibonacci(self, e):
        return self._print_number_function(e, "F")

    # 打印 Tribonacci 函数的表示形式
    def _print_tribonacci(self, e):
        return self._print_number_function(e, "T")

    # 打印 Stieltjes 函数的表示形式
    def _print_stieltjes(self, e):
        # 如果使用 Unicode，返回带有希腊符号的 Stieltjes 函数的表示形式
        if self._use_unicode:
            return self._print_number_function(e, greek_unicode['gamma'])
        # 否则，返回简单的 Stieltjes 函数的表示形式
        else:
            return self._print_number_function(e, "stieltjes")

    # 打印 KroneckerDelta 函数的表示形式
    def _print_KroneckerDelta(self, e):
        # 打印第一个参数
        pform = self._print(e.args[0])
        # 将逗号形式化对象放在第一个参数右侧
        pform = prettyForm(*pform.right(prettyForm(',')))
        # 打印第二个参数
        pform = prettyForm(*pform.right(self._print(e.args[1])))
        # 如果使用 Unicode，创建表示 Delta 的形式化对象
        if self._use_unicode:
            a = stringPict(pretty_symbol('delta'))
        # 否则，创建简单的 'd' 形式化对象
        else:
            a = stringPict('d')
        # 创建包含 Delta 和参数的顶部和底部形式化对象
        b = pform
        top = stringPict(*b.left(' '*a.width()))
        bot = stringPict(*a.right(' '*b.width()))
        # 返回包含顶部和底部的幂形式化对象
        return prettyForm(binding=prettyForm.POW, *bot.below(top))

    # 打印 RandomDomain 类型的表示形式
    def _print_RandomDomain(self, d):
        # 如果对象具有 'as_boolean' 属性
        if hasattr(d, 'as_boolean'):
            # 打印域的标签
            pform = self._print('Domain: ')
            # 将布尔表达式打印为形式化对象
            pform = prettyForm(*pform.right(self._print(d.as_boolean())))
            return pform
        # 如果对象具有 'set' 属性
        elif hasattr(d, 'set'):
            # 打印域的标签
            pform = self._print('Domain: ')
            # 打印符号的集合
            pform = prettyForm(*pform.right(self._print(d.symbols)))
            # 打印 'in' 关键字
            pform = prettyForm(*pform.right(self._print(' in ')))
            # 打印集合
            pform = prettyForm(*pform.right(self._print(d.set)))
            return pform
        # 如果对象具有 'symbols' 属性
        elif hasattr(d, 'symbols'):
            # 打印域的标签
            pform = self._print('Domain on ')
            # 打印符号
            pform = prettyForm(*pform.right(self._print(d.symbols)))
            return pform
        # 否则，打印 None 对象
        else:
            return self._print(None)
    def _print_DMP(self, p):
        try:
            # 检查参数 p 的 ring 属性是否存在
            if p.ring is not None:
                # TODO incorporate order（待办事项：整合顺序）
                # 将 p.ring 转换为 SymPy 对象，并打印其结果
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        # 如果无法转换或出现异常，则返回 p 的字符串表示形式
        return self._print(repr(p))

    def _print_DMF(self, p):
        # 直接调用 _print_DMP 方法来打印 p
        return self._print_DMP(p)

    def _print_Object(self, object):
        # 打印 object 的名称（通过 pretty_symbol 函数美化输出）
        return self._print(pretty_symbol(object.name))

    def _print_Morphism(self, morphism):
        # 定义箭头符号
        arrow = xsym("-->")

        # 打印 morphism 的 domain 和 codomain，以箭头符号连接起来
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        tail = domain.right(arrow, codomain)[0]

        # 返回美化后的形式
        return prettyForm(tail)

    def _print_NamedMorphism(self, morphism):
        # 打印 morphism 的名称和其对应的箭头表达式
        pretty_name = self._print(pretty_symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return prettyForm(pretty_name.right(":", pretty_morphism)[0])

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        # 创建一个 NamedMorphism 对象，并打印其形式
        return self._print_NamedMorphism(
            NamedMorphism(morphism.domain, morphism.codomain, "id"))

    def _print_CompositeMorphism(self, morphism):
        # 定义圆圈符号
        circle = xsym(".")

        # 获取所有组成 morphism 的组件的名称，并反转列表顺序
        component_names_list = [pretty_symbol(component.name) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = circle.join(component_names_list) + ":"

        # 打印构建的复合名称和 morphism 的形式
        pretty_name = self._print(component_names)
        pretty_morphism = self._print_Morphism(morphism)
        return prettyForm(pretty_name.right(pretty_morphism)[0])

    def _print_Category(self, category):
        # 打印 category 的名称
        return self._print(pretty_symbol(category.name))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # 如果 diagram 的 premises 为空，返回一个空集合的打印形式
            return self._print(S.EmptySet)

        # 打印 diagram 的 premises
        pretty_result = self._print(diagram.premises)
        if diagram.conclusions:
            # 如果 diagram 有 conclusions，则打印箭头和 conclusions
            results_arrow = " %s " % xsym("==>")

            pretty_conclusions = self._print(diagram.conclusions)[0]
            pretty_result = pretty_result.right(
                results_arrow, pretty_conclusions)

        # 返回美化后的形式
        return prettyForm(pretty_result[0])

    def _print_DiagramGrid(self, grid):
        from sympy.matrices import Matrix
        # 根据 grid 创建一个 Matrix 对象
        matrix = Matrix([[grid[i, j] if grid[i, j] else Symbol(" ")
                          for j in range(grid.width)]
                         for i in range(grid.height)])
        # 打印 matrix 的内容
        return self._print_matrix_contents(matrix)

    def _print_FreeModuleElement(self, m):
        # 方便起见，将 m 打印为行向量的序列
        return self._print_seq(m, '[', ']')

    def _print_SubModule(self, M):
        # 将 M 的生成元转换为 SymPy 对象，并打印其序列
        gens = [[M.ring.to_sympy(g) for g in gen] for gen in M.gens]
        return self._print_seq(gens, '<', '>')
    # 打印 FreeModule 类的字符串表示，包括环和秩
    def _print_FreeModule(self, M):
        return self._print(M.ring)**self._print(M.rank)

    # 打印 ModuleImplementedIdeal 类的字符串表示，包括环的符号表示和生成元的列表
    def _print_ModuleImplementedIdeal(self, M):
        sym = M.ring.to_sympy
        return self._print_seq([sym(x) for [x] in M._module.gens], '<', '>')

    # 打印 QuotientRing 类的字符串表示，包括环和基理想的字符串表示
    def _print_QuotientRing(self, R):
        return self._print(R.ring) / self._print(R.base_ideal)

    # 打印 QuotientRingElement 类的字符串表示，包括环元素转换为 sympy 的字符串表示和基理想的字符串表示
    def _print_QuotientRingElement(self, R):
        return self._print(R.ring.to_sympy(R)) + self._print(R.ring.base_ideal)

    # 打印 QuotientModuleElement 类的字符串表示，包括模数据的字符串表示和被杀死的模的字符串表示
    def _print_QuotientModuleElement(self, m):
        return self._print(m.data) + self._print(m.module.killed_module)

    # 打印 QuotientModule 类的字符串表示，包括基模和被杀死的模的字符串表示
    def _print_QuotientModule(self, M):
        return self._print(M.base) / self._print(M.killed_module)

    # 打印 MatrixHomomorphism 类的字符串表示，包括矩阵的字符串表示和映射域和值域的字符串表示
    def _print_MatrixHomomorphism(self, h):
        matrix = self._print(h._sympy_matrix())
        matrix.baseline = matrix.height() // 2
        pform = prettyForm(*matrix.right(' : ', self._print(h.domain),
            ' %s> ' % hobj('-', 2), self._print(h.codomain)))
        return pform

    # 打印 Manifold 类的字符串表示，包括流形的名称
    def _print_Manifold(self, manifold):
        return self._print(manifold.name)

    # 打印 Patch 类的字符串表示，包括补丁的名称
    def _print_Patch(self, patch):
        return self._print(patch.name)

    # 打印 CoordSystem 类的字符串表示，包括坐标系的名称
    def _print_CoordSystem(self, coords):
        return self._print(coords.name)

    # 打印 BaseScalarField 类的字符串表示，包括基标量场的符号名
    def _print_BaseScalarField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return self._print(pretty_symbol(string))

    # 打印 BaseVectorField 类的字符串表示，包括基矢量场的符号名
    def _print_BaseVectorField(self, field):
        s = U('PARTIAL DIFFERENTIAL') + '_' + field._coord_sys.symbols[field._index].name
        return self._print(pretty_symbol(s))

    # 打印 Differential 类的字符串表示，包括微分算子和相关的场的字符串表示
    def _print_Differential(self, diff):
        if self._use_unicode:
            d = pretty_atom('Differential')
        else:
            d = 'd'
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys.symbols[field._index].name
            return self._print(d + ' ' + pretty_symbol(string))
        else:
            pform = self._print(field)
            pform = prettyForm(*pform.parens())
            return prettyForm(*pform.left(d))

    # 打印 Tr 类的字符串表示，目前未处理指标
    def _print_Tr(self, p):
        #TODO: Handle indices
        pform = self._print(p.args[0])
        pform = prettyForm(*pform.left('%s(' % (p.__class__.__name__)))
        pform = prettyForm(*pform.right(')'))
        return pform

    # 打印 primenu 函数的字符串表示，包括 nu 符号和函数参数的字符串表示
    def _print_primenu(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        if self._use_unicode:
            pform = prettyForm(*pform.left(greek_unicode['nu']))
        else:
            pform = prettyForm(*pform.left('nu'))
        return pform

    # 打印 primeomega 函数的字符串表示，包括 Omega 符号和函数参数的字符串表示
    def _print_primeomega(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        if self._use_unicode:
            pform = prettyForm(*pform.left(greek_unicode['Omega']))
        else:
            pform = prettyForm(*pform.left('Omega'))
        return pform
    # 打印"Quantity"对象的表达式，根据需要进行格式化输出
    def _print_Quantity(self, e):
        # 如果表达式是"degree"，根据使用unicode标志确定输出形式
        if e.name.name == 'degree':
            if self._use_unicode:
                # 使用漂亮的原子"Degree"来打印
                pform = self._print(pretty_atom('Degree'))
            else:
                # 否则使用ASCII码176来打印度数符号
                pform = self._print(chr(176))
            return pform
        else:
            # 对于其他情况调用空打印机处理
            return self.emptyPrinter(e)
    
    # 打印赋值操作的基类表达式，构建并返回美观的格式化形式
    def _print_AssignmentBase(self, e):
        # 获取操作符的漂亮形式
        op = prettyForm(' ' + xsym(e.op) + ' ')
        # 打印左操作数和右操作数的格式化形式
        l = self._print(e.lhs)
        r = self._print(e.rhs)
        # 创建包含左操作数、操作符和右操作数的漂亮形式对象
        pform = prettyForm(*stringPict.next(l, op, r))
        return pform
    
    # 打印字符串对象的表达式，返回其名称的打印形式
    def _print_Str(self, s):
        return self._print(s.name)
`
# 定义一个装饰器函数，用于打印漂亮的表达式
@print_function(PrettyPrinter)
def pretty(expr, **settings):
    """Returns a string containing the prettified form of expr.

    For information on keyword arguments see pretty_print function.

    """
    # 创建一个 PrettyPrinter 对象，使用传入的设置参数
    pp = PrettyPrinter(settings)

    # XXX: 这是一个丑陋的 hack，但至少能够工作
    # 获取是否使用 Unicode 的设置
    use_unicode = pp._settings['use_unicode']
    # 调用函数设置是否使用 Unicode，并记录原始设置
    uflag = pretty_use_unicode(use_unicode)

    try:
        # 调用 PrettyPrinter 对象的 doprint 方法对表达式进行美化打印
        return pp.doprint(expr)
    finally:
        # 恢复 Unicode 使用设置为原始值
        pretty_use_unicode(uflag)


def pretty_print(expr, **kwargs):
    """Prints expr in pretty form.

    pprint is just a shortcut for this function.

    Parameters
    ==========

    expr : expression
        The expression to print.

    wrap_line : bool, optional (default=True)
        Line wrapping enabled/disabled.

    num_columns : int or None, optional (default=None)
        Number of columns before line breaking (default to None which reads
        the terminal width), useful when using SymPy without terminal.

    use_unicode : bool or None, optional (default=None)
        Use unicode characters, such as the Greek letter pi instead of
        the string pi.

    full_prec : bool or string, optional (default="auto")
        Use full precision.

    order : bool or string, optional (default=None)
        Set to 'none' for long expressions if slow; default is None.

    use_unicode_sqrt_char : bool, optional (default=True)
        Use compact single-character square root symbol (when unambiguous).

    root_notation : bool, optional (default=True)
        Set to 'False' for printing exponents of the form 1/n in fractional form.
        By default exponent is printed in root form.

    mat_symbol_style : string, optional (default="plain")
        Set to "bold" for printing MatrixSymbols using a bold mathematical symbol face.
        By default the standard face is used.

    imaginary_unit : string, optional (default="i")
        Letter to use for imaginary unit when use_unicode is True.
        Can be "i" (default) or "j".
    """
    # 调用 pretty 函数进行表达式的漂亮打印，并直接打印输出
    print(pretty(expr, **kwargs))

# 设置 pprint 为 pretty_print 函数的别名
pprint = pretty_print


def pager_print(expr, **settings):
    """Prints expr using the pager, in pretty form.

    This invokes a pager command using pydoc. Lines are not wrapped
    automatically. This routine is meant to be used with a pager that allows
    sideways scrolling, like ``less -S``.

    Parameters are the same as for ``pretty_print``. If you wish to wrap lines,
    pass ``num_columns=None`` to auto-detect the width of the terminal.

    """
    # 导入必要的模块和函数
    from pydoc import pager
    from locale import getpreferredencoding
    
    # 如果设置中没有指定 num_columns，则设置一个非常大的值来禁用行包装
    if 'num_columns' not in settings:
        settings['num_columns'] = 500000  # disable line wrap
    
    # 调用 pretty 函数生成漂亮的打印字符串，编码为当前首选编码后传递给 pager 函数
    pager(pretty(expr, **settings).encode(getpreferredencoding()))
```