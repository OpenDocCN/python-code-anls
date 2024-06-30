# `D:\src\scipysrc\sympy\sympy\printing\repr.py`

```
"""
A Printer for generating executable code.

The most important function here is srepr that returns a string so that the
relation eval(srepr(expr))=expr holds in an appropriate environment.
"""

from __future__ import annotations
from typing import Any

from sympy.core.function import AppliedUndef  # 导入符号计算库中的函数
from sympy.core.mul import Mul  # 导入符号计算库中的乘法操作
from mpmath.libmp import repr_dps, to_str as mlib_to_str  # 导入高精度数学库中的相关函数

from .printer import Printer, print_function  # 从本地引入打印器类和打印函数


class ReprPrinter(Printer):
    printmethod = "_sympyrepr"  # 类属性，用于指定打印方法名称

    _default_settings: dict[str, Any] = {
        "order": None,  # 默认设置字典，指定打印顺序为无序
        "perm_cyclic": True,  # 默认设置字典，指定循环置换为启用
    }

    def reprify(self, args, sep):
        """
        Prints each item in `args` and joins them with `sep`.
        打印 `args` 中的每个项，并使用 `sep` 连接它们。
        """
        return sep.join([self.doprint(item) for item in args])

    def emptyPrinter(self, expr):
        """
        The fallback printer.
        回退打印器。
        """
        if isinstance(expr, str):  # 如果表达式是字符串，直接返回该字符串
            return expr
        elif hasattr(expr, "__srepr__"):  # 如果表达式有 `__srepr__` 方法，调用其返回值
            return expr.__srepr__()
        elif hasattr(expr, "args") and hasattr(expr.args, "__iter__"):  # 如果表达式有 `args` 属性且可迭代
            l = []
            for o in expr.args:
                l.append(self._print(o))
            return expr.__class__.__name__ + '(%s)' % ', '.join(l)  # 返回表达式类名和其参数列表的字符串表示
        elif hasattr(expr, "__module__") and hasattr(expr, "__name__"):  # 如果表达式有 `__module__` 和 `__name__` 属性
            return "<'%s.%s'>" % (expr.__module__, expr.__name__)  # 返回模块名和对象名的字符串表示
        else:
            return str(expr)  # 其他情况，将表达式转换为字符串并返回

    def _print_Add(self, expr, order=None):
        """
        Prints the Add class instance.
        打印 Add 类实例。
        """
        args = self._as_ordered_terms(expr, order=order)  # 获取按顺序排列的加法项
        args = map(self._print, args)  # 将每个项打印
        clsname = type(expr).__name__  # 获取表达式的类名
        return clsname + "(%s)" % ", ".join(args)  # 返回类名和打印后的参数列表的字符串表示

    def _print_Cycle(self, expr):
        """
        Prints the Cycle class instance.
        打印 Cycle 类实例。
        """
        return expr.__repr__()  # 返回 Cycle 类的字符串表示
    # 定义一个方法来打印排列（Permutation）对象的字符串表示
    def _print_Permutation(self, expr):
        # 导入排列（Permutation）和循环（Cycle）类以及 sympy_deprecation_warning 异常
        from sympy.combinatorics.permutations import Permutation, Cycle
        from sympy.utilities.exceptions import sympy_deprecation_warning

        # 获取当前设定的 Permutation.print_cyclic 的值
        perm_cyclic = Permutation.print_cyclic
        # 如果 perm_cyclic 不为 None，则发出 sympy_deprecation_warning
        if perm_cyclic is not None:
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
            # 否则使用默认的 self._settings 中的 perm_cyclic 值
            perm_cyclic = self._settings.get("perm_cyclic", True)

        # 如果 perm_cyclic 为 True，则处理循环表示
        if perm_cyclic:
            # 如果表达式 expr 的大小为 0，则返回 'Permutation()'
            if not expr.size:
                return 'Permutation()'
            # 否则，获取 Cycle 表示，并处理最后一个元素是否为单元素环的情况
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            return 'Permutation%s' % s
        else:
            # 如果 perm_cyclic 为 False，则获取表达式 expr 的支持集合
            s = expr.support()
            # 如果支持集合为空
            if not s:
                # 如果表达式的大小小于 5，则返回表达式的数组形式字符串表示
                if expr.size < 5:
                    return 'Permutation(%s)' % str(expr.array_form)
                # 否则返回带有大小信息的空排列表示
                return 'Permutation([], size=%s)' % expr.size
            # 否则，修剪表达式的数组形式，并根据长度选择要使用的表示形式
            trim = str(expr.array_form[:s[-1] + 1]) + ', size=%s' % expr.size
            use = full = str(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use

    # 定义一个方法来打印函数（Function）对象的字符串表示
    def _print_Function(self, expr):
        # 打印函数名称及其参数列表的字符串表示
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.args])
        return r

    # 定义一个方法来打印 Heaviside 函数的字符串表示
    def _print_Heaviside(self, expr):
        # 与 _print_Function 类似，但使用 pargs 来忽略第二个参数的默认值
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.pargs])
        return r

    # 定义一个方法来打印函数类（FunctionClass）的字符串表示
    def _print_FunctionClass(self, expr):
        # 如果是 AppliedUndef 类的子类，则返回 'Function' 和函数名的字符串表示
        if issubclass(expr, AppliedUndef):
            return 'Function(%r)' % (expr.__name__)
        else:
            # 否则直接返回类名的字符串表示
            return expr.__name__

    # 定义一个方法来打印 Rational(1, 2) 的字符串表示
    def _print_Half(self, expr):
        return 'Rational(1, 2)'

    # 定义一个方法来打印有理常数（RationalConstant）对象的字符串表示
    def _print_RationalConstant(self, expr):
        return str(expr)

    # 定义一个方法来打印 AtomicExpr 对象的字符串表示
    def _print_AtomicExpr(self, expr):
        return str(expr)

    # 定义一个方法来打印 NumberSymbol 对象的字符串表示
    def _print_NumberSymbol(self, expr):
        return str(expr)

    # 定义一个方法来打印整数（Integer）对象的字符串表示
    def _print_Integer(self, expr):
        return 'Integer(%i)' % expr.p

    # 定义一个方法来打印复数集合（Complexes）的字符串表示
    def _print_Complexes(self, expr):
        return 'Complexes'

    # 定义一个方法来打印整数集合（Integers）的字符串表示
    def _print_Integers(self, expr):
        return 'Integers'

    # 定义一个方法来打印自然数集合（Naturals）的字符串表示
    def _print_Naturals(self, expr):
        return 'Naturals'

    # 定义一个方法来打印非负整数集合（Naturals0）的字符串表示
    def _print_Naturals0(self, expr):
        return 'Naturals0'

    # 定义一个方法来打印有理数集合（Rationals）的字符串表示
    def _print_Rationals(self, expr):
        return 'Rationals'

    # 定义一个方法来打印实数集合（Reals）的字符串表示
    def _print_Reals(self, expr):
        return 'Reals'
    # 返回字符串 'EmptySet'，表示空集合
    def _print_EmptySet(self, expr):
        return 'EmptySet'

    # 返回字符串 'UniversalSet'，表示全集
    def _print_UniversalSet(self, expr):
        return 'UniversalSet'

    # 返回字符串 'EmptySequence'，表示空序列
    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    # 将列表 expr 转换为字符串形式，用逗号分隔元素
    def _print_list(self, expr):
        return "[%s]" % self.reprify(expr, ", ")

    # 将字典 expr 转换为字符串形式，每对键值对用逗号分隔
    def _print_dict(self, expr):
        sep = ", "
        dict_kvs = ["%s: %s" % (self.doprint(key), self.doprint(value)) for key, value in expr.items()]
        return "{%s}" % sep.join(dict_kvs)

    # 将集合 expr 转换为字符串形式，如果集合为空则返回 'set()'，否则用逗号分隔元素
    def _print_set(self, expr):
        if not expr:
            return "set()"
        return "{%s}" % self.reprify(expr, ", ")

    # 将 MatrixBase 类型的 expr 转换为字符串形式
    def _print_MatrixBase(self, expr):
        # 对于某些特殊情况的空矩阵，返回其类名及行数、列数、空列表的字符串表示
        if (expr.rows == 0) ^ (expr.cols == 0):
            return '%s(%s, %s, %s)' % (expr.__class__.__name__,
                                       self._print(expr.rows),
                                       self._print(expr.cols),
                                       self._print([]))
        l = []
        # 遍历矩阵的行和列，构建二维列表 l，其中包含矩阵的元素
        for i in range(expr.rows):
            l.append([])
            for j in range(expr.cols):
                l[-1].append(expr[i, j])
        return '%s(%s)' % (expr.__class__.__name__, self._print(l))

    # 返回字符串 'true'，表示布尔值 True
    def _print_BooleanTrue(self, expr):
        return "true"

    # 返回字符串 'false'，表示布尔值 False
    def _print_BooleanFalse(self, expr):
        return "false"

    # 返回字符串 'nan'，表示 Not a Number（NaN）
    def _print_NaN(self, expr):
        return "nan"

    # 将乘法表达式 expr 转换为字符串形式，根据 order 参数决定是否排序因子
    def _print_Mul(self, expr, order=None):
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            args = Mul.make_args(expr)
        
        # 将所有因子转换为字符串形式，并使用类名加上参数列表的形式表示
        args = map(self._print, args)
        clsname = type(expr).__name__
        return clsname + "(%s)" % ", ".join(args)

    # 返回有理数 expr 的字符串表示，格式为 'Rational(分子, 分母)'
    def _print_Rational(self, expr):
        return 'Rational(%s, %s)' % (self._print(expr.p), self._print(expr.q))

    # 返回 PythonRational 类型 expr 的字符串表示，格式为 'PythonRational(分子, 分母)'
    def _print_PythonRational(self, expr):
        return "%s(%d, %d)" % (expr.__class__.__name__, expr.p, expr.q)

    # 返回分数 expr 的字符串表示，格式为 'Fraction(分子, 分母)'
    def _print_Fraction(self, expr):
        return 'Fraction(%s, %s)' % (self._print(expr.numerator), self._print(expr.denominator))

    # 返回浮点数 expr 的字符串表示，包括其精度信息
    def _print_Float(self, expr):
        r = mlib_to_str(expr._mpf_, repr_dps(expr._prec))
        return "%s('%s', precision=%i)" % (expr.__class__.__name__, r, expr._prec)

    # 返回 Sum2 类型 expr 的字符串表示，格式为 "Sum2(函数, (变量, 起始值, 终止值))"
    def _print_Sum2(self, expr):
        return "Sum2(%s, (%s, %s, %s))" % (self._print(expr.f), self._print(expr.i),
                                           self._print(expr.a), self._print(expr.b))

    # 返回字符串 s 的字符串表示，格式为 'Str(字符串)'
    def _print_Str(self, s):
        return "%s(%s)" % (s.__class__.__name__, self._print(s.name))
    # 定义一个方法，用于打印符号表达式
    def _print_Symbol(self, expr):
        # 获取表达式的原始假设字典
        d = expr._assumptions_orig
        # 如果表达式是一个虚拟符号，则将其虚拟索引作为假设加入字典
        if expr.is_Dummy:
            d['dummy_index'] = expr.dummy_index

        # 如果假设字典为空
        if d == {}:
            return "%s(%s)" % (expr.__class__.__name__, self._print(expr.name))
        else:
            # 构建带有假设属性的表达式字符串
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s)" % (expr.__class__.__name__,
                                   self._print(expr.name), ', '.join(attr))

    # 定义一个方法，用于打印坐标符号表达式
    def _print_CoordinateSymbol(self, expr):
        # 获取表达式的生成器假设字典
        d = expr._assumptions.generator

        # 如果假设字典为空
        if d == {}:
            return "%s(%s, %s)" % (
                expr.__class__.__name__,
                self._print(expr.coord_sys),
                self._print(expr.index)
            )
        else:
            # 构建带有假设属性的表达式字符串
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s, %s)" % (
                expr.__class__.__name__,
                self._print(expr.coord_sys),
                self._print(expr.index),
                ', '.join(attr)
            )

    # 定义一个方法，用于打印谓词表达式
    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    # 定义一个方法，用于打印应用的谓词表达式
    def _print_AppliedPredicate(self, expr):
        # 获取表达式的参数列表
        args = expr._args
        return "%s(%s)" % (expr.__class__.__name__, self.reprify(args, ", "))

    # 定义一个方法，用于打印字符串表达式
    def _print_str(self, expr):
        return repr(expr)

    # 定义一个方法，用于打印元组表达式
    def _print_tuple(self, expr):
        # 如果元组只有一个元素，则返回形如 "(element,)" 的字符串
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            # 否则返回形如 "(element1, element2, ...)" 的字符串
            return "(%s)" % self.reprify(expr, ", ")

    # 定义一个方法，用于打印通配符函数表达式
    def _print_WildFunction(self, expr):
        return "%s('%s')" % (expr.__class__.__name__, expr.name)

    # 定义一个方法，用于打印代数数表达式
    def _print_AlgebraicNumber(self, expr):
        return "%s(%s, %s)" % (expr.__class__.__name__,
            self._print(expr.root), self._print(expr.coeffs()))

    # 定义一个方法，用于打印多项式环表达式
    def _print_PolyRing(self, ring):
        return "%s(%s, %s, %s)" % (ring.__class__.__name__,
            self._print(ring.symbols), self._print(ring.domain), self._print(ring.order))

    # 定义一个方法，用于打印分式域表达式
    def _print_FracField(self, field):
        return "%s(%s, %s, %s)" % (field.__class__.__name__,
            self._print(field.symbols), self._print(field.domain), self._print(field.order))

    # 定义一个方法，用于打印多项式元素表达式
    def _print_PolyElement(self, poly):
        # 获取多项式的项列表，并根据环的顺序进行排序
        terms = list(poly.terms())
        terms.sort(key=poly.ring.order, reverse=True)
        return "%s(%s, %s)" % (poly.__class__.__name__, self._print(poly.ring), self._print(terms))

    # 定义一个方法，用于打印分式元素表达式
    def _print_FracElement(self, frac):
        # 获取分子和分母的项列表，并根据域的顺序进行排序
        numer_terms = list(frac.numer.terms())
        numer_terms.sort(key=frac.field.order, reverse=True)
        denom_terms = list(frac.denom.terms())
        denom_terms.sort(key=frac.field.order, reverse=True)
        numer = self._print(numer_terms)
        denom = self._print(denom_terms)
        return "%s(%s, %s, %s)" % (frac.__class__.__name__, self._print(frac.field), numer, denom)
    # 定义一个方法用于打印分式域的表示
    def _print_FractionField(self, domain):
        # 获取域的类名
        cls = domain.__class__.__name__
        # 打印域的字段
        field = self._print(domain.field)
        # 返回格式化后的字符串表示
        return "%s(%s)" % (cls, field)

    # 定义一个方法用于打印多项式环的基础表示
    def _print_PolynomialRingBase(self, ring):
        # 获取环的类名
        cls = ring.__class__.__name__
        # 打印环的定义域
        dom = self._print(ring.domain)
        # 打印环的生成元
        gens = ', '.join(map(self._print, ring.gens))
        # 获取环的次序
        order = str(ring.order)
        # 如果环的次序不等于默认次序，则附加次序信息
        if order != ring.default_order:
            orderstr = ", order=" + order
        else:
            orderstr = ""
        # 返回格式化后的字符串表示
        return "%s(%s, %s%s)" % (cls, dom, gens, orderstr)

    # 定义一个方法用于打印分段多项式的表示
    def _print_DMP(self, p):
        # 获取分段多项式的类名
        cls = p.__class__.__name__
        # 打印分段多项式的表示
        rep = self._print(p.to_list())
        # 打印分段多项式的定义域
        dom = self._print(p.dom)
        # 返回格式化后的字符串表示
        return "%s(%s, %s)" % (cls, rep, dom)

    # 定义一个方法用于打印单生成有限扩展的表示
    def _print_MonogenicFiniteExtension(self, ext):
        # 因为srepr(ext.modulus)展示的扩展树不实用，直接返回简单的表示
        return "FiniteExtension(%s)" % str(ext.modulus)

    # 定义一个方法用于打印扩展元素的表示
    def _print_ExtensionElement(self, f):
        # 打印扩展元素的表示
        rep = self._print(f.rep)
        # 打印扩展元素的扩展对象
        ext = self._print(f.ext)
        # 返回格式化后的字符串表示
        return "ExtElem(%s, %s)" % (rep, ext)
# 使用装饰器 @print_function(ReprPrinter) 包装函数 srepr，使用 ReprPrinter 打印函数的输出
@print_function(ReprPrinter)
# 定义函数 srepr，接受一个表达式 expr 和额外的设置参数 settings
def srepr(expr, **settings):
    """return expr in repr form"""
    # 创建一个 ReprPrinter 对象，使用给定的设置参数 settings，然后打印表达式 expr 的表示形式（repr form）
    return ReprPrinter(settings).doprint(expr)
```