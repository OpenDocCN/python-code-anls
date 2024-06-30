# `D:\src\scipysrc\sympy\sympy\printing\str.py`

```
"""
A Printer for generating readable representation of most SymPy classes.
"""

# 导入必要的模块和类
from __future__ import annotations
from typing import Any

from sympy.core import S, Rational, Pow, Basic, Mul, Number  # 导入 SymPy 核心类
from sympy.core.mul import _keep_coeff  # 导入 SymPy 中的 _keep_coeff 函数
from sympy.core.numbers import Integer  # 导入 SymPy 中的 Integer 类
from sympy.core.relational import Relational  # 导入 SymPy 中的 Relational 类
from sympy.core.sorting import default_sort_key  # 导入 SymPy 中的 default_sort_key 函数
from sympy.utilities.iterables import sift  # 导入 SymPy 中的 sift 函数
from .precedence import precedence, PRECEDENCE  # 导入本地模块 precedence 和 PRECEDENCE
from .printer import Printer, print_function  # 导入本地模块 Printer 和 print_function
from mpmath.libmp import prec_to_dps, to_str as mlib_to_str  # 导入 mpmath 库中的 prec_to_dps 和 mlib_to_str 函数


class StrPrinter(Printer):
    # 类变量
    printmethod = "_sympystr"
    _default_settings: dict[str, Any] = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
        "abbrev": False,
        "perm_cyclic": True,
        "min": None,
        "max": None,
    }

    _relationals: dict[str, str] = {}  # 空的关系型字典

    # 方法：给表达式添加必要的括号以保证正确的打印顺序
    def parenthesize(self, item, level, strict=False):
        if (precedence(item) < level) or ((not strict) and precedence(item) <= level):
            return "(%s)" % self._print(item)
        else:
            return self._print(item)

    # 方法：将表达式的参数转换成字符串，并用指定的分隔符连接
    def stringify(self, args, sep, level=0):
        return sep.join([self.parenthesize(item, level) for item in args])

    # 方法：处理空打印机情况下的表达式
    def emptyPrinter(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, Basic):
            return repr(expr)
        else:
            return str(expr)

    # 方法：打印 Add 类型的表达式
    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)

        prec = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if t.startswith('-') and not term.is_Add:
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < prec or term.is_Add:
                l.extend([sign, "(%s)" % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    # 方法：打印 BooleanTrue 类型的表达式
    def _print_BooleanTrue(self, expr):
        return "True"

    # 方法：打印 BooleanFalse 类型的表达式
    def _print_BooleanFalse(self, expr):
        return "False"

    # 方法：打印 Not 类型的表达式
    def _print_Not(self, expr):
        return '~%s' % (self.parenthesize(expr.args[0], PRECEDENCE["Not"]))

    # 方法：打印 And 类型的表达式
    def _print_And(self, expr):
        args = list(expr.args)
        for j, i in enumerate(args):
            if isinstance(i, Relational) and (
                    i.canonical.rhs is S.NegativeInfinity):
                args.insert(0, args.pop(j))
        return self.stringify(args, " & ", PRECEDENCE["BitwiseAnd"])

    # 方法：打印 Or 类型的表达式
    def _print_Or(self, expr):
        return self.stringify(expr.args, " | ", PRECEDENCE["BitwiseOr"])

    # 方法：打印 Xor 类型的表达式
    def _print_Xor(self, expr):
        return self.stringify(expr.args, " ^ ", PRECEDENCE["BitwiseXor"])
    def _print_AppliedPredicate(self, expr):
        # 返回一个字符串，格式为函数名(参数列表)，使用 self._print 方法处理函数名和参数
        return '%s(%s)' % (
            self._print(expr.function), self.stringify(expr.arguments, ", "))

    def _print_Basic(self, expr):
        # 创建一个列表 l，其中包含对表达式 expr.args 中每个对象调用 self._print 方法的结果
        l = [self._print(o) for o in expr.args]
        # 返回一个字符串，格式为类名(参数列表)，参数为逗号分隔的列表 l 的内容
        return expr.__class__.__name__ + "(%s)" % ", ".join(l)

    def _print_BlockMatrix(self, B):
        # 检查 B.blocks 的形状是否为 (1, 1)
        if B.blocks.shape == (1, 1):
            # 如果是，则打印 B.blocks[0, 0] 的内容
            self._print(B.blocks[0, 0])
        # 返回一个字符串，内容为 B.blocks 的字符串表示
        return self._print(B.blocks)

    def _print_Catalan(self, expr):
        # 返回字符串 'Catalan'
        return 'Catalan'

    def _print_ComplexInfinity(self, expr):
        # 返回字符串 'zoo'
        return 'zoo'

    def _print_ConditionSet(self, s):
        # 对象 s 的 sym 和 condition 属性进行打印，并格式化为字符串
        args = tuple([self._print(i) for i in (s.sym, s.condition)])
        # 如果 s.base_set 是 S.UniversalSet，则返回格式化后的字符串
        if s.base_set is S.UniversalSet:
            return 'ConditionSet(%s, %s)' % args
        # 否则，将 s.base_set 也添加到格式化后的字符串中返回
        args += (self._print(s.base_set),)
        return 'ConditionSet(%s, %s, %s)' % args

    def _print_Derivative(self, expr):
        # 获取表达式 expr 的主表达式和变量计数，使用 self._print 方法打印结果
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        # 返回一个字符串，格式为 'Derivative(主表达式, 变量列表)'
        return 'Derivative(%s)' % ", ".join((self._print(arg) for arg in [dexpr] + dvars))

    def _print_dict(self, d):
        # 获取字典 d 的键，按照默认排序键排序
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        # 遍历排序后的键列表，创建每个键值对的字符串并添加到 items 列表中
        for key in keys:
            item = "%s: %s" % (self._print(key), self._print(d[key]))
            items.append(item)

        # 返回一个字符串，格式为 '{键1: 值1, 键2: 值2, ...}'
        return "{%s}" % ", ".join(items)

    def _print_Dict(self, expr):
        # 调用 self._print_dict 方法打印表达式 expr
        return self._print_dict(expr)

    def _print_RandomDomain(self, d):
        # 如果对象 d 有属性 'as_boolean'，则返回 'Domain: ' + d.as_boolean() 的打印结果
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        # 如果对象 d 有属性 'set'，则返回 'Domain: ' + d.symbols + ' in ' + d.set 的打印结果
        elif hasattr(d, 'set'):
            return ('Domain: ' + self._print(d.symbols) + ' in ' +
                    self._print(d.set))
        # 否则，返回 'Domain on ' + d.symbols 的打印结果
        else:
            return 'Domain on ' + self._print(d.symbols)

    def _print_Dummy(self, expr):
        # 返回下划线 + 表达式 expr 的名称
        return '_' + expr.name

    def _print_EulerGamma(self, expr):
        # 返回字符串 'EulerGamma'
        return 'EulerGamma'

    def _print_Exp1(self, expr):
        # 返回字符串 'E'
        return 'E'

    def _print_ExprCondPair(self, expr):
        # 返回一个字符串，格式为 '(表达式, 条件)'，使用 self._print 方法处理表达式和条件
        return '(%s, %s)' % (self._print(expr.expr), self._print(expr.cond))

    def _print_Function(self, expr):
        # 返回一个字符串，格式为 函数名(参数列表)，使用 self.stringify 方法处理参数列表
        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

    def _print_GoldenRatio(self, expr):
        # 返回字符串 'GoldenRatio'
        return 'GoldenRatio'

    def _print_Heaviside(self, expr):
        # 返回与 _print_Function 相同的字符串，但使用 pargs 以抑制默认的第二个参数 1/2
        return expr.func.__name__ + "(%s)" % self.stringify(expr.pargs, ", ")

    def _print_TribonacciConstant(self, expr):
        # 返回字符串 'TribonacciConstant'
        return 'TribonacciConstant'

    def _print_ImaginaryUnit(self, expr):
        # 返回字符串 'I'
        return 'I'

    def _print_Infinity(self, expr):
        # 返回字符串 'oo'
        return 'oo'
    # 打印积分表达式的字符串表示，包括积分的限制条件
    def _print_Integral(self, expr):
        # 定义内部函数，将积分的限制条件转换为字符串
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        
        # 将所有限制条件转换为字符串，并用逗号连接
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        # 返回积分表达式的字符串表示，包括函数和限制条件
        return 'Integral(%s, %s)' % (self._print(expr.function), L)

    # 打印区间表达式的字符串表示
    def _print_Interval(self, i):
        fin =  'Interval{m}({a}, {b})'
        a, b, l, r = i.args
        # 根据区间的无限性和开闭性确定字符串格式
        if a.is_infinite and b.is_infinite:
            m = ''
        elif a.is_infinite and not r:
            m = ''
        elif b.is_infinite and not l:
            m = ''
        elif not l and not r:
            m = ''
        elif l and r:
            m = '.open'
        elif l:
            m = '.Lopen'
        else:
            m = '.Ropen'
        # 返回格式化后的区间字符串表示
        return fin.format(**{'a': a, 'b': b, 'm': m})

    # 打印积分累积边界的字符串表示
    def _print_AccumulationBounds(self, i):
        # 返回累积边界对象的字符串表示，包括最小值和最大值
        return "AccumBounds(%s, %s)" % (self._print(i.min),
                                        self._print(i.max))

    # 打印求逆表达式的字符串表示
    def _print_Inverse(self, I):
        # 返回求逆操作的字符串表示，使用适当的优先级括号
        return "%s**(-1)" % self.parenthesize(I.arg, PRECEDENCE["Pow"])

    # 打印Lambda表达式的字符串表示
    def _print_Lambda(self, obj):
        expr = obj.expr
        sig = obj.signature
        # 确保Lambda表达式的签名只有一个符号时输出单个符号而不是元组
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        # 返回Lambda表达式的字符串表示，包括签名和表达式
        return "Lambda(%s, %s)" % (self._print(sig), self._print(expr))

    # 打印格操作的字符串表示
    def _print_LatticeOp(self, expr):
        # 对操作的参数按默认排序键排序，并返回操作名称及其参数的字符串表示
        args = sorted(expr.args, key=default_sort_key)
        return expr.func.__name__ + "(%s)" % ", ".join(self._print(arg) for arg in args)

    # 打印极限表达式的字符串表示
    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args
        # 返回极限表达式的字符串表示，包括表达式、自变量、趋近点和趋近方向
        return "Limit(%s, %s, %s, dir='%s')" % tuple(map(self._print, (e, z, z0, dir)))


    # 打印列表表达式的字符串表示
    def _print_list(self, expr):
        # 返回列表表达式的字符串表示，用逗号分隔每个元素
        return "[%s]" % self.stringify(expr, ", ")

    # 打印列表表达式的字符串表示（别名）
    def _print_List(self, expr):
        return self._print_list(expr)

    # 打印矩阵基类对象的字符串表示
    def _print_MatrixBase(self, expr):
        # 返回矩阵基类对象的格式化字符串表示
        return expr._format_str(self)

    # 打印矩阵元素表达式的字符串表示
    def _print_MatrixElement(self, expr):
        # 返回矩阵元素表达式的字符串表示，包括父矩阵、行索引和列索引
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))

    # 打印矩阵切片表达式的字符串表示
    def _print_MatrixSlice(self, expr):
        # 定义内部函数，将矩阵切片参数转换为字符串表示
        def strslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return ':'.join((self._print(arg) for arg in x))
        
        # 返回矩阵切片表达式的字符串表示，包括父矩阵、行切片和列切片
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + '[' +
                strslice(expr.rowslice, expr.parent.rows) + ', ' +
                strslice(expr.colslice, expr.parent.cols) + ']')

    # 打印延迟向量对象的字符串表示
    def _print_DeferredVector(self, expr):
        # 返回延迟向量对象的名称字符串表示
        return expr.name
    # 将表达式中的系数和乘积分离，返回乘积的字符串表示形式
    def _print_MatMul(self, expr):
        # 将表达式分解为系数和乘积
        c, m = expr.as_coeff_mmul()
    
        sign = ""
        # 如果系数是数值类型
        if c.is_number:
            # 分离实部和虚部
            re, im = c.as_real_imag()
            # 如果虚部为零且实部为负数，则取相反数并记录负号
            if im.is_zero and re.is_negative:
                expr = _keep_coeff(-c, m)
                sign = "-"
            # 如果实部为零且虚部为负数，则取相反数并记录负号
            elif re.is_zero and im.is_negative:
                expr = _keep_coeff(-c, m)
                sign = "-"
    
        # 构造乘积表达式的字符串表示，每个参数使用适当的括号
        return sign + '*'.join(
            [self.parenthesize(arg, precedence(expr)) for arg in expr.args]
        )
    
    # 返回应用元素级函数的字符串表示形式
    def _print_ElementwiseApplyFunction(self, expr):
        return "{}.({})".format(
            expr.function,
            self._print(expr.expr),
        )
    
    # 返回NaN的字符串表示形式
    def _print_NaN(self, expr):
        return 'nan'
    
    # 返回负无穷的字符串表示形式
    def _print_NegativeInfinity(self, expr):
        return '-oo'
    
    # 返回O记号的字符串表示形式
    def _print_Order(self, expr):
        # 如果没有变量或者所有点都是零
        if not expr.variables or all(p is S.Zero for p in expr.point):
            if len(expr.variables) <= 1:
                # 单个变量或点为零，返回简单的O(expr)
                return 'O(%s)' % self._print(expr.expr)
            else:
                # 多个变量或点不为零，返回复杂结构的O(expr, variables)
                return 'O(%s)' % self.stringify((expr.expr,) + expr.variables, ', ', 0)
        else:
            # 否则，返回O(args)
            return 'O(%s)' % self.stringify(expr.args, ', ', 0)
    
    # 返回序数的字符串表示形式
    def _print_Ordinal(self, expr):
        return expr.__str__()
    
    # 返回循环的字符串表示形式
    def _print_Cycle(self, expr):
        return expr.__str__()
    def _print_Permutation(self, expr):
        # 导入排列相关的类和函数
        from sympy.combinatorics.permutations import Permutation, Cycle
        from sympy.utilities.exceptions import sympy_deprecation_warning

        # 获取当前设置中的 Permutation.print_cyclic 值
        perm_cyclic = Permutation.print_cyclic
        # 如果设置不为 None，发出 Sympy 弃用警告
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
            # 如果设置为 None，则从当前设置中获取 perm_cyclic 值
            perm_cyclic = self._settings.get("perm_cyclic", True)

        # 如果 perm_cyclic 为 True，使用循环表示法打印排列
        if perm_cyclic:
            # 如果表达式的大小为 0，返回空循环
            if not expr.size:
                return '()'
            # 在使用循环表示法之前，检查最后一个元素是否为单一循环，并将其移到字符串的开头
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            # 如果不是从头开始的单一循环且没有逗号，则重新排列字符串
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            s = s.replace(',', '')
            return s
        else:
            # 如果 perm_cyclic 为 False，使用支持集合来打印排列
            s = expr.support()
            # 如果支持集合为空
            if not s:
                # 如果表达式的大小小于 5，返回使用数组形式打印的排列
                if expr.size < 5:
                    return 'Permutation(%s)' % self._print(expr.array_form)
                # 否则返回空排列，指定大小
                return 'Permutation([], size=%s)' % self._print(expr.size)
            # 如果支持集合非空，截断数组形式的排列并使用更短的形式打印
            trim = self._print(expr.array_form[:s[-1] + 1]) + ', size=%s' % self._print(expr.size)
            full = self._print(expr.array_form)
            # 如果截断后的字符串长度小于完整形式，则使用截断形式
            use = trim if len(trim) < len(full) else full
            return 'Permutation(%s)' % use

    def _print_Subs(self, obj):
        # 获取 Subs 对象的表达式、旧变量和新变量，并打印 Subs 表达式
        expr, old, new = obj.args
        if len(obj.point) == 1:
            old = old[0]
            new = new[0]
        return "Subs(%s, %s, %s)" % (
            self._print(expr), self._print(old), self._print(new))

    def _print_TensorIndex(self, expr):
        # 打印 TensorIndex 对象
        return expr._print()

    def _print_TensorHead(self, expr):
        # 打印 TensorHead 对象
        return expr._print()

    def _print_Tensor(self, expr):
        # 打印 Tensor 对象
        return expr._print()

    def _print_TensMul(self, expr):
        # 打印 TensMul 对象，格式为 "系数*因子1*因子2*..."
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "*".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        # 打印 TensAdd 对象
        return expr._print()

    def _print_ArraySymbol(self, expr):
        # 打印 ArraySymbol 对象的名称
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        # 打印 ArrayElement 对象，格式为 "数组名称[索引1, 索引2, ...]"
        return "%s[%s]" % (
            self.parenthesize(expr.name, PRECEDENCE["Func"], True), ", ".join([self._print(i) for i in expr.indices]))

    def _print_PermutationGroup(self, expr):
        # 打印 PermutationGroup 对象
        p = ['    %s' % self._print(a) for a in expr.args]
        return 'PermutationGroup([\n%s])' % ',\n'.join(p)
    # 定义一个方法用于打印 Pi，始终返回字符串 'pi'
    def _print_Pi(self, expr):
        return 'pi'

    # 定义一个方法用于打印多项式环对象的字符串表示
    def _print_PolyRing(self, ring):
        # 格式化返回一个描述多项式环的字符串，包括环中符号的列表、定义域和顺序
        return "Polynomial ring in %s over %s with %s order" % \
            (", ".join((self._print(rs) for rs in ring.symbols)),
            self._print(ring.domain), self._print(ring.order))

    # 定义一个方法用于打印有理函数域对象的字符串表示
    def _print_FracField(self, field):
        # 格式化返回一个描述有理函数域的字符串，包括域中符号的列表、定义域和顺序
        return "Rational function field in %s over %s with %s order" % \
            (", ".join((self._print(fs) for fs in field.symbols)),
            self._print(field.domain), self._print(field.order))

    # 定义一个方法用于打印自由群元素对象的字符串表示
    def _print_FreeGroupElement(self, elm):
        # 直接返回自由群元素对象的字符串形式
        return elm.__str__()

    # 定义一个方法用于打印高斯元素对象的字符串表示
    def _print_GaussianElement(self, poly):
        # 格式化返回一个描述高斯元素的字符串，包括实部和虚部
        return "(%s + %s*I)" % (poly.x, poly.y)

    # 定义一个方法用于打印多项式元素对象的字符串表示
    def _print_PolyElement(self, poly):
        # 使用多项式对象的方法生成其字符串形式，并根据指定的优先级和运算符格式化返回
        return poly.str(self, PRECEDENCE, "%s**%s", "*")

    # 定义一个方法用于打印有理函数元素对象的字符串表示
    def _print_FracElement(self, frac):
        # 如果有理函数的分母为1，直接打印其分子
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            # 否则，使用自定义方法加括号打印分子和分母的表达式
            numer = self.parenthesize(frac.numer, PRECEDENCE["Mul"], strict=True)
            denom = self.parenthesize(frac.denom, PRECEDENCE["Atom"], strict=True)
            return numer + "/" + denom
    # 定义一个方法用于打印多项式表达式
    def _print_Poly(self, expr):
        # 设置原子操作符的优先级
        ATOM_PREC = PRECEDENCE["Atom"] - 1
        # 准备生成器表达式列表，用于将表达式的生成器括起来
        gens = [ self.parenthesize(s, ATOM_PREC) for s in expr.gens ]
        terms = []

        # 遍历表达式的每个单项式及其系数
        for monom, coeff in expr.terms():
            s_monom = []

            # 处理单项式中的每个指数
            for i, e in enumerate(monom):
                if e > 0:
                    if e == 1:
                        s_monom.append(gens[i])
                    else:
                        s_monom.append(gens[i] + "**%d" % e)

            # 将单项式的各部分用 '*' 连接起来
            s_monom = "*".join(s_monom)

            # 处理系数部分
            if coeff.is_Add:
                if s_monom:
                    s_coeff = "(" + self._print(coeff) + ")"
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue
                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue
                s_coeff = self._print(coeff)

            # 根据单项式是否为空来构造整个项的字符串表示
            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + "*" + s_monom

            # 根据项的符号决定添加到 terms 列表中
            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        # 处理 terms 列表的第一个元素是符号的情况
        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)
            if modifier == '-':
                terms[0] = '-' + terms[0]

        # 设置格式字符串，用于格式化整个表达式
        format = expr.__class__.__name__ + "(%s, %s"

        # 尝试获取表达式的模量并格式化到格式字符串中，如果失败则获取域信息
        from sympy.polys.polyerrors import PolynomialError
        try:
            format += ", modulus=%s" % expr.get_modulus()
        except PolynomialError:
            format += ", domain='%s'" % expr.get_domain()

        format += ")"

        # 去掉生成器表达式中多余的括号
        for index, item in enumerate(gens):
            if len(item) > 2 and (item[:1] == "(" and item[len(item) - 1:] == ")"):
                gens[index] = item[1:len(item) - 1]

        # 返回格式化后的表达式字符串
        return format % (' '.join(terms), ', '.join(gens))

    # 定义一个方法用于打印 UniversalSet 对象
    def _print_UniversalSet(self, p):
        return 'UniversalSet'

    # 定义一个方法用于打印代数数对象
    def _print_AlgebraicNumber(self, expr):
        # 如果代数数已经被别名化，则打印其作为多项式表达式的形式
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())
    # 定义一个打印函数，用于处理幂运算表达式（Pow）
    
    def _print_Pow(self, expr, rational=False):
        """Printing helper function for ``Pow``
    
        Parameters
        ==========
    
        rational : bool, optional
            If ``True``, it will not attempt printing ``sqrt(x)`` or
            ``x**S.Half`` as ``sqrt``, and will use ``x**(1/2)``
            instead.
    
            See examples for additional details
    
        Examples
        ========
    
        >>> from sympy import sqrt, StrPrinter
        >>> from sympy.abc import x
    
        How ``rational`` keyword works with ``sqrt``:
    
        >>> printer = StrPrinter()
        >>> printer._print_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._print_Pow(sqrt(x), rational=False)
        'sqrt(x)'
        >>> printer._print_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._print_Pow(1/sqrt(x), rational=False)
        '1/sqrt(x)'
    
        Notes
        =====
    
        ``sqrt(x)`` is canonicalized as ``Pow(x, S.Half)`` in SymPy,
        so there is no need of defining a separate printer for ``sqrt``.
        Instead, it should be handled here as well.
        """
        PREC = precedence(expr)
    
        # 处理指数为 S.Half 的情况，根据 rational 参数决定是否输出 sqrt 形式
        if expr.exp is S.Half and not rational:
            return "sqrt(%s)" % self._print(expr.base)
    
        # 对可交换的指数进行处理
        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # 注意：这里不使用 "expr.exp == -S.Half" 的比较，因为会匹配到 -0.5，我们不希望这样。
                return "%s/sqrt(%s)" % tuple((self._print(arg) for arg in (S.One, expr.base)))
            if expr.exp is -S.One:
                # 类似于指数为 S.Half 的情况，不使用 "==" 比较。
                return '%s/%s' % (self._print(S.One),
                                  self.parenthesize(expr.base, PREC, strict=False))
    
        # 对指数进行括号化处理，并输出表达式
        e = self.parenthesize(expr.exp, PREC, strict=False)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and expr.exp.q != 1:
            # 括号化指数应为 '(Rational(a, b))'，因此去除括号，但要做检查以确保。
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e[1:-1])
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e)
    
    # 处理未评估的表达式（UnevaluatedExpr），直接输出其第一个参数的打印形式
    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])
    
    # 处理矩阵幂（MatPow），输出形式为基数和指数的幂运算形式
    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False),
                           self.parenthesize(expr.exp, PREC, strict=False))
    
    # 处理整数（Integer），根据设置输出对应的形式，使用 S() 包装整数
    def _print_Integer(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(%s)" % (expr)
        return str(expr.p)
    
    # 处理整数集合（Integers），直接返回字符串 'Integers'
    def _print_Integers(self, expr):
        return 'Integers'
    
    # 处理自然数集合（Naturals），直接返回字符串 'Naturals'
    def _print_Naturals(self, expr):
        return 'Naturals'
    
    # 处理非负整数集合（Naturals0），直接返回字符串 'Naturals0'
    def _print_Naturals0(self, expr):
        return 'Naturals0'
    # 返回字符串 'Rationals'，表示有理数集合
    def _print_Rationals(self, expr):
        return 'Rationals'

    # 返回字符串 'Reals'，表示实数集合
    def _print_Reals(self, expr):
        return 'Reals'

    # 返回字符串 'Complexes'，表示复数集合
    def _print_Complexes(self, expr):
        return 'Complexes'

    # 返回字符串 'EmptySet'，表示空集
    def _print_EmptySet(self, expr):
        return 'EmptySet'

    # 返回字符串 'EmptySequence'，表示空序列
    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    # 将整数表达式转换为字符串返回
    def _print_int(self, expr):
        return str(expr)

    # 将大整数表达式转换为字符串返回
    def _print_mpz(self, expr):
        return str(expr)

    # 将有理数表达式转换为字符串返回
    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            # 如果设置了 sympy_integers，则返回符号化有理数形式，否则返回分数形式
            if self._settings.get("sympy_integers", False):
                return "S(%s)/%s" % (expr.p, expr.q)
            return "%s/%s" % (expr.p, expr.q)

    # 将 Python 的有理数表达式转换为字符串返回
    def _print_PythonRational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            return "%d/%d" % (expr.p, expr.q)

    # 将分数表达式转换为字符串返回
    def _print_Fraction(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    # 将 MPQ（多精度有理数）表达式转换为字符串返回
    def _print_mpq(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    # 将浮点数表达式转换为字符串返回
    def _print_Float(self, expr):
        # 获取浮点数的精度
        prec = expr._prec
        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec)
        # 根据设置决定是否保留全精度
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1
        low = self._settings["min"] if "min" in self._settings else None
        high = self._settings["max"] if "max" in self._settings else None
        # 将浮点数转换为字符串
        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
        # 处理特殊情况下的字符串格式化
        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        if rv.startswith('+'):
            rv = rv[1:]  # 去除正号
        return rv

    # 将关系表达式转换为字符串返回
    def _print_Relational(self, expr):
        # 定义字符映射，将关系运算符映射为字符串
        charmap = {
            "==": "Eq",
            "!=": "Ne",
            ":=": "Assignment",
            '+=': "AddAugmentedAssignment",
            "-=": "SubAugmentedAssignment",
            "*=": "MulAugmentedAssignment",
            "/=": "DivAugmentedAssignment",
            "%=": "ModAugmentedAssignment",
        }
        # 如果关系运算符在映射中，则格式化成对应的字符串返回
        if expr.rel_op in charmap:
            return '%s(%s, %s)' % (charmap[expr.rel_op], self._print(expr.lhs),
                                   self._print(expr.rhs))
        # 否则按默认格式化返回关系表达式字符串
        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
                           self._relationals.get(expr.rel_op) or expr.rel_op,
                           self.parenthesize(expr.rhs, precedence(expr)))
    # 打印表达式的复数根，返回格式化字符串 "CRootOf(expr, index)"
    def _print_ComplexRootOf(self, expr):
        return "CRootOf(%s, %d)" % (self._print_Add(expr.expr,  order='lex'),
                                    expr.index)

    # 打印根和求和函数的表达式，返回格式化字符串 "RootSum(expr, fun)"
    def _print_RootSum(self, expr):
        args = [self._print_Add(expr.expr, order='lex')]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        return "RootSum(%s)" % ", ".join(args)

    # 打印Groebner基础类的表达式，返回格式化字符串 "cls(exprs, gens, domain, order)"
    def _print_GroebnerBasis(self, basis):
        cls = basis.__class__.__name__

        # 打印基础表达式列表
        exprs = [self._print_Add(arg, order=basis.order) for arg in basis.exprs]
        exprs = "[%s]" % ", ".join(exprs)

        # 打印生成器列表
        gens = [ self._print(gen) for gen in basis.gens ]

        # 打印域和顺序
        domain = "domain='%s'" % self._print(basis.domain)
        order = "order='%s'" % self._print(basis.order)

        args = [exprs] + gens + [domain, order]

        return "%s(%s)" % (cls, ", ".join(args))

    # 打印集合类的表达式，返回格式化字符串 "set(item1, item2, ...)"
    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)

        # 打印集合中的每个项目
        args = ', '.join(self._print(item) for item in items)
        if not args:
            return "set()"
        return '{%s}' % args

    # 打印有限集合类的表达式，返回格式化字符串 "FiniteSet(item1, item2, ...)"
    def _print_FiniteSet(self, s):
        from sympy.sets.sets import FiniteSet
        items = sorted(s, key=default_sort_key)

        # 打印集合中的每个项目
        args = ', '.join(self._print(item) for item in items)
        # 如果集合中包含 FiniteSet，则返回带有 FiniteSet 标签的格式化字符串
        if any(item.has(FiniteSet) for item in items):
            return 'FiniteSet({})'.format(args)
        return '{{{}}}'.format(args)

    # 打印分区类的表达式，返回格式化字符串 "Partition(item1, item2, ...)"
    def _print_Partition(self, s):
        items = sorted(s, key=default_sort_key)

        # 打印分区中的每个项目
        args = ', '.join(self._print(arg) for arg in items)
        return 'Partition({})'.format(args)

    # 打印冻结集合类的表达式，返回格式化字符串 "frozenset(item1, item2, ...)"
    def _print_frozenset(self, s):
        if not s:
            return "frozenset()"
        return "frozenset(%s)" % self._print_set(s)

    # 打印求和表达式，返回格式化字符串 "Sum(function, limits)"
    def _print_Sum(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Sum(%s, %s)' % (self._print(expr.function), L)

    # 打印符号表达式，返回符号名称
    def _print_Symbol(self, expr):
        return expr.name
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    # 打印单位矩阵，返回字符串 "I"
    def _print_Identity(self, expr):
        return "I"

    # 打印零矩阵，返回字符串 "0"
    def _print_ZeroMatrix(self, expr):
        return "0"

    # 打印单位矩阵，返回字符串 "1"
    def _print_OneMatrix(self, expr):
        return "1"

    # 打印谓词表达式，返回格式化字符串 "Q.name"
    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    # 打印字符串表达式，返回字符串表示
    def _print_str(self, expr):
        return str(expr)

    # 打印元组表达式，返回格式化字符串 "(item1, item2, ...)"
    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.stringify(expr, ", ")

    # 打印元组表达式，返回格式化字符串 "(item1, item2, ...)"
    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    # 打印转置表达式，返回格式化字符串 "T.arg.T"
    def _print_Transpose(self, T):
        return "%s.T" % self.parenthesize(T.arg, PRECEDENCE["Pow"])
    # 返回表示 Uniform 分布的字符串，包括最小值和最大值
    def _print_Uniform(self, expr):
        return "Uniform(%s, %s)" % (self._print(expr.a), self._print(expr.b))

    # 根据设置返回 Quantity 的名称或缩写
    def _print_Quantity(self, expr):
        if self._settings.get("abbrev", False):
            return "%s" % expr.abbrev
        return "%s" % expr.name

    # 返回表示 Quaternion 的字符串表达式
    def _print_Quaternion(self, expr):
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True) for i in expr.args]
        a = [s[0]] + [i+"*"+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    # 返回 Dimension 对象的字符串表示
    def _print_Dimension(self, expr):
        return str(expr)

    # 返回 Wild 对象的名称后加下划线的字符串表示
    def _print_Wild(self, expr):
        return expr.name + '_'

    # 返回 WildFunction 对象的名称后加下划线的字符串表示
    def _print_WildFunction(self, expr):
        return expr.name + '_'

    # 返回 WildDot 对象的名称的字符串表示
    def _print_WildDot(self, expr):
        return expr.name

    # 返回 WildPlus 对象的名称的字符串表示
    def _print_WildPlus(self, expr):
        return expr.name

    # 返回 WildStar 对象的名称的字符串表示
    def _print_WildStar(self, expr):
        return expr.name

    # 返回表示整数 0 的字符串表示，根据设置输出 SymPy 整数或者普通整数
    def _print_Zero(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(0)"
        return self._print_Integer(Integer(0))

    # 返回表示 DMP 对象的字符串表示，包括类名、系数表示和定义域表示
    def _print_DMP(self, p):
        cls = p.__class__.__name__
        rep = self._print(p.to_list())
        dom = self._print(p.dom)

        return "%s(%s, %s)" % (cls, rep, dom)

    # 返回表示 DMF 对象的字符串表示，包括类名、分子表示、分母表示和定义域表示
    def _print_DMF(self, expr):
        cls = expr.__class__.__name__
        num = self._print(expr.num)
        den = self._print(expr.den)
        dom = self._print(expr.dom)

        return "%s(%s, %s, %s)" % (cls, num, den, dom)

    # 返回表示 Object 对象的字符串表示，包括对象名称
    def _print_Object(self, obj):
        return 'Object("%s")' % obj.name

    # 返回表示 IdentityMorphism 对象的字符串表示，包括定义域
    def _print_IdentityMorphism(self, morphism):
        return 'IdentityMorphism(%s)' % morphism.domain

    # 返回表示 NamedMorphism 对象的字符串表示，包括定义域、值域和名称
    def _print_NamedMorphism(self, morphism):
        return 'NamedMorphism(%s, %s, "%s")' % \
               (morphism.domain, morphism.codomain, morphism.name)

    # 返回表示 Category 对象的字符串表示，包括分类名称
    def _print_Category(self, category):
        return 'Category("%s")' % category.name

    # 返回表示 Manifold 对象的字符串表示，包括流形名称
    def _print_Manifold(self, manifold):
        return manifold.name.name

    # 返回表示 Patch 对象的字符串表示，包括补丁名称
    def _print_Patch(self, patch):
        return patch.name.name

    # 返回表示 CoordSystem 对象的字符串表示，包括坐标系名称
    def _print_CoordSystem(self, coords):
        return coords.name.name

    # 返回表示 BaseScalarField 对象的字符串表示，包括基本标量场的符号名称
    def _print_BaseScalarField(self, field):
        return field._coord_sys.symbols[field._index].name

    # 返回表示 BaseVectorField 对象的字符串表示，包括基本矢量场的符号名称
    def _print_BaseVectorField(self, field):
        return 'e_%s' % field._coord_sys.symbols[field._index].name

    # 返回表示 Differential 对象的字符串表示，包括形式场名称或符号
    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            return 'd%s' % field._coord_sys.symbols[field._index].name
        else:
            return 'd(%s)' % self._print(field)

    # 返回表示 Tr 对象的字符串表示，目前尚未处理指数情况
    def _print_Tr(self, expr):
        #TODO : Handle indices
        return "%s(%s)" % ("Tr", self._print(expr.args[0]))

    # 返回表示字符串对象的字符串表示
    def _print_Str(self, s):
        return self._print(s.name)
    # 定义一个方法 `_print_AppliedBinaryRelation`，用于打印应用的二元关系表达式
    def _print_AppliedBinaryRelation(self, expr):
        # 获取表达式中的二元关系函数
        rel = expr.function
        # 返回格式化的字符串，包括函数名和左右两个操作数的打印结果
        return '%s(%s, %s)' % (self._print(rel),
                               self._print(expr.lhs),
                               self._print(expr.rhs))
# 使用修饰器将 print_function 应用于 StrPrinter 类，使得函数 sstr 被 StrPrinter 类打印
@print_function(StrPrinter)
def sstr(expr, **settings):
    """Returns the expression as a string.

    For large expressions where speed is a concern, use the setting
    order='none'. If abbrev=True setting is used then units are printed in
    abbreviated form.

    Examples
    ========

    >>> from sympy import symbols, Eq, sstr
    >>> a, b = symbols('a b')
    >>> sstr(Eq(a + b, 0))
    'Eq(a + b, 0)'
    """

    # 使用给定的设置创建 StrPrinter 对象
    p = StrPrinter(settings)
    # 使用 StrPrinter 对象将表达式打印成字符串
    s = p.doprint(expr)

    return s


class StrReprPrinter(StrPrinter):
    """(internal) -- see sstrrepr"""

    # 重载 _print_str 方法，返回字符串的 repr 形式
    def _print_str(self, s):
        return repr(s)

    # 重载 _print_Str 方法，格式化打印 Str 对象
    def _print_Str(self, s):
        # 在此处 Str 不需要像 str 一样打印
        return "%s(%s)" % (s.__class__.__name__, self._print(s.name))


# 使用修饰器将 print_function 应用于 StrReprPrinter 类，使得函数 sstrrepr 被 StrReprPrinter 类打印
@print_function(StrReprPrinter)
def sstrrepr(expr, **settings):
    """return expr in mixed str/repr form

       i.e. strings are returned in repr form with quotes, and everything else
       is returned in str form.

       This function could be useful for hooking into sys.displayhook
    """

    # 使用给定的设置创建 StrReprPrinter 对象
    p = StrReprPrinter(settings)
    # 使用 StrReprPrinter 对象将表达式以混合的 str/repr 形式打印
    s = p.doprint(expr)

    return s
```