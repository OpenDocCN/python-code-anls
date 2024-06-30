# `D:\src\scipysrc\sympy\sympy\printing\mathml.py`

```
"""
A MathML printer.
"""

from __future__ import annotations  # 允许在类定义中使用类型注解
from typing import Any  # 导入 Any 类型用于类型注解

from sympy.core.mul import Mul  # 导入 Mul 类用于乘法表达式
from sympy.core.singleton import S  # 导入 S 类用于单例符号
from sympy.core.sorting import default_sort_key  # 导入排序函数 default_sort_key
from sympy.core.sympify import sympify  # 导入 sympify 函数用于将字符串转换为 SymPy 表达式
from sympy.printing.conventions import split_super_sub, requires_partial  # 导入打印约定相关函数
from sympy.printing.precedence import \
    precedence_traditional, PRECEDENCE, PRECEDENCE_TRADITIONAL  # 导入打印优先级相关常量
from sympy.printing.pretty.pretty_symbology import greek_unicode  # 导入希腊字母打印相关函数
from sympy.printing.printer import Printer, print_function  # 导入打印器和打印函数

from mpmath.libmp import prec_to_dps, repr_dps, to_str as mlib_to_str  # 导入数学库函数

class MathMLPrinterBase(Printer):
    """Contains common code required for MathMLContentPrinter and
    MathMLPresentationPrinter.
    """

    _default_settings: dict[str, Any] = {  # 默认设置字典，指定各种打印选项的默认值
        "order": None,
        "encoding": "utf-8",
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_symbol_style": "plain",
        "mul_symbol": None,
        "root_notation": True,
        "symbol_names": {},
        "mul_symbol_mathml_numbers": '&#xB7;',  # 乘号的 MathML 实体编码
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)  # 调用 Printer 的初始化方法
        from xml.dom.minidom import Document, Text  # 导入 XML DOM 相关模块

        self.dom = Document()  # 创建 XML 文档对象

        # Workaround to allow strings to remain unescaped
        # Based on
        # https://stackoverflow.com/questions/38015864/python-xml-dom-minidom-\
        #                              please-dont-escape-my-strings/38041194
        class RawText(Text):
            def writexml(self, writer, indent='', addindent='', newl=''):
                if self.data:
                    writer.write('{}{}{}'.format(indent, self.data, newl))

        def createRawTextNode(data):
            r = RawText()
            r.data = data
            r.ownerDocument = self.dom
            return r

        self.dom.createTextNode = createRawTextNode  # 重写 createTextNode 方法以支持原始文本节点

    def doprint(self, expr):
        """
        Prints the expression as MathML.
        """
        mathML = Printer._print(self, expr)  # 调用 Printer 类的 _print 方法打印表达式
        unistr = mathML.toxml()  # 将 MathML 对象转换为 XML 字符串
        xmlbstr = unistr.encode('ascii', 'xmlcharrefreplace')  # 将 XML 字符串转换为 ASCII 编码，处理非法字符
        res = xmlbstr.decode()  # 解码得到最终输出结果
        return res  # 返回打印的结果字符串


class MathMLContentPrinter(MathMLPrinterBase):
    """Prints an expression to the Content MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter4.html
    """
    printmethod = "_mathml_content"  # 设置打印方法名称为 "_mathml_content"
    # 定义一个方法，返回表达式的 MathML 标签
    def mathml_tag(self, e):
        # 定义一个字典，用于将表达式类名映射到对应的 MathML 标签
        translate = {
            'Add': 'plus',
            'Mul': 'times',
            'Derivative': 'diff',
            'Number': 'cn',
            'int': 'cn',
            'Pow': 'power',
            'Max': 'max',
            'Min': 'min',
            'Abs': 'abs',
            'And': 'and',
            'Or': 'or',
            'Xor': 'xor',
            'Not': 'not',
            'Implies': 'implies',
            'Symbol': 'ci',
            'MatrixSymbol': 'ci',
            'RandomSymbol': 'ci',
            'Integral': 'int',
            'Sum': 'sum',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'cot': 'cot',
            'csc': 'csc',
            'sec': 'sec',
            'sinh': 'sinh',
            'cosh': 'cosh',
            'tanh': 'tanh',
            'coth': 'coth',
            'csch': 'csch',
            'sech': 'sech',
            'asin': 'arcsin',
            'asinh': 'arcsinh',
            'acos': 'arccos',
            'acosh': 'arccosh',
            'atan': 'arctan',
            'atanh': 'arctanh',
            'atan2': 'arctan',
            'acot': 'arccot',
            'acoth': 'arccoth',
            'asec': 'arcsec',
            'asech': 'arcsech',
            'acsc': 'arccsc',
            'acsch': 'arccsch',
            'log': 'ln',
            'Equality': 'eq',
            'Unequality': 'neq',
            'GreaterThan': 'geq',
            'LessThan': 'leq',
            'StrictGreaterThan': 'gt',
            'StrictLessThan': 'lt',
            'Union': 'union',
            'Intersection': 'intersect',
        }

        # 遍历表达式类的方法解析顺序（Method Resolution Order）
        for cls in e.__class__.__mro__:
            # 获取类名
            n = cls.__name__
            # 如果类名在字典中，则返回对应的 MathML 标签
            if n in translate:
                return translate[n]
        
        # 如果未在字典中找到对应的类名，则返回类名的小写形式作为默认标签
        # 这种情况通常表示表达式类未在预定义的映射中找到对应的标签
        n = e.__class__.__name__
        return n.lower()
    # 定义一个方法来打印乘法表达式，接受一个表达式参数 expr
    def _print_Mul(self, expr):

        # 如果表达式能够提取负号
        if expr.could_extract_minus_sign():
            # 创建一个名为 x 的 XML 元素 'apply'
            x = self.dom.createElement('apply')
            # 在 'apply' 元素下创建一个 'minus' 元素
            x.appendChild(self.dom.createElement('minus'))
            # 将当前方法递归调用，并将参数取反后作为子元素添加到 'minus' 元素下
            x.appendChild(self._print_Mul(-expr))
            return x

        # 导入 sympy 库中的 fraction 函数，用于获取表达式的分子和分母
        from sympy.simplify import fraction
        # 调用 fraction 函数，将表达式分解为分子 numer 和分母 denom
        numer, denom = fraction(expr)

        # 如果分母不是 1
        if denom is not S.One:
            # 创建一个名为 x 的 XML 元素 'apply'
            x = self.dom.createElement('apply')
            # 在 'apply' 元素下创建一个 'divide' 元素
            x.appendChild(self.dom.createElement('divide'))
            # 将分子和分母分别作为子元素添加到 'divide' 元素下
            x.appendChild(self._print(numer))
            x.appendChild(self._print(denom))
            return x

        # 将表达式分解为系数 coeff 和基础项 terms
        coeff, terms = expr.as_coeff_mul()
        # 如果系数是 1 并且基础项 terms 只有一个
        if coeff is S.One and len(terms) == 1:
            # 直接打印基础项 terms[0]
            return self._print(terms[0])

        # 如果不是 'old' 排序方式
        if self.order != 'old':
            # 对 terms 应用有序因子排序
            terms = Mul._from_args(terms).as_ordered_factors()

        # 创建一个名为 x 的 XML 元素 'apply'
        x = self.dom.createElement('apply')
        # 在 'apply' 元素下创建一个 'times' 元素
        x.appendChild(self.dom.createElement('times'))
        # 如果系数不等于 1，则将系数作为子元素添加到 'times' 元素下
        if coeff != 1:
            x.appendChild(self._print(coeff))
        # 遍历基础项 terms，将每一项作为子元素添加到 'times' 元素下
        for term in terms:
            x.appendChild(self._print(term))
        return x

    # 定义一个方法来打印加法表达式，接受一个表达式参数 expr 和一个排序参数 order
    def _print_Add(self, expr, order=None):
        # 使用 _as_ordered_terms 方法对表达式 expr 进行排序，并将结果保存到 args 中
        args = self._as_ordered_terms(expr, order=order)
        # 将第一个排序后的表达式作为 lastProcessed
        lastProcessed = self._print(args[0])
        # 初始化一个空列表 plusNodes
        plusNodes = []
        # 遍历 args 中的每一个表达式 arg（从第二个开始）
        for arg in args[1:]:
            # 如果表达式 arg 能够提取负号
            if arg.could_extract_minus_sign():
                # 创建一个名为 x 的 XML 元素 'apply'
                x = self.dom.createElement('apply')
                # 在 'apply' 元素下创建一个 'minus' 元素
                x.appendChild(self.dom.createElement('minus'))
                # 将上一次处理的表达式作为 'minus' 元素的子元素
                x.appendChild(lastProcessed)
                # 将参数取反后的表达式作为 'minus' 元素的子元素
                x.appendChild(self._print(-arg))
                # 将处理后的结果反转，因为现在表达式被减去了
                lastProcessed = x
                # 如果当前表达式 arg 是 args 的最后一个表达式，则将 lastProcessed 添加到 plusNodes 中
                if arg == args[-1]:
                    plusNodes.append(lastProcessed)
            else:
                # 将上一次处理的表达式添加到 plusNodes 中
                plusNodes.append(lastProcessed)
                # 处理当前表达式 arg，并将结果保存到 lastProcessed
                lastProcessed = self._print(arg)
                # 如果当前表达式 arg 是 args 的最后一个表达式，则将处理后的结果添加到 plusNodes 中
                if arg == args[-1]:
                    plusNodes.append(self._print(arg))

        # 如果 plusNodes 中只有一个元素，则直接返回 lastProcessed
        if len(plusNodes) == 1:
            return lastProcessed

        # 创建一个名为 x 的 XML 元素 'apply'
        x = self.dom.createElement('apply')
        # 在 'apply' 元素下创建一个 'plus' 元素
        x.appendChild(self.dom.createElement('plus'))
        # 将 plusNodes 中的每一个元素依次添加到 'plus' 元素下
        while plusNodes:
            x.appendChild(plusNodes.pop(0))
        return x
    def _print_Piecewise(self, expr):
        # 检查 Piecewise 表达式的最后一个条件是否为 True，否则可能导致生成的函数无法返回结果
        if expr.args[-1].cond != True:
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        # 创建根节点 <piecewise>
        root = self.dom.createElement('piecewise')
        # 遍历 Piecewise 表达式的每个分段
        for i, (e, c) in enumerate(expr.args):
            # 如果是最后一个分段且条件为 True，创建 <otherwise> 节点
            if i == len(expr.args) - 1 and c == True:
                piece = self.dom.createElement('otherwise')
                piece.appendChild(self._print(e))  # 添加表达式内容
            else:
                # 否则创建 <piece> 节点
                piece = self.dom.createElement('piece')
                piece.appendChild(self._print(e))  # 添加表达式内容
                piece.appendChild(self._print(c))  # 添加条件内容
            root.appendChild(piece)  # 将分段节点添加到根节点下
        return root  # 返回整个 <piecewise> 结构

    def _print_MatrixBase(self, m):
        # 创建 <matrix> 节点
        x = self.dom.createElement('matrix')
        # 遍历矩阵的行
        for i in range(m.rows):
            x_r = self.dom.createElement('matrixrow')  # 创建 <matrixrow> 节点
            # 遍历矩阵的列
            for j in range(m.cols):
                x_r.appendChild(self._print(m[i, j]))  # 添加每个元素的打印内容
            x.appendChild(x_r)  # 将行节点添加到矩阵节点下
        return x  # 返回整个矩阵节点

    def _print_Rational(self, e):
        # 如果分母为1，创建 <cn> 节点表示整数
        if e.q == 1:
            x = self.dom.createElement('cn')
            x.appendChild(self.dom.createTextNode(str(e.p)))
            return x
        # 否则创建 <apply><divide> 节点表示有理数的分数形式
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('divide'))
        # 分子
        xnum = self.dom.createElement('cn')
        xnum.appendChild(self.dom.createTextNode(str(e.p)))
        # 分母
        xdenom = self.dom.createElement('cn')
        xdenom.appendChild(self.dom.createTextNode(str(e.q)))
        x.appendChild(xnum)
        x.appendChild(xdenom)
        return x

    def _print_Limit(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))  # 添加数学标记

        x_1 = self.dom.createElement('bvar')
        x_2 = self.dom.createElement('lowlimit')
        x_1.appendChild(self._print(e.args[1]))  # 添加变量
        x_2.appendChild(self._print(e.args[2]))  # 添加下限

        x.appendChild(x_1)
        x.appendChild(x_2)
        x.appendChild(self._print(e.args[0]))  # 添加表达式
        return x

    def _print_ImaginaryUnit(self, e):
        # 创建 <imaginaryi> 节点表示虚数单位
        return self.dom.createElement('imaginaryi')

    def _print_EulerGamma(self, e):
        # 创建 <eulergamma> 节点表示欧拉常数 γ
        return self.dom.createElement('eulergamma')

    def _print_GoldenRatio(self, e):
        """We use unicode #x3c6 for Greek letter phi as defined here
        https://www.w3.org/2003/entities/2007doc/isogrk1.html"""
        # 创建 <cn> 节点表示黄金比例 φ
        x = self.dom.createElement('cn')
        x.appendChild(self.dom.createTextNode("\N{GREEK SMALL LETTER PHI}"))
        return x
    # 创建一个 XML 元素 'exponentiale' 并返回
    def _print_Exp1(self, e):
        return self.dom.createElement('exponentiale')

    # 创建一个 XML 元素 'pi' 并返回
    def _print_Pi(self, e):
        return self.dom.createElement('pi')

    # 创建一个 XML 元素 'infinity' 并返回
    def _print_Infinity(self, e):
        return self.dom.createElement('infinity')

    # 创建一个 XML 元素 'notanumber' 并返回
    def _print_NaN(self, e):
        return self.dom.createElement('notanumber')

    # 创建一个 XML 元素 'emptyset' 并返回
    def _print_EmptySet(self, e):
        return self.dom.createElement('emptyset')

    # 创建一个 XML 元素 'true' 并返回
    def _print_BooleanTrue(self, e):
        return self.dom.createElement('true')

    # 创建一个 XML 元素 'false' 并返回
    def _print_BooleanFalse(self, e):
        return self.dom.createElement('false')

    # 创建表示负无穷的 XML 结构并返回
    def _print_NegativeInfinity(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('minus'))  # 添加减号元素
        x.appendChild(self.dom.createElement('infinity'))  # 添加无穷大元素
        return x

    # 创建表示积分的 XML 结构并返回
    def _print_Integral(self, e):
        # 定义内部函数 lime_recur 处理积分的边界条件
        def lime_recur(limits):
            x = self.dom.createElement('apply')  # 创建 apply 元素
            x.appendChild(self.dom.createElement(self.mathml_tag(e)))  # 添加数学标签元素

            bvar_elem = self.dom.createElement('bvar')  # 创建积分变量元素
            bvar_elem.appendChild(self._print(limits[0][0]))  # 添加积分变量的打印表示
            x.appendChild(bvar_elem)

            if len(limits[0]) == 3:
                low_elem = self.dom.createElement('lowlimit')  # 创建下限元素
                low_elem.appendChild(self._print(limits[0][1]))  # 添加下限的打印表示
                x.appendChild(low_elem)
                up_elem = self.dom.createElement('uplimit')  # 创建上限元素
                up_elem.appendChild(self._print(limits[0][2]))  # 添加上限的打印表示
                x.appendChild(up_elem)
            if len(limits[0]) == 2:
                up_elem = self.dom.createElement('uplimit')  # 创建上限元素
                up_elem.appendChild(self._print(limits[0][1]))  # 添加上限的打印表示
                x.appendChild(up_elem)

            if len(limits) == 1:
                x.appendChild(self._print(e.function))  # 添加积分函数的打印表示
            else:
                x.appendChild(lime_recur(limits[1:]))  # 递归处理更多的积分限制

            return x

        limits = list(e.limits)
        limits.reverse()  # 反转限制列表以便正确处理积分上下限
        return lime_recur(limits)  # 调用递归函数处理积分的所有限制条件

    # 创建表示求和的 XML 结构并返回，与积分共享打印方法
    def _print_Sum(self, e):
        # 由于求和与积分具有相同的内部表示，因此可以共享打印方法
        return self._print_Integral(e)
    # 定义一个方法用于打印符号的 MathML 表示
    def _print_Symbol(self, sym):
        # 创建一个 XML 元素，其标签由 sym 参数决定
        ci = self.dom.createElement(self.mathml_tag(sym))

        # 定义一个内部方法，用于将项目连接起来形成一个 mrow 元素
        def join(items):
            # 如果项目数大于1，则创建一个 mrow 元素并逐个添加项目
            if len(items) > 1:
                mrow = self.dom.createElement('mml:mrow')
                for i, item in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mml:mo')
                        mo.appendChild(self.dom.createTextNode(" "))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mml:mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            # 如果只有一个项目，则直接创建一个 mi 元素
            else:
                mi = self.dom.createElement('mml:mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi

        # 将符号名称、上标和下标分离
        name, supers, subs = split_super_sub(sym.name)
        # 将符号名称翻译成对应的 Unicode 字符
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        # 创建一个 mi 元素表示符号的名称
        mname = self.dom.createElement('mml:mi')
        mname.appendChild(self.dom.createTextNode(name))
        
        # 根据上标和下标的情况，构建不同的 MathML 结构
        if not supers:
            if not subs:
                ci.appendChild(self.dom.createTextNode(name))
            else:
                msub = self.dom.createElement('mml:msub')
                msub.appendChild(mname)
                msub.appendChild(join(subs))
                ci.appendChild(msub)
        else:
            if not subs:
                msup = self.dom.createElement('mml:msup')
                msup.appendChild(mname)
                msup.appendChild(join(supers))
                ci.appendChild(msup)
            else:
                msubsup = self.dom.createElement('mml:msubsup')
                msubsup.appendChild(mname)
                msubsup.appendChild(join(subs))
                msubsup.appendChild(join(supers))
                ci.appendChild(msubsup)

        # 返回创建好的 XML 元素
        return ci

    # 将 _print_MatrixSymbol 和 _print_RandomSymbol 方法与 _print_Symbol 方法绑定，实现相同的打印功能
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol
    # 定义一个方法来打印指数运算表达式
    def _print_Pow(self, e):
        # 如果设置为根号表示且指数是整数的倒数
        if (self._settings['root_notation'] and e.exp.is_Rational
                and e.exp.p == 1):
            # 创建一个 XML 元素 'apply'
            x = self.dom.createElement('apply')
            # 添加 'root' 元素作为子元素
            x.appendChild(self.dom.createElement('root'))
            # 如果指数不是 2，添加 'degree' 元素及其内容
            if e.exp.q != 2:
                xmldeg = self.dom.createElement('degree')
                xmlcn = self.dom.createElement('cn')
                xmlcn.appendChild(self.dom.createTextNode(str(e.exp.q)))
                xmldeg.appendChild(xmlcn)
                x.appendChild(xmldeg)
            # 添加基数和指数的打印结果作为子元素
            x.appendChild(self._print(e.base))
            return x

        # 创建一个 XML 元素 'apply'
        x = self.dom.createElement('apply')
        # 根据表达式类型创建相应的元素，并添加基数和指数的打印结果作为子元素
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        x.appendChild(self._print(e.base))
        x.appendChild(self._print(e.exp))
        return x

    # 定义一个方法来打印数值表达式
    def _print_Number(self, e):
        # 创建一个 XML 元素，元素名称为数值表达式的标签名
        x = self.dom.createElement(self.mathml_tag(e))
        # 添加数值表达式的文本内容作为子节点
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    # 定义一个方法来打印浮点数表达式
    def _print_Float(self, e):
        # 创建一个 XML 元素，元素名称为浮点数表达式的标签名
        x = self.dom.createElement(self.mathml_tag(e))
        # 将浮点数的字符串表示添加为文本节点的内容
        repr_e = mlib_to_str(e._mpf_, repr_dps(e._prec))
        x.appendChild(self.dom.createTextNode(repr_e))
        return x

    # 定义一个方法来打印导数表达式
    def _print_Derivative(self, e):
        # 创建一个 XML 元素 'apply'
        x = self.dom.createElement('apply')
        # 确定使用的微分符号类型
        diff_symbol = self.mathml_tag(e)
        if requires_partial(e.expr):
            diff_symbol = 'partialdiff'
        x.appendChild(self.dom.createElement(diff_symbol))
        # 创建 'bvar' 元素用于表示变量
        x_1 = self.dom.createElement('bvar')

        # 遍历变量及其对应的次数并添加到 'bvar' 元素中
        for sym, times in reversed(e.variable_count):
            x_1.appendChild(self._print(sym))
            if times > 1:
                # 如果次数大于 1，则添加 'degree' 元素并包含次数的打印结果
                degree = self.dom.createElement('degree')
                degree.appendChild(self._print(sympify(times)))
                x_1.appendChild(degree)

        # 将 'bvar' 元素和表达式的打印结果作为子元素添加到 'apply' 元素中
        x.appendChild(x_1)
        x.appendChild(self._print(e.expr))
        return x

    # 定义一个方法来打印函数表达式
    def _print_Function(self, e):
        # 创建一个 XML 元素 'apply'
        x = self.dom.createElement("apply")
        # 添加函数名作为子元素
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        # 遍历函数的参数并将每个参数的打印结果作为子元素添加到 'apply' 元素中
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    # 定义一个方法来打印基础表达式
    def _print_Basic(self, e):
        # 创建一个 XML 元素，元素名称为基础表达式的标签名
        x = self.dom.createElement(self.mathml_tag(e))
        # 遍历基础表达式的参数并将每个参数的打印结果作为子元素添加到 XML 元素中
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    # 定义一个方法来打印关联操作表达式
    def _print_AssocOp(self, e):
        # 创建一个 XML 元素 'apply'
        x = self.dom.createElement('apply')
        # 添加关联操作的标签名作为子元素
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        # 遍历关联操作的参数并将每个参数的打印结果作为子元素添加到 'apply' 元素中
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    # 定义一个方法来打印关系表达式
    def _print_Relational(self, e):
        # 创建一个 XML 元素 'apply'
        x = self.dom.createElement('apply')
        # 添加关系运算符的标签名作为子元素
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        # 添加左侧和右侧表达式的打印结果作为子元素
        x.appendChild(self._print(e.lhs))
        x.appendChild(self._print(e.rhs))
        return x
    # 创建一个名为 _print_list 的方法，用于生成 MathML 中的 <list> 元素
    def _print_list(self, seq):
        """MathML reference for the <list> element:
        https://www.w3.org/TR/MathML2/chapter4.html#contm.list"""
        # 创建一个名为 dom_element 的 XML 元素 <list>
        dom_element = self.dom.createElement('list')
        # 遍历输入的序列 seq 中的每个项，将其转换成对应的 MathML 元素，并添加到 dom_element 中
        for item in seq:
            dom_element.appendChild(self._print(item))
        # 返回生成的 <list> 元素
        return dom_element

    # 创建一个名为 _print_int 的方法，用于生成表示整数的 MathML 元素
    def _print_int(self, p):
        # 根据整数 p 生成对应的 MathML 标签，并创建该标签的 XML 元素
        dom_element = self.dom.createElement(self.mathml_tag(p))
        # 在 dom_element 中添加表示整数 p 的文本节点
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        # 返回生成的整数的 MathML 元素
        return dom_element

    # 将 _print_Implies、_print_Not、_print_Xor 三个方法指向 _print_AssocOp 方法
    _print_Implies = _print_AssocOp
    _print_Not = _print_AssocOp
    _print_Xor = _print_AssocOp

    # 创建一个名为 _print_FiniteSet 的方法，用于生成 MathML 中的 <set> 元素
    def _print_FiniteSet(self, e):
        # 创建一个名为 x 的 <set> 元素
        x = self.dom.createElement('set')
        # 遍历输入的 e 中的每个参数 arg，并将其转换成对应的 MathML 元素，并添加到 x 中
        for arg in e.args:
            x.appendChild(self._print(arg))
        # 返回生成的 <set> 元素
        return x

    # 创建一个名为 _print_Complement 的方法，用于生成 MathML 中的补集操作元素
    def _print_Complement(self, e):
        # 创建一个名为 x 的 <apply> 元素
        x = self.dom.createElement('apply')
        # 在 x 中添加表示集合差集的 <setdiff> 元素
        x.appendChild(self.dom.createElement('setdiff'))
        # 遍历输入的 e 中的每个参数 arg，并将其转换成对应的 MathML 元素，并添加到 x 中
        for arg in e.args:
            x.appendChild(self._print(arg))
        # 返回生成的 <apply> 元素
        return x

    # 创建一个名为 _print_ProductSet 的方法，用于生成 MathML 中的笛卡尔积元素
    def _print_ProductSet(self, e):
        # 创建一个名为 x 的 <apply> 元素
        x = self.dom.createElement('apply')
        # 在 x 中添加表示笛卡尔积的 <cartesianproduct> 元素
        x.appendChild(self.dom.createElement('cartesianproduct'))
        # 遍历输入的 e 中的每个参数 arg，并将其转换成对应的 MathML 元素，并添加到 x 中
        for arg in e.args:
            x.appendChild(self._print(arg))
        # 返回生成的 <apply> 元素
        return x

    # 创建一个名为 _print_Lambda 的方法，用于生成 MathML 中的 lambda 元素
    def _print_Lambda(self, e):
        # MathML 参考链接：https://www.w3.org/TR/MathML2/chapter4.html#id.4.2.1.7
        # 创建一个名为 x 的 XML 元素，标签由 mathml_tag 方法根据 e 的内容确定
        x = self.dom.createElement(self.mathml_tag(e))
        # 遍历 e 的签名（参数列表），为每个参数创建 <bvar> 元素，并将其添加到 x 中
        for arg in e.signature:
            x_1 = self.dom.createElement('bvar')
            x_1.appendChild(self._print(arg))
            x.appendChild(x_1)
        # 将 e 的表达式部分转换成对应的 MathML 元素，并添加到 x 中
        x.appendChild(self._print(e.expr))
        # 返回生成的 XML 元素 x
        return x

    # XXX 对称差（Symmetric difference）不支持 MathML 内容打印器。
class MathMLPresentationPrinter(MathMLPrinterBase):
    """Prints an expression to the Presentation MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter3.html
    """
    # 设置默认的打印方法为 Presentation MathML
    printmethod = "_mathml_presentation"

    def mathml_tag(self, e):
        """Returns the MathML tag for an expression."""
        # 定义数学表达式类型到 MathML 标签的映射表
        translate = {
            'Number': 'mn',  # 数字
            'Limit': '&#x2192;',  # 极限
            'Derivative': '&dd;',  # 导数
            'int': 'mn',  # 积分
            'Symbol': 'mi',  # 符号
            'Integral': '&int;',  # 积分符号
            'Sum': '&#x2211;',  # 求和
            'sin': 'sin',  # 正弦函数
            'cos': 'cos',  # 余弦函数
            'tan': 'tan',  # 正切函数
            'cot': 'cot',  # 余切函数
            'asin': 'arcsin',  # 反正弦函数
            'asinh': 'arcsinh',  # 反双曲正弦函数
            'acos': 'arccos',  # 反余弦函数
            'acosh': 'arccosh',  # 反双曲余弦函数
            'atan': 'arctan',  # 反正切函数
            'atanh': 'arctanh',  # 反双曲正切函数
            'acot': 'arccot',  # 反余切函数
            'atan2': 'arctan',  # 反正切函数 (两参数版本)
            'Equality': '=',  # 等于号
            'Unequality': '&#x2260;',  # 不等于号
            'GreaterThan': '&#x2265;',  # 大于等于号
            'LessThan': '&#x2264;',  # 小于等于号
            'StrictGreaterThan': '>',  # 严格大于号
            'StrictLessThan': '<',  # 严格小于号
            'lerchphi': '&#x3A6;',  # 莱让符号
            'zeta': '&#x3B6;',  # Zeta 函数
            'dirichlet_eta': '&#x3B7;',  # 狄利克雷 eta 函数
            'elliptic_k': '&#x39A;',  # 椭圆积分第一类
            'lowergamma': '&#x3B3;',  # 小 Gamma 函数
            'uppergamma': '&#x393;',  # 大 Gamma 函数
            'gamma': '&#x393;',  # Gamma 函数
            'totient': '&#x3D5;',  # 欧拉函数
            'reduced_totient': '&#x3BB;',  # 简化欧拉函数
            'primenu': '&#x3BD;',  # 素数计数函数 nu
            'primeomega': '&#x3A9;',  # 素数计数函数 omega
            'fresnels': 'S',  # Fresnel S 函数
            'fresnelc': 'C',  # Fresnel C 函数
            'LambertW': 'W',  # Lambert W 函数
            'Heaviside': '&#x398;',  # 海维赛德函数
            'BooleanTrue': 'True',  # 布尔真值
            'BooleanFalse': 'False',  # 布尔假值
            'NoneType': 'None',  # 空类型
            'mathieus': 'S',  # Mathieu S 函数
            'mathieuc': 'C',  # Mathieu C 函数
            'mathieusprime': 'S&#x2032;',  # Mathieu S' 函数
            'mathieucprime': 'C&#x2032;',  # Mathieu C' 函数
            'Lambda': 'lambda',  # Lambda 函数
        }

        # 选择乘号的符号表示方式
        def mul_symbol_selection():
            if (self._settings["mul_symbol"] is None or
                    self._settings["mul_symbol"] == 'None'):
                return '&InvisibleTimes;'  # 不可见乘号
            elif self._settings["mul_symbol"] == 'times':
                return '&#xD7;'  # 真正的乘号
            elif self._settings["mul_symbol"] == 'dot':
                return '&#xB7;'  # 点乘号
            elif self._settings["mul_symbol"] == 'ldot':
                return '&#x2024;'  # 长点乘号
            elif not isinstance(self._settings["mul_symbol"], str):
                raise TypeError
            else:
                return self._settings["mul_symbol"]

        # 遍历表达式类及其父类，返回找到的 MathML 标签
        for cls in e.__class__.__mro__:
            n = cls.__name__
            if n in translate:
                return translate[n]
        
        # 如果在 MRO 中未找到对应的类名，则检查是否为乘法运算符号
        if e.__class__.__name__ == "Mul":
            return mul_symbol_selection()

        # 如果都未匹配，则返回类名的小写形式作为标签
        n = e.__class__.__name__
        return n.lower()
    # 定义一个方法用于将表达式包围在括号中，根据给定的运算级别和是否严格判断是否添加括号
    def parenthesize(self, item, level, strict=False):
        # 获取给定项的传统优先级值
        prec_val = precedence_traditional(item)
        # 如果给定项的优先级低于指定级别，或者（非严格模式且优先级等于指定级别），则添加括号
        if (prec_val < level) or ((not strict) and prec_val <= level):
            # 创建一个名为 'mfenced' 的 XML 元素
            brac = self.dom.createElement('mfenced')
            # 将 item 的打印结果作为子节点添加到 'mfenced' 元素中
            brac.appendChild(self._print(item))
            return brac
        else:
            # 否则直接打印 item
            return self._print(item)

    # 定义一个方法用于打印乘法表达式的 MathML 表示
    def _print_Mul(self, expr):

        # 定义一个内部函数，用于处理乘法表达式的打印
        def multiply(expr, mrow):
            # 导入 fraction 函数用于获取表达式的分子和分母
            from sympy.simplify import fraction
            numer, denom = fraction(expr)
            # 如果分母不为 1，则创建 'mfrac' 元素表示分数
            if denom is not S.One:
                frac = self.dom.createElement('mfrac')
                # 如果设置中允许简化短分数且表达式长度小于 7，则设置 'bevelled' 属性为 'true'
                if self._settings["fold_short_frac"] and len(str(expr)) < 7:
                    frac.setAttribute('bevelled', 'true')
                xnum = self._print(numer)
                xden = self._print(denom)
                frac.appendChild(xnum)
                frac.appendChild(xden)
                mrow.appendChild(frac)
                return mrow

            # 将表达式分解为系数和项
            coeff, terms = expr.as_coeff_mul()
            # 如果系数为 1 且项数为 1，则直接打印这一项
            if coeff is S.One and len(terms) == 1:
                mrow.appendChild(self._print(terms[0]))
                return mrow
            # 如果排序方式不是 'old'，则按顺序处理因子
            if self.order != 'old':
                terms = Mul._from_args(terms).as_ordered_factors()

            # 如果系数不为 1，则打印系数并添加乘号 'mo' 元素
            if coeff != 1:
                x = self._print(coeff)
                y = self.dom.createElement('mo')
                y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                mrow.appendChild(x)
                mrow.appendChild(y)
            # 遍历每个项并打印，用 'mo' 元素作为分隔符
            for term in terms:
                mrow.appendChild(self.parenthesize(term, PRECEDENCE['Mul']))
                if not term == terms[-1]:
                    y = self.dom.createElement('mo')
                    y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                    mrow.appendChild(y)
            return mrow

        # 创建 'mrow' 元素作为乘法表达式的根元素
        mrow = self.dom.createElement('mrow')
        # 如果表达式可以提取负号，则添加负号 '-' 并处理其余部分
        if expr.could_extract_minus_sign():
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode('-'))
            mrow.appendChild(x)
            mrow = multiply(-expr, mrow)
        else:
            # 否则直接处理乘法表达式
            mrow = multiply(expr, mrow)

        return mrow

    # 定义一个方法用于打印加法表达式的 MathML 表示
    def _print_Add(self, expr, order=None):
        # 创建 'mrow' 元素作为加法表达式的根元素
        mrow = self.dom.createElement('mrow')
        # 按顺序获取表达式中的每个项
        args = self._as_ordered_terms(expr, order=order)
        # 打印第一个项
        mrow.appendChild(self._print(args[0]))
        # 遍历每个剩余项并打印，用 'mo' 元素作为分隔符
        for arg in args[1:]:
            if arg.could_extract_minus_sign():
                # 如果项可以提取负号，则使用负号 '-' 打印
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('-'))
                y = self._print(-arg)
            else:
                # 否则使用加号 '+' 打印
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('+'))
                y = self._print(arg)
            mrow.appendChild(x)
            mrow.appendChild(y)

        return mrow
    # 创建一个 XML 元素 'mtable'，用于表示数学矩阵
    table = self.dom.createElement('mtable')
    # 遍历矩阵 m 的行
    for i in range(m.rows):
        # 创建 'mtr' 元素，表示矩阵的一行
        x = self.dom.createElement('mtr')
        # 遍历矩阵 m 的列
        for j in range(m.cols):
            # 创建 'mtd' 元素，表示矩阵的一个单元格
            y = self.dom.createElement('mtd')
            # 将 m[i, j] 的打印结果作为子元素添加到 'mtd'
            y.appendChild(self._print(m[i, j]))
            # 将 'mtd' 添加到当前行 'mtr'
            x.appendChild(y)
        # 将当前行 'mtr' 添加到 'mtable'
        table.appendChild(x)
    # 如果设置中的矩阵分隔符为 ''，直接返回整个表格 'mtable'
    if self._settings["mat_delim"] == '':
        return table
    # 如果设置中的矩阵分隔符为 '[', 则创建 'mfenced' 元素用于包裹 'mtable'
    brac = self.dom.createElement('mfenced')
    if self._settings["mat_delim"] == "[":
        # 设置 'mfenced' 元素的开放和关闭分隔符为 '[' 和 ']'
        brac.setAttribute('close', ']')
        brac.setAttribute('open', '[')
    # 将 'mtable' 添加到 'mfenced' 元素
    brac.appendChild(table)
    # 返回包裹了 'mtable' 的 'mfenced' 元素
    return brac

# 处理有理数打印的方法，根据设置选择是否折叠短分数
def _get_printed_Rational(self, e, folded=None):
    # 如果分子 e.p 小于 0，取其相反数作为分子 p
    if e.p < 0:
        p = -e.p
    else:
        p = e.p
    # 创建 'mfrac' 元素表示分数
    x = self.dom.createElement('mfrac')
    # 如果设置要求折叠或者设置中选择折叠短分数
    if folded or self._settings["fold_short_frac"]:
        # 设置 'mfrac' 元素为斜线形式的分数
        x.setAttribute('bevelled', 'true')
    # 添加分子 p 的打印结果作为 'mfrac' 的子元素
    x.appendChild(self._print(p))
    # 添加分母 e.q 的打印结果作为 'mfrac' 的子元素
    x.appendChild(self._print(e.q))
    # 如果分子 e.p 小于 0，创建 'mrow' 元素表示带负号的分数
    if e.p < 0:
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('-'))
        mrow.appendChild(mo)
        mrow.appendChild(x)
        return mrow
    else:
        # 否则直接返回 'mfrac' 元素
        return x

# 打印有理数的方法，如果分母为 1 则直接打印分子，否则调用 _get_printed_Rational 处理
def _print_Rational(self, e):
    if e.q == 1:
        # 如果分母为 1，则直接返回分子的打印结果
        return self._print(e.p)
    # 否则根据设置选择是否折叠短分数打印分数 e
    return self._get_printed_Rational(e, self._settings["fold_short_frac"])

# 打印极限表达式的方法，创建包含极限符号和表达式的 'mrow' 元素
def _print_Limit(self, e):
    mrow = self.dom.createElement('mrow')
    munder = self.dom.createElement('munder')
    mi = self.dom.createElement('mi')
    mi.appendChild(self.dom.createTextNode('lim'))

    x = self.dom.createElement('mrow')
    x_1 = self._print(e.args[1])
    arrow = self.dom.createElement('mo')
    arrow.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
    x_2 = self._print(e.args[2])
    x.appendChild(x_1)
    x.appendChild(arrow)
    x.appendChild(x_2)

    munder.appendChild(mi)
    munder.appendChild(x)
    mrow.appendChild(munder)
    mrow.appendChild(self._print(e.args[0]))

    return mrow

# 打印虚数单位的方法，创建 'mi' 元素表示虚数单位 i
def _print_ImaginaryUnit(self, e):
    x = self.dom.createElement('mi')
    x.appendChild(self.dom.createTextNode('&ImaginaryI;'))
    return x

# 打印黄金比例的方法，创建 'mi' 元素表示黄金比例 φ
def _print_GoldenRatio(self, e):
    x = self.dom.createElement('mi')
    x.appendChild(self.dom.createTextNode('&#x3A6;'))
    return x

# 打印自然常数 e 的方法，创建 'mi' 元素表示自然常数 e
def _print_Exp1(self, e):
    x = self.dom.createElement('mi')
    x.appendChild(self.dom.createTextNode('&ExponentialE;'))
    return x

# 打印圆周率 π 的方法，创建 'mi' 元素表示圆周率 π
def _print_Pi(self, e):
    x = self.dom.createElement('mi')
    x.appendChild(self.dom.createTextNode('&pi;'))
    return x

# 打印无穷大符号的方法，创建 'mi' 元素表示无穷大符号 ∞
def _print_Infinity(self, e):
    x = self.dom.createElement('mi')
    x.appendChild(self.dom.createTextNode('&#x221E;'))
    return x
    # 创建一个 XML 元素 'mrow'
    def _print_NegativeInfinity(self, e):
        mrow = self.dom.createElement('mrow')
        # 创建一个 XML 元素 'mo'，并添加文本节点 '-'
        y = self.dom.createElement('mo')
        y.appendChild(self.dom.createTextNode('-'))
        # 调用 _print_Infinity 方法，将其结果作为子元素添加到 'mrow'
        x = self._print_Infinity(e)
        mrow.appendChild(y)
        mrow.appendChild(x)
        return mrow
    
    # 创建一个 XML 元素 'mi'，并添加文本节点 '&#x210F;'
    def _print_HBar(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x210F;'))
        return x
    
    # 创建一个 XML 元素 'mi'，并添加文本节点 '&#x3B3;'
    def _print_EulerGamma(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x3B3;'))
        return x
    
    # 创建一个 XML 元素 'mi'，并添加文本节点 'TribonacciConstant'
    def _print_TribonacciConstant(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('TribonacciConstant'))
        return x
    
    # 创建一个 XML 元素 'msup'
    def _print_Dagger(self, e):
        msup = self.dom.createElement('msup')
        # 调用 _print 方法处理 e.args[0]，将其结果作为子元素添加到 'msup'
        msup.appendChild(self._print(e.args[0]))
        # 添加文本节点 '&#x2020;' 作为 'msup' 的第二个子元素
        msup.appendChild(self.dom.createTextNode('&#x2020;'))
        return msup
    
    # 创建一个 XML 元素 'mrow'
    def _print_Contains(self, e):
        mrow = self.dom.createElement('mrow')
        # 调用 _print 方法处理 e.args[0]，将其结果作为子元素添加到 'mrow'
        mrow.appendChild(self._print(e.args[0]))
        # 创建一个 XML 元素 'mo'，并添加文本节点 '&#x2208;'
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2208;'))
        # 将 'mo' 添加为 'mrow' 的子元素
        mrow.appendChild(mo)
        # 调用 _print 方法处理 e.args[1]，将其结果作为子元素添加到 'mrow'
        mrow.appendChild(self._print(e.args[1]))
        return mrow
    
    # 创建一个 XML 元素 'mi'，并添加文本节点 '&#x210B;'
    def _print_HilbertSpace(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x210B;'))
        return x
    
    # 创建一个 XML 元素 'msup'
    def _print_ComplexSpace(self, e):
        msup = self.dom.createElement('msup')
        # 添加文本节点 '&#x1D49E;' 作为 'msup' 的第一个子元素
        msup.appendChild(self.dom.createTextNode('&#x1D49E;'))
        # 调用 _print 方法处理 e.args[0]，将其结果作为 'msup' 的第二个子元素添加
        msup.appendChild(self._print(e.args[0]))
        return msup
    
    # 创建一个 XML 元素 'mi'，并添加文本节点 '&#x2131;'
    def _print_FockSpace(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x2131;'))
        return x
    # 定义一个方法来打印积分表达式
    def _print_Integral(self, expr):
        # 定义不同积分符号的 HTML 实体编码
        intsymbols = {1: "&#x222B;", 2: "&#x222C;", 3: "&#x222D;"}

        # 创建一个名为 mrow 的 MathML 元素
        mrow = self.dom.createElement('mrow')

        # 如果积分符号数量不超过三个且每个积分都只有一个限制条件
        if len(expr.limits) <= 3 and all(len(lim) == 1 for lim in expr.limits):
            # 创建一个名为 mo 的 MathML 元素
            mo = self.dom.createElement('mo')
            # 在 mo 元素中添加积分符号对应的文本节点
            mo.appendChild(self.dom.createTextNode(intsymbols[len(expr.limits)]))
            # 将 mo 元素添加到 mrow 元素中
            mrow.appendChild(mo)
        else:
            # 如果积分符号数量超过三个或者提供了限制条件
            for lim in reversed(expr.limits):
                # 创建一个名为 mo 的 MathML 元素
                mo = self.dom.createElement('mo')
                # 在 mo 元素中添加单个积分符号的文本节点
                mo.appendChild(self.dom.createTextNode(intsymbols[1]))
                # 如果限制条件的数量为 1
                if len(lim) == 1:
                    # 将 mo 元素添加到 mrow 元素中
                    mrow.appendChild(mo)
                # 如果限制条件的数量为 2
                if len(lim) == 2:
                    # 创建一个名为 msup 的 MathML 元素
                    msup = self.dom.createElement('msup')
                    # 将 mo 元素作为 msup 元素的子元素
                    msup.appendChild(mo)
                    # 将 lim[1] 对应的打印结果作为 msup 元素的子元素
                    msup.appendChild(self._print(lim[1]))
                    # 将 msup 元素添加到 mrow 元素中
                    mrow.appendChild(msup)
                # 如果限制条件的数量为 3
                if len(lim) == 3:
                    # 创建一个名为 msubsup 的 MathML 元素
                    msubsup = self.dom.createElement('msubsup')
                    # 将 mo 元素作为 msubsup 元素的子元素
                    msubsup.appendChild(mo)
                    # 将 lim[1] 和 lim[2] 对应的打印结果作为 msubsup 元素的子元素
                    msubsup.appendChild(self._print(lim[1]))
                    msubsup.appendChild(self._print(lim[2]))
                    # 将 msubsup 元素添加到 mrow 元素中
                    mrow.appendChild(msubsup)

        # 调用 parenthesize 方法，将表达式函数部分添加到 mrow 元素中
        mrow.appendChild(self.parenthesize(expr.function, PRECEDENCE["Mul"],
                                           strict=True))
        
        # 打印积分变量
        for lim in reversed(expr.limits):
            # 创建一个名为 mo 的 MathML 元素
            d = self.dom.createElement('mo')
            # 在 mo 元素中添加积分变量符号的 HTML 实体编码
            d.appendChild(self.dom.createTextNode('&dd;'))
            # 将 mo 元素添加到 mrow 元素中
            mrow.appendChild(d)
            # 将积分变量的打印结果添加到 mrow 元素中
            mrow.appendChild(self._print(lim[0]))

        # 返回包含积分表达式的 mrow 元素
        return mrow


    # 定义一个方法来打印求和表达式
    def _print_Sum(self, e):
        # 获取表达式的限制条件列表
        limits = list(e.limits)
        # 创建一个名为 subsup 的 MathML 元素
        subsup = self.dom.createElement('munderover')
        # 获取下限的打印结果
        low_elem = self._print(limits[0][1])
        # 获取上限的打印结果
        up_elem = self._print(limits[0][2])
        # 创建一个名为 summand 的 MathML 元素
        summand = self.dom.createElement('mo')
        # 在 summand 元素中添加求和符号的 MathML 标签
        summand.appendChild(self.dom.createTextNode(self.mathml_tag(e)))

        # 创建一个名为 low 的 MathML 元素
        low = self.dom.createElement('mrow')
        # 获取限制条件中第一个变量的打印结果
        var = self._print(limits[0][0])
        # 创建一个名为 equal 的 MathML 元素
        equal = self.dom.createElement('mo')
        # 在 equal 元素中添加等号的文本节点
        equal.appendChild(self.dom.createTextNode('='))
        # 将变量、等号和下限添加到 low 元素中
        low.appendChild(var)
        low.appendChild(equal)
        low.appendChild(low_elem)

        # 将 summand、low 和 up_elem 元素添加到 subsup 元素中
        subsup.appendChild(summand)
        subsup.appendChild(low)
        subsup.appendChild(up_elem)

        # 创建一个名为 mrow 的 MathML 元素
        mrow = self.dom.createElement('mrow')
        # 将 subsup 元素添加到 mrow 元素中
        mrow.appendChild(subsup)
        # 如果函数的字符串表示长度为 1
        if len(str(e.function)) == 1:
            # 将函数的打印结果添加到 mrow 元素中
            mrow.appendChild(self._print(e.function))
        else:
            # 创建一个名为 fence 的 MathML 元素
            fence = self.dom.createElement('mfenced')
            # 将函数的打印结果添加到 fence 元素中
            fence.appendChild(self._print(e.function))
            # 将 fence 元素添加到 mrow 元素中
            mrow.appendChild(fence)

        # 返回包含求和表达式的 mrow 元素
        return mrow
    # 定义一个方法用于打印数学符号，生成相应的 XML 元素节点
    def _print_Symbol(self, sym, style='plain'):
        # 定义一个内部方法，用于连接多个项目成为一个数学行式
        def join(items):
            if len(items) > 1:
                mrow = self.dom.createElement('mrow')
                for i, item in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode(" "))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi
        
        # 将符号名称、上标和下标转换为 Unicode 字符
        def translate(s):
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s
        
        # 使用辅助函数分离符号的名称、上标和下标
        name, supers, subs = split_super_sub(sym.name)
        # 将名称和上下标转换为 Unicode
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]
        
        # 创建数学标识符节点 mi
        mname = self.dom.createElement('mi')
        mname.appendChild(self.dom.createTextNode(name))
        
        # 根据上下标的数量和存在情况创建相应的数学节点
        if len(supers) == 0:
            if len(subs) == 0:
                x = mname
            else:
                x = self.dom.createElement('msub')
                x.appendChild(mname)
                x.appendChild(join(subs))
        else:
            if len(subs) == 0:
                x = self.dom.createElement('msup')
                x.appendChild(mname)
                x.appendChild(join(supers))
            else:
                x = self.dom.createElement('msubsup')
                x.appendChild(mname)
                x.appendChild(join(subs))
                x.appendChild(join(supers))
        
        # 根据样式设置是否使用粗体字体
        if style == 'bold':
            x.setAttribute('mathvariant', 'bold')
        
        # 返回生成的数学节点 x
        return x

    # 打印矩阵符号，根据设置的样式调用 _print_Symbol 方法
    def _print_MatrixSymbol(self, sym):
        return self._print_Symbol(sym,
                                  style=self._settings['mat_symbol_style'])

    # 将 _print_Symbol 方法绑定为 _print_RandomSymbol 方法的别名
    _print_RandomSymbol = _print_Symbol

    # 打印共轭符号，生成相应的 XML 元素节点
    def _print_conjugate(self, expr):
        enc = self.dom.createElement('menclose')
        enc.setAttribute('notation', 'top')
        enc.appendChild(self._print(expr.args[0]))
        return enc

    # 打印运算符后的表达式，生成相应的 XML 元素节点
    def _print_operator_after(self, op, expr):
        row = self.dom.createElement('mrow')
        # 将表达式和操作符按照函数的优先级进行括号化处理后添加到行中
        row.appendChild(self.parenthesize(expr, PRECEDENCE["Func"]))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(op))
        row.appendChild(mo)
        return row

    # 打印阶乘符号 '!'，生成相应的 XML 元素节点
    def _print_factorial(self, expr):
        return self._print_operator_after('!', expr.args[0])

    # 打印双阶乘符号 '!!'，生成相应的 XML 元素节点
    def _print_factorial2(self, expr):
        return self._print_operator_after('!!', expr.args[0])
    def _print_binomial(self, expr):
        # 创建一个 XML 元素 'mfenced'
        brac = self.dom.createElement('mfenced')
        # 创建一个 XML 元素 'mfrac'
        frac = self.dom.createElement('mfrac')
        # 设置 'mfrac' 的属性 'linethickness' 为 '0'
        frac.setAttribute('linethickness', '0')
        # 将表达式的第一个和第二个参数分别添加到 'mfrac' 元素中
        frac.appendChild(self._print(expr.args[0]))
        frac.appendChild(self._print(expr.args[1]))
        # 将 'mfrac' 元素添加到 'mfenced' 元素中
        brac.appendChild(frac)
        # 返回 'mfenced' 元素作为结果
        return brac

    def _print_Pow(self, e):
        # 如果指数是整数的倒数，并且启用了根号表示法，则使用根号而不是幂
        if (e.exp.is_Rational and abs(e.exp.p) == 1 and e.exp.q != 1 and
                self._settings['root_notation']):
            # 如果指数是 2 的倒数
            if e.exp.q == 2:
                x = self.dom.createElement('msqrt')
                x.appendChild(self._print(e.base))
            # 如果指数不是 2 的倒数
            if e.exp.q != 2:
                x = self.dom.createElement('mroot')
                x.appendChild(self._print(e.base))
                x.appendChild(self._print(e.exp.q))
            # 如果指数为 -1
            if e.exp.p == -1:
                frac = self.dom.createElement('mfrac')
                frac.appendChild(self._print(1))
                frac.appendChild(x)
                return frac
            else:
                return x

        # 如果指数是有理数并且指数的分母不是 1
        if e.exp.is_Rational and e.exp.q != 1:
            # 如果指数是负数
            if e.exp.is_negative:
                top = self.dom.createElement('mfrac')
                top.appendChild(self._print(1))
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._get_printed_Rational(-e.exp,
                                    self._settings['fold_frac_powers']))
                top.appendChild(x)
                return top
            else:
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._get_printed_Rational(e.exp,
                                    self._settings['fold_frac_powers']))
                return x

        # 如果指数是负数
        if e.exp.is_negative:
                top = self.dom.createElement('mfrac')
                top.appendChild(self._print(1))
                # 如果指数是 -1
                if e.exp == -1:
                    top.appendChild(self._print(e.base))
                else:
                    x = self.dom.createElement('msup')
                    x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                    x.appendChild(self._print(-e.exp))
                    top.appendChild(x)
                return top

        # 默认情况，创建一个 'msup' 元素表示幂运算
        x = self.dom.createElement('msup')
        x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
        x.appendChild(self._print(e.exp))
        return x

    def _print_Number(self, e):
        # 创建一个指定标签的 XML 元素，包含数字表达式的文本节点
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x
    # 创建 XML 元素 'mfenced'，用于表示带有括号的数学表达式
    brac = self.dom.createElement('mfenced')
    # 设置 'mfenced' 元素的闭合括号为 '\u27e9'
    brac.setAttribute('close', '\u27e9')
    # 设置 'mfenced' 元素的开启括号为 '\u27e8'
    brac.setAttribute('open', '\u27e8')
    # 将数学表达式 i.min 转换为 MathML 并添加到 'mfenced' 元素中
    brac.appendChild(self._print(i.min))
    # 将数学表达式 i.max 转换为 MathML 并添加到 'mfenced' 元素中
    brac.appendChild(self._print(i.max))
    # 返回创建的 'mfenced' 元素作为输出
    return brac

# 根据给定的表达式 e 打印其导数的 MathML 表示
def _print_Derivative(self, e):

    # 如果需要使用偏导数符号 '&#x2202;'
    if requires_partial(e.expr):
        d = '&#x2202;'
    else:
        # 否则使用数学标签表示
        d = self.mathml_tag(e)

    # 创建一个数学行 'mrow'
    m = self.dom.createElement('mrow')
    # 初始化总的微分维度为 0，用于表示分子
    dim = 0  # Total diff dimension, for numerator
    
    # 遍历变量计数列表（反向）
    for sym, num in reversed(e.variable_count):
        # 累加维度数
        dim += num
        # 如果变量次数大于等于 2
        if num >= 2:
            # 创建 'msup' 元素表示 d 的 num 次方
            x = self.dom.createElement('msup')
            xx = self.dom.createElement('mo')
            xx.appendChild(self.dom.createTextNode(d))
            x.appendChild(xx)
            # 添加 num 的 MathML 表示作为上标
            x.appendChild(self._print(num))
        else:
            # 否则创建 'mo' 元素表示 d
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(d))
        # 将创建的元素添加到 'mrow' 中
        m.appendChild(x)
        # 将符号 sym 的 MathML 表示添加到 'mrow' 中
        y = self._print(sym)
        m.appendChild(y)

    # 创建一个数学行 'mrow' 用于分子
    mnum = self.dom.createElement('mrow')
    # 如果总维度大于等于 2
    if dim >= 2:
        # 创建 'msup' 元素表示 d 的 dim 次方
        x = self.dom.createElement('msup')
        xx = self.dom.createElement('mo')
        xx.appendChild(self.dom.createTextNode(d))
        x.appendChild(xx)
        # 添加 dim 的 MathML 表示作为上标
        x.appendChild(self._print(dim))
    else:
        # 否则创建 'mo' 元素表示 d
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode(d))

    # 将创建的元素添加到分子 'mnum' 中
    mnum.appendChild(x)

    # 创建一个数学行 'mrow' 用于分式
    mrow = self.dom.createElement('mrow')
    # 创建 'mfrac' 元素表示分式
    frac = self.dom.createElement('mfrac')
    # 将分子 'mnum' 添加到分式中
    frac.appendChild(mnum)
    # 将分母 'm' 添加到分式中
    frac.appendChild(m)
    # 将创建的 'mfrac' 元素添加到 'mrow' 中
    mrow.appendChild(frac)

    # 将表达式 e.expr 的 MathML 表示添加到 'mrow' 中
    mrow.appendChild(self._print(e.expr))

    # 返回创建的 'mrow' 元素作为输出
    return mrow

# 根据给定的函数表达式 e 打印其 MathML 表示
def _print_Function(self, e):
    # 创建一个数学行 'mrow'
    mrow = self.dom.createElement('mrow')
    # 创建 'mi' 元素表示函数名，并根据设置选择 'ln' 或者函数名
    x = self.dom.createElement('mi')
    if self.mathml_tag(e) == 'log' and self._settings["ln_notation"]:
        x.appendChild(self.dom.createTextNode('ln'))
    else:
        x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
    # 创建 'mfenced' 元素表示函数参数列表
    y = self.dom.createElement('mfenced')
    # 遍历函数参数列表，并将每个参数的 MathML 表示添加到 'mfenced' 中
    for arg in e.args:
        y.appendChild(self._print(arg))
    # 将函数名 'mi' 和参数列表 'mfenced' 添加到 'mrow' 中
    mrow.appendChild(x)
    mrow.appendChild(y)
    # 返回创建的 'mrow' 元素作为输出
    return mrow
    # 定义一个方法 `_print_Float`，用于将浮点表达式转换为 MathML 格式的 XML 元素
    def _print_Float(self, expr):
        # 根据表达式的精度确定小数点后的位数
        dps = prec_to_dps(expr._prec)
        # 将浮点数转换为字符串表示，去除末尾的零
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=True)

        # 数学公式中必须有一个乘号符号（因为 2.5 10^{20} 看起来很奇怪）
        # 因此我们使用数值分隔符
        separator = self._settings['mul_symbol_mathml_numbers']
        mrow = self.dom.createElement('mrow')
        
        # 如果字符串中包含 'e'，表示科学计数法表示的浮点数
        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            # 科学计数法指数部分去掉正号
            if exp[0] == '+':
                exp = exp[1:]

            # 创建 XML 元素表示底数
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(mant))
            mrow.appendChild(mn)
            
            # 创建 XML 元素表示乘号
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode(separator))
            mrow.appendChild(mo)
            
            # 创建 XML 元素表示指数
            msup = self.dom.createElement('msup')
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode("10"))
            msup.appendChild(mn)
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(exp))
            msup.appendChild(mn)
            mrow.appendChild(msup)
            
            return mrow
        # 如果是正无穷大，返回正无穷大的 MathML 表示
        elif str_real == "+inf":
            return self._print_Infinity(None)
        # 如果是负无穷大，返回负无穷大的 MathML 表示
        elif str_real == "-inf":
            return self._print_NegativeInfinity(None)
        else:
            # 否则，直接创建 XML 元素表示浮点数
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(str_real))
            return mn

    # 定义一个方法 `_print_polylog`，用于将多对数函数表达式转换为 MathML 格式的 XML 元素
    def _print_polylog(self, expr):
        mrow = self.dom.createElement('mrow')
        m = self.dom.createElement('msub')

        # 创建 XML 元素表示多对数函数 Li
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('Li'))
        m.appendChild(mi)
        
        # 添加多对数函数的参数表达式
        m.appendChild(self._print(expr.args[0]))
        mrow.appendChild(m)
        
        # 创建 XML 元素表示参数表达式被圆括号包围
        brac = self.dom.createElement('mfenced')
        brac.appendChild(self._print(expr.args[1]))
        mrow.appendChild(brac)
        
        return mrow

    # 定义一个方法 `_print_Basic`，用于将基本表达式转换为 MathML 格式的 XML 元素
    def _print_Basic(self, e):
        mrow = self.dom.createElement('mrow')
        
        # 创建 XML 元素表示基本表达式的数学标签
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        
        # 创建 XML 元素表示基本表达式的参数被圆括号包围
        brac = self.dom.createElement('mfenced')
        for arg in e.args:
            brac.appendChild(self._print(arg))
        mrow.appendChild(brac)
        
        return mrow

    # 定义一个方法 `_print_Tuple`，用于将元组表达式转换为 MathML 格式的 XML 元素
    def _print_Tuple(self, e):
        mrow = self.dom.createElement('mrow')
        
        # 创建 XML 元素表示元组表达式的参数被圆括号包围
        x = self.dom.createElement('mfenced')
        for arg in e.args:
            x.appendChild(self._print(arg))
        mrow.appendChild(x)
        
        return mrow
    def _print_Interval(self, i):
        # 创建一个名为 mrow 的 XML 元素对象
        mrow = self.dom.createElement('mrow')
        # 创建一个名为 brac 的 mfenced XML 元素对象
        brac = self.dom.createElement('mfenced')
        if i.start == i.end:
            # 如果 Interval 的起始值和结束值相等，通常转换为 FiniteSet
            brac.setAttribute('close', '}')
            brac.setAttribute('open', '{')
            # 将起始值的打印结果作为子节点添加到 brac 中
            brac.appendChild(self._print(i.start))
        else:
            # 如果 Interval 的右侧不开放，则设置关闭符号为 ")"
            if i.right_open:
                brac.setAttribute('close', ')')
            else:
                brac.setAttribute('close', ']')

            # 如果 Interval 的左侧不开放，则设置开放符号为 "("
            if i.left_open:
                brac.setAttribute('open', '(')
            else:
                brac.setAttribute('open', '[')
            
            # 将起始值和结束值的打印结果作为子节点添加到 brac 中
            brac.appendChild(self._print(i.start))
            brac.appendChild(self._print(i.end))

        # 将 brac 作为子节点添加到 mrow 中
        mrow.appendChild(brac)
        # 返回 mrow 对象
        return mrow

    def _print_Abs(self, expr, exp=None):
        # 创建一个名为 mrow 的 XML 元素对象
        mrow = self.dom.createElement('mrow')
        # 创建一个名为 x 的 mfenced XML 元素对象
        x = self.dom.createElement('mfenced')
        x.setAttribute('close', '|')
        x.setAttribute('open', '|')
        # 将表达式的第一个参数的打印结果作为子节点添加到 x 中
        x.appendChild(self._print(expr.args[0]))
        # 将 x 作为子节点添加到 mrow 中
        mrow.appendChild(x)
        # 返回 mrow 对象
        return mrow

    # _print_Determinant 与 _print_Abs 相同
    _print_Determinant = _print_Abs

    def _print_re_im(self, c, expr):
        # 创建一个名为 mrow 的 XML 元素对象
        mrow = self.dom.createElement('mrow')
        # 创建一个名为 mi 的 XML 元素对象
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'fraktur')
        # 在 mi 元素中添加一个文本节点，内容为 c
        mi.appendChild(self.dom.createTextNode(c))
        # 将 mi 作为子节点添加到 mrow 中
        mrow.appendChild(mi)
        # 创建一个名为 brac 的 mfenced XML 元素对象
        brac = self.dom.createElement('mfenced')
        # 将表达式的打印结果作为子节点添加到 brac 中
        brac.appendChild(self._print(expr))
        # 将 brac 作为子节点添加到 mrow 中
        mrow.appendChild(brac)
        # 返回 mrow 对象
        return mrow

    def _print_re(self, expr, exp=None):
        # 调用 _print_re_im 方法，打印实部，并传入 'R' 作为参数
        return self._print_re_im('R', expr.args[0])

    def _print_im(self, expr, exp=None):
        # 调用 _print_re_im 方法，打印虚部，并传入 'I' 作为参数
        return self._print_re_im('I', expr.args[0])

    def _print_AssocOp(self, e):
        # 创建一个名为 mrow 的 XML 元素对象
        mrow = self.dom.createElement('mrow')
        # 创建一个名为 mi 的 XML 元素对象
        mi = self.dom.createElement('mi')
        # 在 mi 元素中添加一个文本节点，内容为 e 的数学标签
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        # 将 mi 作为子节点添加到 mrow 中
        mrow.appendChild(mi)
        # 遍历 e 的参数列表，将每个参数的打印结果作为子节点添加到 mrow 中
        for arg in e.args:
            mrow.appendChild(self._print(arg))
        # 返回 mrow 对象
        return mrow

    def _print_SetOp(self, expr, symbol, prec):
        # 创建一个名为 mrow 的 XML 元素对象
        mrow = self.dom.createElement('mrow')
        # 将表达式的第一个参数用指定的优先级括起来，并将其作为子节点添加到 mrow 中
        mrow.appendChild(self.parenthesize(expr.args[0], prec))
        # 遍历表达式的其余参数
        for arg in expr.args[1:]:
            # 创建一个名为 x 的 mo XML 元素对象，内容为指定的符号
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(symbol))
            # 将参数用指定的优先级括起来，并将其作为子节点添加到 mrow 中
            y = self.parenthesize(arg, prec)
            mrow.appendChild(x)
            mrow.appendChild(y)
        # 返回 mrow 对象
        return mrow

    def _print_Union(self, expr):
        # 获取并设置 Union 操作符的优先级
        prec = PRECEDENCE_TRADITIONAL['Union']
        # 调用 _print_SetOp 方法，打印 Union 表达式，并传入指定的符号和优先级
        return self._print_SetOp(expr, '&#x222A;', prec)

    def _print_Intersection(self, expr):
        # 获取并设置 Intersection 操作符的优先级
        prec = PRECEDENCE_TRADITIONAL['Intersection']
        # 调用 _print_SetOp 方法，打印 Intersection 表达式，并传入指定的符号和优先级
        return self._print_SetOp(expr, '&#x2229;', prec)

    def _print_Complement(self, expr):
        # 获取并设置 Complement 操作符的优先级
        prec = PRECEDENCE_TRADITIONAL['Complement']
        # 调用 _print_SetOp 方法，打印 Complement 表达式，并传入指定的符号和优先级
        return self._print_SetOp(expr, '&#x2216;', prec)
    # 定义一个方法来打印对称差集表达式的数学表示
    def _print_SymmetricDifference(self, expr):
        # 获取对称差集在传统数学运算中的优先级
        prec = PRECEDENCE_TRADITIONAL['SymmetricDifference']
        # 调用通用的集合操作打印方法，返回对称差集的数学表示
        return self._print_SetOp(expr, '&#x2206;', prec)

    # 定义一个方法来打印笛卡尔积集表达式的数学表示
    def _print_ProductSet(self, expr):
        # 获取笛卡尔积集在传统数学运算中的优先级
        prec = PRECEDENCE_TRADITIONAL['ProductSet']
        # 调用通用的集合操作打印方法，返回笛卡尔积集的数学表示
        return self._print_SetOp(expr, '&#x00d7;', prec)

    # 定义一个方法来打印有限集的数学表示
    def _print_FiniteSet(self, s):
        # 调用通用的集合打印方法，返回有限集的数学表示
        return self._print_set(s.args)

    # 定义一个方法来打印集合的数学表示
    def _print_set(self, s):
        # 对集合中的元素按默认排序键进行排序
        items = sorted(s, key=default_sort_key)
        # 创建一个 XML 元素 mfenced 表示集合，设置左右括号
        brac = self.dom.createElement('mfenced')
        brac.setAttribute('close', '}')
        brac.setAttribute('open', '{')
        # 遍历集合中的每个元素，将其打印并添加到 mfenced 元素中
        for item in items:
            brac.appendChild(self._print(item))
        # 返回表示集合的 XML 元素
        return brac

    # 将 _print_frozenset 方法指向 _print_set 方法，打印不可变集合的数学表示与可变集合相同
    _print_frozenset = _print_set

    # 定义一个方法来打印逻辑运算表达式的数学表示
    def _print_LogOp(self, args, symbol):
        # 创建一个 XML 元素 mrow 表示一行数学表达式
        mrow = self.dom.createElement('mrow')
        # 如果第一个参数是布尔类型且不是 Not 操作，则用括号包围
        if args[0].is_Boolean and not args[0].is_Not:
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(args[0]))
            mrow.appendChild(brac)
        else:
            mrow.appendChild(self._print(args[0]))
        # 遍历剩余参数
        for arg in args[1:]:
            # 创建一个 XML 元素 mo 表示操作符，并添加对应的符号文本
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(symbol))
            # 如果参数是布尔类型且不是 Not 操作，则用括号包围
            if arg.is_Boolean and not arg.is_Not:
                y = self.dom.createElement('mfenced')
                y.appendChild(self._print(arg))
            else:
                y = self._print(arg)
            # 将操作符和参数添加到 mrow 元素中
            mrow.appendChild(x)
            mrow.appendChild(y)
        # 返回完整的逻辑运算表达式的数学表示
        return mrow
    def _print_BasisDependent(self, expr):
        from sympy.vector import Vector  # 导入 SymPy 中的 Vector 类

        if expr == expr.zero:
            # 如果表达式为零向量，返回其打印结果
            return self._print(expr.zero)
        if isinstance(expr, Vector):
            # 如果表达式是 Vector 类型，将其分离成项
            items = expr.separate().items()
        else:
            # 否则将表达式作为单个项处理
            items = [(0, expr)]

        mrow = self.dom.createElement('mrow')  # 创建 XML 元素 <mrow>
        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key=lambda x: x[0].__str__())
            for i, (k, v) in enumerate(inneritems):
                if v == 1:
                    if i:  # 第一项不需要添加 "+"
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode('+'))
                        mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))  # 打印向量分量 k
                elif v == -1:
                    mo = self.dom.createElement('mo')
                    mo.appendChild(self.dom.createTextNode('-'))
                    mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))  # 打印负向量分量 k
                else:
                    if i:  # 第一项不需要添加 "+"
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode('+'))
                        mrow.appendChild(mo)
                    mbrac = self.dom.createElement('mfenced')  # 创建带括号的 XML 元素 <mfenced>
                    mbrac.appendChild(self._print(v))  # 打印向量分量 v
                    mrow.appendChild(mbrac)
                    mo = self.dom.createElement('mo')
                    mo.appendChild(self.dom.createTextNode('&InvisibleTimes;'))
                    mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))  # 打印向量分量 k
        return mrow  # 返回生成的 XML 元素 <mrow>


    def _print_And(self, expr):
        args = sorted(expr.args, key=default_sort_key)  # 对表达式中的参数按默认键排序
        return self._print_LogOp(args, '&#x2227;')  # 调用打印逻辑操作符的方法，表示逻辑与


    def _print_Or(self, expr):
        args = sorted(expr.args, key=default_sort_key)  # 对表达式中的参数按默认键排序
        return self._print_LogOp(args, '&#x2228;')  # 调用打印逻辑操作符的方法，表示逻辑或


    def _print_Xor(self, expr):
        args = sorted(expr.args, key=default_sort_key)  # 对表达式中的参数按默认键排序
        return self._print_LogOp(args, '&#x22BB;')  # 调用打印逻辑操作符的方法，表示逻辑异或


    def _print_Implies(self, expr):
        return self._print_LogOp(expr.args, '&#x21D2;')  # 调用打印逻辑操作符的方法，表示逻辑蕴含


    def _print_Equivalent(self, expr):
        args = sorted(expr.args, key=default_sort_key)  # 对表达式中的参数按默认键排序
        return self._print_LogOp(args, '&#x21D4;')  # 调用打印逻辑操作符的方法，表示逻辑等价


    def _print_Not(self, e):
        mrow = self.dom.createElement('mrow')  # 创建 XML 元素 <mrow>
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xAC;'))  # 添加逻辑非符号
        mrow.appendChild(mo)
        if (e.args[0].is_Boolean):
            x = self.dom.createElement('mfenced')  # 创建带括号的 XML 元素 <mfenced>
            x.appendChild(self._print(e.args[0]))  # 打印布尔表达式的内容
        else:
            x = self._print(e.args[0])  # 打印非布尔表达式的内容
        mrow.appendChild(x)
        return mrow  # 返回生成的 XML 元素 <mrow>
    # 创建一个 XML 元素 'mi'，并将数学标签（通过 mathml_tag 函数获得）的文本内容作为子节点添加
    mi = self.dom.createElement('mi')
    mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
    # 返回创建的 XML 元素 'mi'
    return mi

# 将 _print_bool 函数赋值给 _print_BooleanTrue 和 _print_BooleanFalse，以便后续使用相同逻辑处理布尔值 True 和 False

    # 创建一个 XML 元素 'mi'，并将数学标签（通过 mathml_tag 函数获得）的文本内容作为子节点添加
    mi = self.dom.createElement('mi')
    mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
    # 返回创建的 XML 元素 'mi'
    return mi

    # 创建一个 XML 元素 'mi'，并将数学标签（通过 mathml_tag 函数获得）的文本内容作为子节点添加
    mi = self.dom.createElement('mi')
    mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
    # 返回创建的 XML 元素 'mi'
    return mi

    # 设置省略号的 Unicode 表示
    dots = "\u2026"
    # 创建一个 XML 元素 'mfenced'，设置其开放符号为 '{'，关闭符号为 '}'
    brac = self.dom.createElement('mfenced')
    brac.setAttribute('close', '}')
    brac.setAttribute('open', '{')

    # 根据 Range 对象的起始、终止、步长情况，构建打印集合
    if s.start.is_infinite and s.stop.is_infinite:
        if s.step.is_positive:
            printset = dots, -1, 0, 1, dots
        else:
            printset = dots, 1, 0, -1, dots
    elif s.start.is_infinite:
        printset = dots, s[-1] - s.step, s[-1]
    elif s.stop.is_infinite:
        it = iter(s)
        printset = next(it), next(it), dots
    elif len(s) > 4:
        it = iter(s)
        printset = next(it), next(it), dots, s[-1]
    else:
        printset = tuple(s)

    # 遍历打印集合中的元素，根据其是否为省略号或其他值进行处理并添加到 'mfenced' 元素中
    for el in printset:
        if el == dots:
            # 创建一个 XML 元素 'mi'，并将省略号的文本内容添加为子节点
            mi = self.dom.createElement('mi')
            mi.appendChild(self.dom.createTextNode(dots))
            brac.appendChild(mi)
        else:
            # 将当前元素 el 打印到 'mfenced' 元素中
            brac.appendChild(self._print(el))

    # 返回构建好的 'mfenced' 元素
    return brac

    # 对可变参数函数表达式进行处理，创建包含函数名和参数的 XML 元素 'mrow'
    args = sorted(expr.args, key=default_sort_key)
    mrow = self.dom.createElement('mrow')
    mo = self.dom.createElement('mo')
    # 将函数名转换为小写并作为文本节点添加到 'mo' 元素中
    mo.appendChild(self.dom.createTextNode((str(expr.func)).lower()))
    mrow.appendChild(mo)
    brac = self.dom.createElement('mfenced')
    # 遍历参数列表，将每个参数打印并添加到 'mfenced' 元素中
    for symbol in args:
        brac.appendChild(self._print(symbol))
    mrow.appendChild(brac)
    # 返回构建好的 'mrow' 元素
    return mrow

# 将 _hprint_variadic_function 函数赋值给 _print_Min 和 _print_Max，以便后续使用相同逻辑处理 Min 和 Max 函数

    # 创建一个 XML 元素 'msup'，将指数部分的打印结果作为子节点添加
    msup = self.dom.createElement('msup')
    msup.appendChild(self._print_Exp1(None))
    # 将表达式的参数打印结果作为指数的子节点添加到 'msup' 元素中
    msup.appendChild(self._print(expr.args[0]))
    # 返回构建好的 'msup' 元素
    return msup

    # 创建一个 XML 元素 'mrow'
    mrow = self.dom.createElement('mrow')
    # 将关系表达式左侧的打印结果作为子节点添加到 'mrow' 元素中
    mrow.appendChild(self._print(e.lhs))
    x = self.dom.createElement('mo')
    # 将关系运算符（通过 mathml_tag 函数获得）的文本内容添加到 'mo' 元素中
    x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
    mrow.appendChild(x)
    # 将关系表达式右侧的打印结果作为子节点添加到 'mrow' 元素中
    mrow.appendChild(self._print(e.rhs))
    # 返回构建好的 'mrow' 元素
    return mrow

    # 创建一个 XML 元素，元素名称由 mathml_tag 函数获得，文本内容为整数 p 的字符串表示
    dom_element = self.dom.createElement(self.mathml_tag(p))
    dom_element.appendChild(self.dom.createTextNode(str(p)))
    # 返回创建的 XML 元素
    return dom_element
    # 创建一个包含下标的数学标签 'msub'
    msub = self.dom.createElement('msub')
    # 从表达式 e 中获取索引和系统对象
    index, system = e._id
    # 创建一个数学标签 'mi'
    mi = self.dom.createElement('mi')
    # 设置 'mi' 元素的数学变量属性为粗体，并插入系统对象的变量名作为文本节点
    mi.setAttribute('mathvariant', 'bold')
    mi.appendChild(self.dom.createTextNode(system._variable_names[index]))
    # 将 'mi' 元素添加到 'msub' 标签中
    msub.appendChild(mi)
    # 创建另一个数学标签 'mi'
    mi = self.dom.createElement('mi')
    # 设置 'mi' 元素的数学变量属性为粗体，并插入系统对象的名称作为文本节点
    mi.setAttribute('mathvariant', 'bold')
    mi.appendChild(self.dom.createTextNode(system._name))
    # 将 'mi' 元素添加到 'msub' 标签中
    msub.appendChild(mi)
    # 返回创建的数学标签 'msub'
    return msub

    # 创建一个包含下标和上标的数学标签 'msub' 和 'mover'
    msub = self.dom.createElement('msub')
    # 从表达式 e 中获取索引和系统对象
    index, system = e._id
    # 创建一个数学标签 'mover'
    mover = self.dom.createElement('mover')
    # 创建一个数学标签 'mi'
    mi = self.dom.createElement('mi')
    # 设置 'mi' 元素的数学变量属性为粗体，并插入系统对象的向量名作为文本节点
    mi.setAttribute('mathvariant', 'bold')
    mi.appendChild(self.dom.createTextNode(system._vector_names[index]))
    # 将 'mi' 元素添加到 'mover' 标签中
    mover.appendChild(mi)
    # 创建一个数学标签 'mo'
    mo = self.dom.createElement('mo')
    # 插入上标符号 '^' 作为 'mo' 元素的文本节点
    mo.appendChild(self.dom.createTextNode('^'))
    # 将 'mo' 元素添加到 'mover' 标签中
    mover.appendChild(mo)
    # 将 'mover' 标签添加到 'msub' 标签中
    msub.appendChild(mover)
    # 创建另一个数学标签 'mi'
    mi = self.dom.createElement('mi')
    # 设置 'mi' 元素的数学变量属性为粗体，并插入系统对象的名称作为文本节点
    mi.setAttribute('mathvariant', 'bold')
    mi.appendChild(self.dom.createTextNode(system._name))
    # 将 'mi' 元素添加到 'msub' 标签中
    msub.appendChild(mi)
    # 返回创建的数学标签 'msub'
    return msub

    # 创建一个包含上标的数学标签 'mover'
    mover = self.dom.createElement('mover')
    # 创建一个数学标签 'mi'
    mi = self.dom.createElement('mi')
    # 设置 'mi' 元素的数学变量属性为粗体，并插入文本节点 "0"
    mi.setAttribute('mathvariant', 'bold')
    mi.appendChild(self.dom.createTextNode("0"))
    # 将 'mi' 元素添加到 'mover' 标签中
    mover.appendChild(mi)
    # 创建一个数学标签 'mo'
    mo = self.dom.createElement('mo')
    # 插入上标符号 '^' 作为 'mo' 元素的文本节点
    mo.appendChild(self.dom.createTextNode('^'))
    # 将 'mo' 元素添加到 'mover' 标签中
    mover.appendChild(mo)
    # 返回创建的数学标签 'mover'
    return mover

    # 创建一个包含乘法运算的数学标签 'mrow'
    mrow = self.dom.createElement('mrow')
    # 从表达式 expr 中获取第一个向量表达式和第二个向量表达式
    vec1 = expr._expr1
    vec2 = expr._expr2
    # 将第一个向量表达式使用括号包裹，并添加到 'mrow' 标签中
    mrow.appendChild(self.parenthesize(vec1, PRECEDENCE['Mul']))
    # 创建一个数学标签 'mo'
    mo = self.dom.createElement('mo')
    # 插入乘号 '×' 作为 'mo' 元素的文本节点
    mo.appendChild(self.dom.createTextNode('&#xD7;'))
    # 将 'mo' 元素添加到 'mrow' 标签中
    mrow.appendChild(mo)
    # 将第二个向量表达式使用括号包裹，并添加到 'mrow' 标签中
    mrow.appendChild(self.parenthesize(vec2, PRECEDENCE['Mul']))
    # 返回创建的数学标签 'mrow'
    return mrow

    # 创建一个包含旋度运算的数学标签 'mrow'
    mrow = self.dom.createElement('mrow')
    # 创建一个数学标签 'mo'，插入向量算子 '∇' 作为文本节点
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('&#x2207;'))
    # 将 'mo' 元素添加到 'mrow' 标签中
    mrow.appendChild(mo)
    # 创建一个数学标签 'mo'，插入乘号 '×' 作为文本节点
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('&#xD7;'))
    # 将 'mo' 元素添加到 'mrow' 标签中
    mrow.appendChild(mo)
    # 将表达式 expr 的内容使用括号包裹，并添加到 'mrow' 标签中
    mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
    # 返回创建的数学标签 'mrow'
    return mrow

    # 创建一个包含散度运算的数学标签 'mrow'
    mrow = self.dom.createElement('mrow')
    # 创建一个数学标签 'mo'，插入向量算子 '∇' 作为文本节点
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('&#x2207;'))
    # 将 'mo' 元素添加到 'mrow' 标签中
    mrow.appendChild(mo)
    # 创建一个数学标签 'mo'，插入点乘号 '·' 作为文本节点
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('&#xB7;'))
    # 将 'mo' 元素添加到 'mrow' 标签中
    mrow.appendChild(mo)
    # 将表达式 expr 的内容使用括号包裹，并添加到 'mrow' 标签中
    mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
    # 返回创建的数学标签 'mrow'
    return mrow
    # 创建一个包含数学表达式的 XML 元素 <mrow>
    def _print_Dot(self, expr):
        mrow = self.dom.createElement('mrow')
        # 获取表达式的两个向量
        vec1 = expr._expr1
        vec2 = expr._expr2
        # 将第一个向量作为子元素添加到 <mrow> 中，并加上适当的括号
        mrow.appendChild(self.parenthesize(vec1, PRECEDENCE['Mul']))
        # 创建一个表示乘号的 <mo> 元素，并添加到 <mrow> 中
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xB7;'))
        mrow.appendChild(mo)
        # 将第二个向量作为子元素添加到 <mrow> 中，并加上适当的括号
        mrow.appendChild(self.parenthesize(vec2, PRECEDENCE['Mul']))
        # 返回创建好的 <mrow> 元素
        return mrow

    # 创建一个包含梯度符号的数学表达式的 XML 元素 <mrow>
    def _print_Gradient(self, expr):
        mrow = self.dom.createElement('mrow')
        # 创建一个表示梯度符号的 <mo> 元素，并添加到 <mrow> 中
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        # 将表达式的子表达式作为子元素添加到 <mrow> 中，并加上适当的括号
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        # 返回创建好的 <mrow> 元素
        return mrow

    # 创建一个包含拉普拉斯算子的数学表达式的 XML 元素 <mrow>
    def _print_Laplacian(self, expr):
        mrow = self.dom.createElement('mrow')
        # 创建一个表示拉普拉斯算子的 <mo> 元素，并添加到 <mrow> 中
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2206;'))
        mrow.appendChild(mo)
        # 将表达式的子表达式作为子元素添加到 <mrow> 中，并加上适当的括号
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        # 返回创建好的 <mrow> 元素
        return mrow

    # 创建一个表示整数集合的 XML 元素 <mi>
    def _print_Integers(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2124;'))
        return x

    # 创建一个表示复数集合的 XML 元素 <mi>
    def _print_Complexes(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2102;'))
        return x

    # 创建一个表示实数集合的 XML 元素 <mi>
    def _print_Reals(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x211D;'))
        return x

    # 创建一个表示自然数集合的 XML 元素 <mi>
    def _print_Naturals(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2115;'))
        return x

    # 创建一个表示自然数集合（包括零）的 XML 元素 <msub>
    def _print_Naturals0(self, e):
        sub = self.dom.createElement('msub')
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2115;'))
        sub.appendChild(x)
        # 将零作为子元素添加到 <msub> 中，并根据上下文适当格式化
        sub.appendChild(self._print(S.Zero))
        return sub

    # 创建一个包含奇异函数表示的 XML 元素 <msup>
    def _print_SingularityFunction(self, expr):
        # 计算位移和幂次
        shift = expr.args[0] - expr.args[1]
        power = expr.args[2]
        sup = self.dom.createElement('msup')
        # 创建一个带括号的 <mfenced> 元素，并添加位移作为子元素
        brac = self.dom.createElement('mfenced')
        brac.setAttribute('close', '\u27e9')
        brac.setAttribute('open', '\u27e8')
        brac.appendChild(self._print(shift))
        sup.appendChild(brac)
        # 将幂次作为子元素添加到 <msup> 中
        sup.appendChild(self._print(power))
        # 返回创建好的 <msup> 元素
        return sup

    # 创建一个表示 NaN（非数）的 XML 元素 <mi>
    def _print_NaN(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('NaN'))
        return x
    def _print_number_function(self, e, name):
        # 打印函数名及其第一个参数（如果有多个参数，则打印第一个参数及其余参数）
        sub = self.dom.createElement('msub')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(name))
        sub.appendChild(mi)
        sub.appendChild(self._print(e.args[0]))
        if len(e.args) == 1:
            return sub
        # 如果有多个参数，创建一个包含函数名和所有参数的元素
        mrow = self.dom.createElement('mrow')
        y = self.dom.createElement('mfenced')
        for arg in e.args[1:]:
            y.appendChild(self._print(arg))
        mrow.appendChild(sub)
        mrow.appendChild(y)
        return mrow

    def _print_bernoulli(self, e):
        # 打印 Bernoulli 函数
        return self._print_number_function(e, 'B')

    _print_bell = _print_bernoulli

    def _print_catalan(self, e):
        # 打印 Catalan 函数
        return self._print_number_function(e, 'C')

    def _print_euler(self, e):
        # 打印 Euler 函数
        return self._print_number_function(e, 'E')

    def _print_fibonacci(self, e):
        # 打印 Fibonacci 函数
        return self._print_number_function(e, 'F')

    def _print_lucas(self, e):
        # 打印 Lucas 函数
        return self._print_number_function(e, 'L')

    def _print_stieltjes(self, e):
        # 打印 Stieltjes 函数
        return self._print_number_function(e, '&#x03B3;')

    def _print_tribonacci(self, e):
        # 打印 Tribonacci 函数
        return self._print_number_function(e, 'T')

    def _print_ComplexInfinity(self, e):
        # 打印复数无穷大符号
        x = self.dom.createElement('mover')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x221E;'))
        x.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('~'))
        x.appendChild(mo)
        return x

    def _print_EmptySet(self, e):
        # 打印空集符号
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode('&#x2205;'))
        return x

    def _print_UniversalSet(self, e):
        # 打印全集符号
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode('&#x1D54C;'))
        return x

    def _print_Adjoint(self, expr):
        # 打印伴随操作
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(mat))
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2020;'))
        sup.appendChild(mo)
        return sup
    # 定义一个方法用于打印转置操作
    def _print_Transpose(self, expr):
        # 导入必要的库：MatrixSymbol类用于表示矩阵符号
        from sympy.matrices import MatrixSymbol
        # 获取表达式中的矩阵部分
        mat = expr.arg
        # 创建一个上标元素
        sup = self.dom.createElement('msup')
        # 如果矩阵不是MatrixSymbol类型，则用括号包围打印矩阵
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(mat))
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        # 创建一个数学运算符（mo），表示转置操作，添加到上标元素中
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('T'))
        sup.appendChild(mo)
        # 返回包含转置表示的上标元素
        return sup

    # 定义一个方法用于打印矩阵的逆操作
    def _print_Inverse(self, expr):
        # 导入必要的库：MatrixSymbol类用于表示矩阵符号
        from sympy.matrices import MatrixSymbol
        # 获取表达式中的矩阵部分
        mat = expr.arg
        # 创建一个上标元素
        sup = self.dom.createElement('msup')
        # 如果矩阵不是MatrixSymbol类型，则用括号包围打印矩阵
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(mat))
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        # 添加一个上标元素表示-1，即逆的数学运算符
        sup.appendChild(self._print(-1))
        # 返回包含逆表示的上标元素
        return sup

    # 定义一个方法用于打印矩阵乘法操作
    def _print_MatMul(self, expr):
        # 导入必要的库：MatMul类用于表示矩阵乘法表达式
        from sympy.matrices.expressions.matmul import MatMul

        # 创建一个mrow元素，用于包含多个子元素
        x = self.dom.createElement('mrow')
        # 获取乘法表达式的所有参数
        args = expr.args
        # 如果第一个参数是Mul类型，则重新排序因子
        if isinstance(args[0], Mul):
            args = args[0].as_ordered_factors() + list(args[1:])
        else:
            args = list(args)

        # 如果是MatMul类型且可以提取负号，则处理第一个参数
        if isinstance(expr, MatMul) and expr.could_extract_minus_sign():
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            # 创建一个数学运算符mo，表示负号，并添加到mrow中
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('-'))
            x.appendChild(mo)

        # 遍历所有参数（除最后一个外），用括号包围并添加乘号
        for arg in args[:-1]:
            x.appendChild(self.parenthesize(arg, precedence_traditional(expr),
                                            False))
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('&InvisibleTimes;'))
            x.appendChild(mo)
        # 处理最后一个参数，不需要括号和乘号
        x.appendChild(self.parenthesize(args[-1], precedence_traditional(expr),
                                        False))
        # 返回包含矩阵乘法表示的mrow元素
        return x

    # 定义一个方法用于打印矩阵的幂操作
    def _print_MatPow(self, expr):
        # 导入必要的库：MatrixSymbol类用于表示矩阵符号
        from sympy.matrices import MatrixSymbol
        # 获取幂操作的基数和指数
        base, exp = expr.base, expr.exp
        # 创建一个上标元素
        sup = self.dom.createElement('msup')
        # 如果基数不是MatrixSymbol类型，则用括号包围打印基数
        if not isinstance(base, MatrixSymbol):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(base))
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(base))
        # 打印指数并添加到上标元素中
        sup.appendChild(self._print(exp))
        # 返回包含幂表示的上标元素
        return sup
    # 创建一个 XML 元素 'mrow' 用于存放表达式的内容
    x = self.dom.createElement('mrow')
    # 获取表达式的参数列表
    args = expr.args
    # 遍历参数列表，除了最后一个参数之外的每个参数
    for arg in args[:-1]:
        # 将参数通过指定的优先级进行括号化，并添加到 'mrow' 元素中
        x.appendChild(
            self.parenthesize(arg, precedence_traditional(expr), False))
        # 创建一个 'mo' 元素，用于表示 Hadamard 乘积的操作符 '∘'
        mo = self.dom.createElement('mo')
        # 添加文本节点 '&#x2218;' 到 'mo' 元素中，表示 '∘' 符号
        mo.appendChild(self.dom.createTextNode('&#x2218;'))
        # 将 'mo' 元素添加到 'mrow' 元素中，表示连接两个参数及操作符 '∘'
        x.appendChild(mo)
    # 将最后一个参数通过指定的优先级进行括号化，并添加到 'mrow' 元素中
    x.appendChild(
        self.parenthesize(args[-1], precedence_traditional(expr), False))
    # 返回构建好的 XML 元素 'mrow'
    return x

    # 创建一个 XML 元素 'mn' 用于表示零矩阵
    x = self.dom.createElement('mn')
    # 添加文本节点 '&#x1D7D8' 到 'mn' 元素中，表示数学字符 '𝟘'
    x.appendChild(self.dom.createTextNode('&#x1D7D8'))
    # 返回构建好的 XML 元素 'mn'
    return x

    # 创建一个 XML 元素 'mn' 用于表示单位矩阵
    x = self.dom.createElement('mn')
    # 添加文本节点 '&#x1D7D9' 到 'mn' 元素中，表示数学字符 '𝟙'
    x.appendChild(self.dom.createTextNode('&#x1D7D9'))
    # 返回构建好的 XML 元素 'mn'
    return x

    # 创建一个 XML 元素 'mi' 用于表示标识符
    x = self.dom.createElement('mi')
    # 添加文本节点 '&#x1D540;' 到 'mi' 元素中，表示数学字符 '𝕀'
    x.appendChild(self.dom.createTextNode('&#x1D540;'))
    # 返回构建好的 XML 元素 'mi'
    return x

    # 创建一个 XML 元素 'mrow' 用于表示 floor 函数
    mrow = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'mfenced' 用于表示括号
    x = self.dom.createElement('mfenced')
    # 设置 'mfenced' 元素的闭合符为 '\u230B'，开放符为 '\u230A'
    x.setAttribute('close', '\u230B')
    x.setAttribute('open', '\u230A')
    # 将函数参数通过递归调用 _print 方法添加到 'mfenced' 元素中
    x.appendChild(self._print(e.args[0]))
    # 将 'mfenced' 元素添加到 'mrow' 元素中
    mrow.appendChild(x)
    # 返回构建好的 XML 元素 'mrow'
    return mrow

    # 创建一个 XML 元素 'mrow' 用于表示 ceiling 函数
    mrow = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'mfenced' 用于表示括号
    x = self.dom.createElement('mfenced')
    # 设置 'mfenced' 元素的闭合符为 '\u2309'，开放符为 '\u2308'
    x.setAttribute('close', '\u2309')
    x.setAttribute('open', '\u2308')
    # 将函数参数通过递归调用 _print 方法添加到 'mfenced' 元素中
    x.appendChild(self._print(e.args[0]))
    # 将 'mfenced' 元素添加到 'mrow' 元素中
    mrow.appendChild(x)
    # 返回构建好的 XML 元素 'mrow'
    return mrow

    # 创建一个 XML 元素 'mfenced' 用于表示 Lambda 表达式
    x = self.dom.createElement('mfenced')
    # 创建一个 XML 元素 'mrow' 用于包含 Lambda 表达式的各个组成部分
    mrow = self.dom.createElement('mrow')
    # 获取 Lambda 表达式的符号列表
    symbols = e.args[0]
    # 如果符号列表只包含一个符号，将其转换为 XML 元素并添加到 'mrow' 中
    if len(symbols) == 1:
        symbols = self._print(symbols[0])
    else:
        symbols = self._print(symbols)
    mrow.appendChild(symbols)
    # 创建一个 XML 元素 'mo' 用于表示 Lambda 符号 '↦'
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('&#x21A6;'))
    mrow.appendChild(mo)
    # 将 Lambda 表达式的第二部分通过递归调用 _print 方法添加到 'mrow' 中
    mrow.appendChild(self._print(e.args[1]))
    # 将 'mrow' 元素添加到 'mfenced' 元素中
    x.appendChild(mrow)
    # 返回构建好的 XML 元素 'mfenced'
    return x

    # 创建一个 XML 元素 'mfenced' 用于表示元组
    x = self.dom.createElement('mfenced')
    # 遍历元组的每个元素，通过递归调用 _print 方法将其添加到 'mfenced' 元素中
    for i in e:
        x.appendChild(self._print(i))
    # 返回构建好的 XML 元素 'mfenced'
    return x

    # 创建一个 XML 元素 'msub' 用于表示带下标的数学表达式
    x = self.dom.createElement('msub')
    # 将基础部分通过递归调用 _print 方法添加到 'msub' 元素中
    x.appendChild(self._print(e.base))
    # 如果索引的数量为 1，则将索引部分通过递归调用 _print 方法添加到 'msub' 元素中
    if len(e.indices) == 1:
        x.appendChild(self._print(e.indices[0]))
        return x
    # 如果索引的数量不为 1，则将整个索引部分通过递归调用 _print 方法添加到 'msub' 元素中
    x.appendChild(self._print(e.indices))
    # 返回构建好的 XML 元素 'msub'
    return x
    # 创建一个包含给定元素的 XML 元素 'msub'，表示数学表达式的下标
    def _print_MatrixElement(self, e):
        x = self.dom.createElement('msub')
        # 将元素 e 的父元素添加到 'msub' 元素中，加上括号以指定优先级
        x.appendChild(self.parenthesize(e.parent, PRECEDENCE["Atom"], strict=True))
        
        # 创建一个 'mfenced' 元素用于包裹索引，设置空的开始和结束符号
        brac = self.dom.createElement('mfenced')
        brac.setAttribute("close", "")
        brac.setAttribute("open", "")
        
        # 遍历 e 的索引，将每个索引转换成 XML 元素并添加到 'mfenced' 元素中
        for i in e.indices:
            brac.appendChild(self._print(i))
        
        # 将 'mfenced' 元素添加到 'msub' 元素中
        x.appendChild(brac)
        
        # 返回创建的 'msub' 元素
        return x
    
    # 创建一个包含椭圆函数 f 的 XML 元素 'mrow'
    def _print_elliptic_f(self, e):
        x = self.dom.createElement('mrow')
        # 创建一个 'mi' 元素，表示特定的数学标识符
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d5a5;'))  # 插入 Unicode 字符 &#x1d5a5; 到 'mi' 元素中
        x.appendChild(mi)
        
        # 创建一个 'mfenced' 元素，设置分隔符为 "|"
        y = self.dom.createElement('mfenced')
        y.setAttribute("separators", "|")
        
        # 遍历 e 的参数列表，将每个参数转换成 XML 元素并添加到 'mfenced' 元素中
        for i in e.args:
            y.appendChild(self._print(i))
        
        # 将 'mfenced' 元素添加到 'mrow' 元素中
        x.appendChild(y)
        
        # 返回创建的 'mrow' 元素
        return x
    
    # 创建一个包含椭圆函数 e 的 XML 元素 'mrow'
    def _print_elliptic_e(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d5a4;'))  # 插入 Unicode 字符 &#x1d5a4; 到 'mi' 元素中
        x.appendChild(mi)
        
        y = self.dom.createElement('mfenced')
        y.setAttribute("separators", "|")
        
        for i in e.args:
            y.appendChild(self._print(i))
        
        x.appendChild(y)
        return x
    
    # 创建一个包含椭圆函数 π 的 XML 元素 'mrow'
    def _print_elliptic_pi(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d6f1;'))  # 插入 Unicode 字符 &#x1d6f1; 到 'mi' 元素中
        x.appendChild(mi)
        
        y = self.dom.createElement('mfenced')
        # 根据参数数量设置不同的分隔符
        if len(e.args) == 2:
            y.setAttribute("separators", "|")
        else:
            y.setAttribute("separators", ";|")
        
        for i in e.args:
            y.appendChild(self._print(i))
        
        x.appendChild(y)
        return x
    
    # 创建一个包含指数积分函数 Ei 的 XML 元素 'mrow'
    def _print_Ei(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('Ei'))  # 插入文本 'Ei' 到 'mi' 元素中
        x.appendChild(mi)
        
        # 将函数参数转换成 XML 元素并添加到 'mrow' 元素中
        x.appendChild(self._print(e.args))
        
        return x
    
    # 创建一个包含指数函数 expint 的 XML 元素 'mrow'
    def _print_expint(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('E'))  # 插入文本 'E' 到 'mo' 元素中
        y.appendChild(mo)
        
        # 将函数的第一个参数作为下标添加到 'msub' 元素中
        y.appendChild(self._print(e.args[0]))
        
        # 将剩余的参数转换成 XML 元素并添加到 'mrow' 元素中
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        
        return x
    
    # 创建一个包含雅可比函数 P 的 XML 元素 'mrow'
    def _print_jacobi(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))  # 插入文本 'P' 到 'mo' 元素中
        y.appendChild(mo)
        
        # 将函数的第一个参数作为下标添加到 'msubsup' 元素中
        y.appendChild(self._print(e.args[0]))
        
        # 将第二个和第三个参数转换成 XML 元素并添加到 'msubsup' 元素中
        y.appendChild(self._print(e.args[1:3]))
        
        # 将剩余的参数转换成 XML 元素并添加到 'mrow' 元素中
        x.appendChild(y)
        x.appendChild(self._print(e.args[3:]))
        
        return x
    # 创建一个 XML 元素 'mrow'，用于容纳数学表达式
    x = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'msubsup'，用于表示包含上下标的数学表达式
    y = self.dom.createElement('msubsup')
    # 创建一个 XML 元素 'mo'，用于包含数学运算符 'C'
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('C'))
    # 将 'mo' 元素添加到 'msubsup' 元素中
    y.appendChild(mo)
    # 将第一个参数打印为 XML，并添加到 'msubsup' 元素中
    y.appendChild(self._print(e.args[0]))
    # 将第二个参数的子集打印为 XML，并添加到 'msubsup' 元素中
    y.appendChild(self._print(e.args[1:2]))
    # 将 'msubsup' 元素添加到 'mrow' 元素中
    x.appendChild(y)
    # 将剩余的参数打印为 XML，并添加到 'mrow' 元素中
    x.appendChild(self._print(e.args[2:]))
    # 返回包含整个数学表达式的 'mrow' 元素
    return x



    # 创建一个 XML 元素 'mrow'，用于容纳数学表达式
    x = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'msub'，用于表示包含下标的数学表达式
    y = self.dom.createElement('msub')
    # 创建一个 XML 元素 'mo'，用于包含数学运算符 'T'
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('T'))
    # 将 'mo' 元素添加到 'msub' 元素中
    y.appendChild(mo)
    # 将第一个参数打印为 XML，并添加到 'msub' 元素中
    y.appendChild(self._print(e.args[0]))
    # 将剩余的参数打印为 XML，并添加到 'mrow' 元素中
    x.appendChild(y)
    x.appendChild(self._print(e.args[1:]))
    # 返回包含整个数学表达式的 'mrow' 元素
    return x



    # 创建一个 XML 元素 'mrow'，用于容纳数学表达式
    x = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'msub'，用于表示包含下标的数学表达式
    y = self.dom.createElement('msub')
    # 创建一个 XML 元素 'mo'，用于包含数学运算符 'U'
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('U'))
    # 将 'mo' 元素添加到 'msub' 元素中
    y.appendChild(mo)
    # 将第一个参数打印为 XML，并添加到 'msub' 元素中
    y.appendChild(self._print(e.args[0]))
    # 将剩余的参数打印为 XML，并添加到 'mrow' 元素中
    x.appendChild(y)
    x.appendChild(self._print(e.args[1:]))
    # 返回包含整个数学表达式的 'mrow' 元素
    return x



    # 创建一个 XML 元素 'mrow'，用于容纳数学表达式
    x = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'msub'，用于表示包含下标的数学表达式
    y = self.dom.createElement('msub')
    # 创建一个 XML 元素 'mo'，用于包含数学运算符 'P'
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('P'))
    # 将 'mo' 元素添加到 'msub' 元素中
    y.appendChild(mo)
    # 将第一个参数打印为 XML，并添加到 'msub' 元素中
    y.appendChild(self._print(e.args[0]))
    # 将剩余的参数打印为 XML，并添加到 'mrow' 元素中
    x.appendChild(y)
    x.appendChild(self._print(e.args[1:]))
    # 返回包含整个数学表达式的 'mrow' 元素
    return x



    # 创建一个 XML 元素 'mrow'，用于容纳数学表达式
    x = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'msubsup'，用于表示包含上下标的数学表达式
    y = self.dom.createElement('msubsup')
    # 创建一个 XML 元素 'mo'，用于包含数学运算符 'P'
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('P'))
    # 将 'mo' 元素添加到 'msubsup' 元素中
    y.appendChild(mo)
    # 将第一个参数打印为 XML，并添加到 'msubsup' 元素中
    y.appendChild(self._print(e.args[0]))
    # 将第二个参数的子集打印为 XML，并添加到 'msubsup' 元素中
    y.appendChild(self._print(e.args[1:2]))
    # 将 'msubsup' 元素添加到 'mrow' 元素中
    x.appendChild(y)
    # 将剩余的参数打印为 XML，并添加到 'mrow' 元素中
    x.appendChild(self._print(e.args[2:]))
    # 返回包含整个数学表达式的 'mrow' 元素
    return x



    # 创建一个 XML 元素 'mrow'，用于容纳数学表达式
    x = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'msub'，用于表示包含下标的数学表达式
    y = self.dom.createElement('msub')
    # 创建一个 XML 元素 'mo'，用于包含数学运算符 'L'
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('L'))
    # 将 'mo' 元素添加到 'msub' 元素中
    y.appendChild(mo)
    # 将第一个参数打印为 XML，并添加到 'msub' 元素中
    y.appendChild(self._print(e.args[0]))
    # 将剩余的参数打印为 XML，并添加到 'mrow' 元素中
    x.appendChild(y)
    x.appendChild(self._print(e.args[1:]))
    # 返回包含整个数学表达式的 'mrow' 元素
    return x



    # 创建一个 XML 元素 'mrow'，用于容纳数学表达式
    x = self.dom.createElement('mrow')
    # 创建一个 XML 元素 'msubsup'，用于表示包含上下标的数学表达式
    y = self.dom.createElement('msubsup')
    # 创建一个 XML 元素 'mo'，用于包含数学运算符 'L'
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('L'))
    # 将 'mo' 元素添加到 'msubsup' 元素中
    y.appendChild(mo)
    # 将第一个参数打印为 XML，并添加到 'msubsup' 元素中
    y.appendChild(self._print(e.args[0]))
    # 将第二个参数的子集打印为 XML，并添加到 'msubsup' 元素中
    y.appendChild(self._print(e.args[1:2]))
    # 将 'msubsup' 元素添加到 'mrow' 元素中
    x.appendChild(y)
    # 将剩余的参数打印为 XML，并添加到 'mrow' 元素中
    x.appendChild(self._print(e.args[2:]))
    # 返回包含整个数学表达式的 'mrow' 元素
    return x
    # 定义一个方法 `_print_hermite`，用于生成 Hermite 多项式的表示式，并返回 XML 元素对象
    def _print_hermite(self, e):
        # 创建一个空的 XML 元素 'mrow'，用于容纳 Hermite 多项式的表示式
        x = self.dom.createElement('mrow')
        
        # 创建一个 XML 元素 'msub'，用于表示 Hermite 多项式的子表达式
        y = self.dom.createElement('msub')
        
        # 创建一个 XML 元素 'mo'，并添加文本节点 'H'
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('H'))
        
        # 将 'mo' 元素作为 'msub' 元素的子元素
        y.appendChild(mo)
        
        # 将 Hermite 多项式的第一个参数打印为 XML 元素，并添加到 'msub' 元素中
        y.appendChild(self._print(e.args[0]))
        
        # 将 'msub' 元素添加为 'mrow' 元素的子元素，构建 Hermite 多项式的表示式的一部分
        x.appendChild(y)
        
        # 将 Hermite 多项式的剩余参数打印为 XML 元素，并添加到 'mrow' 元素中，作为完整的表示式的一部分
        x.appendChild(self._print(e.args[1:]))
        
        # 返回构建好的 Hermite 多项式的完整 XML 元素表示式
        return x
# 使用装饰器将 print_function 应用于 MathMLPrinterBase 类，确保其 print 方法兼容打印功能
@print_function(MathMLPrinterBase)
# 定义 mathml 函数，返回表达式 expr 的 MathML 表示。根据 printer 参数选择输出 Presentation MathML 或 Content MathML
def mathml(expr, printer='content', **settings):
    """Returns the MathML representation of expr. If printer is presentation
    then prints Presentation MathML else prints content MathML.
    """
    # 如果 printer 参数为 'presentation'，则使用 Presentation MathML 打印器打印 expr
    if printer == 'presentation':
        return MathMLPresentationPrinter(settings).doprint(expr)
    else:
        # 否则使用 Content MathML 打印器打印 expr
        return MathMLContentPrinter(settings).doprint(expr)


# 定义 print_mathml 函数，打印表达式 expr 的 MathML 表示
def print_mathml(expr, printer='content', **settings):
    """
    Prints a pretty representation of the MathML code for expr. If printer is
    presentation then prints Presentation MathML else prints content MathML.

    Examples
    ========

    >>> ##
    >>> from sympy import print_mathml
    >>> from sympy.abc import x
    >>> print_mathml(x+1) #doctest: +NORMALIZE_WHITESPACE
    <apply>
        <plus/>
        <ci>x</ci>
        <cn>1</cn>
    </apply>
    >>> print_mathml(x+1, printer='presentation')
    <mrow>
        <mi>x</mi>
        <mo>+</mo>
        <mn>1</mn>
    </mrow>

    """
    # 根据 printer 参数选择合适的打印器
    if printer == 'presentation':
        # 如果 printer 为 'presentation'，使用 Presentation MathML 打印器
        s = MathMLPresentationPrinter(settings)
    else:
        # 否则使用 Content MathML 打印器
        s = MathMLContentPrinter(settings)
    # 将 sympify 处理后的 expr 转换为 MathML XML，并美化输出
    xml = s._print(sympify(expr))
    pretty_xml = xml.toprettyxml()

    # 打印美化后的 MathML XML
    print(pretty_xml)


# 为了向后兼容，将 MathMLPrinter 设置为 MathMLContentPrinter
MathMLPrinter = MathMLContentPrinter
```