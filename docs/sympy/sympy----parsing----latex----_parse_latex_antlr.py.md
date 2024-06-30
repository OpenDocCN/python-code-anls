# `D:\src\scipysrc\sympy\sympy\parsing\latex\_parse_latex_antlr.py`

```
# 从 importlib.metadata 模块导入 version 函数，用于获取包的版本信息
from importlib.metadata import version
# 导入 sympy 库
import sympy
# 从 sympy.external 模块导入 import_module 函数，用于动态导入模块
from sympy.external import import_module
# 从 sympy.printing.str 模块导入 StrPrinter 类
from sympy.printing.str import StrPrinter
# 从 sympy.physics.quantum.state 模块导入 Bra 和 Ket 类
from sympy.physics.quantum.state import Bra, Ket

# 导入自定义错误类 LaTeXParsingError
from .errors import LaTeXParsingError

# 初始化变量，先置为 None
LaTeXParser = LaTeXLexer = MathErrorListener = None

try:
    # 尝试导入 sympy.parsing.latex._antlr.latexparser 模块中的 LaTeXParser 类
    LaTeXParser = import_module('sympy.parsing.latex._antlr.latexparser',
                                import_kwargs={'fromlist': ['LaTeXParser']}).LaTeXParser
    # 尝试导入 sympy.parsing.latex._antlr.latexlexer 模块中的 LaTeXLexer 类
    LaTeXLexer = import_module('sympy.parsing.latex._antlr.latexlexer',
                               import_kwargs={'fromlist': ['LaTeXLexer']}).LaTeXLexer
except Exception:
    pass

# 尝试导入 antlr4.error.ErrorListener 模块中的 ErrorListener 类
ErrorListener = import_module('antlr4.error.ErrorListener',
                              warn_not_installed=True,
                              import_kwargs={'fromlist': ['ErrorListener']}
                              )

# 如果 ErrorListener 存在，则定义 MathErrorListener 类，继承自 ErrorListener.ErrorListener 类
if ErrorListener:
    class MathErrorListener(ErrorListener.ErrorListener):  # type:ignore # noqa:F811
        # MathErrorListener 类的初始化方法
        def __init__(self, src):
            super(ErrorListener.ErrorListener, self).__init__()
            self.src = src

        # MathErrorListener 类的语法错误处理方法
        def syntaxError(self, recog, symbol, line, col, msg, e):
            fmt = "%s\n%s\n%s"
            marker = "~" * col + "^"

            # 根据不同的错误类型格式化错误信息
            if msg.startswith("missing"):
                err = fmt % (msg, self.src, marker)
            elif msg.startswith("no viable"):
                err = fmt % ("I expected something else here", self.src, marker)
            elif msg.startswith("mismatched"):
                names = LaTeXParser.literalNames
                expected = [
                    names[i] for i in e.getExpectedTokens() if i < len(names)
                ]
                if len(expected) < 10:
                    expected = " ".join(expected)
                    err = (fmt % ("I expected one of these: " + expected, self.src,
                                  marker))
                else:
                    err = (fmt % ("I expected something else here", self.src,
                                  marker))
            else:
                err = fmt % ("I don't understand this", self.src, marker)
            # 抛出自定义的 LaTeXParsingError 异常
            raise LaTeXParsingError(err)


# 定义 parse_latex 函数，接收一个字符串 sympy 和一个布尔型参数 strict
def parse_latex(sympy, strict=False):
    # 导入 antlr4 模块
    antlr4 = import_module('antlr4')

    # 检查是否缺少任何依赖或者版本是否不匹配
    if None in [antlr4, MathErrorListener] or \
            not version('antlr4-python3-runtime').startswith('4.11'):
        # 如果缺少依赖或版本不匹配，则抛出 ImportError 异常
        raise ImportError("LaTeX parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime) or"
                          " conda (antlr-python-runtime), version 4.11")

    # 去除字符串 sympy 的首尾空白字符
    sympy = sympy.strip()
    # 创建 MathErrorListener 的实例 matherror
    matherror = MathErrorListener(sympy)

    # 创建 antlr4.InputStream 对象 stream，从字符串 sympy 中读取输入流
    stream = antlr4.InputStream(sympy)
    # 创建 LaTeXLexer 对象 lex，使用 stream 作为输入流
    lex = LaTeXLexer(stream)
    # 移除 lex 的默认错误监听器
    lex.removeErrorListeners()
    # 添加自定义的错误监听器 matherror 到 lex
    lex.addErrorListener(matherror)

    # 创建 antlr4.CommonTokenStream 对象 tokens，使用 lex 作为词法分析器
    tokens = antlr4.CommonTokenStream(lex)
    # 创建 LaTeXParser 对象 parser，使用 tokens 作为输入流
    parser = LaTeXParser(tokens)
    # 移除默认的控制台错误监听器
    parser.removeErrorListeners()
    # 添加自定义的错误监听器 matherror
    parser.addErrorListener(matherror)

    # 解析输入的 LaTeX 字符串并得到关系表达式
    relation = parser.math().relation()

    # 如果 strict 为 True，并且关系表达式的起始位置不为 0 或结束位置不为字符串 sympy 的长度减 1，则抛出异常
    if strict and (relation.start.start != 0 or relation.stop.stop != len(sympy) - 1):
        raise LaTeXParsingError("Invalid LaTeX")

    # 将关系表达式转换为程序中的表达式对象
    expr = convert_relation(relation)

    # 返回转换后的表达式对象
    return expr
# 将关系表达式转换为 sympy 的关系表达式
def convert_relation(rel):
    # 如果表达式存在，则转换为表达式
    if rel.expr():
        return convert_expr(rel.expr())

    # 分别递归地转换左右两个关系表达式
    lh = convert_relation(rel.relation(0))
    rh = convert_relation(rel.relation(1))
    # 根据关系操作符生成相应的 sympy 关系对象
    if rel.LT():
        return sympy.StrictLessThan(lh, rh)
    elif rel.LTE():
        return sympy.LessThan(lh, rh)
    elif rel.GT():
        return sympy.StrictGreaterThan(lh, rh)
    elif rel.GTE():
        return sympy.GreaterThan(lh, rh)
    elif rel.EQUAL():
        return sympy.Eq(lh, rh)
    elif rel.NEQ():
        return sympy.Ne(lh, rh)


# 将表达式转换为 sympy 的表达式
def convert_expr(expr):
    return convert_add(expr.additive())


# 将加法表达式转换为 sympy 的加法表达式
def convert_add(add):
    # 如果是加法，递归地转换左右两个加法表达式，并返回 sympy 加法对象
    if add.ADD():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.Add(lh, rh, evaluate=False)
    # 如果是减法，递归地转换左右两个加法表达式，并返回 sympy 加法对象
    elif add.SUB():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        # 如果右边是原子表达式，则返回加法和负数的 sympy 加法对象
        if hasattr(rh, "is_Atom") and rh.is_Atom:
            return sympy.Add(lh, -1 * rh, evaluate=False)
        # 否则返回加法和乘法的 sympy 加法对象
        return sympy.Add(lh, sympy.Mul(-1, rh, evaluate=False), evaluate=False)
    else:
        # 否则转换为乘幂表达式
        return convert_mp(add.mp())


# 将乘法表达式转换为 sympy 的乘法表达式
def convert_mp(mp):
    # 根据属性选择适当的乘法子表达式
    if hasattr(mp, 'mp'):
        mp_left = mp.mp(0)
        mp_right = mp.mp(1)
    else:
        mp_left = mp.mp_nofunc(0)
        mp_right = mp.mp_nofunc(1)

    # 根据乘法操作符生成相应的 sympy 乘法对象或乘幂对象
    if mp.MUL() or mp.CMD_TIMES() or mp.CMD_CDOT():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return sympy.Mul(lh, rh, evaluate=False)
    elif mp.DIV() or mp.CMD_DIV() or mp.COLON():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return sympy.Mul(lh, sympy.Pow(rh, -1, evaluate=False), evaluate=False)
    else:
        # 否则转换为一元表达式
        if hasattr(mp, 'unary'):
            return convert_unary(mp.unary())
        else:
            return convert_unary(mp.unary_nofunc())


# 将一元表达式转换为 sympy 的一元表达式
def convert_unary(unary):
    # 根据属性选择适当的一元子表达式
    if hasattr(unary, 'unary'):
        nested_unary = unary.unary()
    else:
        nested_unary = unary.unary_nofunc()

    # 根据一元操作符生成相应的 sympy 对象
    if unary.ADD():
        return convert_unary(nested_unary)
    elif unary.SUB():
        numabs = convert_unary(nested_unary)
        # 使用 Integer(-n) 而不是 Mul(-1, n)
        return -numabs
    elif unary.postfix():
        return convert_postfix_list(unary.postfix())


# 将后缀列表转换为相应的 sympy 对象
def convert_postfix_list(arr, i=0):
    # 如果索引超出范围，则抛出解析错误
    if i >= len(arr):
        raise LaTeXParsingError("Index out of bounds")

    # 转换后缀表达式为 sympy 对象
    res = convert_postfix(arr[i])
    # 检查 res 是否为 sympy.Expr 对象
    if isinstance(res, sympy.Expr):
        # 如果当前位置 i 是 arr 的最后一个元素，则返回 res，没有要乘的内容
        if i == len(arr) - 1:
            return res  # 没有要乘的内容
        else:
            # 如果 i 大于 0，则获取左右两侧的表达式，并判断它们是否为 sympy.Expr 对象
            if i > 0:
                left = convert_postfix(arr[i - 1])
                right = convert_postfix(arr[i + 1])
                if isinstance(left, sympy.Expr) and isinstance(
                        right, sympy.Expr):
                    # 获取左右表达式中的符号（sympy.Symbol 对象）
                    left_syms = convert_postfix(arr[i - 1]).atoms(sympy.Symbol)
                    right_syms = convert_postfix(arr[i + 1]).atoms(
                        sympy.Symbol)
                    # 如果左右两侧都不含变量，并且中间的符号是 'x'，则视为乘法操作
                    if not (left_syms or right_syms) and str(res) == 'x':
                        return convert_postfix_list(arr, i + 1)
            # 返回 res 与下一个元素的乘积
            return sympy.Mul(
                res, convert_postfix_list(arr, i + 1), evaluate=False)
    else:  # 否则，必须是导数操作
        # 获取导数的相对于的变量
        wrt = res[0]
        # 如果当前位置 i 是 arr 的最后一个元素，则抛出异常
        if i == len(arr) - 1:
            raise LaTeXParsingError("Expected expression for derivative")
        else:
            # 否则，获取从当前位置 i+1 开始的表达式并返回其关于 wrt 的导数
            expr = convert_postfix_list(arr, i + 1)
            return sympy.Derivative(expr, wrt)
# 定义函数 do_subs，接受一个表达式 expr 和一个符号 at，并根据 at 的类型进行符号替换操作
def do_subs(expr, at):
    # 如果 at 是一个表达式
    if at.expr():
        # 将 at.expr() 转换为表达式 at_expr
        at_expr = convert_expr(at.expr())
        # 找出 at_expr 中的所有符号
        syms = at_expr.atoms(sympy.Symbol)
        # 如果没有符号，则返回原始表达式 expr
        if len(syms) == 0:
            return expr
        # 如果有符号
        elif len(syms) > 0:
            # 取第一个符号
            sym = next(iter(syms))
            # 使用 at_expr 替换 expr 中的 sym 符号
            return expr.subs(sym, at_expr)
    # 如果 at 是一个等式
    elif at.equality():
        # 将等式左侧 lh 和右侧 rh 转换为表达式
        lh = convert_expr(at.equality().expr(0))
        rh = convert_expr(at.equality().expr(1))
        # 在 expr 中用 lh 替换 rh
        return expr.subs(lh, rh)


# 定义函数 convert_postfix，将后缀表达式 postfix 转换为 SymPy 表达式
def convert_postfix(postfix):
    # 如果 postfix 具有 exp 属性，则将其作为 exp_nested
    if hasattr(postfix, 'exp'):
        exp_nested = postfix.exp()
    else:
        exp_nested = postfix.exp_nofunc()

    # 将 exp_nested 转换为 SymPy 表达式 exp
    exp = convert_exp(exp_nested)
    
    # 处理后缀操作符
    for op in postfix.postfix_op():
        # 处理 '!' 后缀操作符
        if op.BANG():
            # 如果 exp 是列表，则抛出异常，不能对导数应用后缀操作
            if isinstance(exp, list):
                raise LaTeXParsingError("Cannot apply postfix to derivative")
            # 计算 exp 的阶乘，但不进行求值
            exp = sympy.factorial(exp, evaluate=False)
        # 处理 '@' 后缀操作符
        elif op.eval_at():
            ev = op.eval_at()
            at_b = None
            at_a = None
            # 如果 ev 包含 eval_at_sup，则进行符号替换，得到 at_b
            if ev.eval_at_sup():
                at_b = do_subs(exp, ev.eval_at_sup())
            # 如果 ev 包含 eval_at_sub，则进行符号替换，得到 at_a
            if ev.eval_at_sub():
                at_a = do_subs(exp, ev.eval_at_sub())
            # 根据替换结果更新 exp
            if at_b is not None and at_a is not None:
                exp = sympy.Add(at_b, -1 * at_a, evaluate=False)
            elif at_b is not None:
                exp = at_b
            elif at_a is not None:
                exp = at_a

    # 返回转换后的 SymPy 表达式 exp
    return exp


# 定义函数 convert_exp，将表达式 exp 转换为 SymPy 表达式
def convert_exp(exp):
    # 如果 exp 具有 exp 属性，则将其作为 exp_nested
    if hasattr(exp, 'exp'):
        exp_nested = exp.exp()
    else:
        exp_nested = exp.exp_nofunc()

    # 如果 exp_nested 存在，则将其转换为基础 base
    if exp_nested:
        base = convert_exp(exp_nested)
        # 如果 base 是列表，则抛出异常，不能对导数进行指数运算
        if isinstance(base, list):
            raise LaTeXParsingError("Cannot raise derivative to power")
        # 根据 exp 的原子或表达式，确定指数 exponent
        if exp.atom():
            exponent = convert_atom(exp.atom())
        elif exp.expr():
            exponent = convert_expr(exp.expr())
        # 返回 SymPy 中的幂运算结果，但不进行求值
        return sympy.Pow(base, exponent, evaluate=False)
    else:
        # 如果 exp 具有 comp 属性，则将其转换为 SymPy 表达式
        if hasattr(exp, 'comp'):
            return convert_comp(exp.comp())
        else:
            return convert_comp(exp.comp_nofunc())


# 定义函数 convert_comp，根据 comp 的类型，将其转换为 SymPy 表达式
def convert_comp(comp):
    # 如果 comp 是 group 类型，则将其组中的表达式转换为 SymPy 表达式
    if comp.group():
        return convert_expr(comp.group().expr())
    # 如果 comp 是 abs_group 类型，则计算其绝对值并返回 SymPy 表达式
    elif comp.abs_group():
        return sympy.Abs(convert_expr(comp.abs_group().expr()), evaluate=False)
    # 如果 comp 是 atom 类型，则将其转换为 SymPy 原子表达式
    elif comp.atom():
        return convert_atom(comp.atom())
    # 如果 comp 是 floor 类型，则将其转换为 SymPy floor 函数
    elif comp.floor():
        return convert_floor(comp.floor())
    # 如果 comp 是 ceil 类型，则将其转换为 SymPy ceil 函数
    elif comp.ceil():
        return convert_ceil(comp.ceil())
    # 如果 comp 是 func 类型，则将其转换为 SymPy 函数
    elif comp.func():
        return convert_func(comp.func())


# 定义函数 convert_atom，根据 atom 的类型，将其转换为 SymPy 表达式
def convert_atom(atom):
    # 如果 atom 是一个 LETTER 节点
    if atom.LETTER():
        # 获取 LETTER 节点的文本内容作为符号名
        sname = atom.LETTER().getText()
        
        # 如果存在 subexpr 节点
        if atom.subexpr():
            # 如果 subexpr 节点有 expr 子节点（即下标是一个表达式）
            if atom.subexpr().expr():
                # 将下标转换为字符串形式
                subscript = convert_expr(atom.subexpr().expr())
            else:
                # 如果下标是一个 atom（原子）
                subscript = convert_atom(atom.subexpr().atom())
            
            # 在符号名末尾加上下标，形如 sname_{subscript}
            sname += '_{' + StrPrinter().doprint(subscript) + '}'
        
        # 如果存在 SINGLE_QUOTES 节点
        if atom.SINGLE_QUOTES():
            # 将单引号内容添加到符号名末尾，用于更易识别
            sname += atom.SINGLE_QUOTES().getText()  # put after subscript for easy identify
        
        # 返回一个 sympy.Symbol 对象，表示这个符号
        return sympy.Symbol(sname)
    
    # 如果 atom 是 SYMBOL 节点
    elif atom.SYMBOL():
        # 获取 SYMBOL 节点的文本内容，去掉开头的符号（通常是 $）
        s = atom.SYMBOL().getText()[1:]
        
        # 如果是特殊符号 "infty"，返回 sympy 无穷大符号
        if s == "infty":
            return sympy.oo
        else:
            # 如果存在 subexpr 节点
            if atom.subexpr():
                subscript = None
                # 如果 subexpr 节点有 expr 子节点（即下标是一个表达式）
                if atom.subexpr().expr():
                    subscript = convert_expr(atom.subexpr().expr())
                else:
                    # 如果下标是一个 atom（原子）
                    subscript = convert_atom(atom.subexpr().atom())
                
                # 将下标转换为字符串形式
                subscriptName = StrPrinter().doprint(subscript)
                # 在符号名末尾加上下标，形如 s_{subscriptName}
                s += '_{' + subscriptName + '}'
            
            # 返回一个 sympy.Symbol 对象，表示这个符号
            return sympy.Symbol(s)
    
    # 如果 atom 是 number 节点
    elif atom.number():
        # 获取 number 节点的文本内容，并移除可能的逗号（如千位分隔符）
        s = atom.number().getText().replace(",", "")
        # 返回一个 sympy.Number 对象，表示这个数值
        return sympy.Number(s)
    
    # 如果 atom 是 DIFFERENTIAL 节点
    elif atom.DIFFERENTIAL():
        # 获取微分变量名
        var = get_differential_var(atom.DIFFERENTIAL())
        # 返回一个 sympy.Symbol 对象，表示微分符号 dvar
        return sympy.Symbol('d' + var.name)
    
    # 如果 atom 是 mathit 节点
    elif atom.mathit():
        # 将 mathit 节点转换为文本
        text = rule2text(atom.mathit().mathit_text())
        # 返回一个 sympy.Symbol 对象，表示文本所代表的符号
        return sympy.Symbol(text)
    
    # 如果 atom 是 frac 节点
    elif atom.frac():
        # 将 frac 节点转换为分数表达式
        return convert_frac(atom.frac())
    
    # 如果 atom 是 binom 节点
    elif atom.binom():
        # 将 binom 节点转换为二项式表达式
        return convert_binom(atom.binom())
    
    # 如果 atom 是 bra 节点
    elif atom.bra():
        # 将 bra 节点转换为表达式，并返回 Bra 对象
        val = convert_expr(atom.bra().expr())
        return Bra(val)
    
    # 如果 atom 是 ket 节点
    elif atom.ket():
        # 将 ket 节点转换为表达式，并返回 Ket 对象
        val = convert_expr(atom.ket().expr())
        return Ket(val)
# 将规则上下文对象转换为文本表示
def rule2text(ctx):
    # 获取输入流
    stream = ctx.start.getInputStream()
    # 获取起始标记的起始索引
    startIdx = ctx.start.start
    # 获取结束标记的结束索引
    stopIdx = ctx.stop.stop

    # 返回起始索引到结束索引之间的文本内容
    return stream.getText(startIdx, stopIdx)


# 将分式表达式转换为 SymPy 表示
def convert_frac(frac):
    diff_op = False
    partial_op = False

    # 检查是否存在下限和上限
    if frac.lower and frac.upper:
        # 获取下限的源区间
        lower_itv = frac.lower.getSourceInterval()
        lower_itv_len = lower_itv[1] - lower_itv[0] + 1

        # 如果下限是微分操作符
        if (frac.lower.start == frac.lower.stop
                and frac.lower.start.type == LaTeXLexer.DIFFERENTIAL):
            wrt = get_differential_var_str(frac.lower.start.text)
            diff_op = True
        # 如果下限是 \\partial 符号
        elif (lower_itv_len == 2 and frac.lower.start.type == LaTeXLexer.SYMBOL
              and frac.lower.start.text == '\\partial'
              and (frac.lower.stop.type == LaTeXLexer.LETTER
                   or frac.lower.stop.type == LaTeXLexer.SYMBOL)):
            partial_op = True
            wrt = frac.lower.stop.text
            if frac.lower.stop.type == LaTeXLexer.SYMBOL:
                wrt = wrt[1:]

        # 如果是微分或者偏导数操作，则创建符号变量
        if diff_op or partial_op:
            wrt = sympy.Symbol(wrt)
            # 如果是微分操作并且上限是 'd' 符号
            if (diff_op and frac.upper.start == frac.upper.stop
                    and frac.upper.start.type == LaTeXLexer.LETTER
                    and frac.upper.start.text == 'd'):
                return [wrt]
            # 如果是偏导数操作并且上限是 '\\partial' 符号
            elif (partial_op and frac.upper.start == frac.upper.stop
                  and frac.upper.start.type == LaTeXLexer.SYMBOL
                  and frac.upper.start.text == '\\partial'):
                return [wrt]

            # 获取上限文本内容
            upper_text = rule2text(frac.upper)

            expr_top = None
            # 如果是微分操作并且上限文本以 'd' 开头
            if diff_op and upper_text.startswith('d'):
                expr_top = parse_latex(upper_text[1:])
            # 如果是偏导数操作并且上限文本以 '\\partial' 开头
            elif partial_op and frac.upper.start.text == '\\partial':
                expr_top = parse_latex(upper_text[len('\\partial'):])

            # 如果成功解析了表达式，则返回对应的 SymPy 导数表示
            if expr_top:
                return sympy.Derivative(expr_top, wrt)

    # 如果只有上限，转换上限表达式
    if frac.upper:
        expr_top = convert_expr(frac.upper)
    else:
        expr_top = sympy.Number(frac.upperd.text)

    # 如果只有下限，转换下限表达式
    if frac.lower:
        expr_bot = convert_expr(frac.lower)
    else:
        expr_bot = sympy.Number(frac.lowerd.text)

    # 计算倒数的表达式
    inverse_denom = sympy.Pow(expr_bot, -1, evaluate=False)

    # 如果上限为 1，则直接返回倒数
    if expr_top == 1:
        return inverse_denom
    else:
        # 否则返回乘积的表达式
        return sympy.Mul(expr_top, inverse_denom, evaluate=False)


# 将二项式系数表达式转换为 SymPy 表示
def convert_binom(binom):
    # 转换 n 和 k 表达式
    expr_n = convert_expr(binom.n)
    expr_k = convert_expr(binom.k)
    # 返回 SymPy 的二项式系数表示
    return sympy.binomial(expr_n, expr_k, evaluate=False)


# 将向下取整表达式转换为 SymPy 表示
def convert_floor(floor):
    # 转换表达式的值
    val = convert_expr(floor.val)
    # 返回 SymPy 的向下取整函数表示
    return sympy.floor(val, evaluate=False)


# 将向上取整表达式转换为 SymPy 表示
def convert_ceil(ceil):
    # 转换表达式的值
    val = convert_expr(ceil.val)
    # 返回 SymPy 的向上取整函数表示
    return sympy.ceiling(val, evaluate=False)


# 将函数表达式转换为 SymPy 表示
def convert_func(func):
    # 检查普通函数调用是否为真
    if func.func_normal():
        # 如果函数名后带括号，则表示使用括号调用函数
        if func.L_PAREN():  # function called with parenthesis
            # 将函数参数转换为符号表达式
            arg = convert_func_arg(func.func_arg())
        else:
            # 如果函数名后没有括号，则使用无括号参数调用函数
            arg = convert_func_arg(func.func_arg_noparens())

        # 提取函数名，去除开头的字符（通常是一个符号，如“f”）
        name = func.func_normal().start.text[1:]

        # 将特定形式的函数名转换为 SymPy 中对应的函数对象，并生成表达式
        # 如将 "arcsin" 转换为 "asin"
        if name in [
                "arcsin", "arccos", "arctan", "arccsc", "arcsec", "arccot"
        ]:
            name = "a" + name[3:]  # 修改函数名
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        # 处理另一种特殊函数名转换，如将 "arsinh" 转换为 "asinh"
        if name in ["arsinh", "arcosh", "artanh"]:
            name = "a" + name[2:]  # 修改函数名
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        # 处理指数函数 "exp"
        if name == "exp":
            expr = sympy.exp(arg, evaluate=False)

        # 处理对数函数 "log", "lg", "ln"
        if name in ("log", "lg", "ln"):
            if func.subexpr():
                if func.subexpr().expr():
                    base = convert_expr(func.subexpr().expr())
                else:
                    base = convert_atom(func.subexpr().atom())
            elif name == "lg":  # ISO 80000-2:2019
                base = 10
            elif name in ("ln", "log"):  # SymPy's latex printer prints ln as log by default
                base = sympy.E
            expr = sympy.log(arg, base, evaluate=False)

        # 初始化幂函数相关变量
        func_pow = None
        should_pow = True

        # 处理上标表达式，即函数的指数部分
        if func.supexpr():
            if func.supexpr().expr():
                func_pow = convert_expr(func.supexpr().expr())
            else:
                func_pow = convert_atom(func.supexpr().atom())

        # 处理三角函数及其双曲函数
        if name in [
                "sin", "cos", "tan", "csc", "sec", "cot", "sinh", "cosh",
                "tanh"
        ]:
            # 如果指数为 -1，则表示反函数，如 sin^(-1) 转换为 arcsin
            if func_pow == -1:
                name = "a" + name
                should_pow = False
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        # 如果存在指数并且需要应用指数运算，则计算幂函数
        if func_pow and should_pow:
            expr = sympy.Pow(expr, func_pow, evaluate=False)

        # 返回最终表达式
        return expr

    # 处理字母或符号作为函数的情况
    elif func.LETTER() or func.SYMBOL():
        # 获取函数名
        if func.LETTER():
            fname = func.LETTER().getText()
        elif func.SYMBOL():
            fname = func.SYMBOL().getText()[1:]
        fname = str(fname)  # 转换为字符串，不能是Unicode

        # 处理函数的下标表达式
        if func.subexpr():
            if func.subexpr().expr():  # 下标为表达式
                subscript = convert_expr(func.subexpr().expr())
            else:  # 下标为原子
                subscript = convert_atom(func.subexpr().atom())
            subscriptName = StrPrinter().doprint(subscript)
            fname += '_{' + subscriptName + '}'

        # 处理函数名后带单引号的情况
        if func.SINGLE_QUOTES():
            fname += func.SINGLE_QUOTES().getText()

        # 处理函数的参数
        input_args = func.args()
        output_args = []
        while input_args.args():  # 处理函数的多个参数
            output_args.append(convert_expr(input_args.expr()))
            input_args = input_args.args()
        output_args.append(convert_expr(input_args.expr()))

        # 构建并返回 SymPy 的函数对象
        return sympy.Function(fname)(*output_args)
    # 如果函数对象的 FUNC_INT 方法返回 True，则处理积分函数
    elif func.FUNC_INT():
        # 调用处理积分的函数，并返回结果
        return handle_integral(func)
    
    # 如果函数对象的 FUNC_SQRT 方法返回 True，则处理平方根函数
    elif func.FUNC_SQRT():
        # 将函数的基础表达式转换为符号表达式
        expr = convert_expr(func.base)
        # 如果函数有根指数，则将根指数转换为符号表达式，并返回根运算的结果
        if func.root:
            r = convert_expr(func.root)
            return sympy.root(expr, r, evaluate=False)
        # 否则，直接返回平方根运算的结果
        else:
            return sympy.sqrt(expr, evaluate=False)
    
    # 如果函数对象的 FUNC_OVERLINE 方法返回 True，则处理共轭复数函数
    elif func.FUNC_OVERLINE():
        # 将函数的基础表达式转换为符号表达式，并返回共轭复数的结果
        expr = convert_expr(func.base)
        return sympy.conjugate(expr, evaluate=False)
    
    # 如果函数对象的 FUNC_SUM 方法返回 True，则处理求和函数
    elif func.FUNC_SUM():
        # 调用处理求和的函数，并返回结果
        return handle_sum_or_prod(func, "summation")
    
    # 如果函数对象的 FUNC_PROD 方法返回 True，则处理乘积函数
    elif func.FUNC_PROD():
        # 调用处理乘积的函数，并返回结果
        return handle_sum_or_prod(func, "product")
    
    # 如果函数对象的 FUNC_LIM 方法返回 True，则处理极限函数
    elif func.FUNC_LIM():
        # 调用处理极限的函数，并返回结果
        return handle_limit(func)
# 处理函数参数的转换，根据参数对象的属性选择转换方式
def convert_func_arg(arg):
    if hasattr(arg, 'expr'):
        return convert_expr(arg.expr())
    else:
        return convert_mp(arg.mp_nofunc())

# 处理积分函数
def handle_integral(func):
    # 根据函数是否包含加法表达式或分式选择相应的转换
    if func.additive():
        integrand = convert_add(func.additive())
    elif func.frac():
        integrand = convert_frac(func.frac())
    else:
        integrand = 1

    int_var = None
    # 如果函数包含微分符号，则获取微分变量
    if func.DIFFERENTIAL():
        int_var = get_differential_var(func.DIFFERENTIAL())
    else:
        # 否则检查积分表达式中的符号，并假设默认的微分变量为 'x'
        for sym in integrand.atoms(sympy.Symbol):
            s = str(sym)
            if len(s) > 1 and s[0] == 'd':
                if s[1] == '\\':
                    int_var = sympy.Symbol(s[2:])
                else:
                    int_var = sympy.Symbol(s[1:])
                int_sym = sym
        if int_var:
            integrand = integrand.subs(int_sym, 1)
        else:
            int_var = sympy.Symbol('x')  # 默认情况下假设微分变量为 'x'

    # 如果函数有子表达式，则处理下限和上限
    if func.subexpr():
        if func.subexpr().atom():
            lower = convert_atom(func.subexpr().atom())
        else:
            lower = convert_expr(func.subexpr().expr())
        if func.supexpr().atom():
            upper = convert_atom(func.supexpr().atom())
        else:
            upper = convert_expr(func.supexpr().expr())
        return sympy.Integral(integrand, (int_var, lower, upper))
    else:
        return sympy.Integral(integrand, int_var)

# 处理求和或乘积函数
def handle_sum_or_prod(func, name):
    val = convert_mp(func.mp())
    iter_var = convert_expr(func.subeq().equality().expr(0))
    start = convert_expr(func.subeq().equality().expr(1))
    if func.supexpr().expr():  # ^{expr}
        end = convert_expr(func.supexpr().expr())
    else:  # ^atom
        end = convert_atom(func.supexpr().atom())

    if name == "summation":
        return sympy.Sum(val, (iter_var, start, end))
    elif name == "product":
        return sympy.Product(val, (iter_var, start, end))

# 处理极限函数
def handle_limit(func):
    sub = func.limit_sub()
    # 获取极限符号后的变量，若不存在则默认为 'x'
    if sub.LETTER():
        var = sympy.Symbol(sub.LETTER().getText())
    elif sub.SYMBOL():
        var = sympy.Symbol(sub.SYMBOL().getText()[1:])
    else:
        var = sympy.Symbol('x')
    # 确定极限的方向是向正无穷还是向负无穷
    if sub.SUB():
        direction = "-"
    elif sub.ADD():
        direction = "+"
    else:
        direction = "+-"
    approaching = convert_expr(sub.expr())  # 确定极限的逼近值
    content = convert_mp(func.mp())  # 处理极限函数的主体内容

    return sympy.Limit(content, var, approaching, direction)

# 从微分对象中获取微分变量
def get_differential_var(d):
    text = get_differential_var_str(d.getText())
    return sympy.Symbol(text)

# 从字符串中获取微分变量的实际内容
def get_differential_var_str(text):
    for i in range(1, len(text)):
        c = text[i]
        if not (c == " " or c == "\r" or c == "\n" or c == "\t"):
            idx = i
            break
    text = text[idx:]
    if text[0] == "\\":
        text = text[1:]
    return text
```