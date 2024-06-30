# `D:\src\scipysrc\sympy\sympy\parsing\mathematica.py`

```
# 导入必要的模块和函数
from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable

# 导入 sympy 库的多个符号和函数
import sympy
from sympy import Mul, Add, Pow, Rational, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
    acosh, atanh, acoth, asech, acsch, expand, im, flatten, polylog, cancel, expand_trig, sign, simplify, \
    UnevaluatedExpr, S, atan, atan2, Mod, Max, Min, rf, Ei, Si, Ci, airyai, airyaiprime, airybi, primepi, prime, \
    isprime, cot, sec, csc, csch, sech, coth, Function, I, pi, Tuple, GreaterThan, StrictGreaterThan, StrictLessThan, \
    LessThan, Equality, Or, And, Lambda, Integer, Dummy, symbols

# 导入 sympy 库的特殊函数和错误处理函数
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning

# 定义函数 mathematica，用于将 Mathematica 表达式转换为 SymPy 表达式
def mathematica(s, additional_translations=None):
    # 发出 sympy 的弃用警告
    sympy_deprecation_warning(
        """The ``mathematica`` function for the Mathematica parser is now
deprecated. Use ``parse_mathematica`` instead.
The parameter ``additional_translation`` can be replaced by SymPy's
.replace( ) or .subs( ) methods on the output expression instead.""",
        deprecated_since_version="1.11",
        active_deprecations_target="mathematica-parser-new",
    )
    # 创建 MathematicaParser 对象并使用它解析输入字符串 s
    parser = MathematicaParser(additional_translations)
    return sympify(parser._parse_old(s))


# 定义函数 parse_mathematica，用于将 Mathematica 表达式转换为 SymPy 表达式
def parse_mathematica(s):
    """
    Translate a string containing a Wolfram Mathematica expression to a SymPy
    expression.

    If the translator is unable to find a suitable SymPy expression, the
    ``FullForm`` of the Mathematica expression will be output, using SymPy
    ``Function`` objects as nodes of the syntax tree.

    Examples
    ========

    >>> from sympy.parsing.mathematica import parse_mathematica
    >>> parse_mathematica("Sin[x]^2 Tan[y]")
    sin(x)**2*tan(y)
    >>> e = parse_mathematica("F[7,5,3]")
    >>> e
    F(7, 5, 3)
    >>> from sympy import Function, Max, Min
    >>> e.replace(Function("F"), lambda *x: Max(*x)*Min(*x))
    21

    Both standard input form and Mathematica full form are supported:

    >>> parse_mathematica("x*(a + b)")
    x*(a + b)
    >>> parse_mathematica("Times[x, Plus[a, b]]")
    x*(a + b)

    To get a matrix from Wolfram's code:

    >>> m = parse_mathematica("{{a, b}, {c, d}}")
    >>> m
    ((a, b), (c, d))
    >>> from sympy import Matrix
    >>> Matrix(m)
    Matrix([
    [a, b],
    [c, d]])

    If the translation into equivalent SymPy expressions fails, an SymPy
    expression equivalent to Wolfram Mathematica's "FullForm" will be created:

    >>> parse_mathematica("x_.")
    Optional(Pattern(x, Blank()))
    >>> parse_mathematica("Plus @@ {x, y, z}")
    Apply(Plus, (x, y, z))
    >>> parse_mathematica("f[x_, 3] := x^3 /; x > 0")
    SetDelayed(f(Pattern(x, Blank()), 3), Condition(x**3, x > 0))
    """
    # 创建 MathematicaParser 对象并使用它解析输入字符串 s
    parser = MathematicaParser()
    return parser.parse(s)


    # 调用 parser 对象的 parse 方法，传入参数 s 进行解析处理，并返回结果
    return parser.parse(s)
# 定义一个函数 _parse_Function，接受可变数量的参数 *args
def _parse_Function(*args):
    # 如果参数个数为 1
    if len(args) == 1:
        # 取出唯一的参数 arg
        arg = args[0]
        # 创建一个名为 Slot 的符号函数对象
        Slot = Function("Slot")
        # 在 arg 中找出所有的 Slot 符号并收集到 slots 中
        slots = arg.atoms(Slot)
        # 从 slots 中获取所有 Slot 符号的参数值
        numbers = [a.args[0] for a in slots]
        # 找出参数值中的最大值
        number_of_arguments = max(numbers)
        # 如果最大值是整数类型
        if isinstance(number_of_arguments, Integer):
            # 创建一个由符号变量组成的列表，以 dummy0 到最大参数值命名
            variables = symbols(f"dummy0:{number_of_arguments}", cls=Dummy)
            # 返回一个 Lambda 表达式，将 Slot 符号替换为对应的符号变量
            return Lambda(variables, arg.xreplace({Slot(i+1): v for i, v in enumerate(variables)}))
        # 如果最大值不是整数类型，则返回一个没有参数的 Lambda 表达式
        return Lambda((), arg)
    # 如果参数个数为 2
    elif len(args) == 2:
        # 第一个参数是变量列表，第二个参数是函数体
        variables = args[0]
        body = args[1]
        # 返回一个 Lambda 表达式，接受给定的变量列表和函数体
        return Lambda(variables, body)
    # 如果参数个数既不是 1 也不是 2，则抛出语法错误异常
    else:
        raise SyntaxError("Function node expects 1 or 2 arguments")


# 定义一个装饰器函数 _deco，接受一个类 cls 作为参数
def _deco(cls):
    # 调用类的 _initialize_class() 方法进行初始化
    cls._initialize_class()
    # 返回初始化后的类对象
    return cls


# 使用装饰器 @_deco 对 MathematicaParser 类进行装饰
class MathematicaParser:
    """
    该类的一个实例将 Wolfram Mathematica 表达式字符串转换为 SymPy 表达式。

    主要解析器分为三个内部阶段：

    1. tokenizer：对 Mathematica 表达式进行标记化，并添加缺失的 * 运算符。由 ``_from_mathematica_to_tokens(...)`` 处理。
    2. full form list：对 tokenizer 输出的字符串列表进行排序，并形成嵌套列表和字符串的语法树，等同于 Mathematica 的 ``FullForm`` 表达式输出。由函数 ``_from_tokens_to_fullformlist(...)`` 处理。
    3. SymPy 表达式：将以 full form list 形式表达的语法树遍历，并将与 SymPy 中等价类的节点替换。未知的语法树节点将被转换为 SymPy 的 ``Function`` 对象。由 ``_from_fullformlist_to_sympy(...)`` 处理。

    """

    # 左侧是 Mathematica，右侧是 SymPy 的对应关系字典
    CORRESPONDENCES = {
        'Sqrt[x]': 'sqrt(x)',
        'Rational[x,y]': 'Rational(x,y)',
        'Exp[x]': 'exp(x)',
        'Log[x]': 'log(x)',
        'Log[x,y]': 'log(y,x)',
        'Log2[x]': 'log(x,2)',
        'Log10[x]': 'log(x,10)',
        'Mod[x,y]': 'Mod(x,y)',
        'Max[*x]': 'Max(*x)',
        'Min[*x]': 'Min(*x)',
        'Pochhammer[x,y]':'rf(x,y)',
        'ArcTan[x,y]':'atan2(y,x)',
        'ExpIntegralEi[x]': 'Ei(x)',
        'SinIntegral[x]': 'Si(x)',
        'CosIntegral[x]': 'Ci(x)',
        'AiryAi[x]': 'airyai(x)',
        'AiryAiPrime[x]': 'airyaiprime(x)',
        'AiryBi[x]' :'airybi(x)',
        'AiryBiPrime[x]' :'airybiprime(x)',
        'LogIntegral[x]':' li(x)',
        'PrimePi[x]': 'primepi(x)',
        'Prime[x]': 'prime(x)',
        'PrimeQ[x]': 'isprime(x)'
    }

    # 对数学和三角函数等的处理
    for arc, tri, h in product(('', 'Arc'), (
            'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc'), ('', 'h')):
        fm = arc + tri + h + '[x]'
        if arc:  # arc func
            fs = 'a' + tri.lower() + h + '(x)'
        else:    # non-arc func
            fs = tri.lower() + h + '(x)'
        # 更新对应关系字典
        CORRESPONDENCES.update({fm: fs})

    # 替换字典，用于将空格、^、{、}等特殊字符转换为相应的字符串
    REPLACEMENTS = {
        ' ': '',
        '^': '**',
        '{': '[',
        '}': ']',
    }
    `
        RULES = {
            # 定义规则字典，其中包含各种正则表达式匹配模式和对应的替换字符串
            # 规则 'whitespace' 匹配空格，将其替换为 '*'
            'whitespace': (
                re.compile(r'''
                    (?:(?<=[a-zA-Z\d])|(?<=\d\.))     # 匹配字母或数字后，或数字后面的点
                    \s+                               # 匹配一个或多个空格
                    (?:(?=[a-zA-Z\d])|(?=\.\d))       # 匹配字母或数字前，或点和数字前
                    ''', re.VERBOSE),  # 使用 VERBOSE 标志允许多行注释和空格
                '*'),
    
            # 规则 'add*_1' 添加丢失的 '*' 字符
            'add*_1': (
                re.compile(r'''
                    (?:(?<=[])\d])|(?<=\d\.))       # 匹配闭括号或点后跟数字
                                                    # ''
                    (?=[(a-zA-Z])                   # 匹配左括号或字母前
                    ''', re.VERBOSE),  # 使用 VERBOSE 标志允许多行注释和空格
                '*'),
    
            # 规则 'add*_2' 添加丢失的 '*' 字符，匹配前面的字母后跟括号
            'add*_2': (
                re.compile(r'''
                    (?<=[a-zA-Z])       # 匹配字母前
                    \(                  # 匹配左括号字符
                    (?=.)               # 匹配任意字符
                    ''', re.VERBOSE),  # 使用 VERBOSE 标志允许多行注释和空格
                '*('),
    
            # 规则 'Pi' 将 'Pi' 替换为 'pi'
            'Pi': (
                re.compile(r'''
                    (?:
                    \A|(?<=[^a-zA-Z])
                    )
                    Pi                  # 匹配 'Pi'
                    (?=[^a-zA-Z])
                    ''', re.VERBOSE),  # 使用 VERBOSE 标志允许多行注释和空格
                'pi'),
        }
    
        # 定义 Mathematica 函数名称的正则表达式模式
        FM_PATTERN = re.compile(r'''
                    (?:
                    \A|(?<=[^a-zA-Z])   # 匹配字符串开头或非字母字符前
                    )
                    [A-Z][a-zA-Z\d]*    # 匹配以大写字母开头，后跟字母或数字的字符串
                    (?=\[)              # 匹配左中括号
                    ''', re.VERBOSE)  # 使用 VERBOSE 标志允许多行注释和空格
    
        # 定义列表或矩阵的正则表达式模式（供未来使用）
        ARG_MTRX_PATTERN = re.compile(r'''
                    \{.*\}              # 匹配包含任意字符的花括号
                    ''', re.VERBOSE)  # 使用 VERBOSE 标志允许多行注释和空格
    
        # 定义函数参数模式的正则表达式字符串模板
        ARGS_PATTERN_TEMPLATE = r'''
                    (?:
                    \A|(?<=[^a-zA-Z])
                    )
                    {arguments}         # 模型参数，如 x, y,...
                    (?=[^a-zA-Z])
                    '''
    
        # 初始化转换字典，使用元组作为键，字典作为值
        TRANSLATIONS: dict[tuple[str, int], dict[str, Any]] = {}
    
        # 原始用户翻译字典的缓存
        cache_original: dict[tuple[str, int], dict[str, Any]] = {}
    
        # 编译后的用户翻译字典的缓存
        cache_compiled: dict[tuple[str, int], dict[str, Any]] = {}
    
        @classmethod
        def _initialize_class(cls):
            # 调用类方法 _compile_dictionary，将 CORRESPONDENCES 字典转换为新的字典
            d = cls._compile_dictionary(cls.CORRESPONDENCES)
            # 更新 TRANSLATIONS 字典，包含转换后的字典
            cls.TRANSLATIONS.update(d)
    def __init__(self, additional_translations=None):
        # 初始化一个空的翻译字典
        self.translations = {}

        # 使用类常量 TRANSLATIONS 更新翻译字典
        self.translations.update(self.TRANSLATIONS)

        # 如果没有传入额外的翻译字典，则设为一个空字典
        if additional_translations is None:
            additional_translations = {}

        # 检查最新添加的额外翻译字典是否与缓存不同
        if self.__class__.cache_original != additional_translations:
            # 如果传入的额外翻译字典不是 dict 类型，抛出 ValueError 异常
            if not isinstance(additional_translations, dict):
                raise ValueError('The argument must be dict type')

            # 调用 _compile_dictionary 方法，生成经过转换的额外翻译字典
            d = self._compile_dictionary(additional_translations)

            # 更新缓存
            self.__class__.cache_original = additional_translations
            self.__class__.cache_compiled = d

        # 将用户自定义的翻译字典合并到总翻译字典中
        self.translations.update(self.__class__.cache_compiled)

    @classmethod
    def _compile_dictionary(cls, dic):
        # 初始化一个空字典，用于存放处理后的翻译规则
        d = {}

        # 遍历传入的字典 dic
        for fm, fs in dic.items():
            # 检查函数形式是否符合规范
            cls._check_input(fm)
            cls._check_input(fs)

            # 对函数名 fm 和函数内容 fs 进行预处理，去除空格
            fm = cls._apply_rules(fm, 'whitespace')
            fs = cls._apply_rules(fs, 'whitespace')

            # 去除空格
            fm = cls._replace(fm, ' ')
            fs = cls._replace(fs, ' ')

            # 在 fm 中搜索 Mathematica 函数名的模式
            m = cls.FM_PATTERN.search(fm)

            # 如果没有找到匹配项，抛出 ValueError 异常
            if m is None:
                err = "'{f}' function form is invalid.".format(f=fm)
                raise ValueError(err)

            # 获取 Mathematica 函数名，例如 'Log'
            fm_name = m.group()

            # 获取 Mathematica 函数的参数
            args, end = cls._get_args(m)

            # 检查函数形式是否符合规范，例如 '2*Func[x]' 是无效的
            if m.start() != 0 or end != len(fm):
                err = "'{f}' function form is invalid.".format(f=fm)
                raise ValueError(err)

            # 检查最后一个参数的第一个字符
            if args[-1][0] == '*':
                key_arg = '*'
            else:
                key_arg = len(args)

            # 构造字典的键，包括函数名和参数信息
            key = (fm_name, key_arg)

            # 将参数列表中的 '*x' 转换为 '\\*x' 以用于正则表达式
            re_args = [x if x[0] != '*' else '\\' + x for x in args]

            # 构造正则表达式的模式字符串
            xyz = '(?:(' + '|'.join(re_args) + '))'
            patStr = cls.ARGS_PATTERN_TEMPLATE.format(arguments=xyz)

            # 编译正则表达式模式
            pat = re.compile(patStr, re.VERBOSE)

            # 更新字典 d
            d[key] = {}
            d[key]['fs'] = fs  # SymPy 函数模板
            d[key]['args'] = args  # 参数列表，例如 ['x', 'y']
            d[key]['pat'] = pat

        # 返回处理后的字典 d
        return d
    # 将 Mathematica 函数解析为 SymPy 函数

    # 编译后的正则表达式对象
    pat = self.FM_PATTERN

    # 转换后的字符串
    scanned = ''  # converted string

    # 当前位置游标
    cur = 0  # position cursor

    while True:
        # 在字符串 s 中搜索匹配 pat 的结果
        m = pat.search(s)

        # 如果没有找到匹配项
        if m is None:
            # 将剩余的字符串直接添加到 scanned 中
            scanned += s
            break

        # 获取 Mathematica 函数名
        fm = m.group()

        # 获取函数的参数及其结束位置
        args, end = self._get_args(m)

        # 函数名的起始位置
        bgn = m.start()

        # 将 Mathematica 函数转换为 SymPy 函数
        s = self._convert_one_function(s, fm, args, bgn, end)

        # 更新游标位置
        cur = bgn

        # 将转换后的部分添加到 scanned 中
        scanned += s[:cur]

        # 缩小 s 的范围
        s = s[cur:]

    return scanned


def _convert_one_function(self, s, fm, args, bgn, end):
    # 如果是固定长度参数的情况
    if (fm, len(args)) in self.translations:
        key = (fm, len(args))

        # 获取模型参数对应的实际参数
        x_args = self.translations[key]['args']

        # 创建模型参数与实际参数的映射关系
        d = dict(zip(x_args, args))

    # 如果是可变长度参数的情况
    elif (fm, '*') in self.translations:
        key = (fm, '*')

        # 获取模型参数对应的实际参数
        x_args = self.translations[key]['args']

        # 创建模型参数与实际参数的映射关系
        d = {}
        for i, x in enumerate(x_args):
            if x[0] == '*':
                d[x] = ','.join(args[i:])
                break
            d[x] = args[i]

    # 如果不在白名单内
    else:
        err = "'{f}' is out of the whitelist.".format(f=fm)
        raise ValueError(err)

    # 转换函数的模板字符串
    template = self.translations[key]['fs']

    # 模型参数的正则表达式模式
    pat = self.translations[key]['pat']

    scanned = ''
    cur = 0
    while True:
        # 在模板字符串中搜索模型参数的匹配项
        m = pat.search(template)

        # 如果没有找到匹配项
        if m is None:
            scanned += template
            break

        # 获取模型参数
        x = m.group()

        # 获取模型参数的起始位置
        xbgn = m.start()

        # 将对应的实际参数添加到 scanned 中
        scanned += template[:xbgn] + d[x]

        # 更新游标到模型参数的结束位置
        cur = m.end()

        # 缩小模板字符串的范围
        template = template[cur:]

    # 更新原始字符串为替换后的字符串
    s = s[:bgn] + scanned + s[end:]

    return s
    def _get_args(cls, m):
        '''Get arguments of a Mathematica function'''

        s = m.string                # 获取整个字符串
        anc = m.end() + 1           # 指向参数的第一个字母的位置
        square, curly = [], []      # 括号的堆栈，用于跟踪括号匹配情况
        args = []                   # 存储提取的参数列表

        # 当前游标位置
        cur = anc
        for i, c in enumerate(s[anc:], anc):
            # 提取一个参数
            if c == ',' and (not square) and (not curly):
                args.append(s[cur:i])       # 添加一个参数
                cur = i + 1                 # 移动游标到下一个位置

            # 处理列表或矩阵（未来可能使用）
            if c == '{':
                curly.append(c)
            elif c == '}':
                curly.pop()

            # 寻找对应的']'，并跳过不相关的
            if c == '[':
                square.append(c)
            elif c == ']':
                if square:
                    square.pop()
                else:   # 空堆栈
                    args.append(s[cur:i])
                    break

        # 下一个位置是']'括号的末尾（函数的结束）
        func_end = i + 1

        return args, func_end

    @classmethod
    def _replace(cls, s, bef):
        aft = cls.REPLACEMENTS[bef]
        s = s.replace(bef, aft)
        return s

    @classmethod
    def _apply_rules(cls, s, bef):
        pat, aft = cls.RULES[bef]
        return pat.sub(aft, s)

    @classmethod
    def _check_input(cls, s):
        for bracket in (('[', ']'), ('{', '}'), ('(', ')')):
            if s.count(bracket[0]) != s.count(bracket[1]):
                err = "'{f}' 函数格式无效。".format(f=s)
                raise ValueError(err)

        if '{' in s:
            err = "目前不支持列表。"
            raise ValueError(err)

    def _parse_old(self, s):
        # 检查输入
        self._check_input(s)

        # 替换隐藏在空格后面的'*'字符
        s = self._apply_rules(s, 'whitespace')

        # 移除空白字符
        s = self._replace(s, ' ')

        # 添加省略的'*'字符
        s = self._apply_rules(s, 'add*_1')
        s = self._apply_rules(s, 'add*_2')

        # 转换函数
        s = self._convert_function(s)

        # 将'^'替换为'**'
        s = self._replace(s, '^')

        # 将'Pi'替换为'pi'
        s = self._apply_rules(s, 'Pi')

        # 将'{'和'}'分别替换为'['和']'
        # 替换字符串中的 '{'，目前未考虑列表
        s = cls._replace(s, '{')   # currently list is not taken into account
        # 替换字符串中的 '}'
        s = cls._replace(s, '}')

        # 返回处理后的字符串
        return s

    # 解析输入的数学表达式字符串并转换为 SymPy 表达式
    def parse(self, s):
        # 将数学表达式字符串转换为 token 列表
        s2 = self._from_mathematica_to_tokens(s)
        # 将 token 列表转换为 fullform 列表
        s3 = self._from_tokens_to_fullformlist(s2)
        # 将 fullform 列表转换为 SymPy 表达式
        s4 = self._from_fullformlist_to_sympy(s3)
        # 返回最终的 SymPy 表达式
        return s4

    # 定义不同操作符的类型常量
    INFIX = "Infix"
    PREFIX = "Prefix"
    POSTFIX = "Postfix"
    FLAT = "Flat"
    RIGHT = "Right"
    LEFT = "Left"

    # 定义数学表达式操作符的优先级列表
    _mathematica_op_precedence: list[tuple[str, str | None, dict[str, str | Callable]]] = [
        # 后缀操作符的处理规则，例如分号
        (POSTFIX, None, {";": lambda x: x + ["Null"] if isinstance(x, list) and x and x[0] == "CompoundExpression" else ["CompoundExpression", x, "Null"]}),
        # 中缀操作符的处理规则，例如分号
        (INFIX, FLAT, {";": "CompoundExpression"}),
        # 中缀右结合操作符的处理规则，例如等号、加等于等
        (INFIX, RIGHT, {"=": "Set", ":=": "SetDelayed", "+=": "AddTo", "-=": "SubtractFrom", "*=": "TimesBy", "/=": "DivideBy"}),
        # 中缀左结合操作符的处理规则，例如双斜杠
        (INFIX, LEFT, {"//": lambda x, y: [x, y]}),
        # 后缀操作符的处理规则，例如 & 符号
        (POSTFIX, None, {"&": "Function"}),
        # 中缀左结合操作符的处理规则，例如 /. 符号
        (INFIX, LEFT, {"/.": "ReplaceAll"}),
        # 中缀右结合操作符的处理规则，例如 -> 符号
        (INFIX, RIGHT, {"->": "Rule", ":>": "RuleDelayed"}),
        # 中缀左结合操作符的处理规则，例如 /; 符号
        (INFIX, LEFT, {"/;": "Condition"}),
        # 中缀操作符的处理规则，例如 | 符号
        (INFIX, FLAT, {"|": "Alternatives"}),
        # 后缀操作符的处理规则，例如 .. 和 ... 符号
        (POSTFIX, None, {"..": "Repeated", "...": "RepeatedNull"}),
        # 中缀操作符的处理规则，例如 || 符号
        (INFIX, FLAT, {"||": "Or"}),
        # 中缀操作符的处理规则，例如 && 符号
        (INFIX, FLAT, {"&&": "And"}),
        # 前缀操作符的处理规则，例如 ! 符号
        (PREFIX, None, {"!": "Not"}),
        # 中缀操作符的处理规则，例如 === 和 =!= 符号
        (INFIX, FLAT, {"===": "SameQ", "=!=": "UnsameQ"}),
        # 中缀操作符的处理规则，例如 ==、!=、<=、<、>=、> 符号
        (INFIX, FLAT, {"==": "Equal", "!=": "Unequal", "<=": "LessEqual", "<": "Less", ">=": "GreaterEqual", ">": "Greater"}),
        # 中缀操作符的处理规则，例如 ;; 符号
        (INFIX, None, {";;": "Span"}),
        # 中缀操作符的处理规则，例如 + 和 - 符号
        (INFIX, FLAT, {"+": "Plus", "-": "Plus"}),
        # 中缀操作符的处理规则，例如 * 和 / 符号
        (INFIX, FLAT, {"*": "Times", "/": "Times"}),
        # 中缀操作符的处理规则，例如 . 符号
        (INFIX, FLAT, {".": "Dot"}),
        # 前缀操作符的处理规则，例如 - 和 + 符号
        (PREFIX, None, {"-": lambda x: MathematicaParser._get_neg(x),
                        "+": lambda x: x}),
        # 中缀右结合操作符的处理规则，例如 ^ 符号
        (INFIX, RIGHT, {"^": "Power"}),
        # 中缀右结合操作符的处理规则，例如 @@、/@、//、@@@ 符号
        (INFIX, RIGHT, {"@@": "Apply", "/@": "Map", "//@": "MapAll", "@@@": lambda x, y: ["Apply", x, y, ["List", "1"]]}),
        # 后缀操作符的处理规则，例如 ' 符号
        (POSTFIX, None, {"'": "Derivative", "!": "Factorial", "!!": "Factorial2", "--": "Decrement"}),
        # 中缀操作符的处理规则，例如 [ 和 [[ 符号
        (INFIX, None, {"[": lambda x, y: [x, *y], "[[": lambda x, y: ["Part", x, *y]}),
        # 前缀操作符的处理规则，例如 { 和 ( 符号
        (PREFIX, None, {"{": lambda x: ["List", *x], "(": lambda x: x[0]}),
        # 中缀操作符的处理规则，例如 ? 符号
        (INFIX, None, {"?": "PatternTest"}),
        # 后缀操作符的处理规则，例如 _、_.、__、___ 符号
        (POSTFIX, None, {
            "_": lambda x: ["Pattern", x, ["Blank"]],
            "_.": lambda x: ["Optional", ["Pattern", x, ["Blank"]]],
            "__": lambda x: ["Pattern", x, ["BlankSequence"]],
            "___": lambda x: ["Pattern", x, ["BlankNullSequence"]],
        }),
        # 中缀操作符的处理规则，例如 _ 符号
        (INFIX, None, {"_": lambda x, y: ["Pattern", x, ["Blank", y]]}),
        # 前缀操作符的处理规则，例如 # 和 ## 符号
        (PREFIX, None, {"#": "Slot", "##": "SlotSequence"}),
    ]

    # 定义缺少参数时的默认处理规则字典
    _missing_arguments_default = {
        "#": lambda: ["Slot", "1"],
        "##": lambda: ["SlotSequence", "1"],
    }

    # 定义数学表达式中的字面量和数字的正则表达式模式
    _literal = r"[A-Za-z][A-Za-z0-9]*"
    _number = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"

    # 定义数学表达式中的开放符号列表
    _enclosure_open = ["(", "[", "[[", "{"]
    # 定义一个列表，包含了用于闭合结构的各种符号
    _enclosure_close = [")", "]", "]]", "}"]
    
    # 定义一个类方法，根据输入 x 返回其相反数或用于表达数学表达式的列表
    @classmethod
    def _get_neg(cls, x):
        return f"-{x}" if isinstance(x, str) and re.match(MathematicaParser._number, x) else ["Times", "-1", x]
    
    # 定义一个类方法，返回 x 的倒数表示
    @classmethod
    def _get_inv(cls, x):
        return ["Power", x, "-1"]
    
    # 初始化一个类变量，默认为 None，用于存储正则表达式编译后的对象
    _regex_tokenizer = None
    
    # 实例方法，用于获取 token 的正则表达式编译对象
    def _get_tokenizer(self):
        if self._regex_tokenizer is not None:
            # 如果已经编译过正则表达式，则直接返回已编译的对象
            return self._regex_tokenizer
        tokens = [self._literal, self._number]  # 初始 token 包括字面量和数字
        tokens_escape = self._enclosure_open[:] + self._enclosure_close[:]  # 添加开闭符号到转义 token 列表
        for typ, strat, symdict in self._mathematica_op_precedence:
            for k in symdict:
                tokens_escape.append(k)  # 将数学操作符号添加到转义 token 列表
        tokens_escape.sort(key=lambda x: -len(x))  # 根据长度降序排序转义 token 列表
        tokens.extend(map(re.escape, tokens_escape))  # 将转义后的 token 添加到 token 列表
        tokens.append(",")  # 添加逗号作为 token
        tokens.append("\n")  # 添加换行符作为 token
        # 使用 token 构建并编译正则表达式
        tokenizer = re.compile("(" + "|".join(tokens) + ")")
        self._regex_tokenizer = tokenizer  # 缓存编译后的正则表达式对象
        return self._regex_tokenizer  # 返回编译后的正则表达式对象
    # 将 Mathematica 代码转换为标记列表
    def _from_mathematica_to_tokens(self, code: str):
        # 获取标记生成器
        tokenizer = self._get_tokenizer()

        # 查找字符串并拆分代码
        code_splits: list[str | list] = []
        while True:
            string_start = code.find("\"")
            if string_start == -1:
                if len(code) > 0:
                    code_splits.append(code)
                break
            match_end = re.search(r'(?<!\\)"', code[string_start+1:])
            if match_end is None:
                raise SyntaxError('mismatch in string "  " expression')
            string_end = string_start + match_end.start() + 1
            if string_start > 0:
                code_splits.append(code[:string_start])
            code_splits.append(["_Str", code[string_start+1:string_end].replace('\\"', '"')])
            code = code[string_end+1:]

        # 移除代码中的注释
        for i, code_split in enumerate(code_splits):
            if isinstance(code_split, list):
                continue
            while True:
                pos_comment_start = code_split.find("(*")
                if pos_comment_start == -1:
                    break
                pos_comment_end = code_split.find("*)")
                if pos_comment_end == -1 or pos_comment_end < pos_comment_start:
                    raise SyntaxError("mismatch in comment (*  *) code")
                code_split = code_split[:pos_comment_start] + code_split[pos_comment_end+2:]
            code_splits[i] = code_split

        # 使用正则表达式对输入字符串进行标记化
        token_lists = [tokenizer.findall(i) if isinstance(i, str) and i.isascii() else [i] for i in code_splits]
        tokens = [j for i in token_lists for j in i]

        # 移除开头的换行符
        while tokens and tokens[0] == "\n":
            tokens.pop(0)
        # 移除末尾的换行符
        while tokens and tokens[-1] == "\n":
            tokens.pop(-1)

        return tokens

    # 判断是否为操作符或符号
    def _is_op(self, token: str | list) -> bool:
        if isinstance(token, list):
            return False
        if re.match(self._literal, token):
            return False
        if re.match("-?" + self._number, token):
            return False
        return True

    # 判断是否为有效的星号1位置
    def _is_valid_star1(self, token: str | list) -> bool:
        if token in (")", "}"):
            return True
        return not self._is_op(token)

    # 判断是否为有效的星号2位置
    def _is_valid_star2(self, token: str | list) -> bool:
        if token in ("(", "{"):
            return True
        return not self._is_op(token)
    # 将 tokens 转换为完整形式列表的私有方法
    def _from_tokens_to_fullformlist(self, tokens: list):
        # 使用列表作为栈的初始元素，用于存储解析后的结果
        stack: list[list] = [[]]
        # 存储当前打开的嵌套结构的序列
        open_seq = []
        # 指针，用于遍历 tokens 列表
        pointer: int = 0
        # 循环处理 tokens 列表中的每一个 token
        while pointer < len(tokens):
            token = tokens[pointer]
            # 如果 token 是一个开放型的嵌套结构符号
            if token in self._enclosure_open:
                # 将 token 添加到当前栈顶的列表中
                stack[-1].append(token)
                # 记录打开的嵌套结构符号
                open_seq.append(token)
                # 在栈上压入一个新的空列表，用于存储下一个嵌套结构的内容
                stack.append([])
            # 如果 token 是逗号 ","
            elif token == ",":
                # 如果当前栈顶为空，并且前一个栈顶的最后一个元素是当前打开的嵌套结构符号
                if len(stack[-1]) == 0 and stack[-2][-1] == open_seq[-1]:
                    # 抛出语法错误异常，说明在特定的嵌套结构后不应跟随逗号
                    raise SyntaxError("%s cannot be followed by comma ," % open_seq[-1])
                # 对当前栈顶的内容进行解析，并将结果作为新的栈顶
                stack[-1] = self._parse_after_braces(stack[-1])
                # 在栈上压入一个新的空列表，用于存储下一个嵌套结构的内容
                stack.append([])
            # 如果 token 是一个关闭型的嵌套结构符号
            elif token in self._enclosure_close:
                # 获取 token 在闭合符号列表中的索引
                ind = self._enclosure_close.index(token)
                # 检查当前关闭符号与最近打开的嵌套结构符号是否匹配
                if self._enclosure_open[ind] != open_seq[-1]:
                    # 抛出语法错误异常，说明嵌套结构符号不匹配
                    unmatched_enclosure = SyntaxError("unmatched enclosure")
                    # 特殊情况处理，例如 "[[" 与 "]]" 的修正
                    if token == "]]" and open_seq[-1] == "[":
                        if open_seq[-2] == "[":
                            # 在指针位置插入缺失的闭合符号 "]"
                            tokens.insert(pointer+1, "]")
                        elif open_seq[-2] == "[[":
                            # 根据后续 token 的情况修正 "[[" 与 "]]"
                            if tokens[pointer+1] == "]":
                                tokens[pointer+1] = "]]"
                            elif tokens[pointer+1] == "]]":
                                tokens[pointer+1] = "]]"
                                tokens.insert(pointer+2, "]")
                            else:
                                raise unmatched_enclosure
                    else:
                        raise unmatched_enclosure
                # 如果当前栈顶为空，并且前一个栈顶的最后一个元素是左括号 "("
                if len(stack[-1]) == 0 and stack[-2][-1] == "(":
                    # 抛出语法错误异常，说明 "( )" 不是有效的语法
                    raise SyntaxError("( ) not valid syntax")
                # 解析当前栈顶的内容，并将结果作为新的栈顶
                last_stack = self._parse_after_braces(stack[-1], True)
                stack[-1] = last_stack
                # 创建一个新的列表元素，用于存储被弹出的栈内容，直到遇到对应的开放符号
                new_stack_element = []
                while stack[-1][-1] != open_seq[-1]:
                    new_stack_element.append(stack.pop())
                new_stack_element.reverse()
                # 如果是左括号 "("，检查新的栈元素数量是否为 1
                if open_seq[-1] == "(" and len(new_stack_element) != 1:
                    # 抛出语法错误异常，说明左括号 "(" 后应只跟一个表达式
                    raise SyntaxError("( must be followed by one expression, %i detected" % len(new_stack_element))
                # 将新的栈元素作为当前栈顶的一部分
                stack[-1].append(new_stack_element)
                # 弹出已匹配的嵌套结构符号
                open_seq.pop(-1)
            else:
                # 如果 token 是普通的非结构符号，则直接添加到当前栈顶
                stack[-1].append(token)
            # 移动指针到下一个 token
            pointer += 1
        # 最终检查栈的大小是否为 1，否则抛出运行时错误异常
        if len(stack) != 1:
            raise RuntimeError("Stack should have only one element")
        # 对栈顶的内容进行最终解析处理，并返回结果
        return self._parse_after_braces(stack[0])
    # 移除输入的列表中的换行符，根据情况处理包围符号内的换行符
    def _util_remove_newlines(self, lines: list, tokens: list, inside_enclosure: bool):
        pointer = 0
        size = len(tokens)
        while pointer < size:
            token = tokens[pointer]
            if token == "\n":
                if inside_enclosure:
                    # 忽略包围符号内部的换行符
                    tokens.pop(pointer)
                    size -= 1
                    continue
                if pointer == 0:
                    # 如果是第一个 token 是换行符，直接移除
                    tokens.pop(0)
                    size -= 1
                    continue
                if pointer > 1:
                    try:
                        # 尝试解析花括号后的表达式
                        prev_expr = self._parse_after_braces(tokens[:pointer], inside_enclosure)
                    except SyntaxError:
                        # 解析错误则移除当前换行符
                        tokens.pop(pointer)
                        size -= 1
                        continue
                else:
                    prev_expr = tokens[0]
                if len(prev_expr) > 0 and prev_expr[0] == "CompoundExpression":
                    # 如果前一个表达式是复合表达式，则将其扩展到 lines 中
                    lines.extend(prev_expr[1:])
                else:
                    lines.append(prev_expr)
                # 移除前面的所有 token
                for i in range(pointer):
                    tokens.pop(0)
                size -= pointer
                pointer = 0
                continue
            pointer += 1

    # 在 tokens 列表中添加缺失的星号运算符，确保运算顺序正确
    def _util_add_missing_asterisks(self, tokens: list):
        size: int = len(tokens)
        pointer: int = 0
        while pointer < size:
            if (pointer > 0 and
                    self._is_valid_star1(tokens[pointer - 1]) and
                    self._is_valid_star2(tokens[pointer])):
                # 下面这一段代码的技巧是为了在表达式中添加缺失的 * 运算符：
                # `"*" in op_dict` 确保其优先级与 "*" 相同，
                # 而 `not self._is_op(...)` 确保当前和前一个表达式都不是运算符。
                if tokens[pointer] == "(":
                    # 已经处理了左括号 "("，现在替换：
                    tokens[pointer] = "*"
                    tokens[pointer + 1] = tokens[pointer + 1][0]
                else:
                    # 在当前位置插入 "*"
                    tokens.insert(pointer, "*")
                    pointer += 1
                    size += 1
            pointer += 1

    # 检查两个操作符是否兼容，即它们是否可以连续出现
    def _check_op_compatible(self, op1: str, op2: str):
        if op1 == op2:
            return True
        muldiv = {"*", "/"}
        addsub = {"+", "-"}
        if op1 in muldiv and op2 in muldiv:
            return True
        if op1 in addsub and op2 in addsub:
            return True
        return False
    def _from_fullform_to_fullformlist(self, wmexpr: str):
        """
        Parses FullForm[Downvalues[]] generated by Mathematica
        """
        # 初始化一个空列表用于存储解析后的表达式
        out: list = []
        # 使用一个堆栈来帮助解析嵌套结构
        stack = [out]
        # 使用正则表达式找到所有的方括号和逗号
        generator = re.finditer(r'[\[\],]', wmexpr)
        # 记录上一个匹配的位置
        last_pos = 0
        # 遍历每一个匹配项
        for match in generator:
            # 如果没有匹配到，则退出循环
            if match is None:
                break
            # 获取当前匹配项的起始位置
            position = match.start()
            # 提取两个匹配项之间的字符串，去除其中的逗号、方括号，然后去除首尾空格
            last_expr = wmexpr[last_pos:position].replace(',', '').replace(']', '').replace('[', '').strip()

            # 根据匹配项的类型进行处理
            if match.group() == ',':
                # 如果是逗号，且表达式不为空，则将其加入当前堆栈顶部列表
                if last_expr != '':
                    stack[-1].append(last_expr)
            elif match.group() == ']':
                # 如果是右方括号，处理完上一个表达式后，弹出当前堆栈
                if last_expr != '':
                    stack[-1].append(last_expr)
                stack.pop()
            elif match.group() == '[':
                # 如果是左方括号，将一个新的空列表加入当前堆栈顶部列表，并移动堆栈指针
                stack[-1].append([last_expr])
                stack.append(stack[-1][-1])
            # 更新上一个匹配项的结束位置
            last_pos = match.end()
        # 返回解析后的表达式列表的第一个元素
        return out[0]

    def _from_fullformlist_to_fullformsympy(self, pylist: list):
        from sympy import Function, Symbol

        # 定义一个递归函数来转换解析后的列表到 sympy 表达式
        def converter(expr):
            if isinstance(expr, list):
                if len(expr) > 0:
                    head = expr[0]
                    args = [converter(arg) for arg in expr[1:]]
                    return Function(head)(*args)
                else:
                    raise ValueError("Empty list of expressions")
            elif isinstance(expr, str):
                return Symbol(expr)
            else:
                return _sympify(expr)  # 假设 _sympify 函数已定义

        # 调用递归函数开始转换并返回最终的 sympy 表达式
        return converter(pylist)
    # 节点类型到相应 SymPy 函数或类的映射字典
    _node_conversions = {
        "Times": Mul,                               # 将 "Times" 映射到 Mul 类（乘法）
        "Plus": Add,                                # 将 "Plus" 映射到 Add 类（加法）
        "Power": Pow,                               # 将 "Power" 映射到 Pow 类（幂运算）
        "Rational": Rational,                       # 将 "Rational" 映射到 Rational 类（有理数）
        "Log": lambda *a: log(*reversed(a)),        # 将 "Log" 映射到一个函数，用于计算对数
        "Log2": lambda x: log(x, 2),                # 将 "Log2" 映射到一个函数，用于计算以2为底的对数
        "Log10": lambda x: log(x, 10),              # 将 "Log10" 映射到一个函数，用于计算以10为底的对数
        "Exp": exp,                                 # 将 "Exp" 映射到 exp 函数（指数函数）
        "Sqrt": sqrt,                               # 将 "Sqrt" 映射到 sqrt 函数（平方根）
    
        "Sin": sin,                                 # 将 "Sin" 映射到 sin 函数（正弦）
        "Cos": cos,                                 # 将 "Cos" 映射到 cos 函数（余弦）
        "Tan": tan,                                 # 将 "Tan" 映射到 tan 函数（正切）
        "Cot": cot,                                 # 将 "Cot" 映射到 cot 函数（余切）
        "Sec": sec,                                 # 将 "Sec" 映射到 sec 函数（正割）
        "Csc": csc,                                 # 将 "Csc" 映射到 csc 函数（余割）
    
        "ArcSin": asin,                             # 将 "ArcSin" 映射到 asin 函数（反正弦）
        "ArcCos": acos,                             # 将 "ArcCos" 映射到 acos 函数（反余弦）
        "ArcTan": lambda *a: atan2(*reversed(a)) if len(a) == 2 else atan(*a),  # 将 "ArcTan" 映射到一个函数，用于计算反正切
        "ArcCot": acot,                             # 将 "ArcCot" 映射到 acot 函数（反余切）
        "ArcSec": asec,                             # 将 "ArcSec" 映射到 asec 函数（反正割）
        "ArcCsc": acsc,                             # 将 "ArcCsc" 映射到 acsc 函数（反余割）
    
        "Sinh": sinh,                               # 将 "Sinh" 映射到 sinh 函数（双曲正弦）
        "Cosh": cosh,                               # 将 "Cosh" 映射到 cosh 函数（双曲余弦）
        "Tanh": tanh,                               # 将 "Tanh" 映射到 tanh 函数（双曲正切）
        "Coth": coth,                               # 将 "Coth" 映射到 coth 函数（双曲余切）
        "Sech": sech,                               # 将 "Sech" 映射到 sech 函数（双曲正割）
        "Csch": csch,                               # 将 "Csch" 映射到 csch 函数（双曲余割）
    
        "ArcSinh": asinh,                           # 将 "ArcSinh" 映射到 asinh 函数（反双曲正弦）
        "ArcCosh": acosh,                           # 将 "ArcCosh" 映射到 acosh 函数（反双曲余弦）
        "ArcTanh": atanh,                           # 将 "ArcTanh" 映射到 atanh 函数（反双曲正切）
        "ArcCoth": acoth,                           # 将 "ArcCoth" 映射到 acoth 函数（反双曲余切）
        "ArcSech": asech,                           # 将 "ArcSech" 映射到 asech 函数（反双曲正割）
        "ArcCsch": acsch,                           # 将 "ArcCsch" 映射到 acsch 函数（反双曲余割）
    
        "Expand": expand,                           # 将 "Expand" 映射到 expand 函数（展开表达式）
        "Im": im,                                   # 将 "Im" 映射到 im 函数（虚部）
        "Re": sympy.re,                             # 将 "Re" 映射到 sympy.re 函数（实部）
        "Flatten": flatten,                         # 将 "Flatten" 映射到 flatten 函数（扁平化）
        "Polylog": polylog,                         # 将 "Polylog" 映射到 polylog 函数（多对数）
        "Cancel": cancel,                           # 将 "Cancel" 映射到 cancel 函数（化简）
        # Gamma=gamma,                             # Gamma 函数映射暂时注释掉
        "TrigExpand": expand_trig,                  # 将 "TrigExpand" 映射到 expand_trig 函数（三角函数展开）
        "Sign": sign,                               # 将 "Sign" 映射到 sign 函数（符号函数）
        "Simplify": simplify,                       # 将 "Simplify" 映射到 simplify 函数（简化表达式）
        "Defer": UnevaluatedExpr,                   # 将 "Defer" 映射到 UnevaluatedExpr 类（延迟求值表达式）
        "Identity": S,                              # 将 "Identity" 映射到 S 对象（表示同一）
        # Sum=Sum_doit,                            # Sum 函数映射暂时注释掉
        # Module=With,                             # Module 函数映射暂时注释掉
        # Block=With,                              # Block 函数映射暂时注释掉
        "Null": lambda *a: S.Zero,                  # 将 "Null" 映射到一个函数，返回 S.Zero 对象（空值）
        "Mod": Mod,                                 # 将 "Mod" 映射到 Mod 类（模运算）
        "Max": Max,                                 # 将 "Max" 映射到 Max 函数（最大值）
        "Min": Min,                                 # 将 "Min" 映射到 Min 函数（最小值）
        "Pochhammer": rf,                           # 将 "Pochhammer" 映射到 rf 函数（Pochhammer 符号）
        "ExpIntegralEi": Ei,                        # 将 "ExpIntegralEi" 映射到 Ei 函数（指数积分 Ei）
        "SinIntegral": Si,                          # 将 "SinIntegral" 映射到 Si 函数（正弦积分 Si）
        "CosIntegral": Ci,                          # 将 "CosIntegral" 映射到 Ci 函数（余弦积分 Ci）
        "AiryAi": airyai,                           # 将 "AiryAi" 映射到 airyai 函数（Airy 函数 Ai）
        "AiryAiPrime": airyaiprime,                 # 将 "AiryAiPrime" 映射到 airyaiprime 函数（Airy 函数 Ai'）
        "AiryBi": airybi,                           # 将 "AiryBi" 映射到 airybi 函数（Airy 函数 Bi）
        "AiryBiPrime": airybiprime,                 # 将 "AiryBiPrime" 映射到 airybiprime 函数（Airy 函数 Bi'）
        "LogIntegral": li,                          # 将 "LogIntegral" 映射到 li 函数（对数积分）
        "PrimePi": primepi,                         # 将 "PrimePi" 映射到 primepi 函数（素数计数函数）
        "Prime": prime,                             # 将 "Prime" 映射到 prime 函数（判断是否为素数）
        "PrimeQ": isprime,                          # 将 "PrimeQ" 映射到 isprime 函数（
```