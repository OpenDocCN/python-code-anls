# `D:\src\scipysrc\sympy\sympy\parsing\sympy_parser.py`

```
"""Transform a string with Python-like source code into SymPy expression. """

# 导入所需模块和函数
from tokenize import (generate_tokens, untokenize, TokenError,
    NUMBER, STRING, NAME, OP, ENDMARKER, ERRORTOKEN, NEWLINE)
# 导入检查关键字的函数
from keyword import iskeyword
# 导入抽象语法树模块
import ast
# 导入Unicode数据处理模块
import unicodedata
# 导入字符串IO模块
from io import StringIO
# 导入内建函数和类型
import builtins
import types
# 导入类型提示
from typing import Tuple as tTuple, Dict as tDict, Any, Callable, \
    List, Optional, Union as tUnion

# 导入符号代数相关模块和函数
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min

# 空字符串的别名
null = ''

# 定义类型别名
TOKEN = tTuple[int, str]
DICT = tDict[str, Any]
TRANS = Callable[[List[TOKEN], DICT, DICT], List[TOKEN]]

def _token_splittable(token_name: str) -> bool:
    """
    Predicate for whether a token name can be split into multiple tokens.

    A token is splittable if it does not contain an underscore character and
    it is not the name of a Greek letter. This is used to implicitly convert
    expressions like 'xyz' into 'x*y*z'.
    """
    # 检查token_name是否包含下划线
    if '_' in token_name:
        return False
    try:
        # 尝试查找希腊小写字母对应的Unicode字符，如果找不到则说明可以拆分
        return not unicodedata.lookup('GREEK SMALL LETTER ' + token_name)
    except KeyError:
        return len(token_name) > 1

def _token_callable(token: TOKEN, local_dict: DICT, global_dict: DICT, nextToken=None):
    """
    Predicate for whether a token name represents a callable function.

    Essentially wraps ``callable``, but looks up the token name in the
    locals and globals.
    """
    # 从本地字典或全局字典中获取token对应的函数
    func = local_dict.get(token[1])
    if not func:
        func = global_dict.get(token[1])
    # 检查获取到的函数是否可调用且不是Symbol对象
    return callable(func) and not isinstance(func, Symbol)

def _add_factorial_tokens(name: str, result: List[TOKEN]) -> List[TOKEN]:
    # 如果结果为空或最后一个token是左括号，则抛出TokenError异常
    if result == [] or result[-1][1] == '(':
        raise TokenError()

    # 开始构造阶乘函数的token列表
    beginning = [(NAME, name), (OP, '(')]
    end = [(OP, ')')]

    diff = 0
    length = len(result)

    # 反向遍历结果列表，查找适合插入阶乘函数token的位置
    for index, token in enumerate(result[::-1]):
        toknum, tokval = token
        i = length - index - 1

        if tokval == ')':
            diff += 1
        elif tokval == '(':
            diff -= 1

        if diff == 0:
            if i - 1 >= 0 and result[i - 1][0] == NAME:
                return result[:i - 1] + beginning + result[i - 1:] + end
            else:
                return result[:i] + beginning + result[i:] + end

    return result

class ParenthesisGroup(List[TOKEN]):
    """List of tokens representing an expression in parentheses."""
    pass

class AppliedFunction:
    """
    A group of tokens representing a function and its arguments.

    `exponent` is for handling the shorthand sin^2, ln^2, etc.
    """
    pass
    # 初始化方法，用于实例化对象
    def __init__(self, function: TOKEN, args: ParenthesisGroup, exponent=None):
        # 如果指数为空，设为一个空列表
        if exponent is None:
            exponent = []
        # 将参数存储到对象属性中
        self.function = function  # 存储函数标识符
        self.args = args  # 存储参数组对象
        self.exponent = exponent  # 存储指数（默认为空列表）
        self.items = ['function', 'args', 'exponent']  # 存储属性名称的列表

    # 返回一个代表函数的标记列表
    def expand(self) -> List[TOKEN]:
        """Return a list of tokens representing the function"""
        return [self.function, *self.args]  # 返回函数标识符和参数组成的列表

    # 通过索引获取对象的属性
    def __getitem__(self, index):
        return getattr(self, self.items[index])  # 返回指定属性的值

    # 返回对象的字符串表示形式
    def __repr__(self):
        return "AppliedFunction(%s, %s, %s)" % (self.function, self.args,
                                                self.exponent)
# 将列表中的应用函数展平为令牌列表
def _flatten(result: List[tUnion[TOKEN, AppliedFunction]]):
    result2: List[TOKEN] = []
    for tok in result:
        if isinstance(tok, AppliedFunction):
            # 如果是应用函数，则展开并添加到结果列表中
            result2.extend(tok.expand())
        else:
            # 如果不是应用函数，则直接添加到结果列表中
            result2.append(tok)
    return result2


# 使用递归器递归处理括号组内的令牌，并将其分组为 ParenthesisGroup
def _group_parentheses(recursor: TRANS):
    def _inner(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
        """Group tokens between parentheses with ParenthesisGroup.

        Also processes those tokens recursively.

        """
        result: List[tUnion[TOKEN, ParenthesisGroup]] = []
        stacks: List[ParenthesisGroup] = []
        stacklevel = 0
        for token in tokens:
            if token[0] == OP:
                if token[1] == '(':
                    # 开始一个新的 ParenthesisGroup
                    stacks.append(ParenthesisGroup([]))
                    stacklevel += 1
                elif token[1] == ')':
                    # 遇到右括号时，将括号内的内容整体处理为一个 ParenthesisGroup
                    stacks[-1].append(token)
                    stack = stacks.pop()

                    if len(stacks) > 0:
                        # 如果还有上层的 stack，则将当前 stack 内容添加到上层 stack 中
                        stacks[-1].extend(stack)
                    else:
                        # 对内部的括号组进行递归处理
                        # 去除外层括号，以避免无限循环
                        inner = stack[1:-1]
                        inner = recursor(inner, local_dict, global_dict)
                        parenGroup = [stack[0]] + inner + [stack[-1]]
                        result.append(ParenthesisGroup(parenGroup))
                    stacklevel -= 1
                    continue
            if stacklevel:
                # 如果在括号组内，将 token 添加到当前 stack 中
                stacks[-1].append(token)
            else:
                # 如果不在括号组内，直接将 token 添加到结果列表中
                result.append(token)
        if stacklevel:
            # 如果括号不匹配，则抛出 TokenError 异常
            raise TokenError("Mismatched parentheses")
        return result
    return _inner


# 将 NAME 令牌与 ParenthesisGroup 转换为 AppliedFunction
def _apply_functions(tokens: List[tUnion[TOKEN, ParenthesisGroup]], local_dict: DICT, global_dict: DICT):
    """Convert a NAME token + ParenthesisGroup into an AppliedFunction.

    Note that ParenthesisGroups, if not applied to any function, are
    converted back into lists of tokens.

    """
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    symbol = None
    for tok in tokens:
        if isinstance(tok, ParenthesisGroup):
            # 如果是 ParenthesisGroup，则尝试将其应用到之前收集的 NAME 令牌上
            if symbol and _token_callable(symbol, local_dict, global_dict):
                result[-1] = AppliedFunction(symbol, tok)
                symbol = None
            else:
                # 如果不能应用到任何函数，则将其转换回令牌列表
                result.extend(tok)
        elif tok[0] == NAME:
            # 如果是 NAME 令牌，则作为可能的函数调用的标识
            symbol = tok
            result.append(tok)
        else:
            # 其他情况下，重置 symbol，并将 token 直接添加到结果列表中
            symbol = None
            result.append(tok)
    return result


# 隐式添加 '*' 令牌以处理隐式乘法情况
def _implicit_multiplication(tokens: List[tUnion[TOKEN, AppliedFunction]], local_dict: DICT, global_dict: DICT):
    """Implicitly adds '*' tokens.

    Cases:
    
    """
    根据输入的 tokens 列表生成处理后的结果列表，处理规则如下：
    1. 如果当前 token 是操作符且是点号（'.'），并且下一个 token 是名称，则表示点操作符，不做隐式乘法处理。
    2. 如果当前 token 是应用函数（AppliedFunction）：
       - 如果下一个 token 也是应用函数，则插入乘法操作符 '*'.
       - 如果下一个 token 是左括号 '('，表示应用函数后直接跟随左括号，则：
         - 如果当前函数是 "Function"，则转换为 "Symbol"。
         - 插入乘法操作符 '*'.
       - 如果下一个 token 是名称，则表示隐式应用函数，插入乘法操作符 '*'.
    3. 如果当前 token 是右括号 ')'：
       - 如果下一个 token 是应用函数，则插入乘法操作符 '*'.
       - 如果下一个 token 是名称，则表示隐式应用函数，插入乘法操作符 '*'.
       - 如果下一个 token 是左括号 '('，则表示右括号后直接跟随左括号，插入乘法操作符 '*'.
    4. 如果当前 token 是名称且不可调用：
       - 如果下一个 token 是应用函数或者是可调用的名称，则插入乘法操作符 '*'.
       - 如果下一个 token 是左括号 '('，则表示名称后直接跟随左括号，插入乘法操作符 '*'.
       - 如果下一个 token 是名称，则表示名称后直接跟随名称，插入乘法操作符 '*'.
    5. 如果 tokens 列表不为空，则添加最后一个 token 到结果列表中。
    返回处理后的结果列表。
    """
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    skip = False
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if skip:
            skip = False
            continue
        if tok[0] == OP and tok[1] == '.' and nextTok[0] == NAME:
            # Dotted name. Do not do implicit multiplication
            skip = True
            continue
        if isinstance(tok, AppliedFunction):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                # Applied function followed by an open parenthesis
                if tok.function[1] == "Function":
                    tok.function = (tok.function[0], 'Symbol')
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                # Applied function followed by implicitly applied function
                result.append((OP, '*'))
        else:
            if tok == (OP, ')'):
                if isinstance(nextTok, AppliedFunction):
                    # Close parenthesis followed by an applied function
                    result.append((OP, '*'))
                elif nextTok[0] == NAME:
                    # Close parenthesis followed by an implicitly applied function
                    result.append((OP, '*'))
                elif nextTok == (OP, '('):
                    # Close parenthesis followed by an open parenthesis
                    result.append((OP, '*'))
            elif tok[0] == NAME and not _token_callable(tok, local_dict, global_dict):
                if isinstance(nextTok, AppliedFunction) or \
                    (nextTok[0] == NAME and _token_callable(nextTok, local_dict, global_dict)):
                    # Constant followed by (implicitly applied) function
                    result.append((OP, '*'))
                elif nextTok == (OP, '('):
                    # Constant followed by parenthesis
                    result.append((OP, '*'))
                elif nextTok[0] == NAME:
                    # Constant followed by constant
                    result.append((OP, '*'))
    if tokens:
        result.append(tokens[-1])
    return result
# 在函数内部处理隐式的函数应用，即根据上下文添加括号。
def _implicit_application(tokens: List[tUnion[TOKEN, AppliedFunction]], local_dict: DICT, global_dict: DICT):
    """Adds parentheses as needed after functions."""
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    appendParen = 0  # 需要添加的闭括号数目
    skip = 0  # 延迟添加闭括号前需要跳过的 token 数目
    exponentSkip = False  # 标记是否跳过 token 来处理函数的指数运算
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        # 如果当前 token 是 NAME 而下一个 token 不是运算符、终结符或换行符
        if (tok[0] == NAME and nextTok[0] not in [OP, ENDMARKER, NEWLINE]):
            # 检查当前 NAME 是否可调用，如果是则在后面添加一个 '('
            if _token_callable(tok, local_dict, global_dict, nextTok):  # type: ignore
                result.append((OP, '('))
                appendParen += 1
        # 处理 NAME 后面跟着 '**' 的情况，用于函数指数运算
        elif (tok[0] == NAME and nextTok[0] == OP and nextTok[1] == '**'):
            if _token_callable(tok, local_dict, global_dict):  # type: ignore
                exponentSkip = True
        elif exponentSkip:
            # 如果最后添加的 token 是一个应用函数（即函数指数运算的幂）或者是乘法运算符
            if (isinstance(tok, AppliedFunction)
                or (tok[0] == OP and tok[1] == '*')):
                # 如果下一个 token 是乘法运算符，则不添加任何东西；如果是 '('，则停止跳过 token
                if not (nextTok[0] == OP and nextTok[1] == '*'):
                    if not (nextTok[0] == OP and nextTok[1] == '('):
                        result.append((OP, '('))
                        appendParen += 1
                    exponentSkip = False
        elif appendParen:
            # 如果需要添加闭括号，检查下一个 token 是否是 '^', '**', '*'
            if nextTok[0] == OP and nextTok[1] in ('^', '**', '*'):
                skip = 1
                continue
            # 如果有跳过的 token，继续跳过
            if skip:
                skip -= 1
                continue
            # 添加一个闭括号，并减少需要添加的闭括号数目
            result.append((OP, ')'))
            appendParen -= 1

    # 如果 tokens 不为空，则添加最后一个 token 到结果中
    if tokens:
        result.append(tokens[-1])

    # 最后处理剩余需要添加的闭括号数目，将其全部添加到结果中
    if appendParen:
        result.extend([(OP, ')')] * appendParen)
    return result


```  
# 允许函数进行指数运算，例如 ``cos**2(x)``。
def function_exponentiation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows functions to be exponentiated, e.g. ``cos**2(x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, function_exponentiation)
    >>> transformations = standard_transformations + (function_exponentiation,)
    >>> parse_expr('sin**4(x)', transformations=transformations)
    sin(x)**4
    """
    result: List[TOKEN] = []
    exponent: List[TOKEN] = []
    consuming_exponent = False
    level = 0
    # 遍历 tokens 列表，同时迭代每个元素及其后一个元素
    for tok, nextTok in zip(tokens, tokens[1:]):
        # 检查当前 token 是 NAME 类型且下一个 token 是 OP 类型且其值为 '**'
        if tok[0] == NAME and nextTok[0] == OP and nextTok[1] == '**':
            # 如果 _token_callable 函数返回 True，则进入处理指数部分的模式
            if _token_callable(tok, local_dict, global_dict):
                consuming_exponent = True
        # 如果当前正在处理指数部分
        elif consuming_exponent:
            # 如果当前 token 是 NAME 类型且其值为 'Function'，将其替换为 (NAME, 'Symbol')
            if tok[0] == NAME and tok[1] == 'Function':
                tok = (NAME, 'Symbol')
            # 将当前 token 加入指数列表 exponent
            exponent.append(tok)

            # 检查是否应该结束指数部分，例如遇到 )( 或者 )*(
            if tok[0] == nextTok[0] == OP and tok[1] == ')' and nextTok[1] == '(':
                consuming_exponent = False
            # 如果存在隐式乘法形式，如 )*( ，则结束指数部分并移除最后一个元素
            if tok[0] == nextTok[0] == OP and tok[1] == '*' and nextTok[1] == '(':
                consuming_exponent = False
                del exponent[-1]
            continue
        # 如果存在指数部分并且当前不处于处理指数状态
        elif exponent and not consuming_exponent:
            # 如果当前 token 是 OP 类型
            if tok[0] == OP:
                # 如果是左括号 '('，增加嵌套层级 level
                if tok[1] == '(':
                    level += 1
                # 如果是右括号 ')'，减少嵌套层级 level
                elif tok[1] == ')':
                    level -= 1
            # 如果嵌套层级 level 归零，则当前 token 及指数部分应加入最终结果
            if level == 0:
                result.append(tok)
                result.extend(exponent)
                exponent = []
                continue
        # 将当前 token 加入最终结果 result
        result.append(tok)
    
    # 处理 tokens 列表中的最后一个 token
    if tokens:
        result.append(tokens[-1])
    
    # 将剩余的指数部分加入最终结果 result
    if exponent:
        result.extend(exponent)
    
    # 返回处理后的结果列表
    return result
# 定义一个函数 split_symbols_custom，用于创建一个根据特定条件分割符号名称的转换器。
#
# ``predicate`` 参数应当是一个可调用对象，接受一个字符串参数（符号名称），并返回一个布尔值，
# 表示是否应该对该符号名称进行分割操作。
#
# 例如，为了保留默认行为但避免分割特定符号名称，可以创建如下的 predicate 函数：
#
# >>> from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,
# ... standard_transformations, implicit_multiplication,
# ... split_symbols_custom)
# >>> def can_split(symbol):
# ...     if symbol not in ('list', 'of', 'unsplittable', 'names'):
# ...             return _token_splittable(symbol)
# ...     return False
# ...
# >>> transformation = split_symbols_custom(can_split)
# >>> parse_expr('unsplittable', transformations=standard_transformations +
# ... (transformation, implicit_multiplication))
# unsplittable
#
# 上述示例展示了如何使用 split_symbols_custom 函数创建一个 transformation 对象，
# 并在解析表达式时应用该转换以确保特定符号名称不被分割。
#
# 函数的具体实现内容将在函数体中进行定义。
def split_symbols_custom(predicate: Callable[[str], bool]):
    """Creates a transformation that splits symbol names.

    ``predicate`` should return True if the symbol name is to be split.

    For instance, to retain the default behavior but avoid splitting certain
    symbol names, a predicate like this would work:


    >>> from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,
    ... standard_transformations, implicit_multiplication,
    ... split_symbols_custom)
    >>> def can_split(symbol):
    ...     if symbol not in ('list', 'of', 'unsplittable', 'names'):
    ...             return _token_splittable(symbol)
    ...     return False
    ...
    >>> transformation = split_symbols_custom(can_split)
    >>> parse_expr('unsplittable', transformations=standard_transformations +
    ... (transformation, implicit_multiplication))
    unsplittable
    """
    # 定义名为 _split_symbols 的函数，接受三个参数：tokens（TOKEN 类型的列表）、local_dict（DICT 类型）、global_dict（DICT 类型）
    def _split_symbols(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
        # 初始化空列表 result，用于存储处理后的 tokens
        result: List[TOKEN] = []
        # 初始化标志变量 split 和 split_previous 为 False，用于跟踪是否需要分割处理
        split = False
        split_previous = False

        # 遍历 tokens 列表
        for tok in tokens:
            # 如果 split_previous 为 True，则跳过当前循环，表示跳过前一次分割的符号的闭合括号
            if split_previous:
                split_previous = False
                continue
            split_previous = False

            # 如果当前 token 的第一个元素为 NAME，并且第二个元素是 'Symbol' 或 'Function'
            if tok[0] == NAME and tok[1] in ['Symbol', 'Function']:
                # 设置 split 为 True，表示需要进行分割处理
                split = True

            # 如果 split 为 True，并且当前 token 的第一个元素为 NAME
            elif split and tok[0] == NAME:
                # 获取符号的名称（去除首尾的引号）
                symbol = tok[1][1:-1]

                # 如果符合 predicate 函数的条件
                if predicate(symbol):
                    # 获取之前的 token 类型（Symbol 或 Function）
                    tok_type = result[-2][1]
                    # 删除 result 列表中最后两个元素，即删除对 Symbol 的调用
                    del result[-2:]

                    # 遍历符号的每个字符
                    i = 0
                    while i < len(symbol):
                        char = symbol[i]
                        # 如果字符存在于 local_dict 或 global_dict 中，则作为 NAME 类型添加到 result 中
                        if char in local_dict or char in global_dict:
                            result.append((NAME, "%s" % char))
                        # 如果字符是数字，则将其作为 'Number' 类型添加到 result 中，并加上单引号
                        elif char.isdigit():
                            chars = [char]
                            for j in range(i + 1, len(symbol)):
                                if not symbol[j].isdigit():
                                    i = j - 1
                                    break
                                chars.append(symbol[j])
                            char = ''.join(chars)
                            result.extend([(NAME, 'Number'), (OP, '('),
                                           (NAME, "'%s'" % char), (OP, ')')])
                        # 否则，按照 tok_type 添加符号字符到 result 中
                        else:
                            use = tok_type if i == len(symbol) else 'Symbol'
                            result.extend([(NAME, use), (OP, '('),
                                           (NAME, "'%s'" % char), (OP, ')')])
                        i += 1

                    # 设置 split_previous 为 True，以便跳过原始符号的闭合括号
                    split = False
                    split_previous = True
                    continue

                else:
                    # 如果不符合 predicate 函数的条件，则取消分割处理
                    split = False

            # 将当前 token 添加到 result 列表中
            result.append(tok)

        # 返回处理后的 result 列表
        return result

    # 返回 _split_symbols 函数作为结果
    return _split_symbols
#: 分割符号名称以支持隐式乘法。
#:
#: 旨在使表达式如 ``xyz`` 被解析为 ``x*y*z``。不会分割希腊字符名称，因此 ``theta`` 不会变成 ``t*h*e*t*a``。
#: 通常应与 ``implicit_multiplication`` 一起使用。
split_symbols = split_symbols_custom(_token_splittable)


def implicit_multiplication(tokens: List[TOKEN], local_dict: DICT,
                            global_dict: DICT) -> List[TOKEN]:
    """使大多数情况下乘法运算符变为可选。

    在使用 :func:`implicit_application` 之前使用此函数，否则表达式如 ``sin 2x`` 将被解析为 ``x * sin(2)`` 而不是 ``sin(2*x)``。

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication)
    >>> transformations = standard_transformations + (implicit_multiplication,)
    >>> parse_expr('3 x y', transformations=transformations)
    3*x*y
    """
    # 这些步骤是相互依赖的，因此我们不单独公开它们
    res1 = _group_parentheses(implicit_multiplication)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_multiplication(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result


def implicit_application(tokens: List[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> List[TOKEN]:
    """在某些情况下使函数调用中的括号可选。

    在使用 :func:`implicit_multiplication` 之后使用此函数，否则表达式
    如 ``sin 2x`` 将被解析为 ``x * sin(2)`` 而不是 ``sin(2*x)``。

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_application)
    >>> transformations = standard_transformations + (implicit_application,)
    >>> parse_expr('cot z + csc z', transformations=transformations)
    cot(z) + csc(z)
    """
    res1 = _group_parentheses(implicit_application)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_application(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result


def implicit_multiplication_application(result: List[TOKEN], local_dict: DICT,
                                        global_dict: DICT) -> List[TOKEN]:
    """允许稍微放宽的语法规则。

    - 单参数方法调用的括号是可选的。
    - 乘法是隐式的。
    - 符号名称可以被分割（即符号之间不需要空格）。
    - 函数可以被指数化。

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication_application)
    >>> parse_expr("10sin**2 x**2 + 3xyz + tan theta",
    ... transformations=(standard_transformations +
    ... (implicit_multiplication_application,)))
    # 计算一个复杂的数学表达式，包括多项式和三角函数
    3*x*y*z + 10*sin(x**2)**2 + tan(theta)

    """
    # 依次应用多个转换步骤来处理数学表达式
    # 这些步骤包括符号分割、隐式乘法、隐式函数应用和函数指数
    for step in (split_symbols, implicit_multiplication,
                 implicit_application, function_exponentiation):
        # 使用当前步骤对结果进行转换
        result = step(result, local_dict, global_dict)

    # 返回经过所有步骤处理后的结果
    return result
def auto_symbol(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Inserts calls to ``Symbol``/``Function`` for undefined variables."""
    # 初始化结果列表
    result: List[TOKEN] = []
    # 初始化前一个 token
    prevTok = (-1, '')

    # 将一个空 token 添加到 tokens 列表末尾，以便能够遍历所有 token
    tokens.append((-1, ''))  # so zip traverses all tokens
    # 遍历 tokens 列表，同时获取当前 token 和下一个 token
    for tok, nextTok in zip(tokens, tokens[1:]):
        tokNum, tokVal = tok
        nextTokNum, nextTokVal = nextTok
        # 如果当前 token 是 NAME 类型
        if tokNum == NAME:
            name = tokVal

            # 如果变量名是 Python 中的关键字（如 True, False, None）
            # 或者是保留不转换的属性访问
            # 或者是不转换的关键字参数
            # 或者已经在本地字典中定义
            if (name in ['True', 'False', 'None']
                    or iskeyword(name)
                    or (prevTok[0] == OP and prevTok[1] == '.')
                    or (prevTok[0] == OP and prevTok[1] in ('(', ',')
                        and nextTokNum == OP and nextTokVal == '=')
                    or name in local_dict and local_dict[name] is not None):
                # 直接将当前 token 添加到结果列表中
                result.append((NAME, name))
                continue
            # 如果变量名已经在本地字典中
            elif name in local_dict:
                # 将变量名添加到本地字典的集合中
                local_dict.setdefault(name, set()).add(name)
                # 根据下一个 token 判断是否为函数调用，选择创建 Function 或 Symbol 对象
                if nextTokVal == '(':
                    local_dict[name] = Function(name)
                else:
                    local_dict[name] = Symbol(name)
                # 将当前 token 添加到结果列表中
                result.append((NAME, name))
                continue
            # 如果变量名在全局字典中
            elif name in global_dict:
                obj = global_dict[name]
                # 判断对象是否是特定类型或可调用
                if isinstance(obj, (AssumptionKeys, Basic, type)) or callable(obj):
                    # 将当前 token 添加到结果列表中
                    result.append((NAME, name))
                    continue

            # 如果以上条件都不满足，则进行转换处理
            result.extend([
                (NAME, 'Symbol' if nextTokVal != '(' else 'Function'),
                (OP, '('),
                (NAME, repr(str(name))),
                (OP, ')'),
            ])
        else:
            # 如果当前 token 不是 NAME 类型，直接添加到结果列表中
            result.append((tokNum, tokVal))

        # 更新前一个 token 为当前 token
        prevTok = (tokNum, tokVal)

    # 返回处理后的结果列表
    return result


def lambda_notation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Substitutes "lambda" with its SymPy equivalent Lambda().
    However, the conversion does not take place if only "lambda"
    is passed because that is a syntax error.

    """
    # 初始化结果列表
    result: List[TOKEN] = []
    # 初始化标志位
    flag = False
    # 获取第一个 token 的信息
    toknum, tokval = tokens[0]
    # 获取 tokens 列表的长度
    tokLen = len(tokens)
    # 检查当前 token 是否为 NAME 类型且其值为 'lambda'
    if toknum == NAME and tokval == 'lambda':
        # 检查 token 的长度是否为 2 或者 3 并且第二个 token 是 NEWLINE 类型
        if tokLen == 2 or tokLen == 3 and tokens[1][0] == NEWLINE:
            # 在 Python 3.6.7+ 中，如果输入中没有换行符，则会在 tokens 中添加 NEWLINE
            result.extend(tokens)
        elif tokLen > 2:
            # 如果 token 的数量大于 2
            # 添加以下 token 到 result 中，模拟 Lambda 表达式的语法
            result.extend([
                (NAME, 'Lambda'),
                (OP, '('),
                (OP, '('),
                (OP, ')'),
                (OP, ')'),
            ])
            # 遍历 tokens[1:] 中的每个 token
            for tokNum, tokVal in tokens[1:]:
                # 如果 token 是 OP 类型且其值为 ':'，则将其值修改为 ','
                if tokNum == OP and tokVal == ':':
                    tokVal = ','
                    flag = True
                # 如果未开启 flag 并且 token 是 OP 类型且其值为 '*' 或 '**'，则抛出 TokenError
                if not flag and tokNum == OP and tokVal in ('*', '**'):
                    raise TokenError("Starred arguments in lambda not supported")
                # 如果 flag 已开启，则将 token 插入到倒数第二个位置
                if flag:
                    result.insert(-1, (tokNum, tokVal))
                else:
                    # 否则将 token 插入到倒数第三个位置
                    result.insert(-2, (tokNum, tokVal))
    else:
        # 如果 token 不是 'lambda'，则直接将 tokens 添加到 result 中
        result.extend(tokens)

    # 返回处理后的 result 结果
    return result
# 使用 tokens 列表、local_dict 和 global_dict 参数，实现阶乘记号的标准化处理
def factorial_notation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows standard notation for factorial."""
    # 结果列表初始化为空
    result: List[TOKEN] = []
    # nfactorial 变量用于计数阶乘符号的出现次数
    nfactorial = 0
    # 遍历 tokens 列表中的每个元素 (toknum, tokval)
    for toknum, tokval in tokens:
        # 如果 toknum 是操作符(OP)且 tokval 是 "!"
        if toknum == OP and tokval == "!":
            # 计数阶乘符号的数量
            nfactorial += 1
        # 如果 toknum 是错误标记(ERRORTOKEN)
        elif toknum == ERRORTOKEN:
            # 将操作符赋值给变量 op
            op = tokval
            # 如果 op 是 "!"，增加阶乘计数
            if op == '!':
                nfactorial += 1
            else:
                # 如果 op 不是 "!"，重置阶乘计数并将操作符添加到结果列表中
                nfactorial = 0
                result.append((OP, op))
        else:
            # 如果出现连续的阶乘符号，根据计数选择相应的处理函数添加到结果列表中
            if nfactorial == 1:
                result = _add_factorial_tokens('factorial', result)
            elif nfactorial == 2:
                result = _add_factorial_tokens('factorial2', result)
            elif nfactorial > 2:
                # 如果阶乘符号数量超过2个，抛出 TokenError 异常
                raise TokenError
            # 重置阶乘计数并将当前 (toknum, tokval) 添加到结果列表中
            nfactorial = 0
            result.append((toknum, tokval))
    # 返回处理后的结果列表
    return result


# 使用 tokens 列表、local_dict 和 global_dict 参数，将 XOR 操作符 "^" 转换为指数操作符 "**"
def convert_xor(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Treats XOR, ``^``, as exponentiation, ``**``."""
    # 结果列表初始化为空
    result: List[TOKEN] = []
    # 遍历 tokens 列表中的每个元素 (toknum, tokval)
    for toknum, tokval in tokens:
        # 如果 toknum 是操作符(OP)且 tokval 是 "^"
        if toknum == OP:
            if tokval == '^':
                # 将 "^" 替换为 "**"，并添加到结果列表中
                result.append((OP, '**'))
            else:
                # 如果不是 "^"，将当前元素添加到结果列表中
                result.append((toknum, tokval))
        else:
            # 如果不是操作符，将当前元素添加到结果列表中
            result.append((toknum, tokval))
    # 返回处理后的结果列表
    return result


# 使用 tokens 列表、local_dict 和 global_dict 参数，实现重复小数的表示，如 0.2[1] 表示 0.2111... (19/90)
def repeated_decimals(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Allows 0.2[1] notation to represent the repeated decimal 0.2111... (19/90)

    Run this before auto_number.

    """
    # 结果列表初始化为空
    result: List[TOKEN] = []

    # 定义用于判断字符串是否为数字的函数
    def is_digit(s):
        return all(i in '0123456789_' for i in s)

    # num 变量用于匹配 DECIMAL [ INTEGER ] 结构的部分
    num: List[TOKEN] = []
    # 遍历 tokens 中的每个元组 (toknum, tokval)
    for toknum, tokval in tokens:
        # 如果当前 token 类型为 NUMBER
        if toknum == NUMBER:
            # 如果 num 列表为空，并且 tokval 中包含小数点但不包含 'e' 和 'j'
            if (not num and '.' in tokval and 'e' not in tokval.lower() and
                'j' not in tokval.lower()):
                num.append((toknum, tokval))
            # 或者如果 tokval 是数字并且 num 中已经有两个元素
            elif is_digit(tokval) and len(num) == 2:
                num.append((toknum, tokval))
            # 或者如果 tokval 是数字并且 num 中已经有三个元素，并且最后一个元素也是数字
            elif is_digit(tokval) and len(num) == 3 and is_digit(num[-1][1]):
                # Python 2 将 00123 解析为 '00', '123'
                # Python 3 将 01289 解析为 '012', '89'
                num.append((toknum, tokval))
            else:
                num = []
        # 如果当前 token 类型为 OP
        elif toknum == OP:
            # 如果 tokval 是 '[' 并且 num 中有一个元素
            if tokval == '[' and len(num) == 1:
                num.append((OP, tokval))
            # 或者如果 tokval 是 ']' 并且 num 中至少有三个元素
            elif tokval == ']' and len(num) >= 3:
                num.append((OP, tokval))
            # 或者如果 tokval 是 '.' 并且 num 列表为空
            elif tokval == '.' and not num:
                # 处理 .[1]
                num.append((NUMBER, '0.'))
            else:
                num = []
        else:
            num = []

        # 将当前 token 添加到结果列表 result 中
        result.append((toknum, tokval))

        # 如果 num 列表不为空，并且 num 列表中最后一个元素的值是 ']'
        if num and num[-1][1] == ']':
            # 从 result 中移除 num 列表长度个元素
            result = result[:-len(num)]
            # 将 num[0][1] 按 '.' 分割为 pre 和 post，将 num[2][1] 设置为 repetend
            pre, post = num[0][1].split('.')
            repetend = num[2][1]
            # 如果 num 的长度为 5，则将 num[3][1] 加到 repetend 中
            if len(num) == 5:
                repetend += num[3][1]

            # 清除 pre, post, repetend 中的下划线
            pre = pre.replace('_', '')
            post = post.replace('_', '')
            repetend = repetend.replace('_', '')

            # 根据 post 的长度生成相应个数的 '0'
            zeros = '0'*len(post)
            # 去除 post 和 repetend 中的前导 '0'
            post, repetends = [w.lstrip('0') for w in [post, repetend]]
                                        # or else interpreted as octal

            # 设置 a, b, c, d, e 的值
            a = pre or '0'
            b, c = post or '0', '1' + zeros
            d, e = repetends, ('9'*len(repetend)) + zeros

            # 构建一个序列 seq
            seq = [
                (OP, '('),
                    (NAME, 'Integer'),
                    (OP, '('),
                        (NUMBER, a),
                    (OP, ')'),
                    (OP, '+'),
                    (NAME, 'Rational'),
                    (OP, '('),
                        (NUMBER, b),
                        (OP, ','),
                        (NUMBER, c),
                    (OP, ')'),
                    (OP, '+'),
                    (NAME, 'Rational'),
                    (OP, '('),
                        (NUMBER, d),
                        (OP, ','),
                        (NUMBER, e),
                    (OP, ')'),
                (OP, ')'),
            ]
            # 将序列 seq 扩展到结果列表 result 中
            result.extend(seq)
            # 清空 num 列表
            num = []

    # 返回最终的结果列表 result
    return result
def auto_number(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Converts numeric literals to use SymPy equivalents.

    Complex numbers use ``I``, integer literals use ``Integer``, and float
    literals use ``Float``.

    """
    # 初始化空的结果列表
    result: List[TOKEN] = []

    # 遍历 tokens 中的每个 (toknum, tokval) 元组
    for toknum, tokval in tokens:
        # 如果当前 token 是数字类型
        if toknum == NUMBER:
            # 将当前数字值存储到 number 变量中
            number = tokval
            # 初始化后缀列表
            postfix = []

            # 如果数字以 'j' 或 'J' 结尾，将其替换为 'I' 符号
            if number.endswith(('j', 'J')):
                number = number[:-1]
                postfix = [(OP, '*'), (NAME, 'I')]

            # 如果数字包含小数点或科学计数法，使用 Float 类型
            if '.' in number or (('e' in number or 'E' in number) and
                    not (number.startswith(('0x', '0X')))):
                seq = [(NAME, 'Float'), (OP, '('),
                    (NUMBER, repr(str(number))), (OP, ')')]
            else:
                # 否则，使用 Integer 类型
                seq = [(NAME, 'Integer'), (OP, '('), (
                    NUMBER, number), (OP, ')')]

            # 将转换后的序列添加到结果列表中，并添加可能的后缀
            result.extend(seq + postfix)
        else:
            # 如果当前 token 不是数字类型，直接将其添加到结果列表中
            result.append((toknum, tokval))

    # 返回最终的结果列表
    return result


def rationalize(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Converts floats into ``Rational``. Run AFTER ``auto_number``."""
    # 初始化空的结果列表
    result: List[TOKEN] = []
    # 标记是否已经经过了浮点数的处理
    passed_float = False

    # 遍历 tokens 中的每个 (toknum, tokval) 元组
    for toknum, tokval in tokens:
        # 如果当前 token 的类型是 NAME
        if toknum == NAME:
            # 如果当前 token 的值是 'Float'，则将其替换为 'Rational'
            if tokval == 'Float':
                passed_float = True
                tokval = 'Rational'
            # 将处理后的 token 添加到结果列表中
            result.append((toknum, tokval))
        # 如果已经经过浮点数处理，并且当前 token 是数字类型
        elif passed_float == True and toknum == NUMBER:
            # 将当前数字类型替换为字符串类型
            passed_float = False
            result.append((STRING, tokval))
        else:
            # 否则，直接将当前 token 添加到结果列表中
            result.append((toknum, tokval))

    # 返回最终的结果列表
    return result


def _transform_equals_sign(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Transforms the equals sign ``=`` to instances of Eq.

    This is a helper function for ``convert_equals_signs``.
    Works with expressions containing one equals sign and no
    nesting. Expressions like ``(1=2)=False`` will not work with this
    and should be used with ``convert_equals_signs``.

    Examples: 1=2     to Eq(1,2)
              1*2=x   to Eq(1*2, x)

    This does not deal with function arguments yet.

    """
    # 初始化空的结果列表
    result: List[TOKEN] = []
    # 如果 tokens 中包含 '=' 符号
    if (OP, "=") in tokens:
        # 添加 'Eq' 到结果列表中
        result.append((NAME, "Eq"))
        result.append((OP, "("))
        # 遍历 tokens 中的每个 token
        for token in tokens:
            # 如果当前 token 是 '=' 符号，则替换为 ','
            if token == (OP, "="):
                result.append((OP, ","))
                continue
            # 否则，直接将当前 token 添加到结果列表中
            result.append(token)
        result.append((OP, ")"))
    else:
        # 如果 tokens 中没有 '=' 符号，直接使用原始 tokens
        result = tokens
    # 返回最终的结果列表
    return result


def convert_equals_signs(tokens: List[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> List[TOKEN]:
    """ Transforms all the equals signs ``=`` to instances of Eq.

    Parses the equals signs in the expression and replaces them with
    appropriate Eq instances. Also works with nested equals signs.

    Does not yet play well with function arguments.
    For example, the expression ``(x=y)`` is ambiguous and can be interpreted

    """
    # 调用 _transform_equals_sign 函数处理 tokens 中的等号
    result = _transform_equals_sign(tokens, local_dict, global_dict)
    # 返回处理后的结果列表
    return result
    # 将 tokens 使用 convert_equals_signs 转换，并对括号进行分组
    res1 = _group_parentheses(convert_equals_signs)(tokens, local_dict, global_dict)
    # 应用函数到结果中的每个部分
    res2 = _apply_functions(res1, local_dict, global_dict)
    # 转换等号符号，将等式处理成符合预期的形式
    res3 = _transform_equals_sign(res2, local_dict, global_dict)
    # 扁平化结果，确保结果是一个简单的列表
    result = _flatten(res3)
    # 返回处理后的结果
    return result
# 定义用于 parse_expr 函数的标准转换
# 插入到 SymPy 的 Symbol、Integer 等数据类型的调用，并允许使用标准阶乘符号（例如 x!）
standard_transformations: tTuple[TRANS, ...] \
    = (lambda_notation, auto_symbol, repeated_decimals, auto_number,
       factorial_notation)


def stringify_expr(s: str, local_dict: DICT, global_dict: DICT,
        transformations: tTuple[TRANS, ...]) -> str:
    """
    将字符串 ``s`` 转换为 Python 代码，在 ``local_dict`` 中

    通常应使用 ``parse_expr``。
    """

    tokens = []
    input_code = StringIO(s.strip())
    # 生成输入代码的标记流
    for toknum, tokval, _, _, _ in generate_tokens(input_code.readline):
        tokens.append((toknum, tokval))

    # 对于每个转换函数，应用它来修改标记流
    for transform in transformations:
        tokens = transform(tokens, local_dict, global_dict)

    return untokenize(tokens)


def eval_expr(code, local_dict: DICT, global_dict: DICT):
    """
    评估由 ``stringify_expr`` 生成的 Python 代码。

    通常应使用 ``parse_expr``。
    """
    # 使用局部字典优先来评估表达式
    expr = eval(
        code, global_dict, local_dict)
    return expr


def parse_expr(s: str, local_dict: Optional[DICT] = None,
               transformations: tUnion[tTuple[TRANS, ...], str] \
                   = standard_transformations,
               global_dict: Optional[DICT] = None, evaluate=True):
    """将字符串 ``s`` 转换为 SymPy 表达式，在 ``local_dict`` 中.

    Parameters
    ==========

    s : str
        要解析的字符串.

    local_dict : dict, optional
        用于解析时使用的局部变量字典.

    global_dict : dict, optional
        全局变量字典. 默认情况下，会使用 ``from sympy import *`` 初始化;
        可以提供此参数来覆盖此行为（例如，解析 ``"Q & S"``）.

    transformations : tuple or str
        用于在评估前修改解析表达式标记的转换函数元组. 默认的转换函数包括将数值文字转换为 SymPy 对应的形式，
        将未定义的变量转换为 SymPy 符号，并允许使用标准数学阶乘符号（例如 ``x!``）. 也可以通过字符串选择（参见下面）.

    evaluate : bool, optional
        当为 False 时，保留字符串中的参数顺序，抑制通常会发生的自动简化.（参见示例）

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import parse_expr
    >>> parse_expr("1/2")
    1/2
    >>> type(_)
    <class 'sympy.core.numbers.Half'>
    >>> from sympy.parsing.sympy_parser import standard_transformations,\\
    ... implicit_multiplication_application
    >>> transformations = (standard_transformations +
    ...     (implicit_multiplication_application,))
    # 如果 local_dict 为 None，则设为一个空字典
    if local_dict is None:
        local_dict = {}
    # 如果 local_dict 不是字典类型，则抛出 TypeError 异常
    elif not isinstance(local_dict, dict):
        raise TypeError('expecting local_dict to be a dict')
    # 如果 local_dict 中包含 null 键，抛出 ValueError 异常
    elif null in local_dict:
        raise ValueError('cannot use "" in local_dict')

    # 如果 global_dict 为 None，则设为一个空字典，并引入 sympy 模块中的所有对象
    if global_dict is None:
        global_dict = {}
        exec('from sympy import *', global_dict)

        # 将 Python 内置函数添加到 global_dict 中
        builtins_dict = vars(builtins)
        for name, obj in builtins_dict.items():
            if isinstance(obj, types.BuiltinFunctionType):
                global_dict[name] = obj
        # 将 Max 和 Min 函数添加到 global_dict 中
        global_dict['max'] = Max
        global_dict['min'] = Min

    # 如果 global_dict 不是字典类型，则抛出 TypeError 异常
    elif not isinstance(global_dict, dict):
        raise TypeError('expecting global_dict to be a dict')

    # 如果 transformations 为 None，则设为一个空元组
    transformations = transformations or ()
    # 如果 transformations 是字符串类型
    if isinstance(transformations, str):
        # 如果 transformations 是 'all'，则使用所有的转换
        if transformations == 'all':
            _transformations = T[:]
        # 如果 transformations 是 'implicit'，则使用前6个转换
        elif transformations == 'implicit':
            _transformations = T[:6]
        else:
            # 如果 transformations 是未知的名称，则引发 ValueError 异常
            raise ValueError('unknown transformation group name')
    else:
        # 否则，使用 transformations 本身作为 _transformations
        _transformations = transformations

    # 将表达式转换为字符串形式
    code = stringify_expr(s, local_dict, global_dict, _transformations)

    # 如果不需要评估代码，则编译为表达式对象
    if not evaluate:
        code = compile(evaluateFalse(code), '<string>', 'eval')  # type: ignore

    try:
        # 对编译后的表达式进行评估，使用本地和全局字典
        rv = eval_expr(code, local_dict, global_dict)
        # 恢复对于 null 的定义为中立状态的名称
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        return rv
    except Exception as e:
        # 在捕获到异常后，恢复对于 null 的定义为中立状态的名称，并引发新的异常
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        raise e from ValueError(f"Error from parse_expr with transformed code: {code!r}")
def evaluateFalse(s: str):
    """
    Replaces operators with the SymPy equivalent and sets evaluate=False.
    """
    # 解析输入字符串 `s` 成为抽象语法树（AST）
    node = ast.parse(s)
    # 使用自定义的节点转换器进行 AST 的转换
    transformed_node = EvaluateFalseTransformer().visit(node)
    # 将转换后的节点从 Module 类型变为 Expression 类型
    transformed_node = ast.Expression(transformed_node.body[0].value)

    # 修正 AST 中可能缺失的位置信息
    return ast.fix_missing_locations(transformed_node)


class EvaluateFalseTransformer(ast.NodeTransformer):
    operators = {
        ast.Add: 'Add',
        ast.Mult: 'Mul',
        ast.Pow: 'Pow',
        ast.Sub: 'Add',
        ast.Div: 'Mul',
        ast.BitOr: 'Or',
        ast.BitAnd: 'And',
        ast.BitXor: 'Not',
    }
    functions = (
        'Abs', 'im', 're', 'sign', 'arg', 'conjugate',
        'acos', 'acot', 'acsc', 'asec', 'asin', 'atan',
        'acosh', 'acoth', 'acsch', 'asech', 'asinh', 'atanh',
        'cos', 'cot', 'csc', 'sec', 'sin', 'tan',
        'cosh', 'coth', 'csch', 'sech', 'sinh', 'tanh',
        'exp', 'ln', 'log', 'sqrt', 'cbrt',
    )

    relational_operators = {
        ast.NotEq: 'Ne',
        ast.Lt: 'Lt',
        ast.LtE: 'Le',
        ast.Gt: 'Gt',
        ast.GtE: 'Ge',
        ast.Eq: 'Eq'
    }
    
    # 对比较表达式（Compare）节点进行访问和转换
    def visit_Compare(self, node):
        if node.ops[0].__class__ in self.relational_operators:
            # 获取符号代表的 SymPy 等效类名
            sympy_class = self.relational_operators[node.ops[0].__class__]
            # 访问比较操作符右侧的节点
            right = self.visit(node.comparators[0])
            # 访问比较操作符左侧的节点
            left = self.visit(node.left)
            # 创建一个新的调用节点，用于构建 SymPy 函数调用
            new_node = ast.Call(
                func=ast.Name(id=sympy_class, ctx=ast.Load()),
                args=[left, right],
                keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
            )
            return new_node
        return node

    # 扁平化函数调用，将嵌套的函数调用展开
    def flatten(self, args, func):
        result = []
        for arg in args:
            if isinstance(arg, ast.Call):
                arg_func = arg.func
                if isinstance(arg_func, ast.Call):
                    arg_func = arg_func.func
                if arg_func.id == func:
                    result.extend(self.flatten(arg.args, func))
                else:
                    result.append(arg)
            else:
                result.append(arg)
        return result
    # 访问二元操作节点的方法，处理各种二元操作符
    def visit_BinOp(self, node):
        # 检查节点的操作符类型是否在已知操作符集合中
        if node.op.__class__ in self.operators:
            # 获取对应的SymPy类
            sympy_class = self.operators[node.op.__class__]
            # 访问并处理右操作数和左操作数
            right = self.visit(node.right)
            left = self.visit(node.left)

            # 是否需要反转操作数顺序的标志
            rev = False
            # 如果操作符是减法，则转换为乘法的调用
            if isinstance(node.op, ast.Sub):
                right = ast.Call(
                    func=ast.Name(id='Mul', ctx=ast.Load()),
                    args=[ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1)), right],
                    keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                )
            # 如果操作符是除法
            elif isinstance(node.op, ast.Div):
                # 如果左操作数是一元操作，则交换左右操作数并进行特殊处理
                if isinstance(node.left, ast.UnaryOp):
                    left, right = right, left
                    rev = True
                    left = ast.Call(
                        func=ast.Name(id='Pow', ctx=ast.Load()),
                        args=[left, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))],
                        keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                    )
                else:
                    # 否则，对右操作数进行乘法调用的转换
                    right = ast.Call(
                        func=ast.Name(id='Pow', ctx=ast.Load()),
                        args=[right, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))],
                        keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                    )

            # 如果进行了操作数反转，则恢复操作数顺序
            if rev:
                left, right = right, left
            # 创建新的函数调用节点
            new_node = ast.Call(
                func=ast.Name(id=sympy_class, ctx=ast.Load()),
                args=[left, right],
                keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
            )

            # 如果SymPy类是'Add'或'Mul'，则根据需要展开节点
            if sympy_class in ('Add', 'Mul'):
                # 根据操作符类型展开节点以便处理
                new_node.args = self.flatten(new_node.args, sympy_class)

            # 返回新节点
            return new_node
        # 如果操作符类型不在已知操作符集合中，直接返回节点本身
        return node

    # 访问函数调用节点的方法，进行一般的访问处理
    def visit_Call(self, node):
        # 使用通用访问方法获取新节点
        new_node = self.generic_visit(node)
        # 如果函数名为已知函数集合中的函数名，则添加评估参数
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            new_node.keywords.append(ast.keyword(arg='evaluate', value=ast.Constant(value=False)))
        # 返回新节点
        return new_node
# 定义一个字典 `_transformation`，其中存储了一系列变换函数，索引是整数，对应的值是函数名或者函数对象。
# 注意：该字典允许增加新项，但不允许改变项的顺序。
_transformation = {
    0: lambda_notation,
    1: auto_symbol,
    2: repeated_decimals,
    3: auto_number,
    4: factorial_notation,
    5: implicit_multiplication_application,
    6: convert_xor,
    7: implicit_application,
    8: implicit_multiplication,
    9: convert_equals_signs,
    10: function_exponentiation,
    11: rationalize
}

# 使用列表推导式创建一个字符串 `transformations`，将 `_transformation` 中的每个键值对格式化成字符串
# 形式为 `<索引>: <函数名(函数对象)>`，并以换行符连接成一个字符串。
transformations = '\n'.join('%s: %s' % (i, func_name(f)) for i, f in _transformation.items())

# 定义一个类 `_T`，用于从给定的切片中检索变换函数。
class _T():
    """class to retrieve transformations from a given slice

    EXAMPLES
    ========

    >>> from sympy.parsing.sympy_parser import T, standard_transformations
    >>> assert T[:5] == standard_transformations
    """
    
    def __init__(self):
        # 记录 `_transformation` 字典中键值对的数量
        self.N = len(_transformation)

    def __str__(self):
        # 返回变换字符串 `transformations`
        return transformations

    def __getitem__(self, t):
        # 如果 `t` 不是元组，则转换成元组
        if not type(t) is tuple:
            t = (t,)
        i = []
        for ti in t:
            if type(ti) is int:
                # 如果 `t` 中的元素是整数，则将其转换成对应的索引
                i.append(range(self.N)[ti])
            elif type(ti) is slice:
                # 如果 `t` 中的元素是切片对象，则扩展索引范围
                i.extend(range(*ti.indices(self.N)))
            else:
                # 如果 `t` 中的元素类型不符合预期，则引发异常
                raise TypeError('unexpected slice arg')
        # 返回根据索引获取的变换函数组成的元组
        return tuple([_transformation[_] for _ in i])

# 创建 `_T` 的实例 `T`
T = _T()
```