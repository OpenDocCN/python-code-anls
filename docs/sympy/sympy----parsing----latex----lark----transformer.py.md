# `D:\src\scipysrc\sympy\sympy\parsing\latex\lark\transformer.py`

```
import re  # 导入正则表达式模块

import sympy  # 导入 sympy 符号计算库
from sympy.external import import_module  # 导入 sympy 的外部模块导入函数
from sympy.parsing.latex.errors import LaTeXParsingError  # 导入 LaTeX 解析错误

lark = import_module("lark")  # 尝试导入 lark 解析器模块

if lark:
    from lark import Transformer, Token  # 如果导入成功，从 lark 中导入 Transformer 和 Token 类型
else:
    class Transformer:  # 如果导入失败，定义一个空的 Transformer 类
        def transform(self, *args):
            pass


    class Token:  # 如果导入失败，定义一个空的 Token 类
        pass


# noinspection PyPep8Naming,PyMethodMayBeStatic
class TransformToSymPyExpr(Transformer):
    """返回一个 SymPy 表达式，通过遍历传递给 ``transform()`` 函数的 ``lark.Tree`` 生成。

    Notes
    =====

    **This class is never supposed to be used directly.**
    这个类不应直接使用。

    要调整这个类的行为，必须对其进行子类化，然后在进行必要的修改后，将新类的名称通过构造函数的
    ``transformer`` 参数传递给 :py:class:`LarkLaTeXParser` 类。

    Parameters
    ==========

    visit_tokens : bool, optional
        有关此选项的详细信息，请参见 `此处 <https://lark-parser.readthedocs.io/en/latest/visitors.html#lark.visitors.Transformer>`_。

        注意，默认解析器需要将此选项设置为 ``True``。
    """

    SYMBOL = sympy.Symbol  # 定义变量 SYMBOL 为 sympy 符号对象
    DIGIT = sympy.core.numbers.Integer  # 定义变量 DIGIT 为 sympy 整数对象

    def CMD_INFTY(self, tokens):
        # 处理 CMD_INFTY 规则，返回 sympy 无穷大符号
        return sympy.oo

    def GREEK_SYMBOL(self, tokens):
        # 处理 GREEK_SYMBOL 规则，根据正则表达式处理希腊字母符号
        variable_name = re.sub("var", "", tokens[1:])
        return sympy.Symbol(variable_name)

    def BASIC_SUBSCRIPTED_SYMBOL(self, tokens):
        # 处理 BASIC_SUBSCRIPTED_SYMBOL 规则，处理基本的带下标符号
        symbol, sub = tokens.value.split("_")
        if sub.startswith("{"):
            return sympy.Symbol("%s_{%s}" % (symbol, sub[1:-1]))
        else:
            return sympy.Symbol("%s_{%s}" % (symbol, sub))

    def GREEK_SUBSCRIPTED_SYMBOL(self, tokens):
        # 处理 GREEK_SUBSCRIPTED_SYMBOL 规则，处理带下标的希腊字母符号
        greek_letter, sub = tokens.value.split("_")
        greek_letter = re.sub("var", "", greek_letter[1:])
        if sub.startswith("{"):
            return sympy.Symbol("%s_{%s}" % (greek_letter, sub[1:-1]))
        else:
            return sympy.Symbol("%s_{%s}" % (greek_letter, sub))

    def SYMBOL_WITH_GREEK_SUBSCRIPT(self, tokens):
        # 处理 SYMBOL_WITH_GREEK_SUBSCRIPT 规则，处理带希腊字母下标的符号
        symbol, sub = tokens.value.split("_")
        if sub.startswith("{"):
            greek_letter = sub[2:-1]
            greek_letter = re.sub("var", "", greek_letter)
            return sympy.Symbol("%s_{%s}" % (symbol, greek_letter))
        else:
            greek_letter = sub[1:]
            greek_letter = re.sub("var", "", greek_letter)
            return sympy.Symbol("%s_{%s}" % (symbol, greek_letter))

    def multi_letter_symbol(self, tokens):
        # 处理 multi_letter_symbol 规则，返回多字母符号的 sympy 符号对象
        return sympy.Symbol(tokens[2])
    # 定义一个方法，接受一个tokens列表作为参数，处理数字类型的token
    def number(self, tokens):
        # 如果tokens[0]中包含小数点，则将其转换为浮点数对象返回
        if "." in tokens[0]:
            return sympy.core.numbers.Float(tokens[0])
        else:
            # 否则将其转换为整数对象返回
            return sympy.core.numbers.Integer(tokens[0])

    # 定义一个方法，接受一个tokens列表作为参数，处理latex字符串类型的token
    def latex_string(self, tokens):
        # 直接返回tokens中的第一个元素，表示该token是一个latex字符串
        return tokens[0]

    # 定义一个方法，接受一个tokens列表作为参数，处理圆括号分组类型的token
    def group_round_parentheses(self, tokens):
        # 直接返回tokens中的第二个元素，表示该token是一个圆括号分组
        return tokens[1]

    # 定义一个方法，接受一个tokens列表作为参数，处理方括号分组类型的token
    def group_square_brackets(self, tokens):
        # 直接返回tokens中的第二个元素，表示该token是一个方括号分组
        return tokens[1]

    # 定义一个方法，接受一个tokens列表作为参数，处理花括号分组类型的token
    def group_curly_parentheses(self, tokens):
        # 直接返回tokens中的第二个元素，表示该token是一个花括号分组
        return tokens[1]

    # 定义一个方法，接受一个tokens列表作为参数，处理等号表达式类型的token
    def eq(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的等式对象
        return sympy.Eq(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理不等号表达式类型的token
    def ne(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的不等式对象
        return sympy.Ne(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理小于号表达式类型的token
    def lt(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的小于关系对象
        return sympy.Lt(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理小于等于号表达式类型的token
    def lte(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的小于等于关系对象
        return sympy.Le(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理大于号表达式类型的token
    def gt(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的大于关系对象
        return sympy.Gt(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理大于等于号表达式类型的token
    def gte(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的大于等于关系对象
        return sympy.Ge(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理加法表达式类型的token
    def add(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的加法表达式对象
        return sympy.Add(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理减法表达式类型的token
    def sub(self, tokens):
        # 如果tokens的长度为2，表示单目负号，返回tokens[1]的负数
        if len(tokens) == 2:
            return -tokens[1]
        # 如果tokens的长度为3，表示二元减法，返回tokens[0]减去tokens[2]
        elif len(tokens) == 3:
            return sympy.Add(tokens[0], -tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理乘法表达式类型的token
    def mul(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的乘法表达式对象
        return sympy.Mul(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理除法表达式类型的token
    def div(self, tokens):
        # 返回一个用tokens中第0个乘以tokens中第2个的倒数构成的除法表达式对象
        return sympy.Mul(tokens[0], sympy.Pow(tokens[2], -1))

    # 定义一个方法，接受一个tokens列表作为参数，处理相邻表达式类型的token
    def adjacent_expressions(self, tokens):
        # 导入量子物理模块中的Bra和Ket类
        from sympy.physics.quantum import Bra, Ket
        # 如果tokens中第0个是Ket对象且tokens中第1个是Bra对象，则返回它们的外积
        if isinstance(tokens[0], Ket) and isinstance(tokens[1], Bra):
            from sympy.physics.quantum import OuterProduct
            return OuterProduct(tokens[0], tokens[1])
        # 如果tokens中第0个是符号"d"，则很可能是微分操作，返回tokens[0]和tokens[1]
        elif tokens[0] == sympy.Symbol("d"):
            return tokens[0], tokens[1]
        # 如果tokens中第0个是元组，那么很可能是导数操作，返回tokens[1]对tokens[0][1]的导数
        elif isinstance(tokens[0], tuple):
            return sympy.Derivative(tokens[1], tokens[0][1])
        # 否则返回tokens中第0个和第1个元素的乘积
        else:
            return sympy.Mul(tokens[0], tokens[1])

    # 定义一个方法，接受一个tokens列表作为参数，处理上标表达式类型的token
    def superscript(self, tokens):
        # 返回一个用tokens中第0个和第2个元素构成的乘方表达式对象
        return sympy.Pow(tokens[0], tokens[2])

    # 定义一个方法，接受一个tokens列表作为参数，处理分数表达式类型的token
    def fraction(self, tokens):
        numerator = tokens[1]
        # 如果tokens中第2个是元组，则返回"derivative"和tokens中第2个元组的第二个元素
        if isinstance(tokens[2], tuple):
            _, variable = tokens[2]
            return "derivative", variable
        else:
            denominator = tokens[2]
            # 否则返回分子乘以分母的倒数构成的乘法表达式对象
            return sympy.Mul(numerator, sympy.Pow(denominator, -1))

    # 定义一个方法，接受一个tokens列表作为参数，处理二项式表达式类型的token
    def binomial(self, tokens):
        # 返回一个用tokens中第1个和第2个元素构成的二项式系数对象
        return sympy.binomial(tokens[1], tokens[2])
    # 定义一个方法，用于处理带有大括号和圆括号的整数分组，接受 tokens 参数作为输入
    def group_curly_parentheses_int(self, tokens):
        # 如果 tokens 的长度为 3，则返回一个元组，包含分子表达式和积分变量
        if len(tokens) == 3:
            return 1, tokens[1]
        # 如果 tokens 的长度为 4，则返回分子和除数
        elif len(tokens) == 4:
            return tokens[1], tokens[2]
        # 如果以上条件都不满足，则抛出错误，因为没有其他可能性

    # 定义一个方法，处理特殊的分数，接受 tokens 参数作为输入
    def special_fraction(self, tokens):
        # 从 tokens 中提取分子和积分变量
        numerator, variable = tokens[1]
        denominator = tokens[2]

        # 返回 sympy 格式的分数，分子是 numerator，分母是 denominator 的乘法逆
        return sympy.Mul(numerator, sympy.Pow(denominator, -1)), variable

    # 定义一个方法，处理带有特殊分数的积分，接受 tokens 参数作为输入
    def integral_with_special_fraction(self, tokens):
        underscore_index = None
        caret_index = None

        # 如果 "_" 在 tokens 中，找到它的索引
        if "_" in tokens:
            underscore_index = tokens.index("_")

        # 如果 "^" 在 tokens 中，找到它的索引
        if "^" in tokens:
            caret_index = tokens.index("^")

        # 根据 underscore_index 和 caret_index 提取积分的下限和上限
        lower_bound = tokens[underscore_index + 1] if underscore_index else None
        upper_bound = tokens[caret_index + 1] if caret_index else None

        # 检查积分上下限的情况，抛出错误，如果只有下限或只有上限
        if lower_bound is not None and upper_bound is None:
            raise LaTeXParsingError("Integral lower bound was found, but upper bound was not found.")

        if upper_bound is not None and lower_bound is None:
            raise LaTeXParsingError("Integral upper bound was found, but lower bound was not found.")

        # 提取被积函数和微分变量
        integrand, differential_variable = tokens[-1]

        # 如果存在下限，返回定积分
        if lower_bound is not None:
            return sympy.Integral(integrand, (differential_variable, lower_bound, upper_bound))
        else:
            # 否则返回不定积分
            return sympy.Integral(integrand, differential_variable)
    def group_curly_parentheses_special(self, tokens):
        # 找到 tokens 中 "_" 的索引位置
        underscore_index = tokens.index("_")
        # 找到 tokens 中 "^" 的索引位置
        caret_index = tokens.index("^")

        # 给定我们解析的表达式类型，假设底限始终使用大括号括起其参数。
        # 这是因为我们不支持将无约束的求和转换为 SymPy 表达式。

        # 首先，我们隔离底限
        left_brace_index = tokens.index("{", underscore_index)
        right_brace_index = tokens.index("}", underscore_index)

        # 提取底限的内容
        bottom_limit = tokens[left_brace_index + 1: right_brace_index]

        # 接下来，我们隔离上限
        top_limit = tokens[caret_index + 1:]

        # 下面的代码将支持类似 `\sum_{n = 0}^{n = 5} n^2` 这样的情况
        # if "{" in top_limit:
        #     left_brace_index = tokens.index("{", caret_index)
        #     if left_brace_index != -1:
        #         # 如果字符串中有左括号，则需要找到对应的右括号
        #         right_brace_index = tokens.index("}", caret_index)
        #         top_limit = tokens[left_brace_index + 1: right_brace_index]

        # print(f"top  limit = {top_limit}")

        # 底限的索引变量
        index_variable = bottom_limit[0]
        # 底限的下界
        lower_limit = bottom_limit[-1]
        # 上限的上界，目前索引始终为0
        upper_limit = top_limit[0]

        # print(f"return value = ({index_variable}, {lower_limit}, {upper_limit})")

        # 返回解析得到的索引变量、下界和上界
        return index_variable, lower_limit, upper_limit

    def summation(self, tokens):
        # 返回 tokens[2] 和 tokens[1] 构成的 SymPy 求和表达式
        return sympy.Sum(tokens[2], tokens[1])

    def product(self, tokens):
        # 返回 tokens[2] 和 tokens[1] 构成的 SymPy 积分表达式
        return sympy.Product(tokens[2], tokens[1])

    def limit_dir_expr(self, tokens):
        # 找到 tokens 中 "^" 的索引位置
        caret_index = tokens.index("^")

        # 如果 tokens 中包含 "{"
        if "{" in tokens:
            # 找到 "{" 的索引位置
            left_curly_brace_index = tokens.index("{", caret_index)
            # 获取方向信息
            direction = tokens[left_curly_brace_index + 1]
        else:
            # 否则，方向默认为 tokens[caret_index + 1]
            direction = tokens[caret_index + 1]

        # 根据方向返回 tokens[0] 和方向信息
        if direction == "+":
            return tokens[0], "+"
        elif direction == "-":
            return tokens[0], "-"
        else:
            return tokens[0], "+-"

    def group_curly_parentheses_lim(self, tokens):
        # tokens[1] 是限制变量
        limit_variable = tokens[1]
        # 如果 tokens[3] 是元组，则解析出目标和方向
        if isinstance(tokens[3], tuple):
            destination, direction = tokens[3]
        else:
            # 否则，目标为 tokens[3]，方向为 "+-"
            destination = tokens[3]
            direction = "+-"

        # 返回限制变量、目标和方向
        return limit_variable, destination, direction

    def limit(self, tokens):
        # 解析 tokens[2] 得到限制变量、目标和方向
        limit_variable, destination, direction = tokens[2]

        # 返回 SymPy 极限表达式
        return sympy.Limit(tokens[-1], limit_variable, destination, direction)

    def differential(self, tokens):
        # 返回 tokens[1] 作为微分表达式
        return tokens[1]

    def derivative(self, tokens):
        # 返回 tokens[5] 作为导数表达式
        return sympy.Derivative(tokens[-1], tokens[5])
    # 定义一个方法，接受一个名为 tokens 的参数列表
    def list_of_expressions(self, tokens):
        # 如果 tokens 的长度为 1，直接返回 tokens，因为 function_applied 节点期望得到一个列表
        if len(tokens) == 1:
            return tokens
        else:
            # 定义一个内部函数 remove_tokens，用于过滤 tokens 中的特定类型的 Token 对象
            def remove_tokens(args):
                # 如果 args 是 Token 对象
                if isinstance(args, Token):
                    # 如果 Token 的类型不是 "COMMA"，则抛出 LaTeXParsingError 异常
                    raise LaTeXParsingError("A comma token was expected, but some other token was encountered.")
                    return False
                # 如果 args 不是 Token 对象，保留它
                return True

            # 使用 remove_tokens 函数过滤 tokens，并返回结果
            return filter(remove_tokens, tokens)

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示应用函数
    def function_applied(self, tokens):
        # 调用 sympy 库中的 Function 函数，传入 tokens[0] 作为函数名，tokens[2] 作为参数列表，并返回结果
        return sympy.Function(tokens[0])(*tokens[2])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示最小值函数
    def min(self, tokens):
        # 调用 sympy 库中的 Min 函数，传入 tokens[2] 作为参数列表，并返回结果
        return sympy.Min(*tokens[2])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示最大值函数
    def max(self, tokens):
        # 调用 sympy 库中的 Max 函数，传入 tokens[2] 作为参数列表，并返回结果
        return sympy.Max(*tokens[2])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示 Bra（布拉）算符
    def bra(self, tokens):
        # 导入 sympy.physics.quantum 中的 Bra 类，并使用 tokens[1] 创建一个 Bra 对象并返回
        from sympy.physics.quantum import Bra
        return Bra(tokens[1])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示 Ket（凯特）算符
    def ket(self, tokens):
        # 导入 sympy.physics.quantum 中的 Ket 类，并使用 tokens[1] 创建一个 Ket 对象并返回
        from sympy.physics.quantum import Ket
        return Ket(tokens[1])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示内积
    def inner_product(self, tokens):
        # 导入 sympy.physics.quantum 中的 Bra、Ket 和 InnerProduct 类
        from sympy.physics.quantum import Bra, Ket, InnerProduct
        # 创建一个 InnerProduct 对象，其中使用 Bra(tokens[1]) 和 Ket(tokens[3]) 作为参数
        return InnerProduct(Bra(tokens[1]), Ket(tokens[3]))

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示正弦函数
    def sin(self, tokens):
        # 调用 sympy 库中的 sin 函数，传入 tokens[1] 作为参数，并返回结果
        return sympy.sin(tokens[1])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示余弦函数
    def cos(self, tokens):
        # 调用 sympy 库中的 cos 函数，传入 tokens[1] 作为参数，并返回结果
        return sympy.cos(tokens[1])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示正切函数
    def tan(self, tokens):
        # 调用 sympy 库中的 tan 函数，传入 tokens[1] 作为参数，并返回结果
        return sympy.tan(tokens[1])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示余切函数
    def csc(self, tokens):
        # 调用 sympy 库中的 csc 函数，传入 tokens[1] 作为参数，并返回结果
        return sympy.csc(tokens[1])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示正割函数
    def sec(self, tokens):
        # 调用 sympy 库中的 sec 函数，传入 tokens[1] 作为参数，并返回结果
        return sympy.sec(tokens[1])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示余割函数
    def cot(self, tokens):
        # 调用 sympy 库中的 cot 函数，传入 tokens[1] 作为参数，并返回结果
        return sympy.cot(tokens[1])

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示正弦函数的幂
    def sin_power(self, tokens):
        # 从 tokens 中获取指数值
        exponent = tokens[2]
        # 如果指数值为 -1，调用 sympy 库中的 asin 函数，传入 tokens[-1] 作为参数，并返回结果
        if exponent == -1:
            return sympy.asin(tokens[-1])
        else:
            # 否则，调用 sympy 库中的 Pow 函数，对 sin(tokens[-1]) 进行 exponent 次幂运算，并返回结果
            return sympy.Pow(sympy.sin(tokens[-1]), exponent)

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示余弦函数的幂
    def cos_power(self, tokens):
        # 从 tokens 中获取指数值
        exponent = tokens[2]
        # 如果指数值为 -1，调用 sympy 库中的 acos 函数，传入 tokens[-1] 作为参数，并返回结果
        if exponent == -1:
            return sympy.acos(tokens[-1])
        else:
            # 否则，调用 sympy 库中的 Pow 函数，对 cos(tokens[-1]) 进行 exponent 次幂运算，并返回结果
            return sympy.Pow(sympy.cos(tokens[-1]), exponent)

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示正切函数的幂
    def tan_power(self, tokens):
        # 从 tokens 中获取指数值
        exponent = tokens[2]
        # 如果指数值为 -1，调用 sympy 库中的 atan 函数，传入 tokens[-1] 作为参数，并返回结果
        if exponent == -1:
            return sympy.atan(tokens[-1])
        else:
            # 否则，调用 sympy 库中的 Pow 函数，对 tan(tokens[-1]) 进行 exponent 次幂运算，并返回结果
            return sympy.Pow(sympy.tan(tokens[-1]), exponent)

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示余切函数的幂
    def csc_power(self, tokens):
        # 从 tokens 中获取指数值
        exponent = tokens[2]
        # 如果指数值为 -1，调用 sympy 库中的 acsc 函数，传入 tokens[-1] 作为参数，并返回结果
        if exponent == -1:
            return sympy.acsc(tokens[-1])
        else:
            # 否则，调用 sympy 库中的 Pow 函数，对 csc(tokens[-1]) 进行 exponent 次幂运算，并返回结果
            return sympy.Pow(sympy.csc(tokens[-1]), exponent)

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示正割函数的幂
    def sec_power(self, tokens):
        # 从 tokens 中获取指数值
        exponent = tokens[2]
        # 如果指数值为 -1，调用 sympy 库中的 asec 函数，传入 tokens[-1] 作为参数，并返回结果
        if exponent == -1:
            return sympy.asec(tokens[-1])
        else:
            # 否则，调用 sympy 库中的 Pow 函数，对 sec(tokens[-1]) 进行 exponent 次幂运算，并返回结果
            return sympy.Pow(sympy.sec(tokens[-1]), exponent)

    # 定义一个方法，接受一个名为 tokens 的参数列表，表示余割函数的幂
    def cot_power(self, tokens):
        # 从 tokens 中获取指数值
        exponent = tokens[2]
    # 计算 tokens[1] 的反正切值
    def arctan(self, tokens):
        return sympy.atan(tokens[1])

    # 计算 tokens[1] 的反余割值
    def arccsc(self, tokens):
        return sympy.acsc(tokens[1])

    # 计算 tokens[1] 的反正割值
    def arcsec(self, tokens):
        return sympy.asec(tokens[1])

    # 计算 tokens[1] 的反余切值
    def arccot(self, tokens):
        return sympy.acot(tokens[1])

    # 计算 tokens[1] 的双曲正弦值
    def sinh(self, tokens):
        return sympy.sinh(tokens[1])

    # 计算 tokens[1] 的双曲余弦值
    def cosh(self, tokens):
        return sympy.cosh(tokens[1])

    # 计算 tokens[1] 的双曲正切值
    def tanh(self, tokens):
        return sympy.tanh(tokens[1])

    # 计算 tokens[1] 的反双曲正弦值
    def asinh(self, tokens):
        return sympy.asinh(tokens[1])

    # 计算 tokens[1] 的反双曲余弦值
    def acosh(self, tokens):
        return sympy.acosh(tokens[1])

    # 计算 tokens[1] 的反双曲正切值
    def atanh(self, tokens):
        return sympy.atanh(tokens[1])

    # 计算 tokens[1] 的绝对值
    def abs(self, tokens):
        return sympy.Abs(tokens[1])

    # 计算 tokens[1] 的下取整值
    def floor(self, tokens):
        return sympy.floor(tokens[1])

    # 计算 tokens[1] 的上取整值
    def ceil(self, tokens):
        return sympy.ceiling(tokens[1])

    # 计算 tokens[0] 的阶乘
    def factorial(self, tokens):
        return sympy.factorial(tokens[0])

    # 计算 tokens[1] 的共轭复数
    def conjugate(self, tokens):
        return sympy.conjugate(tokens[1])

    # 计算 tokens[1] 的平方根
    def square_root(self, tokens):
        if len(tokens) == 2:
            # 如果 tokens 长度为2，说明没有方括号参数
            return sympy.sqrt(tokens[1])
        elif len(tokens) == 3:
            # 如果 tokens 长度为3，说明有方括号参数
            return sympy.root(tokens[2], tokens[1])

    # 计算 tokens[1] 的指数值
    def exponential(self, tokens):
        return sympy.exp(tokens[1])

    # 计算对数，根据 tokens[0].type 的不同选择不同的基数
    def log(self, tokens):
        if tokens[0].type == "FUNC_LG":
            # 如果 tokens[0].type 为 "FUNC_LG"，返回以10为底的对数
            return sympy.log(tokens[1], 10)
        elif tokens[0].type == "FUNC_LN":
            # 如果 tokens[0].type 为 "FUNC_LN"，返回以e为底的自然对数
            return sympy.log(tokens[1])
        elif tokens[0].type == "FUNC_LOG":
            # 如果 tokens[0].type 为 "FUNC_LOG"，根据是否指定基数返回对数
            if "_" in tokens:
                # 如果指定了基数，返回以指定基数为底的对数
                return sympy.log(tokens[3], tokens[2])
            else:
                # 如果未指定基数，返回以e为底的自然对数
                return sympy.log(tokens[1])

    # 从字符串 s 中提取微分符号，可能是 "d", r"\text{d}", 或 r"\mathrm{d}"
    def _extract_differential_symbol(self, s: str):
        differential_symbols = {"d", r"\text{d}", r"\mathrm{d}"}

        # 查找并返回 s 中的微分符号
        differential_symbol = next((symbol for symbol in differential_symbols if symbol in s), None)

        return differential_symbol
```