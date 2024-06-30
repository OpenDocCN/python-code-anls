# `D:\src\scipysrc\scipy\scipy\io\_harwell_boeing\_fortran_format_parser.py`

```
"""
Preliminary module to handle Fortran formats for IO. Does not use this outside
scipy.sparse io for now, until the API is deemed reasonable.

The *Format classes handle conversion between Fortran and Python format, and
FortranFormatParser can create *Format instances from raw Fortran format
strings (e.g. '(3I4)', '(10I3)', etc...)
"""
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库


__all__ = ["BadFortranFormat", "FortranFormatParser", "IntFormat", "ExpFormat"]  # 模块的公开接口列表


TOKENS = {
    "LPAR": r"\(",   # 左括号的正则表达式
    "RPAR": r"\)",   # 右括号的正则表达式
    "INT_ID": r"I",  # 整数标识的正则表达式
    "EXP_ID": r"E",  # 指数标识的正则表达式
    "INT": r"\d+",   # 整数的正则表达式
    "DOT": r"\.",    # 点号的正则表达式
}


class BadFortranFormat(SyntaxError):
    pass  # 自定义异常类，用于处理不良的Fortran格式错误


def number_digits(n):
    return int(np.floor(np.log10(np.abs(n))) + 1)  # 计算整数的位数


class IntFormat:
    @classmethod
    def from_number(cls, n, min=None):
        """Given an integer, returns a "reasonable" IntFormat instance to represent
        any number between 0 and n if n > 0, -n and n if n < 0

        Parameters
        ----------
        n : int
            max number one wants to be able to represent
        min : int
            minimum number of characters to use for the format

        Returns
        -------
        res : IntFormat
            IntFormat instance with reasonable (see Notes) computed width

        Notes
        -----
        Reasonable should be understood as the minimal string length necessary
        without losing precision. For example, IntFormat.from_number(1) will
        return an IntFormat instance of width 2, so that any 0 and 1 may be
        represented as 1-character strings without loss of information.
        """
        width = number_digits(n) + 1  # 计算数字n的位数加一
        if n < 0:
            width += 1  # 如果n为负数，位数再加一
        repeat = 80 // width  # 计算重复次数，使得每行最多80个字符
        return cls(width, min, repeat=repeat)  # 返回IntFormat的实例对象

    def __init__(self, width, min=None, repeat=None):
        self.width = width   # 设置宽度属性
        self.repeat = repeat  # 设置重复次数属性
        self.min = min        # 设置最小值属性

    def __repr__(self):
        r = "IntFormat("  # 返回IntFormat对象的字符串表示形式
        if self.repeat:
            r += "%d" % self.repeat
        r += "I%d" % self.width
        if self.min:
            r += ".%d" % self.min
        return r + ")"

    @property
    def fortran_format(self):
        r = "("  # 返回Fortran格式字符串
        if self.repeat:
            r += "%d" % self.repeat
        r += "I%d" % self.width
        if self.min:
            r += ".%d" % self.min
        return r + ")"

    @property
    def python_format(self):
        return "%" + str(self.width) + "d"  # 返回Python格式字符串


class ExpFormat:
    @classmethod
    def from_number(cls, n, min=None):
        """Given a float number, returns a "reasonable" ExpFormat instance to
        represent any number between -n and n.

        Parameters
        ----------
        n : float
            max number one wants to be able to represent
        min : int
            minimum number of characters to use for the format

        Returns
        -------
        res : ExpFormat
            ExpFormat instance with reasonable (see Notes) computed width

        Notes
        -----
        Reasonable should be understood as the minimal string length necessary
        to avoid losing precision.
        """
        # 计算指数格式化字符串的宽度：
        # 符号 + 1|0 + "." +
        # 小数部分的数字个数 + 'E' +
        # 指数的符号 + 指数部分的长度
        finfo = np.finfo(n.dtype)
        # 获取小数部分的数字个数
        n_prec = finfo.precision + 1
        # 获取指数部分的长度
        n_exp = number_digits(np.max(np.abs([finfo.maxexp, finfo.minexp])))
        # 计算总宽度
        width = 1 + 1 + n_prec + 1 + n_exp + 1
        if n < 0:
            width += 1
        # 计算重复次数，使格式化后字符串不超过80个字符
        repeat = int(np.floor(80 / width))
        return cls(width, n_prec, min, repeat=repeat)

    def __init__(self, width, significand, min=None, repeat=None):
        """\
        Parameters
        ----------
        width : int
            number of characters taken by the string (includes space).
        """
        # 初始化函数，设置对象的属性
        self.width = width
        self.significand = significand
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        # 返回对象的字符串表示形式，用于显示
        r = "ExpFormat("
        if self.repeat:
            r += "%d" % self.repeat
        r += "E%d.%d" % (self.width, self.significand)
        if self.min:
            r += "E%d" % self.min
        return r + ")"

    @property
    def fortran_format(self):
        # 返回对象的 Fortran 格式字符串表示形式
        r = "("
        if self.repeat:
            r += "%d" % self.repeat
        r += "E%d.%d" % (self.width, self.significand)
        if self.min:
            r += "E%d" % self.min
        return r + ")"

    @property
    def python_format(self):
        # 返回对象的 Python 格式字符串表示形式
        return "%" + str(self.width-1) + "." + str(self.significand) + "E"
# Token 类用于表示词法分析器生成的单个标记，包含类型、值和位置信息
class Token:
    def __init__(self, type, value, pos):
        self.type = type  # 标记类型
        self.value = value  # 标记的值
        self.pos = pos  # 标记在输入字符串中的位置

    def __str__(self):
        return f"""Token('{self.type}', "{self.value}")"""  # 返回标记的可读字符串表示

    def __repr__(self):
        return self.__str__()  # 返回标记的字符串表示，用于调试和输出

# Tokenizer 类用于将输入字符串解析为标记序列
class Tokenizer:
    def __init__(self):
        self.tokens = list(TOKENS.keys())  # 初始化标记类型列表
        self.res = [re.compile(TOKENS[i]) for i in self.tokens]  # 根据正则表达式编译生成对应的模式对象列表

    def input(self, s):
        self.data = s  # 设置输入字符串
        self.curpos = 0  # 当前解析位置初始化为 0
        self.len = len(s)  # 输入字符串的长度

    def next_token(self):
        curpos = self.curpos  # 获取当前解析位置

        while curpos < self.len:
            for i, r in enumerate(self.res):
                m = r.match(self.data, curpos)  # 尝试从当前位置开始匹配模式
                if m is None:
                    continue
                else:
                    self.curpos = m.end()  # 更新当前解析位置到匹配结束位置
                    return Token(self.tokens[i], m.group(), self.curpos)  # 返回匹配到的标记对象
            raise SyntaxError("Unknown character at position %d (%s)"
                              % (self.curpos, self.data[curpos]))  # 如果无法匹配任何模式，抛出语法错误异常

# FortranFormatParser 类用于解析 Fortran 格式字符串，生成格式化信息
class FortranFormatParser:
    """Parser for Fortran format strings. The parse method returns a *Format
    instance.

    Notes
    -----
    Only ExpFormat (exponential format for floating values) and IntFormat
    (integer format) for now.
    """
    def __init__(self):
        self.tokenizer = Tokenizer()  # 初始化词法分析器

    def parse(self, s):
        self.tokenizer.input(s)  # 设置输入字符串到词法分析器

        tokens = []  # 用于存储解析得到的标记列表

        try:
            while True:
                t = self.tokenizer.next_token()  # 获取下一个标记
                if t is None:
                    break
                else:
                    tokens.append(t)  # 将标记加入列表
            return self._parse_format(tokens)  # 解析标记列表并返回解析结果
        except SyntaxError as e:
            raise BadFortranFormat(str(e)) from e  # 捕获语法错误并抛出自定义异常 BadFortranFormat

    def _get_min(self, tokens):
        next = tokens.pop(0)  # 弹出并返回列表中的第一个元素
        if not next.type == "DOT":
            raise SyntaxError()  # 如果不是 DOT 类型的标记，抛出语法错误
        next = tokens.pop(0)  # 弹出并返回列表中的第二个元素
        return next.value  # 返回标记的值

    def _expect(self, token, tp):
        if not token.type == tp:
            raise SyntaxError()  # 如果标记类型与预期不符，抛出语法错误
    # 解析格式化字符串的方法，根据给定的 tokens 参数进行解析
    def _parse_format(self, tokens):
        # 检查第一个 token 是否为左括号，否则抛出语法错误异常
        if not tokens[0].type == "LPAR":
            raise SyntaxError("Expected left parenthesis at position "
                              "%d (got '%s')" % (0, tokens[0].value))
        # 检查最后一个 token 是否为右括号，否则抛出语法错误异常
        elif not tokens[-1].type == "RPAR":
            raise SyntaxError("Expected right parenthesis at position "
                              "%d (got '%s')" % (len(tokens), tokens[-1].value))

        # 去掉 tokens 列表中的首尾括号，保留中间部分作为待处理的 tokens
        tokens = tokens[1:-1]
        # 提取 tokens 列表中每个 token 的类型，存入 types 列表
        types = [t.type for t in tokens]
        
        # 如果第一个 token 类型为 "INT"，则将其弹出转换为整数作为重复次数 repeat
        if types[0] == "INT":
            repeat = int(tokens.pop(0).value)
        else:
            repeat = None

        # 弹出 tokens 列表中的第一个 token，根据其类型决定进一步操作
        next = tokens.pop(0)
        # 如果第一个 token 类型为 "INT_ID"
        if next.type == "INT_ID":
            # 继续处理 tokens，找到下一个 "INT" 类型的 token，并将其值转换为宽度 width
            next = self._next(tokens, "INT")
            width = int(next.value)
            # 如果 tokens 非空，继续找到下一个 "MIN"，将其值转换为最小值 min
            if tokens:
                min = int(self._get_min(tokens))
            else:
                min = None
            # 返回一个 IntFormat 对象，包括宽度 width、最小值 min、重复次数 repeat
            return IntFormat(width, min, repeat)
        # 如果第一个 token 类型为 "EXP_ID"
        elif next.type == "EXP_ID":
            # 继续处理 tokens，找到下一个 "INT" 类型的 token，并将其值转换为宽度 width
            next = self._next(tokens, "INT")
            width = int(next.value)

            # 继续处理 tokens，找到下一个 "DOT" 类型的 token
            next = self._next(tokens, "DOT")

            # 继续处理 tokens，找到下一个 "INT" 类型的 token，并将其值转换为有效数字部分 significand
            next = self._next(tokens, "INT")
            significand = int(next.value)

            # 如果 tokens 非空，继续找到下一个 "EXP_ID"，将其值转换为指数部分的最小值 min
            if tokens:
                next = self._next(tokens, "EXP_ID")

                next = self._next(tokens, "INT")
                min = int(next.value)
            else:
                min = None
            # 返回一个 ExpFormat 对象，包括宽度 width、有效数字部分 significand、指数部分最小值 min、重复次数 repeat
            return ExpFormat(width, significand, min, repeat)
        else:
            # 如果遇到无效的 formatter 类型，则抛出语法错误异常
            raise SyntaxError("Invalid formatter type %s" % next.value)

    # 辅助方法：从 tokens 中取出下一个符合类型 tp 的 token，并返回该 token
    def _next(self, tokens, tp):
        if not len(tokens) > 0:
            raise SyntaxError()
        # 弹出 tokens 列表中的第一个 token，并检查其类型是否符合 tp
        next = tokens.pop(0)
        self._expect(next, tp)
        return next
```