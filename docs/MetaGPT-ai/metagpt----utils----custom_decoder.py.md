# `MetaGPT\metagpt\utils\custom_decoder.py`

```

# 导入json和re模块
import json
import re
from json import JSONDecodeError
from json.decoder import _decode_uXXXX

# 定义一个正则表达式，用于匹配数字
NUMBER_RE = re.compile(r"(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?", (re.VERBOSE | re.MULTILINE | re.DOTALL))

# 定义一个函数，用于创建JSON扫描器
def py_make_scanner(context):
    # 从上下文中获取相应的解析函数和参数
    parse_object = context.parse_object
    parse_array = context.parse_array
    parse_string = context.parse_string
    match_number = NUMBER_RE.match
    strict = context.strict
    parse_float = context.parse_float
    parse_int = context.parse_int
    parse_constant = context.parse_constant
    object_hook = context.object_hook
    object_pairs_hook = context.object_pairs_hook
    memo = context.memo

    # 定义内部函数，用于扫描JSON字符串
    def _scan_once(string, idx):
        try:
            nextchar = string[idx]
        except IndexError:
            raise StopIteration(idx) from None

        # 根据下一个字符的不同情况进行相应的处理
        if nextchar in ("'", '"'):
            # 处理字符串
            if idx + 2 < len(string) and string[idx + 1] == nextchar and string[idx + 2] == nextchar:
                return parse_string(string, idx + 3, strict, delimiter=nextchar * 3)  # triple quote
            else:
                return parse_string(string, idx + 1, strict, delimiter=nextchar)
        elif nextchar == "{":
            # 处理对象
            return parse_object((string, idx + 1), strict, _scan_once, object_hook, object_pairs_hook, memo)
        elif nextchar == "[":
            # 处理数组
            return parse_array((string, idx + 1), _scan_once)
        elif nextchar == "n" and string[idx : idx + 4] == "null":
            return None, idx + 4
        elif nextchar == "t" and string[idx : idx + 4] == "true":
            return True, idx + 4
        elif nextchar == "f" and string[idx : idx + 5] == "false":
            return False, idx + 5

        # 匹配数字
        m = match_number(string, idx)
        if m is not None:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = parse_float(integer + (frac or "") + (exp or ""))
            else:
                res = parse_int(integer)
            return res, m.end()
        elif nextchar == "N" and string[idx : idx + 3] == "NaN":
            return parse_constant("NaN"), idx + 3
        elif nextchar == "I" and string[idx : idx + 8] == "Infinity":
            return parse_constant("Infinity"), idx + 8
        elif nextchar == "-" and string[idx : idx + 9] == "-Infinity":
            return parse_constant("-Infinity"), idx + 9
        else:
            raise StopIteration(idx)

    # 定义一个函数，用于扫描JSON字符串
    def scan_once(string, idx):
        try:
            return _scan_once(string, idx)
        finally:
            memo.clear()

    return scan_once

# 定义一些常量和正则表达式
FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
STRINGCHUNK = re.compile(r'(.*?)(["\\\x00-\x1f])', FLAGS)
STRINGCHUNK_SINGLEQUOTE = re.compile(r"(.*?)([\'\\\x00-\x1f])", FLAGS)
STRINGCHUNK_TRIPLE_DOUBLE_QUOTE = re.compile(r"(.*?)(\"\"\"|[\\\x00-\x1f])", FLAGS)
STRINGCHUNK_TRIPLE_SINGLEQUOTE = re.compile(r"(.*?)('''|[\\\x00-\x1f])", FLAGS)
BACKSLASH = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
WHITESPACE = re.compile(r"[ \t\n\r]*", FLAGS)
WHITESPACE_STR = " \t\n\r"

# 定义一个函数，用于解析JSON字符串中的对象
def JSONObject(
    s_and_end, strict, scan_once, object_hook, object_pairs_hook, memo=None, _w=WHITESPACE.match, _ws=WHITESPACE_STR
def py_scanstring(s, end, strict=True, _b=BACKSLASH, _m=STRINGCHUNK.match, delimiter='"'):
    # 扫描JSON字符串
    # 参数说明见函数注释
    ...

# 定义一个自定义的JSON解码器
class CustomDecoder(json.JSONDecoder):
    def __init__(
        self,
        *,
        object_hook=None,
        parse_float=None,
        parse_int=None,
        parse_constant=None,
        strict=True,
        object_pairs_hook=None
    ):
        # 初始化父类
        ...

    # 重写decode方法
    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        return super().decode(s)

```