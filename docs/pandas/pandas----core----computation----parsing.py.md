# `D:\src\scipysrc\pandas\pandas\core\computation\parsing.py`

```
"""
:func:`~pandas.eval` source string parsing functions
"""

# 从未来导入注解，允许在类型提示中使用字符串
from __future__ import annotations

# 导入 StringIO 类
from io import StringIO
# 导入 iskeyword 函数，用于检查是否为 Python 关键字
from keyword import iskeyword
# 导入 token 模块，用于处理 Python 代码的词法分析
import token
# 导入 tokenize 模块，用于更详细的 Python 代码分析
import tokenize
# 导入 TYPE_CHECKING 常量，用于类型检查时的条件判断
from typing import TYPE_CHECKING

# 如果正在进行类型检查，则导入 Hashable 和 Iterator 类型
if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

# 定义一个 Python 分析器很少使用的 token 值
BACKTICK_QUOTED_STRING = 100


def create_valid_python_identifier(name: str) -> str:
    """
    Create valid Python identifiers from any string.

    Check if name contains any special characters. If it contains any
    special characters, the special characters will be replaced by
    a special string and a prefix is added.

    Raises
    ------
    SyntaxError
        If the returned name is not a Python valid identifier, raise an exception.
        This can happen if there is a hashtag in the name, as the tokenizer will
        than terminate and not find the backtick.
        But also for characters that fall out of the range of (U+0001..U+007F).
    """
    # 如果 name 已经是有效的 Python 标识符并且不是 Python 关键字，则直接返回
    if name.isidentifier() and not iskeyword(name):
        return name

    # 创建一个字典，用于存储特殊字符及其替换字符串
    special_characters_replacements = {
        char: f"_{token.tok_name[tokval]}_"
        for char, tokval in tokenize.EXACT_TOKEN_TYPES.items()
    }
    # 更新字典，添加更多特殊字符的替换规则
    special_characters_replacements.update(
        {
            " ": "_",
            "?": "_QUESTIONMARK_",
            "!": "_EXCLAMATIONMARK_",
            "$": "_DOLLARSIGN_",
            "€": "_EUROSIGN_",
            "°": "_DEGREESIGN_",
            "'": "_SINGLEQUOTE_",
            '"': "_DOUBLEQUOTE_",
        }
    )

    # 根据特殊字符替换规则，对 name 进行替换处理
    name = "".join([special_characters_replacements.get(char, char) for char in name])
    # 添加特定前缀以确保生成的名称是有效的 Python 标识符
    name = f"BACKTICK_QUOTED_STRING_{name}"

    # 如果处理后的名称不是有效的 Python 标识符，则抛出 SyntaxError 异常
    if not name.isidentifier():
        raise SyntaxError(f"Could not convert '{name}' to a valid Python identifier.")

    return name


def clean_backtick_quoted_toks(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Clean up a column name if surrounded by backticks.

    Backtick quoted string are indicated by a certain tokval value. If a string
    is a backtick quoted token it will processed by
    :func:`_create_valid_python_identifier` so that the parser can find this
    string when the query is executed.
    In this case the tok will get the NAME tokval.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tok : Tuple[int, str]
        Either the input or token or the replacement values
    """
    # 接受一个元组作为输入，元组包含两个部分，第一个部分是整数，第二个部分是字符串
    toknum, tokval = tok
    # 如果当前 token 类型为 BACKTICK_QUOTED_STRING
    if toknum == BACKTICK_QUOTED_STRING:
        # 返回 token 类型为 tokenize.NAME，并将 tokval 转换为有效的 Python 标识符
        return tokenize.NAME, create_valid_python_identifier(tokval)
    # 如果当前 token 类型不是 BACKTICK_QUOTED_STRING，则直接返回当前的 toknum 和 tokval
    return toknum, tokval
# 定义一个函数，用于清理列名，使其符合 Python 标识符的命名规范
def clean_column_name(name: Hashable) -> Hashable:
    """
    Function to emulate the cleaning of a backtick quoted name.

    The purpose for this function is to see what happens to the name of
    identifier if it goes to the process of being parsed a Python code
    inside a backtick quoted string and than being cleaned
    (removed of any special characters).

    Parameters
    ----------
    name : hashable
        Name to be cleaned.

    Returns
    -------
    name : hashable
        Returns the name after tokenizing and cleaning.

    Notes
    -----
        For some cases, a name cannot be converted to a valid Python identifier.
        In that case :func:`tokenize_string` raises a SyntaxError.
        In that case, we just return the name unmodified.

        If this name was used in the query string (this makes the query call impossible)
        an error will be raised by :func:`tokenize_backtick_quoted_string` instead,
        which is not caught and propagates to the user level.
    """
    try:
        # 使用 backtick 引用字符串的方式对 name 进行标记化
        tokenized = tokenize_string(f"`{name}`")
        # 获取标记化后的值
        tokval = next(tokenized)[1]
        # 创建一个有效的 Python 标识符
        return create_valid_python_identifier(tokval)
    except SyntaxError:
        # 如果出现 SyntaxError，则返回原始的 name
        return name


def tokenize_backtick_quoted_string(
    token_generator: Iterator[tokenize.TokenInfo], source: str, string_start: int
) -> tuple[int, str]:
    """
    Creates a token from a backtick quoted string.

    Moves the token_generator forwards till right after the next backtick.

    Parameters
    ----------
    token_generator : Iterator[tokenize.TokenInfo]
        The generator that yields the tokens of the source string (Tuple[int, str]).
        The generator is at the first token after the backtick (`)

    source : str
        The Python source code string.

    string_start : int
        This is the start of backtick quoted string inside the source string.

    Returns
    -------
    tok: Tuple[int, str]
        The token that represents the backtick quoted string.
        The integer is equal to BACKTICK_QUOTED_STRING (100).
    """
    # 遍历 token_generator 直到找到下一个反引号 (`)
    for _, tokval, start, _, _ in token_generator:
        if tokval == "`":
            # 记录反引号结束的位置
            string_end = start[1]
            break

    # 返回 BACKTICK_QUOTED_STRING (100) 作为标识和反引号引用字符串的内容
    return BACKTICK_QUOTED_STRING, source[string_start:string_end]


def tokenize_string(source: str) -> Iterator[tuple[int, str]]:
    """
    Tokenize a Python source code string.

    Parameters
    ----------
    source : str
        The Python source code string.

    Returns
    -------
    tok_generator : Iterator[Tuple[int, str]]
        An iterator yielding all tokens with only toknum and tokval (Tuple[ing, str]).
    """
    # 创建一个字符串读取器
    line_reader = StringIO(source).readline
    # 使用 tokenize.generate_tokens 函数生成 token 的生成器
    token_generator = tokenize.generate_tokens(line_reader)

    # 循环处理所有的 token，直到找到反引号 (`)
    # 然后获取直到下一个反引号的所有 token，形成反引号引用的字符串
    # 遍历 token_generator 生成的每一个 token
    for toknum, tokval, start, _, _ in token_generator:
        # 检查当前 token 是否为反引号 "`"
        if tokval == "`":
            # 如果是反引号，则尝试解析反引号引起的字符串
            try:
                # 调用函数 tokenize_backtick_quoted_string 解析反引号引起的字符串
                yield tokenize_backtick_quoted_string(
                    token_generator, source, string_start=start[1] + 1
                )
            except Exception as err:
                # 如果解析失败，则抛出语法错误并指明错误源
                raise SyntaxError(f"Failed to parse backticks in '{source}'.") from err
        else:
            # 如果不是反引号，则直接将当前 token 返回
            yield toknum, tokval
```