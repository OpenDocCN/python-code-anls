# `D:\src\scipysrc\sympy\sympy\printing\conventions.py`

```
"""
A few practical conventions common to all printers.
"""

# 导入正则表达式模块
import re

# 导入可迭代对象抽象类和求导函数
from collections.abc import Iterable
from sympy.core.function import Derivative

# 正则表达式，用于匹配带数字的名称
_name_with_digits_p = re.compile(r'^([^\W\d_]+)(\d+)$', re.U)


def split_super_sub(text):
    """Split a symbol name into a name, superscripts and subscripts

    The first part of the symbol name is considered to be its actual
    'name', followed by super- and subscripts. Each superscript is
    preceded with a "^" character or by "__". Each subscript is preceded
    by a "_" character. The three return values are the actual name, a
    list with superscripts and a list with subscripts.

    Examples
    ========

    >>> from sympy.printing.conventions import split_super_sub
    >>> split_super_sub('a_x^1')
    ('a', ['1'], ['x'])
    >>> split_super_sub('var_sub1__sup_sub2')
    ('var', ['sup'], ['sub1', 'sub2'])

    """
    # 如果输入为空，则直接返回空值
    if not text:
        return text, [], []

    # 初始化变量
    pos = 0
    name = None
    supers = []
    subs = []

    # 遍历输入文本
    while pos < len(text):
        start = pos + 1
        # 处理双下划线开头的上标
        if text[pos:pos + 2] == "__":
            start += 1
        
        # 查找 '^' 符号和 '_' 符号的位置
        pos_hat = text.find("^", start)
        if pos_hat < 0:
            pos_hat = len(text)
        pos_usc = text.find("_", start)
        if pos_usc < 0:
            pos_usc = len(text)
        
        # 确定下一个分隔符的位置
        pos_next = min(pos_hat, pos_usc)
        part = text[pos:pos_next]
        pos = pos_next
        
        # 根据不同的前缀，将部分内容归类到名称、上标或下标列表中
        if name is None:
            name = part
        elif part.startswith("^"):
            supers.append(part[1:])
        elif part.startswith("__"):
            supers.append(part[2:])
        elif part.startswith("_"):
            subs.append(part[1:])
        else:
            raise RuntimeError("This should never happen.")

    # 处理名称以数字结尾的情况，将其视为下标处理
    m = _name_with_digits_p.match(name)
    if m:
        name, sub = m.groups()
        subs.insert(0, sub)

    # 返回解析后的名称、上标和下标
    return name, supers, subs


def requires_partial(expr):
    """Return whether a partial derivative symbol is required for printing

    This requires checking how many free variables there are,
    filtering out the ones that are integers. Some expressions do not have
    free variables. In that case, check its variable list explicitly to
    get the context of the expression.
    """
    # 如果表达式是导数对象，则递归检查其表达式部分
    if isinstance(expr, Derivative):
        return requires_partial(expr.expr)

    # 如果表达式的自由符号不可迭代，则检查其变量列表的长度
    if not isinstance(expr.free_symbols, Iterable):
        return len(set(expr.variables)) > 1

    # 统计非整数的自由符号数量
    return sum(not s.is_integer for s in expr.free_symbols) > 1
```