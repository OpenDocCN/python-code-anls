# `D:\src\scipysrc\sympy\sympy\printing\pretty\pretty_symbology.py`

```
"""Symbolic primitives + unicode/ASCII abstraction for pretty.py"""

# 导入系统模块、警告模块和字符串模块中的 ASCII 字符集和大写小写字母
import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
# 导入 unicodedata 模块，用于处理 Unicode 数据
import unicodedata

# 初始化 Unicode 警告信息为空字符串
unicode_warnings = ''

# 定义函数 U，用于根据名称获取 Unicode 字符，如果找不到则返回 None
def U(name):
    """
    Get a unicode character by name or, None if not found.

    This exists because older versions of Python use older unicode databases.
    """
    try:
        return unicodedata.lookup(name)
    except KeyError:
        global unicode_warnings
        # 如果找不到指定名称的 Unicode 字符，则记录警告信息并返回 None
        unicode_warnings += 'No \'%s\' in unicodedata\n' % name
        return None

# 从 sympy.printing.conventions 模块导入 split_super_sub 函数
from sympy.printing.conventions import split_super_sub
# 从 sympy.core.alphabets 模块导入 greeks 列表
from sympy.core.alphabets import greeks
# 从 sympy.utilities.exceptions 模块导入 sympy_deprecation_warning 异常
from sympy.utilities.exceptions import sympy_deprecation_warning

# 定义全局变量 __all__，包含特定公开的标识符列表
__all__ = ['greek_unicode', 'sub', 'sup', 'xsym', 'vobj', 'hobj', 'pretty_symbol',
           'annotated', 'center_pad', 'center']

# 初始化 _use_unicode 标志，默认为 False
_use_unicode = False

# 定义函数 pretty_use_unicode，用于设置或获取是否默认使用 Unicode 进行漂亮打印
def pretty_use_unicode(flag=None):
    """Set whether pretty-printer should use unicode by default"""
    global _use_unicode
    global unicode_warnings
    # 如果 flag 为 None，则返回当前 _use_unicode 的值
    if flag is None:
        return _use_unicode

    # 如果 flag 为 True，并且存在 Unicode 警告信息，则发出警告
    if flag and unicode_warnings:
        warnings.warn(unicode_warnings)
        unicode_warnings = ''

    # 记录当前 _use_unicode 的值
    use_unicode_prev = _use_unicode
    # 设置 _use_unicode 为传入的 flag 值
    _use_unicode = flag
    return use_unicode_prev

# 定义函数 pretty_try_use_unicode，尝试检测是否可以使用 Unicode 输出
def pretty_try_use_unicode():
    """See if unicode output is available and leverage it if possible"""

    # 获取 sys.stdout 的编码方式
    encoding = getattr(sys.stdout, 'encoding', None)

    # 如果编码方式为 None，则表示 sys.stdout 没有编码方式，直接返回
    if encoding is None:
        return  # sys.stdout has no encoding

    symbols = []

    # 检查是否可以表示希腊字母表中的字符
    symbols += greek_unicode.values()

    # 在这里还应该检查其他符号表，例如 atoms_table，此处省略了代码

    # 对 symbols 中的每个字符尝试编码
    for s in symbols:
        if s is None:
            return  # common symbols not present!

        try:
            s.encode(encoding)
        except UnicodeEncodeError:
            return

    # 如果所有字符都可以成功编码，则设置使用 Unicode 输出
    pretty_use_unicode(True)

# 定义函数 xstr，用于输出一个被弃用警告的字符串形式
def xstr(*args):
    sympy_deprecation_warning(
        """
        The sympy.printing.pretty.pretty_symbology.xstr() function is
        deprecated. Use str() instead.
        """,
        deprecated_since_version="1.7",
        active_deprecations_target="deprecated-pretty-printing-functions"
    )
    return str(*args)

# 定义匿名函数 g，用于生成小写希腊字母的 Unicode 字符
g = lambda l: U('GREEK SMALL LETTER %s' % l.upper())
# 定义匿名函数 G，用于生成大写希腊字母的 Unicode 字符
G = lambda l: U('GREEK CAPITAL LETTER %s' % l.upper())

# 复制希腊字母表并处理 lambda 的拼写
greek_letters = list(greeks) # make a copy
greek_letters[greek_letters.index('lambda')] = 'lamda'

# 创建 greek_unicode 字典，将希腊字母映射到其对应的小写和大写 Unicode 字符
greek_unicode = {L: g(L) for L in greek_letters}
greek_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_letters)

# aliases
# 在这里通常会定义一些别名，此处省略了代码
# 将 greek_unicode 字典中的 'lambda' 键映射到 'lamda' 的值
greek_unicode['lambda'] = greek_unicode['lamda']

# 将 greek_unicode 字典中的 'Lambda' 键映射到 'Lamda' 的值
greek_unicode['Lambda'] = greek_unicode['Lamda']

# 将 greek_unicode 字典中的 'varsigma' 键映射到希腊字符小型终结 sigma 的 Unicode 字符
greek_unicode['varsigma'] = '\N{GREEK SMALL LETTER FINAL SIGMA}'

# BOLD
# 使用 lambda 函数生成带有数学粗体小写字母的 Unicode 名称
b = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
# 使用 lambda 函数生成带有数学粗体大写字母的 Unicode 名称
B = lambda l: U('MATHEMATICAL BOLD CAPITAL %s' % l.upper())

# 创建包含所有 ASCII 小写字母的带粗体的 Unicode 字典
bold_unicode = {l: b(l) for l in ascii_lowercase}
# 将带粗体的 Unicode 字典更新为包含所有 ASCII 大写字母的带粗体的 Unicode 字典
bold_unicode.update((L, B(L)) for L in ascii_uppercase)

# GREEK BOLD
# 使用 lambda 函数生成带有数学粗体希腊小写字母的 Unicode 名称
gb = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
# 使用 lambda 函数生成带有数学粗体希腊大写字母的 Unicode 名称
GB = lambda l: U('MATHEMATICAL BOLD CAPITAL  %s' % l.upper())

# 创建希腊字母列表的副本，并将 lambda 改正为 lamda
greek_bold_letters = list(greeks) # make a copy, not strictly required here
greek_bold_letters[greek_bold_letters.index('lambda')] = 'lamda'

# 创建包含希腊粗体字母的 Unicode 字典
greek_bold_unicode = {L: g(L) for L in greek_bold_letters}
# 将带粗体的希腊 Unicode 字典更新为包含带粗体的希腊大写字母的 Unicode 字典
greek_bold_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_bold_letters)
# 将 greek_bold_unicode 字典中的 'lambda' 键映射到 greek_unicode 字典中 'lamda' 键的值
greek_bold_unicode['lambda'] = greek_unicode['lamda']
# 将 greek_bold_unicode 字典中的 'Lambda' 键映射到 greek_unicode 字典中 'Lamda' 键的值
greek_bold_unicode['Lambda'] = greek_unicode['Lamda']
# 将 greek_bold_unicode 字典中的 'varsigma' 键映射到带数学粗体小型终结 sigma 的 Unicode 字符
greek_bold_unicode['varsigma'] = '\N{MATHEMATICAL BOLD SMALL FINAL SIGMA}'

# 数字到文本的映射字典
digit_2txt = {
    '0':    'ZERO',
    '1':    'ONE',
    '2':    'TWO',
    '3':    'THREE',
    '4':    'FOUR',
    '5':    'FIVE',
    '6':    'SIX',
    '7':    'SEVEN',
    '8':    'EIGHT',
    '9':    'NINE',
}

# 符号到文本的映射字典
symb_2txt = {
    '+':    'PLUS SIGN',
    '-':    'MINUS',
    '=':    'EQUALS SIGN',
    '(':    'LEFT PARENTHESIS',
    ')':    'RIGHT PARENTHESIS',
    '[':    'LEFT SQUARE BRACKET',
    ']':    'RIGHT SQUARE BRACKET',
    '{':    'LEFT CURLY BRACKET',
    '}':    'RIGHT CURLY BRACKET',

    # 非标准
    '{}':   'CURLY BRACKET',
    'sum':  'SUMMATION',
    'int':  'INTEGRAL',
}

# SUBSCRIPT & SUPERSCRIPT
# 创建 lambda 函数，生成拉丁小写字母的下标符号的 Unicode 名称
LSUB = lambda letter: U('LATIN SUBSCRIPT SMALL LETTER %s' % letter.upper())
# 创建 lambda 函数，生成希腊小写字母的下标符号的 Unicode 名称
GSUB = lambda letter: U('GREEK SUBSCRIPT SMALL LETTER %s' % letter.upper())
# 创建 lambda 函数，生成数字的下标符号的 Unicode 名称
DSUB = lambda digit:  U('SUBSCRIPT %s' % digit_2txt[digit])
# 创建 lambda 函数，生成符号的下标符号的 Unicode 名称
SSUB = lambda symb:   U('SUBSCRIPT %s' % symb_2txt[symb])

# 创建 lambda 函数，生成拉丁小写字母的上标符号的 Unicode 名称
LSUP = lambda letter: U('SUPERSCRIPT LATIN SMALL LETTER %s' % letter.upper())
# 创建 lambda 函数，生成数字的上标符号的 Unicode 名称
DSUP = lambda digit:  U('SUPERSCRIPT %s' % digit_2txt[digit])
# 创建 lambda 函数，生成符号的上标符号的 Unicode 名称
SSUP = lambda symb:   U('SUPERSCRIPT %s' % symb_2txt[symb])

# 创建空字典，用于存储符号到下标符号的映射
sub = {}    # symb -> subscript symbol
# 创建空字典，用于存储符号到上标符号的映射
sup = {}    # symb -> superscript symbol

# 创建拉丁小写字母的下标符号和上标符号的映射
for l in 'aeioruvxhklmnpst':
    sub[l] = LSUB(l)

# 创建 'i' 和 'n' 的上标符号映射
for l in 'in':
    sup[l] = LSUP(l)

# 创建希腊字母 'beta', 'gamma', 'rho', 'phi', 'chi' 的下标符号映射
for gl in ['beta', 'gamma', 'rho', 'phi', 'chi']:
    sub[gl] = GSUB(gl)

# 创建数字和其对应下标符号、上标符号的映射
for d in [str(i) for i in range(10)]:
    sub[d] = DSUB(d)
    sup[d] = DSUP(d)

# 创建符号和其对应下标符号、上标符号的映射
for s in '+-=()':
    sub[s] = SSUB(s)
    sup[s] = SSUP(s)

# 变量修饰符字典
# TODO: 让括号根据内容的高度进行调整
modifier_dict = {
    # 重音符号
    'mathring': lambda s: center_accent(s, '\N{COMBINING RING ABOVE}'),
    'ddddot': lambda s: center_accent(s, '\N{COMBINING FOUR DOTS ABOVE}'),
    'dddot': lambda s: center_accent(s, '\N{COMBINING THREE DOTS ABOVE}'),
    'ddot': lambda s: center_accent(s, '\N{COMBINING DIAERESIS}'),
}
    'dot': lambda s: center_accent(s, '\N{COMBINING DOT ABOVE}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING DOT ABOVE}' 放置在 s 的上方
    'check': lambda s: center_accent(s, '\N{COMBINING CARON}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING CARON}' 放置在 s 的上方
    'breve': lambda s: center_accent(s, '\N{COMBINING BREVE}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING BREVE}' 放置在 s 的上方
    'acute': lambda s: center_accent(s, '\N{COMBINING ACUTE ACCENT}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING ACUTE ACCENT}' 放置在 s 的上方
    'grave': lambda s: center_accent(s, '\N{COMBINING GRAVE ACCENT}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING GRAVE ACCENT}' 放置在 s 的上方
    'tilde': lambda s: center_accent(s, '\N{COMBINING TILDE}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING TILDE}' 放置在 s 的上方
    'hat': lambda s: center_accent(s, '\N{COMBINING CIRCUMFLEX ACCENT}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING CIRCUMFLEX ACCENT}' 放置在 s 的上方
    'bar': lambda s: center_accent(s, '\N{COMBINING OVERLINE}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING OVERLINE}' 放置在 s 的上方
    'vec': lambda s: center_accent(s, '\N{COMBINING RIGHT ARROW ABOVE}'),
    # 定义一个 lambda 函数，接受字符串 s，并将 '\N{COMBINING RIGHT ARROW ABOVE}' 放置在 s 的上方
    'prime': lambda s: s+'\N{PRIME}',
    # 定义一个 lambda 函数，接受字符串 s，并在其后添加 '\N{PRIME}' 符号
    'prm': lambda s: s+'\N{PRIME}',
    # 定义一个 lambda 函数，接受字符串 s，并在其后添加 '\N{PRIME}' 符号
    # # Faces -- these are here for some compatibility with latex printing
    # 'bold': lambda s: s,
    # 'bm': lambda s: s,
    # 'cal': lambda s: s,
    # 'scr': lambda s: s,
    # 'frak': lambda s: s,
    # Brackets
    'norm': lambda s: '\N{DOUBLE VERTICAL LINE}'+s+'\N{DOUBLE VERTICAL LINE}',
    # 定义一个 lambda 函数，接受字符串 s，并在其前后添加 '\N{DOUBLE VERTICAL LINE}' 符号
    'avg': lambda s: '\N{MATHEMATICAL LEFT ANGLE BRACKET}'+s+'\N{MATHEMATICAL RIGHT ANGLE BRACKET}',
    # 定义一个 lambda 函数，接受字符串 s，并在其前后添加 '\N{MATHEMATICAL LEFT ANGLE BRACKET}' 和 '\N{MATHEMATICAL RIGHT ANGLE BRACKET}' 符号
    'abs': lambda s: '\N{VERTICAL LINE}'+s+'\N{VERTICAL LINE}',
    # 定义一个 lambda 函数，接受字符串 s，并在其前后添加 '\N{VERTICAL LINE}' 符号
    'mag': lambda s: '\N{VERTICAL LINE}'+s+'\N{VERTICAL LINE}',
    # 定义一个 lambda 函数，接受字符串 s，并在其前后添加 '\N{VERTICAL LINE}' 符号
# 定义一个 lambda 函数，返回指定符号的上钩字符
HUP = lambda symb: U('%s UPPER HOOK' % symb_2txt[symb])
# 定义一个 lambda 函数，返回指定符号的上角字符
CUP = lambda symb: U('%s UPPER CORNER' % symb_2txt[symb])
# 定义一个 lambda 函数，返回指定符号的中间片段字符
MID = lambda symb: U('%s MIDDLE PIECE' % symb_2txt[symb])
# 定义一个 lambda 函数，返回指定符号的扩展字符
EXT = lambda symb: U('%s EXTENSION' % symb_2txt[symb])
# 定义一个 lambda 函数，返回指定符号的下钩字符
HLO = lambda symb: U('%s LOWER HOOK' % symb_2txt[symb])
# 定义一个 lambda 函数，返回指定符号的下角字符
CLO = lambda symb: U('%s LOWER CORNER' % symb_2txt[symb])
# 定义一个 lambda 函数，返回指定符号的顶部字符
TOP = lambda symb: U('%s TOP' % symb_2txt[symb])
# 定义一个 lambda 函数，返回指定符号的底部字符
BOT = lambda symb: U('%s BOTTOM' % symb_2txt[symb])

# 定义一个字典，包含各种垂直对象的 Unicode 表示
_xobj_unicode = {

    # 垂直符号
    # ((扩展, 顶部, 底部, 中间), 字符)
    '(': ((EXT('('), HUP('('), HLO('(')), '('),
    ')': ((EXT(')'), HUP(')'), HLO(')')), ')'),
    '[': ((EXT('['), CUP('['), CLO('[')), '['),
    ']': ((EXT(']'), CUP(']'), CLO(']')), ']'),
    '{': ((EXT('{}'), HUP('{'), HLO('{'), MID('{')) , '{'),
    '}': ((EXT('{}'), HUP('}'), HLO('}'), MID('}')) , '}'),
    '|': U('BOX DRAWINGS LIGHT VERTICAL'),
    'Tee': U('BOX DRAWINGS LIGHT UP AND HORIZONTAL'),
    'UpTack': U('BOX DRAWINGS LIGHT DOWN AND HORIZONTAL'),
    'corner_up_centre': U('LEFT PARENTHESIS EXTENSION'),  # 符号角上中心
    '(_ext': U('RIGHT PARENTHESIS EXTENSION'),  # 左括号扩展
    ')_ext': U('LEFT PARENTHESIS LOWER HOOK'),  # 右括号扩展
    '(_lower_hook': U('RIGHT PARENTHESIS LOWER HOOK'),  # 左括号下钩
    ')_lower_hook': U('LEFT PARENTHESIS UPPER HOOK'),  # 右括号下钩
    '(_upper_hook': U('RIGHT PARENTHESIS UPPER HOOK'),  # 左括号上钩
    ')_upper_hook': U('BOX DRAWINGS LIGHT VERTICAL'),  # 右括号上钩
    '<': ((U('BOX DRAWINGS LIGHT VERTICAL'),
          U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'),
          U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT')), '<'),

    '>': ((U('BOX DRAWINGS LIGHT VERTICAL'),
          U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
          U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), '>'),

    'lfloor': ((EXT('['), EXT('['), CLO('[')), U('LEFT FLOOR')),
    'rfloor': ((EXT(']'), EXT(']'), CLO(']')), U('RIGHT FLOOR')),
    'lceil': ((EXT('['), CUP('['), EXT('[')), U('LEFT CEILING')),
    'rceil': ((EXT(']'), CUP(']'), EXT(']')), U('RIGHT CEILING')),

    'int': ((EXT('int'), U('TOP HALF INTEGRAL'), U('BOTTOM HALF INTEGRAL')), U('INTEGRAL')),
    'sum': ((U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
            '_', U('OVERLINE'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), U('N-ARY SUMMATION')),

    # 水平对象
    '-': U('BOX DRAWINGS LIGHT HORIZONTAL'),
    '_': U('LOW LINE'),
    # We used to use this, but LOW LINE looks better for roots, as it's a
}
    # 对象 '/' 的注释，表示这个字符在 Unicode 中对应的含义是「BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT」
    '/':    U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'),

    # 对象 '\\' 的注释，表示这个字符在 Unicode 中对应的含义是「BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT」
    '\\':   U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
# '}' 后面这部分代码定义了一个名为 _xobj_ascii 的字典，用于存储ASCII艺术风格的符号以及它们的结构信息。
_xobj_ascii = {
    # vertical symbols
    #       (( ext, top, bot, mid ), c1)
    '(':    (( '|', '/', '\\' ), '('),  # '(' 符号及其对应的显示元组
    ')':    (( '|', '\\', '/' ), ')'),  # ')' 符号及其对应的显示元组

# XXX this looks ugly
#   '[':    (( '|', '-', '-' ), '['),  # '[' 符号及其对应的显示元组
#   ']':    (( '|', '-', '-' ), ']'),  # ']' 符号及其对应的显示元组
# XXX not so ugly :(
    '[':    (( '[', '[', '[' ), '['),  # '[' 符号及其对应的显示元组
    ']':    (( ']', ']', ']' ), ']'),  # ']' 符号及其对应的显示元组

    '{':    (( '|', '/', '\\', '<' ), '{'),  # '{' 符号及其对应的显示元组
    '}':    (( '|', '\\', '/', '>' ), '}'),  # '}' 符号及其对应的显示元组
    '|':    '|',  # '|' 符号

    '<':    (( '|', '/', '\\' ), '<'),  # '<' 符号及其对应的显示元组
    '>':    (( '|', '\\', '/' ), '>'),  # '>' 符号及其对应的显示元组

    'int':  ( ' | ', '  /', '/  ' ),  # 'int' 符号及其对应的显示元组

    # horizontal objects
    '-':    '-',  # '-' 符号
    '_':    '_',  # '_' 符号

    # diagonal objects '\' & '/' ?
    '/':    '/',  # '/' 符号
    '\\':   '\\',  # '\\' 符号
}
    # 创建一个元组，包含键值对 (3, 8)，值为 VF('THREE EIGHTHS')
    (3, 8): VF('THREE EIGHTHS'),
    # 创建一个元组，包含键值对 (5, 8)，值为 VF('FIVE EIGHTHS')
    (5, 8): VF('FIVE EIGHTHS'),
    # 创建一个元组，包含键值对 (7, 8)，值为 VF('SEVEN EIGHTHS')
    (7, 8): VF('SEVEN EIGHTHS'),
}

# atom symbols
_xsym = {
    '==':  ('=', '='),
    '<':   ('<', '<'),
    '>':   ('>', '>'),
    '<=':  ('<=', U('LESS-THAN OR EQUAL TO')),
    '>=':  ('>=', U('GREATER-THAN OR EQUAL TO')),
    '!=':  ('!=', U('NOT EQUAL TO')),
    ':=':  (':=', ':='),
    '+=':  ('+=', '+='),
    '-=':  ('-=', '-='),
    '*=':  ('*=', '*='),
    '/=':  ('/=', '/='),
    '%=':  ('%=', '%='),
    '*':   ('*', U('DOT OPERATOR')),
    '-->': ('-->', U('EM DASH') + U('EM DASH') +
            U('BLACK RIGHT-POINTING TRIANGLE') if U('EM DASH')
            and U('BLACK RIGHT-POINTING TRIANGLE') else None),
    '==>': ('==>', U('BOX DRAWINGS DOUBLE HORIZONTAL') +
            U('BOX DRAWINGS DOUBLE HORIZONTAL') +
            U('BLACK RIGHT-POINTING TRIANGLE') if
            U('BOX DRAWINGS DOUBLE HORIZONTAL') and
            U('BOX DRAWINGS DOUBLE HORIZONTAL') and
            U('BLACK RIGHT-POINTING TRIANGLE') else None),
    '.':   ('*', U('RING OPERATOR')),
}

def xsym(sym):
    """get symbology for a 'character'"""
    op = _xsym[sym]

    if _use_unicode:
        return op[1]
    else:
        return op[0]

# SYMBOLS

atoms_table = {
    # class                    how-to-display
    'Exp1':                    U('SCRIPT SMALL E'),
    'Pi':                      U('GREEK SMALL LETTER PI'),
    'Infinity':                U('INFINITY'),
    'NegativeInfinity':        U('INFINITY') and ('-' + U('INFINITY')),  # XXX what to do here
    #'ImaginaryUnit':          U('GREEK SMALL LETTER IOTA'),
    #'ImaginaryUnit':          U('MATHEMATICAL ITALIC SMALL I'),
    'ImaginaryUnit':           U('DOUBLE-STRUCK ITALIC SMALL I'),
    'EmptySet':                U('EMPTY SET'),
    'Naturals':                U('DOUBLE-STRUCK CAPITAL N'),
    'Naturals0':              (U('DOUBLE-STRUCK CAPITAL N') and
                              (U('DOUBLE-STRUCK CAPITAL N') +
                               U('SUBSCRIPT ZERO'))),
    'Integers':                U('DOUBLE-STRUCK CAPITAL Z'),
    'Rationals':               U('DOUBLE-STRUCK CAPITAL Q'),
    'Reals':                   U('DOUBLE-STRUCK CAPITAL R'),
    'Complexes':               U('DOUBLE-STRUCK CAPITAL C'),
    'Universe':                U('MATHEMATICAL DOUBLE-STRUCK CAPITAL U'),
    'IdentityMatrix':          U('MATHEMATICAL DOUBLE-STRUCK CAPITAL I'),
    'ZeroMatrix':              U('MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO'),
    'OneMatrix':               U('MATHEMATICAL DOUBLE-STRUCK DIGIT ONE'),
    'Differential':            U('DOUBLE-STRUCK ITALIC SMALL D'),
    'Union':                   U('UNION'),
    'ElementOf':               U('ELEMENT OF'),
    'SmallElementOf':          U('SMALL ELEMENT OF'),
    'SymmetricDifference':     U('INCREMENT'),
    'Intersection':            U('INTERSECTION'),
    'Ring':                    U('RING OPERATOR'),
    'Multiplication':          U('MULTIPLICATION SIGN'),
    'TensorProduct':           U('N-ARY CIRCLED TIMES OPERATOR'),
    'Dots':                    U('HORIZONTAL ELLIPSIS'),
    'Modifier Letter Low Ring':U('Modifier Letter Low Ring'),
    # 定义常量 'Modifier Letter Low Ring'，其值是 U('Modifier Letter Low Ring') 返回的结果
    'EmptySequence':           'EmptySequence',
    # 定义常量 'EmptySequence'，其值是字符串 'EmptySequence'
    'SuperscriptPlus':         U('SUPERSCRIPT PLUS SIGN'),
    # 定义常量 'SuperscriptPlus'，其值是 U('SUPERSCRIPT PLUS SIGN') 返回的结果
    'SuperscriptMinus':        U('SUPERSCRIPT MINUS'),
    # 定义常量 'SuperscriptMinus'，其值是 U('SUPERSCRIPT MINUS') 返回的结果
    'Dagger':                  U('DAGGER'),
    # 定义常量 'Dagger'，其值是 U('DAGGER') 返回的结果
    'Degree':                  U('DEGREE SIGN'),
    # 定义常量 'Degree'，其值是 U('DEGREE SIGN') 返回的结果
    # 逻辑符号
    'And':                     U('LOGICAL AND'),
    # 定义常量 'And'，其值是 U('LOGICAL AND') 返回的结果
    'Or':                      U('LOGICAL OR'),
    # 定义常量 'Or'，其值是 U('LOGICAL OR') 返回的结果
    'Not':                     U('NOT SIGN'),
    # 定义常量 'Not'，其值是 U('NOT SIGN') 返回的结果
    'Nor':                     U('NOR'),
    # 定义常量 'Nor'，其值是 U('NOR') 返回的结果
    'Nand':                    U('NAND'),
    # 定义常量 'Nand'，其值是 U('NAND') 返回的结果
    'Xor':                     U('XOR'),
    # 定义常量 'Xor'，其值是 U('XOR') 返回的结果
    'Equiv':                   U('LEFT RIGHT DOUBLE ARROW'),
    # 定义常量 'Equiv'，其值是 U('LEFT RIGHT DOUBLE ARROW') 返回的结果
    'NotEquiv':                U('LEFT RIGHT DOUBLE ARROW WITH STROKE'),
    # 定义常量 'NotEquiv'，其值是 U('LEFT RIGHT DOUBLE ARROW WITH STROKE') 返回的结果
    'Implies':                 U('LEFT RIGHT DOUBLE ARROW'),
    # 定义常量 'Implies'，其值是 U('LEFT RIGHT DOUBLE ARROW') 返回的结果
    'NotImplies':              U('LEFT RIGHT DOUBLE ARROW WITH STROKE'),
    # 定义常量 'NotImplies'，其值是 U('LEFT RIGHT DOUBLE ARROW WITH STROKE') 返回的结果
    'Arrow':                   U('RIGHTWARDS ARROW'),
    # 定义常量 'Arrow'，其值是 U('RIGHTWARDS ARROW') 返回的结果
    'ArrowFromBar':            U('RIGHTWARDS ARROW FROM BAR'),
    # 定义常量 'ArrowFromBar'，其值是 U('RIGHTWARDS ARROW FROM BAR') 返回的结果
    'NotArrow':                U('RIGHTWARDS ARROW WITH STROKE'),
    # 定义常量 'NotArrow'，其值是 U('RIGHTWARDS ARROW WITH STROKE') 返回的结果
    'Tautology':               U('BOX DRAWINGS LIGHT UP AND HORIZONTAL'),
    # 定义常量 'Tautology'，其值是 U('BOX DRAWINGS LIGHT UP AND HORIZONTAL') 返回的结果
    'Contradiction':           U('BOX DRAWINGS LIGHT DOWN AND HORIZONTAL')
    # 定义常量 'Contradiction'，其值是 U('BOX DRAWINGS LIGHT DOWN AND HORIZONTAL') 返回的结果
def pretty_atom(atom_name, default=None, printer=None):
    """return pretty representation of an atom"""
    # 如果使用 Unicode 并且打印机不为空，且原子名为 'ImaginaryUnit'，并且打印机设置的虚部单位为 'j'
    if _use_unicode:
        if printer is not None and atom_name == 'ImaginaryUnit' and printer._settings['imaginary_unit'] == 'j':
            return U('DOUBLE-STRUCK ITALIC SMALL J')
        else:
            # 返回原子名对应的 Unicode 美化表示
            return atoms_table[atom_name]
    else:
        # 如果默认值不为空，返回默认值；否则引发 KeyError 异常
        if default is not None:
            return default
        raise KeyError('only unicode')  # 如果不使用 Unicode，抛出异常，只支持 Unicode


def pretty_symbol(symb_name, bold_name=False):
    """return pretty representation of a symbol"""
    # 将符号名拆分为主符号、上标和下标
    # UC: beta1
    # UC: f_beta

    if not _use_unicode:
        return symb_name

    # 分离主符号、上标和下标
    name, sups, subs = split_super_sub(symb_name)

    def translate(s, bold_name):
        # 根据是否要加粗转换符号
        if bold_name:
            gG = greek_bold_unicode.get(s)
        else:
            gG = greek_unicode.get(s)
        if gG is not None:
            return gG
        # 尝试根据修改器字典处理符号
        for key in sorted(modifier_dict.keys(), key=lambda k:len(k), reverse=True):
            if s.lower().endswith(key) and len(s)>len(key):
                return modifier_dict[key](translate(s[:-len(key)], bold_name))
        if bold_name:
            # 如果要加粗，则返回符号的粗体 Unicode 字符串
            return ''.join([bold_unicode[c] for c in s])
        return s

    # 转换主符号的名称
    name = translate(name, bold_name)

    def pretty_list(l, mapping):
        # 根据映射处理上标或下标列表
        result = []
        for s in l:
            pretty = mapping.get(s)
            if pretty is None:
                try:  # 尝试按单个字符匹配
                    pretty = ''.join([mapping[c] for c in s])
                except (TypeError, KeyError):
                    return None
            result.append(pretty)
        return result

    # 美化处理上标和下标列表
    pretty_sups = pretty_list(sups, sup)
    if pretty_sups is not None:
        pretty_subs = pretty_list(subs, sub)
    else:
        pretty_subs = None

    # 将处理结果组合成最终的字符串
    if pretty_subs is None:
        # 如果无法美化处理，直接连接上标和下标到名称
        if subs:
            name += '_'+'_'.join([translate(s, bold_name) for s in subs])
        if sups:
            name += '__'+'__'.join([translate(s, bold_name) for s in sups])
        return name
    else:
        # 否则，连接处理后的上标和下标到名称
        sups_result = ' '.join(pretty_sups)
        subs_result = ' '.join(pretty_subs)

    return ''.join([name, sups_result, subs_result])
    # 定义用于存储Unicode风格图形的字典，每个键值对对应不同的字符图形和格式
    ucode_pics = {
        'F': (2, 0, 2, 0, '\N{BOX DRAWINGS LIGHT DOWN AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\n'
                          '\N{BOX DRAWINGS LIGHT VERTICAL AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\n'
                          '\N{BOX DRAWINGS LIGHT UP}'),
        'G': (3, 0, 3, 1, '\N{BOX DRAWINGS LIGHT ARC DOWN AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\N{BOX DRAWINGS LIGHT ARC DOWN AND LEFT}\n'
                          '\N{BOX DRAWINGS LIGHT VERTICAL}\N{BOX DRAWINGS LIGHT RIGHT}\N{BOX DRAWINGS LIGHT DOWN AND LEFT}\n'
                          '\N{BOX DRAWINGS LIGHT ARC UP AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\N{BOX DRAWINGS LIGHT ARC UP AND LEFT}')
    }
    
    # 定义用于存储ASCII风格图形的字典，每个键值对对应不同的字符图形和格式
    ascii_pics = {
        'F': (3, 0, 3, 0, ' _\n|_\n|\n'),
        'G': (3, 0, 3, 1, ' __\n/__\n\\_|')
    }
    
    # 根据_use_unicode变量的值决定返回Unicode风格还是ASCII风格的字符图形
    if _use_unicode:
        return ucode_pics[letter]
    else:
        return ascii_pics[letter]
# 创建一个字典 _remove_combining，用于存储需要移除的组合字符范围的 Unicode 码点
# 这些范围包括从 COMBINING GRAVE ACCENT 到 COMBINING LATIN SMALL LETTER X 和
# 从 COMBINING LEFT HARPOON ABOVE 到 COMBINING ASTERISK ABOVE 的所有字符
_remove_combining = dict.fromkeys(list(range(ord('\N{COMBINING GRAVE ACCENT}'), ord('\N{COMBINING LATIN SMALL LETTER X}')))
                            + list(range(ord('\N{COMBINING LEFT HARPOON ABOVE}'), ord('\N{COMBINING ASTERISK ABOVE}'))))

def is_combining(sym):
    """检查符号是否是 Unicode 组合修改符。"""
    # 检查符号的 Unicode 码点是否存在于 _remove_combining 字典中
    return ord(sym) in _remove_combining


def center_accent(string, accent):
    """
    在字符串的中间字符处插入组合重音符号。对于放置在符号名称上的组合重音符号（包括多字符名称）很有用。

    Parameters
    ==========

    string : string
        要插入重音符号的字符串。
    accent : string
        要插入的组合重音符号。

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Combining_character
    .. [2] https://en.wikipedia.org/wiki/Combining_Diacritical_Marks

    """

    # 将重音符号放在前一个字符上，尽管这在控制台上可能并不总是看起来这样
    midpoint = len(string) // 2 + 1
    firstpart = string[:midpoint]
    secondpart = string[midpoint:]
    return firstpart + accent + secondpart


def line_width(line):
    """Unicode 组合符号（修改符）不会显示为单独的符号，因此不应计数。"""
    # 返回移除组合符号后的字符串长度
    return len(line.translate(_remove_combining))


def is_subscriptable_in_unicode(subscript):
    """
    检查字符串是否可以在 Unicode 中作为下标使用。

    Parameters
    ==========

    subscript: 需要检查的字符串

    Examples
    ========

    >>> from sympy.printing.pretty.pretty_symbology import is_subscriptable_in_unicode
    >>> is_subscriptable_in_unicode('abc')
    False
    >>> is_subscriptable_in_unicode('123')
    True

    """
    # 检查字符串中的每个字符是否都可以作为 subscript 的下标
    return all(character in subscript for character in subscript)


def center_pad(wstring, wtarget, fillchar=' '):
    """
    返回填充字符串，使得一个 wstring 字符宽的字符串在宽度为 wtarget 的空间中居中。

    应始终确保 wstring 的 line_width 小于或等于 wtarget，否则将引发 ValueError。
    """
    if wstring > wtarget:
        raise ValueError('字符串的空间不足')
    wdelta = wtarget - wstring

    wleft = wdelta // 2  # 更倾向于左边 '1 '
    wright = wdelta - wleft

    left = fillchar * wleft
    right = fillchar * wright

    return left, right


def center(string, width, fillchar=' '):
    """返回一个居中的字符串，其长度由 line_width 确定，使用 fillchar 进行填充。"""
    # 获得左右填充字符串
    left, right = center_pad(line_width(string), width, fillchar)
    return ''.join([left, string, right])
```