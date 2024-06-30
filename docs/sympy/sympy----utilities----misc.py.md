# `D:\src\scipysrc\sympy\sympy\utilities\misc.py`

```
# 导入必要的模块和包
"""Miscellaneous stuff that does not really fit anywhere else."""

from __future__ import annotations  # 导入用于支持类型注解的特性

import operator  # 导入操作符模块
import sys  # 导入系统相关的功能
import os  # 导入操作系统功能
import re as _re  # 导入正则表达式模块，并起别名为_re
import struct  # 导入处理结构化数据的模块
from textwrap import fill, dedent  # 从textwrap模块中导入fill和dedent函数


class Undecidable(ValueError):
    # 一个用于在需要明确答案但无法做出决定时引发的错误
    # where a definitive answer is needed
    pass


def filldedent(s, w=70, **kwargs):
    """
    Strips leading and trailing empty lines from a copy of ``s``, then dedents,
    fills and returns it.

    Empty line stripping serves to deal with docstrings like this one that
    start with a newline after the initial triple quote, inserting an empty
    line at the beginning of the string.

    Additional keyword arguments will be passed to ``textwrap.fill()``.

    See Also
    ========
    strlines, rawlines

    """
    # 去除`s`字符串的开头和结尾的空行，然后进行去缩进、填充，并返回结果
    return '\n' + fill(dedent(str(s)).strip('\n'), width=w, **kwargs)


def strlines(s, c=64, short=False):
    """Return a cut-and-pastable string that, when printed, is
    equivalent to the input.  The lines will be surrounded by
    parentheses and no line will be longer than c (default 64)
    characters. If the line contains newlines characters, the
    `rawlines` result will be returned.  If ``short`` is True
    (default is False) then if there is one line it will be
    returned without bounding parentheses.

    Examples
    ========

    >>> from sympy.utilities.misc import strlines
    >>> q = 'this is a long string that should be broken into shorter lines'
    >>> print(strlines(q, 40))
    (
    'this is a long string that should be b'
    'roken into shorter lines'
    )
    >>> q == (
    ... 'this is a long string that should be b'
    ... 'roken into shorter lines'
    ... )
    True

    See Also
    ========
    filldedent, rawlines
    """
    # 如果`s`不是字符串，抛出值错误异常
    if not isinstance(s, str):
        raise ValueError('expecting string input')
    # 如果`s`中包含换行符，返回其`rawlines`的结果
    if '\n' in s:
        return rawlines(s)
    # 如果`s`以双引号开头，设置引号为双引号，否则为单引号
    q = '"' if repr(s).startswith('"') else "'"
    q = (q,)*2
    # 如果`s`中包含反斜杠，则使用原始字符串
    if '\\' in s:  # use r-string
        m = '(\nr%s%%s%s\n)' % q
        j = '%s\nr%s' % q
        c -= 3
    else:
        m = '(\n%s%%s%s\n)' % q
        j = '%s\n%s' % q
        c -= 2
    # 将`s`按照指定宽度分割成多行，组合成输出格式
    out = []
    while s:
        out.append(s[:c])
        s=s[c:]
    # 如果`short`为True并且`out`只有一行，则返回去除边界括号的结果
    if short and len(out) == 1:
        return (m % out[0]).splitlines()[1]  # strip bounding (\n...\n)
    # 否则返回完整的输出格式
    return m % j.join(out)


def rawlines(s):
    """Return a cut-and-pastable string that, when printed, is equivalent
    to the input. Use this when there is more than one line in the
    string. The string returned is formatted so it can be indented
    nicely within tests; in some cases it is wrapped in the dedent
    function which has to be imported from textwrap.

    Examples
    ========

    Note: because there are characters in the examples below that need
    to be escaped because they are themselves within a triple quoted
    docstring, expressions below look more complicated than they would
    # 将输入字符串 s 按行分割为列表 lines
    lines = s.split('\n')
    
    # 如果字符串只有一行，则返回该行的表示形式
    if len(lines) == 1:
        return repr(lines[0])
    
    # 判断字符串 s 是否包含三引号 """ 或单引号 '''，以及是否以空格结尾或包含反斜杠
    triple = ["'''" in s, '"""' in s]
    
    # 如果任意行以空格结尾，或者字符串包含反斜杠，或者所有行都是三引号形式，则进行以下处理
    if any(li.endswith(' ') for li in lines) or '\\' in s or all(triple):
        rv = []
        # 添加每行的表示形式到列表 rv，包括换行符
        trailing = s.endswith('\n')
        last = len(lines) - 1
        for i, li in enumerate(lines):
            if i != last or trailing:
                rv.append(repr(li + '\n'))
            else:
                rv.append(repr(li))
        # 返回一个多行字符串的表示形式，每行缩进四个空格
        return '(\n    %s\n)' % '\n    '.join(rv)
    else:
        # 否则，将所有行连接为一个多行字符串，保留原始的缩进格式
        rv = '\n    '.join(lines)
        if triple[0]:
            # 如果是三个单引号形式，则返回带有 dedent 的表示形式
            return 'dedent("""\\\n    %s""")' % rv
        else:
            # 如果是三个双引号形式，则返回带有 dedent 的表示形式
            return "dedent('''\\\n    %s''')" % rv
# 计算机架构位数，由指针大小计算得出
ARCH = str(struct.calcsize('P') * 8) + "-bit"

# 检查系统是否启用哈希随机化，如果不支持则返回 False
HASH_RANDOMIZATION = getattr(sys.flags, 'hash_randomization', False)

# 调试时使用的临时变量，用于存储调试信息
_debug_tmp: list[str] = []
# 调试迭代计数器，记录调试过程中的迭代次数
_debug_iter = 0

def debug_decorator(func):
    """如果 SYMPY_DEBUG 是 True，则打印所有装饰函数的参数和结果，否则不做任何操作。"""
    from sympy import SYMPY_DEBUG

    if not SYMPY_DEBUG:
        return func

    def maketree(f, *args, **kw):
        global _debug_tmp
        global _debug_iter
        oldtmp = _debug_tmp
        _debug_tmp = []
        _debug_iter += 1

        def tree(subtrees):
            def indent(s, variant=1):
                x = s.split("\n")
                r = "+-%s\n" % x[0]
                for a in x[1:]:
                    if a == "":
                        continue
                    if variant == 1:
                        r += "| %s\n" % a
                    else:
                        r += "  %s\n" % a
                return r
            if len(subtrees) == 0:
                return ""
            f = []
            for a in subtrees[:-1]:
                f.append(indent(a))
            f.append(indent(subtrees[-1], 2))
            return ''.join(f)

        # 如果出现错误导致算法进入无限循环，可以取消下面几行的注释，
        # 它会在调用主要函数之前打印函数名称和参数
        #from functools import reduce
        #print("%s%s %s%s" % (_debug_iter, reduce(lambda x, y: x + y, \
        #    map(lambda x: '-', range(1, 2 + _debug_iter))), f.__name__, args))

        r = f(*args, **kw)

        _debug_iter -= 1
        s = "%s%s = %s\n" % (f.__name__, args, r)
        if _debug_tmp != []:
            s += tree(_debug_tmp)
        _debug_tmp = oldtmp
        if _debug_iter == 0:
            print(_debug_tmp[0])
            _debug_tmp = []
        return r

    def decorated(*args, **kwargs):
        return maketree(func, *args, **kwargs)

    return decorated


def debug(*args):
    """
    如果 SYMPY_DEBUG 是 True，则将 ``*args`` 打印到标准错误流，否则不做任何操作。
    """
    from sympy import SYMPY_DEBUG
    if SYMPY_DEBUG:
        print(*args, file=sys.stderr)


def debugf(string, args):
    """
    如果 SYMPY_DEBUG 是 True，则使用格式化字符串 ``string%args`` 打印到标准错误流，否则不做任何操作。
    用于打印调试信息的格式化字符串。
    """
    from sympy import SYMPY_DEBUG
    if SYMPY_DEBUG:
        print(string%args, file=sys.stderr)


def find_executable(executable, path=None):
    """
    尝试在指定的路径列表（使用 os.pathsep 分隔的字符串，默认为 os.environ['PATH']）中查找可执行文件名。
    如果找到则返回完整的文件名，否则返回 None。
    """
    from .exceptions import sympy_deprecation_warning
    # 发出 SymPy 的弃用警告信息，指示 sympy.utilities.misc.find_executable() 方法已被弃用，建议使用标准库 shutil.which() 函数代替。
    sympy_deprecation_warning(
        """
        sympy.utilities.misc.find_executable() is deprecated. Use the standard
        library shutil.which() function instead.
        """,
        deprecated_since_version="1.7",
        active_deprecations_target="deprecated-find-executable",
    )
    
    # 如果路径参数为 None，则使用环境变量 PATH 的值作为路径
    if path is None:
        path = os.environ['PATH']
    
    # 将 PATH 路径按操作系统特定的路径分隔符分割为列表
    paths = path.split(os.pathsep)
    
    # 在执行文件的扩展名列表中加入空字符串作为默认值
    extlist = ['']
    
    # 如果操作系统是 OS/2
    if os.name == 'os2':
        # 将执行文件的基础名和扩展名分开
        (base, ext) = os.path.splitext(executable)
        # 在 OS/2 上，执行文件可以有任意扩展名，但如果文件名中没有点，则自动添加 .exe 扩展名
        if not ext:
            executable = executable + ".exe"
    
    # 如果操作系统是 Windows
    elif sys.platform == 'win32':
        # 获取环境变量 PATHEXT 的值，并转换为小写后分割为列表
        pathext = os.environ['PATHEXT'].lower().split(os.pathsep)
        # 将执行文件的基础名和扩展名分开
        (base, ext) = os.path.splitext(executable)
        # 如果执行文件的小写扩展名不在 PATHEXT 列表中，则更新扩展名列表为 PATHEXT 列表的值
        if ext.lower() not in pathext:
            extlist = pathext
    
    # 遍历扩展名列表
    for ext in extlist:
        # 构造具有当前扩展名的执行文件名
        execname = executable + ext
        # 如果当前文件存在，则返回该文件名
        if os.path.isfile(execname):
            return execname
        else:
            # 否则，遍历路径列表
            for p in paths:
                # 将当前路径和执行文件名合并成完整路径
                f = os.path.join(p, execname)
                # 如果完整路径指向一个文件，则返回该路径
                if os.path.isfile(f):
                    return f
    
    # 如果未找到可执行文件，则返回 None
    return None
# 定义一个函数，根据输入的对象 `x` 返回其函数名（如果定义了函数），否则返回其类型名
def func_name(x, short=False):
    # 定义一个映射表，将特定函数名映射为更短的别名
    alias = {
    'GreaterThan': 'Ge',
    'StrictGreaterThan': 'Gt',
    'LessThan': 'Le',
    'StrictLessThan': 'Lt',
    'Equality': 'Eq',
    'Unequality': 'Ne',
    }
    # 获取对象 `x` 的类型
    typ = type(x)
    # 如果类型以 "<type '" 开头，提取其中的类型名
    if str(typ).startswith("<type '"):
        typ = str(typ).split("'")[1].split("'")[0]
    # 如果类型以 "<class '" 开头，提取其中的类名
    elif str(typ).startswith("<class '"):
        typ = str(typ).split("'")[1].split("'")[0]
    # 获取对象 `x` 的函数名或类型名作为初始返回值
    rv = getattr(getattr(x, 'func', x), '__name__', typ)
    # 如果返回值包含 '.'，则只保留最后一部分作为函数名
    if '.' in rv:
        rv = rv.split('.')[-1]
    # 如果指定了 short=True，将返回值替换为别名（如果有的话）
    if short:
        rv = alias.get(rv, rv)
    return rv


# 定义一个辅助函数，返回一个能够在字符串上执行替换操作的函数，替换规则由 `reps` 指定
def _replace(reps):
    # 如果 `reps` 为空字典，则返回一个恒等函数，即不做任何替换
    if not reps:
        return lambda x: x
    # 定义替换函数 D，用于处理匹配项
    D = lambda match: reps[match.group(0)]
    # 构建替换的正则表达式模式，按照长字符串优先的顺序匹配
    pattern = _re.compile("|".join(
        [_re.escape(k) for k, v in reps.items()]), _re.M)
    # 返回一个匿名函数，将输入字符串按照替换规则进行替换
    return lambda string: pattern.sub(D, string)


# 定义一个函数，用于在字符串 `string` 中替换所有 `reps` 中的键为对应的值
def replace(string, *reps):
    # 如果 `reps` 中只有一个元素且为字典，则直接使用字典进行替换
    if len(reps) == 1:
        kv = reps[0]
        if isinstance(kv, dict):
            reps = kv
        else:
            return string.replace(*kv)
    else:
        # 将多个 (key, value) 对转换为字典形式
        reps = dict(reps)
    # 调用 `_replace` 函数返回的函数对字符串 `string` 进行替换操作，并返回结果
    return _replace(reps)(string)


# 定义一个函数，返回字符串 `s` 中经过替换或删除处理后的结果
def translate(s, a, b=None, c=None):
    # 当 `a` 为 None 时，删除 `s` 中所有属于 `deletechars` 中字符
    translate(s, None, deletechars):
        all characters in `deletechars` are deleted
    # mr 是一个空字典，用于存储替换映射关系
    mr = {}

    # 如果 a 是 None
    if a is None:
        # 如果 c 不是 None，则抛出 ValueError 异常，说明 c 应该为 None
        if c is not None:
            raise ValueError('c should be None when a=None is passed, instead got %s' % c)
        # 如果 b 也是 None，则直接返回原始字符串 s
        if b is None:
            return s
        # 否则，将 c 和 b 都赋值为空字符串
        c = b
        a = b = ''
    else:
        # 如果 a 是一个字典
        if isinstance(a, dict):
            # short 是一个空字典，用于存储单字符映射关系
            short = {}
            # 遍历 a 字典的键列表
            for k in list(a.keys()):
                # 如果键 k 的长度为 1，且对应值的长度也为 1
                if len(k) == 1 and len(a[k]) == 1:
                    # 将该键值对移动到 short 字典中，并从 a 字典中删除
                    short[k] = a.pop(k)
            # 将剩余的映射关系存储到 mr 字典中
            mr = a
            # 将 c 赋值为 b
            c = b
            # 如果 short 字典不为空，则将 a 和 b 设置为 short 字典中的键和值
            if short:
                a, b = [''.join(i) for i in list(zip(*short.items()))]
            else:
                # 否则，将 a 和 b 都设置为空字符串
                a = b = ''
        # 如果 a 不是字典，且 oldchars 和 newchars 的长度不相等，则抛出 ValueError 异常
        elif len(a) != len(b):
            raise ValueError('oldchars and newchars have different lengths')

    # 如果 c 不为空字符串
    if c:
        # 生成一个字符串转换表 val，删除其中的所有 c 中的字符
        val = str.maketrans('', '', c)
        # 使用生成的转换表 val 对 s 进行转换
        s = s.translate(val)

    # 使用 replace 函数替换 s 中的 mr 字典中的映射关系
    s = replace(s, mr)

    # 生成一个新的字符串转换表 n，将 a 中的字符替换为 b 中对应位置的字符
    n = str.maketrans(a, b)

    # 使用生成的转换表 n 对 s 进行转换，并返回结果
    return s.translate(n)
# 返回一个整数的序数表示形式的字符串，例如 1 变成 1st。
def ordinal(num):
    """Return ordinal number string of num, e.g. 1 becomes 1st.
    """
    # 导入必要的整数转换函数
    n = as_int(num)
    # 取绝对值后取模 100 的结果
    k = abs(n) % 100
    # 根据 k 的值确定序数后缀
    if 11 <= k <= 13:
        suffix = 'th'
    elif k % 10 == 1:
        suffix = 'st'
    elif k % 10 == 2:
        suffix = 'nd'
    elif k % 10 == 3:
        suffix = 'rd'
    else:
        suffix = 'th'
    # 返回带有序数后缀的字符串表示
    return str(n) + suffix


# 将输入转换为内置整数类型，确保返回值与输入相等。如果输入具有非整数值，则引发 ValueError。
def as_int(n, strict=True):
    """
    Convert the argument to a builtin integer.

    The return value is guaranteed to be equal to the input. ValueError is
    raised if the input has a non-integral value. When ``strict`` is True, this
    uses `__index__ <https://docs.python.org/3/reference/datamodel.html#object.__index__>`_
    and when it is False it uses ``int``.


    Examples
    ========

    >>> from sympy.utilities.misc import as_int
    >>> from sympy import sqrt, S

    The function is primarily concerned with sanitizing input for
    functions that need to work with builtin integers, so anything that
    is unambiguously an integer should be returned as an int:

    >>> as_int(S(3))
    3

    Floats, being of limited precision, are not assumed to be exact and
    will raise an error unless the ``strict`` flag is False. This
    precision issue becomes apparent for large floating point numbers:

    >>> big = 1e23
    >>> type(big) is float
    True
    >>> big == int(big)
    True
    >>> as_int(big)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    >>> as_int(big, strict=False)
    99999999999999991611392

    Input that might be a complex representation of an integer value is
    also rejected by default:

    >>> one = sqrt(3 + 2*sqrt(2)) - sqrt(2)
    >>> int(one) == 1
    True
    >>> as_int(one)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    """
    # 如果 strict 为 True，则使用 __index__ 方法尝试将输入转换为整数
    if strict:
        try:
            if isinstance(n, bool):
                raise TypeError
            return operator.index(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
    # 如果 strict 为 False，则使用 int() 方法尝试将输入转换为整数
    else:
        try:
            result = int(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
        # 如果输入与其整数形式不相等，则抛出 ValueError
        if n - result:
            raise ValueError('%s is not an integer' % (n,))
        return result
```