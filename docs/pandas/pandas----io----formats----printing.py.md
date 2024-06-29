# `D:\src\scipysrc\pandas\pandas\io\formats\printing.py`

```
# 打印工具模块。

# 引入类型注解的未来支持
from __future__ import annotations

# 引入抽象基类中的各种集合类型
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    Sequence,
)

# 引入系统模块
import sys

# 引入类型提示相关的工具
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
)

# 引入 Unicode 数据处理模块
from unicodedata import east_asian_width

# 引入 Pandas 配置模块
from pandas._config import get_option

# 引入 Pandas 推断数据类型模块
from pandas.core.dtypes.inference import is_sequence

# 引入 Pandas 控制台输出格式模块
from pandas.io.formats.console import get_console_size

# 如果是类型检查阶段，引入 ListLike 类型
if TYPE_CHECKING:
    from pandas._typing import ListLike

# 定义一个别名 EscapeChars，可以是字符串映射或字符串迭代器
EscapeChars = Union[Mapping[str, str], Iterable[str]]

# 类型变量 _KT 和 _VT
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def adjoin(space: int, *lists: list[str], **kwargs: Any) -> str:
    """
    使用指定的空格数将两组字符串粘合在一起。
    目的是美化输出。

    ----------
    space : int
        空格数用于填充
    lists : str
        要连接的字符串列表
    strlen : callable
        用于计算每个字符串长度的函数。用于处理 Unicode。
    justfunc : callable
        用于调整字符串对齐方式的函数。用于处理 Unicode。
    """
    # 从 kwargs 中获取 strlen 参数，默认为 len 函数
    strlen = kwargs.pop("strlen", len)
    # 从 kwargs 中获取 justfunc 参数，默认为 _adj_justify 函数
    justfunc = kwargs.pop("justfunc", _adj_justify)

    # 创建一个新的列表
    newLists = []
    # 计算每组字符串的最大长度并加上空格数
    lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
    # 对最后一组字符串特殊处理，计算其长度
    lengths.append(max(map(len, lists[-1])))
    # 计算所有字符串中的最大长度
    maxLen = max(map(len, lists))
    # 遍历字符串列表
    for i, lst in enumerate(lists):
        # 调整字符串的对齐方式
        nl = justfunc(lst, lengths[i], mode="left")
        # 如果字符串长度不足，用空格填充
        nl = ([" " * lengths[i]] * (maxLen - len(lst))) + nl
        # 将调整后的字符串列表添加到新列表中
        newLists.append(nl)
    # 将调整后的字符串列表按行连接起来并返回
    toJoin = zip(*newLists)
    return "\n".join("".join(lines) for lines in toJoin)


def _adj_justify(texts: Iterable[str], max_len: int, mode: str = "right") -> list[str]:
    """
    对字符串或类列表进行左对齐、居中或右对齐的操作。
    """
    if mode == "left":
        return [x.ljust(max_len) for x in texts]
    elif mode == "center":
        return [x.center(max_len) for x in texts]
    else:
        return [x.rjust(max_len) for x in texts]


# Unicode 统一化
# ---------------------
#
# 用于生成对象的 Unicode 文本或字节(对于 3.x)/字符串(对于 2.x) 表示的 pprinting 实用函数。
# 尽可能使用这些函数，而不是自己编写。
#
# 使用时机
# -----------
#
# 1) 如果您正在编写与 pandas 内部相关的代码（不涉及直接 I/O），请使用 pprint_thing()。
#
#    它将始终返回 Unicode 文本，可以被包裹在包的其他部分处理，而不会破坏。
#
# 2) 如果需要将内容写入文件，请使用 pprint_thing_encoded(encoding)。
#
#    如果未指定编码，则默认为 utf-8。由于使用 utf-8 对纯 ascii 进行编码是一个无操作，因此如果您处理纯 ascii，则可以安全地使用默认的 utf-8。
#

def _pprint_seq(
    seq: ListLike, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds: Any
) -> str:
    """
    内部函数。用于迭代对象的 pprinter。通常应使用 pprint_thing()
    """
    rather than calling this directly.

    bounds length of printed sequence, depending on options
    """
    # 如果序列是一个集合（set），使用花括号格式化字符串
    if isinstance(seq, set):
        fmt = "{{{body}}}"
    else:
        # 如果序列具有 __setitem__ 属性，则使用方括号格式化字符串，否则使用圆括号格式化字符串
        fmt = "[{body}]" if hasattr(seq, "__setitem__") else "({body})"

    # 如果 max_seq_items 是 False，则不限制打印的最大项目数
    if max_seq_items is False:
        max_items = None
    else:
        # 否则，设置最大项目数为 max_seq_items，或者从全局选项中获取，或者使用序列的长度
        max_items = max_seq_items or get_option("max_seq_items") or len(seq)

    # 创建序列的迭代器
    s = iter(seq)
    # 处理集合类型，不支持切片操作
    r = []
    max_items_reached = False
    for i, item in enumerate(s):
        # 如果指定了最大项目数，并且当前索引超过或等于最大项目数，则达到了最大项目数限制
        if (max_items is not None) and (i >= max_items):
            max_items_reached = True
            break
        # 对每个项目调用 pprint_thing 函数，获取其格式化后的字符串，并添加到列表 r 中
        r.append(pprint_thing(item, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds))
    # 将所有项目的格式化字符串用逗号连接成一个字符串
    body = ", ".join(r)

    # 如果达到了最大项目数限制，则在最后加上省略号
    if max_items_reached:
        body += ", ..."
    # 如果序列是一个长度为 1 的元组，则在最后一个项目后加上一个逗号
    elif isinstance(seq, tuple) and len(seq) == 1:
        body += ","

    # 根据 fmt 格式化字符串，将 body 插入其中并返回
    return fmt.format(body=body)
def _pprint_dict(
    seq: Mapping, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds: Any
) -> str:
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.
    """
    # 使用大括号格式化字符串作为输出格式
    fmt = "{{{things}}}"
    # 初始化空列表用于存储键值对字符串
    pairs = []

    # 默认格式化字符串模板，用于每对键值对的格式化
    pfmt = "{key}: {val}"

    # 根据参数确定要处理的最大键值对数量
    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or get_option("max_seq_items") or len(seq)

    # 遍历序列中的键值对，格式化每一对的输出字符串，并添加到pairs列表中
    for k, v in list(seq.items())[:nitems]:
        pairs.append(
            pfmt.format(
                key=pprint_thing(k, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
                val=pprint_thing(v, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
            )
        )

    # 如果实际处理的键值对数量小于总数量，则输出格式化后的字符串，显示省略号
    if nitems < len(seq):
        return fmt.format(things=", ".join(pairs) + ", ...")
    else:
        # 否则输出完整的格式化字符串
        return fmt.format(things=", ".join(pairs))


def pprint_thing(
    thing: object,
    _nest_lvl: int = 0,
    escape_chars: EscapeChars | None = None,
    default_escapes: bool = False,
    quote_strings: bool = False,
    max_seq_items: int | None = None,
) -> str:
    """
    This function is the sanctioned way of converting objects
    to a string representation and properly handles nested sequences.

    Parameters
    ----------
    thing : anything to be formatted
    _nest_lvl : internal use only. pprint_thing() is mutually-recursive
        with pprint_sequence, this argument is used to keep track of the
        current nesting level, and limit it.
    escape_chars : list[str] or Mapping[str, str], optional
        Characters to escape. If a Mapping is passed the values are the
        replacements
    default_escapes : bool, default False
        Whether the input escape characters replaces or adds to the defaults
    max_seq_items : int or None, default None
        Pass through to other pretty printers to limit sequence printing

    Returns
    -------
    str
    """

    # 将输入的对象转换为经过转义处理的字符串表示
    def as_escaped_string(
        thing: Any, escape_chars: EscapeChars | None = escape_chars
    ) -> str:
        # 默认转义字符映射，用于替换特定字符的转义序列
        translate = {"\t": r"\t", "\n": r"\n", "\r": r"\r"}

        # 如果传入的转义字符是映射类型，则根据default_escapes决定是替换还是添加
        if isinstance(escape_chars, Mapping):
            if default_escapes:
                translate.update(escape_chars)
            else:
                translate = escape_chars  # type: ignore[assignment]
            escape_chars = list(escape_chars.keys())
        else:
            escape_chars = escape_chars or ()

        # 对输入的对象应用转义字符的映射
        result = str(thing)
        for c in escape_chars:
            result = result.replace(c, translate[c])
        return result

    # 如果对象具有__next__方法，直接返回其字符串表示
    if hasattr(thing, "__next__"):
        return str(thing)
    # 如果对象是映射类型且未超过嵌套层数限制，则调用_pprint_dict处理
    elif isinstance(thing, Mapping) and _nest_lvl < get_option(
        "display.pprint_nest_depth"
    ):
        result = _pprint_dict(
            thing, _nest_lvl, quote_strings=True, max_seq_items=max_seq_items
        )
        # 返回格式化后的字符串表示
        return result
    # 如果 `thing` 是一个序列并且嵌套级别 `_nest_lvl` 小于设定的最大嵌套深度，
    # 调用 `_pprint_seq` 函数对序列进行格式化输出
    elif is_sequence(thing) and _nest_lvl < get_option("display.pprint_nest_depth"):
        result = _pprint_seq(
            thing,  # 将 `thing` 作为第一个参数传递给 `_pprint_seq` 函数
            _nest_lvl,
            escape_chars=escape_chars,
            quote_strings=quote_strings,
            max_seq_items=max_seq_items,
        )
    # 如果 `thing` 是字符串并且需要在输出中引用字符串（quote_strings=True），
    # 将 `thing` 转换成转义后的字符串，并用单引号包裹
    elif isinstance(thing, str) and quote_strings:
        result = f"'{as_escaped_string(thing)}'"
    # 否则，将 `thing` 转换成转义后的字符串
    else:
        result = as_escaped_string(thing)

    # 返回处理后的结果
    return result
# 定义函数，接收一个对象并返回其Unicode表示
def pprint_thing_encoded(
    object: object, encoding: str = "utf-8", errors: str = "replace"
) -> bytes:
    # 获取对象的Unicode表示
    value = pprint_thing(object)  # get unicode representation of object
    # 将Unicode字符串编码为指定编码的字节流
    return value.encode(encoding, errors)


# 定义函数，根据参数决定是否启用数据资源格式化输出
def enable_data_resource_formatter(enable: bool) -> None:
    # 如果没有导入IPython模块，则退出函数
    if "IPython" not in sys.modules:
        # definitely not in IPython
        return
    from IPython import get_ipython

    # 获取IPython对象，忽略静态分析器的未类型化调用错误
    ip = get_ipython()  # type: ignore[no-untyped-call]
    # 如果IPython对象为空，则退出函数
    if ip is None:
        # still not in IPython
        return

    # 获取IPython的显示格式化器
    formatters = ip.display_formatter.formatters
    # 定义数据资源的MIME类型
    mimetype = "application/vnd.dataresource+json"

    # 如果启用标志为True
    if enable:
        # 如果数据资源的MIME类型不在格式化器中
        if mimetype not in formatters:
            # 导入所需的类和函数
            from IPython.core.formatters import BaseFormatter
            from traitlets import ObjectName

            # 定义数据资源格式化器类
            class TableSchemaFormatter(BaseFormatter):
                print_method = ObjectName("_repr_data_resource_")
                _return_type = (dict,)

            # 将数据资源格式化器注册到IPython中
            formatters[mimetype] = TableSchemaFormatter()
        # 启用数据资源格式化器（如果先前已禁用）
        formatters[mimetype].enabled = True
    # 如果启用标志为False且数据资源的MIME类型在格式化器中
    elif mimetype in formatters:
        # 禁用数据资源格式化器
        formatters[mimetype].enabled = False


# 定义函数，对给定的对象进行默认格式化输出
def default_pprint(thing: Any, max_seq_items: int | None = None) -> str:
    # 调用pprint_thing函数进行对象的格式化输出
    return pprint_thing(
        thing,
        escape_chars=("\t", "\r", "\n"),
        quote_strings=True,
        max_seq_items=max_seq_items,
    )


# 定义函数，返回对象的摘要信息的格式化字符串
def format_object_summary(
    obj: ListLike,
    formatter: Callable,
    is_justify: bool = True,
    name: str | None = None,
    indent_for_name: bool = True,
    line_break_each_value: bool = False,
) -> str:
    """
    Return the formatted obj as a unicode string

    Parameters
    ----------
    obj : object
        must be iterable and support __getitem__
    formatter : callable
        string formatter for an element
    is_justify : bool
        should justify the display
    name : name, optional
        defaults to the class name of the obj
    indent_for_name : bool, default True
        Whether subsequent lines should be indented to
        align with the name.
    line_break_each_value : bool, default False
        If True, inserts a line break for each value of ``obj``.
        If False, only break lines when the a line of values gets wider
        than the display width.

    Returns
    -------
    summary string
    """
    # 获取控制台的宽度和显示选项
    display_width, _ = get_console_size()
    # 如果获取不到控制台宽度，则使用默认宽度80
    if display_width is None:
        display_width = get_option("display.width") or 80
    # 如果未提供名称，则使用对象的类名作为名称
    if name is None:
        name = type(obj).__name__

    # 根据名称是否缩进来定义空白字符
    if indent_for_name:
        name_len = len(name)
        space1 = f'\n{(" " * (name_len + 1))}'
        space2 = f'\n{(" " * (name_len + 2))}'
    else:
        space1 = "\n"
        space2 = "\n "  # space for the opening '['

    # 获取对象的长度
    n = len(obj)
    # 如果需要在每个值上垂直对齐对象，我们需要在值之间使用换行符，并缩进值
    sep = ",\n " + " " * len(name)
else:
    # 否则，只用逗号分隔值
    sep = ","

# 获取最大序列项数，可以由用户设置或默认为对象长度 n
max_seq_items = get_option("display.max_seq_items") or n

# 检查是否显示被截断的序列
is_truncated = n > max_seq_items

# 获取调整函数，用于处理Unicode字符的宽度调整
adj = get_adjustment()

def _extend_line(
    s: str, line: str, value: str, display_width: int, next_line_prefix: str
) -> tuple[str, str]:
    # 如果当前行加上值超过了显示宽度，则将当前行添加到结果字符串 s 中并重新开始新行
    if adj.len(line.rstrip()) + adj.len(value.rstrip()) >= display_width:
        s += line.rstrip()
        line = next_line_prefix
    # 添加当前值到当前行
    line += value
    return s, line

def best_len(values: list[str]) -> int:
    # 计算值列表中最长值的长度
    if values:
        return max(adj.len(x) for x in values)
    else:
        return 0

# 用于表示序列结束的字符串
close = ", "

if n == 0:
    # 如果序列长度为 0，生成空序列的摘要字符串
    summary = f"[]{close}"
elif n == 1 and not line_break_each_value:
    # 如果序列长度为 1 并且不需要在每个值之间换行，则生成只包含第一个值的摘要字符串
    first = formatter(obj[0])
    summary = f"[{first}]{close}"
elif n == 2 and not line_break_each_value:
    # 如果序列长度为 2 并且不需要在每个值之间换行，则生成包含第一个和最后一个值的摘要字符串
    first = formatter(obj[0])
    last = formatter(obj[-1])
    summary = f"[{first}, {last}]{close}"

# 返回生成的摘要字符串
return summary
def _justify(
    head: list[Sequence[str]], tail: list[Sequence[str]]
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """
    Justify items in head and tail, so they are right-aligned when stacked.

    Parameters
    ----------
    head : list-like of list-likes of strings
        List of sequences where each sequence contains strings to be justified.
    tail : list-like of list-likes of strings
        List of sequences where each sequence contains strings to be justified.

    Returns
    -------
    tuple of list of tuples of strings
        Tuple containing two lists of tuples. Each tuple represents the
        sequences in head and tail respectively, with strings right-aligned
        according to the maximum length within each position.

    Examples
    --------
    >>> _justify([["a", "b"]], [["abc", "abcd"]])
    ([('  a', '   b')], [('abc', 'abcd')])
    """
    combined = head + tail

    # For each position in the sequences in ``combined``,
    # find the length of the largest string.
    max_length = [0] * len(combined[0])
    for inner_seq in combined:
        length = [len(item) for item in inner_seq]
        max_length = [max(x, y) for x, y in zip(max_length, length)]

    # Justify each item in each list-like in head and tail using max_length
    head_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length)) for seq in head
    ]
    tail_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length)) for seq in tail
    ]
    return head_tuples, tail_tuples


class PrettyDict(dict[_KT, _VT]):
    """
    Dict extension to support abbreviated __repr__
    """

    def __repr__(self) -> str:
        """
        Return string representation of the PrettyDict instance.
        """
        return pprint_thing(self)


class _TextAdjustment:
    def __init__(self) -> None:
        self.encoding = get_option("display.encoding")

    def len(self, text: str) -> int:
        """
        Return the length of a string considering its display width.

        Parameters
        ----------
        text : str
            The string whose length is to be calculated.

        Returns
        -------
        int
            Length of the string considering its display width.
        """
        return len(text)

    def justify(self, texts: Any, max_len: int, mode: str = "right") -> list[str]:
        """
        Justify strings or list-like objects based on specified mode and maximum length.

        Parameters
        ----------
        texts : Any
            String or list-like object containing strings to be justified.
        max_len : int
            Maximum length to which each string should be justified.
        mode : str, optional
            Justification mode: 'left', 'center', or 'right' (default).

        Returns
        -------
        list[str]
            Justified strings or list-like object.
        """
        if mode == "left":
            return [x.ljust(max_len) for x in texts]
        elif mode == "center":
            return [x.center(max_len) for x in texts]
        else:
            return [x.rjust(max_len) for x in texts]

    def adjoin(self, space: int, *lists: Any, **kwargs: Any) -> str:
        """
        Adjoin multiple lists into a single string representation with specified spacing.

        Parameters
        ----------
        space : int
            Number of spaces to use as separation between elements.
        *lists : Any
            Lists or list-like objects to be joined.
        **kwargs : Any
            Additional keyword arguments passed to the adjoin function.

        Returns
        -------
        str
            Joined string representation of the lists.
        """
        return adjoin(space, *lists, strlen=self.len, justfunc=self.justify, **kwargs)


class _EastAsianTextAdjustment(_TextAdjustment):
    def __init__(self) -> None:
        super().__init__()
        if get_option("display.unicode.ambiguous_as_wide"):
            self.ambiguous_width = 2
        else:
            self.ambiguous_width = 1

        # Definition of East Asian Width
        # https://unicode.org/reports/tr11/
        # Ambiguous width can be changed by option
        self._EAW_MAP = {"Na": 1, "N": 1, "W": 2, "F": 2, "H": 1}

    def len(self, text: str) -> int:
        """
        Calculate display width considering unicode East Asian Width.

        Parameters
        ----------
        text : str
            The string whose display width is to be calculated.

        Returns
        -------
        int
            Display width of the string considering East Asian Width properties.
        """
        if not isinstance(text, str):
            return len(text)

        return sum(
            self._EAW_MAP.get(east_asian_width(c), self.ambiguous_width) for c in text
        )
    # 在当前类中定义一个方法 justify，接收参数 texts（字符串迭代器）、max_len（最大长度）、mode（对齐模式，默认为右对齐），返回一个字符串列表
    def justify(
        self, texts: Iterable[str], max_len: int, mode: str = "right"
    ) -> list[str]:
        # 定义一个内部函数 _get_pad，计算每个字符串需要填充的空格数，考虑到东亚宽度字符
        def _get_pad(t: str) -> int:
            return max_len - self.len(t) + len(t)

        # 根据 mode 参数选择对齐方式，并返回对齐后的字符串列表
        if mode == "left":
            # 对每个字符串 x 左对齐，填充空格数由 _get_pad 函数返回
            return [x.ljust(_get_pad(x)) for x in texts]
        elif mode == "center":
            # 对每个字符串 x 居中对齐，填充空格数由 _get_pad 函数返回
            return [x.center(_get_pad(x)) for x in texts]
        else:
            # 对每个字符串 x 右对齐（默认模式），填充空格数由 _get_pad 函数返回
            return [x.rjust(_get_pad(x)) for x in texts]
# 定义函数，返回一个文本调整对象 _TextAdjustment
def get_adjustment() -> _TextAdjustment:
    # 获取显示选项中的东亚宽度设置
    use_east_asian_width = get_option("display.unicode.east_asian_width")
    
    # 如果使用东亚宽度设置为真，则返回东亚文本调整对象 _EastAsianTextAdjustment
    if use_east_asian_width:
        return _EastAsianTextAdjustment()
    # 否则返回默认文本调整对象 _TextAdjustment
    else:
        return _TextAdjustment()
```