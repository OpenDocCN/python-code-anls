# `.\pytorch\torch\package\glob_group.py`

```py
# mypy: allow-untyped-defs
# 引入 re 模块，用于正则表达式操作
import re
# 从 typing 模块中引入 Iterable 和 Union 类型
from typing import Iterable, Union

# 定义 GlobPattern 类型，可以是字符串或字符串的可迭代集合
GlobPattern = Union[str, Iterable[str]]


class GlobGroup:
    """A set of patterns that candidate strings will be matched against.

    A candidate is composed of a list of segments separated by ``separator``, e.g. "foo.bar.baz".

    A pattern contains one or more segments. Segments can be:
        - A literal string (e.g. "foo"), which matches exactly.
        - A string containing a wildcard (e.g. "torch*", or "foo*baz*"). The wildcard matches
          any string, including the empty string.
        - A double wildcard ("**"). This matches against zero or more complete segments.

    Examples:
        ``torch.**``: matches ``torch`` and all its submodules, e.g. ``torch.nn`` and ``torch.nn.functional``.
        ``torch.*``: matches ``torch.nn`` or ``torch.functional``, but not ``torch.nn.functional``.
        ``torch*.**``: matches ``torch``, ``torchvision``, and all their submodules.

    A candidates will match the ``GlobGroup`` if it matches any of the ``include`` patterns and
    none of the ``exclude`` patterns.

    Args:
        include (Union[str, Iterable[str]]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will match if it matches *any* include pattern
        exclude (Union[str, Iterable[str]]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will be excluded from matching if it matches *any* exclude pattern.
        separator (str): A string that delimits segments in candidates and
            patterns. By default this is "." which corresponds to how modules are
            named in Python. Another common value for this is "/", which is
            the Unix path separator.
    """

    def __init__(
        self, include: GlobPattern, *, exclude: GlobPattern = (), separator: str = "."
    ):
        # 调试信息，用于显示对象的 include 和 exclude 属性
        self._dbg = f"GlobGroup(include={include}, exclude={exclude})"
        # 将 include 和 exclude 转换为正则表达式对象列表
        self.include = GlobGroup._glob_list(include, separator)
        self.exclude = GlobGroup._glob_list(exclude, separator)
        self.separator = separator

    def __str__(self):
        # 返回调试信息字符串表示
        return self._dbg

    def __repr__(self):
        # 返回调试信息字符串表示
        return self._dbg

    def matches(self, candidate: str) -> bool:
        # 将 candidate 前加上分隔符，以匹配 include 和 exclude 中的模式
        candidate = self.separator + candidate
        # 判断 candidate 是否匹配任何 include 中的模式，并且不匹配任何 exclude 中的模式
        return any(p.fullmatch(candidate) for p in self.include) and all(
            not p.fullmatch(candidate) for p in self.exclude
        )

    @staticmethod
    def _glob_list(elems: GlobPattern, separator: str = "."):
        # 如果 elems 是字符串，则将其转换为正则表达式对象列表
        if isinstance(elems, str):
            return [GlobGroup._glob_to_re(elems, separator)]
        else:
            # 如果 elems 是可迭代对象，则将每个元素转换为正则表达式对象并组成列表
            return [GlobGroup._glob_to_re(e, separator) for e in elems]

    @staticmethod
    def _glob_to_re(pattern: str, separator: str = "."):
        # 将 glob 模式转换为正则表达式对象

        # 定义将单个组件转换为正则表达式的函数
        def component_to_re(component):
            # 如果组件中包含 "**"，处理特殊情况
            if "**" in component:
                # "**" 只能作为整个路径段的一部分出现，否则抛出异常
                if component == "**":
                    return "(" + re.escape(separator) + "[^" + separator + "]+)*"
                else:
                    raise ValueError("** can only appear as an entire path segment")
            else:
                # 否则，将通配符 "*" 转换为正则表达式
                return re.escape(separator) + ("[^" + separator + "]*").join(
                    re.escape(x) for x in component.split("*")
                )

        # 将每个组件转换为正则表达式并拼接成最终的正则表达式字符串
        result = "".join(component_to_re(c) for c in pattern.split(separator))
        # 编译最终的正则表达式字符串为正则表达式对象并返回
        return re.compile(result)
```