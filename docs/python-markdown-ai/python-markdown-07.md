# PythonMarkdown源码解析 7

# `/markdown/markdown/util.py`

这是一个 Python 实现的 John Gruber 的 Markdown 代码。Markdown 是一种轻量级的标记语言，可以轻松地将普通文本转换为 HTML，具有良好的可读性和可维护性。

该代码维护了一个名为“Python Markdown”的 GitHub 仓库，其中有大量关于 Markdown 的教程、文档和示例。这个仓库的目的是让 Python 用户更方便地使用 Markdown。

该代码还包含了 Python Markdown 库的依赖信息，这个库提供了许多 Markdown 处理的功能，如插入链接、图片、列表、标题等。


```py
# Python Markdown

# A Python implementation of John Gruber's Markdown.

# Documentation: https://python-markdown.github.io/
# GitHub: https://github.com/Python-Markdown/markdown/
# PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
```

这段代码是一个 Python 模块，其中包含了一些 Python 标准库中产生的导入引用、函数和类。具体来说，它引入了 `re` 模块 (Python 标准库中的正则表达式模块),`sys` 模块 (Python 标准库中的系统模块),`warnings` 模块 (Python 标准库中的警告模块),`lru_cache` 函数 (Python 标准库中的自旋缓存函数)，以及一些函数和类。

具体来说，这段代码中的函数和类包括：

- `contact()` 函数：这个函数接受一个 Contact 类对象作为参数，这个类可能是用来表示联系人的。不过，由于这个函数没有定义具体的实现，因此无法确定它具体是如何工作的。
- `remove_duplicates()` 函数：这个函数接受一个列表作为参数，这个列表可能是多个不同的 Contact 对象。它使用了一些正则表达式技术，将列表中的每个 Contact 对象与列表中的其他 Contact 对象进行比较，如果两个 Contact 对象相同，则保留其中的一个，否则删除其中的一个。
- `lru_cache()` 函数：这个函数是一个自旋缓存函数，它使用了 Python 标准库中的 `lru_cache` 类。它会在每次创建缓存对象时，尝试使用最近最少使用的原则来选择缓存对象的存储位置，从而提高缓存的效率。

由于这段代码中包含了很多 Python 标准库中的模块和函数，因此它的具体作用可能取决于具体的应用场景。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
This module contains various contacts, classes and functions which get referenced and used
throughout the code base.
"""

from __future__ import annotations

import re
import sys
import warnings
from functools import wraps, lru_cache
```

这段代码是一个用于遍历并计数 Python 中所有函数的 Python 函数。它使用了一个名为 `itertools` 的标准库函数，用于从一系列迭代器中产生迭代器。函数定义了一个名为 `_T` 的类型变量，但没有具体的类型限制。

接下来，该函数通过 `typing.NamedTuple` 创建了一个名为 `FunctionCount` 的命名元组类型。这个类型包含两个属性：`count` 和 `function_names`。属性 `count` 是一个整数，用于存储计数器的值；属性 `function_names` 是一个字符串，存储了所有函数的名称。

函数的主要部分在 `if TYPE_CHECKING` 语句下，表示只有在 `markdown` 和 `xml.etree.ElementTree` 库正在使用时，函数才会输出它们创建的 `Markdown` 和 `ElementTree` 对象。这意味着，如果您的程序中没有安装这些库，您不会看到函数输出任何内容。

接下来，该函数定义了一个名为 `FunctionCount` 的函数，这个函数会遍历所有定义在 `__all__` 列表中的函数，并执行以下操作：

1. 使用 `count` 初始化一个计数器变量 `count`；
2. 遍历 `all_functions` 列表中的函数；
3. 对于每个函数，使用 `markdown` 库创建一个新的 `Markdown` 对象，并将计数器 `count` 加 1；
4. 遍历 `function_names` 列表，将当前函数名称存储在 `FunctionCount` 对象中。




```py
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown
    import xml.etree.ElementTree as etree

_T = TypeVar('_T')


"""
Constants you might want to modify
-----------------------------------------------------------------------------
"""


```

This code defines a list of HTML tags that are considered as block-level elements. These tags are used for layout and formatting purposes within a Markdown document. The list is created as an updated version of the `block_level_elements` attribute of the [`Markdown`] class, which is a part of the [Markdown](https:// markdown-js.org/) JavaScript library.

The `block_level_elements` attribute is a union of multiple lists, where each list corresponds to a different kind of block-level element. The main purpose of this list is to provide a consistent and comprehensive set of HTML elements that can be safely used in Markdown, avoiding the穿插 of non-block-level HTML elements into the main document structure.

The list is ordered by semantic content, starting with the highest semantic priority ones first. The elements in the list are耗


```py
BLOCK_LEVEL_ELEMENTS: list[str] = [
    # Elements which are invalid to wrap in a `<p>` tag.
    # See https://w3c.github.io/html/grouping-content.html#the-p-element
    'address', 'article', 'aside', 'blockquote', 'details', 'div', 'dl',
    'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3',
    'h4', 'h5', 'h6', 'header', 'hgroup', 'hr', 'main', 'menu', 'nav', 'ol',
    'p', 'pre', 'section', 'table', 'ul',
    # Other elements which Markdown should not be mucking up the contents of.
    'canvas', 'colgroup', 'dd', 'body', 'dt', 'group', 'html', 'iframe', 'li', 'legend',
    'math', 'map', 'noscript', 'output', 'object', 'option', 'progress', 'script',
    'style', 'summary', 'tbody', 'td', 'textarea', 'tfoot', 'th', 'thead', 'tr', 'video'
]
"""
List of HTML tags which get treated as block-level elements. Same as the `block_level_elements`
attribute of the [`Markdown`][markdown.Markdown] class. Generally one should use the
```

这段代码定义了一个名为 "attribute" 的类属性。它的作用是用于在代码中插入一些占位符（placeholder），以保持兼容性，同时避免在代码中直接输出这些占位符。

具体来说，这段代码定义了两个占位符：STX 和 ETX。它们分别表示"开始文本"和"结束文本"的标记。此外，还定义了一个 INLINE_PLACEHOLDER_PREFIX 变量，用于表示将占位符插入到 INLINE_PLACEHOLDER 中的前缀。最后一个定义是 INLINE_PLACEHOLDER，表示用于在 INLINE_PLACEHOLDER_PREFIX 中的占位符，以及一个 regular expression (re) for 匹配 INLINE_PLACEHOLDER 中的占位符。

这个 code 文件的主要作用是定义一个可以插入文本的占位符，以便在需要插入文本时使用。通过将占位符与INLINE_PLACEHOLDER_PREFIX 组合，可以在原始代码中安全地插入占位符。而且，这个占位符也可以在将来的版本中被用来覆盖，而不需要修改现有的代码。


```py
attribute on the class. This remains for compatibility with older extensions.
"""

# Placeholders
STX = '\u0002'
""" "Start of Text" marker for placeholder templates. """
ETX = '\u0003'
""" "End of Text" marker for placeholder templates. """
INLINE_PLACEHOLDER_PREFIX = STX+"klzzwxh:"
""" Prefix for inline placeholder template. """
INLINE_PLACEHOLDER = INLINE_PLACEHOLDER_PREFIX + "%s" + ETX
""" Placeholder template for stashed inline text. """
INLINE_PLACEHOLDER_RE = re.compile(INLINE_PLACEHOLDER % r'([0-9]+)')
""" Regular Expression which matches inline placeholders. """
AMP_SUBSTITUTE = STX+"amp"+ETX
```

这段代码是一个Python模板，其中包含了一些placeholder（占位符）模板，用于在输出HTML时替换一些无用的占位符，以简化代码和提高性能。

具体来说，这段代码定义了两个模板：HTML_PLACEHOLDER和HTML_PLACEHOLDER_RE。其中，HTML_PLACEHOLDER模板用于将占位符替换为HTML实体，而HTML_PLACEHOLDER_RE模板则是一个正则表达式，用于匹配HTML中的占位符，并将它们替换为实体。

此外，还定义了一个名为TAG_PLACEHOLDER的模板，用于将占位符替换为标签。这个模板通常在应用程序中使用，将占位符替换为实际标签，以提高代码的可读性和可维护性。

最后，定义了一个名为RTL_BIDI_RANGES的元组，包含了一些机读的BIDI范围，用于将一些特定的字符匹配为特定类型的内容。


```py
""" Placeholder template for HTML entities. """
HTML_PLACEHOLDER = STX + "wzxhzdk:%s" + ETX
""" Placeholder template for raw HTML. """
HTML_PLACEHOLDER_RE = re.compile(HTML_PLACEHOLDER % r'([0-9]+)')
""" Regular expression which matches HTML placeholders. """
TAG_PLACEHOLDER = STX + "hzzhzkh:%s" + ETX
""" Placeholder template for tags. """


# Constants you probably do not need to change
# -----------------------------------------------------------------------------

RTL_BIDI_RANGES = (
    ('\u0590', '\u07FF'),
    # Hebrew (0590-05FF), Arabic (0600-06FF),
    # Syriac (0700-074F), Arabic supplement (0750-077F),
    # Thaana (0780-07BF), Nko (07C0-07FF).
    ('\u2D30', '\u2D7F')  # Tifinagh
)


```

此代码是一个Python代码，定义了一个名为`get_installed_extensions`的函数。函数使用了`@lru_cache(maxsize=None)`装饰，表示使用LRU缓存来避免重复的缓存。

具体来说，这个函数的作用是返回`markdown.extensions`模块中所有注册的扩展 entry_points的列表。在函数内部，使用`importlib`或`importlib_metadata`来导入`markdown.extensions`模块，并使用`metadata.entry_points`来获取该模块中所有注册的扩展 entry_points。

由于在函数内部使用了`@lru_cache(maxsize=None)`装饰，因此该函数只会在缓存中存储一次扩展 entry_points，并在缓存超时时自动去重。


```py
# AUXILIARY GLOBAL FUNCTIONS
# =============================================================================


@lru_cache(maxsize=None)
def get_installed_extensions():
    """ Return all entry_points in the `markdown.extensions` group. """
    if sys.version_info >= (3, 10):
        from importlib import metadata
    else:  # `<PY310` use backport
        import importlib_metadata as metadata
    # Only load extension entry_points once.
    return metadata.entry_points(group='markdown.extensions')


```

该代码定义了一个名为 "deprecated" 的函数，其接受一个字符串参数 "message" 和一个可选的整数参数 "stacklevel"。函数的作用是在传入一个函数或方法时，抛出一个名为 "DeprecationWarning" 的警告，警告信息中包含字符串 "this method will be removed in version X; use Y instead."。

该函数中使用了两个参数：

1. "message": 该参数表示在警告信息中要显示的警告信息。该警告信息是在函数或方法被调用时显示的。
2. "stacklevel": 该参数表示警告的栈级别。栈级别指定警告在哪个步骤出错，该值越大，警告就越接近于出错的步骤。默认值为 2。

该函数中还定义了一个内部函数 "wrapper"，该函数接受一个函数参数 "func"，并返回一个新函数 "deprecated_func"。该新函数使用 "wraps" 方法对传入的 "func" 函数进行包装，并在内部使用了 "deprecated" 函数的语法。

最后，该函数返回了 "wrapper" 函数。


```py
def deprecated(message: str, stacklevel: int = 2):
    """
    Raise a [`DeprecationWarning`][] when wrapped function/method is called.

    Usage:

    ```python
    @deprecated("This method will be removed in version X; use Y instead.")
    def some_method():
        pass
    ```py
    """
    def wrapper(func):
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                f"'{func.__name__}' is deprecated. {message}",
                category=DeprecationWarning,
                stacklevel=stacklevel
            )
            return func(*args, **kwargs)
        return deprecated_func
    return wrapper


```

这段代码是一个名为 `parseBoolValue` 的函数，它接受一个字符串参数 `value`，一个布尔参数 `fail_on_errors`，以及一个布尔参数 `preserve_none`。它的作用是解析出一个布尔值，并返回相应的结果。

函数首先检查 `value` 是否为字符串类型，如果不是，则执行以下操作：如果 `preserve_none` 为 `True`，则检查 `value` 是否为 `None`，如果是，则返回 `None`；如果 `fail_on_errors` 为 `False`，则忽略此错误，继续尝试解析。如果 `value` 不是字符串类型，则尝试将其转换为字符串，并检查它是否为 `True` 或 `False`。如果转换成功，则返回相应的值；如果转换失败，则尝试根据 `fail_on_errors` 的值来决定如何处理，如果 `fail_on_errors` 为 `True`，则抛出 `ValueError`，否则返回 `None`。


```py
def parseBoolValue(value: str | None, fail_on_errors: bool = True, preserve_none: bool = False) -> bool | None:
    """Parses a string representing a boolean value. If parsing was successful,
       returns `True` or `False`. If `preserve_none=True`, returns `True`, `False`,
       or `None`. If parsing was not successful, raises `ValueError`, or, if
       `fail_on_errors=False`, returns `None`."""
    if not isinstance(value, str):
        if preserve_none and value is None:
            return value
        return bool(value)
    elif preserve_none and value.lower() == 'none':
        return None
    elif value.lower() in ('true', 'yes', 'y', 'on', '1'):
        return True
    elif value.lower() in ('false', 'no', 'n', 'off', '0', 'none'):
        return False
    elif fail_on_errors:
        raise ValueError('Cannot parse bool value: %r' % value)


```

这段代码定义了一个名为 `code_escape` 的函数，用于将给定的字符串中的 HTML 标签转义。函数接受一个字符串参数 `text`，并返回转义后的字符串。

函数内部包含一个 if 语句，用于检查给定的字符串是否包含 HTML 标签。如果是，则函数会执行一系列 if 语句，检查要转义的字符是否包含在给定的转义字符串中。如果是，则替换字符，并将结果返回。

函数还包含一个名为 `_get_stack_depth` 的内部函数，用于获取当前栈层的深度。该函数接受一个参数 `size`，代表要获取的层数。函数返回当前栈层的深度，也可以通过修改 `size` 参数来改变获取栈层数的深度。


```py
def code_escape(text: str) -> str:
    """HTML escape a string of code."""
    if "&" in text:
        text = text.replace("&", "&amp;")
    if "<" in text:
        text = text.replace("<", "&lt;")
    if ">" in text:
        text = text.replace(">", "&gt;")
    return text


def _get_stack_depth(size: int = 2) -> int:
    """Get current stack depth, performantly.
    """
    frame = sys._getframe(size)

    for size in count(size):
        frame = frame.f_back
        if not frame:
            return size


```

这段代码是一个函数，名为 `nearing_recursion_limit()`，但事实上它是一个递归函数，因此我们需要注意它实际上是一个无限递归的函数。

函数的作用是判断当前栈的深度是否已经达到了栈的最大深度（100），如果已经达到了栈的最大深度，那么函数返回 `True`，否则返回 `False`。

该函数的实现依赖于 `sys.getrecursionlimit()` 和 `_get_stack_depth()` 函数，它们用于获取栈的最大深度和当前栈的深度。

需要注意的是，该函数的实现存在栈溢出的风险，因此函数应该被谨慎地使用。


```py
def nearing_recursion_limit() -> bool:
    """Return true if current stack depth is within 100 of maximum limit."""
    return sys.getrecursionlimit() - _get_stack_depth() < 100


# MISC AUXILIARY CLASSES
# =============================================================================


class AtomicString(str):
    """A string which should not be further processed."""
    pass


class Processor:
    """ The base class for all processors.

    Attributes:
        Processor.md: The `Markdown` instance passed in an initialization.

    Arguments:
        md: The `Markdown` instance this processor is a part of.

    """
    def __init__(self, md: Markdown | None = None):
        self.md = md


```

This is a simple class for storing HTML segments and replacing placeholders with their corresponding stash HTML blocks.

It has an initializer which creates an instance of the class and initializes its attributes.

The `store` method takes an HTML segment and returns a placeholder string. It recursively creates and returns a new placeholder string each time an inline HTML segment is processed.

The `reset` method resets the stash by clearing its attributes and removing all processed HTML segments.

The `get_placeholder` method returns a placeholder string for a given stash index.

The `store_tag` method takes a tag, its attributes, the left and right indexes, and the stash index. It recursively creates and returns a placeholder string for the given tag and attributes, using the `TAG_PLACEHOLDER` and `HTML_PLACEHOLDER` variables.

Note that the `etree.Element` class from the `etree` module is used for parsing and creating `Element` objects from the HTML. The `TagData` class is used for storing the tag data for each element in the `store_tag` method.


```py
if TYPE_CHECKING:  # pragma: no cover
    class TagData(TypedDict):
        tag: str
        attrs: dict[str, str]
        left_index: int
        right_index: int


class HtmlStash:
    """
    This class is used for stashing HTML objects that we extract
    in the beginning and replace with place-holders.
    """

    def __init__(self):
        """ Create an `HtmlStash`. """
        self.html_counter = 0  # for counting inline html segments
        self.rawHtmlBlocks: list[str | etree.Element] = []
        self.tag_counter = 0
        self.tag_data: list[TagData] = []  # list of dictionaries in the order tags appear

    def store(self, html: str | etree.Element) -> str:
        """
        Saves an HTML segment for later reinsertion.  Returns a
        placeholder string that needs to be inserted into the
        document.

        Keyword arguments:
            html: An html segment.

        Returns:
            A placeholder string.

        """
        self.rawHtmlBlocks.append(html)
        placeholder = self.get_placeholder(self.html_counter)
        self.html_counter += 1
        return placeholder

    def reset(self) -> None:
        """ Clear the stash. """
        self.html_counter = 0
        self.rawHtmlBlocks = []

    def get_placeholder(self, key: int) -> str:
        return HTML_PLACEHOLDER % key

    def store_tag(self, tag: str, attrs: dict[str, str], left_index: int, right_index: int) -> str:
        """Store tag data and return a placeholder."""
        self.tag_data.append({'tag': tag, 'attrs': attrs,
                              'left_index': left_index,
                              'right_index': right_index})
        placeholder = TAG_PLACEHOLDER % str(self.tag_counter)
        self.tag_counter += 1  # equal to the tag's index in `self.tag_data`
        return placeholder


```

This is a class `_PriorityRegistry` that appears to be used to store and manage a list of items with priorities. It has methods for registering new items and removing items from the registry, as well as sorting the items based on their priorities. The class also has a `_sort` method that is used internally to sort the items, but should never be called by users.


```py
# Used internally by `Registry` for each item in its sorted list.
# Provides an easier to read API when editing the code later.
# For example, `item.name` is more clear than `item[0]`.
class _PriorityItem(NamedTuple):
    name: str
    priority: float


class Registry(Generic[_T]):
    """
    A priority sorted registry.

    A `Registry` instance provides two public methods to alter the data of the
    registry: `register` and `deregister`. Use `register` to add items and
    `deregister` to remove items. See each method for specifics.

    When registering an item, a "name" and a "priority" must be provided. All
    items are automatically sorted by "priority" from highest to lowest. The
    "name" is used to remove ("deregister") and get items.

    A `Registry` instance it like a list (which maintains order) when reading
    data. You may iterate over the items, get an item and get a count (length)
    of all items. You may also check that the registry contains an item.

    When getting an item you may use either the index of the item or the
    string-based "name". For example:

        registry = Registry()
        registry.register(SomeItem(), 'itemname', 20)
        # Get the item by index
        item = registry[0]
        # Get the item by name
        item = registry['itemname']

    When checking that the registry contains an item, you may use either the
    string-based "name", or a reference to the actual item. For example:

        someitem = SomeItem()
        registry.register(someitem, 'itemname', 20)
        # Contains the name
        assert 'itemname' in registry
        # Contains the item instance
        assert someitem in registry

    The method `get_index_for_name` is also available to obtain the index of
    an item using that item's assigned "name".
    """

    def __init__(self):
        self._data: dict[str, _T] = {}
        self._priority: list[_PriorityItem] = []
        self._is_sorted = False

    def __contains__(self, item: str | _T) -> bool:
        if isinstance(item, str):
            # Check if an item exists by this name.
            return item in self._data.keys()
        # Check if this instance exists.
        return item in self._data.values()

    def __iter__(self) -> Iterator[_T]:
        self._sort()
        return iter([self._data[k] for k, p in self._priority])

    @overload
    def __getitem__(self, key: str | int) -> _T:  # pragma: no cover
        ...

    @overload
    def __getitem__(self, key: slice) -> Registry[_T]:  # pragma: no cover
        ...

    def __getitem__(self, key: str | int | slice) -> _T | Registry[_T]:
        self._sort()
        if isinstance(key, slice):
            data: Registry[_T] = Registry()
            for k, p in self._priority[key]:
                data.register(self._data[k], k, p)
            return data
        if isinstance(key, int):
            return self._data[self._priority[key].name]
        return self._data[key]

    def __len__(self) -> int:
        return len(self._priority)

    def __repr__(self):
        return '<{}({})>'.format(self.__class__.__name__, list(self))

    def get_index_for_name(self, name: str) -> int:
        """
        Return the index of the given name.
        """
        if name in self:
            self._sort()
            return self._priority.index(
                [x for x in self._priority if x.name == name][0]
            )
        raise ValueError('No item named "{}" exists.'.format(name))

    def register(self, item: _T, name: str, priority: float) -> None:
        """
        Add an item to the registry with the given name and priority.

        Arguments:
            item: The item being registered.
            name: A string used to reference the item.
            priority: An integer or float used to sort against all items.

        If an item is registered with a "name" which already exists, the
        existing item is replaced with the new item. Treat carefully as the
        old item is lost with no way to recover it. The new item will be
        sorted according to its priority and will **not** retain the position
        of the old item.
        """
        if name in self:
            # Remove existing item of same name first
            self.deregister(name)
        self._is_sorted = False
        self._data[name] = item
        self._priority.append(_PriorityItem(name, priority))

    def deregister(self, name: str, strict: bool = True) -> None:
        """
        Remove an item from the registry.

        Set `strict=False` to fail silently. Otherwise a [`ValueError`][] is raised for an unknown `name`.
        """
        try:
            index = self.get_index_for_name(name)
            del self._priority[index]
            del self._data[name]
        except ValueError:
            if strict:
                raise

    def _sort(self) -> None:
        """
        Sort the registry by priority from highest to lowest.

        This method is called internally and should never be explicitly called.
        """
        if not self._is_sorted:
            self._priority.sort(key=lambda item: item.priority, reverse=True)
            self._is_sorted = True

```

# `/markdown/markdown/__init__.py`

该代码是一个Python实现John Gruber所写的Markdown（Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式）的代码。它包括一个文档、GitHub链接、PyPI链接，说明了这个Markdown项目的来源和维护者。

该代码的作用是提供一个Markdown实例，用于将Markdown文档转换为HTML格式的代码。通过运行这个代码，用户可以轻松地将Markdown文档在线转换为HTML格式，从而方便地将Markdown文档分享给其他人或将其嵌入到网站中。


```py
# Python Markdown

# A Python implementation of John Gruber's Markdown.

# - Documentation: https://python-markdown.github.io/
# - GitHub: https://github.com/Python-Markdown/markdown/
# - PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# - Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# - Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
```

这段代码是一个Python类的导出，定义了Python-Markdown库中的两个公有函数，分别是`markdown.markdown`和`markdown.markdownFromFile`。这两个函数都包装了`markdown.Markdown`类，可以用来生成Markdown格式的文本。

函数的作用是将`markdown.Markdown`类中的方法暴露给外部使用，使得外部可以使用这两个函数来生成Markdown文本。这个库提供了许多Markdown扩展，如语法 highlighting、任务列表、链接、图片、视频等，通过这个库可以方便地生成Markdown文档，使生成的文档更易于阅读。


```py
# - Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
Python-Markdown provides two public functions ([`markdown.markdown`][] and [`markdown.markdownFromFile`][])
both of which wrap the public class [`markdown.Markdown`][]. All submodules support these public functions
and class and/or provide extension support.

Modules:
    core: Core functionality.
    preprocessors: Pre-processors.
    blockparser: Core Markdown block parser.
    blockprocessors: Block processors.
    treeprocessors: Tree processors.
    inlinepatterns: Inline patterns.
    postprocessors: Post-processors.
    serializers: Serializers.
    util: Utility functions.
    htmlparser: HTML parser.
    test_tools: Testing utilities.
    extensions: Markdown extensions.
```

这段代码是一个Python package，包含一个名为"Markdown"的类，以及一个名为"__all__"的列表，其中包含该package中所有导出的类和函数。

具体来说，该package中包含一个名为"Markdown"的类，这个类继承自Python标准中的"Markup"类，提供了将文本转换为Markdown格式的函数。该类还有一个名为"markdown"的静态方法，它接受一个文本参数并返回一个Markdown文档对象。此外，该package中还包含一个名为"markdownFromFile"的静态方法，它从指定文件中读取文本并返回一个Markdown文档对象。

最后，该package定义了一个名为"__all__"的列表，其中包含所有导出的类和函数。这样，当Python代码中直接使用这些导出类或函数时，就可以自动导入它们，而不需要使用"**"星号来表示。


```py
"""

from __future__ import annotations

from .core import Markdown, markdown, markdownFromFile
from .__meta__ import __version__, __version_info__  # noqa

# For backward compatibility as some extensions expect it...
from .extensions import Extension  # noqa

__all__ = ['Markdown', 'markdown', 'markdownFromFile']

```

# `/markdown/markdown/__main__.py`

该代码是一个Python实现John Gruber所写的Markdown的示例。Markdown是一种轻量级的标记语言，它可以让你以类似纯文本的方式结构化文章，非常适合在网络环境下快速创建和共享文档。

该代码将Markdown文档中的内容解析为列表，然后按照列表的语法逐行输出。这样做的好处是可以方便地将Markdown文档转换为列表形式，从而使代码更易于阅读和维护。


```py
# Python Markdown

# A Python implementation of John Gruber's Markdown.

# Documentation: https://python-markdown.github.io/
# GitHub: https://github.com/Python-Markdown/markdown/
# PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
```

这段代码是一个Python脚本，它定义了一个函数`unmark_routine_wrappers`，该函数的作用是：将一段文本内容中的所有`<script>`标签和`<script>`标签之间的内容全部去除，并返回处理后的文本内容。

具体来说，这段代码做以下几件事情：

1. 从`markdown`模块中导入`markdown.sql`，这个模块可以用来解析Markdown文本，应用一些过滤和转换，例如删除`<script>`标签和`<script>`标签之间的内容。
2. 从`optparse`模块中导入`argparse`，这个模块可以用来定义命令行参数。
3. 定义一个`unmark_routine_wrappers`函数，这个函数的参数是一个字符串`text`，表示要处理的目标文本内容。
4. 在函数内部使用`unsafe_load`函数，这个函数可以接受一个Python对象作为参数，并返回一个字符串，不会对参数类型进行检查。在这里，`unsafe_load`函数被用来加载要处理的文本内容，因为有些用户可能需要将其作为实际Python对象传递。
5. 使用`markdown.sql`函数对要处理的目标文本内容进行预处理，例如删除`<script>`标签和`<script>`标签之间的内容。
6. 使用`argparse`模块定义一个命令行参数，用于指定要处理的目标文本文件。
7. 在主程序部分，调用`unmark_routine_wrappers`函数，并将预处理后的目标文本作为参数传入，得到处理后的文本内容。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

from __future__ import annotations

import sys
import optparse
import codecs
import warnings
import markdown
try:
    # We use `unsafe_load` because users may need to pass in actual Python
    # objects. As this is only available from the CLI, the user has much
    # worse problems if an attacker can use this as an attach vector.
    from yaml import unsafe_load as yaml_load
```

It looks like the `extension_configs` parameter is a dictionary that maps Python extensions to their options, such as `'extension_configs': {'extension1': {'options1': 'option1'}, 'extension2': {'options2': 'option2'}}`
It appears that the `extension_configs` parameter is only populated if the `--extension` option is not used. If the `--extension` option is used, the `extension_configs` parameter is read from the `configfile` specified by the user, and if it is not specified, the `extension_configs` parameter is not defined.


```py
except ImportError:  # pragma: no cover
    try:
        # Fall back to PyYAML <5.1
        from yaml import load as yaml_load
    except ImportError:
        # Fall back to JSON
        from json import load as yaml_load

import logging
from logging import DEBUG, WARNING, CRITICAL

logger = logging.getLogger('MARKDOWN')


def parse_options(args=None, values=None):
    """
    Define and parse `optparse` options for command-line usage.
    """
    usage = """%prog [options] [INPUTFILE]
       (STDIN is assumed if no INPUTFILE is given)"""
    desc = "A Python implementation of John Gruber's Markdown. " \
           "https://Python-Markdown.github.io/"
    ver = "%%prog %s" % markdown.__version__

    parser = optparse.OptionParser(usage=usage, description=desc, version=ver)
    parser.add_option("-f", "--file", dest="filename", default=None,
                      help="Write output to OUTPUT_FILE. Defaults to STDOUT.",
                      metavar="OUTPUT_FILE")
    parser.add_option("-e", "--encoding", dest="encoding",
                      help="Encoding for input and output files.",)
    parser.add_option("-o", "--output_format", dest="output_format",
                      default='xhtml', metavar="OUTPUT_FORMAT",
                      help="Use output format 'xhtml' (default) or 'html'.")
    parser.add_option("-n", "--no_lazy_ol", dest="lazy_ol",
                      action='store_false', default=True,
                      help="Observe number of first item of ordered lists.")
    parser.add_option("-x", "--extension", action="append", dest="extensions",
                      help="Load extension EXTENSION.", metavar="EXTENSION")
    parser.add_option("-c", "--extension_configs",
                      dest="configfile", default=None,
                      help="Read extension configurations from CONFIG_FILE. "
                      "CONFIG_FILE must be of JSON or YAML format. YAML "
                      "format requires that a python YAML library be "
                      "installed. The parsed JSON or YAML must result in a "
                      "python dictionary which would be accepted by the "
                      "'extension_configs' keyword on the markdown.Markdown "
                      "class. The extensions must also be loaded with the "
                      "`--extension` option.",
                      metavar="CONFIG_FILE")
    parser.add_option("-q", "--quiet", default=CRITICAL,
                      action="store_const", const=CRITICAL+10, dest="verbose",
                      help="Suppress all warnings.")
    parser.add_option("-v", "--verbose",
                      action="store_const", const=WARNING, dest="verbose",
                      help="Print all warnings.")
    parser.add_option("--noisy",
                      action="store_const", const=DEBUG, dest="verbose",
                      help="Print debug messages.")

    (options, args) = parser.parse_args(args, values)

    if len(args) == 0:
        input_file = None
    else:
        input_file = args[0]

    if not options.extensions:
        options.extensions = []

    extension_configs = {}
    if options.configfile:
        with codecs.open(
            options.configfile, mode="r", encoding=options.encoding
        ) as fp:
            try:
                extension_configs = yaml_load(fp)
            except Exception as e:
                message = "Failed parsing extension config file: %s" % \
                          options.configfile
                e.args = (message,) + e.args[1:]
                raise

    opts = {
        'input': input_file,
        'output': options.filename,
        'extensions': options.extensions,
        'extension_configs': extension_configs,
        'encoding': options.encoding,
        'output_format': options.output_format,
        'lazy_ol': options.lazy_ol
    }

    return opts, options.verbose


```

这段代码是一个Python函数，名为“run”，它执行以下操作：

1. 定义了一个名为“run”的函数，函数内部声明了一个没有参数的函数实参“()”，表示函数的实际参数为自己的空括号“()”。
2. 在函数内部定义了一个名为“options”的变量，该变量存储了从命令行中获得的选项，如果没有选项则被赋值为“None”。
3. 如果“options”的值为“None”，则函数内部的逻辑将无法继续执行，因为缺少了必要的参数。
4. 定义了一个名为“logging_level”的变量，用于存储日志输出级的设置，该变量被设置为“logging.ERROR”，表示将所有错误输出传递给日志库。
5. 创建了一个名为“console_handler”的日志输出级，该输出级将所有日志输出传递给一个名为“console_handler”的实参，该实参是一个Python内置的“logging.StreamHandler”实例。
6. 将“console_handler”添加到“logger”的日志输出级中，设置该输出级的日志级别为刚刚设置好的“logging.ERROR”。
7. 如果日志输出级别“logging_level”小于或等于“warning”，则执行以下操作：
  1. 创建一个名为“warnings”的日志输出级，用于存储警告类输出，该输出级设置为刚刚设置好的“logging.WARNING”。
  2. 创建一个名为“warn_logger”的日志输出级，用于存储警告类输出，该输出级设置为刚刚设置好的“logging.WARNING”。
  3. 将“console_handler”添加到“warn_logger”的日志输出级中，设置该输出级的日志级别为刚刚设置好的“logging.WARNING”。
8. 调用“markdown.markdownFromFile”函数，该函数接收一个名为“options”的参数，用于指定从文件中读取的Markdown内容。


```py
def run():  # pragma: no cover
    """Run Markdown from the command line."""

    # Parse options and adjust logging level if necessary
    options, logging_level = parse_options()
    if not options:
        sys.exit(2)
    logger.setLevel(logging_level)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    if logging_level <= WARNING:
        # Ensure deprecation warnings get displayed
        warnings.filterwarnings('default')
        logging.captureWarnings(True)
        warn_logger = logging.getLogger('py.warnings')
        warn_logger.addHandler(console_handler)

    # Run
    markdown.markdownFromFile(**options)


```

这段代码是一个if语句，判断当前脚本是否作为命令行程序运行。如果脚本作为命令行程序运行，那么它会执行run()函数。

run()函数的功能是启动一个命令行工具，用来运行这个脚本。这个工具会被添加到命令行参数中，允许用户使用“python [options] [args]”命令行选项来运行脚本。选项和args参数可以在命令行中被传递给脚本。


```py
if __name__ == '__main__':  # pragma: no cover
    # Support running module as a command line command.
    #     python -m markdown [options] [args]
    run()

```

# `/markdown/markdown/__meta__.py`

该代码是一个Python实现John Gruber所写的Markdown（Markdown是一种轻量级的标记语言，可以轻松地将纯文本转换为HTML格式）的代码。

该代码定义了一个名为`Markdown`的类，该类提供了一些方法来实现Markdown的基本功能。通过使用这些方法，可以轻松地将Markdown编码为HTML并输出。

该代码还定义了一个名为`A Python implementation of John Gruber's Markdown`的类，该类提供了一个`python-markdown`包的入口点。通过使用这个包，用户可以通过Python调用Markdown的方法来生成Markdown代码。

该代码还引用了Python标准库中的`


```py
# Python Markdown

# A Python implementation of John Gruber's Markdown.

# Documentation: https://python-markdown.github.io/
# GitHub: https://github.com/Python-Markdown/markdown/
# PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
```

这段代码是一个 Python 程序，定义了一个名为 `__version_info__` 的类。这个类的作用是提供关于程序版本的信息，例如程序的版本号、版本名称和版本号。

具体来说，这个程序的版本号是 `1.2.0`，版本名称是 `dev`，版本号是 `alpha` 和 `beta`。此外，还有两个与版本号有关的预发布版本，分别是 `rc` 和 `final`。

这个程序还定义了一个名为 `__future__` 的模块，包含了一些未来版本的说明。例如，这个模块表明 `from __future__ import annotations`，这意味着在使用这个模块时，需要使用 `@annotations` 这样的声明。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

# __version_info__ format:
#     (major, minor, patch, dev/alpha/beta/rc/final, #)
#     (1, 1, 2, 'dev', 0) => "1.1.2.dev0"
#     (1, 1, 2, 'alpha', 1) => "1.1.2a1"
#     (1, 2, 0, 'beta', 2) => "1.2b2"
#     (1, 2, 0, 'rc', 4) => "1.2rc4"
#     (1, 2, 0, 'final', 0) => "1.2"

from __future__ import annotations


```

这段代码定义了一个名为 `__version_info__` 的内置常量，其值为 `(3, 5, 1, 'final', 0)`。

该常量的含义是：

- `version_info` 是一个元组，包含五个元素。
- `len(version_info)` 的值为 5。
- `version_info[3]` 的值为 `'final'`。
- `version_info[2]` 的值为 `0`。

该函数 `_get_version` 使用 `version_info` 来生成一个符合 `PEP 440` 规范的版本号。它的实现方式如下：

- 如果 `version_info` 中只有一项是 `final`，那么直接返回 `3.5.1`。
- 如果 `version_info[3]` 是 `'dev'`，则将 `'alpha'`、`'beta'` 或 `'rc'` 和一个数字 `version_info[4]` 组合成一个字符串，加入版本号中。
- 如果 `version_info[3]` 不是 `'final'`，则使用一个映射 `{'alpha': 'a', 'beta': 'b', 'rc': 'rc'}` 来将 `version_info[3]` 转换成一个字母，然后将转换后的字母和 `version_info[4]` 加入版本号中。

最后，函数返回生成的版本号。


```py
__version_info__ = (3, 5, 1, 'final', 0)


def _get_version(version_info):
    " Returns a PEP 440-compliant version number from `version_info`. "
    assert len(version_info) == 5
    assert version_info[3] in ('dev', 'alpha', 'beta', 'rc', 'final')

    parts = 2 if version_info[2] == 0 else 3
    v = '.'.join(map(str, version_info[:parts]))

    if version_info[3] == 'dev':
        v += '.dev' + str(version_info[4])
    elif version_info[3] != 'final':
        mapping = {'alpha': 'a', 'beta': 'b', 'rc': 'rc'}
        v += mapping[version_info[3]] + str(version_info[4])

    return v


```

这段代码是在 Python 中执行的，它尝试从 `__version_info__` 对象中获取版本信息，然后将获取到的版本信息存储在 `__version__` 变量中。

`__version_info__` 是一个元组，其中包含 Python 版本号的相关信息。例如，对于版本号为 3.9.7，`__version_info__` 可能是一个包含以下元素的元组：


3, 9, 7


在这个元组中，第一个元素是版本号，第二个元素是版本号后面的字符串，表示是支持该版本号的用户。

在这个代码中，`_get_version` 函数是一个自定义的函数，它接受一个参数 `__version_info__`，返回版本号的相关信息。这个函数可能是从 Python 标准库或其他第三方库中获取的。

最后，将获取到的版本信息存储在 `__version__` 变量中，以便在程序的其他部分使用。


```py
__version__ = _get_version(__version_info__)

```

# `/markdown/markdown/extensions/abbr.py`

该代码是一个 Python-Markdown 的自定义 abbreviation 扩展。它通过将 Python 中的大写字母转换为小写字母来简化长单词的表示。例如，在 Python 中，`print()` 函数通常用于输出大量信息，但通过该自定义扩展，可以使用 `p` 代替 `print()`，从而简化输出。

该自定义扩展的实现基于两个贡献者：Waylan Limberg 和 Seemant Kulleen。该自定义扩展最初是在 2008 年由 The Python Markdown Project 发布的，并在随后的几年得到维护和升级。

该自定义扩展的原始代码采用 BSD 授权，这意味着它允许自由地使用、修改和重新分发，前提是在代码中包含原始版权声明和许可证。


```py
# Abbreviation Extension for Python-Markdown
# ==========================================

# This extension adds abbreviation handling to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/abbreviations
# for documentation.

# Original code Copyright 2007-2008 [Waylan Limberg](http://achinghead.com/)
# and [Seemant Kulleen](http://www.kulleen.org/)

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

```

这段代码是一个Python脚本，是对Python-Markdown进行扩展的其中一个功能。具体来说，它定义了一个名为"abbreviation handling"的新扩展，用于处理Python-Markdown中的缩写词。

这个新扩展将使得Python-Markdown能够识别出Markdown中的文本内容中的所有缩写词，并将它们转换为完整的单词。通过使用这个扩展，用户就可以轻松地将一些常用的单词表达出来，例如"争霸"、"裙下之臣"等等。

这个新扩展使用了Python-Markdown的扩展机制，通过定义一个类Extension来实现。在Extension的定义中，使用了Python中一个特殊的装饰器**，这个装饰器用于定义扩展的功能、名称、文档等信息。

另外，这个新扩展还使用了BlockProcessor、InlineProcessor和AtomicString类，这些类都用于处理Markdown文档中的内容。


```py
"""
This extension adds abbreviation handling to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/abbreviations)
for details.
"""

from __future__ import annotations

from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..util import AtomicString
import re
import xml.etree.ElementTree as etree


```

This is an implementation of the `md_pattern.AbbrInlineProcessor` class in the `mark.jsonl` module. It defines a pattern to find and replace all abbreviations (`abbr`) in the text with their expanded form.

The regular expression pattern for an abbreviation is given by:
css
(?P<abbr>[H][T][M][L])

This pattern matches any lowercase letter (`H`, `T`, `M`, or `L`) that is followed by a space and then an optional parameter, which is either a hyphen (`-`) or an underscore (`_`). The space is considered a literal character and the parameter is treated as a group.

The `AbbrInlineProcessor` class is responsible for finding all abbreviations in the input text and replacing them with their expanded form. It does this by searching for abbreviations in the text (using the regular expression provided), and then replacing them in the output text. If a pattern is found, it is added to the `pattern_rules` dictionary, which is used to generate the actual replacement pattern at runtime.

The `test` and `run` methods of the `AbbrInlineProcessor` class are used to test and apply the pattern, respectively. The `test` method takes an input tree element and a list of blocks and returns `True` if the pattern is found and applied to the input text. The `run` method finds all abbreviations in the input text and applies the pattern to each block, returning `True` if any are found and replaced.


```py
class AbbrExtension(Extension):
    """ Abbreviation Extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Insert `AbbrPreprocessor` before `ReferencePreprocessor`. """
        md.parser.blockprocessors.register(AbbrPreprocessor(md.parser), 'abbr', 16)


class AbbrPreprocessor(BlockProcessor):
    """ Abbreviation Preprocessor - parse text for abbr references. """

    RE = re.compile(r'^[*]\[(?P<abbr>[^\]]*)\][ ]?:[ ]*\n?[ ]*(?P<title>.*)$', re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        """
        Find and remove all Abbreviation references from the text.
        Each reference is set as a new `AbbrPattern` in the markdown instance.

        """
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            abbr = m.group('abbr').strip()
            title = m.group('title').strip()
            self.parser.md.inlinePatterns.register(
                AbbrInlineProcessor(self._generate_pattern(abbr), title), 'abbr-%s' % abbr, 2
            )
            if block[m.end():].strip():
                # Add any content after match back to blocks as separate block
                blocks.insert(0, block[m.end():].lstrip('\n'))
            if block[:m.start()].strip():
                # Add any content before match back to blocks as separate block
                blocks.insert(0, block[:m.start()].rstrip('\n'))
            return True
        # No match. Restore block.
        blocks.insert(0, block)
        return False

    def _generate_pattern(self, text: str) -> str:
        """
        Given a string, returns an regex pattern to match that string.

        'HTML' -> r'(?P<abbr>[H][T][M][L])'

        Note: we force each char as a literal match (in brackets) as we don't
        know what they will be beforehand.

        """
        chars = list(text)
        for i in range(len(chars)):
            chars[i] = r'[%s]' % chars[i]
        return r'(?P<abbr>\b%s\b)' % (r''.join(chars))


```

这段代码定义了一个名为 "AbbrInlineProcessor" 的类，该类实现了 StackOverflow 问题 "不要在代码中输出扩展函数名称" 的要求。该类包含了一个内部处理函数 "handleMatch"，用于处理 Abbreviation inline pattern。

该函数的参数包括两个字符串参数 "pattern" 和 "title"，分别表示要匹配的短语模式和显示的文本标题。函数内部首先调用父类的 "handleMatch" 函数，然后创建一个 "abbr" 元素，设置其文本为匹配到的短语，并设置标题为 "title"。最后，返回创建的 "abbr" 元素对象，以及匹配到的短语开始和结束的位置索引。

该类还定义了一个名为 "makeExtension" 的函数，该函数接受一个字典作为参数，然后使用该参数中的键值对创建一个新的 "AbbrExtension" 类实例，从而将 "AbbrInlineProcessor" 和 "AbbrExtension" 类组合在一起，实现一个简洁的短语处理函数。


```py
class AbbrInlineProcessor(InlineProcessor):
    """ Abbreviation inline pattern. """

    def __init__(self, pattern: str, title: str):
        super().__init__(pattern)
        self.title = title

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        abbr = etree.Element('abbr')
        abbr.text = AtomicString(m.group('abbr'))
        abbr.set('title', self.title)
        return abbr, m.start(0), m.end(0)


def makeExtension(**kwargs):  # pragma: no cover
    return AbbrExtension(**kwargs)

```

# `/markdown/markdown/extensions/admonition.py`

这段代码是一个Python-Markdown的admonition扩展，它为Python添加了类似于rST格式的admonitions。这个扩展是基于[rST](https://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions)规范进行的，它定义了一些rST格式的admonitions，可以用来在Python-Markdown中使用。

具体来说，这段代码实现了一个如下功能：

1. 定义了一个名为“# 引言”的admonition。
2. 在引言中提到了rST格式的admonitions，这是受到了rST规范中[specific-admonitions](https://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions)部分内容的启发。
3. 在代码中定义了一些rST格式的admonitions，包括标题、内容、链接等。
4. 在引言的末尾，展示了原始代码的作者信息，以及定义这些admonitions的日期。
5. 在文件末尾，使用了BSD授权协议发布了这个扩展。

总之，这段代码的作用是扩展Python-Markdown，使得用户可以使用rST格式的admonitions来格式化他们的Markdown文档。


```py
# Admonition extension for Python-Markdown
# ========================================

# Adds rST-style admonitions. Inspired by [rST][] feature with the same name.

# [rST]: http://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions

# See https://Python-Markdown.github.io/extensions/admonition
# for documentation.

# Original code Copyright [Tiago Serafim](https://www.tiagoserafim.com/).

# All changes Copyright The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)


```

该代码是一个Python文件中的类定义，属于rST（r人有兴趣的段落）格式的配置文件。它定义了一个名为“Adds rST-style admonitions”的新指令，是相同名称的rST格式的指令的扩展。

该指令的内容是在文档中对特定类别的说明。它通过使用http://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions 来获取rST文档的说明。它还链接了[rST文档](https://Python-Markdown.github.io/extensions/admonition)以获取更多信息。

该指令还定义了一个内部类BlockProcessor，它继承自extensions.BlockProcessor类。该内部类似乎负责处理rST格式的配置文件中的内容，但并没有在该指令中做详细说明。


```py
"""
Adds rST-style admonitions. Inspired by [rST][] feature with the same name.

[rST]: http://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions

See the [documentation](https://Python-Markdown.github.io/extensions/admonition)
for details.
"""

from __future__ import annotations

from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
```

这段代码定义了一个名为 `AdmonitionExtension` 的类，用于在 Python-Markdown 中添加一个名为 "admonition" 的扩展，可以在需要时将 Admonition 段落添加到 Markdown 对象中。

具体来说，代码中首先定义了一个 pragma 注释 `from typing import TYPE_CHECKING`，用于指示代码可以不被用于类型检查。然后，代码导入自 `markdown` 包中的 `blockparser` 类，以便在 Admonition 扩展中使用。

接着，定义了 `AdmonitionExtension` 类，其中包含一个名为 `extendMarkdown` 的方法，用于将 Admonition 扩展添加到 Markdown 对象中。在此方法中，首先注册了扩展的名称(即 `self`)、扩展的类名(即 `AdmonitionProcessor`)、以及扩展的 ID(即 105)。

接着，定义了一个名为 `registerExtension` 的方法，用于注册 Admonition 扩展的处理器。在此方法中，注册了名为 `AdmonitionProcessor` 的处理器，并指定了处理器在 Markdown 解析器中的 ID，该 ID 为 105。

最后，扩展被注册到 Markdown 对象后，会在 `md.parser.blockprocessors.register` 方法返回的列表中始终包含 `AdmonitionProcessor` 处理器，以确保在后续的 Markdown 解析中，能够正确地解析和处理扩展。


```py
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from markdown import blockparser


class AdmonitionExtension(Extension):
    """ Admonition extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Add Admonition to Markdown instance. """
        md.registerExtension(self)

        md.parser.blockprocessors.register(AdmonitionProcessor(md.parser), 'admonition', 105)


```

This is a JavaScript-like language with a class and method name of the same, `matchbox_ parser` which parses an HTML block. It takes an HTML block as input and returns a list of blocks where the indentation is less than a certain threshold, or where the block has no text.

The `matchbox_ parser` uses a regular expression to match lines that contain indentation within the block, and returns a list of these blocks. The blocks are processed in order, and any indentation between blocks is taken into account.

The `matchbox_ parser` also supports nested blocks, where the blocks are indented within other blocks. This is useful for handling inheritance or multiple imports.

If a block has the maximum indentation limit, it will be included in the list of blocks.

It is worth noting that this parser is a simple implementation, it does not handle some edge cases and it can be easily improved.


```py
class AdmonitionProcessor(BlockProcessor):

    CLASSNAME = 'admonition'
    CLASSNAME_TITLE = 'admonition-title'
    RE = re.compile(r'(?:^|\n)!!! ?([\w\-]+(?: +[\w\-]+)*)(?: +"(.*?)")? *(?:\n|$)')
    RE_SPACES = re.compile('  +')

    def __init__(self, parser: blockparser.BlockParser):
        """Initialization."""

        super().__init__(parser)

        self.current_sibling: etree.Element | None = None
        self.content_indention = 0

    def parse_content(self, parent: etree.Element, block: str) -> tuple[etree.Element | None, str, str]:
        """Get sibling admonition.

        Retrieve the appropriate sibling element. This can get tricky when
        dealing with lists.

        """

        old_block = block
        the_rest = ''

        # We already acquired the block via test
        if self.current_sibling is not None:
            sibling = self.current_sibling
            block, the_rest = self.detab(block, self.content_indent)
            self.current_sibling = None
            self.content_indent = 0
            return sibling, block, the_rest

        sibling = self.lastChild(parent)

        if sibling is None or sibling.tag != 'div' or sibling.get('class', '').find(self.CLASSNAME) == -1:
            sibling = None
        else:
            # If the last child is a list and the content is sufficiently indented
            # to be under it, then the content's sibling is in the list.
            last_child = self.lastChild(sibling)
            indent = 0
            while last_child is not None:
                if (
                    sibling is not None and block.startswith(' ' * self.tab_length * 2) and
                    last_child is not None and last_child.tag in ('ul', 'ol', 'dl')
                ):

                    # The expectation is that we'll find an `<li>` or `<dt>`.
                    # We should get its last child as well.
                    sibling = self.lastChild(last_child)
                    last_child = self.lastChild(sibling) if sibling is not None else None

                    # Context has been lost at this point, so we must adjust the
                    # text's indentation level so it will be evaluated correctly
                    # under the list.
                    block = block[self.tab_length:]
                    indent += self.tab_length
                else:
                    last_child = None

            if not block.startswith(' ' * self.tab_length):
                sibling = None

            if sibling is not None:
                indent += self.tab_length
                block, the_rest = self.detab(old_block, indent)
                self.current_sibling = sibling
                self.content_indent = indent

        return sibling, block, the_rest

    def test(self, parent: etree.Element, block: str) -> bool:

        if self.RE.search(block):
            return True
        else:
            return self.parse_content(parent, block)[0] is not None

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        m = self.RE.search(block)

        if m:
            if m.start() > 0:
                self.parser.parseBlocks(parent, [block[:m.start()]])
            block = block[m.end():]  # removes the first line
            block, theRest = self.detab(block)
        else:
            sibling, block, theRest = self.parse_content(parent, block)

        if m:
            klass, title = self.get_class_and_title(m)
            div = etree.SubElement(parent, 'div')
            div.set('class', '{} {}'.format(self.CLASSNAME, klass))
            if title:
                p = etree.SubElement(div, 'p')
                p.text = title
                p.set('class', self.CLASSNAME_TITLE)
        else:
            # Sibling is a list item, but we need to wrap it's content should be wrapped in <p>
            if sibling.tag in ('li', 'dd') and sibling.text:
                text = sibling.text
                sibling.text = ''
                p = etree.SubElement(sibling, 'p')
                p.text = text

            div = sibling

        self.parser.parseChunk(div, block)

        if theRest:
            # This block contained unindented line(s) after the first indented
            # line. Insert these lines as the first block of the master blocks
            # list for future processing.
            blocks.insert(0, theRest)

    def get_class_and_title(self, match: re.Match[str]) -> tuple[str, str | None]:
        klass, title = match.group(1).lower(), match.group(2)
        klass = self.RE_SPACES.sub(' ', klass)
        if title is None:
            # no title was provided, use the capitalized class name as title
            # e.g.: `!!! note` will render
            # `<p class="admonition-title">Note</p>`
            title = klass.split(' ', 1)[0].capitalize()
        elif title == '':
            # an explicit blank title should not be rendered
            # e.g.: `!!! warning ""` will *not* render `p` with a title
            title = None
        return klass, title


```

这段代码定义了一个名为 `makeExtension` 的函数，它接受一个或多个参数 `**kwargs`，并将它们传递给名为 `AdmonitionExtension` 的函数，最终返回该函数。

具体来说，这段代码使用了 Python 的 pragma 指导原则（PEP 8），其中 `**kwargs` 是该指导原则的一种语法，用于表示在函数定义中参数可以是一个或多个键值对，而不是单个键值对。

函数体中，`AdmonitionExtension` 是一个名为 `AdmonitionExtension` 的函数，该函数接受了 `**kwargs` 的传递，并返回了一个 `AdmonitionExtension` 对象。这里，`**kwargs` 中的键和值都被传递给了 `AdmonitionExtension`，因此 `AdmonitionExtension` 函数可以接收一个或多个参数。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return AdmonitionExtension(**kwargs)

```

# `/markdown/markdown/extensions/attr_list.py`

这段代码是一个Python类的扩展，名为Attribute List Extension。它定义了一个新的属性列表语法，类似于Markdown中的[[冽西]]。这个语法通过在属性名称和值之间添加中括号来表示，例如：

[attr1]
   属性1的值
[attr2]
   属性2的值

它的作用是给Python的AttributeList扩展添加一个新的属性列表语法，使得在Markdown中更方便地引用属性列表。


```py
# Attribute List Extension for Python-Markdown
# ============================================

# Adds attribute list syntax. Inspired by
# [Maruku](http://maruku.rubyforge.org/proposal.html#attribute_lists)'s
# feature of the same name.

# See https://Python-Markdown.github.io/extensions/attr_list
# for documentation.

# Original code Copyright 2011 [Waylan Limberg](http://achinghead.com/).

# All changes Copyright 2011-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

```

这段代码定义了一个新的 Python 类 `AttributeListProcessor`，并添加了属性列表语法。它受到 Maruku 的提议的启发，也在 不能不说非常重大中进行了说明。

具体来说，这段代码实现了一个如下功能：

1. 定义了一个名为 `AttributeListProcessor` 的类。
2. 继承自 `Treeprocessor`，实现了 `process_ According to the documentation <https://Python-Markdown.github.io/extensions/attr_list>  
  https://github.com/markdown/docs/reference/https://Python-Markdown.github.io/extensions/attr_list>
3. 实现了 `process_constant` 方法，用于处理属性列表中的常量值。
4. 实现了 `process_甜甜圈` 方法，用于处理属性列表中的甜甜圈表达式。
5. 实现了 `process_header` 方法，用于处理属性列表中的Header表达式。
6. 实现了 `process_filename` 方法，用于处理属性列表中的文件名。
7. 实现了 `process_input` 方法，用于处理属性列表中的输入。
8. 实现了 `process_output` 方法，用于处理属性列表中的输出。
9. 实现了 `run` 方法，用于处理整个属性列表。

由于 `AttributeListProcessor` 类中包含了许多方法，因此它的作用是使代码更加易于理解和维护。它可以帮助用户生成格式良好的文档、表格和模板，从而使得文档更加规范、易于阅读和理解。


```py
"""
 Adds attribute list syntax. Inspired by
[Maruku](http://maruku.rubyforge.org/proposal.html#attribute_lists)'s
feature of the same name.

See the [documentation](https://Python-Markdown.github.io/extensions/attr_list)
for details.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from . import Extension
from ..treeprocessors import Treeprocessor
import re

```

这段代码定义了几个函数来处理不同类型的引用。以下是每个函数的作用：

1. `_handle_double_quote`：用于解析两个双引号之间的键值对。它将输入字符串`s`和`t`分成两个部分，并将它们存储为两个变量`k`和`v`。然后，它返回这两个变量。

2. `_handle_single_quote`：用于解析一个单引号之间的键值对。它与`_handle_double_quote`类似，只是返回一个变量`k`和一个变量`v`，而不是两个。

3. `_handle_key_value`：用于解析一个键值对。它将输入字符串`s`和`t`分成两个部分，并将它们存储为两个变量`k`和`v`。然后，它返回这两个变量。

该代码可能用于为一个输入字符串列表定义不同的解析函数，根据输入字符串的类型进行类型检查。例如，如果输入字符串是一个列表，该代码将能够解析两个引号之间的键值对。


```py
if TYPE_CHECKING:  # pragma: no cover
    from xml.etree.ElementTree import Element


def _handle_double_quote(s, t):
    k, v = t.split('=', 1)
    return k, v.strip('"')


def _handle_single_quote(s, t):
    k, v = t.split('=', 1)
    return k, v.strip("'")


def _handle_key_value(s, t):
    return t.split('=', 1)


```

这段代码是一个正则表达式的字符串处理函数，它的输入参数是一个字符串`s`和一个字符串`t`。函数的作用是在`s`中查找给定的字段，如果找到了相应的字段，则返回该字段，否则返回原字符串。

具体地，函数实现以下操作：

1. 如果`t`以点号`.`开头，则返回`t`中从点号后面的字符开始到结束的字符，即`t[1:]`。
2. 如果`t`以井号`#`开头，则返回`t`中从井号后面的字符开始到结束的字符，即`t[1:]`。
3. 如果`t`包含给定的字段，则返回该字段，即`t`本身。
4. 如果以上步骤都没有找到相应的字段，则返回空字符串`''`。


```py
def _handle_word(s, t):
    if t.startswith('.'):
        return '.', t[1:]
    if t.startswith('#'):
        return 'id', t[1:]
    return t, t


_scanner = re.Scanner([
    (r'[^ =]+=".*?"', _handle_double_quote),
    (r"[^ =]+='.*?'", _handle_single_quote),
    (r'[^ =]+=[^ =]+', _handle_key_value),
    (r'[^ =]+', _handle_word),
    (r' ', None)
])


```

This looks like a CSS parser that allows you to parse CSS written in the相互-exclusive梁文与教科书中文间的格式. It appears to support both HTML and CSS input, and it can parse CSS and extract relevant information such as class names, authors, and licenses. Additionally, it seems to support the "attrs" syntax for attaching custom attributes to elements, which can be useful for dynamic loading of data.


```py
def get_attrs(str: str) -> list[tuple[str, str]]:
    """ Parse attribute list and return a list of attribute tuples. """
    return _scanner.scan(str)[0]


def isheader(elem: Element) -> bool:
    return elem.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']


class AttrListTreeprocessor(Treeprocessor):

    BASE_RE = r'\{\:?[ ]*([^\}\n ][^\}\n]*)[ ]*\}'
    HEADER_RE = re.compile(r'[ ]+{}[ ]*$'.format(BASE_RE))
    BLOCK_RE = re.compile(r'\n[ ]*{}[ ]*$'.format(BASE_RE))
    INLINE_RE = re.compile(r'^{}'.format(BASE_RE))
    NAME_RE = re.compile(r'[^A-Z_a-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02ff'
                         r'\u0370-\u037d\u037f-\u1fff\u200c-\u200d'
                         r'\u2070-\u218f\u2c00-\u2fef\u3001-\ud7ff'
                         r'\uf900-\ufdcf\ufdf0-\ufffd'
                         r'\:\-\.0-9\u00b7\u0300-\u036f\u203f-\u2040]+')

    def run(self, doc: Element) -> None:
        for elem in doc.iter():
            if self.md.is_block_level(elem.tag):
                # Block level: check for `attrs` on last line of text
                RE = self.BLOCK_RE
                if isheader(elem) or elem.tag in ['dt', 'td', 'th']:
                    # header, def-term, or table cell: check for attributes at end of element
                    RE = self.HEADER_RE
                if len(elem) and elem.tag == 'li':
                    # special case list items. children may include a `ul` or `ol`.
                    pos = None
                    # find the `ul` or `ol` position
                    for i, child in enumerate(elem):
                        if child.tag in ['ul', 'ol']:
                            pos = i
                            break
                    if pos is None and elem[-1].tail:
                        # use tail of last child. no `ul` or `ol`.
                        m = RE.search(elem[-1].tail)
                        if m:
                            self.assign_attrs(elem, m.group(1))
                            elem[-1].tail = elem[-1].tail[:m.start()]
                    elif pos is not None and pos > 0 and elem[pos-1].tail:
                        # use tail of last child before `ul` or `ol`
                        m = RE.search(elem[pos-1].tail)
                        if m:
                            self.assign_attrs(elem, m.group(1))
                            elem[pos-1].tail = elem[pos-1].tail[:m.start()]
                    elif elem.text:
                        # use text. `ul` is first child.
                        m = RE.search(elem.text)
                        if m:
                            self.assign_attrs(elem, m.group(1))
                            elem.text = elem.text[:m.start()]
                elif len(elem) and elem[-1].tail:
                    # has children. Get from tail of last child
                    m = RE.search(elem[-1].tail)
                    if m:
                        self.assign_attrs(elem, m.group(1))
                        elem[-1].tail = elem[-1].tail[:m.start()]
                        if isheader(elem):
                            # clean up trailing #s
                            elem[-1].tail = elem[-1].tail.rstrip('#').rstrip()
                elif elem.text:
                    # no children. Get from text.
                    m = RE.search(elem.text)
                    if m:
                        self.assign_attrs(elem, m.group(1))
                        elem.text = elem.text[:m.start()]
                        if isheader(elem):
                            # clean up trailing #s
                            elem.text = elem.text.rstrip('#').rstrip()
            else:
                # inline: check for `attrs` at start of tail
                if elem.tail:
                    m = self.INLINE_RE.match(elem.tail)
                    if m:
                        self.assign_attrs(elem, m.group(1))
                        elem.tail = elem.tail[m.end():]

    def assign_attrs(self, elem: Element, attrs: str) -> None:
        """ Assign `attrs` to element. """
        for k, v in get_attrs(attrs):
            if k == '.':
                # add to class
                cls = elem.get('class')
                if cls:
                    elem.set('class', '{} {}'.format(cls, v))
                else:
                    elem.set('class', v)
            else:
                # assign attribute `k` with `v`
                elem.set(self.sanitize_name(k), v)

    def sanitize_name(self, name: str) -> str:
        """
        Sanitize name as 'an XML Name, minus the ":"'.
        See https://www.w3.org/TR/REC-xml-names/#NT-NCName
        """
        return self.NAME_RE.sub('_', name)


```

这段代码定义了一个名为 `AttrListExtension` 的类，用于在 Python-Markdown 中扩展 `AttrListTreeprocessor`。AttrListTreeprocessor 是一个类，它在运行时创建一个 AttrList，该 AttrList 可以用来在 Python-Markdown 中插入自定义标记和链接。

`AttrListExtension` 类继承自 `Extension` 类，后者是一个 Python 标准库中的类，用于定义扩展。通过 `registerExtension` 方法，可以将 `AttrListTreeprocessor` 注册到 Markdown 的 treeprocessors 属性中，这样它就可以在运行时创建并应用 AttrList。

另外，`makeExtension` 函数用于创建扩展，它接受一个字典 `kwargs`，其中包含用于扩展的选项。选项可以包括 `**kwargs`，这样就可以将 `AttrListTreeprocessor` 和 `AttrList` 的参数传递给 `registerExtension` 方法。


```py
class AttrListExtension(Extension):
    """ Attribute List extension for Python-Markdown """
    def extendMarkdown(self, md):
        md.treeprocessors.register(AttrListTreeprocessor(md), 'attr_list', 8)
        md.registerExtension(self)


def makeExtension(**kwargs):  # pragma: no cover
    return AttrListExtension(**kwargs)

```