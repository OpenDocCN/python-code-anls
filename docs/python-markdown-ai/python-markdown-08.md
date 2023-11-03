# PythonMarkdown源码解析 8

# `/markdown/markdown/extensions/codehilite.py`

这段代码是一个Python-Markdown的扩展，它可以在运行时对标准Python-Markdown代码块进行代码和 syntax highlighting（高亮显示）。它通过将python
# code_hilite_ Extension for Python-Markdown
和python
#  Adds code/syntax highlighting to standard Python-Markdown code blocks.
这两行注释来告诉您如何使用此扩展。

具体来说，这段代码会检查当前代码文件是否使用了Python-Markdown语法，如果是，它会在代码块前添加以下内容：
python
# code_hilite_ Extension for Python-Markdown
# https://Python-Markdown.github.io/extensions/code_hilite

然后，它会检查当前代码块是否包含`code_hilite`标签，如果是，它会将其展开为一系列交互式高亮代码块。
python
# code_hilite_ Extension for Python-Markdown
# https://Python-Markdown.github.io/extensions/code_hilite
# for documentation.

此外，它还要求作者提供原始代码的链接以进行documentation。
python
# code_hilite_ Extension for Python-Markdown
# https://Python-Markdown.github.io/extensions/code_hilite
# for documentation.

另外，它要求源代码使用BSD授权协议。
python
# code_hilite_ Extension for Python-Markdown
# https://opensource.org/licenses/bsd-license.php
# License: [BSD](https://opensource.org/licenses/bsd-license.php)

最后，所有改动需提交至原始代码 repository。


```py
# CodeHilite Extension for Python-Markdown
# ========================================

# Adds code/syntax highlighting to standard Python-Markdown code blocks.

# See https://Python-Markdown.github.io/extensions/code_hilite
# for documentation.

# Original code Copyright 2006-2008 [Waylan Limberg](http://achinghead.com/).

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
```

这段代码的作用是向标准 Python Markdown 代码块中添加代码高亮。它由一个名为 Extension 的类负责，该类是一个 Python 扩展，可以使用 `@extensions.py` 签名来导入。Treeprocessor 是另一个类，它可能负责将代码中的文本转换为树形结构。util.parseBoolValue 是一个函数，可能是用于解析文本中的布尔值。最后，这段代码没有定义任何函数或类，也没有从外部导入任何模块或库。


```py
Adds code/syntax highlighting to standard Python-Markdown code blocks.

See the [documentation](https://Python-Markdown.github.io/extensions/code_hilite)
for details.
"""

from __future__ import annotations

from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue
from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:  # pragma: no cover
    import xml.etree.ElementTree as etree

```

这段代码是一个异常处理程序，用于在程序运行时遇到ImportError时进行回退。

try语句的后面有一个if语句，用于检查是否发生了ImportError。如果没有发生ImportError，则说明pygments模块已经被正确导入，可以继续执行后续代码。如果发生了ImportError，则说明pygments模块尚未被正确导入，程序将无法继续执行。

在if语句的后面是两个import语句，用于导入pygments模块中的highlight、get_lexer_by_name和get_formatter_by_name函数，以及一个ClassNotFound类。

try语句的后面还有一个except语句，用于处理ImportError。如果发生了ImportError，将回滚try语句，但不输出任何信息。

另外，还有一个parse_hl_lines函数，它接受一个表达式参数expr，并返回一个列表，列表中的每个元素都是数字，表示要强调的代码行的行号。函数内部首先检查expr是否为空，如果为空则返回一个空列表。接着，它尝试从expr中解析出要强调的代码行的行号，并将它们转换成整数。如果解析失败，则返回一个空列表。


```py
try:  # pragma: no cover
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import get_formatter_by_name
    from pygments.util import ClassNotFound
    pygments = True
except ImportError:  # pragma: no cover
    pygments = False


def parse_hl_lines(expr: str) -> list[int]:
    """Support our syntax for emphasizing certain lines of code.

    `expr` should be like '1 2' to emphasize lines 1 and 2 of a code block.
    Returns a list of integers, the line numbers to emphasize.
    """
    if not expr:
        return []

    try:
        return list(map(int, expr.split()))
    except ValueError:  # pragma: no cover
        return []


```

This appears to be a Python implementation of a line/column numbering and syntax highlighting reader. It appears to support a number of different syntax highlighting options, including the ability to specify a list of highlight lines using a shell-like syntax. The syntax itself is rather complex, but it appears to be implementing the core functionality of a line/column numbering and syntax highlighting reader.


```py
# ------------------ The Main CodeHilite Class ----------------------
class CodeHilite:
    """
    Determine language of source code, and pass it on to the Pygments highlighter.

    Usage:

    ```python
    code = CodeHilite(src=some_code, lang='python')
    html = code.hilite()
    ```py

    Arguments:
        src: Source string or any object with a `.readline` attribute.

    Keyword arguments:
        lang (str): String name of Pygments lexer to use for highlighting. Default: `None`.
        guess_lang (bool): Auto-detect which lexer to use.
            Ignored if `lang` is set to a valid value. Default: `True`.
        use_pygments (bool): Pass code to Pygments for code highlighting. If `False`, the code is
            instead wrapped for highlighting by a JavaScript library. Default: `True`.
        pygments_formatter (str): The name of a Pygments formatter or a formatter class used for
            highlighting the code blocks. Default: `html`.
        linenums (bool): An alias to Pygments `linenos` formatter option. Default: `None`.
        css_class (str): An alias to Pygments `cssclass` formatter option. Default: 'codehilite'.
        lang_prefix (str): Prefix prepended to the language. Default: "language-".

    Other Options:

    Any other options are accepted and passed on to the lexer and formatter. Therefore,
    valid options include any options which are accepted by the `html` formatter or
    whichever lexer the code's language uses. Note that most lexers do not have any
    options. However, a few have very useful options, such as PHP's `startinline` option.
    Any invalid options are ignored without error.

    * **Formatter options**: <https://pygments.org/docs/formatters/#HtmlFormatter>
    * **Lexer Options**: <https://pygments.org/docs/lexers/>

    Additionally, when Pygments is enabled, the code's language is passed to the
    formatter as an extra option `lang_str`, whose value being `{lang_prefix}{lang}`.
    This option has no effect to the Pygments' builtin formatters.

    Advanced Usage:

    ```python
    code = CodeHilite(
        src = some_code,
        lang = 'php',
        startinline = True,      # Lexer option. Snippet does not start with `<?php`.
        linenostart = 42,        # Formatter option. Snippet starts on line 42.
        hl_lines = [45, 49, 50], # Formatter option. Highlight lines 45, 49, and 50.
        linenos = 'inline'       # Formatter option. Avoid alignment problems.
    )
    html = code.hilite()
    ```py

    """

    def __init__(self, src: str, **options):
        self.src = src
        self.lang: str | None = options.pop('lang', None)
        self.guess_lang: bool = options.pop('guess_lang', True)
        self.use_pygments: bool = options.pop('use_pygments', True)
        self.lang_prefix: str = options.pop('lang_prefix', 'language-')
        self.pygments_formatter: str | Callable = options.pop('pygments_formatter', 'html')

        if 'linenos' not in options:
            options['linenos'] = options.pop('linenums', None)
        if 'cssclass' not in options:
            options['cssclass'] = options.pop('css_class', 'codehilite')
        if 'wrapcode' not in options:
            # Override Pygments default
            options['wrapcode'] = True
        # Disallow use of `full` option
        options['full'] = False

        self.options = options

    def hilite(self, shebang: bool = True) -> str:
        """
        Pass code to the [Pygments](https://pygments.org/) highlighter with
        optional line numbers. The output should then be styled with CSS to
        your liking. No styles are applied by default - only styling hooks
        (i.e.: `<span class="k">`).

        returns : A string of html.

        """

        self.src = self.src.strip('\n')

        if self.lang is None and shebang:
            self._parseHeader()

        if pygments and self.use_pygments:
            try:
                lexer = get_lexer_by_name(self.lang, **self.options)
            except ValueError:
                try:
                    if self.guess_lang:
                        lexer = guess_lexer(self.src, **self.options)
                    else:
                        lexer = get_lexer_by_name('text', **self.options)
                except ValueError:  # pragma: no cover
                    lexer = get_lexer_by_name('text', **self.options)
            if not self.lang:
                # Use the guessed lexer's language instead
                self.lang = lexer.aliases[0]
            lang_str = f'{self.lang_prefix}{self.lang}'
            if isinstance(self.pygments_formatter, str):
                try:
                    formatter = get_formatter_by_name(self.pygments_formatter, **self.options)
                except ClassNotFound:
                    formatter = get_formatter_by_name('html', **self.options)
            else:
                formatter = self.pygments_formatter(lang_str=lang_str, **self.options)
            return highlight(self.src, lexer, formatter)
        else:
            # just escape and build markup usable by JavaScript highlighting libraries
            txt = self.src.replace('&', '&amp;')
            txt = txt.replace('<', '&lt;')
            txt = txt.replace('>', '&gt;')
            txt = txt.replace('"', '&quot;')
            classes = []
            if self.lang:
                classes.append('{}{}'.format(self.lang_prefix, self.lang))
            if self.options['linenos']:
                classes.append('linenums')
            class_str = ''
            if classes:
                class_str = ' class="{}"'.format(' '.join(classes))
            return '<pre class="{}"><code{}>{}\n</code></pre>\n'.format(
                self.options['cssclass'],
                class_str,
                txt
            )

    def _parseHeader(self) -> None:
        """
        Determines language of a code block from shebang line and whether the
        said line should be removed or left in place. If the shebang line
        contains a path (even a single /) then it is assumed to be a real
        shebang line and left alone. However, if no path is given
        (e.i.: `#!python` or `:::python`) then it is assumed to be a mock shebang
        for language identification of a code fragment and removed from the
        code block prior to processing for code highlighting. When a mock
        shebang (e.i: `#!python`) is found, line numbering is turned on. When
        colons are found in place of a shebang (e.i.: `:::python`), line
        numbering is left in the current state - off by default.

        Also parses optional list of highlight lines, like:

            :::python hl_lines="1 3"
        """

        import re

        # split text into lines
        lines = self.src.split("\n")
        # pull first line to examine
        fl = lines.pop(0)

        c = re.compile(r'''
            (?:(?:^::+)|(?P<shebang>^[#]!)) # Shebang or 2 or more colons
            (?P<path>(?:/\w+)*[/ ])?        # Zero or 1 path
            (?P<lang>[\w#.+-]*)             # The language
            \s*                             # Arbitrary whitespace
            # Optional highlight lines, single- or double-quote-delimited
            (hl_lines=(?P<quot>"|')(?P<hl_lines>.*?)(?P=quot))?
            ''',  re.VERBOSE)
        # search first line for shebang
        m = c.search(fl)
        if m:
            # we have a match
            try:
                self.lang = m.group('lang').lower()
            except IndexError:  # pragma: no cover
                self.lang = None
            if m.group('path'):
                # path exists - restore first line
                lines.insert(0, fl)
            if self.options['linenos'] is None and m.group('shebang'):
                # Overridable and Shebang exists - use line numbers
                self.options['linenos'] = True

            self.options['hl_lines'] = parse_hl_lines(m.group('hl_lines'))
        else:
            # No match
            lines.insert(0, fl)

        self.src = "\n".join(lines).strip("\n")


```

这段代码是一个名为 `HiliteTreeprocessor` 的类，它是 `Markdown` 扩展中的一个子类。它的作用是在给定的 `Treeprocessor` 实例中，将 Markdown 代码块高亮显示，并将其存储为 `htmlStash` 对象中。

具体来说，这段代码实现以下两个方法：

1. `code_unescape` 方法：这个方法接收一个 Markdown 代码块，并对其进行处理，以去除其中的 HTML 标签，将“&”标签替换为相应的 HTML 实体，最后返回到给定的代码块。
2. `run` 方法：这个方法接收一个元素的根节点，并遍历该元素中的所有预格式标签（也就是 code 标签）。对于每个预格式标签，如果它的文本内容只有一个字符，那么这个方法会将 Markdown 代码块解析并高亮显示，然后将其存储到 `htmlStash` 对象中。

`HiliteTreeprocessor` 类的实例可以被用来执行以下操作：

1. 提取元素中的所有预格式标签（code 标签），并将它们高亮显示为 HTML。
2. 将 Markdown 代码块存储为 `htmlStash` 对象中。
3. 将预格式标签（code 标签）的文本内容解析为 HTML，并在代码块中插入新的空格，以使生成的 HTML 代码块具有更好的可读性。
4. 将生成的 HTML 代码块存储到 `htmlStash` 对象中。


```py
# ------------------ The Markdown Extension -------------------------------


class HiliteTreeprocessor(Treeprocessor):
    """ Highlight source code in code blocks. """

    config: dict[str, Any]

    def code_unescape(self, text: str) -> str:
        """Unescape code."""
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        # Escaped '&' should be replaced at the end to avoid
        # conflicting with < and >.
        text = text.replace("&amp;", "&")
        return text

    def run(self, root: etree.Element) -> None:
        """ Find code blocks and store in `htmlStash`. """
        blocks = root.iter('pre')
        for block in blocks:
            if len(block) == 1 and block[0].tag == 'code':
                local_config = self.config.copy()
                code = CodeHilite(
                    self.code_unescape(block[0].text),
                    tab_length=self.md.tab_length,
                    style=local_config.pop('pygments_style', 'default'),
                    **local_config
                )
                placeholder = self.md.htmlStash.store(code.hilite())
                # Clear code block in `etree` instance
                block.clear()
                # Change to `p` element which will later
                # be removed when inserting raw html
                block.tag = 'p'
                block.text = placeholder


```

This is a Python class that appears to be a plugin for the `mark.坛` (Markdown) extension, called `HilitePostprocessor`. It appears to be used to highlight code blocks in Markdown, using the `pygments` library for syntax highlighting.

The class has several configuration options, including `css_class`, `pygments_style`, `lang_prefix`, and `pygments_formatter`. These options can be used to customize the appearance of highlighted code blocks in Markdown.

The class also has a `use_pygments` option, which is a boolean value that controls whether to use `pygments` for syntax highlighting. If `use_pygments` is `True`, the class will automatically启用 syntax highlighting, and if it is `False`, the class will not enable syntax highlighting but will still allow `pygments` to be used in conjunction with `mark.坛`.

Overall, this class seems to be a useful tool for highlighting code blocks in Markdown, and provides a variety of configuration options for customize the appearance of highlighted blocks.


```py
class CodeHiliteExtension(Extension):
    """ Add source code highlighting to markdown code blocks. """

    def __init__(self, **kwargs):
        # define default configs
        self.config = {
            'linenums': [
                None, "Use lines numbers. True|table|inline=yes, False=no, None=auto. Default: `None`."
            ],
            'guess_lang': [
                True, "Automatic language detection - Default: `True`."
            ],
            'css_class': [
                "codehilite", "Set class name for wrapper <div> - Default: `codehilite`."
            ],
            'pygments_style': [
                'default', 'Pygments HTML Formatter Style (Colorscheme). Default: `default`.'
            ],
            'noclasses': [
                False, 'Use inline styles instead of CSS classes - Default `False`.'
            ],
            'use_pygments': [
                True, 'Highlight code blocks with pygments. Disable if using a JavaScript library. Default: `True`.'
            ],
            'lang_prefix': [
                'language-', 'Prefix prepended to the language when `use_pygments` is false. Default: `language-`.'
            ],
            'pygments_formatter': [
                'html', 'Use a specific formatter for Pygments highlighting. Default: `html`.'
            ],
        }
        """ Default configuration options. """

        for key, value in kwargs.items():
            if key in self.config:
                self.setConfig(key, value)
            else:
                # manually set unknown keywords.
                if isinstance(value, str):
                    try:
                        # Attempt to parse `str` as a boolean value
                        value = parseBoolValue(value, preserve_none=True)
                    except ValueError:
                        pass  # Assume it's not a boolean value. Use as-is.
                self.config[key] = [value, '']

    def extendMarkdown(self, md):
        """ Add `HilitePostprocessor` to Markdown instance. """
        hiliter = HiliteTreeprocessor(md)
        hiliter.config = self.getConfigs()
        md.treeprocessors.register(hiliter, 'hilite', 30)

        md.registerExtension(self)


```

这段代码定义了一个名为`makeExtension`的函数，它接受一个或多个参数`**kwargs`，并将它们传递给名为`CodeHiliteExtension`的函数，该函数返回一个经优化的`CodeHiliteExtension`对象。

具体来说，这段代码使用了Python中的一个特性——元类（generator type），也称为生成器类型。在这种类型中，可以通过对参数的输入来定义一个函数，而不是为这个函数提供参数。

`makeExtension`函数的实现中，首先定义了一个`**kwargs`的类型，这实际上是一个生成器类型。然后，在这个类型中定义了`makeExtension`函数，它的作用就是接受一个或多个参数，并将它们传递给`CodeHiliteExtension`函数，最终返回一个`CodeHiliteExtension`对象。

由于`makeExtension`函数接受一个或多个参数，并且这个函数本身并没有定义任何局部变量，因此它的作用就是返回一个`CodeHiliteExtension`对象，这个对象可以在调用它的函数中用来定义代码块、函数内等等。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return CodeHiliteExtension(**kwargs)

```

# `/markdown/markdown/extensions/def_list.py`

这段代码是一个Python-Markdown的定义列表扩展，它定义了一个名为“定义列表”的类，它可以让你在Python-Markdown中解析定义列表。

定义列表是一种特殊的Markdown列表，可以包含多个选项，其格式为<li>选项1</li>[选项2]<li>选项3</li>...。这个类在Python-Markdown中提供了一些特殊的处理定义列表的方法，如parse()、to_dict()、filter()等。

因此，这个代码的主要作用是扩展Python-Markdown的功能，使其可以解析定义列表，并提供了定义列表的一些特殊方法。


```py
# Definition List Extension for Python-Markdown
# =============================================

# Adds parsing of Definition Lists to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/definition_lists
# for documentation.

# Original code Copyright 2008 [Waylan Limberg](http://achinghead.com)

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
```

This appears to be a regular expression (regex) for matching terms in a block of text. The regex is using the `raw_block` variable, which is being split from a list by newline characters (`\n`). The regex is using the `m` variable, which contains a match object that represents the start index of the term to match.

The regex is then using the `.split()` method to split the `raw_block` by newline characters, and returning a list of all the terms.

The block is then being built up by concatenating each term in the list, with each term being padded with a placeholder (indented) using the `self.NO_INDENT_RE.match()` method. If the `match()` method does not find a match for the term, the term is直接 concatenated without indentation.

The block is then being passed to the `d` method, which is building up a definition object. If the block does not contain a definition object, the block is being passed to the `p` method. This is building up a paragraph element, and the `state` variable is set to `'looselist'`.

If the block is a list, the `state` variable is set to `'list'`.

The block is then being passed to the `parseBlocks()` method, which is parsing the block using the defined regular expression.

The `.lastChild()` method is being used to access the `dt` element that is the child of the `dl` element.

This script appears to be processing a `.txt` file that contains a list of terms and definitions.


```py
Adds parsing of Definition Lists to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/definition_lists)
for details.
"""

from __future__ import annotations

from . import Extension
from ..blockprocessors import BlockProcessor, ListIndentProcessor
import xml.etree.ElementTree as etree
import re


class DefListProcessor(BlockProcessor):
    """ Process Definition Lists. """

    RE = re.compile(r'(^|\n)[ ]{0,3}:[ ]{1,3}(.*?)(\n|$)')
    NO_INDENT_RE = re.compile(r'^[ ]{0,3}[^ :]')

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.search(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> bool | None:

        raw_block = blocks.pop(0)
        m = self.RE.search(raw_block)
        terms = [term.strip() for term in
                 raw_block[:m.start()].split('\n') if term.strip()]
        block = raw_block[m.end():]
        no_indent = self.NO_INDENT_RE.match(block)
        if no_indent:
            d, theRest = (block, None)
        else:
            d, theRest = self.detab(block)
        if d:
            d = '{}\n{}'.format(m.group(2), d)
        else:
            d = m.group(2)
        sibling = self.lastChild(parent)
        if not terms and sibling is None:
            # This is not a definition item. Most likely a paragraph that
            # starts with a colon at the beginning of a document or list.
            blocks.insert(0, raw_block)
            return False
        if not terms and sibling.tag == 'p':
            # The previous paragraph contains the terms
            state = 'looselist'
            terms = sibling.text.split('\n')
            parent.remove(sibling)
            # Acquire new sibling
            sibling = self.lastChild(parent)
        else:
            state = 'list'

        if sibling is not None and sibling.tag == 'dl':
            # This is another item on an existing list
            dl = sibling
            if not terms and len(dl) and dl[-1].tag == 'dd' and len(dl[-1]):
                state = 'looselist'
        else:
            # This is a new list
            dl = etree.SubElement(parent, 'dl')
        # Add terms
        for term in terms:
            dt = etree.SubElement(dl, 'dt')
            dt.text = term
        # Add definition
        self.parser.state.set(state)
        dd = etree.SubElement(dl, 'dd')
        self.parser.parseBlocks(dd, [d])
        self.parser.state.reset()

        if theRest:
            blocks.insert(0, theRest)


```

这段代码定义了一个名为 `DefListIndentProcessor` 的类，其作用是处理定义列表中所有 `dd` 和 `li` 类型的子元素。

具体来说，这个类继承自另一个名为 `ListIndentProcessor` 的类，它定义了 `ITEM_TYPES` 和 `LIST_TYPES` 两个属性，用于指定要处理的列表元素类型。然后，在这个类的 `create_item` 方法中，根据传入的 `parent` 元素类型，选择创建 `dd` 元素或 `li` 元素，并对其中的 `block` 元素进行解析。最后，这个 `DefListIndentProcessor` 类还定义了一个 `parser.parseBlocks` 方法，用于递归地解析 `dd` 和 `li` 类型的子元素。


```py
class DefListIndentProcessor(ListIndentProcessor):
    """ Process indented children of definition list items. """

    # Definition lists need to be aware of all list types
    ITEM_TYPES = ['dd', 'li']
    """ Include `dd` in list item types. """
    LIST_TYPES = ['dl', 'ol', 'ul']
    """ Include `dl` is list types. """

    def create_item(self, parent: etree.Element, block: str) -> None:
        """ Create a new `dd` or `li` (depending on parent) and parse the block with it as the parent. """

        dd = etree.SubElement(parent, 'dd')
        self.parser.parseBlocks(dd, [block])


```

这段代码定义了一个名为 "DefListExtension" 的扩展，该扩展在 Markdown 中添加定义列表。

具体来说，这个扩展继承自 "Extension"，这是 Python 3.6 中的一个模块，用于在代码中添加自定义功能。在 "DefListExtension" 中，扩展中包含两个函数：`extendMarkdown` 和 `makeExtension`。

`extendMarkdown` 函数将一个 Markdown 对象（md）传递给定义，然后将其中的 `BlockParser` 实例注册为一个名为 "defindent" 的 85 级延迟处理程序，将其注册为 `BlockParser` 的实例。同时，它还注册了一个名为 "deflist" 的 25 级延迟处理程序，将其注册为 `BlockParser` 的实例。这个延迟处理程序会在渲染到输出时添加延迟，以便在页面上输出内容。

`makeExtension` 函数将一个字典作为参数，其中包含定义列表的键和值。它返回一个将定义列表添加到 Markdown 对象的函数。这个函数会将定义列表中的每个条目转换为一个延迟，并将其添加到 `BlockParser` 的实例中。

由于 `makeExtension` 函数中包含延迟处理程序，因此它会在渲染到输出时延迟添加定义列表的内容。这使得用户可以体验到定义列表在页面上呈现的效果，而无需立即将其添加到页面上。


```py
class DefListExtension(Extension):
    """ Add definition lists to Markdown. """

    def extendMarkdown(self, md):
        """ Add an instance of `DefListProcessor` to `BlockParser`. """
        md.parser.blockprocessors.register(DefListIndentProcessor(md.parser), 'defindent', 85)
        md.parser.blockprocessors.register(DefListProcessor(md.parser), 'deflist', 25)


def makeExtension(**kwargs):  # pragma: no cover
    return DefListExtension(**kwargs)

```

# `/markdown/markdown/extensions/extra.py`

这段代码是一个Python-Markdown扩展的编译，其中包括了各种Python-Markdown扩展，以模仿[PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)。它通过引用[PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)中的文本来定义和编译这些扩展。最后，它定义了一个扩展名，以告诉Python-Markdown如何使用它。


```py
# Python-Markdown Extra Extension
# ===============================

# A compilation of various Python-Markdown extensions that imitates
# [PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/).

# See https://Python-Markdown.github.io/extensions/extra
# for documentation.

# Copyright The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
A compilation of various Python-Markdown extensions that imitates
[PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/).

```

这段代码是一个用于确保 Python Markdown 扩展在启动时仅列出一个扩展的 convenience 函数。它通过将所有扩展名存储在一个列表中，这样可以方便地在启动时使用马克down，而无需在运行时逐个加载每个扩展。

对于每个扩展，该函数都会在 `PYTHONPATH` 目录的扩展名列表中添加扩展。但是，该列表并不会包含附录中定义的扩展，这些扩展只能通过创建一个名为自己的 Extra 克隆并在不同的 Python-Markdown 版本中使用自定义配置文件来启用。

因此，该代码的作用是提供一个方便的方式来使用 Python Markdown 的所有扩展，而不会使扩展名称与其 Python-Markdown 版本不匹配。如果需要使用自定义扩展，可以创建一个名为 `Extra` 的克隆，并在 `extensions` 全局变量中更改为所需的扩展名称。不过，如果升级到 Python-Markdown 的任何新版本，这些更改可能会丢失。


```py
Note that each of the individual extensions still need to be available
on your `PYTHONPATH`. This extension simply wraps them all up as a
convenience so that only one extension needs to be listed when
initiating Markdown. See the documentation for each individual
extension for specifics about that extension.

There may be additional extensions that are distributed with
Python-Markdown that are not included here in Extra. Those extensions
are not part of PHP Markdown Extra, and therefore, not part of
Python-Markdown Extra. If you really would like Extra to include
additional extensions, we suggest creating your own clone of Extra
under a different name. You could also edit the `extensions` global
variable defined below, but be aware that such changes may be lost
when you upgrade to any future version of Python-Markdown.

```

这段代码是一个Python文档字符串，它将定义一个名为"extensions"的列表。列表中的每个元素都是一个字符串，它将包含一个扩展，该扩展用于在Python中使用Markdown。

具体来说，这段代码将定义以下扩展：

- fenced_code：一个可以包含Markdown的代码块。
- footnotes：一个可以包含Markdown脚注的扩展。
- attr_list：一个可以包含Markdown属性列表的扩展。
- def_list：一个可以包含Markdown定义列表的扩展。
- tables：一个可以包含Markdown表格的扩展。
- abbr：一个可以包含Markdown缩写的扩展。
- md_in_html：一个可以将Markdown内容转换为HTML的扩展。

此外，这段代码还通过附录（ See the documentation ）链接了Markdown文档，以便用户获得更多信息和示例。


```py
See the [documentation](https://Python-Markdown.github.io/extensions/extra)
for details.
"""

from __future__ import annotations

from . import Extension

extensions = [
    'fenced_code',
    'footnotes',
    'attr_list',
    'def_list',
    'tables',
    'abbr',
    'md_in_html'
]
```

这段代码定义了一个名为ExtraExtension的类，它继承自名为Extension的类。这个类的构造函数接受一个 keyword argument，即kwargs，它用于存储扩展的配置信息。这个类的扩展Markdown方法允许将扩展实例注册到Markdown对象中，并使用config参数来获取扩展的配置信息。

具体来说，这段代码的作用是扩展Markdown文件的语法，允许在Markdown中使用自定义的扩展。通过将ExtraExtension实例注册到Markdown对象中，可以定义扩展的优先级，以使扩展的代码在Markdown文件中具有更高的可见度。


```py
""" The list of included extensions. """


class ExtraExtension(Extension):
    """ Add various extensions to Markdown class."""

    def __init__(self, **kwargs):
        """ `config` is a dumb holder which gets passed to the actual extension later. """
        self.config = kwargs

    def extendMarkdown(self, md):
        """ Register extension instances. """
        md.registerExtensions(extensions, self.config)


```

这段代码定义了一个名为`makeExtension`的函数，它接受一个或多个参数`**kwargs`，并将它们传递给名为`ExtraExtension`的函数，这个函数的接收参数是`**kwargs`。

具体来说，这段代码的作用是创建一个名为`ExtraExtension`的函数，它可以接受一个或多个参数`**kwargs`，然后对这个函数进行装饰，使得这个函数在接收一个或多个参数的情况下，仍然可以接受一个或多个参数。这个装饰的作用是在函数定义时，将传递给`ExtraExtension`的参数数量增加到和定义时参数数量相同。这样，当调用`makeExtension`时，即使传递给它的参数数量发生变化，`ExtraExtension`函数仍然可以接受相同数量的参数。

例如，如果你在调用`makeExtension`时传递了5个参数，那么实际上会传递给`ExtraExtension`函数6个参数（5个参数加上函数定义时的参数数量）。如果你在调用`makeExtension`时传递了10个参数，那么`ExtraExtension`函数仍然会接受这10个参数。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return ExtraExtension(**kwargs)

```

# `/markdown/markdown/extensions/fenced_code.py`

这段代码是一个Python脚本，它是一个带有围栏代码块的扩展，目的是让用户在Python-Markdown中更方便地使用带有文档的代码块。

具体来说，这段代码实现了以下功能：

1. 定义了一个名为`fenced_code_block`的类，该类继承自Python-Markdown中的`code_block`类。
2. 在定义`fenced_code_block`类的过程中，实现了以下几处修改：
a. 移除了`__init__`方法，因为该方法不需要在这里实现。
b. 修改了`code_block`类的`body`属性，使其可以包含一个或多个`f-string`，从而可以将文档字符串插入到代码块中。
c. 添加了一个名为`fenced_code_block_Macro`的新方法，用于在脚本中定义自定义的文档字符串，该方法使用了`f-string`。

3. 在脚本的顶部，定义了一个名为`__fenced_code_block_extension__`的类，该类包含了`fenced_code_block`类的实例。
4. 在`__fenced_code_block_extension__`类中，定义了一个`extension_def_docs`方法，用于定义该扩展的文档字符串。
5. 在定义`fenced_code_block`类和`__fenced_code_block_extension__`类的过程中，使用了以下外部链接：
https://Python-Markdown.github.io/extensions/fenced_code_blocks<https://www.python-markdown.org/extensions/fenced_code_blocks.html>


```py
# Fenced Code Extension for Python Markdown
# =========================================

# This extension adds Fenced Code Blocks to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/fenced_code_blocks
# for documentation.

# Original code Copyright 2007-2008 [Waylan Limberg](http://achinghead.com/).

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
```

这段代码是一个Python扩展，名为Fenced Code Blocks。它向Python-Markdown中添加了带内边距的代码块。内边距代码块在两个高亮度级别上显示，一个是`<pre>`标签，另一个是`<code>`标签。通过将`fenced_code_blocks`参数传递给`CodeHilite`类，可以设置内边距代码块的格式ting。同时，通过`parse_hl_lines`方法解析的hl标签行也被添加到了代码块中。


```py
This extension adds Fenced Code Blocks to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/fenced_code_blocks)
for details.
"""

from __future__ import annotations

from textwrap import dedent
from . import Extension
from ..preprocessors import Preprocessor
from .codehilite import CodeHilite, CodeHiliteExtension, parse_hl_lines
from .attr_list import get_attrs, AttrListExtension
from ..util import parseBoolValue
from ..serializers import _escape_attrib_html
```

这段代码是一个Python语言的类，名为`FencedCodeExtension`。它是一个扩展，用于在Markdown中嵌入fenced代码。这里有一个导入了`re`和`typing.TypesChecking`的函数`markdown`的实例。

`FencedCodeExtension`有两个方法：

1. `__init__`：这是构造函数，用于初始化`FencedCodeExtension`的配置。`lang_prefix`是一个字典，其中包含一个或多个`language-`前缀，这些前缀用于指定将要嵌入的编程语言。如果没有指定前缀，则默认为"language-"`。
2. `extendMarkdown`：这个方法将`FencedBlockPreprocessor`添加到Markdown实例中，并注册一个`FencedCodeBlock`预处理程序。这个预处理程序会在Markdown被输出之前被调用，以便可以添加fenced代码。

这里有一个使用`typing.TypesChecking`的语句，用于确保`FencedCodeExtension`可以正确地使用`markdown`实例。如果没有这个语句，程序可能无法在带有`markdown`的Python环境中运行。


```py
import re
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


class FencedCodeExtension(Extension):
    def __init__(self, **kwargs):
        self.config = {
            'lang_prefix': ['language-', 'Prefix prepended to the language. Default: "language-"']
        }
        """ Default configuration options. """
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add `FencedBlockPreprocessor` to the Markdown instance. """
        md.registerExtension(self)

        md.preprocessors.register(FencedBlockPreprocessor(md, self.getConfigs()), 'fenced_code_block', 25)


```

The `MD` class is a class that uses the Modern Language Disco (MLD) format for writing markdown. It has a number of methods, including `handle_attrs`, which takes an iterable of attrs (short for attributes) and returns a tuple of the form `(id, classes, configs)`.

The `handle_attrs` method takes the attributes for the markdown string and returns a tuple of three parts:

* `id`: The ID of the element that the attribute applies to. This is useful for ensuring that elements are uniquely named.
* `classes`: A list of classes that are affected by the attribute. This allows you to apply the same class to multiple elements.
* `configs`: A dictionary that maps the attribute name to its configuration. This can be used to customize the behavior of the attribute.

The `handle_attrs` method is called by the `md.md. placeholders.generate` method, which is responsible for generating the markdown string. It is called with an iterable of attributes, which are processed by the method and added to the `configs` dictionary. The method then returns the tuple of `id`, `classes`, and `configs`.

The `_escape` method is a simple method for escaping the markdown string. It replaces any HTML entities with their corresponding AMP or HTML entities.


```py
class FencedBlockPreprocessor(Preprocessor):
    """ Find and extract fenced code blocks. """

    FENCED_BLOCK_RE = re.compile(
        dedent(r'''
            (?P<fence>^(?:~{3,}|`{3,}))[ ]*                          # opening fence
            ((\{(?P<attrs>[^\}\n]*)\})|                              # (optional {attrs} or
            (\.?(?P<lang>[\w#.+-]*)[ ]*)?                            # optional (.)lang
            (hl_lines=(?P<quot>"|')(?P<hl_lines>.*?)(?P=quot)[ ]*)?) # optional hl_lines)
            \n                                                       # newline (end of opening fence)
            (?P<code>.*?)(?<=\n)                                     # the code block
            (?P=fence)[ ]*$                                          # closing fence
        '''),
        re.MULTILINE | re.DOTALL | re.VERBOSE
    )

    def __init__(self, md: Markdown, config: dict[str, Any]):
        super().__init__(md)
        self.config = config
        self.checked_for_deps = False
        self.codehilite_conf: dict[str, Any] = {}
        self.use_attr_list = False
        # List of options to convert to boolean values
        self.bool_options = [
            'linenums',
            'guess_lang',
            'noclasses',
            'use_pygments'
        ]

    def run(self, lines: list[str]) -> list[str]:
        """ Match and store Fenced Code Blocks in the `HtmlStash`. """

        # Check for dependent extensions
        if not self.checked_for_deps:
            for ext in self.md.registeredExtensions:
                if isinstance(ext, CodeHiliteExtension):
                    self.codehilite_conf = ext.getConfigs()
                if isinstance(ext, AttrListExtension):
                    self.use_attr_list = True

            self.checked_for_deps = True

        text = "\n".join(lines)
        while 1:
            m = self.FENCED_BLOCK_RE.search(text)
            if m:
                lang, id, classes, config = None, '', [], {}
                if m.group('attrs'):
                    id, classes, config = self.handle_attrs(get_attrs(m.group('attrs')))
                    if len(classes):
                        lang = classes.pop(0)
                else:
                    if m.group('lang'):
                        lang = m.group('lang')
                    if m.group('hl_lines'):
                        # Support `hl_lines` outside of `attrs` for backward-compatibility
                        config['hl_lines'] = parse_hl_lines(m.group('hl_lines'))

                # If `config` is not empty, then the `codehighlite` extension
                # is enabled, so we call it to highlight the code
                if self.codehilite_conf and self.codehilite_conf['use_pygments'] and config.get('use_pygments', True):
                    local_config = self.codehilite_conf.copy()
                    local_config.update(config)
                    # Combine classes with `cssclass`. Ensure `cssclass` is at end
                    # as Pygments appends a suffix under certain circumstances.
                    # Ignore ID as Pygments does not offer an option to set it.
                    if classes:
                        local_config['css_class'] = '{} {}'.format(
                            ' '.join(classes),
                            local_config['css_class']
                        )
                    highliter = CodeHilite(
                        m.group('code'),
                        lang=lang,
                        style=local_config.pop('pygments_style', 'default'),
                        **local_config
                    )

                    code = highliter.hilite(shebang=False)
                else:
                    id_attr = lang_attr = class_attr = kv_pairs = ''
                    if lang:
                        prefix = self.config.get('lang_prefix', 'language-')
                        lang_attr = f' class="{prefix}{_escape_attrib_html(lang)}"'
                    if classes:
                        class_attr = f' class="{_escape_attrib_html(" ".join(classes))}"'
                    if id:
                        id_attr = f' id="{_escape_attrib_html(id)}"'
                    if self.use_attr_list and config and not config.get('use_pygments', False):
                        # Only assign key/value pairs to code element if `attr_list` extension is enabled, key/value
                        # pairs were defined on the code block, and the `use_pygments` key was not set to `True`. The
                        # `use_pygments` key could be either set to `False` or not defined. It is omitted from output.
                        kv_pairs = ''.join(
                            f' {k}="{_escape_attrib_html(v)}"' for k, v in config.items() if k != 'use_pygments'
                        )
                    code = self._escape(m.group('code'))
                    code = f'<pre{id_attr}{class_attr}><code{lang_attr}{kv_pairs}>{code}</code></pre>'

                placeholder = self.md.htmlStash.store(code)
                text = f'{text[:m.start()]}\n{placeholder}\n{text[m.end():]}'
            else:
                break
        return text.split("\n")

    def handle_attrs(self, attrs: Iterable[tuple[str, str]]) -> tuple[str, list[str], dict[str, Any]]:
        """ Return tuple: `(id, [list, of, classes], {configs})` """
        id = ''
        classes = []
        configs = {}
        for k, v in attrs:
            if k == 'id':
                id = v
            elif k == '.':
                classes.append(v)
            elif k == 'hl_lines':
                configs[k] = parse_hl_lines(v)
            elif k in self.bool_options:
                configs[k] = parseBoolValue(v, fail_on_errors=False, preserve_none=True)
            else:
                configs[k] = v
        return id, classes, configs

    def _escape(self, txt: str) -> str:
        """ basic html escaping """
        txt = txt.replace('&', '&amp;')
        txt = txt.replace('<', '&lt;')
        txt = txt.replace('>', '&gt;')
        txt = txt.replace('"', '&quot;')
        return txt


```

这段代码定义了一个名为`makeExtension`的函数，它接受一个或多个参数`**kwargs`，并将它们传递给一个名为`FencedCodeExtension`的函数，这个函数的作用是接收`**kwargs`作为参数，并返回一个经过扩展的版本。

具体来说，这段代码使用了一个被称为`**kwargs`的语法，它允许在函数定义中使用`**`来展开相关的参数。通过这种方法，`makeExtension`函数可以很容易地扩展一个函数，而不需要显式地指定参数。

在函数体中，`makeExtension`函数创建了一个名为`FencedCodeExtension`的函数，这个函数接受一个或多个参数，并将它们传递给`FencedCodeExtension`函数本身。这个`FencedCodeExtension`函数的作用是在函数内部扩展函数，可以在函数外部被调用，也可以作为函数参数传递给其他函数。

总结一下，这段代码定义了一个`makeExtension`函数，它接受一个或多个参数`**kwargs`，并将它们传递给一个名为`FencedCodeExtension`的函数。通过使用`**kwargs`语法，`makeExtension`函数可以方便地扩展一个函数，而不需要显式地指定参数。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return FencedCodeExtension(**kwargs)

```

# `/markdown/markdown/extensions/footnotes.py`

这段代码是一个Python脚本，它添加了一个 footnote 扩展，可以在 Python-Markdown 中使用。footnote 是一种用于在文档中添加 footnote 的工具，通常用于在文章或页面底部添加引用或解释。

具体来说，这段代码做了以下几件事情：

1. 引入了 Python-Markdown 库的 footnote 扩展。
2. 创建了一个名为 `footnotes` 的类，继承自自定义的 `docutils` 扩展的 `generator` 类。
3. 重写了 `generate` 方法，在其中实现了 footnote 的添加。
4. 在 `generate` 方法的示例中使用了 `（ documented footnote 格式 ）` 来指定 footnote 的内容。
5. 在 ` footnotes` 类中添加了 `import footnotes`，以便在需要时可以使用这个类。
6. 最后，定义了代码的著作权和许可证信息。


```py
# Footnotes Extension for Python-Markdown
# =======================================

# Adds footnote handling to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/footnotes
# for documentation.

# Copyright The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Adds footnote handling to Python-Markdown.

```

这段代码是一个Python类，名为Extension。它定义了一个Extension类，该类包含多个BlockProcessor、InlineProcessor、Treeprocessor和Postprocessor类。

具体来说，这段代码实现了一个正则表达式的解析过程，通过这些类对Markdown文档进行处理。在处理过程中，可以对文档进行一些操作，如提取特定类别的块、链接、图片等，并对结果进行排序、提取或者修改。

这些类都实现了Python标准库中的对应函数，如util.parse_image，re.sub等等。同时，还通过OrderedDict实现了有序输出Markdown文档。

该类的使用方式是在需要对Markdown文档进行一些处理时，通过extension = Extension.from_filename()来实例化该类，然后使用该类中的方法对文档进行处理，最后将结果保存到OrderedDict中。


```py
See the [documentation](https://Python-Markdown.github.io/extensions/footnotes)
for details.
"""

from __future__ import annotations

from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..treeprocessors import Treeprocessor
from ..postprocessors import Postprocessor
from .. import util
from collections import OrderedDict
import re
import copy
```

This is a function that takes an `Element` object and returns an `etree.Element` object representing a `div` element with a footnote div.

It works by first checking if the footnotes are empty and if not, it returns `None`. If the footnotes are not empty, it creates an `etree.Element` object with a `div` element and sets it to have a class of "footnote". Then it creates an ordered list (`ol`) and a surrogate parent `div` element.

It then iterates through the footnotes, parsing each chunk of the footnote using the `FootnoteParser` class and appending the parsed chunk to the surrogate parent. Finally, it creates a backward compatibility link to the footnote in the surrogate parent and appends it to the surrogate parent.

It returns the `div` element with the footnote element as its child, so that it can be added to the `body` of an HTML document.


```py
import xml.etree.ElementTree as etree

FN_BACKLINK_TEXT = util.STX + "zz1337820767766393qq" + util.ETX
NBSP_PLACEHOLDER = util.STX + "qq3936677670287331zz" + util.ETX
RE_REF_ID = re.compile(r'(fnref)(\d+)')


class FootnoteExtension(Extension):
    """ Footnote Extension. """

    def __init__(self, **kwargs):
        """ Setup configs. """

        self.config = {
            'PLACE_MARKER': [
                '///Footnotes Go Here///', 'The text string that marks where the footnotes go'
            ],
            'UNIQUE_IDS': [
                False, 'Avoid name collisions across multiple calls to `reset()`.'
            ],
            'BACKLINK_TEXT': [
                '&#8617;', "The text string that links from the footnote to the reader's place."
            ],
            'SUPERSCRIPT_TEXT': [
                '{}', "The text string that links from the reader's place to the footnote."
            ],
            'BACKLINK_TITLE': [
                'Jump back to footnote %d in the text',
                'The text string used for the title HTML attribute of the backlink. '
                '%d will be replaced by the footnote number.'
            ],
            'SEPARATOR': [
                ':', 'Footnote separator.'
            ]
        }
        """ Default configuration options. """
        super().__init__(**kwargs)

        # In multiple invocations, emit links that don't get tangled.
        self.unique_prefix = 0
        self.found_refs: dict[str, int] = {}
        self.used_refs: set[str] = set()

        self.reset()

    def extendMarkdown(self, md):
        """ Add pieces to Markdown. """
        md.registerExtension(self)
        self.parser = md.parser
        self.md = md
        # Insert a `blockprocessor` before `ReferencePreprocessor`
        md.parser.blockprocessors.register(FootnoteBlockProcessor(self), 'footnote', 17)

        # Insert an inline pattern before `ImageReferencePattern`
        FOOTNOTE_RE = r'\[\^([^\]]*)\]'  # blah blah [^1] blah
        md.inlinePatterns.register(FootnoteInlineProcessor(FOOTNOTE_RE, self), 'footnote', 175)
        # Insert a tree-processor that would actually add the footnote div
        # This must be before all other tree-processors (i.e., `inline` and
        # `codehilite`) so they can run on the the contents of the div.
        md.treeprocessors.register(FootnoteTreeprocessor(self), 'footnote', 50)

        # Insert a tree-processor that will run after inline is done.
        # In this tree-processor we want to check our duplicate footnote tracker
        # And add additional `backrefs` to the footnote pointing back to the
        # duplicated references.
        md.treeprocessors.register(FootnotePostTreeprocessor(self), 'footnote-duplicate', 15)

        # Insert a postprocessor after amp_substitute processor
        md.postprocessors.register(FootnotePostprocessor(self), 'footnote', 25)

    def reset(self) -> None:
        """ Clear footnotes on reset, and prepare for distinct document. """
        self.footnotes: OrderedDict[str, str] = OrderedDict()
        self.unique_prefix += 1
        self.found_refs = {}
        self.used_refs = set()

    def unique_ref(self, reference: str, found: bool = False) -> str:
        """ Get a unique reference if there are duplicates. """
        if not found:
            return reference

        original_ref = reference
        while reference in self.used_refs:
            ref, rest = reference.split(self.get_separator(), 1)
            m = RE_REF_ID.match(ref)
            if m:
                reference = '%s%d%s%s' % (m.group(1), int(m.group(2))+1, self.get_separator(), rest)
            else:
                reference = '%s%d%s%s' % (ref, 2, self.get_separator(), rest)

        self.used_refs.add(reference)
        if original_ref in self.found_refs:
            self.found_refs[original_ref] += 1
        else:
            self.found_refs[original_ref] = 1
        return reference

    def findFootnotesPlaceholder(
        self, root: etree.Element
    ) -> tuple[etree.Element, etree.Element, bool] | None:
        """ Return ElementTree Element that contains Footnote placeholder. """
        def finder(element):
            for child in element:
                if child.text:
                    if child.text.find(self.getConfig("PLACE_MARKER")) > -1:
                        return child, element, True
                if child.tail:
                    if child.tail.find(self.getConfig("PLACE_MARKER")) > -1:
                        return child, element, False
                child_res = finder(child)
                if child_res is not None:
                    return child_res
            return None

        res = finder(root)
        return res

    def setFootnote(self, id: str, text: str) -> None:
        """ Store a footnote for later retrieval. """
        self.footnotes[id] = text

    def get_separator(self) -> str:
        """ Get the footnote separator. """
        return self.getConfig("SEPARATOR")

    def makeFootnoteId(self, id: str) -> str:
        """ Return footnote link id. """
        if self.getConfig("UNIQUE_IDS"):
            return 'fn%s%d-%s' % (self.get_separator(), self.unique_prefix, id)
        else:
            return 'fn{}{}'.format(self.get_separator(), id)

    def makeFootnoteRefId(self, id: str, found: bool = False) -> str:
        """ Return footnote back-link id. """
        if self.getConfig("UNIQUE_IDS"):
            return self.unique_ref('fnref%s%d-%s' % (self.get_separator(), self.unique_prefix, id), found)
        else:
            return self.unique_ref('fnref{}{}'.format(self.get_separator(), id), found)

    def makeFootnotesDiv(self, root: etree.Element) -> etree.Element | None:
        """ Return `div` of footnotes as `etree` Element. """

        if not list(self.footnotes.keys()):
            return None

        div = etree.Element("div")
        div.set('class', 'footnote')
        etree.SubElement(div, "hr")
        ol = etree.SubElement(div, "ol")
        surrogate_parent = etree.Element("div")

        # Backward compatibility with old '%d' placeholder
        backlink_title = self.getConfig("BACKLINK_TITLE").replace("%d", "{}")

        for index, id in enumerate(self.footnotes.keys(), start=1):
            li = etree.SubElement(ol, "li")
            li.set("id", self.makeFootnoteId(id))
            # Parse footnote with surrogate parent as `li` cannot be used.
            # List block handlers have special logic to deal with `li`.
            # When we are done parsing, we will copy everything over to `li`.
            self.parser.parseChunk(surrogate_parent, self.footnotes[id])
            for el in list(surrogate_parent):
                li.append(el)
                surrogate_parent.remove(el)
            backlink = etree.Element("a")
            backlink.set("href", "#" + self.makeFootnoteRefId(id))
            backlink.set("class", "footnote-backref")
            backlink.set(
                "title",
                backlink_title.format(index)
            )
            backlink.text = FN_BACKLINK_TEXT

            if len(li):
                node = li[-1]
                if node.tag == "p":
                    node.text = node.text + NBSP_PLACEHOLDER
                    node.append(backlink)
                else:
                    p = etree.SubElement(li, "p")
                    p.append(backlink)
        return div


```

It looks like this is a Python implementation of a regular expression (re)lookahead that matches against a block of text. The `re.lookahead` method is called with a block of text (`block`) and a regular expression (`pattern`) that is looking for a match anywhere in the block.

The regular expression `pattern` is enclosed in single quotes to indicate that it is a non-capturing group. This means that the `pattern` will only match the text within the block and will not capture any matches.

The code then defines a `detectTabbed` method that takes a list of blocks and returns a list of blocks with indentation removed. This method first checks if the block is wrapped in four spaces (`' '*4`), and if it is, it removes the first four spaces and returns the modified block. If the block is not wrapped in four spaces, it adds the block to the list of blocks to be processed and returns it.

The `detectTabbed` method then checks the block for any match using the regular expression `pattern`. If a match is found, it adds the matched content to the list of blocks to be processed and then removes the indentation. If no match is found, it returns the block as is.

Finally, the `detectTabbed` method returns the list of blocks with indentation removed.


```py
class FootnoteBlockProcessor(BlockProcessor):
    """ Find all footnote references and store for later use. """

    RE = re.compile(r'^[ ]{0,3}\[\^([^\]]*)\]:[ ]*(.*)$', re.MULTILINE)

    def __init__(self, footnotes: FootnoteExtension):
        super().__init__(footnotes.parser)
        self.footnotes = footnotes

    def test(self, parent: etree.Element, block: str) -> bool:
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        """ Find, set, and remove footnote definitions. """
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            id = m.group(1)
            fn_blocks = [m.group(2)]

            # Handle rest of block
            therest = block[m.end():].lstrip('\n')
            m2 = self.RE.search(therest)
            if m2:
                # Another footnote exists in the rest of this block.
                # Any content before match is continuation of this footnote, which may be lazily indented.
                before = therest[:m2.start()].rstrip('\n')
                fn_blocks[0] = '\n'.join([fn_blocks[0], self.detab(before)]).lstrip('\n')
                # Add back to blocks everything from beginning of match forward for next iteration.
                blocks.insert(0, therest[m2.start():])
            else:
                # All remaining lines of block are continuation of this footnote, which may be lazily indented.
                fn_blocks[0] = '\n'.join([fn_blocks[0], self.detab(therest)]).strip('\n')

                # Check for child elements in remaining blocks.
                fn_blocks.extend(self.detectTabbed(blocks))

            footnote = "\n\n".join(fn_blocks)
            self.footnotes.setFootnote(id, footnote.rstrip())

            if block[:m.start()].strip():
                # Add any content before match back to blocks as separate block
                blocks.insert(0, block[:m.start()].rstrip('\n'))
            return True
        # No match. Restore block.
        blocks.insert(0, block)
        return False

    def detectTabbed(self, blocks: list[str]) -> list[str]:
        """ Find indented text and remove indent before further processing.

        Returns:
            A list of blocks with indentation removed.
        """
        fn_blocks = []
        while blocks:
            if blocks[0].startswith(' '*4):
                block = blocks.pop(0)
                # Check for new footnotes within this block and split at new footnote.
                m = self.RE.search(block)
                if m:
                    # Another footnote exists in this block.
                    # Any content before match is continuation of this footnote, which may be lazily indented.
                    before = block[:m.start()].rstrip('\n')
                    fn_blocks.append(self.detab(before))
                    # Add back to blocks everything from beginning of match forward for next iteration.
                    blocks.insert(0, block[m.start():])
                    # End of this footnote.
                    break
                else:
                    # Entire block is part of this footnote.
                    fn_blocks.append(self.detab(block))
            else:
                # End of this footnote.
                break
        return fn_blocks

    def detab(self, block: str) -> str:
        """ Remove one level of indent from a block.

        Preserve lazily indented blocks by only removing indent from indented lines.
        """
        lines = block.split('\n')
        for i, line in enumerate(lines):
            if line.startswith(' '*4):
                lines[i] = line[4:]
        return '\n'.join(lines)


```

该代码定义了一个名为 `FootnoteInlineProcessor` 的类，用于处理文档中的 footnote 标记。这个类实现了 `InlineProcessor` 接口，因此继承了 `InlineProcessor` 的属性和方法。

在 `__init__` 方法中，首先调用父类的 `__init__` 方法，然后将 `pattern` 和 `footnotes` 参数传递给父类的构造函数。

在 `handleMatch` 方法中，定义了一个处理 footnote 标记的函数。当遇到一个 footnote 标记时，该函数根据给出的 id 查找其在 `footnotes` 对象中的 footnote 扩展，并返回一个 `sup` 元素、匹配的 start 和 end 位置。如果 id 在 `footnotes` 对象中存在，则使用 `makeFootnoteRefId` 方法生成一个引用 id，并将生成的链接添加到 `sup` 元素中。如果 id 不存在，则直接返回 `None`，因为无法生成 footnote 链接。

`FootnoteInlineProcessor` 可以被用来在文档中添加 footnote 标记。只需创建一个实例，并将要使用的 footnote 模式和扩展传递给 `FootnoteInlineProcessor` 的构造函数即可。例如：


from footnote import FootnoteInlineProcessor

pattern = r'(\\[？< footnote\]?)\\s*(？<=\\]|$)'
footnotes = FootnoteExtension(
   ' footnote1'
)

processor = FootnoteInlineProcessor(pattern, footnotes)

processor.handleMatch('[？< footnote\]?', '这是一段用于展示 footnote 标记的文本')
 

这将检查所有的 `< footnote?>` 标签，并将所有以 `< footnote?>` 开头的行中的 footnote 链接插入到 `这是一段用于展示 footnote 标记的文本` 的文档中。


```py
class FootnoteInlineProcessor(InlineProcessor):
    """ `InlineProcessor` for footnote markers in a document's body text. """

    def __init__(self, pattern: str, footnotes: FootnoteExtension):
        super().__init__(pattern)
        self.footnotes = footnotes

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        id = m.group(1)
        if id in self.footnotes.footnotes.keys():
            sup = etree.Element("sup")
            a = etree.SubElement(sup, "a")
            sup.set('id', self.footnotes.makeFootnoteRefId(id, found=True))
            a.set('href', '#' + self.footnotes.makeFootnoteId(id))
            a.set('class', 'footnote-ref')
            a.text = self.footnotes.getConfig("SUPERSCRIPT_TEXT").format(
                list(self.footnotes.footnotes.keys()).index(id) + 1
            )
            return sup, m.start(0), m.end(0)
        else:
            return None, None, None


```

This is a Python class that handles the crawling and adding duplicate footnotes to an HTML file. The class has a `run` method that crawls through the footnote div and adds the missing footnotes.

The `handle_duplicates` method finds the number of duplicate footnotes in the HTML and adds the missing links. This method uses the `get_num_duplicates` and `handle_duplicates` functions to find and add the duplicate footnotes.

The `get_num_duplicates` function gets the number of duplicate footnotes based on the footnote ID.

The `add_duplicates` function adds the missing footnotes to the current div. This is done by first finding the number of duplicate footnotes, and then using the `handle_duplicates` method to add the missing footnotes.

Overall, this class is useful for crawling and adding missing footnotes to HTML files with duplicate footnotes.


```py
class FootnotePostTreeprocessor(Treeprocessor):
    """ Amend footnote div with duplicates. """

    def __init__(self, footnotes: FootnoteExtension):
        self.footnotes = footnotes

    def add_duplicates(self, li: etree.Element, duplicates: int) -> None:
        """ Adjust current `li` and add the duplicates: `fnref2`, `fnref3`, etc. """
        for link in li.iter('a'):
            # Find the link that needs to be duplicated.
            if link.attrib.get('class', '') == 'footnote-backref':
                ref, rest = link.attrib['href'].split(self.footnotes.get_separator(), 1)
                # Duplicate link the number of times we need to
                # and point the to the appropriate references.
                links = []
                for index in range(2, duplicates + 1):
                    sib_link = copy.deepcopy(link)
                    sib_link.attrib['href'] = '%s%d%s%s' % (ref, index, self.footnotes.get_separator(), rest)
                    links.append(sib_link)
                    self.offset += 1
                # Add all the new duplicate links.
                el = list(li)[-1]
                for link in links:
                    el.append(link)
                break

    def get_num_duplicates(self, li: etree.Element) -> int:
        """ Get the number of duplicate refs of the footnote. """
        fn, rest = li.attrib.get('id', '').split(self.footnotes.get_separator(), 1)
        link_id = '{}ref{}{}'.format(fn, self.footnotes.get_separator(), rest)
        return self.footnotes.found_refs.get(link_id, 0)

    def handle_duplicates(self, parent: etree.Element) -> None:
        """ Find duplicate footnotes and format and add the duplicates. """
        for li in list(parent):
            # Check number of duplicates footnotes and insert
            # additional links if needed.
            count = self.get_num_duplicates(li)
            if count > 1:
                self.add_duplicates(li, count)

    def run(self, root: etree.Element) -> None:
        """ Crawl the footnote div and add missing duplicate footnotes. """
        self.offset = 0
        for div in root.iter('div'):
            if div.attrib.get('class', '') == 'footnote':
                # Footnotes should be under the first ordered list under
                # the footnote div.  So once we find it, quit.
                for ol in div.iter('ol'):
                    self.handle_duplicates(ol)
                    break


```

这段代码定义了一个名为 "FootnoteTreeprocessor" 的类，其父类是 "Treeprocessor"。这个类的目的是将脚注 div 添加到文档的结尾。

在这个类中，有一个 "(__init____)" 方法，用于初始化对象并设置 "footnotes" 属性为传入的 "FootnoteExtension" 对象。这个 "FootnoteExtension" 类据说是一个扩展，但在这里并没有定义具体的实现。

有一个 "run" 方法，这个方法将根元素 "root" 作为参数，并返回 void。在 "run" 方法中，首先调用 "footnotes.makeFootnotesDiv" 方法将 footnotes div 创建出来，如果没有返回值，则表示根元素中包含有脚注 div。接着，使用 "findFootnotesPlaceholder" 方法查找 "root" 元素中包含 footnotes div 的父元素，如果找到了，则执行以下操作：将 footnotes div 插入到父元素中，并将脚注 div 的内容设置为 isText=True，这样就不会对文本内容进行操作。如果查找成功，则执行以下操作：删除 child 元素，将 footnotes div 插入到 parent 元素的第一个子元素位置，并将脚注 div 的尾随属性设置为 None。如果查找失败，则将 footnotes div 添加到 "root" 元素中。


```py
class FootnoteTreeprocessor(Treeprocessor):
    """ Build and append footnote div to end of document. """

    def __init__(self, footnotes: FootnoteExtension):
        self.footnotes = footnotes

    def run(self, root: etree.Element) -> None:
        footnotesDiv = self.footnotes.makeFootnotesDiv(root)
        if footnotesDiv is not None:
            result = self.footnotes.findFootnotesPlaceholder(root)
            if result:
                child, parent, isText = result
                ind = list(parent).index(child)
                if isText:
                    parent.remove(child)
                    parent.insert(ind, footnotesDiv)
                else:
                    parent.insert(ind + 1, footnotesDiv)
                    child.tail = None
            else:
                root.append(footnotesDiv)


```

这段代码定义了一个名为 `FootnotePostprocessor` 的类，它继承自 `Postprocessor` 类（在背后可能还有其他类）。这个类的目标是在遍历文本时，将包含 footnote 样式的引用替换为 HTML 实体。

具体来说，这段代码做了以下几件事情：

1. 定义了一个 `__init__` 方法，这个方法接收一个 `FootnoteExtension` 类的实例作为参数。这个实例被传递给 `__init__` 方法，但并不在方法内部使用。
2. 定义了一个 `run` 方法，这个方法接收一个文本字符串作为参数。首先，它使用 `text.replace` 方法，用 footnote 样式的引用（通过 `FN_BACKLINK_TEXT` 配置）替换为 footnote 样式的 HTML 实体。然后，它使用 `text.replace` 方法，用 `NBSP_PLACEHOLDER` 替换为 `&#160;`，即一个 HTML 换行符（`<br>`）。

另外，这段代码还定义了一个名为 `makeExtension` 的函数，用于创建一个 `FootnoteExtension` 类的实例。这个函数接收一个字典（可能还包含其他参数）作为参数，并将其转换为 `FootnoteExtension` 类的实例。


```py
class FootnotePostprocessor(Postprocessor):
    """ Replace placeholders with html entities. """
    def __init__(self, footnotes: FootnoteExtension):
        self.footnotes = footnotes

    def run(self, text: str) -> str:
        text = text.replace(
            FN_BACKLINK_TEXT, self.footnotes.getConfig("BACKLINK_TEXT")
        )
        return text.replace(NBSP_PLACEHOLDER, "&#160;")


def makeExtension(**kwargs):  # pragma: no cover
    """ Return an instance of the `FootnoteExtension` """
    return FootnoteExtension(**kwargs)

```