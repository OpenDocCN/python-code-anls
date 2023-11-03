# PythonMarkdown源码解析 4

title: WikiLinks Extension

# WikiLinks

## Summary

The WikiLinks extension adds support for [WikiLinks][]. Specifically, any
``[[bracketed]]`` word is converted to a link.

This extension is included in the standard Markdown library.

[WikiLinks]: https://en.wikipedia.org/wiki/Wikilink

## Syntax

A ``[[bracketed]]`` word is any combination of  upper or lower case letters,
number, dashes, underscores and spaces surrounded by double brackets. Therefore

```pymd
[[Bracketed]]
```

would produce the following HTML:

```pyhtml
<a href="/Bracketed/" class="wikilink">Bracketed</a>
```

Note that WikiLinks are automatically assigned `class="wikilink"` making it
easy to style WikiLinks differently from other links on a page if one so
desires. See below for ways to alter the class.

Also note that when a space is used, the space is converted to an underscore in
the link but left as-is in the label. Perhaps an example would illustrate this
best:

```pymd
[[Wiki Link]]
```

becomes

```pyhtml
<a href="/Wiki_Link/" class="wikilink">Wiki Link</a>
```

## Usage

See [Extensions](index.md) for general extension usage. Use `wikilinks` as the
name of the extension.

See the [Library Reference](../reference.md#extensions) for information about
configuring extensions.

The default behavior is to point each link to the document root of the current
domain and close with a trailing slash. Additionally, each link is assigned to
the HTML class `wikilink`.

The following options are provided to change the default behavior:

* **`base_url`**: String to append to beginning of URL.

    Default: `'/'`

* **`end_url`**: String to append to end of URL.

    Default: `'/'`

* **`html_class`**: CSS class. Leave blank for none.

    Default: `'wikilink'`

* **`build_url`**: Callable which formats the URL from its parts.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['wikilinks'])
```

### Examples

For an example, let us suppose links should always point to the sub-directory
`/wiki/` and end with `.html`

```pypycon
>>> from markdown.extensions.wikilinks import WikiLinkExtension
>>> html = markdown.markdown(text,
...     extensions=[WikiLinkExtension(base_url='/wiki/', end_url='.html')]
... )
```

The above would result in the following link for `[[WikiLink]]`.

```pyhtml
<a href="/wiki/WikiLink.html" class="wikilink">WikiLink</a>
```

If you want to do more that just alter the base and/or end of the URL, you
could also pass in a callable which must accept three arguments (``label``,
``base``, and ``end``). The callable must return the URL in it's entirety.

```pypycon
>>> def my_url_builder(label, base, end):
...    # do stuff
...    return url
...
>>> html = markdown.markdown(text,
...     extensions=[WikiLinkExtension(build_url=my_url_builder)],
... )
```

The option is also provided to change or remove the class attribute.

```pypycon
>>> html = markdown.markdown(text,
...     extensions=[WikiLinkExtension(html_class='myclass')]
... )
```

Would cause all WikiLinks to be assigned to the class `myclass`.

```pyhtml
<a href="/WikiLink/" class="myclass">WikiLink</a>
```

## Using with Meta-Data extension

The WikiLink extension also supports the [Meta-Data](meta_data.md) extension.
Please see the documentation for that extension for specifics. The supported
meta-data keywords are:

* `wiki_base_url`
* `wiki_end_url`
* `wiki_html_class`

When used, the meta-data will override the settings provided through the
`extension_configs` interface.

This document:

```pymd
wiki_base_url: http://example.com/
wiki_end_url:  .html
wiki_html_class:

A [[WikiLink]] in the first paragraph.
```

would result in the following output (notice the blank `wiki_html_class`):

```pyhtml
<p>A <a href="http://example.com/WikiLink.html">WikiLink</a> in the first paragraph.</p>
```


# `/markdown/markdown/blockparser.py`

这是一个 Python 实现 John Gruber 的 Markdown 的代码。Markdown 是一种轻量级的标记语言，可以用来编写文档、readme、简历、培训材料等。

这个代码定义了一个 Python 类，实现了 Markdown 的一些常用功能，如
- 标题（h1, h2, h3, ...）
- 列表（ul, ol, li, ...）
- 链接（a, abbr, address, ...）
- 图片（img, src, alt, ...）
- 引用（blockquote, inline, ...）
- 代码块（code, vbnet, nmaq, ...）
- 列表（nested, ordered, custom, ...）
- 字典（dict, key, value, ...）
- 视频（video, source, controls, ...）
- 音频（audio, source, controls, ...）

此外，还支持在一些高级功能，如属性、事件、引用、代码块、表格等。这个实现使用了 Python 内置的一些库，如 `markdown` 和 `html`。


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

这段代码定义了一个名为 `BlockParser` 的类，该类处理 Markdown 文件的块（block）和列表（list）等基本文本文本元素。它不关注带有 `**bold**` 这样的内联元素，而是关注完整的块。

这个 `BlockParser` 类由若干个名为 `BlockProcessor` 的类组成，每个 `BlockProcessor` 负责处理一种不同的块类型。扩展时，可以增加、删除或修改 `BlockProcessor` 来改变 Markdown 块的解析方式。

总之，这段代码描述了一个简单的 Markdown 块解析器，它可以处理基本的文本元素，但不会对内联元素进行处理。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
The block parser handles basic parsing of Markdown blocks.  It doesn't concern
itself with inline elements such as `**bold**` or `*italics*`, but rather just
catches blocks, lists, quotes, etc.

The `BlockParser` is made up of a bunch of `BlockProcessors`, each handling a
different type of block. Extensions may add/replace/remove `BlockProcessors`
as they need to alter how Markdown blocks are parsed.
"""

from __future__ import annotations

```

这段代码是一个名为`State`的类，它用于跟踪`BlockParser`的状态。`BlockParser`是一个可扩展的类，用于在`Markdown`中处理块级格式化内容。

`State`类有两个方法：`set`和`reset`。`set`方法接受一个`state`参数，将其添加到当前状态列表的末尾。`reset`方法接受一个`state`参数，将其从当前状态列表的末尾删除。

这两个方法可以用来设置或重置状态，它们还用于确保每个状态都是唯一的，并在跟踪嵌套状态时保持正确性。

`State`类还有一个方法`isstate`，用于检查给定的状态是否是当前状态的前一级。这个方法在`BlockProcessor`中用于确保在将来自嵌套块的格式化内容转换为列表时，底层文本内容不会丢失。


```py
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Iterable, Any
from . import util

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown
    from .blockprocessors import BlockProcessor


class State(list):
    """ Track the current and nested state of the parser.

    This utility class is used to track the state of the `BlockParser` and
    support multiple levels if nesting. It's just a simple API wrapped around
    a list. Each time a state is set, that state is appended to the end of the
    list. Each time a state is reset, that state is removed from the end of
    the list.

    Therefore, each time a state is set for a nested block, that state must be
    reset when we back out of that level of nesting or the state could be
    corrupted.

    While all the methods of a list object are available, only the three
    defined below need be used.

    """

    def set(self, state: Any):
        """ Set a new state. """
        self.append(state)

    def reset(self) -> None:
        """ Step back one step in nested state. """
        self.pop()

    def isstate(self, state: Any) -> bool:
        """ Test that top (current) level is of given state. """
        if len(self):
            return self[-1] == state
        else:
            return False


```

I understand your concern. However, I'd like to clarify that the `parseChunk` and `parseBlocks` methods should only be called on the ` entire document` and not pieces, which means that the method should be called on the root element of the `etree.Element` object returned by the `parseChunk` method.

If you're looking to process individual lines of text in addition to the blocks, you can use a `lines` list to access those lines. Then, you should call the `parseChunk` method with the appropriate arguments. Here's an example:
markdown
lines = ['This is a line', 'This is another line']
doc = self.parseChunk(etree.Element(self.md.doc_tag), lines)

I apologize for any confusion. Please let me know if you have any more questions or need further clarification.


```py
class BlockParser:
    """ Parse Markdown blocks into an `ElementTree` object.

    A wrapper class that stitches the various `BlockProcessors` together,
    looping through them and creating an `ElementTree` object.

    """

    def __init__(self, md: Markdown):
        """ Initialize the block parser.

        Arguments:
            md: A Markdown instance.

        Attributes:
            BlockParser.md (Markdown): A Markdown instance.
            BlockParser.state (State): Tracks the nesting level of current location in document being parsed.
            BlockParser.blockprocessors (util.Registry): A collection of
                [`blockprocessors`][markdown.blockprocessors].

        """
        self.blockprocessors: util.Registry[BlockProcessor] = util.Registry()
        self.state = State()
        self.md = md

    def parseDocument(self, lines: Iterable[str]) -> etree.ElementTree:
        """ Parse a Markdown document into an `ElementTree`.

        Given a list of lines, an `ElementTree` object (not just a parent
        `Element`) is created and the root element is passed to the parser
        as the parent. The `ElementTree` object is returned.

        This should only be called on an entire document, not pieces.

        Arguments:
            lines: A list of lines (strings).

        Returns:
            An element tree.
        """
        # Create an `ElementTree` from the lines
        self.root = etree.Element(self.md.doc_tag)
        self.parseChunk(self.root, '\n'.join(lines))
        return etree.ElementTree(self.root)

    def parseChunk(self, parent: etree.Element, text: str) -> None:
        """ Parse a chunk of Markdown text and attach to given `etree` node.

        While the `text` argument is generally assumed to contain multiple
        blocks which will be split on blank lines, it could contain only one
        block. Generally, this method would be called by extensions when
        block parsing is required.

        The `parent` `etree` Element passed in is altered in place.
        Nothing is returned.

        Arguments:
            parent: The parent element.
            text: The text to parse.

        """
        self.parseBlocks(parent, text.split('\n\n'))

    def parseBlocks(self, parent: etree.Element, blocks: list[str]) -> None:
        """ Process blocks of Markdown text and attach to given `etree` node.

        Given a list of `blocks`, each `blockprocessor` is stepped through
        until there are no blocks left. While an extension could potentially
        call this method directly, it's generally expected to be used
        internally.

        This is a public method as an extension may need to add/alter
        additional `BlockProcessors` which call this method to recursively
        parse a nested block.

        Arguments:
            parent: The parent element.
            blocks: The blocks of text to parse.

        """
        while blocks:
            for processor in self.blockprocessors:
                if processor.test(parent, blocks[0]):
                    if processor.run(parent, blocks) is not False:
                        # run returns True or None
                        break

```

# `/markdown/markdown/blockprocessors.py`

该代码是一个Python实现了John Gruber的Markdown的实现。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的内容。该代码将Markdown格式的内容转换为HTML格式的内容，并将其存储在指定的位置。

该代码由Mannd小明和Yuri Takhteyev维护，当前由Waylan Limberg、Dmitry Shachnev和Isaac Muse共同维护。该代码使用了Python的Markdown库，该库支持Markdown的解析、生成和转换。


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

这段代码是一个名为 "block\_processor.py" 的 Python 模块，它是一个文本处理工具，主要用于处理 Git 仓库中的 Markdown 文件。该模块定义了一个 Blob 类型，用于表示解析后缀为 .md 或 .markdown 的文件中的文本块。模块中包含了一些静态方法，用于将解析到的文本块添加到 ElementTree 中，ElementTree 是一个 Git 用来表示文档对象的树形数据结构。

具体来说，block\_processor.py 模块的作用是处理以下类型的文本块：

1. 以空白行分隔的文本块，其中空白行不会以换行符为分隔符。
2. 包含代码块的文本，其中代码块可以是单行或多行的文本，以空白行分隔。

由于不同的文本块可能会有不同的语法结构，因此 block\_processor.py 模块会根据不同的文本块采用不同的处理方式。例如，对于单行代码块，会将其转换成一个节点添加到树中。对于多行代码块，则将其转换成一个子节点添加到树中，这个子节点会包含多行代码块以及代码块中的换行符。

block\_processor.py 模块还包含一些静态方法，例如 add_code()、add_math()、add_tables() 等，用于在解析到的文本块中添加代码、数学公式和表格。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
A block processor parses blocks of text and adds new elements to the ElementTree. Blocks of text,
separated from other text by blank lines, may have a different syntax and produce a differently
structured tree than other Markdown. Block processors excel at handling code formatting, equation
layouts, tables, etc.
"""

from __future__ import annotations

import logging
import re
```

这段代码是一个Python脚本，它实现了Markdown中的BlockParser类。BlockParser类是一个抽象类，它注册了一系列的块处理程序，用于处理Markdown文档中的不同类型的块（如标题、列表、链接等）。通过将BlockParser实例注册到BlockProcessor类中，可以定义自己的块类型和样式。

具体来说，这段代码实现了一个名为`build_block_parser`的函数，它接收一个Markdown渲染器（md）和一个或多个参数（**kwargs）作为参数，返回一个BlockParser实例。这个函数通过注册一系列的BlockProcessor实例来处理Markdown文档，然后将处理结果返回给调用者。注册的BlockProcessor类型包括：

- `EmptyBlockProcessor`：用于处理文档中的空块。
- `ListIndentProcessor`：用于处理列表的标题，根据其索引缩进相应的列表项。
- `CodeBlockProcessor`：用于处理Markdown代码块。
- `HashHeaderProcessor`：用于处理文档的标题，将其作为键值对存储，并在键为`[[AUTO]]`时自动生成标题。
- `SetextHeaderProcessor`：用于处理Markdown的标题，将其作为键值对存储。
- `HRProcessor`：用于处理文档的标题，在标题为`[HR]`时自动生成。
- `OListProcessor`：用于处理列表的渲染。
- `UListProcessor`：用于处理无序列表的渲染。
- `BlockQuoteProcessor`：用于处理Markdown中的引用。
- `ReferenceProcessor`：用于处理Markdown中的链接。
- `ParagraphProcessor`：用于处理文档的段落。

通过调用这个函数，可以将Markdown渲染器实例与BlockProcessor实例组合起来，用于处理Markdown文档的块。例如：
python
from markdown import Markdown

def render_markdown(markdown: str) -> str:
   """ Render the specified Markdown into a string. """
   return build_block_parser(markdown)

markdown = """
# 标题1
## 标题2
### 标题3

[链接1]
[链接2]
[链接3]

[图片]

HR
"""

rendered_markdown = render_markdown(markdown)

这段代码将以上Markdown内容渲染为字符串，并使用了BlockParser实例来处理文档中的块。


```py
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown

logger = logging.getLogger('MARKDOWN')


def build_block_parser(md: Markdown, **kwargs: Any) -> BlockParser:
    """ Build the default block parser used by Markdown. """
    parser = BlockParser(md)
    parser.blockprocessors.register(EmptyBlockProcessor(parser), 'empty', 100)
    parser.blockprocessors.register(ListIndentProcessor(parser), 'indent', 90)
    parser.blockprocessors.register(CodeBlockProcessor(parser), 'code', 80)
    parser.blockprocessors.register(HashHeaderProcessor(parser), 'hashheader', 70)
    parser.blockprocessors.register(SetextHeaderProcessor(parser), 'setextheader', 60)
    parser.blockprocessors.register(HRProcessor(parser), 'hr', 50)
    parser.blockprocessors.register(OListProcessor(parser), 'olist', 40)
    parser.blockprocessors.register(UListProcessor(parser), 'ulist', 30)
    parser.blockprocessors.register(BlockQuoteProcessor(parser), 'quote', 20)
    parser.blockprocessors.register(ReferenceProcessor(parser), 'reference', 15)
    parser.blockprocessors.register(ParagraphProcessor(parser), 'paragraph', 10)
    return parser


```

Based on the provided documentation, it appears that the `run` method of the `BlockProcessor` class should be overridden by subclasses. The `run` method is expected to parse the individual lines of the block and append them to the `etree`.

Subclasses of `BlockProcessor` should be able to provide their own implementation of the `run` method, which can include modifications to the parent `etree` element and adding/removing blocks from the input.

It is important to note that the `run` method should not return `False` as this would indicate that the block is not present in the document, which is not the responsibility of the processor.

If you have an implementation of the `run` method for a specific block type, you should be able to use it in the `test` method to determine if the block is of that type. However, this implementation should be able to handle the case where the block type is different depending on the parent element, and should provide the correct behavior for that case.


```py
class BlockProcessor:
    """ Base class for block processors.

    Each subclass will provide the methods below to work with the source and
    tree. Each processor will need to define it's own `test` and `run`
    methods. The `test` method should return True or False, to indicate
    whether the current block should be processed by this processor. If the
    test passes, the parser will call the processors `run` method.

    Attributes:
        BlockProcessor.parser (BlockParser): The `BlockParser` instance this is attached to.
        BlockProcessor.tab_length (int): The tab length set on the `Markdown` instance.

    """

    def __init__(self, parser: BlockParser):
        self.parser = parser
        self.tab_length = parser.md.tab_length

    def lastChild(self, parent: etree.Element) -> etree.Element | None:
        """ Return the last child of an `etree` element. """
        if len(parent):
            return parent[-1]
        else:
            return None

    def detab(self, text: str, length: int | None = None) -> tuple[str, str]:
        """ Remove a tab from the front of each line of the given text. """
        if length is None:
            length = self.tab_length
        newtext = []
        lines = text.split('\n')
        for line in lines:
            if line.startswith(' ' * length):
                newtext.append(line[length:])
            elif not line.strip():
                newtext.append('')
            else:
                break
        return '\n'.join(newtext), '\n'.join(lines[len(newtext):])

    def looseDetab(self, text: str, level: int = 1) -> str:
        """ Remove a tab from front of lines but allowing dedented lines. """
        lines = text.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith(' '*self.tab_length*level):
                lines[i] = lines[i][self.tab_length*level:]
        return '\n'.join(lines)

    def test(self, parent: etree.Element, block: str) -> bool:
        """ Test for block type. Must be overridden by subclasses.

        As the parser loops through processors, it will call the `test`
        method on each to determine if the given block of text is of that
        type. This method must return a boolean `True` or `False`. The
        actual method of testing is left to the needs of that particular
        block type. It could be as simple as `block.startswith(some_string)`
        or a complex regular expression. As the block type may be different
        depending on the parent of the block (i.e. inside a list), the parent
        `etree` element is also provided and may be used as part of the test.

        Keyword arguments:
            parent: An `etree` element which will be the parent of the block.
            block: A block of text from the source which has been split at blank lines.
        """
        pass  # pragma: no cover

    def run(self, parent: etree.Element, blocks: list[str]) -> bool | None:
        """ Run processor. Must be overridden by subclasses.

        When the parser determines the appropriate type of a block, the parser
        will call the corresponding processor's `run` method. This method
        should parse the individual lines of the block and append them to
        the `etree`.

        Note that both the `parent` and `etree` keywords are pointers
        to instances of the objects which should be edited in place. Each
        processor must make changes to the existing objects as there is no
        mechanism to return new/different objects to replace them.

        This means that this method should be adding `SubElements` or adding text
        to the parent, and should remove (`pop`) or add (`insert`) items to
        the list of blocks.

        If `False` is returned, this will have the same effect as returning `False`
        from the `test` method.

        Keyword arguments:
            parent: An `etree` element which is the parent of the current block.
            blocks: A list of all remaining blocks of the document.
        """
        pass  # pragma: no cover


```

This is a `xml.etree.Element` method that parses an XML document. It does so by following a nested structure defined by the `text` attribute of each `li` element.

It starts by creating a new `li` element and setting its text to the `text` attribute of the last child of the `li` element. Then it resets the `state` attribute of the parser to `reset()` to parse a new document.

It then starts parsing the block by calling the `parseChunk()` method of the `parser.`

It has another method `create_item()` which creates a new `li` element and parses the block by calling the `parseBlocks()` method of the parser.

It also has a method `get_level()` which gets the indentation level of a `li` element based on the level of the `text` attribute.


```py
class ListIndentProcessor(BlockProcessor):
    """ Process children of list items.

    Example

        * a list item
            process this part

            or this part

    """

    ITEM_TYPES = ['li']
    """ List of tags used for list items. """
    LIST_TYPES = ['ul', 'ol']
    """ Types of lists this processor can operate on. """

    def __init__(self, *args):
        super().__init__(*args)
        self.INDENT_RE = re.compile(r'^(([ ]{%s})+)' % self.tab_length)

    def test(self, parent: etree.Element, block: str) -> bool:
        return block.startswith(' '*self.tab_length) and \
            not self.parser.state.isstate('detabbed') and \
            (parent.tag in self.ITEM_TYPES or
                (len(parent) and parent[-1] is not None and
                    (parent[-1].tag in self.LIST_TYPES)))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        level, sibling = self.get_level(parent, block)
        block = self.looseDetab(block, level)

        self.parser.state.set('detabbed')
        if parent.tag in self.ITEM_TYPES:
            # It's possible that this parent has a `ul` or `ol` child list
            # with a member.  If that is the case, then that should be the
            # parent.  This is intended to catch the edge case of an indented
            # list whose first member was parsed previous to this point
            # see `OListProcessor`
            if len(parent) and parent[-1].tag in self.LIST_TYPES:
                self.parser.parseBlocks(parent[-1], [block])
            else:
                # The parent is already a `li`. Just parse the child block.
                self.parser.parseBlocks(parent, [block])
        elif sibling.tag in self.ITEM_TYPES:
            # The sibling is a `li`. Use it as parent.
            self.parser.parseBlocks(sibling, [block])
        elif len(sibling) and sibling[-1].tag in self.ITEM_TYPES:
            # The parent is a list (`ol` or `ul`) which has children.
            # Assume the last child `li` is the parent of this block.
            if sibling[-1].text:
                # If the parent `li` has text, that text needs to be moved to a `p`
                # The `p` must be 'inserted' at beginning of list in the event
                # that other children already exist i.e.; a nested sub-list.
                p = etree.Element('p')
                p.text = sibling[-1].text
                sibling[-1].text = ''
                sibling[-1].insert(0, p)
            self.parser.parseChunk(sibling[-1], block)
        else:
            self.create_item(sibling, block)
        self.parser.state.reset()

    def create_item(self, parent: etree.Element, block: str) -> None:
        """ Create a new `li` and parse the block with it as the parent. """
        li = etree.SubElement(parent, 'li')
        self.parser.parseBlocks(li, [block])

    def get_level(self, parent: etree.Element, block: str) -> tuple[int, etree.Element]:
        """ Get level of indentation based on list level. """
        # Get indent level
        m = self.INDENT_RE.match(block)
        if m:
            indent_level = len(m.group(1))/self.tab_length
        else:
            indent_level = 0
        if self.parser.state.isstate('list'):
            # We're in a tight-list - so we already are at correct parent.
            level = 1
        else:
            # We're in a loose-list - so we need to find parent.
            level = 0
        # Step through children of tree to find matching indent level.
        while indent_level > level:
            child = self.lastChild(parent)
            if (child is not None and
               (child.tag in self.LIST_TYPES or child.tag in self.ITEM_TYPES)):
                if child.tag in self.LIST_TYPES:
                    level += 1
                parent = child
            else:
                # No more child levels. If we're short of `indent_level`,
                # we have a code block. So we stop here.
                break
        return level, parent


```

这段代码定义了一个名为 CodeBlockProcessor 的类，继承自 BlockProcessor 类（可能指某个库的 BlockProcessor 类）。这个类的主要目的是处理代码块，即在给定的 parent 元素或文本中提取出的代码块进行处理。

CodeBlockProcessor 类包含一个 test 方法和一个 run 方法。

- test 方法接受两个参数：一个父元素（etree.Element）和一个代码块（str）。这个方法判断给定的代码块是否以空格字符作为开头，如果是，则返回 True，否则返回 False。

- run 方法接受两个参数：一个父元素（etree.Element）和一个代码块列表（list[str]）。这个方法将给定的代码块列表中的第一个代码块处理，并在需要时将整行的缩进处理掉。代码块处理后的结果将添加到父元素中的 theRest 子元素中，如果 theRest 子元素为空，则会将 theRest 子元素（如果有）插入到代码块列表的开头。

该类的方法详细描述了如何处理代码块，包括：

1. 以空白行作为代码块的起始行时，如何处理。
2. 处理代码块中的缩进，如果代码块的起始行包含缩进，则将缩进转换为引人注目的格式。
3. 如果处理的代码块是在一个新代码块的内部，该代码块将包含前一个代码块中的所有行，并将它们添加到新代码块中。
4. 如果新代码块的起始行包含一个或多个缩进，则在新代码块中插入这些缩进。
5. 如果 theRest 子元素是空的，则在代码块列表的开头插入它。


```py
class CodeBlockProcessor(BlockProcessor):
    """ Process code blocks. """

    def test(self, parent: etree.Element, block: str) -> bool:
        return block.startswith(' '*self.tab_length)

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        sibling = self.lastChild(parent)
        block = blocks.pop(0)
        theRest = ''
        if (sibling is not None and sibling.tag == "pre" and
           len(sibling) and sibling[0].tag == "code"):
            # The previous block was a code block. As blank lines do not start
            # new code blocks, append this block to the previous, adding back
            # line breaks removed from the split into a list.
            code = sibling[0]
            block, theRest = self.detab(block)
            code.text = util.AtomicString(
                '{}\n{}\n'.format(code.text, util.code_escape(block.rstrip()))
            )
        else:
            # This is a new code block. Create the elements and insert text.
            pre = etree.SubElement(parent, 'pre')
            code = etree.SubElement(pre, 'code')
            block, theRest = self.detab(block)
            code.text = util.AtomicString('%s\n' % util.code_escape(block.rstrip()))
        if theRest:
            # This block contained unindented line(s) after the first indented
            # line. Insert these lines as the first block of the master blocks
            # list for future processing.
            blocks.insert(0, theRest)


```

这段代码定义了一个名为 BlockQuoteProcessor 的类，继承自 BlockProcessor 类（可能需要从已有的 BlockProcessor 类定义来推断）。这个类的主要目的是处理块引用（blockquotes）。

在 BlockQuoteProcessor 类中，定义了一个名为 test 的方法，用于对传入的父元素（parent）和块引用（block）进行测试，返回测试结果为 True 时表示处理成功，否则表示处理失败。

定义了一个名为 run 的方法，用于递归地处理传入的父元素（parent）和块引用（blocks）列表。当递归调用这个方法时，会处理到列表的第一个元素。方法内部首先找到要处理的块引用（block），然后执行以下操作：

1. 如果找到块引用（block）后，使用正则表达式 RE 进行匹配，将匹配到的内容（包括换行符）从块引用（block）中删除。
2. 根据正则表达式 RE 的结果，删除匹配到的内容后，使用 split('\n') 方法将内容（不包括换行符）分割成多个列表。
3. 对每个分割出的内容列表，调用 clean 方法进行去除非空格（'\'）并去除换行符（'\n'）。
4. 如果找到一个与块引用（block）相同的兄弟元素（sibling），并且这个兄弟元素的标签（tag）为 "blockquote"，那么将兄弟元素（sibling）设置为这个块引用（block）作为新父元素（new parent）。
5. 如果这个方法递归调用后，发现自己处于一个新的块引用（block）中，那么将当前的父元素（parent）设置为一个新的父元素（new parent），并使用 parser 对象的 state（state）设置为 'blockquote'。
6. 使用 parser 对象的 state（state）设置为 reset，然后使用 parser 对象的 parseChunk 方法（可能需要从已有的 BlockProcessor 类定义来推断）对当前的块引用（block）进行解析。
7. 递归调用 run 方法，处理下一个块引用（block）。

最后，定义了一个名为 clean 的方法，用于从块引用（block）中删除 >。这个方法首先使用正则表达式 RE 找到匹配的块引用（block），然后去除非空格（'\'）。


```py
class BlockQuoteProcessor(BlockProcessor):
    """ Process blockquotes. """

    RE = re.compile(r'(^|\n)[ ]{0,3}>[ ]?(.*)')

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.search(block)) and not util.nearing_recursion_limit()

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            before = block[:m.start()]  # Lines before blockquote
            # Pass lines before blockquote in recursively for parsing first.
            self.parser.parseBlocks(parent, [before])
            # Remove `> ` from beginning of each line.
            block = '\n'.join(
                [self.clean(line) for line in block[m.start():].split('\n')]
            )
        sibling = self.lastChild(parent)
        if sibling is not None and sibling.tag == "blockquote":
            # Previous block was a blockquote so set that as this blocks parent
            quote = sibling
        else:
            # This is a new blockquote. Create a new parent element.
            quote = etree.SubElement(parent, 'blockquote')
        # Recursively parse block with blockquote as parent.
        # change parser state so blockquotes embedded in lists use `p` tags
        self.parser.state.set('blockquote')
        self.parser.parseChunk(quote, block)
        self.parser.state.reset()

    def clean(self, line: str) -> str:
        """ Remove `>` from beginning of a line. """
        m = self.RE.match(line)
        if line.strip() == ">":
            return ""
        elif m:
            return m.group(2)
        else:
            return line


```

This appears to be an XML parser that can parse an HTML block. It has a specific behavior for what it can parse, but it also has some general-purpose functionality like breaking down blocks and items.

The `parse_block` method takes an HTML block as input and returns a list of items. It works by parsing the block using the `parseBlocks` method, which recursively handles nested blocks.

The `parseBlocks` method takes two arguments: the first is the list of items that were matched by the block, and the second is the list that will be used to store the parsed items. It uses regular expressions to match the block and its child elements, and appends the matched elements to the second list.

The `indent` method is used to indent the block, which is helpful when it's being passed to the `parseBlocks` method. It takes a string that represents the number of spaces that should be added to the start of each line, and returns a new list that has been indented by that amount.

The `startswith` method is used to check if an element starts with a given string. It takes the string and returns a boolean value indicating whether the string starts with the first character of the element.

Overall, this parser appears to be a simple tool for parsing and processing HTML blocks, and it has some limited functionality for breaking down blocks and items.


```py
class OListProcessor(BlockProcessor):
    """ Process ordered list blocks. """

    TAG: str = 'ol'
    """ The tag used for the the wrapping element. """
    STARTSWITH: str = '1'
    """
    The integer (as a string ) with which the list starts. For example, if a list is initialized as
    `3. Item`, then the `ol` tag will be assigned an HTML attribute of `starts="3"`. Default: `"1"`.
    """
    LAZY_OL: bool = True
    """ Ignore `STARTSWITH` if `True`. """
    SIBLING_TAGS: list[str] = ['ol', 'ul']
    """
    Markdown does not require the type of a new list item match the previous list item type.
    This is the list of types which can be mixed.
    """

    def __init__(self, parser: BlockParser):
        super().__init__(parser)
        # Detect an item (`1. item`). `group(1)` contains contents of item.
        self.RE = re.compile(r'^[ ]{0,%d}\d+\.[ ]+(.*)' % (self.tab_length - 1))
        # Detect items on secondary lines. they can be of either list type.
        self.CHILD_RE = re.compile(r'^[ ]{0,%d}((\d+\.)|[*+-])[ ]+(.*)' %
                                   (self.tab_length - 1))
        # Detect indented (nested) items of either type
        self.INDENT_RE = re.compile(r'^[ ]{%d,%d}((\d+\.)|[*+-])[ ]+.*' %
                                    (self.tab_length, self.tab_length * 2 - 1))

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.match(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        # Check for multiple items in one block.
        items = self.get_items(blocks.pop(0))
        sibling = self.lastChild(parent)

        if sibling is not None and sibling.tag in self.SIBLING_TAGS:
            # Previous block was a list item, so set that as parent
            lst = sibling
            # make sure previous item is in a `p` - if the item has text,
            # then it isn't in a `p`
            if lst[-1].text:
                # since it's possible there are other children for this
                # sibling, we can't just `SubElement` the `p`, we need to
                # insert it as the first item.
                p = etree.Element('p')
                p.text = lst[-1].text
                lst[-1].text = ''
                lst[-1].insert(0, p)
            # if the last item has a tail, then the tail needs to be put in a `p`
            # likely only when a header is not followed by a blank line
            lch = self.lastChild(lst[-1])
            if lch is not None and lch.tail:
                p = etree.SubElement(lst[-1], 'p')
                p.text = lch.tail.lstrip()
                lch.tail = ''

            # parse first block differently as it gets wrapped in a `p`.
            li = etree.SubElement(lst, 'li')
            self.parser.state.set('looselist')
            firstitem = items.pop(0)
            self.parser.parseBlocks(li, [firstitem])
            self.parser.state.reset()
        elif parent.tag in ['ol', 'ul']:
            # this catches the edge case of a multi-item indented list whose
            # first item is in a blank parent-list item:
            #     * * subitem1
            #         * subitem2
            # see also `ListIndentProcessor`
            lst = parent
        else:
            # This is a new list so create parent with appropriate tag.
            lst = etree.SubElement(parent, self.TAG)
            # Check if a custom start integer is set
            if not self.LAZY_OL and self.STARTSWITH != '1':
                lst.attrib['start'] = self.STARTSWITH

        self.parser.state.set('list')
        # Loop through items in block, recursively parsing each with the
        # appropriate parent.
        for item in items:
            if item.startswith(' '*self.tab_length):
                # Item is indented. Parse with last item as parent
                self.parser.parseBlocks(lst[-1], [item])
            else:
                # New item. Create `li` and parse with it as parent
                li = etree.SubElement(lst, 'li')
                self.parser.parseBlocks(li, [item])
        self.parser.state.reset()

    def get_items(self, block: str) -> list[str]:
        """ Break a block into list items. """
        items = []
        for line in block.split('\n'):
            m = self.CHILD_RE.match(line)
            if m:
                # This is a new list item
                # Check first item for the start index
                if not items and self.TAG == 'ol':
                    # Detect the integer value of first list item
                    INTEGER_RE = re.compile(r'(\d+)')
                    self.STARTSWITH = INTEGER_RE.match(m.group(1)).group()
                # Append to the list
                items.append(m.group(3))
            elif self.INDENT_RE.match(line):
                # This is an indented (possibly nested) item.
                if items[-1].startswith(' '*self.tab_length):
                    # Previous item was indented. Append to that item.
                    items[-1] = '{}\n{}'.format(items[-1], line)
                else:
                    items.append(line)
            else:
                # This is another line of previous item. Append to that item.
                items[-1] = '{}\n{}'.format(items[-1], line)
        return items


```

This is a Python class that defines a `HashHeaderProcessor` implementation for the治疗 `xml.睾丸处理器`. This class inherits from the `BlockProcessor` class and contains methods for processing `HashHeader`.

The `HashHeaderProcessor` class takes a `BlockParser` object and uses its `parseBlocks` method to process blocks. The `parseBlocks` method is used to recursively parse each line in the block and inserts the parsed blocks into the main processing tree.

The `HashHeaderProcessor` class contains a `test` method that Detects a header at the start of any line in the block. If the header is found, the `run` method is called. The `run` method uses the regular expression pattern for a header and the `test` method to check if the header is found in the block. If the header is found, the method extracts the header text from the block and inserts it into the header element. If the header is not found, the method logs a warning.

In the `run` method, the header is first detected as the first non-whitespace character in the block. If the header is found, the remaining lines are processed as a new block and the header is inserted into the block as an element.


```py
class UListProcessor(OListProcessor):
    """ Process unordered list blocks. """

    TAG: str = 'ul'
    """ The tag used for the the wrapping element. """

    def __init__(self, parser: BlockParser):
        super().__init__(parser)
        # Detect an item (`1. item`). `group(1)` contains contents of item.
        self.RE = re.compile(r'^[ ]{0,%d}[*+-][ ]+(.*)' % (self.tab_length - 1))


class HashHeaderProcessor(BlockProcessor):
    """ Process Hash Headers. """

    # Detect a header at start of any line in block
    RE = re.compile(r'(?:^|\n)(?P<level>#{1,6})(?P<header>(?:\\.|[^\\])*?)#*(?:\n|$)')

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.search(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            before = block[:m.start()]  # All lines before header
            after = block[m.end():]     # All lines after header
            if before:
                # As the header was not the first line of the block and the
                # lines before the header must be parsed first,
                # recursively parse this lines as a block.
                self.parser.parseBlocks(parent, [before])
            # Create header using named groups from RE
            h = etree.SubElement(parent, 'h%d' % len(m.group('level')))
            h.text = m.group('header').strip()
            if after:
                # Insert remaining lines as first block for future parsing.
                blocks.insert(0, after)
        else:  # pragma: no cover
            # This should never happen, but just in case...
            logger.warn("We've got a problem header: %r" % block)


```

这段代码定义了一个名为 `SetextHeaderProcessor` 的类，该类继承自 `BlockProcessor` 类，用于处理 Setext-style 格式的数据。

具体来说，这个类包含一个名为 `test` 的方法，用于测试某个 `BlockProcessor` 类中的数据是否符合 Setext-style 格式，如果符合，则返回 `True`，否则返回 `False`。

接着，有一个名为 `run` 的方法，用于处理多个 `BlockProcessor` 类的数据。这个方法接收两个参数：一个是包含多个 Setext-style 数据块的元素，另一个是一个字符串，表示要处理的数据。

如果第一个参数中的数据包含 Setext-style 数据，则将其提取出来，并添加到第二个参数中，以供后续处理。如果第一个参数中的数据包含多行，则将多余的行添加到第二个参数中。

最后，这个 `SetextHeaderProcessor` 类还有一个 `__init__` 方法，用于初始化 Setext-style 头信息，包括可能的模式和括号。


```py
class SetextHeaderProcessor(BlockProcessor):
    """ Process Setext-style Headers. """

    # Detect Setext-style header. Must be first 2 lines of block.
    RE = re.compile(r'^.*?\n[=-]+[ ]*(\n|$)', re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.match(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        lines = blocks.pop(0).split('\n')
        # Determine level. `=` is 1 and `-` is 2.
        if lines[1].startswith('='):
            level = 1
        else:
            level = 2
        h = etree.SubElement(parent, 'h%d' % level)
        h.text = lines[0].strip()
        if len(lines) > 2:
            # Block contains additional lines. Add to  master blocks for later.
            blocks.insert(0, '\n'.join(lines[2:]))


```

这段代码定义了一个名为HRProcessor的类，继承自BlockProcessor类（可能需要替换为BlockFilterProcessor）。这个类的主要目的是处理水平规则，使其适应Python的re模块。

具体来说，这个类包含以下方法：

1. `test`：用于检测包含水平的文本块。如果找到匹配的文本，将 match 对象保存，并返回True；否则返回False。
2. `run`：用于解析包含多个水平规则的文本块。首先检查在块之前是否包含 lines，如果包含，则递归处理这些行。然后创建一个hr元素，并将 HR 添加到父元素中。接下来，处理块之后的 lines。如果包含 lines，将它们添加到主块中；否则，只处理包含 HR 的行。

这个类使用了以下字符串正则表达式来找到水平规则：
less
^[ ]{0,3}(?=(?P<atomicgroup>-+[ ]{0,2}){3,}|(_+[ ]{0,2}){3,}|(\*+[ ]{0,2}){3,}))(?P=atomicgroup)[ ]*$

这个正则表达式用于在块中查找水平的、以 `-+` 开头的、具有 `0` 到 `2` 个数字的子元素，并将其保存。


```py
class HRProcessor(BlockProcessor):
    """ Process Horizontal Rules. """

    # Python's `re` module doesn't officially support atomic grouping. However you can fake it.
    # See https://stackoverflow.com/a/13577411/866026
    RE = r'^[ ]{0,3}(?=(?P<atomicgroup>(-+[ ]{0,2}){3,}|(_+[ ]{0,2}){3,}|(\*+[ ]{0,2}){3,}))(?P=atomicgroup)[ ]*$'
    # Detect hr on any line of a block.
    SEARCH_RE = re.compile(RE, re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        m = self.SEARCH_RE.search(block)
        if m:
            # Save match object on class instance so we can use it later.
            self.match = m
            return True
        return False

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        match = self.match
        # Check for lines in block before `hr`.
        prelines = block[:match.start()].rstrip('\n')
        if prelines:
            # Recursively parse lines before `hr` so they get parsed first.
            self.parser.parseBlocks(parent, [prelines])
        # create hr
        etree.SubElement(parent, 'hr')
        # check for lines in block after `hr`.
        postlines = block[match.end():].lstrip('\n')
        if postlines:
            # Add lines after `hr` to master blocks for later parsing.
            blocks.insert(0, postlines)


```

这段代码定义了一个名为 `EmptyBlockProcessor` 的类，其父类是 `BlockProcessor`。这个类的主要目的是处理那些空格或者以空行开始的空腔块。

该类包含一个测试方法 `test`，用于检查给定的空腔块是否为空或者是否以空行开始。如果空腔块为空或者以空行开始，则返回 `True`，否则返回 `False`。

该类包含一个 `run` 方法，用于处理给定的空腔块列表。首先将空腔块列表中的第一个元素保存到一个变量 `block` 中。然后，构造一个包含一系列空行字符的字符串 `filler`，并将它设置为空腔块列表中的第二个元素。接下来，遍历空腔块列表中的所有元素。如果空腔块以空行开始，将剩余的行添加到已经构建好的主块中。在主块中，如果是最后一块，将会将其内容更改为包含空行和 `filler` 的字符串，以保留 whitespace。

在该类的实例中，如果给定一个空腔块列表，则运行 `run` 方法将用 `filler` 填充空腔块列表中的所有空行，并将已经构建好的主块添加到空腔块列表中。


```py
class EmptyBlockProcessor(BlockProcessor):
    """ Process blocks that are empty or start with an empty line. """

    def test(self, parent: etree.Element, block: str) -> bool:
        return not block or block.startswith('\n')

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        filler = '\n\n'
        if block:
            # Starts with empty line
            # Only replace a single line.
            filler = '\n'
            # Save the rest for later.
            theRest = block[1:]
            if theRest:
                # Add remaining lines to master blocks for later.
                blocks.insert(0, theRest)
        sibling = self.lastChild(parent)
        if (sibling is not None and sibling.tag == 'pre' and
           len(sibling) and sibling[0].tag == 'code'):
            # Last block is a code block. Append to preserve whitespace.
            sibling[0].text = util.AtomicString(
                '{}{}'.format(sibling[0].text, filler)
            )


```

这段代码是一个名为 "ReferenceProcessor" 的类，它是 BlockProcessor 类的子类。这个类的目的是处理链接引用，即在文档中有链接出现时，将其链接的引用关系处理下来。

具体来说，这个类包含了一个正则表达式模块 (regex) RE，用于匹配链接引用中的文本。正则表达式中的 RE 部分定义了链接引用的格式，包括开始和结束标签、属性、空格等。通过这个正则表达式，可以匹配大部分的链接引用。

在 test() 方法中，对传入的块 (block) 和其内容进行处理。首先，使用正则表达式找到块中的链接引用，然后检查是否已经处理过链接。如果已经处理过链接，则检查块中的内容是否在链接前和链接后匹配。如果匹配成功，则保存链接和标题，并删除链接前后的内容。如果匹配失败，则不处理块，并将其移动到块的起始位置。

在 run() 方法中，对传入的块 (block) 和其内容进行处理。首先，从块中删除链接，并查找链接的引用关系。如果找到了链接，则根据链接内容添加到已处理的块中。如果链接内容为空或者只包含一个空格，则将其移动到块的起始位置。如果链接内容不包含任何有效的链接，则将其保存到已处理的块中。

最后，如果块中的内容匹配链接，则返回 True，否则返回 False。


```py
class ReferenceProcessor(BlockProcessor):
    """ Process link references. """
    RE = re.compile(
        r'^[ ]{0,3}\[([^\[\]]*)\]:[ ]*\n?[ ]*([^\s]+)[ ]*(?:\n[ ]*)?((["\'])(.*)\4[ ]*|\((.*)\)[ ]*)?$', re.MULTILINE
    )

    def test(self, parent: etree.Element, block: str) -> bool:
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            id = m.group(1).strip().lower()
            link = m.group(2).lstrip('<').rstrip('>')
            title = m.group(5) or m.group(6)
            self.parser.md.references[id] = (link, title)
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


```

这段代码定义了一个名为 ParagraphProcessor 的类，继承自 BlockProcessor 类（可能是一个继承自某个数据结构）。

ParagraphProcessor 的作用是对给定的 Paragraph 文本进行处理，主要包括以下几个方法：

1. test(self, parent: etree.Element, block: str) -> bool：这个方法用于验证 block 是否可以作为一个 Paragraph 运行。如果 block 是空字符串，那么返回 True，否则返回 False。
2. run(self, parent: etree.Element, blocks: list[str]) -> None：这个方法对给定的 Paragraph 元素和处理 blocks 中的每个 Paragraph 文本。首先，它从 blocks 列表中取出第一个元素，然后执行 block 中提供的操作。如果 block 中包含一个空字符串，那么将其追加到 Paragraph 元素中。如果 block 中包含多个字符，那么将其转换为 Paragraph 元素并添加到 Paragraph 元素中。

总的来说，这段代码定义了一个 Paragraph 处理工具类，用于处理 Paragraph 中的文本内容。


```py
class ParagraphProcessor(BlockProcessor):
    """ Process Paragraph blocks. """

    def test(self, parent: etree.Element, block: str) -> bool:
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        if block.strip():
            # Not a blank block. Add to parent, otherwise throw it away.
            if self.parser.state.isstate('list'):
                # The parent is a tight-list.
                #
                # Check for any children. This will likely only happen in a
                # tight-list when a header isn't followed by a blank line.
                # For example:
                #
                #     * # Header
                #     Line 2 of list item - not part of header.
                sibling = self.lastChild(parent)
                if sibling is not None:
                    # Insert after sibling.
                    if sibling.tail:
                        sibling.tail = '{}\n{}'.format(sibling.tail, block)
                    else:
                        sibling.tail = '\n%s' % block
                else:
                    # Append to parent.text
                    if parent.text:
                        parent.text = '{}\n{}'.format(parent.text, block)
                    else:
                        parent.text = block.lstrip()
            else:
                # Create a regular paragraph
                p = etree.SubElement(parent, 'p')
                p.text = block.lstrip()

```

# `/markdown/markdown/core.py`

该代码是一个Python实现John Gruber所写的Markdown的示例。Markdown是一种轻量级的标记语言，它可以让你快速地编写出文档性的内容。

该代码包括以下几个部分：

1. 函数def普通MarkdownToHTML(content):

<!doctype html>
<html>
 <head>
   <meta charset="utf-8">
   <title>Python Markdown</title>
 </head>
 <body>
   <pre>{{content}}</pre>
 </body>
</html>


这个函数的作用是将Markdown内容转换为HTML格式的内容。在转换过程中，函数会将特殊标记如标题、列表和链接转换为对应的HTML标记。

2. 函数defMarkdownToLink(content, link):

<!doctype html>
<html>
 <head>
   <meta charset="utf-8">
   <title>Python Markdown</title>
 </head>
 <body>
   <ul>
     {% for link in linker.links %}
       <li><a href="{{ link.href }}" target="_blank">{{ link.text }}</a></li>
     {% endfor %}
   </ul>
   <pre>{{ content }}</pre>
 </body>
</html>


这个函数的作用是将Markdown内容转换为包含链接的HTML内容。在转换过程中，函数会遍历Markdown中的链接，并将其转换为包含链接的HTML列表项。

3. 定义了两个链接er类MarkdownLinker和Linker，它们的作用将Markdown链接转换为相应的HTML标记。

4. 在项目的描述部分提供了项目的版本信息、作者和维护者等信息。

5. 在最后部分定义了该项目的版权信息。


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

这段代码是一个Python程序，它将不同类型的文本数据文件（如.txt，.xml，.json等）转化为对应的预处理和处理函数，以方便后续的文本分析工作。具体来说，它实现了以下功能：

1. 从sys.argv列表中提取要转换的文件名参数。
2. 读取文件内容，并将其存储在两个Mapping对象中：分别是Tuple[str, BinaryIO]和Tuple[str, TextIO]。
3. 对文件内容进行清洗和处理，包括去除标点符号、大小写转换、删除停用词等。具体实现依赖于要转换的文件类型。
4. 将清洗后的内容存储到两个类Var对象中：分别是TEXT_CONTENTS和TAGGED_CONTENTS。
5. 如果需要，将清洗后的内容写入新的文件。具体实现依赖于要转换的文件类型和文件路径。
6. 导入必要的类型注释，以便在需要时自动进行类型检查。
7. 使用annotations库来提供类型信息，以便在需要时自动进行类型检查。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

from __future__ import annotations

import codecs
import sys
import logging
import importlib
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, ClassVar, Mapping, Sequence
from . import util
from .preprocessors import build_preprocessors
from .blockprocessors import build_block_parser
from .treeprocessors import build_treeprocessors
```

这段代码是一个Python应用程序，包含以下几个主要模块：

1. `build_inlinepatterns` 函数：这个函数从`inlinepatterns`模块中学习到了一些可以用来在Markdown中快速转换普通文本 pattern的规则，然后编译成了一个可以快速转换Markdown文本的中间表示形式，最后返回到Python应用中。

2. `build_postprocessors` 函数：这个函数从`postprocessors`模块中学习到了一些可以用来在Markdown中处理数据的规则，然后编译成了一个可以快速处理Markdown数据的中间表示形式，最后返回到Python应用中。

3. `Extension` 类：这个类实现了`Extension`接口，可能是一个可以让你在Markdown中使用自定义滤波器、转换器等功能的扩展。

4. `to_html_string` 和 `to_xhtml_string` 函数：这两个函数可以将Markdown内容转换为HTML字符串。`to_html_string`使用了Markdown的原始语法，而`to_xhtml_string`使用了XHTML的语法。

5. `BLOCK_LEVEL_ELEMENTS` 变量：这个变量定义了一个枚举类型，用于指定Markdown中的块级元素可以使用哪些级别的元素包裹。


```py
from .inlinepatterns import build_inlinepatterns
from .postprocessors import build_postprocessors
from .extensions import Extension
from .serializers import to_html_string, to_xhtml_string
from .util import BLOCK_LEVEL_ELEMENTS

if TYPE_CHECKING:  # pragma: no cover
    from xml.etree.ElementTree import Element

__all__ = ['Markdown', 'markdown', 'markdownFromFile']


logger = logging.getLogger('MARKDOWN')


```

The `xmlcharrefreplace` function is a part of the `xml.etree.ElementTree` module in Python, which is used for working with XML documents. It replaces references to an XML character reference (such as `\ufeff`) with the corresponding character from the ASCII character set. This is useful when working with entities that may contain non-ASCII characters.


```py
class Markdown:
    """
    A parser which converts Markdown to HTML.

    Attributes:
        Markdown.tab_length (int): The number of spaces which correspond to a single tab. Default: `4`.
        Markdown.ESCAPED_CHARS (list[str]): List of characters which get the backslash escape treatment.
        Markdown.block_level_elements (list[str]): List of HTML tags which get treated as block-level elements.
            See [`markdown.util.BLOCK_LEVEL_ELEMENTS`][] for the full list of elements.
        Markdown.registeredExtensions (list[Extension]): List of extensions which have called
            [`registerExtension`][markdown.Markdown.registerExtension] during setup.
        Markdown.doc_tag (str): Element used to wrap document. Default: `div`.
        Markdown.stripTopLevelTags (bool): Indicates whether the `doc_tag` should be removed. Default: 'True'.
        Markdown.references (dict[str, tuple[str, str]]): A mapping of link references found in a parsed document
             where the key is the reference name and the value is a tuple of the URL and title.
        Markdown.htmlStash (util.HtmlStash): The instance of the `HtmlStash` used by an instance of this class.
        Markdown.output_formats (dict[str, Callable[xml.etree.ElementTree.Element]]): A mapping of known output
             formats by name and their respective serializers. Each serializer must be a callable which accepts an
            [`Element`][xml.etree.ElementTree.Element] and returns a `str`.
        Markdown.output_format (str): The output format set by
            [`set_output_format`][markdown.Markdown.set_output_format].
        Markdown.serializer (Callable[xml.etree.ElementTree.Element]): The serializer set by
            [`set_output_format`][markdown.Markdown.set_output_format].
        Markdown.preprocessors (util.Registry): A collection of [`preprocessors`][markdown.preprocessors].
        Markdown.parser (blockparser.BlockParser): A collection of [`blockprocessors`][markdown.blockprocessors].
        Markdown.inlinePatterns (util.Registry): A collection of [`inlinepatterns`][markdown.inlinepatterns].
        Markdown.treeprocessors (util.Registry): A collection of [`treeprocessors`][markdown.treeprocessors].
        Markdown.postprocessors (util.Registry): A collection of [`postprocessors`][markdown.postprocessors].

    """

    doc_tag = "div"     # Element used to wrap document - later removed

    output_formats: ClassVar[dict[str, Callable[[Element], str]]] = {
        'html':   to_html_string,
        'xhtml':  to_xhtml_string,
    }
    """
    A mapping of known output formats by name and their respective serializers. Each serializer must be a
    callable which accepts an [`Element`][xml.etree.ElementTree.Element] and returns a `str`.
    """

    def __init__(self, **kwargs):
        """
        Creates a new Markdown instance.

        Keyword Arguments:
            extensions (list[Extension | str]): A list of extensions.

                If an item is an instance of a subclass of [`markdown.extensions.Extension`][],
                the instance will be used as-is. If an item is of type `str`, it is passed
                to [`build_extension`][markdown.Markdown.build_extension] with its corresponding
                `extension_configs` and the returned instance  of [`markdown.extensions.Extension`][]
                is used.
            extension_configs (dict[str, dict[str, Any]]): Configuration settings for extensions.
            output_format (str): Format of output. Supported formats are:

                * `xhtml`: Outputs XHTML style tags. Default.
                * `html`: Outputs HTML style tags.
            tab_length (int): Length of tabs in the source. Default: `4`

        """

        self.tab_length: int = kwargs.get('tab_length', 4)

        self.ESCAPED_CHARS: list[str] = [
            '\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!'
        ]
        """ List of characters which get the backslash escape treatment. """

        self.block_level_elements: list[str] = BLOCK_LEVEL_ELEMENTS.copy()

        self.registeredExtensions: list[Extension] = []
        self.docType = ""  # TODO: Maybe delete this. It does not appear to be used anymore.
        self.stripTopLevelTags: bool = True

        self.build_parser()

        self.references: dict[str, tuple[str, str]] = {}
        self.htmlStash: util.HtmlStash = util.HtmlStash()
        self.registerExtensions(extensions=kwargs.get('extensions', []),
                                configs=kwargs.get('extension_configs', {}))
        self.set_output_format(kwargs.get('output_format', 'xhtml'))
        self.reset()

    def build_parser(self) -> Markdown:
        """
        Build the parser from the various parts.

        Assigns a value to each of the following attributes on the class instance:

        * **`Markdown.preprocessors`** ([`Registry`][markdown.util.Registry]) -- A collection of
          [`preprocessors`][markdown.preprocessors].
        * **`Markdown.parser`** ([`BlockParser`][markdown.blockparser.BlockParser]) -- A collection of
          [`blockprocessors`][markdown.blockprocessors].
        * **`Markdown.inlinePatterns`** ([`Registry`][markdown.util.Registry]) -- A collection of
          [`inlinepatterns`][markdown.inlinepatterns].
        * **`Markdown.treeprocessors`** ([`Registry`][markdown.util.Registry]) -- A collection of
          [`treeprocessors`][markdown.treeprocessors].
        * **`Markdown.postprocessors`** ([`Registry`][markdown.util.Registry]) -- A collection of
          [`postprocessors`][markdown.postprocessors].

        This method could be redefined in a subclass to build a custom parser which is made up of a different
        combination of processors and patterns.

        """
        self.preprocessors = build_preprocessors(self)
        self.parser = build_block_parser(self)
        self.inlinePatterns = build_inlinepatterns(self)
        self.treeprocessors = build_treeprocessors(self)
        self.postprocessors = build_postprocessors(self)
        return self

    def registerExtensions(
        self,
        extensions: Sequence[Extension | str],
        configs: Mapping[str, dict[str, Any]]
    ) -> Markdown:
        """
        Load a list of extensions into an instance of the `Markdown` class.

        Arguments:
            extensions (list[Extension | str]): A list of extensions.

                If an item is an instance of a subclass of [`markdown.extensions.Extension`][],
                the instance will be used as-is. If an item is of type `str`, it is passed
                to [`build_extension`][markdown.Markdown.build_extension] with its corresponding `configs` and the
                returned instance  of [`markdown.extensions.Extension`][] is used.
            configs (dict[str, dict[str, Any]]): Configuration settings for extensions.

        """
        for ext in extensions:
            if isinstance(ext, str):
                ext = self.build_extension(ext, configs.get(ext, {}))
            if isinstance(ext, Extension):
                ext.extendMarkdown(self)
                logger.debug(
                    'Successfully loaded extension "%s.%s".'
                    % (ext.__class__.__module__, ext.__class__.__name__)
                )
            elif ext is not None:
                raise TypeError(
                    'Extension "{}.{}" must be of type: "{}.{}"'.format(
                        ext.__class__.__module__, ext.__class__.__name__,
                        Extension.__module__, Extension.__name__
                    )
                )
        return self

    def build_extension(self, ext_name: str, configs: Mapping[str, Any]) -> Extension:
        """
        Build extension from a string name, then return an instance using the given `configs`.

        Arguments:
            ext_name: Name of extension as a string.
            configs: Configuration settings for extension.

        Returns:
            An instance of the extension with the given configuration settings.

        First attempt to load an entry point. The string name must be registered as an entry point in the
        `markdown.extensions` group which points to a subclass of the [`markdown.extensions.Extension`][] class.
        If multiple distributions have registered the same name, the first one found is returned.

        If no entry point is found, assume dot notation (`path.to.module:ClassName`). Load the specified class and
        return an instance. If no class is specified, import the module and call a `makeExtension` function and return
        the [`markdown.extensions.Extension`][] instance returned by that function.
        """
        configs = dict(configs)

        entry_points = [ep for ep in util.get_installed_extensions() if ep.name == ext_name]
        if entry_points:
            ext = entry_points[0].load()
            return ext(**configs)

        # Get class name (if provided): `path.to.module:ClassName`
        ext_name, class_name = ext_name.split(':', 1) if ':' in ext_name else (ext_name, '')

        try:
            module = importlib.import_module(ext_name)
            logger.debug(
                'Successfully imported extension module "%s".' % ext_name
            )
        except ImportError as e:
            message = 'Failed loading extension "%s".' % ext_name
            e.args = (message,) + e.args[1:]
            raise

        if class_name:
            # Load given class name from module.
            return getattr(module, class_name)(**configs)
        else:
            # Expect  `makeExtension()` function to return a class.
            try:
                return module.makeExtension(**configs)
            except AttributeError as e:
                message = e.args[0]
                message = "Failed to initiate extension " \
                          "'%s': %s" % (ext_name, message)
                e.args = (message,) + e.args[1:]
                raise

    def registerExtension(self, extension: Extension) -> Markdown:
        """
        Register an extension as having a resettable state.

        Arguments:
            extension: An instance of the extension to register.

        This should get called once by an extension during setup. A "registered" extension's
        `reset` method is called by [`Markdown.reset()`][markdown.Markdown.reset]. Not all extensions have or need a
        resettable state, and so it should not be assumed that all extensions are "registered."

        """
        self.registeredExtensions.append(extension)
        return self

    def reset(self) -> Markdown:
        """
        Resets all state variables to prepare the parser instance for new input.

        Called once upon creation of a class instance. Should be called manually between calls
        to [`Markdown.convert`][markdown.Markdown.convert].
        """
        self.htmlStash.reset()
        self.references.clear()

        for extension in self.registeredExtensions:
            if hasattr(extension, 'reset'):
                extension.reset()

        return self

    def set_output_format(self, format: str) -> Markdown:
        """
        Set the output format for the class instance.

        Arguments:
            format: Must be a known value in `Markdown.output_formats`.

        """
        self.output_format = format.lower().rstrip('145')  # ignore number
        try:
            self.serializer = self.output_formats[self.output_format]
        except KeyError as e:
            valid_formats = list(self.output_formats.keys())
            valid_formats.sort()
            message = 'Invalid Output Format: "%s". Use one of %s.' \
                % (self.output_format,
                   '"' + '", "'.join(valid_formats) + '"')
            e.args = (message,) + e.args[1:]
            raise
        return self

    # Note: the `tag` argument is type annotated `Any` as ElementTree uses many various objects as tags.
    # As there is no standardization in ElementTree, the type of a given tag is unpredictable.
    def is_block_level(self, tag: Any) -> bool:
        """
        Check if the given `tag` is a block level HTML tag.

        Returns `True` for any string listed in `Markdown.block_level_elements`. A `tag` which is
        not a string always returns `False`.

        """
        if isinstance(tag, str):
            return tag.lower().rstrip('/') in self.block_level_elements
        # Some ElementTree tags are not strings, so return False.
        return False

    def convert(self, source: str) -> str:
        """
        Convert a Markdown string to a string in the specified output format.

        Arguments:
            source: Markdown formatted text as Unicode or ASCII string.

        Returns:
            A string in the specified output format.

        Markdown parsing takes place in five steps:

        1. A bunch of [`preprocessors`][markdown.preprocessors] munge the input text.
        2. A [`BlockParser`][markdown.blockparser.BlockParser] parses the high-level structural elements of the
           pre-processed text into an [`ElementTree`][xml.etree.ElementTree.ElementTree] object.
        3. A bunch of [`treeprocessors`][markdown.treeprocessors] are run against the
           [`ElementTree`][xml.etree.ElementTree.ElementTree] object. One such `treeprocessor`
           ([`markdown.treeprocessors.InlineProcessor`][]) runs [`inlinepatterns`][markdown.inlinepatterns]
           against the [`ElementTree`][xml.etree.ElementTree.ElementTree] object, parsing inline markup.
        4. Some [`postprocessors`][markdown.postprocessors] are run against the text after the
           [`ElementTree`][xml.etree.ElementTree.ElementTree] object has been serialized into text.
        5. The output is returned as a string.

        """

        # Fix up the source text
        if not source.strip():
            return ''  # a blank Unicode string

        try:
            source = str(source)
        except UnicodeDecodeError as e:  # pragma: no cover
            # Customize error message while maintaining original traceback
            e.reason += '. -- Note: Markdown only accepts Unicode input!'
            raise

        # Split into lines and run the line preprocessors.
        self.lines = source.split("\n")
        for prep in self.preprocessors:
            self.lines = prep.run(self.lines)

        # Parse the high-level elements.
        root = self.parser.parseDocument(self.lines).getroot()

        # Run the tree-processors
        for treeprocessor in self.treeprocessors:
            newRoot = treeprocessor.run(root)
            if newRoot is not None:
                root = newRoot

        # Serialize _properly_.  Strip top-level tags.
        output = self.serializer(root)
        if self.stripTopLevelTags:
            try:
                start = output.index(
                    '<%s>' % self.doc_tag) + len(self.doc_tag) + 2
                end = output.rindex('</%s>' % self.doc_tag)
                output = output[start:end].strip()
            except ValueError as e:  # pragma: no cover
                if output.strip().endswith('<%s />' % self.doc_tag):
                    # We have an empty document
                    output = ''
                else:
                    # We have a serious problem
                    raise ValueError('Markdown failed to strip top-level '
                                     'tags. Document=%r' % output.strip()) from e

        # Run the text post-processors
        for pp in self.postprocessors:
            output = pp.run(output)

        return output.strip()

    def convertFile(
        self,
        input: str | BinaryIO | None = None,
        output: str | BinaryIO | None = None,
        encoding: str | None = None,
    ) -> Markdown:
        """
        Converts a Markdown file and returns the HTML as a Unicode string.

        Decodes the file using the provided encoding (defaults to `utf-8`),
        passes the file content to markdown, and outputs the HTML to either
        the provided stream or the file with provided name, using the same
        encoding as the source file. The
        [`xmlcharrefreplace`](https://docs.python.org/3/library/codecs.html#error-handlers)
        error handler is used when encoding the output.

        **Note:** This is the only place that decoding and encoding of Unicode
        takes place in Python-Markdown.  (All other code is Unicode-in /
        Unicode-out.)

        Arguments:
            input: File object or path. Reads from `stdin` if `None`.
            output: File object or path. Writes to `stdout` if `None`.
            encoding: Encoding of input and output files. Defaults to `utf-8`.

        """

        encoding = encoding or "utf-8"

        # Read the source
        if input:
            if isinstance(input, str):
                input_file = codecs.open(input, mode="r", encoding=encoding)
            else:
                input_file = codecs.getreader(encoding)(input)
            text = input_file.read()
            input_file.close()
        else:
            text = sys.stdin.read()

        text = text.lstrip('\ufeff')  # remove the byte-order mark

        # Convert
        html = self.convert(text)

        # Write to file or stdout
        if output:
            if isinstance(output, str):
                output_file = codecs.open(output, "w",
                                          encoding=encoding,
                                          errors="xmlcharrefreplace")
                output_file.write(html)
                output_file.close()
            else:
                writer = codecs.getwriter(encoding)
                output_file = writer(output, errors="xmlcharrefreplace")
                output_file.write(html)
                # Don't close here. User may want to write more.
        else:
            # Encode manually and write bytes to stdout.
            html = html.encode(encoding, "xmlcharrefreplace")
            sys.stdout.buffer.write(html)

        return self


```

这段代码定义了两个函数 `markdown()` 和 `markdownFromFile()`，它们用于将Markdown格式的文本转换为HTML并返回HTML作为Unicode字符串。

`markdown()` 函数的实现比较简单，直接调用 `Markdown` 类的 `convert()` 方法，将Markdown字符串作为参数传递给`Markdown`类，然后返回转换后的HTML字符串。这个函数主要用于一些简单的Markdown场景，如直接输出Markdown原始字符串或者将其转换为标题、列表等格式。

`markdownFromFile()` 函数接受一个名为 `filename` 的参数，表示要转换的Markdown文件。这个函数会读取文件内容，并将其转换为`Markdown` 类的实例，然后再调用 `convert()` 方法将Markdown内容转换为HTML并返回。这个函数主要用于需要从文件中读取Markdown内容的场景，例如在应用程序中从用户输入中读取Markdown内容并进行渲染。


```py
"""
EXPORTED FUNCTIONS
=============================================================================

Those are the two functions we really mean to export: `markdown()` and
`markdownFromFile()`.
"""


def markdown(text: str, **kwargs: Any) -> str:
    """
    Convert a markdown string to HTML and return HTML as a Unicode string.

    This is a shortcut function for [`Markdown`][markdown.Markdown] class to cover the most
    basic use case.  It initializes an instance of [`Markdown`][markdown.Markdown], loads the
    necessary extensions and runs the parser on the given text.

    Arguments:
        text: Markdown formatted text as Unicode or ASCII string.

    Keyword arguments:
        **kwargs: Any arguments accepted by the Markdown class.

    Returns:
        A string in the specified output format.

    """
    md = Markdown(**kwargs)
    return md.convert(text)


```

这段代码定义了一个名为 `markdownFromFile` 的函数，它接受任何数量的 keyword argument，并返回一个 Markdown 对象。

该函数的实现基本上是在 Markdown 类的 `convertFile` 方法的基础上，提供了一个便捷的方式来读取一个 Markdown 文件并将其内容输出到另一个文件或流中。

具体来说，函数接受三个 keyword argument:

- `input(str | BinaryIO):` 表示输入文件名或可读/可写对象。如果传递的是文件名，则函数将读取该文件并将其转换为 Markdown 对象。如果传递的是二进制文件对象，则函数将直接将其转换为 Markdown 对象。
- `output(str | BinaryIO):` 表示输出文件名或可写对象。如果传递的是文件名，则函数将将 Markdown 对象写入该文件中。如果传递的是二进制文件对象，则函数将将其写入该文件中。
- `encoding(str):` 表示输入和输出文件的编码。如果传递的编码与输入或输出文件的编码相同，则函数将不会对输入或输出进行转换。

除了上述三个关键字参数外，还可以传递任何其他 keyword argument，如 `None`。这些 keyword argument 将直接传递给 `Markdown` 类中的 `convertFile` 方法。


```py
def markdownFromFile(**kwargs: Any):
    """
    Read Markdown text from a file and write output to a file or a stream.

    This is a shortcut function which initializes an instance of [`Markdown`][markdown.Markdown],
    and calls the [`convertFile`][markdown.Markdown.convertFile] method rather than
    [`convert`][markdown.Markdown.convert].

    Keyword arguments:
        input (str | BinaryIO): A file name or readable object.
        output (str | BinaryIO): A file name or writable object.
        encoding (str): Encoding of input and output.
        **kwargs: Any arguments accepted by the `Markdown` class.

    """
    md = Markdown(**kwargs)
    md.convertFile(kwargs.get('input', None),
                   kwargs.get('output', None),
                   kwargs.get('encoding', None))

```