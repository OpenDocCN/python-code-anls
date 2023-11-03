# PythonMarkdown源码解析 9

# `/markdown/markdown/extensions/legacy_attrs.py`

该代码是一个Python实现John Gruber所写的Markdown（Markdown扩展渲染语法）的示例。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML，有助于提高文档的可读性和可维护性。

该代码使用了多个第三方库来实现Markdown的渲染，其中包括：
1. `markdown`：一个Python库，提供了许多Markdown渲染和解析功能；
2. `lxml`：一个强大的XML解析库，可以轻松地解析Markdown文档；
3. `requests`：一个用于发送HTTP请求的库，可以用于发送HTTP请求以获取Markdown内容；
4. `pygments`：一个Python库，提供了自动折行和解析Markdown的功能；
5. `pyintlink`：一个Python库，提供了将Markdown链接转换为指向Markdown文件的实际链接的功能。

通过调用这些库，以及一些辅助函数和格式化字符，该代码可以将Markdown文档转换为HTML并将其输出。


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

这段代码是一个Python脚本，它实现了Python Markdown的属性扩展。具体来说，它提供了两个主要功能：

1. 在Python-Markdown版本3.0之前，Markdown类有一个默认的`enable_attributes`关键词，它提供了定义元素属性以使用格式`{@key=value}`的特性。
2. 为了兼容旧的属性格式，该脚本提供了一个新属性`attr_lists`来代替默认的属性格式。新文档应该使用`attr_lists`来编写，但仍然可以继续使用旧的属性格式来编写许多文档。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
An extension to Python Markdown which implements legacy attributes.

Prior to Python-Markdown version 3.0, the Markdown class had an `enable_attributes`
keyword which was on by default and provided for attributes to be defined for elements
using the format `{@key=value}`. This extension is provided as a replacement for
backward compatibility. New documents should be authored using `attr_lists`. However,
numerous documents exist which have been using the old attribute format for many
years. This extension can be used to continue to render those documents correctly.
"""

```

这段代码的作用是定义了一个名为"LegacyAttrs"的类，该类继承自"markdown.treeprocessors.Treeprocessor"类。这个类的目标是解析Markdown文档中的语义元素，并将它们与标记相关的属性相结合。通过使用"from __future__ import annotations"来引入第二个小时的语法定义，这个类可以更好地支持将来的定义。

具体来说，这个类包含以下方法：

- "run(doc:etree.Element)"：这个方法接收一个Markdown文档的元素对象作为参数，并对其中的所有元素进行处理。它的目的是设置文档中所有语义元素的属性。

- "handleAttributes(el:etree.Element, txt:str)=str"：这个方法接收一个语义元素对象和一个字符串作为参数，并返回一个新的字符串，其中标记和属性值已经被设置。它的目的是设置语义元素的属性和值。

- "handleAttributes(el:etree.Element, alt:str)=str"：这个方法与"handleAttributes()"类似，但允许在标记中使用alt属性。

- "handleAttributes(el:etree.Element, text:str)=str"：这个方法与"handleAttributes()"类似，但允许在语义元素中使用text属性。

此外，这个类还包含两个辅助方法：

- "isString(text:str)"：这个方法接收一个字符串作为参数，并返回一个布尔值，表示它是否是一个真正的字符串。

- "as_attribute(tag:str, attr:str):str"：这个方法接收一个标记和属性名称作为参数，并返回一个将属性与标记结合起来的字符串。它旨在将标记和属性名称组合成一个可以用于Markdown文档的属性。


```py
from __future__ import annotations

import re
from markdown.treeprocessors import Treeprocessor, isString
from markdown.extensions import Extension
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import xml.etree.ElementTree as etree


ATTR_RE = re.compile(r'\{@([^\}]*)=([^\}]*)}')  # {@id=123}


class LegacyAttrs(Treeprocessor):
    def run(self, doc: etree.Element) -> None:
        """Find and set values of attributes ({@key=value}). """
        for el in doc.iter():
            alt = el.get('alt', None)
            if alt is not None:
                el.set('alt', self.handleAttributes(el, alt))
            if el.text and isString(el.text):
                el.text = self.handleAttributes(el, el.text)
            if el.tail and isString(el.tail):
                el.tail = self.handleAttributes(el, el.tail)

    def handleAttributes(self, el: etree.Element, txt: str) -> str:
        """ Set attributes and return text without definitions. """
        def attributeCallback(match: re.Match[str]):
            el.set(match.group(1), match.group(2).replace('\n', ' '))
        return ATTR_RE.sub(attributeCallback, txt)


```

这段代码定义了一个名为 `LegacyAttrExtension` 的类，它实现了 `Extension` 接口。这个类的父类是 `Extension`，意味着它可以从其他类继承属性和方法。

在这个类中，有一个名为 `extendMarkdown` 的方法，这个方法接受一个 `md` 参数，表示 Markdown 对象。在这个方法中，使用 `register` 方法将 `LegacyAttrs` 类实例注册到 `md.treeprocessors` 列表中。这里，`treeprocessors` 是一个内置的属性，用于注册 Markdown 处理器。

通过扩展 `Extension`，可以为 Markdown 对象添加一些自定义的处理逻辑。在这个例子中，通过扩展 `Markdown`，添加了一个名为 `legacyattrs` 的处理器，这个处理器可以对 Markdown 对象进行处理，将其转换为包含 `LegacyAttrs` 类的实例。由于 `register` 方法中传递的是一个参数对象 `**kwargs`，因此这个扩展可以被应用于任何需要这个扩展的 Markdown 对象上。


```py
class LegacyAttrExtension(Extension):
    def extendMarkdown(self, md):
        """ Add `LegacyAttrs` to Markdown instance. """
        md.treeprocessors.register(LegacyAttrs(md), 'legacyattrs', 15)


def makeExtension(**kwargs):  # pragma: no cover
    return LegacyAttrExtension(**kwargs)

```

# `/markdown/markdown/extensions/legacy_em.py`

这段代码是一个Python Markdown扩展，它提供了对HTML文档中连接单词（即<br>标签）的处理。函数作用域内包含两个参数：

1. 一个类的自定义扩展函数，名为`_connected_words_`，用于设置或取消默认的连接单词行为。
2. 一个类的自定义扩展函数，名为`__future__`，用于在函数范围内使用Python 3的特性。


```py
# Legacy Em Extension for Python-Markdown
# =======================================

# This extension provides legacy behavior for _connected_words_.

# Copyright 2015-2018 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
This extension provides legacy behavior for _connected_words_.
"""

from __future__ import annotations

```

这段代码定义了一个名为"LegacyUnderscoreProcessor"的类，继承自另一个名为"UnderscoreProcessor"的类。这个类定义了一个处理 strong 和 em 匹配的方法，主要通过在 underscores 上使用给定的模式来增强文本的表达。

具体来说，这个类的长寿模式可以使用以下模式之一来增强匹配：

- `EmStrongItem`：匹配两个模式，一个是 `Em_Strong2_RE`(匹配两个 strong 模式)，另一个是 `EMPHASIS_RE`(匹配一个强调模式)，将它们与一个类中的两个方法组合，以确保匹配两个或更多 strong 模式。
- `EmStrongItem`：匹配两个模式，一个是 `STRONG_EM2_RE`(匹配两个 em 模式)，另一个是 `EMPHASIS_RE`(匹配一个强调模式)，将它们与一个类中的两个方法组合，以确保匹配两个或更多 strong 模式。
- `EmStrongItem`：匹配两个模式，一个是 `STRONG_EM_RE`(匹配两个 em 模式)，另一个是 `EMPHASIS_RE`(匹配一个强调模式)，将它们与一个类中的两个方法组合，以确保匹配两个或更多 strong 模式。
- `EmStrongItem`：匹配两个模式，一个是 `EMPHASIS_RE`(匹配一个强调模式)，另一个是 `STRONG_RE`(匹配一个 strong 模式)，将它们与一个类中的两个方法组合，以确保匹配两个或多个 strong 模式。
- `EmStrongItem`：匹配两个模式，一个是 `EMPHASIS_RE`(匹配一个强调模式)，另一个是 `EMPHASIS_RE`(匹配另一个强调模式)，将它们与一个类中的两个方法组合，以确保匹配两个或多个 strong 模式。

最后，定义了一个名为"LegacyUnderscoreProcessor"的类，继承自"UnderscoreProcessor"，它实现了上述的所有匹配模式，并将它们注册到了自己的 PATTERNS 属性中。这个类的方法被传递给父类的实例，以便在需要时可以调用其处理 strong 和 em 匹配的文本。


```py
from . import Extension
from ..inlinepatterns import UnderscoreProcessor, EmStrongItem, EM_STRONG2_RE, STRONG_EM2_RE
import re

# _emphasis_
EMPHASIS_RE = r'(_)([^_]+)\1'

# __strong__
STRONG_RE = r'(_{2})(.+?)\1'

# __strong_em___
STRONG_EM_RE = r'(_)\1(?!\1)([^_]+?)\1(?!\1)(.+?)\1{3}'


class LegacyUnderscoreProcessor(UnderscoreProcessor):
    """Emphasis processor for handling strong and em matches inside underscores."""

    PATTERNS = [
        EmStrongItem(re.compile(EM_STRONG2_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'),
        EmStrongItem(re.compile(STRONG_EM2_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'),
        EmStrongItem(re.compile(STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'),
        EmStrongItem(re.compile(STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'),
        EmStrongItem(re.compile(EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')
    ]


```

这段代码定义了一个名为`LegacyEmExtension`的类，其父类为`Extension`类。这个类的目的是扩展Markdown文件的语法，使其能够使用`<em>`标签来强调文本，类似于`<strong>`标签，但是它具有更多的功能。

具体来说，这个类的`extendMarkdown`方法修改了Markdown文件的内置直接模式，注册了一个名为`LegacyUnderscoreProcessor`的类，这个类的父类是`Processor`类。在`register`方法中，我们使用了`r'<_>'`这个正则表达式来匹配Markdown文件中的`<em>`标签，并注册了一个名为`em_strong2`的过滤器，其作用是保留`<em>`标签并将其强度级别设置为50。通过这个过滤器，即使一个Markdown文件中存在多个`<em>`标签，这个插件也能够正确地处理它们。

这个插件的类还定义了一个`makeExtension`方法，通过这个方法可以创建一个`LegacyEmExtension`实例，这个实例将调用类中`extendMarkdown`方法来修改Markdown文件的语法。

总的来说，这段代码定义了一个用于在Markdown文件中使用`<em>`标签的插件，通过扩展Markdown文件的语法，使得Markdown文件能够正确地使用了`<em>`标签来强调文本。


```py
class LegacyEmExtension(Extension):
    """ Add legacy_em extension to Markdown class."""

    def extendMarkdown(self, md):
        """ Modify inline patterns. """
        md.inlinePatterns.register(LegacyUnderscoreProcessor(r'_'), 'em_strong2', 50)


def makeExtension(**kwargs):  # pragma: no cover
    """ Return an instance of the `LegacyEmExtension` """
    return LegacyEmExtension(**kwargs)

```

# `/markdown/markdown/extensions/md_in_html.py`

这段代码是一个名为“php-markdown”的开源项目中的一个扩展，实现了对PHP Markdown语法的解析，将Markdown语法转换为原始HTML。

具体来说，这段代码实现了以下功能：

1. 从输入的Markdown字符串中提取语法树。
2. 解析Markdown语法树，生成对应的HTML。
3. 将生成的HTML输出到屏幕。

由于这段代码没有注释，我们无法判断作者的具体意图。但是，从代码中可以看出，它对PHP Markdown语法有很强的解析能力，可以让我们更加轻松地使用Markdown语法来编写文章。


```py
# Python-Markdown Markdown in HTML Extension
# ===============================

# An implementation of [PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)'s
# parsing of Markdown syntax in raw HTML.

# See https://Python-Markdown.github.io/extensions/raw_html
# for documentation.

# Copyright The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
An implementation of [PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)'s
```

这段代码的作用是解析Markdown语法为原始HTML。它使用Python Markdown扩展中的RawHtmlPostprocessor来处理Markdown语法。具体来说，它将Markdown文档中的内容通过HTMLExtractor提取为标记树，然后通过BlockProcessor和Preprocessor进行预处理，最后通过RawHtmlPostprocessor将预处理后的内容再次转化为HTML。

需要注意的是，这段代码并不直接输出处理后的HTML，而是将处理后的内容保存在一个名为`raw_html`的实体内，以便稍后进行处理。这个实体的作用是为了解决HTMLExtractor在处理Markdown时可能出现的一些问题，例如在解析文档时可能会出现空白行。通过保留原始输入，可以更好地处理这些情况。


```py
parsing of Markdown syntax in raw HTML.

See the [documentation](https://Python-Markdown.github.io/extensions/raw_html)
for details.
"""

from __future__ import annotations

from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
```

This is a class called `HTMLExtractor` which appears to be part of the HTML5 parser used by the
Selenium WebDriver. It has methods for handling data andempties in theraw data, which is
润色为与`.content`相比，它是如何做到的呢？似乎两者都不是很好的选择。

handles the case when the parser is in raw data mode.

In the `handle_data` method, it seems to be checking whether the line starts with an empty tag.如果是，则跳过当前
text，否则继续提取文本，并将标记存入mdstack中。

在这里，似乎标记不仅包括在handle\_data`方法中，还包括在parse\_html\_declaration`方法中。然而，它们似乎都继承自同一个超级
handle\_data`方法，因此没有提供更多的选择。

因此，HTMLExtractor似乎只是一个简单的HTML5解析器，它在遇到empty标签时跳过当前text，并将标记存入mdstack中。尽管如此，它仍然处理了相同标记


```py
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


class HTMLExtractorExtra(HTMLExtractor):
    """
    Override `HTMLExtractor` and create `etree` `Elements` for any elements which should have content parsed as
    Markdown.
    """

    def __init__(self, md: Markdown, *args, **kwargs):
        # All block-level tags.
        self.block_level_tags = set(md.block_level_elements.copy())
        # Block-level tags in which the content only gets span level parsing
        self.span_tags = set(
            ['address', 'dd', 'dt', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'legend', 'li', 'p', 'summary', 'td', 'th']
        )
        # Block-level tags which never get their content parsed.
        self.raw_tags = set(['canvas', 'math', 'option', 'pre', 'script', 'style', 'textarea'])

        super().__init__(md, *args, **kwargs)

        # Block-level tags in which the content gets parsed as blocks
        self.block_tags = set(self.block_level_tags) - (self.span_tags | self.raw_tags | self.empty_tags)
        self.span_and_blocks_tags = self.block_tags | self.span_tags

    def reset(self):
        """Reset this instance.  Loses all unprocessed data."""
        self.mdstack: list[str] = []  # When markdown=1, stack contains a list of tags
        self.treebuilder = etree.TreeBuilder()
        self.mdstate: list[Literal['block', 'span', 'off', None]] = []
        super().reset()

    def close(self):
        """Handle any buffered data."""
        super().close()
        # Handle any unclosed tags.
        if self.mdstack:
            # Close the outermost parent. `handle_endtag` will close all unclosed children.
            self.handle_endtag(self.mdstack[0])

    def get_element(self) -> etree.Element:
        """ Return element from `treebuilder` and reset `treebuilder` for later use. """
        element = self.treebuilder.close()
        self.treebuilder = etree.TreeBuilder()
        return element

    def get_state(self, tag, attrs: Mapping[str, str]) -> Literal['block', 'span', 'off', None]:
        """ Return state from tag and `markdown` attribute. One of 'block', 'span', or 'off'. """
        md_attr = attrs.get('markdown', '0')
        if md_attr == 'markdown':
            # `<tag markdown>` is the same as `<tag markdown='1'>`.
            md_attr = '1'
        parent_state = self.mdstate[-1] if self.mdstate else None
        if parent_state == 'off' or (parent_state == 'span' and md_attr != '0'):
            # Only use the parent state if it is more restrictive than the markdown attribute.
            md_attr = parent_state
        if ((md_attr == '1' and tag in self.block_tags) or
                (md_attr == 'block' and tag in self.span_and_blocks_tags)):
            return 'block'
        elif ((md_attr == '1' and tag in self.span_tags) or
              (md_attr == 'span' and tag in self.span_and_blocks_tags)):
            return 'span'
        elif tag in self.block_level_tags:
            return 'off'
        else:  # pragma: no cover
            return None

    def handle_starttag(self, tag, attrs):
        # Handle tags that should always be empty and do not specify a closing tag
        if tag in self.empty_tags and (self.at_line_start() or self.intail):
            attrs = {key: value if value is not None else key for key, value in attrs}
            if "markdown" in attrs:
                attrs.pop('markdown')
                element = etree.Element(tag, attrs)
                data = etree.tostring(element, encoding='unicode', method='html')
            else:
                data = self.get_starttag_text()
            self.handle_empty_tag(data, True)
            return

        if tag in self.block_level_tags and (self.at_line_start() or self.intail):
            # Valueless attribute (ex: `<tag checked>`) results in `[('checked', None)]`.
            # Convert to `{'checked': 'checked'}`.
            attrs = {key: value if value is not None else key for key, value in attrs}
            state = self.get_state(tag, attrs)
            if self.inraw or (state in [None, 'off'] and not self.mdstack):
                # fall back to default behavior
                attrs.pop('markdown', None)
                super().handle_starttag(tag, attrs)
            else:
                if 'p' in self.mdstack and tag in self.block_level_tags:
                    # Close unclosed 'p' tag
                    self.handle_endtag('p')
                self.mdstate.append(state)
                self.mdstack.append(tag)
                attrs['markdown'] = state
                self.treebuilder.start(tag, attrs)
        else:
            # Span level tag
            if self.inraw:
                super().handle_starttag(tag, attrs)
            else:
                text = self.get_starttag_text()
                if self.mdstate and self.mdstate[-1] == "off":
                    self.handle_data(self.md.htmlStash.store(text))
                else:
                    self.handle_data(text)
                if tag in self.CDATA_CONTENT_ELEMENTS:
                    # This is presumably a standalone tag in a code span (see #1036).
                    self.clear_cdata_mode()

    def handle_endtag(self, tag):
        if tag in self.block_level_tags:
            if self.inraw:
                super().handle_endtag(tag)
            elif tag in self.mdstack:
                # Close element and any unclosed children
                while self.mdstack:
                    item = self.mdstack.pop()
                    self.mdstate.pop()
                    self.treebuilder.end(item)
                    if item == tag:
                        break
                if not self.mdstack:
                    # Last item in stack is closed. Stash it
                    element = self.get_element()
                    # Get last entry to see if it ends in newlines
                    # If it is an element, assume there is no newlines
                    item = self.cleandoc[-1] if self.cleandoc else ''
                    # If we only have one newline before block element, add another
                    if not item.endswith('\n\n') and item.endswith('\n'):
                        self.cleandoc.append('\n')
                    self.cleandoc.append(self.md.htmlStash.store(element))
                    self.cleandoc.append('\n\n')
                    self.state = []
                    # Check if element has a tail
                    if not blank_line_re.match(
                            self.rawdata[self.line_offset + self.offset + len(self.get_endtag_text(tag)):]):
                        # More content exists after `endtag`.
                        self.intail = True
            else:
                # Treat orphan closing tag as a span level tag.
                text = self.get_endtag_text(tag)
                if self.mdstate and self.mdstate[-1] == "off":
                    self.handle_data(self.md.htmlStash.store(text))
                else:
                    self.handle_data(text)
        else:
            # Span level tag
            if self.inraw:
                super().handle_endtag(tag)
            else:
                text = self.get_endtag_text(tag)
                if self.mdstate and self.mdstate[-1] == "off":
                    self.handle_data(self.md.htmlStash.store(text))
                else:
                    self.handle_data(text)

    def handle_startendtag(self, tag, attrs):
        if tag in self.empty_tags:
            attrs = {key: value if value is not None else key for key, value in attrs}
            if "markdown" in attrs:
                attrs.pop('markdown')
                element = etree.Element(tag, attrs)
                data = etree.tostring(element, encoding='unicode', method='html')
            else:
                data = self.get_starttag_text()
        else:
            data = self.get_starttag_text()
        self.handle_empty_tag(data, is_block=self.md.is_block_level(tag))

    def handle_data(self, data):
        if self.intail and '\n' in data:
            self.intail = False
        if self.inraw or not self.mdstack:
            super().handle_data(data)
        else:
            self.treebuilder.data(data)

    def handle_empty_tag(self, data, is_block):
        if self.inraw or not self.mdstack:
            super().handle_empty_tag(data, is_block)
        else:
            if self.at_line_start() and is_block:
                self.handle_data('\n' + self.md.htmlStash.store(data) + '\n\n')
            else:
                self.handle_data(self.md.htmlStash.store(data))

    def parse_pi(self, i: int) -> int:
        if self.at_line_start() or self.intail or self.mdstack:
            # The same override exists in `HTMLExtractor` without the check
            # for `mdstack`. Therefore, use parent of `HTMLExtractor` instead.
            return super(HTMLExtractor, self).parse_pi(i)
        # This is not the beginning of a raw block so treat as plain data
        # and avoid consuming any tags which may follow (see #1066).
        self.handle_data('<?')
        return i + 2

    def parse_html_declaration(self, i: int) -> int:
        if self.at_line_start() or self.intail or self.mdstack:
            # The same override exists in `HTMLExtractor` without the check
            # for `mdstack`. Therefore, use parent of `HTMLExtractor` instead.
            return super(HTMLExtractor, self).parse_html_declaration(i)
        # This is not the beginning of a raw block so treat as plain data
        # and avoid consuming any tags which may follow (see #1066).
        self.handle_data('<!')
        return i + 2


```

This is a `js_okumndo.js` file that contains the implementation for the Okumndo高度自定义解析引擎的 `parseTree` 函数。这个函数负责解析 Markdown 中的复杂元素，如列表、段落、图像和链接等。

首先，我们检查元素是否包含文本内容。如果是，我们将其文本内容删除并将其子元素替换为空字符串。然后，我们根据 `md_attr` 的值来决定如何处理该元素。如果 `md_attr` 是 `span`，我们将遍历其 children，并删除任何 `markdown` 属性。

如果 `md_attr` 不是 `span`，我们将处理其 children，并将其插入到父元素中。首先，我们将处理 `md_attr` 等于 `span` 的子元素。然后，我们将 `md_attr` 不为 `span` 的子元素传递给 `self.parse_element_content` 函数。

在 `self.parse_element_content` 函数中，我们将解析 `self.parser.md.htmlStash.rawHtmlBlocks` 数组中的内容，并将其插入到 `element` 的 children 中。我们将 `md_attr` 不为 `span` 的子元素作为 `self.parser.md.htmlStash.rawHtmlBlocks` 数组的索引，并将 `rawHtmlBlocks` 数组中的内容作为参数传递给 `self.parse_element_content` 函数。

最后，我们将更新 `self.parser.md.htmlStash.rawHtmlBlocks` 和 `self.parser.md.htmlStash.rawHtmlBlocks.insert` 方法的索引，以反映已插入的文本内容。我们将原始的 `md_attr` 值作为参数传递给 `self.parser.md.htmlStash.rawHtmlBlocks.insert` 方法，以确保在插入元素时可以保留原始的 `md_attr` 值。


```py
class HtmlBlockPreprocessor(Preprocessor):
    """Remove html blocks from the text and store them for later retrieval."""

    def run(self, lines: list[str]) -> list[str]:
        source = '\n'.join(lines)
        parser = HTMLExtractorExtra(self.md)
        parser.feed(source)
        parser.close()
        return ''.join(parser.cleandoc).split('\n')


class MarkdownInHtmlProcessor(BlockProcessor):
    """Process Markdown Inside HTML Blocks which have been stored in the `HtmlStash`."""

    def test(self, parent: etree.Element, block: str) -> bool:
        # Always return True. `run` will return `False` it not a valid match.
        return True

    def parse_element_content(self, element: etree.Element) -> None:
        """
        Recursively parse the text content of an `etree` Element as Markdown.

        Any block level elements generated from the Markdown will be inserted as children of the element in place
        of the text content. All `markdown` attributes are removed. For any elements in which Markdown parsing has
        been disabled, the text content of it and its children are wrapped in an `AtomicString`.
        """

        md_attr = element.attrib.pop('markdown', 'off')

        if md_attr == 'block':
            # Parse content as block level
            # The order in which the different parts are parsed (text, children, tails) is important here as the
            # order of elements needs to be preserved. We can't be inserting items at a later point in the current
            # iteration as we don't want to do raw processing on elements created from parsing Markdown text (for
            # example). Therefore, the order of operations is children, tails, text.

            # Recursively parse existing children from raw HTML
            for child in list(element):
                self.parse_element_content(child)

            # Parse Markdown text in tail of children. Do this separate to avoid raw HTML parsing.
            # Save the position of each item to be inserted later in reverse.
            tails = []
            for pos, child in enumerate(element):
                if child.tail:
                    block = child.tail.rstrip('\n')
                    child.tail = ''
                    # Use a dummy placeholder element.
                    dummy = etree.Element('div')
                    self.parser.parseBlocks(dummy, block.split('\n\n'))
                    children = list(dummy)
                    children.reverse()
                    tails.append((pos + 1, children))

            # Insert the elements created from the tails in reverse.
            tails.reverse()
            for pos, tail in tails:
                for item in tail:
                    element.insert(pos, item)

            # Parse Markdown text content. Do this last to avoid raw HTML parsing.
            if element.text:
                block = element.text.rstrip('\n')
                element.text = ''
                # Use a dummy placeholder element as the content needs to get inserted before existing children.
                dummy = etree.Element('div')
                self.parser.parseBlocks(dummy, block.split('\n\n'))
                children = list(dummy)
                children.reverse()
                for child in children:
                    element.insert(0, child)

        elif md_attr == 'span':
            # Span level parsing will be handled by inline processors.
            # Walk children here to remove any `markdown` attributes.
            for child in list(element):
                self.parse_element_content(child)

        else:
            # Disable inline parsing for everything else
            if element.text is None:
                element.text = ''
            element.text = util.AtomicString(element.text)
            for child in list(element):
                self.parse_element_content(child)
                if child.tail:
                    child.tail = util.AtomicString(child.tail)

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        m = util.HTML_PLACEHOLDER_RE.match(blocks[0])
        if m:
            index = int(m.group(1))
            element = self.parser.md.htmlStash.rawHtmlBlocks[index]
            if isinstance(element, etree.Element):
                # We have a matched element. Process it.
                blocks.pop(0)
                self.parse_element_content(element)
                parent.append(element)
                # Cleanup stash. Replace element with empty string to avoid confusing postprocessor.
                self.parser.md.htmlStash.rawHtmlBlocks.pop(index)
                self.parser.md.htmlStash.rawHtmlBlocks.insert(index, '')
                # Confirm the match to the `blockparser`.
                return True
        # No match found.
        return False


```

这段代码定义了一个名为 `MarkdownInHTMLPostprocessor` 的类，它继承自 `RawHtmlPostprocessor` 类（在背后可能还有其他类的继承）。

`stash_to_string` 方法接收一个 `text` 参数，它可以选择是字符串实例（如果是，则该方法将直接返回），或者是 `etree.Element` 实例（如果是，则该方法将在原始 HTML 中查找并提取 `etree` 元素）。对于这两种情况，该方法都会将其转换为 Markdown 语法树的一部分，然后使用 `md.serializer` 方法将它们返回。如果 `text` 是字符串，则直接将其转换为 Markdown 语法树的一部分并返回。

`MarkdownInHtmlExtension` 类负责将 Markdown 解析扩展到 HTML 中。在 `extendMarkdown` 方法中，首先用新注册的 `HtmlBlockPreprocessor` 替换原始的 `RawHtmlPostprocessor`。然后注册了一个 `MarkdownInHtmlProcessor` 实例，它被绑定到 `markdown_block` 编号，用于将 Markdown 语法树转换为 HTML。最后，用新注册的 `MarkdownInHTMLPostprocessor` 替换原始的 `RawHtmlPostprocessor`。这样，当需要将 Markdown 解析扩展到 HTML 时，就会首先经过 `MarkdownInHtmlProcessor` 的处理，然后才会被 `HtmlBlockPreprocessor` 处理。


```py
class MarkdownInHTMLPostprocessor(RawHtmlPostprocessor):
    def stash_to_string(self, text: str | etree.Element) -> str:
        """ Override default to handle any `etree` elements still in the stash. """
        if isinstance(text, etree.Element):
            return self.md.serializer(text)
        else:
            return str(text)


class MarkdownInHtmlExtension(Extension):
    """Add Markdown parsing in HTML to Markdown class."""

    def extendMarkdown(self, md):
        """ Register extension instances. """

        # Replace raw HTML preprocessor
        md.preprocessors.register(HtmlBlockPreprocessor(md), 'html_block', 20)
        # Add `blockprocessor` which handles the placeholders for `etree` elements
        md.parser.blockprocessors.register(
            MarkdownInHtmlProcessor(md.parser), 'markdown_block', 105
        )
        # Replace raw HTML postprocessor
        md.postprocessors.register(MarkdownInHTMLPostprocessor(md), 'raw_html', 30)


```

这段代码定义了一个名为 `makeExtension` 的函数，它接受一个或多个参数 `**kwargs`，并将它们传递给 `MarkdownInHtmlExtension` 函数，这个函数将扩展名为 `.md` 或 `.html` 的 Markdown 文件转换为 HTML 格式。

具体来说，该函数的作用是实现将 Markdown 文件中的扩展名 `.md` 或 `.html` 转换为 HTML 格式的代码。在这个过程中，函数将Markdown中的所有引用标记替换为具有相应扩展名的 HTML 标记。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return MarkdownInHtmlExtension(**kwargs)

```

# `/markdown/markdown/extensions/meta.py`

这段代码是一个Python-Markdown的元数据扩展，目的是处理元数据以添加到标记down中。

具体来说，它做了以下几件事情：

1. 在定义函数时使用了`#`注释，这是一个保留的Python注释标记，告诉代码接下来的内容是注释。

2. 定义了一个名为`meta_data_extension`的函数。

3. 在函数内部使用了`# See`注释，这个注释是参考另一个名为`meta_data`的Python-Markdown扩展的文档而写的。

4. 在函数内部定义了一个名为`add_meta_data`的函数，并将它作为`meta_data_extension`函数的一部分。

5. 在函数内部定义了一个名为`requirements.txt`的文件，并将它作为元数据扩展的依赖项列表。

6. 在函数内部定义了一个名为`__version__`的函数，并将它作为元数据扩展的版本号。

7. 在函数内部定义了一个名为`__description__`的函数，并将它作为元数据扩展的描述。

8. 在函数内部创建了一个名为`meta_data`的新类，并使用`__init__`函数来初始化它。

9. 在`meta_data`类中定义了一个名为`metadata`的新属性，类型为字典。

10. 在`meta_data`类中定义了一个名为`add_metadata`的新方法，用于将元数据添加到标记down中。

11. 在`add_meta_data`方法中，使用了`markdown`库中的`add_meta`方法来添加元数据。

12. 在`meta_data`类中定义了一个名为`requirements`的新属性，类型为列表。

13. 在`meta_data`类中定义了一个名为`version`的新属性，类型为字符串。

14. 在`meta_data`类中定义了一个名为`description`的新属性，类型为字符串。

15. 在`meta_data_extension`函数中，将这些元数据添加到了标记down中。

16. 最后，定义了一个名为`__main__`的函数，返回一个布尔值，如果成功定义了这个元数据扩展，则返回True，否则返回False。


```py
# Meta Data Extension for Python-Markdown
# =======================================

# This extension adds Meta Data handling to markdown.

# See https://Python-Markdown.github.io/extensions/meta_data
# for documentation.

# Original code Copyright 2007-2008 [Waylan Limberg](http://achinghead.com).

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
```

这段代码是一个扩展，名为 "MARKDOWN-META-DATA"，它将元数据处理添加到了 Markdown 中。

具体来说，这个扩展将以下内容存储在元数据中：

- 标题（尚不支持）：
- 描述（尚不支持）：
- 更新日期：

这个扩展使用了 Python 的 "Markdown" 库，并添加了 "MARKDOWN-META-DATA" 元数据类别。通过这个扩展，用户可以将标题、描述和更新日期添加到 Markdown 文档中，这将使用 YAML（YAML）格式的选项。

如果你想了解更多关于这个扩展的详细信息，请查看 [官方文档](https://Python-Markdown.github.io/extensions/meta_data)。


```py
This extension adds Meta Data handling to markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/meta_data)
for details.
"""

from __future__ import annotations

from . import Extension
from ..preprocessors import Preprocessor
import re
import logging
from typing import Any

log = logging.getLogger('MARKDOWN')

```

这段代码定义了一个名为 "MetaExtension" 的类，用于在 Python-Markdown 中添加 "Meta-Data" 扩展。

该类包含三个函数：

1. `extendMarkdown`：将 `MetaPreprocessor` 类添加到 Markdown 实例中，并将 `self.md` 变量保存下来。
2. `reset`：将 "Meta" 字典和 `self.md` 变量重置为初始值。
3. `registerExtension`：注册 `MetaPreprocessor` 类作为 Markdown 实例的预处理器，并在 `md.preprocessors.register` 方法中传递 `self`，参数为 `md` 和 `meta` 分别是 Markdown 和 "Meta-Data" 预处理器的实例。第一个参数是预处理器，第二个参数是优先级，这里是 27。

该类继承自 `Extension` 类，可能还会继承其他类，具体取决于扩展的版本和需要。


```py
# Global Vars
META_RE = re.compile(r'^[ ]{0,3}(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)')
META_MORE_RE = re.compile(r'^[ ]{4,}(?P<value>.*)')
BEGIN_RE = re.compile(r'^-{3}(\s.*)?')
END_RE = re.compile(r'^(-{3}|\.{3})(\s.*)?')


class MetaExtension (Extension):
    """ Meta-Data extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Add `MetaPreprocessor` to Markdown instance. """
        md.registerExtension(self)
        self.md = md
        md.preprocessors.register(MetaPreprocessor(md), 'meta', 27)

    def reset(self) -> None:
        self.md.Meta = {}


```

这段代码定义了一个名为MetaPreprocessor的类，其继承自另一个名为Preprocessor的类。这个类的主要作用是解析Markdown文档中的元数据(Meta-Data)，并将解析得到的元数据存储为Markdown.Meta数据结构中。

具体来说，这个类包含了一个run方法，用于运行Meta-Data解析过程。在run方法中，首先通过lines获取输入Markdown文档中的所有行。然后，使用两个循环从每个行中提取出可能包含元数据的关键字和值。如果关键字或值匹配Begin_RE或End_RE模式，就删除该关键字或值，否则就执行try-except语句，将获取到的值添加到元数据字典中。如果既没有Begin_RE也没有End_RE模式，就将当前行作为元数据，然后退出解析过程。

最终，这个MetaPreprocessor类的run方法返回一个被解析为Markdown.Meta数据结构的元数据列表。这个列表将被用于将Markdown文档转换为所需的格式，并可以用于将元数据存储到Markdown.Meta对象中，从而使Markdown文档更具有可读性和可访问性。


```py
class MetaPreprocessor(Preprocessor):
    """ Get Meta-Data. """

    def run(self, lines: list[str]) -> list[str]:
        """ Parse Meta-Data and store in Markdown.Meta. """
        meta: dict[str, Any] = {}
        key = None
        if lines and BEGIN_RE.match(lines[0]):
            lines.pop(0)
        while lines:
            line = lines.pop(0)
            m1 = META_RE.match(line)
            if line.strip() == '' or END_RE.match(line):
                break  # blank line or end of YAML header - done
            if m1:
                key = m1.group('key').lower().strip()
                value = m1.group('value').strip()
                try:
                    meta[key].append(value)
                except KeyError:
                    meta[key] = [value]
            else:
                m2 = META_MORE_RE.match(line)
                if m2 and key:
                    # Add another line to existing key
                    meta[key].append(m2.group('value').strip())
                else:
                    lines.insert(0, line)
                    break  # no meta data - done
        self.md.Meta = meta
        return lines


```

这段代码定义了一个名为 `makeExtension` 的函数，它接受一个或多个参数 `**kwargs`，并将其传递给另一个函数 `MetaExtension`。

具体来说，这段代码的作用是创建了一个装饰函数，用于在调用 `MetaExtension` 函数时自动添加扩展类型信息。这个装饰函数通过 `**kwargs` 的形式接受任意数量的参数，并将它们传递给 `MetaExtension` 函数。这样，当 `makeExtension` 函数被调用时，它实际上调用了 `MetaExtension` 函数，并将所有传递给它的参数传递给它。

由于 `MetaExtension` 函数的实现没有被显示在代码中，因此我们无法看到这段代码是如何实现它的。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return MetaExtension(**kwargs)

```

# `/markdown/markdown/extensions/nl2br.py`

这段代码是一个Python-Markdown扩展，名为`NL2BR`。它的作用是处理新行并将其视为硬性break，类似于GitHub中使用的Markdown。这个扩展将使得Python-Markdown能够更好地支持Markdown语法。

具体来说，这个扩展主要实现了以下功能：

1. 将一个换行符（'\n'）替换为一个行为空的分隔符（''）。
2. 在当前行添加一个空白行（''）。
3. 如果当前行已经包含了一个换行符，则忽略。

对于使用这个extension的Python-Markdown，它们将能够以类似于GitHub Markdown的方式处理新行。例如，你可以将一个具有多个换行符的段落存储为一个列表，然后将其转换为一个具有多个空行的新列表。这将允许你在处理Markdown时更轻松地处理其中的新行。


```py
# `NL2BR` Extension
# ===============

# A Python-Markdown extension to treat newlines as hard breaks; like
# GitHub-flavored Markdown does.

# See https://Python-Markdown.github.io/extensions/nl2br
# for documentation.

# Original code Copyright 2011 [Brian Neal](https://deathofagremmie.com/)

# All changes Copyright 2011-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

```

这段代码是一个Python-Markdown扩展，其目的是在Markdown中处理新行并将其视为硬折行。与GitHub-flavored Markdown一样，这个扩展在支持Markdown的同时，也支持插件和扩展。

具体来说，这段代码定义了一个名为`BR_RE`的元字符常量，它表示一个新行符，用于指定何时在Markdown中应用新行。然后，它定义了一个名为`SubscribeTagInlineProcessor`的类，继承自`inlinepatterns.SubstituteTagInlineProcessor`类，用于在Markdown中插入或替换标签。

最后，它导入了`Extension`类，并在导入了该类的同层级模块后，将其实例化并注册为Markdown的扩展。


```py
"""
A Python-Markdown extension to treat newlines as hard breaks; like
GitHub-flavored Markdown does.

See the [documentation](https://Python-Markdown.github.io/extensions/nl2br)
for details.
"""

from __future__ import annotations

from . import Extension
from ..inlinepatterns import SubstituteTagInlineProcessor

BR_RE = r'\n'


```

这段代码定义了一个名为 Nl2BrExtension 的类，继承自 Extensions 类。这个类的成员函数 extendMarkdown() 在 Markdown 中添加了一个名为 "nl" 的 `SubstituteTagInlineProcessor`。

具体来说，这个 `SubstituteTagInlineProcessor` 会在 Markdown 中查找所有带 "nl" 标签的文本，并将它们中的 "nl" 替换为 "BR"。这样，在显示这些 Markdown 内容时，将用 "BR" 代替 "nl"。

另外，`makeExtension` 函数会接收一个字典中传递的所有参数，并将其传递给 `Nl2BrExtension` 类中。这样，你可以将这个类作为其他类的一个子类，并在需要的时候通过 `makeExtension` 函数来应用这个类。


```py
class Nl2BrExtension(Extension):

    def extendMarkdown(self, md):
        """ Add a `SubstituteTagInlineProcessor` to Markdown. """
        br_tag = SubstituteTagInlineProcessor(BR_RE, 'br')
        md.inlinePatterns.register(br_tag, 'nl', 5)


def makeExtension(**kwargs):  # pragma: no cover
    return Nl2BrExtension(**kwargs)

```

# `/markdown/markdown/extensions/sane_lists.py`

这段代码是一个Python-Markdown的可扩展列表扩展，旨在改变列表在Python-Markdown中的行为，使其更加符合程序员的要求。主要实现如下：

1. 导入所需的库。
2. 创建一个名为`sane_lists`的新类，该类继承自`list`类，覆盖了`__len__`和`__getitem__`方法，实现了列表的 behavior。
3. 实现特殊方法，例如`get_serialized_svg`，用于生成序列化的 SVG 图。
4. 将`sane_lists`类注册到`Python-Markdown`扩展的插件中，这样在创建新的列表时，就可以使用`sane_lists`类来设置列表的行为。
5. 生成了一个文档，描述了`sane_lists`的作用和使用方法。

使用 `sane_lists` 的作用是，使列表在渲染时，可以访问到其元素的一个更友好的接口。在这个接口中，可以使用`markdown` 库中常用的列表操作，例如列表解析、索引和切片等。通过这种方式，可以方便地使用列表的最佳实践，而无需担心在渲染时出现特殊的情况。


```py
# Sane List Extension for Python-Markdown
# =======================================

# Modify the behavior of Lists in Python-Markdown to act in a sane manor.

# See https://Python-Markdown.github.io/extensions/sane_lists
# for documentation.

# Original code Copyright 2011 [Waylan Limberg](http://achinghead.com)

# All changes Copyright 2011-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
```

这段代码是一个扩展列表（extension）的类，名为`ModifyTheBehaviorOfLists`。它通过修改列表的行为来使列表表现得更加理智。

具体来说，这段代码实现了以下几点：

1. 支持对列表进行修改：通过添加新的 `@modify_list` 注解，可以定义扩展函数，对列表进行修改。
2. 支持嵌套列表：通过添加 `@list_inspectable` 注解，可以定义扩展函数，对列表进行访问和修改。
3. 支持无限列表：通过添加 `@is_extension` 注解，可以定义扩展函数，对无限列表进行判断和处理。
4. 支持列表索引：通过添加 `@list_index_out` 注解，可以定义扩展函数，对列表的索引进行输出。

需要注意的是，该扩展列表仅作为示例，并不是一个生产用的模块。要使用该扩展列表，需要将其导入并将其注册到`Python-Markdown`中。


```py
Modify the behavior of Lists in Python-Markdown to act in a sane manor.

See [documentation](https://Python-Markdown.github.io/extensions/sane_lists)
for details.
"""

from __future__ import annotations

from . import Extension
from ..blockprocessors import OListProcessor, UListProcessor
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .. import blockparser


```

这段代码定义了一个类SaneOListProcessor，它继承自类SIBLING_TAGS和类UListProcessor。

在SaneOListProcessor中，除了重写了SIBLING_TAGS为['ul']之外，还重写了LAZY_OL为False。

SIBLING_TAGS定义了是否包括`ul`，而LAZY_OL定义了是否延迟加载列表元素。

具体来说，这段代码的作用是：

1. 将SIBLING_TAGS更改为['ul']，以便不包括`ol`。
2. 将LAZY_OL更改为False，以禁用延迟加载列表元素。
3. 在__init__方法中，初始化OListProcessor类的实例，并设置CHILD_RE为用于解析儿童行的正则表达式，以适应新行为。


```py
class SaneOListProcessor(OListProcessor):
    """ Override `SIBLING_TAGS` to not include `ul` and set `LAZY_OL` to `False`. """

    SIBLING_TAGS = ['ol']
    """ Exclude `ul` from list of siblings. """
    LAZY_OL = False
    """ Disable lazy list behavior. """

    def __init__(self, parser: blockparser.BlockParser):
        super().__init__(parser)
        self.CHILD_RE = re.compile(r'^[ ]{0,%d}((\d+\.))[ ]+(.*)' %
                                   (self.tab_length - 1))


class SaneUListProcessor(UListProcessor):
    """ Override `SIBLING_TAGS` to not include `ol`. """

    SIBLING_TAGS = ['ul']
    """ Exclude `ol` from list of siblings. """

    def __init__(self, parser: blockparser.BlockParser):
        super().__init__(parser)
        self.CHILD_RE = re.compile(r'^[ ]{0,%d}(([*+-]))[ ]+(.*)' %
                                   (self.tab_length - 1))


```

这段代码定义了一个名为 "SaneListExtension" 的类，继承自 "Extension"，用于将 "sane lists"（可能是 Github Flavored Markdown 中的列表项）添加到 Markdown 中。

具体来说，这段代码实现了以下两个方法：

1. "extendMarkdown"，该方法直接重写了 "Processors" 接口，注册了一个名为 "SaneOListProcessor" 和 "SaneUListProcessor" 的处理器。其中，"SaneOListProcessor" 将 "olist" 类型的Block 处理器注册为 "40"，而 "SaneUListProcessor" 将 "ulist" 类型的Block 处理器注册为 "30"。这样，在 Markdown 中，每当遇到 "olist" 和 "ulist" 标签时，处理器就会自动将 "sane lists" 添加到相应的 Markdown 节点中。
2. "makeExtension"，该方法将 "SaneListExtension" 类作为 extension 创建，以便在需要时进行注册。


```py
class SaneListExtension(Extension):
    """ Add sane lists to Markdown. """

    def extendMarkdown(self, md):
        """ Override existing Processors. """
        md.parser.blockprocessors.register(SaneOListProcessor(md.parser), 'olist', 40)
        md.parser.blockprocessors.register(SaneUListProcessor(md.parser), 'ulist', 30)


def makeExtension(**kwargs):  # pragma: no cover
    return SaneListExtension(**kwargs)

```

# `/markdown/markdown/extensions/smarty.py`

这段代码是一个Smarty扩展，用于将Python-Markdown中的ASCII破折号、引号和省略号转换为HTML实体等效。它实现了对Python-Markdown的扩展，使得用户可以使用Markdown语法来更方便地格式化Python代码。

具体来说，这段代码实现了以下功能：

1. 定义了Smarty扩展的配置文件（config.py）路径，并在其中定义了变量：`ASCII_DASHES`，`ASCII_QUOTES` 和 `ASCII_ELOPS`。
2. 在`config.py`中定义了一系列的函数，包括：`make_ascii_dashes`，`make_ascii_quotes` 和 `make_ascii_ellipsis`。这些函数可以将Markdown中的相应部分转换为ASCII实体等效的HTML标记。
3. 在`config.py`中定义了一个`Author`变量，其值为2013年12月12日，作者为`Dmitry Shachnev`（@mitya57@gmail.com）。
4. 在`config.py`中定义了一个`License`变量，其值为BSD授权，对应的PHP文件路径为`LICENSE.txt`。

简而言之，这段代码定义了一个Smarty扩展，用于将Markdown中的ASCII破折号、引号和省略号转换为ASCII实体等效的HTML标记。这使得用户可以在Python-Markdown中更方便地格式化代码。


```py
# Smarty extension for Python-Markdown
# ====================================

# Adds conversion of ASCII dashes, quotes and ellipses to their HTML
# entity equivalents.

# See https://Python-Markdown.github.io/extensions/smarty
# for documentation.

# Author: 2013, Dmitry Shachnev <mitya57@gmail.com>

# All changes Copyright 2013-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

```

这段代码是一个名为 SmartyPants 的包的版权声明，告诉用户可以使用、修改和分发这个包，但必须遵守以下限制条件：

- 源代码和二进制形式的 SmartyPants 包的版权保留。
- 如果用户想要 redistribute（分发）源代码，则必须保留版权通知，这个列表的条件，以及一个名为 " SmartyPants 版权声明" 的说明。
- 如果用户想要在二进制形式上 redistribute，则必须复制上述版权通知、这个列表的条件，以及一个名为 " SmartyPants 版权声明" 的说明，到一个名为 " SmartyPants 说明文档" 的文档中，或者在文档中包含这些说明。


```py
# SmartyPants license:

#    Copyright (c) 2003 John Gruber <https://daringfireball.net/>
#    All rights reserved.

#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:

#    *  Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#    *  Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
```

这段代码是一个名为“distribution.py”的Python文件。它包含一个声明，指出该文件中包含的代码是版权持有者和贡献者的作品，允许在不限于特定事先书面许可的情况下自由地使用或修改该代码，但是任何由此产生的保证或陈述都不能被视为保证或陈述。

进一步指出，由于该代码是作为一个软件Distribution的源代码，因此该代码中包含的可能是为了商业目的而开发的产品，这些产品与Distribution的软件名称或 contributors名称无关。


```py
#       distribution.

#    *  Neither the name "SmartyPants" nor the names of its contributors
#       may be used to endorse or promote products derived from this
#       software without specific prior written permission.

#    This software is provided by the copyright holders and contributors "as
#    is" and any express or implied warranties, including, but not limited
#    to, the implied warranties of merchantability and fitness for a
#    particular purpose are disclaimed. In no event shall the copyright
#    owner or contributors be liable for any direct, indirect, incidental,
#    special, exemplary, or consequential damages (including, but not
#    limited to, procurement of substitute goods or services; loss of use,
#    data, or profits; or business interruption) however caused and on any
#    theory of liability, whether in contract, strict liability, or tort
```

这段代码是一个Python定义， 不能不说非常重大警告。它表示尽管在使用此软件时可能会发生疏忽或其它问题，但无论如何都会产生任何损害时，该软件不会为此承担责任。

具体来说，这段代码是一个 derivative work of SmartyPants 软件，它允许用户在 SmartyPants 警告下使用或 distribution 该软件，前提是不要修改该软件或散布其修改后的软件。


```py
#    (including negligence or otherwise) arising in any way out of the use
#    of this software, even if advised of the possibility of such damage.


# `smartypants.py` license:

#    `smartypants.py` is a derivative work of SmartyPants.
#    Copyright (c) 2004, 2007 Chad Miller <http://web.chad.org/>

#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:

#    *  Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

```

这段代码是一个名为`RedistributionsInBinaryForm.py`的Python脚本。它主要目的是确保二进制形式的软件分发在重新发布时包含上述版权通知、使用说明和/或其他提供的文档中。这意味着，当软件分发给用户时，它将遵循所提供的二进制形式，而不是可执行文件。

具体来说，这段代码定义了一个名为`RedistributionsInBinaryForm`的类。在这个类中，有以下方法：

1. `__init__`方法：该方法初始化一个列表，其中包括一个条件列表和一个免责声明。

2. `download_copyright_notice()`方法：该方法从用户处下载版权通知，并将其存储在`copyright_notice`变量中。

3. `get_download_link()`方法：该方法返回一个下载链接，用户可以通过点击该链接下载上述版权通知。

4. `is_lossy_installation()`方法：该方法用于检查是否可以通过安装程序安装软件。如果安装程序被认为是二进制形式，则该方法将返回`True`，否则将返回`False`。

5. `execute_极限出血（）`方法：该方法执行极限出血操作，这是一种在软件分发时删除锁定文件以避免侵权的手段。

6. `software_version()`方法：该方法返回软件的版本。


```py
#    *  Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.

#    This software is provided by the copyright holders and contributors "as
#    is" and any express or implied warranties, including, but not limited
#    to, the implied warranties of merchantability and fitness for a
#    particular purpose are disclaimed. In no event shall the copyright
#    owner or contributors be liable for any direct, indirect, incidental,
#    special, exemplary, or consequential damages (including, but not
#    limited to, procurement of substitute goods or services; loss of use,
#    data, or profits; or business interruption) however caused and on any
#    theory of liability, whether in contract, strict liability, or tort
#    (including negligence or otherwise) arising in any way out of the use
```

这段代码的作用是定义了一个名为`ASCIIDashExtension`的Python class，它实现了将ASCII级别的文本（包括冒号、引号和斜杠）转换为HTML实体的一种扩展。

具体来说，这个类的实现包括以下几个步骤：

1. 从文档中读取扩展的说明，即使用户被告知可能出现的损坏情况。
2. 定义了一个名为`__init__`的静态方法，用于初始化该扩展类的实例。
3. 从`pyx2`库中导入了一个名为`HtmlInlineProcessor`的类，用于将ASCII级别的文本转换为HTML实体。
4. 从`pyx2`库中导入了一个名为`InlineProcessor`的类，用于处理筒素级别（即单引号内的文本）。
5. 创建一个名为`ASCIIDashExtension`的类，继承自`Extension`类，实现了`register`和`run`方法。

具体来说，`register`方法用于注册该扩展，并将其添加到`pyx2.scanner.INITIAL_LINE_NAMES`列表中。`run`方法用于在用户运行Python时执行该扩展，首先会检查当前是否已经安装了`pyx2`库，如果没有安装，则安装并注册扩展；如果已经安装，则读取`pyx2.scanner.INITIAL_LINE_NAMES`列表，并将其中包含的名称添加到该列表中。在运行过程中，每当检测到用户输入的文本包含ASCII级别的单元格时，就会将其转换为HTML实体，并将其插入到当前行的开始位置。


```py
#    of this software, even if advised of the possibility of such damage.

"""
Adds conversion of ASCII dashes, quotes and ellipses to their HTML
entity equivalents.

See the [documentation](https://Python-Markdown.github.io/extensions/smarty)
for details.
"""

from __future__ import annotations

from . import Extension
from ..inlinepatterns import HtmlInlineProcessor, HTML_RE
from ..treeprocessors import InlineProcessor
```

这段代码定义了一个Registry，包含了多个函数，用于处理Markdown中的引用。Registry是Python的一个标准库，用于在应用程序中管理配置文件。

函数Registry中定义了以下函数：

- openquotes: 这个函数的目的是匹配Markdown中的引用，主要作用是检查它们是否使用了正确的语法。
- closequotes: 这个函数的目的是匹配Markdown中的引用，主要作用是检查它们是否使用了正确的语法。
- reference: 这个函数的目的是获取Markdown中的引用，主要作用是解析引用，并返回其解析后的引用对象。
- template: 这个函数的目的是生成Markdown中的引用，主要作用是解析引用，并返回其解析后的引用对象。

函数Registry中的函数使用了多个正则表达式和字符串处理函数，可以处理Markdown中的引用并且可以防止SQL注入攻击。


```py
from ..util import Registry
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown
    from .. import inlinepatterns
    import re
    import xml.etree.ElementTree as etree

# Constants for quote education.
punctClass = r"""[!"#\$\%'()*+,-.\/:;<=>?\@\[\\\]\^_`{|}~]"""
endOfWordClass = r"[\s.,;:!?)]"
closeClass = r"[^\ \t\r\n\[\{\(\-\u0002\u0003]"

openingQuotesBase = (
    r'(\s'               # a  whitespace char
    r'|&nbsp;'           # or a non-breaking space entity
    r'|--'               # or dashes
    r'|–|—'              # or Unicode
    r'|&[mn]dash;'       # or named dash entities
    r'|&#8211;|&#8212;'  # or decimal entities
    r')'
)

```

这段代码定义了一个名为 `substitutions` 的字典，包含了各种常用的数学公式和符号的替代词。这些替代词是为了在文本中引用数学公式和符号时，使得看起来更美观一些，同时避免因为语言的特殊性而产生的一些问题。

具体来说，这段代码定义了以下几个键：

* `mdash`：表示带斜杠的数学公式，例如 `mdash(x)` 可以替换为 `x`。
* `ndash`：表示带斜杠的数学公式，例如 `ndash(x)` 可以替换为 `x`。
* `ellipsis`：表示一个由星号组成的数学公式，例如 `ellipsis(x)` 可以替换为 `x`。
* `left-angle-quote`：表示一个带有左括号和引用的数学公式，例如 `left-angle-quote(x)` 可以替换为 `x`。
* `right-angle-quote`：表示一个带有右括号和引用的数学公式，例如 `right-angle-quote(x)` 可以替换为 `x`。
* `left-single-quote`：表示一个带有左括号和单引号的数学公式，例如 `left-single-quote(x)` 可以替换为 `x`。
* `right-single-quote`：表示一个带有右括号和单引号的数学公式，例如 `right-single-quote(x)` 可以替换为 `x`。
* `left-double-quote`：表示一个带有左括号和双引号的数学公式，例如 `left-double-quote(x)` 可以替换为 `x`。
* `right-double-quote`：表示一个带有右括号和双引号的数学公式，例如 `right-double-quote(x)` 可以替换为 `x`。

此外，代码还定义了一个特殊 case，即第一个字符是一个引号时，会使用 brute force 关闭引号。这意味着如果第一个字符是一个引号，即使它后面的字符不是句号、斜杠或单引号等任何特殊字符，`substitutions` 中的键仍然会被替换成指定的内容。


```py
substitutions = {
    'mdash': '&mdash;',
    'ndash': '&ndash;',
    'ellipsis': '&hellip;',
    'left-angle-quote': '&laquo;',
    'right-angle-quote': '&raquo;',
    'left-single-quote': '&lsquo;',
    'right-single-quote': '&rsquo;',
    'left-double-quote': '&ldquo;',
    'right-double-quote': '&rdquo;',
}


# Special case if the very first character is a quote
# followed by punctuation at a non-word-break. Close the quotes by brute force:
```

This code appears to be written in PowerShell and it is using regular expressions (regex) to find and capture certain patterns of text in a string.

The script defines two regular expressions, singleQuoteStartRe and doubleQuoteStartRe, which are used to capture the start of a double quote or a double quote with a leading space. The `%s` is a placeholder that represents a match anywhere in the string, and the `\B` is a positive lookahead assertion, which ensures that the match is not immediately followed by a double quote.

The code then defines two sets of rules for matching double quotes or single quotes with certain patterns, `doubleQuoteSetsRe` and `singleQuoteSetsRe`. These sets are likely used to store the matching strings that are being saved in variables or passed to other functions.

Finally, the code defines a regular expression `decadeAbbrRe` to capture century-specific abbreviations, such as "80s".

Without the specific context in which this code is being used, it's hard to know what it's doing and what kind of text it's matching.


```py
singleQuoteStartRe = r"^'(?=%s\B)" % punctClass
doubleQuoteStartRe = r'^"(?=%s\B)' % punctClass

# Special case for double sets of quotes, e.g.:
#   <p>He said, "'Quoted' words in a larger quote."</p>
doubleQuoteSetsRe = r""""'(?=\w)"""
singleQuoteSetsRe = r"""'"(?=\w)"""

# Special case for decade abbreviations (the '80s):
decadeAbbrRe = r"(?<!\w)'(?=\d{2}s)"

# Get most opening double quotes:
openingDoubleQuotesRegex = r'%s"(?=\w)' % openingQuotesBase

# Double closing quotes:
```

这段代码的作用是使用正则表达式匹配文本中的单引号和双引号，以便从给定的输入文本中提取最具代表性的引号。正则表达式中的第一个表达式匹配单引号，第二个表达式匹配双引号。

具体来说，代码首先定义了两个正则表达式：closingDoubleQuotesRegex 和 closingSingleQuotesRegex。这两个正则表达式都使用了全球通配符 % 表示，可以匹配字符串中的任意字符。

接着，代码通过调用 get_most_opening_single_quotes() 函数来获取给定文本中的最具代表性的单引号。这个函数的作用是查找文本中第一个单引号，并返回其正则表达式。

接下来，代码通过调用 get_all_remaining_quotes() 函数来获取给定文本中除最具代表性单引号外的所有剩余引号。这个函数的作用是查找文本中最后一个单引号，并返回其正则表达式。

最后，代码定义了一个 HTML_STRICT_RE 常量，表示 HTML5 语义为严格模式。

总之，这段代码的作用是提取给定文本中的最具代表性的单引号和双引号，以便进行正则表达式匹配。


```py
closingDoubleQuotesRegex = r'"(?=\s)'
closingDoubleQuotesRegex2 = '(?<=%s)"' % closeClass

# Get most opening single quotes:
openingSingleQuotesRegex = r"%s'(?=\w)" % openingQuotesBase

# Single closing quotes:
closingSingleQuotesRegex = r"(?<=%s)'(?!\s|s\b|\d)" % closeClass
closingSingleQuotesRegex2 = r"'(\s|s\b)"

# All remaining quotes should be opening ones
remainingSingleQuotesRegex = r"'"
remainingDoubleQuotesRegex = r'"'

HTML_STRICT_RE = HTML_RE + r'(?!\>)'


```

这段代码定义了一个名为 "SubstituteTextPattern" 的类，该类继承自 "HtmlInlineProcessor" 类。

该类的初始化方法 "__init__" 接受一个字符串 "pattern" 和一个可变元组 "replace"（包含 int、str 或 etree.Element 类型），以及一个 "md" 的实例。

该类的 "handleMatch" 方法接收一个匹配对象 "m" 和匹配到的字符串 "data"。该方法返回一个元组 "result"（包含匹配到的字符串）、匹配的起始位置 "start" 和结束位置 "end"。

该类的 "replace" 变量是一个可变元组，其中包含要替换的匹配项。对于每个匹配项，如果它是整数，该匹配项将替换为 "replace[i]" 中的字符串；否则，该匹配项将使用 "self.md.htmlStash.store" 方法将其作为 "pattern" 的附加部分与 "md" 中的内容连接起来。


```py
class SubstituteTextPattern(HtmlInlineProcessor):
    def __init__(self, pattern: str, replace: Sequence[int | str | etree.Element], md: Markdown):
        """ Replaces matches with some text. """
        HtmlInlineProcessor.__init__(self, pattern)
        self.replace = replace
        self.md = md

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        result = ''
        for part in self.replace:
            if isinstance(part, int):
                result += m.group(part)
            else:
                result += self.md.htmlStash.store(part)
        return result, m.start(0), m.end(0)


```

This is a PyInstaller script that installs the Quill and Prisma离子 products.

Quill is an experimental writing tool that allows you to write Markdown using plain text. It includes syntax highlighting for Markdown elements, which can help you write more efficiently.

Prisma is a tool that can be used to generate the equivalent CSS of a Markdown document. This can be useful for visualizing the formatting of your Markdown document.

The script installs these packages and sets the necessary environment variables for them to work correctly.


```py
class SmartyExtension(Extension):
    """ Add Smarty to Markdown. """
    def __init__(self, **kwargs):
        self.config = {
            'smart_quotes': [True, 'Educate quotes'],
            'smart_angled_quotes': [False, 'Educate angled quotes'],
            'smart_dashes': [True, 'Educate dashes'],
            'smart_ellipses': [True, 'Educate ellipses'],
            'substitutions': [{}, 'Overwrite default substitutions'],
        }
        """ Default configuration options. """
        super().__init__(**kwargs)
        self.substitutions: dict[str, str] = dict(substitutions)
        self.substitutions.update(self.getConfig('substitutions', default={}))

    def _addPatterns(
        self,
        md: Markdown,
        patterns: Sequence[tuple[str, Sequence[int | str | etree.Element]]],
        serie: str,
        priority: int,
    ):
        for ind, pattern in enumerate(patterns):
            pattern += (md,)
            pattern = SubstituteTextPattern(*pattern)
            name = 'smarty-%s-%d' % (serie, ind)
            self.inlinePatterns.register(pattern, name, priority-ind)

    def educateDashes(self, md: Markdown) -> None:
        emDashesPattern = SubstituteTextPattern(
            r'(?<!-)---(?!-)', (self.substitutions['mdash'],), md
        )
        enDashesPattern = SubstituteTextPattern(
            r'(?<!-)--(?!-)', (self.substitutions['ndash'],), md
        )
        self.inlinePatterns.register(emDashesPattern, 'smarty-em-dashes', 50)
        self.inlinePatterns.register(enDashesPattern, 'smarty-en-dashes', 45)

    def educateEllipses(self, md: Markdown) -> None:
        ellipsesPattern = SubstituteTextPattern(
            r'(?<!\.)\.{3}(?!\.)', (self.substitutions['ellipsis'],), md
        )
        self.inlinePatterns.register(ellipsesPattern, 'smarty-ellipses', 10)

    def educateAngledQuotes(self, md: Markdown) -> None:
        leftAngledQuotePattern = SubstituteTextPattern(
            r'\<\<', (self.substitutions['left-angle-quote'],), md
        )
        rightAngledQuotePattern = SubstituteTextPattern(
            r'\>\>', (self.substitutions['right-angle-quote'],), md
        )
        self.inlinePatterns.register(leftAngledQuotePattern, 'smarty-left-angle-quotes', 40)
        self.inlinePatterns.register(rightAngledQuotePattern, 'smarty-right-angle-quotes', 35)

    def educateQuotes(self, md: Markdown) -> None:
        lsquo = self.substitutions['left-single-quote']
        rsquo = self.substitutions['right-single-quote']
        ldquo = self.substitutions['left-double-quote']
        rdquo = self.substitutions['right-double-quote']
        patterns = (
            (singleQuoteStartRe, (rsquo,)),
            (doubleQuoteStartRe, (rdquo,)),
            (doubleQuoteSetsRe, (ldquo + lsquo,)),
            (singleQuoteSetsRe, (lsquo + ldquo,)),
            (decadeAbbrRe, (rsquo,)),
            (openingSingleQuotesRegex, (1, lsquo)),
            (closingSingleQuotesRegex, (rsquo,)),
            (closingSingleQuotesRegex2, (rsquo, 1)),
            (remainingSingleQuotesRegex, (lsquo,)),
            (openingDoubleQuotesRegex, (1, ldquo)),
            (closingDoubleQuotesRegex, (rdquo,)),
            (closingDoubleQuotesRegex2, (rdquo,)),
            (remainingDoubleQuotesRegex, (ldquo,))
        )
        self._addPatterns(md, patterns, 'quotes', 30)

    def extendMarkdown(self, md):
        configs = self.getConfigs()
        self.inlinePatterns: Registry[inlinepatterns.InlineProcessor] = Registry()
        if configs['smart_ellipses']:
            self.educateEllipses(md)
        if configs['smart_quotes']:
            self.educateQuotes(md)
        if configs['smart_angled_quotes']:
            self.educateAngledQuotes(md)
            # Override `HTML_RE` from `inlinepatterns.py` so that it does not
            # process tags with duplicate closing quotes.
            md.inlinePatterns.register(HtmlInlineProcessor(HTML_STRICT_RE, md), 'html', 90)
        if configs['smart_dashes']:
            self.educateDashes(md)
        inlineProcessor = InlineProcessor(md)
        inlineProcessor.inlinePatterns = self.inlinePatterns
        md.treeprocessors.register(inlineProcessor, 'smarty', 2)
        md.ESCAPED_CHARS.extend(['"', "'"])


```

这段代码定义了一个名为 `makeExtension` 的函数，它接收一个或多个参数 `**kwargs`，并将其传递给名为 `SmartyExtension` 的函数，这个函数会使用 `**kwargs` 中的所有键值对，返回一个经过扩展的 `SmartyExtension` 对象。

具体来说，这段代码使用的是 Python 的 pragma 注解，这是一种简写的函数定义方式，可以帮助人们在不写下大量注释的情况下定义函数。 pragma 注解中的 `**kwargs` 表示这个函数接受一个或多个 keyword arguments，也就是带参数的参数。这些 keyword arguments 会被转化为普通参数，传递给 `SmartyExtension` 函数，并且可以通过 `**kwargs` 的方式重复使用这些参数。

这段代码还使用了 `no cover` 注解，这是一种防止函数被继承的技巧，这里使用的是 `**kwargs` 中的所有参数都会被覆盖，也就是 `**kwargs` 中的所有参数都不会被传递给 `SmartyExtension` 函数。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return SmartyExtension(**kwargs)

```