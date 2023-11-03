# PythonMarkdown源码解析 6

# `/markdown/markdown/postprocessors.py`

该代码是一个Python实现John Gruber所写的Markdown（Markdown的Python实现）的示例。它将Markdown的语法解析为文本，并将其转换为Python可以理解的语法树。通过使用BeautifulSoup库，它能够提供更加方便的Markdown解析。

该代码最初由Mefrid动手实现，之后由一系列的维护者共同维护。当前的维护者包括Waylan Limberg、Dmitry Shachnev和Isaac Muse。

该代码已经被发布到了PyPI上，并且GitHub上也有相应的仓库。


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

这段代码定义了一个名为 `postprocessors` 的类，其作用是在将文本序列化为字符串后执行一系列操作。具体来说，这个类定义了一个字符串类 `Postprocessors` 继承自 `Postprocessors` 类，该类包含了一个方法 `run_postprocessors`。

`run_postprocessors` 方法接收一个字符串参数，该参数描述了要执行的后处理操作。该方法返回一个字符串，其中包含后处理操作的结果。这个方法使用了 `__future__` 注解，以便在未来的版本中更改方法的行为。

该代码的目的是提供一个基类，以便其他类可以继承并添加自己的后处理操作。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""

Post-processors run on the text of the entire document after is has been serialized into a string.
Postprocessors should be used to work with the text just before output. Usually, they are used add
back sections that were extracted in a preprocessor, fix up outgoing encodings, or wrap the whole
document.

"""

from __future__ import annotations

```

这段代码使用了多个Python库，包括collections、typing和re。它实现了两个函数，分别是build_postprocessors和Postprocessor。这两个函数的作用如下：

1. build_postprocessors函数：

这个函数接收一个Markdown对象（md）以及任何 keyword arguments（关键字参数），然后返回一个Registry（注册表）类型的Postprocessor对象。它通过register函数来注册Postprocessor类实例。在这个register函数中，函数注册了一个RawHtmlPostprocessor实例，使用了Markdown的RawHtmlPostprocessor类，这个类将原始的Markdown内容作为输入，并返回一个经过处理的Markdown内容。然后函数还注册了一个AndSubstitutePostprocessor实例，使用了AndSubstitutePostprocessor类，这个类将在Markdown内容中替换掉某些关键字。

2. Postprocessor类：

Postprocessor类是这段代码中定义的一个类，它继承了util.Registry类，并实现了postprocess_content和postprocess_title两个方法。其中，postprocess_content方法用于处理Markdown内容，而postprocess_title方法用于处理文章标题。

postprocess_content方法的实现比较简单，直接将Markdown内容作为输入，并返回经过处理的内容。具体来说，它首先创建一个util.Registry对象，然后使用register方法将RawHtmlPostprocessor实例注册到Registry中，最后将Registry对象返回。

postprocess_title方法的实现比较复杂，它将在Markdown内容之前和之后创建两个util.Registry对象，分别用于处理文章标题和内容。它还使用re.sub方法来替换掉文章标题和内容中的所有关键字，然后将替换后的内容作为输入，并注册到Registry中。具体来说，它首先创建一个util.Registry对象，然后使用register方法将util.registry.py`文件中的util.Registry实例注册到Registry中，这个文件中定义了util.Registry的实现，与postprocess_title方法中的行为类似。然后它又定义了一个方法`replace_content`，用于替换文章标题和内容中的所有关键字。这个方法首先创建一个util.Registry对象，然后使用register方法将util.registry.py`文件中的util.Registry实例注册到Registry中，这个文件中定义了util.Registry的实现，与postprocess_title方法中的行为类似。最后它调用replace_content方法，将替换后的内容作为输入，并注册到Registry中。


```py
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from . import util
import re

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


def build_postprocessors(md: Markdown, **kwargs: Any) -> util.Registry[Postprocessor]:
    """ Build the default postprocessors for Markdown. """
    postprocessors = util.Registry()
    postprocessors.register(RawHtmlPostprocessor(md), 'raw_html', 30)
    postprocessors.register(AndSubstitutePostprocessor(), 'amp_substitute', 20)
    return postprocessors


```

这段代码定义了一个名为 `Postprocessor` 的类，继承自 `util.Processor` 类(可能存在，但并没有在代码中提供更多的信息)。

这个 `Postprocessor` 类有一个 `run` 方法，这个方法接受一个 HTML 文档作为输入参数，然后对其进行修改并返回一个新的 HTML 文档。

每个 `Postprocessor` 类都必须实现 `run` 方法，因此如果想要创建一个新的 `Postprocessor` 类，只需要继承 `Postprocessor` 类并实现 `run` 方法即可。

由于 `run` 方法中包含了一些不想输出细节，因此没有提供更多关于这个函数的详细信息。


```py
class Postprocessor(util.Processor):
    """
    Postprocessors are run after the ElementTree it converted back into text.

    Each Postprocessor implements a `run` method that takes a pointer to a
    text string, modifies it as necessary and returns a text string.

    Postprocessors must extend `Postprocessor`.

    """

    def run(self, text: str) -> str:
        """
        Subclasses of `Postprocessor` should implement a `run` method, which
        takes the html document as a single text string and returns a
        (possibly modified) string.

        """
        pass  # pragma: no cover


```

This is a class that appears to parse and substitute placeholders in a Markdown text string. It uses the MD (Markdown) frontend for this purpose. The class has methods for running the MD through a stack for block-level parsing, as well as a method for stashing text to a position in the stack, substitute placeholders in a text string, and return the processed text after running it through the stack.

The class also has a method called `isblocklevel`, that checks if a given piece of text is a block-level Markdown block, such as a `<p>` tag, by checking for certain special characters.

The `stash_to_string` method takes a stashed text and converts it to a string, which is then passed to the `run` method.

The `run` method takes a stashed text and runs it through a stack for block-level parsing. It uses a regular expression to match the text and a function `substitute_match` to replace placeholders in the text with their corresponding text.

It is important to notice that the stash_to_string and run method should be adjusted according to the stash\_to\_string method documentation.


```py
class RawHtmlPostprocessor(Postprocessor):
    """ Restore raw html to the document. """

    BLOCK_LEVEL_REGEX = re.compile(r'^\<\/?([^ >]+)')

    def run(self, text: str) -> str:
        """ Iterate over html stash and restore html. """
        replacements = OrderedDict()
        for i in range(self.md.htmlStash.html_counter):
            html = self.stash_to_string(self.md.htmlStash.rawHtmlBlocks[i])
            if self.isblocklevel(html):
                replacements["<p>{}</p>".format(
                    self.md.htmlStash.get_placeholder(i))] = html
            replacements[self.md.htmlStash.get_placeholder(i)] = html

        def substitute_match(m: re.Match[str]) -> str:
            key = m.group(0)

            if key not in replacements:
                if key[3:-4] in replacements:
                    return f'<p>{ replacements[key[3:-4]] }</p>'
                else:
                    return key

            return replacements[key]

        if replacements:
            base_placeholder = util.HTML_PLACEHOLDER % r'([0-9]+)'
            pattern = re.compile(f'<p>{ base_placeholder }</p>|{ base_placeholder }')
            processed_text = pattern.sub(substitute_match, text)
        else:
            return text

        if processed_text == text:
            return processed_text
        else:
            return self.run(processed_text)

    def isblocklevel(self, html: str) -> bool:
        """ Check is block of HTML is block-level. """
        m = self.BLOCK_LEVEL_REGEX.match(html)
        if m:
            if m.group(1)[0] in ('!', '?', '@', '%'):
                # Comment, PHP etc...
                return True
            return self.md.is_block_level(m.group(1))
        return False

    def stash_to_string(self, text: str) -> str:
        """ Convert a stashed object to a string. """
        return str(text)


```

这两段代码定义了一个名为 `AndSubstitutePostprocessor` 的类和一个名为 `UnescapePostprocessor` 的类。它们都属于一个名为 `Postprocessor` 的接口。

`AndSubstitutePostprocessor` 的实现类重写了 `run` 方法，该方法接收一个字符串参数 `text`，并使用 `util.AMP_SUBSTITUTE` 函数来替换字符中的 `&` 符号，然后返回修改后的字符串。

`UnescapePostprocessor` 的实现类重写了 `run` 方法，该方法接收一个字符串参数 `text`，并使用正则表达式 `util.STX` 和 `util.ETX` 来查找字符串中的特殊字符 `&`，然后使用 `chr` 函数将其转换为 ASCII 码，最后返回修改后的字符串。

两段代码的作用是用于对文本进行处理，特别是对 `&` 符号进行替换，使得在某些报文背景下仍然可读取的内容。`AndSubstitutePostprocessor` 主要关注于处理在 `util.STX` 和 `util.ETX` 控制下的字符，而 `UnescapePostprocessor` 则专注于处理这些字符。


```py
class AndSubstitutePostprocessor(Postprocessor):
    """ Restore valid entities """

    def run(self, text: str) -> str:
        text = text.replace(util.AMP_SUBSTITUTE, "&")
        return text


@util.deprecated(
    "This class is deprecated and will be removed in the future; "
    "use [`UnescapeTreeprocessor`][markdown.treeprocessors.UnescapeTreeprocessor] instead."
)
class UnescapePostprocessor(Postprocessor):
    """ Restore escaped chars. """

    RE = re.compile(r'{}(\d+){}'.format(util.STX, util.ETX))

    def unescape(self, m: re.Match[str]) -> str:
        return chr(int(m.group(1)))

    def run(self, text: str) -> str:
        return self.RE.sub(self.unescape, text)

```

# `/markdown/markdown/preprocessors.py`

该代码是一个Python实现John Gruber所写的Markdown的示例。Markdown是一种轻量级的标记语言，它可以让你以类似纯文本的方式编写HTML内容，让许多人易于阅读和编写。

该代码使用了Python的Markdown库，它是一个Python的实现，而不是一个Wiki的页面。它通过将Markdown代码转换为HTML代码来呈现Markdown内容，然后在浏览器中将其渲染为可视化HTML。

该代码的作用是提供一个Python的Markdown实现，以便那些希望使用Python来编写Markdown内容的人。


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

这段代码定义了一个名为 `preprocessors` 的类，该类使用 Manfred Stienstra 的版权，并在其内部源代码的顶部添加了 BSD 授权。

该类包含一个 `preprocess` 方法，用于对输入文本进行预处理，包括去除不良字符和提取需要进一步处理的文本片段。

该代码的目的是提供一个通用的字符串预处理函数，该函数可以在解析器中处理可能因为包含不良字符而产生问题的字符串。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
Preprocessors work on source text before it is broken down into its individual parts.
This is an excellent place to clean up bad characters or to extract portions for later
processing that the parser may otherwise choke on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from . import util
from .htmlparser import HTMLExtractor
```

这段代码是一个Python库中的函数，用于创建一个注册表，将Markdown中的预处理程序(preprocessor)注册到该注册表中。这个函数接受一个名为md的Markdown对象，以及多个 keyword arguments，其中任何都是可选的。

如果`TYPE_CHECKING`是一个声明(声明时使用`pragma: no cover`)，则不会产生任何输出，否则会输出以下消息：


Warning: 修复了Markdown的警告：在使用痔疮之前，要覆盖__init__函数)


这是因为该库在它被使用时，会覆盖`__init__`函数，从而允许在需要时动态地注册预处理程序。

该函数的实现很简单：只是创建了一个名为`preprocessors`的注册表，并注册了两个预处理程序。具体来说，它注册了一个名为`NormalizeWhitespace`的预处理程序，使用键`'normalize_whitespace'`和值`30`进行注册，以及注册了一个名为`HtmlBlockPreprocessor`的预处理程序，使用键`'html_block'`和值`20`进行注册。

这两个预处理程序的实现都很简单。`NormalizeWhitespace`只是一个简单的函数，只是去掉了一些白色的空格，而`HtmlBlockPreprocessor`也是一个简单的函数，只是简单地将Markdown中的块的起始和结束标记换行。

该函数返回一个名为`preprocessors`的注册表，用于将预处理程序传递给Markdown的`run`方法。

该函数的实现中，有一个`if TYPE_CHECKING`语句，如果没有这个语句，则不会产生任何输出，否则会输出一个警告消息。


```py
import re

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


def build_preprocessors(md: Markdown, **kwargs: Any) -> util.Registry[Preprocessor]:
    """ Build and return the default set of preprocessors used by Markdown. """
    preprocessors = util.Registry()
    preprocessors.register(NormalizeWhitespace(md), 'normalize_whitespace', 30)
    preprocessors.register(HtmlBlockPreprocessor(md), 'html_block', 20)
    return preprocessors


class Preprocessor(util.Processor):
    """
    Preprocessors are run after the text is broken into lines.

    Each preprocessor implements a `run` method that takes a pointer to a
    list of lines of the document, modifies it as necessary and returns
    either the same pointer or a pointer to a new list.

    Preprocessors must extend `Preprocessor`.

    """
    def run(self, lines: list[str]) -> list[str]:
        """
        Each subclass of `Preprocessor` should override the `run` method, which
        takes the document as a list of strings split by newlines and returns
        the (possibly modified) list of lines.

        """
        pass  # pragma: no cover


```



这段代码定义了两个类，一个是`NormalizeWhitespace`，另一个是`HtmlBlockPreprocessor`。两个类都是`Preprocessor`类的子类，说明它们可以从 lines 列表中提取出我们需要进行预处理的内容。

具体来说，`NormalizeWhitespace`类实现了将所有 whitespace 转义，去除换行符，以及在 STX 和 ETX 之外添加一个空格，使得所有内容可以被正确地匹配。然后将其余的内容进行扩展制表，去除多余的空格，并将内容分割为行。

`HtmlBlockPreprocessor`类实现了从所有需要去除的 HTML 标签中提取出块内容，并将内容存储在`htmlStash`对象中。这个对象将会在运行时用来确保所有匹配的 HTML 标签都会被正确地提取出来，并在后续的代码中进行使用。

在这两个类中，`util.HTMLExtractor`和`re`是用于实现预处理的函数。


```py
class NormalizeWhitespace(Preprocessor):
    """ Normalize whitespace for consistent parsing. """

    def run(self, lines: list[str]) -> list[str]:
        source = '\n'.join(lines)
        source = source.replace(util.STX, "").replace(util.ETX, "")
        source = source.replace("\r\n", "\n").replace("\r", "\n") + "\n\n"
        source = source.expandtabs(self.md.tab_length)
        source = re.sub(r'(?<=\n) +\n', '\n', source)
        return source.split('\n')


class HtmlBlockPreprocessor(Preprocessor):
    """
    Remove html blocks from the text and store them for later retrieval.

    The raw HTML is stored in the [`htmlStash`][markdown.util.HtmlStash] of the
    [`Markdown`][markdown.Markdown] instance.
    """

    def run(self, lines: list[str]) -> list[str]:
        source = '\n'.join(lines)
        parser = HTMLExtractor(self.md)
        parser.feed(source)
        parser.close()
        return ''.join(parser.cleandoc).split('\n')

```

# `/markdown/markdown/serializers.py`

这段代码的作用是向 `Elementree` 库中添加了 HTML 序列化功能。它从 `ElementTree` 1.3 预览版中稍作修改来实现这一目的。具体来说，它实现了 HTML 元素与 JavaScript 对象之间的映射，使得在 JavaScript 中更容易地操作 DOM 元素。


```py
# Add x/html serialization to `Elementree`
# Taken from ElementTree 1.3 preview with slight modifications
#
# Copyright (c) 1999-2007 by Fredrik Lundh.  All rights reserved.
#
# fredrik@pythonware.com
# https://www.pythonware.com/
#
# --------------------------------------------------------------------
# The ElementTree toolkit is
#
# Copyright (c) 1999-2007 by Fredrik Lundh
#
# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
```

这段代码是一个带有版权声明的程序，它告知用户该软件可以免费使用，但需要在软件中包含版权声明和支持文档。用户还可以修改和分发该软件，前提是不要将软件用于商业目的或以任何方式侵犯版权。

版权声明在代码的第一行中明确告知。接下来的行说明了软件的使用限制，包括可以在哪些平台上使用、如何复制、修改和分发软件，以及需要包含的版权声明和声明者名称。

最后一行说明了如果用户想要将软件用于商业目的，则需要遵守特殊的许可条款。


```py
# and will comply with the following terms and conditions:
#
# Permission to use, copy, modify, and distribute this software and
# its associated documentation for any purpose and without fee is
# hereby granted, provided that the above copyright notice appears in
# all copies, and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of
# Secret Labs AB or the author not be used in advertising or publicity
# pertaining to distribution of the software without specific, written
# prior permission.
#
# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANT-
# ABILITY AND FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR
# BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
```

这段代码是一个Python类，名为`DAMAGES_WHATSOEVER_RESULTING_FROM_LOSS_OF_USE_DATA_OR_PROFITS_ACTION.` 它定义了两个函数，用于将Python中`ElementTree.Element`对象转换为HTML或XML格式的字符串。

具体来说，这两个函数分别将以下类型的元素对象转换为HTML或XML格式：

- `self-closing`标签（`<tag>` 和 `<tag />`）：在HTML中，这些标签会被转换为`<tag>`，而在XML中，它们会被转换为`<tag />`。
- `boolean`属性（`attrname` 和 `attrname="attrname"`）：在HTML中，这些属性被转换为`attrname`，而在XML中，它们被转换为`attrname="attrname"`。

这两个函数的实现是基于Python的Markdown库，它提供了两种不同的方式将`ElementTree.Element`对象转换为HTML或XML格式的字符串。


```py
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.
# --------------------------------------------------------------------

"""
Python-Markdown provides two serializers which render [`ElementTree.Element`][xml.etree.ElementTree.Element]
objects to a string of HTML. Both functions wrap the same underlying code with only a few minor
differences as outlined below:

1. Empty (self-closing) tags are rendered as `<tag>` for HTML and as `<tag />` for XHTML.
2. Boolean attributes are rendered as `attrname` for HTML and as `attrname="attrname"` for XHTML.
"""

```

这段代码的作用是定义了两个函数 `to_html_string` 和 `to_xhtml_string`，它们接受一个参数 `text`，并返回一个 HTML 格式的字符串。

具体来说，这段代码使用了许多来自 Python 和 XML 元字符的函数和常量，包括 `from __future__ import annotations`、`from xml.etree.ElementTree import ProcessingInstruction`、`from xml.etree.ElementTree import Comment`、`from xml.etree.ElementTree import Element`、`from xml.etree.ElementTree import QName` 和 `from typing import Callable, Literal, NoReturn`。

这两个函数的具体实现可能还有更多的细节，但可以从代码中得到以下信息：

- `to_html_string` 函数接受一个 HTML 标记，它会尝试解析这个标记并将其转换为 XML 格式。为此，它使用了 `from xml.etree.ElementTree import ProcessingInstruction` 和 `from xml.etree.ElementTree import Comment` 函数来获取当前标签的处理器指令和注释信息。

- `to_xhtml_string` 函数与 `to_html_string` 类似，但它使用了 `to_html_string` 函数之后解析生成的 XML 格式，而不是直接将文本转换为 XML 格式。为此，它使用了 `re.sub` 函数来查找所有 XML 标记（包括换行符），并使用 `re.sub` 函数的显式替换功能将其转换为 HTML 标记。


```py
from __future__ import annotations

from xml.etree.ElementTree import ProcessingInstruction
from xml.etree.ElementTree import Comment, ElementTree, Element, QName, HTML_EMPTY
import re
from typing import Callable, Literal, NoReturn

__all__ = ['to_html_string', 'to_xhtml_string']

RE_AMP = re.compile(r'&(?!(?:\#[0-9]+|\#x[0-9a-f]+|[0-9a-z]+);)', re.I)


def _raise_serialization_error(text: str) -> NoReturn:  # pragma: no cover
    raise TypeError(
        "cannot serialize {!r} (type {})".format(text, type(text).__name__)
        )


```

这段代码定义了一个名为 `_escape_cdata` 的函数，用于将字符串数据中的 &、< 和 > 符号替换为它们的 HTML 实体编码。

该函数接收一个字符串参数 `text`，首先使用正则表达式检查该字符串是否包含 &、< 和 > 符号。如果是，则使用 RE_AMP 库中的替换函数将 &、< 和 > 替换为它们的 HTML 实体编码。如果不是，函数将忽略这些符号。

如果函数在尝试执行替换操作时出现错误，例如字符串长度小于 500 或遇到无法访问的属性，则会跳转到错误处理程序。如果没有错误发生，函数返回原始字符串。


```py
def _escape_cdata(text) -> str:
    # escape character data
    try:
        # it's worth avoiding do-nothing calls for strings that are
        # shorter than 500 character, or so.  assume that's, by far,
        # the most common case in most applications.
        if "&" in text:
            # Only replace & when not part of an entity
            text = RE_AMP.sub('&amp;', text)
        if "<" in text:
            text = text.replace("<", "&lt;")
        if ">" in text:
            text = text.replace(">", "&gt;")
        return text
    except (TypeError, AttributeError):  # pragma: no cover
        _raise_serialization_error(text)


```

该函数名为 `_escape_attrib`，它接受一个字符串参数 `text`，并返回一个新的字符串。函数的作用是处理字符串中的特殊符号，将其转义，从而避免潜在的安全漏洞。

函数内部首先使用正则表达式 `RE_AMP` 来查找文本中的 &符号，然后根据符号的不同，执行相应的转义操作。如果找到某个符号，函数就会执行一次转义操作，然后尝试使用 `return` 语句返回转义后的字符串。

如果函数内部出现错误，如 `TypeError` 或 `AttributeError`，则会捕获并报错。


```py
def _escape_attrib(text: str) -> str:
    # escape attribute value
    try:
        if "&" in text:
            # Only replace & when not part of an entity
            text = RE_AMP.sub('&amp;', text)
        if "<" in text:
            text = text.replace("<", "&lt;")
        if ">" in text:
            text = text.replace(">", "&gt;")
        if "\"" in text:
            text = text.replace("\"", "&quot;")
        if "\n" in text:
            text = text.replace("\n", "&#10;")
        return text
    except (TypeError, AttributeError):  # pragma: no cover
        _raise_serialization_error(text)


```

这段代码定义了一个名为 `_escape_attrib_html` 的函数，它接受一个字符串参数 `text`。函数的作用是处理 HTML 实体，例如 `&` 和 `<` 标签，将它们替换为相应的实体名称，例如 `&amp;` 和 `&lt;`。

函数首先会检查传入的字符串是否包含 `&`、`<` 或 `>` 标签，如果是，则执行相应的replace函数，将标签替换为对应的实体名称。在函数内部，还使用了一个特殊的方法 `__escape_attrib_html`，如果该方法失败，则使用 `_raise_serialization_error` 函数抛出异常。


```py
def _escape_attrib_html(text: str) -> str:
    # escape attribute value
    try:
        if "&" in text:
            # Only replace & when not part of an entity
            text = RE_AMP.sub('&amp;', text)
        if "<" in text:
            text = text.replace("<", "&lt;")
        if ">" in text:
            text = text.replace(">", "&gt;")
        if "\"" in text:
            text = text.replace("\"", "&quot;")
        return text
    except (TypeError, AttributeError):  # pragma: no cover
        _raise_serialization_error(text)


```

This is a JavaScript function that takes an HTML tag, text to be serialized, and an HTML format. It determines whether the text has a namespace and serializes it accordingly. If the text does not have a namespace, it is either escaped or left as is. If the tag is a QName, it is interpreted as a URI template and the text is converted to a URI. If the tag is not a QName or the text does not have a namespace, it is left as is. The function returns the serialized HTML.


```py
def _serialize_html(write: Callable[[str], None], elem: Element, format: Literal["html", "xhtml"]) -> None:
    tag = elem.tag
    text = elem.text
    if tag is Comment:
        write("<!--%s-->" % _escape_cdata(text))
    elif tag is ProcessingInstruction:
        write("<?%s?>" % _escape_cdata(text))
    elif tag is None:
        if text:
            write(_escape_cdata(text))
        for e in elem:
            _serialize_html(write, e, format)
    else:
        namespace_uri = None
        if isinstance(tag, QName):
            # `QNAME` objects store their data as a string: `{uri}tag`
            if tag.text[:1] == "{":
                namespace_uri, tag = tag.text[1:].split("}", 1)
            else:
                raise ValueError('QName objects must define a tag.')
        write("<" + tag)
        items = elem.items()
        if items:
            items = sorted(items)  # lexical order
            for k, v in items:
                if isinstance(k, QName):
                    # Assume a text only `QName`
                    k = k.text
                if isinstance(v, QName):
                    # Assume a text only `QName`
                    v = v.text
                else:
                    v = _escape_attrib_html(v)
                if k == v and format == 'html':
                    # handle boolean attributes
                    write(" %s" % v)
                else:
                    write(' {}="{}"'.format(k, v))
        if namespace_uri:
            write(' xmlns="%s"' % (_escape_attrib(namespace_uri)))
        if format == "xhtml" and tag.lower() in HTML_EMPTY:
            write(" />")
        else:
            write(">")
            if text:
                if tag.lower() in ["script", "style"]:
                    write(text)
                else:
                    write(_escape_cdata(text))
            for e in elem:
                _serialize_html(write, e, format)
            if tag.lower() not in HTML_EMPTY:
                write("</" + tag + ">")
    if elem.tail:
        write(_escape_cdata(elem.tail))


```

这段代码定义了一个名为 `_write_html` 的函数，它接受一个根元素（`Element` 类型）和一个格式字符串（Literal 类型，可以是 "html" 或 "xhtml"）。函数的作用是将根元素及其子元素序列化为 HTML5 格式的字符串。

具体实现中，函数首先检查传入的根元素是否为 `None`，如果是，则返回一个空列表（`[]`）。接着定义一个名为 `data` 的列表，用于存储字符串，并定义一个名为 `write` 的函数，用于将列表中的字符串添加到数据列表中。

函数内部调用一个名为 `_serialize_html` 的内部函数，该函数接受两个参数：数据列表（已经包含了字符串列表）和格式字符串。格式字符串指定了将数据序列化为 HTML5 格式的字符串。

最后，函数创建一个字符串列表（`[]`），并调用 `write` 函数将数据添加到列表中，然后调用 `join` 函数将列表中的所有字符串连接成一个字符串。最终，函数返回这个字符串列表。

通过对 `to_html_string` 函数进行调用，可以获取指定元素及其子元素的 HTML5 格式字符串表示。


```py
def _write_html(root: Element, format: Literal["html", "xhtml"] = "html") -> str:
    assert root is not None
    data: list[str] = []
    write = data.append
    _serialize_html(write, root, format)
    return "".join(data)


# --------------------------------------------------------------------
# public functions


def to_html_string(element: Element) -> str:
    """ Serialize element and its children to a string of HTML5. """
    return _write_html(ElementTree(element).getroot(), format="html")


```

这段代码定义了一个名为 `to_xhtml_string` 的函数，用于将一个给定的元素及其子元素序列化为 XHTML 格式的字符串。

函数的核心是 `_write_html` 函数，它接受一个 `ElementTree` 对象，代表根元素及其子元素。这个函数会将元素及其子元素序列化为 XML 格式，并将其输出为字符串。

函数的实现可能还有其他细节，比如可能还定义了一些辅助函数或内部变量，但这些细节在题目中没有给出，因此无法进行详细解释。


```py
def to_xhtml_string(element: Element) -> str:
    """ Serialize element and its children to a string of XHTML. """
    return _write_html(ElementTree(element).getroot(), format="xhtml")

```

# `/markdown/markdown/test_tools.py`

这是一个 Python 实现 John Gruber 的 Markdown 的代码。Markdown 是一种轻量级的标记语言，可以轻松地将普通文本转换为 HTML 或者导出为 PDF、Word、Markdown 列表格式等。

这个实现使用了 Manfred Stienstra 和 Yuri Takhteyev 的原始实现，同时维护了 DWERG 网站上推荐的一些额外功能。这个版本是 1.7，支持 Markdown 列表和链接。

这个项目的详细信息、贡献列表和 GitHub 仓库都可以在代码中找到。


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

这段代码是一个Python代码，定义了一个名为`markdown`的类，该类提供了一组工具来测试Markdown代码基线和扩展。

具体来说，这段代码包含以下几个部分：

1. 一个名为`__future__`的注释，表示该代码将使用Python 2.x版本的事实未来的特性。

2. 导入了三个名为`os`、`sys`和`unittest`的模块。

3. 通过`import`语句导入了`markdown`、`Markdown`和`util`模块。

4. 通过`from typing import Any`导入了任何类型的变量。

5. 在`markdown`类中定义了一个`test_markdown_extensions`方法，该方法使用`unittest`模块的`TestCase`和`assert_true`方法测试Markdown代码的扩展是否正常工作。

6. 在`markdown`类中定义了一个`parse_markdown`方法，该方法读取一个Markdown字符串并返回解析后的Markdown对象。

7. 在`markdown`类中定义了一个`add_extensions`方法，该方法允许用户自定义Markdown扩展。用户可以创建一个`Markdown`对象，使用`add_extensions`方法添加所需的扩展，然后调用`parse_markdown`方法解析该扩展的Markdown字符串。

8. 在`markdown`类中定义了一个`util`类，该类包含了一些通用的工具方法，如`print_bold`、`print_italic`等。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

""" A collection of tools for testing the Markdown code base and extensions. """

from __future__ import annotations

import os
import sys
import unittest
import textwrap
from typing import Any
from . import markdown, Markdown, util

```

It looks like this is a class that extends `unittest.TestCase`. It has a `default_kwargs` parameter which is a dictionary of default keyword arguments to pass to `Markdown` when testing. It also has a `dedent` method that is used to dedent any triple-quoted strings in the source text.

The `assertMarkdownRenders` method is used to test that the source Markdown text renders to the expected output. It accepts the source text, the expected output, and any keywords passed to `Markdown`. It uses the `assertMultiLineEqual` method to test that the output contains only the expected lines of text. It then tests each attribute of the `Markdown` instance using the `assertEqual` method, with the `getattr` method used to retrieve the attribute value from the `Markdown` instance if it is not found in the default keywords.

The `dedent` method is used to dedent any triple-quoted strings in the source text. This is done by using the `textwrap.dedent` function, which takes the triple-quoted text as input and returns the deduced text with all double quotes removed. This is then stripped of any leading or trailing double quotes and returns the cleaned text.


```py
try:
    import tidylib
except ImportError:
    tidylib = None

__all__ = ['TestCase', 'LegacyTestCase', 'Kwargs']


class TestCase(unittest.TestCase):
    """
    A [`unittest.TestCase`][] subclass with helpers for testing Markdown output.

    Define `default_kwargs` as a `dict` of keywords to pass to Markdown for each
    test. The defaults can be overridden on individual tests.

    The `assertMarkdownRenders` method accepts the source text, the expected
    output, and any keywords to pass to Markdown. The `default_kwargs` are used
    except where overridden by `kwargs`. The output and expected output are passed
    to `TestCase.assertMultiLineEqual`. An `AssertionError` is raised with a diff
    if the actual output does not equal the expected output.

    The `dedent` method is available to dedent triple-quoted strings if
    necessary.

    In all other respects, behaves as `unittest.TestCase`.
    """

    default_kwargs: dict[str, Any] = {}
    """ Default options to pass to Markdown for each test. """

    def assertMarkdownRenders(self, source, expected, expected_attrs=None, **kwargs):
        """
        Test that source Markdown text renders to expected output with given keywords.

        `expected_attrs` accepts a `dict`. Each key should be the name of an attribute
        on the `Markdown` instance and the value should be the expected value after
        the source text is parsed by Markdown. After the expected output is tested,
        the expected value for each attribute is compared against the actual
        attribute of the `Markdown` instance using `TestCase.assertEqual`.
        """

        expected_attrs = expected_attrs or {}
        kws = self.default_kwargs.copy()
        kws.update(kwargs)
        md = Markdown(**kws)
        output = md.convert(source)
        self.assertMultiLineEqual(output, expected)
        for key, value in expected_attrs.items():
            self.assertEqual(getattr(md, key), value)

    def dedent(self, text):
        """
        Dedent text.
        """

        # TODO: If/when actual output ends with a newline, then use:
        #     return textwrap.dedent(text.strip('/n'))
        return textwrap.dedent(text).strip()


```

这段代码定义了一个名为 `recursionlimit` 的类，用于在测试时暂时修改 Python 的 recursion 限制，以维持测试框架的一致性。该类包含了一个私有的 `__init__` 方法和一个私有的 `__enter__` 和 `__exit__` 方法。

在 `__init__` 方法中，首先获取当前堆栈深度和设置一个新的 recursion 限制。然后，将旧的 recursion 限制设置为当前限制的两倍。这样，当函数被enter 进入时，会设置一个新的 recursion 限制，同时在enter 返回时恢复旧的 recursion 限制。

在 `__enter__` 方法中，执行当前 recursion 限制，将当前堆栈深度设置为新的限深度，并返回当前堆栈深度。

在 `__exit__` 方法中，恢复旧的 recursion 限制，并清除当前堆栈。


```py
class recursionlimit:
    """
    A context manager which temporarily modifies the Python recursion limit.

    The testing framework, coverage, etc. may add an arbitrary number of levels to the depth. To maintain consistency
    in the tests, the current stack depth is determined when called, then added to the provided limit.

    Example usage:

    ``` python
    with recursionlimit(20):
        # test code here
    ```py

    See <https://stackoverflow.com/a/50120316/866026>.
    """

    def __init__(self, limit):
        self.limit = util._get_stack_depth() + limit
        self.old_limit = sys.getrecursionlimit()

    def __enter__(self):
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)


```

这段代码定义了一个名为`Kwargs`的类，该类类似于Python中的字典，用于存储键值对（关键字参数）。这个类有一个`_normalize_whitespace`方法，用于将字符串中的空白字符和换行符进行normalize，可以有效地去除代码中的关键字参数中的空白字符和换行符，提高代码的可读性。


```py
#########################
# Legacy Test Framework #
#########################


class Kwargs(dict):
    """ A `dict` like class for holding keyword arguments. """
    pass


def _normalize_whitespace(text):
    """ Normalize whitespace for a string of HTML using `tidylib`. """
    output, errors = tidylib.tidy_fragment(text, options={
        'drop_empty_paras': 0,
        'fix_backslash': 0,
        'fix_bad_comments': 0,
        'fix_uri': 0,
        'join_styles': 0,
        'lower_literals': 0,
        'merge_divs': 0,
        'output_xhtml': 1,
        'quote_ampersand': 0,
        'newline': 'LF'
    })
    return output


```

This looks like a Python class that wraps a TinyHorn test suite and inherits from the `unittest.TestCase` class. It appears to perform some customization of the default test behavior, such as adding a `location` parameter to the `__init__` method, allowing for an explicit `normalize` parameter, and handling file paths that end with different extensions than the expected input file type. It also appears to define a `generate_test` function that takes an input file, an output file path, a boolean for whether to normalize the input (default is `True`), and keyword arguments for the test case.


```py
class LegacyTestMeta(type):
    def __new__(cls, name, bases, dct):

        def generate_test(infile, outfile, normalize, kwargs):
            def test(self):
                with open(infile, encoding="utf-8") as f:
                    input = f.read()
                with open(outfile, encoding="utf-8") as f:
                    # Normalize line endings
                    # (on Windows, git may have altered line endings).
                    expected = f.read().replace("\r\n", "\n")
                output = markdown(input, **kwargs)
                if tidylib and normalize:
                    try:
                        expected = _normalize_whitespace(expected)
                        output = _normalize_whitespace(output)
                    except OSError:
                        self.skipTest("Tidylib's c library not available.")
                elif normalize:
                    self.skipTest('Tidylib not available.')
                self.assertMultiLineEqual(output, expected)
            return test

        location = dct.get('location', '')
        exclude = dct.get('exclude', [])
        normalize = dct.get('normalize', False)
        input_ext = dct.get('input_ext', '.txt')
        output_ext = dct.get('output_ext', '.html')
        kwargs = dct.get('default_kwargs', Kwargs())

        if os.path.isdir(location):
            for file in os.listdir(location):
                infile = os.path.join(location, file)
                if os.path.isfile(infile):
                    tname, ext = os.path.splitext(file)
                    if ext == input_ext:
                        outfile = os.path.join(location, tname + output_ext)
                        tname = tname.replace(' ', '_').replace('-', '_')
                        kws = kwargs.copy()
                        if tname in dct:
                            kws.update(dct[tname])
                        test_name = 'test_%s' % tname
                        if tname not in exclude:
                            dct[test_name] = generate_test(infile, outfile, normalize, kws)
                        else:
                            dct[test_name] = unittest.skip('Excluded')(lambda: None)

        return type.__new__(cls, name, bases, dct)


```

这段代码定义了一个名为`LegacyTestCase`的类，继承自`unittest.TestCase`类，用于在Markdown中运行基于文件测试。这个子类实现了以下属性：

* `location`: 测试文件的目录。绝对路径 preferred。
* `exclude`: 排除的测试文件列表。每个测试文件名由文件名和扩展名组成，以空格和斜杠分隔。
* `normalize`: HTML输出是否正常化的开关。默认为False。
* `input_ext`: 输入文件的文件扩展名。默认为`.txt`。
* `output_ext`: 预期输出文件的文件扩展名。默认为`.html`。
* `default_kwargs`: 对于每个目录下的测试文件，该类实例可以定义自己的默认参数。该属性以参数类型和任意关键字的形式指定，并包含一个`Kwargs`实例，用于将该参数组的参数传递给`Markdown`类。该参数组将被“更新”`default_kwargs`。

该类实例还实现了以下方法：

* `__init__(self, *args, **kwargs)`: 通过参数和关键字参数实例化。
* `markdown_test_file(self, markdown_path, **kwargs)`: 在测试文件中运行Markdown测试。
* `run_ tests(self, test_filter='**', **kwargs)`: 运行测试套件中所有测试的并排除测试的测试套件。


```py
class LegacyTestCase(unittest.TestCase, metaclass=LegacyTestMeta):
    """
    A [`unittest.TestCase`][] subclass for running Markdown's legacy file-based tests.

    A subclass should define various properties which point to a directory of
    text-based test files and define various behaviors/defaults for those tests.
    The following properties are supported:

    Attributes:
        location (str): A path to the directory of test files. An absolute path is preferred.
        exclude (list[str]): A list of tests to exclude. Each test name should comprise the filename
            without an extension.
        normalize (bool): A boolean value indicating if the HTML should be normalized. Default: `False`.
        input_ext (str): A string containing the file extension of input files. Default: `.txt`.
        output_ext (str): A string containing the file extension of expected output files. Default: `html`.
        default_kwargs (Kwargs[str, Any]): The default set of keyword arguments for all test files in the directory.

    In addition, properties can be defined for each individual set of test files within
    the directory. The property should be given the name of the file without the file
    extension. Any spaces and dashes in the filename should be replaced with
    underscores. The value of the property should be a `Kwargs` instance which
    contains the keyword arguments that should be passed to `Markdown` for that
    test file. The keyword arguments will "update" the `default_kwargs`.

    When the class instance is created, it will walk the given directory and create
    a separate `Unitttest` for each set of test files using the naming scheme:
    `test_filename`. One `Unittest` will be run for each set of input and output files.
    """
    pass

```

# `/markdown/markdown/treeprocessors.py`

该代码是一个Python实现了John Gruber的Markdown的实现。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的文档。该代码将Markdown格式的文本转换为HTML格式的文档，并且可以插入链接、图片、列表、标题等元素。

该代码的作用是提供一个Python的Markdown实现，以便用户可以将Markdown格式的文本转换为HTML格式的文档。该实现支持Markdown的很多特性，如插入链接、图片、列表、标题等元素，并且可以通过调用一些附加功能来改变文档的样式和排版。


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

这段代码定义了一个名为 `TreeProcessor` 的类，表示树处理程序。这个类实现了两个方法：`processTree` 和 `createElementTree`。

`processTree` 方法接收一个 `ElementTree` 对象，使用其中提供的树块处理器对树进行操作，还可以创建一个新的 `ElementTree` 对象。

`createElementTree` 方法接收一个 `ElementTree` 对象，创建一个新的 `ElementTree` 对象，并将 `None` 作为其子节点。这个方法可以用来作为树的一个起点，然后使用 `processTree` 方法对树进行操作，从而生成新的树或者修改现有的树。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
Tree processors manipulate the tree created by block processors. They can even create an entirely
new `ElementTree` object. This is an excellent place for creating summaries, adding collected
references, or last minute adjustments.

"""

from __future__ import annotations

import re
import xml.etree.ElementTree as etree
```

这段代码的作用是定义了一个名为 `build_treeprocessors` 的函数，它接受一个名为 `md` 的 `Markdown` 对象，并返回一个名为 `Registry` 的类对象 `treeprocessors`。

该函数内部使用 `typing.TYPE_CHECKING` 注解来确保该函数可以正确地使用 `Markdown` 对象，即使 `Markdown` 不是 `typing` 的一部分。

函数内部通过 `util.Registry` 创建了一个新的注册表 `treeprocessors`，该注册表包含四个不同的注册项：

- `InlineProcessor(md)`: 将 `md` 中的 `InlineProcessor` 注册到注册表中，它的值为 `20`，表示该进程器的优先级为最高。
- `PrettifyTreeprocessor(md)`: 将 `md` 中的 `PrettifyTreeprocessor` 注册到注册表中，它的值为 `10`，表示该进程器的优先级为中等。
- `UnescapeTreeprocessor(md)`: 将 `md` 中的 `UnescapeTreeprocessor` 注册到注册表中，它的值为 `0`，表示该进程器的优先级为最低。

最后，函数返回经过注册的注册表对象 `treeprocessors`。


```py
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


def build_treeprocessors(md: Markdown, **kwargs: Any) -> util.Registry[Treeprocessor]:
    """ Build the default  `treeprocessors` for Markdown. """
    treeprocessors = util.Registry()
    treeprocessors.register(InlineProcessor(md), 'inline', 20)
    treeprocessors.register(PrettifyTreeprocessor(md), 'prettify', 10)
    treeprocessors.register(UnescapeTreeprocessor(md), 'unescape', 0)
    return treeprocessors


```

这段代码定义了一个名为 `isString` 的函数，其接收一个名为 `s` 的参数，并返回一个布尔值。函数的实现方式是判断 `s` 是否为字符串类型，如果不是，则将其强制转换为字符串类型。如果 `s` 是字符串类型，则直接返回 `True`；如果不是，则返回 `False`。

接下来，定义了一个名为 `Treeprocessor` 的类，继承自 `markdown.Treeprocessor` 类。 `Treeprocessor` 类的 `run` 方法接收一个名为 `root` 的元素对象，并对它进行修改。由于 `Treeprocessor` 类必须扩展 `markdown.Treeprocessor`，因此需要实现 `run` 方法来完成 `Treeprocessor` 的职责。

然而，以上代码并没有对 `Treeprocessor` 的 `run` 方法进行实现。因此，如果要在实际应用中使用 `Treeprocessor`，需要对 `Treeprocessor` 的 `run` 方法进行实现，以满足 `Treeprocessor` 的要求。


```py
def isString(s: object) -> bool:
    """ Return `True` if object is a string but not an  [`AtomicString`][markdown.util.AtomicString]. """
    if not isinstance(s, util.AtomicString):
        return isinstance(s, str)
    return False


class Treeprocessor(util.Processor):
    """
    `Treeprocessor`s are run on the `ElementTree` object before serialization.

    Each `Treeprocessor` implements a `run` method that takes a pointer to an
    `Element` and modifies it as necessary.

    `Treeprocessors` must extend `markdown.Treeprocessor`.

    """
    def run(self, root: etree.Element) -> etree.Element | None:
        """
        Subclasses of `Treeprocessor` should implement a `run` method, which
        takes a root `Element`. This method can return another `Element`
        object, and the existing root `Element` will be replaced, or it can
        modify the current tree and return `None`.
        """
        pass  # pragma: no cover


```

以下是一个使用Python实现的近义链表树（Ancestors Tree）的实现。请注意，这个实现可能不是最优的，但它能正常工作。

python
class AncestorsTree:
   def __init__(self, tree):
       self.tree = tree

   def __build_ancestors(self, currElement, parents):
       self.ancestors = []

       while currElement:
           parent = self.tree.iter().next()
           self.ancestors.append(parent.tag.lower())
           if parent.text:
               self.ancestors.append(parent.text)
           currElement = parent

   def __processPlaceholders(self, text, element):
       result = []

       for placeholder in text.split('<phosphorus'):
           if place == '<phosphorus':
               result.append(element)
               continue

           parts = place.split('<phosphorus')
           if len(parts) > 1:
               phrase = ''.join(parts)
               result.append(phrase)
               element = result.pop()
               element.text = text[:-1] + place + element.text
               result.append(element)
           else:
               element.text = text[:-1] + place + element.text
               result.append(element)

       return result

   def __handleInline(self, text):
       result = []

       phrases = text.split(' ')
       if len(phrases) > 1:
           phrase = ' '.join(phrases)
           result.append(phrase)
           phrases.pop())

       for phrase in phrases:
           if phrase.startswith('<phosphorus'):
               element = self.tree.find('<phosphorus')
               element.appendChild(self.processPlaceholder(phrase[8:], element))
               continue
               element.appendChild(self.processPlaceholder(phrase[8:], element))
           else:
               element.appendChild(self.processPlaceholder(phrase, element))

       return result

   def insert(self, position, text):
       self.tree.insert(position, text)

   def insertChildren(self, position, children):
       for child in children:
           self.insert(position, child)

   def insertAt(self, position, text, root=None):
       if root:
           self.insertAt(position, text, root.get('r'))
           self.insertAt(position, text, root.get('l'))

   def __str__(self):
       return '<AncestorsTree ' + str(self.tree) + '>'


为了测试这个实现，您可以创建一个测试祖先链表树：

python
def main():
   root = AncestorsTree(' ancestors.xml')
   tree = root.tree
   print(tree)
   root.print_tree()

if __name__ == '__main__':
   main()


这个测试会将`<phosphorus standalone="true">`插入到树中。首先，打印祖先链表树：


<AncestorsTree standalone="true">
       <phosphorus standalone="true">
           text: <b城市>(c城市)(d城市)(e城市)(f城市)</b>
       </phosphorus>
       <phosphorus standalone="true">
           text: <i城市>(j城市)(k城市)</i>
       </phosphorus>
       <phosphorus standalone="true">
           text: <phosphorus standalone="true" style="position: relative;">(g)</phosphorus>
       </phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(h)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(i)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(j)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(k)</phosphorus>
       <phosphorus standalone="true">
           text: <b老子>(l老子)</b>
       </phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(m老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(n老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(o老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(p老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(q老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(r老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(s老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(t老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(u老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(v老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(w老子)</phosphorus>
       <phosphorus standalone="true" style="position: absolute; top: 0; left: 0;">(x老子)</phosphorus>
       <phosphorus standalone="true"


```py
class InlineProcessor(Treeprocessor):
    """
    A `Treeprocessor` that traverses a tree, applying inline patterns.
    """

    def __init__(self, md: Markdown):
        self.__placeholder_prefix = util.INLINE_PLACEHOLDER_PREFIX
        self.__placeholder_suffix = util.ETX
        self.__placeholder_length = 4 + len(self.__placeholder_prefix) \
                                      + len(self.__placeholder_suffix)
        self.__placeholder_re = util.INLINE_PLACEHOLDER_RE
        self.md = md
        self.inlinePatterns = md.inlinePatterns
        self.ancestors: list[str] = []

    def __makePlaceholder(self, type: str) -> tuple[str, str]:
        """ Generate a placeholder """
        id = "%04d" % len(self.stashed_nodes)
        hash = util.INLINE_PLACEHOLDER % id
        return hash, id

    def __findPlaceholder(self, data: str, index: int) -> tuple[str | None, int]:
        """
        Extract id from data string, start from index.

        Arguments:
            data: String.
            index: Index, from which we start search.

        Returns:
            Placeholder id and string index, after the found placeholder.

        """
        m = self.__placeholder_re.search(data, index)
        if m:
            return m.group(1), m.end()
        else:
            return None, index + 1

    def __stashNode(self, node: etree.Element | str, type: str) -> str:
        """ Add node to stash. """
        placeholder, id = self.__makePlaceholder(type)
        self.stashed_nodes[id] = node
        return placeholder

    def __handleInline(self, data: str, patternIndex: int = 0) -> str:
        """
        Process string with inline patterns and replace it with placeholders.

        Arguments:
            data: A line of Markdown text.
            patternIndex: The index of the `inlinePattern` to start with.

        Returns:
            String with placeholders.

        """
        if not isinstance(data, util.AtomicString):
            startIndex = 0
            count = len(self.inlinePatterns)
            while patternIndex < count:
                data, matched, startIndex = self.__applyPattern(
                    self.inlinePatterns[patternIndex], data, patternIndex, startIndex
                )
                if not matched:
                    patternIndex += 1
        return data

    def __processElementText(self, node: etree.Element, subnode: etree.Element, isText: bool = True) -> None:
        """
        Process placeholders in `Element.text` or `Element.tail`
        of Elements popped from `self.stashed_nodes`.

        Arguments:
            node: Parent node.
            subnode: Processing node.
            isText: Boolean variable, True - it's text, False - it's a tail.

        """
        if isText:
            text = subnode.text
            subnode.text = None
        else:
            text = subnode.tail
            subnode.tail = None

        childResult = self.__processPlaceholders(text, subnode, isText)

        if not isText and node is not subnode:
            pos = list(node).index(subnode) + 1
        else:
            pos = 0

        childResult.reverse()
        for newChild in childResult:
            node.insert(pos, newChild[0])

    def __processPlaceholders(
        self,
        data: str | None,
        parent: etree.Element,
        isText: bool = True
    ) -> list[tuple[etree.Element, list[str]]]:
        """
        Process string with placeholders and generate `ElementTree` tree.

        Arguments:
            data: String with placeholders instead of `ElementTree` elements.
            parent: Element, which contains processing inline data.
            isText: Boolean variable, True - it's text, False - it's a tail.

        Returns:
            List with `ElementTree` elements with applied inline patterns.

        """
        def linkText(text: str | None) -> None:
            if text:
                if result:
                    if result[-1][0].tail:
                        result[-1][0].tail += text
                    else:
                        result[-1][0].tail = text
                elif not isText:
                    if parent.tail:
                        parent.tail += text
                    else:
                        parent.tail = text
                else:
                    if parent.text:
                        parent.text += text
                    else:
                        parent.text = text
        result = []
        strartIndex = 0
        while data:
            index = data.find(self.__placeholder_prefix, strartIndex)
            if index != -1:
                id, phEndIndex = self.__findPlaceholder(data, index)

                if id in self.stashed_nodes:
                    node = self.stashed_nodes.get(id)

                    if index > 0:
                        text = data[strartIndex:index]
                        linkText(text)

                    if not isString(node):  # it's Element
                        for child in [node] + list(node):
                            if child.tail:
                                if child.tail.strip():
                                    self.__processElementText(
                                        node, child, False
                                    )
                            if child.text:
                                if child.text.strip():
                                    self.__processElementText(child, child)
                    else:  # it's just a string
                        linkText(node)
                        strartIndex = phEndIndex
                        continue

                    strartIndex = phEndIndex
                    result.append((node, self.ancestors[:]))

                else:  # wrong placeholder
                    end = index + len(self.__placeholder_prefix)
                    linkText(data[strartIndex:end])
                    strartIndex = end
            else:
                text = data[strartIndex:]
                if isinstance(data, util.AtomicString):
                    # We don't want to loose the `AtomicString`
                    text = util.AtomicString(text)
                linkText(text)
                data = ""

        return result

    def __applyPattern(
        self,
        pattern: inlinepatterns.Pattern,
        data: str,
        patternIndex: int,
        startIndex: int = 0
    ) -> tuple[str, bool, int]:
        """
        Check if the line fits the pattern, create the necessary
        elements, add it to `stashed_nodes`.

        Arguments:
            data: The text to be processed.
            pattern: The pattern to be checked.
            patternIndex: Index of current pattern.
            startIndex: String index, from which we start searching.

        Returns:
            String with placeholders instead of `ElementTree` elements.

        """
        new_style = isinstance(pattern, inlinepatterns.InlineProcessor)

        for exclude in pattern.ANCESTOR_EXCLUDES:
            if exclude.lower() in self.ancestors:
                return data, False, 0

        if new_style:
            match = None
            # Since `handleMatch` may reject our first match,
            # we iterate over the buffer looking for matches
            # until we can't find any more.
            for match in pattern.getCompiledRegExp().finditer(data, startIndex):
                node, start, end = pattern.handleMatch(match, data)
                if start is None or end is None:
                    startIndex += match.end(0)
                    match = None
                    continue
                break
        else:  # pragma: no cover
            match = pattern.getCompiledRegExp().match(data[startIndex:])
            leftData = data[:startIndex]

        if not match:
            return data, False, 0

        if not new_style:  # pragma: no cover
            node = pattern.handleMatch(match)
            start = match.start(0)
            end = match.end(0)

        if node is None:
            return data, True, end

        if not isString(node):
            if not isinstance(node.text, util.AtomicString):
                # We need to process current node too
                for child in [node] + list(node):
                    if not isString(node):
                        if child.text:
                            self.ancestors.append(child.tag.lower())
                            child.text = self.__handleInline(
                                child.text, patternIndex + 1
                            )
                            self.ancestors.pop()
                        if child.tail:
                            child.tail = self.__handleInline(
                                child.tail, patternIndex
                            )

        placeholder = self.__stashNode(node, pattern.type())

        if new_style:
            return "{}{}{}".format(data[:start],
                                   placeholder, data[end:]), True, 0
        else:  # pragma: no cover
            return "{}{}{}{}".format(leftData,
                                     match.group(1),
                                     placeholder, match.groups()[-1]), True, 0

    def __build_ancestors(self, parent: etree.Element | None, parents: list[str]) -> None:
        """Build the ancestor list."""
        ancestors = []
        while parent is not None:
            if parent is not None:
                ancestors.append(parent.tag.lower())
            parent = self.parent_map.get(parent)
        ancestors.reverse()
        parents.extend(ancestors)

    def run(self, tree: etree.Element, ancestors: list[str] | None = None) -> etree.Element:
        """Apply inline patterns to a parsed Markdown tree.

        Iterate over `Element`, find elements with inline tag, apply inline
        patterns and append newly created Elements to tree.  To avoid further
        processing of string with inline patterns, instead of normal string,
        use subclass [`AtomicString`][markdown.util.AtomicString]:

            node.text = markdown.util.AtomicString("This will not be processed.")

        Arguments:
            tree: `Element` object, representing Markdown tree.
            ancestors: List of parent tag names that precede the tree node (if needed).

        Returns:
            An element tree object with applied inline patterns.

        """
        self.stashed_nodes: dict[str, etree.Element | str] = {}

        # Ensure a valid parent list, but copy passed in lists
        # to ensure we don't have the user accidentally change it on us.
        tree_parents = [] if ancestors is None else ancestors[:]

        self.parent_map = {c: p for p in tree.iter() for c in p}
        stack = [(tree, tree_parents)]

        while stack:
            currElement, parents = stack.pop()

            self.ancestors = parents
            self.__build_ancestors(currElement, self.ancestors)

            insertQueue = []
            for child in currElement:
                if child.text and not isinstance(
                    child.text, util.AtomicString
                ):
                    self.ancestors.append(child.tag.lower())
                    text = child.text
                    child.text = None
                    lst = self.__processPlaceholders(
                        self.__handleInline(text), child
                    )
                    for item in lst:
                        self.parent_map[item[0]] = child
                    stack += lst
                    insertQueue.append((child, lst))
                    self.ancestors.pop()
                if child.tail:
                    tail = self.__handleInline(child.tail)
                    dumby = etree.Element('d')
                    child.tail = None
                    tailResult = self.__processPlaceholders(tail, dumby, False)
                    if dumby.tail:
                        child.tail = dumby.tail
                    pos = list(currElement).index(child) + 1
                    tailResult.reverse()
                    for newChild in tailResult:
                        self.parent_map[newChild[0]] = currElement
                        currElement.insert(pos, newChild[0])
                if len(child):
                    self.parent_map[child] = currElement
                    stack.append((child, self.ancestors[:]))

            for element, lst in insertQueue:
                for i, obj in enumerate(lst):
                    newChild = obj[0]
                    element.insert(i, newChild)
        return tree


```

这段代码定义了一个名为 `PrettifyTreeprocessor` 的类，它继承自 `Treeprocessor` 类。这个类的功能是在给定的 `etree.Element` 对象上添加行间插件，使得在渲染到页面上时能够正确地添加行间换行。

具体来说，这个类包含了一个名为 `_prettifyETree` 的方法，它会递归地将行间插件添加到给定元素的 `ElementTree` 子元素中。如果给定元素是一个文本元素或者一个不包含文本的元素，而这个元素不是一个代码元素，那么它会将行间插件添加到元素的 `text` 属性中。对于给定元素中的每一个子元素，如果它是一个代码元素，那么它会递归地调用 `_prettifyETree` 方法来处理这个代码元素。

在 `run` 方法中，给定元素的对象调用 `_prettifyETree` 方法，然后处理所有的 `<br>` 元素。在处理 `<br>` 元素时，如果元素对象没有 `tail` 子元素或者 `tail` 子元素的内容为空，那么它会将 `<br>` 元素的内容设置为 `'\n'`。否则，它会遍历给定元素的 `tail` 子元素，并将 `'\n'` 添加到它的 end 属性中。最后，对于给定元素中的 `<pre>` 元素，如果它的内容包含文本，那么它会尝试从 `text` 属性中读取文本，并将文本设置为只包含文本的 `AtomicString` 对象。否则，它不会对 `<pre>` 元素做任何处理。


```py
class PrettifyTreeprocessor(Treeprocessor):
    """ Add line breaks to the html document. """

    def _prettifyETree(self, elem: etree.Element) -> None:
        """ Recursively add line breaks to `ElementTree` children. """

        i = "\n"
        if self.md.is_block_level(elem.tag) and elem.tag not in ['code', 'pre']:
            if (not elem.text or not elem.text.strip()) \
                    and len(elem) and self.md.is_block_level(elem[0].tag):
                elem.text = i
            for e in elem:
                if self.md.is_block_level(e.tag):
                    self._prettifyETree(e)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i

    def run(self, root: etree.Element) -> None:
        """ Add line breaks to `Element` object and its children. """

        self._prettifyETree(root)
        # Do `<br />`'s separately as they are often in the middle of
        # inline content and missed by `_prettifyETree`.
        brs = root.iter('br')
        for br in brs:
            if not br.tail or not br.tail.strip():
                br.tail = '\n'
            else:
                br.tail = '\n%s' % br.tail
        # Clean up extra empty lines at end of code blocks.
        pres = root.iter('pre')
        for pre in pres:
            if len(pre) and pre[0].tag == 'code':
                code = pre[0]
                # Only prettify code containing text only
                if not len(code) and code.text is not None:
                    code.text = util.AtomicString(code.text.rstrip() + '\n')


```

这段代码定义了一个名为 UnescapeTreeprocessor 的类，继承自名为 Treeprocessor 的类。这个 UnescapeTreeprocessor 类的主要作用是处理包含escape转义序列的字符串，将它们恢复成其原始字符。

具体来说，这个类包含了一个名为 _unescape 的方法，它接受一个正则表达式 m，并返回一个字符。正则表达式 `{util.STX}{ATTR_SEARCH}` 匹配文本中的 STX 和 ETX，其中 ATTR_SEARCH 是一个字符，用于匹配元素的属性值。这个正则表达式被用来查找所有具有 ATTR_SEARCH 属性的元素，并将它们内部的 escape 转义序列替换成它内部的匹配表达式（也就是 `{}` 部分）返回的结果。

此外，这个 UnescapeTreeprocessor 类还包含一个名为 unescape 的方法，它接受一个字符串参数，将其中的所有 escape 转义序列替换成它内部的 _unescape 方法处理后的结果。这个方法在两个情况下被调用：一是当字符串内容中有 escape 转义序列时，二是当字符串是元素的属性值时。

最后，这个 UnescapeTreeprocessor 类还包含一个名为 run 的方法，它接受一个 etree.Element 对象根元素，将从根元素开始遍历到所有子元素。在循环内部，它将遍历所有的文本内容，如果文本内容不是代码标签（即以代码标签开始标签和结束标签之间的内容），它将尝试将文本内容中的 escape 转义序列替换成它内部的 _unescape 方法处理后的结果。


```py
class UnescapeTreeprocessor(Treeprocessor):
    """ Restore escaped chars """

    RE = re.compile(r'{}(\d+){}'.format(util.STX, util.ETX))

    def _unescape(self, m: re.Match[str]) -> str:
        return chr(int(m.group(1)))

    def unescape(self, text: str) -> str:
        return self.RE.sub(self._unescape, text)

    def run(self, root: etree.Element) -> None:
        """ Loop over all elements and unescape all text. """
        for elem in root.iter():
            # Unescape text content
            if elem.text and not elem.tag == 'code':
                elem.text = self.unescape(elem.text)
            # Unescape tail content
            if elem.tail:
                elem.tail = self.unescape(elem.tail)
            # Unescape attribute values
            for key, value in elem.items():
                elem.set(key, self.unescape(value))

```