# PythonMarkdown源码解析 16

# `/markdown/tests/test_syntax/extensions/test_footnotes.py`

这是一段Python代码，它是一个Python实现的John Gruber的Markdown。Markdown是一种轻量级的标记语言，它可以让你快速地将你的文档写成易于阅读和学习的格式。

这段代码定义了一个名为`markdown`的类，它实现了Markdown的一些基本功能。首先，它导入了三个Markdown相关的Python库：`markdown`, `htmlentity`, 和`latex`。然后，它定义了一个`render`方法，用于将Markdown内容渲染成HTML。最后，它导入了两个评论对象，分别是用于显示Markdown文档的`DocTypeView`对象和用于显示Markdown文档的`MarkdownView`对象。

所以，这段代码的主要作用是实现一个Python的Markdown解析器和渲染器，可以让你将Markdown文档转换成HTML，并且支持Markdown语法。


```py
"""
Python Markdown

A Python implementation of John Gruber's Markdown.

Documentation: https://python-markdown.github.io/
GitHub: https://github.com/Python-Markdown/markdown/
PyPI: https://pypi.org/project/Markdown/

Started by Manfred Stienstra (http://www.dwerg.net/).
Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
Currently maintained by Waylan Limberg (https://github.com/waylan),
Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
```

It looks like your test is testing the backlink title and text of an超级注释。在你的测试代码中，`test_backlink_title()`和`test_superscript_text()`函数分别测试了这些配置。

在`test_backlink_title()`函数中，你可以看到`test_backlink_title()`在测试以`<sup>`为前缀的超级链接的标题。在这里，`self.assertMarkdownRenders()`函数是用来验证在`<p>`和`<div>`标签之间的内容是否正确地渲染成了一个超级链接。测试结果显示，这个链接的标题应该是"Jump back to footnote"，没有出现错误。

在`test_superscript_text()`函数中，你可以看到`test_superscript_text()`在测试以`<sup>`为前缀的超级链接的文本内容。在这里，`self.assertMarkdownRenders()`函数是用来验证在`<p>`和`<div>`标签之间的内容是否正确地渲染成了一个超级链接。测试结果显示，这个链接的文本内容应该是"A Footnote"，没有出现错误。


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestFootnotes(TestCase):

    default_kwargs = {'extensions': ['footnotes']}
    maxDiff = None

    def test_basic_footnote(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                paragraph[^1]

                [^1]: A Footnote
                """
            ),
            '<p>paragraph<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote&#160;<a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

    def test_multiple_footnotes(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                foo[^1]

                bar[^2]

                [^1]: Footnote 1
                [^2]: Footnote 2
                """
            ),
            '<p>foo<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<p>bar<sup id="fnref:2"><a class="footnote-ref" href="#fn:2">2</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>Footnote 1&#160;<a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '<li id="fn:2">\n'
            '<p>Footnote 2&#160;<a class="footnote-backref" href="#fnref:2"'
            ' title="Jump back to footnote 2 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

    def test_multiple_footnotes_multiline(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                foo[^1]

                bar[^2]

                [^1]: Footnote 1
                    line 2
                [^2]: Footnote 2
                """
            ),
            '<p>foo<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<p>bar<sup id="fnref:2"><a class="footnote-ref" href="#fn:2">2</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>Footnote 1\nline 2&#160;<a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '<li id="fn:2">\n'
            '<p>Footnote 2&#160;<a class="footnote-backref" href="#fnref:2"'
            ' title="Jump back to footnote 2 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

    def test_footnote_multi_line(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                paragraph[^1]
                [^1]: A Footnote
                    line 2
                """
            ),
            '<p>paragraph<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote\nline 2&#160;<a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

    def test_footnote_multi_line_lazy_indent(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                paragraph[^1]
                [^1]: A Footnote
                line 2
                """
            ),
            '<p>paragraph<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote\nline 2&#160;<a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

    def test_footnote_multi_line_complex(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                paragraph[^1]

                [^1]:

                    A Footnote
                    line 2

                    * list item

                    > blockquote
                """
            ),
            '<p>paragraph<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote\nline 2</p>\n'
            '<ul>\n<li>list item</li>\n</ul>\n'
            '<blockquote>\n<p>blockquote</p>\n</blockquote>\n'
            '<p><a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

    def test_footnote_multple_complex(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                foo[^1]

                bar[^2]

                [^1]:

                    A Footnote
                    line 2

                    * list item

                    > blockquote

                [^2]: Second footnote

                    paragraph 2
                """
            ),
            '<p>foo<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<p>bar<sup id="fnref:2"><a class="footnote-ref" href="#fn:2">2</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote\nline 2</p>\n'
            '<ul>\n<li>list item</li>\n</ul>\n'
            '<blockquote>\n<p>blockquote</p>\n</blockquote>\n'
            '<p><a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '<li id="fn:2">\n'
            '<p>Second footnote</p>\n'
            '<p>paragraph 2&#160;<a class="footnote-backref" href="#fnref:2"'
            ' title="Jump back to footnote 2 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

    def test_footnote_multple_complex_no_blank_line_between(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                foo[^1]

                bar[^2]

                [^1]:

                    A Footnote
                    line 2

                    * list item

                    > blockquote
                [^2]: Second footnote

                    paragraph 2
                """
            ),
            '<p>foo<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<p>bar<sup id="fnref:2"><a class="footnote-ref" href="#fn:2">2</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote\nline 2</p>\n'
            '<ul>\n<li>list item</li>\n</ul>\n'
            '<blockquote>\n<p>blockquote</p>\n</blockquote>\n'
            '<p><a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '<li id="fn:2">\n'
            '<p>Second footnote</p>\n'
            '<p>paragraph 2&#160;<a class="footnote-backref" href="#fnref:2"'
            ' title="Jump back to footnote 2 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

    def test_backlink_text(self):
        """Test back-link configuration."""

        self.assertMarkdownRenders(
            'paragraph[^1]\n\n[^1]: A Footnote',
            '<p>paragraph<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote&#160;<a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">back</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>',
            extension_configs={'footnotes': {'BACKLINK_TEXT': 'back'}}
        )

    def test_footnote_separator(self):
        """Test separator configuration."""

        self.assertMarkdownRenders(
            'paragraph[^1]\n\n[^1]: A Footnote',
            '<p>paragraph<sup id="fnref-1"><a class="footnote-ref" href="#fn-1">1</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn-1">\n'
            '<p>A Footnote&#160;<a class="footnote-backref" href="#fnref-1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>',
            extension_configs={'footnotes': {'SEPARATOR': '-'}}
        )

    def test_backlink_title(self):
        """Test back-link title configuration without placeholder."""

        self.assertMarkdownRenders(
            'paragraph[^1]\n\n[^1]: A Footnote',
            '<p>paragraph<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote&#160;<a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>',
            extension_configs={'footnotes': {'BACKLINK_TITLE': 'Jump back to footnote'}}
        )

    def test_superscript_text(self):
        """Test superscript text configuration."""

        self.assertMarkdownRenders(
            'paragraph[^1]\n\n[^1]: A Footnote',
            '<p>paragraph<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">[1]</a></sup></p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>A Footnote&#160;<a class="footnote-backref" href="#fnref:1"'
            ' title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>',
            extension_configs={'footnotes': {'SUPERSCRIPT_TEXT': '[{}]'}}
        )

```

# `/markdown/tests/test_syntax/extensions/test_legacy_attrs.py`

这是一段使用Python实现的Markdown代码。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的文档。

这段代码定义了一个名为Markdown的类，包含了Markdown的一些核心功能，例如：

1. 定义了Markdown的一些保留字，例如#、**、##等，这些保留字可以作为标题的级别前缀。
2. 实现了一些Markdown对象的构造函数，包括`Html`、`Block`、`Heading`、`Link`等，可以用于生成Markdown文档。
3. 通过`import markdown`从标准库中引入了Markdown的相关包。
4. 通过`from markdown import frontmatter`导入了Markdown的FrontMatter模式，可以生成文档的标题、内容等信息。

这段代码的作用是提供一个Python的Markdown实现，可以用来创建Markdown文档并将其转换为HTML格式的文档。


```py
"""
Python Markdown

A Python implementation of John Gruber's Markdown.

Documentation: https://python-markdown.github.io/
GitHub: https://github.com/Python-Markdown/markdown/
PyPI: https://pypi.org/project/Markdown/

Started by Manfred Stienstra (http://www.dwerg.net/).
Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
Currently maintained by Waylan Limberg (https://github.com/waylan),
Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
```

This is a test case for the Fourthought platform that uses legacy attributes. The test case checks if the legacy attributes are rendered correctly in the HTML markup.

The `TestLegacyAtrributes` class inherits from `TestCase` and has a maximum difference of `None`. This is to ensure that the test case does not allow for unexpected differences in the rendering of the legacy attributes.

The `testLegacyAttrs` method checks if the legacy attributes are rendered correctly by using the `assertMarkdownRenders` method. This method checks if the markdown is rendered without any syntax errors. If there are no syntax errors, the method checks if the legacy attributes are rendered correctly in the markdown.

Overall, this test case is useful to ensure that the legacy attributes are rendered correctly in the Fourthought platform.


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestLegacyAtrributes(TestCase):

    maxDiff = None

    def testLegacyAttrs(self):
        self.assertMarkdownRenders(
            self.dedent("""
                # Header {@id=inthebeginning}

                Now, let's try something *inline{@class=special}*, to see if it works

                @id=TABLE.OF.CONTENTS}


                * {@id=TABLEOFCONTENTS}


                Or in the middle of the text {@id=TABLEOFCONTENTS}

                {@id=tableofcontents}

                [![{@style=float: left; margin: 10px; border:
                none;}](http://fourthought.com/images/ftlogo.png "Fourthought
                logo")](http://fourthought.com/)

                ![img{@id=foo}][img]

                [img]: http://example.com/i.jpg
            """),
            self.dedent("""
                <h1 id="inthebeginning">Header </h1>
                <p>Now, let's try something <em class="special">inline</em>, to see if it works</p>
                <p>@id=TABLE.OF.CONTENTS}</p>
                <ul>
                <li id="TABLEOFCONTENTS"></li>
                </ul>
                <p id="TABLEOFCONTENTS">Or in the middle of the text </p>
                <p id="tableofcontents"></p>
                <p><a href="http://fourthought.com/"><img alt="" src="http://fourthought.com/images/ftlogo.png" style="float: left; margin: 10px; border: none;" title="Fourthought logo" /></a></p>
                <p><img alt="img" id="foo" src="http://example.com/i.jpg" /></p>
            """),  # noqa: E501
            extensions=['legacy_attrs']
        )

```

# `/markdown/tests/test_syntax/extensions/test_legacy_em.py`

这是一段Python代码，它是一个Markdown渲染器的实现。Markdown是一种轻量级的标记语言，它可以使得网页更加简洁易读。

在这段代码中，首先定义了一个Python Markdown类，这个类包含了一些方法，用来将Markdown语法转换为HTML语法或者将HTML语法转换为Markdown语法。

接着定义了一个指向GitHub仓库的链接，这个链接指向了Markdown项目的GitHub仓库，可以方便地访问项目的源代码。

然后定义了一个起始日期和一个维护者，用来记录Markdown项目的创建者和维护者。

最后定义了一些版权信息，用来保护Markdown项目的版权。


```py
"""
Python Markdown

A Python implementation of John Gruber's Markdown.

Documentation: https://python-markdown.github.io/
GitHub: https://github.com/Python-Markdown/markdown/
PyPI: https://pypi.org/project/Markdown/

Started by Manfred Stienstra (http://www.dwerg.net/).
Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
Currently maintained by Waylan Limberg (https://github.com/waylan),
Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
```

It looks like your test suite is testing markdown rendering with different extensions, specifically `legacy_em` and `strong`.

The `assertMarkdownRenders` function is checking that the markdown rendering is correct, with the error message containing information about the failure.

The `test_complex_emphasis_underscore` method is testing a specific case where the text contains both bold and italic text within a middle of the word.

The `test_complex_multple_underscore_type` method is testing a similar case but with a different text.

If the test passes for both cases, it means that the markdown rendering is working as expected. If any of the tests fail, it will give an error message with information about the failure.


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestLegacyEm(TestCase):
    def test_legacy_emphasis(self):
        self.assertMarkdownRenders(
            '_connected_words_',
            '<p><em>connected</em>words_</p>',
            extensions=['legacy_em']
        )

    def test_legacy_strong(self):
        self.assertMarkdownRenders(
            '__connected__words__',
            '<p><strong>connected</strong>words__</p>',
            extensions=['legacy_em']
        )

    def test_complex_emphasis_underscore(self):
        self.assertMarkdownRenders(
            'This is text __bold _italic bold___ with more text',
            '<p>This is text <strong>bold <em>italic bold</em></strong> with more text</p>',
            extensions=['legacy_em']
        )

    def test_complex_emphasis_underscore_mid_word(self):
        self.assertMarkdownRenders(
            'This is text __bold_italic bold___ with more text',
            '<p>This is text <strong>bold<em>italic bold</em></strong> with more text</p>',
            extensions=['legacy_em']
        )

    def test_complex_multple_underscore_type(self):

        self.assertMarkdownRenders(
            'traced ___along___ bla __blocked__ if other ___or___',
            '<p>traced <strong><em>along</em></strong> bla <strong>blocked</strong> if other <strong><em>or</em></strong></p>'  # noqa: E501
        )

    def test_complex_multple_underscore_type_variant2(self):

        self.assertMarkdownRenders(
            'on the __1-4 row__ of the AP Combat Table ___and___ receive',
            '<p>on the <strong>1-4 row</strong> of the AP Combat Table <strong><em>and</em></strong> receive</p>'
        )

```

# `/markdown/tests/test_syntax/extensions/test_md_in_html.py`

该代码是一个Python Markdown的实现，它遵循了John Gruber的Markdown规范。它的作用是提供一个Python程序来解析和渲染Markdown格式的文本。它包括一个主函数markdown_document，其中包含了对Markdown文档的解析和渲染。

具体来说，该代码将Markdown文档中的所有元数据（如标题、段落、列表、链接等）解析为Python数据类型，例如列表或字符串。然后，它使用这些数据类型来 render（呈现）Markdown文档，从而将文档转换为纯Python代码。

该代码使用了Python的markdown库，该库是一个Python解析器和渲染器，可以将Markdown格式的文本转换为HTML格式的文档。通过使用该库，该代码可以轻松地将Markdown文档转换为HTML，从而在Python环境中更方便地共享和呈现Markdown内容。


```py
# -*- coding: utf-8 -*-
"""
Python Markdown

A Python implementation of John Gruber's Markdown.

Documentation: https://python-markdown.github.io/
GitHub: https://github.com/Python-Markdown/markdown/
PyPI: https://pypi.org/project/Markdown/

Started by Manfred Stienstra (http://www.dwerg.net/).
Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
Currently maintained by Waylan Limberg (https://github.com/waylan),
Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

```

这段代码是一个 Python 测试框架中的一个测试类，属于标记down测试套件的一部分。其目的是测试 `Markdown` 库中的 `test_html_blocks` 函数的正确性，该函数在收到的 Markdown 块中执行测试。

具体来说，这段代码以下午包含以下步骤：

1. 导入 `unittest`、`markdown.test_tools` 和 `xml.etree.ElementTree` 库。
2. 继承自 `TestCase`，从而获得 `self.assertTrue()` 方法的调用权。
3. 实现 `test_stash_to_string()` 方法，该方法接收一个 `Element` 对象，该对象的 `text` 属性包含一个或多个 Markdown 块。
4. 使用 `Markdown` 库的 `postprocessors` 属性，设置 `md_in_html` 扩展，从而启用 `test_html_blocks` 函数。
5. 使用 `element` 获取一个 `div` 元素，并将其文本设置为 `'Foo bar.'`。
6. 使用 `md.postprocessors['raw_html'].stash_to_string(element)` 调用 `test_html_blocks` 函数，将 `element` 对象传递给函数，并得到一个字符串 `<div>Foo bar.</div>`。
7. 最后，使用 `self.assertEqual()` 方法比较得到的结果和预期结果（空字符串）。

如果 `test_html_blocks` 函数在处理 Markdown 块时出现问题，该测试将无法通过，并生成错误消息。


```py
Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from unittest import TestSuite
from markdown.test_tools import TestCase
from ..blocks.test_html_blocks import TestHTMLBlocks
from markdown import Markdown
from xml.etree.ElementTree import Element


class TestMarkdownInHTMLPostProcessor(TestCase):
    """ Ensure any remaining elements in HTML stash are properly serialized. """

    def test_stash_to_string(self):
        # There should be no known cases where this actually happens so we need to
        # forcefully pass an `etree` `Element` to the method to ensure proper behavior.
        element = Element('div')
        element.text = 'Foo bar.'
        md = Markdown(extensions=['md_in_html'])
        result = md.postprocessors['raw_html'].stash_to_string(element)
        self.assertEqual(result, '<div>Foo bar.</div>')


```

It looks like this is a testing class for the `@markdown` extension. They are testing different scenarios to ensure that the extension is working correctly.

Their tests include:

* testing markdown with `<script>` tag and `<script>` extension
* testing markdown with `<script>` extension and inline `<script>` tag
* testing markdown with `<script>` extension and nested `<script>` tag
* testing markdown with `<script>` extension and nested `<script>` tag
* testing markdown with `<script>` extension and inline `<script>` tag with `target` attribute set to `'self'`
* testing markdown with `<script>` extension and inline `<script>` tag with `target` attribute set to `'self'` and `scoped'` attribute set to `'√'`
* testing markdown with `<script>` extension and inline `<script>` tag with `target` attribute set to `'self'` and `scoped'` attribute set to `'runonly'`
* testing markdown with `<script>` extension and inline `<script>` tag with `target` attribute set to `'self'` and `scoped'` attribute set to `'runonly'`
* testing markdown with `<script>` extension and inline `<script>` tag with `data-script` attribute set to `'√'`
* testing markdown with `<script>` extension and inline `<script>` tag with `data-script` attribute set to `'√'` and `mode` attribute set to `'no_source'`
* testing markdown with `<script>` extension and inline `<script>` tag with `data-script` attribute set to `'√'` and `mode` attribute set to `'no_source'`
* testing markdown with `<script>` extension and inline `<script>` tag with `target` attribute set to `'out'`
* testing markdown with `<script>` extension and inline `<script>` tag with `target` attribute set to `'out'` and `scoped'` attribute set to `'√'`
* testing markdown with `<script>` extension and inline `<script>` tag with `target` attribute set to `'out'` and `scoped'` attribute set to `'runonly'`

These tests are making sure that the `@markdown` extension is functioning correctly in different scenarios and that it renders the markdown as expected.


```py
class TestDefaultwMdInHTML(TestHTMLBlocks):
    """ Ensure the md_in_html extension does not break the default behavior. """

    default_kwargs = {'extensions': ['md_in_html']}


class TestMdInHTML(TestCase):

    default_kwargs = {'extensions': ['md_in_html']}

    def test_md1_paragraph(self):
        self.assertMarkdownRenders(
            '<p markdown="1">*foo*</p>',
            '<p><em>foo</em></p>'
        )

    def test_md1_p_linebreaks(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <p markdown="1">
                *foo*
                </p>
                """
            ),
            self.dedent(
                """
                <p>
                <em>foo</em>
                </p>
                """
            )
        )

    def test_md1_p_blank_lines(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <p markdown="1">

                *foo*

                </p>
                """
            ),
            self.dedent(
                """
                <p>

                <em>foo</em>

                </p>
                """
            )
        )

    def test_md1_div(self):
        self.assertMarkdownRenders(
            '<div markdown="1">*foo*</div>',
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                </div>
                """
            )
        )

    def test_md1_div_linebreaks(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                *foo*
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                </div>
                """
            )
        )

    def test_md1_code_span(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                `<h1>code span</h1>`
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><code>&lt;h1&gt;code span&lt;/h1&gt;</code></p>
                </div>
                """
            )
        )

    def test_md1_code_span_oneline(self):
        self.assertMarkdownRenders(
            '<div markdown="1">`<h1>code span</h1>`</div>',
            self.dedent(
                """
                <div>
                <p><code>&lt;h1&gt;code span&lt;/h1&gt;</code></p>
                </div>
                """
            )
        )

    def test_md1_code_span_unclosed(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                `<p>`
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><code>&lt;p&gt;</code></p>
                </div>
                """
            )
        )

    def test_md1_code_span_script_tag(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                `<script>`
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><code>&lt;script&gt;</code></p>
                </div>
                """
            )
        )

    def test_md1_div_blank_lines(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                *foo*

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                </div>
                """
            )
        )

    def test_md1_div_multi(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                *foo*

                __bar__

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                <p><strong>bar</strong></p>
                </div>
                """
            )
        )

    def test_md1_div_nested(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                <div markdown="1">
                *foo*
                </div>

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div>
                <p><em>foo</em></p>
                </div>
                </div>
                """
            )
        )

    def test_md1_div_multi_nest(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                <div markdown="1">
                <p markdown="1">*foo*</p>
                </div>

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div>
                <p><em>foo</em></p>
                </div>
                </div>
                """
            )
        )

    def text_md1_details(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <details markdown="1">
                <summary>Click to expand</summary>
                *foo*
                </details>
                """
            ),
            self.dedent(
                """
                <details>
                <summary>Click to expand</summary>
                <p><em>foo</em></p>
                </details>
                """
            )
        )

    def test_md1_mix(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                A _Markdown_ paragraph before a raw child.

                <p markdown="1">A *raw* child.</p>

                A _Markdown_ tail to the raw child.
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p>A <em>Markdown</em> paragraph before a raw child.</p>
                <p>A <em>raw</em> child.</p>
                <p>A <em>Markdown</em> tail to the raw child.</p>
                </div>
                """
            )
        )

    def test_md1_deep_mix(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                A _Markdown_ paragraph before a raw child.

                A second Markdown paragraph
                with two lines.

                <div markdown="1">

                A *raw* child.

                <p markdown="1">*foo*</p>

                Raw child tail.

                </div>

                A _Markdown_ tail to the raw child.

                A second tail item
                with two lines.

                <p markdown="1">More raw.</p>

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p>A <em>Markdown</em> paragraph before a raw child.</p>
                <p>A second Markdown paragraph
                with two lines.</p>
                <div>
                <p>A <em>raw</em> child.</p>
                <p><em>foo</em></p>
                <p>Raw child tail.</p>
                </div>
                <p>A <em>Markdown</em> tail to the raw child.</p>
                <p>A second tail item
                with two lines.</p>
                <p>More raw.</p>
                </div>
                """
            )
        )

    def test_md1_div_raw_inline(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                <em>foo</em>

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                </div>
                """
            )
        )

    def test_no_md1_paragraph(self):
        self.assertMarkdownRenders(
            '<p>*foo*</p>',
            '<p>*foo*</p>'
        )

    def test_no_md1_nest(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                A _Markdown_ paragraph before a raw child.

                <p>A *raw* child.</p>

                A _Markdown_ tail to the raw child.
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p>A <em>Markdown</em> paragraph before a raw child.</p>
                <p>A *raw* child.</p>
                <p>A <em>Markdown</em> tail to the raw child.</p>
                </div>
                """
            )
        )

    def test_md1_nested_empty(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                A _Markdown_ paragraph before a raw empty tag.

                <img src="image.png" alt="An image" />

                A _Markdown_ tail to the raw empty tag.
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p>A <em>Markdown</em> paragraph before a raw empty tag.</p>
                <p><img src="image.png" alt="An image" /></p>
                <p>A <em>Markdown</em> tail to the raw empty tag.</p>
                </div>
                """
            )
        )

    def test_md1_nested_empty_block(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                A _Markdown_ paragraph before a raw empty tag.

                <hr />

                A _Markdown_ tail to the raw empty tag.
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p>A <em>Markdown</em> paragraph before a raw empty tag.</p>
                <hr />
                <p>A <em>Markdown</em> tail to the raw empty tag.</p>
                </div>
                """
            )
        )

    def test_empty_tags(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                <div></div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div></div>
                </div>
                """
            )
        )

    def test_orphan_end_tag_in_raw_html(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                <div>
                Test

                </pre>

                Test
                </div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div>
                Test

                </pre>

                Test
                </div>
                </div>
                """
            )
        )

    def test_complex_nested_case(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                **test**
                <div>
                **test**
                <img src=""/>
                <code>Test</code>
                <span>**test**</span>
                <p>Test 2</p>
                </div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><strong>test</strong></p>
                <div>
                **test**
                <img src=""/>
                <code>Test</code>
                <span>**test**</span>
                <p>Test 2</p>
                </div>
                </div>
                """
            )
        )

    def test_complex_nested_case_whitespace(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Text with space\t
                <div markdown="1">\t
                \t
                 <div>
                **test**
                <img src=""/>
                <code>Test</code>
                <span>**test**</span>
                  <div>With whitespace</div>
                <p>Test 2</p>
                </div>
                **test**
                </div>
                """
            ),
            self.dedent(
                """
                <p>Text with space </p>
                <div>
                <div>
                **test**
                <img src=""/>
                <code>Test</code>
                <span>**test**</span>
                  <div>With whitespace</div>
                <p>Test 2</p>
                </div>
                <p><strong>test</strong></p>
                </div>
                """
            )
        )

    def test_md1_intail_md1(self):
        self.assertMarkdownRenders(
            '<div markdown="1">*foo*</div><div markdown="1">*bar*</div>',
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                </div>
                <div>
                <p><em>bar</em></p>
                </div>
                """
            )
        )

    def test_md1_no_blank_line_before(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                A _Markdown_ paragraph with no blank line after.
                <div markdown="1">
                A _Markdown_ paragraph in an HTML block with no blank line before.
                </div>
                """
            ),
            self.dedent(
                """
                <p>A <em>Markdown</em> paragraph with no blank line after.</p>
                <div>
                <p>A <em>Markdown</em> paragraph in an HTML block with no blank line before.</p>
                </div>
                """
            )
        )

    def test_md1_no_line_break(self):
        # The div here is parsed as a span-level element. Bad input equals bad output!
        self.assertMarkdownRenders(
            'A _Markdown_ paragraph with <div markdown="1">no _line break_.</div>',
            '<p>A <em>Markdown</em> paragraph with <div markdown="1">no <em>line break</em>.</div></p>'
        )

    def test_md1_in_tail(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div></div><div markdown="1">
                A _Markdown_ paragraph in an HTML block in tail of previous element.
                </div>
                """
            ),
            self.dedent(
                """
                <div></div>
                <div>
                <p>A <em>Markdown</em> paragraph in an HTML block in tail of previous element.</p>
                </div>
                """
            )
        )

    def test_md1_PI_oneliner(self):
        self.assertMarkdownRenders(
            '<div markdown="1"><?php print("foo"); ?></div>',
            self.dedent(
                """
                <div>
                <?php print("foo"); ?>
                </div>
                """
            )
        )

    def test_md1_PI_multiline(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                <?php print("foo"); ?>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <?php print("foo"); ?>
                </div>
                """
            )
        )

    def test_md1_PI_blank_lines(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                <?php print("foo"); ?>

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <?php print("foo"); ?>
                </div>
                """
            )
        )

    def test_md_span_paragraph(self):
        self.assertMarkdownRenders(
            '<p markdown="span">*foo*</p>',
            '<p><em>foo</em></p>'
        )

    def test_md_block_paragraph(self):
        self.assertMarkdownRenders(
            '<p markdown="block">*foo*</p>',
            self.dedent(
                """
                <p>
                <p><em>foo</em></p>
                </p>
                """
            )
        )

    def test_md_span_div(self):
        self.assertMarkdownRenders(
            '<div markdown="span">*foo*</div>',
            '<div><em>foo</em></div>'
        )

    def test_md_block_div(self):
        self.assertMarkdownRenders(
            '<div markdown="block">*foo*</div>',
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                </div>
                """
            )
        )

    def test_md_span_nested_in_block(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="block">
                <div markdown="span">*foo*</div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div><em>foo</em></div>
                </div>
                """
            )
        )

    def test_md_block_nested_in_span(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="span">
                <div markdown="block">*foo*</div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div><em>foo</em></div>
                </div>
                """
            )
        )

    def test_md_block_after_span_nested_in_block(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="block">
                <div markdown="span">*foo*</div>
                <div markdown="block">*bar*</div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div><em>foo</em></div>
                <div>
                <p><em>bar</em></p>
                </div>
                </div>
                """
            )
        )

    def test_nomd_nested_in_md1(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                *foo*
                <div>
                *foo*
                <p>*bar*</p>
                *baz*
                </div>
                *bar*
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                <div>
                *foo*
                <p>*bar*</p>
                *baz*
                </div>
                <p><em>bar</em></p>
                </div>
                """
            )
        )

    def test_md1_nested_in_nomd(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div>
                <div markdown="1">*foo*</div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div markdown="1">*foo*</div>
                </div>
                """
            )
        )

    def test_md1_single_quotes(self):
        self.assertMarkdownRenders(
            "<p markdown='1'>*foo*</p>",
            '<p><em>foo</em></p>'
        )

    def test_md1_no_quotes(self):
        self.assertMarkdownRenders(
            '<p markdown=1>*foo*</p>',
            '<p><em>foo</em></p>'
        )

    def test_md_no_value(self):
        self.assertMarkdownRenders(
            '<p markdown>*foo*</p>',
            '<p><em>foo</em></p>'
        )

    def test_md1_preserve_attrs(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1" id="parent">

                <div markdown="1" class="foo">
                <p markdown="1" class="bar baz">*foo*</p>
                </div>

                </div>
                """
            ),
            self.dedent(
                """
                <div id="parent">
                <div class="foo">
                <p class="bar baz"><em>foo</em></p>
                </div>
                </div>
                """
            )
        )

    def test_md1_unclosed_div(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                _foo_

                <div class="unclosed">

                _bar_

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                <div class="unclosed">

                _bar_

                </div>
                </div>
                """
            )
        )

    def test_md1_orphan_endtag(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">

                _foo_

                </p>

                _bar_

                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><em>foo</em></p>
                </p>
                <p><em>bar</em></p>
                </div>
                """
            )
        )

    def test_md1_unclosed_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <p markdown="1">_foo_
                <p markdown="1">_bar_
                """
            ),
            self.dedent(
                """
                <p><em>foo</em>
                </p>
                <p><em>bar</em>

                </p>
                """
            )
        )

    def test_md1_nested_unclosed_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                <p markdown="1">_foo_
                <p markdown="1">_bar_
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p><em>foo</em>
                </p>
                <p><em>bar</em>
                </p>
                </div>
                """
            )
        )

    def test_md1_nested_comment(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                A *Markdown* paragraph.
                <!-- foobar -->
                A *Markdown* paragraph.
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <p>A <em>Markdown</em> paragraph.</p>
                <!-- foobar -->
                <p>A <em>Markdown</em> paragraph.</p>
                </div>
                """
            )
        )

    def test_md1_nested_link_ref(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                [link]: http://example.com
                <div markdown="1">
                [link][link]
                </div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div>
                <p><a href="http://example.com">link</a></p>
                </div>
                </div>
                """
            )
        )

    def test_md1_hr_only_start(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                *emphasis1*
                <hr markdown="1">
                *emphasis2*
                """
            ),
            self.dedent(
                """
                <p><em>emphasis1</em></p>
                <hr>
                <p><em>emphasis2</em></p>
                """
            )
        )

    def test_md1_hr_self_close(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                *emphasis1*
                <hr markdown="1" />
                *emphasis2*
                """
            ),
            self.dedent(
                """
                <p><em>emphasis1</em></p>
                <hr>
                <p><em>emphasis2</em></p>
                """
            )
        )

    def test_md1_hr_start_and_end(self):
        # Browsers ignore ending hr tags, so we don't try to do anything to handle them special.
        self.assertMarkdownRenders(
            self.dedent(
                """
                *emphasis1*
                <hr markdown="1"></hr>
                *emphasis2*
                """
            ),
            self.dedent(
                """
                <p><em>emphasis1</em></p>
                <hr>
                <p></hr>
                <em>emphasis2</em></p>
                """
            )
        )

    def test_md1_hr_only_end(self):
        # Browsers ignore ending hr tags, so we don't try to do anything to handle them special.
        self.assertMarkdownRenders(
            self.dedent(
                """
                *emphasis1*
                </hr>
                *emphasis2*
                """
            ),
            self.dedent(
                """
                <p><em>emphasis1</em>
                </hr>
                <em>emphasis2</em></p>
                """
            )
        )

    def test_md1_hr_with_content(self):
        # Browsers ignore ending hr tags, so we don't try to do anything to handle them special.
        # Content is not allowed and will be treated as normal content between two hr tags
        self.assertMarkdownRenders(
            self.dedent(
                """
                *emphasis1*
                <hr markdown="1">
                **content**
                </hr>
                *emphasis2*
                """
            ),
            self.dedent(
                """
                <p><em>emphasis1</em></p>
                <hr>
                <p><strong>content</strong>
                </hr>
                <em>emphasis2</em></p>
                """
            )
        )

    def test_no_md1_hr_with_content(self):
        # Browsers ignore ending hr tags, so we don't try to do anything to handle them special.
        # Content is not allowed and will be treated as normal content between two hr tags
        self.assertMarkdownRenders(
            self.dedent(
                """
                *emphasis1*
                <hr>
                **content**
                </hr>
                *emphasis2*
                """
            ),
            self.dedent(
                """
                <p><em>emphasis1</em></p>
                <hr>
                <p><strong>content</strong>
                </hr>
                <em>emphasis2</em></p>
                """
            )
        )

    def test_md1_nested_abbr_ref(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                *[abbr]: Abbreviation
                <div markdown="1">
                abbr
                </div>
                </div>
                """
            ),
            self.dedent(
                """
                <div>
                <div>
                <p><abbr title="Abbreviation">abbr</abbr></p>
                </div>
                </div>
                """
            ),
            extensions=['md_in_html', 'abbr']
        )

    def test_md1_nested_footnote_ref(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <div markdown="1">
                [^1]: The footnote.
                <div markdown="1">
                Paragraph with a footnote.[^1]
                </div>
                </div>
                """
            ),
            '<div>\n'
            '<div>\n'
            '<p>Paragraph with a footnote.<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n'
            '</div>\n'
            '</div>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>The footnote.&#160;'
            '<a class="footnote-backref" href="#fnref:1" title="Jump back to footnote 1 in the text">&#8617;</a>'
            '</p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>',
            extensions=['md_in_html', 'footnotes']
        )


```

这段代码定义了一个名为 `load_tests` 的函数，它接受三个参数：`loader`、`tests` 和 `pattern`。

函数的作用是加载 `TestSuite` 类中所有的测试用例，并将它们从测试集中排除，以保证同一份测试用例不会被运行 twice。

具体实现是，首先从 `loader` 对象中获取测试用例，然后遍历 `tests` 列表中的测试用例，接着将它们添加到 `suite` 对象中。最后，返回 `suite` 对象。


```py
def load_tests(loader, tests, pattern):
    """ Ensure `TestHTMLBlocks` doesn't get run twice by excluding it here. """
    suite = TestSuite()
    for test_class in [TestDefaultwMdInHTML, TestMdInHTML, TestMarkdownInHTMLPostProcessor]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite

```