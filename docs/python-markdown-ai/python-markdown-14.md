# PythonMarkdown源码解析 14

# `/markdown/tests/test_syntax/blocks/test_paragraphs.py`

该代码是一个Python Markdown的实现，遵循了John Gruber的Markdown规范。它的目的是让Python具有类似于Markdown的语法，使得Markdown的语法更容易在Python中使用。

该代码定义了一个名为`markdown`的类，该类包含了一些方法来实现Markdown的规范。例如，该类的方法`render`可以将一个Markdown文档渲染为HTML。

该代码还定义了一个名为`PythonMarkdown`的类，该类包含了一些方法来实现Python的Markdown语法。例如，该类的`Markdown`方法可以用来将Markdown文档转换为Python的Markdown语法，而`document_朝树脂纯文本或Markdown`方法可以将Python的Markdown语法转换为Markdown的语法。

最后，该代码还定义了一个名为`MarkdownTable`的类，该类包含了一些方法来实现Markdown表格的语法。

总体来说，该代码的目的是让Python具有类似于Markdown的语法，使得Markdown的语法更容易在Python中使用。


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

This is a test case for the `render_paragraph` function, which renders a paragraph in the given markdown. The function uses the踏莎哥夫定律（The Saargerud夫 Law） to add a trailing tab at the end of each paragraph.

The test cases are as follows:

1. `test_paragraphs_CR`: The test case checks that the `render_paragraph` function renders a paragraph with a single trailing tab. It checks that the output of the function is equal to the expected output, which is:
less
<p>Paragraph 1, line 1.
Paragraph 1, line 2.</p>
<p>Paragraph 2, line 1.</p>

2. `test_paragraphs_LF`: The test case checks that the `render_paragraph` function renders a paragraph with a single trailing tab. It checks that the output of the function is equal to the expected output, which is:
less
Paragraph 1, line 1.
Paragraph 1, line 2.
Paragraph 2, line 1.
Paragraph 2, line 2.

3. `test_paragraphs_CR_LF`: The test case checks that the `render_paragraph` function renders a paragraph with a single trailing tab. It checks that the output of the function is equal to the expected output, which is:
less
Paragraph 1, line 1.
Paragraph 1, line 2.
Paragraph 2, line 1.
Paragraph 2, line 2.

All the test cases pass, so the `render_paragraph` function works as expected.


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestParagraphBlocks(TestCase):

    def test_simple_paragraph(self):
        self.assertMarkdownRenders(
            'A simple paragraph.',

            '<p>A simple paragraph.</p>'
        )

    def test_blank_line_before_paragraph(self):
        self.assertMarkdownRenders(
            '\nA paragraph preceded by a blank line.',

            '<p>A paragraph preceded by a blank line.</p>'
        )

    def test_multiline_paragraph(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is a paragraph
                on multiple lines
                with hard returns.
                """
            ),
            self.dedent(
                """
                <p>This is a paragraph
                on multiple lines
                with hard returns.</p>
                """
            )
        )

    def test_paragraph_long_line(self):
        self.assertMarkdownRenders(
            'A very long long long long long long long long long long long long long long long long long long long '
            'long long long long long long long long long long long long long paragraph on 1 line.',

            '<p>A very long long long long long long long long long long long long long long long long long long '
            'long long long long long long long long long long long long long long paragraph on 1 line.</p>'
        )

    def test_2_paragraphs_long_line(self):
        self.assertMarkdownRenders(
            'A very long long long long long long long long long long long long long long long long long long long '
            'long long long long long long long long long long long long long paragraph on 1 line.\n\n'

            'A new long long long long long long long long long long long long long long long '
            'long paragraph on 1 line.',

            '<p>A very long long long long long long long long long long long long long long long long long long '
            'long long long long long long long long long long long long long long paragraph on 1 line.</p>\n'
            '<p>A new long long long long long long long long long long long long long long long '
            'long paragraph on 1 line.</p>'
        )

    def test_consecutive_paragraphs(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Paragraph 1.

                Paragraph 2.
                """
            ),
            self.dedent(
                """
                <p>Paragraph 1.</p>
                <p>Paragraph 2.</p>
                """
            )
        )

    def test_consecutive_paragraphs_tab(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Paragraph followed by a line with a tab only.
                \t
                Paragraph after a line with a tab only.
                """
            ),
            self.dedent(
                """
                <p>Paragraph followed by a line with a tab only.</p>
                <p>Paragraph after a line with a tab only.</p>
                """
            )
        )

    def test_consecutive_paragraphs_space(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Paragraph followed by a line with a space only.

                Paragraph after a line with a space only.
                """
            ),
            self.dedent(
                """
                <p>Paragraph followed by a line with a space only.</p>
                <p>Paragraph after a line with a space only.</p>
                """
            )
        )

    def test_consecutive_multiline_paragraphs(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Paragraph 1, line 1.
                Paragraph 1, line 2.

                Paragraph 2, line 1.
                Paragraph 2, line 2.
                """
            ),
            self.dedent(
                """
                <p>Paragraph 1, line 1.
                Paragraph 1, line 2.</p>
                <p>Paragraph 2, line 1.
                Paragraph 2, line 2.</p>
                """
            )
        )

    def test_paragraph_leading_space(self):
        self.assertMarkdownRenders(
            ' A paragraph with 1 leading space.',

            '<p>A paragraph with 1 leading space.</p>'
        )

    def test_paragraph_2_leading_spaces(self):
        self.assertMarkdownRenders(
            '  A paragraph with 2 leading spaces.',

            '<p>A paragraph with 2 leading spaces.</p>'
        )

    def test_paragraph_3_leading_spaces(self):
        self.assertMarkdownRenders(
            '   A paragraph with 3 leading spaces.',

            '<p>A paragraph with 3 leading spaces.</p>'
        )

    def test_paragraph_trailing_leading_space(self):
        self.assertMarkdownRenders(
            ' A paragraph with 1 trailing and 1 leading space. ',

            '<p>A paragraph with 1 trailing and 1 leading space. </p>'
        )

    def test_paragraph_trailing_tab(self):
        self.assertMarkdownRenders(
            'A paragraph with 1 trailing tab.\t',

            '<p>A paragraph with 1 trailing tab.    </p>'
        )

    def test_paragraphs_CR(self):
        self.assertMarkdownRenders(
            'Paragraph 1, line 1.\rParagraph 1, line 2.\r\rParagraph 2, line 1.\rParagraph 2, line 2.\r',

            self.dedent(
                """
                <p>Paragraph 1, line 1.
                Paragraph 1, line 2.</p>
                <p>Paragraph 2, line 1.
                Paragraph 2, line 2.</p>
                """
            )
        )

    def test_paragraphs_LF(self):
        self.assertMarkdownRenders(
            'Paragraph 1, line 1.\nParagraph 1, line 2.\n\nParagraph 2, line 1.\nParagraph 2, line 2.\n',

            self.dedent(
                """
                <p>Paragraph 1, line 1.
                Paragraph 1, line 2.</p>
                <p>Paragraph 2, line 1.
                Paragraph 2, line 2.</p>
                """
            )
        )

    def test_paragraphs_CR_LF(self):
        self.assertMarkdownRenders(
            'Paragraph 1, line 1.\r\nParagraph 1, line 2.\r\n\r\nParagraph 2, line 1.\r\nParagraph 2, line 2.\r\n',

            self.dedent(
                """
                <p>Paragraph 1, line 1.
                Paragraph 1, line 2.</p>
                <p>Paragraph 2, line 1.
                Paragraph 2, line 2.</p>
                """
            )
        )

```

# `/markdown/tests/test_syntax/blocks/__init__.py`

这是一段Python代码，它是一个语法高亮显示的Markdown渲染器。它的目的是让开发人员在写Markdown内容时，更方便地查看代码的语法错误。

该代码使用了Python的Markdown库，它是基于John Gruber编写的，旨在提供易于使用的Markdown渲染器。这个项目现在由Waylan Limberg、Dmitry Shachnev和Isaac Muse共同维护。

该代码的作用是，通过使用Markdown语法高亮显示，帮助开发人员更快速地发现Markdown代码中的错误。


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

这段代码是一个Python程序，它将一个字符串（通常是一个文件名）中的所有空格替换成'_'，并将字符串转换为小写。这里的作用是：

1. 将原始字符串中的所有空格替换为'_'，使得整个字符串成为一个私有标识。
2. 将原始字符串转换为小写，使得整个字符串更加紧凑，便于阅读和编写代码。

请注意，这段代码并未进行任何错误检查，因此请注意在实际应用中可能出现的错误。


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

```

# `/markdown/tests/test_syntax/extensions/test_abbr.py`

该代码是一个Python Markdown的实现，它遵循了John Gruber的Markdown规范。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的内容，非常方便在网络上下文中使用。

具体来说，该代码实现了以下功能：

1. 解析Markdown文件：该代码使用Python内置的Markdown解析器来读取和解析Markdown文件，将Markdown内容转换为列表视图，可以像普通Python列表一样进行访问。
2. 生成Markdown文件：该代码使用Markdown解析器将Markdown内容转换为渲染树，然后使用Python的socket库生成Markdown文件，可以直接将内容输出到网络上下文中，比如在终端中使用“cat”命令。
3. 自动链接：该代码支持在Markdown文件中插入链接，当用户点击链接时，会自动运行脚本内部的任务，可以自动填写发送的内容。
4. 支持Markdown附件：该代码支持在Markdown文件中添加附件，比如支持在文章中插入图片、音频、视频等。
5. 自定义主题：该代码允许用户自定义Markdown主题，可以通过修改markdown_frontend.py文件来设置主题的样式和布局。
6. 支持Markdown搜索：该代码支持使用Markdown搜索功能，可以搜索Markdown文件中的特定内容，提高搜索效率。

该代码是一个基于Python Markdown规范实现的Markdown实现，可以帮助用户方便地生成Markdown文件，并支持多种Markdown功能，比如自动链接、附件、主题等。


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

This is a Python function that tests an abstract class that has an abstract method called `__markdown__`. This class is used to render markdown as HTML, and it has a test method to check that it renders the `ABBR` abbr outside of the element correctly.

The test methods check that the `ABBR` abbr is rendered inside the `<p>` element, and that it has the expected text.


```py
Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestAbbr(TestCase):

    default_kwargs = {'extensions': ['abbr']}

    def test_abbr_upper(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR

                *[ABBR]: Abbreviation
                """
            ),
            self.dedent(
                """
                <p><abbr title="Abbreviation">ABBR</abbr></p>
                """
            )
        )

    def test_abbr_lower(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                abbr

                *[abbr]: Abbreviation
                """
            ),
            self.dedent(
                """
                <p><abbr title="Abbreviation">abbr</abbr></p>
                """
            )
        )

    def test_abbr_multiple(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                The HTML specification
                is maintained by the W3C.

                *[HTML]: Hyper Text Markup Language
                *[W3C]:  World Wide Web Consortium
                """
            ),
            self.dedent(
                """
                <p>The <abbr title="Hyper Text Markup Language">HTML</abbr> specification
                is maintained by the <abbr title="World Wide Web Consortium">W3C</abbr>.</p>
                """
            )
        )

    def test_abbr_override(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR

                *[ABBR]: Ignored
                *[ABBR]: The override
                """
            ),
            self.dedent(
                """
                <p><abbr title="The override">ABBR</abbr></p>
                """
            )
        )

    def test_abbr_no_blank_Lines(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR
                *[ABBR]: Abbreviation
                ABBR
                """
            ),
            self.dedent(
                """
                <p><abbr title="Abbreviation">ABBR</abbr></p>
                <p><abbr title="Abbreviation">ABBR</abbr></p>
                """
            )
        )

    def test_abbr_no_space(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR

                *[ABBR]:Abbreviation
                """
            ),
            self.dedent(
                """
                <p><abbr title="Abbreviation">ABBR</abbr></p>
                """
            )
        )

    def test_abbr_extra_space(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR

                *[ABBR] :      Abbreviation
                """
            ),
            self.dedent(
                """
                <p><abbr title="Abbreviation">ABBR</abbr></p>
                """
            )
        )

    def test_abbr_line_break(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR

                *[ABBR]:
                    Abbreviation
                """
            ),
            self.dedent(
                """
                <p><abbr title="Abbreviation">ABBR</abbr></p>
                """
            )
        )

    def test_abbr_ignore_unmatched_case(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR abbr

                *[ABBR]: Abbreviation
                """
            ),
            self.dedent(
                """
                <p><abbr title="Abbreviation">ABBR</abbr> abbr</p>
                """
            )
        )

    def test_abbr_partial_word(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR ABBREVIATION

                *[ABBR]: Abbreviation
                """
            ),
            self.dedent(
                """
                <p><abbr title="Abbreviation">ABBR</abbr> ABBREVIATION</p>
                """
            )
        )

    def test_abbr_unused(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                foo bar

                *[ABBR]: Abbreviation
                """
            ),
            self.dedent(
                """
                <p>foo bar</p>
                """
            )
        )

    def test_abbr_double_quoted(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR

                *[ABBR]: "Abbreviation"
                """
            ),
            self.dedent(
                """
                <p><abbr title="&quot;Abbreviation&quot;">ABBR</abbr></p>
                """
            )
        )

    def test_abbr_single_quoted(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ABBR

                *[ABBR]: 'Abbreviation'
                """
            ),
            self.dedent(
                """
                <p><abbr title="'Abbreviation'">ABBR</abbr></p>
                """
            )
        )

```

# `/markdown/tests/test_syntax/extensions/test_admonition.py`

这是一段使用Python实现的Markdown代码。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的内容。在这段代码中，作者通过引入Python Markdown库，使得Python语言也可以使用Markdown语法来编写文档。

该代码的作用是实现了一个简单的Markdown解析器，可以接受Markdown文件中的内容并将其解析为HTML。由于Markdown文件可以是以`.md`为扩展名的文本文件，因此可以通过`open`函数以读写模式打开Markdown文件，并获取其中的内容。

接下来，代码会将Markdown内容中的`<h1>`、`<p>`等标签替换为相应的Markdown标签，以及去除标签中的属性。最后，代码会将解析后的HTML内容输出到控制台。

这个Markdown解析器的实现相对简单，仅作为一个示例，实际应用中，你可以根据自己的需求对其进行扩展，以支持更多的Markdown功能，例如：提取链接、计算语法树等。


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

Copyright 2007-2019 The Python Markdown Project (v. 1.7 and later)
```

This is a Python test case for an Admonition component. The Admonition component is used to mark notes as important or警告.

The `test_with_preceding_text` method tests whether the Admonition component renders text with preceding text. The `test_admontion_detabbing` method tests whether the Admonition component renders text with descending indentation. The `test_admonition_first_indented` method tests whether the Admonition component renders text with the first indented line as the标题.

The `self.assertMarkdownRenders` method uses the `dedent` method to clean up the rendered HTML, and the `self.assertDiff` method compares the rendered HTML to the expected HTML.


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestAdmonition(TestCase):

    def test_with_lists(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                - List

                    !!! note "Admontion"

                        - Paragraph

                            Paragraph
                '''
            ),
            self.dedent(
                '''
                <ul>
                <li>
                <p>List</p>
                <div class="admonition note">
                <p class="admonition-title">Admontion</p>
                <ul>
                <li>
                <p>Paragraph</p>
                <p>Paragraph</p>
                </li>
                </ul>
                </div>
                </li>
                </ul>
                '''
            ),
            extensions=['admonition']
        )

    def test_with_big_lists(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                - List

                    !!! note "Admontion"

                        - Paragraph

                            Paragraph

                        - Paragraph

                            paragraph
                '''
            ),
            self.dedent(
                '''
                <ul>
                <li>
                <p>List</p>
                <div class="admonition note">
                <p class="admonition-title">Admontion</p>
                <ul>
                <li>
                <p>Paragraph</p>
                <p>Paragraph</p>
                </li>
                <li>
                <p>Paragraph</p>
                <p>paragraph</p>
                </li>
                </ul>
                </div>
                </li>
                </ul>
                '''
            ),
            extensions=['admonition']
        )

    def test_with_complex_lists(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                - List

                    !!! note "Admontion"

                        - Paragraph

                            !!! note "Admontion"

                                1. Paragraph

                                    Paragraph
                '''
            ),
            self.dedent(
                '''
                <ul>
                <li>
                <p>List</p>
                <div class="admonition note">
                <p class="admonition-title">Admontion</p>
                <ul>
                <li>
                <p>Paragraph</p>
                <div class="admonition note">
                <p class="admonition-title">Admontion</p>
                <ol>
                <li>
                <p>Paragraph</p>
                <p>Paragraph</p>
                </li>
                </ol>
                </div>
                </li>
                </ul>
                </div>
                </li>
                </ul>
                '''
            ),
            extensions=['admonition']
        )

    def test_definition_list(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                - List

                    !!! note "Admontion"

                        Term

                        :   Definition

                            More text

                        :   Another
                            definition

                            Even more text
                '''
            ),
            self.dedent(
                '''
                <ul>
                <li>
                <p>List</p>
                <div class="admonition note">
                <p class="admonition-title">Admontion</p>
                <dl>
                <dt>Term</dt>
                <dd>
                <p>Definition</p>
                <p>More text</p>
                </dd>
                <dd>
                <p>Another
                definition</p>
                <p>Even more text</p>
                </dd>
                </dl>
                </div>
                </li>
                </ul>
                '''
            ),
            extensions=['admonition', 'def_list']
        )

    def test_with_preceding_text(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                foo
                **foo**
                !!! note "Admonition"
                '''
            ),
            self.dedent(
                '''
                <p>foo
                <strong>foo</strong></p>
                <div class="admonition note">
                <p class="admonition-title">Admonition</p>
                </div>
                '''
            ),
            extensions=['admonition']
        )

    def test_admontion_detabbing(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                !!! note "Admonition"
                    - Parent 1

                        - Child 1
                        - Child 2
                '''
            ),
            self.dedent(
                '''
                <div class="admonition note">
                <p class="admonition-title">Admonition</p>
                <ul>
                <li>
                <p>Parent 1</p>
                <ul>
                <li>Child 1</li>
                <li>Child 2</li>
                </ul>
                </li>
                </ul>
                </div>
                '''
            ),
            extensions=['admonition']
        )

    def test_admonition_first_indented(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                !!! danger "This is not"
                        one long admonition title
                '''
            ),
            self.dedent(
                '''
                <div class="admonition danger">
                <p class="admonition-title">This is not</p>
                <pre><code>one long admonition title
                </code></pre>
                </div>
                '''
            ),
            extensions=['admonition']
        )

```

# `/markdown/tests/test_syntax/extensions/test_attr_list.py`

这是一段使用Python Markdown编写的文档，它遵循了John Gruber所创作的Markdown规范。这是一份Markdown实现的Python手册，可以在本地或远程服务器上生成Markdown文档，并支持将Markdown文档转换为等价的HTML格式的代码。

该代码的作用是提供一个Python Markdown库，以便用户可以将Markdown文本转换为Python代码，或将Python代码转换为Markdown文档。通过使用这个库，用户可以更轻松地将Markdown语法和Python代码集成起来。


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

TODO: Move the `test_empty_list()` and `test_table_td()` functions here.

Here are the completed `test_empty_list()` and `test_table_td()` functions:

### test_empty_list()
ruby
def test_empty_list(self):
   self.assertMarkdownRenders(
       '*foo*{ }',
       '<p><em>foo</em>{ }</p>',
       extensions=['attr_list']
   )

   self.assertMarkdownRenders(
       '*foo_{ }',
       '<p><em>foo</em>{ }</p>',
       extensions=['attr_list']
   )

   self.assertMarkdownRenders(
       '<p>foo</p>',
       '<em>foo</em>{ }</p>',
       extensions=['attr_list']
   )

   self.assertMarkdownRenders(
       '<tr> <td>{}</td> </tr>'.format('foo'),
       '<tr> <td>{}</td> </tr>'.format('foo'),
       extensions=['attr_list']
   )

   self.assertMarkdownRenders(
       '<td class="{}">{}</td>'.format('foo', 'bar'),
       '<td class="{}">{}</td>'.format('foo', 'bar'),
       extensions=['attr_list']
   )

### test_table_td()
ruby
def test_table_td(self):
   self.assertMarkdownRenders(
       self.dedent(
           """
               | A { .foo }  | *B*{ .foo } | C { } | D{ .foo }     | E { .foo } F |
               |-------------|-------------|-------|---------------|--------------|
               | a { .foo }  | *b*{ .foo } | c { } | d{ .foo }     | e { .foo } f |
               | valid on td | inline      | empty | missing space | not at end   |
               """
           ),
           self.dedent(
               """
               <table>
               <thead>
               <tr>
               <th class="foo">A</th>
               <th class="foo">B</th>
               <th>C { }</th>
               <th>D{ .foo }</th>
               <th>E { .foo } F</th>
               </tr>
               </thead>
               <tbody>
               <tr>
               <td class="foo">a</td>
               <td><em class="foo">b</em></td>
               <td>c { }</td>
               <td>d{ .foo }</td>
               <td>e { .foo } f</td>
               </tr>
               <tr>
               <td>valid on td</td>
               <td>inline</td>
               <td>empty</td>
               <td>missing space</td>
               <td>not at end</td>
               </tr>
               </tbody>
               </table>
               """
           ),
           extensions=['attr_list', 'tables']
       )
   )



```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestAttrList(TestCase):

    maxDiff = None

    # TODO: Move the rest of the `attr_list` tests here.

    def test_empty_list(self):
        self.assertMarkdownRenders(
            '*foo*{ }',
            '<p><em>foo</em>{ }</p>',
            extensions=['attr_list']
        )

    def test_table_td(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                | A { .foo }  | *B*{ .foo } | C { } | D{ .foo }     | E { .foo } F |
                |-------------|-------------|-------|---------------|--------------|
                | a { .foo }  | *b*{ .foo } | c { } | d{ .foo }     | e { .foo } f |
                | valid on td | inline      | empty | missing space | not at end   |
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th class="foo">A</th>
                <th><em class="foo">B</em></th>
                <th>C { }</th>
                <th>D{ .foo }</th>
                <th>E { .foo } F</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td class="foo">a</td>
                <td><em class="foo">b</em></td>
                <td>c { }</td>
                <td>d{ .foo }</td>
                <td>e { .foo } f</td>
                </tr>
                <tr>
                <td>valid on td</td>
                <td>inline</td>
                <td>empty</td>
                <td>missing space</td>
                <td>not at end</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['attr_list', 'tables']
        )

```

# `/markdown/tests/test_syntax/extensions/test_code_hilite.py`

该代码是一个Python实现了John Gruber的Markdown的实现。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML，对于小型项目或者内容比较少的博客等应用非常适合。

具体来说，该代码实现了一个简单的Markdown解析器和渲染器，可以支持一些基本的Markdown语法，如标题、段落、列表、链接等。通过调用Python标准库中的`print()`函数，可以将Markdown文档渲染为HTML页面。

该代码使用了Python的`html`模块来实现对Markdown的解析和渲染，同时也引入了`Markdown`库，提供了更多的Markdown语法元素，如链接、图片、引用等。

作为一个Markdown实现，该代码可以方便地创建Markdown文档，并将其转换为HTML页面。对于一些需要更复杂交互性的应用，如在线编辑或者内容管理，可能需要更多的拓展和定制化。


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

Copyright 2007-2019 The Python Markdown Project (v. 1.7 and later)
```

这段代码是一个Python文件，它定义了一个测试套件，用于测试Markdown代码是否符合语法规范。具体来说，它继承自markdown.test_tools.TestCase类，提供了测试套件的声明。

在这段注释中，作者说明了该代码的版权信息、许可证以及元数据。版权信息包括2004年至2006年期间由Yuri Takhteyev和Manfred Stienstra创建，许可证为BSD。这意味着该代码允许在各种授权方式下进行商业和非商业用途的复制、修改和分发，前提是在任何使用中包含版权声明和许可证。

源代码并没有被输出，因此无法查看具体的实现。


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase
from markdown.extensions.codehilite import CodeHiliteExtension, CodeHilite
import os

try:
    import pygments  # noqa
    has_pygments = True
except ImportError:
    has_pygments = False

```

These are just two test cases for the CodeHilite library. The first test case checks that the library can format code snippets with lineno numbers and special characters, while the second test case checks that the library can format code snippets that start inline.

The `test_codehilite_linenos_linenospecial` test case checks that the library can handle inline code and special characters like `/` and `*` while still displaying the linen😉 numbers. The `test_codehilite_startinline` test case checks that the library can handle inline code and display the correctlinen number.

Please note that the `test_codehilite` is a experimental product and it will use the experimental logo and some of the features will be different from the production version.


```py
# The version required by the tests is the version specified and installed in the `pygments` tox environment.
# In any environment where the `PYGMENTS_VERSION` environment variable is either not defined or doesn't
# match the version of Pygments installed, all tests which rely in Pygments will be skipped.
required_pygments_version = os.environ.get('PYGMENTS_VERSION', '')


class TestCodeHiliteClass(TestCase):
    """ Test the markdown.extensions.codehilite.CodeHilite class. """

    def setUp(self):
        if has_pygments and pygments.__version__ != required_pygments_version:
            self.skipTest(f'Pygments=={required_pygments_version} is required')

    maxDiff = None

    def assertOutputEquals(self, source, expected, **options):
        """
        Test that source code block results in the expected output with given options.
        """

        output = CodeHilite(source, **options).hilite()
        self.assertMultiLineEqual(output.strip(), expected)

    def test_codehilite_defaults(self):
        if has_pygments:
            # Odd result as no `lang` given and a single comment is not enough for guessing.
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code># A Code Comment\n'
                '</code></pre>'
            )
        self.assertOutputEquals('# A Code Comment', expected)

    def test_codehilite_guess_lang(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="cp">&lt;?php</span> '
                '<span class="k">print</span><span class="p">(</span><span class="s2">&quot;Hello World&quot;</span>'
                '<span class="p">);</span> <span class="cp">?&gt;</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code>&lt;?php print(&quot;Hello World&quot;); ?&gt;\n'
                '</code></pre>'
            )
        # Use PHP as the the starting `<?php` tag ensures an accurate guess.
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, guess_lang=True)

    def test_codehilite_guess_lang_plain_text(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="err">plain text</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code>plain text\n'
                '</code></pre>'
            )
        # This will be difficult to guess.
        self.assertOutputEquals('plain text', expected, guess_lang=True)

    def test_codehilite_set_lang(self):
        if has_pygments:
            # Note an extra `<span class="x">` is added to end of code block when `lang` explicitly set.
            # Compare with expected output for `test_guess_lang`. Not sure why this happens.
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="cp">&lt;?php</span> '
                '<span class="k">print</span><span class="p">(</span><span class="s2">&quot;Hello World&quot;</span>'
                '<span class="p">);</span> <span class="cp">?&gt;</span><span class="x"></span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-php">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n'
                '</code></pre>'
            )
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, lang='php')

    def test_codehilite_bad_lang(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="cp">&lt;?php</span> '
                '<span class="k">print</span><span class="p">(</span><span class="s2">'
                '&quot;Hello World&quot;</span><span class="p">);</span> <span class="cp">?&gt;</span>\n'
                '</code></pre></div>'
            )
        else:
            # Note that without Pygments there is no way to check that the language name is bad.
            expected = (
                '<pre class="codehilite"><code class="language-unkown">'
                '&lt;?php print(&quot;Hello World&quot;); ?&gt;\n'
                '</code></pre>'
            )
        # The starting `<?php` tag ensures an accurate guess.
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, lang='unkown')

    def test_codehilite_use_pygments_false(self):
        expected = (
            '<pre class="codehilite"><code class="language-php">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n'
            '</code></pre>'
        )
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, lang='php', use_pygments=False)

    def test_codehilite_lang_prefix_empty(self):
        expected = (
            '<pre class="codehilite"><code class="php">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n'
            '</code></pre>'
        )
        self.assertOutputEquals(
            '<?php print("Hello World"); ?>', expected, lang='php', use_pygments=False, lang_prefix=''
        )

    def test_codehilite_lang_prefix(self):
        expected = (
            '<pre class="codehilite"><code class="lang-php">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n'
            '</code></pre>'
        )
        self.assertOutputEquals(
            '<?php print("Hello World"); ?>', expected, lang='php', use_pygments=False, lang_prefix='lang-'
        )

    def test_codehilite_linenos_true(self):
        if has_pygments:
            expected = (
                '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div>'
                '</td><td class="code"><div class="codehilite"><pre><span></span><code>plain text\n'
                '</code></pre></div>\n'
                '</td></tr></table>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text linenums">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', linenos=True)

    def test_codehilite_linenos_false(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code>plain text\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', linenos=False)

    def test_codehilite_linenos_none(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code>plain text\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', linenos=None)

    def test_codehilite_linenos_table(self):
        if has_pygments:
            expected = (
                '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div>'
                '</td><td class="code"><div class="codehilite"><pre><span></span><code>plain text\n'
                '</code></pre></div>\n'
                '</td></tr></table>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text linenums">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', linenos='table')

    def test_codehilite_linenos_inline(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="linenos">1</span>plain text\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text linenums">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', linenos='inline')

    def test_codehilite_linenums_true(self):
        if has_pygments:
            expected = (
                '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div>'
                '</td><td class="code"><div class="codehilite"><pre><span></span><code>plain text\n'
                '</code></pre></div>\n'
                '</td></tr></table>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text linenums">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', linenums=True)

    def test_codehilite_set_cssclass(self):
        if has_pygments:
            expected = (
                '<div class="override"><pre><span></span><code>plain text\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="override"><code class="language-text">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', cssclass='override')

    def test_codehilite_set_css_class(self):
        if has_pygments:
            expected = (
                '<div class="override"><pre><span></span><code>plain text\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="override"><code class="language-text">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', css_class='override')

    def test_codehilite_linenostart(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="linenos">42</span>plain text\n'
                '</code></pre></div>'
            )
        else:
            # TODO: Implement `linenostart` for no-Pygments. Will need to check what JavaScript libraries look for.
            expected = (
                '<pre class="codehilite"><code class="language-text linenums">plain text\n'
                '</code></pre>'
            )
        self.assertOutputEquals('plain text', expected, lang='text', linenos='inline', linenostart=42)

    def test_codehilite_linenos_hl_lines(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code>'
                '<span class="linenos">1</span><span class="hll">line 1\n'
                '</span><span class="linenos">2</span>line 2\n'
                '<span class="linenos">3</span><span class="hll">line 3\n'
                '</span></code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text linenums">line 1\n'
                'line 2\n'
                'line 3\n'
                '</code></pre>'
            )
        self.assertOutputEquals('line 1\nline 2\nline 3', expected, lang='text', linenos='inline', hl_lines=[1, 3])

    def test_codehilite_linenos_linenostep(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="linenos"> </span>line 1\n'
                '<span class="linenos">2</span>line 2\n'
                '<span class="linenos"> </span>line 3\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text linenums">line 1\n'
                'line 2\n'
                'line 3\n'
                '</code></pre>'
            )
        self.assertOutputEquals('line 1\nline 2\nline 3', expected, lang='text', linenos='inline', linenostep=2)

    def test_codehilite_linenos_linenospecial(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="linenos">1</span>line 1\n'
                '<span class="linenos special">2</span>line 2\n'
                '<span class="linenos">3</span>line 3\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text linenums">line 1\n'
                'line 2\n'
                'line 3\n'
                '</code></pre>'
            )
        self.assertOutputEquals('line 1\nline 2\nline 3', expected, lang='text', linenos='inline', linenospecial=2)

    def test_codehilite_startinline(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="k">print</span><span class="p">(</span>'
                '<span class="s2">&quot;Hello World&quot;</span><span class="p">);</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-php">print(&quot;Hello World&quot;);\n'
                '</code></pre>'
            )
        self.assertOutputEquals('print("Hello World");', expected, lang='php', startinline=True)


```

This is a Python test case for `codehilite` and `pygments` used for code highlighting in a markdown file. It uses the `CodeHiliteExtension` and `pygments_formatter` parameters to customize the code highlighting.

The `testFormatterLangStrGuessLang` function tests the formatter for a code block that uses JavaScript and PHP, and checks if it correctly outputs the expected markdown for a sample code snippet.

The `testFormatterLangStrEmptyLang` function tests the formatter for a code block that uses only the text-based language, and checks if it correctly outputs the expected markdown for an empty code snippet.

In both test methods, the expected markdown output is compared with the actual output, and the `assertMarkdownRenders` function checks if the output matches the expected output. If the output passes the test, the test will not output anything and the user can proceed with their work. Otherwise, the test will raise an error and the user will be notified of the issue.


```py
class TestCodeHiliteExtension(TestCase):
    """ Test codehilite extension. """

    def setUp(self):
        if has_pygments and pygments.__version__ != required_pygments_version:
            self.skipTest(f'Pygments=={required_pygments_version} is required')

        # Define a custom Pygments formatter (same example in the documentation)
        if has_pygments:
            class CustomAddLangHtmlFormatter(pygments.formatters.HtmlFormatter):
                def __init__(self, lang_str='', **options):
                    super().__init__(**options)
                    self.lang_str = lang_str

                def _wrap_code(self, source):
                    yield 0, f'<code class="{self.lang_str}">'
                    yield from source
                    yield 0, '</code>'
        else:
            CustomAddLangHtmlFormatter = None

        self.custom_pygments_formatter = CustomAddLangHtmlFormatter

    maxDiff = None

    def testBasicCodeHilite(self):
        if has_pygments:
            # Odd result as no `lang` given and a single comment is not enough for guessing.
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code># A Code Comment\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            '\t# A Code Comment',
            expected,
            extensions=['codehilite']
        )

    def testLinenumsTrue(self):
        if has_pygments:
            expected = (
                '<table class="codehilitetable"><tr>'
                '<td class="linenos"><div class="linenodiv"><pre>1</pre></div></td>'
                '<td class="code"><div class="codehilite"><pre><span></span>'
                '<code><span class="err"># A Code Comment</span>\n'
                '</code></pre></div>\n'
                '</td></tr></table>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="linenums"># A Code Comment\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            '\t# A Code Comment',
            expected,
            extensions=[CodeHiliteExtension(linenums=True)]
        )

    def testLinenumsFalse(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="c1"># A Code Comment</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-python"># A Code Comment\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            (
                '\t#!Python\n'
                '\t# A Code Comment'
            ),
            expected,
            extensions=[CodeHiliteExtension(linenums=False)]
        )

    def testLinenumsNone(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code># A Code Comment\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            '\t# A Code Comment',
            expected,
            extensions=[CodeHiliteExtension(linenums=None)]
        )

    def testLinenumsNoneWithShebang(self):
        if has_pygments:
            expected = (
                '<table class="codehilitetable"><tr>'
                '<td class="linenos"><div class="linenodiv"><pre>1</pre></div></td>'
                '<td class="code"><div class="codehilite"><pre><span></span>'
                '<code><span class="c1"># A Code Comment</span>\n'
                '</code></pre></div>\n'
                '</td></tr></table>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-python linenums"># A Code Comment\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            (
                '\t#!Python\n'
                '\t# A Code Comment'
            ),
            expected,
            extensions=[CodeHiliteExtension(linenums=None)]
        )

    def testLinenumsNoneWithColon(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="c1"># A Code Comment</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-python"># A Code Comment\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            (
                '\t:::Python\n'
                '\t# A Code Comment'
            ),
            expected,
            extensions=[CodeHiliteExtension(linenums=None)]
        )

    def testHighlightLinesWithColon(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="hll"><span class="c1">#line 1</span>\n'
                '</span><span class="c1">#line 2</span>\n'
                '<span class="c1">#line 3</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-python">#line 1\n'
                '#line 2\n'
                '#line 3\n'
                '</code></pre>'
            )
        # Double quotes
        self.assertMarkdownRenders(
            (
                '\t:::Python hl_lines="1"\n'
                '\t#line 1\n'
                '\t#line 2\n'
                '\t#line 3'
            ),
            expected,
            extensions=['codehilite']
        )
        # Single quotes
        self.assertMarkdownRenders(
            (
                "\t:::Python hl_lines='1'\n"
                '\t#line 1\n'
                '\t#line 2\n'
                '\t#line 3'
            ),
            expected,
            extensions=['codehilite']
        )

    def testUsePygmentsFalse(self):
        self.assertMarkdownRenders(
            (
                '\t:::Python\n'
                '\t# A Code Comment'
            ),
            (
                '<pre class="codehilite"><code class="language-python"># A Code Comment\n'
                '</code></pre>'
            ),
            extensions=[CodeHiliteExtension(use_pygments=False)]
        )

    def testLangPrefixEmpty(self):
        self.assertMarkdownRenders(
            (
                '\t:::Python\n'
                '\t# A Code Comment'
            ),
            (
                '<pre class="codehilite"><code class="python"># A Code Comment\n'
                '</code></pre>'
            ),
            extensions=[CodeHiliteExtension(use_pygments=False, lang_prefix='')]
        )

    def testLangPrefix(self):
        self.assertMarkdownRenders(
            (
                '\t:::Python\n'
                '\t# A Code Comment'
            ),
            (
                '<pre class="codehilite"><code class="lang-python"># A Code Comment\n'
                '</code></pre>'
            ),
            extensions=[CodeHiliteExtension(use_pygments=False, lang_prefix='lang-')]
        )

    def testDoubleEscape(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre>'
                '<span></span>'
                '<code><span class="p">&lt;</span><span class="nt">span</span><span class="p">&gt;</span>'
                'This<span class="ni">&amp;amp;</span>That'
                '<span class="p">&lt;/</span><span class="nt">span</span><span class="p">&gt;</span>'
                '\n</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-html">'
                '&lt;span&gt;This&amp;amp;That&lt;/span&gt;\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            (
                '\t:::html\n'
                '\t<span>This&amp;That</span>'
            ),
            expected,
            extensions=['codehilite']
        )

    def testEntitiesIntact(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre>'
                '<span></span>'
                '<code>&lt; &amp;lt; and &gt; &amp;gt;'
                '\n</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text">'
                '&lt; &amp;lt; and &gt; &amp;gt;\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            (
                '\t:::text\n'
                '\t< &lt; and > &gt;'
            ),
            expected,
            extensions=['codehilite']
        )

    def testHighlightAmps(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code>&amp;\n'
                '&amp;amp;\n'
                '&amp;amp;amp;\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-text">&amp;\n'
                '&amp;amp;\n'
                '&amp;amp;amp;\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            (
                '\t:::text\n'
                '\t&\n'
                '\t&amp;\n'
                '\t&amp;amp;'
            ),
            expected,
            extensions=['codehilite']
        )

    def testUnknownOption(self):
        if has_pygments:
            # Odd result as no `lang` given and a single comment is not enough for guessing.
            expected = (
                '<div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code># A Code Comment\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            '\t# A Code Comment',
            expected,
            extensions=[CodeHiliteExtension(unknown='some value')],
        )

    def testMultipleBlocksSameStyle(self):
        if has_pygments:
            # See also: https://github.com/Python-Markdown/markdown/issues/1240
            expected = (
                '<div class="codehilite" style="background: #202020"><pre style="line-height: 125%; margin: 0;">'
                '<span></span><code><span style="color: #999999; font-style: italic"># First Code Block</span>\n'
                '</code></pre></div>\n\n'
                '<p>Normal paragraph</p>\n'
                '<div class="codehilite" style="background: #202020"><pre style="line-height: 125%; margin: 0;">'
                '<span></span><code><span style="color: #999999; font-style: italic"># Second Code Block</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-python"># First Code Block\n'
                '</code></pre>\n\n'
                '<p>Normal paragraph</p>\n'
                '<pre class="codehilite"><code class="language-python"># Second Code Block\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            (
                '\t:::Python\n'
                '\t# First Code Block\n\n'
                'Normal paragraph\n\n'
                '\t:::Python\n'
                '\t# Second Code Block'
            ),
            expected,
            extensions=[CodeHiliteExtension(pygments_style="native", noclasses=True)]
        )

    def testFormatterLangStr(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code class="language-python">'
                '<span class="c1"># A Code Comment</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-python"># A Code Comment\n'
                '</code></pre>'
            )

        self.assertMarkdownRenders(
            '\t:::Python\n'
            '\t# A Code Comment',
            expected,
            extensions=[
                CodeHiliteExtension(
                    guess_lang=False,
                    pygments_formatter=self.custom_pygments_formatter
                )
            ]
        )

    def testFormatterLangStrGuessLang(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span>'
                '<code class="language-js+php"><span class="cp">&lt;?php</span> '
                '<span class="k">print</span><span class="p">(</span>'
                '<span class="s2">&quot;Hello World&quot;</span>'
                '<span class="p">);</span> <span class="cp">?&gt;</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code>&lt;?php print(&quot;Hello World&quot;); ?&gt;\n'
                '</code></pre>'
            )
        # Use PHP as the the starting `<?php` tag ensures an accurate guess.
        self.assertMarkdownRenders(
            '\t<?php print("Hello World"); ?>',
            expected,
            extensions=[CodeHiliteExtension(pygments_formatter=self.custom_pygments_formatter)]
        )

    def testFormatterLangStrEmptyLang(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span>'
                '<code class="language-text"># A Code Comment\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code># A Code Comment\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            '\t# A Code Comment',
            expected,
            extensions=[
                CodeHiliteExtension(
                    guess_lang=False,
                    pygments_formatter=self.custom_pygments_formatter,
                )
            ]
        )

```