# PythonMarkdown源码解析 17

# `/markdown/tests/test_syntax/extensions/test_smarty.py`

该代码是一个Python Markdown的实现，它遵循了John Gruber的Markdown规范。下面是该实现的一些主要功能：

1. 支持Markdown语法：该实现支持Markdown中的常见语法，如标题、段落、列表、链接等。
2. 支持强调：该实现支持强调，可以对文本进行加粗或者斜体。
3. 支持链接：该实现支持链接，可以指向外部网站或内部链接。
4. 支持图片：该实现支持在文章中插入图片，可以通过在HTML标签中使用标签语法插入图片。
5. 支持页眉和页脚：该实现支持在文章的页眉和页脚中插入内容。
6. 支持嵌入代码：该实现支持在文章中嵌入代码，可以通过在HTML标签中使用标签语法插入代码。
7. 支持Markdown对象的导入：该实现支持导入Markdown对象，如H1、H2、H3等，以帮助您生成Markdown内容。
8. 支持Markdown链接：该实现支持生成Markdown链接，可以将链接指向外部网站或内部链接。


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

It looks like you are testing the different parts of your Markdown file, and I will go through them one by one.

1. `![x"x](x)`: This is a testing of an inline image. It renders correctly.
2. `Skip `"code" -- --- \'spans\' ...`.`: This is a testing of code blocks. It should be able to render the code and the missing closing brackets.
3. `<p>Also skip "code" <code> blocks</p>`: This is also a testing of code blocks. It should render the code blocks as an ordered list.
4. `foo -- bar --- baz ...`: This is the last test, which is also a testing of code blocks. It should render the text as a block.

Overall, it looks like your Markdown file is well-tested and covers a good range of features.


```py
Copyright 2007-2022 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestSmarty(TestCase):

    default_kwargs = {'extensions': ['smarty']}

    def test_basic(self):
        self.assertMarkdownRenders(
            "It's fun. What's fun?",
            '<p>It&rsquo;s fun. What&rsquo;s fun?</p>'
        )
        self.assertMarkdownRenders(
            '"Isn\'t this fun"? --- she said...',
            '<p>&ldquo;Isn&rsquo;t this fun&rdquo;? &mdash; she said&hellip;</p>'
        )
        self.assertMarkdownRenders(
            '"\'Quoted\' words in a larger quote."',
            '<p>&ldquo;&lsquo;Quoted&rsquo; words in a larger quote.&rdquo;</p>'
        )
        self.assertMarkdownRenders(
            '\'Quoted "words" in a larger quote.\'',
            '<p>&lsquo;Quoted &ldquo;words&rdquo; in a larger quote.&rsquo;</p>'
        )
        self.assertMarkdownRenders(
            '"quoted" text and **bold "quoted" text**',
            '<p>&ldquo;quoted&rdquo; text and <strong>bold &ldquo;quoted&rdquo; text</strong></p>'
        )
        self.assertMarkdownRenders(
            "'quoted' text and **bold 'quoted' text**",
            '<p>&lsquo;quoted&rsquo; text and <strong>bold &lsquo;quoted&rsquo; text</strong></p>'
        )
        self.assertMarkdownRenders(
            'em-dashes (---) and ellipes (...)',
            '<p>em-dashes (&mdash;) and ellipes (&hellip;)</p>'
        )
        self.assertMarkdownRenders(
            '"[Link](http://example.com)" --- she said.',
            '<p>&ldquo;<a href="http://example.com">Link</a>&rdquo; &mdash; she said.</p>'
        )
        self.assertMarkdownRenders(
            '"Ellipsis within quotes..."',
            '<p>&ldquo;Ellipsis within quotes&hellip;&rdquo;</p>'
        )
        self.assertMarkdownRenders(
            "*Custer*'s Last Stand",
            "<p><em>Custer</em>&rsquo;s Last Stand</p>"
        )

    def test_years(self):
        self.assertMarkdownRenders("1440--80's", '<p>1440&ndash;80&rsquo;s</p>')
        self.assertMarkdownRenders("1440--'80s", '<p>1440&ndash;&rsquo;80s</p>')
        self.assertMarkdownRenders("1440---'80s", '<p>1440&mdash;&rsquo;80s</p>')
        self.assertMarkdownRenders("1960's", '<p>1960&rsquo;s</p>')
        self.assertMarkdownRenders("one two '60s", '<p>one two &rsquo;60s</p>')
        self.assertMarkdownRenders("'60s", '<p>&rsquo;60s</p>')

    def test_wrapping_line(self):
        text = (
            "A line that 'wraps' with\n"
            "*emphasis* at the beginning of the next line."
        )
        html = (
            '<p>A line that &lsquo;wraps&rsquo; with\n'
            '<em>emphasis</em> at the beginning of the next line.</p>'
        )
        self.assertMarkdownRenders(text, html)

    def test_escaped(self):
        self.assertMarkdownRenders(
            'Escaped \\-- ndash',
            '<p>Escaped -- ndash</p>'
        )
        self.assertMarkdownRenders(
            '\\\'Escaped\\\' \\"quotes\\"',
            '<p>\'Escaped\' "quotes"</p>'
        )
        self.assertMarkdownRenders(
            'Escaped ellipsis\\...',
            '<p>Escaped ellipsis...</p>'
        )
        self.assertMarkdownRenders(
            '\'Escaped \\"quotes\\" in real ones\'',
            '<p>&lsquo;Escaped "quotes" in real ones&rsquo;</p>'
        )
        self.assertMarkdownRenders(
            '\\\'"Real" quotes in escaped ones\\\'',
            "<p>'&ldquo;Real&rdquo; quotes in escaped ones'</p>"
        )

    def test_escaped_attr(self):
        self.assertMarkdownRenders(
            '![x\"x](x)',
            '<p><img alt="x&quot;x" src="x" /></p>'
        )

    def test_code_spans(self):
        self.assertMarkdownRenders(
            'Skip `"code" -- --- \'spans\' ...`.',
            '<p>Skip <code>"code" -- --- \'spans\' ...</code>.</p>'
        )

    def test_code_blocks(self):
        text = (
            '    Also skip "code" \'blocks\'\n'
            '    foo -- bar --- baz ...'
        )
        html = (
            '<pre><code>Also skip "code" \'blocks\'\n'
            'foo -- bar --- baz ...\n'
            '</code></pre>'
        )
        self.assertMarkdownRenders(text, html)

    def test_horizontal_rule(self):
        self.assertMarkdownRenders('--- -- ---', '<hr />')


```

这段代码定义了一个名为 `TestSmartyAngledQuotes` 的测试类，用于测试 Smarty 框架中 Angled Quotes 类型的扩展是否正确工作。

在该测试类中，定义了一个名为 `default_kwargs` 的类级默认参数，其中包含了一些扩展配置，包括 `smarty` 扩展和 `smarty_angled_quotes` 扩展。这里，`smarty` 是一个预定义的扩展名称，表示 Smarty 中的高大突破的引用部分；`smarty_angled_quotes` 是一个布尔值，表示是否启用包含括号的高大突破引用。

接着，定义了一个名为 `test_angled_quotes` 的方法，该方法使用 `self.assertMarkdownRenders` 方法来验证在不同的输入下，Smarty 是否正确地将 Angled Quotes 渲染为网页中的字符。

`test_angled_quotes` 方法中共有三个测试用例，分别用来测试以下情况：

1. `'<<hello>>'` 是否正确地将 Angled Quotes 渲染为 `<p>` 标签和 `<strong>` 标签之间的内容；
2. `'Calvinbane://securebn.服药<<极端个案战斗寡妇人民法院疗效的崩盘'` 是否正确地将 Angled Quotes 渲染为 `<p>` 标签和 `<strong>` 标签之间的内容；
3. `'Anfleigh['睫毛'长时间暴露在阳光超广范围米奇妙火花'`是否正确地将 Angled Quotes 渲染为 `<p>` 标签和 `<strong>` 标签之间的内容。

如果以上三个用例的渲染结果均正确，那么说明该扩展名下的智能家居网站 / 应用程序 / 插件等具有正确的功能和接口。


```py
class TestSmartyAngledQuotes(TestCase):

    default_kwargs = {
        'extensions': ['smarty'],
        'extension_configs': {
            'smarty': {
                'smart_angled_quotes': True,
            },
        },
    }

    def test_angled_quotes(self):
        self.assertMarkdownRenders(
            '<<hello>>',
            '<p>&laquo;hello&raquo;</p>'
        )
        self.assertMarkdownRenders(
            'Кавычки-<<ёлочки>>',
            '<p>Кавычки-&laquo;ёлочки&raquo;</p>'
        )
        self.assertMarkdownRenders(
            'Anführungszeichen->>Chevrons<<',
            '<p>Anführungszeichen-&raquo;Chevrons&laquo;</p>'
        )


```

这段代码是一个用于测试Smarty语法的Substitutions测试用例类。其中，TestSmartyCustomSubstitutions类实现了Smarty的两个方法，分别是smarty()和test_custom_substitutions()。

1. smarty()方法用于设置Smarty的配置参数。其中，extensions表示要应用的扩展，extension_configs表示扩展的配置。

2. test_custom_substitutions()方法会创建一个带有三种样式的HTML，然后用Markdown渲染这些HTML。如果渲染结果正确，测试用例就通过了。

具体来说，该测试用例类中的default_kwargs配置了Smarty的一些扩展和扩展的配置，包括smarty()方法中的extensions和extension_configs。这些扩展包括'ndash'、'mdash'、'ellipsis'、'left-single-quote'、'right-single-quote'、'left-double-quote'和'right-double-quote'。这些扩展可以用来代替Markdown中的其他引用，从而使Smarty更易于使用。

在该测试用例类中，test_custom_substitutions()方法创建了两种HTML格式，一种是带有smarty()方法中配置的扩展的HTML，另一种是Markdown中的`<p>`标签和`<code>`标签的组合。然后这些HTML被一起提交给TestSmartyCustomSubstitutions类的一个实例，并要求其使用smarty()方法来渲染这些HTML。

如果test_custom_substitutions()方法成功渲染这些HTML，测试用例就是正确的，而且所有的扩展都应该是正确的。如果该测试用例类未能通过测试，则可以用来修复smarty()方法的扩展配置或修复在测试中使用的其他扩展。


```py
class TestSmartyCustomSubstitutions(TestCase):

    default_kwargs = {
        'extensions': ['smarty'],
        'extension_configs': {
            'smarty': {
                'smart_angled_quotes': True,
                'substitutions': {
                    'ndash': '\u2013',
                    'mdash': '\u2014',
                    'ellipsis': '\u2026',
                    'left-single-quote': '&sbquo;',  # `sb` is not a typo!
                    'right-single-quote': '&lsquo;',
                    'left-double-quote': '&bdquo;',
                    'right-double-quote': '&ldquo;',
                    'left-angle-quote': '[',
                    'right-angle-quote': ']',
                },
            },
        },
    }

    def test_custom_substitutions(self):
        text = (
            '<< The "Unicode char of the year 2014"\n'
            "is the 'mdash': ---\n"
            "Must not be confused with 'ndash'  (--) ... >>"
        )
        html = (
            '<p>[ The &bdquo;Unicode char of the year 2014&ldquo;\n'
            'is the &sbquo;mdash&lsquo;: \u2014\n'
            'Must not be confused with &sbquo;ndash&lsquo;  (\u2013) \u2026 ]</p>'
        )
        self.assertMarkdownRenders(text, html)

```

# `/markdown/tests/test_syntax/extensions/test_tables.py`

该代码是一个Python实现了John Gruber's Markdown的实现。Markdown是一种轻量级的标记语言，可以轻松地将纯文本转换为HTML格式。该代码将Markdown语法与Python代码结合使用，允许用户将Markdown文本转换为HTML，并在需要时对其进行解析。

该代码的主要目的是提供一个Python Markdown的实现，以便那些需要将Markdown文本转换为HTML的开发者能够更容易地在Python中使用它。它还允许开发人员在运行时解析Markdown文本，从而使其更具可读性。此外，该代码还提供了几个重要的内置功能，如标题、列表、链接等，使得用户可以轻松地将Markdown文本转换为格式良好的HTML。


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

This code is a Python script that is using the `markdown.extensions.tables` module to test the functionality of a table. The script is defining a class `TestTableBlocks` that inherits from `TestCase`.

The script has a single test method `test_empty_cells` which tests whether an empty cell (`nbsp`) is rendered correctly in a table. The test method uses the `setUp` method to set up a test environment and the `tearDown` method to tear down the environment after the test is finished.

The `test_empty_cells` method takes an argument `self` which refers to the instance of the `TestCase` class. Inside the method, it creates a string with an HTML table to test the rendering of empty cells. After that, it uses the `assertP自我` method to check if the `<span>` tag with an `id` of `" emptytablecell"'` is present in the rendered table.

Overall, this code is using the `markdown.extensions.tables` module to test the rendering of an HTML table with an empty cell.


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase
from markdown.extensions.tables import TableExtension


class TestTableBlocks(TestCase):

    def test_empty_cells(self):
        """Empty cells (`nbsp`)."""

        text = """
   | Second Header
```

This is a test case for the `Markdown` class's `self.assertMarkdownRenders` method. It tests that the `align_three_legacy` method correctly renders a table with three columns and three rows, including the alignment for each column.

The expected output for this test case is:
less
|foo|bar|baz|
|   |       |   |
|W  |       |   |
|   |       |   |

The actual output can be differ depending on the implementation of the `Markdown` class, but this is the expected output based on the documentation:
scss
<table>
 <thead>
 <tr>
 <th align="left" style="white-space: nowrap;">foo</th>
 <th align="center" style="white-space: nowrap;">bar</th>
 <th align="right" style="white-space: nowrap;">baz</th>
 </tr>
 </thead>
 <tbody>
 <tr>
 <td align="left" style="white-space: nowrap;">Q</td>
 <td align="center" style="white-space: nowrap;">W</td>
 <td align="right" style="white-space: nowrap;">W</td>
 </tr>
 </tbody>
 </table>



```py
------------- | -------------
   | Content Cell
Content Cell  |  
"""

        self.assertMarkdownRenders(
            text,
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th> </th>
                <th>Second Header</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td> </td>
                <td>Content Cell</td>
                </tr>
                <tr>
                <td>Content Cell</td>
                <td> </td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_no_sides(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                First Header  | Second Header
                ------------- | -------------
                Content Cell  | Content Cell
                Content Cell  | Content Cell
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th>First Header</th>
                <th>Second Header</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>Content Cell</td>
                <td>Content Cell</td>
                </tr>
                <tr>
                <td>Content Cell</td>
                <td>Content Cell</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_both_sides(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                | First Header  | Second Header |
                | ------------- | ------------- |
                | Content Cell  | Content Cell  |
                | Content Cell  | Content Cell  |
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th>First Header</th>
                <th>Second Header</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>Content Cell</td>
                <td>Content Cell</td>
                </tr>
                <tr>
                <td>Content Cell</td>
                <td>Content Cell</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_align_columns(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                | Item      | Value |
                | :-------- | -----:|
                | Computer  | $1600 |
                | Phone     |   $12 |
                | Pipe      |    $1 |
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th style="text-align: left;">Item</th>
                <th style="text-align: right;">Value</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td style="text-align: left;">Computer</td>
                <td style="text-align: right;">$1600</td>
                </tr>
                <tr>
                <td style="text-align: left;">Phone</td>
                <td style="text-align: right;">$12</td>
                </tr>
                <tr>
                <td style="text-align: left;">Pipe</td>
                <td style="text-align: right;">$1</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_styles_in_tables(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                | Function name | Description                    |
                | ------------- | ------------------------------ |
                | `help()`      | Display the help window.       |
                | `destroy()`   | **Destroy your computer!**     |
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th>Function name</th>
                <th>Description</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td><code>help()</code></td>
                <td>Display the help window.</td>
                </tr>
                <tr>
                <td><code>destroy()</code></td>
                <td><strong>Destroy your computer!</strong></td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_align_three(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                |foo|bar|baz|
                |:--|:-:|--:|
                |   | Q |   |
                |W  |   |  W|
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th style="text-align: left;">foo</th>
                <th style="text-align: center;">bar</th>
                <th style="text-align: right;">baz</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td style="text-align: left;"></td>
                <td style="text-align: center;">Q</td>
                <td style="text-align: right;"></td>
                </tr>
                <tr>
                <td style="text-align: left;">W</td>
                <td style="text-align: center;"></td>
                <td style="text-align: right;">W</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_three_columns(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                foo|bar|baz
                ---|---|---
                   | Q |
                 W |   | W
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th>foo</th>
                <th>bar</th>
                <th>baz</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td></td>
                <td>Q</td>
                <td></td>
                </tr>
                <tr>
                <td>W</td>
                <td></td>
                <td>W</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_three_spaces_prefix(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Three spaces in front of a table:

                   First Header | Second Header
                   ------------ | -------------
                   Content Cell | Content Cell
                   Content Cell | Content Cell

                   | First Header | Second Header |
                   | ------------ | ------------- |
                   | Content Cell | Content Cell  |
                   | Content Cell | Content Cell  |
                """),
            self.dedent(
                """
                <p>Three spaces in front of a table:</p>
                <table>
                <thead>
                <tr>
                <th>First Header</th>
                <th>Second Header</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>Content Cell</td>
                <td>Content Cell</td>
                </tr>
                <tr>
                <td>Content Cell</td>
                <td>Content Cell</td>
                </tr>
                </tbody>
                </table>
                <table>
                <thead>
                <tr>
                <th>First Header</th>
                <th>Second Header</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>Content Cell</td>
                <td>Content Cell</td>
                </tr>
                <tr>
                <td>Content Cell</td>
                <td>Content Cell</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_code_block_table(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Four spaces is a code block:

                    First Header | Second Header
                    ------------ | -------------
                    Content Cell | Content Cell
                    Content Cell | Content Cell

                | First Header | Second Header |
                | ------------ | ------------- |
                """),
            self.dedent(
                """
                <p>Four spaces is a code block:</p>
                <pre><code>First Header | Second Header
                ------------ | -------------
                Content Cell | Content Cell
                Content Cell | Content Cell
                </code></pre>
                <table>
                <thead>
                <tr>
                <th>First Header</th>
                <th>Second Header</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td></td>
                <td></td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_inline_code_blocks(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                More inline code block tests

                Column 1 | Column 2 | Column 3
                ---------|----------|---------
                word 1   | word 2   | word 3
                word 1   | `word 2` | word 3
                word 1   | \\`word 2 | word 3
                word 1   | `word 2 | word 3
                word 1   | `word |2` | word 3
                words    |`` some | code `` | more words
                words    |``` some | code ```py | more words
                words    |```` some | code ```py` | more words
                words    |`` some ` | ` code `` | more words
                words    |``` some ` | ` code ```py | more words
                words    |```` some ` | ` code ```py` | more words
                """),
            self.dedent(
                """
                <p>More inline code block tests</p>
                <table>
                <thead>
                <tr>
                <th>Column 1</th>
                <th>Column 2</th>
                <th>Column 3</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>word 1</td>
                <td>word 2</td>
                <td>word 3</td>
                </tr>
                <tr>
                <td>word 1</td>
                <td><code>word 2</code></td>
                <td>word 3</td>
                </tr>
                <tr>
                <td>word 1</td>
                <td>`word 2</td>
                <td>word 3</td>
                </tr>
                <tr>
                <td>word 1</td>
                <td>`word 2</td>
                <td>word 3</td>
                </tr>
                <tr>
                <td>word 1</td>
                <td><code>word |2</code></td>
                <td>word 3</td>
                </tr>
                <tr>
                <td>words</td>
                <td><code>some | code</code></td>
                <td>more words</td>
                </tr>
                <tr>
                <td>words</td>
                <td><code>some | code</code></td>
                <td>more words</td>
                </tr>
                <tr>
                <td>words</td>
                <td><code>some | code</code></td>
                <td>more words</td>
                </tr>
                <tr>
                <td>words</td>
                <td><code>some ` | ` code</code></td>
                <td>more words</td>
                </tr>
                <tr>
                <td>words</td>
                <td><code>some ` | ` code</code></td>
                <td>more words</td>
                </tr>
                <tr>
                <td>words</td>
                <td><code>some ` | ` code</code></td>
                <td>more words</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_issue_440(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                A test for issue #440:

                foo | bar
                --- | ---
                foo | (`bar`) and `baz`.
                """),
            self.dedent(
                """
                <p>A test for issue #440:</p>
                <table>
                <thead>
                <tr>
                <th>foo</th>
                <th>bar</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>foo</td>
                <td>(<code>bar</code>) and <code>baz</code>.</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_lists_not_tables(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Lists are not tables

                 - this | should | not
                 - be | a | table
                """),
            self.dedent(
                """
                <p>Lists are not tables</p>
                <ul>
                <li>this | should | not</li>
                <li>be | a | table</li>
                </ul>
                """
            ),
            extensions=['tables']
        )

    def test_issue_449(self):
        self.assertMarkdownRenders(
            self.dedent(
                r"""
                Add tests for issue #449

                Odd backticks | Even backticks
                ------------ | -------------
                ``[!\"\#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]`` | ``[!\"\#$%&'()*+,\-./:;<=>?@\[\\\]^`_`{|}~]``

                Escapes | More Escapes
                ------- | ------
                `` `\`` | `\`

                Only the first backtick can be escaped

                Escaped | Bacticks
                ------- | ------
                \`` \`  | \`\`

                Test escaped pipes

                Column 1 | Column 2
                -------- | --------
                `|` \|   | Pipes are okay in code and escaped. \|

                | Column 1 | Column 2 |
                | -------- | -------- |
                | row1     | row1    \|
                | row2     | row2     |

                Test header escapes

                | `` `\`` \| | `\` \|
                | ---------- | ---- |
                | row1       | row1 |
                | row2       | row2 |

                Escaped pipes in format row should not be a table

                | Column1   | Column2 |
                | ------- \|| ------- |
                | row1      | row1    |
                | row2      | row2    |

                Test escaped code in Table

                Should not be code | Should be code
                ------------------ | --------------
                \`Not code\`       | \\`code`
                \\\`Not code\\\`   | \\\\`code`
                """),
            self.dedent(
                """
                <p>Add tests for issue #449</p>
                <table>
                <thead>
                <tr>
                <th>Odd backticks</th>
                <th>Even backticks</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td><code>[!\\"\\#$%&amp;'()*+,\\-./:;&lt;=&gt;?@\\[\\\\\\]^_`{|}~]</code></td>
                <td><code>[!\\"\\#$%&amp;'()*+,\\-./:;&lt;=&gt;?@\\[\\\\\\]^`_`{|}~]</code></td>
                </tr>
                </tbody>
                </table>
                <table>
                <thead>
                <tr>
                <th>Escapes</th>
                <th>More Escapes</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td><code>`\\</code></td>
                <td><code>\\</code></td>
                </tr>
                </tbody>
                </table>
                <p>Only the first backtick can be escaped</p>
                <table>
                <thead>
                <tr>
                <th>Escaped</th>
                <th>Bacticks</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>`<code>\\</code></td>
                <td>``</td>
                </tr>
                </tbody>
                </table>
                <p>Test escaped pipes</p>
                <table>
                <thead>
                <tr>
                <th>Column 1</th>
                <th>Column 2</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td><code>|</code> |</td>
                <td>Pipes are okay in code and escaped. |</td>
                </tr>
                </tbody>
                </table>
                <table>
                <thead>
                <tr>
                <th>Column 1</th>
                <th>Column 2</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>row1</td>
                <td>row1    |</td>
                </tr>
                <tr>
                <td>row2</td>
                <td>row2</td>
                </tr>
                </tbody>
                </table>
                <p>Test header escapes</p>
                <table>
                <thead>
                <tr>
                <th><code>`\\</code> |</th>
                <th><code>\\</code> |</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>row1</td>
                <td>row1</td>
                </tr>
                <tr>
                <td>row2</td>
                <td>row2</td>
                </tr>
                </tbody>
                </table>
                <p>Escaped pipes in format row should not be a table</p>
                <p>| Column1   | Column2 |
                | ------- || ------- |
                | row1      | row1    |
                | row2      | row2    |</p>
                <p>Test escaped code in Table</p>
                <table>
                <thead>
                <tr>
                <th>Should not be code</th>
                <th>Should be code</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>`Not code`</td>
                <td>\\<code>code</code></td>
                </tr>
                <tr>
                <td>\\`Not code\\`</td>
                <td>\\\\<code>code</code></td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=['tables']
        )

    def test_single_column_tables(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                Single column tables

                | Is a Table |
                | ---------- |

                | Is a Table
                | ----------

                Is a Table |
                ---------- |

                | Is a Table |
                | ---------- |
                | row        |

                | Is a Table
                | ----------
                | row

                Is a Table |
                ---------- |
                row        |

                | Is not a Table
                --------------
                | row

                Is not a Table |
                --------------
                row            |

                | Is not a Table
                | --------------
                row

                Is not a Table |
                -------------- |
                row
                """),
            self.dedent(
                """
                <p>Single column tables</p>
                <table>
                <thead>
                <tr>
                <th>Is a Table</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td></td>
                </tr>
                </tbody>
                </table>
                <table>
                <thead>
                <tr>
                <th>Is a Table</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td></td>
                </tr>
                </tbody>
                </table>
                <table>
                <thead>
                <tr>
                <th>Is a Table</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td></td>
                </tr>
                </tbody>
                </table>
                <table>
                <thead>
                <tr>
                <th>Is a Table</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>row</td>
                </tr>
                </tbody>
                </table>
                <table>
                <thead>
                <tr>
                <th>Is a Table</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>row</td>
                </tr>
                </tbody>
                </table>
                <table>
                <thead>
                <tr>
                <th>Is a Table</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td>row</td>
                </tr>
                </tbody>
                </table>
                <h2>| Is not a Table</h2>
                <p>| row</p>
                <h2>Is not a Table |</h2>
                <p>row            |</p>
                <p>| Is not a Table
                | --------------
                row</p>
                <p>Is not a Table |
                -------------- |
                row</p>
                """
            ),
            extensions=['tables']
        )

    def test_align_columns_legacy(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                | Item      | Value |
                | :-------- | -----:|
                | Computer  | $1600 |
                | Phone     |   $12 |
                | Pipe      |    $1 |
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th align="left">Item</th>
                <th align="right">Value</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td align="left">Computer</td>
                <td align="right">$1600</td>
                </tr>
                <tr>
                <td align="left">Phone</td>
                <td align="right">$12</td>
                </tr>
                <tr>
                <td align="left">Pipe</td>
                <td align="right">$1</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=[TableExtension(use_align_attribute=True)]
        )

    def test_align_three_legacy(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                |foo|bar|baz|
                |:--|:-:|--:|
                |   | Q |   |
                |W  |   |  W|
                """
            ),
            self.dedent(
                """
                <table>
                <thead>
                <tr>
                <th align="left">foo</th>
                <th align="center">bar</th>
                <th align="right">baz</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                <td align="left"></td>
                <td align="center">Q</td>
                <td align="right"></td>
                </tr>
                <tr>
                <td align="left">W</td>
                <td align="center"></td>
                <td align="right">W</td>
                </tr>
                </tbody>
                </table>
                """
            ),
            extensions=[TableExtension(use_align_attribute=True)]
        )

```

# `/markdown/tests/test_syntax/extensions/test_toc.py`

该代码是一个Python实现了John Gruber的Markdown的实现。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的内容。

该代码由Mefkind Stienstra、Yuri Takhteyev、Waylan Limberg和Dmitry Shachnev共同维护。该项目的最新版本是1.7，发布于2021年9月。

该项目的源代码可以从GitHub上找到，地址为https://github.com/Python-Markdown/markdown。


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

This is a test of the TOC extension in Jekyll. The extension allows for custom classes to be added to the TOC, such as "custom1" and "custom2".

The test checks that the TOC can be rendered with a custom header and a custom title class. The test also checks that the TOC can be rendered without a title class.

The test uses the `self.assertMarkdownRenders` method to render the TOC and check the rendered output. The `self.dedent` method is used to remove extraneous whitespace and indentation from the rendered output.


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase
from markdown.extensions.toc import TocExtension
from markdown.extensions.nl2br import Nl2BrExtension


class TestTOC(TestCase):
    maxDiff = None

    # TODO: Move the rest of the TOC tests here.

    def testAnchorLink(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # Header 1

                ## Header *2*
                '''
            ),
            self.dedent(
                '''
                <h1 id="header-1"><a class="toclink" href="#header-1">Header 1</a></h1>
                <h2 id="header-2"><a class="toclink" href="#header-2">Header <em>2</em></a></h2>
                '''
            ),
            extensions=[TocExtension(anchorlink=True)]
        )

    def testAnchorLinkWithSingleInlineCode(self):
        self.assertMarkdownRenders(
            '# This is `code`.',
            '<h1 id="this-is-code">'                        # noqa
                '<a class="toclink" href="#this-is-code">'  # noqa
                    'This is <code>code</code>.'            # noqa
                '</a>'                                      # noqa
            '</h1>',                                        # noqa
            extensions=[TocExtension(anchorlink=True)]
        )

    def testAnchorLinkWithDoubleInlineCode(self):
        self.assertMarkdownRenders(
            '# This is `code` and `this` too.',
            '<h1 id="this-is-code-and-this-too">'                           # noqa
                '<a class="toclink" href="#this-is-code-and-this-too">'     # noqa
                    'This is <code>code</code> and <code>this</code> too.'  # noqa
                '</a>'                                                      # noqa
            '</h1>',                                                        # noqa
            extensions=[TocExtension(anchorlink=True)]
        )

    def testPermalink(self):
        self.assertMarkdownRenders(
            '# Header',
            '<h1 id="header">'                                                            # noqa
                'Header'                                                                  # noqa
                '<a class="headerlink" href="#header" title="Permanent link">&para;</a>'  # noqa
            '</h1>',                                                                      # noqa
            extensions=[TocExtension(permalink=True)]
        )

    def testPermalinkWithSingleInlineCode(self):
        self.assertMarkdownRenders(
            '# This is `code`.',
            '<h1 id="this-is-code">'                                                            # noqa
                'This is <code>code</code>.'                                                    # noqa
                '<a class="headerlink" href="#this-is-code" title="Permanent link">&para;</a>'  # noqa
            '</h1>',                                                                            # noqa
            extensions=[TocExtension(permalink=True)]
        )

    def testPermalinkWithDoubleInlineCode(self):
        self.assertMarkdownRenders(
            '# This is `code` and `this` too.',
            '<h1 id="this-is-code-and-this-too">'                                                            # noqa
                'This is <code>code</code> and <code>this</code> too.'                                       # noqa
                '<a class="headerlink" href="#this-is-code-and-this-too" title="Permanent link">&para;</a>'  # noqa
            '</h1>',                                                                                         # noqa
            extensions=[TocExtension(permalink=True)]
        )

    def testMinMaxLevel(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # Header 1 not in TOC

                ## Header 2 not in TOC

                ### Header 3

                #### Header 4

                ##### Header 5 not in TOC
                '''
            ),
            self.dedent(
                '''
                <h1 id="header-1-not-in-toc">Header 1 not in TOC</h1>
                <h2 id="header-2-not-in-toc">Header 2 not in TOC</h2>
                <h3 id="header-3">Header 3</h3>
                <h4 id="header-4">Header 4</h4>
                <h5 id="header-5-not-in-toc">Header 5 not in TOC</h5>
                '''
            ),
            expected_attrs={
                'toc': (
                    '<div class="toc">\n'
                      '<ul>\n'                                             # noqa
                        '<li><a href="#header-3">Header 3</a>'             # noqa
                          '<ul>\n'                                         # noqa
                            '<li><a href="#header-4">Header 4</a></li>\n'  # noqa
                          '</ul>\n'                                        # noqa
                        '</li>\n'                                          # noqa
                      '</ul>\n'                                            # noqa
                    '</div>\n'                                             # noqa
                ),
                'toc_tokens': [
                    {
                        'level': 3,
                        'id': 'header-3',
                        'name': 'Header 3',
                        'children': [
                            {
                                'level': 4,
                                'id': 'header-4',
                                'name': 'Header 4',
                                'children': []
                            }
                        ]
                    }
                ]
            },
            extensions=[TocExtension(toc_depth='3-4')]
        )

    def testMaxLevel(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # Header 1

                ## Header 2

                ### Header 3 not in TOC
                '''
            ),
            self.dedent(
                '''
                <h1 id="header-1">Header 1</h1>
                <h2 id="header-2">Header 2</h2>
                <h3 id="header-3-not-in-toc">Header 3 not in TOC</h3>
                '''
            ),
            expected_attrs={
                'toc': (
                    '<div class="toc">\n'
                      '<ul>\n'                                             # noqa
                        '<li><a href="#header-1">Header 1</a>'             # noqa
                          '<ul>\n'                                         # noqa
                            '<li><a href="#header-2">Header 2</a></li>\n'  # noqa
                          '</ul>\n'                                        # noqa
                        '</li>\n'                                          # noqa
                      '</ul>\n'                                            # noqa
                    '</div>\n'                                             # noqa
                ),
                'toc_tokens': [
                    {
                        'level': 1,
                        'id': 'header-1',
                        'name': 'Header 1',
                        'children': [
                            {
                                'level': 2,
                                'id': 'header-2',
                                'name': 'Header 2',
                                'children': []
                            }
                        ]
                    }
                ]
            },
            extensions=[TocExtension(toc_depth=2)]
        )

    def testMinMaxLevelwithAnchorLink(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # Header 1 not in TOC

                ## Header 2 not in TOC

                ### Header 3

                #### Header 4

                ##### Header 5 not in TOC
                '''
            ),
            '<h1 id="header-1-not-in-toc">'                                                      # noqa
                '<a class="toclink" href="#header-1-not-in-toc">Header 1 not in TOC</a></h1>\n'  # noqa
            '<h2 id="header-2-not-in-toc">'                                                      # noqa
                '<a class="toclink" href="#header-2-not-in-toc">Header 2 not in TOC</a></h2>\n'  # noqa
            '<h3 id="header-3">'                                                                 # noqa
                '<a class="toclink" href="#header-3">Header 3</a></h3>\n'                        # noqa
            '<h4 id="header-4">'                                                                 # noqa
                '<a class="toclink" href="#header-4">Header 4</a></h4>\n'                        # noqa
            '<h5 id="header-5-not-in-toc">'                                                      # noqa
                '<a class="toclink" href="#header-5-not-in-toc">Header 5 not in TOC</a></h5>',   # noqa
            expected_attrs={
                'toc': (
                    '<div class="toc">\n'
                      '<ul>\n'                                             # noqa
                        '<li><a href="#header-3">Header 3</a>'             # noqa
                          '<ul>\n'                                         # noqa
                            '<li><a href="#header-4">Header 4</a></li>\n'  # noqa
                          '</ul>\n'                                        # noqa
                        '</li>\n'                                          # noqa
                      '</ul>\n'                                            # noqa
                    '</div>\n'                                             # noqa
                ),
                'toc_tokens': [
                    {
                        'level': 3,
                        'id': 'header-3',
                        'name': 'Header 3',
                        'children': [
                            {
                                'level': 4,
                                'id': 'header-4',
                                'name': 'Header 4',
                                'children': []
                            }
                        ]
                    }
                ]
            },
            extensions=[TocExtension(toc_depth='3-4', anchorlink=True)]
        )

    def testMinMaxLevelwithPermalink(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # Header 1 not in TOC

                ## Header 2 not in TOC

                ### Header 3

                #### Header 4

                ##### Header 5 not in TOC
                '''
            ),
            '<h1 id="header-1-not-in-toc">Header 1 not in TOC'                                                # noqa
                '<a class="headerlink" href="#header-1-not-in-toc" title="Permanent link">&para;</a></h1>\n'  # noqa
            '<h2 id="header-2-not-in-toc">Header 2 not in TOC'                                                # noqa
                '<a class="headerlink" href="#header-2-not-in-toc" title="Permanent link">&para;</a></h2>\n'  # noqa
            '<h3 id="header-3">Header 3'                                                                      # noqa
                '<a class="headerlink" href="#header-3" title="Permanent link">&para;</a></h3>\n'             # noqa
            '<h4 id="header-4">Header 4'                                                                      # noqa
                '<a class="headerlink" href="#header-4" title="Permanent link">&para;</a></h4>\n'             # noqa
            '<h5 id="header-5-not-in-toc">Header 5 not in TOC'                                                # noqa
                '<a class="headerlink" href="#header-5-not-in-toc" title="Permanent link">&para;</a></h5>',   # noqa
            expected_attrs={
                'toc': (
                    '<div class="toc">\n'
                      '<ul>\n'                                             # noqa
                        '<li><a href="#header-3">Header 3</a>'             # noqa
                          '<ul>\n'                                         # noqa
                            '<li><a href="#header-4">Header 4</a></li>\n'  # noqa
                          '</ul>\n'                                        # noqa
                        '</li>\n'                                          # noqa
                      '</ul>\n'                                            # noqa
                    '</div>\n'                                             # noqa
                ),
                'toc_tokens': [
                    {
                        'level': 3,
                        'id': 'header-3',
                        'name': 'Header 3',
                        'children': [
                            {
                                'level': 4,
                                'id': 'header-4',
                                'name': 'Header 4',
                                'children': []
                            }
                        ]
                    }
                ]
            },
            extensions=[TocExtension(toc_depth='3-4', permalink=True)]
        )

    def testMinMaxLevelwithBaseLevel(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # First Header

                ## Second Level

                ### Third Level

                #### Forth Level
                '''
            ),
            self.dedent(
                '''
                <h3 id="first-header">First Header</h3>
                <h4 id="second-level">Second Level</h4>
                <h5 id="third-level">Third Level</h5>
                <h6 id="forth-level">Forth Level</h6>
                '''
            ),
            expected_attrs={
                'toc': (
                    '<div class="toc">\n'
                      '<ul>\n'                                                  # noqa
                        '<li><a href="#second-level">Second Level</a>'          # noqa
                          '<ul>\n'                                              # noqa
                            '<li><a href="#third-level">Third Level</a></li>\n' # noqa
                          '</ul>\n'                                             # noqa
                        '</li>\n'                                               # noqa
                      '</ul>\n'                                                 # noqa
                    '</div>\n'                                                  # noqa
                ),
                'toc_tokens': [
                    {
                        'level': 4,
                        'id': 'second-level',
                        'name': 'Second Level',
                        'children': [
                            {
                                'level': 5,
                                'id': 'third-level',
                                'name': 'Third Level',
                                'children': []
                            }
                        ]
                    }
                ]
            },
            extensions=[TocExtension(toc_depth='4-5', baselevel=3)]
        )

    def testMaxLevelwithBaseLevel(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # Some Header

                ## Next Level

                ### Too High
                '''
            ),
            self.dedent(
                '''
                <h2 id="some-header">Some Header</h2>
                <h3 id="next-level">Next Level</h3>
                <h4 id="too-high">Too High</h4>
                '''
            ),
            expected_attrs={
                'toc': (
                    '<div class="toc">\n'
                      '<ul>\n'                                                 # noqa
                        '<li><a href="#some-header">Some Header</a>'           # noqa
                          '<ul>\n'                                             # noqa
                            '<li><a href="#next-level">Next Level</a></li>\n'  # noqa
                          '</ul>\n'                                            # noqa
                        '</li>\n'                                              # noqa
                      '</ul>\n'                                                # noqa
                    '</div>\n'                                                 # noqa
                ),
                'toc_tokens': [
                    {
                        'level': 2,
                        'id': 'some-header',
                        'name': 'Some Header',
                        'children': [
                            {
                                'level': 3,
                                'id': 'next-level',
                                'name': 'Next Level',
                                'children': []
                            }
                        ]
                    }
                ]
            },
            extensions=[TocExtension(toc_depth=3, baselevel=2)]
        )

    def test_escaped_code(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                [TOC]

                # `<test>`
                '''
            ),
            self.dedent(
                '''
                <div class="toc">
                <ul>
                <li><a href="#test">&lt;test&gt;</a></li>
                </ul>
                </div>
                <h1 id="test"><code>&lt;test&gt;</code></h1>
                '''
            ),
            extensions=['toc']
        )

    def test_escaped_char_in_id(self):
        self.assertMarkdownRenders(
            r'# escaped\_character',
            '<h1 id="escaped_character">escaped_character</h1>',
            expected_attrs={
                'toc': (
                    '<div class="toc">\n'
                      '<ul>\n'                                                           # noqa
                        '<li><a href="#escaped_character">escaped_character</a></li>\n'  # noqa
                      '</ul>\n'                                                          # noqa
                    '</div>\n'                                                           # noqa
                ),
                'toc_tokens': [
                    {
                        'level': 1,
                        'id': 'escaped_character',
                        'name': 'escaped_character',
                        'children': []
                    }
                ]
            },
            extensions=['toc']
        )

    def testAnchorLinkWithCustomClass(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # Header 1

                ## Header *2*
                '''
            ),
            self.dedent(
                '''
                <h1 id="header-1"><a class="custom" href="#header-1">Header 1</a></h1>
                <h2 id="header-2"><a class="custom" href="#header-2">Header <em>2</em></a></h2>
                '''
            ),
            extensions=[TocExtension(anchorlink=True, anchorlink_class="custom")]
        )

    def testAnchorLinkWithCustomClasses(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                # Header 1

                ## Header *2*
                '''
            ),
            self.dedent(
                '''
                <h1 id="header-1"><a class="custom1 custom2" href="#header-1">Header 1</a></h1>
                <h2 id="header-2"><a class="custom1 custom2" href="#header-2">Header <em>2</em></a></h2>
                '''
            ),
            extensions=[TocExtension(anchorlink=True, anchorlink_class="custom1 custom2")]
        )

    def testPermalinkWithEmptyText(self):
        self.assertMarkdownRenders(
            '# Header',
            '<h1 id="header">'                                                      # noqa
                'Header'                                                            # noqa
                '<a class="headerlink" href="#header" title="Permanent link"></a>'  # noqa
            '</h1>',                                                                # noqa
            extensions=[TocExtension(permalink="")]
        )

    def testPermalinkWithCustomClass(self):
        self.assertMarkdownRenders(
            '# Header',
            '<h1 id="header">'                                                        # noqa
                'Header'                                                              # noqa
                '<a class="custom" href="#header" title="Permanent link">&para;</a>'  # noqa
            '</h1>',                                                                  # noqa
            extensions=[TocExtension(permalink=True, permalink_class="custom")]
        )

    def testPermalinkWithCustomClasses(self):
        self.assertMarkdownRenders(
            '# Header',
            '<h1 id="header">'                                                                 # noqa
                'Header'                                                                       # noqa
                '<a class="custom1 custom2" href="#header" title="Permanent link">&para;</a>'  # noqa
            '</h1>',                                                                           # noqa
            extensions=[TocExtension(permalink=True, permalink_class="custom1 custom2")]
        )

    def testPermalinkWithCustomTitle(self):
        self.assertMarkdownRenders(
            '# Header',
            '<h1 id="header">'                                                    # noqa
                'Header'                                                          # noqa
                '<a class="headerlink" href="#header" title="custom">&para;</a>'  # noqa
            '</h1>',                                                              # noqa
            extensions=[TocExtension(permalink=True, permalink_title="custom")]
        )

    def testPermalinkWithEmptyTitle(self):
        self.assertMarkdownRenders(
            '# Header',
            '<h1 id="header">'                                                    # noqa
                'Header'                                                          # noqa
                '<a class="headerlink" href="#header">&para;</a>'                 # noqa
            '</h1>',                                                              # noqa
            extensions=[TocExtension(permalink=True, permalink_title="")]
        )

    def testPermalinkWithUnicodeInID(self):
        from markdown.extensions.toc import slugify_unicode
        self.assertMarkdownRenders(
            '# Unicode ヘッダー',
            '<h1 id="unicode-ヘッダー">'                                                            # noqa
                'Unicode ヘッダー'                                                                  # noqa
                '<a class="headerlink" href="#unicode-ヘッダー" title="Permanent link">&para;</a>'  # noqa
            '</h1>',                                                                               # noqa
            extensions=[TocExtension(permalink=True, slugify=slugify_unicode)]
        )

    def testPermalinkWithUnicodeTitle(self):
        from markdown.extensions.toc import slugify_unicode
        self.assertMarkdownRenders(
            '# Unicode ヘッダー',
            '<h1 id="unicode-ヘッダー">'                                                        # noqa
                'Unicode ヘッダー'                                                              # noqa
                '<a class="headerlink" href="#unicode-ヘッダー" title="パーマリンク">&para;</a>'  # noqa
            '</h1>',                                                                           # noqa
            extensions=[TocExtension(permalink=True, permalink_title="パーマリンク", slugify=slugify_unicode)]
        )

    def testPermalinkWithExtendedLatinInID(self):
        self.assertMarkdownRenders(
            '# Théâtre',
            '<h1 id="theatre">'                                                            # noqa
                'Théâtre'                                                                  # noqa
                '<a class="headerlink" href="#theatre" title="Permanent link">&para;</a>'  # noqa
            '</h1>',                                                                       # noqa
            extensions=[TocExtension(permalink=True)]
        )

    def testNl2brCompatibility(self):
        self.assertMarkdownRenders(
            '[TOC]\ntext',
            '<p>[TOC]<br />\ntext</p>',
            extensions=[TocExtension(), Nl2BrExtension()]
        )

    def testTOCWithCustomClass(self):

        self.assertMarkdownRenders(
            self.dedent(
                '''
                [TOC]
                # Header
                '''
            ),
            self.dedent(
                '''
                <div class="custom">
                <ul>
                <li><a href="#header">Header</a></li>
                </ul>
                </div>
                <h1 id="header">Header</h1>
                '''
            ),
            extensions=[TocExtension(toc_class="custom")]
        )

    def testTOCWithCustomClasses(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                [TOC]
                # Header
                '''
            ),
            self.dedent(
                '''
                <div class="custom1 custom2">
                <ul>
                <li><a href="#header">Header</a></li>
                </ul>
                </div>
                <h1 id="header">Header</h1>
                '''
            ),
            extensions=[TocExtension(toc_class="custom1 custom2")]
        )

    def testTOCWithEmptyTitleClass(self):

        self.assertMarkdownRenders(
            self.dedent(
                '''
                [TOC]
                # Header
                '''
            ),
            self.dedent(
                '''
                <div class="toc"><span>ToC</span><ul>
                <li><a href="#header">Header</a></li>
                </ul>
                </div>
                <h1 id="header">Header</h1>
                '''
            ),
            extensions=[TocExtension(title_class="", title='ToC')]
        )

    def testTOCWithCustomTitleClass(self):

        self.assertMarkdownRenders(
            self.dedent(
                '''
                [TOC]
                # Header
                '''
            ),
            self.dedent(
                '''
                <div class="toc"><span class="tocname">ToC</span><ul>
                <li><a href="#header">Header</a></li>
                </ul>
                </div>
                <h1 id="header">Header</h1>
                '''
            ),
            extensions=[TocExtension(title_class="tocname", title='ToC')]
        )

```