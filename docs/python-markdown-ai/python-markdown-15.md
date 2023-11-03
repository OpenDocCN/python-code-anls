# PythonMarkdown源码解析 15

# `/markdown/tests/test_syntax/extensions/test_def_list.py`

该代码是一个Python实现John Gruber所写的Markdown的库，它提供了Markdown解析器和生成器等功能，使得Python开发人员可以轻松地将Markdown内容转换为HTML或另存为PDF文件。

具体来说，该代码包含以下几个主要组件：

1. `markdown`库：该库提供了解析Markdown内容的函数和接口，以及生成Markdown内容的函数。
2. `python-markdown`库：该库是一个Python Markdown库，提供了许多方便的功能，如自定义主题、引用、列表等。
3. `transformers`库：该库是一个Haskell库，提供了一些常用的Markdown解析器和生成器，如`read-in`、`write-in`等。
4. `requests`库：该库是一个用于发送HTTP请求的Python库，可以用于从Markdown仓库中获取或上传文件。

通过组合这些库，该代码可以使得Python开发人员轻松地将Markdown内容转换为HTML或另存为PDF文件。


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

The `def_list` extension is a syntax extension for defining lists in Py火星fmt. It allows you to define a list with a class name and optional numbers before the list, like so:

- [1]
- [1, 2]
- [1, 2, 3]

This syntax is represented in the documentation as follows:
rust
- [ClassName]
- [ListItem]
 [Optional数字]

For example, the following code will define a list with the class `my_class` and the option to be a number:
python
from my_package.models import MyClass

my_list = MyClass(option="number")

This is a valid syntax and will be rendered by Py火星fmt as such:
rust
- MyClass
- [number]

As you can see, the `admonition` syntax extension is also supported in the `def_list` syntax. It is represented in the documentation as follows:
php
admonition
 <p class="admonition-title">Admontion</p>
 <dl>
   <dt>Term</dt>
   <dd>
     <p>Definition</p>
     <ol>
       <li>
         <p>Item 1</p>
         <p>Item 2</p>
       </li>
       <li>
         <p>Item 3</p>
         <p>Continue</p>
       </li>
     </ol>
   </dd>
 </dl>

This syntax is also valid and will be rendered by Py火星fmt as such:
lua
admonition
 <p class="admonition-title">Admontion</p>
 <dl>
   <dt>Term</dt>
   <dd>
     <p>Definition</p>
     <ol>
       <li>
         <p>Item 1</p>
         <p>Item 2</p>
       </li>
       <li>
         <p>Item 3</p>
         <p>Continue</p>
       </li>
     </ol>
   </dd>
 </dl>

Therefore, the `def_list` syntax extensions are supported in Py火星fmt, and you can use them to define your list in a concise and readable way.


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestDefList(TestCase):

    def test_def_list_with_ol(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''

                term

                :   this is a definition for term. it has
                    multiple lines in the first paragraph.

                    1.  first thing

                        first thing details in a second paragraph.

                    1.  second thing

                        second thing details in a second paragraph.

                    1.  third thing

                        third thing details in a second paragraph.
                '''
            ),
            self.dedent(
                '''
                <dl>
                <dt>term</dt>
                <dd>
                <p>this is a definition for term. it has
                multiple lines in the first paragraph.</p>
                <ol>
                <li>
                <p>first thing</p>
                <p>first thing details in a second paragraph.</p>
                </li>
                <li>
                <p>second thing</p>
                <p>second thing details in a second paragraph.</p>
                </li>
                <li>
                <p>third thing</p>
                <p>third thing details in a second paragraph.</p>
                </li>
                </ol>
                </dd>
                </dl>
                '''
            ),
            extensions=['def_list']
        )

    def test_def_list_with_ul(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''

                term

                :   this is a definition for term. it has
                    multiple lines in the first paragraph.

                    -   first thing

                        first thing details in a second paragraph.

                    -   second thing

                        second thing details in a second paragraph.

                    -   third thing

                        third thing details in a second paragraph.
                '''
            ),
            self.dedent(
                '''
                <dl>
                <dt>term</dt>
                <dd>
                <p>this is a definition for term. it has
                multiple lines in the first paragraph.</p>
                <ul>
                <li>
                <p>first thing</p>
                <p>first thing details in a second paragraph.</p>
                </li>
                <li>
                <p>second thing</p>
                <p>second thing details in a second paragraph.</p>
                </li>
                <li>
                <p>third thing</p>
                <p>third thing details in a second paragraph.</p>
                </li>
                </ul>
                </dd>
                </dl>
                '''
            ),
            extensions=['def_list']
        )

    def test_def_list_with_nesting(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''

                term

                :   this is a definition for term. it has
                    multiple lines in the first paragraph.

                    1.  first thing

                        first thing details in a second paragraph.

                        -   first nested thing

                            second nested thing details
                '''
            ),
            self.dedent(
                '''
                <dl>
                <dt>term</dt>
                <dd>
                <p>this is a definition for term. it has
                multiple lines in the first paragraph.</p>
                <ol>
                <li>
                <p>first thing</p>
                <p>first thing details in a second paragraph.</p>
                <ul>
                <li>
                <p>first nested thing</p>
                <p>second nested thing details</p>
                </li>
                </ul>
                </li>
                </ol>
                </dd>
                </dl>
                '''
            ),
            extensions=['def_list']
        )

    def test_def_list_with_nesting_self(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''

                term

                :   this is a definition for term. it has
                    multiple lines in the first paragraph.

                    inception

                    :   this is a definition for term. it has
                        multiple lines in the first paragraph.

                        - bullet point

                          another paragraph
                '''
            ),
            self.dedent(
                '''
                <dl>
                <dt>term</dt>
                <dd>
                <p>this is a definition for term. it has
                multiple lines in the first paragraph.</p>
                <dl>
                <dt>inception</dt>
                <dd>
                <p>this is a definition for term. it has
                multiple lines in the first paragraph.</p>
                <ul>
                <li>bullet point</li>
                </ul>
                <p>another paragraph</p>
                </dd>
                </dl>
                </dd>
                </dl>
                '''
            ),
            extensions=['def_list']
        )

    def test_def_list_unreasonable_nesting(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''

                turducken

                :   this is a definition for term. it has
                    multiple lines in the first paragraph.

                    1.  ordered list

                        - nested list

                            term

                            :   definition

                                -   item 1 paragraph 1

                                    item 1 paragraph 2
                '''
            ),
            self.dedent(
                '''
                <dl>
                <dt>turducken</dt>
                <dd>
                <p>this is a definition for term. it has
                multiple lines in the first paragraph.</p>
                <ol>
                <li>
                <p>ordered list</p>
                <ul>
                <li>
                <p>nested list</p>
                <dl>
                <dt>term</dt>
                <dd>
                <p>definition</p>
                <ul>
                <li>
                <p>item 1 paragraph 1</p>
                <p>item 1 paragraph 2</p>
                </li>
                </ul>
                </dd>
                </dl>
                </li>
                </ul>
                </li>
                </ol>
                </dd>
                </dl>
                '''
            ),
            extensions=['def_list']
        )

    def test_def_list_nested_admontions(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                term

                :   definition

                    !!! note "Admontion"

                        term

                        :   definition

                            1.  list

                                continue
                '''
            ),
            self.dedent(
                '''
                <dl>
                <dt>term</dt>
                <dd>
                <p>definition</p>
                <div class="admonition note">
                <p class="admonition-title">Admontion</p>
                <dl>
                <dt>term</dt>
                <dd>
                <p>definition</p>
                <ol>
                <li>
                <p>list</p>
                <p>continue</p>
                </li>
                </ol>
                </dd>
                </dl>
                </div>
                </dd>
                </dl>
                '''
            ),
            extensions=['def_list', 'admonition']
        )

```

# `/markdown/tests/test_syntax/extensions/test_fenced_code.py`

这是一段Python代码，它是一个Markdown渲染器，它将Markdown文档转换为HTML格式的内容。下面是这段代码的一些作用：

1. 将Markdown文档转换为HTML格式的内容：这个代码段落的核心作用是将其所在的Markdown文档转换为HTML格式的内容。
2. 支持Markdown语法：这个代码段落可以读取和解析Markdown语法，使得它成为一个Markdown渲染器。
3. 提供了一个Python实现：这个代码段落提供了一个Python实现的Markdown渲染器，这使得用户可以使用Python来编写Markdown文档，而不是仅仅使用Markdown编辑器。
4. 可以进一步扩展：虽然这个代码段落提供了Markdown渲染器的基本功能，但你可以通过向其添加更多的扩展来进一步扩展它的功能，比如添加自定义主题、布局等。
5. 遵循MIT许可证：这个代码段落遵循了MIT许可证，这意味着任何人都可以自由地使用、修改和分发它，前提是你别有用得不当。


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

这段代码是一个Python文件，其中包含一个测试套件（test-extension）以及一个Markdown文件。它通过引用自定义的Markdown标签来修改Markdown的语法，从而实现了一些特殊的功能。

具体来说，这段代码的作用如下：

1. 将Python的日期和版本号暴露给外部使用。
2. 导入Markdown并导入了Markdown的测试工具类，以及一个名为`codehilite`的扩展，用于语法高亮显示。
3. 通过一个名为`os`的系统级导入，使系统可以访问Python的`os`模块。
4. 定义了一个`test_extension`的类，其中包含一个`main`方法。
5. 在`main`方法中，使用`pygments`库将Markdown的扩展加载到Python中，并定义了一个`pygments.extensions.codehilite`的实例。
6. 在`codehilite`的扩展中，定义了一个新的`file_extension_replacer`方法，用于在Markdown文件中替换指定的扩展名（例如，`.md`）。
7. 在`test_extension`的`main`方法的`pygments.extensions.codehilite`部分，将这个新的替换器实例与Python的`os`模块结合起来，用于在运行测试时替换`.md`扩展名。
8. 最后，通过测试套件的运行，验证`test_extension`是否能够正常工作。


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase
import markdown
import markdown.extensions.codehilite
import os

try:
    import pygments  # noqa
    import pygments.formatters  # noqa
    has_pygments = True
```

It looks like this is a test of the `FencedCodeExtension` class from the `markdown` package. This extension allows you to include code snippets enclosed in `<code>` tags within the same document that are formatted as if they were running on a Python interpreter. The `FencedCodeExtension` class has two methods, `FencedCodeExtension.FencedCodeExtension` and `FamedCodeExtension.FamedCodeExtension`, which can be used to enable or disable fencing for code snippets.

The `FencedCodeExtension.FancedCodeExtension` method takes a single argument, `lang_prefix`, which specifies the language prefix for the fencing. If `lang_prefix` is not specified, the default prefix is `''`. This means that any fencing code will be written in the language name without any prefix.

The `FamedCodeExtension.FamedCodeExtension` method is a subclass of `FencedCodeExtension.FancedCodeExtension` and adds an `alt_prefix` argument. This specifies the alternative prefix for the fencing code, which should be specified as a URL relative to the `base_url` of the project.

The `FencedCodeExtension` class also includes an `extensions` list, which specifies an extension to be applied to the `FamedCodeExtension` if it is启用. The extensions are `FencedCodeExtension.FencedCodeExtension` and `FamedCodeExtension.FamedCodeExtension`.


```py
except ImportError:
    has_pygments = False

# The version required by the tests is the version specified and installed in the `pygments` tox environment.
# In any environment where the `PYGMENTS_VERSION` environment variable is either not defined or doesn't
# match the version of Pygments installed, all tests which rely in Pygments will be skipped.
required_pygments_version = os.environ.get('PYGMENTS_VERSION', '')


class TestFencedCode(TestCase):

    def testBasicFence(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                A paragraph before a fenced code block:

                ```
                Fenced code block
                ```py
                '''
            ),
            self.dedent(
                '''
                <p>A paragraph before a fenced code block:</p>
                <pre><code>Fenced code block
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testNestedFence(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ````

                ```py
                ````
                '''
            ),
            self.dedent(
                '''
                <pre><code>
                ```py
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedTildes(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ~~~
                # Arbitrary code
                ``` # these backticks will not close the block
                ~~~
                '''
            ),
            self.dedent(
                '''
                <pre><code># Arbitrary code
                ```py # these backticks will not close the block
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedLanguageNoDot(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` python
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedLanguageWithDot(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` .python
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def test_fenced_code_in_raw_html(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                <details>
                ```
                Begone placeholders!
                ```py
                </details>
                """
            ),
            self.dedent(
                """
                <details>

                <pre><code>Begone placeholders!
                </code></pre>

                </details>
                """
            ),
            extensions=['fenced_code']
        )

    def testFencedLanguageInAttr(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` {.python}
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedMultipleClassesInAttr(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` {.python .foo .bar}
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre class="foo bar"><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedIdInAttr(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { #foo }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre id="foo"><code># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedIdAndLangInAttr(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python #foo }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre id="foo"><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedIdAndLangAndClassInAttr(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python #foo .bar }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre id="foo" class="bar"><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedLanguageIdAndPygmentsDisabledInAttrNoCodehilite(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python #foo use_pygments=False }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre id="foo"><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedLanguageIdAndPygmentsEnabledInAttrNoCodehilite(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python #foo use_pygments=True }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre id="foo"><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code']
        )

    def testFencedLanguageNoCodehiliteWithAttrList(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python foo=bar }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="language-python" foo="bar"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code', 'attr_list']
        )

    def testFencedLanguagePygmentsDisabledInAttrNoCodehiliteWithAttrList(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python foo=bar use_pygments=False }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="language-python" foo="bar"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code', 'attr_list']
        )

    def testFencedLanguagePygmentsEnabledInAttrNoCodehiliteWithAttrList(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python foo=bar use_pygments=True }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code', 'attr_list']
        )

    def testFencedLanguageNoPrefix(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` python
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="python"># Some python code
                </code></pre>
                '''
            ),
            extensions=[markdown.extensions.fenced_code.FencedCodeExtension(lang_prefix='')]
        )

    def testFencedLanguageAltPrefix(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` python
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="lang-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=[markdown.extensions.fenced_code.FencedCodeExtension(lang_prefix='lang-')]
        )

    def testFencedCodeEscapedAttrs(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { ."weird #"foo bar=">baz }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre id="&quot;foo"><code class="language-&quot;weird" bar="&quot;&gt;baz"># Some python code
                </code></pre>
                '''
            ),
            extensions=['fenced_code', 'attr_list']
        )


```

This is a test case for the `render_pygments` function in Pygments. The function renders the given markdown string with the specified Pygments formatter on the assumption that the Pygments formatter is installed and configured correctly.

The first test case (`testBasicFormatter`) checks that the expected output is rendered for a simple markdown string with a default Pygments formatter. The expected output is that the markdown string is rendered with the default formatter:
php
<pre class="codehilite"><code>hello world
hello another world
</code></pre>

The second test case (`testInvalidCustomFormatter`) checks that the expected output is not rendered for an invalid Pygments formatter. The expected output is that the markdown string is not rendered with any formatter:
php
<pre class="codehilite"><code>hello world
hello another world
</code></pre>

The third test case (`testCustomFormatter`) checks that the expected output is rendered for a markdown string with a custom Pygments formatter. The expected output is that the markdown string is rendered with the specified formatter:
php
<pre class="codehilite"><code>
物的ctions and attributes
"order",
       "柏   lex   腾

<code>hel



```py
class TestFencedCodeWithCodehilite(TestCase):

    def setUp(self):
        if has_pygments and pygments.__version__ != required_pygments_version:
            self.skipTest(f'Pygments=={required_pygments_version} is required')

    def test_shebang(self):

        if has_pygments:
            expected = '''
            <div class="codehilite"><pre><span></span><code>#!test
            </code></pre></div>
            '''
        else:
            expected = '''
            <pre class="codehilite"><code>#!test
            </code></pre>
            '''

        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```
                #!test
                ```py
                '''
            ),
            self.dedent(
                expected
            ),
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False),
                'fenced_code'
            ]
        )

    def testFencedCodeWithHighlightLines(self):
        if has_pygments:
            expected = self.dedent(
                '''
                <div class="codehilite"><pre><span></span><code><span class="hll">line 1
                </span>line 2
                <span class="hll">line 3
                </span></code></pre></div>
                '''
            )
        else:
            expected = self.dedent(
                    '''
                    <pre class="codehilite"><code>line 1
                    line 2
                    line 3
                    </code></pre>
                    '''
                )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```hl_lines="1 3"
                line 1
                line 2
                line 3
                ```py
                '''
            ),
            expected,
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False),
                'fenced_code'
            ]
        )

    def testFencedLanguageAndHighlightLines(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code>'
                '<span class="hll"><span class="n">line</span> <span class="mi">1</span>\n'
                '</span><span class="n">line</span> <span class="mi">2</span>\n'
                '<span class="hll"><span class="n">line</span> <span class="mi">3</span>\n'
                '</span></code></pre></div>'
            )
        else:
            expected = self.dedent(
                    '''
                    <pre class="codehilite"><code class="language-python">line 1
                    line 2
                    line 3
                    </code></pre>
                    '''
                )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` .python hl_lines="1 3"
                line 1
                line 2
                line 3
                ```py
                '''
            ),
            expected,
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False),
                'fenced_code'
            ]
        )

    def testFencedLanguageAndPygmentsDisabled(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` .python
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(use_pygments=False),
                'fenced_code'
            ]
        )

    def testFencedLanguageDoubleEscape(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code>'
                '<span class="p">&lt;</span><span class="nt">span</span>'
                '<span class="p">&gt;</span>This<span class="ni">&amp;amp;</span>'
                'That<span class="p">&lt;/</span><span class="nt">span</span>'
                '<span class="p">&gt;</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-html">'
                '&lt;span&gt;This&amp;amp;That&lt;/span&gt;\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```html
                <span>This&amp;That</span>
                ```py
                '''
            ),
            expected,
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(),
                'fenced_code'
            ]
        )

    def testFencedAmps(self):
        if has_pygments:
            expected = self.dedent(
                '''
                <div class="codehilite"><pre><span></span><code>&amp;
                &amp;amp;
                &amp;amp;amp;
                </code></pre></div>
                '''
            )
        else:
            expected = self.dedent(
                '''
                <pre class="codehilite"><code class="language-text">&amp;
                &amp;amp;
                &amp;amp;amp;
                </code></pre>
                '''
            )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```text
                &
                &amp;
                &amp;amp;
                ```py
                '''
            ),
            expected,
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(),
                'fenced_code'
            ]
        )

    def testFencedCodeWithHighlightLinesInAttr(self):
        if has_pygments:
            expected = self.dedent(
                '''
                <div class="codehilite"><pre><span></span><code><span class="hll">line 1
                </span>line 2
                <span class="hll">line 3
                </span></code></pre></div>
                '''
            )
        else:
            expected = self.dedent(
                    '''
                    <pre class="codehilite"><code>line 1
                    line 2
                    line 3
                    </code></pre>
                    '''
                )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```{ hl_lines="1 3" }
                line 1
                line 2
                line 3
                ```py
                '''
            ),
            expected,
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False),
                'fenced_code'
            ]
        )

    def testFencedLanguageAndHighlightLinesInAttr(self):
        if has_pygments:
            expected = (
                '<div class="codehilite"><pre><span></span><code>'
                '<span class="hll"><span class="n">line</span> <span class="mi">1</span>\n'
                '</span><span class="n">line</span> <span class="mi">2</span>\n'
                '<span class="hll"><span class="n">line</span> <span class="mi">3</span>\n'
                '</span></code></pre></div>'
            )
        else:
            expected = self.dedent(
                    '''
                    <pre class="codehilite"><code class="language-python">line 1
                    line 2
                    line 3
                    </code></pre>
                    '''
                )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python hl_lines="1 3" }
                line 1
                line 2
                line 3
                ```py
                '''
            ),
            expected,
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False),
                'fenced_code'
            ]
        )

    def testFencedLanguageIdInAttrAndPygmentsDisabled(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python #foo }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre id="foo"><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(use_pygments=False),
                'fenced_code'
            ]
        )

    def testFencedLanguageIdAndPygmentsDisabledInAttr(self):
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python #foo use_pygments=False }
                # Some python code
                ```py
                '''
            ),
            self.dedent(
                '''
                <pre id="foo"><code class="language-python"># Some python code
                </code></pre>
                '''
            ),
            extensions=['codehilite', 'fenced_code']
        )

    def testFencedLanguageAttrCssclass(self):
        if has_pygments:
            expected = self.dedent(
                '''
                <div class="pygments"><pre><span></span><code><span class="c1"># Some python code</span>
                </code></pre></div>
                '''
            )
        else:
            expected = (
                '<pre class="pygments"><code class="language-python"># Some python code\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python css_class='pygments' }
                # Some python code
                ```py
                '''
            ),
            expected,
            extensions=['codehilite', 'fenced_code']
        )

    def testFencedLanguageAttrLinenums(self):
        if has_pygments:
            expected = (
                '<table class="codehilitetable"><tr>'
                '<td class="linenos"><div class="linenodiv"><pre>1</pre></div></td>'
                '<td class="code"><div class="codehilite"><pre><span></span>'
                '<code><span class="c1"># Some python code</span>\n'
                '</code></pre></div>\n'
                '</td></tr></table>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-python linenums"># Some python code\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python linenums=True }
                # Some python code
                ```py
                '''
            ),
            expected,
            extensions=['codehilite', 'fenced_code']
        )

    def testFencedLanguageAttrGuesslang(self):
        if has_pygments:
            expected = self.dedent(
                '''
                <div class="codehilite"><pre><span></span><code># Some python code
                </code></pre></div>
                '''
            )
        else:
            expected = (
                '<pre class="codehilite"><code># Some python code\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { guess_lang=False }
                # Some python code
                ```py
                '''
            ),
            expected,
            extensions=['codehilite', 'fenced_code']
        )

    def testFencedLanguageAttrNoclasses(self):
        if has_pygments:
            expected = (
                '<div class="codehilite" style="background: #f8f8f8">'
                '<pre style="line-height: 125%; margin: 0;"><span></span><code>'
                '<span style="color: #408080; font-style: italic"># Some python code</span>\n'
                '</code></pre></div>'
            )
        else:
            expected = (
                '<pre class="codehilite"><code class="language-python"># Some python code\n'
                '</code></pre>'
            )
        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python noclasses=True }
                # Some python code
                ```py
                '''
            ),
            expected,
            extensions=['codehilite', 'fenced_code']
        )

    def testFencedMultipleBlocksSameStyle(self):
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
            expected = '''
            <pre class="codehilite"><code class="language-python"># First Code Block
            </code></pre>

            <p>Normal paragraph</p>
            <pre class="codehilite"><code class="language-python"># Second Code Block
            </code></pre>
            '''

        self.assertMarkdownRenders(
            self.dedent(
                '''
                ``` { .python }
                # First Code Block
                ```py

                Normal paragraph

                ``` { .python }
                # Second Code Block
                ```py
                '''
            ),
            self.dedent(
                expected
            ),
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(pygments_style="native", noclasses=True),
                'fenced_code'
            ]
        )

    def testCustomPygmentsFormatter(self):
        if has_pygments:
            class CustomFormatter(pygments.formatters.HtmlFormatter):
                def wrap(self, source, outfile):
                    return self._wrap_div(self._wrap_code(source))

                def _wrap_code(self, source):
                    yield 0, '<code>'
                    for i, t in source:
                        if i == 1:
                            t += '<br>'
                        yield i, t
                    yield 0, '</code>'

            expected = '''
            <div class="codehilite"><code>hello world
            <br>hello another world
            <br></code></div>
            '''

        else:
            CustomFormatter = None
            expected = '''
            <pre class="codehilite"><code>hello world
            hello another world
            </code></pre>
            '''

        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```
                hello world
                hello another world
                ```py
                '''
            ),
            self.dedent(
                expected
            ),
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(
                    pygments_formatter=CustomFormatter,
                    guess_lang=False,
                ),
                'fenced_code'
            ]
        )

    def testPygmentsAddLangClassFormatter(self):
        if has_pygments:
            class CustomAddLangHtmlFormatter(pygments.formatters.HtmlFormatter):
                def __init__(self, lang_str='', **options):
                    super().__init__(**options)
                    self.lang_str = lang_str

                def _wrap_code(self, source):
                    yield 0, f'<code class="{self.lang_str}">'
                    yield from source
                    yield 0, '</code>'

            expected = '''
                <div class="codehilite"><pre><span></span><code class="language-text">hello world
                hello another world
                </code></pre></div>
                '''
        else:
            CustomAddLangHtmlFormatter = None
            expected = '''
                <pre class="codehilite"><code class="language-text">hello world
                hello another world
                </code></pre>
                '''

        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```text
                hello world
                hello another world
                ```py
                '''
            ),
            self.dedent(
                expected
            ),
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(
                    guess_lang=False,
                    pygments_formatter=CustomAddLangHtmlFormatter,
                ),
                'fenced_code'
            ]
        )

    def testSvgCustomPygmentsFormatter(self):
        if has_pygments:
            expected = '''
            <?xml version="1.0"?>
            <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
            <svg xmlns="http://www.w3.org/2000/svg">
            <g font-family="monospace" font-size="14px">
            <text x="0" y="14" xml:space="preserve">hello&#160;world</text>
            <text x="0" y="33" xml:space="preserve">hello&#160;another&#160;world</text>
            <text x="0" y="52" xml:space="preserve"></text></g></svg>
            '''

        else:
            expected = '''
            <pre class="codehilite"><code>hello world
            hello another world
            </code></pre>
            '''

        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```
                hello world
                hello another world
                ```py
                '''
            ),
            self.dedent(
                expected
            ),
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(
                    pygments_formatter='svg',
                    linenos=False,
                    guess_lang=False,
                ),
                'fenced_code'
            ]
        )

    def testInvalidCustomPygmentsFormatter(self):
        if has_pygments:
            expected = '''
            <div class="codehilite"><pre><span></span><code>hello world
            hello another world
            </code></pre></div>
            '''

        else:
            expected = '''
            <pre class="codehilite"><code>hello world
            hello another world
            </code></pre>
            '''

        self.assertMarkdownRenders(
            self.dedent(
                '''
                ```
                hello world
                hello another world
                ```py
                '''
            ),
            self.dedent(
                expected
            ),
            extensions=[
                markdown.extensions.codehilite.CodeHiliteExtension(
                    pygments_formatter='invalid',
                    guess_lang=False,
                ),
                'fenced_code'
            ]
        )

```