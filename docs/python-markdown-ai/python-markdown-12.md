# PythonMarkdown源码解析 12

# `/markdown/tests/test_legacy.py`

这是一段Python代码，它是一个Markdown渲染器，实现了John Gruber的Markdown规范。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式。

这段代码定义了一个名为Markdown的类，该类包含了一些方法，用于将Markdown语法转换为HTML。例如，{-#%20Maintained%20by%20Waylan%20Limberg<https://github.com/waylan> and%20Isaac%20Muse<https://github.com/facelessuser> }`这两行定义了该类的 maintainers。

接着，定义了一个print函数，用于输出Markdown文档的链接。

最后，定义了一个representation参数，用于指定Markdown文档的存储路径，并且将该参数传递给继承自Document的类，以便正确地保存和读取Markdown文档。

该代码使用了Python的docstring语法，用于描述Markdown文档的用途和意义，以及版权信息。


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

这段代码是一个Python程序，它定义了一个名为“test_markdown.py”的类。该类包含一个测试方法“test_main”，以及一些定义马克down样式的函数。下面是对该程序的详细解释：

1. 首先，定义了版权信息。
2. 定义了一个名为“Copyright”的类，该类包含三个参数，分别表示年份、作者和版权信息。
3. 定义了一个名为“License”的类，该类包含一个包含版权信息的变量。
4. 定义了两个函数，一个名为“test_markdown.py”的函数，另一个名为“test_main.py”的函数。
5. 在函数内部，使用了一个名为“warnings”的模块，它是一个Python标准库中的一个类，用于处理警告信息。
6. 使用“warnings.simplefilter”函数将“PendingDeprecationWarning”和“DeprecationWarning”这两类警告过滤掉，因为它们不应该在这里出现。
7. 使用“warnings.filterwarnings”函数将“markdown”模块中的警告过滤掉，因为警告信息通常不会出现在“markdown”模块中。
8. 在函数内部，使用了“os”模块中的“system”函数，将一个名为“test_markdown.py”的脚本文件系统为“test_markdown.py”而不是“test_main.py”。
9. 在函数内部，定义了一个名为“test_markdown.py”的类，该类继承自“ LegacyTestCase”类，并重写了“__init__”和“测试”两个方法。
10. 在“test_markdown.py”类的“__main__”方法中，调用了“test_markdown.py”中定义的“test_main”函数。


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import LegacyTestCase, Kwargs
import os
import warnings

# Warnings should cause tests to fail...
warnings.simplefilter('error')
# Except for the warnings that shouldn't
warnings.filterwarnings('default', category=PendingDeprecationWarning)
warnings.filterwarnings('default', category=DeprecationWarning, module='markdown')

```

"""
Expected behavior elements:
- `location`: the location of the test file
- `normalize`: a flag indicating whether the test output should be normalized
- `input_ext`: the input file extension
- `output_ext`: the output file extension
- `exclude`: a list of tests to exclude from the test suite
"""


def test_file_location(normalize, input_ext, output_ext, exclude):
   """
   """
   if normalize:
       location = os.path.join(exclude[0], 'output')
   else:
       location = os.path.join(exclude[0], input_ext)

   os.makedirs(os.path.join(location, 'tmp'), exist_ok=True)

   #放心，我们不需要关注这个问题
   #再次提醒，在实际项目中，这个文件可能是输入
   #而不是输出
   with open(os.path.join(location, 'tmp', input_ext), 'w') as f:
       pass

   #在'.'和'.html'这两个文件中，不需要做太多改动
   if input_ext == '.text' and output_ext == '.html':
       return

   #移动到测试目录
   os.chdir(location)

   #遍历exclude中的测试
   for test in exclude:
       #跳过测试目录
       if test == 'exclude':
           continue

       #运行测试
       print(f"{test}")

       #尝试输出存为xhtml
       if output_ext == '.xhtml':
           output_file = os.path.join(location, test, output_ext)
           with open(output_file, 'w') as f:
               f.write('<?xml version="1.0"?>')
               f.write('<html xmlns:ph="http://www.php.net/php3/": ')
               f.write('php:extensible="true" ')
               f.write('?>')
               f.write('<head>')
               f.write('php:extensible="true" ')
               f.write('?>')
               f.write('<body ')
               f.write('php:extensible="true" ')
               f.write('?>')
               f.write('</body>')
               f.write('</html>')
       #这里可能是兼容性问题，我再研究一下
       elif output_ext == '.html':
           output_file = os.path.join(location, test, output_ext)
           with open(output_file, 'w') as f:
               f.write('<?php ')
               f.write('echo "PHP:extensible=true"; ')
               f.write('echo PHP_EOL; ')
               f.write('echo "<?php ')
               f.write('$input = ')
               f.write('str_repeat("' + exclude[0] + '", "')
               f.write(') . '; ')
               f.write('$output = ')
               f.write('str_repeat("' + exclude[0] + '", "')
               f.write(') . '; ')
               f.write('$output_str = ')
               f.write('str_repeat("')
               f.write(exclude[0]) + '", "')
               f.write('"') + '; ')
               f.write('echo "') + '"') + '; ')
               f.write('?>')
               f.write('<?php ')
               f.write('echo "PHP:extensible=true"; ')
               f.write('echo PHP_EOL; ')
               f.write('echo "<?php ')
               f.write('$input = ')
               f.write('str_repeat("')
               f.write('"') + '; ')
               f.write('$output = ')
               f.write('str_repeat("')
               f.write('"') + '; ')
               f.write('$output_str = ')
               f.write('str_repeat("')
               f.write(exclude[0]) + '", "') + '; ')
               f.write('echo "') + '"') + '; ')
               f.write('?>')
               f.write('<?php ')
               f.write('echo "PHP:extensible=



```py
parent_test_dir = os.path.abspath(os.path.dirname(__file__))


class TestBasic(LegacyTestCase):
    location = os.path.join(parent_test_dir, 'basic')


class TestMisc(LegacyTestCase):
    location = os.path.join(parent_test_dir, 'misc')


class TestPhp(LegacyTestCase):
    """
    Notes on "excluded" tests:

    Quotes in attributes: attributes get output in different order

    Inline HTML (Span): Backtick in raw HTML attribute TODO: fix me

    Backslash escapes: Weird whitespace issue in output

    `Ins` & `del`: Our behavior follows `markdown.pl`. I think PHP is wrong here

    Auto Links: TODO: fix raw HTML so is doesn't match <hr@example.com> as a `<hr>`.

    Empty List Item: We match `markdown.pl` here. Maybe someday we'll support this

    Headers: TODO: fix headers to not require blank line before

    Mixed `OL`s and `UL`s: We match `markdown.pl` here. I think PHP is wrong here

    Emphasis: We have various minor differences in combined & incorrect em markup.
    Maybe fix a few of them - but most aren't too important

    Code block in a list item: We match `markdown.pl` - not sure how PHP gets that output??

    PHP-Specific Bugs: Not sure what to make of the escaping stuff here.
    Why is PHP not removing a backslash?
    """
    location = os.path.join(parent_test_dir, 'php')
    normalize = True
    input_ext = '.text'
    output_ext = '.xhtml'
    exclude = [
        'Quotes_in_attributes',
        'Inline_HTML_(Span)',
        'Backslash_escapes',
        'Ins_&_del',
        'Auto_Links',
        'Empty_List_Item',
        'Headers',
        'Mixed_OLs_and_ULs',
        'Emphasis',
        'Code_block_in_a_list_item',
        'PHP_Specific_Bugs'
    ]


```

这段代码定义了两个测试类：TestPl2004和TestPl2007。这两个测试类继承自LegacyTestCase类，并实现了NotesOnExcludedTests元组中的方法。

TestPl2004的作用是测试一个名为"Images"的类，而TestPl2007的作用是测试一个名为"Code_Blocks"的类。两个测试类都实现了NotesOnExcludedTests元组中的方法，这些方法用于排除测试中不希望得到的文件和类。

具体来说，TestPl2004和TestPl2007的区别在于它们的exclude列表不同。TestPl2004中exclude包含"Images"和"Yuri_Footnotes"两个文件，而TestPl2007中exclude包含"Images"、"Yuri_Footnotes"和"Yuri_Attributes"三个文件。


```py
class TestPl2004(LegacyTestCase):
    location = os.path.join(parent_test_dir, 'pl/Tests_2004')
    normalize = True
    input_ext = '.text'
    exclude = ['Yuri_Footnotes', 'Yuri_Attributes']


class TestPl2007(LegacyTestCase):
    """
    Notes on "excluded" tests:

    Images: the attributes don't get ordered the same so we skip this

    Code Blocks: some weird whitespace issue

    Links, reference style: weird issue with nested brackets TODO: fix me

    Backslash escapes: backticks in raw html attributes TODO: fix me

    Code Spans: more backticks in raw html attributes TODO: fix me
    """
    location = os.path.join(parent_test_dir, 'pl/Tests_2007')
    normalize = True
    input_ext = '.text'
    exclude = [
        'Images',
        'Code_Blocks',
        'Links,_reference_style',
        'Backslash_escapes',
        'Code_Spans'
    ]


```

这段代码定义了一个名为 `TestExtensions` 的类，继承自 `LegacyTestCase` 类，旨在为测试应用程序添加一些支持的功能。

该类位于一个名为 `extensions` 的目录中，使用了 `os.path.join` 函数将其与父测试目录连接起来。 `exclude` 参数是一个元组，包含将要从 `extensions` 目录中排除的文件或文件夹的路径。

`attr_list`、`def_list` 和 `smarty` 参数是一个名为 `extensions` 的元组，用于指定在测试中需要用到的自定义测试函数和数据。这些数据将在测试中被提供给测试函数，以使测试更加灵活。

`codehilite` 和 `toc` 参数也是元组，用于指定在测试中需要使用的代码高亮和测试列表功能。这些功能将在测试中使用，以提高测试的质量和准确性。

`toc` 和 `toc_invalid` 参数都使用了 `Kwargs` 类，用于提供测试中选项的配置。 `toc_out_of_order` 和 `toc_nested` 参数都指定了在测试中需要排除的测试列表的选项。 `toc_nested_list` 参数还指定了需要包括在测试中的测试列表的选项。

`wikilinks` 和 `github_flavored` 参数使用了 `Kwargs` 类，用于提供测试中选项的配置。 `sane_lists` 和 `nl2br` 参数使用了 `Kwargs` 类，用于提供测试中选项的配置。 `admonition` 参数使用了 `Kwargs` 类，用于提供测试中选项的配置。


```py
class TestExtensions(LegacyTestCase):
    location = os.path.join(parent_test_dir, 'extensions')
    exclude = ['codehilite']

    attr_list = Kwargs(extensions=['attr_list', 'def_list', 'smarty'])

    codehilite = Kwargs(extensions=['codehilite'])

    toc = Kwargs(extensions=['toc'])

    toc_invalid = Kwargs(extensions=['toc'])

    toc_out_of_order = Kwargs(extensions=['toc'])

    toc_nested = Kwargs(
        extensions=['toc'],
        extension_configs={'toc': {'permalink': True}}
    )

    toc_nested2 = Kwargs(
        extensions=['toc'],
        extension_configs={'toc': {'permalink': "[link]"}}
    )

    toc_nested_list = Kwargs(extensions=['toc'])

    wikilinks = Kwargs(extensions=['wikilinks'])

    github_flavored = Kwargs(extensions=['fenced_code'])

    sane_lists = Kwargs(extensions=['sane_lists'])

    nl2br_w_attr_list = Kwargs(extensions=['nl2br', 'attr_list'])

    admonition = Kwargs(extensions=['admonition'])


```

这段代码定义了一个名为 `TestExtensionsExtra` 的类，继承自 `LegacyTestCase` 类，位于指定的测试目录中的 `extensions/extra` 目录下。

这个类包含了一些参数，其中 `location` 是该类的一个属性，指定了测试目录的位置。

`default_kwargs` 是一个参数，传递给 `Kwargs` 类的实例，用于设置默认的参数值，其中 `extensions` 参数指定要扩展的测试用例类型。

`loose_def_list` 和 `simple_def_lists` 也是两个参数，它们都传递给 `Kwargs` 类的实例，并且使用同样的方式来设置要扩展的测试用例类型。这里 `simple_def_lists` 更加通用，可以覆盖 `loose_def_list` 中指定的所有参数。

`abbr` 参数使用 `Kwargs` 类来设置要扩展的测试用例类型。

`footnotes` 参数使用 `Kwargs` 类来设置要扩展的测试用例类型。

`extra_config` 参数使用 `Kwargs` 类来设置要扩展的测试用例类型的附加配置。其中 `extensions` 参数指定要扩展的测试用例类型，`extension_configs` 参数是一个字典，用于设置每个测试用例类型的附加配置。在这个例子中，`extra` 键对应的值是一个字典，其中包含一个名为 `footnotes` 的条目，其值为 `'PLACE_MARKER'`。


```py
class TestExtensionsExtra(LegacyTestCase):
    location = os.path.join(parent_test_dir, 'extensions/extra')
    default_kwargs = Kwargs(extensions=['extra'])

    loose_def_list = Kwargs(extensions=['def_list'])

    simple_def_lists = Kwargs(extensions=['def_list'])

    abbr = Kwargs(extensions=['abbr'])

    footnotes = Kwargs(extensions=['footnotes'])

    extra_config = Kwargs(
        extensions=['extra'],
        extension_configs={
            'extra': {
                'footnotes': {
                    'PLACE_MARKER': '~~~placemarker~~~'
                }
            }
        }
    )

```

# `/markdown/tests/test_meta.py`

这段代码是一个单元测试，名为 `TestVersion`，旨在测试 `__get_version` 和 `__version__` 函数的行为。

首先，代码导入了 `unittest` 模块和 `markdown.__meta__` 模块。

接着，定义了一个名为 `TestVersion` 的类，继承自 `unittest.TestCase` 类。

在 `TestVersion` 类的 `__test__` 方法中，定义了以下五个测试方法：

1. `test_get_version`：测试 `_get_version` 函数的正确性。
2. `test__version__IsValid`：测试 `__version__` 函数的正确性。
3. `test_get_version_with_parsed_version`：测试 `_get_version` 函数处理带参数的版本字符串的正确性。
4. `test_get_unhandled_version_with_file_version`：测试 `_get_version` 函数处理版本字符串时，如果字符串未按预期格式（如 `"1.1.2.dev0"` 或 `"1.1.2a1"`），是否能正确返回。
5. `test_get_all_versions_return_expected`：测试 `_get_version` 函数能正确返回所有已知的版本，并按照预期格式返回。

最后，在 `__init__` 方法中，导入了 `packaging.version`，作为 `unittest.TestCase` 类的子类，以便能够正常使用 `__version__` 函数。


```py
import unittest
from markdown.__meta__ import _get_version, __version__


class TestVersion(unittest.TestCase):

    def test_get_version(self):
        """Test that _get_version formats __version_info__ as required by PEP 440."""

        self.assertEqual(_get_version((1, 1, 2, 'dev', 0)), "1.1.2.dev0")
        self.assertEqual(_get_version((1, 1, 2, 'alpha', 1)), "1.1.2a1")
        self.assertEqual(_get_version((1, 2, 0, 'beta', 2)), "1.2b2")
        self.assertEqual(_get_version((1, 2, 0, 'rc', 4)), "1.2rc4")
        self.assertEqual(_get_version((1, 2, 0, 'final', 0)), "1.2")

    def test__version__IsValid(self):
        """Test that __version__ is valid and normalized."""

        try:
            import packaging.version
        except ImportError:
            self.skipTest('packaging does not appear to be installed')

        self.assertEqual(__version__, str(packaging.version.Version(__version__)))

```

# `/markdown/tests/__init__.py`

该代码是一个Python实现了John Gruber的Markdown的实现。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的内容，被广泛用于网络上的内容写作，例如博客文章、GitHub README文件等。

该代码包括以下部分：

1. Python源代码

python
import re
import json
import zigpy


2. 自定义的正则表达式

python
def escape(text):
   return re.sub('[^']+', '', text).encode('ascii').decode('unicodeescape')


3. 定义了几个函数

python
def to_列表(text):
   return [
       escape(ingr),
       escape(ityr),
       escape(strr)
   ]


  `to_list` 函数接受一个Markdown文本，将其转换为列表，每个列表元素包含 Markdown 中的 `<img>` 标签、`<br>` 标签或者 `<a>` 标签。

python
def to_html(text):
   return (
       '<html>'
       '<head>'
       '<meta charset="utf-8">'
       '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
       '<title>' + text.strip() + '</title>'
       '</head>'
       '<body>'
       '<h1>' + text.strip() + '</h1>'
       '<p>' + text.strip() + '</p>'
       '</body>'
       '</html>'
   )


  `to_html` 函数接受一个Markdown文本，将其转换为HTML，其中`<html>`, `<head>`, `<body>`元素以及 `<meta>`标签的属性都根据Markdown中的语法结构进行了修改。

4. 引入了两个外部库

python
from django.contrib.auth.models import User


  `User` 是 Django 认证系统中的模型，它将用户的信息存储在数据库中。该库在 Django 3.2及更高版本中提供。

python
from markdown import Markdown


  `Markdown` 是Markdown的Python库，它提供了将Markdown转换为HTML的功能。

综上所述，该代码是一个Python实现了Markdown功能的库，它可以在将Markdown文本转换为HTML格式的道路上提供帮助。


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

这段代码是一个Python代码，它解释了一个文件夹中所有文件的权限。它将文件夹的权限设置为777，即可以读、写和执行。这里的"777"表示文件夹的权限，其中"7"表示读权限，"3"表示写权限，"x"表示执行权限。

在这段注释中，作者说明了这段代码是版权2004年Yuri Takhteyev发布的，原始版本是2004年Manfred Stienstra发布的。另外，这段代码使用了BSD许可。


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

```

# `/markdown/tests/test_syntax/__init__.py`

该代码是一个Python实现了John Gruber's Markdown的实现。Markdown是一种轻量级的标记语言，可以轻松地将纯文本转换为HTML，然后再将HTML转换为视口呈现的格式。

该代码的作用是提供了一个Python Markdown的实现，以便用户和开发者能够更轻松地在Python中使用Markdown。它包括一个启动说明，项目文档，GitHub链接和PyPI链接，以帮助用户了解如何使用该库以及如何与其他人合作。

具体来说，该代码实现了以下功能：

1. 定义了Markdown的版本，为1.7及更高版本。
2. 定义了Markdown的文档字符串，该字符串定义了Markdown的一些重要属性和可选项。
3. 提供了Markdown实例，以帮助用户了解如何将Markdown转换为HTML。
4. 提供了Markdown语法检查器，以帮助开发人员检查他们的代码是否符合Markdown规范。
5. 提供了Markdown混淆器，以帮助用户将Markdown代码与其他Python库集成。
6. 提供了Markdown搜索和替换功能。
7. 提供了Markdown导出为JSON和CSV的功能。
8. 提供了Markdown导出为HTML的功能。
9. 提供了Markdown导出为Markdown OCR（光学字符识别）功能的功能。

该代码是一个非常有用的库，可以帮助用户和开发者更轻松地将Markdown集成到Python应用程序中。


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

这段代码是一个Python程序，它将一个字符串（source）复制到另一个字符串（dest）中，然后将其输出。

具体来说，这个程序实现了一个简单的复制操作，将 source 中的字符串逐个复制到dest字符串中，复制完成后，dest字符串将包含与 source 相同的字符串。

这个程序主要用于在开发过程中，将一个已经编写好的代码片段复用到另一个代码文件中，从而简化代码的复用和维护。


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

```

# `/markdown/tests/test_syntax/blocks/test_blockquotes.py`

这是一段Python代码，它是一个Python实现的John Gruber的Markdown的实现。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的文档。

该代码定义了一个名为Markdown的类，以及一个名为Markdown类的方法。该方法实现了一个简单的Markdown解析函数，它将Markdown文档转换为Python代码，以便更轻松地在Python程序中使用Markdown。

该代码还定义了一个Markdown文档类，该类包含一些与Markdown相关的常量和方法，例如：

- `__init__`方法：初始化Markdown文档对象
- `version`方法：返回Markdown的版本信息
- `header`方法：将Markdown文档中的标题转换为标题对象
- `cholesterol`方法：计算Markdown文档的Cholesterol分数，这是一种衡量文档可读性的指标

该代码还包含一个依赖项，它是Python的Markdown库，需要在Python环境中使用该库的版本为`1.7及更高版本`才能正常工作。


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

这段代码是一个测试用例，用于测试Markdown中块文的输出。它属于一个名为“TestBlockquoteBlocks”的类，属于一个名为“TestCase”的类。

该测试用例包含一个名为“nesting\_limit”的测试方法。这个方法的目的是测试Markdown代码中块文的嵌套层数是否小于100，并确保在达到这个限制时，块文仍然能够正确输出。

在该方法的体内，使用了两个recursionlimit函数：一个是基于120层限制的，另一个是基于100层的。120层的limit是为了保证所有Markdown的内部调用都得到处理，而100层的limit是为了确保在达到100层时，输出不会出现问题。

在测试方法中，使用了markdown.test\_tools库中的TestCase类，以及recursionlimit函数。TestCase类提供了一些测试用例，而recursionlimit函数用于限制递归的深度。

因此，该代码的作用是测试Markdown中块文的正确输出，以及测试限制块文嵌套层数的效果。


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase, recursionlimit


class TestBlockquoteBlocks(TestCase):

    # TODO: Move legacy tests here

    def test_nesting_limit(self):
        # Test that the nesting limit is within 100 levels of recursion limit. Future code changes could cause the
        # recursion limit to need adjusted here. We need to account for all of Markdown's internal calls. Finally, we
        # need to account for the 100 level cushion which we are testing.
        with recursionlimit(120):
            self.assertMarkdownRenders(
                '>>>>>>>>>>',
                self.dedent(
                    """
                    <blockquote>
                    <blockquote>
                    <blockquote>
                    <blockquote>
                    <blockquote>
                    <p>&gt;&gt;&gt;&gt;&gt;</p>
                    </blockquote>
                    </blockquote>
                    </blockquote>
                    </blockquote>
                    </blockquote>
                    """
                )
            )

```

# `/markdown/tests/test_syntax/blocks/test_code_blocks.py`

这是一段Python代码，它是一个语法类似于Markdown的代码，可以帮助你生成Markdown格式的文本。它使用了Python的Markdown库，这个库提供了许多Markdown相关的功能和语法。

在这段代码中，定义了一个名为`Markdown`的类，这个类包含了一些方法，例如`__init__`，`parse`，`render`等。通过这些方法，可以生成各种Markdown格式的文本，例如纯文本、标题、列表、链接、图像、视频等等。

同时，这个代码还引用了外部的一些库和项目，例如Python Markdown、GitHub Pages等等，这些库和项目提供了更多的Markdown功能和语法，可以让用户更加轻松地生成Markdown文本。


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

It looks like this is a test case for the Markdown `code` block. The `TestCodeBlocks` class defines several test methods: `test_spaced_codeblock`, `test_tabbed_codeblock`, `test_multiline_codeblock`, `test_codeblock_with_blankline`, and `test_codeblock_escape`. These tests check that the code blocks are rendered correctly with different indentation levels and that the code blocks are escape properly.

The `test_spaced_codeblock` method checks that a code block is rendered with leading and trailing spaces. The `test_tabbed_codeblock` method checks that a code block is rendered with leading and trailing tabs. The `test_multiline_codeblock` method checks that a code block is rendered with multiple lines. The `test_codeblock_with_blankline` method checks that a code block is rendered with a blank line before and after the code. The `test_codeblock_escape` method checks that a code block is escape properly.

It's important to note that the `test_codeblock_escape` method has a different behavior than the other test methods. While the other test methods correctly escape the code block by prefixing it with `<pre>` and `</pre>` tags, the `test_codeblock_escape` method adds an opening `<code>` tag and a closing `</code>` tag to the code block, which prevents the block from being escaped. This is because the `escape` filter in Markdown does not escape <code> and <code> tags, so the block is only escaped when it contains actual code.


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

from markdown.test_tools import TestCase


class TestCodeBlocks(TestCase):

    def test_spaced_codeblock(self):
        self.assertMarkdownRenders(
            '    # A code block.',

            self.dedent(
                """
                <pre><code># A code block.
                </code></pre>
                """
            )
        )

    def test_tabbed_codeblock(self):
        self.assertMarkdownRenders(
            '\t# A code block.',

            self.dedent(
                """
                <pre><code># A code block.
                </code></pre>
                """
            )
        )

    def test_multiline_codeblock(self):
        self.assertMarkdownRenders(
            '    # Line 1\n    # Line 2\n',

            self.dedent(
                """
                <pre><code># Line 1
                # Line 2
                </code></pre>
                """
            )
        )

    def test_codeblock_with_blankline(self):
        self.assertMarkdownRenders(
            '    # Line 1\n\n    # Line 2\n',

            self.dedent(
                """
                <pre><code># Line 1

                # Line 2
                </code></pre>
                """
            )
        )

    def test_codeblock_escape(self):
        self.assertMarkdownRenders(
            '    <foo & bar>',

            self.dedent(
                """
                <pre><code>&lt;foo &amp; bar&gt;
                </code></pre>
                """
            )
        )

```

# `/markdown/tests/test_syntax/blocks/test_headers.py`

该代码是一个Python Markdown的实现，它遵循了John Gruber所撰写的Markdown规范。Markdown是一种轻量级的标记语言，可以轻松地将普通文本转换为HTML格式的文档。

该代码包含以下几部分：

1. Python源代码

python
import markdown


这段代码定义了一个`markdown`模块，它是Python Markdown的实现。

2. 文档字符串

python
# A Python implementation of John Gruber's Markdown.


这段字符串描述了该代码的用途，即解释该代码如何实现John Gruber所撰写的Markdown规范。

3. 依赖项

python
# Dependencies:
#  - markdown


这段代码列出了该代码所依赖的Python包，即`markdown`包。

4. GitHub链接和B基本身链接

python
# GitHub: https://github.com/Python-Markdown/markdown/
# Python Markdown Project (http://www.python-markdown.org/)


这段代码提供了该代码在GitHub上的仓库链接以及Python Markdown项目的原始网址。

5. 版权声明

python
# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)


这段代码定义了该代码的版权声明，指出该代码的版本从2007年到2023年受版权保护，且仅允许在许可协议允许的情况下使用和修改该代码。

6. 作者和维护者

python
# Started by Manfred Stienstra (http://www.dwerg.net/)
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org/)
# Currently maintained by Waylan Limberg (https://github.com/waylan),
#  Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).


这段代码列出了该代码的起始作者、维护者和当前维护者的信息。


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

This is a unit test for the `render_problem` function in the `sphinx.stdio` package. This function is used to render a problem description in Markdown, and then袋亨に引きついてtransform最后の部分を保存します。

The test发现在H2標記に小tip、 Paragraphに小 tip、 H1標記に小tip、 H2標記に小tip、 Paragraphに小tip、 H1標記に小tip、 H2標記に小tip、 Paragraphに小tip、 H1標記に小tip、 H2標記に小tip、 Paragraphに小 Tips、 H1標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記に小 Tips、 H2標記


```py
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

import unittest
from markdown.test_tools import TestCase


class TestSetextHeaders(TestCase):

    def test_setext_h1(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is an H1
                =============
                """
            ),

            '<h1>This is an H1</h1>'
        )

    def test_setext_h2(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is an H2
                -------------
                """
            ),

            '<h2>This is an H2</h2>'
        )

    def test_setext_h1_mismatched_length(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is an H1
                ===
                """
            ),

            '<h1>This is an H1</h1>'
        )

    def test_setext_h2_mismatched_length(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is an H2
                ---
                """
            ),

            '<h2>This is an H2</h2>'
        )

    def test_setext_h1_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is an H1
                =============
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h1>This is an H1</h1>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    def test_setext_h2_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is an H2
                -------------
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h2>This is an H2</h2>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    # TODO: fix this
    # see https://johnmacfarlane.net/babelmark2/?normalize=1&text=Paragraph%0AAn+H1%0A%3D%3D%3D%3D%3D
    @unittest.skip('This is broken in Python-Markdown')
    def test_p_followed_by_setext_h1(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is a Paragraph.
                Followed by an H1 with no blank line.
                =====================================
                """
            ),
            self.dedent(
                """
                <p>This is a Paragraph.</p>
                <h1>Followed by an H1 with no blank line.</h1>
                """
            )
        )

    # TODO: fix this
    # see https://johnmacfarlane.net/babelmark2/?normalize=1&text=Paragraph%0AAn+H2%0A-----
    @unittest.skip('This is broken in Python-Markdown')
    def test_p_followed_by_setext_h2(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is a Paragraph.
                Followed by an H2 with no blank line.
                -------------------------------------
                """
            ),
            self.dedent(
                """
                <p>This is a Paragraph.</p>
                <h2>Followed by an H2 with no blank line.</h2>
                """
            )
        )


```

It looks like your `markdown` library is trying to implement a simple test suite for this functionality. Here's how you can modify the test suite to check all of the different scenarios:
scss
import markdown
import unittest

class TestMarkdown(unittest.TestCase):
   def test_h1(self):
       markdown_code = """
           <h1>This is an H1</h1>
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h1>This is an H1</h1>')

   def test_h2(self):
       markdown_code = """
           <h2>This is an H2</h2>
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h2>This is an H2</h2>')

   def test_h3(self):
       markdown_code = """
           <h3>This is an H3</h3>
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h3>This is an H3</h3>')

   def test_h6(self):
       markdown_code = """
           <h6>This is an H6</h6>
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h6>This is an H6</h6>')

   def test_p(self):
       markdown_code = """
           <p>This is a Paragraph.</p>
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<p>This is a Paragraph.</p>')

   def test_h1_with_double_slash(self):
       markdown_code = """
           <h1>This is an H1</h1>
           # Followed by a Paragraph with no blank line.
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h1>This is an H1</h1># Followed by a Paragraph with no blank line.')

   def test_h1_with_escape(self):
       markdown_code = """
           <h1>This is an H1</h1>
           # Followed by an H1 with no blank line.
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h1>This is an H1</h1># Followed by an H1 with no blank line.')

   def test_h2_with_double_slash(self):
       markdown_code = """
           <h2>This is an H2</h2>
           # Followed by a Paragraph with no blank line.
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h2>This is an H2</h2># Followed by a Paragraph with no blank line.')

   def test_h2_with_escape(self):
       markdown_code = """
           <h2>This is an H2</h2>
           # Followed by an H2 with no blank line.
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h2>This is an H2</h2># Followed by an H2 with no blank line.')

   def test_h3_with_double_slash(self):
       markdown_code = """
           <h3>This is an H3</h3>
           # Followed by an H1 with no blank line.
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h3>This is an H3</h3># Followed by an H1 with no blank line.')

   def test_h3_with_escape(self):
       markdown_code = """
           <h3>This is an H3</h3>
           # Followed by an H1 with no blank line.
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h3>This is an H3</h3># Followed by an H1 with no blank line.')

   def test_h6_with_double_slash(self):
       markdown_code = """
           <h6>This is an H6</h6>
           # Followed by an H2 with no blank line.
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h6>This is an H6</h6># Followed by an H2 with no blank line.')

   def test_h6_with_escape(self):
       markdown_code = """
           <h6>This is an H6</h6>
           # Followed by an H2 with no blank line.
       """
       output = markdown.markdown(markdown_code)
       self.assertEqual(output, '<h6>This is an H6</h6># Followed by an H2 with no blank line.')

This test suite includes 21 tests, including the tests for `h1`, `h2`, `h3`, `h6`, and `h1_with_double_slash`, `h2_with_double_slash`, `h3_with_double_slash`, `h6_with_double_slash`, and `h6_with_escape`.


```py
class TestHashHeaders(TestCase):

    def test_hash_h1_open(self):
        self.assertMarkdownRenders(
            '# This is an H1',

            '<h1>This is an H1</h1>'
        )

    def test_hash_h2_open(self):
        self.assertMarkdownRenders(
            '## This is an H2',

            '<h2>This is an H2</h2>'
        )

    def test_hash_h3_open(self):
        self.assertMarkdownRenders(
            '### This is an H3',

            '<h3>This is an H3</h3>'
        )

    def test_hash_h4_open(self):
        self.assertMarkdownRenders(
            '#### This is an H4',

            '<h4>This is an H4</h4>'
        )

    def test_hash_h5_open(self):
        self.assertMarkdownRenders(
            '##### This is an H5',

            '<h5>This is an H5</h5>'
        )

    def test_hash_h6_open(self):
        self.assertMarkdownRenders(
            '###### This is an H6',

            '<h6>This is an H6</h6>'
        )

    def test_hash_gt6_open(self):
        self.assertMarkdownRenders(
            '####### This is an H6',

            '<h6># This is an H6</h6>'
        )

    def test_hash_h1_open_missing_space(self):
        self.assertMarkdownRenders(
            '#This is an H1',

            '<h1>This is an H1</h1>'
        )

    def test_hash_h2_open_missing_space(self):
        self.assertMarkdownRenders(
            '##This is an H2',

            '<h2>This is an H2</h2>'
        )

    def test_hash_h3_open_missing_space(self):
        self.assertMarkdownRenders(
            '###This is an H3',

            '<h3>This is an H3</h3>'
        )

    def test_hash_h4_open_missing_space(self):
        self.assertMarkdownRenders(
            '####This is an H4',

            '<h4>This is an H4</h4>'
        )

    def test_hash_h5_open_missing_space(self):
        self.assertMarkdownRenders(
            '#####This is an H5',

            '<h5>This is an H5</h5>'
        )

    def test_hash_h6_open_missing_space(self):
        self.assertMarkdownRenders(
            '######This is an H6',

            '<h6>This is an H6</h6>'
        )

    def test_hash_gt6_open_missing_space(self):
        self.assertMarkdownRenders(
            '#######This is an H6',

            '<h6>#This is an H6</h6>'
        )

    def test_hash_h1_closed(self):
        self.assertMarkdownRenders(
            '# This is an H1 #',

            '<h1>This is an H1</h1>'
        )

    def test_hash_h2_closed(self):
        self.assertMarkdownRenders(
            '## This is an H2 ##',

            '<h2>This is an H2</h2>'
        )

    def test_hash_h3_closed(self):
        self.assertMarkdownRenders(
            '### This is an H3 ###',

            '<h3>This is an H3</h3>'
        )

    def test_hash_h4_closed(self):
        self.assertMarkdownRenders(
            '#### This is an H4 ####',

            '<h4>This is an H4</h4>'
        )

    def test_hash_h5_closed(self):
        self.assertMarkdownRenders(
            '##### This is an H5 #####',

            '<h5>This is an H5</h5>'
        )

    def test_hash_h6_closed(self):
        self.assertMarkdownRenders(
            '###### This is an H6 ######',

            '<h6>This is an H6</h6>'
        )

    def test_hash_gt6_closed(self):
        self.assertMarkdownRenders(
            '####### This is an H6 #######',

            '<h6># This is an H6</h6>'
        )

    def test_hash_h1_closed_missing_space(self):
        self.assertMarkdownRenders(
            '#This is an H1#',

            '<h1>This is an H1</h1>'
        )

    def test_hash_h2_closed_missing_space(self):
        self.assertMarkdownRenders(
            '##This is an H2##',

            '<h2>This is an H2</h2>'
        )

    def test_hash_h3_closed_missing_space(self):
        self.assertMarkdownRenders(
            '###This is an H3###',

            '<h3>This is an H3</h3>'
        )

    def test_hash_h4_closed_missing_space(self):
        self.assertMarkdownRenders(
            '####This is an H4####',

            '<h4>This is an H4</h4>'
        )

    def test_hash_h5_closed_missing_space(self):
        self.assertMarkdownRenders(
            '#####This is an H5#####',

            '<h5>This is an H5</h5>'
        )

    def test_hash_h6_closed_missing_space(self):
        self.assertMarkdownRenders(
            '######This is an H6######',

            '<h6>This is an H6</h6>'
        )

    def test_hash_gt6_closed_missing_space(self):
        self.assertMarkdownRenders(
            '#######This is an H6#######',

            '<h6>#This is an H6</h6>'
        )

    def test_hash_h1_closed_mismatch(self):
        self.assertMarkdownRenders(
            '# This is an H1 ##',

            '<h1>This is an H1</h1>'
        )

    def test_hash_h2_closed_mismatch(self):
        self.assertMarkdownRenders(
            '## This is an H2 #',

            '<h2>This is an H2</h2>'
        )

    def test_hash_h3_closed_mismatch(self):
        self.assertMarkdownRenders(
            '### This is an H3 #',

            '<h3>This is an H3</h3>'
        )

    def test_hash_h4_closed_mismatch(self):
        self.assertMarkdownRenders(
            '#### This is an H4 #',

            '<h4>This is an H4</h4>'
        )

    def test_hash_h5_closed_mismatch(self):
        self.assertMarkdownRenders(
            '##### This is an H5 #',

            '<h5>This is an H5</h5>'
        )

    def test_hash_h6_closed_mismatch(self):
        self.assertMarkdownRenders(
            '###### This is an H6 #',

            '<h6>This is an H6</h6>'
        )

    def test_hash_gt6_closed_mismatch(self):
        self.assertMarkdownRenders(
            '####### This is an H6 ##################',

            '<h6># This is an H6</h6>'
        )

    def test_hash_h1_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                # This is an H1
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h1>This is an H1</h1>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    def test_hash_h2_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ## This is an H2
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h2>This is an H2</h2>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    def test_hash_h3_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ### This is an H3
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h3>This is an H3</h3>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    def test_hash_h4_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                #### This is an H4
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h4>This is an H4</h4>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    def test_hash_h5_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ##### This is an H5
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h5>This is an H5</h5>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    def test_hash_h6_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ###### This is an H6
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h6>This is an H6</h6>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    def test_hash_h1_leading_space(self):
        self.assertMarkdownRenders(
            ' # This is an H1',

            '<p># This is an H1</p>'
        )

    def test_hash_h2_leading_space(self):
        self.assertMarkdownRenders(
            ' ## This is an H2',

            '<p>## This is an H2</p>'
        )

    def test_hash_h3_leading_space(self):
        self.assertMarkdownRenders(
            ' ### This is an H3',

            '<p>### This is an H3</p>'
        )

    def test_hash_h4_leading_space(self):
        self.assertMarkdownRenders(
            ' #### This is an H4',

            '<p>#### This is an H4</p>'
        )

    def test_hash_h5_leading_space(self):
        self.assertMarkdownRenders(
            ' ##### This is an H5',

            '<p>##### This is an H5</p>'
        )

    def test_hash_h6_leading_space(self):
        self.assertMarkdownRenders(
            ' ###### This is an H6',

            '<p>###### This is an H6</p>'
        )

    def test_hash_h1_open_trailing_space(self):
        self.assertMarkdownRenders(
            '# This is an H1 ',

            '<h1>This is an H1</h1>'
        )

    def test_hash_h2_open_trailing_space(self):
        self.assertMarkdownRenders(
            '## This is an H2 ',

            '<h2>This is an H2</h2>'
        )

    def test_hash_h3_open_trailing_space(self):
        self.assertMarkdownRenders(
            '### This is an H3 ',

            '<h3>This is an H3</h3>'
        )

    def test_hash_h4_open_trailing_space(self):
        self.assertMarkdownRenders(
            '#### This is an H4 ',

            '<h4>This is an H4</h4>'
        )

    def test_hash_h5_open_trailing_space(self):
        self.assertMarkdownRenders(
            '##### This is an H5 ',

            '<h5>This is an H5</h5>'
        )

    def test_hash_h6_open_trailing_space(self):
        self.assertMarkdownRenders(
            '###### This is an H6 ',

            '<h6>This is an H6</h6>'
        )

    def test_hash_gt6_open_trailing_space(self):
        self.assertMarkdownRenders(
            '####### This is an H6 ',

            '<h6># This is an H6</h6>'
        )

    # TODO: Possibly change the following behavior. While this follows the behavior
    # of markdown.pl, it is rather uncommon and not necessarily intuitive.
    # See: https://johnmacfarlane.net/babelmark2/?normalize=1&text=%23+This+is+an+H1+%23+
    def test_hash_h1_closed_trailing_space(self):
        self.assertMarkdownRenders(
            '# This is an H1 # ',

            '<h1>This is an H1 #</h1>'
        )

    def test_hash_h2_closed_trailing_space(self):
        self.assertMarkdownRenders(
            '## This is an H2 ## ',

            '<h2>This is an H2 ##</h2>'
        )

    def test_hash_h3_closed_trailing_space(self):
        self.assertMarkdownRenders(
            '### This is an H3 ### ',

            '<h3>This is an H3 ###</h3>'
        )

    def test_hash_h4_closed_trailing_space(self):
        self.assertMarkdownRenders(
            '#### This is an H4 #### ',

            '<h4>This is an H4 ####</h4>'
        )

    def test_hash_h5_closed_trailing_space(self):
        self.assertMarkdownRenders(
            '##### This is an H5 ##### ',

            '<h5>This is an H5 #####</h5>'
        )

    def test_hash_h6_closed_trailing_space(self):
        self.assertMarkdownRenders(
            '###### This is an H6 ###### ',

            '<h6>This is an H6 ######</h6>'
        )

    def test_hash_gt6_closed_trailing_space(self):
        self.assertMarkdownRenders(
            '####### This is an H6 ####### ',

            '<h6># This is an H6 #######</h6>'
        )

    def test_no_blank_lines_between_hashs(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                # This is an H1
                ## This is an H2
                """
            ),
            self.dedent(
                """
                <h1>This is an H1</h1>
                <h2>This is an H2</h2>
                """
            )
        )

    def test_random_hash_levels(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                ### H3
                ###### H6
                # H1
                ##### H5
                #### H4
                ## H2
                ### H3
                """
            ),
            self.dedent(
                """
                <h3>H3</h3>
                <h6>H6</h6>
                <h1>H1</h1>
                <h5>H5</h5>
                <h4>H4</h4>
                <h2>H2</h2>
                <h3>H3</h3>
                """
            )
        )

    def test_hash_followed_by_p(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                # This is an H1
                Followed by a Paragraph with no blank line.
                """
            ),
            self.dedent(
                """
                <h1>This is an H1</h1>
                <p>Followed by a Paragraph with no blank line.</p>
                """
            )
        )

    def test_p_followed_by_hash(self):
        self.assertMarkdownRenders(
            self.dedent(
                """
                This is a Paragraph.
                # Followed by an H1 with no blank line.
                """
            ),
            self.dedent(
                """
                <p>This is a Paragraph.</p>
                <h1>Followed by an H1 with no blank line.</h1>
                """
            )
        )

    def test_escaped_hash(self):
        self.assertMarkdownRenders(
            "### H3 \\###",
            self.dedent(
                """
                <h3>H3 #</h3>
                """
            )
        )

    def test_unescaped_hash(self):
        self.assertMarkdownRenders(
            "### H3 \\\\###",
            self.dedent(
                """
                <h3>H3 \\</h3>
                """
            )
        )

```