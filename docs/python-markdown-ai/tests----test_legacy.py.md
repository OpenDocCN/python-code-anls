# `markdown\tests\test_legacy.py`

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
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

# 导入所需的模块和库
from markdown.test_tools import LegacyTestCase, Kwargs
import os
import warnings

# 设置警告过滤器，使得警告变成错误
warnings.simplefilter('error')
# 除了特定的警告类型，其他警告都会变成错误
warnings.filterwarnings('default', category=PendingDeprecationWarning)
warnings.filterwarnings('default', category=DeprecationWarning, module='markdown')

# 获取当前文件所在目录的绝对路径
parent_test_dir = os.path.abspath(os.path.dirname(__file__))


# 定义测试类 TestBasic
class TestBasic(LegacyTestCase):
    location = os.path.join(parent_test_dir, 'basic')


# 定义测试类 TestMisc
class TestMisc(LegacyTestCase):
    location = os.path.join(parent_test_dir, 'misc')


# 定义测试类 TestPhp
class TestPhp(LegacyTestCase):
    # 定义一些被排除的测试用例
    """
    Notes on "excluded" tests:
    ...
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


# 定义测试类 TestPl2004
class TestPl2004(LegacyTestCase):
    location = os.path.join(parent_test_dir, 'pl/Tests_2004')
    normalize = True
    input_ext = '.text'
    exclude = ['Yuri_Footnotes', 'Yuri_Attributes']


# 定义测试类 TestPl2007
class TestPl2007(LegacyTestCase):
    # 定义一些被排除的测试用例
    """
    Notes on "excluded" tests:
    ...
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


# 定义测试类 TestExtensions
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


# 定义测试类 TestExtensionsExtra
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