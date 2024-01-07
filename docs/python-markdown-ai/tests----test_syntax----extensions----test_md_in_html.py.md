# `markdown\tests\test_syntax\extensions\test_md_in_html.py`

```

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

Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

# 导入所需的模块和类
from unittest import TestSuite
from markdown.test_tools import TestCase
from ..blocks.test_html_blocks import TestHTMLBlocks
from markdown import Markdown
from xml.etree.ElementTree import Element

# 定义一个测试类，确保 HTML 存储中的剩余元素被正确序列化
class TestMarkdownInHTMLPostProcessor(TestCase):
    def test_stash_to_string(self):
        # 在这里应该没有已知的情况会发生这种情况，所以我们需要强制传递一个 `etree` `Element` 给方法以确保正确的行为。
        element = Element('div')
        element.text = 'Foo bar.'
        md = Markdown(extensions=['md_in_html'])
        result = md.postprocessors['raw_html'].stash_to_string(element)
        self.assertEqual(result, '<div>Foo bar.</div>')

# 定义一个测试类，确保 md_in_html 扩展不会破坏默认行为
class TestDefaultwMdInHTML(TestHTMLBlocks):
    default_kwargs = {'extensions': ['md_in_html']}

# 加载测试用例
def load_tests(loader, tests, pattern):
    # 确保 `TestHTMLBlocks` 不会被重复运行，因此在这里排除它
    suite = TestSuite()
    for test_class in [TestDefaultwMdInHTML, TestMdInHTML, TestMarkdownInHTMLPostProcessor]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite

```