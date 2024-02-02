# `markdown\tests\test_syntax\extensions\test_smarty.py`

```py

# 设置文件编码为 utf-8
# Python Markdown
# 一个实现了 John Gruber 的 Markdown 的 Python 实现。
# 文档：https://python-markdown.github.io/
# GitHub：https://github.com/Python-Markdown/markdown/
# PyPI：https://pypi.org/project/Markdown/
# 由 Manfred Stienstra (http://www.dwerg.net/) 开始。
# 由 Yuri Takhteyev (http://www.freewisdom.org/) 维护了几年。
# 目前由 Waylan Limberg (https://github.com/waylan)、Dmitry Shachnev (https://github.com/mitya57) 和 Isaac Muse (https://github.com/facelessuser) 维护。
# 版权 2007-2022 Python Markdown 项目（v. 1.7 及更高版本）
# 版权 2004, 2005, 2006 Yuri Takhteyev（v. 0.2-1.6b）
# 版权 2004 Manfred Stienstra（原始版本）
# 许可证：BSD（详细信息请参阅 LICENSE.md）。

# 从 markdown.test_tools 导入 TestCase 类
from markdown.test_tools import TestCase

# 测试类 TestSmartyAngledQuotes
class TestSmartyAngledQuotes(TestCase):

    # 默认参数
    default_kwargs = {
        'extensions': ['smarty'],
        'extension_configs': {
            'smarty': {
                'smart_angled_quotes': True,
            },
        },
    }

    # 测试尖角引号
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

# 测试类 TestSmartyCustomSubstitutions
class TestSmartyCustomSubstitutions(TestCase):

    # 默认参数
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

    # 测试自定义替换
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