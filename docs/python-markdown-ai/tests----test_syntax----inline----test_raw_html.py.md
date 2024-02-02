# `markdown\tests\test_syntax\inline\test_raw_html.py`

```py
"""
这段代码是关于 Python Markdown 的文档和测试用例。

从 markdown.test_tools 模块导入 TestCase 类
"""
from markdown.test_tools import TestCase

class TestRawHtml(TestCase):
    # 测试内联 HTML 标签中的角括号
    def test_inline_html_angle_brackets(self):
        # 断言内联 HTML 标签中的角括号被正确转义
        self.assertMarkdownRenders("<span>e<c</span>", "<p><span>e&lt;c</span></p>")
        self.assertMarkdownRenders("<span>e>c</span>", "<p><span>e&gt;c</span></p>")
        self.assertMarkdownRenders("<span>e < c</span>", "<p><span>e &lt; c</span></p>")
        self.assertMarkdownRenders("<span>e > c</span>", "<p><span>e &gt; c</span></p>")

    # 测试内联 HTML 标签中的反斜杠
    def test_inline_html_backslashes(self):
        # 断言内联 HTML 标签中的反斜杠被正确处理
        self.assertMarkdownRenders('<img src="..\\..\\foo.png">', '<p><img src="..\\..\\foo.png"></p>')
"""
```