# `markdown\tests\test_syntax\inline\test_entities.py`

```py

# 导入markdown.test_tools模块中的TestCase类
from markdown.test_tools import TestCase

# 定义TestEntities类，用于测试实体
class TestEntities(TestCase):

    # 测试命名实体
    def test_named_entities(self):
        self.assertMarkdownRenders("&amp;", "<p>&amp;</p>")
        self.assertMarkdownRenders("&sup2;", "<p>&sup2;</p>")
        self.assertMarkdownRenders("&Aacute;", "<p>&Aacute;</p>")

    # 测试十进制实体
    def test_decimal_entities(self):
        self.assertMarkdownRenders("&#38;", "<p>&#38;</p>")
        self.assertMarkdownRenders("&#178;", "<p>&#178;</p>")

    # 测试十六进制实体
    def test_hexadecimal_entities(self):
        self.assertMarkdownRenders("&#x00026;", "<p>&#x00026;</p>")
        self.assertMarkdownRenders("&#xB2;", "<p>&#xB2;</p>")

    # 测试错误的实体
    def test_false_entities(self):
        self.assertMarkdownRenders("&not an entity;", "<p>&amp;not an entity;</p>")
        self.assertMarkdownRenders("&#B2;", "<p>&amp;#B2;</p>")
        self.assertMarkdownRenders("&#xnothex;", "<p>&amp;#xnothex;</p>")

```