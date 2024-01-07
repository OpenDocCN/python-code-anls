# `markdown\tests\test_syntax\extensions\test_tables.py`

```
"""
# 导入模块
from markdown.test_tools import TestCase
from markdown.extensions.tables import TableExtension

# 定义测试类
class TestTableBlocks(TestCase):

    # 定义测试方法
    def test_empty_cells(self):
        """Empty cells (`nbsp`)."""
        # 测试用例描述

        # 测试用例输入
        text = """
   | Second Header
------------- | -------------
   | Content Cell
Content Cell  |  
        """

```