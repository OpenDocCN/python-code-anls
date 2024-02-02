# `MetaGPT\tests\metagpt\utils\test_code_parser.py`

```py

#!/usr/bin/env python
# coding: utf-8
"""
@Time    : 2023/7/10 17:14
@Author  : chengmaoyu
@File    : test_code_parser.py
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.utils.common 模块中导入 CodeParser 类
from metagpt.utils.common import CodeParser

# 定义测试用例的文本内容
t_text = '''
## Required Python third-party packages
...
        '''

# 定义测试类 TestCodeParser
class TestCodeParser:
    # 定义 pytest fixture parser
    @pytest.fixture
    def parser(self):
        return CodeParser()

    # 定义 pytest fixture text
    @pytest.fixture
    def text(self):
        return t_text

    # 测试解析文本块的方法
    def test_parse_blocks(self, parser, text):
        result = parser.parse_blocks(text)
        print(result)
        assert "game.py" in result["Task list"]

    # 测试解析特定块的方法
    def test_parse_block(self, parser, text):
        result = parser.parse_block("Task list", text)
        print(result)
        assert "game.py" in result

    # 测试解析特定代码块的方法
    def test_parse_code(self, parser, text):
        result = parser.parse_code("Task list", text, "python")
        print(result)
        assert "game.py" in result

    # 测试解析特定字符串的方法
    def test_parse_str(self, parser, text):
        result = parser.parse_str("Anything UNCLEAR", text, "python")
        print(result)
        assert "We need clarification on how the high score " in result

    # 测试解析文件列表的方法
    def test_parse_file_list(self, parser, text):
        result = parser.parse_file_list("Task list", text)
        print(result)
        assert "game.py" in result

# 如果是主程序，则执行测试用例
if __name__ == "__main__":
    t = TestCodeParser()
    t.test_parse_file_list(CodeParser(), t_text)
    # TestCodeParser.test_parse_file_list()

```