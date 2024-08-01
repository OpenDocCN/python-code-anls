# `.\DB-GPT-src\dbgpt\rag\knowledge\tests\test_markdown.py`

```py
# 导入所需模块和函数
from unittest.mock import mock_open, patch
import pytest
# 导入要测试的类或函数
from dbgpt.rag.knowledge.markdown import MarkdownKnowledge

# 定义模拟的 Markdown 数据
MOCK_MARKDOWN_DATA = """# Header 1
This is some text under header 1.

## Header 2
This is some text under header 2.
"""

# 定义测试装置（fixture），用于模拟文件打开操作
@pytest.fixture
def mock_file_open():
    # 使用 patch 和 mock_open 创建模拟的文件打开操作
    with patch("builtins.open", mock_open(read_data=MOCK_MARKDOWN_DATA)) as mock_file:
        yield mock_file

# 定义测试函数
def test_load_from_markdown(mock_file_open):
    # 准备测试所需的文件路径
    file_path = "test_document.md"
    # 创建 MarkdownKnowledge 对象实例
    knowledge = MarkdownKnowledge(file_path=file_path)
    # 调用 _load 方法加载文档内容
    documents = knowledge._load()

    # 断言文档列表中应有一份文档
    assert len(documents) == 1
    # 断言文档内容与预期的 MOCK_MARKDOWN_DATA 相符
    assert documents[0].content == MOCK_MARKDOWN_DATA
    # 断言文档元数据中的 source 字段应与文件路径一致
    assert documents[0].metadata["source"] == file_path
```