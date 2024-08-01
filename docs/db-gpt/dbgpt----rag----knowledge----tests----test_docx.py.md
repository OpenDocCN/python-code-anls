# `.\DB-GPT-src\dbgpt\rag\knowledge\tests\test_docx.py`

```py
# 导入必要的模块和函数
from unittest.mock import MagicMock, patch
import pytest
# 导入需要测试的 DocxKnowledge 类
from dbgpt.rag.knowledge.docx import DocxKnowledge

# 定义一个 pytest fixture，用于创建一个模拟的 Docx 文档对象
@pytest.fixture
def mock_docx_document():
    # 创建一个 MagicMock 对象作为文档的模拟
    mock_document = MagicMock()
    # 设置模拟文档的段落内容
    mock_document.paragraphs = [
        MagicMock(text="This is the first paragraph."),
        MagicMock(text="This is the second paragraph."),
    ]
    # 使用 patch 装饰器替换 docx.Document，使其返回模拟文档对象
    with patch("docx.Document", return_value=mock_document):
        # 使用 yield 返回模拟文档对象，供测试函数使用
        yield mock_document

# 定义测试函数，测试从 Docx 文件加载文档内容的功能
def test_load_from_docx(mock_docx_document):
    # 设置测试文件的路径
    file_path = "test_document.docx"
    # 创建 DocxKnowledge 对象，加载指定路径的文档
    knowledge = DocxKnowledge(file_path=file_path)
    # 调用 _load 方法加载文档内容
    documents = knowledge._load()

    # 断言文档列表的长度为1
    assert len(documents) == 1
    # 断言加载的文档内容正确
    assert (
        documents[0].content
        == "This is the first paragraph.\nThis is the second paragraph."
    )
    # 断言加载的文档元数据中的来源路径正确
    assert documents[0].metadata["source"] == file_path
```