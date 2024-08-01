# `.\DB-GPT-src\dbgpt\rag\knowledge\tests\test_html.py`

```py
# 导入所需模块和函数：mock_open、patch
from unittest.mock import mock_open, patch

# 导入 pytest 模块，用于单元测试
import pytest

# 导入需要测试的类 HTMLKnowledge
from dbgpt.rag.knowledge.html import HTMLKnowledge

# 模拟的 HTML 内容，作为被打开文件的内容
MOCK_HTML_CONTENT = b"""
<html>
<head>
<title>Test HTML</title>
</head>
<body>
<p>This is a paragraph.</p>
</body>
</html>
"""

# 模拟 chardet.detect 函数返回的结果
MOCK_CHARDET_RESULT = {"encoding": "utf-8", "confidence": 0.99}


# 定义装置函数，模拟内置函数 open
@pytest.fixture
def mock_file_open():
    with patch(
        "builtins.open", mock_open(read_data=MOCK_HTML_CONTENT), create=True
    ) as mock_file:
        yield mock_file


# 定义装置函数，模拟 chardet.detect 函数
@pytest.fixture
def mock_chardet_detect():
    with patch("chardet.detect", return_value=MOCK_CHARDET_RESULT) as mock_detect:
        yield mock_detect


# 定义测试函数，验证从 HTML 文件加载内容的功能
def test_load_from_html(mock_file_open, mock_chardet_detect):
    # 定义测试文件路径
    file_path = "test_document.html"
    # 创建 HTMLKnowledge 实例
    knowledge = HTMLKnowledge(file_path=file_path)
    # 调用 _load 方法加载文档内容
    documents = knowledge._load()

    # 断言：加载的文档数量为 1
    assert len(documents) == 1
    # 断言：文档内容包含指定段落文本
    assert "This is a paragraph." in documents[0].content
    # 断言：文档元数据中的来源与文件路径一致
    assert documents[0].metadata["source"] == file_path

    # 断言：mock_open 被调用了一次，用于打开指定路径的文件
    mock_file_open.assert_called_once_with(file_path, "rb")

    # 断言：mock_chardet_detect 被调用了一次，用于检测文件编码
    mock_chardet_detect.assert_called_once()
```