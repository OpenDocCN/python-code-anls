# `.\DB-GPT-src\dbgpt\rag\knowledge\tests\test_txt.py`

```py
# 导入所需的模块和函数
from unittest.mock import mock_open, patch
import pytest
from dbgpt.rag.knowledge.txt import TXTKnowledge

# 定义一个模拟的文本内容作为测试用例
MOCK_TXT_CONTENT = b"Sample text content for testing.\nAnother line of text."

# 定义一个模拟的字符检测结果作为测试用例
MOCK_CHARDET_RESULT = {"encoding": "utf-8", "confidence": 0.99}

# 定义一个 pytest 的 fixture，用于模拟文件打开操作
@pytest.fixture
def mock_file_open():
    # 使用 patch 替换内建函数 open，模拟打开文件并返回模拟文件对象
    with patch("builtins.open", mock_open(read_data=MOCK_TXT_CONTENT), create=True) as mock_file:
        yield mock_file  # 将模拟文件对象作为 fixture 的返回值

# 定义一个 pytest 的 fixture，用于模拟字符编码检测操作
@pytest.fixture
def mock_chardet_detect():
    # 使用 patch 替换 chardet.detect 函数，模拟字符编码检测并返回模拟检测结果
    with patch("chardet.detect", return_value=MOCK_CHARDET_RESULT) as mock_detect:
        yield mock_detect  # 将模拟检测函数作为 fixture 的返回值

# 定义测试函数，测试从文本文件加载知识
def test_load_from_txt(mock_file_open, mock_chardet_detect):
    file_path = "test_document.txt"
    knowledge = TXTKnowledge(file_path=file_path)  # 创建 TXTKnowledge 对象，传入文件路径
    documents = knowledge._load()  # 调用 _load 方法加载文档内容

    # 断言文档数量为 1
    assert len(documents) == 1
    # 断言文档内容包含预期的测试文本
    assert "Sample text content for testing." in documents[0].content
    # 断言文档的元数据中包含正确的源文件路径
    assert documents[0].metadata["source"] == file_path

    # 验证 mock_open 是否正确调用，检查是否以二进制模式打开了指定文件
    mock_file_open.assert_called_once_with(file_path, "rb")

    # 验证 mock_chardet_detect 是否正确调用，检查字符编码检测是否执行了一次
    mock_chardet_detect.assert_called_once()
```