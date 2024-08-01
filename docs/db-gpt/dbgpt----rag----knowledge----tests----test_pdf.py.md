# `.\DB-GPT-src\dbgpt\rag\knowledge\tests\test_pdf.py`

```py
from unittest.mock import MagicMock, mock_open, patch  # 导入所需的模块和函数

import pytest  # 导入 pytest 测试框架

from dbgpt.rag.knowledge.pdf import PDFKnowledge  # 导入 PDFKnowledge 类

MOCK_PDF_PAGES = [  # 定义一个模拟的 PDF 页面内容列表
    ("This is the content of the first page.", 0),
    ("This is the content of the second page.", 1),
]


@pytest.fixture  # 定义一个 pytest 的 fixture，用于创建模拟 PDF 文件和阅读器
def mock_pdf_open_and_reader():
    mock_pdf_file = mock_open()  # 创建一个模拟的打开 PDF 文件的函数
    mock_reader = MagicMock()  # 创建一个模拟的 PDF 阅读器对象
    mock_reader.pages = [  # 为模拟的阅读器设置页面列表，每个页面使用 MagicMock 模拟
        MagicMock(extract_text=MagicMock(return_value=page[0]))
        for page in MOCK_PDF_PAGES
    ]
    with patch("builtins.open", mock_pdf_file):  # 使用 patch 临时替换内置的 open 函数为模拟的打开函数
        with patch("pypdf.PdfReader", return_value=mock_reader) as mock:  # 使用 patch 替换 pypdf.PdfReader 为模拟的阅读器
            yield mock  # 返回模拟对象


def test_load_from_pdf(mock_pdf_open_and_reader):  # 定义测试函数，使用 mock_pdf_open_and_reader fixture
    file_path = "test_document.pdf"  # 设置测试 PDF 文件路径
    knowledge = PDFKnowledge(file_path=file_path)  # 创建 PDFKnowledge 实例
    documents = knowledge._load()  # 调用 _load 方法加载文档内容

    assert len(documents) == len(MOCK_PDF_PAGES)  # 断言加载的文档数量与模拟页面数量相同
    for i, document in enumerate(documents):  # 遍历加载的文档列表
        assert MOCK_PDF_PAGES[i][0] in document.content  # 断言每个模拟页面的内容在对应文档的内容中
        assert document.metadata["source"] == file_path  # 断言文档的来源元数据为测试 PDF 文件路径
        assert document.metadata["page"] == MOCK_PDF_PAGES[i][1]  # 断言文档的页码元数据与模拟页面的页码相同

    #
```