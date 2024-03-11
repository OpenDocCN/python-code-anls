# `.\Langchain-Chatchat\tests\document_loader\test_pdfloader.py`

```py
# 导入 sys 模块
import sys
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 获取当前文件的父目录的父目录的父目录作为根路径
root_path = Path(__file__).parent.parent.parent
# 将根路径转换为字符串并添加到 sys.path 中，以便在其中搜索模块
sys.path.append(str(root_path))
# 从 pprint 模块中导入 pprint 函数
from pprint import pprint

# 定义一个包含测试文件路径的字典
test_files = {
    "ocr_test.pdf": str(root_path / "tests" / "samples" / "ocr_test.pdf"),
}

# 定义一个测试函数 test_rapidocrpdfloader
def test_rapidocrpdfloader():
    # 获取测试文件的路径
    pdf_path = test_files["ocr_test.pdf"]
    # 从 document_loaders 模块中导入 RapidOCRPDFLoader 类
    from document_loaders import RapidOCRPDFLoader

    # 创建 RapidOCRPDFLoader 对象，传入 PDF 文件路径
    loader = RapidOCRPDFLoader(pdf_path)
    # 调用 load 方法加载文档
    docs = loader.load()
    # 使用 pprint 函数打印加载的文档
    pprint(docs)
    # 断言加载的文档是列表且长度大于0，并且第一个文档的页面内容是字符串类型
    assert isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0].page_content, str)
```