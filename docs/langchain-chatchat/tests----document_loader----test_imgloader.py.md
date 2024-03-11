# `.\Langchain-Chatchat\tests\document_loader\test_imgloader.py`

```py
# 导入 sys 模块，用于访问 Python 解释器的变量和函数
import sys
# 从 pathlib 模块中导入 Path 类，用于处理文件路径
from pathlib import Path

# 获取当前文件的父目录的父目录的父目录，即项目根目录
root_path = Path(__file__).parent.parent.parent
# 将项目根目录路径转换为字符串并添加到 sys.path 中，以便在运行时能够访问该路径下的模块
sys.path.append(str(root_path))
# 从 pprint 模块中导入 pprint 函数，用于美观打印数据结构
from pprint import pprint

# 定义一个包含测试文件路径的字典
test_files = {
    "ocr_test.jpg": str(root_path / "tests" / "samples" / "ocr_test.jpg"),
}

# 定义一个测试函数 test_rapidocrloader
def test_rapidocrloader():
    # 获取 OCR 测试图片的路径
    img_path = test_files["ocr_test.jpg"]
    # 从 document_loaders 模块中导入 RapidOCRLoader 类
    from document_loaders import RapidOCRLoader

    # 创建 RapidOCRLoader 对象，传入图片路径
    loader = RapidOCRLoader(img_path)
    # 调用 load 方法加载文档
    docs = loader.load()
    # 使用 pprint 函数美观打印加载的文档
    pprint(docs)
    # 断言加载的文档是列表类型且长度大于0，并且第一个文档的页面内容是字符串类型
    assert isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0].page_content, str)
```