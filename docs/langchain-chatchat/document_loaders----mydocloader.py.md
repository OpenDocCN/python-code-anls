# `.\Langchain-Chatchat\document_loaders\mydocloader.py`

```py
# 从 langchain.document_loaders.unstructured 模块中导入 UnstructuredFileLoader 类
from langchain.document_loaders.unstructured import UnstructuredFileLoader
# 从 typing 模块中导入 List 类型
from typing import List
# 从 tqdm 模块中导入 tqdm 函数
import tqdm

# 定义 RapidOCRDocLoader 类，继承自 UnstructuredFileLoader 类
class RapidOCRDocLoader(UnstructuredFileLoader):
    # 如果当前脚本作为主程序执行
    if __name__ == '__main__':
        # 创建 RapidOCRDocLoader 对象，指定文件路径为 "../tests/samples/ocr_test.docx"
        loader = RapidOCRDocLoader(file_path="../tests/samples/ocr_test.docx")
        # 调用 load 方法加载文档
        docs = loader.load()
        # 打印加载的文档
        print(docs)
```