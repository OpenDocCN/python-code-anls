# `.\Langchain-Chatchat\document_loaders\mypdfloader.py`

```
# 从 typing 模块中导入 List 类型
from typing import List
# 从 langchain.document_loaders.unstructured 模块中导入 UnstructuredFileLoader 类
from langchain.document_loaders.unstructured import UnstructuredFileLoader
# 导入 cv2 模块
import cv2
# 从 PIL 模块中导入 Image 类
from PIL import Image
# 导入 numpy 模块，并使用 np 别名
import numpy as np
# 从 configs 模块中导入 PDF_OCR_THRESHOLD 常量
from configs import PDF_OCR_THRESHOLD
# 从 document_loaders.ocr 模块中导入 get_ocr 函数
from document_loaders.ocr import get_ocr
# 导入 tqdm 模块
import tqdm

# 定义 RapidOCRPDFLoader 类，继承自 UnstructuredFileLoader 类
class RapidOCRPDFLoader(UnstructuredFileLoader):
    # 如果当前脚本作为主程序运行
    if __name__ == "__main__":
        # 创建 RapidOCRPDFLoader 对象，指定文件路径为 "/Users/tonysong/Desktop/test.pdf"
        loader = RapidOCRPDFLoader(file_path="/Users/tonysong/Desktop/test.pdf")
        # 调用 load 方法加载文档
        docs = loader.load()
        # 打印加载的文档
        print(docs)
```