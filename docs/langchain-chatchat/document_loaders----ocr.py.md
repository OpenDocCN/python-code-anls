# `.\Langchain-Chatchat\document_loaders\ocr.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 如果是类型检查模式
if TYPE_CHECKING:
    # 尝试导入基于 PaddlePaddle 的 OCR 模块
    try:
        from rapidocr_paddle import RapidOCR
    # 如果导入失败，则导入基于 ONNX Runtime 的 OCR 模块
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

# 定义一个函数，返回 OCR 对象
def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    # 尝试导入基于 PaddlePaddle 的 OCR 模块
    try:
        from rapidocr_paddle import RapidOCR
        # 创建基于 PaddlePaddle 的 OCR 对象
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    # 如果导入失败，则导入基于 ONNX Runtime 的 OCR 模块
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        # 创建基于 ONNX Runtime 的 OCR 对象
        ocr = RapidOCR()
    # 返回 OCR 对象
    return ocr
```