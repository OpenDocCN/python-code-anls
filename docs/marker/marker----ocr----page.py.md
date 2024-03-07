# `.\marker\marker\ocr\page.py`

```
import io  # 导入io模块
from typing import List, Optional  # 导入类型提示相关模块

import fitz as pymupdf  # 导入fitz模块并重命名为pymupdf
import ocrmypdf  # 导入ocrmypdf模块
from spellchecker import SpellChecker  # 从spellchecker模块导入SpellChecker类

from marker.ocr.utils import detect_bad_ocr  # 从marker.ocr.utils模块导入detect_bad_ocr函数
from marker.schema import Block  # 从marker.schema模块导入Block类
from marker.settings import settings  # 从marker.settings模块导入settings变量

ocrmypdf.configure_logging(verbosity=ocrmypdf.Verbosity.quiet)  # 配置ocrmypdf的日志记录级别为quiet

# 对整个页面进行OCR识别，返回Block对象列表
def ocr_entire_page(page, lang: str, spellchecker: Optional[SpellChecker] = None) -> List[Block]:
    # 如果OCR_ENGINE设置为"tesseract"，则调用ocr_entire_page_tess函数
    if settings.OCR_ENGINE == "tesseract":
        return ocr_entire_page_tess(page, lang, spellchecker)
    # 如果OCR_ENGINE设置为"ocrmypdf"，则调用ocr_entire_page_ocrmp函数
    elif settings.OCR_ENGINE == "ocrmypdf":
        return ocr_entire_page_ocrmp(page, lang, spellchecker)
    else:
        raise ValueError(f"Unknown OCR engine {settings.OCR_ENGINE}")  # 抛出数值错误异常，显示未知的OCR引擎

# 使用tesseract对整个页面进行OCR识别，返回Block对象列表
def ocr_entire_page_tess(page, lang: str, spellchecker: Optional[SpellChecker] = None) -> List[Block]:
    try:
        # 获取页面的完整OCR文本页
        full_tp = page.get_textpage_ocr(flags=settings.TEXT_FLAGS, dpi=settings.OCR_DPI, full=True, language=lang)
        # 获取页面的文本块列表
        blocks = page.get_text("dict", sort=True, flags=settings.TEXT_FLAGS, textpage=full_tp)["blocks"]
        # 获取页面的完整文本
        full_text = page.get_text("text", sort=True, flags=settings.TEXT_FLAGS, textpage=full_tp)

        # 如果完整文本长度为0，则返回空列表
        if len(full_text) == 0:
            return []

        # 检查OCR是否成功。如果失败，返回空列表
        # 例如，如果有一张扫描的空白页上有一些淡淡的文本印记，OCR可能会失败
        if detect_bad_ocr(full_text, spellchecker):
            return []
    except RuntimeError:
        return []
    return blocks  # 返回文本块列表

# 使用ocrmypdf对整个页面进行OCR识别，返回Block对象列表
def ocr_entire_page_ocrmp(page, lang: str, spellchecker: Optional[SpellChecker] = None) -> List[Block]:
    # 使用ocrmypdf获取整个页面的OCR文本
    src = page.parent  # 页面所属文档
    blank_doc = pymupdf.open()  # 创建临时的1页文档
    blank_doc.insert_pdf(src, from_page=page.number, to_page=page.number, annots=False, links=False)  # 插入PDF页面
    pdfbytes = blank_doc.tobytes()  # 获取文档字节流
    inbytes = io.BytesIO(pdfbytes)  # 转换为BytesIO对象
    # 创建一个字节流对象，用于存储 ocrmypdf 处理后的结果 PDF
    outbytes = io.BytesIO()  # let ocrmypdf store its result pdf here
    # 使用 ocrmypdf 进行 OCR 处理
    ocrmypdf.ocr(
        inbytes,
        outbytes,
        language=lang,
        output_type="pdf",
        redo_ocr=None if settings.OCR_ALL_PAGES else True,
        force_ocr=True if settings.OCR_ALL_PAGES else None,
        progress_bar=False,
        optimize=False,
        fast_web_view=1e6,
        skip_big=15, # skip images larger than 15 megapixels
        tesseract_timeout=settings.TESSERACT_TIMEOUT,
        tesseract_non_ocr_timeout=settings.TESSERACT_TIMEOUT,
    )
    # 以 fitz PDF 格式打开 OCR 处理后的输出
    ocr_pdf = pymupdf.open("pdf", outbytes.getvalue())  # read output as fitz PDF
    # 获取 OCR 处理后的文本块信息
    blocks = ocr_pdf[0].get_text("dict", sort=True, flags=settings.TEXT_FLAGS)["blocks"]
    # 获取 OCR 处理后的完整文本
    full_text = ocr_pdf[0].get_text("text", sort=True, flags=settings.TEXT_FLAGS)

    # 确保原始 PDF/EPUB/MOBI 的边界框和 OCR 处理后的 PDF 的边界框相同
    assert page.bound() == ocr_pdf[0].bound()

    # 如果完整文本为空，则返回空列表
    if len(full_text) == 0:
        return []

    # 如果检测到 OCR 处理不良，则返回空列表
    if detect_bad_ocr(full_text, spellchecker):
        return []

    # 返回文本块信息
    return blocks
```