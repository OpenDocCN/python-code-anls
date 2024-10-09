# `.\MinerU\magic_pdf\pdf_parse_by_ocr.py`

```
# 从 magic_pdf 模块导入 pdf_parse_union 函数
from magic_pdf.pdf_parse_union_core import pdf_parse_union

# 定义一个解析 PDF 的函数，使用 OCR 方法
def parse_pdf_by_ocr(pdf_bytes,
                     model_list,
                     imageWriter,
                     start_page_id=0,  # 设置解析起始页，默认为 0
                     end_page_id=None,  # 设置解析结束页，默认为 None（解析到最后一页）
                     debug_mode=False,  # 设置调试模式，默认为 False
                     ):
    # 调用 pdf_parse_union 函数解析 PDF，传入相关参数
    return pdf_parse_union(pdf_bytes,
                           model_list,
                           imageWriter,
                           "ocr",  # 指定使用 OCR 方法解析
                           start_page_id=start_page_id,  # 传递起始页参数
                           end_page_id=end_page_id,  # 传递结束页参数
                           debug_mode=debug_mode,  # 传递调试模式参数
                           )
```