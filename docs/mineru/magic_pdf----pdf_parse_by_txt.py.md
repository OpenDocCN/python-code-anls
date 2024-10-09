# `.\MinerU\magic_pdf\pdf_parse_by_txt.py`

```
# 从 magic_pdf.pdf_parse_union_core 模块导入 pdf_parse_union 函数
from magic_pdf.pdf_parse_union_core import pdf_parse_union


# 定义一个解析 PDF 文件的函数，接受多个参数
def parse_pdf_by_txt(
    # PDF 文件的字节数据
    pdf_bytes,
    # 模型列表，用于解析 PDF
    model_list,
    # 图像写入器，用于处理提取的图像
    imageWriter,
    # 开始页 ID，默认值为 0
    start_page_id=0,
    # 结束页 ID，默认为 None，表示解析到最后一页
    end_page_id=None,
    # 调试模式，默认为 False
    debug_mode=False,
):
    # 调用 pdf_parse_union 函数进行 PDF 解析，返回解析结果
    return pdf_parse_union(pdf_bytes,
                           model_list,
                           imageWriter,
                           "txt",  # 指定解析格式为文本
                           start_page_id=start_page_id,  # 设置开始页 ID
                           end_page_id=end_page_id,  # 设置结束页 ID
                           debug_mode=debug_mode,  # 设置调试模式
                           )
```