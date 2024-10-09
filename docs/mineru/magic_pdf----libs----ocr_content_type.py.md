# `.\MinerU\magic_pdf\libs\ocr_content_type.py`

```
# 定义内容类型的类
class ContentType:
    # 定义图像内容类型
    Image = 'image'
    # 定义表格内容类型
    Table = 'table'
    # 定义文本内容类型
    Text = 'text'
    # 定义行内公式内容类型
    InlineEquation = 'inline_equation'
    # 定义行间公式内容类型
    InterlineEquation = 'interline_equation'


# 定义块类型的类
class BlockType:
    # 定义图像块类型
    Image = 'image'
    # 定义图像主体块类型
    ImageBody = 'image_body'
    # 定义图像标题块类型
    ImageCaption = 'image_caption'
    # 定义图像脚注块类型
    ImageFootnote = 'image_footnote'
    # 定义表格块类型
    Table = 'table'
    # 定义表格主体块类型
    TableBody = 'table_body'
    # 定义表格标题块类型
    TableCaption = 'table_caption'
    # 定义表格脚注块类型
    TableFootnote = 'table_footnote'
    # 定义文本块类型
    Text = 'text'
    # 定义标题块类型
    Title = 'title'
    # 定义行间公式块类型
    InterlineEquation = 'interline_equation'
    # 定义脚注块类型
    Footnote = 'footnote'
    # 定义被丢弃的块类型
    Discarded = 'discarded'


# 定义类别 ID 的类
class CategoryId:
    # 定义标题类别 ID
    Title = 0
    # 定义文本类别 ID
    Text = 1
    # 定义废弃类别 ID
    Abandon = 2
    # 定义图像主体类别 ID
    ImageBody = 3
    # 定义图像标题类别 ID
    ImageCaption = 4
    # 定义表格主体类别 ID
    TableBody = 5
    # 定义表格标题类别 ID
    TableCaption = 6
    # 定义表格脚注类别 ID
    TableFootnote = 7
    # 定义行间公式布局类别 ID
    InterlineEquation_Layout = 8
    # 定义行内公式类别 ID
    InlineEquation = 13
    # 定义行间公式 YOLO 类别 ID
    InterlineEquation_YOLO = 14
    # 定义 OCR 文本类别 ID
    OcrText = 15
    # 定义图像脚注类别 ID
    ImageFootnote = 101
```