# `.\MinerU\magic_pdf\libs\drop_tag.py`

```
# 定义背景颜色块的常量
COLOR_BG_HEADER_TXT_BLOCK = "color_background_header_txt_block"
# 定义页码常量
PAGE_NO = "page-no" # 页码
# 定义页眉页脚内文本的常量
CONTENT_IN_FOOT_OR_HEADER = 'in-foot-header-area' # 页眉页脚内的文本
# 定义垂直文本的常量
VERTICAL_TEXT = 'vertical-text' # 垂直文本
# 定义旋转文本的常量
ROTATE_TEXT = 'rotate-text' # 旋转文本
# 定义边缘上的空白块的常量
EMPTY_SIDE_BLOCK = 'empty-side-block' # 边缘上的空白没有任何内容的block
# 定义文本在图片上时的常量
ON_IMAGE_TEXT = 'on-image-text' # 文本在图片上
# 定义文本在表格上时的常量
ON_TABLE_TEXT = 'on-table-text' # 文本在表格上

# 定义一个类，用于表示不同的标签类型
class DropTag:
    # 定义页面编号的常量
    PAGE_NUMBER = "page_no"
    # 定义页眉的常量
    HEADER = "header"
    # 定义页脚的常量
    FOOTER = "footer"
    # 定义脚注的常量
    FOOTNOTE = "footnote"
    # 定义不在布局中的常量
    NOT_IN_LAYOUT = "not_in_layout"
    # 定义跨越重叠的常量
    SPAN_OVERLAP = "span_overlap"
    # 定义块重叠的常量
    BLOCK_OVERLAP = "block_overlap"
```