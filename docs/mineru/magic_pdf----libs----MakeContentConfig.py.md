# `.\MinerU\magic_pdf\libs\MakeContentConfig.py`

```
# 定义一个表示不同模式的类
class MakeMode:
    # 定义一个常量，表示 Markdown 格式的模式
    MM_MD = "mm_markdown"
    # 定义一个常量，表示 NLP 相关的 Markdown 格式模式
    NLP_MD = "nlp_markdown"
    # 定义一个常量，表示标准格式
    STANDARD_FORMAT = "standard_format"


# 定义一个表示丢弃模式的类
class DropMode:
    # 定义一个常量，表示丢弃整个 PDF
    WHOLE_PDF = "whole_pdf"
    # 定义一个常量，表示丢弃单页 PDF
    SINGLE_PAGE = "single_page"
    # 定义一个常量，表示不丢弃任何内容
    NONE = "none"
```