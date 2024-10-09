# `.\MinerU\magic_pdf\libs\markdown_utils.py`

```
# 导入正则表达式模块
import re


# 定义函数，用于转义Markdown中特殊字符
def escape_special_markdown_char(pymu_blocks):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    # 定义需要转义的特殊字符列表
    special_chars = ["*", "`", "~", "$"]
    # 遍历每个块
    for blk in pymu_blocks:
        # 遍历每个块中的行
        for line in blk['lines']:
            # 遍历每行中的跨度
            for span in line['spans']:
                # 遍历每个特殊字符
                for char in special_chars:
                    # 获取当前跨度的文本
                    span_text = span['text']
                    # 获取当前跨度的类型，如果没有则为None
                    span_type = span.get("_type", None)
                    # 如果类型为 'inline-equation' 或 'interline-equation'，跳过此跨度
                    if span_type in ['inline-equation', 'interline-equation']:
                        continue
                    # 如果当前跨度文本不为空
                    elif span_text:
                        # 用转义字符替换特殊字符
                        span['text'] = span['text'].replace(char, "\\" + char)

    # 返回修改后的块列表
    return pymu_blocks


# 定义函数，用于转义字符串中的Markdown特殊字符
def ocr_escape_special_markdown_char(content):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    # 定义需要转义的特殊字符列表
    special_chars = ["*", "`", "~", "$"]
    # 遍历每个特殊字符
    for char in special_chars:
        # 用转义字符替换特殊字符
        content = content.replace(char, "\\" + char)

    # 返回转义后的内容
    return content
```