# `.\MinerU\magic_pdf\pre_proc\main_text_font.py`

```
# 导入 collections 模块以便使用计数器
import collections


# 定义一个函数，用于获取 PDF 文档中主要文本的字体
def get_main_text_font(pdf_docs):
    # 创建一个计数器，用于统计字体名称及其出现次数
    font_names = collections.Counter()
    # 遍历每个页面
    for page in pdf_docs:
        # 获取页面中所有文本块的字典表示
        blocks = page.get_text('dict')['blocks']
        # 如果存在文本块，则继续处理
        if blocks is not None:
            # 遍历每个文本块
            for block in blocks:
                # 获取文本块中的行
                lines = block.get('lines')
                # 如果存在行，则继续处理
                if lines is not None:
                    # 遍历每一行
                    for line in lines:
                        # 获取行中每个跨度的字体及其文本长度
                        span_font = [(span['font'], len(span['text'])) for span in line['spans'] if
                                     'font' in span and len(span['text']) > 0]
                        # 如果存在有效的跨度字体，则继续处理
                        if span_font:
                            # 主要文本字体应基于字数最多的字体进行统计
                            # font_names.append(font_name for font_name in span_font)
                            # block_fonts.append(font_name for font_name in span_font)
                            # 遍历每个跨度字体及其对应的文本长度
                            for font, count in span_font:
                                # 更新计数器中对应字体的字数
                                font_names[font] += count
    # 获取出现次数最多的字体
    main_text_font = font_names.most_common(1)[0][0]
    # 返回主要文本字体
    return main_text_font
```