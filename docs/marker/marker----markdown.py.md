# `.\marker\marker\markdown.py`

```py
# 从 marker.schema 模块中导入 MergedLine, MergedBlock, FullyMergedBlock, Page 类
from marker.schema import MergedLine, MergedBlock, FullyMergedBlock, Page
# 导入 re 模块，用于正则表达式操作
import re
# 从 typing 模块中导入 List 类型
from typing import List

# 定义一个函数，用于在文本两侧添加指定字符
def surround_text(s, char_to_insert):
    # 匹配字符串开头的空白字符
    leading_whitespace = re.match(r'^(\s*)', s).group(1)
    # 匹配字符串结尾的空白字符
    trailing_whitespace = re.search(r'(\s*)$', s).group(1)
    # 去除字符串两侧空白字符
    stripped_string = s.strip()
    # 在去除空白字符后的字符串两侧添加指定字符
    modified_string = char_to_insert + stripped_string + char_to_insert
    # 将添加指定字符后的字符串重新加上空白字符，形成最终字符串
    final_string = leading_whitespace + modified_string + trailing_whitespace
    return final_string

# 定义一个函数，用于合并块
def merge_spans(blocks):
    # 初始化一个空列表用于存储合并后的块
    merged_blocks = []
    return merged_blocks

# 定义一个函数，用于根据块类型对文本进行包围处理
def block_surround(text, block_type):
    if block_type == "Section-header":
        if not text.startswith("#"):
            text = "\n## " + text.strip().title() + "\n"
    elif block_type == "Title":
        if not text.startswith("#"):
            text = "# " + text.strip().title() + "\n"
    elif block_type == "Table":
        text = "\n" + text + "\n"
    elif block_type == "List-item":
        pass
    elif block_type == "Code":
        text = "\n" + text + "\n"
    return text

# 定义一个函数，用于处理文本行之间的分隔符
def line_separator(line1, line2, block_type, is_continuation=False):
    # 包含拉丁衍生语言和俄语的小写字母
    lowercase_letters = "a-zà-öø-ÿа-яşćăâđêôơưþðæøå"
    # 包含拉丁衍生语言和俄语的大写字母
    uppercase_letters = "A-ZÀ-ÖØ-ßА-ЯŞĆĂÂĐÊÔƠƯÞÐÆØÅ"
    # 匹配当前行是否以连字符结尾，且下一行与当前行似乎连接在一起
    hyphen_pattern = re.compile(rf'.*[{lowercase_letters}][-]\s?$', re.DOTALL)
    if line1 and hyphen_pattern.match(line1) and re.match(rf"^[{lowercase_letters}]", line2):
        # 从右侧分割连字符
        line1 = re.split(r"[-—]\s?$", line1)[0]
        return line1.rstrip() + line2.lstrip()

    lowercase_pattern1 = re.compile(rf'.*[{lowercase_letters},]\s?$', re.DOTALL)
    lowercase_pattern2 = re.compile(rf'^\s?[{uppercase_letters}{lowercase_letters}]', re.DOTALL)
    end_pattern = re.compile(r'.*[.?!]\s?$', re.DOTALL)
    # 如果块类型为标题或节标题，则返回去除右侧空格的line1和去除左侧空格的line2拼接的字符串
    if block_type in ["Title", "Section-header"]:
        return line1.rstrip() + " " + line2.lstrip()
    # 如果line1和line2都符合小写模式1和小写模式2，并且块类型为文本，则返回去除右侧空格的line1和去除左侧空格的line2拼接的字符串
    elif lowercase_pattern1.match(line1) and lowercase_pattern2.match(line2) and block_type == "Text":
        return line1.rstrip() + " " + line2.lstrip()
    # 如果是续行，则返回去除右侧空格的line1和去除左侧空格的line2拼接的字符串
    elif is_continuation:
        return line1.rstrip() + " " + line2.lstrip()
    # 如果块类型为文本且line1匹配结束模式，则返回line1后加上两个换行符和line2
    elif block_type == "Text" and end_pattern.match(line1):
        return line1 + "\n\n" + line2
    # 如果块类型为公式，则返回line1后加上一个空格和line2
    elif block_type == "Formula":
        return line1 + " " + line2
    # 其他情况下，返回line1后加上一个换行符和line2
    else:
        return line1 + "\n" + line2
# 定义一个函数，用于确定两个不同类型的文本块之间的分隔符
def block_separator(line1, line2, block_type1, block_type2):
    # 默认分隔符为换行符
    sep = "\n"
    # 如果第一个块的类型是"Text"，则分隔符为两个换行符
    if block_type1 == "Text":
        sep = "\n\n"

    # 返回第二行和分隔符
    return sep + line2


# 合并文本块中的行
def merge_lines(blocks, page_blocks: List[Page]):
    # 存储文本块的列表
    text_blocks = []
    prev_type = None
    prev_line = None
    block_text = ""
    block_type = ""
    # 存储每个页面的常见行高度统计信息
    common_line_heights = [p.get_line_height_stats() for p in page_blocks]
    # 遍历每个页面的文本块
    for page in blocks:
        for block in page:
            # 获取当前文本块的最常见类型
            block_type = block.most_common_block_type()
            # 如果当前类型与前一个类型不同且前一个类型存在，则将前一个文本块添加到列表中
            if block_type != prev_type and prev_type:
                text_blocks.append(
                    FullyMergedBlock(
                        text=block_surround(block_text, prev_type),
                        block_type=prev_type
                    )
                )
                block_text = ""

            prev_type = block_type
            # 将文本块中的行合并在一起
            for i, line in enumerate(block.lines):
                line_height = line.bbox[3] - line.bbox[1]
                prev_line_height = prev_line.bbox[3] - prev_line.bbox[1] if prev_line else 0
                prev_line_x = prev_line.bbox[0] if prev_line else 0
                prev_line = line
                is_continuation = line_height == prev_line_height and line.bbox[0] == prev_line_x
                if block_text:
                    block_text = line_separator(block_text, line.text, block_type, is_continuation)
                else:
                    block_text = line.text

    # 将最后一个文本块添加到列表中
    text_blocks.append(
        FullyMergedBlock(
            text=block_surround(block_text, prev_type),
            block_type=block_type
        )
    )
    return text_blocks


# 获取完整的文本
def get_full_text(text_blocks):
    full_text = ""
    prev_block = None
    # 遍历文本块列表
    for block in text_blocks:
        # 如果存在前一个文本块
        if prev_block:
            # 将前一个文本块、当前文本块、前一个文本块类型和当前文本块类型传入block_separator函数，将返回的结果添加到full_text中
            full_text += block_separator(prev_block.text, block.text, prev_block.block_type, block.block_type)
        else:
            # 如果不存在前一个文本块，直接将当前文本块的内容添加到full_text中
            full_text += block.text
        # 更新prev_block为当前文本块
        prev_block = block
    # 返回完整的文本
    return full_text
```