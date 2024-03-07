# `.\marker\marker\cleaners\code.py`

```
# 导入所需的模块和类
from marker.schema import Span, Line, Page
import re
from typing import List
import fitz as pymupdf

# 判断代码行的长度是否符合阈值
def is_code_linelen(lines, thresh=60):
    # 计算所有代码行中的字母数字字符总数
    total_alnum_chars = sum(len(re.findall(r'\w', line.prelim_text)) for line in lines)
    # 计算总行数
    total_newlines = max(len(lines) - 1, 1)

    # 如果没有字母数字字符，则返回 False
    if total_alnum_chars == 0:
        return False

    # 计算字母数字字符与行数的比率
    ratio = total_alnum_chars / total_newlines
    return ratio < thresh

# 统计代码行中包含注释的行数
def comment_count(lines):
    # 定义匹配注释的正则表达式模式
    pattern = re.compile(r"^(//|#|'|--|/\*|'''|\"\"\"|--\[\[|<!--|%|%{|\(\*)")
    # 统计匹配到的注释行数
    return sum([1 for line in lines if pattern.match(line)])

# 识别代码块
def identify_code_blocks(blocks: List[Page]):
    # 初始化代码块计数和字体信息
    code_block_count = 0
    font_info = None
    # 遍历每个页面
    for p in blocks:
        # 获取页面的字体统计信息
        stats = p.get_font_stats()
        # 如果是第一页，则将字体信息初始化为当前页面的字体信息
        if font_info is None:
            font_info = stats
        else:
            # 否则将当前页面的字体信息与之前页面的字体信息相加
            font_info += stats
    try:
        # 获取最常见的字体
        most_common_font = font_info.most_common(1)[0][0]
    except IndexError:
        # 如果找不到最常见的字体，则打印提示信息
        print(f"Could not find most common font")
        most_common_font = None

    # 初始化最后一个代码块
    last_block = None
    # 遍历每一页的文本块
    for page in blocks:
        try:
            # 获取当前页最小行的起始位置
            min_start = page.get_min_line_start()
        except IndexError:
            # 如果出现索引错误，则跳过当前页
            continue

        # 遍历当前页的文本块
        for block in page.blocks:
            # 如果当前文本块的类型不是"Text"，则跳过
            if block.most_common_block_type() != "Text":
                last_block = block
                continue

            # 初始化用于判断是否为代码的变量
            is_indent = []
            line_fonts = []
            # 遍历当前文本块的每一行
            for line in block.lines:
                # 获取每行中的字体信息
                fonts = [span.font for span in line.spans]
                line_fonts += fonts
                # 获取每行的起始位置
                line_start = line.bbox[0]
                # 判断当前行是否缩进
                if line_start > min_start:
                    is_indent.append(True)
                else:
                    is_indent.append(False)
            # 统计每个文本块中的注释行数
            comment_lines = comment_count([line.prelim_text for line in block.lines])
            # 判断当前文本块是否为代码块
            is_code = [
                len(block.lines) > 3,  # 文本块行数大于3
                sum([f != most_common_font for f in line_fonts]) > len(line_fonts) * .8,  # 至少80%的字体不是最常见的字体，因为代码通常使用与主体文本不同的字体
                is_code_linelen(block.lines),  # 判断代码行长度是否符合规范
                (
                    sum(is_indent) > len(block.lines) * .2  # 20%的行有缩进
                    or
                    comment_lines > len(block.lines) * .2  # 20%的行是注释
                 ), 
            ]

            # 检查前一个文本块是否为代码块，当前文本块是否有缩进
            is_code_prev = [
                last_block and last_block.most_common_block_type() == "Code",  # 前一个文本块是代码块
                sum(is_indent) >= len(block.lines) * .8  # 至少80%的行有缩进
            ]

            # 如果当前文本块被判断为代码块，增加代码块计数并设置文本块类型为"Code"
            if all(is_code) or all(is_code_prev):
                code_block_count += 1
                block.set_block_type("Code")

            last_block = block
    # 返回代码块计数
    return code_block_count
# 缩进代码块，将每个代码块的内容整理成一个新的 Span 对象
def indent_blocks(blocks: List[Page]):
    # 计数器，用于生成新的 Span 对象的 ID
    span_counter = 0
    # 遍历每一页的代码块
    for page in blocks:
        for block in page.blocks:
            # 获取当前代码块中所有行的块类型
            block_types = [span.block_type for line in block.lines for span in line.spans]
            # 如果当前代码块不是代码块，则跳过
            if "Code" not in block_types:
                continue

            # 初始化空列表用于存储处理后的行数据
            lines = []
            # 初始化最左边界和字符宽度
            min_left = 1000  # will contain x- coord of column 0
            col_width = 0  # width of 1 char
            # 遍历当前代码块的每一行
            for line in block.lines:
                text = ""
                # 更新最左边界
                min_left = min(line.bbox[0], min_left)
                # 拼接每行的文本内容
                for span in line.spans:
                    if col_width == 0 and len(span.text) > 0:
                        col_width = (span.bbox[2] - span.bbox[0]) / len(span.text)
                    text += span.text
                lines.append((pymupdf.Rect(line.bbox), text))

            # 初始化空字符串用于存储处理后的代码块文本
            block_text = ""
            blank_line = False
            # 遍历处理后的每一行
            for line in lines:
                text = line[1]
                prefix = " " * int((line[0].x0 - min_left) / col_width)
                current_line_blank = len(text.strip()) == 0
                # 如果当前行和上一行都是空行，则跳过
                if blank_line and current_line_blank:
                    continue

                # 拼接处理后的代码块文本
                block_text += prefix + text + "\n"
                blank_line = current_line_blank

            # 创建新的 Span 对象，用于替换原有的代码块
            new_span = Span(
                text=block_text,
                bbox=block.bbox,
                color=block.lines[0].spans[0].color,
                span_id=f"{span_counter}_fix_code",
                font=block.lines[0].spans[0].font,
                block_type="Code"
            )
            span_counter += 1
            # 替换原有的代码块内容为新的 Span 对象
            block.lines = [Line(spans=[new_span], bbox=block.bbox)]
```