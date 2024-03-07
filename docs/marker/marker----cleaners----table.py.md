# `.\marker\marker\cleaners\table.py`

```
# 从 marker.bbox 模块中导入 merge_boxes 函数
# 从 marker.schema 模块中导入 Line, Span, Block, Page 类
# 从 copy 模块中导入 deepcopy 函数
# 从 tabulate 模块中导入 tabulate 函数
# 从 typing 模块中导入 List 类型
# 导入 re 模块
# 导入 textwrap 模块
from marker.bbox import merge_boxes
from marker.schema import Line, Span, Block, Page
from copy import deepcopy
from tabulate import tabulate
from typing import List
import re
import textwrap


# 合并表格块
def merge_table_blocks(blocks: List[Page]):
    # 初始化当前行列表和当前边界框
    current_lines = []
    current_bbox = None
    # 遍历每一页
    for page in blocks:
        new_page_blocks = []
        pnum = page.pnum
        # 遍历每个块
        for block in page.blocks:
            # 如果块的最常见类型不是表格
            if block.most_common_block_type() != "Table":
                # 如果当前行列表不为空
                if len(current_lines) > 0:
                    # 创建新的块对象，包含当前行列表和当前页码
                    new_block = Block(
                        lines=deepcopy(current_lines),
                        pnum=pnum,
                        bbox=current_bbox
                    )
                    new_page_blocks.append(new_block)
                    current_lines = []
                    current_bbox = None

                # 将当前块添加到新页块列表中
                new_page_blocks.append(block)
                continue

            # 将块的行添加到当前行列表中
            current_lines.extend(block.lines)
            # 如果当前边界框为空，则设置为块的边界框，否则合并边界框
            if current_bbox is None:
                current_bbox = block.bbox
            else:
                current_bbox = merge_boxes(current_bbox, block.bbox)

        # 如果当前行列表不为空
        if len(current_lines) > 0:
            # 创建新的块对象，包含当前行列表和当前页码
            new_block = Block(
                lines=deepcopy(current_lines),
                pnum=pnum,
                bbox=current_bbox
            )
            new_page_blocks.append(new_block)
            current_lines = []
            current_bbox = None

        # 更新当前页的块列表
        page.blocks = new_page_blocks


# 创建新的表格
def create_new_tables(blocks: List[Page]):
    # 初始化表格索引和正则表达式模式
    table_idx = 0
    dot_pattern = re.compile(r'(\s*\.\s*){4,}')
    dot_multiline_pattern = re.compile(r'.*(\s*\.\s*){4,}.*', re.DOTALL)
    # 遍历每一页中的文本块
    for page in blocks:
        # 遍历每个文本块中的块
        for block in page.blocks:
            # 如果块类型不是表格或者行数小于3，则跳过
            if block.most_common_block_type() != "Table" or len(block.lines) < 3:
                continue

            # 初始化表格行列表和y坐标
            table_rows = []
            y_coord = None
            row = []
            # 遍历每行文本
            for line in block.lines:
                # 遍历每个文本块
                for span in line.spans:
                    # 如果y坐标不同于当前span的起始y坐标
                    if y_coord != span.y_start:
                        # 如果当前行有内容，则添加到表格行列表中
                        if len(row) > 0:
                            table_rows.append(row)
                            row = []
                        y_coord = span.y_start

                    # 获取文本内容并处理多行文本
                    text = span.text
                    if dot_multiline_pattern.match(text):
                        text = dot_pattern.sub(' ', text)
                    row.append(text)
            # 如果当前行有内容，则添加到表格行列表中
            if len(row) > 0:
                table_rows.append(row)

            # 如果表格行字符总长度大于300，或者第一行列数大于8或小于2，则跳过
            if max([len("".join(r)) for r in table_rows]) > 300 or len(table_rows[0]) > 8 or len(table_rows[0]) < 2:
                continue

            # 格式化表格行数据并创建新的Span和Line对象
            new_text = tabulate(table_rows, headers="firstrow", tablefmt="github")
            new_span = Span(
                bbox=block.bbox,
                span_id=f"{table_idx}_fix_table",
                font="Table",
                color=0,
                block_type="Table",
                text=new_text
            )
            new_line = Line(
                bbox=block.bbox,
                spans=[new_span]
            )
            # 替换原有文本块的行为新的行
            block.lines = [new_line]
            table_idx += 1
    # 返回处理过的表格数量
    return table_idx
```