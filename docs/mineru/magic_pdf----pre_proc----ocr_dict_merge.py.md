# `.\MinerU\magic_pdf\pre_proc\ocr_dict_merge.py`

```
# 从指定库中导入相关函数和类
from magic_pdf.libs.boxbase import (__is_overlaps_y_exceeds_threshold,
                                    _is_in_or_part_overlap_with_area_ratio,
                                    calculate_overlap_area_in_bbox1_area_ratio)
from magic_pdf.libs.drop_tag import DropTag
from magic_pdf.libs.ocr_content_type import BlockType, ContentType


# 将每一个line中的span从左到右排序
def line_sort_spans_by_left_to_right(lines):
    # 初始化一个空列表，用于存储行对象
    line_objects = []
    # 遍历每一行
    for line in lines:
        # 按照x0坐标对每一行中的span进行排序
        line.sort(key=lambda span: span['bbox'][0])
        # 计算当前行的包围框（bounding box）
        line_bbox = [
            min(span['bbox'][0] for span in line),  # 获取当前行中所有span的最小x0
            min(span['bbox'][1] for span in line),  # 获取当前行中所有span的最小y0
            max(span['bbox'][2] for span in line),  # 获取当前行中所有span的最大x1
            max(span['bbox'][3] for span in line),  # 获取当前行中所有span的最大y1
        ]
        # 将当前行的包围框和span信息添加到line_objects中
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    # 返回所有行的对象
    return line_objects


def merge_spans_to_line(spans):
    # 如果spans列表为空，则返回空列表
    if len(spans) == 0:
        return []
    else:
        # 按照y0坐标对span进行排序
        spans.sort(key=lambda span: span['bbox'][1])

        # 初始化行列表和当前行
        lines = []
        current_line = [spans[0]]  # 将第一个span作为当前行的开始
        # 遍历从第二个span开始的所有span
        for span in spans[1:]:
            # 如果当前span类型为"interline_equation" 或者 当前行中已有"interline_equation"
            # 同理适用于image和table类型
            if span['type'] in [
                    ContentType.InterlineEquation, ContentType.Image,
                    ContentType.Table
            ] or any(s['type'] in [
                    ContentType.InterlineEquation, ContentType.Image,
                    ContentType.Table
            ] for s in current_line):
                # 将当前行添加到lines中，并开始新行
                lines.append(current_line)
                current_line = [span]
                continue

            # 如果当前的span与当前行的最后一个span在y轴上重叠，则添加到当前行
            if __is_overlaps_y_exceeds_threshold(span['bbox'],
                                                 current_line[-1]['bbox']):
                current_line.append(span)  # 添加到当前行
            else:
                # 否则，开始新行
                lines.append(current_line)
                current_line = [span]  # 将当前span作为新行的开始

        # 如果current_line非空，添加最后一行到lines中
        if current_line:
            lines.append(current_line)

        # 返回合并后的行列表
        return lines


def merge_spans_to_line_by_layout(spans, layout_bboxes):
    # 初始化行列表、新span列表和丢弃的span列表
    lines = []
    new_spans = []
    dropped_spans = []
    # 遍历每个布局框
    for item in layout_bboxes:
        layout_bbox = item['layout_bbox']
        # 初始化一个空列表，用于存放当前布局中的span
        layout_sapns = []
        # 遍历所有span，将每个span放入对应的layout中
        for span in spans:
            # 如果span与layout_bbox的重叠面积比例大于0.6，则将其加入layout_sapns
            if calculate_overlap_area_in_bbox1_area_ratio(
                    span['bbox'], layout_bbox) > 0.6:
                layout_sapns.append(span)
        # 如果layout_sapns不为空，则放入new_spans中
        if len(layout_sapns) > 0:
            new_spans.append(layout_sapns)
            # 从spans中删除已经放入layout_sapns中的span
            for layout_sapn in layout_sapns:
                spans.remove(layout_sapn)
    # 如果 new_spans 列表中有元素
    if len(new_spans) > 0:
        # 遍历 new_spans 列表中的每一个布局跨度
        for layout_sapns in new_spans:
            # 将布局跨度合并成行
            layout_lines = merge_spans_to_line(layout_sapns)
            # 将合并后的行添加到 lines 列表中
            lines.extend(layout_lines)

    # 对 lines 中的跨度进行从左到右的排序
    lines = line_sort_spans_by_left_to_right(lines)

    # 遍历所有的跨度
    for span in spans:
        # 将每个跨度的标签设置为不在布局中
        span['tag'] = DropTag.NOT_IN_LAYOUT
        # 将处理后的跨度添加到 dropped_spans 列表中
        dropped_spans.append(span)

    # 返回排序后的行和丢弃的跨度列表
    return lines, dropped_spans
# 将输入的行合并成一个块结构，每个块只包含一个行，并且块的边界框是行的边界框
def merge_lines_to_block(lines):
    # 初始化一个空列表，用于存放块
    blocks = []
    # 遍历每一行
    for line in lines:
        # 将每行的边界框和行本身组成一个块字典，并添加到块列表中
        blocks.append({
            'bbox': line['bbox'],
            'lines': [line],
        })
    # 返回所有块的列表
    return blocks


# 根据布局的边界框对块进行排序
def sort_blocks_by_layout(all_bboxes, layout_bboxes):
    # 初始化新的块列表和排序后的块列表
    new_blocks = []
    sort_blocks = []
    # 遍历每个布局边界框
    for item in layout_bboxes:
        # 获取当前布局的边界框
        layout_bbox = item['layout_bbox']

        # 初始化一个列表，用于存放与当前布局匹配的块
        layout_blocks = []
        # 遍历所有块
        for block in all_bboxes:
            # 如果块类型为脚注，则跳过
            if block[7] == BlockType.Footnote:
                continue
            # 提取块的边界框
            block_bbox = block[:4]
            # 如果块与布局的重叠面积比大于0.8，则将块添加到布局块列表中
            if calculate_overlap_area_in_bbox1_area_ratio(
                    block_bbox, layout_bbox) > 0.8:
                layout_blocks.append(block)

        # 如果找到的布局块不为空，则将其添加到新的块列表中
        if len(layout_blocks) > 0:
            new_blocks.append(layout_blocks)
            # 从所有块中删除已添加到布局块列表的块
            for layout_block in layout_blocks:
                all_bboxes.remove(layout_block)

    # 如果新的块列表不为空，对其中的每个块进行排序
    if len(new_blocks) > 0:
        for bboxes_in_layout_block in new_blocks:
            # 根据块的第二个元素（y0坐标）进行排序
            bboxes_in_layout_block.sort(
                key=lambda x: x[1])  # 一个layout内部的box，按照y0自上而下排序
            # 将排序后的块扩展到排序块列表中
            sort_blocks.extend(bboxes_in_layout_block)

    # 返回排序后的块列表
    return sort_blocks


# 将所有跨度根据位置关系填入块中
def fill_spans_in_blocks(blocks, spans, radio):
    """将allspans中的span按位置关系，放入blocks中."""
    # 初始化一个空列表，用于存放包含跨度的块
    block_with_spans = []
    # 遍历每个块
    for block in blocks:
        # 获取块的类型和边界框
        block_type = block[7]
        block_bbox = block[0:4]
        # 创建一个字典存储块的信息
        block_dict = {
            'type': block_type,
            'bbox': block_bbox,
        }
        # 初始化一个空列表，用于存放与块重叠的跨度
        block_spans = []
        # 遍历所有跨度
        for span in spans:
            # 获取跨度的边界框
            span_bbox = span['bbox']
            # 如果跨度与块的重叠面积比大于给定比例，则将跨度添加到块的跨度列表中
            if calculate_overlap_area_in_bbox1_area_ratio(
                    span_bbox, block_bbox) > radio:
                block_spans.append(span)
        '''行内公式调整, 高度调整至与同行文字高度一致(优先左侧, 其次右侧)'''
        # displayed_list = []
        # text_inline_lines = []
        # modify_y_axis(block_spans, displayed_list, text_inline_lines)
        '''模型识别错误的行间公式, type类型转换成行内公式'''
        # block_spans = modify_inline_equation(block_spans, displayed_list, text_inline_lines)
        '''bbox去除粘连'''  # 去粘连会影响span的bbox，导致后续fill的时候出错
        # block_spans = remove_overlap_between_bbox_for_span(block_spans)

        # 将找到的跨度添加到块字典中
        block_dict['spans'] = block_spans
        # 将块字典添加到包含跨度的块列表中
        block_with_spans.append(block_dict)

        # 从跨度列表中删除已添加到块的跨度
        if len(block_spans) > 0:
            for span in block_spans:
                spans.remove(span)

    # 返回包含跨度的块和剩余的跨度
    return block_with_spans, spans


# 修复块中的跨度，处理嵌套关系
def fix_block_spans(block_with_spans, img_blocks, table_blocks):
    """1、img_block和table_block因为包含caption和footnote的关系，存在block的嵌套关系
    需要将caption和footnote的text_span放入相应img_block和table_block内的
    caption_block和footnote_block中 2、同时需要删除block中的spans字段."""
    # 初始化一个空列表，用于存放修复后的块
    fix_blocks = []
    # 遍历包含各种块的列表
        for block in block_with_spans:
            # 获取当前块的类型
            block_type = block['type']
    
            # 如果块类型是图像
            if block_type == BlockType.Image:
                # 修正图像块
                block = fix_image_block(block, img_blocks)
            # 如果块类型是表格
            elif block_type == BlockType.Table:
                # 修正表格块
                block = fix_table_block(block, table_blocks)
            # 如果块类型是文本或标题
            elif block_type in [BlockType.Text, BlockType.Title]:
                # 修正文本块
                block = fix_text_block(block)
            # 如果块类型是行间公式
            elif block_type == BlockType.InterlineEquation:
                # 修正行间公式块
                block = fix_interline_block(block)
            # 如果块类型不匹配，则跳过
            else:
                continue
            # 将修正后的块添加到修正块列表中
            fix_blocks.append(block)
        # 返回修正后的块列表
        return fix_blocks
# 修复被丢弃的块，并返回修复后的块列表
def fix_discarded_block(discarded_block_with_spans):
    # 初始化修复后的块列表
    fix_discarded_blocks = []
    # 遍历丢弃的块
    for block in discarded_block_with_spans:
        # 修复文本块
        block = fix_text_block(block)
        # 将修复后的块添加到列表中
        fix_discarded_blocks.append(block)
    # 返回修复后的块列表
    return fix_discarded_blocks


# 将给定的跨度合并到块中
def merge_spans_to_block(spans: list, block_bbox: list, block_type: str):
    # 初始化块的跨度列表
    block_spans = []
    # 如果有 img_caption，则将 img_block 中的 text_spans 放入 img_caption_block 中
    for span in spans:
        # 计算当前跨度与块的重叠比例
        if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'],
                                                      block_bbox) > 0.6:
            # 如果重叠比例大于0.6，添加跨度到块中
            block_spans.append(span)
    # 将块的跨度合并成行
    block_lines = merge_spans_to_line(block_spans)
    # 对行中的跨度进行排序
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    # 创建块字典，包含边界框、类型和行
    block = {'bbox': block_bbox, 'type': block_type, 'lines': sort_block_lines}
    # 返回块和其跨度
    return block, block_spans


# 创建主体块
def make_body_block(span: dict, block_bbox: list, block_type: str):
    # 创建主体行字典，包含边界框和跨度
    body_line = {
        'bbox': block_bbox,
        'spans': [span],
    }
    # 创建主体块字典
    body_block = {'bbox': block_bbox, 'type': block_type, 'lines': [body_line]}
    # 返回主体块
    return body_block


# 修复图像块
def fix_image_block(block, img_blocks):
    # 初始化块的子块列表
    block['blocks'] = []
    # 遍历图像块，查找匹配的图像块
    for img_block in img_blocks:
        # 判断当前块与图像块是否重叠
        if _is_in_or_part_overlap_with_area_ratio(block['bbox'],
                                                  img_block['bbox'], 0.95):

            # 创建 img_body_block
            for span in block['spans']:
                # 如果跨度类型为图像，且边界框匹配
                if span['type'] == ContentType.Image and img_block[
                        'img_body_bbox'] == span['bbox']:
                    # 创建图像主体块
                    img_body_block = make_body_block(
                        span, img_block['img_body_bbox'], BlockType.ImageBody)
                    # 添加图像主体块到当前块
                    block['blocks'].append(img_body_block)

                    # 从跨度中移除已添加的跨度
                    block['spans'].remove(span)
                    break

            # 如果图像块有 img_caption
            if img_block['img_caption_bbox'] is not None:
                # 合并跨度到 img_caption 块
                img_caption_block, img_caption_spans = merge_spans_to_block(
                    block['spans'], img_block['img_caption_bbox'],
                    BlockType.ImageCaption)
                # 添加 img_caption 块到当前块
                block['blocks'].append(img_caption_block)

            # 如果图像块有 img_footnote
            if img_block['img_footnote_bbox'] is not None:
                # 合并跨度到 img_footnote 块
                img_footnote_block, img_footnote_spans = merge_spans_to_block(
                    block['spans'], img_block['img_footnote_bbox'],
                    BlockType.ImageFootnote)
                # 添加 img_footnote 块到当前块
                block['blocks'].append(img_footnote_block)
            break
    # 删除当前块的跨度
    del block['spans']
    # 返回更新后的块
    return block


# 修复表格块
def fix_table_block(block, table_blocks):
    # 初始化块的子块列表
    block['blocks'] = []
    # 遍历表格块，查找匹配的表格块
    # 遍历每个表格块
    for table_block in table_blocks:
        # 检查当前块的边界框是否与表格块的边界框重叠，重叠比例大于 0.95
        if _is_in_or_part_overlap_with_area_ratio(block['bbox'],
                                                  table_block['bbox'], 0.95):

            # 创建表格主体块
            for span in block['spans']:
                # 检查 span 类型是否为表格，且其边界框与表格块匹配
                if span['type'] == ContentType.Table and table_block[
                        'table_body_bbox'] == span['bbox']:
                    # 调用函数创建表格主体块
                    table_body_block = make_body_block(
                        span, table_block['table_body_bbox'],
                        BlockType.TableBody)
                    # 将创建的表格主体块添加到当前块的块列表中
                    block['blocks'].append(table_body_block)

                    # 从 spans 列表中移除已经放入的 span
                    block['spans'].remove(span)
                    # 退出循环
                    break

            # 检查表格块是否包含标题
            if table_block['table_caption_bbox'] is not None:
                # 合并与标题相关的 spans，创建标题块
                table_caption_block, table_caption_spans = merge_spans_to_block(
                    block['spans'], table_block['table_caption_bbox'],
                    BlockType.TableCaption)
                # 将创建的标题块添加到当前块的块列表中
                block['blocks'].append(table_caption_block)

                # 如果标题块的 spans 不为空
                if len(table_caption_spans) > 0:
                    # 从 spans 中删除已放入标题块的 spans
                    for span in table_caption_spans:
                        block['spans'].remove(span)

            # 检查表格块是否包含脚注
            if table_block['table_footnote_bbox'] is not None:
                # 合并与脚注相关的 spans，创建脚注块
                table_footnote_block, table_footnote_spans = merge_spans_to_block(
                    block['spans'], table_block['table_footnote_bbox'],
                    BlockType.TableFootnote)
                # 将创建的脚注块添加到当前块的块列表中
                block['blocks'].append(table_footnote_block)

            # 退出循环
            break
    # 删除当前块中的 spans 列表
    del block['spans']
    # 返回处理后的块
    return block
# 修复文本块，将公式类型转换为行内类型
def fix_text_block(block):
    # 遍历文本块中的每个公式 span
    for span in block['spans']:
        # 如果 span 的类型是行间公式
        if span['type'] == ContentType.InterlineEquation:
            # 将其转换为行内公式
            span['type'] = ContentType.InlineEquation
    # 将 span 合并为行
    block_lines = merge_spans_to_line(block['spans'])
    # 按从左到右的顺序对行进行排序
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    # 将排序后的行添加到文本块中
    block['lines'] = sort_block_lines
    # 删除原始的 spans 信息
    del block['spans']
    # 返回修复后的文本块
    return block


# 修复行间块，将其转换为行格式
def fix_interline_block(block):
    # 将 spans 合并为行
    block_lines = merge_spans_to_line(block['spans'])
    # 按从左到右的顺序对行进行排序
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    # 将排序后的行添加到块中
    block['lines'] = sort_block_lines
    # 删除原始的 spans 信息
    del block['spans']
    # 返回修复后的块
    return block
```