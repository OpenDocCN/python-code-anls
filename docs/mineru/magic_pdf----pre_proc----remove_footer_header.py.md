# `.\MinerU\magic_pdf\pre_proc\remove_footer_header.py`

```
# 导入正则表达式模块
import re

# 从magic_pdf库中导入特定的函数和常量
from magic_pdf.libs.boxbase import _is_in_or_part_overlap
from magic_pdf.libs.drop_tag import CONTENT_IN_FOOT_OR_HEADER, PAGE_NO

# 定义一个函数以删除页面中的页眉、页脚和页码
def remove_headder_footer_one_page(text_raw_blocks, image_bboxes, table_bboxes, header_bboxs, footer_bboxs,
                                   page_no_bboxs, page_w, page_h):
    """
    删除页眉页脚，页码
    从line级别进行删除，删除之后观察这个text-block是否是空的，如果是空的，则移动到remove_list中
    """
    # 初始化页眉和页脚的边界列表
    header = []
    footer = []
    # 如果页眉列表为空
    if len(header) == 0:
        # 将传入的页眉边界赋值给模型页眉
        model_header = header_bboxs
        # 如果模型页眉不为空
        if model_header:
            # 计算页眉的最小和最大 x、y 坐标
            x0 = min([x for x, _, _, _ in model_header])
            y0 = min([y for _, y, _, _ in model_header])
            x1 = max([x1 for _, _, x1, _ in model_header])
            y1 = max([y1 for _, _, _, y1 in model_header])
            # 将计算出的边界赋值给页眉
            header = [x0, y0, x1, y1]
    # 如果页脚列表为空
    if len(footer) == 0:
        # 将传入的页脚边界赋值给模型页脚
        model_footer = footer_bboxs
        # 如果模型页脚不为空
        if model_footer:
            # 计算页脚的最小和最大 x、y 坐标
            x0 = min([x for x, _, _, _ in model_footer])
            y0 = min([y for _, y, _, _ in model_footer])
            x1 = max([x1 for _, _, x1, _ in model_footer])
            y1 = max([y1 for _, _, _, y1 in model_footer])
            # 将计算出的边界赋值给页脚
            footer = [x0, y0, x1, y1]

    # 设置页眉的上边界
    header_y0 = 0 if len(header) == 0 else header[3]
    # 设置页脚的下边界
    footer_y0 = page_h if len(footer) == 0 else footer[1]
    # 如果存在页码边界
    if page_no_bboxs:
        # 分别获取上半部分和下半部分的页码边界
        top_part = [b for b in page_no_bboxs if b[3] < page_h / 2]
        btn_part = [b for b in page_no_bboxs if b[1] > page_h / 2]

        # 计算上半部分页码的最大 y 坐标
        top_max_y0 = max([b[1] for b in top_part]) if top_part else 0
        # 计算下半部分页码的最小 y 坐标
        btn_min_y1 = min([b[3] for b in btn_part]) if btn_part else page_h

        # 更新页眉和页脚的边界
        header_y0 = max(header_y0, top_max_y0)
        footer_y0 = min(footer_y0, btn_min_y1)

    # 计算内容的边界
    content_boundry = [0, header_y0, page_w, footer_y0]

    # 更新页眉的边界
    header = [0, 0, page_w, header_y0]
    # 更新页脚的边界
    footer = [0, footer_y0, page_w, page_h]

    """以上计算出来了页眉页脚的边界，下面开始进行删除"""
    # 初始化要删除的文本块列表
    text_block_to_remove = []
    # 首先检查每个文本块
    for blk in text_raw_blocks:
        # 如果文本块中有行
        if len(blk['lines']) > 0:
            # 遍历文本块中的每一行
            for line in blk['lines']:
                # 初始化要删除的行列表
                line_del = []
                # 遍历行中的每个跨度
                for span in line['spans']:
                    # 初始化要删除的跨度列表
                    span_del = []
                    # 如果跨度的下边界在页眉上方
                    if span['bbox'][3] < header_y0:
                        span_del.append(span)
                    # 如果跨度与页眉或页脚有重叠
                    elif _is_in_or_part_overlap(span['bbox'], header) or _is_in_or_part_overlap(span['bbox'], footer):
                        span_del.append(span)
                # 从行中移除要删除的跨度
                for span in span_del:
                    line['spans'].remove(span)
                # 如果行中没有跨度，则标记为删除
                if not line['spans']:
                    line_del.append(line)

            # 从文本块中移除要删除的行
            for line in line_del:
                blk['lines'].remove(line)
        else:
            # 如果文本块没有行，则标记为页脚或页眉内容
            blk['tag'] = CONTENT_IN_FOOT_OR_HEADER
            # 将文本块添加到删除列表中
            text_block_to_remove.append(blk)

    """有的时候由于pageNo太小了，总是会有一点和content_boundry重叠一点，被放入正文，因此对于pageNo，进行span粒度的删除"""
    # 初始化要删除的页码块列表
    page_no_block_2_remove = []
    # 如果有页码的边界框
    if page_no_bboxs:
        # 遍历每个页码的边界框
        for pagenobox in page_no_bboxs:
            # 遍历原始文本块
            for block in text_raw_blocks:
                # 检查块是否与页码边界框重叠或部分重叠
                if _is_in_or_part_overlap(pagenobox, block['bbox']):  # 在span级别删除页码
                    # 遍历块中的每一行
                    for line in block['lines']:
                        # 遍历行中的每个span
                        for span in line['spans']:
                            # 检查span是否与页码边界框重叠或部分重叠
                            if _is_in_or_part_overlap(pagenobox, span['bbox']):
                                # span['text'] = ''  # 注释掉的代码，原本用于清空span文本
                                span['tag'] = PAGE_NO  # 将span标记为页码
                                # 检查这个块是否只有这一个span，如果是，那么就把这个块也删除
                                if len(line['spans']) == 1 and len(block['lines']) == 1:
                                    page_no_block_2_remove.append(block)  # 将块添加到待删除列表
    else:
        # 测试最后一个块是否是页码：规则是，最后一个块仅有1个line, 一个span, 且text是数字、空格、符号组成，不含字母，并且包含数字
        if len(text_raw_blocks) > 0:
            text_raw_blocks.sort(key=lambda x: x['bbox'][1], reverse=True)  # 按照y坐标降序排列块
            last_block = text_raw_blocks[0]  # 获取最后一个块
            if len(last_block['lines']) == 1:  # 检查最后一个块是否只有一行
                last_line = last_block['lines'][0]  # 获取最后一行
                if len(last_line['spans']) == 1:  # 检查最后一行是否只有一个span
                    last_span = last_line['spans'][0]  # 获取最后一个span
                    # 检查span文本是否有效，包含数字且不含字母
                    if last_span['text'].strip() and not re.search('[a-zA-Z]', last_span['text']) and re.search('[0-9]',
                                                                                                                last_span[
                                                                                                                    'text']):
                        last_span['tag'] = PAGE_NO  # 将span标记为页码
                        page_no_block_2_remove.append(last_block)  # 将最后一个块添加到待删除列表

    # 将待删除的页码块添加到待移除文本块列表
    for b in page_no_block_2_remove:
        text_block_to_remove.append(b)

    # 从原始文本块中移除待删除的块
    for blk in text_block_to_remove:
        if blk in text_raw_blocks:
            text_raw_blocks.remove(blk)

    text_block_remain = text_raw_blocks  # 剩余的文本块
    # 获取不与内容边界重叠的图像边界框
    image_bbox_to_remove = [bbox for bbox in image_bboxes if not _is_in_or_part_overlap(bbox, content_boundry)]

    # 获取与内容边界重叠的图像边界框
    image_bbox_remain = [bbox for bbox in image_bboxes if _is_in_or_part_overlap(bbox, content_boundry)]
    # 获取不与内容边界重叠的表格边界框
    table_bbox_to_remove = [bbox for bbox in table_bboxes if not _is_in_or_part_overlap(bbox, content_boundry)]
    # 获取与内容边界重叠的表格边界框
    table_bbox_remain = [bbox for bbox in table_bboxes if _is_in_or_part_overlap(bbox, content_boundry)]

    # 返回剩余的图像和表格边界框、文本块以及待删除的文本和图像、表格边界框
    return image_bbox_remain, table_bbox_remain, text_block_remain, text_block_to_remove, image_bbox_to_remove, table_bbox_to_remove
```