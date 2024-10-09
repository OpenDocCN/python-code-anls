# `.\MinerU\magic_pdf\pre_proc\ocr_detect_all_bboxes.py`

```
# 导入日志记录库
from loguru import logger

# 从特定模块导入函数和类，用于处理边界框和计算重叠
from magic_pdf.libs.boxbase import get_minbox_if_overlap_by_ratio, calculate_overlap_area_in_bbox1_area_ratio, \
    calculate_iou
from magic_pdf.libs.drop_tag import DropTag
from magic_pdf.libs.ocr_content_type import BlockType
from magic_pdf.pre_proc.remove_bbox_overlap import remove_overlap_between_bbox_for_block

# 准备边界框以进行布局分割，接收多个块和页面尺寸作为参数
def ocr_prepare_bboxes_for_layout_split(img_blocks, table_blocks, discarded_blocks, text_blocks,
                                        title_blocks, interline_equation_blocks, page_w, page_h):
    # 初始化空列表以存储所有边界框
    all_bboxes = []
    # 初始化空列表以存储所有被丢弃的块
    all_discarded_blocks = []
    
    # 遍历图像块，提取边界框并添加到列表
    for image in img_blocks:
        x0, y0, x1, y1 = image['bbox']  # 解包图像块的边界框
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Image, None, None, None, None, image["score"]])  # 添加图像边界框信息

    # 遍历表格块，提取边界框并添加到列表
    for table in table_blocks:
        x0, y0, x1, y1 = table['bbox']  # 解包表格块的边界框
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Table, None, None, None, None, table["score"]])  # 添加表格边界框信息

    # 遍历文本块，提取边界框并添加到列表
    for text in text_blocks:
        x0, y0, x1, y1 = text['bbox']  # 解包文本块的边界框
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Text, None, None, None, None, text["score"]])  # 添加文本边界框信息

    # 遍历标题块，提取边界框并添加到列表
    for title in title_blocks:
        x0, y0, x1, y1 = title['bbox']  # 解包标题块的边界框
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Title, None, None, None, None, title["score"]])  # 添加标题边界框信息

    # 遍历行间公式块，提取边界框并添加到列表
    for interline_equation in interline_equation_blocks:
        x0, y0, x1, y1 = interline_equation['bbox']  # 解包行间公式块的边界框
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.InterlineEquation, None, None, None, None, interline_equation["score"]])  # 添加行间公式边界框信息

    '''block嵌套问题解决'''
    '''文本框与标题框重叠，优先信任文本框'''
    all_bboxes = fix_text_overlap_title_blocks(all_bboxes)  # 解决文本框与标题框的重叠问题

    '''任何框体与舍弃框重叠，优先信任舍弃框'''
    all_bboxes = remove_need_drop_blocks(all_bboxes, discarded_blocks)  # 处理与丢弃框的重叠

    # interline_equation 与title或text框冲突的情况，分两种情况处理
    '''interline_equation框与文本类型框iou比较接近1的时候，信任行间公式框'''
    all_bboxes = fix_interline_equation_overlap_text_blocks_with_hi_iou(all_bboxes)  # 处理与文本框的重叠

    '''interline_equation框被包含在文本类型框内，且interline_equation比文本区块小很多时信任文本框，这时需要舍弃公式框'''
    # 通过后续大框套小框逻辑删除

    '''discarded_blocks中只保留宽度超过1/3页面宽度的，高度超过10的，处于页面下半50%区域的（限定footnote）'''
    for discarded in discarded_blocks:
        x0, y0, x1, y1 = discarded['bbox']  # 解包被丢弃块的边界框
        all_discarded_blocks.append([x0, y0, x1, y1, None, None, None, BlockType.Discarded, None, None, None, None, discarded["score"]])  # 添加被丢弃块边界框信息
        
        # 将footnote加入到all_bboxes中，用来计算layout
        if (x1 - x0) > (page_w / 3) and (y1 - y0) > 10 and y0 > (page_h / 2):  # 检查是否符合条件
            all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Footnote, None, None, None, None, discarded["score"]])  # 添加footnote边界框信息

    '''经过以上处理后，还存在大框套小框的情况，则删除小框'''
    all_bboxes = remove_overlaps_min_blocks(all_bboxes)  # 删除小框以避免重叠
    all_discarded_blocks = remove_overlaps_min_blocks(all_discarded_blocks)  # 同样处理被丢弃的块
    
    '''将剩余的bbox做分离处理，防止后面分layout时出错'''
    all_bboxes, drop_reasons = remove_overlap_between_bbox_for_block(all_bboxes)  # 去除重叠边界框以防后续错误
    # 返回包含所有边界框、被丢弃的块和丢弃原因的元组
    return all_bboxes, all_discarded_blocks, drop_reasons
# 修复文本块和行间公式块的重叠
def fix_interline_equation_overlap_text_blocks_with_hi_iou(all_bboxes):
    # 初始化列表以存储所有文本块
    text_blocks = []
    # 遍历所有边界框，提取文本块
    for block in all_bboxes:
        if block[7] == BlockType.Text:
            text_blocks.append(block)
    # 初始化列表以存储行间公式块
    interline_equation_blocks = []
    # 遍历所有边界框，提取行间公式块
    for block in all_bboxes:
        if block[7] == BlockType.InterlineEquation:
            interline_equation_blocks.append(block)

    # 初始化列表以存储需要移除的块
    need_remove = []

    # 检查行间公式块与文本块的重叠
    for interline_equation_block in interline_equation_blocks:
        for text_block in text_blocks:
            interline_equation_block_bbox = interline_equation_block[:4]
            text_block_bbox = text_block[:4]
            # 计算重叠率，如果大于0.8，标记文本块为待移除
            if calculate_iou(interline_equation_block_bbox, text_block_bbox) > 0.8:
                if text_block not in need_remove:
                    need_remove.append(text_block)

    # 如果有需要移除的块，则从原始列表中移除
    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)

    # 返回处理后的边界框列表
    return all_bboxes


# 修复文本块和标题块的重叠
def fix_text_overlap_title_blocks(all_bboxes):
    # 初始化列表以存储所有文本块
    text_blocks = []
    # 遍历所有边界框，提取文本块
    for block in all_bboxes:
        if block[7] == BlockType.Text:
            text_blocks.append(block)
    # 初始化列表以存储标题块
    title_blocks = []
    # 遍历所有边界框，提取标题块
    for block in all_bboxes:
        if block[7] == BlockType.Title:
            title_blocks.append(block)

    # 初始化列表以存储需要移除的块
    need_remove = []

    # 检查文本块与标题块的重叠
    for text_block in text_blocks:
        for title_block in title_blocks:
            text_block_bbox = text_block[:4]
            title_block_bbox = title_block[:4]
            # 计算重叠率，如果大于0.8，标记标题块为待移除
            if calculate_iou(text_block_bbox, title_block_bbox) > 0.8:
                if title_block not in need_remove:
                    need_remove.append(title_block)

    # 如果有需要移除的块，则从原始列表中移除
    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)

    # 返回处理后的边界框列表
    return all_bboxes


# 移除需要丢弃的块
def remove_need_drop_blocks(all_bboxes, discarded_blocks):
    # 初始化列表以存储需要移除的块
    need_remove = []
    # 检查所有边界框与丢弃块的重叠
    for block in all_bboxes:
        for discarded_block in discarded_blocks:
            block_bbox = block[:4]
            # 计算重叠面积比率，如果大于0.6，标记该块为待移除
            if calculate_overlap_area_in_bbox1_area_ratio(block_bbox, discarded_block['bbox']) > 0.6:
                if block not in need_remove:
                    need_remove.append(block)
                    break

    # 如果有需要移除的块，则从原始列表中移除
    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)
    # 返回处理后的边界框列表
    return all_bboxes


# 移除重叠的较小块
def remove_overlaps_min_blocks(all_bboxes):
    # 处理重叠块，较小的块不能直接删除，需要与大的块合并
    # 初始化列表以存储需要移除的块
    need_remove = []
    # 遍历所有边界框
    for block1 in all_bboxes:
        # 对每个边界框，再次遍历所有边界框
        for block2 in all_bboxes:
            # 确保不比较同一个边界框
            if block1 != block2:
                # 获取 block1 的边界框坐标
                block1_bbox = block1[:4]
                # 获取 block2 的边界框坐标
                block2_bbox = block2[:4]
                # 获取两个边界框如果重叠，且重叠比例大于 0.8 的最小外接框
                overlap_box = get_minbox_if_overlap_by_ratio(block1_bbox, block2_bbox, 0.8)
                # 如果存在重叠框
                if overlap_box is not None:
                    # 查找与重叠框匹配的需要移除的边界框
                    block_to_remove = next((block for block in all_bboxes if block[:4] == overlap_box), None)
                    # 如果找到了需要移除的边界框且不在待移除列表中
                    if block_to_remove is not None and block_to_remove not in need_remove:
                        # 确定较大的边界框
                        large_block = block1 if block1 != block_to_remove else block2
                        # 解包大边界框的坐标
                        x1, y1, x2, y2 = large_block[:4]
                        # 解包需要移除的边界框的坐标
                        sx1, sy1, sx2, sy2 = block_to_remove[:4]
                        # 更新大边界框的左上角坐标
                        x1 = min(x1, sx1)
                        y1 = min(y1, sy1)
                        # 更新大边界框的右下角坐标
                        x2 = max(x2, sx2)
                        y2 = max(y2, sy2)
                        # 将更新后的坐标赋值给大边界框
                        large_block[:4] = [x1, y1, x2, y2]
                        # 将需要移除的边界框添加到待移除列表中
                        need_remove.append(block_to_remove)

    # 如果待移除的边界框列表不为空
    if len(need_remove) > 0:
        # 遍历所有待移除的边界框
        for block in need_remove:
            # 从所有边界框中移除该边界框
            all_bboxes.remove(block)

    # 返回更新后的所有边界框
    return all_bboxes
```