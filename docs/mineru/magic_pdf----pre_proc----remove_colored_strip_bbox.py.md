# `.\MinerU\magic_pdf\pre_proc\remove_colored_strip_bbox.py`

```
# 从magic_pdf.libs.boxbase模块导入必要的函数
from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap, calculate_overlap_area_2_minbox_area_ratio
# 从loguru模块导入logger，用于日志记录
from loguru import logger

# 从magic_pdf.libs.drop_tag模块导入COLOR_BG_HEADER_TXT_BLOCK常量
from magic_pdf.libs.drop_tag import COLOR_BG_HEADER_TXT_BLOCK


# 计算给定矩形的面积
def __area(box):
    # 返回矩形的宽度乘以高度
    return (box[2] - box[0]) * (box[3] - box[1])


# 判断矩形是否在页面中轴线附近
def rectangle_position_determination(rect, p_width):
    """
    判断矩形是否在页面中轴线附近。

    Args:
        rect (list): 矩形坐标，格式为[x1, y1, x2, y2]。
        p_width (int): 页面宽度。

    Returns:
        bool: 若矩形在页面中轴线附近则返回True，否则返回False。
    """
    # 计算页面中轴线的x坐标
    x_axis = p_width / 2
    # 判断矩形是否跨越中轴线
    is_span = rect[0] < x_axis and rect[2] > x_axis
    # 如果矩形跨越中轴线，返回True
    if is_span:
        return True
    else:
        # 计算矩形与中轴线的距离，只考虑近的一侧
        distance = rect[0] - x_axis if rect[0] > x_axis else x_axis - rect[2]
        # 判断矩形与中轴线的距离是否小于页面宽度的20%
        if distance < p_width * 0.2:
            return True  # 若距离小于20%，返回True
        else:
            return False  # 否则返回False

# 根据页面中特定颜色和大小过滤文本块
def remove_colored_strip_textblock(remain_text_blocks, page):
    """
    根据页面中特定颜色和大小过滤文本块，将符合条件的文本块从remain_text_blocks中移除，并返回移除的文本块列表colored_strip_textblock。

    Args:
        remain_text_blocks (list): 剩余文本块列表。
        page (Page): 页面对象。

    Returns:
        tuple: 剩余文本块列表和移除的文本块列表。
    """
    # 创建一个空列表，用于存放移除的文本块
    colored_strip_textblocks = []  # 先构造一个空的返回
    # 检查剩余文本块是否存在
        if len(remain_text_blocks) > 0:
            # 获取页面的宽度和高度
            p_width, p_height = page.rect.width, page.rect.height
            # 获取页面中的所有绘制块
            blocks = page.get_cdrawings()
            # 初始化用于存储彩色条背景矩形的列表
            colored_strip_bg_rect = []
            # 遍历每个绘制块
            for block in blocks:
                # 检查块是否填充且填充颜色不是白色
                is_filled = 'fill' in block and block['fill'] and block['fill'] != (1.0, 1.0, 1.0)  # 过滤掉透明的
                # 获取块的矩形坐标
                rect = block['rect']
                # 检查矩形区域是否足够大
                area_is_large_enough = __area(rect) > 100  # 过滤掉特别小的矩形
                # 确定矩形的位置
                rectangle_position_determination_result = rectangle_position_determination(rect, p_width)
                # 检查矩形是否位于页面上半部分
                in_upper_half_page = rect[3] < p_height * 0.3  # 找到位于页面上半部分的矩形，下边界小于页面高度的30%
                # 检查矩形的长宽比是否超过4
                aspect_ratio_exceeds_4 = (rect[2] - rect[0]) > (rect[3] - rect[1]) * 4  # 找到长宽比超过4的矩形
    
                # 如果所有条件都满足，将矩形添加到列表中
                if is_filled and area_is_large_enough and rectangle_position_determination_result and in_upper_half_page and aspect_ratio_exceeds_4:
                    colored_strip_bg_rect.append(rect)
    
            # 如果找到任何彩色条背景矩形
            if len(colored_strip_bg_rect) > 0:
                # 遍历每个彩色条背景矩形
                for colored_strip_block_bbox in colored_strip_bg_rect:
                    # 遍历剩余文本块
                    for text_block in remain_text_blocks:
                        # 获取文本块的边界框
                        text_bbox = text_block['bbox']
                        # 检查文本块是否在彩色条矩形内或部分重叠且重叠比率大于0.6
                        if _is_in(text_bbox, colored_strip_block_bbox) or (_is_in_or_part_overlap(text_bbox, colored_strip_block_bbox) and calculate_overlap_area_2_minbox_area_ratio(text_bbox, colored_strip_block_bbox) > 0.6):
                            # 记录移除的文本块信息
                            logger.info(f'remove_colored_strip_textblock: {text_bbox}, {colored_strip_block_bbox}')
                            # 标记文本块为彩色背景标题文本块
                            text_block['tag'] = COLOR_BG_HEADER_TXT_BLOCK
                            # 添加到彩色条文本块列表
                            colored_strip_textblocks.append(text_block)
    
                    # 如果找到了彩色条文本块
                    if len(colored_strip_textblocks) > 0:
                        # 遍历并从剩余文本块中移除彩色条文本块
                        for colored_strip_textblock in colored_strip_textblocks:
                            if colored_strip_textblock in remain_text_blocks:
                                remain_text_blocks.remove(colored_strip_textblock)
    
        # 返回剩余文本块和彩色条文本块
        return remain_text_blocks, colored_strip_textblocks
```