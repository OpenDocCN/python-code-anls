# `.\MinerU\magic_pdf\pre_proc\pdf_pre_filter.py`

```
# 从magic_pdf库中导入fitz模块，用于处理PDF页面
from magic_pdf.libs.commons import fitz
# 导入用于矩形包含关系检查的函数
from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap
# 导入用于处理丢弃原因的类
from magic_pdf.libs.drop_reason import DropReason


# 定义一个计算矩形面积的函数
def __area(box):
    # 计算并返回矩形的面积
    return (box[2] - box[0]) * (box[3] - box[1])

# 定义检查页面是否包含有颜色背景矩形的函数
def __is_contain_color_background_rect(page:fitz.Page, text_blocks, image_bboxes) -> bool:
    """
    检查page是包含有颜色背景的矩形
    """
    # 存储找到的颜色背景矩形
    color_bg_rect = []
    # 获取页面的宽度和高度
    p_width, p_height = page.rect.width, page.rect.height
    
    # 获取页面上的所有绘图块
    blocks = page.get_cdrawings()
    for block in blocks:
        
        # 过滤掉透明填充的块
        if 'fill' in block and block['fill']: # 过滤掉透明的
            fill = list(block['fill'])
            # 将填充颜色转换为整数
            fill[0], fill[1], fill[2] = int(fill[0]), int(fill[1]), int(fill[2])
            # 跳过白色背景
            if fill==(1.0,1.0,1.0):
                continue
            rect = block['rect']
            # 过滤掉面积小于100的矩形
            if __area(rect) < 10*10:
                continue
            # 过滤掉与图片框重叠的矩形
            if any([_is_in_or_part_overlap(rect, img_bbox) for img_bbox in image_bboxes]):
                continue
            # 将符合条件的矩形加入列表
            color_bg_rect.append(rect)
            
    # 如果找到颜色背景矩形
    if len(color_bg_rect) > 0:
        # 找到面积最大的背景矩形
        max_rect = max(color_bg_rect, key=lambda x:__area(x))
        # 将最大的矩形转换为整数
        max_rect_int = (int(max_rect[0]), int(max_rect[1]), int(max_rect[2]), int(max_rect[3]))
        # 检查该矩形的宽度和高度是否符合条件
        if max_rect[2]-max_rect[0] > 0.2*p_width and  max_rect[3]-max_rect[1] > 0.1*p_height:#宽度符合
            # 检查文本块是否落在这个矩形中
            for text_block in text_blocks:
                box = text_block['bbox']
                box_int = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                # 如果文本块在矩形内，返回True
                if _is_in(box_int, max_rect_int):
                    return True
    
    # 如果没有符合条件的矩形，返回False
    return False


# 定义检查表格与文本块重叠的函数
def __is_table_overlap_text_block(text_blocks, table_bbox):
    """
    检查table_bbox是否覆盖了text_blocks里的文本块
    TODO
    """
    # 遍历所有文本块
    for text_block in text_blocks:
        box = text_block['bbox']
        # 如果表格与文本块重叠，返回True
        if _is_in_or_part_overlap(table_bbox, box):
            return True
    # 如果没有重叠，返回False
    return False


# 定义PDF过滤函数，检查PDF是否符合要求
def pdf_filter(page:fitz.Page, text_blocks, table_bboxes, image_bboxes) -> tuple:
    """
    return:(True|False, err_msg)
        True, 如果pdf符合要求
        False, 如果pdf不符合要求
        
    """
    # 如果页面包含颜色背景矩形，则返回False和丢弃原因
    if __is_contain_color_background_rect(page, text_blocks, image_bboxes):
        return False, {"_need_drop": True, "_drop_reason": DropReason.COLOR_BACKGROUND_TEXT_BOX}

    # 如果符合要求，返回True和None
    return True, None
```