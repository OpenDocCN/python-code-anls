# `.\comic-translate\modules\utils\textblock.py`

```py
from typing import List, Tuple  # 导入需要使用的类型提示工具
import numpy as np  # 导入 NumPy 库，用于处理数值计算和数组操作
import cv2  # 导入 OpenCV 库，用于图像处理
from functools import cached_property  # 导入 functools 库中的 cached_property 装饰器

class TextBlock(object):
    """
    Object that stores a block of text. Optionally stores the list of lines
    """
    def __init__(self, 
                 text_bbox: np.ndarray,
                 text_segm_points: np.ndarray = None, 
                 bubble_bbox: np.ndarray = None,
                 text_class: str = "",
                 lines: List = None,
                 texts: List[str] = None,
                 translation: str = "",
                 line_spacing = 1,
                 alignment: str = '',
                 source_lang: str = "",
                 target_lang: str = "",
                 **kwargs) -> None:
        # 初始化方法，接受各种参数来构造 TextBlock 对象
        
        self.xyxy = text_bbox  # 文本框的边界框坐标
        self.segm_pts = text_segm_points  # 文本的分割点
        self.bubble_xyxy = bubble_bbox  # 气泡框的边界框坐标
        self.text_class = text_class  # 文本类别
        
        self.lines = np.array(lines, dtype=np.int32) if lines else []  # 将行列表转换为 NumPy 数组
        self.texts = texts if texts is not None else []  # 文本列表
        self.text = ' '.join(self.texts)  # 将文本列表合并为一个字符串
        self.translation = translation  # 翻译文本
        
        self.line_spacing = line_spacing  # 行间距
        self.alignment = alignment  # 对齐方式
        
        self.source_lang = source_lang  # 源语言
        self.target_lang = target_lang  # 目标语言

    @cached_property
    def xywh(self):
        # 计算并返回文本框的 (x, y, w, h)
        x1, y1, x2, y2 = self.xyxy
        return np.array([x1, y1, x2-x1, y2-y1]).astype(np.int32)

    @cached_property
    def center(self) -> np.ndarray:
        # 计算并返回文本框的中心坐标
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2
    
    @cached_property
    def source_lang_direction(self):
        # 根据源语言返回文本方向
        if self.source_lang == 'ja':
            return 'ver_rtl'  # 如果是日语，返回垂直从右到左
        else:
            return 'hor_ltr'  # 否则返回水平从左到右

def sort_blk_list(blk_list: List[TextBlock], right_to_left=True) -> List[TextBlock]:
    # 对文本块列表进行排序，从上到下，从右到左（可选）
    sorted_blk_list = []  # 初始化排序后的文本块列表
    for blk in sorted(blk_list, key=lambda blk: blk.center[1]):
        # 按文本块中心的 y 坐标排序
        for i, sorted_blk in enumerate(sorted_blk_list):
            if blk.center[1] > sorted_blk.xyxy[3]:
                continue
            if blk.center[1] < sorted_blk.xyxy[1]:
                sorted_blk_list.insert(i + 1, blk)
                break

            # 如果文本块的中心在已排序块内，则按 x 坐标排序
            if right_to_left and blk.center[0] > sorted_blk.center[0]:
                sorted_blk_list.insert(i, blk)
                break
            if not right_to_left and blk.center[0] < sorted_blk.center[0]:
                sorted_blk_list.insert(i, blk)
                break
        else:
            sorted_blk_list.append(blk)  # 如果未插入，则追加到末尾
    return sorted_blk_list

def sort_textblock_rectangles(coords_text_list: List[Tuple[Tuple[int, int, int, int], str]], direction: str = 'ver_rtl', threshold: int = 5):
    # 对文本块矩形坐标列表进行排序
    # 判断两个词框是否在同一行或同一列
    def in_same_line(coor_a, coor_b):
        # 如果是水平文本，检查词框是否在同一水平线上
        if 'hor' in direction:
            return abs(coor_a[1] - coor_b[1]) <= threshold
        # 如果是垂直文本，检查词框是否在同一垂直线上
        elif 'ver' in direction:
            return abs(coor_a[0] - coor_b[0]) <= threshold

    # 将词框分组成行
    lines = []
    remaining_boxes = coords_text_list[:]  # 创建一个副本

    while remaining_boxes:
        box = remaining_boxes.pop(0)  # 从剩余词框中取出第一个作为起点
        current_line = [box]

        # 对比剩余的词框，将与当前词框在同一行（或列）的词框加入当前行
        boxes_to_check_against = remaining_boxes[:]
        for comparison_box in boxes_to_check_against:
            if in_same_line(box[0], comparison_box[0]):
                remaining_boxes.remove(comparison_box)
                current_line.append(comparison_box)

        lines.append(current_line)

    # 根据阅读方向对每一行内的词框进行排序
    for i, line in enumerate(lines):
        if direction == 'hor_ltr':
            lines[i] = sorted(line, key=lambda box: box[0][0])  # 按照最左端的 x 坐标排序（从左到右）
        elif direction == 'hor_rtl':
            lines[i] = sorted(line, key=lambda box: -box[0][0])  # 按照最左端的 x 坐标排序（从右到左）
        elif direction in ['ver_ltr', 'ver_rtl']:
            lines[i] = sorted(line, key=lambda box: box[0][1])  # 按照最顶端的 y 坐标排序

    # 根据文本方向对行进行排序
    if 'hor' in direction:
        lines.sort(key=lambda line: min(box[0][1] for box in line))  # 对于水平文本，按照每行最顶端的 y 坐标排序
    elif direction == 'ver_ltr':
        lines.sort(key=lambda line: min(box[0][0] for box in line))  # 对于垂直从左到右文本，按照每行最左端的 x 坐标排序
    elif direction == 'ver_rtl':
        lines.sort(key=lambda line: min(box[0][0] for box in line), reverse=True)  # 对于垂直从右到左文本，按照每行最左端的 x 坐标排序，逆序

    # 展开行列表，返回一个包含所有分组词框的单一列表
    grouped_boxes = [box for line in lines for box in line]
    
    return grouped_boxes
# 可视化文本块在画布上的显示
def visualize_textblocks(canvas, blk_list: List[TextBlock]):
    # 计算线条宽度，基于画布尺寸的一定比例
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width

    # 遍历文本块列表
    for i, blk in enumerate(blk_list):
        # 提取文本块的边界框坐标
        bx1, by1, bx2, by2 = blk.xyxy

        # 绘制边界框
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)

        # 遍历文本块中的每一行
        for j, line in enumerate(blk.lines):
            # 在画布上标注行号
            cv2.putText(canvas, str(j), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 127, 0), 1)
            # 绘制连接文本块中各行的多边形
            cv2.polylines(canvas, [line], True, (0, 127, 255), 2)

        # 在文本块边界框左上角附近标注文本块序号
        cv2.putText(canvas, str(i), (bx1, by1 + lw), 0, lw / 3, (255, 127, 127), max(lw - 1, 1), cv2.LINE_AA)

    return canvas

# 可视化语音气泡在画布上的显示
def visualize_speech_bubbles(canvas, blk_list: List[TextBlock]):
    # 计算线条宽度，基于画布尺寸的一定比例
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width

    # 定义每个类别对应的颜色
    class_colors = {
        'text_free': (255, 0, 0),    # 类别 text_free 的颜色为蓝色
        'text_bubble': (0, 255, 0),  # 类别 text_bubble 的颜色为绿色
    }

    # 遍历语音气泡块列表
    for blk in blk_list:
        # 提取语音气泡的边界框坐标
        bx1, by1, bx2, by2 = blk.bubble_xyxy

        # 根据类别选择颜色，如果类别未定义则使用默认颜色
        color = class_colors.get(blk.text_class, (127, 255, 127))

        # 绘制边界框
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, lw)

    return canvas

# 调整文本行坐标
def adjust_text_line_coordinates(coords, width_expansion_percentage: int, height_expansion_percentage: int):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = coords

    # 计算当前坐标框的宽度和高度
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y

    # 计算宽度和高度的扩展偏移量
    width_expansion_offset = int(((width * width_expansion_percentage) / 100) / 2)
    height_expansion_offset = int(((height * height_expansion_percentage) / 100) / 2)

    # 计算扩展后的矩形起点坐标（左上角）
    pt1_expanded = (
        top_left_x - width_expansion_offset,
        top_left_y - height_expansion_offset,
    )
    # 根据传入的参数计算出第二个点的扩展位置
    pt2_expanded = (
        bottom_right_x + width_expansion_offset,  # 计算右下角点的新 x 坐标
        bottom_right_y + height_expansion_offset,  # 计算右下角点的新 y 坐标
    )
    
    # 返回扩展后的第一个点和第二个点的坐标信息
    return pt1_expanded[0], pt1_expanded[1], pt2_expanded[0], pt2_expanded[1]
def adjust_blks_size(blk_list: List[TextBlock], img_shape: Tuple[int, int, int], w_expan: int = 0, h_expan: int = 0):
    # 提取图像的高度和宽度
    im_h, im_w = img_shape[:2]
    # 遍历文本块列表
    for blk in blk_list:
        # 获取文本块的坐标信息
        coords = blk.xyxy
        # 根据指定的扩展参数调整文本行坐标
        expanded_coords = adjust_text_line_coordinates(coords, w_expan, h_expan)

        # 确保文本框不超出图像边界
        new_x1 = max(expanded_coords[0], 0)  # 新的左上角 x 坐标
        new_y1 = max(expanded_coords[1], 0)  # 新的左上角 y 坐标
        new_x2 = min(expanded_coords[2], im_w)  # 新的右下角 x 坐标
        new_y2 = min(expanded_coords[3], im_h)  # 新的右下角 y 坐标

        # 更新文本块的坐标信息
        blk.xyxy[:] = [new_x1, new_y1, new_x2, new_y2]
```