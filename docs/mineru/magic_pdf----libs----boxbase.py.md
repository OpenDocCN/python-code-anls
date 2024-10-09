# `.\MinerU\magic_pdf\libs\boxbase.py`

```
import math  # 导入数学模块以便使用数学函数


def _is_in_or_part_overlap(box1, box2) -> bool:
    """两个bbox是否有部分重叠或者包含."""
    if box1 is None or box2 is None:  # 检查任一边界框是否为 None
        return False  # 如果是，则返回 False

    x0_1, y0_1, x1_1, y1_1 = box1  # 解包 box1 的四个坐标
    x0_2, y0_2, x1_2, y1_2 = box2  # 解包 box2 的四个坐标

    return not (x1_1 < x0_2 or  # 如果 box1 的右边界小于 box2 的左边界
                x0_1 > x1_2 or  # 或者 box1 的左边界大于 box2 的右边界
                y1_1 < y0_2 or  # 或者 box1 的下边界小于 box2 的上边界
                y0_1 > y1_2)  # 或者 box1 的上边界大于 box2 的下边界


def _is_in_or_part_overlap_with_area_ratio(box1,
                                           box2,
                                           area_ratio_threshold=0.6):
    """判断box1是否在box2里面，或者box1和box2有部分重叠，且重叠面积占box1的比例超过area_ratio_threshold."""
    if box1 is None or box2 is None:  # 检查任一边界框是否为 None
        return False  # 如果是，则返回 False

    x0_1, y0_1, x1_1, y1_1 = box1  # 解包 box1 的四个坐标
    x0_2, y0_2, x1_2, y1_2 = box2  # 解包 box2 的四个坐标

    if not _is_in_or_part_overlap(box1, box2):  # 如果 box1 和 box2 没有重叠或包含
        return False  # 则返回 False

    # 计算重叠面积
    x_left = max(x0_1, x0_2)  # 重叠区域的左边界
    y_top = max(y0_1, y0_2)  # 重叠区域的上边界
    x_right = min(x1_1, x1_2)  # 重叠区域的右边界
    y_bottom = min(y1_1, y1_2)  # 重叠区域的下边界
    overlap_area = (x_right - x_left) * (y_bottom - y_top)  # 计算重叠区域的面积

    # 计算box1的面积
    box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)  # box1 的面积

    return overlap_area / box1_area > area_ratio_threshold  # 返回重叠面积占 box1 面积的比例是否大于阈值


def _is_in(box1, box2) -> bool:
    """box1是否完全在box2里面."""
    x0_1, y0_1, x1_1, y1_1 = box1  # 解包 box1 的四个坐标
    x0_2, y0_2, x1_2, y1_2 = box2  # 解包 box2 的四个坐标

    return (x0_1 >= x0_2 and  # box1 的左边界不在 box2 的左边外
            y0_1 >= y0_2 and  # box1 的上边界不在 box2 的上边外
            x1_1 <= x1_2 and  # box1 的右边界不在 box2 的右边外
            y1_1 <= y1_2)  # box1 的下边界不在 box2 的下边外


def _is_part_overlap(box1, box2) -> bool:
    """两个bbox是否有部分重叠，但不完全包含."""
    if box1 is None or box2 is None:  # 检查任一边界框是否为 None
        return False  # 如果是，则返回 False

    return _is_in_or_part_overlap(box1, box2) and not _is_in(box1, box2)  # 返回部分重叠且不完全包含的结果


def _left_intersect(left_box, right_box):
    """检查两个box的左边界是否有交集，也就是left_box的右边界是否在right_box的左边界内."""
    if left_box is None or right_box is None:  # 检查任一边界框是否为 None
        return False  # 如果是，则返回 False

    x0_1, y0_1, x1_1, y1_1 = left_box  # 解包左边框的四个坐标
    x0_2, y0_2, x1_2, y1_2 = right_box  # 解包右边框的四个坐标

    return x1_1 > x0_2 and x0_1 < x0_2 and (y0_1 <= y0_2 <= y1_1  # 检查左边框右边界是否在右边框左边界内且 y 方向的交集
                                            or y0_1 <= y1_2 <= y1_1)  # 或者右边框的下边界是否在左边框的 y 范围内


def _right_intersect(left_box, right_box):
    """检查box是否在右侧边界有交集，也就是left_box的左边界是否在right_box的右边界内."""
    if left_box is None or right_box is None:  # 检查任一边界框是否为 None
        return False  # 如果是，则返回 False

    x0_1, y0_1, x1_1, y1_1 = left_box  # 解包左边框的四个坐标
    x0_2, y0_2, x1_2, y1_2 = right_box  # 解包右边框的四个坐标

    return x0_1 < x1_2 and x1_1 > x1_2 and (y0_1 <= y0_2 <= y1_1  # 检查左边框左边界是否在右边框右边界内且 y 方向的交集
                                            or y0_1 <= y1_2 <= y1_1)  # 或者右边框的下边界是否在左边框的 y 范围内


def _is_vertical_full_overlap(box1, box2, x_torlence=2):
    """x方向上：要么box1包含box2, 要么box2包含box1。不能部分包含 y方向上：box1和box2有重叠."""
    # 解析box的坐标
    x11, y11, x12, y12 = box1  # 左上角和右下角的坐标 (x1, y1, x2, y2)
    x21, y21, x22, y22 = box2  # 解包 box2 的四个坐标

    # 在x轴方向上，box1是否包含box2 或 box2包含box1
    contains_in_x = (x11 - x_torlence <= x21 and x12 + x_torlence >= x22) or (  # 检查 box1 是否包含 box2
        x21 - x_torlence <= x11 and x22 + x_torlence >= x12)  # 或者 box2 是否包含 box1

    # 在y轴方向上，box1和box2是否有重叠
    overlap_in_y = not (y12 < y21 or y11 > y22)  # 检查在 y 方向上是否有重叠
    # 返回两个条件的逻辑与结果：contains_in_x 和 overlap_in_y
        return contains_in_x and overlap_in_y
# 检查 box1 下方和 box2 上方是否存在轻微的重叠
def _is_bottom_full_overlap(box1, box2, y_tolerance=2):
    # 如果 box1 或 box2 为 None，返回 False
    if box1 is None or box2 is None:
        return False

    # 解构 box1 和 box2 的坐标
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    # 定义容忍边距
    tolerance_margin = 2
    # 判断 x 方向上是否存在完全重叠
    is_xdir_full_overlap = (
        (x0_1 - tolerance_margin <= x0_2 <= x1_1 + tolerance_margin
         and x0_1 - tolerance_margin <= x1_2 <= x1_1 + tolerance_margin)
        or (x0_2 - tolerance_margin <= x0_1 <= x1_2 + tolerance_margin
            and x0_2 - tolerance_margin <= x1_1 <= x1_2 + tolerance_margin))

    # 返回 y 方向重叠情况及 x 方向重叠情况
    return y0_2 < y1_1 and 0 < (y1_1 -
                                y0_2) < y_tolerance and is_xdir_full_overlap


# 检查 box1 的左侧是否与 box2 重叠
def _is_left_overlap(
    box1,
    box2,
):
    # 定义一个内部函数检查 y 轴上的重叠
    def __overlap_y(Ay1, Ay2, By1, By2):
        return max(0, min(Ay2, By2) - max(Ay1, By1))

    # 如果 box1 或 box2 为 None，返回 False
    if box1 is None or box2 is None:
        return False

    # 解构 box1 和 box2 的坐标
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    # 计算 y 方向的重叠长度
    y_overlap_len = __overlap_y(y0_1, y1_1, y0_2, y1_2)
    # 计算 box1 和 box2 的重叠比例
    ratio_1 = 1.0 * y_overlap_len / (y1_1 - y0_1) if y1_1 - y0_1 != 0 else 0
    ratio_2 = 1.0 * y_overlap_len / (y1_2 - y0_2) if y1_2 - y0_2 != 0 else 0
    # 判断是否满足垂直重叠条件
    vertical_overlap_cond = ratio_1 >= 0.5 or ratio_2 >= 0.5

    # 返回 x 方向重叠情况及垂直重叠条件
    return x0_1 <= x0_2 <= x1_1 and vertical_overlap_cond


# 检查两个边界框在 y 轴上的重叠是否超过一定阈值
def __is_overlaps_y_exceeds_threshold(bbox1,
                                      bbox2,
                                      overlap_ratio_threshold=0.8):
    # 解构 bbox1 和 bbox2 的 y 坐标
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2

    # 计算重叠长度
    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    # 计算高度
    height1, height2 = y1_1 - y0_1, y1_2 - y0_2
    # 获取较小的高度
    min_height = min(height1, height2)

    # 返回重叠高度占较小高度的比例是否超过阈值
    return (overlap / min_height) > overlap_ratio_threshold


# 计算两个边界框的交并比 (IOU)
def calculate_iou(bbox1, bbox2):
    """计算两个边界框的交并比(IOU)。

    Args:
        bbox1 (list[float]): 第一个边界框的坐标，格式为 [x1, y1, x2, y2]，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (list[float]): 第二个边界框的坐标，格式与 `bbox1` 相同。

    Returns:
        float: 两个边界框的交并比(IOU)，取值范围为 [0, 1]。
    """
    # 确定交集矩形的坐标
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # 如果没有重叠，返回 0.0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算重叠区域的面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个边界框的面积
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # 计算交并比（IoU），通过获取交集区域
    # 然后将其面积除以两个区域面积之和减去交集面积
    iou = intersection_area / float(bbox1_area + bbox2_area -
                                    intersection_area)
    # 返回计算得到的交并比
    return iou
# 计算 box1 和 box2 的重叠面积占最小面积的比例
def calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2):
    # 确定重叠矩形的左、上、右、下坐标
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # 如果没有重叠，返回 0.0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算重叠面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # 计算两个 box 的最小面积
    min_box_area = min([(bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]),
                        (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])])
    # 如果最小面积为 0，返回 0
    if min_box_area == 0:
        return 0
    else:
        # 返回重叠面积占最小面积的比例
        return intersection_area / min_box_area


# 计算重叠面积占 bbox1 的比例
def calculate_overlap_area_in_bbox1_area_ratio(bbox1, bbox2):
    # 确定重叠矩形的左、上、右、下坐标
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # 如果没有重叠，返回 0.0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算重叠面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # 计算 bbox1 的面积
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    # 如果 bbox1 的面积为 0，返回 0
    if bbox1_area == 0:
        return 0
    else:
        # 返回重叠面积占 bbox1 面积的比例
        return intersection_area / bbox1_area


# 通过重叠面积比例判断返回较小的 bbox
def get_minbox_if_overlap_by_ratio(bbox1, bbox2, ratio):
    # 解包两个 bbox 的坐标
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    # 计算两个 bbox 的面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    # 计算重叠面积占最小面积的比例
    overlap_ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
    # 如果重叠比例大于给定比例
    if overlap_ratio > ratio:
        # 返回较小的 bbox
        if area1 <= area2:
            return bbox1
        else:
            return bbox2
    else:
        # 否则返回 None
        return None


# 从给定边界内筛选 bbox
def get_bbox_in_boundary(bboxes: list, boundary: tuple) -> list:
    # 解包边界坐标
    x0, y0, x1, y1 = boundary
    # 筛选在边界内的 bbox
    new_boxes = [
        box for box in bboxes
        if box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1
    ]
    # 返回筛选结果
    return new_boxes


# 判断一个 bbox 是否在 PDF 页面边缘
def is_vbox_on_side(bbox, width, height, side_threshold=0.2):
    # 解包 bbox 的左、右坐标
    x0, x1 = bbox[0], bbox[2]
    # 判断 bbox 是否在边缘
    if x1 <= width * side_threshold or x0 >= width * (1 - side_threshold):
        return True
    # 如果不在边缘返回 False
    return False


# 找到与给定 bbox 最近的上方文本 bbox
def find_top_nearest_text_bbox(pymu_blocks, obj_bbox):
    # 设置容差边距
    tolerance_margin = 4
    # 筛选出距离 obj_bbox 上方的文本 bbox
    top_boxes = [
        box for box in pymu_blocks
        if obj_bbox[1] - box['bbox'][3] >= -tolerance_margin
        and not _is_in(box['bbox'], obj_bbox)
    ]
    # 然后找到 X 方向上有互相重叠的
    # 过滤 top_boxes 列表，保留与 obj_bbox 有交集的盒子
        top_boxes = [
            box for box in top_boxes if any([
                # 检查 box 的左边是否在 obj_bbox 的范围内，加上容差
                obj_bbox[0] - tolerance_margin <= box['bbox'][0] <= obj_bbox[2] +
                tolerance_margin, 
                # 检查 box 的右边是否在 obj_bbox 的范围内，加上容差
                obj_bbox[0] - tolerance_margin <= box['bbox'][2] <= obj_bbox[2] +
                tolerance_margin, 
                # 检查 obj_bbox 的左边是否在 box 的范围内，加上容差
                box['bbox'][0] - tolerance_margin <= obj_bbox[0] <= box['bbox'][2] +
                tolerance_margin, 
                # 检查 obj_bbox 的右边是否在 box 的范围内，加上容差
                box['bbox'][0] - tolerance_margin <= obj_bbox[2] <= box['bbox'][2] +
                tolerance_margin
            ])
        ]
    
        # 如果过滤后的 top_boxes 非空，找出 y1 最大的盒子
        if len(top_boxes) > 0:
            # 根据 y1 值（bbox[3]）对 top_boxes 进行排序，降序
            top_boxes.sort(key=lambda x: x['bbox'][3], reverse=True)
            # 返回 y1 最大的盒子
            return top_boxes[0]
        else:
            # 如果没有符合条件的盒子，返回 None
            return None
# 查找距离给定边界框最近的底部文本框
def find_bottom_nearest_text_bbox(pymu_blocks, obj_bbox):
    # 筛选出在给定边界框下面的文本框，且不在给定边界框内
    bottom_boxes = [
        box for box in pymu_blocks if box['bbox'][1] -
        obj_bbox[3] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # 找到X方向上有互相重叠的文本框
    bottom_boxes = [
        box for box in bottom_boxes if any([
            obj_bbox[0] - 2 <= box['bbox'][0] <= obj_bbox[2] + 2, obj_bbox[0] -
            2 <= box['bbox'][2] <= obj_bbox[2] + 2, box['bbox'][0] -
            2 <= obj_bbox[0] <= box['bbox'][2] + 2, box['bbox'][0] -
            2 <= obj_bbox[2] <= box['bbox'][2] + 2
        ])
    ]

    # 如果存在满足条件的底部文本框
    if len(bottom_boxes) > 0:
        # 根据y坐标对底部文本框排序，找到y0最小的那个
        bottom_boxes.sort(key=lambda x: x['bbox'][1], reverse=False)
        # 返回距离最近的底部文本框
        return bottom_boxes[0]
    else:
        # 如果没有找到，返回None
        return None


# 查找距离给定边界框最近的左侧文本框
def find_left_nearest_text_bbox(pymu_blocks, obj_bbox):
    """寻找左侧最近的文本block."""
    # 筛选出在给定边界框左侧的文本框，且不在给定边界框内
    left_boxes = [
        box for box in pymu_blocks if obj_bbox[0] -
        box['bbox'][2] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # 找到Y方向上有互相重叠的文本框
    left_boxes = [
        box for box in left_boxes if any([
            obj_bbox[1] - 2 <= box['bbox'][1] <= obj_bbox[3] + 2, obj_bbox[1] -
            2 <= box['bbox'][3] <= obj_bbox[3] + 2, box['bbox'][1] -
            2 <= obj_bbox[1] <= box['bbox'][3] + 2, box['bbox'][1] -
            2 <= obj_bbox[3] <= box['bbox'][3] + 2
        ])
    ]

    # 如果存在满足条件的左侧文本框
    if len(left_boxes) > 0:
        # 根据x坐标对左侧文本框排序，找到x1最大的那个
        left_boxes.sort(key=lambda x: x['bbox'][2], reverse=True)
        # 返回距离最近的左侧文本框
        return left_boxes[0]
    else:
        # 如果没有找到，返回None
        return None


# 查找距离给定边界框最近的右侧文本框
def find_right_nearest_text_bbox(pymu_blocks, obj_bbox):
    """寻找右侧最近的文本block."""
    # 筛选出在给定边界框右侧的文本框，且不在给定边界框内
    right_boxes = [
        box for box in pymu_blocks if box['bbox'][0] -
        obj_bbox[2] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # 找到Y方向上有互相重叠的文本框
    right_boxes = [
        box for box in right_boxes if any([
            obj_bbox[1] - 2 <= box['bbox'][1] <= obj_bbox[3] + 2, obj_bbox[1] -
            2 <= box['bbox'][3] <= obj_bbox[3] + 2, box['bbox'][1] -
            2 <= obj_bbox[1] <= box['bbox'][3] + 2, box['bbox'][1] -
            2 <= obj_bbox[3] <= box['bbox'][3] + 2
        ])
    ]

    # 如果存在满足条件的右侧文本框
    if len(right_boxes) > 0:
        # 根据x坐标对右侧文本框排序，找到x0最小的那个
        right_boxes.sort(key=lambda x: x['bbox'][0], reverse=False)
        # 返回距离最近的右侧文本框
        return right_boxes[0]
    else:
        # 如果没有找到，返回None
        return None


# 判断两个矩形框的相对位置关系
def bbox_relative_pos(bbox1, bbox2):
    """判断两个矩形框的相对位置关系.

    Args:
        bbox1: 一个四元组，表示第一个矩形框的左上角和右下角的坐标，格式为(x1, y1, x1b, y1b)
        bbox2: 一个四元组，表示第二个矩形框的左上角和右下角的坐标，格式为(x2, y2, x2b, y2b)

    Returns:
        一个四元组，表示矩形框1相对于矩形框2的位置关系，格式为(left, right, bottom, top)
        其中，left表示矩形框1是否在矩形框2的左侧，right表示矩形框1是否在矩形框2的右侧，
        bottom表示矩形框1是否在矩形框2的下方，top表示矩形框1是否在矩形框2的上方
    """
    # 解包边界框坐标
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    # 判断bbox1是否在bbox2的左侧
    left = x2b < x1
    # 判断bbox1是否在bbox2的右侧
    right = x1b < x2
    # 判断bbox1是否在bbox2的下方
    bottom = y2b < y1
    # 判断bbox1是否在bbox2的上方
    top = y1b < y2
    # 返回相对位置关系的四元组
    return left, right, bottom, top


# 计算两个矩形框的距离
def bbox_distance(bbox1, bbox2):
    """计算两个矩形框的距离。
    # 定义函数的参数，bbox1和bbox2分别为两个矩形框的坐标
    Args:
        bbox1 (tuple): 第一个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (tuple): 第二个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。

    # 定义返回值的类型，表示两个矩形框之间的距离
    Returns:
        float: 矩形框之间的距离。
    """

    # 定义计算两点之间距离的函数
    def dist(point1, point2):
        # 使用勾股定理计算并返回两点之间的欧几里得距离
        return math.sqrt((point1[0] - point2[0])**2 +
                         (point1[1] - point2[1])**2)

    # 解包第一个矩形框的坐标
    x1, y1, x1b, y1b = bbox1
    # 解包第二个矩形框的坐标
    x2, y2, x2b, y2b = bbox2

    # 调用函数获取两个矩形框的相对位置，返回左、右、下、上的布尔值
    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)

    # 检查矩形框的相对位置，并计算相应的距离
    if top and left:
        # 当第一个框在第二个框的左上方时，计算其右下角与第二个框左上角的距离
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        # 当第一个框在第二个框的左下方时，计算其左上角与第二个框右下角的距离
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        # 当第一个框在第二个框的右下方时，计算其右上角与第二个框左下角的距离
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        # 当第一个框在第二个框的右上方时，计算其左下角与第二个框右上角的距离
        return dist((x1b, y1b), (x2, y2))
    elif left:
        # 当第一个框在第二个框的左侧时，返回它们之间的水平距离
        return x1 - x2b
    elif right:
        # 当第一个框在第二个框的右侧时，返回它们之间的水平距离
        return x2 - x1b
    elif bottom:
        # 当第一个框在第二个框的下方时，返回它们之间的垂直距离
        return y1 - y2b
    elif top:
        # 当第一个框在第二个框的上方时，返回它们之间的垂直距离
        return y2 - y1b
    # 如果矩形框重叠，返回距离为0
    return 0.0
# 定义一个计算边界框面积的函数
def box_area(bbox):
    # 计算并返回边界框的宽度乘以高度，即面积
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


# 定义一个计算两个边界框重叠面积比例的函数
def get_overlap_area(bbox1, bbox2):
    """计算box1和box2的重叠面积占bbox1的比例."""
    # 确定交集矩形的左、上、右、下坐标
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # 如果没有重叠，返回0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算并返回重叠面积
    return (x_right - x_left) * (y_bottom - y_top)
```