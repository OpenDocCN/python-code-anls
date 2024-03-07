# `.\marker\marker\bbox.py`

```py
import fitz as pymupdf

# 判断两个矩形框是否应该合并
def should_merge_blocks(box1, box2, tol=5):
    # 在 tol y 像素内，并且在右侧在 tol 像素内
    merge = [
        box2[0] > box1[0], # 在 x 坐标上在后面
        abs(box2[1] - box1[1]) < tol, # 在 y 坐标上在 tol 像素内
        abs(box2[3] - box1[3]) < tol, # 在 y 坐标上在 tol 像素内
        abs(box2[0] - box1[2]) < tol, # 在 x 坐标上在 tol 像素内
    ]
    return all(merge)

# 合并两个矩形框
def merge_boxes(box1, box2):
    return (min(box1[0], box2[0]), min(box1[1], box2[1]), max(box2[2], box1[2]), max(box1[3], box2[3]))

# 判断两个矩形框是否相交
def boxes_intersect(box1, box2):
    # 矩形框1与矩形框2相交
    return box1[0] < box2[2] and box1[2] > box2[0] and box1[1] < box2[3] and box1[3] > box2[1]

# 判断两个矩形框的相交面积占比是否大于给定百分比
def boxes_intersect_pct(box1, box2, pct=.9):
    # 确定相交矩形的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 两个轴对齐边界框的交集始终是一个轴对齐边界框
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个边界框的面积
    bb1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    bb2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou > pct

# 判断一个矩形框是否与多个矩形框相交
def multiple_boxes_intersect(box1, boxes):
    for box2 in boxes:
        if boxes_intersect(box1, box2):
            return True
    return False

# 判断一个矩形框是否包含在另一个矩形框内
def box_contained(box1, box2):
    # 矩形框1在矩形框2内部
    return box1[0] > box2[0] and box1[1] > box2[1] and box1[2] < box2[2] and box1[3] < box2[3]

# 将归一化的矩形框坐标还原为原始坐标
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

# 修正矩形框的旋转
def correct_rotation(bbox, page):
    #bbox base is (x0, y0, x1, y1)
    # 获取页面的旋转角度
    rotation = page.rotation
    # 如果旋转角度为0，则直接返回原始边界框
    if rotation == 0:
        return bbox

    # 计算旋转后的左上角和右下角坐标
    tl = pymupdf.Point(bbox[0], bbox[1]) * page.rotation_matrix
    br = pymupdf.Point(bbox[2], bbox[3]) * page.rotation_matrix

    # 根据不同的旋转角度进行边界框的调整
    if rotation == 90:
        bbox = [br[0], tl[1], tl[0], br[1]]
    elif rotation == 180:
        bbox = [br[0], br[1], tl[0], tl[1]]
    elif rotation == 270:
        bbox = [tl[0], br[1], br[0], tl[1]]

    # 返回调整后的边界框
    return bbox
```