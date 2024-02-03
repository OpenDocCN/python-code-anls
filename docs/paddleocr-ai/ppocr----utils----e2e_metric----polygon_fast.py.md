# `.\PaddleOCR\ppocr\utils\e2e_metric\polygon_fast.py`

```
# 导入所需的库
import numpy as np
from shapely.geometry import Polygon

# 计算多边形的面积
def area(x, y):
    # 通过顶点坐标生成多边形对象
    polygon = Polygon(np.stack([x, y], axis=1))
    # 返回多边形的面积
    return float(polygon.area)

# 计算两个多边形的近似交集面积
def approx_area_of_intersection(det_x, det_y, gt_x, gt_y):
    """
    This helper determine if both polygons are intersecting with each others with an approximation method.
    Area of intersection represented by the minimum bounding rectangular [xmin, ymin, xmax, ymax]
    """
    # 计算检测框和真实框的最大最小坐标
    det_ymax = np.max(det_y)
    det_xmax = np.max(det_x)
    det_ymin = np.min(det_y)
    det_xmin = np.min(det_x)

    gt_ymax = np.max(gt_y)
    gt_xmax = np.max(gt_x)
    gt_ymin = np.min(gt_y)
    gt_xmin = np.min(gt_x)

    # 计算交集的高度
    all_min_ymax = np.minimum(det_ymax, gt_ymax)
    all_max_ymin = np.maximum(det_ymin, gt_ymin)
    intersect_heights = np.maximum(0.0, (all_min_ymax - all_max_ymin))

    # 计算交集的宽度
    all_min_xmax = np.minimum(det_xmax, gt_xmax)
    all_max_xmin = np.maximum(det_xmin, gt_xmin)
    intersect_widths = np.maximum(0.0, (all_min_xmax - all_max_xmin))
    # 返回两个矩形相交区域的面积，即高度和宽度的乘积
    return intersect_heights * intersect_widths
# 计算两个多边形的交集面积
def area_of_intersection(det_x, det_y, gt_x, gt_y):
    # 根据检测框的顶点坐标创建 Polygon 对象，并进行缓冲处理
    p1 = Polygon(np.stack([det_x, det_y], axis=1)).buffer(0)
    # 根据真实框的顶点坐标创建 Polygon 对象，并进行缓冲处理
    p2 = Polygon(np.stack([gt_x, gt_y], axis=1)).buffer(0)
    # 返回两个多边形的交集面积
    return float(p1.intersection(p2).area)


# 计算两个多边形的并集面积
def area_of_union(det_x, det_y, gt_x, gt_y):
    # 根据检测框的顶点坐标创建 Polygon 对象，并进行缓冲处理
    p1 = Polygon(np.stack([det_x, det_y], axis=1)).buffer(0)
    # 根据真实框的顶点坐标创建 Polygon 对象，并进行缓冲处理
    p2 = Polygon(np.stack([gt_x, gt_y], axis=1)).buffer(0)
    # 返回两个多边形的并集面积
    return float(p1.union(p2).area)


# 计算两个多边形的交并比
def iou(det_x, det_y, gt_x, gt_y):
    # 返回交集面积除以并集面积加上一个小数的结果
    return area_of_intersection(det_x, det_y, gt_x, gt_y) / (
        area_of_union(det_x, det_y, gt_x, gt_y) + 1.0)


# 计算检测框的交并比
def iod(det_x, det_y, gt_x, gt_y):
    """
    This helper determine the fraction of intersection area over detection area
    """
    # 返回交集面积除以检测框面积加上一个小数的结果
    return area_of_intersection(det_x, det_y, gt_x, gt_y) / (
        area(det_x, det_y) + 1.0)
```