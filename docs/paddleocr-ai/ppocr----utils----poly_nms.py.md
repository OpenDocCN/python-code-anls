# `.\PaddleOCR\ppocr\utils\poly_nms.py`

```py
# 版权声明
#
# 版权所有 (c) 2022 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以获取特定语言下的权限和限制。

import numpy as np
from shapely.geometry import Polygon

def points2polygon(points):
    """将 k 个点转换为一个多边形。

    Args:
        points (ndarray or list): 一个形状为 (2k) 的 ndarray 或列表，表示 k 个点。

    Returns:
        polygon (Polygon): 一个多边形对象。
    """
    if isinstance(points, list):
        points = np.array(points)

    assert isinstance(points, np.ndarray)
    assert (points.size % 2 == 0) and (points.size >= 8)

    point_mat = points.reshape([-1, 2])
    return Polygon(point_mat)

def poly_intersection(poly_det, poly_gt, buffer=0.0001):
    """计算两个多边形之间的交集面积。

    Args:
        poly_det (Polygon): 检测器预测的多边形。
        poly_gt (Polygon): 真实多边形。

    Returns:
        intersection_area (float): 两个多边形之间的交集面积。
    """
    assert isinstance(poly_det, Polygon)
    assert isinstance(poly_gt, Polygon)

    if buffer == 0:
        poly_inter = poly_det & poly_gt
    else:
        poly_inter = poly_det.buffer(buffer) & poly_gt.buffer(buffer)
    return poly_inter.area, poly_inter

def poly_union(poly_det, poly_gt):
    """计算两个多边形之间的并集面积。

    Args:
        poly_det (Polygon): 检测器预测的多边形。
        poly_gt (Polygon): 真实多边形。
    # 返回两个多边形之间的并集面积
    Returns:
        union_area (float): The union area between two polygons.
    """
    # 断言 poly_det 是 Polygon 类型
    assert isinstance(poly_det, Polygon)
    # 断言 poly_gt 是 Polygon 类型
    assert isinstance(poly_gt, Polygon)

    # 计算 poly_det 的面积
    area_det = poly_det.area
    # 计算 poly_gt 的面积
    area_gt = poly_gt.area
    # 计算两个多边形的交集面积
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    # 返回两个多边形的并集面积
    return area_det + area_gt - area_inters
# 检查边界是否有效，根据边界长度和是否包含分数来判断
def valid_boundary(x, with_score=True):
    # 获取边界点的数量
    num = len(x)
    # 如果点的数量小于8，则返回 False
    if num < 8:
        return False
    # 如果点的数量为偶数且不包含分数，则返回 True
    if num % 2 == 0 and (not with_score):
        return True
    # 如果点的数量为奇数且包含分数，则返回 True
    if num % 2 == 1 and with_score:
        return True
    # 其他情况返回 False

# 计算两个边界之间的 IOU
def boundary_iou(src, target):
    """Calculate the IOU between two boundaries.

    Args:
       src (list): Source boundary.
       target (list): Target boundary.

    Returns:
       iou (float): The iou between two boundaries.
    """
    # 确保源边界和目标边界都是有效的
    assert valid_boundary(src, False)
    assert valid_boundary(target, False)
    # 将边界点转换为多边形
    src_poly = points2polygon(src)
    target_poly = points2polygon(target)
    # 计算多边形之间的 IOU
    return poly_iou(src_poly, target_poly)

# 计算两个多边形之间的 IOU
def poly_iou(poly_det, poly_gt):
    """Calculate the IOU between two polygons.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        iou (float): The IOU between two polygons.
    """
    # 确保输入的多边形是 Polygon 类型
    assert isinstance(poly_det, Polygon)
    assert isinstance(poly_gt, Polygon)
    # 计算多边形的交集和并集面积
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    area_union = poly_union(poly_det, poly_gt)
    # 如果并集面积为0，则返回0.0
    if area_union == 0:
        return 0.0
    # 计算 IOU 并返回
    return area_inters / area_union

# 多边形的非极大值抑制
def poly_nms(polygons, threshold):
    # 确保输入的多边形是列表类型
    assert isinstance(polygons, list)
    # 将多边形按照最后一个元素（分数）排序
    polygons = np.array(sorted(polygons, key=lambda x: x[-1]))
    # 保留的多边形列表
    keep_poly = []
    # 索引列表
    index = [i for i in range(polygons.shape[0])]
    # 循环直到索引列表为空
    while len(index) > 0:
        # 将最后一个多边形添加到保留列表中
        keep_poly.append(polygons[index[-1]].tolist())
        A = polygons[index[-1]][:-1]
        index = np.delete(index, -1)
        iou_list = np.zeros((len(index), ))
        # 计算当前多边形与其他多边形的 IOU
        for i in range(len(index)):
            B = polygons[index[i]][:-1]
            iou_list[i] = boundary_iou(A, B)
        # 根据阈值删除重叠的多边形
        remove_index = np.where(iou_list > threshold)
        index = np.delete(index, remove_index)
    # 返回保留的多边形列表
    return keep_poly
```