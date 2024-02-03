# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\data_loader\modules\make_border_map.py`

```
import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pyclipper
from shapely.geometry import Polygon

# 定义一个类用于生成边界地图
class MakeBorderMap():
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        # 初始化函数，设置缩小比例、最小阈值和最大阈值
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        # 从输入数据中获取图像、文本多边形和忽略标签
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        # 创建一个全零数组作为画布和掩模
        canvas = np.zeros(im.shape[:2], dtype=np.float32)
        mask = np.zeros(im.shape[:2], dtype=np.float32)

        # 遍历文本多边形
        for i in range(len(text_polys)):
            # 如果是忽略标签，则跳过
            if ignore_tags[i]:
                continue
            # 绘制边界地图
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        # 根据最小和最大阈值调整画布
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        # 将结果存入数据字典中
        data['threshold_map'] = canvas
        data['threshold_mask'] = mask
        return data
    # 计算点到直线的距离
    # ys: 第一轴的坐标
    # xs: 第二轴的坐标
    # point_1, point_2: (x, y)，直线的两个端点

    # 获取 xs 的高度和宽度
    height, width = xs.shape[:2]

    # 计算点到 point_1 的距离的平方
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])

    # 计算点到 point_2 的距离的平方
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])

    # 计算两点之间的距离的平方
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    # 计算余弦值
    cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))

    # 计算正弦值的平方
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)

    # 计算点到直线的距离
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

    # 如果余弦值小于 0，则取最小距离
    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]

    # 返回计算结果
    return result
    # 根据给定的两个点，通过缩放比例计算出延长线的起点
    ex_point_1 = (int(
        round(point_1[0] + (point_1[0] - point_2[0]) * (
            1 + self.shrink_ratio))), int(
                round(point_1[1] + (point_1[1] - point_2[1]) * (
                    1 + self.shrink_ratio))))
    # 在结果图像上绘制从延长线起点到原始点的线段
    cv2.line(
        result,
        tuple(ex_point_1),
        tuple(point_1),
        4096.0,
        1,
        lineType=cv2.LINE_AA,
        shift=0)
    # 根据给定的两个点，通过缩放比例计算出延长线的终点
    ex_point_2 = (int(
        round(point_2[0] + (point_2[0] - point_1[0]) * (
            1 + self.shrink_ratio))), int(
                round(point_2[1] + (point_2[1] - point_1[1]) * (
                    1 + self.shrink_ratio))))
    # 在结果图像上绘制从延长线终点到原始点的线段
    cv2.line(
        result,
        tuple(ex_point_2),
        tuple(point_2),
        4096.0,
        1,
        lineType=cv2.LINE_AA,
        shift=0)
    # 返回延长线的起点和终点
    return ex_point_1, ex_point_2
```