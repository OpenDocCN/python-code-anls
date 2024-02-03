# `.\PaddleOCR\ppocr\data\imaug\make_border_map.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""
# 引入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

# 设置 numpy 的错误处理方式
np.seterr(divide='ignore', invalid='ignore')
# 导入 pyclipper 库
import pyclipper
# 导入 Polygon 类
from shapely.geometry import Polygon
import sys
import warnings

# 忽略警告
warnings.simplefilter("ignore")

# 定义 MakeBorderMap 类
__all__ = ['MakeBorderMap']

class MakeBorderMap(object):
    # 初始化函数
    def __init__(self,
                 shrink_ratio=0.4,
                 thresh_min=0.3,
                 thresh_max=0.7,
                 **kwargs):
        # 设置收缩比例、最小阈值和最大阈值
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        # 如果参数中包含 'total_epoch' 和 'epoch'，并且 'epoch' 不为 "None"
        if 'total_epoch' in kwargs and 'epoch' in kwargs and kwargs['epoch'] != "None":
            # 根据训练周期调整收缩比例
            self.shrink_ratio = self.shrink_ratio + 0.2 * kwargs['epoch'] / float(kwargs['total_epoch'])
    # 定义一个方法，用于处理输入数据
    def __call__(self, data):
        
        # 从输入数据中获取图像数据、文本多边形坐标和忽略标签
        img = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']
        
        # 创建一个与图像大小相同的全零矩阵和全零矩阵
        canvas = np.zeros(img.shape[:2], dtype=np.float32)
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        
        # 遍历文本多边形坐标
        for i in range(len(text_polys)):
            # 如果对应的忽略标签为真，则跳过当前循环
            if ignore_tags[i]:
                continue
            # 调用方法绘制边界地图
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        # 对生成的地图进行线性变换
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        
        # 将处理后的地图和掩码保存到输入数据中
        data['threshold_map'] = canvas
        data['threshold_mask'] = mask
        return data

    # 定义一个方法，用于计算点到直线的距离
    def _distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        # 获取坐标数组的高度和宽度
        height, width = xs.shape[:2]
        # 计算点到直线的距离
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result
    # 根据给定的两个点，通过缩放比例计算出延长线的起点
    ex_point_1 = (int(
        round(point_1[0] + (point_1[0] - point_2[0]) * (1 + shrink_ratio))),
                  int(
                      round(point_1[1] + (point_1[1] - point_2[1]) * (
                          1 + shrink_ratio))))
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
        round(point_2[0] + (point_2[0] - point_1[0]) * (1 + shrink_ratio))),
                  int(
                      round(point_2[1] + (point_2[1] - point_1[1]) * (
                          1 + shrink_ratio))))
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