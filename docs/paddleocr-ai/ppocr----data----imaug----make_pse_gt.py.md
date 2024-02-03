# `.\PaddleOCR\ppocr\data\imaug\make_pse_gt.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

# 定义 MakePseGt 类
__all__ = ['MakePseGt']

class MakePseGt(object):
    # 初始化函数，设置参数
    def __init__(self, kernel_num=7, size=640, min_shrink_ratio=0.4, **kwargs):
        self.kernel_num = kernel_num
        self.min_shrink_ratio = min_shrink_ratio
        self.size = size
    # 定义一个类的调用方法，接受数据作为输入
    def __call__(self, data):

        # 从输入数据中获取图像、文本多边形和忽略标签
        image = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']

        # 获取图像的高度、宽度和通道数
        h, w, _ = image.shape
        # 计算图像的短边长度
        short_edge = min(h, w)
        # 如果短边长度小于指定大小self.size，则进行缩放
        if short_edge < self.size:
            # 计算缩放比例
            scale = self.size / short_edge
            # 对图像进行缩放
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            # 调整文本多边形坐标
            text_polys *= scale

        # 初始化存储文本区域的列表
        gt_kernels = []
        # 生成多个不同尺度的文本区域
        for i in range(1, self.kernel_num + 1):
            # 计算当前尺度的收缩比例
            rate = 1.0 - (1.0 - self.min_shrink_ratio) / (self.kernel_num - 1) * i
            # 生成当前尺度的文本区域和忽略标签
            text_kernel, ignore_tags = self.generate_kernel(
                image.shape[0:2], rate, text_polys, ignore_tags)
            # 将生成的文本区域添加到列表中
            gt_kernels.append(text_kernel)

        # 创建训练掩码，用于标记忽略区域
        training_mask = np.ones(image.shape[0:2], dtype='uint8')
        # 根据忽略标签填充训练掩码
        for i in range(text_polys.shape[0]):
            if ignore_tags[i]:
                cv2.fillPoly(training_mask,
                             text_polys[i].astype(np.int32)[np.newaxis, :, :],
                             0)

        # 将文本区域列表转换为NumPy数组，并将大于0的值设置为1
        gt_kernels = np.array(gt_kernels)
        gt_kernels[gt_kernels > 0] = 1

        # 更新数据字典中的图像、文本多边形、文本区域、文本标签和掩码
        data['image'] = image
        data['polys'] = text_polys
        data['gt_kernels'] = gt_kernels[0:]
        data['gt_text'] = gt_kernels[0]
        data['mask'] = training_mask.astype('float32')
        # 返回更新后的数据字典
        return data
    # 生成文本区域的核，用于文本检测
    def generate_kernel(self,
                        img_size,
                        shrink_ratio,
                        text_polys,
                        ignore_tags=None):
        """
        参考部分代码：
        https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/textdet_targets/base_textdet_targets.py
        """

        # 获取图像的高度和宽度
        h, w = img_size
        # 创建一个全零数组，用于存储文本区域的核
        text_kernel = np.zeros((h, w), dtype=np.float32)
        # 遍历文本多边形列表
        for i, poly in enumerate(text_polys):
            # 创建多边形对象
            polygon = Polygon(poly)
            # 计算缩小系数
            distance = polygon.area * (1 - shrink_ratio * shrink_ratio) / (
                polygon.length + 1e-6)
            # 将多边形转换为 Pyclipper 可接受的格式
            subject = [tuple(l) for l in poly]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            # 执行缩小操作
            shrinked = np.array(pco.Execute(-distance))

            # 如果缩小后的多边形为空，则跳过当前多边形
            if len(shrinked) == 0 or shrinked.size == 0:
                # 如果存在忽略标签列表，则将当前索引位置的标签设为 True
                if ignore_tags is not None:
                    ignore_tags[i] = True
                continue
            try:
                # 尝试将缩小后的多边形转换为二维数组
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
            except:
                # 如果转换失败，则跳过当前多边形
                if ignore_tags is not None:
                    ignore_tags[i] = True
                continue
            # 使用缩小后的多边形填充文本核数组
            cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)], i + 1)
        # 返回文本核数组和忽略标签列表
        return text_kernel, ignore_tags
```