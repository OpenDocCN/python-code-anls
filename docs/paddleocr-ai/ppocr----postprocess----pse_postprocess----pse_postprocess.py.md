# `.\PaddleOCR\ppocr\postprocess\pse_postprocess\pse_postprocess.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/whai362/PSENet/blob/python3/models/head/psenet_head.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import paddle
from paddle.nn import functional as F

from ppocr.postprocess.pse_postprocess.pse import pse

# 定义 PSE 后处理类
class PSEPostProcess(object):
    """
    The post process for PSE.
    """

    # 初始化方法
    def __init__(self,
                 thresh=0.5,
                 box_thresh=0.85,
                 min_area=16,
                 box_type='quad',
                 scale=4,
                 **kwargs):
        # 断言，只支持 'quad' 和 'poly' 两种框类型
        assert box_type in ['quad', 'poly'], 'Only quad and poly is supported'
        # 设置阈值
        self.thresh = thresh
        # 设置框阈值
        self.box_thresh = box_thresh
        # 设置最小区域
        self.min_area = min_area
        # 设置框类型
        self.box_type = box_type
        # 设置缩放比例
        self.scale = scale
    # 定义一个类方法，用于从输出字典和形状列表中获取预测结果
    def __call__(self, outs_dict, shape_list):
        # 从输出字典中获取预测结果
        pred = outs_dict['maps']
        # 如果预测结果不是 PaddlePaddle 的张量，则转换为张量
        if not isinstance(pred, paddle.Tensor):
            pred = paddle.to_tensor(pred)
        # 对预测结果进行插值，将其缩放到指定比例
        pred = F.interpolate(
            pred, scale_factor=4 // self.scale, mode='bilinear')

        # 从预测结果中获取分数
        score = F.sigmoid(pred[:, 0, :, :])

        # 根据阈值生成内核
        kernels = (pred > self.thresh).astype('float32')
        text_mask = kernels[:, 0, :, :]
        text_mask = paddle.unsqueeze(text_mask, axis=1)

        # 将内核应用于文本掩码
        kernels[:, 0:, :, :] = kernels[:, 0:, :, :] * text_mask

        # 将分数和内核转换为 NumPy 数组
        score = score.numpy()
        kernels = kernels.numpy().astype(np.uint8)

        # 初始化存储框和分数的列表
        boxes_batch = []
        # 遍历每个批次的预测结果
        for batch_index in range(pred.shape[0]):
            # 从分数和内核中生成框和分数
            boxes, scores = self.boxes_from_bitmap(score[batch_index],
                                                   kernels[batch_index],
                                                   shape_list[batch_index])

            # 将框和分数添加到列表中
            boxes_batch.append({'points': boxes, 'scores': scores})
        # 返回存储框和分数的列表
        return boxes_batch

    # 从位图中生成框
    def boxes_from_bitmap(self, score, kernels, shape):
        # 使用 PSE 算法从内核中生成标签
        label = pse(kernels, self.min_area)
        # 生成框并返回
        return self.generate_box(score, label, shape)
    # 生成包围框，根据得分、标签和形状信息
    def generate_box(self, score, label, shape):
        # 解析形状信息
        src_h, src_w, ratio_h, ratio_w = shape
        # 获取标签中的最大值，即标签数
        label_num = np.max(label) + 1

        # 初始化空列表用于存储包围框和得分
        boxes = []
        scores = []
        # 遍历标签数，从1开始
        for i in range(1, label_num):
            # 获取标签中等于当前值的索引
            ind = label == i
            # 获取这些索引对应的坐标点
            points = np.array(np.where(ind)).transpose((1, 0))[:, ::-1]

            # 如果坐标点数量小于最小面积要求，则将对应标签置为0并继续下一次循环
            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue

            # 计算当前标签的平均得分
            score_i = np.mean(score[ind])
            # 如果得分低于阈值，则将对应标签置为0并继续下一次循环
            if score_i < self.box_thresh:
                label[ind] = 0
                continue

            # 根据包围框类型生成包围框
            if self.box_type == 'quad':
                # 使用坐标点拟合最小外接矩形
                rect = cv2.minAreaRect(points)
                # 获取矩形的四个顶点坐标
                bbox = cv2.boxPoints(rect)
            elif self.box_type == 'poly':
                # 计算多边形包围框的高度和宽度
                box_height = np.max(points[:, 1]) + 10
                box_width = np.max(points[:, 0]) + 10

                # 创建一个空白图像作为掩模
                mask = np.zeros((box_height, box_width), np.uint8)
                # 在掩模上标记坐标点
                mask[points[:, 1], points[:, 0]] = 255

                # 寻找掩模中的轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                # 提取轮廓的坐标点
                bbox = np.squeeze(contours[0], 1)
            else:
                # 如果包围框类型不是'quad'或'poly'，则抛出未实现错误
                raise NotImplementedError

            # 根据比例因子将包围框坐标映射回原始图像尺寸，并进行边界裁剪
            bbox[:, 0] = np.clip(np.round(bbox[:, 0] / ratio_w), 0, src_w)
            bbox[:, 1] = np.clip(np.round(bbox[:, 1] / ratio_h), 0, src_h)
            # 将包围框和得分添加到列表中
            boxes.append(bbox)
            scores.append(score_i)
        # 返回包围框列表和得分列表
        return boxes, scores
```