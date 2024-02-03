# `.\PaddleOCR\ppocr\postprocess\db_postprocess.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则不得使用此文件；
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
"""
# 引用来源
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/post_processing/seg_detector_representer.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入所需的库
import numpy as np
import cv2
import paddle
from shapely.geometry import Polygon
import pyclipper

# 定义 DBPostProcess 类，用于不同iable Binarization（DB）的后处理
class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    # 初始化函数，设置参数
    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 box_type='quad',
                 **kwargs):
        # 设置阈值
        self.thresh = thresh
        # 设置框的阈值
        self.box_thresh = box_thresh
        # 设置最大候选框数
        self.max_candidates = max_candidates
        # 设置解除裁剪比例
        self.unclip_ratio = unclip_ratio
        # 设置最小尺寸
        self.min_size = 3
        # 设置评分模式
        self.score_mode = score_mode
        # 设置框的类型
        self.box_type = box_type
        # 断言，确保评分模式在指定范围内
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        # 如果使用膨胀，则设置膨胀核
        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])
    # 从二进制位图中提取多边形，返回多边形的顶点坐标和得分
    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: 单通道位图，形状为 (1, H, W)，值为 {0, 1}
        '''

        # 复制位图
        bitmap = _bitmap
        # 获取位图的高度和宽度
        height, width = bitmap.shape

        # 初始化多边形框和得分列表
        boxes = []
        scores = []

        # 寻找位图的轮廓
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓
        for contour in contours[:self.max_candidates]:
            # 计算轮廓的周长
            epsilon = 0.002 * cv2.arcLength(contour, True)
            # 近似轮廓
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 将近似的轮廓转换为点坐标
            points = approx.reshape((-1, 2))
            # 如果点的数量小于4，则跳过
            if points.shape[0] < 4:
                continue

            # 计算多边形框的得分
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            # 如果得分低于阈值，则跳过
            if self.box_thresh > score:
                continue

            # 如果点的数量大于2
            if points.shape[0] > 2:
                # 对多边形框进行解裁
                box = self.unclip(points, self.unclip_ratio)
                # 如果解裁后的多边形框数量大于1，则跳过
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            # 获取最小外接矩形的角点和最短边长
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            # 如果最短边长小于最小尺寸加2，则跳过
            if sside < self.min_size + 2:
                continue

            # 将多边形框的坐标映射到目标尺寸
            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            # 将多边形框和得分添加到列表中
            boxes.append(box.tolist())
            scores.append(score)
        # 返回多边形框和得分列表
        return boxes, scores
    # 从二值化的位图中提取文本框信息
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        # 复制位图数据
        bitmap = _bitmap
        # 获取位图的高度和宽度
        height, width = bitmap.shape

        # 寻找位图中的轮廓
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        # 根据返回的轮廓数量进行处理
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        # 确定要处理的轮廓数量
        num_contours = min(len(contours), self.max_candidates)

        # 初始化文本框和得分列表
        boxes = []
        scores = []
        # 遍历每个轮廓
        for index in range(num_contours):
            # 获取当前轮廓
            contour = contours[index]
            # 获取轮廓的最小外接矩形的顶点和边长
            points, sside = self.get_mini_boxes(contour)
            # 如果外接矩形边长小于最小尺寸要求，则跳过
            if sside < self.min_size:
                continue
            # 将顶点转换为 NumPy 数组
            points = np.array(points)
            # 根据评分模式选择快速或慢速计算得分
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            # 如果得分低于阈值，则跳过
            if self.box_thresh > score:
                continue

            # 对外接矩形进行解裁
            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            # 如果解裁后的外接矩形边长小于最小尺寸要求，则跳过
            if sside < self.min_size + 2:
                continue
            # 将外接矩形坐标映射到目标尺寸
            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            # 将文本框和得分添加到列表中
            boxes.append(box.astype("int32"))
            scores.append(score)
        # 返回文本框数组和得分列表
        return np.array(boxes, dtype="int32"), scores
    # 根据给定的边界框和扩展比例进行解除裁剪操作
    def unclip(self, box, unclip_ratio):
        # 将边界框转换为多边形对象
        poly = Polygon(box)
        # 计算扩展距离
        distance = poly.area * unclip_ratio / poly.length
        # 创建 PyclipperOffset 对象
        offset = pyclipper.PyclipperOffset()
        # 添加路径到 PyclipperOffset 对象中
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 执行扩展操作并返回结果
        expanded = np.array(offset.Execute(distance))
        return expanded

    # 获取最小外接矩形的四个顶点坐标
    def get_mini_boxes(self, contour):
        # 获取最小外接矩形
        bounding_box = cv2.minAreaRect(contour)
        # 获取外接矩形的四个顶点坐标并按 x 坐标排序
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # 根据顶点坐标的 y 值确定顶点的顺序
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        # 根据确定的顶点顺序构建外接矩形的四个顶点坐标
        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        # 返回外接矩形的四个顶点坐标和最小外接矩形的最小边长
        return box, min(bounding_box[1])

    # 计算边界框内部的像素均值作为得分
    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        # 获取位图的高度和宽度
        h, w = bitmap.shape[:2]
        # 复制边界框
        box = _box.copy()
        # 计算边界框在 x 轴方向的最小和最大值
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        # 计算边界框在 y 轴方向的最小和最大值
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        # 创建掩码图像
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        # 将边界框坐标转换为相对于左上角的偏移量
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        # 在掩码图像上填充多边形区域
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        # 计算边界框内部像素的均值作为得分
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    # 使用多边形的平均分数作为平均分数
    def box_score_slow(self, bitmap, contour):
        # 获取位图的高度和宽度
        h, w = bitmap.shape[:2]
        # 复制轮廓数据
        contour = contour.copy()
        # 重塑轮廓数据的形状
        contour = np.reshape(contour, (-1, 2))

        # 计算边界框的最小和最大值
        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        # 创建一个与边界框大小相同的零矩阵
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        # 调整轮廓坐标
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        # 使用轮廓数据填充 mask
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        # 返回位图中指定区域的平均值
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    # 对象调用函数
    def __call__(self, outs_dict, shape_list):
        # 获取预测结果
        pred = outs_dict['maps']
        # 如果预测结果是张量，则转换为 numpy 数组
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        # 选择第一个通道的数据
        pred = pred[:, 0, :, :]
        # 根据阈值生成分割结果
        segmentation = pred > self.thresh

        # 存储边界框的列表
        boxes_batch = []
        # 遍历每个批次的预测结果
        for batch_index in range(pred.shape[0]):
            # 获取原始图像的高度、宽度和缩放比例
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            # 如果存在膨胀核，则对分割结果进行膨胀操作
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            # 根据边界框类型生成边界框和分数
            if self.box_type == 'poly':
                boxes, scores = self.polygons_from_bitmap(pred[batch_index],
                                                          mask, src_w, src_h)
            elif self.box_type == 'quad':
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                       src_w, src_h)
            else:
                raise ValueError("box_type can only be one of ['quad', 'poly']")

            # 将边界框信息添加到列表中
            boxes_batch.append({'points': boxes})
        # 返回边界框列表
        return boxes_batch
# 定义一个DistillationDBPostProcess类，用于后处理模型输出结果
class DistillationDBPostProcess(object):
    # 初始化方法，接受一系列参数
    def __init__(self,
                 model_name=["student"],  # 模型名称，默认为"student"
                 key=None,  # 关键字参数，默认为None
                 thresh=0.3,  # 阈值参数，默认为0.3
                 box_thresh=0.6,  # 边界框阈值参数，默认为0.6
                 max_candidates=1000,  # 最大候选框数量，默认为1000
                 unclip_ratio=1.5,  # 解除裁剪比例，默认为1.5
                 use_dilation=False,  # 是否使用膨胀，默认为False
                 score_mode="fast",  # 分数模式，默认为"fast"
                 box_type='quad',  # 边界框类型，默认为'quad'
                 **kwargs):  # 其他关键字参数
        # 初始化模型名称和关键字参数
        self.model_name = model_name
        self.key = key
        # 创建一个DBPostProcess对象，用于后处理
        self.post_process = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            box_type=box_type)

    # 定义__call__方法，用于调用对象
    def __call__(self, predicts, shape_list):
        # 初始化结果字典
        results = {}
        # 遍历模型名称列表
        for k in self.model_name:
            # 对每个模型名称进行后处理，将结果存入结果字典
            results[k] = self.post_process(predicts[k], shape_list=shape_list)
        # 返回结果字典
        return results
```