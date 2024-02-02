# `arknights-mower\arknights_mower\ocr\decode.py`

```py
# 导入需要的库
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

# 定义 SegDetectorRepresenter 类
class SegDetectorRepresenter:
    # 初始化函数，设置默认阈值、框阈值、最大候选数和解除裁剪比例
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=2.0):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    # 调用函数，处理预测结果
    def __call__(self, pred, height, width):
        """
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        """

        # 获取预测结果的第一个通道
        pred = pred[0, :, :]
        # 通过阈值将预测结果二值化
        segmentation = self.binarize(pred)

        # 从二值化结果中获取文本框和得分
        boxes, scores = self.boxes_from_bitmap(
            pred, segmentation, width, height)

        return boxes, scores

    # 通过阈值将预测结果二值化
    def binarize(self, pred):
        return pred > self.thresh
    # 从位图中获取包围框
    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
        """

        # 断言位图的形状为二维
        assert len(bitmap.shape) == 2
        # 获取位图的高度和宽度
        height, width = bitmap.shape
        # 寻找位图的轮廓
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 确定要处理的轮廓数量
        num_contours = min(len(contours), self.max_candidates)
        # 创建用于存储包围框坐标的数组
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        # 创建用于存储分数的数组
        scores = np.zeros((num_contours,), dtype=np.float32)
        # 遍历每个轮廓
        for index in range(num_contours):
            # 将轮廓展平为一维数组
            contour = contours[index].squeeze(1)
            # 获取最小外接矩形的顶点坐标和最短边的长度
            points, sside = self.get_mini_boxes(contour)
            # 如果最短边的长度小于最小尺寸要求，则跳过
            if sside < self.min_size:
                continue
            # 将顶点坐标转换为数组
            points = np.array(points)
            # 计算包围框的分数
            score = self.box_score_fast(pred, contour)
            # 如果包围框的分数低于阈值，则跳过
            if self.box_thresh > score:
                continue
            # 对包围框进行解裁
            box = self.unclip(
                points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            # 获取解裁后的包围框和最短边的长度
            box, sside = self.get_mini_boxes(box)
            # 如果解裁后的包围框最短边的长度小于最小尺寸要求，则跳过
            if sside < self.min_size + 2:
                continue
            # 将包围框转换为数组
            box = np.array(box)
            # 如果目标宽度和高度不是整数，则转换为整数
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            # 对包围框的坐标进行缩放和裁剪
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            # 将包围框坐标和分数存储到数组中
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        # 返回包围框数组和分数数组
        return boxes, scores
    # 根据给定的边界框和放大比例，对多边形进行放大
    def unclip(self, box, unclip_ratio=1.5):
        # 将边界框转换为多边形
        poly = Polygon(box)

        # 计算放大的距离
        distance = poly.area * unclip_ratio / (poly.length)
        # 创建 PyclipperOffset 对象
        offset = pyclipper.PyclipperOffset()
        # 添加路径到 PyclipperOffset 对象中
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 执行放大操作
        expanded = np.array(offset.Execute(distance))
        return expanded

    # 获取最小外接矩形的四个顶点坐标和最小外接矩形的高度
    def get_mini_boxes(self, contour):
        # 获取最小外接矩形
        bounding_box = cv2.minAreaRect(contour)
        # 将顶点按照 x 坐标排序
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # 根据顶点的 y 坐标值确定顶点的顺序
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

        # 根据顶点的顺序构建边界框
        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        # 返回边界框和最小外接矩形的高度
        return box, min(bounding_box[1])

    # 计算边界框内部的平均灰度值
    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        # 复制边界框
        box = _box.copy()
        # 计算边界框的 x 和 y 范围
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        # 创建一个与边界框大小相同的零矩阵
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        # 将边界框坐标转换为相对于左上角的偏移量
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        # 在 mask 上填充边界框
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        # 计算边界框内部的平均灰度值
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
```