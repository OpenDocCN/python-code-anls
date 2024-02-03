# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\post_processing\seg_detector_representer.py`

```py
# 导入需要的库
import cv2
import numpy as np
import pyclipper
import paddle
from shapely.geometry import Polygon

# 定义 SegDetectorRepresenter 类
class SegDetectorRepresenter():
    # 初始化函数，设置默认阈值、框阈值、最大候选数和未裁剪比例
    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    # 调用函数，处理预测结果
    def __call__(self, batch, pred, is_output_polygon=False):
        '''
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
        '''
        # 将 paddle.Tensor 转换为 numpy 数组
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        # 获取预测结果的第一个通道
        pred = pred[:, 0, :, :]
        # 二值化预测结果
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        # 遍历每个 batch
        for batch_index in range(pred.shape[0]):
            height, width = batch['shape'][batch_index]
            # 如果输出多边形
            if is_output_polygon:
                # 从位图中获取多边形
                boxes, scores = self.polygons_from_bitmap(
                    pred[batch_index], segmentation[batch_index], width, height)
            else:
                # 从位图中获取框
                boxes, scores = self.boxes_from_bitmap(
                    pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        # 返回框和得分
        return boxes_batch, scores_batch
    # 将预测结果二值化，大于阈值的为True，小于等于阈值的为False
    def binarize(self, pred):
        return pred > self.thresh

    # 从位图中提取多边形信息
    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: 单通道位图，形状为 (H, W)，值为 {0, 1}
        '''

        # 断言位图的维度为2
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap  # 第一个通道
        height, width = bitmap.shape
        boxes = []
        scores = []

        # 寻找位图的轮廓
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓
        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            # 计算盒子得分
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            # 如果目标宽度不是整数，则转换为整数
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            # 对盒子坐标进行裁剪和缩放
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        # 确保输入的位图是二维的
        assert len(_bitmap.shape) == 2
        # 获取位图的高度和宽度
        bitmap = _bitmap  # The first channel
        height, width = bitmap.shape
        # 寻找位图的轮廓
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 确定要处理的轮廓数量
        num_contours = min(len(contours), self.max_candidates)
        # 初始化存储框和分数的数组
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)

        # 遍历每个轮廓
        for index in range(num_contours):
            # 获取当前轮廓的点和最小边长
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            # 如果最小边长小于阈值，则跳过
            if sside < self.min_size:
                continue
            points = np.array(points)
            # 计算当前框的得分
            score = self.box_score_fast(pred, contour)
            # 如果得分低于阈值，则跳过
            if self.box_thresh > score:
                continue

            # 对框进行解裁
            box = self.unclip(
                points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            # 如果解裁后的框边长小于阈值，则跳过
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            # 确保目标宽度和高度是整数
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            # 将框的坐标映射到目标宽度和高度上
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            # 存储框和得分
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        # 返回框和得分
        return boxes, scores
    # 对输入的四边形框进行解除裁剪，返回扩展后的四边形框
    def unclip(self, box, unclip_ratio=1.5):
        # 将四边形框转换为多边形对象
        poly = Polygon(box)
        # 计算扩展距离
        distance = poly.area * unclip_ratio / poly.length
        # 创建 PyclipperOffset 对象
        offset = pyclipper.PyclipperOffset()
        # 添加路径到 PyclipperOffset 对象中
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 执行扩展操作，返回扩展后的多边形
        expanded = np.array(offset.Execute(distance))
        return expanded

    # 获取最小外接矩形框及其高度
    def get_mini_boxes(self, contour):
        # 获取最小外接矩形框
        bounding_box = cv2.minAreaRect(contour)
        # 获取外接矩形框的四个顶点，并按 x 坐标排序
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # 根据顶点的 y 坐标确定顶点的顺序
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

        # 根据确定的顶点顺序构建四边形框
        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        # 返回四边形框及其高度
        return box, min(bounding_box[1])

    # 计算四边形框内的像素平均值
    def box_score_fast(self, bitmap, _box):
        # 获取位图的高度和宽度
        h, w = bitmap.shape[:2]
        # 复制输入的四边形框
        box = _box.copy()
        # 计算四边形框的最小外接矩形的左上角和右下角坐标
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        # 创建一个与最小外接矩形大小相同的全零数组
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        # 将四边形框坐标转换为相对于最小外接矩形的坐标
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        # 在 mask 上填充四边形框
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        # 计算四边形框内像素的平均值
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
```