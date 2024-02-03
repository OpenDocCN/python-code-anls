# `.\PaddleOCR\ppocr\postprocess\picodet_postprocess.py`

```py
# 导入所需的库
import numpy as np
from scipy.special import softmax

# 定义一个函数，实现硬性非极大值抑制
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    # 提取盒子的得分
    scores = box_scores[:, -1]
    # 提取盒子的坐标
    boxes = box_scores[:, :-1]
    # 初始化一个空列表用于存储选中的盒子的索引
    picked = []
    # 对盒子得分进行排序，获取排序后的索引
    indexes = np.argsort(scores)
    # 仅保留候选盒子的数量
    indexes = indexes[-candidate_size:]
    # 循环直到所有候选盒子都被处理
    while len(indexes) > 0:
        # 获取当前盒子的索引
        current = indexes[-1]
        # 将当前盒子的索引添加到选中列表中
        picked.append(current)
        # 如果达到了指定的 top_k 数量或者所有盒子都被处理完，则退出循环
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        # 获取当前盒子的坐标
        current_box = boxes[current, :]
        # 移除当前盒子，计算其余盒子与当前盒子的 IoU
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        # 根据 IoU 阈值筛选出与当前盒子重叠度低的盒子
        indexes = indexes[iou <= iou_threshold]

    # 返回选中的盒子及其得分
    return box_scores[picked, :]

# 计算两组盒子之间的交并比（IoU）
def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    # 计算两组框的左上角坐标的最大值，即两组框的交集的左上角坐标
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    # 计算两组框的右下角坐标的最小值，即两组框的交集的右下角坐标
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    # 计算交集区域的面积
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    # 计算第一组框的面积
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    # 计算第二组框的面积
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    # 计算两组框的交集 over 两组框的并集的比值，加上一个很小的数 eps 避免分母为零
    return overlap_area / (area0 + area1 - overlap_area + eps)
def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    # 计算矩形的面积，给定两个角
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


class PicoDetPostProcess(object):
    """
    Args:
        input_shape (int): network input image size
        ori_shape (int): ori image shape of before padding
        scale_factor (float): scale factor of ori image
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 layout_dict_path,
                 strides=[8, 16, 32, 64],
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 nms_top_k=1000,
                 keep_top_k=100):
        # 加载标签字典
        self.labels = self.load_layout_dict(layout_dict_path)
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def load_layout_dict(self, layout_dict_path):
        # 读取标签字典文件
        with open(layout_dict_path, 'r', encoding='utf-8') as fp:
            labels = fp.readlines()
        return [label.strip('\n') for label in labels]
    # 对边界框应用变换
    def warp_boxes(self, boxes, ori_shape):
        """Apply transform to boxes
        """
        # 获取原始图像的宽度和高度
        width, height = ori_shape[1], ori_shape[0]
        # 获取边界框的数量
        n = len(boxes)
        if n:
            # 变换点坐标
            xy = np.ones((n * 4, 3))
            # 将边界框坐标转换为点坐标形式
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            # 对点坐标进行缩放
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # 创建新的边界框
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # 限制边界框的范围在图像内
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes

    # 获取图像信息
    def img_info(self, ori_img, img):
        # 获取原始图像和调整大小后的图像的形状
        origin_shape = ori_img.shape
        resize_shape = img.shape
        # 计算图像在y轴和x轴上的缩放比例
        im_scale_y = resize_shape[2] / float(origin_shape[0])
        im_scale_x = resize_shape[3] / float(origin_shape[1])
        scale_factor = np.array([im_scale_y, im_scale_x], dtype=np.float32)
        img_shape = np.array(img.shape[2:], dtype=np.float32)

        input_shape = np.array(img).astype('float32').shape[2:]
        ori_shape = np.array((img_shape, )).astype('float32')
        scale_factor = np.array((scale_factor, )).astype('float32')
        return ori_shape, input_shape, scale_factor
```