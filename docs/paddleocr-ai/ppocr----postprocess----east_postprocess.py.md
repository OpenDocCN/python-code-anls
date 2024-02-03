# `.\PaddleOCR\ppocr\postprocess\east_postprocess.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，不附带任何担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from .locality_aware_nms import nms_locality
import cv2
import paddle
import os
from ppocr.utils.utility import check_install
import sys

# 定义 EASTPostProcess 类，用于 EAST 模型的后处理
class EASTPostProcess(object):
    """
    The post process for EAST.
    """

    def __init__(self,
                 score_thresh=0.8,
                 cover_thresh=0.1,
                 nms_thresh=0.2,
                 **kwargs):

        # 初始化阈值参数
        self.score_thresh = score_thresh
        self.cover_thresh = cover_thresh
        self.nms_thresh = nms_thresh

    # 从四边形恢复矩形
    def restore_rectangle_quad(self, origin, geometry):
        """
        Restore rectangle from quadrangle.
        """
        # 将原始坐标复制四份，拼接成 (n, 8) 的数组
        origin_concat = np.concatenate(
            (origin, origin, origin, origin), axis=1)  # (n, 8)
        # 通过几何信息计算预测的四边形坐标
        pred_quads = origin_concat - geometry
        # 重塑数组形状为 (n, 4, 2)
        pred_quads = pred_quads.reshape((-1, 4, 2))  # (n, 4, 2)
        return pred_quads
    # 检测文本框，根据分数图和几何图还原文本框
    def detect(self,
               score_map,
               geo_map,
               score_thresh=0.8,
               cover_thresh=0.1,
               nms_thresh=0.2):
        """
        restore text boxes from score map and geo map
        """

        # 获取第一个元素的分数图
        score_map = score_map[0]
        # 调整几何图的维度
        geo_map = np.swapaxes(geo_map, 1, 0)
        geo_map = np.swapaxes(geo_map, 1, 2)
        # 过滤分数图
        xy_text = np.argwhere(score_map > score_thresh)
        if len(xy_text) == 0:
            return []
        # 根据 y 轴对文本框进行排序
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # 还原四边形提议
        text_box_restored = self.restore_rectangle_quad(
            xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

        try:
            # 检查并导入 lanms 库
            check_install('lanms', 'lanms-nova')
            import lanms
            # 使用 lanms 库中的方法进行文本框合并
            boxes = lanms.merge_quadrangle_n9(boxes, nms_thresh)
        except:
            print(
                'You should install lanms by pip3 install lanms-nova to speed up nms_locality'
            )
            # 如果无法导入 lanms 库，则使用 nms_locality 方法进行文本框合并
            boxes = nms_locality(boxes.astype(np.float64), nms_thresh)
        if boxes.shape[0] == 0:
            return []
        # 在这里通过平均分数图过滤一些低分数的文本框，这与原始论文有所不同
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape(
                (-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        # 过滤掉覆盖率低于阈值的文本框
        boxes = boxes[boxes[:, 8] > cover_thresh]
        return boxes
    # 对多边形进行排序
    def sort_poly(self, p):
        # 找到四个顶点中横纵坐标和最小的顶点索引
        min_axis = np.argmin(np.sum(p, axis=1))
        # 重新排列多边形的顶点顺序
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        # 判断多边形的长边是横向还是纵向，返回重新排序后的多边形
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    # 对检测结果进行处理
    def __call__(self, outs_dict, shape_list):
        # 获取检测结果中的得分和几何信息
        score_list = outs_dict['f_score']
        geo_list = outs_dict['f_geo']
        # 如果得分和几何信息是PaddlePaddle的Tensor类型，则转换为numpy数组
        if isinstance(score_list, paddle.Tensor):
            score_list = score_list.numpy()
            geo_list = geo_list.numpy()
        # 获取图像数量
        img_num = len(shape_list)
        dt_boxes_list = []
        # 遍历每张图像
        for ino in range(img_num):
            # 获取当前图像的得分和几何信息
            score = score_list[ino]
            geo = geo_list[ino]
            # 进行文本检测，得到检测框
            boxes = self.detect(
                score_map=score,
                geo_map=geo,
                score_thresh=self.score_thresh,
                cover_thresh=self.cover_thresh,
                nms_thresh=self.nms_thresh)
            boxes_norm = []
            # 如果检测到文本框
            if len(boxes) > 0:
                # 获取图像的高度和宽度
                h, w = score.shape[1:]
                # 获取原始图像的高度、宽度和缩放比例
                src_h, src_w, ratio_h, ratio_w = shape_list[ino]
                # 将检测框坐标转换为原始图像坐标
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                # 对每个检测框进行处理
                for i_box, box in enumerate(boxes):
                    # 对多边形进行排序
                    box = self.sort_poly(box.astype(np.int32))
                    # 如果多边形的某两个顶点距离小于5，则跳过
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    boxes_norm.append(box)
            # 将处理后的检测框添加到列表中
            dt_boxes_list.append({'points': np.array(boxes_norm)})
        # 返回处理后的检测框列表
        return dt_boxes_list
```