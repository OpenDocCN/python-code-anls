# `.\PaddleOCR\ppocr\postprocess\sast_postprocess.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(__file__)
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将当前目录的上一级目录添加到系统路径中
sys.path.append(os.path.join(__dir__, '..'))

import numpy as np
# 导入局部感知 NMS 模块
from .locality_aware_nms import nms_locality
import paddle
import cv2
import time

# 定义 SAST 后处理类
class SASTPostProcess(object):
    """
    The post process for SAST.
    """

    def __init__(self,
                 score_thresh=0.5,
                 nms_thresh=0.2,
                 sample_pts_num=2,
                 shrink_ratio_of_width=0.3,
                 expand_scale=1.0,
                 tcl_map_thresh=0.5,
                 **kwargs):

        # 设置阈值
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.sample_pts_num = sample_pts_num
        self.shrink_ratio_of_width = shrink_ratio_of_width
        self.expand_scale = expand_scale
        self.tcl_map_thresh = tcl_map_thresh

        # 判断是否为 Python 3.5 版本
        self.is_python35 = False
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            self.is_python35 = True
    # 将垂直的点对转换为顺时针方向的多边形点
    def point_pair2poly(self, point_pair_list):
        """
        Transfer vertical point_pairs into poly point in clockwise.
        """
        # 计算点的数量
        point_num = len(point_pair_list) * 2
        # 初始化点列表
        point_list = [0] * point_num
        # 遍历点对列表，将点对中的点按顺序添加到点列表中
        for idx, point_pair in enumerate(point_pair_list):
            point_list[idx] = point_pair[0]
            point_list[point_num - 1 - idx] = point_pair[1]
        # 将点列表转换为 numpy 数组，并按照二维形状返回
        return np.array(point_list).reshape(-1, 2)

    # 沿着宽度生成缩小的四边形
    def shrink_quad_along_width(self,
                                quad,
                                begin_width_ratio=0.,
                                end_width_ratio=1.):
        """ 
        Generate shrink_quad_along_width.
        """
        # 构建宽度比例对
        ratio_pair = np.array(
            [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
        # 计算四边形的两个顶点
        p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
        p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
        # 返回缩小后的四边形顶点坐标
        return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0])
    # 沿着宽度扩展多边形
    def expand_poly_along_width(self, poly, shrink_ratio_of_width=0.3):
        """
        expand poly along width.
        """
        # 获取多边形的点数
        point_num = poly.shape[0]
        # 左侧四边形的点
        left_quad = np.array(
            [poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
        # 计算左侧四边形的缩小比例
        left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                     (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
        # 根据缩小比例扩展左侧四边形
        left_quad_expand = self.shrink_quad_along_width(left_quad, left_ratio,
                                                        1.0)
        # 右侧四边形的点
        right_quad = np.array(
            [
                poly[point_num // 2 - 2], poly[point_num // 2 - 1],
                poly[point_num // 2], poly[point_num // 2 + 1]
            ],
            dtype=np.float32)
        # 计算右侧四边形的缩小比例
        right_ratio = 1.0 + \
                      shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                      (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
        # 根据缩小比例扩展右侧四边形
        right_quad_expand = self.shrink_quad_along_width(right_quad, 0.0,
                                                         right_ratio)
        # 更新多边形的点
        poly[0] = left_quad_expand[0]
        poly[-1] = left_quad_expand[-1]
        poly[point_num // 2 - 1] = right_quad_expand[1]
        poly[point_num // 2] = right_quad_expand[2]
        # 返回更新后的多边形
        return poly
    # 恢复四边形区域
    def restore_quad(self, tcl_map, tcl_map_thresh, tvo_map):
        """Restore quad."""
        # 找到文本区域的坐标
        xy_text = np.argwhere(tcl_map[:, :, 0] > tcl_map_thresh)
        xy_text = xy_text[:, ::-1]  # (n, 2)

        # 根据 y 轴对文本框进行排序
        xy_text = xy_text[np.argsort(xy_text[:, 1])]

        # 获取文本框的得分
        scores = tcl_map[xy_text[:, 1], xy_text[:, 0], 0]
        scores = scores[:, np.newaxis]

        # 恢复文本框
        point_num = int(tvo_map.shape[-1] / 2)
        assert point_num == 4
        tvo_map = tvo_map[xy_text[:, 1], xy_text[:, 0], :]
        xy_text_tile = np.tile(xy_text, (1, point_num))  # (n, point_num * 2)
        quads = xy_text_tile - tvo_map

        return scores, quads, xy_text

    # 计算四边形区域的面积
    def quad_area(self, quad):
        """
        compute area of a quad.
        """
        edge = [(quad[1][0] - quad[0][0]) * (quad[1][1] + quad[0][1]),
                (quad[2][0] - quad[1][0]) * (quad[2][1] + quad[1][1]),
                (quad[3][0] - quad[2][0]) * (quad[3][1] + quad[2][1]),
                (quad[0][0] - quad[3][0]) * (quad[0][1] + quad[3][1])]
        return np.sum(edge) / 2.

    # 非极大值抑制
    def nms(self, dets):
        if self.is_python35:
            from ppocr.utils.utility import check_install
            check_install('lanms', 'lanms-nova')
            import lanms
            # 使用 lanms 库进行非极大值抑制
            dets = lanms.merge_quadrangle_n9(dets, self.nms_thresh)
        else:
            # 使用自定义的非极大值抑制函数
            dets = nms_locality(dets, self.nms_thresh)
        return dets
    # 基于四边形将 tcl_map 中的像素进行聚类
    def cluster_by_quads_tco(self, tcl_map, tcl_map_thresh, quads, tco_map):
        """
        Cluster pixels in tcl_map based on quads.
        """
        # 计算实例数量，包括背景
        instance_count = quads.shape[0] + 1
        # 创建一个与 tcl_map 相同大小的全零矩阵，用于存储实例标签
        instance_label_map = np.zeros(tcl_map.shape[:2], dtype=np.int32)
        # 如果实例数量为 1，则直接返回结果
        if instance_count == 1:
            return instance_count, instance_label_map

        # 预测文本中心点
        xy_text = np.argwhere(tcl_map[:, :, 0] > tcl_map_thresh)
        n = xy_text.shape[0]
        xy_text = xy_text[:, ::-1]  # (n, 2)
        tco = tco_map[xy_text[:, 1], xy_text[:, 0], :]  # (n, 2)
        pred_tc = xy_text - tco

        # 获取真实文本中心点
        m = quads.shape[0]
        gt_tc = np.mean(quads, axis=1)  # (m, 2)

        # 复制预测文本中心点以匹配真实文本中心点的维度
        pred_tc_tile = np.tile(pred_tc[:, np.newaxis, :], (1, m, 1))  # (n, m, 2)
        gt_tc_tile = np.tile(gt_tc[np.newaxis, :, :], (n, 1, 1))  # (n, m, 2)
        # 计算预测文本中心点与真实文本中心点之间的距离矩阵
        dist_mat = np.linalg.norm(pred_tc_tile - gt_tc_tile, axis=2)  # (n, m)
        # 将每个预测文本中心点分配给最近的真实文本中心点
        xy_text_assign = np.argmin(dist_mat, axis=1) + 1  # (n,)

        # 将分配的实例标签填充到实例标签图中
        instance_label_map[xy_text[:, 1], xy_text[:, 0]] = xy_text_assign
        return instance_count, instance_label_map
    # 估算采样点数量
    def estimate_sample_pts_num(self, quad, xy_text):
        """
        Estimate sample points number.
        """
        # 计算四边形的高度
        eh = (np.linalg.norm(quad[0] - quad[3]) +
              np.linalg.norm(quad[1] - quad[2])) / 2.0
        # 计算四边形的宽度
        ew = (np.linalg.norm(quad[0] - quad[1]) +
              np.linalg.norm(quad[2] - quad[3])) / 2.0

        # 计算密集采样点的数量
        dense_sample_pts_num = max(2, int(ew))
        # 在文本坐标中获取密集采样点的中心线
        dense_xy_center_line = xy_text[np.linspace(
            0,
            xy_text.shape[0] - 1,
            dense_sample_pts_num,
            endpoint=True,
            dtype=np.float32).astype(np.int32)]

        # 计算密集采样点中心线的差异
        dense_xy_center_line_diff = dense_xy_center_line[
            1:] - dense_xy_center_line[:-1]
        # 估算弧长
        estimate_arc_len = np.sum(
            np.linalg.norm(
                dense_xy_center_line_diff, axis=1))

        # 计算最终采样点数量
        sample_pts_num = max(2, int(estimate_arc_len / eh))
        return sample_pts_num
    # 定义一个方法，用于处理模型输出结果和图像形状信息，返回检测到的多边形列表
    def __call__(self, outs_dict, shape_list):
        # 从输出字典中获取得分、边界、垂直偏移和水平偏移列表
        score_list = outs_dict['f_score']
        border_list = outs_dict['f_border']
        tvo_list = outs_dict['f_tvo']
        tco_list = outs_dict['f_tco']
        # 如果得分列表是 paddle.Tensor 类型，则转换为 numpy 数组
        if isinstance(score_list, paddle.Tensor):
            score_list = score_list.numpy()
            border_list = border_list.numpy()
            tvo_list = tvo_list.numpy()
            tco_list = tco_list.numpy()

        # 获取图像数量
        img_num = len(shape_list)
        # 初始化多边形列表
        poly_lists = []
        # 遍历每张图像
        for ino in range(img_num):
            # 调整得分、边界、垂直偏移和水平偏移的维度顺序
            p_score = score_list[ino].transpose((1, 2, 0))
            p_border = border_list[ino].transpose((1, 2, 0))
            p_tvo = tvo_list[ino].transpose((1, 2, 0))
            p_tco = tco_list[ino].transpose((1, 2, 0))
            # 获取当前图像的形状信息
            src_h, src_w, ratio_h, ratio_w = shape_list[ino]

            # 调用 detect_sast 方法进行文本检测，得到多边形列表
            poly_list = self.detect_sast(
                p_score,
                p_tvo,
                p_border,
                p_tco,
                ratio_w,
                ratio_h,
                src_w,
                src_h,
                shrink_ratio_of_width=self.shrink_ratio_of_width,
                tcl_map_thresh=self.tcl_map_thresh,
                offset_expand=self.expand_scale)
            # 将多边形列表添加到 poly_lists 中
            poly_lists.append({'points': np.array(poly_list)})

        # 返回多边形列表
        return poly_lists
```