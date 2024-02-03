# `.\PaddleOCR\ppocr\postprocess\fce_postprocess.py`

```
# 版权声明
# 2022年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制。
"""
# 代码来源
# https://github.com/open-mmlab/mmocr/blob/v0.3.0/mmocr/models/textdet/postprocess/wrapper.py

import cv2
import paddle
import numpy as np
from numpy.fft import ifft
from ppocr.utils.poly_nms import poly_nms, valid_boundary

# 填充孔洞函数
def fill_hole(input_mask):
    # 获取输入mask的高度和宽度
    h, w = input_mask.shape
    # 创建一个高度和宽度各增加2的全零矩阵
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    # 将输入mask复制到新创建的矩阵中间
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    # 创建一个高度和宽度各增加4的全零矩阵
    mask = np.zeros((h + 4, w + 4), np.uint8)

    # 使用洪泛填充算法填充孔洞
    cv2.floodFill(canvas, mask, (0, 0), 1)
    # 将填充后的结果截取到原始大小，并转换为布尔类型
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool_)

    # 返回填充后的结果
    return ~canvas | input_mask

# 傅里叶系数转多边形函数
def fourier2poly(fourier_coeff, num_reconstr_points=50):
    """ Inverse Fourier transform
        Args:
            fourier_coeff (ndarray): Fourier coefficients shaped (n, 2k+1),
                with n and k being candidates number and Fourier degree
                respectively.
            num_reconstr_points (int): Number of reconstructed polygon points.
        Returns:
            Polygons (ndarray): The reconstructed polygons shaped (n, n')
        """

    # 创建一个复数类型的数组用于存储傅里叶系数
    a = np.zeros((len(fourier_coeff), num_reconstr_points), dtype='complex')
    # 计算傅里叶系数的一半
    k = (len(fourier_coeff[0]) - 1) // 2

    # 将傅里叶系数重新排列
    a[:, 0:k + 1] = fourier_coeff[:, k:]
    a[:, -k:] = fourier_coeff[:, :k]

    # 进行逆傅里叶变换并乘以重构点数
    poly_complex = ifft(a) * num_reconstr_points
    # 创建一个用于存储多边形的数组
    polygon = np.zeros((len(fourier_coeff), num_reconstr_points, 2)
    # 将 poly_complex 数组中的实部赋值给 polygon 数组的第一维
    polygon[:, :, 0] = poly_complex.real
    # 将 poly_complex 数组中的虚部赋值给 polygon 数组的第二维
    polygon[:, :, 1] = poly_complex.imag
    # 将 polygon 数组转换为 int32 类型，并重新调整形状为(len(fourier_coeff), -1)，然后返回
    return polygon.astype('int32').reshape((len(fourier_coeff), -1))
class FCEPostProcess(object):
    """
    The post process for FCENet.
    """

    def __init__(self,
                 scales,
                 fourier_degree=5,
                 num_reconstr_points=50,
                 decoding_type='fcenet',
                 score_thr=0.3,
                 nms_thr=0.1,
                 alpha=1.0,
                 beta=1.0,
                 box_type='poly',
                 **kwargs):

        # 初始化 FCEPostProcess 类的属性
        self.scales = scales
        self.fourier_degree = fourier_degree
        self.num_reconstr_points = num_reconstr_points
        self.decoding_type = decoding_type
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.box_type = box_type

    def __call__(self, preds, shape_list):
        # 初始化 score_maps 列表
        score_maps = []
        # 遍历 preds 字典中的键值对
        for key, value in preds.items():
            # 如果值是 paddle.Tensor 类型，则转换为 numpy 数组
            if isinstance(value, paddle.Tensor):
                value = value.numpy()
            # 获取分类结果和回归结果
            cls_res = value[:, :4, :, :]
            reg_res = value[:, 4:, :, :]
            # 将分类结果和回归结果添加到 score_maps 列表中
            score_maps.append([cls_res, reg_res])

        # 调用 get_boundary 方法处理 score_maps 和 shape_list
        return self.get_boundary(score_maps, shape_list)

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.

        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
            with size 2k+1 with k>=4.
            scale_factor(ndarray): The scale factor of size (4,).

        Returns:
            boundaries (list[list[float]]): The scaled boundaries.
        """
        # 初始化 boxes 和 scores 列表
        boxes = []
        scores = []
        # 遍历 boundaries 列表
        for b in boundaries:
            sz = len(b)
            # 调用 valid_boundary 函数验证边界
            valid_boundary(b, True)
            # 将最后一个元素作为分数添加到 scores 列表中
            scores.append(b[-1])
            # 根据 scale_factor 对边界进行缩放
            b = (np.array(b[:sz - 1]) *
                 (np.tile(scale_factor[:2], int(
                     (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
            # 将缩放后的边界添加到 boxes 列表中
            boxes.append(np.array(b).reshape([-1, 2]))

        # 返回缩放后的 boxes 和 scores
        return np.array(boxes, dtype=np.float32), scores
    # 获取边界框的函数，输入为分数图和形状列表
    def get_boundary(self, score_maps, shape_list):
        # 确保分数图的数量与比例尺的数量相同
        assert len(score_maps) == len(self.scales)
        # 初始化边界框列表
        boundaries = []
        # 遍历每个分数图
        for idx, score_map in enumerate(score_maps):
            # 获取当前比例尺
            scale = self.scales[idx]
            # 调用私有方法获取单个分数图的边界框
            boundaries = boundaries + self._get_boundary_single(score_map,
                                                                scale)

        # 非极大值抑制
        boundaries = poly_nms(boundaries, self.nms_thr)
        # 调整边界框大小
        boundaries, scores = self.resize_boundary(
            boundaries, (1 / shape_list[0, 2:]).tolist()[::-1])

        # 将边界框和分数组成字典列表
        boxes_batch = [dict(points=boundaries, scores=scores)]
        # 返回字典列表
        return boxes_batch

    # 获取单个分数图的边界框
    def _get_boundary_single(self, score_map, scale):
        # 确保分数图的长度为2
        assert len(score_map) == 2
        # 确保第二个分数图的形状符合要求
        assert score_map[1].shape[1] == 4 * self.fourier_degree + 2

        # 调用FCENet解码函数，返回边界框
        return self.fcenet_decode(
            preds=score_map,
            fourier_degree=self.fourier_degree,
            num_reconstr_points=self.num_reconstr_points,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            box_type=self.box_type,
            score_thr=self.score_thr,
            nms_thr=self.nms_thr)
```