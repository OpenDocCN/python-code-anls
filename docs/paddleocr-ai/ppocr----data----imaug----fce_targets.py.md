# `.\PaddleOCR\ppocr\data\imaug\fce_targets.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发，基于"原样"的基础，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
"""
# 代码来源：
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/textdet_targets/fcenet_targets.py

import cv2
import numpy as np
from numpy.fft import fft
from numpy.linalg import norm
import sys

# 计算向量斜率
def vector_slope(vec):
    assert len(vec) == 2
    return abs(vec[1] / (vec[0] + 1e-8))

# FCENetTargets 类用于生成 FCENet 的地面实况目标：傅里叶轮廓嵌入
# 用于任意形状文本检测
# [https://arxiv.org/abs/2104.10442]
class FCENetTargets:
    """
    生成 FCENet 的地面实况目标：傅里叶轮廓嵌入
    用于任意形状文本检测

    Args:
        fourier_degree (int): 最大傅里叶变换度 k
        resample_step (float): 重新采样文本中心线（TCL）的步长
            最好不要超过最小宽度的一半
        center_region_shrink_ratio (float): 文本中心区域的收缩比例
        level_size_divisors (tuple(int)): 每个级别的下采样比率
        level_proportion_range (tuple(tuple(int))): 分配给每个级别的文本大小范围
    """
    # 初始化函数，设置默认参数和接收额外参数
    def __init__(self,
                 fourier_degree=5,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0)),
                 orientation_thr=2.0,
                 **kwargs):

        # 调用父类的初始化函数
        super().__init__()
        # 断言确保 level_size_divisors 是元组类型
        assert isinstance(level_size_divisors, tuple)
        # 断言确保 level_proportion_range 是元组类型
        assert isinstance(level_proportion_range, tuple)
        # 断言确保 level_size_divisors 和 level_proportion_range 长度相同
        assert len(level_size_divisors) == len(level_proportion_range)
        # 设置各个参数的值
        self.fourier_degree = fourier_degree
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range

        self.orientation_thr = orientation_thr

    # 计算两个向量之间的夹角
    def vector_angle(self, vec1, vec2):
        # 如果 vec1 是多维数组
        if vec1.ndim > 1:
            # 计算单位向量
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
        # 如果 vec2 是多维数组
        if vec2.ndim > 1:
            # 计算单位向量
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
        # 返回两个向量的夹角
        return np.arccos(
            np.clip(
                np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))
    def resample_line(self, line, n):
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """

        # 确保输入的线段是二维数组
        assert line.ndim == 2
        # 确保线段包含至少两个点
        assert line.shape[0] >= 2
        # 确保每个点是二维的
        assert line.shape[1] == 2
        # 确保 n 是整数且大于 0
        assert isinstance(n, int)
        assert n > 0

        # 计算线段中相邻点之间的距离
        length_list = [
            norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
        ]
        # 计算线段的总长度
        total_length = sum(length_list)
        # 计算累积长度
        length_cumsum = np.cumsum([0.0] + length_list)
        # 计算每个线段的长度
        delta_length = total_length / (float(n) + 1e-8)

        current_edge_ind = 0
        resampled_line = [line[0]]

        # 对每个点进行重新采样
        for i in range(1, n):
            current_line_len = i * delta_length

            # 找到当前点所在的线段
            while current_edge_ind + 1 < len(
                    length_cumsum) and current_line_len >= length_cumsum[
                        current_edge_ind + 1]:
                current_edge_ind += 1

            current_edge_end_shift = current_line_len - length_cumsum[
                current_edge_ind]

            # 计算当前点的位置
            if current_edge_ind >= len(length_list):
                break
            end_shift_ratio = current_edge_end_shift / length_list[
                current_edge_ind]
            current_point = line[current_edge_ind] + (line[current_edge_ind + 1]
                                                      - line[current_edge_ind]
                                                      ) * end_shift_ratio
            resampled_line.append(current_point)
        resampled_line.append(line[-1])
        resampled_line = np.array(resampled_line)

        return resampled_line
    def reorder_poly_edge(self, points):
        """重新排列文本多边形的边缘点，获取头部边缘、尾部边缘、顶部曲线边线和底部曲线边线的对应点。

        Args:
            points (ndarray): 组成文本多边形的点集。

        Returns:
            head_edge (ndarray): 组成文本多边形头部边缘的两个点。
            tail_edge (ndarray): 组成文本多边形尾部边缘的两个点。
            top_sideline (ndarray): 组成文本多边形顶部曲线边线的点集。
            bot_sideline (ndarray): 组成文本多边形底部曲线边线的点集。
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        # 找到头部和尾部边缘的索引
        head_inds, tail_inds = self.find_head_tail(points, self.orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        # 复制点集并拼接，以处理边缘跨越数组末尾的情况
        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        # 提取两个侧边线的点集
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        # 计算侧边线的平均偏移
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        # 根据侧边线的平均偏移确定顶部和底部曲线边线
        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        return head_edge, tail_edge, top_sideline, bot_sideline
    def resample_sidelines(self, sideline1, sideline2, resample_step):
        """Resample two sidelines to be of the same points number according to
        step size.

        Args:
            sideline1 (ndarray): The points composing a sideline of a text
                polygon.
            sideline2 (ndarray): The points composing another sideline of a
                text polygon.
            resample_step (float): The resampled step size.

        Returns:
            resampled_line1 (ndarray): The resampled line 1.
            resampled_line2 (ndarray): The resampled line 2.
        """

        # 检查两个侧线的维度是否为2
        assert sideline1.ndim == sideline2.ndim == 2
        # 检查两个侧线的每个点的维度是否为2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        # 检查侧线1的点数是否大于等于2
        assert sideline1.shape[0] >= 2
        # 检查侧线2的点数是否大于等于2
        assert sideline2.shape[0] >= 2
        # 检查resample_step是否为float类型
        assert isinstance(resample_step, float)

        # 计算侧线1的总长度
        length1 = sum([
            norm(sideline1[i + 1] - sideline1[i])
            for i in range(len(sideline1) - 1)
        ])
        # 计算侧线2的总长度
        length2 = sum([
            norm(sideline2[i + 1] - sideline2[i])
            for i in range(len(sideline2) - 1)
        ])

        # 计算两侧线的平均长度
        total_length = (length1 + length2) / 2
        # 根据给定的步长计算需要重新采样的点数
        resample_point_num = max(int(float(total_length) / resample_step), 1)

        # 对侧线1进行重新采样
        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        # 对侧线2进行重新采样
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        # 返回重新采样后的两侧线
        return resampled_line1, resampled_line2
    def resample_polygon(self, polygon, n=400):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        # 计算每个边的长度
        length = []

        # 遍历多边形的每个点
        for i in range(len(polygon)):
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]
            # 计算两点之间的距离并添加到长度列表中
            length.append(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)

        # 计算总长度
        total_length = sum(length)
        # 计算每条边上应该有多少个点
        n_on_each_line = (np.array(length) / (total_length + 1e-8)) * n
        n_on_each_line = n_on_each_line.astype(np.int32)
        new_polygon = []

        # 遍历多边形的每个点
        for i in range(len(polygon)):
            num = n_on_each_line[i]
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]

            if num == 0:
                continue

            # 计算每个点之间的增量
            dxdy = (p2 - p1) / num
            # 在每条边上均匀采样点
            for j in range(num):
                point = p1 + dxdy * j
                new_polygon.append(point)

        return np.array(new_polygon)

    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        # 将多边形平移到均值为原点
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        # 重新排列多边形，使起点在最右边
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon
    def poly2fourier(self, polygon, fourier_degree):
        """Perform Fourier transformation to generate Fourier coefficients ck
        from polygon.

        Args:
            polygon (ndarray): An input polygon.
            fourier_degree (int): The maximum Fourier degree K.
        Returns:
            c (ndarray(complex)): Fourier coefficients.
        """
        # 将多边形的点转换为复数形式
        points = polygon[:, 0] + polygon[:, 1] * 1j
        # 对点进行傅里叶变换并计算傅里叶系数
        c_fft = fft(points) / len(points)
        # 将傅里叶系数合并成对称形式
        c = np.hstack((c_fft[-fourier_degree:], c_fft[:fourier_degree + 1]))
        return c

    def clockwise(self, c, fourier_degree):
        """Make sure the polygon reconstructed from Fourier coefficients c in
        the clockwise direction.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon in clockwise point order.
        """
        # 检查傅里叶系数以确保多边形按顺时针方向重建
        if np.abs(c[fourier_degree + 1]) > np.abs(c[fourier_degree - 1]):
            return c
        elif np.abs(c[fourier_degree + 1]) < np.abs(c[fourier_degree - 1]):
            return c[::-1]
        else:
            if np.abs(c[fourier_degree + 2]) > np.abs(c[fourier_degree - 2]):
                return c
            else:
                return c[::-1]
    # 计算输入多边形的傅里叶描述符

    # 参数：
    #   polygon (ndarray): 输入的多边形
    #   fourier_degree (int): 最大傅里叶次数 K
    # 返回值：
    #   fourier_signature (ndarray): 一个形状为 (2k+1, 2) 的数组，包含了2k+1个傅里叶系数的实部和虚部

    # 对多边形进行重新采样
    resampled_polygon = self.resample_polygon(polygon)
    # 对重新采样后的多边形进行归一化处理
    resampled_polygon = self.normalize_polygon(resampled_polygon)

    # 将多边形转换为傅里叶系数
    fourier_coeff = self.poly2fourier(resampled_polygon, fourier_degree)
    # 对傅里叶系数进行顺时针处理
    fourier_coeff = self.clockwise(fourier_coeff, fourier_degree)

    # 提取傅里叶系数的实部并重塑为列向量
    real_part = np.real(fourier_coeff).reshape((-1, 1))
    # 提取傅里叶系数的虚部并重塑为列向量
    image_part = np.imag(fourier_coeff).reshape((-1, 1))
    # 将实部和虚部合并成傅里叶描述符
    fourier_signature = np.hstack([real_part, image_part])

    # 返回傅里叶描述符
    return fourier_signature
    def generate_fourier_maps(self, img_size, text_polys):
        """Generate Fourier coefficient maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            fourier_real_map (ndarray): The Fourier coefficient real part maps.
            fourier_image_map (ndarray): The Fourier coefficient image part
                maps.
        """

        assert isinstance(img_size, tuple)

        # 获取图像的高度和宽度
        h, w = img_size
        # 获取Fourier级数的阶数
        k = self.fourier_degree
        # 初始化Fourier系数的实部和虚部的数组
        real_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)
        imag_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)

        # 遍历每个文本多边形
        for poly in text_polys:
            # 创建一个与图像大小相同的全零掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            # 将多边形转换为OpenCV可以处理的格式，并在掩码上填充多边形区域
            polygon = np.array(poly).reshape((1, -1, 2))
            cv2.fillPoly(mask, polygon.astype(np.int32), 1)
            # 计算多边形的Fourier描述符
            fourier_coeff = self.cal_fourier_signature(polygon[0], k)
            # 遍历Fourier系数的每一项
            for i in range(-k, k + 1):
                if i != 0:
                    # 更新实部和虚部的值，根据掩码选择更新的位置
                    real_map[i + k, :, :] = mask * fourier_coeff[i + k, 0] + (
                        1 - mask) * real_map[i + k, :, :]
                    imag_map[i + k, :, :] = mask * fourier_coeff[i + k, 1] + (
                        1 - mask) * imag_map[i + k, :, :]
                else:
                    # 对于i=0的情况，更新特定位置的实部和虚部值
                    yx = np.argwhere(mask > 0.5)
                    k_ind = np.ones((len(yx)), dtype=np.int64) * k
                    y, x = yx[:, 0], yx[:, 1]
                    real_map[k_ind, y, x] = fourier_coeff[k, 0] - x
                    imag_map[k_ind, y, x] = fourier_coeff[k, 1] - y

        # 返回实部和虚部的Fourier系数数组
        return real_map, imag_map
    def generate_text_region_mask(self, img_size, text_polys):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)

        h, w = img_size
        # 创建一个与图像大小相同的全零数组，用于存储文本区域掩码
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            # 将文本多边形转换为 numpy 数组，并填充多边形内部为1
            polygon = np.array(poly, dtype=np.int32).reshape((1, -1, 2))
            cv2.fillPoly(text_region_mask, polygon, 1)

        return text_region_mask

    def generate_effective_mask(self, mask_size: tuple, polygons_ignore):
        """Generate effective mask by setting the ineffective regions to 0 and
        effective regions to 1.

        Args:
            mask_size (tuple): The mask size.
            polygons_ignore (list[[ndarray]]: The list of ignored text
                polygons.

        Returns:
            mask (ndarray): The effective mask of (height, width).
        """

        # 创建一个与掩码大小相同的全1数组，用于存储有效掩码
        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in polygons_ignore:
            # 将被忽略的文本多边形转换为 numpy 数组，并将其内部填充为0
            instance = poly.reshape(-1, 2).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)

        return mask
    # 为 FCENet 生成地面真实目标

    # 断言输入的结果是一个字典
    assert isinstance(results, dict)
    # 获取输入结果中的图像、多边形和忽略标签
    image = results['image']
    polygons = results['polys']
    ignore_tags = results['ignore_tags']
    # 获取图像的高度、宽度和通道数
    h, w, _ = image.shape

    # 初始化多边形掩码列表
    polygon_masks = []
    polygon_masks_ignore = []
    # 遍历忽略标签和多边形，根据标签将多边形分别添加到不同的列表中
    for tag, polygon in zip(ignore_tags, polygons):
        if tag is True:
            polygon_masks_ignore.append(polygon)
        else:
            polygon_masks.append(polygon)

    # 生成级别目标地图
    level_maps = self.generate_level_targets((h, w), polygon_masks,
                                             polygon_masks_ignore)

    # 构建映射关系，将生成的地图分别存储到对应的键中
    mapping = {
        'p3_maps': level_maps[0],
        'p4_maps': level_maps[1],
        'p5_maps': level_maps[2]
    }
    # 将生成的地图存储到结果字典中
    for key, value in mapping.items():
        results[key] = value

    # 返回更新后的结果字典
    return results

    # 调用函数时执行生成目标地图的操作
    def __call__(self, results):
        results = self.generate_targets(results)
        return results
```