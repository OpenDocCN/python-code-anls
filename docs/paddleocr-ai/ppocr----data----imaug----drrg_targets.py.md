# `.\PaddleOCR\ppocr\data\imaug\drrg_targets.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
# 代码来源于：
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/textdet_targets/drrg_targets.py

# 导入所需的库
import cv2
import numpy as np
from ppocr.utils.utility import check_install
from numpy.linalg import norm

# 定义 DRRGTargets 类
class DRRGTargets(object):
    # 初始化函数，设置各种参数的默认值
    def __init__(self,
                 orientation_thr=2.0,
                 resample_step=8.0,
                 num_min_comps=9,
                 num_max_comps=600,
                 min_width=8.0,
                 max_width=24.0,
                 center_region_shrink_ratio=0.3,
                 comp_shrink_ratio=1.0,
                 comp_w_h_ratio=0.3,
                 text_comp_nms_thr=0.25,
                 min_rand_half_height=8.0,
                 max_rand_half_height=24.0,
                 jitter_level=0.2,
                 **kwargs):

        # 调用父类的初始化函数
        super().__init__()
        # 设置各个参数的值
        self.orientation_thr = orientation_thr
        self.resample_step = resample_step
        self.num_max_comps = num_max_comps
        self.num_min_comps = num_min_comps
        self.min_width = min_width
        self.max_width = max_width
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.comp_shrink_ratio = comp_shrink_ratio
        self.comp_w_h_ratio = comp_w_h_ratio
        self.text_comp_nms_thr = text_comp_nms_thr
        self.min_rand_half_height = min_rand_half_height
        self.max_rand_half_height = max_rand_half_height
        self.jitter_level = jitter_level
        self.eps = 1e-8

    # 计算两个向量之间的夹角
    def vector_angle(self, vec1, vec2):
        # 如果向量维度大于1，则计算单位向量
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps)
        # 计算夹角并返回
        return np.arccos(
            np.clip(
                np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    # 计算向量的斜率
    def vector_slope(self, vec):
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + self.eps))

    # 计算向量的正弦值
    def vector_sin(self, vec):
        assert len(vec) == 2
        return vec[1] / (norm(vec) + self.eps)
    # 计算向量的余弦值
    def vector_cos(self, vec):
        # 确保向量长度为2
        assert len(vec) == 2
        # 返回向量的第一个元素除以向量的模长加上一个很小的数eps
        return vec[0] / (norm(vec) + self.eps)

    # 重新排序多边形的边
    def reorder_poly_edge(self, points):
        # 确保点的维度为2
        assert points.ndim == 2
        # 确保点的行数大于等于4
        assert points.shape[0] >= 4
        # 确保点的列数为2
        assert points.shape[1] == 2

        # 找到多边形的头部和尾部索引
        head_inds, tail_inds = self.find_head_tail(points, self.orientation_thr)
        # 获取头部边和尾部边
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        # 复制点数组并拼接成两倍长度的数组
        pad_points = np.vstack([points, points])
        # 如果尾部索引的第二个元素小于1，则将其设置为点的长度
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        # 获取侧线1和侧线2
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        # 计算侧线均值的偏移量
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        # 如果侧线均值的y坐标大于0，则将顶部侧线和底部侧线设置为侧线2和侧线1
        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        # 返回头部边、尾部边、顶部侧线和底部侧线
        return head_edge, tail_edge, top_sideline, bot_sideline

    # 计算曲线的长度
    def cal_curve_length(self, line):
        # 确保线的维度为2
        assert line.ndim == 2
        # 确保线的长度大于等于2
        assert len(line) >= 2

        # 计算每条边的长度
        edges_length = np.sqrt((line[1:, 0] - line[:-1, 0])**2 + (line[
            1:, 1] - line[:-1, 1])**2)
        # 计算总长度
        total_length = np.sum(edges_length)
        return edges_length, total_length
    # 对给定的线段进行重新采样，使其包含 n 个等距点
    def resample_line(self, line, n):

        # 断言线段是二维数组
        assert line.ndim == 2
        # 断言线段至少包含两个点
        assert line.shape[0] >= 2
        # 断言线段的每个点是二维的
        assert line.shape[1] == 2
        # 断言 n 是整数
        assert isinstance(n, int)
        # 断言 n 大于 2
        assert n > 2

        # 计算线段的边长和总长度
        edges_length, total_length = self.cal_curve_length(line)
        # 计算原始的参数化时间 t
        t_org = np.insert(np.cumsum(edges_length), 0, 0)
        # 计算单位参数化时间
        unit_t = total_length / (n - 1)
        # 计算等距参数化时间
        t_equidistant = np.arange(1, n - 1, dtype=np.float32) * unit_t
        edge_ind = 0
        points = [line[0]]
        # 对等距参数化时间进行遍历，插值得到新的点
        for t in t_equidistant:
            while edge_ind < len(edges_length) - 1 and t > t_org[edge_ind + 1]:
                edge_ind += 1
            t_l, t_r = t_org[edge_ind], t_org[edge_ind + 1]
            weight = np.array(
                [t_r - t, t - t_l], dtype=np.float32) / (t_r - t_l + self.eps)
            p_coords = np.dot(weight, line[[edge_ind, edge_ind + 1]])
            points.append(p_coords)
        points.append(line[-1])
        # 将插值得到的点组合成新的线段
        resampled_line = np.vstack(points)

        return resampled_line

    # 对给定的两个侧线进行重新采样，使其包含相同数量的等距点
    def resample_sidelines(self, sideline1, sideline2, resample_step):

        # 断言侧线是二维数组
        assert sideline1.ndim == sideline2.ndim == 2
        # 断言侧线的每个点是二维的
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        # 断言侧线至少包含两个点
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        # 断言 resample_step 是浮点数
        assert isinstance(resample_step, float)

        # 计算两个侧线的长度
        _, length1 = self.cal_curve_length(sideline1)
        _, length2 = self.cal_curve_length(sideline2)

        # 计算平均长度
        avg_length = (length1 + length2) / 2
        # 计算重新采样点的数量
        resample_point_num = max(int(float(avg_length) / resample_step) + 1, 3)

        # 对两个侧线分别进行重新采样
        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2

    # 计算点到直线的距离
    def dist_point2line(self, point, line):

        # 断言直线是一个包含两个点的元组
        assert isinstance(line, tuple)
        point1, point2 = line
        # 计算点到直线的距离
        d = abs(np.cross(point2 - point1, point - point1)) / (
            norm(point2 - point1) + 1e-8)
        return d
    def jitter_comp_attribs(self, comp_attribs, jitter_level):
        """Jitter text components attributes.

        Args:
            comp_attribs (ndarray): The text component attributes.
            jitter_level (float): The jitter level of text components
                attributes.

        Returns:
            jittered_comp_attribs (ndarray): The jittered text component
                attributes (x, y, h, w, cos, sin, comp_label).
        """

        # 检查文本组件属性的形状是否为 (n, 7)
        assert comp_attribs.shape[1] == 7
        # 检查文本组件属性的数量是否大于 0
        assert comp_attribs.shape[0] > 0
        # 检查 jitter_level 是否为 float 类型
        assert isinstance(jitter_level, float)

        # 提取文本组件属性中的 x 坐标
        x = comp_attribs[:, 0].reshape((-1, 1))
        # 提取文本组件属性中的 y 坐标
        y = comp_attribs[:, 1].reshape((-1, 1))
        # 提取文本组件属性中的高度
        h = comp_attribs[:, 2].reshape((-1, 1))
        # 提取文本组件属性中的宽度
        w = comp_attribs[:, 3].reshape((-1, 1))
        # 提取文本组件属性中的 cos 值
        cos = comp_attribs[:, 4].reshape((-1, 1))
        # 提取文本组件属性中的 sin 值
        sin = comp_attribs[:, 5].reshape((-1, 1))
        # 提取文本组件属性中的标签
        comp_labels = comp_attribs[:, 6].reshape((-1, 1))

        # 对 x 坐标进行抖动
        x += (np.random.random(size=(len(comp_attribs), 1)) - 0.5) * (
            h * np.abs(cos) + w * np.abs(sin)) * jitter_level
        # 对 y 坐标进行抖动
        y += (np.random.random(size=(len(comp_attribs), 1)) - 0.5) * (
            h * np.abs(sin) + w * np.abs(cos)) * jitter_level

        # 对高度进行抖动
        h += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
              ) * h * jitter_level
        # 对宽度进行抖动
        w += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
              ) * w * jitter_level

        # 对 cos 值进行抖动
        cos += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
                ) * 2 * jitter_level
        # 对 sin 值进行抖动
        sin += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
                ) * 2 * jitter_level

        # 计算缩放比例
        scale = np.sqrt(1.0 / (cos**2 + sin**2 + 1e-8))
        cos = cos * scale
        sin = sin * scale

        # 组合抖动后的文本组件属性
        jittered_comp_attribs = np.hstack([x, y, h, w, cos, sin, comp_labels])

        return jittered_comp_attribs
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
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            # 将文本多边形转换为 numpy 数组，并重塑形状
            polygon = np.array(poly, dtype=np.int32).reshape((1, -1, 2))
            # 使用多边形填充生成文本区域掩码
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
        # 创建全为1的有效掩码
        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in polygons_ignore:
            # 将忽略的文本多边形转换为 numpy 数组，并重塑形状
            instance = poly.astype(np.int32).reshape(1, -1, 2)
            # 使用多边形填充生成有效掩码
            cv2.fillPoly(mask, instance, 0)

        return mask
    # 生成 DRRG 的 gt 目标
    def generate_targets(self, data):
        """Generate the gt targets for DRRG.

        Args:
            data (dict): The input result dictionary.

        Returns:
            data (dict): The output result dictionary.
        """

        # 断言 data 是一个字典类型
        assert isinstance(data, dict)

        # 从输入结果字典中获取图像数据、多边形数据和忽略标签
        image = data['image']
        polygons = data['polys']
        ignore_tags = data['ignore_tags']
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

        # 生成文本区域掩码
        gt_text_mask = self.generate_text_region_mask((h, w), polygon_masks)
        # 生成有效掩码
        gt_mask = self.generate_effective_mask((h, w), polygon_masks_ignore)
        # 生成中心掩码属性映射
        (center_lines, gt_center_region_mask, gt_top_height_map,
         gt_bot_height_map, gt_sin_map,
         gt_cos_map) = self.generate_center_mask_attrib_maps((h, w),
                                                             polygon_masks)

        # 生成组合属性
        gt_comp_attribs = self.generate_comp_attribs(
            center_lines, gt_text_mask, gt_center_region_mask,
            gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map)

        # 构建映射关系字典
        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_top_height_map': gt_top_height_map,
            'gt_bot_height_map': gt_bot_height_map,
            'gt_sin_map': gt_sin_map,
            'gt_cos_map': gt_cos_map
        }

        # 更新结果字典
        data.update(mapping)
        data['gt_comp_attribs'] = gt_comp_attribs
        return data

    # 调用函数，生成目标
    def __call__(self, data):
        data = self.generate_targets(data)
        return data
```