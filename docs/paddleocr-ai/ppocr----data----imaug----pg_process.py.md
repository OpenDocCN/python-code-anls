# `.\PaddleOCR\ppocr\data\imaug\pg_process.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“按原样”分发，没有任何形式的担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入所需的库
import math
import cv2
import numpy as np
from skimage.morphology._skeletonize import thin
from ppocr.utils.e2e_utils.extract_textpoint_fast import sort_and_expand_with_direction_v2

# 定义 PGProcessTrain 类
__all__ = ['PGProcessTrain']

class PGProcessTrain(object):
    # 初始化方法
    def __init__(self,
                 character_dict_path,
                 max_text_length,
                 max_text_nums,
                 tcl_len,
                 batch_size=14,
                 use_resize=True,
                 use_random_crop=False,
                 min_crop_size=24,
                 min_text_size=4,
                 max_text_size=512,
                 point_gather_mode=None,
                 **kwargs):
        # 初始化参数
        self.tcl_len = tcl_len
        self.max_text_length = max_text_length
        self.max_text_nums = max_text_nums
        self.batch_size = batch_size
        # 如果使用随机裁剪，则设置最小裁剪尺寸和使用随机裁剪标志
        if use_random_crop is True:
            self.min_crop_size = min_crop_size
            self.use_random_crop = use_random_crop
        self.min_text_size = min_text_size
        self.max_text_size = max_text_size
        self.use_resize = use_resize
        self.point_gather_mode = point_gather_mode
        # 获取字符字典
        self.Lexicon_Table = self.get_dict(character_dict_path)
        self.pad_num = len(self.Lexicon_Table)
        self.img_id = 0
    def get_dict(self, character_dict_path):
        # 初始化字符字符串
        character_str = ""
        # 打开字符字典文件
        with open(character_dict_path, "rb") as fin:
            # 读取文件所有行
            lines = fin.readlines()
            # 遍历每一行
            for line in lines:
                # 解码每一行内容为 UTF-8 格式，并去除换行符
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                # 将每一行内容添加到字符字符串中
                character_str += line
            # 将字符字符串转换为字符列表
            dict_character = list(character_str)
        # 返回字符列表
        return dict_character

    def quad_area(self, poly):
        """
        compute area of a polygon
        :param poly: 多边形的顶点坐标
        :return: 多边形的面积
        """
        # 计算多边形的各边
        edge = [(poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
                (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
                (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
                (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])]
        # 返回多边形的面积
        return np.sum(edge) / 2.

    def gen_quad_from_poly(self, poly):
        """
        Generate min area quad from poly.
        :param poly: 多边形的顶点坐标
        :return: 最小面积四边形的顶点坐标
        """
        # 获取多边形的顶点数量
        point_num = poly.shape[0]
        # 初始化最小面积四边形的顶点坐标数组
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        # 获取最小外接矩形
        rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
        # 获取外接矩形的四个顶点
        box = np.array(cv2.boxPoints(rect))

        # 初始化第一个顶点索引和最小距离
        first_point_idx = 0
        min_dist = 1e4
        # 遍历四个顶点
        for i in range(4):
            # 计算当前四边形的周长
            dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                   np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                   np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                   np.linalg.norm(box[(i + 3) % 4] - poly[-1])
            # 更新最小距离和第一个顶点索引
            if dist < min_dist:
                min_dist = dist
                first_point_idx = i
        # 根据第一个顶点索引获取最小面积四边形的顶点坐标
        for i in range(4):
            min_area_quad[i] = box[(first_point_idx + i) % 4]

        # 返回最小面积四边形的顶点坐标
        return min_area_quad
    # 检查并验证文本多边形，确保它们朝向相同，并过滤一些无效的多边形
    def check_and_validate_polys(self, polys, tags, im_size):
        """
        check so that the text poly is in the same direction,
        and also filter some invalid polygons
        :param polys: 多边形的坐标
        :param tags: 多边形的标签
        :param im_size: 图像的尺寸
        :return: 经过验证的多边形、标签和水平/垂直标签
        """
        (h, w) = im_size
        # 如果多边形数量为0，则直接返回空数组
        if polys.shape[0] == 0:
            return polys, np.array([]), np.array([])
        # 将多边形的 x 坐标限制在图像宽度范围内
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        # 将多边形的 y 坐标限制在图像高度范围内
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

        validated_polys = []
        validated_tags = []
        hv_tags = []
        # 遍历每个多边形和其对应的标签
        for poly, tag in zip(polys, tags):
            # 从多边形生成四边形
            quad = self.gen_quad_from_poly(poly)
            # 计算四边形的面积
            p_area = self.quad_area(quad)
            # 如果面积小于1，则认为是无效多边形，跳过
            if abs(p_area) < 1:
                print('invalid poly')
                continue
            # 如果面积大于0
            if p_area > 0:
                # 如果标签为 False，则多边形朝向错误，将其翻转
                if tag == False:
                    print('poly in wrong direction')
                    tag = True  # reversed cases should be ignore
                # 调整多边形和四边形的顶点顺序
                poly = poly[(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                             1), :]
                quad = quad[(0, 3, 2, 1), :]

            # 计算四边形的宽度和高度
            len_w = np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] -
                                                                       quad[2])
            len_h = np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] -
                                                                       quad[2])
            hv_tag = 1

            # 如果宽度的两倍小于高度，则水平/垂直标签为0
            if len_w * 2.0 < len_h:
                hv_tag = 0

            # 将验证后的多边形、标签和水平/垂直标签添加到相应列表中
            validated_polys.append(poly)
            validated_tags.append(tag)
            hv_tags.append(hv_tag)
        # 返回经过验证的多边形、标签和水平/垂直标签的数组
        return np.array(validated_polys), np.array(validated_tags), np.array(
            hv_tags)
    # 生成方向映射，将多边形四边形列表转换为方向映射
    def generate_direction_map(self, poly_quads, n_char, direction_map):
        # 初始化宽度列表和高度列表
        width_list = []
        height_list = []
        # 遍历每个四边形
        for quad in poly_quads:
            # 计算四边形的宽度和高度
            quad_w = (np.linalg.norm(quad[0] - quad[1]) +
                      np.linalg.norm(quad[2] - quad[3])) / 2.0
            quad_h = (np.linalg.norm(quad[0] - quad[3]) +
                      np.linalg.norm(quad[2] - quad[1])) / 2.0
            # 将宽度和高度添加到对应的列表中
            width_list.append(quad_w)
            height_list.append(quad_h)
        # 计算规范化宽度和平均高度
        norm_width = max(sum(width_list) / n_char, 1.0)
        average_height = max(sum(height_list) / len(height_list), 1.0)
        k = 1
        # 遍历每个四边形
        for quad in poly_quads:
            # 计算方向向量
            direct_vector_full = (
                (quad[1] + quad[2]) - (quad[0] + quad[3])) / 2.0
            direct_vector = direct_vector_full / (
                np.linalg.norm(direct_vector_full) + 1e-6) * norm_width
            # 计算方向标签
            direction_label = tuple(
                map(float,
                    [direct_vector[0], direct_vector[1], 1.0 / average_height]))
            # 填充多边形区域的方向标签到方向映射中
            cv2.fillPoly(direction_map,
                         quad.round().astype(np.int32)[np.newaxis, :, :],
                         direction_label)
            k += 1
        # 返回方向映射
        return direction_map

    # 计算平均高度
    def calculate_average_height(self, poly_quads):
        # 初始化高度列表
        height_list = []
        # 遍历每个四边形
        for quad in poly_quads:
            # 计算四边形的高度
            quad_h = (np.linalg.norm(quad[0] - quad[3]) +
                      np.linalg.norm(quad[2] - quad[1])) / 2.0
            # 将高度添加到列表中
            height_list.append(quad_h)
        # 计算平均高度
        average_height = max(sum(height_list) / len(height_list), 1.0)
        # 返回平均高度
        return average_height
    # 调整多边形的顶点顺序
    def adjust_point(self, poly):
        """
        adjust point order.
        """
        # 获取多边形的顶点数量
        point_num = poly.shape[0]
        # 如果顶点数量为4
        if point_num == 4:
            # 计算多边形各边的长度
            len_1 = np.linalg.norm(poly[0] - poly[1])
            len_2 = np.linalg.norm(poly[1] - poly[2])
            len_3 = np.linalg.norm(poly[2] - poly[3])
            len_4 = np.linalg.norm(poly[3] - poly[0])

            # 如果第一条边和第三条边的长度之和乘以1.5小于第二条边和第四条边的长度之和
            if (len_1 + len_3) * 1.5 < (len_2 + len_4):
                # 调整顶点顺序
                poly = poly[[1, 2, 3, 0], :]

        # 如果顶点数量大于4
        elif point_num > 4:
            # 计算第一条边和第二条边的向量
            vector_1 = poly[0] - poly[1]
            vector_2 = poly[1] - poly[2]
            # 计算两向量夹角的余弦值
            cos_theta = np.dot(vector_1, vector_2) / (
                np.linalg.norm(vector_1) * np.linalg.norm(vector_2) + 1e-6)
            # 计算夹角的弧度值
            theta = np.arccos(np.round(cos_theta, decimals=4))

            # 如果夹角的绝对值大于70度
            if abs(theta) > (70 / 180 * math.pi):
                # 调整顶点顺序
                index = list(range(1, point_num)) + [0]
                poly = poly[np.array(index), :]
        # 返回调整后的多边形顶点
        return poly
    def gen_min_area_quad_from_poly(self, poly):
        """
        从多边形生成最小面积的四边形。
        """
        # 获取多边形的顶点数
        point_num = poly.shape[0]
        # 初始化最小面积四边形数组
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        # 如果多边形有4个顶点，则直接将多边形作为最小面积四边形
        if point_num == 4:
            min_area_quad = poly
            # 计算多边形中心点
            center_point = np.sum(poly, axis=0) / 4
        else:
            # 获取包围多边形的最小矩形
            rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
            center_point = rect[0]
            box = np.array(cv2.boxPoints(rect))

            first_point_idx = 0
            min_dist = 1e4
            # 寻找使得四边形边长最短的起始点
            for i in range(4):
                dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                       np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                       np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                       np.linalg.norm(box[(i + 3) % 4] - poly[-1])
                if dist < min_dist:
                    min_dist = dist
                    first_point_idx = i

            # 根据起始点重新排列四边形的顶点
            for i in range(4):
                min_area_quad[i] = box[(first_point_idx + i) % 4]

        return min_area_quad, center_point

    def shrink_quad_along_width(self,
                                quad,
                                begin_width_ratio=0.,
                                end_width_ratio=1.):
        """
        沿着宽度方向缩小四边形。
        """
        # 计算缩小比例
        ratio_pair = np.array(
            [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
        # 计算四边形两侧点的位置
        p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
        p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
        return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0])

    def prepare_text_label(self, label_str, Lexicon_Table):
        """
        根据给定的词典表准备文本标签。
        """
        # 如果词典表长度为36，则将标签转换为小写
        if len(Lexicon_Table) == 36:
            return label_str.lower()
        else:
            return label_str
    def vector_angle(self, A, B):
        """
        Calculate the angle between vector AB and x-axis positive direction.
        """
        # Calculate the vector AB
        AB = np.array([B[1] - A[1], B[0] - A[0])
        # Return the arctangent of AB
        return np.arctan2(*AB)

    def theta_line_cross_point(self, theta, point):
        """
        Calculate the line through given point and angle in ax + by + c =0 form.
        """
        x, y = point
        cos = np.cos(theta)
        sin = np.sin(theta)
        # Calculate the line equation in ax + by + c = 0 form
        return [sin, -cos, cos * y - sin * x]

    def line_cross_two_point(self, A, B):
        """
        Calculate the line through given point A and B in ax + by + c =0 form.
        """
        # Calculate the angle between points A and B
        angle = self.vector_angle(A, B)
        # Calculate the line equation in ax + by + c = 0 form
        return self.theta_line_cross_point(angle, A)

    def average_angle(self, poly):
        """
        Calculate the average angle between left and right edge in given poly.
        """
        p0, p1, p2, p3 = poly
        # Calculate the angles between points p3-p0 and p2-p1
        angle30 = self.vector_angle(p3, p0)
        angle21 = self.vector_angle(p2, p1)
        # Return the average of the two angles
        return (angle30 + angle21) / 2

    def line_cross_point(self, line1, line2):
        """
        line1 and line2 in  0=ax+by+c form, compute the cross point of line1 and line2
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        # Calculate the determinant
        d = a1 * b2 - a2 * b1

        if d == 0:
            print('Cross point does not exist')
            return np.array([0, 0], dtype=np.float32)
        else:
            # Calculate the cross point of the two lines
            x = (b1 * c2 - b2 * c1) / d
            y = (a2 * c1 - a1 * c2) / d

        return np.array([x, y], dtype=np.float32)

    def quad2tcl(self, poly, ratio):
        """
        Generate center line by poly clock-wise point. (4, 2)
        """
        ratio_pair = np.array(
            [[0.5 - ratio / 2], [0.5 + ratio / 2]], dtype=np.float32)
        # Calculate the center line based on the polygon's clock-wise points
        p0_3 = poly[0] + (poly[3] - poly[0]) * ratio_pair
        p1_2 = poly[1] + (poly[2] - poly[1]) * ratio_pair
        return np.array([p0_3[0], p1_2[0], p1_2[1], p0_3[1]])
    def poly2tcl(self, poly, ratio):
        """
        Generate center line by poly clock-wise point.
        """
        # 创建一个包含两个元素的数组，用于计算中心线
        ratio_pair = np.array(
            [[0.5 - ratio / 2], [0.5 + ratio / 2]], dtype=np.float32)
        # 创建一个与 poly 相同形状的零矩阵
        tcl_poly = np.zeros_like(poly)
        # 获取 poly 的点数
        point_num = poly.shape[0]

        # 遍历一半的点，生成中心线
        for idx in range(point_num // 2):
            # 计算两个点的中心线
            point_pair = poly[idx] + (poly[point_num - 1 - idx] - poly[idx]
                                      ) * ratio_pair
            tcl_poly[idx] = point_pair[0]
            tcl_poly[point_num - 1 - idx] = point_pair[1]
        return tcl_poly

    def gen_quad_tbo(self, quad, tcl_mask, tbo_map):
        """
        Generate tbo_map for give quad.
        """
        # 计算上下两条线的函数：ax + by + c = 0;
        up_line = self.line_cross_two_point(quad[0], quad[1])
        lower_line = self.line_cross_two_point(quad[3], quad[2])

        # 计算四边形的高和宽
        quad_h = 0.5 * (np.linalg.norm(quad[0] - quad[3]) +
                        np.linalg.norm(quad[1] - quad[2]))
        quad_w = 0.5 * (np.linalg.norm(quad[0] - quad[1]) +
                        np.linalg.norm(quad[2] - quad[3]))

        # 计算左右两条线的平均角度
        angle = self.average_angle(quad)

        # 获取二值化 mask 中为 1 的点的坐标
        xy_in_poly = np.argwhere(tcl_mask == 1)
        for y, x in xy_in_poly:
            point = (x, y)
            # 计算经过该点的直线
            line = self.theta_line_cross_point(angle, point)
            cross_point_upper = self.line_cross_point(up_line, line)
            cross_point_lower = self.line_cross_point(lower_line, line)
            ##FIX, offset reverse
            # 计算上下两条线与点的偏移
            upper_offset_x, upper_offset_y = cross_point_upper - point
            lower_offset_x, lower_offset_y = cross_point_lower - point
            # 将偏移信息存储到 tbo_map 中
            tbo_map[y, x, 0] = upper_offset_y
            tbo_map[y, x, 1] = upper_offset_x
            tbo_map[y, x, 2] = lower_offset_y
            tbo_map[y, x, 3] = lower_offset_x
            tbo_map[y, x, 4] = 1.0 / max(min(quad_h, quad_w), 1.0) * 2
        return tbo_map
    def poly2quads(self, poly):
        """
        Split poly into quads.
        """
        # 初始化一个空列表用于存储四边形
        quad_list = []
        # 获取多边形的顶点数
        point_num = poly.shape[0]

        # 存储点对
        point_pair_list = []
        # 遍历多边形的一半顶点数
        for idx in range(point_num // 2):
            # 获取对称的点对
            point_pair = [poly[idx], poly[point_num - 1 - idx]]
            point_pair_list.append(point_pair)

        # 计算四边形的数量
        quad_num = point_num // 2 - 1
        # 遍历四边形数量
        for idx in range(quad_num):
            # 重新排列并调整为顺时针方向
            quad_list.append((np.array(point_pair_list)[[idx, idx + 1]]
                              ).reshape(4, 2)[[0, 2, 3, 1]])

        return np.array(quad_list)

    def rotate_im_poly(self, im, text_polys):
        """
        rotate image with 90 / 180 / 270 degre
        """
        # 获取图像的宽度和高度
        im_w, im_h = im.shape[1], im.shape[0]
        # 复制图像
        dst_im = im.copy()
        dst_polys = []
        # 生成随机角度
        rand_degree_ratio = np.random.rand()
        rand_degree_cnt = 1
        if rand_degree_ratio > 0.5:
            rand_degree_cnt = 3
        # 根据随机角度旋转图像
        for i in range(rand_degree_cnt):
            dst_im = np.rot90(dst_im)
        rot_degree = -90 * rand_degree_cnt
        rot_angle = rot_degree * math.pi / 180.0
        n_poly = text_polys.shape[0]
        cx, cy = 0.5 * im_w, 0.5 * im_h
        ncx, ncy = 0.5 * dst_im.shape[1], 0.5 * dst_im.shape[0]
        # 遍历文本多边形
        for i in range(n_poly):
            wordBB = text_polys[i]
            poly = []
            # 遍历多边形的四个顶点
            for j in range(4):  # 16->4
                sx, sy = wordBB[j][0], wordBB[j][1]
                # 计算旋转后的顶点坐标
                dx = math.cos(rot_angle) * (sx - cx) - math.sin(rot_angle) * (
                    sy - cy) + ncx
                dy = math.sin(rot_angle) * (sx - cx) + math.cos(rot_angle) * (
                    sy - cy) + ncy
                poly.append([dx, dy])
            dst_polys.append(poly)
        return dst_im, np.array(dst_polys, dtype=np.float32)
```