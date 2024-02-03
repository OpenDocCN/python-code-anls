# `.\PaddleOCR\ppocr\utils\e2e_utils\extract_textpoint_slow.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”提供的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。
"""包含各种 CTC 解码器。"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入所需的库
import cv2
import math

import numpy as np
from itertools import groupby
from skimage.morphology._skeletonize import thin

# 从字符字典文件中获取字符列表
def get_dict(character_dict_path):
    character_str = ""
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            # 解码每行内容为 UTF-8 格式，并去除换行符
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            character_str += line
        # 将字符串转换为字符列表
        dict_character = list(character_str)
    return dict_character

# 将垂直点对转换为顺时针的多边形点
def point_pair2poly(point_pair_list):
    """
    Transfer vertical point_pairs into poly point in clockwise.
    """
    # 计算每个点对的长度信息
    pair_length_list = []
    for point_pair in point_pair_list:
        pair_length = np.linalg.norm(point_pair[0] - point_pair[1])
        pair_length_list.append(pair_length)
    pair_length_list = np.array(pair_length_list)
    pair_info = (pair_length_list.max(), pair_length_list.min(),
                 pair_length_list.mean())

    # 计算总点数
    point_num = len(point_pair_list) * 2
    point_list = [0] * point_num
    # 构建多边形点列表
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2), pair_info
# 缩小四边形沿宽度方向的函数，可以指定起始和结束宽度比例
def shrink_quad_along_width(quad, begin_width_ratio=0., end_width_ratio=1.):
    # 创建起始和结束宽度比例的数组
    ratio_pair = np.array(
        [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
    # 计算第一个点和第二个点的位置
    p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
    # 计算第三个点和第四个点的位置
    p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
    # 返回缩小后的四边形
    return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])


# 沿宽度方向扩展多边形的函数，可以指定缩小宽度的比例
def expand_poly_along_width(poly, shrink_ratio_of_width=0.3):
    # 获取多边形的点数
    point_num = poly.shape[0]
    # 获取左侧四边形
    left_quad = np.array(
        [poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
    # 计算左侧四边形的缩小比例
    left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                 (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    # 获取左侧四边形的扩展后的位置
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    # 获取右侧四边形
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
    # 获取右侧四边形的扩展后的位置
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    # 更新多边形的点位置
    poly[0] = left_quad_expand[0]
    poly[-1] = left_quad_expand[-1]
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2] = right_quad_expand[2]
    # 返回更新后的多边形
    return poly


# softmax 函数，用于计算 logits 的 softmax 分布
def softmax(logits):
    """
    logits: N x d
    """
    # 计算每行的最大值
    max_value = np.max(logits, axis=1, keepdims=True)
    # 计算指数
    exp = np.exp(logits - max_value)
    # 计算指数的和
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    # 计算 softmax 分布
    dist = exp / exp_sum
    # 返回 softmax 分布
    return dist


# 获取要保留项目的位置索引的函数
def get_keep_pos_idxs(labels, remove_blank=None):
    """
    Remove duplicate and get pos idxs of keep items.
    The value of keep_blank should be [None, 95].
    """
    # 初始化重复长度列表和要保留项目的位置索引列表
    duplicate_len_list = []
    keep_pos_idx_list = []
    # 用于存储保留字符的索引列表
    keep_char_idx_list = []
    # 根据标签进行分组，返回键和对应的值
    for k, v_ in groupby(labels):
        # 计算当前分组的长度
        current_len = len(list(v_))
        # 如果键不等于指定要移除的字符
        if k != remove_blank:
            # 计算当前字符的索引位置
            current_idx = int(sum(duplicate_len_list) + current_len // 2)
            # 将当前字符的索引位置添加到保留位置索引列表中
            keep_pos_idx_list.append(current_idx)
            # 将当前字符添加到保留字符索引列表中
            keep_char_idx_list.append(k)
        # 将当前分组长度添加到重复长度列表中
        duplicate_len_list.append(current_len)
    # 返回保留字符索引列表和保留位置索引列表
    return keep_char_idx_list, keep_pos_idx_list
# 从标签列表中移除指定的空白标签
def remove_blank(labels, blank=0):
    # 使用列表推导式生成一个新的标签列表，移除了指定的空白标签
    new_labels = [x for x in labels if x != blank]
    # 返回新的标签列表
    return new_labels


# 在标签列表中插入指定的空白标签
def insert_blank(labels, blank=0):
    # 创建一个新的标签列表，初始值为指定的空白标签
    new_labels = [blank]
    # 遍历原标签列表，在每个标签后插入指定的空白标签
    for l in labels:
        new_labels += [l, blank]
    # 返回新的标签列表
    return new_labels


# CTC贪婪解码器
def ctc_greedy_decoder(probs_seq, blank=95, keep_blank_in_idxs=True):
    """
    CTC贪婪（最佳路径）解码器。
    """
    # 获取每个时间步最大概率的标签
    raw_str = np.argmax(np.array(probs_seq), axis=1)
    # 根据参数决定是否保留空白标签的位置
    remove_blank_in_pos = None if keep_blank_in_idxs else blank
    # 获取去重后的标签序列和保留的索引列表
    dedup_str, keep_idx_list = get_keep_pos_idxs(
        raw_str, remove_blank=remove_blank_in_pos)
    # 移除空白标签，得到最终解码结果
    dst_str = remove_blank(dedup_str, blank=blank)
    # 返回最终解码结果和保留的索引列表
    return dst_str, keep_idx_list


# 单个实例的CTC贪婪解码器
def instance_ctc_greedy_decoder(gather_info, logits_map, keep_blank_in_idxs=True):
    """
    gather_info: [[x, y], [x, y] ...]
    logits_map: H x W X (n_chars + 1)
    """
    _, _, C = logits_map.shape
    ys, xs = zip(*gather_info)
    # 从logits_map中提取对应位置的logits序列
    logits_seq = logits_map[list(ys), list(xs)]  # n x 96
    # 计算概率序列
    probs_seq = softmax(logits_seq)
    # 使用CTC贪婪解码器获取解码结果和保留的索引列表
    dst_str, keep_idx_list = ctc_greedy_decoder(
        probs_seq, blank=C - 1, keep_blank_in_idxs=keep_blank_in_idxs)
    # 根据保留的索引列表获取保留的gather_info列表
    keep_gather_list = [gather_info[idx] for idx in keep_idx_list]
    # 返回解码结果和保留的gather_info列表
    return dst_str, keep_gather_list


# 图像的CTC解码器，使用多个进程
def ctc_decoder_for_image(gather_info_list, logits_map, keep_blank_in_idxs=True):
    """
    CTC解码器，使用多个进程。
    """
    decoder_results = []
    # 遍历gather_info_list，对每个gather_info进行解码
    for gather_info in gather_info_list:
        res = instance_ctc_greedy_decoder(
            gather_info, logits_map, keep_blank_in_idxs=keep_blank_in_idxs)
        decoder_results.append(res)
    # 返回解码结果列表
    return decoder_results


# 根据给定的方向对位置列表进行排序
def sort_with_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    # 根据给定的位置列表和方向列表对部分进行排序
    def sort_part_with_direction(pos_list, point_direction):
        # 将位置列表转换为二维数组
        pos_list = np.array(pos_list).reshape(-1, 2)
        # 将方向列表转换为二维数组
        point_direction = np.array(point_direction).reshape(-1, 2)
        # 计算平均方向
        average_direction = np.mean(point_direction, axis=0, keepdims=True)
        # 计算位置在平均方向上的投影长度
        pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
        # 根据投影长度对位置列表进行排序
        sorted_list = pos_list[np.argsort(pos_proj_leng)].tolist()
        # 根据排序后的投影长度对方向列表进行排序
        sorted_direction = point_direction[np.argsort(pos_proj_leng)].tolist()
        # 返回排序后的位置列表和方向列表
        return sorted_list, sorted_direction
    
    # 将位置列表转换为二维数组
    pos_list = np.array(pos_list).reshape(-1, 2)
    # 根据位置列表的坐标获取方向列表
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]  # x, y
    # 调整方向列表的顺序
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    # 对位置列表和方向列表进行排序
    sorted_point, sorted_direction = sort_part_with_direction(pos_list, point_direction)
    
    # 获取位置列表的长度
    point_num = len(sorted_point)
    # 如果位置列表长度大于等于16
    if point_num >= 16:
        # 计算中间位置
        middle_num = point_num // 2
        # 分割前半部分位置列表和方向列表
        first_part_point = sorted_point[:middle_num]
        first_point_direction = sorted_direction[:middle_num]
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(
            first_part_point, first_point_direction)
    
        # 分割后半部分位置列表和方向列表
        last_part_point = sorted_point[middle_num:]
        last_point_direction = sorted_direction[middle_num:]
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(
            last_part_point, last_point_direction)
        # 合并排序后的前半部分和后半部分位置列表
        sorted_point = sorted_fist_part_point + sorted_last_part_point
        # 合并排序后的前半部分和后半部分方向列表
        sorted_direction = sorted_fist_part_direction + sorted_last_part_direction
    
    # 返回排序后的位置列表和方向列表
    return sorted_point, np.array(sorted_direction)
# 为聚合特征添加 ID，用于推断
def add_id(pos_list, image_id=0):
    """
    Add id for gather feature, for inference.
    """
    # 创建一个新列表
    new_list = []
    # 遍历原始位置列表，为每个位置添加图像 ID，并组成新的列表
    for item in pos_list:
        new_list.append((image_id, item[0], item[1]))
    # 返回新列表
    return new_list


# 根据方向对位置列表进行排序和扩展
def sort_and_expand_with_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    # 获取方向数组的形状
    h, w, _ = f_direction.shape
    # 调用排序函数对位置列表进行排序，并返回排序后的列表和方向
    sorted_list, point_direction = sort_with_direction(pos_list, f_direction)

    # 沿着方向扩展
    point_num = len(sorted_list)
    sub_direction_len = max(point_num // 3, 2)
    left_direction = point_direction[:sub_direction_len, :]
    right_dirction = point_direction[point_num - sub_direction_len:, :]

    # 计算左侧平均方向和步长
    left_average_direction = -np.mean(left_direction, axis=0, keepdims=True)
    left_average_len = np.linalg.norm(left_average_direction)
    left_start = np.array(sorted_list[0])
    left_step = left_average_direction / (left_average_len + 1e-6)

    # 计算右侧平均方向和步长
    right_average_direction = np.mean(right_dirction, axis=0, keepdims=True)
    right_average_len = np.linalg.norm(right_average_direction)
    right_step = right_average_direction / (right_average_len + 1e-6)
    right_start = np.array(sorted_list[-1])

    # 计算追加的位置数量
    append_num = max(
        int((left_average_len + right_average_len) / 2.0 * 0.15), 1)
    left_list = []
    right_list = []
    # 遍历追加的位置，计算左侧和右侧的位置列表
    for i in range(append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ly < h and lx < w and (ly, lx) not in left_list:
            left_list.append((ly, lx))
        ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ry < h and rx < w and (ry, rx) not in right_list:
            right_list.append((ry, rx))

    # 组合左侧、排序后和右侧的位置列表
    all_list = left_list[::-1] + sorted_list + right_list
    return all_list


# 根据方向对位置列表进行排序和扩展，同时考虑二进制 TCl 映射
def sort_and_expand_with_direction_v2(pos_list, f_direction, binary_tcl_map):
    """
    f_direction: h x w x 2
    """
    # 定义一个包含位置坐标的列表，每个位置坐标为[y, x]
    # 定义一个二维数组，表示地图的高度和宽度
    """
    # 获取方向数组的形状信息，包括高度、宽度和通道数
    h, w, _ = f_direction.shape
    # 对位置列表进行排序，并返回排序后的列表和对应的方向信息
    sorted_list, point_direction = sort_with_direction(pos_list, f_direction)

    # 计算左侧点的数量
    point_num = len(sorted_list)
    sub_direction_len = max(point_num // 3, 2)
    left_direction = point_direction[:sub_direction_len, :]
    right_dirction = point_direction[point_num - sub_direction_len:, :]

    # 计算左侧平均方向和长度
    left_average_direction = -np.mean(left_direction, axis=0, keepdims=True)
    left_average_len = np.linalg.norm(left_average_direction)
    left_start = np.array(sorted_list[0])
    left_step = left_average_direction / (left_average_len + 1e-6)

    # 计算右侧平均方向和长度
    right_average_direction = np.mean(right_dirction, axis=0, keepdims=True)
    right_average_len = np.linalg.norm(right_average_direction)
    right_step = right_average_direction / (right_average_len + 1e-6)
    right_start = np.array(sorted_list[-1])

    # 计算追加点的数量
    append_num = max(
        int((left_average_len + right_average_len) / 2.0 * 0.15), 1)
    max_append_num = 2 * append_num

    # 初始化左侧和右侧点列表
    left_list = []
    right_list = []
    # 遍历左侧点列表，根据步长计算新的点坐标并添加到列表中
    for i in range(max_append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ly < h and lx < w and (ly, lx) not in left_list:
            if binary_tcl_map[ly, lx] > 0.5:
                left_list.append((ly, lx))
            else:
                break

    # 遍历右侧点列表，根据步长计算新的点坐标并添加到列表中
    for i in range(max_append_num):
        ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ry < h and rx < w and (ry, rx) not in right_list:
            if binary_tcl_map[ry, rx] > 0.5:
                right_list.append((ry, rx))
            else:
                break

    # 组合左侧、排序后的和右侧点列表，返回所有点的列表
    all_list = left_list[::-1] + sorted_list + right_list
    return all_list
# 生成 TCL 实例的中心点和端点列表，根据字符映射进行过滤
def generate_pivot_list_curved(p_score,
                               p_char_maps,
                               f_direction,
                               score_thresh=0.5,
                               is_expand=True,
                               is_backbone=False,
                               image_id=0):
    """
    返回 TCL 实例的中心点和端点；根据字符映射进行过滤
    """
    # 获取第一个元素的概率分数
    p_score = p_score[0]
    # 调整方向数组的维度顺序
    f_direction = f_direction.transpose(1, 2, 0)
    # 创建 TCL 实例地图，根据分数阈值进行筛选
    p_tcl_map = (p_score > score_thresh) * 1.0
    # 对 TCL 实例地图进行细化处理
    skeleton_map = thin(p_tcl_map)
    # 计算 TCL 实例的数量和实例标签地图
    instance_count, instance_label_map = cv2.connectedComponents(
        skeleton_map.astype(np.uint8), connectivity=8)

    # 获取 TCL 实例
    all_pos_yxs = []
    center_pos_yxs = []
    end_points_yxs = []
    instance_center_pos_yxs = []
    pred_strs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            ### FIX-ME, 消除异常值
            if len(pos_list) < 3:
                continue

            if is_expand:
                # 根据方向扩展和排序位置列表
                pos_list_sorted = sort_and_expand_with_direction_v2(
                    pos_list, f_direction, p_tcl_map)
            else:
                # 根据方向排序位置列表
                pos_list_sorted, _ = sort_with_direction(pos_list, f_direction)
            all_pos_yxs.append(pos_list_sorted)

    # 使用解码器过滤背景点
    p_char_maps = p_char_maps.transpose([1, 2, 0])
    decode_res = ctc_decoder_for_image(
        all_pos_yxs, logits_map=p_char_maps, keep_blank_in_idxs=True)
    # 遍历解码结果列表，其中每个元素包含解码后的字符串和保留的坐标列表
    for decoded_str, keep_yxs_list in decode_res:
        # 如果是主干网络
        if is_backbone:
            # 为保留的坐标列表添加ID，并将其添加到实例中心位置坐标列表中
            keep_yxs_list_with_id = add_id(keep_yxs_list, image_id=image_id)
            instance_center_pos_yxs.append(keep_yxs_list_with_id)
            # 将解码后的字符串添加到预测字符串列表中
            pred_strs.append(decoded_str)
        else:
            # 如果不是主干网络，将保留坐标列表的第一个和最后一个坐标添加到端点坐标列表中
            end_points_yxs.extend((keep_yxs_list[0], keep_yxs_list[-1]))
            # 将保留坐标列表添加到中心位置坐标列表中
            center_pos_yxs.extend(keep_yxs_list)

    # 如果是主干网络，返回预测字符串列表和实例中心位置坐标列表
    if is_backbone:
        return pred_strs, instance_center_pos_yxs
    else:
        # 如果不是主干网络，返回中心位置坐标列表和端点坐标列表
        return center_pos_yxs, end_points_yxs
# 生成水平方向的中心点和端点列表，根据得分、字符映射、方向过滤；默认得分阈值为0.5，不是主干结构，默认图像ID为0
def generate_pivot_list_horizontal(p_score,
                                   p_char_maps,
                                   f_direction,
                                   score_thresh=0.5,
                                   is_backbone=False,
                                   image_id=0):
    """
    返回 TCL 实例的中心点和端点；根据字符映射进行过滤；
    """
    # 获取得分的第一个元素
    p_score = p_score[0]
    # 调整方向的维度顺序
    f_direction = f_direction.transpose(1, 2, 0)
    # 根据得分阈值生成二值化 TCL 地图
    p_tcl_map_bi = (p_score > score_thresh) * 1.0
    # 计算 TCL 实例的数量和实例标签地图
    instance_count, instance_label_map = cv2.connectedComponents(
        p_tcl_map_bi.astype(np.uint8), connectivity=8)

    # 获取 TCL 实例
    all_pos_yxs = []
    center_pos_yxs = []
    end_points_yxs = []
    instance_center_pos_yxs = []
    # 如果实例数量大于0，则进入循环
    if instance_count > 0:
        # 遍历实例ID范围
        for instance_id in range(1, instance_count):
            # 初始化位置列表
            pos_list = []
            # 获取实例ID对应的坐标
            ys, xs = np.where(instance_label_map == instance_id)
            # 将坐标组成位置列表
            pos_list = list(zip(ys, xs))

            ### FIX-ME, 消除异常值
            # 如果位置列表长度小于5，则跳过当前实例
            if len(pos_list) < 5:
                continue

            # 添加规则
            # 提取主方向
            main_direction = extract_main_direction(pos_list, f_direction)  # y x
            # 设置参考方向
            reference_directin = np.array([0, 1]).reshape([-1, 2])  # y x
            # 判断是否为水平角度
            is_h_angle = abs(np.sum(
                main_direction * reference_directin)) < math.cos(math.pi / 180 *
                                                                 70)

            # 获取点的坐标
            point_yxs = np.array(pos_list)
            # 获取最大和最小的y、x坐标
            max_y, max_x = np.max(point_yxs, axis=0)
            min_y, min_x = np.min(point_yxs, axis=0)
            # 判断是否为水平长度
            is_h_len = (max_y - min_y) < 1.5 * (max_x - min_x)

            # 初始化最终位置列表
            pos_list_final = []
            # 如果是水平长度
            if is_h_len:
                # 获取唯一的x坐标
                xs = np.unique(xs)
                # 遍历x坐标
                for x in xs:
                    ys = instance_label_map[:, x].copy().reshape((-1, ))
                    y = int(np.where(ys == instance_id)[0].mean())
                    pos_list_final.append((y, x))
            else:
                # 获取唯一的y坐标
                ys = np.unique(ys)
                # 遍历y坐标
                for y in ys:
                    xs = instance_label_map[y, :].copy().reshape((-1, ))
                    x = int(np.where(xs == instance_id)[0].mean())
                    pos_list_final.append((y, x))

            # 根据方向对位置列表进行排序
            pos_list_sorted, _ = sort_with_direction(pos_list_final,
                                                     f_direction)
            # 将排序后的位置列表添加到所有位置列表中
            all_pos_yxs.append(pos_list_sorted)

    # 使用解码器过滤背景点
    # 调整p_char_maps的维度
    p_char_maps = p_char_maps.transpose([1, 2, 0])
    # 使用图像的CTC解码器
    decode_res = ctc_decoder_for_image(
        all_pos_yxs, logits_map=p_char_maps, keep_blank_in_idxs=True)
    # 遍历解码结果中的每个解码后的字符串和对应的保留的坐标列表
    for decoded_str, keep_yxs_list in decode_res:
        # 如果是主干网络
        if is_backbone:
            # 为保留的坐标列表添加ID，并将结果添加到实例中心位置坐标列表中
            keep_yxs_list_with_id = add_id(keep_yxs_list, image_id=image_id)
            instance_center_pos_yxs.append(keep_yxs_list_with_id)
        else:
            # 如果不是主干网络，将保留坐标列表的第一个和最后一个坐标添加到端点坐标列表中
            end_points_yxs.extend((keep_yxs_list[0], keep_yxs_list[-1]))
            # 将保留的坐标列表添加到中心位置坐标列表中
            center_pos_yxs.extend(keep_yxs_list)

    # 如果是主干网络，返回实例中心位置坐标列表
    if is_backbone:
        return instance_center_pos_yxs
    else:
        # 如果不是主干网络，返回中心位置坐标列表和端点坐标列表
        return center_pos_yxs, end_points_yxs
# 定义一个函数，根据给定参数生成一个枢轴列表，该函数运行较慢
def generate_pivot_list_slow(p_score,
                             p_char_maps,
                             f_direction,
                             score_thresh=0.5,
                             is_backbone=False,
                             is_curved=True,
                             image_id=0):
    """
    Warp all the function together.
    """
    # 如果 is_curved 为 True，则调用 generate_pivot_list_curved 函数
    if is_curved:
        return generate_pivot_list_curved(
            p_score,
            p_char_maps,
            f_direction,
            score_thresh=score_thresh,
            is_expand=True,
            is_backbone=is_backbone,
            image_id=image_id)
    # 否则调用 generate_pivot_list_horizontal 函数
    else:
        return generate_pivot_list_horizontal(
            p_score,
            p_char_maps,
            f_direction,
            score_thresh=score_thresh,
            is_backbone=is_backbone,
            image_id=image_id)


# 定义一个函数，用于提取主方向
def extract_main_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    # 将 pos_list 转换为 NumPy 数组
    pos_list = np.array(pos_list)
    # 从 f_direction 中获取指定位置的方向
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]
    # 将方向坐标进行转换，x, y -> y, x
    point_direction = point_direction[:, ::-1]
    # 计算平均方向
    average_direction = np.mean(point_direction, axis=0, keepdims=True)
    # 对平均方向进行归一化处理
    average_direction = average_direction / (
        np.linalg.norm(average_direction) + 1e-6)
    return average_direction


# 定义一个函数，根据方向和图像 ID 对位置列表进行排序（已弃用）
def sort_by_direction_with_image_id_deprecated(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[id, y, x], [id, y, x], [id, y, x] ...]
    """
    # 将 pos_list 转换为 NumPy 数组，并重新整形
    pos_list_full = np.array(pos_list).reshape(-1, 3)
    pos_list = pos_list_full[:, 1:]
    # 从 f_direction 中获取指定位置的方向
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]  # x, y
    # 将方向坐标进行转换，x, y -> y, x
    point_direction = point_direction[:, ::-1]
    # 计算平均方向
    average_direction = np.mean(point_direction, axis=0, keepdims=True)
    # 计算位置在平均方向上的投影长度
    pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
    # 根据投影长度对位置列表进行排序
    sorted_list = pos_list_full[np.argsort(pos_proj_leng)].tolist()
    # 返回排序后的列表
    return sorted_list
def sort_by_direction_with_image_id(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """

    # 定义一个内部函数，根据给定的方向对位置列表进行排序
    def sort_part_with_direction(pos_list_full, point_direction):
        # 将完整的位置列表转换为 NumPy 数组，并去除第一列
        pos_list_full = np.array(pos_list_full).reshape(-1, 3)
        pos_list = pos_list_full[:, 1:]
        # 将给定的方向转换为 NumPy 数组，并去除第一列
        point_direction = np.array(point_direction).reshape(-1, 2)
        # 计算平均方向
        average_direction = np.mean(point_direction, axis=0, keepdims=True)
        # 计算位置在平均方向上的投影长度
        pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
        # 根据投影长度对位置列表进行排序
        sorted_list = pos_list_full[np.argsort(pos_proj_leng)].tolist()
        sorted_direction = point_direction[np.argsort(pos_proj_leng)].tolist()
        return sorted_list, sorted_direction

    # 将位置列表转换为 NumPy 数组
    pos_list = np.array(pos_list).reshape(-1, 3)
    # 根据位置列表中的坐标获取对应方向
    point_direction = f_direction[pos_list[:, 1], pos_list[:, 2]]  # x, y
    # 调整方向的顺序，从 x, y 调整为 y, x
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    # 调用内部函数对位置列表进行排序
    sorted_point, sorted_direction = sort_part_with_direction(pos_list, point_direction)

    # 获取排序后的点的数量
    point_num = len(sorted_point)
    # 如果点的数量大于等于 16
    if point_num >= 16:
        # 计算中间点的数量
        middle_num = point_num // 2
        # 分割前半部分点和方向
        first_part_point = sorted_point[:middle_num]
        first_point_direction = sorted_direction[:middle_num]
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(
            first_part_point, first_point_direction)

        # 分割后半部分点和方向
        last_part_point = sorted_point[middle_num:]
        last_point_direction = sorted_direction[middle_num:]
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(
            last_part_point, last_point_direction)
        # 合并排序后的前半部分和后半部分点和方向
        sorted_point = sorted_fist_part_point + sorted_last_part_point
        sorted_direction = sorted_fist_part_direction + sorted_last_part_direction

    # 返回排序后的点
    return sorted_point
# 生成 TCL 实例的中心点和端点列表；根据字符映射进行过滤
def generate_pivot_list_tt_inference(p_score,
                                     p_char_maps,
                                     f_direction,
                                     score_thresh=0.5,
                                     is_backbone=False,
                                     is_curved=True,
                                     image_id=0):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    # 获取第一个元素的概率分数
    p_score = p_score[0]
    # 调整方向数组的维度顺序
    f_direction = f_direction.transpose(1, 2, 0)
    # 根据分数阈值生成 TCL 地图
    p_tcl_map = (p_score > score_thresh) * 1.0
    # 对 TCL 地图进行细化处理
    skeleton_map = thin(p_tcl_map)
    # 获取 TCL 实例的数量和实例标签地图
    instance_count, instance_label_map = cv2.connectedComponents(
        skeleton_map.astype(np.uint8), connectivity=8)

    # 获取 TCL 实例
    all_pos_yxs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            # 获取实例的坐标
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))
            ### FIX-ME, eliminate outlier
            # 如果坐标列表长度小于3，则跳过
            if len(pos_list) < 3:
                continue
            # 对坐标列表进行排序和扩展，根据方向和 TCL 地图
            pos_list_sorted = sort_and_expand_with_direction_v2(
                pos_list, f_direction, p_tcl_map)
            # 添加实例 ID，并将结果添加到列表中
            pos_list_sorted_with_id = add_id(pos_list_sorted, image_id=image_id)
            all_pos_yxs.append(pos_list_sorted_with_id)
    return all_pos_yxs
```