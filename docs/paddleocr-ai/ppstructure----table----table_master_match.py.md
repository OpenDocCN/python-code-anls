# `.\PaddleOCR\ppstructure\table\table_master_match.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/match.py

import os
import re
import cv2
import glob
import copy
import math
import pickle
import numpy as np

from shapely.geometry import Polygon, MultiPoint
"""
匹配中的有用函数
"""

def remove_empty_bboxes(bboxes):
    """
    移除结构化主要边界框中的 [0., 0., 0., 0.]
    bboxes.shape 的长度必须为 2
    :param bboxes: 边界框
    :return: 移除空边界框后的结果
    """
    new_bboxes = []
    for bbox in bboxes:
        if sum(bbox) == 0.:
            continue
        new_bboxes.append(bbox)
    return np.array(new_bboxes)


def xywh2xyxy(bboxes):
    if len(bboxes.shape) == 1:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[0] = bboxes[0] - bboxes[2] / 2
        new_bboxes[1] = bboxes[1] - bboxes[3] / 2
        new_bboxes[2] = bboxes[0] + bboxes[2] / 2
        new_bboxes[3] = bboxes[1] + bboxes[3] / 2
        return new_bboxes
    elif len(bboxes.shape) == 2:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
        return new_bboxes
    else:
        raise ValueError


def xyxy2xywh(bboxes):
    # 待实现
    # 如果边界框的维度为1，则进行以下操作
    if len(bboxes.shape) == 1:
        # 创建一个与原始边界框相同形状的新边界框
        new_bboxes = np.empty_like(bboxes)
        # 计算新边界框的中心点横坐标
        new_bboxes[0] = bboxes[0] + (bboxes[2] - bboxes[0]) / 2
        # 计算新边界框的中心点纵坐标
        new_bboxes[1] = bboxes[1] + (bboxes[3] - bboxes[1]) / 2
        # 计算新边界框的宽度
        new_bboxes[2] = bboxes[2] - bboxes[0]
        # 计算新边界框的高度
        new_bboxes[3] = bboxes[3] - bboxes[1]
        # 返回新边界框
        return new_bboxes
    # 如果边界框的维度为2，则进行以下操作
    elif len(bboxes.shape) == 2:
        # 创建一个与原始边界框相同形状的新边界框
        new_bboxes = np.empty_like(bboxes)
        # 计算新边界框每个边界框的中心点横坐标
        new_bboxes[:, 0] = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2
        # 计算新边界框每个边界框的中心点纵坐标
        new_bboxes[:, 1] = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2
        # 计算新边界框每个边界框的宽度
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        # 计算新边界框每个边界框的高度
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        # 返回新边界框
        return new_bboxes
    # 如果边界框的维度既不为1也不为2，则抛出数值错误异常
    else:
        raise ValueError
# 从指定路径加载 pickle 文件，如果是文件则直接加载，如果是目录则加载目录下所有以指定前缀开头的 pickle 文件并合并数据
def pickle_load(path, prefix='end2end'):
    # 如果路径是文件
    if os.path.isfile(path):
        # 读取 pickle 文件内容
        data = pickle.load(open(path, 'rb'))
    # 如果路径是目录
    elif os.path.isdir(path):
        # 初始化数据字典
        data = dict()
        # 搜索目录下所有以指定前缀开头的 pickle 文件
        search_path = os.path.join(path, '{}_*.pkl'.format(prefix))
        pkls = glob.glob(search_path)
        # 遍历所有找到的 pickle 文件
        for pkl in pkls:
            # 读取 pickle 文件内容
            this_data = pickle.load(open(pkl, 'rb'))
            # 合并数据到总数据字典
            data.update(this_data)
    else:
        # 抛出数值错误异常
        raise ValueError
    # 返回加载的数据
    return data


# 将两点坐标格式转换为四点坐标格式
def convert_coord(xyxy):
    """
    Convert two points format to four points format.
    :param xyxy:
    :return:
    """
    # 初始化四点坐标数组
    new_bbox = np.zeros([4, 2], dtype=np.float32)
    # 赋值四个点的坐标
    new_bbox[0, 0], new_bbox[0, 1] = xyxy[0], xyxy[1]
    new_bbox[1, 0], new_bbox[1, 1] = xyxy[2], xyxy[1]
    new_bbox[2, 0], new_bbox[2, 1] = xyxy[2], xyxy[3]
    new_bbox[3, 0], new_bbox[3, 1] = xyxy[0], xyxy[3]
    # 返回四点坐标数组
    return new_bbox


# 计算两个边界框的 IoU（交并比）
def cal_iou(bbox1, bbox2):
    # 将边界框转换为多边形
    bbox1_poly = Polygon(bbox1).convex_hull
    bbox2_poly = Polygon(bbox2).convex_hull
    union_poly = np.concatenate((bbox1, bbox2))

    # 如果两个多边形不相交，则 IoU 为 0
    if not bbox1_poly.intersects(bbox2_poly):
        iou = 0
    else:
        # 计算交集面积和并集面积
        inter_area = bbox1_poly.intersection(bbox2_poly).area
        union_area = MultiPoint(union_poly).convex_hull.area
        # 计算 IoU
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area
    return iou


# 计算两点之间的欧氏距离
def cal_distance(p1, p2):
    # 计算两点在 x 和 y 方向上的差值
    delta_x = p1[0] - p2[0]
    delta_y = p1[1] - p2[1]
    # 计算欧氏距离
    d = math.sqrt((delta_x**2) + (delta_y**2))
    return d


# 判断中心点是否在边界框内部
def is_inside(center_point, corner_point):
    """
    Find if center_point inside the bbox(corner_point) or not.
    :param center_point: center point (x, y)
    :param corner_point: corner point ((x1,y1),(x2,y2))
    :return:
    """
    x_flag = False
    y_flag = False
    # 判断中心点的 x 坐标是否在边界框的 x 范围内
    if (center_point[0] >= corner_point[0][0]) and (
            center_point[0] <= corner_point[1][0]):
        x_flag = True
    # 检查中心点的 y 坐标是否在矩形的上下边界之间
    if (center_point[1] >= corner_point[0][1]) and (
            center_point[1] <= corner_point[1][1]):
        y_flag = True
    # 如果 x_flag 和 y_flag 都为 True，则返回 True
    if x_flag and y_flag:
        return True
    # 否则返回 False
    else:
        return False
# 查找在之前匹配列表中没有匹配的 end2end 边框框
def find_no_match(match_list, all_end2end_nums, type='end2end'):
    """
    Find out no match end2end bbox in previous match list.
    :param match_list: matching pairs.
    :param all_end2end_nums: numbers of end2end_xywh
    :param type: 'end2end' corresponding to idx 0, 'master' corresponding to idx 1.
    :return: no match pse bbox index list
    """
    # 根据 type 确定 idx 的值
    if type == 'end2end':
        idx = 0
    elif type == 'master':
        idx = 1
    else:
        raise ValueError

    no_match_indexs = []
    # 获取已匹配边框的索引
    matched_bbox_indexs = [m[idx] for m in match_list]
    # 遍历所有 end2end 边框的索引，找出未匹配的边框
    for n in range(all_end2end_nums):
        if n not in matched_bbox_indexs:
            no_match_indexs.append(n)
    return no_match_indexs


# 判断两个边框的 y 轴坐标差值是否小于阈值
def is_abs_lower_than_threshold(this_bbox, target_bbox, threshold=3):
    # 只考虑 y 轴坐标，用于在同一行进行分组
    delta = abs(this_bbox[1] - target_bbox[1])
    if delta < threshold:
        return True
    else:
        return False


# 对同一行（组）中的边框进行排序
def sort_line_bbox(g, bg):
    """
    Sorted the bbox in the same line(group)
    compare coord 'x' value, where 'y' value is closed in the same group.
    :param g: index in the same group
    :param bg: bbox in the same group
    :return:
    """
    # 获取所有边框的 x 坐标值并排序
    xs = [bg_item[0] for bg_item in bg]
    xs_sorted = sorted(xs)

    g_sorted = [None] * len(xs_sorted)
    bg_sorted = [None] * len(xs_sorted)
    # 根据排序后的 x 坐标值重新排列边框和索引
    for g_item, bg_item in zip(g, bg):
        idx = xs_sorted.index(bg_item[0])
        bg_sorted[idx] = bg_item
        g_sorted[idx] = g_item

    return g_sorted, bg_sorted


# 将排序后的组和边框组展开成索引和边框列表
def flatten(sorted_groups, sorted_bbox_groups):
    idxs = []
    bboxes = []
    for group, bbox_group in zip(sorted_groups, sorted_bbox_groups):
        for g, bg in zip(group, bbox_group):
            idxs.append(g)
            bboxes.append(bg)
    return idxs, bboxes


# 对 end2end_xywh_bboxes 中未匹配的边框进行排序和分组
def sort_bbox(end2end_xywh_bboxes, no_match_end2end_indexes):
    """
    This function will group the render end2end bboxes in row.
    """
    :param end2end_xywh_bboxes: 传入的端到端框的坐标信息
    :param no_match_end2end_indexes: 传入的未匹配的端到端框的索引
    :return: 返回排序后的索引列表、排序后的边界框列表、排序后的组列表、排序后的边界框组列表
    """
    # 初始化空列表用于存储组和边界框组
    groups = []
    bbox_groups = []
    # 遍历未匹配的端到端框的索引和坐标信息
    for index, end2end_xywh_bbox in zip(no_match_end2end_indexes, end2end_xywh_bboxes):
        this_bbox = end2end_xywh_bbox
        # 如果组列表为空，则创建新的组和边界框组
        if len(groups) == 0:
            groups.append([index])
            bbox_groups.append([this_bbox])
        else:
            flag = False
            # 遍历已有的组和边界框组
            for g, bg in zip(groups, bbox_groups):
                # 判断当前边界框是否属于该行
                if is_abs_lower_than_threshold(this_bbox, bg[0]):
                    g.append(index)
                    bg.append(this_bbox)
                    flag = True
                    break
            if not flag:
                # 如果当前边界框不属于任何行，则创建新的行
                groups.append([index])
                bbox_groups.append([this_bbox])

    # 对每个组内的边界框进行排序
    tmp_groups, tmp_bbox_groups = [], []
    for g, bg in zip(groups, bbox_groups):
        g_sorted, bg_sorted = sort_line_bbox(g, bg)
        tmp_groups.append(g_sorted)
        tmp_bbox_groups.append(bg_sorted)

    # 按照 y 坐标值对组进行排序
    sorted_groups = [None] * len(tmp_groups)
    sorted_bbox_groups = [None] * len(tmp_bbox_groups)
    ys = [bg[0][1] for bg in tmp_bbox_groups]
    sorted_ys = sorted(ys)
    for g, bg in zip(tmp_groups, tmp_bbox_groups):
        idx = sorted_ys.index(bg[0][1])
        sorted_groups[idx] = g
        sorted_bbox_groups[idx] = bg

    # 展开列表，获取最终结果
    end2end_sorted_idx_list, end2end_sorted_bbox_list = flatten(sorted_groups, sorted_bbox_groups)

    return end2end_sorted_idx_list, end2end_sorted_bbox_list, sorted_groups, sorted_bbox_groups
# 定义一个函数，用于将end2end结果和structure master结果转换为xyxy格式的bbox列表和xywh格式的bbox列表
def get_bboxes_list(end2end_result, structure_master_result):
    """
    This function is use to convert end2end results and structure master results to
    List of xyxy bbox format and List of xywh bbox format
    :param end2end_result: bbox's format is xyxy
    :param structure_master_result: bbox's format is xywh
    :return: 4 kind list of bbox ()
    """
    # end2end
    end2end_xyxy_list = []  # 存储xyxy格式的bbox列表
    end2end_xywh_list = []  # 存储xywh格式的bbox列表
    for end2end_item in end2end_result:
        src_bbox = end2end_item['bbox']  # 获取end2end结果中的bbox
        end2end_xyxy_list.append(src_bbox)  # 将bbox添加到xyxy格式的bbox列表中
        xywh_bbox = xyxy2xywh(src_bbox)  # 将xyxy格式的bbox转换为xywh格式的bbox
        end2end_xywh_list.append(xywh_bbox)  # 将转换后的bbox添加到xywh格式的bbox列表中
    end2end_xyxy_bboxes = np.array(end2end_xyxy_list)  # 将xyxy格式的bbox列表转换为numpy数组
    end2end_xywh_bboxes = np.array(end2end_xywh_list)  # 将xywh格式的bbox列表转换为numpy数组

    # structure master
    src_bboxes = structure_master_result['bbox']  # 获取structure master结果中的bbox
    src_bboxes = remove_empty_bboxes(src_bboxes)  # 移除空的bbox
    structure_master_xyxy_bboxes = src_bboxes  # 将bbox赋值给xyxy格式的bbox列表
    xywh_bbox = xyxy2xywh(src_bboxes)  # 将bbox转换为xywh格式的bbox
    structure_master_xywh_bboxes = xywh_bbox  # 将转换后的bbox赋值给xywh格式的bbox列表

    return end2end_xyxy_bboxes, end2end_xywh_bboxes, structure_master_xywh_bboxes, structure_master_xyxy_bboxes


# 定义一个函数，用于判断end2end Bbox的中心点是否在structure master Bbox内，如果在，则获取匹配对
def center_rule_match(end2end_xywh_bboxes, structure_master_xyxy_bboxes):
    """
    Judge end2end Bbox's center point is inside structure master Bbox or not,
    if end2end Bbox's center is in structure master Bbox, get matching pair.
    :param end2end_xywh_bboxes:
    :param structure_master_xyxy_bboxes:
    :return: match pairs list, e.g. [[0,1], [1,2], ...]
    """
    match_pairs_list = []  # 存储匹配对列表
    # 遍历end2end_xywh_bboxes列表中的每个元素，同时获取索引和值
    for i, end2end_xywh in enumerate(end2end_xywh_bboxes):
        # 遍历structure_master_xyxy_bboxes列表中的每个元素，同时获取索引和值
        for j, master_xyxy in enumerate(structure_master_xyxy_bboxes):
            # 解构end2end_xywh元组，获取x和y坐标
            x_end2end, y_end2end = end2end_xywh[0], end2end_xywh[1]
            # 解构master_xyxy元组，获取四个坐标值
            x_master1, y_master1, x_master2, y_master2 \
                = master_xyxy[0], master_xyxy[1], master_xyxy[2], master_xyxy[3]
            # 计算end2end的中心点坐标
            center_point_end2end = (x_end2end, y_end2end)
            # 计算master的对角点坐标
            corner_point_master = ((x_master1, y_master1),
                                   (x_master2, y_master2))
            # 如果end2end的中心点在master的对角点内部
            if is_inside(center_point_end2end, corner_point_master):
                # 将匹配的索引对添加到match_pairs_list列表中
                match_pairs_list.append([i, j])
    # 返回匹配对列表
    return match_pairs_list
# 使用 IOU 计算规则匹配 end2end_xyxy_bboxes 和 structure_master_xyxy_bboxes 中的边界框
def iou_rule_match(end2end_xyxy_bboxes, end2end_xyxy_indexes,
                   structure_master_xyxy_bboxes):
    # 初始化匹配对列表
    match_pair_list = []
    # 遍历原始 end2end 索引和边界框
    for end2end_xyxy_index, end2end_xyxy in zip(end2end_xyxy_indexes,
                                                end2end_xyxy_bboxes):
        # 初始化最大 IOU 值和匹配对
        max_iou = 0
        max_match = [None, None]
        # 遍历结构主边界框
        for j, master_xyxy in enumerate(structure_master_xyxy_bboxes):
            # 转换坐标格式
            end2end_4xy = convert_coord(end2end_xyxy)
            master_4xy = convert_coord(master_xyxy)
            # 计算 IOU
            iou = cal_iou(end2end_4xy, master_4xy)
            # 更新最大 IOU 值和匹配对
            if iou > max_iou:
                max_match[0], max_match[1] = end2end_xyxy_index, j
                max_iou = iou

        # 如果没有匹配，则继续下一个循环
        if max_match[0] is None:
            continue
        # 将匹配对添加到匹配列表中
        match_pair_list.append(max_match)
    # 返回匹配对列表
    return match_pair_list


# 使用最小距离规则匹配 end2end_bboxes 和 master_bboxes 中的边界框
def distance_rule_match(end2end_indexes, end2end_bboxes, master_indexes,
                        master_bboxes):
    # 初始化最小距离匹配列表
    min_match_list = []
    # 遍历主要索引和主要边界框
    for j, master_bbox in zip(master_indexes, master_bboxes):
        # 初始化最小距离为无穷大
        min_distance = np.inf
        # 初始化最小匹配为 [0, 0]，表示索引 i 和 j
        min_match = [0, 0]  # i, j
        # 遍历端到端索引和端到端边界框
        for i, end2end_bbox in zip(end2end_indexes, end2end_bboxes):
            # 获取端到端边界框的 x 和 y 坐标
            x_end2end, y_end2end = end2end_bbox[0], end2end_bbox[1]
            # 获取主要边界框的 x 和 y 坐标
            x_master, y_master = master_bbox[0], master_bbox[1]
            # 创建端到端点和主要点的元组
            end2end_point = (x_end2end, y_end2end)
            master_point = (x_master, y_master)
            # 计算主要点和端到端点之间的距离
            dist = cal_distance(master_point, end2end_point)
            # 如果距离小于最小距离，则更新最小匹配和最小距离
            if dist < min_distance:
                min_match[0], min_match[1] = i, j
                min_distance = dist
        # 将最小匹配添加到列表中
        min_match_list.append(min_match)
    # 返回最小匹配列表
    return min_match_list
# 创建一些虚拟的主 bbox，并与未匹配的 end2end 索引进行匹配
def extra_match(no_match_end2end_indexes, master_bbox_nums):
    # 计算虚拟主 bbox 的数量
    end_nums = len(no_match_end2end_indexes) + master_bbox_nums
    # 存储额外匹配的列表
    extra_match_list = []
    # 遍历创建额外匹配的索引对
    for i in range(master_bbox_nums, end_nums):
        end2end_index = no_match_end2end_indexes[i - master_bbox_nums]
        extra_match_list.append([end2end_index, i])
    return extra_match_list


# 将匹配列表转换为字典，其中键是主 bbox 的索引，值是 end2end bbox 的索引
def get_match_dict(match_list):
    match_dict = dict()
    # 遍历匹配列表，将索引对添加到字典中
    for match_pair in match_list:
        end2end_index, master_index = match_pair[0], match_pair[1]
        if master_index not in match_dict.keys():
            match_dict[master_index] = [end2end_index]
        else:
            match_dict[master_index].append(end2end_index)
    return match_dict


# 处理文本中连续的空格字符
def deal_successive_space(text):
    # 替换连续的三个空格为 '<space>'，表示真实的空格
    text = text.replace(' ' * 3, '<space>')
    # 移除空格，这些空格是分隔符，不是真正的空格
    text = text.replace(' ', '')
    # 将 '<space>' 替换为真实的空格
    text = text.replace('<space>', ' ')
    return text


# 将文本列表中的重复部分合并为一个元素
def reduce_repeat_bb(text_list, break_token):
    count = 0
    for text in text_list:
        if text.startswith('<b>'):
            count += 1
    # 如果计数等于文本列表的长度
    if count == len(text_list):
        # 创建一个新的文本列表
        new_text_list = []
        # 遍历原始文本列表中的每个文本
        for text in text_list:
            # 替换文本中的 '<b>' 和 '</b>' 为空字符串
            text = text.replace('<b>', '').replace('</b>', '')
            # 将处理后的文本添加到新的文本列表中
            new_text_list.append(text)
        # 返回一个包含所有处理后文本的列表，用断点标记连接
        return ['<b>' + break_token.join(new_text_list) + '</b>']
    else:
        # 如果计数不等于文本列表的长度，则直接返回原始文本列表
        return text_list
# 根据匹配字典和端到端信息生成匹配文本字典
def get_match_text_dict(match_dict, end2end_info, break_token=' '):
    # 初始化匹配文本字典
    match_text_dict = dict()
    # 遍历匹配字典，获取主索引和端到端索引列表
    for master_index, end2end_index_list in match_dict.items():
        # 从端到端信息中获取文本列表
        text_list = [
            end2end_info[end2end_index]['text']
            for end2end_index in end2end_index_list
        ]
        # 调用函数去除重复的文本，并使用指定的分隔符连接文本列表
        text_list = reduce_repeat_bb(text_list, break_token)
        text = break_token.join(text_list)
        # 将主索引和对应的文本添加到匹配文本字典中
        match_text_dict[master_index] = text
    # 返回匹配文本字典
    return match_text_dict


def merge_span_token(master_token_list):
    """
    Merge the span style token (row span or col span).
    :param master_token_list: 主令牌列表
    :return: 无
    """
    # 初始化新的主令牌列表
    new_master_token_list = []
    # 初始化指针
    pointer = 0
    # 如果主令牌列表的最后一个元素不是 '</tbody>'，则添加 '</tbody>' 到主令牌列表中
    if master_token_list[-1] != '</tbody>':
        master_token_list.append('</tbody>')
    # 当指针指向的元素不是 '</tbody>' 时，执行循环
    while master_token_list[pointer] != '</tbody>':
        try:
            # 如果当前元素是 '<td'，则进入条件判断
            if master_token_list[pointer] == '<td':
                # 如果下一个元素以 ' colspan=' 或 ' rowspan=' 开头，则执行以下操作
                if master_token_list[pointer + 1].startswith(
                        ' colspan=') or master_token_list[
                            pointer + 1].startswith(' rowspan='):
                    """
                    example:
                    pattern <td colspan="3">
                    '<td' + 'colspan=" "' + '>' + '</td>'
                    """
                    # 将当前元素及其后两个元素拼接成字符串，加入到新列表中
                    tmp = ''.join(master_token_list[pointer:pointer + 3 + 1])
                    pointer += 4
                    new_master_token_list.append(tmp)

                # 如果第二个元素以 ' colspan=' 或 ' rowspan=' 开头，则执行以下操作
                elif master_token_list[pointer + 2].startswith(
                        ' colspan=') or master_token_list[
                            pointer + 2].startswith(' rowspan='):
                    """
                    example:
                    pattern <td rowspan="2" colspan="3">
                    '<td' + 'rowspan=" "' + 'colspan=" "' + '>' + '</td>'
                    """
                    # 将当前元素及其后四个元素拼接成字符串，加入到新列表中
                    tmp = ''.join(master_token_list[pointer:pointer + 4 + 1])
                    pointer += 5
                    new_master_token_list.append(tmp)

                else:
                    # 如果不满足上述条件，则将当前元素加入到新列表中
                    new_master_token_list.append(master_token_list[pointer])
                    pointer += 1
            else:
                # 如果当前元素不是 '<td'，则将其加入到新列表中
                new_master_token_list.append(master_token_list[pointer])
                pointer += 1
        except:
            # 捕获异常并打印信息
            print("Break in merge...")
            # 跳出循环
            break
    # 将 '</tbody>' 加入到新列表中
    new_master_token_list.append('</tbody>')

    # 返回处理后的新列表
    return new_master_token_list
# 处理带有 <eb></eb>, <eb1></eb1>, ... 标记的文本，替换成对应的 <td></td>, <td> </td>, ... 标记
def deal_eb_token(master_token):
    # 定义空白标记字典，包含不同情况下的标记替换关系
    emptyBboxTokenDict = {
        "[]": '<eb></eb>',
        "[' ']": '<eb1></eb1>',
        "['<b>', ' ', '</b>']": '<eb2></eb2>',
        "['\\u2028', '\\u2028']": '<eb3></eb3>',
        "['<sup>', ' ', '</sup>']": '<eb4></eb4>',
        "['<b>', '</b>']": '<eb5></eb5>',
        "['<i>', ' ', '</i>']": '<eb6></eb6>',
        "['<b>', '<i>', '</i>', '</b>']": '<eb7></eb7>',
        "['<b>', '<i>', ' ', '</i>', '</b>']": '<eb8></eb8>',
        "['<i>', '</i>']": '<eb9></eb9>',
        "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']": '<eb10></eb10>',
    }
    # 替换文本中的标记
    master_token = master_token.replace('<eb></eb>', '<td></td>')
    master_token = master_token.replace('<eb1></eb1>', '<td> </td>')
    master_token = master_token.replace('<eb2></eb2>', '<td><b> </b></td>')
    master_token = master_token.replace('<eb3></eb3>', '<td>\u2028\u2028</td>')
    master_token = master_token.replace('<eb4></eb4>', '<td><sup> </sup></td>')
    master_token = master_token.replace('<eb5></eb5>', '<td><b></b></td>')
    master_token = master_token.replace('<eb6></eb6>', '<td><i> </i></td>')
    master_token = master_token.replace('<eb7></eb7>', '<td><b><i></i></b></td>')
    master_token = master_token.replace('<eb8></eb8>', '<td><b><i> </i></b></td>')
    master_token = master_token.replace('<eb9></eb9>', '<td><i></i></td>')
    master_token = master_token.replace('<eb10></eb10>', '<td><b> \u2028 \u2028 </b></td>')
    return master_token

# 将 OCR 文本结果插入到结构化标记中
def insert_text_to_token(master_token_list, match_text_dict):
    # 合并连续的标记
    master_token_list = merge_span_token(master_token_list)
    # 初始化合并结果列表
    merged_result_list = []
    # 初始化文本计数器
    text_count = 0
    # 遍历主标记列表中的每个标记
    for master_token in master_token_list:
        # 检查标记是否以'<td'开头
        if master_token.startswith('<td'):
            # 如果文本计数器大于匹配文本字典的长度减1，则增加计数器并继续下一次循环
            if text_count > len(match_text_dict) - 1:
                text_count += 1
                continue
            # 如果文本计数器不在匹配文本字典的键中，则增加计数器并继续下一次循环
            elif text_count not in match_text_dict.keys():
                text_count += 1
                continue
            else:
                # 替换标记中的文本内容为匹配文本字典中对应的值
                master_token = master_token.replace(
                    '><', '>{}<'.format(match_text_dict[text_count]))
                text_count += 1
        # 处理特殊标记
        master_token = deal_eb_token(master_token)
        # 将处理后的标记添加到合并结果列表中
        merged_result_list.append(master_token)

    # 返回合并后的结果字符串
    return ''.join(merged_result_list)
# 处理孤立的跨度标记，这是由结构识别模型中的错误预测引起的。
# 例如，将 <td rowspan="2"></td> 预测为 <td></td> rowspan="2"></b></td>。
def deal_isolate_span(thead_part):
    # 1. 查找孤立的跨度标记。
    isolate_pattern = "<td></td> rowspan=\"(\d)+\" colspan=\"(\d)+\"></b></td>|" \
                      "<td></td> colspan=\"(\d)+\" rowspan=\"(\d)+\"></b></td>|" \
                      "<td></td> rowspan=\"(\d)+\"></b></td>|" \
                      "<td></td> colspan=\"(\d)+\"></b></td>"
    isolate_iter = re.finditer(isolate_pattern, thead_part)
    isolate_list = [i.group() for i in isolate_iter]

    # 2. 根据步骤 1 的结果找出跨度数。
    span_pattern = " rowspan=\"(\d)+\" colspan=\"(\d)+\"|" \
                   " colspan=\"(\d)+\" rowspan=\"(\d)+\"|" \
                   " rowspan=\"(\d)+\"|" \
                   " colspan=\"(\d)+\""
    corrected_list = []
    for isolate_item in isolate_list:
        span_part = re.search(span_pattern, isolate_item)
        spanStr_in_isolateItem = span_part.group()
        # 3. 将跨度数合并到跨度标记格式字符串中。
        if spanStr_in_isolateItem is not None:
            corrected_item = '<td{}></td>'.format(spanStr_in_isolateItem)
            corrected_list.append(corrected_item)
        else:
            corrected_list.append(None)

    # 4. 替换原始的孤立标记。
    for corrected_item, isolate_item in zip(corrected_list, isolate_list):
        if corrected_item is not None:
            thead_part = thead_part.replace(isolate_item, corrected_item)
        else:
            pass
    return thead_part


# 处理替换后的重复的 <b> 或 </b>。
# 在 <td></td> 标记中保留一个 <b></b>。
def deal_duplicate_bb(thead_part):
    # 1. 在 <thead></thead> 中查找 <td></td>。
    # 定义匹配 <td></td> 标签的正则表达式模式，包括不同的 rowspan 和 colspan 情况
    td_pattern = "<td rowspan=\"(\d)+\" colspan=\"(\d)+\">(.+?)</td>|" \
                 "<td colspan=\"(\d)+\" rowspan=\"(\d)+\">(.+?)</td>|" \
                 "<td rowspan=\"(\d)+\">(.+?)</td>|" \
                 "<td colspan=\"(\d)+\">(.+?)</td>|" \
                 "<td>(.*?)</td>"
    # 使用正则表达式在 thead_part 中查找所有匹配的 <td></td> 标签
    td_iter = re.finditer(td_pattern, thead_part)
    # 将匹配到的 <td></td> 标签存储在列表中
    td_list = [t.group() for t in td_iter]

    # 2. is multiply <b></b> in <td></td> or not?
    # 遍历每个 <td></td> 标签，检查是否存在多个 <b></b> 标签
    new_td_list = []
    for td_item in td_list:
        if td_item.count('<b>') > 1 or td_item.count('</b>') > 1:
            # 存在多个 <b></b> 标签的情况
            # 1. 移除所有 <b></b> 标签
            td_item = td_item.replace('<b>', '').replace('</b>', '')
            # 2. 替换 <td> 标签为 <td><b>，</td> 标签为 </b></td>
            td_item = td_item.replace('<td>', '<td><b>').replace('</td>',
                                                                 '</b></td>')
            new_td_list.append(td_item)
        else:
            new_td_list.append(td_item)

    # 3. replace original thead part.
    # 替换原始 thead_part 中的 <td></td> 标签为处理后的新标签
    for td_item, new_td_item in zip(td_list, new_td_list):
        thead_part = thead_part.replace(td_item, new_td_item)
    # 返回处理后的 thead_part
    return thead_part
def deal_bb(result_token):
    """
    In our opinion, <b></b> always occurs in <thead></thead> text's context.
    This function will find out all tokens in <thead></thead> and insert <b></b> by manual.
    :param result_token:
    :return:
    """
    # find out <thead></thead> parts.
    thead_pattern = '<thead>(.*?)</thead>'
    # 如果结果中没有匹配到<thead></thead>部分，则直接返回结果
    if re.search(thead_pattern, result_token) is None:
        return result_token
    # 获取匹配到的<thead></thead>部分
    thead_part = re.search(thead_pattern, result_token).group()
    # 复制一份原始的<thead></thead>部分
    origin_thead_part = copy.deepcopy(thead_part)

    # check "rowspan" or "colspan" occur in <thead></thead> parts or not .
    # 检查<thead></thead>部分是否包含"rowspan"或"colspan"
    span_pattern = "<td rowspan=\"(\d)+\" colspan=\"(\d)+\">|<td colspan=\"(\d)+\" rowspan=\"(\d)+\">|<td rowspan=\"(\d)+\">|<td colspan=\"(\d)+\">"
    span_iter = re.finditer(span_pattern, thead_part)
    span_list = [s.group() for s in span_iter]
    # 判断<thead></thead>部分是否包含"rowspan"或"colspan"
    has_span_in_head = True if len(span_list) > 0 else False

    if not has_span_in_head:
        # <thead></thead> not include "rowspan" or "colspan" branch 1.
        # 1. replace <td> to <td><b>, and </td> to </b></td>
        # 2. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b></b> to </b>
        # 如果<thead></thead>部分不包含"rowspan"或"colspan"，则进行以下操作：
        # 1. 将<td>替换为<td><b>，将</td>替换为</b></td>
        # 2. 通过文本行识别，可能预测文本中包含<b>或</b>，因此将<b><b>替换为<b>，将</b></b>替换为</b>
        thead_part = thead_part.replace('<td>', '<td><b>')\
            .replace('</td>', '</b></td>')\
            .replace('<b><b>', '<b>')\
            .replace('</b></b>', '</b>')
    else:
        # 如果不是第一种情况，则处理包含 "rowspan" 或 "colspan" 的第二种情况
        # 首先，处理 rowspan 或 colspan 的情况
        # 1. 将 > 替换为 ><b>
        # 2. 将 </td> 替换为 </b></td>
        # 3. 可以通过文本行识别预测文本是否包含 <b> 或 </b>，因此将 <b><b> 替换为 <b>，将 </b><b> 替换为 </b>

        # 其次，处理类似第一种情况的普通情况

        # 将 ">" 替换为 "<b>"
        replaced_span_list = []
        for sp in span_list:
            replaced_span_list.append(sp.replace('>', '><b>'))
        for sp, rsp in zip(span_list, replaced_span_list):
            thead_part = thead_part.replace(sp, rsp)

        # 将 "</td>" 替换为 "</b></td>"
        thead_part = thead_part.replace('</td>', '</b></td>')

        # 通过 re.sub 去除重复的 <b>
        mb_pattern = "(<b>)+"
        single_b_string = "<b>"
        thead_part = re.sub(mb_pattern, single_b_string, thead_part)

        mgb_pattern = "(</b>)+"
        single_gb_string = "</b>"
        thead_part = re.sub(mgb_pattern, single_gb_string, thead_part)

        # 处理类似第一种情况的普通情况
        thead_part = thead_part.replace('<td>', '<td><b>').replace('<b><b>',
                                                                   '<b>')

    # 将 <tb><b></b></tb> 转换回 <tb></tb>，空单元格没有 <b></b>
    # 但是空格单元格（<tb> </tb>）适合 <td><b> </b></td>
    thead_part = thead_part.replace('<td><b></b></td>', '<td></td>')
    # 处理重复的 <b></b>
    thead_part = deal_duplicate_bb(thead_part)
    # 处理由结构预测错误导致的孤立的 span 标记
    # 例如：PMC5994107_011_00.png
    thead_part = deal_isolate_span(thead_part)
    # 用新的 thead 部分替换原始结果
    result_token = result_token.replace(origin_thead_part, thead_part)
    return result_token
class Matcher:
    def __init__(self, end2end_file, structure_master_file):
        """
        This class process the end2end results and structure recognition results.
        :param end2end_file: end2end results predict by end2end inference.
        :param structure_master_file: structure recognition results predict by structure master inference.
        """
        # 初始化 Matcher 类，接收 end2end 文件和 structure_master 文件作为参数
        self.end2end_file = end2end_file
        self.structure_master_file = structure_master_file
        # 使用 pickle_load 函数加载 end2end 文件的结果，以 'end2end' 为前缀
        self.end2end_results = pickle_load(end2end_file, prefix='end2end')
        # 使用 pickle_load 函数加载 structure_master 文件的结果，以 'structure' 为前缀
        self.structure_master_results = pickle_load(
            structure_master_file, prefix='structure')
    def _format(self, match_result, file_name):
        """
        Extend the master token(insert virtual master token), and format matching result.
        :param match_result: 匹配结果
        :param file_name: 文件名
        :return: 返回格式化后的匹配结果
        """
        # 获取文件名对应的端到端信息
        end2end_info = self.end2end_results[file_name]
        # 获取文件名对应的结构化主信息
        master_info = self.structure_master_results[file_name]
        # 获取主信息中的文本
        master_token = master_info['text']
        # 获取匹配结果中的排序后的组
        sorted_groups = match_result['sorted_groups']

        # 创建虚拟主标记
        virtual_master_token_list = []
        for line_group in sorted_groups:
            tmp_list = ['<tr>']
            item_nums = len(line_group)
            for _ in range(item_nums):
                tmp_list.append('<td></td>')
            tmp_list.append('</tr>')
            virtual_master_token_list.extend(tmp_list)

        # 插入虚拟主标记
        master_token_list = master_token.split(',')
        if master_token_list[-1] == '</tbody>':
            # 完整预测（未按最大长度截断）
            # 在这种情况下插入虚拟主标记会降低验证集中的TEDs分数。
            # 因此，在这种情况下我们不会扩展虚拟标记。

            # 伪扩展虚拟
            master_token_list[:-1].extend(virtual_master_token_list)

            # 真实扩展虚拟
            # master_token_list = master_token_list[:-1]
            # master_token_list.extend(virtual_master_token_list)
            # master_token_list.append('</tbody>')

        elif master_token_list[-1] == '<td></td>':
            master_token_list.append('</tr>')
            master_token_list.extend(virtual_master_token_list)
            master_token_list.append('</tbody>')
        else:
            master_token_list.extend(virtual_master_token_list)
            master_token_list.append('</tbody>')

        # 格式化输出
        match_result.setdefault('matched_master_token_list', master_token_list)
        return match_result
    # 定义一个方法，用于将OCR结果合并到结构化令牌中，以获取最终结果
    def get_merge_result(self, match_results):
        """
        Merge the OCR result into structure token to get final results.
        :param match_results: 匹配结果的字典
        :return: 合并后的结果字典
        """
        # 创建一个空字典，用于存储合并后的结果
        merged_results = dict()

        # 定义一个分隔符，用于表示一个主边界框有多个端到端边界框时的换行符
        break_token = ' '

        # 遍历匹配结果字典中的每个文件名及其匹配信息
        for idx, (file_name, match_info) in enumerate(match_results.items()):
            # 获取该文件名对应的端到端信息
            end2end_info = self.end2end_results[file_name]
            # 获取匹配的主令牌列表
            master_token_list = match_info['matched_master_token_list']
            # 获取匹配列表，包括额外匹配
            match_list = match_info['match_list_add_extra_match']

            # 将匹配列表转换为匹配字典
            match_dict = get_match_dict(match_list)
            # 根据匹配字典、端到端信息和分隔符获取匹配文本字典
            match_text_dict = get_match_text_dict(match_dict, end2end_info,
                                                  break_token)
            # 将匹配文本插入到主令牌列表中
            merged_result = insert_text_to_token(master_token_list,
                                                 match_text_dict)
            # 处理合并后的结果中的边界框
            merged_result = deal_bb(merged_result)

            # 将合并后的结果存入结果字典中
            merged_results[file_name] = merged_result

        # 返回合并后的结果字典
        return merged_results
class TableMasterMatcher(Matcher):
    # 定义一个TableMasterMatcher类，继承自Matcher类
    def __init__(self):
        # 初始化函数，不执行任何操作

    def __call__(self, structure_res, dt_boxes, rec_res, img_name=1):
        # 定义一个调用函数，接受structure_res, dt_boxes, rec_res和img_name参数
        end2end_results = {img_name: []}
        # 创建一个字典end2end_results，键为img_name，值为空列表
        for dt_box, res in zip(dt_boxes, rec_res):
            # 遍历dt_boxes和rec_res的元素
            d = dict(
                bbox=np.array(dt_box),
                text=res[0], )
            # 创建一个字典d，包含bbox和text字段
            end2end_results[img_name].append(d)
            # 将d添加到end2end_results字典中img_name对应的值列表中

        self.end2end_results = end2end_results
        # 将end2end_results赋值给类属性end2end_results

        structure_master_result_dict = {img_name: {}}
        # 创建一个字典structure_master_result_dict，键为img_name，值为空字典
        pred_structures, pred_bboxes = structure_res
        # 将structure_res解包为pred_structures和pred_bboxes
        pred_structures = ','.join(pred_structures[3:-3])
        # 将pred_structures的第3个到倒数第3个元素以逗号连接成字符串
        structure_master_result_dict[img_name]['text'] = pred_structures
        # 将pred_structures赋值给structure_master_result_dict字典img_name对应的text字段
        structure_master_result_dict[img_name]['bbox'] = pred_bboxes
        # 将pred_bboxes赋值给structure_master_result_dict字典img_name对应的bbox字段
        self.structure_master_results = structure_master_result_dict
        # 将structure_master_result_dict赋值给类属性structure_master_results

        # match
        match_results = self.match()
        # 调用类的match方法，得到匹配结果
        merged_results = self.get_merge_result(match_results)
        # 调用类的get_merge_result方法，得到合并结果
        pred_html = merged_results[img_name]
        # 获取合并结果中img_name对应的值
        pred_html = '<html><body><table>' + pred_html + '</table></body></html>'
        # 将pred_html包装成HTML格式
        return pred_html
        # 返回HTML格式的预测结果
```