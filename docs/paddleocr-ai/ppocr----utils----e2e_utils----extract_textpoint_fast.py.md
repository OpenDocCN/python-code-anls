# `.\PaddleOCR\ppocr\utils\e2e_utils\extract_textpoint_fast.py`

```
# 版权声明和许可信息
# 本代码受 Apache 许可证版本 2.0 保护
# 请在遵守许可证的情况下使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入必要的库
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
            # 将字节流解码为 UTF-8 格式的字符串，并去除换行符
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            character_str += line
        # 将字符串转换为字符列表
        dict_character = list(character_str)
    return dict_character

# 实现 softmax 函数
def softmax(logits):
    """
    logits: N x d
    """
    # 计算每行的最大值
    max_value = np.max(logits, axis=1, keepdims=True)
    # 计算指数
    exp = np.exp(logits - max_value)
    # 计算指数和
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    # 计算 softmax 分布
    dist = exp / exp_sum
    return dist

# 获取保留项目的位置索引
def get_keep_pos_idxs(labels, remove_blank=None):
    """
    Remove duplicate and get pos idxs of keep items.
    The value of keep_blank should be [None, 95].
    """
    duplicate_len_list = []
    keep_pos_idx_list = []
    keep_char_idx_list = []
    # 根据标签进行分组，返回键和对应的值迭代器
    for k, v_ in groupby(labels):
        # 计算当前分组的长度
        current_len = len(list(v_))
        # 如果当前键不等于指定要移除的空白字符
        if k != remove_blank:
            # 计算当前索引位置，将其插入到重复长度列表的总和位置
            current_idx = int(sum(duplicate_len_list) + current_len // 2)
            # 将当前索引位置添加到保留位置索引列表中
            keep_pos_idx_list.append(current_idx)
            # 将当前键添加到保留字符索引列表中
            keep_char_idx_list.append(k)
        # 将当前分组长度添加到重复长度列表中
        duplicate_len_list.append(current_len)
    # 返回保留的字符索引列表和位置索引列表
    return keep_char_idx_list, keep_pos_idx_list
# 从标签列表中移除指定的空白标签
def remove_blank(labels, blank=0):
    # 使用列表推导式创建一个新的标签列表，排除掉指定的空白标签
    new_labels = [x for x in labels if x != blank]
    # 返回新的标签列表
    return new_labels


# 在标签列表中插入指定的空白标签
def insert_blank(labels, blank=0):
    # 创建一个新的标签列表，初始值为指定的空白标签
    new_labels = [blank]
    # 遍历原标签列表，每个标签之间插入指定的空白标签
    for l in labels:
        new_labels += [l, blank]
    # 返回新的标签列表
    return new_labels


# CTC 贪婪解码器，找出最佳路径
def ctc_greedy_decoder(probs_seq, blank=95, keep_blank_in_idxs=True):
    """
    CTC greedy (best path) decoder.
    """
    # 获取每个时间步最大概率的标签
    raw_str = np.argmax(np.array(probs_seq), axis=1)
    # 如果需要保留空白标签的索引，则设置移除空白标签的位置为 None
    remove_blank_in_pos = None if keep_blank_in_idxs else blank
    # 获取去重后的标签列表和保留的索引列表
    dedup_str, keep_idx_list = get_keep_pos_idxs(
        raw_str, remove_blank=remove_blank_in_pos)
    # 移除空白标签，得到最终的标签列表
    dst_str = remove_blank(dedup_str, blank=blank)
    # 返回最终的标签列表和保留的索引列表
    return dst_str, keep_idx_list


# 实例化 CTC 贪婪解码器，处理 logits_map 中的数据
def instance_ctc_greedy_decoder(gather_info,
                                logits_map,
                                pts_num=4,
                                point_gather_mode=None):
    # 获取 logits_map 的形状信息
    _, _, C = logits_map.shape
    # 如果点聚合模式为'align'
    if point_gather_mode == 'align':
        # 初始化插入数量为0
        insert_num = 0
        # 将gather_info转换为numpy数组
        gather_info = np.array(gather_info)
        # 获取gather_info的长度
        length = len(gather_info) - 1
        # 遍历gather_info数组
        for index in range(length):
            # 计算y方向和x方向的步长
            stride_y = np.abs(gather_info[index + insert_num][0] - gather_info[index + 1 + insert_num][0])
            stride_x = np.abs(gather_info[index + insert_num][1] - gather_info[index + 1 + insert_num][1])
            # 取步长的最大值
            max_points = int(max(stride_x, stride_y))
            # 计算步长
            stride = (gather_info[index + insert_num] - gather_info[index + 1 + insert_num]) / (max_points)
            # 计算插入数量
            insert_num_temp = max_points - 1

            # 插入数据
            for i in range(int(insert_num_temp)):
                insert_value = gather_info[index + insert_num] - (i + 1) * stride
                insert_index = index + i + 1 + insert_num
                gather_info = np.insert(gather_info, insert_index, insert_value, axis=0)
            # 更新插入数量
            insert_num += insert_num_temp
        # 将gather_info转换为列表
        gather_info = gather_info.tolist()
    else:
        pass

    # 将gather_info中的y坐标和x坐标分离
    ys, xs = zip(*gather_info)
    # 从logits_map中获取对应坐标的logits序列
    logits_seq = logits_map[list(ys), list(xs)]
    # 将logits_seq作为probs_seq
    probs_seq = logits_seq
    # 获取labels，即在每个位置概率最大的类别
    labels = np.argmax(probs_seq, axis=1)
    # 去除类别为C-1的连续重复元素
    dst_str = [k for k, v_ in groupby(labels) if k != C - 1]
    # 计算gather_info的间隔
    detal = len(gather_info) // (pts_num - 1)
    # 选择保留的索引列表
    keep_idx_list = [0] + [detal * (i + 1) for i in range(pts_num - 2)] + [-1]
    # 根据索引列表获取保留的gather_info
    keep_gather_list = [gather_info[idx] for idx in keep_idx_list]
    # 返回结果
    return dst_str, keep_gather_list
# 使用多进程的 CTC 解码器
def ctc_decoder_for_image(gather_info_list,
                          logits_map,
                          Lexicon_Table,
                          pts_num=6,
                          point_gather_mode=None):
    """
    CTC decoder using multiple processes.
    """
    decoder_str = []  # 存储解码后的字符串
    decoder_xys = []  # 存储解码后的坐标
    for gather_info in gather_info_list:
        if len(gather_info) < pts_num:  # 如果 gather_info 的长度小于指定的点数，则跳过
            continue
        dst_str, xys_list = instance_ctc_greedy_decoder(
            gather_info,
            logits_map,
            pts_num=pts_num,
            point_gather_mode=point_gather_mode)
        dst_str_readable = ''.join([Lexicon_Table[idx] for idx in dst_str])  # 将解码后的索引转换为可读字符串
        if len(dst_str_readable) < 2:  # 如果解码后的字符串长度小于2，则跳过
            continue
        decoder_str.append(dst_str_readable)  # 将解码后的字符串添加到列表中
        decoder_xys.append(xys_list)  # 将解码后的坐标添加到列表中
    return decoder_str, decoder_xys  # 返回解码后的字符串列表和坐标列表


# 根据给定方向对位置列表进行排序
def sort_with_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """

    # 根据给定方向对部分位置列表进行排序
    def sort_part_with_direction(pos_list, point_direction):
        pos_list = np.array(pos_list).reshape(-1, 2)
        point_direction = np.array(point_direction).reshape(-1, 2)
        average_direction = np.mean(point_direction, axis=0, keepdims=True)
        pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
        sorted_list = pos_list[np.argsort(pos_proj_leng)].tolist()
        sorted_direction = point_direction[np.argsort(pos_proj_leng)].tolist()
        return sorted_list, sorted_direction

    pos_list = np.array(pos_list).reshape(-1, 2)
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]  # 获取给定位置的方向信息
    point_direction = point_direction[:, ::-1]  # 调整方向信息的顺序
    sorted_point, sorted_direction = sort_part_with_direction(pos_list,
                                                              point_direction)  # 调用排序函数对位置列表进行排序

    point_num = len(sorted_point)  # 获取排序后的位置列表的长度
    # 如果点的数量大于等于16个
    if point_num >= 16:
        # 计算中间点的位置
        middle_num = point_num // 2
        # 将点按照位置划分为两部分，取前一部分的点和方向
        first_part_point = sorted_point[:middle_num]
        first_point_direction = sorted_direction[:middle_num]
        # 对前一部分的点和方向进行排序
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(
            first_part_point, first_point_direction)

        # 取后一部分的点和方向
        last_part_point = sorted_point[middle_num:]
        last_point_direction = sorted_direction[middle_num:]
        # 对后一部分的点和方向进行排序
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(
            last_part_point, last_point_direction)
        # 合并排序后的前一部分和后一部分的点
        sorted_point = sorted_fist_part_point + sorted_last_part_point
        # 合并排序后的前一部分和后一部分的方向
        sorted_direction = sorted_fist_part_direction + sorted_last_part_direction

    # 返回排序后的点和方向
    return sorted_point, np.array(sorted_direction)
# 为聚合特征添加 ID，用于推断
def add_id(pos_list, image_id=0):
    """
    Add id for gather feature, for inference.
    """
    # 创建一个新列表
    new_list = []
    # 遍历原始位置列表，为每个位置添加图像 ID，并加入新列表
    for item in pos_list:
        new_list.append((image_id, item[0], item[1]))
    # 返回添加了 ID 的新列表
    return new_list


# 根据方向对位置列表进行排序和扩展
def sort_and_expand_with_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    # 获取方向数组的高度、宽度和通道数
    h, w, _ = f_direction.shape
    # 调用排序函数对位置列表进行排序，并获取排序后的列表和方向
    sorted_list, point_direction = sort_with_direction(pos_list, f_direction)

    # 计算一些参数
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
    # 遍历追加的位置，计算左侧和右侧位置列表
    for i in range(append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ly < h and lx < w and (ly, lx) not in left_list:
            left_list.append((ly, lx))
        ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ry < h and rx < w and (ry, rx) not in right_list:
            right_list.append((ry, rx))

    # 组合左侧、排序后和右侧位置列表
    all_list = left_list[::-1] + sorted_list + right_list
    return all_list


# 根据方向对位置列表进行排序和扩展，版本 2
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

    # 获取排序后的位置数量
    point_num = len(sorted_list)
    # 计算左右两侧方向的子数组长度
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
    # 遍历左侧追加点
    for i in range(max_append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype(
            'int32').tolist()
        # 判断点是否在地图范围内且未被访问过
        if ly < h and lx < w and (ly, lx) not in left_list:
            # 判断二值化地图中对应位置的值是否大于0.5
            if binary_tcl_map[ly, lx] > 0.5:
                left_list.append((ly, lx))
            else:
                break

    # 遍历右侧追加点
    for i in range(max_append_num):
        ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype(
            'int32').tolist()
        # 判断点是否在地图范围内且未被访问过
        if ry < h and rx < w and (ry, rx) not in right_list:
            # 判断二值化地图中对应位置的值是否大于0.5
            if binary_tcl_map[ry, rx] > 0.5:
                right_list.append((ry, rx))
            else:
                break

    # 组合左侧、排序后的和右侧点列表，返回结果
    all_list = left_list[::-1] + sorted_list + right_list
    return all_list
# 将垂直点对转换为顺时针方向的多边形点
def point_pair2poly(point_pair_list):
    # 计算点对数量
    point_num = len(point_pair_list) * 2
    # 初始化点列表
    point_list = [0] * point_num
    # 遍历点对列表，将点对中的点按顺序填入点列表
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    # 将点列表转换为 numpy 数组，并按照二维形状返回
    return np.array(point_list).reshape(-1, 2)


# 沿着宽度收缩四边形
def shrink_quad_along_width(quad, begin_width_ratio=0., end_width_ratio=1.):
    # 创建收缩比例数组
    ratio_pair = np.array([[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
    # 计算四边形的两个顶点
    p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
    p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
    # 返回收缩后的四边形
    return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0])


# 沿着宽度扩展多边形
def expand_poly_along_width(poly, shrink_ratio_of_width=0.3):
    # 计算多边形的点数
    point_num = poly.shape[0]
    # 计算左侧四边形
    left_quad = np.array([poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
    # 计算左侧四边形的收缩比例
    left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                 (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    # 计算左侧四边形的扩展后的四边形
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    # 计算右侧四边形
    right_quad = np.array([
        poly[point_num // 2 - 2], poly[point_num // 2 - 1],
        poly[point_num // 2], poly[point_num // 2 + 1]
    ], dtype=np.float32)
    # 计算右侧四边形的收缩比例
    right_ratio = 1.0 + shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                  (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    # 计算右侧四边形的扩展后的四边形
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    # 更新多边形的顶点坐标
    poly[0] = left_quad_expand[0]
    poly[-1] = left_quad_expand[-1]
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2] = right_quad_expand[2]
    return poly


# 恢复多边形
def restore_poly(instance_yxs_list, seq_strs, p_border, ratio_w, ratio_h, src_w,
                 src_h, valid_set):
    # 初始化多边形列表
    poly_list = []
    # 初始化保留字符串列表
    keep_str_list = []
    # 遍历实例的中心线和对应的序列字符串
    for yx_center_line, keep_str in zip(instance_yxs_list, seq_strs):
        # 如果序列字符串长度小于2，打印提示信息并继续下一次循环
        if len(keep_str) < 2:
            print('--> too short, {}'.format(keep_str))
            continue

        # 设置偏移扩展系数
        offset_expand = 1.0
        # 如果是'totaltext'数据集，则将偏移扩展系数设置为1.2
        if valid_set == 'totaltext':
            offset_expand = 1.2

        # 初始化点对列表
        point_pair_list = []
        # 遍历中心线的每个点
        for y, x in yx_center_line:
            # 根据像素边界获取偏移量，并根据偏移扩展系数进行调整
            offset = p_border[:, y, x].reshape(2, 2) * offset_expand
            # 将原始坐标转换为浮点型数组
            ori_yx = np.array([y, x], dtype=np.float32)
            # 计算点对坐标并进行比例缩放
            point_pair = (ori_yx + offset)[:, ::-1] * 4.0 / np.array([ratio_w, ratio_h]).reshape(-1, 2)
            point_pair_list.append(point_pair)

        # 将点对坐标转换为多边形
        detected_poly = point_pair2poly(point_pair_list)
        # 沿着宽度方向扩展多边形
        detected_poly = expand_poly_along_width(detected_poly, shrink_ratio_of_width=0.2)
        # 将多边形坐标限制在图像范围内
        detected_poly[:, 0] = np.clip(detected_poly[:, 0], a_min=0, a_max=src_w)
        detected_poly[:, 1] = np.clip(detected_poly[:, 1], a_min=0, a_max=src_h)

        # 将序列字符串添加到列表中
        keep_str_list.append(keep_str)
        # 根据数据集类型进行处理
        if valid_set == 'partvgg':
            # 如果是'partvgg'数据集，取中间点并截取多边形
            middle_point = len(detected_poly) // 2
            detected_poly = detected_poly[[0, middle_point - 1, middle_point, -1], :]
            poly_list.append(detected_poly)
        elif valid_set == 'totaltext':
            # 如果是'totaltext'数据集，直接添加多边形
            poly_list.append(detected_poly)
        else:
            # 如果不支持的数据集类型，打印提示信息并退出程序
            print('--> Not supported format.')
            exit(-1)
    # 返回多边形列表和序列字符串列表
    return poly_list, keep_str_list
# 快速生成中心点和结束点的 TCL 实例；使用字符映射进行过滤
def generate_pivot_list_fast(p_score,
                             p_char_maps,
                             f_direction,
                             Lexicon_Table,
                             score_thresh=0.5,
                             point_gather_mode=None):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    # 获取第一个元素
    p_score = p_score[0]
    # 调整维度顺序
    f_direction = f_direction.transpose(1, 2, 0)
    # 创建 TCL 实例地图
    p_tcl_map = (p_score > score_thresh) * 1.0
    # 对 TCL 实例地图进行细化
    skeleton_map = thin(p_tcl_map.astype(np.uint8))
    # 计算 TCL 实例数量和实例标签地图
    instance_count, instance_label_map = cv2.connectedComponents(
        skeleton_map.astype(np.uint8), connectivity=8)

    # 获取 TCL 实例
    all_pos_yxs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            if len(pos_list) < 3:
                continue

            # 对位置列表根据方向进行排序和扩展
            pos_list_sorted = sort_and_expand_with_direction_v2(
                pos_list, f_direction, p_tcl_map)
            all_pos_yxs.append(pos_list_sorted)

    # 调整字符映射的维度顺序
    p_char_maps = p_char_maps.transpose([1, 2, 0])
    # 使用 CTC 解码器对图像进行解码
    decoded_str, keep_yxs_list = ctc_decoder_for_image(
        all_pos_yxs,
        logits_map=p_char_maps,
        Lexicon_Table=Lexicon_Table,
        point_gather_mode=point_gather_mode)
    return keep_yxs_list, decoded_str


# 提取主方向
def extract_main_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    # 将位置列表转换为 NumPy 数组
    pos_list = np.array(pos_list)
    # 获取点的方向
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]
    # 调整方向顺序
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    # 计算平均方向
    average_direction = np.mean(point_direction, axis=0, keepdims=True)
    # 归一化平均方向
    average_direction = average_direction / (
        np.linalg.norm(average_direction) + 1e-6)
    return average_direction


# 根据方向和图像 ID 进行排序（已弃用）
def sort_by_direction_with_image_id_deprecated(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[id, y, x], [id, y, x], [id, y, x] ...]
    """
    # 将 pos_list 转换为 numpy 数组，并重新组织为二维数组
    pos_list_full = np.array(pos_list).reshape(-1, 3)
    # 从 pos_list_full 中提取出位置信息，去除 id 列
    pos_list = pos_list_full[:, 1:]
    # 根据 pos_list 中的位置信息在 f_direction 中查找对应的方向信息
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]  # x, y
    # 将 point_direction 中的坐标顺序进行反转，从 x, y 调整为 y, x
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    # 计算所有点的平均方向
    average_direction = np.mean(point_direction, axis=0, keepdims=True)
    # 计算每个点到平均方向的投影长度
    pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
    # 根据投影长度对 pos_list_full 进行排序
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
    # 根据位置列表的坐标获取对应的方向
    point_direction = f_direction[pos_list[:, 1], pos_list[:, 2]]  # x, y
    # 调整方向的顺序，从 x, y 调整为 y, x
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    # 调用内部函数对位置列表进行排序
    sorted_point, sorted_direction = sort_part_with_direction(pos_list, point_direction)

    # 获取排序后的位置数量
    point_num = len(sorted_point)
    # 如果位置数量大于等于 16
    if point_num >= 16:
        # 计算中间位置的索引
        middle_num = point_num // 2
        # 分割位置列表为两部分，分别进行排序
        first_part_point = sorted_point[:middle_num]
        first_point_direction = sorted_direction[:middle_num]
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(
            first_part_point, first_point_direction)

        last_part_point = sorted_point[middle_num:]
        last_point_direction = sorted_direction[middle_num:]
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(
            last_part_point, last_point_direction)
        # 合并排序后的两部分位置列表和方向列表
        sorted_point = sorted_fist_part_point + sorted_last_part_point
        sorted_direction = sorted_fist_part_direction + sorted_last_part_direction

    # 返回排序后的位置列表
    return sorted_point
```