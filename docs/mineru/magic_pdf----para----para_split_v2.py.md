# `.\MinerU\magic_pdf\para\para_split_v2.py`

```
# 导入标准库和第三方库
import copy

# 从 sklearn 导入 DBSCAN 聚类算法
from sklearn.cluster import DBSCAN
# 导入 NumPy 库
import numpy as np
# 导入 loguru 日志库
from loguru import logger
# 导入正则表达式库
import re
# 从自定义库中导入函数以检查区域重叠
from magic_pdf.libs.boxbase import _is_in_or_part_overlap_with_area_ratio as is_in_layout
# 从自定义库中导入内容类型和块类型
from magic_pdf.libs.ocr_content_type import ContentType, BlockType
# 从自定义模型中导入 MagicModel 类
from magic_pdf.model.magic_model import MagicModel
# 导入常量
from magic_pdf.libs.Constants import *

# 定义行结束符的列表
LINE_STOP_FLAG = ['.', '!', '?', '。', '！', '？', "：", ":", ")", "）", ";"]
# 定义内联方程的内容类型
INLINE_EQUATION = ContentType.InlineEquation
# 定义行间方程的内容类型
INTERLINE_EQUATION = ContentType.InterlineEquation
# 定义文本的内容类型
TEXT = ContentType.Text
# 初始化调试标志为 False
debug_able = False


# 定义私有函数以获取 span 的文本内容
def __get_span_text(span):
    # 尝试从 span 中获取内容，如果没有则尝试获取图像路径
    c = span.get('content', '')
    if len(c) == 0:
        c = span.get('image_path', '')

    # 返回找到的内容或路径
    return c


# 定义私有函数以检测列表行并将其分开
def __detect_list_lines(lines, new_layout_bboxes, lang):
    # 声明调试标志为全局变量
    global debug_able
    """
    探测是否包含了列表，并且把列表的行分开.
    这样的段落特点是，顶格字母大写/数字，紧跟着几行缩进的。缩进的行首字母含小写的。
    """

    # 定义内部函数以查找重复模式
    def find_repeating_patterns2(lst):
        # 初始化索引列表
        indices = []
        ones_indices = []
        i = 0
        while i < len(lst):  # 循环遍历整个列表
            if lst[i] == 1:  # 如果遇到 '1'，可能是模式的开始
                start = i
                ones_in_this_interval = [i]
                i += 1
                # 遍历值为 1、2 或 3 的元素，直到遇到其他值
                while i < len(lst) and lst[i] in [1, 2, 3]:
                    if lst[i] == 1:
                        ones_in_this_interval.append(i)
                    i += 1
                # 如果条件符合，记录起始和结束索引
                if len(ones_in_this_interval) > 1 or (
                        start < len(lst) - 1 and ones_in_this_interval and lst[start + 1] in [2, 3]):
                    indices.append((start, i - 1))
                    ones_indices.append(ones_in_this_interval)
            else:
                i += 1
        # 返回找到的索引和 1 的索引列表
        return indices, ones_indices

    # 定义内部函数以查找重复模式
    def find_repeating_patterns(lst):
        # 初始化索引列表
        indices = []
        ones_indices = []
        i = 0
        while i < len(lst) - 1:  # 确保余下元素至少有2个
            if lst[i] == 1 and lst[i + 1] in [2, 3]:  # 额外检查以防止连续出现的1
                start = i
                ones_in_this_interval = [i]
                i += 1
                while i < len(lst) and lst[i] in [2, 3]:
                    i += 1
                # 验证下一个序列是否符合条件
                if i < len(lst) - 1 and lst[i] == 1 and lst[i + 1] in [2, 3] and lst[i - 1] in [2, 3]:
                    while i < len(lst) and lst[i] in [1, 2, 3]:
                        if lst[i] == 1:
                            ones_in_this_interval.append(i)
                        i += 1
                    # 记录起始和结束索引
                    indices.append((start, i - 1))
                    ones_indices.append(ones_in_this_interval)
                else:
                    i += 1
            else:
                i += 1
        # 返回找到的索引和 1 的索引列表
        return indices, ones_indices

    """===================="""
    # 定义一个函数，用于根据给定的长度和索引数组分割区间
    def split_indices(slen, index_array):
        # 初始化结果列表，用于存储区间信息
        result = []
        # 记录上一个区间的结束位置
        last_end = 0
    
        # 对索引数组进行排序，并遍历每个区间
        for start, end in sorted(index_array):
            if start > last_end:
                # 前一个区间结束到下一个区间开始之间的部分标记为"text"
                result.append(('text', last_end, start - 1))
            # 当前区间标记为"list"
            result.append(('list', start, end))
            # 更新上一个区间的结束位置
            last_end = end + 1
    
        if last_end < slen:
            # 如果最后一个区间结束后还有剩余的字符串，将其标记为"text"
            result.append(('text', last_end, slen - 1))
    
        # 返回包含所有区间信息的结果
        return result
    
    """===================="""
    
    # 如果语言不是英语，直接返回行和空值
    if lang != 'en':
        return lines, None
    
    # 获取行的总数
    total_lines = len(lines)
    # 初始化一个列表，用于存储每行的特征编码
    line_fea_encode = []
    """
    对每一行进行特征编码，编码规则如下：
    1. 如果行顶格，且大写字母开头或者数字开头，编码为1
    2. 如果顶格，其他非大写开头编码为4
    3. 如果非顶格，首字符大写，编码为2
    4. 如果非顶格，首字符非大写编码为3
    """
    # 如果行数大于0，调用函数生成x_map_tag_dict和min_x_tag
    if len(lines) > 0:
        x_map_tag_dict, min_x_tag = cluster_line_x(lines)
    # 遍历每一行
    for l in lines:
        # 获取当前行的跨度文本
        span_text = __get_span_text(l['spans'][0])
        if not span_text:
            # 如果跨度文本为空，编码为0并继续下一个循环
            line_fea_encode.append(0)
            continue
        # 获取跨度文本的首字符
        first_char = span_text[0]
        # 根据当前行的边界框在新布局中查找布局
        layout = __find_layout_bbox_by_line(l['bbox'], new_layout_bboxes)
        if not layout:
            # 如果找不到布局，编码为0
            line_fea_encode.append(0)
        else:
            #
            if x_map_tag_dict[round(l['bbox'][0])] == min_x_tag:
                # 如果首字符不是字母数字或与参考列表匹配，编码为1
                if not first_char.isalnum() or if_match_reference_list(span_text):
                    line_fea_encode.append(1)
                else:
                    # 否则编码为4
                    line_fea_encode.append(4)
            else:
                # 如果首字符是大写字母，编码为2
                if first_char.isupper():
                    line_fea_encode.append(2)
                else:
                    # 否则编码为3
                    line_fea_encode.append(3)
    
        # 然后根据编码进行分段，选出连续出现至少2次的1,2,3编码的行，认为是列表。
        
        list_indice, list_start_idx = find_repeating_patterns2(line_fea_encode)
        if len(list_indice) > 0:
            if debug_able:
                # 如果调试开启，记录发现的列表及行数
                logger.info(f"发现了列表，列表行数：{list_indice}， {list_start_idx}")
    
        # TODO check一下这个特列表里缩进的行左侧是不是对齐的。
        # 初始化一个列表，用于存储分段信息
        segments = []
        # 遍历找到的列表索引
        for start, end in list_indice:
            for i in range(start, end + 1):
                if i > 0:
                    # 如果当前行编码为4，则表示不是顶格行
                    if line_fea_encode[i] == 4:
                        if debug_able:
                            logger.info(f"列表行的第{i}行不是顶格的")
                        break
            else:
                # 如果没有跳出循环，说明所有行都是列表
                if debug_able:
                    logger.info(f"列表行的第{start}到第{end}行是列表")
    
        # 返回分割后的结果和列表起始索引
        return split_indices(total_lines, list_indice), list_start_idx
# 定义一个函数，用于对给定行的边界框的 x0 值进行聚类
def cluster_line_x(lines: list) -> dict:
    """
    对一个block内所有lines的bbox的x0聚类
    """
    # 设置聚类的最小距离和最小样本数
    min_distance = 5
    min_sample = 1
    # 从每个行的边界框中提取 x0 值，并封装成 NumPy 数组
    x0_lst = np.array([[round(line['bbox'][0]), 0] for line in lines])
    # 使用 DBSCAN 算法对 x0 值进行聚类
    x0_clusters = DBSCAN(eps=min_distance, min_samples=min_sample).fit(x0_lst)
    # 获取唯一的聚类标签
    x0_uniq_label = np.unique(x0_clusters.labels_)
    # 创建一个字典用于存储旧值对应的新值的映射
    x0_2_new_val = {}  # 存储旧值对应的新值映射
    # 记录初始的最小 x0 值
    min_x0 = round(lines[0]["bbox"][0])
    # 遍历每一个唯一的聚类标签
    for label in x0_uniq_label:
        # 跳过噪声点（标签为 -1）
        if label == -1:
            continue
        # 获取当前聚类标签对应的索引
        x0_index_of_label = np.where(x0_clusters.labels_ == label)
        # 提取当前聚类中的原始 x0 值
        x0_raw_val = x0_lst[x0_index_of_label][:, 0]
        # 找到当前聚类中的最小 x0 值
        x0_new_val = np.min(x0_lst[x0_index_of_label][:, 0])
        # 更新字典，将原始值映射到新值
        x0_2_new_val.update({round(raw_val): round(x0_new_val) for raw_val in x0_raw_val})
        # 更新最小 x0 值
        if x0_new_val < min_x0:
            min_x0 = x0_new_val
    # 返回旧值到新值的映射和最小 x0 值
    return x0_2_new_val, min_x0


# 定义一个函数，用于检查给定文本是否与参考列表匹配
def if_match_reference_list(text: str) -> bool:
    # 编译一个正则表达式，用于匹配以数字开头并包含句点的文本
    pattern = re.compile(r'^\d+\..*')
    # 检查文本是否与模式匹配
    if pattern.match(text):
        return True
    else:
        return False


# 定义一个私有函数，用于在布局框内对齐行的左右侧
def __valign_lines(blocks, layout_bboxes):
    """
    在一个layoutbox内对齐行的左侧和右侧。
    扫描行的左侧和右侧，如果x0, x1差距不超过一个阈值，就强行对齐到所处layout的左右两侧（和layout有一段距离）。
    3是个经验值，TODO，计算得来，可以设置为1.5个正文字符。
    """
    
    # 设置对齐的最小距离和最小样本数
    min_distance = 3
    min_sample = 2
    # 初始化一个新的布局边界框列表
    new_layout_bboxes = []
    # 为每个块添加深拷贝的边界框以进行段落分割计算
    for block in blocks:
        block["bbox_fs"] = copy.deepcopy(block["bbox"])
    # 遍历每个布局框
    for layout_box in layout_bboxes:
        # 筛选出在当前布局框内的文本块
        blocks_in_layoutbox = [b for b in blocks if
                               b["type"] == BlockType.Text and is_in_layout(b['bbox'], layout_box['layout_bbox'])]
        # 如果没有找到文本块或文本块没有行，添加当前布局框到新布局框列表并继续
        if len(blocks_in_layoutbox) == 0 or len(blocks_in_layoutbox[0]["lines"]) == 0:
            new_layout_bboxes.append(layout_box['layout_bbox'])
            continue

        # 获取每个文本行的左边界坐标并构成数组
        x0_lst = np.array([[line['bbox'][0], 0] for block in blocks_in_layoutbox for line in block['lines']])
        # 获取每个文本行的右边界坐标并构成数组
        x1_lst = np.array([[line['bbox'][2], 0] for block in blocks_in_layoutbox for line in block['lines']])
        # 对左边界坐标进行 DBSCAN 聚类
        x0_clusters = DBSCAN(eps=min_distance, min_samples=min_sample).fit(x0_lst)
        # 对右边界坐标进行 DBSCAN 聚类
        x1_clusters = DBSCAN(eps=min_distance, min_samples=min_sample).fit(x1_lst)
        # 获取左边界聚类的唯一标签
        x0_uniq_label = np.unique(x0_clusters.labels_)
        # 获取右边界聚类的唯一标签
        x1_uniq_label = np.unique(x1_clusters.labels_)

        # 创建映射字典以存储旧值到新值的映射
        x0_2_new_val = {}  # 存储旧值对应的新值映射
        x1_2_new_val = {}
        # 为每个左边界的聚类标签创建新值
        for label in x0_uniq_label:
            if label == -1:
                continue
            x0_index_of_label = np.where(x0_clusters.labels_ == label)
            x0_raw_val = x0_lst[x0_index_of_label][:, 0]
            x0_new_val = np.min(x0_lst[x0_index_of_label][:, 0])
            # 更新映射字典
            x0_2_new_val.update({idx: x0_new_val for idx in x0_raw_val})
        # 为每个右边界的聚类标签创建新值
        for label in x1_uniq_label:
            if label == -1:
                continue
            x1_index_of_label = np.where(x1_clusters.labels_ == label)
            x1_raw_val = x1_lst[x1_index_of_label][:, 0]
            x1_new_val = np.max(x1_lst[x1_index_of_label][:, 0])
            # 更新映射字典
            x1_2_new_val.update({idx: x1_new_val for idx in x1_raw_val})

        # 遍历每个文本块，更新行的边界框
        for block in blocks_in_layoutbox:
            for line in block['lines']:
                x0, x1 = line['bbox'][0], line['bbox'][2]
                # 如果左边界在映射中，更新其值
                if x0 in x0_2_new_val:
                    line['bbox'][0] = int(x0_2_new_val[x0])

                # 如果右边界在映射中，更新其值
                if x1 in x1_2_new_val:
                    line['bbox'][2] = int(x1_2_new_val[x1])
            # 其余对不齐的保持不动

        # 由于修改了block里的line长度，现在需要重新计算block的bbox
        for block in blocks_in_layoutbox:
            if len(block["lines"]) > 0:
                # 重新计算每个块的边界框
                block['bbox_fs'] = [min([line['bbox'][0] for line in block['lines']]),
                                    min([line['bbox'][1] for line in block['lines']]),
                                    max([line['bbox'][2] for line in block['lines']]),
                                    max([line['bbox'][3] for line in block['lines']])]
        # 新计算layout的bbox，因为block的bbox变了。
        layout_x0 = min([block['bbox_fs'][0] for block in blocks_in_layoutbox])
        layout_y0 = min([block['bbox_fs'][1] for block in blocks_in_layoutbox])
        layout_x1 = max([block['bbox_fs'][2] for block in blocks_in_layoutbox])
        layout_y1 = max([block['bbox_fs'][3] for block in blocks_in_layoutbox])
        # 将新计算的布局框添加到列表中
        new_layout_bboxes.append([layout_x0, layout_y0, layout_x1, layout_y1])

    # 返回新的布局框列表
    return new_layout_bboxes
# 对文本块进行对齐处理，确保超出部分被布局边界截断
def __align_text_in_layout(blocks, layout_bboxes):
    """
    由于ocr出来的line，有时候会在前后有一段空白，这个时候需要对文本进行对齐，超出的部分被layout左右侧截断。
    """
    # 遍历所有布局边界框
    for layout in layout_bboxes:
        # 获取当前布局的边界框
        lb = layout['layout_bbox']
        # 过滤出在当前布局内的文本块
        blocks_in_layoutbox = [block for block in blocks if
                               block["type"] == BlockType.Text and is_in_layout(block['bbox'], lb)]
        # 如果没有文本块在当前布局内，则跳过
        if len(blocks_in_layoutbox) == 0:
            continue

        # 对每个在布局内的文本块进行处理
        for block in blocks_in_layoutbox:
            # 遍历文本块中的每一行
            for line in block.get("lines", []):
                x0, x1 = line['bbox'][0], line['bbox'][2]
                # 如果行的左边界小于布局的左边界，则调整左边界
                if x0 < lb[0]:
                    line['bbox'][0] = lb[0]
                # 如果行的右边界大于布局的右边界，则调整右边界
                if x1 > lb[2]:
                    line['bbox'][2] = lb[2]


# 对文本块进行通用预处理，不区分语言
def __common_pre_proc(blocks, layout_bboxes):
    """
    不分语言的，对文本进行预处理
    """
    # 调用行段落添加函数（当前被注释）
    # __add_line_period(blocks, layout_bboxes)
    # 对文本块进行布局对齐处理
    __align_text_in_layout(blocks, layout_bboxes)
    # 对齐后的布局框进行垂直对齐处理
    aligned_layout_bboxes = __valign_lines(blocks, layout_bboxes)

    # 返回对齐后的布局框
    return aligned_layout_bboxes


# 对中文文本块进行预处理
def __pre_proc_zh_blocks(blocks, layout_bboxes):
    """
    对中文文本进行分段预处理
    """
    # 当前函数未实现
    pass


# 对英文文本块进行预处理
def __pre_proc_en_blocks(blocks, layout_bboxes):
    """
    对英文文本进行分段预处理
    """
    # 当前函数未实现
    pass


# 根据布局聚合文本行
def __group_line_by_layout(blocks, layout_bboxes):
    """
    每个layout内的行进行聚合
    """
    # 初始化用于存储聚合结果的列表
    blocks_group = []
    # 遍历所有布局边界框
    for lyout in layout_bboxes:
        # 过滤出在当前布局内的文本块
        blocks_in_layout = [block for block in blocks if is_in_layout(block.get('bbox_fs', None), lyout['layout_bbox'])]
        # 将当前布局的文本块添加到聚合列表
        blocks_group.append(blocks_in_layout)
    # 返回聚合后的文本块
    return blocks_group


# 在布局框内对段落进行分段处理
def __split_para_in_layoutbox(blocks_group, new_layout_bbox, lang="en"):
    """
    lines_group 进行行分段——layout内部进行分段。lines_group内每个元素是一个Layoutbox内的所有行。
    1. 先计算每个group的左右边界。
    2. 然后根据行末尾特征进行分段。
        末尾特征：以句号等结束符结尾。并且距离右侧边界有一定距离。
        且下一行开头不留空白。

    """
    # 初始化用于存储每个布局信息的列表
    list_info = []  # 这个layout最后是不是列表,记录每一个layout里是不是列表开头，列表结尾
    # 遍历块组中的每个块
    for blocks in blocks_group:
        # 初始化开始和结束标志列表
        is_start_list = None
        is_end_list = None
        # 如果块为空，添加 [False, False] 到信息列表并继续
        if len(blocks) == 0:
            list_info.append([False, False])
            continue
        # 如果第一个和最后一个块的类型都不是文本，添加 [False, False] 到信息列表并继续
        if blocks[0]["type"] != BlockType.Text and blocks[-1]["type"] != BlockType.Text:
            list_info.append([False, False])
            continue
        # 如果第一个块不是文本，设置开始标志为 False
        if blocks[0]["type"] != BlockType.Text:
            is_start_list = False
        # 如果最后一个块不是文本，设置结束标志为 False
        if blocks[-1]["type"] != BlockType.Text:
            is_end_list = False

        # 从块中提取所有文本行
        lines = [line for block in blocks if
                 block["type"] == BlockType.Text for line in
                 block['lines']]
        # 计算总行数
        total_lines = len(lines)
        # 如果总行数为 0 或 1，添加 [False, False] 到信息列表并继续
        if total_lines == 1 or total_lines == 0:
            list_info.append([False, False])
            continue
        """在进入到真正的分段之前，要对文字块从统计维度进行对齐方式的探测，
                    对齐方式分为以下：
                    1. 左对齐的文本块(特点是左侧顶格，或者左侧不顶格但是右侧顶格的行数大于非顶格的行数，顶格的首字母有大写也有小写)
                        1) 右侧对齐的行，单独成一段
                        2) 中间对齐的行，按照字体/行高聚合成一段
                    2. 左对齐的列表块（其特点是左侧顶格的行数小于等于非顶格的行数，非定格首字母会有小写，顶格90%是大写。并且左侧顶格行数大于1，大于1是为了这种模式连续出现才能称之为列表）
                        这样的文本块，顶格的为一个段落开头，紧随其后非顶格的行属于这个段落。
        """
        # 检测文本段落和列表起始行
        text_segments, list_start_line = __detect_list_lines(lines, new_layout_bbox, lang)
        """根据list_range，把lines分成几个部分
        """
        # 遍历列表起始行
        for list_start in list_start_line:
            # 如果起始行不止一个，处理这些行
            if len(list_start) > 1:
                for i in range(0, len(list_start)):
                    index = list_start[i] - 1
                    # 如果索引有效，检查内容并添加换行符
                    if index >= 0:
                        if "content" in lines[index]["spans"][-1] and lines[index]["spans"][-1].get('type', '') not in [
                            ContentType.InlineEquation, ContentType.InterlineEquation]:
                            lines[index]["spans"][-1]["content"] += '\n\n'
        # 初始化布局信息，记录每个布局是否是列表开头或结尾
        layout_list_info = [False, False]  
        # 遍历文本段信息
        for content_type, start, end in text_segments:
            # 如果内容类型是列表，更新布局信息
            if content_type == 'list':
                if start == 0 and is_start_list is None:
                    layout_list_info[0] = True
                if end == total_lines - 1 and is_end_list is None:
                    layout_list_info[1] = True

        # 将布局信息添加到列表信息中
        list_info.append(layout_list_info)
    # 返回列表信息
    return list_info
# 定义一个函数，用于将文本行和文本块分割成段落
def __split_para_lines(lines: list, text_blocks: list) -> list:
    # 初始化文本段落、其他段落和文本行的列表
    text_paras = []
    other_paras = []
    text_lines = []
    # 遍历每一行
    for line in lines:
        # 获取当前行的所有内容类型
        spans_types = [span["type"] for span in line]
        # 如果当前行是表格类型，将其添加到其他段落中并继续
        if ContentType.Table in spans_types:
            other_paras.append([line])
            continue
        # 如果当前行是图片类型，将其添加到其他段落中并继续
        if ContentType.Image in spans_types:
            other_paras.append([line])
            continue
        # 如果当前行是行间方程类型，将其添加到其他段落中并继续
        if ContentType.InterlineEquation in spans_types:
            other_paras.append([line])
            continue
        # 将文本行添加到文本行列表中
        text_lines.append(line)

    # 遍历每个文本块
    for block in text_blocks:
        # 获取当前文本块的边界框
        block_bbox = block["bbox"]
        para = []
        # 遍历文本行
        for line in text_lines:
            bbox = line["bbox"]
            # 检查文本行是否在文本块的布局内
            if is_in_layout(bbox, block_bbox):
                para.append(line)
        # 如果找到段落，添加到文本段落列表
        if len(para) > 0:
            text_paras.append(para)
    # 将其他段落与文本段落合并
    paras = other_paras.extend(text_paras)
    # 对合并后的段落进行排序，按每个段落的顶部边界进行排序
    paras_sorted = sorted(paras, key=lambda x: x[0]["bbox"][1])
    # 返回排序后的段落
    return paras_sorted


# 定义一个函数，用于连接跨布局的列表
def __connect_list_inter_layout(blocks_group, new_layout_bbox, layout_list_info, page_num, lang):
    global debug_able
    """
    如果上个layout的最后一个段落是列表，下一个layout的第一个段落也是列表，那么将它们连接起来。
    根据layout_list_info判断是不是列表，下个layout的第一个段落如果不是列表，那么查看它们是否有几行具有相同的缩进。
    """
    # 如果 blocks_group 为空，返回原值和 False 状态
    if len(blocks_group) == 0 or len(blocks_group) == 0:  # 0的时候最后的return 会出错
        return blocks_group, [False, False]

    # 遍历块组中的每个布局
    for i in range(1, len(blocks_group)):
        # 如果当前或前一个布局没有段落，继续
        if len(blocks_group[i]) == 0 or len(blocks_group[i - 1]) == 0:
            continue
        # 获取前一个和下一个布局的列表信息
        pre_layout_list_info = layout_list_info[i - 1]
        next_layout_list_info = layout_list_info[i]
        # 获取前一个布局的最后一个段落和下一个布局的段落
        pre_last_para = blocks_group[i - 1][-1].get("lines", [])
        next_paras = blocks_group[i]
        next_first_para = next_paras[0]

        # 如果前一个布局是列表结尾而下一个布局不是，并且下一个的第一个段落是文本
        if pre_layout_list_info[1] and not next_layout_list_info[0] and next_first_para[
            "type"] == BlockType.Text:  # 前一个是列表结尾，后一个是非列表开头，此时检测是否有相同的缩进
            # 如果调试开启，记录信息
            if debug_able:
                logger.info(f"连接page {page_num} 内的list")
            # 寻找具有相同缩进的连续行
            may_list_lines = []
            lines = next_first_para.get("lines", [])

            # 遍历下一个段落的行
            for line in lines:
                # 检查行的边界框是否在新布局的边界框内
                if line['bbox'][0] > __find_layout_bbox_by_line(line['bbox'], new_layout_bbox)[0]:
                    may_list_lines.append(line)
                else:
                    break
            # 如果这些行的缩进相等，将它们连接到前一个段落
            if len(may_list_lines) > 0 and len(set([x['bbox'][0] for x in may_list_lines])) == 1:
                pre_last_para.extend(may_list_lines)
                next_first_para["lines"] = next_first_para["lines"][len(may_list_lines):]

    # 返回更新后的块组和页面级别的列表信息
    return blocks_group, [layout_list_info[0][0], layout_list_info[-1][1]]  # 同时还返回了这个页面级别的开头、结尾是不是列表的信息


# 定义一个函数，用于连接跨页面的列表
def __connect_list_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox,
                              pre_page_list_info, next_page_list_info, page_num, lang):
    #
    # 如果上一个布局的最后一个段落是列表，而下一个布局的第一个段落也是列表，连接它们。TODO: 目前无法区分列表和段落，因此此方法尚未实现。
    # 根据layout_list_info判断下一个布局的第一个段落是否是列表，若不是则检查它们的缩进是否相同。
    """
    # 检查上一个和下一个页面段落是否为空，若为空则返回False以避免出错
    if len(pre_page_paras) == 0 or len(next_page_paras) == 0:  # 0的时候最后的return 会出错
        return False
    # 检查上一个页面的最后段落和下一个页面的第一个段落是否为空
    if len(pre_page_paras[-1]) == 0 or len(next_page_paras[0]) == 0:
        return False
    # 确保上一个页面最后段落和下一个页面第一个段落都是文本类型
    if pre_page_paras[-1][-1]["type"] != BlockType.Text or next_page_paras[0][0]["type"] != BlockType.Text:
        return False
    # 如果前一个布局是列表结尾而后一个布局不是列表开头，检查缩进是否相同
    if pre_page_list_info[1] and not next_page_list_info[0]:  # 前一个是列表结尾，后一个是非列表开头，此时检测是否有相同的缩进
        if debug_able:  # 若调试模式开启，记录日志
            logger.info(f"连接page {page_num} 内的list")
        # 定义一个列表用于存储可能的连续行
        may_list_lines = []
        next_page_first_para = next_page_paras[0][0]  # 获取下一个页面的第一个段落
        # 如果该段落为文本类型，提取其所有行
        if next_page_first_para["type"] == BlockType.Text:
            lines = next_page_first_para["lines"]  # 获取行内容
            for line in lines:  # 遍历行
                # 检查行的起始位置是否在布局边界内
                if line['bbox'][0] > __find_layout_bbox_by_line(line['bbox'], next_page_layout_bbox)[0]:
                    may_list_lines.append(line)  # 添加到可能的列表行中
                else:
                    break  # 一旦遇到不符合条件的行则停止
        # 如果这些行的缩进是相等的，则连接到上一个布局的最后一段
        if len(may_list_lines) > 0 and len(set([x['bbox'][0] for x in may_list_lines])) == 1:
            # pre_page_paras[-1].append(may_list_lines)  # 这行被注释掉了
            # 合并下一页内容到上一页最后一段，并标记为跨页内容
            for line in may_list_lines:  # 遍历可能的列表行
                for span in line["spans"]:  # 遍历行内的所有跨度
                    span[CROSS_PAGE] = True  # 标记为跨页
            pre_page_paras[-1][-1]["lines"].extend(may_list_lines)  # 将行添加到上一个段落
            next_page_first_para["lines"] = next_page_first_para["lines"][len(may_list_lines):]  # 更新下一个段落的行
            return True  # 返回连接成功的标志

    return False  # 如果未满足条件，则返回False
# 根据给定行的边界框找到其所在的布局边界框
def __find_layout_bbox_by_line(line_bbox, layout_bboxes):
    # 文档字符串，说明函数的功能
    """
    根据line找到所在的layout
    """
    # 遍历所有布局边界框
    for layout in layout_bboxes:
        # 检查行边界框是否在当前布局中
        if is_in_layout(line_bbox, layout):
            # 如果在，则返回该布局
            return layout
    # 如果没有找到，则返回 None
    return None


# 连接不同布局之间的段落
def __connect_para_inter_layoutbox(blocks_group, new_layout_bbox):
    # 文档字符串，说明函数的功能和连接条件
    """
    layout之间进行分段。
    主要是计算前一个layOut的最后一行和后一个layout的第一行是否可以连接。
    连接的条件需要同时满足：
    1. 上一个layout的最后一行沾满整个行。并且没有结尾符号。
    2. 下一行开头不留空白。

    """
    # 用于存储连接的布局块
    connected_layout_blocks = []
    # 如果块组为空，返回空列表
    if len(blocks_group) == 0:
        return connected_layout_blocks
    # 将第一个块添加到连接的布局块中
    connected_layout_blocks.append(blocks_group[0])
    # 遍历 blocks_group 中从第 1 个到最后一个元素
    for i in range(1, len(blocks_group)):
        # 尝试捕获可能的异常
        try:
            # 如果当前块为空，则跳过
            if len(blocks_group[i]) == 0:
                continue
            # 如果前一个块为空，则将当前块加入到连接的布局块中
            if len(blocks_group[i - 1]) == 0:  # TODO 考虑连接问题，
                connected_layout_blocks.append(blocks_group[i])
                continue
            # 只有 text 类型的块才考虑合并布局
            if blocks_group[i - 1][-1]["type"] != BlockType.Text or blocks_group[i][0]["type"] != BlockType.Text:
                connected_layout_blocks.append(blocks_group[i])
                continue
            # 如果前一个块的最后一行或当前块的第一行没有行数据，则加入连接的布局块
            if len(blocks_group[i - 1][-1]["lines"]) == 0 or len(blocks_group[i][0]["lines"]) == 0:
                connected_layout_blocks.append(blocks_group[i])
                continue
            # 获取前一个块最后一行和当前块第一行
            pre_last_line = blocks_group[i - 1][-1]["lines"][-1]
            next_first_line = blocks_group[i][0]["lines"][0]
        # 捕获异常并记录日志
        except Exception as e:
            logger.error(f"page layout {i} has no line")
            continue
        # 获取前一行的文本内容
        pre_last_line_text = ''.join([__get_span_text(span) for span in pre_last_line['spans']])
        # 获取前一行最后一个 span 的类型
        pre_last_line_type = pre_last_line['spans'][-1]['type']
        # 获取当前行的文本内容
        next_first_line_text = ''.join([__get_span_text(span) for span in next_first_line['spans']])
        # 获取当前行第一个 span 的类型
        next_first_line_type = next_first_line['spans'][0]['type']
        # 如果前一行或当前行的类型不是文本或行内方程，则将当前块加入连接的布局块
        if pre_last_line_type not in [TEXT, INLINE_EQUATION] or next_first_line_type not in [TEXT, INLINE_EQUATION]:
            connected_layout_blocks.append(blocks_group[i])
            continue
        # 根据前一行的边界框找到布局
        pre_layout = __find_layout_bbox_by_line(pre_last_line['bbox'], new_layout_bbox)
        # 根据当前行的边界框找到布局
        next_layout = __find_layout_bbox_by_line(next_first_line['bbox'], new_layout_bbox)

        # 获取前一行的最大 x2 值，如果没有则设为 -1
        pre_x2_max = pre_layout[2] if pre_layout else -1
        # 获取当前行的最小 x0 值，如果没有则设为 -1
        next_x0_min = next_layout[0] if next_layout else -1

        # 去除前一行文本的空白字符
        pre_last_line_text = pre_last_line_text.strip()
        # 去除当前行文本的空白字符
        next_first_line_text = next_first_line_text.strip()
        # 检查前一行的边界和文本内容，决定是否合并段落
        if pre_last_line['bbox'][2] == pre_x2_max and pre_last_line_text and pre_last_line_text[
            -1] not in LINE_STOP_FLAG and \
                next_first_line['bbox'][0] == next_x0_min:  # 前面一行沾满了整个行，并且没有结尾符号.下一行没有空白开头。
            """连接段落条件成立，将前一个layout的段落和后一个layout的段落连接。"""
            # 将当前块的第一行添加到前一个块的最后一行中
            connected_layout_blocks[-1][-1]["lines"].extend(blocks_group[i][0]["lines"])
            # 清空当前块的第一行的行数据，因为它已经被合并
            blocks_group[i][0]["lines"] = []  
            # 标记当前块的行已删除
            blocks_group[i][0][LINES_DELETED] = True
            # if len(layout_paras[i]) == 0:
            #     layout_paras.pop(i)
            # else:
            #     connected_layout_paras.append(layout_paras[i])
            # 将当前块加入连接的布局块
            connected_layout_blocks.append(blocks_group[i])
        else:
            """连接段落条件不成立，将前一个layout的段落加入到结果中。"""
            # 直接将当前块加入连接的布局块
            connected_layout_blocks.append(blocks_group[i])
    # 返回连接的布局块
    return connected_layout_blocks
# 连接相邻两个页面的段落
def __connect_para_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, page_num,
                              lang):
    """
    连接起来相邻两个页面的段落——前一个页面最后一个段落和后一个页面的第一个段落。
    是否可以连接的条件：
    1. 前一个页面的最后一个段落最后一行沾满整个行。并且没有结尾符号。
    2. 后一个页面的第一个段落第一行没有空白开头。
    """
    # 检查前后页面段落是否为空或缺少文字
    if len(pre_page_paras) == 0 or len(next_page_paras) == 0 or len(pre_page_paras[0]) == 0 or len(
            next_page_paras[0]) == 0:  # TODO [[]]为什么出现在pre_page_paras里？
        return False
    # 获取前一个页面最后一个段落和后一个页面第一个段落
    pre_last_block = pre_page_paras[-1][-1]
    next_first_block = next_page_paras[0][0]
    # 检查最后一个段落和第一个段落的类型是否为文本
    if pre_last_block["type"] != BlockType.Text or next_first_block["type"] != BlockType.Text:
        return False
    # 检查段落是否有行
    if len(pre_last_block["lines"]) == 0 or len(next_first_block["lines"]) == 0:
        return False
    # 获取最后一个段落和第一个段落的行
    pre_last_para = pre_last_block["lines"]
    next_first_para = next_first_block["lines"]
    pre_last_line = pre_last_para[-1]
    next_first_line = next_first_para[0]
    # 获取最后一行的文本内容和类型
    pre_last_line_text = ''.join([__get_span_text(span) for span in pre_last_line['spans']])
    pre_last_line_type = pre_last_line['spans'][-1]['type']
    # 获取第一行的文本内容和类型
    next_first_line_text = ''.join([__get_span_text(span) for span in next_first_line['spans']])
    next_first_line_type = next_first_line['spans'][0]['type']

    # 检查行的类型是否有效
    if pre_last_line_type not in [TEXT, INLINE_EQUATION] or next_first_line_type not in [TEXT,
                                                                                         INLINE_EQUATION]:  # TODO，真的要做好，要考虑跨table, image, 行间的情况
        # 不是文本，不连接
        return False

    # 查找前一行的最大边界框
    pre_x2_max_bbox = __find_layout_bbox_by_line(pre_last_line['bbox'], pre_page_layout_bbox)
    if not pre_x2_max_bbox:
        return False
    # 查找下一行的最小边界框
    next_x0_min_bbox = __find_layout_bbox_by_line(next_first_line['bbox'], next_page_layout_bbox)
    if not next_x0_min_bbox:
        return False

    # 获取最大和最小的边界值
    pre_x2_max = pre_x2_max_bbox[2]
    next_x0_min = next_x0_min_bbox[0]

    # 去除前一行和下一行的空白字符
    pre_last_line_text = pre_last_line_text.strip()
    next_first_line_text = next_first_line_text.strip()
    # 检查连接段落的条件
    if pre_last_line['bbox'][2] == pre_x2_max and pre_last_line_text[-1] not in LINE_STOP_FLAG and \
            next_first_line['bbox'][0] == next_x0_min:  # 前面一行沾满了整个行，并且没有结尾符号.下一行没有空白开头。
        """连接段落条件成立，将前一个layout的段落和后一个layout的段落连接。"""
        # 将下一页的段落标记为跨页面并合并
        for line in next_first_para:
            for span in line["spans"]:
                span[CROSS_PAGE] = True
        pre_last_para.extend(next_first_para)

        # 清空后一个页面的第一个段落的行
        next_page_paras[0][0]["lines"] = []
        next_page_paras[0][0][LINES_DELETED] = True
        return True
    else:
        return False


# 查找连续的真值区域
def find_consecutive_true_regions(input_array):
    start_index = None  # 连续True区域的起始索引
    regions = []  # 用于保存所有连续True区域的起始和结束索引
    # 遍历输入数组的每个元素
    for i in range(len(input_array)):
        # 如果当前元素为True且未进入连续True区域
        if input_array[i] and start_index is None:
            # 记录当前连续True区域的起始索引
            start_index = i  

        # 如果当前元素为False且已在连续True区域中
        elif not input_array[i] and start_index is not None:
            # 检查当前连续True区域的长度是否大于1
            if i - start_index > 1:
                # 将该区域的起始索引和结束索引添加到结果列表
                regions.append((start_index, i - 1))
            # 重置起始索引为None，以开始新的检查
            start_index = None  

    # 如果数组最后一个元素为True，处理最后的连续True区域
    if start_index is not None and len(input_array) - start_index > 1:
        # 将最后一个区域添加到结果列表中
        regions.append((start_index, len(input_array) - 1))

    # 返回所有找到的连续True区域的列表
    return regions
# 定义一个连接中间对齐文本的函数，接收页面段落、布局边界框、页面编号和语言参数
def __connect_middle_align_text(page_paras, new_layout_bbox, page_num, lang):
    # 声明全局变量，用于调试
    global debug_able
    """
    找出来中间对齐的连续单行文本，如果连续行高度相同，那么合并为一个段落。
    一个line居中的条件是：
    1. 水平中心点跨越layout的中心点。
    2. 左右两侧都有空白
    """

    # 遍历每个页面段落及其索引
    for layout_i, layout_para in enumerate(page_paras):
        # 获取当前段落的布局框
        layout_box = new_layout_bbox[layout_i]
        # 初始化单行段落标记列表
        single_line_paras_tag = []
        # 遍历当前段落的每一行
        for i in range(len(layout_para)):
            # 判断每行是否为单行文本，并添加到标记列表
            single_line_paras_tag.append(layout_para[i]['type'] == BlockType.Text and len(layout_para[i]["lines"]) == 1)
        """找出来连续的单行文本，如果连续行高度相同，那么合并为一个段落。"""
        # 找出连续的单行文本的索引
        consecutive_single_line_indices = find_consecutive_true_regions(single_line_paras_tag)
        # 如果找到了连续单行文本
        if len(consecutive_single_line_indices) > 0:
            """检查这些行是否是高度相同的，居中的"""
            # 遍历找到的连续单行文本的起始和结束索引
            for start, end in consecutive_single_line_indices:
                # start += index_offset  # 这行被注释掉
                # end += index_offset    # 这行被注释掉
                # 计算连续行的高度差
                line_hi = np.array([block["lines"][0]['bbox'][3] - block["lines"][0]['bbox'][1] for block in
                                    layout_para[start:end + 1]])
                # 获取首行文本
                first_line_text = ''.join([__get_span_text(span) for span in layout_para[start]["lines"][0]['spans']])
                # 如果首行包含“Table”或“Figure”，则跳过合并
                if "Table" in first_line_text or "Figure" in first_line_text:
                    pass
                # 如果调试模式开启，记录行高的标准差
                if debug_able:
                    logger.info(line_hi.std())

                # 如果行高标准差小于2，则判断是否居中
                if line_hi.std() < 2:
                    """行高度相同，那么判断是否居中"""
                    # 获取所有行的左侧和右侧坐标
                    all_left_x0 = [block["lines"][0]['bbox'][0] for block in layout_para[start:end + 1]]
                    all_right_x1 = [block["lines"][0]['bbox'][2] for block in layout_para[start:end + 1]]
                    # 计算布局的中心点
                    layout_center = (layout_box[0] + layout_box[2]) / 2
                    # 检查所有行是否居中并且左右两侧都有空白
                    if all([x0 < layout_center < x1 for x0, x1 in zip(all_left_x0, all_right_x1)]) \
                            and not all([x0 == layout_box[0] for x0 in all_left_x0]) \
                            and not all([x1 == layout_box[2] for x1 in all_right_x1]):
                        # 合并符合条件的段落行
                        merge_para = [block["lines"][0] for block in layout_para[start:end + 1]]
                        # 获取合并后的文本内容
                        para_text = ''.join([__get_span_text(span) for line in merge_para for span in line['spans']])
                        # 如果调试模式开启，记录合并后的段落文本
                        if debug_able:
                            logger.info(para_text)
                        # 更新首行的文本为合并后的内容
                        layout_para[start]["lines"] = merge_para
                        # 清空后续行的内容，并标记为删除
                        for i_para in range(start + 1, end + 1):
                            layout_para[i_para]["lines"] = []
                            layout_para[i_para][LINES_DELETED] = True
                        # layout_para[start:end + 1] = [merge_para]  # 这行被注释掉

                        # index_offset -= end - start  # 这行被注释掉

    return


# 定义一个合并单行文本的函数，接收页面段落、布局边界框、页面编号和语言参数
def __merge_signle_list_text(page_paras, new_layout_bbox, page_num, lang):
    """
    找出来连续的单行文本，如果首行顶格，接下来的几个单行段落缩进对齐，那么合并为一个段落。
    """

    pass
# 根据行和布局情况进行页面分段
def __do_split_page(blocks, layout_bboxes, new_layout_bbox, page_num, lang):
    # 先实现一个根据行末尾特征分段的简单方法
    """
    算法思路：
    1. 扫描layout里每一行，找出来行尾距离layout有边界有一定距离的行
    2. 从上述行中找到末尾是句号等可作为断行标志的行
    3. 参照上述行尾特征进行分段
    4. 图、表，目前独占一行，不考虑分段
    """
    # 按照布局将行块进行分组
    blocks_group = __group_line_by_layout(blocks, layout_bboxes)  # block内分段
    # 在指定布局框中分割段落
    layout_list_info = __split_para_in_layoutbox(blocks_group, new_layout_bbox, lang)  # layout内分段
    # 在不同布局之间连接列表段落
    blocks_group, page_list_info = __connect_list_inter_layout(blocks_group, new_layout_bbox, layout_list_info,
                                                               page_num, lang)  # layout之间连接列表段落
    # 在布局框之间链接段落
    connected_layout_blocks = __connect_para_inter_layoutbox(blocks_group, new_layout_bbox)  # layout间链接段落

    # 返回连接后的布局块和页面信息
    return connected_layout_blocks, page_list_info


# 分段函数，处理 PDF 信息字典
def para_split(pdf_info_dict, debug_mode, lang="en"):
    global debug_able  # 声明全局变量以控制调试模式
    debug_able = debug_mode  # 设置调试模式
    new_layout_of_pages = []  # 数组的数组，每个元素是一个页面的layout
    all_page_list_info = []  # 保存每个页面开头和结尾是否是列表
    # 遍历每一页的 PDF 信息
    for page_num, page in pdf_info_dict.items():
        # 深拷贝预处理块以防止修改原数据
        blocks = copy.deepcopy(page['preproc_blocks'])
        layout_bboxes = page['layout_bboxes']  # 获取当前页面的布局边界框
        # 对块和布局框进行常见预处理
        new_layout_bbox = __common_pre_proc(blocks, layout_bboxes)
        # 将新的布局框添加到页面布局列表
        new_layout_of_pages.append(new_layout_bbox)
        # 根据行和布局情况进行分段
        splited_blocks, page_list_info = __do_split_page(blocks, layout_bboxes, new_layout_bbox, page_num, lang)
        # 将页面信息保存到列表
        all_page_list_info.append(page_list_info)
        page['para_blocks'] = splited_blocks  # 将分段结果存储到页面中

    # 连接页面与页面之间的可能合并的段落
    pdf_infos = list(pdf_info_dict.values())  # 获取所有页面信息的列表
    # 遍历每一页以连接段落
    for page_num, page in enumerate(pdf_info_dict.values()):
        if page_num == 0:  # 跳过第一页
            continue
        pre_page_paras = pdf_infos[page_num - 1]['para_blocks']  # 获取前一页的段落
        next_page_paras = pdf_infos[page_num]['para_blocks']  # 获取当前页的段落
        pre_page_layout_bbox = new_layout_of_pages[page_num - 1]  # 前一页的布局框
        next_page_layout_bbox = new_layout_of_pages[page_num]  # 当前页的布局框

        # 检查并连接前后页之间的段落
        is_conn = __connect_para_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox,
                                            next_page_layout_bbox, page_num, lang)
        # 如果在调试模式下，记录连接信息
        if debug_able:
            if is_conn:
                logger.info(f"连接了第{page_num - 1}页和第{page_num}页的段落")

        # 检查并连接前后页之间的列表段落
        is_list_conn = __connect_list_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox,
                                                 next_page_layout_bbox, all_page_list_info[page_num - 1],
                                                 all_page_list_info[page_num], page_num, lang)
        # 如果在调试模式下，记录连接信息
        if debug_able:
            if is_list_conn:
                logger.info(f"连接了第{page_num - 1}页和第{page_num}页的列表段落")

    # 接下来可能会漏掉一些特别的一些可以合并的内容，对他们进行段落连接
    # 1. 正文中有时出现一个行顶格，接下来几行缩进的情况
    # 2. 居中的一些连续单行，如果高度相同，那么可能是一个段落
    # 遍历 pdf_info_dict 字典的值，获取每一页的信息及其页码
        for page_num, page in enumerate(pdf_info_dict.values()):
            # 获取当前页的段落块
            page_paras = page['para_blocks']
            # 获取新布局的边界框，基于当前页的页码
            new_layout_bbox = new_layout_of_pages[page_num]
            # 连接中间对齐的文本，使用当前页的段落和布局边界框
            __connect_middle_align_text(page_paras, new_layout_bbox, page_num, lang)
            # 合并单个文本块，使用当前页的段落和布局边界框
            __merge_signle_list_text(page_paras, new_layout_bbox, page_num, lang)
    
        # 对布局进行展平处理
        for page_num, page in enumerate(pdf_info_dict.values()):
            # 获取当前页的段落块
            page_paras = page['para_blocks']
            # 将当前页的所有段落块展平为单一列表
            page_blocks = [block for layout in page_paras for block in layout]
            # 更新当前页的段落块为展平后的列表
            page["para_blocks"] = page_blocks
```