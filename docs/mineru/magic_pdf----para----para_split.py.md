# `.\MinerU\magic_pdf\para\para_split.py`

```
# 从 sklearn 库导入 DBSCAN 聚类算法
from sklearn.cluster import DBSCAN
# 导入 numpy 库以支持数组和数学操作
import numpy as np
# 从 loguru 库导入 logger 用于日志记录
from loguru import logger

# 导入自定义函数，检查元素是否在布局区域内或部分重叠
from magic_pdf.libs.boxbase import _is_in_or_part_overlap_with_area_ratio as is_in_layout
# 导入内容类型定义
from magic_pdf.libs.ocr_content_type import ContentType

# 定义表示行结束的标志符号列表
LINE_STOP_FLAG = ['.', '!', '?', '。', '！', '？',"：", ":", ")", "）", ";"]
# 定义内容类型常量：内联方程
INLINE_EQUATION = ContentType.InlineEquation
# 定义内容类型常量：行间方程
INTERLINE_EQUATION = ContentType.InterlineEquation
# 定义内容类型常量：文本
TEXT = ContentType.Text

# 定义获取文本内容的辅助函数
def __get_span_text(span):
    # 从 span 中获取内容，如果没有则尝试获取图片路径
    c = span.get('content', '')
    if len(c)==0:
        c = span.get('image_path', '')
    # 返回获取到的内容
    return c

# 定义检测列表行的辅助函数
def __detect_list_lines(lines, new_layout_bboxes, lang):
    """
    探测是否包含了列表，并且把列表的行分开.
    这样的段落特点是，顶格字母大写/数字，紧跟着几行缩进的。缩进的行首字母含小写的。
    """
    # 定义查找重复模式的内部函数
    def find_repeating_patterns(lst):
        indices = []  # 存储匹配到的模式索引
        ones_indices = []  # 存储包含1的索引
        i = 0  # 初始化循环索引
        while i < len(lst) - 1:  # 确保余下元素至少有2个
            if lst[i] == 1 and lst[i+1] in [2, 3]:  # 检查是否以1开始，并跟随2或3
                start = i  # 记录匹配模式的起始索引
                ones_in_this_interval = [i]  # 初始化当前区间内的1的索引
                i += 1  # 移动到下一个元素
                while i < len(lst) and lst[i] in [2, 3]:  # 遍历跟随的2和3
                    i += 1
                # 验证下一个序列是否符合条件
                if i < len(lst) - 1 and lst[i] == 1 and lst[i+1] in [2, 3] and lst[i-1] in [2, 3]:
                    while i < len(lst) and lst[i] in [1, 2, 3]:  # 查找连续的1、2、3
                        if lst[i] == 1:
                            ones_in_this_interval.append(i)  # 添加1的索引
                        i += 1
                    indices.append((start, i - 1))  # 添加匹配模式的起始和结束索引
                    ones_indices.append(ones_in_this_interval)  # 添加1的索引列表
                else:
                    i += 1  # 否则移动到下一个元素
            else:
                i += 1  # 否则移动到下一个元素
        # 返回匹配模式的索引和1的索引列表
        return indices, ones_indices
    
    """===================="""
    
    # 定义分割索引的内部函数
    def split_indices(slen, index_array):
        result = []  # 初始化结果列表
        last_end = 0  # 记录上一个区间的结束位置
        
        for start, end in sorted(index_array):  # 遍历排序后的索引数组
            if start > last_end:
                # 前一个区间结束到下一个区间开始之间的部分标记为"text"
                result.append(('text', last_end, start - 1))
            # 区间内标记为"list"
            result.append(('list', start, end))
            last_end = end + 1  # 更新上一个区间的结束位置

        if last_end < slen:
            # 如果最后一个区间结束后还有剩余的字符串，将其标记为"text"
            result.append(('text', last_end, slen - 1))

        # 返回分割后的结果
        return result
    
    """===================="""

    # 如果语言不是英语，直接返回原行和 None
    if lang!='en':
        return lines, None
    else:
        # 获取行的总数
        total_lines = len(lines)
        # 初始化特征编码列表
        line_fea_encode = []
        """
        对每一行进行特征编码，编码规则如下：
        1. 如果行顶格，且大写字母开头或者数字开头，编码为1
        2. 如果顶格，其他非大写开头编码为4
        3. 如果非顶格，首字符大写，编码为2
        4. 如果非顶格，首字符非大写编码为3
        """
        # 遍历每一行进行编码
        for l in lines:
            # 获取当前行的首字符
            first_char = __get_span_text(l['spans'][0])[0]
            # 查找当前行的布局左侧边界
            layout_left = __find_layout_bbox_by_line(l['bbox'], new_layout_bboxes)[0]
            # 判断当前行是否顶格
            if l['bbox'][0] == layout_left:
                # 如果首字符为大写字母或数字，编码为1
                if first_char.isupper() or first_char.isdigit():
                    line_fea_encode.append(1)
                # 否则编码为4
                else:
                    line_fea_encode.append(4)
            else:
                # 如果非顶格，首字符为大写，编码为2
                if first_char.isupper():
                    line_fea_encode.append(2)
                # 否则编码为3
                else:
                    line_fea_encode.append(3)
                    
        # 根据编码寻找连续出现至少2次的行，认为是列表
        list_indice, list_start_idx  = find_repeating_patterns(line_fea_encode)
        # 如果找到列表，记录日志
        if len(list_indice) > 0:
            logger.info(f"发现了列表，列表行数：{list_indice}， {list_start_idx}")
        
        # TODO 检查特列表里缩进行左侧是否对齐
        segments = []
        # 遍历找到的列表索引
        for start, end in list_indice:
            # 检查列表行的每一行
            for i in range(start, end + 1):
                # 如果不是第一行
                if i > 0:
                    # 检查是否有行不是顶格的
                    if line_fea_encode[i] == 4:
                        logger.info(f"列表行的第{i}行不是顶格的")
                        break
            else:
                # 如果所有行都是顶格，记录日志
                logger.info(f"列表行的第{start}到第{end}行是列表")
        
        # 返回分段的索引和列表起始索引
        return split_indices(total_lines, list_indice), list_start_idx
# 在一个layoutbox内对齐行的左侧和右侧
def __valign_lines(blocks, layout_bboxes):
    # 定义用于对齐的最小距离阈值
    min_distance = 3
    # 定义最小样本数量以进行对齐
    min_sample = 2
    # 初始化一个空列表以存储新的布局边界框
    new_layout_bboxes = []
    # 遍历每个布局框
        for layout_box in layout_bboxes:
            # 过滤出在当前布局框内的所有块
            blocks_in_layoutbox = [b for b in blocks if is_in_layout(b['bbox'], layout_box['layout_bbox'])]
            # 如果没有块在布局框内，则跳过本次循环
            if len(blocks_in_layoutbox)==0:
                continue
            
            # 提取所有块的左边界坐标并构造数组
            x0_lst = np.array([[line['bbox'][0], 0] for block in blocks_in_layoutbox for line in block['lines']])
            # 提取所有块的右边界坐标并构造数组
            x1_lst = np.array([[line['bbox'][2], 0] for block in blocks_in_layoutbox for line in block['lines']])
            # 对左边界坐标进行聚类
            x0_clusters = DBSCAN(eps=min_distance, min_samples=min_sample).fit(x0_lst)
            # 对右边界坐标进行聚类
            x1_clusters = DBSCAN(eps=min_distance, min_samples=min_sample).fit(x1_lst)
            # 获取左边界聚类的唯一标签
            x0_uniq_label = np.unique(x0_clusters.labels_)
            # 获取右边界聚类的唯一标签
            x1_uniq_label = np.unique(x1_clusters.labels_)
            
            # 存储左边界旧值到新值的映射
            x0_2_new_val = {} 
            # 存储右边界旧值到新值的映射
            x1_2_new_val = {}
            # 更新左边界的新值映射
            for label in x0_uniq_label:
                if label==-1:
                    continue
                x0_index_of_label = np.where(x0_clusters.labels_==label)
                x0_raw_val = x0_lst[x0_index_of_label][:,0]
                x0_new_val = np.min(x0_lst[x0_index_of_label][:,0])
                x0_2_new_val.update({idx: x0_new_val for idx in x0_raw_val})
            # 更新右边界的新值映射
            for label in x1_uniq_label:
                if label==-1:
                    continue
                x1_index_of_label = np.where(x1_clusters.labels_==label)
                x1_raw_val = x1_lst[x1_index_of_label][:,0]
                x1_new_val = np.max(x1_lst[x1_index_of_label][:,0])
                x1_2_new_val.update({idx: x1_new_val for idx in x1_raw_val})
            
            # 更新每个块中行的边界坐标
            for block in blocks_in_layoutbox:
                for line in block['lines']:
                    x0, x1 = line['bbox'][0], line['bbox'][2]
                    # 如果左边界有新的值，更新之
                    if x0 in x0_2_new_val:
                        line['bbox'][0] = int(x0_2_new_val[x0])
    
                    # 如果右边界有新的值，更新之
                    if x1 in x1_2_new_val:
                        line['bbox'][2] = int(x1_2_new_val[x1])
                # 其余对不齐的保持不动
                
            # 由于行的边界变化，重新计算每个块的边界框
            for block in blocks_in_layoutbox:
                block['bbox'] = [min([line['bbox'][0] for line in block['lines']]), 
                                min([line['bbox'][1] for line in block['lines']]), 
                                max([line['bbox'][2] for line in block['lines']]), 
                                max([line['bbox'][3] for line in block['lines']])]
                
            # 计算新的布局框，基于块的边界框
            layout_x0 = min([block['bbox'][0] for block in blocks_in_layoutbox])
            layout_y0 = min([block['bbox'][1] for block in blocks_in_layoutbox])
            layout_x1 = max([block['bbox'][2] for block in blocks_in_layoutbox])
            layout_y1 = max([block['bbox'][3] for block in blocks_in_layoutbox])
            # 将新的布局框添加到结果列表中
            new_layout_bboxes.append([layout_x0, layout_y0, layout_x1, layout_y1])
                
        # 返回更新后的布局框列表
        return new_layout_bboxes
# 对文本块进行对齐处理，限制在给定的布局边界内
def __align_text_in_layout(blocks, layout_bboxes):
    """
    由于ocr出来的line，有时候会在前后有一段空白，这个时候需要对文本进行对齐，超出的部分被layout左右侧截断。
    """
    # 遍历每个布局框
    for layout in layout_bboxes:
        # 获取当前布局的边界框
        lb = layout['layout_bbox']
        # 筛选出在当前布局框内的文本块
        blocks_in_layoutbox = [b for b in blocks if is_in_layout(b['bbox'], lb)]
        # 如果没有文本块在当前布局框内，继续下一个布局
        if len(blocks_in_layoutbox) == 0:
            continue
        
        # 遍历当前布局框内的所有文本块
        for block in blocks_in_layoutbox:
            # 遍历文本块中的所有行
            for line in block['lines']:
                # 获取行的左右边界
                x0, x1 = line['bbox'][0], line['bbox'][2]
                # 如果行的左边界小于布局框的左边界，则对齐到左边界
                if x0 < lb[0]:
                    line['bbox'][0] = lb[0]
                # 如果行的右边界大于布局框的右边界，则对齐到右边界
                if x1 > lb[2]:
                    line['bbox'][2] = lb[2]

# 对文本块和布局框进行通用预处理
def __common_pre_proc(blocks, layout_bboxes):
    """
    不分语言的，对文本进行预处理
    """
    # 调用对齐文本的函数
    # __add_line_period(blocks, layout_bboxes)
    __align_text_in_layout(blocks, layout_bboxes)
    # 调用垂直对齐行的函数
    aligned_layout_bboxes = __valign_lines(blocks, layout_bboxes)
    
    # 返回对齐后的布局框
    return aligned_layout_bboxes

# 对中文文本块进行预处理的占位函数
def __pre_proc_zh_blocks(blocks, layout_bboxes):
    """
    对中文文本进行分段预处理
    """
    pass

# 对英文文本块进行预处理的占位函数
def __pre_proc_en_blocks(blocks, layout_bboxes):
    """
    对英文文本进行分段预处理
    """
    pass

# 根据布局聚合每个块内的行
def __group_line_by_layout(blocks, layout_bboxes, lang="en"):
    """
    每个layout内的行进行聚合
    """
    # 初始化行组列表
    lines_group = []
    
    # 遍历每个布局框
    for lyout in layout_bboxes:
        # 获取在当前布局框内的所有行
        lines = [line for block in blocks if is_in_layout(block['bbox'], lyout['layout_bbox']) for line in block['lines']]
        # 将行添加到行组列表中
        lines_group.append(lines)

    # 返回行组
    return lines_group

# 在布局框内进行行的分段处理
def __split_para_in_layoutbox(lines_group, new_layout_bbox, lang="en", char_avg_len=10):
    """
    lines_group 进行行分段——layout内部进行分段。lines_group内每个元素是一个Layoutbox内的所有行。
    1. 先计算每个group的左右边界。
    2. 然后根据行末尾特征进行分段。
        末尾特征：以句号等结束符结尾。并且距离右侧边界有一定距离。
        且下一行开头不留空白。
    
    """
    # 初始化布局段落信息和列表信息
    list_info = [] # 这个layout最后是不是列表,记录每一个layout里是不是列表开头，列表结尾
    layout_paras = []
    # 设定右侧尾部距离
    right_tail_distance = 1.5 * char_avg_len
    
    # 返回段落和列表信息
    return layout_paras, list_info

# 连接跨布局的列表段落
def __connect_list_inter_layout(layout_paras, new_layout_bbox, layout_list_info, page_num, lang):
    """
    如果上个layout的最后一个段落是列表，下一个layout的第一个段落也是列表，那么将他们连接起来。 TODO 因为没有区分列表和段落，所以这个方法暂时不实现。
    根据layout_list_info判断是不是列表。，下个layout的第一个段如果不是列表，那么看他们是否有几行都有相同的缩进。
    """
    # 如果段落或列表信息为空，返回原始信息
    if len(layout_paras) == 0 or len(layout_list_info) == 0: # 0的时候最后的return 会出错
        return layout_paras, [False, False]
    # 遍历 layout_paras 列表，索引从 1 开始
        for i in range(1, len(layout_paras)):
            # 获取前一个布局列表的信息
            pre_layout_list_info = layout_list_info[i-1]
            # 获取当前布局列表的信息
            next_layout_list_info = layout_list_info[i]
            # 获取前一个段落的最后一项
            pre_last_para = layout_paras[i-1][-1]
            # 获取当前布局的段落列表
            next_paras = layout_paras[i]
            # 获取当前段落列表的第一项
            next_first_para = next_paras[0]
            
            # 检测前一个布局是列表结尾而后一个布局不是列表开头的情况
            if pre_layout_list_info[1] and not next_layout_list_info[0]: # 前一个是列表结尾，后一个是非列表开头，此时检测是否有相同的缩进
                # 记录连接页面的信息
                logger.info(f"连接page {page_num} 内的list")
                # 初始化一个空列表以存储可能的列表行
                may_list_lines = []
                # 遍历当前段落列表
                for j in range(len(next_paras)):
                    # 获取当前行
                    line = next_paras[j]
                    # 检测行是否只有一项
                    if len(line)==1: # 只可能是一行，多行情况再需要分析了
                        # 检测当前行的边界框是否在新的布局边界框之右
                        if line[0]['bbox'][0] > __find_layout_bbox_by_line(line[0]['bbox'], new_layout_bbox)[0]:
                            # 将符合条件的行添加到可能的列表行中
                            may_list_lines.append(line[0])
                        else:
                            # 如果不符合条件，结束循环
                            break
                    else:
                        # 如果当前行有多项，结束循环
                        break
                # 检查可能的列表行是否具有相同的缩进
                if len(may_list_lines)>0 and len(set([x['bbox'][0] for x in may_list_lines]))==1:
                    # 将可能的列表行扩展到前一个段落的最后
                    pre_last_para.extend(may_list_lines)
                    # 更新当前布局的段落列表，移除已连接的行
                    layout_paras[i] = layout_paras[i][len(may_list_lines):]
                               
        # 返回更新后的段落列表和页面级别的开头、结尾信息
        return layout_paras, [layout_list_info[0][0], layout_list_info[-1][1]] # 同时还返回了这个页面级别的开头、结尾是不是列表的信息
# 连接两个页面之间的列表段落，如果满足特定条件
def __connect_list_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox,  pre_page_list_info, next_page_list_info, page_num, lang):
    """
    如果上个layout的最后一个段落是列表，下一个layout的第一个段落也是列表，那么将他们连接起来。 
    TODO 因为没有区分列表和段落，所以这个方法暂时不实现。
    根据layout_list_info判断是不是列表。，下个layout的第一个段如果不是列表，那么看他们是否有几行都有相同的缩进。
    """
    # 如果前一页或下一页的段落列表为空，则返回 False
    if len(pre_page_paras)==0 or len(next_page_paras)==0: # 0的时候最后的return 会出错
        return False
    
    # 检查前一段是否为列表的结尾，后一段是否为非列表的开头
    if pre_page_list_info[1] and not next_page_list_info[0]: # 前一个是列表结尾，后一个是非列表开头，此时检测是否有相同的缩进
        # 记录需要连接的行
        logger.info(f"连接page {page_num} 内的list")
        # 初始化可能的列表行
        may_list_lines = []
        # 遍历下一页的第一个段落的每一行
        for j in range(len(next_page_paras[0])):
            line = next_page_paras[0][j]
            # 如果这一行只有一行内容
            if len(line)==1: # 只可能是一行，多行情况再需要分析了
                # 检查当前行的缩进是否符合条件
                if line[0]['bbox'][0] > __find_layout_bbox_by_line(line[0]['bbox'], next_page_layout_bbox)[0]:
                    may_list_lines.append(line[0]) # 添加符合条件的行
                else:
                    break # 如果不符合条件，则终止循环
            else:
                break # 如果有多行内容，直接终止循环
        # 如果找到的行具有相同的缩进，则连接到前一段落
        if len(may_list_lines)>0 and len(set([x['bbox'][0] for x in may_list_lines]))==1:
            pre_page_paras[-1].append(may_list_lines) # 将行添加到前一段落
            next_page_paras[0] = next_page_paras[0][len(may_list_lines):] # 更新下一段落
            return True # 返回连接成功
    
    return False # 如果未满足连接条件，则返回 False


# 根据行的边界框查找其所在的布局
def __find_layout_bbox_by_line(line_bbox, layout_bboxes):
    """
    根据line找到所在的layout
    """
    # 遍历每个布局框
    for layout in layout_bboxes:
        # 检查当前行是否在布局中
        if is_in_layout(line_bbox, layout):
            return layout # 如果在，则返回该布局
    return None # 如果没有找到，则返回 None


# 在不同布局之间连接段落
def __connect_para_inter_layoutbox(layout_paras, new_layout_bbox, lang):
    """
    layout之间进行分段。
    主要是计算前一个layOut的最后一行和后一个layout的第一行是否可以连接。
    连接的条件需要同时满足：
    1. 上一个layout的最后一行沾满整个行。并且没有结尾符号。
    2. 下一行开头不留空白。
    """
    # 初始化连接的布局段落列表
    connected_layout_paras = []
    # 如果布局段落为空，则返回空列表
    if len(layout_paras)==0:
        return connected_layout_paras
    
    # 将第一个布局段落添加到连接列表中
    connected_layout_paras.append(layout_paras[0])
    # 遍历 layout_paras 列表，忽略第一个元素
    for i in range(1, len(layout_paras)):
        try:
            # 如果当前段落或前一个段落为空，跳过当前循环
            if len(layout_paras[i])==0 or len(layout_paras[i-1])==0: #  TODO 考虑连接问题，
                continue
            # 获取前一个段落的最后一行
            pre_last_line = layout_paras[i-1][-1][-1]
            # 获取当前段落的第一行
            next_first_line = layout_paras[i][0][0]
        except Exception as e:
            # 如果发生异常，记录错误信息并跳过当前循环
            logger.error(f"page layout {i} has no line")
            continue
        # 将前一行的所有 span 文本连接成字符串
        pre_last_line_text = ''.join([__get_span_text(span) for span in pre_last_line['spans']])
        # 获取前一行最后一个 span 的类型
        pre_last_line_type = pre_last_line['spans'][-1]['type']
        # 将当前行的所有 span 文本连接成字符串
        next_first_line_text = ''.join([__get_span_text(span) for span in next_first_line['spans']])
        # 获取当前行第一个 span 的类型
        next_first_line_type = next_first_line['spans'][0]['type']
        # 检查前后行的类型是否在可连接的类型中
        if pre_last_line_type not in [TEXT, INLINE_EQUATION] or next_first_line_type not in [TEXT, INLINE_EQUATION]:
            # 如果不在，则将当前段落添加到连接段落列表
            connected_layout_paras.append(layout_paras[i])
            continue
        
        # 找到前一行的最大 x 坐标
        pre_x2_max = __find_layout_bbox_by_line(pre_last_line['bbox'], new_layout_bbox)[2]
        # 找到当前行的最小 x 坐标
        next_x0_min = __find_layout_bbox_by_line(next_first_line['bbox'], new_layout_bbox)[0]
        
        # 去除前一行文本首尾空格
        pre_last_line_text = pre_last_line_text.strip()
        next_first_line_text = next_first_line_text.strip()
        # 检查连接条件是否满足
        if pre_last_line['bbox'][2] == pre_x2_max and pre_last_line_text[-1] not in LINE_STOP_FLAG and next_first_line['bbox'][0]==next_x0_min: # 前面一行沾满了整个行，并且没有结尾符号.下一行没有空白开头。
            """连接段落条件成立，将前一个layout的段落和后一个layout的段落连接。"""
            # 将当前段落的第一行添加到前一个段落的最后一行
            connected_layout_paras[-1][-1].extend(layout_paras[i][0])
            # 删除当前段落的第一行
            layout_paras[i].pop(0) # 删除后一个layout的第一个段落， 因为他已经被合并到前一个layout的最后一个段落了。
            # 如果当前段落为空，则从列表中移除
            if len(layout_paras[i])==0:
                layout_paras.pop(i)
            else:
                # 否则将当前段落添加到连接段落列表
                connected_layout_paras.append(layout_paras[i])
        else:                            
            """连接段落条件不成立，将前一个layout的段落加入到结果中。"""
            # 将当前段落添加到连接段落列表
            connected_layout_paras.append(layout_paras[i])
    
    # 返回连接后的段落列表
    return connected_layout_paras
# 连接相邻两个页面的段落，判断连接条件
def __connect_para_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, page_num, lang):
    # 文档字符串，描述连接相邻页面段落的条件
    """
    连接起来相邻两个页面的段落——前一个页面最后一个段落和后一个页面的第一个段落。
    是否可以连接的条件：
    1. 前一个页面的最后一个段落最后一行沾满整个行。并且没有结尾符号。
    2. 后一个页面的第一个段落第一行没有空白开头。
    """
    # 检查页面是否没有段落内容
    if len(pre_page_paras)==0 or len(next_page_paras)==0 or len(pre_page_paras[0])==0 or len(next_page_paras[0])==0: # TODO [[]]为什么出现在pre_page_paras里？
        return False  # 返回False表示无法连接
    pre_last_para = pre_page_paras[-1][-1]  # 获取前页面最后一个段落
    next_first_para = next_page_paras[0][0]  # 获取后页面第一个段落
    pre_last_line = pre_last_para[-1]  # 获取前页面最后段落的最后一行
    next_first_line = next_first_para[0]  # 获取后页面第一个段落的第一行
    # 将前最后一行的所有文本拼接成一个字符串
    pre_last_line_text = ''.join([__get_span_text(span) for span in pre_last_line['spans']])
    pre_last_line_type = pre_last_line['spans'][-1]['type']  # 获取前最后一行的类型
    # 将后第一行的所有文本拼接成一个字符串
    next_first_line_text = ''.join([__get_span_text(span) for span in next_first_line['spans']])
    next_first_line_type = next_first_line['spans'][0]['type']  # 获取后第一行的类型
    
    # 检查行类型是否为文本或内联方程
    if pre_last_line_type not in [TEXT, INLINE_EQUATION] or next_first_line_type not in [TEXT, INLINE_EQUATION]: # TODO，真的要做好，要考虑跨table, image, 行间的情况
        return False  # 返回False表示无法连接
    # 计算前最后一行的右边界
    pre_x2_max = __find_layout_bbox_by_line(pre_last_line['bbox'], pre_page_layout_bbox)[2]
    # 计算后第一行的左边界
    next_x0_min = __find_layout_bbox_by_line(next_first_line['bbox'], next_page_layout_bbox)[0]
    
    pre_last_line_text = pre_last_line_text.strip()  # 去掉前最后一行的前后空白
    next_first_line_text = next_first_line_text.strip()  # 去掉后第一行的前后空白
    # 检查前最后一行是否沾满行且没有结尾符号，后第一行是否没有空白开头
    if pre_last_line['bbox'][2] == pre_x2_max and pre_last_line_text[-1] not in LINE_STOP_FLAG and next_first_line['bbox'][0]==next_x0_min: # 前面一行沾满了整个行，并且没有结尾符号.下一行没有空白开头。
        """连接段落条件成立，将前一个layout的段落和后一个layout的段落连接。"""
        pre_last_para.extend(next_first_para)  # 连接段落
        next_page_paras[0].pop(0)  # 删除后一个页面的第一个段落， 因为他已经被合并到前一个页面的最后一个段落了。
        return True  # 返回True表示连接成功
    else:
        return False  # 返回False表示无法连接

# 查找连续的True区域
def find_consecutive_true_regions(input_array):
    start_index = None  # 连续True区域的起始索引
    regions = []  # 用于保存所有连续True区域的起始和结束索引

    for i in range(len(input_array)):  # 遍历输入数组
        # 如果我们找到了一个True值，并且当前并没有在连续True区域中
        if input_array[i] and start_index is None:
            start_index = i  # 记录连续True区域的起始索引

        # 如果我们找到了一个False值，并且当前在连续True区域中
        elif not input_array[i] and start_index is not None:
            # 如果连续True区域长度大于1，那么将其添加到结果列表中
            if i - start_index > 1: 
                regions.append((start_index, i-1))  # 添加连续True区域的索引
            start_index = None  # 重置起始索引

    # 如果最后一个元素是True，那么需要将最后一个连续True区域加入到结果列表中
    if start_index is not None and len(input_array) - start_index > 1:
        regions.append((start_index, len(input_array)-1))  # 添加最后一个True区域的索引

    return regions  # 返回所有连续True区域的列表

# 查找中间对齐的连续单行文本
def __connect_middle_align_text(page_paras, new_layout_bbox, page_num, lang, debug_mode):
    # 文档字符串，描述如何合并中间对齐的文本行
    """
    找出来中间对齐的连续单行文本，如果连续行高度相同，那么合并为一个段落。
    一个line居中的条件是：
    1. 水平中心点跨越layout的中心点。
    2. 左右两侧都有空白
    """
    # 遍历每个页面的段落
    for layout_i, layout_para in enumerate(page_paras):
        # 获取当前布局框
        layout_box = new_layout_bbox[layout_i]
        # 初始化单行段落标记列表
        single_line_paras_tag = []
        # 遍历当前布局段落
        for i in range(len(layout_para)):
            # 检查当前段落是否为单行文本并标记
            single_line_paras_tag.append(len(layout_para[i])==1 and layout_para[i][0]['spans'][0]['type']==TEXT)
            
        """找出来连续的单行文本，如果连续行高度相同，那么合并为一个段落。"""
        # 获取连续单行段落的索引
        consecutive_single_line_indices = find_consecutive_true_regions(single_line_paras_tag)
        # 如果有找到的连续单行段落
        if len(consecutive_single_line_indices)>0:
            index_offset = 0
            """检查这些行是否是高度相同的，居中的"""
            # 遍历连续单行段落的起止索引
            for start, end in consecutive_single_line_indices:
                # 更新起始和结束索引考虑偏移
                start += index_offset
                end += index_offset
                # 计算当前段落的行高
                line_hi = np.array([line[0]['bbox'][3]-line[0]['bbox'][1] for line in layout_para[start:end+1]])
                # 获取首行文本内容
                first_line_text = ''.join([__get_span_text(span) for span in layout_para[start][0]['spans']])
                # 如果首行文本包含"Table"或"Figure"，则跳过
                if "Table" in first_line_text or "Figure" in first_line_text:
                    pass
                # 如果调试模式开启，记录行高标准差
                if debug_mode:
                    logger.debug(line_hi.std())
                
                # 如果行高标准差小于2
                if line_hi.std()<2:
                    """行高度相同，那么判断是否居中"""
                    # 获取所有左边界和右边界的x坐标
                    all_left_x0 = [line[0]['bbox'][0] for line in layout_para[start:end+1]]
                    all_right_x1 = [line[0]['bbox'][2] for line in layout_para[start:end+1]]
                    # 计算布局中心
                    layout_center = (layout_box[0] + layout_box[2]) / 2
                    # 检查所有行是否居中且不全为边界
                    if all([x0 < layout_center < x1 for x0, x1 in zip(all_left_x0, all_right_x1)]) \
                    and not all([x0==layout_box[0] for x0 in all_left_x0]) \
                    and not all([x1==layout_box[2] for x1 in all_right_x1]):
                        # 合并符合条件的段落
                        merge_para = [l[0] for l in layout_para[start:end+1]]
                        para_text = ''.join([__get_span_text(span) for line in merge_para for span in line['spans']])
                        # 如果调试模式开启，记录合并后的段落文本
                        if debug_mode:
                            logger.debug(para_text)
                        # 将合并后的段落替换原有段落
                        layout_para[start:end+1] = [merge_para]
                        # 更新索引偏移
                        index_offset -= end-start
                        
    # 返回，结束函数
    return
# 定义一个私有函数，用于合并单行文本段落
def __merge_signle_list_text(page_paras, new_layout_bbox, page_num, lang):
    """
    找出来连续的单行文本，如果首行顶格，接下来的几个单行段落缩进对齐，那么合并为一个段落。
    """
    
    # 函数体为空，待实现
    pass


# 定义一个私有函数，用于根据行和布局情况进行页面分段
def __do_split_page(blocks, layout_bboxes, new_layout_bbox, page_num, lang):
    """
    根据line和layout情况进行分段
    先实现一个根据行末尾特征分段的简单方法。
    """
    """
    算法思路：
    1. 扫描layout里每一行，找出来行尾距离layout有边界有一定距离的行。
    2. 从上述行中找到末尾是句号等可作为断行标志的行。
    3. 参照上述行尾特征进行分段。
    4. 图、表，目前独占一行，不考虑分段。
    """
    # 检查当前页是否为343，如果是，则不进行处理
    if page_num==343:
        pass
    # 根据布局将行进行分组
    lines_group = __group_line_by_layout(blocks, layout_bboxes, lang) # block内分段
    # 在布局框内对段落进行拆分
    layout_paras, layout_list_info = __split_para_in_layoutbox(lines_group, new_layout_bbox, lang) # layout内分段
    # 连接布局之间的列表段落
    layout_paras2, page_list_info = __connect_list_inter_layout(layout_paras, new_layout_bbox, layout_list_info, page_num, lang) # layout之间连接列表段落
    # 连接布局框内的段落
    connected_layout_paras = __connect_para_inter_layoutbox(layout_paras2, new_layout_bbox, lang) # layout间链接段落
    
    
    # 返回连接后的段落和页面列表信息
    return connected_layout_paras, page_list_info
       
    
# 定义一个用于段落拆分的函数
def para_split(pdf_info_dict, debug_mode, lang="en"):
    """
    根据line和layout情况进行分段
    """
    # 初始化一个数组，存储每个页面的布局
    new_layout_of_pages = [] # 数组的数组，每个元素是一个页面的layoutS
    # 初始化一个数组，保存每个页面的列表信息
    all_page_list_info = [] # 保存每个页面开头和结尾是否是列表
    # 遍历每个页面的编号和内容
    for page_num, page in pdf_info_dict.items():
        # 获取页面中的预处理块和布局框
        blocks = page['preproc_blocks']
        layout_bboxes = page['layout_bboxes']
        # 对块和布局框进行公共预处理
        new_layout_bbox = __common_pre_proc(blocks, layout_bboxes)
        # 将新的布局框添加到页面布局数组中
        new_layout_of_pages.append(new_layout_bbox)
        # 根据行和布局情况拆分块，并获取页面列表信息
        splited_blocks, page_list_info = __do_split_page(blocks, layout_bboxes, new_layout_bbox, page_num, lang)
        # 将页面列表信息添加到总信息中
        all_page_list_info.append(page_list_info)
        # 将拆分后的块存储回页面中
        page['para_blocks'] = splited_blocks
        
    """连接页面与页面之间的可能合并的段落"""
    # 将 PDF 信息转换为列表形式
    pdf_infos = list(pdf_info_dict.values())
    # 遍历每个页面的编号和内容
    for page_num, page in enumerate(pdf_info_dict.values()):
        # 跳过第一页
        if page_num==0:
            continue
        # 获取前一页和当前页的段落
        pre_page_paras = pdf_infos[page_num-1]['para_blocks']
        next_page_paras = pdf_infos[page_num]['para_blocks']
        # 获取前一页和当前页的布局框
        pre_page_layout_bbox = new_layout_of_pages[page_num-1]
        next_page_layout_bbox = new_layout_of_pages[page_num]
        
        # 检查前一页和当前页的段落是否可以连接
        is_conn = __connect_para_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, page_num, lang)
        # 如果调试模式开启，记录连接信息
        if debug_mode:
            if is_conn:
                logger.info(f"连接了第{page_num-1}页和第{page_num}页的段落")
            
        # 检查前一页和当前页的列表段落是否可以连接
        is_list_conn = __connect_list_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, all_page_list_info[page_num-1], all_page_list_info[page_num], page_num, lang)
        # 如果调试模式开启，记录连接信息
        if debug_mode:
            if is_list_conn:
                logger.info(f"连接了第{page_num-1}页和第{page_num}页的列表段落")
            
    """接下来可能会漏掉一些特别的一些可以合并的内容，对他们进行段落连接
    1. 正文中有时出现一个行顶格，接下来几行缩进的情况。
    2. 居中的一些连续单行，如果高度相同，那么可能是一个段落。
    """
    # 遍历 pdf_info_dict 字典中的所有页面信息，获取页面编号和内容
        for page_num, page in enumerate(pdf_info_dict.values()):
            # 从当前页面信息中提取段落块
            page_paras = page['para_blocks']
            # 获取当前页面的新布局边界框
            new_layout_bbox = new_layout_of_pages[page_num]
            # 调用函数连接中间对齐的文本，传入段落块、布局边界框、页面编号、语言和调试模式
            __connect_middle_align_text(page_paras, new_layout_bbox, page_num, lang, debug_mode=debug_mode)
            # 调用函数合并单一列表中的文本，传入段落块、布局边界框、页面编号和语言
            __merge_signle_list_text(page_paras, new_layout_bbox, page_num, lang)
```