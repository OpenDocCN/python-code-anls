# `.\MinerU\magic_pdf\pre_proc\fix_table.py`

```
# 从 magic_pdf.libs.commons 导入 fitz，用于处理 PDF 文件
from magic_pdf.libs.commons import fitz             # pyMuPDF库
# 导入正则表达式模块
import re
# 从 magic_pdf.libs.boxbase 导入多个函数，用于处理表格和区域重叠
from magic_pdf.libs.boxbase import _is_in_or_part_overlap, _is_part_overlap, find_bottom_nearest_text_bbox, find_left_nearest_text_bbox, find_right_nearest_text_bbox, find_top_nearest_text_bbox             # json

## version 2
# 定义函数，提取并合并当前页的水平线段
def get_merged_line(page):
    """
    这个函数是为了从pymuPDF中提取出的矢量里筛出水平的横线，并且将断开的线段进行了合并。
    :param page :fitz读取的当前页的内容
    """
    drawings_bbox = []  # 用于存储每条线的边界框
    drawings_line = []  # 记录线段
    drawings = page.get_drawings()  # 提取所有的矢量
    for p in drawings:
        # 将每个图形的边界框添加到列表中
        drawings_bbox.append(p["rect"].irect)  # (L, U, R, D)

    lines = []  # 存储水平线段
    for L, U, R, D in drawings_bbox:
        # 筛选出高度差不超过3的水平线
        if abs(D - U) <= 3: # 筛出水平的横线
            lines.append((L, U, R, D))  # 添加符合条件的线段

    U_groups = []  # 存储高度一致的线段组
    visited = [False for _ in range(len(lines))]  # 记录线段是否已访问
    for i, (L1, U1, R1, D1) in enumerate(lines):
        # 跳过已访问的线段
        if visited[i] == True:
            continue
        tmp_g = [(L1, U1, R1, D1)]  # 当前组初始化
        for j, (L2, U2, R2, D2) in enumerate(lines):
            if i == j:
                continue  # 跳过自身
            if visited[j] == True:
                continue  # 跳过已访问的线段
            # 判断是否高度一致，若一致则加入当前组
            if max(U1, D1, U2, D2) - min(U1, D1, U2, D2) <= 5:   # 把高度一致的线放进一个group
                tmp_g.append((L2, U2, R2, D2))  # 添加到组中
                visited[j] = True  # 标记已访问
        U_groups.append(tmp_g)  # 将组添加到列表中
        
    res = []  # 存储最终结果
    for group in U_groups:
        # 根据左边界和右边界排序
        group.sort(key = lambda LURD: (LURD[0], LURD[2]))
        LL, UU, RR, DD = group[0]  # 初始化边界值
        for i, (L1, U1, R1, D1) in enumerate(group):
            # 判断当前线段与上一个线段的间距
            if (L1 - RR) >= 5:
                cur_line = (LL, UU, RR, DD)  # 形成当前线段
                res.append(cur_line)  # 添加到结果中
                LL = L1  # 更新左边界
            else:
                # 更新右边界为最大值
                RR = max(RR, R1)
        cur_line = (LL, UU, RR, DD)  # 形成最后一条线段
        res.append(cur_line)  # 添加到结果中
    return res  # 返回合并后的线段列表

# 定义函数以修正表格，包含标题选项
def fix_tables(page: fitz.Page, table_bboxes: list, include_table_title: bool, scan_line_num: int):
    """
    :param page :fitz读取的当前页的内容
    :param table_bboxes: list类型，每一个元素是一个元祖 (L, U, R, D)
    :param include_table_title: 是否将表格的标题也圈进来
    :param scan_line_num: 在与表格框临近的上下几个文本框里扫描搜索标题
    """
    
    drawings_lines = get_merged_line(page)  # 获取合并后的线段
    fix_table_bboxes = []  # 存储修正后的表格边界框
    # 遍历每个表格的边界框
        for table in table_bboxes:
            # 解包表格的左、上、右、下边界
            (L, U, R, D) = table
            # 初始化左边界修正列表
            fix_table_L = []
            # 初始化上边界修正列表
            fix_table_U = []
            # 初始化右边界修正列表
            fix_table_R = []
            # 初始化下边界修正列表
            fix_table_D = []
            # 计算表格的宽度
            width = R - L
            # 计算允许的宽度偏差范围（10%）
            width_range = width * 0.1 # 只看距离表格整体宽度10%之内偏差的线
            # 计算表格的高度
            height = D - U
            # 计算允许的高度偏差范围（10%）
            height_range = height * 0.1 # 只看距离表格整体高度10%之内偏差的线
            # 遍历绘图中的每一条线
            for line in drawings_lines:
                # 检查线的起点和终点是否在宽度允许范围内
                if (L - width_range) <= line[0] <= (L + width_range) and (R - width_range) <= line[2] <= (R + width_range): # 相近的宽度
                    # 检查线的Y坐标是否在上边界允许范围内
                    if (U - height_range) < line[1] < (U + height_range): # 上边界，在一定的高度范围内
                        # 添加该线的Y坐标到上边界修正列表
                        fix_table_U.append(line[1])
                        # 添加该线的起点X坐标到左边界修正列表
                        fix_table_L.append(line[0])
                        # 添加该线的终点X坐标到右边界修正列表
                        fix_table_R.append(line[2])
                    # 检查线的Y坐标是否在下边界允许范围内
                    elif (D - height_range) < line[1] < (D + height_range): # 下边界，在一定的高度范围内
                        # 添加该线的Y坐标到下边界修正列表
                        fix_table_D.append(line[1])
                        # 添加该线的起点X坐标到左边界修正列表
                        fix_table_L.append(line[0])
                        # 添加该线的终点X坐标到右边界修正列表
                        fix_table_R.append(line[2])
    
            # 如果存在上边界修正线
            if fix_table_U:
                # 更新上边界为修正后的最小值
                U = min(fix_table_U)
            # 如果存在下边界修正线
            if fix_table_D:
                # 更新下边界为修正后的最大值
                D = max(fix_table_D)
            # 如果存在左边界修正线
            if fix_table_L:
                # 更新左边界为修正后的最小值
                L = min(fix_table_L)
            # 如果存在右边界修正线
            if fix_table_R:
                # 更新右边界为修正后的最大值
                R = max(fix_table_R)
                
            # 检查是否需要包含表格标题
            if include_table_title:   # 需要将表格标题包括
                # 获取当前页面所有文本块，作为字典返回
                text_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]   # 所有的text的block
                # 筛选出与表格无重叠的文本块
                incolumn_text_blocks = [block for block in text_blocks if not ((block['bbox'][0] < L and block['bbox'][2] < L) or (block['bbox'][0] > R and block['bbox'][2] > R))]  # 将与表格完全没有任何遮挡的文字筛除掉（比如另一栏的文字）
                # 筛选出在表格上方的文本块
                upper_text_blocks = [block for block in incolumn_text_blocks if (U - block['bbox'][3]) > 0]  # 将在表格线以上的text block筛选出来
                # 按照文本块的下边界距离表格上边界的距离升序排序
                sorted_filtered_text_blocks = sorted(upper_text_blocks, key=lambda x: (U - x['bbox'][3], x['bbox'][0])) # 按照text block的下边界距离表格上边界的距离升序排序，如果是同一个高度，则先左再右
                
                # 遍历指定数量的扫描行
                for idx in range(scan_line_num):   
                    # 检查当前索引是否在排序后的文本块范围内
                    if idx+1 <= len(sorted_filtered_text_blocks):
                        # 获取当前文本块的行内容
                        line_temp = sorted_filtered_text_blocks[idx]['lines']
                        # 如果该文本块有内容
                        if line_temp:
                            # 提取出第一个span里的文本内容
                            text = line_temp[0]['spans'][0]['text'] # 提取出第一个span里的text内容
                            # 检查文本是否以"Table"开头（英文）
                            check_en = re.match('Table', text) # 检查是否有Table开头的(英文）
                            # 检查文本是否以"表"开头（中文）
                            check_ch = re.match('表', text) # 检查是否有Table开头的(中文）
                            # 如果是英文或中文的表格标题
                            if check_en or check_ch:
                                # 确保文本块的下边界在下边界之上
                                if sorted_filtered_text_blocks[idx]['bbox'][1] < D: # 以防出现负的bbox
                                    # 更新上边界为文本块的下边界
                                    U = sorted_filtered_text_blocks[idx]['bbox'][1]
                                      
            # 将修正后的表格边界添加到结果列表中
            fix_table_bboxes.append([L-2, U-2, R+2, D+2])
        
        # 返回所有修正后的表格边界框
        return fix_table_bboxes
# 检查文本段是否是表格的标题
def __check_table_title_pattern(text):
    # 定义用于匹配表格标题的正则表达式模式
    patterns = [r'^table\s\d+']
    
    # 遍历所有定义的模式
    for pattern in patterns:
        # 尝试与文本进行模式匹配，忽略大小写
        match = re.match(pattern, text, re.IGNORECASE)
        # 如果匹配成功，返回 True
        if match:
            return True
        # 否则，返回 False
        else:
            return False
         
         
# 调整表格的边界，以与相邻文本块对齐
def fix_table_text_block(pymu_blocks, table_bboxes: list):
    # 遍历所有表格边界框
    for tb in table_bboxes:
        # 解包边界框的左、上、右、下坐标
        (L, U, R, D) = tb
        # 遍历所有文本块
        for block in pymu_blocks:
            # 检查文本块与表格边界框是否重叠
            if _is_in_or_part_overlap((L, U, R, D), block['bbox']):
                # 将文本块中的所有文本合并成一个字符串
                txt = " ".join(span['text'] for line in block['lines'] for span in line['spans'])
                # 检查文本是否为表格标题，如果不是则调整边界
                if not __check_table_title_pattern(txt) and block.get("_table", False) is False:
                    # 更新表格边界框的左、上、右、下坐标
                    tb[0] = min(tb[0], block['bbox'][0])
                    tb[1] = min(tb[1], block['bbox'][1])
                    tb[2] = max(tb[2], block['bbox'][2])
                    tb[3] = max(tb[3], block['bbox'][3])
                    block['_table'] = True  # 标记该块为已占用，防止重复使用
                    
                # 如果文本块是表格标题并且部分重叠，调整标题边界
                if _is_part_overlap(tb, block['bbox']) and __check_table_title_pattern(txt):
                    block['bbox'] = list(block['bbox'])  # 转换为列表以便修改
                    # 如果标题的下边界超过表格上边界，则调整
                    if block['bbox'][3] > U:
                        block['bbox'][3] = U-1
                    # 如果标题的上边界低于表格下边界，则调整
                    if block['bbox'][1] < D:
                        block['bbox'][1] = D+1
                
    # 返回调整后的表格边界框
    return table_bboxes


# 获取文本块中的表格标题和行数
def __get_table_caption_text(text_block):
    # 将文本块中所有的文本合并成一个字符串
    txt = " ".join(span['text'] for line in text_block['lines'] for span in line['spans'])
    # 计算文本块的行数
    line_cnt = len(text_block['lines'])
    # 清除不需要的文本部分
    txt = txt.replace("Ž . ", '')
    # 返回标题文本和行数
    return txt, line_cnt


# 将表格标题包含到边界框中
def include_table_title(pymu_blocks, table_bboxes: list):
    # 返回包含表格标题的边界框
    return table_bboxes
```