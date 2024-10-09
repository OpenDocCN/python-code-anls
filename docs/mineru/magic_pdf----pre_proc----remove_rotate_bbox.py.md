# `.\MinerU\magic_pdf\pre_proc\remove_rotate_bbox.py`

```
# 导入数学模块以进行数学计算
import math

# 从库中导入检测盒子和标记相关的功能
from magic_pdf.libs.boxbase import is_vbox_on_side
from magic_pdf.libs.drop_tag import EMPTY_SIDE_BLOCK, ROTATE_TEXT, VERTICAL_TEXT


# 定义函数以检测文档中的非水平文本
def detect_non_horizontal_texts(result_dict):
    """
    该函数用于检测文档中的水印和垂直边注。

    通过找到具有相同坐标和在多个页面上频繁出现的相同文本来识别水印。
    如果满足这些条件，则这些块很可能是水印，而不是页眉或页脚，它们可能会在页面之间变化。
    如果这些块的方向不是水平的，则肯定被视为水印。

    通过找到具有相同坐标和在多个页面上频繁出现的相同文本来识别垂直边注。
    如果满足这些条件，则这些块很可能是垂直边注，通常出现在页面的左右侧。
    如果这些块的方向是垂直的，则肯定被视为垂直边注。

    参数
    ----------
    result_dict : dict
        结果字典。

    返回
    -------
    result_dict : dict
        更新后的结果字典。
    """
    # 创建字典以存储潜在水印的信息
    potential_watermarks = {}
    # 创建字典以存储潜在边注的信息
    potential_margin_notes = {}

    # 遍历结果字典中的每一页
    for page_id, page_content in result_dict.items():
        # 仅处理以 "page_" 开头的页面
        if page_id.startswith("page_"):
            # 遍历页面内容中的每个块
            for block_id, block_data in page_content.items():
                # 仅处理以 "block_" 开头的块
                if block_id.startswith("block_"):
                    # 检查块数据中是否包含方向信息
                    if "dir" in block_data:
                        # 创建坐标和文本的元组
                        coordinates_text = (block_data["bbox"], block_data["text"])  # 坐标和文本的元组

                        # 计算块的方向角度
                        angle = math.atan2(block_data["dir"][1], block_data["dir"][0])
                        angle = abs(math.degrees(angle))  # 转换为绝对角度

                        # 检查角度以判断是否为水印方向
                        if angle > 5 and angle < 85:  # 检查方向是否为水印
                            # 如果坐标文本已存在于潜在水印中，则计数增加
                            if coordinates_text in potential_watermarks:
                                potential_watermarks[coordinates_text] += 1
                            else:
                                # 否则初始化计数
                                potential_watermarks[coordinates_text] = 1

                        # 检查角度以判断是否为垂直方向
                        if angle > 85 and angle < 105:  # 检查方向是否为垂直
                            # 如果坐标文本已存在于潜在边注中，则计数增加
                            if coordinates_text in potential_margin_notes:
                                potential_margin_notes[coordinates_text] += 1  # 增加计数
                            else:
                                # 否则初始化计数
                                potential_margin_notes[coordinates_text] = 1  # 初始化计数

    # 通过查找计数高于阈值的条目来识别水印（例如，出现次数超过一半页面）
    watermark_threshold = len(result_dict) // 2
    # 创建水印字典，过滤出计数超过阈值的潜在水印
    watermarks = {k: v for k, v in potential_watermarks.items() if v > watermark_threshold}
    # 通过找到出现次数超过阈值的条目来识别边注（例如，出现在超过一半页面上的条目）
    margin_note_threshold = len(result_dict) // 2
    # 创建一个字典，包含出现次数超过阈值的潜在边注
    margin_notes = {k: v for k, v in potential_margin_notes.items() if v > margin_note_threshold}

    # 将水印信息添加到结果字典中
    for page_id, blocks in result_dict.items():
        # 检查页面 ID 是否以 "page_" 开头
        if page_id.startswith("page_"):
            # 遍历每个页面的块
            for block_id, block_data in blocks.items():
                # 获取块的边界框和文本
                coordinates_text = (block_data["bbox"], block_data["text"])
                # 如果块的坐标文本在水印列表中，则标记为水印
                if coordinates_text in watermarks:
                    block_data["is_watermark"] = 1
                else:
                    # 否则标记为非水印
                    block_data["is_watermark"] = 0

                # 如果块的坐标文本在边注列表中，则标记为垂直边注
                if coordinates_text in margin_notes:
                    block_data["is_vertical_margin_note"] = 1
                else:
                    # 否则标记为非边注
                    block_data["is_vertical_margin_note"] = 0

    # 返回更新后的结果字典
    return result_dict
"""
# 注释: 用于描述如何处理特定条件下的文本块
1. 当一个block里全部文字都不是dir=(1,0)，这个block整体去掉
2. 当一个block里全部文字都是dir=(1,0)，但是每行只有一个字，这个block整体去掉。这个block必须出现在页面的四周，否则不去掉
"""
# 导入正则表达式模块
import re

def __is_a_word(sentence):
    # 检查输入的句子是否为单个中文字符
    if re.fullmatch(r'[\u4e00-\u9fa5]', sentence):
        return True
    # 检查输入是否为单个英文单词或字符（包括ASCII标点）
    elif re.fullmatch(r'[a-zA-Z0-9]+', sentence) and len(sentence) <=2:
        return True
    # 如果不符合条件，返回False
    else:
        return False


def __get_text_color(num):
    """获取字体的颜色RGB值"""
    # 提取蓝色分量
    blue = num & 255
    # 提取绿色分量
    green = (num >> 8) & 255
    # 提取红色分量
    red = (num >> 16) & 255
    # 返回RGB颜色值
    return red, green, blue


def __is_empty_side_box(text_block):
    """
    检查该文本块是否是边缘的空白块
    """
    # 遍历文本块中的每一行
    for line in text_block['lines']:
        # 遍历行中的每个字体范围
        for span in line['spans']:
            # 获取字体颜色
            font_color = span['color']
            r, g, b = __get_text_color(font_color)
            # 如果有非空文本且颜色不是白色，返回False
            if len(span['text'].strip()) > 0 and (r, g, b) != (255, 255, 255):
                return False
            
    # 如果所有条件都满足，返回True
    return True


def remove_rotate_side_textblock(pymu_text_block, page_width, page_height):
    """
    返回删除了垂直，水印，旋转的textblock
    删除的内容打上tag返回
    """
    # 用于存放被删除的文本块
    removed_text_block = []
    
    # 遍历每一个文本块
    for i, block in enumerate(pymu_text_block): # 格式参考test/assets/papre/pymu_textblocks.json
        lines = block['lines']  # 获取文本块中的行
        block_bbox = block['bbox']  # 获取文本块的边界框
        # 检查文本块是否在页面的两侧
        if not is_vbox_on_side(block_bbox, page_width, page_height, 0.2): # 保证这些box必须在页面的两边
           continue
        
        # 检查所有行是否都是单个单词，并且行数大于1
        if all([__is_a_word(line['spans'][0]["text"]) for line in lines if len(line['spans']) > 0]) and len(lines) > 1 and all([len(line['spans']) == 1 for line in lines]):
            # 检查文本块是否在垂直方向上对齐
            is_box_valign = (len(set([int(line['spans'][0]['bbox'][0]) for line in lines if len(line['spans']) > 0])) == 1) and (len([int(line['spans'][0]['bbox'][0]) for line in lines if len(line['spans']) > 0]) > 1)  # 测试bbox在垂直方向是不是x0都相等，也就是在垂直方向排列.同时必须大于等于2个字
            
            # 如果满足条件，打上标签并添加到移除列表
            if is_box_valign:
                block['tag'] = VERTICAL_TEXT
                removed_text_block.append(block)
                continue
        
        # 遍历每一行，检查方向
        for line in lines:
            # 如果有任意一行的方向不是(1,0)，则标记并删除整个块
            if line['dir'] != (1, 0):
                block['tag'] = ROTATE_TEXT
                removed_text_block.append(block)  # 只要有一个line不是dir=(1,0)，就把整个block都删掉
                break
        
    # 从原始文本块中移除标记的块
    for block in removed_text_block:
        pymu_text_block.remove(block)
    
    # 返回处理后的文本块和移除的文本块
    return pymu_text_block, removed_text_block

def get_side_boundry(rotate_bbox, page_width, page_height):
    """
    根据rotate_bbox，返回页面的左右正文边界
    """
    left_x = 0  # 初始化左边界
    right_x = page_width  # 初始化右边界
    # 遍历每个旋转边界框
    for x in rotate_bbox:
        box = x['bbox']  # 获取边界框
        # 如果边界框在页面左半部分，更新左边界
        if box[2] < page_width / 2:
            left_x = max(left_x, box[2])
        else:
            # 更新右边界
            right_x = min(right_x, box[0])
            
    # 返回计算后的左右边界
    return left_x + 1, right_x - 1


def remove_side_blank_block(pymu_text_block, page_width, page_height):
    """
    删除页面两侧的空白block
    """
    # 用于存放被删除的文本块
    removed_text_block = []
    # 遍历 pymu_text_block 列表，获取索引和每个块
    for i, block in enumerate(pymu_text_block): # 格式参考test/assets/papre/pymu_textblocks.json
        # 获取当前块的边界框信息
        block_bbox = block['bbox']
        # 检查当前边界框是否在页面的两侧，容忍度为 0.2
        if not is_vbox_on_side(block_bbox, page_width, page_height, 0.2): # 保证这些box必须在页面的两边
           # 如果不在两侧，跳过当前块
           continue
            
        # 检查当前块是否为空侧块
        if __is_empty_side_box(block):
            # 将当前块标记为 EMPTY_SIDE_BLOCK
            block['tag'] = EMPTY_SIDE_BLOCK
            # 将当前块添加到 removed_text_block 列表
            removed_text_block.append(block)
            # 跳过当前块的后续处理
            continue
        
    # 遍历被移除的文本块列表
    for block in removed_text_block:
        # 从原始文本块列表中移除当前块
        pymu_text_block.remove(block)
    
    # 返回处理后的文本块列表和移除的文本块列表
    return pymu_text_block, removed_text_block
```