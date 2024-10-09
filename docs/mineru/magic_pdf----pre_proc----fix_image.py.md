# `.\MinerU\magic_pdf\pre_proc\fix_image.py`

```
# 导入正则表达式模块
import re    
# 从 magic_pdf.libs.boxbase 导入多个函数
from magic_pdf.libs.boxbase import  _is_in_or_part_overlap, _is_part_overlap, find_bottom_nearest_text_bbox, find_left_nearest_text_bbox, find_right_nearest_text_bbox, find_top_nearest_text_bbox

# 从 magic_pdf.libs.textbase 导入获取文本块基础信息的函数
from magic_pdf.libs.textbase import get_text_block_base_info

# 修正图片位置的函数，接受图片边界和文本块作为参数
def fix_image_vertical(image_bboxes:list, text_blocks:list):
    """
    修正图片的位置
    如果图片与文字block发生一定重叠（也就是图片切到了一部分文字），那么减少图片边缘，让文字和图片不再重叠。
    只对垂直方向进行。
    """
    # 遍历所有图片边界
    for image_bbox in image_bboxes:
        # 遍历所有文本块
        for text_block in text_blocks:
            text_bbox = text_block["bbox"]  # 获取文本块的边界
            # 检查文本块和图片边界是否部分重叠，并且是否在水平方向上有重叠
            if _is_part_overlap(text_bbox, image_bbox) and any([text_bbox[0]>=image_bbox[0] and text_bbox[2]<=image_bbox[2], text_bbox[0]<=image_bbox[0] and text_bbox[2]>=image_bbox[2]]):
                # 如果文本块在图片上方，调整图片底部
                if text_bbox[1] < image_bbox[1]:#在图片上方
                    image_bbox[1] = text_bbox[3]+1  # 将图片顶部调整到文本块底部加1
                # 如果文本块在图片下方，调整图片顶部
                elif text_bbox[3]>image_bbox[3]:#在图片下方
                    image_bbox[3] = text_bbox[1]-1  # 将图片底部调整到文本块顶部减1
                
    return image_bboxes  # 返回调整后的图片边界

# 合并有共同边的两个边界的辅助函数
def __merge_if_common_edge(bbox1, bbox2):
    x_min_1, y_min_1, x_max_1, y_max_1 = bbox1  # 解包第一个边界的坐标
    x_min_2, y_min_2, x_max_2, y_max_2 = bbox2  # 解包第二个边界的坐标

    # 检查是否有公共的水平边
    if y_min_1 == y_min_2 or y_max_1 == y_max_2:
        # 确保一个框的x范围在另一个框的x范围内
        if max(x_min_1, x_min_2) <= min(x_max_1, x_max_2):
            return [min(x_min_1, x_min_2), min(y_min_1, y_min_2), max(x_max_1, x_max_2), max(y_max_1, y_max_2)]  # 返回合并后的边界

    # 检查是否有公共的垂直边
    if x_min_1 == x_min_2 or x_max_1 == x_max_2:
        # 确保一个框的y范围在另一个框的y范围内
        if max(y_min_1, y_min_2) <= min(y_max_1, y_max_2):
            return [min(x_min_1, x_min_2), min(y_min_1, y_min_2), max(x_max_1, x_max_2), max(y_max_1, y_max_2)]  # 返回合并后的边界

    # 如果没有公共边，则返回 None
    return None

# 修正分离的图片的函数
def fix_seperated_image(image_bboxes:list):
    """
    如果2个图片有一个边重叠，那么合并2个图片
    """
    new_images = []  # 用于存储合并后的新图片边界
    droped_img_idx = []  # 用于记录被合并的图片索引
            
    # 遍历所有图片边界
    for i in range(0, len(image_bboxes)):
        # 比较每对图片边界
        for j in range(i+1, len(image_bboxes)):
            new_img = __merge_if_common_edge(image_bboxes[i], image_bboxes[j])  # 尝试合并两个边界
            if new_img is not None:  # 如果合并成功
                new_images.append(new_img)  # 添加新边界到列表
                droped_img_idx.append(i)  # 记录被合并的索引
                droped_img_idx.append(j)  # 记录被合并的索引
                break  # 结束内层循环
            
    # 遍历所有图片边界
    for i in range(0, len(image_bboxes)):
        if i not in droped_img_idx:  # 如果该索引未被合并
            new_images.append(image_bboxes[i])  # 添加到新列表
            
    return new_images  # 返回新图片边界列表


# 检查文本段是否是表格标题的辅助函数
def __check_img_title_pattern(text):
    """
    检查文本段是否是表格的标题
    """
    patterns = [r"^(fig|figure).*", r"^(scheme).*"]  # 定义标题模式
    text = text.strip()  # 去除文本两端空白
    # 遍历所有模式
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)  # 尝试匹配模式
        if match:  # 如果匹配成功
            return True  # 返回 True
    return False  # 返回 False


# 获取图片说明文本的辅助函数
def __get_fig_caption_text(text_block):
    # 组合文本块中的所有文本为一个字符串
    txt = " ".join(span['text'] for line in text_block['lines'] for span in line['spans'])
    line_cnt = len(text_block['lines'])  # 计算行数
    txt = txt.replace("Ž . ", '')  # 替换特定字符串
    return txt, line_cnt  # 返回文本和行数


# 查找并扩展底部标题的辅助函数
def __find_and_extend_bottom_caption(text_block, pymu_blocks, image_box):
    #
    # 继续向下方寻找和图片 caption 字号、字体、颜色一样的文字框，合并入 caption。
    # text_block 是已经找到的图片 caption（这个 caption 可能不全，多行被划分到多个 pymu block 里了）
    combined_image_caption_text_block = list(text_block.copy()['bbox'])  # 创建一个新的列表，存储图片 caption 的边界框信息
    base_font_color, base_font_size, base_font_type = get_text_block_base_info(text_block)  # 获取当前 text_block 的基本字体信息
    while True:  # 开始无限循环，直到找到合适的 text block 或者结束条件满足
        tb_add = find_bottom_nearest_text_bbox(pymu_blocks, combined_image_caption_text_block)  # 寻找与当前 caption 边界框最近的下方文本框
        if not tb_add:  # 如果没有找到合适的文本框，则退出循环
            break
        tb_font_color, tb_font_size, tb_font_type = get_text_block_base_info(tb_add)  # 获取找到的文本框的基本字体信息
        if tb_font_color==base_font_color and tb_font_size==base_font_size and tb_font_type==base_font_type:  # 判断找到的文本框的字体信息是否与当前 caption 相同
            combined_image_caption_text_block[0] = min(combined_image_caption_text_block[0], tb_add['bbox'][0])  # 更新左边界为最小值
            combined_image_caption_text_block[2] = max(combined_image_caption_text_block[2], tb_add['bbox'][2])  # 更新右边界为最大值
            combined_image_caption_text_block[3] = tb_add['bbox'][3]  # 更新下边界为找到文本框的下边界
        else:  # 如果字体信息不同，则退出循环
            break
            
    image_box[0] = min(image_box[0], combined_image_caption_text_block[0])  # 更新 image_box 的左边界为最小值
    image_box[1] = min(image_box[1], combined_image_caption_text_block[1])  # 更新 image_box 的上边界为最小值
    image_box[2] = max(image_box[2], combined_image_caption_text_block[2])  # 更新 image_box 的右边界为最大值
    image_box[3] = max(image_box[3], combined_image_caption_text_block[3])  # 更新 image_box 的下边界为最大值
    text_block['_image_caption'] = True  # 标记该 text_block 为图片 caption
# 定义一个函数，用于合并与图片相关的文本块和边框
def include_img_title(pymu_blocks, image_bboxes: list):
    """
    向上方和下方寻找符合图片title的文本block，合并到图片里
    如果图片上下都有fig的情况怎么办？寻找标题距离最近的那个。
    ---
    增加对左侧和右侧图片标题的寻找
    """
    
    # 返回传入的图片边框列表，未做任何处理
    return image_bboxes


# 定义一个函数，用于合并重叠的图片边框
def combine_images(image_bboxes:list):
    """
    合并图片，如果图片有重叠，那么合并
    """
    # 初始化一个新的图片列表，用于存储合并后的图片
    new_images = []
    # 初始化一个列表，用于记录被合并掉的图片索引
    droped_img_idx = []
            
    # 外层循环，遍历每个图片边框
    for i in range(0, len(image_bboxes)):
        # 内层循环，检查与当前图片边框的重叠情况
        for j in range(i+1, len(image_bboxes)):
            # 如果当前图片未被合并且存在重叠
            if j not in droped_img_idx and _is_in_or_part_overlap(image_bboxes[i], image_bboxes[j]):
                # 合并当前图片边框的坐标到最小和最大值
                image_bboxes[i][0], image_bboxes[i][1],image_bboxes[i][2],image_bboxes[i][3] = min(image_bboxes[i][0], image_bboxes[j][0]), min(image_bboxes[i][1], image_bboxes[j][1]), max(image_bboxes[i][2], image_bboxes[j][2]), max(image_bboxes[i][3], image_bboxes[j][3])
                # 记录被合并掉的图片索引
                droped_img_idx.append(j)
            
    # 遍历所有图片边框，保留未被合并的图片
    for i in range(0, len(image_bboxes)):
        if i not in droped_img_idx:
            # 将未被合并的图片添加到新列表中
            new_images.append(image_bboxes[i])
            
    # 返回合并后的图片列表
    return new_images
```