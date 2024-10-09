# `.\MinerU\magic_pdf\pre_proc\resolve_bbox_conflict.py`

```
"""
# 从 PDF 中提取 API 给出的 bbox，并根据重叠情况进行处理
# 1. 首先去掉出现在图片上的 bbox，包括表格和图片
# 2. 然后去掉出现在文字 block 上的图片 bbox
"""

# 从指定库导入判断 bbox 重叠和位置的工具函数
from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap, _is_left_overlap
from magic_pdf.libs.drop_tag import ON_IMAGE_TEXT, ON_TABLE_TEXT


def resolve_bbox_overlap_conflict(images: list, tables: list, interline_equations: list, inline_equations: list,
                                  text_raw_blocks: list):
    """
    # text_raw_blocks 结构是从 pymupdf 直接获取的，具体样例参考提供的 JSON 文件
    # 采用一种粗暴的方式处理重叠情况：
    # 1. 去掉图片上的公式
    # 2. 去掉表格上的公式
    # 3. 图片和文字 block 重叠时丢弃图片
    # 4. 去掉文字 bbox 中完全位于图片、表格内部的文字
    # 5. 去掉表格上的文字
    """
    text_block_removed = []  # 存储已去掉的文字 block
    images_backup = []  # 存储与文字重叠的图片

    # 去掉位于图片上的文字 block
    for image_box in images:
        for text_block in text_raw_blocks:
            text_bbox = text_block["bbox"]  # 获取文字 block 的 bbox
            # 检查文字 block 是否完全在图片 bbox 内
            if _is_in(text_bbox, image_box):
                text_block['tag'] = ON_IMAGE_TEXT  # 标记为在图片上的文字
                text_block_removed.append(text_block)  # 添加到已去掉列表

    # 去掉位于表格上的文字 block
    for table_box in tables:
        for text_block in text_raw_blocks:
            text_bbox = text_block["bbox"]  # 获取文字 block 的 bbox
            # 检查文字 block 是否完全在表格 bbox 内
            if _is_in(text_bbox, table_box):
                text_block['tag'] = ON_TABLE_TEXT  # 标记为在表格上的文字
                text_block_removed.append(text_block)  # 添加到已去掉列表

    # 从原始文字 blocks 中移除已标记的文字 block
    for text_block in text_block_removed:
        if text_block in text_raw_blocks:
            text_raw_blocks.remove(text_block)

    # 第一步：去掉在图片上出现的公式 box
    temp = []  # 临时存储要移除的公式
    for image_box in images:
        for eq1 in interline_equations:
            # 检查公式是否与图片 bbox 有重叠
            if _is_in_or_part_overlap(image_box, eq1[:4]):
                temp.append(eq1)  # 添加到临时列表
        for eq2 in inline_equations:
            # 检查公式是否与图片 bbox 有重叠
            if _is_in_or_part_overlap(image_box, eq2[:4]):
                temp.append(eq2)  # 添加到临时列表

    # 从 interline 和 inline 公式中移除已标记的公式
    for eq in temp:
        if eq in interline_equations:
            interline_equations.remove(eq)  # 移除 interline 公式
        if eq in inline_equations:
            inline_equations.remove(eq)  # 移除 inline 公式

    # 第二步：去掉在表格上出现的公式 box
    temp = []  # 临时存储要移除的公式
    for table_box in tables:
        for eq1 in interline_equations:
            # 检查公式是否与表格 bbox 有重叠
            if _is_in_or_part_overlap(table_box, eq1[:4]):
                temp.append(eq1)  # 添加到临时列表
        for eq2 in inline_equations:
            # 检查公式是否与表格 bbox 有重叠
            if _is_in_or_part_overlap(table_box, eq2[:4]):
                temp.append(eq2)  # 添加到临时列表

    # 从 interline 和 inline 公式中移除已标记的公式
    for eq in temp:
        if eq in interline_equations:
            interline_equations.remove(eq)  # 移除 interline 公式
        if eq in inline_equations:
            inline_equations.remove(eq)  # 移除 inline 公式

    # 图片和文字重叠时，丢掉重叠的图片
    for image_box in images:
        for text_block in text_raw_blocks:
            text_bbox = text_block["bbox"]  # 获取文字 block 的 bbox
            # 检查图片与文字 block 是否重叠
            if _is_in_or_part_overlap(image_box, text_bbox):
                images_backup.append(image_box)  # 添加到备份列表
                break  # 找到重叠后跳出循环
    # 从原始图片列表中移除重叠的图片
    for image_box in images_backup:
        images.remove(image_box)

    # 图片和图片重叠时，暂时不参与版面计算
    images_dup_index = []  # 存储重叠图片的索引
    # 遍历所有图像，使用双重循环比较每一对图像
    for i in range(len(images)):
        # 从当前图像的下一个开始，避免重复比较
        for j in range(i + 1, len(images)):
            # 检查这两个图像是否重叠或部分重叠
            if _is_in_or_part_overlap(images[i], images[j]):
                # 将重叠的图像索引添加到重复索引列表
                images_dup_index.append(i)
                images_dup_index.append(j)

    # 将重复索引转换为集合，以去除重复项
    dup_idx = set(images_dup_index)
    # 遍历唯一的重复图像索引
    for img_id in dup_idx:
        # 将重复图像备份到备份列表中
        images_backup.append(images[img_id])
        # 将原图像设置为 None，标记为删除
        images[img_id] = None

    # 过滤掉所有为 None 的图像，保留有效图像
    images = [img for img in images if img is not None]

    # 如果行间公式与文字块重叠，将其放入临时数据中，以避免影响布局计算
    # 通过计算 IOU 合并行间公式和文字块
    # 删除这样的文本块，保持行间公式的大小不变
    # 在布局计算完毕后再合并回来
    text_block_removed_2 = []
    # 遍历原始文本块，判断重叠情况
    # for text_block in text_raw_blocks:
    #     text_bbox = text_block["bbox"]
    #     # 对每个行间公式进行比较
    #     for eq in interline_equations:
    #         # 计算重叠面积与最小外接框面积的比率
    #         ratio = calculate_overlap_area_2_minbox_area_ratio(text_bbox, eq[:4])
    #         # 如果重叠比率超过阈值，则标记并保存
    #         if ratio > 0.05:
    #             text_block['tag'] = "belong-to-interline-equation"
    #             text_block_removed_2.append(text_block)
    #             break

    # 移除被标记的文本块
    # for tb in text_block_removed_2:
    #     if tb in text_raw_blocks:
    #         text_raw_blocks.remove(tb)

    # 合并被移除的文本块
    # text_block_removed = text_block_removed + text_block_removed_2

    # 返回图像、表格、行间公式、内联公式、原始文本块、被移除的文本块、图像备份和被移除的文本块列表
    return images, tables, interline_equations, inline_equations, text_raw_blocks, text_block_removed, images_backup, text_block_removed_2
# 检查文本block之间的水平重叠情况，如果发现重叠，返回 True
def check_text_block_horizontal_overlap(text_blocks: list, header, footer) -> bool:
    """
    检查文本block之间的水平重叠情况，这种情况如果发生，那么这个pdf就不再继续处理了。
    因为这种情况大概率发生了公式没有被检测出来。
    
    """
    # 如果文本块列表为空，返回 False
    if len(text_blocks) == 0:
        return False

    # 初始化页面最小Y坐标
    page_min_y = 0
    # 计算页面最大Y坐标
    page_max_y = max(yy['bbox'][3] for yy in text_blocks)

    # 定义一个获取最大Y坐标的函数
    def __max_y(lst: list):
        # 如果列表不为空，返回列表中第二个元素的最大值
        if len(lst) > 0:
            return max([item[1] for item in lst])
        return page_min_y

    # 定义一个获取最小Y坐标的函数
    def __min_y(lst: list):
        # 如果列表不为空，返回列表中第四个元素的最小值
        if len(lst) > 0:
            return min([item[3] for item in lst])
        return page_max_y

    # 获取头部的最大Y坐标
    clip_y0 = __max_y(header)
    # 获取底部的最小Y坐标
    clip_y1 = __min_y(footer)

    # 初始化文本框的边界框列表
    txt_bboxes = []
    # 遍历每个文本块
    for text_block in text_blocks:
        bbox = text_block["bbox"]
        # 如果文本块在头部和底部之间，添加其边界框
        if bbox[1] >= clip_y0 and bbox[3] <= clip_y1:
            txt_bboxes.append(bbox)

    # 检查文本框之间的重叠情况
    for i in range(len(txt_bboxes)):
        for j in range(i + 1, len(txt_bboxes)):
            # 如果两个文本框有重叠，返回 True
            if _is_left_overlap(txt_bboxes[i], txt_bboxes[j]) or _is_left_overlap(txt_bboxes[j], txt_bboxes[i]):
                return True

    # 如果没有重叠，返回 False
    return False


# 检查有用的文本block之间的水平重叠情况
def check_useful_block_horizontal_overlap(useful_blocks: list) -> bool:
    """
    检查文本block之间的水平重叠情况，这种情况如果发生，那么这个pdf就不再继续处理了。
    因为这种情况大概率发生了公式没有被检测出来。

    """
    # 如果有用的文本块列表为空，返回 False
    if len(useful_blocks) == 0:
        return False

    # 初始化页面最小Y坐标
    page_min_y = 0
    # 计算页面最大Y坐标
    page_max_y = max(yy['bbox'][3] for yy in useful_blocks)

    # 初始化有用文本框的边界框列表
    useful_bboxes = []
    # 遍历每个有用的文本块
    for text_block in useful_blocks:
        bbox = text_block["bbox"]
        # 如果文本块在页面范围内，添加其边界框
        if bbox[1] >= page_min_y and bbox[3] <= page_max_y:
            useful_bboxes.append(bbox)

    # 检查有用文本框之间的重叠情况
    for i in range(len(useful_bboxes)):
        for j in range(i + 1, len(useful_bboxes)):
            # 计算两个文本框的面积
            area_i = (useful_bboxes[i][2] - useful_bboxes[i][0]) * (useful_bboxes[i][3] - useful_bboxes[i][1])
            area_j = (useful_bboxes[j][2] - useful_bboxes[j][0]) * (useful_bboxes[j][3] - useful_bboxes[j][1])
            # 如果两个文本框有重叠，返回相关信息
            if _is_left_overlap(useful_bboxes[i], useful_bboxes[j]) or _is_left_overlap(useful_bboxes[j], useful_bboxes[i]):
                if area_i > area_j:
                    return True, useful_bboxes[j], useful_bboxes[i]
                else:
                    return True, useful_bboxes[i], useful_bboxes[j]

    # 如果没有重叠，返回 False 和 None
    return False, None, None
```