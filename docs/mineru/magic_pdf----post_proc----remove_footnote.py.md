# `.\MinerU\magic_pdf\post_proc\remove_footnote.py`

```
# 从 magic_pdf.libs.boxbase 导入函数
from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap
# 导入 collections 库以进行统计操作
import collections      # 统计库



# 判断 bbox1 是否在 bbox2 下面
def is_below(bbox1, bbox2):
    # 如果 bbox1 的上边 y 坐标大于 bbox2 的下边 y 坐标，返回 True
    return bbox1[1] > bbox2[3]


# 合并多个 bbox，返回一个新的 bbox
def merge_bboxes(bboxes):
    # 找出所有 bboxes 的最小 x0，最大 y1，最大 x1，最小 y0，这就是合并后的 bbox
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[2] for bbox in bboxes)
    y1 = max(bbox[3] for bbox in bboxes)
    # 返回合并后的 bbox
    return [x0, y0, x1, y1]


# 合并脚注块，初始化 merged_bboxes 列表
def merge_footnote_blocks(page_info, main_text_font):
    # 在 page_info 中添加一个空列表以存储合并后的 bboxes
    page_info['merged_bboxes'] = []
    # 返回更新后的 page_info
    return page_info


# 从页面信息中移除脚注块
def remove_footnote_blocks(page_info):
    # 如果 merged_bboxes 存在
    if page_info.get('merged_bboxes'):
        # 从文字中去掉脚注
        remain_text_blocks, removed_footnote_text_blocks = remove_footnote_text(page_info['preproc_blocks'], page_info['merged_bboxes'])
        # 从图片中去掉脚注
        image_blocks, removed_footnote_imgs_blocks = remove_footnote_image(page_info['images'], page_info['merged_bboxes'])
        # 更新 page_info，保留剩余的文本和图像块
        page_info['preproc_blocks'] = remain_text_blocks
        page_info['images'] = image_blocks
        # 将移除的文本和图像块添加到丢弃列表中
        page_info['droped_text_block'].extend(removed_footnote_text_blocks)
        page_info['droped_image_block'].extend(removed_footnote_imgs_blocks)
        # 删除临时的 merged_bboxes
        del page_info['merged_bboxes']
    # 删除临时的脚注 bboxes
    del page_info['footnote_bboxes_tmp']
    # 返回更新后的 page_info
    return page_info


# 移除原始文本块中的脚注文本
def remove_footnote_text(raw_text_block, footnote_bboxes):
    """
    :param raw_text_block: str类型，是当前页的文本内容
    :param footnoteBboxes: list类型，是当前页的脚注 bbox
    """
    # 初始化脚注文本块列表
    footnote_text_blocks = []
    # 遍历原始文本块
    for block in raw_text_block:
        text_bbox = block['bbox']
        # 检查文本块是否与任何脚注 bbox 重叠
        if any([_is_in_or_part_overlap(text_bbox, footnote_bbox) for footnote_bbox in footnote_bboxes]):
            # 标记为脚注
            block['tag'] = 'footnote'
            # 将该块添加到脚注文本块列表
            footnote_text_blocks.append(block)
            # raw_text_block.remove(block)  # 这个行已被注释掉以避免错误

    # 移除脚注文本块，不能在循环内直接移除
    for block in footnote_text_blocks:
        raw_text_block.remove(block)

    # 返回更新后的原始文本块和脚注文本块列表
    return raw_text_block, footnote_text_blocks


# 移除图像块中的脚注图像
def remove_footnote_image(image_blocks, footnote_bboxes):
    """
    :param image_bboxes: list类型，是当前页的图片 bbox(结构体)
    :param footnoteBboxes: list类型，是当前页的脚注 bbox
    """
    # 初始化脚注图像块列表
    footnote_imgs_blocks = []
    # 遍历图像块
    for image_block in image_blocks:
        # 检查图像块是否在任何脚注 bbox 内
        if any([_is_in(image_block['bbox'], footnote_bbox) for footnote_bbox in footnote_bboxes]):
            # 将符合条件的图像块添加到脚注图像块列表
            footnote_imgs_blocks.append(image_block)

    # 移除脚注图像块
    for footnote_imgs_block in footnote_imgs_blocks:
        image_blocks.remove(footnote_imgs_block)

    # 返回更新后的图像块和脚注图像块列表
    return image_blocks, footnote_imgs_blocks
```