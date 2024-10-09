# `.\MinerU\magic_pdf\post_proc\pdf_post_filter.py`

```
# 从 loguru 导入 logger，用于记录日志
from loguru import logger

# 从 magic_pdf.layout.layout_sort 导入获取布局列数的函数
from magic_pdf.layout.layout_sort import get_columns_cnt_of_layout
# 从 magic_pdf.libs 导入 DropReason 类，用于定义丢弃原因
from magic_pdf.libs.drop_reason import DropReason


# 定义一个函数，用于判断页面是否伪单列，返回布尔值
def __is_pseudo_single_column(page_info) -> bool:
    """
    判断一个页面是否伪单列。

    Args:
        page_info (dict): 页面信息字典，包括'_layout_tree'和'preproc_blocks'。

    Returns:
        Tuple[bool, Optional[str]]: 如果页面伪单列返回(True, extra_info)，否则返回(False, None)。

    """
    # 从页面信息中提取布局树
    layout_tree = page_info['_layout_tree']
    # 获取布局列数
    layout_column_width = get_columns_cnt_of_layout(layout_tree)
    # 如果布局列数为1，则继续判断
    if layout_column_width == 1:
        # 提取预处理文本块
        text_blocks = page_info['preproc_blocks']
        # 遍历每一个text_block
        for text_block in text_blocks:
            # 提取文本块中的行
            lines = text_block['lines']
            # 统计行数
            num_lines = len(lines)
            # 初始化满足条件的行数计数
            num_satisfying_lines = 0

            # 遍历行，判断相邻行的边界框
            for i in range(num_lines - 1):
                current_line = lines[i]
                next_line = lines[i + 1]

                # 获取当前行和下一行的边界框属性
                current_bbox = current_line['bbox']
                next_bbox = next_line['bbox']

                # 检查相邻行是否满足条件
                if next_bbox[0] > current_bbox[2] or next_bbox[2] < current_bbox[0]:
                    num_satisfying_lines += 1
            # 如果有一半以上的行满足条件，则标记为需要丢弃
            # print("num_satisfying_lines:", num_satisfying_lines, "num_lines:", num_lines)
            if num_lines > 20:
                # 计算满足条件的行数比例
                radio = num_satisfying_lines / num_lines
                # 如果比例大于等于0.5，则记录额外信息并返回
                if radio >= 0.5:
                    extra_info = f"{{num_lines: {num_lines}, num_satisfying_lines: {num_satisfying_lines}}}"
                    block_text = []
                    # 提取文本块中的所有文本
                    for line in lines:
                        if line['spans']:
                            for span in line['spans']:
                                block_text.append(span['text'])
                    # 记录警告日志
                    logger.warning(f"pseudo_single_column block_text: {block_text}")
                    return True, extra_info

    # 如果不满足伪单列条件，返回False和None
    return False, None


# 定义一个函数，用于后处理 PDF 页面，判断其是否符合要求
def pdf_post_filter(page_info) -> tuple:
    """
    return:(True|False, err_msg)
        True, 如果pdf符合要求
        False, 如果pdf不符合要求

    """
    # 调用 __is_pseudo_single_column 函数检查页面
    bool_is_pseudo_single_column, extra_info = __is_pseudo_single_column(page_info)
    # 如果页面被标记为伪单列，则返回丢弃信息
    if bool_is_pseudo_single_column:
        return False, {"_need_drop": True, "_drop_reason": DropReason.PSEUDO_SINGLE_COLUMN, "extra_info": extra_info}

    # 如果符合要求，返回True和None
    return True, None
```