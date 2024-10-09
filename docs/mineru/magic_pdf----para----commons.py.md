# `.\MinerU\magic_pdf\para\commons.py`

```
# 导入系统模块
import sys

# 从魔法 PDF 库导入 fitz 模块
from magic_pdf.libs.commons import fitz
# 导入支持彩色打印的 cprint 函数
from termcolor import cprint


# 检查 Python 版本是否为 3 或更高
if sys.version_info[0] >= 3:
    # 设置标准输出编码为 UTF-8，以支持中文输出，忽略类型检查
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


# 定义打开 PDF 文件的函数
def open_pdf(pdf_path):
    try:
        # 使用 fitz 打开 PDF 文件
        pdf_document = fitz.open(pdf_path)  # type: ignore
        # 返回打开的 PDF 文档
        return pdf_document
    except Exception as e:
        # 打印错误信息，说明无法打开 PDF 文件
        print(f"无法打开PDF文件：{pdf_path}。原因是：{e}")
        # 抛出异常
        raise e


# 定义打印绿色文本在红色背景的函数
def print_green_on_red(text):
    # 使用 cprint 打印绿色文本在红色背景，并加粗
    cprint(text, "green", "on_red", attrs=["bold"], end="\n\n")


# 定义打印绿色文本的函数
def print_green(text):
    # 打印空行
    print()
    # 使用 cprint 打印绿色文本，并加粗
    cprint(text, "green", attrs=["bold"], end="\n\n")


# 定义打印红色文本的函数
def print_red(text):
    # 打印空行
    print()
    # 使用 cprint 打印红色文本，并加粗
    cprint(text, "red", attrs=["bold"], end="\n\n")


# 定义打印黄色文本的函数
def print_yellow(text):
    # 打印空行
    print()
    # 使用 cprint 打印黄色文本，并加粗
    cprint(text, "yellow", attrs=["bold"], end="\n\n")


# 定义安全获取字典值的函数
def safe_get(dict_obj, key, default):
    # 尝试从字典中获取指定键的值
    val = dict_obj.get(key)
    # 如果值为 None，则返回默认值
    if val is None:
        return default
    else:
        # 否则返回获取的值
        return val


# 定义检查两个边界框是否重叠的函数
def is_bbox_overlap(bbox1, bbox2):
    """
    This function checks if bbox1 and bbox2 overlap or not

    Parameters
    ----------
    bbox1 : list
        bbox1
    bbox2 : list
        bbox2

    Returns
    -------
    bool
        True if bbox1 and bbox2 overlap, else False
    """
    # 解包第一个边界框的坐标
    x0_1, y0_1, x1_1, y1_1 = bbox1
    # 解包第二个边界框的坐标
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # 检查 x 方向是否重叠
    if x0_1 > x1_2 or x0_2 > x1_1:
        return False
    # 检查 y 方向是否重叠
    if y0_1 > y1_2 or y0_2 > y1_1:
        return False

    # 如果没有重叠条件成立，返回 True
    return True


# 定义检查一个边界框是否在另一个边界框内的函数
def is_in_bbox(bbox1, bbox2):
    """
    This function checks if bbox1 is in bbox2

    Parameters
    ----------
    bbox1 : list
        bbox1
    bbox2 : list
        bbox2

    Returns
    -------
    bool
        True if bbox1 is in bbox2, else False
    """
    # 解包第一个边界框的坐标
    x0_1, y0_1, x1_1, y1_1 = bbox1
    # 解包第二个边界框的坐标
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # 检查第一个边界框是否在第二个边界框内
    if x0_1 >= x0_2 and y0_1 >= y0_2 and x1_1 <= x1_2 and y1_1 <= y1_2:
        return True
    else:
        return False


# 定义计算段落最小边界框的函数
def calculate_para_bbox(lines):
    """
    This function calculates the minimum bbox of the paragraph

    Parameters
    ----------
    lines : list
        lines

    Returns
    -------
    para_bbox : list
        bbox of the paragraph
    """
    # 计算所有行中最小的 x0 坐标
    x0 = min(line["bbox"][0] for line in lines)
    # 计算所有行中最小的 y0 坐标
    y0 = min(line["bbox"][1] for line in lines)
    # 计算所有行中最大的 x1 坐标
    x1 = max(line["bbox"][2] for line in lines)
    # 计算所有行中最大的 y1 坐标
    y1 = max(line["bbox"][3] for line in lines)
    # 返回计算得到的段落边界框
    return [x0, y0, x1, y1]


# 定义检查当前行是否右对齐的函数
def is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction=2):
    """
    This function checks if the line is right aligned from its neighbors

    Parameters
    ----------
    curr_line_bbox : list
        bbox of the current line
    prev_line_bbox : list
        bbox of the previous line
    next_line_bbox : list
        bbox of the next line
    avg_char_width : float
        average of char widths
    direction : int
        0 for prev, 1 for next, 2 for both

    Returns
    -------
    bool
        True if the line is right aligned from its neighbors, False otherwise.
    """
    # 定义水平比率，用于判断右对齐
    horizontal_ratio = 0.5
    # 计算水平阈值，通过水平比率与平均字符宽度相乘得到
        horizontal_thres = horizontal_ratio * avg_char_width
    
        # 从当前行边界框中解包，获取 x1 坐标
        _, _, x1, _ = curr_line_bbox
        # 从前一行边界框中解包，若为空则返回默认值 (0, 0, 0, 0)
        _, _, prev_x1, _ = prev_line_bbox if prev_line_bbox else (0, 0, 0, 0)
        # 从下一行边界框中解包，若为空则返回默认值 (0, 0, 0, 0)
        _, _, next_x1, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)
    
        # 根据方向进行不同的阈值比较
        if direction == 0:
            # 判断当前行与前一行 x1 的距离是否小于水平阈值
            return abs(x1 - prev_x1) < horizontal_thres
        elif direction == 1:
            # 判断当前行与下一行 x1 的距离是否小于水平阈值
            return abs(x1 - next_x1) < horizontal_thres
        elif direction == 2:
            # 判断当前行与前一行和下一行 x1 的距离是否均小于水平阈值
            return abs(x1 - prev_x1) < horizontal_thres and abs(x1 - next_x1) < horizontal_thres
        else:
            # 如果方向不在预定义范围内，返回 False
            return False
# 检查当前行是否相对于相邻行左对齐的函数
def is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction=2):
    """
    此函数检查该行是否相对于其相邻行左对齐

    参数
    ----------
    curr_line_bbox : list
        当前行的边界框
    prev_line_bbox : list
        前一行的边界框
    next_line_bbox : list
        下一行的边界框
    avg_char_width : float
        字符宽度的平均值
    direction : int
        0 表示前一行，1 表示下一行，2 表示两者

    返回
    -------
    bool
        如果该行相对于相邻行左对齐则返回 True，否则返回 False。
    """
    # 设置水平比率
    horizontal_ratio = 0.5
    # 计算水平阈值
    horizontal_thres = horizontal_ratio * avg_char_width

    # 解构当前行的边界框，获取左侧坐标
    x0, _, _, _ = curr_line_bbox
    # 获取前一行的左侧坐标，如果不存在则默认为 (0, 0, 0, 0)
    prev_x0, _, _, _ = prev_line_bbox if prev_line_bbox else (0, 0, 0, 0)
    # 获取下一行的左侧坐标，如果不存在则默认为 (0, 0, 0, 0)
    next_x0, _, _, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)

    # 根据方向判断左对齐情况
    if direction == 0:
        # 检查当前行是否相对于前一行左对齐
        return abs(x0 - prev_x0) < horizontal_thres
    elif direction == 1:
        # 检查当前行是否相对于下一行左对齐
        return abs(x0 - next_x0) < horizontal_thres
    elif direction == 2:
        # 检查当前行是否同时相对于前一行和下一行左对齐
        return abs(x0 - prev_x0) < horizontal_thres and abs(x0 - next_x0) < horizontal_thres
    else:
        # 如果方向无效则返回 False
        return False


# 检查行文本是否以标点符号结尾的函数
def end_with_punctuation(line_text):
    """
    此函数检查该行是否以标点符号结尾
    """

    # 定义英语和中文的结束标点
    english_end_puncs = [".", "?", "!"]
    chinese_end_puncs = ["。", "？", "！"]
    # 合并结束标点列表
    end_puncs = english_end_puncs + chinese_end_puncs

    last_non_space_char = None
    # 反向遍历行文本中的字符
    for ch in line_text[::-1]:
        # 检查字符是否不是空白
        if not ch.isspace():
            # 记录最后一个非空白字符
            last_non_space_char = ch
            break

    # 如果没有找到非空白字符，则返回 False
    if last_non_space_char is None:
        return False

    # 检查最后一个非空白字符是否在结束标点列表中
    return last_non_space_char in end_puncs


# 检查列表是否为嵌套列表的函数
def is_nested_list(lst):
    # 判断输入是否为列表
    if isinstance(lst, list):
        # 检查列表中的任意元素是否为列表
        return any(isinstance(sub, list) for sub in lst)
    # 如果不是列表则返回 False
    return False
```