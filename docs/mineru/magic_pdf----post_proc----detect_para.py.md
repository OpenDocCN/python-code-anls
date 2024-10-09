# `.\MinerU\magic_pdf\post_proc\detect_para.py`

```
# 导入操作系统模块
import os
# 导入系统参数模块
import sys
# 导入JSON处理模块
import json
# 导入正则表达式模块
import re
# 导入数学模块
import math
# 导入Unicode字符处理模块
import unicodedata
# 从collections导入Counter类，用于计数
from collections import Counter


# 导入NumPy库
import numpy as np
# 导入cprint函数用于带颜色打印
from termcolor import cprint


# 从magic_pdf库导入fitz模块，处理PDF文件
from magic_pdf.libs.commons import fitz
# 从magic_pdf库导入NLPModels，用于自然语言处理
from magic_pdf.libs.nlp_utils import NLPModels


# 检查Python版本，如果是3及以上版本，则重新配置输出编码为UTF-8
if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


# 定义打开PDF文件的函数
def open_pdf(pdf_path):
    try:
        # 尝试打开指定路径的PDF文件
        pdf_document = fitz.open(pdf_path)  # type: ignore
        # 返回打开的PDF文档对象
        return pdf_document
    except Exception as e:
        # 如果发生异常，打印错误信息并抛出异常
        print(f"无法打开PDF文件：{pdf_path}。原因是：{e}")
        raise e


# 定义一个函数，用于以绿色字体在红色背景上打印文本
def print_green_on_red(text):
    # 使用cprint打印文本，设置颜色和样式
    cprint(text, "green", "on_red", attrs=["bold"], end="\n\n")


# 定义一个函数，用于以绿色字体打印文本
def print_green(text):
    # 打印换行
    print()
    # 使用cprint打印文本，设置颜色和样式
    cprint(text, "green", attrs=["bold"], end="\n\n")


# 定义一个函数，用于以红色字体打印文本
def print_red(text):
    # 打印换行
    print()
    # 使用cprint打印文本，设置颜色和样式
    cprint(text, "red", attrs=["bold"], end="\n\n")


# 定义一个函数，用于以黄色字体打印文本
def print_yellow(text):
    # 打印换行
    print()
    # 使用cprint打印文本，设置颜色和样式
    cprint(text, "yellow", attrs=["bold"], end="\n\n")


# 定义一个安全获取字典值的函数
def safe_get(dict_obj, key, default):
    # 从字典中获取指定键的值
    val = dict_obj.get(key)
    # 如果值为None，返回默认值
    if val is None:
        return default
    else:
        # 否则返回获取的值
        return val


# 定义一个函数，检查两个边界框是否重叠
def is_bbox_overlap(bbox1, bbox2):
    """
    该函数检查bbox1和bbox2是否重叠

    参数
    ----------
    bbox1 : list
        bbox1
    bbox2 : list
        bbox2

    返回
    -------
    bool
        如果bbox1和bbox2重叠，则返回True，否则返回False
    """
    # 解包第一个边界框的坐标
    x0_1, y0_1, x1_1, y1_1 = bbox1
    # 解包第二个边界框的坐标
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # 如果bbox1的左侧在bbox2的右侧或bbox2的左侧在bbox1的右侧，则不重叠
    if x0_1 > x1_2 or x0_2 > x1_1:
        return False
    # 如果bbox1的上侧在bbox2的下侧或bbox2的上侧在bbox1的下侧，则不重叠
    if y0_1 > y1_2 or y0_2 > y1_1:
        return False

    # 否则返回重叠
    return True


# 定义一个函数，检查一个边界框是否在另一个边界框内
def is_in_bbox(bbox1, bbox2):
    """
    该函数检查bbox1是否在bbox2内

    参数
    ----------
    bbox1 : list
        bbox1
    bbox2 : list
        bbox2

    返回
    -------
    bool
        如果bbox1在bbox2内，则返回True，否则返回False
    """
    # 解包第一个边界框的坐标
    x0_1, y0_1, x1_1, y1_1 = bbox1
    # 解包第二个边界框的坐标
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # 检查bbox1的所有边界是否在bbox2的边界内
    if x0_1 >= x0_2 and y0_1 >= y0_2 and x1_1 <= x1_2 and y1_1 <= y1_2:
        return True
    else:
        # 否则返回False
        return False


# 定义一个函数，计算段落的最小边界框
def calculate_para_bbox(lines):
    """
    该函数计算段落的最小边界框

    参数
    ----------
    lines : list
        lines

    返回
    -------
    para_bbox : list
        段落的边界框
    """
    # 获取所有行的最小x坐标
    x0 = min(line["bbox"][0] for line in lines)
    # 获取所有行的最小y坐标
    y0 = min(line["bbox"][1] for line in lines)
    # 获取所有行的最大x坐标
    x1 = max(line["bbox"][2] for line in lines)
    # 获取所有行的最大y坐标
    y1 = max(line["bbox"][3] for line in lines)
    # 返回计算出的边界框
    return [x0, y0, x1, y1]


# 定义一个函数，检查当前行是否与邻近行右对齐
def is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction=2):
    """
    该函数检查该行是否与其邻近行右对齐

    参数
    ----------
    curr_line_bbox : list
        当前行的边界框
    prev_line_bbox : list
        前一行的边界框
    next_line_bbox : list
        下一行的边界框
    avg_char_width : float
        字符的平均宽度
    direction : int
        # direction 参数为整数，表示对齐方向：0 表示与前一行对齐，1 表示与后一行对齐，2 表示与两者对齐

    Returns
    -------
    bool
        # 返回值为布尔类型，表示当前行是否与邻近行右对齐，如果是则返回 True，否则返回 False。
    """
    horizontal_ratio = 0.5
    # 定义一个水平对齐比率，设置为 0.5，表示使用平均字符宽度的一半作为对齐的阈值
    horizontal_thres = horizontal_ratio * avg_char_width
    # 计算当前行边界框的 x1 坐标
    _, _, x1, _ = curr_line_bbox
    # 获取前一行边界框的 x1 坐标，如果没有前一行，则默认值为 0
    _, _, prev_x1, _ = prev_line_bbox if prev_line_bbox else (0, 0, 0, 0)
    # 获取后一行边界框的 x1 坐标，如果没有后一行，则默认值为 0
    _, _, next_x1, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)

    # 如果方向为 0，判断当前行是否与前一行右对齐
    if direction == 0:
        return abs(x1 - prev_x1) < horizontal_thres
    # 如果方向为 1，判断当前行是否与后一行右对齐
    elif direction == 1:
        return abs(x1 - next_x1) < horizontal_thres
    # 如果方向为 2，判断当前行是否与前后两行都右对齐
    elif direction == 2:
        return abs(x1 - prev_x1) < horizontal_thres and abs(x1 - next_x1) < horizontal_thres
    # 如果方向不是 0、1 或 2，返回 False
    else:
        return False
# 检查当前行是否从邻近行左对齐的函数
def is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction=2):
    # 定义水平比率
    horizontal_ratio = 0.5
    # 计算水平阈值
    horizontal_thres = horizontal_ratio * avg_char_width

    # 获取当前行的左边界
    x0, _, _, _ = curr_line_bbox
    # 获取前一行的左边界，如果不存在则默认为0
    prev_x0, _, _, _ = prev_line_bbox if prev_line_bbox else (0, 0, 0, 0)
    # 获取下一行的左边界，如果不存在则默认为0
    next_x0, _, _, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)

    # 根据方向检查当前行是否左对齐
    if direction == 0:
        return abs(x0 - prev_x0) < horizontal_thres
    elif direction == 1:
        return abs(x0 - next_x0) < horizontal_thres
    elif direction == 2:
        return abs(x0 - prev_x0) < horizontal_thres and abs(x0 - next_x0) < horizontal_thres
    else:
        return False

# 检查行文本是否以标点符号结尾的函数
def end_with_punctuation(line_text):
    # 定义英文和中文的结束标点
    english_end_puncs = [".", "?", "!"]
    chinese_end_puncs = ["。", "？", "！"]
    # 合并所有结束标点
    end_puncs = english_end_puncs + chinese_end_puncs

    last_non_space_char = None
    # 从行文本的末尾开始遍历字符
    for ch in line_text[::-1]:
        # 如果字符不是空白，则记录该字符
        if not ch.isspace():
            last_non_space_char = ch
            break

    # 如果没有找到非空白字符，则返回False
    if last_non_space_char is None:
        return False

    # 检查最后一个非空白字符是否是结束标点
    return last_non_space_char in end_puncs

# 检查列表是否为嵌套列表的函数
def is_nested_list(lst):
    # 如果参数是列表，则检查其中是否有子列表
    if isinstance(lst, list):
        return any(isinstance(sub, list) for sub in lst)
    return False

# 定义密集单行块异常的类
class DenseSingleLineBlockException(Exception):
    # 初始化异常消息
    def __init__(self, message="DenseSingleLineBlockException"):
        self.message = message
        super().__init__(self.message)

    # 返回异常消息的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常消息的正式表示
    def __repr__(self):
        return f"{self.message}"

# 定义标题检测异常的类
class TitleDetectionException(Exception):
    # 初始化异常消息
    def __init__(self, message="TitleDetectionException"):
        self.message = message
        super().__init__(self.message)

    # 返回异常消息的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常消息的正式表示
    def __repr__(self):
        return f"{self.message}"

# 定义标题级别异常的类
class TitleLevelException(Exception):
    # 初始化异常消息
    def __init__(self, message="TitleLevelException"):
        self.message = message
        super().__init__(self.message)

    # 返回异常消息的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常消息的正式表示
    def __repr__(self):
        return f"{self.message}"

# 定义段落分隔异常的类
class ParaSplitException(Exception):
    # 该类定义了段落拆分的异常类型
    """
    This class defines the exception type for paragraph splitting.
    """

    # 初始化异常对象，接受一个可选的消息参数，默认为"ParaSplitException"
    def __init__(self, message="ParaSplitException"):
        # 将传入的消息存储为对象的属性
        self.message = message
        # 调用父类的初始化方法
        super().__init__(self.message)

    # 定义字符串表示方法，返回异常消息
    def __str__(self):
        return f"{self.message}"

    # 定义表示方法，返回异常消息，通常用于调试
    def __repr__(self):
        return f"{self.message}"
# 定义一个异常类，用于段落合并时的异常处理
class ParaMergeException(Exception):
    """
    该类定义了段落合并的异常类型。
    """

    # 初始化异常，默认消息为 "ParaMergeException"
    def __init__(self, message="ParaMergeException"):
        self.message = message  # 设置异常消息
        super().__init__(self.message)  # 调用父类构造函数

    # 返回异常消息的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常消息的可表示形式
    def __repr__(self):
        return f"{self.message}"


# 定义一个类，通过异常来丢弃 PDF 文件
class DiscardByException:
    """
    该类通过异常丢弃 PDF 文件
    """

    # 初始化方法，不需要任何参数
    def __init__(self) -> None:
        pass

    # 根据单行块异常丢弃 PDF 文件
    def discard_by_single_line_block(self, pdf_dic, exception: DenseSingleLineBlockException):
        """
        该函数根据单行块异常丢弃 PDF 文件

        参数
        ----------
        pdf_dic : dict
            PDF 文件字典
        exception : str
            异常消息

        返回
        -------
        error_message : str
        """
        exception_page_nums = 0  # 初始化异常页面计数
        page_num = 0  # 初始化页面计数
        for page_id, page in pdf_dic.items():  # 遍历 PDF 字典中的每一页
            if page_id.startswith("page_"):  # 检查页面 ID 是否以 "page_" 开头
                page_num += 1  # 增加页面计数
                if "preproc_blocks" in page.keys():  # 检查页面是否包含预处理块
                    preproc_blocks = page["preproc_blocks"]  # 获取预处理块

                    all_single_line_blocks = []  # 初始化单行块列表
                    for block in preproc_blocks:  # 遍历预处理块
                        if len(block["lines"]) == 1:  # 检查块是否只有一行
                            all_single_line_blocks.append(block)  # 添加到单行块列表

                    # 检查单行块占比，如果占比超过 90% 则计入异常页面
                    if len(preproc_blocks) > 0 and len(all_single_line_blocks) / len(preproc_blocks) > 0.9:
                        exception_page_nums += 1  # 增加异常页面计数

        if page_num == 0:  # 如果没有页面
            return None  # 返回 None

        # 如果异常页面占比超过 10%，返回异常消息
        if exception_page_nums / page_num > 0.1:  # 低比例意味着基本上，出现这种情况时，将被丢弃
            return exception.message

        return None  # 否则返回 None

    # 根据标题检测异常丢弃 PDF 文件
    def discard_by_title_detection(self, pdf_dic, exception: TitleDetectionException):
        """
        该函数根据标题检测异常丢弃 PDF 文件

        参数
        ----------
        pdf_dic : dict
            PDF 文件字典
        exception : str
            异常消息

        返回
        -------
        error_message : str
        """
        # return exception.message  # 返回异常消息（被注释掉）
        return None  # 返回 None

    # 根据标题级别异常丢弃 PDF 文件
    def discard_by_title_level(self, pdf_dic, exception: TitleLevelException):
        """
        该函数根据标题级别异常丢弃 PDF 文件

        参数
        ----------
        pdf_dic : dict
            PDF 文件字典
        exception : str
            异常消息

        返回
        -------
        error_message : str
        """
        # return exception.message  # 返回异常消息（被注释掉）
        return None  # 返回 None
    # 定义一个方法，根据拆分段落异常处理 PDF 文件
    def discard_by_split_para(self, pdf_dic, exception: ParaSplitException):
        """
        该函数通过拆分段落异常来丢弃 PDF 文件
    
        参数
        ----------
        pdf_dic : dict
            PDF 字典
        exception : str
            异常消息
    
        返回
        -------
        error_message : str
        """
        # 返回异常的消息（被注释掉）
        # return exception.message
        # 返回 None，表示没有错误消息
        return None
    
    # 定义一个方法，根据合并段落异常处理 PDF 文件
    def discard_by_merge_para(self, pdf_dic, exception: ParaMergeException):
        """
        该函数通过合并段落异常来丢弃 PDF 文件
    
        参数
        ----------
        pdf_dic : dict
            PDF 字典
        exception : str
            异常消息
    
        返回
        -------
        error_message : str
        """
        # 返回异常的消息（被注释掉）
        # return exception.message
        # 返回 None，表示没有错误消息
        return None
# 定义布局过滤器处理器类
class LayoutFilterProcessor:
    # 初始化方法
    def __init__(self) -> None:
        # 构造函数，不执行任何操作
        pass

    # 批量处理 PDF 块的方法
    def batch_process_blocks(self, pdf_dict):
        """
        该函数批量处理块。

        参数
        ----------
        self : object
            类的实例。

        pdf_dict : dict
            PDF 字典

        返回
        -------
        pdf_dict : dict
            PDF 字典
        """
        # 遍历 PDF 字典中的每一页及其块
        for page_id, blocks in pdf_dict.items():
            # 检查页面 ID 是否以 "page_" 开头
            if page_id.startswith("page_"):
                # 检查块中是否包含 "layout_bboxes" 和 "para_blocks" 这两个键
                if "layout_bboxes" in blocks.keys() and "para_blocks" in blocks.keys():
                    # 获取布局边界框对象
                    layout_bbox_objs = blocks["layout_bboxes"]
                    # 如果布局边界框对象为空，则跳过当前块
                    if layout_bbox_objs is None:
                        continue
                    # 提取布局边界框的坐标
                    layout_bboxes = [bbox_obj["layout_bbox"] for bbox_obj in layout_bbox_objs]

                    # 增大每个布局边界框的坐标值，以防文本丢失
                    layout_bboxes = [
                        [math.ceil(x0), math.ceil(y0), math.ceil(x1), math.ceil(y1)] for x0, y0, x1, y1 in layout_bboxes
                    ]

                    # 获取段落块
                    para_blocks = blocks["para_blocks"]
                    # 如果段落块为空，则跳过当前块
                    if para_blocks is None:
                        continue

                    # 遍历每个布局边界框
                    for lb_bbox in layout_bboxes:
                        # 遍历段落块及其索引
                        for i, para_block in enumerate(para_blocks):
                            # 获取段落块的边界框
                            para_bbox = para_block["bbox"]
                            # 初始化段落块的 in_layout 属性为 0
                            para_blocks[i]["in_layout"] = 0
                            # 检查段落块是否在布局边界框内
                            if is_in_bbox(para_bbox, lb_bbox):
                                # 如果在布局内，设置 in_layout 属性为 1
                                para_blocks[i]["in_layout"] = 1

                    # 更新块中的段落块
                    blocks["para_blocks"] = para_blocks

        # 返回处理后的 PDF 字典
        return pdf_dict


# 定义原始块处理器类
class RawBlockProcessor:
    # 初始化方法
    def __init__(self) -> None:
        # 设置 y 方向的容差值
        self.y_tolerance = 2
        # 初始化 PDF 字典
        self.pdf_dic = {}

    # 私有方法，分解字体标志
    def __span_flags_decomposer(self, span_flags):
        """
        将字体标志转化为人类可读格式。

        参数
        ----------
        self : object
            类的实例。

        span_flags : int
            字体标志

        返回
        -------
        l : dict
            分解后的标志
        """

        # 创建一个字典来存储分解后的标志
        l = {
            "is_superscript": False,
            "is_italic": False,
            "is_serifed": False,
            "is_sans_serifed": False,
            "is_monospaced": False,
            "is_proportional": False,
            "is_bold": False,
        }

        # 检查字体标志是否包含上标标志
        if span_flags & 2**0:
            l["is_superscript"] = True  # 表示上标

        # 检查字体标志是否包含斜体标志
        if span_flags & 2**1:
            l["is_italic"] = True  # 表示斜体

        # 检查字体标志是否包含衬线字体标志
        if span_flags & 2**2:
            l["is_serifed"] = True  # 表示衬线字体
        else:
            l["is_sans_serifed"] = True  # 表示非衬线字体

        # 检查字体标志是否包含等宽字体标志
        if span_flags & 2**3:
            l["is_monospaced"] = True  # 表示等宽字体
        else:
            l["is_proportional"] = True  # 表示比例字体

        # 检查字体标志是否包含粗体标志
        if span_flags & 2**4:
            l["is_bold"] = True  # 表示粗体

        # 返回分解后的字体标志字典
        return l
    # 定义一个私有方法，用于创建新行
    def __make_new_lines(self, raw_lines):
        """
        该函数用于生成新行。
    
        参数
        ----------
        self : object
            类的实例。
    
        raw_lines : list
            原始行列表。
    
        返回
        -------
        new_lines : list
            新行列表。
        """
        # 初始化新行列表
        new_lines = []
        # 初始化新行变量
        new_line = None
    
        # 遍历原始行列表
        for raw_line in raw_lines:
            # 获取原始行的边界框
            raw_line_bbox = raw_line["bbox"]
            # 获取原始行的跨度信息
            raw_line_spans = raw_line["spans"]
            # 合并跨度的文本为一个字符串
            raw_line_text = "".join([span["text"] for span in raw_line_spans])
            # 获取原始行的方向，默认为 None
            raw_line_dir = raw_line.get("dir", None)
    
            # 初始化解构后的行跨度列表
            decomposed_line_spans = []
            # 遍历每个跨度
            for span in raw_line_spans:
                # 获取原始标志
                raw_flags = span["flags"]
                # 解构标志
                decomposed_flags = self.__span_flags_decomposer(raw_flags)
                # 将解构后的标志添加到跨度中
                span["decomposed_flags"] = decomposed_flags
                # 添加到解构后的行跨度列表
                decomposed_line_spans.append(span)
    
            # 如果新行尚未初始化，处理第一行
            if new_line is None:  
                new_line = {
                    # 设置新行的边界框
                    "bbox": raw_line_bbox,
                    # 设置新行的文本
                    "text": raw_line_text,
                    # 设置新行的方向，默认为 (0, 0)
                    "dir": raw_line_dir if raw_line_dir else (0, 0),
                    # 设置新行的解构跨度
                    "spans": decomposed_line_spans,
                }
            else:  # 处理后续行
                # 检查当前行与新行的 Y 轴位置差异是否在容忍范围内
                if (
                    abs(raw_line_bbox[1] - new_line["bbox"][1]) <= self.y_tolerance
                    and abs(raw_line_bbox[3] - new_line["bbox"][3]) <= self.y_tolerance
                ):
                    # 更新新行的边界框
                    new_line["bbox"] = (
                        min(new_line["bbox"][0], raw_line_bbox[0]),  # 左边界
                        new_line["bbox"][1],  # 上边界
                        max(new_line["bbox"][2], raw_line_bbox[2]),  # 右边界
                        raw_line_bbox[3],  # 下边界
                    )
                    # 添加当前行的文本到新行
                    new_line["text"] += raw_line_text
                    # 将当前行的跨度添加到新行的跨度中
                    new_line["spans"].extend(raw_line_spans)
                    # 更新新行的方向
                    new_line["dir"] = (
                        new_line["dir"][0] + raw_line_dir[0],
                        new_line["dir"][1] + raw_line_dir[1],
                    )
                else:
                    # 如果不在容忍范围，保存当前新行并初始化新行
                    new_lines.append(new_line)
                    new_line = {
                        # 设置新行的边界框
                        "bbox": raw_line_bbox,
                        # 设置新行的文本
                        "text": raw_line_text,
                        # 设置新行的方向，默认为 (0, 0)
                        "dir": raw_line_dir if raw_line_dir else (0, 0),
                        # 设置新行的跨度
                        "spans": raw_line_spans,
                    }
        # 如果新行存在，添加到新行列表中
        if new_line:
            new_lines.append(new_line)
    
        # 返回生成的新行列表
        return new_lines
    # 定义一个私有方法用于生成新的块
    def __make_new_block(self, raw_block):
        """
        该函数创建一个新的块。
    
        参数
        ----------
        self : object
            类的实例。
        ----------
        raw_block : dict
            原始块
    
        返回
        -------
        new_block : dict
        """
        # 初始化一个空字典用于存储新块的信息
        new_block = {}
    
        # 从原始块中获取块的编号
        block_id = raw_block["number"]
        # 从原始块中获取边界框信息
        block_bbox = raw_block["bbox"]
        # 从原始块中提取文本，将所有行的所有片段的文本连接起来
        block_text = "".join(span["text"] for line in raw_block["lines"] for span in line["spans"])
        # 获取原始块中的所有行
        raw_lines = raw_block["lines"]
        # 调用方法生成新的行信息
        block_lines = self.__make_new_lines(raw_lines)
    
        # 将块的编号添加到新块中
        new_block["block_id"] = block_id
        # 将边界框信息添加到新块中
        new_block["bbox"] = block_bbox
        # 将提取的文本添加到新块中
        new_block["text"] = block_text
        # 将处理后的行信息添加到新块中
        new_block["lines"] = block_lines
    
        # 返回新生成的块
        return new_block
    
    # 定义一个方法用于批量处理块
    def batch_process_blocks(self, pdf_dic):
        """
        该函数批量处理块。
    
        参数
        ----------
        self : object
            类的实例。
        ----------
        blocks : list
            输入块是原始块的列表。
    
        返回
        -------
        result_dict : dict
            结果字典
        """
    
        # 遍历 PDF 字典中的每一页及其块
        for page_id, blocks in pdf_dic.items():
            # 检查页面 ID 是否以 "page_" 开头
            if page_id.startswith("page_"):
                # 初始化一个空列表用于存储处理后的块
                para_blocks = []
                # 检查原始块中是否包含预处理块
                if "preproc_blocks" in blocks.keys():
                    # 获取预处理块
                    input_blocks = blocks["preproc_blocks"]
                    # 遍历每一个原始块
                    for raw_block in input_blocks:
                        # 生成新的块并添加到 para_blocks 列表
                        new_block = self.__make_new_block(raw_block)
                        para_blocks.append(new_block)
    
                # 将处理后的块存储回原始块的字典中
                blocks["para_blocks"] = para_blocks
    
        # 返回处理后的 PDF 字典
        return pdf_dic
# 定义一个用于计算区块统计信息的类
class BlockStatisticsCalculator:
    """
    该类计算区块的统计信息。
    """

    # 初始化方法，当前不执行任何操作
    def __init__(self) -> None:
        pass

    # 定义一个私有方法，用于生成新的区块
    def __make_new_block(self, input_block):
        # 创建一个新的空字典以存储新区块的信息
        new_block = {}

        # 从输入区块中获取原始行数据
        raw_lines = input_block["lines"]
        # 计算原始行的统计信息
        stats = self.__calc_stats_of_new_lines(raw_lines)

        # 获取输入区块的 ID
        block_id = input_block["block_id"]
        # 获取输入区块的边界框
        block_bbox = input_block["bbox"]
        # 获取输入区块的文本内容
        block_text = input_block["text"]
        # 将原始行赋值给区块行
        block_lines = raw_lines
        # 从统计信息中提取各类统计数据
        block_avg_left_boundary = stats[0]
        block_avg_right_boundary = stats[1]
        block_avg_char_width = stats[2]
        block_avg_char_height = stats[3]
        block_font_type = stats[4]
        block_font_size = stats[5]
        block_direction = stats[6]
        block_median_font_size = stats[7]

        # 将各类统计信息添加到新区块字典中
        new_block["block_id"] = block_id
        new_block["bbox"] = block_bbox
        new_block["text"] = block_text
        new_block["dir"] = block_direction
        new_block["X0"] = block_avg_left_boundary
        new_block["X1"] = block_avg_right_boundary
        new_block["avg_char_width"] = block_avg_char_width
        new_block["avg_char_height"] = block_avg_char_height
        new_block["block_font_type"] = block_font_type
        new_block["block_font_size"] = block_font_size
        new_block["lines"] = block_lines
        new_block["median_font_size"] = block_median_font_size

        # 返回生成的新区块字典
        return new_block

    # 定义一个批量处理区块的方法
    def batch_process_blocks(self, pdf_dic):
        """
        此函数批量处理区块。

        Parameters
        ----------
        self : object
            类的实例。
        ----------
        blocks : list
            输入区块是原始区块的列表。
            模式可以参考键“preproc_blocks”的值。

        Returns
        -------
        result_dict : dict
            结果字典
        """

        # 遍历 PDF 字典中的每一页及其区块
        for page_id, blocks in pdf_dic.items():
            # 检查页面 ID 是否以 "page_" 开头
            if page_id.startswith("page_"):
                # 初始化一个空列表以存储段落区块
                para_blocks = []
                # 如果区块中包含 "para_blocks" 键
                if "para_blocks" in blocks.keys():
                    # 获取输入的段落区块
                    input_blocks = blocks["para_blocks"]
                    # 遍历每个输入区块
                    for input_block in input_blocks:
                        # 生成新的区块并添加到段落区块列表中
                        new_block = self.__make_new_block(input_block)
                        para_blocks.append(new_block)

                # 将处理后的段落区块更新回原始区块字典中
                blocks["para_blocks"] = para_blocks

        # 返回处理后的 PDF 字典
        return pdf_dic


# 定义一个用于计算文档统计信息的类
class DocStatisticsCalculator:
    """
    该类计算文档的统计信息。
    """

    # 初始化方法，当前不执行任何操作
    def __init__(self) -> None:
        pass

# 定义一个用于处理标题的类
class TitleProcessor:
    """
    该类处理标题。
    """
    # 初始化方法，接受可变数量的文档统计数据
        def __init__(self, *doc_statistics) -> None:
            # 如果传入的文档统计数据不为空
            if len(doc_statistics) > 0:
                # 将第一个文档统计数据赋值给实例变量
                self.doc_statistics = doc_statistics[0]
    
            # 创建 NLPModels 的实例，赋值给实例变量
            self.nlp_model = NLPModels()
            # 定义最大标题层级，赋值给实例变量
            self.MAX_TITLE_LEVEL = 3
            # 定义正则表达式模式，用于匹配各种格式的标题
            self.numbered_title_pattern = r"""
                ^                                 # 行首
                (                                 # 开始捕获组
                    [\(\（]\d+[\)\）]              # 括号内数字，支持中文和英文括号，例如：(1) 或 （1）
                    |\d+[\)\）]\s                  # 数字后跟右括号和空格，支持中文和英文括号，例如：2) 或 2）
                    |[\(\（][A-Z][\)\）]            # 括号内大写字母，支持中文和英文括号，例如：(A) 或 （A）
                    |[A-Z][\)\）]\s                # 大写字母后跟右括号和空格，例如：A) 或 A）
                    |[\(\（][IVXLCDM]+[\)\）]       # 括号内罗马数字，支持中文和英文括号，例如：(I) 或 （I）
                    |[IVXLCDM]+[\)\）]\s            # 罗马数字后跟右括号和空格，例如：I) 或 I）
                    |\d+(\.\d+)*\s                # 数字或复合数字编号后跟空格，例如：1. 或 3.2.1 
                    |[一二三四五六七八九十百千]+[、\s]       # 中文序号后跟顿号和空格，例如：一、
                    |[\（|\(][一二三四五六七八九十百千]+[\）|\)]\s*  # 中文括号内中文序号后跟空格，例如：（一）
                    |[A-Z]\.\d+(\.\d+)?\s         # 大写字母后跟点和数字，例如：A.1 或 A.1.1
                    |[\(\（][a-z][\)\）]            # 括号内小写字母，支持中文和英文括号，例如：(a) 或 （a）
                    |[a-z]\)\s                    # 小写字母后跟右括号和空格，例如：a) 
                    |[A-Z]-\s                     # 大写字母后跟短横线和空格，例如：A- 
                    |\w+:\s                       # 英文序号词后跟冒号和空格，例如：First: 
                    |第[一二三四五六七八九十百千]+[章节部分条款]\s # 以“第”开头的中文标题后跟空格
                    |[IVXLCDM]+\.                 # 罗马数字后跟点，例如：I.
                    |\d+\.\s                      # 单个数字后跟点和空格，例如：1. 
                )                                 # 结束捕获组
                .+                                # 标题的其余部分
            """
    
        # 定义方法，用于判断当前行是否可能是标题
        def _is_potential_title(
            self,
            curr_line,                      # 当前行内容
            prev_line,                      # 前一行内容
            prev_line_is_title,            # 前一行是否是标题的标志
            next_line,                      # 下一行内容
            avg_char_width,                # 字符的平均宽度
            avg_char_height,               # 字符的平均高度
            median_font_size,              # 字体的中位大小
    # 检测每个段落块的标题
    def _detect_title(self, input_block):
        """
        使用 'is_potential_title' 函数检测每个段落块的标题。
        如果某行是标题，则该行的 'is_title' 键的值将被设置为 True。
        """
    
        # 获取输入块中的原始行
        raw_lines = input_block["lines"]
    
        # 上一行是否为标题的标志
        prev_line_is_title_flag = False
    
        # 遍历每一行及其索引
        for i, curr_line in enumerate(raw_lines):
            # 获取当前行的前一行，如果不存在则为 None
            prev_line = raw_lines[i - 1] if i > 0 else None
            # 获取当前行的下一行，如果不存在则为 None
            next_line = raw_lines[i + 1] if i < len(raw_lines) - 1 else None
    
            # 获取块的平均字符宽度
            blk_avg_char_width = input_block["avg_char_width"]
            # 获取块的平均字符高度
            blk_avg_char_height = input_block["avg_char_height"]
            # 获取块的中位字体大小
            blk_media_font_size = input_block["median_font_size"]
    
            # 检测当前行是否为标题及是否为作者或组织列表
            is_title, is_author_or_org_list = self._is_potential_title(
                curr_line,
                prev_line,
                prev_line_is_title_flag,
                next_line,
                blk_avg_char_width,
                blk_avg_char_height,
                blk_media_font_size,
            )
    
            # 如果当前行是标题，则设置其 'is_title' 属性为 True
            if is_title:
                curr_line["is_title"] = is_title
                # 更新标志，表示上一行是标题
                prev_line_is_title_flag = True
            else:
                # 当前行不是标题，设置其 'is_title' 属性为 False
                curr_line["is_title"] = False
                # 更新标志，表示上一行不是标题
                prev_line_is_title_flag = False
    
            # 打印调试信息（被注释掉）
            # print(f"curr_line['text']: {curr_line['text']}")
            # print(f"curr_line['is_title']: {curr_line['is_title']}")
            # print(f"prev_line['text']: {prev_line['text'] if prev_line else None}")
            # print(f"prev_line_is_title_flag: {prev_line_is_title_flag}")
            # print()
    
            # 如果当前行是作者或组织列表，则设置其属性
            if is_author_or_org_list:
                curr_line["is_author_or_org_list"] = is_author_or_org_list
            else:
                # 否则设置其 'is_author_or_org_list' 属性为 False
                curr_line["is_author_or_org_list"] = False
    
        # 返回更新后的输入块
        return input_block
    # 批量处理 PDF 字典中的块以检测标题
    def batch_detect_titles(self, pdf_dic):
        """
        该函数批量处理块以检测标题。
    
        参数
        ----------
        pdf_dict : dict
            结果字典
    
        返回
        -------
        pdf_dict : dict
            结果字典
        """
        # 初始化标题计数器
        num_titles = 0
    
        # 遍历 PDF 字典中的每一页及其对应的块
        for page_id, blocks in pdf_dic.items():
            # 检查页面 ID 是否以 "page_" 开头
            if page_id.startswith("page_"):
                # 初始化段落块列表
                para_blocks = []
                # 如果存在 "para_blocks" 键，则获取其值
                if "para_blocks" in blocks.keys():
                    para_blocks = blocks["para_blocks"]
    
                    # 初始化单行块列表
                    all_single_line_blocks = []
                    # 遍历段落块，筛选出单行块
                    for block in para_blocks:
                        if len(block["lines"]) == 1:
                            all_single_line_blocks.append(block)
    
                    # 初始化新的段落块列表
                    new_para_blocks = []
                    # 如果并非所有块都是单行块
                    if not len(all_single_line_blocks) == len(para_blocks):
                        # 遍历段落块，检测每个块的标题
                        for para_block in para_blocks:
                            new_block = self._detect_title(para_block)
                            new_para_blocks.append(new_block)
                            # 统计新块中标题的数量
                            num_titles += sum([line.get("is_title", 0) for line in new_block["lines"]])
                    else:  # 如果所有块都是单行块
                        # 直接将单行块添加到新的段落块列表
                        for para_block in para_blocks:
                            new_para_blocks.append(para_block)
                            # 统计单行块中标题的数量
                            num_titles += sum([line.get("is_title", 0) for line in para_block["lines"]])
                    # 更新段落块为新的段落块列表
                    para_blocks = new_para_blocks
    
                # 将更新后的段落块列表存回块中
                blocks["para_blocks"] = para_blocks
    
                # 遍历每个段落块，检测是否为标题
                for para_block in para_blocks:
                    # 检查所有行是否都是标题
                    all_titles = all(safe_get(line, "is_title", False) for line in para_block["lines"])
                    # 计算段落中所有行的文本长度
                    para_text_len = sum([len(line["text"]) for line in para_block["lines"]])
                    # 如果所有行都是标题且段落长度小于 200
                    if (
                        all_titles and para_text_len < 200
                    ):  # 段落总长度小于 200，超过此长度不应为标题
                        para_block["is_block_title"] = 1
                    else:
                        para_block["is_block_title"] = 0
    
                    # 检查段落中是否所有行都是作者或组织列表
                    all_name_or_org_list_to_be_removed = all(
                        safe_get(line, "is_author_or_org_list", False) for line in para_block["lines"]
                    )
                    # 如果是作者或组织列表且在第一页
                    if all_name_or_org_list_to_be_removed and page_id == "page_0":
                        para_block["is_block_an_author_or_org_list"] = 1
                    else:
                        para_block["is_block_an_author_or_org_list"] = 0
    
        # 更新 PDF 字典中的标题统计
        pdf_dic["statistics"]["num_titles"] = num_titles
    
        # 返回更新后的 PDF 字典
        return pdf_dic
    # 根据字体大小识别标题层级
    def _recog_title_level(self, title_blocks):
        """
        该函数根据标题的字体大小确定标题层级。
    
        参数
        ----------
        title_blocks : list
    
        返回
        -------
        title_blocks : list
        """
    
        # 从每个标题块中提取字体大小，形成 NumPy 数组
        font_sizes = np.array([safe_get(tb["block"], "block_font_size", 0) for tb in title_blocks])
    
        # 使用字体大小的均值和标准差去除极端值
        mean_font_size = np.mean(font_sizes)  # 计算字体大小的均值
        std_font_size = np.std(font_sizes)    # 计算字体大小的标准差
        min_extreme_font_size = mean_font_size - std_font_size  # 极小字体大小阈值
        max_extreme_font_size = mean_font_size + std_font_size  # 极大字体大小阈值
    
        # 计算标题层级的阈值
        middle_font_sizes = font_sizes[(font_sizes > min_extreme_font_size) & (font_sizes < max_extreme_font_size)]
        if middle_font_sizes.size > 0:
            middle_mean_font_size = np.mean(middle_font_sizes)  # 中间字体大小的均值
            level_threshold = middle_mean_font_size  # 设置层级阈值为中间均值
        else:
            level_threshold = mean_font_size  # 没有中间值时使用均值
    
        # 遍历每个标题块
        for tb in title_blocks:
            title_block = tb["block"]  # 获取当前标题块
            title_font_size = safe_get(title_block, "block_font_size", 0)  # 获取标题字体大小
    
            current_level = 1  # 初始化标题层级，最大的层级为 1
    
            # print(f"Before adjustment by font size, {current_level}")
            if title_font_size >= max_extreme_font_size:
                current_level = 1  # 字体过大，层级为 1
            elif title_font_size <= min_extreme_font_size:
                current_level = 3  # 字体过小，层级为 3
            elif float(title_font_size) >= float(level_threshold):
                current_level = 2  # 字体在阈值以上，层级为 2
            else:
                current_level = 3  # 字体在阈值以下，层级为 3
            # print(f"After adjustment by font size, {current_level}")
    
            title_block["block_title_level"] = current_level  # 更新标题块的层级
    
        return title_blocks  # 返回更新后的标题块列表
    
    # 批量处理标题层级识别
    def batch_recog_title_level(self, pdf_dic):
        """
        该函数批量处理块以识别标题层级。
    
        参数
        ----------
        pdf_dict : dict
            结果字典
    
        返回
        -------
        pdf_dict : dict
            结果字典
        """
        title_blocks = []  # 初始化标题块列表
    
        # 收集所有标题
        for page_id, blocks in pdf_dic.items():
            if page_id.startswith("page_"):  # 处理以 "page_" 开头的页面
                para_blocks = blocks.get("para_blocks", [])  # 获取段落块
                for block in para_blocks:
                    if block.get("is_block_title"):  # 检查块是否为标题
                        title_obj = {"page_id": page_id, "block": block}  # 创建标题对象
                        title_blocks.append(title_obj)  # 添加到标题块列表
    
        # 确定标题层级
        if title_blocks:  # 如果有标题块
            # 根据字体大小确定标题层级
            title_blocks = self._recog_title_level(title_blocks)
    
        return pdf_dic  # 返回包含标题层级的结果字典
# 处理块结束的类
class BlockTerminationProcessor:
    """
    该类用于处理块终止。
    """

    # 初始化方法
    def __init__(self) -> None:
        # 该方法目前不执行任何操作
        pass

    # 检查当前行是否与其邻居一致
    def _is_consistent_lines(
        self,
        curr_line,
        prev_line,
        next_line,
        consistent_direction,  # 0表示前一行，1表示后一行，2表示两者
    ):
        """
        检查当前行是否与其邻居一致

        参数
        ----------
        curr_line : dict
            当前行
        prev_line : dict
            前一行
        next_line : dict
            后一行
        consistent_direction : int
            0表示前一行，1表示后一行，2表示两者

        返回
        -------
        bool
            如果当前行与邻居一致则返回True，否则返回False。
        """

        # 获取当前行的字体大小
        curr_line_font_size = curr_line["spans"][0]["size"]
        # 获取当前行的字体类型并转为小写
        curr_line_font_type = curr_line["spans"][0]["font"].lower()

        # 如果一致方向为0（与前一行比较）
        if consistent_direction == 0:
            # 检查前一行是否存在
            if prev_line:
                # 获取前一行的字体大小
                prev_line_font_size = prev_line["spans"][0]["size"]
                # 获取前一行的字体类型并转为小写
                prev_line_font_type = prev_line["spans"][0]["font"].lower()
                # 返回当前行是否与前一行一致
                return curr_line_font_size == prev_line_font_size and curr_line_font_type == prev_line_font_type
            else:
                # 如果前一行不存在，返回False
                return False

        # 如果一致方向为1（与后一行比较）
        elif consistent_direction == 1:
            # 检查后一行是否存在
            if next_line:
                # 获取后一行的字体大小
                next_line_font_size = next_line["spans"][0]["size"]
                # 获取后一行的字体类型并转为小写
                next_line_font_type = next_line["spans"][0]["font"].lower()
                # 返回当前行是否与后一行一致
                return curr_line_font_size == next_line_font_size and curr_line_font_type == next_line_font_type
            else:
                # 如果后一行不存在，返回False
                return False

        # 如果一致方向为2（与前后一行比较）
        elif consistent_direction == 2:
            # 检查前一行和后一行是否都存在
            if prev_line and next_line:
                # 获取前一行的字体大小
                prev_line_font_size = prev_line["spans"][0]["size"]
                # 获取前一行的字体类型并转为小写
                prev_line_font_type = prev_line["spans"][0]["font"].lower()
                # 获取后一行的字体大小
                next_line_font_size = next_line["spans"][0]["size"]
                # 获取后一行的字体类型并转为小写
                next_line_font_type = next_line["spans"][0]["font"].lower()
                # 返回当前行是否与前一行和后一行都一致
                return (curr_line_font_size == prev_line_font_size and curr_line_font_type == prev_line_font_type) and (
                    curr_line_font_size == next_line_font_size and curr_line_font_type == next_line_font_type
                )
            else:
                # 如果前一行或后一行不存在，返回False
                return False

        # 如果一致方向不合法，返回False
        else:
            return False
    # 定义一个检查行是否为常规行的私有方法
    def _is_regular_line(self, curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, X0, X1, avg_line_height):
        """
        该函数检查给定的行是否为常规行

        参数
        ----------
        curr_line_bbox : list
            当前行的边界框
        prev_line_bbox : list
            上一行的边界框
        next_line_bbox : list
            下一行的边界框
        avg_char_width : float
            字符宽度的平均值
        X0 : float
            X0 值的中位数，表示页面左边界的平均边界
        X1 : float
            X1 值的中位数，表示页面右边界的平均边界
        avg_line_height : float
            行高的平均值

        返回
        -------
        bool
            如果该行是常规行，则返回 True，否则返回 False。
        """
        # 定义水平和垂直的比例
        horizontal_ratio = 0.5
        vertical_ratio = 0.5
        # 计算水平和垂直的阈值
        horizontal_thres = horizontal_ratio * avg_char_width
        vertical_thres = vertical_ratio * avg_line_height

        # 解构当前行的边界框，获取坐标
        x0, y0, x1, y1 = curr_line_bbox

        # 检查当前行的左边界是否接近页面左边界
        x0_near_X0 = abs(x0 - X0) < horizontal_thres
        # 检查当前行的右边界是否接近页面右边界
        x1_near_X1 = abs(x1 - X1) < horizontal_thres

        # 检查上一行是否为段落的结束
        prev_line_is_end_of_para = prev_line_bbox and (abs(prev_line_bbox[2] - X1) > avg_char_width)

        # 初始化上方的间距是否充足的标志
        sufficient_spacing_above = False
        # 如果存在上一行，计算上方的间距
        if prev_line_bbox:
            vertical_spacing_above = y1 - prev_line_bbox[3]
            # 判断上方的间距是否大于阈值
            sufficient_spacing_above = vertical_spacing_above > vertical_thres

        # 初始化下方的间距是否充足的标志
        sufficient_spacing_below = False
        # 如果存在下一行，计算下方的间距
        if next_line_bbox:
            vertical_spacing_below = next_line_bbox[1] - y0
            # 判断下方的间距是否大于阈值
            sufficient_spacing_below = vertical_spacing_below > vertical_thres

        # 返回是否为常规行的判断结果
        return (
            (sufficient_spacing_above or sufficient_spacing_below)  # 上下方的间距充足
            or (not x0_near_X0 and not x1_near_X1)  # 左右边界都不接近页面边界
            or prev_line_is_end_of_para  # 上一行是段落的结束
        )
    # 定义一个方法，检查当前行是否可能是段落的结束
    def _is_possible_end_of_para(self, curr_line, next_line, X0, X1, avg_char_width):
        """
        该函数检查当前行是否可能是段落的结束

        参数
        ----------
        curr_line : dict
            当前行
        next_line : dict
            下一行
        X0 : float
            表示页面左侧平均边界的 x0 值的中位数
        X1 : float
            表示页面右侧平均边界的 x1 值的中位数
        avg_char_width : float
            字符宽度的平均值

        返回
        -------
        bool
            如果该行可能是段落的结束则返回 True，否则返回 False。
        """

        # 初始化行作为段落结束的置信度
        end_confidence = 0.5  
        # 记录决策路径
        decision_path = []  

        # 获取当前行的边界框
        curr_line_bbox = curr_line["bbox"]
        # 获取下一行的边界框，如果不存在则为 None
        next_line_bbox = next_line["bbox"] if next_line else None

        # 左侧和右侧的水平比率
        left_horizontal_ratio = 0.5
        right_horizontal_ratio = 0.5

        # 解包当前行的边界框坐标
        x0, _, x1, y1 = curr_line_bbox
        # 解包下一行的边界框坐标，如果不存在则默认为 (0, 0, 0, 0)
        next_x0, next_y0, _, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)

        # 检查当前行的 x0 值是否接近页面左侧边界
        x0_near_X0 = abs(x0 - X0) < left_horizontal_ratio * avg_char_width
        # 如果接近，增加置信度，并记录决策路径
        if x0_near_X0:
            end_confidence += 0.1
            decision_path.append("x0_near_X0")

        # 检查当前行的 x1 值是否小于页面右侧边界
        x1_smaller_than_X1 = x1 < X1 - right_horizontal_ratio * avg_char_width
        # 如果小于，增加置信度，并记录决策路径
        if x1_smaller_than_X1:
            end_confidence += 0.1
            decision_path.append("x1_smaller_than_X1")

        # 检查下一行是否可能是段落的开始
        next_line_is_start_of_para = (
            next_line_bbox
            and (next_x0 > X0 + left_horizontal_ratio * avg_char_width)
            and (not is_line_left_aligned_from_neighbors(curr_line_bbox, None, next_line_bbox, avg_char_width, direction=1))
        )
        # 如果下一行是段落的开始，增加置信度，并记录决策路径
        if next_line_is_start_of_para:
            end_confidence += 0.2
            decision_path.append("next_line_is_start_of_para")

        # 检查当前行是否与邻近行左对齐
        is_line_left_aligned_from_neighbors_bool = is_line_left_aligned_from_neighbors(
            curr_line_bbox, None, next_line_bbox, avg_char_width
        )
        # 如果左对齐，增加置信度，并记录决策路径
        if is_line_left_aligned_from_neighbors_bool:
            end_confidence += 0.1
            decision_path.append("line_is_left_aligned_from_neighbors")

        # 检查当前行是否与邻近行右对齐
        is_line_right_aligned_from_neighbors_bool = is_line_right_aligned_from_neighbors(
            curr_line_bbox, None, next_line_bbox, avg_char_width
        )
        # 如果没有右对齐，增加置信度，并记录决策路径
        if not is_line_right_aligned_from_neighbors_bool:
            end_confidence += 0.1
            decision_path.append("line_is_not_right_aligned_from_neighbors")

        # 检查当前行是否以标点符号结束，并结合其他条件
        is_end_of_para = end_with_punctuation(curr_line["text"]) and (
            (x0_near_X0 and x1_smaller_than_X1)
            or (is_line_left_aligned_from_neighbors_bool and not is_line_right_aligned_from_neighbors_bool)
        )

        # 返回是否为段落结束、置信度及决策路径
        return (is_end_of_para, end_confidence, decision_path)
    # 批量处理 PDF 字典中所有页面的块
    def batch_process_blocks(self, pdf_dict):
        """
        解析所有页面的块。

        参数
        ----------
        pdf_dict : dict
            PDF 字典。
        filter_blocks : list
            用于过滤的边界框列表。

        返回
        -------
        result_dict : dict
            结果字典。

        """

        # 初始化段落计数器
        num_paras = 0

        # 遍历 PDF 字典中的每一页
        for page_id, page in pdf_dict.items():
            # 检查页面 ID 是否以 "page_" 开头
            if page_id.startswith("page_"):
                # 初始化当前页面的段落块列表
                para_blocks = []
                # 检查页面中是否存在 "para_blocks" 键
                if "para_blocks" in page.keys():
                    # 获取输入的块
                    input_blocks = page["para_blocks"]
                    # 遍历每个输入块
                    for input_block in input_blocks:
                        # 处理每个输入块，获取新的块
                        new_block = self._cut_paras_per_block(input_block)
                        # 将新的块添加到段落块列表
                        para_blocks.append(new_block)
                        # 增加段落计数器
                        num_paras += len(new_block["paras"])

                # 将处理后的段落块更新回页面字典
                page["para_blocks"] = para_blocks

        # 更新 PDF 字典中的统计信息，记录段落总数
        pdf_dict["statistics"]["num_paras"] = num_paras
        # 返回更新后的 PDF 字典
        return pdf_dict
# 定义一个类，用于处理块以检测块的延续性
class BlockContinuationProcessor:
    """
    This class is used to process the blocks to detect block continuations.
    """

    # 初始化方法，创建 BlockContinuationProcessor 实例
    def __init__(self) -> None:
        # 当前没有任何初始化操作
        pass

    # 定义一个私有方法，用于检查两个字体类型是否相似
    def __is_similar_font_type(self, font_type_1, font_type_2, prefix_length_ratio=0.3):
        """
        This function checks if the two font types are similar.
        Definition of similar font types: the two font types have a common prefix,
        and the length of the common prefix is at least a certain ratio of the length of the shorter font type.

        Parameters
        ----------
        font_type1 : str
            font type 1
        font_type2 : str
            font type 2
        prefix_length_ratio : float
            minimum ratio of the common prefix length to the length of the shorter font type

        Returns
        -------
        bool
            True if the two font types are similar, False otherwise.
        """

        # 如果 font_type_1 是列表，取其第一个元素，若列表为空则设为空字符串
        if isinstance(font_type_1, list):
            font_type_1 = font_type_1[0] if font_type_1 else ""
        # 如果 font_type_2 是列表，取其第一个元素，若列表为空则设为空字符串
        if isinstance(font_type_2, list):
            font_type_2 = font_type_2[0] if font_type_2 else ""

        # 如果两个字体类型完全相同，则返回 True
        if font_type_1 == font_type_2:
            return True

        # 找到两个字体类型的公共前缀长度
        common_prefix_length = len(os.path.commonprefix([font_type_1, font_type_2]))

        # 根据最小比例计算最小公共前缀长度
        min_prefix_length = int(min(len(font_type_1), len(font_type_2)) * prefix_length_ratio)

        # 返回是否公共前缀长度大于等于最小前缀长度
        return common_prefix_length >= min_prefix_length
    # 定义一个私有方法，用于比较两个块的字体是否相同
    def __is_same_block_font(self, block_1, block_2):
        """
        该函数比较 block1 和 block2 的字体

        参数
        ----------
        block1 : dict
            第一个块
        block2 : dict
            第二个块

        返回值
        -------
        is_same : bool
            如果 block1 和 block2 具有相同的字体，则返回 True，否则返回 False
        """
        # 安全地获取 block_1 的字体类型
        block_1_font_type = safe_get(block_1, "block_font_type", "")
        # 安全地获取 block_1 的字体大小
        block_1_font_size = safe_get(block_1, "block_font_size", 0)
        # 安全地获取 block_1 的平均字符宽度
        block_1_avg_char_width = safe_get(block_1, "avg_char_width", 0)

        # 安全地获取 block_2 的字体类型
        block_2_font_type = safe_get(block_2, "block_font_type", "")
        # 安全地获取 block_2 的字体大小
        block_2_font_size = safe_get(block_2, "block_font_size", 0)
        # 安全地获取 block_2 的平均字符宽度
        block_2_avg_char_width = safe_get(block_2, "avg_char_width", 0)

        # 检查 block_1 的字体大小是否为列表，如果是，取其第一个元素
        if isinstance(block_1_font_size, list):
            block_1_font_size = block_1_font_size[0] if block_1_font_size else 0
        # 检查 block_2 的字体大小是否为列表，如果是，取其第一个元素
        if isinstance(block_2_font_size, list):
            block_2_font_size = block_2_font_size[0] if block_2_font_size else 0

        # 安全地获取 block_1 的文本内容
        block_1_text = safe_get(block_1, "text", "")
        # 安全地获取 block_2 的文本内容
        block_2_text = safe_get(block_2, "text", "")

        # 如果 block_1 或 block_2 的平均字符宽度为 0，则返回 False
        if block_1_avg_char_width == 0 or block_2_avg_char_width == 0:
            return False

        # 如果 block_1 或 block_2 的文本为空，则返回 False
        if not block_1_text or not block_2_text:
            return False
        else:
            # 计算 block_2 文本长度与 block_1 文本长度的比率
            text_len_ratio = len(block_2_text) / len(block_1_text)
            # 根据文本长度比率决定平均字符宽度的条件
            if text_len_ratio < 0.2:
                avg_char_width_condition = (
                    abs(block_1_avg_char_width - block_2_avg_char_width) / min(block_1_avg_char_width, block_2_avg_char_width)
                    < 0.5
                )
            else:
                avg_char_width_condition = (
                    abs(block_1_avg_char_width - block_2_avg_char_width) / min(block_1_avg_char_width, block_2_avg_char_width)
                    < 0.2
                )

        # 检查 block_1 和 block_2 的字体大小差异是否小于 1
        block_font_size_condition = abs(block_1_font_size - block_2_font_size) < 1

        # 返回字体类型相似、平均字符宽度条件和字体大小条件同时满足的结果
        return (
            self.__is_similar_font_type(block_1_font_type, block_2_font_type)
            and avg_char_width_condition
            and block_font_size_condition
        )

    # 定义一个私有方法，用于判断字符是否为字母字符
    def _is_alphabet_char(self, char):
        # 检查字符是否在大写字母或小写字母的范围内
        if (char >= "\u0041" and char <= "\u005a") or (char >= "\u0061" and char <= "\u007a"):
            return True
        else:
            return False

    # 定义一个私有方法，用于判断字符是否为汉字字符
    def _is_chinese_char(self, char):
        # 检查字符是否在汉字的范围内
        if char >= "\u4e00" and char <= "\u9fa5":
            return True
        else:
            return False

    # 定义一个私有方法，用于判断字符是否为其他字母字符
    def _is_other_letter_char(self, char):
        try:
            # 获取字符的 Unicode 类别
            cat = unicodedata.category(char)
            # 检查字符是否为大写字母或小写字母，同时排除字母和汉字
            if cat == "Lu" or cat == "Ll":
                return not self._is_alphabet_char(char) and not self._is_chinese_char(char)
        except TypeError:
            # 捕获类型错误并输出错误信息
            print("The input to the function must be a single character.")
        return False
    # 检查给定字符串是否表示一个有效的年份
    def _is_year(self, s: str):
        # 尝试将字符串转换为整数
        try:
            number = int(s)
            # 检查该年份是否在 1900 到 2099 之间
            return 1900 <= number <= 2099
        # 如果转换失败，返回 False
        except ValueError:
            return False
    
    # 检查文本的开头是否是指定的括号
    def _match_brackets(self, text):
        # 定义正则表达式，匹配特定的括号字符
        pattern = r"^[\(\)\]（）】{}｛｝>＞〕〙\"\'“”‘’]"
        # 返回文本是否匹配该模式
        return bool(re.match(pattern, text))
    
    # 比较两个段落的字体一致性
    def _is_para_font_consistent(self, para_1, para_2):
        """
        此函数比较 para1 和 para2 的字体
    
        参数
        ----------
        para1 : dict
            para1
        para2 : dict
            para2
    
        返回
        -------
        is_same : bool
            如果 para1 和 para2 的字体相同，则返回 True，否则返回 False
        """
        # 如果任一段落为 None，返回 False
        if para_1 is None or para_2 is None:
            return False
    
        # 获取第一个段落的字体类型、大小和颜色
        para_1_font_type = safe_get(para_1, "para_font_type", "")
        para_1_font_size = safe_get(para_1, "para_font_size", 0)
        para_1_font_color = safe_get(para_1, "para_font_color", "")
    
        # 获取第二个段落的字体类型、大小和颜色
        para_2_font_type = safe_get(para_2, "para_font_type", "")
        para_2_font_size = safe_get(para_2, "para_font_size", 0)
        para_2_font_color = safe_get(para_2, "para_font_color", "")
    
        # 如果 para_1 的字体类型是列表，则获取最常见的字体类型
        if isinstance(para_1_font_type, list):  # get the most common font type
            para_1_font_type = max(set(para_1_font_type), key=para_1_font_type.count)
        # 如果 para_2 的字体类型是列表，则获取最常见的字体类型
        if isinstance(para_2_font_type, list):
            para_2_font_type = max(set(para_2_font_type), key=para_2_font_type.count)
        # 如果 para_1 的字体大小是列表，则计算平均字体大小
        if isinstance(para_1_font_size, list):  # compute average font type
            para_1_font_size = sum(para_1_font_size) / len(para_1_font_size)
        # 如果 para_2 的字体大小是列表，则计算平均字体大小
        if isinstance(para_2_font_size, list):  # compute average font type
            para_2_font_size = sum(para_2_font_size) / len(para_2_font_size)
    
        # 检查字体类型是否相似且字体大小差异小于 1.5
        return (
            self.__is_similar_font_type(para_1_font_type, para_2_font_type)
            and abs(para_1_font_size - para_2_font_size) < 1.5
            # and para_font_color1 == para_font_color2
        )
    
    # 检查两个块是否来自同一个块
    def _is_block_consistent(self, block_1, block_2):
        """
        此函数确定 block1 和 block2 是否最初来自同一个块
    
        参数
        ----------
        block1 : dict
            block1s
        block2 : dict
            block2
    
        返回
        -------
        is_same : bool
            如果 block1 和 block2 来自同一个块，则返回 True，否则返回 False
        """
        # 调用方法检查两个块的字体是否相同
        return self.__is_same_block_font(block_1, block_2)
    # 检查两个段落是否来自同一段落
    def _is_para_continued(self, para_1, para_2):
        """
        此函数确定 para1 和 para2 是否最初来自同一段落
    
        参数
        ----------
        para1 : dict
            para1
        para2 : dict
            para2
    
        返回值
        -------
        is_same : bool
            如果 para1 和 para2 来自同一段落则为 True，否则为 False
        """
        # 检查段落的字体是否一致
        is_para_font_consistent = self._is_para_font_consistent(para_1, para_2)
        # 检查段落的标点符号是否一致
        is_para_puncs_consistent = self._is_para_puncs_consistent(para_1, para_2)
    
        # 返回字体和标点符号一致性的逻辑与结果
        return is_para_font_consistent and is_para_puncs_consistent
    
    # 检查两个块的边界是否一致
    def _are_boundaries_of_block_consistent(self, block_1, block_2):
        """
        此函数检查 block1 和 block2 的边界是否一致
    
        参数
        ----------
        block1 : dict
            block1
    
        block2 : dict
            block2
    
        返回值
        -------
        is_consistent : bool
            如果 block1 和 block2 的边界一致则为 True，否则为 False
        """
    
        # 获取 block_1 的最后一行
        last_line_of_block_1 = block_1["lines"][-1]
        # 获取 block_2 的第一行
        first_line_of_block_2 = block_2["lines"][0]
    
        # 获取最后一行的字符跨度信息
        spans_of_last_line_of_block_1 = last_line_of_block_1["spans"]
        # 获取第一行的字符跨度信息
        spans_of_first_line_of_block_2 = first_line_of_block_2["spans"]
    
        # 获取最后一行的字体类型并转为小写
        font_type_of_last_line_of_block_1 = spans_of_last_line_of_block_1[0]["font"].lower()
        # 获取最后一行的字体大小
        font_size_of_last_line_of_block_1 = spans_of_last_line_of_block_1[0]["size"]
        # 获取最后一行的字体颜色
        font_color_of_last_line_of_block_1 = spans_of_last_line_of_block_1[0]["color"]
        # 获取最后一行的字体标志
        font_flags_of_last_line_of_block_1 = spans_of_last_line_of_block_1[0]["flags"]
    
        # 获取第一行的字体类型并转为小写
        font_type_of_first_line_of_block_2 = spans_of_first_line_of_block_2[0]["font"].lower()
        # 获取第一行的字体大小
        font_size_of_first_line_of_block_2 = spans_of_first_line_of_block_2[0]["size"]
        # 获取第一行的字体颜色
        font_color_of_first_line_of_block_2 = spans_of_first_line_of_block_2[0]["color"]
        # 获取第一行的字体标志
        font_flags_of_first_line_of_block_2 = spans_of_first_line_of_block_2[0]["flags"]
    
        # 返回边界一致性的判断结果
        return (
            self.__is_similar_font_type(font_type_of_last_line_of_block_1, font_type_of_first_line_of_block_2)
            and abs(font_size_of_last_line_of_block_1 - font_size_of_first_line_of_block_2) < 1
            # and font_color_of_last_line_of_block1 == font_color_of_first_line_of_block2
            and font_flags_of_last_line_of_block_1 == font_flags_of_first_line_of_block_2
        )
    # 定义一个方法，判断下一个段落是否应该与当前段落合并
    def should_merge_next_para(self, curr_para, next_para):
        """
        此函数检查下一个段落是否应合并到当前段落。
    
        参数
        ----------
        curr_para : dict
            当前段落。
        next_para : dict
            下一个段落。
    
        返回值
        -------
        bool
            如果下一个段落应该合并到当前段落，则返回 True，否则返回 False。
        """
        # 调用私有方法检查当前段落与下一个段落是否属于连续
        if self._is_para_continued(curr_para, next_para):
            # 如果是连续，则返回 True
            return True
        else:
            # 如果不是连续，则返回 False
            return False
    
    # 定义一个方法，通过 ID 查找段落块
    def find_block_by_id(self, para_blocks, block_id):
        """
        此函数通过 ID 查找块。
    
        参数
        ----------
        para_blocks : list
            块的列表。
        block_id : int
            要查找的块的 ID。
    
        返回值
        -------
        block : dict
            具有给定 ID 的块。
        """
        # 遍历段落块列表，并获取每个块的索引和内容
        for blk_idx, block in enumerate(para_blocks):
            # 检查当前块的 ID 是否与要查找的 ID 匹配
            if block.get("block_id") == block_id:
                # 如果匹配，返回该块
                return block
        # 如果没有找到，返回 None
        return None
# 定义一个用于在 PDF 文件上绘制注释的类
class DrawAnnos:
    """
    该类在 PDF 文件上绘制注释

    ----------------------------------------
                颜色代码
    ----------------------------------------
        红色: (1, 0, 0)
        绿色: (0, 1, 0)
        蓝色: (0, 0, 1)
        黄色: (1, 1, 0) - 红色和绿色的混合
        青色: (0, 1, 1) - 绿色和蓝色的混合
        品红: (1, 0, 1) - 红色和蓝色的混合
        白色: (1, 1, 1) - 红、绿、蓝全强度
        黑色: (0, 0, 0) - 没有任何颜色成分
        灰色: (0.5, 0.5, 0.5) - 红、绿、蓝等强度的中等强度
        橙色: (1, 0.65, 0) - 红色最大强度，绿色中等强度，没有蓝色成分
    """

    # 初始化方法
    def __init__(self) -> None:
        pass

    # 判断给定列表是否为嵌套列表
    def __is_nested_list(self, lst):
        """
        如果给定列表是任意深度的嵌套列表，返回 True。
        """
        # 如果 lst 是列表类型
        if isinstance(lst, list):
            # 检查是否存在嵌套列表或列表中的元素是列表
            return any(self.__is_nested_list(i) for i in lst) or any(isinstance(i, list) for i in lst)
        return False  # 如果不是列表，则返回 False

    # 验证矩形框的有效性
    def __valid_rect(self, bbox):
        # 确保矩形不是空的或无效的
        if isinstance(bbox[0], list):
            return False  # 如果是嵌套列表，则不能是有效矩形
        else:
            # 确保左下角坐标小于右上角坐标
            return bbox[0] < bbox[2] and bbox[1] < bbox[3]

    # 绘制嵌套框
    def __draw_nested_boxes(self, page, nested_bbox, color=(0, 1, 1)):
        """
        该函数绘制嵌套框

        参数
        ----------
        page : fitz.Page
            页面对象
        nested_bbox : list
            嵌套的边界框
        color : tuple
            颜色，默认为 (0, 1, 1)    # 使用青色绘制组合段落
        """
        # 如果嵌套边界框是嵌套列表
        if self.__is_nested_list(nested_bbox):
            # 遍历嵌套边界框
            for bbox in nested_bbox:
                # 递归调用该函数绘制嵌套框
                self.__draw_nested_boxes(page, bbox, color)
        # 如果是有效矩形
        elif self.__valid_rect(nested_bbox):
            para_rect = fitz.Rect(nested_bbox)  # 创建矩形对象
            para_anno = page.add_rect_annot(para_rect)  # 在页面上添加矩形注释
            para_anno.set_colors(stroke=color)  # 设置边框颜色为青色
            para_anno.set_border(width=1)  # 设置边框宽度
            para_anno.update()  # 更新注释

# 定义一个用于处理段落的处理管道类
class ParaProcessPipeline:
    # 初始化方法
    def __init__(self) -> None:
        pass

"""
运行此脚本以测试功能，命令为：

python detect_para.py [pdf_path] [output_pdf_path]

参数：
- pdf_path: PDF 文件的路径
- output_pdf_path: 输出 PDF 文件的路径
"""

# 检查是否为主模块
if __name__ == "__main__":
    # 定义默认 PDF 文件路径，根据操作系统选择路径格式
    DEFAULT_PDF_PATH = (
        "app/pdf_toolbox/tests/assets/paper/paper.pdf" if os.name != "nt" else "app\\pdf_toolbox\\tests\\assets\\paper\\paper.pdf"
    )
    # 从命令行参数获取输入 PDF 路径，若没有则使用默认路径
    input_pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    # 从命令行参数获取输出 PDF 路径，若没有则根据输入路径生成输出路径
    output_pdf_path = sys.argv[2] if len(sys.argv) > 2 else input_pdf_path.split(".")[0] + "_recogPara.pdf"
    # 根据命令行参数设置输出 JSON 文件路径，如果参数不足则默认以输入 PDF 文件名生成
        output_json_path = sys.argv[3] if len(sys.argv) > 3 else input_pdf_path.split(".")[0] + "_recogPara.json"
    
        # 导入文件状态模块
        import stat
    
        # 如果输出 PDF 文件已存在，则删除它
        if os.path.exists(output_pdf_path):
            # 设置文件为可写，防止删除时的权限问题
            os.chmod(output_pdf_path, stat.S_IWRITE)
            # 删除输出 PDF 文件
            os.remove(output_pdf_path)
    
        # 打开输入 PDF 文档
        input_pdf_doc = open_pdf(input_pdf_path)
    
        # 初始化段落后处理管道
        paraProcessPipeline = ParaProcessPipeline()
    
        # 创建空字典用于存储 PDF 数据
        pdf_dic = {}
    
        # 初始化块终止处理器
        blockInnerParasProcessor = BlockTerminationProcessor()
    
        """
        构建 PDF 字典。
        """
    
        # 遍历输入 PDF 文档的每一页
        for page_id, page in enumerate(input_pdf_doc):  # type: ignore
            # 处理当前页面（注释掉的调试输出）
            # print(f"Processing page {page_id}")
            # print(f"page: {page}")
            # 获取页面中的文本块
            raw_blocks = page.get_text("dict")["blocks"]
    
            # 保存符合条件的文本块到“预处理块”列表
            preproc_blocks = []
            for block in raw_blocks:
                # 仅保留类型为0的块（一般为文本块）
                if block["type"] == 0:
                    preproc_blocks.append(block)
    
            # 初始化布局边界框列表
            layout_bboxes = []
    
            # 构建当前页面的 PDF 字典
            page_dict = {
                "para_blocks": None,
                "preproc_blocks": preproc_blocks,
                "images": None,
                "tables": None,
                "interline_equations": None,
                "inline_equations": None,
                "layout_bboxes": None,
                "pymu_raw_blocks": None,
                "global_statistic": None,
                "droped_text_block": None,
                "droped_image_block": None,
                "droped_table_block": None,
                "image_backup": None,
                "table_backup": None,
            }
    
            # 将当前页面的字典添加到 PDF 字典中
            pdf_dic[f"page_{page_id}"] = page_dict
    
        # 打印生成的 PDF 字典（注释掉的调试输出）
        # print(f"pdf_dic: {pdf_dic}")
    
        # 将 PDF 字典写入指定的 JSON 文件
        with open(output_json_path, "w", encoding="utf-8") as f:
            # 将 PDF 字典序列化为 JSON 格式并写入文件
            json.dump(pdf_dic, f, ensure_ascii=False, indent=4)
    
        # 对生成的 JSON 文件进行段落处理，并返回处理结果
        pdf_dic = paraProcessPipeline.para_process_pipeline(output_json_path, input_pdf_doc, output_pdf_path)
```