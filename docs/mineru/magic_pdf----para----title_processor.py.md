# `.\MinerU\magic_pdf\para\title_processor.py`

```
# 导入操作系统相关的模块
import os
# 导入正则表达式模块
import re
# 导入 NumPy 库并命名为 np
import numpy as np

# 从 magic_pdf.libs.nlp_utils 模块导入 NLPModels 类
from magic_pdf.libs.nlp_utils import NLPModels

# 从 magic_pdf.para.commons 模块导入所有内容
from magic_pdf.para.commons import *

# 检查 Python 版本是否为 3 及以上
if sys.version_info[0] >= 3:
    # 将标准输出的编码设置为 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

# 定义 TitleProcessor 类
class TitleProcessor:
    # 初始化方法，接受可变数量的文档统计数据
    def __init__(self, *doc_statistics) -> None:
        # 如果传入的文档统计数据数量大于 0
        if len(doc_statistics) > 0:
            # 将第一个统计数据保存到实例变量中
            self.doc_statistics = doc_statistics[0]

        # 创建 NLPModels 实例
        self.nlp_model = NLPModels()
        # 设置最大标题级别为 3
        self.MAX_TITLE_LEVEL = 3
        # 定义用于匹配编号标题的正则表达式模式
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

    # 定义私有方法，检查当前行是否可能为标题
    def _is_potential_title(
        self,
        curr_line,                     # 当前行文本
        prev_line,                     # 前一行文本
        prev_line_is_title,           # 前一行是否为标题的布尔值
        next_line,                     # 下一行文本
        avg_char_width,                # 平均字符宽度
        avg_char_height,               # 平均字符高度
        median_font_size,              # 中位数字体大小
    # 定义一个检测块标题的私有方法，接受一个输入块作为参数
        def _detect_block_title(self, input_block):
            """
            使用函数 'is_potential_title' 检测每个段落块的标题。
            如果某行是标题，则该行的键 'is_title' 的值将设置为 True。
            """
    
            # 获取输入块中的原始行
            raw_lines = input_block["lines"]
    
            # 初始化前一行是否为标题的标志
            prev_line_is_title_flag = False
    
            # 遍历每一行及其索引
            for i, curr_line in enumerate(raw_lines):
                # 获取当前行的前一行（如果存在）和下一行（如果存在）
                prev_line = raw_lines[i - 1] if i > 0 else None
                next_line = raw_lines[i + 1] if i < len(raw_lines) - 1 else None
    
                # 获取块的平均字符宽度、高度和中位字体大小
                blk_avg_char_width = input_block["avg_char_width"]
                blk_avg_char_height = input_block["avg_char_height"]
                blk_media_font_size = input_block["median_font_size"]
    
                # 调用 '_is_potential_title' 方法检测当前行是否为标题，并获取可能的作者或组织列表
                is_title, is_author_or_org_list = self._is_potential_title(
                    curr_line,
                    prev_line,
                    prev_line_is_title_flag,
                    next_line,
                    blk_avg_char_width,
                    blk_avg_char_height,
                    blk_media_font_size,
                )
    
                # 如果当前行被判定为标题，设置其标题标志为 True
                if is_title:
                    curr_line["is_title"] = is_title
                    # 更新前一行是否为标题的标志
                    prev_line_is_title_flag = True
                else:
                    # 否则设置标题标志为 False
                    curr_line["is_title"] = False
                    prev_line_is_title_flag = False
    
                # 如果检测到可能的作者或组织列表，设置相关标志
                if is_author_or_org_list:
                    curr_line["is_author_or_org_list"] = is_author_or_org_list
                else:
                    # 否则设置为 False
                    curr_line["is_author_or_org_list"] = False
    
            # 返回修改后的输入块
            return input_block
    # 批量处理块以检测标题的函数
    def batch_process_blocks_detect_titles(self, pdf_dic):
        """
        此函数批量处理块以检测标题。
    
        参数
        ----------
        pdf_dict : dict
            结果字典
    
        返回
        -------
        pdf_dict : dict
            结果字典
        """
        # 初始化标题计数
        num_titles = 0
    
        # 遍历每个页面ID和对应的块
        for page_id, blocks in pdf_dic.items():
            # 检查页面ID是否以“page_”开头
            if page_id.startswith("page_"):
                # 初始化段落块列表
                para_blocks = []
                # 检查块中是否包含“para_blocks”键
                if "para_blocks" in blocks.keys():
                    # 获取段落块
                    para_blocks = blocks["para_blocks"]
    
                    # 存储所有单行块
                    all_single_line_blocks = []
                    # 遍历段落块
                    for block in para_blocks:
                        # 如果块只有一行，添加到单行块列表
                        if len(block["lines"]) == 1:
                            all_single_line_blocks.append(block)
    
                    # 存储新的段落块
                    new_para_blocks = []
                    # 检查是否并非所有块都是单行块
                    if not len(all_single_line_blocks) == len(para_blocks):  # 不是所有块都是单行块
                        # 遍历段落块
                        for para_block in para_blocks:
                            # 检测每个块的标题
                            new_block = self._detect_block_title(para_block)
                            # 添加到新的段落块列表
                            new_para_blocks.append(new_block)
                            # 累加标题数量
                            num_titles += sum([line.get("is_title", 0) for line in new_block["lines"]])
                    else:  # 所有块都是单行块
                        # 遍历段落块
                        for para_block in para_blocks:
                            # 直接添加到新的段落块列表
                            new_para_blocks.append(para_block)
                            # 累加标题数量
                            num_titles += sum([line.get("is_title", 0) for line in para_block["lines"]])
                    # 更新段落块为新的段落块
                    para_blocks = new_para_blocks
    
                # 更新块中的段落块
                blocks["para_blocks"] = para_blocks
    
                # 遍历每个段落块
                for para_block in para_blocks:
                    # 检查所有行是否都是标题
                    all_titles = all(safe_get(line, "is_title", False) for line in para_block["lines"])
                    # 计算段落文本长度
                    para_text_len = sum([len(line["text"]) for line in para_block["lines"]])
                    # 检查段落是否为标题（长度小于200且所有行都是标题）
                    if (
                        all_titles and para_text_len < 200
                    ):  # 段落总长度小于200，超过这个长度不应为标题
                        para_block["is_block_title"] = 1
                    else:
                        para_block["is_block_title"] = 0
    
                    # 检查是否所有行都是作者或机构列表
                    all_name_or_org_list_to_be_removed = all(
                        safe_get(line, "is_author_or_org_list", False) for line in para_block["lines"]
                    )
                    # 如果是且在第一页，则标记为作者或机构列表
                    if all_name_or_org_list_to_be_removed and page_id == "page_0":
                        para_block["is_block_an_author_or_org_list"] = 1
                    else:
                        para_block["is_block_an_author_or_org_list"] = 0
    
        # 在结果字典中更新标题数量
        pdf_dic["statistics"]["num_titles"] = num_titles
    
        # 返回更新后的结果字典
        return pdf_dic
    def __determine_size_based_level(self, title_blocks):
        """
        This function determines the title level based on the font size of the title.

        Parameters
        ----------
        title_blocks : list

        Returns
        -------
        title_blocks : list
        """

        # 从每个标题块中提取字体大小，生成一个 NumPy 数组
        font_sizes = np.array([safe_get(tb["block"], "block_font_size", 0) for tb in title_blocks])

        # 使用字体大小的均值和标准差来去除极端值
        mean_font_size = np.mean(font_sizes)  # 计算字体大小的均值
        std_font_size = np.std(font_sizes)    # 计算字体大小的标准差
        min_extreme_font_size = mean_font_size - std_font_size  # 计算最小极端字体大小
        max_extreme_font_size = mean_font_size + std_font_size  # 计算最大极端字体大小

        # 计算标题级别的阈值
        middle_font_sizes = font_sizes[(font_sizes > min_extreme_font_size) & (font_sizes < max_extreme_font_size)]  # 筛选中间范围的字体大小
        if middle_font_sizes.size > 0:  # 如果有中间字体大小
            middle_mean_font_size = np.mean(middle_font_sizes)  # 计算中间字体大小的均值
            level_threshold = middle_mean_font_size  # 将均值作为级别阈值
        else:
            level_threshold = mean_font_size  # 否则使用全体均值作为级别阈值

        for tb in title_blocks:  # 遍历每个标题块
            title_block = tb["block"]  # 获取标题块
            title_font_size = safe_get(title_block, "block_font_size", 0)  # 提取字体大小

            current_level = 1  # 初始化标题级别，最大的级别是 1

            # print(f"Before adjustment by font size, {current_level}")
            if title_font_size >= max_extreme_font_size:  # 如果字体大小大于最大极端值
                current_level = 1  # 设置级别为 1
            elif title_font_size <= min_extreme_font_size:  # 如果字体大小小于最小极端值
                current_level = 3  # 设置级别为 3
            elif float(title_font_size) >= float(level_threshold):  # 如果字体大小大于等于阈值
                current_level = 2  # 设置级别为 2
            else:
                current_level = 3  # 否则设置级别为 3
            # print(f"After adjustment by font size, {current_level}")

            title_block["block_title_level"] = current_level  # 将计算得到的级别赋值给标题块

        return title_blocks  # 返回修改后的标题块列表

    def batch_process_blocks_recog_title_level(self, pdf_dic):
        title_blocks = []  # 初始化标题块列表

        # 收集所有标题
        for page_id, blocks in pdf_dic.items():  # 遍历 PDF 字典中的每一页
            if page_id.startswith("page_"):  # 仅处理以 "page_" 开头的页码
                para_blocks = blocks.get("para_blocks", [])  # 获取段落块
                for block in para_blocks:  # 遍历每个段落块
                    if block.get("is_block_title"):  # 如果该块被标记为标题
                        title_obj = {"page_id": page_id, "block": block}  # 创建包含页码和块的对象
                        title_blocks.append(title_obj)  # 将对象添加到标题块列表

        # 确定标题级别
        if title_blocks:  # 如果有标题块
            # 根据字体大小确定标题级别
            title_blocks = self.__determine_size_based_level(title_blocks)  # 调用函数以确定级别

        return pdf_dic  # 返回原始 PDF 字典
```