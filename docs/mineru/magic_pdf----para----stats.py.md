# `.\MinerU\magic_pdf\para\stats.py`

```
# 从 collections 模块导入 Counter 类，用于计数
from collections import Counter
# 导入 numpy 库，用于数值计算
import numpy as np

# 从 magic_pdf.para.commons 模块导入所有内容
from magic_pdf.para.commons import *


# 检查 Python 版本，如果是 3.x 及以上
if sys.version_info[0] >= 3:
    # 配置标准输出的编码为 UTF-8，确保中文等字符正确显示
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


# 定义一个计算块统计信息的类
class BlockStatisticsCalculator:
    # 初始化方法
    def __init__(self) -> None:
        # 初始化时不进行任何操作
        pass

    # 私有方法，用于创建新块
    def __make_new_block(self, input_block):
        # 初始化一个字典以存储新块信息
        new_block = {}

        # 从输入块中提取原始行数据
        raw_lines = input_block["lines"]
        # 计算原始行的统计信息
        stats = self.__calc_stats_of_new_lines(raw_lines)

        # 从输入块中提取块的 ID
        block_id = input_block["block_id"]
        # 从输入块中提取块的边界框信息
        block_bbox = input_block["bbox"]
        # 从输入块中提取块的文本内容
        block_text = input_block["text"]
        # 保存原始行信息
        block_lines = raw_lines
        # 从统计信息中提取左边界平均值
        block_avg_left_boundary = stats[0]
        # 从统计信息中提取右边界平均值
        block_avg_right_boundary = stats[1]
        # 从统计信息中提取字符宽度平均值
        block_avg_char_width = stats[2]
        # 从统计信息中提取字符高度平均值
        block_avg_char_height = stats[3]
        # 从统计信息中提取字体类型
        block_font_type = stats[4]
        # 从统计信息中提取字体大小
        block_font_size = stats[5]
        # 从统计信息中提取文本方向
        block_direction = stats[6]
        # 从统计信息中提取中位数字体大小
        block_median_font_size = stats[7]

        # 将块的各项信息添加到新块字典中
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

        # 返回新创建的块
        return new_block

    # 批处理块的方法
    def batch_process_blocks(self, pdf_dic):
        """
        该函数用于批量处理块。

        参数
        ----------
        self : object
            类的实例。
        ----------
        blocks : list
            输入块是原始块的列表。架构可以参考 "preproc_blocks" 键的值，示例文件是 app/pdf_toolbox/tests/preproc_2_parasplit_example.json

        返回
        -------
        result_dict : dict
            结果字典
        """

        # 遍历 PDF 字典中的每一页
        for page_id, blocks in pdf_dic.items():
            # 检查页 ID 是否以 "page_" 开头
            if page_id.startswith("page_"):
                # 初始化一个空列表用于存储段落块
                para_blocks = []
                # 检查块中是否包含 "para_blocks" 键
                if "para_blocks" in blocks.keys():
                    # 提取输入块
                    input_blocks = blocks["para_blocks"]
                    # 遍历输入块
                    for input_block in input_blocks:
                        # 创建新块
                        new_block = self.__make_new_block(input_block)
                        # 将新块添加到段落块列表中
                        para_blocks.append(new_block)

                # 将处理后的段落块列表添加到原块中
                blocks["para_blocks"] = para_blocks

        # 返回处理后的 PDF 字典
        return pdf_dic


# 定义一个文档统计信息计算器类
class DocStatisticsCalculator:
    # 初始化方法
    def __init__(self) -> None:
        # 初始化时不进行任何操作
        pass
```