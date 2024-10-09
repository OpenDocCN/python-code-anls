# `.\MinerU\magic_pdf\para\denoise.py`

```
# 导入数学库
import math

# 从 collections 模块导入 defaultdict，用于默认字典
from collections import defaultdict
# 从 magic_pdf.para.commons 导入所有内容
from magic_pdf.para.commons import *

# 检查 Python 版本是否为 3 或更高
if sys.version_info[0] >= 3:
    # 如果是，重新配置标准输出的编码为 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


# 定义 HeaderFooterProcessor 类
class HeaderFooterProcessor:
    # 初始化方法
    def __init__(self) -> None:
        pass

    # 获取最常见的边界框方法
    def get_most_common_bboxes(self, bboxes, page_height, position="top", threshold=0.25, num_bboxes=3, min_frequency=2):
        """
        此函数从 bboxes 获取最常见的边界框

        参数
        ----------
        bboxes : list
            边界框列表
        page_height : float
            页面的高度
        position : str, optional
            "top" 或 "bottom"，默认为 "top"
        threshold : float, optional
            阈值，默认为 0.25
        num_bboxes : int, optional
            返回的边界框数量，默认为 3
        min_frequency : int, optional
            边界框的最小频率，默认为 2

        返回
        -------
        common_bboxes : list
            常见的边界框
        """
        # 根据位置过滤边界框
        if position == "top":
            # 选择顶部位置的边界框
            filtered_bboxes = [bbox for bbox in bboxes if bbox[1] < page_height * threshold]
        else:
            # 选择底部位置的边界框
            filtered_bboxes = [bbox for bbox in bboxes if bbox[3] > page_height * (1 - threshold)]

        # 统计最常见的边界框
        bbox_count = defaultdict(int)
        for bbox in filtered_bboxes:
            # 更新边界框的计数
            bbox_count[tuple(bbox)] += 1

        # 获取频率超过 min_frequency 的最常出现的边界框
        common_bboxes = [
            bbox for bbox, count in sorted(bbox_count.items(), key=lambda item: item[1], reverse=True) if count >= min_frequency
        ][:num_bboxes]
        # 返回常见的边界框
        return common_bboxes

# 定义 NonHorizontalTextProcessor 类
class NonHorizontalTextProcessor:
    # 初始化方法
    def __init__(self) -> None:
        pass

# 定义 NoiseRemover 类
class NoiseRemover:
    # 初始化方法
    def __init__(self) -> None:
        pass
    # 跳过数据噪声的函数，包括重叠块、头部、尾部、水印、垂直边注和标题
    def skip_data_noises(self, result_dict):
        # 初始化一个空字典，用于存储过滤后的结果
        filtered_result_dict = {}
        # 遍历结果字典中的每一页及其对应的块
        for page_id, blocks in result_dict.items():
            # 只处理以 "page_" 开头的页面 ID
            if page_id.startswith("page_"):
                # 初始化一个空字典，用于存储过滤后的块
                filtered_blocks = {}
                # 遍历当前页面中的每一个块
                for block_id, block in blocks.items():
                    # 只处理以 "block_" 开头的块 ID
                    if block_id.startswith("block_"):
                        # 检查块是否包含任何噪声标志
                        if any(
                            block.get(key, 0)
                            for key in [
                                "is_overlap",  # 是否重叠
                                "is_header",   # 是否为头部
                                "is_footer",   # 是否为尾部
                                "is_watermark",# 是否为水印
                                "is_vertical_margin_note", # 是否为垂直边注
                                "is_block_title", # 是否为块标题
                            ]
                        ):
                            continue  # 如果是噪声，则跳过该块
                        # 将过滤后的块添加到字典中
                        filtered_blocks[block_id] = block
                # 如果过滤后的块不为空，则将其添加到结果字典中
                if filtered_blocks:
                    filtered_result_dict[page_id] = filtered_blocks
    
        # 返回过滤后的结果字典
        return filtered_result_dict
```