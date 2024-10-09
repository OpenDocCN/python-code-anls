# `.\MinerU\magic_pdf\para\block_termination_processor.py`

```
# 从 magic_pdf.para.commons 模块导入所有内容
from magic_pdf.para.commons import *


# 检查 Python 版本是否为 3 及以上
if sys.version_info[0] >= 3:
    # 重新配置标准输出的编码为 UTF-8，忽略类型检查
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore



# 定义 BlockTerminationProcessor 类
class BlockTerminationProcessor:
    # 初始化方法，构造函数
    def __init__(self) -> None:
        # 构造函数中没有具体实现，使用默认
        pass

    # 定义检查当前行与邻居行一致性的方法
    def _is_consistent_lines(
        self,
        curr_line,  # 当前行
        prev_line,  # 前一行
        next_line,  # 下一行
        consistent_direction,  # 一致性方向：0 表示前一行，1 表示下一行，2 表示两者
    ):
        """
        此函数检查当前行是否与邻居行一致

        参数
        ----------
        curr_line : dict
            当前行
        prev_line : dict
            前一行
        next_line : dict
            下一行
        consistent_direction : int
            0 表示前一行，1 表示下一行，2 表示两者

        返回
        -------
        bool
            如果当前行与邻居行一致则返回 True，否则返回 False。
        """

        # 获取当前行的字体大小
        curr_line_font_size = curr_line["spans"][0]["size"]
        # 获取当前行的字体类型并转换为小写
        curr_line_font_type = curr_line["spans"][0]["font"].lower()

        # 检查与前一行一致性
        if consistent_direction == 0:
            # 如果前一行存在
            if prev_line:
                # 获取前一行的字体大小
                prev_line_font_size = prev_line["spans"][0]["size"]
                # 获取前一行的字体类型并转换为小写
                prev_line_font_type = prev_line["spans"][0]["font"].lower()
                # 返回当前行与前一行字体大小和类型是否一致
                return curr_line_font_size == prev_line_font_size and curr_line_font_type == prev_line_font_type
            else:
                # 如果前一行不存在，返回 False
                return False

        # 检查与下一行一致性
        elif consistent_direction == 1:
            # 如果下一行存在
            if next_line:
                # 获取下一行的字体大小
                next_line_font_size = next_line["spans"][0]["size"]
                # 获取下一行的字体类型并转换为小写
                next_line_font_type = next_line["spans"][0]["font"].lower()
                # 返回当前行与下一行字体大小和类型是否一致
                return curr_line_font_size == next_line_font_size and curr_line_font_type == next_line_font_type
            else:
                # 如果下一行不存在，返回 False
                return False

        # 检查与前后两行一致性
        elif consistent_direction == 2:
            # 如果前一行和下一行都存在
            if prev_line and next_line:
                # 获取前一行的字体大小
                prev_line_font_size = prev_line["spans"][0]["size"]
                # 获取前一行的字体类型并转换为小写
                prev_line_font_type = prev_line["spans"][0]["font"].lower()
                # 获取下一行的字体大小
                next_line_font_size = next_line["spans"][0]["size"]
                # 获取下一行的字体类型并转换为小写
                next_line_font_type = next_line["spans"][0]["font"].lower()
                # 检查当前行是否同时与前一行和下一行一致
                return (curr_line_font_size == prev_line_font_size and curr_line_font_type == prev_line_font_type) and (
                    curr_line_font_size == next_line_font_size and curr_line_font_type == next_line_font_type
                )
            else:
                # 如果前一行或下一行不存在，返回 False
                return False

        # 如果一致性方向不合法，返回 False
        else:
            return False
    # 定义一个检查当前行是否为规则行的函数
    def _is_regular_line(self, curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, X0, X1, avg_line_height):
        """
        该函数检查该行是否为规则行
    
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
        X0 : float
            表示页面左边界的 x0 值的中位数
        X1 : float
            表示页面右边界的 x1 值的中位数
        avg_line_height : float
            行高的平均值
    
        返回
        -------
        bool
            如果该行为规则行则返回 True，否则返回 False。
        """
        # 设置横向和纵向的比例阈值
        horizontal_ratio = 0.5
        vertical_ratio = 0.5
        # 计算横向和纵向的阈值
        horizontal_thres = horizontal_ratio * avg_char_width
        vertical_thres = vertical_ratio * avg_line_height
    
        # 解包当前行的边界框坐标
        x0, y0, x1, y1 = curr_line_bbox
    
        # 检查当前行的左边界是否接近页面左边界
        x0_near_X0 = abs(x0 - X0) < horizontal_thres
        # 检查当前行的右边界是否接近页面右边界
        x1_near_X1 = abs(x1 - X1) < horizontal_thres
    
        # 检查前一行是否是段落的结束
        prev_line_is_end_of_para = prev_line_bbox and (abs(prev_line_bbox[2] - X1) > avg_char_width)
    
        # 初始化上方有足够间距的标志
        sufficient_spacing_above = False
        if prev_line_bbox:
            # 计算当前行与前一行之间的垂直间距
            vertical_spacing_above = y1 - prev_line_bbox[3]
            # 判断上方是否有足够的间距
            sufficient_spacing_above = vertical_spacing_above > vertical_thres
    
        # 初始化下方有足够间距的标志
        sufficient_spacing_below = False
        if next_line_bbox:
            # 计算当前行与下一行之间的垂直间距
            vertical_spacing_below = next_line_bbox[1] - y0
            # 判断下方是否有足够的间距
            sufficient_spacing_below = vertical_spacing_below > vertical_thres
    
        # 返回判断结果：是否为规则行
        return (
            (sufficient_spacing_above or sufficient_spacing_below)  # 检查上下间距
            or (not x0_near_X0 and not x1_near_X1)  # 检查边界接近性
            or prev_line_is_end_of_para  # 检查前一行是否为段落结束
        )
    # 检查当前行是否可能是段落的结束
        def _is_possible_end_of_para(self, curr_line, next_line, X0, X1, avg_char_width):
            """
            此函数检查该行是否可能是段落结束
    
            参数
            ----------
            curr_line : dict
                当前行
            next_line : dict
                下一行
            X0 : float
                x0 值的中位数，表示页面的左平均边界
            X1 : float
                x1 值的中位数，表示页面的右平均边界
            avg_char_width : float
                字符宽度的平均值
    
            返回
            -------
            bool
                如果该行可能是段落结束，则返回 True，否则返回 False。
            """
    
            end_confidence = 0.5  # 段落结束的初始置信度
            decision_path = []  # 记录决策路径
    
            curr_line_bbox = curr_line["bbox"]  # 获取当前行的边界框
            next_line_bbox = next_line["bbox"] if next_line else None  # 获取下一行的边界框，若不存在则为 None
    
            left_horizontal_ratio = 0.5  # 左侧水平比例
            right_horizontal_ratio = 0.5  # 右侧水平比例
    
            x0, _, x1, y1 = curr_line_bbox  # 解包当前行的边界框
            next_x0, next_y0, _, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)  # 解包下一行的边界框
    
            # 检查当前行的 x0 是否接近左边界 X0
            x0_near_X0 = abs(x0 - X0) < left_horizontal_ratio * avg_char_width  
            if x0_near_X0:  # 如果接近
                end_confidence += 0.1  # 增加置信度
                decision_path.append("x0_near_X0")  # 记录决策路径
    
            # 检查当前行的 x1 是否小于右边界 X1
            x1_smaller_than_X1 = x1 < X1 - right_horizontal_ratio * avg_char_width  
            if x1_smaller_than_X1:  # 如果小于
                end_confidence += 0.1  # 增加置信度
                decision_path.append("x1_smaller_than_X1")  # 记录决策路径
    
            # 检查下一行是否是段落的开始
            next_line_is_start_of_para = (
                next_line_bbox
                and (next_x0 > X0 + left_horizontal_ratio * avg_char_width)  # 下一行的 x0 是否在左边界右侧
                and (not is_line_left_aligned_from_neighbors(curr_line_bbox, None, next_line_bbox, avg_char_width, direction=1))  # 检查是否未与邻居左对齐
            )
            if next_line_is_start_of_para:  # 如果是段落开始
                end_confidence += 0.2  # 增加置信度
                decision_path.append("next_line_is_start_of_para")  # 记录决策路径
    
            # 检查当前行是否与邻居左对齐
            is_line_left_aligned_from_neighbors_bool = is_line_left_aligned_from_neighbors(
                curr_line_bbox, None, next_line_bbox, avg_char_width
            )
            if is_line_left_aligned_from_neighbors_bool:  # 如果左对齐
                end_confidence += 0.1  # 增加置信度
                decision_path.append("line_is_left_aligned_from_neighbors")  # 记录决策路径
    
            # 检查当前行是否与邻居右对齐
            is_line_right_aligned_from_neighbors_bool = is_line_right_aligned_from_neighbors(
                curr_line_bbox, None, next_line_bbox, avg_char_width
            )
            if not is_line_right_aligned_from_neighbors_bool:  # 如果未右对齐
                end_confidence += 0.1  # 增加置信度
                decision_path.append("line_is_not_right_aligned_from_neighbors")  # 记录决策路径
    
            # 检查当前行是否以标点符号结束，结合其他条件判断是否为段落结束
            is_end_of_para = end_with_punctuation(curr_line["text"]) and (
                (x0_near_X0 and x1_smaller_than_X1)  # 检查 x0 和 x1 条件
                or (is_line_left_aligned_from_neighbors_bool and not is_line_right_aligned_from_neighbors_bool)  # 或左对齐且不右对齐
            )
    
            return (is_end_of_para, end_confidence, decision_path)  # 返回是否为段落结束的布尔值、置信度和决策路径
    
        def _cut_paras_per_block(  # 定义函数切割每个块中的段落
            self,
            block,
    # 批量处理 PDF 字典中的页面块
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
                # 初始化段落块列表
                para_blocks = []
                # 检查页面中是否包含 "para_blocks" 键
                if "para_blocks" in page.keys():
                    # 获取输入块
                    input_blocks = page["para_blocks"]
                    # 遍历输入块
                    for input_block in input_blocks:
                        # 对每个输入块进行段落切割
                        new_block = self._cut_paras_per_block(input_block)
                        # 将新的块添加到段落块列表中
                        para_blocks.append(new_block)
                        # 更新段落计数器
                        num_paras += len(new_block["paras"])
    
                # 将处理后的段落块保存回页面字典
                page["para_blocks"] = para_blocks
    
        # 在 PDF 字典的统计信息中记录段落数量
        pdf_dict["statistics"]["num_paras"] = num_paras
        # 返回更新后的 PDF 字典
        return pdf_dict
```