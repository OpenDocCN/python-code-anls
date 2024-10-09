# `.\MinerU\magic_pdf\para\block_continuation_processor.py`

```
# 导入操作系统模块
import os
# 导入用于 Unicode 处理的模块
import unicodedata

# 从 magic_pdf.para.commons 导入所有内容
from magic_pdf.para.commons import *


# 检查 Python 版本是否为 3 或更高
if sys.version_info[0] >= 3:
    # 将标准输出的编码重新配置为 UTF-8，以支持中文字符，忽略类型检查
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


class BlockContinuationProcessor:
    """
    该类用于处理块以检测块的延续。
    """

    def __init__(self) -> None:
        # 初始化方法，当前不执行任何操作
        pass

    def __is_similar_font_type(self, font_type1, font_type2, prefix_length_ratio=0.3):
        """
        该函数检查两种字体类型是否相似。
        相似字体类型的定义：两种字体类型具有共同前缀，
        并且共同前缀的长度至少是较短字体类型长度的一定比例。

        参数
        ----------
        font_type1 : str
            字体类型 1
        font_type2 : str
            字体类型 2
        prefix_length_ratio : float
            共同前缀长度与较短字体类型长度的最小比例

        返回
        -------
        bool
            如果两种字体类型相似，则返回 True，否则返回 False。
        """

        # 如果 font_type1 是列表，取其第一个元素，若为空则赋值为空字符串
        if isinstance(font_type1, list):
            font_type1 = font_type1[0] if font_type1 else ""
        # 如果 font_type2 是列表，取其第一个元素，若为空则赋值为空字符串
        if isinstance(font_type2, list):
            font_type2 = font_type2[0] if font_type2 else ""

        # 如果两种字体类型完全相同，返回 True
        if font_type1 == font_type2:
            return True

        # 找到两种字体类型的共同前缀的长度
        common_prefix_length = len(os.path.commonprefix([font_type1, font_type2]))

        # 根据比例计算最小前缀长度
        min_prefix_length = int(min(len(font_type1), len(font_type2)) * prefix_length_ratio)

        # 返回共同前缀长度是否大于等于最小前缀长度
        return common_prefix_length >= min_prefix_length
    # 定义一个私有方法，用于比较两个文本块的字体
    def __is_same_block_font(self, block1, block2):
        """
        此函数比较 block1 和 block2 的字体

        参数
        ----------
        block1 : dict
            第一个文本块
        block2 : dict
            第二个文本块

        返回
        -------
        is_same : bool
            如果 block1 和 block2 的字体相同，则返回 True，否则返回 False
        """
        # 从 block1 中安全获取字体类型，默认为空字符串
        block_1_font_type = safe_get(block1, "block_font_type", "")
        # 从 block1 中安全获取字体大小，默认为 0
        block_1_font_size = safe_get(block1, "block_font_size", 0)
        # 从 block1 中安全获取平均字符宽度，默认为 0
        block_1_avg_char_width = safe_get(block1, "avg_char_width", 0)

        # 从 block2 中安全获取字体类型，默认为空字符串
        block_2_font_type = safe_get(block2, "block_font_type", "")
        # 从 block2 中安全获取字体大小，默认为 0
        block_2_font_size = safe_get(block2, "block_font_size", 0)
        # 从 block2 中安全获取平均字符宽度，默认为 0
        block_2_avg_char_width = safe_get(block2, "avg_char_width", 0)

        # 检查 block_1_font_size 是否为列表，如果是，则取第一个元素或默认 0
        if isinstance(block_1_font_size, list):
            block_1_font_size = block_1_font_size[0] if block_1_font_size else 0
        # 检查 block_2_font_size 是否为列表，如果是，则取第一个元素或默认 0
        if isinstance(block_2_font_size, list):
            block_2_font_size = block_2_font_size[0] if block_2_font_size else 0

        # 从 block1 中安全获取文本内容，默认为空字符串
        block_1_text = safe_get(block1, "text", "")
        # 从 block2 中安全获取文本内容，默认为空字符串
        block_2_text = safe_get(block2, "text", "")

        # 如果平均字符宽度为 0，则返回 False，表示字体不相同
        if block_1_avg_char_width == 0 or block_2_avg_char_width == 0:
            return False

        # 如果任一文本内容为空，则返回 False，表示字体不相同
        if not block_1_text or not block_2_text:
            return False
        else:
            # 计算 block_2_text 与 block_1_text 的长度比
            text_len_ratio = len(block_2_text) / len(block_1_text)
            # 根据文本长度比决定平均字符宽度的条件
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

        # 检查字体大小的差异是否小于 1
        block_font_size_condtion = abs(block_1_font_size - block_2_font_size) < 1

        # 返回字体类型、平均字符宽度和字体大小是否相似的结果
        return (
            self.__is_similar_font_type(block_1_font_type, block_2_font_type)
            and avg_char_width_condition
            and block_font_size_condtion
        )

    # 定义一个私有方法，用于判断字符是否为字母
    def _is_alphabet_char(self, char):
        # 检查字符是否在大写字母或小写字母范围内
        if (char >= "\u0041" and char <= "\u005a") or (char >= "\u0061" and char <= "\u007a"):
            return True
        else:
            return False

    # 定义一个私有方法，用于判断字符是否为汉字
    def _is_chinese_char(self, char):
        # 检查字符是否在汉字的 Unicode 范围内
        if char >= "\u4e00" and char <= "\u9fa5":
            return True
        else:
            return False

    # 定义一个私有方法，用于判断字符是否为其他字母字符
    def _is_other_letter_char(self, char):
        try:
            # 获取字符的 Unicode 类别
            cat = unicodedata.category(char)
            # 如果类别为大写字母或小写字母，且不是字母或汉字，则返回 True
            if cat == "Lu" or cat == "Ll":
                return not self._is_alphabet_char(char) and not self._is_chinese_char(char)
        except TypeError:
            # 捕获类型错误，说明输入的字符不合法
            print("The input to the function must be a single character.")
        # 默认返回 False
        return False
    # 检查给定字符串是否代表一个有效的年份（范围：1900到2099）
    def _is_year(self, s: str):
        try:
            # 尝试将字符串转换为整数
            number = int(s)
            # 检查该数字是否在1900到2099之间
            return 1900 <= number <= 2099
        except ValueError:
            # 如果转换失败，则返回False
            return False

    # 比较两个段落的字体是否一致
    def __is_para_font_consistent(self, para_1, para_2):
        """
        This function compares the font of para1 and para2

        Parameters
        ----------
        para1 : dict
            para1
        para2 : dict
            para2

        Returns
        -------
        is_same : bool
            True if para1 and para2 have the same font, else False
        """
        # 如果任一段落为None，则返回False
        if para_1 is None or para_2 is None:
            return False

        # 从第一个段落中安全获取字体类型、大小和颜色
        para_1_font_type = safe_get(para_1, "para_font_type", "")
        para_1_font_size = safe_get(para_1, "para_font_size", 0)
        para_1_font_color = safe_get(para_1, "para_font_color", "")

        # 从第二个段落中安全获取字体类型、大小和颜色
        para_2_font_type = safe_get(para_2, "para_font_type", "")
        para_2_font_size = safe_get(para_2, "para_font_size", 0)
        para_2_font_color = safe_get(para_2, "para_font_color", "")

        # 如果第一个段落的字体类型是列表，获取最常见的字体类型
        if isinstance(para_1_font_type, list):  # get the most common font type
            para_1_font_type = max(set(para_1_font_type), key=para_1_font_type.count)
        # 如果第二个段落的字体类型是列表，获取最常见的字体类型
        if isinstance(para_2_font_type, list):
            para_2_font_type = max(set(para_2_font_type), key=para_2_font_type.count)
        # 如果第一个段落的字体大小是列表，计算平均字体大小
        if isinstance(para_1_font_size, list):  # compute average font type
            para_1_font_size = sum(para_1_font_size) / len(para_1_font_size)
        # 如果第二个段落的字体大小是列表，计算平均字体大小
        if isinstance(para_2_font_size, list):  # compute average font type
            para_2_font_size = sum(para_2_font_size) / len(para_2_font_size)

        # 返回字体类型相似且字体大小差异小于1.5的结果
        return (
            self.__is_similar_font_type(para_1_font_type, para_2_font_type)
            and abs(para_1_font_size - para_2_font_size) < 1.5
            # and para_font_color1 == para_font_color2
        )

    # 判断两个块是否来自同一块
    def _is_block_consistent(self, block1, block2):
        """
        This function determines whether block1 and block2 are originally from the same block

        Parameters
        ----------
        block1 : dict
            block1s
        block2 : dict
            block2

        Returns
        -------
        is_same : bool
            True if block1 and block2 are from the same block, else False
        """
        # 调用私有方法检查两个块的字体是否相同
        return self.__is_same_block_font(block1, block2)

    # 判断两个段落是否来自同一段落
    def _is_para_continued(self, para1, para2):
        """
        This function determines whether para1 and para2 are originally from the same paragraph

        Parameters
        ----------
        para1 : dict
            para1
        para2 : dict
            para2

        Returns
        -------
        is_same : bool
            True if para1 and para2 are from the same paragraph, else False
        """
        # 检查段落的字体一致性
        is_para_font_consistent = self.__is_para_font_consistent(para1, para2)
        # 检查段落的标点一致性
        is_para_puncs_consistent = self._is_para_puncs_consistent(para1, para2)

        # 返回字体和标点一致性的结果
        return is_para_font_consistent and is_para_puncs_consistent
    # 检查 block1 和 block2 的边界是否一致
    def _are_boundaries_of_block_consistent(self, block1, block2):
        """
        该函数检查 block1 和 block2 的边界是否一致

        参数
        ----------
        block1 : dict
            block1

        block2 : dict
            block2

        返回
        -------
        is_consistent : bool
            如果 block1 和 block2 的边界一致则返回 True，否则返回 False
        """

        # 获取 block1 的最后一行
        last_line_of_block1 = block1["lines"][-1]
        # 获取 block2 的第一行
        first_line_of_block2 = block2["lines"][0]

        # 获取最后一行的跨度信息
        spans_of_last_line_of_block1 = last_line_of_block1["spans"]
        # 获取第一行的跨度信息
        spans_of_first_line_of_block2 = first_line_of_block2["spans"]

        # 获取最后一行的字体类型并转换为小写
        font_type_of_last_line_of_block1 = spans_of_last_line_of_block1[0]["font"].lower()
        # 获取最后一行的字体大小
        font_size_of_last_line_of_block1 = spans_of_last_line_of_block1[0]["size"]
        # 获取最后一行的字体颜色
        font_color_of_last_line_of_block1 = spans_of_last_line_of_block1[0]["color"]
        # 获取最后一行的字体标志
        font_flags_of_last_line_of_block1 = spans_of_last_line_of_block1[0]["flags"]

        # 获取第一行的字体类型并转换为小写
        font_type_of_first_line_of_block2 = spans_of_first_line_of_block2[0]["font"].lower()
        # 获取第一行的字体大小
        font_size_of_first_line_of_block2 = spans_of_first_line_of_block2[0]["size"]
        # 获取第一行的字体颜色
        font_color_of_first_line_of_block2 = spans_of_first_line_of_block2[0]["color"]
        # 获取第一行的字体标志
        font_flags_of_first_line_of_block2 = spans_of_first_line_of_block2[0]["flags"]

        # 返回两个行的字体类型相似且字体大小差异小于1，同时字体标志一致的结果
        return (
            self.__is_similar_font_type(font_type_of_last_line_of_block1, font_type_of_first_line_of_block2)
            and abs(font_size_of_last_line_of_block1 - font_size_of_first_line_of_block2) < 1
            # and font_color_of_last_line_of_block1 == font_color_of_first_line_of_block2
            and font_flags_of_last_line_of_block1 == font_flags_of_first_line_of_block2
        )

    # 获取块中最后一个段落
    def _get_last_paragraph(self, block):
        """
        从块中检索最后一个段落。

        参数
        ----------
        block : dict
            从中检索段落的块。

        返回
        -------
        dict
            块的最后一个段落。
        """
        # 如果块中有段落
        if block["paras"]:
            # 获取最后一个段落的键
            last_para_key = list(block["paras"].keys())[-1]
            # 返回最后一个段落
            return block["paras"][last_para_key]
        else:
            # 如果没有段落，返回 None
            return None

    # 获取块中第一个段落
    def _get_first_paragraph(self, block):
        """
        从块中检索第一个段落。

        参数
        ----------
        block : dict
            从中检索段落的块。

        返回
        -------
        dict
            块的第一个段落。
        """
        # 如果块中有段落
        if block["paras"]:
            # 获取第一个段落的键
            first_para_key = list(block["paras"].keys())[0]
            # 返回第一个段落
            return block["paras"][first_para_key]
        else:
            # 如果没有段落，返回 None
            return None

    # 判断当前段落是否与下一个段落合并
    def should_merge_next_para(self, curr_para, next_para):
        # 检查当前段落是否继续
        if self._is_para_continued(curr_para, next_para):
            # 如果继续，返回 True
            return True
        else:
            # 否则返回 False
            return False
    # 根据给定的段落块列表和块 ID 查找特定块
    def find_block_by_id(self, para_blocks, block_id):
        # 遍历段落块列表
        for block in para_blocks:
            # 检查当前块的 ID 是否与目标 ID 匹配
            if block.get("block_id") == block_id:
                # 返回匹配的块
                return block
        # 如果没有找到，返回 None
        return None

    # 批量合并段落
    def batch_merge_paras(self, pdf_dict):
        # 遍历 PDF 字典中的每个页面
        for page_id, page_content in pdf_dict.items():
            # 检查页面 ID 是否以 "page_" 开头，并且包含段落块
            if page_id.startswith("page_") and page_content.get("para_blocks", []):
                # 获取当前页面的段落块
                para_blocks_of_page = page_content["para_blocks"]

                # 遍历当前页面的每个段落块
                for i in range(len(para_blocks_of_page)):
                    # 获取当前段落块
                    current_block = para_blocks_of_page[i]
                    # 获取当前段落块中的段落
                    paras = current_block["paras"]

                    # 遍历段落字典中的每个段落
                    for para_id, curr_para in list(paras.items()):
                        # 跳过标题段落
                        if curr_para.get("is_para_title"):
                            continue

                        # 如果当前段落需要合并下一个段落
                        while curr_para.get("merge_next_para"):
                            # 获取下一个段落的位置
                            next_para_location = curr_para.get("next_para_location")
                            # 如果没有下一个段落位置，退出循环
                            if not next_para_location:
                                break

                            # 解构下一个段落的位置
                            next_page_idx, next_block_id, next_para_id = next_para_location
                            # 构造下一个页面的 ID
                            next_page_id = f"page_{next_page_idx}"
                            # 获取下一个页面的内容
                            next_page_content = pdf_dict.get(next_page_id)
                            # 如果下一个页面不存在，退出循环
                            if not next_page_content:
                                break

                            # 根据块 ID 查找下一个块
                            next_block = self.find_block_by_id(next_page_content.get("para_blocks", []), next_block_id)
                            # 如果没有找到下一个块，退出循环
                            if not next_block:
                                break

                            # 获取下一个段落
                            next_para = next_block["paras"].get(f"para_{next_para_id}")
                            # 如果下一个段落不存在或是标题段落，退出循环
                            if not next_para or next_para.get("is_para_title"):
                                break

                            # 合并当前段落的文本与下一个段落的文本
                            curr_para_text = curr_para.get("para_text", "")
                            next_para_text = next_para.get("para_text", "")
                            curr_para["para_text"] = curr_para_text + " " + next_para_text

                            # 更新当前段落的下一个段落位置
                            curr_para["next_para_location"] = next_para.get("next_para_location")

                            # 将下一个段落的文本置为空，表示已被合并
                            next_para["para_text"] = ""

                            # 更新当前段落的合并标记
                            curr_para["merge_next_para"] = next_para.get("merge_next_para", False)

        # 返回更新后的 PDF 字典
        return pdf_dict
```