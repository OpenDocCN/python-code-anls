# `.\MinerU\magic_pdf\pre_proc\equations_replace.py`

```
# 对pymupdf返回的结构里的公式进行替换，替换为模型识别的公式结果
"""

# 导入所需的库和模块
from magic_pdf.libs.commons import fitz  # 导入fitz模块，用于PDF处理
import json  # 导入json模块，用于处理JSON数据
import os  # 导入os模块，用于操作系统相关的功能
from pathlib import Path  # 从pathlib导入Path，用于路径操作
from loguru import logger  # 导入loguru模块，用于日志记录
from magic_pdf.libs.ocr_content_type import ContentType  # 导入ContentType，用于定义内容类型

# 定义两种内容类型：行内公式和行间公式
TYPE_INLINE_EQUATION = ContentType.InlineEquation
TYPE_INTERLINE_EQUATION = ContentType.InterlineEquation


def combine_chars_to_pymudict(block_dict, char_dict):
    """
    把block级别的pymupdf 结构里加入char结构
    """
    # 将字符字典与块字典对齐，以便后续补充
    char_map = {tuple(item["bbox"]): item for item in char_dict}

    for i in range(len(block_dict)):  # 遍历每个块
        block = block_dict[i]  # 当前块
        key = block["bbox"]  # 获取块的边界框
        char_dict_item = char_map[tuple(key)]  # 对应字符字典项
        char_dict_map = {tuple(item["bbox"]): item for item in char_dict_item["lines"]}  # 创建字符字典映射
        for j in range(len(block["lines"])):  # 遍历每个块的行
            lines = block["lines"][j]  # 当前行
            with_char_lines = char_dict_map[lines["bbox"]]  # 获取对应字符行
            for k in range(len(lines["spans"])):  # 遍历每个行的跨度
                spans = lines["spans"][k]  # 当前跨度
                try:
                    chars = with_char_lines["spans"][k]["chars"]  # 获取字符
                except Exception as e:
                    logger.error(char_dict[i]["lines"][j])  # 记录错误信息

                spans["chars"] = chars  # 将字符赋值给跨度

    return block_dict  # 返回更新后的块字典


def calculate_overlap_area_2_minbox_area_ratio(bbox1, min_bbox):
    """
    计算box1和box2的重叠面积占最小面积的box的比例
    """
    # 确定重叠矩形的坐标
    x_left = max(bbox1[0], min_bbox[0])  # 左侧坐标
    y_top = max(bbox1[1], min_bbox[1])  # 上侧坐标
    x_right = min(bbox1[2], min_bbox[2])  # 右侧坐标
    y_bottom = min(bbox1[3], min_bbox[3])  # 下侧坐标

    if x_right < x_left or y_bottom < y_top:  # 检查是否有重叠
        return 0.0  # 无重叠，返回0

    # 计算重叠面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)  # 重叠面积
    min_box_area = (min_bbox[3] - min_bbox[1]) * (min_bbox[2] - min_bbox[0])  # 最小框的面积
    if min_box_area == 0:  # 如果最小框的面积为0
        return 0  # 返回0
    else:
        return intersection_area / min_box_area  # 返回重叠面积与最小框面积的比例


def _is_xin(bbox1, bbox2):
    area1 = abs(bbox1[2] - bbox1[0]) * abs(bbox1[3] - bbox1[1])  # 计算bbox1的面积
    area2 = abs(bbox2[2] - bbox2[0]) * abs(bbox2[3] - bbox2[1])  # 计算bbox2的面积
    if area1 < area2:  # 判断面积大小
        ratio = calculate_overlap_area_2_minbox_area_ratio(bbox2, bbox1)  # 计算重叠比例
    else:
        ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)  # 计算重叠比例

    return ratio > 0.6  # 返回重叠比例是否大于0.6


def remove_text_block_in_interline_equation_bbox(interline_bboxes, text_blocks):
    """消除掉整个块都在行间公式块内部的文本块"""
    for eq_bbox in interline_bboxes:  # 遍历行间公式的边界框
        removed_txt_blk = []  # 存储被移除的文本块
        for text_blk in text_blocks:  # 遍历文本块
            text_bbox = text_blk["bbox"]  # 获取文本块的边界框
            if (
                calculate_overlap_area_2_minbox_area_ratio(eq_bbox["bbox"], text_bbox)  # 计算重叠比例
                >= 0.7  # 判断是否大于等于0.7
            ):
                removed_txt_blk.append(text_blk)  # 添加到移除列表
        for blk in removed_txt_blk:  # 遍历被移除的文本块
            text_blocks.remove(blk)  # 从文本块列表中移除

    return text_blocks  # 返回更新后的文本块列表


def _is_in_or_part_overlap(box1, box2) -> bool:
    """
    两个bbox是否有部分重叠或者包含
    """
    if box1 is None or box2 is None:  # 检查边界框是否为空
        return False  # 如果为空，返回False
    # 解包 box1 的坐标值，分别赋值给 x0_1, y0_1, x1_1, y1_1
    x0_1, y0_1, x1_1, y1_1 = box1
    # 解包 box2 的坐标值，分别赋值给 x0_2, y0_2, x1_2, y1_2
    x0_2, y0_2, x1_2, y1_2 = box2

    # 返回两个框是否相交，使用逻辑非操作符
    return not (
        x1_1 < x0_2  # box1在box2的左边，如果是，则不相交
        or x0_1 > x1_2  # box1在box2的右边，如果是，则不相交
        or y1_1 < y0_2  # box1在box2的上边，如果是，则不相交
        or y0_1 > y1_2  # box1在box2的下边，如果是，则不相交
    )  # 如果以上条件均不成立，表示 box1 和 box2 相交
# 定义一个函数，移除行间公式与文本块的重叠部分，同时重新计算文本块的大小
def remove_text_block_overlap_interline_equation_bbox(
    interline_eq_bboxes, pymu_block_list
):

    # 文档字符串，说明函数的功能
    """消除掉行行内公式有部分重叠的文本块的内容。
    同时重新计算消除重叠之后文本块的大小"""
    # 存储被删除的文本块
    deleted_block = []
    # 遍历给定的文本块列表
    for text_block in pymu_block_list:
        # 存储被删除的行
        deleted_line = []
        # 遍历文本块中的每一行
        for line in text_block["lines"]:
            # 存储被删除的 span
            deleted_span = []
            # 遍历行中的每一个 span
            for span in line["spans"]:
                # 存储被删除的字符
                deleted_chars = []
                # 遍历 span 中的每一个字符
                for char in span["chars"]:
                    # 检查字符是否与任何行间公式的边界框有重叠
                    if any(
                            [
                                (calculate_overlap_area_2_minbox_area_ratio(eq_bbox["bbox"], char["bbox"]) > 0.5)
                                for eq_bbox in interline_eq_bboxes
                            ]
                    ):
                        # 如果有重叠，记录该字符
                        deleted_chars.append(char)
                # 检查 span 里没有字符则删除这个 span
                for char in deleted_chars:
                    span["chars"].remove(char)
                # 重新计算这个 span 的边界框大小
                if len(span["chars"]) == 0:  # 如果没有字符则删除这个 span
                    deleted_span.append(span)
                else:
                    # 更新 span 的边界框
                    span["bbox"] = (
                        min([b["bbox"][0] for b in span["chars"]]),  # 计算最小 x 坐标
                        min([b["bbox"][1] for b in span["chars"]]),  # 计算最小 y 坐标
                        max([b["bbox"][2] for b in span["chars"]]),  # 计算最大 x 坐标
                        max([b["bbox"][3] for b in span["chars"]]),  # 计算最大 y 坐标
                    )

            # 检查被删除的 span
            for span in deleted_span:
                line["spans"].remove(span)  # 从行中移除被删除的 span
            if len(line["spans"]) == 0:  # 如果行中没有 span 则删除这行
                deleted_line.append(line)
            else:
                # 更新行的边界框
                line["bbox"] = (
                    min([b["bbox"][0] for b in line["spans"]]),  # 计算最小 x 坐标
                    min([b["bbox"][1] for b in line["spans"]]),  # 计算最小 y 坐标
                    max([b["bbox"][2] for b in line["spans"]]),  # 计算最大 x 坐标
                    max([b["bbox"][3] for b in line["spans"]]),  # 计算最大 y 坐标
                )

        # 检查是否可以删除整个文本块
        for line in deleted_line:
            text_block["lines"].remove(line)  # 从文本块中移除被删除的行
        if len(text_block["lines"]) == 0:  # 如果文本块没有行则删除这个文本块
            deleted_block.append(text_block)
        else:
            # 更新文本块的边界框
            text_block["bbox"] = (
                min([b["bbox"][0] for b in text_block["lines"]]),  # 计算最小 x 坐标
                min([b["bbox"][1] for b in text_block["lines"]]),  # 计算最小 y 坐标
                max([b["bbox"][2] for b in text_block["lines"]]),  # 计算最大 x 坐标
                max([b["bbox"][3] for b in text_block["lines"]]),  # 计算最大 y 坐标
            )

    # 检查并删除已标记的文本块
    for block in deleted_block:
        pymu_block_list.remove(block)  # 从列表中移除已删除的文本块
    if len(pymu_block_list) == 0:  # 如果列表为空
        return []  # 返回空列表

    return pymu_block_list  # 返回处理后的文本块列表


# 定义一个函数，在行间公式对应的地方插入一个伪造的文本块
def insert_interline_equations_textblock(interline_eq_bboxes, pymu_block_list):
    # 文档字符串，说明函数的功能
    """在行间公式对应的地方插上一个伪造的block"""
    # 遍历所有的行内公式边界框
    for eq in interline_eq_bboxes:
        # 获取当前公式的边界框信息
        bbox = eq["bbox"]
        # 获取当前公式的 LaTeX 内容
        latex_content = eq["latex"]
        # 创建一个文本块字典，包含公式的相关信息
        text_block = {
            # 记录当前文本块的编号
            "number": len(pymu_block_list),
            # 设置文本块类型为 0
            "type": 0,
            # 存储边界框信息
            "bbox": bbox,
            # 定义文本块内的行内容
            "lines": [
                {
                    # 定义行内的跨度信息
                    "spans": [
                        {
                            # 设置字体大小
                            "size": 9.962599754333496,
                            # 设置跨度类型为行内公式类型
                            "type": TYPE_INTERLINE_EQUATION,
                            # 设置标志位
                            "flags": 4,
                            # 设置字体类型
                            "font": TYPE_INTERLINE_EQUATION,
                            # 设置字体颜色
                            "color": 0,
                            # 设置上升线
                            "ascender": 0.9409999847412109,
                            # 设置下降线
                            "descender": -0.3050000071525574,
                            # 存储 LaTeX 内容
                            "latex": latex_content,
                            # 设置起始点为边界框左下角
                            "origin": [bbox[0], bbox[1]],
                            # 存储边界框信息
                            "bbox": bbox,
                        }
                    ],
                    # 设置绘制模式为 0
                    "wmode": 0,
                    # 设置文本方向为水平方向
                    "dir": [1.0, 0.0],
                    # 存储边界框信息
                    "bbox": bbox,
                }
            ],
        }
        # 将文本块添加到 pymu_block_list 中
        pymu_block_list.append(text_block)
# 计算两个矩形在X轴方向上的重叠比例
def x_overlap_ratio(box1, box2):
    # 解包box1的边界值
    a, _, c, _ = box1
    # 解包box2的边界值
    e, _, g, _ = box2

    # 计算重叠宽度
    overlap_x = max(min(c, g) - max(a, e), 0)

    # 计算box1的宽度
    width1 = g - e

    # 计算重叠比例
    overlap_ratio = overlap_x / width1 if width1 != 0 else 0

    # 返回重叠比例
    return overlap_ratio


# 判断两个矩形在X轴方向上是否重叠
def __is_x_dir_overlap(bbox1, bbox2):
    # 如果bbox1的右边界在bbox2的左边界左侧或bbox1的左边界在bbox2的右边界右侧，则不重叠
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2])


# 计算两个矩形在Y轴方向上的重叠比例
def __y_overlap_ratio(box1, box2):
    """"""
    # 解包box1的边界值
    _, b, _, d = box1
    # 解包box2的边界值
    _, f, _, h = box2

    # 计算重叠高度
    overlap_y = max(min(d, h) - max(b, f), 0)

    # 计算box1的高度
    height1 = d - b

    # 计算重叠比例
    overlap_ratio = overlap_y / height1 if height1 != 0 else 0

    # 返回重叠比例
    return overlap_ratio


# 替换公式行中的字符
def replace_line_v2(eqinfo, line):
    """
    扫描这一行所有的和公式框X方向重叠的char,然后计算char的左、右x0, x1,位于这个区间内的span删除掉。
    最后与这个x0,x1有相交的span0, span1内部进行分割。
    """
    # 初始化重叠span的索引和标识
    first_overlap_span = -1
    first_overlap_span_idx = -1
    last_overlap_span = -1
    delete_chars = []

    # 遍历行中的每个span
    for i in range(0, len(line["spans"])):
        # 检查span中是否包含字符
        if "chars" not in line["spans"][i]:
            continue

        # 检查span是否为插入的伪造公式
        if line["spans"][i].get("_type", None) is not None:
            continue  # 忽略，因为已经是插入的伪造span公式了

        # 遍历span中的每个字符
        for char in line["spans"][i]["chars"]:
            # 检查字符是否与公式框在X方向上重叠
            if __is_x_dir_overlap(eqinfo["bbox"], char["bbox"]):
                # 初始化行文本
                line_txt = ""
                # 遍历行中的每个span
                for span in line["spans"]:
                    span_txt = "<span>"
                    # 遍历span中的每个字符
                    for ch in span["chars"]:
                        span_txt = span_txt + ch["c"]

                    # 添加span结束标签
                    span_txt = span_txt + "</span>"

                    # 连接所有span文本
                    line_txt = line_txt + span_txt

                # 记录第一个重叠span
                if first_overlap_span_idx == -1:
                    first_overlap_span = line["spans"][i]
                    first_overlap_span_idx = i
                # 更新最后一个重叠span
                last_overlap_span = line["spans"][i]
                # 记录重叠的字符
                delete_chars.append(char)

    # 检查第一个和最后一个字符的重叠比例
    if len(delete_chars) > 0:
        ch0_bbox = delete_chars[0]["bbox"]
        # 如果重叠比例小于0.51，则移除第一个字符
        if x_overlap_ratio(eqinfo["bbox"], ch0_bbox) < 0.51:
            delete_chars.remove(delete_chars[0])
    if len(delete_chars) > 0:
        ch0_bbox = delete_chars[-1]["bbox"]
        # 如果重叠比例小于0.51，则移除最后一个字符
        if x_overlap_ratio(eqinfo["bbox"], ch0_bbox) < 0.51:
            delete_chars.remove(delete_chars[-1])

    # 计算被删除区间内字符的真实x0, x1
    if len(delete_chars):
        x0, x1 = min([b["bbox"][0] for b in delete_chars]), max(
            [b["bbox"][2] for b in delete_chars]
        )
    else:
        # 如果没有字符被删除，则返回False
        return False

    # 删除位于x0, x1这两个中间的span
    delete_span = []
    for span in line["spans"]:
        span_box = span["bbox"]
        # 检查span是否在删除区间内
        if x0 <= span_box[0] and span_box[2] <= x1:
            delete_span.append(span)
    # 从行中移除这些span
    for span in delete_span:
        line["spans"].remove(span)
    # 创建一个包含方程信息的字典
        equation_span = {
            # 设置字典的大小属性
            "size": 9.962599754333496,
            # 设置字典的类型为内联方程
            "type": TYPE_INLINE_EQUATION,
            # 设置标志位
            "flags": 4,
            # 设置字体为内联方程字体
            "font": TYPE_INLINE_EQUATION,
            # 设置颜色属性
            "color": 0,
            # 设置上升基线
            "ascender": 0.9409999847412109,
            # 设置下降基线
            "descender": -0.3050000071525574,
            # 初始化 LaTeX 字符串为空
            "latex": "",
            # 设置方程的原点坐标
            "origin": [337.1410153102337, 216.0205245153934],
            # 从 eqinfo 中获取边界框信息
            "bbox": eqinfo["bbox"]
        }
        # 复制第一段的内容到 equation_span（已注释掉）
        # equation_span = line['spans'][0].copy()
        # 从 eqinfo 中提取 LaTeX 表达式并赋值给字典
        equation_span["latex"] = eqinfo['latex']
        # 更新边界框，包含新的 x 坐标
        equation_span["bbox"] = [x0, equation_span["bbox"][1], x1, equation_span["bbox"][3]]
        # 更新原点为新边界框的左下角坐标
        equation_span["origin"] = [equation_span["bbox"][0], equation_span["bbox"][1]]
        # 设置字符信息
        equation_span["chars"] = delete_chars
        # 确保类型为内联方程
        equation_span["type"] = TYPE_INLINE_EQUATION
        # 记录方程的边界框信息
        equation_span["_eq_bbox"] = eqinfo["bbox"]
        # 在指定索引处插入方程信息到行的 spans 列表中
        line["spans"].insert(first_overlap_span_idx + 1, equation_span)  # 放入公式
    
        # 记录信息日志（已注释掉）
        # logger.info(f"==>text is 【{line_txt}】, equation is 【{eqinfo['latex_text']}】")
    
        # 处理第一个和最后一个重叠的 span，分割并插入
        first_span_chars = [
            # 遍历第一个重叠 span 的字符，过滤出左边的字符
            char
            for char in first_overlap_span["chars"]
            if (char["bbox"][2] + char["bbox"][0]) / 2 < x0
        ]
        tail_span_chars = [
            # 遍历最后一个重叠 span 的字符，过滤出右边的字符
            char
            for char in last_overlap_span["chars"]
            if (char["bbox"][0] + char["bbox"][2]) / 2 > x1
        ]
    
        # 如果有第一个 span 的字符存在
        if len(first_span_chars) > 0:
            # 更新第一个重叠 span 的字符
            first_overlap_span["chars"] = first_span_chars
            # 拼接文本
            first_overlap_span["text"] = "".join([char["c"] for char in first_span_chars])
            # 更新边界框，使用字符中最右边的边界值
            first_overlap_span["bbox"] = (
                first_overlap_span["bbox"][0],
                first_overlap_span["bbox"][1],
                max([chr["bbox"][2] for chr in first_span_chars]),
                first_overlap_span["bbox"][3],
            )
            # 更新类型为 "first"（已注释掉）
            # first_overlap_span['_type'] = "first"
        else:
            # 如果没有字符，则删除第一个重叠 span
            if first_overlap_span not in delete_span:
                line["spans"].remove(first_overlap_span)
    # 检查尾部字符是否存在
    if len(tail_span_chars) > 0:
        # 获取尾部字符的最小 x 坐标
        min_of_tail_span_x0 = min([chr["bbox"][0] for chr in tail_span_chars])
        # 获取尾部字符的最小 y 坐标
        min_of_tail_span_y0 = min([chr["bbox"][1] for chr in tail_span_chars])
        # 获取尾部字符的最大 x 坐标
        max_of_tail_span_x1 = max([chr["bbox"][2] for chr in tail_span_chars])
        # 获取尾部字符的最大 y 坐标
        max_of_tail_span_y1 = max([chr["bbox"][3] for chr in tail_span_chars])

        # 判断最后重叠的范围与第一个重叠的范围是否相同
        if last_overlap_span == first_overlap_span:  # 这个时候应该插入一个新的
            # 将尾部字符拼接成字符串
            tail_span_txt = "".join([char["c"] for char in tail_span_chars])
            # 复制最后重叠的范围
            last_span_to_insert = last_overlap_span.copy()
            # 更新插入范围的字符和文本
            last_span_to_insert["chars"] = tail_span_chars
            last_span_to_insert["text"] = "".join(
                [char["c"] for char in tail_span_chars]
            )
            # 判断公式的边界是否大于等于最后重叠范围的右边界
            if equation_span["bbox"][2] >= last_overlap_span["bbox"][2]:
                # 更新插入范围的边界
                last_span_to_insert["bbox"] = (
                    min_of_tail_span_x0,
                    min_of_tail_span_y0,
                    max_of_tail_span_x1,
                    max_of_tail_span_y1
                )
            else:
                # 更新插入范围的边界（与最后重叠范围相关）
                last_span_to_insert["bbox"] = (
                    min([chr["bbox"][0] for chr in tail_span_chars]),
                    last_overlap_span["bbox"][1],
                    last_overlap_span["bbox"][2],
                    last_overlap_span["bbox"][3],
                )
            # 插入到公式对象之后
            equation_idx = line["spans"].index(equation_span)
            line["spans"].insert(equation_idx + 1, last_span_to_insert)  # 放入公式
        else:  # 直接修改原来的span
            # 更新最后重叠范围的字符和文本
            last_overlap_span["chars"] = tail_span_chars
            last_overlap_span["text"] = "".join([char["c"] for char in tail_span_chars])
            # 更新最后重叠范围的边界（保持 y 方向不变）
            last_overlap_span["bbox"] = (
                min([chr["bbox"][0] for chr in tail_span_chars]),
                last_overlap_span["bbox"][1],
                last_overlap_span["bbox"][2],
                last_overlap_span["bbox"][3],
            )
    else:
        # 删掉
        # 检查最后重叠范围是否不在删除列表中且不等于第一个重叠范围
        if (
            last_overlap_span not in delete_span
            and last_overlap_span != first_overlap_span
        ):
            # 从行中移除最后重叠的范围
            line["spans"].remove(last_overlap_span)

    # 初始化剩余文本为一个空字符串
    remain_txt = ""
    # 遍历行中的每个 span
    for span in line["spans"]:
        # 初始化 span 的 HTML 标签
        span_txt = "<span>"
        # 拼接 span 中的字符
        for char in span["chars"]:
            span_txt = span_txt + char["c"]

        # 关闭 span 的 HTML 标签
        span_txt = span_txt + "</span>"

        # 将当前 span 的 HTML 文本添加到剩余文本中
        remain_txt = remain_txt + span_txt

    # logger.info(f"<== succ replace, text is 【{remain_txt}】, equation is 【{eqinfo['latex_text']}】")

    # 返回成功标志
    return True
# 定义函数，用于替换行内公式
def replace_eq_blk(eqinfo, text_block):
    """替换行内公式"""
    # 遍历文本块中的每一行
    for line in text_block["lines"]:
        # 获取当前行的边界框
        line_bbox = line["bbox"]
        # 检查公式的边界框是否与行的边界框重叠
        if (
            _is_xin(eqinfo["bbox"], line_bbox)
            or __y_overlap_ratio(eqinfo["bbox"], line_bbox) > 0.6
        ):  # 定位到行, 使用y方向重合率是因为有的时候，一个行的宽度会小于公式位置宽度：行很高，公式很窄，
            # 尝试在当前行替换公式
            replace_succ = replace_line_v2(eqinfo, line)
            # 检查替换是否成功
            if (
                not replace_succ
            ):  # 有的时候，一个pdf的line高度从API里会计算的有问题，因此在行内span级别会替换不成功，这就需要继续重试下一行
                # 如果替换不成功，继续下一个行
                continue
            else:
                # 如果替换成功，退出循环
                break
    else:
        # 如果没有成功替换任何行，返回 False
        return False
    # 返回 True 表示替换成功
    return True


# 定义函数，用于替换行内公式
def replace_inline_equations(inline_equation_bboxes, raw_text_blocks):
    """替换行内公式"""
    # 遍历所有行内公式的边界框
    for eqinfo in inline_equation_bboxes:
        # 获取公式的边界框
        eqbox = eqinfo["bbox"]
        # 遍历所有原始文本块
        for blk in raw_text_blocks:
            # 检查公式边界框是否与文本块边界框重叠
            if _is_xin(eqbox, blk["bbox"]):
                # 尝试在文本块中替换公式
                if not replace_eq_blk(eqinfo, blk):
                    # 如果替换不成功，记录警告信息
                    logger.warning(f"行内公式没有替换成功：{eqinfo} ")
                else:
                    # 替换成功，退出循环
                    break

    # 返回处理后的文本块
    return raw_text_blocks


# 定义函数，用于删除文本块中的字符
def remove_chars_in_text_blocks(text_blocks):
    """删除text_blocks里的char"""
    # 遍历所有文本块
    for blk in text_blocks:
        # 遍历文本块中的每一行
        for line in blk["lines"]:
            # 遍历行中的每个跨度
            for span in line["spans"]:
                # 删除跨度中的字符，若不存在则返回默认值
                _ = span.pop("chars", "no such key")
    # 返回处理后的文本块
    return text_blocks


# 定义函数，用于在文本块中替换行间和行内公式为 LaTeX
def replace_equations_in_textblock(
    raw_text_blocks, inline_equation_bboxes, interline_equation_bboxes
):
    """
    替换行间和和行内公式为latex
    """
    # 从原始文本块中移除行间公式的文本块
    raw_text_blocks = remove_text_block_in_interline_equation_bbox(
        interline_equation_bboxes, raw_text_blocks
    )  # 消除重叠：第一步，在公式内部的

    # 从原始文本块中移除与行间公式重叠的文本块
    raw_text_blocks = remove_text_block_overlap_interline_equation_bbox(
        interline_equation_bboxes, raw_text_blocks
    )  # 消重，第二步，和公式覆盖的

    # 插入行间公式的文本块
    insert_interline_equations_textblock(interline_equation_bboxes, raw_text_blocks)
    # 替换行内公式
    raw_text_blocks = replace_inline_equations(inline_equation_bboxes, raw_text_blocks)
    # 返回处理后的文本块
    return raw_text_blocks


# 定义函数，用于在 PDF 中绘制带替换行内公式的文本块
def draw_block_on_pdf_with_txt_replace_eq_bbox(json_path, pdf_path):
    """ """
    # 创建新的 PDF 文件路径
    new_pdf = f"{Path(pdf_path).parent}/{Path(pdf_path).stem}.step3-消除行内公式text_block.pdf"
    # 以读取模式打开 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        # 解析 JSON 文件内容
        obj = json.loads(f.read())

    # 如果新 PDF 文件已存在，则删除
    if os.path.exists(new_pdf):
        os.remove(new_pdf)
    # 打开一个新的 PDF 文档
    new_doc = fitz.open("")

    # 打开原始 PDF 文件
    doc = fitz.open(pdf_path)
    # 将原始 PDF 文件复制到新的 PDF 文档
    new_doc = fitz.open(pdf_path)
    # 遍历新文档中的每一页
    for i in range(len(new_doc)):
        # 获取当前页对象
        page = new_doc[i]
        # 获取当前页中的行内公式的边界框
        inline_equation_bboxes = obj[f"page_{i}"]["inline_equations"]
        # 获取当前页中的行间公式的边界框
        interline_equation_bboxes = obj[f"page_{i}"]["interline_equations"]
        # 获取当前页的原始文本块
        raw_text_blocks = obj[f"page_{i}"]["preproc_blocks"]
        # 移除与行间公式重叠的文本块
        raw_text_blocks = remove_text_block_in_interline_equation_bbox(
            interline_equation_bboxes, raw_text_blocks
        )  # 消除重叠：第一步，在公式内部的
        # 再次移除与行间公式重叠的文本块
        raw_text_blocks = remove_text_block_overlap_interline_equation_bbox(
            interline_equation_bboxes, raw_text_blocks
        )  # 消重，第二步，和公式覆盖的
        # 插入行间公式文本块
        insert_interline_equations_textblock(interline_equation_bboxes, raw_text_blocks)
        # 替换行内公式文本块
        raw_text_blocks = replace_inline_equations(
            inline_equation_bboxes, raw_text_blocks
        )

        # 为了检验公式是否重复，把每一行里，含有公式的span背景改成黄色的
        color_map = [fitz.pdfcolor["blue"], fitz.pdfcolor["green"]]  # 定义颜色映射
        j = 0  # 计数器初始化
        # 遍历每个文本块
        for blk in raw_text_blocks:
            # 遍历文本块中的每一行
            for i, line in enumerate(blk["lines"]):

                # 下面的代码被注释掉了，用于绘制行的矩形框
                # line_box = line['bbox']
                # shape = page.new_shape()
                # shape.draw_rect(line_box)
                # shape.finish(color=fitz.pdfcolor['red'], fill=color_map[j%2], fill_opacity=0.3)
                # shape.commit()
                # j = j+1

                # 遍历行中的每个span
                for i, span in enumerate(line["spans"]):
                    shape_page = page.new_shape()  # 创建新的形状对象
                    span_type = span.get("_type")  # 获取span的类型
                    color = fitz.pdfcolor["blue"]  # 默认颜色为蓝色
                    # 根据span类型设置颜色
                    if span_type == "first":
                        color = fitz.pdfcolor["blue"]  # 首个公式颜色
                    elif span_type == "tail":
                        color = fitz.pdfcolor["green"]  # 尾部公式颜色
                    elif span_type == TYPE_INLINE_EQUATION:
                        color = fitz.pdfcolor["black"]  # 行内公式颜色
                    else:
                        color = None  # 其他类型不设置颜色

                    b = span["bbox"]  # 获取span的边界框
                    shape_page.draw_rect(b)  # 在页面上绘制边界框

                    shape_page.finish(color=None, fill=color, fill_opacity=0.3)  # 完成形状，设置填充颜色和透明度
                    shape_page.commit()  # 提交形状到页面

    new_doc.save(new_pdf)  # 保存新文档为新的PDF文件
    logger.info(f"save ok {new_pdf}")  # 记录保存成功的信息
    final_json = json.dumps(obj, ensure_ascii=False, indent=2)  # 将对象转换为JSON格式字符串
    with open("equations_test/final_json.json", "w") as f:  # 打开JSON文件以写入
        f.write(final_json)  # 将JSON字符串写入文件

    return new_pdf  # 返回新生成的PDF文件名
# 判断当前模块是否为主程序入口
if __name__ == "__main__":
    # 调用函数，绘制 PDF 中的块并替换文本，参数为 JSON 文件路径和方程颜色 PDF
    # draw_block_on_pdf_with_txt_replace_eq_bbox(new_json_path, equation_color_pdf)
    # 占位符，暂时不执行任何操作
    pass
```