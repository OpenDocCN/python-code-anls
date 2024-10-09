# `.\MinerU\magic_pdf\pre_proc\ocr_span_list_modify.py`

```
# 从 loguru 库导入 logger，用于记录日志
from loguru import logger

# 从 magic_pdf.libs.boxbase 导入相关函数，用于计算重叠区域和 IOU
from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio, get_minbox_if_overlap_by_ratio, \
    __is_overlaps_y_exceeds_threshold, calculate_iou
# 从 magic_pdf.libs.drop_tag 导入 DropTag，用于标记被删除的 spans
from magic_pdf.libs.drop_tag import DropTag
# 从 magic_pdf.libs.ocr_content_type 导入 ContentType 和 BlockType，可能用于处理 OCR 内容类型
from magic_pdf.libs.ocr_content_type import ContentType, BlockType

# 定义函数以删除置信度低的重叠 spans
def remove_overlaps_low_confidence_spans(spans):
    dropped_spans = []  # 初始化被删除的 spans 列表
    # 遍历所有 spans，比较每对 spans 的重叠情况
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:  # 确保不比较自身
                # 检查 span1 或 span2 是否已经在 dropped_spans 中
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue  # 如果在，跳过此对
                else:
                    # 计算两个 spans 的 IOU，如果超过 0.9，表示重叠
                    if calculate_iou(span1['bbox'], span2['bbox']) > 0.9:
                        # 判断哪个 span 置信度低，决定需要删除的 span
                        if span1['score'] < span2['score']:
                            span_need_remove = span1
                        else:
                            span_need_remove = span2
                        # 如果需要删除的 span 不在已删除列表中，添加进去
                        if span_need_remove is not None and span_need_remove not in dropped_spans:
                            dropped_spans.append(span_need_remove)

    # 如果有需要删除的 spans，进行标记和删除
    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)  # 从原列表中移除
            span_need_remove['tag'] = DropTag.SPAN_OVERLAP  # 标记为重叠

    # 返回处理后的 spans 和已删除的 spans
    return spans, dropped_spans

# 定义函数以删除较小的重叠 spans
def remove_overlaps_min_spans(spans):
    dropped_spans = []  # 初始化被删除的 spans 列表
    # 遍历所有 spans，比较每对 spans 的重叠情况
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:  # 确保不比较自身
                # 获取重叠区域的最小边界框，重叠比率大于 0.65
                overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.65)
                if overlap_box is not None:  # 如果存在重叠区域
                    # 查找该重叠区域对应的 span
                    span_need_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                    # 如果需要删除的 span 不在已删除列表中，添加进去
                    if span_need_remove is not None and span_need_remove not in dropped_spans:
                        dropped_spans.append(span_need_remove)

    # 如果有需要删除的 spans，进行标记和删除
    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)  # 从原列表中移除
            span_need_remove['tag'] = DropTag.SPAN_OVERLAP  # 标记为重叠

    # 返回处理后的 spans 和已删除的 spans
    return spans, dropped_spans

# 定义函数根据边界框删除指定 spans
def remove_spans_by_bboxes(spans, need_remove_spans_bboxes):
    # 遍历 spans，判断是否在需要删除的边界框中
    need_remove_spans = []  # 初始化需要删除的 spans 列表
    for span in spans:
        for removed_bbox in need_remove_spans_bboxes:
            # 计算重叠面积与比例，如果大于 0.5，则标记为需要删除
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], removed_bbox) > 0.5:
                if span not in need_remove_spans:
                    need_remove_spans.append(span)  # 添加到需要删除列表
                    break  # 找到重叠后跳出循环

    # 如果有需要删除的 spans，进行删除
    if len(need_remove_spans) > 0:
        for span in need_remove_spans:
            spans.remove(span)  # 从原列表中移除

    # 返回处理后的 spans
    return spans

# 定义函数根据边界框字典删除指定 spans
def remove_spans_by_bboxes_dict(spans, need_remove_spans_bboxes_dict):
    dropped_spans = []  # 初始化被删除的 spans 列表
    # 遍历需要移除的 span 的边界框字典中的每个条目
    for drop_tag, removed_bboxes in need_remove_spans_bboxes_dict.items():
        # 记录移除操作的信息（注释掉的日志信息）
        # logger.info(f"remove spans by bbox dict, drop_tag: {drop_tag}, removed_bboxes: {removed_bboxes}")
        # 初始化一个列表，用于存储需要移除的 spans
        need_remove_spans = []
        # 遍历所有的 span
        for span in spans:
            # 通过判断 span 的 bbox 是否在 removed_bboxes 中，判断是否需要删除该 span
            for removed_bbox in removed_bboxes:
                # 计算 span 的 bbox 与 removed_bbox 的重叠面积比，如果超过 0.5，则标记该 span 需要移除
                if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], removed_bbox) > 0.5:
                    need_remove_spans.append(span)
                    break
                # 当 drop_tag 为 DropTag.FOOTNOTE 时，判断 span 是否在 removed_bboxes 中任意一个的下方
                # 如果是，则标记该 span 需要移除
                elif drop_tag == DropTag.FOOTNOTE and (span['bbox'][1] + span['bbox'][3]) / 2 > removed_bbox[3] and \
                        removed_bbox[0] < (span['bbox'][0] + span['bbox'][2]) / 2 < removed_bbox[2]:
                    need_remove_spans.append(span)
                    break

        # 遍历所有需要移除的 spans
        for span in need_remove_spans:
            # 从 spans 中移除该 span
            spans.remove(span)
            # 将 span 的标签设置为 drop_tag
            span['tag'] = drop_tag
            # 将该 span 添加到已移除的 spans 列表中
            dropped_spans.append(span)

    # 返回剩余的 spans 和已移除的 spans
    return spans, dropped_spans
# 调整独立块的边界框以适应包含的文本
def adjust_bbox_for_standalone_block(spans):
    # 对类型为 "interline_equation", "image", "table" 的元素进行额外处理
    for sb_span in spans:
        # 检查当前 span 是否为特定类型
        if sb_span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            # 遍历所有 spans 以查找文本相关的 span
            for text_span in spans:
                # 检查当前 span 是否为文本或行内方程
                if text_span['type'] in [ContentType.Text, ContentType.InlineEquation]:
                    # 判断当前 span 的 y 轴是否被文本 span 所覆盖
                    if sb_span['bbox'][1] < text_span['bbox'][1] and sb_span['bbox'][3] > text_span['bbox'][3]:
                        # 判断文本 span 是否位于当前 span 的左侧
                        if text_span['bbox'][0] < sb_span['bbox'][0]:
                            # 调整当前 span 的 y0 使其与文本 span 的 y0 一致
                            sb_span['bbox'][1] = text_span['bbox'][1]
    # 返回调整后的 spans 列表
    return spans


# 修改 y 轴以适应给定的 spans 和显示列表
def modify_y_axis(spans: list, displayed_list: list, text_inline_lines: list):
    # 初始化显示列表（注释掉的代码）
    # displayed_list = []
    # 如果 spans 为空，则不进行处理
    if len(spans) == 0:
        pass
    else:
        # 按照每个 span 的 y 坐标进行排序
        spans.sort(key=lambda span: span['bbox'][1])

        # 初始化行的集合
        lines = []
        # 将第一个 span 加入当前行
        current_line = [spans[0]]
        # 如果第一个 span 是特定类型，则添加到显示列表中
        if spans[0]["type"] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            displayed_list.append(spans[0])

        # 记录当前行的 y 坐标信息
        line_first_y0 = spans[0]["bbox"][1]
        line_first_y = spans[0]["bbox"][3]
        # 用于存放行内的公式搜索
        # text_inline_lines = []
        for span in spans[1:]:
            # 如果当前的 span 类型为特定类型，或当前行已有特定类型
            if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] or any(
                    s['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] for s in
                    current_line):
                # 将当前 span 添加到显示列表中
                if span["type"] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
                    displayed_list.append(span)
                # 结束当前行，开始新行
                lines.append(current_line)
                # 如果当前行包含多个 span 或类型符合条件，则记录行内信息
                if len(current_line) > 1 or current_line[0]["type"] in [ContentType.Text, ContentType.InlineEquation]:
                    text_inline_lines.append((current_line, (line_first_y0, line_first_y)))
                # 重置当前行为仅当前 span
                current_line = [span]
                # 更新行的 y 坐标信息
                line_first_y0 = span["bbox"][1]
                line_first_y = span["bbox"][3]
                continue

            # 如果当前 span 与当前行最后一个 span 在 y 轴上重叠，则添加到当前行
            if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox']):
                # 如果当前 span 类型为文本，更新 y 坐标
                if span["type"] == "text":
                    line_first_y0 = span["bbox"][1]
                    line_first_y = span["bbox"][3]
                # 将 span 添加到当前行
                current_line.append(span)

            else:
                # 否则，结束当前行，开始新行
                lines.append(current_line)
                # 记录当前行信息
                text_inline_lines.append((current_line, (line_first_y0, line_first_y)))
                # 重置当前行为仅当前 span
                current_line = [span]
                # 更新 y 坐标信息
                line_first_y0 = span["bbox"][1]
                line_first_y = span["bbox"][3]

        # 如果当前行非空，添加到行集合
        if current_line:
            lines.append(current_line)
            # 如果当前行符合条件，记录行内信息
            if len(current_line) > 1 or current_line[0]["type"] in [ContentType.Text, ContentType.InlineEquation]:
                text_inline_lines.append((current_line, (line_first_y0, line_first_y)))
        # 对每行的文本进行排序
        for line in text_inline_lines:
            # 获取当前行的 span
            current_line = line[0]
            # 根据 x0 坐标排序
            current_line.sort(key=lambda span: span['bbox'][0])

        # 统一每个行内的 bbox 信息
        for line in text_inline_lines:
            current_line, (line_first_y0, line_first_y) = line
            for span in current_line:
                # 更新每个 span 的 y 坐标
                span["bbox"][1] = line_first_y0
                span["bbox"][3] = line_first_y

        # return spans, displayed_list, text_inline_lines
# 修改行间公式为行内公式
def modify_inline_equation(spans: list, displayed_list: list, text_inline_lines: list):
    # 初始化索引 j
    j = 0
    # 遍历已显示的列表
    for i in range(len(displayed_list)):
        # 获取当前 span 对象
        span = displayed_list[i]
        # 获取 span 的 y 坐标
        span_y0, span_y = span["bbox"][1], span["bbox"][3]

        # 遍历文本行
        while j < len(text_inline_lines):
            # 获取当前文本行
            text_line = text_inline_lines[j]
            y0, y1 = text_line[1]
            # 判断 span 和文本行的 y 坐标是否重叠
            if (
                    span_y0 < y0 < span_y or span_y0 < y1 < span_y or span_y0 < y0 and span_y > y1
            ) and __is_overlaps_y_exceeds_threshold(
                span['bbox'], (0, y0, 0, y1)
            ):
                # 调整公式类型为行内公式
                if span["type"] == ContentType.InterlineEquation:
                    # 检查是否为最后一行
                    if j + 1 >= len(text_inline_lines):
                        span["type"] = ContentType.InlineEquation
                        span["bbox"][1] = y0
                        span["bbox"][3] = y1
                    else:
                        # 检查是否满足转换条件
                        y0_next, y1_next = text_inline_lines[j + 1][1]
                        if not __is_overlaps_y_exceeds_threshold(span['bbox'], (0, y0_next, 0, y1_next)) and 3 * (
                                y1 - y0) > span_y - span_y0:
                            span["type"] = ContentType.InlineEquation
                            span["bbox"][1] = y0
                            span["bbox"][3] = y1
                break
            # 判断当前 span 已经低于文本行，退出循环
            elif span_y < y0 or span_y0 < y0 < span_y and not __is_overlaps_y_exceeds_threshold(span['bbox'],
                                                                                                (0, y0, 0, y1)):
                break
            # 否则继续向下遍历文本行
            else:
                j += 1

    # 返回修改后的 spans 列表
    return spans


# 获取所需的图像、表格和公式列表
def get_qa_need_list(blocks):
    # 创建图像、表格、行间公式和行内公式的副本
    images = []
    tables = []
    interline_equations = []
    inline_equations = []

    # 遍历块
    for block in blocks:
        # 遍历块中的每一行
        for line in block["lines"]:
            # 遍历行中的每个 span
            for span in line["spans"]:
                # 根据类型将 span 添加到相应的列表
                if span["type"] == ContentType.Image:
                    images.append(span)
                elif span["type"] == ContentType.Table:
                    tables.append(span)
                elif span["type"] == ContentType.InlineEquation:
                    inline_equations.append(span)
                elif span["type"] == ContentType.InterlineEquation:
                    interline_equations.append(span)
                else:
                    continue
    # 返回所有收集到的列表
    return images, tables, interline_equations, inline_equations


# 获取所需的图像、表格和行间公式列表（版本2）
def get_qa_need_list_v2(blocks):
    # 创建图像、表格和行间公式的副本
    images = []
    tables = []
    interline_equations = []
    # 遍历每个区块
        for block in blocks:
            # 如果区块类型是图片，则将其添加到图片列表
            if block["type"] == BlockType.Image:
                images.append(block)
            # 如果区块类型是表格，则将其添加到表格列表
            elif block["type"] == BlockType.Table:
                tables.append(block)
            # 如果区块类型是行间方程，则将其添加到行间方程列表
            elif block["type"] == BlockType.InterlineEquation:
                interline_equations.append(block)
        # 返回包含所有图片、表格和行间方程的元组
        return images, tables, interline_equations
```