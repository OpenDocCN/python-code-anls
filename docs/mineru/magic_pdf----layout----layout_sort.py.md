# `.\MinerU\magic_pdf\layout\layout_sort.py`

```
"""对pdf上的box进行layout识别，并对内部组成的box进行排序."""  # 说明模块功能：识别PDF布局中的框并对其进行排序。

from loguru import logger  # 导入日志记录库loguru，用于记录日志信息。

from magic_pdf.layout.bbox_sort import (CONTENT_IDX, CONTENT_TYPE_IDX,  # 从bbox_sort模块导入相关常量。
                                        X0_EXT_IDX, X0_IDX, X1_EXT_IDX, X1_IDX,
                                        Y0_EXT_IDX, Y0_IDX, Y1_EXT_IDX, Y1_IDX,
                                        paper_bbox_sort)  # 导入纸张bbox排序相关功能。

from magic_pdf.layout.layout_det_utils import (  # 从layout_det_utils模块导入布局检测工具函数。
    find_all_bottom_bbox_direct, find_all_left_bbox_direct,  # 导入查找边框的工具函数。
    find_all_right_bbox_direct, find_all_top_bbox_direct,
    find_bottom_bbox_direct_from_left_edge,
    find_bottom_bbox_direct_from_right_edge,
    find_top_bbox_direct_from_left_edge, find_top_bbox_direct_from_right_edge,
    get_left_edge_bboxes, get_right_edge_bboxes)  # 导入获取边缘框的工具函数。

from magic_pdf.libs.boxbase import get_bbox_in_boundary  # 从boxbase模块导入获取边界内框的函数。

LAYOUT_V = 'V'  # 定义垂直布局的标识符。
LAYOUT_H = 'H'  # 定义水平布局的标识符。
LAYOUT_UNPROC = 'U'  # 定义未处理布局的标识符。
LAYOUT_BAD = 'B'  # 定义不良布局的标识符。

def _is_single_line_text(bbox):  # 定义函数，检查给定的框是否仅包含单行文本。
    """检查bbox里面的文字是否只有一行."""  # 函数文档字符串，描述其功能。
    return True  # TODO  # 占位返回，待实现具体逻辑。
    box_type = bbox[CONTENT_TYPE_IDX]  # 获取框的内容类型。
    if box_type != 'text':  # 如果内容类型不是文本。
        return False  # 返回False，表示不是单行文本。
    paras = bbox[CONTENT_IDX]['paras']  # 获取框内段落信息。
    text_content = ''  # 初始化文本内容为空字符串。
    for para_id, para in paras.items():  # 遍历每个段落。
        is_title = para['is_title']  # 获取段落是否为标题的标识。
        if is_title != 0:  # 如果是标题段落。
            text_content += f"## {para['text']}"  # 将标题格式化并添加到文本内容。
        else:  # 如果不是标题段落。
            text_content += para['text']  # 直接添加段落文本。
        text_content += '\n\n'  # 在段落后添加换行符。

    return bbox[CONTENT_TYPE_IDX] == 'text' and len(text_content.split('\n\n')) <= 1  # 返回是否是文本类型且仅有一段内容。

def _horizontal_split(bboxes: list, boundary: tuple, avg_font_size=20) -> list:  # 定义函数，对框进行水平切割。
    """
    对bboxes进行水平切割
    方法是：找到左侧和右侧都没有被直接遮挡的box，然后进行扩展，之后进行切割
    return:
        返回几个大的Layout区域 [[x0, y0, x1, y1, "h|u|v"], ], h代表水平，u代表未探测的，v代表垂直布局
    """  # 函数文档字符串，描述切割方法和返回值格式。
    sorted_layout_blocks = []  # 初始化用于存储最终返回布局的列表。

    bound_x0, bound_y0, bound_x1, bound_y1 = boundary  # 解包边界元组，获取边界坐标。
    all_bboxes = get_bbox_in_boundary(bboxes, boundary)  # 获取在给定边界内的所有框。
    # all_bboxes = paper_bbox_sort(all_bboxes, abs(bound_x1-bound_x0), abs(bound_y1-bound_x0)) # 大致拍下序, 这个是基于直接遮挡的。
    """
    首先在水平方向上扩展独占一行的bbox
    """  # 注释，说明接下来的步骤。

    last_h_split_line_y1 = bound_y0  # 记录下上次的水平分割线的y坐标。
    """
    此时独占一行的被成功扩展到指定的边界上，这个时候利用边界条件合并连续的bbox，成为一个group
    然后合并所有连续水平方向的bbox.
    """  # 注释，说明合并框的策略。

    all_bboxes.sort(key=lambda x: x[Y0_IDX])  # 根据y坐标对所有框进行排序。
    h_bboxes = []  # 初始化用于存储水平框组的列表。
    h_bbox_group = []  # 初始化当前框组为空列表。

    for bbox in all_bboxes:  # 遍历所有框。
        if bbox[X0_EXT_IDX] == bound_x0 and bbox[X1_EXT_IDX] == bound_x1:  # 如果框的左右边界与边界匹配。
            h_bbox_group.append(bbox)  # 将该框添加到当前组。
        else:  # 如果框不在边界内。
            if len(h_bbox_group) > 0:  # 如果当前组不为空。
                h_bboxes.append(h_bbox_group)  # 将当前组添加到水平框组列表。
                h_bbox_group = []  # 重置当前组为空列表。
    # 最后一个group
    if len(h_bbox_group) > 0:  # 如果最后一个框组不为空。
        h_bboxes.append(h_bbox_group)  # 将最后一个框组添加到列表中。
    """
    现在h_bboxes里面是所有的group了，每个group都是一个list
    对h_bboxes里的每个group进行计算放回到sorted_layouts里
    """  # 注释，说明目前存储的框组及后续处理。
    h_layouts = []  # 初始化用于存储水平布局的列表。
    # 遍历所有的水平边界框
    for gp in h_bboxes:
        # 按照每个边界框的y0坐标进行排序
        gp.sort(key=lambda x: x[Y0_IDX])
        # 然后计算这个group的layout_bbox，最小的x0,y0和最大的x1,y1
        x0, y0, x1, y1 = (
            gp[0][X0_EXT_IDX],  # 取第一个边界框的x0坐标
            gp[0][Y0_EXT_IDX],  # 取第一个边界框的y0坐标
            gp[-1][X1_EXT_IDX],  # 取最后一个边界框的x1坐标
            gp[-1][Y1_EXT_IDX],  # 取最后一个边界框的y1坐标
        )
        # 将计算出的布局添加到水平布局列表中
        h_layouts.append([x0, y0, x1, y1, LAYOUT_H])  # 水平的布局
    """
    接下来利用这些连续的水平bbox的layout_bbox的y0, y1，从水平上切分开其余的为几个部分
    """
    # 初始化水平分割线列表，包含上边界y坐标
    h_split_lines = [bound_y0]
    # 遍历所有的水平边界框
    for gp in h_bboxes:  # gp是一个list[bbox_list]
        y0, y1 = gp[0][1], gp[-1][3]  # 获取当前组的y0和y1
        # 将y0和y1添加到分割线列表中
        h_split_lines.append(y0)
        h_split_lines.append(y1)
    # 添加下边界y坐标到分割线列表中
    h_split_lines.append(bound_y1)

    # 初始化未分割边界框列表
    unsplited_bboxes = []
    # 遍历分割线，步长为2
    for i in range(0, len(h_split_lines), 2):
        start_y0, start_y1 = h_split_lines[i : i + 2]  # 取当前的y0和y1
        # 然后找出[start_y0, start_y1]之间的其他bbox，这些组成一个未分割板块
        bboxes_in_block = [
            bbox
            for bbox in all_bboxes  # 遍历所有边界框
            if bbox[Y0_IDX] >= start_y0 and bbox[Y1_IDX] <= start_y1  # 选择在范围内的边界框
        ]
        # 将未分割板块添加到列表中
        unsplited_bboxes.append(bboxes_in_block)
    # 接着把未处理的加入到h_layouts里
    for bboxes_in_block in unsplited_bboxes:
        if len(bboxes_in_block) == 0:  # 如果当前块为空，跳过
            continue
        # 计算当前未分割块的边界框
        x0, y0, x1, y1 = (
            bound_x0,  # 左边界x0
            min([bbox[Y0_IDX] for bbox in bboxes_in_block]),  # 当前块的最小y0
            bound_x1,  # 右边界x1
            max([bbox[Y1_IDX] for bbox in bboxes_in_block]),  # 当前块的最大y1
        )
        # 将未处理的布局添加到水平布局列表中
        h_layouts.append([x0, y0, x1, y1, LAYOUT_UNPROC])

    # 按照y0坐标进行排序，从上到下
    h_layouts.sort(key=lambda x: x[1])  # 按照y0排序, 也就是从上到下的顺序
    """
    转换成如下格式返回
    """
    # 遍历每个布局，转换格式并添加到结果列表中
    for layout in h_layouts:
        sorted_layout_blocks.append(
            {
                'layout_bbox': layout[:4],  # 提取布局边界框
                'layout_label': layout[4],  # 提取布局标签
                'sub_layout': [],  # 初始化子布局为空
            }
        )
    # 返回排序后的布局块
    return sorted_layout_blocks
###############################################################################################
#
#  垂直方向的处理
#
#
###############################################################################################
# 定义一个函数，用于计算垂直方向对齐并分割bboxes为布局
def _vertical_align_split_v1(bboxes: list, boundary: tuple) -> list:
    """
    计算垂直方向上的对齐， 并分割bboxes成layout。负责对一列多行的进行列维度分割。
    如果不能完全分割，剩余部分作为layout_lable为u的layout返回
    -----------------------
    |     |           |
    |     |           |
    |     |           |
    |     |           |
    -------------------------
    此函数会将：以上布局将会切分出来2列
    """
    sorted_layout_blocks = []  # 初始化要返回的布局列表
    new_boundary = [boundary[0], boundary[1], boundary[2], boundary[3]]  # 复制输入边界

    v_blocks = []  # 用于存储垂直块的列表
    """
    先从左到右切分
    """
    while True:  # 开始一个无限循环，直到没有更多可处理的框
        all_bboxes = get_bbox_in_boundary(bboxes, new_boundary)  # 获取在当前边界内的所有bbox
        left_edge_bboxes = get_left_edge_bboxes(all_bboxes)  # 获取所有左边缘bbox
        if len(left_edge_bboxes) == 0:  # 如果没有左边缘bbox，退出循环
            break
        right_split_line_x1 = max([bbox[X1_IDX] for bbox in left_edge_bboxes]) + 1  # 计算右侧分割线的位置
        # 检查分割线是否与其他bbox的左边界相交或重合
        if any(
            [bbox[X0_IDX] <= right_split_line_x1 <= bbox[X1_IDX] for bbox in all_bboxes]
        ):
            # 如果相交，说明无法进行完全的垂直切分，退出循环
            break
        else:  # 否则成功分割出一列
            # 找到左侧边界最靠左的bbox作为layout的x0
            layout_x0 = min(
                [bbox[X0_IDX] for bbox in left_edge_bboxes]
            )  # 计算最左边的边界，以留出间距
            v_blocks.append(
                [
                    layout_x0,
                    new_boundary[1],
                    right_split_line_x1,
                    new_boundary[3],
                    LAYOUT_V,
                ]
            )  # 将新分割的布局块添加到v_blocks中
            new_boundary[0] = right_split_line_x1  # 更新边界，准备下次循环
    """
    再从右到左切， 此时如果还是无法完全切分，那么剩余部分作为layout_lable为u的layout返回
    """
    unsplited_block = []  # 初始化未分割块的列表
    # 无限循环，直到满足退出条件
    while True:
        # 获取在新边界内的所有边界框
        all_bboxes = get_bbox_in_boundary(bboxes, new_boundary)
        # 获取右侧边界的边界框
        right_edge_bboxes = get_right_edge_bboxes(all_bboxes)
        # 如果没有右侧边界的边界框，则退出循环
        if len(right_edge_bboxes) == 0:
            break
        # 找到右侧边界中最小的x坐标，并向左扩展1个单位
        left_split_line_x0 = min([bbox[X0_IDX] for bbox in right_edge_bboxes]) - 1
        # 检查新线是否与其他边界框的左边界相交或重合
        if any(
            [bbox[X0_IDX] <= left_split_line_x0 <= bbox[X1_IDX] for bbox in all_bboxes]
        ):
            # 如果相交，记录未分割的块
            unsplited_block.append(
                [
                    new_boundary[0],
                    new_boundary[1],
                    new_boundary[2],
                    new_boundary[3],
                    LAYOUT_UNPROC,
                ]
            )
            # 退出循环
            break
        else:
            # 否则，找到右侧边界中最右的边界框作为布局的x1
            layout_x1 = max([bbox[X1_IDX] for bbox in right_edge_bboxes])
            # 记录新的垂直块
            v_blocks.append(
                [
                    left_split_line_x0,
                    new_boundary[1],
                    layout_x1,
                    new_boundary[3],
                    LAYOUT_V,
                ]
            )
            # 更新右边界为新的分割线
            new_boundary[2] = left_split_line_x0  # 更新右边界
    """
    # 最后将垂直块拼装成布局格式返回
    """
    # 遍历每个垂直块并添加到排序后的布局块中
    for block in v_blocks:
        sorted_layout_blocks.append(
            {
                'layout_bbox': block[:4],
                'layout_label': block[4],
                'sub_layout': [],
            }
        )
    # 遍历未分割的块并添加到排序后的布局块中
    for block in unsplited_block:
        sorted_layout_blocks.append(
            {
                'layout_bbox': block[:4],
                'layout_label': block[4],
                'sub_layout': [],
            }
        )

    # 根据左边界x0进行排序
    sorted_layout_blocks.sort(key=lambda x: x['layout_bbox'][0])
    # 返回排序后的布局块
    return sorted_layout_blocks
# 定义一个函数，用于改进垂直对齐的盒子分割算法
def _vertical_align_split_v2(bboxes: list, boundary: tuple) -> list:
    # 文档字符串，说明该算法的改进原因和工作原理
    """改进的
    _vertical_align_split算法，原算法会因为第二列的box由于左侧没有遮挡被认为是左侧的一部分，导致整个layout多列被识别为一列。
    利用从左上角的box开始向下看的方法，不断扩展w_x0, w_x1，直到不能继续向下扩展，或者到达边界下边界。"""
    # 初始化一个空列表，用于最终返回的布局块
    sorted_layout_blocks = []  # 这是要最终返回的值
    # 复制边界参数，以便后续使用
    new_boundary = [boundary[0], boundary[1], boundary[2], boundary[3]]
    # 初始化一个空列表，用于存放被割中的盒子
    bad_boxes = []  # 被割中的box
    # 初始化一个空列表，用于存放垂直块
    v_blocks = []
    # 无限循环，直到手动中断
    while True:
        # 获取在新边界内的所有盒子
        all_bboxes = get_bbox_in_boundary(bboxes, new_boundary)
        # 如果没有盒子，退出循环
        if len(all_bboxes) == 0:
            break
        # 找到在所有盒子中左上角的盒子
        left_top_box = min(
            all_bboxes, key=lambda x: (x[X0_IDX], x[Y0_IDX])
        )  # 这里应该加强，检查一下必须是在第一列的 TODO
        # 初始化起始盒子的坐标
        start_box = [
            left_top_box[X0_IDX],
            left_top_box[Y0_IDX],
            left_top_box[X1_IDX],
            left_top_box[Y1_IDX],
        ]
        # 获取左上角盒子的左侧和右侧边界
        w_x0, w_x1 = left_top_box[X0_IDX], left_top_box[X1_IDX]
        """
        然后沿着这个box线向下找最近的那个box, 然后扩展w_x0, w_x1
        扩展之后，宽度会增加，随后用x=w_x1来检测在边界内是否有box与相交，如果相交，那么就说明不能再扩展了。
        当不能扩展的时候就要看是否到达下边界：
        1. 达到，那么更新左边界继续分下一个列
        2. 没有达到，那么此时开始从右侧切分进入下面的循环里
        """
        # 向下查找盒子
        while left_top_box is not None:  # 向下去找
            # 创建一个虚拟盒子以检查扩展
            virtual_box = [w_x0, left_top_box[Y0_IDX], w_x1, left_top_box[Y1_IDX]]
            # 从左侧边缘查找下一个盒子
            left_top_box = find_bottom_bbox_direct_from_left_edge(
                virtual_box, all_bboxes
            )
            # 如果找到了下一个盒子，更新宽度
            if left_top_box:
                w_x0, w_x1 = min(virtual_box[X0_IDX], left_top_box[X0_IDX]), max(
                    [virtual_box[X1_IDX], left_top_box[X1_IDX]]
                )
        # 如果初始盒子在列的中间，向上查找
        start_box = [
            w_x0,
            start_box[Y0_IDX],
            w_x1,
            start_box[Y1_IDX],
        ]  # 扩展一下宽度更鲁棒
        # 从左侧边缘向上查找盒子
        left_top_box = find_top_bbox_direct_from_left_edge(start_box, all_bboxes)
        # 向上查找盒子
        while left_top_box is not None:  # 向上去找
            # 创建一个虚拟盒子以检查扩展
            virtual_box = [w_x0, left_top_box[Y0_IDX], w_x1, left_top_box[Y1_IDX]]
            # 从左侧边缘查找上一个盒子
            left_top_box = find_top_bbox_direct_from_left_edge(virtual_box, all_bboxes)
            # 如果找到了上一个盒子，更新宽度
            if left_top_box:
                w_x0, w_x1 = min(virtual_box[X0_IDX], left_top_box[X0_IDX]), max(
                    [virtual_box[X1_IDX], left_top_box[X1_IDX]]
                )

        # 检查是否与其他盒子相交
        if any([bbox[X0_IDX] <= w_x1 + 1 <= bbox[X1_IDX] for bbox in all_bboxes]):
            # 如果相交，记录下被割中的盒子
            for b in all_bboxes:
                if b[X0_IDX] <= w_x1 + 1 <= b[X1_IDX]:
                    bad_boxes.append([b[X0_IDX], b[Y0_IDX], b[X1_IDX], b[Y1_IDX]])
            # 退出循环
            break
        else:  # 说明成功分割出一列
            # 将分割出的盒子添加到垂直块列表中
            v_blocks.append([w_x0, new_boundary[1], w_x1, new_boundary[3], LAYOUT_V])
            # 更新边界的左侧值
            new_boundary[0] = w_x1  # 更新边界
    """
    接着开始从右上角的box扫描
    """
    # 初始化右侧边界
    w_x0, w_x1 = 0, 0
    # 初始化一个空列表，用于存放未分割的块
    unsplited_block = []
    # 无限循环，直到满足退出条件
    while True:
        # 获取在新边界内的所有边框
        all_bboxes = get_bbox_in_boundary(bboxes, new_boundary)
        # 如果没有找到边框，则退出循环
        if len(all_bboxes) == 0:
            break
        # 根据 X1 值降序排列边框，找出 X1 最大的边框
        bbox_list_sorted = sorted(
            all_bboxes, key=lambda bbox: bbox[X1_IDX], reverse=True
        )
        # 找到 X1 最大的边框的值
        bigest_x1 = bbox_list_sorted[0][X1_IDX]
        # 找到所有 X1 值等于最大值的边框
        boxes_with_bigest_x1 = [
            bbox for bbox in bbox_list_sorted if bbox[X1_IDX] == bigest_x1
        ]  # 这些边框是最靠右的
        # 找到 Y0 值最小的边框，即右上角的边框
        right_top_box = min(
            boxes_with_bigest_x1, key=lambda bbox: bbox[Y0_IDX]
        )  # y0最小的那个
        # 初始化起始边框的坐标
        start_box = [
            right_top_box[X0_IDX],
            right_top_box[Y0_IDX],
            right_top_box[X1_IDX],
            right_top_box[Y1_IDX],
        ]
        # 获取右上角边框的 X0 和 X1 值
        w_x0, w_x1 = right_top_box[X0_IDX], right_top_box[X1_IDX]

        # 继续查找下方边框，直到没有找到为止
        while right_top_box is not None:
            # 创建虚拟边框以查找下方边框
            virtual_box = [w_x0, right_top_box[Y0_IDX], w_x1, right_top_box[Y1_IDX]]
            # 从虚拟边框的右边界寻找下方边框
            right_top_box = find_bottom_bbox_direct_from_right_edge(
                virtual_box, all_bboxes
            )
            # 如果找到新的右上角边框，更新宽度
            if right_top_box:
                w_x0, w_x1 = min([w_x0, right_top_box[X0_IDX]]), max(
                    [w_x1, right_top_box[X1_IDX]]
                )
        # 向上扫描
        start_box = [
            w_x0,
            start_box[Y0_IDX],
            w_x1,
            start_box[Y1_IDX],
        ]  # 扩展一下宽度以提高鲁棒性
        # 从新起始边框向上查找边框
        right_top_box = find_top_bbox_direct_from_right_edge(start_box, all_bboxes)
        # 继续查找上方边框，直到没有找到为止
        while right_top_box is not None:
            # 创建虚拟边框以查找上方边框
            virtual_box = [w_x0, right_top_box[Y0_IDX], w_x1, right_top_box[Y1_IDX]]
            # 从虚拟边框的右边界寻找上方边框
            right_top_box = find_top_bbox_direct_from_right_edge(
                virtual_box, all_bboxes
            )
            # 如果找到新的右上角边框，更新宽度
            if right_top_box:
                w_x0, w_x1 = min([w_x0, right_top_box[X0_IDX]]), max(
                    [w_x1, right_top_box[X1_IDX]]
                )

        # 检查是否与其他边框相交，若有相交则无法完全垂直分割
        if any([bbox[X0_IDX] <= w_x0 - 1 <= bbox[X1_IDX] for bbox in all_bboxes]):
            # 若无法分割，则将新边界添加到未处理块
            unsplited_block.append(
                [
                    new_boundary[0],
                    new_boundary[1],
                    new_boundary[2],
                    new_boundary[3],
                    LAYOUT_UNPROC,
                ]
            )
            # 记录与分割线相交的边框
            for b in all_bboxes:
                if b[X0_IDX] <= w_x0 - 1 <= b[X1_IDX]:
                    bad_boxes.append([b[X0_IDX], b[Y0_IDX], b[X1_IDX], b[Y1_IDX]])
            break
        else:  # 说明成功分割出一列
            # 将分割出的边框添加到垂直块列表
            v_blocks.append([w_x0, new_boundary[1], w_x1, new_boundary[3], LAYOUT_V])
            # 更新新边界的右侧边界
            new_boundary[2] = w_x0
    """转换数据结构"""
    # 将垂直块转换为特定格式
    for block in v_blocks:
        sorted_layout_blocks.append(
            {
                'layout_bbox': block[:4],
                'layout_label': block[4],
                'sub_layout': [],
            }
        )
    # 遍历未分割的块
        for block in unsplited_block:
            # 将每个块的信息添加到已排序布局块列表中
            sorted_layout_blocks.append(
                {
                    # 存储布局边界框的前四个元素
                    'layout_bbox': block[:4],
                    # 存储布局标签
                    'layout_label': block[4],
                    # 初始化子布局为空列表
                    'sub_layout': [],
                    # 记录被割中的盒子
                    'bad_boxes': bad_boxes,  # 记录下来，这个box是被割中的
                }
            )
    
        # 按照布局边界框的x0坐标进行排序
        sorted_layout_blocks.sort(key=lambda x: x['layout_bbox'][0])
        # 返回排序后的布局块列表
        return sorted_layout_blocks
# 尝试进行水平切分，若无法切分则返回一个BAD_LAYOUT
def _try_horizontal_mult_column_split(bboxes: list, boundary: tuple) -> list:
    # 函数主体为空，待实现
    pass


# 从垂直方向进行切割，返回切分后的块
def _vertical_split(bboxes: list, boundary: tuple) -> list:
    # 用于存储最终返回的布局块
    sorted_layout_blocks = []  

    # 解包边界坐标
    bound_x0, bound_y0, bound_x1, bound_y1 = boundary  
    # 获取在边界内的所有 bbox
    all_bboxes = get_bbox_in_boundary(bboxes, boundary)  
    """
    all_bboxes = fix_vertical_bbox_pos(all_bboxes) # 垂直方向解覆盖
    all_bboxes = fix_hor_bbox_pos(all_bboxes)  # 水平解覆盖

    这两行代码目前先不执行，因处理时间过长。它们的作用是：
    如果存在重叠的bbox，则压缩面积较小的box，避免重叠。
    """

    # all_bboxes = paper_bbox_sort(all_bboxes, abs(bound_x1-bound_x0), abs(bound_y1-bound_x0)) # 按大小排序
    """
    在垂直方向上扩展占用一行的 bbox
    """
    # 遍历所有边界内的 bbox
    for bbox in all_bboxes:
        # 找到 bbox 上方的所有 bbox
        top_nearest_bbox = find_all_top_bbox_direct(bbox, all_bboxes)  
        # 找到 bbox 下方的所有 bbox
        bottom_nearest_bbox = find_all_bottom_bbox_direct(bbox, all_bboxes)  
        # 检查 bbox 是否独占一列且与其他 bbox 不重叠
        if (
            top_nearest_bbox is None
            and bottom_nearest_bbox is None
            and not any(
                [
                    b[X0_IDX] < bbox[X1_IDX] < b[X1_IDX]
                    or b[X0_IDX] < bbox[X0_IDX] < b[X1_IDX]
                    for b in all_bboxes
                ]
            )
        ):  # 独占一列, 且不和其他重叠
            # 设置扩展后的 bbox 边界
            bbox[X0_EXT_IDX] = bbox[X0_IDX]  
            bbox[Y0_EXT_IDX] = bound_y0  
            bbox[X1_EXT_IDX] = bbox[X1_IDX]  
            bbox[Y1_EXT_IDX] = bound_y1  
        """
        将独占一列的 bbox 扩展到指定边界，合并连续的 bbox 成为一个 group
        """
    # 按照 x0 坐标对 bbox 进行排序
    all_bboxes.sort(key=lambda x: x[X0_IDX])  
    # 创建存储垂直 bbox 的列表
    v_bboxes = []  
    # 遍历所有的 bbox，筛选出满足条件的 bbox
    for box in all_bboxes:
        if box[Y0_EXT_IDX] == bound_y0 and box[Y1_EXT_IDX] == bound_y1:
            v_bboxes.append(box)  
    """
    v_bboxes 现在包含了所有的 group，每个 group 都是一个列表
    对 v_bboxes 中的每个 group 进行计算，放回到 sorted_layouts 中
    """
    v_layouts = []  
    for vbox in v_bboxes:
        # 计算 group 的布局边界，获得最小的 x0,y0 和最大的 x1,y1
        x0, y0, x1, y1 = (
            vbox[X0_EXT_IDX],
            vbox[Y0_EXT_IDX],
            vbox[X1_EXT_IDX],
            vbox[Y1_EXT_IDX],
        )
        # 将计算出的布局信息添加到 v_layouts 中
        v_layouts.append([x0, y0, x1, y1, LAYOUT_V])  # 垂直的布局
    """
    # 说明接下来将利用连续的垂直边界框的布局边界的x0, x1，从垂直方向切分其余部分为多个部分
    """
    v_split_lines = [bound_x0]  # 初始化垂直切分线列表，包含左边界
    for gp in v_bboxes:  # 遍历所有垂直边界框
        x0, x1 = gp[X0_IDX], gp[X1_IDX]  # 获取每个边界框的左边界x0和右边界x1
        v_split_lines.append(x0)  # 将左边界x0添加到切分线列表
        v_split_lines.append(x1)  # 将右边界x1添加到切分线列表
    v_split_lines.append(bound_x1)  # 添加右边界到切分线列表

    unsplited_bboxes = []  # 初始化未分割的边界框列表
    for i in range(0, len(v_split_lines), 2):  # 每次处理两个切分线
        start_x0, start_x1 = v_split_lines[i : i + 2]  # 获取当前切分区间的x0和x1
        # 找出[start_x0, start_x1]之间的其他边界框，这些组成一个未分割的板块
        bboxes_in_block = [
            bbox  # 遍历所有边界框
            for bbox in all_bboxes  # 从所有边界框中筛选
            if bbox[X0_IDX] >= start_x0 and bbox[X1_IDX] <= start_x1  # 确保边界框在当前切分区间内
        ]
        unsplited_bboxes.append(bboxes_in_block)  # 将找到的未分割边界框添加到列表中
    # 将未处理的边界框加入到v_layouts中
    for bboxes_in_block in unsplited_bboxes:  # 遍历每个未分割的边界框块
        if len(bboxes_in_block) == 0:  # 如果当前块为空，跳过
            continue
        x0, y0, x1, y1 = (  # 计算当前块的边界
            min([bbox[X0_IDX] for bbox in bboxes_in_block]),  # 计算块内所有边界框的最小x0
            bound_y0,  # 使用给定的y0边界
            max([bbox[X1_IDX] for bbox in bboxes_in_block]),  # 计算块内所有边界框的最大x1
            bound_y1,  # 使用给定的y1边界
        )
        v_layouts.append(  # 将计算出的边界框添加到布局列表
            [x0, y0, x1, y1, LAYOUT_UNPROC]  # 说明该区域未能分析出可靠的版面
        )

    v_layouts.sort(key=lambda x: x[0])  # 根据x0对布局进行排序，从左到右

    for layout in v_layouts:  # 遍历已排序的布局
        sorted_layout_blocks.append(  # 将布局信息添加到结果列表
            {
                'layout_bbox': layout[:4],  # 存储布局边界
                'layout_label': layout[4],  # 存储布局标签
                'sub_layout': [],  # 初始化子布局为空
            }
        )
    """
    # 说明到此为止，垂直方向已切分为两种类型，一种是独占一列的，另一种是未处理的。
    # 接下来将对这些未处理的进行垂直方向的切分，以切分出类似“吕”字的垂直布局
    """
    for i, layout in enumerate(sorted_layout_blocks):  # 遍历已排序的布局块
        if layout['layout_label'] == LAYOUT_UNPROC:  # 如果布局标签为未处理
            x0, y0, x1, y1 = layout['layout_bbox']  # 获取布局的边界
            v_split_layouts = _vertical_align_split_v2(bboxes, [x0, y0, x1, y1])  # 对布局进行垂直切分
            sorted_layout_blocks[i] = {  # 更新当前布局块的信息
                'layout_bbox': [x0, y0, x1, y1],  # 更新布局边界
                'layout_label': LAYOUT_H,  # 设置布局标签为水平
                'sub_layout': v_split_layouts,  # 存储子布局
            }
            layout['layout_label'] = LAYOUT_H  # 标记为已被垂直线切分成了水平布局

    return sorted_layout_blocks  # 返回最终的已排序布局块列表
# 定义函数，接受边界框列表、边界元组和页码，返回布局列表
def split_layout(bboxes: list, boundary: tuple, page_num: int) -> list:
    # 函数文档字符串，描述输入输出格式
    """
    把bboxes切割成layout
    return:
    [
        {
            "layout_bbox": [x0,y0,x1,y1],
            "layout_label":"u|v|h|b", 未处理|垂直|水平|BAD_LAYOUT
            "sub_layout":[] #每个元素都是[
                                            x0,y0,
                                            x1,y1,
                                            block_content,
                                            idx_x,idx_y,
                                            content_type,
                                            ext_x0,ext_y0,
                                            ext_x1,ext_y1
                                        ], 并且顺序就是阅读顺序
        }
    ]
    example:
    [
        {
            "layout_bbox": [0, 0, 100, 100],
            "layout_label":"u|v|h|b",
            "sub_layout":[

            ]
        },
        {
            "layout_bbox": [0, 0, 100, 100],
            "layout_label":"u|v|h|b",
            "sub_layout":[
                {
                    "layout_bbox": [0, 0, 100, 100],
                    "layout_label":"u|v|h|b",
                    "content_bboxes":[
                        [],
                        [],
                        []
                    ]
                },
                {
                    "layout_bbox": [0, 0, 100, 100],
                    "layout_label":"u|v|h|b",
                    "sub_layout":[

                    ]
                }
        }
    ]
    """
    # 初始化一个空列表，用于存储最终结果
    sorted_layouts = []  # 最终返回的结果

    # 解包边界元组，分别赋值给四个变量
    boundary_x0, boundary_y0, boundary_x1, boundary_y1 = boundary
    # 如果边界框数量小于等于1，返回一个默认布局
    if len(bboxes) <= 1:
        return [
            {
                'layout_bbox': [boundary_x0, boundary_y0, boundary_x1, boundary_y1],
                'layout_label': LAYOUT_V,  # 默认标签设置为垂直
                'sub_layout': [],  # 子布局为空
            }
        ]
    # 接下来按照先水平后垂直的顺序进行切分
    """
    接下来按照先水平后垂直的顺序进行切分
    """
    # 对边界框进行排序，依据给定边界的宽度和高度
    bboxes = paper_bbox_sort(
        bboxes, boundary_x1 - boundary_x0, boundary_y1 - boundary_y0
    )
    # 通过水平分割函数获得已排序的布局
    sorted_layouts = _horizontal_split(bboxes, boundary)  # 通过水平分割出来的layout
    # 遍历已排序的布局列表，获取每个布局的索引和内容
    for i, layout in enumerate(sorted_layouts):
        # 解构布局边界框的坐标，分别为左上角(x0, y0)和右下角(x1, y1)
        x0, y0, x1, y1 = layout['layout_bbox']
        # 获取当前布局的标签类型
        layout_type = layout['layout_label']
        # 检查布局类型是否为未处理状态，意味着需要进行垂直切分
        if layout_type == LAYOUT_UNPROC:  # 说明是非独占单行的，这些需要垂直切分
            # 进行垂直切分，传入边界框和当前布局的坐标
            v_split_layouts = _vertical_split(bboxes, [x0, y0, x1, y1])
            """
            最后这里有个逻辑问题：如果这个函数只分离出来了一个column layout，那么这个layout分割肯定超出了算法能力范围。因为我们假定的是传进来的
            box已经把行全部剥离了，所以这里必须十多个列才可以。如果只剥离出来一个layout，并且是多个box，那么就说明这个layout是无法分割的，标记为LAYOUT_UNPROC
            """
            # 默认标记为垂直布局
            layout_label = LAYOUT_V
            # 如果只得到了一个垂直切分结果
            if len(v_split_layouts) == 1:
                # 检查这个结果是否没有子布局
                if len(v_split_layouts[0]['sub_layout']) == 0:
                    # 如果没有子布局，标记为未处理布局
                    layout_label = LAYOUT_UNPROC
                    # logger.warning(f"WARNING: pageno={page_num}, 无法分割的layout: ", v_split_layouts)
            """
            组合起来最终的layout
            """
            # 更新当前布局为新的布局信息
            sorted_layouts[i] = {
                'layout_bbox': [x0, y0, x1, y1],  # 维持原有的边界框
                'layout_label': layout_label,      # 设置布局标签
                'sub_layout': v_split_layouts,     # 设置子布局
            }
            # 将当前布局的标签改为水平布局
            layout['layout_label'] = LAYOUT_H
    """
    水平和垂直方向都切分完毕了。此时还有一些未处理的，这些未处理的可能是因为水平和垂直方向都无法切分。
    这些最后调用_try_horizontal_mult_block_split做一次水平多个block的联合切分，如果也不能切分最终就当做BAD_LAYOUT返回
    """
    # TODO  # 这里待实现水平多个块的联合切分逻辑

    # 返回最终处理后的排序布局列表
    return sorted_layouts
# 对给定的框进行布局处理和排序
def get_bboxes_layout(all_boxes: list, boundary: tuple, page_id: int):
    """
    对利用layout排序之后的box，进行排序
    return:
    [
        {
            "layout_bbox": [x0, y0, x1, y1],
            "layout_label":"u|v|h|b", 未处理|垂直|水平|BAD_LAYOUT
        }，
    ]
    """

    # 定义一个前序遍历的辅助函数，用于对布局节点进行排序
    def _preorder_traversal(layout):
        """对sorted_layouts的叶子节点，也就是len(sub_layout)==0的节点进行排序。排序按照前序遍历的顺序，也就是从上到
        下，从左到右的顺序."""
        sorted_layout_blocks = []  # 存储排序后的布局块
        for layout in layout:  # 遍历当前布局
            sub_layout = layout['sub_layout']  # 获取子布局
            if len(sub_layout) == 0:  # 如果没有子布局，说明是叶子节点
                sorted_layout_blocks.append(layout)  # 直接添加到结果中
            else:
                s = _preorder_traversal(sub_layout)  # 递归处理子布局
                sorted_layout_blocks.extend(s)  # 合并结果
        return sorted_layout_blocks  # 返回排序后的布局块

    # -------------------------------------------------------------------------------------------------------------------------
    sorted_layouts = split_layout(
        all_boxes, boundary, page_id
    )  # 将输入框切分为布局，得到一个树形结构
    total_sorted_layout_blocks = _preorder_traversal(sorted_layouts)  # 对切分后的布局进行前序遍历排序
    return total_sorted_layout_blocks, sorted_layouts  # 返回排序后的布局块和原始布局

# 获取布局树的列数
def get_columns_cnt_of_layout(layout_tree):
    """获取一个layout的宽度."""
    max_width_list = [0]  # 初始化一个元素，防止max,min函数报错

    for items in layout_tree:  # 针对每一层（横切）计算列数，横着的算一列
        layout_type = items['layout_label']  # 获取当前布局的类型
        sub_layouts = items['sub_layout']  # 获取子布局
        if len(sub_layouts) == 0:  # 如果没有子布局
            max_width_list.append(1)  # 列数加1
        else:
            if layout_type == LAYOUT_H:  # 如果是水平布局
                max_width_list.append(1)  # 列数加1
            else:
                width = 0  # 初始化宽度计数
                for sub_layout in sub_layouts:  # 遍历子布局
                    if len(sub_layout['sub_layout']) == 0:  # 如果是叶子节点
                        width += 1  # 列数加1
                    else:
                        for lay in sub_layout['sub_layout']:  # 递归计算子布局的列数
                            width += get_columns_cnt_of_layout([lay])
                max_width_list.append(width)  # 将计算的宽度添加到列表中

    return max(max_width_list)  # 返回最大列数

# 对框进行布局排序
def sort_with_layout(bboxes: list, page_width, page_height) -> (list, list):
    """输入是一个bbox的list.

    获取到输入之后，先进行layout切分，然后对这些bbox进行排序。返回排序后的bboxes
    """

    new_bboxes = []  # 初始化新的框列表
    for box in bboxes:  # 遍历所有框
        # new_bboxes.append([box[0], box[1], box[2], box[3], None, None, None, 'text', None, None, None, None])
        new_bboxes.append(  # 将框的信息添加到新列表中
            [
                box[0],
                box[1],
                box[2],
                box[3],
                None,
                None,
                None,
                'text',  # 设置类型为'text'
                None,
                None,
                None,
                None,
                box[4],  # 保留原始框的额外信息
            ]
        )

    layout_bboxes, _ = get_bboxes_layout(  # 获取布局信息
        new_bboxes, tuple([0, 0, page_width, page_height]), 0
    )
    if any([lay['layout_label'] == LAYOUT_UNPROC for lay in layout_bboxes]):  # 检查布局是否未处理
        logger.warning('drop this pdf, reason: 复杂版面')  # 记录警告信息
        return None, None  # 返回空值

    sorted_bboxes = []  # 初始化排序后的框列表
    # 利用layout bbox每次框定一些box，然后排序
    # 遍历所有的布局边界框
    for layout in layout_bboxes:
        # 获取当前布局的边界框
        lbox = layout['layout_bbox']
        # 获取在当前布局边界框内的新边界框
        bbox_in_layout = get_bbox_in_boundary(new_bboxes, lbox)
        # 根据布局边界框的宽度和高度对边界框进行排序
        sorted_bbox = paper_bbox_sort(
            bbox_in_layout, lbox[2] - lbox[0], lbox[3] - lbox[1]
        )
        # 将排序后的边界框添加到结果列表中
        sorted_bboxes.extend(sorted_bbox)

    # 返回排序后的边界框和原始布局边界框
    return sorted_bboxes, layout_bboxes
# 对一页的text_block进行排序，并返回排序后的文本框列表
def sort_text_block(text_block, layout_bboxes):
    # 初始化排序后的文本框列表
    sorted_text_bbox = []
    # 初始化所有文本框列表
    all_text_bbox = []
    # 创建一个从框到文本的映射
    box_to_text = {}
    # 遍历每个文本块
    for blk in text_block:
        # 获取文本块的边界框
        box = blk['bbox']
        # 将边界框坐标与文本块映射在一起
        box_to_text[(box[0], box[1], box[2], box[3])] = blk
        # 将边界框添加到所有文本框列表中
        all_text_bbox.append(box)

    # 按照layout_bboxes的顺序，对text_block进行排序
    for layout in layout_bboxes:
        # 获取当前布局的边界框
        layout_box = layout['layout_bbox']
        # 获取在当前布局边界框内的文本框
        text_bbox_in_layout = get_bbox_in_boundary(
            all_text_bbox,
            [
                layout_box[0] - 1,
                layout_box[1] - 1,
                layout_box[2] + 1,
                layout_box[3] + 1,
            ],
        )
        # 按照y0坐标对布局内的文本框进行排序
        text_bbox_in_layout.sort(
            key=lambda x: x[1]
        )  # 一个layout内部的box，按照y0自上而下排序
        # 遍历排序后的文本框
        for sb in text_bbox_in_layout:
            # 根据边界框坐标从映射中获取对应的文本块，并添加到排序列表中
            sorted_text_bbox.append(box_to_text[(sb[0], sb[1], sb[2], sb[3])])

    # 返回排序后的文本框列表
    return sorted_text_bbox
```