# `.\MinerU\magic_pdf\layout\mcol_sort.py`

```
# 这是一个高级的 PyMuPDF 工具，用于检测多列页面。
"""
This is an advanced PyMuPDF utility for detecting multi-column pages.
"""

# 可以在 shell 脚本中使用，或者将其主函数导入并调用。
"""
It can be used in a shell script, or its main function can be imported and
invoked as descript below.
"""

# 特性部分，描述功能。
"""
Features
---------
"""

# 识别页面上属于（可变数量的）列的文本。
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#
    # 定义函数 can_extend，判断矩形 'temp' 是否可以被 'bb' 扩展
    # 而不与 'bboxlist' 中的矩形相交。
    def can_extend(temp, bb, bboxlist):
        """判断矩形 'temp' 是否可以被 'bb' 扩展
        而不与 'bboxlist' 中的任何矩形相交。

        bboxlist 的项可能为 None，如果它们已被移除。

        返回值：
            如果 'temp' 与 'bboxlist' 中的项没有交集，返回 True。
        """
        # 遍历 bboxlist 中的每个矩形 b
        for b in bboxlist:
            # 判断 temp 与 vert_bboxes 是否不相交，并且 b 为 None 或者等于 bb，或者 temp 与 b 的交集为空
            if not intersects_bboxes(temp, vert_bboxes) and (
                b == None or b == bb or (temp & b).is_empty
            ):
                continue  # 如果条件成立，继续检查下一个矩形
            return False  # 如果有交集，返回 False

        return True  # 所有检查通过，返回 True

    # 定义函数 in_bbox，返回 bbox 中是否包含 bb 的 1 基数索引，否则返回 0
    def in_bbox(bb, bboxes):
        """如果 bbox 包含 bb，返回 1 基数的编号，否则返回 0。"""
        # 遍历 bboxes 的索引和矩形
        for i, bbox in enumerate(bboxes):
            # 如果 bb 在 bbox 中
            if bb in bbox:
                return i + 1  # 返回 1 基数的索引
        return 0  # 没有找到，返回 0

    # 定义函数 intersects_bboxes，检查 bb 是否与 bboxes 中的任何矩形相交
    def intersects_bboxes(bb, bboxes):
        """如果 bbox 与 bb 相交，则返回 True，否则返回 False。"""
        # 遍历 bboxes 中的每个矩形
        for bbox in bboxes:
            # 如果 bb 与 bbox 的交集不为空
            if not (bb & bbox).is_empty:
                return True  # 返回 True，表示相交
        return False  # 所有矩形均不相交，返回 False

    # 定义函数 extend_right，将 bbox 扩展到右侧页面边界
    def extend_right(bboxes, width, path_bboxes, vert_bboxes, img_bboxes):
        """将 bbox 扩展到右侧页面边界。

        当 bbox 右侧没有文本时，将其扩展到右侧页面边界。

        参数：
            bboxes: (list[IRect]) 需要检查的 bboxes
            width: (int) 页面宽度
            path_bboxes: (list[IRect]) 背景颜色的 bboxes
            vert_bboxes: (list[IRect]) 纵向文本的 bboxes
            img_bboxes: (list[IRect]) 图像的 bboxes
        返回值：
            潜在修改后的 bboxes。
        """
        # 遍历 bboxes 的索引和矩形
        for i, bb in enumerate(bboxes):
            # 不扩展有背景颜色的文本
            if in_bbox(bb, path_bboxes):
                continue  # 继续检查下一个矩形

            # 不扩展图像中的文本
            if in_bbox(bb, img_bboxes):
                continue  # 继续检查下一个矩形

            # temp 是扩展到右侧页面边界的 bb
            temp = +bb
            temp.x1 = width  # 将 temp 的右侧设置为页面宽度

            # 不要切割有颜色背景或图像的区域
            if intersects_bboxes(temp, path_bboxes + vert_bboxes + img_bboxes):
                continue  # 继续检查下一个矩形

            # 也不与其他文本 bboxes 相交
            check = can_extend(temp, bb, bboxes)  # 检查 temp 是否可以扩展
            if check:
                bboxes[i] = temp  # 如果可以扩展，用扩展后的矩形替换原矩形

        # 返回不为 None 的 bboxes 列表
        return [b for b in bboxes if b != None]
    # 定义清理块的函数，进行一些基本的清理工作
    def clean_nblocks(nblocks):
        """Do some elementary cleaning."""

        # 1. 移除任何重复的块
        blen = len(nblocks)  # 获取块的数量
        if blen < 2:  # 如果块少于2个，直接返回
            return nblocks
        start = blen - 1  # 设置开始索引为最后一个块
        for i in range(start, -1, -1):  # 从最后一个块向前遍历
            bb1 = nblocks[i]  # 当前块
            bb0 = nblocks[i - 1]  # 上一个块
            if bb0 == bb1:  # 如果当前块和上一个块相同
                del nblocks[i]  # 删除当前块

        # 2. 修复特殊情况下的顺序：
        # 处理底部值几乎相同的连续块，按 x 坐标升序排序
        y1 = nblocks[0].y1  # 获取第一个块的底部坐标
        i0 = 0  # 当前块的索引
        i1 = -1  # 上一个相同底部块的索引

        # 遍历块，识别底部值相近的段落
        # 用其排序版本替换每个段落
        for i in range(1, len(nblocks)):  # 从第二个块开始遍历
            b1 = nblocks[i]  # 当前块
            if abs(b1.y1 - y1) > 10:  # 如果底部值不同
                if i1 > i0:  # 如果段落长度大于1，进行排序
                    nblocks[i0 : i1 + 1] = sorted(
                        nblocks[i0 : i1 + 1], key=lambda b: b.x0
                    )
                y1 = b1.y1  # 存储新的底部值
                i0 = i  # 存储新的开始索引
            i1 = i  # 更新当前索引
        if i1 > i0:  # 如果有待排序的段落
            nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda b: b.x0)
        return nblocks  # 返回清理后的块

    # 提取矢量图形
    for p in paths:  # 遍历路径
        path_rects.append(p["rect"].irect)  # 将路径矩形添加到列表中
    path_bboxes = path_rects  # 将路径矩形赋值给路径边界框

    # 按照升序对路径边界框进行排序，首先按顶部，然后按左侧坐标
    path_bboxes.sort(key=lambda b: (b.y0, b.x0))

    # 页面上的图像边界框，不需要排序
    for item in page.get_images():  # 获取页面中的所有图像
        img_bboxes.extend(page.get_image_rects(item[0]))  # 添加图像矩形到列表中

    # 页面上的文本块
    blocks = page.get_text(
        "dict",  # 以字典格式获取文本
        flags=fitz.TEXTFLAGS_TEXT,  # 设置文本标志
        clip=clip,  # 设置剪辑区域
    )["blocks"]  # 提取文本块

    # 创建块矩形，忽略非水平文本
    for b in blocks:  # 遍历所有文本块
        bbox = fitz.IRect(b["bbox"])  # 获取块的边界框

        # 忽略写在图像上的文本
        if no_image_text and in_bbox(bbox, img_bboxes):  # 如果设置了忽略图像文本并且边界框在图像内
            continue  # 跳过该文本块

        # 确认第一行是水平的
        line0 = b["lines"][0]  # 获取第一行
        if line0["dir"] != (1, 0):  # 只接受水平文本
            vert_bboxes.append(bbox)  # 如果是垂直文本，将边界框添加到垂直边界框列表中
            continue  # 跳过后续处理

        srect = fitz.EMPTY_IRECT()  # 创建一个空的矩形
        for line in b["lines"]:  # 遍历块中的所有行
            lbbox = fitz.IRect(line["bbox"])  # 获取行的边界框
            text = "".join([s["text"].strip() for s in line["spans"]])  # 合并行中所有文本
            if len(text) > 1:  # 如果文本长度大于1
                srect |= lbbox  # 扩展矩形以包括行的边界框
        bbox = +srect  # 将矩形转换为边界框

        if not bbox.is_empty:  # 如果边界框不为空
            bboxes.append(bbox)  # 将边界框添加到边界框列表中

    # 按照升序对文本边界框进行排序，首先按背景，接着按顶部，然后按左侧坐标
    bboxes.sort(key=lambda k: (in_bbox(k, path_bboxes), k.y0, k.x0))

    # 在可能的情况下向右扩展边界框
    # 扩展右侧边界框，返回新的边界框列表
    bboxes = extend_right(
        bboxes, int(page.rect.width), path_bboxes, vert_bboxes, img_bboxes
    )

    # 如果没有找到文本，立即返回空列表
    if bboxes == []:
        return []

    # --------------------------------------------------------------------
    # 将边界框连接以建立某种列结构
    # --------------------------------------------------------------------
    # 初始化页面上的最终边界框列表，预填充第一个边界框
    nblocks = [bboxes[0]]  # pre-fill with first bbox
    # 剩余的旧边界框
    bboxes = bboxes[1:]  # remaining old bboxes

    # 遍历旧边界框
    for i, bb in enumerate(bboxes):  # iterate old bboxes
        # 标志位，表示是否有不需要的连接
        check = False  # indicates unwanted joins

        # 检查 bb 是否可以扩展到新块之一
        for j in range(len(nblocks)):
            nbb = nblocks[j]  # 一个新的块

            # 不允许跨列连接
            if bb == None or nbb.x1 < bb.x0 or bb.x1 < nbb.x0:
                continue

            # 不允许跨不同背景颜色连接
            if in_bbox(nbb, path_bboxes) != in_bbox(bb, path_bboxes):
                continue

            # 暂时扩展新的块
            temp = bb | nbb  # temporary extension of new block
            # 检查是否可以扩展
            check = can_extend(temp, nbb, nblocks)
            if check == True:
                break

        # 如果 bb 不能用于扩展任何新的边界框
        if not check:  # bb cannot be used to extend any of the new bboxes
            # 将其添加到列表中
            nblocks.append(bb)  # so add it to the list
            # 获取新添加的边界框的索引
            j = len(nblocks) - 1  # index of it
            # 获取新添加的边界框
            temp = nblocks[j]  # new bbox added

        # 检查是否有剩余的边界框包含在 temp 中
        check = can_extend(temp, bb, bboxes)
        # 如果不能扩展，添加 bb
        if check == False:
            nblocks.append(bb)
        else:
            # 否则更新 nblocks 中的元素
            nblocks[j] = temp
        # 标记 bboxes 中的当前元素为 None
        bboxes[i] = None

    # 进行一些基本的清理
    nblocks = clean_nblocks(nblocks)

    # 返回识别出的文本边界框
    return nblocks
if __name__ == "__main__":  # 检查是否为主程序执行
    """Only for debugging purposes, currently.

    Draw red borders around the returned text bboxes and insert
    the bbox number.
    Then save the file under the name "input-blocks.pdf".
    """  # 文档字符串，说明当前代码用于调试，绘制文本边界框并保存为新文件

    # get the file name  # 获取文件名
    filename = sys.argv[1]  # 从命令行参数中获取输入文件名

    # check if footer margin is given  # 检查是否提供了页脚边距
    if len(sys.argv) > 2:  # 如果命令行参数数量大于2
        footer_margin = int(sys.argv[2])  # 将页脚边距参数转换为整数
    else:  # use default value  # 否则使用默认值
        footer_margin = 50  # 设置默认页脚边距为50

    # check if header margin is given  # 检查是否提供了页眉边距
    if len(sys.argv) > 3:  # 如果命令行参数数量大于3
        header_margin = int(sys.argv[3])  # 将页眉边距参数转换为整数
    else:  # use default value  # 否则使用默认值
        header_margin = 50  # 设置默认页眉边距为50

    # open document  # 打开文档
    doc = fitz.open(filename)  # 使用fitz库打开指定的PDF文件

    # iterate over the pages  # 遍历文档中的每一页
    for page in doc:  # 对每一页执行循环
        # remove any geometry issues  # 解决任何几何问题
        page.wrap_contents()  # 包装页面内容以修复几何问题

        # get the text bboxes  # 获取文本边界框
        bboxes = column_boxes(page, footer_margin=footer_margin, header_margin=header_margin)  # 根据页脚和页眉边距获取文本边界框

        # prepare a canvas to draw rectangles and text  # 准备一个画布以绘制矩形和文本
        shape = page.new_shape()  # 创建新的形状对象用于绘图

        # iterate over the bboxes  # 遍历所有的边界框
        for i, rect in enumerate(bboxes):  # 通过枚举获取边界框及其索引
            shape.draw_rect(rect)  # draw a border  # 绘制边界矩形

            # write sequence number  # 写入序列号
            shape.insert_text(rect.tl + (5, 15), str(i), color=fitz.pdfcolor["red"])  # 在边界框的左上角插入序列号

        # finish drawing / text with color red  # 完成绘制，使用红色
        shape.finish(color=fitz.pdfcolor["red"])  # 结束绘图并设置颜色为红色
        shape.commit()  # store to the page  # 将绘制的形状保存到页面

    # save document with text bboxes  # 保存带有文本边界框的文档
    doc.ez_save(filename.replace(".pdf", "-blocks.pdf"))  # 将文件保存为新文件，名称后缀为“-blocks.pdf”
```