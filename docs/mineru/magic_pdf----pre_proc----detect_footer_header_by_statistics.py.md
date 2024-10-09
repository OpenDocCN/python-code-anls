# `.\MinerU\magic_pdf\pre_proc\detect_footer_header_by_statistics.py`

```
# 从 collections 模块导入 defaultdict，用于创建字典并提供默认值
from collections import defaultdict

# 从 magic_pdf.libs.boxbase 模块导入 calculate_iou 函数，用于计算 IOU
from magic_pdf.libs.boxbase import calculate_iou


# 比较一个边界框与边界框列表，判断是否有相似的边界框
def compare_bbox_with_list(bbox, bbox_list, tolerance=1):
    # 检查 bbox 是否与 bbox_list 中的任意边界框在容差范围内相等
    return any(all(abs(a - b) < tolerance for a, b in zip(bbox, common_bbox)) for common_bbox in bbox_list)

# 判断给定的块是否为单行块
def is_single_line_block(block):
    # 根据块的宽度和高度进行判断
    block_width = block["X1"] - block["X0"]  # 计算块的宽度
    block_height = block["bbox"][3] - block["bbox"][1]  # 计算块的高度

    # 如果块的高度接近平均字符高度且宽度较大，则认为是单行块
    return block_height <= block["avg_char_height"] * 3 and block_width > block["avg_char_width"] * 3

# 获取最常见的边界框
def get_most_common_bboxes(bboxes, page_height, position="top", threshold=0.25, num_bboxes=3, min_frequency=2):
    """
    此函数从 bboxes 中获取最常见的边界框

    参数
    ----------
    bboxes : list
        边界框列表
    page_height : float
        页面高度
    position : str, optional
        "top" 或 "bottom"，默认是 "top"
    threshold : float, optional
        阈值，默认是 0.25
    num_bboxes : int, optional
        返回的边界框数量，默认是 3
    min_frequency : int, optional
        边界框的最小频率，默认是 2

    返回
    -------
    common_bboxes : list
        常见的边界框
    """
    # 根据位置过滤边界框
    if position == "top":
        filtered_bboxes = [bbox for bbox in bboxes if bbox[1] < page_height * threshold]  # 筛选顶部边界框
    else:
        filtered_bboxes = [bbox for bbox in bboxes if bbox[3] > page_height * (1 - threshold)]  # 筛选底部边界框

    # 计算最常见的边界框
    bbox_count = defaultdict(int)  # 创建一个默认值为 0 的字典
    for bbox in filtered_bboxes:
        bbox_count[tuple(bbox)] += 1  # 更新边界框计数

    # 获取最频繁出现的边界框，且频率需超过 min_frequency
    common_bboxes = [
        bbox for bbox, count in sorted(bbox_count.items(), key=lambda item: item[1], reverse=True) if count >= min_frequency
    ][:num_bboxes]  # 按频率排序并取出前 num_bboxes 个
    return common_bboxes  # 返回常见边界框列表

# 检测文档的页眉和页脚
def detect_footer_header2(result_dict, similarity_threshold=0.5):
    """
    此函数检测文档的页眉和页脚。

    参数
    ----------
    result_dict : dict
        结果字典

    返回
    -------
    result_dict : dict
        结果字典
    """
    # 遍历文档中的所有块
    single_line_blocks = 0  # 单行块计数器
    total_blocks = 0  # 总块计数器

    for page_id, blocks in result_dict.items():  # 遍历每一页的块
        if page_id.startswith("page_"):  # 确保是页码
            for block_key, block in blocks.items():  # 遍历每个块
                if block_key.startswith("block_"):  # 确保是块
                    total_blocks += 1  # 总块计数加 1
                    if is_single_line_block(block):  # 检查是否为单行块
                        single_line_blocks += 1  # 单行块计数加 1

    # 如果没有块，则跳过页眉和页脚检测
    if total_blocks == 0:
        print("No blocks found. Skipping header/footer detection.")  # 打印提示信息
        return result_dict  # 返回原始结果字典
    # 如果大多数块是单行，则跳过头部和底部的检测
    if single_line_blocks / total_blocks > 0.5:  # 50% 的块是单行
        # print("跳过文本密集型文档的头部/底部检测。")
        return result_dict  # 返回结果字典

    # 收集所有块的边界框
    all_bboxes = []  # 存储所有块的边界框
    all_texts = []  # 存储所有块的文本

    # 遍历结果字典中的每一页和对应的块
    for page_id, blocks in result_dict.items():
        if page_id.startswith("page_"):  # 确保是以 "page_" 开头的页面
            for block_key, block in blocks.items():
                if block_key.startswith("block_"):  # 确保是以 "block_" 开头的块
                    all_bboxes.append(block["bbox"])  # 将边界框添加到列表中

    # 获取页面的高度
    page_height = max(bbox[3] for bbox in all_bboxes)  # 从所有边界框中获取最大高度

    # 获取头部和底部的最常见边界框列表
    common_header_bboxes = get_most_common_bboxes(all_bboxes, page_height, position="top") if all_bboxes else []  # 获取头部边界框
    common_footer_bboxes = get_most_common_bboxes(all_bboxes, page_height, position="bottom") if all_bboxes else []  # 获取底部边界框

    # 检测并标记头部和底部
    for page_id, blocks in result_dict.items():
        if page_id.startswith("page_"):  # 确保是以 "page_" 开头的页面
            for block_key, block in blocks.items():
                if block_key.startswith("block_"):  # 确保是以 "block_" 开头的块
                    bbox = block["bbox"]  # 获取当前块的边界框
                    text = block["text"]  # 获取当前块的文本

                    is_header = compare_bbox_with_list(bbox, common_header_bboxes)  # 检查当前块是否是头部
                    is_footer = compare_bbox_with_list(bbox, common_footer_bboxes)  # 检查当前块是否是底部
                    block["is_header"] = int(is_header)  # 将结果转换为整数并存储
                    block["is_footer"] = int(is_footer)  # 将结果转换为整数并存储

    return result_dict  # 返回更新后的结果字典
# 获取页面大小的函数，接受页面尺寸列表作为输入
def __get_page_size(page_sizes:list):
    """
    页面大小可能不一样
    """
    # 计算页面宽度的平均值
    w = sum([w for w,h in page_sizes])/len(page_sizes)
    # 计算页面高度的平均值
    h = sum([h for w,h  in page_sizes])/len(page_sizes)
    # 返回计算得到的宽度和高度
    return w, h

# 计算两个边界框（bbox）之间的 IOU（交并比）
def __calculate_iou(bbox1, bbox2):
    # 调用外部函数计算 IOU 值
    iou = calculate_iou(bbox1, bbox2)
    # 返回计算得到的 IOU 值
    return iou

# 判断两个边界框是否在同一位置，基于 IOU 阈值
def __is_same_pos(box1, box2, iou_threshold):
    # 计算两个边界框之间的 IOU 值
    iou = __calculate_iou(box1, box2)
    # 判断 IOU 是否大于等于阈值
    return iou >= iou_threshold

# 获取最常见的边界框
def get_most_common_bbox(bboxes:list, page_size:list, page_cnt:int, page_range_threshold=0.2, iou_threshold=0.9):
    """
    common bbox必须大于page_cnt的1/3
    """
    # 设置最小出现次数为总页面数的四分之一或 3，取较大值
    min_occurance_cnt = max(3, page_cnt//4)
    # 初始化头部和底部检测边界框列表
    header_det_bbox = []
    footer_det_bbox = []
    
    # 用于存储同一位置的头部和底部边界框组
    hdr_same_pos_group = []
    btn_same_pos_group = []
    
    # 获取页面的宽度和高度
    page_w, page_h = __get_page_size(page_size)
    # 计算页面上部和下部的阈值
    top_y, bottom_y = page_w*page_range_threshold, page_h*(1-page_range_threshold)
    
    # 获取位于页面上部的边界框
    top_bbox = [b for b in bboxes if b[3]<top_y]
    # 获取位于页面下部的边界框
    bottom_bbox = [b for b in bboxes if b[1]>bottom_y]
    # 然后开始排序，寻找最经常出现的 bbox, 寻找的时候如果 IOU > iou_threshold 就算是一个
    for i in range(0, len(top_bbox)):
        # 将当前上部边界框添加到同一位置组中
        hdr_same_pos_group.append([top_bbox[i]])
        for j in range(i+1, len(top_bbox)):
            # 检查两个边界框是否在同一位置
            if __is_same_pos(top_bbox[i], top_bbox[j], iou_threshold):
                # 如果是，则将第二个边界框添加到当前组中
                hdr_same_pos_group[i].append(top_bbox[j])
                
    for i in range(0, len(bottom_bbox)):
        # 将当前下部边界框添加到同一位置组中
        btn_same_pos_group.append([bottom_bbox[i]])
        for j in range(i+1, len(bottom_bbox)):
            # 检查两个边界框是否在同一位置
            if __is_same_pos(bottom_bbox[i], bottom_bbox[j], iou_threshold):
                # 如果是，则将第二个边界框添加到当前组中
                btn_same_pos_group[i].append(bottom_bbox[j])
                
    # 然后看下每一组的 bbox，是否符合大于 page_cnt 一定比例
    hdr_same_pos_group = [g for g in hdr_same_pos_group if len(g)>=min_occurance_cnt]
    btn_same_pos_group = [g for g in btn_same_pos_group if len(g)>=min_occurance_cnt]
    
    # 平铺两个 list[list]
    hdr_same_pos_group = [bbox for g in hdr_same_pos_group for bbox in g]
    btn_same_pos_group = [bbox for g in btn_same_pos_group for bbox in g]
    # 寻找 hdr_same_pos_group 中的 box[3] 最大值，btn_same_pos_group 中的 box[1] 最小值
    hdr_same_pos_group.sort(key=lambda b:b[3])
    btn_same_pos_group.sort(key=lambda b:b[1])
    
    # 获取头部边界框的最大 y 值
    hdr_y = hdr_same_pos_group[-1][3] if hdr_same_pos_group else 0
    # 获取底部边界框的最小 y 值
    btn_y = btn_same_pos_group[0][1] if btn_same_pos_group else page_h
    
    # 设置头部和底部检测边界框
    header_det_bbox = [0, 0, page_w, hdr_y]
    footer_det_bbox = [0, btn_y, page_w, page_h]
    # logger.warning(f"header: {header_det_bbox}, footer: {footer_det_bbox}")
    # 返回头部和底部边界框以及页面宽高
    return header_det_bbox, footer_det_bbox, page_w, page_h
    
# 删除页眉和页脚的函数
def drop_footer_header(pdf_info_dict:dict):
    """
    启用规则探测,在全局的视角上通过统计的方法。
    """
    # 初始化 header 和 footer 列表，用于存储页眉和页脚的边界框
    header = []
    footer = []
    
    # 从 pdf_info_dict 中提取所有文本块的边界框
    all_text_bboxes = [blk['bbox'] for _, val in pdf_info_dict.items() for blk in val['preproc_blocks']]
    # 从 pdf_info_dict 中提取所有图片的边界框，以及备份图片的边界框
    image_bboxes = [img['bbox'] for _, val in pdf_info_dict.items() for img in val['images']] + [img['bbox'] for _, val in pdf_info_dict.items() for img in val['image_backup']]
    # 从 pdf_info_dict 中提取每页的大小
    page_size = [val['page_size'] for _, val in pdf_info_dict.items()]
    # 计算 PDF 的总页数
    page_cnt = len(pdf_info_dict.keys()) # 一共多少页
    # 获取最常见的边界框信息，包括页眉、页脚和页面宽高
    header, footer, page_w, page_h = get_most_common_bbox(all_text_bboxes+image_bboxes, page_size, page_cnt)
    
    """"
    # 将页眉的范围扩展到页面的整个水平
    """        
    if header:
        # 设置页眉的边界框，顶部位置加一像素
        header = [0, 0, page_w, header[3]+1]
        
    if footer:
        # 设置页脚的边界框，底部位置减一像素
        footer = [0, footer[1]-1, page_w, page_h]
        
    # 找到 footer 和 header 的范围后，针对每一页 PDF，从文本和图片中删除这些范围内的内容
    # 移除文本块
    # 遍历 pdf_info_dict 字典中的每个页面信息
    for _, page_info in pdf_info_dict.items():
        # 初始化用于存储头部文本块的列表
        header_text_blk = []
        # 初始化用于存储底部文本块的列表
        footer_text_blk = []
        # 遍历页面的预处理文本块
        for blk in page_info['preproc_blocks']:
            # 获取当前文本块的边界框
            blk_bbox = blk['bbox']
            # 如果有头部信息并且文本块的底部在头部上方，则标记为头部
            if header and blk_bbox[3] <= header[3]:
                blk['tag'] = "header"
                # 将头部文本块添加到列表中
                header_text_blk.append(blk)
            # 如果有底部信息并且文本块的顶部在底部下方，则标记为底部
            elif footer and blk_bbox[1] >= footer[1]:
                blk['tag'] = "footer"
                # 将底部文本块添加到列表中
                footer_text_blk.append(blk)
                
        # 将头部文本块添加到页面信息的 dropped_text_block 中
        page_info['droped_text_block'].extend(header_text_blk)
        # 将底部文本块添加到页面信息的 dropped_text_block 中
        page_info['droped_text_block'].extend(footer_text_blk)
        
        # 从页面的预处理块中移除头部文本块
        for blk in header_text_blk:
            page_info['preproc_blocks'].remove(blk)
        # 从页面的预处理块中移除底部文本块
        for blk in footer_text_blk:
            page_info['preproc_blocks'].remove(blk)
            
        """接下来把footer、header上的图片也删除掉。图片包括正常的和backup的"""
        # 初始化用于存储头部图片的列表
        header_image = []
        # 初始化用于存储底部图片的列表
        footer_image = []
        
        # 遍历页面的图片信息
        for image_info in page_info['images']:
            # 获取当前图片的边界框
            img_bbox = image_info['bbox']
            # 如果有头部信息并且图片的底部在头部上方，则标记为头部
            if header and img_bbox[3] <= header[3]:
                image_info['tag'] = "header"
                # 将头部图片添加到列表中
                header_image.append(image_info)
            # 如果有底部信息并且图片的顶部在底部下方，则标记为底部
            elif footer and img_bbox[1] >= footer[1]:
                image_info['tag'] = "footer"
                # 将底部图片添加到列表中
                footer_image.append(image_info)
                
        # 将头部图片添加到页面信息的 dropped_image_block 中
        page_info['droped_image_block'].extend(header_image)
        # 将底部图片添加到页面信息的 dropped_image_block 中
        page_info['droped_image_block'].extend(footer_image)
        
        # 从页面的图片列表中移除头部图片
        for img in header_image:
            page_info['images'].remove(img)
        # 从页面的图片列表中移除底部图片
        for img in footer_image:
            page_info['images'].remove(img)
            
        """接下来把backup的图片也删除掉"""
        # 初始化用于存储头部备份图片的列表
        header_image = []
        # 初始化用于存储底部备份图片的列表
        footer_image = []
        
        # 遍历页面的备份图片信息
        for image_info in page_info['image_backup']:
            # 获取当前备份图片的边界框
            img_bbox = image_info['bbox']
            # 如果有头部信息并且备份图片的底部在头部上方，则标记为头部
            if header and img_bbox[3] <= header[3]:
                image_info['tag'] = "header"
                # 将头部备份图片添加到列表中
                header_image.append(image_info)
            # 如果有底部信息并且备份图片的顶部在底部下方，则标记为底部
            elif footer and img_bbox[1] >= footer[1]:
                image_info['tag'] = "footer"
                # 将底部备份图片添加到列表中
                footer_image.append(image_info)
                
        # 将头部备份图片添加到页面信息的 dropped_image_block 中
        page_info['droped_image_block'].extend(header_image)
        # 将底部备份图片添加到页面信息的 dropped_image_block 中
        page_info['droped_image_block'].extend(footer_image)
        
        # 从页面的备份图片列表中移除头部备份图片
        for img in header_image:
            page_info['image_backup'].remove(img)
        # 从页面的备份图片列表中移除底部备份图片
        for img in footer_image:
            page_info['image_backup'].remove(img)
            
    # 返回头部和底部信息
    return header, footer
```