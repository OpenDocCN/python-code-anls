# `.\MinerU\magic_pdf\pre_proc\detect_equation.py`

```
# 从指定库中导入判断包含关系和计算重叠区域与最小框面积比的函数
from magic_pdf.libs.boxbase import _is_in, calculate_overlap_area_2_minbox_area_ratio              # 正则
# 从 pyMuPDF 库中导入 fitz，用于处理 PDF 文档
from magic_pdf.libs.commons import fitz             # pyMuPDF库


# 定义一个函数，判断给定的边界框列表中的包含关系
def __solve_contain_bboxs(all_bbox_list: list):

    """将两个公式的bbox做判断是否有包含关系，若有的话则删掉较小的bbox"""

    # 初始化一个空列表，用于存储需要删除的边界框
    dump_list = []
    # 遍历边界框列表
    for i in range(len(all_bbox_list)):
        for j in range(i + 1, len(all_bbox_list)):
            # 获取当前两个边界框的前四个坐标
            bbox1 = all_bbox_list[i][:4]
            bbox2 = all_bbox_list[j][:4]
            
            # 判断bbox1是否在bbox2内，若是则将bbox1添加到待删除列表
            if _is_in(bbox1, bbox2):
                dump_list.append(all_bbox_list[i])
            # 判断bbox2是否在bbox1内，若是则将bbox2添加到待删除列表
            elif _is_in(bbox2, bbox1):
                dump_list.append(all_bbox_list[j])
            else:
                # 计算两个边界框的重叠区域与最小框面积的比率
                ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
                # 如果重叠比率大于0.7
                if ratio > 0.7:
                    # 计算bbox1的面积
                    s1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) 
                    # 计算bbox2的面积
                    s2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    # 如果bbox2的面积大于bbox1，则将bbox1添加到待删除列表
                    if s2 > s1:  
                        dump_list.append(all_bbox_list[i])
                    else:
                        # 否则将bbox1添加到待删除列表
                        dump_list.append(all_bbox_list[i]) 

    # 遍历需要删除的边界框列表中的每个元素
    for item in dump_list:
        # 只要item在all_bbox_list中，就将其移除
        while item in all_bbox_list:
            all_bbox_list.remove(item)
    # 返回处理后的边界框列表
    return all_bbox_list


# 定义解析方程的函数
def parse_equations(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """
    # 设置每英寸的分辨率为72
    DPI = 72  # use this resolution
    # 获取当前页面的像素映射
    pix = page.get_pixmap(dpi=DPI)
    # 定义页面的左边界和右边界
    pageL = 0
    pageR = int(pix.w)
    # 定义页面的上边界和下边界
    pageU = 0
    pageD = int(pix.h)
    

    #--------- 通过json_from_DocXchain来获取 table ---------#
    # 初始化两个空列表，用于存储提取的边界框
    equationEmbedding_from_DocXChain_bboxs = []
    equationIsolated_from_DocXChain_bboxs = []
    
    # 将传入的JSON对象赋值给xf_json
    xf_json = json_from_DocXchain_obj
    # 从JSON中提取页面宽度
    width_from_json = xf_json['page_info']['width']
    # 从JSON中提取页面高度
    height_from_json = xf_json['page_info']['height']
    # 计算左右方向的缩放比率
    LR_scaleRatio = width_from_json / (pageR - pageL)
    # 计算上下方向的缩放比率
    UD_scaleRatio = height_from_json / (pageD - pageU)
    
    # 遍历JSON中的布局检测信息
    for xf in xf_json['layout_dets']:
    # 该注释说明了每个索引对应的文本类型
    # {0: 'title', 1: 'figure', 2: 'plain text', 3: 'header', 4: 'page number', 5: 'footnote', 6: 'footer', 7: 'table', 8: 'table caption', 9: 'figure caption', 10: 'equation', 11: 'full column', 12: 'sub column'}
        # 从 xf 字典中获取多边形的左边界，并根据 LR_scaleRatio 进行缩放
        L = xf['poly'][0] / LR_scaleRatio
        # 从 xf 字典中获取多边形的上边界，并根据 UD_scaleRatio 进行缩放
        U = xf['poly'][1] / UD_scaleRatio
        # 从 xf 字典中获取多边形的右边界，并根据 LR_scaleRatio 进行缩放
        R = xf['poly'][2] / LR_scaleRatio
        # 从 xf 字典中获取多边形的下边界，并根据 UD_scaleRatio 进行缩放
        D = xf['poly'][5] / UD_scaleRatio
        # 调整 L 和 R 的值，考虑到某些页面的 artBox 可能偏移（注释掉的代码）
        # L += pageL          
        # R += pageL
        # 调整 U 和 D 的值，考虑到某些页面的 artBox 可能偏移（注释掉的代码）
        # U += pageU
        # D += pageU
        # 计算并更新左边界 L 和 右边界 R 的最小和最大值
        L, R = min(L, R), max(L, R)
        # 计算并更新上边界 U 和 下边界 D 的最小和最大值
        U, D = min(U, D), max(U, D)
        # 创建公式的文件名，包含页面ID及边界坐标
        img_suffix = f"{page_ID}_{int(L)}_{int(U)}_{int(R)}_{int(D)}"
        # 如果类别为13且得分大于等于0.3，则处理嵌入的公式
        if xf['category_id'] == 13 and xf['score'] >= 0.3:      
            # 从字典中获取LaTeX文本，如果不存在则使用默认值
            latex_text = xf.get("latex", "EmptyInlineEquationResult")
            # 生成可调试的LaTeX文本，包含公式文件名后缀
            debugable_latex_text = f"{latex_text}|{img_suffix}"
            # 将公式边界和LaTeX文本添加到列表中
            equationEmbedding_from_DocXChain_bboxs.append((L, U, R, D, latex_text))
        # 如果类别为14且得分大于等于0.3，则处理孤立的公式
        if xf['category_id'] == 14 and xf['score'] >= 0.3:
            # 从字典中获取LaTeX文本，如果不存在则使用默认值
            latex_text = xf.get("latex", "EmptyInterlineEquationResult")
            # 生成可调试的LaTeX文本，包含公式文件名后缀
            debugable_latex_text = f"{latex_text}|{img_suffix}"
            # 将公式边界和LaTeX文本添加到列表中
            equationIsolated_from_DocXChain_bboxs.append((L, U, R, D, latex_text))
    
    #---------------------------------------- 排序，编号，保存 -----------------------------------------#
    # 对孤立公式的边界进行排序，首先按上边界U排序，然后按左边界L排序
    equationIsolated_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 对孤立公式的边界进行再次排序（可能是多余的）
    equationIsolated_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    
    # 初始化嵌入公式名称列表
    equationEmbedding_from_DocXChain_names = []
    # 初始化嵌入公式ID计数器
    equationEmbedding_ID = 0
    
    # 初始化孤立公式名称列表
    equationIsolated_from_DocXChain_names = []
    # 初始化孤立公式ID计数器
    equationIsolated_ID = 0
    
    # 遍历嵌入公式的边界列表
    for L, U, R, D, _ in equationEmbedding_from_DocXChain_bboxs:
        # 检查边界是否有效（左边界小于右边界且上边界小于下边界）
        if not(L < R and U < D):
            continue
        try:
            # 获取指定边界的图像快照（注释掉的代码）
            # cur_equation = page.get_pixmap(clip=(L,U,R,D))
            # 生成嵌入公式的文件名
            new_equation_name = "equationEmbedding_{}_{}.png".format(page_ID, equationEmbedding_ID)        # 公式name
            # 保存图像快照到指定文件夹（注释掉的代码）
            # cur_equation.save(res_dir_path + '/' + new_equation_name)                       # 把公式存出在新建的文件夹，并命名
            # 将生成的文件名添加到嵌入公式名称列表中
            equationEmbedding_from_DocXChain_names.append(new_equation_name)                         # 把公式的名字存在list中，方便在md中插入引用
            # 增加嵌入公式ID计数
            equationEmbedding_ID += 1
        except:
            # 如果发生异常，则跳过当前公式
            pass

    # 遍历孤立公式的边界列表
    for L, U, R, D, _ in equationIsolated_from_DocXChain_bboxs:
        # 检查边界是否有效（左边界小于右边界且上边界小于下边界）
        if not(L < R and U < D):
            continue
        try:
            # 获取指定边界的图像快照（注释掉的代码）
            # cur_equation = page.get_pixmap(clip=(L,U,R,D))
            # 生成孤立公式的文件名
            new_equation_name = "equationEmbedding_{}_{}.png".format(page_ID, equationIsolated_ID)        # 公式name
            # 保存图像快照到指定文件夹（注释掉的代码）
            # cur_equation.save(res_dir_path + '/' + new_equation_name)                       # 把公式存出在新建的文件夹，并命名
            # 将生成的文件名添加到孤立公式名称列表中
            equationIsolated_from_DocXChain_names.append(new_equation_name)                         # 把公式的名字存在list中，方便在md中插入引用
            # 增加孤立公式ID计数
            equationIsolated_ID += 1
        except:
            # 如果发生异常，则跳过当前公式
            pass
    
    # 对嵌入公式的边界进行排序，首先按上边界U排序，然后按左边界L排序
    equationEmbedding_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 对 equationIsolated_from_DocXChain_bboxs 列表进行排序，排序的依据是每个元素的第二个和第一个值
    equationIsolated_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    
    # 文档注释：根据 PDF 可视区域，调整 bbox 的坐标
    cropbox = page.cropbox  # 获取页面的裁剪框坐标
    # 检查裁剪框的左下角坐标是否与页面的矩形坐标不同
    if cropbox[0]!=page.rect[0] or cropbox[1]!=page.rect[1]:
        # 遍历 equationEmbedding_from_DocXChain_bboxs 列表中的每个边界框
        for eq_box in equationEmbedding_from_DocXChain_bboxs:
            # 调整边界框的坐标，增加裁剪框的坐标偏移
            eq_box = [eq_box[0]+cropbox[0], eq_box[1]+cropbox[1], eq_box[2]+cropbox[0], eq_box[3]+cropbox[1], eq_box[4]]
        # 遍历 equationIsolated_from_DocXChain_bboxs 列表中的每个边界框
        for eq_box in equationIsolated_from_DocXChain_bboxs:
            # 调整边界框的坐标，增加裁剪框的坐标偏移
            eq_box = [eq_box[0]+cropbox[0], eq_box[1]+cropbox[1], eq_box[2]+cropbox[0], eq_box[3]+cropbox[1], eq_box[4]]
        
    # 调用函数 __solve_contain_bboxs，去重处理 equationEmbedding_from_DocXChain_bboxs 中的边界框
    deduped_embedding_eq_bboxes = __solve_contain_bboxs(equationEmbedding_from_DocXChain_bboxs)
    # 返回去重后的边界框和原始的孤立边界框列表
    return deduped_embedding_eq_bboxes, equationIsolated_from_DocXChain_bboxs
```