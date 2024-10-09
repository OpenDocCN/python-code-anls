# `.\MinerU\magic_pdf\pre_proc\detect_tables.py`

```
# 从 magic_pdf.libs.commons 模块导入 fitz（pyMuPDF库）
from magic_pdf.libs.commons import fitz             # pyMuPDF库


# 定义解析 PDF 页面的表格的函数，接收页面ID、页面对象和 JSON 数据
def parse_tables(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_ID页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """
    # 设置分辨率为72 DPI
    DPI = 72  # use this resolution
    # 根据当前页面生成位图（pixmap），指定分辨率
    pix = page.get_pixmap(dpi=DPI)
    # 初始化页面的左边界
    pageL = 0
    # 初始化页面的右边界，取位图宽度
    pageR = int(pix.w)
    # 初始化页面的上边界
    pageU = 0
    # 初始化页面的下边界，取位图高度
    pageD = int(pix.h)
    

    #--------- 通过json_from_DocXchain来获取 table ---------#
    # 用于存储从 DocXChain 获取的表格边界框
    table_bbox_from_DocXChain = []

    # 将传入的 JSON 对象赋值给变量
    xf_json = json_from_DocXchain_obj
    # 从 JSON 中获取页面宽度
    width_from_json = xf_json['page_info']['width']
    # 从 JSON 中获取页面高度
    height_from_json = xf_json['page_info']['height']
    # 计算左右缩放比例
    LR_scaleRatio = width_from_json / (pageR - pageL)
    # 计算上下缩放比例
    UD_scaleRatio = height_from_json / (pageD - pageU)

    
    # 遍历从 JSON 中获取的布局信息
    for xf in xf_json['layout_dets']:
    # {0: 'title', 1: 'figure', 2: 'plain text', 3: 'header', 4: 'page number', 5: 'footnote', 6: 'footer', 7: 'table', 8: 'table caption', 9: 'figure caption', 10: 'equation', 11: 'full column', 12: 'sub column'}
    #  13: 'embedding',     # 嵌入公式
    #  14: 'isolated'}      # 单行公式
        # 根据缩放比例计算左边界
        L = xf['poly'][0] / LR_scaleRatio
        # 根据缩放比例计算上边界
        U = xf['poly'][1] / UD_scaleRatio
        # 根据缩放比例计算右边界
        R = xf['poly'][2] / LR_scaleRatio
        # 根据缩放比例计算下边界
        D = xf['poly'][5] / UD_scaleRatio
        # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
        # R += pageL
        # U += pageU
        # D += pageU
        # 确保左、右边界的顺序正确
        L, R = min(L, R), max(L, R)
        # 确保上、下边界的顺序正确
        U, D = min(U, D), max(U, D)
        # 如果类别是表格且得分高于阈值，添加边界框到列表
        if xf['category_id'] == 7 and xf['score'] >= 0.3:
            table_bbox_from_DocXChain.append((L, U, R, D))
            
    
    # 用于存储最终表格的名称和边界框
    table_final_names = []
    table_final_bboxs = []
    # 表格的ID初始化为0
    table_ID = 0
    # 遍历收集到的表格边界框
    for L, U, R, D in table_bbox_from_DocXChain:
        # cur_table = page.get_pixmap(clip=(L,U,R,D))
        # 生成表格的名称，格式为 "table_页ID_表格ID.png"
        new_table_name = "table_{}_{}.png".format(page_ID, table_ID)      # 表格name
        # cur_table.save(res_dir_path + '/' + new_table_name)        # 把表格存出在新建的文件夹，并命名
        # 将表格名称添加到列表中，以便后续引用
        table_final_names.append(new_table_name)                      # 把表格的名字存在list中，方便在md中插入引用
        # 将表格的边界框添加到列表中
        table_final_bboxs.append((L, U, R, D))
        # 表格ID自增
        table_ID += 1
        

    # 根据上边界和左边界对表格边界框进行排序
    table_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 当前页面所有表格的边界框
    curPage_all_table_bboxs = table_final_bboxs
    # 返回当前页面的所有表格边界框
    return curPage_all_table_bboxs
```