# `.\MinerU\magic_pdf\pre_proc\detect_footer_by_model.py`

```
# 从magic_pdf库导入fitz模块（pyMuPDF库），用于处理PDF文件
from magic_pdf.libs.commons import fitz             # pyMuPDF库
# 从coordinate_transform模块导入获取缩放比例的函数
from magic_pdf.libs.coordinate_transform import get_scale_ratio


def parse_footers(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """

    #--------- 通过json_from_DocXchain来获取 footer ---------#
    # 初始化一个空列表，用于存放从DocXChain提取的页脚边界框
    footer_bbox_from_DocXChain = []

    # 将输入的json对象赋值给变量
    xf_json = json_from_DocXchain_obj
    # 获取水平和垂直的缩放比例
    horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(xf_json, page)

    # 对每个布局元素进行遍历
    for xf in xf_json['layout_dets']:
        # 根据缩放比例计算左、上、右、下边界
        L = xf['poly'][0] / horizontal_scale_ratio
        U = xf['poly'][1] / vertical_scale_ratio
        R = xf['poly'][2] / horizontal_scale_ratio
        D = xf['poly'][5] / vertical_scale_ratio
        # 处理页面偏移（注释掉的代码）
        # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
        # R += pageL
        # U += pageU
        # D += pageU
        # 获取左右边界的最小值和最大值
        L, R = min(L, R), max(L, R)
        # 获取上下边界的最小值和最大值
        U, D = min(U, D), max(U, D)
        # 如果元素类别为6（页脚）且分数大于等于0.3，则添加到页脚边界框列表中
        if xf['category_id'] == 6 and xf['score'] >= 0.3:
            footer_bbox_from_DocXChain.append((L, U, R, D))
            
    # 初始化存放页脚名称和边界框的列表
    footer_final_names = []
    footer_final_bboxs = []
    footer_ID = 0  # 页脚ID初始化
    # 遍历每个页脚的边界框
    for L, U, R, D in footer_bbox_from_DocXChain:
        # cur_footer = page.get_pixmap(clip=(L,U,R,D))  # 获取当前页脚的位图（注释掉的代码）
        # 生成页脚名称，格式为"footer_pageID_footerID.png"
        new_footer_name = "footer_{}_{}.png".format(page_ID, footer_ID)    # 脚注name
        # cur_footer.save(res_dir_path + '/' + new_footer_name)  # 保存页脚位图到指定目录（注释掉的代码）
        # 将页脚名称添加到列表中
        footer_final_names.append(new_footer_name)                        # 把脚注的名字存在list中
        # 将页脚边界框添加到列表中
        footer_final_bboxs.append((L, U, R, D))
        footer_ID += 1  # 页脚ID递增
        
    # 对页脚边界框按照上边界和左边界进行排序
    footer_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 当前页的所有页脚边界框
    curPage_all_footer_bboxs = footer_final_bboxs
    # 返回当前页所有的页脚边界框
    return curPage_all_footer_bboxs
```