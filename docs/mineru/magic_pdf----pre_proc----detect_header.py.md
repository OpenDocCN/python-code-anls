# `.\MinerU\magic_pdf\pre_proc\detect_header.py`

```
# 从magic_pdf.libs.commons导入fitz库，pyMuPDF的核心功能
from magic_pdf.libs.commons import fitz             # pyMuPDF库
# 从magic_pdf.libs.coordinate_transform导入get_scale_ratio函数
from magic_pdf.libs.coordinate_transform import get_scale_ratio

# 定义解析页眉的函数，接收页码、页面对象和JSON对象作为参数
def parse_headers(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """

    # 初始化一个空列表用于存储从DocXChain获取的页眉边界框
    header_bbox_from_DocXChain = []

    # 将传入的JSON对象赋值给变量
    xf_json = json_from_DocXchain_obj
    # 获取水平和垂直缩放比例
    horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(xf_json, page)

    # 遍历JSON对象中'layout_dets'的每个元素
    for xf in xf_json['layout_dets']:
        # 计算左边界，使用水平缩放比例
        L = xf['poly'][0] / horizontal_scale_ratio
        # 计算上边界，使用垂直缩放比例
        U = xf['poly'][1] / vertical_scale_ratio
        # 计算右边界，使用水平缩放比例
        R = xf['poly'][2] / horizontal_scale_ratio
        # 计算下边界，使用垂直缩放比例
        D = xf['poly'][5] / vertical_scale_ratio
        # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
        # R += pageL
        # U += pageU
        # D += pageU
        # 确保左边界和右边界的顺序正确
        L, R = min(L, R), max(L, R)
        # 确保上边界和下边界的顺序正确
        U, D = min(U, D), max(U, D)
        # 检查类别是否为页眉且分数高于0.3
        if xf['category_id'] == 3 and xf['score'] >= 0.3:
            # 将有效的页眉边界框添加到列表中
            header_bbox_from_DocXChain.append((L, U, R, D))
            
    # 初始化用于存储最终页眉名称和边界框的列表
    header_final_names = []
    header_final_bboxs = []
    # 初始化页眉ID
    header_ID = 0
    # 遍历从DocXChain获取的页眉边界框
    for L, U, R, D in header_bbox_from_DocXChain:
        # cur_header = page.get_pixmap(clip=(L,U,R,D))  # 获取当前页眉的位图
        # 生成页眉的文件名
        new_header_name = "header_{}_{}.png".format(page_ID, header_ID)    # 页眉name
        # cur_header.save(res_dir_path + '/' + new_header_name)           # 把页眉存储在新建的文件夹，并命名
        # 将新生成的页眉名称添加到列表中
        header_final_names.append(new_header_name)                        # 把页面的名字存在list中
        # 将页眉的边界框添加到列表中
        header_final_bboxs.append((L, U, R, D))
        # 增加页眉ID以便于命名
        header_ID += 1
        

    # 根据上边界和左边界对页眉边界框进行排序
    header_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 将当前页的所有页眉边界框赋值给变量
    curPage_all_header_bboxs = header_final_bboxs
    # 返回当前页所有页眉的边界框
    return curPage_all_header_bboxs
```