# `.\MinerU\magic_pdf\pre_proc\detect_page_number.py`

```
# 从magic_pdf.libs.commons导入fitz库，用于处理PDF文件
from magic_pdf.libs.commons import fitz             # pyMuPDF库
# 从magic_pdf.libs.coordinate_transform导入get_scale_ratio函数，用于获取缩放比例
from magic_pdf.libs.coordinate_transform import get_scale_ratio


def parse_pageNos(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_ID页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """

    # 初始化一个空列表，用于存储从DocXChain提取的页码边界框
    #--------- 通过json_from_DocXchain来获取 pageNo ---------#
    pageNo_bbox_from_DocXChain = []

    # 将传入的json对象赋值给xf_json变量
    xf_json = json_from_DocXchain_obj
    # 获取水平和垂直的缩放比例
    horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(xf_json, page)

    # 以下注释部分是文档中各类别的说明
    # {0: 'title',  # 标题
    # 1: 'figure', # 图片
    #  2: 'plain text',  # 文本
    #  3: 'header',      # 页眉
    #  4: 'page number', # 页码
    #  5: 'footnote',    # 脚注
    #  6: 'footer',      # 页脚
    #  7: 'table',       # 表格
    #  8: 'table caption',  # 表格描述
    #  9: 'figure caption', # 图片描述
    #  10: 'equation',      # 公式
    #  11: 'full column',   # 单栏
    #  12: 'sub column',    # 多栏
    #  13: 'embedding',     # 嵌入公式
    #  14: 'isolated'}      # 单行公式
    # 遍历json对象中的layout_dets字段，提取每个元素
    for xf in xf_json['layout_dets']:
        # 按水平缩放比例计算左边界
        L = xf['poly'][0] / horizontal_scale_ratio
        # 按垂直缩放比例计算上边界
        U = xf['poly'][1] / vertical_scale_ratio
        # 按水平缩放比例计算右边界
        R = xf['poly'][2] / horizontal_scale_ratio
        # 按垂直缩放比例计算下边界
        D = xf['poly'][5] / vertical_scale_ratio
        # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
        # R += pageL
        # U += pageU
        # D += pageU
        # 确保左边界小于或等于右边界
        L, R = min(L, R), max(L, R)
        # 确保上边界小于或等于下边界
        U, D = min(U, D), max(U, D)
        # 如果类别ID为4（页码）且得分大于等于0.3，则将边界框添加到列表
        if xf['category_id'] == 4 and xf['score'] >= 0.3:
            pageNo_bbox_from_DocXChain.append((L, U, R, D))
            
    # 初始化存储页码名称和边界框的列表
    pageNo_final_names = []
    pageNo_final_bboxs = []
    pageNo_ID = 0  # 页码计数器初始化为0
    # 遍历提取的页码边界框
    for L, U, R, D in pageNo_bbox_from_DocXChain:
        # cur_pageNo = page.get_pixmap(clip=(L,U,R,D))  # 根据边界框获取页码的图像
        # 创建新的页码名称，格式为"pageNo_页码ID_页码计数.png"
        new_pageNo_name = "pageNo_{}_{}.png".format(page_ID, pageNo_ID)    # 页码name
        # cur_pageNo.save(res_dir_path + '/' + new_pageNo_name)           # 把页码存储在新建的文件夹，并命名
        # 将新生成的页码名称添加到列表中
        pageNo_final_names.append(new_pageNo_name)                        # 把页码的名字存在list中
        # 将当前边界框添加到最终边界框列表中
        pageNo_final_bboxs.append((L, U, R, D))
        pageNo_ID += 1  # 页码计数器增加1
        

    # 按照上边界和左边界排序最终的边界框
    pageNo_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 将当前页的所有页码边界框赋值给变量
    curPage_all_pageNo_bboxs = pageNo_final_bboxs
    # 返回当前页的所有页码边界框
    return curPage_all_pageNo_bboxs
```