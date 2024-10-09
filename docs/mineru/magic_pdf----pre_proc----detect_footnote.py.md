# `.\MinerU\magic_pdf\pre_proc\detect_footnote.py`

```
# 导入Counter类用于计数
from collections import Counter
# 从pyMuPDF库导入fitz模块以处理PDF文件
from magic_pdf.libs.commons import fitz             # pyMuPDF库
# 导入获取缩放比例的函数
from magic_pdf.libs.coordinate_transform import get_scale_ratio


# 定义解析脚注的函数，接收页面ID、页面对象、JSON数据和其他参数
def parse_footnotes_by_model(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict, md_bookname_save_path=None, debug_mode=False):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """

    # 初始化用于存储脚注的边界框列表
    footnote_bbox_from_DocXChain = []

    # 将传入的JSON对象赋值给变量
    xf_json = json_from_DocXchain_obj
    # 获取页面的水平和垂直缩放比例
    horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(xf_json, page)

    # 遍历JSON中的布局信息
    for xf in xf_json['layout_dets']:
        # 根据缩放比例计算多边形的左、上、右、下坐标
        L = xf['poly'][0] / horizontal_scale_ratio
        U = xf['poly'][1] / vertical_scale_ratio
        R = xf['poly'][2] / horizontal_scale_ratio
        D = xf['poly'][5] / vertical_scale_ratio
        # 对坐标进行排序以确保左、右、上、下的正确性
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        # 检查分类ID和分数以确定是否为脚注
        if xf['category_id'] == 5 and xf['score'] >= 0.43:  # 新的footnote阈值
            # 如果是脚注，则将边界框添加到列表中
            footnote_bbox_from_DocXChain.append((L, U, R, D))
            
    # 初始化脚注名称和边界框列表
    footnote_final_names = []
    footnote_final_bboxs = []
    footnote_ID = 0
    # 遍历脚注的边界框
    for L, U, R, D in footnote_bbox_from_DocXChain:
        if debug_mode:
            # 根据脚注的边界框生成脚注图像
            new_footnote_name = "footnote_{}_{}.png".format(page_ID, footnote_ID)    # 脚注name
            # 保存脚注图像到指定路径
            footnote_final_names.append(new_footnote_name)                        # 把脚注的名字存在list中
        # 将脚注的边界框添加到列表中
        footnote_final_bboxs.append((L, U, R, D))
        footnote_ID += 1  # 增加脚注ID

    # 对脚注边界框进行排序
    footnote_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 当前页面的所有脚注边界框
    curPage_all_footnote_bboxs = footnote_final_bboxs
    # 返回当前页面的所有脚注边界框
    return curPage_all_footnote_bboxs


# 定义需要移除的块的函数
def need_remove(block):
    # 检查 block 中是否包含 'lines' 并且 'lines' 的长度大于 0
    if 'lines' in block and len(block['lines']) > 0:
        # 检查 block 中是否只有一行，且该行包含的文本全为大写字母或字体为粗体（包含关键词 'SB'、'bold'、'Bold'），如果满足条件则返回 True
        if len(block['lines']) == 1:
            # 检查第一行中是否有 'spans'，且 'spans' 的长度为 1
            if 'spans' in block['lines'][0] and len(block['lines'][0]['spans']) == 1:
                # 定义字体关键词列表
                font_keywords = ['SB', 'bold', 'Bold']
                # 检查该行的文本是否全为大写，或字体中是否包含关键词
                if block['lines'][0]['spans'][0]['text'].isupper() or any(keyword in block['lines'][0]['spans'][0]['font'] for keyword in font_keywords):
                    # 满足条件，返回 True
                    return True
        # 遍历 block 中的每一行
        for line in block['lines']:
            # 检查每一行中是否有 'spans'，且 'spans' 的长度大于 0
            if 'spans' in line and len(line['spans']) > 0:
                # 遍历每个 span
                for span in line['spans']:
                    # 检测 "keyword" 是否在 span 的文本中，忽略大小写
                    if "keyword" in span['text'].lower():
                        # 找到关键词，返回 True
                        return True
    # 如果以上条件均不满足，返回 False
    return False
# 根据给定的文本块、页高和页码，解析出符合规则的脚注文本块，并返回其边界框。
def parse_footnotes_by_rule(remain_text_blocks, page_height, page_id, main_text_font):
    # 定义一个函数，接受文本块列表、页面高度、页面ID和主要字体作为参数
    """
    根据给定的文本块、页高和页码，解析出符合规则的脚注文本块，并返回其边界框。

    Args:
        remain_text_blocks (list): 包含所有待处理的文本块的列表。
        page_height (float): 页面的高度。
        page_id (int): 页面的ID。

    Returns:
        list: 符合规则的脚注文本块的边界框列表。

    """
    # 检查页面ID是否大于2以筛选前3页
    if page_id > 2:  # 为保证精确度，先只筛选前3页
        # 如果页面ID大于2，返回一个空列表
        return []
```