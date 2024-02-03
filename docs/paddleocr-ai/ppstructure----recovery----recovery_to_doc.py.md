# `.\PaddleOCR\ppstructure\recovery\recovery_to_doc.py`

```
# 导入所需的模块和库
import os
from copy import deepcopy
from docx import Document
from docx import shared
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT
from ppstructure.recovery.table_process import HtmlToDocx
from ppocr.utils.logging import get_logger

# 获取日志记录器对象
logger = get_logger()

# 定义一个函数，将信息转换为 docx 格式
def convert_info_docx(img, res, save_folder, img_name):
    # 创建一个空白的 Word 文档对象
    doc = Document()
    # 设置文档的默认字体为 Times New Roman
    doc.styles['Normal'].font.name = 'Times New Roman'
    # 设置文档的默认中文字体为宋体
    doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')
    # 设置文档的默认字体大小为 6.5 磅
    doc.styles['Normal'].font.size = shared.Pt(6.5)

    # 初始化一个标志变量为 1
    flag = 1
    # 遍历结果列表中的每个区域，并获取索引和区域内容
    for i, region in enumerate(res):
        # 如果区域内容为空，则跳过当前循环
        if len(region['res']) == 0:
            continue
        # 获取当前区域的图片索引
        img_idx = region['img_idx']
        # 根据条件判断，设置文档的布局方式
        if flag == 2 and region['layout'] == 'single':
            section = doc.add_section(WD_SECTION.CONTINUOUS)
            section._sectPr.xpath('./w:cols')[0].set(qn('w:num'), '1')
            flag = 1
        elif flag == 1 and region['layout'] == 'double':
            section = doc.add_section(WD_SECTION.CONTINUOUS)
            section._sectPr.xpath('./w:cols')[0].set(qn('w:num'), '2')
            flag = 2

        # 根据区域类型进行不同的处理
        if region['type'].lower() == 'figure':
            # 设置图片保存路径和段落样式
            excel_save_folder = os.path.join(save_folder, img_name)
            img_path = os.path.join(excel_save_folder, '{}_{}.jpg'.format(region['bbox'], img_idx))
            paragraph_pic = doc.add_paragraph()
            paragraph_pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph_pic.add_run("")
            # 根据布局方式添加不同尺寸的图片
            if flag == 1:
                run.add_picture(img_path, width=shared.Inches(5))
            elif flag == 2:
                run.add_picture(img_path, width=shared.Inches(2))
        elif region['type'].lower() == 'title':
            # 添加标题
            doc.add_heading(region['res'][0]['text'])
        elif region['type'].lower() == 'table':
            # 处理表格类型的区域
            parser = HtmlToDocx()
            parser.table_style = 'TableGrid'
            parser.handle_table(region['res']['html'], doc)
        else:
            # 处理其他类型的区域
            paragraph = doc.add_paragraph()
            paragraph_format = paragraph.paragraph_format
            for i, line in enumerate(region['res']):
                if i == 0:
                    paragraph_format.first_line_indent = shared.Inches(0.25)
                text_run = paragraph.add_run(line['text'] + ' ')
                text_run.font.size = shared.Pt(10)

    # 保存文档为 docx 格式
    docx_path = os.path.join(save_folder, '{}_ocr.docx'.format(img_name))
    doc.save(docx_path)
    logger.info('docx save to {}'.format(docx_path))
# 对文本框按照从上到下，从左到右的顺序进行排序
def sorted_layout_boxes(res, w):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        res(list):ppstructure results
    return:
        sorted results(list)
    """
    # 获取文本框的数量
    num_boxes = len(res)
    # 如果只有一个文本框，则设置其布局为'single'，并返回结果
    if num_boxes == 1:
        res[0]['layout'] = 'single'
        return res

    # 根据文本框的坐标进行排序
    sorted_boxes = sorted(res, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    # 复制排序后的文本框列表
    _boxes = list(sorted_boxes)

    # 初始化新的结果列表和左右两个子列表
    new_res = []
    res_left = []
    res_right = []
    i = 0
    # 进入循环，处理每个文本框的布局
    while True:
        # 如果当前处理的文本框索引超过总文本框数量，则跳出循环
        if i >= num_boxes:
            break
        # 如果当前处理的文本框是最后一个
        if i == num_boxes - 1:
            # 检查当前文本框是否符合单列布局条件
            if _boxes[i]['bbox'][1] > _boxes[i - 1]['bbox'][3] and _boxes[i]['bbox'][0] < w / 2 and _boxes[i]['bbox'][2] > w / 2:
                # 将当前文本框添加到新结果中，并设置为单列布局
                new_res += res_left
                new_res += res_right
                _boxes[i]['layout'] = 'single'
                new_res.append(_boxes[i])
            else:
                # 如果不符合单列布局条件，则根据位置添加到左列或右列
                if _boxes[i]['bbox'][2] > w / 2:
                    _boxes[i]['layout'] = 'double'
                    res_right.append(_boxes[i])
                    new_res += res_left
                    new_res += res_right
                elif _boxes[i]['bbox'][0] < w / 2:
                    _boxes[i]['layout'] = 'double'
                    res_left.append(_boxes[i])
                    new_res += res_left
                    new_res += res_right
            # 清空左列和右列
            res_left = []
            res_right = []
            break
        # 如果当前文本框在左侧1/4且右侧3/4范围内
        elif _boxes[i]['bbox'][0] < w / 4 and _boxes[i]['bbox'][2] < 3 * w / 4:
            # 将当前文本框添加到左列，并移动到下一个文本框
            _boxes[i]['layout'] = 'double'
            res_left.append(_boxes[i])
            i += 1
        # 如果当前文本框在左侧1/4且右侧超过1/2范围
        elif _boxes[i]['bbox'][0] > w / 4 and _boxes[i]['bbox'][2] > w / 2:
            # 将当前文本框添加到右列，并移动到下一个文本框
            _boxes[i]['layout'] = 'double'
            res_right.append(_boxes[i])
            i += 1
        else:
            # 如果不符合以上条件，则将当前文本框添加到新结果中，并设置为单列布局
            new_res += res_left
            new_res += res_right
            _boxes[i]['layout'] = 'single'
            new_res.append(_boxes[i])
            # 清空左列和右列
            res_left = []
            res_right = []
            i += 1
    # 如果左列还有文本框，添加到新结果中
    if res_left:
        new_res += res_left
    # 如果右列还有文本框，添加到新结果中
    if res_right:
        new_res += res_right
    # 返回处理后的文本框列表
    return new_res
```