# `.\MinerU\magic_pdf\libs\draw_bbox.py`

```
# 从指定的库导入 PyMuPDF，用于处理 PDF 文件
from magic_pdf.libs.commons import fitz  # PyMuPDF
# 从常量库中导入 CROSS_PAGE 常量
from magic_pdf.libs.Constants import CROSS_PAGE
# 从 OCR 内容类型库中导入 BlockType、CategoryId 和 ContentType
from magic_pdf.libs.ocr_content_type import BlockType, CategoryId, ContentType
# 从模型库中导入 MagicModel 类
from magic_pdf.model.magic_model import MagicModel


# 定义一个函数，用于在 PDF 页面上绘制不带编号的边界框
def draw_bbox_without_number(i, bbox_list, page, rgb_config, fill_config):
    new_rgb = []  # 初始化一个空列表，用于存储新的 RGB 颜色值
    # 将 RGB 配置中的每个值转换为 0 到 1 之间的浮点数，并添加到新列表
    for item in rgb_config:
        item = float(item) / 255  # 将 0-255 的 RGB 值转换为 0-1 的浮点数
        new_rgb.append(item)  # 将转换后的值添加到新 RGB 列表中
    page_data = bbox_list[i]  # 获取指定索引 i 的边界框数据
    # 遍历页面数据中的每个边界框
    for bbox in page_data:
        x0, y0, x1, y1 = bbox  # 解包边界框的坐标
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # 定义矩形区域
        # 如果 fill_config 为真，表示需要填充矩形
        if fill_config:
            # 在页面上绘制填充的矩形
            page.draw_rect(
                rect_coords,  # 矩形坐标
                color=None,  # 不设置边框颜色
                fill=new_rgb,  # 使用新 RGB 值填充
                fill_opacity=0.3,  # 填充透明度为 0.3
                width=0.5,  # 边框宽度为 0.5
                overlay=True,  # 在页面上覆盖绘制
            )  # 绘制矩形
        else:
            # 在页面上绘制边框的矩形
            page.draw_rect(
                rect_coords,  # 矩形坐标
                color=new_rgb,  # 设置边框颜色为新 RGB 值
                fill=None,  # 不进行填充
                fill_opacity=1,  # 填充透明度为 1（不透明）
                width=0.5,  # 边框宽度为 0.5
                overlay=True,  # 在页面上覆盖绘制
            )  # 绘制矩形


# 定义一个函数，用于在 PDF 页面上绘制带编号的边界框
def draw_bbox_with_number(i, bbox_list, page, rgb_config, fill_config):
    new_rgb = []  # 初始化一个空列表，用于存储新的 RGB 颜色值
    # 将 RGB 配置中的每个值转换为 0 到 1 之间的浮点数，并添加到新列表
    for item in rgb_config:
        item = float(item) / 255  # 将 0-255 的 RGB 值转换为 0-1 的浮点数
        new_rgb.append(item)  # 将转换后的值添加到新 RGB 列表中
    page_data = bbox_list[i]  # 获取指定索引 i 的边界框数据
    # 遍历页面数据中的每个边界框，同时获取索引
    for j, bbox in enumerate(page_data):
        x0, y0, x1, y1 = bbox  # 解包边界框的坐标
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # 定义矩形区域
        # 如果 fill_config 为真，表示需要填充矩形
        if fill_config:
            # 在页面上绘制填充的矩形
            page.draw_rect(
                rect_coords,  # 矩形坐标
                color=None,  # 不设置边框颜色
                fill=new_rgb,  # 使用新 RGB 值填充
                fill_opacity=0.3,  # 填充透明度为 0.3
                width=0.5,  # 边框宽度为 0.5
                overlay=True,  # 在页面上覆盖绘制
            )  # 绘制矩形
        else:
            # 在页面上绘制边框的矩形
            page.draw_rect(
                rect_coords,  # 矩形坐标
                color=new_rgb,  # 设置边框颜色为新 RGB 值
                fill=None,  # 不进行填充
                fill_opacity=1,  # 填充透明度为 1（不透明）
                width=0.5,  # 边框宽度为 0.5
                overlay=True,  # 在页面上覆盖绘制
            )  # 绘制矩形
        # 在矩形左上角插入编号文本
        page.insert_text(
            (x0, y0 + 10), str(j + 1), fontsize=10, color=new_rgb
        )  # 在矩形的左上角插入索引文本


# 定义一个函数，用于绘制 PDF 文档的布局边界框
def draw_layout_bbox(pdf_info, pdf_bytes, out_path, filename):
    layout_bbox_list = []  # 初始化一个空列表，用于存储布局边界框
    dropped_bbox_list = []  # 初始化一个空列表，用于存储丢弃的边界框
    tables_list, tables_body_list = [], []  # 初始化表格相关的空列表
    tables_caption_list, tables_footnote_list = [], []  # 初始化表格标题和脚注的空列表
    imgs_list, imgs_body_list, imgs_caption_list = [], [], []  # 初始化图像相关的空列表
    imgs_footnote_list = []  # 初始化图像脚注的空列表
    titles_list = []  # 初始化标题的空列表
    texts_list = []  # 初始化文本的空列表
    interequations_list = []  # 初始化内部公式的空列表
    # 遍历每一页的 PDF 信息
        for page in pdf_info:
            # 初始化页面布局列表
            page_layout_list = []
            # 初始化丢弃的块列表
            page_dropped_list = []
            # 初始化各类表格相关的列表
            tables, tables_body, tables_caption, tables_footnote = [], [], [], []
            # 初始化各类图片相关的列表
            imgs, imgs_body, imgs_caption, imgs_footnote = [], [], [], []
            # 初始化标题列表
            titles = []
            # 初始化文本列表
            texts = []
            # 初始化行间方程列表
            interequations = []
            # 遍历当前页面的布局边界框
            for layout in page['layout_bboxes']:
                # 将布局边界框添加到页面布局列表
                page_layout_list.append(layout['layout_bbox'])
            # 将页面布局列表添加到总体布局边界框列表
            layout_bbox_list.append(page_layout_list)
            # 遍历当前页面的丢弃块
            for dropped_bbox in page['discarded_blocks']:
                # 将丢弃的边界框添加到页面丢弃列表
                page_dropped_list.append(dropped_bbox['bbox'])
            # 将页面丢弃列表添加到总体丢弃边界框列表
            dropped_bbox_list.append(page_dropped_list)
            # 遍历当前页面的段落块
            for block in page['para_blocks']:
                # 获取块的边界框
                bbox = block['bbox']
                # 如果块类型为表格
                if block['type'] == BlockType.Table:
                    # 将表格边界框添加到表格列表
                    tables.append(bbox)
                    # 遍历块中的嵌套块
                    for nested_block in block['blocks']:
                        # 获取嵌套块的边界框
                        bbox = nested_block['bbox']
                        # 根据嵌套块类型添加到相应的表格相关列表
                        if nested_block['type'] == BlockType.TableBody:
                            tables_body.append(bbox)
                        elif nested_block['type'] == BlockType.TableCaption:
                            tables_caption.append(bbox)
                        elif nested_block['type'] == BlockType.TableFootnote:
                            tables_footnote.append(bbox)
                # 如果块类型为图片
                elif block['type'] == BlockType.Image:
                    # 将图片边界框添加到图片列表
                    imgs.append(bbox)
                    # 遍历块中的嵌套块
                    for nested_block in block['blocks']:
                        # 获取嵌套块的边界框
                        bbox = nested_block['bbox']
                        # 根据嵌套块类型添加到相应的图片相关列表
                        if nested_block['type'] == BlockType.ImageBody:
                            imgs_body.append(bbox)
                        elif nested_block['type'] == BlockType.ImageCaption:
                            imgs_caption.append(bbox)
                        elif nested_block['type'] == BlockType.ImageFootnote:
                            imgs_footnote.append(bbox)
                # 如果块类型为标题
                elif block['type'] == BlockType.Title:
                    # 将标题边界框添加到标题列表
                    titles.append(bbox)
                # 如果块类型为文本
                elif block['type'] == BlockType.Text:
                    # 将文本边界框添加到文本列表
                    texts.append(bbox)
                # 如果块类型为行间方程
                elif block['type'] == BlockType.InterlineEquation:
                    # 将方程边界框添加到行间方程列表
                    interequations.append(bbox)
            # 将各类表格相关的列表添加到总体列表
            tables_list.append(tables)
            tables_body_list.append(tables_body)
            tables_caption_list.append(tables_caption)
            tables_footnote_list.append(tables_footnote)
            # 将各类图片相关的列表添加到总体列表
            imgs_list.append(imgs)
            imgs_body_list.append(imgs_body)
            imgs_caption_list.append(imgs_caption)
            imgs_footnote_list.append(imgs_footnote)
            # 将标题、文本和行间方程的列表添加到总体列表
            titles_list.append(titles)
            texts_list.append(texts)
            interequations_list.append(interequations)
    
        # 打开 PDF 文档
        pdf_docs = fitz.open('pdf', pdf_bytes)
    # 遍历 PDF 文档列表，获取索引和页面内容
        for i, page in enumerate(pdf_docs):
            # 在页面上绘制带编号的边界框，颜色为红色
            draw_bbox_with_number(i, layout_bbox_list, page, [255, 0, 0], False)
            # 在页面上绘制不带编号的边界框，颜色为灰色，表示丢弃的边界框
            draw_bbox_without_number(i, dropped_bbox_list, page, [158, 158, 158],
                                     True)
            # 在页面上绘制不带编号的边界框，颜色为黄色，表示表格
            draw_bbox_without_number(i, tables_list, page, [153, 153, 0],
                                     True)  # color !
            # 在页面上绘制不带编号的边界框，颜色为淡黄色，表示表格主体
            draw_bbox_without_number(i, tables_body_list, page, [204, 204, 0],
                                     True)
            # 在页面上绘制不带编号的边界框，颜色为浅黄色，表示表格标题
            draw_bbox_without_number(i, tables_caption_list, page, [255, 255, 102],
                                     True)
            # 在页面上绘制不带编号的边界框，颜色为浅绿色，表示表格脚注
            draw_bbox_without_number(i, tables_footnote_list, page,
                                     [229, 255, 204], True)
            # 在页面上绘制不带编号的边界框，颜色为深绿色，表示图片
            draw_bbox_without_number(i, imgs_list, page, [51, 102, 0], True)
            # 在页面上绘制不带编号的边界框，颜色为亮绿色，表示图片主体
            draw_bbox_without_number(i, imgs_body_list, page, [153, 255, 51], True)
            # 在页面上绘制不带编号的边界框，颜色为蓝色，表示图片标题
            draw_bbox_without_number(i, imgs_caption_list, page, [102, 178, 255],
                                     True)
            # 在页面上绘制带编号的边界框，颜色为橙色，表示图片脚注
            draw_bbox_with_number(i, imgs_footnote_list, page, [255, 178, 102],
                                  True),
            # 在页面上绘制不带编号的边界框，颜色为蓝色，表示标题
            draw_bbox_without_number(i, titles_list, page, [102, 102, 255], True)
            # 在页面上绘制不带编号的边界框，颜色为深红色，表示文本
            draw_bbox_without_number(i, texts_list, page, [153, 0, 76], True)
            # 在页面上绘制不带编号的边界框，颜色为绿色，表示交互方程
            draw_bbox_without_number(i, interequations_list, page, [0, 255, 0],
                                     True)
    
        # 保存处理后的 PDF 文件到指定路径
        pdf_docs.save(f'{out_path}/{filename}_layout.pdf')
# 绘制 PDF 页面的边界框信息，并保存到指定路径
def draw_span_bbox(pdf_info, pdf_bytes, out_path, filename):
    # 初始化用于存储不同内容类型的列表
    text_list = []
    inline_equation_list = []
    interline_equation_list = []
    image_list = []
    table_list = []
    dropped_list = []
    next_page_text_list = []
    next_page_inline_equation_list = []

    # 定义获取 span 信息的内部函数
    def get_span_info(span):
        # 处理文本类型的 span
        if span['type'] == ContentType.Text:
            # 判断是否为跨页文本
            if span.get(CROSS_PAGE, False):
                next_page_text_list.append(span['bbox'])
            else:
                page_text_list.append(span['bbox'])
        # 处理行内方程类型的 span
        elif span['type'] == ContentType.InlineEquation:
            if span.get(CROSS_PAGE, False):
                next_page_inline_equation_list.append(span['bbox'])
            else:
                page_inline_equation_list.append(span['bbox'])
        # 处理行间方程类型的 span
        elif span['type'] == ContentType.InterlineEquation:
            page_interline_equation_list.append(span['bbox'])
        # 处理图像类型的 span
        elif span['type'] == ContentType.Image:
            page_image_list.append(span['bbox'])
        # 处理表格类型的 span
        elif span['type'] == ContentType.Table:
            page_table_list.append(span['bbox'])

    # 遍历 PDF 信息中的每一页
    for page in pdf_info:
        # 初始化当前页的各类内容列表
        page_text_list = []
        page_inline_equation_list = []
        page_interline_equation_list = []
        page_image_list = []
        page_table_list = []
        page_dropped_list = []

        # 将跨页的文本 span 移动到当前页的列表中
        if len(next_page_text_list) > 0:
            page_text_list.extend(next_page_text_list)
            next_page_text_list.clear()
        if len(next_page_inline_equation_list) > 0:
            page_inline_equation_list.extend(next_page_inline_equation_list)
            next_page_inline_equation_list.clear()

        # 构造被丢弃的内容列表
        for block in page['discarded_blocks']:
            if block['type'] == BlockType.Discarded:
                for line in block['lines']:
                    for span in line['spans']:
                        page_dropped_list.append(span['bbox'])
        dropped_list.append(page_dropped_list)
        # 构造其余有用的内容列表
        for block in page['para_blocks']:
            if block['type'] in [
                    BlockType.Text,
                    BlockType.Title,
                    BlockType.InterlineEquation,
            ]:
                for line in block['lines']:
                    for span in line['spans']:
                        get_span_info(span)
            elif block['type'] in [BlockType.Image, BlockType.Table]:
                for sub_block in block['blocks']:
                    for line in sub_block['lines']:
                        for span in line['spans']:
                            get_span_info(span)
        # 将当前页的内容添加到各自的列表中
        text_list.append(page_text_list)
        inline_equation_list.append(page_inline_equation_list)
        interline_equation_list.append(page_interline_equation_list)
        image_list.append(page_image_list)
        table_list.append(page_table_list)
    # 打开 PDF 文档以处理字节数据
    pdf_docs = fitz.open('pdf', pdf_bytes)
    # 遍历 PDF 文档中的每一页，i 为页码，page 为当前页对象
    for i, page in enumerate(pdf_docs):
        # 在当前页面绘制文本框，颜色为红色
        draw_bbox_without_number(i, text_list, page, [255, 0, 0], False)
        # 在当前页面绘制行内公式的框，颜色为绿色
        draw_bbox_without_number(i, inline_equation_list, page, [0, 255, 0],
                                 False)
        # 在当前页面绘制行间公式的框，颜色为蓝色
        draw_bbox_without_number(i, interline_equation_list, page, [0, 0, 255],
                                 False)
        # 在当前页面绘制图像的框，颜色为橙色
        draw_bbox_without_number(i, image_list, page, [255, 204, 0], False)
        # 在当前页面绘制表格的框，颜色为紫色
        draw_bbox_without_number(i, table_list, page, [204, 0, 255], False)
        # 在当前页面绘制丢失内容的框，颜色为灰色
        draw_bbox_without_number(i, dropped_list, page, [158, 158, 158], False)

    # 保存处理后的 PDF 文件到指定路径
    pdf_docs.save(f'{out_path}/{filename}_spans.pdf')
# 定义函数，接受模型列表、PDF字节流、输出路径和文件名作为参数
def drow_model_bbox(model_list: list, pdf_bytes, out_path, filename):
    # 初始化被丢弃的边界框列表
    dropped_bbox_list = []
    # 初始化各种表格和图像的边界框列表
    tables_body_list, tables_caption_list, tables_footnote_list = [], [], []
    imgs_body_list, imgs_caption_list, imgs_footnote_list = [], [], []
    titles_list = []
    texts_list = []
    interequations_list = []
    # 打开PDF字节流
    pdf_docs = fitz.open('pdf', pdf_bytes)
    # 创建魔法模型实例，使用模型列表和PDF文档
    magic_model = MagicModel(model_list, pdf_docs)
    # 遍历模型列表的索引
    for i in range(len(model_list)):
        # 初始化页面的被丢弃边界框列表
        page_dropped_list = []
        # 初始化当前页面的各种边界框列表
        tables_body, tables_caption, tables_footnote = [], [], []
        imgs_body, imgs_caption, imgs_footnote = [], [], []
        titles = []
        texts = []
        interequations = []
        # 获取当前页面的模型信息
        page_info = magic_model.get_model_list(i)
        # 提取页面的布局检测信息
        layout_dets = page_info['layout_dets']
        # 遍历布局检测信息
        for layout_det in layout_dets:
            # 获取当前布局检测的边界框
            bbox = layout_det['bbox']
            # 根据类别ID分类边界框
            if layout_det['category_id'] == CategoryId.Text:
                texts.append(bbox)  # 添加文本边界框
            elif layout_det['category_id'] == CategoryId.Title:
                titles.append(bbox)  # 添加标题边界框
            elif layout_det['category_id'] == CategoryId.TableBody:
                tables_body.append(bbox)  # 添加表格主体边界框
            elif layout_det['category_id'] == CategoryId.TableCaption:
                tables_caption.append(bbox)  # 添加表格标题边界框
            elif layout_det['category_id'] == CategoryId.TableFootnote:
                tables_footnote.append(bbox)  # 添加表格脚注边界框
            elif layout_det['category_id'] == CategoryId.ImageBody:
                imgs_body.append(bbox)  # 添加图像主体边界框
            elif layout_det['category_id'] == CategoryId.ImageCaption:
                imgs_caption.append(bbox)  # 添加图像标题边界框
            elif layout_det[
                    'category_id'] == CategoryId.InterlineEquation_YOLO:
                interequations.append(bbox)  # 添加行间方程边界框
            elif layout_det['category_id'] == CategoryId.Abandon:
                page_dropped_list.append(bbox)  # 添加被丢弃的边界框
            elif layout_det['category_id'] == CategoryId.ImageFootnote:
                imgs_footnote.append(bbox)  # 添加图像脚注边界框

        # 将当前页面的边界框添加到对应的列表中
        tables_body_list.append(tables_body)
        tables_caption_list.append(tables_caption)
        tables_footnote_list.append(tables_footnote)
        imgs_body_list.append(imgs_body)
        imgs_caption_list.append(imgs_caption)
        titles_list.append(titles)
        texts_list.append(texts)
        interequations_list.append(interequations)
        dropped_bbox_list.append(page_dropped_list)
        imgs_footnote_list.append(imgs_footnote)
    # 遍历 pdf_docs 列表中的每一页，获取索引和页面对象
        for i, page in enumerate(pdf_docs):
            # 在页面上绘制掉落的边框，颜色为灰色，标记为可见
            draw_bbox_with_number(i, dropped_bbox_list, page, [158, 158, 158],
                                  True)  # color !
            # 在页面上绘制表格主体的边框，颜色为黄色，标记为可见
            draw_bbox_with_number(i, tables_body_list, page, [204, 204, 0], True)
            # 在页面上绘制表格标题的边框，颜色为浅黄色，标记为可见
            draw_bbox_with_number(i, tables_caption_list, page, [255, 255, 102],
                                  True)
            # 在页面上绘制表格脚注的边框，颜色为浅绿色，标记为可见
            draw_bbox_with_number(i, tables_footnote_list, page, [229, 255, 204],
                                  True)
            # 在页面上绘制图像主体的边框，颜色为亮绿色，标记为可见
            draw_bbox_with_number(i, imgs_body_list, page, [153, 255, 51], True)
            # 在页面上绘制图像标题的边框，颜色为蓝色，标记为可见
            draw_bbox_with_number(i, imgs_caption_list, page, [102, 178, 255],
                                  True)
            # 在页面上绘制图像脚注的边框，颜色为橙色，标记为可见
            draw_bbox_with_number(i, imgs_footnote_list, page, [255, 178, 102],
                                  True)
            # 在页面上绘制标题的边框，颜色为深蓝色，标记为可见
            draw_bbox_with_number(i, titles_list, page, [102, 102, 255], True)
            # 在页面上绘制文本的边框，颜色为暗红色，标记为可见
            draw_bbox_with_number(i, texts_list, page, [153, 0, 76], True)
            # 在页面上绘制内部方程的边框，颜色为绿色，标记为可见
            draw_bbox_with_number(i, interequations_list, page, [0, 255, 0], True)
    
        # 保存修改后的 PDF 文档，路径为 out_path，文件名为 filename 加上后缀
        pdf_docs.save(f'{out_path}/{filename}_model.pdf')
```