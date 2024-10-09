# `.\MinerU\magic_pdf\libs\vis_utils.py`

```
# 从 magic_pdf.libs.commons 导入 fitz 模块
from magic_pdf.libs.commons import fitz
# 导入操作系统相关的模块
import os

# 在给定的 PDF 页面上绘制边界框并保存到指定路径
def draw_bbox_on_page(raw_pdf_doc: fitz.Document, paras_dict:dict, save_path: str):
    """
    在page上画出bbox，保存到save_path
    """
    # 初始化标志，检查是否为新创建的 PDF
    is_new_pdf = False
    # 检查保存路径是否已经存在
    if os.path.exists(save_path):
        # 如果存在，打开该 PDF 文件
        doc = fitz.open(save_path)
    else:
        # 如果不存在，设置标志为新 PDF，并创建一个新的空白 PDF
        is_new_pdf = True
        doc = fitz.open('')

    # 定义颜色映射，用于不同类型的边界框
    color_map = {
        'image': fitz.pdfcolor["yellow"],
        'text': fitz.pdfcolor['blue'],
        "table": fitz.pdfcolor['green']
    }
    
    # 遍历参数字典中的每个项
    for k, v in paras_dict.items():
        # 获取当前页面索引
        page_idx = v['page_idx']
        # 获取页面的宽度和高度
        width = raw_pdf_doc[page_idx].rect.width
        height = raw_pdf_doc[page_idx].rect.height
        # 在文档中创建一个新页面，大小与原页面相同
        new_page = doc.new_page(width=width, height=height)

        # 创建一个新形状对象用于绘制
        shape = new_page.new_shape()
        # 遍历预处理块以绘制边界框
        for order, block in enumerate(v['preproc_blocks']):
            # 获取块的边界框并转换为矩形
            rect = fitz.Rect(block['bbox'])
            # 创建一个新形状对象
            shape = new_page.new_shape()
            # 在页面上绘制矩形
            shape.draw_rect(rect)
            # 完成形状的填充，使用文本颜色
            shape.finish(color=None, fill=color_map['text'], fill_opacity=0.2)
            # 最终完成形状
            shape.finish()
            # 提交形状到页面
            shape.commit()
            
        # 遍历图像以绘制原始边界框
        for img in v['images']:
            # 获取图像的边界框
            rect = fitz.Rect(img['bbox'])
            # 创建一个新形状对象
            shape = new_page.new_shape()
            # 在页面上绘制矩形
            shape.draw_rect(rect)
            # 完成形状的填充，使用黄色
            shape.finish(color=None, fill=fitz.pdfcolor['yellow'])
            # 最终完成形状
            shape.finish()
            # 提交形状到页面
            shape.commit()

        # 遍历备份图像以绘制原始边界框
        for img in v['image_backup']:
            # 获取备份图像的边界框
            rect = fitz.Rect(img['bbox'])
            # 创建一个新形状对象
            shape = new_page.new_shape()
            # 在页面上绘制矩形
            shape.draw_rect(rect)
            # 完成形状的填充，使用黄色边框，无填充
            shape.finish(color=fitz.pdfcolor['yellow'],  fill=None)
            # 最终完成形状
            shape.finish()
            # 提交形状到页面
            shape.commit()
            
        # 遍历被丢弃的文本块以绘制边界框
        for tb in v['droped_text_block']:
            # 获取文本块的边界框
            rect = fitz.Rect(tb['bbox'])
            # 创建一个新形状对象
            shape = new_page.new_shape()
            # 在页面上绘制矩形
            shape.draw_rect(rect)
            # 完成形状的填充，使用黑色并设置填充透明度
            shape.finish(color=None, fill=fitz.pdfcolor['black'], fill_opacity=0.4)
            # 最终完成形状
            shape.finish()
            # 提交形状到页面
            shape.commit()
            
        # TODO: 处理表格边界框
        for tb in v['tables']:
            # 获取表格的边界框
            rect = fitz.Rect(tb['bbox'])
            # 创建一个新形状对象
            shape = new_page.new_shape()
            # 在页面上绘制矩形
            shape.draw_rect(rect)
            # 完成形状的填充，使用绿色并设置填充透明度
            shape.finish(color=None, fill=fitz.pdfcolor['green'], fill_opacity=0.2)
            # 最终完成形状
            shape.finish()
            # 提交形状到页面
            shape.commit()

    # 获取保存路径的父目录
    parent_dir = os.path.dirname(save_path)
    # 如果父目录不存在，则创建目录
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # 根据是否为新 PDF 保存文件
    if is_new_pdf:
        # 如果是新 PDF，保存文件
        doc.save(save_path)
    else:
        # 如果是已有 PDF，增量保存文件
        doc.saveIncr()
    # 关闭 PDF 文档
    doc.close()
    

# 以覆盖的方式生成临时 PDF 进行调试
def debug_show_bbox(raw_pdf_doc: fitz.Document, page_idx: int, bboxes: list, droped_bboxes:list,  expect_drop_bboxes:list, save_path: str, expected_page_id:int):
    """
    以覆盖的方式写个临时的pdf，用于debug
    """
    # 检查当前页面索引是否与预期页面一致
    if page_idx!=expected_page_id:
        return
        
    # 检查保存路径是否已存在
    if os.path.exists(save_path):
        # 删除已存在的文件
        os.remove(save_path)
    # 创建一个新的空白 PDF 文件
    # 打开一个新的 PDF 文档
        doc = fitz.open('')
    
        # 获取指定页面的宽度和高度
        width = raw_pdf_doc[page_idx].rect.width
        height = raw_pdf_doc[page_idx].rect.height
        # 在文档中创建一个新页面，设置宽度和高度
        new_page = doc.new_page(width=width, height=height)
    
        # 创建一个新形状用于绘制
        shape = new_page.new_shape()
        # 遍历每个框的边界
        for bbox in bboxes:
            # 根据框的坐标创建矩形
            rect = fitz.Rect(*bbox[0:4])
            # 为新页面创建一个新形状
            shape = new_page.new_shape()
            # 绘制矩形
            shape.draw_rect(rect)
            # 设置矩形的边框颜色、填充颜色及填充透明度
            shape.finish(color=fitz.pdfcolor['red'], fill=fitz.pdfcolor['blue'], fill_opacity=0.2)
            # 完成当前形状
            shape.finish()
            # 提交当前形状以渲染
            shape.commit()
            
        # 遍历被丢弃的框的边界
        for bbox in droped_bboxes:
            # 根据框的坐标创建矩形
            rect = fitz.Rect(*bbox[0:4])
            # 为新页面创建一个新形状
            shape = new_page.new_shape()
            # 绘制矩形
            shape.draw_rect(rect)
            # 设置矩形的边框颜色、填充颜色及填充透明度
            shape.finish(color=None, fill=fitz.pdfcolor['yellow'], fill_opacity=0.2)
            # 完成当前形状
            shape.finish()
            # 提交当前形状以渲染
            shape.commit()
            
        # 遍历期望丢弃的框的边界
        for bbox in expect_drop_bboxes:
            # 根据框的坐标创建矩形
            rect = fitz.Rect(*bbox[0:4])
            # 为新页面创建一个新形状
            shape = new_page.new_shape()
            # 绘制矩形
            shape.draw_rect(rect)
            # 设置矩形的边框颜色，填充颜色设置为无
            shape.finish(color=fitz.pdfcolor['red'], fill=None)
            # 完成当前形状
            shape.finish()
            # 提交当前形状以渲染
            shape.commit()
    
        # shape.insert_textbox(fitz.Rect(200, 0, 600, 20), f"total bboxes: {len(bboxes)}", fontname="helv", fontsize=12,
        #                      color=(0, 0, 0))
        # shape.finish(color=fitz.pdfcolor['black'])
        # shape.commit()
    
        # 获取保存路径的父目录
        parent_dir = os.path.dirname(save_path)
        # 检查父目录是否存在，若不存在则创建
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    
        # 保存文档到指定路径
        doc.save(save_path)
        # 关闭文档
        doc.close()
# 定义一个用于调试的函数，显示指定页面和多个边界框
def debug_show_page(page, bboxes1: list, bboxes2: list, bboxes3: list,):
    # 设置保存生成的 PDF 文件路径
    save_path = "./tmp/debug.pdf"
    # 如果文件已存在，则删除该文件
    if os.path.exists(save_path):
        # 删除已经存在的文件
        os.remove(save_path)
    # 创建一个新的空白 PDF 文件
    doc = fitz.open('')

    # 获取页面的宽度和高度
    width = page.rect.width
    height = page.rect.height
    # 在 PDF 文档中创建一个新的页面，大小与原页面相同
    new_page = doc.new_page(width=width, height=height)
    
    # 创建一个新的形状对象
    shape = new_page.new_shape()
    # 遍历第一个边界框列表
    for bbox in bboxes1:
        # 原始box画上去
        # 根据边界框的坐标创建矩形对象
        rect = fitz.Rect(*bbox[0:4])
        # 创建一个新的形状对象
        shape = new_page.new_shape()
        # 在页面上绘制矩形
        shape.draw_rect(rect)
        # 设置矩形的边框颜色和填充颜色
        shape.finish(color=fitz.pdfcolor['red'], fill=fitz.pdfcolor['blue'], fill_opacity=0.2)
        # 完成绘制形状
        shape.finish()
        # 提交形状以显示在页面上
        shape.commit()
        
    # 遍历第二个边界框列表
    for bbox in bboxes2:
        # 原始box画上去
        # 根据边界框的坐标创建矩形对象
        rect = fitz.Rect(*bbox[0:4])
        # 创建一个新的形状对象
        shape = new_page.new_shape()
        # 在页面上绘制矩形
        shape.draw_rect(rect)
        # 设置矩形的填充颜色
        shape.finish(color=None, fill=fitz.pdfcolor['yellow'], fill_opacity=0.2)
        # 完成绘制形状
        shape.finish()
        # 提交形状以显示在页面上
        shape.commit()
        
    # 遍历第三个边界框列表
    for bbox in bboxes3:
        # 原始box画上去
        # 根据边界框的坐标创建矩形对象
        rect = fitz.Rect(*bbox[0:4])
        # 创建一个新的形状对象
        shape = new_page.new_shape()
        # 在页面上绘制矩形
        shape.draw_rect(rect)
        # 设置矩形的边框颜色
        shape.finish(color=fitz.pdfcolor['red'], fill=None)
        # 完成绘制形状
        shape.finish()
        # 提交形状以显示在页面上
        shape.commit()
        
    # 获取保存路径的父目录
    parent_dir = os.path.dirname(save_path)
    # 如果父目录不存在，则创建该目录
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # 将生成的 PDF 文档保存到指定路径
    doc.save(save_path)
    # 关闭 PDF 文档
    doc.close() 
    
    
    
    
# 定义一个函数，在指定页面上绘制边界框并保存到文件
def draw_layout_bbox_on_page(raw_pdf_doc: fitz.Document, paras_dict: dict, header, footer, pdf_path: str):
    """
    在page上画出bbox，保存到save_path
    """
    # 检查文件是否存在
    is_new_pdf = False
    # 如果指定路径的 PDF 文件存在
    if os.path.exists(pdf_path):
        # 打开现有的 PDF 文件
        doc = fitz.open(pdf_path)
    else:
        # 创建一个新的空白 PDF 文件
        is_new_pdf = True
        doc = fitz.open('')
    # 遍历参数字典中的每个项
    for k, v in paras_dict.items():
        # 获取页面索引
        page_idx = v['page_idx']
        # 获取布局边界框
        layouts = v['layout_bboxes']
        # 获取对应页面
        page = doc[page_idx]
        # 创建新形状对象
        shape = page.new_shape()
        # 遍历布局列表并获取顺序和布局信息
        for order, layout in enumerate(layouts):
            # 定义边框偏移量
            border_offset = 1
            # 获取布局边界框
            rect_box = layout['layout_bbox']
            # 获取布局标签
            layout_label = layout['layout_label']
            # 根据布局标签设置填充颜色
            fill_color = fitz.pdfcolor['pink'] if layout_label=='U' else None
            # 调整边界框坐标以添加边框
            rect_box = [rect_box[0]+1, rect_box[1]-border_offset, rect_box[2]-1, rect_box[3]+border_offset]
            # 创建矩形对象
            rect = fitz.Rect(*rect_box)
            # 在页面上绘制矩形
            shape.draw_rect(rect)
            # 结束形状，设置颜色和填充
            shape.finish(color=fitz.pdfcolor['red'], fill=fill_color, fill_opacity=0.4)
            """
            在布局框上绘制顺序文本
            """
            # 定义字体大小
            font_size = 10
            # 在矩形框中插入顺序文本
            shape.insert_text((rect_box[0] + 1, rect_box[1] + font_size), f"{order}", fontsize=font_size, color=(0, 0, 0))
        
        """绘制页眉和页脚"""
        # 如果存在页眉，绘制页眉矩形
        if header:
            shape.draw_rect(fitz.Rect(header))
            # 结束形状，设置颜色和填充
            shape.finish(color=None, fill=fitz.pdfcolor['black'], fill_opacity=0.2)
        # 如果存在页脚，绘制页脚矩形
        if footer:
            shape.draw_rect(fitz.Rect(footer))
            # 结束形状，设置颜色和填充
            shape.finish(color=None, fill=fitz.pdfcolor['black'], fill_opacity=0.2)
        
        # 提交形状的绘制
        shape.commit()
    
    # 如果是新 PDF，则保存为新文件
    if is_new_pdf:
        doc.save(pdf_path)
    # 否则以增量方式保存
    else:
        doc.saveIncr()
    # 关闭文档
    doc.close()
# 使用弃用警告装饰器标记此函数为过时
@DeprecationWarning
# 定义在 PDF 文档中绘制布局的函数，接受 PDF 文档、页面索引、布局列表和 PDF 路径作为参数
def draw_layout_on_page(raw_pdf_doc: fitz.Document,  page_idx: int, page_layout: list, pdf_path: str):
    """
    把layout的box用红色边框花在pdf_path的page_idx上
    """
    # 定义绘制形状的内部函数，接受形状、布局和填充颜色作为参数
    def draw(shape, layout, fill_color=fitz.pdfcolor['pink']):
        border_offset = 1  # 边框偏移量
        rect_box = layout['layout_bbox']  # 获取布局的边界框
        layout_label = layout['layout_label']  # 获取布局标签
        sub_layout = layout['sub_layout']  # 获取子布局
        # 检查子布局是否为空
        if len(sub_layout) == 0:
            # 根据布局标签设置填充颜色
            fill_color = fill_color if layout_label == 'U' else None
            # 调整边界框以适应边框偏移量
            rect_box = [rect_box[0] + 1, rect_box[1] - border_offset, rect_box[2] - 1, rect_box[3] + border_offset]
            # 创建矩形对象
            rect = fitz.Rect(*rect_box)
            # 绘制矩形
            shape.draw_rect(rect)
            # 完成形状，设置边框颜色和填充颜色及透明度
            shape.finish(color=fitz.pdfcolor['red'], fill=fill_color, fill_opacity=0.2)
            # 注释掉的代码块用于处理布局标签为 'U' 的情况
            # if layout_label=='U':
            #     bad_boxes = layout.get("bad_boxes", [])
            #     for bad_box in bad_boxes:
            #         rect = fitz.Rect(*bad_box)
            #         shape.draw_rect(rect)
            #         shape.finish(color=fitz.pdfcolor['red'], fitz.pdfcolor['red'], fill_opacity=0.2)
        # else:  # 注释掉的部分，用于处理非空子布局的情况
        #     rect = fitz.Rect(*rect_box)
        #     shape.draw_rect(rect)
        #     shape.finish(color=fitz.pdfcolor['blue'])
        
        # 遍历子布局并递归调用绘制函数
        for sub_layout in sub_layout:
            draw(shape, sub_layout)
        # 提交形状以完成绘制
        shape.commit()
    
    # 检查指定的 PDF 文件是否存在
    is_new_pdf = False
    if os.path.exists(pdf_path):
        # 如果文件存在，则打开现有的 PDF 文件
        doc = fitz.open(pdf_path)
    else:
        # 如果文件不存在，则创建一个新的空白 PDF 文件
        is_new_pdf = True
        doc = fitz.open('')

    # 获取指定索引的页面
    page = doc[page_idx]
    # 创建新的形状对象
    shape = page.new_shape()
    # 遍历页面布局并绘制
    for order, layout in enumerate(page_layout):
        draw(shape, layout, fitz.pdfcolor['yellow'])  # 使用黄色填充绘制

    # 注释掉的代码块用于在页面上插入文本框
    # shape.insert_textbox(fitz.Rect(200, 0, 600, 20), f"total bboxes: {len(layout)}", fontname="helv", fontsize=12,
    #                      color=(0, 0, 0))
    # shape.finish(color=fitz.pdfcolor['black'])
    # shape.commit()

    # 获取 PDF 文件的父目录
    parent_dir = os.path.dirname(pdf_path)
    # 如果父目录不存在，则创建它
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # 根据文件是否为新建的 PDF 决定保存方式
    if is_new_pdf:
        doc.save(pdf_path)  # 保存新 PDF 文件
    else:
        doc.saveIncr()  # 增量保存现有 PDF 文件
    # 关闭 PDF 文档
    doc.close()
```