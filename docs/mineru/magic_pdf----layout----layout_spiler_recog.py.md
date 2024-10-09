# `.\MinerU\magic_pdf\layout\layout_spiler_recog.py`

```
"""
找到能分割布局的水平的横线、色块
"""

# 导入操作系统模块
import os
# 从魔法 PDF 库导入 fitz 模块
from magic_pdf.libs.commons import fitz
# 从魔法 PDF 库导入重叠检查函数
from magic_pdf.libs.boxbase import _is_in_or_part_overlap

# 定义根据宽度过滤矩形的函数
def __rect_filter_by_width(rect, page_w, page_h):
    # 计算页面中间的 x 坐标
    mid_x = page_w / 2
    # 检查矩形的左边缘是否小于中间 x，且中间 x 小于右边缘
    if rect[0] < mid_x < rect[2]:
        return True  # 如果符合条件，则返回 True
    return False  # 否则返回 False

# 定义根据位置过滤矩形的函数
def __rect_filter_by_pos(rect, image_bboxes, table_bboxes):
    """
    不能出现在 table 和 image 的位置
    """
    # 遍历图像边界框列表
    for box in image_bboxes:
        # 检查矩形是否与图像框重叠
        if _is_in_or_part_overlap(rect, box):
            return False  # 如果重叠，返回 False
    
    # 遍历表格边界框列表
    for box in table_bboxes:
        # 检查矩形是否与表格框重叠
        if _is_in_or_part_overlap(rect, box):
            return False  # 如果重叠，返回 False
    
    return True  # 如果没有重叠，返回 True

# 定义用于调试页面的函数
def __debug_show_page(page, bboxes1: list, bboxes2: list, bboxes3: list):
    # 定义保存调试 PDF 的路径
    save_path = "./tmp/debug.pdf"
    if os.path.exists(save_path):
        # 如果调试文件已经存在，删除它
        os.remove(save_path)
    # 创建一个新的空白 PDF 文件
    doc = fitz.open('')

    # 获取页面的宽度和高度
    width = page.rect.width
    height = page.rect.height
    # 在文档中创建一个新页面
    new_page = doc.new_page(width=width, height=height)
    
    # 创建一个新形状对象
    shape = new_page.new_shape()
    # 遍历第一个边界框列表
    for bbox in bboxes1:
        # 将边界框转换为矩形
        rect = fitz.Rect(*bbox[0:4])
        shape = new_page.new_shape()  # 创建新形状
        # 绘制矩形
        shape.draw_rect(rect)
        # 设置矩形的颜色和填充
        shape.finish(color=fitz.pdfcolor['red'], fill=fitz.pdfcolor['blue'], fill_opacity=0.2)
        shape.finish()  # 完成形状
        shape.commit()  # 提交形状

    # 遍历第二个边界框列表
    for bbox in bboxes2:
        # 将边界框转换为矩形
        rect = fitz.Rect(*bbox[0:4])
        shape = new_page.new_shape()  # 创建新形状
        # 绘制矩形
        shape.draw_rect(rect)
        # 设置矩形的颜色和填充
        shape.finish(color=None, fill=fitz.pdfcolor['yellow'], fill_opacity=0.2)
        shape.finish()  # 完成形状
        shape.commit()  # 提交形状

    # 遍历第三个边界框列表
    for bbox in bboxes3:
        # 将边界框转换为矩形
        rect = fitz.Rect(*bbox[0:4])
        shape = new_page.new_shape()  # 创建新形状
        # 绘制矩形
        shape.draw_rect(rect)
        # 设置矩形的边框颜色
        shape.finish(color=fitz.pdfcolor['red'], fill=None)
        shape.finish()  # 完成形状
        shape.commit()  # 提交形状
        
    # 获取调试文件的父目录
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)  # 如果目录不存在，创建目录

    # 保存调试 PDF
    doc.save(save_path)
    doc.close()  # 关闭文档
    
# 定义获取页面分割元素的函数
def get_spilter_of_page(page, image_bboxes, table_bboxes):
    """
    获取到色块和横线
    """
    # 获取页面的所有绘制对象
    cdrawings = page.get_cdrawings()
    
    # 用于存储分割边界框的列表
    spilter_bbox = []
    # 遍历所有绘制对象
    for block in cdrawings:
        if 'fill' in block:  # 检查是否有填充属性
            fill = block['fill']
        # 如果填充不为空且不为白色
        if 'fill' in block and block['fill'] and block['fill'] != (1.0, 1.0, 1.0):
            rect = block['rect']  # 获取矩形边界框
            # 过滤矩形是否符合条件
            if __rect_filter_by_width(rect, page.rect.width, page.rect.height) and __rect_filter_by_pos(rect, image_bboxes, table_bboxes):
                spilter_bbox.append(list(rect))  # 如果符合条件，添加到分割边界框列表
    
    """过滤、修正一下这些box。因为有时候会有一些矩形，高度为0或者为负数，造成layout计算无限循环。如果是负高度或者0高度，统一修正为高度为1"""
    # 遍历分割边界框列表
    for box in spilter_bbox:
        # 检查矩形的高度是否小于等于0
        if box[3] - box[1] <= 0:
            box[3] = box[1] + 1  # 修正高度为1
            
    # __debug_show_page(page, spilter_bbox, [], [])  # 调试显示页面（被注释掉）
    
    return spilter_bbox  # 返回分割边界框列表
```