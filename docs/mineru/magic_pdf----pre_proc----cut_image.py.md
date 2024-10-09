# `.\MinerU\magic_pdf\pre_proc\cut_image.py`

```
# 从 loguru 导入 logger，用于记录日志信息
from loguru import logger

# 从 commons 模块导入 join_path 函数，用于连接路径
from magic_pdf.libs.commons import join_path
# 从 ocr_content_type 模块导入 ContentType 类，表示内容类型
from magic_pdf.libs.ocr_content_type import ContentType
# 从 pdf_image_tools 模块导入 cut_image 函数，用于裁剪图片
from magic_pdf.libs.pdf_image_tools import cut_image


# 定义 ocr_cut_image_and_table 函数，处理图像和表格的裁剪
def ocr_cut_image_and_table(spans, page, page_id, pdf_bytes_md5, imageWriter):
    # 定义返回路径的内部函数，使用 PDF 的 MD5 和类型生成路径
    def return_path(type):
        return join_path(pdf_bytes_md5, type)

    # 遍历每个 span，处理其类型
    for span in spans:
        # 获取当前 span 的类型
        span_type = span['type']
        # 如果 span 类型是图片
        if span_type == ContentType.Image:
            # 检查图片边界框是否有效
            if not check_img_bbox(span['bbox']):
                continue  # 如果无效，跳过当前 span
            # 裁剪图片并保存路径
            span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('images'),
                                           imageWriter=imageWriter)
        # 如果 span 类型是表格
        elif span_type == ContentType.Table:
            # 检查表格边界框是否有效
            if not check_img_bbox(span['bbox']):
                continue  # 如果无效，跳过当前 span
            # 裁剪表格并保存路径
            span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('tables'),
                                           imageWriter=imageWriter)

    # 返回包含所有处理后的 spans
    return spans


# 定义 txt_save_images_by_bboxes 函数，保存图片并返回信息
def txt_save_images_by_bboxes(page_num: int, page, pdf_bytes_md5: str,
                              image_bboxes: list, images_overlap_backup: list, table_bboxes: list,
                              equation_inline_bboxes: list,
                              equation_interline_bboxes: list, imageWriter) -> dict:
    """
    返回一个dict, key为bbox, 值是图片地址
    """
    # 初始化用于存储信息的列表
    image_info = []
    image_backup_info = []
    table_info = []
    inline_eq_info = []
    interline_eq_info = []

    # 图片的保存路径组成是这样的： {s3_or_local_path}/{book_name}/{images|tables|equations}/{page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg

    # 定义返回路径的内部函数
    def return_path(type):
        return join_path(pdf_bytes_md5, type)

    # 遍历每个图片边界框
    for bbox in image_bboxes:
        # 检查边界框是否有效
        if not check_img_bbox(bbox):
            continue  # 如果无效，跳过当前 bbox
        # 裁剪图片并保存路径
        image_path = cut_image(bbox, page_num, page, return_path("images"), imageWriter)
        # 将信息添加到列表中
        image_info.append({"bbox": bbox, "image_path": image_path})

    # 遍历每个重叠的图片边界框
    for bbox in images_overlap_backup:
        # 检查边界框是否有效
        if not check_img_bbox(bbox):
            continue  # 如果无效，跳过当前 bbox
        # 裁剪图片并保存路径
        image_path = cut_image(bbox, page_num, page, return_path("images"), imageWriter)
        # 将信息添加到列表中
        image_backup_info.append({"bbox": bbox, "image_path": image_path})

    # 遍历每个表格边界框
    for bbox in table_bboxes:
        # 检查边界框是否有效
        if not check_img_bbox(bbox):
            continue  # 如果无效，跳过当前 bbox
        # 裁剪表格并保存路径
        image_path = cut_image(bbox, page_num, page, return_path("tables"), imageWriter)
        # 将信息添加到列表中
        table_info.append({"bbox": bbox, "image_path": image_path})

    # 返回所有收集的信息
    return image_info, image_backup_info, table_info, inline_eq_info, interline_eq_info


# 定义 check_img_bbox 函数，检查边界框的有效性
def check_img_bbox(bbox) -> bool:
    # 如果边界框的左上角坐标大于等于右下角坐标，则认为无效
    if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
        # 记录警告日志，提示错误的边界框
        logger.warning(f"image_bboxes: 错误的box, {bbox}")
        return False  # 返回无效
    return True  # 返回有效
```