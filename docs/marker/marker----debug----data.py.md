# `.\marker\marker\debug\data.py`

```py
import base64
import json
import os
import zlib
from typing import List

from marker.schema import Page
from marker.settings import settings
from PIL import Image
import io

# 定义一个函数，用于将公式的调试数据转储到文件中
def dump_equation_debug_data(doc, images, converted_spans):
    # 如果未设置调试数据文件夹或调试级别为0，则直接返回
    if not settings.DEBUG_DATA_FOLDER or settings.DEBUG_LEVEL == 0:
        return

    # 如果图片列表为空，则直接返回
    if len(images) == 0:
        return

    # 断言每个图片都有对应的转换结果
    assert len(converted_spans) == len(images)

    data_lines = []
    # 遍历图片和对应的转换结果
    for idx, (pil_image, converted_span) in enumerate(zip(images, converted_spans)):
        # 如果转换结果为空，则跳过当前图片
        if converted_span is None:
            continue
        # 将 PIL 图像保存为 BytesIO 对象
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="WEBP", lossless=True)
        # 将图片数据进行 base64 编码
        b64_image = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        # 将图片数据、转换后的文本和边界框信息添加到数据行中
        data_lines.append({
            "image": b64_image,
            "text": converted_span.text,
            "bbox": converted_span.bbox
        })

    # 从文档名称中去除扩展名
    doc_base = os.path.basename(doc.name).rsplit(".", 1)[0]

    # 构建调试数据文件路径
    debug_file = os.path.join(settings.DEBUG_DATA_FOLDER, f"{doc_base}_equations.json")
    # 将数据行写入到 JSON 文件中
    with open(debug_file, "w+") as f:
        json.dump(data_lines, f)

# 定义一个函数，用于将边界框的调试数据转储到文件中
def dump_bbox_debug_data(doc, blocks: List[Page]):
    # 如果未设置调试数据文件夹或调试级别小于2，则直接返回
    if not settings.DEBUG_DATA_FOLDER or settings.DEBUG_LEVEL < 2:
        return

    # 从文档名称中去除扩展名
    doc_base = os.path.basename(doc.name).rsplit(".", 1)[0]

    # 构建调试数据文件路径
    debug_file = os.path.join(settings.DEBUG_DATA_FOLDER, f"{doc_base}_bbox.json")
    debug_data = []
    # 遍历每个页面的块索引和块数据
    for idx, page_blocks in enumerate(blocks):
        # 获取当前页面对象
        page = doc[idx]

        # 获取页面的像素图像
        pix = page.get_pixmap(dpi=settings.TEXIFY_DPI, annots=False, clip=page_blocks.bbox)
        # 将像素图像转换为 PNG 格式的字节流
        png = pix.pil_tobytes(format="PNG")
        # 从 PNG 字节流创建图像对象
        png_image = Image.open(io.BytesIO(png))
        # 获取图像的宽度和高度
        width, height = png_image.size
        # 设置最大尺寸
        max_dimension = 6000
        # 如果图像宽度或高度超过最大尺寸
        if width > max_dimension or height > max_dimension:
            # 计算缩放因子
            scaling_factor = min(max_dimension / width, max_dimension / height)
            # 缩放图像
            png_image = png_image.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.ANTIALIAS)

        # 创建一个字节流对象
        img_bytes = io.BytesIO()
        # 将图像以 WEBP 格式保存到字节流中
        png_image.save(img_bytes, format="WEBP", lossless=True, quality=100)
        # 将字节流编码为 base64 字符串
        b64_image = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        # 获取页面块的模型数据
        page_data = page_blocks.model_dump()
        # 将图像数据添加到页面数据中
        page_data["image"] = b64_image
        # 将页面数据添加到调试数据列表中
        debug_data.append(page_data)

    # 将调试数据以 JSON 格式写入调试文件
    with open(debug_file, "w+") as f:
        json.dump(debug_data, f)
```