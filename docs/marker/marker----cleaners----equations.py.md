# `.\marker\marker\cleaners\equations.py`

```
# 导入所需的库
import io
from copy import deepcopy
from functools import partial
from typing import List

import torch
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
import re
from PIL import Image, ImageDraw

# 导入自定义模块
from marker.bbox import should_merge_blocks, merge_boxes
from marker.debug.data import dump_equation_debug_data
from marker.settings import settings
from marker.schema import Page, Span, Line, Block, BlockType
import os

# 设置环境变量，禁用 tokenizers 的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载处理器
processor = load_processor()

# 加载 Texify 模型
def load_texify_model():
    texify_model = load_model(checkpoint=settings.TEXIFY_MODEL_NAME, device=settings.TORCH_DEVICE_MODEL, dtype=settings.TEXIFY_DTYPE)
    return texify_model

# 创建遮罩区域
def mask_bbox(png_image, bbox, selected_bboxes):
    # 创建一个与图片大小相同的灰度图像
    mask = Image.new('L', png_image.size, 0)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(mask)
    first_x = bbox[0]
    first_y = bbox[1]
    bbox_height = bbox[3] - bbox[1]
    bbox_width = bbox[2] - bbox[0]

    for box in selected_bboxes:
        # 将框适配到选定区域
        new_box = (box[0] - first_x, box[1] - first_y, box[2] - first_x, box[3] - first_y)
        # 将遮罩适配到图像边界与 PDF 边界
        resized = (
           new_box[0] / bbox_width * png_image.size[0],
           new_box[1] / bbox_height * png_image.size[1],
           new_box[2] / bbox_width * png_image.size[0],
           new_box[3] / bbox_height * png_image.size[1]
        )
        draw.rectangle(resized, fill=255)

    # 通过遮罩创建结果图像
    result = Image.composite(png_image, Image.new('RGBA', png_image.size, 'white'), mask)
    return result

# 获取遮罩后的图像
def get_masked_image(page, bbox, selected_bboxes):
    # 获取页面的像素图
    pix = page.get_pixmap(dpi=settings.TEXIFY_DPI, clip=bbox)
    png = pix.pil_tobytes(format="PNG")
    png_image = Image.open(io.BytesIO(png))
    # 创建遮罩后的图像
    png_image = mask_bbox(png_image, bbox, selected_bboxes)
    png_image = png_image.convert("RGB")
    return png_image
# 批量处理 LaTeX 图像，根据指定的区域长度重新格式化，使用指定的模型进行转换
def get_latex_batched(images, reformat_region_lens, texify_model, batch_size):
    # 如果图像列表为空，则返回空列表
    if len(images) == 0:
        return []

    # 初始化预测结果列表
    predictions = [""] * len(images)

    # 按批次处理图像
    for i in range(0, len(images), batch_size):
        # 动态设置最大长度以节省推理时间
        min_idx = i
        max_idx = min(min_idx + batch_size, len(images))
        max_length = max(reformat_region_lens[min_idx:max_idx])
        max_length = min(max_length, settings.TEXIFY_MODEL_MAX)
        max_length += settings.TEXIFY_TOKEN_BUFFER

        # 对图像批次进行推理
        model_output = batch_inference(images[min_idx:max_idx], texify_model, processor, max_tokens=max_length)

        # 遍历模型输出
        for j, output in enumerate(model_output):
            token_count = get_total_texify_tokens(output)
            # 如果 token 数量超过最大长度减一，则将输出置为空字符串
            if token_count >= max_length - 1:
                output = ""

            # 计算图像索引
            image_idx = i + j
            predictions[image_idx] = output
    return predictions


# 获取文本中的总 LaTeX token 数量
def get_total_texify_tokens(text):
    tokenizer = processor.tokenizer
    tokens = tokenizer(text)
    return len(tokens["input_ids"])


# 查找页面中的数学公式区域
def find_page_equation_regions(pnum, page, block_types):
    i = 0
    # 提取数学公式区域的边界框
    equation_boxes = [b.bbox for b in block_types[pnum] if b.block_type == "Formula"]
    reformatted_blocks = set()
    reformat_regions = []
    block_lens = []
    return reformat_regions, block_lens


# 获取区域内的边界框
def get_bboxes_for_region(page, region):
    bboxes = []
    merged_box = None
    for idx in region:
        block = page.blocks[idx]
        bbox = block.bbox
        if merged_box is None:
            merged_box = bbox
        else:
            merged_box = merge_boxes(merged_box, bbox)
        bboxes.append(bbox)
    return bboxes, merged_box


# 替换页面块中的文本块为 LaTeX
def replace_blocks_with_latex(page_blocks: Page, merged_boxes, reformat_regions, predictions, pnum):
    new_blocks = []
    converted_spans = []
    current_region = 0
    idx = 0
    success_count = 0
    fail_count = 0
    # 当索引小于页面块列表的长度时，继续循环
    while idx < len(page_blocks.blocks):
        # 获取当前索引对应的页面块
        block = page_blocks.blocks[idx]
        # 如果当前区域索引超过重新格式化区域列表的长度，或者当前索引小于重新格式化区域的起始索引
        if current_region >= len(reformat_regions) or idx < reformat_regions[current_region][0]:
            # 将当前页面块添加到新的块列表中
            new_blocks.append(block)
            # 索引加一
            idx += 1
            # 继续下一次循环
            continue

        # 获取重新格式化区域的原始文本
        orig_block_text = " ".join([page_blocks.blocks[i].prelim_text for i in reformat_regions[current_region]])
        # 获取预测的 LaTeX 文本
        latex_text = predictions[current_region]
        # 定义条件列表
        conditions = [
            len(latex_text) > 0,
            get_total_texify_tokens(latex_text) < settings.TEXIFY_MODEL_MAX,  # 确保没有达到总体令牌最大值
            len(latex_text) > len(orig_block_text) * .8,
            len(latex_text.strip()) > 0
        ]

        # 更新索引为重新格式化区域的结束索引加一
        idx = reformat_regions[current_region][-1] + 1
        # 如果条件不满足
        if not all(conditions):
            # 失败计数加一
            fail_count += 1
            # 将转换后的区域添加为 None
            converted_spans.append(None)
            # 将重新格式化区域中的页面块添加到新的块列表中
            for i in reformat_regions[current_region]:
                new_blocks.append(page_blocks.blocks[i])
        else:
            # 成功计数加一
            success_count += 1
            # 创建一个包含 LaTeX 文本的行对象
            block_line = Line(
                spans=[
                    Span(
                        text=latex_text,
                        bbox=merged_boxes[current_region],
                        span_id=f"{pnum}_{idx}_fixeq",
                        font="Latex",
                        color=0,
                        block_type="Formula"
                    )
                ],
                bbox=merged_boxes[current_region]
            )
            # 深拷贝第一个 span 对象并添加到转换后的区域列表中
            converted_spans.append(deepcopy(block_line.spans[0]))
            # 创建一个新的块对象，包含上述行对象
            new_blocks.append(Block(
                lines=[block_line],
                bbox=merged_boxes[current_region],
                pnum=pnum
            ))
        # 更新当前区域索引
        current_region += 1
    # 返回新的块列表、成功计数、失败计数和转换后的区域列表
    return new_blocks, success_count, fail_count, converted_spans
def replace_equations(doc, blocks: List[Page], block_types: List[List[BlockType]], texify_model, batch_size=settings.TEXIFY_BATCH_SIZE):
    # 初始化未成功 OCR 的计数和成功 OCR 的计数
    unsuccessful_ocr = 0
    successful_ocr = 0

    # 查找潜在的方程区域，并计算每个区域中文本的长度
    reformat_regions = []
    reformat_region_lens = []
    for pnum, page in enumerate(blocks):
        regions, region_lens = find_page_equation_regions(pnum, page, block_types)
        reformat_regions.append(regions)
        reformat_region_lens.append(region_lens)

    # 计算方程的总数
    eq_count = sum([len(x) for x in reformat_regions])

    # 获取每个区域的图像
    flat_reformat_region_lens = [item for sublist in reformat_region_lens for item in sublist]
    images = []
    merged_boxes = []
    for page_idx, reformat_regions_page in enumerate(reformat_regions):
        page_obj = doc[page_idx]
        for reformat_region in reformat_regions_page:
            bboxes, merged_box = get_bboxes_for_region(blocks[page_idx], reformat_region)
            png_image = get_masked_image(page_obj, merged_box, bboxes)
            images.append(png_image)
            merged_boxes.append(merged_box)

    # 进行批量预测
    predictions = get_latex_batched(images, flat_reformat_region_lens, texify_model, batch_size)

    # 替换区域中的文本块为预测结果
    page_start = 0
    converted_spans = []
    # 遍历重排后的区域列表，获取每一页的预测结果和合并后的框
    for page_idx, reformat_regions_page in enumerate(reformat_regions):
        # 获取当前页的预测结果和合并后的框
        page_predictions = predictions[page_start:page_start + len(reformat_regions_page)]
        page_boxes = merged_boxes[page_start:page_start + len(reformat_regions_page)]
        # 替换块内容为 LaTeX，并返回新的块列表、成功计数、失败计数和转换的跨度
        new_page_blocks, success_count, fail_count, converted_span = replace_blocks_with_latex(
            blocks[page_idx],
            page_boxes,
            reformat_regions_page,
            page_predictions,
            page_idx
        )
        # 将转换的跨度添加到列表中
        converted_spans.extend(converted_span)
        # 更新当前页的块列表
        blocks[page_idx].blocks = new_page_blocks
        # 更新页起始位置
        page_start += len(reformat_regions_page)
        # 更新成功 OCR 计数和失败 OCR 计数
        successful_ocr += success_count
        unsuccessful_ocr += fail_count

    # 如果调试模式开启，输出转换结果以供比较
    dump_equation_debug_data(doc, images, converted_spans)

    # 返回更新后的块列表和 OCR 结果统计信息
    return blocks, {"successful_ocr": successful_ocr, "unsuccessful_ocr": unsuccessful_ocr, "equations": eq_count}
```