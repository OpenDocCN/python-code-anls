# `.\marker\marker\ordering.py`

```py
# 导入必要的模块
from copy import deepcopy
from typing import List
import torch
import sys, os
from marker.extract_text import convert_single_page
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
from PIL import Image
import io
from marker.schema import Page
from marker.settings import settings

# 从设置中加载 LayoutLMv3Processor 模型
processor = LayoutLMv3Processor.from_pretrained(settings.ORDERER_MODEL_NAME)

# 加载 LayoutLMv3ForSequenceClassification 模型
def load_ordering_model():
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        settings.ORDERER_MODEL_NAME,
        torch_dtype=settings.MODEL_DTYPE,
    ).to(settings.TORCH_DEVICE_MODEL)
    model.eval()
    return model

# 获取推理数据
def get_inference_data(page, page_blocks: Page):
    # 深拷贝页面块的边界框
    bboxes = deepcopy([block.bbox for block in page_blocks.blocks])
    # 初始化单词列表
    words = ["."] * len(bboxes)

    # 获取页面的像素图像
    pix = page.get_pixmap(dpi=settings.LAYOUT_DPI, annots=False, clip=page_blocks.bbox)
    # 将像素图像转换为 PNG 格式
    png = pix.pil_tobytes(format="PNG")
    # 将 PNG 数据转换为 RGB 图像
    rgb_image = Image.open(io.BytesIO(png)).convert("RGB")

    # 获取页面块的边界框和宽高
    page_box = page_blocks.bbox
    pwidth = page_blocks.width
    pheight = page_blocks.height

    # 调整边界框的值
    for box in bboxes:
        if box[0] < page_box[0]:
            box[0] = page_box[0]
        if box[1] < page_box[1]:
            box[1] = page_box[1]
        if box[2] > page_box[2]:
            box[2] = page_box[2]
        if box[3] > page_box[3]:
            box[3] = page_box[3]

        # 将边界框的值转换为相对于页面宽高的比例
        box[0] = int(box[0] / pwidth * 1000)
        box[1] = int(box[1] / pheight * 1000)
        box[2] = int(box[2] / pwidth * 1000)
        box[3] = int(box[3] / pheight * 1000)

    return rgb_image, bboxes, words

# 批量推理
def batch_inference(rgb_images, bboxes, words, model):
    # 对 RGB 图像、单词和边界框进行编码
    encoding = processor(
        rgb_images,
        text=words,
        boxes=bboxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # 将像素值转换为模型的数据类型
    encoding["pixel_values"] = encoding["pixel_values"].to(model.dtype)
    # 进入推断模式，不进行梯度计算
    with torch.inference_mode():
        # 将指定的键对应的值移动到模型所在设备上
        for k in ["bbox", "input_ids", "pixel_values", "attention_mask"]:
            encoding[k] = encoding[k].to(model.device)
        # 使用模型进行推理，获取输出
        outputs = model(**encoding)
        # 获取模型输出的预测结果
        logits = outputs.logits

    # 获取预测结果中概率最大的类别索引，并转换为列表
    predictions = logits.argmax(-1).squeeze().tolist()
    # 如果预测结果是整数，则转换为列表
    if isinstance(predictions, int):
        predictions = [predictions]
    # 将预测结果转换为类别标签
    predictions = [model.config.id2label[p] for p in predictions]
    # 返回预测结果
    return predictions
# 为文档中的每个块添加列数计数
def add_column_counts(doc, doc_blocks, model, batch_size):
    # 按照批量大小遍历文档块
    for i in range(0, len(doc_blocks), batch_size):
        # 创建当前批量的索引范围
        batch = range(i, min(i + batch_size, len(doc_blocks)))
        # 初始化空列表用于存储 RGB 图像、边界框和单词
        rgb_images = []
        bboxes = []
        words = []
        # 遍历当前批量的页码
        for pnum in batch:
            # 获取推理数据：RGB 图像、页边界框和页单词
            page = doc[pnum]
            rgb_image, page_bboxes, page_words = get_inference_data(page, doc_blocks[pnum])
            rgb_images.append(rgb_image)
            bboxes.append(page_bboxes)
            words.append(page_words)

        # 进行批量推理，获取预测结果
        predictions = batch_inference(rgb_images, bboxes, words, model)
        # 将预测结果与页码对应，更新文档块的列数计数
        for pnum, prediction in zip(batch, predictions):
            doc_blocks[pnum].column_count = prediction

# 对文档块进行排序
def order_blocks(doc, doc_blocks: List[Page], model, batch_size=settings.ORDERER_BATCH_SIZE):
    # 添加列数计数
    add_column_counts(doc, doc_blocks, model, batch_size)

    # 遍历文档块中的每一页
    for page_blocks in doc_blocks:
        # 如果该页的列数大于1
        if page_blocks.column_count > 1:
            # 根据位置重新排序块
            split_pos = page_blocks.x_start + page_blocks.width / 2
            left_blocks = []
            right_blocks = []
            # 遍历该页的每个块
            for block in page_blocks.blocks:
                # 根据位置将块分为左右两部分
                if block.x_start <= split_pos:
                    left_blocks.append(block)
                else:
                    right_blocks.append(block)
            # 更新该页的块顺序
            page_blocks.blocks = left_blocks + right_blocks
    # 返回排序后的文档块
    return doc_blocks
```