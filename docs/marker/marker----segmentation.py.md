# `.\marker\marker\segmentation.py`

```py
# 导入所需的库
from concurrent.futures import ThreadPoolExecutor
from typing import List

from transformers import LayoutLMv3ForTokenClassification

# 导入自定义的模块
from marker.bbox import unnormalize_box
from transformers.models.layoutlmv3.image_processing_layoutlmv3 import normalize_box
import io
from PIL import Image
from transformers import LayoutLMv3Processor
import numpy as np
from marker.settings import settings
from marker.schema import Page, BlockType
import torch
from math import isclose

# 设置图像最大像素值，避免部分图像被截断
Image.MAX_IMAGE_PIXELS = None

# 从预训练模型加载 LayoutLMv3Processor
processor = LayoutLMv3Processor.from_pretrained(settings.LAYOUT_MODEL_NAME, apply_ocr=False)

# 定义需要分块的键和不需要分块的键
CHUNK_KEYS = ["input_ids", "attention_mask", "bbox", "offset_mapping"]
NO_CHUNK_KEYS = ["pixel_values"]

# 加载 LayoutLMv3ForTokenClassification 模型
def load_layout_model():
    # 从预训练模型加载 LayoutLMv3ForTokenClassification 模型
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        settings.LAYOUT_MODEL_NAME,
        torch_dtype=settings.MODEL_DTYPE,
    ).to(settings.TORCH_DEVICE_MODEL)

    # 设置模型的标签映射
    model.config.id2label = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title"
    }

    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    return model

# 检测文档块类型
def detect_document_block_types(doc, blocks: List[Page], layoutlm_model, batch_size=settings.LAYOUT_BATCH_SIZE):
    # 获取特征编码、元数据和样本长度
    encodings, metadata, sample_lengths = get_features(doc, blocks)
    # 预测块类型
    predictions = predict_block_types(encodings, layoutlm_model, batch_size)
    # 将预测结果与框匹配
    block_types = match_predictions_to_boxes(encodings, predictions, metadata, sample_lengths, layoutlm_model)
    # 断言块类型数量与块数量相等
    assert len(block_types) == len(blocks)
    return block_types

# 获取临时框
def get_provisional_boxes(pred, box, is_subword, start_idx=0):
    # 从预测结果中获取临时框
    prov_predictions = [pred_ for idx, pred_ in enumerate(pred) if not is_subword[idx]][start_idx:]
    # 从列表中筛选出不是子词的元素，并从指定索引开始切片，得到新的列表
    prov_boxes = [box_ for idx, box_ in enumerate(box) if not is_subword[idx]][start_idx:]
    # 返回处理后的预测结果和框
    return prov_predictions, prov_boxes
# 获取页面编码信息，输入参数为页面和页面块对象
def get_page_encoding(page, page_blocks: Page):
    # 如果页面块中的所有行数为0，则返回空列表
    if len(page_blocks.get_all_lines()) == 0:
        return [], []

    # 获取页面块的边界框、宽度和高度
    page_box = page_blocks.bbox
    pwidth = page_blocks.width
    pheight = page_blocks.height

    # 获取页面块的像素图，并转换为 PNG 格式
    pix = page.get_pixmap(dpi=settings.LAYOUT_DPI, annots=False, clip=page_blocks.bbox)
    png = pix.pil_tobytes(format="PNG")
    png_image = Image.open(io.BytesIO(png))
    # 如果图像太大，则缩小以适应模型
    rgb_image = png_image.convert('RGB')
    rgb_width, rgb_height = rgb_image.size

    # 确保图像大小与 PDF 页面的比例正确
    assert isclose(rgb_width / pwidth, rgb_height / pheight, abs_tol=2e-2)

    # 获取页面块中的所有行
    lines = page_blocks.get_all_lines()

    boxes = []
    text = []
    for line in lines:
        box = line.bbox
        # 处理边界框溢出的情况
        if box[0] < page_box[0]:
            box[0] = page_box[0]
        if box[1] < page_box[1]:
            box[1] = page_box[1]
        if box[2] > page_box[2]:
            box[2] = page_box[2]
        if box[3] > page_box[3]:
            box[3] = page_box[3]

        # 处理边界框宽度或高度为0或负值的情况
        if box[2] <= box[0]:
            print("Zero width box found, cannot convert properly")
            raise ValueError
        if box[3] <= box[1]:
            print("Zero height box found, cannot convert properly")
            raise ValueError
        boxes.append(box)
        text.append(line.prelim_text)

    # 将边界框归一化为模型（缩放为1000x1000）
    boxes = [normalize_box(box, pwidth, pheight) for box in boxes]
    for box in boxes:
        # 验证所有边界框都是有效的
        assert(len(box) == 4)
        assert(max(box)) <= 1000
        assert(min(box)) >= 0
    # 使用 processor 处理 RGB 图像，传入文本、框、返回偏移映射等参数
    encoding = processor(
        rgb_image,
        text=text,
        boxes=boxes,
        return_offsets_mapping=True,
        truncation=True,
        return_tensors="pt",
        stride=settings.LAYOUT_CHUNK_OVERLAP,
        padding="max_length",
        max_length=settings.LAYOUT_MODEL_MAX,
        return_overflowing_tokens=True
    )
    # 从 encoding 中弹出 offset_mapping 和 overflow_to_sample_mapping
    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
    # 将 encoding 中的 bbox、input_ids、attention_mask、pixel_values 转换为列表
    bbox = list(encoding["bbox"])
    input_ids = list(encoding["input_ids"])
    attention_mask = list(encoding["attention_mask"])
    pixel_values = list(encoding["pixel_values"])

    # 断言各列表长度相等
    assert len(bbox) == len(input_ids) == len(attention_mask) == len(pixel_values) == len(offset_mapping)

    # 将各列表中的元素组成字典，放入 list_encoding 列表中
    list_encoding = []
    for i in range(len(bbox)):
        list_encoding.append({
            "bbox": bbox[i],
            "input_ids": input_ids[i],
            "attention_mask": attention_mask[i],
            "pixel_values": pixel_values[i],
            "offset_mapping": offset_mapping[i]
        })

    # 其他数据包括原始框、pwidth 和 pheight
    other_data = {
        "original_bbox": boxes,
        "pwidth": pwidth,
        "pheight": pheight,
    }
    # 返回 list_encoding 和 other_data
    return list_encoding, other_data
# 获取文档的特征信息
def get_features(doc, blocks):
    # 初始化编码、元数据和样本长度列表
    encodings = []
    metadata = []
    sample_lengths = []
    # 遍历每个块
    for i in range(len(blocks)):
        # 调用函数获取页面编码和其他数据
        encoding, other_data = get_page_encoding(doc[i], blocks[i])
        # 将页面编码添加到编码列表中
        encodings.extend(encoding)
        # 将其他数据添加到元数据列表中
        metadata.append(other_data)
        # 记录当前页面编码的长度
        sample_lengths.append(len(encoding))
    # 返回编码、元数据和样本长度
    return encodings, metadata, sample_lengths


# 预测块类型
def predict_block_types(encodings, layoutlm_model, batch_size):
    # 初始化所有预测结果列表
    all_predictions = []
    # 按批次处理编码
    for i in range(0, len(encodings), batch_size):
        # 计算当前批次的起始和结束索引
        batch_start = i
        batch_end = min(i + batch_size, len(encodings))
        # 获取当前批次的编码
        batch = encodings[batch_start:batch_end]

        # 构建模型输入
        model_in = {}
        for k in ["bbox", "input_ids", "attention_mask", "pixel_values"]:
            model_in[k] = torch.stack([b[k] for b in batch]).to(layoutlm_model.device)

        model_in["pixel_values"] = model_in["pixel_values"].to(layoutlm_model.dtype)

        # 进入推理模式
        with torch.inference_mode():
            # 使用模型进行推理
            outputs = layoutlm_model(**model_in)
            logits = outputs.logits

        # 获取预测结果
        predictions = logits.argmax(-1).squeeze().tolist()
        if len(predictions) == settings.LAYOUT_MODEL_MAX:
            predictions = [predictions]
        # 将预测结果添加到所有预测结果列表中
        all_predictions.extend(predictions)
    # 返回所有预测结果
    return all_predictions


# 将预测结果与框匹配
def match_predictions_to_boxes(encodings, predictions, metadata, sample_lengths, layoutlm_model) -> List[List[BlockType]]:
    # 断言编码、预测结果和样本长度的长度相等
    assert len(encodings) == len(predictions) == sum(sample_lengths)
    assert len(metadata) == len(sample_lengths)

    # 初始化页面起始索引和页面块类型列表
    page_start = 0
    page_block_types = []
    # 返回页面块类型列表
    return page_block_types
```