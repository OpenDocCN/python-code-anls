# `.\transformers\pipelines\document_question_answering.py`

```py
# 2022年版权声明

# 导入正则表达式模块
import re
# 导入类型提示模块中的特定类和函数
from typing import List, Optional, Tuple, Union
# 导入numpy模块
import numpy as np
# 从模块中导入特定功能
from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_pytesseract_available,
    is_torch_available,
    is_vision_available,
    logging,
)
# 从base模块中导入特定类
from .base import PIPELINE_INIT_ARGS, ChunkPipeline
# 从question_answering模块中导入特定功能
from .question_answering import select_starts_ends

# 如果引入了图像处理模块
if is_vision_available():
    # 从PIL模块中导入Image类
    from PIL import Image
    # 从image_utils模块中导入特定功能
    from ..image_utils import load_image

# 如果引入了torch模块
if is_torch_available():
    # 从torch模块中导入特定功能
    import torch
    # 从模型自动模块中导入特定功能
    from ..models.auto.modeling_auto import MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES

# 定义变量TESSERACT_LOADED
TESSERACT_LOADED = False
# 如果引入了pytesseract模块
if is_pytesseract_available():
    TESSERACT_LOADED = True
    # 从pytesseract模块中导入特定功能
    import pytesseract

# 获取日志记录器
logger = logging.get_logger(__name__)

# 对normalize_box()和apply_tesseract()进行了从models/layoutlmv3/feature_extraction_layoutlmv3.py中apply_tesseract得出的修改
# 然而，因为pipeline可能会从layoutlmv3当前所做的事情发展，所以它是被拷贝而不是被引入，以避免产生不必要的依赖关系。
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

# 定义函数apply_tesseract
def apply_tesseract(image: "Image.Image", lang: Optional[str], tesseract_config: Optional[str]):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    # 运用 OCR
    data = pytesseract.image_to_data(image, lang=lang, output_type="dict", config=tesseract_config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # 过滤空单词和对应的坐标
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # 将坐标转换为(left, top, left+width, top+height)格式
    actual_boxes = []
    # 遍历列表中的 left、top、width、height 四个列表，分别赋值给 x、y、w、h
    for x, y, w, h in zip(left, top, width, height):
        # 计算当前框的实际坐标，即左上角和右下角的坐标
        actual_box = [x, y, x + w, y + h]
        # 将当前框的实际坐标添加到实际框列表中
        actual_boxes.append(actual_box)

    # 获取图像的宽度和高度
    image_width, image_height = image.size

    # 最终，将边界框归一化
    normalized_boxes = []
    # 遍历实际框列表中的每个框
    for box in actual_boxes:
        # 将当前框进行归一化处理，并添加到归一化框列表中
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    # 如果单词列表的长度与归一化框列表的长度不相等，则抛出 ValueError 异常
    if len(words) != len(normalized_boxes):
        raise ValueError("Not as many words as there are bounding boxes")

    # 返回单词列表和归一化框列表
    return words, normalized_boxes
# 定义了一个枚举类 ModelType，用于表示模型类型
class ModelType(ExplicitEnum):
    LayoutLM = "layoutlm"
    LayoutLMv2andv3 = "layoutlmv2andv3"
    VisionEncoderDecoder = "vision_encoder_decoder"

# 利用装饰器 @add_end_docstrings(PIPELINE_INIT_ARGS) 添加额外的文档字符串描述到 DocumentQuestionAnsweringPipeline 类
class DocumentQuestionAnsweringPipeline(ChunkPipeline):
    # TODO: Update task_summary docs to include an example with document QA and then update the first sentence
    """
    Document Question Answering pipeline using any `AutoModelForDocumentQuestionAnswering`. The inputs/outputs are
    similar to the (extractive) question answering pipeline; however, the pipeline takes an image (and optional OCR'd
    words/boxes) as input instead of text context.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> document_qa = pipeline(model="impira/layoutlm-document-qa")
    >>> document_qa(
    ...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    ...     question="What is the invoice number?",
    ... )
    [{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]
    ```py

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This document question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"document-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a document question answering task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=document-question-answering).
    """
    
    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 调用 ChunkPipeline 父类的初始化方法
        super().__init__(*args, **kwargs)
        # 检查是否使用的是快速分词器，如果不是则抛出错误
        if self.tokenizer is not None and not self.tokenizer.__class__.__name__.endswith("Fast"):
            raise ValueError(
                "`DocumentQuestionAnsweringPipeline` requires a fast tokenizer, but a slow tokenizer "
                f"(`{self.tokenizer.__class__.__name__}`) is provided."
            )
        # 检查模型配置是否为 VisionEncoderDecoderConfig，如果是则设置模型类型为 VisionEncoderDecoder
        if self.model.config.__class__.__name__ == "VisionEncoderDecoderConfig":
            self.model_type = ModelType.VisionEncoderDecoder
            if self.model.config.encoder.model_type != "donut-swin":
                raise ValueError("Currently, the only supported VisionEncoderDecoder model is Donut")
        else:
            # 检查模型类型是否匹配 MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES 列表中的模型
            self.check_model_type(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES)
            # 如果模型配置是 LayoutLMConfig 类型则设置模型类型为 LayoutLM，否则设置为 LayoutLMv2andv3
            if self.model.config.__class__.__name__ == "LayoutLMConfig":
                self.model_type = ModelType.LayoutLM
            else:
                self.model_type = ModelType.LayoutLMv2andv3

    # 处理参数的方法
    def _sanitize_parameters(
        self,
        padding=None,
        doc_stride=None,
        max_question_len=None,
        lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        max_answer_len=None,
        max_seq_len=None,
        top_k=None,
        handle_impossible_answer=None,
        timeout=None,
        **kwargs,
    # 初始化预处理和后处理参数字典
    preprocess_params, postprocess_params = {}, {}
    # 如果存在填充参数，添加到预处理参数字典中
    if padding is not None:
        preprocess_params["padding"] = padding
    # 如果存在文档步距参数，添加到预处理参数字典中
    if doc_stride is not None:
        preprocess_params["doc_stride"] = doc_stride
    # 如果存在最大问题长度参数，添加到预处理参数字典中
    if max_question_len is not None:
        preprocess_params["max_question_len"] = max_question_len
    # 如果存在最大序列长度参数，添加到预处理参数字典中
    if max_seq_len is not None:
        preprocess_params["max_seq_len"] = max_seq_len
    # 如果存在语言参数，添加到预处理参数字典中
    if lang is not None:
        preprocess_params["lang"] = lang
    # 如果存在tesseract配置参数，添加到预处理参数字典中
    if tesseract_config is not None:
        preprocess_params["tesseract_config"] = tesseract_config
    # 如果存在超时参数，添加到预处理参数字典中
    if timeout is not None:
        preprocess_params["timeout"] = timeout

    # 如果存在top_k参数，检查是否大于等于1，若不是，抛出异常
    if top_k is not None:
        if top_k < 1:
            raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")
        # 添加top_k参数到后处理参数字典中
        postprocess_params["top_k"] = top_k
    # 如果存在最大答案长度参数，检查是否大于等于1，若不是，抛出异常
    if max_answer_len is not None:
        if max_answer_len < 1:
            raise ValueError(f"max_answer_len parameter should be >= 1 (got {max_answer_len}")
        # 添加最大答案长度参数到后处理参数字典中
        postprocess_params["max_answer_len"] = max_answer_len
    # 如果存在处理不可能答案参数，添加到后处理参数字典中
    if handle_impossible_answer is not None:
        postprocess_params["handle_impossible_answer"] = handle_impossible_answer

    # 返回预处理参数，空字典，后处理参数
    return preprocess_params, {}, postprocess_params


    # 定义__call__方法，接受图像、问题、单词框等输入
    def __call__(
        self,
        image: Union["Image.Image", str],
        question: Optional[str] = None,
        word_boxes: Tuple[str, List[float]] = None,
        **kwargs,


    # 定义预处理方法，接受输入，填充参数、文档步距、最大序列长度、单词框、语言、tesseract配置、超时等参数
    def preprocess(
        self,
        input,
        padding="do_not_pad",
        doc_stride=None,
        max_seq_len=None,
        word_boxes: Tuple[str, List[float]] = None,
        lang=None,
        tesseract_config="",
        timeout=None,


    # 定义_forward方法，接受模型输入，p_mask、word_ids、words、is_last参数
    def _forward(self, model_inputs):
        # 弹出p_mask，word_ids，words，is_last参数
        p_mask = model_inputs.pop("p_mask", None)
        word_ids = model_inputs.pop("word_ids", None)
        words = model_inputs.pop("words", None)
        is_last = model_inputs.pop("is_last", False)

        # 如果模型类型是VisionEncoderDecoder，执行generate方法
        if self.model_type == ModelType.VisionEncoderDecoder:
            model_outputs = self.model.generate(**model_inputs)
        else:
            model_outputs = self.model(**model_inputs)

        # 将模型输出组成字典
        model_outputs = dict(model_outputs.items())
        model_outputs["p_mask"] = p_mask
        model_outputs["word_ids"] = word_ids
        model_outputs["words"] = words
        model_outputs["attention_mask"] = model_inputs.get("attention_mask", None)
        model_outputs["is_last"] = is_last
        # 返回模型输出
        return model_outputs


    # 定义后处理方法，接受模型输出和top_k参数等
    def postprocess(self, model_outputs, top_k=1, **kwargs):
        # 如果模型类型是VisionEncoderDecoder，执行postprocess_encoder_decoder_single方法
        if self.model_type == ModelType.VisionEncoderDecoder:
            answers = [self.postprocess_encoder_decoder_single(o) for o in model_outputs]
        else:
            # 执行postprocess_extractive_qa方法
            answers = self.postprocess_extractive_qa(model_outputs, top_k=top_k, **kwargs)

        # 根据得分倒序排列答案，并取前top_k个
        answers = sorted(answers, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        # 返回答案
        return answers
    # 对编码解码模型的预测结果进行后处理，输出格式化的答案
    def postprocess_encoder_decoder_single(self, model_outputs, **kwargs):
        # 将模型输出处理为字符串序列
        sequence = self.tokenizer.batch_decode(model_outputs["sequences"])[0]

        # TODO: 大部分逻辑是针对Donut模型的，可能应该在tokenizer中处理
        # (参见 https://github.com/huggingface/transformers/pull/18414/files#r961747408 获取更多上下文)
        # 从序列中移除结束标记和填充标记
        sequence = sequence.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
        # 从序列中移除第一个任务起始标记
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        # 初始化返回结果字典
        ret = {
            "answer": None,
        }

        # 从序列中提取答案
        answer = re.search(r"<s_answer>(.*)</s_answer>", sequence)
        if answer is not None:
            # 将答案加入返回结果字典
            ret["answer"] = answer.group(1).strip()
        # 返回结果字典
        return ret

    # 对抽取式问答模型的预测结果进行后处理，输出格式化的答案列表
    def postprocess_extractive_qa(
        self, model_outputs, top_k=1, handle_impossible_answer=False, max_answer_len=15, **kwargs
    ):
        # 初始化最小空答案得分
        min_null_score = 1000000  # large and positive
        # 初始化答案列表
        answers = []
        # 遍历模型输出列表
        for output in model_outputs:
            # 获取词汇表
            words = output["words"]

            # 根据开始和结束位置选择答案，并更新最小空答案得分
            starts, ends, scores, min_null_score = select_starts_ends(
                start=output["start_logits"],
                end=output["end_logits"],
                p_mask=output["p_mask"],
                attention_mask=output["attention_mask"].numpy()
                if output.get("attention_mask", None) is not None
                else None,
                min_null_score=min_null_score,
                top_k=top_k,
                handle_impossible_answer=handle_impossible_answer,
                max_answer_len=max_answer_len,
            )
            # 获取单词id列表
            word_ids = output["word_ids"]
            # 遍历开始和结束位置
            for start, end, score in zip(starts, ends, scores):
                # 获取单词的开始和结束索引
                word_start, word_end = word_ids[start], word_ids[end]
                if word_start is not None and word_end is not None:
                    # 将答案添加到答案列表
                    answers.append(
                        {
                            "score": float(score),
                            "answer": " ".join(words[word_start : word_end + 1]),
                            "start": word_start,
                            "end": word_end,
                        }
                    )

        # 如果需要处理无法回答的问题，则加入一个空答案
        if handle_impossible_answer:
            answers.append({"score": min_null_score, "answer": "", "start": 0, "end": 0})

        # 返回答案列表
        return answers
```