# `.\pipelines\document_question_answering.py`

```
# 导入正则表达式模块
import re
# 导入类型提示相关模块
from typing import List, Optional, Tuple, Union

# 导入第三方库 numpy
import numpy as np

# 导入自定义工具函数和类
from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_pytesseract_available,
    is_torch_available,
    is_vision_available,
    logging,
)

# 导入基础类 ChunkPipeline 和函数 build_pipeline_init_args
from .base import ChunkPipeline, build_pipeline_init_args
# 导入问题回答相关函数 select_starts_ends
from .question_answering import select_starts_ends

# 如果视觉处理库可用，则导入 PIL 图像处理模块和 load_image 函数
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果 PyTorch 可用，则导入 PyTorch 库
if is_torch_available():
    import torch
    from ..models.auto.modeling_auto import MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES

# 初始化 TESSERACT_LOADED 标志
TESSERACT_LOADED = False
# 如果 pytesseract 可用，则将 TESSERACT_LOADED 设置为 True，并导入 pytesseract 库
if is_pytesseract_available():
    TESSERACT_LOADED = True
    import pytesseract

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# normalize_box() 和 apply_tesseract() 函数从 models/layoutlmv3/feature_extraction_layoutlmv3.py 中的 apply_tesseract 派生而来。
# 由于管道可能会从 layoutlmv3 当前的实现中演变，因此此处将其复制（而非导入），以避免创建不必要的依赖关系。

def normalize_box(box, width, height):
    """根据图像宽度和高度，归一化边界框的坐标值，并返回归一化后的边界框列表。"""
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def apply_tesseract(image: "Image.Image", lang: Optional[str], tesseract_config: Optional[str]):
    """对文档图像应用 Tesseract OCR，返回识别的单词及其归一化的边界框。"""
    # 应用 OCR
    data = pytesseract.image_to_data(image, lang=lang, output_type="dict", config=tesseract_config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # 过滤空单词及其对应的坐标
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # 将坐标转换为 (left, top, left+width, top+height) 格式
    actual_boxes = []
    # 使用 zip 函数并行迭代 left, top, width, height 四个列表，每次迭代取出一个元组 (x, y, w, h)
    for x, y, w, h in zip(left, top, width, height):
        # 根据左上角坐标和宽高计算出实际边界框的坐标 [left, top, right, bottom]
        actual_box = [x, y, x + w, y + h]
        # 将计算得到的实际边界框添加到 actual_boxes 列表中
        actual_boxes.append(actual_box)

    # 获取图像的宽度和高度
    image_width, image_height = image.size

    # 创建一个空列表来存储标准化后的边界框
    normalized_boxes = []
    # 遍历所有实际边界框，对每个边界框调用 normalize_box 函数进行标准化处理
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    # 检查单词列表和标准化边界框列表的长度是否相等，如果不相等则抛出 ValueError 异常
    if len(words) != len(normalized_boxes):
        raise ValueError("Not as many words as there are bounding boxes")

    # 返回处理后的单词列表和标准化后的边界框列表作为结果
    return words, normalized_boxes
class ModelType(ExplicitEnum):
    LayoutLM = "layoutlm"
    LayoutLMv2andv3 = "layoutlmv2andv3"
    VisionEncoderDecoder = "vision_encoder_decoder"


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True, has_tokenizer=True))
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
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This document question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"document-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a document question answering task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=document-question-answering).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 检查是否提供了非快速的分词器，如果提供了，抛出值错误异常
        if self.tokenizer is not None and not self.tokenizer.__class__.__name__.endswith("Fast"):
            raise ValueError(
                "`DocumentQuestionAnsweringPipeline` requires a fast tokenizer, but a slow tokenizer "
                f"(`{self.tokenizer.__class__.__name__}`) is provided."
            )

        # 如果模型配置为 VisionEncoderDecoderConfig 类型，则设置模型类型为 VisionEncoderDecoder
        if self.model.config.__class__.__name__ == "VisionEncoderDecoderConfig":
            self.model_type = ModelType.VisionEncoderDecoder
            # 如果模型编码器类型不是 "donut-swin"，则抛出值错误异常
            if self.model.config.encoder.model_type != "donut-swin":
                raise ValueError("Currently, the only supported VisionEncoderDecoder model is Donut")
        else:
            # 否则，检查模型类型是否在 DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES 中
            self.check_model_type(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES)
            # 如果模型配置为 LayoutLMConfig 类型，则设置模型类型为 LayoutLM
            if self.model.config.__class__.__name__ == "LayoutLMConfig":
                self.model_type = ModelType.LayoutLM
            else:
                # 否则，设置模型类型为 LayoutLMv2andv3
                self.model_type = ModelType.LayoutLMv2andv3
    # 对输入参数进行清理和预处理，返回预处理参数和空的后处理参数字典
    def _sanitize_parameters(
        self,
        padding=None,  # 如果指定了填充参数，设置预处理参数字典中的填充
        doc_stride=None,  # 如果指定了文档步幅参数，设置预处理参数字典中的文档步幅
        max_question_len=None,  # 如果指定了最大问题长度参数，设置预处理参数字典中的最大问题长度
        lang: Optional[str] = None,  # 如果指定了语言参数，设置预处理参数字典中的语言
        tesseract_config: Optional[str] = None,  # 如果指定了 Tesseract 配置参数，设置预处理参数字典中的 Tesseract 配置
        max_answer_len=None,  # 如果指定了最大答案长度参数，设置后处理参数字典中的最大答案长度
        max_seq_len=None,  # 如果指定了最大序列长度参数，设置预处理参数字典中的最大序列长度
        top_k=None,  # 如果指定了 top_k 参数，设置后处理参数字典中的 top_k
        handle_impossible_answer=None,  # 如果指定了处理不可能答案的参数，设置后处理参数字典中的处理方式
        timeout=None,  # 如果指定了超时参数，设置预处理参数字典中的超时时间
        **kwargs,  # 其他未命名的参数，不做特定处理
    ):
        preprocess_params, postprocess_params = {}, {}

        if padding is not None:
            preprocess_params["padding"] = padding
        if doc_stride is not None:
            preprocess_params["doc_stride"] = doc_stride
        if max_question_len is not None:
            preprocess_params["max_question_len"] = max_question_len
        if max_seq_len is not None:
            preprocess_params["max_seq_len"] = max_seq_len
        if lang is not None:
            preprocess_params["lang"] = lang
        if tesseract_config is not None:
            preprocess_params["tesseract_config"] = tesseract_config
        if timeout is not None:
            preprocess_params["timeout"] = timeout

        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")
            postprocess_params["top_k"] = top_k
        if max_answer_len is not None:
            if max_answer_len < 1:
                raise ValueError(f"max_answer_len parameter should be >= 1 (got {max_answer_len}")
            postprocess_params["max_answer_len"] = max_answer_len
        if handle_impossible_answer is not None:
            postprocess_params["handle_impossible_answer"] = handle_impossible_answer

        return preprocess_params, {}, postprocess_params

    # 处理调用对象的输入，支持图片或文件路径、问题文本、词框列表等输入
    def __call__(
        self,
        image: Union["Image.Image", str],  # 图片或文件路径
        question: Optional[str] = None,  # 可选的问题文本
        word_boxes: Tuple[str, List[float]] = None,  # 包含词框的元组
        **kwargs,  # 其他未命名的参数，不做特定处理
    ):
    
    # 对输入进行预处理，支持输入、填充方式、文档步幅、最大序列长度、词框列表、语言、Tesseract 配置及超时设置
    def preprocess(
        self,
        input,
        padding="do_not_pad",  # 默认不填充
        doc_stride=None,  # 可选的文档步幅
        max_seq_len=None,  # 可选的最大序列长度
        word_boxes: Tuple[str, List[float]] = None,  # 可选的词框列表
        lang=None,  # 可选的语言设置
        tesseract_config="",  # 默认空的 Tesseract 配置
        timeout=None,  # 可选的超时设置
    ):
    
    # 执行模型的前向传播，处理模型输入和生成参数
    def _forward(self, model_inputs, **generate_kwargs):
        p_mask = model_inputs.pop("p_mask", None)  # 弹出并获取模型输入中的 p_mask
        word_ids = model_inputs.pop("word_ids", None)  # 弹出并获取模型输入中的 word_ids
        words = model_inputs.pop("words", None)  # 弹出并获取模型输入中的 words
        is_last = model_inputs.pop("is_last", False)  # 弹出并获取模型输入中的 is_last，默认为 False

        if self.model_type == ModelType.VisionEncoderDecoder:
            model_outputs = self.model.generate(**model_inputs, **generate_kwargs)  # 生成视觉编码器解码器模型的输出
        else:
            model_outputs = self.model(**model_inputs)  # 调用普通模型的前向传播

        model_outputs = dict(model_outputs.items())  # 将模型输出转换为字典形式
        model_outputs["p_mask"] = p_mask  # 将 p_mask 放回模型输出
        model_outputs["word_ids"] = word_ids  # 将 word_ids 放回模型输出
        model_outputs["words"] = words  # 将 words 放回模型输出
        model_outputs["attention_mask"] = model_inputs.get("attention_mask", None)  # 获取模型输入中的 attention_mask 并放入模型输出
        model_outputs["is_last"] = is_last  # 将 is_last 放回模型输出
        return model_outputs  # 返回模型输出
    # 根据模型类型确定后处理方法，对模型输出进行处理并返回答案列表
    def postprocess(self, model_outputs, top_k=1, **kwargs):
        if self.model_type == ModelType.VisionEncoderDecoder:
            # 如果模型类型是 VisionEncoderDecoder，则调用相应的单一处理方法
            answers = [self.postprocess_encoder_decoder_single(o) for o in model_outputs]
        else:
            # 否则，调用抽取式问答的后处理方法
            answers = self.postprocess_extractive_qa(model_outputs, top_k=top_k, **kwargs)

        # 按照答案的分数从高到低进行排序，并选取前 top_k 个答案
        answers = sorted(answers, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        return answers

    # 处理单个 VisionEncoderDecoder 模型输出的后处理方法
    def postprocess_encoder_decoder_single(self, model_outputs, **kwargs):
        # 解码模型输出的序列为文本
        sequence = self.tokenizer.batch_decode(model_outputs["sequences"])[0]

        # TODO: A lot of this logic is specific to Donut and should probably be handled in the tokenizer
        # (see https://github.com/huggingface/transformers/pull/18414/files#r961747408 for more context).
        
        # 以下逻辑大部分特定于 Donut，可能应该在 tokenizer 中处理
        # 参考链接：https://github.com/huggingface/transformers/pull/18414/files#r961747408
        
        # 替换序列中的 eos_token 和 pad_token
        sequence = sequence.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
        # 使用正则表达式移除第一个任务开始标记之后的内容
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        ret = {
            "answer": None,
        }

        # 从序列中寻找 <s_answer>...</s_answer> 匹配的内容作为答案
        answer = re.search(r"<s_answer>(.*)</s_answer>", sequence)
        if answer is not None:
            ret["answer"] = answer.group(1).strip()
        return ret

    # 处理抽取式问答模型输出的后处理方法
    def postprocess_extractive_qa(
        self, model_outputs, top_k=1, handle_impossible_answer=False, max_answer_len=15, **kwargs
    ):
        # 设置一个较大的初始空值分数
        min_null_score = 1000000  # large and positive
        answers = []
        for output in model_outputs:
            words = output["words"]

            # 选择起始和结束位置，并更新最小空值分数
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
            word_ids = output["word_ids"]
            for start, end, score in zip(starts, ends, scores):
                word_start, word_end = word_ids[start], word_ids[end]
                if word_start is not None and word_end is not None:
                    # 将答案及其相关信息添加到答案列表中
                    answers.append(
                        {
                            "score": float(score),
                            "answer": " ".join(words[word_start : word_end + 1]),
                            "start": word_start,
                            "end": word_end,
                        }
                    )

        # 如果处理不可能的答案，则将最小空值分数的答案添加到答案列表中
        if handle_impossible_answer:
            answers.append({"score": min_null_score, "answer": "", "start": 0, "end": 0})

        return answers
```