# `.\models\dpr\tokenization_dpr_fast.py`

```py
# coding=utf-8
# 代码文件的编码声明，使用UTF-8格式

# 版权声明和许可信息，指明代码的版权归属和许可条件
# Copyright 2018 The HuggingFace Inc. team, The Hugging Face Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes for DPR."""
# 本模块提供了DPR模型的tokenization类

# 导入必要的库和模块
import collections
from typing import List, Optional, Union

# 导入基础的tokenization工具和辅助函数
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging

# 导入BERT模型的快速tokenization类
from ..bert.tokenization_bert_fast import BertTokenizerFast

# 导入DPR模型的tokenization类
from .tokenization_dpr import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRReaderTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 上下文编码器预训练模型的词汇文件映射
CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": (
            "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/vocab.txt"
        ),
        "facebook/dpr-ctx_encoder-multiset-base": (
            "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": (
            "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/tokenizer.json"
        ),
        "facebook/dpr-ctx_encoder-multiset-base": (
            "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/tokenizer.json"
        ),
    },
}

# 问题编码器预训练模型的词汇文件映射
QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-question_encoder-single-nq-base": (
            "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/vocab.txt"
        ),
        "facebook/dpr-question_encoder-multiset-base": (
            "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "facebook/dpr-question_encoder-single-nq-base": (
            "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/tokenizer.json"
        ),
        "facebook/dpr-question_encoder-multiset-base": (
            "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/tokenizer.json"
        ),
    },
}

# 阅读器预训练模型的词汇文件映射
READER_PRETRAINED_VOCAB_FILES_MAP = {
    # 定义一个字典，存储不同模型名称到其对应的词汇表文件 URL 的映射关系
    "vocab_file": {
        "facebook/dpr-reader-single-nq-base": (
            "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/vocab.txt"
        ),
        "facebook/dpr-reader-multiset-base": (
            "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/vocab.txt"
        ),
    },
    # 定义一个字典，存储不同模型名称到其对应的分词器文件 URL 的映射关系
    "tokenizer_file": {
        "facebook/dpr-reader-single-nq-base": (
            "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/tokenizer.json"
        ),
        "facebook/dpr-reader-multiset-base": (
            "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/tokenizer.json"
        ),
    },
}

CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-ctx_encoder-single-nq-base": 512,  # 定义上下文编码器单一模型的位置编码大小为512
    "facebook/dpr-ctx_encoder-multiset-base": 512,   # 定义上下文编码器多集模型的位置编码大小为512
}
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-question_encoder-single-nq-base": 512,   # 定义问题编码器单一模型的位置编码大小为512
    "facebook/dpr-question_encoder-multiset-base": 512,    # 定义问题编码器多集模型的位置编码大小为512
}
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-reader-single-nq-base": 512,     # 定义阅读器单一模型的位置编码大小为512
    "facebook/dpr-reader-multiset-base": 512,      # 定义阅读器多集模型的位置编码大小为512
}


CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-ctx_encoder-single-nq-base": {"do_lower_case": True},    # 上下文编码器单一模型的初始化配置，设置为小写敏感
    "facebook/dpr-ctx_encoder-multiset-base": {"do_lower_case": True},     # 上下文编码器多集模型的初始化配置，设置为小写敏感
}
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-question_encoder-single-nq-base": {"do_lower_case": True},   # 问题编码器单一模型的初始化配置，设置为小写敏感
    "facebook/dpr-question_encoder-multiset-base": {"do_lower_case": True},    # 问题编码器多集模型的初始化配置，设置为小写敏感
}
READER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-reader-single-nq-base": {"do_lower_case": True},    # 阅读器单一模型的初始化配置，设置为小写敏感
    "facebook/dpr-reader-multiset-base": {"do_lower_case": True},     # 阅读器多集模型的初始化配置，设置为小写敏感
}


class DPRContextEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" DPRContextEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRContextEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES   # 设置词汇文件的名称列表为已定义的全局变量 VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP   # 预训练词汇文件的映射表为上下文编码器的预定义映射
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES   # 最大模型输入尺寸为上下文编码器的位置编码大小
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION   # 预训练模型的初始化配置为上下文编码器的初始化配置
    slow_tokenizer_class = DPRContextEncoderTokenizer   # 慢速分词器类为 DPRContextEncoderTokenizer


class DPRQuestionEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a "fast" DPRQuestionEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRQuestionEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES   # 设置词汇文件的名称列表为已定义的全局变量 VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP   # 预训练词汇文件的映射表为问题编码器的预定义映射
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES   # 最大模型输入尺寸为问题编码器的位置编码大小
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION   # 预训练模型的初始化配置为问题编码器的初始化配置
    slow_tokenizer_class = DPRQuestionEncoderTokenizer   # 慢速分词器类为 DPRQuestionEncoderTokenizer


DPRSpanPrediction = collections.namedtuple(
    "DPRSpanPrediction", ["span_score", "relevance_score", "doc_id", "start_index", "end_index", "text"]
)

DPRReaderOutput = collections.namedtuple("DPRReaderOutput", ["start_logits", "end_logits", "relevance_logits"])


CUSTOM_DPR_READER_DOCSTRING = r"""
    # 返回一个包含输入字符串的token id及其他信息的字典，用于传递给 `.decode_best_spans` 函数。
    # 使用分词器和词汇表将问题和不同段落（标题和文本）的字符串转换为一系列整数ID。结果的 `input_ids` 是一个大小为 `(n_passages, sequence_length)` 的矩阵，
    # 其格式为：
    #
    # [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>
    #
    # 返回：
    # `Dict[str, List[List[int]]]`: 包含以下键的字典：
    #
    # - `input_ids`: 要输入模型的token id列表。
    # - `attention_mask`: 指定模型应关注哪些token的索引列表。
# 将自定义的文档字符串添加到类上，通常用于API文档生成
@add_start_docstrings(CUSTOM_DPR_READER_DOCSTRING)
# 定义一个混合类，用于处理DPR Reader的自定义Tokenizer功能
class CustomDPRReaderTokenizerMixin:
    # 定义__call__方法，使对象可以像函数一样调用
    def __call__(
        self,
        questions,  # 输入的问题或问题列表
        titles: Optional[str] = None,  # 可选参数，输入的标题或单个标题字符串
        texts: Optional[str] = None,  # 可选参数，输入的文本或单个文本字符串
        padding: Union[bool, str] = False,  # 是否进行填充，可以是布尔值或填充策略字符串
        truncation: Union[bool, str] = False,  # 是否进行截断，可以是布尔值或截断策略字符串
        max_length: Optional[int] = None,  # 可选参数，最大长度限制
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力遮罩
        **kwargs,  # 其他未命名的关键字参数
    ) -> BatchEncoding:  # 返回值为BatchEncoding类型的对象
        # 如果标题和文本均未提供，则直接调用父类的__call__方法
        if titles is None and texts is None:
            return super().__call__(
                questions,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        # 如果标题或文本中有一个为None，则将其作为文本对处理
        elif titles is None or texts is None:
            text_pair = titles if texts is None else texts
            return super().__call__(
                questions,
                text_pair,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        # 如果titles是字符串，则转换为列表
        titles = titles if not isinstance(titles, str) else [titles]
        # 如果texts是字符串，则转换为列表
        texts = texts if not isinstance(texts, str) else [texts]
        # 计算标题的数量，作为文本对的数量
        n_passages = len(titles)
        # 如果问题是字符串，则复制为问题列表，使每个问题对应一个文本对
        questions = questions if not isinstance(questions, str) else [questions] * n_passages
        # 断言标题和文本的数量应该相同
        assert len(titles) == len(
            texts
        ), f"There should be as many titles than texts but got {len(titles)} titles and {len(texts)} texts."
        # 调用父类的__call__方法对问题和标题进行编码，禁用填充和截断
        encoded_question_and_titles = super().__call__(questions, titles, padding=False, truncation=False)["input_ids"]
        # 调用父类的__call__方法对文本进行编码，禁用特殊令牌、填充和截断
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        # 合并编码后的问题和标题与文本，并根据最大长度和截断策略进行处理
        encoded_inputs = {
            "input_ids": [
                (encoded_question_and_title + encoded_text)[:max_length]
                if max_length is not None and truncation
                else encoded_question_and_title + encoded_text
                for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
            ]
        }
        # 如果不返回注意力遮罩，则创建注意力遮罩列表
        if return_attention_mask is not False:
            attention_mask = []
            for input_ids in encoded_inputs["input_ids"]:
                attention_mask.append([int(input_id != self.pad_token_id) for input_id in input_ids])
            encoded_inputs["attention_mask"] = attention_mask
        # 调用pad方法对编码输入进行填充，根据填充策略和最大长度进行处理
        return self.pad(encoded_inputs, padding=padding, max_length=max_length, return_tensors=return_tensors)
    def decode_best_spans(
        self,
        reader_input: BatchEncoding,
        reader_output: DPRReaderOutput,
        num_spans: int = 16,
        max_answer_length: int = 64,
        num_spans_per_passage: int = 4,
    ):
        """
        解码最佳跨度的函数，用于从抽取式问答模型中找出一个段落的最佳答案跨度。它按照降序的 `span_score` 排序，并保留最多 `top_spans` 个跨度。超过 `max_answer_length` 的跨度将被忽略。
        """
        scores = []
        for start_index, start_score in enumerate(start_logits):
            for answer_length, end_score in enumerate(end_logits[start_index : start_index + max_answer_length]):
                scores.append(((start_index, start_index + answer_length), start_score + end_score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []
        for (start_index, end_index), score in scores:
            assert start_index <= end_index, f"Wrong span indices: [{start_index}:{end_index}]"
            length = end_index - start_index + 1
            assert length <= max_answer_length, f"Span is too long: {length} > {max_answer_length}"
            if any(
                start_index <= prev_start_index <= prev_end_index <= end_index
                or prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals
            ):
                continue
            chosen_span_intervals.append((start_index, end_index))

            if len(chosen_span_intervals) == top_spans:
                break
        return chosen_span_intervals
# 应用装饰器 @add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING) 来添加自定义文档字符串到类 DPRReaderTokenizerFast
@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
# 声明 DPRReaderTokenizerFast 类，继承自 CustomDPRReaderTokenizerMixin 和 BertTokenizerFast
class DPRReaderTokenizerFast(CustomDPRReaderTokenizerMixin, BertTokenizerFast):
    # 构造函数说明
    r"""
    构造一个“快速” DPRReader 分词器（由 HuggingFace 的 *tokenizers* 库支持）。

    [`DPRReaderTokenizerFast`] 几乎与 [`BertTokenizerFast`] 相同，并运行端到端的分词：
    标点符号拆分和 wordpiece。区别在于它有三个输入字符串：问题、标题和文本，这些被组合后供 [`DPRReader`] 模型使用。

    参考超类 [`BertTokenizerFast`] 以获取有关参数的使用示例和文档。

    """

    # 定义词汇文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练词汇文件的映射
    pretrained_vocab_files_map = READER_PRETRAINED_VOCAB_FILES_MAP
    # 定义模型最大输入大小
    max_model_input_sizes = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义预训练初始化配置
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速分词器类的定义
    slow_tokenizer_class = DPRReaderTokenizer
```