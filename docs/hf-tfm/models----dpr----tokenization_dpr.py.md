# `.\models\dpr\tokenization_dpr.py`

```py
# coding=utf-8
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


import collections
from typing import List, Optional, Union

from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging
from ..bert.tokenization_bert import BertTokenizer


# 获取名为 logging 的日志记录器对象
logger = logging.get_logger(__name__)

# 定义词汇文件名字典，包括词汇文件和分词器文件
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 上下文编码器预训练模型的词汇文件映射字典
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

# 问题编码器预训练模型的词汇文件映射字典
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

# 读者预训练模型的词汇文件映射字典
READER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-reader-single-nq-base": (
            "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/vocab.txt"
        ),
        "facebook/dpr-reader-multiset-base": (
            "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/vocab.txt"
        ),
    },
    # tokenizer_file 字典包含两个条目，每个条目的键是模型名称，值是其对应的 tokenizer.json 文件的 URL
    "tokenizer_file": {
        "facebook/dpr-reader-single-nq-base": (
            "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/tokenizer.json"
        ),
        "facebook/dpr-reader-multiset-base": (
            "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/tokenizer.json"
        ),
    },
}

# 定义用于不同模型的预训练位置嵌入大小的映射字典
CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-ctx_encoder-single-nq-base": 512,
    "facebook/dpr-ctx_encoder-multiset-base": 512,
}
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-question_encoder-single-nq-base": 512,
    "facebook/dpr-question_encoder-multiset-base": 512,
}
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-reader-single-nq-base": 512,
    "facebook/dpr-reader-multiset-base": 512,
}

# 定义用于不同模型的预训练初始化配置的字典
CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-ctx_encoder-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-ctx_encoder-multiset-base": {"do_lower_case": True},
}
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-question_encoder-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-question_encoder-multiset-base": {"do_lower_case": True},
}
READER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-reader-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-reader-multiset-base": {"do_lower_case": True},
}

# 定义一个自定义的类，继承自BertTokenizer，用于DPR上下文编码器的分词器
class DPRContextEncoderTokenizer(BertTokenizer):
    r"""
    Construct a DPRContextEncoder tokenizer.

    [`DPRContextEncoderTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """

    # 设置词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 设置预训练词汇文件的映射字典
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    # 设置最大模型输入大小的映射字典
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 设置预训练初始化配置的映射字典
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION

# 定义一个自定义的类，继承自BertTokenizer，用于DPR问题编码器的分词器
class DPRQuestionEncoderTokenizer(BertTokenizer):
    r"""
    Constructs a DPRQuestionEncoder tokenizer.

    [`DPRQuestionEncoderTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """

    # 设置词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 设置预训练词汇文件的映射字典
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    # 设置最大模型输入大小的映射字典
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 设置预训练初始化配置的映射字典
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION

# 定义一个命名元组，用于存储DPR阅读器的输出结果
DPRSpanPrediction = collections.namedtuple(
    "DPRSpanPrediction", ["span_score", "relevance_score", "doc_id", "start_index", "end_index", "text"]
)

# 定义一个命名元组，用于存储DPR阅读器的输出结果
DPRReaderOutput = collections.namedtuple("DPRReaderOutput", ["start_logits", "end_logits", "relevance_logits"])

# 自定义DPR阅读器文档字符串
CUSTOM_DPR_READER_DOCSTRING = r"""
    Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`.
    It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers),
    """
    Prepares input data for a question answering model by tokenizing passages and creating input IDs and attention masks.
    
    Returns:
        `Dict[str, List[List[int]]]`: A dictionary containing the following keys:
        
        - `input_ids`: List of token IDs formatted as `[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>`.
        - `attention_mask`: List indicating which tokens should be attended to by the model.
    """
# 将自定义的文档字符串添加到类的注释中
@add_start_docstrings(CUSTOM_DPR_READER_DOCSTRING)
# 定义一个混合类，用于处理DPR阅读器的定制标记器
class CustomDPRReaderTokenizerMixin:
    # 实现__call__方法，使类实例可以像函数一样调用
    def __call__(
        self,
        questions,  # 输入的问题或问题列表
        titles: Optional[str] = None,  # 输入的标题或标题列表（可选）
        texts: Optional[str] = None,  # 输入的文本或文本列表（可选）
        padding: Union[bool, str] = False,  # 是否填充到最大长度或指定填充方法
        truncation: Union[bool, str] = False,  # 是否截断到最大长度或指定截断方法
        max_length: Optional[int] = None,  # 最大序列长度（可选）
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型（可选）
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力遮罩（可选）
        **kwargs,  # 其他关键字参数
    ) -> BatchEncoding:  # 返回类型为BatchEncoding对象
        # 如果标题和文本均为空，则调用父类的__call__方法
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
        # 如果标题或文本有一个为空，则处理成对的标题-文本
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
        # 如果标题和文本均为单个字符串，则转换为列表
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)  # 获取标题的数量
        # 如果问题是单个字符串，则复制为与标题数量相匹配的列表
        questions = questions if not isinstance(questions, str) else [questions] * n_passages
        # 检查标题和文本的数量是否相同，不同则引发值错误
        if len(titles) != len(texts):
            raise ValueError(
                f"There should be as many titles than texts but got {len(titles)} titles and {len(texts)} texts."
            )
        # 获取问题和标题的编码输入（input_ids）
        encoded_question_and_titles = super().__call__(questions, titles, padding=False, truncation=False)["input_ids"]
        # 获取文本的编码输入（input_ids），不添加特殊标记
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        # 构建编码输入字典
        encoded_inputs = {
            "input_ids": [
                (encoded_question_and_title + encoded_text)[:max_length]  # 若截断则截断到最大长度
                if max_length is not None and truncation
                else encoded_question_and_title + encoded_text  # 否则直接拼接
                for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
            ]
        }
        # 如果需要返回attention_mask，则生成对应的attention_mask
        if return_attention_mask is not False:
            attention_mask = []
            for input_ids in encoded_inputs["input_ids"]:
                attention_mask.append([int(input_id != self.pad_token_id) for input_id in input_ids])
            encoded_inputs["attention_mask"] = attention_mask
        # 调用pad方法进行填充处理，并返回结果
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
        解码最佳跨度，用于从抽取式问答模型中找出一个段落的最佳答案跨度。
        按照 `span_score` 降序排列，保留最大的 `top_spans` 个跨度。忽略超过 `max_answer_length` 的跨度。
        """
        scores = []
        for start_index, start_score in enumerate(start_logits):
            for answer_length, end_score in enumerate(end_logits[start_index : start_index + max_answer_length]):
                scores.append(((start_index, start_index + answer_length), start_score + end_score))
        # 根据得分降序排序所有跨度
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []
        for (start_index, end_index), score in scores:
            # 检查跨度索引的合法性
            if start_index > end_index:
                raise ValueError(f"Wrong span indices: [{start_index}:{end_index}]")
            length = end_index - start_index + 1
            # 检查跨度长度是否超过最大答案长度
            if length > max_answer_length:
                raise ValueError(f"Span is too long: {length} > {max_answer_length}")
            # 检查是否存在重叠的跨度
            if any(
                start_index <= prev_start_index <= prev_end_index <= end_index
                or prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals
            ):
                continue
            chosen_span_intervals.append((start_index, end_index))

            # 如果已选出了指定数量的跨度，则停止
            if len(chosen_span_intervals) == top_spans:
                break
        return chosen_span_intervals
# 使用自定义的文档字符串装饰器来添加文档字符串到类定义中，基于给定的自定义文档字符串
@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
# 定义一个类 DPRReaderTokenizer，继承自 CustomDPRReaderTokenizerMixin 和 BertTokenizer
class DPRReaderTokenizer(CustomDPRReaderTokenizerMixin, BertTokenizer):
    """
    Construct a DPRReader tokenizer.

    [`DPRReaderTokenizer`] is almost identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts that are
    combined to be fed to the [`DPRReader`] model.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """

    # 类属性：词汇表文件名列表，值为 VOCAB_FILES_NAMES
    vocab_files_names = VOCAB_FILES_NAMES
    # 类属性：预训练词汇文件映射，值为 READER_PRETRAINED_VOCAB_FILES_MAP
    pretrained_vocab_files_map = READER_PRETRAINED_VOCAB_FILES_MAP
    # 类属性：最大模型输入尺寸列表，值为 READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    max_model_input_sizes = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 类属性：预训练初始化配置，值为 READER_PRETRAINED_INIT_CONFIGURATION
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION
    # 类属性：模型输入名称列表，包含 "input_ids" 和 "attention_mask"
    model_input_names = ["input_ids", "attention_mask"]
```