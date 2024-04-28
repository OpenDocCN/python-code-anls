# `.\models\dpr\tokenization_dpr.py`

```py
# 设置文件编码和版权信息
# 版权声明
# 许可证信息

"""DPR 的标记类。"""

# 导入必要的库和模块
import collections
from typing import List, Optional, Union

# 导入其他模块提供的函数和类
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging
from ..bert.tokenization_bert import BertTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 上下文编码器预训练词汇文件映射
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

# 问题编码器预训练词汇文件映射
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

# 读者预训练词汇文件映射
READER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-reader-single-nq-base": (
            "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/vocab.txt"
        ),
        "facebook/dpr-reader-multiset-base": (
            "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/vocab.txt"
        ),
    },
    # 定义一个字典，用于存储模型名称与其对应的tokenizer文件链接
    "tokenizer_file": {
        # 模型名称 "facebook/dpr-reader-single-nq-base" 对应的tokenizer文件链接
        "facebook/dpr-reader-single-nq-base": (
            "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/tokenizer.json"
        ),
        # 模型名称 "facebook/dpr-reader-multiset-base" 对应的tokenizer文件链接
        "facebook/dpr-reader-multiset-base": (
            "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/tokenizer.json"
        ),
    },
# This section contains constant variables for the sizes and configurations of the pretrained position embeddings for the context encoder, question encoder, and reader
# The values correspond to different models from the "facebook/dpr" family

# Size of pretrained positional embeddings for the context encoder
CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-ctx_encoder-single-nq-base": 512,  # Size for a specific model
    "facebook/dpr-ctx_encoder-multiset-base": 512,  # Size for another specific model
}

# Size of pretrained positional embeddings for the question encoder
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-question_encoder-single-nq-base": 512,  # Size for a specific model
    "facebook/dpr-question_encoder-multiset-base": 512,  # Size for another specific model
}

# Size of pretrained positional embeddings for the reader
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-reader-single-nq-base": 512,  # Size for a specific model
    "facebook/dpr-reader-multiset-base": 512,  # Size for another specific model
}

# Configuration of pretrained initialization for the context encoder
CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-ctx_encoder-single-nq-base": {"do_lower_case": True},  # Configuration for a specific model
    "facebook/dpr-ctx_encoder-multiset-base": {"do_lower_case": True},  # Configuration for another specific model
}

# Configuration of pretrained initialization for the question encoder
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-question_encoder-single-nq-base": {"do_lower_case": True},  # Configuration for a specific model
    "facebook/dpr-question_encoder-multiset-base": {"do_lower_case": True},  # Configuration for another specific model
}

# Configuration of pretrained initialization for the reader
READER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-reader-single-nq-base": {"do_lower_case": True},  # Configuration for a specific model
    "facebook/dpr-reader-multiset-base": {"do_lower_case": True},  # Configuration for another specific model
}

# Definition of a tokenizer for the DPR Context Encoder
class DPRContextEncoderTokenizer(BertTokenizer):
    r"""
    Construct a DPRContextEncoder tokenizer.

    [`DPRContextEncoderTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """

    # Set vocab files names
    vocab_files_names = VOCAB_FILES_NAMES
    # Set pretrained vocab files map
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    # Set max model input sizes
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # Set pretrained initialization configuration
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION

# Definition of a tokenizer for the DPR Question Encoder
class DPRQuestionEncoderTokenizer(BertTokenizer):
    r"""
    Constructs a DPRQuestionEncoder tokenizer.

    [`DPRQuestionEncoderTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """

    # Set vocab files names
    vocab_files_names = VOCAB_FILES_NAMES
    # Set pretrained vocab files map
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    # Set max model input sizes
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # Set pretrained initialization configuration
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION

# Definition of a named tuple representing the output of the DPR reader
DPRReaderOutput = collections.namedtuple("DPRReaderOutput", ["start_logits", "end_logits", "relevance_logits"])

# This documentation describes the purpose of the `CUSTOM_DPR_READER_DOCSTRING` function
CUSTOM_DPR_READER_DOCSTRING = r"""
    Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`.
    It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers),
    # 使用分词器和词汇表生成输入序列的编码
    # 结果的 input_ids 是一个大小为 (n_passages, sequence_length) 的矩阵，格式如下：
    # [CLS] <问题的编码> [SEP] <标题的编码> [SEP] <文本的编码>
    # 返回一个字典，包含以下键值对：
    # - input_ids: 要输入模型的标记 id 的列表
    # - attention_mask: 指定哪些标记需要被模型关注的索引列表
# 添加自定义的DPR Reader文档字符串的描述到类上
@add_start_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class CustomDPRReaderTokenizerMixin:
    # 定义了一个可调用的方法，接收多个参数，并返回BatchEncoding对象
    def __call__(
        self,
        questions,  # 问题
        titles: Optional[str] = None,  # 标题，可选参数
        texts: Optional[str] = None,  # 文本内容，可选参数
        padding: Union[bool, str] = False,  # 填充方式，可选参数
        truncation: Union[bool, str] = False,  # 截断方式，可选参数
        max_length: Optional[int] = None,  # 最大长度，可选参数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型，可选参数
        return_attention_mask: Optional[bool] = None,  # 返回注意力遮罩，可选参数
        **kwargs,  # 其他参数
    ) -> BatchEncoding:  # 返回BatchEncoding对象
        # 如果标题和文本内容都为空，则只处理问题
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
        # 如果标题或文本内容其中一个为空，则处理问题和另一个不为空的内容
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
        # 如果标题和文本内容都不为空，则处理多个标题和文本
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)  # 获取标题数量
        questions = questions if not isinstance(questions, str) else [questions] * n_passages
        # 如果标题的数量和文本数量不一致，则抛出数值错误
        if len(titles) != len(texts):
            raise ValueError(
                f"There should be as many titles than texts but got {len(titles)} titles and {len(texts)} texts."
            )
        # 对问题和标题进行编码处理
        encoded_question_and_titles = super().__call__(questions, titles, padding=False, truncation=False)["input_ids"]
        # 对文本内容进行编码处理
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        # 将编码后的问题和标题及文本内容组合，返回编码输入
        encoded_inputs = {
            "input_ids": [
                (encoded_question_and_title + encoded_text)[:max_length]
                if max_length is not None and truncation
                else encoded_question_and_title + encoded_text
                for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
            ]
        }
        # 如果不返回注意力遮罩，则生成注意力遮罩
        if return_attention_mask is not False:
            attention_mask = []
            for input_ids in encoded_inputs["input_ids"]:
                attention_mask.append([int(input_id != self.pad_token_id) for input_id in input_ids])
            encoded_inputs["attention_mask"] = attention_mask
        # 执行填充处理，返回填充后的输入
        return self.pad(encoded_inputs, padding=padding, max_length=max_length, return_tensors=return_tensors)
    # 定义一个方法来解码最佳抽取式问题回答的跨度
    def decode_best_spans(
        self,
        reader_input: BatchEncoding,
        reader_output: DPRReaderOutput,
        num_spans: int = 16,
        max_answer_length: int = 64,
        num_spans_per_passage: int = 4,
        
    # 定义一个私有方法来获取最佳跨度
    def _get_best_spans(
        self,
        start_logits: List[int],
        end_logits: List[int],
        max_answer_length: int,
        top_spans: int,
    ) -> List[DPRSpanPrediction]:
        """
        找到一个段落中抽取式问答模型的最佳回答跨度。按照`span_score`的降序顺序返回最佳跨度，并保留最大的`top_spans`个跨度。忽略超过`max_answer_length`的跨度。
        """
        scores = []
        for start_index, start_score in enumerate(start_logits):
            for answer_length, end_score in enumerate(end_logits[start_index: start_index + max_answer_length]):
                scores.append(((start_index, start_index + answer_length), start_score + end_score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []
        for (start_index, end_index), score in scores:
            # 检查索引范围是否合法
            if start_index > end_index:
                raise ValueError(f"Wrong span indices: [{start_index}:{end_index}]")
            length = end_index - start_index + 1
            # 检查跨度长度是否超过最大长度
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

            if len(chosen_span_intervals) == top_spans:
                break
        return chosen_span_intervals
# 使用装饰器add_end_docstrings()将自定义文档字符串添加到DPRReaderTokenizer类上
@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
# 定义DPRReaderTokenizer类，继承自CustomDPRReaderTokenizerMixin和BertTokenizer类
class DPRReaderTokenizer(CustomDPRReaderTokenizerMixin, BertTokenizer):
    r"""
    构造一个DPRReader分词器。

    [`DPRReaderTokenizer`]几乎与[`BertTokenizer`]相同，并且运行端到端的分词：标点分割和词块分割。 不同之处在于它有三个输入字符串：问题，标题和文本，这些字符串被组合并送入[`DPRReader`]模型。

    可参考超类[`BertTokenizer`]的用法示例和参数说明。
    """

    # 设置词汇文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 设置预训练词汇文件映射
    pretrained_vocab_files_map = READER_PRETRAINED_VOCAB_FILES_MAP
    # 设置模型最大输入尺寸
    max_model_input_sizes = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 设置预训练初始化配置
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION
    # 设置模型输入名称
    model_input_names = ["input_ids", "attention_mask"]
```