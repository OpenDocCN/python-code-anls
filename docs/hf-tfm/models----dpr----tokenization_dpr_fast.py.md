# `.\models\dpr\tokenization_dpr_fast.py`

```py
# 导入必要的模块和函数
import collections
from typing import List, Optional, Union

from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_dpr import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRReaderTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇表文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义上下文编码器预训练模型的词汇表文件资源地址映射
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

# 定义问题编码器预训练模型的词汇表文件资源地址映射
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

# 定义阅读器预训练模型的词汇表文件资源地址映射
READER_PRETRAINED_VOCAB_FILES_MAP = {
    ...
}
    # 定义一个字典，键为模型名称，值为对应的词汇表文件的 URL
    "vocab_file": {
        "facebook/dpr-reader-single-nq-base": (
            "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/vocab.txt"
        ),
        "facebook/dpr-reader-multiset-base": (
            "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/vocab.txt"
        ),
    },
    # 定义一个字典，键为模型名称，值为对应的分词器文件的 URL
    "tokenizer_file": {
        "facebook/dpr-reader-single-nq-base": (
            "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/tokenizer.json"
        ),
        "facebook/dpr-reader-multiset-base": (
            "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/tokenizer.json"
        ),
    },
# 定义了一个名为 CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 的字典，包含了预训练模型名称和对应的位置嵌入尺寸
CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-ctx_encoder-single-nq-base": 512,  # 单一查询的 DPR 上下文编码器模型的位置嵌入尺寸
    "facebook/dpr-ctx_encoder-multiset-base": 512,   # 多重集的 DPR 上下文编码器模型的位置嵌入尺寸
}

# 定义了一个名为 QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 的字典，包含了预训练模型名称和对应的位置嵌入尺寸
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-question_encoder-single-nq-base": 512,  # 单一查询的 DPR 问题编码器模型的位置嵌入尺寸
    "facebook/dpr-question_encoder-multiset-base": 512,   # 多重集的 DPR 问题编码器模型的位置嵌入尺寸
}

# 定义了一个名为 READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 的字典，包含了预训练模型名称和对应的位置嵌入尺寸
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-reader-single-nq-base": 512,   # 单一查询的 DPR 阅读器模型的位置嵌入尺寸
    "facebook/dpr-reader-multiset-base": 512,    # 多重集的 DPR 阅读器模型的位置嵌入尺寸
}


# 定义了一个名为 CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION 的字典，包含了预训练模型名称和对应的初始化配置
CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-ctx_encoder-single-nq-base": {"do_lower_case": True},    # 单一查询的 DPR 上下文编码器模型的初始化配置
    "facebook/dpr-ctx_encoder-multiset-base": {"do_lower_case": True},     # 多重集的 DPR 上下文编码器模型的初始化配置
}

# 定义了一个名为 QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION 的字典，包含了预训练模型名称和对应的初始化配置
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-question_encoder-single-nq-base": {"do_lower_case": True},  # 单一查询的 DPR 问题编码器模型的初始化配置
    "facebook/dpr-question_encoder-multiset-base": {"do_lower_case": True},   # 多重集的 DPR 问题编码器模型的初始化配置
}

# 定义了一个名为 READER_PRETRAINED_INIT_CONFIGURATION 的字典，包含了预训练模型名称和对应的初始化配置
READER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-reader-single-nq-base": {"do_lower_case": True},  # 单一查询的 DPR 阅读器模型的初始化配置
    "facebook/dpr-reader-multiset-base": {"do_lower_case": True},   # 多重集的 DPR 阅读器模型的初始化配置
}


# 定义了一个名为 DPRContextEncoderTokenizerFast 的类，继承自 BertTokenizerFast 类
class DPRContextEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" DPRContextEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRContextEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """
    
    # 定义了一系列类属性，指定了词汇文件的名称、预训练词汇文件的映射、模型输入的最大尺寸和预训练模型的初始化配置
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = DPRContextEncoderTokenizer


# 定义了一个名为 DPRQuestionEncoderTokenizerFast 的类，继承自 BertTokenizerFast 类
class DPRQuestionEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a "fast" DPRQuestionEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRQuestionEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    # 定义了一系列类属性，指定了词汇文件的名称、预训练词汇文件的映射、模型输入的最大尺寸和预训练模型的初始化配置
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = DPRQuestionEncoderTokenizer


# 定义了一个命名元组 DPRSpanPrediction，用于存储 DPR 模型的预测结果
DPRSpanPrediction = collections.namedtuple(
    "DPRSpanPrediction", ["span_score", "relevance_score", "doc_id", "start_index", "end_index", "text"]
)

# 定义了一个命名元组 DPRReaderOutput，用于存储 DPR 阅读器模型的输出结果
DPRReaderOutput = collections.namedtuple("DPRReaderOutput", ["start_logits", "end_logits", "relevance_logits"])


# 定义了一个名为 CUSTOM_DPR_READER_DOCSTRING 的字符串常量，用于自定义 DPR 阅读器模型的文档字符串
CUSTOM_DPR_READER_DOCSTRING = r"""
    返回一个字典，其中包含输入字符串的标记 ID 和其他信息，以便传递给`.decode_best_spans`。它将问题字符串和不同段落（标题和正文）的字符串转换为一系列 ID（整数），使用分词器和词汇表。生成的`input_ids`是一个大小为`(n_passages, sequence_length)`的矩阵，格式如下所示：
    
    [CLS] <问题标记 ID> [SEP] <标题标记 ID> [SEP] <正文标记 ID>
    
    返回：
    `Dict[str, List[List[int]]]`：一个字典，包含以下键：
    
    - `input_ids`：要输入模型的标记 ID 列表。
    - `attention_mask`：指定模型应注意哪些标记的索引列表。
# 将给定的自定义 DPR 读取器的文档字符串添加到类上
@add_start_docstrings(CUSTOM_DPR_READER_DOCSTRING)
# 定义一个混合类，用于自定义 DPR 读取器的 Tokenizer
class CustomDPRReaderTokenizerMixin:
    # 定义 __call__ 方法，用于调用 Tokenizer 对象进行编码
    def __call__(
        self,
        questions,  # 问题文本或问题文本列表
        titles: Optional[str] = None,  # 标题文本或标题文本列表（可选）
        texts: Optional[str] = None,  # 正文文本或正文文本列表（可选）
        padding: Union[bool, str] = False,  # 是否填充序列（可选）
        truncation: Union[bool, str] = False,  # 是否截断序列（可选）
        max_length: Optional[int] = None,  # 最大序列长度（可选）
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型（可选）
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码（可选）
        **kwargs,  # 其他关键字参数
    ) -> BatchEncoding:  # 返回编码后的批次
        # 如果标题和正文都为 None，则只编码问题文本
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
        # 如果标题或正文有一个为 None，则将文本作为标题-正文对进行编码
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
        # 如果标题和正文都不为 None，则分别处理标题和正文
        titles = titles if not isinstance(titles, str) else [titles]  # 将标题转换为列表
        texts = texts if not isinstance(texts, str) else [texts]  # 将正文转换为列表
        n_passages = len(titles)  # 标题的数量即为段落的数量
        # 如果问题是字符串，则将其复制 n_passages 次
        questions = questions if not isinstance(questions, str) else [questions] * n_passages
        # 检查标题和正文的数量是否一致
        assert len(titles) == len(
            texts
        ), f"There should be as many titles than texts but got {len(titles)} titles and {len(texts)} texts."
        # 编码问题和标题，不进行填充和截断
        encoded_question_and_titles = super().__call__(questions, titles, padding=False, truncation=False)["input_ids"]
        # 编码正文，不添加特殊标记，不进行填充和截断
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        # 组合编码后的问题标题和正文
        encoded_inputs = {
            "input_ids": [
                (encoded_question_and_title + encoded_text)[:max_length]
                if max_length is not None and truncation
                else encoded_question_and_title + encoded_text
                for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
            ]
        }
        # 如果不返回注意力掩码，则跳过此步骤
        if return_attention_mask is not False:
            attention_mask = []
            # 为每个输入序列生成注意力掩码
            for input_ids in encoded_inputs["input_ids"]:
                attention_mask.append([int(input_id != self.pad_token_id) for input_id in input_ids])
            # 将注意力掩码添加到编码后的输入中
            encoded_inputs["attention_mask"] = attention_mask
        # 在需要的情况下对编码后的输入进行填充
        return self.pad(encoded_inputs, padding=padding, max_length=max_length, return_tensors=return_tensors)
    # 解码最佳答案跨度的方法，针对抽取式问答模型的一个段落。它按降序返回最佳跨度，并保留最大数量的 `top_spans` 跨度。超过 `max_answer_length` 的跨度将被忽略。
    def decode_best_spans(
        self,
        reader_input: BatchEncoding,
        reader_output: DPRReaderOutput,
        num_spans: int = 16,
        max_answer_length: int = 64,
        num_spans_per_passage: int = 4,
    # 获取最佳跨度的内部方法，根据起始和结束的概率分布。
    def _get_best_spans(
        self,
        start_logits: List[int],
        end_logits: List[int],
        max_answer_length: int,
        top_spans: int,
    ) -> List[DPRSpanPrediction]:
        """
        找到一个段落的抽取式 Q&A 模型的最佳答案跨度。它按 `span_score` 降序返回最佳跨度，并保留最大 `top_spans` 个跨度。
        超过 `max_answer_length` 的跨度将被忽略。
        """
        # 存储每个跨度的得分
        scores = []
        # 遍历起始位置的概率分布
        for start_index, start_score in enumerate(start_logits):
            # 遍历结束位置的概率分布，但限制最大答案长度
            for answer_length, end_score in enumerate(end_logits[start_index : start_index + max_answer_length]):
                # 计算当前跨度的得分，并将其添加到得分列表中
                scores.append(((start_index, start_index + answer_length), start_score + end_score))
        # 按照得分降序排列跨度
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # 存储已选择的跨度间隔
        chosen_span_intervals = []
        # 遍历排序后的跨度及其得分
        for (start_index, end_index), score in scores:
            # 确保起始索引小于等于结束索引
            assert start_index <= end_index, f"Wrong span indices: [{start_index}:{end_index}]"
            # 计算跨度长度
            length = end_index - start_index + 1
            # 确保跨度长度不超过最大答案长度
            assert length <= max_answer_length, f"Span is too long: {length} > {max_answer_length}"
            # 检查是否存在重叠的跨度
            if any(
                start_index <= prev_start_index <= prev_end_index <= end_index
                or prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals
            ):
                continue
            # 将当前跨度添加到已选择的跨度列表中
            chosen_span_intervals.append((start_index, end_index))

            # 如果已选择的跨度数量达到了设定的最大数量，就退出循环
            if len(chosen_span_intervals) == top_spans:
                break
        # 返回已选择的跨度间隔列表
        return chosen_span_intervals
# 导入 add_end_docstrings 函数并使用 CUSTOM_DPR_READER_DOCSTRING 作为参数为 DPRReaderTokenizerFast 类添加文档字符串
@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
# 定义 DPRReaderTokenizerFast 类，该类继承自 CustomDPRReaderTokenizerMixin 和 BertTokenizerFast
class DPRReaderTokenizerFast(CustomDPRReaderTokenizerMixin, BertTokenizerFast):
    r"""
    构造一个“快速”的 DPRReader 分词器（由 HuggingFace 的 *tokenizers* 库支持）。

    [`DPRReaderTokenizerFast`] 几乎与 [`BertTokenizerFast`] 相同，并且执行端到端的分词：标点符号分割和 WordPiece。区别在于它有三个输入字符串：问题、标题和文本，这些字符串被组合在一起供 [`DPRReader`] 模型使用。

    参考超类 [`BertTokenizerFast`] 以获取有关参数的使用示例和文档。

    """

    # 定义类属性，包含词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义类属性，包含预训练词汇文件映射
    pretrained_vocab_files_map = READER_PRETRAINED_VOCAB_FILES_MAP
    # 定义类属性，包含最大模型输入尺寸列表
    max_model_input_sizes = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义类属性，包含预训练初始化配置
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION
    # 定义类属性，包含模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 定义类属性，指定慢速分词器类
    slow_tokenizer_class = DPRReaderTokenizer
```