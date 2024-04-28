# `.\models\layoutlmv2\tokenization_layoutlmv2_fast.py`

```
# 设置编码格式为 UTF-8
# 版权声明
#
# 授权条款
# 本文件基于Apache许可证 2.0版授权, 除非在协议条款下不得使用本文件
# 您可以获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件按"原样"分发基础上
# 没有任何明示或暗示的担保或条件，包括但不限于
# 特定目的实用性，不违反权利或不侵犯权利

"""
快速tokenization 类别为 LayoutLMv2。重写了慢 tokenizer 类别的 2 个方法，即 _batch_encode_plus 和 _encode_plus，其中使用了 Rust tokenizer。
"""
import json  # 导入 json 模块
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示必需的模块

from tokenizers import normalizers  # 从 tokenizers 模块中导入 normalizers

from ...tokenization_utils_base import (  # 从 tokenization_utils_base 模块导入 BatchEncoding, EncodedInput, PaddingStrategy, PreTokenizedInput, TensorType, TextInput, TextInputPair, TruncationStrategy
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 从 tokenization_utils_fast 模块中导入 PreTrainedTokenizerFast
from ...utils import add_end_docstrings, logging  # 从 utils 模块中导入 add_end_docstrings, logging
from .tokenization_layoutlmv2 import (  # 从 tokenization_layoutlmv2 模块中导入 LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING, LayoutLMv2Tokenizer
    LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING,
    LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    LayoutLMv2Tokenizer,
)

logger = logging.get_logger(__name__)  # 获得 logger 对象

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}  # VOCAB_FILES_NAMES 字典

PRETRAINED_VOCAB_FILES_MAP = {  # PRETRAINED_VOCAB_FILES_MAP 字典
    "vocab_file": {  # "vocab_file" 字典
        "microsoft/layoutlmv2-base-uncased": (  # "microsoft/layoutlmv2-base-uncased" 键值对
            "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/vocab.txt"  # 对应的值
        ),
    },
    "tokenizer_file": {  # "tokenizer_file"字典
        "microsoft/layoutlmv2-base-uncased": (  # "microsoft/layoutlmv2-base-uncased"键值对
            "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/tokenizer.json"  # 对应的值
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {  # PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 字典
    "microsoft/layoutlmv2-base-uncased": 512,  # "microsoft/layoutlmv2-base-uncased" 键值对
}

PRETRAINED_INIT_CONFIGURATION = {  # PRETRAINED_INIT_CONFIGURATION 字典
    "microsoft/layoutlmv2-base-uncased": {"do_lower_case": True},  # "microsoft/layoutlmv2-base-uncased" 键值对
}


class LayoutLMv2TokenizerFast(PreTrainedTokenizerFast):  # 定义 LayoutLMv2TokenizerFast 类，继承自 PreTrainedTokenizerFast 类
    r"""
    使用 HuggingFace 的 *tokenizers* 库构建 "fast" LayoutLMv2 tokenizer。基于 WordPiece。

    该 tokenizer 继承自 [`PreTrainedTokenizerFast`]，其中包含大部分主要方法。用户应参考该超类以获取有关这些方法的更多信息。
    # 参数说明:
    # vocab_file (`str`): 词汇表文件名。
    # do_lower_case (`bool`, *optional*, defaults to `True`): 在标记化时是否将输入转换为小写。
    # unk_token (`str`, *optional*, defaults to `"[UNK]"`): 未知标记。词汇表中没有的标记无法转换为 ID，会被设置为此标记。
    # sep_token (`str`, *optional*, defaults to `"[SEP]"`): 分隔符标记，用于从多个序列构建序列时使用，例如用于序列分类或文本和问题的问题回答。还用作带有特殊标记的序列的最后一个标记。
    # pad_token (`str`, *optional*, defaults to `"[PAD]"`): 用于填充的标记，例如当批处理不同长度的序列时。
    # cls_token (`str`, *optional*, defaults to `"[CLS]"`): 分类器标记，在进行序列分类（整个序列而不是每个标记的分类）时使用。它是使用特殊标记构建序列时的第一个标记。
    # mask_token (`str`, *optional*, defaults to `"[MASK]"`): 用于掩码值的标记。这是在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
    # cls_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`): 用于特殊 [CLS] 标记的边界框。
    # sep_token_box (`List[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`): 用于特殊 [SEP] 标记的边界框。
    # pad_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`): 用于特殊 [PAD] 标记的边界框。
    # pad_token_label (`int`, *optional*, defaults to -100): 用于填充标记的标签。默认为 -100，这是 PyTorch 的 CrossEntropyLoss 的 `ignore_index`。
    # only_label_first_subword (`bool`, *optional*, defaults to `True`): 是否仅标记第一个子词（如果提供了单词标签）。
    # tokenize_chinese_chars (`bool`, *optional*, defaults to `True`): 是否标记化中文字符。对于日语，这可能应该取消激活（参见此问题）。
    # strip_accents (`bool`, *optional*): 是否去除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值确定（与原始 LayoutLMv2 相同）。
    """

    # 词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 初始化类的静态变量，使用预训练的位置嵌入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 指定慢速分词器的类为 LayoutLMv2Tokenizer

    slow_tokenizer_class = LayoutLMv2Tokenizer

    # 初始化方法
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 调用父类的初始化方法，传入相应的参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # 获取预训练分词器的状态信息
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查是否需要调整预训练分词器的参数
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            # 获取预训练分词器的类
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            # 更新参数
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            # 更新预训练分词器
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        # 设置实例变量的值
        self.do_lower_case = do_lower_case

        # 添加额外属性
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    # 添加文档结束字符串
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义类的调用方法，用于对输入文本或文本对进行编码
    def __call__(
        self,
        # 输入文本，可以是单个文本、预分词后的文本或其列表形式
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        # 输入文本对，可以是预分词后的文本对或其列表形式
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        # 文本框坐标，可以是列表形式，用于处理图像中的文本
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        # 单词标签，可以是单个标签列表或其列表形式，用于处理图像中的标签
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        # 是否添加特殊标记，如[CLS]和[SEP]
        add_special_tokens: bool = True,
        # 填充策略，可以是布尔值、字符串或填充策略对象
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，可以是布尔值、字符串或截断策略对象
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 步长
        stride: int = 0,
        # 填充至的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回标记类型 IDs
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的标记
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记掩码
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回长度
        return_length: bool = False,
        # 是否冗长输出
        verbose: bool = True,
        # 其他关键字参数
        **kwargs,
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 批量编码多个文本或文本对
    def batch_encode_plus(
        self,
        # 批量文本或文本对，可以是文本输入、文本对输入或预分词文本输入的列表形式
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        # 是否为文本对
        is_pair: bool = None,
        # 文本框坐标，可以是列表形式，用于处理图像中的文本
        boxes: Optional[List[List[List[int]]]] = None,
        # 单词标签，可以是单个标签列表或其列表形式，用于处理图像中的标签
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        # 是否添加特殊标记，如[CLS]和[SEP]
        add_special_tokens: bool = True,
        # 填充策略，可以是布尔值、字符串或填充策略对象
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，可以是布尔值、字符串或截断策略对象
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 步长
        stride: int = 0,
        # 填充至的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回标记类型 IDs
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的标记
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记掩码
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回长度
        return_length: bool = False,
        # 是否冗长输出
        verbose: bool = True,
        # 其他关键字参数
        **kwargs,
    ) -> BatchEncoding:
        # 为了向后兼容 'truncation_strategy', 'pad_to_max_length'，获取填充和截断策略以及其他相关参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用 _batch_encode_plus 方法进行批量编码
        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    # 对文本进行分词
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        # 如果有文本对，则将其组成批量输入
        batched_input = [(text, pair)] if pair else [text]
        # 使用 Tokenizer 的 encode_batch 方法对批量输入进行编码
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        # 返回编码结果的第一个样本的 tokens
        return encodings[0].tokens

    # 对文本进行编码，并提供额外参数的接口
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    def __call__(
        self,
        text: Union[
            str,
            List[str],
            List[List[str]]
        ],
        text_pair: Optional[
            Union[
                List[str],
                List[int]
            ]
        ] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略以及相关参数，用于后续处理
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法进行编码处理，返回编码结果
        return self._encode_plus(
            text=text,
            boxes=boxes,
            text_pair=text_pair,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        boxes: Optional[List[List[List[int]]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    # 定义一个内部函数，用于将文本数据转换为 BatchEncoding 格式
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 输入文本数据，可以是单文本或预分词后的文本
        text_pair: Optional[PreTokenizedInput] = None,  # 第二个文本数据，可为空
        boxes: Optional[List[List[int]]] = None,  # 文本框的坐标信息，可为空
        word_labels: Optional[List[int]] = None,  # 单词的标签信息，可为空
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 步幅大小
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的长度
        return_tensors: Optional[bool] = None,  # 返回张量或数组
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型 ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力蒙版
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token
        return_special_tokens_mask: bool = False,  # 是否返回特殊 token 蒙版
        return_offsets_mapping: bool = False,  # 返回偏移映射
        return_length: bool = False,  # 返回长度
        verbose: bool = True,  # 是否输出详细信息
        **kwargs,  # 其他关键字参数
    ) -> BatchEncoding:  # 返回 BatchEncoding 对象
        # 将输入转换为批处理输入
        batched_input = [(text, text_pair)] if text_pair else [text]  # 创建批处理输入
        batched_boxes = [boxes]  # 创建批处理文本框
        batched_word_labels = [word_labels] if word_labels is not None else None  # 创建批处理单词标签
        # 调用内部的批处理编码方法
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pair=bool(text_pair is not None),  # 是否有第二个文本
            boxes=batched_boxes,  # 文本框信息
            word_labels=batched_word_labels,  # 单词标签信息
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
            padding_strategy=padding_strategy,  # 填充策略
            truncation_strategy=truncation_strategy,  # 截断策略
            max_length=max_length,  # 最大长度限制
            stride=stride,  # 步幅大小
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到指定长度的倍数
            return_tensors=return_tensors,  # 返回张量或数组
            return_token_type_ids=return_token_type_ids,  # 返回 token 类型 ID
            return_attention_mask=return_attention_mask,  # 返回注意力蒙版
            return_overflowing_tokens=return_overflowing_tokens,  # 返回溢出的 token
            return_special_tokens_mask=return_special_tokens_mask,  # 返回特殊 token 蒙版
            return_offsets_mapping=return_offsets_mapping,  # 返回偏移映射
            return_length=return_length,  # 返回长度
            verbose=verbose,  # 输出详细信息
            **kwargs,  # 其他关键字参数
        )

        # 如果返回张量为空，并且不返回溢出的 token，则删除首个批处理轴
        # 如果溢出的 token 返回为批处理输出，则在此情况下保留它们
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,  # 编码信息
            )

        # 检查序列长度是否过长，并给出警告
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output  # 返回批处理输出
    # 定义一个内部方法用于填充输入序列，可以接受单个字典或批量编码的字典
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    # 从序列构建特殊标记的模型输入，用于序列分类任务
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # 从两个序列中创建用于序列对分类任务的 mask
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format: :: 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second
        sequence | If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 保存词汇表到指定文件夹
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```