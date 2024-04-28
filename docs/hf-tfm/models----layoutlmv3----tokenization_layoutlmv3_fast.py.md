# `.\models\layoutlmv3\tokenization_layoutlmv3_fast.py`

```
# 设置编码方式为 UTF-8
# 版权声明
# 版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可 2.0 版本授权
# 除非符合许可协议要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按 "原样" 分发软件
# 无论明示或暗示，均无任何保证或条件
# 有关限制性语言，请参阅许可协议
"""
LayoutLMv3 的快速标记化类。它重写了慢标记化类的两个方法，即 _batch_encode_plus 和 _encode_plus，
其中使用了 Rust 标记器。
"""

# 导入所需模块和类型提示
import json
from typing import Dict, List, Optional, Tuple, Union

# 导入 tokenizers 库的预处理器和处理器
from tokenizers import pre_tokenizers, processors

# 导入基类中的相关类型和工具函数
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

# 导入快速标记器基类
from ...tokenization_utils_fast import PreTrainedTokenizerFast

# 导入日志记录工具
from ...utils import add_end_docstrings, logging

# 导入 LayoutLMv3 标记化器
from .tokenization_layoutlmv3 import (
    LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING,
    LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    LayoutLMv3Tokenizer,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/raw/main/vocab.json",
        "microsoft/layoutlmv3-large": "https://huggingface.co/microsoft/layoutlmv3-large/raw/main/vocab.json",
    },
    "merges_file": {
        "microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/raw/main/merges.txt",
        "microsoft/layoutlmv3-large": "https://huggingface.co/microsoft/layoutlmv3-large/raw/main/merges.txt",
    },
}

# 定义预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlmv3-base": 512,
    "microsoft/layoutlmv3-large": 512,
}


class LayoutLMv3TokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个 "快速" LayoutLMv3 标记化器（由 HuggingFace 的 *tokenizers* 库支持）。基于 BPE。

    此标记化器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该参考此超类以获取有关这些方法的更多信息。

    """

    # 定义词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 定义慢标记化器类
    slow_tokenizer_class = LayoutLMv3Tokenizer

```  
    # 初始化函数，用于创建一个新的Tokenizer对象
    def __init__(
        self,
        # 词汇文件路径，用于加载词汇表
        vocab_file=None,
        # 合并文件路径，用于加载合并表
        merges_file=None,
        # 分词器文件路径，用于加载分词器
        tokenizer_file=None,
        # 错误处理方式，默认为替换
        errors="replace",
        # 句首标记
        bos_token="<s>",
        # 句尾标记
        eos_token="</s>",
        # 分隔符标记
        sep_token="</s>",
        # 类别开始标记
        cls_token="<s>",
        # 未知标记
        unk_token="<unk>",
        # 填充标记
        pad_token="<pad>",
        # 掩码标记
        mask_token="<mask>",
        # 是否在前缀之前添加空格，默认为True
        add_prefix_space=True,
        # 是否修剪偏移量，默认为True
        trim_offsets=True,
        # 类别开始标记的边框坐标，默认为[0, 0, 0, 0]
        cls_token_box=[0, 0, 0, 0],
        # 分隔符标记的边框坐标，默认为[0, 0, 0, 0]
        sep_token_box=[0, 0, 0, 0],
        # 填充标记的边框坐标，默认为[0, 0, 0, 0]
        pad_token_box=[0, 0, 0, 0],
        # 填充标记的类别，默认为-100
        pad_token_label=-100,
        # 是否仅对标签的第一个子词进行标记，默认为True
        only_label_first_subword=True,
        # 其他参数，可传递给父类
        **kwargs,
    # 调用父类的构造函数来初始化 LayoutLMv3TokenizerFast 实例
    ):
        super().__init__(
            vocab_file,  # 词汇表文件路径
            merges_file,  # 合并文件路径
            tokenizer_file=tokenizer_file,  # 分词器文件路径
            errors=errors,  # 错误处理方式
            bos_token=bos_token,  # 开始标记
            eos_token=eos_token,  # 结束标记
            sep_token=sep_token,  # 分隔标记
            cls_token=cls_token,  # 分类标记
            unk_token=unk_token,  # 未知标记
            pad_token=pad_token,  # 填充标记
            mask_token=mask_token,  # 掩码标记
            add_prefix_space=add_prefix_space,  # 是否在每个子词前添加空格
            trim_offsets=trim_offsets,  # 是否修剪偏移量
            cls_token_box=cls_token_box,  # 分类标记框
            sep_token_box=sep_token_box,  # 分隔标记框
            pad_token_box=pad_token_box,  # 填充标记框
            pad_token_label=pad_token_label,  # 填充标记标签
            only_label_first_subword=only_label_first_subword,  # 是否仅标记第一个子词
            **kwargs,  # 其它参数
        )

        # 将前处理器的状态转换为 JSON 格式
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果 add_prefix_space 属性与给定值不同，更新为给定值
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置实例的 add_prefix_space 属性
        self.add_prefix_space = add_prefix_space

        # 获取后处理器实例
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # 将列表 'sep' 和 'cls' 转换为元组，以便于 `post_processor_class`
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            # 检查是否需要应用更改
            changes_to_apply = False

            # 如果 add_prefix_space 属性与给定值不同，更新为给定值
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            # 如果 trim_offsets 属性与给定值不同，更新为给定值
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果需要应用更改，则创建新的后处理器实例并设置
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

        # 设置额外的属性
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    # 添加额外的文档字符串
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.__call__ 复制
    # 定义类方法__call__，用于对输入文本进行编码
    def __call__(
        self,
        # 输入文本，可以是单个文本、预处理后的文本或文本列表
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        # 第二个输入文本（可选），用于文本对任务
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        # 文本框的坐标信息，用于布局任务
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        # 单词标签，用于标记单词边界
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        # 是否添加特殊标记（如[CLS]、[SEP]）
        add_special_tokens: bool = True,
        # 是否进行填充
        padding: Union[bool, str, PaddingStrategy] = False,
        # 是否进行截断
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 步长
        stride: int = 0,
        # 填充到指定的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回token类型id
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的tokens
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记的掩码
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回文本长度
        return_length: bool = False,
        # 是否启用详细输出
        verbose: bool = True,
        # 其他参数
        **kwargs,
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.batch_encode_plus复制过来的
    # 定义batch_encode_plus方法，用于对文本批量进行编码
    def batch_encode_plus(
        self,
        # 批量输入文本，可以是单个文本、文本对或预处理后的文本
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        # 是否为文本对
        is_pair: bool = None,
        # 文本框的坐标信息，用于布局任务
        boxes: Optional[List[List[List[int]]]] = None,
        # 单词标签，用于标记单词边界
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        # 是否添加特殊标记（如[CLS]、[SEP]）
        add_special_tokens: bool = True,
        # 是否进行填充
        padding: Union[bool, str, PaddingStrategy] = False,
        # 是否进行截断
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 步长
        stride: int = 0,
        # 填充到指定的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回token类型id
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的tokens
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记的掩码
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回文本长度
        return_length: bool = False,
        # 是否启用详细输出
        verbose: bool = True,
        # 其他参数
        **kwargs,
    # 返回一个批量编码的结果
    ) -> BatchEncoding:
        # 获取填充和截断策略以及其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法进行批量编码处理
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

    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.tokenize 复制而来
    # 对文本进行标记化处理
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        # 将文本封装成批量输入
        batched_input = [(text, pair)] if pair else [text]
        # 使用内部的编码批量处理方法对输入进行编码
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        # 返回编码结果的 tokens 部分
        return encodings[0].tokens

    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.encode_plus 复制而来
    # 对输入进行编码处理
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
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
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings, or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略的方法，以及最大长度和其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 使用_encode_plus方法对输入进行编码处理
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
    # 从transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast._encode_plus方法中复制而来，用于对文本进行编码并返回各种信息
    def _encode_plus(
        # 待编码的文本，可以是单个文本或者是预分词后的文本
        text: Union[TextInput, PreTokenizedInput],
        # 第二个文本序列（可选），通常用于处理文本对任务
        text_pair: Optional[PreTokenizedInput] = None,
        # 文本对应的包围框信息（可选），通常用于布局感知的模型
        boxes: Optional[List[List[int]]] = None,
        # 单词级别的标签（可选），通常用于处理序列标注任务
        word_labels: Optional[List[int]] = None,
        # 是否添加特殊标记（可选），默认为True
        add_special_tokens: bool = True,
        # 填充策略（可选），默认为不进行填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略（可选），默认为不进行截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度（可选），用于控制编码后的文本长度
        max_length: Optional[int] = None,
        # 步长（可选），用于滑动窗口截断文本
        stride: int = 0,
        # 填充到的倍数（可选），用于填充后的文本长度控制
        pad_to_multiple_of: Optional[int] = None,
        # 是否返回张量（可选），默认为None
        return_tensors: Optional[bool] = None,
        # 是否返回标记类型ID（可选），默认为None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码（可选），默认为None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的标记（可选），默认为False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记的掩码（可选），默认为False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射（可选），默认为False
        return_offsets_mapping: bool = False,
        # 是否返回编码后文本的长度（可选），默认为False
        return_length: bool = False,
        # 是否启用详细输出（可选），默认为True
        verbose: bool = True,
        # 其他参数（可选）
        **kwargs,
    # 定义一个函数，输入为文本和可选的文本对，输出为批处理后的编码
    def __call__(
        self,
        text: Union[List[str], List[List[str]]],
        text_pair: Optional[Union[List[str], List[List[str]]]] = None,
        boxes: Optional[List[List[Tuple[int]]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy, str] = False,
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
        verbose: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        # 将输入文本批处理化
        # 两种选项：
        # 1) 只有文本，此时文本必须是一个字符串列表
        # 2) 文本 + 文本对，此时文本是一个字符串，文本对是一个字符串列表
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_boxes = [boxes]
        batched_word_labels = [word_labels] if word_labels is not None else None
        # 调用_batch_encode_plus方法进行编码
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pair=bool(text_pair is not None),
            boxes=batched_boxes,
            word_labels=batched_word_labels,
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

        # 如果返回的张量为空，则可以删除前导批处理轴
        # 在这种情况下，超出的标记作为输出的批处理返回，因此在这种情况下保留它们
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        # 检查序列长度是否过长，并在需要时发出警告
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        # 返回批处理后的输出
        return batched_output

    # 从transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast中复制的_pad方法
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        
    # 从transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast中复制的save_vocabulary方法
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用_tokenizer.model.save方法保存词汇表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名
        return tuple(files)
    # 构建包含特殊令牌的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 在输入序列的开头添加开始令牌
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # 如果有第二个输入序列
        if token_ids_1 is None:
            # 返回输出序列
            return output
        # 添加分隔符和第二个输入序列的令牌，并在末尾添加结束令牌
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    # 从输入序列创建令牌类型ID
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Args:
        从两个序列中创建一个用于序列对分类任务的掩码。RoBERTa 没有使用令牌类型ID，因此返回了一个全为零的列表。
            token_ids_0 (`List[int]`):
                ID的列表.
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的ID列表，用于序列对.
        Returns:
            `List[int]`: 全为零的列表.
        """
        # 分隔符和CLS令牌的ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果没有第二个序列，则返回全为零的列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 创建包含两个序列的令牌类型ID，并在最后添加分隔符的ID
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```