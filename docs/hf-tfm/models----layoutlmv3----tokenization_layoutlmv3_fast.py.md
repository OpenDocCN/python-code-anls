# `.\models\layoutlmv3\tokenization_layoutlmv3_fast.py`

```py
# 导入必要的模块和类
import json  # 导入用于处理 JSON 数据的模块
from typing import Dict, List, Optional, Tuple, Union  # 引入类型提示，用于函数参数和返回值类型检查

from tokenizers import pre_tokenizers, processors  # 导入tokenizers库中的预处理器和处理器

# 导入基础的 tokenization_utils_base 模块中定义的类和函数
from ...tokenization_utils_base import (
    BatchEncoding,  # 批量编码结果的数据结构
    EncodedInput,  # 编码后的输入数据
    PaddingStrategy,  # 填充策略枚举类型
    PreTokenizedInput,  # 预分词化的输入数据结构
    TensorType,  # 张量类型标识
    TextInput,  # 文本输入数据类型
    TextInputPair,  # 文本对输入数据类型
    TruncationStrategy,  # 截断策略枚举类型
)

# 导入快速 tokenization_utils_fast 模块中定义的类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 快速预训练分词器基类

# 导入工具函数和日志记录
from ...utils import add_end_docstrings, logging  # 导入添加文档结束字符串的装饰器和日志记录功能

# 导入 LayoutLMv3Tokenizer 类
from .tokenization_layoutlmv3 import (
    LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING,  # 编码关键字参数文档字符串
    LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,  # 编码加强版关键字参数文档字符串
    LayoutLMv3Tokenizer,  # LayoutLMv3Tokenizer 类
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
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

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlmv3-base": 512,
    "microsoft/layoutlmv3-large": 512,
}


class LayoutLMv3TokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" LayoutLMv3 tokenizer (backed by HuggingFace's *tokenizers* library). Based on BPE.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """

    # 词汇文件名称映射
    vocab_files_names = VOCAB_FILES_NAMES

    # 预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    # 预训练模型的最大输入尺寸映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 模型的输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 慢速分词器的类，用于提供后备
    slow_tokenizer_class = LayoutLMv3Tokenizer
    # 定义一个初始化方法，用于初始化对象的各种属性和参数
    def __init__(
        self,
        vocab_file=None,                # 词汇表文件路径，默认为None
        merges_file=None,               # 合并文件路径，默认为None
        tokenizer_file=None,            # 分词器文件路径，默认为None
        errors="replace",               # 编码错误处理方式，默认为替换
        bos_token="<s>",                # 开始符号标记，默认为"<s>"
        eos_token="</s>",               # 结束符号标记，默认为"</s>"
        sep_token="</s>",               # 分隔符号标记，默认为"</s>"
        cls_token="<s>",                # 类别标记，默认为"<s>"
        unk_token="<unk>",              # 未知标记，默认为"<unk>"
        pad_token="<pad>",              # 填充标记，默认为"<pad>"
        mask_token="<mask>",            # 掩码标记，默认为"<mask>"
        add_prefix_space=True,          # 是否在标记前添加空格，默认为True
        trim_offsets=True,              # 是否修剪偏移量，默认为True
        cls_token_box=[0, 0, 0, 0],     # 类别标记框，默认为[0, 0, 0, 0]
        sep_token_box=[0, 0, 0, 0],     # 分隔符号标记框，默认为[0, 0, 0, 0]
        pad_token_box=[0, 0, 0, 0],     # 填充标记框，默认为[0, 0, 0, 0]
        pad_token_label=-100,           # 填充标记的标签，默认为-100
        only_label_first_subword=True,  # 是否只标记第一个子词，默认为True
        **kwargs,                       # 其他未命名参数，以字典形式接收
        )
        # 调用父类的构造函数，初始化 LayoutLMv3TokenizerFast 实例
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            **kwargs,
        )

        # 获取当前的前处理器（pre_tokenizer）状态并转换为 JSON 格式
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())

        # 如果当前前处理器的 add_prefix_space 属性与参数 add_prefix_space 不一致
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 获取当前前处理器的类型
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            # 更新前处理器状态的 add_prefix_space 属性为参数值
            pre_tok_state["add_prefix_space"] = add_prefix_space
            # 根据更新后的状态重新创建前处理器对象
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置对象的 add_prefix_space 属性
        self.add_prefix_space = add_prefix_space

        # 获取后处理器（post_processor）组件的实例
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)

        # 如果后处理器组件实例存在
        if tokenizer_component_instance:
            # 获取当前后处理器实例的状态并转换为 JSON 格式
            state = json.loads(tokenizer_component_instance.__getstate__())

            # 如果状态中包含 'sep' 列表，则将其转换为元组
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            # 如果状态中包含 'cls' 列表，则将其转换为元组
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            # 初始化变量，用于记录是否有更新需要应用到后处理器实例
            changes_to_apply = False

            # 如果后处理器状态中的 add_prefix_space 属性与参数 add_prefix_space 不一致
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                # 更新状态中的 add_prefix_space 属性为参数值
                state["add_prefix_space"] = add_prefix_space
                # 标记需要应用变更
                changes_to_apply = True

            # 如果后处理器状态中的 trim_offsets 属性与参数 trim_offsets 不一致
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                # 更新状态中的 trim_offsets 属性为参数值
                state["trim_offsets"] = trim_offsets
                # 标记需要应用变更
                changes_to_apply = True

            # 如果有需要应用的变更
            if changes_to_apply:
                # 获取后处理器的类
                component_class = getattr(processors, state.pop("type"))
                # 创建新的后处理器实例
                new_value = component_class(**state)
                # 将新的后处理器实例赋给后处理器组件
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

        # 设置额外的属性值
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword
    # 通过 add_end_docstrings 装饰器添加文档字符串，参考 LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING 和 LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.__call__ 复制而来
    # 定义一个特殊方法 __call__，使实例对象可以像函数一样被调用
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
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
    ):
        # 使用 layoutlmv3 的特定参数文档来扩展函数的文档字符串
        @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
        # 方法批量编码多个文本或文本对，并返回处理后的结果
        # 这是从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.batch_encode_plus 复制过来的
        def batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
            ],
            is_pair: bool = None,
            boxes: Optional[List[List[List[int]]]] = None,
            word_labels: Optional[Union[List[int], List[List[int]]]] = None,
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
        ):
    ) -> BatchEncoding:
        # 获取填充和截断策略，以及相关参数，用于向后兼容 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用 _batch_encode_plus 方法进行批量编码处理
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
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        # 如果有文本对，则构建批次输入
        batched_input = [(text, pair)] if pair else [text]
        # 使用 _tokenizer 对象进行批量编码处理，返回编码结果
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        # 返回第一个编码结果的 tokens
        return encodings[0].tokens

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.encode_plus 复制而来
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
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，以及其他参数，用于后续编码过程
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用 _encode_plus 方法进行编码处理，返回编码结果
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
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast._encode_plus 复制而来的方法定义
    def _encode_plus(
        self,
        # 输入的文本，可以是单个文本或者预分词后的输入
        text: Union[TextInput, PreTokenizedInput],
        # 可选参数，第二个输入文本对
        text_pair: Optional[PreTokenizedInput] = None,
        # 可选参数，文本框的坐标列表
        boxes: Optional[List[List[int]]] = None,
        # 可选参数，单词标签列表
        word_labels: Optional[List[int]] = None,
        # 是否添加特殊标记（如[CLS]和[SEP]）
        add_special_tokens: bool = True,
        # 填充策略，默认不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 滑动窗口步长
        stride: int = 0,
        # 填充到的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 是否返回张量
        return_tensors: Optional[bool] = None,
        # 是否返回token类型ID
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的tokens
        return_overflowing_tokens: bool = False,
        # 是否返回特殊tokens的掩码
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回长度
        return_length: bool = False,
        # 是否显示详细信息
        verbose: bool = True,
        # 其它关键字参数
        **kwargs,
    ) -> BatchEncoding:
        # 将输入转换为批量输入
        # 两种选项：
        # 1) 只有文本，此时文本必须是一个字符串列表
        # 2) 文本 + 文本对，此时文本是一个字符串，text_pair 是一个字符串列表
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_boxes = [boxes]
        batched_word_labels = [word_labels] if word_labels is not None else None
        # 调用 _batch_encode_plus 方法进行批量编码
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

        # 如果 return_tensors 是 None，并且不返回溢出的 token，则去除前导批次轴
        # 溢出的 token 作为输出的批次返回，因此在这种情况下保留它们
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        # 对批处理的输入进行长度检查，如果长度超出设定的最大长度则警告
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output

    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast._pad 复制而来
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.save_vocabulary 复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 使用 _tokenizer.model.save 方法保存词汇表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名元组
        return tuple(files)
    # 创建一个包含特殊标记的输入序列，用于模型输入
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 构建输出列表，加入起始标记、token_ids_0和结束标记
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # 如果没有第二个序列token_ids_1，则直接返回output
        if token_ids_1 is None:
            return output

        # 否则在output后加入结束标记、token_ids_1和再次结束标记
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    # 根据输入的两个序列token_ids_0和token_ids_1，创建用于序列对分类任务的token类型ID列表
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Args:
            token_ids_0 (`List[int]`):
                第一个序列的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的ID列表，用于序列对任务。

        Returns:
            `List[int]`: 全零列表，用作RoBERTa模型中不使用token类型ID的占位。
        """
        # 分隔标记
        sep = [self.sep_token_id]
        # 分类标记
        cls = [self.cls_token_id]

        # 如果没有token_ids_1，则返回长度为cls + token_ids_0 + sep的全零列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        # 否则返回长度为cls + token_ids_0 + sep + sep + token_ids_1 + sep的全零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```