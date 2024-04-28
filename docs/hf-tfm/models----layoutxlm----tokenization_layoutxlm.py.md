# `.\models\layoutxlm\tokenization_layoutxlm.py`

```
# 设定文件编码格式为 utf-8
# 版权声明，遵循 Apache License 2.0
# 代码逻辑从 LayoutXLM 模型 tokenizer 继承而来
# 导入所需的库和模块
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
# 导入 HuggingFace 库中的 Tokenizer 相关类和方法
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging
# 导入 xlm_roberta 模型的 tokenizer 相关类和方法
from ..xlm_roberta.tokenization_xlm_roberta import (
    # 导入预训练位置嵌入大小字典
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
    # 导入预训练词汇表文件字典
    PRETRAINED_VOCAB_FILES_MAP,
    # 使用下划线
    SPIECE_UNDERLINE,
    # 词汇表文件名
    VOCAB_FILES_NAMES,
)
# 日志记录器
logger = logging.get_logger(__name__)

class LayoutXLMTokenizer(PreTrainedTokenizer):
    """
    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    # 词汇表文件名字典
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
        # 定义 mask_token 方法，并指定返回类型为 None
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        # 如果 sp_model_kwargs 为 None，则初始化为空字典，否则使用传入的 sp_model_kwargs
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 使用 sp_model_kwargs 初始化 sp_model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件到 sp_model
        self.sp_model.Load(str(vocab_file))
        # 保存词汇文件路径到 vocab_file
        self.vocab_file = vocab_file

        # Mimic fairseq token-to-id alignment for the first 4 token
        # 初始化 fairseq_tokens_to_ids 字典，包括"<s>", "<pad>", "</s>", "<unk>"
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        # fairseq_vocab 和 spm_vocab 的对齐关系
        # ...
        
        # 设置 fairseq_offset 为 1
        self.fairseq_offset = 1
        # 添加 "<mask>" 到 fairseq_tokens_to_ids 字典中，值为 len(self.sp_model) + self.fairseq_offset
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        # 初始化 fairseq_ids_to_tokens 字典，反转 fairseq_tokens_to_ids 字典
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # 将其他属性保存如 cls_token_box, sep_token_box 等
        self.cls_token_box = cls_token_box
        # ...

        # 调用父类的 __init__ 方法，初始化相关参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    # 序列化对象时调用的方法
    def __getstate__(self):
        state = self.__dict__.copy()
        # 将 sp_model 设置为 None
        state["sp_model"] = None
        # 保存 sp_model 的 serialized_model_proto 到 sp_model_proto
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    # 反序列化对象时调用的方法
    def __setstate__(self, d):
        self.__dict__ = d
        # 兼容旧版本，如果不存在 sp_model_kwargs，则初始化为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
        # 重新初始化 sp_model，并加载 sp_model_proto
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    # 构建包含特殊 tokens 的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从序列或序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。XLM-RoBERTa 序列具有以下格式：

        - 单个序列：`
    ) -> List[int]:
        """
        声明一个方法，用于在序列对分类任务中创建一个 mask
        """
        # 初始化分隔符的 token id
        sep = [self.sep_token_id]
        # 初始化类别标识符的 token id
        cls = [self.cls_token_id]

        # 如果第二个序列的 token ids 为空
        if token_ids_1 is None:
            # 返回由 0 组成的列表，长度为类别标识符 + 第一个序列的长度 + 分隔符的长度
            return len(cls + token_ids_0 + sep) * [0]
        # 否则
        # 返回由 0 组成的列表，长度为类别标识符 + 第一个序列的长度 + 分隔符的长度 + 另一个序列的长度 + 分隔符的长度
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    # 定义属性方法，返回语料库的大小
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    # 获取语料库的方法
    def get_vocab(self):
        # 创建包含转换后的 tokens 的字典
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 更新 tokens 编码器的内容
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 分词方法，将文本转换为 tokens 列表
    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    # 将 token 转换为 id 的方法
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 如果 token 在 fairseq_tokens_to_ids 中
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        # 否则
        spm_id = self.sp_model.PieceToId(token)

        # 如果 SP 模型返回的 id 为 0，则返回未知 token id
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 如果 index 在 fairseq_ids_to_tokens 中
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        # 否则
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    # 将 tokens 转换为单个字符串的方法
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # 保存语料库的方法
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存路径不是��个目录
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 定义输出的语料库文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前的语料库文件路径不等于输出的语料库文件路径，并且当前的语料库文件存在
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            # 复制当前的语料库文件到输出路径
            copyfile(self.vocab_file, out_vocab_file)
        # 否则，如果当前的语料库文件不存在
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    # 添加文档字符串的装饰器方法
    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)
    # 定义一个方法，接受多种类型的文本输入，并可选地接受文本对、框、单词标签以及各种其他参数
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
        
        # 定义一个方法，批量编码文本或文本对，并可选地接受框、单词标签以及各种其他参数
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
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    # 定义一个方法，用于准备输入批量数据，以便模型使用
    def __call__(
        self,
        batch_text_or_text_pairs: Union[List[str], List[Tuple[str, Optional[str]]]],
        is_pair: bool = None,
        boxes: Optional[List[List[int]]] = None,
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
        return_length: bool = False,
        verbose: bool = True,
    ):
        # 如果要返回偏移映射，则抛出未实现错误，因为 Python tokenizers 不支持此功能
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 准备模型输入的批量数据
        batch_outputs = self._batch_prepare_for_model(
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
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        # 返回批量编码的结果
        return BatchEncoding(batch_outputs)

    # 添加LAYOUTXLM_ENCODE_KWARGS_DOCSTRING中描述的文档字符串到方法中
    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_text_or_text_pairs,
        is_pair: bool = None,
        boxes: Optional[List[List[int]]] = None,
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
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        # 初始化一个空字典用于存储批处理的输出
        batch_outputs = {}
        # 遍历批处理中的每个示例
        for idx, example in enumerate(zip(batch_text_or_text_pairs, boxes)):
            # 从输入的批处理中提取文本或文本对和对应的边界框
            batch_text_or_text_pair, boxes_example = example
            # 准备输入序列以供模型使用
            outputs = self.prepare_for_model(
                # 如果是文本对，则提取第一个文本；否则提取文本
                batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair,
                # 如果是文本对，则提取第二个文本；否则为 None
                batch_text_or_text_pair[1] if is_pair else None,
                # 当前示例的边界框
                boxes_example,
                # 如果提供了单词标签，则传入当前示例的标签；否则为 None
                word_labels=word_labels[idx] if word_labels is not None else None,
                # 是否添加特殊标记
                add_special_tokens=add_special_tokens,
                # 填充策略：不填充，在批处理后进行填充
                padding=PaddingStrategy.DO_NOT_PAD.value,
                # 截断策略
                truncation=truncation_strategy.value,
                # 最大长度
                max_length=max_length,
                # 步长
                stride=stride,
                # 填充到倍数的策略：在批处理后进行填充
                pad_to_multiple_of=None,
                # 是否返回注意力掩码：在批处理后返回
                return_attention_mask=False,
                # 是否返回 token 类型 id：在批处理后返回
                return_token_type_ids=return_token_type_ids,
                # 是否返回溢出的 token：在批处理后返回
                return_overflowing_tokens=return_overflowing_tokens,
                # 是否返回特殊标记掩码：在批处理后返回
                return_special_tokens_mask=return_special_tokens_mask,
                # 是否返回长度：在批处理后返回
                return_length=return_length,
                # 是否返回张量：在最后将整个批次转换为张量
                return_tensors=None,
                # 是否在输出中添加批次轴
                prepend_batch_axis=False,
                # 是否冗长的输出
                verbose=verbose,
            )

            # 将输出添加到批输出字典中
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # 对批输出进行填充
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 将批输出转换为 BatchEncoding 类型
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # 返回批输出
        return batch_outputs
    # 对输入进行编码，并返回编码后的批量编码
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 文本输入，可以是文本或预先标记的输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的第二个文本输入，可以是预先标记的输入
        boxes: Optional[List[List[int]]] = None,  # 文本框的坐标列表
        word_labels: Optional[List[int]] = None,  # 文字标签列表
        add_special_tokens: bool = True,  # 是否添加特殊令牌
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回令牌类型的ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出令牌
        return_special_tokens_mask: bool = False,  # 是否返回特殊令牌的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否冗长输出
        **kwargs,  # 其他参数
    ) -> BatchEncoding:  # 返回类型为 BatchEncoding
        # 如果要返回偏移映射，抛出未实现的错误
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        return self.prepare_for_model(  # 返回经过模型准备的编码
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)  # 添加文档字符串
    # 为模型准备输入数据，根据给定的参数对输入进行处理
    def prepare_for_model(
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
        prepend_batch_axis: bool = False,
        **kwargs,
    # 截断输入序列，根据给定的策略截断输入序列以便模型处理
    def truncate_sequences(
        self,
        ids: List[int],
        token_boxes: List[List[int]],
        pair_ids: Optional[List[int]] = None,
        pair_token_boxes: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    # 对输入进行填充，根据给定的策略对输入进行填充以满足模型输入要求
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
````
```