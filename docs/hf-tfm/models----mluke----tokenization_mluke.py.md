# `.\transformers\models\mluke\tokenization_mluke.py`

```
# 设置文件编码为utf-8
# 版权声明
# 导入必要的库和模块
from collections.abc import Mapping
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
# 导入numpy和sentencepiece库
import numpy as np
import sentencepiece as spm
# 导入HuggingFace提供的工具类
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
    to_py_obj,
)
from ...utils import add_end_docstrings, is_tf_tensor, is_torch_tensor, logging
# 设置日志记录器
logger = logging.get_logger(__name__)
# 定义类型别名
EntitySpan = Tuple[int, int]
EntitySpanInput = List[EntitySpan]
Entity = str
EntityInput = List[Entity]

# 设置SentencePiece用于连接符
SPIECE_UNDERLINE = "▁"
# 定义词汇文件名的映射
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "entity_vocab_file": "entity_vocab.json"}
# 预训练模型中的词汇文件名映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "studio-ousia/mluke-base": "https://huggingface.co/studio-ousia/mluke-base/resolve/main/vocab.json",
    },
    "merges_file": {
        "studio-ousia/mluke-base": "https://huggingface.co/studio-ousia/mluke-base/resolve/main/merges.txt",
    },
    "entity_vocab_file": {
        "studio-ousia/mluke-base": "https://huggingface.co/studio-ousia/mluke-base/resolve/main/entity_vocab.json",
    },
}
# 预训练位置嵌入尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "studio-ousia/mluke-base": 512,
}

class MLukeTokenizer(PreTrainedTokenizer):
    """
    Adapted from [`XLMRobertaTokenizer`] and [`LukeTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    # 设置属性值
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化函数，接收多个参数，包括词汇文件、实体词汇文件、特殊符号等
    def __init__(
        self,
        vocab_file,
        entity_vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        task=None,
        max_entity_length=32,
        max_mention_length=30,
        entity_token_1="<ent>",
        entity_token_2="<ent2>",
        entity_unk_token="[UNK]",
        entity_pad_token="[PAD]",
        entity_mask_token="[MASK]",
        entity_mask2_token="[MASK2]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    @property
    # 获取词汇表的大小，包括sp_model的长度、fairseq_offset和一个<mask> token
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    # 获取词汇表，包括已添加的标记编码
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 对文本进行分词处理
    def _tokenize(self, text: str) -> List[str]:
        # TODO check if the t5/llama PR also applies here
        return self.sp_model.encode(text, out_type=str)

    # 将token转换为id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    # 将id转换为token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    # 将token序列转换为字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # 获取当前状态的拷贝
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state
    
    # 设置当前状态
    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 将ENCODE_KWARGS_DOCSTRING和ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING添加到函数的文档字符串（docstring）中
    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        text_pair: Optional[Union[TextInput, List[TextInput]]] = None,
        entity_spans: Optional[Union[EntitySpanInput, List[EntitySpanInput]]] = None,
        entity_spans_pair: Optional[Union[EntitySpanInput, List[EntitySpanInput]]] = None,
        entities: Optional[Union[EntityInput, List[EntityInput]]] = None,
        entities_pair: Optional[Union[EntityInput, List[EntityInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: Optional[bool] = False,
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
    # 调用该函数将对输入的文本进行编码，并返回编码结果
    # text: 输入的文本或文本列表
    # text_pair: 第二个文本或文本列表（可选）
    # entity_spans: 实体的位置信息或位置信息列表（可选）
    # entity_spans_pair: 第二个实体的位置信息或位置信息列表（可选）
    # entities: 实体的编码信息或编码信息列表（可选）
    # entities_pair: 第二个实体的编码信息或编码信息列表（可选）
    # add_special_tokens: 是否在编码结果中加入特殊标记（默认为True）
    # padding: 填充策略，可以是布尔值、字符串或PaddingStrategy（默认为False）
    # truncation: 截断策略，可以是布尔值、字符串或TruncationStrategy（默认为None）
    # max_length: 编码后的文本的最大长度（可选）
    # max_entity_length: 编码后的实体的最大长度（可选）
    # stride: 编码时两个实体间的跨度（默认为0）
    # is_split_into_words: 输入的文本是否已经被分成单词（可选，默认为False）
    # pad_to_multiple_of: 对序列进行填充时的长度倍数（可选）
    # return_tensors: 返回的张量类型（可选）
    # return_token_type_ids: 是否返回token_type_ids（可选）
    # return_attention_mask: 是否返回attention_mask（可选）
    # return_overflowing_tokens: 是否返回溢出的token（默认为False）
    # return_special_tokens_mask: 是否返回特殊标记的掩码（默认为False）
    # return_offsets_mapping: 是否返回字符偏移映射（默认为False）
    # return_length: 是否返回文本长度（默认为False）
    # verbose: 是否打印过程信息（默认为True）
    # **kwargs: 其它关键字参数
    # 下面两行的函数和注释与上面相同，只是函数名和参数名不同，这里没有提供
    def _encode_plus(
        self,
        text: Union[TextInput],
        text_pair: Optional[Union[TextInput]] = None,
        entity_spans: Optional[EntitySpanInput] = None,
        entity_spans_pair: Optional[EntitySpanInput] = None,
        entities: Optional[EntityInput] = None,
        entities_pair: Optional[EntityInput] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: Optional[bool] = False,
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
        # 如果需要返回偏移映射，则抛出异常
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 如果is_split_into_words为真，则抛出异常，此tokenizer不支持该特性
        if is_split_into_words:
            raise NotImplementedError("is_split_into_words is not supported in this tokenizer.")

        # 调用_create_input_sequence函数创建输入序列
        (
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        ) = self._create_input_sequence(
            text=text,
            text_pair=text_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=entity_spans,
            entity_spans_pair=entity_spans_pair,
            **kwargs,
        )

        # 准备传给模型的输入数据，包括创建attenton_mask和token_type_ids
        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            entity_ids=first_entity_ids,
            pair_entity_ids=second_entity_ids,
            entity_token_spans=first_entity_token_spans,
            pair_entity_token_spans=second_entity_token_spans,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            max_entity_length=max_entity_length,
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

    # 从transformers.models.luke.tokenization_luke.LukeTokenizer._batch_encode_plus复制的代码
    # 定义一个方法用于批量编码输入文本或文本对，支持不同的参数设置
    def _batch_encode_plus(
        # 批量输入文本或文本对的列表
        batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair]],
        # 批量实体标记或实体标记对的列表，可选参数，默认为None
        batch_entity_spans_or_entity_spans_pairs: Optional[
            Union[List[EntitySpanInput], List[Tuple[EntitySpanInput, EntitySpanInput]]]
        ] = None,
        # 批量实体或实体对的列表，可选参数，默认为None
        batch_entities_or_entities_pairs: Optional[
            Union[List[EntityInput], List[Tuple[EntityInput, EntityInput]]]
        ] = None,
        # 是否添加特殊标记，默认为True
        add_special_tokens: bool = True,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认为不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制，默认为None
        max_length: Optional[int] = None,
        # 最大实体长度限制，默认为None
        max_entity_length: Optional[int] = None,
        # 步幅，默认为0
        stride: int = 0,
        # 是否拆分为单词，默认为False
        is_split_into_words: Optional[bool] = False,
        # 填充到的倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 返回张量的类型，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 返回标记类型ID，默认为None
        return_token_type_ids: Optional[bool] = None,
        # 返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出标记，默认为False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记的掩码，默认为False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为False
        return_length: bool = False,
        # 是否详细输出，默认为True
        verbose: bool = True,
        # 其他关键字参数
        **kwargs,
    # 从transformers.models.luke.tokenization_luke.LukeTokenizer._check_entity_input_format复制而来
    def _check_entity_input_format(self, entities: Optional[EntityInput], entity_spans: Optional[EntitySpanInput]):
        # 如果实体标记不是列表，则引发值错误
        if not isinstance(entity_spans, list):
            raise ValueError("entity_spans should be given as a list")
        # 如果实体标记不为空且第一个元素不是元组，则引发值错���
        elif len(entity_spans) > 0 and not isinstance(entity_spans[0], tuple):
            raise ValueError(
                "entity_spans should be given as a list of tuples containing the start and end character indices"
            )

        # 如果存在实体，则校验实体输入格式
        if entities is not None:
            if not isinstance(entities, list):
                raise ValueError("If you specify entities, they should be given as a list")

            if len(entities) > 0 and not isinstance(entities[0], str):
                raise ValueError("If you specify entities, they should be given as a list of entity names")

            if len(entities) != len(entity_spans):
                raise ValueError("If you specify entities, entities and entity_spans must be the same length")

    # 从transformers.models.luke.tokenization_luke.LukeTokenizer._create_input_sequence复制而来
    def _create_input_sequence(
        # 文本输入
        text: Union[TextInput],
        # 文本对输入，可选参数，默认为None
        text_pair: Optional[Union[TextInput]] = None,
        # 实体输入，可选参数，默认为None
        entities: Optional[EntityInput] = None,
        # 实体对输入，可选参数，默认为None
        entities_pair: Optional[EntityInput] = None,
        # 实体标记输入，可选参数，默认为None
        entity_spans: Optional[EntitySpanInput] = None,
        # 实体标记对输入，可选参数，默认为None
        entity_spans_pair: Optional[EntitySpanInput] = None,
        # 其他关键字参数
        **kwargs,
    # 从transformers.models.luke.tokenization_luke.LukeTokenizer._batch_prepare_for_model复制而来
    # 为模型准备批处理数据
    def _batch_prepare_for_model(
        # 批量的文本 ID 对，包括一个文本 ID 列表和一个空
        batch_ids_pairs: List[Tuple[List[int], None]],
        # 批量的实体 ID 对，包括一个实体 ID 列表和一个空
        batch_entity_ids_pairs: List[Tuple[Optional[List[int]], Optional[List[int]]],
        # 实体令牌跨度对，包括一个实体令牌跨度的列表和一个空
        batch_entity_token_spans_pairs: List[Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]],
        # 是否添加特殊标记
        add_special_tokens: bool = True,
        # 填充策略
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度
        max_length: Optional[int] = None,
        # 最大实体长度
        max_entity_length: Optional[int] = None,
        # 步进值
        stride: int = 0,
        # 填充到某个值的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回张量
        return_tensors: Optional[str] = None,
        # 返回令牌类型 ID
        return_token_type_ids: Optional[bool] = None,
        # 返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 返回溢出的令牌
        return_overflowing_tokens: bool = False,
        # 返回特殊令牌掩码
        return_special_tokens_mask: bool = False,
        # 返回长度
        return_length: bool = False,
        # 详细输出
        verbose: bool = True,
```  
    def prepare_input_ids(
        self,
        batch_ids_pairs: List[Union[List[int], Tuple[List[int], List[int]]]],
        batch_entity_ids_pairs: List[Union[List[int], Tuple[List[int], List[int]]]],
        batch_entity_token_spans_pairs: List[Union[List[Tuple[int, int]], Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]],
        max_entity_length: Optional[int] = None,
        add_special_tokens: bool = True,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        max_length: Optional[int] = None,
        stride: int = 0,
        return_token_type_ids: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        padding_strategy: Union[str, PaddingStrategy] = "longest",
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens


        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
            batch_entity_ids_pairs: list of entity ids or entity ids pairs
            batch_entity_token_spans_pairs: list of entity spans or entity spans pairs
            max_entity_length: The maximum length of the entity sequence.
        """

        # Initialize dictionary to store batch outputs
        batch_outputs = {}
        # Iterate over input batches
        for input_ids, entity_ids, entity_token_span_pairs in zip(
            batch_ids_pairs, batch_entity_ids_pairs, batch_entity_token_spans_pairs
        ):
            # Unpack input ids, entity ids, and entity token spans
            first_ids, second_ids = input_ids
            first_entity_ids, second_entity_ids = entity_ids
            first_entity_token_spans, second_entity_token_spans = entity_token_span_pairs
            # Prepare inputs for the model
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                entity_ids=first_entity_ids,
                pair_entity_ids=second_entity_ids,
                entity_token_spans=first_entity_token_spans,
                pair_entity_token_spans=second_entity_token_spans,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # Aggregate outputs into batch_outputs dictionary
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # Pad batch outputs according to padding strategy
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # Convert batch outputs to BatchEncoding object
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # Return batch outputs
        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从transformers.models.luke.tokenization_luke.LukeTokenizer.prepare_for_model复制而来
    def prepare_for_model(
        self,
        # 输入的 token ids 列表
        ids: List[int],
        # 第二个句子的 token ids 列表
        pair_ids: Optional[List[int]] = None,
        # 实体 token ids 列表
        entity_ids: Optional[List[int]] = None,
        # 第二个句子的实体 token ids 列表
        pair_entity_ids: Optional[List[int]] = None,
        # 实体 token 的起始和结束位置
        entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        # 第二个句子的实体 token 的起始和结束位置
        pair_entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        # 是否添加特殊 token
        add_special_tokens: bool = True,
        # 填充策略
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 最大实体长度限制
        max_entity_length: Optional[int] = None,
        # 步长
        stride: int = 0,
        # 填充到指定长度的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回类型，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 返回 token 类型 id
        return_token_type_ids: Optional[bool] = None,
        # 返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 返回溢出的 token
        return_overflowing_tokens: bool = False,
        # 返回特殊 token 掩码
        return_special_tokens_mask: bool = False,
        # 返回偏移映射
        return_offsets_mapping: bool = False,
        # 返回长度
        return_length: bool = False,
        # 冗长输出
        verbose: bool = True,
        # 是否在返回之前添加批次维度
        prepend_batch_axis: bool = False,
        **kwargs,
    
    # 从transformers.models.luke.tokenization_luke.LukeTokenizer.pad复制而来
    def pad(
        self,
        # 编码输入数据
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        # 填充策略
        padding: Union[bool, str, PaddingStrategy] = True,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 最大实体长度限制
        max_entity_length: Optional[int] = None,
        # 填充到指定长度的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 返回类型，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 冗长输出
        verbose: bool = True,
    
    # 从transformers.models.luke.tokenization_luke.LukeTokenizer._pad复制而来
    def _pad(
        self,
        # 编码输入数据
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        # 最大长度限制
        max_length: Optional[int] = None,
        # 最大实体长度限制
        max_entity_length: Optional[int] = None,
        # 填充策略
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 填充到指定长度的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回注意力掩码
        return_attention_mask: Optional[bool] = None,
    # 将词汇表保存到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        # 如果保存目录不存在，记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
    
        # 构造词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
    
        # 如果词汇表文件路径与当前路径不同且当前词汇表文件存在，则将其复制到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化的模型写入目标文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
    
        # 构造实体词汇表文件路径
        entity_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["entity_vocab_file"]
        )
    
        # 将实体词汇表以 JSON 格式写入目标文件
        with open(entity_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.entity_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    
        # 返回保存后的两个文件路径
        return out_vocab_file, entity_vocab_file
    
    # 构建用于序列分类任务的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        构建模型输入，包括添加特殊标记符。
        XLM-RoBERTa 序列格式如下:
        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s></s> B </s>`
        
        参数:
            token_ids_0 (`List[int]`):
                要添加特殊标记符的序列 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                可选的第二个序列 ID 列表。
    
        返回:
            `List[int]`: 包含适当特殊标记符的输入 ID 列表。
        """
        # 如果只有一个序列，在前后添加 cls 和 sep 标记符
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 如果有两个序列，在前后分别添加 cls 和 sep 标记符
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep
    
    # 获取输入序列中特殊标记符的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    # 从没有添加特殊标记的令牌列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。
    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False,
        ) -> List[int]:
            """
            Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
            special tokens using the tokenizer `prepare_for_model` method.
    
            Args:
                token_ids_0 (`List[int]`):
                    List of IDs.
                token_ids_1 (`List[int]`, *optional*):
                    Optional second list of IDs for sequence pairs.
                already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                    Whether or not the token list is already formatted with special tokens for the model.
    
            Returns:
                `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
            """
    
            # 如果令牌列表已经包含特殊标记,则直接返回父类的实现
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                )
    
            # 如果只有 token_ids_0,在开头和结尾添加特殊标记,其他部分标记为序列标记
            if token_ids_1 is None:
                return [1] + ([0] * len(token_ids_0)) + [1]
            # 如果有 token_ids_1,在开头、两个序列之间、结尾添加特殊标记,其他部分标记为序列标记
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]
    
        # 从 transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer 复制
        def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
            """
            Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
            not make use of token type ids, therefore a list of zeros is returned.
    
            Args:
                token_ids_0 (`List[int]`):
                    List of IDs.
                token_ids_1 (`List[int]`, *optional*):
                    Optional second list of IDs for sequence pairs.
    
            Returns:
                `List[int]`: List of zeros.
    
            """
    
            # 定义分隔符和分类器标记 ID
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]
    
            # 如果只有 token_ids_0,返回全 0 列表
            if token_ids_1 is None:
                return len(cls + token_ids_0 + sep) * [0]
            # 如果有 token_ids_1,返回全 0 列表
            return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```