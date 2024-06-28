# `.\models\mluke\tokenization_mluke.py`

```py
# 导入所需的模块和库
import itertools  # 导入itertools模块，用于高效循环操作
import json  # 导入json模块，用于处理JSON格式的数据
import os  # 导入os模块，用于与操作系统进行交互
from collections.abc import Mapping  # 从collections.abc模块导入Mapping抽象基类
from shutil import copyfile  # 从shutil模块导入copyfile函数，用于复制文件
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示所需的类和函数

import numpy as np  # 导入numpy库，用于数值计算
import sentencepiece as spm  # 导入sentencepiece库，用于处理文本分词

# 导入transformers库中的相关模块和函数
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
from ...utils import add_end_docstrings, is_tf_tensor, is_torch_tensor, logging  # 导入一些工具函数和类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

EntitySpan = Tuple[int, int]  # 定义EntitySpan类型为(int, int)元组
EntitySpanInput = List[EntitySpan]  # 定义EntitySpanInput类型为元素为EntitySpan的列表
Entity = str  # 定义Entity类型为字符串
EntityInput = List[Entity]  # 定义EntityInput类型为元素为Entity的列表

SPIECE_UNDERLINE = "▁"  # 定义SPIECE_UNDERLINE为"▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "entity_vocab_file": "entity_vocab.json"}  # 定义VOCAB_FILES_NAMES字典

# 定义PRETRAINED_VOCAB_FILES_MAP字典，包含预训练模型和其对应的文件路径
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

# 定义PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES字典，包含预训练模型和其对应的位置嵌入大小
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

    # 定义MLukeTokenizer类，继承自PreTrainedTokenizer类

    vocab_files_names = VOCAB_FILES_NAMES  # 设置类属性vocab_files_names为VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置类属性pretrained_vocab_files_map为PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 设置类属性max_model_input_sizes为PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]  # 设置类属性model_input_names为包含"input_ids"和"attention_mask"的列表
    # 初始化方法，用于创建一个新的 tokenizer 对象
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
    ):
        # 继承父类构造方法
        super().__init__(**kwargs)

    @property
    # 计算属性：返回 tokenizer 的词汇量大小
    # 从 transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer.vocab_size 复制而来
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 1  # 添加 <mask> token 的数量到词汇量中

    # 获取词汇表的方法
    # 从 transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer.get_vocab 复制而来
    def get_vocab(self):
        # 创建词汇表字典，包括转换后的 token 到 id 的映射
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 分词方法
    # 从 transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer._tokenize 复制而来
    def _tokenize(self, text: str) -> List[str]:
        # 使用 SentencePiece 模型进行文本编码
        # TODO 检查是否适用于 t5/llama PR
        return self.sp_model.encode(text, out_type=str)

    # 将 token 转换为 id 的方法
    # 从 transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer._convert_token_to_id 复制而来
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 如果 token 存在于 fairseq_tokens_to_ids 中，直接返回对应的 id
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        # 否则，使用 SentencePiece 模型将 token 转换为 id
        spm_id = self.sp_model.PieceToId(token)

        # 如果 spm_id 为 0，返回未知 token 的 id
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    # 将 id 转换为 token 的方法
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 如果 index 存在于 fairseq_ids_to_tokens 中，直接返回对应的 token
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        # 否则，使用 SentencePiece 模型将 id 转换为 token
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    # 将 token 序列转换为字符串的方法
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        # 将 token 序列连接成字符串，并替换 SPIECE_UNDERLINE 为空格，然后去除首尾空格
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # 序列化对象状态的方法
    def __getstate__(self):
        state = self.__dict__.copy()
        # 清空 sp_model，因为其不可序列化
        state["sp_model"] = None
        # 将 sp_model_proto 序列化并保存在状态中
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    # 反序列化对象状态的方法
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容，如果不存在 sp_model_kwargs，设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 根据 sp_model_kwargs 创建 SentencePieceProcessor 对象，并从序列化的 proto 中加载模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
    # 将两个参数字典化，用于增强函数文档的功能
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从 transformers.models.luke.tokenization_luke.LukeTokenizer.__call__ 复制而来的函数定义
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
    ):
        # 从 transformers.models.luke.tokenization_luke.LukeTokenizer._encode_plus 复制而来的函数定义
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
        ):
    # 定义函数签名，声明返回类型为 BatchEncoding
    ) -> BatchEncoding:
        # 如果要求返回偏移映射，则抛出 NotImplementedError 异常
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 如果指定 is_split_into_words 参数为 True，则抛出 NotImplementedError 异常
        if is_split_into_words:
            raise NotImplementedError("is_split_into_words is not supported in this tokenizer.")

        (
            # 调用内部方法 _create_input_sequence，生成输入序列所需的各个 ID 和实体标记跨度
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        ) = self._create_input_sequence(
            # 传入文本和其配对文本、实体和其配对实体、实体标记跨度等参数
            text=text,
            text_pair=text_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=entity_spans,
            entity_spans_pair=entity_spans_pair,
            **kwargs,  # 接受其他可能的关键字参数
        )

        # 调用 prepare_for_model 方法，生成模型输入所需的 attention_mask 和 token_type_ids
        # 返回结果作为函数结果
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

    # 从 transformers.models.luke.tokenization_luke.LukeTokenizer._batch_encode_plus 复制而来的代码
    # 定义一个方法用于批量编码文本或文本对，同时处理实体和实体跨度输入
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair]],
        batch_entity_spans_or_entity_spans_pairs: Optional[
            Union[List[EntitySpanInput], List[Tuple[EntitySpanInput, EntitySpanInput]]]
        ] = None,
        batch_entities_or_entities_pairs: Optional[
            Union[List[EntityInput], List[Tuple[EntityInput, EntityInput]]]
        ] = None,
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
    ):
        # 检查实体输入格式的方法，确保实体跨度是列表且长度匹配
        # 如果 entity_spans 不是列表，抛出 ValueError
        if not isinstance(entity_spans, list):
            raise ValueError("entity_spans should be given as a list")
        # 如果 entity_spans 长度大于 0 且第一个元素不是元组，抛出 ValueError
        elif len(entity_spans) > 0 and not isinstance(entity_spans[0], tuple):
            raise ValueError(
                "entity_spans should be given as a list of tuples containing the start and end character indices"
            )

        # 如果指定了 entities，则需检查其格式是否正确
        if entities is not None:
            # 如果 entities 不是列表，抛出 ValueError
            if not isinstance(entities, list):
                raise ValueError("If you specify entities, they should be given as a list")
            # 如果 entities 长度大于 0 且第一个元素不是字符串，抛出 ValueError
            if len(entities) > 0 and not isinstance(entities[0], str):
                raise ValueError("If you specify entities, they should be given as a list of entity names")
            # 如果 entities 和 entity_spans 长度不一致，抛出 ValueError
            if len(entities) != len(entity_spans):
                raise ValueError("If you specify entities, entities and entity_spans must be the same length")

    # 创建输入序列的方法，用于构建模型输入
    def _create_input_sequence(
        self,
        text: Union[TextInput],
        text_pair: Optional[Union[TextInput]] = None,
        entities: Optional[EntityInput] = None,
        entities_pair: Optional[EntityInput] = None,
        entity_spans: Optional[EntitySpanInput] = None,
        entity_spans_pair: Optional[EntitySpanInput] = None,
        **kwargs,
    ):
        pass  # 此方法的实现可能包括文本编码、实体标记等操作，这里未提供具体实现

    # 批量为模型准备输入的方法，扩展了 _batch_encode_plus 方法的功能
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 该装饰器通常用于增强方法的文档说明，添加了关于编码的额外参数的文档说明
    # 实际功能由所在类中的 _batch_encode_plus 方法实现
    # 批量准备模型输入数据，为每个批次准备模型所需的输入
    def _batch_prepare_for_model(
        # 批次中每个样本的ID与空值对，即[(ids, None), ...]
        batch_ids_pairs: List[Tuple[List[int], None]],
        # 批次中每个样本的实体ID对，即[(entity_ids1, entity_ids2), ...]
        batch_entity_ids_pairs: List[Tuple[Optional[List[int]], Optional[List[int]]]],
        # 批次中每个样本的实体token span对，即[(token_spans1, token_spans2), ...]
        batch_entity_token_spans_pairs: List[Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]],
        # 是否添加特殊token，如[CLS]和[SEP]
        add_special_tokens: bool = True,
        # 填充策略，默认不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制，可以为None
        max_length: Optional[int] = None,
        # 实体最大长度限制，可以为None
        max_entity_length: Optional[int] = None,
        # 步进值，默认为0
        stride: int = 0,
        # 填充到某个倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，可以为None
        return_tensors: Optional[str] = None,
        # 是否返回token类型ID，可以为None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，可以为None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的token，默认为False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊token掩码，默认为False
        return_special_tokens_mask: bool = False,
        # 是否返回长度信息，默认为False
        return_length: bool = False,
        # 是否输出详细信息，即使默认为True
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

        batch_outputs = {}  # 初始化一个空字典用于存储每个批次的输出结果
        for input_ids, entity_ids, entity_token_span_pairs in zip(
            batch_ids_pairs, batch_entity_ids_pairs, batch_entity_token_spans_pairs
        ):
            first_ids, second_ids = input_ids  # 将输入的 ids 对分解为第一个和第二个序列
            first_entity_ids, second_entity_ids = entity_ids  # 将实体 ids 对分解为第一个和第二个实体序列
            first_entity_token_spans, second_entity_token_spans = entity_token_span_pairs  # 将实体 token spans 对分解为第一个和第二个实体 token spans

            # 调用 self.prepare_for_model 方法处理输入，准备模型输入
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                entity_ids=first_entity_ids,
                pair_entity_ids=second_entity_ids,
                entity_token_spans=first_entity_token_spans,
                pair_entity_token_spans=second_entity_token_spans,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 在之后的批次中进行填充
                truncation=truncation_strategy.value,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                pad_to_multiple_of=None,  # 在之后的批次中进行填充
                return_attention_mask=False,  # 在之后的批次中返回注意力掩码
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # 最终将整个批次转换为张量
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # 将每个输出添加到 batch_outputs 字典中对应的列表中
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # 调用 self.pad 方法对批次进行填充处理
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 将处理后的 batch_outputs 转换为 BatchEncoding 对象
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # 返回最终的 batch_outputs
        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer.prepare_for_model
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        entity_ids: Optional[List[int]] = None,
        pair_entity_ids: Optional[List[int]] = None,
        entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        pair_entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
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
    ):
        # 准备输入数据以供模型使用，可以设置添加特殊标记、填充、截断等策略
        pass
    
    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer.pad
    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ):
        # 对输入进行填充处理，支持不同的填充策略和最大长度设置
        pass
    
    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer._pad
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        # 内部方法：根据指定的填充策略对输入进行填充，支持最大长度和多样化填充倍数
        pass
    # 将词汇表保存到指定目录下的文件中，并返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建输出词汇表文件路径，根据可选的前缀和文件名构造
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将模型序列化后的内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 构建实体词汇表文件路径
        entity_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["entity_vocab_file"]
        )

        # 将实体词汇表以 JSON 格式写入文件
        with open(entity_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.entity_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 返回保存的词汇表文件路径和实体词汇表文件路径
        return out_vocab_file, entity_vocab_file

    # 从 XLM-RoBERTa Tokenizer 类中复制的方法：构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        # 如果只有一个输入序列，添加起始和结束标记
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 如果有两个输入序列，添加起始标记、两个结束标记和分隔标记
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # 从 XLM-RoBERTa Tokenizer 类中复制的方法：获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    # 从没有添加特殊标记的标记列表中提取序列 ID。当使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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

        if already_has_special_tokens:
            # 如果已经存在特殊标记，直接调用父类的方法获取特殊标记掩码
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # 如果没有第二个序列，返回一个列表：开始标记、token_ids_0 的长度个零、结束标记
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 如果有第二个序列，返回一个列表：开始标记、token_ids_0 的长度个零、结束标记、两个分隔符、token_ids_1 的长度个零、结束标记
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # 从 transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer.create_token_type_ids_from_sequences 复制
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

        sep = [self.sep_token_id]  # 分隔符的 token ID
        cls = [self.cls_token_id]  # 开始标记的 token ID

        if token_ids_1 is None:
            # 如果没有第二个序列，返回一个全零列表，长度为开始标记、token_ids_0、分隔符的总长度
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有第二个序列，返回一个全零列表，长度为开始标记、token_ids_0、两个分隔符、token_ids_1、分隔符的总长度
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```