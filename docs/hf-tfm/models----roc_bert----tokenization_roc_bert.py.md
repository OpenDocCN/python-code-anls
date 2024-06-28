# `.\models\roc_bert\tokenization_roc_bert.py`

```
# coding=utf-8
# Copyright 2022 WeChatAI and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for RoCBert."""

import collections  # 导入 collections 模块，用于处理数据集合
import itertools  # 导入 itertools 模块，用于高效循环操作
import json  # 导入 json 模块，用于处理 JSON 数据
import os  # 导入 os 模块，用于操作系统相关的功能
import unicodedata  # 导入 unicodedata 模块，用于 Unicode 字符处理
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  # 导入 tokenization_utils 中的函数和类
from ...tokenization_utils_base import (  # 导入 tokenization_utils_base 中的各种类和函数
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import add_end_docstrings, logging  # 导入 utils 中的函数和类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {  # 定义用于存储词汇文件名的字典
    "vocab_file": "vocab.txt",
    "word_shape_file": "word_shape.json",
    "word_pronunciation_file": "word_pronunciation.json",
}

PRETRAINED_VOCAB_FILES_MAP = {  # 预训练模型的词汇文件映射
    "vocab_file": {
        "weiweishi/roc-bert-base-zh": "https://huggingface.co/weiweishi/roc-bert-base-zh/resolve/main/vocab.txt"
    },
    "word_shape_file": {
        "weiweishi/roc-bert-base-zh": "https://huggingface.co/weiweishi/roc-bert-base-zh/resolve/main/word_shape.json"
    },
    "word_pronunciation_file": {
        "weiweishi/roc-bert-base-zh": (
            "https://huggingface.co/weiweishi/roc-bert-base-zh/resolve/main/word_pronunciation.json"
        )
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {  # 预训练位置嵌入大小的映射
    "weiweishi/roc-bert-base-zh": 512,
}

PRETRAINED_INIT_CONFIGURATION = {  # 预训练初始化配置
    "weiweishi/roc-bert-base-zh": {"do_lower_case": True},
}


# Copied from transformers.models.bert.tokenization_bert.load_vocab
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建有序字典来存储词汇
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇文件进行读取
        tokens = reader.readlines()  # 逐行读取词汇文件内容
    for index, token in enumerate(tokens):  # 遍历读取的词汇列表
        token = token.rstrip("\n")  # 去除每个词汇末尾的换行符
        vocab[token] = index  # 将词汇和对应的索引添加到字典中
    return vocab  # 返回构建好的词汇字典


# Copied from transformers.models.bert.tokenization_bert.whitespace_tokenize
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本首尾的空白字符
    if not text:  # 如果文本为空
        return []  # 返回空列表
    tokens = text.split()  # 使用空白字符分割文本，得到词汇列表
    return tokens  # 返回分割后的词汇列表


class RoCBertTokenizer(PreTrainedTokenizer):
    r"""
    Args:
    # 构建一个 RoCBert 分词器，基于 WordPiece。该分词器继承自 `PreTrainedTokenizer`，其中包含大多数主要方法。
    # 用户应参考该超类以获取有关这些方法的更多信息。
    def __init__(
        self,
        vocab_file: str,
        word_shape_file: str,
        word_pronunciation_file: str,
        do_lower_case: bool = True,
        do_basic_tokenize: bool = True,
        never_split: Iterable[str] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        tokenize_chinese_chars: bool = True,
        strip_accents: bool = None
    ):
        # 词汇表文件名
        self.vocab_files_names = VOCAB_FILES_NAMES
        # 预训练模型的词汇文件映射
        self.pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 预训练模型的初始化配置
        self.pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
        # 预训练位置嵌入的最大输入尺寸
        self.max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    def __init__(
        self,
        vocab_file,
        word_shape_file,
        word_pronunciation_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 检查并确保提供的文件路径有效，如果不存在或者不是文件，则抛出异常
        for cur_file in [vocab_file, word_shape_file, word_pronunciation_file]:
            if cur_file is None or not os.path.isfile(cur_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google "
                    "pretrained model use `tokenizer = RoCBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )

        # 加载词汇表文件并存储到 self.vocab 中
        self.vocab = load_vocab(vocab_file)

        # 使用 UTF-8 编码打开词形文件，并加载其内容到 self.word_shape 中
        with open(word_shape_file, "r", encoding="utf8") as in_file:
            self.word_shape = json.load(in_file)

        # 使用 UTF-8 编码打开发音文件，并加载其内容到 self.word_pronunciation 中
        with open(word_pronunciation_file, "r", encoding="utf8") as in_file:
            self.word_pronunciation = json.load(in_file)

        # 创建一个从 ID 到 token 的有序字典，用于反向查找
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        # 设置是否执行基本的分词操作
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            # 如果需要基本分词，则初始化 RoCBertBasicTokenizer 对象
            self.basic_tokenizer = RoCBertBasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        
        # 使用给定的未知 token 初始化 RoCBertWordpieceTokenizer 对象
        self.wordpiece_tokenizer = RoCBertWordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        
        # 调用父类的初始化方法，设置通用的 tokenizer 参数
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        # 返回当前 tokenizer 是否执行小写化处理的状态
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回当前词汇表的大小（词汇表中 token 的数量）
        return len(self.vocab)

    # 从 transformers 库中复制的方法，返回当前 tokenizer 的完整词汇表（包括添加的特殊 token）
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 从 transformers 库中复制的方法，用于实现 tokenization_bert.BertTokenizer._tokenize 的功能
    # 使用self对象的basic_tokenizer对文本进行基本的tokenization处理
    def _tokenize(self, text, split_special_tokens=False):
        # 初始化空列表，用于存储分割后的token
        split_tokens = []
        # 如果设置了do_basic_tokenize标志为True
        if self.do_basic_tokenize:
            # 调用basic_tokenizer的tokenize方法对文本进行处理
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果token在never_split集合中，则直接加入split_tokens列表
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                # 否则使用wordpiece_tokenizer对token进行进一步的分割处理，并将结果加入split_tokens列表
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果do_basic_tokenize标志为False，则直接使用wordpiece_tokenizer对文本进行处理
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回处理后的token列表
        return split_tokens

    # 使用@add_end_docstrings注解来添加文档字符串，描述函数的各个参数
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: List[int],
        shape_ids: List[int],
        pronunciation_ids: List[int],
        pair_ids: Optional[List[int]] = None,
        pair_shape_ids: Optional[List[int]] = None,
        pair_pronunciation_ids: Optional[List[int]] = None,
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
    ):
        # 函数用于将输入的ids、shape_ids和pronunciation_ids编码成模型输入
        # 如果提供了pair_ids、pair_shape_ids和pair_pronunciation_ids，也会进行相应处理
        # 设置是否添加特殊token，默认为True
        # 设置padding策略，默认不进行padding
        # 设置截断策略，默认不进行截断
        # 设置最大长度限制，默认为None
        # 设置stride，默认为0
        # 设置是否将输入拆分成单词，默认为False
        # 设置pad_to_multiple_of，用于设置多少的倍数进行padding，默认为None
        # 设置返回的张量类型，默认为None
        # 设置是否返回token类型ID，默认为None
        # 设置是否返回注意力掩码，默认为None
        # 设置是否返回溢出的token，默认为False
        # 设置是否返回特殊token掩码，默认为False
        # 设置是否返回偏移映射，默认为False
        # 设置是否返回长度，默认为False
        # 设置是否输出详细信息，默认为True
        # 设置是否在前面添加批次轴，默认为False
        # kwargs用于接收其它可能的关键字参数
        pass  # 函数体未提供，暂时占位
    # 在类中定义一个方法 `_pad`，用于填充输入数据以达到指定的最大长度
    def _pad(
        # 接受一个字典或批量编码作为输入，其中键是字符串，值是编码后的输入
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        # 可选参数，指定填充后的最大长度
        max_length: Optional[int] = None,
        # 可选参数，填充策略，默认不进行填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 可选参数，指定填充后的长度是某数的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 可选参数，是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        # 如果 return_attention_mask 为 None，则根据模型输入名称判断是否需要返回 attention_mask
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # 获取必需的输入，通常为第一个模型输入的编码结果
        required_input = encoded_inputs[self.model_input_names[0]]

        # 根据 padding_strategy 是否为 LONGEST，确定 max_length
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        # 如果 max_length 和 pad_to_multiple_of 都有值，并且 max_length 不是 pad_to_multiple_of 的倍数，调整 max_length
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # 判断是否需要进行填充
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # 如果需要返回 attention_mask 并且 encoded_inputs 中没有 "attention_mask"，则初始化 attention_mask
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        # 如果需要填充
        if needs_to_be_padded:
            difference = max_length - len(required_input)

            # 如果填充在右侧
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                for key in ["input_shape_ids", "input_pronunciation_ids"]:
                    if key in encoded_inputs:
                        encoded_inputs[key] = encoded_inputs[key] + [self.pad_token_id] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            # 如果填充在左侧
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                for key in ["input_shape_ids", "input_pronunciation_ids"]:
                    if key in encoded_inputs:
                        encoded_inputs[key] = [self.pad_token_id] * difference + encoded_inputs[key]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                # 如果填充策略不是 "left" 或 "right"，抛出异常
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        # 返回填充后的 encoded_inputs 字典
        return encoded_inputs
    # 定义一个方法用于批量编码文本或文本对，支持多种输入类型
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 步长，默认为0
        is_split_into_words: bool = False,  # 输入是否已经分成单词
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊tokens的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否详细输出
        **kwargs,  # 其他关键字参数
    ):
        # 方法功能的详细描述和补充参数文档参见ENCODE_KWARGS_DOCSTRING和ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
        pass

    # 定义一个方法用于为模型准备批量数据，支持多种输入类型
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        batch_shape_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        batch_pronunciation_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 步长，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数
        return_tensors: Optional[str] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊tokens的掩码
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否详细输出
        **kwargs,  # 其他关键字参数
    ):
        # 方法用于准备模型输入数据的详细描述和补充参数文档参见相应文档
        pass
    ) -> BatchEncoding:
        """
        准备一个输入 id 序列，或者一对输入 id 序列，以便可以被模型使用。它添加特殊标记，根据特殊标记截断序列，同时考虑特殊标记，并管理一个移动窗口（带有用户定义的步长）来处理溢出的标记。

        Args:
            batch_ids_pairs: tokenized input ids 或者 input ids pairs 的列表
            batch_shape_ids_pairs: tokenized input shape ids 或者 input shape ids pairs 的列表
            batch_pronunciation_ids_pairs: tokenized input pronunciation ids 或者 input pronunciation ids pairs 的列表
        """

        batch_outputs = {}
        for i, (first_ids, second_ids) in enumerate(batch_ids_pairs):
            first_shape_ids, second_shape_ids = batch_shape_ids_pairs[i]
            first_pronunciation_ids, second_pronunciation_ids = batch_pronunciation_ids_pairs[i]
            outputs = self.prepare_for_model(
                first_ids,
                first_shape_ids,
                first_pronunciation_ids,
                pair_ids=second_ids,
                pair_shape_ids=second_shape_ids,
                pair_pronunciation_ids=second_pronunciation_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 在后续批处理中进行填充
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # 在后续批处理中进行填充
                return_attention_mask=False,  # 在后续批处理中进行填充
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # 最终将整个批次转换为张量
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer._convert_token_to_id 复制过来的
    def _convert_token_to_id(self, token):
        """使用词汇表将一个标记（字符串）转换为对应的 id。"""
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    # 使用 shape vocab 将给定的 token 转换成对应的 shape_id
    def _convert_token_to_shape_id(self, token):
        """Converts a token (str) in an shape_id using the shape vocab."""
        return self.word_shape.get(token, self.word_shape.get(self.unk_token))

    # 将一组 token 转换成对应的 shape_ids 列表或单个 shape_id
    def convert_tokens_to_shape_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_shape_id(token))
        return ids

    # 使用 pronunciation vocab 将给定的 token 转换成对应的 pronunciation_id
    def _convert_token_to_pronunciation_id(self, token):
        """Converts a token (str) in an shape_id using the shape vocab."""
        return self.word_pronunciation.get(token, self.word_pronunciation.get(self.unk_token))

    # 将一组 token 转换成对应的 pronunciation_ids 列表或单个 pronunciation_id
    def convert_tokens_to_pronunciation_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_pronunciation_id(token))
        return ids

    # 从词汇表中将给定的 index 转换成对应的 token
    # 这里的词汇表是 ids_to_tokens 字典
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将一组 token 序列转换成单个字符串，去除特殊标记 "##"
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建包含特殊 token 的模型输入序列，用于序列分类任务
    # 可能是单个序列 `[CLS] X [SEP]` 或者序列对 `[CLS] A [SEP] B [SEP]`
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        cls_token_id: int = None,
        sep_token_id: int = None,
    ) -> List[int]:
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
        cls = [self.cls_token_id] if cls_token_id is None else [cls_token_id]
        sep = [self.sep_token_id] if sep_token_id is None else [sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取特殊 token 的掩码
    # 由于该函数被省略了，文档中提到的函数为 transformers.models.bert.tokenization_bert.BertTokenizer.get_special_tokens_mask
    # 检查是否已经包含特殊标记，如果是，则调用父类方法返回特殊标记掩码
    if already_has_special_tokens:
        return super().get_special_tokens_mask(
            token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
        )

    # 如果有第二个序列token_ids_1，则返回带有特殊标记的掩码列表
    if token_ids_1 is not None:
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    # 否则，只返回第一个序列token_ids_0带有特殊标记的掩码列表
    return [1] + ([0] * len(token_ids_0)) + [1]



    # 从给定的序列中创建token type IDs，用于序列对分类任务
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]  # 分隔符token的ID列表
        cls = [self.cls_token_id]  # 类别标记token的ID列表
        if token_ids_1 is None:
            # 如果没有第二个序列，返回仅包含第一个序列的token type IDs列表（都是0）
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回包含两个序列的token type IDs列表，第一个序列为0，第二个序列为1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 将词汇表和相关文件保存到指定的目录中，并返回文件路径的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str, str]:
        index = 0  # 初始化索引变量
        if os.path.isdir(save_directory):  # 检查保存目录是否存在
            # 构建词汇表文件的路径，如果有前缀，则添加前缀，否则直接使用文件名
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"],
            )
            # 构建词形文件的路径，同上
            word_shape_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["word_shape_file"],
            )
            # 构建发音文件的路径，同上
            word_pronunciation_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["word_pronunciation_file"],
            )
        else:
            # 如果目录不存在，则抛出值错误
            raise ValueError(
                f"Can't find a directory at path '{save_directory}'. To load the vocabulary from a Google "
                "pretrained model use `tokenizer = RoCBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        # 打开并写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的每个单词及其索引
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 检查索引是否连续，若不连续则记录警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入单词到文件中，并换行
                writer.write(token + "\n")
                index += 1

        # 打开并写入词形文件，以 JSON 格式存储
        with open(word_shape_file, "w", encoding="utf8") as writer:
            json.dump(self.word_shape, writer, ensure_ascii=False, indent=4, separators=(", ", ": "))

        # 打开并写入发音文件，以 JSON 格式存储
        with open(word_pronunciation_file, "w", encoding="utf8") as writer:
            json.dump(self.word_pronunciation, writer, ensure_ascii=False, indent=4, separators=(", ", ": "))

        # 返回保存的文件路径的元组
        return (
            vocab_file,
            word_shape_file,
            word_pronunciation_file,
        )
# 从 transformers.models.bert.tokenization_bert.BasicTokenizer 复制的 RoCBertBasicTokenizer 类定义，
# 将 BasicTokenizer 更名为 RoCBertBasicTokenizer
class RoCBertBasicTokenizer(object):
    """
    构建一个 RoCBertBasicTokenizer 类，用于执行基本的分词（如标点符号分割、转换为小写等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在分词时不会被拆分的标记集合。仅在 `do_basic_tokenize=True` 时有效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行分词。

            对于日文，这个选项应该关闭（参见此 [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果未指定此选项，则将由 `lowercase` 的值确定（与原始 BERT 一样）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕获单词的完整上下文，例如缩略词。
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case  # 是否将输入转换为小写
        self.never_split = set(never_split)  # 在分词时不拆分的标记集合
        self.tokenize_chinese_chars = tokenize_chinese_chars  # 是否对中文字符进行分词
        self.strip_accents = strip_accents  # 是否去除所有重音符号
        self.do_split_on_punc = do_split_on_punc  # 是否在基本标点符号上进行分割
    # Tokenize a piece of text using basic tokenization rules.
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # Create a set of tokens that should never be split during tokenization.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # Clean the input text.
        text = self._clean_text(text)

        # Handle Chinese character tokenization if enabled.
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        
        # Normalize the text to Unicode NFC form to ensure uniform representation of characters.
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # Tokenize the text on whitespace.
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        
        # Iterate over each token and process according to tokenizer settings.
        for token in orig_tokens:
            if token not in never_split:
                # Lowercase the token if required by the tokenizer settings.
                if self.do_lower_case:
                    token = token.lower()
                    # Strip accents from the token if specified.
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # Split the token on punctuation marks while respecting tokens in never_split.
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # Tokenize the final output tokens on whitespace and return them.
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # Remove accents (diacritical marks) from a piece of text.
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # Normalize the text to Unicode NFD form to decompose characters and accents.
        text = unicodedata.normalize("NFD", text)
        output = []
        # Iterate over each character in the normalized text.
        for char in text:
            # Check the Unicode category of the character; "Mn" denotes a nonspacing mark (accents).
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue  # Skip combining characters (accents).
            output.append(char)
        # Join the processed characters back into a string without accents.
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割，或者文本在 never_split 中，直接返回文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则将其作为新的单独列表项添加到 output 中
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，根据 start_new_word 的状态判断是否开始一个新的单词
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将字符添加到当前的最后一个列表项中
                output[-1].append(char)
            i += 1

        # 将列表中的列表项连接成字符串并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是中日韩字符，则在其前后添加空格并加入输出列表
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 否则直接加入输出列表
                output.append(char)
        # 将输出列表连接成字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的码点是否在中日韩字符的 Unicode 块范围内
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则替换为单个空格；否则直接加入输出列表
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将输出列表连接成字符串并返回
        return "".join(output)
# Copied from  transformers.models.bert.tokenization_bert.WordpieceTokenizer with WordpieceTokenizer->RoCBertWordpieceTokenizer
class RoCBertWordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化方法，接受词汇表、未知token以及每个单词的最大字符数作为参数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """
        # 将文本分词为 wordpiece tokens 的方法
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # 如果单词长度超过设定的最大字符数，将其标记为未知token
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                # 如果无法找到匹配的子串，则将整个单词标记为未知token
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```