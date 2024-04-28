# `.\transformers\models\roc_bert\tokenization_roc_bert.py`

```py
# 设置代码文件的编码格式为 UTF-8
# 版权声明，标注了文件所有者和版权信息
# 使用 Apache License 2.0 开源协议，允许使用者在遵守协议的前提下自由使用该文件
# 引入 collections 模块，用于创建字典
# 引入 itertools 模块，用于创建迭代器
# 引入 json 模块，用于处理 JSON 数据
# 引入 os 模块，提供了丰富的方法用于处理文件和目录
# 引入 unicodedata 模块，用于处理 Unicode 字符
# 引入 typing 模块，提供了类型相关的操作
# 从 tokenization_utils 模块中导入 PreTrainedTokenizer 类和一些辅助函数
# 从 tokenization_utils_base 模块中导入一些与编码、输入等相关的类和函数
# 引入 add_end_docstrings 装饰器，用于添加函数文档字符串的结尾
# 引入 logging 模块，用于记录日志信息

import collections
import itertools
import json
import os
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
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
from ...utils import add_end_docstrings, logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "word_shape_file": "word_shape.json",
    "word_pronunciation_file": "word_pronunciation.json",
}

# 预训练模型的词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
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

# 预训练模型的位置嵌入大小映射字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "weiweishi/roc-bert-base-zh": 512,
}

# 预训练模型的初始化配置字典
PRETRAINED_INIT_CONFIGURATION = {
    "weiweishi/roc-bert-base-zh": {"do_lower_case": True},
}


# 从 transformers.models.bert.tokenization_bert.load_vocab 函数复制过来
# 加载词汇文件到字典中
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 函数复制过来
# 对文本进行基本的空格清理和分割
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


# 定义 RoCBertTokenizer 类，继承自 PreTrainedTokenizer 类
class RoCBertTokenizer(PreTrainedTokenizer):
    r"""
    Args:
    # 构建一个 RoCBert tokenizer。基于 WordPiece 算法实现。这个 tokenizer 继承自 PreTrainedTokenizer 类，包含了大多数主要方法。用户可以参考该父类的文档获得更多信息。
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
    ):
        # 词汇表文件
        vocab_file (`str`):
            包含词汇表的文件。
        # 词形信息文件    
        word_shape_file (`str`):
            包含词 => 词形信息的文件。
        # 词发音信息文件
        word_pronunciation_file (`str`):
            包含词 => 发音信息的文件。
        # 是否转换为小写
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在标记化时将输入转换为小写。
        # 是否进行基本标记化
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            是否在 WordPiece 之前进行基本标记化。
        # 永不拆分的标记
        never_split (`Iterable`, *optional*):
            永不拆分的标记集合。只有在 `do_basic_tokenize=True` 时生效。
        # 未知标记
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。一个不在词汇表中的标记将被转换为此标记。
        # 分隔标记
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔标记。用于构建多个序列的序列，如序列分类或问答任务中的文本和问题。也用作序列的最后一个标记。
        # 填充标记
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            填充标记。用于填充不同长度序列以便批处理。
        # 分类标记
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类标记。用于序列分类任务（对整个序列而不是每个标记进行分类）。是构建有特殊标记序列时的第一个标记。
        # 掩蔽标记
        mask_token (`str`, *optional`, defaults to `"[MASK]"`):
            掩蔽标记。用于掩蔽值。这是在使用掩蔽语言模型训练该模型时使用的标记。是模型试图预测的标记。
        # 是否标记化中文字符
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否标记化中文字符。对日语可能需要禁用（参见此 issue）。
        # 是否去除重音
        strip_accents (`bool`, *optional*):
            是否去除所有重音。如果未指定此选项，则将由 `lowercase` 的值决定（与原始 BERT 一致）。
    
        # 词汇表文件名
        vocab_files_names = VOCAB_FILES_NAMES
        # 预训练词汇表文件映射
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 预训练初始化配置
        pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
        # 预训练位置嵌入大小
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
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
        # 检查给定的文件路径是否存在，如果不存在则引发异常
        for cur_file in [vocab_file, word_shape_file, word_pronunciation_file]:
            if cur_file is None or not os.path.isfile(cur_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google "
                    "pretrained model use `tokenizer = RoCBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )

        # 加载词汇表文件内容并存储在self.vocab中
        self.vocab = load_vocab(vocab_file)

        # 加载词形文件内容并存储在self.word_shape中
        with open(word_shape_file, "r", encoding="utf8") as in_file:
            self.word_shape = json.load(in_file)

        # 加载发音文件内容并存储在self.word_pronunciation中
        with open(word_pronunciation_file, "r", encoding="utf8") as in_file:
            self.word_pronunciation = json.load(in_file)

        # 创建ids到tokens的映射关系，存储在self.ids_to_tokens中
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        # 初始化基本tokenize标志
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            # 如果进行基本tokenize，则初始化RoCBertBasicTokenizer
            self.basic_tokenizer = RoCBertBasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 初始化RoCBertWordpieceTokenizer
        self.wordpiece_tokenizer = RoCBertWordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        super().__init__(
            # 初始化超类
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
        # 获取基本tokenizer的小写���志
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 获取词汇表大小
        return len(self.vocab)

    # 从transformers.models.bert.tokenization_bert.BertTokenizer.get_vocab复制而来
    def get_vocab(self):
        # 获取词汇表及附加的tokens映射关系
        return dict(self.vocab, **self.added_tokens_encoder)

    # 从transformers.models.bert.tokenization_bert.BertTokenizer._tokenize复制而来
    # 将文本进行分词处理，返回分词后的列表
    def _tokenize(self, text, split_special_tokens=False):
        # 初始化一个空列表用于存储分词结果
        split_tokens = []
        # 如果需要进行基础分词处理
        if self.do_basic_tokenize:
            # 调用基础分词器对文本进行分词
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果分词结果在不分割的特殊标记集合中
                if token in self.basic_tokenizer.never_split:
                    # 将该分词结果直接添加到结果列表中
                    split_tokens.append(token)
                else:
                    # 否则，使用 WordPiece 分词器对该分词结果进行进一步分词
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 否则，直接使用 WordPiece 分词器对文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回分词结果列表
        return split_tokens

    # 对输入进行编码，生成模型输入所需的特征
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
```  
    # 在当前类中定义一个私有方法，用于填充输入序列
    def _pad(
        # 输入参数：已编码输入的字典或批量编码对象，类型可以是 EncodedInput 字典或 BatchEncoding 对象
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        # 最大长度参数，表示填充后的序列的最大长度，默认为 None
        max_length: Optional[int] = None,
        # 填充策略参数，指定如何进行填充，默认为不进行填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 填充到的长度的倍数参数，指定填充后序列长度应为多少的倍数，默认为 None
        pad_to_multiple_of: Optional[int] = None,
        # 是否返回注意力掩码参数，指定是否返回注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        # 如果未指定 return_attention_mask 参数，则根据模型输入名称判断是否需要返回 attention_mask
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # 获取必需的模型输入
        required_input = encoded_inputs[self.model_input_names[0]]

        # 如果 padding_strategy 为 LONGEST，则将 max_length 设置为必需输入的长度
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        # 如果 max_length 和 pad_to_multiple_of 都有值，并且 (max_length % pad_to_multiple_of) 不为0，则重新计算 max_length
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # 根据 padding_strategy 和需要填充的长度确定是否需要填充
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # 如果需要返回 attention_mask 且 encoded_inputs 中没有 attention_mask，则初始化 attention_mask
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        # 如果需要填充
        if needs_to_be_padded:
            # 计算需要填充的长度
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                # 在右侧填充
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
            elif self.padding_side == "left":
                # 在左侧填充
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
                # 抛出错误，无效的填充策略
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        # 返回经填充后的 encoded_inputs 字典
        return encoded_inputs
    # 定义一个方法用于批量编码文本或文本对，接受多种输入格式的列表作为参数
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[  # 输入参数可以是文本列表、文本对列表、预标记输入列表等多种格式的联合类型
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认为不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认为不截断
        max_length: Optional[int] = None,  # 最大长度限制，默认为None
        stride: int = 0,  # 步长，默认为0
        is_split_into_words: bool = False,  # 输入是否已经分词，默认为False
        pad_to_multiple_of: Optional[int] = None,  # 填充到的长度的倍数，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型，默认为None
        return_token_type_ids: Optional[bool] = None,  # 是否返回标记类型的ID，默认为None
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码，默认为None
        return_overflowing_tokens: bool = False,  # 是否返回溢出的标记，默认为False
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码，默认为False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为False
        return_length: bool = False,  # 是否返回长度，默认为False
        verbose: bool = True,  # 是否显示详细信息，默认为True
        **kwargs,  # 其他关键字参数
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)  # 添加文档字符串的装饰器，说明编码方法的其他关键字参数
    # 方法用于为模型准备批量数据，接受预标记输入对的列表作为参数
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],  # 输入对的ID列表
        batch_shape_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],  # 形状ID对的列表
        batch_pronunciation_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],  # 发音ID对的列表
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认为不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认为不截断
        max_length: Optional[int] = None,  # 最大长度限制，默认为None
        stride: int = 0,  # 步长，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 填充到的长度的倍数，默认为None
        return_tensors: Optional[str] = None,  # 返回的张量类型，默认为None
        return_token_type_ids: Optional[bool] = None,  # 是否返回标记类型的ID，默认为None
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码，默认为None
        return_overflowing_tokens: bool = False,  # 是否返回溢出的标记，默认为False
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码，默认为False
        return_length: bool = False,  # 是否返回长度，默认为False
        verbose: bool = True,  # 是否显示详细信息，默认为True
        ) -> BatchEncoding:
        """
        准备一个输入 ID 序列，或者一对输入 ID 序列，以便模型使用。它添加特殊标记，根据特殊标记截断序列并在考虑特殊标记的情况下管理溢出令牌的移动窗口（具有用户定义的跨度）。

        Args:
            batch_ids_pairs: tokenized 输入 IDs 列表或输入 IDs 对
            batch_shape_ids_pairs: tokenized 输入 shape IDs 列表或输入 shape IDs 对
            batch_pronunciation_ids_pairs: tokenized 输入 pronunciation IDs 列表或输入 pronunciation IDs 对
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
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 我们之后会在批处理中进行填充
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # 我们之后会在批处理中进行填充
                return_attention_mask=False,  # 我们之后会在批处理中进行填充
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # 我们最后将整个批次转换为张量
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
        """使用词汇表将令牌（str）转换为 ID。"""
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    # 将 token 转换为 shape_id，使用 shape 词汇表
    def _convert_token_to_shape_id(self, token):
        """Converts a token (str) in an shape_id using the shape vocab."""
        return self.word_shape.get(token, self.word_shape.get(self.unk_token))

    # 将 tokens 序列转换为对应的 shape_ids 序列
    def convert_tokens_to_shape_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_shape_id(token))
        return ids

    # 将 token 转换为 pronunciation_id，使用 pronunciation 词汇表
    def _convert_token_to_pronunciation_id(self, token):
        """Converts a token (str) in an shape_id using the shape vocab."""
        return self.word_pronunciation.get(token, self.word_pronunciation.get(self.unk_token))

    # 将 tokens 序列转换为对应的 pronunciation_ids 序列
    def convert_tokens_to_pronunciation_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_pronunciation_id(token))
        return ids

    # 从索引 (index) 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将 tokens 序列转换为单个字符串，去除特殊标记 "##"
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建带有特殊标记的输入序列，用于序列分类任务
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
        # 如果未提供 cls_token_id 和 sep_token_id，则使用默认值
        cls = [self.cls_token_id] if cls_token_id is None else [cls_token_id]
        sep = [self.sep_token_id] if sep_token_id is None else [sep_token_id]
        # 如果仅有一个 token_ids，则返回 cls + token_ids_0 + sep
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        # 如果有两个 token_ids，则返回 cls + token_ids_0 + sep + token_ids_1 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取特殊标记的掩码
    # (此处的注释因为代码中断，所以无法提供完整的功能说明)
```py   
    # 获取没有添加特殊符号的 token 列表的特殊符号 mask。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊符号时会调用此方法。
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
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # 从给定的 token 序列中创建 token 类型 ID 的方法，用于用于序列对分类任务。BERT 序列对 mask 的格式如下：
    # 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    # | first sequence    | second sequence |
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

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
    # 保存词汇表到指定目录，并返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str, str]:
        # 初始化索引
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"],
            )
            # 构建词形文件路径
            word_shape_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["word_shape_file"],
            )
            # 构建发音文件路径
            word_pronunciation_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["word_pronunciation_file"],
            )
        else:
            # 如果保存目录不存在，引发 ValueError
            raise ValueError(
                f"Can't find a directory at path '{save_directory}'. To load the vocabulary from a Google "
                "pretrained model use `tokenizer = RoCBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        # 打开词汇表文件并写入词汇
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的 token 和其对应的索引，按索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 检查索引是否连续，如果不连续则发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入 token 到文件
                writer.write(token + "\n")
                index += 1

        # 打开词形文件并写入词形字典
        with open(word_shape_file, "w", encoding="utf8") as writer:
            json.dump(self.word_shape, writer, ensure_ascii=False, indent=4, separators=(", ", ": "))

        # 打开发音文件并写入发音字典
        with open(word_pronunciation_file, "w", encoding="utf8") as writer:
            json.dump(self.word_pronunciation, writer, ensure_ascii=False, indent=4, separators=(", ", ": "))

        # 返回保存的文件路径
        return (
            vocab_file,
            word_shape_file,
            word_pronunciation_file,
        )
# 从 transformers.models.bert.tokenization_bert.BasicTokenizer 复制而来，创建 RoCBertBasicTokenizer 类
class RoCBertBasicTokenizer(object):
    """
    构造一个 RoCBertBasicTokenizer，执行基本的分词（标点符号分割、小写转换等）。

    Args:
        do_lower_case (`bool`, *可选*, 默认为 `True`):
            在分词时是否将输入转换为小写。
        never_split (`Iterable`, *可选*):
            在分词时永远不会被拆分的标记集合。只在 `do_basic_tokenize=True` 时起作用。
        tokenize_chinese_chars (`bool`, *可选*, 默认为 `True`):
            是否对中文字符进行分词。

            对于日文，这可能应该关闭（参见此处
            [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *可选*):
            是否去除所有重音符号。如果未指定此选项，则将根据 `lowercase` 的值来确定（与原始的 BERT 一样）。
        do_split_on_punc (`bool`, *可选*, 默认为 `True`):
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
        # 如果 never_split 为 None，则设为空列表
        if never_split is None:
            never_split = []
        # 将输入参数保存为对象的属性
        self.do_lower_case = do_lower_case
        # 将 never_split 转换为集合，用于快速查找
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    # 定义一个方法用于对文本进行基本的分词。若需要子词分词，请参考 WordPieceTokenizer。
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果存在 never_split 参数，则将其与类属性 never_split 合并
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本
        text = self._clean_text(text)

        # 添加于2018年11月1日，用于多语言和中文模型。
        # 现在也应用于英语模型，但这并不重要，因为英语模型没有在任何中文数据上进行训练，
        # 通常也不包含任何中文数据（英文维基百科中有一些中文单词，因此词汇表中有中文字符）。
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 将文本中的 Unicode 规范化为 NFC 形式
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白符进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历分词后的原始 token
        for token in orig_tokens:
            # 如果 token 不在不分割列表中
            if token not in never_split:
                # 如果设置了小写化参数
                if self.do_lower_case:
                    # 将 token 转换为小写
                    token = token.lower()
                    # 如果启用了去除重音的功能
                    if self.strip_accents is not False:
                        # 对 token 执行去除重音
                        token = self._run_strip_accents(token)
                # 如果启用了去除重音的功能
                elif self.strip_accents:
                    # 对 token 执行去除重音
                    token = self._run_strip_accents(token)
            # 将处理后的 token 添加到分割后的 token 列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空白符重新分割 token 并返回结果
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 从文本中去除重音
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本中的字符标准化为 NFD 形式
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的字符
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符为 Mn 类型（非重音标记）
            if cat == "Mn":
                # 跳过该字符
                continue
            # 将字符添加到输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串并返回
        return "".join(output)
```  
    # 对给定的文本进行标点符号切割，返回切割后的列表
    def _run_split_on_punc(self, text, never_split=None):
        # 如果不需要对标点符号进行切割，或者文本在 never_split 列表中，则直接返回包含原文本的列表
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
            # 如果当前字符是标点符号
            if _is_punctuation(char):
                # 将其添加到输出列表中，并标记开始新单词
                output.append([char])
                start_new_word = True
            else:
                # 如果是非标点符号字符
                if start_new_word:
                    # 添加一个新的空列表到输出列表
                    output.append([])
                # 标记不是开始新单词
                start_new_word = False
                # 将当前字符添加到最后一个输出列表中
                output[-1].append(char)
            i += 1
        
        # 将输出列表中的每个子列表连接成字符串，返回结果
        return ["".join(x) for x in output]
    
    # 给中文字符添加空格
    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果是中文字符
            if self._is_chinese_char(cp):
                # 在字符前后添加空格
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 否则直接添加字符
                output.append(char)
        return "".join(output)
    
    # 判断给定的 Unicode 码点是否是中文字符
    def _is_chinese_char(self, cp):
        # 检查码点是否在 CJK Unified Ideographs 区域
        return (
            (cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)
        )
    
    # 清理文本，移除无效字符并清理空白字符
    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            # 跳过 0, 0xFFFD 和控制字符
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 将空白字符替换为空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer中复制代码，将WordpieceTokenizer更名为RoCBertWordpieceTokenizer
class RoCBertWordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化方法，接受词汇表、未知标记和每个单词的最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将文本标记化为它的单词片段。这使用贪婪的最长匹配算法，使用给定的词汇表执行标记化。

        例如，`input = "unaffable"`将作为输出返回`["un", "##aff", "##able"]`。

        Args:
            text: 单个标记或以空格分隔的标记。这应该已经通过*BasicTokenizer*。

        Returns:
            一个单词片段标记列表。
        """

        output_tokens = []
        for token in whitespace_tokenize(text):  # 使用whitespace_tokenize对文本进行标记化
            chars = list(token)  # 将标记转换为字符列表
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)  # 如果字符数超过单词的最大输入字符数，则添加未知标记
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])  # 将字符列表[start:end]连接成子串
                    if start > 0:
                        substr = "##" + substr  # 如果起始位置大于0，则在子串前加上"##"
                    if substr in self.vocab:  # 如果子串在词汇表中
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:  # 如果找不到匹配的子串
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)  # 如果存在无法匹配的子串，则添加未知标记
            else:
                output_tokens.extend(sub_tokens)  # 否则添加匹配的子串列表
        return output_tokens
```