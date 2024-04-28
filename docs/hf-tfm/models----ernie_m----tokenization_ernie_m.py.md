# `.\models\ernie_m\tokenization_ernie_m.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证版本 2.0 使用该文件，详见 http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面协议，分发的软件基于"AS IS"基础分发，没有任何明示或暗示的保证或条件
# 请查阅许可证以获取更多信息
"""Ernie-M 的分词器类。"""

import io
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取 logger
logger = logging.get_logger(__name__)

# 定义分词用的下划线符号
SPIECE_UNDERLINE = "▁"

# 定义词汇表文件名和句子片段模型检查点文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "sentencepiece_model_ckpt": "sentencepiece.bpe.model"}

# 定义资源文件名，包括句子片段模型文件和词汇表文件
RESOURCE_FILES_NAMES = {
    "sentencepiece_model_file": "sentencepiece.bpe.model",
    "vocab_file": "vocab.txt",
}

# 预训练词汇表映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "ernie-m-base": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/vocab.txt",
        "ernie-m-large": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/vocab.txt",
    },
    "sentencepiece_model_file": {
        "ernie-m-base": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/sentencepiece.bpe.model",
        "ernie-m-large": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/sentencepiece.bpe.model",
    },
}

# 预训练定位嵌入维度
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "ernie-m-base": 514,
    "ernie-m-large": 514,
}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "ernie-m-base": {"do_lower_case": False},
    "ernie-m-large": {"do_lower_case": False},
}

# 从 paddlenlp.transformers.ernie_m.tokenizer.ErnieMTokenizer 调整的代码
class ErnieMTokenizer(PreTrainedTokenizer):
    r"""
    构建一个 Ernie-M 分词器。使用 sentencepiece 工具将单词切分为子词。
    Args:
        sentencepiece_model_file (`str`):
            The file path of sentencepiece model.
        vocab_file (`str`, *optional*):
            The file path of the vocabulary.
        do_lower_case (`str`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            A special token representing the `unknown (out-of-vocabulary)` token. An unknown token is set to be
            `unk_token` inorder to be converted to an ID.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            A special token separating two different sentences in the same input.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            A special token used to make arrays of tokens the same size for batching purposes.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            A special token used for sequence classification. It is the last token of the sequence when built with
            special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            A special token representing a masked token. This is the token used in the masked language modeling task
            which the model tries to predict the original unmasked ones.
    """

    # Ernie-M model doesn't have token_type embedding.
    model_input_names: List[str] = ["input_ids"]  # 定义了模型输入的名称列表，此模型没有 token_type embedding。

    vocab_files_names = VOCAB_FILES_NAMES  # 词汇表文件名列表，从全局变量 VOCAB_FILES_NAMES 中获取。
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION  # 预训练模型的初始化配置，从全局变量 PRETRAINED_INIT_CONFIGURATION 中获取。
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练模型的最大输入尺寸，从全局变量 PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 中获取。
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练模型的词汇表文件映射，从全局变量 PRETRAINED_VOCAB_FILES_MAP 中获取。
    resource_files_names = RESOURCE_FILES_NAMES  # 资源文件名列表，从全局变量 RESOURCE_FILES_NAMES 中获取。

    def __init__(
        self,
        sentencepiece_model_ckpt,
        vocab_file=None,
        do_lower_case=False,
        encoding="utf8",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 注释描述标记符号行为，表现得如同普通词汇，包括前面的空格，且包含在原始文本中，在非标准化句子中应有对应的匹配。

        # 如果提供的sp_model_kwargs为空，则使用空字典，否则使用传入的sp_model_kwargs
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置是否将文本转为小写的标志
        self.do_lower_case = do_lower_case
        # 存储SentencePiece模型的检查点
        self.sentencepiece_model_ckpt = sentencepiece_model_ckpt
        # 创建一个SentencePiece处理器实例
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载指定的SentencePiece模型
        self.sp_model.Load(sentencepiece_model_ckpt)

        # 如果vocab_file不为空，加载词汇表；否则从SentencePiece模型中生成词汇表
        if vocab_file is not None:
            self.vocab = self.load_vocab(filepath=vocab_file)
        else:
            self.vocab = {self.sp_model.id_to_piece(id): id for id in range(self.sp_model.get_piece_size())}
        # 创建从词汇ID到词汇的反向映射
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # 调用基类的构造器，初始化基础配置
        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            vocab_file=vocab_file,
            encoding=encoding,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def get_offset_mapping(self, text):
        # 如果输入文本为空，返回None
        if text is None:
            return None

        # 对文本进行分词
        split_tokens = self.tokenize(text)
        normalized_text, char_mapping = "", []

        # 遍历原始文本的每个字符，进行字符映射和正规化处理
        for i, ch in enumerate(text):
            # 如果字符在特定映射中，替换之
            if ch in self.SP_CHAR_MAPPING:
                ch = self.SP_CHAR_MAPPING.get(ch)
            else:
                # 将字符正规化为NFKC形式
                ch = unicodedata.normalize("NFKC", ch)
            # 忽略空白字符
            if self.is_whitespace(ch):
                continue
            # 累加处理后的字符到新的文本中
            normalized_text += ch
            # 累加字符索引映射
            char_mapping.extend([i] * len(ch))

        # 初始化文本分析结果
        text, token_mapping, offset = normalized_text, [], 0

        # 如果设置为小写，则将整个文本转为小写
        if self.do_lower_case:
            text = text.lower()

        # 遍历分词结果，计算每个token在原文中的映射范围
        for token in split_tokens:
            # 如果token以特定前缀开始，移除该前缀
            if token[:1] == "▁":
                token = token[1:]
            # 找到token在处理后文本中的起始位置
            start = text[offset:].index(token) + offset
            # 计算token的结束位置
            end = start + len(token)

            # 记录token在原文中的字符范围
            token_mapping.append((char_mapping[start], char_mapping[end - 1] + 1))
            # 更新偏移量
            offset = end
        # 返回处理结果
        return token_mapping

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.vocab)

    def get_vocab(self):
        # 返回当前词汇表，包括动态添加的token
        return dict(self.vocab, **self.added_tokens_encoder)

    def __getstate__(self):
        # 获取对象的状态，用于序列化
        state = self.__dict__.copy()
        # 移除sp_model以避免序列化错误
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 反序列化时设置对象状态
        self.__dict__ = d

        # 为了向后兼容
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新创建和加载SentencePiece处理器
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.sentencepiece_model_ckpt)
    def clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        # 对文本进行非法字符移除和空格清理
        return "".join((self.SP_CHAR_MAPPING.get(c, c) for c in text))

    def _tokenize(self, text, enable_sampling=False, nbest_size=64, alpha=0.1):
        """Tokenize a string."""
        # 设置 tokenization 的参数，如是否启用采样、nbest_size 和 alpha
        if self.sp_model_kwargs.get("enable_sampling") is True:
            enable_sampling = True
        if self.sp_model_kwargs.get("alpha") is not None:
            alpha = self.sp_model_kwargs.get("alpha")
        if self.sp_model_kwargs.get("nbest_size") is not None:
            nbest_size = self.sp_model_kwargs.get("nbest_size")

        # 根据参数对文本进行分词处理
        if not enable_sampling:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, nbest_size, alpha)
        new_pieces = []
        for pi, piece in enumerate(pieces):
            if piece == SPIECE_UNDERLINE:
                if not pieces[pi + 1].startswith(SPIECE_UNDERLINE) and pi != 0:
                    new_pieces.append(SPIECE_UNDERLINE)
                    continue
                else:
                    continue
            lst_i = 0
            for i, chunk in enumerate(piece):
                if chunk == SPIECE_UNDERLINE:
                    continue
                if self.is_ch_char(chunk) or self.is_punct(chunk):
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    new_pieces.append(chunk)
                    lst_i = i + 1
                elif chunk.isdigit() and i > 0 and not piece[i - 1].isdigit():
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    lst_i = i
                elif not chunk.isdigit() and i > 0 and piece[i - 1].isdigit():
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    lst_i = i
            if len(piece) > lst_i:
                new_pieces.append(piece[lst_i:])
        return new_pieces

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        # 将 tokens 转换成一个字符串
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def convert_ids_to_string(self, ids):
        """
        Converts a sequence of tokens (strings for sub-words) in a single string.
        """
        # 将 token ID 序列转换成一个字符串
        tokens = self.convert_ids_to_tokens(ids)
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # to mimic paddlenlp.transformers.ernie_m.tokenizer.ErnieMTokenizer functioning
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # to mimic paddlenlp.transformers.ernie_m.tokenizer.ErnieMTokenizer functioning
    # 将索引（整数）转换为标记（字符串），使用词汇表进行转换
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.reverse_vocab.get(index, self.unk_token)

    # 为序列分类任务构建模型输入，包括连接和添加特殊标记
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An ErnieM sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of input_id with the appropriate special tokens.
        """
        # 如果只有一个序列，则返回带有特殊标记的输入列表
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        # 否则，返回带有特殊标记的输入列表（对于序列对）
        return _cls + token_ids_0 + _sep + _sep + token_ids_1 + _sep

    # 为偏移映射对构建偏移映射，通过连接和添加特殊标记的偏移来实现
    def build_offset_mapping_with_special_tokens(self, offset_mapping_0, offset_mapping_1=None):
        r"""
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens. An Ernie-M
        offset_mapping has the following format:

        - single sequence: `(0,0) X (0,0)`
        - pair of sequences: `(0,0) A (0,0) (0,0) B (0,0)`

        Args:
            offset_mapping_ids_0 (`List[tuple]`):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (`List[tuple]`, *optional*):
                Optional second list of wordpiece offsets for offset mapping pairs.
        Returns:
            `List[tuple]`: List of wordpiece offsets with the appropriate offsets of special tokens.
        """
        # 如果只有一个偏移映射，则返回带有特殊标记偏移的偏移映射列表
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        # 否则，返回带有特殊标记偏移的偏移映射列表（对于偏移映射对）
        return [(0, 0)] + offset_mapping_0 + [(0, 0), (0, 0)] + offset_mapping_1 + [(0, 0)]
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        r"""
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `encode` method.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`str`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`:
                The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        # If the token list already has special tokens added
        if already_has_special_tokens:
            # Raise an error if a second sequence is provided, as it shouldn't be provided in this case
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            # Generate a mask where special tokens are marked as 1 and sequence tokens as 0
            return [1 if x in [self.sep_token_id, self.cls_token_id] else 0 for x in token_ids_0]

        # If the token list does not have special tokens added
        if token_ids_1 is not None:
            # Generate a mask for sequence pairs with special tokens included
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]
        # Generate a mask for a single sequence with special tokens included
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids) Should be overridden in a subclass if the model has a special way of
        building: those.

        Args:
            token_ids_0 (`List[int]`):
                The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*):
                The second tokenized sequence.
        Returns:
            `List[int]`: The token type ids.
        """
        # Called when `add_special_tokens` is True, so align with `build_inputs_with_special_tokens` method
        if token_ids_1 is None:
            # For a single sequence, return token type ids where all tokens are marked as belonging to the same segment
            return (len(token_ids_0) + 2) * [0]

        # For sequence pairs, return token type ids where tokens of the first sequence are marked as 0 and tokens of the second sequence are marked as 1
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 3)

    def is_ch_char(self, char):
        """
        is_ch_char
        """
        # Check if the character is a Chinese character
        if "\u4e00" <= char <= "\u9fff":
            return True
        return False

    def is_alpha(self, char):
        """
        is_alpha
        """
        # Check if the character is an alphabetical character
        if ("a" <= char <= "z") or ("A" <= char <= "Z"):
            return True
        return False

    def is_punct(self, char):
        """
        is_punct
        """
        # Check if the character is a punctuation mark
        if char in ",;:.?!~，；：。？！《》【】":
            return True
        return False
    # 判断一个字符是否为空白字符
    def is_whitespace(self, char):
        """
        is whitespace
        """
        # 如果字符是空格、制表符、换行符、回车符中的一个，则返回 True
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        # 如果字符长度为1，则判断其 Unicode 分类是否为 Zs（空白分隔符）,是则返回 True
        if len(char) == 1:
            cat = unicodedata.category(char)
            if cat == "Zs":
                return True
        # 其它情况返回 False
        return False

    # 加载词汇表
    def load_vocab(self, filepath):
        # 初始化一个空字典
        token_to_idx = {}
        # 打开文件并循环读取每一行
        with io.open(filepath, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                # 去除行末的换行符，得到 token
                token = line.rstrip("\n")
                # 将 token 和它的索引存入字典
                token_to_idx[token] = int(index)
        # 返回装有 token 到索引的字典
        return token_to_idx

    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引为0
        index = 0
        # 如果保存目录已经存在
        if os.path.isdir(save_directory):
            # 构造词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 构造词汇表文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件并写入
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历排序后的词汇表字典，按照索引顺序写入词汇和索引
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1

        # 构造 tokenizer 模型文件路径
        tokenizer_model_file = os.path.join(save_directory, "sentencepiece.bpe.model")
        # 打开文件并写入内容
        with open(tokenizer_model_file, "wb") as fi:
            content_spiece_model = self.sp_model.serialized_model_proto()
            fi.write(content_spiece_model)

        # 返回保存的文件路径
        return (vocab_file,)
```