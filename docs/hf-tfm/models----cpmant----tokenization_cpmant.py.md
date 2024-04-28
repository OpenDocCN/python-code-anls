# `.\models\cpmant\tokenization_cpmant.py`

```py
# Tokenization classes for CPMAnt: CPMAnt的分词类


import collections
import os
from typing import List, Optional, Tuple

from transformers.utils import is_jieba_available, requires_backends


if is_jieba_available():
    import jieba

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openbmb/cpm-ant-10b": "https://huggingface.co/openbmb/cpm-ant-10b/blob/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openbmb/cpm-ant-10b": 1024,
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class WordpieceTokenizer(object):
    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
            else:
                sub_tokens.append(cur_substr)
                start = end

        return sub_tokens


class CpmAntTokenizer(PreTrainedTokenizer):
    """
    Construct a CPMAnt tokenizer. Based on byte-level Byte-Pair-Encoding.
    """
    # 定义一个类（初始化参数包括词汇文件路径以及可选的特殊标记）
    class SomeClass:
        # 设置一些默认的词汇文件名和映射
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        model_input_names = ["input_ids", "attention_mask"]
        add_prefix_space = False
    
        # 初始化函数，包括词汇文件路径和一些可选特殊标记
        def __init__(
            self,
            vocab_file,
            bod_token="<d>",
            eod_token="</d>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            line_token="</n>",
            space_token="</_>",
            padding_side="left",
            **kwargs,
        ):
            # 使用外部库jieba，确保其可用
            requires_backends(self, ["jieba"])
            self.bod_token = bod_token
            self.eod_token = eod_token
            # 从词汇文件加载编码器
            self.encoder = load_vocab(vocab_file)
            # 将空格和换行符加入编码器中
            self.encoder[" "] = self.encoder[space_token]
            self.encoder["\n"] = self.encoder[line_token]
            # 从编码器中删除空格和换行符
            del self.encoder[space_token]
            del self.encoder[line_token]
            # 对编码器按值进行排序并创建其对应的解码器
            self.encoder = collections.OrderedDict(sorted(self.encoder.items(), key=lambda x: x[1]))
            self.decoder = {v: k for k, v in self.encoder.items()}
            # 使用编码器和未知标记初始化词分词仪
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder, unk_token=unk_token)
            # 调用父类初始化函数
            super().__init__(
                bod_token=bod_token,
                eod_token=eod_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                unk_token=unk_token,
                line_token=line_token,
                space_token=space_token,
                padding_side=padding_side,
                **kwargs,
            )
    
        # 返回开始文档标记的编码
        @property
        def bod_token_id(self):
            return self.encoder[self.bod_token]
    
        # 返回结束文档标记的编码
        @property
        def eod_token_id(self):
            return self.encoder[self.eod_token]
    
        # 返回换行符的编码
        @property
        def newline_id(self):
            return self.encoder["\n"]
    
        # 返回词汇大小
        @property
        def vocab_size(self) -> int:
            return len(self.encoder)
    
        # 获取编码器
        def get_vocab(self):
            return dict(self.encoder, **self.added_tokens_encoder)
    # 将字符串进行分词处理
    def _tokenize(self, text):
        """Tokenize a string."""
        # 初始化一个空列表用于存储分词后的结果
        output_tokens = []
        # 使用结巴分词对字符串进行分词处理
        for x in jieba.cut(text, cut_all=False):
            # 将结巴分词结果逐个进行词片段处理，并将处理后的词片段添加到结果列表中
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        # 返回分词结果列表
        return output_tokens

    # 将 id 序列解码成字符串
    def _decode(self, token_ids, **kwargs):
        """Decode ids into a string."""
        # 过滤掉小于 0 的 token_ids
        token_ids = [i for i in token_ids if i >= 0]
        # 过滤掉特殊标记的 token_ids（pad_token_id, eos_token_id, bos_token_id）
        token_ids = [
            x for x in token_ids if x != self.pad_token_id and x != self.eos_token_id and x != self.bos_token_id
        ]
        # 调用父类的 _decode 方法进行解码，并传入其他方式的参数
        return super()._decode(token_ids, **kwargs)

    # 检查 token 是否在编码器中
    def check(self, token):
        return token in self.encoder

    # 将 tokens 列表转换为字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    # 保存词汇表到文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存的目录已经存在，则生成词汇表文件路径
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:  # 如果保存的目录不存在，则直接使用保存路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 初始化 index 变量
        index = 0
        # 如果空格存在于编码器中，则进行替换操作
        if " " in self.encoder:
            self.encoder["</_>"] = self.encoder[" "]
            del self.encoder[" "]
        # 如果换行存在于编码器中，则进行替换操作
        if "\n" in self.encoder:
            self.encoder["</n>"] = self.encoder["\n"]
            del self.encoder["\n"]
        # 对编码器进行排序
        self.encoder = collections.OrderedDict(sorted(self.encoder.items(), key=lambda x: x[1]))
        # 将编码器中的信息保存到文件中
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in self.encoder.items():
                # 检查词汇表索引是否连续，如果不连续则输出警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    # 更新索引值
                    index = token_index
                # 写入 token 到文件中
                writer.write(token + "\n")
                # 更新索引值
                index += 1
        # 返回保存的词汇表文件路径
        return (vocab_file,)
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
        """
        从一个或一对序列构建用于序列分类任务的模型输入，通过连接和添加特殊标记。一个 CPMAnt 序列的格式如下：

        - 单个序列: `[BOS] Sequence`.

        Args:
            token_ids_0 (`List[int]`): 将要添加特殊标记的第一个标记化序列。
            token_ids_1 (`List[int]`, *可选*): 将要添加特殊标记的可选第二个标记化序列。

        Returns:
            `List[int]`: 带有特殊标记的模型输入。
        """
        # 如果没有第二个序列，则只添加起始特殊标记并返回
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0
        # 否则，连接两个序列并添加起始特殊标记，并返回
        return [self.bos_token_id] + token_ids_0 + [self.bos_token_id] + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。此方法在使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用。

        Args:
            token_ids_0 (`List[int]`): ID 列表。
            token_ids_1 (`List[int]`, *可选*): 序列对的可选第二个 ID 列表。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围在 [0, 1] 内：1 表示特殊标记，0 表示序列标记。
        """

        # 如果已经包含特殊标记，则调用父类的方法返回结果
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果有第二个序列，则返回带有特殊标记的掩码列表
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
        # 否则，返回只包含第一个序列的带有特殊标记的掩码列表
        return [1] + ([0] * len(token_ids_0))
```