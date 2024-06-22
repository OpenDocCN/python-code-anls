# `.\models\gptsan_japanese\tokenization_gptsan_japanese.py`

```py
# 设置编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可证版本 2.0 许可
# 仅在符合许可证的情况下使用此文件
# 您可以获取许可证的副本
# 在这个网址：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件是基于"现状"提供的
# 没有任何种类的担保或条件，无论是明示的还是暗示的
# 请阅读许可证以了解特定语言上的许可证要求和限制
"""GPTSANJapanese"的标记类
导入所需的依赖库并设置一些常量
"""
import collections  # 导入collections模块，用来创建有序字典等数据结构
import json  # 导入json模块，用来处理JSON数据
import os  # 导入os模块，用来进行系统操作
import re  # 导入re模块，用来进行正则表达式匹配
from typing import List, Optional, Tuple, Union  # 导入类型提示模块，用于类型注解

import numpy as np  # 导入numpy库，用于数值计算

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练的Tokenizer类
from ...tokenization_utils_base import (  # 导入基础的Tokenizer类和相关类型
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import PaddingStrategy, logging  # 导入Padding策略和日志记录功能

# 获取logger对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义词汇表文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "emoji_file": "emoji.json"}

# 定义预训练词汇表文件映射关系
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Tanrei/GPTSAN-japanese": "https://huggingface.co/Tanrei/GPTSAN-japanese/blob/main/vocab.txt",
    },
    "emoji_file": {
        "Tanrei/GPTSAN-japanese": "https://huggingface.co/Tanrei/GPTSAN-japanese/blob/main/emoji.json",
    },
}

# 定义预训练位置嵌入的尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Tanrei/GPTSAN-japanese": 1280,
}


def load_vocab_and_emoji(vocab_file, emoji_file):
    """加载词汇表文件和表情符号文件到字典中"""
    # 从emoji文件中读取表情符号并转为字典
    with open(emoji_file, "r", encoding="utf-8") as f:
        emoji = json.loads(f.read())

    # 初始化词汇表和原始词汇表等字典
    vocab = collections.OrderedDict()
    raw_vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as f:
        token = f.readlines()
    # 将词汇表文件拆分为单词列表
    token = [
        [t.rstrip("\n")] if (t == ",\n" or "," not in t) else t.rstrip("\n").split(",") for t in token
    ]
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b
        raw_vocab[",".join(b)] = idx
        for wd in b:
            vocab[wd] = idx

    return vocab, raw_vocab, ids_to_tokens, emoji


class GPTSanJapaneseTokenizer(PreTrainedTokenizer):
    """
    这个标记器基于GPTNeoXJapaneseTokenizer，并进行了以下修改
    - 正确解码字节0~字节255标记
    - 增加bagofword标记处理
    - 为前缀语言模型返回token_type_ids
    bagofword标记表示重复前一个标记，并在解码时转换为3个连续标记
    此外，原始的日文特殊子词编码已发布在此存储库中
    (https://github.com/tanreinama/Japanese-BPEEncoder_V2)。token_type_ids是指示前缀输入的掩码
    # 设置 Prefix-LM 模型的前缀位置。要指定前缀位置，为 prefix_text 指定前缀输入，或者指定前缀部分和后面部分的文本对作为批量输入
    position of the Prefix-LM model. To specify a prefix position, specify a prefix input for prefix_text, or specify a sentence of the prefix part and the part after it as a text pair.

    示例:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> # You can confirm both 慶応 and 慶應 are encoded to 17750
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [35993, 35998, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Both 慶応 and 慶應 are decoded to 慶応
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    ```py

    Prefix-LM 示例:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["input_ids"]
    [35993, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 35998, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["token_type_ids"]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ```py

    批量编码 示例:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["input_ids"]
    [[35993, 8640, 25948, 35998, 30647, 35675, 35999, 35999], [35993, 10382, 9868, 35998, 30646, 9459, 30646, 35675]]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["token_type_ids"]
    [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]

    >>> # Mask for padding
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["attention_mask"]
    [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
    ```py
    Args:
        vocab_file (`str`):
            包含词汇表的文件。
        emoji_file (`str`):
            包含表情符号的文件。
        unk_token (`str`, *optional*, 默认为 `"<|nottoken|>"`):
            用于未知字符的标记
        pad_token (`str`, *optional*, 默认为 `"<|separator|>"`):
            用于填充的标记
        bos_token (`str`, *optional*, 默认为 `"<|startoftext|>"`):
            序列开始的标记。
        eos_token (`str`, *optional*, 默认为 `"<|endoftext|>"`):
            序列结束的标记。
        sep_token (`str`, *optional*, 默认为 `"<|segmenter|>"`):
            用于分隔前缀部分和一般输入部分的特殊标记。
        do_clean_text (`bool`, *optional*, 默认为 `False`):
            是否对 URL、邮箱、电话、日文日期和日文价格进行文本清洗。

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __init__(
        self,
        vocab_file,
        emoji_file,
        unk_token="<|nottoken|>",
        pad_token="<|separator|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        sep_token="<|segmenter|>",
        do_clean_text=False,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = GPTSanJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        if not os.path.isfile(emoji_file):
            raise ValueError(
                f"Can't find a emoji file at path '{emoji_file}'. To load the emoji information from a Google"
                " pretrained model use `tokenizer = GPTSanJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_clean_text = do_clean_text
        self.vocab, self.raw_vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        self.subword_tokenizer = SubWordJapaneseTokenizer(
            vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji
        )

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            do_clean_text=do_clean_text,
            **kwargs,
        )

    @property
    # 从 tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizer.vocab_size 复制而来
    def vocab_size(self):
        # self.vocab 包含了对日语特有字符波动的支持，并且具有大量的词汇量
        return len(self.raw_vocab)
    # 从 tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizer.get_vocab 复制的方法，返回词汇表（字典形式）
    def get_vocab(self):
        return dict(self.raw_vocab, **self.added_tokens_encoder)

    # 从 tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizer._tokenize 复制的方法，将文本分词
    def _tokenize(self, text):
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    # 从 tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizer._convert_token_to_id 复制的方法，将 token 转换为 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 从 tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizer._convert_id_to_token 复制的方法，将 id 转换为 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.subword_tokenizer.convert_id_to_token(index)

    # 将 tokens 序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        words = []
        byte_tokens = []
        for word in tokens:
            # 处理字节标记
            if word[:6] == "<|byte" and word[-2:] == "|>":
                byte_tokens.append(int(word[6:-2]))
            else:
                # 将 byte_tokens 转换为字符添加到结果中
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                    byte_tokens = []
                # 处理特殊的单词标记和条件
                # ...
                else:
                    words.append(word)
        # 处理剩余的字节标记
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        text = "".join(words)
        return text

    @property
    def default_chat_template(self):
        """
        A simple chat template that adds standard BOS, SEP and EOS tokens between messages while discarding role
        information.
        """
        # 如果没有为此分词器定义聊天模板，则使用默认模板，并发出警告消息
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板
        return (
            "{% for message in messages %}"
            "{% if not loop.first %}{{ bos_token}}{% endif %}"
            "{{ sep_token }}{{ message.content }} {{ eos_token }}"
            "{% endfor %}"
        )

    # 从 tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizer.save_vocabulary 复制而来
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
            # 构建表情符号文件路径
            emoji_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["emoji_file"]
            )
        else:
            # 构建词汇表文件路径
            vocab_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["vocab_file"]
            )
            # 构建表情符号文件路径
            emoji_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["emoji_file"]
            )
        # 写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token_index, token in self.ids_to_tokens.items():
                if index != token_index:
                    # 如果词汇表索引不是连续的，则发出警告消息
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(",".join(token) + "\n")
                index += 1
        # 写入表情符号文件
        with open(emoji_file, "w", encoding="utf-8") as writer:
            json.dump(self.emoji, writer)
        # 返回词汇表文件路径和表情符号文件路径
        return vocab_file, emoji_file

    # 从序列创建 token_type_ids
    # 将 token_ids_0 和 token_ids_1 转换为 token_type_ids
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # 标记类型 ID 作为前缀部分和其余部分之间的分隔符
        # 前缀部分的 token_type_ids 为 1，其余部分为 0

        # 示例:
        ```python
        >>> from transformers import GPTSanJapaneseTokenizer

        >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("ｱｲｳｴ")
        >>> # input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |

        >>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
        >>> # input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
        >>> # token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |

        >>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
        >>> # input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
        ```py"""
        # 计算前缀长度
        prefix_len = 0
        if self.sep_token in self.vocab:  # 如果分隔符在词汇表中
            segid = self.vocab[self.sep_token]  # 获取分隔符的 ID
            if segid in token_ids_0:  # 如果分隔符 ID 存在于 token_ids_0 中
                prefix_len = token_ids_0.index(segid)  # 计算前缀长度为分隔符之前的 token 数量
        if token_ids_1 is None:  # 如果 token_ids_1 为空
            total_len = len(token_ids_0)  # 计算总长度为 token_ids_0 的长度
        else:  # 如果 token_ids_1 不为空
            total_len = len(token_ids_0 + token_ids_1)  # 计算总长度为两个 token_ids 的长度之和
        return prefix_len * [1] + (total_len - prefix_len) * [0]  # 返回前缀部分的 token_type_ids

    def prepare_for_tokenization(self, text, prefix_text=None, add_sep_token=None, **kwargs):
        # GPTSAN 在文本生成的前缀语言模型中除了 SOT 之外还会插入额外的 SEP token
        # 文本开头为 SOT，前缀部分和其余部分之间为 SEP

        if add_sep_token is None:
            add_sep_token = self.sep_token not in text  # 如果在非前缀位置显式插入 SEP token
        prepared = self.bos_token if self.bos_token in self.vocab else ""  # 如果 BOS token 在词汇表中，则加入到 prepared 中
        prepared += prefix_text if prefix_text is not None else ""  # 加入前缀文本
        if add_sep_token:  # 如果需要插入 SEP token
            prepared += self.sep_token if self.sep_token in self.vocab else ""  # 如果 SEP token 在词汇表中，则加入到 prepared 中
        prepared += text  # 加入文本
        return (prepared, kwargs)  # 返回准备好的文本和额外的参数
    # 定义一个方法用于批量编码文本或文本对
    def _batch_encode_plus(
        self,
        # 输入参数可以是文本列表、文本对列表、预分词输入列表或预分词输入对列表的联合类型
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        # 是否添加特殊标记，默认为 True
        add_special_tokens: bool = True,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认为不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制，默认为 None
        max_length: Optional[int] = None,
        # 步长，默认为 0
        stride: int = 0,
        # 是否已分词，默认为 False
        is_split_into_words: bool = False,
        # 填充到指定长度的倍数，默认为 None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为 None
        return_tensors: Optional[str] = None,
        # 是否返回标记类型的张量，默认为 None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的标记，默认为 False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记的掩码，默认为 False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为 False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为 False
        return_length: bool = False,
        # 是否显示详细信息，默认为 True
        verbose: bool = True,
    ) -> BatchEncoding:
        # 这个分词器将输入文本对转换为前缀输入和随后输入
        if isinstance(batch_text_or_text_pairs[0], tuple) or isinstance(tuple(batch_text_or_text_pairs[0]), list):
            # 将每个文本对添加前缀并合并成一个文本列表
            batch_prefix_texts = []
            for pref, txt in batch_text_or_text_pairs:
                batch_prefix_texts.append(pref + self.sep_token + txt)
            batch_text_or_text_pairs = batch_prefix_texts

        # 调用父类的 _batch_encode_plus 方法进行编码处理并返回结果
        return super()._batch_encode_plus(
            batch_text_or_text_pairs,
            add_special_tokens,
            padding_strategy,
            truncation_strategy,
            max_length,
            stride,
            is_split_into_words,
            pad_to_multiple_of,
            return_tensors,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_offsets_mapping,
            return_length,
            verbose,
        )
# 定义一个名为 SubWordJapaneseTokenizer 的类
class SubWordJapaneseTokenizer(object):
    """
    This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications
    - Decoding byte0~byte255 tokens correctly
    - Added bagofword token handling

    https://github.com/tanreinama/Japanese-BPEEncoder_V2 This tokenizer class is under MIT Lisence according to the
    original repository.

    MIT License

    Copyright (c) 2020 tanreinama

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    # Copied from tokenization_gpt_neox_japanese.SubWordJapaneseTokenizer.__init__
    # 初始化函数，设置类的属性
    def __init__(self, vocab, ids_to_tokens, emoji):
        self.vocab = vocab  # 将参数 vocab 赋值给实例属性 vocab
        self.ids_to_tokens = ids_to_tokens  # 将参数 ids_to_tokens 赋值给实例属性 ids_to_tokens
        self.emoji = emoji  # 将参数 emoji 赋值给实例属性 emoji
        self.maxlen = np.max([len(w) for w in self.vocab.keys()])  # 计算实例属性 maxlen 的最大值
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")  # 编译正则表达式
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")  # 编译正则表达式
        self.content_repatter3 = re.compile(r"[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}")  # 编译正则表达式
        self.content_repatter4 = re.compile(
            r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )  # 编译正则表达式
        self.content_repatter5 = re.compile(
            r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )  # 编译正则表达式
        self.content_repatter6 = re.compile(
            r"((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*"
        )  # 编译正则表达式
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k: "<BLOCK>" for k in keisen + blocks})  # 使用 str.maketrans 方法创建替换字符字典

    # 返回 ids_to_tokens 的长度
    def __len__(self):
        return len(self.ids_to_tokens)

    # 清洗文本内容
    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>", content)  # 使用正则表达式替换内容
        content = self.content_repatter2.sub("<EMAIL>", content)  # 使用正则表达式替换内容
        content = self.content_repatter3.sub("<TEL>", content)  # 使用正则表达式替换内容
        content = self.content_repatter4.sub("<DATE>", content)  # 使用正则表达式替换内容
        content = self.content_repatter5.sub("<DATE>", content)  # 使用正则表达式替换内容
        content = self.content_repatter6.sub("<PRICE>", content)  # 使用正则表达式替换内容
        content = content.translate(self.content_trans1)  # 使用 maketrans 方法替换内容
        while "<BLOCK><BLOCK>" in content:
            content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")  # 替换内容中的重复标记
        return content
    # 用于将文本进行分词处理，可选择是否进行清洗
    def tokenize(self, text, clean=False):
        # 替换空格为<SP>
        text = text.replace(" ", "<SP>")
        # 替换全角空格为<SP>
        text = text.replace("　", "<SP>")
        # 替换换行符为<BR>
        text = text.replace("\r\n", "<BR>")
        text = text.replace("\n", "<BR>")
        text = text.replace("\r", "<BR>")
        # 替换制表符为<TAB>
        text = text.replace("\t", "<TAB>")
        # 替换特殊破折号为标准破折号
        text = text.replace("—", "ー")
        text = text.replace("−", "ー")
        # 遍历emoji字典，如果文本中存在键对应的内容，则替换为对应的值
        for k, v in self.emoji["emoji"].items():
            if k in text:
                text = text.replace(k, v)
        # 如果需要清洗，则调用clean_text方法进行清洗
        if clean:
            text = self.clean_text(text)

        # 定义检查特殊符号的方法
        def check_simbol(x):
            # 如果字符长度为1且编码长度为2，则进行特殊符号检查
            e = x.encode()
            if len(x) == 1 and len(e) == 2:
                c = (int(e[0]) << 8) + int(e[1])
                # 若符合特殊符号的编码范围，则返回True
                if (
                    (c >= 0xC2A1 and c <= 0xC2BF)
                    or (c >= 0xC780 and c <= 0xC783)
                    or (c >= 0xCAB9 and c <= 0xCBBF)
                    or (c >= 0xCC80 and c <= 0xCDA2)
                ):
                    return True
            return False

        # 定义检查U+2000-U+2BFF编码范围的方法
        def checku2e(x):
            # 如果字符长度为1且编码长度为3，则进行U+2000-U+2BFF编码范围检查
            e = x.encode()
            if len(x) == 1 and len(e) == 3:
                c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
                # 若符合U+2000-U+2BFF编码范围，则返回True
                if c >= 0xE28080 and c <= 0xE2B07F:
                    return True
            return False

        # 初始化位置变量
        pos = 0
        result = []  # 结果存储列表
        # 开始遍历文本进行分词处理
        while pos < len(text):
            # 设定结束位置，如果当前字符为"<"，则结束位置为最大长度+1，否则为当前位置加3
            end = min(len(text), pos + self.maxlen + 1) if text[pos] == "<" else pos + 3
            candidates = []  # 候选词列表，格式为(token_id, token, pos)
            for e in range(end, pos, -1):
                wd = text[pos:e]
                # 如果词在vocab中，则加入候选词列表
                if wd in self.vocab:
                    if wd[0] == "<" and len(wd) > 2:
                        candidates = [(self.vocab[wd], wd, e)]
                        break
                    else:
                        candidates.append((self.vocab[wd], wd, e))
            if len(candidates) > 0:
                # 选择候选词中token_id最小的token，加入结果列表中
                _, wd, e = sorted(candidates, key=lambda x: x[0])[0]
                result.append(wd)
                pos = e
            else:
                # 如果没有找到匹配的词，则截取当前位置到结束位置的字符作为一个token
                end = pos + 1
                wd = text[pos:end]
                # 检查是否为特殊符号，若是则添加"<KIGOU>"到结果列表，若不是则按utf-8编码每一个字节添加到结果列表
                if check_simbol(wd):
                    result.append("<KIGOU>")
                elif checku2e(wd):
                    result.append("<U2000U2BFF>")
                else:
                    for i in wd.encode("utf-8"):
                        result.append("<|byte%d|>" % i)
                pos = end
        return result  # 返回分词后的结果列表

    # 转化token_id为token并返回
    def convert_id_to_token(self, index):
        return self.ids_to_tokens[index][0]
```