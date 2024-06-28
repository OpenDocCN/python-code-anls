# `.\models\gptsan_japanese\tokenization_gptsan_japanese.py`

```
# 指定文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发的软件，
# 没有任何明示或暗示的担保或条件
# 请参阅许可证获取具体语言的权限或限制
"""GPTSANJapanese 的标记化类"""
import collections  # 导入集合模块，用于处理有序字典等
import json  # 导入 JSON 模块，用于处理 JSON 数据
import os  # 导入 OS 模块，用于处理操作系统相关功能
import re  # 导入正则表达式模块，用于字符串匹配操作
from typing import List, Optional, Tuple, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 模块，用于数值计算

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练标记器类
from ...tokenization_utils_base import (  # 导入基础标记化相关模块
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import PaddingStrategy, logging  # 导入填充策略和日志模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "emoji_file": "emoji.json"}  # 定义词汇文件名和表情符号文件名

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Tanrei/GPTSAN-japanese": "https://huggingface.co/Tanrei/GPTSAN-japanese/blob/main/vocab.txt",
    },
    "emoji_file": {
        "Tanrei/GPTSAN-japanese": "https://huggingface.co/Tanrei/GPTSAN-japanese/blob/main/emoji.json",
    },
}  # 预训练词汇文件映射，指定 GPTSAN-japanese 模型的词汇和表情符号文件

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Tanrei/GPTSAN-japanese": 1280,
}  # 预训练位置嵌入尺寸映射，指定 GPTSAN-japanese 模型的位置嵌入尺寸


def load_vocab_and_emoji(vocab_file, emoji_file):
    """加载词汇文件和表情符号文件到字典中。"""
    with open(emoji_file, "r", encoding="utf-8") as f:
        emoji = json.loads(f.read())  # 读取并解析 JSON 格式的表情符号文件内容

    vocab = collections.OrderedDict()  # 创建有序字典用于存储词汇表
    raw_vocab = collections.OrderedDict()  # 创建有序字典用于存储原始词汇表
    ids_to_tokens = collections.OrderedDict()  # 创建有序字典用于存储从索引到标记的映射关系
    with open(vocab_file, "r", encoding="utf-8") as f:
        token = f.readlines()  # 逐行读取词汇文件内容
    token = [[t.rstrip("\n")] if (t == ",\n" or "," not in t) else t.rstrip("\n").split(",") for t in token]  # 对每行进行处理，将其拆分为标记列表
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b  # 将索引与标记映射关系存入字典
        raw_vocab[",".join(b)] = idx  # 将标记列表转换为字符串作为键，索引作为值存入原始词汇表
        for wd in b:
            vocab[wd] = idx  # 将标记与索引的映射关系存入词汇表

    return vocab, raw_vocab, ids_to_tokens, emoji  # 返回词汇表、原始词汇表、索引到标记映射和表情符号字典


class GPTSanJapaneseTokenizer(PreTrainedTokenizer):
    """
    本标记器基于 GPTNeoXJapaneseTokenizer，并进行以下修改：
    - 正确解码字节0~255的标记
    - 添加 bagofword 标记处理
    - 为 Prefix-LM 模型返回 token_type_ids
    bagofword 标记表示前一个标记的重复，并在解码时转换为三个连续的标记
    此外，原始的日本特殊 Sub-Word-Encoding 已在此存储库中发布
    (https://github.com/tanreinama/Japanese-BPEEncoder_V2)。token_type_ids 是一个指示前缀输入的掩码
    """
    pass  # GPTSanJapaneseTokenizer 类目前无具体实现，仅有文档字符串说明其基本功能
    >>> from transformers import GPTSanJapaneseTokenizer
    引入 GPTSanJapaneseTokenizer 类从 transformers 库
    
    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    使用预训练模型 "Tanrei/GPTSAN-japanese" 初始化一个 tokenizer 对象
    
    >>> # You can confirm both 慶応 and 慶應 are encoded to 17750
    # 使用 tokenizer 对字符串进行编码，返回输入文本的 token IDs 列表
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [35993, 35998, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]
    
    >>> # Both 慶応 and 慶應 are decoded to 慶応
    # 使用 tokenizer 对 token IDs 进行解码，返回原始文本
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    
    
    
    
    Example for Prefix-LM:
    
    >>> from transformers import GPTSanJapaneseTokenizer
    引入 GPTSanJapaneseTokenizer 类从 transformers 库
    
    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    使用预训练模型 "Tanrei/GPTSAN-japanese" 初始化一个 tokenizer 对象
    
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["input_ids"]
    # 使用 tokenizer 对带有前缀文本的字符串进行编码，返回 token IDs 列表
    [35993, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 35998, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]
    
    >>> # Mask for Prefix-LM inputs
    # 返回带有前缀文本的输入的 token 类型 IDs
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["token_type_ids"]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    
    
    
    Example for batch encode:
    
    >>> from transformers import GPTSanJapaneseTokenizer
    引入 GPTSanJapaneseTokenizer 类从 transformers 库
    
    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    使用预训练模型 "Tanrei/GPTSAN-japanese" 初始化一个 tokenizer 对象
    
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["input_ids"]
    # 使用 tokenizer 对批量输入进行编码，返回填充后的 token IDs 列表
    [[35993, 8640, 25948, 35998, 30647, 35675, 35999, 35999], [35993, 10382, 9868, 35998, 30646, 9459, 30646, 35675]]
    
    >>> # Mask for Prefix-LM inputs
    # 返回带有前缀文本的批量输入的 token 类型 IDs
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["token_type_ids"]
    [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]
    
    >>> # Mask for padding
    # 返回填充后的批量输入的注意力掩码
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["attention_mask"]
    [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        unk_token (`str`, *optional*, defaults to `"<|nottoken|>"`):
            The token used for unknown characters.
        pad_token (`str`, *optional*, defaults to `"<|separator|>"`):
            The token used for padding.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"<|segmenter|>"`):
            A special token to separate tokens into prefix and general input parts.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URLs, emails, telephone numbers, Japanese dates, and Japanese prices.
    """
    # Define constants for files related to vocabulary and model configurations
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
        # Check if vocabulary file exists; raise an error if not found
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = GPTSanJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Check if emoji file exists; raise an error if not found
        if not os.path.isfile(emoji_file):
            raise ValueError(
                f"Can't find an emoji file at path '{emoji_file}'. To load the emoji information from a Google"
                " pretrained model use `tokenizer = GPTSanJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        
        # Initialize the tokenizer with the provided parameters
        self.do_clean_text = do_clean_text
        self.vocab, self.raw_vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        self.subword_tokenizer = SubWordJapaneseTokenizer(
            vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji
        )

        # Initialize the superclass (TokenizerBase) with tokenizer specific parameters
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
    # Property to get the size of the vocabulary
    # Copied from tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizer.vocab_size
    def vocab_size(self):
        # The vocab_size property returns the length of the raw_vocab, which contains character variations unique to Japanese
        return len(self.raw_vocab)
    # 从 raw_vocab 和 added_tokens_encoder 构建并返回词汇表字典
    def get_vocab(self):
        return dict(self.raw_vocab, **self.added_tokens_encoder)

    # 使用 subword_tokenizer 对文本进行分词处理并返回结果
    def _tokenize(self, text):
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    # 根据 token 查找词汇表中的对应 id，如果找不到则返回 unk_token 的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 根据 id 查找词汇表中的对应 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.subword_tokenizer.convert_id_to_token(index)

    # 将一系列 token 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        words = []
        byte_tokens = []
        for word in tokens:
            if word[:6] == "<|byte" and word[-2:] == "|>":
                byte_tokens.append(int(word[6:-2]))
            else:
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                    byte_tokens = []
                if word[:7] == "<|emoji" and word[-2:] == "|>":
                    words.append(self.emoji["emoji_inv"][word])
                elif word == "<SP>":
                    words.append(" ")
                elif word == "<BR>":
                    words.append("\n")
                elif word == "<TAB>":
                    words.append("\t")
                elif word == "<BLOCK>":
                    words.append("▀")
                elif word == "<KIGOU>":
                    words.append("ǀ")
                elif word == "<U2000U2BFF>":
                    words.append("‖")
                elif word == "<|bagoftoken|>":
                    if len(words) > 0:
                        words.append(words[-1])
                        words.append(words[-1])
                        words.append(words[-1])
                elif word.startswith("<|") and word.endswith("|>"):
                    words.append("")
                else:
                    words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        text = "".join(words)
        return text
    # 默认的聊天模板，用于在消息之间添加标准的BOS、SEP和EOS标记，并且不包含角色信息。
    def default_chat_template(self):
        """
        A simple chat template that adds standard BOS, SEP and EOS tokens between messages while discarding role
        information.
        """
        # 如果未为此分词器定义聊天模板，则警告并使用默认模板
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回格式化后的聊天模板字符串
        return (
            "{% for message in messages %}"
            "{% if not loop.first %}{{ bos_token}}{% endif %}"
            "{{ sep_token }}{{ message.content }} {{ eos_token }}"
            "{% endfor %}"
        )

    # 从 GPTNeoXJapaneseTokenizer.save_vocabulary 复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径和表情符号文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
            emoji_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["emoji_file"]
            )
        else:
            # 构建词汇表文件路径和表情符号文件路径（不是目录）
            vocab_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["vocab_file"]
            )
            emoji_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["emoji_file"]
            )
        # 写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表映射，将索引和对应的 token 写入文件
            for token_index, token in self.ids_to_tokens.items():
                if index != token_index:
                    # 若词汇表索引不连续，发出警告
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将 token 写入文件，每个 token 用逗号分隔
                writer.write(",".join(token) + "\n")
                index += 1
        # 写入表情符号文件
        with open(emoji_file, "w", encoding="utf-8") as writer:
            json.dump(self.emoji, writer)
        # 返回词汇表文件和表情符号文件的路径
        return vocab_file, emoji_file

    # 创建 token_type_ids 从 token_ids_0 和 token_ids_1 中
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # docstyle-ignore
        """
        The tokenizer returns token_type_ids as separators between the Prefix part and the rest.
        token_type_ids is 1 for the Prefix part and 0 for the rest of the token.

        Example:
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
        ```"""
        # 计算前缀长度的初始值为 0
        prefix_len = 0
        # 检查分隔符在词汇表中存在
        if self.sep_token in self.vocab:
            # 获取分隔符在词汇表中的索引
            segid = self.vocab[self.sep_token]
            # 如果 token_ids_0 中存在分隔符的索引
            if segid in token_ids_0:
                # 计算前缀长度为分隔符索引之前的长度
                prefix_len = token_ids_0.index(segid)
        # 如果 token_ids_1 为 None，则总长度为 token_ids_0 的长度
        if token_ids_1 is None:
            total_len = len(token_ids_0)
        else:
            # 否则总长度为 token_ids_0 和 token_ids_1 的长度之和
            total_len = len(token_ids_0 + token_ids_1)
        # 返回前缀长度数量的 1，后面补充 (总长度 - 前缀长度) 个 0 组成的列表
        return prefix_len * [1] + (total_len - prefix_len) * [0]

    def prepare_for_tokenization(self, text, prefix_text=None, add_sep_token=None, **kwargs):
        # GPTSAN 在 Prefix-LM 中除了在文本生成中插入的 SOT，还额外插入 SEP 标记。
        # 文本开头的 SOT，以及在前缀部分和其余部分之间的 SEP 标记。
        if add_sep_token is None:
            # 如果未明确在非前缀位置插入 SEP 标记
            add_sep_token = self.sep_token not in text
        # 准备 tokenization 的文本，初始为空字符串或者以 BOS 标记开头的字符串
        prepared = self.bos_token if self.bos_token in self.vocab else ""
        # 如果有前缀文本，则将其添加到准备的文本中
        prepared += prefix_text if prefix_text is not None else ""
        # 如果需要添加 SEP 标记，则将其添加到准备的文本中
        if add_sep_token:
            prepared += self.sep_token if self.sep_token in self.vocab else ""
        # 将原始文本添加到准备的文本中
        prepared += text
        # 返回包含准备好的文本和其他关键字参数的元组
        return (prepared, kwargs)
    # 定义了一个方法 `_batch_encode_plus`，用于批量编码文本或文本对
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        # 此标记器将输入文本对转换为前缀输入和后续输入
        if isinstance(batch_text_or_text_pairs[0], tuple) or isinstance(tuple(batch_text_or_text_pairs[0]), list):
            # 如果输入是文本对或文本对列表，则处理成前缀加分隔符后的单一文本列表
            batch_prefix_texts = []
            for pref, txt in batch_text_or_text_pairs:
                batch_prefix_texts.append(pref + self.sep_token + txt)
            batch_text_or_text_pairs = batch_prefix_texts

        # 调用父类的 `_batch_encode_plus` 方法，传递所有参数，并返回结果
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
# 定义 SubWordJapaneseTokenizer 类，用于日语分词，基于 GPTNeoXJapaneseTokenizer 并进行了以下修改
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

    # 从 tokenization_gpt_neox_japanese.SubWordJapaneseTokenizer.__init__ 复制而来
    def __init__(self, vocab, ids_to_tokens, emoji):
        self.vocab = vocab  # 初始化词汇表属性，与参数swe相同
        self.ids_to_tokens = ids_to_tokens  # 初始化 ID 到词汇映射属性，与参数bpe相同
        self.emoji = emoji  # 初始化表情符号属性
        self.maxlen = np.max([len(w) for w in self.vocab.keys()])  # 计算词汇表中最长词的长度并赋值给maxlen
        # 初始化用于匹配文本中各种模式的正则表达式
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r"[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}")
        self.content_repatter4 = re.compile(
            r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter5 = re.compile(
            r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter6 = re.compile(
            r"((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*"
        )
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k: "<BLOCK>" for k in keisen + blocks})  # 创建字符替换映射表

    # 从tokenization_gpt_neox_japanese.SubWordJapaneseTokenizer.__len__中复制而来
    def __len__(self):
        return len(self.ids_to_tokens)  # 返回ids_to_tokens的长度作为对象的长度

    # 从tokenization_gpt_neox_japanese.SubWordJapaneseTokenizer.clean_text中复制而来
    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>", content)  # 将文本中的URL替换为"<URL>"
        content = self.content_repatter2.sub("<EMAIL>", content)  # 将文本中的邮箱地址替换为"<EMAIL>"
        content = self.content_repatter3.sub("<TEL>", content)  # 将文本中的电话号码替换为"<TEL>"
        content = self.content_repatter4.sub("<DATE>", content)  # 将文本中的日期替换为"<DATE>"
        content = self.content_repatter5.sub("<DATE>", content)  # 将文本中的日期替换为"<DATE>"
        content = self.content_repatter6.sub("<PRICE>", content)  # 将文本中的价格替换为"<PRICE>"
        content = content.translate(self.content_trans1)  # 使用content_trans1进行文本的字符替换
        while "<BLOCK><BLOCK>" in content:
            content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")  # 将连续的"<BLOCK><BLOCK>"替换为单个"<BLOCK>"
        return content

    # 从tokenization_gpt_neox_japanese.SubWordJapaneseTokenizer.tokenize中复制而来
    # 将文本中的空格替换为"<SP>"
    text = text.replace(" ", "<SP>")
    # 将全角空格替换为"<SP>"
    text = text.replace("　", "<SP>")
    # 将 Windows 换行符"\r\n"替换为"<BR>"
    text = text.replace("\r\n", "<BR>")
    # 将普通换行符"\n"替换为"<BR>"
    text = text.replace("\n", "<BR>")
    # 将老式 Mac 换行符"\r"替换为"<BR>"
    text = text.replace("\r", "<BR>")
    # 将制表符"\t"替换为"<TAB>"
    text = text.replace("\t", "<TAB>")
    # 将"—"替换为"ー"
    text = text.replace("—", "ー")
    # 将"−"替换为"ー"
    text = text.replace("−", "ー")
    
    # 遍历表情字典中的每个键值对，如果文本中包含某个键，则用对应的值替换文本中的键
    for k, v in self.emoji["emoji"].items():
        if k in text:
            text = text.replace(k, v)
    
    # 如果 clean 参数为 True，则对文本进行清洗处理
    if clean:
        text = self.clean_text(text)

    # 定义检查单个字符是否为特定符号的函数
    def check_simbol(x):
        e = x.encode()
        # 检查字符长度为1且编码长度为2的情况
        if len(x) == 1 and len(e) == 2:
            c = (int(e[0]) << 8) + int(e[1])
            # 检查是否符合特定范围内的字符编码
            if (
                (c >= 0xC2A1 and c <= 0xC2BF)
                or (c >= 0xC780 and c <= 0xC783)
                or (c >= 0xCAB9 and c <= 0xCBBF)
                or (c >= 0xCC80 and c <= 0xCDA2)
            ):
                return True
        return False

    # 定义检查单个字符是否为 Unicode 表意文字扩展区域的函数
    def checku2e(x):
        e = x.encode()
        # 检查字符长度为1且编码长度为3的情况
        if len(x) == 1 and len(e) == 3:
            c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
            # 检查是否符合特定范围内的字符编码
            if c >= 0xE28080 and c <= 0xE2B07F:
                return True
        return False

    # 初始化位置变量为0
    pos = 0
    # 初始化结果列表
    result = []
    # 当位置小于文本长度时循环处理文本
    while pos < len(text):
        # 如果当前字符是"<"，则结束位置为当前位置加上最大长度加1；否则结束位置为当前位置加3
        end = min(len(text), pos + self.maxlen + 1) if text[pos] == "<" else pos + 3
        # 候选词列表初始化为空
        candidates = []  # (token_id, token, pos)
        # 从结束位置向当前位置遍历
        for e in range(end, pos, -1):
            # 获取当前位置到结束位置的子串
            wd = text[pos:e]
            # 如果该子串在词汇表中存在
            if wd in self.vocab:
                # 如果子串以"<"开头且长度大于2，则将其作为一个候选项加入列表
                if wd[0] == "<" and len(wd) > 2:
                    candidates = [(self.vocab[wd], wd, e)]
                    break
                else:
                    candidates.append((self.vocab[wd], wd, e))
        # 如果候选词列表不为空
        if len(candidates) > 0:
            # 根据 token_id 最小的原则选取候选项中的一个进行处理
            _, wd, e = sorted(candidates, key=lambda x: x[0])[0]
            # 将选取的词添加到结果列表中
            result.append(wd)
            # 更新位置为 e
            pos = e
        else:
            # 如果候选词列表为空，则处理当前位置到结束位置的子串
            end = pos + 1
            wd = text[pos:end]
            # 如果子串为特定符号，则将"<KIGOU>"加入结果列表
            if check_simbol(wd):
                result.append("<KIGOU>")
            # 如果子串为 Unicode 表意文字扩展区域的字符，则将"<U2000U2BFF>"加入结果列表
            elif checku2e(wd):
                result.append("<U2000U2BFF>")
            else:
                # 否则将子串中的每个字节按照格式"<|byte%d|>"添加到结果列表中
                for i in wd.encode("utf-8"):
                    result.append("<|byte%d|>" % i)
            # 更新位置为 end
            pos = end
    
    # 返回处理后的结果列表
    return result
```