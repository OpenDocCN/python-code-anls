# `.\models\blenderbot\tokenization_blenderbot.py`

```py
# coding=utf-8
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for Blenderbot."""

import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import regex as re  # 引入 regex 库，用于处理正则表达式

from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入自定义的 Token 和预训练 Tokenizer
from ...utils import logging  # 导入日志模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "merges_file": "merges.txt",  # 合并文件名
    "tokenizer_config_file": "tokenizer_config.json",  # 分词器配置文件名
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/vocab.json"},
    "merges_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/merges.txt"},
    "tokenizer_config_file": {
        "facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/tokenizer_config.json"
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/blenderbot-3B": 128}  # 预训练位置嵌入的大小

@lru_cache()
# 从 transformers.models.roberta.tokenization_roberta 中复制，用于将字节转换为 Unicode 字符
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# 从 transformers.models.roberta.tokenization_roberta 中复制，用于获取单词中的符号对
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    # 对单词中除第一个字符外的每个字符进行迭代
    for char in word[1:]:
        # 将前一个字符和当前字符作为一个元组加入到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，以便下一次迭代使用
        prev_char = char
    # 返回包含所有字符对的集合
    return pairs
# 定义 BlenderbotTokenizer 类，继承自 PreTrainedTokenizer
class BlenderbotTokenizer(PreTrainedTokenizer):
    """
    Constructs a Blenderbot tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import BlenderbotTokenizer

    >>> tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3B")
    >>> tokenizer.add_prefix_space = False
    >>> tokenizer("Hello world")["input_ids"]
    [47, 921, 86, 1085, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [6950, 1085, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Specifies the error handling scheme to use for decoding bytes to UTF-8.
            See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for details.

        bos_token (`str`, *optional*, defaults to `"<s>"`):
            Beginning of sequence token used during pretraining. Often employed as a sequence classifier token.

            <Tip>
            This token is not typically used as the beginning of sequence when special tokens are employed. 
            Instead, the `cls_token` is used.
            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            End of sequence token.

            <Tip>
            When constructing sequences with special tokens, this is not used as the end of sequence.
            The `sep_token` is used instead.
            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            Separator token used for constructing sequences from multiple sources, 
            such as for sequence classification or question answering.

        cls_token (`str`, *optional*, defaults to `"<s>"`):
            Classifier token used in sequence classification tasks. It is the first token in the sequence when using special tokens.

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            Token representing unknown words or tokens not in the vocabulary.

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            Token used for padding sequences to equal lengths during batching.

        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            Token used during masked language modeling, indicating positions where the model will predict.

        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Indicates whether to add an initial space to the input, treating the leading word like any other word.

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.__init__中复制而来，用于初始化Blenderbot的Tokenizer类
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs,
    ):
        # 如果bos_token是字符串，则创建一个对应的AddedToken对象，用于表示序列开始的特殊标记
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果pad_token是字符串，则创建一个对应的AddedToken对象，用于表示填充的特殊标记
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果eos_token是字符串，则创建一个对应的AddedToken对象，用于表示序列结束的特殊标记
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果unk_token是字符串，则创建一个对应的AddedToken对象，用于表示未知标记的特殊标记
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果sep_token是字符串，则创建一个对应的AddedToken对象，用于表示分隔符的特殊标记
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        # 如果cls_token是字符串，则创建一个对应的AddedToken对象，用于表示类别标记的特殊标记
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
    
        # mask_token的行为类似于普通单词，即在其前面包含空格
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )
    
        # 这些特殊标记不包含在vocab.json中，因此将它们按正确顺序添加
        # 用UTF-8编码打开vocab_file，并加载其中的编码器内容为字典self.encoder
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 通过self.encoder创建反向映射字典self.decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置解码过程中的错误处理方式
        self.errors = errors
        # 创建字节到Unicode的编码映射字典self.byte_encoder
        self.byte_encoder = bytes_to_unicode()
        # 通过self.byte_encoder创建反向映射字典self.byte_decoder
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 用UTF-8编码打开merges_file，读取内容并分割成行，排除第一行和最后一行空行后，将其转换为元组列表bpe_merges
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将bpe_merges列表中的每个合并规则字符串转换为元组，并构建合并规则到索引的映射字典self.bpe_ranks
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存字典self.cache为空字典
        self.cache = {}
        # 设置是否在特殊标记前添加空格的标志
        self.add_prefix_space = add_prefix_space
    
        # 应该添加re.IGNORECASE，以便对缩写的大写版本进行BPE合并
        # 编译正则表达式，用于识别缩写、字母和数字、非空白非字母数字字符、空白（排除非空白字符后的空白）
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
        # 调用父类的初始化方法，传递参数设置
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
    # 返回当前词汇表的大小，即编码器的长度
    def vocab_size(self):
        return len(self.encoder)

    # 从Blenderbot的词汇表中获取完整的词汇表，包括添加的特殊标记
    def get_vocab(self):
        # 复制编码器中的内容到vocab字典中
        vocab = dict(self.encoder).copy()
        # 将添加的特殊标记编码器内容更新到vocab字典中
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 根据Blenderbot的BPE算法处理给定的token，返回处理后的字符串
    def bpe(self, token):
        # 如果token已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        # 使用Blenderbot的BPE算法处理token，生成pairs
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # 找到当前pairs中优先级最低的bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    # 使用Blenderbot的BPE算法对给定的文本进行分词，返回分词后的结果
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        # 使用正则表达式找到所有匹配的token，并逐个处理
        for token in re.findall(self.pat, text):
            # 将token编码成字节，并通过Blenderbot的字节编码器映射成unicode字符串
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 使用Blenderbot的BPE算法对编码后的token进行分词，将分词结果添加到bpe_tokens列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 将给定的token转换为其在Blenderbot词汇表中的ID，如果token不存在，则使用未知标记的ID
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将给定的ID转换为其在Blenderbot词汇表中对应的token，如果ID不存在，则返回对应的未知标记
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)
    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.convert_tokens_to_string复制而来，将Roberta->Blenderbot，RoBERTa->Blenderbot
    def convert_tokens_to_string(self, tokens):
        """将一系列的tokens（字符串）转换为单个字符串。"""
        # 将tokens列表中的所有字符串连接成一个字符串
        text = "".join(tokens)
        # 使用self.byte_decoder中的映射将text中的每个字符解码为UTF-8编码的字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.save_vocabulary复制而来，将Roberta->Blenderbot，RoBERTa->Blenderbot
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果save_directory不是一个目录，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"词汇表路径 ({save_directory}) 应为一个目录")
            return
        # 构建词汇文件的路径，如果提供了filename_prefix，则使用它作为前缀
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件的路径，如果提供了filename_prefix，则使用它作为前缀
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将self.encoder中的内容以UTF-8编码格式写入vocab_file
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将BPE（Byte Pair Encoding）的tokens和它们的索引写入merge_file
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 按照token_index排序self.bpe_ranks.items()，并将每个bpe_tokens列表写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"保存词汇到 {merge_file}: BPE合并索引不是连续的。请确保分词器未损坏！"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_special_tokens_mask复制而来，将Roberta->Blenderbot，RoBERTa->Blenderbot
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        # 返回一个掩码，指示哪些token是特殊token（如[PAD]、[CLS]、[SEP]等）
        pass
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
        # Check if the token list already has special tokens; if so, delegate to superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If no special tokens are present and there is only one token list, add special tokens at the start and end
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # If there are two token lists, add special tokens appropriately for sequence pairs
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer.create_token_type_ids_from_sequences with Roberta->Blenderbot, RoBERTa->Blenderbot
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. Blenderbot does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If there's only one sequence, return a list of zeros of appropriate length
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # If there are two sequences, return a list of zeros of appropriate length
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer.prepare_for_tokenization with Roberta->Blenderbot, RoBERTa->Blenderbot
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        Prepare the text for tokenization, ensuring correct formatting based on tokenizer settings.

        Args:
            text (str): The input text to be tokenized.
            is_split_into_words (bool, optional): Whether the text is already split into words.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[str, Dict]: A tuple containing the processed text and any additional kwargs.
        """
        # Determine if a prefix space should be added and apply if necessary
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        # Return processed text and remaining keyword arguments
        return (text, kwargs)
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Blenderbot sequence has the following format:
        - single sequence: ` X </s>`
    
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Will be ignored
        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 将 token_ids_0 和 EOS（结束符号）的 token ID 进行连接，构建包含特殊标记的模型输入
        return token_ids_0 + [self.eos_token_id]
    
    @property
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        # 如果未为此分词器定义聊天模板，则记录警告并返回默认的聊天模板字符串
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回一个简单的聊天模板字符串，用于在消息之间添加空格
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
```