# `.\transformers\models\led\tokenization_led.py`

```py
# 设定文件编码格式和版权声明
# 使用 Apache 许可证版本 2.0
# 可在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 如果没有合法许可，不允许使用此文件中的内容
# 分发的软件基于“按原样”发布的基础，没有任何明示或暗示的担保或条件
# 有关特定语言管理权限和限制的详细信息，请参阅许可证
"""LED模型的Tokenization类。"""

import json  # 导入JSON库
import os  # 导入操作系统库
from functools import lru_cache  # 导入 lru_cache 装饰器
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关库

import regex as re  # 导入正则表达式库

from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入预训练分词器类
from ...tokenization_utils_base import BatchEncoding, EncodedInput  # 导入批次编码和编码输入
from ...utils import PaddingStrategy, logging  # 导入填充策略和日志记录

# 导入日志
logger = logging.get_logger(__name__)

# 词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 映射预训练词汇文件
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/vocab.json",
    },
    "merges_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer.json",
    },
}

# 位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/led-base-16384": 16384,
}

@lru_cache()  # 缓存装饰器
def bytes_to_unicode():
    """
    返回utf-8字节列表和与unicode字符串的映射
    返回utf-8字节列表和与unicode字符串的映射。在此处专门避免映射到空格/控制字符。
    可逆的bpe代码适用于unicode字符串。这意味着如果您想避免 UNK，你需要大量的unicode字符。例如，10B令牌的数据集需要大约5000个unicode字符以获得良好的覆盖率。这占了正常32K bpe词汇表的相当大比重。为了避免这种情况，我们希望utf-8字节与unicode字符串之间的查找表
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

def get_pairs(word):
    """返回单词中的符号对集合。单词表示为符号的元组（符号���可变长度字符串）。"""
    # 创建一个空集合用来存储字符对
    pairs = set()
    # 取单词的第一个字符作为前一个字符
    prev_char = word[0]
    # 遍历单词中的字符，将每两个相邻字符作为一个字符对加入集合中
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    # 返回字符对集合
    return pairs
# LEDTokenizer 类继承自 PreTrainedTokenizer 类，用于构建 LED 分词器，该分词器类似于 ROBERTa 分词器，使用字节级别的字节对编码。

"""
Constructs a LED tokenizer, which is smilar to the ROBERTa tokenizer, using byte-level Byte-Pair-Encoding.
构造 LED 分词器，它类似于 ROBERTa 分词器，使用字节级别的字节对编码。

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
be encoded differently whether it is at the beginning of the sentence (without space) or not:

这个分词器已经训练成将空格视为标记的一部分（有点像 sentencepiece），因此一个单词的编码会根据它是否在句子的开头（没有空格）而不同：


>>> from transformers import LEDTokenizer

>>> tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]


You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

您可以通过在实例化此分词器时或在对某些文本进行调用时传递 `add_prefix_space=True` 来避免这种行为，但由于模型不是这样预训练的，因此可能会导致性能下降。

<Tip>

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

<Tip>

使用 `is_split_into_words=True` 时，此分词器将在每个单词之前添加一个空格（即使是第一个单词）。

This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.
此分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应该参考这个超类以获取有关这些方法的更多信息。
    Args:
        vocab_file (`str`):
            Path to the vocabulary file. 词汇表文件的路径
        merges_file (`str`):
            Path to the merges file. 合并文件的路径
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information. 解码字节为 UTF-8 时要遵循的范例。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (BART tokenizer detect beginning of words by the preceding space).
    """

    # 以下三个变量存储特定文件名的映射关系
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer.__init__
    # 初始化函数，设置各种默认参数和属性
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
        # 如果 bos_token 是字符串，则将其转换为 AddedToken 对象，同时不去除左右空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则将其转换为 AddedToken 对象，同时不去除左右空格
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 sep_token 是字符串，则将其转换为 AddedToken 对象，同时不去除左右空格
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        # 如果 cls_token 是字符串，则将其转换为 AddedToken 对象，同时不去除左右空格
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        # 如果 unk_token 是字符串，则将其转换为 AddedToken 对象，同时不去除左右空格
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则将其转换为 AddedToken 对象，同时不去除左右空格
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 创建一个包含字符串 mask_token 的 AddedToken 对象，并去除左空格，不去除右空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 读取 vocab_file 文件的内容，将其解析为 JSON 格式，赋值给 self.encoder
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 将 self.encoder 中的 key 和 value 互换位置，赋值给 self.decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置错误处理方式
        self.errors = errors
        # 创建一个字节到 Unicode 的映射
        self.byte_encoder = bytes_to_unicode()
        # 创建一个 Unicode 到字节的映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 读取 merges_file 文件的内容，按换行符分割为列表，并去掉首尾空行
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将 bpe_merges 列表中的每个元素按空格分割为元组，组成新的列表
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建一个包含 bpe_merges 元组和对应索引的字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 创建一个空字典
        self.cache = {}
        # 设置是否在前缀空格后添加空格
        self.add_prefix_space = add_prefix_space

        # 创建一个正则表达式，用来匹配字符串中的特定模式
        # 匹配 's|'t|'re|'ve|'m|'ll|'d 或者匹配任意字母或数字或非空白字符或多个空白字符
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化函数，传入指定参数和额外的参数
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

    @property
    # 返回编码器的大小
    def vocab_size(self):
        return len(self.encoder)

    # 返回编码器的字典
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 这里应该有一个注释来解释函数的作用
    # 与 transformers.models.bart.tokenization_bart.BartTokenizer.bpe 相同
    def bpe
    # 对给定的 token 进行 BPE 处理
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        # 获取 token 的所有 pairs
        pairs = get_pairs(word)

        # 如果 token 没有 pairs，则返回原始 token
        if not pairs:
            return token

        # 循环处理 token
        while True:
            # 选择当前 pairs 中权重最小的 bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果选中的 bigram 不在 bpe_ranks 中，则退出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 根据选择的 bigram 对 word 进行拆分并重组
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
        # 将处理后的 word 转换为字符串形式
        word = " ".join(word)
        # 缓存处理结果，并返回
        self.cache[token] = word
        return word

    # 从给定的文本中进行 BPE 处理
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        # 使用正则表达式将文本分割为 token，并处理每个 token
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 将处理后的 token 拆分为多个 bpe token，并加入结果列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 根据给定的 token 返回其对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 根据给定的 id 返回对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 将一系列 tokens 转换为字符串形式
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # 保存词汇表（暂不需要添加注释）
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 设置词汇表文件和合并文件的路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器(encoder)以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引为0，打开合并文件进行写操作，写入版本信息，遍历BPE记录写入合并文件
        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # 生成带有特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A LED sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int)`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果没有第二个token序列，直接返回特殊标记拼接后的第一个token序列
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 否则，构建带有特殊标记的输入序列并返回
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # 获取带有特殊标记的掩码
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
        # If the token list already has special tokens, call the get_special_tokens_mask method from the parent class
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If token_ids_1 is None, add special tokens to the token_ids_0 list
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # If token_ids_1 is not None, add special tokens to both token_ids_0 and token_ids_1 lists
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer.create_token_type_ids_from_sequences with BART->LED
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. LED does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If token_ids_1 is None, return a list of zeros with the length of cls + token_ids_0 + sep
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # If token_ids_1 is not None, return a list of zeros with the length of cls + token_ids_0 + 2(sep) + token_ids_1 + sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer.prepare_for_tokenization
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # If the text is split into words or add_prefix_space is True and the first character of the text is not a space, add a space at the beginning of the text
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    # 声明函数返回一个字典类型
    ) -> dict:
        # 在调用父类的_pad方法进行填充时，传入参数
        # encoded_inputs：编码后的输入
        # max_length：最大长度
        # padding_strategy：填充策略
        # pad_to_multiple_of：填充到最大长度的倍数
        # return_attention_mask：是否返回注意力掩码
        encoded_inputs = super()._pad(
            encoded_inputs=encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 从模型默认值加载
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # 如果需要返回注意力掩码并且编码后的输入中包含全局注意力掩码
        if return_attention_mask and "global_attention_mask" in encoded_inputs:
            required_input = encoded_inputs[self.model_input_names[0]]
            # 'global_attention_mask'需要与其他(序列)输入的长度相同
            needs_to_be_padded = len(encoded_inputs["global_attention_mask"]) != len(required_input)

            if needs_to_be_padded:
                difference = len(required_input) - len(encoded_inputs["global_attention_mask"])

                if self.padding_side == "right":
                    # 使用'-1'，因为'global_attention_mask'中的'0'表示'局部注意力'而不是'不需要关注'
                    encoded_inputs["global_attention_mask"] = (
                        encoded_inputs["global_attention_mask"] + [-1] * difference
                    )
                elif self.padding_side == "left":
                    encoded_inputs["global_attention_mask"] = [-1] * difference + encoded_inputs[
                        "global_attention_mask"
                    ]
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        # 返回编码后的输入
        return encoded_inputs
```