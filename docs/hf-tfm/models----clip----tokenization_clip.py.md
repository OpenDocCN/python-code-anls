# `.\models\clip\tokenization_clip.py`

```
# 设置文件编码为utf-8
# 版权声明
# 根据Apache 2.0许可证，除非符合许可证要求，否则不得使用此文件
# 您可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，
# 没有任何明示或暗示的担保或条件。请查看具体语言的许可协议以了解权限和限制
"""CLIP的标记化类"""

import json
import os
import unicodedata
from functools import lru_cache
from typing import List, Optional, Tuple

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取logger实例
logger = logging.get_logger(__name__)

# 词汇文件的名称控制
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
    },
}

# 预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai/clip-vit-base-patch32": 77,
}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "openai/clip-vit-base-patch32": {},
}

@lru_cache()
def bytes_to_unicode():
    """
    返回utf-8字节的列表以及到Unicode字符串的映射。
    我们特意避免了空格/控制字符的映射，因为BPE编码会因此而失败。
    可逆的BPE代码适用于Unicode字符串。这意味着如果你想避免UNK（未知标记），则需要大量的Unicode字符。
    例如，如果你有大约100亿标记的数据集，你最终需要大约5000个Unicode字符才能获得良好的覆盖率。
    这是常规32K BPE词汇表的相当大的百分比。为了避免这种情况，我们需要在utf-8字节和Unicode字符串之间建立查找表。
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
    """
    返回单词中的符号对集合。
    单词被表示为符号的元组（符号是可变长度的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def whitespace_clean(text):
    """
    清除文本中的空白字符，将多个空格替换为单个空格，然后去除首尾空格
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
# 从transformers.models.bert.tokenization_bert.whitespace_tokenize中复制的代码，函数用于对文本进行基本的空格清理和分词
def whitespace_tokenize(text):
    # 去除文本两端的空格
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到分词结果
    tokens = text.split()
    # 返回分词结果
    return tokens


# 从transformers.models.bert.tokenization_bert.BasicTokenizer中复制的代码，类用于进行基本的分词（标点分隔、小写处理等）
class BasicTokenizer(object):
    """
    构造一个BasicTokenizer，用于进行基本的分词（标点分隔、小写处理等）。

    参数:
        do_lower_case （`bool`，可选，默认为`True`）：
            是否在进行分词时将输入文本转为小写。
        never_split （`Iterable`，可选）：
            在分词过程中永远不需要切分的词汇集合。仅在`do_basic_tokenize=True`时生效。
        tokenize_chinese_chars （`bool`，可选，默认为`True`）：
            是否对中文字符进行分词。

            对于日文，此项应当关闭（参见
            [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents （`bool`，可选）：
            是否去除所有的重音符号。如果未指定该选项，则将根据`lowercase`的取值来确定（即和原始BERT一样）。
        do_split_on_punc （`bool`，可选，默认为`True`）：
            在某些情况下，我们希望跳过基本的标点切分，以便后续的分词可以捕捉到完整的词汇内容，例如缩写词。

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
        # 是否转小写
        self.do_lower_case = do_lower_case
        # 从不进行切分的词汇集合
        self.never_split = set(never_split)
        # 是否tokenize中文字符
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 是否去除重音符号
        self.strip_accents = strip_accents
        # 是否进行基本的标点切分
        self.do_split_on_punc = do_split_on_punc
    # 对文本进行基本的分词处理。有关子词分词，请参见 WordPieceTokenizer。

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果存在 never_split 参数，则将其与 self.never_split 求并集，构成新的不分割词集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除无效字符
        text = self._clean_text(text)

        # 添加于2018年11月1日，用于多语言和中文模型。现在也应用于英语模型，但是这没有关系，因为英语模型没有在任何中文数据上进行训练，
        # 通常也不包含任何中文数据（英语维基百科中有一些中文单词，因此词汇表中有中文字符。）
        if self.tokenize_chinese_chars:
            # 对文本中的中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 将文本中的 Unicode 标准化为 NFC 形式，以防止将具有不同 Unicode 码点的相同字符视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白符对文本进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            # 如果 token 不在不分割词集合中
            if token not in never_split:
                # 如果设置为小写，则将 token 转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果 strip_accents 不为 False，则去除 token 中的重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果 strip_accents 为 True，则去除 token 中的重音符号
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 使用标点符号分割 token，并将分割后的结果加入到 split_tokens 中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空白符对 split_tokens 进行再次分词，并返回分词结果
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 从文本中去除重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本中的字符标准化为 NFD 形式
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符的 Unicode 分类为 Mn（Nonspacing_Mark），则跳过该字符，不添加到输出结果中
            if cat == "Mn":
                continue
            output.append(char)
        # 将去除重音符号后的字符列表拼接成字符串，并返回结果
        return "".join(output)
    # 在文本上进行标点符号切分
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号上切分或者文本在never_split中，则返回原文本
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    # 在中文字符周围添加空格
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果是中文字符，则在两侧添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    # 检查是否是中文字符
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 定义"中文字符"为CJK Unicode块中的任何字符
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True

        return False

    # 在文本上执行无效字符删除和空格清理
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果是无效字符或控制字符
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
class CLIPTokenizer(PreTrainedTokenizer):
    """
    Construct a CLIP tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file. 词汇表文件的路径。
        merges_file (`str`):
            Path to the merges file. 合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
            解码字节到 UTF-8 时使用的范例。更多信息请参阅[bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            未知的标记。词汇表中没有的标记无法转换为 ID，并被设为此标记。
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
            序列的开头标记。
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
            序列的结尾标记。
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
            用于填充的标记，例如在批处理不同长度的序列时。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # hack to enable padding
        **kwargs,
    
    # 生成用于序列分类任务的模型输入，通过连接和添加特殊标记的方式。一个 CLIP 序列的格式如下：
    
    #   - 单个序列：`<|startoftext|> X <|endoftext|>`
    
    # 序列对不是预期的使用情况，但也可以处理，不需要分隔符。
    
    # 参数：
    #   - token_ids_0 (`List[int]`) ：用于添加特殊标记的 ID 列表。
    #   - token_ids_1 (`List[int]`, *optional*): 可选的第二个用于序列对的 ID 列表。
    
    # 返回值：
    #   - `List[int]`: 列表，包含适当的特殊标记的输入 ID。
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        
        # 创建开始标记的 ID 列表
        bos_token = [self.bos_token_id]
        # 创建结束标记的 ID 列表
        eos_token = [self.eos_token_id]
    
        # 如果没有第二个序列，那么只返回开始标记、token_ids_0 和结束标记的拼接结果
        if token_ids_1 is None:
            return bos_token + token_ids_0 + eos_token
        
        # 如果有第二个序列，返回开始标记、token_ids_0、结束标记、结束标记、token_ids_1 和结束标记的拼接结果
        return bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token
    
    
    # 从没有添加特殊标记的标记列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时，会调用此方法。
    
    # 参数：
    #   - token_ids_0 (`List[int]`) : ID 列表。
    #   - token_ids_1 (`List[int]`, *optional*): 可选的第二个 ID 列表，用于序列对。
    #   - already_has_special_tokens (`bool`, *optional*, 默认为 `False`): 标记列表是否已经使用了模型的特殊标记。
    
    # 返回值：
    #   - `List[int]`: 一个整数列表，范围在[0, 1]之间：1 表示特殊标记，0 表示序列标记。
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        
        # 如果标记列表已经添加了特殊标记，直接调用父类的 `get_special_tokens_mask` 方法
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        # 如果没有第二个序列，返回 1、0 重复 token_ids_0 的长度次数、1
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        
        # 如果有第二个序列，返回 1、0 重复 token_ids_0 的长度次数、1、1、0 重复 token_ids_1 的长度次数、1
        return [1] + ([0] * len(token_ids_0)) + [1] + [1] + ([0] * len(token_ids_1)) + [1]
    
    
    # 从序列 ID 创建 token_type_ids，用于序列对的输入。
    
    # 参数：
    #   - token_ids_0 (`List[int]`) : ID 列表。
    #   - token_ids_1 (`List[int]`, *optional*): 可选的第二个 ID 列表，用于序列对。
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed. CLIP does not make use of token type ids, therefore a list of
        zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # Define beginning of sequence and end of sequence tokens
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]

        # If only one sequence is passed
        if token_ids_1 is None:
            # Return a mask of zeros with the length of the concatenated sequence
            return len(bos_token + token_ids_0 + eos_token) * [0]
        # If two sequences are passed
        # Return a mask of zeros with the length of the concatenated sequences
        return len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token) * [0]

    def bpe(self, token):
        # If token is in cache, return its corresponding value
        if token in self.cache:
            return self.cache[token]
        # Add special token to end of token and split into pairs
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        # If no pairs are found, return token with special token appended
        if not pairs:
            return token + "</w>"

        # Loop until no more pairs can be merged
        while True:
            # Find the pair with the smallest rank
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # If the bigram is not in the ranks, break the loop
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # Merge pairs in word
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
            # If word is reduced to a single token, break the loop
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        # Cache the token and its corresponding value
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        # If fix_text is not provided, tokenize the text using the NLP model and join tokens with spaces
        if self.fix_text is None:
            text = " ".join(self.nlp.tokenize(text))
        # If fix_text is provided, clean the text and convert to lowercase
        else:
            text = whitespace_clean(self.fix_text(text)).lower()

        # Loop through tokens found using regex
        for token in re.findall(self.pat, text):
            # Convert bytes to unicode strings and split using BPE
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # Return the ID of the token from the vocab, or the ID of the unknown token if not found
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    # 根据给定的索引（整数），使用词汇表将索引转换为对应的词汇（字符串）
    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    # 将一系列的词汇（字符串）转换成单个字符串
    def convert_tokens_to_string(self, tokens):
        # 将词汇列表连接成一个字符串
        text = "".join(tokens)
        # 创建字节数组，将字符串中每个字符转换为其对应的字节值
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        # 将字节数组解码为 UTF-8 编码的字符串，并替换特定子串为指定字符，然后去除两侧空格
        text = byte_array.decode("utf-8", errors=self.errors).replace("</w>", " ").strip()
        return text

    # 将词汇表保存到指定目录，返回保存的文件名
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 判断保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        # 组合词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 组合合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        # 写入词汇表到文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入合并文件到文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            # 写入版本信息
            writer.write("#version: 0.2\n")
            # 遍历 BPE 分词和它们的索引，并按索引排序后写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果索引不连续，则输出警告信息
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merge_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file
```