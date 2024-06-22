# `.\transformers\models\whisper\tokenization_whisper.py`

```py
# 设置文件编码为 utf-8
# 版权声明，声明代码的版权归 The HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 许可，你可以在遵守许可的前提下使用本文件中的代码
# 你可以在以下网址获取许可的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件按“原样”提供，不提供任何形式的担保或条件
# 请查阅许可获取更多信息

# 导入所需库
import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import regex as re

# 导入父类 PreTrainedTokenizer 和其他所需工具函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 导入英文文本规范化类
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer

# 定义用于存储文件名的字典常量
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_file": "tokenizer.json",
    "merges_file": "merges.txt",
    "normalizer_file": "normalizer.json",
}

# 定义预训练模型的文件映射常量
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/vocab.json",
    },
    "merges_file": {"openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/merges_file.txt"},
    "normalizer_file": {
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/normalizer.json"
    },
}

# 定义最大模型输入尺寸的常量
MAX_MODEL_INPUT_SIZES = {
    "openai/whisper-base": 448,
}

# 从 GPT-2 的字节到 Unicode 的映射函数
def bytes_to_unicode():
    """
    返回 utf-8 字节列表及其对应的 Unicode 字符列表的映射关系。我们特意避免将空格和控制字符映射到Unicode字符串，
    因为这会导致 bpe 代码出错。

    可逆的 bpe 代码是在 Unicode 字符串上运行的。这意味着如果你想避免 UNKs，你需要在词汇表中有大量的 Unicode 字符。
    当你处理大约 100 亿令牌的数据集时，你最终需要大约 5000 个字符来达到良好的覆盖率。这在你的常规 32K bpe 词汇表中占有很大的比例。
    为了避免这种情况，我们希望有 utf-8 字节和 Unicode 字符串之间的查找表。
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

# 从 GPT-2 的字节到 Unicode 的映射函数中获取字符对的函数
def get_pairs(word):
    """
    返回一个单词中的符号对集合。

    单词表示为符号的元组（符号为可变长度字符串）。
    """
    pairs = set()
    prev_char = word[0]
    # 遍历单词中除第一个字符外的每个字符
    for char in word[1:]:
        # 将当前字符与前一个字符组成的元组添加到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符
        prev_char = char
    # 返回包含字符配对的集合
    return pairs
# 定义一个包含语言代码和语言名的字典
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    ...
    # 其他语言代码和语言名的对应关系
}

# 根据语言名查找对应的语言代码，包括一些语言别名
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    ...
    # 其他一些语言别名和对应的语言代码
}

# 定义任务类型列表
TASK_IDS = ["translate", "transcribe"]

# 定义一个名为WhisperTokenizer的类，并继承自PreTrainedTokenizer
class WhisperTokenizer(PreTrainedTokenizer):
    """
    Construct a Whisper tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.
    # 构建一个Whisper标记器
    # 这个标记器继承自`PreTrainedTokenizer`，其中包含一些主要方法。用户应该参考超类以获取有关这些方法的更多信息。
    # 定义函数的参数和说明
    Args:
        vocab_file (`str`):
            Path to the vocabulary file. 词汇文件的路径
        merges_file (`str`):
            Path to the merges file. 合并文件的路径
        normalizer_file (`str`, *optional*):
            Path to the normalizer_file file. 归一化文件的路径（可选）
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information. 解码字节为 UTF-8 时遵循的模式
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. 未知标记，不在词汇表中的标记无法转换为 ID，并设置为该标记
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
            `"<|startoftranscript|>"` when generating. 序列的开始标记，`decoder_start_token_id` 用于在生成时将第一个标记设置为 `"<|startoftranscript|>"`
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token. 序列的结束标记
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths. 用于填充的标记，例如在批处理不同长度的序列时使用
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. 是否在输入中添加初始空格，这样可以将第一个词视为其他词处理
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
            转录文本的语言，相应的语言 ID 标记被附加到序列的开头，用于多语言语音识别和语音翻译任务
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
            应该使用此标识符用于多语言微调，对于语音识别使用 "transcribe"，对于语音翻译使用 "translate"
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence. 是否在序列开头省略 `<|notimestamps|>` 标记
    """
    # 定义一些类属性
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化函数
    def __init__(
        self,
        vocab_file,
        merges_file,
        normalizer_file=None,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        language=None,
        task=None,
        predict_timestamps=False,
        **kwargs,
    # 如果 bos_token 是字符串，则创建一个添加特殊标记的 AddedToken 对象，否则保持原样
    bos_token = (
        AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False, special=True)
        if isinstance(bos_token, str)
        else bos_token
    )
    # 如果 eos_token 是字符串，则创建一个添加特殊标记的 AddedToken 对象，否则保持原样
    eos_token = (
        AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False, special=True)
        if isinstance(eos_token, str)
        else eos_token
    )
    # 如果 unk_token 是字符串，则创建一个添加特殊标记的 AddedToken 对象，否则保持原样
    unk_token = (
        AddedToken(unk_token, lstrip=False, rstrip=False, normalized=False, special=True)
        if isinstance(unk_token, str)
        else unk_token
    )
    # 如果 pad_token 是字符串，则创建一个添加特殊标记的 AddedToken 对象，否则保持原样
    pad_token = (
        AddedToken(pad_token, lstrip=False, rstrip=False, normalized=False, special=True)
        if isinstance(pad_token, str)
        else pad_token
    )
    
    # 使用 utf-8 编码打开 vocab_file，加载其中的 JSON 数据到 encoder 中
    with open(vocab_file, encoding="utf-8") as vocab_handle:
        self.encoder = json.load(vocab_handle)
    # 创建一个反转键值对的字典，将 encoder 中的值作为键，键作为值，存储到 decoder 中
    self.decoder = {v: k for k, v in self.encoder.items()}
    # 设置错误处理方式
    self.errors = errors
    # 使用 bytes_to_unicode 函数创建字节编码器和解码器
    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    # 使用 utf-8 编码打开 merges_file，读取并解析内容，存储到 bpe_merges 中
    with open(merges_file, encoding="utf-8") as merges_handle:
        bpe_merges = merges_handle.read().split("\n")[1:-1]
    # 将 bpe_merges 列表中的每个元素按空格拆分成元组，存储到 bpe_merges 中
    bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
    # 创建 BPE 合并到排名的映射，存储到 bpe_ranks 中
    self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    # 初始化缓存为空字典
    self.cache = {}
    # 设置是否在前缀空格后添加空格
    self.add_prefix_space = add_prefix_space
    
    # 如果 normalizer_file 不为 None，则使用 utf-8 编码打开并加载其中的 JSON 数据到 english_spelling_normalizer 中
    # 否则 english_spelling_normalizer 为 None
    if normalizer_file is not None:
        with open(normalizer_file, encoding="utf-8") as vocab_handle:
            self.english_spelling_normalizer = json.load(vocab_handle)
    else:
        self.english_spelling_normalizer = None
    
    # 设置正则表达式模式
    self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")
    
    # 设置语言，将错误处理方式和其他参数传递给父类的初始化方法
    self.language = language
    super().__init__(
        errors=errors,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )
    
    # 设置任务和预测时间戳
    self.task = task
    self.predict_timestamps = predict_timestamps
    
    # 声明 vocab_size 属性为只读属性，返回 encoder 的长度
    @property
    def vocab_size(self) -> int:
        return len(self.encoder)
    
    # 获取词汇表，将 token 转换成索引，并添加到 vocab 中
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe 复制过来
    # BPE（字节对编码）处理函数，将输入的 token 进行 BPE 编码
    def bpe(self, token):
        # 如果 token 已经在缓存中，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为元组形式
        word = tuple(token)
        # 获取 token 的所有字节对
        pairs = get_pairs(word)

        # 如果没有字节对，则返回原始 token
        if not pairs:
            return token

        # 循环处理字节对，直到无法再合并为止
        while True:
            # 找到当前字节对中频率最低的一个
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字节对不在 BPE 编码的字典中，则停止循环
            if bigram not in self.bpe_ranks:
                break
            # 将 bigram 拆分为两个字节
            first, second = bigram
            # 新建一个列表用于存放拆分后的 token
            new_word = []
            i = 0
            # 遍历 token 中的每个字符
            while i < len(word):
                # 尝试找到当前 bigram 的起始位置
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到，将剩余的字符加入新 token 列表并结束循环
                    new_word.extend(word[i:])
                    break
                else:
                    # 如果找到了，将 bigram 之前的字符加入新 token 列表
                    new_word.extend(word[i:j])
                    i = j

                # 如果当前字符与 bigram 的第一个字符相同，并且下一个字符也与 bigram 的第二个字符相同，则合并这两个字符
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则，将当前字符加入新 token 列表，并移动到下一个字符
                    new_word.append(word[i])
                    i += 1
            # 将新 token 转换为元组
            new_word = tuple(new_word)
            word = new_word
            # 如果新 token 的长度为 1，说明无法再合并，结束循环
            if len(word) == 1:
                break
            else:
                # 否则，继续获取新 token 的字节对
                pairs = get_pairs(word)
        # 将新 token 转换为字符串形式，并缓存结果
        word = " ".join(word)
        self.cache[token] = word
        # 返回 BPE 编码后的 token
        return word

    # 设置前缀 token 的方法，用于在标签序列的开头添加前缀 token
    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```python
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
        >>> # now switch the prefix token from Spanish to French
        >>> tokenizer.set_prefix_tokens(language="french")
        ```py

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
        # 更新语言前缀 token
        self.language = language if language is not None else self.language
        # 更新任务前缀 token
        self.task = task if task is not None else self.task
        # 更新预测时间戳前缀 token
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

    # 属性装饰器用于获取前缀 token 的方法
    @property
    def prefix_tokens(self) -> List[int]:
        # 将特殊起始标记转换为对应的 token id
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        translate_token_id = self.convert_tokens_to_ids("<|translate|>")
        transcribe_token_id = self.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token_id = self.convert_tokens_to_ids("<|notimestamps|>")
        # 获取支持的语言列表
        langs = tuple(LANGUAGES.keys())

        # 如果指定了语言
        if self.language is not None:
            # 将语言转换为小写
            self.language = self.language.lower()
            # 如果语言在 TO_LANGUAGE_CODE 中
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            # 如果语言在 TO_LANGUAGE_CODE 的值中
            elif self.language in TO_LANGUAGE_CODE.values():
                language_id = self.language
            # 否则抛出异常
            else:
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        # 如果指定了任务
        if self.task is not None:
            # 如果任务不在 TASK_IDS 中，抛出异常
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        # 构建起始 token 序列
        bos_sequence = [bos_token_id]
        # 如果指定了语言，则添加语言相关的 token id
        if self.language is not None:
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        # 如果指定了任务，则添加相应的 token id
        if self.task is not None:
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        # 如果不需要预测时间戳，则添加相应的 token id
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence

    # 从序列中构建模型输入，并在末尾添加 eos token id
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # 对于双序列，将 prefix_tokens、token_ids_0、token_ids_1 和 eos_token_id 连接起来
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # 获取特殊 token 的 mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
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
            # If the input token list already contains special tokens, directly call the base class method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)  # Create a list of 1s for the special tokens prefix
        suffix_ones = [1]  # Create a list with a single 1 for the special tokens suffix
        if token_ids_1 is None:
            # If there's no second token list (for sequence pairs), return the special tokens mask for the first list only
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        # If there are two token lists (for sequence pairs), return the special tokens mask for both lists
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize with GPT2 -> Whisper
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        # Tokenize the input text using byte pair encoding (BPE)
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id with GPT2 -> Whisper
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # Convert a token into its corresponding ID using the vocabulary
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab. Whisper's base tokenizer always decodes OOV
        tokens as "", thus we do not use the `unk_token` here.
        """
        # Convert an ID into its corresponding token using the vocabulary
        return self.decoder.get(index, "")

    def _normalize(self, text):
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which preforms commons transformation on
        english text.
        """
        # Normalize the input text using the EnglishTextNormalizer class
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        return normalizer(text)

    @staticmethod
    # 使用 BasicTextNormalizer 类对给定文本进行基本归一化处理
    def _basic_normalize(text, remove_diacritics=False):
        """
        Normalize a given string using the `BasicTextNormalizer` class, which preforms commons transformation on
        multilingual text.
        """
        # 创建 BasicTextNormalizer 实例，根据参数决定是否移除文本中的变音符号
        normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
        # 使用 normalizer 实例对文本进行归一化处理并返回结果
        return normalizer(text)
    
    # 使用 decode_with_timestamps 方法解码带有时间戳的 token 序列
    def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        # 时间戳 token 的 ID 范围比特殊 token 的 ID 范围大
        timestamp_begin = self.all_special_ids[-1] + 1
        # 存放解码结果的列表
        outputs = [[]]
    
        # 当前最大时间戳和之前段落的总时长
        cur_max_timestamp = 0.0
        prev_segments_len = 0.0
    
        # 遍历 token 序列
        for token in token_ids:
            # 如果当前 token 是时间戳 token
            if token >= timestamp_begin:
                # 根据 token 计算出对应的时间戳
                timestamp = float((token - timestamp_begin) * time_precision)
    
                # 如果当前时间戳小于之前的最大时间戳，说明进入了下一个段落
                if timestamp < cur_max_timestamp:
                    # 更新之前段落的总时长
                    prev_segments_len += cur_max_timestamp
    
                # 更新当前最大时间戳
                cur_max_timestamp = timestamp
    
                # 添加时间戳注解
                outputs.append(f"<|{(timestamp + prev_segments_len):.2f}|>")
                outputs.append([])
            # 如果当前 token 不是时间戳 token
            else:
                # 添加到当前段落的 token 列表中
                outputs[-1].append(token)
    
        # 遍历 outputs 列表，将每个元素解码为字符串
        outputs = [
            s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
        ]
        # 将所有字符串拼接成最终结果并返回
        return "".join(outputs)
    def _compute_offsets(self, token_ids, time_precision=0.02):
        """
        根据给定的标记化输入计算偏移量
    
        参数:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                标记化输入 id 列表，可通过 `__call__` 方法获得。
            time_precision (`float`, `optional`, 默认为 0.02):
                从标记到时间的转换时间精度。
        """
        # 存储计算得到的偏移量
        offsets = []
    
        # 如果输入的是 torch 的张量，则将其放置在 cpu 上
        if "torch" in str(type(token_ids)) and (hasattr(token_ids, "cpu") and callable(token_ids.cpu)):
            token_ids = token_ids.cpu()
    
        # 将 token_ids 转换为 numpy 数组
        token_ids = np.array(token_ids)
    
        # 如果 token_ids 的第一个维度大于1且 token_ids 的维度大于1，则抛出 ValueError
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
    
        # 从 self.all_special_ids 中获取时间戳起始的 id 值
        timestamp_begin = self.all_special_ids[-1] + 1
    
        # 获取标记化输入中的时间戳标记的位置
        timestamp_tokens = token_ids >= timestamp_begin
    
        # 获取连续的时间戳标记的索引
        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
    
        # 如果没有连续的时间戳标记，且时间戳标记的数量小于等于1，则返回空列表
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            return []
        # 如果最后一个时间戳标记的索引不在连续的时间戳标记列表中，则将其添加到列表中
        elif np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)
    
        # 设置初始切片的结束位置
        last_slice = np.where(timestamp_tokens)[0][0]
    
        # 遍历连续的时间戳标记的索引
        for current_slice in consecutive:
            # 获取当前切片的标记化输入
            sliced_tokens = token_ids[last_slice:current_slice]
    
            # 如果切片的长度大于1，则进行以下操作
            if len(sliced_tokens) > 1:
                # 计算起始和结束的时间戳位置
                start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin
    
                # 从文本输出中去掉时间戳标记
                sliced_tokens = self._preprocess_token_ids(sliced_tokens)
                text = self._decode(sliced_tokens)
                text = self._filter_timestamp_ids(text)
    
                # 将文本和时间戳位置添加到 offsets 中
                offsets.append(
                    {
                        "text": text,
                        "timestamp": (
                            start_timestamp_position * time_precision,
                            end_timestamp_position * time_precision,
                        ),
                    }
                )
    
            # 更新上一个切片的结束位置
            last_slice = current_slice
    
        # 返回偏移量列表
        return offsets
    
    
    @lru_cache
    def timestamp_ids(self, time_precision=0.02):
        """
        计算给定精度的时间戳标记的 id，并保存到最近最少使用 (LRU) 缓存中。
    
        参数:
            time_precision (`float`, `optional`, 默认为 0.02):
                从 token 到时间的转换时间精度。
        """
        return self.convert_tokens_to_ids([("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)])
    # 对 token ids 进行预处理，去除提示和时间戳等特殊 token ids
    def _preprocess_token_ids(self, token_ids, skip_special_tokens: bool = False):
        """
        Pre-process the token ids for decoding by removing the prompt tokens ids and timestamp token ids.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Typically, obtained using the `__call__` method of the tokenizer.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens from the token ids. If `True`, the prompt token ids will be
                removed.
        """
        if skip_special_tokens:
            # 获取特殊 token 的 id
            prompt_token_id = self.convert_tokens_to_ids("<|startofprev|>")
            decoder_start_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
            # 去除特殊 token ids
            token_ids = self._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)

        return token_ids

    # 过滤时间戳 ids
    def _filter_timestamp_ids(self, token_ids):
        return re.sub(self.timestamp_pat, "", token_ids)

    # 解码方法
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_offsets: bool = False,
        time_precision: float = 0.02,
        decode_with_timestamps: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    
    # 内部解码方法
    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    ) -> str:
        # 从关键字参数中弹出"use_source_tokenizer"，默认为False
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)
        # 将 token_ids 转换为特殊标记跳过后的 tokens
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # 为了避免字节级别和 unicode 混合在一起，需要分别构建添加标记和字节级别标记的字符串
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        text = "".join(sub_texts)

        if normalize:
            # 标准化文本
            clean_text = self._normalize(text)
            return clean_text
        elif basic_normalize:
            # 简单标准化文本
            clean_text = self._basic_normalize(text, remove_diacritics=remove_diacritics)
            return clean_text
        else:
            return text

    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string 复制到 WhisperTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """将 tokens（字符串）序列转换为单个字符串。"""
        # 将 tokens 组合为字符串
        text = "".join(tokens)
        # 解码字节数组转换为 utf-8 编码的字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 将词汇保存到指定目录下
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接词汇文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 拼接合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        # 拼接规范化器文件路径
        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
        )

        # 将编码器保存到词汇文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将BPE标记写入合并文件
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

        # 如果存在英语拼写规范器，则将其保存到规范器文件
        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

        return vocab_file, merge_file, normalizer_file

    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.prepare_for_tokenization复制，将GPT2更改为Whisper
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经分词或需要在前面添加空格，则在文本前面添加空格
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    @property
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template复制
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        # 如果没有定义聊天模板，则使用默认模板，并发出警告信息
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板字符串
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        # 设置前缀 tokens
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        # 获取强制 decoder 的 tokens
        forced_tokens = self.prefix_tokens[1:]
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
        return forced_decoder_ids

    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        # 调用内部的 _decode_asr 函数
        return _decode_asr(
            self,
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )

    def get_prompt_ids(self, text: str, return_tensors="np"):
        """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
        # 将提示文本转换为 IDs
        batch_encoding = self("<|startofprev|>", " " + text.strip(), add_special_tokens=False)

        # 检查是否存在特殊 tokens
        prompt_text_ids = batch_encoding["input_ids"][1:]
        special_token_id = next((x for x in prompt_text_ids if x >= self.all_special_ids[0]), None)
        if special_token_id is not None:
            token = self.convert_ids_to_tokens(special_token_id)
            raise ValueError(f"Encountered text in the prompt corresponding to disallowed special token: {token}.")

        batch_encoding.convert_to_tensors(tensor_type=return_tensors)
        return batch_encoding["input_ids"]

    @staticmethod
    def _strip_prompt(token_ids: List[int], prompt_token_id: int, decoder_start_token_id: int):
        # 检查是否存在提示
        has_prompt = isinstance(token_ids, list) and token_ids and token_ids[0] == prompt_token_id
        # 如果存在提示，则返回 decoder 的 tokens，��则返回空列表
        if has_prompt:
            if decoder_start_token_id in token_ids:
                return token_ids[token_ids.index(decoder_start_token_id) :]
            else:
                return []

        return token_ids
# 该函数用于解码自动语音识别模型的输出结果，并处理各种选项和特殊情况
def _decode_asr(tokenizer, model_outputs, *, return_timestamps, return_language, time_precision):
    # 概览:
    # - 遍历所有输出
    # - 每个输出包含多个token
    # - 每个token可以是以下类型之一:
    #   - 语言标记
    #   - 特殊标记
    #   - 时间戳标记
    #   - 文本标记
    # - 我们累积文本标记
    # - 我们根据结束时间戳进行分块
    # - 很多复杂性来自于跨度和时间戳

    # 上一个语言
    last_language = None

    # 创建新的分块
    def new_chunk():
        return {"language": last_language, "timestamp": [None, None], "text": ""}

    # 状态机欢迎您!
    chunks = []
    chunk = new_chunk()
    # 时间偏移
    time_offset = 0.0
    # 时间戳结束标记的 token ID
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
    # 先前的 tokens 和时间戳
    previous_tokens = []
    previous_token_timestamps = []
    # 是否跳过
    skip = False
    # 右侧跨度的起始点
    right_stride_start = None

    # 所有特殊 ID 的集合
    all_special_ids = set(tokenizer.all_special_ids)

    # 遍历所有输出
    if previous_tokens:
        if return_timestamps:
            # 发出警告: Whisper 未能预测结束时间戳, 可能是因为音频被截断
            logger.warning(...)
        # 当我们不使用时间戳时发生这种情况
        resolved_tokens, resolved_token_timestamps = _find_longest_common_sequence(
            previous_tokens, previous_token_timestamps
        )
        resolved_text = tokenizer.decode(resolved_tokens)
        chunk["text"] = resolved_text
        if return_timestamps == "word":
            chunk["words"] = _collate_word_timestamps(
                tokenizer, resolved_tokens, resolved_token_timestamps, last_language
            )
        chunks.append(chunk)

    # 准备和清理管道输出
    full_text = "".join(chunk["text"] for chunk in chunks)
    if return_timestamps or return_language:
        for chunk in chunks:
            if not return_timestamps:
                chunk.pop("timestamp")
            else:
                chunk["timestamp"] = tuple(chunk["timestamp"])
            if not return_language:
                chunk.pop("language")

        if return_timestamps == "word":
            new_chunks = []
            for chunk in chunks:
                new_chunks.extend(chunk["words"])
            optional = {"chunks": new_chunks}
        else:
            optional = {"chunks": chunks}
    else:
        optional = {}
    return full_text, optional


# 找到最长的公共子序列
def _find_longest_common_sequence(sequences, token_timestamp_sequences=None):
    # 这里使用的是 O(n) 的复杂度, 因为我们有一个很好的属性:
    # 总序列必须是这些子序列按顺序排列的.
    # 如果提供了token_timestamp_sequences，则以完全相同的方式拆分这些序列。
    
    # 取出sequences中的第一个序列
    left_sequence = sequences[0]
    # 计算左序列的长度
    left_length = len(left_sequence)
    # 初始化总序列为空列表
    total_sequence = []

    # 如果token_timestamp_sequences存在
    if token_timestamp_sequences:
        # 取出token_timestamp_sequences中的第一个序列
        left_token_timestamp_sequence = token_timestamp_sequences[0]
        # 初始化总的token_timestamp_sequence为空列表
        total_token_timestamp_sequence = []
    for seq_idx, right_sequence in enumerate(sequences[1:]):
        # 遍历索引和右边序列，其中右边序列从第二个开始到最后一个
        max_ = 0.0
        # 初始化最大值为0
        max_indices = (left_length, left_length, 0, 0)
        # 初始化最大匹配索引为左边长度，左边长度，0，0
        right_length = len(right_sequence)
        # 计算右边序列的长度
        for i in range(1, left_length + right_length):
            # 遍历范围为1到左边长度加右边长度
            eps = i / 10000.0
            # 定义eps为i除以10000.0，用于稍微偏向于长完美匹配

            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])
            # 分别计算左段和右段的起始索引和结束索引，以及对应的数组

            if len(left) != len(right):
                raise RuntimeError("There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference.")
            # 如果左右段长度不相等，则发出运行时错误

            matches = np.sum(left == right)
            # 计算匹配的数目
            matching = matches / i + eps
            # 计算匹配比例
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)
            # 更新最大匹配比例和对应的索引

        (left_start, left_stop, right_start, right_stop) = max_indices
        # 解构赋值最大匹配索引

        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)
        # 对左中间和右中间进行处理，更新左边序列和左边序列长度

        if token_timestamp_sequences:
            total_token_timestamp_sequence.extend(left_token_timestamp_sequence[:left_mid])
            left_token_timestamp_sequence = token_timestamp_sequences[seq_idx + 1][right_mid:]
        # 如果有token_timestamp_sequences，则进行处理，更新token序列和对应的时间戳序列

    total_sequence.extend(left_sequence)
    # 将最终左边序列加入总序列中

    if token_timestamp_sequences is None:
        return total_sequence
    # 如果没有token时间戳序列，则返回总序列
    # 如果token_timestamp_sequences列表的长度大于0
    if len(token_timestamp_sequences) > 0:
        # 将left_token_timestamp_sequence添加到total_token_timestamp_sequence中
        total_token_timestamp_sequence.extend(left_token_timestamp_sequence)
        # 返回total_sequence和total_token_timestamp_sequence
        return total_sequence, total_token_timestamp_sequence
    else:
        # 如果token_timestamp_sequences列表的长度不大于0，则返回total_sequence和空列表
        return total_sequence, []
# 将标记组合成单词及其时间戳
def _collate_word_timestamps(tokenizer, tokens, token_timestamps, language):
    # 将标记组合成单词及其对应的时间戳
    words, _, token_indices = _combine_tokens_into_words(tokenizer, tokens, language)
    # 创建包含单词及其时间戳的列表
    timings = [
        {
            "text": word,  # 单词文本
            "timestamp": (token_timestamps[indices[0]][0], token_timestamps[indices[-1]][1]),  # 单词的开始时间和结束时间
        }
        for word, indices in zip(words, token_indices)
    ]
    # 返回包含单词及其时间戳的列表
    return timings


def _combine_tokens_into_words(
    tokenizer,
    tokens: List[int],
    language: str = None,
    prepend_punctuations: str = "\"'“¡¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
):
    """
    将标记组合成单词。返回包含单词列表和标记 ID 序列的元组。
    """
    # 如果未指定语言，则使用标记器的默认语言
    if language is None:
        language = tokenizer.language
    # 如果仍未指定语言，则默认为英语
    if language is None:
        language = "english"

    # 对于不使用空格的语言，以 Unicode 分割标记
    if language in {"chinese", "japanese", "thai", "lao", "myanmar", "cantonese"}:
        # 这些语言通常不使用空格
        # 将标记根据 Unicode 分割成单词
        words, word_tokens, token_indices = _split_tokens_on_unicode(tokenizer, tokens)
    else:
        # 对于其他语言，使用空格分割标记
        words, word_tokens, token_indices = _split_tokens_on_spaces(tokenizer, tokens)

    # 合并标点符号到单词中
    _merge_punctuations(words, word_tokens, token_indices, prepend_punctuations, append_punctuations)
    # 返回包含单词列表、单词标记列表和标记索引列表的元组
    return words, word_tokens, token_indices


def _split_tokens_on_unicode(tokenizer, tokens: List[int]):
    """通过在标记解码为有效 Unicode 码点的任何位置分割标记，将标记分割成单词。"""
    # 使用标记器解码标记，并保留时间戳信息
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    # 替换字符
    replacement_char = "\ufffd"

    # 初始化列表
    words = []
    word_tokens = []
    token_indices = []
    current_tokens = []
    current_indices = []
    unicode_offset = 0

    # 遍历标记
    for token_idx, token in enumerate(tokens):
        # 将标记及其索引添加到当前列表中
        current_tokens.append(token)
        current_indices.append(token_idx)
        # 使用标记器解码当前标记，并保留时间戳信息
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)

        # 如果替换字符不在解码后的字符串中，或者替换字符出现在完整解码后的字符串中，则说明是一个单词的结束
        if (
            replacement_char not in decoded
            or decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char
        ):
            # 将单词、单词标记和标记索引添加到相应列表中
            words.append(decoded)
            word_tokens.append(current_tokens)
            token_indices.append(current_indices)
            # 重置当前列表
            current_tokens = []
            current_indices = []
            # 更新 Unicode 偏移量
            unicode_offset += len(decoded)

    # 返回单词列表、单词标记列表和标记索引列表
    return words, word_tokens, token_indices


def _split_tokens_on_spaces(tokenizer, tokens: List[int]):
    """通过在空格和标点符号标记处分割标记，将标记分割成单词。"""
    # 调用 _split_tokens_on_unicode 函数，以 Unicode 分割标记
    subwords, subword_tokens_list, subword_indices_list = _split_tokens_on_unicode(tokenizer, tokens)
    # 初始化列表
    words = []
    word_tokens = []
    token_indices = []
    # 遍历subwords、subword_tokens_list和subword_indices_list的对应元素
    for subword, subword_tokens, subword_indices in zip(subwords, subword_tokens_list, subword_indices_list):
        # 判断当前subword是否为特殊标记
        special = subword_tokens[0] >= tokenizer.eos_token_id
        # 判断subword是否以空格开头
        with_space = subword.startswith(" ")
        # 判断subword是否为标点符号
        punctuation = subword.strip() in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        # 如果满足特殊标识、以空格开头、为标点符号或者words为空的条件之一
        if special or with_space or punctuation or len(words) == 0:
            # 将当前subword添加到words列表中
            words.append(subword)
            # 将当前subword_tokens添加到word_tokens列表中
            word_tokens.append(subword_tokens)
            # 将当前subword_indices添加到token_indices列表中
            token_indices.append(subword_indices)
        else:
            # 将当前subword连接到上一个words中的子单词
            words[-1] = words[-1] + subword
            # 将当前subword_tokens扩展到上一个word_tokens中的子单词tokens中
            word_tokens[-1].extend(subword_tokens)
            # 将当前subword_indices扩展到上一个token_indices中的子单词indices中
            token_indices[-1].extend(subword_indices)

    # 返回处理后的words、word_tokens和token_indices
    return words, word_tokens, token_indices
# 合并标点符号与相邻单词
def _merge_punctuations(words, tokens, indices, prepended, appended):
    """Merges punctuation tokens with neighboring words."""
    # 在单词列表末尾添加标点符号
    i = len(words) - 2
    j = len(words) - 1
    while i >= 0:
        if words[i].startswith(" ") and words[i].strip() in prepended:  # 如果上一个单词以空格开头且在prepended列表中
            words[j] = words[i] + words[j]  # 将上一个单词和当前单词合并
            tokens[j] = tokens[i] + tokens[j]  # 将上一个单词的token和当前单词的token合并
            indices[j] = indices[i] + indices[j]  # 将上一个单词的索引和当前单词的索引合并
            words[i] = ""  # 清空上一个单词
            tokens[i] = []  # 清空上一个单词对应的token
            indices[i] = []  # 清空上一个单词的索引
        else:
            j = i
        i -= 1

    # 在单词列表开头添加标点符号
    i = 0
    j = 1
    while j < len(words):
        if not words[i].endswith(" ") and words[j] in appended:  # 如果当前单词不以空格结尾且下一个单词在appended列表中
            words[i] += words[j]  # 将当前单词和下一个单词合并
            tokens[i] += tokens[j]  # 将当前单词的token和下一个单词的token合并
            indices[i] += indices[j]  # 将当前单词的索引和下一个单词的索引合并
            words[j] = ""  # 清空下一个单词
            tokens[j] = []  # 清空下一个单词对应的token
            indices[j] = []  # 清空下一个单词的索引
        else:
            i = j
        j += 1

    # 移除现在为空的元素
    words[:] = [word for word in words if word]  # 移除为空的单词
    tokens[:] = [token for token in tokens if token]  # 移除为空的token
    indices[:] = [idx for idx in indices if idx]  # 移除为空的索引
```