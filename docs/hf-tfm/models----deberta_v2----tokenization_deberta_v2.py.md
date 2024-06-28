# `.\models\deberta_v2\tokenization_deberta_v2.py`

```py
# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
"""
Tokenization class for model DeBERTa.
"""

import os  # 导入标准库os，用于处理操作系统相关功能
import unicodedata  # 导入unicodedata库，用于Unicode字符数据库的访问
from typing import Any, Dict, List, Optional, Tuple  # 导入类型提示相关的库

import sentencepiece as sp  # 导入sentencepiece库，用于分词模型的处理

from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入自定义模块中的类和函数
from ...utils import logging  # 从自定义模块中导入logging模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# 预定义的词汇文件映射，指定不同预训练模型的SentencePiece模型文件的下载链接
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model"
        ),
        "microsoft/deberta-v2-xxlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model"
        ),
    }
}

# 预定义的位置嵌入大小映射，指定不同预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-v2-xlarge": 512,
    "microsoft/deberta-v2-xxlarge": 512,
    "microsoft/deberta-v2-xlarge-mnli": 512,
    "microsoft/deberta-v2-xxlarge-mnli": 512,
}

# 预定义的初始化配置映射，指定不同预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v2-xlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xlarge-mnli": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge-mnli": {"do_lower_case": False},
}

# 词汇文件名称映射，指定模型的SentencePiece模型文件名称
VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}


class DebertaV2Tokenizer(PreTrainedTokenizer):
    r"""
    Constructs a DeBERTa-v2 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件名称映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置预训练模型的词汇文件映射
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION  # 设置预训练模型的初始化配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 设置模型的最大输入大小

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        split_by_punct=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize a DebertaV2Tokenizer with essential parameters.

        Args:
            vocab_file (str): The vocabulary file path.
            do_lower_case (bool): Whether to convert tokens to lowercase.
            split_by_punct (bool): Whether to split tokens by punctuation.
            bos_token (str): Beginning of sequence token.
            eos_token (str): End of sequence token.
            unk_token (str): Token for unknown or unrecognized tokens.
            sep_token (str): Separator token.
            pad_token (str): Token used for padding sequences.
            cls_token (str): Classification token.
            mask_token (str): Mask token for masked language modeling.
            sp_model_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments for SentencePiece model.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.vocab_file = vocab_file  # 设置词汇文件路径
        self.do_lower_case = do_lower_case  # 设置是否将词汇转换为小写
        self.split_by_punct = split_by_punct  # 设置是否按标点符号分割词汇
        self.sp_model_kwargs = sp_model_kwargs if sp_model_kwargs is not None else {}  # 设置SentencePiece模型的额外参数
    ) -> None:
        # 初始化一个空字典作为分词模型的参数，如果没有指定则使用空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 检查给定的词汇文件路径是否是一个文件，如果不是则抛出数值错误异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        # 设置是否小写化文本的标志
        self.do_lower_case = do_lower_case
        # 设置是否通过标点符号分割的标志
        self.split_by_punct = split_by_punct
        # 设置词汇文件路径
        self.vocab_file = vocab_file

        # 使用SPMTokenizer初始化分词器，传入词汇文件路径、None作为模型路径、是否通过标点符号分割的标志、以及分词模型参数字典
        self._tokenizer = SPMTokenizer(
            vocab_file, None, split_by_punct=split_by_punct, sp_model_kwargs=self.sp_model_kwargs
        )

        # 如果unk_token是字符串类型，则创建一个AddedToken对象，标记为特殊且已规范化；否则直接使用unk_token
        unk_token = AddedToken(unk_token, normalized=True, special=True) if isinstance(unk_token, str) else unk_token

        # 调用父类的初始化方法，设置分词器的各种特殊标记以及其他关键字参数
        super().__init__(
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            split_by_punct=split_by_punct,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        # 将特殊标记列表赋值给分词器的特殊标记属性
        self._tokenizer.special_tokens = self.all_special_tokens

    @property
    def vocab_size(self):
        # 返回分词器词汇表的大小（词汇表的长度）
        return len(self.vocab)

    @property
    def vocab(self):
        # 返回分词器的词汇表
        return self._tokenizer.vocab

    def get_vocab(self):
        # 获取分词器的完整词汇表，包括额外添加的词汇
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        # 如果设定为小写化，则将输入文本转换为小写
        if self.do_lower_case:
            text = text.lower()
        # 调用分词器的tokenize方法，将文本分词为字符串列表（token列表）
        return self._tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用分词器的spm对象将token（字符串）转换为对应的id（整数）
        return self._tokenizer.spm.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用分词器的spm对象将id（整数）转换为对应的token（字符串），如果id超出词汇表大小，则返回unk_token
        return self._tokenizer.spm.IdToPiece(index) if index < self.vocab_size else self.unk_token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 使用分词器的decode方法将token序列（字符串列表）转换为单个字符串
        return self._tokenizer.decode(tokens)
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        从序列或序列对中构建模型输入，用于序列分类任务，通过连接和添加特殊标记。DeBERTa 序列的格式如下：

        - 单个序列：[CLS] X [SEP]
        - 序列对：[CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对输入。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """

        if token_ids_1 is None:
            # 返回只含有一个序列的特殊标记的输入列表
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回包含序列对的特殊标记的输入列表
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        从没有添加特殊标记的标记列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 或 `encode_plus` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对输入。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经格式化为模型所需的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，取值为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            # 返回包含序列对特殊标记掩码的列表
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # 返回只包含单个序列特殊标记掩码的列表
        return [1] + ([0] * len(token_ids_0)) + [1]
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

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
        # Define separator and classification token IDs
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If only one sequence is provided
        if token_ids_1 is None:
            # Return token type IDs for single sequence (all 0s)
            return len(cls + token_ids_0 + sep) * [0]
        
        # Return token type IDs for two sequences (0s for first sequence, 1s for second sequence)
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # Extract 'add_prefix_space' from kwargs
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        
        # Add prefix space if required
        if is_split_into_words or add_prefix_space:
            text = " " + text
        
        # Return text and remaining kwargs
        return (text, kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Save vocabulary using the underlying tokenizer's method
        return self._tokenizer.save_pretrained(save_directory, filename_prefix=filename_prefix)
    r"""
    Constructs a tokenizer based on [SentencePiece](https://github.com/google/sentencepiece).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    def __init__(
        self, vocab_file, special_tokens, split_by_punct=False, sp_model_kwargs: Optional[Dict[str, Any]] = None
    ):
        # 是否按标点符号进行分割
        self.split_by_punct = split_by_punct
        # 词汇文件路径
        self.vocab_file = vocab_file
        # SentencePiece 参数，如果未提供则为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 使用给定的参数初始化 SentencePieceProcessor 对象
        spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        # 检查词汇文件是否存在，不存在则抛出 FileNotFoundError 异常
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"{vocab_file} does not exist!")
        # 加载词汇文件到 SentencePieceProcessor 对象
        spm.load(vocab_file)
        # 获取 BPE 词汇表大小
        bpe_vocab_size = spm.GetPieceSize()
        # 构建词汇映射表
        self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
        # 根据编号获取词汇表
        self.ids_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        # 设置特殊标记（未使用的代码段）
        # self.vocab['[PAD]'] = 0
        # self.vocab['[CLS]'] = 1
        # self.vocab['[SEP]'] = 2
        # self.vocab['[UNK]'] = 3

        # 保存 SentencePieceProcessor 对象和特殊标记
        self.spm = spm
        self.special_tokens = special_tokens

    def __getstate__(self):
        # 复制当前对象的状态，但将 spm 属性置为 None
        state = self.__dict__.copy()
        state["spm"] = None
        return state

    def __setstate__(self, d):
        # 恢复对象状态
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新初始化 SentencePieceProcessor 对象并加载词汇文件
        self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        self.spm.Load(self.vocab_file)

    def tokenize(self, text):
        # 使用 SentencePiece 对文本进行分词
        return self._encode_as_pieces(text)

    def convert_ids_to_tokens(self, ids):
        # 将编号转换为对应的标记
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    # 解码给定的 token 序列成原始文本。如果 raw_text 为 None，则根据 tokens 进行解码；否则根据 raw_text 进行解码。
    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        if raw_text is None:
            current_sub_tokens = []  # 存储当前正在处理的子 token 序列
            out_string = ""  # 存储最终解码的文本字符串
            prev_is_special = False  # 标记前一个 token 是否为特殊 token
            for token in tokens:
                # 如果 token 是特殊 token，则不使用 sentencepiece 模型解码
                if token in self.special_tokens:
                    if not prev_is_special:
                        out_string += " "  # 如果前一个 token 不是特殊 token，则添加空格分隔
                    out_string += self.spm.decode_pieces(current_sub_tokens) + token  # 解码当前子 token 序列并添加当前 token
                    prev_is_special = True
                    current_sub_tokens = []  # 清空当前子 token 序列，准备处理下一个特殊 token
                else:
                    current_sub_tokens.append(token)  # 将 token 添加到当前子 token 序列中
                    prev_is_special = False
            out_string += self.spm.decode_pieces(current_sub_tokens)  # 解码剩余的子 token 序列并添加到最终文本中
            return out_string.strip()  # 返回去除首尾空格的最终文本
        else:
            words = self.split_to_words(raw_text)  # 根据原始文本分割成单词列表
            word_tokens = [self.tokenize(w) for w in words]  # 对每个单词进行分词得到 token 序列
            token2words = [0] * len(tokens)  # 创建一个与 tokens 等长的列表，用于映射 token 到单词索引
            tid = 0
            for i, w in enumerate(word_tokens):
                for k, t in enumerate(w):
                    token2words[tid] = i  # 将 token 的索引映射到对应的单词索引
                    tid += 1
            word_start = token2words[start]  # 获取起始 token 对应的单词索引
            word_end = token2words[end] if end < len(tokens) else len(words)  # 获取结束 token 对应的单词索引，如果超出 tokens 则取单词列表的末尾
            text = "".join(words[word_start:word_end])  # 根据单词索引拼接原始文本
            return text  # 返回拼接后的文本

    # 添加特殊 token 到 tokenizer 中，如果 token 不存在于特殊 token 列表中，则添加，并更新词汇表和 id 到 token 的映射
    def add_special_token(self, token):
        if token not in self.special_tokens:
            self.special_tokens.append(token)  # 将新的特殊 token 添加到列表中
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) - 1  # 将新的 token 添加到词汇表中
                self.ids_to_tokens.append(token)  # 更新 id 到 token 的映射
        return self.id(token)  # 返回特殊 token 对应的 id

    # 判断 token 是否为整个单词的一部分。如果 is_bos 为 True，则始终返回 True；否则根据 token 的首字符判断是否为单词的一部分。
    def part_of_whole_word(self, token, is_bos=False):
        logger.warning_once(
            "The `DebertaTokenizer.part_of_whole_word` method is deprecated and will be removed in `transformers==4.35`"
        )
        if is_bos:
            return True
        if (
            len(token) == 1
            and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0]))
        ) or token in self.special_tokens:
            return False

        word_start = b"\xe2\x96\x81".decode("utf-8")
        return not token.startswith(word_start)  # 判断 token 是否以词的起始字符开头

    # 返回填充 token
    def pad(self):
        return "[PAD]"

    # 返回文本的开头 token
    def bos(self):
        return "[CLS]"

    # 返回文本的结尾 token
    def eos(self):
        return "[SEP]"

    # 返回未知 token
    def unk(self):
        return "[UNK]"

    # 返回掩码 token
    def mask(self):
        return "[MASK]"

    # 根据 id 返回对应的 token
    def sym(self, id):
        return self.ids_to_tokens[id]

    # 根据 token 返回对应的 id，如果 token 不在词汇表中则返回默认 id 1
    def id(self, sym):
        logger.warning_once(
            "The `DebertaTokenizer.id` method is deprecated and will be removed in `transformers==4.35`"
        )
        return self.vocab[sym] if sym in self.vocab else 1
    # 将输入文本转换为Unicode格式
    def _encode_as_pieces(self, text):
        text = convert_to_unicode(text)
        
        # 如果设置了按标点符号分割，则在文本上运行标点符号分割
        if self.split_by_punct:
            words = self._run_split_on_punc(text)
            # 对每个分割后的单词进行SPM编码，并转换为字符串列表
            pieces = [self.spm.encode(w, out_type=str) for w in words]
            # 展平嵌套列表，将编码后的片段放入一个列表中
            return [p for w in pieces for p in w]
        else:
            # 否则直接对整个文本进行SPM编码
            return self.spm.encode(text, out_type=str)

    # 将文本分割成单词
    def split_to_words(self, text):
        pieces = self._encode_as_pieces(text)
        # 定义用于标记单词开始的特殊字符
        word_start = b"\xe2\x96\x81".decode("utf-8")
        words = []
        offset = 0
        prev_end = 0
        
        # 遍历编码后的片段
        for i, p in enumerate(pieces):
            # 如果片段以单词开始字符开头
            if p.startswith(word_start):
                # 如果当前偏移量大于上一个单词结束的位置
                if offset > prev_end:
                    # 将上一个单词的内容添加到单词列表中
                    words.append(text[prev_end:offset])
                prev_end = offset
                # 移除单词开始字符，获取真正的单词内容
                w = p.replace(word_start, "")
            else:
                w = p
            
            try:
                # 在文本中查找当前单词的起始位置
                s = text.index(w, offset)
                pn = ""
                k = i + 1
                # 查找下一个非空白片段
                while k < len(pieces):
                    pn = pieces[k].replace(word_start, "")
                    if len(pn) > 0:
                        break
                    k += 1
                
                # 如果下一个片段非空且在当前单词范围内，则增加偏移量
                if len(pn) > 0 and pn in text[offset:s]:
                    offset = offset + 1
                else:
                    offset = s + len(w)
            except Exception:
                offset = offset + 1
        
        # 添加最后一个单词到单词列表中
        if prev_end < offset:
            words.append(text[prev_end:offset])
        
        return words

    # 在文本上运行标点符号分割
    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        
        # 遍历文本中的每个字符
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号，则开始一个新单词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 否则将字符添加到当前单词的最后一个片段中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        
        # 将分割后的列表中的子列表连接成字符串，并返回结果
        return ["".join(x) for x in output]

    # 将当前模型保存到指定路径下
    def save_pretrained(self, path: str, filename_prefix: str = None):
        # 获取保存的文件名
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        
        # 拼接保存文件的完整路径
        full_path = os.path.join(path, filename)
        
        # 将序列化后的模型写入文件
        with open(full_path, "wb") as fs:
            fs.write(self.spm.serialized_model_proto())
        
        # 返回保存的文件路径
        return (full_path,)
# 检查字符是否为空白字符
def _is_whitespace(char):
    # 如果字符是空格、制表符、换行符或回车符，则返回 True
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    # 如果分类是 Zs（空格分隔符），则返回 True
    if cat == "Zs":
        return True
    # 否则返回 False
    return False


# 检查字符是否为控制字符
def _is_control(char):
    # 如果字符是制表符、换行符或回车符，则返回 False
    if char == "\t" or char == "\n" or char == "\r":
        return False
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    # 如果分类以 C 开头（控制字符），则返回 True
    if cat.startswith("C"):
        return True
    # 否则返回 False
    return False


# 检查字符是否为标点符号
def _is_punctuation(char):
    # 获取字符的 Unicode 码点
    cp = ord(char)
    # 检查是否在 ASCII 范围内的非字母/数字字符，认定为标点符号
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    # 如果分类以 P 开头（标点字符），则返回 True
    if cat.startswith("P"):
        return True
    # 否则返回 False
    return False


# 将文本转换为 Unicode 编码（如果尚未）
def convert_to_unicode(text):
    # 如果输入已经是字符串，则直接返回
    if isinstance(text, str):
        return text
    # 如果输入是字节流，则用 UTF-8 解码为字符串并忽略错误
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    # 如果输入既不是字符串也不是字节流，则引发异常
    else:
        raise ValueError(f"Unsupported string type: {type(text)}")
```