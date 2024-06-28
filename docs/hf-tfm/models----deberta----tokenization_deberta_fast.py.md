# `.\models\deberta\tokenization_deberta_fast.py`

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
""" Fast Tokenization class for model DeBERTa."""

import json
from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers  # 导入 pre_tokenizers 模块

from ...tokenization_utils_base import AddedToken, BatchEncoding  # 导入 tokenization_utils_base 模块中的 AddedToken 和 BatchEncoding 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入 tokenization_utils_fast 模块中的 PreTrainedTokenizerFast 类
from ...utils import logging  # 导入 utils 模块中的 logging 函数
from .tokenization_deberta import DebertaTokenizer  # 导入当前目录下的 tokenization_deberta 模块中的 DebertaTokenizer 类

logger = logging.get_logger(__name__)  # 获取当前模块的 logger 对象

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/vocab.json",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/vocab.json",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/vocab.json",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/vocab.json",
        "microsoft/deberta-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/merges.txt",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/merges.txt",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/merges.txt",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/merges.txt",
        "microsoft/deberta-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/merges.txt"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
    "microsoft/deberta-xlarge": 512,
    "microsoft/deberta-base-mnli": 512,
    "microsoft/deberta-large-mnli": 512,
    "microsoft/deberta-xlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False},
    # 预训练模型 "microsoft/deberta-base" 的初始化配置，指定 do_lower_case 为 False
}
    "microsoft/deberta-large": {"do_lower_case": False},


    # 定义一个键为 "microsoft/deberta-large" 的字典条目，其值为包含一个布尔值 False 的键 "do_lower_case"
}



class DebertaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" DeBERTa tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import DebertaTokenizerFast

    >>> tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
    >>> tokenizer("Hello world")["input_ids"]
    [1, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [1, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    # 定义常量：词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义常量：预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义常量：预训练位置嵌入的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义常量：模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    # 慢速分词器类的引用
    slow_tokenizer_class = DebertaTokenizer

    # 初始化方法，用于创建一个新的 DebertaTokenizer 对象
    def __init__(
        self,
        vocab_file=None,           # 词汇文件的路径（可选）
        merges_file=None,          # 合并文件的路径（可选）
        tokenizer_file=None,       # 分词器文件的路径（可选）
        errors="replace",          # 解码字节流时的错误处理方式，默认为替换
        bos_token="[CLS]",         # 序列起始标记（可选，默认为 "[CLS]"）
        eos_token="[SEP]",         # 序列结束标记（可选，默认为 "[SEP]"）
        sep_token="[SEP]",         # 分隔标记，用于多序列构建等情况（可选，默认为 "[SEP]"）
        cls_token="[CLS]",         # 分类器标记，用于序列分类任务（可选，默认为 "[CLS]"）
        unk_token="[UNK]",         # 未知标记，当词汇中不存在时使用（可选，默认为 "[UNK]"）
        pad_token="[PAD]",         # 填充标记，用于填充不同长度的序列（可选，默认为 "[PAD]"）
        mask_token="[MASK]",       # 掩码标记，用于掩码语言建模任务（可选，默认为 "[MASK]"）
        add_prefix_space=False,    # 是否在输入前添加空格，用于 Deberta 分词器（可选，默认为 False）
        **kwargs,                  # 其他关键字参数
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
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
        self.add_bos_token = kwargs.pop("add_bos_token", False)

        # 获取当前预处理器（pre_tokenizer）的状态，并转换为字典
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 检查预处理器是否需要更新 `add_prefix_space` 属性
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 获取预处理器类型并重新设置 `add_prefix_space` 属性
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            # 更新后的预处理器重新赋值给当前实例的 backend_tokenizer.pre_tokenizer
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 保存 add_prefix_space 属性到实例变量
        self.add_prefix_space = add_prefix_space

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        Deberta tokenizer has a special mask token to be used in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *[MASK]*.
        """
        # 如果 _mask_token 尚未设置，记录错误信息并返回 None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        # 返回 _mask_token 的字符串表示
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.
        """
        # 将 mask token 设置为包含前导空格的特殊词，确保 lstrip 为 True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # If only one sequence is provided, return it with [CLS] and [SEP] tokens
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # For sequence pairs, concatenate them with [CLS], [SEP] tokens in between
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
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
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If only one sequence is provided, return token type IDs for the first sequence only
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # For sequence pairs, return token type IDs with 0s for the first sequence and 1s for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._batch_encode_plus
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """
        Encodes a batch of inputs into token IDs, attention masks, and other relevant information.

        Args:
            *args:
                Positional arguments for encoding.
            **kwargs:
                Keyword arguments for encoding, including `is_split_into_words`.

        Returns:
            `BatchEncoding`: Encoded batch containing token IDs, attention masks, and other metadata.
        """
        is_split_into_words = kwargs.get("is_split_into_words", False)
        
        # Asserts that the tokenizer is compatible with pretokenized inputs if `add_prefix_space=True` is not set
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        # Call the superclass method to perform batch encoding
        return super()._batch_encode_plus(*args, **kwargs)
    # 对输入参数进行编码处理，并返回BatchEncoding对象
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取是否已经分词的标志位，默认为False
        is_split_into_words = kwargs.get("is_split_into_words", False)
        
        # 断言条件：如果add_prefix_space为True或者is_split_into_words为False，则通过；否则抛出异常
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        
        # 调用父类方法对输入参数进行编码处理并返回结果
        return super()._encode_plus(*args, **kwargs)

    # 从transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast中复制的方法：保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用_tokenizer.model的save方法保存模型到指定目录，并返回保存的文件名列表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        
        # 返回保存的文件名列表作为元组
        return tuple(files)
```