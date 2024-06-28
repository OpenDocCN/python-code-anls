# `.\models\deprecated\retribert\tokenization_retribert_fast.py`

```py
# coding=utf-8
# 版权所有 2018 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本进行许可；
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件按“原样”分发，不提供任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
"""RetriBERT 的分词类。"""

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

# 导入预训练的 tokenizer
from ....tokenization_utils_fast import PreTrainedTokenizerFast
from ....utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件和 tokenizer 文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型对应的词汇文件和 tokenizer 文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "yjernite/retribert-base-uncased": (
            "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "yjernite/retribert-base-uncased": (
            "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练模型对应的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "yjernite/retribert-base-uncased": 512,
}

# 预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "yjernite/retribert-base-uncased": {"do_lower_case": True},
}


class RetriBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”RetriBERT 分词器（基于 HuggingFace 的 *tokenizers* 库）。

    [`RetriBertTokenizerFast`] 与 [`BertTokenizerFast`] 相同，并且支持端到端的分词：标点符号拆分和 wordpiece。

    此分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类获取有关这些方法的更多信息。
    """
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """
    # 以下代码段是变量声明，用于配置和初始化BERT类型的分词器的各种设置和参数

    # 文件名列表，指定了与模型相关的词汇表文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇表文件映射，指定了不同预训练模型的词汇表文件路径
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型输入的最大长度，指定了不同预训练模型的最大输入长度
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 预训练模型的初始化配置，包含了不同预训练模型的初始化参数
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 用于慢速分词器的类，指定了用于BERT类型模型的慢速分词器类
    slow_tokenizer_class = RetriBertTokenizer
    # 模型输入名称列表，指定了模型的输入ID和注意力掩码
    model_input_names = ["input_ids", "attention_mask"]

    # 以下代码段的功能与transformers库中的BertTokenizerFast.__init__方法相同，但未完全展示
    # 初始化方法，用于创建一个新的实例对象
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件路径，默认为None
        tokenizer_file=None,  # 分词器文件路径，默认为None
        do_lower_case=True,  # 是否将输入文本转换为小写，默认为True
        unk_token="[UNK]",  # 未知标记的字符串表示，默认为"[UNK]"
        sep_token="[SEP]",  # 分隔标记的字符串表示，默认为"[SEP]"
        pad_token="[PAD]",  # 填充标记的字符串表示，默认为"[PAD]"
        cls_token="[CLS]",  # 类别标记的字符串表示，默认为"[CLS]"
        mask_token="[MASK]",  # 掩码标记的字符串表示，默认为"[MASK]"
        tokenize_chinese_chars=True,  # 是否分词中文字符，默认为True
        strip_accents=None,  # 是否去除重音，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # 获取后端分词器的正常化状态
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查正常化状态是否与当前初始化参数匹配，若不匹配则更新分词器的正常化类
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 设置实例对象的小写处理状态
        self.do_lower_case = do_lower_case

    # 从给定的token_ids_0和可选的token_ids_1构建包含特殊标记的模型输入序列，用于序列分类任务
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 构建包含特殊标记的输入序列，以用于模型输入
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 若存在第二个token_ids_1，则构建双序列的输入格式
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # 根据token_ids_0和可选的token_ids_1创建token类型ID序列
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_bert_seq_classification_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of token IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of token IDs for the second sequence in sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define the separator and classification tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If there is no second sequence, return a mask with all zeros for the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Return a mask that identifies the token type IDs for both sequences
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's vocabulary to files in the specified directory.

        Args:
            save_directory (str):
                Directory where vocabulary files will be saved.
            filename_prefix (str, *optional*):
                Optional prefix for the saved vocabulary filenames.

        Returns:
            Tuple[str]: Tuple containing the filenames of the saved vocabulary files.
        """
        # Save the vocabulary files using the underlying tokenizer model
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```