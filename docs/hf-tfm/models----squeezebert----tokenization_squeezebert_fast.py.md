# `.\models\squeezebert\tokenization_squeezebert_fast.py`

```py
# coding=utf-8
# 版权所有 2020 年的 SqueezeBert 作者和 HuggingFace Inc. 团队。
#
# 根据 Apache 许可 2.0 版本许可下许可;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按"原样"分发，不附带任何明示或暗示的保证或条件。
# 有关特定语言的条款，请参阅许可证。
"""SqueezeBERT 的标记化类。"""

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

# 从 transformers 库中导入必要的模块和类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_squeezebert import SqueezeBertTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义与词汇文件和标记器文件相关的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "squeezebert/squeezebert-uncased": (
            "https://huggingface.co/squeezebert/squeezebert-uncased/resolve/main/vocab.txt"
        ),
        "squeezebert/squeezebert-mnli": "https://huggingface.co/squeezebert/squeezebert-mnli/resolve/main/vocab.txt",
        "squeezebert/squeezebert-mnli-headless": (
            "https://huggingface.co/squeezebert/squeezebert-mnli-headless/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "squeezebert/squeezebert-uncased": (
            "https://huggingface.co/squeezebert/squeezebert-uncased/resolve/main/tokenizer.json"
        ),
        "squeezebert/squeezebert-mnli": (
            "https://huggingface.co/squeezebert/squeezebert-mnli/resolve/main/tokenizer.json"
        ),
        "squeezebert/squeezebert-mnli-headless": (
            "https://huggingface.co/squeezebert/squeezebert-mnli-headless/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练模型的位置嵌入尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "squeezebert/squeezebert-uncased": 512,
    "squeezebert/squeezebert-mnli": 512,
    "squeezebert/squeezebert-mnli-headless": 512,
}

# 预训练模型的初始化配置映射
PRETRAINED_INIT_CONFIGURATION = {
    "squeezebert/squeezebert-uncased": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli-headless": {"do_lower_case": True},
}

# 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 复制的 SqueezeBERT 快速标记器类
class SqueezeBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”SqueezeBERT标记器（由 HuggingFace 的 *tokenizers* 库支持）。基于 WordPiece。

    此标记器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。

    ```
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
                value for `lowercase` (as in the original SqueezeBERT).
            wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
                The prefix for subwords.
        """
    
        # 引入全局变量
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        slow_tokenizer_class = SqueezeBertTokenizer
    
        # 初始化函数，接收多个参数，包括必填和可选参数
        def __init__(
            self,
            vocab_file=None,
            tokenizer_file=None,
            do_lower_case=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            **kwargs,
        ):
        ):
        # 调用父类初始化方法，设置词汇文件、分词器文件、大小写处理、未知标记、分隔符标记、填充标记、类别标记、掩码标记、中文字符分词处理和重音处理等参数
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

        # 获取当前标准化器的状态，并转换成 JSON 格式
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查当前标准化器的设置与初始化时传入的设置是否一致，若不一致则更新标准化器
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取当前标准化器的类别，并更新设置参数
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 更新对象的小写处理设置
        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A SqueezeBERT sequence has the following format:

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
        # 构建带有特殊标记的输入序列，根据是否提供第二个序列决定是否添加第二个分隔符标记
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
    def create_seq_pair_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A SqueezeBERT sequence
        pair mask has the following format:
    
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
    
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
    
        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
    
        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s). 0 represents the first sequence, and 1 represents the second sequence.
        """
        # Define special tokens
        sep = [self.sep_token_id]  # Separator token ID
        cls = [self.cls_token_id]  # Classification token ID
    
        # If token_ids_1 is not provided, return mask for token_ids_0 only
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
    
        # Return mask including both token_ids_0 and token_ids_1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary associated with the tokenizer's model to a specified directory.
    
        Args:
            save_directory (str):
                Directory where the vocabulary files will be saved.
            filename_prefix (str, *optional*):
                Optional prefix for the saved files.
    
        Returns:
            Tuple[str]: Tuple containing the filenames of the saved vocabulary files.
        """
        # Save the vocabulary files using the tokenizer's model
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```