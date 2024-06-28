# `.\models\realm\tokenization_realm_fast.py`

```py
# coding=utf-8
# 版权 2022 年 REALM 作者和 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件根据“原样”分发，
# 没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
"""REALM 的快速分词类。"""

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers  # 导入 tokenizers 包中的 normalizers 模块

from ...tokenization_utils_base import BatchEncoding  # 导入 BatchEncoding 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入 PreTrainedTokenizerFast 类
from ...utils import PaddingStrategy, logging  # 导入 PaddingStrategy 和 logging 类
from .tokenization_realm import RealmTokenizer  # 从当前目录导入 RealmTokenizer 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}  # 定义 VOCAB_FILES_NAMES 字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/realm-cc-news-pretrained-embedder": (
            "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/vocab.txt"
        ),
        "google/realm-cc-news-pretrained-encoder": (
            "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/vocab.txt"
        ),
        "google/realm-cc-news-pretrained-scorer": (
            "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/vocab.txt"
        ),
        "google/realm-cc-news-pretrained-openqa": (
            "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/vocab.txt"
        ),
        "google/realm-orqa-nq-openqa": "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/vocab.txt",
        "google/realm-orqa-nq-reader": "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/vocab.txt",
        "google/realm-orqa-wq-openqa": "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/vocab.txt",
        "google/realm-orqa-wq-reader": "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/vocab.txt",
    },
    # 定义 PRETRAINED_VOCAB_FILES_MAP 字典，包含不同模型的预训练词汇文件 URL
}
    # 定义一个字典，存储多个模型的名称和对应的 tokenizer 文件的 URL
    "tokenizer_file": {
        # 模型 google/realm-cc-news-pretrained-embedder 的 tokenizer 文件 URL
        "google/realm-cc-news-pretrained-embedder": (
            "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/tokenizer.jsont"
        ),
        # 模型 google/realm-cc-news-pretrained-encoder 的 tokenizer 文件 URL
        "google/realm-cc-news-pretrained-encoder": (
            "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/tokenizer.json"
        ),
        # 模型 google/realm-cc-news-pretrained-scorer 的 tokenizer 文件 URL
        "google/realm-cc-news-pretrained-scorer": (
            "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/tokenizer.json"
        ),
        # 模型 google/realm-cc-news-pretrained-openqa 的 tokenizer 文件 URL
        "google/realm-cc-news-pretrained-openqa": (
            "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/tokenizer.json"
        ),
        # 模型 google/realm-orqa-nq-openqa 的 tokenizer 文件 URL
        "google/realm-orqa-nq-openqa": (
            "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/tokenizer.json"
        ),
        # 模型 google/realm-orqa-nq-reader 的 tokenizer 文件 URL
        "google/realm-orqa-nq-reader": (
            "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/tokenizer.json"
        ),
        # 模型 google/realm-orqa-wq-openqa 的 tokenizer 文件 URL
        "google/realm-orqa-wq-openqa": (
            "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/tokenizer.json"
        ),
        # 模型 google/realm-orqa-wq-reader 的 tokenizer 文件 URL
        "google/realm-orqa-wq-reader": (
            "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/tokenizer.json"
        ),
    },
}

# 定义预训练模型的位置嵌入大小字典，每个模型名称映射到其对应的位置嵌入大小（均为512）
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/realm-cc-news-pretrained-embedder": 512,
    "google/realm-cc-news-pretrained-encoder": 512,
    "google/realm-cc-news-pretrained-scorer": 512,
    "google/realm-cc-news-pretrained-openqa": 512,
    "google/realm-orqa-nq-openqa": 512,
    "google/realm-orqa-nq-reader": 512,
    "google/realm-orqa-wq-openqa": 512,
    "google/realm-orqa-wq-reader": 512,
}

# 定义预训练模型初始化配置字典，每个模型名称映射到其对应的初始化配置字典，这里只设置了一个通用项 do_lower_case=True
PRETRAINED_INIT_CONFIGURATION = {
    "google/realm-cc-news-pretrained-embedder": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-encoder": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-scorer": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-openqa": {"do_lower_case": True},
    "google/realm-orqa-nq-openqa": {"do_lower_case": True},
    "google/realm-orqa-nq-reader": {"do_lower_case": True},
    "google/realm-orqa-wq-openqa": {"do_lower_case": True},
    "google/realm-orqa-wq-reader": {"do_lower_case": True},
}


class RealmTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" REALM tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    [`RealmTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    # 定义预置的词汇文件名列表，通常包含不同语言的词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型的词汇文件映射，用于加载不同预训练模型的词汇文件
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义预训练模型的初始化配置，可能包含模型的特定参数配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 定义预训练模型的最大输入长度限制，通常用于限制输入序列的最大长度
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义慢速分词器类，通常用于特定语言或对分词速度要求不高的场景
    slow_tokenizer_class = RealmTokenizer

    # 初始化函数，用于创建一个新的实例对象
    def __init__(
        self,
        vocab_file=None,  # 词汇文件路径，用于加载模型的词汇
        tokenizer_file=None,  # 分词器文件路径，用于加载保存的分词器模型
        do_lower_case=True,  # 是否将输入文本转为小写
        unk_token="[UNK]",  # 未知标记，用于词汇中未出现的词的表示
        sep_token="[SEP]",  # 分隔符标记，用于组合多个序列的标记
        pad_token="[PAD]",  # 填充标记，用于批处理不同长度的序列
        cls_token="[CLS]",  # 分类器标记，用于序列分类任务的开始标记
        mask_token="[MASK]",  # 掩码标记，用于掩码语言模型训练中的预测
        tokenize_chinese_chars=True,  # 是否分词中文字符
        strip_accents=None,  # 是否去除所有的重音符号
        **kwargs,  # 其他可选参数，用于灵活设置
        ):
        # 调用父类的构造函数初始化对象，设置各种参数
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

        # 从后端分词器获取规范化器的状态并反序列化
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查当前对象的参数是否与规范化器的状态匹配，如果不匹配则更新规范化器
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

        # 设置对象的小写处理标志位
        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A REALM sequence has the following format:

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
        # 构建带有特殊标记的模型输入序列
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A REALM sequence
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
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define the separator and classification token IDs
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If token_ids_1 is None, return a mask with zeros for only the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Otherwise, concatenate masks for both sequences where the first sequence has 0s and the second has 1s
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's vocabulary to the specified directory.

        Args:
            save_directory (str):
                Directory path where the vocabulary will be saved.
            filename_prefix (str, *optional*):
                Optional prefix for the saved vocabulary files.

        Returns:
            Tuple[str]: Tuple containing the paths to the saved files.
        """
        # Save the tokenizer's model (vocabulary) to the specified directory with an optional filename prefix
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        
        # Return a tuple of file paths that were saved
        return tuple(files)
```