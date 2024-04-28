# `.\transformers\models\realm\tokenization_realm_fast.py`

```
# 导入所需模块
import json
from typing import List, Optional, Tuple

# 导入 tokenizers 库中的 normalizers 模块
from tokenizers import normalizers

# 导入所需的基础类和函数
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, logging

# 导入 REALM 的 tokenization_realm 模块中的 RealmTokenizer 类
from .tokenization_realm import RealmTokenizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练词汇文件的映射表
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
    # 定义了一个字典，用于存储各个模型的 tokenizer 文件的 URL 地址
    "tokenizer_file": {
        # Realm-CC-News 预训练 Embedder 模型的 tokenizer 文件 URL 地址
        "google/realm-cc-news-pretrained-embedder": (
            "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/tokenizer.jsont"
        ),
        # Realm-CC-News 预训练 Encoder 模型的 tokenizer 文件 URL 地址
        "google/realm-cc-news-pretrained-encoder": (
            "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/tokenizer.json"
        ),
        # Realm-CC-News 预训练 Scorer 模型的 tokenizer 文件 URL 地址
        "google/realm-cc-news-pretrained-scorer": (
            "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/tokenizer.json"
        ),
        # Realm-CC-News 预训练 OpenQA 模型的 tokenizer 文件 URL 地址
        "google/realm-cc-news-pretrained-openqa": (
            "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/tokenizer.json"
        ),
        # Realm-ORQA-NQ 预训练 OpenQA 模型的 tokenizer 文件 URL 地址
        "google/realm-orqa-nq-openqa": (
            "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/tokenizer.json"
        ),
        # Realm-ORQA-NQ 预训练 Reader 模型的 tokenizer 文件 URL 地址
        "google/realm-orqa-nq-reader": (
            "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/tokenizer.json"
        ),
        # Realm-ORQA-WQ 预训练 OpenQA 模型的 tokenizer 文件 URL 地址
        "google/realm-orqa-wq-openqa": (
            "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/tokenizer.json"
        ),
        # Realm-ORQA-WQ 预训练 Reader 模型的 tokenizer 文件 URL 地址
        "google/realm-orqa-wq-reader": (
            "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/tokenizer.json"
        ),
    },
# 定义了一个字典，存储不同预训练模型的位置编码大小
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

# 定义了一个字典，存储不同预训练模型的初始化配置
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

# 定义了一个 RealmTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
# 该类用于构造一个 REALM 模型的快速分词器
# 它与 BertTokenizerFast 类似，实现了标点符号分割和词块分词的端到端tokenization
class RealmTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" REALM tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    [`RealmTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    Args:
        vocab_file (`str`):
            包含词汇表的文件名。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            在进行分词时，是否将输入转换为小写。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。不在词汇表中的标记无法转换为 ID，并将设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，用于从多个序列构建序列，例如，用于序列分类或问题回答中的文本和问题。也用作使用特殊标记构建的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，在将不同长度的序列拼接在一起时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，用于进行序列分类（整个序列的分类，而不是每个标记的分类）。使用特殊标记构建时，它是序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于掩码值的标记。这是在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
        clean_text (`bool`, *optional*, defaults to `True`):
            在分词之前是否清理文本，通过删除任何控制字符并将所有空格替换为标准空格。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行分词。这对日语可能应该被禁用（参见此问题）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值确定（与原始 BERT 中相同）。
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            用于子词的前缀。

    vocab_files_names = VOCAB_FILES_NAMES 
    # 词汇表文件名列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练词汇文件映射
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 预训练初始化配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 预训练位置嵌入大小
    slow_tokenizer_class = RealmTokenizer
    # 慢速分词器类

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
        # 调用父类构造函数初始化对象
        super().__init__(
            vocab_file,  # 词汇表文件路径
            tokenizer_file=tokenizer_file,  # 分词器文件路径
            do_lower_case=do_lower_case,  # 是否转换为小写
            unk_token=unk_token,  # 未知标记
            sep_token=sep_token,  # 分隔标记
            pad_token=pad_token,  # 填充标记
            cls_token=cls_token,  # 类标记
            mask_token=mask_token,  # 掩码标记
            tokenize_chinese_chars=tokenize_chinese_chars,  # 是否分词中文字符
            strip_accents=strip_accents,  # 是否去除重音符号
            **kwargs,  # 其他关键字参数
        )

        # 从后端分词器中加载规范化器状态
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查规范化器状态是否与当前设置一致，如果不一致，则重新设置规范化器
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取规范化器类
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            # 更新规范化器状态
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 使用更新后的状态重新设置后端分词器的规范化器
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 保存当前是否转换为小写的设置
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
        # 构建带有特殊标记的模型输入
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果存在第二个序列，添加第二个序列及其特殊标记
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
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
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 分隔符标记，用于表示两个序列之间的分隔
        sep = [self.sep_token_id]
        # 分类标记，用于表示序列的开头
        cls = [self.cls_token_id]
        # 如果第二个序列为空，则返回只有第一个序列的部分掩码（全为0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回两个序列组成的掩码，第一个序列为0，第二个序列为1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 使用 Tokenizer 模型保存词汇表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名元组
        return tuple(files)
```