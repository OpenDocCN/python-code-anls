# `.\models\distilbert\tokenization_distilbert_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""DistilBERT 的 Tokenization 类。"""

# 导入必要的库
import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

# 导入其他模块
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_distilbert import DistilBertTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/vocab.txt",
        "distilbert-base-uncased-distilled-squad": (
            "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/vocab.txt"
        ),
        "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/vocab.txt",
        "distilbert-base-cased-distilled-squad": (
            "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/vocab.txt"
        ),
        "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/vocab.txt",
        "distilbert-base-multilingual-cased": (
            "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/tokenizer.json",
        "distilbert-base-uncased-distilled-squad": (
            "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/tokenizer.json"
        ),
        "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/tokenizer.json",
        "distilbert-base-cased-distilled-squad": (
            "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json"
        ),
        "distilbert-base-german-cased": (
            "https://huggingface.co/distilbert-base-german-cased/resolve/main/tokenizer.json"
        ),
        "distilbert-base-multilingual-cased": (
            "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "distilbert-base-uncased": 512,
    "distilbert-base-uncased-distilled-squad": 512,
    "distilbert-base-cased": 512,
    "distilbert-base-cased-distilled-squad": 512,
    # 键为模型名称，值为模型的最大输入长度
    "distilbert-base-german-cased": 512,
    "distilbert-base-multilingual-cased": 512,
# 预训练模型的初始化配置，包含了不同模型的大小写设置
PRETRAINED_INIT_CONFIGURATION = {
    "distilbert-base-uncased": {"do_lower_case": True},
    "distilbert-base-uncased-distilled-squad": {"do_lower_case": True},
    "distilbert-base-cased": {"do_lower_case": False},
    "distilbert-base-cased-distilled-squad": {"do_lower_case": False},
    "distilbert-base-german-cased": {"do_lower_case": False},
    "distilbert-base-multilingual-cased": {"do_lower_case": False},
}

# 定义一个 DistilBERT 的快速分词器类，继承自 PreTrainedTokenizerFast
class DistilBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”DistilBERT分词器（由HuggingFace的*tokenizers*库支持）。基于WordPiece。

    这个分词器继承自[`PreTrainedTokenizerFast`]，其中包含了大部分主要方法。用户应该参考这个超类以获取更多关于这些方法的信息。
    Args:
        vocab_file (`str`):
            包含词汇表的文件。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            在标记化时是否将输入转换为小写。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。词汇表中不存在的标记无法转换为 ID，并将被设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，在从多个序列构建序列时使用，例如用于序列分类的两个序列或用于文本和问题的问答。它也用作使用特殊标记构建的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，例如在批处理不同长度的序列时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，用于进行序列分类（整个序列的分类而不是每个标记的分类）。它是使用特殊标记构建的序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于屏蔽值的标记。在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
        clean_text (`bool`, *optional*, defaults to `True`):
            在标记化之前是否清理文本，通过删除所有控制字符并将所有空格替换为经典空格。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否标记化中文字符。这可能应该在日语中停用（参见[此问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由`lowercase`的值确定（与原始BERT相同）。
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            子词的前缀。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = DistilBertTokenizer

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
    # 调用父类的构造函数，初始化 BertTokenizerFast 对象
    ):
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

        # 获取当前正则化器的状态
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查当前正则化器的设置是否与传入参数一致，如果不一致则更新正则化器
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

        # 设置对象的小写参数
        self.do_lower_case = do_lower_case

    # 从特殊标记构建输入序列
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
            `List[int`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 构建包含特殊标记的输入序列
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # 从序列创建 token 类型 ID
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个方法，用于生成用于序列对分类任务的掩码。BERT序列对掩码的格式如下：
    # 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    # | 第一个序列 | 第二个序列 |
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        # 分隔符标记
        sep = [self.sep_token_id]
        # 分类标记
        cls = [self.cls_token_id]
        # 如果第二个序列为空，则只返回掩码的第一部分（全为0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回第一个序列和第二个序列的掩码
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 从transformers.models.bert.tokenization_bert_fast.BertTokenizerFast中复制的方法，用于保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用_tokenizer.model.save方法保存词汇表文件
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名
        return tuple(files)
```