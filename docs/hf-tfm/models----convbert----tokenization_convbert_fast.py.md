# `.\models\convbert\tokenization_convbert_fast.py`

```
# 指定文件编码为UTF-8
# 版权声明
# 根据Apache License, Version 2.0许可证，您可以使用此文件，但需遵守许可证规定
# 如需了解许可证的详细信息，请访问http://www.apache.org/licenses/LICENSE-2.0
"""ConvBERT的标记化类。"""
# 导入所需模块
import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

# 导入基类PreTrainedTokenizerFast和日志记录模块
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
# 导入ConvBERT的标记化器
from .tokenization_convbert import ConvBertTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇表文件的名称字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练模型对应的词汇表文件的映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/vocab.txt",
        "YituTech/conv-bert-medium-small": (
            "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/vocab.txt"
        ),
        "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/vocab.txt",
    }
}

# 定义预训练模型对应的位置嵌入大小的字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "YituTech/conv-bert-base": 512,
    "YituTech/conv-bert-medium-small": 512,
    "YituTech/conv-bert-small": 512,
}

# 定义预训练模型的初始配置字典
PRETRAINED_INIT_CONFIGURATION = {
    "YituTech/conv-bert-base": {"do_lower_case": True},
    "YituTech/conv-bert-medium-small": {"do_lower_case": True},
    "YituTech/conv-bert-small": {"do_lower_case": True},
}

# 从transformers.models.bert.tokenization_bert_fast.BertTokenizerFast中复制代码，并修改为ConvBERT相关的内容
class ConvBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个"快速"的ConvBERT标记化器（由HuggingFace的*tokenizers*库支持）。基于WordPiece。

    此标记化器继承自[`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    # 参数：
    # vocab_file (`str`):
    #     包含词汇表的文件。
    # do_lower_case (`bool`, *optional*, 默认为 `True`):
    #     在标记化时是否将输入转换为小写。
    # unk_token (`str`, *optional*, 默认为 `"[UNK]"`):
    #     未知标记。词汇表中没有的标记无法转换为 ID，而是设置为该标记。
    # sep_token (`str`, *optional*, 默认为 `"[SEP]"`):
    #     分隔标记，用于从多个序列构建序列时使用，例如用于序列分类的两个序列，或用于问题回答的文本和问题。还用作建立带有特殊标记的序列的最后一个标记。
    # pad_token (`str`, *optional*, 默认为 `"[PAD]"`):
    #     用于填充的标记，例如在对不同长度的序列进行批处理时。
    # cls_token (`str`, *optional*, 默认为 `"[CLS]"`):
    #     用于序列分类（整个序列的分类，而不是每个标记的分类）时使用的分类器标记。这是建立带有特殊标记的序列时的第一个标记。
    # mask_token (`str`, *optional*, 默认为 `"[MASK]"`):
    #     用于屏蔽值的标记。这是在训练具有屏蔽语言建模的模型时使用的标记。这是模型将尝试预测的标记。
    # clean_text (`bool`, *optional*, 默认为 `True`):
    #     在标记化之前是否清理文本，即删除所有控制字符并将所有空白替换为经典的空格。
    # tokenize_chinese_chars (`bool`, *optional*, 默认为 `True`):
    #     是否标记化中文字符。这在日语中可能应该停用（请参阅[此问题](https://github.com/huggingface/transformers/issues/328)）。
    # strip_accents (`bool`, *optional*):
    #     是否去除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值确定（就像原始 ConvBERT 中一样）。
    # wordpieces_prefix (`str`, *optional*, 默认为 `"##"`):
    #     子词的前缀。
    # """
    
    # 定义了一些文件名和映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = ConvBertTokenizer
    
    # 初始化方法
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
    # 调用父类的初始化方法，设置模型参数
    def __init__(
        self,
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
    ):
        # 调用父类的初始化方法，传入必要参数和可选参数
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

        # 从后端分词器中获取正则化器的状态信息
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查当前初始化参数与正则化器状态是否一致，若不一致则更新正则化器
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取正则化器类
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            # 更新正则化器状态
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 实例化新的正则化器
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 设置当前对象的小写参数
        self.do_lower_case = do_lower_case

    # 根据输入的 token_ids 构建带有特殊标记的模型输入
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A ConvBERT sequence has the following format:

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
        # 构建模型输入，加入特殊标记
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果有第二个序列，则加入其特殊标记
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # 根据 token_ids 构建 token 类型 ID
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ConvBERT sequence
        pair mask has the following format:

        """
        # 定义分隔符和类别标记的 ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果第二个序列的 tokens 为 None，则只返回第一个序列的 mask (全为0)
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回两个序列合并后的 mask，第一个序列对应0，第二个序列对应1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用 tokenizer 中的 model.save 方法保存词汇表文件
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件路径
        return tuple(files)
```