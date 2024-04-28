# `.\models\layoutlm\tokenization_layoutlm_fast.py`

```py
# coding=utf-8
# 版本声明和作者授权声明
# 指定代码文件的编码格式和版权信息
# 这段代码是为 LayoutLM 模型设计的标记化类

import json  # 导入 json 模块
from typing import List, Optional, Tuple  # 导入类型提示模块

from tokenizers import normalizers  # 从 tokenizers 模块中导入 normalizers 模块

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 从上级目录的 tokenization_utils_fast 模块中导入 PreTrainedTokenizerFast 类
from ...utils import logging  # 从上级目录的 utils 模块中导入 logging 模块
from .tokenization_layoutlm import LayoutLMTokenizer  # 从当前目录的 tokenization_layoutlm 模块中导入 LayoutLMTokenizer 类

logger = logging.get_logger(__name__)  # 获取名为 __name__ 的 logger 对象

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}  # 词汇文件名和 tokenizer 文件名

PRETRAINED_VOCAB_FILES_MAP = {  # 预先训练的词汇文件映射
    "vocab_file": {
        "microsoft/layoutlm-base-uncased": (
            "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/vocab.txt"  # 预先训练的词汇文件的 URL
        ),
        "microsoft/layoutlm-large-uncased": (
            "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/vocab.txt"  # 预先训练的词汇文件的 URL
        ),
    },
    "tokenizer_file": {
        "microsoft/layoutlm-base-uncased": (
            "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/tokenizer.json"  # 预先训练的 tokenizer 文件的 URL
        ),
        "microsoft/layoutlm-large-uncased": (
            "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/tokenizer.json"  # 预先训练的 tokenizer 文件的 URL
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {  # 预先训练的位置嵌入大小
    "microsoft/layoutlm-base-uncased": 512,  # 模型 layoutlm-base-uncased 的位置嵌入大小
    "microsoft/layoutlm-large-uncased": 512,  # 模型 layoutlm-large-uncased 的位置嵌入大小
}

PRETRAINED_INIT_CONFIGURATION = {  # 预先训练的初始化配置
    "microsoft/layoutlm-base-uncased": {"do_lower_case": True},  # 模型 layoutlm-base-uncased 的初始化配置
    "microsoft/layoutlm-large-uncased": {"do_lower_case": True},  # 模型 layoutlm-large-uncased 的初始化配置
}


# 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 中复制来的 LayoutLMTokenizerFast 类
class LayoutLMTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" LayoutLM tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary. 词汇表文件的路径
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing. 是否在进行标记化时将输入转换为小写
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. 未知标记。不在词汇表中的标记不能转换为ID，会被替换成这个标记
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens. 分隔符标记，用于构建来自多个序列的序列，例如用于序列分类的两个序列，或者用于问题回答中的文本和问题。还用作使用特殊标记构建的序列的最后一个标记
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths. 用于填充的标记，例如在对不同长度的序列进行批处理时使用
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens. 用于进行序列分类（整个序列的分类而不是按标记的分类）时使用的分类器标记。使用特殊标记构建序列时，它是序列的第一个标记
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict. 用于屏蔽值的标记。在使用屏蔽语言建模训练此模型时使用的标记。这是模型将尝试预测的标记
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one. 在进行标记化之前是否清理文本，通过删除任何控制字符并将所有空白字符替换为一个经典的空格
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)). 是否对中文字符进行标记化。这对于日文可能需要取消激活(参见此问题)
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original LayoutLM). 是否删除所有重音符号。如果没有指定此选项，则其取决于`lowercase`的值(与原始LayoutLM相同)
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords. 子词的前缀
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = LayoutLMTokenizer

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
    ):  # 定义类的构造函数，接受多个参数
        # 调用父类的构造函数，传入参数初始化父类的实例
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
            **kwargs, # 传入其他未命名参数
        )

        # 获取规范器的状态信息
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查是否需要重新设置规范器的参数
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取规范器的类
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            # 更新规范器的参数
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 使用新的参数创建规范器对象
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 保存参数
        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        通过连接和添加特殊标记，从一个序列或一对序列构建用于序列分类任务的模型输入。
        LayoutLM序列的格式如下：

        - 单个序列: `[CLS] X [SEP]`
        - 一对序列: `[CLS] A [SEP] B [SEP]`

        参数:
            token_ids_0 (`List[int]`):
                要添加特殊标记的ID列表。
            token_ids_1 (`List[int]`, *可选*):
                用于序列对的第二个ID列表。

        返回:
            `List[int]`: 添加了适当的特殊标记的[输入ID](../glossary#input-ids)列表。
        """
        # 添加特殊标记到输入ID列表
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果存在第二个ID列表，则将其添加到输出列表中
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        # 返回输出列表
        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个函数，用于创建用于序列对分类任务的序列掩码
    def create_sequence_pair_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A LayoutLM sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """

        # 定义分隔符和类别标识符
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果第二个序列的 IDs 为空，则只返回掩码的第一部分（0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回两个序列的掩码
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 保存词汇表到文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```