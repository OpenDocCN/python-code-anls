# `.\models\convbert\tokenization_convbert_fast.py`

```py
# coding=utf-8
# 版权归 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证版本 2.0 授权使用此文件；
# 除非符合许可证的要求，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”发布的，
# 没有任何形式的明示或暗示保证或条件。
# 请参阅许可证以获取有关特定语言的权限和限制。
"""ConvBERT 的分词类。"""
import json
from typing import List, Optional, Tuple

from tokenizers import normalizers  # 导入 tokenizers 库中的 normalizers 模块

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入预训练的快速分词器类
from ...utils import logging  # 导入 logging 模块，用于记录日志
from .tokenization_convbert import ConvBertTokenizer  # 导入 ConvBERT 分词器类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}  # 定义词汇表文件名映射字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/vocab.txt",
        "YituTech/conv-bert-medium-small": (
            "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/vocab.txt"
        ),
        "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "YituTech/conv-bert-base": 512,
    "YituTech/conv-bert-medium-small": 512,
    "YituTech/conv-bert-small": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "YituTech/conv-bert-base": {"do_lower_case": True},  # 预训练配置字典，指定小写处理为真
    "YituTech/conv-bert-medium-small": {"do_lower_case": True},
    "YituTech/conv-bert-small": {"do_lower_case": True},
}

# 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 复制，将 bert-base-cased->YituTech/conv-bert-base, Bert->ConvBert, BERT->ConvBERT
class ConvBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    使用 HuggingFace 的 *tokenizers* 库构建“快速”ConvBERT分词器，基于 WordPiece。

    该分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    ```
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
            value for `lowercase` (as in the original ConvBERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    ```
    # 定义一些预定义的变量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = ConvBertTokenizer
    
    ```
    # 初始化方法，用于实例化一个新的 ConvBertTokenizer 对象
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
    ```
        ):
        # 调用父类的初始化方法，设置模型的词汇文件、分词器文件、大小写敏感性、未知标记、分隔标记、填充标记、类标记、掩码标记、中文字符分词选项和重音符号处理选项
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

        # 获取当前后端分词器的正常化状态
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查正常化状态是否与参数中的设置一致，如果不一致则更新分词器的正常化器
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取当前正常化器类
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            # 更新正常化状态的设置
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 使用更新后的设置重新初始化后端分词器的正常化器
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 更新当前对象的大小写敏感性设置
        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。ConvBERT 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 一对序列：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的 ID 列表（可选）。

        Returns:
            `List[int]`: 包含适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        # 初始化输出列表，以 [CLS] 标记开始，然后是 token_ids_0，最后加上 [SEP] 标记
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果提供了 token_ids_1，将其添加到输出列表中，并以 [SEP] 标记结尾
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_convbert_sequence_classification_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ConvBERT sequence
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
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define separator and classification tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If only one sequence is provided, return a mask with all 0s
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # If two sequences are provided, concatenate their lengths to create the mask
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's vocabulary to the specified directory.

        Args:
            save_directory (str):
                Directory where the vocabulary files will be saved.
            filename_prefix (str, *optional*):
                Prefix for the saved files.

        Returns:
            `Tuple[str]`: Tuple containing the filenames where the vocabulary was saved.
        """
        # Call the tokenizer's model save method to save the vocabulary
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # Return the filenames where the vocabulary was saved
        return tuple(files)
```