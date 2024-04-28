# `.\transformers\models\mobilebert\tokenization_mobilebert_fast.py`

```
# 导入必要的模块和类型提示
import json
from typing import List, Optional, Tuple

# 导入 tokenizers 模块中的 normalizers 函数
from tokenizers import normalizers

# 从 tokenization_utils_fast 中导入 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast

# 导入 logging 模块
from ...utils import logging

# 从 tokenization_mobilebert 中导入 MobileBertTokenizer 类
from .tokenization_mobilebert import MobileBertTokenizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件名的映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/vocab.txt"},
    "tokenizer_file": {
        "mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/tokenizer.json"
    },
}

# 定义预训练位置嵌入大小的映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"mobilebert-uncased": 512}

# 定义预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {}


# 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 复制而来，将其中的 BERT 替换为 MobileBERT，Bert 替换为 MobileBert
class MobileBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”的 MobileBERT 分词器（基于 HuggingFace 的 *tokenizers* 库）。基于 WordPiece。

    该分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该参考此超类以获取有关这些方法的更多信息。
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
            value for `lowercase` (as in the original MobileBERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """
    # 定义一些预设的参数和值，用于构建tokenizer实例
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = MobileBertTokenizer

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
        # 调用父类的初始化方法，传入参数
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

        # 获取当前 backend_tokenizer 对象的 normalizer_state
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查 normalizer_state 是否与实例化时传入的参数不一致，若不一致则重新设置 normalizer_state
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 重新设置 normalizer_state
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 更新 backend_tokenizer 的 normalizer
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 设置实例对象属性 do_lower_case
        self.do_lower_case = do_lower_case

    # 创建特殊标记的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        通过连接和添加特殊标记，从序列或序列对构建模型输入，用于序列分类任务。MobileBERT 序列具有以下格式：

        - 单个序列: `[CLS] X [SEP]`
        - 序列对: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列ID列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的[input IDs](../glossary#input-ids)列表。
        """
        # 构建包含特殊标记的输入序列
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 若有第二个序列，则将第二个序列的 token_ids_1 和特殊标记添加到输出中
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # 从序列构建 token 类型 ID
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义函数，用于生成用于序列对分类任务的遮罩
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A MobileBERT sequence
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
        # 定义分隔符和CLS标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果token_ids_1为None，只返回第一部分遮罩（全为0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 定义函数，保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用tokenizer模块的save方法将词汇表保存到指定目录下
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件路径元组
        return tuple(files)
```