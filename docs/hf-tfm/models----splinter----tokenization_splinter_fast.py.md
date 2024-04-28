# `.\transformers\models\splinter\tokenization_splinter_fast.py`

```
# 文件编码为 utf-8
# 版权声明
# 基于 Apache 2.0 协议
# 返回许可证的文本版本
# 基于适用法律或书面同意要求的情况下，分发的软件为"原样"分发，没有任何保证或条件，无论是明示还是默示
# 请查看具体语言控制权限和限制的许可证
"""Splinter 的快速分词类。"""
# 导入
import json
from typing import List, Optional, Tuple
# 从 tokenizers 模块导入 normalizers
from tokenizers import normalizers
# 从 tokenization_utils_fast 模块导入 PreTrainedTokenizerFast
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从 utils 模块导入 logging
from ...utils import logging
# 从 tokenization_splinter 模块导入 SplinterTokenizer
from .tokenization_splinter import SplinterTokenizer

# 获取 logger
logger = logging.get_logger(__name__)

# 定义 VOCAB_FILES_NAMES
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义 PRETRAINED_VOCAB_FILES_MAP
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "tau/splinter-base": "https://huggingface.co/tau/splinter-base/resolve/main/vocab.txt",
        "tau/splinter-base-qass": "https://huggingface.co/tau/splinter-base-qass/resolve/main/vocab.txt",
        "tau/splinter-large": "https://huggingface.co/tau/splinter-large/resolve/main/vocab.txt",
        "tau/splinter-large-qass": "https://huggingface.co/tau/splinter-large-qass/resolve/main/vocab.txt",
    }
}

# 定义 PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "tau/splinter-base": 512,
    "tau/splinter-base-qass": 512,
    "tau/splinter-large": 512,
    "tau/splinter-large-qass": 512,
}

# 定义 PRETRAINED_INIT_CONFIGURATION
PRETRAINED_INIT_CONFIGURATION = {
    "tau/splinter-base": {"do_lower_case": False},
    "tau/splinter-base-qass": {"do_lower_case": False},
    "tau/splinter-large": {"do_lower_case": False},
    "tau/splinter-large-qass": {"do_lower_case": False},
}

# SplinterTokenizerFast 类，继承了 PreTrainedTokenizerFast
class SplinterTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" Splinter tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
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
        question_token (`str`, *optional*, defaults to `"[QUESTION]"`):
            The token used for constructing question representations.
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


    vocab_files_names = VOCAB_FILES_NAMES
    # 词汇表文件名列表，从常量中获取
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练词汇文件映射，从常量中获取
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 预训练初始化配置，从常量中获取
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 最大模型输入尺寸，从常量中获取
    slow_tokenizer_class = SplinterTokenizer
    # 慢速分词器类，设置为 SplinterTokenizer 类
    # 初始化函数，接受多个参数，设置默认值，并调用父类的初始化函数
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件
        tokenizer_file=None,  # 分词器文件
        do_lower_case=True,  # 是否转换为小写
        unk_token="[UNK]",  # 未知标记
        sep_token="[SEP]",  # 分隔标记
        pad_token="[PAD]",  # 填充标记
        cls_token="[CLS]",  # 类别标记
        mask_token="[MASK]",  # 掩盖标记
        question_token="[QUESTION]",  # 问题标记
        tokenize_chinese_chars=True,  # 是否分词中文字符
        strip_accents=None,  # 去除重音符号
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化函数
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
            additional_special_tokens=(question_token,),
            **kwargs,
        )

        # 获取预处理后的标记状态
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 如果预处理后的小写状态不等于传入的小写状态，或者预处理后的去重音符号状态不等于传入的去重音符号状态
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            # 获取预处理类
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            # 更新预处理状态的小写状态和去重音符号状态
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            # 设置后端分词器的预处理为更新后的预处理类和状态
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        # 设置当前实例的小写状态
        self.do_lower_case = do_lower_case

    @property
    def question_token_id(self):
        """
        `Optional[int]`: Id of the question token in the vocabulary, used to condition the answer on a question
        representation.
        """
        # 返回问题标记在词汇表中的标记ID
        return self.convert_tokens_to_ids(self.question_token)

    # 构建包含特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一对序列构建模型输入，用于问答任务，通过连接和添加特殊标记。Splinter 序列具有以下格式：

        - 单个序列：`[CLS] X [SEP]`
        - 问答任务的序列对：`[CLS] question_tokens [QUESTION] . [SEP] context_tokens [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                如果 pad_on_right，则是问题令牌的 ID，否则是上下文令牌的 ID
            token_ids_1 (`List[int]`, *optional*):
                如果 pad_on_right，则是上下文令牌的 ID，否则是问题令牌的 ID

        Returns:
            `List[int]`: 具有适当特殊标记的[输入 ID](../glossary#input-ids)列表。
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if self.padding_side == "right":
            # 输入是问题-然后-上下文
            return cls + token_ids_0 + question_suffix + sep + token_ids_1 + sep
        else:
            # 输入是上下文-然后-问题
            return cls + token_ids_0 + sep + token_ids_1 + question_suffix + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        创建与传递的序列相对应的标记类型 ID。[什么是标记类型 ID？](../glossary#token-type-ids)

        如果模型有特殊的方式构建它们，应该在子类中重写此方法。

        Args:
            token_ids_0 (`List[int]`): 第一个标记化序列。
            token_ids_1 (`List[int]`, *optional*): 第二个标记化序列。

        Returns:
            `List[int]`: 标记类型 ID。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        if self.padding_side == "right":
            # 输入是问题-然后-上下文
            return len(cls + token_ids_0 + question_suffix + sep) * [0] + len(token_ids_1 + sep) * [1]
        else:
            # 输入是上下文-然后-问题
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + question_suffix + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```