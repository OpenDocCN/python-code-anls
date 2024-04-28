# `.\transformers\models\lxmert\tokenization_lxmert_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本许可，除依法要求或书面同意外，不得使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非受适用法律要求或书面同意，否则根据许可证分发的软件将根据“原样”基础分发，不包括任何形式的保证或条件，无论是明示的还是暗示的
# 请查看协议以获取有关特定语言的权限和限制

import json
from typing import List, Optional, Tuple
#从tokenizers库中导入normalizers模块
from tokenizers import normalizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_lxmert import LxmertTokenizer

# 指定词汇文件名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "unc-nlp/lxmert-base-uncased": "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "unc-nlp/lxmert-base-uncased": (
            "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "unc-nlp/lxmert-base-uncased": 512,
}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "unc-nlp/lxmert-base-uncased": {"do_lower_case": True},
}

# 从transformers.models.bert.tokenization_bert_fast.BertTokenizerFast复制LXMERT的快速令牌化器
# 替换 bert-base-cased->unc-nlp/lxmert-base-uncased, BERT->Lxmert, Bert->Lxmert
class LxmertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速” Lxmert 令牌化器 (使用 HuggingFace 的 *tokenizers* 库支持)。基于WordPiece。

    该令牌化器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法，用户应该参考这个超类以获取有关这些方法的更多信息。
    ```
    # 以下是关于构造函数的参数说明
    Args:
        # 词汇表文件路径
        vocab_file (`str`):
            File containing the vocabulary.
        # 是否在标记化时转换为小写，默认为 `True`
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        # 未知标记，当词汇表中不包含时使用的标记
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        # 分隔符标记，用于构建多个序列的序列
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        # 填充标记，用于将序列填充到相同长度
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        # 分类器标记，用于序列分类
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        # 掩码标记，用于掩码语言建模训练
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        # 是否清理文本，例如去除控制字符、替换所有空格
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        # 是否标记化中文字符，建议对于日文不启用
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        # 是否去除所有重音符号
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original Lxmert).
        # 子词的前缀
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    
    """
    # 以下是类属性的赋值
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = LxmertTokenizer

    # 以下是构造函数
    def __init__(
        # 词汇表文件路径
        self,
        vocab_file=None,
        # 分词器文件路径
        tokenizer_file=None,
        # 是否在标记化时转换为小写，默认为 `True`
        do_lower_case=True,
        # 未知标记，当词汇表中不包含时使用的标记
        unk_token="[UNK]",
        # 分隔符标记
        sep_token="[SEP]",
        # 填充标记
        pad_token="[PAD]",
        # 分类器标记
        cls_token="[CLS]",
        # 掩码标记
        mask_token="[MASK]",
        # 是否标记化中文字符
        tokenize_chinese_chars=True,
        # 是否去除所有重音符号
        strip_accents=None,
        # 其他参数
        **kwargs,
        # 调用父类构造函数来初始化Tokenizer，传入所需的参数
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

        # 将当前的标准化器状态转换为JSON格式
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查当前标准化器是否需要更新
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 根据当前标准化器的类型创建新的标准化器对象并更新相关状态
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 更新do_lower_case参数
        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        根据输入的文本序列或一对文本序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。Lxmert序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 两个序列：`[CLS] A [SEP] B [SEP]`

        参数:
            token_ids_0 (`List[int]`):
                要添加特殊标记的ID列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的可选第二个ID列表。

        返回:
            `List[int]`: 包含适当特殊标记的[输入ID](../glossary#input-ids)列表。
        """
        # 构建带有特殊标记的输入序列
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。一个 Lxmert 序列对掩码具有以下格式：

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | 第一个序列    | 第二个序列 |
        ```

        如果 `token_ids_1` 是 `None`，这个方法只返回掩码的第一部分（0s）。

        参数:
            token_ids_0 (`List[int]`):
                ID 列表.
            token_ids_1 (`List[int]`, *optional*):
                序列对的可选第二个 ID 列表。

        返回:
            `List[int]`: 根据给定序列(s)的 [token 类型 ID](../glossary#token-type-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```