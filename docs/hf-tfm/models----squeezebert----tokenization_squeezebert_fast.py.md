# `.\transformers\models\squeezebert\tokenization_squeezebert_fast.py`

```
# 设定文件编码为 UTF-8
# 版权声明，引用了 SqueezeBert 作者和 HuggingFace Inc. 团队的版权声明
# 根据 Apache License 2.0 进行许可
# 只有在遵守许可证的情况下才能使用此文件
# 可以从以下链接获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得使用此文件分发软件
# 软件按"原样"提供，没有任何形式的担保或条件，明示或默示
# 请参阅许可证以获取特定语言的权限
"""SqueezeBERT 的标记化类。"""

# 导入必要的库
import json
from typing import List, Optional, Tuple

# 导入 tokenizers 库中的 normalizers 模块
from tokenizers import normalizers

# 导入日志记录工具
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging

# 导入 SqueezeBertTokenizer 类
from .tokenization_squeezebert import SqueezeBertTokenizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件映射
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

# 定义预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "squeezebert/squeezebert-uncased": 512,
    "squeezebert/squeezebert-mnli": 512,
    "squeezebert/squeezebert-mnli-headless": 512,
}

# 定义预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "squeezebert/squeezebert-uncased": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli-headless": {"do_lower_case": True},
}

# 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 复制的类，将 Bert->SqueezeBert，BERT->SqueezeBERT
class SqueezeBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”SqueezeBERT分词器（由HuggingFace的*tokenizers*库支持）。基于 WordPiece。

    该分词器继承自[`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该
    参考此超类以获取有关这些方法的更多信息。
    Args:
        vocab_file (`str`):
            File containing the vocabulary. 词汇表文件的路径
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing. 在进行分词时是否将输入转换为小写
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. 未知标记，用于词汇表中不存在的标记
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens. 分隔标记，用于在多个序列构建一条序列时进行分隔
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths. 用于填充的标记，例如在批处理不同长度的序列时使用
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens. 分类器标记，用于进行序列分类
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict. 用于屏蔽值的标记，用于进行屏蔽语言建模时的训练
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one. 在进行分词前是否清理文本
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)). 是否对中文字符进行分词
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original SqueezeBERT). 是否去除所有的重音符号
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords. 子词的前缀
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 词汇表文件的名称列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练模型的词汇表文件映射
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION  # 预训练模型的初始化配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练模型的最大输入尺寸
    slow_tokenizer_class = SqueezeBertTokenizer  # 慢速分词器的类

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
        **kwargs,  # 其他参数
    ):  # 定义参数列表结束
        super().__init__(  # 调用父类的初始化方法
            vocab_file,  # 词汇表文件路径
            tokenizer_file=tokenizer_file,  # 分词器文件路径
            do_lower_case=do_lower_case,  # 是否转换为小写
            unk_token=unk_token,  # 未知标记
            sep_token=sep_token,  # 分隔符标记
            pad_token=pad_token,  # 填充标记
            cls_token=cls_token,  # 类标记
            mask_token=mask_token,  # 掩码标记
            tokenize_chinese_chars=tokenize_chinese_chars,  # 是否分词中文字符
            strip_accents=strip_accents,  # 是否去除重音符号
            **kwargs,  # 其他参数
        )

        # 使用后端分词器的规范化状态
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查规范化状态参数，若与输入参数不符则修改后端分词器的规范化器
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

        self.do_lower_case = do_lower_case  # 保存是否转换为小写参数

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
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]  # 构建输入序列，加入特殊标记

        if token_ids_1 is not None:  # 若输入包含第二个序列
            output += token_ids_1 + [self.sep_token_id]  # 加入第二个序列及其特殊标记

        return output  # 返回输入序列及特殊标记

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None  # 创建 token type ids
    ) -> List[int]:
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
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 创建分隔符和类别标记的列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果第二个序列的 token_ids_1 是 None，则只返回 mask 的第一部分（全为 0）
        if token_ids_1 is None:
            # 返回全为 0 的列表，长度为类别标记 + 第一个序列长度 + 分隔符长度的总和
            return len(cls + token_ids_0 + sep) * [0]
        # 返回第一部分全为 0，第二部分全为 1 的 mask
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 保存词汇表文件到指定目录下
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名组成的元组
        return tuple(files)
```