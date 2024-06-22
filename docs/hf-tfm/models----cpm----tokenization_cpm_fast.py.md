# `.\models\cpm\tokenization_cpm_fast.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据Apache License, Version 2.0进行授权，禁止违规使用
# 获取许可证的网址
# 分发软件基于“原样”基础，无明示或暗示的担保或条件
# 查看特定语言的限制和许可
"""Tokenization classes."""

# 导入模块
import os
from shutil import copyfile
from typing import List, Optional, Tuple
# 导入logging实用程序
from ...tokenization_utils_fast import AddedToken, PreTrainedTokenizerFast
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/tokenizer.json",
    },
}

# 创建 CpmTokenizerFast 类
class CpmTokenizerFast(PreTrainedTokenizerFast):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""

    # 初始化函数
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        additional_special_tokens=["<eop>", "<eod>"],
        **kwargs,
    ):
    # 定义可以保存慢速分词器的属性
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 从transformers.models.xlnet.tokenization_xlnet_fast.XLNetTokenizerFast.build_inputs_with_special_tokens 复制
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个函数，用于构建用于序列分类任务的模型输入，通过连接和添加特殊 token。一个 XLNet 序列的格式如下：
    # - 单个序列：`X <sep> <cls>`
    # - 一对序列：`A <sep> B <sep> <cls>`
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]  # 创建包含特殊 token 分隔符的列表
        cls_segment_id = [2]  # 创建包含特殊 token 分类标识的列表

        if token_ids_1 is None:  # 如果没有第二个 token 列表
            return len(token_ids_0 + sep) * [0] + cls_segment_id  # 返回由0组成的序列与分类标识的组合
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id  # 返回由0和1组成的序列与分类标识的组合


    # 从输入的序列或一对序列构建用于序列分类任务的模型输入，通过连接和添加特殊 token。一个 XLNet 序列的格式如下：
    # - 单个序列：`X <sep> <cls>`
    # - 一对序列：`A <sep> B <sep> <cls>`
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]  # 创建包含特殊 token 分隔符的列表
        cls = [self.cls_token_id]  # 创建包含特殊 token 分类标识的列表
        if token_ids_1 is None:  # 如果没有第二个 token 列表
            return token_ids_0 + sep + cls  # 返回第一个 token 列表与分隔符和分类标识的组合
        return token_ids_0 + sep + token_ids_1 + sep + cls  # 返回两个 token 列表与分隔符和分类标识的组合
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则抛出数值错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )
        
        # 如果保存目录不存在，记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 拼接输出词汇文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 如果词汇文件路径与输出词汇文件路径不一致，则复制词汇文件到输出词汇文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        
        # 返回输出词汇文件路径的元组
        return (out_vocab_file,)
    
    # 批量编码文本或文本对
    def _batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        # 对批量的文本或文本对进行分词，然后拼接成字符串
        batch_text_or_text_pairs = [
            " ".join([x.translate(self.translator) for x in self.jieba.cut(text, cut_all=False)])
            for text in batch_text_or_text_pairs
        ]
        # 调用父类的批量编码方法
        return super()._batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)
    
    # 解码文本
    def _decode(self, *args, **kwargs):
        # 调用父类的解码方法
        text = super()._decode(*args, **kwargs)
        # 替换空格和特殊符号
        text = text.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")
        # 返回处理后的文本
        return text
```