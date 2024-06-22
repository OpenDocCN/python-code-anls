# `.\transformers\models\roformer\tokenization_roformer_fast.py`

```py
# 指定编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权，仅在符合许可证下才能使用该文件
# 可以在以下链接获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律或书面同意，否则根据"AS IS"基础分发软件
# 没有任何明示或暗示的担保或条件，参见许可证以获取特定语言的权限和
# 限制
"""RoFormer"的标记化类。"""
# 导入必要的库和模块
import json
# 导入类型提示的工具
from typing import List, Optional, Tuple
# 导入 tokenizers 库中的必要组件
from tokenizers import normalizers
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
# 导入 logging 模块
from ...utils import logging
from .tokenization_roformer import RoFormerTokenizer
# 导入 JiebaPreTokenizer 类
from .tokenization_utils import JiebaPreTokenizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预先训练的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_char_small": (
            "https://huggingface.co/junnyu/roformer_chinese_char_small/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_chinese_char_base": (
            "https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_discriminator": (
            "https://huggingface.co/junnyu/roformer_small_discriminator/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_generator": (
            "https://huggingface.co/junnyu/roformer_small_generator/resolve/main/vocab.txt"
        ),
    }
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "junnyu/roformer_chinese_small": 1536,
    "junnyu/roformer_chinese_base": 1536,
    "junnyu/roformer_chinese_char_small": 512,
    "junnyu/roformer_chinese_char_base": 512,
    "junnyu/roformer_small_discriminator": 128,
    "junnyu/roformer_small_generator": 128,
}

# 预先训练的初始配置
PRETRAINED_INIT_CONFIGURATION = {
    "junnyu/roformer_chinese_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_base": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_base": {"do_lower_case": True},
    "junnyu/roformer_small_discriminator": {"do_lower_case": True},
    "junnyu/roformer_small_generator": {"do_lower_case": True},
}
    # RoFormerTokenizerFast 类几乎与 BertTokenizerFast 完全相同,都是从头到尾进行标记化:
    # 标点分割和词块。在对中文进行标记时,它们之间存在一些差异。
    
    # 该标记器继承自 PreTrainedTokenizerFast 类,其中包含大多数主要方法。
    # 用户应该参考这个父类以了解这些方法的更多信息。
    
    # 示例:
    # 从"junnyu/roformer_chinese_base"加载预训练的 RoFormerTokenizerFast 
    # 并使用它对句子进行标记化。
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
        # 调用父类 PreTrainedTokenizerFast 的 __init__ 方法
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
    
        # 获取标准化器的状态,并检查是否需要更新标准化器的参数
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)
    
        # 设置自定义的预标记器
        vocab = self.backend_tokenizer.get_vocab()
        self.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer(vocab))
    
        self.do_lower_case = do_lower_case
    
    # 保存和加载状态时,需要更新预标记器为默认的 BertPreTokenizer
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"].pre_tokenizer = BertPreTokenizer()
        return state
    
    def __setstate__(self, d):
        self.__dict__ = d
        vocab = self.__dict__["_tokenizer"].get_vocab()
        self.__dict__["_tokenizer"].pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer(vocab))
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        从一个序列或一个序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。RoFormer 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的可选第二个 ID 列表。

        Returns:
            `List[int]`: 带有适当特殊标记的 [input IDs](../glossary#input-ids) 列表。
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。RoFormer 序列对掩码的格式如下：

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | 第一个序列    | 第二个序列 |
        ```py

        如果 `token_ids_1` 是 `None`，则此方法仅返回掩码的第一部分（0s）。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的可选第二个 ID 列表。

        Returns:
            `List[int]`: 根据给定序列返回的 [token type IDs](../glossary#token-type-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def save_pretrained(
        self,
        save_directory,
        legacy_format=None,
        filename_prefix=None,
        push_to_hub=False,
        **kwargs,
    ):
        self.backend_tokenizer.pre_tokenizer = BertPreTokenizer()
        return super().save_pretrained(save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)
```