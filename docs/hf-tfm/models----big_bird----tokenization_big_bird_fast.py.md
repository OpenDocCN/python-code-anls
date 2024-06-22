# `.\transformers\models\big_bird\tokenization_big_bird_fast.py`

```py
# 设置脚本的字符编码为 UTF-8
# 版权声明，指明代码版权归谷歌人工智能、谷歌Brain以及HuggingFace公司团队所有
# 使用 Apache 许可证版本 2.0 发布，除非你遵守许可证中规定的条款，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则按“原样”提供软件，不提供任何形式的保证或条件
# 查看许可证获取更多详细信息
""" 用于 Big Bird 模型的分词类 """

# 导入所需模块
import os  # 导入操作系统相关模块
from shutil import copyfile  # 从 shutil 模块中导入 copyfile 函数
from typing import List, Optional, Tuple  # 导入类型提示模块中的 List、Optional、Tuple 类型

# 导入 tokenization_utils 模块中的 AddedToken 类
from ...tokenization_utils import AddedToken  
# 导入 tokenization_utils_fast 模块中的 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  
# 导入 utils 模块中的 is_sentencepiece_available、logging 函数
from ...utils import is_sentencepiece_available, logging  

# 如果 SentencePiece 可用
if is_sentencepiece_available():  
    # 从当前目录中的 tokenization_big_bird 模块中导入 BigBirdTokenizer 类
    from .tokenization_big_bird import BigBirdTokenizer  
else:
    BigBirdTokenizer = None  # 否则 BigBirdTokenizer 设置为 None

# 获取 logging 模块中的 logger 对象
logger = logging.get_logger(__name__)
# 设置词汇表文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型词汇表文件的映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/spiece.model",
        "google/bigbird-roberta-large": (
            "https://huggingface.co/google/bigbird-roberta-large/resolve/main/spiece.model"
        ),
        "google/bigbird-base-trivia-itc": (
            "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/spiece.model"
        ),
    },
    "tokenizer_file": {
        "google/bigbird-roberta-base": (
            "https://huggingface.co/google/bigbird-roberta-base/resolve/main/tokenizer.json"
        ),
        "google/bigbird-roberta-large": (
            "https://huggingface.co/google/bigbird-roberta-large/resolve/main/tokenizer.json"
        ),
        "google/bigbird-base-trivia-itc": (
            "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练模型位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/bigbird-roberta-base": 4096,
    "google/bigbird-roberta-large": 4096,
    "google/bigbird-base-trivia-itc": 4096,
}

# 用于标识 SentencePiece 特殊符号的字符串
SPIECE_UNDERLINE = "▁"

# BigBirdTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class BigBirdTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速” BigBird 分词器（由 HuggingFace 的 *tokenizers* 库支持）。
    基于[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models)。
    此分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息
    """
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    # 定义一些属性，包括文件名、预训练的词汇文件映射、预训练位置嵌入的最大模型输入尺寸、慢速分词器类、模型输入的名称列表、前缀令牌列表
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = BigBirdTokenizer
    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sep_token="[SEP]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        **kwargs,
```  
    ):
        # 如果 bos_token 是字符串，则将其转换为 AddedToken 对象，保留左右空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则将其转换为 AddedToken 对象，保留左右空格
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 unk_token 是字符串，则将其转换为 AddedToken 对象，保留左右空格
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则将其转换为 AddedToken 对象，保留左右空格
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果 cls_token 是字符串，则将其转换为 AddedToken 对象，保留左右空格
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        # 如果 sep_token 是字符串，则将其转换为 AddedToken 对象，保留左右空格
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token

        # Mask token 表现为普通单词，即包括其前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的初始化方法，传入参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 设置实例变量 vocab_file
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 判断是否可以保存慢速 tokenizer，如果 vocab_file 存在则返回 True，否则返回 False
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊标记，从序列或序列对构建用于序列分类任务的模型输入。BigBird 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表
            token_ids_1 (`List[int]`, *optional*):
                用于序列对的可选第二个 ID 列表。

        Returns:
            `List[int]`: 具有适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从不含特殊标记的令牌列表中检索序列 ID。当使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                如果令牌列表已经使用模型的特殊标记格式化，则设置为 True。

        Returns:
            `List[int]`: 在范围 [0, 1] 中的整数列表：1 表示特殊令牌，0 表示序列令牌。
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "如果提供的 ID 序列已经使用模型的特殊标记格式化，请不要提供第二个序列。"
                )
            return [1 if x in [self.sep_token_id, self.cls_token_id] else 0 for x in token_ids_0]

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。ALBERT 序列对掩码的格式如下：

        ```py
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | 第一个序列     | 第二个序列 |
        ```

        如果 token_ids_1 为 None，则仅返回掩码的第一部分（0）。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的 ID 列表，用于序列对。

        Returns:
            `List[int]`: 根据给定序列(s)生成的[令牌类型 ID](../glossary#token-type-ids)列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查是否可以保存慢速分词器的词汇表
        if not self.can_save_slow_tokenizer:
            # 如果不能保存，则引发值错误
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            # 如果不存在，则记录错误并返回
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出词汇表文件路径不同
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            # 复制当前词汇表文件到输出词汇表文件路径
            copyfile(self.vocab_file, out_vocab_file)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)
```