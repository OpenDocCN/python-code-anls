# `.\models\big_bird\tokenization_big_bird_fast.py`

```
# 设定文件编码为 UTF-8
# 版权声明及许可信息
# 根据 Apache License 2.0 许可使用代码
# 如果不符合许可条件，则不能使用本文件
# 获取许可副本地址：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，本软件是按“原样”基础分发的，不提供任何明示或暗示的担保或条件。
# 请查阅许可证以了解具体的法律权限和限制。
""" Big Bird 模型的 Tokenization 类 """

# 导入标准库和模块
import os
from shutil import copyfile
from typing import List, Optional, Tuple

# 导入依赖的工具和函数
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 如果 SentencePiece 可用，则导入 BigBirdTokenizer 类，否则置为 None
if is_sentencepiece_available():
    from .tokenization_big_bird import BigBirdTokenizer
else:
    BigBirdTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射
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

# 定义预训练模型的位置编码嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/bigbird-roberta-base": 4096,
    "google/bigbird-roberta-large": 4096,
    "google/bigbird-base-trivia-itc": 4096,
}

# 定义 SentencePiece 中的特殊字符
SPIECE_UNDERLINE = "▁"

# BigBirdTokenizerFast 类继承自 PreTrainedTokenizerFast 类
class BigBirdTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”的 BigBird 分词器（由 HuggingFace 的 tokenizers 库支持）。基于 Unigram 模型。
    该分词器继承自 `PreTrainedTokenizerFast`，包含大多数主要方法。用户应参考其超类以获取更多关于这些方法的信息。
    """
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
    # 定义一些预先设置好的常量和类，用于初始化 tokenizer
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
    ):
        """
        构造函数，初始化一个新的 tokenizer 对象。

        Args:
            vocab_file (str, optional): SentencePiece 文件的路径，包含了实例化 tokenizer 所需的词汇表。
            tokenizer_file (str, optional): tokenizer 文件的路径，如果提供了，将会加载现有的 tokenizer。
            unk_token (str, optional): 未知 token，当词汇表中没有某个词时，将使用该 token。
            bos_token (str, optional): 序列的开头 token，用于序列分类或者特殊 token 序列的起始。
            eos_token (str, optional): 序列的结尾 token，用于特殊 token 序列的结束。
            pad_token (str, optional): 填充 token，在批处理不同长度序列时使用。
            sep_token (str, optional): 分隔 token，用于构建来自多个序列的单一序列。
            cls_token (str, optional): 分类器 token，用于序列分类任务中整个序列的分类。
            mask_token (str, optional): 掩码 token，用于预测被 mask 的词语。
            **kwargs: 其他关键字参数，用于额外配置 tokenizer。
        """
    ):
        # 如果 bos_token 是字符串类型，则创建一个 AddedToken 对象，保持左右两端的空白不变
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串类型，则创建一个 AddedToken 对象，保持左右两端的空白不变
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 unk_token 是字符串类型，则创建一个 AddedToken 对象，保持左右两端的空白不变
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串类型，则创建一个 AddedToken 对象，保持左右两端的空白不变
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果 cls_token 是字符串类型，则创建一个 AddedToken 对象，保持左右两端的空白不变
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        # 如果 sep_token 是字符串类型，则创建一个 AddedToken 对象，去除左侧空白，保持右侧空白
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的初始化方法，传入参数进行初始化
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

        # 设置实例的 vocab_file 属性
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查 vocab_file 是否存在，如果存在返回 True，否则返回 False
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An BigBird sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果 token_ids_1 为空，则返回 `[CLS] + token_ids_0 + [SEP]`
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        # 否则返回 `[CLS] + token_ids_0 + [SEP] + token_ids_1 + [SEP]`
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:
    
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
    
        if token_ids_1 is None, only returns the first portion of the mask (0s).
    
        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
    
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 定义分隔和类别标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
    
        # 如果没有第二个序列，则返回只包含第一个序列和分隔符的长度的 0 组成的列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # 否则返回一个列表，其中包含第一个序列、分隔符以及第二个序列和分隔符的长度的 0 和 1 组成的列表
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 定义一个保存词汇表的方法，接受一个保存目录和可选的文件名前缀作为参数，并返回一个包含文件路径的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查是否能够保存慢速分词器的词汇表，否则抛出数值错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建输出词汇表文件的路径，如果提供了前缀则加在文件名前面，否则直接使用默认文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件的绝对路径不等于输出路径的绝对路径，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回保存的词汇表文件路径的元组
        return (out_vocab_file,)
```