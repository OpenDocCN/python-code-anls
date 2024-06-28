# `.\models\xlnet\tokenization_xlnet_fast.py`

```py
# coding=utf-8
# 版权声明和许可信息

# 导入必要的库和模块
import os  # 导入操作系统相关模块
from shutil import copyfile  # 导入文件复制函数
from typing import List, Optional, Tuple  # 导入类型提示相关模块

# 导入所需的Tokenization相关工具函数和类
from ...tokenization_utils import AddedToken  
from ...tokenization_utils_fast import PreTrainedTokenizerFast  
from ...utils import is_sentencepiece_available, logging  # 导入SentencePiece可用性检查和日志模块

# 如果SentencePiece可用，则导入XLNetTokenizer类
if is_sentencepiece_available():
    from .tokenization_xlnet import XLNetTokenizer
else:
    XLNetTokenizer = None  # 否则将XLNetTokenizer设置为None

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xlnet/xlnet-base-cased": "https://huggingface.co/xlnet/xlnet-base-cased/resolve/main/spiece.model",
        "xlnet/xlnet-large-cased": "https://huggingface.co/xlnet/xlnet-large-cased/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "xlnet/xlnet-base-cased": "https://huggingface.co/xlnet/xlnet-base-cased/resolve/main/tokenizer.json",
        "xlnet/xlnet-large-cased": "https://huggingface.co/xlnet/xlnet-large-cased/resolve/main/tokenizer.json",
    },
}

# 定义预训练位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlnet/xlnet-base-cased": None,
    "xlnet/xlnet-large-cased": None,
}

# 定义句子片段标识符常量
SPIECE_UNDERLINE = "▁"

# 定义不同句子段的标识符常量
SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

class XLNetTokenizerFast(PreTrainedTokenizerFast):
    """
    快速构建XLNet分词器，基于HuggingFace的tokenizers库。基于Unigram模型。

    该分词器继承自PreTrainedTokenizerFast类，包含大部分主要方法。详细方法信息请参考其超类。
    """
    pass  # 占位符，该类暂时没有自定义方法或属性，只是继承了PreTrainedTokenizerFast的功能
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether to lowercase the input when tokenizing.
        remove_space (`bool`, *optional*, defaults to `True`):
            Whether to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether to keep accents when tokenizing.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
            Additional special tokens used by the tokenizer.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    # VOCAB_FILES_NAMES is typically a constant or a configuration that defines
    # the names or paths of various vocabulary-related files used by the tokenizer.
    vocab_files_names = VOCAB_FILES_NAMES
    # 使用预先定义的词汇文件映射 PRETRAINED_VOCAB_FILES_MAP
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 使用预先定义的最大模型输入大小映射 PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 指定填充位置为左侧
    padding_side = "left"
    # 指定使用的慢速分词器类为 XLNetTokenizer

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
        # 如果 mask_token 是字符串，则创建一个 AddedToken 对象，用于处理去除左侧空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的构造方法，初始化分词器对象
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # 设置特殊的 pad_token_type_id 为 3
        self._pad_token_type_id = 3
        # 初始化其他属性
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查是否可以保存慢速分词器，基于是否存在 vocab_file 文件
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。对于 XLNet 模型，输入的格式如下：

        - 单个序列: `X <sep> <cls>`
        - 一对序列: `A <sep> B <sep> <cls>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        return token_ids_0 + sep + token_ids_1 + sep + cls

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        # 这个方法的作用没有具体注释，应该补充一个注释来解释它的功能
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
        sequence pair mask has the following format:

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
        # Separator token [SEP] used to separate sequences
        sep = [self.sep_token_id]
        # Classification segment ID indicating the segment for classification
        cls_segment_id = [2]

        # If only one sequence (`token_ids_1` is None), return a mask for the first sequence only
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        # Otherwise, return a mask for both sequences
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if the fast tokenizer can save vocabulary; raise an error if not
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # Check if save_directory is a valid directory; log an error and return if not
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # Define the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the current vocabulary file path is different from the output path, copy the vocabulary file
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # Return the path to the saved vocabulary file
        return (out_vocab_file,)
```