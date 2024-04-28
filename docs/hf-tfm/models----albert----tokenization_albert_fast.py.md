# `.\transformers\models\albert\tokenization_albert_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google AI、Google Brain 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" ALBERT 模型的分词类 """

# 导入所需的库
import os
from shutil import copyfile
from typing import List, Optional, Tuple

# 导入相关的模块和函数
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 如果安装了 sentencepiece 库，则导入 AlbertTokenizer 类
if is_sentencepiece_available():
    from .tokenization_albert import AlbertTokenizer
else:
    AlbertTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/spiece.model",
        "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/spiece.model",
        "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/spiece.model",
        "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/spiece.model",
        "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/spiece.model",
        "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/spiece.model",
        "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/spiece.model",
        "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/tokenizer.json",
        "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/tokenizer.json",
        "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/tokenizer.json",
        "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/tokenizer.json",
        "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json",
        "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/tokenizer.json",
        "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/tokenizer.json",
        "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/tokenizer.json",
    },
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "albert-base-v1": 512,
    "albert-large-v1": 512,
    "albert-xlarge-v1": 512,
    "albert-xxlarge-v1": 512,
    # 定义了四个不同的 ALBERT 模型，每个模型的隐层大小均为 512
    "albert-base-v2": 512,
    "albert-large-v2": 512,
    "albert-xlarge-v2": 512,
    "albert-xxlarge-v2": 512,
# 定义一个特殊字符，用于表示下划线
SPIECE_UNDERLINE = "▁"

# 定义一个类 AlbertTokenizerFast，继承自 PreTrainedTokenizerFast 类
class AlbertTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”ALBERT分词器（由HuggingFace的*tokenizers*库支持）。基于Unigram模型。该分词器继承自PreTrainedTokenizerFast类，其中包含大多数主要方法。用户应参考该超类以获取有关这些方法的更多信息
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
            用于实例化分词器所需的词汇文件（通常具有*.spm*扩展名）。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
            是否在分词时将输入转换为小写。
        remove_space (`bool`, *optional*, defaults to `True`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
            是否在分词时去除文本的空格（在字符串前后去除多余的空格）。
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether or not to keep accents when tokenizing.
            是否在分词时保留重音符号。
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            在预训练期间使用的序列起始标记。可用作序列分类器标记。
            <Tip>
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.
            构建带有特殊标记的序列时，这不是用于序列开头的标记。使用的标记是`cls_token`。
            </Tip>
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the `sep_token`.
            序列结束标记。注意：构建带有特殊标记的序列时，这不是用于序列结尾的标记。使用的标记是`sep_token`。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            未知标记。词汇表中不存在的标记无法转换为ID，因此设置为该标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            分隔标记，用于从多个序列构建序列，例如，用于序列分类的两个序列，或用于问答任务中的文本和问题。也用作使用特殊标记构建的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            用于填充的标记，例如在批处理不同长度的序列时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            在执行序列分类（整个序列的分类而不是每个标记的分类）时使用的分类器标记。在使用特殊标记构建时，它是序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
            用于屏蔽值的标记。在使用遮蔽语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = AlbertTokenizer
    # 初始化方法，用于创建一个新的Tokenizer对象
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        # 如果mask_token是字符串类型，则将其转换为AddedToken对象
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 调用父类的初始化方法，传入参数来构建Tokenizer对象
        super().__init__(
            vocab_file,
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
            **kwargs,
        )

        # 设置对象的属性值
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    # 返回是否可以保存慢速分词器的布尔值
    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 判断词汇文件是否存在来确定是否可以保存慢速分词器
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An ALBERT sequence has the following format:

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
        # 如果只有一个序列，返回带有特殊标记的单个序列
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        # 如果有两个序列，返回带有特殊标记的两个序列
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 从序列中创建token type IDs
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
        # Define separator token
        sep = [self.sep_token_id]
        # Define classification token
        cls = [self.cls_token_id]

        # If only one sequence is provided
        if token_ids_1 is None:
            # Return a mask with all zeros for the first sequence
            return len(cls + token_ids_0 + sep) * [0]
        # If two sequences are provided
        # Return a mask with zeros for the first sequence and ones for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if slow tokenizer can be saved
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # Check if save directory exists
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # Define the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the output vocabulary file path is different from the current vocabulary file path, copy the vocabulary
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # Return the output vocabulary file path
        return (out_vocab_file,)
```