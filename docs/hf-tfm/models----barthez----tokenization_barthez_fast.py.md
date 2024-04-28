# `.\transformers\models\barthez\tokenization_barthez_fast.py`

```
# 指定编码格式为 UTF-8
# 版权声明和许可信息
""" Tokenization classes for the BARThez model."""

# 导入必要的库和模块
import os
from shutil import copyfile
from typing import List, Optional, Tuple

# 导入 tokenization_utils 模块中的 AddedToken 类
from ...tokenization_utils import AddedToken
# 导入 tokenization_utils_fast 模块中的 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 导入 is_sentencepiece_available 函数
from ...utils import is_sentencepiece_available, logging

# 判断是否有 sentencepiece 库可用
if is_sentencepiece_available():
    # 如果可用，从 tokenization_barthez 模块导入 BarthezTokenizer 类
    from .tokenization_barthez import BarthezTokenizer
else:
    # 如果不可用，则 BarthezTokenizer 为 None
    BarthezTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "moussaKam/mbarthez": "https://huggingface.co/moussaKam/mbarthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez": "https://huggingface.co/moussaKam/barthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez-orangesum-title": (
            "https://huggingface.co/moussaKam/barthez-orangesum-title/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "moussaKam/mbarthez": "https://huggingface.co/moussaKam/mbarthez/resolve/main/tokenizer.json",
        "moussaKam/barthez": "https://huggingface.co/moussaKam/barthez/resolve/main/tokenizer.json",
        "moussaKam/barthez-orangesum-title": (
            "https://huggingface.co/moussaKam/barthez-orangesum-title/resolve/main/tokenizer.json"
        ),
    },
}

# 定义预训练位置嵌入大小字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "moussaKam/mbarthez": 1024,
    "moussaKam/barthez": 1024,
    "moussaKam/barthez-orangesum-title": 1024,
}

# 定义特殊标记分隔符
SPIECE_UNDERLINE = "▁"


# 定义 BarthezTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class BarthezTokenizerFast(PreTrainedTokenizerFast):
    """
    Adapted from [`CamembertTokenizer`] and [`BartTokenizer`]. Construct a "fast" BARThez tokenizer. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
```  
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) 文件（通常具有 *.spm* 扩展名），其中包含实例化分词器所需的词汇表。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            在预训练期间使用的序列开始标记。可以用作序列分类器标记。

            <Tip>

            构建序列时使用特殊标记时，这不是用于序列开始的标记。用于序列开始的标记是 `cls_token`。

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            序列结束标记。

            <Tip>

            构建序列时使用特殊标记时，这不是用于序列结束的标记。用于序列结束的标记是 `sep_token`。

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            分隔符标记，在从多个序列构建序列时使用，例如，用于序列分类或用于文本和问题的问答。它也用作使用特殊标记构建的序列的最后一个标记。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            分类器标记，用于进行序列分类（对整个序列进行分类，而不是对每个标记进行分类）。它是使用特殊标记构建的序列的第一个标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中没有的标记无法转换为 ID，并将设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如，当批处理不同长度的序列时。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            用于屏蔽值的标记。这是在使用遮蔽语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            分词器使用的其他特殊标记。
    """

    # 词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速分词器类
    slow_tokenizer_class = BarthezTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
        # 如果mask_token是字符串，则使其像普通单词一样，即包含它前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类初始化方法，传入参数包括：词汇文件、标记器文件、bos标记、eos标记、unk标记、sep标记、cls标记、pad标记、mask标记以及其他参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 将词汇文件赋值给实例变量
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 如果词汇文件存在，则返回True；否则返回False
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BARThez sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 构建输入，将特殊标记和token_ids_0、token_ids_1连接起来
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # 如果没有token_ids_1，则返回与cls、token_ids_0、sep的长度相同的零列表
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回与cls、token_ids_0、sep、sep、token_ids_1、sep连接后长度相同的零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查是否能够保存慢速分词器的词汇表，若不能则引发异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 检查保存目录是否存在，若不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建输出词汇表文件路径，加上前缀并使用正确的文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 检查当前词汇表文件路径与输出词汇表文件路径是否相同，若不同则复制词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)
```