# `.\models\reformer\tokenization_reformer_fast.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可协议信息
# 引入操作系统模块和复制文件函数
# 引入类型提示模块中的 Optional 和 Tuple 类型
import os
from shutil import copyfile
from typing import Optional, Tuple

# 从 tokenization_utils_fast 中引入 PreTrainedTokenizerFast 类
# 从 utils 中引入 is_sentencepiece_available 和 logging 函数
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 如果 sentencepiece 可用，从 tokenization_reformer 中引入 ReformerTokenizer 类，否则为 None
if is_sentencepiece_available():
    from .tokenization_reformer import ReformerTokenizer
else:
    ReformerTokenizer = None

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义特殊的单词分隔符
SPIECE_UNDERLINE = "▁"

# 定义词汇文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/reformer-crime-and-punishment": (
            "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/spiece.model"
        )
    },
    "tokenizer_file": {
        "google/reformer-crime-and-punishment": (
            "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/tokenizer.json"
        )
    },
}

# 定义预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/reformer-crime-and-punishment": 524288,
}

# 定义 ReformerTokenizerFast 类，继承自 PreTrainedTokenizerFast
class ReformerTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”Reformer分词器（由HuggingFace的tokenizers库支持）。基于Unigram模型。

    这个分词器继承自 PreTrainedTokenizerFast，包含大多数主要方法。用户应该参考这个超类来获取更多关于这些方法的信息。
    """
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
    """

    # 获取预定义的文件名常量列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 获取预训练模型使用的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 获取预训练位置嵌入的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速标记器类定义为 ReformerTokenizer
    slow_tokenizer_class = ReformerTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        additional_special_tokens=[],
        **kwargs,
    ):
        # 调用父类的初始化方法，传递参数以设置词汇文件、标记器文件、特殊标记等
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # 将参数中的词汇文件路径保存到对象属性中
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查当前对象是否具备保存慢速标记器所需的信息，主要是检查词汇文件是否存在
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速标记器，则引发 ValueError 异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存路径不是一个目录，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 指定输出词汇文件的路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇文件路径与输出路径不一致，则复制词汇文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回保存的词汇文件路径的元组
        return (out_vocab_file,)
```