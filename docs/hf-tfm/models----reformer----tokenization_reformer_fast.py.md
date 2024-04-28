# `.\transformers\models\reformer\tokenization_reformer_fast.py`

```
# 设定文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件
# 按"原样"分发，不提供任何形式的担保或条件
# 有关特定语言的权限，请参见许可证
"""Reformer 模型的分词类"""


# 导入所需模块
import os
from shutil import copyfile
from typing import Optional, Tuple

# 导入句子分词模块
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging


# 如果句子分词模块可用，则导入 ReformerTokenizer 类，否则设置为 None
if is_sentencepiece_available():
    from .tokenization_reformer import ReformerTokenizer
else:
    ReformerTokenizer = None


# 获取日志记录器
logger = logging.get_logger(__name__)


# 定义分词时的下划线符号
SPIECE_UNDERLINE = "▁"

# 定义词汇表文件名
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 预训练词汇表文件映射
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

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/reformer-crime-and-punishment": 524288,
}


# 定义 ReformerTokenizerFast 类，继承自 PreTrainedTokenizerFast
class ReformerTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”Reformer 分词器（由 HuggingFace 的 *tokenizers* 库支持）。基于
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models)。

    该分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大部分主要方法。用户应该
    参考此超类以获取有关这些方法的更多信息。
    """
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) 文件（通常具有 *.spm* 扩展名），包含实例化分词器所需的词汇表。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            结束序列的特殊标记。

            <Tip>

            在构建序列时使用特殊标记时，这并不是用于表示序列结束的标记。用于表示序列结束的标记是 `sep_token`。

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中不存在的标记将无法转换为 ID，并设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如在对不同长度的序列进行批处理时。
        additional_special_tokens (`List[str]`, *optional*):
            分词器使用的其他特殊标记。
    """

    # 定义相关文件名和映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
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
        # 初始化基类
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # 设置词汇表文件路径
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 判断是否能保存慢速分词器的词汇表
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 保存词汇表到指定目录
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 确保保存目录存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果输出路径与当前词汇表路径不同，将词汇表复制到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
```