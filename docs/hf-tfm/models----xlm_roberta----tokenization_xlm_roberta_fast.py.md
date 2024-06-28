# `.\models\xlm_roberta\tokenization_xlm_roberta_fast.py`

```py
# 设置 Python 文件的编码格式为 UTF-8
# 版权声明和许可证信息
# 本模块提供了 XLM-RoBERTa 模型的分词类

# 导入必要的模块和函数
import os
from shutil import copyfile
from typing import List, Optional, Tuple

# 导入自定义的分词相关工具函数和类
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 如果安装了 sentencepiece 库，则导入 XLMRobertaTokenizer 类
if is_sentencepiece_available():
    from .tokenization_xlm_roberta import XLMRobertaTokenizer
else:
    XLMRobertaTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义用于 XLM-RoBERTa 模型的词汇文件名称映射
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型和对应的词汇文件 URL 映射关系
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "FacebookAI/xlm-roberta-base": "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/sentencepiece.bpe.model",
        "FacebookAI/xlm-roberta-large": "https://huggingface.co/FacebookAI/xlm-roberta-large/resolve/main/sentencepiece.bpe.model",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/sentencepiece.bpe.model"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/sentencepiece.bpe.model"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english/resolve/main/sentencepiece.bpe.model"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll03-german": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-german/resolve/main/sentencepiece.bpe.model"
        ),
    },
    # 定义一个字典，包含多个键值对，每个键是模型名称，对应的值是该模型的 tokenizer.json 文件的 URL
    "tokenizer_file": {
        "FacebookAI/xlm-roberta-base": "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/tokenizer.json",
        "FacebookAI/xlm-roberta-large": "https://huggingface.co/FacebookAI/xlm-roberta-large/resolve/main/tokenizer.json",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/tokenizer.json"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/tokenizer.json"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english/resolve/main/tokenizer.json"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll03-german": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-german/resolve/main/tokenizer.json"
        ),
    },
}

# 定义一个预训练位置嵌入大小的字典，不同的模型对应不同的嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "FacebookAI/xlm-roberta-base": 512,  # XLM-RoBERTa base 模型的位置嵌入大小为 512
    "FacebookAI/xlm-roberta-large": 512,  # XLM-RoBERTa large 模型的位置嵌入大小为 512
    "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch": 512,  # 细调为荷兰语的 XLM-RoBERTa large 模型的位置嵌入大小为 512
    "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish": 512,  # 细调为西班牙语的 XLM-RoBERTa large 模型的位置嵌入大小为 512
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english": 512,  # 细调为英语的 XLM-RoBERTa large 模型的位置嵌入大小为 512
    "FacebookAI/xlm-roberta-large-finetuned-conll03-german": 512,  # 细调为德语的 XLM-RoBERTa large 模型的位置嵌入大小为 512
}


class XLMRobertaTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”XLM-RoBERTa tokenizer（由HuggingFace的*tokenizers*库支持）。改编自[`RobertaTokenizer`]和[`XLNetTokenizer`]。
    基于[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models)。

    此tokenizer继承自[`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.

        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining.
            Can be used a sequence classifier token.

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

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.

        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.

        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.

        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
    """
    
    # Define constants
    vocab_files_names = VOCAB_FILES_NAMES  # Constant mapping vocabulary file names
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # Mapping of pretrained vocabulary files
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # Maximum model input sizes
    model_input_names = ["input_ids", "attention_mask"]  # Names of model input tensors
    slow_tokenizer_class = XLMRobertaTokenizer  # Tokenizer class used

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
    ):
        """
        Constructor for the tokenizer class.

        Args:
            vocab_file (str, optional): Path to the vocabulary file.
            tokenizer_file (str, optional): Path to the tokenizer file.
            bos_token (str, optional): The beginning of sequence token.
            eos_token (str, optional): The end of sequence token.
            sep_token (str, optional): The separator token.
            cls_token (str, optional): The classifier token.
            unk_token (str, optional): The unknown token.
            pad_token (str, optional): The padding token.
            mask_token (str, optional): The masking token.
            **kwargs: Additional keyword arguments.
        """
    ):
        # 如果 mask_token 是字符串，创建一个剥离左边空格的 AddedToken 对象
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的初始化方法，传入参数进行初始化
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 将 vocab_file 赋值给 self.vocab_file
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 如果 self.vocab_file 存在且是文件，则返回 True，否则返回 False
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊 token，为序列分类任务构建模型输入。XLM-RoBERTa 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊 token 的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的 ID 列表，用于序列对。

        Returns:
            `List[int]`: 带有适当特殊 token 的输入 ID 列表。
        """

        if token_ids_1 is None:
            # 返回包含特殊 token 的单个序列的输入 ID 列表
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回包含特殊 token 的序列对的输入 ID 列表
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        创建用于序列对分类任务的 mask。XLM-RoBERTa 不使用 token 类型 ID，因此返回一个全为零的列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的 ID 列表，用于序列对。

        Returns:
            `List[int]`: 全为零的列表。

        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # 返回单个序列的 token 类型 ID 列表，全为零
            return len(cls + token_ids_0 + sep) * [0]
        # 返回序列对的 token 类型 ID 列表，全为零
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
    # 定义保存词汇表的方法，接受一个保存目录路径和可选的文件名前缀参数，返回一个包含文件路径字符串的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则引发值错误异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，记录错误日志并返回空
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return

        # 组合输出词汇表文件的路径，结合可选的文件名前缀和标准的词汇表文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不一致，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回包含输出词汇表文件路径的元组
        return (out_vocab_file,)
```