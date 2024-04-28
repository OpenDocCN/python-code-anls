# `.\models\fnet\tokenization_fnet_fast.py`

```
# 导入必要的库
import os  # 导入操作系统库
from shutil import copyfile  # 导入文件复制函数copyfile
from typing import List, Optional, Tuple  # 导入类型提示相关模块

# 导入必要的类和函数
from ...tokenization_utils import AddedToken  # 导入AddedToken类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入PreTrainedTokenizerFast类
from ...utils import is_sentencepiece_available, logging  # 导入判断SentencePiece是否可用和日志模块

# 如果SentencePiece可用，则导入FNetTokenizer类
if is_sentencepiece_available():
    from .tokenization_fnet import FNetTokenizer
else:
    FNetTokenizer = None  # 否则设置FNetTokenizer为None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/tokenizer.json",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/tokenizer.json",
    },
}

# 预训练位置编码的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/fnet-base": 512,
    "google/fnet-large": 512,
}

# SentencePiece模型的下划线字符
SPIECE_UNDERLINE = "▁"


# 定义FNetTokenizerFast类，继承自PreTrainedTokenizerFast类
class FNetTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" FNetTokenizer (backed by HuggingFace's *tokenizers* library). Adapted from
    [`AlbertTokenizerFast`]. Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
    tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods
    """
```  
    # 参数说明：
    # vocab_file (`str`): SentencePiece 文件的路径，包含了初始化分词器所需的词汇表
    # do_lower_case (`bool`, *optional*, defaults to `False`): 是否在分词时将输入转为小写
    # remove_space (`bool`, *optional*, defaults to `True`): 在分词时是否去除文本中的空格
    # keep_accents (`bool`, *optional*, defaults to `True`): 在分词时是否保留重音符号
    # unk_token (`str`, *optional*, defaults to `"<unk>"`): 未知标记，不在词汇表中的标记将被设置为此标记
    # sep_token (`str`, *optional*, defaults to `"[SEP]"`): 分隔符标记，用于构建多个序列时的分隔
    # pad_token (`str`, *optional*, defaults to `"<pad>"`): 用于填充的标记，在批处理长度不同的序列时使用
    # cls_token (`str`, *optional*, defaults to `"[CLS]"`): 用于序列分类的分类器标记。构建包含特殊标记的序列时为第一个标记
    # mask_token (`str`, *optional*, defaults to `"[MASK]"`): 用于掩码值的标记，在掩码语言建模训练时使用

    vocab_files_names = VOCAB_FILES_NAMES  # 词汇表文件名
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇表文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 最大模型输入尺寸
    model_input_names = ["input_ids", "token_type_ids"]  # 模型输入名称列表
    slow_tokenizer_class = FNetTokenizer  # 慢速分词器类

    def __init__(
        self,
        vocab_file=None,  # 词汇表文件路径
        tokenizer_file=None,  # 分词器文件路径
        do_lower_case=False,  # 是否小写化
        remove_space=True,  # 是否去除空格
        keep_accents=True,  # 是否保留重音符号
        unk_token="<unk>",  # 未知标记
        sep_token="[SEP]",  # 分隔符标记
        pad_token="<pad>",  # 填充标记
        cls_token="[CLS]",  # 分类器标记
        mask_token="[MASK]",  # 掩码标记
        **kwargs,  # 其他关键字参数
    ):
        # 如果 mask_token 是字符串类型，则创建一个 AddedToken 对象并设置其左侧空格剥离，右侧不剥离
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        # 如果 cls_token 是字符串类型，则创建一个 AddedToken 对象并设置其左侧不剥离，右侧不剥离
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        # 如果 sep_token 是字符串类型，则创建一个 AddedToken 对象并设置其左侧不剥离，右侧不剥离
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token

        # 调用父类的初始化方法，传入相应参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 设置属性值
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 判断是否存在 vocab_file 文件，若存在返回 True，否则返回 False
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。一个 FNet 序列的格式如下：

        - 单个序列: `[CLS] X [SEP]`
        - 一对序列: `[CLS] A [SEP] B [SEP]`

        参数:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表
            token_ids_1 (`List[int]`, *optional*):
                用于序列对的可选第二个 ID 列表

        返回:
            `List[int]`: 含有适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        创建用于序列对分类任务的掩码。一个 FNet 序列对掩码的格式如下：

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | 第一个序列       | 第二个序列     |
        ```

        如果 token_ids_1 为 None，则仅返回掩码的第一个部分（全为0）。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的可选 ID 列表，用于序列对。

        Returns:
            `List[int]`: 根据给定序列的标记类型 ID 列表。
        """
        sep = [self.sep_token_id]  # 分隔符标记的 ID
        cls = [self.cls_token_id]  # 类别标记的 ID

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # 返回第一个序列的掩码
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]  # 返回序列对的掩码

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )  # 输出的词汇表文件路径

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)  # 复制词汇表文件到输出路径

        return (out_vocab_file,)  # 返回输出的词汇表文件路径元组
```