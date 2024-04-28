# `.\transformers\models\xglm\tokenization_xglm_fast.py`

```py
# 设置编码格式为 utf-8
# 版本版权声明
# 根据Apache许可证版本2.0获得授权方可使用此文件
# 可到 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 如非依法要求，或经授权的书面同意，只有在符合本许可证的条件下方能使用本文件
# 本许可证限制代码的发布，基于 "不做任何担保或条件"，无论是明示的或暗示的
# 请从许可证中查看具体语言许可许可的权限以及限制。
"""XGLM"""的标识处理类

# 导入所需的模块和库
import os
from shutil import copyfile
from typing import List, Optional, Tuple
# 从 tokenization_utils_fast 模块中导入 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从 utils 模块中导入 is_sentencepiece_available 和 logging 函数
from ...utils import is_sentencepiece_available, logging
# 如 sentencepiece 可用，从 tokenization_xglm 导入 XGLMTokenizer 类
if is_sentencepiece_available():
    from .tokenization_xglm import XGLMTokenizer
else:
    XGLMTokenizer = None

# 获取日志处理对象
logger = logging.get_logger(__name__)
# 设置词汇文件名的映射
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 设置预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/xglm-564M": "https://huggingface.co/facebook/xglm-564M/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "facebook/xglm-564M": "https://huggingface.co/facebook/xglm-564M/resolve/main/tokenizer.json",
    },
}
# 设置预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/xglm-564M": 2048,
}

# 定义 XGLMTokenizerFast 类，继承 PreTrainedTokenizerFast 类
class XGLMTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" XGLM tokenizer (backed by HuggingFace's *tokenizers* library). Adapted from [`RobertaTokenizer`]
    and [`XLNetTokenizer`]. Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    <s>NOTUSED", "</s>NOTUSED"]`):
                # 标记器使用的其他特殊标记。
                Additional special tokens used by the tokenizer.
        """
    
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        model_input_names = ["input_ids", "attention_mask"]
        slow_tokenizer_class = XGLMTokenizer
    
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
            **kwargs,
    ):
        # 兼容原始分词器
        self.num_madeup_words = 7
        # 创建虚构单词列表
        madeup_words = [f"<madeupword{i}>" for i in range(self.num_madeup_words)]

        # 如果未指定附加特殊标记，则初始化为空列表
        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        # 合并虚构单词到附加特殊标记中
        kwargs["additional_special_tokens"] += [
            word for word in madeup_words if word not in kwargs["additional_special_tokens"]
        ]

        # 调用父类的构造函数，传递所需参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

        # 保存词汇文件路径
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 判断词汇文件是否存在
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接并添加特殊标记，为序列分类任务构建模型输入。
        XLM-RoBERTa 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                需要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 具有适当特殊标记的输入 ID 列表。
        """

        if token_ids_1 is None:
            return [self.sep_token_id] + token_ids_0
        sep = [self.sep_token_id]
        return sep + token_ids_0 + sep + sep + token_ids_1

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。
        XLM-RoBERTa 不使用令牌类型 ID，因此返回零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零列表。
        """

        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return len(sep + token_ids_0) * [0]
        return len(sep + token_ids_0 + sep + sep + token_ids_1) * [0]
    # 保存词汇表到指定目录，返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则引发数值错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )
    
        # 如果保存目录不存在，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        # 组装输出词汇表文件的路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
    
        # 如果当前词汇表文件的绝对路径不等于输出文件的绝对路径，则复制当前词汇表文件到输出文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
    
        # 返回保存的文件路径的元组
        return (out_vocab_file,)
```