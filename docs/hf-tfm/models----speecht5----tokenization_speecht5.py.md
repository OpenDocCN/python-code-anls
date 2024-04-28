# `.\transformers\models\speecht5\tokenization_speecht5.py`

```py
# 设置文件编码格式为 UTF-8
# 版权声明与许可协议，如何获取协议等信息
"""Tokenization class for SpeechT5."""
# 引入所需的包
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
# 引入 sentencepiece
import sentencepiece as spm
# 引入 logging 模块
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from .number_normalizer import EnglishNumberNormalizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm_char.model"}

# 预训练模型使用的词汇文件
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/speecht5_asr": "https://huggingface.co/microsoft/speecht5_asr/resolve/main/spm_char.model",
        "microsoft/speecht5_tts": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/spm_char.model",
        "microsoft/speecht5_vc": "https://huggingface.co/microsoft/speecht5_vc/resolve/main/spm_char.model",
    }
}
# 预训练模型的位置编码大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/speecht5_asr": 1024,
    "microsoft/speecht5_tts": 1024,
    "microsoft/speecht5_vc": 1024,
}

# 定义 SpeechT5Tokenizer 类，继承自 PreTrainedTokenizer
class SpeechT5Tokenizer(PreTrainedTokenizer):
    """
    Construct a SpeechT5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            SentencePiece 文件的路径，包含了实例化分词器所需的词汇表，通常具有 *.spm* 扩展名。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            序列开始标记。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            序列结束标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中不存在的标记将无法转换为 ID，并被设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如在将不同长度的序列进行批处理时使用。
        normalize (`bool`, *optional*, defaults to `False`):
            是否将文本中的数值量转换为它们拼写出的英文形式。
        sp_model_kwargs (`dict`, *optional*):
            将传递给 `SentencePieceProcessor.__init__()` 方法的参数字典。可以用于设置 SentencePiece 的各种参数，如：

            - `enable_sampling`: 启用子词正则化。
            - `nbest_size`: 用于unigram的采样参数。对于BPE-Dropout无效。

              - `nbest_size = {0,1}`: 不执行采样。
              - `nbest_size > 1`: 从nbest_size个结果中采样。
              - `nbest_size < 0`: 假设nbest_size为无穷大，并使用前向-后向采样算法从所有假设（格）中采样。

            - `alpha`: unigram采样的平滑参数，以及BPE-dropout的合并操作的丢弃概率。

    Attributes:
        sp_model (`SentencePieceProcessor`):
            用于每次转换（字符串、标记和 ID）的 SentencePiece 处理器。
    """

    # 词汇表文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入大小的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        normalize=False,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
```  
    ) -> None:
        # 初始化方法，设置分词模型的关键字参数
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 设置词汇表文件路径
        self.vocab_file = vocab_file
        # 设置是否进行文本规范化
        self.normalize = normalize
        # 初始化文本规范化器
        self._normalizer = None

        # 使用给定的关键字参数初始化 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 载入指定的词汇表文件
        self.sp_model.Load(vocab_file)

        # 调用父类初始化方法，设置特殊符号和关键字参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            normalize=normalize,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 获取是否进行文本规范化的参数
        normalize = kwargs.pop("normalize", self.normalize)
        # 如果文本已经是分词后的单词列表，则在其前添加空格
        if is_split_into_words:
            text = " " + text
        # 如果需要进行文本规范化，则使用规范化器对文本进行处理
        if normalize:
            text = self.normalizer(text)
        # 返回处理后的文本和额外的关键字参数
        return (text, kwargs)

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return self.sp_model.get_piece_size()

    @property
    def normalizer(self):
        # 获取文本规范化器，如果不存在则初始化一个英文数字规范化器
        if self._normalizer is None:
            self._normalizer = EnglishNumberNormalizer()
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value):
        # 设置文本规范化器
        self._normalizer = value

    def get_vocab(self):
        # 获取词汇表，包括转换后的特殊符号和自定义添加的符号
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 获取当前对象的状态，剔除 sp_model 对象以便序列化
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 恢复对象状态
        self.__dict__ = d

        # 为了向后兼容性，如果不存在 sp_model_kwargs，则初始化为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用之前的关键字参数重新初始化 SentencePieceProcessor 对象并载入词汇表
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        # 使用 SentencePiece 模型对文本进行编码，返回分词后的字符串列表
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将 token 转换为其在词汇表中的 id
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将 id 转换为对应的 token
        token = self.sp_model.IdToPiece(index)
        return token

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 初始化一个空列表用于存储当前的子 token
        current_sub_tokens = []
        # 初始化一个空字符串，用于存储最终的输出字符串
        out_string = ""
        # 初始化一个变量，用于标记前一个 token 是否是特殊 token
        prev_is_special = False
        # 遍历输入的 tokens 序列
        for token in tokens:
            # 检查当前 token 是否是特殊 token
            if token in self.all_special_tokens:
                # 如果前一个 token 不是特殊 token，则在输出字符串中添加一个空格
                if not prev_is_special:
                    out_string += " "
                # 解码当前子 token 列表并添加当前 token，更新输出
```