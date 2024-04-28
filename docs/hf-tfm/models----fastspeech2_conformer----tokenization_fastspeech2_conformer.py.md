# `.\models\fastspeech2_conformer\tokenization_fastspeech2_conformer.py`

```
# 设置代码文件的编码格式为utf-8
# 版权声明部分，保留版权和许可信息
# 加载必要库和模块
"""Tokenization classes for FastSpeech2Conformer."""
# 导入需要的库和模块
import json
import os
from typing import Optional, Tuple

import regex

# 从tokenization_utils模块中导入PreTrainedTokenizer类
from ...tokenization_utils import PreTrainedTokenizer
# 从utils模块中导入logging和requires_backends
from ...utils import logging, requires_backends


# 获取logger实例
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

# 指定预训练词汇文件的路径
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "espnet/fastspeech2_conformer": "https://huggingface.co/espnet/fastspeech2_conformer/raw/main/vocab.json",
    },
}

# 设置预训练位置编码大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    # 设置为相对较大的数字，因为模型输入并不受相对位置编码的限制
    "espnet/fastspeech2_conformer": 4096,
}

# 定义FastSpeech2ConformerTokenizer类，继承自PreTrainedTokenizer
class FastSpeech2ConformerTokenizer(PreTrainedTokenizer):
    """
    Construct a FastSpeech2Conformer tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The begin of sequence token. Note that for FastSpeech2, it is the same as the `eos_token`.
        eos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The end of sequence token. Note that for FastSpeech2, it is the same as the `bos_token`.
        pad_token (`str`, *optional*, defaults to `"<blank>"`):
            The token used for padding, for example when batching sequences of different lengths.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        should_strip_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to strip the spaces from the list of tokens.
    """
    
    # 定义类属性
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 初始化方法
    def __init__(
        self,
        vocab_file,
        bos_token="<sos/eos>",
        eos_token="<sos/eos>",
        pad_token="<blank>",
        unk_token="<unk>",
        should_strip_spaces=False,
        **kwargs,
        ):
            # 检查是否需要使用"g2p_en"后端
            requires_backends(self, "g2p_en")

            # 以 UTF-8 编码打开词汇文件，并加载到编解码器中
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)

            # 导入g2p_en库
            import g2p_en

            # 创建g2p_en的G2p对象
            self.g2p = g2p_en.G2p()

            # 使用编码器创建解码器
            self.decoder = {v: k for k, v in self.encoder.items()}

            # 调用父类的初始化方法，并传入相应的参数
            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                should_strip_spaces=should_strip_spaces,
                **kwargs,
            )

            # 设置是否需要去除空格
            self.should_strip_spaces = should_strip_spaces

        @property
        def vocab_size(self):
            # 返回解码器的长度作为词汇表大小
            return len(self.decoder)

        def get_vocab(self):
            # 返回包含编码器和已添加tokens编码器的词汇表字典
            "Returns vocab as a dict"
            return dict(self.encoder, **self.added_tokens_encoder)

        def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
            # 扩展符号
            text = regex.sub(";", ",", text)
            text = regex.sub(":", ",", text)
            text = regex.sub("-", " ", text)
            text = regex.sub("&", "and", text)

            # 去除不必要的符号
            text = regex.sub(r"[\(\)\[\]\<\>\"]+", "", text)

            # 去除空格
            text = regex.sub(r"\s+", " ", text)

            # 转换为大写
            text = text.upper()

            return text, kwargs

        def _tokenize(self, text):
            """Returns a tokenized string."""
            # 对文本进行音素划分
            tokens = self.g2p(text)

            # 如果需要去除空格，则去除空格
            if self.should_strip_spaces:
                tokens = list(filter(lambda s: s != " ", tokens))

            # 添加结束标记
            tokens.append(self.eos_token)

            return tokens

        def _convert_token_to_id(self, token):
            """Converts a token (str) to an id using the vocab."""
            # 使用编码器将token转换为id
            return self.encoder.get(token, self.encoder.get(self.unk_token))

        def _convert_id_to_token(self, index):
            """Converts an index (integer) to a token (str) using the vocab."""
            # 使用解码器将id转换为token
            return self.decoder.get(index, self.unk_token)

        # 由于音素无法可靠地转换回字符串，因此需要重写
        def decode(self, token_ids, **kwargs):
            logger.warn(
                "Phonemes cannot be reliably converted to a string due to the one-many mapping, converting to tokens instead."
            )
            return self.convert_ids_to_tokens(token_ids)

        # 由于音素无法可靠地转换回字符串，因此需要重写
        def convert_tokens_to_string(self, tokens, **kwargs):
            logger.warn(
                "Phonemes cannot be reliably converted to a string due to the one-many mapping, returning the tokens."
            )
            return tokens
    # 将词汇表和特殊标记文件保存到指定目录中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            # 如果保存目录不存在，则记录错误并返回
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇表文件的路径，包括可选的文件名前缀
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 将词汇表内容写入文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.get_vocab(), ensure_ascii=False))

        # 返回保存的文件路径
        return (vocab_file,)

    # 定义对象的状态保存方法
    def __getstate__(self):
        # 复制对象的字典状态
        state = self.__dict__.copy()
        # 设置 g2p 属性为 None，因为 g2p 属性不可序列化
        state["g2p"] = None
        return state

    # 定义对象的状态加载方法
    def __setstate__(self, d):
        # 用传入的状态字典设置对象的状态
        self.__dict__ = d

        try:
            # 尝试导入 g2p_en 模块
            import g2p_en

            # 初始化对象的 g2p 属性
            self.g2p = g2p_en.G2p()
        except ImportError:
            # 如果导入失败，则抛出 ImportError 异常
            raise ImportError(
                "You need to install g2p-en to use FastSpeech2ConformerTokenizer. "
                "See https://pypi.org/project/g2p-en/ for installation."
            )
```