# `.\transformers\models\vits\tokenization_vits.py`

```
# 设置文件编码为 utf-8
# 版权声明，包括作者和团队信息
# 根据 Apache 许可协议 Version 2.0 使用此文件
# 可以访问 http://www.apache.org/licenses/LICENSE-2.0 获取许可协议的副本
# 根据适用法律或书面约定，分发的软件是基于"AS IS"的基础分发的，没有任何保证或条件，无论是明示还是默示
# 请查看许可协议以获取有关权限和限制的详细信息
"""VITS 的 Tokenization 类"""  # 对 VITS 进行 Tokenization

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_phonemizer_available, logging  # 导入必要的库

if is_phonemizer_available():  # 如果 phonemizer 可用
    import phonemizer  # 导入 phonemizer

logger = logging.get_logger(__name__)  # 获取 logger 对象

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}  # 定义词汇文件名字典

PRETRAINED_VOCAB_FILES_MAP = {  # 预训练词汇文件映射
    "vocab_file": {  # 词汇文件
        "facebook/mms-tts-eng": "https://huggingface.co/facebook/mms-tts-eng/resolve/main/vocab.json",  # 模型对应的词汇文件链接
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {  # 预训练位置嵌入大小
    # 此模型没有最大输入长度
    "facebook/mms-tts-eng": 4096,  # 模型的最大输入长度
}


def has_non_roman_characters(input_string):  # 检查输入字符串是否含有非罗马字符的函数
    # 查找 ASCII 范围外的字符
    non_roman_pattern = re.compile(r"[^\x00-\x7F]")  # 定义正则表达式模式查找非罗马字符

    # 在输入字符串中搜索非罗马字符
    match = non_roman_pattern.search(input_string)  # 在输入字符串中搜索非罗马字符
    has_non_roman = match is not None  # 判断是否找到非罗马字符
    return has_non_roman  # 返回结果，是否包含非罗马字符


class VitsTokenizer(PreTrainedTokenizer):
    """
    构造 VITS Tokenizer，同时支持 MMS-TTS。

    这个 Tokenizer 继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考这个超类以获取有关这些方法的更多信息。

    Args:
        vocab_file (`str`):
            词汇文件的路径。
        language (`str`, *optional*):
            语言标识符。
        add_blank (`bool`, *optional*, defaults to `True`):
            是否在其他标记之间插入标记 id 0。
        normalize (`bool`, *optional*, defaults to `True`):
            是否通过删除所有大小写和标点来规范化输入文本。
        phonemize (`bool`, *optional*, defaults to `True`):
            是否将输入文本转换为音素。
        is_uroman (`bool`, *optional*, defaults to `False`):
            是否需要在对输入文本进行分词之前应用 'uroman' 罗马化器。
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 定义词汇文件名字典
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练位置嵌入大小
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入的名称
    # 初始化方法，接受参数并进行相关设置
    def __init__(
        self,
        vocab_file,
        pad_token="<pad>",
        unk_token="<unk>",
        language=None,
        add_blank=True,
        normalize=True,
        phonemize=True,
        is_uroman=False,
        **kwargs,
    ) -> None:
        # 从文件中读取词汇表内容，使用 UTF-8 编码
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        # 根据词汇表内容构建反向映射
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置语言
        self.language = language
        # 是否添加空白标记
        self.add_blank = add_blank
        # 是否进行规范化处理
        self.normalize = normalize
        # 是否进行音素化处理
        self.phonemize = phonemize
        # 是否是uroman
        self.is_uroman = is_uroman

        # 调用父类的初始化方法
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            language=language,
            add_blank=add_blank,
            normalize=normalize,
            phonemize=phonemize,
            is_uroman=is_uroman,
            **kwargs,
        )

    # 返回词汇表的大小
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 获取词汇表内容
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 对输入字符串进行规范化处理
    def normalize_text(self, input_string):
        """Lowercase the input string, respecting any special token ids that may be part or entirely upper-cased."""
        all_vocabulary = list(self.encoder.keys()) + list(self.added_tokens_encoder.keys())
        filtered_text = ""

        i = 0
        # 遍历输入字符串
        while i < len(input_string):
            found_match = False
            # 在词汇表中查找匹配的单词
            for word in all_vocabulary:
                if input_string[i : i + len(word)] == word:
                    filtered_text += word
                    i += len(word)
                    found_match = True
                    break

            # 如果没有找到匹配的单词，将字符转换为小写
            if not found_match:
                filtered_text += input_string[i].lower()
                i += 1

        return filtered_text

    # 特定语言的字符预处理方法
    def _preprocess_char(self, text):
        """Special treatment of characters in certain languages"""
        if self.language == "ron":
            # 如果语言是罗马���亚，则将字符 "ț" 替换为 "ţ"
            text = text.replace("ț", "ţ")
        return text

    # 准备进行分词处理
    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, normalize: Optional[bool] = None, **kwargs
    # 定义一个方法，执行令牌化之前的任何必要转换，并返回剩余的 `kwargs`
    def _pre_tokenize(self, text: str, is_split_into_words: bool = False, normalize: bool = None, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # 如果未指定 normalize 参数，则使用默认的 self.normalize 参数
        normalize = normalize if normalize is not None else self.normalize
    
        # 如果需要进行标准化
        if normalize:
            # 对文本进行标准化处理
            text = self.normalize_text(text)
    
        # 将文本预处理为字符
        filtered_text = self._preprocess_char(text)
    
        # 如果预处理后的文本包含非罗马字符，并且使用罗马字符标记器
        if has_non_roman_characters(filtered_text) and self.is_uroman:
            # 发出警告信息
            logger.warning(
                "Text to the tokenizer contains non-Roman characters. Ensure the `uroman` Romanizer is "
                "applied to the text prior to passing it to the tokenizer. See "
                "`https://github.com/isi-nlp/uroman` for details."
            )
    
        # 如果启用了音素标记
        if self.phonemize:
            # 检查是否安装了 phonemizer 包
            if not is_phonemizer_available():
                raise ImportError("Please install the `phonemizer` Python package to use this tokenizer.")
            
            # 对过滤文本进行音素标记处理
            filtered_text = phonemizer.phonemize(
                filtered_text,
                language="en-us",
                backend="espeak",
                strip=True,
                preserve_punctuation=True,
                with_stress=True,
            )
            # 移除多余的空格
            filtered_text = re.sub(r"\s+", " ", filtered_text)
        # 如果需要进行标准化
        elif normalize:
            # 去除词汇表之外的字符（标点符号）
            filtered_text = "".join(list(filter(lambda char: char in self.encoder, filtered_text))).strip()
    
        # 返回处理后的文本和剩余的 kwargs
        return filtered_text, kwargs
    # 通过在相邻字符之间插入 `<pad>` token 来将字符串进行分词
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string by inserting the `<pad>` token at the boundary between adjacent characters."""
        # 将字符串转换成字符列表
        tokens = list(text)

        # 如果需要添加空白token
        if self.add_blank:
            # 创建一个有默认token填充的列表
            interspersed = [self._convert_id_to_token(0)] * (len(tokens) * 2 + 1)
            # 将原始token插入到默认token的位置上
            interspersed[1::2] = tokens
            tokens = interspersed

        # 返回分词结果
        return tokens

    # 将tokens列表转换为字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # 如果需要添加空白token，并且tokens列表长度大于1
        if self.add_blank and len(tokens) > 1:
            # 从tokens列表中提取原始的token
            tokens = tokens[1::2]
        # 将tokens列表拼接成字符串并返回
        return "".join(tokens)

    # 将token转换成对应的id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将token转换成id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将id转换成对应的token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将id转换成token
        return self.decoder.get(index)

    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Union[Tuple[str], None]:
        # 如果保存目录不存在，则报错并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 拼接保存文件的路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 将词汇表以 JSON 格式写入文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 返回保存文件的路径
        return (vocab_file,)
```