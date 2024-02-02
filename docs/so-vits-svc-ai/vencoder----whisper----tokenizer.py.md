# `so-vits-svc\vencoder\whisper\tokenizer.py`

```py
# 导入必要的库
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import GPT2TokenizerFast

# 定义语言代码和对应的语言名称的映射关系
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
}
    # "mt": "maltese"，键值对，键为"mt"，值为"maltese"
    "mt": "maltese",
    # "sa": "sanskrit"，键值对，键为"sa"，值为"sanskrit"
    "sa": "sanskrit",
    # "lb": "luxembourgish"，键值对，键为"lb"，值为"luxembourgish"
    "lb": "luxembourgish",
    # "my": "myanmar"，键值对，键为"my"，值为"myanmar"
    "my": "myanmar",
    # "bo": "tibetan"，键值对，键为"bo"，值为"tibetan"
    "bo": "tibetan",
    # "tl": "tagalog"，键值对，键为"tl"，值为"tagalog"
    "tl": "tagalog",
    # "mg": "malagasy"，键值对，键为"mg"，值为"malagasy"
    "mg": "malagasy",
    # "as": "assamese"，键值对，键为"as"，值为"assamese"
    "as": "assamese",
    # "tt": "tatar"，键值对，键为"tt"，值为"tatar"
    "tt": "tatar",
    # "haw": "hawaiian"，键值对，键为"haw"，值为"hawaiian"
    "haw": "hawaiian",
    # "ln": "lingala"，键值对，键为"ln"，值为"lingala"
    "ln": "lingala",
    # "ha": "hausa"，键值对，键为"ha"，值为"hausa"
    "ha": "hausa",
    # "ba": "bashkir"，键值对，键为"ba"，值为"bashkir"
    "ba": "bashkir",
    # "jw": "javanese"，键值对，键为"jw"，值为"javanese"
    "jw": "javanese",
    # "su": "sundanese"，键值对，键为"su"，值为"sundanese"
    "su": "sundanese",
}
# 定义一个语言名称到语言代码的映射，包括一些语言的别名
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

# 定义一个名为 Tokenizer 的数据类，用于包装 GPT2TokenizerFast，提供对特殊标记的快速访问
@dataclass(frozen=True)
class Tokenizer:
    """A thin wrapper around `GPT2TokenizerFast` providing quick access to special tokens"""

    tokenizer: "GPT2TokenizerFast"
    language: Optional[str]
    sot_sequence: Tuple[int]

    # 编码文本的方法
    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    # 解码标记的方法
    def decode(self, token_ids: Union[int, List[int], np.ndarray, torch.Tensor], **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    # 带时间戳的解码方法
    def decode_with_timestamps(self, tokens) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        outputs = [[]]
        for token in tokens:
            if token >= self.timestamp_begin:
                timestamp = f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        return "".join(outputs)

    # 获取结束标记的属性
    @property
    @lru_cache()
    def eot(self) -> int:
        return self.tokenizer.eos_token_id

    # 获取开始转录的属性
    @property
    @lru_cache()
    def sot(self) -> int:
        return self._get_single_token_id("<|startoftranscript|>")

    # 获取语言模型的开始标记的属性
    @property
    @lru_cache()
    def sot_lm(self) -> int:
        return self._get_single_token_id("<|startoflm|>")

    # 获取缓存的属性
    @property
    @lru_cache()
    # 返回特殊标记 "<|startofprev|>" 对应的 token id
    def sot_prev(self) -> int:
        return self._get_single_token_id("<|startofprev|>")

    # 返回特殊标记 "<|nospeech|>" 对应的 token id
    @property
    @lru_cache()
    def no_speech(self) -> int:
        return self._get_single_token_id("<|nospeech|>")

    # 返回特殊标记 "<|notimestamps|>" 对应的 token id
    @property
    @lru_cache()
    def no_timestamps(self) -> int:
        return self._get_single_token_id("<|notimestamps|>")

    # 返回时间戳开始的 token id
    @property
    @lru_cache()
    def timestamp_begin(self) -> int:
        return self.tokenizer.all_special_ids[-1] + 1

    # 返回与 `language` 字段值对应的 token id
    @property
    @lru_cache()
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        additional_tokens = dict(
            zip(
                self.tokenizer.additional_special_tokens,
                self.tokenizer.additional_special_tokens_ids,
            )
        )
        candidate = f"<|{self.language}|>"
        if candidate in additional_tokens:
            return additional_tokens[candidate]

        raise KeyError(f"Language {self.language} not found in tokenizer.")

    # 返回所有语言 token 的元组
    @property
    @lru_cache()
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in zip(
            self.tokenizer.additional_special_tokens,
            self.tokenizer.additional_special_tokens_ids,
        ):
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    # 返回所有语言代码的元组
    @property
    @lru_cache()
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([l]).strip("<|>") for l in self.all_language_tokens)

    # 返回包括不带时间戳的 SOT 序列的元组
    @property
    @lru_cache()
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    # 返回所有语言 token 的元组
    @property
    @lru_cache()
    # 定义一个方法，返回需要屏蔽的标记列表，以避免包含说话者标记或非语音注释，以防止采样非实际语音的文本，例如：
    # - ♪♪♪
    # - ( SPEAKING FOREIGN LANGUAGE )
    # - [DAVID] Hey there,
    # 保留基本的标点符号，如逗号、句号、问号、感叹号等。

    def non_speech_tokens(self) -> Tuple[int]:
        # 定义需要屏蔽的符号列表
        symbols = list("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")
        # 添加额外的符号到列表中
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # 可能是单个标记或多个标记的符号，取决于分词器。
        # 如果它们是多个标记，则屏蔽第一个标记，这是安全的，因为：
        # 这些是 U+2640 到 U+267F 之间的杂项符号，在生成中可以屏蔽，它们在 3 字节的 UTF-8 表示中共享前两个字节。
        miscellaneous = set("♩♪♫♬♭♮♯")
        # 断言所有杂项符号的 Unicode 编码在 0x2640 到 0x267F 之间
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # 允许连字符 "-" 和单引号 "'" 在单词之间，但不允许它们出现在单词开头
        result = {self.tokenizer.encode(" -")[0], self.tokenizer.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.tokenizer.encode(symbol), self.tokenizer.encode(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        # 返回结果的元组
        return tuple(sorted(result))

    # 定义一个私有方法，获取文本的单个标记 ID
    def _get_single_token_id(self, text) -> int:
        # 使用分词器对文本进行编码
        tokens = self.tokenizer.encode(text)
        # 断言编码后的标记长度为 1
        assert len(tokens) == 1, f"{text} is not encoded as a single token"
        # 返回第一个标记的 ID
        return tokens[0]
# 使用 lru_cache 装饰器缓存 build_tokenizer 函数的结果
@lru_cache(maxsize=None)
# 构建 tokenizer 对象的函数，接受一个字符串参数 name，默认为 "gpt2"
def build_tokenizer(name: str = "gpt2"):
    # 设置环境变量，禁用 tokenizers 的并行处理
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 拼接文件路径，获取 tokenizer 模型文件的路径
    path = os.path.join(os.path.dirname(__file__), "assets", name)
    # 从预训练模型路径加载 GPT2TokenizerFast 对象
    tokenizer = GPT2TokenizerFast.from_pretrained(path)

    # 定义特殊标记列表
    specials = [
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]

    # 向 tokenizer 添加特殊标记
    tokenizer.add_special_tokens(dict(additional_special_tokens=specials))
    # 返回构建好的 tokenizer 对象
    return tokenizer


# 使用 lru_cache 装饰器缓存 get_tokenizer 函数的结果
@lru_cache(maxsize=None)
# 获取 tokenizer 对象的函数，接受一个布尔型参数 multilingual 和两个可选参数 task 和 language
def get_tokenizer(
    multilingual: bool,
    *,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    language: Optional[str] = None,
) -> Tokenizer:
    # 如果指定了 language 参数
    if language is not None:
        # 将 language 转换为小写
        language = language.lower()
        # 如果 language 不在 LANGUAGES 中
        if language not in LANGUAGES:
            # 如果 language 在 TO_LANGUAGE_CODE 中
            if language in TO_LANGUAGE_CODE:
                # 将 language 转换为对应的语言代码
                language = TO_LANGUAGE_CODE[language]
            else:
                # 抛出异常，指定的语言不受支持
                raise ValueError(f"Unsupported language: {language}")

    # 如果 multilingual 为 True
    if multilingual:
        # 设置 tokenizer_name 为 "multilingual"
        tokenizer_name = "multilingual"
        # 如果 task 为 None，则设置 task 为 "transcribe"
        task = task or "transcribe"
        # 如果 language 为 None，则设置 language 为 "en"
        language = language or "en"
    else:
        # 设置 tokenizer_name 为 "gpt2"
        tokenizer_name = "gpt2"
        # 将 task 和 language 设置为 None
        task = None
        language = None

    # 调用 build_tokenizer 函数，获取 tokenizer 对象
    tokenizer = build_tokenizer(name=tokenizer_name)
    # 获取 tokenizer 对象的所有特殊标记的 id 列表
    all_special_ids: List[int] = tokenizer.all_special_ids
    # 获取特殊标记 "<|startoftranscript|>" 的 id
    sot: int = all_special_ids[1]
    # 获取特殊标记 "<|translate|>" 的 id
    translate: int = all_special_ids[-6]
    # 获取特殊标记 "<|transcribe|>" 的 id
    transcribe: int = all_special_ids[-5]

    # 获取 LANGUAGES 的键组成的元组
    langs = tuple(LANGUAGES.keys())
    # 创建特殊标记序列列表，初始值为 sot
    sot_sequence = [sot]
    # 如果 language 不为 None
    if language is not None:
        # 将 sot + 1 + langs.index(language) 添加到特殊标记序列列表中
        sot_sequence.append(sot + 1 + langs.index(language))
    # 如果 task 不为 None
    if task is not None:
        # 如果 task 为 "transcribe"，则将 transcribe 添加到特殊标记序列列表中，否则将 translate 添加到特殊标记序列列表中
        sot_sequence.append(transcribe if task == "transcribe" else translate)

    # 返回 Tokenizer 对象，包括 tokenizer、language 和 sot_sequence
    return Tokenizer(tokenizer=tokenizer, language=language, sot_sequence=tuple(sot_sequence))
```