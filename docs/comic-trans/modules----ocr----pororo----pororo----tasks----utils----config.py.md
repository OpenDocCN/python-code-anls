# `.\comic-translate\modules\ocr\pororo\pororo\tasks\utils\config.py`

```py
from dataclasses import dataclass
from typing import Union

# 定义数据类 TransformerConfig，用于存储 Transformer 模型的配置信息
@dataclass
class TransformerConfig:
    src_dict: Union[str, None]   # 源语言词典文件路径或者 None
    tgt_dict: Union[str, None]   # 目标语言词典文件路径或者 None
    src_tok: Union[str, None]    # 源语言的分词器文件路径或者 None
    tgt_tok: Union[str, None]    # 目标语言的分词器文件路径或者 None

# 配置字典，存储各种 Transformer 模型的具体配置
CONFIGS = {
    "transformer.base.ko.const":
        TransformerConfig(
            "dict.transformer.base.ko.const",   # 源语言和目标语言词典文件路径相同
            "dict.transformer.base.ko.const",
            None,   # 没有指定分词器文件路径
            None,
        ),
    "transformer.base.ko.pg":
        TransformerConfig(
            "dict.transformer.base.ko.mt",     # 源语言词典文件路径
            "dict.transformer.base.ko.mt",     # 目标语言词典文件路径
            "bpe8k.ko",                        # 源语言分词器文件路径
            None,                              # 没有指定目标语言分词器文件路径
        ),
    "transformer.base.ko.pg_long":
        TransformerConfig(
            "dict.transformer.base.ko.mt",     # 源语言词典文件路径
            "dict.transformer.base.ko.mt",     # 目标语言词典文件路径
            "bpe8k.ko",                        # 源语言分词器文件路径
            None,                              # 没有指定目标语言分词器文件路径
        ),
    "transformer.base.en.gec":
        TransformerConfig(
            "dict.transformer.base.en.mt",     # 源语言词典文件路径
            "dict.transformer.base.en.mt",     # 目标语言词典文件路径
            "bpe32k.en",                       # 源语言分词器文件路径
            None,                              # 没有指定目标语言分词器文件路径
        ),
    "transformer.base.zh.pg":
        TransformerConfig(
            "dict.transformer.base.zh.mt",     # 源语言词典文件路径
            "dict.transformer.base.zh.mt",     # 目标语言词典文件路径
            None,                              # 没有指定源语言分词器文件路径
            None,
        ),
    "transformer.base.ja.pg":
        TransformerConfig(
            "dict.transformer.base.ja.mt",     # 源语言词典文件路径
            "dict.transformer.base.ja.mt",     # 目标语言词典文件路径
            "bpe8k.ja",                        # 源语言分词器文件路径
            None,                              # 没有指定目标语言分词器文件路径
        ),
    "transformer.base.zh.const":
        TransformerConfig(
            "dict.transformer.base.zh.const",  # 源语言和目标语言词典文件路径相同
            "dict.transformer.base.zh.const",
            None,   # 没有指定分词器文件路径
            None,
        ),
    "transformer.base.en.const":
        TransformerConfig(
            "dict.transformer.base.en.const",  # 源语言和目标语言词典文件路径相同
            "dict.transformer.base.en.const",
            None,   # 没有指定分词器文件路径
            None,
        ),
    "transformer.base.en.pg":
        TransformerConfig(
            "dict.transformer.base.en.mt",     # 源语言词典文件路径
            "dict.transformer.base.en.mt",     # 目标语言词典文件路径
            "bpe32k.en",                       # 源语言分词器文件路径
            None,                              # 没有指定目标语言分词器文件路径
        ),
    "transformer.base.ko.gec":
        TransformerConfig(
            "dict.transformer.base.ko.gec",    # 源语言和目标语言词典文件路径相同
            "dict.transformer.base.ko.gec",
            "bpe8k.ko",                        # 源语言分词器文件路径
            None,                              # 没有指定目标语言分词器文件路径
        ),
    "transformer.base.en.char_gec":
        TransformerConfig(
            "dict.transformer.base.en.char_gec",  # 源语言和目标语言词典文件路径相同
            "dict.transformer.base.en.char_gec",
            None,   # 没有指定分词器文件路径
            None,
        ),
    "transformer.base.en.caption":
        TransformerConfig(
            None,   # 没有指定源语言词典文件路径
            None,   # 没有指定目标语言词典文件路径
            None,   # 没有指定源语言分词器文件路径
            None,   # 没有指定目标语言分词器文件路径
        ),
    "transformer.base.ja.p2g":
        TransformerConfig(
            "dict.transformer.base.ja.p2g",     # 源语言和目标语言词典文件路径相同
            "dict.transformer.base.ja.p2g",
            None,   # 没有指定分词器文件路径
            None,
        ),
    "transformer.large.multi.mtpg":
        TransformerConfig(
            "dict.transformer.large.multi.mtpg",  # 源语言和目标语言词典文件路径相同
            "dict.transformer.large.multi.mtpg",
            "bpe32k.en",                          # 源语言分词器文件路径
            None,                                 # 没有指定目标语言分词器文件路径
        ),
    # 定义一个键为字符串 "transformer.large.multi.fast.mtpg"，值为 TransformerConfig 对象的字典项
    "transformer.large.multi.fast.mtpg":
        # 创建 TransformerConfig 对象，参数依次为字典文件名、模型文件名、bpe 模型文件名和附加信息（此处为 None）
        TransformerConfig(
            "dict.transformer.large.multi.mtpg",
            "dict.transformer.large.multi.mtpg",
            "bpe32k.en",
            None,
        ),
    # 定义一个键为字符串 "transformer.large.ko.wsd"，值为 TransformerConfig 对象的字典项
    "transformer.large.ko.wsd":
        # 创建 TransformerConfig 对象，参数依次为字典文件名、模型文件名、bpe 模型文件名和附加信息（这两项均为 None）
        TransformerConfig(
            "dict.transformer.large.ko.wsd",
            "dict.transformer.large.ko.wsd",
            None,
            None,
        ),
}



# 这行代码是一个代码块的结尾，用于结束一个代码块，通常用于结束循环、条件语句或函数定义。
```