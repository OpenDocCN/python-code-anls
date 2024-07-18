# `.\graphrag\graphrag\index\text_splitting\text_splitting.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""包含 'Tokenizer', 'TextSplitter', 'NoopTextSplitter' 和 'TokenTextSplitter' 模型的模块。"""

import json  # 导入 json 模块
import logging  # 导入 logging 模块
from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 类和 abstractmethod 装饰器
from collections.abc import Callable, Collection, Iterable  # 从 collections.abc 模块导入 Callable、Collection 和 Iterable 抽象基类
from dataclasses import dataclass  # 导入 dataclass 装饰器
from enum import Enum  # 导入 Enum 枚举类
from typing import Any, Literal, cast  # 导入类型提示 Any、Literal 和 cast 函数

import pandas as pd  # 导入 pandas 库
import tiktoken  # 导入 tiktoken 模块

from graphrag.index.utils import num_tokens_from_string  # 从 graphrag.index.utils 模块导入 num_tokens_from_string 函数

EncodedText = list[int]  # 定义 EncodedText 类型别名为整数列表
DecodeFn = Callable[[EncodedText], str]  # 定义 DecodeFn 类型别名为接受 EncodedText 返回字符串的可调用对象
EncodeFn = Callable[[str], EncodedText]  # 定义 EncodeFn 类型别名为接受字符串返回 EncodedText 的可调用对象
LengthFn = Callable[[str], int]  # 定义 LengthFn 类型别名为接受字符串返回整数的可调用对象

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer 数据类。"""
    
    chunk_overlap: int
    """chunk_overlap 属性：片段之间的重叠标记数"""
    tokens_per_chunk: int
    """tokens_per_chunk 属性：每个片段的最大标记数"""
    decode: DecodeFn
    """decode 属性：将标记 id 列表解码为字符串的函数"""
    encode: EncodeFn
    """encode 属性：将字符串编码为标记 id 列表的函数"""


class TextSplitter(ABC):
    """文本拆分器类定义。"""

    _chunk_size: int
    """_chunk_size 属性：片段大小"""
    _chunk_overlap: int
    """_chunk_overlap 属性：片段重叠数"""
    _length_function: LengthFn
    """_length_function 属性：计算字符串长度的函数"""
    _keep_separator: bool
    """_keep_separator 属性：是否保留分隔符"""
    _add_start_index: bool
    """_add_start_index 属性：是否添加起始索引"""
    _strip_whitespace: bool
    """_strip_whitespace 属性：是否去除空白"""

    def __init__(
        self,
        chunk_size: int = 8191,
        chunk_overlap: int = 100,
        length_function: LengthFn = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ):
        """初始化方法定义。"""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """拆分文本方法定义。"""


class NoopTextSplitter(TextSplitter):
    """空操作文本拆分器类定义。"""

    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """拆分文本方法定义。"""
        return [text] if isinstance(text, str) else text


class TokenTextSplitter(TextSplitter):
    """标记文本拆分器类定义。"""

    _allowed_special: Literal["all"] | set[str]
    """_allowed_special 属性：允许的特殊标记集合"""
    _disallowed_special: Literal["all"] | Collection[str]
    """_disallowed_special 属性：不允许的特殊标记集合"""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: str | None = None,
        allowed_special: Literal["all"] | set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ):
        """初始化方法定义。"""
        super().__init__(**kwargs)
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special
    ):
        """Init method definition."""
        # 调用父类的初始化方法，传递关键字参数
        super().__init__(**kwargs)
        # 如果指定了模型名称，尝试获取其对应的编码器
        if model_name is not None:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            # 如果指定的模型名称不存在，则记录异常并使用默认编码器
            except KeyError:
                log.exception("Model %s not found, using %s", model_name, encoding_name)
                enc = tiktoken.get_encoding(encoding_name)
        else:
            # 如果未指定模型名称，则使用指定的编码器名称获取编码器
            enc = tiktoken.get_encoding(encoding_name)
        # 将获取到的编码器赋值给对象的_tokenizer属性
        self._tokenizer = enc
        # 设置对象的特殊允许字符集，如果未指定则使用空集合
        self._allowed_special = allowed_special or set()
        # 设置对象的特殊禁止字符集
        self._disallowed_special = disallowed_special

    def encode(self, text: str) -> list[int]:
        """Encode the given text into an int-vector."""
        # 调用_tokenizer对象的encode方法进行文本编码
        return self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        # 返回文本中的 token 数量，通过调用encode方法计算其长度
        return len(self.encode(text))

    def split_text(self, text: str | list[str]) -> list[str]:
        """Split text method."""
        # 如果输入的文本为NaN或空字符串，则返回空列表
        if cast(bool, pd.isna(text)) or text == "":
            return []
        # 如果输入的文本为列表，则将其合并为一个字符串
        if isinstance(text, list):
            text = " ".join(text)
        # 如果输入的文本不是字符串，则抛出类型错误异常
        if not isinstance(text, str):
            msg = f"Attempting to split a non-string value, actual is {type(text)}"
            raise TypeError(msg)

        # 创建一个新的分词器对象，用于处理文本的分词
        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda text: self.encode(text),
        )

        # 调用外部函数split_text_on_tokens，使用创建的分词器对文本进行分割
        return split_text_on_tokens(text=text, tokenizer=tokenizer)
class TextListSplitterType(str, Enum):
    """Enum for the type of the TextListSplitter."""
    DELIMITED_STRING = "delimited_string"
    JSON = "json"


class TextListSplitter(TextSplitter):
    """Text list splitter class definition."""

    def __init__(
        self,
        chunk_size: int,
        splitter_type: TextListSplitterType = TextListSplitterType.JSON,
        input_delimiter: str | None = None,
        output_delimiter: str | None = None,
        model_name: str | None = None,
        encoding_name: str | None = None,
    ):
        """Initialize the TextListSplitter with a chunk size."""
        # Set the chunk overlap to 0 as we use full strings
        super().__init__(chunk_size, chunk_overlap=0)
        self._type = splitter_type  # 设置文本分割器的类型
        self._input_delimiter = input_delimiter  # 设置输入文本列表的分隔符
        self._output_delimiter = output_delimiter or "\n"  # 设置输出文本列表的分隔符，默认为换行符
        self._length_function = lambda x: num_tokens_from_string(
            x, model=model_name, encoding_name=encoding_name
        )  # 定义一个计算字符串长度的函数，用于估算文本长度

    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split a string list into a list of strings for a given chunk size."""
        if not text:
            return []

        result: list[str] = []  # 初始化结果列表
        current_chunk: list[str] = []  # 初始化当前分块列表

        # Add the brackets
        current_length: int = self._length_function("[]")  # 计算包含空列表符号的长度

        # Input should be a string list joined by a delimiter
        string_list = self._load_text_list(text)  # 加载文本列表数据

        if len(string_list) == 1:
            return string_list  # 如果只有一个元素，直接返回该元素

        for item in string_list:
            # Count the length of the item and add comma
            item_length = self._length_function(f"{item},")  # 计算每个元素及其后面逗号的长度

            if current_length + item_length > self._chunk_size:
                if current_chunk and len(current_chunk) > 0:
                    # Add the current chunk to the result
                    self._append_to_result(result, current_chunk)  # 将当前分块加入结果列表

                    # Start a new chunk
                    current_chunk = [item]  # 开始一个新的分块
                    current_length = item_length  # 更新当前分块的长度
            else:
                # Add the item to the current chunk
                current_chunk.append(item)  # 将元素加入当前分块
                current_length += item_length  # 更新当前分块的长度

        # Add the last chunk to the result
        self._append_to_result(result, current_chunk)  # 将最后一个分块加入结果列表

        return result  # 返回最终的分块结果列表

    def _load_text_list(self, text: str | list[str]):
        """Load the text list based on the type."""
        if isinstance(text, list):
            string_list = text  # 如果输入已经是列表，则直接使用
        elif self._type == TextListSplitterType.JSON:
            string_list = json.loads(text)  # 如果是 JSON 类型，解析成列表
        else:
            string_list = text.split(self._input_delimiter)  # 否则按照指定分隔符切分成列表
        return string_list  # 返回加载后的文本列表
    def _append_to_result(self, chunk_list: list[str], new_chunk: list[str]):
        """将当前块添加到结果中。"""
        # 检查新块是否存在且非空
        if new_chunk and len(new_chunk) > 0:
            # 根据输出类型进行处理
            if self._type == TextListSplitterType.JSON:
                # 如果输出类型为 JSON，则将新块转换为 JSON 格式并添加到列表中
                chunk_list.append(json.dumps(new_chunk))
            else:
                # 否则，将新块使用输出分隔符连接成字符串后添加到列表中
                chunk_list.append(self._output_delimiter.join(new_chunk))
# 使用指定的分词器对文本进行分块处理，并返回分块后的结果列表
def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]:
    """Split incoming text and return chunks using tokenizer."""
    # 初始化空列表，用于存储分块后的文本片段
    splits: list[str] = []
    
    # 使用 tokenizer 对文本进行编码，得到输入文本的标记 ID 列表
    input_ids = tokenizer.encode(text)
    
    # 设置起始索引为 0
    start_idx = 0
    
    # 计算当前分块的结束索引，不超过标记 ID 列表的长度
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    
    # 获取当前分块的标记 ID 子列表
    chunk_ids = input_ids[start_idx:cur_idx]
    
    # 当起始索引小于输入标记 ID 列表的长度时，执行循环
    while start_idx < len(input_ids):
        # 将当前分块的标记 ID 子列表解码为文本，并添加到 splits 列表中
        splits.append(tokenizer.decode(chunk_ids))
        
        # 更新起始索引，移动到下一个分块的起始位置
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        
        # 计算下一个分块的结束索引，不超过标记 ID 列表的长度
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        
        # 获取下一个分块的标记 ID 子列表
        chunk_ids = input_ids[start_idx:cur_idx]
    
    # 返回分块后的文本片段列表
    return splits
```