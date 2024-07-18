# `.\graphrag\graphrag\index\verbs\text\chunk\strategies\tokens.py`

```py
# 从 collections.abc 模块导入 Iterable 类型，用于声明函数返回类型
# 从 typing 模块导入 Any 类型，用于声明参数类型
import tiktoken  # 导入 tiktoken 模块
from datashaper import ProgressTicker  # 从 datashaper 模块导入 ProgressTicker 类
import graphrag.config.defaults as defs  # 导入 graphrag.config.defaults 模块中的 defs 对象
from graphrag.index.text_splitting import Tokenizer  # 从 graphrag.index.text_splitting 模块导入 Tokenizer 类
from graphrag.index.verbs.text.chunk.typing import TextChunk  # 从 graphrag.index.verbs.text.chunk.typing 模块导入 TextChunk 类


def run(
    input: list[str], args: dict[str, Any], tick: ProgressTicker
) -> Iterable[TextChunk]:
    """Chunks text into multiple parts. A pipeline verb."""
    tokens_per_chunk = args.get("chunk_size", defs.CHUNK_SIZE)  # 获取 chunk_size 参数，默认为 defs.CHUNK_SIZE
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)  # 获取 chunk_overlap 参数，默认为 defs.CHUNK_OVERLAP
    encoding_name = args.get("encoding_name", defs.ENCODING_MODEL)  # 获取 encoding_name 参数，默认为 defs.ENCODING_MODEL
    enc = tiktoken.get_encoding(encoding_name)  # 使用 encoding_name 获取编码器对象

    def encode(text: str) -> list[int]:
        """Encodes text into a list of integers using the specified encoding."""
        if not isinstance(text, str):
            text = f"{text}"  # 如果输入不是字符串，则转换为字符串
        return enc.encode(text)  # 使用 enc 对象对文本进行编码，并返回编码后的整数列表

    def decode(tokens: list[int]) -> str:
        """Decodes a list of integers back into text using the specified encoding."""
        return enc.decode(tokens)  # 使用 enc 对象对整数列表进行解码，并返回解码后的文本

    return split_text_on_tokens(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
        tick,
    )


# 从 https://github.com/langchain-ai/langchain/blob/77b359edf5df0d37ef0d539f678cf64f5557cb54/libs/langchain/langchain/text_splitter.py#L471 改编
# 以便更好地控制分块过程
def split_text_on_tokens(
    texts: list[str], enc: Tokenizer, tick: ProgressTicker
) -> list[TextChunk]:
    """Split incoming text and return chunks."""
    result = []  # 用于存储最终结果的空列表
    mapped_ids = []  # 用于存储映射 ID 的空列表

    # 遍历输入的文本列表，对每个文本进行编码和处理
    for source_doc_idx, text in enumerate(texts):
        encoded = enc.encode(text)  # 使用 Tokenizer 对象 enc 对文本进行编码
        tick(1)  # 调用 tick 函数，通知进度
        mapped_ids.append((source_doc_idx, encoded))  # 将文档索引和编码结果添加到 mapped_ids 列表中

    # 构建 input_ids 列表，其中每个元素为文档索引和对应的编码 ID
    input_ids: list[tuple[int, int]] = [
        (source_doc_idx, id) for source_doc_idx, ids in mapped_ids for id in ids
    ]

    start_idx = 0  # 开始索引位置初始化为 0
    cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))  # 当前索引位置初始化为 tokens_per_chunk 或者列表长度的最小值
    chunk_ids = input_ids[start_idx:cur_idx]  # 获取当前分块的编码 ID 列表

    # 使用 while 循环处理所有文本块
    while start_idx < len(input_ids):
        chunk_text = enc.decode([id for _, id in chunk_ids])  # 解码当前分块的编码 ID 列表
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})  # 获取当前分块的文档索引列表
        result.append(
            TextChunk(
                text_chunk=chunk_text,  # 当前文本块的文本内容
                source_doc_indices=doc_indices,  # 当前文本块所涉及的文档索引列表
                n_tokens=len(chunk_ids),  # 当前文本块的标记数量
            )
        )
        start_idx += enc.tokens_per_chunk - enc.chunk_overlap  # 更新下一个分块的起始索引位置
        cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))  # 更新当前分块的结束索引位置
        chunk_ids = input_ids[start_idx:cur_idx]  # 获取下一个分块的编码 ID 列表

    return result  # 返回最终的文本块列表
```