# `.\DB-GPT-src\dbgpt\rag\text_splitter\token_splitter.py`

```py
"""Token splitter."""
# 导入所需模块和类型提示
from typing import Callable, List, Optional

# 导入基础模型、字段、私有属性工具
from dbgpt._private.pydantic import BaseModel, Field, PrivateAttr
# 导入全局帮助函数
from dbgpt.util.global_helper import globals_helper
# 导入字符串分割工具函数
from dbgpt.util.splitter_utils import split_by_char, split_by_sep

# 默认的元数据格式长度、重叠块大小和块大小
DEFAULT_METADATA_FORMAT_LEN = 2
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_CHUNK_SIZE = 1024

# TokenTextSplitter 类，用于按单词标记拆分文本
class TokenTextSplitter(BaseModel):
    """Implementation of splitting text that looks at word tokens."""

    # 每个块的标记块大小，默认为 DEFAULT_CHUNK_SIZE
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, description="The token chunk size for each chunk."
    )
    # 拆分时每个块的重叠量，默认为 DEFAULT_CHUNK_OVERLAP
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        description="The token overlap of each chunk when splitting.",
    )
    # 默认用于按单词拆分的分隔符，默认为空格
    separator: str = Field(
        default=" ", description="Default separator for splitting into words"
    )
    # 额外的分隔符列表，用于拆分
    backup_separators: List = Field(
        default_factory=list, description="Additional separators for splitting."
    )
    # 回调管理器，用于回调函数（已注释掉）
    # callback_manager: CallbackManager = Field(
    #     default_factory=CallbackManager, exclude=True
    # )
    # 标记器，用于将单词拆分为标记的回调函数，默认为全局帮助函数中的标记器
    tokenizer: Callable = Field(
        default_factory=globals_helper.tokenizer,  # type: ignore
        description="Tokenizer for splitting words into tokens.",
        exclude=True,
    )

    # 私有属性，用于存储拆分函数列表
    _split_fns: List[Callable] = PrivateAttr()

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        # callback_manager: Optional[CallbackManager] = None,
        separator: str = " ",
        backup_separators=None,
    ):
        """Initialize with parameters."""
        # 如果没有提供备用分隔符，则默认为换行符
        if backup_separators is None:
            backup_separators = ["\n"]
        # 如果重叠块大于块大小，则引发值错误异常
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        # 如果未指定标记器，则使用全局帮助函数中的标记器
        # callback_manager = callback_manager or CallbackManager([])
        tokenizer = tokenizer or globals_helper.tokenizer

        # 组合所有分隔符，包括主分隔符和备用分隔符
        all_seps = [separator] + (backup_separators or [])

        # 调用父类的初始化方法，设置初始属性值
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            backup_separators=backup_separators,
            # callback_manager=callback_manager,
            tokenizer=tokenizer,
        )

        # 创建用于拆分的函数列表，基于所有分隔符
        self._split_fns = [split_by_sep(sep) for sep in all_seps] + [split_by_char()]

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        # 返回类的名称字符串
        return "TokenTextSplitter"
    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        """Split text into chunks, reserving space required for metadata str."""
        # 计算元数据字符串经过分词器处理后的长度，并加上默认的元数据格式长度
        metadata_len = len(self.tokenizer(metadata_str)) + DEFAULT_METADATA_FORMAT_LEN
        # 计算有效的分块大小，减去元数据长度
        effective_chunk_size = self.chunk_size - metadata_len
        # 如果有效分块大小小于等于零，则抛出数值错误异常
        if effective_chunk_size <= 0:
            raise ValueError(
                f"Metadata length ({metadata_len}) is longer than chunk size "
                f"({self.chunk_size}). Consider increasing the chunk size or "
                "decreasing the size of your metadata to avoid this."
            )
        # 如果有效分块大小小于50，打印警告信息
        elif effective_chunk_size < 50:
            print(
                f"Metadata length ({metadata_len}) is close to chunk size "
                f"({self.chunk_size}). Resulting chunks are less than 50 tokens. "
                "Consider increasing the chunk size or decreasing the size of "
                "your metadata to avoid this.",
                flush=True,
            )

        # 调用内部方法 _split_text 进行文本分块处理
        return self._split_text(text, chunk_size=effective_chunk_size)

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        # 调用内部方法 _split_text 进行文本分块处理，使用默认的分块大小
        return self._split_text(text, chunk_size=self.chunk_size)

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks up to chunk_size."""
        # 如果文本为空，则返回空列表
        if text == "":
            return []

        # 调用 _split 方法进行实际的文本分块处理
        splits = self._split(text, chunk_size)
        # 合并分块结果，确保每个分块大小不超过指定的 chunk_size
        chunks = self._merge(splits, chunk_size)
        return chunks

    def _split(self, text: str, chunk_size: int) -> List[str]:
        """Break text into splits that are smaller than chunk size.

        The order of splitting is:
        1. split by separator
        2. split by backup separators (if any)
        3. split by characters

        NOTE: the splits contain the separators.
        """
        # 如果文本的分词长度小于等于 chunk_size，则直接返回包含整个文本的列表
        if len(self.tokenizer(text)) <= chunk_size:
            return [text]

        # 遍历所有的分割函数进行文本分块操作
        for split_fn in self._split_fns:
            splits = split_fn(text)
            # 如果分割后的结果数量大于1，则认为成功分块
            if len(splits) > 1:
                break

        new_splits = []
        # 遍历所有分块结果，根据每个分块的长度判断是否需要进一步递归分块处理
        for split in splits:
            split_len = len(self.tokenizer(split))
            if split_len <= chunk_size:
                new_splits.append(split)
            else:
                # 递归调用自身进行分块处理
                new_splits.extend(self._split(split, chunk_size=chunk_size))
        return new_splits
    def _merge(self, splits: List[str], chunk_size: int) -> List[str]:
        """Merge splits into chunks.

        The high-level idea is to keep adding splits to a chunk until we
        exceed the chunk size, then we start a new chunk with overlap.

        When we start a new chunk, we pop off the first element of the previous
        chunk until the total length is less than the chunk size.
        """
        # 初始化一个空列表用于存储最终的合并后的块
        chunks: List[str] = []

        # 初始化当前块的内容和长度
        cur_chunk: List[str] = []
        cur_len = 0

        # 遍历每个分割片段
        for split in splits:
            # 计算当前分割片段的长度
            split_len = len(self.tokenizer(split))

            # 如果当前分割片段的长度大于指定的块大小，则输出警告信息
            if split_len > chunk_size:
                print(
                    f"Got a split of size {split_len}, ",
                    f"larger than chunk size {chunk_size}.",
                )

            # 如果加入当前分割片段后超过了块大小，则结束当前块并开始一个新块
            if cur_len + split_len > chunk_size:
                # 结束前一个块，并将其添加到块列表中
                chunk = "".join(cur_chunk).strip()
                if chunk:
                    chunks.append(chunk)

                # 开始一个新的块并处理重叠部分
                # 持续移除前一个块的第一个元素，直到当前块长度小于重叠部分的长度
                # 或者总长度小于块大小
                while cur_len > self.chunk_overlap or cur_len + split_len > chunk_size:
                    # 弹出前一个块的第一个元素
                    first_chunk = cur_chunk.pop(0)
                    cur_len -= len(self.tokenizer(first_chunk))

            # 将当前分割片段加入当前块中，并更新当前块长度
            cur_chunk.append(split)
            cur_len += split_len

        # 处理最后一个块
        chunk = "".join(cur_chunk).strip()
        if chunk:
            chunks.append(chunk)

        # 返回合并后的块列表
        return chunks
```