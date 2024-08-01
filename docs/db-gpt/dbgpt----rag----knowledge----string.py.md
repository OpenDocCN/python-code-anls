# `.\DB-GPT-src\dbgpt\rag\knowledge\string.py`

```py
"""String Knowledge."""
# 导入必要的模块和类
from typing import Any, Dict, List, Optional, Union

from dbgpt.core import Document
from dbgpt.rag.knowledge.base import ChunkStrategy, Knowledge, KnowledgeType

# 定义 StringKnowledge 类，继承自 Knowledge 类
class StringKnowledge(Knowledge):
    """String Knowledge."""

    def __init__(
        self,
        text: str = "",
        knowledge_type: KnowledgeType = KnowledgeType.TEXT,
        encoding: Optional[str] = "utf-8",
        loader: Optional[Any] = None,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create String knowledge parameters.

        Args:
            text(str): 文本内容
            knowledge_type(KnowledgeType): 知识类型
            encoding(str): 编码方式
            loader(Any): 载入器
            metadata(Optional[Dict[str, Union[str, List[str]]]]): 元数据
        """
        # 调用父类的构造函数进行初始化
        super().__init__(
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        # 设置对象的私有属性
        self._text = text
        self._encoding = encoding

    def _load(self) -> List[Document]:
        """Load raw text from loader."""
        # 设置元数据信息
        metadata = {"source": "raw text"}
        # 如果存在额外的元数据，进行更新
        if self._metadata:
            metadata.update(self._metadata)  # type: ignore
        # 创建文档对象列表，包含从文本创建的 Document 对象
        docs = [Document(content=self._text, metadata=metadata)]
        return docs

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy."""
        # 返回支持的分块策略列表
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
        ]

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy."""
        # 返回默认的分块策略
        return ChunkStrategy.CHUNK_BY_SIZE

    @classmethod
    def type(cls):
        """Return knowledge type."""
        # 返回知识对象的类型
        return KnowledgeType.TEXT
```