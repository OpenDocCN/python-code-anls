# `.\DB-GPT-src\dbgpt\rag\knowledge\markdown.py`

```py
"""Markdown Knowledge."""
# 导入所需模块和类
from typing import Any, Dict, List, Optional, Union

from dbgpt.core import Document
from dbgpt.rag.knowledge.base import (
    ChunkStrategy,
    DocumentType,
    Knowledge,
    KnowledgeType,
)

class MarkdownKnowledge(Knowledge):
    """Markdown Knowledge."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        knowledge_type: KnowledgeType = KnowledgeType.DOCUMENT,
        encoding: Optional[str] = "utf-8",
        loader: Optional[Any] = None,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create Markdown Knowledge with Knowledge arguments.

        Args:
            file_path(str,  optional): file path
            knowledge_type(KnowledgeType, optional): knowledge type
            encoding(str, optional): csv encoding
            loader(Any, optional): loader
        """
        # 调用父类构造函数初始化基础知识对象
        super().__init__(
            path=file_path,
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        # 设置编码
        self._encoding = encoding

    def _load(self) -> List[Document]:
        """Load markdown document from loader."""
        # 如果存在加载器，则使用加载器加载文档
        if self._loader:
            documents = self._loader.load()
        else:
            # 如果路径不存在，则抛出数值错误
            if not self._path:
                raise ValueError("file path is required")
            # 使用指定编码打开文件，并忽略错误
            with open(self._path, encoding=self._encoding, errors="ignore") as f:
                markdown_text = f.read()
                metadata = {"source": self._path}
                # 如果有元数据，则更新到现有元数据中
                if self._metadata:
                    metadata.update(self._metadata)  # type: ignore
                # 创建包含单个 Markdown 文档的文档列表
                documents = [Document(content=markdown_text, metadata=metadata)]
                return documents  # 返回文档列表
        # 将加载得到的文档转换为 Document 对象列表
        return [Document.langchain2doc(lc_document) for lc_document in documents]

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy."""
        # 返回支持的文档分块策略列表
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_MARKDOWN_HEADER,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
        ]

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy."""
        # 返回默认的文档分块策略
        return ChunkStrategy.CHUNK_BY_MARKDOWN_HEADER

    @classmethod
    def type(cls) -> KnowledgeType:
        """Return knowledge type."""
        # 返回知识对象的类型
        return KnowledgeType.DOCUMENT

    @classmethod
    def document_type(cls) -> DocumentType:
        """Return document type."""
        # 返回文档对象的类型
        return DocumentType.MARKDOWN
```