# `.\DB-GPT-src\dbgpt\rag\knowledge\csv.py`

```py
"""CSV Knowledge."""
import csv
from typing import Any, Dict, List, Optional, Union

from dbgpt.core import Document
from dbgpt.rag.knowledge.base import (
    ChunkStrategy,
    DocumentType,
    Knowledge,
    KnowledgeType,
)

class CSVKnowledge(Knowledge):
    """CSV Knowledge."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = KnowledgeType.DOCUMENT,
        source_column: Optional[str] = None,
        encoding: Optional[str] = "utf-8",
        loader: Optional[Any] = None,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create CSV Knowledge with Knowledge arguments.

        Args:
            file_path(str,  optional): file path
            knowledge_type(KnowledgeType, optional): knowledge type
            source_column(str, optional): source column
            encoding(str, optional): csv encoding
            loader(Any, optional): loader
        """
        # 调用父类的初始化方法，传入文件路径、知识类型、数据加载器和元数据
        super().__init__(
            path=file_path,
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        # 设置编码和源列属性
        self._encoding = encoding
        self._source_column = source_column

    def _load(self) -> List[Document]:
        """Load csv document from loader."""
        if self._loader:
            # 如果存在加载器，使用加载器加载文档
            documents = self._loader.load()
        else:
            # 否则手动加载文档
            docs = []
            if not self._path:
                # 如果路径为空，抛出数值错误
                raise ValueError("file path is required")
            with open(self._path, newline="", encoding=self._encoding) as csvfile:
                # 打开 CSV 文件
                csv_reader = csv.DictReader(csvfile)
                for i, row in enumerate(csv_reader):
                    strs = []
                    for k, v in row.items():
                        # 清除键和值的前后空格，并组装成字符串列表
                        if k is None or v is None:
                            continue
                        strs.append(f"{k.strip()}: {v.strip()}")
                    content = "\n".join(strs)
                    try:
                        source = (
                            row[self._source_column]
                            if self._source_column is not None
                            else self._path
                        )
                    except KeyError:
                        # 如果指定的源列不存在，抛出数值错误
                        raise ValueError(
                            f"Source column '{self._source_column}' not in CSV "
                            f"file."
                        )
                    metadata = {"source": source, "row": i}
                    if self._metadata:
                        # 更新元数据
                        metadata.update(self._metadata)  # type: ignore
                    # 创建文档对象并添加到文档列表
                    doc = Document(content=content, metadata=metadata)
                    docs.append(doc)

            return docs
        # 将加载的文档列表转换为适当的 Document 对象并返回
        return [Document.langchain2doc(lc_document) for lc_document in documents]

    @classmethod
    # 返回支持的分块策略列表
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy."""
        return [
            ChunkStrategy.CHUNK_BY_SIZE,      # 按大小分块策略
            ChunkStrategy.CHUNK_BY_SEPARATOR, # 按分隔符分块策略
        ]

    # 返回默认的分块策略
    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy."""
        return ChunkStrategy.CHUNK_BY_SIZE   # 默认按大小分块策略

    # 返回CSV文件的知识类型
    @classmethod
    def type(cls) -> KnowledgeType:
        """Knowledge type of CSV."""
        return KnowledgeType.DOCUMENT       # CSV文档的知识类型

    # 返回文档的类型，此处为CSV
    @classmethod
    def document_type(cls) -> DocumentType:
        """Return document type."""
        return DocumentType.CSV              # 返回CSV文档类型
```