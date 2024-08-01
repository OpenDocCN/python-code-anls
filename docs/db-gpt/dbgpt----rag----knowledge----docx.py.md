# `.\DB-GPT-src\dbgpt\rag\knowledge\docx.py`

```py
"""Docx Knowledge."""
# 引入所需模块和类
from typing import Any, Dict, List, Optional, Union

import docx  # 导入处理 docx 文件的模块

from dbgpt.core import Document  # 导入文档处理相关的核心类
from dbgpt.rag.knowledge.base import (  # 导入知识相关的基础类和枚举
    ChunkStrategy,
    DocumentType,
    Knowledge,
    KnowledgeType,
)


class DocxKnowledge(Knowledge):
    """Docx Knowledge."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        knowledge_type: Any = KnowledgeType.DOCUMENT,
        encoding: Optional[str] = "utf-8",
        loader: Optional[Any] = None,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create Docx Knowledge with Knowledge arguments.

        Args:
            file_path(str,  optional): file path
            knowledge_type(KnowledgeType, optional): knowledge type
            encoding(str, optional): csv encoding
            loader(Any, optional): loader
        """
        # 调用父类构造函数初始化
        super().__init__(
            path=file_path,
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        # 设置编码格式
        self._encoding = encoding

    def _load(self) -> List[Document]:
        """Load docx document from loader."""
        # 如果有指定加载器，则使用加载器加载文档
        if self._loader:
            documents = self._loader.load()
        else:
            # 否则，直接读取 docx 文件
            docs = []
            doc = docx.Document(self._path)  # 创建一个 docx 文档对象
            content = []
            # 遍历文档中的段落
            for i in range(len(doc.paragraphs)):
                para = doc.paragraphs[i]
                text = para.text  # 获取段落文本内容
                content.append(text)  # 将文本内容添加到内容列表中
            metadata = {"source": self._path}  # 设置元数据，指定文档来源
            if self._metadata:
                metadata.update(self._metadata)  # 更新元数据（如果有额外的元数据）
            # 创建 Document 对象并添加到文档列表中
            docs.append(Document(content="\n".join(content), metadata=metadata))
            return docs  # 返回包含一个 Document 对象的列表
        # 返回所有加载的文档列表
        return [Document.langchain2doc(lc_document) for lc_document in documents]

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy."""
        # 返回支持的分块策略列表
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_PARAGRAPH,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
        ]

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy."""
        # 返回默认的分块策略
        return ChunkStrategy.CHUNK_BY_SIZE

    @classmethod
    def type(cls) -> KnowledgeType:
        """Return knowledge type."""
        # 返回知识类型
        return KnowledgeType.DOCUMENT

    @classmethod
    def document_type(cls) -> DocumentType:
        """Return document type."""
        # 返回文档类型
        return DocumentType.DOCX
```