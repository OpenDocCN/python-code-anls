# `.\DB-GPT-src\dbgpt\rag\knowledge\datasource.py`

```py
"""Datasource Knowledge."""
# 导入所需模块和类
from typing import Any, Dict, List, Optional, Union

from dbgpt.core import Document  # 导入Document类
from dbgpt.datasource import BaseConnector  # 导入BaseConnector类

from ..summary.gdbms_db_summary import _parse_db_summary as _parse_gdb_summary  # 导入函数_parse_db_summary并重命名为_parse_gdb_summary
from ..summary.rdbms_db_summary import _parse_db_summary  # 导入函数_parse_db_summary

from .base import ChunkStrategy, DocumentType, Knowledge, KnowledgeType  # 导入相关类和枚举类型


class DatasourceKnowledge(Knowledge):
    """Datasource Knowledge."""

    def __init__(
        self,
        connector: BaseConnector,
        summary_template: str = "{table_name}({columns})",
        knowledge_type: Optional[KnowledgeType] = KnowledgeType.DOCUMENT,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create Datasource Knowledge with Knowledge arguments.

        Args:
            connector(BaseConnector): connector对象，用于连接数据源
            summary_template(str, optional): 摘要模板，用于生成摘要信息
            knowledge_type(KnowledgeType, optional): 知识类型，指明数据源的知识类型
            metadata(Dict[str, Union[str, List[str]]], optional): 元数据，包含关于数据源的附加信息
        """
        self._connector = connector  # 初始化连接器对象
        self._summary_template = summary_template  # 初始化摘要模板
        super().__init__(knowledge_type=knowledge_type, metadata=metadata, **kwargs)  # 调用父类构造函数初始化

    def _load(self) -> List[Document]:
        """Load datasource document from data_loader."""
        docs = []  # 初始化文档列表
        if self._connector.is_graph_type():
            db_summary = _parse_gdb_summary(self._connector, self._summary_template)  # 解析图数据库的摘要信息
        else:
            db_summary = _parse_db_summary(self._connector, self._summary_template)  # 解析关系型数据库的摘要信息
        for table_summary in db_summary:
            metadata = {"source": "database"}  # 设置文档的元数据来源为数据库
            if self._metadata:
                metadata.update(self._metadata)  # 如果有额外的元数据，更新到文档的元数据中
            docs.append(Document(content=table_summary, metadata=metadata))  # 创建文档对象并添加到文档列表中
        return docs  # 返回文档列表

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy."""
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
            ChunkStrategy.CHUNK_BY_PAGE,
        ]  # 返回支持的分块策略列表

    @classmethod
    def type(cls) -> KnowledgeType:
        """Knowledge type of Datasource."""
        return KnowledgeType.DOCUMENT  # 返回数据源的知识类型为文档类型

    @classmethod
    def document_type(cls) -> DocumentType:
        """Return document type."""
        return DocumentType.DATASOURCE  # 返回文档的数据源类型为DATASOURCE

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy.

        Returns:
            ChunkStrategy: default chunk strategy
        """
        return ChunkStrategy.CHUNK_BY_PAGE  # 返回默认的分块策略为按页分块
```