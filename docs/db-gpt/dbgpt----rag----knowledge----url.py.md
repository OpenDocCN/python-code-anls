# `.\DB-GPT-src\dbgpt\rag\knowledge\url.py`

```py
"""URL Knowledge."""
# 导入必要的模块和类型声明
from typing import Any, List, Optional

from dbgpt.core import Document
from dbgpt.rag.knowledge.base import ChunkStrategy, Knowledge, KnowledgeType


class URLKnowledge(Knowledge):
    """URL Knowledge."""

    def __init__(
        self,
        url: str = "",
        knowledge_type: KnowledgeType = KnowledgeType.URL,
        source_column: Optional[str] = None,
        encoding: Optional[str] = "utf-8",
        loader: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Create URL Knowledge with Knowledge arguments.

        Args:
            url(str,  optional): url
            knowledge_type(KnowledgeType, optional): knowledge type
            source_column(str, optional): source column
            encoding(str, optional): csv encoding
            loader(Any, optional): loader
        """
        # 调用父类构造函数初始化基类的路径、知识类型和加载器
        super().__init__(
            path=url, knowledge_type=knowledge_type, loader=loader, **kwargs
        )
        # 设置编码和源列属性
        self._encoding = encoding
        self._source_column = source_column

    def _load(self) -> List[Document]:
        """Fetch URL document from loader."""
        # 如果存在加载器，调用其加载方法获取文档列表
        if self._loader:
            documents = self._loader.load()
        else:
            # 如果没有加载器，则使用默认的 WebBaseLoader 加载器加载网页内容
            from langchain.document_loaders import WebBaseLoader  # mypy: ignore

            if self._path is not None:
                # 创建 WebBaseLoader 对象并加载网页内容
                web_reader = WebBaseLoader(web_path=self._path, encoding="utf8")
                documents = web_reader.load()
            else:
                # 处理 self._path 为 None 的情况，抛出数值错误
                raise ValueError("web_path cannot be None")
        # 将 LangChain 文档转换为通用文档对象后返回文档列表
        return [Document.langchain2doc(lc_document) for lc_document in documents]

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
        # 返回默认的分块策略 CHUNK_BY_SIZE
        return ChunkStrategy.CHUNK_BY_SIZE

    @classmethod
    def type(cls):
        """Return knowledge type."""
        # 返回知识对象的类型，即 KnowledgeType.URL
        return KnowledgeType.URL
```