# `.\DB-GPT-src\dbgpt\rag\knowledge\pptx.py`

```py
"""PPTX Knowledge."""
# 导入必要的模块和类
from typing import Any, Dict, List, Optional, Union

from dbgpt.core import Document  # 导入Document类
from dbgpt.rag.knowledge.base import (
    ChunkStrategy,  # 导入ChunkStrategy枚举类型
    DocumentType,   # 导入DocumentType枚举类型
    Knowledge,      # 导入Knowledge类
    KnowledgeType,  # 导入KnowledgeType枚举类型
)


class PPTXKnowledge(Knowledge):
    """PPTX Knowledge."""

    def __init__(
        self,
        file_path: Optional[str] = None,  # 文件路径，可选
        knowledge_type: KnowledgeType = KnowledgeType.DOCUMENT,  # 知识类型，默认为DOCUMENT
        loader: Optional[Any] = None,  # 加载器，可选
        language: Optional[str] = "zh",  # 语言，默认为中文
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,  # 元数据，可选
        **kwargs: Any,
    ) -> None:
        """Create PPTX knowledge with PDF Knowledge arguments.

        Args:
            file_path:(Optional[str]) file path
            knowledge_type:(KnowledgeType) knowledge type
            loader:(Optional[Any]) loader
        """
        super().__init__(
            path=file_path,
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        self._language = language  # 设置语言属性

    def _load(self) -> List[Document]:
        """Load pdf document from loader."""
        if self._loader:  # 如果存在加载器
            documents = self._loader.load()  # 使用加载器加载文档
        else:
            from pptx import Presentation  # 导入Presentation类

            pr = Presentation(self._path)  # 使用文件路径创建Presentation对象
            docs = []
            for slide in pr.slides:  # 遍历每一页幻灯片
                content = ""
                for shape in slide.shapes:  # 遍历每个形状
                    if hasattr(shape, "text") and shape.text:  # 如果形状有文本属性且文本不为空
                        content += shape.text  # 将文本添加到内容中
                metadata = {"source": self._path}  # 元数据包含源文件路径
                if self._metadata:
                    metadata.update(self._metadata)  # 如果存在额外元数据，更新到元数据中
                docs.append(Document(content=content, metadata=metadata))  # 创建Document对象并添加到列表中
            return docs  # 返回文档列表
        return [Document.langchain2doc(lc_document) for lc_document in documents]  # 将加载的文档转换为Document对象并返回列表

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy.

        Returns:
            List[ChunkStrategy]: support chunk strategy
        """
        return [
            ChunkStrategy.CHUNK_BY_SIZE,  # 按大小分块策略
            ChunkStrategy.CHUNK_BY_PAGE,  # 按页数分块策略
            ChunkStrategy.CHUNK_BY_SEPARATOR,  # 按分隔符分块策略
        ]

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy.

        Returns:
            ChunkStrategy: default chunk strategy
        """
        return ChunkStrategy.CHUNK_BY_SIZE  # 默认按大小分块策略

    @classmethod
    def type(cls) -> KnowledgeType:
        """Knowledge type of PPTX.

        Returns:
            KnowledgeType: knowledge type
        """
        return KnowledgeType.DOCUMENT  # 返回PPTX文档的知识类型

    @classmethod
    def document_type(cls) -> DocumentType:
        """Document type of PPTX.

        Returns:
            DocumentType: document type
        """
        return DocumentType.PPTX  # 返回PPTX文档的文档类型
```