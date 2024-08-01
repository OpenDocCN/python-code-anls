# `.\DB-GPT-src\dbgpt\rag\knowledge\pdf.py`

```py
"""PDF Knowledge."""
# 导入必要的模块和类
from typing import Any, Dict, List, Optional, Union

from dbgpt.core import Document
from dbgpt.rag.knowledge.base import (
    ChunkStrategy,
    DocumentType,
    Knowledge,
    KnowledgeType,
)

# PDFKnowledge 类，继承自 Knowledge 类
class PDFKnowledge(Knowledge):
    """PDF Knowledge."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        knowledge_type: KnowledgeType = KnowledgeType.DOCUMENT,
        loader: Optional[Any] = None,
        language: Optional[str] = "zh",
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create PDF Knowledge with Knowledge arguments.

        Args:
            file_path(str,  optional): file path
            knowledge_type(KnowledgeType, optional): knowledge type
            loader(Any, optional): loader
            language(str, optional): language
        """
        # 调用父类的构造方法初始化
        super().__init__(
            path=file_path,
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        # 设置语言属性
        self._language = language

    def _load(self) -> List[Document]:
        """Load pdf document from loader."""
        # 如果有指定 loader，则使用 loader 加载文档
        if self._loader:
            documents = self._loader.load()
        else:
            # 否则使用默认的 PDF 解析库加载文档
            import pypdf

            pages = []
            documents = []
            # 如果未提供文件路径，则抛出数值错误异常
            if not self._path:
                raise ValueError("file path is required")
            with open(self._path, "rb") as file:
                reader = pypdf.PdfReader(file)
                # 遍历 PDF 的每一页
                for page_num in range(len(reader.pages)):
                    _page = reader.pages[page_num]
                    # 提取页面文本和页码，存入 pages 列表
                    pages.append((_page.extract_text(), page_num))

            # 清理页面内容
            for page, page_num in pages:
                lines = page.splitlines()

                cleaned_lines = []
                # 根据语言选择分割单词的方法，并清理行内容
                for line in lines:
                    if self._language == "en":
                        words = list(line)  # noqa: F841
                    else:
                        words = line.split()  # noqa: F841
                    cleaned_lines.append(line)
                page = "\n".join(cleaned_lines)
                
                # 构建文档的元数据
                metadata = {"source": self._path, "page": page_num}
                if self._metadata:
                    metadata.update(self._metadata)  # type: ignore
                # 创建 Document 对象并添加到 documents 列表中
                document = Document(content=page, metadata=metadata)
                documents.append(document)
            return documents
        # 将 documents 列表中的 lc_document 转换为 Document 对象并返回
        return [Document.langchain2doc(lc_document) for lc_document in documents]

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy."""
        # 返回支持的分块策略列表
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_PAGE,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
        ]

    @classmethod
    # 返回默认的分块策略，这里使用了类方法声明
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy."""
        return ChunkStrategy.CHUNK_BY_SIZE

    # 返回知识类型，这里使用了类方法声明
    @classmethod
    def type(cls) -> KnowledgeType:
        """Return knowledge type."""
        return KnowledgeType.DOCUMENT

    # 返回文档类型，这里使用了类方法声明
    @classmethod
    def document_type(cls) -> DocumentType:
        """Document type of PDF."""
        return DocumentType.PDF
```