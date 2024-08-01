# `.\DB-GPT-src\dbgpt\rag\knowledge\html.py`

```py
"""HTML Knowledge."""
# 引入必要的模块和类型定义
from typing import Any, Dict, List, Optional, Union

import chardet  # 导入字符编码检测模块

# 导入需要的类和枚举类型
from dbgpt.core import Document
from dbgpt.rag.knowledge.base import (
    ChunkStrategy,
    DocumentType,
    Knowledge,
    KnowledgeType,
)


class HTMLKnowledge(Knowledge):
    """HTML Knowledge."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        knowledge_type: KnowledgeType = KnowledgeType.DOCUMENT,
        loader: Optional[Any] = None,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create HTML Knowledge with Knowledge arguments.

        Args:
            file_path(str, optional): 文件路径
            knowledge_type(KnowledgeType, optional): 知识类型
            loader(Any, optional): 加载器
        """
        super().__init__(
            path=file_path,
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )

    def _load(self) -> List[Document]:
        """Load html document from loader."""
        # 如果有加载器，从加载器中加载文档
        if self._loader:
            documents = self._loader.load()
        else:
            # 如果没有加载器，则从文件路径加载文档
            if not self._path:
                raise ValueError("file path is required")
            with open(self._path, "rb") as f:
                raw_text = f.read()
                # 检测文本编码
                result = chardet.detect(raw_text)
                if result["encoding"] is None:
                    text = raw_text.decode("utf-8")
                else:
                    text = raw_text.decode(result["encoding"])
            metadata = {"source": self._path}
            if self._metadata:
                metadata.update(self._metadata)  # 更新元数据
            return [Document(content=text, metadata=metadata)]

        return [Document.langchain2doc(lc_document) for lc_document in documents]

    def _postprocess(self, documents: List[Document]):
        import markdown  # 导入markdown模块

        # 对加载的文档进行后处理
        for i, d in enumerate(documents):
            content = markdown.markdown(d.content)  # 将文档内容转换为Markdown格式
            from bs4 import BeautifulSoup  # 导入BeautifulSoup模块用于解析HTML

            soup = BeautifulSoup(content, "html.parser")
            # 移除特定的HTML标签
            for tag in soup(["!doctype", "meta", "i.fa"]):
                tag.extract()
            documents[i].content = soup.get_text()  # 获取处理后的文本内容
            documents[i].content = documents[i].content.replace("\n", " ")  # 替换换行符为单个空格
        return documents

    @classmethod
    def support_chunk_strategy(cls):
        """Return support chunk strategy."""
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
        ]

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy."""
        return ChunkStrategy.CHUNK_BY_SIZE

    @classmethod
    def type(cls) -> KnowledgeType:
        """Return knowledge type."""
        return KnowledgeType.DOCUMENT

    @classmethod
    # 定义一个类方法 document_type，返回类型为 DocumentType 的枚举值
    def document_type(cls) -> DocumentType:
        """Return document type."""
        # 返回枚举值 DocumentType.HTML，表示返回 HTML 类型的文档
        return DocumentType.HTML
```