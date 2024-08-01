# `.\DB-GPT-src\dbgpt\rag\knowledge\factory.py`

```py
"""Knowledge Factory to create knowledge from file path and url."""
from typing import Dict, List, Optional, Type, Union

from dbgpt.rag.knowledge.base import Knowledge, KnowledgeType
from dbgpt.rag.knowledge.string import StringKnowledge
from dbgpt.rag.knowledge.url import URLKnowledge


class KnowledgeFactory:
    """Knowledge Factory to create knowledge from file path and url."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = KnowledgeType.DOCUMENT,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
    ):
        """Create Knowledge Factory with file path and knowledge type.

        Args:
            file_path(str, optional): file path
            knowledge_type(KnowledgeType, optional): knowledge type
        """
        self._file_path = file_path  # 设置文件路径变量
        self._knowledge_type = knowledge_type  # 设置知识类型变量
        self._metadata = metadata  # 设置元数据变量

    @classmethod
    def create(
        cls,
        datasource: str = "",
        knowledge_type: KnowledgeType = KnowledgeType.DOCUMENT,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
    ):
        """Create knowledge from file path, url or text.

        Args:
             datasource: path of the file to convert
             knowledge_type: type of knowledge
             metadata: Optional[Dict[str, Union[str, List[str]]]]

        Examples:
            .. code-block:: python

                from dbgpt.rag.knowledge.factory import KnowledgeFactory

                url_knowlege = KnowledgeFactory.create(
                    datasource="https://www.baidu.com", knowledge_type=KnowledgeType.URL
                )
                doc_knowlege = KnowledgeFactory.create(
                    datasource="path/to/document.pdf",
                    knowledge_type=KnowledgeType.DOCUMENT,
                )

        """
        match knowledge_type:  # 根据不同的知识类型进行分支处理
            case KnowledgeType.DOCUMENT:  # 如果是文档类型
                return cls.from_file_path(
                    file_path=datasource,  # 使用文件路径创建
                    knowledge_type=knowledge_type,
                    metadata=metadata,
                )
            case KnowledgeType.URL:  # 如果是 URL 类型
                return cls.from_url(url=datasource, knowledge_type=knowledge_type)  # 使用 URL 创建
            case KnowledgeType.TEXT:  # 如果是文本类型
                return cls.from_text(
                    text=datasource,  # 使用文本创建
                    knowledge_type=knowledge_type,
                    metadata=metadata
                )
            case _:  # 对于不支持的类型，抛出异常
                raise Exception(f"Unsupported knowledge type '{knowledge_type}'")

    @classmethod
    def from_file_path(
        cls,
        file_path: str = "",
        knowledge_type: Optional[KnowledgeType] = KnowledgeType.DOCUMENT,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> Knowledge:
        """Create knowledge from path.

        Args:
            param file_path: path of the file to convert
            param knowledge_type: type of knowledge

        Examples:
            .. code-block:: python

                from dbgpt.rag.knowledge.factory import KnowledgeFactory

                doc_knowlege = KnowledgeFactory.create(
                    datasource="path/to/document.pdf",
                    knowledge_type=KnowledgeType.DOCUMENT,
                )

        """
        # 使用类方法创建特定路径的知识对象
        factory = cls(file_path=file_path, knowledge_type=knowledge_type)
        # 调用内部方法选择文档知识对象，并返回
        return factory._select_document_knowledge(
            file_path=file_path, knowledge_type=knowledge_type, metadata=metadata
        )

    @staticmethod
    def from_url(
        url: str = "",
        knowledge_type: KnowledgeType = KnowledgeType.URL,
    ) -> Knowledge:
        """Create knowledge from url.

        Args:
            param url: url of the file to convert
            param knowledge_type: type of knowledge

        Examples:
            .. code-block:: python

                from dbgpt.rag.knowledge.factory import KnowledgeFactory

                url_knowlege = KnowledgeFactory.create(
                    datasource="https://www.baidu.com", knowledge_type=KnowledgeType.URL
                )
        """
        # 直接创建 URL 类型的知识对象并返回
        return URLKnowledge(
            url=url,
            knowledge_type=knowledge_type,
        )

    @staticmethod
    def from_text(
        text: str = "",
        knowledge_type: KnowledgeType = KnowledgeType.TEXT,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> Knowledge:
        """Create knowledge from text.

        Args:
            param text: text to convert
            param knowledge_type: type of knowledge
        """
        # 直接创建文本类型的知识对象并返回
        return StringKnowledge(
            text=text,
            knowledge_type=knowledge_type,
            metadata=metadata,
        )

    def _select_document_knowledge(self, **kwargs):
        """Select document knowledge from file path."""
        # 从文件路径中获取扩展名
        extension = self._file_path.rsplit(".", 1)[-1]
        # 获取所有知识子类
        knowledge_classes = self._get_knowledge_subclasses()
        implementation = None
        # 遍历所有子类，匹配扩展名并选择对应的实现类
        for cls in knowledge_classes:
            if cls.document_type() and cls.document_type().value == extension:
                implementation = cls(**kwargs)
        # 如果没有找到匹配的实现类，则抛出异常
        if implementation is None:
            raise Exception(f"Unsupported knowledge document type '{extension}'")
        return implementation

    @classmethod
    def all_types(cls):
        """Get all knowledge types."""
        # 返回所有知识对象的类型值列表
        return [knowledge.type().value for knowledge in cls._get_knowledge_subclasses()]

    @classmethod
    def subclasses(cls) -> List["Type[Knowledge]"]:
        """Get all knowledge subclasses."""
        # 返回所有知识对象的子类列表
        return cls._get_knowledge_subclasses()

    @staticmethod
    def _get_knowledge_subclasses() -> List["Type[Knowledge]"]:
        """Get all knowledge subclasses."""
        # 导入所需知识类别模块，这些模块包括不同类型的知识（如CSV、Excel等）
        from dbgpt.rag.knowledge.base import Knowledge  # noqa: F401
        from dbgpt.rag.knowledge.csv import CSVKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.datasource import DatasourceKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.docx import DocxKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.excel import ExcelKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.html import HTMLKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.markdown import MarkdownKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.pdf import PDFKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.pptx import PPTXKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.string import StringKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.txt import TXTKnowledge  # noqa: F401
        from dbgpt.rag.knowledge.url import URLKnowledge  # noqa: F401

        # 返回 Knowledge 类的所有子类列表
        return Knowledge.__subclasses__()
```