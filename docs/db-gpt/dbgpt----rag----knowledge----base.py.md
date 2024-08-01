# `.\DB-GPT-src\dbgpt\rag\knowledge\base.py`

```py
"""Module for Knowledge Base."""

# 导入必要的模块
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# 导入文档处理相关的模块
from dbgpt.core import Document
from dbgpt.rag.text_splitter.text_splitter import (
    MarkdownHeaderTextSplitter,
    PageTextSplitter,
    ParagraphTextSplitter,
    RecursiveCharacterTextSplitter,
    SeparatorTextSplitter,
    TextSplitter,
)

# 定义文档类型的枚举
class DocumentType(Enum):
    """Document Type Enum."""
    
    PDF = "pdf"
    CSV = "csv"
    MARKDOWN = "md"
    PPTX = "pptx"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    DATASOURCE = "datasource"
    EXCEL = "xlsx"

# 定义知识类型的枚举
class KnowledgeType(Enum):
    """Knowledge Type Enum."""
    
    DOCUMENT = "DOCUMENT"
    URL = "URL"
    TEXT = "TEXT"
    # TODO: Remove this type
    FIN_REPORT = "FIN_REPORT"

    @property
    def type(self):
        """Get type."""
        return DocumentType

    @classmethod
    def get_by_value(cls, value) -> "KnowledgeType":
        """Get Enum member by value.

        Args:
            value(any): value

        Returns:
            KnowledgeType: Enum member
        """
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid value for {cls.__name__}")

# 定义文本分块策略的枚举
_STRATEGY_ENUM_TYPE = Tuple[Type[TextSplitter], List, str, str]

class ChunkStrategy(Enum):
    """Chunk Strategy Enum."""
    
    CHUNK_BY_SIZE: _STRATEGY_ENUM_TYPE = (
        RecursiveCharacterTextSplitter,
        [
            {
                "param_name": "chunk_size",
                "param_type": "int",
                "default_value": 512,
                "description": "The size of the data chunks used in processing.",
            },
            {
                "param_name": "chunk_overlap",
                "param_type": "int",
                "default_value": 50,
                "description": "The amount of overlap between adjacent data chunks.",
            },
        ],
        "chunk size",
        "split document by chunk size",
    )
    CHUNK_BY_PAGE: _STRATEGY_ENUM_TYPE = (
        PageTextSplitter,
        [],
        "page",
        "split document by page",
    )
    CHUNK_BY_PARAGRAPH: _STRATEGY_ENUM_TYPE = (
        ParagraphTextSplitter,
        [
            {
                "param_name": "separator",
                "param_type": "string",
                "default_value": "\\n",
                "description": "paragraph separator",
            }
        ],
        "paragraph",
        "split document by paragraph",
    )
    CHUNK_BY_SEPARATOR: _STRATEGY_ENUM_TYPE = (
        SeparatorTextSplitter,  # 使用 SeparatorTextSplitter 类来处理分割
        [  # 参数列表
            {
                "param_name": "separator",  # 参数名
                "param_type": "string",  # 参数类型
                "default_value": "\\n",  # 默认值为换行符
                "description": "chunk separator",  # 参数描述
            },
            {
                "param_name": "enable_merge",  # 参数名
                "param_type": "boolean",  # 参数类型
                "default_value": False,  # 默认值为 False
                "description": "Whether to merge according to the chunk_size after "
                "splitting by the separator.",  # 参数描述
            },
        ],
        "separator",  # 别名为 "separator"
        "split document by separator",  # 描述为使用分隔符分割文档
    )

    CHUNK_BY_MARKDOWN_HEADER: _STRATEGY_ENUM_TYPE = (
        MarkdownHeaderTextSplitter,  # 使用 MarkdownHeaderTextSplitter 类来处理分割
        [],  # 无额外参数
        "markdown header",  # 别名为 "markdown header"
        "split document by markdown header",  # 描述为使用 Markdown 标题分割文档
    )

    def __init__(self, splitter_class, parameters, alias, description):
        """Create a new ChunkStrategy with the given splitter_class."""
        self.splitter_class = splitter_class  # 初始化分割器类
        self.parameters = parameters  # 初始化参数列表
        self.alias = alias  # 初始化别名
        self.description = description  # 初始化描述

    def match(self, *args, **kwargs) -> TextSplitter:
        """Match and build splitter."""
        kwargs = {k: v for k, v in kwargs.items() if v is not None}  # 筛选非空参数
        return self.value[0](*args, **kwargs)  # 返回根据参数构建的分割器对象
class Knowledge(ABC):
    """Knowledge Base Class."""

    def __init__(
        self,
        path: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        loader: Optional[Any] = None,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with Knowledge arguments."""
        # 设置私有属性_path，用于存储知识库的路径
        self._path = path
        # 设置私有属性_type，用于存储知识库的类型
        self._type = knowledge_type
        # 设置私有属性_loader，用于存储数据加载器
        self._loader = loader
        # 设置私有属性_metadata，用于存储元数据
        self._metadata = metadata

    def load(self) -> List[Document]:
        """Load knowledge from data loader."""
        # 调用内部方法_load()加载知识数据到documents列表
        documents = self._load()
        # 调用内部方法_postprocess()对加载后的documents进行后处理
        return self._postprocess(documents)

    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract knowledge from text."""
        # 直接返回传入的documents列表，表示从文本中提取知识的方法暂时未实现
        return documents

    @classmethod
    @abstractmethod
    def type(cls) -> KnowledgeType:
        """Get knowledge type."""
        # 抽象方法，需要在子类中实现，用于获取知识库的类型

    @classmethod
    def document_type(cls) -> Any:
        """Get document type."""
        # 返回None，表示此类知识库中的文档类型未定义或不适用
        return None

    def _postprocess(self, docs: List[Document]) -> List[Document]:
        """Post process knowledge from data loader."""
        # 直接返回传入的docs列表，表示在加载后的知识进行后处理的方法暂时未实现
        return docs

    @property
    def file_path(self):
        """Get file path."""
        # 返回存储在_path属性中的文件路径
        return self._path

    @abstractmethod
    def _load(self) -> List[Document]:
        """Preprocess knowledge from data loader."""
        # 抽象方法，需要在子类中实现，用于从数据加载器预处理知识数据

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return supported chunk strategy."""
        # 返回支持的分块策略列表
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_PAGE,
            ChunkStrategy.CHUNK_BY_PARAGRAPH,
            ChunkStrategy.CHUNK_BY_MARKDOWN_HEADER,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
        ]

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy.

        Returns:
            ChunkStrategy: default chunk strategy
        """
        # 返回默认的分块策略，即按大小进行分块
        return ChunkStrategy.CHUNK_BY_SIZE
```