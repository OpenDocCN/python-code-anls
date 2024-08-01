# `.\DB-GPT-src\dbgpt\core\interface\knowledge.py`

```py
"""Chunk document schema."""

# 导入必要的库
import json
import uuid
from typing import Any, Dict

# 导入基础模型和字段定义函数
from dbgpt._private.pydantic import BaseModel, Field, model_to_dict

# 定义 Document 类，包括文档内容和元数据
class Document(BaseModel):
    """Document including document content, document metadata."""

    content: str = Field(default="", description="document text content")

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="metadata fields",
    )

    def set_content(self, content: str) -> None:
        """Set document content."""
        self.content = content

    def get_content(self) -> str:
        """Get document content."""
        return self.content

    @classmethod
    def langchain2doc(cls, document):
        """Transform Langchain to Document format."""
        metadata = document.metadata or {}
        return cls(content=document.page_content, metadata=metadata)

    @classmethod
    def doc2langchain(cls, chunk):
        """Transform Document to Langchain format."""
        from langchain.schema import Document as LCDocument

        return LCDocument(page_content=chunk.content, metadata=chunk.metadata)

# 定义 Chunk 类，继承自 Document 类，包括块文档的内容、元数据、摘要、关系等
class Chunk(Document):
    """The chunk document schema.

    Document Chunk including chunk content, chunk metadata, chunk summary, chunk
    relations.
    """

    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="unique id for the chunk"
    )
    content: str = Field(default="", description="chunk text content")

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="metadata fields",
    )
    score: float = Field(default=0.0, description="chunk text similarity score")
    summary: str = Field(default="", description="chunk text summary")
    separator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Convert Chunk to dict."""
        data = model_to_dict(self, **kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        """Convert Chunk to json."""
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    def __hash__(self):
        """Hash function."""
        return hash((self.chunk_id,))

    def __eq__(self, other):
        """Equal function."""
        return self.chunk_id == other.chunk_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any):  # type: ignore
        """Create Chunk from dict."""
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any):  # type: ignore
        """Create Chunk from json."""
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)

    @classmethod
    # 将 Langchain 对象转换为 Chunk 格式
    def langchain2chunk(cls, document):
        """Transform Langchain to Chunk format."""
        metadata = document.metadata or {}  # 获取文档的元数据，如果不存在则使用空字典
        # 创建并返回一个 Chunk 对象，使用页面内容和获取的元数据
        return cls(content=document.page_content, metadata=metadata)

    @classmethod
    # 将 llamaindex 转换为 Chunk 格式
    def llamaindex2chunk(cls, node):
        """Transform llama-index to Chunk format."""
        metadata = node.metadata or {}  # 获取节点的元数据，如果不存在则使用空字典
        # 创建并返回一个 Chunk 对象，使用节点的内容和获取的元数据
        return cls(content=node.content, metadata=metadata)

    @classmethod
    # 将 Chunk 对象转换为 Langchain 格式
    def chunk2langchain(cls, chunk):
        """Transform Chunk to Langchain format."""
        try:
            from langchain.schema import Document as LCDocument  # 导入 Langchain 的 Document 类
        except ImportError:
            raise ValueError(
                "Could not import python package: langchain "
                "Please install langchain by command `pip install langchain"
            )
        # 使用 Langchain 的 Document 类创建一个新对象，使用 Chunk 的内容和元数据
        return LCDocument(page_content=chunk.content, metadata=chunk.metadata)

    @classmethod
    # 将 Chunk 对象转换为 llamaindex 格式
    def chunk2llamaindex(cls, chunk):
        """Transform Chunk to llama-index format."""
        try:
            from llama_index.schema import TextNode  # 导入 llama_index 的 TextNode 类
        except ImportError:
            raise ValueError(
                "Could not import python package: llama_index "
                "Please install llama_index by command `pip install llama_index"
            )
        # 使用 llama_index 的 TextNode 类创建一个新对象，使用 Chunk 的内容和元数据
        return TextNode(text=chunk.content, metadata=chunk.metadata)
```