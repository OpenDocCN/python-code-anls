# `MetaGPT\metagpt\document.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/8 14:03
@Author  : alexanderwu
@File    : document.py
@Desc    : Classes and Operations Related to Files in the File System.
"""

# 导入所需的模块和库
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from langchain.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
from metagpt.repo_parser import RepoParser

# 验证数据框中是否存在指定的列
def validate_cols(content_col: str, df: pd.DataFrame):
    if content_col not in df.columns:
        raise ValueError("Content column not found in DataFrame.")

# 读取数据
def read_data(data_path: Path):
    suffix = data_path.suffix
    if ".xlsx" == suffix:
        data = pd.read_excel(data_path)
    elif ".csv" == suffix:
        data = pd.read_csv(data_path)
    elif ".json" == suffix:
        data = pd.read_json(data_path)
    elif suffix in (".docx", ".doc"):
        data = UnstructuredWordDocumentLoader(str(data_path), mode="elements").load()
    elif ".txt" == suffix:
        data = TextLoader(str(data_path)).load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=256, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        data = texts
    elif ".pdf" == suffix:
        data = UnstructuredPDFLoader(str(data_path), mode="elements").load()
    else:
        raise NotImplementedError("File format not supported.")
    return data

# 定义文档状态枚举类
class DocumentStatus(Enum):
    """Indicates document status, a mechanism similar to RFC/PEP"""
    DRAFT = "draft"
    UNDERREVIEW = "underreview"
    APPROVED = "approved"
    DONE = "done"

# 定义文档类
class Document(BaseModel):
    """
    Document: Handles operations related to document files.
    """
    path: Path = Field(default=None)
    name: str = Field(default="")
    content: str = Field(default="")
    author: str = Field(default="")
    status: DocumentStatus = Field(default=DocumentStatus.DRAFT)
    reviews: list = Field(default_factory=list)

    # 从文件路径创建文档实例
    @classmethod
    def from_path(cls, path: Path):
        """
        Create a Document instance from a file path.
        """
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found.")
        content = path.read_text()
        return cls(content=content, path=path)

    # 从文本创建文档实例
    @classmethod
    def from_text(cls, text: str, path: Optional[Path] = None):
        """
        Create a Document from a text string.
        """
        return cls(content=text, path=path)

    # 将内容保存到指定的文件路径
    def to_path(self, path: Optional[Path] = None):
        """
        Save content to the specified file path.
        """
        if path is not None:
            self.path = path
        if self.path is None:
            raise ValueError("File path is not set.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.content, encoding="utf-8")

    # 持久化文档到磁盘
    def persist(self):
        """
        Persist document to disk.
        """
        return self.to_path()

# 可索引文档类
class IndexableDocument(Document):
    """
    Advanced document handling: For vector databases or search engines.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: Union[pd.DataFrame, list]
    content_col: Optional[str] = Field(default="")
    meta_col: Optional[str] = Field(default="")

    # 从文件路径创建可索引文档实例
    @classmethod
    def from_path(cls, data_path: Path, content_col="content", meta_col="metadata"):
        if not data_path.exists():
            raise FileNotFoundError(f"File {data_path} not found.")
        data = read_data(data_path)
        if isinstance(data, pd.DataFrame):
            validate_cols(content_col, data)
            return cls(data=data, content=str(data), content_col=content_col, meta_col=meta_col)
        else:
            content = data_path.read_text()
            return cls(data=data, content=content, content_col=content_col, meta_col=meta_col)

    # 从数据框中获取文档和元数据
    def _get_docs_and_metadatas_by_df(self) -> (list, list):
        # 省略部分代码
        pass

    # 从langchain获取文档和元数据
    def _get_docs_and_metadatas_by_langchain(self) -> (list, list):
        # 省略部分代码
        pass

    # 获取文档和元数据
    def get_docs_and_metadatas(self) -> (list, list):
        if isinstance(self.data, pd.DataFrame):
            return self._get_docs_and_metadatas_by_df()
        elif isinstance(self.data, list):
            return self._get_docs_and_metadatas_by_langchain()
        else:
            raise NotImplementedError("Data type not supported for metadata extraction.")

# 仓库元数据类
class RepoMetadata(BaseModel):
    name: str = Field(default="")
    n_docs: int = Field(default=0)
    n_chars: int = Field(default=0)
    symbols: list = Field(default_factory=list)

# 仓库类
class Repo(BaseModel):
    name: str = Field(default="")
    docs: dict[Path, Document] = Field(default_factory=dict)
    codes: dict[Path, Document] = Field(default_factory=dict)
    assets: dict[Path, Document] = Field(default_factory=dict)
    path: Path = Field(default=None)

    # 从文件路径创建仓库实例
    @classmethod
    def from_path(cls, path: Path):
        """Load documents, code, and assets from a repository path."""
        path.mkdir(parents=True, exist_ok=True)
        repo = Repo(path=path, name=path.name)
        for file_path in path.rglob("*"):
            # FIXME: These judgments are difficult to support multiple programming languages and need to be more general
            if file_path.is_file() and file_path.suffix in [".json", ".txt", ".md", ".py", ".js", ".css", ".html"]:
                repo._set(file_path.read_text(), file_path)
        return repo

    # 持久化所有文档、代码和资产到指定的仓库路径
    def to_path(self):
        """Persist all documents, code, and assets to the given repository path."""
        for doc in self.docs.values():
            doc.to_path()
        for code in self.codes.values():
            code.to_path()
        for asset in self.assets.values():
            asset.to_path()

    # 添加文档到适当的类别
    def _set(self, content: str, path: Path):
        """Add a document to the appropriate category based on its file extension."""
        # 省略部分代码
        pass

    # 设置文档并将其持久化到磁盘
    def set(self, filename: str, content: str):
        """Set a document and persist it to disk."""
        # 省略部分代码
        pass

    # 获取指定文件名的文档
    def get(self, filename: str) -> Optional[Document]:
        """Get a document by its filename."""
        # 省略部分代码
        pass

    # 获取文本文档
    def get_text_documents(self) -> list[Document]:
        return list(self.docs.values()) + list(self.codes.values())

    # 数据探索分析
    def eda(self) -> RepoMetadata:
        # 省略部分代码
        pass

```