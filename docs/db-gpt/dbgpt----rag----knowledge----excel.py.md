# `.\DB-GPT-src\dbgpt\rag\knowledge\excel.py`

```py
"""Excel Knowledge."""
# 导入必要的模块和类
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from dbgpt.core import Document
from dbgpt.rag.knowledge.base import (
    ChunkStrategy,
    DocumentType,
    Knowledge,
    KnowledgeType,
)

# 定义 ExcelKnowledge 类，继承自 Knowledge 类
class ExcelKnowledge(Knowledge):
    """Excel Knowledge."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = KnowledgeType.DOCUMENT,
        source_column: Optional[str] = None,
        encoding: Optional[str] = "utf-8",
        loader: Optional[Any] = None,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create xlsx Knowledge with Knowledge arguments.

        Args:
            file_path(str,  optional): file path
            knowledge_type(KnowledgeType, optional): knowledge type
            source_column(str, optional): source column
            encoding(str, optional): csv encoding
            loader(Any, optional): loader
        """
        # 调用父类 Knowledge 的初始化方法
        super().__init__(
            path=file_path,
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        # 设置编码属性
        self._encoding = encoding
        # 设置源列属性
        self._source_column = source_column
    # 加载文档的方法，返回一个文档对象列表
    def _load(self) -> List[Document]:
        """Load csv document from loader."""
        # 如果存在加载器对象，则使用加载器加载文档
        if self._loader:
            documents = self._loader.load()
        else:
            # 否则手动加载文档
            docs = []
            # 如果未指定文件路径，则抛出数值错误
            if not self._path:
                raise ValueError("file path is required")

            # 使用 pandas 打开 Excel 文件
            excel_file = pd.ExcelFile(self._path)
            # 获取 Excel 文件中所有工作表的名称
            sheet_names = excel_file.sheet_names
            # 遍历每个工作表名
            for sheet_name in sheet_names:
                # 解析工作表数据为 DataFrame
                df = excel_file.parse(sheet_name)
                # 遍历 DataFrame 的每一行
                for index, row in df.iterrows():
                    # 初始化空列表用于存储每一行的字符串表示
                    strs = []
                    # 遍历每一列的列名和列值
                    for column_name, column_value in row.items():
                        # 如果列名或列值为空，则跳过
                        if column_name is None or column_value is None:
                            continue

                        # 将列名和列值转换为字符串，并添加到 strs 列表中
                        column_name = str(column_name)
                        column_value = str(column_value)
                        strs.append(f"{column_name.strip()}: {column_value.strip()}")

                    # 将 strs 列表中的字符串用换行符连接成文本内容
                    content = "\n".join(strs)
                    # 尝试获取源数据，如果指定了源列，则使用该列的值作为源，否则使用文件路径作为源
                    try:
                        source = (
                            row[self._source_column]
                            if self._source_column is not None
                            else self._path
                        )
                    except KeyError:
                        # 如果指定的源列不存在，则抛出数值错误
                        raise ValueError(
                            f"Source column '{self._source_column}' not in CSV "
                            f"file."
                        )

                    # 创建文档的元数据字典，包括源和行号
                    metadata = {"source": source, "row": index}
                    # 如果存在其他元数据，则更新到元数据字典中
                    if self._metadata:
                        metadata.update(self._metadata)  # type: ignore
                    # 创建 Document 对象，并添加到 docs 列表中
                    doc = Document(content=content, metadata=metadata)
                    docs.append(doc)

            # 返回加载的所有文档列表
            return docs

        # 如果使用加载器加载了文档，则将加载的文档转换为文档对象列表并返回
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
    def type(cls) -> KnowledgeType:
        """Knowledge type of CSV."""
        # 返回 CSV 文件对应的知识类型 DOCUMENT
        return KnowledgeType.DOCUMENT

    @classmethod
    def document_type(cls) -> DocumentType:
        """Return document type."""
        # 返回文档类型 EXCEL
        return DocumentType.EXCEL
```