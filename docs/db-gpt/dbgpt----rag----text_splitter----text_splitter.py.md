# `.\DB-GPT-src\dbgpt\rag\text_splitter\text_splitter.py`

```py
"""Text splitter module for splitting text into chunks."""

import copy  # 导入用于深拷贝对象的模块
import logging  # 导入日志记录模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import Any, Callable, Dict, Iterable, List, Optional, TypedDict, Union, cast  # 导入类型提示相关的模块

from dbgpt.core import Chunk, Document  # 导入自定义模块中的 Chunk 和 Document 类
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource  # 导入自定义模块中的其他类和函数
from dbgpt.util.i18n_utils import _  # 导入国际化工具函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class TextSplitter(ABC):
    """Interface for splitting text into chunks.

    Refer to `Langchain Text Splitter <https://github.com/langchain-ai/langchain/blob/
    master/libs/langchain/langchain/text_splitter.py>`_
    """

    outgoing_edges = 1  # 类属性，表示出边的数量为1

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        filters=None,
        separator: str = "",
    ):
        """Create a new TextSplitter."""
        if filters is None:
            filters = []  # 如果没有指定过滤器，则初始化为空列表
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )  # 如果重叠大小大于分块大小，抛出数值错误异常
        self._chunk_size = chunk_size  # 设置分块大小
        self._chunk_overlap = chunk_overlap  # 设置分块重叠大小
        self._length_function = length_function  # 设置用于计算长度的函数
        self._filter = filters  # 设置文本过滤器
        self._separator = separator  # 设置文本分隔符

    @abstractmethod
    def split_text(self, text: str, **kwargs) -> List[str]:
        """Split text into multiple components."""
        # 抽象方法，需要在子类中实现，用于将文本分割为多个部分

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        separator: Optional[str] = None,
        **kwargs,
    ) -> List[Chunk]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)  # 处理元数据列表，如果未提供则使用空字典
        chunks = []  # 初始化文档块列表
        for i, text in enumerate(texts):
            for chunk in self.split_text(text, separator=separator, **kwargs):
                new_doc = Chunk(content=chunk, metadata=copy.deepcopy(_metadatas[i]))
                chunks.append(new_doc)  # 根据文本列表创建文档对象列表
        return chunks  # 返回文档块列表

    def split_documents(self, documents: Iterable[Document], **kwargs) -> List[Chunk]:
        """Split documents."""
        texts = []  # 初始化文本列表
        metadatas = []  # 初始化元数据列表
        for doc in documents:
            texts.append(doc.content)  # 将文档内容加入文本列表
            metadatas.append(doc.metadata)  # 将文档元数据加入元数据列表
        return self.create_documents(texts, metadatas, **kwargs)  # 调用创建文档方法，返回文档块列表

    def _join_docs(self, docs: List[str], separator: str, **kwargs) -> Optional[str]:
        """Join a list of document parts into a single document."""
        text = separator.join(docs)  # 使用指定分隔符连接文档部分
        text = text.strip()  # 去除文本首尾空白字符
        if text == "":
            return None  # 如果文本为空，则返回空
        else:
            return text  # 否则返回连接后的文本

    def _merge_splits(
        self,
        splits: Iterable[str | dict],
        separator: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,

        ) -> Optional[str]:
        """Merge splits into a single text block."""
        text = separator.join(splits) if separator else "".join(splits)  # 使用指定分隔符或空字符连接分割文本块
        return text.strip() if text else None  # 返回去除首尾空白字符的文本块，如果文本为空则返回空
    ) -> List[str]:
        # 现在我们希望将这些小片段组合成中等大小的块以发送给LLM。
        # 如果未指定块大小，则使用默认块大小
        if chunk_size is None:
            chunk_size = self._chunk_size
        # 如果未指定块重叠大小，则使用默认块重叠大小
        if chunk_overlap is None:
            chunk_overlap = self._chunk_overlap
        # 如果未指定分隔符，则使用默认分隔符
        if separator is None:
            separator = self._separator
        # 计算分隔符的长度
        separator_len = self._length_function(separator)

        # 初始化文档列表和当前文档
        docs = []
        current_doc: List[str] = []
        total = 0

        # 遍历分割后的文本片段
        for s in splits:
            # 强制将s转换为字符串类型
            d = cast(str, s)
            # 计算当前文档片段的长度
            _len = self._length_function(d)
            
            # 检查是否需要开始一个新的块
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > chunk_size
            ):
                # 如果当前块大小超过了指定的块大小，则记录警告信息
                if total > chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {chunk_size}"
                    )
                
                # 如果当前文档不为空，则将其组合为一个完整的文档并添加到文档列表中
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    
                    # 继续弹出文档片段直到满足以下条件之一：
                    # - 当前块大小大于块重叠大小
                    # - 或者仍然有任何块并且长度过长
                    while total > chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            # 将当前文档片段添加到当前文档中，并更新当前块的总长度
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)

        # 将最后一个文档片段组合为完整的文档并添加到文档列表中
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)

        # 返回最终的文档列表
        return docs

    def clean(self, documents: List[dict], filters: List[str]):
        """清理文档中的特殊字符。"""
        # 遍历每个特殊字符，并在所有文档的内容中替换特殊字符为空字符串
        for special_character in filters:
            for doc in documents:
                doc["content"] = doc["content"].replace(special_character, "")
        # 返回清理后的文档列表
        return documents

    def run(  # type: ignore
        self,
        documents: Union[dict, List[dict]],
        meta: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
        separator: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        filters: Optional[List[str]] = None,
    ):
        """Run the text splitter."""
        # 如果未提供分隔符，则使用默认分隔符
        if separator is None:
            separator = self._separator
        # 如果未提供分块大小，则使用默认分块大小
        if chunk_size is None:
            chunk_size = self._chunk_size
        # 如果未提供重叠大小，则使用默认重叠大小
        if chunk_overlap is None:
            chunk_overlap = self._chunk_overlap
        # 如果未提供过滤器，则使用默认过滤器
        if filters is None:
            filters = self._filter
        # 初始化返回结果列表
        ret = []
        # 如果输入是一个字典（单个文档）
        if type(documents) == dict:
            # 对文档内容进行分割
            text_splits = self.split_text(
                documents["content"],
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            # 遍历分割后的文本片段
            for i, txt in enumerate(text_splits):
                # 深拷贝输入文档
                doc = copy.deepcopy(documents)
                # 更新文档的内容为当前文本片段
                doc["content"] = txt

                # 如果文档中没有 meta 字段或者 meta 字段为 None，则初始化为空字典
                if "meta" not in doc.keys() or doc["meta"] is None:
                    doc["meta"] = {}

                # 将分割的序号存入 meta 中
                doc["meta"]["_split_id"] = i
                # 将更新后的文档添加到结果列表中
                ret.append(doc)

        # 如果输入是一个列表（多个文档）
        elif type(documents) == list:
            # 遍历每个文档
            for document in documents:
                # 对每个文档的内容进行分割
                text_splits = self.split_text(
                    document["content"],
                    separator=separator,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                # 遍历分割后的文本片段
                for i, txt in enumerate(text_splits):
                    # 深拷贝当前文档
                    doc = copy.deepcopy(document)
                    # 更新文档的内容为当前文本片段
                    doc["content"] = txt

                    # 如果文档中没有 meta 字段或者 meta 字段为 None，则初始化为空字典
                    if "meta" not in doc.keys() or doc["meta"] is None:
                        doc["meta"] = {}

                    # 将分割的序号存入 meta 中
                    doc["meta"]["_split_id"] = i
                    # 将更新后的文档添加到结果列表中
                    ret.append(doc)

        # 如果存在过滤器并且过滤器列表不为空，则对结果进行清理操作
        if filters is not None and len(filters) > 0:
            ret = self.clean(ret, filters)
        
        # 构造输出结果字典
        result = {"documents": ret}
        # 返回结果字典和固定的输出标识符
        return result, "output_1"
@register_resource(
    _("Character Text Splitter"),  # 注册一个资源，名称为"Character Text Splitter"
    "character_text_splitter",  # 资源标识符为"character_text_splitter"
    category=ResourceCategory.RAG,  # 资源类别为RAG
    parameters=[  # 定义资源的参数列表
        Parameter.build_from(
            _("Separator"),  # 参数名为"Separator"
            "separator",  # 参数标识符为"separator"
            str,  # 参数类型为字符串
            description=_("Separator to split the text."),  # 参数描述为"用于分割文本的分隔符"
            optional=True,  # 参数是可选的
            default="\n\n",  # 参数的默认值为两个换行符
        ),
    ],
    description="Split text by characters.",  # 资源的描述为"按字符分割文本"
)
class CharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters.

    Refer to `Langchain Test Splitter <https://github.com/langchain-ai/langchain/blob/
    master/libs/langchain/langchain/text_splitter.py>`_
    """

    def __init__(self, separator: str = "\n\n", filters=None, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        if filters is None:
            filters = []  # 如果未提供过滤器，则设置为空列表
        self._separator = separator  # 设置实例变量_separator为传入的分隔符
        self._filter = filters  # 设置实例变量_filter为传入的过滤器列表

    def split_text(
        self, text: str, separator: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        if separator is None:
            separator = self._separator  # 如果未指定分隔符，则使用实例变量中的分隔符
        if separator:
            splits = text.split(separator)  # 使用指定的分隔符分割文本
        else:
            splits = list(text)  # 如果分隔符为空，则将文本转换为字符列表
        return self._merge_splits(splits, separator, **kwargs)  # 调用内部方法_merge_splits处理分割后的结果



@register_resource(
    _("Recursive Character Text Splitter"),  # 注册一个资源，名称为"Recursive Character Text Splitter"
    "recursive_character_text_splitter",  # 资源标识符为"recursive_character_text_splitter"
    category=ResourceCategory.RAG,  # 资源类别为RAG
    parameters=[
        # TODO: Support list of separators
        # Parameter.build_from(
        #     "Separators",
        #     "separators",
        #     List[str],
        #     description="List of separators to split the text.",
        #     optional=True,
        #     default=["###", "\n", " ", ""],
        # ),
    ],
    description=_("Split text by characters recursively."),  # 资源描述为"递归按字符分割文本"
)
class RecursiveCharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters.

    Recursively tries to split by different characters to find one
    that works.

    Refer to `Langchain Test Splitter <https://github.com/langchain-ai/langchain/blob/
    master/libs/langchain/langchain/text_splitter.py>`_
    """

    def __init__(self, separators: Optional[List[str]] = None, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self._separators = separators or ["###", "\n", " ", ""]  # 设置实例变量_separators为传入的分隔符列表或默认列表

    def split_text(
        self, text: str, separator: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Override split_text method to split text recursively."""
        # Implement splitting logic here (not fully provided in the given code snippet)
        pass  # 此处需要实现递归分割文本的逻辑，但在提供的代码片段中未完全给出
    ) -> List[str]:
        """Split incoming text and return chunks."""
        # 初始化最终结果列表
        final_chunks = []
        # 获取要使用的分隔符
        separator = self._separators[-1]
        for _s in self._separators:
            # 如果遇到空分隔符，直接使用空分隔符并退出循环
            if _s == "":
                separator = _s
                break
            # 如果文本中包含当前分隔符，选择该分隔符并退出循环
            if _s in text:
                separator = _s
                break
        # 根据选择的分隔符拆分文本
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # 递归地合并长文本，以便分割成合适大小的块
        _good_splits = []
        for s in splits:
            # 如果当前分割的文本长度小于设定的块大小，加入到有效分割列表中
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                # 如果有积累的有效分割文本，进行合并处理
                if _good_splits:
                    merged_text = self._merge_splits(
                        _good_splits,
                        separator,
                        chunk_size=kwargs.get("chunk_size", None),
                        chunk_overlap=kwargs.get("chunk_overlap", None),
                    )
                    final_chunks.extend(merged_text)
                    _good_splits = []
                # 对于超长文本，递归进行文本分割处理
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        # 处理剩余的有效分割文本
        if _good_splits:
            merged_text = self._merge_splits(
                _good_splits,
                separator,
                chunk_size=kwargs.get("chunk_size", None),
                chunk_overlap=kwargs.get("chunk_overlap", None),
            )
            final_chunks.extend(merged_text)
        # 返回最终的文本块列表
        return final_chunks
@register_resource(
    # 注册资源，名称为 "Spacy Text Splitter"
    _("Spacy Text Splitter"),
    # 标识为 "spacy_text_splitter"
    "spacy_text_splitter",
    # 资源类别为 RAG
    category=ResourceCategory.RAG,
    # 参数列表包括一个 pipeline 参数
    parameters=[
        Parameter.build_from(
            # 参数名为 "Pipeline"
            _("Pipeline"),
            # 参数标识为 "pipeline"
            "pipeline",
            # 参数类型为 str
            str,
            # 参数描述为 "Spacy pipeline to use for tokenization."
            description=_("Spacy pipeline to use for tokenization."),
            # 参数为可选项，默认值为 "zh_core_web_sm"
            optional=True,
            default="zh_core_web_sm",
        ),
    ],
    # 描述为 "Split text by sentences using Spacy."
    description=_("Split text by sentences using Spacy."),
)
class SpacyTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at sentences using Spacy.

    Refer to `Langchain Test Splitter <https://github.com/langchain-ai/langchain/blob/
    master/libs/langchain/langchain/text_splitter.py>`_
    """

    def __init__(self, pipeline: str = "zh_core_web_sm", **kwargs: Any) -> None:
        """Initialize the spacy text splitter."""
        # 调用父类初始化方法
        super().__init__(**kwargs)
        try:
            import spacy
        except ImportError:
            # 如果导入失败，抛出 ImportError 异常
            raise ImportError(
                "Spacy is not installed, please install it with `pip install spacy`."
            )
        try:
            # 尝试加载指定的 Spacy pipeline
            self._tokenizer = spacy.load(pipeline)
        except Exception:
            # 如果加载失败，尝试下载指定的 pipeline
            spacy.cli.download(pipeline)
            # 然后加载该 pipeline
            self._tokenizer = spacy.load(pipeline)

    def split_text(
        self, text: str, separator: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Split incoming text and return chunks."""
        # 如果文本长度超过 1000000，设置 tokenizer 的最大长度
        if len(text) > 1000000:
            self._tokenizer.max_length = len(text) + 100
        # 使用 tokenizer 对文本进行分句，生成器表达式
        splits = (str(s) for s in self._tokenizer(text).sents)
        # 调用 _merge_splits 方法合并分割结果
        return self._merge_splits(splits, separator, **kwargs)


class HeaderType(TypedDict):
    """Header type as typed dict."""

    level: int  # 字典包含 level 键，值为 int 类型
    name: str   # 字典包含 name 键，值为 str 类型
    data: str   # 字典包含 data 键，值为 str 类型


class LineType(TypedDict):
    """Line type as typed dict."""

    metadata: Dict[str, str]  # 字典包含 metadata 键，值为 Dict[str, str] 类型
    content: str              # 字典包含 content 键，值为 str 类型


@register_resource(
    # 注册资源，名称为 "Markdown Header Text Splitter"
    _("Markdown Header Text Splitter"),
    # 标识为 "markdown_header_text_splitter"
    "markdown_header_text_splitter",
    # 资源类别为 RAG
    category=ResourceCategory.RAG,
    # 参数列表包括四个参数
    parameters=[
        Parameter.build_from(
            # 参数名为 "Return Each Line"
            _("Return Each Line"),
            # 参数标识为 "return_each_line"
            "return_each_line",
            # 参数类型为 bool
            bool,
            # 参数描述为 "Return each line with associated headers."
            description=_("Return each line with associated headers."),
            # 参数为可选项，默认值为 False
            optional=True,
            default=False,
        ),
        Parameter.build_from(
            # 参数名为 "Chunk Size"
            _("Chunk Size"),
            # 参数标识为 "chunk_size"
            "chunk_size",
            # 参数类型为 int
            int,
            # 参数描述为 "Size of each chunk."
            description=_("Size of each chunk."),
            # 参数为可选项，默认值为 4000
            optional=True,
            default=4000,
        ),
        Parameter.build_from(
            # 参数名为 "Chunk Overlap"
            _("Chunk Overlap"),
            # 参数标识为 "chunk_overlap"
            "chunk_overlap",
            # 参数类型为 int
            int,
            # 参数描述为 "Overlap between chunks."
            description=_("Overlap between chunks."),
            # 参数为可选项，默认值为 200
            optional=True,
            default=200,
        ),
        Parameter.build_from(
            # 参数名为 "Separator"
            _("Separator"),
            # 参数标识为 "separator"
            "separator",
            # 参数类型为 str
            str,
            # 参数描述为 "Separator to split the text."
            description=_("Separator to split the text."),
            # 参数为可选项，默认值为 "\n"
            optional=True,
            default="\n",
        ),
    ],
    # 描述为 "Split markdown text by headers."
    description=_("Split markdown text by headers."),
)
class MarkdownHeaderTextSplitter(TextSplitter):
    """Implementation of splitting markdown files based on specified headers.

    Refer to `Langchain Text Splitter <https://github.com/langchain-ai/langchain/blob/
    master/libs/langchain/langchain/text_splitter.py>`_
    """

    outgoing_edges = 1  # 定义类变量 outgoing_edges 为 1，表示类的出边数量为 1

    def __init__(
        self,
        headers_to_split_on=None,
        return_each_line: bool = False,
        filters=None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        separator="\n",
    ):
        """Create a new MarkdownHeaderTextSplitter.

        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
        """
        # 根据传入的参数设置类的初始状态
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header1"),
                ("##", "Header2"),
                ("###", "Header3"),
                ("####", "Header4"),
                ("#####", "Header5"),
                ("######", "Header6"),
            ]
        if filters is None:
            filters = []
        self.return_each_line = return_each_line  # 设置是否每行返回及相关头部信息的标志
        self._chunk_size = chunk_size  # 设置文本分块的大小
        # 根据传入的头部信息进行排序，按长度降序排列
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
        self._filter = filters  # 设置文本过滤器
        self._length_function = length_function  # 设置文本长度计算函数
        self._separator = separator  # 设置文本分隔符
        self._chunk_overlap = chunk_overlap  # 设置文本分块的重叠部分大小

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        separator: Optional[str] = None,
        **kwargs,
    ) -> List[Chunk]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)  # 如果没有元数据，则创建空字典
        chunks = []  # 初始化文档块列表
        for i, text in enumerate(texts):
            # 对每个文本进行分割，生成文档块，并添加到列表中
            for chunk in self.split_text(text, separator, **kwargs):
                metadata = chunk.metadata or {}  # 获取块的元数据，如果不存在则为空字典
                metadata.update(_metadatas[i])  # 更新块的元数据
                new_doc = Chunk(content=chunk.content, metadata=metadata)  # 创建新的文档块
                chunks.append(new_doc)  # 将新文档块添加到列表中
        return chunks  # 返回文档块列表
    def aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[Chunk]:
        """Aggregate lines into chunks based on common metadata.

        Args:
            lines: List of dictionaries representing lines of text with associated metadata
        """
        aggregated_chunks: List[LineType] = []  # Initialize an empty list to store aggregated lines with metadata

        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == line["metadata"]
            ):
                # Check if the last aggregated line has the same metadata as the current line
                # If true, concatenate the current line's content to the last line's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            else:
                # If metadata is different or it's the first line, append the current line to aggregated_chunks
                subtitles = "-".join((list(line["metadata"].values())))  # Create subtitles from metadata values
                line["content"] = f'"{subtitles}": ' + line["content"]  # Format content with subtitles
                aggregated_chunks.append(line)

        return [
            Chunk(content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]  # Return a list of Chunk objects from aggregated_chunks

    def split_text(  # type: ignore
        self,
        text: str,
        separator: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs,
    ):
        """Split text into chunks."""
        # Function definition for splitting text, parameters include optional separator, chunk size, and overlap

    def clean(self, documents: List[dict], filters: Optional[List[str]] = None):
        """Clean documents by removing specified characters."""
        if filters is None:
            filters = self._filter  # Use default filters if none provided
        for special_character in filters:
            for doc in documents:
                doc["content"] = doc["content"].replace(special_character, "")  # Remove special characters from content
        return documents  # Return cleaned documents

    def _join_docs(self, docs: List[str], separator: str, **kwargs) -> Optional[str]:
        """Join documents into a single text with specified separator."""
        text = separator.join(docs)  # Join documents with separator
        text = text.strip()  # Strip leading and trailing whitespace
        if text == "":
            return None  # Return None if resulting text is empty
        else:
            return text  # Otherwise, return the joined text

    def _merge_splits(
        self,
        documents: Iterable[str | dict],
        separator: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Merge split documents."""
        # Function definition for merging split documents, with optional separator, chunk size, and overlap
    ) -> List[str]:
        # 现在我们希望将这些较小的片段合并成中等大小的块，发送到LLM。
        # 如果未指定块大小，则使用默认的块大小。
        if chunk_size is None:
            chunk_size = self._chunk_size
        # 如果未指定重叠部分的大小，则使用默认的重叠部分大小。
        if chunk_overlap is None:
            chunk_overlap = self._chunk_overlap
        # 如果未指定分隔符，则使用默认的分隔符。
        if separator is None:
            separator = self._separator
        # 计算分隔符的长度。
        separator_len = self._length_function(separator)

        # 初始化空的文档列表
        docs = []
        # 初始化当前文档为一个空列表
        current_doc: List[str] = []
        # 初始化总长度为0
        total = 0

        # 遍历输入的文档列表
        for _doc in documents:
            # 将_doc强制类型转换为字典类型
            dict_doc = cast(dict, _doc)
            # 如果文档的元数据不为空
            if dict_doc["metadata"] != {}:
                # 获取元数据中最大的值作为头部
                head = sorted(
                    dict_doc["metadata"].items(), key=lambda x: x[0], reverse=True
                )[0][1]
                # 构建包含头部和页面内容的字符串
                d = head + separator + dict_doc["page_content"]
            else:
                # 否则，只使用页面内容作为字符串
                d = dict_doc["page_content"]
            
            # 计算当前文档d的长度
            _len = self._length_function(d)
            
            # 如果当前文档加上新文档长度和可能的分隔符长度超过了块大小
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > chunk_size
            ):
                # 如果总长度已经超过了块大小，则记录警告信息
                if total > chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {chunk_size}"
                    )
                
                # 如果当前文档列表不为空，则将其连接成一个文档，并添加到文档列表中
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    
                    # 如果当前总长度超过了重叠部分的大小，或者当前文档列表仍然有内容且长度超过了块大小，则继续弹出文档
                    while total > chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            # 将当前文档d添加到当前文档列表中，并更新总长度
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        
        # 将最后一个文档列表连接成一个文档，并添加到文档列表中
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        
        # 返回最终的文档列表
        return docs
    ):
        """Run the text splitter."""
        # 如果未指定过滤器，则使用对象内部的默认过滤器
        if filters is None:
            filters = self._filter
        # 如果未指定块大小，则使用对象内部的默认块大小
        if chunk_size is None:
            chunk_size = self._chunk_size
        # 如果未指定块重叠量，则使用对象内部的默认块重叠量
        if chunk_overlap is None:
            chunk_overlap = self._chunk_overlap
        # 如果未指定分隔符，则使用对象内部的默认分隔符
        if separator is None:
            separator = self._separator
        # 初始化返回结果列表
        ret = []
        # 如果输入的文档是一个列表
        if type(documents) == list:
            # 遍历列表中的每个文档
            for document in documents:
                # 调用split_text方法，将文档内容按照指定的分隔符和块大小、块重叠量进行分割
                text_splits = self.split_text(
                    document["content"], separator, chunk_size, chunk_overlap
                )
                # 遍历分割后的文本片段
                for i, txt in enumerate(text_splits):
                    # 构建一个新的文档字典
                    doc = {"content": txt}
                    # 如果文档中不存在meta键或者meta为None，则创建一个空的meta字典
                    if "meta" not in doc.keys() or doc["meta"] is None:
                        doc["meta"] = {}  # type: ignore
                    # 设置meta中的_split_id属性为当前分割片段的索引
                    doc["meta"]["_split_id"] = i
                    # 将构建好的文档字典添加到结果列表中
                    ret.append(doc)
        # 如果输入的文档是一个字典
        elif type(documents) == dict:
            # 调用split_text方法，将字典中内容按照指定的分隔符和块大小、块重叠量进行分割
            text_splits = self.split_text(
                documents["content"], separator, chunk_size, chunk_overlap
            )
            # 遍历分割后的文本片段
            for i, txt in enumerate(text_splits):
                # 构建一个新的文档字典
                doc = {"content": txt}
                # 如果文档中不存在meta键或者meta为None，则创建一个空的meta字典
                if "meta" not in doc.keys() or doc["meta"] is None:
                    doc["meta"] = {}  # type: ignore
                # 设置meta中的_split_id属性为当前分割片段的索引
                doc["meta"]["_split_id"] = i
                # 将构建好的文档字典添加到结果列表中
                ret.append(doc)
        # 如果未指定过滤器，则再次使用对象内部的默认过滤器（这里的代码逻辑似乎重复了）
        if filters is None:
            filters = self._filter
        # 如果存在过滤器并且过滤器列表不为空，则调用clean方法清洗结果列表
        if filters is not None and len(filters) > 0:
            ret = self.clean(ret, filters)
        # 构建最终的输出结果字典，包含整理好的文档列表
        result = {"documents": ret}
        # 返回结果字典和一个字符串标识
        return result, "output_1"
class ParagraphTextSplitter(CharacterTextSplitter):
    """Implementation of splitting text that looks at paragraphs."""

    def __init__(
        self,
        separator="\n",
        chunk_size: int = 0,
        chunk_overlap: int = 0,
    ):
        """Create a new ParagraphTextSplitter."""
        # 初始化段落文本分割器
        self._separator = separator
        # 如果分隔符为None，则默认为换行符"\n"
        if self._separator is None:
            self._separator = "\n"
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # _is_paragraph被误用，实际意图是使用chunk_overlap
        self._is_paragraph = chunk_overlap

    def split_text(
        self, text: str, separator: Optional[str] = "\n", **kwargs
    ) -> List[str]:
        """Split incoming text and return chunks."""
        # 去除首尾空白后，按分隔符分割文本形成段落列表
        paragraphs = text.strip().split(self._separator)
        # 去除每个段落的首尾空白，并排除空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip() != ""]
        return paragraphs


@register_resource(
    _("Separator Text Splitter"),
    "separator_text_splitter",
    category=ResourceCategory.RAG,
    parameters=[
        Parameter.build_from(
            _("Separator"),
            "separator",
            str,
            description=_("Separator to split the text."),
            optional=True,
            default="\\n",
        ),
    ],
    description=_("Split text by separator."),
)
class SeparatorTextSplitter(CharacterTextSplitter):
    """The SeparatorTextSplitter class."""

    def __init__(self, separator: str = "\n", filters=None, **kwargs: Any):
        """Create a new TextSplitter."""
        # 初始化分隔符文本分割器
        if filters is None:
            filters = []
        self._merge = kwargs.pop("enable_merge") or False
        super().__init__(**kwargs)
        self._separator = separator
        self._filter = filters

    def split_text(
        self, text: str, separator: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Split incoming text and return chunks."""
        # 如果未指定分隔符，则使用默认分隔符self._separator
        if separator is None:
            separator = self._separator
        # 根据分隔符分割文本
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # 如果启用了合并选项，返回合并后的分割结果
        if self._merge:
            return self._merge_splits(splits, separator, chunk_overlap=0, **kwargs)
        # 过滤掉空字符串后返回分割结果列表
        return list(filter(None, text.split(separator)))


@register_resource(
    _("Page Text Splitter"),
    "page_text_splitter",
    category=ResourceCategory.RAG,
    parameters=[
        Parameter.build_from(
            _("Separator"),
            "separator",
            str,
            description=_("Separator to split the text."),
            optional=True,
            default="\n\n",
        ),
    ],
    description=_("Split text by page."),
)
class PageTextSplitter(TextSplitter):
    """The PageTextSplitter class."""

    def __init__(self, separator: str = "\n\n", filters=None, **kwargs: Any):
        """Create a new TextSplitter."""
        # 初始化页面文本分割器
        super().__init__(**kwargs)
        if filters is None:
            filters = []
        self._separator = separator
        self._filter = filters
    # 文本分割函数，将输入的文本按指定分隔符分割成列表
    def split_text(
        self, text: str, separator: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Split incoming text and return chunks."""
        # 目前仅返回包含整个文本的列表，实际上未执行分割操作
        return [text]

    # 创建文档函数，从文本列表创建文档对象列表
    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        separator: Optional[str] = None,
        **kwargs,
    ) -> List[Chunk]:
        """Create documents from a list of texts."""
        # 如果未提供元数据列表，则初始化一个空的元数据列表，长度与文本列表相同
        _metadatas = metadatas or [{}] * len(texts)
        # 初始化空的文档对象列表
        chunks = []
        # 遍历文本列表
        for i, text in enumerate(texts):
            # 使用文本和对应的元数据创建一个新的文档对象，并深度复制元数据以避免引用问题
            new_doc = Chunk(content=text, metadata=copy.deepcopy(_metadatas[i]))
            # 将新创建的文档对象添加到文档对象列表中
            chunks.append(new_doc)
        # 返回创建好的文档对象列表
        return chunks
```