# `.\DB-GPT-src\dbgpt\rag\chunk_manager.py`

```py
"""Module for ChunkManager."""

# 从 enum 模块导入 Enum 类型
from enum import Enum
# 从 typing 模块导入 Any, List, Optional 类型
from typing import Any, List, Optional

# 从 dbgpt._private.pydantic 模块导入 BaseModel, Field 类
from dbgpt._private.pydantic import BaseModel, Field
# 从 dbgpt.core 模块导入 Chunk, Document 类
from dbgpt.core import Chunk, Document
# 从 dbgpt.core.awel.flow 模块导入 Parameter, ResourceCategory, register_resource 函数
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
# 从 dbgpt.rag.extractor.base 模块导入 Extractor 类
from dbgpt.rag.extractor.base import Extractor
# 从 dbgpt.rag.knowledge.base 模块导入 ChunkStrategy, Knowledge 类
from dbgpt.rag.knowledge.base import ChunkStrategy, Knowledge
# 从 dbgpt.rag.text_splitter 模块导入 TextSplitter 类
from dbgpt.rag.text_splitter import TextSplitter
# 从 dbgpt.util.i18n_utils 模块导入 _ 函数
from dbgpt.util.i18n_utils import _


class SplitterType(str, Enum):
    """The type of splitter."""
    
    # 定义枚举类型 LANGCHAIN，值为 "langchain"
    LANGCHAIN = "langchain"
    # 定义枚举类型 LLAMA_INDEX，值为 "llama-index"
    LLAMA_INDEX = "llama-index"
    # 定义枚举类型 USER_DEFINE，值为 "user_define"
    USER_DEFINE = "user_define"


@register_resource(
    _("Chunk Parameters"),  # 资源名称，使用国际化函数 _
    "chunk_parameters",  # 资源标识符
    category=ResourceCategory.RAG,  # 资源类别为 RAG
    parameters=[  # 参数列表开始
        Parameter.build_from(
            _("Chunk Strategy"),  # 参数名称，使用国际化函数 _
            "chunk_strategy",  # 参数标识符
            str,  # 参数类型为字符串
            description=_("chunk strategy"),  # 参数描述，使用国际化函数 _
            optional=True,  # 参数可选
            default=None,  # 默认值为 None
        ),
        Parameter.build_from(
            _("Text Splitter"),  # 参数名称，使用国际化函数 _
            "text_splitter",  # 参数标识符
            TextSplitter,  # 参数类型为 TextSplitter 类型
            description=_(
                "Text splitter, if not set, will use the default text splitter."
            ),  # 参数描述，使用国际化函数 _
            optional=True,  # 参数可选
            default=None,  # 默认值为 None
        ),
        Parameter.build_from(
            _("Splitter Type"),  # 参数名称，使用国际化函数 _
            "splitter_type",  # 参数标识符
            str,  # 参数类型为字符串
            description=_("Splitter type"),  # 参数描述，使用国际化函数 _
            optional=True,  # 参数可选
            default=SplitterType.USER_DEFINE.value,  # 默认值为 SplitterType.USER_DEFINE 的值
        ),
        Parameter.build_from(
            _("Chunk Size"),  # 参数名称，使用国际化函数 _
            "chunk_size",  # 参数标识符
            int,  # 参数类型为整数
            description=_("Chunk size"),  # 参数描述，使用国际化函数 _
            optional=True,  # 参数可选
            default=512,  # 默认值为 512
        ),
        Parameter.build_from(
            _("Chunk Overlap"),  # 参数名称，使用国际化函数 _
            "chunk_overlap",  # 参数标识符
            int,  # 参数类型为整数
            description="Chunk overlap",  # 参数描述
            optional=True,  # 参数可选
            default=50,  # 默认值为 50
        ),
        Parameter.build_from(
            _("Separator"),  # 参数名称，使用国际化函数 _
            "separator",  # 参数标识符
            str,  # 参数类型为字符串
            description=_("Chunk separator"),  # 参数描述，使用国际化函数 _
            optional=True,  # 参数可选
            default="\n",  # 默认值为换行符
        ),
        Parameter.build_from(
            _("Enable Merge"),  # 参数名称，使用国际化函数 _
            "enable_merge",  # 参数标识符
            bool,  # 参数类型为布尔值
            description=_("Enable chunk merge by chunk_size."),  # 参数描述，使用国际化函数 _
            optional=True,  # 参数可选
            default=False,  # 默认值为 False
        ),
    ],  # 参数列表结束
)
class ChunkParameters(BaseModel):
    """The parameters for chunking."""
    
    chunk_strategy: str = Field(
        default=None,  # 默认值为 None
        description="chunk strategy",  # 字段描述
    )
    text_splitter: Optional[Any] = Field(
        default=None,  # 默认值为 None
        description="text splitter",  # 字段描述
    )

    splitter_type: SplitterType = Field(
        default=SplitterType.USER_DEFINE,  # 默认值为 SplitterType.USER_DEFINE
        description="splitter type",  # 字段描述
    )

    chunk_size: int = Field(
        default=512,  # 默认值为 512
        description="chunk size",  # 字段描述
    )
    chunk_overlap: int = Field(
        default=50,  # 默认值为 50
        description="chunk overlap",  # 字段描述
    )
    # 分隔符，用于定义块的分隔方式，默认为换行符
    separator: str = Field(
        default="\n",
        description="chunk separator",
    )
    # 是否启用根据块大小合并块的选项，默认为 None，需要根据具体情况指定
    enable_merge: bool = Field(
        default=None,
        description="enable chunk merge by chunk_size.",
    )
    class ChunkManager:
        """Manager for chunks."""

        def __init__(
            self,
            knowledge: Knowledge,
            chunk_parameter: Optional[ChunkParameters] = None,
            extractor: Optional[Extractor] = None,
        ):
            """Create a new ChunkManager with the given knowledge.

            Args:
                knowledge: (Knowledge) Knowledge datasource.
                chunk_parameter: (Optional[ChunkParameter]) Chunk parameter.
                extractor: (Optional[Extractor]) Extractor to use for summarization.
            """
            # 初始化 ChunkManager 实例，设置知识源
            self._knowledge = knowledge

            # 设置文本分割器和分块参数
            self._extractor = extractor
            self._chunk_parameters = chunk_parameter or ChunkParameters()

            # 设置分块策略，如果未提供则使用默认的知识源分块策略
            self._chunk_strategy = (
                chunk_parameter.chunk_strategy
                if chunk_parameter and chunk_parameter.chunk_strategy
                else self._knowledge.default_chunk_strategy().name
            )

            # 设置文本分割器和分割器类型
            self._text_splitter = self._chunk_parameters.text_splitter
            self._splitter_type = self._chunk_parameters.splitter_type

        def split(self, documents: List[Document]) -> List[Chunk]:
            """Split a document into chunks."""
            # 获取当前选择的文本分割器
            text_splitter = self._select_text_splitter()

            # 根据分割器类型进行文档分割和处理
            if SplitterType.LANGCHAIN == self._splitter_type:
                documents = text_splitter.split_documents(documents)
                return [Chunk.langchain2chunk(document) for document in documents]
            elif SplitterType.LLAMA_INDEX == self._splitter_type:
                nodes = text_splitter.split_documents(documents)
                return [Chunk.llamaindex2chunk(node) for node in nodes]
            else:
                return text_splitter.split_documents(documents)

        def split_with_summary(
            self, document: Any, chunk_strategy: ChunkStrategy
        ) -> List[Chunk]:
            """Split a document into chunks and summary."""
            # 此方法尚未实现，抛出未实现错误
            raise NotImplementedError

        def extract(self, chunks: List[Chunk]) -> None:
            """Extract metadata from chunks."""
            # 如果存在提取器，从分块中提取元数据
            if self._extractor:
                self._extractor.extract(chunks)

        @property
        def chunk_parameters(self) -> ChunkParameters:
            """Get chunk parameters."""
            # 返回分块参数对象
            return self._chunk_parameters

        def set_text_splitter(
            self,
            text_splitter: TextSplitter,
            splitter_type: SplitterType = SplitterType.LANGCHAIN,
        ) -> None:
            """Add text splitter."""
            # 设置文本分割器和分割器类型
            self._text_splitter = text_splitter
            self._splitter_type = splitter_type

        def get_text_splitter(
            self,
        ) -> TextSplitter:
            """Return text splitter."""
            # 返回当前选择的文本分割器
            return self._select_text_splitter()

        def _select_text_splitter(
            self,
        ) -> TextSplitter:
            """Select the appropriate text splitter."""
            # 内部方法：根据当前的分割器类型返回相应的文本分割器
            # 这里假设有一个根据条件选择文本分割器的逻辑，具体实现未给出
            pass
    ) -> TextSplitter:
        """根据块策略选择文本分割器。"""
        # 如果已经设置了文本分割器，则直接返回该文本分割器
        if self._text_splitter:
            return self._text_splitter
        # 如果没有设置块策略或者块策略为"Automatic"，则使用默认块策略名称
        if not self._chunk_strategy or self._chunk_strategy == "Automatic":
            self._chunk_strategy = self._knowledge.default_chunk_strategy().name
        # 检查当前块策略是否在支持的块策略列表中
        if self._chunk_strategy not in [
            support_chunk_strategy.name
            for support_chunk_strategy in self._knowledge.support_chunk_strategy()
        ]:
            # 获取当前知识类型值，如果存在文档类型，则使用文档类型的值
            current_type = self._knowledge.type().value
            if self._knowledge.document_type():
                current_type = self._knowledge.document_type().value
            # 抛出值错误，说明当前知识类型不支持所选的块策略
            raise ValueError(
                f"{current_type} knowledge not supported chunk strategy "
                f"{self._chunk_strategy} "
            )
        # 根据选择的块策略名称获取对应的枚举类型ChunkStrategy对象
        strategy = ChunkStrategy[self._chunk_strategy]
        # 调用选择的块策略的match方法，返回文本分割器
        return strategy.match(
            chunk_size=self._chunk_parameters.chunk_size,
            chunk_overlap=self._chunk_parameters.chunk_overlap,
            separator=self._chunk_parameters.separator,
            enable_merge=self._chunk_parameters.enable_merge,
        )
```