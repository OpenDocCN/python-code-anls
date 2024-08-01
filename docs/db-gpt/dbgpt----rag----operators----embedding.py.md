# `.\DB-GPT-src\dbgpt\rag\operators\embedding.py`

```py
"""Embedding retriever operator."""

# 导入reduce函数，用于在参数列表中将函数应用于序列的所有元素
from functools import reduce
# 导入类型提示模块中的相关类和类型
from typing import List, Optional, Union

# 导入核心功能模块
from dbgpt.core import Chunk
# 导入 AWEL 流模块中的特定类和函数
from dbgpt.core.awel.flow import IOField, OperatorCategory, Parameter, ViewMetadata
# 导入检索操作接口中的操作类
from dbgpt.core.interface.operators.retriever import RetrieverOperator
# 导入国际化文本函数
from dbgpt.util.i18n_utils import _

# 导入当前目录下的嵌入汇编器
from ..assembler.embedding import EmbeddingAssembler
# 导入当前目录下的块管理器参数
from ..chunk_manager import ChunkParameters
# 导入当前目录下的基础索引存储接口
from ..index.base import IndexStoreBase
# 导入当前目录下的知识模块
from ..knowledge import Knowledge
# 导入当前目录下的嵌入检索器
from ..retriever.embedding import EmbeddingRetriever
# 导入当前目录下的重新排序模块
from ..retriever.rerank import Ranker
# 导入当前目录下的查询重写模块
from ..retriever.rewrite import QueryRewrite
# 导入当前目录下的装配器操作
from .assembler import AssemblerOperator


class EmbeddingRetrieverOperator(RetrieverOperator[Union[str, List[str]], List[Chunk]]):
    """The Embedding Retriever Operator."""

    # 操作元数据，包括名称、描述、参数、输入和输出
    metadata = ViewMetadata(
        label=_("Embedding Retriever Operator"),
        name="embedding_retriever_operator",
        description=_("Retrieve candidates from vector store."),
        category=OperatorCategory.RAG,  # 操作类别为RAG
        parameters=[
            # 存储索引存储的参数
            Parameter.build_from(
                _("Storage Index Store"),
                "index_store",
                IndexStoreBase,
                description=_("The vector store connector."),
                alias=["vector_store_connector"],
            ),
            # Top K参数，指定候选项数量
            Parameter.build_from(
                _("Top K"),
                "top_k",
                int,
                description=_("The number of candidates."),
            ),
            # 分数阈值参数，低于此分数的候选项将被过滤
            Parameter.build_from(
                _("Score Threshold"),
                "score_threshold",
                float,
                description=_(
                    "The score threshold, if score of candidate is less than it, it "
                    "will be filtered."
                ),
                optional=True,
                default=0.3,  # 默认分数阈值为0.3
            ),
            # 查询重写资源参数
            Parameter.build_from(
                _("Query Rewrite"),
                "query_rewrite",
                QueryRewrite,
                description=_("The query rewrite resource."),
                optional=True,
                default=None,
            ),
            # 重新排序参数
            Parameter.build_from(
                _("Rerank"),
                "rerank",
                Ranker,
                description=_("The rerank."),
                optional=True,
                default=None,
            ),
        ],
        inputs=[
            # 输入字段：查询字符串
            IOField.build_from(
                _("Query"),
                "query",
                str,
                description=_("The query to retrieve."),
            )
        ],
        outputs=[
            # 输出字段：候选项列表
            IOField.build_from(
                _("Candidates"),
                "candidates",
                Chunk,
                description=_("The retrieved candidates."),
                is_list=True,
            )
        ],
    )
    def __init__(
        self,
        index_store: IndexStoreBase,
        top_k: int,
        score_threshold: float = 0.3,
        query_rewrite: Optional[QueryRewrite] = None,
        rerank: Optional[Ranker] = None,
        **kwargs
    ):
        """
        Create a new EmbeddingRetrieverOperator.

        Args:
            index_store (IndexStoreBase): An object that provides access to the index store.
            top_k (int): The maximum number of retrievals to return.
            score_threshold (float, optional): The minimum score threshold for retrieval results.
                                               Defaults to 0.3.
            query_rewrite (Optional[QueryRewrite], optional): An optional query rewriting component.
            rerank (Optional[Ranker], optional): An optional reranking component for retrieval results.
            **kwargs: Additional keyword arguments passed to the superclass constructor.
        """
        super().__init__(**kwargs)
        self._score_threshold = score_threshold
        self._retriever = EmbeddingRetriever(
            index_store=index_store,
            top_k=top_k,
            query_rewrite=query_rewrite,
            rerank=rerank,
        )

    def retrieve(self, query: Union[str, List[str]]) -> List[Chunk]:
        """
        Retrieve the candidates.

        Args:
            query (Union[str, List[str]]): A single query string or a list of query strings.

        Returns:
            List[Chunk]: A list of Chunk objects representing the retrieved candidates.
        """
        if isinstance(query, str):
            # Retrieve candidates for a single query string
            return self._retriever.retrieve_with_scores(query, self._score_threshold)
        elif isinstance(query, list):
            # Retrieve candidates for each query string in the list
            candidates = [
                self._retriever.retrieve_with_scores(q, self._score_threshold)
                for q in query
            ]
            # Reduce the list of candidate lists into a single list
            return reduce(lambda x, y: x + y, candidates)
class EmbeddingAssemblerOperator(AssemblerOperator[Knowledge, List[Chunk]]):
    """The Embedding Assembler Operator."""

    metadata = ViewMetadata(
        label=_("Embedding Assembler Operator"),
        name="embedding_assembler_operator",
        description=_("Load knowledge and assemble embedding chunks to vector store."),
        category=OperatorCategory.RAG,
        parameters=[
            Parameter.build_from(
                _("Vector Store Connector"),
                "index_store",
                IndexStoreBase,
                description=_("The vector store connector."),
                alias=["vector_store_connector"],
            ),
            Parameter.build_from(
                _("Chunk Parameters"),
                "chunk_parameters",
                ChunkParameters,
                description=_("The chunk parameters."),
                optional=True,
                default=None,
            ),
        ],
        inputs=[
            IOField.build_from(
                _("Knowledge"),
                "knowledge",
                Knowledge,
                description=_("The knowledge to be loaded."),
            )
        ],
        outputs=[
            IOField.build_from(
                _("Chunks"),
                "chunks",
                Chunk,
                description=_(
                    "The assembled chunks, it has been persisted to vector " "store."
                ),
                is_list=True,
            )
        ],
    )

    def __init__(
        self,
        index_store: IndexStoreBase,
        chunk_parameters: Optional[ChunkParameters] = None,
        **kwargs
    ):
        """Create a new EmbeddingAssemblerOperator.

        Args:
            index_store (IndexStoreBase): The index storage.
            chunk_parameters (Optional[ChunkParameters], optional): The chunk
                parameters. Defaults to ChunkParameters(chunk_strategy="CHUNK_BY_SIZE").
        """
        # 如果未提供 chunk_parameters，则使用默认的 ChunkParameters 对象
        if not chunk_parameters:
            chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")
        # 初始化私有属性 _chunk_parameters 和 _index_store
        self._chunk_parameters = chunk_parameters
        self._index_store = index_store
        # 调用父类的构造函数
        super().__init__(**kwargs)

    def assemble(self, knowledge: Knowledge) -> List[Chunk]:
        """Assemble knowledge for input value."""
        # 使用知识和给定的 chunk_parameters、index_store 创建 EmbeddingAssembler 对象
        assembler = EmbeddingAssembler.load_from_knowledge(
            knowledge=knowledge,
            chunk_parameters=self._chunk_parameters,
            index_store=self._index_store,
        )
        # 将组装的结果持久化
        assembler.persist()
        # 返回组装后的 Chunk 对象列表
        return assembler.get_chunks()
```