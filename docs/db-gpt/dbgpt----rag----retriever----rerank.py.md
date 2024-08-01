# `.\DB-GPT-src\dbgpt\rag\retriever\rerank.py`

```py
"""Rerank module for RAG retriever."""

# 导入必要的模块和类
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

# 导入具体功能模块
from dbgpt.core import Chunk, RerankEmbeddings
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
from dbgpt.util.executor_utils import blocking_func_to_async_no_executor
from dbgpt.util.i18n_utils import _

# 定义一个类型别名，表示排名函数的类型
RANK_FUNC = Callable[[List[Chunk]], List[Chunk]]


class Ranker(ABC):
    """Base Ranker."""

    def __init__(self, topk: int, rank_fn: Optional[RANK_FUNC] = None) -> None:
        """Create abstract base ranker.

        Args:
            topk: int - 排名结果返回的前k个元素
            rank_fn: Optional[callable] - 可选的排名函数
        """
        self.topk = topk  # 初始化 topk 属性
        self.rank_fn = rank_fn  # 初始化 rank_fn 属性

    @abstractmethod
    def rank(
        self, candidates_with_scores: List[Chunk], query: Optional[str] = None
    ) -> List[Chunk]:
        """Return top k chunks after ranker.

        Rank algorithm implementation return topk documents by candidates
        similarity score

        Args:
            candidates_with_scores: List[Tuple] - 带有得分的候选项列表
            query: Optional[str] - 可选的查询字符串
        Return:
            List[Chunk] - 返回排名后的前k个 Chunk 对象
        """

    async def arank(
        self, candidates_with_scores: List[Chunk], query: Optional[str] = None
    ) -> List[Chunk]:
        """Return top k chunks after ranker.

        Rank algorithm implementation return topk documents by candidates
        similarity score

        Args:
            candidates_with_scores: List[Tuple] - 带有得分的候选项列表
            query: Optional[str] - 可选的查询字符串
        Return:
            List[Chunk] - 返回排名后的前k个 Chunk 对象
        """
        # 使用非阻塞的方式调用阻塞函数 self.rank
        return await blocking_func_to_async_no_executor(
            self.rank, candidates_with_scores, query
        )

    def _filter(self, candidates_with_scores: List) -> List[Chunk]:
        """Filter duplicate candidates documents."""
        # 根据得分对候选项进行排序
        candidates_with_scores = sorted(
            candidates_with_scores, key=lambda x: x.score, reverse=True
        )
        visited_docs = set()  # 使用集合记录已访问的文档内容
        new_candidates = []
        for candidate_chunk in candidates_with_scores:
            if candidate_chunk.content not in visited_docs:
                new_candidates.append(candidate_chunk)
                visited_docs.add(candidate_chunk.content)
        return new_candidates  # 返回过滤后的候选项列表

    def _rerank_with_scores(
        self, candidates_with_scores: List[Chunk], rank_scores: List[float]
    ) -> List[Chunk]:
        """Rerank candidates with scores."""
        # 更新候选项的得分
        for candidate, score in zip(candidates_with_scores, rank_scores):
            candidate.score = float(score)

        # 根据新的得分再次对候选项进行排序
        new_candidates_with_scores = sorted(
            candidates_with_scores, key=lambda x: x.score, reverse=True
        )
        return new_candidates_with_scores  # 返回重新排名后的候选项列表


@register_resource(
    _("Default Ranker"),
    "default_ranker",
    category=ResourceCategory.RAG,
    description=_("Default ranker(Rank by score)."),
    parameters=[
        # 创建一个参数列表，每个参数是一个 Parameter 对象
        Parameter.build_from(
            # 参数的显示名称为 "Top k"
            _("Top k"),
            # 参数的名称为 "topk"
            "topk",
            # 参数的类型为整数 (int)
            int,
            # 参数的描述为 "The number of top k documents."
            description=_("The number of top k documents."),
        ),
        # 下面是被注释掉的部分，未被实际使用
        # Parameter.build_from(
        #     # 参数的显示名称为 "Rank Function"
        #     _("Rank Function"),
        #     # 参数的名称为 "rank_fn"
        #     "rank_fn",
        #     # 参数的类型为 RANK_FUNC（假设是一个预定义的类型）
        #     RANK_FUNC,
        #     # 参数的描述为 "The rank function."
        #     description=_("The rank function."),
        #     # 参数是可选的，且默认值为 None
        #     optional=True,
        #     default=None,
        # ),
    ],
# 闭合括号，可能是代码中的错误，需要检查
)

# 默认排名器类，继承自Ranker类
class DefaultRanker(Ranker):
    """Default Ranker."""

    # 初始化方法，接受topk和rank_fn两个参数
    def __init__(
        self,
        topk: int = 4,
        rank_fn: Optional[RANK_FUNC] = None,
    ):
        """Create Default Ranker with topk and rank_fn."""
        # 调用父类的初始化方法
        super().__init__(topk, rank_fn)

    # 排名方法，返回排名前k个chunks
    def rank(
        self, candidates_with_scores: List[Chunk], query: Optional[str] = None
    ) -> List[Chunk]:
        """Return top k chunks after ranker.

        Return top k documents by candidates similarity score

        Args:
            candidates_with_scores: List[Tuple]

        Return:
            List[Chunk]: List of top k documents
        """
        # 过滤候选项
        candidates_with_scores = self._filter(candidates_with_scores)
        # 如果rank_fn不为空，则使用rank_fn进行排名
        if self.rank_fn is not None:
            candidates_with_scores = self.rank_fn(candidates_with_scores)
        else:
            # 否则按照分数降序排名
            candidates_with_scores = sorted(
                candidates_with_scores, key=lambda x: x.score, reverse=True
            )
        return candidates_with_scores[: self.topk]


# RRF(Reciprocal Rank Fusion)排名器类，继承自Ranker类
class RRFRanker(Ranker):
    """RRF(Reciprocal Rank Fusion) Ranker."""

    # 初始化方法，接受topk和rank_fn两个参数
    def __init__(
        self,
        topk: int = 4,
        rank_fn: Optional[RANK_FUNC] = None,
    ):
        """RRF rank algorithm implementation."""
        # 调用父类的初始化方法
        super().__init__(topk, rank_fn)

    # 排名方法，暂时返回候选项
    def rank(
        self, candidates_with_scores: List[Chunk], query: Optional[str] = None
    ) -> List[Chunk]:
        """RRF rank algorithm implementation.

        This code implements an algorithm called Reciprocal Rank Fusion (RRF), is a
        method for combining multiple result sets with different relevance indicators
        into a single result set. RRF requires no tuning, and the different relevance
        indicators do not have to be related to each other to achieve high-quality
        results.

        RRF uses the following formula to determine the score for ranking each document:
        score = 0.0
        for q in queries:
            if d in result(q):
                score += 1.0 / ( k + rank( result(q), d ) )
        return score
        reference:https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
        """
        # it will be implemented soon when multi recall is implemented
        return candidates_with_scores


# 注册资源的装饰器，用于CrossEncoder Rerank
@register_resource(
    _("CrossEncoder Rerank"),
    "cross_encoder_ranker",
    category=ResourceCategory.RAG,
    description=_("CrossEncoder ranker."),
    parameters=[
        Parameter.build_from(
            _("Top k"),
            "topk",
            int,
            description=_("The number of top k documents."),
        ),
        Parameter.build_from(
            _("Rerank Model"),
            "model",
            str,
            description=_("rerank model name, e.g., 'BAAI/bge-reranker-base'."),
        ),
        Parameter.build_from(
            _("device"),
            "device",
            str,
            description=_("device name, e.g., 'cpu'."),
        ),
    ],
)
# 交叉编码器排名器类，继承自Ranker类
class CrossEncoderRanker(Ranker):
    """CrossEncoder Ranker."""

    def __init__(
        self,
        topk: int = 4,
        model: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        rank_fn: Optional[RANK_FUNC] = None,
    ):
        """Cross Encoder rank algorithm implementation.

        Args:
            topk: int - The number of top k documents.
            model: str - rerank model name, e.g., 'BAAI/bge-reranker-base'.
            device: str - device name, e.g., 'cpu'.
            rank_fn: Optional[callable] - The rank function.
        Refer: https://www.sbert.net/examples/applications/cross-encoder/README.html
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "please `pip install sentence-transformers`",
            )
        # 使用指定的模型和设备创建一个CrossEncoder对象，设定最大长度为512
        self._model = CrossEncoder(model, max_length=512, device=device)
        # 调用父类的初始化方法，设置topk和rank_fn属性
        super().__init__(topk, rank_fn)

    def rank(
        self, candidates_with_scores: List[Chunk], query: Optional[str] = None
    ) -> List[Chunk]:
        """Cross Encoder rank algorithm implementation.

        Args:
            candidates_with_scores: List[Chunk], candidates with scores
            query: Optional[str], query text
        Returns:
            List[Chunk], reranked candidates
        """
        # 如果候选项数量小于等于1，则直接返回候选项列表
        if len(candidates_with_scores) <= 1:
            return candidates_with_scores
        # 从candidates_with_scores中提取每个候选项的内容
        contents = [candidate.content for candidate in candidates_with_scores]
        # 根据查询文本和候选项内容创建查询-内容对列表
        query_content_pairs = [
            [
                query if query is not None else "",
                content if content is not None else "",
            ]
            for content in contents
        ]
        # 使用CrossEncoder模型预测每个查询-内容对的排名分数
        rank_scores = self._model.predict(sentences=query_content_pairs)

        # 将每个候选项的分数更新为预测的排名分数
        for candidate, score in zip(candidates_with_scores, rank_scores):
            candidate.score = float(score)

        # 根据候选项的分数进行降序排序，得到重新排名的候选项列表
        new_candidates_with_scores = sorted(
            candidates_with_scores, key=lambda x: x.score, reverse=True
        )
        # 返回排序后的前topk个候选项
        return new_candidates_with_scores[: self.topk]
    # 定义一个名为 RerankEmbeddingsRanker 的类，继承自 Ranker 类
    """Rerank Embeddings Ranker."""
    
    def __init__(
        self,
        rerank_embeddings: RerankEmbeddings,
        topk: int = 4,
        rank_fn: Optional[RANK_FUNC] = None,
    ):
        """Rerank Embeddings rank algorithm implementation."""
        # 使用传入的 rerank_embeddings 参数初始化 _model 属性
        self._model = rerank_embeddings
        # 调用父类 Ranker 的初始化方法，设置 topk 和 rank_fn 属性
        super().__init__(topk, rank_fn)

    def rank(
        self, candidates_with_scores: List[Chunk], query: Optional[str] = None
    ) -> List[Chunk]:
        """Rerank Embeddings rank algorithm implementation.

        Args:
            candidates_with_scores: List[Chunk], candidates with scores
            query: Optional[str], query text
        Returns:
            List[Chunk], reranked candidates
        """
        # 如果 candidates_with_scores 为空或者 query 为空，则直接返回 candidates_with_scores
        if not candidates_with_scores or not query:
            return candidates_with_scores

        # 提取 candidates_with_scores 中每个 candidate 的 content，存入 contents 列表
        contents = [candidate.content for candidate in candidates_with_scores]
        # 使用 _model 的 predict 方法对 query 和 contents 进行预测，得到 rank_scores
        rank_scores = self._model.predict(query, contents)
        # 使用 _rerank_with_scores 方法对 candidates_with_scores 进行重新排序
        new_candidates_with_scores = self._rerank_with_scores(
            candidates_with_scores, rank_scores
        )
        # 返回重新排序后的前 self.topk 个 candidates_with_scores
        return new_candidates_with_scores[: self.topk]

    async def arank(
        self, candidates_with_scores: List[Chunk], query: Optional[str] = None
    ) -> List[Chunk]:
        """Rerank Embeddings rank algorithm implementation.

        Args:
            candidates_with_scores: List[Chunk], candidates with scores
            query: Optional[str], query text
        Returns:
            List[Chunk], reranked candidates
        """
        # 如果 candidates_with_scores 为空或者 query 为空，则直接返回 candidates_with_scores
        if not candidates_with_scores or not query:
            return candidates_with_scores

        # 提取 candidates_with_scores 中每个 candidate 的 content，存入 contents 列表
        contents = [candidate.content for candidate in candidates_with_scores]
        # 使用 _model 的 apredict 方法对 query 和 contents 进行异步预测，得到 rank_scores
        rank_scores = await self._model.apredict(query, contents)
        # 使用 _rerank_with_scores 方法对 candidates_with_scores 进行重新排序
        new_candidates_with_scores = self._rerank_with_scores(
            candidates_with_scores, rank_scores
        )
        # 返回重新排序后的前 self.topk 个 candidates_with_scores
        return new_candidates_with_scores[: self.topk]
```