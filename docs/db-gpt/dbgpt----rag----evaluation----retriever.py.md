# `.\DB-GPT-src\dbgpt\rag\evaluation\retriever.py`

```py
"""Evaluation for retriever."""
# 导入必要的模块和类
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence, Type

from dbgpt.core import Embeddings, LLMClient
from dbgpt.core.interface.evaluation import (
    BaseEvaluationResult,
    DatasetType,
    EvaluationMetric,
    EvaluationResult,
    Evaluator,
)
from dbgpt.core.interface.operators.retriever import RetrieverOperator
from dbgpt.util.similarity_util import calculate_cosine_similarity

# 导入自定义的评估运算符
from ..operators.evaluation import RetrieverEvaluatorOperator


class RetrieverEvaluationMetric(EvaluationMetric[List[str], str], ABC):
    """Evaluation metric for retriever.

    The prediction is a list of str(content from chunks) and the context is a string.
    """


class RetrieverSimilarityMetric(RetrieverEvaluationMetric):
    """Similarity metric for retriever."""

    def __init__(self, embeddings: Embeddings):
        """Create a SimilarityMetric with embeddings."""
        # 初始化方法，接受嵌入向量对象作为参数
        self._embeddings = embeddings

    def sync_compute(
        self,
        prediction: List[str],
        contexts: Optional[Sequence[str]] = None,
    ) -> BaseEvaluationResult:
        """Compute the evaluation metric.

        Args:
            prediction(List[str]): The retrieved chunks from the retriever.
            contexts(Sequence[str]): The contexts from dataset.

        Returns:
            BaseEvaluationResult: The evaluation result.
                The score is the mean of the cosine similarity between the prediction
                and the contexts.
        """
        # 如果预测值或上下文为空，则返回一个零分数的基本评估结果对象
        if not prediction or not contexts:
            return BaseEvaluationResult(
                prediction=prediction,
                contexts=contexts,
                score=0.0,
            )
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for RelevancySimilarityMetric")

        # 计算预测值与上下文之间的余弦相似度，返回一个 numpy 数组
        similarity: np.ndarray = calculate_cosine_similarity(
            self._embeddings, contexts[0], prediction
        )
        # 返回包含预测值、上下文和相似度均值分数的基本评估结果对象
        return BaseEvaluationResult(
            prediction=prediction,
            contexts=contexts,
            score=float(similarity.mean()),
        )


class RetrieverMRRMetric(RetrieverEvaluationMetric):
    """Retriever Mean Reciprocal Rank metric.

    For each query, MRR evaluates the system’s accuracy by looking at the rank of the
    highest-placed relevant document. Specifically, it’s the average of the reciprocals
    of these ranks across all the queries. So, if the first relevant document is the
    top result, the reciprocal rank is 1; if it’s second, the reciprocal rank is 1/2,
    and so on.
    """

    def sync_compute(
        self,
        prediction: List[str],
        contexts: Optional[Sequence[str]] = None,
    ) -> BaseEvaluationResult:
        """Compute the evaluation metric for Mean Reciprocal Rank.

        Args:
            prediction(List[str]): The retrieved chunks from the retriever.
            contexts(Sequence[str]): The contexts from dataset.

        Returns:
            BaseEvaluationResult: The evaluation result.
                The score is the Mean Reciprocal Rank computed from the prediction
                and contexts.
        """
    # 定义函数 compute_mrr，计算平均倒数排名（MRR）指标
    ) -> BaseEvaluationResult:
        """Compute MRR metric.

        Args:
            prediction(Optional[List[str]]): The retrieved chunks from the retriever.
            contexts(Optional[List[str]]): The contexts from dataset.
        Returns:
            BaseEvaluationResult: The evaluation result.
                The score is the reciprocal rank of the first relevant chunk.
        """
        # 如果 prediction 或 contexts 为空，返回得分为 0 的评估结果对象
        if not prediction or not contexts:
            return BaseEvaluationResult(
                prediction=prediction,
                contexts=contexts,
                score=0.0,
            )
        # 遍历预测的检索结果
        for i, retrieved_chunk in enumerate(prediction):
            # 如果某个检索结果在上下文中存在，返回其倒数排名的评估结果对象
            if retrieved_chunk in contexts:
                return BaseEvaluationResult(
                    score=1.0 / (i + 1),
                )
        # 如果没有找到匹配的检索结果，返回得分为 0 的评估结果对象
        return BaseEvaluationResult(
            score=0.0,
        )
# 检验检索器的命中率指标，继承自检索评估指标基类
class RetrieverHitRateMetric(RetrieverEvaluationMetric):
    """Retriever Hit Rate metric.

    Hit rate calculates the fraction of queries where the correct answer is found
    within the top-k retrieved documents. In simpler terms, it’s about how often our
    system gets it right within the top few guesses.
    """

    def sync_compute(
        self,
        prediction: List[str],
        contexts: Optional[Sequence[str]] = None,
    ) -> BaseEvaluationResult:
        """Compute HitRate metric.

        Args:
            prediction(Optional[List[str]]): The retrieved chunks from the retriever.
            contexts(Optional[Sequence[str]]): The contexts from dataset.
        Returns:
            BaseEvaluationResult: The evaluation result.
        """
        # 如果预测结果或上下文为空，则返回评估结果对象，得分为0.0
        if not prediction or not contexts:
            return BaseEvaluationResult(
                prediction=prediction,
                contexts=contexts,
                score=0.0,
            )
        # 检查任意一个上下文是否在预测结果中，以确定是否命中
        is_hit = any(context in prediction for context in contexts)
        # 返回基础评估结果对象，得分为1.0（命中）或0.0（未命中）
        return BaseEvaluationResult(
            score=1.0 if is_hit else 0.0,
        )


class RetrieverEvaluator(Evaluator):
    """Evaluator for relevancy.
    def __init__(
        self,
        operator_cls: Type[RetrieverOperator],
        llm_client: Optional[LLMClient] = None,
        embeddings: Optional[Embeddings] = None,
        operator_kwargs: Optional[Dict] = None,
    ):
        """
        创建一个新的 RetrieverEvaluator 对象。

        Args:
            operator_cls (Type[RetrieverOperator]): 用于检索操作的操作器类。
            llm_client (Optional[LLMClient], optional): LLM 客户端对象，可选。默认为 None。
            embeddings (Optional[Embeddings], optional): 嵌入向量对象，可选。默认为 None。
            operator_kwargs (Optional[Dict], optional): 操作器的额外参数字典，可选。默认为 None。
        """
        # 如果 operator_kwargs 为空，则初始化为一个空字典
        if not operator_kwargs:
            operator_kwargs = {}
        # 设置操作器类
        self._operator_cls = operator_cls
        # 设置操作器的关键字参数
        self._operator_kwargs: Dict[str, Any] = operator_kwargs
        # 设置嵌入向量对象
        self.embeddings = embeddings
        # 调用父类的初始化方法，传入 LLM 客户端对象
        super().__init__(llm_client=llm_client)

    async def evaluate(
        self,
        dataset: DatasetType,
        metrics: Optional[List[EvaluationMetric]] = None,
        query_key: str = "query",
        contexts_key: str = "contexts",
        prediction_key: str = "prediction",
        parallel_num: int = 1,
        **kwargs,
    ):
        """
        异步评估给定数据集。

        Args:
            dataset (DatasetType): 包含查询和上下文的数据集。
            metrics (Optional[List[EvaluationMetric]], optional): 评估指标列表，可选。默认为 None。
            query_key (str, optional): 查询键名，可选。默认为 "query"。
            contexts_key (str, optional): 上下文键名，可选。默认为 "contexts"。
            prediction_key (str, optional): 预测结果键名，可选。默认为 "prediction"。
            parallel_num (int, optional): 并行执行的数量，可选。默认为 1。
            **kwargs: 其他可选参数。

        Returns:
            EvaluationResults: 包含评估结果的对象。
        """
        """Evaluate the dataset."""
        # 导入必要的模块和类
        from dbgpt.core.awel import DAG, IteratorTrigger, MapOperator
        
        # 如果未提供评估指标，根据是否存在嵌入向量决定是否抛出异常
        if not metrics:
            if not self.embeddings:
                raise ValueError("embeddings are required for SimilarityMetric")
            metrics = [RetrieverSimilarityMetric(self.embeddings)]

        # 创建一个名为 "relevancy_evaluation_dag" 的DAG对象上下文
        with DAG("relevancy_evaluation_dag"):
            # 创建一个迭代器触发器，以 dataset 作为输入
            input_task = IteratorTrigger(dataset)
            
            # 创建一个 MapOperator 实例，用于从数据中提取查询键对应的值
            query_task: MapOperator = MapOperator(lambda x: x[query_key])
            
            # 使用指定的运算符和参数创建一个检索器任务
            retriever_task = self._operator_cls(**self._operator_kwargs)
            
            # 创建一个检索器评估器任务，传入评估指标和 llm_client
            retriever_eva_task = RetrieverEvaluatorOperator(
                evaluation_metrics=metrics, llm_client=self.llm_client
            )
            
            # 设置 DAG 中的任务依赖关系
            input_task >> query_task
            query_task >> retriever_eva_task
            query_task >> retriever_task >> retriever_eva_task
            input_task >> MapOperator(lambda x: x[contexts_key]) >> retriever_eva_task
            input_task >> retriever_eva_task
        
        # 触发输入任务，并行处理指定数量的任务
        results = await input_task.trigger(parallel_num=parallel_num)
        # 返回结果列表
        return [item for _, item in results]
```