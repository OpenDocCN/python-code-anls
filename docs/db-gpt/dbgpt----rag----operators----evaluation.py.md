# `.\DB-GPT-src\dbgpt\rag\operators\evaluation.py`

```py
"""Evaluation operators."""
import asyncio  # 导入异步IO库，用于并发执行任务
from typing import Any, List, Optional  # 导入类型提示相关的模块

from dbgpt.core import Chunk  # 导入Chunk类
from dbgpt.core.awel import JoinOperator  # 导入JoinOperator类
from dbgpt.core.interface.evaluation import EvaluationMetric, EvaluationResult  # 导入评估相关的类
from dbgpt.core.interface.llm import LLMClient  # 导入LLMClient类


class RetrieverEvaluatorOperator(JoinOperator[List[EvaluationResult]]):
    """Evaluator for retriever."""

    def __init__(
        self,
        evaluation_metrics: List[EvaluationMetric],
        llm_client: Optional[LLMClient] = None,
        **kwargs,
    ):
        """Create a new RetrieverEvaluatorOperator."""
        self.llm_client = llm_client  # 初始化LLM客户端
        self.evaluation_metrics = evaluation_metrics  # 初始化评估指标列表
        super().__init__(combine_function=self._do_evaluation, **kwargs)  # 调用父类的初始化方法设置合并函数为_do_evaluation

    async def _do_evaluation(
        self,
        query: str,
        prediction: List[Chunk],
        contexts: List[str],
        raw_dataset: Any = None,
    ) -> List[EvaluationResult]:
        """Run evaluation.

        Args:
            query(str): The query string.
            prediction(List[Chunk]): The retrieved chunks from the retriever.
            contexts(List[str]): The contexts from dataset.
            raw_dataset(Any): The raw data(single row) from dataset.
        """
        if isinstance(contexts, str):
            contexts = [contexts]  # 如果contexts是字符串，则转换为单元素列表
        prediction_strs = [chunk.content for chunk in prediction]  # 获取prediction中每个Chunk对象的内容组成列表
        tasks = []
        for metric in self.evaluation_metrics:
            tasks.append(metric.compute(prediction_strs, contexts))  # 调用每个评估指标的compute方法，生成异步任务列表
        task_results = await asyncio.gather(*tasks)  # 并发执行异步任务，获取结果列表
        results = []
        for result, metric in zip(task_results, self.evaluation_metrics):
            results.append(
                EvaluationResult(
                    query=query,
                    prediction=prediction,
                    score=result.score,
                    contexts=contexts,
                    passing=result.passing,
                    raw_dataset=raw_dataset,
                    metric_name=metric.name,
                )
            )  # 根据评估结果创建EvaluationResult对象列表
        return results  # 返回评估结果列表
```