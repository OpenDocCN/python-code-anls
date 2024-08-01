# `.\DB-GPT-src\dbgpt\core\interface\evaluation.py`

```py
"""Evaluation module."""
# 导入 asyncio 库，支持异步编程
import asyncio
# 导入 string 库，提供字符串处理相关的函数和常量
import string
# 导入 ABC 抽象基类，支持定义抽象类
from abc import ABC, abstractmethod
# 导入 TYPE_CHECKING，用于类型检查
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

# 导入 BaseModel 和 Field，来自 pydantic 库，用于数据模型定义和字段描述
from dbgpt._private.pydantic import BaseModel, Field
# 导入 calculate_cosine_similarity 函数，来自 similarity_util 模块，用于计算余弦相似度
from dbgpt.util.similarity_util import calculate_cosine_similarity

# 导入本地模块中的 Embeddings 类和 LLMClient 类
from .embeddings import Embeddings
from .llm import LLMClient

# 如果在类型检查模式下，导入 InputSource 类
if TYPE_CHECKING:
    from dbgpt.core.awel.task.base import InputSource

# 定义类型别名
QueryType = Union[str, Any]
PredictionType = Union[str, Any]
ContextType = Union[str, Sequence[str], Any]
DatasetType = Union["InputSource", Iterator, AsyncIterator]


# 定义基础评估结果数据模型
class BaseEvaluationResult(BaseModel):
    """Base evaluation result."""

    # 预测数据，可以是来自LLM的输出或检索的数据等
    prediction: Optional[PredictionType] = Field(
        None,
        description="Prediction data(including the output of LLM, the data from "
        "retrieval, etc.)",
    )
    # 上下文数据
    contexts: Optional[ContextType] = Field(None, description="Context data")
    # 预测得分
    score: Optional[float] = Field(None, description="Score for the prediction")
    # 二进制评估结果（通过或未通过）
    passing: Optional[bool] = Field(
        None, description="Binary evaluation result (passing or not)"
    )
    # 评估指标名称
    metric_name: Optional[str] = Field(None, description="Name of the metric")


# 定义评估结果数据模型，继承自 BaseEvaluationResult
class EvaluationResult(BaseEvaluationResult):
    """Evaluation result.

    Output of an BaseEvaluator.
    """

    # 查询数据
    query: Optional[QueryType] = Field(None, description="Query data")
    # 原始数据集
    raw_dataset: Optional[Any] = Field(None, description="Raw dataset")


# 定义评估指标的基类
Q = TypeVar("Q")
P = TypeVar("P")
C = TypeVar("C")


class EvaluationMetric(ABC, Generic[P, C]):
    """Base class for evaluation metric."""

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self.__class__.__name__

    # 异步计算评估指标
    async def compute(
        self,
        prediction: P,
        contexts: Optional[Sequence[C]] = None,
    ) -> BaseEvaluationResult:
        """Compute the evaluation metric.

        Args:
            prediction(P): The prediction data.
            contexts(Optional[Sequence[C]]): The context data.

        Returns:
            BaseEvaluationResult: The evaluation result.
        """
        # 在当前运行的事件循环中异步执行同步计算方法
        return await asyncio.get_running_loop().run_in_executor(
            None, self.sync_compute, prediction, contexts
        )

    # 同步计算评估指标
    def sync_compute(
        self,
        prediction: P,
        contexts: Optional[Sequence[C]] = None,
    ) -> BaseEvaluationResult:
        """Compute the evaluation metric.

        Args:
            prediction(P): The prediction data.
            contexts(Optional[Sequence[C]]): The context data.

        Returns:
            BaseEvaluationResult: The evaluation result.
        """
        # 抛出未实现异常，子类需要实现此方法
        raise NotImplementedError("sync_compute is not implemented")


# 基于函数的评估指标类，继承自 EvaluationMetric
class FunctionMetric(EvaluationMetric[P, C], Generic[P, C]):
    """Evaluation metric based on a function."""
    def __init__(
        self,
        name: str,
        func: Callable[
            [P, Optional[Sequence[C]]],
            BaseEvaluationResult,
        ],
    ):
        """
        Create a FunctionMetric instance.

        Args:
            name(str): The name of the metric.
            func(Callable[[P, Optional[Sequence[C]]], BaseEvaluationResult]):
                The function to use for evaluation.
        """
        self._name = name  # 将传入的名称保存在实例变量中
        self.func = func    # 将传入的函数保存在实例变量中

    @property
    def name(self) -> str:
        """Return the name of the metric."""
        return self._name   # 返回保存的名称

    async def compute(
        self,
        prediction: P,
        context: Optional[Sequence[C]] = None,
    ) -> BaseEvaluationResult:
        """
        Compute the evaluation metric.

        This method calls the stored function with the provided
        prediction and optional context, returning the evaluation result.
        """
        return self.func(prediction, context)  # 调用存储的函数，并返回其计算结果
class ExactMatchMetric(EvaluationMetric[str, str]):
    """Exact match metric.

    Just support string prediction and context.
    """

    def __init__(self, ignore_case: bool = False, ignore_punctuation: bool = False):
        """Create an ExactMatchMetric instance.

        Args:
            ignore_case: Whether to ignore case sensitivity in evaluation.
            ignore_punctuation: Whether to ignore punctuation in evaluation.
        """
        self._ignore_case = ignore_case
        self._ignore_punctuation = ignore_punctuation

    async def compute(
        self,
        prediction: str,
        contexts: Optional[Sequence[str]] = None,
    ) -> BaseEvaluationResult:
        """Compute the exact match evaluation metric.

        Args:
            prediction: The predicted string.
            contexts: Optional list of context strings.

        Returns:
            BaseEvaluationResult: Result object containing prediction, contexts, and score.
        """
        if self._ignore_case:
            prediction = prediction.lower()
            if contexts:
                contexts = [c.lower() for c in contexts]
        if self._ignore_punctuation:
            prediction = prediction.translate(str.maketrans("", "", string.punctuation))
            if contexts:
                contexts = [
                    c.translate(str.maketrans("", "", string.punctuation))
                    for c in contexts
                ]
        score = 0 if not contexts else float(prediction in contexts)
        return BaseEvaluationResult(
            prediction=prediction,
            contexts=contexts,
            score=score,
        )


class SimilarityMetric(EvaluationMetric[str, str]):
    """Similarity metric.

    Calculate the cosine similarity between a prediction and a list of contexts.
    """

    def __init__(self, embeddings: Embeddings):
        """Create a SimilarityMetric instance with embeddings.

        Args:
            embeddings: The embeddings used for computing similarity.
        """
        self._embeddings = embeddings

    def sync_compute(
        self,
        prediction: str,
        contexts: Optional[Sequence[str]] = None,
    ) -> BaseEvaluationResult:
        """Compute the similarity evaluation metric.

        Args:
            prediction: The predicted string.
            contexts: Optional list of context strings.

        Returns:
            BaseEvaluationResult: Result object containing prediction, contexts, and score.
        """
        if not contexts:
            return BaseEvaluationResult(
                prediction=prediction,
                contexts=contexts,
                score=0.0,
            )
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for SimilarityMetric")

        similarity: np.ndarray = calculate_cosine_similarity(
            self._embeddings, prediction, contexts
        )
        return BaseEvaluationResult(
            prediction=prediction,
            contexts=contexts,
            score=float(similarity.mean()),
        )


class Evaluator(ABC):
    """Base Evaluator class."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
    ):
        """Create an Evaluator instance.

        Args:
            llm_client: Optional client for accessing Language Model services.
        """
        self.llm_client = llm_client

    @abstractmethod
    async def evaluate(
        self,
        dataset: DatasetType,
        metrics: Optional[List[EvaluationMetric]] = None,
        query_key: str = "query",
        contexts_key: str = "contexts",
        prediction_key: str = "prediction",
        parallel_num: int = 1,
        **kwargs
    ):
        """Abstract method to evaluate the model.

        Args:
            dataset: The dataset to evaluate.
            metrics: Optional list of evaluation metrics to use.
            query_key: Key to access query in dataset.
            contexts_key: Key to access contexts in dataset.
            prediction_key: Key to access predictions in dataset.
            parallel_num: Number of parallel processes for evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            Evaluation result based on provided metrics.
        """
    ) -> List[List[EvaluationResult]]:
        """
        Run evaluation with a dataset and metrics.

        Args:
            dataset(DatasetType): The dataset to evaluate.
            metrics(Optional[List[EvaluationMetric]]): The metrics to use for
                evaluation.
            query_key(str): The key for query in the dataset.
            contexts_key(str): The key for contexts in the dataset.
            prediction_key(str): The key for prediction in the dataset.
            parallel_num(int): The number of parallel tasks.
            kwargs: Additional arguments.

        Returns:
            List[List[EvaluationResult]]: The evaluation results, the length of the
                result equals to the length of the dataset. The first element in the
                list is the list of evaluation results for metrics.
        """
```